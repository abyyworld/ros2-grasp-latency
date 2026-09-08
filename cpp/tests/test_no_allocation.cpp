// "The measured path does not allocate" is a claim, and a claim in a
// measurement repository has to be checked rather than asserted. Global
// operator new is replaced here with a counting version; grasp_core is a
// static archive linked into this binary, so every allocation it makes goes
// through these definitions. The pipeline is warmed once, the counter is
// armed, and then a run of frames must move it by exactly zero.
//
// The counter is deliberately not compiled into the library: it exists only
// in this test binary, so nothing about the benchmark's own code path changes
// to make the check pass.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

#include "grasp_core/config.hpp"
#include "grasp_core/pipeline.hpp"

#include "check.hpp"
#include "scene.hpp"

namespace
{
std::atomic<long> g_allocations{0};
std::atomic<bool> g_armed{false};

inline void note_allocation()
{
  if (g_armed.load(std::memory_order_relaxed)) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  }
}
}  // namespace

void * operator new(std::size_t size)
{
  note_allocation();
  void * p = std::malloc(size == 0 ? 1 : size);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}

void * operator new[](std::size_t size)
{
  return ::operator new(size);
}

void * operator new(std::size_t size, std::align_val_t alignment)
{
  note_allocation();
  void * p = std::aligned_alloc(static_cast<std::size_t>(alignment), size == 0 ? 1 : size);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}

void * operator new[](std::size_t size, std::align_val_t alignment)
{
  return ::operator new(size, alignment);
}

void operator delete(void * p) noexcept {std::free(p);}
void operator delete[](void * p) noexcept {std::free(p);}
void operator delete(void * p, std::size_t) noexcept {std::free(p);}
void operator delete[](void * p, std::size_t) noexcept {std::free(p);}
void operator delete(void * p, std::align_val_t) noexcept {std::free(p);}
void operator delete[](void * p, std::align_val_t) noexcept {std::free(p);}
void operator delete(void * p, std::size_t, std::align_val_t) noexcept {std::free(p);}
void operator delete[](void * p, std::size_t, std::align_val_t) noexcept {std::free(p);}

int main()
{
  const grasp_core::Config config =
    grasp_core::Config::load(grasp_test::asset("assets/pipeline_config.json"));

  // Two resolutions, so the check covers the buffer growth path as well: the
  // second reserve() must be the last allocation either of them ever causes.
  const grasp_test::Scene small = grasp_test::render_box_on_table(config, 320, 240);
  const grasp_test::Scene large = grasp_test::render_box_on_table(config, 640, 480);

  grasp_core::Pipeline pipeline(
    grasp_test::asset("assets/pipeline_config.json"),
    grasp_test::asset("assets/franka/panda_chain.json"),
    grasp_test::asset("assets/ransac_uniform.bin"));

  pipeline.reserve(large.width, large.height);
  pipeline.run(large.depth.data(), large.rgb.data(), large.width, large.height);

  g_allocations.store(0);
  g_armed.store(true);
  const int frames = 32;
  for (int i = 0; i < frames; ++i) {
    const grasp_core::Result & r = pipeline.run(
      large.depth.data(), large.rgb.data(), large.width, large.height);
    // Read something out of the result so an optimiser cannot decide the calls
    // are dead and delete the thing under test.
    if (r.total_ns == 0) {
      std::fprintf(stderr, "a frame took no measurable time, which cannot be right\n");
      g_armed.store(false);
      return 1;
    }
  }
  g_armed.store(false);
  const long steady_state = g_allocations.load();

  std::fprintf(
    stderr, "%d frames at %dx%d allocated %ld times\n",
    frames, large.width, large.height, steady_state);
  CHECK(steady_state == 0);

  // Changing resolution is allowed to allocate, once, and only through
  // reserve(); after that the smaller resolution must also be clean.
  pipeline.reserve(small.width, small.height);
  pipeline.run(small.depth.data(), small.rgb.data(), small.width, small.height);
  g_allocations.store(0);
  g_armed.store(true);
  for (int i = 0; i < frames; ++i) {
    pipeline.run(small.depth.data(), small.rgb.data(), small.width, small.height);
  }
  g_armed.store(false);
  const long small_state = g_allocations.load();
  std::fprintf(
    stderr, "%d frames at %dx%d allocated %ld times\n",
    frames, small.width, small.height, small_state);
  CHECK(small_state == 0);

  // Positive control. Everything above is a zero, and a broken counter also
  // reports zero, so the two assertions so far are equally consistent with the
  // pipeline being clean and with the instrument being dead. Allocate on
  // purpose and require the counter to see it.
  //
  // The allocation chosen is the one the API deliberately avoids: Pipeline::run
  // returns a const reference, and copying the Result reallocates the
  // trajectory's storage. So this also measures the cost of the interface
  // decision rather than inventing an unrelated new[].
  g_allocations.store(0);
  g_armed.store(true);
  long observed_copies = 0;
  for (int i = 0; i < frames; ++i) {
    const grasp_core::Result copied = pipeline.run(
      small.depth.data(), small.rgb.data(), small.width, small.height);
    observed_copies += static_cast<long>(copied.trajectory.size());
  }
  g_armed.store(false);
  const long copy_state = g_allocations.load();

  std::fprintf(
    stderr,
    "positive control: %d frames taking the Result by value allocated %ld times "
    "(%ld waypoints copied)\n",
    frames, copy_state, observed_copies);
  // One allocation per frame at least: the trajectory vector's buffer. If this
  // is zero the counter is not working and the zeros above prove nothing.
  CHECK(copy_state >= frames);

  return grasp_test::report("test_no_allocation");
}
