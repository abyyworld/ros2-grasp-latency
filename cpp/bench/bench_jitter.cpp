// Jitter, not latency.
//
// bench_pipeline answers "how long does a frame take". A control loop is not
// bought by that number. It is bought by whether the answer arrives on the
// same phase every period, and by what the worst period does, because a
// servo consuming this output has a deadline and does not care about the
// median. This binary measures that instead: it releases work on a fixed
// period and records, per cycle, how late the wake-up was, how long the work
// took, and how much slack was left against the deadline.
//
// Two design points that decide whether the numbers mean anything:
//
// * Releases are absolute offsets from one monotonic origin, slept to with
//   clock_nanosleep(TIMER_ABSTIME). A loop that sleeps for `period` after
//   finishing accumulates its own overhead into the schedule and measures its
//   own drift rather than the system's jitter.
// * The scheduling policy is a flag, not an assumption. SCHED_OTHER is what a
//   node gets by default and SCHED_FIFO is what a real-time one asks for, and
//   the difference between them on the same machine is the result worth
//   reporting: absolute jitter on a shared vCPU says more about the vCPU.
//
// Allocation and page-fault counters ride along because the same run answers
// them: a hot path that allocates or faults cannot hold a deadline, and both
// are claims this repository previously asserted rather than measured. They go
// into a run header at the top of the output file rather than onto stderr,
// because a number quoted in a document has to be readable back out of a
// committed artifact.
//
// The recorder holds itself to the same rule as the code it measures. Its own
// per-cycle log buffer is pre-touched, since pages first written inside the
// measured window fault there and would be charged to the pipeline;
// --no-pretouch runs the identical loop without that and is the control.
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "grasp_core/pipeline.hpp"

namespace
{

// Counting operator new, armed only for the measured window. Defined in this
// binary and not in the library, so the code under test is byte-identical to
// what bench_pipeline runs.
std::atomic<long> g_allocs{0};
std::atomic<long> g_alloc_bytes{0};
std::atomic<bool> g_armed{false};

inline void note(std::size_t size)
{
  if (g_armed.load(std::memory_order_relaxed)) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add(static_cast<long>(size), std::memory_order_relaxed);
  }
}

struct Cycle
{
  long seq;
  long frame;
  long release_ns;    // when this cycle was scheduled to start
  long wake_ns;       // when it actually started
  long compute_ns;    // how long the pipeline took
  long slack_ns;      // deadline minus finish; negative is an overrun
  long allocs;        // allocations during this cycle
};

struct Options
{
  std::string dataset;
  std::string out;
  std::string policy = "other";
  std::string label;
  int priority = 80;
  int cpu = -1;
  double rate_hz = 30.0;
  long frames = 2000;
  long warmup = 200;
  bool mlock = false;
  bool pretouch = true;
};

[[noreturn]] void usage(const char * program, const std::string & why)
{
  std::fprintf(stderr, "%s\n", why.c_str());
  std::fprintf(
    stderr,
    "usage: %s --dataset DIR --out FILE\n"
    "       [--policy other|fifo] [--priority N] [--cpu N] [--mlock]\n"
    "       [--rate HZ] [--frames N] [--warmup N] [--label NAME]\n"
    "       [--no-pretouch]\n",
    program);
  std::exit(2);
}

long now_ns()
{
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<long>(ts.tv_sec) * 1000000000L + ts.tv_nsec;
}

void sleep_until(long absolute_ns)
{
  timespec ts{};
  ts.tv_sec = static_cast<time_t>(absolute_ns / 1000000000L);
  ts.tv_nsec = static_cast<long>(absolute_ns % 1000000000L);
  // Restart on EINTR: a signal must not shorten a period.
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
  }
}

// Fault counts are read outside every measured window, so the syscall never
// lands inside a cycle it would then be reported as part of.
struct Faults
{
  long minor;
  long major;
};

Faults read_faults()
{
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return {ru.ru_minflt, ru.ru_majflt};
}

std::string apply_policy(const Options & opt)
{
  std::string notes;

  if (opt.cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(opt.cpu, &set);
    notes += sched_setaffinity(0, sizeof(set), &set) == 0
      ? "pinned;" : "pin-failed;";
  }

  if (opt.policy == "fifo") {
    sched_param param{};
    param.sched_priority = opt.priority;
    if (sched_setscheduler(0, SCHED_FIFO, &param) == 0) {
      notes += "fifo;";
    } else {
      // Reported rather than fatal: the comparison is still worth running as
      // SCHED_OTHER against SCHED_OTHER, and a silent downgrade would be a
      // result that looks like a null finding.
      notes += std::string("fifo-denied(") + std::strerror(errno) + ");";
    }
  } else {
    notes += "other;";
  }

  if (opt.mlock) {
    notes += mlockall(MCL_CURRENT | MCL_FUTURE) == 0
      ? "mlocked;" : std::string("mlock-denied(") + std::strerror(errno) + ");";
  }

  return notes;
}

std::vector<std::string> frame_paths(const std::string & manifest_text, const char * key)
{
  // The manifest is small and its shape is fixed by docs/FORMATS.md, so a scan
  // for the key is enough and avoids a JSON dependency in the benchmark.
  std::vector<std::string> out;
  const std::string needle = std::string("\"") + key + "\": \"";
  std::size_t pos = 0;
  while ((pos = manifest_text.find(needle, pos)) != std::string::npos) {
    pos += needle.size();
    const std::size_t end = manifest_text.find('"', pos);
    out.push_back(manifest_text.substr(pos, end - pos));
    pos = end;
  }
  return out;
}

long read_int_field(const std::string & text, const char * key)
{
  const std::string needle = std::string("\"") + key + "\": ";
  const std::size_t pos = text.find(needle);
  return pos == std::string::npos ? 0 : std::atol(text.c_str() + pos + needle.size());
}

}  // namespace

void * operator new(std::size_t size)
{
  note(size);
  void * p = std::malloc(size == 0 ? 1 : size);
  if (p == nullptr) { throw std::bad_alloc(); }
  return p;
}
void * operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }

int main(int argc, char ** argv)
{
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto value = [&](void) -> std::string {
      if (i + 1 >= argc) { usage(argv[0], "missing value for " + flag); }
      return argv[++i];
    };
    if (flag == "--dataset") { opt.dataset = value(); }
    else if (flag == "--out") { opt.out = value(); }
    else if (flag == "--policy") { opt.policy = value(); }
    else if (flag == "--label") { opt.label = value(); }
    else if (flag == "--priority") { opt.priority = std::atoi(value().c_str()); }
    else if (flag == "--cpu") { opt.cpu = std::atoi(value().c_str()); }
    else if (flag == "--rate") { opt.rate_hz = std::atof(value().c_str()); }
    else if (flag == "--frames") { opt.frames = std::atol(value().c_str()); }
    else if (flag == "--warmup") { opt.warmup = std::atol(value().c_str()); }
    else if (flag == "--mlock") { opt.mlock = true; }
    else if (flag == "--no-pretouch") { opt.pretouch = false; }
    else { usage(argv[0], "unknown flag " + flag); }
  }
  if (opt.dataset.empty() || opt.out.empty()) {
    usage(argv[0], "--dataset and --out are both required");
  }
  if (opt.label.empty()) { opt.label = opt.policy; }

  std::ifstream manifest_file(opt.dataset + "/manifest.json");
  if (!manifest_file) {
    std::fprintf(stderr, "cannot open %s/manifest.json\n", opt.dataset.c_str());
    return 1;
  }
  std::stringstream buffer;
  buffer << manifest_file.rdbuf();
  const std::string manifest = buffer.str();

  const int width = static_cast<int>(read_int_field(manifest, "width"));
  const int height = static_cast<int>(read_int_field(manifest, "height"));
  const auto depth_names = frame_paths(manifest, "depth");
  const auto rgb_names = frame_paths(manifest, "rgb");
  if (depth_names.empty() || depth_names.size() != rgb_names.size()) {
    std::fprintf(stderr, "manifest lists no usable frames\n");
    return 1;
  }

  // The whole store is resident before the loop starts: a read inside a cycle
  // would be measured as jitter that the pipeline did not cause.
  std::vector<std::vector<uint16_t>> depths(depth_names.size());
  std::vector<std::vector<uint8_t>> rgbs(rgb_names.size());
  for (std::size_t i = 0; i < depth_names.size(); ++i) {
    std::ifstream d(opt.dataset + "/" + depth_names[i], std::ios::binary);
    std::ifstream r(opt.dataset + "/" + rgb_names[i], std::ios::binary);
    depths[i].resize(static_cast<std::size_t>(width) * height);
    rgbs[i].resize(static_cast<std::size_t>(width) * height * 3);
    d.read(reinterpret_cast<char *>(depths[i].data()),
           static_cast<std::streamsize>(depths[i].size() * sizeof(uint16_t)));
    r.read(reinterpret_cast<char *>(rgbs[i].data()),
           static_cast<std::streamsize>(rgbs[i].size()));
  }

  grasp_core::Pipeline pipeline(
    "assets/pipeline_config.json", "assets/franka/panda_chain.json",
    "assets/ransac_uniform.bin");

  const std::string policy_notes = apply_policy(opt);
  const long period_ns = static_cast<long>(1e9 / opt.rate_hz);

  // Warm-up runs under the same policy and is discarded: first-touch page
  // faults and branch predictor state belong to start-up, not to steady state.
  for (long i = 0; i < opt.warmup; ++i) {
    const std::size_t f = static_cast<std::size_t>(i) % depths.size();
    pipeline.run(depths[f].data(), rgbs[f].data(), width, height);
  }

  // The record buffer is written by the loop, so its pages are first touched
  // inside the measured window unless they are touched here. reserve() alone
  // is not enough: it reserves address space, and the kernel still hands over
  // each page on first write. Measured, that is a minor fault every 4 KB of
  // records, 19 of them over 2000 cycles, charged to the pipeline by anyone
  // reading the fault count. --no-pretouch runs the same loop without this and
  // is the control that shows the difference; see docs/REALTIME.md.
  //
  // This is the instrument's own real-time discipline, and it is the same rule
  // the pipeline is held to: a hot path touches no page it has not already
  // touched.
  std::vector<Cycle> cycles;
  if (opt.pretouch) {
    cycles.assign(static_cast<std::size_t>(opt.frames), Cycle{});
    cycles.clear();
  } else {
    cycles.reserve(static_cast<std::size_t>(opt.frames));
  }

  const Faults faults_before = read_faults();
  g_allocs.store(0);
  g_alloc_bytes.store(0);
  g_armed.store(true);

  const long origin = now_ns();
  long previous_allocs = 0;
  for (long k = 0; k < opt.frames; ++k) {
    const long release = origin + k * period_ns;
    sleep_until(release);
    const long wake = now_ns();

    const std::size_t f = static_cast<std::size_t>(k) % depths.size();
    pipeline.run(depths[f].data(), rgbs[f].data(), width, height);

    const long done = now_ns();
    const long allocs = g_allocs.load(std::memory_order_relaxed);
    cycles.push_back(Cycle{
      k, static_cast<long>(f), release, wake, done - wake,
      (release + period_ns) - done, allocs - previous_allocs});
    previous_allocs = allocs;
  }

  g_armed.store(false);
  const Faults faults_after = read_faults();

  std::FILE * out = std::fopen(opt.out.c_str(), "w");
  if (out == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", opt.out.c_str());
    return 1;
  }
  long overruns = 0;
  for (const Cycle & c : cycles) { if (c.slack_ns < 0) { ++overruns; } }

  // A run header before the cycles. The fault and allocation counts are
  // properties of the whole window rather than of any one cycle, and a number
  // quoted in a document has to be readable back out of a committed file: left
  // on stderr they would be a claim about a run nobody else can inspect.
  std::fprintf(
    out,
    "{\"record\":\"run\",\"impl\":\"cpp\",\"label\":\"%s\",\"policy\":\"%s\","
    "\"policy_notes\":\"%s\",\"rate_hz\":%.17g,\"frames\":%ld,\"warmup\":%ld,"
    "\"dataset\":\"%s\",\"width\":%d,\"height\":%d,\"cpu\":%d,\"priority\":%d,"
    "\"pretouch\":%s,"
    "\"allocations\":%ld,\"allocated_bytes\":%ld,\"minor_faults\":%ld,"
    "\"major_faults\":%ld,\"overruns\":%ld}\n",
    opt.label.c_str(), opt.policy.c_str(), policy_notes.c_str(), opt.rate_hz,
    opt.frames, opt.warmup, opt.dataset.c_str(), width, height, opt.cpu,
    opt.priority, opt.pretouch ? "true" : "false",
    g_allocs.load(), g_alloc_bytes.load(),
    faults_after.minor - faults_before.minor,
    faults_after.major - faults_before.major, overruns);

  for (const Cycle & c : cycles) {
    std::fprintf(
      out,
      "{\"impl\":\"cpp\",\"label\":\"%s\",\"policy\":\"%s\",\"rate_hz\":%.17g,"
      "\"seq\":%ld,\"frame\":%ld,\"release_jitter_ns\":%ld,\"compute_ns\":%ld,"
      "\"slack_ns\":%ld,\"allocs\":%ld}\n",
      opt.label.c_str(), opt.policy.c_str(), opt.rate_hz, c.seq, c.frame,
      c.wake_ns - c.release_ns, c.compute_ns, c.slack_ns, c.allocs);
  }
  std::fclose(out);

  std::fprintf(
    stderr,
    "%s: %ld cycles at %.1f Hz, policy=%s [%s%s]\n"
    "  allocations in the measured window: %ld (%ld bytes)\n"
    "  minor faults: %ld, major faults: %ld\n"
    "  overruns: %ld (%.2f%%)\n",
    opt.label.c_str(), opt.frames, opt.rate_hz, opt.policy.c_str(),
    policy_notes.c_str(), opt.pretouch ? "pretouched;" : "not-pretouched;",
    g_allocs.load(), g_alloc_bytes.load(),
    faults_after.minor - faults_before.minor,
    faults_after.major - faults_before.major,
    overruns, 100.0 * static_cast<double>(overruns) / static_cast<double>(opt.frames));
  return 0;
}
