#pragma once

#include <cstdint>
#include <ctime>

namespace grasp_core
{

// CLOCK_MONOTONIC, read directly rather than through std::chrono, because the
// Python side calls time.perf_counter_ns() which is the same syscall on the
// same clock, and going through steady_clock would add a conversion the other
// implementation does not pay. On glibc this resolves through the vDSO, so it
// costs tens of nanoseconds rather than a trap.
inline std::uint64_t monotonic_ns() noexcept
{
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

// Median cost of one monotonic_ns() call. Nine of them bracket the eight
// stages of a frame, so the per-frame instrumentation cost is nine times this
// and belongs in the run metadata rather than in a footnote.
double calibrate_timer_overhead_ns(int samples);

}  // namespace grasp_core
