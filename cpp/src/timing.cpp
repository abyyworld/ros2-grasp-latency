#include "grasp_core/timing.hpp"

#include <algorithm>
#include <vector>

namespace grasp_core
{

double calibrate_timer_overhead_ns(int samples)
{
  if (samples < 1) {
    return 0.0;
  }
  // The median, not the mean: a single preemption between two reads would
  // otherwise dominate the estimate and make the reported overhead a
  // scheduling artefact.
  std::vector<std::uint64_t> deltas;
  deltas.reserve(static_cast<std::size_t>(samples));
  std::uint64_t previous = monotonic_ns();
  for (int i = 0; i < samples; ++i) {
    const std::uint64_t now = monotonic_ns();
    deltas.push_back(now - previous);
    previous = now;
  }
  std::sort(deltas.begin(), deltas.end());
  return static_cast<double>(deltas[deltas.size() / 2]);
}

}  // namespace grasp_core
