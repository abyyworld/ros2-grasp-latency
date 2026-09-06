#include "grasp_core/trajectory.hpp"

#include <algorithm>
#include <cmath>

#include "grasp_core/chain.hpp"
#include "grasp_core/config.hpp"

namespace grasp_core
{

void generate_trajectory(
  const Config & config, const Chain & chain, const std::array<double, kDof> & q_goal,
  std::vector<Waypoint> & out, double & duration_s)
{
  const int count = config.trajectory.waypoints;

  double duration = 0.0;
  std::array<double, kDof> delta{};
  for (int i = 0; i < kDof; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    delta[idx] = q_goal[idx] - chain.q_neutral[idx];
    const double limit = config.trajectory.velocity_fraction *
      chain.joints[idx].velocity;
    duration = std::max(duration, std::abs(delta[idx]) / limit);
  }
  duration = std::clamp(
    duration, config.trajectory.min_duration_s, config.trajectory.max_duration_s);
  duration_s = duration;

  const double inv_duration = 1.0 / duration;
  const double inv_duration2 = inv_duration * inv_duration;
  const double inv_span = 1.0 / static_cast<double>(count - 1);

  for (int k = 0; k < count; ++k) {
    const double s = static_cast<double>(k) * inv_span;
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double s4 = s3 * s;
    const double s5 = s4 * s;
    const double h = 10.0 * s3 - 15.0 * s4 + 6.0 * s5;
    const double hd = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) * inv_duration;
    const double hdd = (60.0 * s - 180.0 * s2 + 120.0 * s3) * inv_duration2;

    Waypoint & w = out[static_cast<std::size_t>(k)];
    for (int i = 0; i < kDof; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      w.position[idx] = chain.q_neutral[idx] + h * delta[idx];
      w.velocity[idx] = hd * delta[idx];
      w.acceleration[idx] = hdd * delta[idx];
    }
    w.time_from_start = s * duration;
  }
}

}  // namespace grasp_core
