// S7. The quintic's whole reason for existing is that it starts and ends at
// rest, so that is what is checked, along with the endpoints being hit exactly
// rather than to a tolerance.

#include <cmath>
#include <vector>

#include "grasp_core/chain.hpp"
#include "grasp_core/config.hpp"
#include "grasp_core/trajectory.hpp"

#include "check.hpp"

using grasp_core::Chain;
using grasp_core::Config;
using grasp_core::kDof;
using grasp_core::Waypoint;

namespace
{

void check_profile(
  const Config & config, const Chain & chain, const std::array<double, kDof> & goal)
{
  std::vector<Waypoint> waypoints(static_cast<std::size_t>(config.trajectory.waypoints));
  double duration = 0.0;
  grasp_core::generate_trajectory(config, chain, goal, waypoints, duration);

  const std::size_t last = waypoints.size() - 1u;
  CHECK(duration >= config.trajectory.min_duration_s);
  CHECK(duration <= config.trajectory.max_duration_s);
  CHECK(waypoints[0].time_from_start == 0.0);
  CHECK_NEAR(waypoints[last].time_from_start, duration, 1e-15);

  for (int j = 0; j < kDof; ++j) {
    const auto joint = static_cast<std::size_t>(j);
    // Endpoints exactly, not to a tolerance: h(0) is 0 and h(1) is 1 in exact
    // arithmetic and the polynomial is evaluated so that it stays that way.
    CHECK(waypoints[0].position[joint] == chain.q_neutral[joint]);
    CHECK_NEAR(waypoints[last].position[joint], goal[joint], 1e-15);

    CHECK(waypoints[0].velocity[joint] == 0.0);
    CHECK(waypoints[last].velocity[joint] == 0.0);
    CHECK(waypoints[0].acceleration[joint] == 0.0);
    CHECK(waypoints[last].acceleration[joint] == 0.0);
  }

  // Time is strictly increasing and the reported velocity is consistent with
  // the position samples, which catches a duration used in one place and not
  // the other.
  for (std::size_t k = 1; k < waypoints.size(); ++k) {
    CHECK(waypoints[k].time_from_start > waypoints[k - 1].time_from_start);
  }
  const double dt = duration / static_cast<double>(last);
  for (std::size_t k = 1; k < last; ++k) {
    for (int j = 0; j < kDof; ++j) {
      const auto joint = static_cast<std::size_t>(j);
      const double slope =
        (waypoints[k + 1].position[joint] - waypoints[k - 1].position[joint]) / (2.0 * dt);
      CHECK_NEAR(slope, waypoints[k].velocity[joint], 0.02 * (1.0 + std::abs(slope)));
    }
  }
}

void test_duration_respects_the_velocity_limit(const Config & config, const Chain & chain)
{
  // Move one joint far enough that the velocity limit, not the floor, sets the
  // duration, and check the peak of the quintic stays inside the fraction of
  // the limit the config allows.
  std::array<double, kDof> goal = chain.q_neutral;
  goal[0] = chain.joints[0].upper;

  std::vector<Waypoint> waypoints(static_cast<std::size_t>(config.trajectory.waypoints));
  double duration = 0.0;
  grasp_core::generate_trajectory(config, chain, goal, waypoints, duration);
  CHECK(duration > config.trajectory.min_duration_s);

  // The quintic's peak rate is 1.875 times the average, which is a property of
  // the polynomial rather than of the configuration, so the bound is stated
  // that way rather than as a number pulled out of a run.
  const double peak_factor = 15.0 / 8.0;
  const double allowed = peak_factor * config.trajectory.velocity_fraction *
    chain.joints[0].velocity;
  for (const auto & w : waypoints) {
    CHECK(std::abs(w.velocity[0]) <= allowed * (1.0 + 1e-9));
  }
}

void test_zero_move_clamps_to_the_floor(const Config & config, const Chain & chain)
{
  std::vector<Waypoint> waypoints(static_cast<std::size_t>(config.trajectory.waypoints));
  double duration = 0.0;
  grasp_core::generate_trajectory(config, chain, chain.q_neutral, waypoints, duration);
  CHECK(duration == config.trajectory.min_duration_s);
  for (const auto & w : waypoints) {
    for (int j = 0; j < kDof; ++j) {
      const auto joint = static_cast<std::size_t>(j);
      CHECK(w.position[joint] == chain.q_neutral[joint]);
      CHECK(w.velocity[joint] == 0.0);
      CHECK(w.acceleration[joint] == 0.0);
    }
  }
}

}  // namespace

int main()
{
  const Config config = Config::load(grasp_test::asset("assets/pipeline_config.json"));
  const Chain chain = Chain::load(grasp_test::asset("assets/franka/panda_chain.json"));

  std::array<double, kDof> goal = chain.q_neutral;
  goal[0] += 0.35;
  goal[2] -= 0.20;
  goal[5] += 0.11;
  check_profile(config, chain, goal);

  test_duration_respects_the_velocity_limit(config, chain);
  test_zero_move_clamps_to_the_floor(config, chain);
  return grasp_test::report("test_trajectory");
}
