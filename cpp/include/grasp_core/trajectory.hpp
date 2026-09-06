#pragma once

// docs/ALGORITHM.md S7. Waypoint is part of the public API: the rclcpp node
// copies straight out of it into a JointTrajectory message, so it holds plain
// arrays and pulls in nothing beyond <array>.

#include <array>
#include <vector>

#include "grasp_core/dof.hpp"

namespace grasp_core
{

struct Waypoint
{
  std::array<double, kDof> position{};
  std::array<double, kDof> velocity{};
  std::array<double, kDof> acceleration{};
  double time_from_start{0.0};
};

struct Chain;
struct Config;

// `out` must already hold config.trajectory.waypoints entries: this is called
// inside the measured path and must not allocate.
void generate_trajectory(
  const Config & config, const Chain & chain, const std::array<double, kDof> & q_goal,
  std::vector<Waypoint> & out, double & duration_s);

}  // namespace grasp_core
