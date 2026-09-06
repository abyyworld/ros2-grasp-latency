// End to end over S0..S7 on a scene whose answer is known in closed form.

#include <cmath>

#include <Eigen/Core>

#include "grasp_core/chain.hpp"
#include "grasp_core/kinematics.hpp"
#include "grasp_core/config.hpp"
#include "grasp_core/pipeline.hpp"

#include "check.hpp"
#include "scene.hpp"

using grasp_core::Chain;
using grasp_core::Config;
using grasp_core::kDof;
using grasp_core::Pipeline;
using grasp_core::Result;

int main()
{
  const Config config = Config::load(grasp_test::asset("assets/pipeline_config.json"));
  const grasp_test::Scene scene = grasp_test::render_box_on_table(config, 320, 240);

  Pipeline pipeline(
    grasp_test::asset("assets/pipeline_config.json"),
    grasp_test::asset("assets/franka/panda_chain.json"),
    grasp_test::asset("assets/ransac_uniform.bin"));

  const Result & r = pipeline.run(
    scene.depth.data(), scene.rgb.data(), scene.width, scene.height);

  CHECK(r.points > config.plane.min_inliers);

  // The table is level in this fixture, so the refit normal is +z and the
  // offset is zero to within the depth quantisation.
  CHECK(r.plane_found);
  CHECK_NEAR(r.plane[0], 0.0, 1e-6);
  CHECK_NEAR(r.plane[1], 0.0, 1e-6);
  CHECK_NEAR(r.plane[2], 1.0, 1e-9);
  CHECK_NEAR(r.plane[3], 0.0, 1e-3);

  CHECK(r.cluster_points >= config.cluster.min_points);

  // The box top is 4 cm across its narrow axis, plus the finger clearance.
  CHECK(r.graspable);
  CHECK_NEAR(r.width, 2.0 * scene.box_half_x + config.grasp.finger_clearance_m, 4e-3);

  // Approach straight down, so the TCP z column is the configured approach
  // axis and the frame is right-handed.
  CHECK_NEAR(r.tcp[2], config.grasp.approach_axis_base[0], 1e-12);
  CHECK_NEAR(r.tcp[6], config.grasp.approach_axis_base[1], 1e-12);
  CHECK_NEAR(r.tcp[10], config.grasp.approach_axis_base[2], 1e-12);

  // The fingers close along the narrow axis of the box top, which is base x.
  CHECK_NEAR(std::abs(r.tcp[1]), 1.0, 1e-6);
  CHECK_NEAR(r.tcp[5], 0.0, 1e-6);

  CHECK_NEAR(r.tcp[3], scene.box_centre_x, 3e-3);
  CHECK_NEAR(r.tcp[7], scene.box_centre_y, 3e-3);
  CHECK_NEAR(r.tcp[11], scene.box_height - config.grasp.grasp_depth_m, 3e-3);
  CHECK(r.tcp[15] == 1.0);

  // The returned q has to put the TCP on the requested pose. Whether the
  // solver also raises `converged` depends on the null-space pull of S6, which
  // holds a millimetre-scale steady-state error on a top-down grasp reached
  // from q_neutral, so the flag is checked for truthfulness rather than
  // assumed true.
  CHECK(r.iterations > 0);
  CHECK(r.iterations <= config.ik.max_iterations);
  CHECK(r.converged == (r.iterations < config.ik.max_iterations));
  for (int i = 0; i < kDof; ++i) {
    CHECK(std::isfinite(r.q[static_cast<std::size_t>(i)]));
  }
  {
    const Chain chain = Chain::load(grasp_test::asset("assets/franka/panda_chain.json"));
    const Eigen::Matrix4d reached = grasp_core::fk_tcp(chain, r.q.data());
    CHECK_NEAR(reached(0, 3), r.tcp[3], 5e-3);
    CHECK_NEAR(reached(1, 3), r.tcp[7], 5e-3);
    CHECK_NEAR(reached(2, 3), r.tcp[11], 5e-3);
    CHECK_NEAR(reached(2, 2), r.tcp[10], 5e-3);
  }

  CHECK(r.trajectory.size() == static_cast<std::size_t>(config.trajectory.waypoints));
  CHECK(r.duration_s >= config.trajectory.min_duration_s);
  for (int i = 0; i < kDof; ++i) {
    const auto j = static_cast<std::size_t>(i);
    CHECK_NEAR(r.trajectory.back().position[j], r.q[j], 1e-12);
  }

  // total_ns is the sum of the eight stages by construction, and a drift here
  // would mean a stage boundary was measured twice or not at all.
  const auto & s = r.stage_ns;
  const std::uint64_t sum = s.decode + s.deproject + s.transform_crop + s.plane +
    s.cluster + s.grasp + s.ik + s.traj;
  CHECK(sum == r.total_ns);
  CHECK(r.total_ns > 0);
  CHECK(pipeline.timer_overhead_ns() >= 0.0);

  // Determinism: the same bytes must give the same answer, since the whole
  // equivalence gate rests on there being no state carried between frames.
  const Result first = r;
  const Result & again = pipeline.run(
    scene.depth.data(), scene.rgb.data(), scene.width, scene.height);
  CHECK(again.cluster_points == first.cluster_points);
  CHECK(again.iterations == first.iterations);
  for (int i = 0; i < kDof; ++i) {
    const auto j = static_cast<std::size_t>(i);
    CHECK(again.q[j] == first.q[j]);
  }

  // An all-zero depth frame is an entirely invalid return, which must bail out
  // rather than crash: a legitimate outcome with a legitimate cost.
  std::vector<std::uint16_t> blank(scene.depth.size(), 0u);
  const Result & empty = pipeline.run(
    blank.data(), scene.rgb.data(), scene.width, scene.height);
  CHECK(empty.points == 0);
  CHECK(!empty.plane_found);
  CHECK(empty.cluster_points == 0);
  CHECK(!empty.graspable);
  CHECK(!empty.converged);
  CHECK(empty.trajectory.size() == static_cast<std::size_t>(config.trajectory.waypoints));

  return grasp_test::report("test_pipeline");
}
