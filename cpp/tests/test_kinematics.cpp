// Forward kinematics against MuJoCo, and an IK round trip that checks the
// pose rather than the joint vector: the arm is redundant, so a solver that
// lands on a different point of the null-space manifold is correct and a test
// that demanded q back would be testing the seed instead of the solver.

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "grasp_core/chain.hpp"
#include "grasp_core/config.hpp"
#include "grasp_core/kinematics.hpp"

#include "check.hpp"
#include "fk_reference.hpp"

using grasp_core::Chain;
using grasp_core::Config;
using grasp_core::IkSolver;
using grasp_core::kDof;

namespace
{

// tests/test_urdf_mjcf_agree.py measures the URDF-against-MJCF residual at
// 6.1e-12 m and 8.4e-8 rad; these admit that and nothing larger.
constexpr double kPositionTol = 1e-9;
constexpr double kRotationTol = 5e-7;

Eigen::Matrix4d reference_pose(int c)
{
  Eigen::Matrix4d t;
  for (int r = 0; r < 4; ++r) {
    for (int col = 0; col < 4; ++col) {
      t(r, col) = kFkTcpRowMajor[c][4 * r + col];
    }
  }
  return t;
}

void test_forward_kinematics(const Chain & chain)
{
  for (int c = 0; c < kFkCases; ++c) {
    const Eigen::Matrix4d expected = reference_pose(c);
    const Eigen::Matrix4d actual = grasp_core::fk_tcp(chain, kFkQ[c]);
    for (int i = 0; i < 3; ++i) {
      CHECK_NEAR(actual(i, 3), expected(i, 3), kPositionTol);
      for (int j = 0; j < 3; ++j) {
        CHECK_NEAR(actual(i, j), expected(i, j), kRotationTol);
      }
    }
  }
}

void test_rodrigues_is_a_rotation()
{
  const Eigen::Vector3d axis(0.3, -0.7, 0.65);
  const Eigen::Matrix3d r = grasp_core::axis_rotation(axis, 1.234);
  CHECK_NEAR(r.determinant(), 1.0, 1e-12);
  CHECK_NEAR((r.transpose() * r - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12);
  // A rotation about its own axis leaves the axis fixed.
  const Eigen::Vector3d unit = axis.normalized();
  CHECK_NEAR((r * unit - unit).norm(), 0.0, 1e-12);
}

void test_ik_round_trip(const Config & config, const Chain & chain)
{
  const IkSolver solver(config, chain);

  // Reachable configurations, well inside the joint limits, so a failure is
  // the solver's and not the workspace's.
  const double seeds[][kDof] = {
    {0.15, -0.55, -0.10, -2.10, 0.05, 1.65, 0.70},
    {-0.40, -0.30, 0.25, -1.90, -0.20, 1.45, 1.05},
    {0.62, -0.95, 0.05, -2.35, 0.31, 1.90, 0.35},
  };

  // What the returned flag promises, and what the returned q promises when the
  // flag is false. The second bound is loose on purpose: the null-space pull of
  // docs/ALGORITHM.md S6 leaks through the damped projector and holds a
  // millimetre-scale steady-state error on some targets, so a test that
  // demanded convergence everywhere would be asserting something the specified
  // algorithm does not provide. What must always hold is that the flag is
  // truthful and that q is a legal, close configuration.
  const double kResidualBound = 5e-3;

  for (const auto & seed : seeds) {
    const Eigen::Matrix4d target = grasp_core::fk_tcp(chain, seed);
    std::array<double, kDof> q{};
    int iterations = 0;
    const bool converged = solver.solve(target, q, iterations);

    const Eigen::Matrix4d reached = grasp_core::fk_tcp(chain, q.data());
    const Eigen::Vector3d position_error =
      target.topRightCorner<3, 1>() - reached.topRightCorner<3, 1>();
    const Eigen::Matrix3d r_des = target.topLeftCorner<3, 3>();
    const Eigen::Matrix3d r_cur = reached.topLeftCorner<3, 3>();
    const Eigen::Vector3d orientation_error = 0.5 *
      (r_cur.col(0).cross(r_des.col(0)) + r_cur.col(1).cross(r_des.col(1)) +
      r_cur.col(2).cross(r_des.col(2)));

    if (converged) {
      CHECK(iterations < config.ik.max_iterations);
      CHECK(position_error.norm() < config.ik.position_tolerance_m);
      CHECK(orientation_error.norm() < config.ik.orientation_tolerance_rad);
    } else {
      CHECK(iterations == config.ik.max_iterations);
    }
    // The pose, not q: the arm is redundant, so landing on a different point
    // of the null-space manifold is a correct answer.
    CHECK(position_error.norm() < kResidualBound);
    CHECK(orientation_error.norm() < kResidualBound);

    for (int i = 0; i < kDof; ++i) {
      const auto & joint = chain.joints[static_cast<std::size_t>(i)];
      CHECK(q[static_cast<std::size_t>(i)] >= joint.lower);
      CHECK(q[static_cast<std::size_t>(i)] <= joint.upper);
    }
  }
}

// The solver has to be able to converge at all, or the contract test above
// passes vacuously on a solver that always reports failure.
void test_ik_converges_on_a_nearby_pose(const Config & config, const Chain & chain)
{
  const IkSolver solver(config, chain);
  std::array<double, kDof> near = chain.q_neutral;
  near[0] += 0.08;
  near[3] += 0.05;
  const Eigen::Matrix4d target = grasp_core::fk_tcp(chain, near.data());

  std::array<double, kDof> q{};
  int iterations = 0;
  CHECK(solver.solve(target, q, iterations));
  CHECK(iterations > 0);
  CHECK(iterations < config.ik.max_iterations);
}

void test_ik_reports_failure_out_of_reach(const Config & config, const Chain & chain)
{
  const IkSolver solver(config, chain);
  Eigen::Matrix4d target = Eigen::Matrix4d::Identity();
  target(0, 3) = 5.0;   // metres, well past the Panda's reach
  std::array<double, kDof> q{};
  int iterations = 0;
  CHECK(!solver.solve(target, q, iterations));
  CHECK(iterations == config.ik.max_iterations);
}

}  // namespace

int main()
{
  const Config config = Config::load(grasp_test::asset("assets/pipeline_config.json"));
  const Chain chain = Chain::load(grasp_test::asset("assets/franka/panda_chain.json"));

  test_forward_kinematics(chain);
  test_rodrigues_is_a_rotation();
  test_ik_round_trip(config, chain);
  test_ik_converges_on_a_nearby_pose(config, chain);
  test_ik_reports_failure_out_of_reach(config, chain);
  return grasp_test::report("test_kinematics");
}
