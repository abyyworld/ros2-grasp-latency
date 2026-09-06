#include "grasp_core/kinematics.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>

namespace grasp_core
{
namespace
{

Eigen::Matrix4d to_matrix(const std::array<double, 16> & row_major)
{
  Eigen::Matrix4d m;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      m(r, c) = row_major[static_cast<std::size_t>(4 * r + c)];
    }
  }
  return m;
}

}  // namespace

Eigen::Matrix3d axis_rotation(const Eigen::Vector3d & axis, double angle)
{
  const Eigen::Vector3d k = axis.normalized();
  Eigen::Matrix3d kx;
  kx << 0.0, -k.z(), k.y(),
    k.z(), 0.0, -k.x(),
    -k.y(), k.x(), 0.0;
  return Eigen::Matrix3d::Identity() + std::sin(angle) * kx +
         (1.0 - std::cos(angle)) * (kx * kx);
}

Eigen::Matrix4d fk(const Chain & chain, const double * q, int upto)
{
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  for (int i = 0; i < upto; ++i) {
    const Joint & j = chain.joints[static_cast<std::size_t>(i)];
    t = t * to_matrix(j.fixed);
    const Eigen::Matrix3d r =
      axis_rotation(Eigen::Vector3d(j.axis[0], j.axis[1], j.axis[2]), q[i]);
    t.topLeftCorner<3, 3>() = t.topLeftCorner<3, 3>() * r;
  }
  return t;
}

Eigen::Matrix4d fk_tcp(const Chain & chain, const double * q)
{
  return fk(chain, q, kDof) * to_matrix(chain.flange_to_tcp);
}

IkSolver::IkSolver(const Config & config, const Chain & chain)
: ik_(config.ik)
{
  for (int i = 0; i < kDof; ++i) {
    const Joint & j = chain.joints[static_cast<std::size_t>(i)];
    fixed_[static_cast<std::size_t>(i)] = to_matrix(j.fixed);
    axis_[static_cast<std::size_t>(i)] =
      Eigen::Vector3d(j.axis[0], j.axis[1], j.axis[2]).normalized();
    lower_[static_cast<std::size_t>(i)] = j.lower;
    upper_[static_cast<std::size_t>(i)] = j.upper;
    q_neutral_(i) = chain.q_neutral[static_cast<std::size_t>(i)];
  }
  flange_to_tcp_ = to_matrix(chain.flange_to_tcp);
}

bool IkSolver::solve(
  const Eigen::Matrix4d & target, std::array<double, kDof> & q, int & iterations) const
{
  const Eigen::Matrix3d r_des = target.topLeftCorner<3, 3>();
  const Eigen::Vector3d p_des = target.topRightCorner<3, 1>();

  Vec7 qv = q_neutral_;
  iterations = 0;

  Eigen::Matrix<double, 6, kDof> jac;
  Eigen::Matrix<double, 6, 1> err;
  std::array<Eigen::Vector3d, kDof> z_axis;
  std::array<Eigen::Vector3d, kDof> origin;
  const Eigen::Matrix<double, 6, 6> damping_term =
    (ik_.damping * ik_.damping) * Eigen::Matrix<double, 6, 6>::Identity();

  bool converged = false;
  for (int it = 0; it < ik_.max_iterations; ++it) {
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    for (int i = 0; i < kDof; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      t = t * fixed_[idx];
      // The axis and origin are read before the joint rotation is applied:
      // rotating about the axis moves neither of them.
      origin[idx] = t.topRightCorner<3, 1>();
      z_axis[idx] = t.topLeftCorner<3, 3>() * axis_[idx];
      t.topLeftCorner<3, 3>() = t.topLeftCorner<3, 3>() * axis_rotation(axis_[idx], qv(i));
    }
    t = t * flange_to_tcp_;

    const Eigen::Vector3d p_cur = t.topRightCorner<3, 1>();
    const Eigen::Matrix3d r_cur = t.topLeftCorner<3, 3>();
    const Eigen::Vector3d e_p = p_des - p_cur;
    const Eigen::Vector3d e_r = 0.5 *
      (r_cur.col(0).cross(r_des.col(0)) + r_cur.col(1).cross(r_des.col(1)) +
      r_cur.col(2).cross(r_des.col(2)));

    if (e_p.norm() < ik_.position_tolerance_m && e_r.norm() < ik_.orientation_tolerance_rad) {
      iterations = it;
      converged = true;
      break;
    }

    for (int i = 0; i < kDof; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      jac.block<3, 1>(0, i) = z_axis[idx].cross(p_cur - origin[idx]);
      jac.block<3, 1>(3, i) = z_axis[idx];
    }
    err.head<3>() = e_p;
    err.tail<3>() = e_r;

    // One factorisation serves both the task-space step and the null-space
    // projector; JJ^T + damping^2 I is symmetric positive definite for any
    // non-zero damping, so LLT is the right decomposition and it stays on the
    // stack at this fixed size.
    const Eigen::Matrix<double, 6, 6> a = jac * jac.transpose() + damping_term;
    const Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(a);
    Vec7 dq = jac.transpose() * llt.solve(err);
    const Eigen::Matrix<double, kDof, kDof> nullspace =
      Eigen::Matrix<double, kDof, kDof>::Identity() - jac.transpose() * llt.solve(jac);
    dq += ik_.nullspace_gain * (nullspace * (q_neutral_ - qv));

    for (int i = 0; i < kDof; ++i) {
      const double step = std::clamp(dq(i), -ik_.max_step_rad, ik_.max_step_rad);
      qv(i) = std::clamp(
        qv(i) + step, lower_[static_cast<std::size_t>(i)], upper_[static_cast<std::size_t>(i)]);
    }
    iterations = it + 1;
  }

  for (int i = 0; i < kDof; ++i) {
    q[static_cast<std::size_t>(i)] = qv(i);
  }
  return converged;
}

}  // namespace grasp_core
