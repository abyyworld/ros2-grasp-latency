#pragma once

// Forward kinematics and the damped-least-squares IK of docs/ALGORITHM.md S6.
// Eigen appears here and in the other internal headers but never in
// pipeline.hpp, so a consumer of the installed target does not inherit an
// Eigen include path it did not ask for.

#include <array>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "grasp_core/chain.hpp"
#include "grasp_core/config.hpp"
#include "grasp_core/dof.hpp"

namespace grasp_core
{

using Vec7 = Eigen::Matrix<double, kDof, 1>;

// Rodrigues' formula about an arbitrary axis. Every Panda joint happens to
// rotate about its child frame's z, but the chain file states an axis per
// joint and nothing here assumes what it will say.
Eigen::Matrix3d axis_rotation(const Eigen::Vector3d & axis, double angle);

// Base -> joint frame after the first `upto` joints have been applied.
// `upto == kDof` gives base -> flange; `fk_tcp` continues to the TCP.
Eigen::Matrix4d fk(const Chain & chain, const double * q, int upto);
Eigen::Matrix4d fk_tcp(const Chain & chain, const double * q);

class IkSolver
{
public:
  IkSolver(const Config & config, const Chain & chain);

  // Seeded from q_neutral every call, so cost does not depend on tracking
  // history and every frame is an independent sample. Allocation-free: every
  // intermediate is a fixed-size Eigen object on the stack.
  bool solve(const Eigen::Matrix4d & target, std::array<double, kDof> & q, int & iterations) const;

private:
  Config::Ik ik_;
  std::array<Eigen::Matrix4d, kDof> fixed_;
  std::array<Eigen::Vector3d, kDof> axis_;
  std::array<double, kDof> lower_;
  std::array<double, kDof> upper_;
  Eigen::Matrix4d flange_to_tcp_;
  Vec7 q_neutral_;
};

}  // namespace grasp_core
