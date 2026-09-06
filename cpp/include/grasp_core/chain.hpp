#pragma once

// The flattened 7-DOF Panda chain, as produced by tools/extract_chain.py from
// the vendored URDF. Nothing here is derived from the URDF at run time: the
// extraction happens offline so that the C++ and Python cores read identical
// numbers from identical bytes.

#include <array>
#include <string>
#include <vector>

#include "grasp_core/dof.hpp"

namespace grasp_core
{

struct Joint
{
  std::string name;
  std::array<double, 16> fixed{};   // row-major parent -> joint frame
  std::array<double, 3> axis{};     // in the joint frame
  double lower{0.0};
  double upper{0.0};
  double velocity{0.0};
};

struct Chain
{
  std::array<Joint, kDof> joints;
  std::array<double, 16> flange_to_tcp{};   // row-major, includes the TCP offset
  std::array<double, kDof> q_neutral{};

  static Chain load(const std::string & path);
};

}  // namespace grasp_core
