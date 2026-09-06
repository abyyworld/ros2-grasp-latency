#include "grasp_core/chain.hpp"

#include <stdexcept>

#include "grasp_core/json.hpp"

namespace grasp_core
{

Chain Chain::load(const std::string & path)
{
  const json::Value root = json::parse_file(path);
  Chain chain;

  const int dof = root["dof"].integer();
  if (dof != kDof) {
    throw std::runtime_error(
            path + " declares " + std::to_string(dof) +
            " degrees of freedom; this core is compiled for " + std::to_string(kDof));
  }

  const auto & joints = root["joints"].array();
  if (joints.size() != static_cast<std::size_t>(kDof)) {
    throw std::runtime_error(path + " does not list one entry per degree of freedom");
  }
  for (int i = 0; i < kDof; ++i) {
    Joint & j = chain.joints[static_cast<std::size_t>(i)];
    const auto & src = joints[static_cast<std::size_t>(i)];
    j.name = src["name"].string();
    src["fixed"].fill(j.fixed.data(), j.fixed.size());
    src["axis"].fill(j.axis.data(), j.axis.size());
    j.lower = src["lower"].number();
    j.upper = src["upper"].number();
    j.velocity = src["velocity"].number();
    if (!(j.velocity > 0.0)) {
      throw std::runtime_error("joint " + j.name + " has a non-positive velocity limit");
    }
  }

  root["flange_to_tcp"].fill(chain.flange_to_tcp.data(), chain.flange_to_tcp.size());
  root["q_neutral"].fill(chain.q_neutral.data(), chain.q_neutral.size());
  return chain;
}

}  // namespace grasp_core
