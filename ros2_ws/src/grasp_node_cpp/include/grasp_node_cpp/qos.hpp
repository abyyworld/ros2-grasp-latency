#pragma once

// QoS is a property of the experiment rather than of the code. If the two
// nodes offer different profiles the run measures a DDS policy difference and
// reports it as a language difference, so the profiles are declared as
// parameters and supplied from config/qos.yaml, which
// docker/run_ros_benchmark.sh checks is byte-identical between grasp_node_cpp
// and grasp_node_py before it starts a run.

#include <cstddef>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace grasp_node_cpp
{

inline rclcpp::QoS make_qos(
  const std::string & key, const std::string & reliability,
  const std::string & durability, int depth)
{
  if (depth < 1) {
    throw std::invalid_argument("qos." + key + ".history_depth must be at least 1");
  }

  rclcpp::QoS qos(rclcpp::KeepLast(static_cast<std::size_t>(depth)));

  if (reliability == "reliable") {
    qos.reliable();
  } else if (reliability == "best_effort") {
    qos.best_effort();
  } else {
    throw std::invalid_argument(
      "qos." + key + ".reliability must be 'reliable' or 'best_effort', got '" +
      reliability + "'");
  }

  if (durability == "volatile") {
    qos.durability_volatile();
  } else if (durability == "transient_local") {
    qos.transient_local();
  } else {
    throw std::invalid_argument(
      "qos." + key + ".durability must be 'volatile' or 'transient_local', got '" +
      durability + "'");
  }

  return qos;
}

// Three parameters rather than one packed string, so `ros2 param get` on a
// running node shows the profile it is actually using and a typo fails at
// declaration rather than at DDS matching time, where it would surface as a
// topic that silently never connects.
inline void declare_qos(
  rclcpp_lifecycle::LifecycleNode & node, const std::string & key,
  const std::string & reliability, const std::string & durability, int depth)
{
  node.declare_parameter<std::string>("qos." + key + ".reliability", reliability);
  node.declare_parameter<std::string>("qos." + key + ".durability", durability);
  node.declare_parameter<int>("qos." + key + ".history_depth", depth);
}

inline rclcpp::QoS resolve_qos(rclcpp_lifecycle::LifecycleNode & node, const std::string & key)
{
  return make_qos(
    key,
    node.get_parameter("qos." + key + ".reliability").as_string(),
    node.get_parameter("qos." + key + ".durability").as_string(),
    static_cast<int>(node.get_parameter("qos." + key + ".history_depth").as_int()));
}

}  // namespace grasp_node_cpp
