// Standalone entry point for the composable GraspNode.
//
// The component library is the primary artefact; this executable exists so the
// node can also be run in its own process, which is the inter-process arm of
// the intra-process comparison and the only arm the Python node can offer.

#include <chrono>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>

#include "grasp_node_cpp/grasp_node.hpp"

namespace
{

using grasp_node_cpp::GraspNode;
namespace lifecycle = lifecycle_msgs::msg;

bool has_lifecycle_services(const GraspNode & node)
{
  const std::string prefix = node.get_fully_qualified_name();
  const std::map<std::string, std::vector<std::string>> services =
    node.get_service_names_and_types();
  return services.count(prefix + "/change_state") != 0 &&
         services.count(prefix + "/get_state") != 0;
}

// Each step runs only if the previous one landed where it should: asking the
// state machine for a transition it does not currently offer throws.
int walk_transitions(GraspNode & node)
{
  if (node.configure().id() != lifecycle::State::PRIMARY_STATE_INACTIVE) {
    RCLCPP_ERROR(node.get_logger(), "smoke test: configure did not reach 'inactive'");
    return 1;
  }
  if (node.activate().id() != lifecycle::State::PRIMARY_STATE_ACTIVE) {
    RCLCPP_ERROR(node.get_logger(), "smoke test: activate did not reach 'active'");
    return 1;
  }
  if (node.deactivate().id() != lifecycle::State::PRIMARY_STATE_INACTIVE) {
    RCLCPP_ERROR(node.get_logger(), "smoke test: deactivate did not reach 'inactive'");
    return 1;
  }
  if (node.cleanup().id() != lifecycle::State::PRIMARY_STATE_UNCONFIGURED) {
    RCLCPP_ERROR(node.get_logger(), "smoke test: cleanup did not reach 'unconfigured'");
    return 1;
  }
  RCLCPP_INFO(node.get_logger(), "smoke test: configure, activate, deactivate, cleanup all held");
  return 0;
}

// What `smoke_test:=true` is for: CI cannot supply a frame store or a bag, so
// the most it can prove is that this binary links against the RMW the image
// ships, reaches the graph and answers for its own state. That is exactly the
// class of failure a ROS package written without a ROS installation is likely
// to have, which is what makes the check worth its wall time.
//
// When the pipeline parameters happen to be set as well, the smoke run walks
// the full transition square, which additionally proves that on_configure
// builds the pipeline and on_cleanup gives it back.
int run_smoke_test(GraspNode & node)
{
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node.get_node_base_interface());

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  bool alive = false;
  while (rclcpp::ok() && !alive && std::chrono::steady_clock::now() < deadline) {
    executor.spin_once(std::chrono::milliseconds(100));
    alive = has_lifecycle_services(node);
  }

  int status = 0;
  if (!alive) {
    RCLCPP_ERROR(
      node.get_logger(),
      "smoke test: the change_state and get_state services of %s did not appear in "
      "the graph within 10 s, so the lifecycle interface is not up",
      node.get_fully_qualified_name());
    status = 1;
  } else if (node.get_current_state().id() != lifecycle::State::PRIMARY_STATE_UNCONFIGURED) {
    RCLCPP_ERROR(
      node.get_logger(), "smoke test: expected state 'unconfigured', found '%s'",
      node.get_current_state().label().c_str());
    status = 1;
  } else {
    RCLCPP_INFO(
      node.get_logger(), "smoke test: alive, lifecycle interface up, state '%s'",
      node.get_current_state().label().c_str());
    if (node.get_parameter("config_path").as_string().empty()) {
      RCLCPP_INFO(
        node.get_logger(),
        "smoke test: config_path is unset, so the transition cycle was skipped");
    } else {
      status = walk_transitions(node);
    }
  }

  executor.remove_node(node.get_node_base_interface());
  return status;
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GraspNode>(rclcpp::NodeOptions());

  int status = 0;
  if (node->smoke_test()) {
    // An exception escaping here would abort and report the failure as a
    // signal, which reads as a crashed test rather than a failed one.
    try {
      status = run_smoke_test(*node);
    } catch (const std::exception & error) {
      RCLCPP_ERROR(node->get_logger(), "smoke test threw: %s", error.what());
      status = 1;
    }
  } else {
    rclcpp::spin(node->get_node_base_interface());
  }

  rclcpp::shutdown();
  return status;
}
