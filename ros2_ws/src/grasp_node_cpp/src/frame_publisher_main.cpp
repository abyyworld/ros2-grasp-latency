// Standalone entry point for the composable FramePublisher, used when the
// driver has to sit in its own process: that is every run except the composed
// ones, and it is the arm the intra-process runs are compared against.

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "grasp_node_cpp/frame_publisher.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<grasp_node_cpp::FramePublisher>(rclcpp::NodeOptions());
  rclcpp::spin(node->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}
