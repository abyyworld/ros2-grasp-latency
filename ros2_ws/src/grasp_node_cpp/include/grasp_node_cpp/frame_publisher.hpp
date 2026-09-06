#pragma once

// Live driver for the frame store, standing in for a camera.
//
// Why this exists rather than just `ros2 bag play`: rosbag2 replays the bytes
// it recorded, header stamp included, and offers no way to restamp on the way
// out (checked against the Jazzy play verb: there is --clock, --rate, --loop
// and --start-offset, but nothing that rewrites a message field). So under bag
// play, `now - header.stamp` is the true latency plus the unknown, constant
// offset between the bag's time origin and the moment playback started, and an
// absolute end-to-end figure cannot be recovered from it. This node stamps at
// publish, so end_to_end_ns from a run it drives is the real number.
//
// It is a component and a managed node for one reason: intra-process delivery
// only happens between two nodes in the same container, so measuring what
// zero copy is worth requires the driver to be loadable next to the consumer.
// It is a lifecycle node so that the container can hold it idle while the
// consumer configures, and start it only once the consumer is active.

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace grasp_node_cpp
{

class FramePublisher : public rclcpp_lifecycle::LifecycleNode
{
public:
  using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  explicit FramePublisher(const rclcpp::NodeOptions & options);

  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & previous) override;

  std::uint64_t published() const {return published_;}

private:
  void publish_info();
  void tick();

  std::vector<sensor_msgs::msg::Image> frames_;
  sensor_msgs::msg::CameraInfo info_;
  std::uint64_t published_{0};
  std::uint64_t total_frames_{0};
  std::chrono::nanoseconds period_{0};

  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>> depth_pub_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::CameraInfo>> info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr info_timer_;
};

}  // namespace grasp_node_cpp
