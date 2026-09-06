#pragma once

// rclcpp front end for the ROS-independent grasp pipeline, as a managed node.
//
// Measurement boundaries, stated once so the numbers can be argued with:
//
//   compute_ns    starts at subscription-callback entry, before any validation,
//                 and stops the instant before the JointTrajectory is handed to
//                 the publisher. It therefore includes message inspection and
//                 outgoing message construction, which the in-process benchmark
//                 does not pay.
//   end_to_end_ns is the node clock at that same instant minus the depth header
//                 stamp, so it additionally carries transport, deserialisation
//                 and executor dispatch. Under intra-process delivery the first
//                 two of those collapse to a pointer handoff, and the size of
//                 that collapse is one of the results this repository reports.
//
// Why a LifecycleNode: the pipeline reads three files and builds every scratch
// buffer at construction, which is tens of milliseconds of work that must not
// be attributable to a frame. A plain node does that in its constructor, before
// the executor exists, and there is then no observable instant at which the
// node is ready. on_configure gives that instant a name, and the benchmark
// script waits for it rather than sleeping and hoping.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <grasp_core/pipeline.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include <grasp_msgs/msg/grasp_latency.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace grasp_node_cpp
{

class GraspNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  explicit GraspNode(const rclcpp::NodeOptions & options);

  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & previous) override;

  // Read in the constructor because the smoke path in grasp_node_main.cpp has
  // to decide what to do with the node before any transition is requested.
  bool smoke_test() const {return smoke_test_;}

  // Frames processed since the last activation. The smoke path does not use it;
  // it exists so a run that produced no latency records can be told apart from
  // a run whose records were lost by the recorder.
  std::uint32_t processed() const {return seq_;}

private:
  void on_camera_info(const sensor_msgs::msg::CameraInfo & msg);
  void on_depth(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void fill_traj(const grasp_core::Result & result);
  void fill_stage_ns(const grasp_core::Result & result);

  std::unique_ptr<grasp_core::Pipeline> pipeline_;
  std::vector<std::string> joint_names_;
  std::vector<std::uint8_t> rgb_scratch_;

  trajectory_msgs::msg::JointTrajectory traj_;
  grasp_msgs::msg::GraspLatency latency_;
  std::uint32_t seq_{0};
  bool smoke_test_{false};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<
      trajectory_msgs::msg::JointTrajectory>> traj_pub_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<
      grasp_msgs::msg::GraspLatency>> latency_pub_;
};

}  // namespace grasp_node_cpp
