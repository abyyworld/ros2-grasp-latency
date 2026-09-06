#include "grasp_node_cpp/frame_publisher.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <ios>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <rclcpp_components/register_node_macro.hpp>

#include "grasp_node_cpp/qos.hpp"

namespace grasp_node_cpp
{

FramePublisher::FramePublisher(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("frame_publisher", options)
{
  declare_parameter<std::string>("frames_dir", "");
  declare_parameter<int>("width", 0);
  declare_parameter<int>("height", 0);
  declare_parameter<int>("frame_count", 0);
  declare_parameter<int>("total_frames", 0);
  declare_parameter<double>("rate_hz", 30.0);
  declare_parameter<std::string>("frame_id", "camera_depth_optical_frame");
  declare_parameter<double>("fx", 0.0);
  declare_parameter<double>("fy", 0.0);
  declare_parameter<double>("cx", 0.0);
  declare_parameter<double>("cy", 0.0);

  // The same two profiles the consumer resolves, from the same config/qos.yaml.
  // A driver that offered a different reliability would simply never match.
  declare_qos(*this, "depth", "best_effort", "volatile", 5);
  declare_qos(*this, "camera_info", "best_effort", "volatile", 5);
}

FramePublisher::CallbackReturn FramePublisher::on_configure(const rclcpp_lifecycle::State &)
{
  const auto frames_dir = get_parameter("frames_dir").as_string();
  const auto width = static_cast<int>(get_parameter("width").as_int());
  const auto height = static_cast<int>(get_parameter("height").as_int());
  const auto frame_count = static_cast<int>(get_parameter("frame_count").as_int());
  const auto rate_hz = get_parameter("rate_hz").as_double();
  const auto frame_id = get_parameter("frame_id").as_string();
  total_frames_ = static_cast<std::uint64_t>(get_parameter("total_frames").as_int());

  if (frames_dir.empty() || width <= 0 || height <= 0 || frame_count <= 0 || rate_hz <= 0.0) {
    RCLCPP_ERROR(
      get_logger(),
      "frames_dir, width, height, frame_count and rate_hz are all required");
    return CallbackReturn::FAILURE;
  }

  // Every frame is read from disk here and turned into a complete Image
  // message. Publishing then writes one header stamp and hands the message
  // over: no file is touched once the timer is running.
  const std::size_t bytes = static_cast<std::size_t>(width) * height * 2;
  try {
    frames_.clear();
    frames_.reserve(static_cast<std::size_t>(frame_count));
    for (int i = 0; i < frame_count; ++i) {
      char name[32];
      std::snprintf(name, sizeof(name), "/%06d.depth.bin", i);
      const std::string path = frames_dir + name;

      std::ifstream in(path, std::ios::binary);
      if (!in) {
        throw std::runtime_error("cannot open frame " + path);
      }

      sensor_msgs::msg::Image msg;
      msg.header.frame_id = frame_id;
      msg.height = static_cast<std::uint32_t>(height);
      msg.width = static_cast<std::uint32_t>(width);
      msg.encoding = "16UC1";
      msg.is_bigendian = 0;
      msg.step = static_cast<std::uint32_t>(width) * 2;
      msg.data.resize(bytes);
      in.read(reinterpret_cast<char *>(msg.data.data()), static_cast<std::streamsize>(bytes));
      if (static_cast<std::size_t>(in.gcount()) != bytes) {
        throw std::runtime_error("short read on " + path);
      }
      frames_.push_back(std::move(msg));
    }

    depth_pub_ = create_publisher<sensor_msgs::msg::Image>(
      "depth/image_raw", resolve_qos(*this, "depth"));
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      "depth/camera_info", resolve_qos(*this, "camera_info"));
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "configure failed: %s", error.what());
    frames_.clear();
    depth_pub_.reset();
    info_pub_.reset();
    return CallbackReturn::FAILURE;
  }

  info_.header.frame_id = frame_id;
  info_.height = static_cast<std::uint32_t>(height);
  info_.width = static_cast<std::uint32_t>(width);
  info_.distortion_model = "plumb_bob";
  info_.d.assign(5, 0.0);
  const double fx = get_parameter("fx").as_double();
  const double fy = get_parameter("fy").as_double();
  const double cx = get_parameter("cx").as_double();
  const double cy = get_parameter("cy").as_double();
  info_.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
  info_.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  info_.p = {fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0};

  period_ = std::chrono::nanoseconds(static_cast<std::int64_t>(1e9 / rate_hz));
  published_ = 0;

  RCLCPP_INFO(
    get_logger(), "configured: %d frames of %dx%d in memory, %.1f Hz, intra-process comms %s",
    frame_count, width, height, rate_hz,
    get_node_options().use_intra_process_comms() ? "on" : "off");
  return CallbackReturn::SUCCESS;
}

FramePublisher::CallbackReturn FramePublisher::on_activate(const rclcpp_lifecycle::State & previous)
{
  const auto base = LifecycleNode::on_activate(previous);
  if (base != CallbackReturn::SUCCESS) {
    return base;
  }

  // The timers exist only while active, so the container can hold this node
  // loaded and silent until the consumer under measurement is ready. Frames
  // published into an inactive consumer would be dropped and would shift the
  // warm-up window by an amount nobody recorded.
  published_ = 0;
  timer_ = create_wall_timer(period_, [this]() {tick();});
  // The consumer latches intrinsics once and then drops the subscription, so
  // republishing CameraInfo per frame would only burn bandwidth on a topic
  // nobody is listening to after the first message.
  info_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {publish_info();});
  return CallbackReturn::SUCCESS;
}

FramePublisher::CallbackReturn FramePublisher::on_deactivate(
  const rclcpp_lifecycle::State & previous)
{
  timer_.reset();
  info_timer_.reset();
  return LifecycleNode::on_deactivate(previous);
}

FramePublisher::CallbackReturn FramePublisher::on_cleanup(const rclcpp_lifecycle::State &)
{
  timer_.reset();
  info_timer_.reset();
  depth_pub_.reset();
  info_pub_.reset();
  frames_.clear();
  frames_.shrink_to_fit();
  return CallbackReturn::SUCCESS;
}

FramePublisher::CallbackReturn FramePublisher::on_shutdown(
  const rclcpp_lifecycle::State & previous)
{
  return on_cleanup(previous);
}

void FramePublisher::publish_info()
{
  info_.header.stamp = now();
  info_pub_->publish(info_);
}

void FramePublisher::tick()
{
  if (total_frames_ != 0 && published_ >= total_frames_) {
    RCLCPP_INFO_ONCE(
      get_logger(), "published %lu frames, idling",
      static_cast<unsigned long>(published_));
    timer_->cancel();
    return;
  }

  // A unique_ptr rather than a const reference, because that is the only shape
  // rclcpp can hand to a single intra-process subscriber without copying. The
  // copy out of the frame store happens here, before the stamp is written, so
  // it is outside every latency this run reports.
  auto msg = std::make_unique<sensor_msgs::msg::Image>(frames_[published_ % frames_.size()]);
  msg->header.stamp = now();
  depth_pub_->publish(std::move(msg));
  ++published_;
}

}  // namespace grasp_node_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(grasp_node_cpp::FramePublisher)
