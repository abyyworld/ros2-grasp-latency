// Live driver for the frame store, standing in for a camera.
//
// Why this exists rather than just `ros2 bag play`: rosbag2 replays the bytes
// it recorded, header stamp included, and offers no way to restamp on the way
// out (checked against the Jazzy play verb: there is --clock, --rate, --loop
// and --start-offset, but nothing that rewrites a message field). So under bag
// play, `now - header.stamp` is the true latency plus the unknown, constant
// offset between the bag's time origin and the moment playback started, and an
// absolute end-to-end figure cannot be recovered from it. This node stamps at
// publish, so end_to_end_ns from a run driven by it is the real number.
//
// Every frame is read from disk and turned into a complete Image message at
// construction. Publishing writes one header stamp and hands the message over;
// no file is touched and nothing is allocated once the timer is running.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace grasp_node_cpp
{

class FramePublisher : public rclcpp::Node
{
public:
  explicit FramePublisher(const rclcpp::NodeOptions & options)
  : rclcpp::Node("frame_publisher", options)
  {
    const auto frames_dir = declare_parameter<std::string>("frames_dir", "");
    const auto width = declare_parameter<int>("width", 0);
    const auto height = declare_parameter<int>("height", 0);
    const auto frame_count = declare_parameter<int>("frame_count", 0);
    const auto rate_hz = declare_parameter<double>("rate_hz", 30.0);
    total_frames_ = static_cast<std::uint64_t>(declare_parameter<int>("total_frames", 0));
    const auto frame_id = declare_parameter<std::string>("frame_id", "camera_depth_optical_frame");

    if (frames_dir.empty() || width <= 0 || height <= 0 || frame_count <= 0 || rate_hz <= 0.0) {
      throw std::runtime_error(
        "frames_dir, width, height, frame_count and rate_hz are all required parameters");
    }

    const std::size_t bytes = static_cast<std::size_t>(width) * height * 2;
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

    info_.header.frame_id = frame_id;
    info_.height = static_cast<std::uint32_t>(height);
    info_.width = static_cast<std::uint32_t>(width);
    info_.distortion_model = "plumb_bob";
    info_.d.assign(5, 0.0);
    const double fx = declare_parameter<double>("fx", 0.0);
    const double fy = declare_parameter<double>("fy", 0.0);
    const double cx = declare_parameter<double>("cx", 0.0);
    const double cy = declare_parameter<double>("cy", 0.0);
    info_.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
    info_.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    info_.p = {fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0};

    depth_pub_ = create_publisher<sensor_msgs::msg::Image>(
      "/camera/depth/image_raw", rclcpp::SensorDataQoS());
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      "/camera/depth/camera_info", rclcpp::SensorDataQoS());

    const auto period = std::chrono::nanoseconds(
      static_cast<std::int64_t>(1e9 / rate_hz));
    timer_ = create_wall_timer(period, [this]() {tick();});

    // The consuming node latches intrinsics once and then drops the
    // subscription, so republishing CameraInfo per frame would only burn
    // bandwidth on a topic nobody is listening to after the first message.
    info_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {publish_info();});
  }

private:
  void publish_info()
  {
    info_.header.stamp = now();
    info_pub_->publish(info_);
  }

  void tick()
  {
    if (total_frames_ != 0 && published_ >= total_frames_) {
      RCLCPP_INFO(
        get_logger(), "published %lu frames, shutting down",
        static_cast<unsigned long>(published_));
      timer_->cancel();
      rclcpp::shutdown();
      return;
    }

    auto & msg = frames_[published_ % frames_.size()];
    msg.header.stamp = now();
    depth_pub_->publish(msg);
    ++published_;
  }

  std::vector<sensor_msgs::msg::Image> frames_;
  sensor_msgs::msg::CameraInfo info_;
  std::uint64_t published_{0};
  std::uint64_t total_frames_{0};

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr info_timer_;
};

}  // namespace grasp_node_cpp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<grasp_node_cpp::FramePublisher>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
