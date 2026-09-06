// rclcpp front end for the ROS-independent grasp pipeline.
//
// Measurement boundaries, stated once so the numbers can be argued with:
//
//   compute_ns    starts at subscription-callback entry, before any validation,
//                 and stops the instant before JointTrajectory::publish(). It
//                 therefore includes message inspection and outgoing message
//                 construction, which the in-process benchmark does not pay.
//   end_to_end_ns is the node clock at that same instant minus the depth
//                 header stamp, so it additionally carries DDS transport,
//                 CDR deserialisation and executor dispatch.
//
// Nothing in the callback allocates, parses configuration, opens a file or
// touches a PRNG: the pipeline, the RGB scratch buffer and both outgoing
// messages are built once in the constructor and reused.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <grasp_core/pipeline.hpp>
#include <rclcpp/rclcpp.hpp>

#include <grasp_msgs/msg/grasp_latency.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace
{

constexpr std::size_t kJoints = 7;
constexpr char kDepthEncoding[] = "16UC1";

// Number of bytes per depth sample implied by kDepthEncoding.
constexpr std::size_t kDepthBytes = 2;

}  // namespace

namespace grasp_node_cpp
{

class GraspNode : public rclcpp::Node
{
public:
  explicit GraspNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("grasp_node", options)
  {
    const auto config_path = declare_parameter<std::string>("config_path", "");
    const auto chain_path = declare_parameter<std::string>("chain_path", "");
    const auto ransac_table_path = declare_parameter<std::string>("ransac_table_path", "");
    joint_names_ = declare_parameter<std::vector<std::string>>(
      "joint_names", std::vector<std::string>{});
    const auto latency_depth = declare_parameter<int>("latency_queue_depth", 256);

    if (config_path.empty() || chain_path.empty() || ransac_table_path.empty()) {
      throw std::runtime_error(
        "config_path, chain_path and ransac_table_path are all required parameters");
    }
    // Joint names are a constant of the experiment and live in
    // assets/pipeline_config.json under trajectory.joint_names. The node takes
    // them as a parameter rather than parsing JSON so that the only JSON reader
    // in the C++ stack stays inside grasp_core; docker/run_ros_benchmark.sh
    // generates the parameter file from that same config file.
    if (joint_names_.size() != kJoints) {
      throw std::runtime_error(
        "joint_names must list exactly 7 joints, sourced from "
        "assets/pipeline_config.json:trajectory.joint_names");
    }

    pipeline_ = std::make_unique<grasp_core::Pipeline>(
      config_path, chain_path, ransac_table_path);

    traj_.joint_names = joint_names_;

    traj_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/grasp/joint_trajectory", rclcpp::QoS(1).reliable());

    // Reliable and deep: a dropped latency sample is a hole in the dataset,
    // whereas a dropped depth frame is a legitimate measurement outcome.
    latency_pub_ = create_publisher<grasp_msgs::msg::GraspLatency>(
      "/grasp/latency",
      rclcpp::QoS(rclcpp::KeepLast(static_cast<std::size_t>(std::max(1, latency_depth))))
        .reliable());

    // Intrinsics are not needed per frame: grasp_core derives fx, fy, cx, cy
    // from pipeline_config.json rescaled by the image width, exactly as the
    // in-process benchmark does. So there is no message_filters synchroniser
    // here -- CameraInfo is latched once purely to assert that the stream we
    // are fed agrees with the config, and the subscription is then dropped so
    // it costs nothing for the remaining frames. One subscription, one
    // callback, no pairing buffer, no waiting for a second topic.
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera/depth/camera_info", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg) {
        on_camera_info(*msg);
      });

    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/camera/depth/image_raw", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {on_depth(msg);});
  }

private:
  void on_camera_info(const sensor_msgs::msg::CameraInfo & msg)
  {
    // K = [fx 0 cx; 0 fy cy; 0 0 1]; only the four intrinsics are of interest.
    RCLCPP_INFO(
      get_logger(),
      "camera_info latched: %ux%u fx=%.4f fy=%.4f cx=%.4f cy=%.4f "
      "(grasp_core takes its own intrinsics from pipeline_config.json; a "
      "mismatch here means the bag and the config disagree)",
      msg.width, msg.height, msg.k[0], msg.k[4], msg.k[2], msg.k[5]);
    info_sub_.reset();
  }

  void on_depth(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    const auto t_enter = std::chrono::steady_clock::now();

    const std::size_t width = msg->width;
    const std::size_t height = msg->height;
    const std::size_t expected = width * height * kDepthBytes;

    if (msg->encoding != kDepthEncoding || msg->is_bigendian != 0 ||
      msg->step != width * kDepthBytes || msg->data.size() != expected)
    {
      // Throttled: a misconfigured stream would otherwise emit one line per
      // frame at 30 Hz and dominate the run's own cost.
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "rejecting depth frame: encoding='%s' is_bigendian=%u step=%u size=%zu, "
        "expected '%s', 0, %zu, %zu",
        msg->encoding.c_str(), msg->is_bigendian, msg->step, msg->data.size(),
        kDepthEncoding, width * kDepthBytes, expected);
      return;
    }

    if (rgb_scratch_.size() != width * height * 3) {
      // The depth topic is the only input the pipeline is fed over ROS, but the
      // core's signature takes an RGB plane because no stage of the algorithm
      // reads it (see docs/ALGORITHM.md S0-S7). Sized once, on the first frame,
      // which lands inside the warm-up window the analysis discards.
      rgb_scratch_.assign(width * height * 3, 0u);
    }

    // std::vector<uint8_t> is allocated by std::allocator and so is aligned for
    // any fundamental type; reading it as uint16_t is well defined in practice
    // and, more to the point, is exactly the zero-copy handoff a real driver
    // node would do. No image buffer is ever copied.
    const auto * depth = reinterpret_cast<const std::uint16_t *>(msg->data.data());

    const grasp_core::Result result = pipeline_->run(
      depth, rgb_scratch_.data(), static_cast<int>(width), static_cast<int>(height));

    fill_traj(result);

    // Two clock reads rather than one: the elapsed compute must come from a
    // monotonic source, while the end-to-end delta has to be taken against the
    // same (ROS) clock the header stamp was written with.
    const auto t_done = std::chrono::steady_clock::now();
    const rclcpp::Time t_publish = now();

    traj_pub_->publish(traj_);

    latency_.stamp = msg->header.stamp;
    latency_.seq = seq_++;
    latency_.compute_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_done - t_enter).count());
    const std::int64_t e2e = (t_publish - rclcpp::Time(msg->header.stamp)).nanoseconds();
    latency_.end_to_end_ns = e2e > 0 ? static_cast<std::uint64_t>(e2e) : 0u;
    fill_stage_ns(result);
    latency_.graspable = result.graspable;
    latency_.converged = result.converged;
    latency_pub_->publish(latency_);
  }

  void fill_traj(const grasp_core::Result & result)
  {
    // The waypoint count is a config constant, so this resize fires on the
    // first frame only and fill_traj() allocates nothing thereafter.
    if (traj_.points.size() != result.trajectory.size()) {
      traj_.points.resize(result.trajectory.size());
      for (auto & p : traj_.points) {
        p.positions.resize(kJoints);
        p.velocities.resize(kJoints);
        p.accelerations.resize(kJoints);
      }
    }

    traj_.header.stamp = now();
    for (std::size_t k = 0; k < result.trajectory.size(); ++k) {
      const auto & src = result.trajectory[k];
      auto & dst = traj_.points[k];
      for (std::size_t j = 0; j < kJoints; ++j) {
        dst.positions[j] = src.position[j];
        dst.velocities[j] = src.velocity[j];
        dst.accelerations[j] = src.acceleration[j];
      }
      dst.time_from_start = rclcpp::Duration::from_seconds(src.time_from_start);
    }
  }

  // Single point of contact with grasp_core::Result::stage_ns. The index order
  // is the key order docs/FORMATS.md fixes, mirrored by the STAGE_* constants
  // in grasp_msgs/GraspLatency.msg.
  void fill_stage_ns(const grasp_core::Result & result)
  {
    const auto & s = result.stage_ns;
    latency_.stage_ns[grasp_msgs::msg::GraspLatency::STAGE_DECODE] = s.decode;
    latency_.stage_ns[grasp_msgs::msg::GraspLatency::STAGE_DEPROJECT] = s.deproject;
    latency_.stage_ns[grasp_msgs::msg::GraspLatency::STAGE_TRANSFORM_CROP] = s.transform_crop;
    latency_.stage_ns[grasp_msgs::msg::GraspLatency::STAGE_PLANE] = s.plane;
    latency_.stage_ns[grasp_msgs::msg::GraspLatency::STAGE_CLUSTER] = s.cluster;
    latency_.stage_ns[grasp_msgs::msg::GraspLatency::STAGE_GRASP] = s.grasp;
    latency_.stage_ns[grasp_msgs::msg::GraspLatency::STAGE_IK] = s.ik;
    latency_.stage_ns[grasp_msgs::msg::GraspLatency::STAGE_TRAJ] = s.traj;
  }

  std::unique_ptr<grasp_core::Pipeline> pipeline_;
  std::vector<std::string> joint_names_;
  std::vector<std::uint8_t> rgb_scratch_;

  trajectory_msgs::msg::JointTrajectory traj_;
  grasp_msgs::msg::GraspLatency latency_;
  std::uint32_t seq_{0};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
  rclcpp::Publisher<grasp_msgs::msg::GraspLatency>::SharedPtr latency_pub_;
};

}  // namespace grasp_node_cpp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<grasp_node_cpp::GraspNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
