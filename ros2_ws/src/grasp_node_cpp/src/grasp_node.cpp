#include "grasp_node_cpp/grasp_node.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <string>
#include <vector>

#include <rclcpp_components/register_node_macro.hpp>

#include "grasp_node_cpp/qos.hpp"

namespace
{

constexpr std::size_t kJoints = 7;
constexpr char kDepthEncoding[] = "16UC1";

// Bytes per depth sample implied by kDepthEncoding.
constexpr std::size_t kDepthBytes = 2;

}  // namespace

namespace grasp_node_cpp
{

GraspNode::GraspNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("grasp_node", options)
{
  // The constructor declares and does nothing else. Reading a file here would
  // put model loading ahead of the executor, where no transition can be
  // observed to have finished and no failure can be reported as a failed
  // transition.
  declare_parameter<std::string>("config_path", "");
  declare_parameter<std::string>("chain_path", "");
  declare_parameter<std::string>("ransac_table_path", "");
  // Joint names are a constant of the experiment and live in
  // assets/pipeline_config.json under trajectory.joint_names. The node takes
  // them as a parameter rather than parsing JSON so that the only JSON reader
  // in the C++ stack stays inside grasp_core; docker/run_ros_benchmark.sh
  // generates the parameter file from that same config file.
  declare_parameter<std::vector<std::string>>("joint_names", std::vector<std::string>{});

  // Defaults reproduce what config/qos.yaml says, so a node started without
  // the file behaves the way the recorded runs did rather than differently.
  declare_qos(*this, "depth", "best_effort", "volatile", 5);
  declare_qos(*this, "camera_info", "best_effort", "volatile", 5);
  declare_qos(*this, "trajectory", "reliable", "volatile", 1);
  declare_qos(*this, "latency", "reliable", "volatile", 256);

  smoke_test_ = declare_parameter<bool>("smoke_test", false);
}

GraspNode::CallbackReturn GraspNode::on_configure(const rclcpp_lifecycle::State &)
{
  const auto config_path = get_parameter("config_path").as_string();
  const auto chain_path = get_parameter("chain_path").as_string();
  const auto ransac_table_path = get_parameter("ransac_table_path").as_string();
  joint_names_ = get_parameter("joint_names").as_string_array();

  if (config_path.empty() || chain_path.empty() || ransac_table_path.empty()) {
    RCLCPP_ERROR(
      get_logger(),
      "config_path, chain_path and ransac_table_path are all required; refusing to "
      "configure rather than measuring a pipeline built from defaults");
    return CallbackReturn::FAILURE;
  }
  if (joint_names_.size() != kJoints) {
    RCLCPP_ERROR(
      get_logger(),
      "joint_names lists %zu joints, expected 7, sourced from "
      "assets/pipeline_config.json:trajectory.joint_names",
      joint_names_.size());
    return CallbackReturn::FAILURE;
  }

  try {
    // Three files read, every scratch buffer sized, and the wrist reference FK
    // solved: all of it here, none of it in a callback.
    pipeline_ = std::make_unique<grasp_core::Pipeline>(
      config_path, chain_path, ransac_table_path);

    traj_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "joint_trajectory", resolve_qos(*this, "trajectory"));
    // Reliable and deep: a dropped latency sample is a hole in the dataset,
    // whereas a dropped depth frame is a legitimate measurement outcome.
    latency_pub_ = create_publisher<grasp_msgs::msg::GraspLatency>(
      "latency", resolve_qos(*this, "latency"));
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "configure failed: %s", error.what());
    pipeline_.reset();
    traj_pub_.reset();
    latency_pub_.reset();
    return CallbackReturn::FAILURE;
  }

  traj_.joint_names = joint_names_;
  seq_ = 0;

  RCLCPP_INFO(
    get_logger(),
    "configured: timer overhead %.1f ns per clock read, intra-process comms %s",
    pipeline_->timer_overhead_ns(),
    get_node_options().use_intra_process_comms() ? "on" : "off");
  return CallbackReturn::SUCCESS;
}

GraspNode::CallbackReturn GraspNode::on_activate(const rclcpp_lifecycle::State & previous)
{
  // The base implementation flips every managed entity, which is both
  // publishers, to activated. Skipping it would leave publish() dropping
  // messages with a warning.
  const auto base = LifecycleNode::on_activate(previous);
  if (base != CallbackReturn::SUCCESS) {
    return base;
  }

  // Subscriptions are created here and destroyed on deactivate, rather than
  // created once in on_configure and guarded by a state check. An inactive
  // node that still ran the pipeline would burn a core it does not need and
  // would put its own scheduling noise into the active node's tail.
  info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    "depth/camera_info", resolve_qos(*this, "camera_info"),
    [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg) {on_camera_info(*msg);});

  depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "depth/image_raw", resolve_qos(*this, "depth"),
    [this](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {on_depth(msg);});

  return CallbackReturn::SUCCESS;
}

GraspNode::CallbackReturn GraspNode::on_deactivate(const rclcpp_lifecycle::State & previous)
{
  depth_sub_.reset();
  info_sub_.reset();
  return LifecycleNode::on_deactivate(previous);
}

GraspNode::CallbackReturn GraspNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  // Everything on_configure took, given back, so a configure/cleanup cycle is
  // a real test of the release path rather than a leak that happens to work.
  depth_sub_.reset();
  info_sub_.reset();
  traj_pub_.reset();
  latency_pub_.reset();
  pipeline_.reset();
  rgb_scratch_.clear();
  rgb_scratch_.shrink_to_fit();
  traj_.points.clear();
  traj_.joint_names.clear();
  joint_names_.clear();
  return CallbackReturn::SUCCESS;
}

GraspNode::CallbackReturn GraspNode::on_shutdown(const rclcpp_lifecycle::State & previous)
{
  return on_cleanup(previous);
}

void GraspNode::on_camera_info(const sensor_msgs::msg::CameraInfo & msg)
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

void GraspNode::on_depth(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
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
  // node would do. No image buffer is ever copied, and under intra-process
  // delivery this pointer is into the publisher's own message.
  const auto * depth = reinterpret_cast<const std::uint16_t *>(msg->data.data());

  // By reference: Result owns a std::vector of waypoints, so binding this to a
  // value would allocate and copy once per frame inside the measured region.
  const grasp_core::Result & result = pipeline_->run(
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
  latency_.plane_found = result.plane_found;
  latency_.graspable = result.graspable;
  latency_.converged = result.converged;
  latency_.points = static_cast<std::uint32_t>(std::max(0, result.points));
  latency_.cluster_points = static_cast<std::uint32_t>(std::max(0, result.cluster_points));
  latency_.ik_iterations = static_cast<std::uint32_t>(std::max(0, result.iterations));
  latency_pub_->publish(latency_);
}

void GraspNode::fill_traj(const grasp_core::Result & result)
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
void GraspNode::fill_stage_ns(const grasp_core::Result & result)
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

}  // namespace grasp_node_cpp

// Registered so the node can be loaded into a component container alongside
// the frame publisher, which is the only way the depth image is delivered
// without a trip through the RMW. docker/run_ros_benchmark.sh measures both
// paths and tags the output with which one produced it.
RCLCPP_COMPONENTS_REGISTER_NODE(grasp_node_cpp::GraspNode)
