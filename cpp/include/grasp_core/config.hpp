#pragma once

// Every tuning constant in the pipeline, read once from
// assets/pipeline_config.json. tests/test_config_is_sole_source.py greps the
// sources to make sure none of them is also typed in by hand.

#include <array>
#include <string>
#include <vector>

namespace grasp_core
{

struct Config
{
  struct Camera
  {
    int reference_width{0};
    int reference_height{0};
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
    double depth_scale_m{0.0};
    std::array<double, 16> T_base_cam{};   // row-major
  } camera;

  struct Deproject
  {
    int stride{1};
    double z_min_m{0.0};
    double z_max_m{0.0};
  } deproject;

  struct Workspace
  {
    double x_min{0.0}, x_max{0.0};
    double y_min{0.0}, y_max{0.0};
    double z_min{0.0}, z_max{0.0};
  } workspace;

  struct Plane
  {
    int iterations{0};
    double inlier_threshold_m{0.0};
    double min_horizontal_cos{0.0};
    int min_inliers{0};
    double clearance_m{0.0};
  } plane;

  struct Cluster
  {
    double voxel_size_m{0.0};
    int connectivity{0};
    int min_points{0};
  } cluster;

  struct Grasp
  {
    std::array<double, 3> approach_axis_base{};
    double grasp_depth_m{0.0};
    double min_height_above_plane_m{0.0};
    double finger_clearance_m{0.0};
    double max_width_m{0.0};
  } grasp;

  struct Ik
  {
    int max_iterations{0};
    double position_tolerance_m{0.0};
    double orientation_tolerance_rad{0.0};
    double damping{0.0};
    double nullspace_gain{0.0};
    double max_step_rad{0.0};
  } ik;

  struct Trajectory
  {
    int waypoints{0};
    double velocity_fraction{0.0};
    double min_duration_s{0.0};
    double max_duration_s{0.0};
    std::vector<std::string> joint_names;
  } trajectory;

  struct RansacTable
  {
    int count{0};
  } ransac_table;

  struct Benchmark
  {
    int warmup_frames{0};
    int measured_frames{0};
    double deadline_hz{0.0};
  } benchmark;

  static Config load(const std::string & path);
};

// The deviate table backing S3. Its length must equal ransac_table.count, and
// the pipeline reads three entries per RANSAC iteration, so a short table is a
// start-up error rather than an out-of-range read on frame one.
std::vector<double> load_ransac_table(const std::string & path, int expected_count);

}  // namespace grasp_core
