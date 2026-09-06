#include "grasp_core/config.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "grasp_core/json.hpp"

namespace grasp_core
{

Config Config::load(const std::string & path)
{
  const json::Value root = json::parse_file(path);
  Config c;

  const auto & cam = root["camera"];
  c.camera.reference_width = cam["reference_width"].integer();
  c.camera.reference_height = cam["reference_height"].integer();
  c.camera.fx = cam["fx"].number();
  c.camera.fy = cam["fy"].number();
  c.camera.cx = cam["cx"].number();
  c.camera.cy = cam["cy"].number();
  c.camera.depth_scale_m = cam["depth_scale_m"].number();
  cam["T_base_cam"].fill(c.camera.T_base_cam.data(), c.camera.T_base_cam.size());

  const auto & dep = root["deproject"];
  c.deproject.stride = dep["stride"].integer();
  c.deproject.z_min_m = dep["z_min_m"].number();
  c.deproject.z_max_m = dep["z_max_m"].number();
  if (c.deproject.stride < 1) {
    throw std::runtime_error("deproject.stride must be at least 1");
  }

  const auto & ws = root["workspace"];
  c.workspace.x_min = ws["x_min"].number();
  c.workspace.x_max = ws["x_max"].number();
  c.workspace.y_min = ws["y_min"].number();
  c.workspace.y_max = ws["y_max"].number();
  c.workspace.z_min = ws["z_min"].number();
  c.workspace.z_max = ws["z_max"].number();

  const auto & pl = root["plane"];
  c.plane.iterations = pl["iterations"].integer();
  c.plane.inlier_threshold_m = pl["inlier_threshold_m"].number();
  c.plane.min_horizontal_cos = pl["min_horizontal_cos"].number();
  c.plane.min_inliers = pl["min_inliers"].integer();
  c.plane.clearance_m = pl["clearance_m"].number();

  const auto & cl = root["cluster"];
  c.cluster.voxel_size_m = cl["voxel_size_m"].number();
  c.cluster.connectivity = cl["connectivity"].integer();
  c.cluster.min_points = cl["min_points"].integer();
  if (c.cluster.connectivity != 6) {
    throw std::runtime_error(
            "only 6-connectivity is implemented; cluster.connectivity says otherwise");
  }

  const auto & gr = root["grasp"];
  gr["approach_axis_base"].fill(
    c.grasp.approach_axis_base.data(), c.grasp.approach_axis_base.size());
  c.grasp.grasp_depth_m = gr["grasp_depth_m"].number();
  c.grasp.min_height_above_plane_m = gr["min_height_above_plane_m"].number();
  c.grasp.finger_clearance_m = gr["finger_clearance_m"].number();
  c.grasp.max_width_m = gr["max_width_m"].number();

  const auto & ik = root["ik"];
  c.ik.max_iterations = ik["max_iterations"].integer();
  c.ik.position_tolerance_m = ik["position_tolerance_m"].number();
  c.ik.orientation_tolerance_rad = ik["orientation_tolerance_rad"].number();
  c.ik.damping = ik["damping"].number();
  c.ik.nullspace_gain = ik["nullspace_gain"].number();
  c.ik.max_step_rad = ik["max_step_rad"].number();

  const auto & tr = root["trajectory"];
  c.trajectory.waypoints = tr["waypoints"].integer();
  c.trajectory.velocity_fraction = tr["velocity_fraction"].number();
  c.trajectory.min_duration_s = tr["min_duration_s"].number();
  c.trajectory.max_duration_s = tr["max_duration_s"].number();
  for (const auto & name : tr["joint_names"].array()) {
    c.trajectory.joint_names.push_back(name.string());
  }
  if (c.trajectory.waypoints < 2) {
    throw std::runtime_error("trajectory.waypoints must be at least 2");
  }

  c.ransac_table.count = root["ransac_table"]["count"].integer();

  const auto & bm = root["benchmark"];
  c.benchmark.warmup_frames = bm["warmup_frames"].integer();
  c.benchmark.measured_frames = bm["measured_frames"].integer();
  c.benchmark.deadline_hz = bm["deadline_hz"].number();

  return c;
}

std::vector<double> load_ransac_table(const std::string & path, int expected_count)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open RANSAC deviate table '" + path + "'");
  }
  in.seekg(0, std::ios::end);
  const std::streamoff bytes = in.tellg();
  in.seekg(0, std::ios::beg);

  const std::size_t count = static_cast<std::size_t>(bytes) / sizeof(double);
  if (count * sizeof(double) != static_cast<std::size_t>(bytes)) {
    throw std::runtime_error(path + " is not a whole number of float64 values");
  }
  if (expected_count > 0 && count != static_cast<std::size_t>(expected_count)) {
    throw std::runtime_error(
            path + " holds " + std::to_string(count) + " deviates, but " +
            "pipeline_config.json:ransac_table.count says " + std::to_string(expected_count));
  }

  std::vector<double> table(count);
  in.read(reinterpret_cast<char *>(table.data()), bytes);
  if (!in) {
    throw std::runtime_error("short read on " + path);
  }
  for (const double u : table) {
    if (!(u >= 0.0) || !(u < 1.0)) {
      throw std::runtime_error(path + " holds a deviate outside [0, 1)");
    }
  }
  return table;
}

}  // namespace grasp_core
