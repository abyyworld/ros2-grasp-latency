#pragma once

// A noise-free top-down render of one box on a level table, built by inverting
// the config's own camera transform. It is deliberately not a frame from
// data/: the C++ unit tests must run in CI before harness/make_dataset.py has
// been given a chance to write a corpus, and a fixture whose right answer is
// known in closed form is a better test than one whose right answer has to be
// looked up.

#include <cmath>
#include <cstdint>
#include <vector>

#include "grasp_core/config.hpp"

namespace grasp_test
{

struct Scene
{
  int width{0};
  int height{0};
  std::vector<std::uint16_t> depth;
  std::vector<std::uint8_t> rgb;

  double box_centre_x{0.0};
  double box_centre_y{0.0};
  double box_half_x{0.0};
  double box_half_y{0.0};
  double box_height{0.0};
};

inline Scene render_box_on_table(const grasp_core::Config & config, int width, int height)
{
  Scene scene;
  scene.width = width;
  scene.height = height;
  scene.box_centre_x = 0.45;
  scene.box_centre_y = -0.05;
  scene.box_half_x = 0.02;    // narrow direction: what the fingers close along
  scene.box_half_y = 0.035;
  scene.box_height = 0.05;

  const double scale = static_cast<double>(width) /
    static_cast<double>(config.camera.reference_width);
  const double fx = config.camera.fx * scale;
  const double fy = config.camera.fy * scale;
  const double cx = config.camera.cx * scale;
  const double cy = config.camera.cy * scale;

  // p_base = R * p_cam + t, with R and t straight out of the config, so the
  // fixture cannot drift from the transform the pipeline applies.
  const auto & m = config.camera.T_base_cam;
  const double r[3][3] = {
    {m[0], m[1], m[2]},
    {m[4], m[5], m[6]},
    {m[8], m[9], m[10]}};
  const double t[3] = {m[3], m[7], m[11]};

  auto to_base = [&](double a, double b, double z, double * out) {
      const double c[3] = {a * z, b * z, z};
      for (int i = 0; i < 3; ++i) {
        out[i] = r[i][0] * c[0] + r[i][1] * c[1] + r[i][2] * c[2] + t[i];
      }
    };

  // Depth of the table and of the box top, found by walking z until the base
  // height matches. Both surfaces are level, so one bisection each is enough.
  auto depth_for_height = [&](double target_z) {
      double lo = config.deproject.z_min_m;
      double hi = config.deproject.z_max_m;
      for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        double p[3];
        to_base(0.0, 0.0, mid, p);
        if (p[2] > target_z) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      return 0.5 * (lo + hi);
    };

  const double z_table = depth_for_height(0.0);
  const double z_box = depth_for_height(scene.box_height);

  scene.depth.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0u);
  scene.rgb.assign(
    static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u, 0u);

  const double table_centre_x = 0.40;
  const double table_half = 0.35;

  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const double a = (static_cast<double>(u) - cx) / fx;
      const double b = (static_cast<double>(v) - cy) / fy;

      double p[3];
      to_base(a, b, z_box, p);
      double z = 0.0;
      if (std::abs(p[0] - scene.box_centre_x) <= scene.box_half_x &&
        std::abs(p[1] - scene.box_centre_y) <= scene.box_half_y)
      {
        z = z_box;
      } else {
        to_base(a, b, z_table, p);
        if (std::abs(p[0] - table_centre_x) <= table_half && std::abs(p[1]) <= table_half) {
          z = z_table;
        }
      }
      const auto millimetres = static_cast<std::uint16_t>(
        std::lround(z / config.camera.depth_scale_m));
      scene.depth[static_cast<std::size_t>(v) * static_cast<std::size_t>(width) +
        static_cast<std::size_t>(u)] = millimetres;
    }
  }
  return scene;
}

}  // namespace grasp_test
