#include "grasp_core/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "grasp_core/chain.hpp"
#include "grasp_core/cluster.hpp"
#include "grasp_core/config.hpp"
#include "grasp_core/kinematics.hpp"
#include "grasp_core/plane.hpp"
#include "grasp_core/timing.hpp"
#include "grasp_core/trajectory.hpp"

namespace grasp_core
{
namespace
{

constexpr int kTimerCalibrationSamples = 4096;

void canonicalise2(Eigen::Vector2d & v)
{
  const int best = (std::abs(v(1)) > std::abs(v(0))) ? 1 : 0;
  if (v(best) < 0.0) {
    v = -v;
  }
}

}  // namespace

struct Pipeline::Impl
{
  Impl(
    const std::string & config_path, const std::string & chain_path,
    const std::string & table_path)
  : config(Config::load(config_path)),
    chain(Chain::load(chain_path)),
    deviates(load_ransac_table(table_path, config.ransac_table.count)),
    plane(config, deviates),
    cluster(config),
    ik(config, chain),
    timer_overhead_ns(calibrate_timer_overhead_ns(kTimerCalibrationSamples))
  {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        rotation(r, c) = config.camera.T_base_cam[static_cast<std::size_t>(4 * r + c)];
      }
      translation(r) = config.camera.T_base_cam[static_cast<std::size_t>(4 * r + 3)];
    }
    result.trajectory.resize(static_cast<std::size_t>(config.trajectory.waypoints));

    // The wrist's rest orientation, used in S5 to pick which of the two
    // equivalent closing-axis signs to report. Computed once, here, because
    // it is a property of the chain and never of a frame.
    const Eigen::Matrix4d neutral = fk_tcp(chain, chain.q_neutral.data());
    y_reference = Eigen::Vector2d(neutral(0, 1), neutral(1, 1));
    const double horizontal = y_reference.norm();
    if (horizontal > 0.0) {
      y_reference /= horizontal;
    }
  }

  void reserve(int width, int height);

  Config config;
  Chain chain;
  std::vector<double> deviates;
  PlaneFitter plane;
  Clusterer cluster;
  IkSolver ik;
  double timer_overhead_ns;

  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
  Eigen::Vector2d y_reference{Eigen::Vector2d::Zero()};

  // Intrinsics for the reserved resolution, derived from the reference stream
  // by the rule in ALGORITHM.md S1. Recomputed on reserve() rather than per
  // frame: a division that never changes has no business inside a measured
  // stage.
  double fx{0.0}, fy{0.0}, cx{0.0}, cy{0.0};
  int reserved_width{0};
  int reserved_height{0};

  std::vector<float> points_cam;
  std::vector<float> points_base;
  std::vector<float> points_object;
  std::vector<int> cluster_indices;

  // S0 leaves typed views of the caller's buffers here. They are read by S1,
  // which is what stops the compiler from folding the stage away.
  const std::uint16_t * depth_view{nullptr};
  const std::uint8_t * rgb_view{nullptr};
  int view_width{0};
  int view_height{0};

  Result result;
};

void Pipeline::Impl::reserve(int width, int height)
{
  if (width == reserved_width && height == reserved_height) {
    return;
  }
  const int stride = config.deproject.stride;
  const int cols = (width + stride - 1) / stride;
  const int rows = (height + stride - 1) / stride;
  const std::size_t max_points = static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows);

  points_cam.resize(3u * max_points);
  points_base.resize(3u * max_points);
  points_object.resize(3u * max_points);
  cluster_indices.resize(max_points);
  plane.reserve(max_points);
  cluster.reserve(max_points);

  // ALGORITHM.md S1: square pixels, so one focal length, and it tracks the
  // vertical resolution because a wider sensor mode is a wider field of view
  // at the same focal length rather than a stretched image. The principal
  // point is the centre of the image that arrives. Written in this order
  // rather than folded into a scale factor so it rounds the same way as the
  // Python, which the 1e-6 equivalence gate is tight enough to notice.
  fx = fy = config.camera.fy * static_cast<double>(height) /
    static_cast<double>(config.camera.reference_height);
  cx = (static_cast<double>(width) - 1.0) / 2.0;
  cy = (static_cast<double>(height) - 1.0) / 2.0;

  reserved_width = width;
  reserved_height = height;
}

Pipeline::Pipeline(
  const std::string & config_path, const std::string & chain_path,
  const std::string & ransac_table_path)
: impl_(std::make_unique<Impl>(config_path, chain_path, ransac_table_path)) {}

Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline &&) noexcept = default;
Pipeline & Pipeline::operator=(Pipeline &&) noexcept = default;

void Pipeline::reserve(int width, int height) {impl_->reserve(width, height);}
const Config & Pipeline::config() const {return impl_->config;}
const Chain & Pipeline::chain() const {return impl_->chain;}
double Pipeline::timer_overhead_ns() const {return impl_->timer_overhead_ns;}

const Result & Pipeline::run(
  const std::uint16_t * depth, const std::uint8_t * rgb, int width, int height)
{
  Impl & s = *impl_;
  s.reserve(width, height);

  Result & r = s.result;
  const Config & cfg = s.config;

  const std::uint64_t t_start = monotonic_ns();

  // ---- S0 decode -------------------------------------------------------
  s.depth_view = depth;
  s.rgb_view = rgb;
  s.view_width = width;
  s.view_height = height;
  const std::uint64_t t_decode = monotonic_ns();

  // ---- S1 deproject ----------------------------------------------------
  const int stride = cfg.deproject.stride;
  const double depth_scale = cfg.camera.depth_scale_m;
  const double z_min = cfg.deproject.z_min_m;
  const double z_max = cfg.deproject.z_max_m;
  float * cam = s.points_cam.data();
  int n_cam = 0;
  for (int v = 0; v < s.view_height; v += stride) {
    const std::uint16_t * row = s.depth_view + static_cast<std::size_t>(v) *
      static_cast<std::size_t>(s.view_width);
    for (int u = 0; u < s.view_width; u += stride) {
      const double z = static_cast<double>(row[u]) * depth_scale;
      if (z < z_min || z > z_max) {
        continue;
      }
      cam[3 * n_cam + 0] = static_cast<float>((static_cast<double>(u) - s.cx) * z / s.fx);
      cam[3 * n_cam + 1] = static_cast<float>((static_cast<double>(v) - s.cy) * z / s.fy);
      cam[3 * n_cam + 2] = static_cast<float>(z);
      ++n_cam;
    }
  }
  const std::uint64_t t_deproject = monotonic_ns();

  // ---- S2 transform and crop -------------------------------------------
  const double r00 = s.rotation(0, 0), r01 = s.rotation(0, 1), r02 = s.rotation(0, 2);
  const double r10 = s.rotation(1, 0), r11 = s.rotation(1, 1), r12 = s.rotation(1, 2);
  const double r20 = s.rotation(2, 0), r21 = s.rotation(2, 1), r22 = s.rotation(2, 2);
  const double tx = s.translation(0), ty = s.translation(1), tz = s.translation(2);
  const Config::Workspace & ws = cfg.workspace;
  float * base = s.points_base.data();
  int n_base = 0;
  for (int i = 0; i < n_cam; ++i) {
    const double px = cam[3 * i + 0];
    const double py = cam[3 * i + 1];
    const double pz = cam[3 * i + 2];
    const double bx = r00 * px + r01 * py + r02 * pz + tx;
    if (bx < ws.x_min || bx > ws.x_max) {
      continue;
    }
    const double by = r10 * px + r11 * py + r12 * pz + ty;
    if (by < ws.y_min || by > ws.y_max) {
      continue;
    }
    const double bz = r20 * px + r21 * py + r22 * pz + tz;
    if (bz < ws.z_min || bz > ws.z_max) {
      continue;
    }
    base[3 * n_base + 0] = static_cast<float>(bx);
    base[3 * n_base + 1] = static_cast<float>(by);
    base[3 * n_base + 2] = static_cast<float>(bz);
    ++n_base;
  }
  r.points = n_base;
  const std::uint64_t t_transform = monotonic_ns();

  // ---- S3 plane removal -------------------------------------------------
  int n_object = 0;
  s.plane.run(base, n_base, r.plane, r.plane_found, s.points_object.data(), n_object);
  const std::uint64_t t_plane = monotonic_ns();

  // ---- S4 cluster --------------------------------------------------------
  const float * object = s.points_object.data();
  const int n_cluster = s.cluster.run(object, n_object, s.cluster_indices.data());
  r.cluster_points = n_cluster;
  const std::uint64_t t_cluster = monotonic_ns();

  // ---- S5 grasp synthesis -------------------------------------------------
  Eigen::Matrix4d target = Eigen::Matrix4d::Identity();
  r.graspable = false;
  r.width = 0.0;
  r.tcp.fill(0.0);
  if (n_cluster > 0) {
    const int * idx = s.cluster_indices.data();
    const double inv_count = 1.0 / static_cast<double>(n_cluster);
    double sum_x = 0.0, sum_y = 0.0;
    double z_top = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_cluster; ++i) {
      const float * p = object + 3 * idx[i];
      sum_x += static_cast<double>(p[0]);
      sum_y += static_cast<double>(p[1]);
      z_top = std::max(z_top, static_cast<double>(p[2]));
    }
    const double centre_x = sum_x * inv_count;
    const double centre_y = sum_y * inv_count;

    double sxx = 0.0, sxy = 0.0, syy = 0.0;
    for (int i = 0; i < n_cluster; ++i) {
      const float * p = object + 3 * idx[i];
      const double dx = static_cast<double>(p[0]) - centre_x;
      const double dy = static_cast<double>(p[1]) - centre_y;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
    }
    Eigen::Matrix2d cov;
    cov << sxx * inv_count, sxy * inv_count,
      sxy * inv_count, syy * inv_count;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
    // Ascending eigenvalues, so column zero is the narrow direction: the one
    // the fingers close along.
    Eigen::Vector2d minor = solver.eigenvectors().col(0);
    canonicalise2(minor);

    double proj_min = std::numeric_limits<double>::infinity();
    double proj_max = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_cluster; ++i) {
      const float * p = object + 3 * idx[i];
      const double proj = minor(0) * static_cast<double>(p[0]) +
        minor(1) * static_cast<double>(p[1]);
      proj_min = std::min(proj_min, proj);
      proj_max = std::max(proj_max, proj);
    }
    r.width = (proj_max - proj_min) + cfg.grasp.finger_clearance_m;

    const Eigen::Vector3d z_tcp(
      cfg.grasp.approach_axis_base[0], cfg.grasp.approach_axis_base[1],
      cfg.grasp.approach_axis_base[2]);
    Eigen::Vector3d y_tcp(minor(0), minor(1), 0.0);
    y_tcp.normalize();

    // A parallel-jaw gripper is symmetric about its closing axis, so +y and -y
    // name the same physical grasp. Rule 3 canonicalises the eigenvector
    // against its own components, which knows nothing about the arm and can
    // demand up to half a turn of joint 7. Folding toward the wrist's rest
    // orientation caps the demanded yaw at a quarter turn and changes which of
    // two equivalent frames is reported, not what is grasped.
    const double alignment = y_tcp.x() * s.y_reference.x() + y_tcp.y() * s.y_reference.y();
    if (alignment < 0.0) {
      y_tcp = -y_tcp;
    } else if (alignment == 0.0 && (y_tcp.x() < 0.0 || (y_tcp.x() == 0.0 && y_tcp.y() < 0.0))) {
      y_tcp = -y_tcp;
    }
    const Eigen::Vector3d x_tcp = y_tcp.cross(z_tcp);

    Eigen::Matrix3d basis;
    basis.col(0) = x_tcp;
    basis.col(1) = y_tcp;
    basis.col(2) = z_tcp;

    double z_grasp = z_top - cfg.grasp.grasp_depth_m;
    if (r.plane_found) {
      // Height of the refit plane under the cluster centroid. The table is
      // tilted by a few degrees in every frame, so this is not z = 0.
      const double plane_z =
        -(r.plane[0] * centre_x + r.plane[1] * centre_y + r.plane[3]) / r.plane[2];
      z_grasp = std::max(z_grasp, plane_z + cfg.grasp.min_height_above_plane_m);
    }

    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        r.tcp[static_cast<std::size_t>(4 * row + col)] = basis(row, col);
      }
    }
    r.tcp[3] = centre_x;
    r.tcp[7] = centre_y;
    r.tcp[11] = z_grasp;
    r.tcp[15] = 1.0;

    // A right-handed frame is an invariant of the construction above, not a
    // property of the data, so a failure here means the minor axis degenerated
    // rather than that this object cannot be grasped.
    const bool right_handed = std::abs(basis.determinant() - 1.0) < 1e-9;
    r.graspable = right_handed && (r.width <= cfg.grasp.max_width_m);

    target.topLeftCorner<3, 3>() = basis;
    target(0, 3) = centre_x;
    target(1, 3) = centre_y;
    target(2, 3) = z_grasp;
  }
  const std::uint64_t t_grasp = monotonic_ns();

  // ---- S6 inverse kinematics ----------------------------------------------
  // Run whenever a pose exists. `graspable` is a gripper-width verdict, and a
  // frame that fails it still has a target the solver can be asked for, so
  // excluding it here would make the latency distribution depend on the width
  // test rather than on the language.
  if (n_cluster > 0) {
    r.converged = s.ik.solve(target, r.q, r.iterations);
  } else {
    r.q = s.chain.q_neutral;
    r.converged = false;
    r.iterations = 0;
  }
  const std::uint64_t t_ik = monotonic_ns();

  // ---- S7 trajectory --------------------------------------------------------
  if (n_cluster > 0) {
    generate_trajectory(cfg, s.chain, r.q, r.trajectory, r.duration_s);
  } else {
    for (auto & w : r.trajectory) {
      w.position.fill(0.0);
      w.velocity.fill(0.0);
      w.acceleration.fill(0.0);
      w.time_from_start = 0.0;
    }
    r.duration_s = 0.0;
  }
  const std::uint64_t t_traj = monotonic_ns();

  r.stage_ns.decode = t_decode - t_start;
  r.stage_ns.deproject = t_deproject - t_decode;
  r.stage_ns.transform_crop = t_transform - t_deproject;
  r.stage_ns.plane = t_plane - t_transform;
  r.stage_ns.cluster = t_cluster - t_plane;
  r.stage_ns.grasp = t_grasp - t_cluster;
  r.stage_ns.ik = t_ik - t_grasp;
  r.stage_ns.traj = t_traj - t_ik;
  r.total_ns = t_traj - t_start;
  return r;
}

}  // namespace grasp_core
