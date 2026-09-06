#include "grasp_core/plane.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace grasp_core
{
namespace
{

// Rule 3 of docs/ALGORITHM.md: flip so the largest-magnitude component is
// positive, and let the lower index decide when two magnitudes tie.
void canonicalise(Eigen::Vector3d & v)
{
  int best = 0;
  for (int i = 1; i < 3; ++i) {
    if (std::abs(v(i)) > std::abs(v(best))) {
      best = i;
    }
  }
  if (v(best) < 0.0) {
    v = -v;
  }
}

}  // namespace

PlaneFitter::PlaneFitter(const Config & config, const std::vector<double> & deviates)
: cfg_(config.plane), deviates_(deviates)
{
  const std::size_t needed = static_cast<std::size_t>(cfg_.iterations) * 3u;
  if (deviates_.size() < needed) {
    throw std::runtime_error(
            "the RANSAC deviate table holds " + std::to_string(deviates_.size()) +
            " values but plane.iterations needs " + std::to_string(needed));
  }
}

void PlaneFitter::reserve(std::size_t max_points)
{
  if (inliers_.size() < max_points) {
    inliers_.resize(max_points);
  }
}

void PlaneFitter::run(
  const float * points, int n, std::array<double, 4> & plane, bool & found,
  float * out, int & out_count)
{
  plane = {0.0, 0.0, 0.0, 0.0};
  found = false;

  if (n < 3) {
    std::memcpy(out, points, static_cast<std::size_t>(n) * 3u * sizeof(float));
    out_count = n;
    return;
  }

  const double threshold = cfg_.inlier_threshold_m;

  int best_score = 0;
  double best_n[3] = {0.0, 0.0, 0.0};
  double best_d = 0.0;

  for (int k = 0; k < cfg_.iterations; ++k) {
    const std::size_t base = static_cast<std::size_t>(k) * 3u;
    // floor(u * M) clamped to M-1: u < 1 by construction, but the clamp is
    // what the spec says and it costs nothing.
    const int i0 = std::min(
      n - 1, static_cast<int>(std::floor(deviates_[base + 0] * static_cast<double>(n))));
    const int i1 = std::min(
      n - 1, static_cast<int>(std::floor(deviates_[base + 1] * static_cast<double>(n))));
    const int i2 = std::min(
      n - 1, static_cast<int>(std::floor(deviates_[base + 2] * static_cast<double>(n))));
    if (i0 == i1 || i1 == i2 || i0 == i2) {
      continue;
    }

    const float * p0 = points + 3 * i0;
    const float * p1 = points + 3 * i1;
    const float * p2 = points + 3 * i2;
    const double ax = static_cast<double>(p1[0]) - p0[0];
    const double ay = static_cast<double>(p1[1]) - p0[1];
    const double az = static_cast<double>(p1[2]) - p0[2];
    const double bx = static_cast<double>(p2[0]) - p0[0];
    const double by = static_cast<double>(p2[1]) - p0[1];
    const double bz = static_cast<double>(p2[2]) - p0[2];

    double nx = ay * bz - az * by;
    double ny = az * bx - ax * bz;
    double nz = ax * by - ay * bx;
    const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (norm < 1e-12) {
      continue;
    }
    nx /= norm;
    ny /= norm;
    nz /= norm;
    if (nz < 0.0) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }
    if (nz < cfg_.min_horizontal_cos) {
      continue;
    }
    const double d = -(nx * static_cast<double>(p0[0]) + ny * static_cast<double>(p0[1]) +
      nz * static_cast<double>(p0[2]));

    int score = 0;
    for (int i = 0; i < n; ++i) {
      const float * p = points + 3 * i;
      const double dist = nx * static_cast<double>(p[0]) + ny * static_cast<double>(p[1]) +
        nz * static_cast<double>(p[2]) + d;
      score += (std::abs(dist) < threshold) ? 1 : 0;
    }

    // Strictly greater, so the earliest iteration wins a tie.
    if (score > best_score) {
      best_score = score;
      best_n[0] = nx;
      best_n[1] = ny;
      best_n[2] = nz;
      best_d = d;
    }
  }

  if (best_score < cfg_.min_inliers) {
    std::memcpy(out, points, static_cast<std::size_t>(n) * 3u * sizeof(float));
    out_count = n;
    return;
  }

  int inlier_count = 0;
  for (int i = 0; i < n; ++i) {
    const float * p = points + 3 * i;
    const double dist = best_n[0] * static_cast<double>(p[0]) +
      best_n[1] * static_cast<double>(p[1]) + best_n[2] * static_cast<double>(p[2]) + best_d;
    if (std::abs(dist) < threshold) {
      inliers_[static_cast<std::size_t>(inlier_count++)] = i;
    }
  }

  double cx = 0.0;
  double cy = 0.0;
  double cz = 0.0;
  for (int i = 0; i < inlier_count; ++i) {
    const float * p = points + 3 * inliers_[static_cast<std::size_t>(i)];
    cx += static_cast<double>(p[0]);
    cy += static_cast<double>(p[1]);
    cz += static_cast<double>(p[2]);
  }
  const double inv_inliers = 1.0 / static_cast<double>(inlier_count);
  cx *= inv_inliers;
  cy *= inv_inliers;
  cz *= inv_inliers;

  double sxx = 0.0, sxy = 0.0, sxz = 0.0, syy = 0.0, syz = 0.0, szz = 0.0;
  for (int i = 0; i < inlier_count; ++i) {
    const float * p = points + 3 * inliers_[static_cast<std::size_t>(i)];
    const double dx = static_cast<double>(p[0]) - cx;
    const double dy = static_cast<double>(p[1]) - cy;
    const double dz = static_cast<double>(p[2]) - cz;
    sxx += dx * dx;
    sxy += dx * dy;
    sxz += dx * dz;
    syy += dy * dy;
    syz += dy * dz;
    szz += dz * dz;
  }
  Eigen::Matrix3d cov;
  cov << sxx, sxy, sxz,
    sxy, syy, syz,
    sxz, syz, szz;
  cov *= inv_inliers;

  // Eigenvalues come back in increasing order, so column zero is the direction
  // of least variance: the plane normal.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
  Eigen::Vector3d normal = solver.eigenvectors().col(0);
  canonicalise(normal);
  if (normal.z() < 0.0) {
    normal = -normal;
  }
  const double d = -(normal.x() * cx + normal.y() * cy + normal.z() * cz);

  plane = {normal.x(), normal.y(), normal.z(), d};
  found = true;

  // One comparison removes the table and everything under it, which is what
  // leaves only objects standing on it.
  const double cut = threshold + cfg_.clearance_m;
  int kept = 0;
  for (int i = 0; i < n; ++i) {
    const float * p = points + 3 * i;
    const double dist = normal.x() * static_cast<double>(p[0]) +
      normal.y() * static_cast<double>(p[1]) + normal.z() * static_cast<double>(p[2]) + d;
    if (dist >= cut) {
      out[3 * kept + 0] = p[0];
      out[3 * kept + 1] = p[1];
      out[3 * kept + 2] = p[2];
      ++kept;
    }
  }
  out_count = kept;
}

}  // namespace grasp_core
