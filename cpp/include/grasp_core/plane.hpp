#pragma once

// docs/ALGORITHM.md S3: RANSAC table removal against a precomputed deviate
// table, least-squares refit over the winning inlier set, then removal of the
// table and everything below it.

#include <array>
#include <cstddef>
#include <vector>

#include "grasp_core/config.hpp"

namespace grasp_core
{

class PlaneFitter
{
public:
  PlaneFitter(const Config & config, const std::vector<double> & deviates);

  // Sizes the inlier index buffer. Call before the measured path; calling it
  // again with a smaller bound is a no-op.
  void reserve(std::size_t max_points);

  // `points` is 3*n interleaved float32 in the base frame. Writes the surviving
  // points to `out` (which must hold 3*n floats) and their count to
  // `out_count`. `plane` is (nx, ny, nz, d) with nz > 0 when `found`, and all
  // zeros when it is not.
  void run(
    const float * points, int n, std::array<double, 4> & plane, bool & found,
    float * out, int & out_count);

private:
  Config::Plane cfg_;
  const std::vector<double> & deviates_;
  std::vector<int> inliers_;
};

}  // namespace grasp_core
