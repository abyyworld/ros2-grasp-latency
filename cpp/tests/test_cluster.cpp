// S4 on two blobs far enough apart that no chain of 6-connected voxels joins
// them. The spec asks for the component with the most *points*, not the most
// voxels, so one fixture makes the two differ.

#include <cstdint>
#include <vector>

#include "grasp_core/cluster.hpp"
#include "grasp_core/config.hpp"

#include "check.hpp"

using grasp_core::Clusterer;
using grasp_core::Config;

namespace
{

void add_blob(
  std::vector<float> & points, double cx, double cy, double cz, double half, int count)
{
  for (int i = 0; i < count; ++i) {
    // A deterministic lattice rather than noise: the fixture must not depend
    // on a generator's stream position.
    const double t = static_cast<double>(i) / static_cast<double>(count);
    points.push_back(static_cast<float>(cx + half * (2.0 * t - 1.0)));
    points.push_back(static_cast<float>(cy + half * (2.0 * ((3 * i) % 7) / 6.0 - 1.0)));
    points.push_back(static_cast<float>(cz + half * (2.0 * ((5 * i) % 11) / 10.0 - 1.0)));
  }
}

void test_picks_the_blob_with_more_points(const Config & config)
{
  std::vector<float> points;
  const int small_count = 300;
  const int large_count = 900;
  add_blob(points, 0.30, -0.15, 0.05, 0.015, small_count);
  add_blob(points, 0.55, 0.18, 0.05, 0.015, large_count);
  const int n = small_count + large_count;

  Clusterer clusterer(config);
  clusterer.reserve(static_cast<std::size_t>(n));
  std::vector<int> indices(static_cast<std::size_t>(n));
  const int count = clusterer.run(points.data(), n, indices.data());

  CHECK(count == large_count);
  // The winning component is the second blob, so every index is past the first.
  for (int i = 0; i < count; ++i) {
    CHECK(indices[static_cast<std::size_t>(i)] >= small_count);
  }
  // Indices come back ascending, which is what makes the downstream reductions
  // order-independent between the two implementations.
  for (int i = 1; i < count; ++i) {
    CHECK(indices[static_cast<std::size_t>(i - 1)] < indices[static_cast<std::size_t>(i)]);
  }
}

void test_more_voxels_does_not_beat_more_points(const Config & config)
{
  // A sparse sheet spanning many voxels against a dense cube in few. Point
  // count decides, so the cube wins.
  std::vector<float> points;
  const double voxel = config.cluster.voxel_size_m;
  int sheet_count = 0;
  for (int i = 0; i < 60; ++i) {
    points.push_back(static_cast<float>(0.10 + static_cast<double>(i) * voxel));
    points.push_back(0.30f);
    points.push_back(0.10f);
    ++sheet_count;
  }
  const int cube_count = 400;
  add_blob(points, 0.50, -0.20, 0.08, 0.008, cube_count);

  Clusterer clusterer(config);
  const int n = sheet_count + cube_count;
  clusterer.reserve(static_cast<std::size_t>(n));
  std::vector<int> indices(static_cast<std::size_t>(n));
  const int count = clusterer.run(points.data(), n, indices.data());

  CHECK(count == cube_count);
  CHECK(indices[0] == sheet_count);
}

void test_below_min_points_returns_empty(const Config & config)
{
  std::vector<float> points;
  const int count = config.cluster.min_points - 1;
  add_blob(points, 0.40, 0.0, 0.05, 0.005, count);

  Clusterer clusterer(config);
  clusterer.reserve(static_cast<std::size_t>(count));
  std::vector<int> indices(static_cast<std::size_t>(count));
  CHECK(clusterer.run(points.data(), count, indices.data()) == 0);
}

void test_six_connectivity_does_not_bridge_a_diagonal(const Config & config)
{
  // Two single-voxel groups touching only at a corner. Under 26-connectivity
  // they would merge; under the 6-connectivity the spec fixes they must not,
  // and with both below min_points the answer is empty.
  const double v = config.cluster.voxel_size_m;
  std::vector<float> points;
  const int per_voxel = config.cluster.min_points - 1;
  for (int i = 0; i < per_voxel; ++i) {
    points.push_back(static_cast<float>(0.5 * v));
    points.push_back(static_cast<float>(0.5 * v));
    points.push_back(static_cast<float>(0.5 * v));
  }
  for (int i = 0; i < per_voxel; ++i) {
    points.push_back(static_cast<float>(1.5 * v));
    points.push_back(static_cast<float>(1.5 * v));
    points.push_back(static_cast<float>(1.5 * v));
  }

  Clusterer clusterer(config);
  const int n = 2 * per_voxel;
  clusterer.reserve(static_cast<std::size_t>(n));
  std::vector<int> indices(static_cast<std::size_t>(n));
  CHECK(clusterer.run(points.data(), n, indices.data()) == 0);
}

}  // namespace

int main()
{
  const Config config = Config::load(grasp_test::asset("assets/pipeline_config.json"));
  test_picks_the_blob_with_more_points(config);
  test_more_voxels_does_not_beat_more_points(config);
  test_below_min_points_returns_empty(config);
  test_six_connectivity_does_not_bridge_a_diagonal(config);
  return grasp_test::report("test_cluster");
}
