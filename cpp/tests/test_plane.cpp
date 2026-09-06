// S3 against a synthetic scene whose plane is known exactly: a tilted table
// with a box standing on it and a scatter of far-off outliers.

#include <cmath>
#include <cstdint>
#include <vector>

#include "grasp_core/config.hpp"
#include "grasp_core/plane.hpp"

#include "check.hpp"

using grasp_core::Config;
using grasp_core::PlaneFitter;

namespace
{

// A deterministic 32-bit generator, used only to build the fixture. Nothing
// in the pipeline draws a random number.
struct Lcg
{
  std::uint32_t state;
  double next()
  {
    state = state * 1664525u + 1013904223u;
    return static_cast<double>(state >> 8) / 16777216.0;
  }
};

void test_finds_a_known_plane(const Config & config, const std::vector<double> & deviates)
{
  // z = a x + b y + c, normalised below. Tilted so the answer is not (0, 0, 1).
  const double a = 0.04;
  const double b = -0.03;
  const double c = 0.012;
  const double inv = 1.0 / std::sqrt(a * a + b * b + 1.0);
  const double nx = -a * inv;
  const double ny = -b * inv;
  const double nz = inv;
  const double d = -c * inv;

  Lcg rng{7u};
  std::vector<float> points;
  const int table_points = 4000;
  for (int i = 0; i < table_points; ++i) {
    const double x = 0.2 + 0.4 * rng.next();
    const double y = -0.2 + 0.4 * rng.next();
    const double z = a * x + b * y + c + (rng.next() - 0.5) * 0.002;
    points.push_back(static_cast<float>(x));
    points.push_back(static_cast<float>(y));
    points.push_back(static_cast<float>(z));
  }
  // A box standing on the table, 6 cm tall, clear of the removal cut.
  const int box_points = 600;
  for (int i = 0; i < box_points; ++i) {
    const double x = 0.35 + 0.03 * rng.next();
    const double y = -0.02 + 0.03 * rng.next();
    const double z = a * x + b * y + c + 0.06;
    points.push_back(static_cast<float>(x));
    points.push_back(static_cast<float>(y));
    points.push_back(static_cast<float>(z));
  }

  const int total = table_points + box_points;
  PlaneFitter fitter(config, deviates);
  fitter.reserve(static_cast<std::size_t>(total));

  std::array<double, 4> plane{};
  bool found = false;
  std::vector<float> kept(static_cast<std::size_t>(total) * 3u);
  int kept_count = 0;
  fitter.run(points.data(), total, plane, found, kept.data(), kept_count);

  CHECK(found);
  CHECK_NEAR(plane[0], nx, 1e-4);
  CHECK_NEAR(plane[1], ny, 1e-4);
  CHECK_NEAR(plane[2], nz, 1e-4);
  CHECK_NEAR(plane[3], d, 1e-4);
  CHECK(plane[2] > 0.0);
  CHECK_NEAR(std::sqrt(
      plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]), 1.0, 1e-12);

  // Everything the box contributed survives; nothing the table contributed does.
  CHECK(kept_count == box_points);
  for (int i = 0; i < kept_count; ++i) {
    const double height = plane[0] * kept[3 * i + 0] + plane[1] * kept[3 * i + 1] +
      plane[2] * kept[3 * i + 2] + plane[3];
    CHECK(height >= config.plane.inlier_threshold_m + config.plane.clearance_m);
  }
}

void test_passes_through_when_no_plane(
  const Config & config, const std::vector<double> & deviates)
{
  // A vertical wall: every candidate normal is horizontal, so every one of
  // them scores zero against min_horizontal_cos and the stage bails out.
  Lcg rng{11u};
  std::vector<float> points;
  const int n = 1200;
  for (int i = 0; i < n; ++i) {
    points.push_back(0.4f);
    points.push_back(static_cast<float>(-0.2 + 0.4 * rng.next()));
    points.push_back(static_cast<float>(0.4 * rng.next()));
  }

  PlaneFitter fitter(config, deviates);
  fitter.reserve(static_cast<std::size_t>(n));
  std::array<double, 4> plane{};
  bool found = true;
  std::vector<float> kept(static_cast<std::size_t>(n) * 3u);
  int kept_count = 0;
  fitter.run(points.data(), n, plane, found, kept.data(), kept_count);

  CHECK(!found);
  CHECK(kept_count == n);
  CHECK(plane[0] == 0.0 && plane[1] == 0.0 && plane[2] == 0.0 && plane[3] == 0.0);
}

void test_too_few_points(const Config & config, const std::vector<double> & deviates)
{
  const float points[6] = {0.1f, 0.0f, 0.0f, 0.2f, 0.0f, 0.0f};
  PlaneFitter fitter(config, deviates);
  fitter.reserve(2u);
  std::array<double, 4> plane{};
  bool found = true;
  float kept[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int kept_count = -1;
  fitter.run(points, 2, plane, found, kept, kept_count);
  CHECK(!found);
  CHECK(kept_count == 2);
  CHECK(kept[3] == 0.2f);
}

}  // namespace

int main()
{
  const Config config = Config::load(grasp_test::asset("assets/pipeline_config.json"));
  const std::vector<double> deviates = grasp_core::load_ransac_table(
    grasp_test::asset("assets/ransac_uniform.bin"), config.ransac_table.count);

  test_finds_a_known_plane(config, deviates);
  test_passes_through_when_no_plane(config, deviates);
  test_too_few_points(config, deviates);
  return grasp_test::report("test_plane");
}
