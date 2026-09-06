#pragma once

// The whole ROS-independent core, behind one class. This header is the only
// one a consumer needs and it deliberately pulls in nothing but the standard
// library, so linking grasp_core does not drag Eigen into a ROS package's
// include path.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "grasp_core/dof.hpp"
#include "grasp_core/trajectory.hpp"

namespace grasp_core
{

struct Config;
struct Chain;

// Per-stage wall time, in the key order docs/FORMATS.md fixes.
struct StageNs
{
  std::uint64_t decode{0};
  std::uint64_t deproject{0};
  std::uint64_t transform_crop{0};
  std::uint64_t plane{0};
  std::uint64_t cluster{0};
  std::uint64_t grasp{0};
  std::uint64_t ik{0};
  std::uint64_t traj{0};
};

struct Result
{
  StageNs stage_ns;
  std::uint64_t total_ns{0};

  // Points entering S3, that is after deprojection and the workspace crop.
  int points{0};
  int cluster_points{0};

  bool plane_found{false};
  std::array<double, 4> plane{};        // (nx, ny, nz, d), all zero when not found

  bool graspable{false};
  double width{0.0};
  std::array<double, 16> tcp{};         // T_base_tcp, row-major

  bool converged{false};
  int iterations{0};
  std::array<double, kDof> q{};

  double duration_s{0.0};
  std::vector<Waypoint> trajectory;
};

class Pipeline
{
public:
  Pipeline(
    const std::string & config_path, const std::string & chain_path,
    const std::string & ransac_table_path);
  ~Pipeline();

  Pipeline(Pipeline &&) noexcept;
  Pipeline & operator=(Pipeline &&) noexcept;
  Pipeline(const Pipeline &) = delete;
  Pipeline & operator=(const Pipeline &) = delete;

  // Sizes every scratch buffer for a given resolution. run() calls this
  // itself, outside the stage clocks, so the first frame at a new resolution
  // pays for the allocation and no later frame does. Call it explicitly before
  // a measured run to keep even that first frame clean.
  void reserve(int width, int height);

  // The returned reference is owned by the Pipeline and is overwritten by the
  // next call. Returning by reference rather than by value is what keeps the
  // trajectory vector from being reallocated once per frame.
  const Result & run(const std::uint16_t * depth, const std::uint8_t * rgb, int width, int height);

  const Config & config() const;
  const Chain & chain() const;

  // Median cost of one clock read, measured at construction. Nine reads
  // bracket the eight stages, so this is the instrumentation floor of a frame.
  double timer_overhead_ns() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace grasp_core
