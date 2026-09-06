#pragma once

// docs/ALGORITHM.md S4: voxelise, take 6-connected components over the
// occupied voxels, and return the component holding the most points.
//
// Implemented as an open-addressing hash of voxel keys plus a BFS, which is
// the structure that stays proportional to the number of occupied voxels
// rather than to the bounding box. A dense grid would be simpler and is what
// the Python side uses through scipy.ndimage; the spec fixes the answer, not
// the data structure.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "grasp_core/config.hpp"

namespace grasp_core
{

class Clusterer
{
public:
  explicit Clusterer(const Config & config);

  void reserve(std::size_t max_points);

  // `points` is 3*n interleaved float32. Writes the winning component's point
  // indices, ascending, into `out_indices` (capacity n) and returns how many.
  // Returns zero when the best component holds fewer than cluster.min_points.
  int run(const float * points, int n, int * out_indices);

private:
  struct Key
  {
    std::int32_t x{0};
    std::int32_t y{0};
    std::int32_t z{0};
  };

  int find_voxel(const Key & key) const;
  int intern_voxel(const Key & key);

  Config::Cluster cfg_;

  std::vector<std::int32_t> bucket_;      // hash slot -> voxel id, or -1
  std::size_t mask_{0};                   // bucket_ window in use, minus one

  std::vector<Key> key_;                  // voxel id -> key
  std::vector<std::int32_t> count_;       // voxel id -> point count
  std::vector<std::int32_t> component_;   // voxel id -> component id, or -1
  std::vector<std::int32_t> voxel_of_;    // point index -> voxel id
  std::vector<std::int32_t> queue_;       // BFS frontier
  int voxels_{0};
};

}  // namespace grasp_core
