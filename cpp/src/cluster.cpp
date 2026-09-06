#include "grasp_core/cluster.hpp"

#include <algorithm>
#include <cmath>

namespace grasp_core
{
namespace
{

std::size_t next_power_of_two(std::size_t n)
{
  std::size_t p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

std::uint64_t hash_key(std::int32_t x, std::int32_t y, std::int32_t z)
{
  std::uint64_t h = static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) *
    0x9E3779B185EBCA87ull;
  h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(y)) * 0xC2B2AE3D27D4EB4Full;
  h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(z)) * 0x165667B19E3779F9ull;
  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 32;
  return h;
}

bool key_less(std::int32_t ax, std::int32_t ay, std::int32_t az,
  std::int32_t bx, std::int32_t by, std::int32_t bz)
{
  if (ax != bx) {
    return ax < bx;
  }
  if (ay != by) {
    return ay < by;
  }
  return az < bz;
}

}  // namespace

Clusterer::Clusterer(const Config & config)
: cfg_(config.cluster) {}

void Clusterer::reserve(std::size_t max_points)
{
  if (voxel_of_.size() >= max_points && !bucket_.empty()) {
    return;
  }
  // One voxel per point is the worst case, and the hash stays under half full
  // so probe chains remain short.
  const std::size_t buckets = next_power_of_two(2u * max_points + 2u);
  bucket_.assign(buckets, -1);
  key_.resize(max_points);
  count_.resize(max_points);
  component_.resize(max_points);
  voxel_of_.resize(max_points);
  queue_.resize(max_points);
}

int Clusterer::find_voxel(const Key & key) const
{
  std::size_t slot = static_cast<std::size_t>(hash_key(key.x, key.y, key.z)) & mask_;
  for (;;) {
    const std::int32_t id = bucket_[slot];
    if (id < 0) {
      return -1;
    }
    const Key & k = key_[static_cast<std::size_t>(id)];
    if (k.x == key.x && k.y == key.y && k.z == key.z) {
      return id;
    }
    slot = (slot + 1u) & mask_;
  }
}

int Clusterer::intern_voxel(const Key & key)
{
  std::size_t slot = static_cast<std::size_t>(hash_key(key.x, key.y, key.z)) & mask_;
  for (;;) {
    const std::int32_t id = bucket_[slot];
    if (id < 0) {
      const int fresh = voxels_++;
      bucket_[slot] = static_cast<std::int32_t>(fresh);
      key_[static_cast<std::size_t>(fresh)] = key;
      count_[static_cast<std::size_t>(fresh)] = 0;
      component_[static_cast<std::size_t>(fresh)] = -1;
      return fresh;
    }
    const Key & k = key_[static_cast<std::size_t>(id)];
    if (k.x == key.x && k.y == key.y && k.z == key.z) {
      return id;
    }
    slot = (slot + 1u) & mask_;
  }
}

int Clusterer::run(const float * points, int n, int * out_indices)
{
  voxels_ = 0;
  if (n <= 0) {
    return 0;
  }

  // Only the part of the bucket array this frame can touch is cleared, so the
  // cost tracks the point count rather than the worst-case resolution.
  const std::size_t buckets =
    std::min(bucket_.size(), next_power_of_two(2u * static_cast<std::size_t>(n) + 2u));
  mask_ = buckets - 1u;
  std::fill(bucket_.begin(), bucket_.begin() + static_cast<std::ptrdiff_t>(buckets), -1);

  const double inv_voxel = 1.0 / cfg_.voxel_size_m;
  for (int i = 0; i < n; ++i) {
    const float * p = points + 3 * i;
    Key key;
    key.x = static_cast<std::int32_t>(std::floor(static_cast<double>(p[0]) * inv_voxel));
    key.y = static_cast<std::int32_t>(std::floor(static_cast<double>(p[1]) * inv_voxel));
    key.z = static_cast<std::int32_t>(std::floor(static_cast<double>(p[2]) * inv_voxel));
    const int id = intern_voxel(key);
    voxel_of_[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(id);
    ++count_[static_cast<std::size_t>(id)];
  }

  int best_component = -1;
  int best_points = 0;
  Key best_key{};
  int components = 0;

  for (int seed = 0; seed < voxels_; ++seed) {
    if (component_[static_cast<std::size_t>(seed)] >= 0) {
      continue;
    }
    const int id = components++;
    int head = 0;
    int tail = 0;
    queue_[static_cast<std::size_t>(tail++)] = static_cast<std::int32_t>(seed);
    component_[static_cast<std::size_t>(seed)] = static_cast<std::int32_t>(id);

    int total = 0;
    Key smallest = key_[static_cast<std::size_t>(seed)];

    while (head < tail) {
      const int v = queue_[static_cast<std::size_t>(head++)];
      const Key k = key_[static_cast<std::size_t>(v)];
      total += count_[static_cast<std::size_t>(v)];
      if (key_less(k.x, k.y, k.z, smallest.x, smallest.y, smallest.z)) {
        smallest = k;
      }
      for (int axis = 0; axis < 3; ++axis) {
        for (int step = -1; step <= 1; step += 2) {
          Key nb = k;
          if (axis == 0) {
            nb.x += step;
          } else if (axis == 1) {
            nb.y += step;
          } else {
            nb.z += step;
          }
          const int found = find_voxel(nb);
          if (found >= 0 && component_[static_cast<std::size_t>(found)] < 0) {
            component_[static_cast<std::size_t>(found)] = static_cast<std::int32_t>(id);
            queue_[static_cast<std::size_t>(tail++)] = static_cast<std::int32_t>(found);
          }
        }
      }
    }

    // Most points wins; the lexicographically smallest key in the component
    // breaks a tie, so the answer never depends on traversal order.
    const bool better = total > best_points ||
      (total == best_points && best_component >= 0 &&
      key_less(smallest.x, smallest.y, smallest.z, best_key.x, best_key.y, best_key.z));
    if (best_component < 0 || better) {
      best_component = id;
      best_points = total;
      best_key = smallest;
    }
  }

  if (best_component < 0 || best_points < cfg_.min_points) {
    return 0;
  }

  int written = 0;
  for (int i = 0; i < n; ++i) {
    if (component_[static_cast<std::size_t>(voxel_of_[static_cast<std::size_t>(i)])] ==
      best_component)
    {
      out_indices[written++] = i;
    }
  }
  return written;
}

}  // namespace grasp_core
