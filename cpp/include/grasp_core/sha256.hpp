#pragma once

// SHA-256, vendored because docs/FORMATS.md defines the trajectory checksum in
// terms of it and pulling in OpenSSL for one digest over 440 doubles per frame
// would be a dependency the Docker build has to carry for no other reason.
// It runs after the benchmark loop, never inside a measured stage.

#include <cstddef>
#include <cstdint>
#include <string>

namespace grasp_core
{

class Sha256
{
public:
  Sha256();
  void update(const void * data, std::size_t length);
  std::string hex_digest();

private:
  void compress(const std::uint8_t * block);

  std::uint32_t state_[8];
  std::uint8_t buffer_[64];
  std::size_t buffered_{0};
  std::uint64_t total_bits_{0};
};

}  // namespace grasp_core
