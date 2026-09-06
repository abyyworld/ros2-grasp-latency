#pragma once

#include <cstddef>

namespace grasp_core
{

// The arm is a fixed 7-DOF Panda, so every joint-space quantity is a
// std::array rather than a vector: no heap, and a length mismatch between the
// chain file and the code is a compile-time shape rather than a runtime check
// inside the measured path. Chain::load() verifies the file agrees.
constexpr int kDof = 7;

// S0..S7 of docs/ALGORITHM.md.
constexpr int kStages = 8;

}  // namespace grasp_core
