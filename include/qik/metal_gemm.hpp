#pragma once

#include <cstdint>
#include <string>

namespace qik {

// GPU backend for the int8 GEMM, via Metal.
//
// The CPU kernels in this repo are bound by how many multiply-accumulates a
// single core can issue. This asks a different question: what does the same
// arithmetic cost on 18 GPU cores, and where does the crossover sit? Small
// matrices should lose on the GPU -- dispatch overhead dominates -- and large
// ones should win. Finding that crossover is the point.
//
// Apple Silicon makes one part of this unusually clean. CPU and GPU share
// physical memory, so buffers created with MTLResourceStorageModeShared are
// visible to both with no copy and no staging. A discrete GPU would need an
// explicit host-to-device transfer that often dominates a GEMM this size, and
// pretending otherwise is how GPU benchmarks end up dishonest. Here the zero
// copy is real, and the numbers below are dispatch plus compute only.

// Returns false and fills `error` when Metal is unavailable.
bool metal_available(std::string* error = nullptr);

// Name of the GPU that will run the kernel.
std::string metal_device_name();

// C = A * Bt^T, int8 in, int32 accumulate. Bt is n-major, matching the CPU
// kernels exactly, so both backends consume identical buffers.
// Returns false on failure and writes the reason into `error`.
bool gemm_int8_metal(const std::int8_t* a, const std::int8_t* bt,
                     std::int32_t* c, int m_dim, int n_dim, int k_dim,
                     std::string* error = nullptr);

}  // namespace qik
