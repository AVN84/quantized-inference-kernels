#pragma once

#include <cstddef>
#include <cstdint>

namespace qik {

// FP32 reference. Deliberately the naive triple loop: this is the correctness
// oracle and the accuracy baseline, so it is written to be obviously right
// rather than fast. Every speedup claim in this repo is measured against it.
inline void gemm_fp32_reference(const float* a, const float* b, float* c,
                                int m_dim, int n_dim, int k_dim) noexcept {
  for (int m = 0; m < m_dim; ++m) {
    for (int n = 0; n < n_dim; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < k_dim; ++k) {
        sum += a[static_cast<std::size_t>(m) * k_dim + k] *
               b[static_cast<std::size_t>(k) * n_dim + n];
      }
      c[static_cast<std::size_t>(m) * n_dim + n] = sum;
    }
  }
}


// FP32 with the same transposed-B layout the int8 kernel uses.
//
// This exists because the naive reference above is not a fair opponent. Its
// inner loop strides through B by n_dim, so it misses cache on nearly every
// access, and comparing a vectorized int8 kernel against it produces speedups
// north of 100x that say far more about the access pattern than about int8.
// Holding the layout fixed and changing only the datatype isolates the thing
// actually being measured.
inline void gemm_fp32_transposed(const float* a, const float* bt, float* c,
                                 int m_dim, int n_dim, int k_dim) noexcept {
  for (int m = 0; m < m_dim; ++m) {
    const float* a_row = a + static_cast<std::size_t>(m) * k_dim;
    for (int n = 0; n < n_dim; ++n) {
      const float* b_row = bt + static_cast<std::size_t>(n) * k_dim;
      float sum = 0.0f;
      for (int k = 0; k < k_dim; ++k) {
        sum += a_row[k] * b_row[k];
      }
      c[static_cast<std::size_t>(m) * n_dim + n] = sum;
    }
  }
}

// Transpose a K x N matrix into N x K, matching the int8 weight layout.
inline void transpose_fp32(const float* b, float* bt, int k_dim,
                           int n_dim) noexcept {
  for (int n = 0; n < n_dim; ++n) {
    for (int k = 0; k < k_dim; ++k) {
      bt[static_cast<std::size_t>(n) * k_dim + k] =
          b[static_cast<std::size_t>(k) * n_dim + n];
    }
  }
}

// Scalar int8 GEMM against a pre-transposed B (n-major, K contiguous).
//
// This is the semantic definition of the quantized kernel. The NEON version
// must produce bit-identical int32 accumulators, which is the strongest
// correctness property available here -- accumulation in int32 is exact, so
// unlike a float kernel there is no tolerance to hide behind.
inline void gemm_int8_scalar(const std::int8_t* a, const std::int8_t* bt,
                             std::int32_t* c, int m_dim, int n_dim,
                             int k_dim) noexcept {
  for (int m = 0; m < m_dim; ++m) {
    const std::int8_t* a_row = a + static_cast<std::size_t>(m) * k_dim;
    for (int n = 0; n < n_dim; ++n) {
      const std::int8_t* b_row = bt + static_cast<std::size_t>(n) * k_dim;
      std::int32_t sum = 0;
      for (int k = 0; k < k_dim; ++k) {
        sum += static_cast<std::int32_t>(a_row[k]) *
               static_cast<std::int32_t>(b_row[k]);
      }
      c[static_cast<std::size_t>(m) * n_dim + n] = sum;
    }
  }
}

}  // namespace qik
