#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace qik {

// Symmetric linear quantization to int8.
//
// Symmetric rather than affine, deliberately. An affine scheme carries a zero
// point, so the GEMM inner product expands to
//
//   sum_k (a_k - za)(b_k - zb)
//     = sum_k a_k b_k  -  za * sum_k b_k  -  zb * sum_k a_k  +  K * za * zb
//
// which costs two extra reduction passes and a correction term per output.
// Weights are close to zero-centered, so the accuracy that buys is small
// relative to what it costs the hot loop. Per-channel scales recover most of
// the remaining gap for far less work -- which is the whole point of the
// per-channel path below.
//
// Note the divisor is 127, not 128. Clamping the negative end to -127 keeps the
// range symmetric so that negating a quantized value stays representable.

inline float compute_scale(const float* data, std::size_t count) noexcept {
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < count; ++i) {
    max_abs = std::max(max_abs, std::fabs(data[i]));
  }
  return max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
}

inline std::int8_t quantize_one(float value, float scale) noexcept {
  const float scaled = std::nearbyint(value / scale);
  return static_cast<std::int8_t>(std::clamp(scaled, -127.0f, 127.0f));
}

// Quantize a whole tensor against one shared scale.
inline float quantize_per_tensor(const float* src, std::int8_t* dst,
                                 std::size_t count) noexcept {
  const float scale = compute_scale(src, count);
  for (std::size_t i = 0; i < count; ++i) {
    dst[i] = quantize_one(src[i], scale);
  }
  return scale;
}

// Quantize a K x N weight matrix into an N x K transposed int8 buffer, one
// scale per output channel (per column of B, per row of the result).
//
// Two things happen here at once and both matter. The transpose turns the inner
// product into a contiguous walk down both operands, which is what lets the
// NEON kernel issue wide loads. The per-channel scale means one badly scaled
// output column can no longer drag the quantization error of every other column
// with it, which is the usual failure mode of per-tensor weight quantization.
inline void quantize_weights_per_channel(const float* b, std::int8_t* bt,
                                         float* scales, int k_dim,
                                         int n_dim) noexcept {
  for (int n = 0; n < n_dim; ++n) {
    float max_abs = 0.0f;
    for (int k = 0; k < k_dim; ++k) {
      max_abs = std::max(max_abs, std::fabs(b[static_cast<std::size_t>(k) * n_dim + n]));
    }
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    scales[n] = scale;
    for (int k = 0; k < k_dim; ++k) {
      bt[static_cast<std::size_t>(n) * k_dim + k] =
          quantize_one(b[static_cast<std::size_t>(k) * n_dim + n], scale);
    }
  }
}

// Same transpose, one shared scale. Kept so the per-tensor and per-channel
// paths can be compared with only the scaling policy changed.
inline float quantize_weights_per_tensor(const float* b, std::int8_t* bt,
                                         int k_dim, int n_dim) noexcept {
  const float scale =
      compute_scale(b, static_cast<std::size_t>(k_dim) * static_cast<std::size_t>(n_dim));
  for (int n = 0; n < n_dim; ++n) {
    for (int k = 0; k < k_dim; ++k) {
      bt[static_cast<std::size_t>(n) * k_dim + k] =
          quantize_one(b[static_cast<std::size_t>(k) * n_dim + n], scale);
    }
  }
  return scale;
}

// int32 accumulators -> float, folding both operand scales back in.
inline void dequantize_per_channel(const std::int32_t* acc, float* out,
                                   int m_dim, int n_dim, float a_scale,
                                   const float* b_scales) noexcept {
  for (int m = 0; m < m_dim; ++m) {
    for (int n = 0; n < n_dim; ++n) {
      const std::size_t i = static_cast<std::size_t>(m) * n_dim + n;
      out[i] = static_cast<float>(acc[i]) * a_scale * b_scales[n];
    }
  }
}

}  // namespace qik
