#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace qik {

// Symmetric int4 quantization with two values packed per byte.
//
// int8 gives 255 representable levels; int4 gives 15. That is not a small step
// down, it is a 17x coarser grid, and it is where per-tensor scaling stops
// being merely suboptimal and starts being wrong on ordinary data rather than
// only on adversarial data.
//
// The range is [-7, 7], not [-8, 7]. Two's complement gives 4 bits an asymmetric
// range, and using the extra negative slot would mean negating a value could
// leave the representable set -- which breaks the symmetry the whole scheme
// depends on. Costing one level to keep the grid symmetric is the standard
// trade and it is why the divisor below is 7.
//
// Packing is little-nibble-first: element 2i goes in the low nibble of byte i,
// element 2i+1 in the high nibble. An odd K leaves the final high nibble zero,
// which is harmless because the kernel is bounded by K rather than by bytes.

inline constexpr int kInt4Max = 7;

inline float compute_scale_int4(const float* data, std::size_t count) noexcept {
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < count; ++i) {
    max_abs = std::max(max_abs, std::fabs(data[i]));
  }
  return max_abs > 0.0f ? max_abs / static_cast<float>(kInt4Max) : 1.0f;
}

inline std::int8_t quantize_one_int4(float value, float scale) noexcept {
  const float scaled = std::nearbyint(value / scale);
  return static_cast<std::int8_t>(
      std::clamp(scaled, static_cast<float>(-kInt4Max),
                 static_cast<float>(kInt4Max)));
}

// Bytes needed to hold `count` packed nibbles.
inline constexpr std::size_t packed_bytes(std::size_t count) noexcept {
  return (count + 1) / 2;
}

// Sign-extend a 4-bit two's complement nibble to int32. Shifting left then
// arithmetic-right is the branch-free way to do it; masking alone would read
// -1 as 15.
inline std::int32_t unpack_lo(std::uint8_t byte) noexcept {
  return static_cast<std::int32_t>(static_cast<std::int8_t>(byte << 4) >> 4);
}
inline std::int32_t unpack_hi(std::uint8_t byte) noexcept {
  return static_cast<std::int32_t>(static_cast<std::int8_t>(byte) >> 4);
}

inline std::uint8_t pack_pair(std::int8_t lo, std::int8_t hi) noexcept {
  return static_cast<std::uint8_t>((static_cast<std::uint8_t>(hi) << 4) |
                                   (static_cast<std::uint8_t>(lo) & 0x0Fu));
}

// Quantize a contiguous run into packed nibbles against one shared scale.
inline float quantize_per_tensor_int4(const float* src, std::uint8_t* dst,
                                      std::size_t count) noexcept {
  const float scale = compute_scale_int4(src, count);
  for (std::size_t i = 0; i < count; i += 2) {
    const std::int8_t lo = quantize_one_int4(src[i], scale);
    const std::int8_t hi =
        (i + 1 < count) ? quantize_one_int4(src[i + 1], scale) : 0;
    dst[i / 2] = pack_pair(lo, hi);
  }
  return scale;
}

// K x N weights -> N x K packed nibbles, one scale per output channel. Same
// transpose and same per-channel policy as the int8 path, so the two differ
// only in bit width and the comparison isolates that.
inline void quantize_weights_per_channel_int4(const float* b, std::uint8_t* bt,
                                              float* scales, int k_dim,
                                              int n_dim) noexcept {
  const std::size_t stride = packed_bytes(static_cast<std::size_t>(k_dim));
  for (int n = 0; n < n_dim; ++n) {
    float max_abs = 0.0f;
    for (int k = 0; k < k_dim; ++k) {
      max_abs = std::max(max_abs,
                         std::fabs(b[static_cast<std::size_t>(k) * n_dim + n]));
    }
    const float scale =
        max_abs > 0.0f ? max_abs / static_cast<float>(kInt4Max) : 1.0f;
    scales[n] = scale;

    std::uint8_t* row = bt + static_cast<std::size_t>(n) * stride;
    for (int k = 0; k < k_dim; k += 2) {
      const std::int8_t lo =
          quantize_one_int4(b[static_cast<std::size_t>(k) * n_dim + n], scale);
      const std::int8_t hi =
          (k + 1 < k_dim)
              ? quantize_one_int4(
                    b[static_cast<std::size_t>(k + 1) * n_dim + n], scale)
              : 0;
      row[k / 2] = pack_pair(lo, hi);
    }
  }
}

// Same, one shared scale across the whole matrix. Kept so per-tensor and
// per-channel can be compared with only the scaling policy changed.
inline float quantize_weights_per_tensor_int4(const float* b, std::uint8_t* bt,
                                              int k_dim, int n_dim) noexcept {
  const float scale = compute_scale_int4(
      b, static_cast<std::size_t>(k_dim) * static_cast<std::size_t>(n_dim));
  const std::size_t stride = packed_bytes(static_cast<std::size_t>(k_dim));
  for (int n = 0; n < n_dim; ++n) {
    std::uint8_t* row = bt + static_cast<std::size_t>(n) * stride;
    for (int k = 0; k < k_dim; k += 2) {
      const std::int8_t lo =
          quantize_one_int4(b[static_cast<std::size_t>(k) * n_dim + n], scale);
      const std::int8_t hi =
          (k + 1 < k_dim)
              ? quantize_one_int4(
                    b[static_cast<std::size_t>(k + 1) * n_dim + n], scale)
              : 0;
      row[k / 2] = pack_pair(lo, hi);
    }
  }
  return scale;
}

}  // namespace qik
