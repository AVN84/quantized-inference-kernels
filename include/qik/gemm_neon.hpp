#pragma once

#include <cstddef>
#include <cstdint>

#include "qik/gemm.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace qik {

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
inline constexpr bool kHasNeonDotProduct = true;
#else
inline constexpr bool kHasNeonDotProduct = false;
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)


// Hand-vectorized fp32 baseline, same transposed layout, same 4-wide n blocking.
//
// This is the only honest opponent for the int8 kernel. The autovectorizer will
// not touch the scalar fp32 reduction because floating-point addition is not
// associative, so reassociating the sum changes the result and the compiler is
// forbidden from doing it without -ffast-math. int32 accumulation *is*
// associative, so the scalar int8 loop vectorizes for free. Comparing those two
// measures the optimizer's permissions, not the datatype.
//
// Vectorizing the fp32 side by hand makes the same reassociation explicitly, so
// both kernels now use four independent accumulators reduced at the end. What
// is left is the real ratio: FMLA retires 4 fp32 multiply-accumulates per
// instruction, SDOT retires 16 int8 ones, so the ceiling is 4x.
inline void gemm_fp32_neon(const float* a, const float* bt, float* c,
                           int m_dim, int n_dim, int k_dim) noexcept {
  for (int m = 0; m < m_dim; ++m) {
    const float* a_row = a + static_cast<std::size_t>(m) * k_dim;
    float* c_row = c + static_cast<std::size_t>(m) * n_dim;

    int n = 0;
    for (; n + 4 <= n_dim; n += 4) {
      const float* b0 = bt + static_cast<std::size_t>(n + 0) * k_dim;
      const float* b1 = bt + static_cast<std::size_t>(n + 1) * k_dim;
      const float* b2 = bt + static_cast<std::size_t>(n + 2) * k_dim;
      const float* b3 = bt + static_cast<std::size_t>(n + 3) * k_dim;

      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);

      int k = 0;
      for (; k + 4 <= k_dim; k += 4) {
        const float32x4_t av = vld1q_f32(a_row + k);
        acc0 = vfmaq_f32(acc0, av, vld1q_f32(b0 + k));
        acc1 = vfmaq_f32(acc1, av, vld1q_f32(b1 + k));
        acc2 = vfmaq_f32(acc2, av, vld1q_f32(b2 + k));
        acc3 = vfmaq_f32(acc3, av, vld1q_f32(b3 + k));
      }

      float s0 = vaddvq_f32(acc0);
      float s1 = vaddvq_f32(acc1);
      float s2 = vaddvq_f32(acc2);
      float s3 = vaddvq_f32(acc3);

      for (; k < k_dim; ++k) {
        s0 += a_row[k] * b0[k];
        s1 += a_row[k] * b1[k];
        s2 += a_row[k] * b2[k];
        s3 += a_row[k] * b3[k];
      }

      c_row[n + 0] = s0;
      c_row[n + 1] = s1;
      c_row[n + 2] = s2;
      c_row[n + 3] = s3;
    }

    for (; n < n_dim; ++n) {
      const float* b_row = bt + static_cast<std::size_t>(n) * k_dim;
      float32x4_t acc = vdupq_n_f32(0.0f);
      int k = 0;
      for (; k + 4 <= k_dim; k += 4) {
        acc = vfmaq_f32(acc, vld1q_f32(a_row + k), vld1q_f32(b_row + k));
      }
      float sum = vaddvq_f32(acc);
      for (; k < k_dim; ++k) {
        sum += a_row[k] * b_row[k];
      }
      c_row[n] = sum;
    }
  }
}

// int8 GEMM built on SDOT.
//
// vdotq_s32 consumes 16 int8 lanes from each operand and produces 4 int32
// partial sums, so one instruction retires 16 multiply-accumulates. That is the
// entire reason int8 is fast here: the arithmetic is not cheaper than fp32 per
// element, but four times as many elements fit in a register and SDOT retires
// them in one go.
//
// The loop holds four accumulators and walks four rows of B against one row of
// A. Widening past four is where a real kernel goes next, but four already
// amortizes the A load 4x and keeps the working set inside L1, which is most of
// the available win before tiling.
inline void gemm_int8_neon(const std::int8_t* a, const std::int8_t* bt,
                           std::int32_t* c, int m_dim, int n_dim,
                           int k_dim) noexcept {
  for (int m = 0; m < m_dim; ++m) {
    const std::int8_t* a_row = a + static_cast<std::size_t>(m) * k_dim;
    std::int32_t* c_row = c + static_cast<std::size_t>(m) * n_dim;

    int n = 0;
    for (; n + 4 <= n_dim; n += 4) {
      const std::int8_t* b0 = bt + static_cast<std::size_t>(n + 0) * k_dim;
      const std::int8_t* b1 = bt + static_cast<std::size_t>(n + 1) * k_dim;
      const std::int8_t* b2 = bt + static_cast<std::size_t>(n + 2) * k_dim;
      const std::int8_t* b3 = bt + static_cast<std::size_t>(n + 3) * k_dim;

      int32x4_t acc0 = vdupq_n_s32(0);
      int32x4_t acc1 = vdupq_n_s32(0);
      int32x4_t acc2 = vdupq_n_s32(0);
      int32x4_t acc3 = vdupq_n_s32(0);

      int k = 0;
      for (; k + 16 <= k_dim; k += 16) {
        const int8x16_t av = vld1q_s8(a_row + k);
        acc0 = vdotq_s32(acc0, av, vld1q_s8(b0 + k));
        acc1 = vdotq_s32(acc1, av, vld1q_s8(b1 + k));
        acc2 = vdotq_s32(acc2, av, vld1q_s8(b2 + k));
        acc3 = vdotq_s32(acc3, av, vld1q_s8(b3 + k));
      }

      std::int32_t s0 = vaddvq_s32(acc0);
      std::int32_t s1 = vaddvq_s32(acc1);
      std::int32_t s2 = vaddvq_s32(acc2);
      std::int32_t s3 = vaddvq_s32(acc3);

      // Tail. int32 accumulation is exact, so this must match the scalar
      // kernel bit for bit, not approximately.
      for (; k < k_dim; ++k) {
        const auto av = static_cast<std::int32_t>(a_row[k]);
        s0 += av * static_cast<std::int32_t>(b0[k]);
        s1 += av * static_cast<std::int32_t>(b1[k]);
        s2 += av * static_cast<std::int32_t>(b2[k]);
        s3 += av * static_cast<std::int32_t>(b3[k]);
      }

      c_row[n + 0] = s0;
      c_row[n + 1] = s1;
      c_row[n + 2] = s2;
      c_row[n + 3] = s3;
    }

    for (; n < n_dim; ++n) {
      const std::int8_t* b_row = bt + static_cast<std::size_t>(n) * k_dim;
      int32x4_t acc = vdupq_n_s32(0);
      int k = 0;
      for (; k + 16 <= k_dim; k += 16) {
        acc = vdotq_s32(acc, vld1q_s8(a_row + k), vld1q_s8(b_row + k));
      }
      std::int32_t sum = vaddvq_s32(acc);
      for (; k < k_dim; ++k) {
        sum += static_cast<std::int32_t>(a_row[k]) *
               static_cast<std::int32_t>(b_row[k]);
      }
      c_row[n] = sum;
    }
  }
}


// int8 GEMM with 4x4 register blocking.
//
// The kernel above holds one row of A against four rows of B. Per 16-element
// step that is five vector loads feeding four SDOTs -- 0.8 multiply-accumulates
// per load. Every row of A re-streams the whole of B, so at 4096x4096 the 16 MB
// weight matrix crosses the memory bus once per output row and the kernel
// spends its time waiting on DRAM rather than issuing arithmetic.
//
// Blocking four rows of A against four rows of B at once turns that into eight
// loads feeding sixteen SDOTs: 2.0 MACs per load, a 2.5x improvement in
// arithmetic intensity, and B is fetched once per *block* of four output rows
// instead of once per row. Sixteen accumulators plus eight operand registers is
// 24 of the 32 architectural SIMD registers, which leaves the compiler enough
// room to schedule without spilling -- go wider and it spills, and the spill
// costs more than the extra reuse buys.
//
// This does nothing for M=1. A single output row has no reuse to recover, so
// that shape stays bandwidth-bound no matter how the loop is arranged. The
// honest claim here is a win on tall problems, not a universal one.
inline void gemm_int8_neon_blocked(const std::int8_t* a, const std::int8_t* bt,
                                   std::int32_t* c, int m_dim, int n_dim,
                                   int k_dim) noexcept {
  constexpr int kMR = 4;  // rows of A per block
  constexpr int kNR = 4;  // rows of Bt per block

  const int m_main = (m_dim / kMR) * kMR;
  const int n_main = (n_dim / kNR) * kNR;

  for (int m0 = 0; m0 < m_main; m0 += kMR) {
    for (int n0 = 0; n0 < n_main; n0 += kNR) {
      const std::int8_t* ap[kMR];
      const std::int8_t* bp[kNR];
      for (int i = 0; i < kMR; ++i) {
        ap[i] = a + static_cast<std::size_t>(m0 + i) * k_dim;
      }
      for (int j = 0; j < kNR; ++j) {
        bp[j] = bt + static_cast<std::size_t>(n0 + j) * k_dim;
      }

      int32x4_t acc[kMR][kNR];
      for (int i = 0; i < kMR; ++i) {
        for (int j = 0; j < kNR; ++j) {
          acc[i][j] = vdupq_n_s32(0);
        }
      }

      int k = 0;
      for (; k + 16 <= k_dim; k += 16) {
        const int8x16_t a0 = vld1q_s8(ap[0] + k);
        const int8x16_t a1 = vld1q_s8(ap[1] + k);
        const int8x16_t a2 = vld1q_s8(ap[2] + k);
        const int8x16_t a3 = vld1q_s8(ap[3] + k);

        const int8x16_t b0 = vld1q_s8(bp[0] + k);
        const int8x16_t b1 = vld1q_s8(bp[1] + k);
        const int8x16_t b2 = vld1q_s8(bp[2] + k);
        const int8x16_t b3 = vld1q_s8(bp[3] + k);

        acc[0][0] = vdotq_s32(acc[0][0], a0, b0);
        acc[0][1] = vdotq_s32(acc[0][1], a0, b1);
        acc[0][2] = vdotq_s32(acc[0][2], a0, b2);
        acc[0][3] = vdotq_s32(acc[0][3], a0, b3);
        acc[1][0] = vdotq_s32(acc[1][0], a1, b0);
        acc[1][1] = vdotq_s32(acc[1][1], a1, b1);
        acc[1][2] = vdotq_s32(acc[1][2], a1, b2);
        acc[1][3] = vdotq_s32(acc[1][3], a1, b3);
        acc[2][0] = vdotq_s32(acc[2][0], a2, b0);
        acc[2][1] = vdotq_s32(acc[2][1], a2, b1);
        acc[2][2] = vdotq_s32(acc[2][2], a2, b2);
        acc[2][3] = vdotq_s32(acc[2][3], a2, b3);
        acc[3][0] = vdotq_s32(acc[3][0], a3, b0);
        acc[3][1] = vdotq_s32(acc[3][1], a3, b1);
        acc[3][2] = vdotq_s32(acc[3][2], a3, b2);
        acc[3][3] = vdotq_s32(acc[3][3], a3, b3);
      }

      for (int i = 0; i < kMR; ++i) {
        std::int32_t* c_row = c + static_cast<std::size_t>(m0 + i) * n_dim;
        for (int j = 0; j < kNR; ++j) {
          std::int32_t sum = vaddvq_s32(acc[i][j]);
          for (int kk = k; kk < k_dim; ++kk) {
            sum += static_cast<std::int32_t>(ap[i][kk]) *
                   static_cast<std::int32_t>(bp[j][kk]);
          }
          c_row[n0 + j] = sum;
        }
      }
    }

    // Columns past the last full 4-wide block.
    for (int n = n_main; n < n_dim; ++n) {
      const std::int8_t* b_row = bt + static_cast<std::size_t>(n) * k_dim;
      for (int i = 0; i < kMR; ++i) {
        const std::int8_t* a_row = a + static_cast<std::size_t>(m0 + i) * k_dim;
        int32x4_t acc1 = vdupq_n_s32(0);
        int k = 0;
        for (; k + 16 <= k_dim; k += 16) {
          acc1 = vdotq_s32(acc1, vld1q_s8(a_row + k), vld1q_s8(b_row + k));
        }
        std::int32_t sum = vaddvq_s32(acc1);
        for (; k < k_dim; ++k) {
          sum += static_cast<std::int32_t>(a_row[k]) *
                 static_cast<std::int32_t>(b_row[k]);
        }
        c[static_cast<std::size_t>(m0 + i) * n_dim + n] = sum;
      }
    }
  }

  // Rows past the last full 4-row block fall back to the unblocked kernel.
  if (m_main < m_dim) {
    gemm_int8_neon(a + static_cast<std::size_t>(m_main) * k_dim, bt,
                   c + static_cast<std::size_t>(m_main) * n_dim,
                   m_dim - m_main, n_dim, k_dim);
  }
}

#else

// Portability shim so the harness builds and the tests still mean something on
// a machine without SDOT. Reports itself as unavailable via
// kHasNeonDotProduct, so benchmarks do not silently claim a NEON number.
inline void gemm_int8_neon(const std::int8_t* a, const std::int8_t* bt,
                           std::int32_t* c, int m_dim, int n_dim,
                           int k_dim) noexcept {
  gemm_int8_scalar(a, bt, c, m_dim, n_dim, k_dim);
}

inline void gemm_int8_neon_blocked(const std::int8_t* a, const std::int8_t* bt,
                                   std::int32_t* c, int m_dim, int n_dim,
                                   int k_dim) noexcept {
  gemm_int8_scalar(a, bt, c, m_dim, n_dim, k_dim);
}

#endif

}  // namespace qik
