#pragma once

#include <cstddef>
#include <cstdint>

#include "qik/gemm.hpp"
#include "qik/gemm_neon.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace qik {

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
inline constexpr bool kHasNeonMatmulInt8 = true;
#else
inline constexpr bool kHasNeonMatmulInt8 = false;
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)

// int8 GEMM built on SMMLA.
//
// SDOT retires 16 multiply-accumulates per instruction: four independent
// length-4 dot products. SMMLA retires 32 -- a 2x8 by 8x2 matrix product, so
// four length-8 dot products, in one go. Same register width, twice the work,
// because the instruction reuses each loaded operand against two of the other
// operand's rows instead of one.
//
// The layout requirement is what makes this awkward. SMMLA wants its two 2x8
// operands packed into single registers: lanes 0-7 are one row, lanes 8-15 the
// next. A and Bt are stored row-major with k contiguous, so two rows that are
// adjacent in the output block are k_dim bytes apart in memory and have to be
// stitched together before the instruction can see them. That is the vcombine
// below, and it is the price of the 2x.
//
// Accumulator layout after vmmlaq_s32(acc, A01, B01):
//
//     lane 0 = A_row0 . B_col0      lane 1 = A_row0 . B_col1
//     lane 2 = A_row1 . B_col0      lane 3 = A_row1 . B_col1
//
// which is a row-major 2x2 tile of C. Four of those tile a 4x4 block, so this
// kernel covers the same block shape as gemm_int8_neon_blocked and the two are
// directly comparable: 8 SMMLA per 16 elements of k against 16 SDOT, from the
// same 8 loads.
//
// Register pressure drops as a side effect. The SDOT kernel needs 16
// accumulators plus 8 operands, 24 of 32 SIMD registers. This needs 4
// accumulators plus operands, which leaves room to widen the block further --
// but widening changes two variables at once, so the comparison here holds the
// block shape fixed and varies only the instruction.
inline void gemm_int8_neon_smmla(const std::int8_t* a, const std::int8_t* bt,
                                 std::int32_t* c, int m_dim, int n_dim,
                                 int k_dim) noexcept {
  constexpr int kMR = 4;
  constexpr int kNR = 4;

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

      // acc[i][j] holds the 2x2 tile of C rooted at (m0 + 2i, n0 + 2j).
      int32x4_t acc00 = vdupq_n_s32(0);
      int32x4_t acc01 = vdupq_n_s32(0);
      int32x4_t acc10 = vdupq_n_s32(0);
      int32x4_t acc11 = vdupq_n_s32(0);

      int k = 0;

      // Main path: 16 bytes loaded per row, consumed as two 8-element steps.
      // Eight loads feed eight SMMLA, against the SDOT kernel's eight loads
      // feeding sixteen SDOT for the identical block of output.
      for (; k + 16 <= k_dim; k += 16) {
        const int8x16_t a0 = vld1q_s8(ap[0] + k);
        const int8x16_t a1 = vld1q_s8(ap[1] + k);
        const int8x16_t a2 = vld1q_s8(ap[2] + k);
        const int8x16_t a3 = vld1q_s8(ap[3] + k);
        const int8x16_t b0 = vld1q_s8(bp[0] + k);
        const int8x16_t b1 = vld1q_s8(bp[1] + k);
        const int8x16_t b2 = vld1q_s8(bp[2] + k);
        const int8x16_t b3 = vld1q_s8(bp[3] + k);

        const int8x16_t a01_lo = vcombine_s8(vget_low_s8(a0), vget_low_s8(a1));
        const int8x16_t a23_lo = vcombine_s8(vget_low_s8(a2), vget_low_s8(a3));
        const int8x16_t b01_lo = vcombine_s8(vget_low_s8(b0), vget_low_s8(b1));
        const int8x16_t b23_lo = vcombine_s8(vget_low_s8(b2), vget_low_s8(b3));

        acc00 = vmmlaq_s32(acc00, a01_lo, b01_lo);
        acc01 = vmmlaq_s32(acc01, a01_lo, b23_lo);
        acc10 = vmmlaq_s32(acc10, a23_lo, b01_lo);
        acc11 = vmmlaq_s32(acc11, a23_lo, b23_lo);

        const int8x16_t a01_hi =
            vcombine_s8(vget_high_s8(a0), vget_high_s8(a1));
        const int8x16_t a23_hi =
            vcombine_s8(vget_high_s8(a2), vget_high_s8(a3));
        const int8x16_t b01_hi =
            vcombine_s8(vget_high_s8(b0), vget_high_s8(b1));
        const int8x16_t b23_hi =
            vcombine_s8(vget_high_s8(b2), vget_high_s8(b3));

        acc00 = vmmlaq_s32(acc00, a01_hi, b01_hi);
        acc01 = vmmlaq_s32(acc01, a01_hi, b23_hi);
        acc10 = vmmlaq_s32(acc10, a23_hi, b01_hi);
        acc11 = vmmlaq_s32(acc11, a23_hi, b23_hi);
      }

      // One more 8-element step if k_dim % 16 >= 8, so the scalar tail below
      // never runs more than seven times.
      if (k + 8 <= k_dim) {
        const int8x16_t a01 =
            vcombine_s8(vld1_s8(ap[0] + k), vld1_s8(ap[1] + k));
        const int8x16_t a23 =
            vcombine_s8(vld1_s8(ap[2] + k), vld1_s8(ap[3] + k));
        const int8x16_t b01 =
            vcombine_s8(vld1_s8(bp[0] + k), vld1_s8(bp[1] + k));
        const int8x16_t b23 =
            vcombine_s8(vld1_s8(bp[2] + k), vld1_s8(bp[3] + k));

        acc00 = vmmlaq_s32(acc00, a01, b01);
        acc01 = vmmlaq_s32(acc01, a01, b23);
        acc10 = vmmlaq_s32(acc10, a23, b01);
        acc11 = vmmlaq_s32(acc11, a23, b23);
        k += 8;
      }

      // Scatter the four 2x2 tiles back into C, adding the scalar tail. int32
      // accumulation is exact, so this has to match the scalar kernel bit for
      // bit -- there is no tolerance to hide an indexing mistake in.
      const int32x4_t tile[2][2] = {{acc00, acc01}, {acc10, acc11}};
      for (int ti = 0; ti < 2; ++ti) {
        for (int tj = 0; tj < 2; ++tj) {
          std::int32_t lanes[4];
          vst1q_s32(lanes, tile[ti][tj]);
          for (int si = 0; si < 2; ++si) {
            for (int sj = 0; sj < 2; ++sj) {
              const int i = 2 * ti + si;
              const int j = 2 * tj + sj;
              std::int32_t sum = lanes[2 * si + sj];
              for (int kk = k; kk < k_dim; ++kk) {
                sum += static_cast<std::int32_t>(ap[i][kk]) *
                       static_cast<std::int32_t>(bp[j][kk]);
              }
              c[static_cast<std::size_t>(m0 + i) * n_dim + n0 + j] = sum;
            }
          }
        }
      }
    }

    // Columns past the last full 4-wide block. SMMLA needs pairs of columns,
    // so a 1- to 3-column remainder has nothing to pair with and falls back
    // to SDOT rather than padding.
    for (int n = n_main; n < n_dim; ++n) {
      const std::int8_t* b_row = bt + static_cast<std::size_t>(n) * k_dim;
      for (int i = 0; i < kMR; ++i) {
        const std::int8_t* a_row = a + static_cast<std::size_t>(m0 + i) * k_dim;
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
        c[static_cast<std::size_t>(m0 + i) * n_dim + n] = sum;
      }
    }
  }

  // Rows past the last full 4-row block. Same reasoning as the column
  // remainder: no pair, no SMMLA.
  if (m_main < m_dim) {
    gemm_int8_neon(a + static_cast<std::size_t>(m_main) * k_dim, bt,
                   c + static_cast<std::size_t>(m_main) * n_dim,
                   m_dim - m_main, n_dim, k_dim);
  }
}

#else

// Same shim discipline as gemm_neon.hpp: the symbol exists so the harness
// builds anywhere, kHasNeonMatmulInt8 reports false, and the benchmark is
// expected to check that flag rather than print SDOT numbers under an SMMLA
// heading.
inline void gemm_int8_neon_smmla(const std::int8_t* a, const std::int8_t* bt,
                                 std::int32_t* c, int m_dim, int n_dim,
                                 int k_dim) noexcept {
  gemm_int8_neon_blocked(a, bt, c, m_dim, n_dim, k_dim);
}

#endif

}  // namespace qik
