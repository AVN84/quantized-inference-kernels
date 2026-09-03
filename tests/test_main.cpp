#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "qik/gemm.hpp"
#include "qik/gemm_neon.hpp"
#include "qik/quantize.hpp"

namespace {

std::vector<float> random_matrix(int rows, int cols, float spread,
                                 std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, spread);
  std::vector<float> out(static_cast<std::size_t>(rows) * cols);
  for (auto& value : out) {
    value = dist(rng);
  }
  return out;
}

// Mean relative error against the fp32 reference, normalized by the magnitude
// of the reference output rather than per element, so near-zero outputs do not
// blow the metric up.
double relative_error(const std::vector<float>& got,
                      const std::vector<float>& want) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    num += std::abs(static_cast<double>(got[i]) - want[i]);
    den += std::abs(static_cast<double>(want[i]));
  }
  return den > 0.0 ? num / den : 0.0;
}

// The load-bearing test. int32 accumulation is exact, so the vectorized kernel
// has no tolerance to hide behind -- it either matches the scalar definition
// exactly or it is wrong. Shapes deliberately straddle the 16-wide SDOT step
// and the 4-wide n blocking so every tail path is exercised.
void test_neon_matches_scalar_bit_exactly() {
  const int shapes[][3] = {
      {1, 1, 1},    {1, 1, 16},   {1, 1, 17},   {3, 5, 7},
      {4, 4, 16},   {4, 4, 15},   {8, 9, 33},   {16, 16, 64},
      {7, 13, 100}, {32, 32, 128}, {5, 4, 16},  {2, 7, 48},
  };

  for (const auto& shape : shapes) {
    const int m = shape[0];
    const int n = shape[1];
    const int k = shape[2];

    const auto a_f = random_matrix(m, k, 1.0f, 1234u + static_cast<unsigned>(k));
    const auto b_f = random_matrix(k, n, 1.0f, 5678u + static_cast<unsigned>(n));

    std::vector<std::int8_t> a_q(a_f.size());
    std::vector<std::int8_t> bt_q(b_f.size());
    std::vector<float> b_scales(static_cast<std::size_t>(n));

    qik::quantize_per_tensor(a_f.data(), a_q.data(), a_f.size());
    qik::quantize_weights_per_channel(b_f.data(), bt_q.data(), b_scales.data(),
                                      k, n);

    std::vector<std::int32_t> scalar(static_cast<std::size_t>(m) * n, 0);
    std::vector<std::int32_t> neon(static_cast<std::size_t>(m) * n, 0);

    qik::gemm_int8_scalar(a_q.data(), bt_q.data(), scalar.data(), m, n, k);
    qik::gemm_int8_neon(a_q.data(), bt_q.data(), neon.data(), m, n, k);

    for (std::size_t i = 0; i < scalar.size(); ++i) {
      assert(scalar[i] == neon[i]);
    }
  }
}

void test_quantization_round_trip_is_bounded() {
  const auto values = random_matrix(64, 64, 2.0f, 99u);
  std::vector<std::int8_t> quantized(values.size());
  const float scale = qik::quantize_per_tensor(values.data(), quantized.data(),
                                               values.size());

  // Round-to-nearest, so no element may be off by more than half a step.
  for (std::size_t i = 0; i < values.size(); ++i) {
    const float restored = static_cast<float>(quantized[i]) * scale;
    assert(std::abs(restored - values[i]) <= scale * 0.5f + 1e-5f);
  }

  // A zero tensor must not produce a zero scale or a division by it.
  std::vector<float> zeros(16, 0.0f);
  std::vector<std::int8_t> zeros_q(zeros.size());
  const float zero_scale =
      qik::quantize_per_tensor(zeros.data(), zeros_q.data(), zeros.size());
  assert(zero_scale > 0.0f);
  for (const auto value : zeros_q) {
    assert(value == 0);
  }
}

void test_quantized_gemm_tracks_fp32_reference() {
  constexpr int m = 32;
  constexpr int n = 32;
  constexpr int k = 256;

  const auto a_f = random_matrix(m, k, 1.0f, 11u);
  const auto b_f = random_matrix(k, n, 1.0f, 22u);

  std::vector<float> want(static_cast<std::size_t>(m) * n);
  qik::gemm_fp32_reference(a_f.data(), b_f.data(), want.data(), m, n, k);

  std::vector<std::int8_t> a_q(a_f.size());
  std::vector<std::int8_t> bt_q(b_f.size());
  std::vector<float> b_scales(n);
  const float a_scale =
      qik::quantize_per_tensor(a_f.data(), a_q.data(), a_f.size());
  qik::quantize_weights_per_channel(b_f.data(), bt_q.data(), b_scales.data(), k, n);

  std::vector<std::int32_t> acc(static_cast<std::size_t>(m) * n);
  qik::gemm_int8_neon(a_q.data(), bt_q.data(), acc.data(), m, n, k);

  std::vector<float> got(static_cast<std::size_t>(m) * n);
  qik::dequantize_per_channel(acc.data(), got.data(), m, n, a_scale,
                              b_scales.data());

  // Gaussian inputs, K=256, symmetric int8 both sides. Errors are independent
  // enough to partially cancel across the reduction, so a few percent is the
  // expected regime. This is an assertion about the scheme, not a fudge factor:
  // if it regresses, the quantization changed.
  const double error = relative_error(got, want);
  assert(error < 0.05);
}

// Per-channel scaling exists to stop one wide column from setting the step size
// for every other column. Build exactly that pathology and show it works.
void test_per_channel_beats_per_tensor_on_skewed_columns() {
  constexpr int m = 8;
  constexpr int n = 8;
  constexpr int k = 128;

  const auto a_f = random_matrix(m, k, 1.0f, 33u);
  auto b_f = random_matrix(k, n, 1.0f, 44u);

  // One column three orders of magnitude wider than the rest.
  for (int row = 0; row < k; ++row) {
    b_f[static_cast<std::size_t>(row) * n + 0] *= 1000.0f;
  }

  std::vector<float> want(static_cast<std::size_t>(m) * n);
  qik::gemm_fp32_reference(a_f.data(), b_f.data(), want.data(), m, n, k);

  std::vector<std::int8_t> a_q(a_f.size());
  const float a_scale =
      qik::quantize_per_tensor(a_f.data(), a_q.data(), a_f.size());

  std::vector<std::int8_t> bt_channel(b_f.size());
  std::vector<float> channel_scales(n);
  qik::quantize_weights_per_channel(b_f.data(), bt_channel.data(),
                                    channel_scales.data(), k, n);

  std::vector<std::int8_t> bt_tensor(b_f.size());
  const float tensor_scale =
      qik::quantize_weights_per_tensor(b_f.data(), bt_tensor.data(), k, n);
  std::vector<float> tensor_scales(n, tensor_scale);

  std::vector<std::int32_t> acc(static_cast<std::size_t>(m) * n);
  std::vector<float> channel_out(acc.size());
  std::vector<float> tensor_out(acc.size());

  qik::gemm_int8_neon(a_q.data(), bt_channel.data(), acc.data(), m, n, k);
  qik::dequantize_per_channel(acc.data(), channel_out.data(), m, n, a_scale,
                              channel_scales.data());

  qik::gemm_int8_neon(a_q.data(), bt_tensor.data(), acc.data(), m, n, k);
  qik::dequantize_per_channel(acc.data(), tensor_out.data(), m, n, a_scale,
                              tensor_scales.data());

  // Compare only the narrow columns, which are the ones per-tensor destroys.
  double channel_err = 0.0;
  double tensor_err = 0.0;
  double magnitude = 0.0;
  for (int row = 0; row < m; ++row) {
    for (int col = 1; col < n; ++col) {
      const std::size_t i = static_cast<std::size_t>(row) * n + col;
      channel_err += std::abs(channel_out[i] - want[i]);
      tensor_err += std::abs(tensor_out[i] - want[i]);
      magnitude += std::abs(want[i]);
    }
  }
  channel_err /= magnitude;
  tensor_err /= magnitude;

  std::printf("  skewed columns: per-channel %.4f vs per-tensor %.4f\n",
              channel_err, tensor_err);

  assert(channel_err < 0.05);
  assert(tensor_err > channel_err * 5.0);
}


// The fp32 NEON baseline must agree with the scalar fp32 definition. Not bit
// exact: four independent accumulators reassociate the sum, which is exactly
// the reassociation the compiler is forbidden from doing on its own, so a small
// tolerance is the honest expectation rather than a fudge.
void test_fp32_neon_matches_scalar_within_tolerance() {
  const int shapes[][3] = {
      {1, 1, 4}, {1, 1, 5}, {3, 5, 7}, {4, 4, 16}, {8, 9, 33}, {16, 16, 128},
  };

  for (const auto& shape : shapes) {
    const int m = shape[0];
    const int n = shape[1];
    const int k = shape[2];

    const auto a_f = random_matrix(m, k, 1.0f, 71u + static_cast<unsigned>(k));
    const auto b_f = random_matrix(k, n, 1.0f, 83u + static_cast<unsigned>(n));

    std::vector<float> bt_f(b_f.size());
    qik::transpose_fp32(b_f.data(), bt_f.data(), k, n);

    std::vector<float> want(static_cast<std::size_t>(m) * n);
    std::vector<float> got(static_cast<std::size_t>(m) * n);
    qik::gemm_fp32_transposed(a_f.data(), bt_f.data(), want.data(), m, n, k);
    qik::gemm_fp32_neon(a_f.data(), bt_f.data(), got.data(), m, n, k);

    assert(relative_error(got, want) < 1e-5);
  }
}

// The transpose must be an actual transpose. Cheap to check, and every layout
// bug downstream looks like a mysterious accuracy problem instead.
void test_transpose_round_trips() {
  constexpr int k = 5;
  constexpr int n = 3;
  const auto b = random_matrix(k, n, 1.0f, 7u);
  std::vector<float> bt(b.size());
  qik::transpose_fp32(b.data(), bt.data(), k, n);
  for (int row = 0; row < k; ++row) {
    for (int col = 0; col < n; ++col) {
      assert(bt[static_cast<std::size_t>(col) * k + row] ==
             b[static_cast<std::size_t>(row) * n + col]);
    }
  }
}


// The blocked kernel has three distinct paths -- the 4x4 main block, the
// leftover columns, and the leftover rows that fall back to the unblocked
// kernel -- so the shapes below deliberately land in every combination of
// m % 4 and n % 4, crossed with k on and off the 16-wide SDOT step. int32
// accumulation is exact, so agreement must be bit for bit.
void test_blocked_matches_scalar_bit_exactly() {
  const int shapes[][3] = {
      {4, 4, 16},   {4, 4, 17},   {8, 8, 64},    {8, 8, 63},
      {5, 4, 16},   {4, 5, 16},   {5, 5, 17},    {7, 6, 33},
      {3, 3, 16},   {1, 1, 16},   {2, 9, 48},    {16, 16, 128},
      {9, 13, 100}, {12, 8, 32},  {6, 11, 65},   {32, 32, 256},
  };

  for (const auto& shape : shapes) {
    const int m = shape[0];
    const int n = shape[1];
    const int k = shape[2];

    const auto a_f = random_matrix(m, k, 1.0f, 4321u + static_cast<unsigned>(k));
    const auto b_f = random_matrix(k, n, 1.0f, 8765u + static_cast<unsigned>(n));

    std::vector<std::int8_t> a_q(a_f.size());
    std::vector<std::int8_t> bt_q(b_f.size());
    std::vector<float> b_scales(static_cast<std::size_t>(n));
    qik::quantize_per_tensor(a_f.data(), a_q.data(), a_f.size());
    qik::quantize_weights_per_channel(b_f.data(), bt_q.data(), b_scales.data(), k, n);

    std::vector<std::int32_t> scalar(static_cast<std::size_t>(m) * n, 0);
    std::vector<std::int32_t> blocked(static_cast<std::size_t>(m) * n, 0);

    qik::gemm_int8_scalar(a_q.data(), bt_q.data(), scalar.data(), m, n, k);
    qik::gemm_int8_neon_blocked(a_q.data(), bt_q.data(), blocked.data(), m, n, k);

    for (std::size_t i = 0; i < scalar.size(); ++i) {
      assert(scalar[i] == blocked[i]);
    }
  }
}

// Every output must be written exactly once. A blocking bug that skips or
// double-visits a tile shows up as a leftover sentinel rather than as a wrong
// number, which the equality test above would not necessarily catch.
void test_blocked_writes_every_output() {
  constexpr int m = 7;
  constexpr int n = 6;
  constexpr int k = 33;
  constexpr std::int32_t kSentinel = 0x5EED5EED;

  const auto a_f = random_matrix(m, k, 1.0f, 555u);
  const auto b_f = random_matrix(k, n, 1.0f, 666u);
  std::vector<std::int8_t> a_q(a_f.size());
  std::vector<std::int8_t> bt_q(b_f.size());
  std::vector<float> b_scales(n);
  qik::quantize_per_tensor(a_f.data(), a_q.data(), a_f.size());
  qik::quantize_weights_per_channel(b_f.data(), bt_q.data(), b_scales.data(), k, n);

  std::vector<std::int32_t> out(static_cast<std::size_t>(m) * n, kSentinel);
  qik::gemm_int8_neon_blocked(a_q.data(), bt_q.data(), out.data(), m, n, k);
  for (const auto value : out) {
    assert(value != kSentinel);
  }
}

}  // namespace

int main() {
  std::printf("SDOT available: %s\n", qik::kHasNeonDotProduct ? "yes" : "no");
  test_transpose_round_trips();
  test_neon_matches_scalar_bit_exactly();
  test_fp32_neon_matches_scalar_within_tolerance();
  test_blocked_matches_scalar_bit_exactly();
  test_blocked_writes_every_output();
  test_quantization_round_trip_is_bounded();
  test_quantized_gemm_tracks_fp32_reference();
  test_per_channel_beats_per_tensor_on_skewed_columns();
  std::printf("all tests passed\n");
  return 0;
}
