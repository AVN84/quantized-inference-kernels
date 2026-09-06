// Measures what quantization actually costs in accuracy, as a function of bit
// width, scaling policy, and reduction length.
//
// Throughput numbers say what a kernel can do. They say nothing about whether
// the output is still usable, which is the question that decides whether you
// can ship the kernel at all. This is that measurement.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "qik/gemm.hpp"
#include "qik/quantize.hpp"
#include "qik/quantize4.hpp"

namespace {

std::vector<float> random_matrix(int rows, int cols, std::uint32_t seed,
                                 float spread = 1.0f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, spread);
  std::vector<float> out(static_cast<std::size_t>(rows) * cols);
  for (auto& v : out) { v = dist(rng); }
  return out;
}

// Aggregate relative error: total absolute deviation over total magnitude.
// Normalizing per element would let near-zero outputs dominate the metric.
double relative_error(const std::vector<float>& got,
                      const std::vector<float>& want) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    num += std::abs(static_cast<double>(got[i]) - want[i]);
    den += std::abs(static_cast<double>(want[i]));
  }
  return den > 0.0 ? num / den : 0.0;
}

struct Result { double int8_pc, int8_pt, int4_pc, int4_pt; };

// Error restricted to a subset of output columns. The aggregate metric
// normalizes by total magnitude, so a single very large column inflates the
// denominator and hides whatever is happening to the small ones. Measuring the
// narrow columns alone is the difference between "6% error" and "these outputs
// are gone".
double relative_error_cols(const std::vector<float>& got,
                           const std::vector<float>& want, int m, int n,
                           int first_col) {
  double num = 0.0, den = 0.0;
  for (int row = 0; row < m; ++row) {
    for (int col = first_col; col < n; ++col) {
      const std::size_t i = static_cast<std::size_t>(row) * n + col;
      num += std::abs(static_cast<double>(got[i]) - want[i]);
      den += std::abs(static_cast<double>(want[i]));
    }
  }
  return den > 0.0 ? num / den : 0.0;
}

Result measure(int m, int n, int k, std::uint32_t seed, bool skew) {
  auto a_f = random_matrix(m, k, seed);
  auto b_f = random_matrix(k, n, seed + 1);
  if (skew) {
    // One output channel three orders of magnitude wider than the rest.
    for (int row = 0; row < k; ++row) {
      b_f[static_cast<std::size_t>(row) * n + 0] *= 1000.0f;
    }
  }

  std::vector<float> want(static_cast<std::size_t>(m) * n);
  qik::gemm_fp32_reference(a_f.data(), b_f.data(), want.data(), m, n, k);

  std::vector<std::int8_t> a_q(a_f.size());
  const float a_scale = qik::quantize_per_tensor(a_f.data(), a_q.data(), a_f.size());

  const std::size_t stride4 = qik::packed_bytes(static_cast<std::size_t>(k));
  std::vector<std::int8_t> bt8(b_f.size());
  std::vector<std::uint8_t> bt4(stride4 * static_cast<std::size_t>(n));
  std::vector<float> sc_pc(static_cast<std::size_t>(n));
  std::vector<float> sc4_pc(static_cast<std::size_t>(n));

  qik::quantize_weights_per_channel(b_f.data(), bt8.data(), sc_pc.data(), k, n);
  qik::quantize_weights_per_channel_int4(b_f.data(), bt4.data(), sc4_pc.data(), k, n);

  std::vector<std::int8_t> bt8_pt(b_f.size());
  std::vector<std::uint8_t> bt4_pt(bt4.size());
  const float s8 = qik::quantize_weights_per_tensor(b_f.data(), bt8_pt.data(), k, n);
  const float s4 = qik::quantize_weights_per_tensor_int4(b_f.data(), bt4_pt.data(), k, n);
  std::vector<float> sc_pt(static_cast<std::size_t>(n), s8);
  std::vector<float> sc4_pt(static_cast<std::size_t>(n), s4);

  std::vector<std::int32_t> acc(static_cast<std::size_t>(m) * n);
  std::vector<float> out(acc.size());
  Result r{};

  qik::gemm_int8_scalar(a_q.data(), bt8.data(), acc.data(), m, n, k);
  qik::dequantize_per_channel(acc.data(), out.data(), m, n, a_scale, sc_pc.data());
  r.int8_pc = relative_error(out, want);

  qik::gemm_int8_scalar(a_q.data(), bt8_pt.data(), acc.data(), m, n, k);
  qik::dequantize_per_channel(acc.data(), out.data(), m, n, a_scale, sc_pt.data());
  r.int8_pt = relative_error(out, want);

  qik::gemm_int4_scalar(a_q.data(), bt4.data(), acc.data(), m, n, k);
  qik::dequantize_per_channel(acc.data(), out.data(), m, n, a_scale, sc4_pc.data());
  r.int4_pc = relative_error(out, want);

  qik::gemm_int4_scalar(a_q.data(), bt4_pt.data(), acc.data(), m, n, k);
  qik::dequantize_per_channel(acc.data(), out.data(), m, n, a_scale, sc4_pt.data());
  r.int4_pt = relative_error(out, want);

  return r;
}

}  // namespace

int main() {
  std::printf("Relative error vs fp32. Lower is better. m=n=64, Gaussian inputs.\n");
  std::printf("pc = per-channel weight scales, pt = one scale for the whole matrix.\n\n");

  std::printf("%8s  %10s %10s   %10s %10s\n", "K", "int8 pc", "int8 pt", "int4 pc", "int4 pt");
  std::printf("%s\n", std::string(56, '-').c_str());
  for (int k : {32, 128, 512, 2048, 8192}) {
    const Result r = measure(64, 64, k, 7u + static_cast<unsigned>(k), false);
    std::printf("%8d  %10.4f %10.4f   %10.4f %10.4f\n", k, r.int8_pc, r.int8_pt,
                r.int4_pc, r.int4_pt);
  }

  std::printf("\nSame, with one weight column scaled 1000x (the pathology\n");
  std::printf("per-channel scaling exists to prevent):\n\n");
  std::printf("%8s  %10s %10s   %10s %10s\n", "K", "int8 pc", "int8 pt", "int4 pc", "int4 pt");
  std::printf("%s\n", std::string(56, '-').c_str());
  for (int k : {128, 512, 2048}) {
    const Result r = measure(64, 64, k, 91u + static_cast<unsigned>(k), true);
    std::printf("%8d  %10.4f %10.4f   %10.4f %10.4f\n", k, r.int8_pc, r.int8_pt,
                r.int4_pc, r.int4_pt);
  }

  // The aggregate numbers above understate the damage. Re-measure the skewed
  // case over only the narrow columns -- the ones per-tensor scaling destroys.
  std::printf("\nSkewed case again, measuring ONLY the narrow columns\n");
  std::printf("(excluding the 1000x column that inflates the denominator):\n\n");
  std::printf("%8s  %10s %10s   %10s %10s\n", "K", "int8 pc", "int8 pt", "int4 pc", "int4 pt");
  std::printf("%s\n", std::string(56, '-').c_str());
  for (int k : {128, 512, 2048}) {
    constexpr int m = 64, n = 64;
    auto a_f = random_matrix(m, k, 91u + static_cast<unsigned>(k));
    auto b_f = random_matrix(k, n, 92u + static_cast<unsigned>(k));
    for (int row = 0; row < k; ++row) {
      b_f[static_cast<std::size_t>(row) * n + 0] *= 1000.0f;
    }
    std::vector<float> want(static_cast<std::size_t>(m) * n);
    qik::gemm_fp32_reference(a_f.data(), b_f.data(), want.data(), m, n, k);

    std::vector<std::int8_t> a_q(a_f.size());
    const float a_scale = qik::quantize_per_tensor(a_f.data(), a_q.data(), a_f.size());
    const std::size_t stride4 = qik::packed_bytes(static_cast<std::size_t>(k));

    std::vector<std::int8_t> bt8(b_f.size()), bt8_pt(b_f.size());
    std::vector<std::uint8_t> bt4(stride4 * n), bt4_pt(stride4 * n);
    std::vector<float> sc_pc(n), sc4_pc(n);
    qik::quantize_weights_per_channel(b_f.data(), bt8.data(), sc_pc.data(), k, n);
    qik::quantize_weights_per_channel_int4(b_f.data(), bt4.data(), sc4_pc.data(), k, n);
    const float s8 = qik::quantize_weights_per_tensor(b_f.data(), bt8_pt.data(), k, n);
    const float s4 = qik::quantize_weights_per_tensor_int4(b_f.data(), bt4_pt.data(), k, n);
    std::vector<float> sc_pt(n, s8), sc4_pt(n, s4);

    std::vector<std::int32_t> acc(static_cast<std::size_t>(m) * n);
    std::vector<float> out(acc.size());
    double e[4];
    qik::gemm_int8_scalar(a_q.data(), bt8.data(), acc.data(), m, n, k);
    qik::dequantize_per_channel(acc.data(), out.data(), m, n, a_scale, sc_pc.data());
    e[0] = relative_error_cols(out, want, m, n, 1);
    qik::gemm_int8_scalar(a_q.data(), bt8_pt.data(), acc.data(), m, n, k);
    qik::dequantize_per_channel(acc.data(), out.data(), m, n, a_scale, sc_pt.data());
    e[1] = relative_error_cols(out, want, m, n, 1);
    qik::gemm_int4_scalar(a_q.data(), bt4.data(), acc.data(), m, n, k);
    qik::dequantize_per_channel(acc.data(), out.data(), m, n, a_scale, sc4_pc.data());
    e[2] = relative_error_cols(out, want, m, n, 1);
    qik::gemm_int4_scalar(a_q.data(), bt4_pt.data(), acc.data(), m, n, k);
    qik::dequantize_per_channel(acc.data(), out.data(), m, n, a_scale, sc4_pt.data());
    e[3] = relative_error_cols(out, want, m, n, 1);
    std::printf("%8d  %10.4f %10.4f   %10.4f %10.4f\n", k, e[0], e[1], e[2], e[3]);
  }

  std::printf("\nWeight storage per element: fp32 4.0 B, int8 1.0 B, int4 0.5 B.\n");
  return 0;
}
