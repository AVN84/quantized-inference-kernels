#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "qik/gemm.hpp"
#include "qik/gemm_neon.hpp"
#include "qik/quantize.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Shape {
  int m;
  int n;
  int k;
};

std::vector<float> random_matrix(int rows, int cols, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> out(static_cast<std::size_t>(rows) * cols);
  for (auto& value : out) {
    value = dist(rng);
  }
  return out;
}

// Median of repeated runs rather than the mean. A single scheduler preemption
// or a thermal blip skews a mean and leaves no trace; the median ignores it.
// Reporting the best-of would be worse still -- it measures the luckiest run,
// not the machine.
template <typename Fn>
double median_seconds(Fn&& fn, int repetitions) {
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repetitions));
  for (int i = 0; i < repetitions; ++i) {
    const auto start = Clock::now();
    fn();
    const auto end = Clock::now();
    samples.push_back(
        std::chrono::duration<double>(end - start).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

double gops(const Shape& shape, double seconds) {
  // Two operations per multiply-accumulate.
  const double ops = 2.0 * shape.m * shape.n * shape.k;
  return ops / seconds / 1e9;
}

}  // namespace

int main() {
  if (!qik::kHasNeonDotProduct) {
    std::printf(
        "WARNING: built without __ARM_FEATURE_DOTPROD. The \"neon\" column below\n"
        "         is the scalar kernel and is not a NEON measurement.\n\n");
  }

  const std::vector<Shape> shapes = {
      {128, 128, 128},   {256, 256, 256},  {512, 512, 512},
      {1, 4096, 4096},   {8, 4096, 4096},  {256, 1024, 1024},
  };

  std::printf("%-16s %10s %10s %10s %10s\n", "shape (m,n,k)", "fp32 scalar",
              "fp32 NEON", "i8 NEON", "i8 / fp32");
  std::printf("%s\n", std::string(62, '-').c_str());

  std::vector<std::string> rows;

  for (const auto& shape : shapes) {
    const auto a_f = random_matrix(shape.m, shape.k, 1u);
    const auto b_f = random_matrix(shape.k, shape.n, 2u);

    std::vector<std::int8_t> a_q(a_f.size());
    std::vector<std::int8_t> bt_q(b_f.size());
    std::vector<float> b_scales(static_cast<std::size_t>(shape.n));
    qik::quantize_per_tensor(a_f.data(), a_q.data(), a_f.size());
    qik::quantize_weights_per_channel(b_f.data(), bt_q.data(), b_scales.data(),
                                      shape.k, shape.n);

    std::vector<float> bt_f(b_f.size());
    qik::transpose_fp32(b_f.data(), bt_f.data(), shape.k, shape.n);

    std::vector<float> c_f(static_cast<std::size_t>(shape.m) * shape.n);
    std::vector<std::int32_t> c_scalar(c_f.size());
    std::vector<std::int32_t> c_neon(c_f.size());

    // Large fp32 reference runs are slow enough that fewer repetitions still
    // give a stable median.
    const bool large = static_cast<long long>(shape.m) * shape.n * shape.k >
                       64LL * 1024 * 1024;
    const int fp32_reps = large ? 3 : 7;

    const double fp32_s = median_seconds(
        [&] {
          qik::gemm_fp32_reference(a_f.data(), b_f.data(), c_f.data(), shape.m,
                                   shape.n, shape.k);
        },
        fp32_reps);

    const double fp32t_s = median_seconds(
        [&] {
          qik::gemm_fp32_transposed(a_f.data(), bt_f.data(), c_f.data(),
                                    shape.m, shape.n, shape.k);
        },
        fp32_reps);

    const double fp32n_s = median_seconds(
        [&] {
          qik::gemm_fp32_neon(a_f.data(), bt_f.data(), c_f.data(), shape.m,
                              shape.n, shape.k);
        },
        11);

    const double scalar_s = median_seconds(
        [&] {
          qik::gemm_int8_scalar(a_q.data(), bt_q.data(), c_scalar.data(),
                                shape.m, shape.n, shape.k);
        },
        fp32_reps);

    const double neon_s = median_seconds(
        [&] {
          qik::gemm_int8_neon(a_q.data(), bt_q.data(), c_neon.data(), shape.m,
                              shape.n, shape.k);
        },
        11);

    // Never report a number without checking the kernel was still correct.
    for (std::size_t i = 0; i < c_scalar.size(); ++i) {
      if (c_scalar[i] != c_neon[i]) {
        std::fprintf(stderr, "MISMATCH at %zu -- refusing to report timings\n", i);
        return 1;
      }
    }

    char label[64];
    std::snprintf(label, sizeof(label), "%d,%d,%d", shape.m, shape.n, shape.k);
    std::printf("%-16s %10.2f %10.2f %10.2f %9.2fx\n", label,
                gops(shape, fp32t_s), gops(shape, fp32n_s),
                gops(shape, neon_s), fp32n_s / neon_s);
    (void)fp32_s;
    (void)scalar_s;
  }

  std::printf(
      "\nNotes: single threaded, no tiling. All kernels share the transposed-B\n"
      "layout and 4-wide n blocking, so the last column isolates the datatype and\n"
      "nothing else. Theoretical ceiling is 4x: FMLA retires 4 fp32 MACs per\n"
      "instruction, SDOT retires 16 int8 ones. Not a comparison to a tuned BLAS.\n");
  return 0;
}
