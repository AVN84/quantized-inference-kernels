#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "qik/gemm.hpp"
#include "qik/gemm_neon.hpp"
#include "qik/metal_gemm.hpp"
#include "qik/quantize.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Shape { int m, n, k; };

std::vector<float> random_matrix(int rows, int cols, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> out(static_cast<std::size_t>(rows) * cols);
  for (auto& v : out) { v = dist(rng); }
  return out;
}

template <typename Fn>
double median_seconds(Fn&& fn, int reps) {
  std::vector<double> s;
  s.reserve(static_cast<std::size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    const auto t0 = Clock::now();
    fn();
    s.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
  }
  std::sort(s.begin(), s.end());
  return s[s.size() / 2];
}

double gops(const Shape& sh, double sec) {
  return 2.0 * sh.m * sh.n * sh.k / sec / 1e9;
}

}  // namespace

int main() {
  std::string err;
  if (!qik::metal_available(&err)) {
    std::printf("Metal unavailable: %s\n", err.c_str());
    std::printf("Skipping GPU benchmark. CPU kernels are unaffected.\n");
    return 0;
  }
  std::printf("GPU: %s\n\n", qik::metal_device_name().c_str());

  // Small shapes are included deliberately. The GPU is expected to lose on
  // them -- dispatch and buffer setup are fixed costs that a small problem
  // cannot amortize -- and the crossover is the interesting number.
  const std::vector<Shape> shapes = {
      {64, 64, 64},      {128, 128, 128},   {256, 256, 256},
      {512, 512, 512},   {1024, 1024, 1024}, {256, 1024, 1024},
      {1024, 4096, 4096},
  };

  std::printf("%-18s %11s %11s %10s %9s\n", "shape (m,n,k)", "CPU blocked",
              "GPU Metal", "GPU/CPU", "exact");
  std::printf("%s\n", std::string(64, '-').c_str());

  for (const auto& sh : shapes) {
    const auto a_f = random_matrix(sh.m, sh.k, 11u);
    const auto b_f = random_matrix(sh.k, sh.n, 22u);

    std::vector<std::int8_t> a_q(a_f.size());
    std::vector<std::int8_t> bt_q(b_f.size());
    std::vector<float> scales(static_cast<std::size_t>(sh.n));
    qik::quantize_per_tensor(a_f.data(), a_q.data(), a_f.size());
    qik::quantize_weights_per_channel(b_f.data(), bt_q.data(), scales.data(), sh.k, sh.n);

    std::vector<std::int32_t> c_cpu(static_cast<std::size_t>(sh.m) * sh.n);
    std::vector<std::int32_t> c_gpu(c_cpu.size());

    const long long work = 1LL * sh.m * sh.n * sh.k;
    const int reps = work > (1LL << 32) ? 3 : work > (1LL << 26) ? 5 : 15;

    qik::gemm_int8_neon_blocked(a_q.data(), bt_q.data(), c_cpu.data(), sh.m, sh.n, sh.k);
    const double cpu_s = median_seconds(
        [&] {
          qik::gemm_int8_neon_blocked(a_q.data(), bt_q.data(), c_cpu.data(),
                                      sh.m, sh.n, sh.k);
        }, reps);

    if (!qik::gemm_int8_metal(a_q.data(), bt_q.data(), c_gpu.data(),
                              sh.m, sh.n, sh.k, &err)) {
      std::printf("GPU failed at %d,%d,%d: %s\n", sh.m, sh.n, sh.k, err.c_str());
      return 1;
    }
    const double gpu_s = median_seconds(
        [&] {
          qik::gemm_int8_metal(a_q.data(), bt_q.data(), c_gpu.data(),
                               sh.m, sh.n, sh.k, nullptr);
        }, reps);

    // int32 accumulation is exact on both sides, so this is bit equality, not
    // a tolerance. A GPU kernel that "almost" matches is a broken GPU kernel.
    bool exact = true;
    for (std::size_t i = 0; i < c_cpu.size(); ++i) {
      if (c_cpu[i] != c_gpu[i]) { exact = false; break; }
    }

    char label[48];
    std::snprintf(label, sizeof(label), "%d,%d,%d", sh.m, sh.n, sh.k);
    std::printf("%-18s %11.2f %11.2f %9.2fx %9s\n", label,
                gops(sh, cpu_s), gops(sh, gpu_s), cpu_s / gpu_s,
                exact ? "yes" : "NO");
    if (!exact) { std::printf("  MISMATCH -- refusing to trust these timings\n"); return 1; }
  }

  std::printf(
      "\nNotes: GPU timing includes buffer creation and dispatch, not shader\n"
      "compilation (done once at startup). Apple Silicon has unified memory, so\n"
      "MTLResourceStorageModeShared buffers are not copied -- a discrete GPU\n"
      "would additionally pay a host-to-device transfer this table does not show.\n");
  return 0;
}
