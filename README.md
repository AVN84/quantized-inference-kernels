# Quantized Inference Kernels

Single-threaded ARM NEON int8 GEMM kernels, int4 weight packing, and a Metal
reference backend. The CPU kernels use the same transposed-B layout as a
hand-vectorized fp32 baseline.

## Implemented

- Flat SDOT and 4-by-4 register-blocked int8 GEMM.
- An experimental SMMLA path, retained for comparison.
- Per-channel and per-tensor quantization, including int4 packing.
- A Metal backend with exact int32 output checks against the CPU reference.
- Correctness, tail-shape, quantization-error, and sanitizer checks.

## Checked-in CPU results

The following is the existing record in
[benchmarks/apple-m3-pro.txt](benchmarks/apple-m3-pro.txt), on an Apple M3 Pro.
Values are GOPS; speedups depend on shape, memory traffic, and measurement noise.

| Shape (m,n,k) | fp32 NEON | int8 flat | int8 blocked | Blocked / flat | Blocked / fp32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128,128,128 | 41.34 | 186.07 | 267.73 | 1.44x | 6.48x |
| 256,256,256 | 30.35 | 187.19 | 272.61 | 1.46x | 8.98x |
| 512,512,512 | 27.34 | 131.01 | 293.10 | 2.24x | 10.72x |
| 1,4096,4096 | 14.55 | 48.31 | 54.88 | 1.14x | 3.77x |
| 8,4096,4096 | 11.52 | 72.35 | 292.11 | 4.04x | 25.36x |
| 256,1024,1024 | 26.09 | 122.83 | 305.39 | 2.49x | 11.71x |
| 1024,4096,4096 | 12.44 | 86.65 | 173.05 | 2.00x | 13.91x |

This record peaks at 305.39 GOPS. The single-row case has no inter-row reuse,
so its observed 1.14x difference should not be interpreted as a structural
register-blocking gain. These are checked-in measurements, not a new run.

## Metal results

The raw [Metal record](benchmarks/apple-m3-pro-metal.txt) includes buffer
creation and dispatch, but excludes shader compilation. It uses Apple unified
memory; it is not a discrete-GPU or CUDA transfer benchmark.

| Shape (m,n,k) | CPU blocked GOPS | Metal GOPS | GPU / CPU | Exact |
| --- | ---: | ---: | ---: | --- |
| 64,64,64 | 246.72 | 2.53 | 0.01x | yes |
| 128,128,128 | 271.34 | 16.05 | 0.06x | yes |
| 256,256,256 | 268.62 | 42.29 | 0.16x | yes |
| 512,512,512 | 297.74 | 83.13 | 0.28x | yes |
| 1024,1024,1024 | 306.55 | 166.13 | 0.54x | yes |
| 256,1024,1024 | 309.45 | 181.78 | 0.59x | yes |
| 1024,4096,4096 | 296.90 | 298.74 | 1.01x | yes |

The naive Metal kernel is slower on smaller shapes and approximately tied on
the largest recorded shape. This comparison does not establish the performance
of an optimized, tiled GPU kernel.

## Accuracy and limitations

[Quantization-error measurements](benchmarks/accuracy-int8-vs-int4.txt) compare
int8/int4 and per-channel/per-tensor scaling. Per-tensor scaling can zero narrow
columns when another column has much larger magnitude; aggregate relative error
can hide that failure.

These are educational kernels, not a tuned BLAS replacement. The CPU comparison
is single-threaded; the Metal implementation has no threadgroup tiling or buffer
reuse. Results should always be paired with the exact workload and baseline.

## Build and checks

```sh
make test
make sanitize
make bench
make metal
make accuracy
```

The vectorized CPU path requires ARMv8.2 dot-product support; Metal requires
macOS. See the Makefile and source for the available fallback behavior.
