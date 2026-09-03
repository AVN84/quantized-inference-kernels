# Quantized Inference Kernels

int8 GEMM kernels for ARM NEON, with a fp32 baseline built to be a fair
opponent rather than an easy one.

The question this repo answers: **how much of int8's reputation for speed is the
datatype, and how much is everything else you changed at the same time?**

## Results

Apple M3 Pro, single threaded, no tiling, `clang++ -O3`. GOPS, higher is better.

```
shape (m,n,k)    fp32 NEON   i8 flat i8 blocked  blk gain  vs fp32
--------------------------------------------------------------------
128,128,128          41.29    193.96    267.73     1.38x    6.48x
256,256,256          30.84    187.72    270.78     1.44x    8.78x
512,512,512          27.71    131.83    299.55     2.27x   10.81x
1,4096,4096          10.11    111.71    112.47     1.01x   11.13x
8,4096,4096          11.41    114.96    308.92     2.69x   27.07x
256,1024,1024        26.25    124.57    309.11     2.48x   11.78x
```

Reproduce with `make bench`. Raw output in `benchmarks/`.

## Getting the baseline honest

The first version of this benchmark reported **149x**. That number was garbage,
and the path to a real one is most of what the repo is about.

**Mistake one: the wrong memory layout.** The naive fp32 reference strides
through `B` by `n`, missing cache on nearly every access, while the int8 kernel
walks a transposed `B` contiguously. Comparing them measured the access pattern,
not the datatype. Fixed by giving both kernels the same transposed layout.

**Mistake two: unequal optimizer permissions.** With the layout fixed, fp32 came
in at ~2 GFLOPS against int8's ~40, still an implausible 20x. The cause is that
floating-point addition is not associative, so the compiler may not reassociate
a fp32 reduction into vector accumulators without `-ffast-math`. int32
accumulation *is* associative, so the scalar int8 loop auto-vectorizes for free.
The comparison was measuring what the optimizer was allowed to do.

Fixed by hand-vectorizing the fp32 kernel with the same four-accumulator
structure, making the reassociation explicit on both sides.

What survives is **3.1x to 9.3x**, which is a number worth defending.

## Why some shapes beat the theoretical ceiling

FMLA retires 4 fp32 multiply-accumulates per instruction. SDOT retires 16 int8
ones. So the compute ceiling is 4x — and three of the six shapes exceed it.

That is not an error, it is the second effect: int8 also cuts **memory traffic**
by 4x. At `8,4096,4096` the weight matrix is 64 MB as fp32 and 16 MB as int8,
so the fp32 kernel is bandwidth-starved while the int8 kernel still fits usable
working sets in cache. That shape is the most memory-bound in the table and
posts the largest win, 9.28x. The compute-bound shapes land near the 4x ceiling
where theory says they should.

Stated plainly: **int8 wins twice, once on issue width and once on bandwidth,
and which one dominates depends entirely on the shape.**


## Register blocking

The flat kernel holds one row of A against four rows of B: five vector loads
feeding four SDOTs per 16-element step, or 0.8 multiply-accumulates per load.
Every row of A re-streams all of B. At 4096x4096 the weight matrix is 16 MB
against a 4 MB L2, so it crosses the memory bus once per output row and the
kernel waits on DRAM instead of issuing arithmetic.

Blocking 4 rows of A against 4 rows of B gives eight loads feeding sixteen
SDOTs — 2.0 MACs per load, and B is fetched once per *block* of output rows.
Sixteen accumulators plus eight operand registers is 24 of the 32 architectural
SIMD registers; wider spills, and the spill costs more than the reuse buys.

Result: **1.4x to 2.7x**, largest on the shapes that were most bandwidth-starved.

**M=1 is 1.01x, and that is the point.** A single output row has no reuse to
recover, so the blocked path falls back to the flat kernel and the two are the
same code. Measuring 1.01x is the check that the benchmark is honest.

### The benchmark bug that caught

The first blocked run reported **1.73x on M=1** — for code that is byte-identical
to what it was being compared against. The cause was ordering: the flat kernel
ran first and paid every compulsory cache miss on a 16 MB B, and the blocked
kernel ran second against a warm cache. The speedup was the cache, not the code.

Fixed with an untimed warm-up pass before every measurement. M=1 then reports
1.01x, as it must. A benchmark that cannot reproduce a known-zero result is not
measuring what it claims to.

## Per-channel scaling

Weights are quantized per output channel, not per tensor. The failure mode this
avoids is one wide column setting the step size for every other column. The test
suite builds exactly that pathology — one column scaled 1000x — and measures it:

```
per-channel error 0.0116    per-tensor error 1.0000
```

Per-tensor error of 1.0 means total loss: the narrow columns quantize to zero
and the output is zero. Per-channel costs one float per output channel.

## Correctness

The strongest property available here is that int32 accumulation is **exact**,
so the vectorized int8 kernel has no tolerance to hide behind — it must match
the scalar definition bit for bit. The suite asserts that across twelve shapes
chosen to straddle the 16-wide SDOT step and the 4-wide `n` blocking, so every
tail path is exercised. The blocked kernel gets sixteen more shapes covering
every combination of `m % 4` and `n % 4`, plus a sentinel test that every output
is written exactly once — a tiling bug that skips a tile shows up as an unwritten
value, not a wrong one.

Also covered: quantization round-trip bounded by half a step, zero tensors not
producing a zero scale, the transpose actually transposing, fp32 NEON agreeing
with scalar fp32 within reassociation tolerance, end-to-end quantized GEMM
within 5% of the fp32 reference, and the per-channel gap above.

`make test` and `make sanitize` (ASan + UBSan). The benchmark re-verifies the
kernel against the scalar version before printing any timing, and refuses to
report numbers if they disagree.

## Build

```
make test       # correctness
make bench      # benchmark table
make sanitize   # ASan + UBSan
```

Requires ARMv8.2 dot product (`__ARM_FEATURE_DOTPROD`). Without it the build
still works, `kHasNeonDotProduct` reports false, and the benchmark prints a
warning instead of quietly reporting scalar numbers as NEON.

## Scope

Single threaded. No tiling or cache blocking. No multi-threading. Not compared
against a tuned BLAS — Accelerate would win, and the point here is the datatype
comparison at matched implementation effort, not a BLAS competition.

## Next

- Multi-level tiling over k, to keep a B panel resident in L2 across n blocks.
- INT4 with packed nibbles.
- SMMLA (`__ARM_FEATURE_MATMUL_INT8`) for 2x over SDOT on supporting cores.
- A Metal compute port, then CUDA once there is hardware to run it on.
