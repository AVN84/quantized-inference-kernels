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
1,4096,4096          11.15    104.34    107.16     1.03x    9.61x
8,4096,4096          11.27    105.50    292.04     2.77x   25.91x
256,1024,1024        26.11    123.37    306.24     2.48x   11.73x
1024,4096,4096       11.09    108.35    306.15     2.83x   27.59x
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


## GPU backend (Metal)

The same kernel on the M3 Pro's 18 GPU cores, dispatched through Metal. Output
is **bit-identical to the CPU kernel at every shape** — int32 accumulation is
exact, so a GPU kernel that only "almost" matches is a broken GPU kernel, and
the benchmark refuses to print timings if any element disagrees.

```
shape (m,n,k)      CPU blocked   GPU Metal    GPU/CPU     exact
----------------------------------------------------------------
64,64,64                246.72        1.32      0.01x       yes
128,128,128             269.16        9.87      0.04x       yes
256,256,256             270.33       92.67      0.34x       yes
512,512,512             293.21      199.15      0.68x       yes
1024,1024,1024          307.14      274.10      0.89x       yes
256,1024,1024           307.27      200.28      0.65x       yes
1024,4096,4096          311.78      297.79      0.96x       yes
```

`make metal`. Raw output in `benchmarks/apple-m3-pro-metal.txt`.

### The GPU loses, and that is the finding

At 64x64x64 the GPU is roughly **100x slower** than one CPU core. It closes the
gap as problems grow and still does not clearly win at 1024x4096x4096. Two
reasons, and neither is "GPUs are overrated":

**Fixed costs dominate small problems.** Buffer creation, encoding, and
dispatch are paid per call regardless of size. A 64x64x64 GEMM is about a
million multiply-accumulates — less work than the dispatch costs to set up.
There is a real crossover here and it sits far to the right of where people
assume.

**The kernel is deliberately naive.** One thread per output element, each
re-reading a full row of A and a full row of Bt from device memory with no
threadgroup tiling and no shared-memory reuse. It is bandwidth-bound by
construction. The CPU kernel it is being compared against is the *optimized*
one, with 4x4 register blocking. This is a tuned CPU kernel against an
untuned GPU kernel, and the table should be read that way.

### Two things this measurement does not do

**Buffers are allocated inside the timed region.** A real inference server
allocates once and reuses across calls; doing it per dispatch penalizes the GPU
on every row above. That is a deliberate choice to keep the harness honest
about what is currently implemented, not an argument that it is optimal.

**Unified memory flatters these numbers.** Apple Silicon shares physical memory
between CPU and GPU, so `MTLResourceStorageModeShared` buffers are views, not
copies. A discrete GPU would pay a host-to-device transfer that often dominates
a GEMM this size. Do not read this table as generalizing to CUDA.

### Implementation notes worth knowing

The shader is compiled **at runtime** from source rather than shipped as a
`.metallib`. The offline `metal` compiler ships with full Xcode, not the
Command Line Tools, so precompiling would make the repo un-buildable on a
machine that has a working GPU and the Metal framework but no Xcode.

`MTLCreateSystemDefaultDevice()` returns nil without a window server session —
over SSH, or in a build agent — on a machine whose GPU is present and working.
`MTLCopyAllDevices()` still enumerates it. The code tries the default and falls
back, rather than reporting "no GPU" on hardware that has one.

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

### Why there is no cache-tiling pass

The obvious next step looked like blocking a panel of B into L2. Before writing
it I worked out what each benchmark shape actually costs in DRAM traffic, and
five of the six could not have shown a difference: their entire B fits in L2, so
re-streaming it is free because it never leaves cache. Optimizing against them
would have measured nothing and proved nothing.

The shape that does stress it is `1024,4096,4096` — 16 MB of weights against a
4 MB L2, re-read once per block of four output rows, about **4 GB** of traffic.
Added it, and register blocking already carries it to **306 GOPS at 2.83x**,
right alongside the shapes whose weights fit in cache entirely.

So the bandwidth problem is already solved: cutting B re-reads by 4x moved the
kernel off the memory wall, and an L2 panel pass would be optimizing something
that is no longer the bottleneck. Measuring first saved the work.

## Next

- INT4 with packed nibbles.
- SMMLA (`__ARM_FEATURE_MATMUL_INT8`) for 2x over SDOT on supporting cores.
- Threadgroup tiling in the Metal kernel, which is where the GPU gap is.
- Buffer reuse across dispatches rather than per-call allocation.
- CUDA on Perlmutter, once the reference and harness exist (they now do).
