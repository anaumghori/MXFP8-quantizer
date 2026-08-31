# MXFP8 Quantizer in CuTeDSL for Blackwell

A CuTeDSL implementation of an MXFP8 quantizer for NVIDIA Blackwell (SM100) GPUs, written and benchmarked on a B200. It takes an FP16 or BF16 matrix and produces an FP8 E4M3 payload along with one UE8M0 scale byte for every 1×32 block of values along the K dimension. In MXFP8, each contiguous block of 32 values along a row has its own scale, and that scale is restricted to a power of two. The scale bytes are written directly in the packed layout that Blackwell's block-scaled Tensor Cores expect, so a downstream block-scaled GEMM can use the output as is, with no repacking step in between.

This is a from scratch reconstruction of the design described in FAL's blog post [Chasing 6+ TB/s: an MXFP8 quantizer on Blackwell](https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/). On large, full sized problems it sustains about 6.4 TB/s of effective bandwidth on a B200. The claim about the scale layout is checked end to end: the packed output is fed straight into an unmodified NVIDIA SM100 block-scaled GEMM kernel, and the result is compared against an fp32 reference.

The kernel takes an input matrix `X` of shape `(M, K)`, with every value stored in FP16 or BF16, and produces two output tensors from it:

- `Q` -> The quantized payload has the same (M, K) tensor shape as the input, but its entries/values are stored using FP8 E4M3, an 8-bit floating-point format with 4 exponent bits and 3 mantissa bits. As a result, Q requires only half the memory of the original FP16 or BF16 matrix.
- `S` -> The scale tensor has a logical shape of `(M, K/32)`, since every contiguous block of 32 values along a row of `X` shares a single scale. Each of these scales is stored as one byte in the UE8M0 format. That byte does not hold the scale directly, it holds a plain number `b` between 0 and 255, and you get the real scale by computing `2^(b - 127)`. The reason 127 is there at all is just to let this one unsigned byte cover both small and large scales: if `b` is exactly 127, the scale is `2^0`, which is 1, meaning no scaling happens. If `b` is smaller than 127, say 126, the scale becomes `2^-1`, or 0.5, so subtracting 127 is what lets the byte produce a fraction even though the byte itself can never go negative. If `b` is larger, say 128, the scale becomes `2^1`, or 2. So 127 is simply the "zero point" of the byte, everything above it means scale up, everything below it means scale down, and the actual multiplier applied to a block of 32 values in `X` is whatever power of two that byte works out to.

Both output tensors come from the same recipe, applied independently to every 32 value block along K in every row. For each block, the kernel:

1. Looks at all 32 values in the block and finds the one with the largest magnitude, ignoring sign, since it's the size of the value that matters for fitting it into FP8.
2. Picks a scale for the whole block based on that largest value. Specifically, it chooses the smallest power of two that, once the block is divided by it, brings that largest value down to fit inside FP8's representable range.
3. Applies the reciprocal of the scale to every value in the block.
4. Converts each of those scaled values into the FP8 E4M3 format. If a value still ends up slightly outside what FP8 can represent, due to rounding, it gets clipped to the nearest value FP8 can actually hold.
5. Writes the 32 quantized values into their place in the output payload `Q`.
6. Writes the single scale byte for the block into its place in the output scale tensor `S`.

The scales cannot be stored as a plain dense `(M, K/32)` tensor because Blackwell's block-scaled Tensor Core path, through the `tcgen05.mma` instruction, expects them in a specific packed layout. CuTeDSL models this layout as `BlockScaledBasicChunk`: chunks of 128 rows by 4 scale columns, packed into 512-byte units. If the scales were first computed in the simple dense layout, we would need an extra pass afterward to rearrange them into this format before a block-scaled GEMM could use them.

The kernel skips that extra pass entirely: every scale is written straight into its final position in the packed layout, with no dense intermediate tensor in between. Within each 128-row by 4-column chunk, groups of rows are interleaved at 4-byte strides, while the scale bytes belonging to consecutive K blocks in the same row are packed together into a single 32-bit store. For shapes aligned to the 128-row by 4-column chunk boundaries, the scale buffer has no padding: every byte corresponds directly to a scale value and is already in the layout required by the GEMM. Shapes that do not align are zero-padded up to those boundaries before launch (the benchmark table's Padded fallback rows).

The execution structure is defined by the following parameters, which determine how the input is tiled, how scales are computed, and how the scale output is organized:

| Parameter             | Value | What it means in practice                      |
| --------------------- | ----: | ---------------------------------------------- |
| `rows_per_cta`        |     8 | One CTA (thread block) covers 8 rows at a time   |
| `k_tile`              |   256 | One CTA covers 256 values along K              |
| `threads_per_cta`     |    32 | One warp per CTA                               |
| `SCALE_VEC_SIZE`      |    32 | One shared scale per contiguous 32-value block |
| `SCALE_PACK_COLS`     |     4 | Scale bytes are packed in groups of 4 columns  |
| `SCALE_ROWS_PER_PACK` |   128 | Scale storage is organized in 128-row chunks   |

Each CTA therefore handles an `8 × 256` tile of the input. It loads the tile from global memory in a single bulk TMA (Tensor Memory Accelerator) transfer and processes it with one warp of 32 threads. Each thread handles 8 values, with four threads working together on each 32-value scale block, allowing the warp to process all 8 rows in parallel. The launch grid is two-dimensional, with `ceil(M / 8)` tiles along the rows and `ceil(K / 256)` tiles along K. This splits the work across both dimensions of the matrix. 

<br><br>

## Repository layout

The project is deliberately small: two CLI entry points, one implementation module, and one vendored file.

```
benchmark.py         CLI entry point for benchmarking the quantizer kernel on
                     the local B200: verify against the torch reference, then
                     time the kernel and report effective bandwidth.

e2e.py               CLI entry point for end-to-end verification: quantize A
                     and B, feed the packed FP8 payloads and tcgen05 scale
                     bytes into the vendored block-scaled GEMM (no repacking),
                     validate C against an fp32 reference.

MXFP8_quantizer.py   The main implementation. Everything that does real work
                     lives here: the CuTeDSL kernel, the launch and shape logic
                     (padding, validation, alignment checks, CUDA error
                     surfacing), the torch reference the kernel is verified
                     against, the benchmark harness, and the end-to-end GEMM
                     verification.

blockscaled_gemm.py  The persistent dense block-scaled GEMM example, provided in 
                     the file `dense_blockscaled_gemm_persistent.py`, taken from 
                     NVIDIA's CUTLASS repository. We keep the persistent SM100 
                     block-scaled GEMM kernel and launch path that consume the 
                     quantizer's output (verbatim), and drop the parts of the 
                     example that are not needed here — its CLI, benchmark harness, 
                     tensor factories, and the dense-to-atom scale repacking 
                     helpers. The file retains NVIDIA's BSD-3-Clause license 
                     header; see the top of the file for the full license text.

pyproject.toml       Project metadata and runtime dependencies (CuTeDSL wheel,
                     PyTorch cu128 build, cuda-python, numpy, packaging).
```

<br><br>

## Installation and running

The project has two plain Python CLI entry points, `benchmark.py` and `e2e.py`, that run on the local NVIDIA B200 GPU and print a JSON summary when they complete: `benchmark.py` measures the quantizer's throughput and `e2e.py` verifies the full quantize-to-GEMM pipeline.

Requirements:

- NVIDIA B200 (SM100) GPU with the CUDA 13 toolkit and a matching driver
- Python >= 3.12
- `uv` (recommended) or `pip`

```bash
uv sync
```

With `pip`, install the cu128 PyTorch build first (it is not published to PyPI), then the project:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

### benchmark.py — quantizer benchmark

`benchmark.py` measures how fast the quantizer moves data. It generates a seeded `(M, K)` input, launches the kernel once and verifies the output against the independent torch reference, then times the kernel over warmup and timed launches and reports the effective bandwidth. The reference check is strict: the scale bytes must match the torch reference bit-exactly and the FP8 payload must match to within 2 byte values (ULPs). The default invocation runs the large full-tile configuration (16384 × 16384 bf16, 10 warmup + 50 timed launches, 5 benchmark passes).

```bash
python benchmark.py
```

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--m` | `int` | `16384` | Number of input rows |
| `--k` | `int` | `16384` | Number of input columns |
| `--input-dtype` | `str` | `bf16` | Input dtype, either `bf16` or `fp16` |
| `--warmup-iterations` | `int` | `10` | Number of warmup iterations before timing |
| `--iterations` | `int` | `50` | Number of timed benchmark iterations per run |
| `--benchmark-repeats` | `int` | `5` | Number of repeated benchmark passes used to collect min/mean/std timing |
| `--cold-l2` | `bool` flag | `False` | Benchmarks with a cold L2 cache by cycling through distinct memory workspaces |
| `--seed` | `int` | `0` | Random seed used for input generation |


### e2e.py — end-to-end verification

`e2e.py` verifies the full pipeline: it quantizes `A (M, K)` and `B (N, K)` with the kernel, feeds the packed FP8 payloads and scale bytes into the vendored SM100 block-scaled GEMM, and validates the result `C` against an fp32 reference.

The default single-configuration run (1024 × 1024 × 2048 bf16):

```bash
python e2e.py
python e2e.py --suite # run 6 configs covering both input dtypes,
                      # cluster multicasts, and multiple seeds
```

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--m` | `int` | `1024` | Number of A rows |
| `--n` | `int` | `1024` | Number of B rows |
| `--k` | `int` | `2048` | Contraction dimension (must be a multiple of 256) |
| `--input-dtype` | `str` | `bf16` | Input dtype, either `bf16` or `fp16` |
| `--seed` | `int` | `0` | Random seed for A/B generation |
| `--gemm-tolerance` | `float` | `1e-2` | `assert_close` `rtol = atol` for the GEMM check |
| `--mma-tiler-mn` | `str` | `128,128` | MMA tiler as `"M,N"`, e.g. `256,128` for 2-CTA MMA |
| `--cluster-shape-mn` | `str` | `1,1` | Cluster shape as `"M,N"`, e.g. `2,1` |
| `--suite` | `bool` flag | `False` | Run the full `DEFAULT_E2E_SUITE` instead of a single configuration |

<br><br>

## Results and verification

All of the numbers below come from the two programs in the previous section, run on a B200. They answer two questions: whether the quantizer produces correct output and how fast it moves data. The correctness checks always run first. If any check fails, the run aborts, so a timing is only ever reported for a kernel whose output has been verified.

### End-to-end verification

The most convincing check hands the quantizer's output to the hardware it was designed for. Two random matrices are quantized, and their packed FP8 payloads and scale bytes are passed straight into an unmodified NVIDIA block-scaled GEMM, with no repacking in between. The GEMM reads the scale bytes in the exact layout it expects, so if the quantizer had written them in the wrong arrangement, the GEMM's output would be garbage. That output is compared against a full-precision (fp32) reference computed from the same quantized data.

The table below, produced by `e2e.py`, shows the six configurations that make up the full verification suite. Each row is one test and the columns mean the following:

- **Input**: the data type of the two quantized matrices, `bf16` or `fp16`.
- **GEMM tiling**: how the GEMM splits the output matrix into tiles for its matrix-multiply hardware. 128×128 means one tile covers 128 rows by 128 columns; 256×128 spreads each tile across two cooperating thread blocks.
- **Cluster**: how many thread blocks work on a tile together. 1×1 is a single block; 2×1 is two blocks side by side.
- **Max error**: the largest difference between the GEMM's output and the full-precision reference, anywhere in the result.
- **Pass margin**: how much room the worst element leaves below the allowed error limit. 1× would mean it sits exactly at the limit; 650× means it is 650 times closer to the reference than required.

The suite deliberately varies these settings so a pass means the quantizer's output works for every path through the GEMM.

| M | N | K | Input | GEMM tiling | Cluster | Max error | Pass margin |
| ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| 1024 | 1024 | 2048 | `bf16` | 128×128 | 1×1 | 1.07e-4 | 650× |
| 2048 | 2048 | 2048 | `bf16` | 128×128 | 2×1 | 1.07e-4 | 420× |
| 1024 | 2048 | 4096 | `fp16` | 128×128 | 1×2 | 3.36e-4 | 265× |
| 2048 | 2048 | 4096 | `bf16` | 256×128 | 2×1 | 2.44e-4 | 259× |
| 2048 | 1024 | 2048 | `fp16` | 256×128 | 2×2 | 1.07e-4 | 440× |
| 1024 | 1024 | 2048 | `bf16` | 128×64 | 1×1 | 9.16e-5 | 379× |

Every configuration passes with a margin of roughly 250× or more: the worst element is hundreds of times closer to the reference than the check requires. The check allows each element to be off by up to 1% of its magnitude, plus a small fixed allowance. The errors are around 1e-4 on output values of magnitude ~200, which is the level of rounding you expect from a dot product accumulated in full precision. The error also grows with the contraction depth, as expected: doubling K (2048 → 4096) roughly doubles the worst error (1.07e-4 → 2.44e-4 for bf16). The two rows compared here differ in M and N as well, so this is a trend rather than a controlled K-only experiment.

### Throughput

The table below, produced by `benchmark.py`, shows the quantizer's throughput at different problem sizes, measured on a B200 with correctness verification enabled. Each row is one configuration, and the columns mean the following:

- **Mode**: whether the requested shape lined up exactly with how the kernel processes data (Full tile) or had to be padded internally to a larger size first (Padded fallback).
- **Runtime min / Runtime std**: the best time observed across the repeated benchmark passes, and the spread between them.
- **Effective TB/s**: total bytes moved per second.

| Mode | Input | M | K | Runtime min (us) | Runtime std (us) | Effective TB/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full tile | `bf16` | 16384 | 16384 | 127.229 | 0.099 | 6.395 |
| Full tile | `fp16` | 16384 | 16384 | 128.482 | 0.051 | 6.333 |
| Full tile | `bf16` | 32768 | 16384 | 252.231 | 0.198 | 6.452 |
| Full tile | `bf16` | 32768 | 8192 | 129.148 | 0.044 | 6.300 |
| Full tile, cold L2 | `bf16` | 16384 | 16384 | 128.720 | 0.048 | 6.321 |
| Padded fallback | `bf16` | 16385 | 16384 | 128.389 | 0.054 | 6.338 |
| Padded fallback | `fp16` | 2051 | 2080 | 15.514 | 0.582 | 0.834 |
| Full tile | `bf16` | 2048 | 2048 | 8.487 | 0.600 | 1.498 |

Two patterns stand out. First, the large problems all sustain roughly 6.3–6.5 TB/s, and time scales with bytes moved rather than shape: 16384×16384 and 32768×8192 both move about 814 MB and both take about 127–129 us. That is the signature of a purely memory-bound kernel: it is limited by how fast the GPU can stream data, not by its own work. Second, the small problems (8.5 us at 2048×2048, 15.5 us for the padded 2051×2080) run nowhere near that plateau because launching the kernel and ramping up the memory pipeline dominate when there is little work to do. The 6+ TB/s figures are a genuine bandwidth limit.

The cold-L2 row shows why cache state does not matter at these sizes: the working set (~814 MB) is far larger than the GPU's L2 cache, so the data streams from main memory whether the cache is warm or cold. That is why the cold measurement (6.321) matches the warm one (6.395) to within about 1.2%. Flushing the cache matters only for small problems that could otherwise fit in L2.
