# MXFP8 Quantizer in CuTeDSL for Blackwell

This repository contains a CuTeDSL implementation of a Blackwell-oriented MXFP8 quantizer for FP16 and BF16 inputs. The kernel is designed around Blackwell's block-scaled Tensor Core path and emits scale factors directly in the final `tcgen05` layout expected by downstream block-scaled GEMMs.

The implementation was built as a reconstruction of the ideas described in FAL's [Chasing 6+ TB/s: an MXFP8 quantizer on Blackwell](https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/), but with a full CuTeDSL implementation and repository-local benchmarking harness.

### What the Kernel Produces

Given an input matrix `X` of shape (M, K) in FP16 or BF16, the kernel emits:

- `Q`: an (M, K) FP8 E4M3 payload stored as bytes
- `S`: one UE8M0 scale byte per (1 x 32) block along `K`, emitted directly in the final Blackwell `tcgen05` scale layout

For each row and each contiguous 32-value block along `K`, the kernel:

1. Computes the absolute maximum over the 32-value block
2. Converts that magnitude into a shared power-of-two `UE8M0` scale byte
3. Multiplies the 32 values by the reciprocal scale
4. Converts the scaled values to FP8 `E4M3` with finite saturation
5. Writes the quantized values through packed wide stores
6. Writes the scale byte directly in the final `tcgen05` layout, with no later repacking stage

The logical scale granularity is `1 x 32` along `K`, but the kernel is organized so that this unit also matches the execution structure. Each lane handles 8 half-precision values, 4 lanes cooperate on one 32-value scale block, and one warp therefore processes 8 rows in parallel. This keeps the mapping between scale computation, register work, and output writes very direct.

### Why This Exists

The main motivation is to avoid an extra scale-repacking stage after quantization.

A conventional reference quantizer naturally produces a dense logical scale tensor of shape `(M, K / 32)`. That representation is easy to reason about, but it is not the layout Blackwell's block-scaled Tensor Core path wants to read directly. If scales are written densely first, a later repacking step is required before a block-scaled GEMM can consume them.

This kernel removes that extra step entirely. It does not produce an intermediate dense scale tensor and then transform it afterward. Instead, it emits the final `tcgen05`-compatible scale layout directly as part of quantization, so the output is already in the form the downstream block-scaled consumer expects.

## Kernel Structure

At a high level, the implementation is built around:

- one CTA per `(rows_per_cta, k_tile)` tile
- a single bulk TMA load from HBM to shared memory
- one warp per CTA
- four lanes cooperating on each `1 x 32` scale block
- packed scale stores using `Uint32` words instead of scattered byte writes

The default kernel configuration and the resulting work decomposition are:

| Parameter | Value | What it means in practice |
| --- | ---  |  ---  |
| `rows_per_cta` | 8 | One CTA covers 8 rows at a time |
| `k_tile` | 256 | One CTA covers 256 values along `K` |
| `threads_per_cta` | 32 | One warp per CTA |
| `SCALE_VEC_SIZE` | 32 | One shared scale per contiguous 32-value block |
| `SCALE_PACK_COLS` | 4 | Scale bytes are packed in groups of 4 columns |
| `SCALE_ROWS_PER_PACK` | 128 | Scale storage is organized in 128-row chunks |

With this configuration, each CTA processes 8 rows and 8 scale blocks per row (`256 / 32 = 8`). The launch grid is two-dimensional, with `grid_m = ceil_div(M, rows_per_cta)` and `grid_k = ceil_div(K, k_tile)`, so work is split across both the row dimension and the `K` dimension instead of giving each CTA a long serial walk across all of `K`. In practice this keeps CTAs short-lived and gives the GPU a large pool of independent tiles to schedule.

## Performance

The current implementation sustains a little over `6.4 TB/s` of effective bandwidth on B200 for large full-tile problem sizes.

### Representative Results

| Mode | Input dtype | M | K | Runtime min (us) | Runtime std (us) | Effective TB/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full tile | `fp16` | 16384 | 16384 | 128.482 | 0.051 | 6.333 |
| Full tile | `bf16` | 16384 | 16384 | 127.111 | 0.037 | 6.401 |
| Full tile | `bf16` | 32768 | 16384 | 252.231 | 0.198 | 6.452 |
| Full tile | `bf16` | 32768 | 8192 | 129.148 | 0.044 | 6.300 |
| Padded fallback | `bf16` | 16385 | 16384 | 128.389 | 0.054 | 6.338 |
| Padded fallback | `fp16` | 2051 | 2080 | 15.514 | 0.582 | 0.834 |
| Full tile | `bf16` | 2048 | 2048 | 8.487 | 0.600 | 1.498 |

#### How to Read These Results

- `Mode` indicates whether the requested shape matched the kernel's internal tiling exactly or required internal padding before launch
- `Full tile` means the problem size was already aligned to the main `(rows_per_cta=8, k_tile=256)` execution path
- `Padded fallback` means the benchmarked shape was rounded up internally before execution
- `Runtime min (us)` is the best observed runtime across repeated benchmark passes
- `Runtime std (us)` shows the variability across those repeated passes
- `Effective TB/s` is the headline user-facing throughput computed from the logical problem size originally requested by the user

The smaller `2048 x 2048` and `2051 x 2080` cases were also used as exact host-side correctness-check runs in the benchmark harness.

## How to run

First install Modal:

```bash
uv add modal
```

or

```bash
pip install modal
```

Then authenticate your Modal account:

```bash
modal token set --token-id <your-token-id> --token-secret <your-token-secret>
```

After that, run the code with

```bash
modal run MXFP8-quantizer.py
```

If you want to run the same kernel with different arguments, pass them directly on the command line:

```bash
modal run MXFP8-quantizer.py --m 16384 --k 16384 --input-dtype fp16 --warmup-iterations 10 --iterations 100 --benchmark-repeats 3 --cold-l2
modal run MXFP8-quantizer.py --m 32768 --k 16384 --input-dtype bf16 --iterations 100 --benchmark-repeats 3 --cold-l2
modal run MXFP8-quantizer.py --m 2051 --k 2080 --input-dtype fp16 --iterations 100 --benchmark-repeats 3 --cold-l2
```

### Available Arguments

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--m` | `int` | `16384` | Number of input rows |
| `--k` | `int` | `16384` | Number of input columns |
| `--input-dtype` | `str` | `bf16` | Input dtype, either `bf16` or `fp16` |
| `--warmup-iterations` | `int` | `10` | Number of warmup iterations before timing |
| `--iterations` | `int` | `50` | Number of timed benchmark iterations per run |
| `--benchmark-repeats` | `int` | `5` | Number of repeated benchmark passes used to collect min/mean/std timing |
| `--cold-l2` | `bool` flag | `False` | Enables the cold-L2-aware benchmark path in the harness |
| `--seed` | `int` | `0` | Random seed used for input generation |
