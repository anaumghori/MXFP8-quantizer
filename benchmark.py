"""Benchmark the MXFP8 quantizer kernel on a local B200.

This entrypoint measures the quantizer in isolation. It runs the CuTeDSL
kernel on an (M, K) matrix of bf16/fp16 data, verifies the kernel's output
against an independent torch reference that reproduces the quantization math
in fp32, and then times the kernel to report the sustained effective
bandwidth (input read + FP8 payload + scale bytes, divided by runtime).

Verification always runs before any timing is collected: the scale bytes must
match the reference bit-exactly and the FP8 payload must match within 2 ULPs.
A failed check raises an error instead of producing timing numbers, so a
broken kernel can never report a benchmark result.

The default invocation (no arguments) benchmarks the large full-tile
configuration (16384 x 16384, bf16, 10 warmup + 50 timed launches, 5 passes):

    python benchmark.py

See README.md for the full argument reference and worked examples.

Compared with e2e.py: benchmark.py measures how fast the quantizer runs and
confirms its outputs are numerically correct, in isolation. e2e.py proves
correctness of the complete pipeline: it feeds the quantizer's packed FP8
payloads and tcgen05 scale bytes into the vendored SM100 block-scaled GEMM
with no repacking and validates the GEMM output against an fp32 reference.
Run benchmark.py for the performance number; run e2e.py for the end-to-end
correctness proof that the emitted scale layout is directly consumable by the
downstream hardware path.
"""

from __future__ import annotations

import argparse
import json

from MXFP8_quantizer import DEFAULT_K_TILE, DEFAULT_ROWS_PER_CTA, run_quantizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the MXFP8 quantizer kernel on a local B200."
    )
    parser.add_argument("--m", type=int, default=16384, help="Number of input rows (default: 16384)")
    parser.add_argument("--k", type=int, default=16384, help="Number of input columns (default: 16384)")
    parser.add_argument(
        "--input-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Input dtype (default: bf16)",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Warmup kernel launches before timing (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Timed kernel launches per benchmark pass (default: 50)",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=5,
        help="Repeated benchmark passes for min/mean/std timing (default: 5)",
    )
    parser.add_argument(
        "--cold-l2",
        action="store_true",
        help="Benchmark with a cold L2 cache by cycling through distinct memory workspaces",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for input generation (default: 0)",
    )
    args = parser.parse_args()

    result = run_quantizer(
        m=args.m,
        k=args.k,
        rows_per_cta=DEFAULT_ROWS_PER_CTA,
        k_tile=DEFAULT_K_TILE,
        input_dtype=args.input_dtype,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        benchmark_repeats=args.benchmark_repeats,
        cold_l2=args.cold_l2,
        skip_verification=False,
        seed=args.seed,
    )

    summary = {
        "launch_success": result["launch_success"],
        "verification_ran": result["verification_ran"],
        "verification_passed": result["verification_passed"],
        "q_max_byte_diff": result["q_max_byte_diff"],
        "q_exact_fraction": result["q_exact_fraction"],
        "scale_bytes_exact": result["scale_bytes_exact"],
        "dispatch_mode": result["dispatch_mode"],
        "runtime_min_s": result["runtime_min_s"],
        "runtime_min_us": result["runtime_min_us"],
        "runtime_mean_s": result["runtime_mean_s"],
        "runtime_mean_us": result["runtime_mean_us"],
        "runtime_std_us": result["runtime_std_us"],
        "effective_gb_s": result["effective_gb_s"],
        "effective_gb_s_mean": result["effective_gb_s_mean"],
        "effective_tb_s": result["effective_tb_s"],
        "logical_bytes_moved": result["logical_bytes_moved"],
        "physical_bytes_moved": result["physical_bytes_moved"],
        "input_dtype": result["input_dtype"],
        "rows_per_cta": result["rows_per_cta"],
        "k_tile": result["k_tile"],
        "grid_m": result["grid_m"],
        "grid_k": result["grid_k"],
        "total_ctas": result["total_ctas"],
        "benchmark_repeats": result["benchmark_repeats"],
        "workspace_count": result["workspace_count"],
        "cold_l2": result["cold_l2"],
        "is_full_tile": result["is_full_tile"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
