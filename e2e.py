"""End-to-end verification: MXFP8 quantizer -> vendored SM100 blockscaled GEMM.

This entrypoint proves the quantizer's central claim end to end: that the
scale bytes it emits are written directly in the tcgen05 atom layout
(``BlockScaledBasicChunk``) that Blackwell's block-scaled Tensor Core path
expects, with no repacking stage.

It quantizes A (M, K) and B (N, K) with the CuTeDSL kernel, feeds the packed
FP8 payloads and scale buffers directly into the vendored SM100 persistent
block-scaled GEMM as SFA/SFB pointers, and validates the GEMM output C
against an fp32 reference ``C_ref = (Q_a * S_a) @ (Q_b * S_b)^T``. If the
scale layout or payload layout ever diverged from what the hardware path
expects, C would be garbage and the check would fail.

Single configuration (default 1024 x 1024 x 2048, bf16):

    python e2e.py

Full verification suite (shapes, tilers, clusters, dtypes, and seeds; see
DEFAULT_E2E_SUITE in MXFP8_quantizer.py):

    python e2e.py --suite

Compared with benchmark.py: benchmark.py measures the quantizer's sustained
effective bandwidth and verifies its outputs against the torch reference, in
isolation. e2e.py does not measure performance; it verifies correctness of
the complete quantize -> GEMM pipeline, which is what proves the emitted
scale layout is directly consumable by the downstream hardware. Run
benchmark.py for the performance number and e2e.py for the end-to-end
correctness proof.
"""

from __future__ import annotations
import argparse
import json

from MXFP8_quantizer import run_e2e_suite, run_e2e_verification

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end MXFP8 quantizer + blockscaled GEMM verification on a local B200."
    )
    parser.add_argument("--m", type=int, default=1024, help="A rows (default: 1024)")
    parser.add_argument("--n", type=int, default=1024, help="B rows (default: 1024)")
    parser.add_argument("--k", type=int, default=2048, help="Contraction dimension (default: 2048)")
    parser.add_argument(
        "--input-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Input dtype (default: bf16)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--gemm-tolerance",
        type=float,
        default=1.0e-2,
        help="assert_close rtol=atol for the GEMM check (default: 1e-2)",
    )
    parser.add_argument(
        "--mma-tiler-mn",
        type=str,
        default="128,128",
        help="MMA tiler as 'M,N', e.g. 256,128 for 2-CTA MMA (default: 128,128)",
    )
    parser.add_argument(
        "--cluster-shape-mn",
        type=str,
        default="1,1",
        help="Cluster shape as 'M,N', e.g. 2,1 (default: 1,1)",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run the full verification suite (DEFAULT_E2E_SUITE) instead of a single configuration",
    )
    args = parser.parse_args()

    if args.suite:
        result = run_e2e_suite(gemm_tolerance=args.gemm_tolerance)
    else:
        mma_tiler = tuple(int(x) for x in args.mma_tiler_mn.split(","))
        cluster_shape = tuple(int(x) for x in args.cluster_shape_mn.split(","))
        result = run_e2e_verification(
            m=args.m,
            n=args.n,
            k=args.k,
            input_dtype=args.input_dtype,
            seed=args.seed,
            gemm_tolerance=args.gemm_tolerance,
            mma_tiler_mn=mma_tiler,
            cluster_shape_mn=cluster_shape,
        )
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
