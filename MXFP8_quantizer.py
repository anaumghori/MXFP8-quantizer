"""MXFP8 quantizer implementation for Blackwell (SM100): the CuTeDSL kernel,
torch-reference verification, boundary robustness, and the end-to-end
verification against the vendored SM100 blockscaled GEMM consumer.

Public API:

- :func:`run_quantizer`  - quantize + verify + benchmark one (m, k) problem
- :func:`run_e2e_verification` - quantize A/B and verify the vendored GEMM output
  (one (m, n, k, dtype, seed, mma_tiler, cluster) configuration)
- :func:`run_e2e_suite` - run the e2e verification across a suite of shapes,
  tilers, clusters, dtypes, and seeds; all must pass
- :func:`prepare_and_quantize` - lower-level: quantize one matrix, no benchmark
"""

import dataclasses
import statistics
from typing import Any

# Default tiling used by the quantizer kernel: one CTA per (rows_per_cta, k_tile) tile, one warp per CTA
DEFAULT_ROWS_PER_CTA = 8
DEFAULT_K_TILE = 256

DEFAULT_E2E_SUITE: tuple[
    tuple[int, int, int, str, int, tuple[int, int], tuple[int, int]], ...
] = (
    # Baseline config (same as the original single-run default).
    (1024, 1024, 2048, "bf16", 0, (128, 128), (1, 1)),
    # Larger M/N with an M-direction cluster multicast.
    (2048, 2048, 2048, "bf16", 1, (128, 128), (2, 1)),
    # Deeper K and fp16 input with an N-direction cluster multicast.
    (1024, 2048, 4096, "fp16", 2, (128, 128), (1, 2)),
    # 2-CTA MMA (M tiler 256) with an M cluster.
    (2048, 2048, 4096, "bf16", 0, (256, 128), (2, 1)),
    # 2-CTA MMA with a 2x2 cluster.
    (2048, 1024, 2048, "fp16", 1, (256, 128), (2, 2)),
    # Narrow N tiler (64) to exercise the N=64 epilogue path.
    (1024, 1024, 2048, "bf16", 3, (128, 64), (1, 1)),
)


def build_runtime() -> dict[str, Any]:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    import torch
    from cutlass import BFloat16, Float16, Float32, Int32, Uint8, Uint16, Uint32, Uint64
    try:
        from cutlass import testing
    except ImportError:
        from cutlass.cute import testing
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.nvgpu import cpasync
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op

    FP8_E4M3_MAX = 448.0
    # Maximum allowed |Q_byte - ref_byte| (in byte/ULP units) for the fp8 payload check.
    # The kernel and the torch reference may round fp32->fp8 with different modes,
    # which differs by at most 1 ULP; 2 leaves slack for saturation-boundary ties.
    Q_BYTE_DIFF_TOLERANCE = 2
    SCALE_VEC_SIZE = 32
    SCALE_PACK_COLS = 4
    SCALE_ROWS_PER_PACK = 128
    _compile_cache: dict[tuple[Any, ...], Any] = {}

    @dsl_user_op
    def bitcast_f32_to_u32(x: Float32, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),  # This inline assembly returns a 32-bit integer
                # x is a CuTeDSL proxy object for a future device-side SSA value
                # x.ir_value(...) unwraps that proxy into the underlying compiler IR value
                # That IR value is passed as the first input operand ($1) to the inline assembly
                [x.ir_value(loc=loc, ip=ip)],
                "mov.b32 $0, $1;",  # Move 32 bits from operand 1 into operand 0
                # Tell LLVM that the output goes in a general register (r)
                # and the input comes from a floating-point register (f)
                "=r,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,  # Interpret the assembly text using this syntax convention
            )
        )

    @dsl_user_op
    def bitcast_u32_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [x.ir_value(loc=loc, ip=ip)],
                "mov.b32 $0, $1;",
                "=f,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def bitcast_bf16_to_u16(x: BFloat16, *, loc=None, ip=None) -> Uint16:
        return Uint16(llvm.bitcast(T.i16(), x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))

    @dsl_user_op
    def bitcast_u16_to_bf16(x: Uint16, *, loc=None, ip=None) -> BFloat16:
        return BFloat16(llvm.bitcast(T.bf16(), x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))

    @dsl_user_op
    def bitcast_half_to_u16(x: Float16, *, loc=None, ip=None) -> Uint16:
        return Uint16(llvm.bitcast(T.i16(), x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))

    @dsl_user_op
    def bitcast_u16_to_half(x: Uint16, *, loc=None, ip=None) -> Float16:
        return Float16(llvm.bitcast(T.f16(), x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))

    # Take two Float32 values, convert them to FP8 E4M3, and pack them into one 16-bit result
    @dsl_user_op
    def pack_e4m3x2(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Uint16:
        return Uint16(
            llvm.inline_asm(
                T.i16(),
                [x0.ir_value(loc=loc, ip=ip), x1.ir_value(loc=loc, ip=ip)],

                # rn means round to nearest and e4m3x2 means produce two packed E4M3 FP8 values
                # satfinite clamps finite values into the representable FP8 range
                # $0, $2, $1 mean output operand, first input operand, second input operand
                # $1 corresponds to x0, $2 corresponds to x1
                "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",

                # Output is one 16-bit register because two FP8 values take 16 bits total
                # Inputs are both Float32 register operands
                "=h,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def fma_rn_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [
                    a.ir_value(loc=loc, ip=ip),
                    b.ir_value(loc=loc, ip=ip),
                    c.ir_value(loc=loc, ip=ip),
                ],
                "fma.rn.f32 $0, $1, $2, $3;",  # fma means fused multiply-add
                "=f,f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    def threadquad_reduce_umax(val: Uint16) -> Uint16:
        val = cutlass.max(
            val,
            # Exchange register values between warp lanes using a butterfly shuffle pattern
            # offset=2 pairs lanes like 0<->2 and 1<->3 for the quad reduction
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = cutlass.max(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    # Answer the question: what power-of-two scale should this block use
    @cute.jit
    def ceil_pow2_scale_from_amax(amax: Float32):
        eps = Float32(1.0e-12)
        inv_scale_f32 = Float32(1.0)
        scale_u8 = Uint8(127)

        if amax != Float32(0.0):
            scaled = fma_rn_f32(amax, Float32(1.0 / FP8_E4M3_MAX), eps)
            bits = bitcast_f32_to_u32(scaled)

            # The 32 bits look like
            # bit 31 = sign, bits 30..23 = exponent, bits 22..0 = mantissa

            # Shift right by 23 because the exponent lives in bits 30..23
            # That moves the exponent field down into the low 8 bits
            # [ sign ][ exponent ][ mantissa ] -> 00000000 [ sign ][ exponent ]
            # Then `& 0xFF` keeps only the lowest 8 bits
            exponent = (bits >> 23) & Uint32(0xFF)

            # This mask keeps the lower 23 bits and removes everything above them
            mantissa = bits & Uint32(0x7FFFFF)

            if mantissa != Uint32(0):
                exponent = exponent + Uint32(1)

            scale_u8 = Uint8(exponent)
            inv_scale_bits = (Uint32(254) - exponent) << 23
            inv_scale_f32 = bitcast_u32_to_f32(inv_scale_bits)

        return inv_scale_f32, scale_u8

    @cute.jit
    def pack_four_fp8_values(q0: Float32, q1: Float32, q2: Float32, q3: Float32):
        # 32-bit result: [ hi 16 bits ][ lo 16 bits ]
        lo = pack_e4m3x2(q0, q1)
        hi = pack_e4m3x2(q2, q3)
        # Shift the upper half left by 16 and combine it with the lower half
        return (Uint32(hi) << 16) | Uint32(lo)

    @cute.jit
    def pack_eight_fp8_values(
        q0: Float32,
        q1: Float32,
        q2: Float32,
        q3: Float32,
        q4: Float32,
        q5: Float32,
        q6: Float32,
        q7: Float32,
    ):
        lo = pack_four_fp8_values(q0, q1, q2, q3)
        hi = pack_four_fp8_values(q4, q5, q6, q7)
        return (Uint64(hi) << 32) | Uint64(lo)

    @cute.jit
    def pack_scale_word(b0: Uint8, b1: Uint8, b2: Uint8, b3: Uint8):
        # The final 32-bit layout becomes [b3][b2][b1][b0]
        return (
            Uint32(b0)
            | (Uint32(b1) << 8)
            | (Uint32(b2) << 16)
            | (Uint32(b3) << 24)
        )

    @dataclasses.dataclass(frozen=True)
    class MXFP8QuantizerConfig:
        input_dtype: type[cutlass.Numeric] = cutlass.BFloat16
        rows_per_cta: int = 8
        k_tile: int = 256
        smem_alignment_bytes: int = 1024

        @property
        def values_per_lane(self) -> int:
            return 8

        @property
        def lanes_per_scale_block(self) -> int:
            return 4

        @property
        def rows_per_warp(self) -> int:
            return 8

        @property
        def threads_per_cta(self) -> int:
            return 32

        @property
        def blocks_per_tile(self) -> int:
            return self.k_tile // SCALE_VEC_SIZE

    class BlackwellMXFP8Quantizer:
        def __init__(self, config: MXFP8QuantizerConfig | None = None):
            self.cfg = config if config is not None else MXFP8QuantizerConfig()

        @cute.jit
        def __call__(
            self,
            x_ptr: cute.Pointer,
            q_dwords_ptr: cute.Pointer,  # points to the packed FP8 output buffer
            s_words_ptr: cute.Pointer,  # points to the packed scale buffer
            m: Int32,
            k: Int32,
            s_word_count: Int32,
            stream: cuda.CUstream,
        ):
            cfg = self.cfg

            if cutlass.const_expr(cfg.k_tile % SCALE_VEC_SIZE != 0):
                raise ValueError("k_tile must be divisible by 32")
            if cutlass.const_expr(cfg.rows_per_cta != 8):
                raise ValueError("blog-faithful TMA kernel currently requires rows_per_cta=8")
            if cutlass.const_expr(cfg.blocks_per_tile % SCALE_PACK_COLS != 0):
                raise ValueError("blocks_per_tile must be divisible by 4")
            if cutlass.const_expr(
                cfg.input_dtype != cutlass.BFloat16 and cfg.input_dtype != cutlass.Float16
            ):
                raise ValueError("optimized CuTe kernel only supports bf16/fp16 inputs")

            # View the input pointer as a row-major (m, k) tensor in global memory
            input_layout = cute.make_layout(
                (m, cute.assume(k, SCALE_VEC_SIZE)),
                stride=(cute.assume(k, SCALE_VEC_SIZE), 1),
            )

            # Each element in this buffer is one Uint64 holding 8 FP8 values
            packed_fp8_word_count = (m * k) // 8
            packed_fp8_word_layout = cute.make_layout((packed_fp8_word_count,), stride=(1,))

            # Each element in this buffer is one Uint32 holding 4 scale bytes
            packed_scale_word_layout = cute.make_layout((s_word_count,), stride=(1,))

            # make_tensor combines a base pointer with a layout so the kernel can index memory logically
            input_tensor = cute.make_tensor(x_ptr, input_layout)
            packed_fp8_output_tensor = cute.make_tensor(q_dwords_ptr, packed_fp8_word_layout)
            packed_scale_output_tensor = cute.make_tensor(s_words_ptr, packed_scale_word_layout)

            # This is the shared-memory view of one CTA tile
            # With the default config it is an 8 x 256 row-major tile
            shared_tile_layout = cute.make_layout(
                (cfg.rows_per_cta, cfg.k_tile),
                stride=(cfg.k_tile, 1),
            )

            # product_each normalizes the logical tile extent that the TMA copy should move per CTA
            cta_tile_shape = cute.product_each(shared_tile_layout.shape)

            # Build the TMA load description for copying one CTA tile from global memory into shared memory
            input_tma_atom, input_tma_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                input_tensor,  # input_tensor tells TMA what the source tensor is
                shared_tile_layout,  # shared_tile_layout tells what the destination tile should look like in shared memory
                cta_tile_shape,  # cta_tile_shape tells it the extent of the tile that one CTA is responsible for
            )

            @cute.struct
            class SharedStorage:
                # One shared-memory barrier slot used to track completion of the TMA load
                barrier_storage: cute.struct.MemRange[cutlass.Int64, 1]

                # cosize(shared_tile_layout) is the number of input elements needed to back the full shared tile
                # Align the tile buffer so the shared-memory region starts on a TMA-friendly boundary
                shared_input_tile_storage: cute.struct.Align[
                    cute.struct.MemRange[cfg.input_dtype, cute.cosize(shared_tile_layout)],
                    cfg.smem_alignment_bytes,
                ]

            self.shared_storage = SharedStorage
            # Total bytes transferred by one TMA load into the shared tile
            self.input_tma_load_bytes = cute.size_in_bytes(cfg.input_dtype, shared_tile_layout)

            self.kernel(
                input_tma_atom,
                input_tma_tensor,
                packed_fp8_output_tensor,
                packed_scale_output_tensor,
                shared_tile_layout,
                k,
            ).launch(
                grid=[cute.ceil_div(m, cfg.rows_per_cta), cute.ceil_div(k, cfg.k_tile), 1],
                block=[cfg.threads_per_cta, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            input_tma_atom: cute.CopyAtom,
            input_tma_tensor: cute.Tensor,
            packed_fp8_output_tensor: cute.Tensor,
            packed_scale_output_tensor: cute.Tensor,
            shared_tile_layout: cute.Layout,
            k: Int32,
        ):
            cfg = self.cfg
            thread_idx, _, _ = cute.arch.thread_idx()
            cta_row_tile_idx, cta_k_tile_idx, _ = cute.arch.block_idx()
            shared_memory_allocator = utils.SmemAllocator()
            shared_storage = shared_memory_allocator.allocate(self.shared_storage)

            # Get the shared-memory barrier address from the structured storage block
            tma_barrier_ptr = shared_storage.barrier_storage.data_ptr()
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tma_barrier_ptr, 1)
                cute.arch.mbarrier_expect_tx(tma_barrier_ptr, self.input_tma_load_bytes)
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()

            warp_id = thread_idx // 32
            lane_id = thread_idx % 32
            row_within_cta = warp_id * cfg.rows_per_warp + lane_id // cfg.lanes_per_scale_block
            lane_within_quad = lane_id % cfg.lanes_per_scale_block
            global_row_idx = cta_row_tile_idx * cfg.rows_per_cta + row_within_cta
            tile_k_start = cta_k_tile_idx * cfg.k_tile

            # Turn the raw shared-memory range into a tensor view using the CTA tile layout
            shared_input_tile = shared_storage.shared_input_tile_storage.get_tensor(shared_tile_layout)
            shared_to_register_copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cfg.input_dtype,
                num_bits_per_copy=128,
            )

            # This thread-value mapping says each row uses 4 lanes and each lane pulls 8 values
            shared_to_register_thread_layout = cute.make_layout(
                (cfg.rows_per_cta, cfg.lanes_per_scale_block),
                stride=(cfg.lanes_per_scale_block, 1),
            )
            shared_to_register_value_layout = cute.make_layout((1, cfg.values_per_lane))
            shared_to_register_tiled_copy = cute.make_tiled_copy_tv(
                shared_to_register_copy_atom,
                shared_to_register_thread_layout,
                shared_to_register_value_layout,
            )
            thread_shared_to_register_copy = shared_to_register_tiled_copy.get_slice(thread_idx)
            tiled_input_tma_tensor = cute.local_tile(
                input_tma_tensor,
                (cfg.rows_per_cta, cfg.k_tile),
                (None, None),
            )

            # Partition the TMA source and destination views so this CTA can issue one tile load
            tma_shared_tile_view, tma_global_tile_view = cpasync.tma_partition(
                input_tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(shared_input_tile, 0, 2),
                cute.group_modes(tiled_input_tma_tensor, 0, 2),
            )
            cta_global_input_tile = tma_global_tile_view[(None, cta_row_tile_idx, cta_k_tile_idx)]
            cute.copy(
                input_tma_atom,
                cta_global_input_tile,
                tma_shared_tile_view,
                tma_bar_ptr=tma_barrier_ptr,
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(tma_barrier_ptr)
            cute.arch.mbarrier_wait(tma_barrier_ptr, 0)

            scale_bytes_per_tile = cute.make_rmem_tensor(
                cute.make_layout((cfg.blocks_per_tile,)),
                Uint8,
            )

            for scale_block_idx in cutlass.range_constexpr(cfg.blocks_per_tile):
                scale_block_k_start = tile_k_start + scale_block_idx * SCALE_VEC_SIZE
                lane_local_absmax_bits = Uint16(0)
                lane_input_values_f32 = cute.make_rmem_tensor(
                    cute.make_layout((cfg.values_per_lane,)),
                    Float32,
                )
                shared_scale_block_tile = cute.local_tile(
                    shared_input_tile,
                    (cfg.rows_per_cta, SCALE_VEC_SIZE),
                    (0, scale_block_idx),
                )
                thread_shared_block_view = thread_shared_to_register_copy.partition_S(shared_scale_block_tile)
                thread_register_block_fragment = cute.make_fragment_like(thread_shared_block_view)
                cute.copy(
                    shared_to_register_tiled_copy,
                    thread_shared_block_view,
                    thread_register_block_fragment,
                )
                lane_input_values = thread_register_block_fragment.load()
                for value_idx in cutlass.range_constexpr(cfg.values_per_lane):
                    lane_input_values_f32[value_idx] = lane_input_values[value_idx].to(Float32)
                    if cutlass.const_expr(cfg.input_dtype == cutlass.BFloat16):
                        abs_value_bits = bitcast_bf16_to_u16(lane_input_values[value_idx]) & Uint16(0x7FFF)
                    else:
                        abs_value_bits = bitcast_half_to_u16(lane_input_values[value_idx]) & Uint16(0x7FFF)
                    lane_local_absmax_bits = cutlass.max(lane_local_absmax_bits, abs_value_bits)

                quad_absmax_bits = threadquad_reduce_umax(lane_local_absmax_bits)
                if cutlass.const_expr(cfg.input_dtype == cutlass.BFloat16):
                    quad_absmax = bitcast_u16_to_bf16(quad_absmax_bits).to(Float32)
                else:
                    quad_absmax = bitcast_u16_to_half(quad_absmax_bits).to(Float32)

                block_inverse_scale, block_scale_byte = ceil_pow2_scale_from_amax(quad_absmax)
                if lane_within_quad == 0:
                    scale_bytes_per_tile[scale_block_idx] = block_scale_byte
                lane_scaled_values_f32 = cute.make_rmem_tensor(
                    cute.make_layout((cfg.values_per_lane,)),
                    Float32,
                )
                for pair_start_idx in cutlass.range_constexpr(0, cfg.values_per_lane, 2):
                    scaled_value_pair = cute.arch.mul_packed_f32x2(
                        (
                            lane_input_values_f32[pair_start_idx],
                            lane_input_values_f32[pair_start_idx + 1],
                        ),
                        (block_inverse_scale, block_inverse_scale),
                    )
                    lane_scaled_values_f32[pair_start_idx] = scaled_value_pair[0]
                    lane_scaled_values_f32[pair_start_idx + 1] = scaled_value_pair[1]
                packed_fp8_word_idx = (
                    (global_row_idx * (k // 8))
                    + (scale_block_k_start // 8)
                    + lane_within_quad
                )
                packed_fp8_output_tensor[packed_fp8_word_idx] = pack_eight_fp8_values(
                    lane_scaled_values_f32[0],
                    lane_scaled_values_f32[1],
                    lane_scaled_values_f32[2],
                    lane_scaled_values_f32[3],
                    lane_scaled_values_f32[4],
                    lane_scaled_values_f32[5],
                    lane_scaled_values_f32[6],
                    lane_scaled_values_f32[7],
                )
            if lane_within_quad == 0:
                total_scale_columns = k // SCALE_VEC_SIZE
                packed_scale_column_block_count = cute.ceil_div(total_scale_columns, SCALE_PACK_COLS)
                scale_row_block_idx = global_row_idx // SCALE_ROWS_PER_PACK
                row_within_scale_block = global_row_idx % SCALE_ROWS_PER_PACK
                row_group_idx = row_within_scale_block // 32
                row_position_in_group = row_within_scale_block % 32
                tile_scale_col_start = tile_k_start // SCALE_VEC_SIZE
                for scale_pack_idx in cutlass.range_constexpr(cfg.blocks_per_tile // SCALE_PACK_COLS):
                    packed_scale_col = tile_scale_col_start + scale_pack_idx * SCALE_PACK_COLS
                    packed_scale_byte_offset = (
                        (
                            (scale_row_block_idx * packed_scale_column_block_count)
                            + (packed_scale_col // SCALE_PACK_COLS)
                        )
                        * 512
                    ) + (row_position_in_group * 16) + (row_group_idx * 4)
                    packed_scale_word = pack_scale_word(
                        scale_bytes_per_tile[scale_pack_idx * SCALE_PACK_COLS + 0],
                        scale_bytes_per_tile[scale_pack_idx * SCALE_PACK_COLS + 1],
                        scale_bytes_per_tile[scale_pack_idx * SCALE_PACK_COLS + 2],
                        scale_bytes_per_tile[scale_pack_idx * SCALE_PACK_COLS + 3],
                    )
                    packed_scale_output_tensor[packed_scale_byte_offset // 4] = packed_scale_word

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b  # Integer division rounded up so partial blocks still count as one full block

    # computes how many bytes are needed to store all scales in the packed format for an m x k matrix
    def packed_scale_storage_bytes(m: int, k: int, scale_vec_size: int) -> int:
        # One scale byte covers scale_vec_size values along K, so this is the logical scale-matrix width
        scale_cols = ceil_div(k, scale_vec_size)
        # The packed format stores rows in groups of 128, so round rows up to that boundary
        padded_rows = ceil_div(m, SCALE_ROWS_PER_PACK) * SCALE_ROWS_PER_PACK
        # The packed format stores scale columns in groups of 4, so round columns up to that boundary
        padded_cols = ceil_div(scale_cols, SCALE_PACK_COLS) * SCALE_PACK_COLS
        # Each logical scale entry is one byte, so rows times columns gives total storage bytes
        return padded_rows * padded_cols

    def packed_scale_word_count(m: int, k: int, scale_vec_size: int) -> int:
        return packed_scale_storage_bytes(m, k, scale_vec_size) // 4  # Four scale bytes are packed into one Uint32 word

    def _check_buffer_alignment(name: str, tensor, required_bytes: int) -> None:
        # TMA loads and wide packed stores require specific address alignment; a
        # misaligned torch view/slice would silently corrupt results, so fail loudly.
        if tensor.data_ptr() % required_bytes != 0:
            raise ValueError(
                f"{name} buffer must be {required_bytes}-byte aligned for "
                f"TMA/packed access, got data_ptr={tensor.data_ptr()} "
                f"(misaligned by {tensor.data_ptr() % required_bytes} bytes)"
            )

    def materialize_quantizer_buffers(
        m: int,
        k: int,
        torch_input_dtype,
        cutlass_input_dtype,
        x_source,
    ):
        x = torch.zeros((m, k), dtype=torch_input_dtype, device="cuda")
        x[: x_source.shape[0], : x_source.shape[1]] = x_source
        quantized_fp8 = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda")
        quantized_fp8_bytes = quantized_fp8.view(torch.int8)
        packed_fp8_words = quantized_fp8_bytes.view(torch.uint64)
        packed_scale_word_count_value = packed_scale_word_count(m, k, SCALE_VEC_SIZE)
        packed_scale_words = torch.zeros(
            (packed_scale_word_count_value,),
            dtype=torch.uint32,
            device="cuda",
        )
        packed_scale_bytes = packed_scale_words.view(torch.uint8)

        # Boundary robustness: validate the alignment assumptions the kernel and
        # TMA rely on before they are baked into the compiled kernel's pointers.
        _check_buffer_alignment("x", x, 16)  # TMA global loads
        _check_buffer_alignment("q", quantized_fp8, 8)  # packed Uint64 stores
        _check_buffer_alignment("s", packed_scale_words, 4)  # packed Uint32 stores

        x_ptr = make_ptr(
            cutlass_input_dtype,
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        q_dwords_ptr = make_ptr(
            cutlass.Uint64,
            packed_fp8_words.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=8,
        )
        s_words_ptr = make_ptr(
            cutlass.Uint32,
            packed_scale_words.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        )

        return {
            "x": x,
            "q_fp8": quantized_fp8,
            "q_bytes": quantized_fp8_bytes,
            "s_storage_bytes": packed_scale_bytes,
            "s_storage_word_count": packed_scale_word_count_value,
            "x_ptr": x_ptr,
            "q_dwords_ptr": q_dwords_ptr,
            "s_words_ptr": s_words_ptr,
        }

    def reference_quantize(x_padded):
        # Reproduce the kernel's quantization math in torch on the zero-padded
        # input, so blocks that straddle the requested M/K boundary see exactly
        # the same values the kernel reads.
        m, k = x_padded.shape
        x_f32 = x_padded.to(torch.float32)
        blocks = x_f32.view(m, k // SCALE_VEC_SIZE, SCALE_VEC_SIZE)
        amax = blocks.abs().amax(dim=2)  # (m, k // SCALE_VEC_SIZE)
        device = x_padded.device
        inv_448 = torch.tensor(1.0 / FP8_E4M3_MAX, dtype=torch.float32, device=device)
        eps = torch.tensor(1.0e-12, dtype=torch.float32, device=device)
        scaled = (
            amax.to(torch.float64) * inv_448.to(torch.float64)
            + eps.to(torch.float64)
        ).to(torch.float32)
        bits = scaled.view(torch.int32)
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF
        scale_bytes = exponent + (mantissa != 0).to(torch.int32)
        # The kernel special-cases all-zero blocks to byte 127 (scale = 1.0).
        scale_bytes = torch.where(
            amax == 0,
            torch.full_like(scale_bytes, 127),
            scale_bytes,
        )

        # The reciprocal scale is an exact power of two, so the fp32 product
        # x * (1/scale) is exact; the only rounding is the final fp8 conversion.
        inv_scale = torch.exp2((127 - scale_bytes).to(torch.float32))
        inv_scale_expanded = inv_scale.repeat_interleave(SCALE_VEC_SIZE, dim=1)
        q_fp8 = (x_f32 * inv_scale_expanded).to(torch.float8_e4m3fn)
        return scale_bytes, q_fp8

    def decode_scale_bytes(s_bytes, m, k):
        # Invert the kernel's tcgen05 scale-word write layout back into a dense
        # (m, k // SCALE_VEC_SIZE) matrix of UE8M0 bytes. The layout is the
        # inverse of the kernel's byte-offset formula, so a correct kernel must
        # reproduce the reference scales bit-exactly after this decode.
        scale_cols = ceil_div(k, SCALE_VEC_SIZE)
        packed_col_blocks = ceil_div(scale_cols, SCALE_PACK_COLS)
        rows = torch.arange(m, device=s_bytes.device)[:, None]
        cols = torch.arange(scale_cols, device=s_bytes.device)[None, :]
        byte_offsets = (
            (
                (rows // SCALE_ROWS_PER_PACK) * packed_col_blocks
                + cols // SCALE_PACK_COLS
            )
            * (SCALE_ROWS_PER_PACK * SCALE_PACK_COLS)
            + ((rows % SCALE_ROWS_PER_PACK) % 32) * 16
            + ((rows % SCALE_ROWS_PER_PACK) // 32) * 4
            + cols % SCALE_PACK_COLS
        )
        return s_bytes[byte_offsets]

    def verify_quantizer_outputs(buffers, kernel_m, kernel_k, real_m, real_k):
        # Compare the kernel's q/s outputs against the independent torch
        # reference. Scale bytes must match bit-exactly; the fp8 payload may
        # differ by at most Q_BYTE_DIFF_TOLERANCE byte values (ULPs) to absorb
        # fp32->fp8 rounding-mode differences in the reference conversion.
        #
        # Boundary robustness: only the real (non-padded) region is compared.
        # Padded rows/columns carry no semantic meaning and are never consumed;
        # the padding is semantics-preserving for the real region by construction
        # (zero-padding cannot change an absmax).

        scale_bytes_ref, q_fp8_ref = reference_quantize(buffers["x"])
        q_bytes = buffers["q_bytes"].view(torch.uint8).to(torch.int32)
        q_ref_bytes = q_fp8_ref.view(torch.uint8).to(torch.int32)
        q_diff = (q_bytes - q_ref_bytes).abs()[:real_m, :real_k]
        q_max_byte_diff = int(q_diff.max().item())
        q_exact_fraction = float((q_diff == 0).float().mean().item())

        dense_scales = decode_scale_bytes(buffers["s_storage_bytes"], kernel_m, kernel_k)
        real_scale_cols = ceil_div(real_k, SCALE_VEC_SIZE)
        scale_bytes_exact = bool(
            (
                dense_scales[:real_m, :real_scale_cols]
                == scale_bytes_ref[:real_m, :real_scale_cols]
            ).all().item()
        )

        verification_passed = scale_bytes_exact and q_max_byte_diff <= Q_BYTE_DIFF_TOLERANCE
        return {
            "verification_passed": verification_passed,
            "q_max_byte_diff": q_max_byte_diff,
            "q_exact_fraction": q_exact_fraction,
            "scale_bytes_exact": scale_bytes_exact,
        }

    def make_compiled_quantizer(
        cfg: MXFP8QuantizerConfig,
        x_ptr: cute.Pointer,
        q_dwords_ptr: cute.Pointer,
        s_words_ptr: cute.Pointer,
        m: int,
        k: int,
        s_word_count: int,
        stream: cuda.CUstream,
    ):
        key = (
            cfg.input_dtype,
            cfg.rows_per_cta,
            cfg.k_tile,
        )
        compiled = _compile_cache.get(key)
        if compiled is None:
            quantizer = BlackwellMXFP8Quantizer(cfg)
            compiled = cute.compile(
                quantizer,
                x_ptr,
                q_dwords_ptr,
                s_words_ptr,
                Int32(m),
                Int32(k),
                Int32(s_word_count),
                stream,
            )
            _compile_cache[key] = compiled
        return compiled

    def benchmark_compiled_quantizer(
        compiled_quantizer,
        stream: cuda.CUstream,
        warmup: int,
        iters: int,
        workspace_generator,
        workspace_count: int,
        repeats: int,
    ) -> dict[str, float]:
        samples_us: list[float] = []
        for _ in range(repeats):
            samples_us.append(
                testing.benchmark(
                    compiled_quantizer,
                    workspace_generator=workspace_generator,
                    workspace_count=workspace_count,
                    warmup_iterations=warmup,
                    iterations=iters,
                    stream=stream,
                )
            )
        min_runtime_us = min(samples_us)
        mean_runtime_us = statistics.fmean(samples_us)
        std_runtime_us = statistics.pstdev(samples_us) if len(samples_us) > 1 else 0.0
        return {
            "min_runtime_us": min_runtime_us,
            "mean_runtime_us": mean_runtime_us,
            "std_runtime_us": std_runtime_us,
            "min_runtime_s": min_runtime_us * 1.0e-6,
            "mean_runtime_s": mean_runtime_us * 1.0e-6,
            "std_runtime_s": std_runtime_us * 1.0e-6,
        }

    _GRID_DIM_Y_LIMIT = 65535
    _INT32_MAX = 2**31 - 1

    def shape_problem(m: int, k: int, rows_per_cta: int, k_tile: int):
        """Map a logical (m, k) problem to the physical (padded) problem the
        kernel executes, validating every real constraint along the way.
        """
        if m < 1 or k < 1:
            raise ValueError(f"m and k must be >= 1, got m={m}, k={k}")
        if m > _INT32_MAX or k > _INT32_MAX:
            raise ValueError(
                f"m and k must fit Int32 (<= {_INT32_MAX}) because they are passed "
                f"to the kernel as Int32 scalars, got m={m}, k={k}"
            )
        kernel_m = ceil_div(m, rows_per_cta) * rows_per_cta
        kernel_k = ceil_div(k, k_tile) * k_tile
        grid_m = kernel_m // rows_per_cta
        grid_k = kernel_k // k_tile
        if grid_m > _INT32_MAX:
            raise ValueError(
                f"gridDim.x limit exceeded: {grid_m} row-tiles > {_INT32_MAX}"
            )
        if grid_k > _GRID_DIM_Y_LIMIT:
            raise ValueError(
                f"gridDim.y limit exceeded: k={k} with k_tile={k_tile} needs "
                f"grid_k={grid_k} > {_GRID_DIM_Y_LIMIT} K-tiles. Reduce k_tile or "
                "reshape the problem."
            )
        return kernel_m, kernel_k, grid_m, grid_k

    def check_cuda_launch_errors(cuda, stream, context: str = "kernel launch") -> None:
        result = cuda.cuStreamSynchronize(stream)
        status = result[0] if isinstance(result, tuple) else result
        if status != cuda.CUresult.CUDA_SUCCESS:
            err = cuda.cuGetErrorName(status)
            name = err[1] if isinstance(err, tuple) else err
            raise RuntimeError(f"CUDA error after {context}: {name}")

    def _resolve_input_dtype(dtype_name: str):
        dtype_name = dtype_name.lower()
        if dtype_name == "bf16":
            return torch.bfloat16, cutlass.BFloat16
        if dtype_name == "fp16":
            return torch.float16, cutlass.Float16
        raise ValueError("input_dtype must be one of: bf16, fp16")

    def prepare_and_quantize(m: int, k: int, dtype_name: str, seed: int, stream):
        """Quantize one (m, k) matrix end to end: shape, pad, launch, verify.

        Returns the materialized buffers (including the packed atom-layout scale
        bytes), the compiled kernel, and the per-matrix verification stats. Used
        by the end-to-end blockscaled GEMM path.
        """
        torch.manual_seed(seed)
        torch_input_dtype, cutlass_input_dtype = _resolve_input_dtype(dtype_name)
        kernel_m, kernel_k, _, _ = shape_problem(
            m, k, DEFAULT_ROWS_PER_CTA, DEFAULT_K_TILE
        )
        x = torch.randn((m, k), dtype=torch_input_dtype, device="cuda")
        buffers = materialize_quantizer_buffers(
            kernel_m, kernel_k, torch_input_dtype, cutlass_input_dtype, x
        )
        cfg = MXFP8QuantizerConfig(input_dtype=cutlass_input_dtype)
        compiled = make_compiled_quantizer(
            cfg,
            buffers["x_ptr"],
            buffers["q_dwords_ptr"],
            buffers["s_words_ptr"],
            kernel_m,
            kernel_k,
            buffers["s_storage_word_count"],
            stream,
        )
        compiled(
            buffers["x_ptr"],
            buffers["q_dwords_ptr"],
            buffers["s_words_ptr"],
            Int32(kernel_m),
            Int32(kernel_k),
            Int32(buffers["s_storage_word_count"]),
            stream,
        )
        torch.cuda.synchronize()
        check_cuda_launch_errors(cuda, stream, "quantizer launch")
        verify = verify_quantizer_outputs(buffers, kernel_m, kernel_k, m, k)
        if not verify["verification_passed"]:
            raise ValueError(
                f"quantizer verification failed for ({m}, {k}): {verify}"
            )
        return {
            "m": m,
            "k": k,
            "kernel_m": kernel_m,
            "kernel_k": kernel_k,
            "x": x,
            "buffers": buffers,
            "compiled": compiled,
            "verify": verify,
        }

    def run_quantizer(
        m: int,
        k: int,
        rows_per_cta: int,
        k_tile: int,
        input_dtype: str,
        warmup_iterations: int,
        iterations: int,
        benchmark_repeats: int,
        cold_l2: bool,
        skip_verification: bool,
        seed: int,
    ) -> dict[str, Any]:
        torch.manual_seed(seed)
        dtype_name = input_dtype.lower()
        torch_input_dtype, cutlass_input_dtype = _resolve_input_dtype(input_dtype)

        x = torch.randn((m, k), dtype=torch_input_dtype, device="cuda")

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        is_full_tile = (m % rows_per_cta == 0) and (k % k_tile == 0)
        kernel_m, kernel_k, grid_m, grid_k = shape_problem(m, k, rows_per_cta, k_tile)
        buffers = materialize_quantizer_buffers(
            kernel_m,
            kernel_k,
            torch_input_dtype,
            cutlass_input_dtype,
            x,
        )
        cfg = MXFP8QuantizerConfig(
            input_dtype=cutlass_input_dtype,
            rows_per_cta=rows_per_cta,
            k_tile=k_tile,
        )
        compiled_quantizer = make_compiled_quantizer(
            cfg,
            buffers["x_ptr"],
            buffers["q_dwords_ptr"],
            buffers["s_words_ptr"],
            kernel_m,
            kernel_k,
            buffers["s_storage_word_count"],
            stream,
        )

        verify_results = None
        if not skip_verification:
            # Launch once on the seeded input and compare against the independent
            # torch reference before any timing is collected, so a wrong kernel
            # can never silently produce benchmark numbers.
            compiled_quantizer(
                buffers["x_ptr"],
                buffers["q_dwords_ptr"],
                buffers["s_words_ptr"],
                Int32(kernel_m),
                Int32(kernel_k),
                Int32(buffers["s_storage_word_count"]),
                stream,
            )
            torch.cuda.synchronize()
            check_cuda_launch_errors(cuda, stream, "quantizer verification launch")
            verify_results = verify_quantizer_outputs(
                buffers, kernel_m, kernel_k, m, k
            )
            if not verify_results["verification_passed"]:
                raise ValueError(
                    "MXFP8 quantizer correctness check FAILED: "
                    f"scale bytes bit-exact={verify_results['scale_bytes_exact']}, "
                    f"fp8 payload max byte diff={verify_results['q_max_byte_diff']} "
                    f"(tolerance {Q_BYTE_DIFF_TOLERANCE}), "
                    f"exact byte fraction={verify_results['q_exact_fraction']:.6f}"
                )
            print(
                "Correctness check passed: "
                f"scale bytes bit-exact={verify_results['scale_bytes_exact']}, "
                f"fp8 payload max byte diff={verify_results['q_max_byte_diff']}, "
                f"exact byte fraction={verify_results['q_exact_fraction']:.6f}"
            )

        def workspace_generator():
            # Each call returns a fresh set of buffers, so the cold-L2 benchmark
            # path genuinely cycles through distinct memory workspaces instead
            # of re-using one buffer set.
            workspace_x = torch.randn(
                (kernel_m, kernel_k), dtype=torch_input_dtype, device="cuda"
            )
            workspace_buffers = materialize_quantizer_buffers(
                kernel_m, kernel_k, torch_input_dtype, cutlass_input_dtype, workspace_x
            )
            return testing.JitArguments(
                workspace_buffers["x_ptr"],
                workspace_buffers["q_dwords_ptr"],
                workspace_buffers["s_words_ptr"],
                Int32(kernel_m),
                Int32(kernel_k),
                Int32(workspace_buffers["s_storage_word_count"]),
                stream,
            )

        logical_bytes_moved = (2 * m * k) + (1 * m * k) + (1 * m * (k // SCALE_VEC_SIZE))
        physical_bytes_moved = (
            (2 * kernel_m * kernel_k)
            + (1 * kernel_m * kernel_k)
            + (1 * kernel_m * (kernel_k // SCALE_VEC_SIZE))
        )
        one_workspace_bytes = (
            buffers["x"].numel() * buffers["x"].element_size()
            + buffers["q_bytes"].numel() * buffers["q_bytes"].element_size()
            + buffers["s_storage_bytes"].numel() * buffers["s_storage_bytes"].element_size()
        )
        workspace_count = (
            testing.get_workspace_count(one_workspace_bytes, warmup_iterations, iterations)
            if cold_l2
            else 1
        )

        timing = benchmark_compiled_quantizer(
            compiled_quantizer,
            stream=stream,
            warmup=warmup_iterations,
            iters=iterations,
            workspace_generator=workspace_generator,
            workspace_count=workspace_count,
            repeats=benchmark_repeats,
        )
        runtime_min_s = timing["min_runtime_s"]
        runtime_mean_s = timing["mean_runtime_s"]
        runtime_std_s = timing["std_runtime_s"]

        effective_tb_s = logical_bytes_moved / runtime_min_s / 1.0e12
        effective_gb_s = logical_bytes_moved / runtime_min_s / 1.0e9
        effective_gb_s_mean = logical_bytes_moved / runtime_mean_s / 1.0e9

        result = {
            "launch_success": True,
            "verification_ran": verify_results is not None,
            "verification_passed": (
                verify_results["verification_passed"] if verify_results is not None else None
            ),
            "q_max_byte_diff": verify_results["q_max_byte_diff"] if verify_results is not None else None,
            "q_exact_fraction": verify_results["q_exact_fraction"] if verify_results is not None else None,
            "scale_bytes_exact": (
                verify_results["scale_bytes_exact"] if verify_results is not None else None
            ),
            "input_dtype": dtype_name,
            "rows_per_cta": cfg.rows_per_cta,
            "k_tile": cfg.k_tile,
            "threads_per_cta": cfg.threads_per_cta,
            "blocks_per_tile": cfg.blocks_per_tile,
            "kernel_m": kernel_m,
            "kernel_k": kernel_k,
            "grid_m": grid_m,
            "grid_k": grid_k,
            "total_ctas": grid_m * grid_k,
            "runtime_min_s": runtime_min_s,
            "runtime_mean_s": runtime_mean_s,
            "runtime_std_s": runtime_std_s,
            "runtime_min_us": timing["min_runtime_us"],
            "runtime_mean_us": timing["mean_runtime_us"],
            "runtime_std_us": timing["std_runtime_us"],
            "effective_gb_s": effective_gb_s,
            "effective_gb_s_mean": effective_gb_s_mean,
            "effective_tb_s": effective_tb_s,
            "logical_bytes_moved": logical_bytes_moved,
            "physical_bytes_moved": physical_bytes_moved,
            "benchmark_repeats": benchmark_repeats,
            "workspace_count": workspace_count,
            "cold_l2": cold_l2,
            "is_full_tile": is_full_tile,
            "dispatch_mode": "cute_full_tile" if is_full_tile else "cute_padded_fallback",
        }
        return result

    def run_e2e_verification(
        m: int,
        n: int,
        k: int,
        input_dtype: str,
        seed: int = 0,
        gemm_tolerance: float = 1.0e-2,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        cluster_shape_mn: tuple[int, int] = (1, 1),
    ) -> dict[str, Any]:
        """Quantize A (m, k) and B (n, k) with the MXFP8 kernel and feed the
        outputs directly into the vendored SM100 blockscaled GEMM consumer.
        """
        from blockscaled_gemm import (
            Sm100BlockScaledPersistentDenseGemmKernel,
            scaled_mm,
        )

        # Validate the GEMM consumer's tiler/cluster constraints up front so a
        # misconfigured suite entry fails loudly before any compile or launch
        # (mirrors is_valid_mma_tiler_and_cluster_shape in blockscaled_gemm.py).
        if (
            mma_tiler_mn[0] not in (128, 256)
            or mma_tiler_mn[1] not in (64, 128, 192, 256)
        ):
            raise ValueError(
                f"mma_tiler_mn must be M in (128, 256) and N in (64, 128, 192, 256), "
                f"got {mma_tiler_mn}"
            )
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or cluster_shape_mn[0] * cluster_shape_mn[1] > 16
        ):
            raise ValueError(
                f"cluster_shape_mn entries must be powers of two in [1, 4] with a "
                f"product <= 16 (SF multicast limit), got {cluster_shape_mn}"
            )
        if mma_tiler_mn[0] == 256 and cluster_shape_mn[0] % 2 != 0:
            raise ValueError(
                f"cluster_shape_mn[0] must be a multiple of 2 when "
                f"mma_tiler_mn[0] == 256 (2-CTA MMA), got cluster_shape_mn="
                f"{cluster_shape_mn}"
            )
        if m % mma_tiler_mn[0] != 0 or n % mma_tiler_mn[1] != 0:
            raise ValueError(
                f"GEMM consumer requires M % {mma_tiler_mn[0]} == 0 and "
                f"N % {mma_tiler_mn[1]} == 0 for mma_tiler_mn={mma_tiler_mn}, "
                f"got m={m}, n={n}"
            )
        if k % DEFAULT_K_TILE != 0:
            raise ValueError(
                f"GEMM consumer requires K % {DEFAULT_K_TILE} == 0 (mma K tiler + "
                "quantizer k_tile, so the emitted scale layout is never padded), "
                f"got k={k}"
            )

        torch.manual_seed(seed)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        a_res = prepare_and_quantize(m, k, input_dtype, seed, stream)
        b_res = prepare_and_quantize(n, k, input_dtype, seed + 1, stream)

        # Aligned problems => the quantizer emits unpadded buffers, so the
        # fp8 payloads are exactly (m, k) / (n, k) and the packed scale bytes
        # are exactly the tcgen05 atom layout for the GEMM's (m, k) / (n, k).
        a_q = a_res["buffers"]["q_fp8"]
        b_q = b_res["buffers"]["q_fp8"]
        a_s_bytes = a_res["buffers"]["s_storage_bytes"]
        b_s_bytes = b_res["buffers"]["s_storage_bytes"]
        c = torch.zeros((m, n), dtype=torch.float32, device="cuda")

        gemm_obj = Sm100BlockScaledPersistentDenseGemmKernel(
            32,  # sf_vec_size, matching the quantizer's SCALE_VEC_SIZE
            mma_tiler_mn,
            cluster_shape_mn,
        )
        if not gemm_obj.can_implement(
            (m, n, k, 1),
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            cutlass.Float32,
            "k",  # a_major: A is row-major (M, K) => K is the contiguous dim
            "k",  # b_major: B is row-major (N, K) => K is the contiguous dim
            "n",  # c_major: C is row-major (M, N)
            32,
            mma_tiler_mn,
            cluster_shape_mn,
        ):
            raise ValueError("GEMM cannot implement this problem configuration")

        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )
        compiled_gemm = scaled_mm(
            gemm_obj,
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.Float32,
            cutlass.Float8E8M0FNU,
            "k",
            "k",  
            "n",  
            max_active_clusters,
            stream,
        )

        a_ptr = make_ptr(
            cutlass.Float8E4M3FN, a_q.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        b_ptr = make_ptr(
            cutlass.Float8E4M3FN, b_q.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        sfa_ptr = make_ptr(
            cutlass.Float8E8M0FNU,
            a_s_bytes.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        sfb_ptr = make_ptr(
            cutlass.Float8E8M0FNU,
            b_s_bytes.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        c_ptr = make_ptr(
            cutlass.Float32, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )

        # The scale-factor pointers point directly at the quantizer's packed
        # buffers: the GEMM reads them in the tcgen05 atom layout with no
        # intermediate repacking stage. If this layout ever diverged from
        # BlockScaledBasicChunk, C would be garbage and the check below fails.
        compiled_gemm(
            a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, 1), stream
        )
        torch.cuda.synchronize()
        check_cuda_launch_errors(cuda, stream, "blockscaled GEMM launch")
        sfa_scale = torch.exp2(
            decode_scale_bytes(a_s_bytes, m, k).to(torch.float32) - 127.0
        ).repeat_interleave(SCALE_VEC_SIZE, dim=1)[:, :k]
        sfb_scale = torch.exp2(
            decode_scale_bytes(b_s_bytes, n, k).to(torch.float32) - 127.0
        ).repeat_interleave(SCALE_VEC_SIZE, dim=1)[:, :k]
        ref = (a_q.to(torch.float32) * sfa_scale) @ (
            b_q.to(torch.float32) * sfb_scale
        ).T

        abs_err = (c - ref).abs()
        max_abs_err = float(abs_err.max().item())
        allowed_err = gemm_tolerance + gemm_tolerance * ref.abs()
        max_err_ratio = float((abs_err / allowed_err).max().item())
        pass_margin = 1.0 / max_err_ratio
        torch.testing.assert_close(c, ref, rtol=gemm_tolerance, atol=gemm_tolerance)

        print(
            "End-to-end blockscaled GEMM check passed: "
            f"max_abs_err={max_abs_err:.3e}, "
            f"max_err_ratio={max_err_ratio:.3f} "
            f"(atol=rtol={gemm_tolerance:.0e}, pass margin={pass_margin:.1f}x)"
        )
        return {
            "e2e_verification_passed": True,
            "m": m,
            "n": n,
            "k": k,
            "input_dtype": input_dtype,
            "gemm_config": {
                "mma_tiler_mn": list(mma_tiler_mn),
                "cluster_shape_mn": list(cluster_shape_mn),
                "sf_vec_size": SCALE_VEC_SIZE,
                "sf_dtype": "Float8E8M0FNU",
                "a/b_dtype": "Float8E4M3FN",
                "c_dtype": "Float32",
            },
            "sfa_layout": "tcgen05_atom_layout_packed_direct",
            "quantizer_verify_a": a_res["verify"],
            "quantizer_verify_b": b_res["verify"],
            "max_abs_err": max_abs_err,
            "max_err_ratio": max_err_ratio,
            "pass_margin": pass_margin,
            "gemm_tolerance": gemm_tolerance,
        }

    def run_e2e_suite(
        configs: tuple[
            tuple[int, int, int, str, int, tuple[int, int], tuple[int, int]], ...
        ] = DEFAULT_E2E_SUITE,
        gemm_tolerance: float = 1.0e-2,
    ) -> dict[str, Any]:
        """Run the e2e quantizer -> blockscaled GEMM verification across a
        suite of (m, n, k, dtype, seed, mma_tiler, cluster) configs.

        Every config must pass its quantizer verification and the GEMM check;
        any failure raises with the failing config's details. Configs that
        share an (mma_tiler_mn, cluster_shape_mn) pair can reuse the compiled
        GEMM, so the default suite groups them; each distinct pair costs one
        GEMM compile (the dominant cost), which is why the default suite is
        kept small.
        """
        results = []
        for cfg in configs:
            m, n, k, input_dtype, seed, mma_tiler_mn, cluster_shape_mn = cfg
            result = run_e2e_verification(
                m=m,
                n=n,
                k=k,
                input_dtype=input_dtype,
                seed=seed,
                gemm_tolerance=gemm_tolerance,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
            )
            results.append(result)

        config_summary = []
        for cfg, r in zip(configs, results):
            m, n, k, input_dtype, seed, mma_tiler_mn, cluster_shape_mn = cfg
            config_summary.append(
                {
                    "m": m,
                    "n": n,
                    "k": k,
                    "input_dtype": input_dtype,
                    "seed": seed,
                    "mma_tiler_mn": list(mma_tiler_mn),
                    "cluster_shape_mn": list(cluster_shape_mn),
                    "quantizer_verify_passed": (
                        r["quantizer_verify_a"]["verification_passed"]
                        and r["quantizer_verify_b"]["verification_passed"]
                    ),
                    "max_abs_err": r["max_abs_err"],
                    "max_err_ratio": r["max_err_ratio"],
                    "pass_margin": r["pass_margin"],
                }
            )
        worst_abs = max(r["max_abs_err"] for r in results)
        worst_ratio = max(r["max_err_ratio"] for r in results)
        print(
            f"e2e suite passed: {len(results)}/{len(results)} configs "
            f"(worst max_abs_err={worst_abs:.3e}, "
            f"worst max_err_ratio={worst_ratio:.3f})"
        )
        return {
            "e2e_suite_passed": True,
            "num_configs": len(results),
            "gemm_tolerance": gemm_tolerance,
            "configs": config_summary,
        }

    return {
        "run_quantizer": run_quantizer,
        "run_e2e_verification": run_e2e_verification,
        "run_e2e_suite": run_e2e_suite,
        "prepare_and_quantize": prepare_and_quantize,
    }


# build_runtime() is a factory, not an entrypoint: it exists so the heavy
# CUDA/CuTeDSL/torch imports stay inside one deferred scope (this module stays
# importable anywhere without the GPU stack) and so a fresh compile cache is
# created per process invocation.

def run_quantizer(
    m: int = 16384,
    k: int = 16384,
    rows_per_cta: int = DEFAULT_ROWS_PER_CTA,
    k_tile: int = DEFAULT_K_TILE,
    input_dtype: str = "bf16",
    warmup_iterations: int = 10,
    iterations: int = 50,
    benchmark_repeats: int = 5,
    cold_l2: bool = False,
    skip_verification: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    return build_runtime()["run_quantizer"](
        m=m,
        k=k,
        rows_per_cta=rows_per_cta,
        k_tile=k_tile,
        input_dtype=input_dtype,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        benchmark_repeats=benchmark_repeats,
        cold_l2=cold_l2,
        skip_verification=skip_verification,
        seed=seed,
    )

def run_e2e_verification(
    m: int = 1024,
    n: int = 1024,
    k: int = 2048,
    input_dtype: str = "bf16",
    seed: int = 0,
    gemm_tolerance: float = 1.0e-2,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    cluster_shape_mn: tuple[int, int] = (1, 1),
) -> dict[str, Any]:
    return build_runtime()["run_e2e_verification"](
        m=m,
        n=n,
        k=k,
        input_dtype=input_dtype,
        seed=seed,
        gemm_tolerance=gemm_tolerance,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )

def run_e2e_suite(
    configs: tuple[
        tuple[int, int, int, str, int, tuple[int, int], tuple[int, int]], ...
    ] = DEFAULT_E2E_SUITE,
    gemm_tolerance: float = 1.0e-2,
) -> dict[str, Any]:
    return build_runtime()["run_e2e_suite"](
        configs=configs, gemm_tolerance=gemm_tolerance
    )

def prepare_and_quantize(m: int, k: int, dtype_name: str, seed: int, stream) -> dict[str, Any]:
    """Lower-level helper: quantize one (m, k) matrix end to end (shape, pad,
    launch, verify) and return the buffers, compiled kernel, and verification
    stats. Used by :func:`run_e2e_verification`; exposed for tests."""
    return build_runtime()["prepare_and_quantize"](m, k, dtype_name, seed, stream)
