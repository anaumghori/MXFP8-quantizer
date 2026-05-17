from __future__ import annotations
import dataclasses
import json
import statistics
from typing import Any
import modal


APP_NAME = "mxfp8-quantizer-blackwell"
CUTLASS_DSL_WHEEL = "nvidia-cutlass-dsl[cu13]==4.4.2"
TORCH_WHEEL = "torch==2.7.1"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CUDA_PYTHON_WHEEL = "cuda-python>=13.0"
ROWS_PER_CTA = 8
K_TILE = 256


image = (
    modal.Image.debian_slim(python_version="3.12")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            # Keep file-backed CuTe artifacts alive inside the Modal container.
            "CUTE_DSL_CACHE_DIR": "/tmp/cute_dsl_cache",
        }
    )
    .uv_pip_install(
        CUTLASS_DSL_WHEEL,
        CUDA_PYTHON_WHEEL,
        "numpy",
        "packaging",
    )
    .uv_pip_install(
        TORCH_WHEEL,
        extra_index_url=TORCH_CUDA_INDEX,
    )
)

app = modal.App(APP_NAME, image=image)


def _build_runtime():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.testing as testing
    import cutlass.utils as utils
    import torch
    from cutlass import BFloat16, Float16, Float32, Int32, Uint8, Uint16, Uint32, Uint64
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.nvgpu import cpasync
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op

    FP8_E4M3_MAX = 448.0
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
            q_dwords_ptr: cute.Pointer, # points to the packed FP8 output buffer
            s_words_ptr: cute.Pointer, # points to the packed scale buffer
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
        return (a + b - 1) // b   # Integer division rounded up so partial blocks still count as one full block

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
        seed: int,
    ) -> dict[str, Any]:
        torch.manual_seed(seed)
        if k % SCALE_VEC_SIZE != 0:
            raise ValueError("K must be divisible by 32")
        dtype_name = input_dtype.lower()
        if dtype_name == "bf16":
            torch_input_dtype = torch.bfloat16
            cutlass_input_dtype = cutlass.BFloat16
        elif dtype_name == "fp16":
            torch_input_dtype = torch.float16
            cutlass_input_dtype = cutlass.Float16
        else:
            raise ValueError("input_dtype must be one of: bf16, fp16")

        x = torch.randn((m, k), dtype=torch_input_dtype, device="cuda")

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        selected_rows_per_cta = rows_per_cta
        selected_k_tile = k_tile
        full_rows_supported = (m % selected_rows_per_cta) == 0
        full_tile_fast_path = full_rows_supported and (k % selected_k_tile == 0)
        kernel_m = (
            m if full_tile_fast_path else ceil_div(m, selected_rows_per_cta) * selected_rows_per_cta
        )
        kernel_k = k if full_tile_fast_path else ceil_div(k, selected_k_tile) * selected_k_tile
        buffers = materialize_quantizer_buffers(
            kernel_m,
            kernel_k,
            torch_input_dtype,
            cutlass_input_dtype,
            x,
        )
        best_cfg = MXFP8QuantizerConfig(
            input_dtype=cutlass_input_dtype,
            rows_per_cta=selected_rows_per_cta,
            k_tile=selected_k_tile,
        )
        compiled_quantizer = make_compiled_quantizer(
            best_cfg,
            buffers["x_ptr"],
            buffers["q_dwords_ptr"],
            buffers["s_words_ptr"],
            kernel_m,
            kernel_k,
            buffers["s_storage_word_count"],
            stream,
        )

        kernel_args = testing.JitArguments(
            buffers["x_ptr"],
            buffers["q_dwords_ptr"],
            buffers["s_words_ptr"],
            Int32(kernel_m),
            Int32(kernel_k),
            Int32(buffers["s_storage_word_count"]),
            stream,
        )

        def workspace_generator():
            return kernel_args

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
            "input_dtype": dtype_name,
            "rows_per_cta": best_cfg.rows_per_cta,
            "k_tile": best_cfg.k_tile,
            "threads_per_cta": best_cfg.threads_per_cta,
            "blocks_per_tile": best_cfg.blocks_per_tile,
            "kernel_m": kernel_m,
            "kernel_k": kernel_k,
            "grid_m": ceil_div(kernel_m, best_cfg.rows_per_cta),
            "grid_k": ceil_div(kernel_k, best_cfg.k_tile),
            "total_ctas": ceil_div(kernel_m, best_cfg.rows_per_cta) * ceil_div(kernel_k, best_cfg.k_tile),
            "runtime_min_s": runtime_min_s,
            "runtime_mean_s": runtime_mean_s,
            "runtime_std_s": runtime_std_s,
            "runtime_min_us": timing["min_runtime_us"],
            "runtime_mean_us": timing["mean_runtime_us"],
            "runtime_std_us": timing["std_runtime_us"],
            "avg_runtime_s": runtime_mean_s,
            "avg_runtime_us": timing["mean_runtime_us"],
            "effective_gb_s": effective_gb_s,
            "effective_gb_s_mean": effective_gb_s_mean,
            "effective_tb_s": effective_tb_s,
            "logical_bytes_moved": logical_bytes_moved,
            "physical_bytes_moved": physical_bytes_moved,
            "benchmark_repeats": benchmark_repeats,
            "workspace_count": workspace_count,
            "cold_l2": cold_l2,
            "fast_full_tile_path": full_tile_fast_path,
            "dispatch_mode": "cute_full_tile" if full_tile_fast_path else "cute_padded_fallback",
        }
        return result

    return {"run_quantizer": run_quantizer}


@app.function(gpu="B200", timeout=60 * 60)
def run_on_b200(
    m: int = 16384,
    k: int = 16384,
    input_dtype: str = "bf16",
    warmup_iterations: int = 10,
    iterations: int = 50,
    benchmark_repeats: int = 5,
    cold_l2: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    runtime = _build_runtime()
    return runtime["run_quantizer"](
        m=m,
        k=k,
        rows_per_cta=ROWS_PER_CTA,
        k_tile=K_TILE,
        input_dtype=input_dtype,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        benchmark_repeats=benchmark_repeats,
        cold_l2=cold_l2,
        seed=seed,
    )


@app.local_entrypoint()
def main(
    m: int = 16384,
    k: int = 16384,
    input_dtype: str = "bf16",
    warmup_iterations: int = 10,
    iterations: int = 50,
    benchmark_repeats: int = 5,
    cold_l2: bool = False,
    seed: int = 0,
):
    result = run_on_b200.remote(
        m=m,
        k=k,
        input_dtype=input_dtype,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        benchmark_repeats=benchmark_repeats,
        cold_l2=cold_l2,
        seed=seed,
    )
    summary = {
        "launch_success": result["launch_success"],
        "dispatch_mode": result["dispatch_mode"],
        "runtime_min_s": result["runtime_min_s"],
        "runtime_min_us": result["runtime_min_us"],
        "avg_runtime_s": result["avg_runtime_s"],
        "avg_runtime_us": result["avg_runtime_us"],
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
        "fast_full_tile_path": result["fast_full_tile_path"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
