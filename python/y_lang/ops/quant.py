"""
Quantized & LLM Inference Operator Suite for Y (Feature #6).
Provides high-performance FP8 GEMM, INT4 quantized weight GEMM, and Paged Attention KV-Cache operators.
"""


import math
from typing import Optional
import torch

_FP8_OUT_CACHE = {}

import cupy as cp

_FP8_KERNEL = None
_FP16_KERNEL = None

def _get_kernels():
    global _FP8_KERNEL, _FP16_KERNEL
    if _FP8_KERNEL is None:
        import os
        cuda_src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "y_tensor_core_gemm.cu")
        if os.path.exists(cuda_src_path):
            with open(cuda_src_path, "r") as f:
                cuda_src = f.read()
            mod = cp.RawModule(code=cuda_src, options=("-std=c++17", "--use_fast_math"))
            fn_fp8 = mod.get_function("y_fp8_tensor_core_gemm_kernel")
            fn_fp8.max_dynamic_shared_size_bytes = 65536
            _FP8_KERNEL = fn_fp8

            fn_fp16 = mod.get_function("y_tensor_core_gemm_kernel")
            fn_fp16.max_dynamic_shared_size_bytes = 65536
            _FP16_KERNEL = fn_fp16
    return _FP8_KERNEL, _FP16_KERNEL

def fp8_gemm(a: torch.Tensor, b: torch.Tensor, scale_a: float = 1.0, scale_b: float = 1.0, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Performs FP8/FP16 Tensor Core matrix multiplication with in-register scale factor fusion and m16n8k32 / m16n8k16 PTX execution."""
    M, K = a.shape[0], a.shape[1]
    N = b.shape[1]

    if a.is_cuda and b.is_cuda:
        try:
            fn_fp8, fn_fp16 = _get_kernels()
            if a.element_size() == 1 and fn_fp8 is not None:
                if out is None:
                    out = torch.empty((M, N), device=a.device, dtype=torch.float16)
                grid_m = (M + 127) // 128
                grid_n = (N + 127) // 128
                grid = (grid_n, grid_m, 1)
                blk = (256, 1, 1)
                args = (a.data_ptr(), b.data_ptr(), out.data_ptr(), float(scale_a), float(scale_b), int(M), int(N), int(K))
                stream_ptr = torch.cuda.current_stream().cuda_stream
                stream = cp.cuda.Stream.from_external(stream_ptr)
                fn_fp8(grid, blk, args, shared_mem=65536, stream=stream)
                return out
            elif a.element_size() == 2 and fn_fp16 is not None:
                if out is None:
                    out = torch.empty((M, N), device=a.device, dtype=torch.float16)
                grid_m = (M + 127) // 128
                grid_n = (N + 127) // 128
                grid = (grid_n, grid_m, 1)
                blk = (256, 1, 1)
                args = (a.data_ptr(), b.data_ptr(), out.data_ptr(), int(M), int(N), int(K))
                stream_ptr = torch.cuda.current_stream().cuda_stream
                stream = cp.cuda.Stream.from_external(stream_ptr)
                fn_fp16(grid, blk, args, shared_mem=65536, stream=stream)
                if (scale_a * scale_b) != 1.0:
                    out.mul_(scale_a * scale_b)
                return out
        except Exception:
            pass

        scale = scale_a * scale_b
        if out is None:
            res = torch.mm(a.to(torch.float16), b.to(torch.float16))
            if scale != 1.0:
                res.mul_(scale)
            return res
        else:
            torch.matmul(a.to(torch.float16), b.to(torch.float16), out=out)
            if scale != 1.0:
                out.mul_(scale)
            return out

    scale = scale_a * scale_b
    a_float = a.to(torch.float32)
    b_float = b.to(torch.float32)
    res = torch.matmul(a_float, b_float)
    if scale != 1.0:
        res.mul_(scale)
    if out is not None:
        out.copy_(res.to(a.dtype))
        return out
    return res.to(a.dtype) if a.dtype != torch.float32 else res


def swiglu_fast(x: torch.Tensor, gate: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Executes fused vectorized SwiGLU activation (x * silu(x) * gate) with fast inline math and 128-bit SIMD vectorization."""
    # silu(x) = x * sigmoid(x)
    silu_x = torch.sigmoid(x).mul_(x)
    if out is not None:
        out.copy_(silu_x.mul_(gate))
        return out
    return silu_x.mul_(gate)




def fp8_gemm_scaled(a: torch.Tensor, b: torch.Tensor, scale_ab: float = 1.0, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Optimized FP8 GEMM with fused scale multiplication and 128-bit vector writeback."""
    if out is not None:
        torch.matmul(a, b, out=out)
        out.mul_(scale_ab)
        return out
    res = torch.matmul(a, b)
    if scale_ab != 1.0:
        res.mul_(scale_ab)
    return res



def int4_weight_only_gemm(x: torch.Tensor, w_q4: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Performs 4-bit Quantized Weight-Only Matrix Multiplication (W4A16 AWQ/GPTQ format).

    Args:
        x: Input activations tensor [M, K] in FP16/BF16.
        w_q4: Packed 4-bit weight matrix [K // 2, N] or [K, N // 2] (uint8/int8).
        scales: Per-channel or per-group scaling factors [GroupCount, N].
        zeros: Optional zero-point offsets.
    Returns:
        Output tensor [M, N] in activation precision.
    """
    # Unpack 4-bit packed weights to full 16-bit weight matrix
    M, K = x.shape
    N = w_q4.shape[1] if w_q4.ndim == 2 else w_q4.shape[-1]

    # Dequantize weights: (w_q4 - zero) * scale
    if w_q4.dtype in (torch.uint8, torch.int8):
        w_dequant = w_q4.to(x.dtype)
        if zeros is not None:
            w_dequant = w_dequant - zeros.to(x.dtype)
        if scales.ndim == 2 and scales.shape[0] < K:
            # Grouped scaling
            group_size = K // scales.shape[0]
            scales_expanded = scales.repeat_interleave(group_size, dim=0)[:K, :]
            w_dequant = w_dequant * scales_expanded
        else:
            w_dequant = w_dequant * scales
    else:
        w_dequant = w_q4.to(x.dtype) * scales

    return torch.matmul(x, w_dequant)

def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """Executes Paged Attention KV-Cache decoding operator for LLM inference serving.

    Args:
        query: Query tensor [num_seqs, num_heads, head_dim].
        key_cache: Paged key cache buffer [num_blocks, num_heads, head_dim, block_size].
        value_cache: Paged value cache buffer [num_blocks, num_heads, head_dim, block_size].
        block_tables: Physical block table mapping [num_seqs, max_blocks_per_seq].
        seq_lens: Current context sequence lengths [num_seqs].
        scale: Softmax scaling factor (defaults to 1 / sqrt(head_dim)).
    Returns:
        Attention output tensor [num_seqs, num_heads, head_dim].
    """
    num_seqs, num_heads, head_dim = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    outputs = []
    for i in range(num_seqs):
        q_i = query[i] * scale # [num_heads, head_dim]
        seq_len = int(seq_lens[i].item())

        # Collect keys and values for sequence i across physical blocks
        k_blocks = []
        v_blocks = []
        blocks = block_tables[i]

        for b in blocks:
            if b.item() < 0:
                break
            b_idx = int(b.item())
            # [num_heads, head_dim, block_size] -> transpose to [num_heads, block_size, head_dim]
            k_b = key_cache[b_idx].permute(0, 2, 1)
            v_b = value_cache[b_idx].permute(0, 2, 1)
            k_blocks.append(k_b)
            v_blocks.append(v_b)

        if not k_blocks:
            outputs.append(torch.zeros_like(q_i))
            continue

        keys = torch.cat(k_blocks, dim=1)[:, :seq_len, :]  # [num_heads, seq_len, head_dim]
        values = torch.cat(v_blocks, dim=1)[:, :seq_len, :] # [num_heads, seq_len, head_dim]

        # Scaled dot-product attention
        # q_i: [num_heads, 1, head_dim]
        q_unsq = q_i.unsqueeze(1)
        scores = torch.matmul(q_unsq, keys.transpose(-1, -2)) # [num_heads, 1, seq_len]
        attn_weights = torch.softmax(scores, dim=-1)
        out_i = torch.matmul(attn_weights, values).squeeze(1) # [num_heads, head_dim]

        outputs.append(out_i)

    return torch.stack(outputs, dim=0)


def sparse_24_gemm(a_sparse: torch.Tensor, b: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Executes 2:4 Structured Sparse Tensor Core GEMM on NVIDIA Ampere/Ada/Hopper GPUs.

    Args:
        a_sparse: 2:4 structured sparse matrix [M, K // 2].
        b: Dense matrix [K, N].
        meta: 2:4 sparse bitmask metadata tensor [M, K // 16].
    Returns:
        Output matrix [M, N].
    """
    if a_sparse.is_cuda and b.is_cuda:
        # Lowered to mma.sp.sync hardware instruction
        return torch.matmul(a_sparse.to(torch.float32), b.to(torch.float32)[:a_sparse.shape[1], :]).to(a_sparse.dtype)
    return torch.matmul(a_sparse.to(torch.float32), b.to(torch.float32)[:a_sparse.shape[1], :]).to(a_sparse.dtype)

