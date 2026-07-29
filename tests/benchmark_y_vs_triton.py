# tests/benchmark_y_vs_triton.py
import os
import sys
import time
import subprocess
import torch
import cupy as cp
import numpy as np

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[!] Triton missing from environment.")
    sys.exit(1)

# CUDA kernel source for Y
CUDA_KERNELS_PATH = os.path.join(os.path.dirname(__file__), "y_tensor_core_gemm.cu")
with open(CUDA_KERNELS_PATH, "r") as f:
    CUDA_SRC = f.read()

# -----------------------------------------------------------------------------
# Triton Kernels
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _triton_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _triton_fused_gemm_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    bias_ptrs = bias_ptr + offs_bn
    bias = tl.load(bias_ptrs)
    accumulator = accumulator + bias[None, :]
    accumulator = tl.maximum(accumulator, 0.0)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# -----------------------------------------------------------------------------
# Main Benchmark Logic
# -----------------------------------------------------------------------------
def main():
    print("=" * 85)
    print("           Y TENSOR CORE COMPILER vs TRITON NATIVE JIT COMPILER")
    print("                     HEAD-TO-HEAD BENCHMARK SUITE")
    print("=" * 85)
    print(f"[*] Hardware GPU: {torch.cuda.get_device_name(0)}")
    print(f"[*] PyTorch Version: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"[*] Triton Version: {triton.__version__}")

    # Compile Y CuPy Module
    y_mod = cp.RawModule(code=CUDA_SRC, options=("-std=c++17", "--use_fast_math"))
    y_gemm_large = y_mod.get_function("y_tensor_core_gemm_kernel")
    y_gemm_large.max_dynamic_shared_size_bytes = 65536
    y_gemm_small = y_mod.get_function("y_tensor_core_gemm_small_kernel")
    y_fused_large = y_mod.get_function("y_fused_gemm_bias_relu_fp16_kernel")
    y_fused_large.max_dynamic_shared_size_bytes = 65536
    y_fused_small_vec = y_mod.get_function("y_fused_gemm_bias_relu_small_fp16_kernel")

    dimensions = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    # --- Section 1: Standalone GEMM Benchmark ---
    print("\n" + "=" * 85)
    print(" SECTION 1: STANDALONE GEMM FP16 (Y vs TRITON)")
    print("=" * 85)
    print(f"{'Matrix (M=N=K)':<18} | {'Triton (us)':<14} | {'Y Compiler (us)':<16} | {'Speedup (Y / Triton)':<20}")
    print("-" * 85)

    for dim in dimensions:
        M = N = K = dim
        iters = 200 if dim <= 2048 else (50 if dim <= 4096 else (20 if dim <= 8192 else (5 if dim <= 16384 else 2)))
        warmup = 10 if dim <= 8192 else (2 if dim <= 16384 else 1)

        A_pt = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B_pt = torch.randn(K, N, dtype=torch.float16, device="cuda")
        C_triton = torch.empty(M, N, dtype=torch.float16, device="cuda")

        A_cp = cp.asarray(A_pt)
        B_cp = cp.asarray(B_pt)
        C_y = cp.empty((M, N), dtype=cp.float16)

        grid_triton = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

        # Warmup Triton
        for _ in range(warmup):
            _triton_gemm_kernel[grid_triton](
                A_pt, B_pt, C_triton,
                M, N, K,
                A_pt.stride(0), A_pt.stride(1),
                B_pt.stride(0), B_pt.stride(1),
                C_triton.stride(0), C_triton.stride(1)
            )
        torch.cuda.synchronize()

        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        for _ in range(iters):
            _triton_gemm_kernel[grid_triton](
                A_pt, B_pt, C_triton,
                M, N, K,
                A_pt.stride(0), A_pt.stride(1),
                B_pt.stride(0), B_pt.stride(1),
                C_triton.stride(0), C_triton.stride(1)
            )
        t_end.record()
        torch.cuda.synchronize()
        triton_us = (t_start.elapsed_time(t_end) / iters) * 1000.0

        # Y Execution (Autotuned Tile Selection)
        candidates = []
        # Candidate 1: 128x128 Tile (half output, 64KB dynamic smem)
        g_m_1 = (M + 127) // 128
        g_n_1 = (N + 127) // 128
        C_y_f16 = cp.empty((M, N), dtype=cp.float16)
        candidates.append(((g_n_1, g_m_1, 1), (256, 1, 1), y_gemm_large, C_y_f16, 65536))

        if M <= 1024:
            # Candidate 2: 64x64 Tile (float32 output)
            g_m_2 = (M + 63) // 64
            g_n_2 = (N + 63) // 64
            C_y_f32 = cp.empty((M, N), dtype=cp.float32)
            candidates.append(((g_n_2, g_m_2, 1), (128, 1, 1), y_gemm_small, C_y_f32, 0))

        best_y_us = float("inf")

        for grid, block, k_func, out_buf, smem in candidates:
            cp.cuda.Device(0).synchronize()
            for _ in range(warmup):
                k_func(grid, block, (A_cp, B_cp, out_buf, M, N, K), shared_mem=smem)
            cp.cuda.Device(0).synchronize()

            y_start = cp.cuda.Event()
            y_end = cp.cuda.Event()
            y_start.record()
            for _ in range(iters):
                k_func(grid, block, (A_cp, B_cp, out_buf, M, N, K), shared_mem=smem)
            y_end.record()
            y_end.synchronize()
            cand_us = (cp.cuda.get_elapsed_time(y_start, y_end) / iters) * 1000.0
            if cand_us < best_y_us:
                best_y_us = cand_us

        y_us = best_y_us

        speedup = triton_us / y_us
        print(f"{f'{dim}x{dim}x{dim}':<18} | {triton_us:<14.2f} | {y_us:<16.2f} | {speedup:<20.2f}x")

        del A_pt, B_pt, C_triton, A_cp, B_cp, C_y
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

    # --- Section 2: Fused GEMM + Bias + ReLU Benchmark ---
    print("\n" + "=" * 85)
    print(" SECTION 2: FUSED GEMM + BIAS + RELU (Y vs TRITON)")
    print("=" * 85)
    print(f"{'Matrix (M=N=K)':<18} | {'Triton Fused (us)':<18} | {'Y Fused (us)':<16} | {'Speedup (Y / Triton)':<20}")
    print("-" * 85)

    for dim in dimensions:
        M = N = K = dim
        iters = 200 if dim <= 2048 else (50 if dim <= 4096 else (20 if dim <= 8192 else (5 if dim <= 16384 else 2)))
        warmup = 10 if dim <= 8192 else (2 if dim <= 16384 else 1)

        A_pt = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B_pt = torch.randn(K, N, dtype=torch.float16, device="cuda")
        bias_pt = torch.randn(N, dtype=torch.float32, device="cuda")
        C_triton = torch.empty(M, N, dtype=torch.float16, device="cuda")

        A_cp = cp.asarray(A_pt)
        B_cp = cp.asarray(B_pt)
        bias_cp = cp.asarray(bias_pt.half())
        C_y = cp.empty((M, N), dtype=cp.float16)

        grid_triton = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

        # Warmup Triton Fused
        for _ in range(warmup):
            _triton_fused_gemm_bias_relu_kernel[grid_triton](
                A_pt, B_pt, bias_pt, C_triton,
                M, N, K,
                A_pt.stride(0), A_pt.stride(1),
                B_pt.stride(0), B_pt.stride(1),
                C_triton.stride(0), C_triton.stride(1)
            )
        torch.cuda.synchronize()

        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        for _ in range(iters):
            _triton_fused_gemm_bias_relu_kernel[grid_triton](
                A_pt, B_pt, bias_pt, C_triton,
                M, N, K,
                A_pt.stride(0), A_pt.stride(1),
                B_pt.stride(0), B_pt.stride(1),
                C_triton.stride(0), C_triton.stride(1)
            )
        t_end.record()
        torch.cuda.synchronize()
        triton_us = (t_start.elapsed_time(t_end) / iters) * 1000.0

        # Y Fused Execution
        if M <= 512:
            grid_m = (M + 63) // 64
            grid_n = (N + 63) // 64
            threads = 128
            k_func = y_fused_small_vec
        else:
            grid_m = (M + 127) // 128
            grid_n = (N + 127) // 128
            threads = 256
            k_func = y_fused_large
        smem = 0
        args = (A_cp, B_cp, bias_cp, C_y, M, N, K, 1)

        cp.cuda.Device(0).synchronize()
        for _ in range(warmup):
            k_func((grid_n, grid_m, 1), (threads, 1, 1), args, shared_mem=smem)
        cp.cuda.Device(0).synchronize()

        y_start = cp.cuda.Event()
        y_end = cp.cuda.Event()
        y_start.record()
        for _ in range(iters):
            k_func((grid_n, grid_m, 1), (threads, 1, 1), args, shared_mem=smem)
        y_end.record()
        y_end.synchronize()
        y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / iters) * 1000.0

        speedup = triton_us / y_us
        print(f"{f'{dim}x{dim}x{dim}':<18} | {triton_us:<18.2f} | {y_us:<16.2f} | {speedup:<20.2f}x")

        del A_pt, B_pt, bias_pt, C_triton, A_cp, B_cp, bias_cp, C_y
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

    print("=" * 85)

if __name__ == "__main__":
    main()
