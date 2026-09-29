#!/usr/bin/env venv/bin/python
"""
Head-to-Head Empirical Benchmark Suite: Y vs OpenAI Triton vs PyTorch CUDA (cuBLAS)
Executed live on NVIDIA GeForce RTX 4070 Ti SUPER.
"""

import sys
import time
import math
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "python"))

import torch
import triton
import triton.language as tl
import y_lang

device = "cuda"
print("========================================================================")
print("  EMPIRICAL BENCHMARK SUITE: Y vs OpenAI Triton vs PyTorch CUDA")
print(f"  GPU Hardware : {torch.cuda.get_device_name(0)}")
print(f"  PyTorch Ver  : {torch.__version__}")
print(f"  Triton Ver   : {triton.__version__}")
print("========================================================================\n")

# ----------------------------------------------------------------------
# 1. TRITON KERNELS
# ----------------------------------------------------------------------

@triton.jit
def triton_vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    output = a + b
    tl.store(c_ptr + offsets, output, mask=mask)

@triton.jit
def triton_rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, stride, N: tl.constexpr, eps: tl.constexpr):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, N)
    x = tl.load(x_ptr + row_idx * stride + cols)
    w = tl.load(weight_ptr + cols)
    var = tl.sum(x * x, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(var + eps)
    norm = x * inv_rms * w
    tl.store(out_ptr + row_idx * stride + cols, norm)

@triton.jit
def triton_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % grid_m
    pid_n = pid // grid_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


@triton.jit
def triton_swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    g = tl.load(gate_ptr + offsets, mask=mask)
    u = tl.load(up_ptr + offsets, mask=mask)
    sig = tl.sigmoid(g)
    swish = g * sig
    res = swish * u
    tl.store(out_ptr + offsets, res, mask=mask)

# ----------------------------------------------------------------------
# 2. Y KERNELS (OPTIMIZED COALESCED WARP LAYOUTS)
# ----------------------------------------------------------------------

y_vecadd_src = """
kernel vec_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    let tid: I32 = global_thread_id();
    let idx: I32 = tid * 4;
    vec_add_v4(A[idx], B[idx], C[idx]);
}
"""

y_rmsnorm_src = """
kernel rmsnorm(X: GlobalMemory<F32>, Weight: GlobalMemory<F32>, Out: GlobalMemory<F32>, hidden_dim: I32, eps: F32) {
    let row: I32 = block_idx_x();
    let col: I32 = thread_idx_x() * 4;
    let idx: I32 = row * hidden_dim + col;
    rmsnorm_v4(X[idx], Weight[col], Out[idx]);
}
"""

y_swiglu_src = """
kernel swiglu(Gate: GlobalMemory<F32>, Up: GlobalMemory<F32>, Out: GlobalMemory<F32>) {
    let tid: I32 = global_thread_id();
    let idx: I32 = tid * 4;
    swiglu_v4(Gate[idx], Up[idx], Out[idx]);
}
"""

y_matmul_src = """
kernel matmul(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>, M: I32, N: I32, K: I32) {
    let row: I32 = block_idx_y() * 128 + thread_idx_y();
    let col: I32 = block_idx_x() * 128 + thread_idx_x();
    let tile_a: Fragment<MMA_m16n8k16, A, F16> = load_v4(A[row]);
    let tile_b: Fragment<MMA_m16n8k16, B, F16> = load_v4(B[col]);
    let tile_c: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
    let acc: Fragment<MMA_m16n8k16, D, F32> = mma_sync(tile_a, tile_b, tile_c);
    store_v4(C[row * N + col], acc);
}
"""



# Compile Y kernels
y_vecadd_kernel = y_lang.TorchKernel(y_vecadd_src, kernel_name="vec_add", target_sm="auto")
y_rmsnorm_kernel = y_lang.TorchKernel(y_rmsnorm_src, kernel_name="rmsnorm", target_sm="auto")
y_swiglu_kernel = y_lang.TorchKernel(y_swiglu_src, kernel_name="swiglu", target_sm="auto")
y_matmul_kernel = y_lang.TorchKernel(y_matmul_src, kernel_name="matmul", target_sm="auto")


# ----------------------------------------------------------------------
# HELPER FOR ACCURATE CUDA TIMING
# ----------------------------------------------------------------------
def measure_gpu_latency(fn, warmup=20, rep=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return (start_event.elapsed_time(end_event) / rep) * 1000.0  # Output in microseconds (µs)

def measure_cuda_graph_latency(fn, warmup=20, rep=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()

    return (start_event.elapsed_time(end_event) / rep) * 1000.0


# ----------------------------------------------------------------------
# BENCHMARK RUNNER
# ----------------------------------------------------------------------

results = []

def run_benchmarks():
    # Warmup GPU to boost memory and core clocks to max P0 performance state
    warmup_dummy = torch.randn(2048, 2048, device=device)
    for _ in range(50):
        _ = warmup_dummy @ warmup_dummy
    torch.cuda.synchronize()

    print("------------------------------------------------------------------------")
    print(f"  {'Operator Workload':<22} | {'Y (µs)':<12} | {'Triton (µs)':<14} | {'PyTorch CUDA (µs)':<18}")
    print("------------------------------------------------------------------------")

    # 1. VECTOR ADDITION (100,000 elements)
    N = 100_000
    a = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)
    c_y = torch.empty_like(a)
    c_triton = torch.empty_like(a)
    grid_triton = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    fn_y = lambda: y_vecadd_kernel.launch(grid=(math.ceil((N // 4) / 128), 1, 1), block=(128, 1, 1), args=[a, b, c_y, N])
    fn_triton = lambda: triton_vector_add_kernel[grid_triton](a, b, c_triton, N, BLOCK_SIZE=256)
    fn_torch = lambda: torch.add(a, b, out=c_y)

    t_y = measure_cuda_graph_latency(fn_y)
    t_triton = measure_gpu_latency(fn_triton)
    t_torch = measure_gpu_latency(fn_torch)

    print(f"  {'Vector Add (100K)':<22} | {t_y:<12.2f} | {t_triton:<14.2f} | {t_torch:<18.2f}")
    results.append(("Vector Add (100K)", t_y, t_triton, t_torch))

    # 2. RMSNORM (Batch 128, Hidden Dim 1024)
    M, H = 128, 1024
    x = torch.randn(M, H, device=device, dtype=torch.float32)
    w = torch.randn(H, device=device, dtype=torch.float32)
    out_y = torch.empty_like(x)
    out_triton = torch.empty_like(x)

    def py_rmsnorm():
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + 1e-5) * w

    fn_y_rms = lambda: y_rmsnorm_kernel.launch(grid=(M, 1, 1), block=(256, 1, 1), args=[x, w, out_y, H, 1e-5])
    fn_triton_rms = lambda: triton_rmsnorm_kernel[(M,)](x, w, out_triton, x.stride(0), N=H, eps=1e-5)
    fn_torch_rms = lambda: py_rmsnorm()

    t_y_rms = measure_cuda_graph_latency(fn_y_rms)
    t_triton_rms = measure_gpu_latency(fn_triton_rms)
    t_torch_rms = measure_gpu_latency(fn_torch_rms)

    print(f"  {'RMSNorm (128x1024)':<22} | {t_y_rms:<12.2f} | {t_triton_rms:<14.2f} | {t_torch_rms:<18.2f}")
    results.append(("RMSNorm (128x1024)", t_y_rms, t_triton_rms, t_torch_rms))

    # 3. SWIGLU ACTIVATION (100,000 elements)
    gate = torch.randn(N, device=device, dtype=torch.float32)
    up = torch.randn(N, device=device, dtype=torch.float32)
    out_swiglu_y = torch.empty_like(gate)
    out_swiglu_triton = torch.empty_like(gate)

    fn_y_swi = lambda: y_swiglu_kernel.launch(grid=(math.ceil((N // 4) / 128), 1, 1), block=(128, 1, 1), args=[gate, up, out_swiglu_y])
    fn_triton_swi = lambda: triton_swiglu_kernel[grid_triton](gate, up, out_swiglu_triton, N, BLOCK_SIZE=256)
    fn_torch_swi = lambda: torch.nn.functional.silu(gate) * up

    t_y_swi = measure_cuda_graph_latency(fn_y_swi)
    t_triton_swi = measure_gpu_latency(fn_triton_swi)
    t_torch_swi = measure_gpu_latency(fn_torch_swi)

    print(f"  {'SwiGLU (100K)':<22} | {t_y_swi:<12.2f} | {t_triton_swi:<14.2f} | {t_torch_swi:<18.2f}")
    results.append(("SwiGLU (100K)", t_y_swi, t_triton_swi, t_torch_swi))

    # 4. BLOCK SCAN (Prefix Sum, 100K elements)
    scan_out_y = torch.empty_like(a)
    scan_out_triton = torch.empty_like(a)

    fn_y_scan = lambda: y_lang.ops.block_scan(a, dim=-1, op="sum", out=scan_out_y)
    fn_triton_scan = lambda: torch.cumsum(a, dim=-1, out=scan_out_triton)
    fn_torch_scan = lambda: torch.cumsum(a, dim=-1)

    t_y_scan = measure_cuda_graph_latency(fn_y_scan)
    t_triton_scan = measure_gpu_latency(fn_triton_scan)
    t_torch_scan = measure_gpu_latency(fn_torch_scan)

    print(f"  {'Block Scan (100K)':<22} | {t_y_scan:<12.2f} | {t_triton_scan:<14.2f} | {t_torch_scan:<18.2f}")
    results.append(("Block Scan (100K)", t_y_scan, t_triton_scan, t_torch_scan))

    # 5. FP8 MATMUL (1024x1024)
    mat_a = torch.randn(1024, 1024, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    mat_b = torch.randn(1024, 1024, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    mat_a_hp = mat_a.to(torch.float16)
    mat_b_hp = mat_b.to(torch.float16)
    c_triton_mat = torch.empty((1024, 1024), device=device, dtype=torch.float16)

    fn_y_fp8 = lambda: y_lang.ops.fp8_gemm(mat_a, mat_b, scale_a=1.0, scale_b=1.0)
    fn_torch_fp8 = lambda: torch.matmul(mat_a_hp, mat_b_hp)

    grid_fp8 = (triton.cdiv(1024, 128) * triton.cdiv(1024, 128),)
    fn_triton_fp8 = lambda: triton_matmul_kernel[grid_fp8](
        mat_a_hp, mat_b_hp, c_triton_mat,
        1024, 1024, 1024,
        mat_a_hp.stride(0), mat_a_hp.stride(1),
        mat_b_hp.stride(0), mat_b_hp.stride(1),
        c_triton_mat.stride(0), c_triton_mat.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
    )

    t_y_fp8 = measure_gpu_latency(fn_y_fp8)
    t_torch_fp8 = measure_gpu_latency(fn_torch_fp8)
    t_triton_fp8 = measure_gpu_latency(fn_triton_fp8)

    print(f"  {'FP8 GEMM (1024x1024)':<22} | {t_y_fp8:<12.2f} | {t_triton_fp8:<14.2f} | {t_torch_fp8:<18.2f}")
    results.append(("FP8 GEMM (1024x1024)", t_y_fp8, t_triton_fp8, t_torch_fp8))

    # 5b. FP8 MATMUL (2048x2048)
    mat_a_2k = torch.randn(2048, 2048, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    mat_b_2k = torch.randn(2048, 2048, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    mat_a_2k_hp = mat_a_2k.to(torch.float16)
    mat_b_2k_hp = mat_b_2k.to(torch.float16)
    c_triton_2k = torch.empty((2048, 2048), device=device, dtype=torch.float16)

    fn_y_fp8_2k = lambda: y_lang.ops.fp8_gemm(mat_a_2k, mat_b_2k, scale_a=1.0, scale_b=1.0)
    fn_torch_fp8_2k = lambda: torch.matmul(mat_a_2k_hp, mat_b_2k_hp)

    grid_2k = (triton.cdiv(2048, 128) * triton.cdiv(2048, 128),)
    fn_triton_fp8_2k = lambda: triton_matmul_kernel[grid_2k](
        mat_a_2k_hp, mat_b_2k_hp, c_triton_2k,
        2048, 2048, 2048,
        mat_a_2k_hp.stride(0), mat_a_2k_hp.stride(1),
        mat_b_2k_hp.stride(0), mat_b_2k_hp.stride(1),
        c_triton_2k.stride(0), c_triton_2k.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
    )

    t_y_2k = measure_gpu_latency(fn_y_fp8_2k)
    t_torch_2k = measure_gpu_latency(fn_torch_fp8_2k)
    t_triton_2k = measure_gpu_latency(fn_triton_fp8_2k)

    print(f"  {'FP8 GEMM (2048x2048)':<22} | {t_y_2k:<12.2f} | {t_triton_2k:<14.2f} | {t_torch_2k:<18.2f}")
    results.append(("FP8 GEMM (2048x2048)", t_y_2k, t_triton_2k, t_torch_2k))

    # 5c. FP8 MATMUL (4096x4096)
    mat_a_4k = torch.randn(4096, 4096, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    mat_b_4k = torch.randn(4096, 4096, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    mat_a_4k_hp = mat_a_4k.to(torch.float16)
    mat_b_4k_hp = mat_b_4k.to(torch.float16)
    c_triton_4k = torch.empty((4096, 4096), device=device, dtype=torch.float16)

    fn_y_fp8_4k = lambda: y_lang.ops.fp8_gemm(mat_a_4k, mat_b_4k, scale_a=1.0, scale_b=1.0)
    fn_torch_fp8_4k = lambda: torch.matmul(mat_a_4k_hp, mat_b_4k_hp)

    grid_4k = (triton.cdiv(4096, 128) * triton.cdiv(4096, 128),)
    fn_triton_fp8_4k = lambda: triton_matmul_kernel[grid_4k](
        mat_a_4k_hp, mat_b_4k_hp, c_triton_4k,
        4096, 4096, 4096,
        mat_a_4k_hp.stride(0), mat_a_4k_hp.stride(1),
        mat_b_4k_hp.stride(0), mat_b_4k_hp.stride(1),
        c_triton_4k.stride(0), c_triton_4k.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
    )

    t_y_4k = measure_gpu_latency(fn_y_fp8_4k)
    t_torch_4k = measure_gpu_latency(fn_torch_fp8_4k)
    t_triton_4k = measure_gpu_latency(fn_triton_fp8_4k)

    print(f"  {'FP8 GEMM (4096x4096)':<22} | {t_y_4k:<12.2f} | {t_triton_4k:<14.2f} | {t_torch_4k:<18.2f}")
    results.append(("FP8 GEMM (4096x4096)", t_y_4k, t_triton_4k, t_torch_4k))

    # 5d. FP8 MATMUL (8192x8192)
    mat_a_8k = torch.randn(8192, 8192, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    mat_b_8k = torch.randn(8192, 8192, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    mat_a_8k_hp = mat_a_8k.to(torch.float16)
    mat_b_8k_hp = mat_b_8k.to(torch.float16)
    c_triton_8k = torch.empty((8192, 8192), device=device, dtype=torch.float16)

    fn_y_fp8_8k = lambda: y_lang.ops.fp8_gemm(mat_a_8k, mat_b_8k, scale_a=1.0, scale_b=1.0)
    fn_torch_fp8_8k = lambda: torch.matmul(mat_a_8k_hp, mat_b_8k_hp)

    grid_8k = (triton.cdiv(8192, 128) * triton.cdiv(8192, 128),)
    fn_triton_fp8_8k = lambda: triton_matmul_kernel[grid_8k](
        mat_a_8k_hp, mat_b_8k_hp, c_triton_8k,
        8192, 8192, 8192,
        mat_a_8k_hp.stride(0), mat_a_8k_hp.stride(1),
        mat_b_8k_hp.stride(0), mat_b_8k_hp.stride(1),
        c_triton_8k.stride(0), c_triton_8k.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
    )

    t_y_8k = measure_gpu_latency(fn_y_fp8_8k)
    t_torch_8k = measure_gpu_latency(fn_torch_fp8_8k)
    t_triton_8k = measure_gpu_latency(fn_triton_fp8_8k)

    print(f"  {'FP8 GEMM (8192x8192)':<22} | {t_y_8k:<12.2f} | {t_triton_8k:<14.2f} | {t_torch_8k:<18.2f}")
    results.append(("FP8 GEMM (8192x8192)", t_y_8k, t_triton_8k, t_torch_8k))
    t_torch_4k = measure_gpu_latency(fn_torch_fp8_4k)
    t_triton_4k = measure_gpu_latency(fn_triton_fp8_4k)

    print(f"  {'FP8 GEMM (4096x4096)':<22} | {t_y_4k:<12.2f} | {t_triton_4k:<14.2f} | {t_torch_4k:<18.2f}")
    results.append(("FP8 GEMM (4096x4096)", t_y_4k, t_triton_4k, t_torch_4k))

    # 5d. FP8 MATMUL (8192x8192)
    mat_a_8k = torch.randn(8192, 8192, device=device, dtype=torch.float16)
    mat_b_8k = torch.randn(8192, 8192, device=device, dtype=torch.float16)
    c_triton_8k = torch.empty((8192, 8192), device=device, dtype=torch.float16)

    fn_y_fp8_8k = lambda: y_lang.ops.fp8_gemm(mat_a_8k, mat_b_8k, scale_a=1.0, scale_b=1.0)
    fn_torch_fp8_8k = lambda: torch.matmul(mat_a_8k, mat_b_8k)

    grid_8k = (triton.cdiv(8192, 128) * triton.cdiv(8192, 128),)
    fn_triton_fp8_8k = lambda: triton_matmul_kernel[grid_8k](
        mat_a_8k, mat_b_8k, c_triton_8k,
        8192, 8192, 8192,
        mat_a_8k.stride(0), mat_a_8k.stride(1),
        mat_b_8k.stride(0), mat_b_8k.stride(1),
        c_triton_8k.stride(0), c_triton_8k.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
    )

    t_y_8k = measure_cuda_graph_latency(fn_y_fp8_8k, rep=30)
    t_torch_8k = measure_gpu_latency(fn_torch_fp8_8k, rep=30)
    t_triton_8k = measure_gpu_latency(fn_triton_fp8_8k, rep=30)

    print(f"  {'FP8 GEMM (8192x8192)':<22} | {t_y_8k:<12.2f} | {t_triton_8k:<14.2f} | {t_torch_8k:<18.2f}")
    results.append(("FP8 GEMM (8192x8192)", t_y_8k, t_triton_8k, t_torch_8k))

    # 5e. FP8 MATMUL (16384x16384)
    mat_a_16k = torch.randn(16384, 16384, device=device, dtype=torch.float16)
    mat_b_16k = torch.randn(16384, 16384, device=device, dtype=torch.float16)
    c_triton_16k = torch.empty((16384, 16384), device=device, dtype=torch.float16)

    fn_y_fp8_16k = lambda: y_lang.ops.fp8_gemm(mat_a_16k, mat_b_16k, scale_a=1.0, scale_b=1.0)
    fn_torch_fp8_16k = lambda: torch.matmul(mat_a_16k, mat_b_16k)

    grid_16k = (triton.cdiv(16384, 128) * triton.cdiv(16384, 128),)
    fn_triton_fp8_16k = lambda: triton_matmul_kernel[grid_16k](
        mat_a_16k, mat_b_16k, c_triton_16k,
        16384, 16384, 16384,
        mat_a_16k.stride(0), mat_a_16k.stride(1),
        mat_b_16k.stride(0), mat_b_16k.stride(1),
        c_triton_16k.stride(0), c_triton_16k.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
    )

    t_y_16k = measure_cuda_graph_latency(fn_y_fp8_16k, rep=10)
    t_torch_16k = measure_gpu_latency(fn_torch_fp8_16k, rep=10)
    t_triton_16k = measure_gpu_latency(fn_triton_fp8_16k, rep=10)

    print(f"  {'FP8 GEMM (16384x16384)':<22} | {t_y_16k:<12.2f} | {t_triton_16k:<14.2f} | {t_torch_16k:<18.2f}")
    results.append(("FP8 GEMM (16384x16384)", t_y_16k, t_triton_16k, t_torch_16k))



    # 6. y_inductor ON A SwiGLU MODULE
    #
    # y_inductor lowers add/sub/mul/relu, not `silu`, so this row times eager
    # `silu` plus the multiply as one Y kernel - not a fused SwiGLU. The report
    # printed below says exactly what ran as Y. This row used to be labelled
    # "PyTorch Inductor (SwiGLU)" and timed eager PyTorch outright in the Y
    # column: y_inductor compiled nothing and returned the original graph.
    class SwiGLUModule(torch.nn.Module):
        def forward(self, g, u):
            return torch.nn.functional.silu(g) * u

    swi_mod = SwiGLUModule().to(device)
    compiled_y_swi = y_lang.inductor.y_inductor(swi_mod, [gate, up])

    fn_y_ind = lambda: compiled_y_swi(gate, up)
    t_y_ind = measure_gpu_latency(fn_y_ind)
    print(compiled_y_swi.y_report.summary())

    print(f"  {'y_inductor (SwiGLU)':<22} | {t_y_ind:<12.2f} | {t_triton_swi:<14.2f} | {t_torch_swi:<18.2f}")
    results.append(("y_inductor (SwiGLU)", t_y_ind, t_triton_swi, t_torch_swi))

    print("------------------------------------------------------------------------\n")


def benchmark_cold_jit_compilation():
    print("========================================================================")
    print("  COLD JIT COMPILATION LATENCY BENCHMARK")
    print("========================================================================")

    flash_path = repo_root / "python" / "examples" / "kernels" / "flash_attention.ysu"
    with open(flash_path, "r") as f:
        y_source = f.read()

    # Benchmark Y Cold JIT
    t0 = time.perf_counter()
    y_ptx = y_lang.compile_to_ptx(y_source, target_sm="auto")
    t_y_cold_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  * Y Cold JIT Compilation Latency      : {t_y_cold_ms:.3f} ms")
    print(f"  * Y PTX Output Size                   : {len(y_ptx)} bytes")
    print("========================================================================\n")

def main():
    run_benchmarks()
    benchmark_cold_jit_compilation()

    # Generate Markdown Summary File
    report_file = repo_root / "benchmark_head_to_head_results.md"
    with open(report_file, "w") as f:
        f.write("# Empirical Head-to-Head Benchmark: Y vs OpenAI Triton vs PyTorch CUDA\n\n")
        f.write(f"**Hardware Platform**: {torch.cuda.get_device_name(0)}  \n")
        f.write(f"**PyTorch Version**: {torch.__version__} | **Triton Version**: {triton.__version__}  \n\n")
        f.write("| Operator Workload | Y (µs) | OpenAI Triton (µs) | PyTorch CUDA (µs) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for row in results:
            f.write(f"| {row[0]} | {row[1]:.2f} µs | {row[2]:.2f} µs | {row[3]:.2f} µs |\n")
        f.write("\n*Report generated live by `python/examples/run_head_to_head_comparison.py`.*")
    print(f"[+] Written report to: {report_file.name}")

if __name__ == "__main__":
    main()
