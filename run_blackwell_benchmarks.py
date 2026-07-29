import os
import sys

# Auto-detect virtual environment if torch or cupy is missing
try:
    import torch
    import cupy as cp
except ModuleNotFoundError:
    possible_venvs = [
        os.path.join(os.path.dirname(__file__), "venv/bin/python3"),
        os.path.join(os.path.dirname(__file__), "../venv/bin/python3"),
        os.path.join(os.getcwd(), "venv/bin/python3"),
        os.path.join(os.getcwd(), "../venv/bin/python3"),
    ]
    venv_py = None
    for v in possible_venvs:
        if os.path.exists(v):
            venv_py = v
            break
    if venv_py and sys.executable != venv_py:
        print(f"[*] Auto-activating virtual environment: {venv_py}")
        os.execv(venv_py, [venv_py] + sys.argv)
    else:
        print("[!] PyTorch / CuPy not found in python environment.")
        print("[!] Please activate your virtual environment or run with your python environment:")
        print("    source venv/bin/activate && python3 run_blackwell_benchmarks.py")
        print("    or: ./venv/bin/python3 run_blackwell_benchmarks.py")
        sys.exit(1)

import gc
import numpy as np

CUDA_KERNELS_PATH = os.path.join(os.path.dirname(__file__), "tests/y_tensor_core_gemm.cu")
with open(CUDA_KERNELS_PATH, "r") as f:
    CUDA_SRC = f.read()

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Triton Fused GEMM + Bias + ReLU Kernel
if HAS_TRITON:
    @triton.jit
    def triton_fused_bias_relu_kernel(
        a_ptr, b_ptr, bias_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        acc = acc + bias[None, :]
        acc = tl.maximum(acc, 0.0)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)

def run_benchmarks():
    device_name = torch.cuda.get_device_name(0)
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    compute_cap = f"{cap_major}.{cap_minor}"
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print("=" * 90)
    print("      Y LANGUAGE COMPILER — BLACKWELL / NEXT-GEN GPU BENCHMARK SUITE")
    print("=" * 90)
    print(f"[*] GPU Device:         {device_name}")
    print(f"[*] Compute Capability: {compute_cap}")
    print(f"[*] Total VRAM:         {total_mem_gb:.2f} GB")
    print(f"[*] PyTorch / CUDA:     {torch.__version__} / {torch.version.cuda}")
    print(f"[*] Triton Available:   {HAS_TRITON}")
    print("=" * 90)

    # NVRTC Compile Options
    compile_options = ["-std=c++17", "--use_fast_math"]
    if cap_major >= 10:
        print("[*] Blackwell GPU detected (SM 10.0+)! Enabling Blackwell architecture targets.")
        compile_options.append(f"-arch=compute_{cap_major}0")

    y_mod = cp.RawModule(code=CUDA_SRC, options=tuple(compile_options))
    y_gemm_large = y_mod.get_function("y_tensor_core_gemm_kernel")
    y_splitk_ws = y_mod.get_function("y_fused_gemm_splitk_workspace_kernel")
    y_splitk_red = y_mod.get_function("y_splitk_reduction_kernel")
    y_fused_bias_relu = y_mod.get_function("y_fused_gemm_bias_relu_fp16_kernel")

    report_lines = [
        f"# Blackwell / Next-Gen GPU Benchmark Report",
        f"- **GPU Hardware**: `{device_name}`",
        f"- **Compute Capability**: `sm_{cap_major}{cap_minor}`",
        f"- **VRAM**: `{total_mem_gb:.2f} GB`",
        f"- **PyTorch / CUDA**: `{torch.__version__}` / `{torch.version.cuda}`",
        f"",
        "## 1. Standalone Dense GEMM Benchmarks (M=N=K)",
        "| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |",
        "|---|:---:|:---:|:---:|:---:|:---:|"
    ]

    # --- SUITE 1: Standalone Dense GEMM ---
    print("\n[+] SUITE 1: STANDALONE DENSE MATRIX MULTIPLICATION (FP16)")
    print("-" * 90)
    print(f"{'Matrix (M=N=K)':<18} | {'cuBLAS (us)':<12} | {'Y Compiler (us)':<16} | {'Y TFLOPS':<10} | {'Speedup':<10} | {'Parity':<8}")
    print("-" * 90)

    gemm_sizes = [512, 1024, 2048, 4096, 8192, 16384]
    for dim in gemm_sizes:
        gc.collect()
        torch.cuda.empty_cache()

        M = N = K = dim
        iters = 50 if dim <= 2048 else (15 if dim <= 4096 else (5 if dim <= 8192 else 2))
        warmup = 10 if dim <= 2048 else (3 if dim <= 8192 else 1)

        A_t = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B_t = torch.randn(K, N, dtype=torch.float16, device="cuda")
        C_ref = torch.matmul(A_t, B_t)

        A_cp = cp.asarray(A_t)
        B_cp = cp.asarray(B_t)
        C_y = cp.zeros((M, N), dtype=cp.float16)

        # Warmup cuBLAS
        for _ in range(warmup):
            _ = torch.matmul(A_t, B_t)
        torch.cuda.synchronize()

        start_c = torch.cuda.Event(enable_timing=True)
        end_c = torch.cuda.Event(enable_timing=True)
        start_c.record()
        for _ in range(iters):
            _ = torch.matmul(A_t, B_t)
        end_c.record()
        torch.cuda.synchronize()
        cublas_us = (start_c.elapsed_time(end_c) / float(iters)) * 1000.0

        grid_m = (M + 127) // 128
        grid_n = (N + 127) // 128
        threads = 256

        # Warmup Y
        for _ in range(50):
            y_gemm_large((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K))
        cp.cuda.Device(0).synchronize()

        y_start = cp.cuda.Event()
        y_end = cp.cuda.Event()
        y_start.record()
        for _ in range(iters):
            y_gemm_large((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K))
        y_end.record()
        y_end.synchronize()
        y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / float(iters)) * 1000.0

        C_y_torch = torch.from_dlpack(C_y)
        is_close = torch.allclose(C_y_torch, C_ref, atol=1e-1, rtol=1e-1)
        parity = "PASSED" if is_close else "WARN"

        tflops = (2.0 * M * N * K) / (y_us * 1e-6) / 1e12
        speedup = cublas_us / y_us

        print(f"{M}x{N}x{K:<12} | {cublas_us:<12.2f} | {y_us:<16.2f} | {tflops:<10.2f} | {speedup:<10.2f}x | {parity:<8}", flush=True)
        report_lines.append(f"| {M}x{N}x{K} | {cublas_us:.2f} | {y_us:.2f} | {tflops:.2f} | {speedup:.2f}x | {parity} |")

    # --- SUITE 2: Real-World LLM Inference Shapes ---
    report_lines.extend([
        "",
        "## 2. Real-World LLM Inference & Prompt Decoding Shapes",
        "| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|"
    ])
    print("\n[+] SUITE 2: REAL-WORLD LLM INFERENCE & PROMPT DECODING SHAPES")
    print("-" * 90)
    print(f"{'Shape (M x N x K)':<20} | {'Workload':<24} | {'cuBLAS (us)':<12} | {'Y Split-K (us)':<14} | {'GB/s':<8} | {'Speedup':<8}")
    print("-" * 90)

    llm_shapes = [
        (1, 4096, 4096, "LLaMA 7B Single-Token Decode"),
        (1, 11008, 4096, "LLaMA 7B SwiGLU FFN Gate/Up"),
        (16, 4096, 4096, "Batch 16 Prompt Evaluation"),
        (32, 4096, 4096, "Batch 32 Prompt Evaluation"),
    ]

    for M, N, K, label in llm_shapes:
        gc.collect()
        torch.cuda.empty_cache()

        A_t = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B_t = torch.randn(K, N, dtype=torch.float16, device="cuda")
        C_ref = torch.matmul(A_t, B_t)

        A_cp = cp.asarray(A_t)
        B_cp = cp.asarray(B_t)
        C_y = cp.zeros((M, N), dtype=cp.float16)

        # Warmup cuBLAS
        for _ in range(10):
            _ = torch.matmul(A_t, B_t)
        torch.cuda.synchronize()

        start_c = torch.cuda.Event(enable_timing=True)
        end_c = torch.cuda.Event(enable_timing=True)
        start_c.record()
        for _ in range(50):
            _ = torch.matmul(A_t, B_t)
        end_c.record()
        torch.cuda.synchronize()
        cublas_us = (start_c.elapsed_time(end_c) / 50.0) * 1000.0

        k_splits = 16 if M == 1 else 4
        workspace = cp.zeros((k_splits, M, N), dtype=cp.float32)
        grid_m = (M + 31) // 32
        grid_n = (N + 63) // 64
        threads = 128
        total_elems = M * N
        red_blocks = (total_elems + 255) // 256

        # Warmup Y
        for _ in range(10):
            workspace.fill(0)
            y_splitk_ws((grid_n, grid_m, k_splits), (threads, 1, 1), (A_cp, B_cp, workspace, M, N, K, k_splits))
            y_splitk_red((red_blocks, 1, 1), (256, 1, 1), (workspace, C_y, total_elems, M, N, k_splits))
        cp.cuda.Device(0).synchronize()

        y_start = cp.cuda.Event()
        y_end = cp.cuda.Event()
        y_start.record()
        for _ in range(50):
            workspace.fill(0)
            y_splitk_ws((grid_n, grid_m, k_splits), (threads, 1, 1), (A_cp, B_cp, workspace, M, N, K, k_splits))
            y_splitk_red((red_blocks, 1, 1), (256, 1, 1), (workspace, C_y, total_elems, M, N, k_splits))
        y_end.record()
        y_end.synchronize()
        y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / 50.0) * 1000.0

        C_y_torch = torch.from_dlpack(C_y)
        is_close = torch.allclose(C_y_torch, C_ref, atol=1e-1, rtol=1e-1)
        parity = "PASSED" if is_close else "WARN"

        bytes_loaded = 2.0 * (M * K + K * N + M * N)
        bandwidth_gbps = (bytes_loaded / (y_us * 1e-6)) / 1e9
        speedup = cublas_us / y_us

        shape_str = f"{M}x{N}x{K}"
        print(f"{shape_str:<20} | {label:<24} | {cublas_us:<12.2f} | {y_us:<14.2f} | {bandwidth_gbps:<8.1f} | {speedup:<8.2f}x")
        sys.stdout.flush()
        report_lines.append(f"| {shape_str} | {label} | {cublas_us:.2f} | {y_us:.2f} | {bandwidth_gbps:.1f} GB/s | {speedup:.2f}x | {parity} |")

    # --- SUITE 3: Fused Layers vs PyTorch Fused (FP16 Fused GEMM + Bias + ReLU) ---
    report_lines.extend([
        "",
        "## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)",
        "| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |",
        "|---|:---:|:---:|:---:|"
    ])
    print("\n[+] SUITE 3: FUSED NEURAL NETWORK LAYERS (GEMM + BIAS + RELU)")
    print("-" * 90)
    print(f"{'Matrix (M=N=K)':<18} | {'PyTorch Fused (us)':<18} | {'Y Compiler Fused (us)':<22} | {'Speedup':<10}")
    print("-" * 90)

    for dim in [512, 1024, 2048, 4096]:
        M = N = K = dim
        iters = 50 if dim <= 2048 else 15

        A_t = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B_t = torch.randn(K, N, dtype=torch.float16, device="cuda")
        bias_t = torch.randn(N, dtype=torch.float16, device="cuda")

        A_cp = cp.asarray(A_t)
        B_cp = cp.asarray(B_t)
        bias_cp = cp.asarray(bias_t)
        C_y = cp.zeros((M, N), dtype=cp.float16)

        # Warmup PyTorch Fused
        for _ in range(10):
            _ = torch.relu(torch.matmul(A_t, B_t) + bias_t)
        torch.cuda.synchronize()

        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        for _ in range(iters):
            _ = torch.relu(torch.matmul(A_t, B_t) + bias_t)
        t_end.record()
        torch.cuda.synchronize()
        pt_fused_us = (t_start.elapsed_time(t_end) / float(iters)) * 1000.0

        grid_m = (M + 127) // 128
        grid_n = (N + 127) // 128
        threads = 256

        # Warmup Y Fused
        for _ in range(10):
            y_fused_bias_relu((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, bias_cp, C_y, M, N, K, 1))
        cp.cuda.Device(0).synchronize()

        y_start = cp.cuda.Event()
        y_end = cp.cuda.Event()
        y_start.record()
        for _ in range(iters):
            y_fused_bias_relu((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, bias_cp, C_y, M, N, K, 1))
        y_end.record()
        y_end.synchronize()
        y_fused_us = (cp.cuda.get_elapsed_time(y_start, y_end) / float(iters)) * 1000.0

        speedup = pt_fused_us / y_fused_us
        print(f"{M}x{N}x{K:<12} | {pt_fused_us:<18.2f} | {y_fused_us:<22.2f} | {speedup:<10.2f}x", flush=True)
        report_lines.append(f"| {M}x{N}x{K} | {pt_fused_us:.2f} | {y_fused_us:.2f} | {speedup:.2f}x |")

    print("\n" + "=" * 90)
    print("[*] Benchmark Run Complete!")
    print("=" * 90)

    # Save artifact report
    report_content = "\n".join(report_lines)
    artifact_path = os.path.join(os.path.dirname(__file__), "blackwell_benchmark_report.md")
    with open(artifact_path, "w") as f:
        f.write(report_content)
    print(f"[*] Benchmark report saved to: {artifact_path}\n")

if __name__ == "__main__":
    run_benchmarks()
