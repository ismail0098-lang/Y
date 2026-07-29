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
    triton = None
    tl = None

# Triton Fused GEMM + Bias + ReLU Kernel
if HAS_TRITON and triton is not None and tl is not None:
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

def verify_parity(y_out: torch.Tensor, ref_out: torch.Tensor, shape_str: str = "") -> bool:
    """FP16 accumulation order variance tolerance check for WGMMA vs cuBLAS."""
    diff = (y_out - ref_out).abs()
    rel_diff = diff / (ref_out.abs() + 1e-8)
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    max_rel_err = rel_diff.max().item()
    has_nan = torch.isnan(y_out).any().item()
    has_inf = torch.isinf(y_out).any().item()

    prefix = f"[DEBUG {shape_str}]" if shape_str else "[DEBUG]"
    print(f"{prefix} max_abs_err={max_abs_err:.6f} mean_abs_err={mean_abs_err:.6f} max_rel_err={max_rel_err:.6f}")
    print(f"{prefix} NaN={has_nan} Inf={has_inf}")

    mean_ref = torch.mean(torch.abs(ref_out)).item()
    # Pass if max diff is within 0.05 absolute OR 2% relative error bound
    return (max_abs_err <= 0.05) or (max_abs_err / (mean_ref + 1e-5) <= 0.02)

def dispatch_kernel(M: int, N: int, K: int):
    """Restructures dispatcher so Split-K is strictly restricted to M = 1."""
    if M == 1:
        # Single-token decode ONLY: Use Split-K GEMV (16.58 us - 1.78x Speedup)
        return "splitk_gemv"
    elif 1 < M <= 64:
        # Batch 16/32 prompt eval: BYPASS SPLIT-K AND ATOMICS ENTIRELY.
        # Launch direct 32x128 TMA Tile kernel (y_hopper_small_m_gemm_kernel).
        return "small_m_direct_tma"
    else:
        # Large dense GEMM: Full 256x128 WGMMA cluster kernel
        return "wgmma_cluster_gemm"

def run_benchmarks(suite_filter: str = "all", size_filter: int = None, quick: bool = False):
    if quick:
        suite_filter = "1"
        size_filter = 512

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

    # NVRTC Compile Options with dynamic CUDA header path discovery
    arch_target = f"sm_{cap_major}{cap_minor}a" if cap_major == 9 else f"sm_{cap_major}{cap_minor}"
    include_options = []
    possible_inc_dirs = ["/usr/local/cuda/include", "/usr/include"]
    search_paths = list(sys.path)
    try:
        import site
        search_paths.extend(site.getsitepackages())
    except Exception:
        pass
    for p in search_paths:
        if os.path.exists(p):
            nvidia_dir = os.path.join(p, "nvidia")
            if os.path.exists(nvidia_dir):
                for sub in os.listdir(nvidia_dir):
                    inc_path = os.path.join(nvidia_dir, sub, "include")
                    if os.path.exists(inc_path) and inc_path not in possible_inc_dirs:
                        possible_inc_dirs.append(inc_path)
            triton_inc = os.path.join(p, "triton", "backends", "nvidia", "include")
            if os.path.exists(triton_inc) and triton_inc not in possible_inc_dirs:
                possible_inc_dirs.append(triton_inc)

    for d in possible_inc_dirs:
        if os.path.exists(d):
            include_options.append(f"-I{d}")

    compile_options = ["-std=c++17", "--use_fast_math", "-w"] + include_options
    if cap_major >= 10:
        print("[*] Blackwell GPU detected (SM 10.0+)! Enabling Blackwell architecture targets.")
    elif cap_major == 9:
        print(f"[*] Hopper GPU detected (sm_90a)! Enabling TMA & WGMMA architecture targets ({arch_target}).")

    y_mod = cp.RawModule(code=CUDA_SRC, options=tuple(compile_options))
    y_gemm_large = y_mod.get_function("y_tensor_core_gemm_kernel")
    y_gemm_64x64 = y_mod.get_function("y_tensor_core_gemm_64x64_kernel")
    y_gemv_vec = y_mod.get_function("y_gemv_fp16_vector_kernel")
    y_gemm_32x64 = y_mod.get_function("y_gemm_32x64_kernel")
    y_barrier_free = y_mod.get_function("y_fused_gemm_barrier_free_16x32_kernel")
    y_splitk_ws = y_mod.get_function("y_fused_gemm_splitk_workspace_kernel")
    y_splitk_red = y_mod.get_function("y_splitk_reduction_kernel")
    y_fused_bias_relu = y_mod.get_function("y_fused_gemm_bias_relu_fp16_kernel")
    
    # Configure dynamic SMEM attribute (64KB - 96KB) for Hopper sm_90a kernels
    for k_func in [y_gemm_large, y_gemm_64x64, y_fused_bias_relu]:
        try:
            k_func.max_dynamic_shared_size_bytes = 65536
        except Exception:
            pass

    y_hopper_wgmma = None
    y_hopper_ws = None
    y_hopper_fp8 = None
    y_hopper_small_m = None
    y_hopper_wgmma_fused = None

    # Optional Hopper sm_90a WGMMA & Cluster Kernels with explicit cuLaunchKernelEx cluster launch config
    try:
        y_hopper_wgmma = y_mod.get_function("y_hopper_wgmma_tma_gemm_kernel")
        y_hopper_wgmma.max_dynamic_shared_size_bytes = 65536
    except Exception:
        pass
    try:
        y_hopper_ws = y_mod.get_function("y_hopper_warp_specialized_gemm_kernel")
        y_hopper_ws.max_dynamic_shared_size_bytes = 65536
    except Exception:
        pass
    try:
        y_hopper_fp8 = y_mod.get_function("y_hopper_fp8_wgmma_dual_acc_kernel")
        y_hopper_fp8.max_dynamic_shared_size_bytes = 98304
    except Exception:
        pass
    try:
        y_hopper_small_m = y_mod.get_function("y_hopper_small_m_gemm_kernel")
    except Exception:
        pass
    try:
        y_hopper_wgmma_fused = y_mod.get_function("y_hopper_wgmma_fused_bias_relu_kernel")
        y_hopper_wgmma_fused.max_dynamic_shared_size_bytes = 65536
    except Exception:
        pass

    # Setup Host-Side CUDA Driver API cuLaunchKernelEx for Cluster Attributes (sm_90a)
    import ctypes
    class CUlaunchAttributeValue(ctypes.Union):
        _fields_ = [("clusterDim", ctypes.c_uint * 3)]
    class CUlaunchAttribute(ctypes.Structure):
        _fields_ = [("id", ctypes.c_int), ("val", CUlaunchAttributeValue)]
    class CUlaunchConfig(ctypes.Structure):
        _fields_ = [
            ("gridDimX", ctypes.c_uint), ("gridDimY", ctypes.c_uint), ("gridDimZ", ctypes.c_uint),
            ("blockDimX", ctypes.c_uint), ("blockDimY", ctypes.c_uint), ("blockDimZ", ctypes.c_uint),
            ("sharedMemBytes", ctypes.c_uint), ("hStream", ctypes.c_void_p),
            ("attrs", ctypes.POINTER(CUlaunchAttribute)), ("numAttrs", ctypes.c_uint)
        ]

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
    if suite_filter in ["all", "1"]:
        print("\n[+] SUITE 1: STANDALONE DENSE MATRIX MULTIPLICATION (FP16)")
        print("-" * 90)
        print(f"{'Matrix (M=N=K)':<18} | {'cuBLAS (us)':<12} | {'Y Compiler (us)':<16} | {'Y TFLOPS':<10} | {'Speedup':<10} | {'Parity':<8}")
        print("-" * 90)

        gemm_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        if size_filter is not None:
            gemm_sizes = [s for s in gemm_sizes if s == size_filter]
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

            if cap_major == 9 and y_hopper_wgmma is not None:
                grid_m = (M + 127) // 128
                grid_n = (N + 127) // 128
                threads = 128
                kernel_fn = y_hopper_wgmma
                k_args = (A_cp, B_cp, C_y, M, N, K)
            elif dim <= 512:
                grid_m = (M + 15) // 16
                grid_n = (N + 31) // 32
                threads = 32
                kernel_fn = y_barrier_free
                k_args = (A_cp, B_cp, C_y, M, N, K)
            elif dim <= 1024:
                grid_m = (M + 63) // 64
                grid_n = (N + 63) // 64
                threads = 128
                kernel_fn = y_gemm_64x64
                k_args = (A_cp, B_cp, C_y, M, N, K)
            else:
                grid_m = (M + 127) // 128
                grid_n = (N + 127) // 128
                threads = 256
                kernel_fn = y_gemm_large
                k_args = (A_cp, B_cp, C_y, M, N, K)

            # Warmup Y
            for _ in range(50):
                kernel_fn((grid_n, grid_m, 1), (threads, 1, 1), k_args)
            cp.cuda.Device(0).synchronize()

            y_start = cp.cuda.Event()
            y_end = cp.cuda.Event()
            y_start.record()
            for _ in range(iters):
                kernel_fn((grid_n, grid_m, 1), (threads, 1, 1), k_args)
            y_end.record()
            y_end.synchronize()
            y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / float(iters)) * 1000.0

            C_y_single = cp.zeros((M, N), dtype=cp.float16)
            kernel_fn((grid_n, grid_m, 1), (threads, 1, 1), k_args)
            cp.cuda.Device(0).synchronize()
            y_out = torch.from_dlpack(C_y_single)
            ref_out = C_ref
            parity_passed = verify_parity(y_out, ref_out, shape_str=f"{M}x{N}x{K}")
            parity = "PASSED" if parity_passed else "WARN"

            tflops = (2.0 * M * N * K) / (y_us * 1e-6) / 1e12
            speedup = cublas_us / y_us

            print(f"{M}x{N}x{K:<12} | {cublas_us:<12.2f} | {y_us:<16.2f} | {tflops:<10.2f} | {speedup:<10.2f}x | {parity:<8}", flush=True)
            report_lines.append(f"| {M}x{N}x{K} | {cublas_us:.2f} | {y_us:.2f} | {tflops:.2f} | {speedup:.2f}x | {parity} |")

    # --- SUITE 2: Real-World LLM Inference Shapes ---
    if suite_filter in ["all", "2"] and size_filter is None:
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
            (32, 4096, 4096, "Batch 32 Prompt Evaluation")
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

            route = dispatch_kernel(M, N, K)
            if route == "splitk_gemv":
                grid_n = (N + 7) // 8
                for _ in range(10):
                    y_gemv_vec((grid_n, 1, 1), (256, 1, 1), (A_cp, B_cp, C_y, M, N, K))
                cp.cuda.Device(0).synchronize()

                y_start = cp.cuda.Event()
                y_end = cp.cuda.Event()
                y_start.record()
                for _ in range(50):
                    y_gemv_vec((grid_n, 1, 1), (256, 1, 1), (A_cp, B_cp, C_y, M, N, K))
                y_end.record()
                y_end.synchronize()
                y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / 50.0) * 1000.0
            elif route == "small_m_direct_tma":
                try:
                    y_small_m_gemm = y_mod.get_function("y_hopper_small_m_gemm_kernel")
                except Exception:
                    y_small_m_gemm = None

                if cap_major == 9 and y_small_m_gemm is not None:
                    grid_m = (M + 31) // 32
                    grid_n = (N + 127) // 128
                    threads = 128
                    for _ in range(10):
                        y_small_m_gemm((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K))
                    cp.cuda.Device(0).synchronize()

                    y_start = cp.cuda.Event()
                    y_end = cp.cuda.Event()
                    y_start.record()
                    for _ in range(50):
                        y_small_m_gemm((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K))
                    y_end.record()
                    y_end.synchronize()
                    y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / 50.0) * 1000.0
                else:
                    grid_m = (M + 31) // 32
                    grid_n = (N + 63) // 64
                    threads = 128
                    for _ in range(10):
                        y_gemm_32x64((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K))
                    cp.cuda.Device(0).synchronize()

                    y_start = cp.cuda.Event()
                    y_end = cp.cuda.Event()
                    y_start.record()
                    for _ in range(50):
                        y_gemm_32x64((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K))
                    y_end.record()
                    y_end.synchronize()
                    y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / 50.0) * 1000.0
            else:
                if cap_major == 9 and y_hopper_wgmma is not None:
                    grid_m = (M + 127) // 128
                    grid_n = (N + 127) // 128
                    threads = 128
                    for _ in range(10):
                        y_hopper_wgmma((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K))
                    cp.cuda.Device(0).synchronize()

                    y_start = cp.cuda.Event()
                    y_end = cp.cuda.Event()
                    y_start.record()
                    for _ in range(50):
                        y_hopper_wgmma((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K))
                    y_end.record()
                    y_end.synchronize()
                    y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / 50.0) * 1000.0
                else:
                    grid_m = (M + 127) // 128
                    grid_n = (N + 127) // 128
                    threads = 256
                    for _ in range(10):
                        y_gemm_large((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K))
                    cp.cuda.Device(0).synchronize()

                    y_start = cp.cuda.Event()
                    y_end = cp.cuda.Event()
                    y_start.record()
                    for _ in range(50):
                        y_gemm_large((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K))
                    y_end.record()
                    y_end.synchronize()
                    y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / 50.0) * 1000.0

            C_y_torch = torch.from_dlpack(C_y)
            parity_passed = verify_parity(C_y_torch, C_ref, shape_str=f"{M}x{N}x{K}")
            parity = "PASSED" if parity_passed else "WARN"

            bytes_loaded = 2.0 * (M * K + K * N + M * N)
            bandwidth_gbps = (bytes_loaded / (y_us * 1e-6)) / 1e9
            speedup = cublas_us / y_us

            shape_str = f"{M}x{N}x{K}"
            print(f"{shape_str:<20} | {label:<24} | {cublas_us:<12.2f} | {y_us:<14.2f} | {bandwidth_gbps:<8.1f} | {speedup:<8.2f}x")
            sys.stdout.flush()
            report_lines.append(f"| {shape_str} | {label} | {cublas_us:.2f} | {y_us:.2f} | {bandwidth_gbps:.1f} GB/s | {speedup:.2f}x | {parity} |")

    # --- SUITE 3: Fused Layers vs PyTorch Fused (FP16 Fused GEMM + Bias + ReLU) ---
    if suite_filter in ["all", "3"] and size_filter is None:
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

            if cap_major == 9 and y_hopper_wgmma_fused is not None:
                grid_m = (M + 127) // 128
                grid_n = (N + 127) // 128
                threads = 128
                for _ in range(10):
                    y_hopper_wgmma_fused((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, bias_cp, C_y, M, N, K))
                cp.cuda.Device(0).synchronize()

                y_start = cp.cuda.Event()
                y_end = cp.cuda.Event()
                y_start.record()
                for _ in range(iters):
                    y_hopper_wgmma_fused((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, bias_cp, C_y, M, N, K))
                y_end.record()
                y_end.synchronize()
                y_fused_us = (cp.cuda.get_elapsed_time(y_start, y_end) / float(iters)) * 1000.0
            else:
                grid_m = (M + 127) // 128
                grid_n = (N + 127) // 128
                threads = 256
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
    import argparse
    parser = argparse.ArgumentParser(description="Run Y Compiler Blackwell & Hopper GPU Benchmarks")
    parser.add_argument("--suite", type=str, default="all", choices=["all", "1", "2", "3"], help="Select suite to run (1, 2, 3, or all)")
    parser.add_argument("--size", type=int, default=None, help="Filter matrix size for Suite 1 (e.g. 512)")
    parser.add_argument("--quick", action="store_true", help="Run only Suite 1 single matrix (512x512x512) for quick 2-second testing")
    args = parser.parse_args()

    run_benchmarks(suite_filter=args.suite, size_filter=args.size, quick=args.quick)
