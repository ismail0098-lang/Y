# tests/benchmark_arbitrary_shapes.py
import os
import gc
os.environ["CUPY_CACHE_DIR"] = "/tmp/cupy_cache"

import sys
import torch
import cupy as cp
import numpy as np

CUDA_KERNELS_PATH = os.path.join(os.path.dirname(__file__), "y_tensor_core_gemm.cu")
with open(CUDA_KERNELS_PATH, "r") as f:
    CUDA_SRC = f.read()

def print_header(title):
    print("\n" + "=" * 90)
    print(f"{title:^90}")
    print("=" * 90)

def main():
    print_header("Y TENSOR CORE COMPILER — UNIVERSAL MATRIX SHAPE BENCHMARK SUITE")
    print(f"[*] Hardware GPU: {torch.cuda.get_device_name(0)}")
    print(f"[*] PyTorch Version: {torch.__version__} | CUDA: {torch.version.cuda}")

    device = torch.device("cuda:0")

    # Compile Y CuPy Module
    y_mod = cp.RawModule(code=CUDA_SRC, options=("-std=c++17", "--use_fast_math"))
    y_gemm_large = y_mod.get_function("y_tensor_core_gemm_kernel")
    y_gemm_large.max_dynamic_shared_size_bytes = 65536
    y_gemm_256x128 = y_mod.get_function("y_tensor_core_gemm_256x128_kernel")
    y_gemm_256x128.max_dynamic_shared_size_bytes = 98304
    y_gemm_small64 = y_mod.get_function("y_fused_gemm_small_64x64_kernel")
    y_gemm_micro = y_mod.get_function("y_fused_gemm_barrier_free_16x32_kernel")
    y_gemm_splitk = y_mod.get_function("y_fused_gemm_splitk_32x64_kernel")

    shape_categories = [
        ("CATEGORY 1: STANDARD POWER-OF-2 SHAPES", [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]),
        ("CATEGORY 2: UNALIGNED / ODD NON-POWER-OF-2 SHAPES", [
            (317, 511, 768),
            (768, 1024, 1536),
            (1337, 1337, 1337),
            (3000, 3000, 3000),
        ]),
        ("CATEGORY 3: LLM INFERENCE / TALL-AND-SKINNY SHAPES", [
            (1, 4096, 4096),
            (16, 4096, 4096),
        ]),
    ]

    for category_name, shapes in shape_categories:
        print_header(category_name)
        print(f"{'Shape (M x N x K)':<24} | {'cuBLAS (us)':<14} | {'Y Compiler (us)':<16} | {'TFLOPS (Y)':<14} | {'Parity':<8} | {'Speedup':<10}")
        print("-" * 90)

        for M, N, K in shapes:
            gc.collect()
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()

            iters = 300 if M*N <= 1024*1024 else (50 if M*N <= 4096*4096 else 10)
            warmup = 20

            A_torch = torch.randn(M, K, dtype=torch.float16, device=device)
            B_torch = torch.randn(K, N, dtype=torch.float16, device=device)
            C_ref = torch.matmul(A_torch, B_torch)

            A_cp = cp.asarray(A_torch)
            B_cp = cp.asarray(B_torch)
            C_y = cp.zeros((M, N), dtype=cp.float16)

            # cuBLAS Measurement
            torch.cuda.synchronize()
            for _ in range(warmup):
                _ = torch.matmul(A_torch, B_torch)
            torch.cuda.synchronize()

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(iters):
                _ = torch.matmul(A_torch, B_torch)
            end_evt.record()
            torch.cuda.synchronize()
            cublas_us = (start_evt.elapsed_time(end_evt) / iters) * 1000.0

            # Y Kernel Selection & Execution
            smem_size = 0
            is_unaligned = (M % 64 != 0) or (N % 64 != 0) or (K % 16 != 0)

            if is_unaligned:
                grid_m = (M + 127) // 128
                grid_n = (N + 127) // 128
                threads = 256
                target_kernel = y_gemm_large
                smem_size = 65536
            elif M <= 16:
                grid_m = (M + 31) // 32
                grid_n = (N + 63) // 64
                threads = 64
                target_kernel = y_gemm_splitk
            elif M <= 256 and N <= 256:
                grid_m = (M + 15) // 16
                grid_n = (N + 31) // 32
                threads = 32
                target_kernel = y_gemm_micro
            elif M <= 512 and N <= 512:
                grid_m = (M + 63) // 64
                grid_n = (N + 63) // 64
                threads = 128
                target_kernel = y_gemm_small64
            elif M >= 4096 and N >= 4096:
                grid_m = (M + 255) // 256
                grid_n = (N + 127) // 128
                threads = 256
                target_kernel = y_gemm_256x128
                smem_size = 98304
            else:
                grid_m = (M + 127) // 128
                grid_n = (N + 127) // 128
                threads = 256
                target_kernel = y_gemm_large
                smem_size = 65536

            cp.cuda.Device(0).synchronize()
            for _ in range(warmup):
                C_y.fill(0)
                target_kernel((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K), shared_mem=smem_size)
            cp.cuda.Device(0).synchronize()

            y_start = cp.cuda.Event()
            y_end = cp.cuda.Event()
            y_start.record()
            for _ in range(iters):
                target_kernel((grid_n, grid_m, 1), (threads, 1, 1), (A_cp, B_cp, C_y, M, N, K), shared_mem=smem_size)
            y_end.record()
            y_end.synchronize()
            y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / iters) * 1000.0

            # Parity Check
            C_y_torch = torch.from_dlpack(C_y)
            parity_pass = torch.allclose(C_y_torch, C_ref, atol=1e-1, rtol=1e-1)
            parity_str = "PASSED" if parity_pass else "WARN"

            total_flops = 2.0 * M * N * K
            tflops = (total_flops / (y_us * 1e-6)) / 1e12
            speedup = cublas_us / y_us

            shape_str = f"{M} x {N} x {K}"
            print(f"{shape_str:<24} | {cublas_us:<14.2f} | {y_us:<16.2f} | {tflops:<14.2f} | {parity_str:<8} | {speedup:<10.2f}x")

if __name__ == "__main__":
    main()
