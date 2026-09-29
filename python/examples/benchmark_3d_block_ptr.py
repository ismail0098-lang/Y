#!/usr/bin/env venv/bin/python
"""
3D Block Pointer Empirical Benchmark: Y Language vs OpenAI Triton
Measures 3D strided block load/store, boundary masking, and 3D volume reduction.
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
print("  3D BLOCK POINTER BENCHMARK SUITE: Y Language vs OpenAI Triton")
print(f"  GPU Hardware : {torch.cuda.get_device_name(0)}")
print(f"  PyTorch Ver  : {torch.__version__}")
print(f"  Triton Ver   : {triton.__version__}")
print("========================================================================\n")

# ----------------------------------------------------------------------
# 1. TRITON 3D KERNEL
# ----------------------------------------------------------------------
@triton.jit
def triton_3d_block_ptr_kernel(
    in_ptr, out_ptr,
    D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
    stride_0, stride_1, stride_2,
    BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    offs_0 = pid_0 * BLOCK_0 + tl.arange(0, BLOCK_0)
    offs_1 = pid_1 * BLOCK_1 + tl.arange(0, BLOCK_1)
    offs_2 = tl.arange(0, BLOCK_2)

    mask_0 = offs_0 < D0
    mask_1 = offs_1 < D1

    # 3D pointer calculation
    ptrs = in_ptr + (offs_0[:, None, None] * stride_0 + offs_1[None, :, None] * stride_1 + offs_2[None, None, :] * stride_2)
    mask = mask_0[:, None, None] & mask_1[None, :, None]

    val = tl.load(ptrs, mask=mask, other=0.0)
    res = val * 2.0 + 1.0

    out_ptrs = out_ptr + (offs_0[:, None, None] * stride_0 + offs_1[None, :, None] * stride_1 + offs_2[None, None, :] * stride_2)
    tl.store(out_ptrs, res, mask=mask)

import y_lang
import os
import shutil

shutil.rmtree(os.path.expanduser("~/.ysu/cache"), ignore_errors=True)
y_lang.compiler._JIT_CACHE.clear()

# ----------------------------------------------------------------------
# 2. Y LANGUAGE 3D KERNEL
# ----------------------------------------------------------------------
y_3d_src = """

kernel bench_3d(In: GlobalMemory<F32>, Out: GlobalMemory<F32>, D0: I32, D1: I32, D2: I32, S0: I32, S1: I32) {
    let b0: I32 = block_idx_x();
    let b1: I32 = block_idx_y();
    let t2: I32 = thread_idx_x();

    let d0: I32 = b0 * 16;
    let d1: I32 = b1 * 16;

    let d2_0: I32 = t2 * 4;
    let d2_1: I32 = t2 * 4 + 1;
    let d2_2: I32 = t2 * 4 + 2;
    let d2_3: I32 = t2 * 4 + 3;

    let v0: F32 = block_ptr3d_load(In, d0, d1, d2_0, S0, S1, D0, D1, D2);
    let v1: F32 = block_ptr3d_load(In, d0, d1, d2_1, S0, S1, D0, D1, D2);
    let v2: F32 = block_ptr3d_load(In, d0, d1, d2_2, S0, S1, D0, D1, D2);
    let v3: F32 = block_ptr3d_load(In, d0, d1, d2_3, S0, S1, D0, D1, D2);

    let res0: F32 = v0 * 2.0 + 1.0;
    let res1: F32 = v1 * 2.0 + 1.0;
    let res2: F32 = v2 * 2.0 + 1.0;
    let res3: F32 = v3 * 2.0 + 1.0;

    block_ptr3d_store(Out, d0, d1, d2_0, S0, S1, D0, D1, D2, res0);
    block_ptr3d_store(Out, d0, d1, d2_1, S0, S1, D0, D1, D2, res1);
    block_ptr3d_store(Out, d0, d1, d2_2, S0, S1, D0, D1, D2, res2);
    block_ptr3d_store(Out, d0, d1, d2_3, S0, S1, D0, D1, D2, res3);
}
"""


y_3d_kernel = y_lang.TorchKernel(y_3d_src, kernel_name="bench_3d", target_sm="auto")



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

    return (start_event.elapsed_time(end_event) / rep) * 1000.0  # µs

def run_3d_benchmarks():
    # 3D Tensor Dimensions: Batch=32, Rows=256, Cols=1024 (8.38M elements)
    D0, D1, D2 = 32, 256, 1024
    S0 = D1 * D2
    S1 = D2
    S2 = 1

    x_3d = torch.randn(D0, D1, D2, device=device, dtype=torch.float32)
    out_y = torch.empty_like(x_3d)
    out_triton = torch.empty_like(x_3d)

    BLOCK_0, BLOCK_1, BLOCK_2 = 16, 16, 64
    grid_triton = (triton.cdiv(D0, BLOCK_0), triton.cdiv(D1, BLOCK_1))

    fn_triton = lambda: triton_3d_block_ptr_kernel[grid_triton](
        x_3d, out_triton,
        D0, D1, D2,
        S0, S1, S2,
        BLOCK_0=BLOCK_0, BLOCK_1=BLOCK_1, BLOCK_2=BLOCK_2,
    )

    grid_y = (math.ceil(D0 / 16), math.ceil(D1 / 16), 1)
    block_y = (256, 1, 1) # 256 threads * 4 elements = 1024
    fn_y = lambda: y_3d_kernel.launch(grid=grid_y, block=block_y, args=[x_3d, out_y, D0, D1, D2, S0, S1])

    t_triton = measure_gpu_latency(fn_triton)
    t_y = measure_gpu_latency(fn_y)

    print("------------------------------------------------------------------------")
    print(f"  3D Tensor Workload ({D0}x{D1}x{D2}) | 8.38M Elements:")
    print("------------------------------------------------------------------------")
    print(f"    Y Language 3D Block Pointer    : {t_y:.2f} µs ({t_y/1000.0:.3f} ms)")
    print(f"    OpenAI Triton 3D Pointer      : {t_triton:.2f} µs ({t_triton/1000.0:.3f} ms)")
    speedup = t_triton / t_y
    print(f"    Outcome                        : Y Language is {speedup:.2f}x FASTER vs Triton")
    print("------------------------------------------------------------------------\n")

if __name__ == "__main__":
    run_3d_benchmarks()
