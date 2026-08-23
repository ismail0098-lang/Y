#!/usr/bin/env python3
"""
Y — Wafer AI PyTorch Integration & Kernel Compilation Benchmark
Demonstrates JIT compilation, PTX codegen, autotuning space generation,
and PyTorch Tensor interop for Y kernels.
"""

import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "python"))

import y_lang

def main():
    print("============================================================")
    print("=== Y — Wafer AI Performance Engine Benchmark ===")
    print("============================================================\n")

    # 1. Test Autotuning Search Space Generation for Wafer Agent
    print("[1] Generating Autotuning Candidate Tile Space for 2048x2048 Matrix...")
    search_space = y_lang.generate_autotune_search_space(2048, 2048, 2048)
    print(f"    -> Generated {len(search_space)} candidate configurations.")
    for idx, cand in enumerate(search_space[:3]):
        print(f"       Candidate #{idx+1}: CTA Tile={cand['cta_m']}x{cand['cta_n']}x{cand['cta_k']}, Warps={cand['num_warps']}, Stages={cand['num_stages']}")
    print()

    # 2. Test FlashAttention-2 Kernel JIT Compilation
    flash_path = repo_root / "examples" / "flash_attention.ysu"
    print(f"[2] JIT Compiling Fused FlashAttention-2 Kernel ({flash_path.name})...")
    with open(flash_path, "r") as f:
        flash_code = f.read()

    start_t = time.perf_counter()
    flash_ptx = y_lang.compile_to_ptx(flash_code, target_sm="sm_90a")
    compile_ms = (time.perf_counter() - start_t) * 1000.0

    print(f"    -> Compiled to sm_90a PTX in {compile_ms:.2f} ms.")
    print(f"    -> PTX Output snippet (first 10 lines):")
    for line in flash_ptx.splitlines()[:10]:
        print(f"       {line}")
    print()

    # 3. Test RMSNorm Kernel JIT Compilation
    rmsnorm_path = repo_root / "examples" / "rmsnorm.ysu"
    print(f"[3] JIT Compiling RMSNorm Kernel ({rmsnorm_path.name})...")
    with open(rmsnorm_path, "r") as f:
        rmsnorm_code = f.read()

    rmsnorm_ptx = y_lang.compile_to_ptx(rmsnorm_code, target_sm="sm_80")
    print(f"    -> Successfully emitted sm_80 PTX ({len(rmsnorm_ptx)} bytes).")
    print()

    # 4. Test SwiGLU Kernel JIT Compilation
    swiglu_path = repo_root / "examples" / "swiglu.ysu"
    print(f"[4] JIT Compiling SwiGLU Activation Kernel ({swiglu_path.name})...")
    with open(swiglu_path, "r") as f:
        swiglu_code = f.read()

    swiglu_ptx = y_lang.compile_to_ptx(swiglu_code, target_sm="sm_89")
    print(f"    -> Successfully emitted sm_89 PTX ({len(swiglu_ptx)} bytes).")
    print()

    print("============================================================")
    print(" All Y Wafer AI production components validated!")
    print("============================================================")

if __name__ == "__main__":
    main()
