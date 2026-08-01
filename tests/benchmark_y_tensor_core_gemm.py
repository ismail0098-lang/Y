#!/usr/bin/env python3
"""
Benchmarks REAL Y-compiler-emitted, tile-adaptive Tensor Core GEMM PTX
(ptx_emitter::emit_tensor_core_gemm_kernel, dispatched via kernel-level
`@tile(M, N, K)` - see tests/gemm_f16_*.ysu) against cuBLAS (torch.matmul
FP16), on this kernel's actual compiled output.

This is deliberately a separate script from benchmark_y_vs_cudnn_cublas.py:
that file's Suites 1-2 measure a hand-written CUDA C++ reference kernel
(tests/y_tensor_core_gemm.cu, compiled via NVRTC) representing the tiling
strategy Y's PTX emitter targets - explicitly NOT Y's own compiler output
(see that file's header comment). This script measures the real thing: for
each size, it shells out to `target/release/Y tests/gemm_f16_<N>.ysu
--emit-ptx`, loads the resulting .ptx directly (cp.RawModule(path=...), the
same mechanism that file's Suite 3 already uses for real Y output), and
times *that*.

Timing discipline: median of REPEAT_RUNS independent process re-launches,
each doing its own in-process warmup + N-iteration cuda-event average -
matching src/bin/autotune_verify.rs's REPEAT_RUNS pattern, not a single
in-process measurement. That file's doc comment records why this matters:
two back-to-back runs of an IDENTICAL candidate, moments apart on the same
GPU, produced opposite A/B verdicts (608% slower vs 66% faster) purely from
GPU clock/power-state transitions across process launches. A single-sample
number here would be exactly as untrustworthy.

Usage:
    python3 tests/benchmark_y_tensor_core_gemm.py               # full suite, all sizes, median-of-N, writes report
    python3 tests/benchmark_y_tensor_core_gemm.py --sizes 256,512  # subset
    python3 tests/benchmark_y_tensor_core_gemm.py --once 4096    # single in-process measurement (internal subprocess worker)
"""
import os
import sys
import re
import json
import subprocess
import statistics
import argparse

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
Y_BIN = os.path.join(REPO_ROOT, "target/release/Y")
ALL_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]
REPEAT_RUNS = 5  # matches src/bin/autotune_verify.rs's REPEAT_RUNS

CTA_COMMENT_RE = re.compile(
    r"\[Y TENSOR CORE GEMM\] M=(\d+) N=(\d+) K=(\d+) \| CTA (\d+)x(\d+)x(\d+) \| (\d+)x(\d+) warps"
)
# emit_tensor_core_gemm_kernel's second comment line: present (with a byte
# count) only on the cp.async-pipelined path - the K-too-small synchronous
# fallback's comment has no "Dynamic shared memory required" clause at all,
# which DYN_SMEM_RE.search below simply won't match (dyn_smem_bytes stays 0,
# same as "no dynamic shared memory needed").
DYN_SMEM_RE = re.compile(r"Dynamic shared memory required: (\d+) bytes")


def iterations_for(size):
    return 500 if size <= 1024 else (100 if size <= 4096 else (20 if size <= 8192 else 5))


def ysu_path(size):
    return os.path.join(REPO_ROOT, f"tests/gemm_f16_{size}.ysu")


def ptx_path(size):
    return os.path.join(REPO_ROOT, f"tests/gemm_f16_{size}.ptx")


def compile_kernel(size):
    """Invokes the real Y CLI to (re)compile tests/gemm_f16_<size>.ysu, then
    parses the CTA/warp config straight out of emit_tensor_core_gemm_kernel's
    own PTX comment - so the launch config used below is guaranteed to match
    what the kernel was actually compiled for, not a second, independently
    (and possibly divergently) computed guess."""
    res = subprocess.run([Y_BIN, ysu_path(size), "--emit-ptx"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Y compile failed for size {size}:\n{res.stdout}\n{res.stderr}")
    with open(ptx_path(size)) as f:
        ptx_text = f.read()
    m = CTA_COMMENT_RE.search(ptx_text)
    if not m:
        raise RuntimeError(
            f"could not find the '[Y TENSOR CORE GEMM]' comment in {ptx_path(size)} - "
            "did emit_tensor_core_gemm_kernel not fire for this kernel? (falls back to "
            "generic scalar lowering silently if @tile dispatch didn't match)"
        )
    M, N, K, cta_m, cta_n, cta_k, warps_m, warps_n = map(int, m.groups())
    smem_m = DYN_SMEM_RE.search(ptx_text)
    return {
        "size": size,
        "M": M, "N": N, "K": K,
        "cta": f"{cta_m}x{cta_n}x{cta_k}",
        "grid": ((N + cta_n - 1) // cta_n, (M + cta_m - 1) // cta_m),
        "threads": warps_m * warps_n * 32,
        "dyn_smem_bytes": int(smem_m.group(1)) if smem_m else 0,
    }


def run_once(size, cfg):
    """Runs entirely inside one fresh process: loads the already-compiled
    PTX, verifies correctness against a fresh cuBLAS reference, times both
    the Y kernel and cuBLAS (same process, back-to-back, same GPU state) via
    in-process warmup + N-iteration averaging. Prints one JSON line."""
    import torch
    import cupy as cp

    device = torch.device("cuda:0")
    M, N, K = cfg["M"], cfg["N"], cfg["K"]

    torch.manual_seed(0)
    A_torch = torch.randn(M, K, dtype=torch.float16, device=device)
    B_torch = torch.randn(K, N, dtype=torch.float16, device=device)
    C_torch = torch.zeros(M, N, dtype=torch.float32, device=device)

    A_cp = cp.asarray(A_torch)
    B_cp = cp.asarray(B_torch)
    C_cp = cp.asarray(C_torch)

    mod = cp.RawModule(path=ptx_path(size))
    fn = mod.get_function(f"gemm_f16_{size}")

    # The cp.async-pipelined kernel declares its stage buffers as one
    # `.extern .shared` (dynamic) array sized only at launch time - see
    # emit_tensor_core_gemm_kernel's doc comment for why (a statically-sized
    # `.shared` array is hard-capped at 48KB by ptxas regardless of this
    # GPU's real ~100KB capacity, confirmed by direct experiment). cupy
    # defaults `max_dynamic_shared_size_bytes` to that same 48KB, so any
    # config needing more (every pipelined config observed so far) must
    # explicitly opt in before launch, in addition to passing the actual
    # byte count via `shared_mem=` on every launch below.
    dyn_smem_bytes = cfg.get("dyn_smem_bytes", 0)
    if dyn_smem_bytes > 0:
        fn.max_dynamic_shared_size_bytes = dyn_smem_bytes

    iters = iterations_for(size)
    # cfg arrives via JSON (env var, from the outer driver) - JSON has no
    # tuple type, and cupy's launch API rejects a plain list for `grid`.
    grid, threads = tuple(cfg["grid"]), cfg["threads"]

    # ---- Y kernel: warmup + timed average ----
    for _ in range(10):
        fn(grid, (threads, 1, 1), (A_cp, B_cp, C_cp), shared_mem=dyn_smem_bytes)
    cp.cuda.Device(0).synchronize()
    y_start, y_end = cp.cuda.Event(), cp.cuda.Event()
    y_start.record()
    for _ in range(iters):
        fn(grid, (threads, 1, 1), (A_cp, B_cp, C_cp), shared_mem=dyn_smem_bytes)
    y_end.record()
    y_end.synchronize()
    y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / iters) * 1000.0

    # ---- correctness: fresh cuBLAS reference (C_cp aliases C_torch's storage) ----
    ref = torch.matmul(A_torch, B_torch).float()
    max_abs_diff = (C_torch - ref).abs().max().item()
    ref_scale = ref.abs().max().item()
    # Same tolerance reasoning as benchmark_y_vs_cudnn_cublas.py's check_correctness:
    # legitimate FP16-accumulation rounding grows with reduction depth (K) and output
    # magnitude, so atol scales with the reference's own magnitude rather than a bare
    # constant - a real correctness bug produces errors orders of magnitude past this.
    correct = bool(torch.allclose(C_torch, ref, rtol=0.02, atol=max(0.75, 0.02 * ref_scale)))

    # ---- cuBLAS: warmup + timed average, same process immediately after ----
    torch.cuda.synchronize()
    for _ in range(10):
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

    print(json.dumps({
        "size": size, "y_us": y_us, "cublas_us": cublas_us,
        "correct": correct, "max_abs_diff": max_abs_diff,
    }))


def median_range(values):
    return statistics.median(values), min(values), max(values)


def ranges_overlap(a_min, a_max, b_min, b_max):
    return a_min <= b_max and b_min <= a_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sizes", type=str, default=None, help="comma-separated subset of sizes")
    args = parser.parse_args()

    if args.once is not None:
        # Internal subprocess-worker mode: config is passed by the outer
        # driver so this process never has to recompile.
        cfg = json.loads(os.environ["Y_GEMM_BENCH_CFG"])
        run_once(args.once, cfg)
        return

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else ALL_SIZES

    print("=" * 100)
    print("REAL Y-COMPILER TENSOR CORE GEMM (tile-adaptive @tile) vs cuBLAS".center(100))
    print("=" * 100)
    print("[*] Unlike benchmark_y_vs_cudnn_cublas.py's Suite 1 (hand-written CUDA C++ reference),")
    print("    every number below comes from `target/release/Y tests/gemm_f16_<N>.ysu --emit-ptx`'s")
    print("    actual output, loaded and run as-is (cp.RawModule(path=...), no hand-editing).")
    print(f"[*] Timing: median of {REPEAT_RUNS} independent process launches per size (see module docstring).")

    print("\n[*] Building Y compiler release binary...")
    res = subprocess.run("cargo build --release", shell=True, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Y compiler build failed:\n{res.stderr}")
        sys.exit(1)

    header = f"{'Matrix (M=N=K)':<18} | {'CTA Tile':<14} | {'cuBLAS us (median)':<22} | {'Y us (median)':<22} | {'Y vs cuBLAS':<14} | Correct"
    print("\n" + header)
    print("-" * len(header))

    results = []
    for size in sizes:
        cfg = compile_kernel(size)
        env = dict(os.environ)
        env["Y_GEMM_BENCH_CFG"] = json.dumps(cfg)

        y_samples, cublas_samples, correct_flags, max_diffs = [], [], [], []
        for _ in range(REPEAT_RUNS):
            proc = subprocess.run(
                [sys.executable, __file__, "--once", str(size)],
                capture_output=True, text=True, env=env, cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                print(f"[!] worker run failed for size {size}:\n{proc.stdout}\n{proc.stderr}")
                sys.exit(1)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("{")), None)
            if line is None:
                print(f"[!] no JSON output from worker for size {size}:\n{proc.stdout}\n{proc.stderr}")
                sys.exit(1)
            data = json.loads(line)
            y_samples.append(data["y_us"])
            cublas_samples.append(data["cublas_us"])
            correct_flags.append(data["correct"])
            max_diffs.append(data["max_abs_diff"])

        y_med, y_min, y_max = median_range(y_samples)
        c_med, c_min, c_max = median_range(cublas_samples)
        inconclusive = ranges_overlap(y_min, y_max, c_min, c_max)
        all_correct = all(correct_flags)

        vs_cublas = c_med / y_med
        verdict = "inconclusive (ranges overlap)" if inconclusive else (f"{vs_cublas:.2f}x")

        results.append({
            "size": size, "cta": cfg["cta"],
            "y_med": y_med, "y_min": y_min, "y_max": y_max,
            "cublas_med": c_med, "cublas_min": c_min, "cublas_max": c_max,
            "vs_cublas": vs_cublas, "inconclusive": inconclusive,
            "correct": all_correct, "max_abs_diff": max(max_diffs),
        })

        correctness_tag = "OK" if all_correct else f"FAIL (max|diff|={max(max_diffs):.4f})"
        print(
            f"{f'{size}x{size}x{size}':<18} | {cfg['cta']:<14} | "
            f"{f'{c_med:.2f} [{c_min:.2f},{c_max:.2f}]':<22} | "
            f"{f'{y_med:.2f} [{y_min:.2f},{y_max:.2f}]':<22} | "
            f"{verdict:<14} | {correctness_tag}"
        )

    report_path = os.path.join(REPO_ROOT, "benchmark_y_tensor_core_gemm_results.md")
    with open(report_path, "w") as f:
        f.write("# Real Y-Compiler Tensor Core GEMM vs cuBLAS\n\n")
        f.write(
            "Every number below is measured from `target/release/Y tests/gemm_f16_<N>.ysu "
            "--emit-ptx`'s actual output (`ptx_emitter::emit_tensor_core_gemm_kernel`, "
            "dispatched via kernel-level `@tile(M, N, K)`, tile/warp/stage selection from "
            "`Autotuner::autotune`) - not the hand-written CUDA C++ reference "
            "benchmark_y_vs_cudnn_cublas.py's Suite 1 measures.\n\n"
        )
        f.write(
            f"Timing is the median of {REPEAT_RUNS} independent process launches per size "
            "(range shown in brackets); a row is marked inconclusive rather than given a "
            "speedup number if the Y and cuBLAS ranges overlap - see the module docstring "
            "in tests/benchmark_y_tensor_core_gemm.py for why a single-sample number isn't "
            "trustworthy here.\n\n"
        )
        f.write("| Matrix (M=N=K) | CTA Tile | cuBLAS us (median [range]) | Y us (median [range]) | Y vs cuBLAS | Correct |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            verdict = "inconclusive" if r["inconclusive"] else f"**{r['vs_cublas']:.2f}x**"
            f.write(
                f"| {r['size']}x{r['size']}x{r['size']} | {r['cta']} | "
                f"{r['cublas_med']:.2f} [{r['cublas_min']:.2f}, {r['cublas_max']:.2f}] | "
                f"{r['y_med']:.2f} [{r['y_min']:.2f}, {r['y_max']:.2f}] | "
                f"{verdict} | {'OK' if r['correct'] else 'FAIL'} |\n"
            )
    print(f"\n[*] Wrote {report_path}")

    if not all(r["correct"] for r in results):
        print("\n[!] One or more sizes FAILED correctness - their timing numbers are not meaningful.")
        sys.exit(1)


if __name__ == "__main__":
    main()
