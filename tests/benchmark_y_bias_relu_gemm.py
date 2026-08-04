#!/usr/bin/env python3
"""
Benchmarks REAL Y-compiler-emitted, fused GEMM+Bias+ReLU Tensor Core PTX
(ptx_emitter::emit_gemm_bias_relu_epilogue, dispatched via the 4-param
kernel-level `@tile(M, N, K)` shape - A, B: GlobalMemory<F16>, Bias, C:
GlobalMemory<F32> - see tests/gemm_f16_bias_relu_*.ysu) against cuDNN's
fused-linear path (torch.nn.functional.linear + relu), on this epilogue's
actual compiled output.

This is the real-compiler counterpart to benchmark_y_vs_cudnn_cublas.py's
Suite 2, which measures a hand-written CUDA C++ reference kernel
(tests/y_tensor_core_gemm.cu's y_fused_gemm_bias_relu_kernel) - explicitly
NOT Y's own compiler output. This script measures the real thing: for each
size, it shells out to `target/release/Y tests/gemm_f16_bias_relu_<N>.ysu
--emit-ptx`, loads the resulting .ptx directly (cp.RawModule(path=...)), and
times *that*.

Correctness reference: torch.nn.functional.linear(A, B.T, bias) followed by
relu - the same math a real nn.Linear + activation layer produces (NOT bias/
relu bolted onto a separately-computed matmul in some other framework path -
functional.linear's own fused kernel is the ground truth here, matching what
a real LLM inference consumer of this op would actually call).

Timing discipline: median of REPEAT_RUNS independent process re-launches,

each doing its own in-process warmup + N-iteration cuda-event average -
same discipline as tests/benchmark_y_tensor_core_gemm.py (see that file's
docstring for why a single in-process sample is not trustworthy here).

Usage:
    python3 tests/benchmark_y_bias_relu_gemm.py               # full suite, all sizes, median-of-N, writes report
    python3 tests/benchmark_y_bias_relu_gemm.py --sizes 512,1024  # subset
    python3 tests/benchmark_y_bias_relu_gemm.py --once 4096    # single in-process measurement (internal subprocess worker)
"""
import os
import sys
import re
import json
import time
import subprocess
import statistics
import argparse

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
Y_BIN = os.path.join(REPO_ROOT, "target/release/Y")
ALL_SIZES = [512, 1024, 2048, 4096, 8192]
REPEAT_RUNS = 5  # matches tests/benchmark_y_tensor_core_gemm.py

# --- GPU clock-ramp control (ported from benchmark_y_decode_gemm.py) ---
#
# The SM clock on this project's dev GPU idles at ~210 MHz and needs ~3s of
# sustained load to reach ~2670 MHz (12.7x), and clocks cannot be locked on
# this box. A 10-iteration warmup does not ramp them, so a cold-started
# process measures a kernel at a fraction of a warm one's clock. Timing all
# of Y and then all of cuDNN also gave cuDNN the hotter clock every time - a
# systematic bias, not noise. Both are now controlled: ramp first, then
# A/B/C-interleave the three timed variants so all see the same clock.
RAMP_SECONDS = 3.0
INTERLEAVE_ROUNDS = 5

CTA_COMMENT_RE = re.compile(
    r"\[Y TENSOR CORE GEMM\] M=(\d+) N=(\d+) K=(\d+) \| CTA (\d+)x(\d+)x(\d+) \| (\d+)x(\d+) warps"
)
DYN_SMEM_RE = re.compile(r"Dynamic shared memory required: (\d+) bytes")


def iterations_for(size):
    return 300 if size <= 2048 else (50 if size <= 4096 else 20)


def ysu_path(size):
    return os.path.join(REPO_ROOT, f"tests/gemm_f16_bias_relu_{size}.ysu")


def ptx_path(size):
    return os.path.join(REPO_ROOT, f"tests/gemm_f16_bias_relu_{size}.ptx")


def compile_kernel(size):
    """Invokes the real Y CLI to (re)compile tests/gemm_f16_bias_relu_<size>.ysu,
    then parses the CTA/warp/smem config straight out of
    emit_tensor_core_gemm_kernel's own PTX comment - see
    tests/benchmark_y_tensor_core_gemm.py's compile_kernel for why this
    matters (guarantees the launch config used below matches what the
    kernel was actually compiled for)."""
    res = subprocess.run([Y_BIN, ysu_path(size), "--emit-ptx"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Y compile failed for size {size}:\n{res.stdout}\n{res.stderr}")
    with open(ptx_path(size)) as f:
        ptx_text = f.read()
    m = CTA_COMMENT_RE.search(ptx_text)
    if not m:
        raise RuntimeError(
            f"could not find the '[Y TENSOR CORE GEMM]' comment in {ptx_path(size)} - "
            "did emit_tensor_core_gemm_kernel not fire for this kernel?"
        )
    M, N, K, cta_m, cta_n, cta_k, warps_m, warps_n = map(int, m.groups())
    if "EPI_BIAS_RELU" not in ptx_text:
        raise RuntimeError(
            f"{ptx_path(size)} does not contain the fused Bias+ReLU epilogue (EPI_BIAS_RELU) - "
            "did the 4-param @tile shape not dispatch to emit_gemm_bias_relu_epilogue?"
        )
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
    PTX, verifies correctness against a fresh torch.nn.functional.linear +
    relu reference, times both the Y fused kernel and cuDNN's fused-linear
    path (same process, back-to-back, same GPU state)."""
    import torch
    import cupy as cp

    device = torch.device("cuda:0")
    M, N, K = cfg["M"], cfg["N"], cfg["K"]

    torch.manual_seed(0)
    A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    bias_fp32 = torch.randn(N, dtype=torch.float32, device=device)
    C_torch = torch.zeros(M, N, dtype=torch.float32, device=device)

    A_cp = cp.asarray(A_fp16)
    B_cp = cp.asarray(B_fp16)
    bias_cp = cp.asarray(bias_fp32)
    C_cp = cp.asarray(C_torch)

    mod = cp.RawModule(path=ptx_path(size))
    fn = mod.get_function(f"gemm_f16_bias_relu_{size}")

    dyn_smem_bytes = cfg.get("dyn_smem_bytes", 0)
    if dyn_smem_bytes > 0:
        fn.max_dynamic_shared_size_bytes = dyn_smem_bytes

    # Total iterations split across INTERLEAVE_ROUNDS rounds so the measured
    # work per size stays comparable to the pre-interleave version.
    iters = max(3, iterations_for(size) // INTERLEAVE_ROUNDS)
    grid, threads = tuple(cfg["grid"]), cfg["threads"]

    A_fp32 = A_fp16.float()
    linear_layer = torch.nn.Linear(K, N, bias=True, device=device)
    linear_layer_fp16 = torch.nn.Linear(K, N, bias=True, device=device, dtype=torch.float16)
    with torch.no_grad():
        linear_layer.weight.copy_(B_fp16.t().float())
        linear_layer.bias.copy_(bias_fp32)
        linear_layer_fp16.weight.copy_(B_fp16.t())
        linear_layer_fp16.bias.copy_(bias_fp32.to(torch.float16))

    def time_y():
        cp.cuda.Device(0).synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        for _ in range(iters):
            fn(grid, (threads, 1, 1), (A_cp, B_cp, bias_cp, C_cp), shared_mem=dyn_smem_bytes)
        e1.record(); e1.synchronize()
        return (cp.cuda.get_elapsed_time(e0, e1) / iters) * 1000.0

    def _time_torch(op):
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            _ = op()
        e1.record(); torch.cuda.synchronize()
        return (e0.elapsed_time(e1) / iters) * 1000.0

    # ---- 1. ramp clocks to steady state on a throwaway workload ----
    ramp = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    t0 = time.time()
    while time.time() - t0 < RAMP_SECONDS:
        for _ in range(20):
            _ = ramp @ ramp
        torch.cuda.synchronize()
    del ramp

    # ---- 2. interleaved timing so all three see the same clock ----
    y_r, c32_r, c16_r = [], [], []
    with torch.no_grad():
        for _ in range(INTERLEAVE_ROUNDS):
            y_r.append(time_y())
            c32_r.append(_time_torch(lambda: torch.relu(linear_layer(A_fp32))))
            c16_r.append(_time_torch(lambda: torch.relu(linear_layer_fp16(A_fp16))))
    y_us = statistics.median(y_r)
    cudnn_us = statistics.median(c32_r)
    cudnn_fp16_us = statistics.median(c16_r)

    # ---- correctness: fresh functional.linear(A, B.T, bias) + relu reference ----
    ref = torch.relu(torch.nn.functional.linear(A_fp16, B_fp16.t().contiguous(), bias_fp32.to(torch.float16)).float())
    max_abs_diff = (C_torch - ref).abs().max().item()
    ref_scale = ref.abs().max().item()
    correct = bool(torch.allclose(C_torch, ref, rtol=0.02, atol=max(0.75, 0.02 * ref_scale)))

    print(json.dumps({
        "size": size, "y_us": y_us, "cudnn_us": cudnn_us, "cudnn_fp16_us": cudnn_fp16_us,
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
        cfg = json.loads(os.environ["Y_GEMM_BENCH_CFG"])
        run_once(args.once, cfg)
        return

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else ALL_SIZES

    print("=" * 100)
    print("REAL Y-COMPILER FUSED GEMM+BIAS+RELU (tile-adaptive @tile) vs cuDNN FUSED LINEAR".center(100))
    print("=" * 100)
    print("[*] Unlike benchmark_y_vs_cudnn_cublas.py's Suite 2 (hand-written CUDA C++ reference),")
    print("    every number below comes from `target/release/Y tests/gemm_f16_bias_relu_<N>.ysu")
    print("    --emit-ptx`'s actual output, loaded and run as-is (cp.RawModule(path=...)).")
    print(f"[*] Timing: median of {REPEAT_RUNS} independent process launches per size (see module docstring).")

    print("\n[*] Building Y compiler release binary...")
    res = subprocess.run("cargo build --release", shell=True, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Y compiler build failed:\n{res.stderr}")
        sys.exit(1)

    header = (
        f"{'Matrix (M=N=K)':<18} | {'CTA Tile':<14} | {'cuDNN FP32 us (median)':<24} | "
        f"{'cuDNN FP16 us (median)':<24} | {'Y Fused us (median)':<22} | {'Y vs cuDNN FP32':<16} | {'Y vs cuDNN FP16':<16} | Correct"
    )
    print("\n" + header)
    print("-" * len(header))

    results = []
    for size in sizes:
        cfg = compile_kernel(size)
        env = dict(os.environ)
        env["Y_GEMM_BENCH_CFG"] = json.dumps(cfg)

        y_samples, cudnn_samples, cudnn_fp16_samples, correct_flags, max_diffs = [], [], [], [], []
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
            cudnn_samples.append(data["cudnn_us"])
            cudnn_fp16_samples.append(data["cudnn_fp16_us"])
            correct_flags.append(data["correct"])
            max_diffs.append(data["max_abs_diff"])

        y_med, y_min, y_max = median_range(y_samples)
        c_med, c_min, c_max = median_range(cudnn_samples)
        cf16_med, cf16_min, cf16_max = median_range(cudnn_fp16_samples)
        inconclusive = ranges_overlap(y_min, y_max, c_min, c_max)
        inconclusive_fp16 = ranges_overlap(y_min, y_max, cf16_min, cf16_max)
        all_correct = all(correct_flags)

        vs_cudnn = c_med / y_med
        vs_cudnn_fp16 = cf16_med / y_med
        verdict = "inconclusive" if inconclusive else (f"{vs_cudnn:.2f}x")
        verdict_fp16 = "inconclusive" if inconclusive_fp16 else (f"{vs_cudnn_fp16:.2f}x")

        results.append({
            "size": size, "cta": cfg["cta"],
            "y_med": y_med, "y_min": y_min, "y_max": y_max,
            "cudnn_med": c_med, "cudnn_min": c_min, "cudnn_max": c_max,
            "cudnn_fp16_med": cf16_med, "cudnn_fp16_min": cf16_min, "cudnn_fp16_max": cf16_max,
            "vs_cudnn": vs_cudnn, "inconclusive": inconclusive,
            "vs_cudnn_fp16": vs_cudnn_fp16, "inconclusive_fp16": inconclusive_fp16,
            "correct": all_correct, "max_abs_diff": max(max_diffs),
        })

        correctness_tag = "OK" if all_correct else f"FAIL (max|diff|={max(max_diffs):.4f})"
        print(
            f"{f'{size}x{size}x{size}':<18} | {cfg['cta']:<14} | "
            f"{f'{c_med:.2f} [{c_min:.2f},{c_max:.2f}]':<24} | "
            f"{f'{cf16_med:.2f} [{cf16_min:.2f},{cf16_max:.2f}]':<24} | "
            f"{f'{y_med:.2f} [{y_min:.2f},{y_max:.2f}]':<22} | "
            f"{verdict:<16} | {verdict_fp16:<16} | {correctness_tag}"
        )

    report_path = os.path.join(REPO_ROOT, "benchmark_y_bias_relu_gemm_results.md")
    with open(report_path, "w") as f:
        f.write("# Real Y-Compiler Fused GEMM+Bias+ReLU vs cuDNN Fused Linear\n\n")
        f.write(
            "Every number below is measured from `target/release/Y tests/gemm_f16_bias_relu_<N>.ysu "
            "--emit-ptx`'s actual output (`ptx_emitter::emit_gemm_bias_relu_epilogue`, dispatched via "
            "the 4-param kernel-level `@tile(M, N, K)` shape) - not the hand-written CUDA C++ reference "
            "benchmark_y_vs_cudnn_cublas.py Suite 2 measures.\n\n"
        )
        f.write(
            f"Timing is the median of {REPEAT_RUNS} independent process launches per size "
            "(range shown in brackets); a row is marked inconclusive rather than given a "
            "speedup number if the Y and cuDNN ranges overlap.\n\n"
        )
        f.write(
            "Two cuDNN columns, deliberately: **FP32** matches "
            "benchmark_y_vs_cudnn_cublas.py Suite 2's exact convention (FP32 "
            "`torch.nn.Linear`, the same baseline that file's 1.16x-1.31x number "
            "came from) - but FP32 cuBLAS/cuDNN gemm is intrinsically much slower "
            "per-FLOP than FP16 tensor-core gemm regardless of fusion quality, so "
            "a large margin there conflates \"the fused epilogue is good\" with "
            "\"FP16 beats FP32\". **FP16** feeds `torch.nn.Linear` the same FP16 "
            "operands the Y kernel itself consumes - the apples-to-apples "
            "precision-matched comparison.\n\n"
        )
        f.write(
            "| Matrix (M=N=K) | CTA Tile | cuDNN FP32 us (median [range]) | cuDNN FP16 us (median [range]) | "
            "Y Fused us (median [range]) | Y vs cuDNN FP32 | Y vs cuDNN FP16 | Correct |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            verdict = "inconclusive" if r["inconclusive"] else f"**{r['vs_cudnn']:.2f}x**"
            verdict_fp16 = "inconclusive" if r["inconclusive_fp16"] else f"**{r['vs_cudnn_fp16']:.2f}x**"
            f.write(
                f"| {r['size']}x{r['size']}x{r['size']} | {r['cta']} | "
                f"{r['cudnn_med']:.2f} [{r['cudnn_min']:.2f}, {r['cudnn_max']:.2f}] | "
                f"{r['cudnn_fp16_med']:.2f} [{r['cudnn_fp16_min']:.2f}, {r['cudnn_fp16_max']:.2f}] | "
                f"{r['y_med']:.2f} [{r['y_min']:.2f}, {r['y_max']:.2f}] | "
                f"{verdict} | {verdict_fp16} | {'OK' if r['correct'] else 'FAIL'} |\n"
            )
    print(f"\n[*] Wrote {report_path}")

    if not all(r["correct"] for r in results):
        print("\n[!] One or more sizes FAILED correctness - their timing numbers are not meaningful.")
        sys.exit(1)


if __name__ == "__main__":
    main()
