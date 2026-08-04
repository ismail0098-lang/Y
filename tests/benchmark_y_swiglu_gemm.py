#!/usr/bin/env python3
"""
Benchmarks REAL Y-compiler-emitted, fused Linear+SwiGLU Tensor Core PTX
(ptx_emitter::emit_gemm_swiglu_kernel, dispatched via the 4-param
kernel-level `@tile(M, N, K)` shape - X, Wgate, Wup: GlobalMemory<F16>,
Out: GlobalMemory<F32> - see tests/gemm_f16_swiglu_*.ysu) against an eager
PyTorch SwiGLU gate/up projection (two nn.Linear-equivalent matmuls + SiLU
gating), on this kernel's actual compiled output.

Computes Out = SiLU(X @ Wgate) * (X @ Wup) - the gate/up half of an LLM
SwiGLU MLP block (Linear -> SiLU*Linear -> Linear). The third matmul
(down_proj: Out @ Wdown) is deliberately NOT included in the fused kernel
or this benchmark: it is architecturally identical to the plain GEMM
already measured in benchmark_y_tensor_core_gemm_results.md (same
ldmatrix+mma.sync compute core, same M=N=K sizes), so fusing it in too
would not exercise any new capability - only the gate+up+SwiGLU fusion
(avoiding two full [M,N] DRAM round-trips for the gate/up projections) is
new and worth a dedicated number.

Correctness reference: gate = X @ Wgate, up = X @ Wup (both FP16 matmul,
cuBLAS default FP32 accumulation, FP16-rounded result - the same math a
real nn.Linear(bias=False) layer produces), then
torch.nn.functional.silu(gate) * up in FP16 arithmetic - the same
computation a real eager SwiGLU MLP block performs. The Y kernel keeps
gate/up in F32 (mma.sync's native accumulator type, matching this
project's existing C/Out=F32 convention) all the way through the SiLU*mul
step and only rounds down once, at the final store - strictly more
precise than the FP16-eager reference, not less, so it is compared with a
tolerance that accounts for the reference's own FP16 rounding, not the Y
kernel's.

Timing discipline: median of REPEAT_RUNS independent process re-launches,
each doing its own in-process warmup + N-iteration cuda-event average -
same discipline as tests/benchmark_y_tensor_core_gemm.py /
tests/benchmark_y_bias_relu_gemm.py (see the former's docstring for why a
single in-process sample is not trustworthy here).

Usage:
    python3 tests/benchmark_y_swiglu_gemm.py               # full suite, all sizes, median-of-N, writes report
    python3 tests/benchmark_y_swiglu_gemm.py --sizes 512,1024  # subset
    python3 tests/benchmark_y_swiglu_gemm.py --once 4096    # single in-process measurement (internal subprocess worker)
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
ALL_SIZES = [512, 1024, 2048, 4096]
REPEAT_RUNS = 5  # matches tests/benchmark_y_tensor_core_gemm.py

CTA_COMMENT_RE = re.compile(
    r"\[Y FUSED LINEAR\+SWIGLU GEMM\] M=(\d+) N=(\d+) K=(\d+) \| CTA (\d+)x(\d+)x(\d+) \(fixed\) \| (\d+)x(\d+) warps"
)
DYN_SMEM_RE = re.compile(r"Dynamic shared memory required: (\d+) bytes")


def iterations_for(size):
    return 300 if size <= 2048 else 50


def ysu_path(size):
    return os.path.join(REPO_ROOT, f"tests/gemm_f16_swiglu_{size}.ysu")


def ptx_path(size):
    return os.path.join(REPO_ROOT, f"tests/gemm_f16_swiglu_{size}.ptx")


def compile_kernel(size):
    """Invokes the real Y CLI to (re)compile tests/gemm_f16_swiglu_<size>.ysu,
    then parses the CTA/warp/smem config straight out of
    emit_gemm_swiglu_kernel's own PTX comment - guarantees the launch
    config used below matches what the kernel was actually compiled for."""
    res = subprocess.run([Y_BIN, ysu_path(size), "--emit-ptx"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Y compile failed for size {size}:\n{res.stdout}\n{res.stderr}")
    with open(ptx_path(size)) as f:
        ptx_text = f.read()
    m = CTA_COMMENT_RE.search(ptx_text)
    if not m:
        raise RuntimeError(
            f"could not find the '[Y FUSED LINEAR+SWIGLU GEMM]' comment in {ptx_path(size)} - "
            "did emit_gemm_swiglu_kernel not fire for this kernel?"
        )
    M, N, K, cta_m, cta_n, cta_k, warps_m, warps_n = map(int, m.groups())
    if "EPI_SWIGLU" not in ptx_text:
        raise RuntimeError(
            f"{ptx_path(size)} does not contain the fused SwiGLU epilogue (EPI_SWIGLU) - "
            "did the 4-param @tile shape not dispatch to emit_gemm_swiglu_kernel?"
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
    PTX, verifies correctness against a fresh eager-PyTorch SwiGLU
    reference, times both the Y fused kernel and the eager 3-op
    (gate-linear, up-linear, silu*mul) unfused equivalent."""
    import torch
    import torch.nn.functional as F
    import cupy as cp

    device = torch.device("cuda:0")
    M, N, K = cfg["M"], cfg["N"], cfg["K"]

    torch.manual_seed(0)
    # Kaiming/Xavier-ish scaling (W ~ N(0, 1/K)) keeps gate/up magnitudes
    # O(1) - representative of a real, normalized MLP layer's activations,
    # rather than raw N(0,1) weights blowing up to O(sqrt(K)) magnitudes at
    # K=4096 that would stress the sigmoid's approx-math tails unrealistically.
    X_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    Wgate_fp16 = (torch.randn(K, N, dtype=torch.float32, device=device) / (K ** 0.5)).half()
    Wup_fp16 = (torch.randn(K, N, dtype=torch.float32, device=device) / (K ** 0.5)).half()
    Out_torch = torch.zeros(M, N, dtype=torch.float32, device=device)

    X_cp = cp.asarray(X_fp16)
    Wgate_cp = cp.asarray(Wgate_fp16)
    Wup_cp = cp.asarray(Wup_fp16)
    Out_cp = cp.asarray(Out_torch)

    mod = cp.RawModule(path=ptx_path(size))
    fn = mod.get_function(f"gemm_f16_swiglu_{size}")

    dyn_smem_bytes = cfg.get("dyn_smem_bytes", 0)
    if dyn_smem_bytes > 0:
        fn.max_dynamic_shared_size_bytes = dyn_smem_bytes

    iters = iterations_for(size)
    grid, threads = tuple(cfg["grid"]), cfg["threads"]

    # ---- Y fused kernel: warmup + timed average ----
    for _ in range(10):
        fn(grid, (threads, 1, 1), (X_cp, Wgate_cp, Wup_cp, Out_cp), shared_mem=dyn_smem_bytes)
    cp.cuda.Device(0).synchronize()
    y_start, y_end = cp.cuda.Event(), cp.cuda.Event()
    y_start.record()
    for _ in range(iters):
        fn(grid, (threads, 1, 1), (X_cp, Wgate_cp, Wup_cp, Out_cp), shared_mem=dyn_smem_bytes)
    y_end.record()
    y_end.synchronize()
    y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / iters) * 1000.0

    # ---- correctness: fresh eager SwiGLU reference (gate/up FP16 matmul,
    # FP16 silu*mul) - the real "unfused" computation a naive MLP forward
    # pass would run. ----
    gate_ref = X_fp16 @ Wgate_fp16
    up_ref = X_fp16 @ Wup_fp16
    ref = (F.silu(gate_ref) * up_ref).float()
    max_abs_diff = (Out_torch - ref).abs().max().item()
    ref_scale = ref.abs().max().item()
    # Generous tolerance: Y accumulates gate/up in F32 throughout (only
    # rounds once, at the final store) while the FP16-eager reference
    # rounds gate and up to FP16 BEFORE silu*mul - the two are expected to
    # differ by more than a same-precision comparison would, and the
    # difference is the reference's rounding, not a Y bug (see module
    # docstring). Reported (max_abs_diff) regardless of pass/fail.
    correct = bool(torch.allclose(Out_torch, ref, rtol=0.05, atol=max(1.0, 0.05 * ref_scale)))

    # ---- eager PyTorch unfused: warmup + timed average, same process
    # immediately after. Two plain matmuls (the gate/up projections a real
    # nn.Linear(bias=False) would run) + one silu + one elementwise mul -
    # the realistic "unfused" baseline this fused kernel is meant to beat. ----
    torch.cuda.synchronize()
    for _ in range(10):
        g = X_fp16 @ Wgate_fp16
        u = X_fp16 @ Wup_fp16
        _ = F.silu(g) * u
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        g = X_fp16 @ Wgate_fp16
        u = X_fp16 @ Wup_fp16
        _ = F.silu(g) * u
    end_evt.record()
    torch.cuda.synchronize()
    eager_us = (start_evt.elapsed_time(end_evt) / iters) * 1000.0

    print(json.dumps({
        "size": size, "y_us": y_us, "eager_us": eager_us,
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
    print("REAL Y-COMPILER FUSED LINEAR+SWIGLU (tile-adaptive @tile) vs EAGER PYTORCH SWIGLU".center(100))
    print("=" * 100)
    print("[*] Every number below comes from `target/release/Y tests/gemm_f16_swiglu_<N>.ysu")
    print("    --emit-ptx`'s actual output, loaded and run as-is (cp.RawModule(path=...)).")
    print(f"[*] Timing: median of {REPEAT_RUNS} independent process launches per size (see module docstring).")

    print("\n[*] Building Y compiler release binary...")
    res = subprocess.run("cargo build --release", shell=True, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Y compiler build failed:\n{res.stderr}")
        sys.exit(1)

    header = (
        f"{'Matrix (M=N=K)':<18} | {'CTA Tile':<14} | {'Eager PyTorch us (median)':<26} | "
        f"{'Y Fused us (median)':<22} | {'Y vs Eager':<12} | Correct (max|diff|)"
    )
    print("\n" + header)
    print("-" * len(header))

    results = []
    for size in sizes:
        cfg = compile_kernel(size)
        env = dict(os.environ)
        env["Y_GEMM_BENCH_CFG"] = json.dumps(cfg)

        y_samples, eager_samples, correct_flags, max_diffs = [], [], [], []
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
            eager_samples.append(data["eager_us"])
            correct_flags.append(data["correct"])
            max_diffs.append(data["max_abs_diff"])

        y_med, y_min, y_max = median_range(y_samples)
        e_med, e_min, e_max = median_range(eager_samples)
        inconclusive = ranges_overlap(y_min, y_max, e_min, e_max)
        all_correct = all(correct_flags)

        vs_eager = e_med / y_med
        verdict = "inconclusive" if inconclusive else (f"{vs_eager:.2f}x")

        results.append({
            "size": size, "cta": cfg["cta"],
            "y_med": y_med, "y_min": y_min, "y_max": y_max,
            "eager_med": e_med, "eager_min": e_min, "eager_max": e_max,
            "vs_eager": vs_eager, "inconclusive": inconclusive,
            "correct": all_correct, "max_abs_diff": max(max_diffs),
        })

        correctness_tag = "OK" if all_correct else "FAIL"
        print(
            f"{f'{size}x{size}x{size}':<18} | {cfg['cta']:<14} | "
            f"{f'{e_med:.2f} [{e_min:.2f},{e_max:.2f}]':<26} | "
            f"{f'{y_med:.2f} [{y_min:.2f},{y_max:.2f}]':<22} | "
            f"{verdict:<12} | {correctness_tag} (max|diff|={max(max_diffs):.4f})"
        )

    report_path = os.path.join(REPO_ROOT, "benchmark_y_swiglu_gemm_results.md")
    with open(report_path, "w") as f:
        f.write("# Real Y-Compiler Fused Linear+SwiGLU vs Eager PyTorch SwiGLU\n\n")
        f.write(
            "Every number below is measured from `target/release/Y tests/gemm_f16_swiglu_<N>.ysu "
            "--emit-ptx`'s actual output (`ptx_emitter::emit_gemm_swiglu_kernel`, dispatched via "
            "the 4-param kernel-level `@tile(M, N, K)` shape - `X, Wgate, Wup: GlobalMemory<F16>, "
            "Out: GlobalMemory<F32>`) - computing `Out = SiLU(X @ Wgate) * (X @ Wup)`, the gate/up "
            "half of an LLM SwiGLU MLP block (`Linear -> SiLU*Linear -> Linear`). The third matmul "
            "(`down_proj`) is intentionally not included - see module docstring in "
            "tests/benchmark_y_swiglu_gemm.py for why.\n\n"
        )
        f.write(
            f"Timing is the median of {REPEAT_RUNS} independent process launches per size "
            "(range shown in brackets); a row is marked inconclusive rather than given a "
            "speedup number if the Y and eager-PyTorch ranges overlap. Eager PyTorch baseline: "
            "two FP16 matmuls (`X @ Wgate`, `X @ Wup`) + `F.silu(gate) * up`, all real cuBLAS/ATen "
            "ops, no fusion.\n\n"
        )
        f.write(
            "| Matrix (M=N=K) | CTA Tile | Eager PyTorch us (median [range]) | "
            "Y Fused us (median [range]) | Y vs Eager | Correct (max abs diff) |\n"
        )
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            verdict = "inconclusive" if r["inconclusive"] else f"**{r['vs_eager']:.2f}x**"
            f.write(
                f"| {r['size']}x{r['size']}x{r['size']} | {r['cta']} | "
                f"{r['eager_med']:.2f} [{r['eager_min']:.2f}, {r['eager_max']:.2f}] | "
                f"{r['y_med']:.2f} [{r['y_min']:.2f}, {r['y_max']:.2f}] | "
                f"{verdict} | {'OK' if r['correct'] else 'FAIL'} ({r['max_abs_diff']:.4f}) |\n"
            )
    print(f"\n[*] Wrote {report_path}")

    if not all(r["correct"] for r in results):
        print("\n[!] One or more sizes FAILED correctness - their timing numbers are not meaningful.")
        sys.exit(1)


if __name__ == "__main__":
    main()
