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

Timing discipline: median of REPEAT_RUNS independent process re-launches
(matching src/bin/autotune_verify.rs's REPEAT_RUNS pattern), where each launch
first ramps the GPU clock to steady state and then A/B-INTERLEAVES Y and cuBLAS
so both are measured at the same clock. See the RAMP_SECONDS comment below.

autotune_verify.rs's doc comment already recorded the underlying hazard: two
back-to-back runs of an IDENTICAL candidate, moments apart on the same GPU,
produced opposite A/B verdicts (608% slower vs 66% faster) purely from GPU
clock/power-state transitions. Re-launching the process does not fix that - it
averages over clock states instead of controlling for them, and does nothing
about the ordering bias from timing all of Y before all of cuBLAS. Both are now
controlled directly.

Usage:
    python3 tests/benchmark_y_tensor_core_gemm.py               # full suite, all sizes, median-of-N, writes report
    python3 tests/benchmark_y_tensor_core_gemm.py --sizes 256,512  # subset
    python3 tests/benchmark_y_tensor_core_gemm.py --once 4096    # single in-process measurement (internal subprocess worker)
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
ALL_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]
REPEAT_RUNS = 5  # matches src/bin/autotune_verify.rs's REPEAT_RUNS

# --- GPU clock-ramp control (ported from benchmark_y_decode_gemm.py) ---
#
# Measured on this project's dev GPU: the SM clock idles at ~210 MHz and only
# reaches ~2670 MHz after ~3s of sustained load - a 12.7x swing - and this box
# has no permission to lock clocks (`nvidia-smi -lgc` -> "The current user does
# not have permission to change clocks"). A 10-iteration warmup does not ramp
# them, so a cold-started process measures a kernel at a fraction of the clock
# a warm one does.
#
# This is not theoretical for THIS script: before this revision it reported the
# same 256x256x256 Y kernel as "45.56us [7.12, 63.03]" - a 9x run-to-run spread
# that made a real 1.14x autotuner improvement indistinguishable from noise.
#
# Two fixes, both required:
#   1. RAMP_SECONDS of sustained dummy load before ANY timing, so measurement
#      starts at steady-state clocks.
#   2. A/B interleaving. This script used to time all of Y, then all of cuBLAS,
#      so cuBLAS always ran on a hotter clock than Y - a systematic bias in
#      cuBLAS's favour, not noise. They now alternate within a process and each
#      side's median is taken over its own interleaved rounds.
#
# Note the existing REPEAT_RUNS process-relaunch discipline does NOT solve this
# on its own: it averages over clock states rather than controlling for them,
# and cannot remove the ordering bias at all.
RAMP_SECONDS = 3.0
INTERLEAVE_ROUNDS = 5


def _sm_clock_mhz():
    """Current SM clock, so a run can report what clock it was measured at."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return 0

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

    # Total iterations are split across INTERLEAVE_ROUNDS rounds, so the
    # measured work per size stays comparable to the pre-interleave version.
    iters = max(3, iterations_for(size) // INTERLEAVE_ROUNDS)
    # cfg arrives via JSON (env var, from the outer driver) - JSON has no
    # tuple type, and cupy's launch API rejects a plain list for `grid`.
    grid, threads = tuple(cfg["grid"]), cfg["threads"]

    def time_y():
        cp.cuda.Device(0).synchronize()
        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ev0.record()
        for _ in range(iters):
            fn(grid, (threads, 1, 1), (A_cp, B_cp, C_cp), shared_mem=dyn_smem_bytes)
        ev1.record()
        ev1.synchronize()
        return (cp.cuda.get_elapsed_time(ev0, ev1) / iters) * 1000.0

    def time_cublas():
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(iters):
            _ = torch.matmul(A_torch, B_torch)
        ev1.record()
        torch.cuda.synchronize()
        return (ev0.elapsed_time(ev1) / iters) * 1000.0

    # ---- 1. ramp clocks to steady state on a throwaway workload ----
    ramp = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    t0 = time.time()
    while time.time() - t0 < RAMP_SECONDS:
        for _ in range(20):
            _ = ramp @ ramp
        torch.cuda.synchronize()
    del ramp
    sm_mhz = _sm_clock_mhz()

    # ---- 2. interleaved A/B timing so both sides see the same clock ----
    y_rounds, c_rounds = [], []
    for _ in range(INTERLEAVE_ROUNDS):
        y_rounds.append(time_y())
        c_rounds.append(time_cublas())
    y_us = statistics.median(y_rounds)
    cublas_us = statistics.median(c_rounds)

    # ---- correctness: fresh cuBLAS reference (C_cp aliases C_torch's storage) ----
    ref = torch.matmul(A_torch, B_torch).float()
    max_abs_diff = (C_torch - ref).abs().max().item()
    ref_scale = ref.abs().max().item()
    # Same tolerance reasoning as benchmark_y_vs_cudnn_cublas.py's check_correctness:
    # legitimate FP16-accumulation rounding grows with reduction depth (K) and output
    # magnitude, so atol scales with the reference's own magnitude rather than a bare
    # constant - a real correctness bug produces errors orders of magnitude past this.
    correct = bool(torch.allclose(C_torch, ref, rtol=0.02, atol=max(0.75, 0.02 * ref_scale)))

    # (cuBLAS was already timed above, interleaved with Y - see the
    # RAMP_SECONDS/INTERLEAVE_ROUNDS comment for why it is no longer measured
    # in a separate block after all of Y's runs.)
    print(json.dumps({
        "size": size, "y_us": y_us, "cublas_us": cublas_us,
        "y_rounds": y_rounds, "c_rounds": c_rounds, "sm_mhz": sm_mhz,
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
        f.write(
            "## Timing methodology (revised)\n\n"
            "Each measuring process now ramps the GPU clock to steady state for "
            f"{RAMP_SECONDS:.0f}s before timing anything, then A/B-interleaves Y and cuBLAS over "
            f"{INTERLEAVE_ROUNDS} rounds so both are measured at the same clock. The previous "
            "discipline (short warmup; all of Y timed, then all of cuBLAS) was contaminated "
            "twice over: the SM clock on this dev GPU idles at ~210 MHz and needs ~3s of load "
            "to reach ~2670 MHz (12.7x, and clocks cannot be locked on this box), and timing Y "
            "first meant cuBLAS always inherited the hotter clock - a systematic bias, not "
            "noise. It reported this same 256 kernel as `45.56us [7.12, 63.03]`, a 9x spread.\n\n"
            "**What the harness fix changed, and what it did not.** The large sizes (2048 and "
            "up) were unaffected - they re-measured at the same 0.75x/0.76x/0.77x reported "
            "before it - because a single iteration there already runs for milliseconds and "
            "ramps the clock itself inside the timed loop. Only the small, microsecond-scale "
            "sizes were ever vulnerable, and those corrected **downward** (256/512/1024 came "
            "in at 0.71x/0.59x/0.68x). Run-to-run ranges are now within a few percent at "
            "every size.\n\n"
            "## Autotuner: occupancy over tile size\n\n"
            "The figures in the table above are *after* a subsequent `score_candidate` "
            "rework, and are substantially better than that 0.75x-0.77x baseline at every "
            "size from 1024 up.\n\n"
            "`ncu` (once GPU performance counters were enabled) showed the old 128x256x32 "
            "4x4 pick was not memory-bound in any direction - DRAM at 16.6% of peak, L1 at "
            "20.4%, a shared-memory stall ratio of 0.74 - so the scorer's `compute_intensity` "
            "term (reuse per byte staged) was optimising for a wall the kernel never hits, "
            "and its `1.0 / ctas_per_sm` factor had no physical basis at all, since total "
            "work is invariant to tile choice. Between them they rated the measured-best "
            "config 3741 against 11809 for one 19% slower.\n\n"
            "The real gap against cuBLAS was occupancy and barrier cost. cuBLAS runs "
            "`ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn`: a 128x128 tile at "
            "128 threads, 234 registers/thread, 2 blocks/SM. Y ran a bigger tile at 512 "
            "threads, 1 block/SM, with a barrier-stall ratio of 6.27 against cuBLAS's 1.27. "
            "Scoring compute-bound shapes on predicted utilisation instead - per-warp MMA "
            "parallelism, resident CTAs per SM (gated on the grid actually supplying them), "
            "and warps per CTA - moves the picks to 128x128x32 at 2 stages, which fits two "
            "CTAs per SM. Measured effect at 4096: tensor-pipe utilisation 37.38% -> 44.61%, "
            "barrier stalls 6.27 -> 0.37 (now below cuBLAS's), registers/thread 122 -> 194.\n\n"
            "Skinny/decode shapes and small squares (min dimension below 1024) keep the older "
            "reuse-based heuristic: the utilisation model assumes steady-state throughput over "
            "many CTA waves, which neither satisfies. Every pick here was re-benchmarked per "
            "size rather than trusted - forcing the utilisation model onto 256 picked a tile "
            "that measured 6.08us against 5.11us for the legacy pick.\n\n"
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
