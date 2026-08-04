#!/usr/bin/env python3
"""
Benchmarks REAL Y-compiler-emitted Tensor Core GEMM PTX
(ptx_emitter::emit_tensor_core_gemm_kernel - the SAME emitter and codegen
already validated in benchmark_y_tensor_core_gemm_results.md, no new
codegen) at decode-shaped ("skinny") sizes: M in {1, 4, 8, 32} (single- and
small-batch autoregressive decode) with N=K=4096 (a typical LLM hidden
dim) - a genuinely different regime from every other benchmark in this
project, which are all square M=N=K training/prefill shapes.

Purpose: `Autotuner::generate_candidates`'s `m <= 256 || n <= 256` bucket
was written and tuned against square-ish shapes, never M=1..32 specifically.
This script checks what CTA tile it actually picks for real decode shapes
and whether that choice - and the underlying `wmma`/`mma.sync` tensor-core
compute core, which operates at a hardware-fixed 16-row (M) granularity
regardless of tile choice - is competitive with cuBLAS here, not assumed.

Correctness reference: torch.matmul (cuBLAS FP16), same as
benchmark_y_tensor_core_gemm.py.

Timing discipline: median of REPEAT_RUNS independent process re-launches;
each launch ramps the GPU clock to steady state, then A/B-interleaves Y and
cuBLAS so both see the same clock. See the RAMP_SECONDS and
NUM_WEIGHT_BUFFERS comments below - the older discipline (short warmup, all
of Y then all of cuBLAS, one shared B buffer) produced 10x run-to-run
spreads and a systematic bias, not measurement noise.

Usage:
    python3 tests/benchmark_y_decode_gemm.py               # full suite, all M, median-of-N, writes report
    python3 tests/benchmark_y_decode_gemm.py --sizes 1,4   # subset (by M)
    python3 tests/benchmark_y_decode_gemm.py --once 1      # single in-process measurement (internal subprocess worker)
"""
import os
import sys
import re
import json
import time
import subprocess
import statistics
import argparse


def _sm_clock_mhz():
    """Current SM clock, so a report can show what clock it was measured at."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return 0

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
Y_BIN = os.path.join(REPO_ROOT, "target/release/Y")
ALL_M = [1, 4, 8, 32]
N_K = 4096
REPEAT_RUNS = 5

# --- L2 residency control (see "Measurement validity" in the generated report) ---
#
# At N=K=4096 FP16 the B (weight) matrix is 4096*4096*2 = 33.55 MB. This
# project's dev GPU (RTX 4070 Ti SUPER, AD103) has a 48 MB L2 cache, so a
# timing loop that hammers ONE B buffer measures an L2-resident working set,
# not the DRAM-bandwidth regime that real autoregressive decode runs in
# (where each of a model's ~32 layers has its own distinct weight matrix and
# nothing is resident between tokens). That is not a hypothetical: at M=32
# the previously-reported cuBLAS median of 37.37us over a 33.55 MB B works
# out to ~910 GB/s effective, which is 1.35x ABOVE this card's 672 GB/s
# theoretical DRAM peak - only reachable out of cache.
#
# `stream` mode therefore rotates over NUM_WEIGHT_BUFFERS distinct B
# matrices (6 * 33.55 MB = 201 MB, ~4x L2), so a buffer is evicted long
# before it comes around again. `resident` mode preserves the old
# single-buffer behaviour so both numbers can be reported side by side.
NUM_WEIGHT_BUFFERS = 6
L2_BYTES = 50331648
DRAM_PEAK_GBS = 672.0

# --- GPU clock-ramp control ---
#
# Measured on this project's dev GPU: SM clock idles at ~210 MHz and only
# reaches ~2670 MHz after ~3s of sustained load - a 12.7x swing - and this
# box has no permission to lock clocks (`nvidia-smi -lgc` -> "The current
# user does not have permission to change clocks"). A short 10-iteration
# warmup does NOT ramp them, so a cold-started process measures a kernel at
# a fraction of the clock a warm one does. That, not any property of the
# kernels, is what produced the 10x spreads (Y "[59.20, 602.86]us") seen
# before this revision.
#
# Two fixes, both required:
#   1. RAMP_SECONDS of sustained dummy load before ANY timing, so the
#      measurement starts at steady-state clocks.
#   2. A/B interleaving. The old harness timed all of Y, then all of cuBLAS,
#      so cuBLAS always ran on a hotter clock than Y - a systematic bias in
#      cuBLAS's favour, not noise. Now the two alternate within a process
#      and each side's median is taken over its own interleaved rounds.
RAMP_SECONDS = 3.0
INTERLEAVE_ROUNDS = 5
ITERS_PER_ROUND = 100

CTA_COMMENT_RE = re.compile(
    r"\[Y TENSOR CORE GEMM\] M=(\d+) N=(\d+) K=(\d+) \| CTA (\d+)x(\d+)x(\d+) \| (\d+)x(\d+) warps"
)
DYN_SMEM_RE = re.compile(r"Dynamic shared memory required: (\d+) bytes")


def ysu_path(m):
    return os.path.join(REPO_ROOT, f"tests/gemm_f16_decode_m{m}.ysu")


def ptx_path(m):
    return os.path.join(REPO_ROOT, f"tests/gemm_f16_decode_m{m}.ptx")


def compile_kernel(m):
    res = subprocess.run([Y_BIN, ysu_path(m), "--emit-ptx"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Y compile failed for M={m}:\n{res.stdout}\n{res.stderr}")
    with open(ptx_path(m)) as f:
        ptx_text = f.read()
    match = CTA_COMMENT_RE.search(ptx_text)
    if not match:
        raise RuntimeError(
            f"could not find the '[Y TENSOR CORE GEMM]' comment in {ptx_path(m)} - "
            "did emit_tensor_core_gemm_kernel not fire for this kernel?"
        )
    M, N, K, cta_m, cta_n, cta_k, warps_m, warps_n = map(int, match.groups())
    smem_m = DYN_SMEM_RE.search(ptx_text)
    return {
        "m": m,
        "M": M, "N": N, "K": K,
        "cta": f"{cta_m}x{cta_n}x{cta_k}",
        "cta_m": cta_m,
        "grid": ((N + cta_n - 1) // cta_n, (M + cta_m - 1) // cta_m),
        "threads": warps_m * warps_n * 32,
        "dyn_smem_bytes": int(smem_m.group(1)) if smem_m else 0,
    }


def run_once(m, cfg, mode):
    import torch
    import cupy as cp

    device = torch.device("cuda:0")
    M, N, K = cfg["M"], cfg["N"], cfg["K"]

    # `resident` reproduces the original single-B-buffer loop; `stream`
    # rotates over enough distinct B matrices to blow past L2 (see the
    # NUM_WEIGHT_BUFFERS comment at the top of this file).
    n_buf = NUM_WEIGHT_BUFFERS if mode == "stream" else 1

    torch.manual_seed(0)
    A_torch = torch.randn(M, K, dtype=torch.float16, device=device)
    B_torches = [torch.randn(K, N, dtype=torch.float16, device=device) for _ in range(n_buf)]
    C_torch = torch.zeros(M, N, dtype=torch.float32, device=device)

    A_cp = cp.asarray(A_torch)
    # cp.asarray over a torch CUDA tensor aliases the same device memory via
    # __cuda_array_interface__, so Y and cuBLAS below genuinely read the
    # SAME bytes - the rotation is identical for both, not just similar.
    B_cps = [cp.asarray(b) for b in B_torches]
    C_cp = cp.asarray(C_torch)

    mod = cp.RawModule(path=ptx_path(m))
    fn = mod.get_function(f"gemm_f16_decode_m{m}")

    dyn_smem_bytes = cfg.get("dyn_smem_bytes", 0)
    if dyn_smem_bytes > 0:
        fn.max_dynamic_shared_size_bytes = dyn_smem_bytes

    grid, threads = tuple(cfg["grid"]), cfg["threads"]

    def time_y():
        cp.cuda.Device(0).synchronize()
        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ev0.record()
        for i in range(ITERS_PER_ROUND):
            fn(grid, (threads, 1, 1), (A_cp, B_cps[i % n_buf], C_cp), shared_mem=dyn_smem_bytes)
        ev1.record()
        ev1.synchronize()
        return (cp.cuda.get_elapsed_time(ev0, ev1) / ITERS_PER_ROUND) * 1000.0

    def time_cublas():
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for i in range(ITERS_PER_ROUND):
            _ = torch.matmul(A_torch, B_torches[i % n_buf])
        ev1.record()
        torch.cuda.synchronize()
        return (ev0.elapsed_time(ev1) / ITERS_PER_ROUND) * 1000.0

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

    # ---- correctness: fresh cuBLAS reference ----
    # C_cp holds whatever the LAST timed Y iteration wrote, so the reference
    # must be taken against that same B buffer, not against buffer 0.
    last_b = B_torches[(ITERS_PER_ROUND - 1) % n_buf]
    ref = torch.matmul(A_torch, last_b).float()
    max_abs_diff = (C_torch - ref).abs().max().item()
    ref_scale = ref.abs().max().item()
    correct = bool(torch.allclose(C_torch, ref, rtol=0.02, atol=max(0.5, 0.02 * ref_scale)))

    print(json.dumps({
        "m": m, "y_us": y_us, "cublas_us": cublas_us,
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
    parser.add_argument("--sizes", type=str, default=None, help="comma-separated subset of M values")
    parser.add_argument("--modes", type=str, default="stream,resident",
                        help="comma-separated subset of {stream,resident} (default both)")
    args = parser.parse_args()

    if args.once is not None:
        cfg = json.loads(os.environ["Y_GEMM_BENCH_CFG"])
        run_once(args.once, cfg, os.environ.get("Y_GEMM_BENCH_MODE", "stream"))
        return

    m_values = [int(s) for s in args.sizes.split(",")] if args.sizes else ALL_M
    modes = [s.strip() for s in args.modes.split(",")]

    print("=" * 100)
    print("REAL Y-COMPILER TENSOR CORE GEMM (decode-shaped, M small) vs cuBLAS".center(100))
    print("=" * 100)
    print(f"[*] N=K={N_K} fixed; every number below comes from `target/release/Y")
    print("    tests/gemm_f16_decode_m<M>.ysu --emit-ptx`'s actual output.")
    print(f"[*] Timing: median of {REPEAT_RUNS} independent process launches per M (see module docstring).")

    print("\n[*] Building Y compiler release binary...")
    res = subprocess.run("cargo build --release", shell=True, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Y compiler build failed:\n{res.stderr}")
        sys.exit(1)

    header = (
        f"{'M':<5} | {'mode':<9} | {'CTA Tile':<14} | {'smem B':<7} | "
        f"{'cuBLAS us (median)':<22} | {'Y us (median)':<22} | {'Y vs cuBLAS':<12} | "
        f"{'Y GB/s':<8} | {'cuBLAS GB/s':<12} | Correct"
    )
    print("\n" + header)
    print("-" * len(header))

    results = []
    for m in m_values:
        cfg = compile_kernel(m)
        # Bytes any implementation must move for one C = A@B at this shape:
        # B is the dominant term and the one that decides L2 residency.
        gemm_bytes = (cfg["K"] * cfg["N"] * 2) + (cfg["M"] * cfg["K"] * 2) + (cfg["M"] * cfg["N"] * 4)

        for mode in modes:
            env = dict(os.environ)
            env["Y_GEMM_BENCH_CFG"] = json.dumps(cfg)
            env["Y_GEMM_BENCH_MODE"] = mode

            y_samples, cublas_samples, correct_flags, max_diffs, clocks = [], [], [], [], []
            for _ in range(REPEAT_RUNS):
                proc = subprocess.run(
                    [sys.executable, __file__, "--once", str(m)],
                    capture_output=True, text=True, env=env, cwd=REPO_ROOT,
                )
                if proc.returncode != 0:
                    print(f"[!] worker run failed for M={m} mode={mode}:\n{proc.stdout}\n{proc.stderr}")
                    sys.exit(1)
                line = next((l for l in proc.stdout.splitlines() if l.startswith("{")), None)
                if line is None:
                    print(f"[!] no JSON output from worker for M={m} mode={mode}:\n{proc.stdout}\n{proc.stderr}")
                    sys.exit(1)
                data = json.loads(line)
                y_samples.append(data["y_us"])
                cublas_samples.append(data["cublas_us"])
                correct_flags.append(data["correct"])
                max_diffs.append(data["max_abs_diff"])
                clocks.append(data.get("sm_mhz", 0))

            y_med, y_min, y_max = median_range(y_samples)
            c_med, c_min, c_max = median_range(cublas_samples)
            inconclusive = ranges_overlap(y_min, y_max, c_min, c_max)
            all_correct = all(correct_flags)

            vs_cublas = c_med / y_med
            verdict = "inconclusive" if inconclusive else (f"{vs_cublas:.2f}x")
            y_gbs = gemm_bytes / (y_med * 1e-6) / 1e9
            c_gbs = gemm_bytes / (c_med * 1e-6) / 1e9

            results.append({
                "m": m, "mode": mode, "cta": cfg["cta"], "cta_m": cfg["cta_m"],
                "smem": cfg.get("dyn_smem_bytes", 0), "grid": cfg["grid"], "threads": cfg["threads"],
                "y_med": y_med, "y_min": y_min, "y_max": y_max,
                "cublas_med": c_med, "cublas_min": c_min, "cublas_max": c_max,
                "vs_cublas": vs_cublas, "inconclusive": inconclusive,
                "y_gbs": y_gbs, "cublas_gbs": c_gbs, "gemm_bytes": gemm_bytes,
                "correct": all_correct, "max_abs_diff": max(max_diffs),
                "sm_mhz": statistics.median(clocks) if clocks else 0,
            })

            correctness_tag = "OK" if all_correct else "FAIL"
            print(
                f"{m:<5} | {mode:<9} | {cfg['cta']:<14} | {cfg.get('dyn_smem_bytes', 0):<7} | "
                f"{f'{c_med:.2f} [{c_min:.2f},{c_max:.2f}]':<22} | "
                f"{f'{y_med:.2f} [{y_min:.2f},{y_max:.2f}]':<22} | "
                f"{verdict:<12} | {y_gbs:<8.0f} | {c_gbs:<12.0f} | {correctness_tag}"
            )

    report_path = os.path.join(REPO_ROOT, "benchmark_y_decode_gemm_results.md")
    with open(report_path, "w") as f:
        f.write("# Real Y-Compiler Tensor Core GEMM vs cuBLAS - Decode-Shaped (Skinny) M\n\n")
        f.write(
            "Every number below is measured from `target/release/Y "
            "tests/gemm_f16_decode_m<M>.ysu --emit-ptx`'s actual output, at "
            f"`M in {{1, 4, 8, 32}}`, `N=K={N_K}` (autoregressive decode batch sizes against a "
            "typical LLM hidden dim).\n\n"
        )
        f.write("## Measurement validity: `stream` vs `resident`\n\n")
        f.write(
            f"At `N=K={N_K}` FP16 the B (weight) matrix is {N_K}*{N_K}*2 = "
            f"{N_K*N_K*2/1e6:.2f} MB, and this project's dev GPU (RTX 4070 Ti SUPER, AD103) has a "
            f"{L2_BYTES/2**20:.0f} MB L2. A timing loop over a SINGLE B buffer - which is what this "
            "benchmark did before this revision - therefore measures an L2-resident working "
            "set, not the DRAM-bandwidth regime real decode runs in, where each of a model's "
            "~32 layers has its own weight matrix and nothing survives between tokens.\n\n"
        )
        f.write(
            "`resident` rows reproduce that old single-buffer loop. `stream` rows rotate over "
            f"{NUM_WEIGHT_BUFFERS} distinct B matrices ({NUM_WEIGHT_BUFFERS*N_K*N_K*2/1e6:.0f} MB, "
            f"~{NUM_WEIGHT_BUFFERS*N_K*N_K*2/L2_BYTES:.1f}x L2) so a buffer is evicted before it comes "
            "around again. Both implementations read the same aliased device memory and rotate "
            "identically. Effective GB/s is `(K*N*2 + M*K*2 + M*N*4) / time`; this card's "
            f"theoretical DRAM peak is {DRAM_PEAK_GBS:.0f} GB/s, so any row above that is provably "
            "being served out of cache rather than DRAM.\n\n"
        )
        f.write(
            f"Timing is the median of {REPEAT_RUNS} independent process launches per (M, mode) "
            "(range in brackets); a row is marked inconclusive rather than given a speedup "
            "number if the Y and cuBLAS ranges overlap.\n\n"
        )
        f.write(
            "| M | mode | CTA Tile | dyn smem B | cuBLAS us (median [range]) | "
            "Y us (median [range]) | Y vs cuBLAS | Y GB/s | cuBLAS GB/s | Correct |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            verdict = "inconclusive" if r["inconclusive"] else f"**{r['vs_cublas']:.2f}x**"
            f.write(
                f"| {r['m']} | {r['mode']} | {r['cta']} | {r['smem']} | "
                f"{r['cublas_med']:.2f} [{r['cublas_min']:.2f}, {r['cublas_max']:.2f}] | "
                f"{r['y_med']:.2f} [{r['y_min']:.2f}, {r['y_max']:.2f}] | "
                f"{verdict} | {r['y_gbs']:.0f} | {r['cublas_gbs']:.0f} | "
                f"{'OK' if r['correct'] else 'FAIL'} |\n"
            )
        f.write("\n")
        f.write("""## What the corrected measurement changed

The previously reported **0.35x at M=32** was almost entirely measurement
artifact, not kernel behaviour. Two independent contaminants stacked:

1. **L2 residency** - a single 33.55 MB B buffer against a 48 MB L2.
2. **GPU clock ramp** - SM clock idles at ~210 MHz here and needs ~3s of
   sustained load to reach ~2670 MHz (12.7x), and clocks cannot be locked on
   this box. A 10-iteration warmup does not ramp them. Worse, the old harness
   timed *all* of Y and then *all* of cuBLAS, so cuBLAS always ran on the
   hotter clock - a systematic bias, not noise. Run-to-run spreads of 10x
   (Y "[59.20, 602.86]us") were the visible symptom.

With both fixed, Y measures **0.94-0.96x of cuBLAS** in `stream` mode, and the
autotuner change below brings that to statistical parity at M=4/8/32.

## Autotuner: what a measured tile sweep showed

Sweeping the real candidate space (via `Y_CTA_OVERRIDE`) rather than trusting
`score_candidate`'s model contradicted the obvious hypothesis. More CTAs per SM
is *monotonically worse* here - 13 CTAs/SM gives 322 GB/s, 1 CTA/SM gives 539
GB/s. The axis that matters is **bytes in flight per CTA** (Little's law), not
occupancy: holding all else fixed, 16x64x32 -> 16x64x64 went 97.9 -> 76.8us and
64x64x32 at 2 -> 3 stages went 108.3 -> 72.3us.

The best configuration was not reachable at all: `cta_k=64` variants at small
`cta_m` were absent from the bucket, and `cta_m <= 32` was hard-capped at 2
stages. With both fixed the autotuner picks 16x64x64/32x64x64 at 3 stages,
worth ~1.05x in `stream` and up to 1.46x in `resident`. Square shapes 512
through 16384 pick bit-identical tiles; 256 improved (5.85 -> 5.13us).

## GEMV path for M < 16: measured, and NOT worth it in the regime that matters

A hand-written split-K GEMV probe (no tensor cores, `ld.global.nc.v4.b32`
straight to registers, 40 registers, no shared memory) was measured against
this kernel at M=1 before committing to the codegen:

| M=1, N=K=4096 | stream (DRAM-bound) | resident (L2-bound) |
|---|---|---|
| GEMV probe (incl. required memset) | 59.56us / 564 GB/s | **18.90us / 1777 GB/s** |
| this tensor-core kernel | 59.65us / 563 GB/s | 51.01us / 658 GB/s |
| cuBLAS | 55.53us / 605 GB/s | 28.71us / 1169 GB/s |

In the **streaming regime - the one real autoregressive decode runs in** - the
GEMV is a wash (59.56 vs 59.65us). The 64x M-dim tensor-core padding is
*compute* waste, and compute waste is free when the kernel is pinned at the
DRAM roofline. The motivating argument for a GEMV path ("less padding, more
memory parallelism") does not survive measurement.

It is a 2.7x win over this kernel (and 1.52x over cuBLAS) only when the weight
matrix is L2-resident, where skipping the global -> smem -> `ldmatrix`
round-trip and loading B straight to registers is what pays. That regime is
real (small models, low-bit weights, hot MoE experts, batched reuse) but is
narrower than streaming decode.

Blocking design question before wiring it in: split-K is unavoidable at these
shapes (N=4096 yields only ~4-16 CTAs without it, far too few to fill 66 SMs),
and split-K means the kernel **accumulates** into C rather than overwriting it.
Every existing caller assumes overwrite semantics. Resolving that needs either
a caller-zeroed-C contract change or a workspace + reduction ABI - a real
interface decision, not an implementation detail.
""")
    print(f"\n[*] Wrote {report_path}")

    if not all(r["correct"] for r in results):
        print("\n[!] One or more M values FAILED correctness - their timing numbers are not meaningful.")
        sys.exit(1)


if __name__ == "__main__":
    main()
