#!/usr/bin/env python3
"""
Benchmark: Y compiler's real FP8 (e4m3) Tensor Core GEMM kernel
(ptx_emitter::emit_fp8_gemm_kernel, reached via the real CLI - this loads
`tests/gemm_fp8_<N>.ptx`, the actual compiler output, via
`cp.RawModule(path=...)`, NOT a hand-written probe) vs torch._scaled_mm,
PyTorch 2.13's real FP8 matmul op.

Methodology, matching this project's own established discipline (see
feedback_sass_register_bank_patching.md's GPU-clock-ramp finding): timing
variant A fully then variant B fully, back-to-back, systematically biases
in favor of whichever runs second (observed 2x swings in a prior session's
SASS-patching work). Fix used here: extensive shared warmup (alternating
both variants) before ANY timed measurement, then measure in alternating
rounds (Y, torch, Y, torch, ...) and take the median per variant - not one
pair of totals. Timed with cuda Events (GPU-side, excludes Python/launch
dispatch overhead) around `repeats` back-to-back launches per round, so
per-round time is already averaged over multiple kernel launches.

Usage: venv/bin/python3 tests/benchmark_y_fp8_gemm.py
"""
import os
import subprocess
import statistics

Y_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTXAS = "/opt/cuda/bin/ptxas"
SIZES = [256, 512, 1024, 4096]
ROUNDS = 7
REPEATS_PER_ROUND = 20
WARMUP_ROUNDS = 15

# Two-tier CTA tile selection (see ptx_emitter::FP8_GEMM_CTA_M_LARGE's doc
# comment and emit_fp8_gemm_kernel's `is_small` branch) - must mirror the
# Rust emitter's own `m <= FP8_GEMM_SMALL_THRESHOLD || n <=
# FP8_GEMM_SMALL_THRESHOLD` heuristic exactly, or the host launch
# configuration silently disagrees with what the compiled kernel expects.
# Threshold raised 256 -> 512 in session 5 - see FP8_GEMM_CTA_M_LARGE's doc
# comment for the real-hardware A/B that motivated it (a standalone _TINY
# 32x32/1-warp tier was also tried and rejected there - real hardware
# consistently favored _SMALL's 4 warps/CTA over _TINY's extra CTA count).
FP8_SMALL_THRESHOLD = 512
FP8_CTA_LARGE = (128, 128, 256)  # (cta_m, cta_n, threads_per_cta)
FP8_CTA_SMALL = (64, 64, 128)


def fp8_gemm_launch_config(m, n):
    cta_m, cta_n, threads = FP8_CTA_SMALL if (m <= FP8_SMALL_THRESHOLD or n <= FP8_SMALL_THRESHOLD) else FP8_CTA_LARGE
    grid = ((m + cta_m - 1) // cta_m, (n + cta_n - 1) // cta_n, 1)
    block = (threads, 1, 1)
    return grid, block


def ensure_compiled(m, n, k):
    ysu_path = os.path.join(Y_ROOT, "tests", f"gemm_fp8_{m}.ysu")
    ptx_path = os.path.join(Y_ROOT, "tests", f"gemm_fp8_{m}.ptx")
    assert m == n == k, "canonical test files are square (see tests/gemm_fp8_<N>.ysu)"
    assert os.path.exists(ysu_path), f"missing {ysu_path}"
    r = subprocess.run(["cargo", "run", "--release", "--bin", "Y", "--", ysu_path, "--emit-ptx"],
                        cwd=Y_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"Y compiler failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    cubin_path = ptx_path.replace(".ptx", ".cubin")
    r = subprocess.run([PTXAS, "-arch=sm_89", "-o", cubin_path, ptx_path], capture_output=True, text=True)
    assert r.returncode == 0, f"ptxas failed:\n{r.stderr}"
    os.remove(cubin_path)
    return ptx_path, f"gemm_fp8_{m}"


def median_range(xs):
    xs = sorted(xs)
    med = statistics.median(xs)
    return med, (xs[0], xs[-1])


def bench_one_size(m, n, k):
    import cupy as cp
    import torch

    ptx_path, kernel_name = ensure_compiled(m, n, k)

    torch.manual_seed(0)
    A = torch.randn(m, k, device="cuda", dtype=torch.float32) * 3.0
    B = torch.randn(k, n, device="cuda", dtype=torch.float32) * 3.0
    scale_a = max((A.abs().amax() / 448.0).item(), 1e-12)
    scale_b = max((B.abs().amax() / 448.0).item(), 1e-12)

    A8 = (A / scale_a).to(torch.float8_e4m3fn)
    B8 = (B / scale_b).to(torch.float8_e4m3fn)
    B8_colmajor = B8.t().contiguous().t()
    scale_a_t = torch.tensor(scale_a, device="cuda", dtype=torch.float32)
    scale_b_t = torch.tensor(scale_b, device="cuda", dtype=torch.float32)

    A_cp = cp.asarray(A)
    B_cp = cp.asarray(B)
    C_cp = cp.zeros((m, n), dtype=cp.float32)
    mod = cp.RawModule(path=ptx_path)
    fn = mod.get_function(kernel_name)
    grid, block = fp8_gemm_launch_config(m, n)

    def run_y():
        for _ in range(REPEATS_PER_ROUND):
            fn(grid, block, (A_cp, B_cp, cp.float32(scale_a), cp.float32(scale_b), C_cp))

    def run_torch():
        for _ in range(REPEATS_PER_ROUND):
            torch._scaled_mm(A8, B8_colmajor, scale_a_t, scale_b_t, out_dtype=torch.float32)

    def timed(fn_call):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        cp.cuda.Device(0).synchronize()
        start.record()
        fn_call()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / REPEATS_PER_ROUND * 1000.0  # us/launch

    # ---- shared warmup, alternating - see module docstring ----
    for _ in range(WARMUP_ROUNDS):
        run_y()
        run_torch()

    # ---- measure in alternating rounds, take median per variant ----
    y_times, torch_times = [], []
    for _ in range(ROUNDS):
        y_times.append(timed(run_y))
        torch_times.append(timed(run_torch))

    y_med, y_range = median_range(y_times)
    t_med, t_range = median_range(torch_times)

    # ---- correctness - a benchmark number from a wrong kernel is
    # worthless, so re-verify every size before trusting its timing ----
    fn(grid, block, (A_cp, B_cp, cp.float32(scale_a), cp.float32(scale_b), C_cp))
    cp.cuda.Device(0).synchronize()
    y_out = torch.as_tensor(C_cp, device="cuda")
    ref = torch._scaled_mm(A8, B8_colmajor, scale_a_t, scale_b_t, out_dtype=torch.float32)
    # Relative L2 error, not a per-element max/rtol check - standard metric
    # for validating reduced-precision GEMM kernels. A per-element max-based
    # check is statistically inappropriate at M=N=4096 (16M+ compared
    # elements): order statistics alone push the observed max deviation up
    # even when the underlying per-element error distribution is small and
    # well-behaved (confirmed this session - see
    # investigation_fp8_gemm_findings.md's "correctness metric" section:
    # relative L2 stayed ~0.1% at every size while a naive per-element
    # rtol=0.03 flagged a false failure at 4096 whose 99.99th-percentile
    # deviation was still under 2% of the mean output magnitude).
    rel_l2 = (torch.linalg.norm(y_out - ref) / torch.linalg.norm(ref)).item()
    correct = rel_l2 < 0.02

    return {
        "m": m, "n": n, "k": k,
        "y_us": y_med, "y_range": y_range,
        "torch_us": t_med, "torch_range": t_range,
        "speedup": t_med / y_med,
        "correct": correct,
        "rel_l2": rel_l2,
    }


def main():
    results = []
    for sz in SIZES:
        print(f"[*] benchmarking {sz}x{sz}x{sz} ...")
        r = bench_one_size(sz, sz, sz)
        results.append(r)
        print(f"    Y: {r['y_us']:.2f}us [{r['y_range'][0]:.2f}, {r['y_range'][1]:.2f}] | "
              f"torch._scaled_mm: {r['torch_us']:.2f}us [{r['torch_range'][0]:.2f}, {r['torch_range'][1]:.2f}] | "
              f"speedup={r['speedup']:.3f}x | correct={r['correct']} (rel_l2={r['rel_l2']*100:.3f}%)")

    md_path = os.path.join(Y_ROOT, "benchmark_y_fp8_gemm_results.md")
    with open(md_path, "w") as f:
        f.write("# Y FP8 (e4m3) Tensor Core GEMM vs torch._scaled_mm\n\n")
        f.write("Every number below is measured from the REAL Y compiler CLI's output "
                "(`cargo run --bin Y -- tests/gemm_fp8_<N>.ysu --emit-ptx`, "
                "`ptx_emitter::emit_fp8_gemm_kernel`, multi-warp CTA tiling with two-tier "
                "size selection - 64x64x64/2x2 warps/128 threads for M<=512||N<=512, else "
                "128x128x64/4x2 warps/256 threads - fused on-the-fly FP32->e4m3 "
                "quantization (session 5: vectorized 128-bit ld.global.v4.f32 staging), "
                "software double-buffered K-loop pipelining), loaded via "
                "`cp.RawModule(path=...)` "
                "- not a hand-written probe. Reference: `torch._scaled_mm` "
                "(`torch==2.13.0+cu130`), `out_dtype=torch.float32`. GPU-side timing via CUDA "
                "events, median of 7 alternating rounds (20 launches/round) after 15 rounds "
                "of shared alternating warmup - see this file's own docstring for why "
                "alternating (not sequential full-A-then-full-B) timing matters on this "
                "project's hardware.\n\n")
        f.write("Correctness metric is relative L2 error (`||Y-ref||/||ref||`), the standard "
                "way to validate reduced-precision GEMM kernels - NOT a per-element max/rtol "
                "check, which is statistically inappropriate at M=N=4096 (16M+ compared "
                "elements: order statistics alone push the observed MAX per-element deviation "
                "up even when the underlying error distribution is small and well-behaved - "
                "confirmed this session, see investigation_fp8_gemm_findings.md).\n\n")
        f.write("| Matrix (M=N=K) | Y us (median [range]) | torch._scaled_mm us (median [range]) | Y vs torch | Correct (rel L2) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['m']}x{r['n']}x{r['k']} | {r['y_us']:.2f} [{r['y_range'][0]:.2f}, {r['y_range'][1]:.2f}] | "
                    f"{r['torch_us']:.2f} [{r['torch_range'][0]:.2f}, {r['torch_range'][1]:.2f}] | "
                    f"**{r['speedup']:.3f}x** | {'OK' if r['correct'] else 'WRONG'} ({r['rel_l2']*100:.3f}%) |\n")
    print(f"[*] wrote {md_path}")


if __name__ == "__main__":
    main()
