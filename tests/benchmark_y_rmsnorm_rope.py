#!/usr/bin/env python3
"""
Benchmarks REAL Y-compiler-emitted fused Add+RMSNorm and RoPE PTX
(ptx_emitter::emit_rmsnorm_residual_kernel / emit_rope_kernel, dispatched
via kernel name + param-shape detection - no `@tile`, see
tests/rmsnorm_residual_4096.ysu / tests/rope_128.ysu) against eager
PyTorch equivalents, on each kernel's actual compiled output.

RMSNorm+residual: h = X + Residual; Out = RMSNorm(h) * Weight; writes both
Out and h (as NewResidual) - the fused "add & RMSNorm" pattern real
inference engines use. Eager baseline: h = X + Residual (materialized),
variance = h.pow(2).mean(-1) (a separate reduction pass reading h back),
Out = h * rsqrt(variance + eps) * Weight - real ATen ops, no fusion,
consistent with how a naive PyTorch RMSNorm module is actually written.

RoPE: rotates each token's head_dim query/key vector by a
position-dependent angle, computed entirely on-device (no precomputed
cos/sin table). Eager baseline: builds the same cos/sin table PyTorch-side
(matching the interleaved-pairs convention this kernel uses - see
emit_rope_kernel's doc comment) and applies the same rotation via tensor
ops - the realistic unfused cost of NOT fusing the trig computation into
the elementwise kernel.

Correctness reference: fresh eager PyTorch computation each run, FP16
throughout (matching the kernels' own FP16 in/out contract), compared via
torch.allclose.

Timing discipline: median of REPEAT_RUNS independent process re-launches,
each doing its own in-process warmup + N-iteration cuda-event average -
same discipline as the other benchmark_y_*.py scripts in this directory.

Usage:
    python3 tests/benchmark_y_rmsnorm_rope.py                  # both kernels, all row counts, writes report
    python3 tests/benchmark_y_rmsnorm_rope.py --kernel rmsnorm  # just RMSNorm+residual
    python3 tests/benchmark_y_rmsnorm_rope.py --kernel rope     # just RoPE
"""
import os
import sys
import json
import subprocess
import statistics
import argparse

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
Y_BIN = os.path.join(REPO_ROOT, "target/release/Y")
ROW_COUNTS = [128, 1024, 8192]
REPEAT_RUNS = 5
ITERS = 500
HIDDEN_DIM = 4096
HEAD_DIM = 128


def build():
    res = subprocess.run("cargo build --release", shell=True, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Y compiler build failed:\n{res.stderr}")
        sys.exit(1)


def compile_once(ysu_name):
    res = subprocess.run([Y_BIN, f"tests/{ysu_name}.ysu", "--emit-ptx"], capture_output=True, text=True, cwd=REPO_ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"Y compile failed for {ysu_name}:\n{res.stdout}\n{res.stderr}")


def run_once_rmsnorm(rows):
    import torch
    import cupy as cp

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    X = torch.randn(rows, HIDDEN_DIM, dtype=torch.float16, device=device)
    Residual = torch.randn(rows, HIDDEN_DIM, dtype=torch.float16, device=device)
    Weight = torch.randn(HIDDEN_DIM, dtype=torch.float16, device=device) * 0.1 + 1.0
    Out = torch.zeros(rows, HIDDEN_DIM, dtype=torch.float16, device=device)
    NewResidual = torch.zeros(rows, HIDDEN_DIM, dtype=torch.float16, device=device)

    X_cp, Res_cp, W_cp = cp.asarray(X), cp.asarray(Residual), cp.asarray(Weight)
    Out_cp, NewRes_cp = cp.asarray(Out), cp.asarray(NewResidual)

    mod = cp.RawModule(path=os.path.join(REPO_ROOT, "tests/rmsnorm_residual_4096.ptx"))
    fn = mod.get_function("rmsnorm_residual_4096")

    # 256 threads/row (8 warps), not 32 - must match
    # emit_rmsnorm_residual_kernel's own num_warps for HIDDEN_DIM=4096 (see
    # benchmark_y_vs_flashinfer.py's copy of this comment for the formula).
    grid, threads = (rows, 1), (256, 1, 1)

    for _ in range(10):
        fn(grid, threads, (X_cp, Res_cp, W_cp, Out_cp, NewRes_cp))
    cp.cuda.Device(0).synchronize()
    y_start, y_end = cp.cuda.Event(), cp.cuda.Event()
    y_start.record()
    for _ in range(ITERS):
        fn(grid, threads, (X_cp, Res_cp, W_cp, Out_cp, NewRes_cp))
    y_end.record()
    y_end.synchronize()
    y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / ITERS) * 1000.0

    # ---- correctness: fresh eager reference ----
    h_ref = X + Residual
    variance = h_ref.float().pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + 1e-5)
    out_ref = (h_ref.float() * inv_rms).to(torch.float16) * Weight

    max_abs_diff_out = (Out.float() - out_ref.float()).abs().max().item()
    max_abs_diff_res = (NewResidual.float() - h_ref.float()).abs().max().item()
    correct = bool(
        torch.allclose(Out, out_ref, rtol=0.02, atol=0.02)
        and torch.allclose(NewResidual, h_ref, rtol=0.01, atol=0.01)
    )

    # ---- eager PyTorch: warmup + timed average ----
    torch.cuda.synchronize()
    for _ in range(10):
        h = X + Residual
        var = h.float().pow(2).mean(-1, keepdim=True)
        inv = torch.rsqrt(var + 1e-5)
        _ = (h.float() * inv).to(torch.float16) * Weight
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(ITERS):
        h = X + Residual
        var = h.float().pow(2).mean(-1, keepdim=True)
        inv = torch.rsqrt(var + 1e-5)
        _ = (h.float() * inv).to(torch.float16) * Weight
    end_evt.record()
    torch.cuda.synchronize()
    eager_us = (start_evt.elapsed_time(end_evt) / ITERS) * 1000.0

    print(json.dumps({
        "rows": rows, "y_us": y_us, "eager_us": eager_us, "correct": correct,
        "max_abs_diff": max(max_abs_diff_out, max_abs_diff_res),
    }))


def run_once_rope(rows):
    import torch
    import cupy as cp
    import math

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    X = torch.randn(rows, HEAD_DIM, dtype=torch.float16, device=device)
    Positions = torch.randint(0, 4096, (rows,), dtype=torch.int32, device=device)
    Out = torch.zeros(rows, HEAD_DIM, dtype=torch.float16, device=device)

    X_cp, Pos_cp, Out_cp = cp.asarray(X), cp.asarray(Positions), cp.asarray(Out)

    mod = cp.RawModule(path=os.path.join(REPO_ROOT, "tests/rope_128.ptx"))
    fn = mod.get_function("rope_128")

    grid, threads = (rows, 1), (32, 1, 1)

    for _ in range(10):
        fn(grid, threads, (X_cp, Pos_cp, Out_cp))
    cp.cuda.Device(0).synchronize()
    y_start, y_end = cp.cuda.Event(), cp.cuda.Event()
    y_start.record()
    for _ in range(ITERS):
        fn(grid, threads, (X_cp, Pos_cp, Out_cp))
    y_end.record()
    y_end.synchronize()
    y_us = (cp.cuda.get_elapsed_time(y_start, y_end) / ITERS) * 1000.0

    # ---- correctness: fresh eager reference (interleaved-pairs convention,
    # matching emit_rope_kernel's own doc comment exactly) ----
    i_idx = torch.arange(HEAD_DIM // 2, dtype=torch.float32, device=device)
    inv_freq = 10000.0 ** (-2.0 * i_idx / HEAD_DIM)  # [head_dim/2]
    pos_f = Positions.float().unsqueeze(1)  # [rows, 1]
    theta = pos_f * inv_freq.unsqueeze(0)   # [rows, head_dim/2]
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)

    x_f = X.float().view(rows, HEAD_DIM // 2, 2)
    x0, x1 = x_f[..., 0], x_f[..., 1]
    out0 = x0 * cos_t - x1 * sin_t
    out1 = x0 * sin_t + x1 * cos_t
    out_ref = torch.stack([out0, out1], dim=-1).reshape(rows, HEAD_DIM).to(torch.float16)

    max_abs_diff = (Out.float() - out_ref.float()).abs().max().item()
    correct = bool(torch.allclose(Out, out_ref, rtol=0.02, atol=0.02))

    # ---- eager PyTorch: warmup + timed average (rebuild sin/cos table every
    # iteration - the realistic unfused cost, see module docstring) ----
    torch.cuda.synchronize()
    for _ in range(10):
        theta = pos_f * inv_freq.unsqueeze(0)
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        x_f = X.float().view(rows, HEAD_DIM // 2, 2)
        x0, x1 = x_f[..., 0], x_f[..., 1]
        out0 = x0 * cos_t - x1 * sin_t
        out1 = x0 * sin_t + x1 * cos_t
        _ = torch.stack([out0, out1], dim=-1).reshape(rows, HEAD_DIM).to(torch.float16)
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(ITERS):
        theta = pos_f * inv_freq.unsqueeze(0)
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        x_f = X.float().view(rows, HEAD_DIM // 2, 2)
        x0, x1 = x_f[..., 0], x_f[..., 1]
        out0 = x0 * cos_t - x1 * sin_t
        out1 = x0 * sin_t + x1 * cos_t
        _ = torch.stack([out0, out1], dim=-1).reshape(rows, HEAD_DIM).to(torch.float16)
    end_evt.record()
    torch.cuda.synchronize()
    eager_us = (start_evt.elapsed_time(end_evt) / ITERS) * 1000.0

    print(json.dumps({
        "rows": rows, "y_us": y_us, "eager_us": eager_us, "correct": correct,
        "max_abs_diff": max_abs_diff,
    }))


def median_range(values):
    return statistics.median(values), min(values), max(values)


def ranges_overlap(a_min, a_max, b_min, b_max):
    return a_min <= b_max and b_min <= a_max


def run_suite(kernel_name, ysu_name, run_once_fn, report_path, title):
    print("\n" + "=" * 100)
    print(title.center(100))
    print("=" * 100)
    compile_once(ysu_name)

    header = f"{'Rows':<10} | {'Eager PyTorch us (median)':<26} | {'Y Fused us (median)':<22} | {'Y vs Eager':<12} | Correct"
    print("\n" + header)
    print("-" * len(header))

    results = []
    for rows in ROW_COUNTS:
        y_samples, eager_samples, correct_flags, max_diffs = [], [], [], []
        for _ in range(REPEAT_RUNS):
            env = dict(os.environ)
            env["Y_BENCH_KERNEL"] = kernel_name
            env["Y_BENCH_ROWS"] = str(rows)
            proc = subprocess.run(
                [sys.executable, __file__, "--once"],
                capture_output=True, text=True, env=env, cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                print(f"[!] worker run failed for rows={rows}:\n{proc.stdout}\n{proc.stderr}")
                sys.exit(1)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("{")), None)
            if line is None:
                print(f"[!] no JSON output from worker for rows={rows}:\n{proc.stdout}\n{proc.stderr}")
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
        verdict = "inconclusive" if inconclusive else f"{vs_eager:.2f}x"

        results.append({
            "rows": rows, "y_med": y_med, "y_min": y_min, "y_max": y_max,
            "eager_med": e_med, "eager_min": e_min, "eager_max": e_max,
            "vs_eager": vs_eager, "inconclusive": inconclusive,
            "correct": all_correct, "max_abs_diff": max(max_diffs),
        })
        print(
            f"{rows:<10} | {f'{e_med:.2f} [{e_min:.2f},{e_max:.2f}]':<26} | "
            f"{f'{y_med:.2f} [{y_min:.2f},{y_max:.2f}]':<22} | {verdict:<12} | "
            f"{'OK' if all_correct else 'FAIL'} (max|diff|={max(max_diffs):.4f})"
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kernel", choices=["rmsnorm", "rope", "both"], default="both")
    args = parser.parse_args()

    if args.once:
        kernel = os.environ["Y_BENCH_KERNEL"]
        rows = int(os.environ["Y_BENCH_ROWS"])
        if kernel == "rmsnorm":
            run_once_rmsnorm(rows)
        else:
            run_once_rope(rows)
        return

    print("[*] Building Y compiler release binary...")
    build()

    all_results = {}
    if args.kernel in ("rmsnorm", "both"):
        all_results["rmsnorm"] = run_suite(
            "rmsnorm", "rmsnorm_residual_4096", run_once_rmsnorm,
            "benchmark_y_rmsnorm_rope_results.md",
            f"REAL Y FUSED ADD+RMSNORM (hidden_dim={HIDDEN_DIM}) vs EAGER PYTORCH",
        )
    if args.kernel in ("rope", "both"):
        all_results["rope"] = run_suite(
            "rope", "rope_128", run_once_rope,
            "benchmark_y_rmsnorm_rope_results.md",
            f"REAL Y FUSED ROPE (head_dim={HEAD_DIM}) vs EAGER PYTORCH",
        )

    report_path = os.path.join(REPO_ROOT, "benchmark_y_rmsnorm_rope_results.md")
    with open(report_path, "w") as f:
        f.write("# Real Y-Compiler Fused Add+RMSNorm and RoPE vs Eager PyTorch\n\n")
        f.write(
            "Every number below is measured from the real Y CLI's `--emit-ptx` output "
            "(`ptx_emitter::emit_rmsnorm_residual_kernel` / `emit_rope_kernel`, dispatched by "
            "kernel name + param shape - no `@tile` - see `tests/rmsnorm_residual_4096.ysu` / "
            "`tests/rope_128.ysu`), loaded and run as-is (`cp.RawModule(path=...)`).\n\n"
        )
        f.write(
            f"Timing is the median of {REPEAT_RUNS} independent process launches per row count "
            "(range shown in brackets); a row is marked inconclusive rather than given a speedup "
            "number if the Y and eager-PyTorch ranges overlap.\n\n"
        )
        if "rmsnorm" in all_results:
            f.write(f"## Fused Add+RMSNorm (hidden_dim={HIDDEN_DIM})\n\n")
            f.write(
                "`h = X + Residual`; `Out = RMSNorm(h) * Weight`; writes both `Out` and `h` "
                "(as `NewResidual`) from one launch, one warp (32 threads) per row. Eager "
                "baseline: real ATen ops (add, `.pow(2).mean(-1)`, `rsqrt`, two multiplies), no "
                "fusion - the way a naive PyTorch RMSNorm module is actually written.\n\n"
            )
            f.write("| Rows | Eager PyTorch us (median [range]) | Y Fused us (median [range]) | Y vs Eager | Correct (max abs diff) |\n")
            f.write("|---|---|---|---|---|\n")
            for r in all_results["rmsnorm"]:
                verdict = "inconclusive" if r["inconclusive"] else f"**{r['vs_eager']:.2f}x**"
                f.write(
                    f"| {r['rows']} | {r['eager_med']:.2f} [{r['eager_min']:.2f}, {r['eager_max']:.2f}] | "
                    f"{r['y_med']:.2f} [{r['y_min']:.2f}, {r['y_max']:.2f}] | {verdict} | "
                    f"{'OK' if r['correct'] else 'FAIL'} ({r['max_abs_diff']:.4f}) |\n"
                )
            f.write("\n")
        if "rope" in all_results:
            f.write(f"## Fused RoPE (head_dim={HEAD_DIM}, interleaved-pairs convention)\n\n")
            f.write(
                "Rotates each token's query/key vector by a position-dependent angle computed "
                "entirely on-device (`ex2.approx.f32`/`sin.approx.f32`/`cos.approx.f32`, no "
                "precomputed cos/sin table input). Eager baseline builds the same table "
                "PyTorch-side every call (the realistic unfused cost) and applies the same "
                "interleaved-pairs rotation.\n\n"
            )
            f.write("| Rows | Eager PyTorch us (median [range]) | Y Fused us (median [range]) | Y vs Eager | Correct (max abs diff) |\n")
            f.write("|---|---|---|---|---|\n")
            for r in all_results["rope"]:
                verdict = "inconclusive" if r["inconclusive"] else f"**{r['vs_eager']:.2f}x**"
                f.write(
                    f"| {r['rows']} | {r['eager_med']:.2f} [{r['eager_min']:.2f}, {r['eager_max']:.2f}] | "
                    f"{r['y_med']:.2f} [{r['y_min']:.2f}, {r['y_max']:.2f}] | {verdict} | "
                    f"{'OK' if r['correct'] else 'FAIL'} ({r['max_abs_diff']:.4f}) |\n"
                )
    print(f"\n[*] Wrote {report_path}")

    all_correct = all(r["correct"] for results in all_results.values() for r in results)
    if not all_correct:
        print("\n[!] One or more configurations FAILED correctness - their timing numbers are not meaningful.")
        sys.exit(1)


if __name__ == "__main__":
    main()
