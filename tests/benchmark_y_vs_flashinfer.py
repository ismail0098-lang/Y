#!/usr/bin/env python3
"""
Benchmarks REAL Y-compiler-emitted fused Add+RMSNorm and RoPE PTX against
FlashInfer's real, production fused kernels (`flashinfer.fused_add_rmsnorm`,
`flashinfer.apply_rope_pos_ids`) - the actual competitive bar (FlashInfer's
kernels are used by vLLM, SGLang, and others in production), not eager
PyTorch. See benchmark_y_rmsnorm_rope_results.md for the eager-PyTorch
comparison; this script exists because that comparison alone overstates
what "beating eager PyTorch" means for a real deployment decision.

Semantics match exactly, not just approximately:
  - `flashinfer.fused_add_rmsnorm(input, residual, weight, eps)`: in-place
    `residual += input; input = (residual / rms(residual)) * weight` - the
    same `h = X + Residual; Out = RMSNorm(h) * Weight` (writing back both
    `Out` and updated `h`) `emit_rmsnorm_residual_kernel` computes.
  - `flashinfer.apply_rope_pos_ids(q, k, pos_ids, interleave=True,
    rope_theta=10000.0)`: interleaved-pairs RoPE, on-the-fly cos/sin - the
    same convention and base `emit_rope_kernel` uses (see that function's
    doc comment).

Timing discipline: median of REPEAT_RUNS independent process re-launches,
matching every other benchmark_y_*.py script in this directory.

Usage:
    python3 tests/benchmark_y_vs_flashinfer.py
    python3 tests/benchmark_y_vs_flashinfer.py --kernel rmsnorm
    python3 tests/benchmark_y_vs_flashinfer.py --kernel rope
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
    import flashinfer

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

    # 256 threads/row (8 warps), not 32: emit_rmsnorm_residual_kernel picks
    # num_warps = largest of {8,4,2,1} dividing (hidden_dim/8)/32 evenly - 8
    # at HIDDEN_DIM=4096 (512 vector-groups, 512%256==0). Must match the
    # kernel's own real thread count or it silently only executes on a
    # fraction of its threads (wrong, not a crash) - see
    # emit_rmsnorm_residual_kernel's doc comment.
    grid, threads = (rows, 1), (256, 1, 1)

    # ---- Y: warmup + timed average ----
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

    # ---- correctness: fresh eager reference (same reference the other doc uses) ----
    h_ref = X + Residual
    variance = h_ref.float().pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + 1e-5)
    out_ref = (h_ref.float() * inv_rms).to(torch.float16) * Weight
    correct = bool(
        torch.allclose(Out, out_ref, rtol=0.02, atol=0.02)
        and torch.allclose(NewResidual, h_ref, rtol=0.01, atol=0.01)
    )
    max_abs_diff = max(
        (Out.float() - out_ref.float()).abs().max().item(),
        (NewResidual.float() - h_ref.float()).abs().max().item(),
    )

    # ---- FlashInfer: real production fused kernel, same math, in-place ----
    fi_input = X.clone()
    fi_residual = Residual.clone()
    for _ in range(10):
        inp, res = fi_input.clone(), fi_residual.clone()
        flashinfer.fused_add_rmsnorm(inp, res, Weight, eps=1e-5)
    torch.cuda.synchronize()

    # Correctness of FlashInfer itself against the same reference (sanity check
    # that the reference/convention match, not just that Y matches something).
    fi_check_in, fi_check_res = X.clone(), Residual.clone()
    flashinfer.fused_add_rmsnorm(fi_check_in, fi_check_res, Weight, eps=1e-5)
    fi_correct = bool(
        torch.allclose(fi_check_in, out_ref, rtol=0.02, atol=0.02)
        and torch.allclose(fi_check_res, h_ref, rtol=0.01, atol=0.01)
    )

    fi_input = X.clone()
    fi_residual = Residual.clone()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(ITERS):
        flashinfer.fused_add_rmsnorm(fi_input, fi_residual, Weight, eps=1e-5)
    end_evt.record()
    torch.cuda.synchronize()
    fi_us = (start_evt.elapsed_time(end_evt) / ITERS) * 1000.0

    print(json.dumps({
        "rows": rows, "y_us": y_us, "fi_us": fi_us, "correct": correct,
        "fi_correct": fi_correct, "max_abs_diff": max_abs_diff,
    }))


def run_once_rope(rows):
    import torch
    import cupy as cp
    import flashinfer
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

    # ---- correctness: fresh eager reference (interleaved-pairs, base=10000 -
    # same convention emit_rope_kernel and flashinfer's interleave=True use) ----
    i_idx = torch.arange(HEAD_DIM // 2, dtype=torch.float32, device=device)
    inv_freq = 10000.0 ** (-2.0 * i_idx / HEAD_DIM)
    pos_f = Positions.float().unsqueeze(1)
    theta = pos_f * inv_freq.unsqueeze(0)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    x_f = X.float().view(rows, HEAD_DIM // 2, 2)
    x0, x1 = x_f[..., 0], x_f[..., 1]
    out0 = x0 * cos_t - x1 * sin_t
    out1 = x0 * sin_t + x1 * cos_t
    out_ref = torch.stack([out0, out1], dim=-1).reshape(rows, HEAD_DIM).to(torch.float16)
    correct = bool(torch.allclose(Out, out_ref, rtol=0.02, atol=0.02))
    max_abs_diff = (Out.float() - out_ref.float()).abs().max().item()

    # ---- FlashInfer: real production fused kernel (Q and K share the API;
    # feed X as both, only inspect the Q output) ----
    fi_q, fi_k = X.clone(), X.clone()
    for _ in range(10):
        _ = flashinfer.apply_rope_pos_ids(
            fi_q.unsqueeze(1), fi_k.unsqueeze(1), Positions.long(),
            interleave=True, rope_theta=10000.0,
        )
    torch.cuda.synchronize()

    fi_q_out, _ = flashinfer.apply_rope_pos_ids(
        X.clone().unsqueeze(1), X.clone().unsqueeze(1), Positions.long(),
        interleave=True, rope_theta=10000.0,
    )
    fi_correct = bool(torch.allclose(fi_q_out.squeeze(1), out_ref, rtol=0.02, atol=0.02))

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(ITERS):
        _ = flashinfer.apply_rope_pos_ids(
            fi_q.unsqueeze(1), fi_k.unsqueeze(1), Positions.long(),
            interleave=True, rope_theta=10000.0,
        )
    end_evt.record()
    torch.cuda.synchronize()
    fi_us = (start_evt.elapsed_time(end_evt) / ITERS) * 1000.0

    print(json.dumps({
        "rows": rows, "y_us": y_us, "fi_us": fi_us, "correct": correct,
        "fi_correct": fi_correct, "max_abs_diff": max_abs_diff,
    }))


def median_range(values):
    return statistics.median(values), min(values), max(values)


def ranges_overlap(a_min, a_max, b_min, b_max):
    return a_min <= b_max and b_min <= a_max


def run_suite(kernel_name, ysu_name, title):
    print("\n" + "=" * 100)
    print(title.center(100))
    print("=" * 100)
    compile_once(ysu_name)

    header = f"{'Rows':<10} | {'FlashInfer us (median)':<24} | {'Y Fused us (median)':<22} | {'Y vs FlashInfer':<16} | Correct"
    print("\n" + header)
    print("-" * len(header))

    results = []
    for rows in ROW_COUNTS:
        y_samples, fi_samples, correct_flags, fi_correct_flags, max_diffs = [], [], [], [], []
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
            fi_samples.append(data["fi_us"])
            correct_flags.append(data["correct"])
            fi_correct_flags.append(data["fi_correct"])
            max_diffs.append(data["max_abs_diff"])

        y_med, y_min, y_max = median_range(y_samples)
        f_med, f_min, f_max = median_range(fi_samples)
        inconclusive = ranges_overlap(y_min, y_max, f_min, f_max)
        all_correct = all(correct_flags)
        all_fi_correct = all(fi_correct_flags)
        vs_fi = f_med / y_med
        verdict = "inconclusive" if inconclusive else f"{vs_fi:.2f}x"

        results.append({
            "rows": rows, "y_med": y_med, "y_min": y_min, "y_max": y_max,
            "fi_med": f_med, "fi_min": f_min, "fi_max": f_max,
            "vs_fi": vs_fi, "inconclusive": inconclusive,
            "correct": all_correct, "fi_correct": all_fi_correct, "max_abs_diff": max(max_diffs),
        })
        tag = "OK" if all_correct else "FAIL"
        fi_tag = "" if all_fi_correct else " [WARNING: FlashInfer itself didn't match the reference - convention mismatch, not a Y bug]"
        print(
            f"{rows:<10} | {f'{f_med:.2f} [{f_min:.2f},{f_max:.2f}]':<24} | "
            f"{f'{y_med:.2f} [{y_min:.2f},{y_max:.2f}]':<22} | {verdict:<16} | {tag}{fi_tag}"
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

    import flashinfer
    print(f"[*] flashinfer version: {flashinfer.__version__}")

    all_results = {}
    if args.kernel in ("rmsnorm", "both"):
        all_results["rmsnorm"] = run_suite(
            "rmsnorm", "rmsnorm_residual_4096",
            f"REAL Y FUSED ADD+RMSNORM (hidden_dim={HIDDEN_DIM}) vs FLASHINFER (real production kernel)",
        )
    if args.kernel in ("rope", "both"):
        all_results["rope"] = run_suite(
            "rope", "rope_128",
            f"REAL Y FUSED ROPE (head_dim={HEAD_DIM}) vs FLASHINFER (real production kernel)",
        )

    report_path = os.path.join(REPO_ROOT, "benchmark_y_vs_flashinfer_results.md")
    with open(report_path, "w") as f:
        f.write("# Real Y-Compiler Fused Add+RMSNorm and RoPE vs FlashInfer (real production kernels)\n\n")
        f.write(
            "This is the comparison `benchmark_y_rmsnorm_rope_results.md` (Y vs eager PyTorch) does NOT "
            "answer: eager PyTorch is the weakest possible baseline, and production inference stacks "
            "(vLLM, SGLang) already use FlashInfer's hand-tuned fused kernels for exactly these ops, not "
            "eager PyTorch. `flashinfer.fused_add_rmsnorm`/`flashinfer.apply_rope_pos_ids(interleave=True)` "
            "compute the same math, same convention (interleaved-pairs RoPE, base=10000, in-place "
            "add+RMSNorm) as Y's kernels - not an approximate comparison.\n\n"
        )
        f.write(
            f"Timing is the median of {REPEAT_RUNS} independent process launches per row count "
            "(range shown in brackets); a row is marked inconclusive rather than given a ratio if the "
            "Y and FlashInfer ranges overlap.\n\n"
        )
        for key, title in [("rmsnorm", f"Fused Add+RMSNorm (hidden_dim={HIDDEN_DIM})"), ("rope", f"Fused RoPE (head_dim={HEAD_DIM}, interleaved)")]:
            if key not in all_results:
                continue
            f.write(f"## {title}\n\n")
            f.write("| Rows | FlashInfer us (median [range]) | Y Fused us (median [range]) | Y vs FlashInfer | Correct |\n")
            f.write("|---|---|---|---|---|\n")
            for r in all_results[key]:
                verdict = "inconclusive" if r["inconclusive"] else f"**{r['vs_fi']:.2f}x**"
                f.write(
                    f"| {r['rows']} | {r['fi_med']:.2f} [{r['fi_min']:.2f}, {r['fi_max']:.2f}] | "
                    f"{r['y_med']:.2f} [{r['y_min']:.2f}, {r['y_max']:.2f}] | {verdict} | "
                    f"{'OK' if r['correct'] else 'FAIL'} |\n"
                )
            f.write("\n")
    print(f"\n[*] Wrote {report_path}")

    all_correct = all(r["correct"] for results in all_results.values() for r in results)
    all_fi_correct = all(r["fi_correct"] for results in all_results.values() for r in results)
    if not all_fi_correct:
        print("\n[!] FlashInfer itself did not match the shared reference at some size - check convention (interleave/base/eps) before trusting any ratio.")
    if not all_correct:
        print("\n[!] Y FAILED correctness at some size - its timing numbers are not meaningful.")
        sys.exit(1)


if __name__ == "__main__":
    main()
