#!/usr/bin/env python3
"""
Follow-up to benchmark_y_vs_flashinfer.py's RoPE result: head_dim=128 showed
a clear win (2.42x at 128 rows in the pre-vectorization measurement, 1.85-
1.92x after) but 1024/8192 rows were inconclusive (overlapping ranges).
Working hypothesis from that doc: FlashInfer pays real per-call overhead for
a much broader contract (ragged-tensor batching, arbitrary rotary_dim,
runtime dtype dispatch) that a kernel compiled for one fixed shape doesn't -
if true, the win should hold up (or grow) across OTHER head_dims, not just
128. This script tests that: same math, same convention, same shared
reference as benchmark_y_vs_flashinfer.py, swept over head_dim in {64, 128,
256} (all common LLM values - 64 is GPT-NeoX/some Llama variants, 128 is
Llama/Mistral/Qwen, 256 is used by some larger-head-dim models) x more
repeat runs than the main script (10, not 5) specifically to try to resolve
the 1024/8192 "inconclusive" calls with a tighter confidence signal.

Usage:
    python3 tests/benchmark_y_rope_headdim_sweep.py
    python3 tests/benchmark_y_rope_headdim_sweep.py --once   (internal, one worker sample)
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
HEAD_DIMS = [64, 128, 256]
REPEAT_RUNS = 10
ITERS = 500


def build():
    res = subprocess.run("cargo build --release", shell=True, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Y compiler build failed:\n{res.stderr}")
        sys.exit(1)


def compile_once(ysu_name):
    res = subprocess.run([Y_BIN, f"tests/{ysu_name}.ysu", "--emit-ptx"], capture_output=True, text=True, cwd=REPO_ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"Y compile failed for {ysu_name}:\n{res.stdout}\n{res.stderr}")


def run_once(rows, head_dim):
    import torch
    import cupy as cp
    import flashinfer

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    X = torch.randn(rows, head_dim, dtype=torch.float16, device=device)
    Positions = torch.randint(0, 4096, (rows,), dtype=torch.int32, device=device)
    Out = torch.zeros(rows, head_dim, dtype=torch.float16, device=device)

    X_cp, Pos_cp, Out_cp = cp.asarray(X), cp.asarray(Positions), cp.asarray(Out)

    kernel_name = f"rope_{head_dim}"
    mod = cp.RawModule(path=os.path.join(REPO_ROOT, f"tests/{kernel_name}.ptx"))
    fn = mod.get_function(kernel_name)

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

    # ---- correctness: fresh eager reference (interleaved-pairs, base=10000) ----
    i_idx = torch.arange(head_dim // 2, dtype=torch.float32, device=device)
    inv_freq = 10000.0 ** (-2.0 * i_idx / head_dim)
    pos_f = Positions.float().unsqueeze(1)
    theta = pos_f * inv_freq.unsqueeze(0)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    x_f = X.float().view(rows, head_dim // 2, 2)
    x0, x1 = x_f[..., 0], x_f[..., 1]
    out0 = x0 * cos_t - x1 * sin_t
    out1 = x0 * sin_t + x1 * cos_t
    out_ref = torch.stack([out0, out1], dim=-1).reshape(rows, head_dim).to(torch.float16)
    correct = bool(torch.allclose(Out, out_ref, rtol=0.02, atol=0.02))
    max_abs_diff = (Out.float() - out_ref.float()).abs().max().item()

    # ---- FlashInfer: real production fused kernel ----
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
        "rows": rows, "head_dim": head_dim, "y_us": y_us, "fi_us": fi_us,
        "correct": correct, "fi_correct": fi_correct, "max_abs_diff": max_abs_diff,
    }))


def median_range(values):
    return statistics.median(values), min(values), max(values)


def ranges_overlap(a_min, a_max, b_min, b_max):
    return a_min <= b_max and b_min <= a_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.once:
        rows = int(os.environ["Y_BENCH_ROWS"])
        head_dim = int(os.environ["Y_BENCH_HEAD_DIM"])
        run_once(rows, head_dim)
        return

    print("[*] Building Y compiler release binary...")
    build()
    for hd in HEAD_DIMS:
        compile_once(f"rope_{hd}")

    import flashinfer
    print(f"[*] flashinfer version: {flashinfer.__version__}")

    all_results = {}
    for head_dim in HEAD_DIMS:
        print("\n" + "=" * 100)
        print(f"REAL Y FUSED ROPE (head_dim={head_dim}) vs FLASHINFER - {REPEAT_RUNS} repeats/row".center(100))
        print("=" * 100)
        header = f"{'Rows':<10} | {'FlashInfer us (median)':<24} | {'Y Fused us (median)':<22} | {'Y vs FlashInfer':<16} | Correct"
        print("\n" + header)
        print("-" * len(header))

        results = []
        for rows in ROW_COUNTS:
            y_samples, fi_samples, correct_flags, fi_correct_flags, max_diffs = [], [], [], [], []
            for _ in range(REPEAT_RUNS):
                env = dict(os.environ)
                env["Y_BENCH_ROWS"] = str(rows)
                env["Y_BENCH_HEAD_DIM"] = str(head_dim)
                proc = subprocess.run(
                    [sys.executable, __file__, "--once"],
                    capture_output=True, text=True, env=env, cwd=REPO_ROOT,
                )
                if proc.returncode != 0:
                    print(f"[!] worker run failed for rows={rows} head_dim={head_dim}:\n{proc.stdout}\n{proc.stderr}")
                    sys.exit(1)
                line = next((l for l in proc.stdout.splitlines() if l.startswith("{")), None)
                if line is None:
                    print(f"[!] no JSON output for rows={rows} head_dim={head_dim}:\n{proc.stdout}\n{proc.stderr}")
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
            print(
                f"{rows:<10} | {f'{f_med:.2f} [{f_min:.2f},{f_max:.2f}]':<24} | "
                f"{f'{y_med:.2f} [{y_min:.2f},{y_max:.2f}]':<22} | {verdict:<16} | {tag}"
            )
        all_results[head_dim] = results

    report_path = os.path.join(REPO_ROOT, "benchmark_y_rope_headdim_sweep_results.md")
    with open(report_path, "w") as f:
        f.write("# RoPE head_dim sweep: Y vs FlashInfer, more repeats to resolve prior inconclusive calls\n\n")
        f.write(
            f"Follow-up to `benchmark_y_vs_flashinfer_results.md`'s RoPE section - {REPEAT_RUNS} repeats/row "
            "(not 5) across head_dim in {64, 128, 256}, testing whether the head_dim=128 win generalizes or "
            "was specific to that shape. Same math, same convention, same reference formula as the main script.\n\n"
        )
        for head_dim in HEAD_DIMS:
            f.write(f"## head_dim={head_dim}\n\n")
            f.write("| Rows | FlashInfer us (median [range]) | Y Fused us (median [range]) | Y vs FlashInfer | Correct |\n")
            f.write("|---|---|---|---|---|\n")
            for r in all_results[head_dim]:
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
        print("\n[!] FlashInfer itself did not match the shared reference at some size - check convention before trusting any ratio.")
    if not all_correct:
        print("\n[!] Y FAILED correctness at some size - its timing numbers are not meaningful.")
        sys.exit(1)


if __name__ == "__main__":
    main()
