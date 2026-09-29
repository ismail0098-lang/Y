#!/usr/bin/env python3
"""Strict A/B of two or more CPU-GEMM benchmark binaries, in one session.

    LD_LIBRARY_PATH=/tmp/ob_avx512 python3 tests/ab_cpu_gemm.py \
        --arm before=/tmp/verif/bench_before --arm after=/tmp/verif/bench_after \
        --threads 16 --launches 3 --shapes 2,3,5 --ob

`tests/run_cpu_gemm_bench.py` is the single-binary Y-vs-OpenBLAS sweep; this is
its A/B sibling, for deciding whether a source change moved anything.  It shares
that file's isolation rules (one shape, one library, one process) and adds the
three that A/B specifically needs:

  - **Arms are interleaved WITHIN a launch and their order ROTATES between
    launches.**  Whichever arm runs second is systematically penalised on this
    box: a 3-6% "regression" on the square shapes reproduced 3/3 launches and
    then reversed sign when the order was flipped.  Rotation (not just reversal)
    matters once there are three arms, so that each arm takes each position.

  - **OpenBLAS is measured in the same session as the arms**, when `--ob` is
    given.  Two sweeps taken minutes apart had OpenBLAS's own column move 35%,
    which silently corrupts any Y/OB geomean carried between them.

  - **The within-arm spread is printed next to every ratio.**  Several shapes
    swing 10-50% between launches at 16 threads.  A ratio inside the spread is
    a tie and must be reported as one.

Ranking is by the best (== minimum time) over launches, because contamination
is one-sided: nothing makes a run spuriously fast.
"""

import argparse
import os
import re
import statistics
import subprocess
import sys

SHAPES = [
    "nice    256^3", "nice    512^3", "nice   1024^3", "nice   2048^3",
    "ragged  250^3", "ragged 1000^3", "ragged 1021^3 prime",
    "ragged 137x391x1013", "ragged 333x777x64", "flatK  4096x4096x8",
    "deepK  64x64x32768", "skinny 33x4096x4096", "skinny 17x4096x4096",
    "decode  8x4096x4096", "decode  4x4096x4096", "gemv    1x4096x4096",
    "gemv    1x8192x8192", "tiny    48^3",
    # Appended, so 0-17 keep the meaning the doc quotes. These four sit in the
    # 48^3 -> 250^3 gap, where the copy-free/packed crossover lives.
    "small   64^3", "small   128x64x128", "small   256x64x256", "small   128^3",
    # One per `nv` bucket of the copy-free path (nv = 1, 2, 4; 48^3 is nv=3).
    "small   64x16x64", "small   48x32x64", "small   32x64x48",
]

ROW = re.compile(r"^(.*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.e+-]+)\s*$")


def run_one(binary, index, mode, threads, rounds):
    """Return (gflops, relL2) for one shape, one library, one fresh process."""
    env = dict(os.environ)
    env["SHAPE_INDEX"] = str(index)
    env.pop("Y_NUM_THREADS", None)
    env.pop("OMP_WAIT_POLICY", None)
    if mode == "y":
        env["Y_NUM_THREADS"] = str(threads)
        env["OMP_NUM_THREADS"] = "1"          # reference GEMM only
        env["OMP_WAIT_POLICY"] = "passive"    # and it must not spin afterwards
    else:
        env["OMP_NUM_THREADS"] = str(threads)

    out = subprocess.run([binary, str(threads), str(rounds), mode],
                         env=env, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"FAILED {binary} mode={mode} shape={index}:\n"
                 f"{out.stdout}\n{out.stderr}")
    for line in out.stdout.splitlines():
        if line.startswith("#") or line.startswith("shape") or not line.strip():
            continue
        m = ROW.match(line)
        if m and m.group(1).strip():
            return float(m.group(3)), float(m.group(5))
    sys.exit(f"no result row for {binary} mode={mode} shape={index}:\n{out.stdout}")


def spread(values):
    """Within-arm dispersion, (max-min)/max.  0 when a single launch was run."""
    return (max(values) - min(values)) / max(values) if max(values) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable; the first arm is the A/B denominator")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--shapes", default="",
                    help="comma-separated shape indices; default all 18")
    ap.add_argument("--ob", action="store_true",
                    help="also measure OpenBLAS, in this same session")
    a = ap.parse_args()

    arms = []
    for spec in a.arm:
        if "=" not in spec:
            sys.exit(f"--arm wants NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            sys.exit(f"no such binary: {path}")
        arms.append((name, path))
    if len(arms) < 2 and not a.ob:
        sys.exit("need at least two arms, or one arm and --ob")
    base = arms[0][0]

    indices = ([int(x) for x in a.shapes.split(",")] if a.shapes
               else list(range(len(SHAPES))))

    print(f"strict A/B | {a.threads} threads | {a.launches} launches x "
          f"{a.rounds} rounds | interleaved, order rotated, ranked by best")
    print("arms: " + ", ".join(f"{n}={p}" for n, p in arms))
    # Column widths come from the labels, so a long arm name cannot run its
    # header into the next column and silently mislabel a number.
    cols = [(n, max(10, len(n) + 2)) for n, _ in arms]
    rcols = [(f"{n}/{base}", max(11, len(n) + len(base) + 3)) for n, _ in arms[1:]]
    ocols = [(f"{n}/OB", max(10, len(n) + 5)) for n, _ in arms]
    hdr = f"{'shape':<22}" + "".join(f"{t:>{w}}" for t, w in cols)
    hdr += "".join(f"{t:>{w}}" for t, w in rcols)
    hdr += f"{'spread':>9}"
    if a.ob:
        hdr += f"{'OB':>10}" + "".join(f"{t:>{w}}" for t, w in ocols)
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    ratios = {n: [] for n, _ in arms[1:]}
    vs_ob = {n: [] for n, _ in arms}
    for index in indices:
        name = SHAPES[index]
        samples = {n: [] for n, _ in arms}
        obs = []
        for launch in range(a.launches):
            # Rotate, so over >= len(arms) launches each arm holds each slot.
            order = arms[launch % len(arms):] + arms[:launch % len(arms)]
            for n, path in order:
                gf, rel = run_one(path, index, "y", a.threads, a.rounds)
                samples[n].append(gf)
                if rel >= 1e-5:
                    print(f"  *** {name} arm {n}: relL2={rel:.1e} MISMATCH")
            if a.ob:
                # Same session, same interleave; OpenBLAS is measured through
                # whichever binary ran last, they all link the same libopenblas.
                obs.append(run_one(order[-1][1], index, "ob",
                                   a.threads, a.rounds)[0])

        best = {n: max(v) for n, v in samples.items()}
        sp = max(spread(v) for v in samples.values())
        line = f"{name:<22}" + "".join(
            f"{best[n]:>{w}.1f}" for (n, _), (_, w) in zip(arms, cols)
        )
        for (n, _), (_, w) in zip(arms[1:], rcols):
            r = best[n] / best[base]
            ratios[n].append(r)
            line += f"{r:>{w}.3f}"
        line += f"{sp * 100:>8.1f}%"
        if a.ob:
            ob = max(obs)
            line += f"{ob:>10.1f}"
            for (n, _), (_, w) in zip(arms, ocols):
                vs_ob[n].append(best[n] / ob)
                line += f"{best[n] / ob:>{w}.2f}"
        print(line, flush=True)

    print("-" * len(hdr))
    for n, rs in ratios.items():
        print(f"geomean {n}/{base}: {statistics.geometric_mean(rs):.3f}"
              f"   (min {min(rs):.3f}, max {max(rs):.3f})")
    if a.ob:
        for n, rs in vs_ob.items():
            print(f"geomean {n}/OB: {statistics.geometric_mean(rs):.3f}"
                  f"   wins {sum(1 for r in rs if r > 1.0)}/{len(rs)}")
    print("\nA ratio inside the printed spread is a TIE, not a result. "
          "0.9-1.1 at 16 threads is routinely noise on this box.")


if __name__ == "__main__":
    main()
