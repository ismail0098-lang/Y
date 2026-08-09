#!/usr/bin/env python3
"""Y vs OpenBLAS CPU GEMM: one shape, one library, one process.

Usage:
    LD_LIBRARY_PATH=/tmp/ob_avx512 \
    python3 tests/run_cpu_gemm_bench.py --bin tests/benchmark_cpu_gemm_avx512 \
        --threads 16 --launches 3 --isa avx512

Why one process per shape per library
-------------------------------------
Every shortcut cheaper than this was measured and every one of them biased the
result, always in Y's favour:

  1. Both libraries in ONE process.  OpenBLAS's idle threads spin before
     parking, so whichever ran second was timed against a busy machine.
     512^3 scaled 1.95x on 16 threads interleaved and 7.35x standalone.

  2. openblas_set_num_threads() on a USE_OPENMP build.  Driving the count
     through the API instead of OMP_NUM_THREADS cost the baseline 12% at
     1024^3 (1621 vs 1825 GFLOPS).  Thread counts are environment-only now.

  3. A batch size derived from an assumed 100 GFLOPS.  That gave THREE
     iterations at 1024^3 on a machine doing 2000+, so the timed region was
     4 ms rather than the 20 ms the comment claimed.  The C driver calibrates
     from a measured call.

  4. Warming at one thread and timing at sixteen.  The first timed round paid
     for OpenMP team creation.

  5. Ramping the clock on ONE core, leaving fifteen at the 624 MHz idle floor.

  6. Running all 18 shapes in one process.  The later shapes are depressed,
     and Y more than OpenBLAS: 2048^3 measures 2289 GFLOPS as the 4th shape of
     a sweep and 3082 alone, moving Y/OB from 0.70 to 0.88.  This is the one
     that cannot be fixed inside the process, hence this script.

Ranking is by the MINIMUM time across rounds and across launches, because
contamination is one-sided.  The reported spread is what the run can actually
resolve; a ratio inside it is a tie, not a win.
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
]

# Flops per cycle per core, single precision. Zen 5 issues two full-width
# 512-bit FMAs per cycle: 16 lanes * 2 flops * 2 pipes = 64.
FLOPS_PER_CYCLE = {"avx512": 64.0, "avx2": 32.0}

ROW = re.compile(r"^(.*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.e+-]+)\s*$")


def run_one(binary, index, mode, threads, rounds):
    """Return (gflops, jitter, relL2, ghz) for one shape in a fresh process.

    Selected by INDEX, which must stay in step with SHAPES[] in the C driver.
    A substring filter was tried first and quietly reported 2048^3 under the
    "tiny 48^3" label.
    """
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
        sys.exit(f"FAILED {mode} {shape}:\n{out.stdout}\n{out.stderr}")

    ghz = 0.0
    for line in out.stdout.splitlines():
        if "achieved clock" in line:
            ghz = float(line.split(":")[1].strip().split()[0])
        if line.startswith("#") or line.startswith("shape") or not line.strip():
            continue
        m = ROW.match(line)
        if m and m.group(1).strip():
            return float(m.group(3)), float(m.group(4)), float(m.group(5)), ghz
    sys.exit(f"no result row for {mode} {shape}:\n{out.stdout}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./tests/benchmark_cpu_gemm")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--isa", choices=["avx512", "avx2"], default="avx2",
                    help="instruction set of the OpenBLAS build being compared")
    a = ap.parse_args()

    print(f"Y vs OpenBLAS  |  {a.threads} threads  |  {a.launches} launches x "
          f"{a.rounds} rounds  |  one process per shape per library")
    print(f"baseline OpenBLAS ISA: {a.isa}")
    hdr = (f"{'shape':<24}{'Y GF':>9}{'OB GF':>9}{'Y/OB':>8}"
           f"{'Y %pk':>7}{'OB %pk':>8}{'jit':>6}{'relL2':>10}")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    ratios, ghz_seen = [], []
    for index, shape in enumerate(SHAPES):
        ys, obs, jits, rels = [], [], [], []
        for _ in range(a.launches):
            g, j, r, hz = run_one(a.bin, index, "y", a.threads, a.rounds)
            ys.append(g); jits.append(j); rels.append(r); ghz_seen.append(hz)
            g, j, _, hz = run_one(a.bin, index, "ob", a.threads, a.rounds)
            obs.append(g); jits.append(j); ghz_seen.append(hz)
        y, ob = max(ys), max(obs)          # min time == max GFLOPS
        ratios.append(y / ob)
        ghz = statistics.median(h for h in ghz_seen if h > 0) or 5.0
        peak_y = FLOPS_PER_CYCLE["avx512"] * a.threads * ghz
        peak_ob = FLOPS_PER_CYCLE[a.isa] * a.threads * ghz
        print(f"{shape:<24}{y:9.1f}{ob:9.1f}{y/ob:8.2f}"
              f"{100*y/peak_y:6.0f}%{100*ob/peak_ob:7.0f}%"
              f"{max(jits):6.2f}{max(rels):10.1e}")
        sys.stdout.flush()

    print("-" * len(hdr))
    geo = statistics.geometric_mean(ratios)
    print(f"{'geomean Y/OB':<24}{'':9}{'':9}{geo:8.2f}")
    print(f"{'arith mean Y/OB':<24}{'':9}{'':9}{sum(ratios)/len(ratios):8.2f}")
    print(f"{'shapes Y wins':<24}{'':9}{'':9}"
          f"{sum(1 for r in ratios if r > 1.0):8d} / {len(ratios)}")
    print(f"\nall-core clock during load: "
          f"{statistics.median(h for h in ghz_seen if h > 0):.2f} GHz")
    print("%pk is against that clock. jit = worst median/min seen for the "
          "shape; >1.10 means the box was contended and the ratio is soft.")


if __name__ == "__main__":
    main()
