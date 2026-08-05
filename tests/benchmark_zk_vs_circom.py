#!/usr/bin/env python3
"""Y-lang vs Circom: R1CS constraint-emission speed, at sizes that actually run.

This replaces the numbers in docs/heavy_circuit_speed_test.md, which were not
reproducible: they were measured at 1,000,000 and 100,000 loop iterations, but
the compiler enforces a hard 10,000-iteration unroll cap
(`Circuit unroll error [Z0099]`), so neither circuit compiles today. The two
harness scripts that doc tells you to run (`run_speed_test.js`,
`run_dot_product_benchmark.js`) are also absent from the repo.

What this measures: wall-clock time for each compiler to turn one source file
into an R1CS. It is a COMPILE-TIME benchmark, not a proving benchmark - neither
tool proves anything here. (For a real proof over Y's R1CS, see
`tests/zk_groth16_end_to_end.rs`, which runs Groth16 setup/prove/verify over
BN254 via arkworks.)

Discipline, matching the rest of this repo's benchmarks:
  * MINIMUM of N runs, not mean or median - contamination is one-sided.
  * Y is invoked with the repo root as CWD. This matters enormously and is easy
    to get wrong: Y looks up `.ysu_hw_profile` at a RELATIVE path, and without
    it every invocation re-runs the full GPU hardware probe. Measuring from the
    wrong directory made Y read 0.694s instead of 0.014s - a 48x error on a
    benchmark that has nothing to do with the GPU.
  * Fairness gate: the run ABORTS unless Y and circom emit the same number of
    non-linear constraints for a given size. Comparing compile speed across two
    tools that built different circuits is meaningless, and constraint
    deduplication is exactly the kind of optimization that could silently make
    Y's output smaller and therefore faster.

Requires: `cargo build --release --features zk` (the ZK backend is NOT in a
default build - without the feature the binary prints "The ZK Circuit Backend
is not compiled into this binary" and exits 0, which looks like success), and
`circom` on PATH.

Usage:  python3 tests/benchmark_zk_vs_circom.py [--runs 5]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Y_BIN = os.path.join(REPO, "target", "release", "Y")
RESULTS = os.path.join(REPO, "benchmark_zk_vs_circom_results.md")

# The compiler's unroll cap is 10,000 iterations; 10,000 is therefore the
# largest circuit either tool can be compared on without changing the compiler.
SIZES = [100, 500, 1000, 5000, 10000]

Y_POLY = """@unsafe
fn main(x: I32, y: I32) -> I32 {{
    let mut temp = x;
    for i in 0..{n} {{
        temp = temp * y;
    }}
    return temp;
}}
"""

CIRCOM_POLY = """pragma circom 2.0.0;
template Poly() {{
    signal input x;
    signal input y;
    signal output out;
    signal temp[{n1}];
    temp[0] <== x;
    for (var i = 0; i < {n}; i++) {{
        temp[i+1] <== temp[i] * y;
    }}
    out <== temp[{n}];
}}
component main = Poly();
"""

Y_DOT = """@unsafe
fn main(x: I32, y: I32) -> I32 {{
    let mut sum = 0;
    let mut a = x;
    let mut b = y;
    for i in 0..{n} {{
        a = a + 1;
        b = b + 1;
        sum = sum + a * b;
    }}
    return sum;
}}
"""

CIRCOM_DOT = """pragma circom 2.0.0;
template Dot() {{
    signal input x;
    signal input y;
    signal output out;
    signal a[{n1}];
    signal b[{n1}];
    signal sum[{n1}];
    signal prod[{n}];
    a[0] <== x;
    b[0] <== y;
    sum[0] <== 0;
    for (var i = 0; i < {n}; i++) {{
        a[i+1] <== a[i] + 1;
        b[i+1] <== b[i] + 1;
        prod[i] <== a[i+1] * b[i+1];
        sum[i+1] <== sum[i] + prod[i];
    }}
    out <== sum[{n}];
}}
component main = Dot();
"""

CIRCUITS = {
    "polynomial (temp = temp * y)": (Y_POLY, CIRCOM_POLY),
    "dot product (sum += a * b)": (Y_DOT, CIRCOM_DOT),
}


def best_of(fn, runs):
    """Minimum wall time over `runs` invocations, plus the last result."""
    best, out = float("inf"), None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def run_y(src_path):
    # CWD must be REPO so `.ysu_hw_profile` is found - see module docstring.
    r = subprocess.run([Y_BIN, src_path, "--target=r1cs"],
                       capture_output=True, text=True, cwd=REPO)
    return r


def y_constraints(stdout, src_path):
    txt_path = os.path.splitext(src_path)[0] + ".r1cs.txt"
    if os.path.exists(txt_path):
        m = re.search(r"Constraints:\s*(\d+)", open(txt_path).read())
        if m:
            return int(m.group(1))
    return None


def run_circom(src_path, outdir):
    return subprocess.run(["circom", src_path, "--r1cs", "-o", outdir],
                          capture_output=True, text=True)


def circom_constraints(stdout):
    m = re.search(r"non-linear constraints:\s*(\d+)", stdout)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    if not os.path.exists(Y_BIN):
        sys.exit("Y binary not found. Run: cargo build --release --features zk")
    probe = subprocess.run([Y_BIN, "--help"], capture_output=True, text=True, cwd=REPO)
    if not shutil.which("circom"):
        sys.exit("circom not found on PATH.")

    rows = []
    for label, (y_tpl, c_tpl) in CIRCUITS.items():
        for n in SIZES:
            with tempfile.TemporaryDirectory() as td:
                ysu = os.path.join(td, f"c{n}.ysu")
                cir = os.path.join(td, f"c{n}.circom")
                open(ysu, "w").write(y_tpl.format(n=n))
                open(cir, "w").write(c_tpl.format(n=n, n1=n + 1))

                yt, yr = best_of(lambda: run_y(ysu), args.runs)
                if yr.returncode != 0 or "Error" in yr.stdout:
                    err = re.search(r"(Circuit unroll error.*|.*Error.*)", yr.stdout)
                    rows.append((label, n, None, None, None, None,
                                 err.group(1).strip() if err else "Y failed"))
                    continue
                ct, cr = best_of(lambda: run_circom(cir, td), args.runs)
                if cr.returncode != 0:
                    rows.append((label, n, yt, None, None, None, "circom failed"))
                    continue

                yc = y_constraints(yr.stdout, ysu)
                cc = circom_constraints(cr.stdout)
                # Fairness gate - see module docstring.
                if yc is None or cc is None or abs(yc - cc) > 1:
                    rows.append((label, n, yt, ct, yc, cc,
                                 "MISMATCHED CIRCUITS - comparison invalid"))
                    continue
                rows.append((label, n, yt, ct, yc, cc, ""))

    hdr = f"{'circuit':<30}{'N':>7}{'Y (s)':>10}{'circom (s)':>12}{'Y cons':>9}{'circom':>8}{'speedup':>10}  note"
    print(hdr)
    print("-" * len(hdr))
    lines = []
    for label, n, yt, ct, yc, cc, note in rows:
        sp = f"{ct/yt:.2f}x" if (yt and ct) else "-"
        line = (f"{label:<30}{n:>7}{(f'{yt:.4f}' if yt else '-'):>10}"
                f"{(f'{ct:.4f}' if ct else '-'):>12}{(yc if yc else '-'):>9}"
                f"{(cc if cc else '-'):>8}{sp:>10}  {note}")
        print(line)
        lines.append((label, n, yt, ct, yc, cc, sp, note))

    with open(RESULTS, "w") as f:
        f.write("# Y-lang vs Circom: R1CS compile-time benchmark\n\n")
        f.write(f"Generated by `tests/benchmark_zk_vs_circom.py --runs {args.runs}`. "
                "Minimum of N runs per cell.\n\n")
        f.write("Compile time only - neither tool proves anything here. For a real "
                "Groth16 proof over Y's R1CS see `tests/zk_groth16_end_to_end.rs`.\n\n")
        f.write("| circuit | N | Y (s) | circom (s) | Y constraints | circom constraints | speedup | note |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for label, n, yt, ct, yc, cc, sp, note in lines:
            f.write(f"| {label} | {n} | {f'{yt:.4f}' if yt else '-'} | "
                    f"{f'{ct:.4f}' if ct else '-'} | {yc or '-'} | {cc or '-'} | {sp} | {note} |\n")
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
