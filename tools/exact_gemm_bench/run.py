#!/usr/bin/env python3
"""Three-arm GEMM benchmark for the verified exact `vpdpwssd` kernel.

    python3 tools/exact_gemm_bench/run.py            # the README's table
    python3 tools/exact_gemm_bench/run.py --scaling  # thread scaling
    python3 tools/exact_gemm_bench/run.py --isa      # the two ISA ceilings

Three arms, because two of them answer different questions and the README
needs both:

  * **Y exact**   - the emitted `vpdpwssd` kernel the Rocq proofs cover.
  * **Y f32**     - the SAME nest in f32 through the SAME emitter, so "what
                    does exactness cost" is not confounded with "what does
                    Y's GEMM quality cost".
  * **OpenBLAS**  - the industry baseline. Taken from the `scipy-openblas`
                    numpy bundles, which is `DYNAMIC_ARCH` and dispatches to
                    its `SkylakeX` AVX-512 kernel here. The harness prints the
                    kernel it selected, because the repo's own tuning document
                    records a distro OpenBLAS with **zero** `zmm` instructions
                    being used as a baseline for a while.

Discipline, all of it learned the hard way in `docs/cpu_gemm_tuning.md`:

  * ONE shape per process per arm. Both libraries in one process measured
    512^3 16T scaling at 1.95x against 7.35x standalone, because OpenBLAS's
    idle threads spin before parking.
  * Arms interleaved and each run twice, minimum taken, so a clock ramp or a
    background process cannot land on one arm only.
  * Absolute numbers on a desktop move between sessions. Read the RATIO
    columns; the run-to-run spread on this box is ~7% on 16-thread GEMM.
"""

import argparse, os, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

SHAPES = [(512, 512, 512, 20, 4), (1024, 1024, 1024, 8, 4),
          (2048, 2048, 2048, 3, 3), (4096, 4096, 4096, 1, 3)]


def find_openblas():
    """The scipy-openblas numpy ships, or $Y_OPENBLAS_SO."""
    if os.environ.get("Y_OPENBLAS_SO"):
        return Path(os.environ["Y_OPENBLAS_SO"]), "scipy_"
    try:
        import numpy
    except ImportError:
        return None, None
    libs = Path(numpy.__file__).parent.parent / "numpy.libs"
    for so in sorted(libs.glob("libscipy_openblas*.so")):
        return so, "scipy_"
    return None, None


def sym_defines(so, prefix):
    """scipy's build renames and suffixes every symbol; a distro one does not."""
    out = subprocess.run(["nm", "-D", "--defined-only", str(so)],
                         capture_output=True, text=True).stdout
    for gemm, thr, core in ((f"{prefix}cblas_sgemm64_", f"{prefix}openblas_set_num_threads64_",
                             f"{prefix}openblas_get_corename64_"),
                            ("cblas_sgemm", "openblas_set_num_threads",
                             "openblas_get_corename")):
        if all(f" {s}" in out for s in (gemm, thr, core)):
            return [f"-DCBLAS_SGEMM={gemm}", f"-DCBLAS_SET_THREADS={thr}",
                    f"-DCBLAS_CORENAME={core}"]
    return None


def build(workdir, arms):
    ybin = REPO / "target" / "release" / "Y"
    if not ybin.exists():
        sys.exit("build the compiler first: cargo build --release")
    exes = {}
    for name, src, entry, flag in (("exact", "exact.ysu", "y_matmul", "-DARM_EXACT=1"),
                                   ("yf32", "f32.ysu", "f32_matmul", "-DARM_YF32=1")):
        if name not in arms:
            continue
        ysu = workdir / f"{name}.ysu"
        shutil.copy(HERE / src, ysu)
        r = subprocess.run([str(ybin), str(ysu), "--emit-llvm"],
                           capture_output=True, text=True, cwd=REPO)
        if r.returncode != 0:
            sys.exit(f"{src} did not compile:\n{r.stdout}{r.stderr}")
        if name == "exact" and "EXACT vpdpwssd kernel substituted" not in r.stdout:
            sys.exit("the exact kernel was NOT substituted - nothing to measure")
        exe = workdir / f"run_{name}"
        subprocess.run(["clang", "-O2", flag, "-o", str(exe), str(HERE / "driver.c"),
                        str(workdir / f"{name}.ll"), "-lpthread", "-lm"],
                       check=True, capture_output=True)
        exes[name] = exe

    if "openblas" in arms:
        so, prefix = find_openblas()
        if so is None:
            print("note: no OpenBLAS found (pip install numpy, or set Y_OPENBLAS_SO)",
                  file=sys.stderr)
        else:
            defs = sym_defines(so, prefix)
            if defs is None:
                print(f"note: {so.name} exports no recognised cblas_sgemm", file=sys.stderr)
            else:
                exe = workdir / "run_openblas"
                subprocess.run(["clang", "-O2", "-DARM_OPENBLAS=1", *defs, "-o", str(exe),
                                str(HERE / "driver.c"), str(so),
                                f"-Wl,-rpath,{so.parent}", "-lm"],
                               check=True, capture_output=True)
                exes["openblas"] = exe
    return exes


def time_once(exe, M, N, K, reps, rounds, threads, show_core=False):
    env = dict(os.environ, Y_NUM_THREADS=str(threads))
    if show_core:
        env["Y_SHOW_CORE"] = "1"
    r = subprocess.run([str(exe), str(M), str(N), str(K), str(reps), str(rounds),
                        str(threads)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit(f"{exe.name} failed: {r.stderr}")
    if show_core and "openblas kernel" in r.stderr:
        print("  " + r.stderr.strip().splitlines()[0])
    return float(r.stdout.strip())


def best(exe, *a):
    return min(time_once(exe, *a) for _ in range(2))


def table(exes, threads):
    have_ob = "openblas" in exes
    print(f"\n{'shape':>14} {'thr':>4} | {'Y exact':>9} {'Y f32':>9} " +
          (f"{'OpenBLAS':>9} " if have_ob else "") +
          f"| {'exact/Yf32':>10} " + (f"{'exact/OB':>9}" if have_ob else ""))
    print("-" * (78 if have_ob else 58))
    for (M, N, K, reps, rounds) in SHAPES:
        mac = M * N * K
        for t in threads:
            e = best(exes["exact"], M, N, K, reps, rounds, t)
            f = best(exes["yf32"], M, N, K, reps, rounds, t)
            row = (f"{M}^3{'':>8} {t:>4} | {mac/e/1e9:>9.1f} {mac/f/1e9:>9.1f} ")
            tail = f"| {f/e:>9.2f}x "
            if have_ob:
                o = best(exes["openblas"], M, N, K, reps, rounds, t)
                row += f"{mac/o/1e9:>9.1f} "
                tail += f"{o/e:>8.2f}x"
            print(row + tail)
    print("\n(G MAC/s. A f32 FMA is 2 flops per MAC; MAC/s is the unit both")
    print(" datapaths share. Run --isa for what each one can actually issue.)")


def scaling(exes, shapes):
    for (M, N, K, reps, rounds) in shapes:
        mac = M * N * K
        print(f"\n--- {M}x{N}x{K}, exact vpdpwssd ---")
        one = None
        for t in (1, 2, 4, 8, 16, 32):
            e = best(exes["exact"], M, N, K, reps, rounds, t)
            one = one or e
            print(f"  {t:>3} threads  {e*1e3:>9.3f} ms  {mac/e/1e9:>7.1f} G MAC/s"
                  f"   {one/e:>5.2f}x")


def isa():
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "isa_ceiling"
        subprocess.run(["clang", "-O2", "-o", str(exe), str(HERE / "isa_ceiling.c")],
                       check=True, capture_output=True)
        n = subprocess.run(["objdump", "-d", str(exe)], capture_output=True,
                           text=True).stdout.count("vpdpwssd")
        if n == 0:
            sys.exit("the probe contains no vpdpwssd - it was folded away")
        print(f"(disassembly contains {n} vpdpwssd, so the loop is real)")
        print(subprocess.run([str(exe)], capture_output=True, text=True).stdout, end="")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", default="16,32")
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--isa", action="store_true")
    args = ap.parse_args()

    if args.isa:
        isa()
        return
    threads = [int(t) for t in args.threads.split(",")]
    with tempfile.TemporaryDirectory(prefix="y_gemm_bench_") as d:
        exes = build(Path(d), {"exact", "yf32", "openblas"})
        if "openblas" in exes:
            time_once(exes["openblas"], 64, 64, 64, 1, 1, threads[0], show_core=True)
        if args.scaling:
            scaling(exes, [SHAPES[1], SHAPES[3]])
        else:
            table(exes, threads)


if __name__ == "__main__":
    main()
