#!/bin/bash
# Runs Y and OpenBLAS in SEPARATE processes and merges.
#
# Thread count is passed through the ENVIRONMENT to both sides, never through
# openblas_set_num_threads(): this OpenBLAS is a USE_OPENMP build, and driving
# it through the API instead of OMP_NUM_THREADS costs it 12% at 1024^3.
#
# Y's process runs OpenBLAS single-threaded and with OMP_WAIT_POLICY=passive,
# so the reference GEMM cannot leave a spinning libgomp team competing with the
# kernel under test.
set -e
T=${1:-16}; R=${2:-7}; BIN=${BIN:-./tests/benchmark_cpu_gemm}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/tmp/openblas_build}

env -u OMP_NUM_THREADS \
    Y_NUM_THREADS=$T OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive \
    $BIN $T $R y  > /tmp/bench_y.txt
sleep 2
env -u Y_NUM_THREADS \
    OMP_NUM_THREADS=$T \
    $BIN $T $R ob > /tmp/bench_ob.txt
sleep 1

python3 - "$T" <<'PY'
import sys
def load(p):
    d={}
    for l in open(p):
        if l.startswith('#') or l.startswith('shape') or l.startswith('ramping') or not l.strip():
            continue
        f=l.split()
        if len(f)<5: continue
        try: d[" ".join(f[:-4])]=(float(f[-4]),float(f[-3]),float(f[-2]))
        except ValueError: continue
    return d
y,o=load('/tmp/bench_y.txt'),load('/tmp/bench_ob.txt')
T=int(sys.argv[1])
# ISA peak: Zen5 core does 2x512-bit FMA/cycle = 64 flop/cycle for AVX-512,
# 32 flop/cycle for AVX2. This OpenBLAS contains zero zmm instructions.
GHZ=5.0
peak_y  = 64*T*GHZ
peak_ob = 32*T*GHZ
print(f"\nY vs OpenBLAS, {T} threads, separate processes, env-driven thread count")
print(f"{'shape':<24}{'Y GF':>9}{'OB GF':>9}{'Y/OB':>8}{'Y %pk':>7}{'OB %pk':>8}{'ISA-norm':>10}{'jitter':>8}")
print("-"*83)
rs=[];ns=[]
for k in y:
    if k not in o: continue
    r=y[k][1]/o[k][1]; rs.append(r)
    n=(y[k][1]/peak_y)/(o[k][1]/peak_ob); ns.append(n)
    jit=max(y[k][2],o[k][2])
    print(f"{k:<24}{y[k][1]:9.1f}{o[k][1]:9.1f}{r:8.2f}"
          f"{100*y[k][1]/peak_y:6.0f}%{100*o[k][1]/peak_ob:7.0f}%{n:10.2f}{jit:8.2f}")
print("-"*83)
g=lambda v:(lambda p:p**(1.0/len(v)))(__import__('functools').reduce(lambda a,b:a*b,v,1.0))
print(f"{'geomean':<24}{'':9}{'':9}{g(rs):8.2f}{'':6}{'':7}{g(ns):10.2f}")
print(f"{'arith mean':<24}{'':9}{'':9}{sum(rs)/len(rs):8.2f}{'':6}{'':7}{sum(ns)/len(ns):10.2f}")
print("\nISA-norm = ratio of achieved fraction-of-peak. >1 means Y's *algorithm*")
print("is ahead; the raw Y/OB column also contains a 2.0x instruction-set gap,")
print("because this OpenBLAS build emits no AVX-512 at all (0 zmm in the .so).")
print("jitter = worst median/min across the two runs; >1.05 means contended.")
PY
