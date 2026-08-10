# CPU GEMM: what was built, what was measured, what was ruled out

The LLVM/CPU backend now recognises the canonical Y matmul nest and emits a
packed, register-blocked AVX-512 kernel for it. This file records the numbers
behind every constant in `src/cpu_gemm.rs` and — more importantly — the ideas
that looked right and measured wrong, so they are not re-chased.

## Reference hardware

- AMD Ryzen 9 9950X (Zen 5), 16 cores / 32 threads
- L1d 48 KB/core · L2 1 MB/core · L3 32 MB × 2 · 64 B lines
- Full-width AVX-512: 2 × 512-bit FMA per cycle = **64 flops/cycle/core**
- Sustained ~5.0 GHz under load ⇒ **~320 GFLOPS/core** f32 peak
- Single-core sustained read bandwidth: **53 GB/s** (measured, 256 MB stream)
- Idles at 624 MHz and needs ~3 s of load to reach boost — see the ramp in
  `tests/benchmark_cpu_gemm.c`

## The baseline was not "slightly behind" — it was wrong

Before any of this, `tests/y_cpu_matmul.ysu` measured **0.52–0.79 GFLOPS** and
**did not match OpenBLAS on any shape**. Two independent miscompiles, both in
the shared LLVM backend rather than anything GEMM-specific:

1. `block_ptr2d_load`/`_store` fell through to the generic call path, where an
   unknown function's parameters default to `i32`. That emitted
   `ptrtoint ptr %B to i32` — **a 64-bit pointer truncated to 32 bits**. The
   old benchmark only worked because it allocated with `mmap(MAP_32BIT)`.
2. The helper returned a float's *bit pattern* as `i32`, and the declared `F32`
   result was produced with `sitofp` — an integer→float *conversion* of a value
   that was already a float. Every element was garbage.

Both are fixed by lowering the intrinsics natively to GEP + typed load/store.
That alone took the kernel to ~2.7–8.7 GFLOPS (still scalar) and made every
shape match. `tests/cpu_gemm_end_to_end.rs` guards it, on ordinary 64-bit
`malloc`, and asserts the emitted IR contains no `ptrtoint ptr`.

Separately, the emitted module hardcoded `target triple = x86_64-pc-windows-msvc`
and `m:w` (Windows) mangling on every host, and named `skylake-avx512` for any
AVX-512 CPU. Both are host-derived now; `sentinel::host_x86_uarch` returns
`znver4`/`znver5` from CPUID, and `None` rather than a guess for anything it
cannot name.

## OpenBLAS is not weak on ragged dimensions

The premise this work started from — that ragged shapes are where OpenBLAS
gives up — is **false on this machine**, and measuring it first saved building
the wrong thing. Single-threaded:

| shape | OpenBLAS GF |
|---|---|
| nice 1024³ | 159.5 |
| ragged 1000³ | 161.0 |
| ragged 1021³ (prime) | 159.0 |

Ragged dimensions cost it essentially nothing. What it *is* weak at is **low
arithmetic intensity**: GEMV (9.4 GF), rank-k (`4096×4096×8`, 37.5 GF), decode
(`4×4096×4096`, 41.3 GF), and tiny shapes where call overhead dominates.

## The baseline was ISA-handicapped, and the fair one has now been built

`objdump` over the original `libopenblas.so` finds **zero `zmm` instructions**
and 40,968 `ymm`. It reports `corename: ZEN`, and `kernel/x86_64/KERNEL.ZEN`
sets `SGEMMKERNEL = sgemm_kernel_8x4_haswell_2.c` — an AVX2 8×4 micro-kernel.
AVX-512 peak on this part is exactly 2× AVX2 peak (2 × 512-bit FMA/cycle =
64 flop/cycle/core, vs 32). So a large part of the early "win" was Y using an
instruction set the reference did not.

OpenBLAS *does* ship an AVX-512 sgemm — `KERNEL.SKYLAKEX` sets
`SGEMMKERNEL = sgemm_kernel_16x4_skylakex_3.c`. Rebuilding the same source with
`TARGET=SKYLAKEX USE_OPENMP=1` produces a library with **39,476 `zmm`
instructions**, and that is the baseline every headline number below is quoted
against. The AVX2 build is kept and reported separately, because it is what a
default `TARGET=ZEN` build gives you on this machine and is therefore what many
users would actually measure.

```bash
cp -a <openblas-src> /tmp/ob_avx512 && cd /tmp/ob_avx512 && make clean
make TARGET=SKYLAKEX USE_OPENMP=1 NUM_THREADS=32 NO_LAPACK=1 NO_LAPACKE=1 -j32
```

## Structure and constants

Packed BLIS shape, chosen by sweep (`MR` swept over 4/6/8/12/16):

| constant | value | why |
|---|---|---|
| `MR` | 12 | rows of C in registers; `MR × NRV` = 24 zmm accumulators |
| `NR` | 32 | 2 × `<16 x float>`; FMA:load ratio `MR·v/(MR+v)` = 1.7 |
| `KC` | 256 | `NR·KC·4` = 32 KB, fits the 48 KB L1d with the A panel |
| `MC` | 192 | `MC·KC·4` = 196 KB, fits the 1 MB L2 |
| `NC` | 2048 | `KC·NC·4` = 2 MB, streams through the 32 MB L3 |
| `SM_MR` | 8 | rows broadcast in the small-M kernel |
| `SM_MAX_M` | 8 | dispatch boundary to the small-M kernel |

`MR = 8` and `MR = 12` are within noise at 1024³ (1.89× both); 12 wins at 2048³
(1.48 vs 1.35). `MR = 16` is clearly worse everywhere (1.16 at 1024³) — 32
accumulators leaves nothing for operands.

Induction variables and accumulators are emitted as `alloca`s, not `phi` nodes.
`mem2reg` promotes them in the first function pass; the emitted inner loop
compiles to 72 FMAs, 34 broadcasts, 3 `vmovaps` and **zero spills**. The vector
width, FMA structure and blocking are explicit — that is where the performance
is — while register allocation is left to the backend.

> **Ratios in the next three sections predate the harness fixes** described
> under "Every number before this revision was inflated", and are against the
> **AVX2** OpenBLAS. Read them as *relative* A/B results within one harness —
> which is what they were used for, and which the biases largely cancel out of,
> since both arms shared them — and **not** as current standings. Anything
> quoted as a standing against OpenBLAS lives in the two results tables below.

## Two shapes needed a different loop order, not a different tile

**Small M.** At `4×4096×4096` the packed path measured 0.33× OpenBLAS. The
fix is `__y_gemm_small_m`: `k` outer, `j` inner. B is then read exactly once in
perfect linear order (it is the large operand and the shape is bandwidth-bound),
while C — only `M×N` with M small — stays cache-resident and absorbs the
accumulation. Measured effect: decode `4×4096×4096` **0.33 → 0.97**, and GEMV
`1×4096×4096` reaches **4.73×**.

**First K-block.** The micro-kernel epilogue used a `select` between "store" and
"accumulate", which needs the accumulate operand — so it read C back on *every*
K-block including the first. At `4096×4096×8` that was a 64 MB read supporting
268 MFLOP. Branching instead of selecting moved 1024³ 1.749 → 1.841 and flatK
0.803 → 0.845.

## Ruled out by measurement — do not re-chase

- **Padding waste is not what costs the skinny shapes.** `17×4096×4096` at
  `MR = 4` (15 % padding) measures 0.56, at `MR = 12` (41 % padding) 0.54.
  Sweeping `MR` does not move it.
- **Skipping the B pack when M is small is NOT a win**, despite the traffic
  arithmetic saying it should be 3×. Measured 0.29 → 0.33 at `4×4096×4096`.
  Packing is not overhead here — it is what makes B's access pattern linear.
  Reading B in place walks a 32-float column panel down 256 rows at stride N,
  touching a new page every row and revisiting each page once per panel.
- **Streaming B with `j` outer and `p` inner (prototype "variant C") is much
  worse** — 0.30 at `4×4096×4096`, 0.55–0.64 elsewhere. It walks A by column,
  one cache miss per element. The winning small-M order is `p` outer.
- **Raising `SM_MAX_M` past 8 does not help.** Thresholds 8 / 16 / 24 measure
  mean 1.622 / 1.638 / 1.599 over the shape set; at 24 the shape it was meant
  to help (`17×4096×4096`) *regresses* to 0.467 from 0.663, because B is
  re-read once per 8-row block.
- **`clang -O3` over `-O2` is worth ~3 %**, not the gap it was suspected of.
- **A single cold call is not a measurement.** The kernel read 195 GF timed
  that way and 292 GF with warmup + ramp + interleaving. The first number was
  an artifact and briefly sent this in the wrong direction.


## What OpenBLAS has that Y did not

Read from the OpenBLAS source in `/tmp/openblas_build`, not inferred:

| OpenBLAS mechanism | where | Y before | Y now |
|---|---|---|---|
| Persistent thread server | `driver/others/blas_server.c` | none | condvar-parked pool, `emit_pool_worker` |
| 2-D thread partition `nthreads_m x nthreads_n` with `switch_ratio` guard | `driver/level3/level3_thread.c:817-850` | none | 1-D, axis chosen by which yields more micro-tiles |
| `nthreads = M*N*K / (SMP_THRESHOLD_MIN * GEMM_MULTITHREAD_THRESHOLD)` | `interface/gemm.c:227` | none | same shape, larger constant (see `WORK_PER_THREAD`) |
| Shared packed-B across threads, with work-stealing over buffers (`divide_rate`, `current` rotation) | `level3_thread.c:469-570` | n/a | **still missing** — each thread re-packs B |
| Runtime-computed block sizes `sgemm_p/q/r` | `param.h` | fixed constants | fixed constants (measured) |
| `alpha`/`beta` scaling | `BETA_OPERATION` | n/a — Y's source computes `C = A*B` | n/a |
| Small-matrix bypass | `SMALL_MATRIX_OPT` | n/a | n/a — **disabled on ZEN in OpenBLAS too** (`kernel/generic/gemm_small_matrix_permit.c` returns 0) |
| Separate GEMV path | level 2 | Y's `__y_gemm_small_m` | Y's `__y_gemm_small_m` |

The last row is why Y is 3-4x OpenBLAS at `M=1`: `cblas_sgemm` with `M=1` stays
in the level-3 packed path and does not dispatch to GEMV, so it runs at
8 GFLOPS. Y routes `M <= SM_MAX_M` to a k-outer kernel.

## Threading, and four bugs found building it

1. **Every emitted function needs `#0`.** Helpers previously got AVX-512 only
   because they were inlined into the `#0` kernel. Adding the worker
   indirection stopped that, `__y_gemm_run` compiled for baseline x86-64, and
   the `<16 x float>` masked ops scalarised into per-lane `movss`/`shufps`
   bit-test chains — **290 -> 40 GFLOPS**, with correctness unchanged, so only
   the benchmark caught it.
2. **A 2.3 MB `alloca` costs on every call, not just on its path.** The
   malloc-failure fallback panel was an entry-block alloca; it emitted
   `sub $0x23b100,%rsp` on every GEMM and took `tiny 48^3` from 1.74 to 9.44 us.
   It is a BSS global now.
3. **`pthread_create` per call is far too expensive at these sizes.** ~25 us
   per thread, so ~400 us of fork/join against a 256^3 GEMM that takes 135 us
   on one thread — threading made small shapes *slower than not threading*
   (0.29x). Hence the persistent pool.
4. **The pool's completion flag published the wrong generation.** A worker
   reaching the work block via the *blocked* path carried the stale generation
   it had read on the spin path, so `done[]` lagged by one and the dispatcher
   waited forever. Found by reading the emitted IR, not by reasoning.

Idle workers park on a condition variable. An earlier version spun 8192 times
then polled `usleep(100)` forever; 15 threads waking 10k times a second
**corrupted the A/B benchmark used to tune them**, reading OpenBLAS at 15
GFLOPS on a shape where it does 460.

### A measurement warning worth more than the numbers

A hung process from the deadlock in (4) sat at 99.6% CPU for several minutes.
Every measurement taken in that window was wrong — OpenBLAS read 11 GFLOPS at
250^3 where it really does 531 — and the wrongness looked like a plausible
result (Y "winning" by 27x). **Check `ps` and the load average before trusting
any A/B run on this box**, and be suspicious of a win that large.

## Every number before this revision was inflated. Six measurement bugs.

The first published tables claimed a 16-thread mean of **1.60–1.63x** and a
single-thread mean of **1.60x**. Both were wrong. Six separate biases were
found, and **every one of them favoured Y**. That one-sidedness is the real
lesson: a harness written by the author of the thing being measured will drift
in its favour unless each step is checked against an independent measurement.

| # | bias | size of the error |
|---|---|---|
| 1 | Both libraries timed **in one process**. OpenBLAS's idle threads spin before parking, so whichever ran second was timed against a busy machine. | 512³ 16T scaling read **1.95x** interleaved vs **7.35x** standalone |
| 2 | `openblas_set_num_threads()` on a **USE_OPENMP** build. This OpenBLAS is built with libgomp; driving the count through the OpenBLAS API instead of `OMP_NUM_THREADS` handicaps it. | 1024³ 16T: **1621** GF via the API vs **1825** GF via the env var |
| 3 | Timed batch sized from an **assumed 100 GFLOPS**: `iters = 20ms / (flops/100e9)`. On a machine doing 2000+ GF that is **three** iterations at 1024³ — a 4 ms timed region, not the 20 ms the comment claimed. | contributed to #6 |
| 4 | **Warm-up ran at one thread**, then timing ran at sixteen, so the first timed round paid for OpenMP team creation. | small, but always against the baseline |
| 5 | Clock ramp spun **one core**, leaving fifteen at the 624 MHz idle floor. | biases whichever side is threaded |
| 6 | All 18 shapes run **in one process**. Later shapes are depressed, and Y more than OpenBLAS. | 2048³ measures **2289** GF as the 4th shape of a sweep and **3082** GF alone; Y/OB moves 0.70 → 0.88 |
| 7 | The `SHAPE_FILTER` added to isolate #6 was a **substring** match, and `"48^3"` matches `"2048^3"`. | the `tiny 48^3` row silently reported 2048³'s numbers |

Bug 7 is the same shape as this repo's standing design rule: a filter that
matches the wrong row does not fail, it **relabels**. The filter is an index now.

The clock probe added in the same revision was also **constant-foldable** — it
timed a dependent FMA chain, `clang -O2` deleted it, and it reported
**4,000,406 GHz**. It reads `scaling_cur_freq` from sysfs now. That is the same
trap `measure_accumulate_costs` hit on the GPU side; a probe reporting an absurd
number is lucky, one reporting a plausible wrong number is not.

**Methodology now: one shape, one library, one process, three launches, seven
rounds, ranked by the minimum.** Thread counts come only from the environment
(`Y_NUM_THREADS`, `OMP_NUM_THREADS`); the harness never calls
`openblas_set_num_threads`. Y's process runs OpenBLAS at one thread with
`OMP_WAIT_POLICY=passive`, so even the single-threaded reference GEMM cannot
leave a spinning team behind. Driver: `tests/run_cpu_gemm_bench.py`.

### Run-to-run spread, and what it means for the ratios

Measured by launching the identical binary on the identical shape six times:

| | 1024³, 16 threads |
|---|---|
| Y | 2241 – 2646 GF (±8%) |
| OpenBLAS | 1688 – 1895 GF (±5.5%) |

So the combined uncertainty on a ratio is roughly **±13%**, and the `jit`
column (worst median/min for the shape) flags runs that were worse than that.
**A ratio between 0.9 and 1.1 on this box is a tie, not a result.** Several
shapes below are exactly that, and are reported as parity rather than as wins.

> **The next three tables are the BASELINE state**, before the tile sweep and
> the runtime `kc`. They are kept because they are the correctly-measured
> starting point every later gain is quoted against, and because their `%pk`
> columns are what identified the mainloop as the thing to fix. Current
> standings are under "Results after both changes".

## Results, 16 threads, vs AVX-512 OpenBLAS (the fair baseline)

`TARGET=SKYLAKEX`, 39,476 `zmm`. All-core clock 5.09 GHz, so AVX-512 peak is
64 × 16 × 5.09 = 5212 GF. Every shape gated on relative L2 < 1e-5.

| shape | Y GF | OB GF | Y/OB | Y %pk | OB %pk |
|---|---|---|---|---|---|
| gemv 1×8192×8192 | 30.0 | 15.4 | **1.94** | 1% | 0% |
| nice 512³ | 2422 | 1723 | **1.41** | 46% | 33% |
| skinny 17×4096×4096 | 869 | 618 | **1.41** | 17% | 12% |
| flatK 4096×4096×8 | 433 | 325 | **1.33** | 8% | 6% |
| skinny 33×4096×4096 | 1172 | 918 | **1.28** | 22% | 18% |
| ragged 333×777×64 | 681 | 557 | **1.22** | 13% | 11% |
| gemv 1×4096×4096 | 54.8 | 48.8 | 1.12 | 1% | 1% |
| nice 1024³ | 2983 | 2701 | 1.10 | 57% | 52% |
| ragged 250³ | 625 | 624 | 1.00 | 12% | 12% |
| deepK 64×64×32768 | 396 | 400 | 0.99 | 8% | 8% |
| nice 256³ | 822 | 842 | 0.98 | 15% | 16% |
| ragged 1000³ | 2663 | 2756 | 0.97 | 51% | 53% |
| ragged 1021³ prime | 2446 | 2789 | 0.88 | 47% | 54% |
| nice 2048³ | 2813 | 3542 | 0.79 | 53% | 67% |
| ragged 137×391×1013 | 759 | 1006 | 0.75 | 15% | 19% |
| decode 4×4096×4096 | 114 | 178 | 0.64 | 2% | 3% |
| tiny 48³ | 125 | 289 | 0.43 | 2% | 6% |
| decode 8×4096×4096 | 108 | 366 | 0.30 | 2% | 7% |

**Geomean 0.95, arithmetic mean 1.03, Y ahead on 9 of 18.** An earlier run of
the same protocol gave geomean 1.02; the honest statement is **parity overall,
±0.05**.

Read by category rather than by mean, which is where the useful signal is:

- **Skinny / flat / rank-k / GEMV — Y ahead, 1.2–1.9x.** This is the class the
  work was aimed at, and the advantage survived every correction.
- **Large square — OpenBLAS ahead, 0.79–1.10.** 2048³ is the clearest loss and
  is a real one, not noise (`jit` 1.06).
- **Small square — tie.** 256³ and 250³ are 0.98–1.00.
- **Decode (M = 4–8) and tiny 48³ — Y clearly behind**, 0.30–0.64. Real, and
  diagnosed under "Known gaps".

## Results, single-threaded, vs AVX-512 OpenBLAS

The most revealing table in this file, and the one that most contradicts the
original claim of a 1.60x single-thread mean.

| shape | Y GF | OB GF | Y/OB | Y %pk | OB %pk |
|---|---|---|---|---|---|
| gemv 1×4096×4096 | 33.4 | 3.8 | **8.85** | 9% | 1% |
| gemv 1×8192×8192 | 25.5 | 3.5 | **7.25** | 7% | 1% |
| decode 4×4096×4096 | 30.6 | 14.9 | **2.05** | 9% | 4% |
| decode 8×4096×4096 | 53.5 | 28.7 | **1.86** | 15% | 8% |
| ragged 333×777×64 | 283 | 278 | 1.02 | 80% | 79% |
| ragged 1021³ prime | 298 | 308 | 0.97 | 84% | 87% |
| nice 1024³ | 295 | 310 | 0.95 | 84% | 88% |
| nice 512³ | 287 | 306 | 0.94 | 81% | 87% |
| ragged 1000³ | 295 | 317 | 0.93 | 84% | 90% |
| flatK 4096×4096×8 | 35.9 | 38.6 | 0.93 | 10% | 11% |
| skinny 33×4096×4096 | 86.4 | 96.9 | 0.89 | 24% | 27% |
| skinny 17×4096×4096 | 51.8 | 58.0 | 0.89 | 15% | 16% |
| ragged 137×391×1013 | 244 | 277 | 0.88 | 69% | 78% |
| ragged 250³ | 249 | 286 | 0.87 | 71% | 81% |
| nice 256³ | 242 | 298 | 0.81 | 69% | 85% |
| nice 2048³ | 236 | 307 | 0.77 | 67% | 87% |
| deepK 64×64×32768 | 138 | 252 | 0.55 | 39% | 72% |
| tiny 48³ | 126 | 297 | 0.42 | 36% | 84% |

**Geomean 1.17. Arithmetic mean 1.77. Y ahead on 5 of 18.**

Those three numbers describe the same data and the arithmetic mean is the
useless one: it is 1.77 only because two GEMV rows are 8.85 and 7.25, and a
single-threaded sgemm at M=1 is a case OpenBLAS routes through its packed GEMM
path instead of a GEMV kernel. **Quote the geomean and the win count**; the
original "mean 1.60x" headline was an arithmetic mean over a set containing
exactly this kind of outlier.

What the `%pk` columns say, which no ratio does: on the square shapes
OpenBLAS's single-core micro-kernel reaches **85–90% of AVX-512 peak** and Y's
reaches **67–84%**. **Y's mainloop is roughly 5–15% behind OpenBLAS's**, and the
16-thread tables were partly hiding that behind better thread scaling. The two
outright single-core losses — `tiny 48³` at 0.42 and `deepK 64×64×32768` at
0.55 — are both cases where Y pays a full pack for very little reuse.

## Results, 16 threads, vs AVX2 OpenBLAS (`TARGET=ZEN`, as shipped)

Same protocol, same Y binary, only the baseline library differs. This is the
comparison the earlier tables were making, measured correctly.

| shape | Y GF | OB GF | Y/OB |
|---|---|---|---|
| flatK 4096×4096×8 | 511 | 237 | **2.15** |
| nice 512³ | 2402 | 1291 | **1.86** |
| nice 1024³ | 2772 | 1833 | **1.51** |
| nice 2048³ | 3033 | 2025 | **1.50** |
| ragged 1000³ | 2531 | 1709 | **1.48** |
| ragged 333×777×64 | 675 | 456 | **1.48** |
| ragged 1021³ prime | 2430 | 1688 | **1.44** |
| skinny 17×4096×4096 | 780 | 553 | **1.41** |
| gemv 1×8192×8192 | 28.9 | 20.9 | **1.39** |
| gemv 1×4096×4096 | 60.4 | 48.7 | **1.24** |
| skinny 33×4096×4096 | 1145 | 986 | **1.16** |
| nice 256³ | 772 | 689 | 1.12 |
| ragged 250³ | 625 | 590 | 1.06 |
| deepK 64×64×32768 | 398 | 388 | 1.03 |
| tiny 48³ | 122 | 119 | 1.02 |
| ragged 137×391×1013 | 684 | 777 | 0.88 |
| decode 4×4096×4096 | 119 | 195 | 0.61 |
| decode 8×4096×4096 | 117 | 349 | 0.34 |

**Geomean 1.17, arithmetic mean 1.26, Y ahead on 15 of 18** — against the
1.60–1.63 arithmetic mean the broken harness reported for this same pairing.
So roughly **half** the original claim was measurement error and the other half
was the AVX2/AVX-512 instruction-set gap.

## The register shape was never swept, and that was worth 40%

`MR` had been swept over 4/6/8/12/16. `NRV` — vectors of C per micro-kernel row
— had **not**, because it was a `const` and moving it meant editing and
rebuilding the compiler. So `MR x NRV = 12x2` was the default by inheritance,
not by measurement.

The blocking parameters are runtime-settable now (`Y_GEMM_TILE=mr,nr,kc,mc,nc`,
validated by `Tile::check`, which **refuses** a tile needing more than 32 zmm
registers rather than emitting one that spills every accumulator in the
innermost loop and reads as a mysterious 3x slowdown). The sweep is a shell
loop. Single-threaded GFLOPS:

| tile (mr,nr,kc,mc,nc) | 1024³ | 2048³ | 512³ | 256³ |
|---|---|---|---|---|
| 12,32,256,192,2048 *(old default)* | 294 | 208 | 292 | 249 |
| 14,32,256,192,2048 | 296 | 211 | 281 | 247 |
| 8,48,256,192,2048 | 302 | 232 | 300 | 258 |
| 6,64,256,192,2048 | 310 | 243 | 302 | 281 |
| 6,64,512,256,2048 | 307 | 275 | 306 | 279 |
| **6,64,1024,384,2048** | **311** | **298** | 307 | 279 |

FMA-per-memory-op in the micro-kernel is `mr*nrv / (mr + nrv)`: 1.71 at 12x2,
2.25 at 8x3, **2.40 at 6x4**. The ranking follows it, which is why this is a
structural result rather than a lucky point.

One cliff worth knowing: `mc=512` measures **185 GFLOPS** at 2048³ where
`mc=384` measures 296 and `mc=768` measures 295. That is an associativity
artefact, not a trend — do not interpolate through it.

## `kc` cannot be a constant: its optimum depends on the thread count

Tuning `kc` on one thread and shipping it is a trap, and this repo fell into it
for one afternoon. The two cases pull in opposite directions:

- **One thread wants `kc` large.** C is read and written once per K-panel, so C
  traffic scales as `K/kc`. Raising 256 → 1024 took 2048³ from 208 to 298
  GFLOPS, **+44%**.
- **Sixteen threads want `kc` small.** Each thread packs a private `kc x nc` B
  panel, so the aggregate is `nthr*kc*nc*4` bytes. At `kc=1024, nc=2048` that is
  8.4 MB per thread and **134 MB across 16 threads, against a 64 MB L3**. With
  the single-threaded optimum baked in, 1024³ collapsed from 2983 to **851**
  GFLOPS — 0.34x OpenBLAS. The single-thread table above is entirely silent
  about this.

`kc` is therefore chosen at runtime in `emit_driver`, budgeting the aggregate
panel against L3:

```
kc = clamp(L3_PANEL_FLOATS / (nthr * nc), KC_MIN, KC_MAX)
```

At `nc=2048` this yields **1024 for one thread and 256 for sixteen** — the two
values the independent sweeps picked. That agreement is the only reason to
trust a formula here rather than a lookup table.

It also beats both fixed values, because `nc` is the *per-thread* N extent: when
the partition splits N, each thread's `nc` is small and `kc` can go deep
without any L3 cost. 2048³ on 16 threads measured 1957 GFLOPS at fixed
`kc=256`, 2074 at fixed `kc=1024`, and **2882 with the runtime rule**.

**Trap this introduced:** the packed-B pointer used to be `scratch + (mc+mr)*kc`
with the *tile's* `kc`. Once `kc` is a runtime value that can exceed it, the A
panel's tail lands on top of the B panel. The offset is `(MC_MAX+MR_MAX)*KC_MAX`
now, matching what `SCRATCH_FLOATS` already reserved. A tile-derived offset here
is a silent wrong answer on exactly the shapes the runtime `kc` exists to speed
up.

## Results after both changes

Same protocol as the tables above (one shape, one library, one process), two
launches rather than three, against the AVX-512 OpenBLAS.

| | before | after |
|---|---|---|
| **1 thread, geomean** | 1.17 | **1.27** |
| 1 thread, 2048³ | 0.77 (67% of peak) | **0.94** (81% of peak) |
| 1 thread, 1024³ | 0.95 | 0.94 |
| 1 thread, deepK 64×64×32768 | 0.55 | **0.75** |
| 1 thread, skinny 17×4096×4096 | 0.89 | **1.38** |
| **16 threads, geomean** | 0.95 | **0.97** |
| 16 threads, 137×391×1013 | 0.75 | **1.00** |
| 16 threads, deepK 64×64×32768 | 0.99 | **1.68** |
| 16 threads, 1021³ prime | 0.88 | **1.00** |
| 16 threads, 1024³ | 1.10 | **0.89** |
| 16 threads, skinny 33×4096×4096 | 1.28 | **0.93** |

Single-thread square GEMM now runs at **78–84% of AVX-512 peak** against
OpenBLAS's 84–89%, where it was 67–84%. The 16-thread picture is a wash on the
mean with large movements underneath — 1024³ and the skinny shapes regressed and
have not been explained yet. **Do not read the 16-thread mean as "unchanged";
read the rows.**

## Decode was two separate bugs, and the obvious one was the smaller one

`decode 8x4096x4096` at **0.36x** on 16 threads was the worst ratio in the
table. The standing diagnosis — "the split axis is wrong" — was right, and was
not the whole story: **single-threaded the same shape moved B at 13 GB/s
against a 53 GB/s single-core read bandwidth**, so the inner loop was already
leaving 4x on the table before any thread was added. Both had to be fixed;
either alone leaves most of the gap.

### 1. The partition: cut K, not N

`__y_gemm_small_m` runs `p` outer and `j` inner, so a thread given `N/nthr`
columns reads a **1 KB slice of every 16 KB row of B**. It walks B's whole
64 MB address span to consume 4 MB of it, and no prefetcher can follow that.
The aggregate is one pass over B, so the traffic arithmetic says nothing is
wrong — the *pattern* is what costs.

A K-band is contiguous: thread `t` reads rows `[t*K/nt, (t+1)*K/nt)` end to end.
The price is that every thread now contributes a partial sum to every element of
C, so each needs a private `M x N` panel and the panels must be summed. That
panel is the thread's existing packing scratch, which this path never otherwise
touches, so it costs no allocation; the reduction is 2 MB against B's 64 MB.

Isolated in a standalone probe (identical inner loop, only the partition
differing), 8x4096x4096 at 16 threads, three runs of the same protocol:

| partition | run 1 | run 2 | run 3 |
|---|---|---|---|
| N-split (what shipped) | 62 | 74 | 66 |
| K-split | 405 | 427 | 371 |

**Read the ratio, not the values.** B is 64 MB against this part's 64 MB L3, so
the probe's absolutes move about 2x between runs — an earlier variant list put
"K-split plain" at 221-242 on exactly the same code. See the measurement note
below.

### 2. The inner loop: C is read and written once per K-step

One B vector drives `mw` C loads, `mw` FMAs and `mw` stores, so the load/store
ports run out long before the FMA pipes do — and at `M = 8, N = 4096` the C
strip is 128 KB, so those accesses miss L1 and run at L2 bandwidth. Holding C
in a register across `SM_PU` consecutive K-steps divides that traffic by
`SM_PU` while the FMA count is unchanged.

`SM_PU = 4`, and **the register-pressure argument against it is wrong**. The A
broadcasts are `SM_MR * SM_PU` values live across the j loop, so 8x4 is 32
vectors and LLVM must spill. It is still faster, because a spilled broadcast
reloads from L1 while the C traffic it removes was missing L1 entirely.
Measured against `SM_PU = 2` through the real emitter and the real harness,
three interleaved launches, best of each:

| shape | 1 thread | 16 threads |
|---|---|---|
| decode 8x4096x4096 | **1.19x** | **1.18x** |
| decode 4x4096x4096 | **1.41x** | **1.26x** |
| gemv 1x4096x4096 | 0.98x | 0.98x |
| gemv 1x8192x8192 | 0.98x | 1.00x |

The unroll is **gated on the row span**, and the threshold is `SM_PU_MIN_ROWS`
= 2, deliberately *not* `SM_PU`. At one live row there is no C traffic to
divide — a `1 x N` strip is L1-resident already — while the broadcast block is
emitted regardless: `gemv 1x4096x4096` measured 31.2 GFLOPS un-unrolled against
27.4 unrolled.

### Results, against the same OpenBLAS build

Strict A/B: one shape per process, arms interleaved within a launch, arm order
alternated between launches, three launches, ranked by the best. OpenBLAS
measured in the **same session** — quoting geomeans from two sweeps taken at
different times is not valid here, because the OB column alone moved 35%
between two such sweeps.

**16 threads**

| shape | before | after | OpenBLAS | before/OB | after/OB |
|---|---|---|---|---|---|
| decode 8x4096x4096 | 126.8 | **607.9** | 356.3 | 0.36 | **1.71** |
| decode 4x4096x4096 | 123.6 | **599.2** | 184.2 | 0.67 | **3.25** |
| gemv 1x4096x4096 | 58.3 | **96.9** | 49.6 | 1.18 | **1.95** |
| gemv 1x8192x8192 | 31.3 | **34.7** | 14.6 | 2.15 | **2.38** |

**Geomean over all 18 shapes: 0.96 -> 1.20. Y ahead on 8 -> 11 of 18.**

**1 thread** (the unroll alone; the K-split does not run below two threads)

| shape | before | after | OpenBLAS | before/OB | after/OB |
|---|---|---|---|---|---|
| decode 8x4096x4096 | 52.8 | **93.8** | 28.9 | 1.82 | **3.24** |
| decode 4x4096x4096 | 31.2 | **69.6** | 14.6 | 2.14 | **4.77** |
| gemv 1x4096x4096 | 34.1 | 35.9 | 3.8 | 9.03 | 9.50 |
| gemv 1x8192x8192 | 26.0 | 26.6 | 3.3 | 7.78 | 7.96 |

**Geomean over all 18 shapes: 1.27 -> 1.38.**

**Nothing else moved.** Single-threaded, every one of the fourteen shapes that
does not route to the small-M kernel measures between **0.98 and 1.03**. That
is the load-bearing control, not the 16-thread table: at one thread the pool
and the partition are bypassed entirely and the launch-to-launch spread is a
few percent, whereas at 16 threads several shapes swing 10-50% between
launches and the same controls read 0.96-1.15 — inside their own spread, and
therefore evidence of nothing either way.

`gemv 1x8192x8192` is the one target shape that barely moved (1.11x at 16
threads). Its B is 256 MB, four times the L3, so it is genuinely DRAM-bound and
the partition can only buy prefetch efficiency, not reuse.

### Four traps, three of them found the hard way

1. **`cut_m` is true for shapes whose M is 8.** It is `pm >= pn` with
   `pn = N / NR`, so at `N = 33` and `NR = 64` both sides are 1. The task fill
   then assigned the reduction's *column* bounds to the *M* range, thread 0's
   band came out empty, it took the driver's early return, and it never reached
   a barrier sized for three threads — the dispatcher spun forever. **Every
   shape in the benchmark has `N >= NR` and hid this.** `cut_m` is now
   explicitly `cut_m_raw && !cut_k`: a new partition mode has to be subtracted
   from the old one at every consumer, not only where it is introduced.
2. **Under the K-split an empty column band is not an idle thread.** `[n0, n1)`
   is the band of the *reduction*; the thread still has a K-band to accumulate
   and still has to arrive at the barrier. The driver's "nothing to do" early
   return therefore tests `n0 >= n1` only when `ksplit == 0`.
3. **The unroll depth and the row gate must not be the same constant.** They
   were, and it made `SM_PU = 8` look catastrophic: `decode 4x4096x4096`
   measured **0.47x** against `SM_PU = 4` single-threaded, because a 4-row
   shape fails `mspan >= 8` and falls all the way to the un-unrolled loop. That
   is the gate firing, not the depth being wrong, and reading it as "8 is too
   deep" would have been the wrong conclusion from a real measurement. Hence
   `SM_PU_MIN_ROWS`. (8 is still not shipped: with the gate decoupled it wins
   `decode 8` by 1.06-1.14x, inside the spread, and was not pursued further.)
4. **A `select` on the unrolled loop's exit bound is not free.** `kmain` is the
   unrolled loop's exit and the remainder loop's entry, so choosing it with a
   `select` hides from LLVM that the two ranges partition `[k0, k1)`. Emitting
   two whole loop nests under one branch keeps both bounds affine. Measured at
   `SM_PU = 2`, best of three in one session: 69.4 GFLOPS affine against 59.8
   with the `select` — arms not order-alternated, and not re-measured since
   `SM_PU` moved to 4, so treat the 14% as indicative rather than as a figure.

### The measurement note that matters more than any number here

**B is 64 MB and this part's L3 is 64 MB, so this shape sits on a capacity
cliff, and the standalone probe is not a reliable instrument at this size.**
Repeated calls leave B partly resident, and which side of the cliff a variant
lands on flips between runs: the same probe configuration measured **689 and
270 GFLOPS** on two runs of an identical interleaved, min-ranked protocol, and
"K-split plain" read 221-242 with one variant list and 371-427 with another.

Every number in this section that is used to *decide* something therefore comes
from the real emitter through `tests/run_cpu_gemm_bench.py`-style isolation,
not from the probe. The probe is cited only for the partition ratio, where the
effect is 5-6x and survives the instability. Two consequences worth carrying:

- **Anything tuned at this size to finer than ~2x is tuning noise.** Prefer a
  fix with no constant to pick.
- **Measure the inner loop single-threaded.** Jitter drops to a few percent,
  the pool and partition drop out, and that is what separated a real 18%
  codegen effect from run-to-run drift — and what proved the packed path
  untouched.

### Ruled out by measurement — do not re-chase

- **Blocking the `j` loop so the C strip fits L1 is not a shippable win.**
  It is the textbook fix for the L2-bound C traffic and it does help in
  isolation, but the best block width swung 2.3x between neighbouring values
  and reversed order between runs. That is the capacity cliff above, not a
  tuning surface. The K-unroll gets the same benefit structurally, without a
  constant to pick.
- **Padding the private panel's row stride** to break the 16 KB L1 set
  aliasing measured *worse* than not padding at `M = 8` (463 vs 568) and better
  at `M = 1`. Same instability; no consistent effect.

## Known gaps

Ordered by measured size against the AVX-512 baseline, not by guess.

1. **`tiny 48³` at 0.43 (1T) / 0.45 (16T)** — the largest single ratio, and the
   least explained. Y packs A and B unconditionally; below roughly `mc x nc`
   there is nothing to amortise the pack against, while OpenBLAS has a cheap
   small-size route. A copy-free direct path for `M, N, K < 64` is the fix.
   Note Y holds 36% of peak here and OpenBLAS 84%, so this is not call overhead.
2. ~~**`decode 8×4096×4096` at 0.38 on 16 threads** — the split-axis
   problem.~~ **Fixed** — see "Decode was two separate bugs" above. The K-split
   was indeed the fix, and the prediction that "the kernel is fine and the
   partition is not" was only half right: the inner loop was also leaving 4x on
   the table single-threaded, and the K-unroll that addresses it is worth a
   further 1.18–1.19x on top of the partition. **0.36 -> 1.71x on 16 threads,
   1.82 -> 3.24x on one.**
3. **`1024³` regressed 1.10 → 0.89 and the skinny shapes 1.28/1.41 → 0.93/0.90
   on 16 threads.** New, from today's changes, and unexplained. The runtime
   `kc` rule uses the per-thread `nc`, so an M-split leaves `nc` at full width
   and pins `kc` to 256 while an N-split lets it go deep — the two partitions
   now get very different K-panels and that interaction was never measured.
   **This is the first thing to look at next.**
4. **`deepK 64×64×32768` at 0.75 single-threaded** (was 0.55). M=N=64 means one
   `mc` block and one `nc` block, so the whole cost is packing against very
   little reuse.
5. **The split is 1-D.** OpenBLAS partitions M and N together. With one axis, a
   shape small in both caps out at `max(M/mr, N/nr)` threads. There are three
   axes to *choose* from now — M, N, and K for the small-M path — but still only
   one is cut per call, so the cap stands for the packed path.
6. **Zero prefetch instructions** in the emitted kernel — `objdump | grep -c
   prefetch` returns 0. OpenBLAS's micro-kernels prefetch the next A and B
   panels aggressively. Untried.
7. **The `SHARE_B_WORK` gate rests on contaminated data.** Shared packed-B was
   measured a regression using the interleaved harness, which is now known to
   have been wrong. Re-measure before trusting the gate.
8. **f32 only**, and the kernel is **not reentrant** — one process-wide pool and
   one task array, so concurrent calls from different threads would corrupt
   each other. Single-threaded callers are the assumption.
9. `cpu_specializer.rs` and the `emit_*_kernel` additions in `cpu_emitter.rs`
   are unrelated earlier scaffolding: they classify shapes and emit naive
   scalar Rust text, are reachable from no backend path, and are not what any
   number here measures.

## Reproducing

```bash
cargo build --release
./target/release/Y tests/y_cpu_matmul.ysu --emit-llvm
clang -O3 -c tests/y_cpu_matmul.ll -o /tmp/y_cpu_matmul.o

# fair baseline (AVX-512)
clang -O2 -o tests/benchmark_cpu_gemm_avx512 tests/benchmark_cpu_gemm.c \
      /tmp/y_cpu_matmul.o -L/tmp/ob_avx512 -lopenblas -lm -lpthread
LD_LIBRARY_PATH=/tmp/ob_avx512 python3 tests/run_cpu_gemm_bench.py \
      --bin ./tests/benchmark_cpu_gemm_avx512 --threads 16 --isa avx512

# as-shipped baseline (AVX2)
LD_LIBRARY_PATH=/tmp/openblas_build python3 tests/run_cpu_gemm_bench.py \
      --bin ./tests/benchmark_cpu_gemm --threads 16 --isa avx2

cargo test --release --test cpu_gemm_end_to_end          # correctness, 1 thread
cargo test --release --test cpu_gemm_threaded            # correctness + no hang, 1..16
```

`cpu_gemm_end_to_end` never sets `Y_NUM_THREADS` and its largest shape is
`193x65x257` — below `WORK_PER_THREAD` — so **every shape it runs is
single-threaded**, and the pool, the partition, the barrier and the K-split
reduction had no coverage at all until `cpu_gemm_threaded` was added. That file
runs each shape under 1/2/3/7/16 threads with a timeout, because the failure
mode of a miscounted barrier is a hang rather than a wrong answer, and it
carries the tiny-`N` shapes (33, 15) that the benchmark set cannot express —
those are what caught trap 2 above.

**Check the box is idle first.** A hung process at 99.6% CPU once made
OpenBLAS read 11 GF at 250³ where it really does 531, and the result looked
like a plausible 27x win rather than like an error:

```bash
top -bn2 -d1 | grep '^%Cpu' | tail -1     # want >= 93% idle
```

`SHAPE_INDEX=<n>` on the C driver runs one shape in isolation, which is how
bias #6 above was found and is the right tool for re-checking any single ratio.
