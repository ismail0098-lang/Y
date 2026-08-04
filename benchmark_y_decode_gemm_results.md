# Real Y-Compiler Tensor Core GEMM vs cuBLAS - Decode-Shaped (Skinny) M

Every number below is measured from `target/release/Y tests/gemm_f16_decode_m<M>.ysu --emit-ptx`'s actual output, at `M in {1, 4, 8, 32}`, `N=K=4096` (autoregressive decode batch sizes against a typical LLM hidden dim).

## Measurement validity: `stream` vs `resident`

At `N=K=4096` FP16 the B (weight) matrix is 4096*4096*2 = 33.55 MB, and this project's dev GPU (RTX 4070 Ti SUPER, AD103) has a 48 MB L2. A timing loop over a SINGLE B buffer - which is what this benchmark did before this revision - therefore measures an L2-resident working set, not the DRAM-bandwidth regime real decode runs in, where each of a model's ~32 layers has its own weight matrix and nothing survives between tokens.

`resident` rows reproduce that old single-buffer loop. `stream` rows rotate over 6 distinct B matrices (201 MB, ~4.0x L2) so a buffer is evicted before it comes around again. Both implementations read the same aliased device memory and rotate identically. Effective GB/s is `(K*N*2 + M*K*2 + M*N*4) / time`; this card's theoretical DRAM peak is 672 GB/s, so any row above that is provably being served out of cache rather than DRAM.

Timing is the median of 5 independent process launches per (M, mode) (range in brackets); a row is marked inconclusive rather than given a speedup number if the Y and cuBLAS ranges overlap.

| M | mode | CTA Tile | dyn smem B | cuBLAS us (median [range]) | Y us (median [range]) | Y vs cuBLAS | Y GB/s | cuBLAS GB/s | Correct |
|---|---|---|---|---|---|---|---|---|---|
| 1 | stream | 32x64x64 | 41472 | 57.77 [56.34, 60.15] | 57.62 [56.28, 59.83] | inconclusive | 583 | 581 | OK |
| 1 | resident | 32x64x64 | 41472 | 28.08 [27.31, 37.09] | 22.00 [21.18, 23.58] | **1.28x** | 1526 | 1196 | OK |
| 4 | stream | 32x64x64 | 41472 | 58.92 [58.71, 60.66] | 57.97 [56.48, 59.99] | inconclusive | 581 | 571 | OK |
| 4 | resident | 32x64x64 | 41472 | 57.62 [54.82, 60.01] | 25.74 [24.66, 26.89] | **2.24x** | 1307 | 584 | OK |
| 8 | stream | 32x64x64 | 41472 | 60.64 [57.55, 61.55] | 56.49 [56.33, 58.35] | inconclusive | 597 | 557 | OK |
| 8 | resident | 32x64x64 | 41472 | 53.42 [53.10, 53.76] | 25.11 [24.91, 29.76] | **2.13x** | 1344 | 632 | OK |
| 32 | stream | 32x64x64 | 41472 | 59.44 [57.61, 61.88] | 59.36 [57.08, 63.26] | inconclusive | 579 | 578 | OK |
| 32 | resident | 32x64x64 | 41472 | 24.39 [23.62, 24.71] | 32.31 [30.87, 33.69] | **0.75x** | 1063 | 1408 | OK |

## What the corrected measurement changed

The previously reported **0.35x at M=32** was almost entirely measurement
artifact, not kernel behaviour. Two independent contaminants stacked:

1. **L2 residency** - a single 33.55 MB B buffer against a 48 MB L2.
2. **GPU clock ramp** - SM clock idles at ~210 MHz here and needs ~3s of
   sustained load to reach ~2670 MHz (12.7x), and clocks cannot be locked on
   this box. A 10-iteration warmup does not ramp them. Worse, the old harness
   timed *all* of Y and then *all* of cuBLAS, so cuBLAS always ran on the
   hotter clock - a systematic bias, not noise. Run-to-run spreads of 10x
   (Y "[59.20, 602.86]us") were the visible symptom.

With both fixed, Y measures **0.94-0.96x of cuBLAS** in `stream` mode, and the
autotuner change below brings that to statistical parity at M=4/8/32.

## Autotuner: what a measured tile sweep showed

Sweeping the real candidate space (via `Y_CTA_OVERRIDE`) rather than trusting
`score_candidate`'s model contradicted the obvious hypothesis. More CTAs per SM
is *monotonically worse* here - 13 CTAs/SM gives 322 GB/s, 1 CTA/SM gives 539
GB/s. The axis that matters is **bytes in flight per CTA** (Little's law), not
occupancy: holding all else fixed, 16x64x32 -> 16x64x64 went 97.9 -> 76.8us and
64x64x32 at 2 -> 3 stages went 108.3 -> 72.3us.

The best configuration was not reachable at all: `cta_k=64` variants at small
`cta_m` were absent from the bucket, and `cta_m <= 32` was hard-capped at 2
stages. With both fixed the autotuner picks 16x64x64/32x64x64 at 3 stages,
worth ~1.05x in `stream` and up to 1.46x in `resident`. Square shapes 512
through 16384 pick bit-identical tiles; 256 improved (5.85 -> 5.13us).

## GEMV path for M < 16: measured, and NOT worth it in the regime that matters

A hand-written split-K GEMV probe (no tensor cores, `ld.global.nc.v4.b32`
straight to registers, 40 registers, no shared memory) was measured against
this kernel at M=1 before committing to the codegen:

| M=1, N=K=4096 | stream (DRAM-bound) | resident (L2-bound) |
|---|---|---|
| GEMV probe (incl. required memset) | 59.56us / 564 GB/s | **18.90us / 1777 GB/s** |
| this tensor-core kernel | 59.65us / 563 GB/s | 51.01us / 658 GB/s |
| cuBLAS | 55.53us / 605 GB/s | 28.71us / 1169 GB/s |

In the **streaming regime - the one real autoregressive decode runs in** - the
GEMV is a wash (59.56 vs 59.65us). The 64x M-dim tensor-core padding is
*compute* waste, and compute waste is free when the kernel is pinned at the
DRAM roofline. The motivating argument for a GEMV path ("less padding, more
memory parallelism") does not survive measurement.

It is a 2.7x win over this kernel (and 1.52x over cuBLAS) only when the weight
matrix is L2-resident, where skipping the global -> smem -> `ldmatrix`
round-trip and loading B straight to registers is what pays. That regime is
real (small models, low-bit weights, hot MoE experts, batched reuse) but is
narrower than streaming decode.

Blocking design question before wiring it in: split-K is unavoidable at these
shapes (N=4096 yields only ~4-16 CTAs without it, far too few to fill 66 SMs),
and split-K means the kernel **accumulates** into C rather than overwriting it.
Every existing caller assumes overwrite semantics. Resolving that needs either
a caller-zeroed-C contract change or a workspace + reduction ABI - a real
interface decision, not an implementation detail.
