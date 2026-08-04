# Real Y-Compiler Tensor Core GEMM vs cuBLAS

Every number below is measured from `target/release/Y tests/gemm_f16_<N>.ysu --emit-ptx`'s actual output (`ptx_emitter::emit_tensor_core_gemm_kernel`, dispatched via kernel-level `@tile(M, N, K)`, tile/warp/stage selection from `Autotuner::autotune`) - not the hand-written CUDA C++ reference benchmark_y_vs_cudnn_cublas.py's Suite 1 measures.

Timing is the median of 5 independent process launches per size (range shown in brackets); a row is marked inconclusive rather than given a speedup number if the Y and cuBLAS ranges overlap - see the module docstring in tests/benchmark_y_tensor_core_gemm.py for why a single-sample number isn't trustworthy here.

## Timing methodology (revised)

Each measuring process now ramps the GPU clock to steady state for 3s before timing anything, then A/B-interleaves Y and cuBLAS over 5 rounds so both are measured at the same clock. The previous discipline (short warmup; all of Y timed, then all of cuBLAS) was contaminated twice over: the SM clock on this dev GPU idles at ~210 MHz and needs ~3s of load to reach ~2670 MHz (12.7x, and clocks cannot be locked on this box), and timing Y first meant cuBLAS always inherited the hotter clock - a systematic bias, not noise. It reported this same 256 kernel as `45.56us [7.12, 63.03]`, a 9x spread.

**What the harness fix changed, and what it did not.** The large sizes (2048 and up) were unaffected - they re-measured at the same 0.75x/0.76x/0.77x reported before it - because a single iteration there already runs for milliseconds and ramps the clock itself inside the timed loop. Only the small, microsecond-scale sizes were ever vulnerable, and those corrected **downward** (256/512/1024 came in at 0.71x/0.59x/0.68x). Run-to-run ranges are now within a few percent at every size.

## Autotuner: occupancy over tile size

The figures in the table above are *after* a subsequent `score_candidate` rework, and are substantially better than that 0.75x-0.77x baseline at every size from 1024 up.

`ncu` (once GPU performance counters were enabled) showed the old 128x256x32 4x4 pick was not memory-bound in any direction - DRAM at 16.6% of peak, L1 at 20.4%, a shared-memory stall ratio of 0.74 - so the scorer's `compute_intensity` term (reuse per byte staged) was optimising for a wall the kernel never hits, and its `1.0 / ctas_per_sm` factor had no physical basis at all, since total work is invariant to tile choice. Between them they rated the measured-best config 3741 against 11809 for one 19% slower.

The real gap against cuBLAS was occupancy and barrier cost. cuBLAS runs `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn`: a 128x128 tile at 128 threads, 234 registers/thread, 2 blocks/SM. Y ran a bigger tile at 512 threads, 1 block/SM, with a barrier-stall ratio of 6.27 against cuBLAS's 1.27. Scoring compute-bound shapes on predicted utilisation instead - per-warp MMA parallelism, resident CTAs per SM (gated on the grid actually supplying them), and warps per CTA - moves the picks to 128x128x32 at 2 stages, which fits two CTAs per SM. Measured effect at 4096: tensor-pipe utilisation 37.38% -> 44.61%, barrier stalls 6.27 -> 0.37 (now below cuBLAS's), registers/thread 122 -> 194.

Skinny/decode shapes and small squares (min dimension below 1024) keep the older reuse-based heuristic: the utilisation model assumes steady-state throughput over many CTA waves, which neither satisfies. Every pick here was re-benchmarked per size rather than trusted - forcing the utilisation model onto 256 picked a tile that measured 6.08us against 5.11us for the legacy pick.

| Matrix (M=N=K) | CTA Tile | cuBLAS us (median [range]) | Y us (median [range]) | Y vs cuBLAS | Correct |
|---|---|---|---|---|---|
| 2048x2048x2048 | 128x128x32 | 204.95 [202.99, 221.63] | 240.20 [233.06, 248.01] | **0.85x** | OK |
| 4096x4096x4096 | 128x128x32 | 1685.85 [1685.66, 1695.28] | 1847.14 [1838.81, 1861.89] | **0.91x** | OK |
| 8192x8192x8192 | 128x128x32 | 13521.14 [13400.99, 13645.06] | 14376.19 [14288.46, 14430.81] | **0.94x** | OK |
| 16384x16384x16384 | 128x128x32 | 104112.50 [103738.37, 105080.50] | 115368.49 [114876.41, 115962.53] | **0.90x** | OK |
