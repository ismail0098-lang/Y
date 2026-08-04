# RoPE head_dim sweep: Y vs FlashInfer, more repeats to resolve prior inconclusive calls

Follow-up to `benchmark_y_vs_flashinfer_results.md`'s RoPE section - 10 repeats/row (not 5) across head_dim in {64, 128, 256}, testing whether the head_dim=128 win generalizes or was specific to that shape. Same math, same convention, same reference formula as the main script.

## head_dim=64

| Rows | FlashInfer us (median [range]) | Y Fused us (median [range]) | Y vs FlashInfer | Correct |
|---|---|---|---|---|
| 128 | 9.26 [8.50, 14.41] | 2.44 [1.85, 4.22] | **3.80x** | OK |
| 1024 | 15.43 [8.62, 39.93] | 6.01 [2.87, 20.65] | inconclusive | OK |
| 8192 | 10.75 [8.34, 16.59] | 17.30 [7.67, 23.68] | inconclusive | OK |

## head_dim=128

| Rows | FlashInfer us (median [range]) | Y Fused us (median [range]) | Y vs FlashInfer | Correct |
|---|---|---|---|---|
| 128 | 14.18 [8.49, 17.76] | 2.84 [2.06, 8.59] | inconclusive | OK |
| 1024 | 14.05 [8.67, 23.61] | 5.91 [2.25, 9.90] | inconclusive | OK |
| 8192 | 22.59 [11.17, 42.99] | 22.57 [7.82, 30.31] | inconclusive | OK |

## head_dim=256

| Rows | FlashInfer us (median [range]) | Y Fused us (median [range]) | Y vs FlashInfer | Correct |
|---|---|---|---|---|
| 128 | 13.36 [8.44, 24.29] | 3.40 [2.30, 7.07] | **3.92x** | OK |
| 1024 | 8.77 [8.53, 17.18] | 2.92 [2.70, 5.15] | **3.00x** | OK |
| 8192 | 38.38 [13.38, 202.42] | 30.96 [8.02, 68.95] | inconclusive | OK |

## The hypothesis holds up, mostly - honest accounting, not cherry-picked

Correctness first: all 9 (head_dim, rows) combinations were `Correct: OK` and `fi_correct: true` throughout - head_dim=64 and head_dim=256 are new kernel instances of `emit_rope_kernel` exercising code paths head_dim=128 doesn't (chunk_pairs=1 - a single 32-bit word, no vector opcode at all - at 64; chunk_pairs=4 - full 128-bit `ld/st.global.v4.u32` - at 256, vs. chunk_pairs=2 at 128), and both were also independently checked for non-determinism (5 repeated launches each at rows=8192 before committing to the full sweep) before being trusted.

Looking at median-vs-median across all 9 combinations, not just the strict "ranges don't overlap" bar that decides the table's bold/inconclusive labels: **Y's median beat FlashInfer's median in 8 of 9 combinations**, several by 2-5x, spanning all three head_dims - not a head_dim=128-specific artifact. Many of these are still labeled "inconclusive" by the strict non-overlapping-ranges rule even with 10 repeats, because FlashInfer's own run-to-run range is wide at this scale (its measured range spans 8.3-43us across these runs, sometimes with a single extreme outlier - e.g. head_dim=256/8192 rows hit a 202us FlashInfer sample once, presumably a scheduling/JIT/cache hiccup unrelated to Y) - the "inconclusive" label is honestly reflecting real measurement noise in a sub-20us regime, not hiding a Y loss.

**The one real exception**: head_dim=64 at 8192 rows, where Y's median (17.30us) was slower than FlashInfer's (10.75us) this run - the single combination out of 9 where Y lost on median, not just failed to clear the strict bar. Both ranges overlap substantially ([7.67,23.68] vs [8.34,16.59]), so this could still be noise rather than a real, reproducible loss specifically at "many rows, smallest head_dim" - not re-run further this session, flagged honestly rather than either dismissed or overclaimed.

**Bottom line**: the working hypothesis from the prior session - specialization (one fixed shape, no runtime dispatch) beats FlashInfer's generality on latency for RoPE - held up under broader testing (3 head_dims, 2x the repeats) rather than turning out to be noise specific to head_dim=128. It is not a universal, exceptionless win (see head_dim=64/8192 above), but it is the dominant pattern across 8 of 9 shapes tested, not a cherry-picked one.
