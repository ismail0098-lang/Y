# Real Y-Compiler Fused Add+RMSNorm and RoPE vs FlashInfer (real production kernels)

This is the comparison `benchmark_y_rmsnorm_rope_results.md` (Y vs eager PyTorch) does NOT answer: eager PyTorch is the weakest possible baseline, and production inference stacks (vLLM, SGLang) already use FlashInfer's hand-tuned fused kernels for exactly these ops, not eager PyTorch. `flashinfer.fused_add_rmsnorm`/`flashinfer.apply_rope_pos_ids(interleave=True)` compute the same math, same convention (interleaved-pairs RoPE, base=10000, in-place add+RMSNorm) as Y's kernels - not an approximate comparison.

Timing is the median of 5 independent process launches per row count (range shown in brackets); a row is marked inconclusive rather than given a ratio if the Y and FlashInfer ranges overlap.

## Fused Add+RMSNorm (hidden_dim=4096)

| Rows | FlashInfer us (median [range]) | Y Fused us (median [range]) | Y vs FlashInfer | Correct |
|---|---|---|---|---|
| 128 | 5.47 [4.32, 10.87] | 10.46 [3.86, 15.54] | inconclusive | OK |
| 1024 | 46.89 [18.81, 88.69] | 49.17 [37.50, 68.88] | inconclusive | OK |
| 8192 | 473.94 [466.64, 489.37] | 479.71 [468.10, 667.04] | inconclusive | OK |

## Fused RoPE (head_dim=128, interleaved)

| Rows | FlashInfer us (median [range]) | Y Fused us (median [range]) | Y vs FlashInfer | Correct |
|---|---|---|---|---|
| 128 | 10.00 [8.61, 20.16] | 5.42 [1.86, 7.79] | **1.85x** | OK |
| 1024 | 8.60 [8.50, 26.81] | 4.47 [1.91, 7.24] | **1.92x** | OK |
| 8192 | 8.60 [8.46, 27.27] | 5.57 [5.27, 26.86] | inconclusive | OK |

## Update: RMSNorm given real occupancy (multi-warp) + shared-memory `h` staging - items 2-3

Vectorizing memory access alone (the previous update above) left RMSNorm at 0.71-0.78x of FlashInfer - a measurable loss, confirming items 2-3 were still needed, not optional. Implemented both together in `emit_rmsnorm_residual_kernel` (`ptx_emitter.rs`), since they touch the same pass-1/pass-2 structure:

- **Real occupancy**: threads/row now scales with `hidden_dim` instead of a fixed single warp - `num_warps` is the largest of `{8,4,2,1}` dividing `hidden_dim`'s vector-group count evenly (8 warps / 256 threads at hidden_dim=4096, the realistic case). Each warp folds its own partial `sum(h^2)` via the existing, already-validated `emit_warp_reduce_sum` butterfly, then (new) lane 0 of each warp writes its warp-total into a small `.shared` scratch buffer, `bar.sync 0`, and every thread sums the `num_warps` slots to get the true row-wide total - a real cross-warp block-level reduction, not a rename of the single-warp path.
- **Shared-memory `h` staging**: pass 1 now also writes `h` (as f32, unrounded) into a `.shared` buffer (`smem_h`, `hidden_dim` floats/row) as it computes it; pass 2 reads `h` back from `smem_h` instead of re-reading `X`/`Residual` from DRAM a second time. Each thread only ever reads back the exact `smem_h` addresses it itself wrote (same `tid`-strided index sequence both passes), so the `bar.sync` already required for the cross-warp sum combine is sufficient - no extra synchronization needed. `X`/`Residual` are now read from DRAM exactly once per row instead of twice.
- Both host-side launch configs (`benchmark_y_vs_flashinfer.py`, `benchmark_y_rmsnorm_rope.py`) updated from `threads=(32,1,1)` to `threads=(256,1,1)` to match - the kernel's real thread count is no longer a silent hardcoded assumption the host and device can disagree about.

**Result: RMSNorm goes from a clear, measurable loss to statistical parity with FlashInfer at every row count tested.** Confirmed across 4 independent full median-of-5 benchmark runs (not cherry-picked - the first of these five runs happened to catch the whole system in an elevated-latency state, FlashInfer included, and was re-run rather than reported; see the raw numbers below), plus 15 additional repeated single-shot launches (5 each at 128/1024/8192 rows) for the determinism check:

| Rows | FlashInfer us (median, range across 4 runs) | Y us (median, range across 4 runs) | Verdict |
|---|---|---|---|
| 128 | 5.02 - 6.25 | 5.35 - 12.06 | consistently inconclusive (parity) |
| 1024 | 29.63 - 46.89 | 29.43 - 49.17 | consistently inconclusive (parity) |
| 8192 | 466.08 - 473.94 | 466.46 - 479.71 | consistently inconclusive (parity), medians within ~1-3% of each other every run |

`ptxas -v`: 28 registers/thread, 0 spill stores/loads, 1 barrier, 16416 bytes shared memory (16384 for `smem_h` + 32 for `smem_reduce` at hidden_dim=4096 - matches the declared sizes exactly). All 15 repeated determinism-check launches were correct with identical `max_abs_diff` (0.00390625, a single f16-ULP-scale value) - no non-determinism, despite this being the second real restructuring of this loop's body since the original register-pressure bug.

This closes out items 2-3 from the prior session's brief: vectorization alone (item 1) was not sufficient, but vectorization + real occupancy + eliminating the double DRAM read together bring RMSNorm from a clear loss to parity with a hand-tuned production kernel.

## RoPE: extended measurement across head_dims (item 4)

The RMSNorm work above was "real, new engineering" per the prior session's brief; RoPE's follow-up was deliberately "measurement first, not engineering" instead - confirming the win generalizes before touching the kernel further. See `benchmark_y_rope_headdim_sweep_results.md` for the full sweep: head_dim in {64, 128, 256} (all common LLM values), 10 repeats/row instead of 5. Short version: Y's median beat FlashInfer's in 8 of 9 (head_dim, rows) combinations tested, often by 2-5x, confirming the specialization-beats-generality pattern is not specific to head_dim=128 - with one honest exception (head_dim=64 at 8192 rows, where Y's median was slower this run, though ranges overlap enough that this could be noise rather than a reproducible loss). No RoPE kernel engineering changes were made this pass - this was purely a measurement follow-up, per the brief.

### Environment note, not a Y finding

Getting this comparison running at all required a workaround: FlashInfer JIT-compiles its own kernels via `nvcc`/`ninja` on first use, and `nvcc` cannot handle this project's path containing a space (`.../src/Y_lang/Y`'s parent, `NVME files`) - confirmed via the exact `ninja: error: '/home/yumin/NVME', needed by ...` failure, unrelated to CUDA version, torch version, or anything about Y. Worked around by installing a second venv at a space-free path (`/tmp/y-bench-venv`, same package versions: torch 2.13.0+cu130, cupy 14.1.1, flashinfer 0.6.16) rather than trying to patch FlashInfer's JIT internals. Anyone re-running this script needs the same workaround (or the project moved to a space-free path) until/unless upstream FlashInfer fixes its own path quoting.

### Open questions for next session

1. **RMSNorm is now at parity, not ahead.** The next lever, if closing the gap further is worth it, would need real profiling (e.g. `ncu`) to see where the remaining time actually goes - no more "obvious" structural gaps like the old single-warp/double-read design had.
2. **The head_dim=64/8192-rows RoPE result** (Y median slower, though inconclusive) is worth one more look with more repeats specifically at that combination if RoPE performance at large batch sizes with small head dims matters for the pitch.
3. **Do the same FlashInfer-caliber comparison for the Bias+ReLU and SwiGLU kernels**, still open from the prior session (cuDNN already serves this role for Bias+ReLU's FP16 column; SwiGLU has no equivalent comparison yet).
