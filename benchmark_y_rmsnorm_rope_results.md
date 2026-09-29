# Real Y-Compiler Fused Add+RMSNorm and RoPE vs Eager PyTorch

Every number below is measured from the real Y CLI's `--emit-ptx` output (`ptx_emitter::emit_rmsnorm_residual_kernel` / `emit_rope_kernel`, dispatched by kernel name + param shape - no `@tile` - see `tests/rmsnorm_residual_4096.ysu` / `tests/rope_128.ysu`), loaded and run as-is (`cp.RawModule(path=...)`).

Timing is the median of 5 independent process launches per row count (range shown in brackets); a row is marked inconclusive rather than given a speedup number if the Y and eager-PyTorch ranges overlap.

## Fused Add+RMSNorm (hidden_dim=4096)

`h = X + Residual`; `Out = RMSNorm(h) * Weight`; writes both `Out` and `h` (as `NewResidual`) from one launch, one warp (32 threads) per row. Eager baseline: real ATen ops (add, `.pow(2).mean(-1)`, `rsqrt`, two multiplies), no fusion - the way a naive PyTorch RMSNorm module is actually written.

| Rows | Eager PyTorch us (median [range]) | Y Fused us (median [range]) | Y vs Eager | Correct (max abs diff) |
|---|---|---|---|---|
| 128 | 54.78 [52.95, 80.08] | 46.13 [42.71, 64.96] | inconclusive | OK (0.0039) |
| 1024 | 159.42 [158.74, 364.12] | 30.12 [28.66, 78.93] | **5.29x** | OK (0.0039) |
| 8192 | 3104.77 [3082.96, 3170.46] | 627.89 [611.88, 630.90] | **4.94x** | OK (0.0039) |

## Fused RoPE (head_dim=128, interleaved-pairs convention)

Rotates each token's query/key vector by a position-dependent angle computed entirely on-device (`ex2.approx.f32`/`sin.approx.f32`/`cos.approx.f32`, no precomputed cos/sin table input). Eager baseline builds the same table PyTorch-side every call (the realistic unfused cost) and applies the same interleaved-pairs rotation.

| Rows | Eager PyTorch us (median [range]) | Y Fused us (median [range]) | Y vs Eager | Correct (max abs diff) |
|---|---|---|---|---|
| 128 | 46.51 [36.84, 69.24] | 2.67 [1.87, 3.33] | **17.40x** | OK (0.0020) |
| 1024 | 68.77 [40.27, 105.68] | 4.75 [4.03, 6.31] | **14.49x** | OK (0.0020) |
| 8192 | 133.78 [96.90, 201.10] | 12.92 [9.46, 17.58] | **10.35x** | OK (0.0020) |

## This session: both kernels built from scratch (no prior code existed), and a real, real-hardware-only bug found and fixed along the way

Neither kernel had any prior implementation in this codebase - both `emit_rmsnorm_residual_kernel` and `emit_rope_kernel`, plus their dispatch (`rmsnorm_residual_operands`/`rope_operands`, matched by kernel name + param shape rather than `@tile` - these kernels have no compile-time M, row count is a pure runtime grid dimension, so the only genuinely compile-time-needed integer, `hidden_dim`/`head_dim`, is parsed from the kernel name), are new this session. Both use one warp (32 threads) per row - no CTA tiling, no shared memory, no cross-CTA synchronization - deliberately the simplest structure that could work for a memory-bound, per-row elementwise/reduction op.

**Every non-trivial PTX primitive was validated standalone on real sm_89 hardware before being wired into either kernel** (matching this project's established discipline elsewhere): `rsqrt.approx.f32` (max relative error 1.2e-7 vs Python `math.sqrt` over 2000 samples), `sin.approx.f32`/`cos.approx.f32` (max absolute error ~6e-6 vs `math.sin`/`math.cos` over 2000 angles spanning +/-50 radians), the 5-step `shfl.sync.bfly.b32` warp-reduction (reproduced a reference 32-value sum exactly to float rounding, real 32-thread launch), and scalar F16<->F32 conversion (`ld.global.u16`+`cvt.f32.f16`+`cvt.rn.f16.f32`+`st.global.u16` - zero prior precedent in this codebase, which only ever moves F16 as raw bits through `ldmatrix`/`wmma`/`mma.sync`, never converts it to a scalar F32 register - bit-exact over 200 random samples).

**A real bug was found in the first working version of `emit_rmsnorm_residual_kernel` and is worth recording in detail.** The first implementation fully unrolled both passes over `hidden_dim` (128 iterations/thread at `hidden_dim=4096`, fresh SSA registers per iteration - 202 registers/thread per `ptxas -v`, 0 reported spills). It had two issues, found in this order:

1. **A real indexing bug**: `Weight` (which should be a single `[hidden_dim]` vector broadcast across every row - the standard RMSNorm gamma contract) was indexed with the same row-inclusive byte offset as `X`/`Residual`/`Out`, silently reading past the end of the `[hidden_dim]`-sized `Weight` buffer for every row after the first. Caught by code review before ever running on hardware, fixed by giving `Weight` its own `idx`-only (not `elem = row_offset + idx`) byte offset.
2. **A non-deterministic wrong-output bug, found only by running on real hardware**: with a *fixed* random seed (identical inputs every run), repeated process launches of the *same* compiled kernel produced a small number of wrong output elements (`Out` only - `NewResidual`, computed without depending on the cross-lane warp-shuffle reduction, was correct every single time) at a *different* `(row, col)` location each run, with an error magnitude too large to be rounding (single elements off by several units, occasionally exactly `0.0`). Bisecting (re-testing the identical kernel logic at `hidden_dim=128`, 34 registers/thread, 4 unrolled iterations - deterministically correct every time, 4/4 runs) isolated the difference to something specific to the much larger, fully-unrolled `hidden_dim=4096` instruction sequence, not the warp-reduction logic itself (independently validated standalone, see above, and structurally identical at both sizes). SASS inspection (`cuobjdump --dump-sass`) of the `hidden_dim=4096` binary showed a completely ordinary, correct-looking `SHFL.BFLY`/`FADD` reduction chain - nothing visibly wrong. **Root cause not fully identified** - rather than continue chasing it blind, both passes were rewritten as real runtime loops (loop-carried `idx`/`running_sum` registers, branching back, matching the style `emit_gemm_tile_load`'s loop already uses elsewhere in this codebase) instead of fully unrolled, which is an independently-justified design improvement regardless (202 -> 24 registers/thread, dramatically better occupancy for a 1-warp/CTA memory-bound kernel) and empirically eliminated the bug (5/5 repeated runs at `hidden_dim=4096` now deterministically correct, matching the `hidden_dim=128` case). Recorded here in full rather than silently "fixed" without disclosure, since the underlying mechanism (something about very large fully-unrolled register-heavy PTX interacting badly with `shfl.sync` on this toolchain/hardware) is not understood and could plausibly recur elsewhere in this codebase's other heavily-unrolled kernels.

### Open questions for next session

1. **Root-cause the register-pressure/unroll-size correctness issue** above properly (not just work around it) - would need `ncu`/hardware perf-counter access this sandboxed session doesn't have (see `benchmark_y_tensor_core_gemm_results.md`'s "ncu access note"), or a controlled bisection sweep (try hidden_dim=512/1024/2048 fully-unrolled to find the exact threshold) to characterize it precisely. Worth checking whether any of this project's other kernels are at risk (the plain/Bias+ReLU/SwiGLU GEMM kernels are all real *loops* already, not full unrolls of a runtime-sized dimension, so likely not exposed - but not confirmed).
2. This session's `hidden_dim=4096`/`head_dim=128` are single fixed configurations; sweeping other realistic LLM sizes (hidden_dim 2048-8192, head_dim 64/256) would round out the picture, though the underlying kernel structure (one warp/row, real loop) is generic over both already.
