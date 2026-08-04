# FP8 (e4m3) Quantized GEMM: A Real, Working, Reachable Kernel (Not Yet Competitive)

This is a follow-up to `investigation_fp8_int8_quantization_findings.md` (prior session, "No Working Benchmark" - every FP8 code path was unreachable dead code or a confirmed Hopper-only hardware dead end). This session did the four things that doc's "What a real next-session attempt would need, in order" section listed, in that order, and each was independently verified before the next started. The result: **a genuinely correct, genuinely reachable FP8 GEMM kernel now exists in this codebase** - a real `.ysu` file compiles through the real CLI to a real, `ptxas`-clean, hardware-validated PTX kernel that matches `torch._scaled_mm` to within 0.02%-0.13% relative L2 error. It is **not performance-competitive** with cuBLASLt (2.7x-70x slower, gap widening with size) - by deliberate, disclosed design, not by accident. Both facts are reported here plainly.

## 1. PTX ISA version bump (`ptx_version_for_sm`)

`sm_89`'s arm in `src/ptx_emitter.rs` changed from `.version 7.8` to `.version 8.4` - FP8 `mma.sync` requires it (confirmed empirically last session). Small, contained, low-risk as predicted: `cargo test` was 71/71 before and after, still 73/73 now (2 new FP8 tests added later, see below). No other kernel's emitted PTX text uses a `.version`-sensitive feature that this bump could regress - `sm_89`'s only consumers are the F16 GEMM/RMSNorm/RoPE/SwiGLU kernels and this new FP8 one, none of which touch anything gated between 7.8 and 8.4.

## 2. Fragment layout for `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, derived and hardware-validated

Never derived or validated in this codebase before (unlike `m16n8k16` f16 - see `benchmark_y_tensor_core_gemm_results.md`). Derived directly from a saved copy of the PTX ISA 8.5 PDF text (`/tmp/ptx_isa.txt`, the same source last session's WGMMA-dead-end finding came from), not memory or paraphrase:

- **Matrix A** (16x32, `.row`): PTX ISA 9.7.13.4.10's `.s8`/`.u8`/`.e4m3`/`.e5m2` bullet (one shared formula) - 4 `.b32` registers, `groupID=lane>>2, tig=lane&3`; `row=groupID` (or `+8`), `col=tig*4+[0..3]` (or `+16` for the K 16-31 half).
- **Matrix B** (32x8, `.col`): same section's B bullet - 2 `.b32` registers; `row=tig*4+[0..3]` (or `+16`), `col=groupID`.
- **Accumulator C/D** (16x8, f32): 9.7.13.4.9's formula (m16n8k16 integer path), reused verbatim - confirmed by direct text comparison that the m16n8k16-float, m16n8k16-integer, and m16n8k32 sections all state the identical accumulator formula (it depends only on M=16/N=8, not K, and 9.7.13.4.10 doesn't re-derive it, just references the same figure by number).
- **A real, non-obvious discovery along the way**: PTX ISA 8.5's `ldmatrix` instruction (9.7.13.4.15) supports `.type = {.b16}` **only** - no 8-bit variant exists at all, on any architecture, confirmed by reading the syntax block directly. This is why the F16 kernel's `ldmatrix.x4`/`ldmatrix.x2.trans` pattern couldn't just be reused for FP8 - fragment loads here are hand-computed `ld.shared` addresses instead. A is loaded with one `ld.shared.b32` per register (its 4 elements/register are contiguous columns in a row-major smem layout); B needs 4x `ld.shared.u8` + shift/or packing per register (its 4 elements/register are 4 *rows* at a fixed column - not contiguous, since B's smem layout matches global B's row-major/N-contiguous layout directly, with no `.trans`-style load to reconcile the mismatch the way `ldmatrix.x2.trans` did for f16).

**Standalone validation** (`validate_fp8_mma.py`, project scratchpad, same discipline as the existing `validate_mma.py`): a single-warp hand-written `.ptx` file, assembled with `ptxas -arch=sm_89` (assembles clean at `.version 8.4`), loaded via `cp.RawModule(path=...)` (raw PTX through the CUDA driver's JIT, not NVRTC), run on real hardware. Three independent checks, all passing: (1) raw A-fragment register dump vs the ISA formula, (2) raw B-fragment register dump vs the ISA formula, (3) `mma.sync` output vs an exact-integer CPU reference (small integers 0-15, exactly representable in e4m3 - confirmed by round-tripping through `torch.float8_e4m3fn`). Stable across 2 different seeds and repeated runs. One real bug caught and fixed during this: the first comparison run failed because the harness compared raw e4m3 *byte encodings* against *decoded integer values* - a test-harness bug, not a kernel bug, but exactly the kind of thing this discipline exists to catch before it reaches real emitter code.

**Quantization instruction** (`cvt.rn.satfinite.e4m3x2.f32`) was validated the same way, separately (`validate_fp8_quantize.py`): 32 test values (zero, positive/negative integers, fractions needing round-to-nearest, values beyond e4m3's max normal 448 needing `.satfinite` saturation), checked bit-for-bit against `torch.float8_e4m3fn`'s own conversion - both byte order (`cvt`'s "input a -> upper 8 bits, input b -> lower 8 bits" convention) and numeric encoding matched exactly.

## 3. The real kernel: `emit_fp8_gemm_kernel` (`src/ptx_emitter.rs`)

Dispatched from a validated 5-parameter `@tile(M, N, K)` shape - `tile_gemm_fp8_operands`, mirroring the existing `tile_gemm_operands`'s re-validation discipline (never trusts `type_checker` alone, since `PtxEmitter` can be driven directly with no type-checking pass at all, as this file's own unit tests do). `type_checker::verify_tile_gemm_kernel` was extended with a matching branch so the two can never disagree about which shape a kernel is - tested directly (a malformed 5-param kernel is rejected with a clear compile error, see below).

**Kernel contract**: `kernel(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>)`. A/B arrive as plain F32 and are quantized to e4m3 **on the fly**, fused into the tile-staging step - no separate quantize kernel or pass. `quantization_pass.rs`'s own `emit_fp32_to_fp8` was checked first and found unusable: it's a placeholder that zeros a register with a `// placeholder for cvt.rn.satfinite.e4m3x4` comment and never actually converts anything, and it's structurally coupled to the RT/Tensor coprocessor's `coprocessor_smem` symbol, which the sibling investigation doc already found never produces valid PTX. Writing a real, working quantization step directly in the new kernel (using the standalone-validated `cvt.rn.satfinite.e4m3x2.f32`) was more direct than trying to rehabilitate that module.

**Deliberate, disclosed scope for this first pass** (see `benchmark_y_tensor_core_gemm_results.md` for how the F16 kernel's own history went from correctness-first `wmma` to optimized `ldmatrix`/`mma.sync` + `cp.async` pipelining over *multiple* sessions - this is that same first step for FP8, not a finished optimization target):
- CTA tile fixed at exactly one `mma` instruction's shape (M=16, N=8, K=32), one warp (32 threads) per CTA. Grid: `(ceil(M/16), ceil(N/8), 1)`.
- No `cp.async` pipelining, no grid swizzle, no Autotuner integration.
- K must be an exact multiple of 32 (`debug_assert`). M and N may be **any** positive value - both the quantize/stage step (predicated loads, zero-filled) and the epilogue (predicated per-element stores, cheap at 16x8/CTA) are boundary-masked, not just exact-multiple shapes.
- `scale_a`/`scale_b` are per-tensor scalars the caller computes (typically `amax/448.0`) and passes as plain kernel params - not computed on-device.
- Static (non-`extern`) `.shared` arrays - the fixed 768B footprint is far under `ptxas`'s 48KB static cap, so the F16 kernel's dynamic-shared machinery (needed for its much larger, autotuned tiles) isn't needed here.

The loop-carried-accumulator trap flagged in `feedback_gemm_kernel_validation.md` was specifically guarded against: the 4 f32 accumulator registers are allocated once before the K-loop and the `mma.sync` instruction writes back into those same registers every dynamic iteration (`{d0..d3}, {a}, {b}, {d0..d3}`), never a fresh register per iteration.

## 4. Reachable from the real CLI - verified, not asserted

`tests/gemm_fp8_256.ysu` / `_512` / `_1024` / `_4096`, each a real `.ysu` source file:
```
@tile(256, 256, 256)
kernel gemm_fp8_256(A: GlobalMemory<F32>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {
}
```
`cargo run --release --bin Y -- tests/gemm_fp8_256.ysu --emit-ptx` (the same invocation any user would run, no special flags) writes `tests/gemm_fp8_256.ptx`. That file - the real compiler's real output, not a hand-written stand-in - assembles clean with `ptxas -arch=sm_89`: **48 registers/thread, 0 spill stores/loads, 768 bytes shared memory** (exactly 512+256, the two smem tile declarations). A malformed 5-param kernel (wrong element type on A) is correctly rejected by `type_checker` with a specific error message, confirmed by running it through the real CLI. Two new `cargo test` regression guards were added (`test_fp8_gemm_kernel_reachable_and_uses_mma_sync_m16n8k32`, `test_fp8_gemm_operands_rejects_wrong_element_type`, both going through the real lexer/parser/`emit_program` path, not calling `emit_fp8_gemm_kernel` directly) - `cargo test`: **73 passed, 0 failed** (up from the 71-test baseline). These are regression guards, not the primary verification; the primary verification is the real-hardware runs below.

## 5. Correctness on real hardware, vs `torch._scaled_mm`

`torch._scaled_mm`'s actual signature was confirmed via `torch.ops.aten._scaled_mm.default._schema` (not assumed, as the task brief warned against): `(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False)`. Two real constraints found empirically, neither guessed: **`mat2` (B) must be column-major** (row-major B fails outright with a `CUBLAS_STATUS_NOT_SUPPORTED` heuristic error), and **`mat2`'s dimensions must be divisible by 16** (a cuBLASLt constraint on the *reference op*, not on Y's kernel, which has no such restriction). `out_dtype=torch.float32` is needed for a directly comparable f32 result (the default output dtype is `float8_e4m3fn`).

- **256/512/1024/4096 (M=N=K), seed 0**: relative L2 error 0.018% / 0.032% / 0.052% / 0.131% - growing smoothly with K, all clearly correct.
- **5 seeds at 256x256x256**: all pass, consistent ~0.002-0.003 relative error range.
- **5 repeated identical-seed launches at 256x256x256, separate process each time**: bit-for-bit identical output every run - **not** the non-deterministic-wrong-output bug class `feedback_gemm_kernel_validation.md` warns about (checked explicitly, per that memory's own guidance to verify before assuming either bug class).
- **Ragged shapes** (250x100x256, 33x17x64, 15x7x32, 1x1x32 - none a multiple of 16 or 8): `torch._scaled_mm` itself rejects all of these (the divisibility-by-16 constraint above). Used an independent manual dequant-then-fp32-matmul reference instead (`(A_e4m3.float()*scale_a) @ (B_e4m3.float()*scale_b)`, using the same e4m3-quantized values - a fair, independently-computed check since quantization was already validated bit-exact against `torch.float8_e4m3fn`, not a tautology against Y's own quantization). All four passed, confirming the boundary-masking logic works for genuinely arbitrary shapes - a capability broader than what the reference op itself accepts.
- **K-sweep at M=N=16 (single CTA, isolating accumulation-length from grid-size effects)**: error is **exactly zero** at K=32 (a single `mma` call, no accumulation loop at all), then grows smoothly - 0.07% at K=128, 0.26% at K=512, 0.85% at K=4096. This rules out a threshold/overflow/indexing bug at large sizes; the growth is accumulation-order sensitivity, an expected property of reduced-precision numerics, not a logic error.
- **Multiple seeds + repeated launches at 4096x4096x4096** (the largest, most demanding size): consistently ~0.131% relative L2 error, deterministic across launches.

**A real methodology lesson, worth recording on its own**: a first-pass per-element `rtol=0.03/atol=0.5` correctness check *falsely* flagged the 4096x4096x4096 case as wrong (only 99.83% of ~16.8M compared elements within tolerance, reported as `[FAIL]`). Root-caused before writing anything up (per this project's own debugging discipline, not dismissed): the underlying per-element error distribution is small and well-behaved (mean absolute difference ~0.5 against a ~459 mean output magnitude), but **order statistics over millions of compared elements push the observed maximum deviation up** even when nothing is wrong - the 99.99th-percentile deviation was still under 2% of the mean output magnitude. Relative L2 error (`||Y-ref||/||ref||`) - the standard metric for validating reduced-precision GEMM kernels in the ML-quantization literature - was 0.13% at this size, unambiguously correct. Both `tests/test_fp8_gemm_correctness.py` and `tests/benchmark_y_fp8_gemm.py` were fixed to gate on relative L2 (reporting the per-element figure alongside, for context, not as the pass/fail bar).

## 6. Performance on real hardware, vs `torch._scaled_mm` - not competitive, honestly

| Matrix (M=N=K) | Y us (median [range]) | torch._scaled_mm us (median [range]) | Y vs torch |
|---|---|---|---|
| 256x256x256 | 16.47 [16.44, 18.23] | 6.16 [2.87, 6.25] | **0.374x** |
| 512x512x512 | 143.00 [132.45, 155.39] | 5.99 [4.61, 11.88] | **0.042x** |
| 1024x1024x1024 | 996.56 [975.62, 1072.52] | 18.28 [16.74, 22.68] | **0.018x** |
| 4096x4096x4096 | 66074.73 [66004.58, 66248.99] | 896.10 [872.45, 917.35] | **0.014x** |

(Full table with methodology notes: `benchmark_y_fp8_gemm_results.md`. Timing via CUDA events, median of 7 alternating rounds - 20 launches/round - after 15 rounds of shared alternating warmup, matching the GPU-clock-ramp-bias fix documented in `feedback_sass_register_bank_patching.md`: timing one variant fully then the other, back-to-back, was previously found to bias in favor of whichever runs second.)

Y is **2.7x to 70x slower** than cuBLASLt's FP8 path, and the gap widens sharply with size. This is a direct, expected consequence of the scope cuts in section 3, not a mystery:
- **One warp (32 threads) per CTA, one 16x8 output tile per CTA.** At 4096x4096x4096 the grid is `(256, 512)` = 131,072 CTAs, each running only 32 threads through 128 sequential K-iterations - an eighth of the F16 kernel's typical 256-thread CTA, with zero intra-CTA parallelism across warps.
- **No `cp.async` pipelining** - global loads and `mma` compute are strictly serialized (`bar.sync` before and after every K-substep), never overlapped.
- **No Autotuner integration** - the CTA tile is fixed at 16x8x32 regardless of problem size, unlike the F16 path's per-shape tile selection.
- **No grid swizzle** - no L2-locality reuse across concurrently-scheduled CTAs' A/B tiles.

## What a next session would need, in order (mirrors the F16 kernel's own multi-session arc from correctness-first to competitive)

1. **Multi-warp CTA tiling** - the single biggest lever. Restructure `emit_fp8_gemm_kernel` to select a real CTA tile (e.g. 64x64x64+) with several `mma` calls per warp (`num_i`/`num_j` loops, mirroring `emit_gemm_compute_block`'s structure), directly attacking the "131K tiny CTAs, 32 threads each" problem above.
2. **`cp.async` pipelining** for the quantize+stage step, overlapping global loads with compute the way the F16 kernel's pipelined path does.
3. **Autotuner integration** (a `Precision::FP8` path or equivalent) to select CTA tile per real problem shape instead of a fixed 16x8x32.
4. **Grid swizzle** for L2 locality once the tile size is large enough for it to matter.
5. **`ncu` profiling** once 1-4 land, to find the *actual* remaining bottleneck rather than guessing further - this project's `ncu` access constraint (see `benchmark_y_tensor_core_gemm_results.md`'s own note) applies here too.

This is a description of a reasonable next step, not a promise of the outcome - the F16 kernel took multiple real, independently-validated sessions to go from correctness-first to 0.77x of cuBLAS, and there is no reason to expect FP8 to close a much larger gap (currently 70x at 4096, vs F16's starting point) in a single follow-up pass.

## Unchanged from the prior session

`emit_wgmma_fp8_dual_accumulator_gemm`, `emit_fp8_scaling_mma`, `emit_fp8_adaptive_gemm`, `emit_fast_int4_dequant_lop3`, `emit_sparse_24_mma` are still unreachable dead code - untouched this session, out of scope (the new `emit_fp8_gemm_kernel` is an independent, reachable path alongside them, not a fix to them). `quantization_pass.rs`'s placeholder FP8/INT8 paths are likewise untouched. WGMMA is still a confirmed hardware dead end on this GPU (`sm_89`, Ada Lovelace) - not revisited, no new information.

## Session 2: Multi-warp CTA tiling - a real, large win at 1024+, a regression at 256

This session did item 1 from the "next session" list above: restructured `emit_fp8_gemm_kernel` from a single-warp/single-`mma`-instruction CTA tile to a real multi-warp CTA tile, directly attacking the "131,072 tiny CTAs, 32 threads each, zero intra-CTA parallelism" problem Session 1 identified as the root cause of the 2.7x-70x slowdown vs cuBLASLt.

### 1. Design: CTA 128x128x64, 4x2 warps (256 threads/CTA)

Rather than guessing a shape, the new `FP8_GEMM_CTA_M`/`_N`/`_K`/`_WARPS_M`/`_WARPS_N` constants (`src/ptx_emitter.rs`) directly mirror a REAL, already-proven F16 autotuner candidate - `(128, 128, 64, 4, 2)` from `Autotuner::generate_candidates`'s `m/n > 512` branch. The resulting per-warp tile (32x64) needs `num_i=2`/`num_j=8` `mma.m16n8k32` calls per K-substep (FP8's N=8 mma granularity is half F16 wmma's N=16, so `num_j` is doubled vs the analogous F16 config), but each accumulator fragment is half the register count (4 f32 regs vs 8), so total accumulator register pressure per warp comes out identical either way (64 registers/warp). No Autotuner integration yet (`Autotuner::generate_candidates` accepts but ignores `Precision::Fp8` - unchanged, still a deliberate scope cut, see below) - this is a single, fixed, hand-picked shape, not a per-problem-size search.

Combined A+B shared-memory footprint at this tile size is `128*64 + 64*128` = 16384 bytes (e4m3 is 1 byte/element, unlike f16's 2) - comfortably under `ptxas`'s 48KB static-`.shared` cap, so - unlike the F16 kernel's much larger, autotuned, `cp.async`-pipelined tiles - this still uses simple static `.shared` arrays, not the dynamic/`extern` mechanism. Confirmed directly: `ptxas -arch=sm_89 -v` on all four regenerated `tests/gemm_fp8_{256,512,1024,4096}.ptx` reports **125 registers/thread, 0 spill stores/loads, 16384 bytes smem** - clean across every size.

### 2. Standalone validation before touching the emitter

Per this project's established discipline ([[feedback_gemm_kernel_validation]]), the new multi-warp address composition was validated standalone BEFORE any Rust code was written - not re-deriving the `mma.sync.m16n8k32` instruction's own per-lane fragment formula (already proven in Session 1), but specifically the NEW integration-level risk multi-warp tiling introduces: `lane = %tid.x & 31` (not `%tid.x` directly, now that a CTA is 256 threads/8 warps, not 32/1), per-warp CTA-local tile origin (`warp_row0_local`/`warp_col0_local`) composing correctly with `cta_row0`/`cta_col0` and the `i`/`j`/`kk` loop offsets, and the generalized 256-thread-cooperative runtime-loop quantize+stage step for the larger tile.

Two scratchpad scripts (independently hand-authored in Python - generating raw PTX text via simple loops, not derived from any Rust code, since none existed yet):
- `validate_fp8_multiwarp.py`: 8 cases (single-CTA, multi-CTA exact-multiple, multi-K-tile, ragged M/N down to 17x9, combinations of these) at the exact planned CTA shape, small-integer inputs (exactly representable in e4m3) with `scale=1.0` so the ENTIRE pipeline - quantize, stage, mma, epilogue - had to reproduce an **exact** integer matmul with zero rounding error anywhere. All 8 passed with `max_abs_diff=0` against a numpy int64 reference, on real sm_89 hardware.
- `validate_fp8_multiwarp_race.py`: specifically targeting the NEW non-determinism risk class this task flagged - cross-warp shared-memory races if `bar.sync` placement isn't right (8 warps now share one single-buffered `smem_a`/`smem_b` pair per CTA, unlike Session 1's one warp racing only itself). 30 repeated launches of the most warp/K-tile-heavy case (M=384, N=160, K=192) with fixed input: bit-for-bit identical every time, and still exact-matching the integer reference.

Only after both passed was `ptx_emitter.rs` touched.

### 3. Rust implementation

`emit_fp8_load_a_one`/`_fragment` and `emit_fp8_load_b_one`/`_fragment` gained `warp_row0_local`/`i`/`kk` and `warp_col0_local`/`j`/`kk` parameters respectively (generalizing the fixed single-fragment addressing). `emit_fp8_quantize_stage_a`/`_b` were rewritten from a fixed-32-lane, compile-time-unrolled sequence to a `threads_per_cta`-wide cooperative RUNTIME loop (mirroring `emit_gemm_tile_load`'s thread-striding `bra`-loop pattern), so the instruction count no longer grows linearly with tile size. A new `emit_fp8_gemm_compute_block` mirrors `emit_gemm_compute_block`'s "B-fragments-per-j computed once and reused across every i, A-fragment-per-i computed once and reused across every j" structure (not copy-pasted - FP8's `mma.sync.m16n8k32` already consumes a full 2-register B fragment and 4-register A fragment in one call per `(i,j)`, unlike F16's `mma.sync.m16n8k16`, so there's no further N-half split). `emit_fp8_accum_row_col` (the per-lane local-row/col-within-16x8-tile formula) needed no changes at all - it was already independent of warp/CTA placement, with the caller now adding `warp_row0_local + i*16` / `warp_col0_local + j*8` before adding `cta_row0`/`cta_col0`.

Both `bar.sync 0` calls from the Session-1 kernel (before and after the compute block) are unchanged in number and placement - now load-bearing across 8 warps sharing one buffer instead of 1 warp synchronizing with itself, confirmed race-free by the standalone check above and by real-hardware repeated-launch checks below.

`cargo test`: **74 passed, 0 failed** (up from 73 - one existing FP8 regression test updated for the new smem sizes, one new test added: `test_fp8_gemm_kernel_multiwarp_cta_tiling`, checking the launch-geometry comment, the `lane = tid.x & 31`/`warp_id = tid.x >> 5` decomposition, and that the mma instruction appears exactly `num_i*num_j*k_substeps` = 32 times in the static PTX text).

### 4. Correctness on real hardware - still holds at the new tile size

Re-ran the same shape categories Session 1 validated, against the REAL Y-compiler-emitted PTX (not the standalone prototype above) via `tests/test_fp8_gemm_correctness.py` (grid/block launch formula updated to `ceil(M/128), ceil(N/128)` / `(256,1,1)`, and extended this session with the repeated-launch determinism check described below):

- **Square, multiple seeds**: 256/512/1024/4096 (M=N=K) all pass, relative L2 error 0.018% / 0.032% / 0.052% / 0.131% - essentially IDENTICAL to Session 1's single-warp numbers at every size (0.018%/0.032%/0.052%/0.131%) - expected, since the underlying e4m3 quantization and f32 accumulation math is unchanged, only its distribution across warps changed.
- **Ragged shapes**: 250x100x256, 33x17x64, 200x150x128, 130x130x64 (straddles the 128-wide CTA tile edge in both dimensions), 17x9x64 (smaller than one warp-tile edge), 1x1x64 (extreme edge case) - all pass, relative L2 0.007%-0.022%.
- **Repeated launches, fixed input** (NEW this session, the cross-warp-race check): 20 repeated launches at every size tested above, including 4096x4096x4096 (1024 CTAs x 8 warps = 8192 concurrently-scheduled warps, the most demanding case for a `bar.sync`-placement bug) - bit-for-bit identical output every launch, at every size.
- **Note**: K must now be a multiple of `FP8_GEMM_CTA_K` (64), narrowed from Session 1's "multiple of 32" - a `debug_assert`, not a `type_checker`-enforced constraint (same pre-existing gap Session 1 had, just narrower; not newly introduced this session). All test shapes above respect it.

### 5. Performance on real hardware - a real, large win at 1024+, a regression at 256

| Matrix (M=N=K) | Session 1 (single-warp 16x8x32) | Session 2 (multi-warp 128x128x64) | Change |
|---|---|---|---|
| 256x256x256 | 0.374x | **0.111x** (range 0.08x-0.13x across 4 repeated invocations) | **worse** |
| 512x512x512 | 0.042x | **0.080x** (range 0.078x-0.103x) | ~2x better |
| 1024x1024x1024 | 0.018x | **0.150x** (range 0.134x-0.154x) | ~8x better |
| 4096x4096x4096 | 0.014x | **0.139x** (range 0.139x-0.140x, tight) | ~10x better |

(Methodology unchanged from Session 1: CUDA events, median of 7 alternating rounds/20 launches per round, after 15 rounds of shared alternating warmup - see `benchmark_y_fp8_gemm_results.md` for the exact canonical run and `tests/benchmark_y_fp8_gemm.py`'s docstring for why alternating timing matters on this hardware. The ranges above are across 4 SEPARATE invocations of the whole benchmark script, run back to back this session, reported in addition to the usual within-run [min,max] to show how stable each size's number actually is - 4096 barely moved between runs; 256 swung nearly 2x.)

**This is a genuine, large win at 1024 and 4096** (both ~8-10x faster in absolute terms, and the gap to cuBLASLt closed from 56x-70x down to ~7x-13x) - multi-warp tiling was correctly identified as "the single biggest lever" in Session 1's next-steps list. It is **not** a uniform win, and the 256 regression is worth taking seriously rather than averaging away:

- **Root cause of the 256 regression, not yet fixed**: at M=N=K=256 with a 128x128 CTA tile, the grid is only `(2,2)` = **4 CTAs total** - on a GPU with dozens of SMs, that leaves the overwhelming majority of the chip idle no matter how efficient each of those 4 CTAs is internally. Session 1's tiny 16x8x32 tile produced a `(16,32)` = 512-CTA grid at this same size - far more CTAs than SMs, so despite each CTA doing almost no work, the GPU's SMs were at least all busy. This is the textbook failure mode a FIXED CTA tile has on small problems, and it is a direct, predictable consequence of shipping multi-warp tiling WITHOUT Autotuner integration (item 3 on the original next-steps list) - not a new bug, just a known, disclosed gap now made concretely visible with a real number.
- **512 and 1024 sit in between**: `(4,4)`=16 CTAs and `(8,8)`=64 CTAs respectively - still well under typical SM counts, but each CTA now does enough real work (256 threads, 32 mma calls/K-substep) that the improved per-CTA efficiency outweighs the under-occupancy, netting a real (if partial, at 512) win.
- **4096 is where this tile shape is actually well-matched**: `(32,32)`=1024 CTAs, comfortably saturating the GPU, and this is also the tightest, most reproducible number across repeated invocations (0.139x-0.140x every time) - the clearest evidence this session's change is a real structural improvement, not noise.

### What a next session would need, in order (updated)

1. **Autotuner integration** (item 3 from Session 1's list) is now the clear priority, promoted ahead of `cp.async` pipelining: this session's own 256 regression is a direct, disclosed consequence of NOT having it, and is likely fixable on its own (without any further kernel-structure work) by selecting a smaller CTA tile - or falling back to something closer to Session 1's single-warp shape - when the problem is too small to fill the grid at 128x128. A `Precision::Fp8` branch in `Autotuner::generate_candidates`/`score_candidate`, or even a simpler standalone heuristic in `emit_fp8_gemm_kernel` (e.g. picking between 2-3 fixed tile sizes by M/N before defaulting to 128x128x64), would both be reasonable first attempts.
2. **`cp.async` pipelining** for the quantize+stage step (still not implemented) - the next lever once tile selection is no longer fighting itself at small sizes.
3. **Grid swizzle** for L2 locality, once the tile size is large enough (and adaptively chosen) for it to matter.
4. **`ncu` profiling** once 1-3 land - this project's `ncu` access constraint (see `benchmark_y_tensor_core_gemm_results.md`'s own note) still applies; in particular it would help confirm the grid-under-utilization explanation for the 256 regression directly (occupancy/active-SM count) rather than relying on the CTA-count arithmetic above alone.

Even at 4096 (this session's best result), Y is still ~7x slower than cuBLASLt (0.139x) - closing most of a 70x gap to a 7x one in one session is real progress, not a finished job, and the 256 regression is a concrete reminder that a fixed tile shape trades one failure mode (too-small CTAs) for another (too-few CTAs) rather than eliminating the tradeoff outright.

## Session 3: Two-tier tile selection - the 256 regression improves, but doesn't fully recover

This session did item 1 from Session 2's updated "next session" list: added a second, smaller CTA tile and a compile-time size heuristic to pick between them, directly targeting the 256 regression Session 2 found and root-caused (a fixed 128x128 CTA tile only produces a 2x2=4-CTA grid at M=N=256, under-utilizing the GPU).

### 1. Design: a SMALL tier (64x64x64, 2x2 warps) alongside the existing LARGE tier, chosen by a fixed M/N threshold

`FP8_GEMM_CTA_M_LARGE`/`_N_LARGE`/`_WARPS_M_LARGE`/`_WARPS_N_LARGE` (the Session 2 shape, renamed) sit alongside new `FP8_GEMM_CTA_M_SMALL`/`_N_SMALL`/`_WARPS_M_SMALL`/`_WARPS_N_SMALL` (64x64x64, 2x2 warps = 128 threads - per-warp tile 32x32, `num_i=2`/`num_j=4`, half the LARGE tier's accumulator register pressure per warp, consistent with the smaller tile). `FP8_GEMM_CTA_K` (64) is shared by both tiers unchanged - only M/N/warp-count vary. `emit_fp8_gemm_kernel` picks between them with `m <= FP8_GEMM_SMALL_THRESHOLD (256) || n <= FP8_GEMM_SMALL_THRESHOLD`, mirroring `Autotuner::generate_candidates`'s own F16 small-shape threshold convention. This is still explicitly NOT real Autotuner integration - two fixed shapes and one threshold, not a search over a real candidate space - but it is a real, disclosed, working stand-in, and it directly targets the exact failure mode Session 2 measured rather than a hypothetical one.

This required refactoring `emit_fp8_load_b_one`/`_fragment`, `emit_fp8_quantize_stage_a`/`_b`, and `emit_fp8_gemm_compute_block` to take `cta_m`/`cta_n` as real parameters instead of referencing Session 2's fixed constants directly (`emit_fp8_load_a_one`/`_fragment` needed NO changes - their address formula only depends on `FP8_GEMM_CTA_K`, which is tier-invariant, plus caller-supplied `warp_row0_local`/`i`, never `cta_m` itself). `emit_fp8_gemm_kernel` now computes `cta_m`/`cta_n`/`warps_m`/`warps_n` as local variables from the tier heuristic and threads them through everywhere Session 2 used the fixed constants directly.

### 2. Standalone validation before touching the emitter

Same discipline as Session 2 (see [[feedback_gemm_kernel_validation]]): a near-duplicate of `validate_fp8_multiwarp.py` (`validate_fp8_smalltile.py`, project scratchpad) reconfigured for the SMALL tier's shape (64x64x64, 2x2 warps) - genuinely re-validated, not assumed to "just work" by extrapolation from the LARGE tier's already-proven address-composition logic, since `num_i`/`num_j`/warp counts are actually different. 10 cases (single-CTA, multi-CTA, multi-K-tile, ragged M/N down to 17x9, an asymmetric case with M<=256 but N>256) all exact-matched an integer reference (`max_abs_diff=0`) on real sm_89 hardware (62-64 registers/thread, 8192 bytes smem, 0 spills). A repeated-launch race check (30 launches, fixed input) was also re-run for this tier specifically - bit-for-bit identical, confirming the `bar.sync` placement stays correct at the smaller warp count (4 warps instead of 8) too.

### 3. Correctness on real hardware, both tiers, including the tier boundary

`cargo test`: **76 passed, 0 failed** (up from 74 - two tests that assumed M=N=256 always selects the tile Session 2 shipped were fixed: `test_fp8_gemm_kernel_reachable_and_uses_mma_sync_m16n8k32`'s smem-size assertions updated for the SMALL tier now selected at that shape, and `test_fp8_gemm_kernel_multiwarp_cta_tiling` moved to M=N=1024 to keep testing the LARGE tier specifically; two new tests added: `test_fp8_gemm_kernel_small_tile_selection` (SMALL tier at M=N=256) and `test_fp8_gemm_kernel_tile_selection_boundary` (confirms M=N=257, one past the threshold, selects LARGE)).

Real-hardware correctness via `tests/test_fp8_gemm_correctness.py` (launch config now mirrors the Rust emitter's own tier heuristic - `fp8_gemm_launch_config()` - rather than a single hardcoded grid/block formula): 256/512/1024/4096 square, the exact tier boundary (257x257), sub-tile-edge shapes (64x64, 128x128), ragged shapes including one that straddles the boundary asymmetrically (M=200<=256 selects SMALL despite N=300>256, per the `||` condition) - all pass, relative L2 0.000%-0.131%, and all deterministic across 20 repeated launches including at 4096x4096x4096 (still the most warp/CTA-heavy real-hardware case).

### 4. Performance: 256 improves substantially but does not fully recover to Session 1's number; 512/1024/4096 unaffected

| Matrix (M=N=K) | Session 1 (single-warp 16x8x32) | Session 2 (LARGE tile only) | Session 3 (two-tier) | Change vs Session 2 |
|---|---|---|---|---|
| 256x256x256 | 0.374x | 0.111x | **~0.17x-0.19x** (one of four runs read 0.071x - see note below) | better, not fully recovered |
| 512x512x512 | 0.042x | 0.080x | **0.077x-0.084x** | unchanged |
| 1024x1024x1024 | 0.018x | 0.150x | **0.129x-0.142x** | unchanged (within noise) |
| 4096x4096x4096 | 0.014x | 0.139x | **0.133x-0.139x** | unchanged |

(4 separate full-benchmark-script invocations run back to back this session, same methodology as Session 2's stability check. 512/1024/4096 landed in the same range Session 2 measured, confirming the tiering refactor didn't regress the LARGE-tier code path. 256 was noisier: three of four runs read 34-37us/0.17x-0.19x, consistent and reproducible; one run read 178.79us/0.071x, with `torch._scaled_mm`'s OWN time also roughly doubling in that same run (12.75us vs the usual ~6us) - `nvidia-smi` immediately after showed the GPU idle and cool (54C, 585MHz of a 3120MHz max, 29% utilization), consistent with a transient system-level effect unrelated to either kernel, matching a noise pattern already seen and documented in this project's own memory (see project_wafer_pitch_status.md's "Recurring gotcha" note about occasional whole-system slowdowns hitting both variants at once). The canonical table written by the benchmark script (`benchmark_y_fp8_gemm_results.md`) reflects the 4th, clean run.)

**Honest bottom line at 256**: this is a real, reproducible ~1.5-1.7x improvement over Session 2's regression (0.111x -> ~0.17x-0.19x), but it does NOT fully recover to Session 1's original 0.374x at this specific size. The reason is directly visible in CTA counts: Session 1's tiny 16x8x32 tile produced a 512-CTA grid at M=N=256 (more CTAs than SMs, hiding launch/tail-effect overhead even though each CTA did almost no work); the new SMALL tier's 64x64 tile produces only a 16-CTA grid at this same size - far better than the LARGE tier's 4, but still well short of Session 1's 512. **512/1024/4096 are unaffected** (within run-to-run noise of Session 2's own numbers), confirming the two-tier change is additive, not a tradeoff against the sizes that were already working.

### What a next session would need, in order (updated again)

1. **A third, finer-grained tier (or real Autotuner integration) for very small shapes** - the 256 case shows the SMALL tier (64x64x64) is a real improvement but not yet CTA-count-competitive with Session 1's original single-mma tile at the smallest sizes. A `32x32x64`/1-warp tier (using this session's now-generic `warps_m=1`/`warps_n=1` code path - untested but structurally supported) would roughly quadruple the CTA count again at 256 relative to the current SMALL tier, worth standalone-validating before shipping. Real Autotuner integration remains the more durable fix (see below).
2. **`cp.async` pipelining** for the quantize+stage step (still not implemented) - unaffected by this session's change, still the next lever once tile selection stops being the dominant effect at small sizes.
3. **Grid swizzle** for L2 locality, once tile selection is no longer the dominant lever.
4. **`ncu` profiling** once 1-3 land - would help confirm whether 256's remaining gap (vs Session 1) is really CTA-count/occupancy as the arithmetic above suggests, or whether something else (e.g. per-CTA launch overhead at this thread count) is also contributing - this project's `ncu` access constraint (see `benchmark_y_tensor_core_gemm_results.md`'s own note) still applies.

## Session 4: Software double-buffered K-loop pipelining - correct, thoroughly validated, but no measurable speedup

This session attempted item 2 from Session 3's list (`cp.async` pipelining for the quantize+stage step) and hit a real design constraint before writing any code: **`cp.async` copies raw bytes global->shared with NO type conversion**, so it cannot perform this kernel's F32->e4m3 `cvt.rn.satfinite.e4m3x2.f32` staging conversion the way the F16 kernel's `emit_gemm_tile_load_async` uses it for a same-format copy. Staging RAW (unconverted) F32 instead, then converting from shared, would need 4x the shared-memory bytes (e4m3 is 1 byte/element, f32 is 4) - computed out to ~147KB for the `_LARGE` tier's tile, over this GPU's real ~100KB per-CTA shared-memory ceiling. This is a genuine difference from the F16 kernel's situation, not a scope cut - `cp.async` pipelining as literally described in Session 2/3's next-steps list is not directly applicable here.

### 1. Design and implementation: software double-buffering instead of `cp.async`

Given real `cp.async` doesn't fit, this session implemented pipelining with ordinary SYNCHRONOUS `ld.global.f32`/`cvt`/`st.shared` instructions instead, restructured so each K-iteration issues the NEXT iteration's quantize+stage (into one of two double-buffered e4m3 shared-memory buffers) BEFORE consuming the CURRENT iteration's already-staged buffer - the latency-hiding comes from issuing high-latency global loads earlier in program order relative to when their result is needed, not from a dedicated async-copy engine. This cut `bar.sync` from 2/K-iteration (Session 2/3's design) to 1/iteration: a thread's own later reads always see its own earlier writes without a barrier, so a single barrier placed after both the prefetch-write and the compute-read suffices for cross-thread visibility in both directions. A prologue unconditionally stages K-tile 0 (always exists, since `K % FP8_GEMM_CTA_K == 0` and K is a positive literal guarantee `k_tiles >= 1`); the steady-state loop's prefetch is predicated off (`next_k_iter < k_tiles`) on the last iteration, where there is no next tile to stage.

The doubled shared-memory footprint (32768 bytes for `_LARGE`, 16384 for `_SMALL`) stays comfortably under `ptxas`'s 48KB static-shared cap for both tiers - confirmed directly, all four regenerated `tests/gemm_fp8_{256,512,1024,4096}.ptx` still assemble with **0 spill stores/loads**, 81-117 registers/thread depending on tier.

### 2. Standalone validation before touching the emitter

Same discipline as every prior FP8 session (see [[feedback_gemm_kernel_validation]]): `validate_fp8_pipeline.py` (project scratchpad, LARGE tier - the higher-warp-count, higher-risk shape) generated the full double-buffered design as raw PTX via Python, independently of any Rust code (none existed yet). Exact-integer correctness confirmed across K depths from 1 K-tile (prologue-only - the steady-state loop's single iteration must fully predicate off its own prefetch, the trickiest edge case) up to 8 K-tiles, plus **30 repeated launches at the deepest depth tested - bit-for-bit identical**, specifically checking whether cutting `bar.sync` from 2 to 1 per iteration reopens a cross-warp shared-memory race. It does not.

### 3. Correctness and determinism on real hardware - unaffected

`cargo test`: **77 passed, 0 failed** (up from 76 - two smem-size assertions updated for the doubled footprint, one new test added: `test_fp8_gemm_kernel_pipelined_double_buffering`, checking the exact static `bar.sync`/`and.pred` occurrence counts derived from and cross-checked against the real compiler's own output, not guessed). Real-hardware correctness via the actual compiled kernel: square shapes 256/512/1024/4096, K-depths from 1 to 64 K-tiles, ragged shapes, and the M=N=1 extreme case all pass with relative L2 error **identical to Session 3's pre-pipelining numbers** (0.018%/0.032%/0.052%/0.131% at 256/512/1024/4096, as expected since pipelining changes scheduling, not the math) - and all deterministic across 20 repeated launches per shape.

### 4. Performance: no measurable improvement, honestly reported

| Matrix (M=N=K) | Session 3 (single-buffered, median of 4 runs) | Session 4 (double-buffered, median of 5 runs) |
|---|---|---|
| 256x256x256 | 0.174x | 0.155x (range 0.111x-0.208x across 5 runs) |
| 512x512x512 | 0.079x | 0.078x |
| 1024x1024x1024 | 0.133x | 0.120x |
| 4096x4096x4096 | 0.135x | 0.135x |

**This is a null result, reported plainly rather than dressed up**: 512 and 4096 are statistically unchanged; 256 is within this environment's demonstrated noise band (a single run swung from 0.111x to 0.208x - a wider range than the before/after difference itself); 1024 shows a small, fairly consistent ~10% decrease across the 5 runs (0.120, 0.119, 0.140, 0.124, 0.116 - four of five cluster tightly, one outlier high) that may be real but is not dramatic and was not confirmed via profiling.

**Why this pipelining technique likely didn't help, reasoned from first principles (not confirmed - no `ncu` access, same constraint noted throughout this project)**: a back-of-envelope occupancy calculation (mirroring `Autotuner::estimate_occupancy`'s own methodology) using the real `ptxas -v` register counts suggests REGISTER pressure, not shared memory, was already the binding occupancy constraint at the `_LARGE` tier before this change (117-125 registers/thread x 256 threads leaves room for only ~2 CTAs/SM by the register limit alone on a ~64K-register SM, versus ~6 CTAs/SM by the pre-pipelining 16KB smem footprint - registers were already the tighter constraint, so doubling smem to 32KB, which still permits ~3 CTAs/SM, does not further reduce occupancy). If occupancy is unchanged, the most likely explanation for the lack of improvement is that the GPU was already hiding global-memory latency effectively via warp-level scheduling (switching between the SM's already-resident warps) rather than needing help from instruction-level reordering within a single warp's own stream - and/or the quantize+stage step's cost is dominated by its sheer scalar instruction count (unvectorized `ld.global.f32`+`cvt`+`st.shared` per pair, unlike the F16 kernel's 128-bit vectorized tile loads) rather than by raw memory latency, in which case reordering WHEN a load is issued doesn't help if the bottleneck is a different resource entirely.

**Decision: kept, not reverted.** The implementation is correct, thoroughly validated, and not measurably worse in a way clearly attributable to the change (the occupancy arithmetic above gives no mechanistic reason to expect a regression, and the one metric that did move (1024) moved by less than this environment's typical run-to-run noise elsewhere in this same session). Reverting to Session 3's single-buffered design remains a trivial, low-risk option for a future session if further measurement confirms a real regression - but discarding a correct, validated implementation on an inconclusive signal would be premature.

### What a next session would need, in order (updated again)

1. **`ncu` profiling** is now the highest-value next step specifically BECAUSE of this session's inconclusive result - guessing further about pipelining, occupancy, or instruction-count bottlenecks without real profiler data has reached diminishing returns; this project's `ncu` access constraint (see `benchmark_y_tensor_core_gemm_results.md`'s own note) still applies, so this needs the user to run it directly.
2. **Vectorizing the quantize+stage step** (currently scalar `ld.global.f32`/`cvt`/`st.shared` per pair, unlike the F16 kernel's 128-bit `ld.global.v4.u32` tile loads) is now a concrete, testable hypothesis for what's actually limiting this kernel, raised directly by this session's pipelining result - worth trying before further pipelining work.
3. **A third, finer-grained tile tier (or real Autotuner integration)** for very small shapes - unaffected by this session, still open from Session 3.
4. **Grid swizzle** for L2 locality, once the above are no longer the dominant levers.

## Session 5: Vectorized quantize+stage (a real, large win) - plus two hypotheses tried and rejected on real hardware

This session acted on item 2 from Session 4's list - vectorizing the quantize+stage step - and separately investigated two more items raised across this doc's history: a finer-grained tile tier for small shapes (Session 3's own next-step #1), and shared-memory bank-conflict swizzling. All three were standalone-designed/analyzed, wired in, and measured on real sm_89 hardware before any decision was made to keep or revert - two were kept, one was reverted after real measurement showed a regression, following this project's established discipline of reporting negative results plainly rather than silently dropping them (precedent: Session 4 itself).

### 1. Vectorized quantize+stage: `ld.global.v4.f32` + packed `.b32` stores

`emit_fp8_quantize_stage_a`/`_b` (`src/ptx_emitter.rs`) were restructured from 2-element/thread/iteration (scalar `ld.global.f32` x2 + `cvt.rn.satfinite.e4m3x2.f32` + `st.shared.b16`) to 4-element/thread/iteration ("quads"): one 128-bit `ld.global.v4.f32`, two `cvt.rn.satfinite.e4m3x2.f32` calls combined via a new `emit_fp8_pack_quad` helper (`cvt.u32.u16` zero-extend + `shl`/`or`) into a single `.b32`, and one `st.shared.b32` - a 4x reduction in load-instruction count, 2x in store-instruction count. PTX ISA 8.5 confirmed (again) no `.e4m3x4` conversion exists (only `.e4m3x2`), matching Session 1's original finding - the packing has to be done by hand from two `.e4m3x2` results, not a single wider `cvt`.

**A is unconditionally safe to vectorize**: its boundary mask is per-ROW only (K is never boundary-masked, `K % FP8_GEMM_CTA_K == 0` always required), so all 4 elements of a quad always share the same row/validity - one predicate per quad, same granularity the original per-pair version already used.

**B needs a hybrid fast/slow path AND has a real alignment hazard**: N's boundary can fall inside a quad, and a single `@p ld.global.v4.f32` can't express "3 of 4 lanes valid". `emit_fp8_quantize_stage_b_vectorized` takes a FAST path (whole quad in-bounds, common case) using the same vectorized approach as A, and falls back to a SLOW path (two independently-predicated `emit_fp8_quantize_pair` calls, the original per-element-masked scalar logic) when a quad straddles the boundary - preserving the exact masking granularity the ragged-shape tests already depend on. Standalone-validated first (`validate_fp8_vectorized_stage.py`, project scratchpad, hand-written raw PTX, checked bit-exact against `torch.float8_e4m3fn`'s own encoding): 16/16 cases passed, including every mid-quad boundary residue (n_bound mod 4 = 0/1/2/3) and extremes (n_bound=0, n_bound=1).

**A real bug found by real-hardware ragged-shape testing, before this ever reached a released kernel**: wiring the vectorized B path in unconditionally and running `tests/test_fp8_gemm_correctness.py 33 17 64` crashed with `CUDA_ERROR_MISALIGNED_ADDRESS`. Root cause: `ld.global.v4.f32` requires its address 16-byte aligned. B's row byte-stride is `n*4` (real N, arbitrary) - A's own row-stride (`k*4`) is always a multiple of 16 since `K % FP8_GEMM_CTA_K(64) == 0` is already required, but B has no equivalent guarantee on N. When `n % 4 != 0` (17 % 4 == 1 here), `(grow*n + col0)*4` is not 16-byte aligned for most K-rows even though `col0` itself is always a multiple of 4. Fix: since `n` is a compile-time `@tile` constant (not a runtime value), this is resolved at Rust codegen time - `emit_fp8_quantize_stage_b` dispatches to the vectorized path only when `n % 4 == 0`, else to `emit_fp8_quantize_stage_b_scalar` (the preserved original pair-at-a-time implementation, whose 4-byte `ld.global.f32` accesses are always alignment-safe regardless of N). Re-confirmed fixed at M=33,N=17,K=64 and a wide sweep of other non-multiple-of-4 N shapes (9, 7, 150, 257, ...) afterward.

`cargo test`: 77 passed (one existing test's hardcoded `and.pred` count updated 67->70 for the new predicate-combination shape - 1 in A's single predicate + 1 in B's FAST combined predicate + 4 in B's SLOW per-element predicates, empirically confirmed against real compiler output before updating, not guessed). All four `tests/gemm_fp8_{256,512,1024,4096}.ptx` regenerated, `ptxas -arch=sm_89 -v`: 0 spill stores/loads at every size.

**Real-hardware benchmark result - a real, large win at 1024+**:

| Size | Session 4 (pre-vectorization) | Session 5 (vectorized) | Change |
|---|---|---|---|
| 256 | 0.155x (noisy, 0.111x-0.208x) | ~0.15x-0.26x (still noisy - see note below) | within noise |
| 512 | 0.078x | ~0.10x-0.11x (before the threshold change below) | ~1.3x better |
| 1024 | 0.120x | ~0.20x-0.21x | **~1.7x better** |
| 4096 | 0.135x | ~0.219x (tight, reproducible) | **~1.6x better** |

This directly confirms Session 4's own hypothesis ("the quantize+stage step's cost is dominated by its sheer scalar instruction count... rather than by raw memory latency") - cutting that instruction count 2-4x produced a real, large, reproducible speedup at the sizes where the K-loop dominates total kernel time, exactly as predicted.

### 2. `FP8_GEMM_SMALL_THRESHOLD` raised 256 -> 512 (a real win, confirmed by A/B, not guessed)

Investigated whether Session 3's own next-step #1 ("a third, finer-grained tile tier... or real Autotuner integration for very small shapes") should extend `_SMALL`'s (64x64x64) range further, since `_LARGE`'s 4-CTA grid at M=N=512 (`(4,4)`) looked like the same grid-starvation pattern that motivated `_SMALL` over `_LARGE` in the first place. Real hardware A/B at exactly M=N=K=512 (both variants compiled for real via a temporarily-adjusted threshold, not launched with mismatched grid/block against the wrong kernel geometry - an early attempt at decoupling launch config from the compiled tier caused exactly that mistake and produced `WRONG`/`nan` results before the bug was caught): `_SMALL`'s 64-CTA grid measured **~0.148x-0.150x**, reproducibly, vs `_LARGE`'s **~0.098x-0.112x** across three separate benchmark runs. A real, disclosed ~1.4x win, not a guess - `FP8_GEMM_SMALL_THRESHOLD` raised to 512. `cargo test`'s tile-selection-boundary test updated (M=N=257 -> 513, the new boundary one past 512) and re-passes; `tests/test_fp8_gemm_correctness.py`/`tests/benchmark_y_fp8_gemm.py`'s own Python mirrors of the threshold updated to match (they must agree with the Rust emitter's compile-time choice or the host launch config silently disagrees with what the compiled kernel expects).

### 3. A `_TINY` (32x32x64, 1-warp) tier: built, standalone-benchmarked, REJECTED

Session 3's list flagged this directly: "A `32x32x64`/1-warp tier... would roughly quadruple the CTA count again at 256... worth standalone-validating before shipping." Implemented as a real third tier (`FP8_GEMM_CTA_M_TINY` etc., `warps_m=1`/`warps_n=1`, reusing the already-generic cta_m/cta_n/warps_m/warps_n code path from Session 3's refactor) and A/B'd against `_SMALL` on real hardware at three shapes chosen to be favorable to `_TINY`'s CTA-count argument:

| Shape | `_SMALL` grid | `_TINY` grid | `_SMALL` speedup | `_TINY` speedup |
|---|---|---|---|---|
| M=N=K=64 | 1 CTA | 4 CTAs | **0.81x** | 0.57x |
| M=64,N=2048,K=128 | 32 CTAs | 128 CTAs | **0.26x** | 0.24x-0.49x (noisy) |
| M=32,N=2048,K=128 (exact fit for `_TINY`'s M-tile, `_SMALL` wastes half its M-tile) | 32 CTAs | 64 CTAs | **0.48x** | 0.33x |

`_SMALL` won in every single case, including the M=32 case specifically engineered to favor `_TINY` (exact tile fit vs `_SMALL`'s 50%-wasted M-tile) and the skinny 64x2048 case with 4x more CTAs available to `_TINY`. Conclusion: `_SMALL`'s 4 warps/CTA apparently hide global-memory latency via intra-CTA warp scheduling more effectively than `_TINY`'s extra CTA-level parallelism compensates for losing - raw CTA count is not the whole story, contrary to the pure grid-occupancy argument that motivated trying it. **Reverted, not shipped** - a real, disclosed negative result, matching this project's own Session 4 precedent for reporting inconclusive/negative findings plainly rather than quietly dropping them.

### 4. Shared-memory XOR swizzle for A's fragment load: mathematically solved the conflict, REJECTED due to occupancy cost

Investigated the user-proposed hypothesis that shared-memory bank conflicts in the FP8 kernel's hand-computed `ld.shared` fragment loads (no `ldmatrix` 8-bit variant exists - see Session 1) were costing real performance. Direct simulation of the exact 32-lane address pattern (`emit_fp8_load_a_one`, project scratchpad's `analyze_fp8_banks.py`, not guessed) confirmed a REAL, though moderate, conflict: `FP8_GEMM_CTA_K` (64 bytes = 16 4-byte words) is exactly half the GPU's 32-bank/128-byte cycle, so `bank = (row*16 + word_col) mod 32` only depends on `row`'s LSB - the 8 distinct rows one `ld.shared.b32` fragment-load instruction touches collapse into 2 bank groups of 4, i.e. a genuine 4-way conflict (not the "32-way" originally guessed, but real).

An exhaustive search over range-constrained XOR masks (`mask(row) = (row&1) XOR ((row&7)<<1)`, a true permutation of one row's own 16 words, never spilling into another row's bytes) found a fully conflict-free (max 1 access/bank) swizzle, mathematically verified against all 4 fragment-load instruction variants before any PTX was written. Implemented consistently on both the read side (`emit_fp8_load_a_one`) and write side (`emit_fp8_quantize_stage_a`'s vectorized quad write) via a new `emit_fp8_a_swizzle_col` helper. Correctness held on real hardware (square and ragged shapes alike, including the exact 4096 boundary) - the swizzle only changes WHERE a logical element physically lives in `smem_a`, not what value is there, so read/write consistency was the only correctness risk and it held.

**Performance: a real regression at the largest size, reverted.** The swizzle math itself (6 extra instructions/call: `and`/`and`/`shl`/`xor`/`shl`/`xor`) pushed register usage from 81->94 at `_SMALL` and, more importantly, **117->144 at `_LARGE`**. Real-hardware benchmark: 256/512/1024 were roughly neutral (within this environment's usual run-to-run noise), but **4096x4096x4096 regressed from 0.219x to 0.120x** - confirmed reproducible, not a one-off. The extra register pressure at `_LARGE`'s already-high baseline (117 registers x 256 threads) evidently cost more in lost SM occupancy than the eliminated bank conflict saved in reduced shared-memory latency. **Reverted, not shipped** - correct, standalone-validated, and still a real negative result worth recording: bank-conflict elimination is not free, and on this kernel/hardware the register-pressure cost outweighed the benefit at the size that matters most.

### 5. Final benchmark, all Session-5 changes combined

| Matrix (M=N=K) | Y us (median [range]) | torch._scaled_mm us (median [range]) | Y vs torch | Correct (rel L2) |
|---|---|---|---|---|
| 256x256x256 | 24.01 [23.92, 26.57] | 6.14 [5.89, 17.31] | **0.256x** | OK (0.018%) |
| 512x512x512 | 44.13 [44.03, 44.24] | 6.96 [6.90, 7.53] | **0.158x** | OK (0.032%) |
| 1024x1024x1024 | 123.55 [123.49, 126.05] | 25.24 [25.09, 31.28] | **0.204x** | OK (0.052%) |
| 4096x4096x4096 | 3726.49 [3725.36, 3866.33] | 821.25 [815.63, 860.36] | **0.220x** | OK (0.131%) |

(Full canonical run: `benchmark_y_fp8_gemm_results.md`, same methodology as every prior session - CUDA events, median of 7 alternating rounds/20 launches per round, after 15 rounds of shared alternating warmup. 256 remains this environment's noisiest size across every session so far, not new to Session 5.) Gap to cuBLASLt at 4096 closed further: Session 3's ~0.135x -> Session 5's ~0.220x, roughly 4.5x slower now vs the ~62x gap Session 1 started from.

Correctness unaffected throughout: relative L2 error identical to every prior session at every size (0.018%/0.032%/0.052%/0.131% at 256/512/1024/4096 - pipelining and vectorization change scheduling and instruction count, not the underlying quantize/accumulate math), and re-validated across the full ragged-shape suite (17x9, 33x17, 130x130, 200x150, 513x513, plus the new alignment-fix regression cases) with 20 repeated launches per shape for determinism.

### What a next session would need, in order (updated again)

1. **`ncu` profiling** remains the single highest-value next step - this session's swizzle result is a concrete example of why: guessing at occupancy/register-pressure tradeoffs from `ptxas -v` register counts alone got the DIRECTION right (more registers can cost occupancy) but real profiler data would settle it far faster than build-and-measure iteration. Project's `ncu` access constraint still applies.
2. **A third, finer-grained tile tier for very small shapes (M,N < 256ish) remains open** - this session's `_TINY` (1-warp) attempt was rejected, but that doesn't rule out a 2-warp or 2x1-warp intermediate tier between `_SMALL` and something smaller; any such attempt should be A/B'd on real hardware the same way, not assumed.
3. **A bank-conflict fix with LOWER register cost** (e.g. precomputing the swizzle mask once per warp/lane rather than once per fragment-load call, or a cheaper mask formula even if it only partially reduces the conflict) could still be worth trying now that this session established both that the conflict is real (4-way, not imagined) and that the naive fix's register cost is what killed it - the next attempt's job is to find a cheaper fix, not to re-litigate whether the conflict exists.
4. **Grid swizzle** for L2 locality - still open, unaffected by this session.
