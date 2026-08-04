# FP8/INT8 Quantized GEMM: Investigation Findings (No Working Benchmark)

**UPDATE, follow-up session: superseded by a real, working, reachable kernel.** Everything below is unchanged as a historical record of that session's findings (WGMMA dead end, dead-code inventory, the `mma.sync` version-bump finding this doc's own "next session" list called for). A later session followed this doc's own "What a real next-session attempt would need, in order" list, step by step, and got a real `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` GEMM kernel built, standalone-validated, wired into the real CLI, and correctness/performance-benchmarked against `torch._scaled_mm` on real hardware - see `investigation_fp8_gemm_findings.md` for the full account (short version: correct, reachable, but 2.7x-70x slower than cuBLASLt, honestly not yet competitive). The title/scope-decision below ("No Working Benchmark", "No new kernel was built") describe that earlier session only.

This is an investigation doc, not a benchmark results doc: every quantized-GEMM code path in this codebase turned out to be unreachable from the real Y CLI, and the one path with a "structural PTX-content test" is architecturally incompatible with this session's hardware. No new kernel was built this session (see "Scope decision" at the bottom for why) - this records what was found so a future session doesn't have to re-derive it.

## 1. Every quantization/FP8 code path is unreachable dead code

`grep`-verified: none of `emit_wgmma_fp8_dual_accumulator_gemm`, `emit_fp8_scaling_mma`, `emit_fp8_adaptive_gemm`, `emit_fast_int4_dequant_lop3`, `emit_sparse_24_mma` (all in `ptx_emitter.rs`) are called from `main.rs`, any CLI dispatch path, or any other production code - only from their own unit tests. `quantization_pass.rs`'s `QuantizationPass` is used exactly once outside its own module, by `coprocessor_scheduler.rs` - which `investigation_rt_tensor_coprocessor_findings.md` (this session) already established never produces valid, executable PTX. There is no `.ysu` source construct today that reaches any of this code. Same pattern as this session's other two "verify existing capability" items (the RT Core coprocessor pipeline, and the pre-existing `SwiGLU` code before this session rewrote it) - unreachable scaffolding, not a working-but-unbenchmarked capability.

## 2. `emit_wgmma_fp8_dual_accumulator_gemm`'s WGMMA instructions cannot run on this session's hardware at all

This function - the one with a "structural PTX-content test" (`test_wgmma_fp8_dual_accumulator`, which only asserts the emitted PTX *text* contains certain substrings, never assembles or runs it) - emits `wgmma.fence.sync.aligned` / `wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3` / `wgmma.commit_group.sync.aligned` / `wgmma.wait_group.sync.aligned`. **WGMMA is a Hopper-only (sm_90a+) warpgroup-level instruction family.** This project's dev GPU (`torch.cuda.get_device_properties(0)`: `NVIDIA GeForce RTX 4070 Ti SUPER`, compute capability **8.9**, Ada Lovelace) does not have it. Confirmed empirically, not just from background knowledge: a minimal standalone PTX file issuing exactly these four instructions, assembled with `ptxas -arch=sm_89`, fails outright:

```
error: Instruction 'wgmma.fence' not supported on .target 'sm_89'
error: Instruction 'wgmma.commit_group' not supported on .target 'sm_89'
error: Instruction 'wgmma.wait_group' not supported on .target 'sm_89'
```

This is not "unverified" - it is **verified impossible on this machine**. Separately, even on Hopper hardware, the function's own emitted code references `desc_A`/`desc_B` (TMA tensor-map descriptors) that are never declared or computed anywhere in the function - the same "references undefined symbols, not a complete kernel body" pattern found in the RT Core emitter (item 1) and the pre-rewrite SwiGLU code (item 2).

## 3. The Ada-compatible alternative (`mma.sync` FP8) is architecturally viable here, but needs two real fixes before it could work

Ada Lovelace's 4th-gen tensor cores *do* support FP8 (e4m3/e5m2) via the non-warpgroup `mma.sync.aligned.m16n8k32...e4m3.e4m3.f32` instruction - the instruction `emit_fp8_scaling_mma` already (if unreachably) emits. Verified empirically this session:

1. **`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` assembles successfully on `ptxas -arch=sm_89`** - confirmed with a minimal standalone probe (zero-valued operands, structure only, not a numeric correctness check) - **but only at PTX ISA `.version 8.4` or later**; `.version 7.8` (identical instruction, identical `-arch=sm_89`) fails with `Feature 'mma with FP8 floating point type' requires PTX ISA .version 8.4 or later`.
2. **`ptx_emitter::ptx_version_for_sm` currently maps `sm_89` to `.version 7.8`** (`src/ptx_emitter.rs`, the `sm_89 | sm_8.9` arm). This means even a fully correct FP8 `mma.sync` kernel body would fail to assemble today, purely because the compiler's own emitted version header is one PTX ISA revision too old for this specific feature on this specific target - a small, well-scoped, low-risk fix (bump that one match arm to `.version 8.4`+ for `sm_89`), separate from and much smaller than writing a real kernel.
3. **`emit_fp8_scaling_mma` itself is not a complete kernel body** even setting the above aside: it references `%f_acc`, `%r_a`, `%r_b` as if already defined, with no CTA tiling, no shared-memory staging, no fragment-loading (`ldmatrix` or otherwise) - a one-instruction sketch, not a starting point that just needs wiring up.

## Scope decision: investigated, not implemented, this session

Given (a) every path is unreachable dead code requiring a from-scratch rewrite either way (the same situation SwiGLU was in, which took substantial effort this session to rebuild correctly), (b) the flagship WGMMA path is a confirmed hardware dead end here, and (c) even the viable `mma.sync`-based alternative needs real fragment-layout derivation (the `m16n8k32` e4m3 per-lane layout has not been derived or validated on real hardware by any prior session, unlike `m16n8k16` f16 - see `benchmark_y_tensor_core_gemm_results.md`'s derivation for what that work looks like) before a single correct instruction could be issued, a full FP8 GEMM kernel was judged too large to attempt reliably in this session's remaining time alongside the other four items. This is a validate-feasibility-before-implementing call, not an oversight - see this session's brief.

**PyTorch's own quantized-matmul baseline is available in this environment** if a future session does this work: `torch.float8_e4m3fn` is a real, working dtype in the installed `torch==2.13.0+cu130` (confirmed: constructing and reading back an `e4m3fn` tensor round-trips correctly on this GPU). TensorRT-LLM was not checked for installation (moot until there's a Y-side kernel to compare against).

## What a real next-session attempt would need, in order

1. Bump `ptx_version_for_sm`'s `sm_89` arm to `.version 8.4`+ (small, contained, low-risk relative to the GEMM-kernel changes flagged elsewhere as high-risk).
2. Derive and standalone-validate `mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32`'s real per-lane A/B/D fragment layout from the PTX ISA (same discipline as the existing `m16n8k16` f16 derivation) - not assumed, not guessed.
3. Only then, a real kernel: CTA tiling + shared-memory staging for 1-byte FP8 elements (half the byte-width of the F16 kernels' padding/stride math - not a copy-paste), a quantization step (FP32/FP16 -> e4m3 with a scale factor - `quantization_pass.rs` may or may not be salvageable for this, not evaluated), and a correctness reference against `torch.float8_e4m3fn` matmul (check what op PyTorch 2.13 actually exposes for this - `torch._scaled_mm` or similar - before promising a specific comparison).
