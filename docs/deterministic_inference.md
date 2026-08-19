# Deterministic Quantized Inference

**Bit-identical model outputs across batch size, split configuration, thread
count and GPU model — at competitive throughput, because exact accumulation in
the quantized regime is not a tax.**

| | |
|---|---|
| Horizon | 12–24 months to revenue |
| First demoable result | ~6–10 weeks |
| Status | **M0-M4 complete; M5 demo working** (2026-08-17) - see the milestone log |
| Relationship | The near-term commercial arm of [`proof_carrying_kernels.md`](proof_carrying_kernels.md) |
| Drafted | 2026-08-16 |

> Every claim about this repository below was verified against the source while
> drafting, not recalled. Line numbers are as of the drafting date.

---

## 1. The thesis

Two facts that were established separately in this repo, never connected, and
together describe a product:

1. **Exact accumulation restores associativity.** Integer and fixed-point
   addition are associative and commutative, so any reordering of a reduction
   — a different tile, a different K-split, a different thread count, a
   different batch size — produces a **bit-identical** result. This is the
   whole premise of `proof_carrying_kernels.md` §2.
2. **In the quantized regime exactness is roughly free.** Measured on a Ryzen 9
   9950X, exact `vpdpwssd` + int64 flush runs at **1.10–1.15x f32 FMA** at
   equal MAC count and equal register budget.

The second fact is what makes this a business rather than a research note.
Every prior attempt at deterministic numerics has had to sell a slowdown. This
one does not — **in the quantized regime.** What is traded is *range*, not
speed: operands are int16 and the flush interval bounds their magnitude, which
is a description of where inference already lives rather than a limitation.

> ### Correction, 2026-08-17: this used to claim 1.88x, and that was wrong
>
> `proof_carrying_kernels.md`'s Phase 0 recorded exact VNNI at **314.5 G MAC/s
> against f32's 166.9, i.e. 1.88x**, with the int32→int64 flush costing 7.7%.
> Re-measured here on the same class of machine, with the loop-hoisting guards
> described below:
>
> | kernel | doc | measured | verdict |
> |---|---|---|---|
> | f32 FMA, mr=6 nr=64 | 166.9 | **172.7** | reproduces |
> | VNNI **raw** (no flush, overflows) | 340.8 | **348.0** | reproduces |
> | VNNI **exact** (flushed) | 314.5 | **146.3** | **does not reproduce** |
>
> Two of three numbers reproduce, so the methodology is sound and the third is
> the error. **The flush cost 137.8%, not 7.7%** — and it is not the flush's
> *frequency*: sweeping the interval from 64 to 65536 k-pairs moved throughput
> by under 5%. It is the flush block's *presence* inside the k-loop. A block
> that reads and writes all 24 accumulators forces them out of registers for the
> whole loop; the hot loop carried 24 `vpdpwssd` alongside 15 spill stores and
> 14 reloads.
>
> **Hoisting the flush into an outer chunk loop recovers most of it: 0.85x →
> 1.10–1.15x**, with the flush gone from the hot loop entirely and stack traffic
> down from 29 to 20 ops. The remaining gap to raw's 2.26x is register pressure
> — 24 accumulators + 4 B + 1 A broadcast is 29 of 32 zmm — and is a tiling
> question, not a structural one.
>
> **The thesis survives but its margin does not.** "Exactness is 1.88x *faster*"
> was a headline; "exactness costs nothing measurable" is a footnote you can
> still sell determinism on, because the product was never speed. Every claim
> downstream of the 1.88x must be restated. `proof_carrying_kernels.md`'s Phase 0
> section needs the same correction.

## 2. What is being sold, and what is not

**Sold:** *same weights, same prompt, same tokens — every time, on any batch
size, on any GPU, at throughput within noise of the fast path.*

**Not sold:** quantization. Quantization is commodity — GPTQ, AWQ,
bitsandbytes, TensorRT and llama.cpp all do it for free and it is table stakes
in every serving stack. Quantization is the *enabling condition* here, not the
product. Any pitch that leads with it is filed under commodity and lost.

Keep the distinction sharp in every sentence: **the product is determinism; int8
is merely the regime in which determinism is affordable.**

### The claim is "deterministic", not "exact"

A transformer is not only matmuls. Exact integer accumulation covers GEMM and
attention scores cleanly. Softmax needs `exp`, which cannot be exact in
integer; RMSNorm and layernorm have reduction-order issues in their
mean/variance passes.

So the product claim is **deterministic** — fixed reduction order and
fixed-precision approximation where exactness does not apply — with exactness
as the mechanism where it does. Stating "exact everywhere" is a claim that
would have to be walked back in the first technical conversation.

**It also has a stated context ceiling**, which M5 step 15 derived with Z3 and
which is worth carrying at the top rather than in a milestone: exactness in the
`W·V` accumulator needs `T · 2^28 · V_LEVELS < 2^53`, i.e. **T ≤ 264,208 tokens**
at the shipped V width. Past it the compiler refuses rather than rounding
quietly. That is well beyond any context this prototype is aimed at, but it is a
real edge of the guarantee, it scales as `2^25/V`, and a buyer running 1M-token
contexts needs to hear it before they find it.

### Who buys

Ranked by sharpness of pain, not size:

1. **RL training teams.** Rollout and training numerics disagree, so
   off-policy corrections correct for a bug. Currently the most expensive,
   least-solved version of this problem, and those teams have budgets.
2. **Eval / regression infrastructure.** A model score that moves because the
   batch size moved is not a score.
3. **Regulated deployment** — auditability, reproducible inference. Slower
   sales, larger contracts, and the on-ramp to the certificate work in
   `proof_carrying_kernels.md`.

### Why competitive-not-fastest is the right bar

Y's inference kernels measure 0.93x cuBLAS on FP16 GEMM and 0.76–1.06x
FlashInfer on paged decode attention. **In a throughput race that is a loss.
Under this thesis it is table stakes already met**: "within 10% of FlashInfer
and bit-reproducible" is a winning product where "8% faster than FlashInfer" is
not a product at all.

This is the same two-axis lesson as the ZK constraint-count trade — do not race
on the axis that is not the one being sold.

---

## 3. What exists today, verified

| Component | Where | State |
|---|---|---|
| `@ZeroDrift` representation selection | `zero_drift.rs` (706 lines) | **real**, `DriftRepr::{FixedQ16_16, FixedQ32_32, Int64, Float64, KahanF32}`, `is_exact` keeps exact and precise apart |
| `@ZeroDrift` LLVM lowering | `llvm_emitter.rs`, 4 sites | **real, tested** |
| `@ZeroDrift` PTX lowering | `ptx_emitter.rs`, `emit_drift_to_fixed` / `emit_drift_from_fixed` | **real** |
| Bit-identity end-to-end test | `tests/zero_drift_end_to_end.rs` | **real** — 4001 terms summed in opposite orders, bit-identical, with a control proving f32 genuinely drifts |
| GEMM recogniser carries the drift request | `cpu_gemm.rs:321` | **real** — records `DriftAccumulator { ty, bounds }` |
| Exact VNNI micro-kernel | `tests/probes/vnni_kernels.c` | **real, measured, exact-verified** — but a C probe, not a Y kernel |
| Integer datapath on GPU | `ptx_emitter.rs` | **real for 32/64-bit**: typed ld/st, typed strides, signed vs unsigned, `mul_wide_u32` |
| Device differential gate | `tests/ptx_integer_datapath.rs` | **real** — 16 ops over 4096 full-range `u32` pairs on device vs plain Rust, mutation-verified |
| Attention / RMSNorm / RoPE kernels | `ptx_emitter.rs` | **real**, competitive (RoPE 1.92x FlashInfer) |

## 4. What does not exist — the actual work

| Gap | Where | Why it blocks |
|---|---|---|
| **Sub-word element types refused** | `ptx_emitter.rs:1018` `reject_unsupported_element_types` | `U8/U16/I8/I16` are refused because an element type is also a *stride* and address math shifts by a hardcoded log2(4). **Nothing int8 on GPU is reachable until this lands.** |
| **No integer tensor-core path** | `ptx_emitter.rs` | `mma.sync...m16n8k32...e4m3` (FP8) exists; `...s32.s8.s8.s32` does not. |
| **Exact GEMM not wired** | `llvm_emitter.rs:1537` | `try_emit_gemm_kernel` returns `None` for a drifted nest and falls through to scalar lowering — deliberately *slow and right* rather than fast and wrong (`tests/cpu_gemm_exact_accumulation.rs`). |
| **No batch-invariance harness** | — | Nothing currently asserts that two *different* launch configurations agree bit-for-bit. This is the product's central claim and it is untested. |
| **Softmax / norm reduction order unfixed** | `ptx_emitter.rs` | Determinism here needs a fixed reduction tree, not exactness. |
| **No end-to-end model path** | — | No weight loading, no tokenizer, no serving integration. |

---

## 5. Milestones

Ordered by dependency. Each produces something demoable or publishable on its
own, so the programme can be stopped at any boundary without the prior work
being wasted.

### M0 — Exact GEMM on CPU, bit-identical across every schedule · 3–4 weeks

Port `tests/probes/vnni_kernels.c`'s exact `vpdpwssd` + int64-flush kernel into
`cpu_gemm.rs` as a real emitted kernel, and let `try_emit_gemm_kernel` dispatch
a `@ZeroDrift` nest to it instead of returning `None`.

- **Done when** — a tiled, threaded, K-split GEMM returns results
  **bit-identical** to the naive nest across every thread count, every blocking
  parameter, and both `Y_NUM_THREADS` settings; and the 1.88x holds inside the
  real kernel, not only in the probe.
- **Also done when** — the range obligation is checked, not assumed.
  `@bounds(min, max)` becomes load-bearing: the flush interval bounds operand
  magnitude (`FLUSH_T = 64` k-pairs needs `|a|,|b| <= 1024`), and a nest whose
  bounds do not license the representation must be **refused**, not silently
  widened. Per the design rule.
- **Why first** — completes Phase 0 of `proof_carrying_kernels.md`, is
  independently publishable, needs no new hardware, and validates the thesis
  before any GPU work is committed to.

#### M0 progress

**Landed 2026-08-16 — the range obligation** (`zero_drift.rs`: `VnniExact`,
`OperandBounds`, `license_vnni_exact`; 6 tests, 5-mutation verified). Built
*before* the kernel deliberately: an int32 accumulator that overflows does not
signal, it wraps, so a kernel written first would silently produce plausible
wrong numbers from a routine whose only purpose is exactness.

Three findings, none of them anticipated:

1. **The accumulator's `@bounds` does not license the kernel — operand bounds
   do, and they are a separate thing the front end does not yet carry.**
   `@bounds(min, max)` constrains `C[i,j]`; the overflow obligation is about
   `A[i,k]` and `B[k,j]`. A bound on a sum implies nothing about its terms
   because they cancel — a result bounded by 1.0 is consistent with operands of
   1e9. So `license_vnni_exact(None, ..)` **refuses**. *This adds a front-end
   task to M0 that was not in the original scope: a way to state operand
   ranges.*
2. **The real bound at the default interval is 4095, not the 1024 in the
   probe's header comment.** 1024 is sound but 4x conservative. The derivation
   is `2 * flush_k_pairs * m^2 <= i32::MAX`; the factor of 2 is `vpdpwssd`
   doing two MACs per int32 lane, and dropping it is caught by three tests.
3. **The int16 width clamp was dead code, and only mutation testing showed
   it.** Removing `.min(OPERAND_WIDTH_LIMIT)` passed all 17 tests, because
   `products >= 2` makes the overflow bound at most `sqrt((2^31-1)/2)` =
   exactly `i16::MAX` — the two bounds coincide at the boundary rather than one
   being redundant by a margin. Replaced with a proven invariant
   (`the_derivation_can_never_exceed_int16`) that fails if the MACs-per-lane
   count ever changes. The width check survives in `license()`, where a
   caller-supplied magnitude makes it genuinely reachable.

**Landed 2026-08-16 — operand bounds reach the emitter** (`cpu_gemm.rs`:
`DriftAccumulator::{a_bounds, b_bounds, operand_bounds}`; 3 tests, 4-mutation
verified). The front-end task that finding 1 above created, closed.

- **No parser or AST change was needed.** `@bounds` is a statement-prefix
  attribute and the operands are `Stmt::Let` bindings, so
  `@bounds(min=-1024, max=1024) let a_val = block_ptr2d_load(A, i, k, K, M, K);`
  already parsed — the recogniser was simply discarding it behind a `..`. The
  bound is now stated exactly where the operand is loaded, which is also where
  a reader looks for it.
- **The two operand bounds combine by the LARGER magnitude**, because the
  overflow derivation uses one bound covering both (`m^2` per product). Taking
  the smaller, or their product, would licence a nest that can overflow;
  mutation-checked in that direction specifically.
- One bounded operand does not licence the pair, and an unbounded one is a
  refusal rather than an inherited default.

**Landed 2026-08-16 — the dispatch and its diagnostics** (`cpu_gemm.rs`:
`ExactGemmPlan`, `plan_exact_gemm`; `llvm_emitter.rs`: `try_emit_gemm_kernel`;
1 unit + 2 integration tests, 3-mutation verified).

**The design point, and it is the opposite of what the previous "Next" note
said.** That note proposed pushing an unlicensed nest's reason to `emit_errors`.
That would be **wrong and would break working programs**: a nest that cannot use
the fast path is still compiled *exactly*, because scalar lowering honours
`@ZeroDrift` by selecting an exact representation. Only the speed is lost. So
the emitter distinguishes two cases that a single error channel would conflate:

| condition | channel | build |
|---|---|---|
| no exact representation exists at all | `emit_errors` | **fails** — the guarantee cannot be delivered |
| exact representation exists, this one fast kernel is not licensed | `drift_report` | succeeds — the guarantee *is* delivered, slowly |

Both messages are user-facing and were checked by reading them, not only by
asserting on them:

```
-> @ZeroDrift matmul MxN: exact vpdpwssd kernel is LICENSED (operands |x| <= 1024,
   flush every 64 k-pairs) but not yet implemented - using scalar lowering, which
   is exact and slow
-> @ZeroDrift matmul MxN: using scalar lowering, which is still EXACT. The fast
   vpdpwssd kernel is unavailable because no operand bounds: `@bounds` on the
   accumulator states the range of the RESULT, and a bound on a sum implies no
   bound on its terms - they can cancel. ...
```

The unlicensed message has to carry both halves. Without the reason the user
cannot reach the fast path — the fix is tighter operand `@bounds`, which is not
guessable. Without "still EXACT" it reads as a dropped guarantee, which is the
opposite of the truth.

**Landed 2026-08-16 — the exact micro-kernel, and the thesis has its first
evidence** (`cpu_gemm.rs`: `emit_vnni_micro_module`, `emit_vnni_flush`;
`tests/cpu_gemm_vnni_micro.rs`; 5-mutation verified).

`emit_vnni_micro_module` emits a standalone LLVM IR module computing
`C[0..6][0..64] += A^T B` over int16 operands into int64, using real
`vpdpwssd`. Validated by compiling it with `clang` and **running it on this
host** (which has `avx512_vnni`):

| property | result |
|---|---|
| Exact vs a scalar reference, 8 sizes | **0 of 384 elements differ**, every size |
| At the licensed maximum (\|x\| = 4095, 512 pairs) | **0 differ** |
| Order independence, 6 different k-splits | **byte-identical**, every split |
| f32 control under the same splits | **39 of 384 differ** |
| `vpdpwssd` in the generated asm | 24 = `MR * NRV`, not scalarised |

**The order-independence row is the product claim, demonstrated rather than
argued.** Splitting the k-range moves the flush boundaries, which regroups the
outer additions; integer addition absorbs the regroup and f32 does not.

Four findings:

1. **The intrinsic signature is not the obvious one.** It is
   `@llvm.x86.avx512.vpdpwssd.512(<16 x i32>, <32 x i16>, <32 x i16>)` — the
   accumulator is `<16 x i32>` but the *multiplicands are `<32 x i16>`*. Writing
   `<16 x i32>` for the operands, which is what `_mm512_dpwssd_epi32`'s C
   signature suggests, produces IR that fails to verify. Derived by compiling
   `tests/probes/vnni_kernels.c` with clang and reading the `declare`.
2. **The first control was vacuous and said so.** It summed f32 in p-ascending
   order and chunked the loop — which reorders nothing, so it did not drift, so
   the order-independence result proved nothing. The reassociation is between
   the kernel's *two levels*: a narrow accumulator flushed into a wide one at
   boundaries that a split moves. The control had to mirror that structure
   before it could fail.
3. **The exactness cases could not see a missing flush.** At \|x\| = 1024 over
   512 pairs the whole reduction is 2^30, which fits int32 outright — a kernel
   with no flush at all stays exact. The `BOUND` case at \|x\| = 4095 is what
   actually exercises the int32→int64 flush, and it doubles as **empirical
   confirmation that the derived 4095 is safe on hardware** rather than only on
   paper. Mutation-checked: dropping the conditional flush fails `BOUND` and
   nothing else.
4. `sext` in the flush is load-bearing — `zext` turns every negative partial
   into a huge positive and fails 102+ elements.

**Landed 2026-08-17 — packing, a blocked GEMM, and the measurement that
overturned §1** (`cpu_gemm.rs`: `emit_vnni_pack_a`, `emit_vnni_pack_b`,
`emit_vnni_gemm_driver`, `emit_vnni_gemm_module`).

`__y_gemm_exact_vnni` computes `C += A * B` for int16 `A`/`B` into int64 `C`,
accumulating into `C` so a caller may split the k-range across calls, tiles or
threads. Validated by running it: exact against a naive integer reference at
8 shapes including odd `K`, `M`/`N` not multiples of the tile, and operands at
the licence maximum; and **byte-identical across k-splits** at three shapes.

Three findings beyond the 1.88x correction in §1:

1. **My first benchmark measured 6.3e7 G MAC/s for f32** — the compiler had
   hoisted the entire timing loop, because the kernel was `static` and every
   call identical. Exactly the trap `feedback_associativity_eats_benchmarks`
   describes. Fixed with `noinline`, a consumed output and a memory barrier;
   the corrected f32 figure then reproduced the doc's to within 3%, which is
   what made the VNNI discrepancy attributable rather than ambiguous.
2. **The flush's placement is worth 1.3x and has no correctness symptom.** Both
   placements pass every exactness and order-independence assertion in the
   suite. It needed its own guard — `the_hot_loop_does_not_spill_the_accumulators`
   asserts the flush's signature instructions (`vpmovsxdq`, `vpaddq`) appear
   nowhere in the block containing the `vpdpwssd`, mutation-verified by emitting
   a flush into the loop.
3. **The power-of-two flush interval is no longer a hardware requirement.** The
   C probe needs it for its `(p & (T-1))` test; the hoisted form steps an outer
   loop by `T` and would accept any positive value. The restriction is kept as a
   canonical form — `flush_interval_for` only ever returns powers of two — and
   the comments saying otherwise have been corrected rather than left to rot.

**Landed 2026-08-17 — M0's completion criterion is MET**
(`tests/cpu_gemm_exact_threaded.rs`, 3-mutation verified).

The packing was restructured first: A is packed once before either loop and B
once per column panel, so packing is `M*K + N*K` instead of the `M*N*K/MR` the
first version paid by repacking B for every row panel.

Then the criterion itself — **36 configurations, all byte-identical**:

| axis | values |
|---|---|
| thread count | 1, 2, 3, 5, 8, 16 (each owns a k-slice and a private `C`) |
| flush interval | 64, 512, 65536 |
| k-slice shape | even, and deliberately ragged so boundaries miss the interval |

with an f32 control over the same data and the same two-level structure
differing on **627 of 3071** elements.

**The control needed two rounds to become meaningful, and the reason is worth
keeping.** It first drifted on only 4 of 3071 — technically non-zero, so the
test passed, but a margin that thin would go to zero under another seed and
turn a real property into a flaky test. The cause was not the split shape: the
products are integers, and with `|x| <= 1024` over `K = 301` the dot products
peak near 1.7e7, while **f32 represents every integer below 2^24 (1.68e7)
exactly**. There was nothing to drift. Raising `K` to 1201 puts the sums past
that range and the control jumps to 627. *A control can be structurally correct
and still measure nothing, if the data cannot express the failure.*

**M0 is complete apart from a performance number worth quoting**, which needs
`mc`/`nc` blocking to cap the packed-A working set. The reproducibility claim —
which is the product — is demonstrated end to end.

### M1 — Sub-word element types in the PTX emitter · 2–3 weeks

Thread a byte width through the address computations and lift the refusal at
`ptx_emitter.rs:1018`.

- **Done when** — `GlobalMemory<I8>` / `<U8>` / `<I16>` / `<U16>` load, store
  and index correctly, gated by an extension of `tests/ptx_integer_datapath.rs`
  running on the real device against plain Rust over the full value range.
- **Mutation requirement** — that file's own history says a differential test
  whose two sides are the same width cannot see a conversion that became an
  identity. Width-crossing cases are mandatory, and the gate must be
  mutation-verified before it is trusted.

#### M1 progress — DONE, 2026-08-17

`U8`/`I8`/`U16`/`I16` are lowered as **buffer element types**, with the stride
threaded through `ScalarTy::log2_bytes` (0 and 1, not a hardcoded 2) and the
sign extension through `ScalarTy::mem` (`ld.global.s8` vs `u8`). Verified on the
device against a Rust reference: `tests/ptx_subword_ops.ysu` +
`the_sub_word_datapath_matches_a_cpu_reference_on_the_gpu`, 4-mutation verified.

Three things worth carrying forward:

1. **`mem()` had to split into `mem()` and `reg_mem()`.** One method served both
   `ld`/`st` (memory type) and `mov` (register type), which is fine while every
   type has a register of its own width and wrong the moment one does not —
   `mov.u8` is not a PTX instruction. Five `mov` sites had to move to
   `reg_mem()`.
2. **A sub-word LOCAL is refused, deliberately.** PTX has no sub-word register
   class, so `let x: I8` would be a 32-bit value whose declared type promises
   8-bit wraparound it will not perform. The promotion rule (load widens,
   arithmetic is 32-bit, store truncates) is C's, and is what quantized
   inference wants — but it is only sound to leave *unstated* where the width is
   unambiguously a storage format, i.e. on a buffer. `SharedMemory` is likewise
   still refused: it indexes in 16-byte units and has no byte-addressed form.
3. **The promotion's signedness was initially untested, and a mutation proved
   it.** Making `I8` promote to `U32` passed the whole file, because every load
   in the fixture was bound to an explicit `let x: I32`, which converts and
   hides what the promotion decided. It is observable only when a sub-word load
   feeds a signedness-sensitive operator *inline*; the fixture now shifts one
   right by 2, where the mutation gives 1073741792 instead of -32. Same lesson
   as this file's own header: **a differential whose operands are normalised
   before comparison cannot see the normalisation being wrong.**

### M2 — Integer tensor cores · 3–4 weeks

`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` on sm_89.

- **Done when** — the per-lane A/B/D fragment layout is **derived from the PTX
  ISA and validated on hardware**, not assumed. This is the discipline the FP8
  path required and the one the WGMMA disaster skipped.
- **Note** — `ptx_version_for_sm` may need the same bump the FP8 path needed.
  Check before assuming it assembles.

#### M2 progress — the layout is settled, 2026-08-17

**No PTX version bump is needed.** Unlike the FP8 variant, which requires
`.version 8.4` on `sm_89`, `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
assembles at **7.8 and 8.4 alike** — checked with a minimal probe rather than
assumed from the FP8 precedent. `sm_89` already maps to 8.4, so nothing changes.

**The fragment layout is derived, validated on the device, and the validation
is mutation-verified** (`tests/ptx_int8_mma_layout.rs`). With
`g = laneid >> 2` and `t = laneid & 3`:

| fragment | shape | per lane | layout |
|---|---|---|---|
| A | 16x32 int8, row-major | 4 x `.b32` | `(g, 4t)`, `(g+8, 4t)`, `(g, 4t+16)`, `(g+8, 4t+16)` |
| B | 32x8 int8, col-major | 2 x `.b32` | `(k=4t, n=g)`, `(k=4t+16, n=g)` |
| D | 16x8 int32 | 4 x `.s32` | `(g, 2t)`, `(g, 2t+1)`, `(g+8, 2t)`, `(g+8, 2t+1)` |

Every fragment's four k-values are contiguous, so each register is a single
32-bit load. **B's byte offset works out to the same expression as A's**, which
is a coincidence of these two shapes and not a shared rule — do not factor it
into one helper.

**Four mutations of the probe were run and all four produce plausible wrong
numbers rather than a crash or an assembler error** (1575, 15265 and -4544
where -29772 was expected). That is the whole argument for validating a layout
on the device: swapping two A registers or halving a stride is invisible to
`ptxas`, invisible to a substring test, and returns a well-formed matrix.

#### M2 — the emitter path is in, 2026-08-17

`@tile(M, N, K) kernel f(A: GlobalMemory<I8>, B: GlobalMemory<I8>, C: GlobalMemory<I32>)`
now lowers to a real int8 tensor-core GEMM
(`ptx_emitter.rs`: `tile_gemm_int8_operands`, `emit_int8_gemm_kernel`;
`tests/int8_gemm.ysu`; 4-mutation verified on the device).

One warp owns one 16x8 output tile and walks K; grid `(N/8, M/16, 1)`, block
`(32,1,1)`. **Fragments are loaded straight from global memory, with no
shared-memory staging, and that is deliberate for this milestone.** The layout
is the part that fails silently, and it is validated in exactly this form;
adding CTA tiling in the same step would change the addressing and leave no way
to tell which half was wrong. The kernel re-reads A and B once per output tile
— correct and reproducible, not fast.

Three things to know:

1. **No PTX version bump** — unlike FP8, the int8 mma assembles at 7.8 as well
   as 8.4. Checked, not inherited.
2. **B is `[N][K]`, not `[K][N]`.** That is what `.col` means for this
   instruction and is the natural inference weight layout, but passing a
   `[K][N]` buffer computes a different function with no diagnostic — so it is
   stated in the fixture and in the emitter's doc comment.
3. **The type checker holds a second copy of the shape whitelist.** Its own
   comment says it mirrors the emitter's dispatch chain "so type-checking and
   codegen never disagree", which makes this two implementations of one rule —
   the hazard `CLAUDE.md`'s design-rule table describes. A shape accepted there
   but unrecognised here falls through to generic scalar lowering and silently
   computes something else. Both were changed together and the coupling is now
   written down at both sites.

Shapes that do not divide the mma tile are **refused** (`M % 16`, `N % 8`,
`K % 32`) rather than padded: a partial tile needs predication on every
fragment load, and rounding the shape up would compute a different matrix than
the source asked for.

**Remaining for M2:** shared-memory staging and CTA tiling, which is a
performance change and must be re-validated against the same device test.

**Next is M3**, which is now unblocked: an exact int32 accumulator over a
tensor core is order-independent by the same argument as the CPU path, so the
batch-invariance harness can finally be written against a GPU kernel.

### M3 — The batch-invariance harness · 1–2 weeks

The product's central claim, made testable.

- **Done when** — one kernel, run under N different launch configurations
  (batch size, split-K factor, CTA tile, thread count), produces
  **byte-identical** output buffers; with a **control** asserting the f32 path
  genuinely differs under the same sweep, so the result cannot be vacuous.
- The control is not optional. A test that passes because the f32 path also
  happens to agree proves nothing — same shape as the `INTT(NTT(a)) == a`
  failure and the shared-memory barrier race that never fired.

#### M3 — DONE on the GPU, 2026-08-17

`tests/gpu_batch_invariance.rs`. **21 launches across 7 split-K geometries
(`gridDim.z` ∈ 1, 2, 3, 5, 8, 16, 32), three repeats each, all byte-identical**
and all matching the CPU reference. The f32 control, given the same partial-sum
structure and the same reorderings, disagrees on **1804 of 2048** elements.

**The kernel gained split-K for this milestone, because without it the result
would have been vacuous.** One warp per output tile over the whole K range is
trivially reproducible — nothing is being combined. The kernel now stripes K
across `%ctaid.z` and combines partials with **`red.global.add.s32`**, which is
where the claim becomes interesting:

> An atomic add is *the* canonical reason GPU results are not reproducible.
> CTAs finish in whatever order the scheduler picks, so with a float
> accumulator the identical binary on the identical input gives different
> answers between launches. Integer addition is associative and commutative, so
> the same atomic cannot care. Nothing is serialised, no ordering is imposed,
> and no determinism flag is set.

Striped rather than blocked partitioning means **any** `gridDim.z` divides the
work with no divisibility precondition, which is what lets the harness sweep the
split factor without recompiling. `C` must now be zero-initialised by the
caller, since the kernel reduces into it rather than overwriting.

Mutation-verified — overwrite-instead-of-reduce and an overlapping partition
are both caught. One mutation (`kstep` from `ctaid.z` rather than `nctaid.z`)
**hung the GPU**, because CTA 0 then steps by zero; a mutation that makes the
kernel not terminate is a bad choice of mutation, not a finding, and was
replaced with one that perturbs the partition without risking a zero stride.

### M4 — Deterministic attention, softmax and norms · 4–6 weeks

Fixed reduction trees where exactness does not apply. The split-K decode
attention kernel is the hard case: its cross-split combine is exactly where
batch-dependent nondeterminism enters.

- **Done when** — decode output is bit-identical across `splits` and `warps`
  settings and across batch sizes.

#### M4 step 1 — the reduction claim holds on attention, 2026-08-17

**This milestone's own framing above was wrong, in the favourable direction.**
It assumed exactness does not reach attention and that the fallback is a fixed
reduction tree — i.e. the same weaker claim a competitor constraining f32
ordering would make. It does reach attention. `tests/gpu_attention_invariance.rs`
demonstrates attention whose every reduction over the sequence is
order-independent *by construction*, on the device.

The structure is two-pass, with no online softmax state:

| step | operation | why order-independent |
|---|---|---|
| 1 | `s_i = q · k_i` | int8 × int8 → int32, exact |
| 2 | `m = max_i s_i` | integer max — associative *and* commutative, exactly |
| 3 | `p_i = round(2^16 · 2^(C(s_i - m)))` | **per element**; not a reduction |
| 4 | `l = Σ p_i` | integer sum, exact |
| 5 | `o_d = Σ p_i · v_id` | integer sum, exact |
| 6 | `out_d = o_d / l` | one float divide, at the very end |

Step 2 is the load-bearing one: knowing the global max *before* any
accumulation begins is what removes the `exp(m_old - m_new)` rescale that a
FlashAttention-style kernel applies once per tile. That rescale is the
mechanism of batch-dependent nondeterminism — its *sequence* changes when the
tiling changes, and every server tunes `splits` and `warps` per batch size.

Steps 4 and 5 run as `red.global.add.u64` — a fire-and-forget atomic, the
canonical source of GPU irreproducibility — and are bit-identical anyway.
Nothing is serialised and no determinism flag is set.

Measured: **27 launches over 9 geometries, bit-identical**, sweeping three
independent axes of the partition (`blockDim.x` 32–256, `gridDim.x` 1–8,
`gridDim.z` 1–32; 32 to 4,096 workers). Agrees with an f64 softmax to
3.47e-4 absolute on values in ±127. The control — the same attention as an
f32 online softmax with split-K — moves on **742 of 768** (element, split)
pairs, so the result is not a statement about benign data.

**The cost is one extra pass, and it is cheap in the shape that matters.**
Pass 2 re-reads `s` at 4 bytes per key, not `K` at `head_dim` bytes per key —
about 3% of the traffic at `head_dim = 128`, in a regime that is DRAM-bound.
Not yet measured; see the open items below.

**The exp is a per-element function and therefore outside every claim made
here — but it is inside M5's.** `ex2.approx.f32` is a fixed-function unit:
same input, same output, every launch, on a given architecture. That is
sufficient for batch and partition invariance, because step 3 has no
reduction in it. It is *not* sufficient for M5's "same answer on two
different GPUs", where a different SM generation may round it differently.
Closing that needs an integer exp (fixed-point table plus polynomial), which
is architecture-independent by construction. **DONE 2026-08-17** — see M4 step
1e; `src/fixed_exp.rs` is 8.8x more accurate than the hardware unit and agrees
with the device bit for bit. The claim is still limited to
"architecture-independent by construction, verified on one device" until a
second GPU is available to run it on.

Mutation-verified, six mutations, all caught: a partition that drops
`nctaid.x`, one that drops `ctaid.z`, `max` → `min` on the global reduction,
overwrite-instead-of-reduce on each of `l` and `o`, and **accumulating the
numerator in f32** — the product claim's own negation. Two independent
assertions carry the file: a serial host sum over the device's own `p` (which
pins the reductions exactly, so "all geometries agree" cannot pass by all
being wrong together), and the geometry sweep. The `ctaid.z` mutation is
caught by the *sweep*, which is the evidence the sweep has teeth rather than
riding on the oracle.

One guard earned its place immediately: the temperature `C` is a hand-picked
constant, and the first value was chosen from an *estimated* score spread of
-1,500 when the real spread is -92,000. The softmax collapsed to 15 distinct
weights out of 256 — every reduction below it would have been measuring
nothing, and every invariance assertion would still have passed. The
degeneracy assertion caught it; the constant is now derived from a measured
spread.

**Not done, and not claimed:** this is a standalone PTX probe, not a path
through the emitter — no `.ysu` surface, no paged KV cache, no GQA head
sharing, and no throughput number. Those are M4 steps 2–3.

#### M4 step 1b — exactness survives optimisation, 2026-08-17

The obvious objection to the above is that determinism is easy if you write
the kernel badly: one global atomic per `(key, element)` is 64 atomics per key
and nothing like a real kernel. So the accumulation is now a **three-level
reduction** — a plain register add for `l`, `red.shared.add.u64` into a
CTA-private shared buffer for `o`, then one global atomic per CTA per output
element — and the naive version is kept beside it.

Both are run at all nine geometries, and **they agree bit for bit with each
other and with the reference**. That is a stronger statement than either being
self-consistent: the two group the terms completely differently, and at every
block size the tree has a different *shape*. Nothing about the order is pinned
down anywhere — the shared atomics interleave in hardware-chosen order exactly
as the global ones do. The kernel is order-independent because integer
addition is.

**A mutation survived, and it is the one this repo has already been burned
by.** Deleting the barrier between zeroing the shared accumulators and
accumulating into them leaves the entire file green, five runs of five. The
race is real, but its window never opens: between the zero loop and the first
`red.shared.add` sit a global load of `Scores`, an `ex2` chain and a global
store of `P`, so every warp finishes zeroing long before any warp reaches an
accumulate. Identical in shape to the `bar.sync` finding in gotcha #8 — **a
race that does not fire is not a test** — and the same answer applies:
`the_two_level_reduction_is_ordered_by_real_barriers` asserts the ordering in
the emitted PTX. The guarantee is absent rather than unused, and a larger `D`
or a scheduling change would open the window. The *other* barrier (before the
global flush) is caught on the device, 0/5. Two further mutations — a nonzero
seed for the shared accumulators, and a flush stride of 1 instead of
`blockDim.x` — are also caught 0/5.

#### M4 step 1c — what exactness costs in decode, measured, 2026-08-17

§6 item 3 is the open question: the 1.88x-style figures are compute-bound
micro-kernels, while decode is DRAM-bound, and "3%" had been asserted from a
byte count and never run. `tests/gpu_attention_cost.rs` runs it.

The pair is the attention **accumulation phase** in the layout a decode kernel
actually uses — one thread owns one output dimension `d` and walks the key
range, so `V` is perfectly coalesced and `o` needs no cross-thread reduction.
Identical layout, identical partition, identical bytes read. The only
difference is the inner loop: `acc += p*v` against the genuine
FlashAttention `acc = acc*corr + p*v` with a running max and two `ex2`.

RTX 4070 Ti SUPER, peak DRAM 672 GB/s, head_dim 128, seq 4096, 32 rows
(17.3 MB per pass):

| splits | exact µs | GB/s | %peak | f32 µs | GB/s | %peak | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 560.5 | 30.9 | 5% | 650.1 | 26.6 | 4% | 0.86x |
| 4 | 134.6 | 128.5 | 19% | 160.6 | 107.7 | 16% | 0.84x |
| 8 | 67.5 | 256.4 | 38% | 75.8 | 228.2 | 34% | 0.89x |
| 16 | 33.3 | 519.7 | 77% | 39.9 | 434.1 | 65% | 0.84x |
| 32 | 30.3 | 571.2 | 85% | 37.4 | 462.7 | 69% | 0.81x |

> **Correction, same day.** The first version of this section read the table
> above as "exactness is 11–19% FASTER". That was **my own baseline being
> unfair to itself**, caught by re-checking rather than by a test. In this
> layout all 128 threads of a block redundantly recompute the same per-key
> `max`/`corr`/`p`, and worse, the f32 path is charged for `ex2` work that the
> exact path had already done in the *untimed* score pass. The `online` column
> below is kept for the record but **must not be quoted**.

The decisive control is a **fair twin**: an f32 kernel structurally identical
to the exact one — same precomputed `p`, same loads, no `ex2`, no running max
— so the only difference is an integer accumulator against a float one.

**Exact is 1.01–1.02x the twin: parity, and if anything a hair slower.**
Stable at every split count, so it is signal rather than noise.

So the honest accounting for exact decode attention is:

| component | cost |
|---|---|
| accumulation arithmetic (int vs float) | **1.01–1.02x** — free |
| the separate score/max pass | `4/D` extra traffic = **3.1%** at head_dim 128 |
| the online rescale exactness deletes | real, but a tiled flash kernel amortises it per tile, so small |

**Total: about 3%, which is exactly what the pre-measurement prediction said.**
The prediction was right; the measurement briefly claimed better than the
prediction and was wrong. Against the "within ~10% of FlashInfer is table
stakes" bar in §2 this passes with room, but it is *parity plus a traffic
premium*, not a win, and nothing downstream should be built on a win.

What the bandwidth column does establish: the exact path reaches **84–85% of
DRAM peak**, so this phase is genuinely bandwidth-bound and the arithmetic is
not the constraint — which is *why* the integer and float accumulators tie.

**Not measured, and therefore not claimed:** a production fused-score flash
kernel, which computes the scores in the same pass and keeps them in
registers. The 3.1% above is a byte count, not a timing. **No end-to-end ratio
against FlashInfer is quoted and none should be** until that kernel exists.

The device-side control is in the same file and is what makes the comparison
mean anything: run at 6 split counts, the exact kernel moves **0 of 20,480**
outputs and the f32 kernel being timed moves **19,970 of 20,480**. The
baseline demonstrably has the defect exactness removes.

#### M4 step 1d — the accuracy question, and a real bug it found, 2026-08-17

§6 item 1 says to test accuracy at M0/M3, not at M5. A full task-accuracy run
is blocked on M4 steps 2–3 (no emitter path, no paged KV, no model). What
*could* be tested now is the part that is **this project's design choice**
rather than quantization's: the softmax weights held as `Q0.F` fixed point.

`tests/attention_quantization_error.rs` states the bar as **"is the exact path
more wrong than the f32 online softmax production flash attention already
ships?"** Both read the same int8 `V`, so int8's own error is common and
cancels.

**It failed, badly, and the design was wrong.** At the hand-picked `F = 16`,
the *attention sink* shape — one key ~18 logits above the rest, ubiquitous in
real transformers — came out **2109x worse than f32 flash**. Cause: every
non-sink weight is `2^-18 · 2^16 = 0.25` and rounds to **zero**, so the entire
tail (~1.5% of the softmax mass) disappears and the output is just `v[argmax]`.
Not an exotic corner; the single most common shape in a real model.

**F is now derived, not picked.** The sweep shows no knee — error falls
monotonically to F=30 — so the width is set by the **range obligation**
instead: `p ≤ 2^F` must fit an i32, `p·v` needs `F+7` bits, and summing `S` of
them needs `F+7+log2(S)` inside the i64 accumulator. **F = 28** leaves three
bits of i32 headroom and supports sequences to 2^28 keys.

| F | worst error | vs f32 flash |
|---|---|---|
| 8 | 5.043e0 | 25268x |
| 16 | 1.588e-1 | **796x** |
| 24 | 1.453e-4 | 0.73x |
| **28** | **1.288e-5** | **0.06x** |
| 30 | 2.369e-6 | 0.01x |

At F=28 the exact path is **0.01–0.20x** flash's error across all five shapes
(Diffuse, Moderate, Peaked, Sink, HeavyTail) and three sequence lengths — i.e.
**5–100x more accurate than what ships**, because it accumulates exactly and
flash does not. On device, agreement with an f64 softmax went 3.47e-4 →
**6.62e-7**.

**The cost of the fix is negative.** `p·v` is 35 bits at F=28, so `mul.wide.s32`
replaces `mul.lo.s32` + `cvt.s64.s32` — one instruction *fewer*. Re-measured:
determinism unchanged at all nine geometries, twin ratio still 0.99–1.02x, peak
still 85% of DRAM.

**Ranking the two error sources**: one int8 step in `V` costs **90,000x** more
than the Q0.28 softmax representation. So the design's own error is nowhere
near binding, and the remaining accuracy question is int8 quantization itself —
the customer's existing decision, not this project's.

**Still not answered, and §6 item 1 stays open:** no model, no calibration, no
activation outliers, no per-channel scales, no task score. These are synthetic
distributions chosen to *include* hard shapes, not observed ones. This bounds
the numerics; it does not close the kill-criterion.

#### M4 step 1e — the cross-GPU gap is closed, 2026-08-17

Every step above carried a caveat: `ex2.approx.f32` is a fixed-function unit
whose rounding belongs to the SM generation. Deterministic per launch and per
architecture — enough for batch invariance, since the step is per-element and
cannot see the partition — and **not** enough for M5's "the same across two
different GPUs". The docs said not to quote a cross-GPU claim until an integer
exp existed. It exists.

`src/fixed_exp.rs` computes `2^-t` for Q16.16 `t` in pure integer arithmetic:
split `t = n + f`, index a 64-entry table with the top six bits of `f`, and
handle the low ten with `1 - y + y²/2 - y³/6` where `y = δ·ln2`. `δ < 2^-6`
gives `y < 0.0108`, so the first omitted term is 0.15 ulp — three terms are
needed and four would be waste, which is pinned by a test that drops `y³/6`
and asserts the error explodes (it costs 57 ulp).

| | worst error vs f64 | portable? |
|---|---|---|
| `ex2.approx.f32` → Q0.28 | **8.02 ulp** | no — SM-generation dependent |
| `fixed_exp` (integer) | **0.91 ulp** | yes — integers are specified |

**It is 8.8x more accurate *and* architecture-independent** — not a trade. The
two differ on 61,372 of 65,536 arguments, so the exposure being removed is
real rather than theoretical.

**`tests/ptx_fixed_exp.rs` is what makes portability a checked property**: a
PTX transcription against the Rust reference, **bit for bit on all 100,427
arguments** — every fractional value exhaustively, every integer part, and the
saturation boundary. Two independent implementations of one integer recipe,
agreeing exactly. Mutation-verified: dropping `y³/6`, truncating instead of
rounding, indexing the table with the wrong bits, and dropping `y²` are all
caught.

Dropped into the attention accuracy harness it is a clean substitution —
**0.00–0.21x f32 flash's error** across all five shapes and three lengths,
matching or beating the `ex2.approx` version everywhere. The conversion into
the exp's argument is a **shift, not a multiply** (`t = (m-s)·2^-13`, so
Q16.16 is `(m-s) << 3`), so the softmax path now contains no floating-point
operation at all until the final divide.

**What this does and does not license.** The *arithmetic* is now portable by
construction. A full cross-GPU claim still needs the whole kernel run on two
different architectures and compared — this repo has one GPU, so that remains
untested, and M5 should say "architecture-independent by construction,
verified on one device" until a second one exists.

#### M5 step 1 — real activations, and a claim that does not replicate, 2026-08-17

`tools/attention_real_activations.py` captures **post-RoPE Q/K/V from a real
model** (Qwen2.5-0.5B-Instruct, 24 layers, 14 Q / 2 KV heads) on real text and
reruns M4 step 1d's comparison on it, decode-shaped (T ≈ 660–710, attending
from the last position over the whole cache).

| quantity | median, relative |
|---|---|
| softmax representation (Q0.28 + integer exp) | **6.6e-7** |
| f32 tiled flash, same int8 V | ~1e-7 – 1e-6 |
| **int8 V, per-TENSOR scale** | **1.96e-1** |
| **int8 V, per-CHANNEL scale** | **1.15e-2** |

**Three things follow, and the first is a correction.**

1. **"The exact path is 5–100x MORE accurate than f32 flash" was a synthetic
   result and does NOT replicate on real activations.** On real data the ratio
   runs 0.2x–8.6x, i.e. sometimes better and sometimes worse. Both sit around
   1e-6 relative, five orders of magnitude below the quantization error beside
   them, so the honest statement is **"indistinguishable, and both irrelevant"**
   — not a win. The synthetic distributions were chosen to include hard shapes,
   which turns out not to be the same thing as being hard in the way real ones
   are. Do not quote the 0.01–0.20x figure for real workloads.

2. **The softmax representation is a non-issue.** 6.6e-7 relative is nowhere
   near binding. Q0.28 and the integer exp are settled; the F=16 bug M4 step 1d
   found was the only real risk there and it is fixed.

3. **§6 item 1 is real, and it is about the QUANTIZATION SCHEME, not about
   determinism.** Per-tensor int8 on V costs **19.6%** relative error;
   per-channel costs **1.15%**, a 17x improvement from a change that has nothing
   to do with this project. int8 V outweighs the entire exact-softmax path by
   ~300,000x. So a demo must use a real quantization scheme (per-channel at
   minimum; SmoothQuant/AWQ-class in practice) — and if the buyer rejects the
   accuracy, they are rejecting int8, not exactness.

**Two harness bugs, the second a repeat offence.** The first run reported the
exact path 10^6x worse than flash. That was not a finding: the baseline was
given exact fp32 logits while the exact path derived its scores from int8 q·k,
so the comparison charged one side for quantization the other skipped — the
**same bias class** as the `ex2` accounting error corrected in M4 step 1c. And
the first prompts were 13–21 tokens, where a flash kernel barely reassociates
anything; that is the regime least like decode and least favourable to the
thesis. Both fixed before any number above was recorded.

**Still not done:** no tokens generated, no serving path, no throughput, no
task-accuracy score. This measures one layer's attention on real inputs. It
narrows §6 item 1 to "pick a real quantization scheme"; it does not close it.

#### M5 step 2 — the demo works, 2026-08-17

`tools/batch_invariance_demo.py`. A real model (Qwen2.5-0.5B-Instruct), one
prompt replicated to fill each batch — so row 0 is the identical computation
every time and anything that changes is the kernel's reduction order following
the batch size.

```
--- stock  (bf16 + SDPA, what production serves) ---
  prompt 0  b8: DIVERGES@157  b32: DIVERGES@6
      batch  1: ' ... written in the first person and include at least three distinct characters'
      batch 32: ' ... written in third person and include at least one character who is not'
  prompt 1  b8: DIVERGES@39   b32: DIVERGES@39
      batch  1: ' Octopuses are capable of using echolocation to navigate and locate prey.'
      batch  8: ' Octopuses are capable of moving through water using their arms and legs'
  => 3/3 prompts produced different text

--- exact  (int8, order-independent reductions) ---
  => 0/3 prompts produced different text
```

Not cosmetic: *first person* vs *third person*, *echolocation* vs *arms and
legs*. Different answers to the same question, from the same weights.

**Three reductions had to be replaced, not one.** Attention, the **169 linear
layers** (a GEMM picks its tiling from the problem size, so the summation order
follows the batch) and the **49 RMSNorms**. `tools/exact_model.py` does the last
two: W8A8 with per-channel weight and per-token activation scales, accumulated
in exact int32 via `torch._int_mm`, falling back to float64 with a 2^53 range
assertion — float64 represents every integer below that exactly, so no partial
sum rounds and every order agrees. Same associativity argument as the integer
kernels, carried by a different type.

**Fixing attention ALONE made it 10,000x worse**, and this is the trap worth
recording. Max logit delta at batch 32: stock **3.8e-5**, exact-attention-only
**3.9e-1**, everything-exact **0**. Quantisation is a *step function* —
`round(q/s)` near a boundary turns a 1e-5 upstream wobble into a full int8
jump. **Partial determinism is worse than none**: it amplifies exactly the
noise it was meant to remove. Ship the whole chain or none of it.

**The control has to be bf16, and the first version of this demo got that
wrong.** In fp32 the batch-dependent logit delta is ~4e-5 — real (99% of logits
differ) but below a greedy argmax's decision margin, so the *tokens* matched
and the demo appeared to prove nothing. In bf16 the delta is **0.34** against a
typical top-2 margin of **0.25**: the noise exceeds the margin and the text
moves. Nobody serves fp32, so bf16 is also the honest comparison.

**Scope, stated plainly:** this is **torch, not the PTX kernels** — it
demonstrates the property end to end, while
`tests/gpu_attention_invariance.rs` is the fast path and was measured
separately (~3% overhead, 85% of DRAM peak). **No throughput number for this
configuration**; the float64 fallback is slow and was never meant to be
serving code. One 0.5B model, greedy decode, three prompts. A larger model and
a real serving integration are still ahead.

#### M5 step 3 — the gaps, closed and measured, 2026-08-17

**The kernel is the compiler's now, not a test's.** The exact-attention PTX
moved from inside `tests/gpu_attention_invariance.rs` into
`src/exact_attention.rs`, and `Y --emit-attention-ptx <head_dim> <seq_len>`
prints it. One copy feeds the device tests, the CLI and the Python bridge — a
kernel that exists only in a test file cannot honestly be called compiler
output.

**The temperature is a runtime parameter now, and had to be.** The kernel
converted a score to the exp's Q16.16 argument with `shl 3`, which is correct
only for `C = 2^-13` — the value the synthetic tests use. A real model's
softmax scale is `q_scale * k_scale / sqrt(d)`, an arbitrary number, so the
kernel was usable on synthetic data *only*. It now takes `KFix` and computes
`t = ((m - s) * KFix + 2^15) >> 16` in 64-bit. The tests pass `8 << 16`, which
is exactly the old shift, so their semantics are unchanged.

**`tools/ptx_bridge.py` closes the "torch, not the kernel" gap.** It loads the
compiler's own PTX through the CUDA driver API into torch's context and
launches it on **real post-RoPE Q/K/V captured from Qwen2.5-0.5B**, then checks
against the torch path:

```
layer/head pairs checked : 12
bit-identical            : 12
mismatched               : 0
```

Four layers x three heads, `p`, `l` and the accumulator all compared as
integers. **The demo's numbers are the kernel's numbers.**

> **Stale as measured, 2026-08-19.** That run passed `KFix = C * 2^16` — the
> torch demo's multiplier — where the kernel needs `C * 2^32`, because the
> kernel consumes one factor of `2^16` in its `>> 16`. Every exponent came out
> 65536x too small, i.e. a **uniform** softmax, and both arms of the comparison
> shared the mistake so 12/12 was reported anyway. The convention is fixed and
> pinned by CPU tests, and a `max(p)/mean(p)` control now refuses a uniform
> result, but the bridge has not been re-run — it needs the GPU. See
> `docs/bit_identical_decode.md` finding 06.

**Throughput, stated plainly because it was unflattering: 6.7x slower.**
Qwen2.5-0.5B, 64 tokens, batch 32, 24.7 tok/s/seq against stock's 165.9. That
was recorded as "the prototype's cost, not exactness's" — true, and useless,
because it named no cause and pointed at no fix. An unattributed 6.7x is
indistinguishable from an excuse. See the next section.

#### M5 step 4 — closing the throughput gap, 2026-08-17

**6.7x → 1.43x, and not one of the four causes was exactness.** Every change
below is verified BIT-IDENTICAL to the float64 path it replaced
(`tools/exact_selftest.py`), because both are exact integer arithmetic and so
anything but equality is a bug rather than a tolerance question.

| | tok/s/seq, stock | tok/s/seq, exact | ratio |
|---|---|---|---|
| before | 165.9 | 24.7 | 6.7x |
| after, eager | 176.9 | 38.0 | 4.66x |
| after, `--compile` (**both arms**) | 251.2 | 175.3 | **1.43x** |

`tools/exact_throughput.py` runs it. **`--compile` is applied to the stock arm
too** — compiling only the challenger is the same bias already caught twice in
this project (the f32 baseline charged for work the exact path did untimed; the
baseline given exact fp32 logits against a quantised candidate). A speedup over
a handicapped control is not a speedup.

The ratio holds across batch size, so it is a property and not one lucky point
(compiled both arms, 64 tokens, best of 3):

| batch | stock tok/s/seq | exact tok/s/seq | ratio |
|---|---|---|---|
| 1 | 328.5 | 224.8 | 1.46x |
| 8 | 286.1 | 208.1 | 1.37x |
| 32 | 251.2 | 175.3 | 1.43x |

(Batch 32 is **1.06x** on full `generate` and **1.01x** on decode alone after
the NCU and tuning work in steps 5-9 below — and **0.97x** on decode device
time, i.e. ahead of stock on the GPU itself.)

The four causes, largest first:

1. **Three safety assertions were 61% of attention time.** `assert_exact_range`
   did `float(x.abs().max()) < 2^53` — a device-to-host copy, three per call,
   each stalling the pipeline. 1049 us/call with them, 404 us without. The
   replacement is *stronger*, not weaker: the bounds are derived from shapes and
   dtypes (`|score| <= d*127^2`, `l <= T*2^28`, `|acc| <= T*2^28*127`), so they
   hold for every input where a measurement only ever spoke for the tensor in
   front of it. `Y_EXACT_CHECK=1` runs the measured form beside the derived one
   and asserts the derivation is not merely satisfied but *correct*. **A safety
   check that synchronises is a performance bug wearing a correctness costume.**
2. **The `_int_mm` guard never fired once.** `ExactLinear` gated the integer
   tensor-core path on `M % 8 == 0`; the real rule is **`M % 32 == 0`**
   (measured: 32/64/96/128/256 accepted, 1–31, 33–47, 48 and 56 refused). At
   decode `M` is the batch size, so every linear layer in every step fell
   through to the float64 fallback — which runs at **1/64 rate** on a GeForce
   part, 24–27x slower than fp32 at these shapes. Padding the row count to a
   multiple of 32 is exact (a zero row gives a zero row; matmul rows are
   independent) and `check_pad_is_load_bearing` fails if a future torch accepts
   M=1, so the comment cannot go stale silently.
3. **`repeat_interleave` to widen K/V to the query heads: 2.0 ms per call.** The
   GQA group shares one K and one V, so the expansion quantised 7 identical
   copies. Folding the query group into the row dimension instead removes it
   entirely — the same restructuring the `paged_decode_attention_split` kernel
   already does on the device side (CLAUDE.md gotcha #5).
4. **float64 where the bound did not require it.** Scores fit fp32 exactly
   (`d*127^2 = 1.03e6 < 2^24`, and so does every partial sum, so there is no
   rounding and hence no order dependence); the `exp2` chain is int64. The one
   product that genuinely does not fit is `p @ V` — a Q0.28 weight times an int8
   activation is 35 bits before summing — so `exact_pv` splits `p` into digits
   narrow enough that each partial matmul is exact in fp32 and recombines them
   in fp64. 949 us → 125 us. `check_pv_bound_is_load_bearing` forces a too-wide
   digit and asserts the answer really does break, so the derivation is not
   decorative.

**Then `torch.compile`, which is the honest remaining answer.** After the four
fixes the path is **launch-bound, not arithmetic-bound**: ~7 us per op on
35K-element tensors, dozens of ops per module. Compiling fuses the elementwise
chains — attention 416 → 89 us (4.67x), `ExactLinear` at M=1 267 → 36 us
(7.46x) — and is bit-identical in both cases. Stock gets the same treatment and
also speeds up, which is why the ratio only falls to 1.43x rather than to
parity.

**What 1.43x is and is not.** It is a torch prototype against a torch baseline,
both compiled, on one 0.5B model. It is **not** the cost of exactness: the
kernel-level figure remains ~3% with a twin control (M4 step 1c), and the PTX
kernels still are not in the serving path. The residue is per-token
quantisation and op count in Python. **Do not quote 1.43x as the cost of
determinism and do not quote 3% as an end-to-end result** — but 1.43x is close
enough to argue from, where 6.7x was not.

One incidental: TorchInductor's C++ link step does not quote its library path,
so this repo's `NVME files` directory breaks compilation with
`cannot find -ltorch_cpu`. `tools/_nospace.py` re-execs through a `/tmp`
symlink.

#### M5 step 5 — what NCU said, 2026-08-17

**1.43x → 1.32x**, and the interesting part is the two things that did *not*
work. Profiled with `ncu --set basic` over two **steady-state decode steps**,
prefill excluded — `tools/ncu_workload.py`. Excluding prefill was the first
correction: at batch 32 / prompt 64 a prefill GEMM has M = 2048 against decode's
M = 32, and blending them reported a decode GEMM at 76.8 us that actually costs
12.

**Finding 1: the int8 GEMM was throwing its own advantage away.**

| | exact | stock |
|---|---|---|
| GEMM total | 4803 us | 4957 us |
| GEMM **% of DRAM peak** | **36.1%** | **69.5%** |

int8 weights are half the bytes of bf16, so the exact arm should have been
roughly twice as fast on the linears. It was level, because `torch._int_mm`
dispatches a CUTLASS kernel built for large M and at decode M is the batch
size. A Triton replacement (`tools/w8a8_gemv.py`, int32 `tl.dot`, so still
exact and still order-independent) is **1.4–4.9x** faster standalone at every
shape from M=1 to M=4096.

**Finding 2: that 1.4–4.9x kernel bought ZERO end-to-end, and NCU is the only
reason that was understood rather than argued about.** Re-profiling showed the
GEMM down 658 us and a *new* 1449 us kernel where there had been none: taking
the linear out of inductor's graph turned a fused norm+quantise (830 us) into a
bare norm (409 us) plus a standalone quantiser. **The fusion that was broken
was worth more than the GEMM that was fixed** — 1.43x to 1.42x, i.e. nothing.
Quantising in torch and calling Triton for the GEMM *only* is what actually
paid: **1.32x, 189.0 tok/s/seq against stock's 250.1.** A faster kernel is not
a faster model; check what it was fused to.

**Two exactness bugs, both found by the self-test, both of the same species.**
Neither is a logic error — each is the toolchain quietly substituting a
*different* floating-point operation, which is the exact hazard this project
exists to remove:

- **Triton's default float divide is the approximate one.** `x / s` came out as
  exactly -22.5 where IEEE `div.rn` gives -22.5000019, so round-half-to-even
  answered -22 against torch's -23. **One element in 458,752** — invisible at
  decode shapes, and it matters because quantisation is a step function, so a
  1-ulp divide becomes a whole int8 level and then a different token.
  `tl.fdiv(..., ieee_rounding=True)` does **not** fix it; `libdevice.div_rn`
  does. `check_quantiser_at_volume` now runs 12.3M elements, because the bug
  needs volume to surface.
- **`acc * sx * sw + bias` was contracted into an FMA**, rounding once where the
  reference rounds twice — 25% of elements off by 1 ulp. `libdevice` exposes no
  `fadd_rn`, so the bias add is done in torch, where it cannot contract. Only a
  layer *with* a bias can catch this, which is why every no-bias shape passed
  while q/k/v were wrong.

**The next bottleneck is named and measured, not guessed:** attention's
quantisation, ~29% of decode, in kernels running at 8.9% of DRAM peak and 12
instructions per thread — latency-stalled on almost no work. The cause is
structural rather than a tuning problem: **K and V are re-quantised in full on
every decode step** although the cache grows by one row, so the work is O(T) per
step where it should be O(1). Fixing it means an int8 KV cache with per-row
scales, which changes the arithmetic and so needs its own accuracy pass — a
design change, not a patch. It was attempted; see step 6.

#### M5 step 6 — the O(1) KV cache: right, and unusable, 2026-08-17

**1.32x → 1.31x, and the accuracy improved 1.7x.** The scheme change landed; the
O(1) cache it was supposed to enable did not, for a reason worth more than the
speedup would have been.

**The scheme.** K and V now carry **per-token** scales instead of one scale per
head over all `(T, d)` and one per feature channel over all `T`. A row's
quantisation then depends on that row alone, which is the property an
append-only cache needs. The monotone-max trick — rebuild only when a scale
actually grows — rescues K and **fails for V**: with `b*nkv*d` = 4096
independent per-channel maxima, the expected number that move at step `t` is
`4096/t`, so some channel moves nearly every step and a full pass happens
anyway.

**Exactness survives, via algebra rather than assertion.** A per-token `sv` does
not factor out of `sum_t p[t]·sv[t]·v8[t,d]`, which is what makes the obvious
version of this idea silently stop being exact. It is recovered by folding `sv`
into the weight and requantising to one common scale: `W[t] = round(p[t]·sv[t]·
2^28 / max_t(p·sv))`, leaving the numerator an exact integer sum times a scalar,
and the denominator (which uses `p`, not `W`) untouched.

**The accuracy pass says the change was worth making on its own.** Measured on
real post-RoPE activations against an fp64 softmax oracle over the *original*
fp32 K/V (`tools/exact_accuracy.py`, 21 layer/head pairs):

| path | median rel. error |
|---|---|
| old (per-head K, per-channel V) | 2.70e-2 |
| **new (per-token K and V)** | **1.60e-2** |
| f32 online softmax, same int8 operands | 1.60e-2 |
| fp64 softmax over int8 K/V (quantisation alone) | 1.60e-2 |

**0.59x the error**, and identical to three decimal places to the
quantisation-only figure — so the fixed-point softmax contributes ~4e-7 where
int8 contributes 1.6e-2, about 40,000x smaller. A control asserts the four paths
are not accidentally the same code, since agreeing that closely is otherwise
indistinguishable from a bug.

**The cache is off by default, and this is why.** It does strictly less work and
loses under `torch.compile` every way it can be wired in, because it is stateful
and the compiler is not:

| | ms/step, batch 32 |
|---|---|
| stateless, in-graph | **5.24** ← shipped |
| cache, visible to dynamo | **2882** |
| cache, `@torch._dynamo.disable` | 27.8 |
| stateless, `@torch._dynamo.disable` | 27.3 |

`self.t` is a Python `int` that changes every decode step and dynamo guards on
it, so the whole model **recompiles once per token** — 550x, and it presents as
a hang rather than a slowdown. Hiding the cache from the tracer instead costs a
graph break per attention call (3 compiled frames → 51) which is ~5x worse than
the O(T) quantisation it was removing. **An asymptotically better algorithm lost
to the compiler twice over.** Same lesson as step 5's fusion, one level up.

The code and its tests stay: it is the right structure for a serving path that
is not a traced graph — the PTX kernels — where there is no dynamo to lose to.
`Y_EXACT_KVCACHE=1` enables it, and `check_kv_cache_matches_stateless` proves 80
incremental appends equal quantising from scratch, plus both invalidation rules
(shape change, `T` going backwards).

#### M5 step 7 — the fp64 quantiser, and a harness that was lying, 2026-08-17

**Device time 12.097 → 10.161 ms; ratio to stock 1.33x → 1.12x.** The profiler
was asked what the bottleneck was rather than guessed at, and the answer was one
dtype.

`quantize_rows` widened K and V to **float64** before dividing. fp64 is 1/64
rate on a GeForce part, and NCU had the kernel at **SM 61.2% with DRAM at 8.9%**
— saturating the fp64 divide pipe on 270,336 elements, not moving memory.
Isolated on the real shape: **68.71 us in float64 against 12.27 us in float32,
5.60x.** In the model:

| kernel | float64 | float32 | SM% |
|---|---|---|---|
| K/V quantise | 2132 us | **510 us** | 61.2 → **6.8** |
| its amax partner | 705 us | 485 us | 49.7 → 4.7 |

The float64 existed to preserve bit-identity with the *original* reference,
which step 6 had already replaced — so it was protecting nothing. Its cost is
measured, not argued: the two disagree on **one int8 value in 270,336**
(0.0004%), the same divide-boundary effect as the Triton `div_rn` bug, and the
accuracy table above is unchanged (`new` still 1.60e-2, still 0.59x the old
scheme, still equal to quantisation alone). Determinism is untouched: an
elementwise divide is reproducible in any precision — it is *reassociation* and
*unspecified* operations that break batch invariance, not narrowness. The logit
and exp arithmetic stay in float64, where the tensor is `rep`x smaller and
precision actually matters.

**A measurement caught the harness lying, which is the more useful half.** The
first run after this change reported 1.19x — and the *stock* arm had moved
250.7 → 223.7 tok/s/seq, 12%, because an unrelated process had taken the GPU.
`exact_throughput.py` ran stock to completion and then exact, so any drift —
thermal, clock, another tenant — landed as a difference between the things being
compared. It interleaves the arms round-robin now and prints the median-minus-
minimum spread, with a warning above 5%. Interleaved: **1.09x wall clock**
(spread 1.1% / 1.4%) on a machine that was still busy, against **1.12x by NCU
device time**, which is immune to other tenants because it serialises kernels.
Quote the device-time figure; the wall clock's absolute throughputs were
depressed on both arms.

**The bottleneck is now unambiguous and singular: `_w8a8_gemm`, 4224 us, 41.6%
of decode, at SM 11.8% and ~35% of DRAM peak.** It already beats stock's bf16
GEMM in absolute terms (4224 vs 4957 us). Step 8 tried to close it.

#### M5 step 8 — the GEMM, and two measurements that lied, 2026-08-17

**1.12x → 1.07x**, from autotuning the GEMM. The other two things tried both
failed, and one of my own measurements was the cause of a false result.

**Autotuning the tile shape (kept).** One default cannot serve these shapes:
within a single measurement window the spread between best and worst config is
**1.9–4.0x**. `triton.autotune` keyed on `(M, N, K)` is worth **1.08x** on the
GEMM and takes the model to **1.07x** wall clock against stock, interleaved,
spread 0.5–1.3% — matching NCU device time (9.759 / 9.105 = 1.07x) exactly.

**Autotuning is legitimate here for a reason that would not hold for a float
GEMM**, and the reason is the project's whole thesis: the accumulator is int32,
integer addition is associative and commutative, so **every config produces the
identical bits**. Without that, the tuner would be free to pick a different
*function* depending on what a benchmark happened to measure at startup — a
model whose output depends on its own warm-up. `check_all_gemm_configs_agree`
pins it: 10 configs x 8 shapes, 80 runs, zero disagreements.

**A contaminated sweep produced a false 1.82x and inverted the ranking.** The
first sweep said BLOCK_N=32 beat BLOCK_N=128 on `down_proj` by **3.97x** and
that 1.82x was available overall. Re-run uncontended, BLOCK_N=128/BLOCK_K=128 is
the *best* config for that shape — which is what the autotuner had already
picked. **The tuner was right and my measurement was wrong**, and the cause was
that I had left one of my own benchmark jobs running in the background and was
timing on top of it. Absolute timings on this machine moved **4–5x** between
windows; only comparisons made inside one window mean anything, and NCU device
time is what to trust because it serialises kernels.

**Split-K: 2x on the kernel, nothing on the model (default off).** `down_proj`
is K=4864, N=896, so at BLOCK_N=128 it launches **7 CTAs onto 66 SMs** — block-
starved, not bandwidth-starved, at 111 GB/s. Cutting K and summing with
`tl.atomic_add` on **int32** (exact and order-independent, so legal here where a
float atomic would not be) took it from 39.3 to 20.0 us/call, verified bitwise
equal at every SPLIT_K and tile tried. End to end: **1.07x without it, 1.08x
with it**, on runs whose own spread was 0.6–1.3%. The extra zero-fill and
epilogue launches cost what the split saves. `Y_EXACT_SPLITK=1` enables it.

**That is the third time in this session that a faster kernel did not make a
faster model** — after the Triton GEMM that broke a fusion (step 5) and the O(1)
KV cache that broke the tracer (step 6). The pattern is now explicit enough to
state as a rule: *in a compiled inference stack, a kernel's standalone speed
predicts almost nothing; measure the model, and measure both arms in the same
window.*

**Where decode time now sits** (NCU, 2 steps, batch 32, 9.759 ms total): GEMM
3897 us (39.9%), attention matmuls 1356 us (13.9%), and everything else is a
~5 us launch-bound kernel. The GEMM is at ~13% SM and the remaining lever is
launch overhead rather than arithmetic — which is what CUDA graphs, or a
serving path that is not eager torch, exist to remove.

**Still open:** one 0.5B model, greedy decode, three prompts. A larger model,
sampled decode, and a real serving integration remain.

> **Two claims in this section were overturned by step 9 and are kept only as
> history.** Replacing `triton.autotune` with a tuner whose candidate list
> includes split-K, measured interleaved, is worth **1.07x on decode device
> time** — so "1.08x from autotuning" was not the end of it, and "the tuner was
> right and my measurement was wrong" was only half true: my measurement *was*
> wrong, and the tuner was also leaving 7% on the table. Clearing your own error
> and re-testing the incumbent are two different jobs.
>
> The `9.759 / 9.105 = 1.07x` NCU device-time figure is also not comparable to a
> CUPTI one: NCU replays each kernel with cache invalidation, which inflates
> cache-sensitive kernels, and the exact arm has 1.45x more of them.

#### M5 step 9 — the GEMM again, and a benchmark that measured cache, 2026-08-18

**Decode device time 1.045x → 0.97x: the exact path is now FASTER than stock on
the GPU, and slower only in launch overhead.** Wall clock **1.06x** on the full
`generate` and **1.01x** on decode alone. The headline is honest and the
explanation I first wrote for it was not — see the correction below.

The step began by testing the previous step's closing sentence — "what remains
is launch overhead, not arithmetic" — instead of believing it, because it made
a checkable prediction. `tools/exact_launch_audit.py` splits a decode step into
device time and the gap where the GPU is running nothing:

| arm | kernels/step | device ms | wall ms | gap |
|---|---|---|---|---|
| stock | 508 | 3.363 | 3.617 | 7% |
| exact | 739 | 3.278 | 3.670 | 11% |

**A benchmark that reported 108% of DRAM peak is what cracked it open.** Sweeping
the seven decode GEMM shapes, `gate/up` came out above the roofline — which is
not a good result, it is a wrong question. The cause: timing one GEMM in a loop
leaves its 4.36 MB of weights in the 48 MB L2, so the number was cache
bandwidth. A decode step reads ~493 MB of *distinct* weights and gets no such
reuse. `triton.autotune` benchmarks the same way, on one buffer in a tight loop.

That produced a hypothesis — *the tuner is choosing tiles for a cache regime the
model never enters* — and a microbenchmark that appeared to confirm it at
**1.20x on the GEMM**, with every corrected choice being a smaller `BLOCK_N`
(more CTAs; `q_proj` and `o_proj` had been launching **7 CTAs onto 66 SMs**).

**The hypothesis did not survive the model, and the microbenchmark did not
survive repetition.** Two of the six shapes are not reproducible in that sweep —
`gate/up` measured 6.0, then 8.9, then 17.6 us across three runs of my own
tool — so its per-shape table cannot carry a conclusion. Measured where the
numbers do reproduce (`exact_launch_audit.py`; stock sat at 3.355–3.385 ms
across six runs), decode device time:

| GEMM tiles | exact device ms | stock ms | ratio |
|---|---|---|---|
| `triton.autotune`, as shipped before | 3.513 / 3.515 | 3.385 / 3.355 | **1.04–1.05x** |
| new tuner, **cold** (ships) | 3.278 | 3.363 | **0.97x** |
| new tuner, **hot** (`Y_EXACT_GEMM_TUNE=hot`) | 3.231 | 3.355 | 0.96x |

**The rework is worth 1.07x on decode device time — and the cache regime is not
why.** Cold and hot land 1.5% apart, inside the exact arm's own ~3% run-to-run
spread, so on this model they are a wash. What actually separates the new tuner
from `triton.autotune` is the *candidate list* (split-K is in it, worth 2.6% on
`down_proj` alone) and interleaved measurement — not the temperature of L2.

The cold apparatus ships anyway, because a decode step genuinely is cold and
because the reporting bugs it fixed were real. But **"108% of peak" justified
investigating the benchmark, not the 1.20x that the investigation then
produced.** A striking diagnostic is a reason to look, not a result; the model
is the arbiter, and here it cut a 20% claim to 7% and reassigned its cause.

**Split-K stopped being a special case.** It had been a separate kernel behind a
hand-written guard (`blocks >= sms or k < 2048`) which at M=32 fired on
`down_proj` and nothing else — 24 of 169 GEMMs — which is exactly why it
measured 2x on that kernel and moved the model 1.07x → 1.08x. The guard also
keyed on K when the starvation is caused by N. It is one entry in the candidate
list now, and the measurement picks.

**Four ways this tuner measured the wrong thing, three of them mine.** Worth
listing because each was invisible in the output and each changed the answer:

1. **CUDA events around back-to-back calls measured CPU dispatch**, not the
   kernel — a flat 13.5–13.8 us across shapes whose weight bytes differ by 40x.
   *A flat column across a swept axis means the axis is not what is being
   measured.* CUPTI device time instead.
2. **The "cold" ring was capped at 48 buffers**, which for a 0.80 MB weight is
   38 MB — inside L2. The fix for a hot benchmark, still hot.
3. **The round loop restarted the buffer index**, so 5 rounds x 20 calls touched
   20 buffers five times rather than 100 distinct ones; taking the minimum then
   selected the most cache-contaminated round. Fixing 2 and 3 moved `q_proj`
   from 3.43 to 4.58 us and brought it into agreement with an independent
   interleaved sweep — *two harnesses agreeing is the check, not either one
   looking plausible.*
4. **Candidates were run to completion in turn rather than interleaved**, so
   drift landed on whichever was running; the same shape tuned three times gave
   three different winners. Round-robin, which this project already required of
   its model-level A/Bs and had not applied here.

**The one that reversed twice is the interesting one.** `_bench_all` timed the
split-K kernel without the epilogue `acc * sx * sw` that `w8a8_matmul` must run
afterwards, while the plain kernel's epilogue is inside the kernel — so split4
"beat" plain by 1.19x. Adding the epilogue made it *lose* by 8%. **Both were
wrong**: in the model, split-K adds **24** launches per step, one per layer, not
48 — so inductor fuses that epilogue into neighbouring elementwise work, and an
isolated benchmark has no neighbour to fuse into. The model says split4 wins by
2.6% of decode device time (3.332 → 3.245 ms, stock steady at 3.36). *An
isolated benchmark cannot price an epilogue whose cost depends on the graph
around it.*

**Tuning stability needed a cache, not a better tuner.** Several candidates sit
inside the 5% tie band — 5% being what min-over-rounds actually disperses by
here, measured at 0.2–9.6% — so *which* are inside it moves with the noise.
Raising the round count from 5 to 9 simply moved the flip from `gate/up` to
`q_proj`. Measuring once per GPU and persisting to `.ysu_exact_gemm` is the fix,
and it is the convention this repo already uses (CLAUDE.md gotcha #3) — with the
same trapdoor: **the cache cannot tell that the kernel changed**, so re-tune with
`Y_EXACT_GEMM_TUNE=force` after editing `_w8a8_gemm`.

None of this can change what the model computes, and that is the point: every
candidate is bit-identical because the accumulator is int32.
`check_all_gemm_configs_agree` now covers the atomic split-K path too, where the
claim has teeth — `tl.atomic_add` genuinely does complete in a nondeterministic
order, and int32 is what makes that not matter. 11 candidates, 76 runs, zero
disagreements; the same test over a float accumulator would be expected to fail.

**Where it leaves the bottleneck.** Decode is now 0.97x on device time and 1.01x
on wall, so the remaining deficit is entirely the 1.45x kernel launches. The
full-`generate` figure is worse (1.06x) than the decode-only one because prefill
is a large share at 64 new tokens — that, and CUDA graphs, are the next levers.
CUDA graphs would need a static cache, and padding attention to a fixed length
has a cost of its own that has not been measured.

#### M5 step 10 — what exactness costs in quality, 2026-08-18

**Task accuracy: net +2 items out of 3000. Perplexity: +6.1%.** §6 item 1 has
been open since the project started, and every throughput figure in this
document was worthless until it was measured — "determinism is free" means
nothing if the model got worse.

`tools/exact_task_accuracy.py`, Qwen2.5-0.5B-Instruct, 1000 items per
multiple-choice task, 250k tokens of wikitext-2:

| arm | wikitext ppl | HellaSwag (norm) | ARC-Easy (norm) | ARC-Chal (norm) |
|---|---|---|---|---|
| fp32 (reference) | 13.863 | 48.5% | 59.4% | 33.7% |
| stock (bf16+SDPA) | 13.871 | 48.5% | 58.6% | 33.7% |
| **exact (int8)** | **14.714** | **47.6%** | **60.5%** | **32.9%** |
| crude4 (CONTROL) | 40.035 | 41.5% | 54.2% | 29.9% |

**The multiple-choice result is a genuine null, and the paired view is what
establishes that.** Comparing two percentages cannot distinguish "identical" from
"different in offsetting ways". Both arms answer the same items, so the right
statistic is the discordant pairs: exact-only 142, stock-only 140, **net +2
against 282 disagreements**. That is 0.12 standard deviations — there is no
effect. It also shows the model *did* change: **9.4% churn**, roughly one item in
eleven flips. Quantisation is reshuffling borderline items, not removing
knowledge.

**Perplexity is a real regression and must be quoted.** bf16 costs +0.06% against
fp32, so essentially all of the 6.1% is int8. Perplexity is the more sensitive
metric here and it is the honest caveat: this is not "free", it is "free on the
benchmarks people score models with, and 6% on the metric that sees everything".

**The control is the reason any of this is believable.** `crude4` is the same
weights rounded onto a 4-bit per-channel grid, and it scores 40.0 ppl against
13.9 with accuracy down 5–7 points everywhere. A harness that reports "no
regression" without a damaged arm beside it has not shown that it *could* have
reported one — the same discipline as `check_pv_bound_is_load_bearing`.

**What this is not.** Self-computed, not `lm-evaluation-harness`: every arm gets
byte-identical prompts, tokenisation, batching and scoring, so differences
between arms are meaningful and absolute values are not comparable to published
leaderboards. Multiple-choice sets are subsampled at a fixed seed. One 0.5B
model.

**The 6.1% is quantisation, not determinism, and the distinction is the whole
pitch.** The exactness machinery adds no error on top of the quantisation it is
applied to — M5 step 7 measured the fixed-point path and a plain int8 path with
fp64 softmax at **identical** error (ratio 1.00x). So a non-deterministic int8
model would pay the same 6.1%; determinism is what is free, quantisation is what
costs. An `q8lin` attribution arm (int8 linears, untouched attention) is wired
up in the tool to split that 6.1% between the weights and the K/V cache, and has
not been run yet.

#### M5 step 11 — the perplexity cost is K, and K was the free half, 2026-08-18

**+6.1% perplexity → +2.3%, at zero throughput cost.** Step 10 left one number
as the deliverable's weakness; this attributes it and removes most of it.

**Attribution first, and it inverted the obvious guess.** An arm with int8
linears and untouched fp32 attention (`q8lin`) splits the cost:

| | wikitext ppl | |
|---|---|---|
| fp32 | 13.863 | |
| + int8 **linears** | 14.166 | **+2.2%** |
| + int8 **K/V** on top | 14.714 | **+3.9%** |

So the KV cache costs nearly twice what the weights do. Group-wise weight
scales — the standard first move, and the one I had proposed — would have
chased the smaller half.

> **Correction, step 16.** Read "+2.2% linears" as *the linear layers*, which is
> what the `q8lin` arm actually varies — not the weight tensors. Splitting it
> further shows the weights contribute **−0.12%**, i.e. nothing, and all +2.2%
> is the *activations*. Group-wise weight scales would therefore have chased not
> merely the smaller half but essentially zero. The imprecise label is what let
> the wrong lever stay on the roadmap for two steps.

**Then the widths, and the second surprise: it is all K, and none of it is V.**
`K_LEVELS`/`V_LEVELS` are parameters now, with every exactness bound *derived*
from them rather than restating 127:

| K | V | ppl | |
|---|---|---|---|
| 127 | 127 | 14.714 | shipping |
| **1023** | **127** | **14.183** | **all of the K/V penalty, free** |
| 127 | 1023 | 14.723 | V alone: nothing, slightly worse |
| 1023 | 1023 | 14.182 | V adds 0.001 on top of K |
| 511 | 511 | 14.228 | |

14.183 against `q8lin`'s 14.166 means an 11-bit K costs **0.12%** over
full-precision attention — the K/V term is essentially gone.

**The asymmetry is structural, which is why this was cheap.** Q·K^T is one
matmul at any width, so widening K changes no work at all. Widening V is
bounded by `exact_pv`'s digit split (`T * (2^dbits - 1) * V_LEVELS < 2^24`), so
it forces narrower digits and MORE matmuls — 8 instead of 5 at 10 bits — and the
sweep says it buys nothing for them. **The expensive side to widen was the side
that did not need widening.**

**Per-channel K is permanently unavailable, and that is worth stating as a
limitation of the approach rather than a gap in the implementation.** A
per-token scale factors out of the dot product and leaves an integer sum; a
per-channel scale sits *inside* it (`sum_c q_c k_c s_c`) and makes the reduction
float-weighted, i.e. order-dependent again. The literature's better K scheme
(KIVI-style per-channel K) is exactly the one exactness forbids. Widening K is
the move that remains, and it happens to be enough.

**The width must be derived per head_dim, not fixed.** The score budget is
`d * Q_LEVELS * K_LEVELS < 2^24`, so the ceiling falls as head_dim rises:

| head_dim | max K_levels | K=511 | K=1023 |
|---|---|---|---|
| 64 (this model) | 2064 | 25% of budget | 50% |
| 128 (most larger models) | 1032 | 50% | 99.1% |
| 256 | 516 | 99.0% | **refuses** |

A hardcoded 1023 would fail closed on head_dim 256 — a default that will not run
some real models. **Using 99% of the budget is exactly as safe as using 6%**,
because `|score| <= d * Q * K` is a proof and bounds every partial sum too, not
a statistical estimate. So the width is chosen from head_dim.

**Cost, measured rather than argued: none.** Decode device time 3.278 -> 3.296
ms and kernels/step unchanged at 739 — inside the exact arm's own ~3% run-to-run
spread, which is what the structural argument predicted (Q*K^T is one matmul at
any width). Wall clock against stock went 1.06x -> **1.04x** on full `generate`,
1.02x on decode alone, 0.98x on decode device time. Quantised K/V are held as
float32 integer-valued tensors here either way, so no bytes move. In a serving
path with real int8 storage an 11-bit K needs int16, i.e. **1.5x KV cache
memory** (K doubles, V does not) — a genuine trade against long context, which is
why the knob stays.

**Task accuracy at the new width, same protocol as step 10:** perplexity
14.183 vs stock's 13.871 (**+2.2%**, was +6.1%); paired multiple choice **net
+40 of 3000 items** with 228 disagreements, and churn down 9.4% -> 7.6%. The +40
is nominally in exact's favour but comes almost entirely from ARC-Easy on one
1000-item subsample — **read it as unchanged, not as an improvement.** Gates:
self-test 11/11, demo stock 3/3 diverge vs exact 0/3, `cargo test` 361 passed.

**Six instances of one bug class, found while parameterising.** A width written
as a literal in one place and *assumed* in another:
`quantize_rows(x, levels=None)` defaulting to **K's** width, so a caller
quantising V without saying so got K's; the self-test's float64 reference
hardcoding 127, so it reported a *width change* as a correctness failure;
`exact_accuracy.py`'s `q_rows` doing the same, so at any non-default width that
tool would have measured a scheme that does not ship; Q's width as a bare 127 at
three sites; and a **dead per-channel quantiser** left beside the reference —
worse than unused, since per-channel is the one family this approach cannot use.
`levels` is required everywhere now and every bound derives from a named width.
Exactness re-verified at both widths: self-test 11/11 at 127 and at 1023,
batch-invariance included.

#### M5 step 12 — the vLLM baseline, and it vindicates the control, 2026-08-18

**Against production vLLM the exact path is 1.54x slower, and 100% of that gap
is CUDA graphs.** Every throughput figure in this document compares against
`stock` = bf16 + SDPA under `torch.compile`, which is the same framework and the
same harness — a fair control, but not what anyone deploys. This measures the
third point. Same prompt, batch 32, 64 new tokens, `ignore_eos` so every arm
emits exactly 2048 tokens (asserted, not assumed):

| arm | tok/s/seq | vs stock |
|---|---|---|
| vllm bf16, **CUDA graphs on** (production default) | **356.4** | 1.47x faster |
| vllm bf16, **CUDA graphs off** (`--enforce-eager`) | 241.2 | **1.00x** |
| stock, compiled HF bf16 + SDPA | 241.9 | — |
| exact, int8, deterministic | 232.0 | 0.96x |

**vLLM with its graphs disabled is statistically identical to compiled
HuggingFace (241.2 vs 241.9, spreads 2.0% and 0.4%).** Two consequences, and the
first matters more than the number:

1. **The baseline was not weak.** The obvious criticism of every figure here has
   been "you benchmarked against HuggingFace". It turns out compiled HF *is*
   vLLM-minus-CUDA-graphs on this workload, so the 1.04x parity claim was
   measured against a legitimately-tuned stack. Continuous batching, paged KV and
   a fused sampler buy **nothing** at batch 32 on a 0.5B model — they are
   throughput features for regimes this benchmark does not enter.
2. **The whole remaining deficit is launch overhead**, which is exactly what
   `exact_launch_audit.py` said before vLLM was installed: exact is 0.98x of
   stock on decode *device* time and 1.02x on wall, with 1.45x the kernel
   launches. Independent measurement, same conclusion.

**So CUDA graphs stop being speculation and become the ranked next lever, with a
measured prize of up to 1.47x** — and the exact path should gain *more* than
stock, because it launches 1.45x the kernels and therefore has more overhead to
remove. The cost that has to be weighed against it is unchanged and still
unmeasured: graphs need static shapes, so a static KV cache, and padding
attention to a fixed length costs work proportional to the padding. At the
benchmark's ~104-token horizon that is a ~1.4x padding factor on attention.

**Do not restate this as "exact is 1.54x off production".** The comparison is
exact-int8-deterministic against vllm-bf16-nondeterministic, and it mixes two
differences: the arithmetic (parity, per row 3 vs row 4) and the launch
machinery (all of it). The honest sentence is: *at equal launch machinery the
deterministic path is at parity with production; production's advantage over
both is CUDA graphs, which nothing about exactness prevents.*

Method notes, because they were both traps. `ignore_eos=True` with fixed
`max_tokens` is required — without it a short generation makes vLLM look faster
by doing less work, and the script asserts the produced token count. vLLM lives
in its own venv: it pins its own torch, and installing it beside the exact path
risks downgrading `torch` under a working environment.

#### M5 step 13 — invariance under production-shaped batching, 2026-08-18

**stock 16/16, exact 0/16.** The demo decodes one prompt replicated to fill the
batch, so every sequence has the same length — and production never looks like
that. Continuous batching mixes lengths, pads to the longest, and reshuffles
composition as sequences join and finish. This was the cheapest remaining
experiment that could have invalidated the central claim, and it does not.

`tools/exact_ragged_batch.py` decodes the *same* prompt 160 tokens greedily in 17
batch compositions, varying four axes at once: batch size (1 → 32), neighbour
lengths (2 to 460 tokens), the target's position (first, middle, last), and which
neighbours (same lengths, different text). Every stock composition changes the
target's output; no exact composition does.

**Three implementation-level dependencies on batch composition were genuinely
exercised, and each was a real chance to break:**

1. **`exact_pv` picks its digit width from the key length.** `dbits` falls below
   8 once `T >= 2^24/(255*V_LEVELS)` = 518, and the test reaches **T = 173 to
   744** — so padding the batch to a long neighbour makes the target's `p @ V` be
   computed by a *different decomposition* than when it decodes alone. Exact at
   every width is the claim; this is the evidence.
2. **`w8a8_matmul` picks `BLOCK_M` from `M = batch * seq`**, so a different batch
   size tiles every GEMM differently. int32 accumulation is what makes that
   bit-identical — the same property that licenses the tuner in step 9.
3. **Padded key positions must contribute exactly zero.** `p` is forced to 0
   where the mask blocks, so the denominator `l` and the fold's scale `wmax` are
   untouched by padding. Had any of those leaked, a longer neighbour would have
   shifted the target's softmax.

**Two harness bugs, and the first is the reason this file has a control.** At 24
new tokens **stock passed all 11 compositions** — a bf16 reduction-order delta
needs room to flip a greedy argmax and then diverge visibly, and the demo uses
160 for exactly that reason. A too-short generation makes a broken comparison
look like a clean pass, and without the "stock must fail" assertion this would
have been reported as a result. Second, the digit-width claim in the file's own
docstring was **false when written**: the LONG prompts top out at 169 tokens, so
T reached only ~329 and never crossed 518. The tool now prints the range of T it
reached and says so when the boundary is not crossed, so the claim cannot quietly
become false again.

Also fixed on the way: `_nospace.guard()` derived the repo root from `argv[0]`
assuming `<root>/tools/<script>.py`, so a `python - <<EOF` invocation (argv[0] is
`-`) repointed the **shared** `/tmp` symlink one directory too high. Every later
tool then re-exec'd through a path with no `tools/` in it, and it self-healed on
the next real script run — which is what made it confusing rather than obvious.

**What this does and does not establish.** It covers ragged *prefill* batching
under HF `generate`, which pads to the longest prompt. It does not cover a
serving engine's *continuous* batching, where sequences join and leave a
partially-decoded batch and the KV cache is paged — the mechanism is the same
(row-local quantisation, integer reductions) but the code path is not this one.
That belongs with the vLLM integration, not with this prototype.

#### M5 step 14 — CUDA graphs: the economics are good, the route is blocked, 2026-08-18

Step 12 showed vLLM's entire advantage is CUDA graphs, which made them the ranked
next lever with a measured prize of up to 1.47x. This prices the cost side and
tries the route. **The trade is clearly worth taking and the HF path segfaults.**

**The cost is 5-8%, not the ~1.4x on attention I had assumed.** A static cache is
the prerequisite for capture, and it pads attention to `max_cache_len` on every
step. Measured on both arms, interleaved:

| arm | dynamic cache | static cache | cost |
|---|---|---|---|
| stock | 241.9 tok/s/seq | 222.8 | −7.9% |
| exact | 232.0 | 221.3 | −4.6% |
| **exact / stock** | 1.04x | **1.01x** | |

The exact arm pays *less* for static shapes than stock does, so the ratio
improves to **1.01x** — the closest parity measured. Pay ~6% to unlock up to 48%:
the economics are not close.

**`torch.compile(mode="reduce-overhead")` + `cache_implementation="static"`
segfaults, on BOTH arms.** Exit 139, with inductor's cudagraph-trees warning
naming the cause: HF's `StaticCache.update` does `self.cumulative_length.add_()`,
an in-place mutation of cache state inside the captured region. Adding the
documented `torch.compiler.cudagraph_mark_step_begin()` per invocation fixes the
first error ("accessing tensor output of CUDAGraphs that has been overwritten")
and the process then dies in the same place. **Both arms fail identically, so
this is transformers 5.15 + torch 2.13, not the deterministic path.**

**Two properties were checked before concluding a static cache is acceptable,
and the first result was alarming until it was localised.** Under `generate`, the
exact path's output DIFFERS between dynamic and static cache. That looked like a
determinism violation, and it is not:

* Driving the cache directly with explicit `cache_position`, dynamic and static
  produce **bit-identical logits** through prefill and four decode steps. So
  padded positions really do contribute exactly zero — `p` is forced to 0 where
  the mask blocks, leaving `l` and the fold's `wmax` untouched.
* Under `generate` the first difference is at step 2, **max|logit diff| 5.7e-06**,
  with the same top-1 token; it accumulates until a token flips. That is a
  rounding-scale difference in the parts of the model this project does **not**
  make exact — float32 RoPE `cos`/`sin` and the residual adds — fused differently
  under the changed shapes.
* **Batch invariance still holds under a static cache: exact 0/3, stock 3/3**
  (`batch_invariance_demo.py --static-cache`).

So the distinction that matters: **batch invariance is the claim and it survives;
cache-implementation invariance is a different claim and was never made.**
Switching to a static cache moves the output once, like any version change, and
is stable thereafter. Worth stating plainly because "the output changed when I
enabled a static cache" reads as a broken guarantee unless the guarantee's scope
is written down.

**Where that leaves CUDA graphs.** The prize is real (up to 1.47x), the cost is
small (~6%), the property survives, and the blocker is a library bug in one
version pair. It is therefore not a research question but a plumbing one, and its
proper home is a serving integration rather than this prototype — the same
conclusion the O(1) KV cache reached in step 6, for the same underlying reason:
**this prototype's remaining costs are all framework costs, and the framework is
the thing a serving path replaces.**

#### Step 15 — the exactness bounds, checked by Z3 over their whole domain

`tools/exact_bounds_check.py`. The determinism claim reduces to one sentence —
*every partial sum stays below the point where its float type starts rounding* —
and that sentence was discharged by four hand-derived bounds in two files:

| | bound | where |
|---|---|---|
| A | `d · Q_LEVELS · k_lv < 2²⁴` | score sum, fp32 |
| B | `T · (2^dbits − 1) · V_LEVELS < 2²⁴` | each digit matmul, fp32 |
| C | `T · 2^P_BITS · V_LEVELS < 2⁵³` | `W·V` accumulator, float64 |
| D | `T · 2^P_BITS < 2⁵³` | softmax denominator |

Each is trivial in isolation, which is exactly why checking them one at a time is
not enough. **The failure mode is an interaction**: two adaptive mechanisms
(`k_levels_for` narrows K as head_dim grows, `exact_pv` narrows its digits as T
grows and falls back to float64 when it runs out), four bounds and three
env-settable widths. A configuration where one escape hatch fires and another does
not is a path the code *believes* is exact and is not. That is decidable integer
arithmetic, so it is a solver question.

**The headline result was not the one predicted, and the surprise is the useful
part.** I expected to find a window where the digit path succeeds while the float64
recombination overflows, and to report it as a documented ceiling. Z3 returns
**UNSAT**: with `dbits ≥ 1`, bound B already forces `T·V < 2²⁴`, while C only fails
at `T·V ≥ 2²⁵`. **B is strictly 2× tighter and subsumes C** — taking the digit path
is *itself* a proof that the recombination is exact. That is a structural
guarantee, not luck, and it is stronger than the code's own comments claimed.

What that leaves is one route into the unsafe region, and it is real:

* `dbits` hits 0 at `T ≥ 2²⁴/V`, C fails at `T ≥ 2²⁵/V` — so **the float64
  fallback is live on exactly one octave of context length**, and past its far
  edge nothing is exact.
* **The exact path therefore has a hard context ceiling**: `T ≤ 264,208` at
  V=127, **32,800 at V=1023**. Previously unwritten anywhere. It also prices the
  V-widening experiment of step 10 properly — widening V to 1023 would have cut
  the maximum context by 8×, on top of costing three extra digit matmuls for no
  perplexity gain.
* Both edges are **refused, not computed**: verified by driving `assert_bound`
  at `T = ceiling ± 1`, and `head_dim = 1041` (where `k_levels_for`'s 127 floor
  first exceeds the score budget) at the other bound.

**Z3 proves the derivations; exhaustion closes the model-vs-code gap** — the same
split `proofs/ZkControlFlow.v` states about itself. `k_levels_for` is exhausted
over *every* head_dim in 1..4096 (a pure function of one small integer, so this is
complete rather than sampled), and `exp2_neg_q16_16`'s `p ≤ 2²⁸` claim over its
entire reachable domain of ~2M inputs. **That last one is the load-bearing check**:
bounds C and D both rest on it, nothing asserts it, and it is a property of a
twelve-step integer polynomial — precisely the kind of claim a docstring makes and
nobody verifies. It holds, with equality at `t=0`.

**A second result came from refusing to accept a tautology.** The first version of
check A asked "does `k_lv · d · q ≤ 2²⁴ − 1` imply `d · q · k_lv < 2²⁴`" — which is
true by reading and tests nothing. Stated as the *actual* spec, `k_lv ≤
⌊2²⁴/(d·q)⌋`, it comes back **SAT**: the cap is a floor and the bound is strict,
so whenever `d·q` divides 2²⁴ the product lands on 2²⁴ exactly. Adding the
power-of-two rounding is still SAT, at `k_lv = 1, d·q = 2²⁴`. Only with the
`max(127, …)` floor as well is it UNSAT. So **both mechanisms in `k_levels_for`
are load-bearing and neither suffices alone**, and `k_lv = 1` is the unique
violating width (`k_lv ≥ 3` forces `d·q ≤ 5,592,405` and hence a product below
2²⁴). That is not a bug — it is the reason a line that looks like defensive
padding cannot be deleted, which is exactly the kind of thing that gets deleted.

Two process notes, both of which changed the tool rather than the code:

* The first version of the digit check **re-implemented the selector it was
  checking**, which only ever proves I can copy a line twice. `digit_width` is
  extracted from `exact_pv` so the checker exhausts the real one — the same reason
  the ZK fuzzer's oracle is written against an independent IR rather than Y's AST.
* **Mutation-verified, 7/7 caught**, and the first round was 5/6. The miss was a
  `>=` → `>` off-by-one in the digit loop, and checking rather than patching showed
  it is **not a bug at all**: 127 ∤ 2²⁴ so the two forms can never disagree, and
  2²⁴ is itself exactly representable in fp32. A confirmation, not a hole — the
  distinction `feedback-mutation-holes-hide-behind-later-checks` exists to make.
  It was replaced by a mutation that *is* a bug (bounding against 2²⁵), plus a
  shift-loop mutation that drops `p`'s top bit — which nothing caught until a
  **differential of `exact_pv` against exact integer arithmetic** was added beside
  the bound checks. Bound arithmetic cannot see a lowering that satisfies every
  bound and still computes the wrong product.

#### Step 16 — the last accuracy gap is activations, not weights

`tools/exact_quant_attribution.py`. Step 10 left +2.2% of perplexity on the
table and attributed it to "the linears", with **group-wise weight scales** named
as the obvious next lever. Measuring the ceiling before building it says that
lever is the wrong one, by a wide margin. Fake quantisation in fp32 — no kernel,
no int8, accuracy only — over wikitext-2 at 60k tokens:

| arm | ppl | vs fp32 | recovers |
|---|---|---|---|
| fp32 | 14.212 | — | |
| **w8** (weights only) | 14.213 | **−0.12%** | |
| **a8** (activations only) | 14.510 | **+1.96%** | |
| w8a8 (shipped) | 14.549 | +2.24% | |
| w8g128a8 (group 128) | 14.530 | +2.10% | **6.2%** |
| w8g32a8 (group 32) | 14.543 | +2.19% | 1.9% |
| w8a255 | 14.327 | +0.68% | 69.6% |
| **w8a511** | **14.225** | **−0.04%** | **101.8%** |
| w8a2047 | 14.219 | −0.08% | 103.7% |

**Weight quantisation is free** — −0.12% is noise in fp32's favour. The entire
remaining gap is the *activations*, group-wise weight scales recover 6% of it,
and **one extra bit of activation width recovers all of it.** The `w8` and `a8`
arms are what make this an attribution rather than a ranking; without them the
grouped arms' failure would look like a tuning problem.

**Why this was mis-ranked**: step 10's split was between *the linears* and *the
KV cache*, so "+2.2% weights" meant "+2.2% for the linear layers", of which the
weight tensors turn out to contribute nothing. The label was imprecise in a way
that survived because nobody split it further. Re-read a phase's label before
building on the number attached to it.

**The fix preserves exactness, and the rejected one would not have.** Group
scales split the reduction into `G` integer partials that must then be combined
with *float* weights — order-dependent again, recoverable only by requantising
the scales onto a common grid. Widening the activation needs no such rescue:

    q = hi*128 + lo          (hi in [-4, 3], lo in [0, 127], both int8)
    q @ w = 128*(hi @ w) + (lo @ w)

Two `tl.dot`s, **one load of the weight tile**, combined in int32. Every step is
integer, so associativity is untouched and the batch-invariance argument is
verbatim the one the int8 path makes. It is the same digit-split device
`exact_pv` already uses for `p @ V`, applied to the other operand — and the
economic case is that at decode `M` the GEMM is DRAM-bound on weights, so a
second dot over a tile already in registers adds arithmetic to a kernel that is
not arithmetic-bound.

The width is **derived, not fixed**: `act_levels_for(K)` spends the int32
accumulator budget `K · a_levels · 127 < 2³¹`, which permits 2047 at this model's
widest `K=4864` and would fall to 511 at a 32k-wide MLP — the same shape of
budget as `k_levels_for`, one operand over. `Y_EXACT_ACT_LEVELS=511` enables it.

**What the second dot costs, measured rather than predicted: 5.9% of decode
throughput.** 234.7 → 221.7 tok/s/seq at batch 32, taking exact/stock from
**1.04x to 1.09x**. Each arm carries its own `stock` run as a control (244.5 vs
242.5, 0.8% apart), so the effect is real and about 2-3x the run-to-run spread
(1.9-2.5%) — quote it as ~6%, not to three figures. The prediction was "nearly
free, because decode is DRAM-bound on weights"; it is cheap but not free, and
saying so is the point of measuring. Compare the same reasoning's track record in
`feedback-gpu-ntt-vs-icicle`, where a 2x prediction measured 1.06x.

**On the real path, not just the simulation: perplexity 14.183 -> 13.887**,
against stock's 13.871 and fp32's 13.863. That is **+0.12% over stock**, down
from +2.2%, i.e. the accuracy cost of this scheme is now at the level where it
stops being a number anyone argues about. It also confirms the fake-quant
harness predicted the real path correctly, which is what makes that harness
reusable for the next such question.

**The trade, stated once, both halves measured:**

| | ppl vs stock | decode vs stock |
|---|---|---|
| `Y_EXACT_ACT_LEVELS=127` (default) | +2.2% | **1.04x** |
| `Y_EXACT_ACT_LEVELS=511` | **+0.12%** | 1.09x |

**Recommendation: 511, and the default is left at 127 anyway.** §6 item 1 names
accuracy as a kill condition and throughput as merely needing to be competitive,
and the target buyer (RL training teams) cares about reproducibility far more
than 6% of decode - so on the merits the wide activation is the better product
configuration. The default is not flipped here because every throughput figure
in this document was measured at 127, and silently re-baselining them is a
decision to take deliberately rather than as a side effect of an accuracy fix.

**Verified before being believed**, three ways: `q == hi*128 + lo` exhaustively
over every representable `q` (the negative half is the risky one — two's
complement shift-and-mask is "obviously" exact right up until it isn't); all 11
GEMM candidates including the K-split atomic kernel producing **bit-identical**
output; and that shared output matching the exact integer product from
`torch._int_mm`. **The last of those is what makes the second worth anything** —
eleven identically wrong kernels would agree perfectly.

**And the bounds check earned its keep immediately.** Adding the new
accumulator bound (E) to `tools/exact_bounds_check.py` found two things. First,
the obvious budget `K · a · 127 < 2³¹` is **too loose by 128 levels**: the kernel
accumulates `128·(hi@w)` and `(lo@w)` into one int32 *interleaved* across K
blocks, so what must fit is the sum of the two parts' bounds, not the bound on
the value they reconstruct. Second — and this is about code that has been
shipping for weeks — **the plain int8 path's int32 accumulator was never bounded
at all**, and silently *wraps* at `K ≥ 133,153`. Unreachable in any model that
exists (a 70B MLP is K=28,672), but unstated, and a wrap is a wrong answer rather
than an imprecise one. `ExactLinear` now refuses at construction with the maximum
K for its width. Z3 also reported that the floor cap alone cannot meet a strict
bound — the identical lesson bound A taught, arrived at independently, because
every one of these caps is a floor and every one of these bounds is strict.

**One self-inflicted hole, caught by reading the output rather than the status.**
Both new self-tests passed on their first run *while testing nothing*: with the
shipped default still 127, `act_levels_for` returned 127, `hi` was confined to
[−1, 0] and the split was degenerate. `act_levels_for` takes a `requested`
override now so a test can exercise the wide path while the default stays
narrow. A green check whose printed parameters say `a_levels 127` is not a green
check.

### M5 — The demo · 3–4 weeks

A real quantized model, decoding the same prompt at batch 1 / 32 / 64,
token-for-token identical, with a throughput number beside it that is within
noise of vLLM. Then the same across two different GPUs.

- **Done when** — it is a screen recording that a stranger believes.
- This is the artifact that validates the technical claim and the pitch
  simultaneously. Take it to RL training teams first.

---

## 6. What would kill this

Written down in advance so the answer is not negotiated after the fact.

1. **Accuracy loss at int16/int8 operands is unacceptable to the buyer.** The
   range trade is real. If deterministic output requires a quantization the
   customer will not accept, there is no product. **STILL OPEN — but narrowed,
   2026-08-17.** M4 step 1d shows the *softmax representation* is not the
   problem: at F=28 the exact path is 0.01–0.20x f32 flash's error, and one
   int8 step in `V` costs 90,000x more than the fixed-point weights do. So the
   accuracy question is int8 quantization itself — the customer's existing
   decision — not determinism. That same test found and killed a real design
   bug (F=16 lost the entire tail on an attention sink, 2109x worse than
   flash).
   **MEASURED 2026-08-18 (M5 steps 10-11): task accuracy unchanged, perplexity
   +2.2%.** Against `stock` (bf16+SDPA) on Qwen2.5-0.5B-Instruct, paired over
   3000 multiple-choice items the two arms differ by **net +40**, and wikitext-2
   perplexity is 14.183 vs 13.871. It was +6.1% until the cost was attributed:
   the KV cache accounted for 3.9% of it against the weights' 2.2%, and **all of
   the KV term was K**, which is the side that is free to widen. This risk is
   now *one small number a buyer can be shown* rather than an unknown — though
   still on one 0.5B model, and perplexity must be quoted, not buried.
   **CLOSED, conditionally, 2026-08-18 (M5 step 16): +0.12% at
   `Y_EXACT_ACT_LEVELS=511`.** The residual +2.2% was the *activations*, not the
   weight tensors (weights alone cost −0.12%, i.e. nothing), and carrying the
   activation as two int8 digits takes perplexity to **13.887 against stock's
   13.871** while keeping the accumulation integer and therefore
   order-independent. It costs **5.9% of decode** (1.04x → 1.09x vs stock), so
   this is now a stated trade rather than a risk: buy the accuracy back for 6% of
   throughput, or don't. The default is still 127 — flipping it re-baselines
   every throughput figure here and is a decision to take deliberately.
2. **Nobody will pay for determinism.** Plausible. Validate by taking M3 to RL
   teams before committing to M4–M5.
3. **The end-to-end penalty in memory-bound regimes.** ~~Unanswered.~~
   **ANSWERED 2026-08-17: about 3%, i.e. survivable** — see M4 step 1c. Against
   a structurally identical f32 twin, exact integer accumulation is
   **1.01–1.02x** (parity) and reaches **84–85% of DRAM peak**, so the phase is
   bandwidth-bound and the arithmetic is free. The cost is the separate score
   pass: 4/D = **3.1%** of traffic at head_dim 128. An earlier reading of this
   experiment claimed 0.81–0.89x (*faster*); that compared against a baseline
   charged for `ex2` the exact path did in an untimed pass, 128x redundantly.
   Still open: no fused score+accumulate flash kernel exists, so no end-to-end
   ratio against FlashInfer is quoted. **End-to-end on a real model, both arms
   compiled, is now 1.43x** (M5 step 4) — a torch prototype, not the kernels,
   but the four causes of the previous 6.7x were all plumbing and none was
   exactness. **Table stakes per §7 is ~10%; 6% on full `generate` and 1% on
   decode alone now clear it** - on a torch prototype against a torch baseline,
   both compiled, on one 0.5B model. **On decode DEVICE time the exact path is
   ahead, 0.97x** (M5 step 9): int8 tensor cores move half the weight bytes at
   twice the rate, and what is left is launch overhead — the exact path issues
   1.45x the kernels. That is a framework cost, not an arithmetic one, which is
   the same conclusion steps 6 and 8 reached from the other two levers (an O(1)
   quantised KV cache, split-K on the GEMM).
4. **A funded team ships batch-invariant kernels first.** Likely, and survivable
   — they are constraining f32 reduction order, which is a weaker claim than
   order-independence by construction, and it cannot extend to a certificate.

## 7. Non-goals

- **Beating vLLM/FlashInfer on throughput.** Table stakes is within ~10%. Racing
  here is the mistake this document exists to avoid.
- **Training.** Exactness is expensive where the hardware lacks an integer
  dot-product instruction, which is where training lives.
- **Selling the compiler.** Y is the factory. It is why a kernel takes weeks
  instead of months. It never appears on an invoice.
- **ZK.** Distinct programme, distinct market. `y-gpu` stays free and open as
  a portfolio artifact.
