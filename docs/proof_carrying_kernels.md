# Proof-Carrying Kernels

**A compiler that emits, alongside the binary, a machine-checkable certificate
that the optimized kernel computes its specification — not a test suite that
failed to find a counterexample.**

| | |
|---|---|
| Horizon | 4–6 years |
| First publishable result | ~6 months |
| Status | pre-Phase 0 |
| Drafted | 2026-08-12 |

> Every claim about this repository below was verified against the source
> while drafting, not recalled.

---

## 1. The problem: nobody can tell you an optimized kernel is correct

The entire industry validates numerical kernels the same way — run the
optimized version against a naive reference on random inputs, compare within a
tolerance, ship it. That is a search for counterexamples, and it is a weak one.

This repository is an unusually well-documented catalogue of it failing. Each
of these passed a differential test and was wrong anyway:

- A `u32` datapath compiled as `ld.global.f32`. It assembled, launched, and
  silently rounded every value above 2^24. No `ptxas` gate can see this.
- `INTT(NTT(a)) == a` passed with the twiddle multiply deleted entirely — the
  error cancels between the two directions. Checking a fast transform against a
  fast transform shares the bug.
- A relative-L2 norm over C cannot see a kernel writing *past* N into padding;
  every element it was asked to produce is still right.
- Twelve address computations in the CPU GEMM were correct only because
  `lda == K` made stride and extent the same number. Mutation testing found two
  of the new tests vacuous — one never entered the threaded path at all.

The last one happened this week, in a codebase whose author is more careful
about measurement than almost anyone. That is the point: **testing is not a
weak methodology being applied lazily. It is the strongest methodology
available, and it is not sufficient.**

---

## 2. The insight: exact accumulation makes equivalence provable

Kernel verification is considered impractical, and there is one specific
reason. You cannot prove that a tiled, K-split, multi-threaded reduction equals
the naive loop, because in floating point *it does not*. Addition is
non-associative, so every reordering gives a different answer. Every effort in
this space hits that wall and retreats to "within 1e-5", which is not a proof
of anything.

> **Integer and fixed-point addition *are* associative and commutative.** Under
> exact accumulation, any reordering of a reduction produces a bit-identical
> result.
>
> So "the optimized kernel equals the reference" stops being a tolerance and
> becomes an equality — and equalities are what proof systems are good at.

Y already has both halves of this, built for unrelated reasons and never
connected:

- `@ZeroDrift` selects an exact fixed-point representation per device and
  lowers it on **both** backends. `tests/zero_drift_end_to_end.rs` compiles a
  program, runs it, sums 4001 terms in opposite orders and asserts the results
  are bit-identical — with a control proving the same sequence genuinely drifts
  in `f32`.
- `recognize_gemm` takes a naive matmul nest and substitutes a packed AVX-512
  kernel.

That substitution is an **unverified rewrite**, and it is exactly the
obligation to discharge.

---

## 3. Inventory: what exists today, honestly

Six pieces are already in the repository. None was built for this purpose,
which is why the programme is credible and also why nothing is ready.

| Component | Where | State | What it still needs |
|---|---|---|---|
| Exact accumulation | `zero_drift.rs` | **real, tested** | Lowered on LLVM *and* PTX. Verified 2026-08-12. |
| Kernel rewrite | `cpu_gemm.rs` | **real, tested** | Explicitly *refuses* a `@ZeroDrift` accumulator today. |
| SMT discharge | `type_checker.rs` | narrow | Z3 wired in, but only for loop invariants over integer scalars. |
| Linear resource types | `linear_tracker.rs` | **real, tested** | Proves async tokens consumed once. Needed for the GPU pipeline. |
| Interval arithmetic | `type_checker.rs` | narrow | Basis for Phase 4 error bounds; currently only index bounds. |
| Empirical validation | `empirical_autotune.rs` | real | Already correctness-checks candidates against a CPU reference. |
| Certificate format | — | **does not exist** | Phase 5. The actual deliverable to a certification authority. |

The single most useful fact in this table: `recognize_gemm` contains an
explicit refusal of exact accumulators (`src/cpu_gemm.rs:289`), with a comment
explaining that substituting one would silently discard the guarantee. **The
two halves have never met, and the code declines to introduce them.** That
refusal is Phase 0.

Two corrections to `CLAUDE.md` found while drafting:

- It lists `@ZeroDrift` lowering as "4 sites in `llvm_emitter.rs`", which reads
  as CPU-only. `ptx_emitter.rs` has 24 references including
  `emit_drift_to_fixed` / `emit_drift_from_fixed`. **The GPU half is real
  today.**

  > **CORRECTION, 2026-08-25 — it was real for `+=` and WRONG for `acc = acc + e`.**
  > The PTX backend had a drift arm for `CompoundAssign` and none for `Assign`,
  > so the running-sum spelling — which is how the recognised GEMM nest is
  > written — fell through to the ordinary assignment path and accumulated in
  > **f32**, then wrote back through an unsigned truncating convert, under a
  > comment reading `accumulated exactly as I64`. Separately,
  > `emit_drift_from_fixed` narrowed `s64 → f64 → f32` on *every* read, so even
  > the working `+=` path lost an exact `I64` accumulator above 2^24. Both fixed;
  > the rule for what counts as an exact accumulation now lives in
  > `zero_drift::running_sum` and both backends call it, and
  > `tests/zero_drift_backend_agreement.rs` pins that the two spellings emit
  > byte-identical PTX. **Found by running every backend over the Phase 0 nest** —
  > the source did not exist when the empty-artifact sweep ran.

---

## 4. The phases

Ordered by dependency, not preference — each is unreachable until the previous
lands. Every phase produces something publishable or sellable on its own, so
the programme can be abandoned at any boundary without the prior work being
wasted.

### Phase 0 — Make the two halves meet · 3–6 months

Remove the refusal. Let `recognize_gemm` accept a `@ZeroDrift` accumulator and
emit an exact-accumulation packed GEMM on both backends. Nothing is proven yet
— this establishes that the object the proof will talk about can exist at all,
and what it costs.

- **Done when** — the tiled, threaded, K-split GEMM returns bit-identical
  results to the naive nest across every thread count and both backends. The
  property that is merely *tested* here becomes the theorem in Phase 1.

#### Phase 0 status, 2026-08-25 — DONE on the LLVM backend

`try_emit_gemm_kernel` substitutes `__y_gemm_exact_vnni_threaded` for a
recognised `@ZeroDrift` nest over `I16` operands and an `I64` accumulator, and
`tests/exact_gemm_thread_invariance.rs` pins the claim: byte-identical output
at 1, 2, 3, 5, 8 and 16 threads over a ragged K, each run also checked against
an integer reference, with a `--wrap=pthread_create` counter proving the split
actually forked. Measured 37.7 → 10.2 ms at 8 threads on 256×256×65536, same
checksum throughout.

**The operand domain is the source's types**, which is the decision that made
the substitution legal: `VnniExact::license` is stated over the int16 values
fed to `vpdpwssd`, and an `F32` nest cannot reach that domain without a
quantization scale — at which point the licence would have been granted against
the source's magnitude, not the kernel's. `I16`/`I64` *is* the kernel's
contract, so nothing is converted. An `F32` `@ZeroDrift` nest still falls back
to scalar exact lowering, permanently.

**The "both backends" clause is NOT met, and its premise needs correcting.**
`--emit-cpu` cannot express the recognised nest at all — it refuses
`block_ptr2d_load` by name, since it targets host code and that is a GPU
intrinsic. Its GEMM kernels come from a different mechanism entirely: a shape
dispatcher keyed on literal `M`/`N`/`K` that emits hand-written Rust/AVX, with
no `@ZeroDrift` path anywhere in the file (zero references). So satisfying this
clause is not "wire the same kernel into a second emitter" but "write a second
exact GEMM, in Rust source form, for a backend Y never compiles" — the
`--emit-cpu` output is printed for the user to paste. Re-scope it deliberately
or drop it; do not quietly treat the LLVM result as covering it.
- **Exit value** — deterministic GEMM is independently saleable. Reproducible
  numerics across thread counts and hardware is a real, unmet want in regulated
  ML and financial model validation.

#### Phase 0 gating measurement — ANSWERED, 2026-08-12

The cost of exact accumulation was the unknown that decided whether any of the
rest is worth doing. Measured with a standalone micro-kernel probe at equal
register budget (24 zmm accumulators), single core, best of 7 interleaved
rounds:

| kernel | tile | G MAC/s | % of f32 FMA peak |
|---|---|---|---|
| f32 FMA (Y's shipped shape) | mr=6, nr=64 | **166.9** | 95% |
| exact Q16.16 → int64 | mr=6, nr=32 | **65.2** | — |

**Exact accumulation costs 2.56x**, and theory agrees within 6%: 2×512-bit FMA
gives 32 f32 MAC/cycle, while `vpmuldq` issues ~1.5/cycle at 8 lanes for ~12.
Two separate costs are inside that number, and only one is the instruction mix:

1. **The tile is forced to halve.** An int64 accumulator holds 8 lanes per
   register where a float holds 16, so covering nr=64 would need 48 zmm against
   a 32 cap. `nr=64 → nr=32` costs arithmetic intensity independently of the
   arithmetic itself.
2. **Two instructions per product** (`vpmuldq` + `vpaddq`) against one FMA.

**First verdict: proceed, 2.56x is a tax the certificate market will bear.**
That verdict survived about an hour — see the next subsection, which replaces
it. It is left in because the int64 result is still the right answer for any
computation whose operands do *not* fit in int16, and because the reasoning
that produced it was sound and still wrong.

#### …and then AVX512-VNNI inverted it

The `vpdpwssd` follow-up was expected to improve the ratio. It reversed the
sign of the result. Same box, same protocol, verified **exact against a scalar
reference** (0 of 384 elements differ):

| kernel | tile | G MAC/s | vs f32 |
|---|---|---|---|
| f32 FMA | mr=6, nr=64 | 166.9 | 1.00x |
| exact int64 (`vpmuldq`) | mr=6, nr=32 | 65.2 | 0.39x |
| VNNI raw (no flush, overflows) | mr=6, nr=64 | 340.8 | — |
| **exact VNNI (`vpdpwssd` + int64 flush)** | mr=6, nr=64 | **314.5** | **1.88x** |

**Exact accumulation is 1.88x FASTER than float, not 2.56x slower.** Both of
the int64 design's costs vanish: `vpdpwssd` accumulates into int32, which holds
16 lanes like a float, so the tile stays at nr=64; and one instruction does
32 MACs against an FMA's 16. Flushing int32 into int64 often enough that it can
never overflow costs 7.7%. The raw kernel hits 97% of the `vpdpwssd` issue
ceiling and the exact one 89%, so both figures are understood rather than
merely observed.

> **CORRECTION, 2026-08-17 — the 314.5 / 1.88x row above is WRONG. The real
> figure is 1.10–1.15x.** Re-measured on a Ryzen 9 9950X while building the
> emitted kernel (`cpu_gemm::emit_vnni_micro_module`,
> `tests/cpu_gemm_vnni_micro.rs`): f32 **172.7** (doc: 166.9, reproduces) and
> VNNI raw **348.0** (doc: 340.8, reproduces), but exact VNNI **146.3** against
> the 314.5 claimed. Two of three reproduce, so the third is the error.
>
> **The flush costs 137.8%, not 7.7%**, and not because of how often it runs —
> sweeping the interval 64 → 65536 k-pairs moves throughput under 5%. It is the
> flush *block being inside the k-loop*: touching all 24 accumulators forces
> them out of registers for the whole loop (24 `vpdpwssd` + 15 spills + 14
> reloads). Hoisting it into an outer chunk loop gives **1.10–1.15x**, which is
> the number to quote. The residual gap to raw is register pressure (29 of 32
> zmm live), i.e. a tiling question.
>
> **This does not change the programme's direction**, since exactness was never
> being sold on speed — but "1.88x faster than float" must not be repeated, and
> §2's argument should rest on *order-independence*, which is measured and holds
> (`tests/cpu_gemm_vnni_micro.rs` shows byte-identical results across six k-range
> splits where the f32 control differs on 39 of 384 elements).
> See `docs/deterministic_inference.md` §1 for the full account.

**What is actually being traded is RANGE, not speed.** Operands are int16, and
the flush interval bounds their magnitude — at `FLUSH_T = 64` k-pairs,
`|a|,|b| <= 1024` keeps every partial sum inside int32. That is far less
precision than f32's 24-bit mantissa over its enormous dynamic range, so this
is **not** a drop-in replacement for an f32 GEMM; it is a different numeric
contract. Quoting 1.88x as though the two computed the same thing would be the
same error as measuring against a single baseline.

**This makes `@bounds(min, max)` load-bearing rather than advisory.** The
range claim is exactly the proof obligation that licenses the representation —
so Phase 1's first theorem is not only "the tiled kernel equals the naive nest"
but "and no accumulator overflowed", which is a bounds proof over the interval
arithmetic already in the type checker.

**Strategic consequence worth stating plainly:** `vpdpwssd` is the same
instruction the quantized-inference world uses for int8/int16. The exact
kernel this programme needs and the int8 capability the CPU business case
needed are **the same kernel**. That was not the plan; it is what the
measurement says.

One follow-up this still does not answer: **the end-to-end penalty in
memory-bound regimes.** These are compute-bound micro-kernel numbers, while
large GEMM sits near the DRAM roofline and decode and GEMV are outright
bandwidth-bound. int16 operands *halve* B's footprint against f32, so
memory-bound shapes plausibly gain more — but that must be measured, not
assumed, and it needs the emitter wiring to measure.

> **Two measurement bugs were found and fixed getting to this number, and the
> second one matters generally.** The first probe read 698 G MAC/s — 4x above
> hardware peak — because the pure call was hoisted out of the timing loop.
> With that fixed, the *exact* arm read 540 G MAC/s, ~6x what `vpmuldq` can
> issue: the loop cycled through 8 A panels, and **integer addition is
> associative, so the compiler may compute each panel's contribution once and
> multiply by the repeat count.** It cannot do that to the f32 arm, because
> float addition is not associative. **The property that makes exact
> accumulation provable is the same property that lets a compiler eat its own
> benchmark.** Fixed by putting the kernels in their own translation unit and
> linking without LTO. Both bugs were caught by comparing against the
> hardware's issue ceiling, which the probe now prints and shouts about.

#### Phase 0 progress, 2026-08-12

**Landed — the request now reaches the emitter.**

- `recognize_gemm` no longer refuses a `@ZeroDrift` nest. It records the
  accumulator's declared type and its resolved `@bounds` in
  `GemmShape::drift`, so a representation can be selected for it.
- `try_emit_gemm_kernel` **guards the seam this opened.** With the request
  visible, the obvious next step is to emit the packed kernel — and that would
  be a silently wrong answer, because the packed kernel accumulates in f32 and
  is neither exact nor order-independent. It returns `None` for a drifted nest,
  falling through to scalar lowering, which honours `@ZeroDrift` correctly.
  Slow and right rather than fast and wrong. The original refusal was correct;
  it was just in the wrong place.
- `tests/cpu_gemm_exact_accumulation.rs` pins it, with a control asserting a
  plain matmul *does* still reach the packed kernel — without which the first
  test would also pass if the recogniser stopped firing altogether.
  Mutation-verified: deleting the guard fails the drift test and leaves the
  control green, which is the split it exists to produce.
- 43 test binaries pass.

**Next, in order:** an exact micro-kernel behind that guard (`vpdpwssd` +
periodic int64 flush, per the measurement above), then the packing layout it
needs, then routing it through the existing threaded driver — at which point
the Phase 0 "done when" becomes testable: bit-identical results across every
thread count.

### Phase 1 — Mechanize one rewrite end to end · 6–12 months

Prove — not test — that the packed GEMM computes the naive nest under exact
accumulation. One transformation, fully discharged: the 2-D partition, the
K-split reduction, `pack_a`/`pack_b`, the micro-kernel, the masked tails.

- **Done when** — a machine-checked proof, regenerated by CI, that the emitted
  kernel refines the source nest for all M, N, K and thread counts.
- **Why now** — the motivating example is already written down and measured:
  twelve address sites correct only by coincidence, and two vacuous tests found
  by mutation. That is the paper's first section, already in hand.
- **Exit value** — the first publication, and the credential that makes the
  rest fundable.

#### Phase 1 progress, 2026-08-25 — the SCHEDULE is proved

`proofs/ExactGemmKsplit.v` (Rocq 9.1, no axioms, nothing admitted) proves the
K-split half of the Phase 0 kernel. Three results:

- **`bands_tile`** — the band decomposition the emitted wrapper computes
  (`base = K/nthr`, `rem = K%nthr`, first `rem` bands one longer) covers
  `[0, K)` exactly, for every `K` and every positive `nthr`.
- **`ksplit_exact`** — summing the per-band partials equals the naive sum over
  the whole of K. No hypothesis that K divides evenly, none that the bands are
  equal, none about the thread count beyond it being positive.
- **`any_thread_count_agrees`** — the corollary
  `tests/exact_gemm_thread_invariance.rs` asserts, *derived*. Note it says
  nothing about threads: two counts agree because each separately equals the
  naive nest.

**Section 2's central claim is now machine-checked rather than asserted.**
`rounding_breaks_the_split` and `exact_survives_the_same_split` are the same
`f`, the same `K = 201` and the same `nthr = 2`, differing only in the
accumulate. The rounded one gives **1100 against its own reference's 1000** —
it does not lose precision, it *disagrees*, and no tolerance makes that a proof
of anything. The exact one gives 1200 either way. A control asserts the two
accumulates really do differ on this input, so the exact case is not passing
because the rounding happens to be inert.

**What is NOT proved, stated because it is most of the kernel.** A band's
partial is modelled as the exact sum over its indices: packing, the 2-D
register tile, the masked tails and the int32 → int64 flush are all *assumed*
to compute that. This file proves the schedule around them. Phase 1's remaining
work is the micro-kernel, which is the harder half.

**The model-to-code gap is narrowed in one specific place rather than waved
at.** `cpu_gemm::ksplit_bands` / `ksplit_threads` are the Rust transcription of
the same definitions; `tests/exact_gemm_ksplit_model.rs` checks the
transcription against the theorem over 192,000 cases and asserts it *agrees*
with the emitted module's constants instead of restating them. And
`the_min_band_floor_is_what_the_model_says` sweeps K across the min-band floor
with the thread request held fixed, asserting the observed `pthread_create`
count is the model's answer — the one place the model predicts something the
shipped code reveals. Drift in either direction fails it, verified by mutating
each side separately.

**Mutation-verified 8/8**, and one result is worth keeping: making the
emitter's uneven split even (`icmp slt` → `sle`) is caught by the new floor
sweep at K=256 and **passes** the pre-existing K=4099 invariance test. A
correctness test at one shape does not cover a schedule.

#### Phase 1 progress, 2026-08-25 (2) — the OUTPUT tiling is proved, and it found a bug

`proofs/ExactGemmTiling.v` finishes the schedule. Where the K-split is a
*reduction* (the bands are summed, so the obligation is that they tile the
range), the output tiling is a *partition* (each tile writes a disjoint
rectangle of C, so the obligation is that every element is written **exactly
once**). Different theorem, and the difference matters: a coverage count is
satisfied by a tiling that writes one element twice and another never.

- **`tiles_cover`** — uniform `MR`/`NR` tiles with a clamped ragged tail
  account for `[0, extent)` exactly.
- **`tile_index_injective` / `tile_index_surjective`** — `(tile, offset)` is a
  bijection onto the axis. This is the "exactly once" obligation.
- **`c_written_exactly_once`** — the 2-D consequence over the whole `vg.j` /
  `vg.i` / `vg.fi` / `vg.fj` nest.
- **`unclamped_tail_writes_out_of_bounds`** turns the emitter's own comment
  ("Letting it write C directly would run past the last row and column … an
  out-of-bounds WRITE, not a wrong number") into a machine-checked refutation.

**Writing the theorem surfaced a precondition nobody had written down, and
testing that precondition found a live out-of-bounds heap write.** The 2-D
result needs `N <= ldc`; with a shorter row stride two distinct `(row, col)`
pairs collapse onto one address. Nothing in the compiler states that, because
every caller passes `ldc = N` — so nothing had ever called the exact kernel
with a padded C. Doing so found **three sites in `emit_vnni_threaded_module`
that treat C as contiguous `M*N`, ignoring `ldc`**:

| site | consequence with `ldc > N` |
|---|---|
| `memset(C, 0, M*N*8)` | zeroes into early rows' padding and leaves the **last rows' live cells unzeroed** — and the kernel accumulates, so that is a wrong answer |
| worker's job slot 8 stores the caller's `%ldc` | the worker's private C is a compact `M*N` buffer, so it writes `(M-1)*(ldc-N)` elements past the end — **heap overflow**, observed as `double free or corruption` |
| the reduction walks source and destination with one flat index | correct only when the two strides are equal |

Fixed: the zeroing is row-wise at the caller's stride, workers are handed
`ldc = N` for their compact buffer, and the reduction crosses the two strides
explicitly. Verified at 1, 2, 4 and 8 threads with `lda = K+5`, `ldb = N+9`,
`ldc = N+7` and both axes ragged.

**This is exactly the class §1 of this document cites** — *"twelve address
computations in the CPU GEMM were correct only because `lda == K` made stride
and extent the same number"* — found again, in the exact kernel, by the
programme that exists because of it. It is not reachable from the compiler
today; it is reachable from the public symbol that is Phase 0's deliverable.

**The honest causal story: the proof did not find the bug, and it is the more
interesting version.** Formalising the tiling forced the `N <= ldc` hypothesis
to be stated; stating it suggested the test; the test found the bug. Mutation-
verified 8/8 across the three fixed sites, the Rust transcription, the emitter's
clamp and two hypotheses of the proof.


#### Phase 1 progress, 2026-08-25 (3) — the PACKING is proved, and it says less than expected

`proofs/ExactGemmPacking.v` (Rocq 9.1, no axioms, nothing admitted) covers the
third obligation: `pack_a` and `pack_b` move a live `MR x kc` / `kc x NR` tile
into a contiguous panel, and the micro-kernel then runs the panel at **full**
width regardless of how ragged the tile was. Two theorems make that legal:

- `pack_a_slot_bijective` / `pack_b_slot_bijective` (with `_in_panel` and
  `_onto`): each destination map is a bijection onto its panel, so nothing is
  written twice and no slot is left holding the previous tile.
- `padded_product_is_the_live_dot_product`: the full `kpairs * 2` product over
  the padded panel equals the dot product over the live `kc`. This is THE
  theorem — it is what licenses running a ragged tile at full width — and it
  rests entirely on the packers' `select ... i16 0`.
  `garbage_in_the_pad_changes_the_answer` refutes the zero-fill-free version
  concretely, so the mask is machine-checked as load-bearing rather than
  asserted to be.

**The interesting result is negative, and it corrects a claim this repo was
making.** The emitted `pack_b` destination map is `(j/16)*32 + (j%16)*2 + h`,
written that way to document the `vpdpwssd` lane layout: group `v = j/16` is one
`<32 x i16>` vector and lane `l = j%16` inside it consumes int16 elements `2l`
and `2l+1`. The docstring and an earlier draft of the model test both claimed
the split was "not a convenience". It is: `16*(j/16) + (j mod 16) = j`, so the
whole decomposition folds and the map **is** the plain interleave `2*j + h`.
`slot_b_is_the_plain_interleave` states that as a theorem.

So the gap between what is proved and what is true is wider than the usual
"bijectivity cannot distinguish two layouts" — there are not two layouts. That
a hardware lane consumes elements `2l`/`2l+1` of its own vector is an ISA fact
with no arithmetic content, and it is pinned only by
`tests/cpu_gemm_vnni_micro.rs` running the real instruction against a scalar
reference. Writing the proof is what forced that to be said precisely.

**And the behavioural tie is weaker than it looks — measured, not assumed.**
`tests/exact_gemm_packing_model.rs` poisons the operand padding with
live-range values (a `calloc` buffer would supply the zeros that are the
property under test, the same trap as pre-zeroing C one section up) and runs
five shapes. Removing a packer's mask and sweeping them:

| mask removed | 53x71x301 | 48x128x301 | 53x128x300 | 48x71x300 | 48x128x300 |
|---|---|---|---|---|---|
| `pack_a` only | 0 | 0 | 0 | 0 | 0 |
| `pack_b` only | 0 | 0 | 0 | 0 | 0 |
| both | 3763 | 6144 | 0 | 0 | 0 |

Two facts, neither visible from reading the code:

1. **The two masks are redundant with each other**, and no driver can separate
   them: a padding term is `a_pad * b_pad`, so a zero on either side kills it.
   Removing one leaves the kernel correct and undefended, not wrong. The
   obligation the test really pins is the conjunction — *the padding
   contributes nothing* — which is exactly what the theorem states.
2. **Only the phantom k-half can corrupt an answer.** The ragged M and ragged N
   shapes report 0 with both masks gone, because those accumulator rows and
   columns are discarded by the C store mask before anything reads them — the
   property proved in `ExactGemmTiling.v`, doing double duty. So one of the five
   shapes is load-bearing and four are controls, and the test says so rather
   than implying five-fold coverage.

The row and column masks are therefore defence in depth against a future change
to the store, not correctness today. Recorded rather than deleted.

Mutation-verified 10/10 (three model mutations, three emitter mutations, three
proof-gate mutations, one kernel mutation), with three further mutations sorted
as **mis-aimed rather than survivors**: `replace(old, new, 1)` had been landing
on a docstring occurrence of the slot expression instead of the code. *A
survivor is a hypothesis about the test; check where the mutation landed before
recording it as a hole.*

With this, Phase 1's schedule is complete: the K-split reduction, the output
partition, and the operand packing. What remains unproved in the exact GEMM is
the micro-kernel itself — the 2-D register tile, the masked tails, and the
int32 accumulate with its periodic int64 flush, whose no-overflow obligation is
discharged exhaustively over the finite int16 domain in
`tests/exact_gemm_licence_obligations.rs` rather than by proof.

#### Phase 1 progress, 2026-08-25 (4) — the FLUSH is proved, and the licence is now known to be necessary

`proofs/ExactGemmMicro.v` (Rocq 9.1, no axioms, nothing admitted) closes the
last schedule obligation. Two things inside the micro-kernel are provable, and
the file says plainly which parts are not.

**The flush.** `vpdpwssd` accumulates into int32, which wraps. The kernel runs a
bounded number of k-pairs into int32 accumulators, then widens them into an
int64 running sum and re-zeroes them. `flush_exact` proves the chunks
`[c, min(c+F, kpairs))` sum to the whole range — with no hypothesis that `F`
divides `kpairs`, since the emitted `select` clamp carries the final partial
chunk. `flush_exact_in_int32` proves the int32 arithmetic (modelled with an
explicit `wrap32`) agrees with `Z` exactly when no partial sum leaves the
range, and `operand_bound_gives_no_overflow` derives that hypothesis from
`2 * F * m^2 <= i32::MAX` — which is precisely what
`VnniExact::max_operand_magnitude` computes. `the_licence_makes_the_chunk_exact`
composes the two.

Note the chunking is the **clamped-tile** shape of `ExactGemmTiling.v`, not the
uneven-band shape of `ExactGemmKsplit.v`: a flush interval is a fixed overflow
budget, not a work split. Confusing the two is how a schedule proof gets pointed
at the wrong theorem.

**The lane round trip, which is a CROSS-FILE obligation and the first of its
kind here.** `ExactGemmPacking.v` says `pack_b` puts column `j`'s `h`-th
k-value at panel slot `2j + h`. `vpdpwssd` consumes that slot in vector
`slot / 32`, lane `(slot mod 32) / 2`; the store writes vector `v` lane `l` to
column `16v + l`. `the_packed_column_is_the_stored_column` proves the
composition is the identity. **A mismatch here is a correctly-summed but
column-PERMUTED tile** — every partial sum right, every value in the wrong
place, and no bijection or bound in any other file able to see it. The control
`a_wrong_lane_stride_permutes_the_columns` stops the theorem being satisfied by
any pair of maps that happen to compose.

**The behavioural half answers a question none of the existing tests asked: is
the licence necessary, or is it paperwork?** `cpu_gemm_exact_threaded.rs`
already links three flush intervals into one process and compares them bit for
bit, so interval *invariance* was covered. Nothing checked the other direction.
Since `emit_vnni_micro_module` takes the interval directly, the emitted symbols
can be driven past the bound even though the compiler refuses to:

| operand magnitude | interval sum | exact answer | kernel returns |
|---|---|---|---|
| 4095 (licensed) | 2,146,435,200 | 2,146,435,200 | 2,146,435,200 |
| 4096 (refused) | 2,147,483,648 | 2,147,483,648 | **-2,147,483,648** |

`K = 128` is `kpairs = 64`, exactly one full default interval, with *constant*
operands because the bound is a worst case that random data never reaches. One
unit past the licence the int32 accumulator wraps to the negative of the right
answer. So the one-unit-wide boundary that
`tests/exact_gemm_licence_obligations.rs` finds by exhausting the int16 domain,
and that `the_4096_case_exceeds_by_exactly_one` states as arithmetic, is now
also observable in a running kernel. **A licence nothing can violate is
indistinguishable from a licence that certifies nothing.**

Mutation-verified 8/8, and the load-bearing one is M6: forcing the emitter to
flush every k-pair — so no overflow is possible — fails the refutation test.
Without that mutation, "the bound is real" and "the test cannot tell" look the
same.

One incidental finding: `malloc`/`free` are declared by `llvm_emitter`'s
prelude, not by either `cpu_gemm` module, so the modules are not
self-contained when emitted standalone. Not a bug — nothing in production
emits them that way — but `emit_vnni_threaded_module`'s `need_libc_decls` flag
does not cover them, which is worth knowing before writing another standalone
harness.

**Phase 1's schedule is now complete: K-split reduction, output partition,
operand packing, and the flush.** What remains unproved in the exact GEMM is
the 2-D register tile and the masked tails — a lane's contribution is taken as
given rather than derived from the packed panels, so nothing here says the
accumulator for row `i` is fed row `i`'s broadcast. That and the `vpdpwssd`
semantics themselves are ISA facts, pinned by `tests/cpu_gemm_vnni_micro.rs`
against a scalar reference.

#### Phase 1 progress, 2026-08-25 (5) — the REGISTER TILE's routing, and a compensating pair

`proofs/ExactGemmRegisterTile.v` (Rocq 9.1, 8 `Print Assumptions`, no axioms)
covers the routing half of the arithmetic core — the piece the previous entry
listed as remaining.

`the_lane_consumes_its_own_column` proves that the emitted broadcast, the four
`<32 x i16>` B loads at `Bp + p*NR*2 + v*32` and `vpdpwssd` send panel slot
`2j + h` to the accumulator lane whose store column is `j`. That is the join
between `ExactGemmPacking.v` (which says what a slot holds) and the kernel
(which says which lane reads it); neither is interesting alone, and a mismatch
between them is a correctly-summed, **column-permuted** tile.

`the_i32_load_is_the_packed_pair` proves the emitter's `getelementptr inbounds
i32` at element `p*MR + i` aliases exactly the packed int16 slots `2i` and
`2i+1` of group `p`. Which half is `2p` rather than `2p+1` is little-endianness
— an ISA fact, taken as a definition — and
`swapping_the_pair_halves_computes_a_different_function` shows the assumption is
load-bearing, paired with `a_symmetric_operand_hides_the_swap`, which is why
the fixture is asymmetric in every axis.

The masked tails are here too, and they explain a measurement from the packing
entry: `a_padding_column_never_reaches_c` and `a_padding_row_never_reaches_c`
are *why* the packers' row and column masks turned out redundant, while the
phantom k-half's mask — which contributes to a live column — did not.

**The behavioural tie drives `__y_gemm_micro_vnni` DIRECTLY with hand-built
panels, and the reason is demonstrated rather than asserted.** Every other
exact-GEMM test goes through the full driver, where the packers and the routing
are composed. A single-site mutation does not expose that: breaking the B
vector offset alone fails every GEMM test in the repo. A **compensating pair**
does — XOR the vector index in both `emit_vnni_pack_b` and the flush's store
column, and the two cancel:

| | packing_model | thread_invariance | cpu_gemm_exact_threaded | register_tile |
|---|---|---|---|---|
| compensating pair | ok | ok | ok | **FAILED** |

Building the panel by hand removes the composition, so there is nothing left
for a packer to compensate with.

**What is still not proved**, and the file says so: `vpdpwssd`'s own semantics
and the endianness are definitions, pinned empirically by
`tests/cpu_gemm_vnni_micro.rs` against a scalar reference. What Phase 1 now has
is the whole schedule plus the tile's routing; what it does not have is a
machine-checked model of the instruction itself, which no proof over `Z` can
supply.

#### Phase 1 progress, 2026-08-25 (6) — where the five proofs meet

`proofs/ExactGemmComposition.v` (Rocq 9.1, 7 `Print Assumptions`, no axioms)
does two things.

**It makes the five files agree.** Each was written self-contained, so three of
them define the B panel's slot map — `ExactGemmPacking.slot_b` as the emitted
`(j/16)*32 + (j%16)*2 + h`, `ExactGemmMicro.slot` and
`ExactGemmRegisterTile.slot_b` as the plain `2j + h` — and `MR`, `NR` and
`col_of` are each defined twice. Nothing checked that any of those denoted the
same thing.

**The first draft of that argument overstated the risk, and measuring it is
what showed so.** The header claimed each file would go on type-checking while
silently describing a different kernel. Three attempts to produce such a drift
— `RegisterTile.slot_b` shifted by one, its `NR` set to 32, `Micro.slot` with
the halves swapped — were all **caught by the file itself**, because each
definition happens to be pinned by a theorem that file already proves. So the
agreement theorems do not close a demonstrably open hole; they make an
incidental pinning explicit and cross-file. That is a smaller claim, and it is
the true one.

**The composition step is the new content.**
`the_lane_accumulates_the_source_elements` says one `vpdpwssd` step on the
packed panels contributes exactly the two k-terms of the *source* matrices' dot
product for `(row i, column 16v + l)`, masked — so a padding row, column or
k-half contributes zero. Packing says what a slot holds; the register tile says
which lane reads it; only together do they say the lane reads the right source
element. Neither file can state it alone, which is the reason the split was
worth making in the first place.

**The packers' contract is taken as a HYPOTHESIS, and naming it is the point.**
`ExactGemmPacking.v` proves the slot maps are bijections and that the padded
product equals the live dot product; it never states "slot `s` holds source
element `x`" as one reusable lemma. The composition assumes that explicitly.
That is the single step of Phase 1 which is assumed rather than proved, and it
is written down rather than buried in a definition.

**What is still not composed**, also stated in the file: this joins packing to
routing for *one* `vpdpwssd` step. It does not chain through the k-pair loop
into the flush, nor through the output partition, nor through the K-split, into
one "the emitted kernel equals the naive nest" theorem. Each of those is proved
over its own model. A genuine end-to-end statement needs a single shared model
of the kernel — which is Phase 2's subject, not a missing lemma here.

### Phase 2 — Turn the proof into a mechanism · 1–2 years

Phase 1 proves one kernel by hand. This makes it structural: a transformation
IR in which tiling, packing, vectorization, loop reordering and thread
partitioning are each *individually* obligation-carrying, composed
automatically. Adding an optimization means discharging its obligation, not
reproving the kernel.

- **Done when** — a second, structurally different kernel (attention, or a
  stencil) is verified with no new hand-written proof.
- **Hard part** — this is the research risk of the whole programme. If
  obligations don't compose, the thing is a one-off proof rather than a
  compiler.
- **Exit value** — a verified kernel compiler for CPU. Narrow, but complete and
  defensible.

### Phase 3 — Extend to the GPU pipeline · 2–3 years

The transformations that make GPU kernels fast are the ones most likely to be
silently wrong: `cp.async` staging, `ldmatrix`, `mma.sync`, XOR bank swizzling,
and the mbarrier discipline. Fold in `linear_tracker`, which already proves
async tokens are consumed exactly once — a property this backend was
*discarding* at the emitter until it was caught.

- **Done when** — a tensor-core GEMM carries a proof covering its memory
  pipeline and its swizzle, not only its arithmetic.
- **Trust boundary** — Y emits PTX; `ptxas` produces SASS and is closed source.
  The proof covers source-to-PTX, and `ptxas` is trusted or validated
  per-translation. **This must be stated in the certificate, never papered
  over.**

### Phase 4 — Bounded error where exactness is impossible · 3–4 years

Exact accumulation covers reductions and fixed-point pipelines. It does not
cover transcendentals, division, or normalization — and real kernels have all
three. For those, prove the implementation is within a *stated, machine-checked
bound* of the ideal real-number result, building on the interval arithmetic
already in the type checker.

- **Done when** — a softmax or a normalization layer carries a proven error
  bound rather than an empirical one.
- **Why it matters** — without this the addressable set is "kernels that are
  pure reductions", which is too narrow to build a company on.

### Phase 5 — Emit the certification packet · 4–6 years

A proof is not evidence until it is in the form an auditor accepts. Generate
the artifact set: the machine-checkable proof, traceability from source
requirement to emitted instruction, a statement of the trusted computing base,
and the tool-qualification material the standards require of the compiler
itself.

- **Done when** — one real certification effort accepts the packet as evidence.
- **Note** — tool qualification is itself a substantial programme; DO-330
  exists for exactly this. Budget it as a phase, not a formality.

---

## 5. End goal

You write the naive loop nest — the specification, readable and obviously
correct. The compiler emits a tiled, swizzled, async-pipelined AVX-512 or PTX
kernel, **and a certificate that the two compute the same thing.** Not a
benchmark. Not a test report. A proof an auditor can check without trusting you
or the compiler.

**Who buys it.** Safety-critical avionics, automotive and medical software
cannot currently certify GPU compute at all — which is why that hardware sits
idle in systems that would benefit from it enormously. The blocker is evidence,
not performance. Regulated finance and auditable ML have the same shape: the
deliverable is the argument, not the speed.

This is the one direction where the question that killed every other candidate
in this repository — *who gives this away free?* — has the answer **nobody**.
cuBLAS is free and irrelevant here, because it ships no certification artifact
and never will.

---

## 6. What could kill it

**Obligations may not compose.** Phase 2 is the real research risk. If each
optimization's proof cannot be discharged independently and composed, you have
one proof of one kernel and no compiler. This is where the programme most
plausibly stops.

**The trusted computing base never reaches zero.** `ptxas` is closed source;
LLVM is enormous. CompCert solved this for C by verifying the whole chain over
many years with a full team. You will be certifying source-to-IR and declaring
the rest trusted — honest and standard, but less than the pitch implies, and it
must be said plainly every time.

**Exact arithmetic is a real constraint.** Until Phase 4 lands, the addressable
set is reductions and fixed-point pipelines. If Phase 4 proves harder than
expected, the programme's ceiling is much lower than section 5 claims.

**Certification markets are slow and conservative.** They buy from established
vendors with track records, on multi-year cycles. A single-developer research
compiler is not a credible supplier to them, whatever the proof says. The
realistic path is partnering with or being acquired by an existing supplier —
plan for that rather than discovering it in year five.

**It is one person.** Four to six years is long enough that funding,
collaborators or an institutional home stop being optional. Phase 1's
publication is the instrument for getting them, which is why it should not
slip.

---

## 7. The first move

Delete four lines and find out what happens.

`src/cpu_gemm.rs:289` refuses a `@ZeroDrift` accumulator on the grounds that
substituting the packed kernel would discard the exactness guarantee. That
reasoning is correct *today*, because the packed kernel accumulates in `f32`.

Make the packed kernel accumulate exactly instead, and the refusal becomes
unnecessary. Then measure what it costs — **that number decides whether any of
the rest is worth doing.**
