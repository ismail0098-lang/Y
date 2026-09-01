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
| Certificate format | `exact_gemm_certificate.rs` | **real, emitted, gated** | Was "does not exist / Phase 5". See the 2026-08-30 entry. |

The single most useful fact in this table: `recognize_gemm` contains an
explicit refusal of exact accumulators (`src/cpu_gemm.rs:289`), with a comment
explaining that substituting one would silently discard the guarantee. **The
two halves have never met, and the code declines to introduce them.** That
refusal is Phase 0.

> **STATUS, 2026-08-31 — this table is the state at the programme's start and
> is kept as written.** Three of its rows have moved since. The refusal is
> gone: `recognize_gemm` substitutes the exact `vpdpwssd` kernel, and the
> "Kernel rewrite" row's *"explicitly refuses"* is history. The certificate
> format exists, is emitted per compilation, and is gated. And the proofs
> themselves went from zero to **fourteen files, ~215 theorems, no axioms**.
> What has NOT moved: the SMT discharge and the interval arithmetic are still
> narrow, and Phase 4's error bounds have not been started — they are not on
> the path, because exactness makes them unnecessary for this kernel family.
>
> The method is written up separately in
> [The process: taking a kernel from *fast* to *verified*](verified_kernel_process.md).

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

`proofs/ExactGemmComposition.v` (Rocq 9.1, 8 `Print Assumptions`, no axioms)
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

**The packers' contract was taken as a HYPOTHESIS, and naming it is what made
it get fixed.** `ExactGemmPacking.v` proved the slot maps are bijections and
that the padded product equals the live dot product; it never stated "slot `s`
holds source element `x`" as one reusable lemma, so the composition assumed
that explicitly and the file called it the single step of Phase 1 which is
assumed rather than proved. See the entry below — the assumption turned out to
be false of the real panel.

**What is still not composed**, also stated in the file: this joins packing to
routing for *one* `vpdpwssd` step. It does not chain through the k-pair loop
into the flush, nor through the output partition, nor through the K-split, into
one "the emitted kernel equals the naive nest" theorem. Each of those is proved
over its own model. A genuine end-to-end statement needs a single shared model
of the kernel — which is Phase 2's subject, not a missing lemma here.

#### Phase 1 progress, 2026-08-26 — the assumed step, and why it was wrong

**The seam the last entry named was closed, and trying to violate it first is
what found a defect in it.** The composition's two hypotheses said panel slot
`p*(2*MR) + slot_a i h` holds `A[i][2p+h]` masked, for **every** `i` — and no
real panel satisfies that. At `i = MR` the index named is the FIRST slot of
k-pair group `p+1`, which holds that group's data rather than a pad, while an
unbounded contract demands a zero there whenever row `MR` is past `mrows`.

So the composition theorem was true and unusable: applying it meant supplying a
premise satisfied only by a panel one k-pair group long, never by the panel the
emitted loop builds. **A hypothesis nothing is shown to satisfy is the
proof-shaped version of a licence nothing can violate** — a check this repo
already runs on its licences (`the_add_formula_really_is_incomplete`,
`garbage_in_the_pad_changes_the_answer`) and had not run on its own premises.

The fix is a bound plus a discharge:

- `ExactGemmPacking.panel` models the panel as a FUNCTION — decode the index
  into `(group, row-or-column, half)` — and `panel_decodes_its_own_write`
  proves the contract of it, with `idx < width`.
- `panel_is_the_only_solution` proves ANY array satisfying the packer's writes
  agrees with `panel` over the whole panel range. That is what makes the first
  a claim about the emitted loop rather than about a model chosen to make the
  composition go through, and it is where the bijection lemmas finally earn
  their keep at panel scale: injective ⇒ the write specification is consistent,
  onto ⇒ it is complete.
- `the_group_bound_is_load_bearing` refutes the unbounded form on concrete
  numbers, with `inside_the_group_the_same_panel_agrees` as the control.
- `the_packed_panels_route_to_the_right_source_elements` is the composition
  with no hypothesis about panel contents at all.

**The behavioural tie runs the real packers** (`tests/exact_gemm_panel_model.rs`):
`__y_gemm_vnni_pack_a` / `_pack_b` are called on poisoned panels over five
shapes (full, phantom k-half, ragged M, ragged N, all three) with every stride
differing from its extent, and **every slot** is compared against
`panel_slot_decode`. `the_next_group_is_not_padding` refutes the unbounded
contract against real bytes rather than against the model.

**It also isolates something no end-to-end test can.** The packing entry above
records that removing `pack_b`'s zero-fill *alone* leaves every answer correct,
because a padding term is `a_pad * b_pad` and a zero on either side kills it —
so the two masks could only ever be pinned as a conjunction. Measured, with
`pack_b`'s mask removed:

| | packing_model | thread_invariance | cpu_gemm_exact_threaded | tiling_model | panel_model |
|---|---|---|---|---|---|
| pack_b mask dropped | ok | ok | ok | ok | **FAILED** |

Mutation-verified 8/8: four against the emitted packers and the decode, four
against the proofs — including reverting the contract to its unbounded form,
which now fails where the previous commit compiled it happily.

**A gate bug fell out of writing the prose.** `nothing_in_any_proof_is_admitted`
stripped Coq comments LINE-LOCALLY, so a continuation line inside a multi-line
comment was scanned as code and the ordinary English word "admit" failed the
build. Contorting the prose would have left the hole. Probing the fixed gate
with what it exists to catch then found a second, older one: it tested
`code == "Admitted."`, `starts_with("admit")` and `contains(" admit.")`, and
`Proof. Admitted.` on ONE line — the commonest way to stub a Coq lemma — is
none of the three. It matches the TOKEN now, and covers `Abort.` as well, which
discards the lemma outright while the file still compiles.

#### Phase 1 progress, 2026-08-26 (2) — the chain closes for one lane

`proofs/ExactGemmChain.v` (Rocq 9.1, 11 `Print Assumptions`, no axioms) joins
four files into one statement. **`the_emitted_lane_computes_the_source_dot_product`**:
run the emitted flush schedule, accumulating each chunk in **int32** and adding
it into an int64 running sum, over the panels the emitted packers produce,
through the emitted routing — and lane `l` of vector `v` for row `i` holds
exactly `sum_k A[i][k] * B[k][16v+l]`, the dot product of the **source**
matrices.

No hypothesis about panel contents, none that the flush interval divides the
k-pair count, none that the tile is full. The one assumption is the **licence**,
`2·Fl·m² ≤ i32::MAX`, which is precisely what
`VnniExact::max_operand_magnitude` computes — and which
`tests/exact_gemm_licence_obligations.rs` discharges by exhausting the finite
int16 domain.

**The connective tissue is the new content, and it was not obviously going to
fit.** `kloop_is_the_padded_product` turns a loop of routing steps into the
padded product `ExactGemmPacking` already evaluates; `kloop_is_sum_from_step`
re-presents the same loop in the shape `ExactGemmMicro`'s flush theorem
quantifies over. Those two models were written independently — `sum_pairs`
counts k-*pairs* carrying both halves, `sum_from` is a flat range — so the join
is a small lemma each way rather than a definitional coincidence. That the
pieces met at all is the first evidence that the file split was the right
decomposition rather than five unrelated models.

**The licence is load-bearing for the whole chain, refuted symbolically and
behaviourally at the same numbers.** `violating_the_licence_breaks_the_chain`
evaluates the chain at operand magnitude 4096 with the shipped interval of 64
and gets **−2,147,483,648 where the answer is 2,147,483,648**;
`at_the_licensed_magnitude_the_chain_holds` is the same chain at 4095.
`tests/exact_gemm_chain_model.rs` reaches the identical pair on the real
`vpdpwssd` kernel — from **source** matrices through the **real packers**, where
`exact_gemm_micro_model.rs` reaches it only with hand-built panels.

**The behavioural tie also crosses more than one flush chunk**, which no sibling
model test does: `kc = 133` is 67 k-pairs against a 64-pair interval, so the
clamped final chunk is a real case. Mutating the flush to overwrite `C` rather
than accumulate — a bug invisible with a single chunk — is caught.

**And what it does not catch is recorded beside what it does.** Five emitter
mutations across every exact-GEMM suite:

| mutation | packing | panel | regtile | micro | thr-inv | exact-thr | chain |
|---|---|---|---|---|---|---|---|
| flush OVERWRITES C | FAIL | ok | ok | ok | FAIL | FAIL | **FAIL** |
| pack_b vector stride 32→16 | FAIL | FAIL | ok | ok | FAIL | FAIL | **FAIL** |
| pack_b zero-fill dropped | ok | FAIL | ok | ok | ok | ok | **FAIL** |
| compensating pair (pack_b `v^1` **and** store column `v^1`) | ok | FAIL | FAIL | ok | ok | ok | **ok** |

The last row is the point: **the chain test does not subsume the register-tile
test and cannot.** It composes packing with routing, so an inverse pair cancels
here exactly as in a full GEMM. Isolating it needs panels stated by the test, or
a panel checked slot by slot. *Adding a test does not retire the tests it
resembles.* (That row also dates the register-tile file's own table, written
before `exact_gemm_panel_model.rs` existed — the panel model catches the pair
too, by the other route.)

**What is still not chained**, stated in the file: this is one **lane**, not the
tile (the tile needs the store, over a different model of C); it does not reach
`C` through `ExactGemmTiling`'s output partition or `ExactGemmKsplit`'s band
reduction, both of which sit above it and are proved over their own models; and
`vpdpwssd`'s semantics plus i32 half-order remain definitions, pinned by
`tests/cpu_gemm_vnni_micro.rs` on the real instruction. Joining the remaining
layers still needs one shared model of the kernel — Phase 2's subject.

Mutation-verified 8/8: four against the emitter, four against the proof
(dropping the licence bound, dropping `0 < Fl`, seeding the accumulator at 1,
and reading the next k-pair group's base).

#### Phase 1 progress, 2026-08-26 (3) — the tile lift

`the_tile_holds_the_source_dot_products` takes the previous entry's statement
from one accumulator lane to the whole `MR × NR` tile: for every live `(i, j)`

> `C[i][j] = C0[i][j] + sum over k < kc of A[i][k] · B[k][j]`

and for every dead one `C[i][j] = C0[i][j]` exactly. Note the **accumulate** —
`C0` is what was there before, which is what lets a caller split K across
threads and is the property `ExactGemmKsplit.v` rests on.

**The join it needed is the inverse of `col_of`.**
`ExactGemmRegisterTile.tile_position_surjective` says an inverse exists;
`the_lane_map_is_a_two_sided_inverse` names it (`vec_of j = j/16`,
`lane_of j = j mod 16`) and proves it inverts *both* ways, which is what
licenses `distinct_columns_use_distinct_lanes` — no two tile positions share an
accumulator lane, so 384 values really do live in 24 registers of 16 lanes with
nothing aliasing.

**The store predicate turns out to do no work, and that is a theorem.** The
emitted micro-kernel writes all `MR × NR` positions unconditionally — the
live-rectangle clamp is the *driver's* — so the tile model was carrying two
things at once. `the_store_predicate_is_redundant` shows a dead row or column
accumulates zero by the packers' masks, so adding it to `C0` leaves `C0`. That
settles the faithfulness question (the tile theorem describes the micro-kernel's
own effect on C, clamp or no clamp) and it is the proved twin of a redundancy
`tests/exact_gemm_packing_model.rs` had only **measured** — removing a packer's
row or column mask leaves every answer correct because the clamp discards it.

**The behavioural gap this exposed is the more useful half.** The tile theorem
says `C0 + dot`, and the chain driver zeroed `C` — so the *accumulate clause was
never exercised*. A kernel that ASSIGNS instead of accumulating is
indistinguishable from a correct one when `C` starts at zero and there is one
flush chunk. `tests/exact_gemm_chain_model.rs` now runs every shape twice, with
`C` zeroed and with `C` pre-loaded, and the pre-load is load-bearing: with the
multi-chunk shapes removed so `kc = 133` cannot do the work, mutating the flush
to overwrite `C` is caught **only** on the pre-loaded arm, at the very first
shape. Pre-loading also makes "a dead position is left as it was" testable at
all — with `C` zeroed, *untouched* and *zeroed* are the same observation.

**What is still missing to reach the whole of C**, named rather than left to be
discovered: three layers sit above the tile. `ExactGemmTiling.v`'s output
partition and `ExactGemmKsplit.v`'s band reduction are each proved over their
own model and are not joined to this one. The third is proved **nowhere** — the
**kc-panel loop**, which cuts K into panels of `kc` inside a single thread. It
is the same decomposition shape as `ExactGemmKsplit.bands_tile`, so it is
probably cheap; "probably cheap" is not "done".

Mutation-verified 4/4 on the proof (`vec_of` at /32, `lane_of` at mod 8,
dropping the `j < NR` bound, dropping the store predicate) plus the behavioural
demonstration above. 18 `Print Assumptions`, no axioms.

Also: `proofs/` now gitignores Rocq build artifacts. The gate compiles in a temp
directory precisely so they never appear, but the proof headers tell a reader to
run `coqc` by hand, which drops a `.lia.cache` beside the source — as mine did.

#### Phase 1 progress, 2026-08-26 (4) — **all six proofs chained: the whole of C**

`proofs/ExactGemmWhole.v` (Rocq 9.1, 6 `Print Assumptions`, no axioms) states the
thing the six files were built for.
**`the_threaded_gemm_holds_the_source_dot_products`**: sum the partials of
`nthr` threads, each handed a K band, each running the emitted driver — packing,
the register tile's routing, the k-pair loop, the int32 flush, the scratch tile
and the fold-back — and for every `(r, c)` inside `M × N` the result is exactly

> `sum over k < K of A[r][k] · B[k][c]`

with **no** hypothesis that `MR` divides `M`, that `NR` divides `N`, that `nthr`
divides `K`, or that `K` is even. The only assumption is the licence,
`2·Fl·m² ≤ i32::MAX`, discharged by exhausting int16 in
`tests/exact_gemm_licence_obligations.rs`.

| file | contributes |
|---|---|
| `ExactGemmPacking.v` | what a packed panel slot holds |
| `ExactGemmRegisterTile.v` | which accumulator lane reads it |
| `ExactGemmComposition.v` | their join, for one `vpdpwssd` step |
| `ExactGemmMicro.v` | the int32 flush chunking |
| `ExactGemmChain.v` | the k-pair loop, and the lift to a tile |
| `ExactGemmTiling.v` | the output partition — **joined here** |
| `ExactGemmKsplit.v` | the K-band reduction — **joined here** |

**A correction to what the previous entry recorded as missing.** It named a
**kc-panel loop** — K cut into panels of `kc` inside one thread — as the layer
proved nowhere, and guessed it was "probably cheap" because it looked like
`bands_tile`. **There is no such loop.** `emit_vnni_gemm_driver` passes the full
`K` to both packers and `kpairs = (K+1)/2` to the micro-kernel; the only cut of
the K axis is the K-split across threads. So `kc` in every sibling file *is* K,
and the gap that actually remained was the fold-back into C. *Read the emitter
before recording a gap* — the guess was wrong in both directions, naming a layer
that does not exist while missing the one that did.

**No new behavioural test, and that is deliberate.**
`tests/exact_gemm_thread_invariance.rs` already runs M=53, N=71, K=4099 — all
three ragged against the `6 × 64` tile — at 1, 2, 3, 5, 8 and 16 threads, each
checked against an independent integer reference; `cpu_gemm_exact_threaded.rs`
sweeps flush intervals across the same; `exact_gemm_tiling_model.rs` covers a
padded `ldc`. That *is* this theorem's behavioural tie. A fourth near-duplicate
would add coverage of nothing.

**What is still not proved**, stated in the file: `vpdpwssd`'s semantics and the
little-endian order of an i32's halves remain **definitions** (pinned by
`tests/cpu_gemm_vnni_micro.rs` on the real instruction); and the loop
**structure** is modelled, not extracted — `gemm_position` says what `(r, c)`
receives on the reading that the driver visits tile `(r/MR, c/NR)` at offset
`(r mod MR, c mod NR)`. `the_position_decomposition_is_the_tilings` ties that to
`ExactGemmTiling.addr`, so `c_written_exactly_once` applies — but the tie is
between two models rather than to the emitted LLVM. **That is precisely the gap
Phase 2 exists to close**, and it is now the only structural one left.

Mutation-verified 4/4 (column offset at `mod MR`, dropping `r < M`, reading band
`t` rather than `t'`, scaling the row by `NR` in the address tie).

#### Phase 1 progress, 2026-08-26 (5) — narrowing the model-to-code gap

The capstone's one remaining structural gap is that the loop **structure** is
modelled rather than extracted: `gemm_position` says what `(r, c)` receives *on
the reading that* the driver visits tile `(r/MR, c/NR)`. Nothing had asked the
emitted driver how many tiles it runs, or in what order.

`tests/exact_gemm_tile_enumeration.rs` asks. **A correct tiling is invisible in
the answer** — `c_written_exactly_once` is exactly that statement — so the
answer cannot arbitrate it, the same bind the K-split was in when the
`--wrap=pthread_create` spawn count turned out to be the only observable. Here
the observable is the sequence of micro-kernel invocations.

`--wrap` does not work for it: `__y_gemm_micro_vnni` is called from the driver
in the *same module*, so the call is resolved at compile time and there is no
relocation to redirect. Instead the emitted module has that definition
**excised** and replaced by a `declare`, and the C driver supplies a recording
stub; the driver under test is byte-identical otherwise. Three things are then
pinned:

1. the call count is `mn_tiles(M, MR).len() × mn_tiles(N, NR).len()`;
2. the **order** is column-panel outer, row-panel inner — the row-tile sequence
   is `0..ntiles_m` repeated `ntiles_n` times, which is what makes "B is packed
   once per column panel" a checked fact rather than a comment;
3. every call receives the full `kpairs = (K+1)/2` and `ldc = NR` — the
   machine-checked form of the fact the previous entry got wrong from
   structural resemblance.

**What it isolates is less than it first looked, and the table says so.** Five
driver mutations:

| mutation | tile_enum | tiling_model | thr-inv | exact-thr | packing | chain |
|---|---|---|---|---|---|---|
| row loop strides by `MR-1` | caught | caught | caught | caught | – | – |
| column loop strides by `2·NR` | caught | caught | caught | caught | – | – |
| micro-kernel handed half the k-pairs | caught | caught | caught | caught | – | – |
| scratch row stride is `ldc`, not `NR` | caught | caught | caught | caught | – | – |
| **one extra row panel past M** | **caught** | ok | caught | ok | ok | ok |

The first four are caught by everything — they change the answer, and the
correctness suites see that. The last row is what the file earns its place on:
an extra panel is clamped to zero width, so the fold-back writes nothing and
**the answer is unchanged**; four of six suites miss it. So the honest claim is
"covers enumeration errors that do not change the answer, and diagnoses the rest
by name rather than as a bad number" — not "catches what the others cannot".

Also, the shared-temp-dir race, hit for the **fourth** time. It is a property of
the `build()` helper rather than of any one test, so the per-test tag now lives
in the signature instead of in a comment asking the next author to remember.

#### Phase 1 progress, 2026-08-26 (6) — the other half of the driver, and an invisible out-of-bounds read

Entry (5) tied the driver's *tile* loop to the model. Its other loop — the one
that prepares the panels — was still guarded by nothing but a paragraph of
prose. `emit_vnni_gemm_driver`'s own comment states the schedule as a measured
fact:

> The first version packed A inside the i-loop and B inside the j-loop nested
> within it, so B was re-packed once per ROW panel — `M/MR` times over … At
> MR=6 that is a sixth of the total run spent copying B.

**A re-packing bug is invisible in the answer.** Packing A inside the j-loop, or
B inside the i-loop, computes exactly the right result — it just does `ntiles_n`
or `ntiles_m` times the packing work. So no correctness test in this repository
can fail on it. Same shape as `a_deep_constant_chain_collapses_completely` (a
convergence property, guarded by a *size* assertion) and the shared-memory
swizzle (a bank-conflict property, guarded by a measurement).

`tests/exact_gemm_packing_schedule.rs` reuses (5)'s technique on all three
callees at once — `__y_gemm_vnni_pack_a`, `__y_gemm_vnni_pack_b` and
`__y_gemm_micro_vnni` are excised and replaced by `declare`s, with recording C
stubs. Recording all three in **one event stream** is what makes the schedule
checkable rather than three separate counts; the whole of it is one assertion:

```text
[A(0) .. A(ntm-1)]  ++  concat over j of ( [B(j)] ++ [K(0) .. K(ntm-1)] )
```

which says at once that every A pack precedes every B pack, that there are `ntm`
and `ntn` of them rather than `ntm × ntn`, and that each B panel is **live for
exactly its own column's tiles** — not merely written the right number of times.
(The two packers are `internal`, so the excision has to match `define internal
void @…` as well; a `declare` is never `internal`.)

**Six mutations, each suite run separately.** `cargo test` aborts the remaining
binaries after one fails, so a run listing several `--test` targets can leave
the important one unmeasured.

| mutation | pack_sched | tile_enum | tiling_model | thr-inv | exact-thr | packing_model |
|---|---|---|---|---|---|---|
| **A packed inside the j-loop** | **caught** | ok | ok | ok | ok | ok |
| **B packed inside the i-loop** | **caught** | ok | ok | ok | ok | ok |
| **pack_a's row count unclamped (`MR`)** | **caught** | ok | ok | ok | ok | ok |
| **pack_b's column count unclamped (`NR`)** | **caught** | ok | ok | ok | ok | ok |
| A panel offset is the row index, not the tile index | caught | ok | caught | caught | caught | caught |
| both packers handed `K-1` | caught | ok | caught | caught | caught | caught |

**Four of six are caught by this file and nothing else, and the two I did not
predict are the more serious pair.** I expected the unclamped-packer mutations
to be caught by the correctness suites and wrote that into the docstring before
measuring; they are not. Dropping either packer's clamp *at the call site* makes
it read up to `MR-1` rows past the end of `A`, or `NR-1` columns past the end of
`B` — and every answer stays bit-identical, because the fold-back's own `mw`/`nw`
clamp discards exactly the rows and columns the packer over-read. **The
observable consequence of a live out-of-bounds read is nothing at all**, until
the buffer happens to end at a page boundary. That is the redundant-guard
pattern `exact_gemm_packing_model` already records for the masks *inside* the
packers, one layer up at the call site — and it is why "the correctness suites
cover the packers" is false.

`exact_gemm_tile_enumeration` catches **none** of the six, which is the result
that says the two files are complementary rather than overlapping: it excises the
micro-kernel only, so what the packers are handed, and in what order, is
invisible to it. Between them they cover the driver's two loops — that one the
tiles it visits, this one the panels it prepares.

Also, the shared-temp-dir race, hit for the **fifth** time — and this time in a
file written two entries ago (`exact_gemm_panel_model.rs`), which had been green
for two commits before a new test binary changed the scheduling enough to fire
it. That is the "passed for several runs before failing" signature exactly. The
rest of `tests/` was then swept for the same defect rather than waiting for a
sixth: two candidates turned out to be false positives (per-`name` filenames with
no `remove_dir_all`, and a `main(` inside an embedded Y source string), so the
suite is now clean.

#### Phase 1 progress, 2026-08-27 — the schedule has ONE source, and a written measurement was wrong

Phase 1 is mathematically complete, and the gap it names about itself is that
the loop **structure** is modelled rather than extracted — "the tie is between
two models rather than to the LLVM". That gap has two halves. The extraction
half is Phase 2. The other half is cheaper and was never stated: the schedule
existed in **three** places with nothing structurally forcing agreement.

- `src/cpu_gemm.rs`, where the emitter reads it.
- The proof files, each declaring its own copy — `MR` and `NR` twice, `col_of`
  twice, the B slot map three times, and `DEFAULT_FLUSH_K_PAIRS` as a bare
  `64` inside two theorem *statements*.
- `proofs/ExactGemmComposition.v`, which asserts the copies agree.

That third one is a theorem somebody remembered to write.

**The measurement came first, and it corrected a claim this repo had already
written down.** `ExactGemmComposition.v`'s header records three attempts —
`RegisterTile.slot_b` shifted by one, its `NR` set to 32, `Micro.slot` with the
halves swapped — all caught by the file itself, and concludes that "each
definition turns out to be pinned by a theorem in its own file". Re-running the
sweep over *every* duplicated definition confirms those three and adds two more
(`RegisterTile.NRV` 4→2, `Packing.NR` 64→32) — and finds the exception nobody
tried:

> **`MR` set to 8, in EITHER `ExactGemmPacking` or `ExactGemmRegisterTile`,
> leaves that file compiling perfectly.** Both are caught only by
> `ExactGemmComposition.v` and by the downstream chain.

So the pinning was incidental for five definitions of six and **absent for the
sixth**. The honest claim for this work is therefore "makes the pinning
structural instead of incidental, and closes one constant that had none" — not
"fixes a live drift bug". Nothing had drifted.

**`proofs/ExactGemmSchedule.v` is generated from `cpu_gemm.rs`'s own constants,
committed, and gated on byte-identity** by `tests/exact_gemm_schedule_proof.rs`.
Every sibling proof takes its constants and index maps from it. Same shape as
`tools/extract_poseidon.py` → `src/zk_poseidon_constants.rs`, and as
`tests/committed_ptx_artifacts.rs` for the `.ptx` corpus.

**The generator LINKS rather than PARSES, and that is the whole design.** A
`tools/` script would have to recover `VNNI_MR` with a regex over `cpu_gemm.rs`
— a fourth copy of the value, living in the generator, which is the bug and not
the fix. `extract_poseidon.py` parses because its input is *foreign*
(circomlib's `.circom`); this input is our own Rust and can be `use`d. That
leaves `[[bin]]` or `#[test]`; a fifth `[[bin]]` is a permanent user surface
(`tests/source_surface.rs` gates those) bought for a file regenerated twice a
year, so it is a `#[test]` with `Y_REWRITE_SCHEDULE_PROOF=1`. There is no
`build.rs`.

**The content-control collision was a real design decision, not an obstacle.**
`tests/proofs_are_checked.rs::every_proof_has_a_content_control` rejects a `.v`
that names no load-bearing theorem — so a definitions-only generated file fails
it. It takes **no exemption**. Two of its three theorems are *structural*: they
constrain the shape of the emitted expressions rather than restating their
values, so they are not made true merely by being generated alongside what they
describe.

- `slot_b_is_the_plain_interleave` — the two sides come from different places
  in `cpu_gemm.rs` (`pack_b_slot`, and the bare `/2` that `panel_slot_decode`
  inverts it with). The Rust asserts their agreement in a doc comment; this
  proves it.
- `the_tile_geometry_is_consistent` — `NR = NRV * LANES`, `VEC_ELEMS = 2*LANES`
  and no degenerate zero. `LANES` is *derived* in the generator as
  `VNNI_NR / VNNI_NRV` rather than written down, so it cannot become a fourth
  copy of 16.
- `ksplit_threads_is_never_zero` — a genuine cross-file join. Every theorem in
  `ExactGemmKsplit.v` is stated under `0 < nthr` and `ksplit_bands` asserts the
  same precondition at runtime; nothing proved the emitted thread count
  satisfies it. The floor was argued in a comment.

The fourth, `the_schedule_is_the_shipped_one`, is **self-fulfilling under
generation and the file says so**. Its job is the other direction: it makes the
values load-bearing inside `coqc`, so a hand-edit fails twice.

**`slot_b` and `slot_b_interleave` are deliberately NOT collapsed.** The
generated file carries the B map in both the emitted `vpdpwssd` vector-group
form and the plain interleave. Collapsing them would make
`slot_b_is_the_plain_interleave` and
`ExactGemmComposition.the_agreement_is_not_vacuous` true by `reflexivity` and
worth nothing.

**Two theorem statements changed text, declared rather than slipped through.**
`ExactGemmMicro.the_default_interval_licenses_4095_and_not_4096` and
`the_4096_case_exceeds_by_exactly_one` carried the flush interval as a literal
`64` — the only place in the nine proofs where a schedule constant sat inside a
*statement*. They now read `ExactGemmSchedule.FLUSH_K_PAIRS`. Same proposition
at the shipped value; the difference is that a drift in
`DEFAULT_FLUSH_K_PAIRS` now makes them FALSE and fails `coqc` instead of
quietly pinning an interval the compiler no longer uses. 4095 and 4096 stay
literals on purpose — they are the *licence's* answer at that interval
(`floor(sqrt(i32::MAX / 2Fl))`), not a schedule constant, and stating them is
what makes the edge one unit wide rather than a tautology. Everything else is
byte-identical in statement; the sibling files gained only `*_unfold` helper
lemmas proved by `reflexivity`, which restate a generated definition in the
shape a tactic script manipulates.

**What it does not close.** The loop **nest** is still hand-written `IrBuilder`
calls in `cpu_gemm.rs`. This closes constant drift, not extraction.

**The mutation table**, thirteen exact-GEMM suites, each `--test` target run
separately:

| mutation | schedule gate | `proofs_are_checked` | the other 11 |
|---|---|---|---|
| Rust `VNNI_MR` 6→8 | **FAIL** | ok | 6 FAIL |
| Rust `KSPLIT_MIN_BAND` 128→256 | **FAIL** | ok | all ok |
| committed `.v` hand-edited, `MR := 8` | **FAIL** | **FAIL** | all ok |
| `RegisterTile.NR := SCH.MR` | ok | **FAIL** | all ok |
| `render` echoes the committed file | **FAIL** | ok | all ok |
| `render` hardcodes `MR := 6` | **FAIL** | ok | all ok |

`KSPLIT_MIN_BAND` **is caught by nothing else** — the clearest single
justification. A hand-edited `.v` is caught only by the two gates, because every
model test drives the Rust and never reads the proofs; that is the Coq-side half
of the drift, previously uncovered. The alias mistake is correctly *not* caught
by byte-identity — the generated file is untouched — and is what demonstrates
that the agreement theorems' weakened claim is still a real one.

**One mutation survived and was sorted rather than recorded**, and it was a
hole in the control written to prevent exactly this. The first version of
`the_generated_text_actually_depends_on_the_rust_constants` asserted the output
*contains* `Definition MR : nat := 6.` — and a generator neutered to echo the
committed file passes that, because the committed file contains that line. The
gate became a check on a constant string and the whole run stayed green. It now
renders a **perturbed** schedule and requires the result to differ from the
committed text, which an echoing generator cannot do.
`feedback-null-metrics-pass-dead-components`, committed by me, in the control
written to apply it — for the second time in this file's history.

A second survivor was genuinely mis-aimed: `Schedule::shipped` reading `mr: 6`
instead of `mr: VNNI_MR` changes nothing observable, since the two are the same
number. Hardcode `6` *and* move `VNNI_MR` to 8, or read the wrong constant
outright, and the tie assertion fails in both — a hardcode that agrees today is
caught the moment it stops agreeing.

#### Phase 1 progress, 2026-08-27 (2) — the mutation row was misattributed, and the register bound was prose

The entry above records `VNNI_MR` 6→8 as caught by "6 model suites". **Diagnosing
why produced a different answer, and it corrects that row.**

**The kernel is not wrong at MR=8.** `exact_gemm_thread_invariance` — the suite
that runs the real threaded kernel at ragged shapes against an independent
integer reference and compares bit-identically across thread counts — **passes**.
So whatever those six suites were reporting, it was not a wrong answer.

**Seven harnesses carried their own copy of the tile shape.** Each embeds a C
driver with `#define MR 6` / `#define NR 64` hardcoded, while its Rust half
reads `VNNI_MR`/`VNNI_NR` from the crate. So moving the constant did not make
those tests report a schedule mismatch — it made each test **disagree with
itself**: `exact_gemm_panel_model` compared a 48-element panel against a
64-element expectation, and `exact_gemm_register_tile_model`'s child process
simply crashed (empty stdout, no `DONE`) on buffers sized for the wrong tile.

That is the same defect `ExactGemmSchedule.v` was built to remove, one layer
down and in the half of the harness that allocates the memory the emitted kernel
writes into. All seven now take the constants from `cpu_gemm.rs`:
`std::fs::write(&drv, schedule_defines() + DRIVER)`. Prepending rather than
templating the whole driver, because C source is full of braces and `format!`
is not the right tool for it.

**With the harnesses fixed, six of the eight pass at MR=8** — confirming the
original signal was self-disagreement. What still fails is the real constraint:

> `cpu_gemm_vnni_micro::the_hot_loop_does_not_spill_the_accumulators` —
> *"hot-loop stack traffic regressed to 17 spills + 17 reloads; it was 10 + 10
> when this bound was set"*.

**The constraint existed, as a comment.** That test's own prose says "24
accumulators + 4 B vectors + 1 A broadcast is 29 of 32 zmm, so the allocator has
almost no slack". Nothing stated it as a property of the schedule — not
`cpu_gemm.rs`, not any of the nine proofs.

**The predicate is measured, not guessed.** Sweeping `VNNI_MR` and reading real
compiled spill traffic:

| MR | `MR*NRV + NRV + 1` | hot-loop spills + reloads |
|---|---|---|
| 5 | 25/32 | within bound |
| 6 | 29/32 | within bound (10 + 10, the shipped kernel) |
| 7 | 33/32 | 16 + 16 |
| 8 | 37/32 | 17 + 17 |

The cliff falls exactly where the inequality flips, so the *form* of the bound
is the measurement's rather than an invention. It is now
`ExactGemmSchedule.the_tile_fits_the_register_file`, and it **bites**: with
`VNNI_MR = 8` the regenerated file fails `coqc` at that theorem's `lia`, so the
generator cannot emit a schedule that does not fit the register file. Nine
theorems about an unallocatable tile is not a state this should be able to reach.

`ZMM_REGISTERS = 32` is emitted as an **ISA fact, not a schedule constant** —
there is no `cpu_gemm.rs` constant for it and no proof over `nat` establishes
it. It sits at the same TCB boundary as `vpdpwssd`'s semantics, pinned
empirically by the spill test reading real compiled output.

Note what the theorem does *not* claim: a spilling kernel is **slow, not wrong**.
Separating those two is what the `thread_invariance` result above is for, and it
is why the bound is a realizability constraint rather than a correctness one.

**One remaining MR=8 failure was left alone, deliberately.**
`exact_gemm_tiling_model::the_unclamped_tail_would_write_past_the_end` asserts
`last_off == 48` against a computed `8 * VNNI_MR`. That looks like an eighth
hardcode and is not: it mirrors `ExactGemmTiling.unclamped_tail_writes_out_of_bounds`'s
**concrete counterexample**, which is stated at MR = 6. Parameterising it would
make the assertion `8*VNNI_MR == 8*VNNI_MR` and worth nothing. Failing when the
schedule moves is the correct behaviour — it says the proof's concrete
refutation no longer matches the shipped tile.

#### Phase 1 progress, 2026-08-27 (3) — the first slice of the loop nest is EXTRACTED, not modelled

The gap Phase 1 names about itself is that the loop **structure** is modelled
rather than extracted, so the tie is between two models. This closes that for
one slice, using the same move that closed constant drift: **one description,
two consumers** — not by proving a translator correct, which needs an LLVM
semantics and would eat the programme, but by removing the second description.

`cpu_gemm::Ix` is a small index expression rendered two ways from one value:
`Ix::emit` produces the driver's LLVM, `Ix::coq` produces the definitions in
`proofs/ExactGemmSchedule.v`. Two expressions are covered:

- `tile_width_ix()` = `min(ext - iv, T)` — the clamped live width of a tile,
  at three sites (pack-A, the j-loop, the i-loop).
- `panel_index_ix()` = `iv / T` — which packed panel a tile reads, at two.

That is deliberately the arithmetic §1 of this document says the bugs live in:
*"twelve address computations in the CPU GEMM were correct only because
`lda == K` made stride and extent the same number"*. Which loops exist, in what
order, and what they call is still hand-written, and so are the k-split bands
and the flush chunking.

**The refactor is proved faithful by byte-identity**: all three emitted modules
(`emit_vnni_gemm_module`, `emit_vnni_threaded_module`, `emit_vnni_micro_module`)
are byte-for-byte what they were before. The driver used to spell the clamp as
three separate `IrBuilder` calls; it now renders it from the shared expression
and emits the same instructions.

Two new theorems are the **join** that was previously implicit. `tw` is stated
over the tile INDEX and the emitted loop has the induction variable instead:

- `the_emitted_width_is_the_tiling_model_at_the_loop_variable` —
  `tw ext T t = tile_width ext (toff T t) T`.
- `the_emitted_panel_index_is_the_tile_index` —
  `panel_index (toff T t) T = t`, i.e. the emitted `sdiv iv, T` really does
  recover the tile index, so a tile reads its own panel.

**The gate had to become universal, and mutation is what showed it.** The first
version searched for the rendered instruction sequence *somewhere* in the
module. That is satisfied while one site of three diverges — and the
discriminating mutation proves it: swapping the `min` operands at the i-loop
computes the **same value** by different instructions, and it passed the gate
*and* all five correctness suites. Counting the occurrences per site turns the
existential into a universal, and that mutation is now caught **by this gate
alone**, because no answer can see it.

The honest limits of the other mutations, measured rather than assumed:

| mutation | schedule gate | correctness suites |
|---|---|---|
| shared expression reversed, `.v` stale | **FAIL** | ok |
| shared expression reversed, `.v` regenerated | **FAIL** (and `coqc` FAILS) | ok |
| driver bypasses helper, reversed clamp | **FAIL** | 3 FAIL |
| driver drops one clamp | **FAIL** (after counting) | 1 FAIL |
| **min operands swapped — same value** | **FAIL** (after counting) | **all ok** |
| driver bypasses helper, *identical* clamp | ok | ok |

The third and fourth rows are worth reading as limits: the gate does **not**
extend coverage there — a reversed or missing clamp is a wrong answer and the
correctness suites catch it too. What the gate adds for those is a diagnosis by
name instead of a bad number. Its unique coverage is the fifth row.

The last row is a design confirmation, not a gap: a hand-written clamp that is
*identical* passes, and should — there is no divergence. The gate checks the
property (the proof's arithmetic is the emitter's arithmetic), not the plumbing
(that a particular helper was called).

#### Phase 1 progress, 2026-08-27 (4) — the flush and the K-split bands are extracted too

The two slices named as next in the entry above. `ExactGemmSchedule.v` already
held `cw`/`nchunks` and `blen`/`boff`; the emitter did not consume them. It does
now, through the same `Ix` layer.

**These sites are emitted as RAW LLVM, not through `IrBuilder`**, with
hand-chosen register names (`%cend0`, `%base`, `%klen`) — so extracting them
needed a second renderer, `render_named`, which takes the result names in
emission order. Supplying the names the emitter already used is what let the
refactor be checked by byte-identity: all three modules are byte-for-byte
unchanged, again.

Three expressions, and the split between them is forced by the code rather than
chosen:

- `chunk_end_ix() = min(iv + T, ext)` — the micro-kernel's flush clamp. Note the
  emitter computes an **end** where `cw` computes a **width**; they are the same
  clamp from two sides, and `the_emitted_chunk_end_is_the_flush_model` is that
  identity.
- `band_base_ix() = K / nthr` and `band_rem_ix() = K mod nthr` — loop-invariant,
  so the emitted wrapper hoists both into `many:`.
- `band_len_ix() = base + (if t < rem then 1 else 0)` — in `spawn.body:`, over
  the already-hoisted terms.

**An expression split across basic blocks is not one contiguous instruction
sequence**, and modelling it as one would have changed the emitted code. Hence
three expressions and a composition theorem rather than a single term:
`the_emitted_band_length_is_the_ksplit_model` recomposes them and ties the
result to `blen`. Every theorem in `ExactGemmKsplit.v` is stated about `blen`;
that is what now says the emitted spawn loop computes it.

The gate for these is **simpler and stronger** than the driver's: those
emitters choose their own register names, so there is nothing to normalise and
the rendered text is compared verbatim.

Mutations, each `--test` target run separately:

| mutation | schedule gate | correctness suites |
|---|---|---|
| **flush clamp operands swapped — same value** | **FAIL** | **all ok** |
| band's extra k moved to the last bands | FAIL | `thread_invariance` FAIL |
| micro-kernel bypasses the helper, *identical* text | ok | ok |

The first row is the one worth having, and it is the second time this session a
same-value divergence has been caught by this gate and by nothing else. The
second row was **mislabelled in my own sweep as "same total" and is not**:
flipping the condition gives `nthr*base + (nthr - rem)`, which equals `K` only
when `rem = nthr - rem`, so it breaks `bands_tile` and the answer moves. Worth
recording because the label was a guess and the measurement corrected it. The
third row is the design confirmation — the gate checks the property, not that a
particular helper was called.

**Where the extraction now stands.** The schedule's *arithmetic* is extracted:
the output tiling's width and panel index, the flush chunking, and the K-split
bands. What is still hand-written is the loop nest's **shape** — which loops
exist, in what order, which blocks they live in, and what they call. That is a
larger change than an expression layer and is squarely Phase 2.

#### Phase 1 progress, 2026-08-27 (5) — a sweep for un-extracted arithmetic, and the clearest isolation result yet

Rather than start on the loop nest's shape, the emitted IR was **swept** for
arithmetic carrying a schedule literal. Three model functions turned out to
exist in `ExactGemmSchedule.v` and be computed independently by the emitter:
`kpairs`, `ntiles`, and `a_i32_element`. The first two are now extracted.

**`kpairs` is the one that mattered.** It is spelled at **five** sites — both
packers, the driver, and twice in the threaded wrapper — and *every packing and
flush theorem is stated in terms of it*. `ExactGemmPacking.kpairs` is
`SCH.kpairs`; `padded_product_is_the_live_dot_product` quantifies over
`kpairs kc`; the micro-kernel's flush decomposes `kpairs`. Nothing said the
compiler computed the same number. `Definition kpairs` is now rendered from
`cpu_gemm::kpairs_ix`, and the five sites emit from it.

`tile_count_ix` covers the threaded wrapper's `(M + MR - 1) / MR`. `T - 1` is a
separate bound name rather than a subterm, because **the emitter folds it at
compile time into a literal** (`add i64 %M, 5`) — modelling it as `Sub(T, 1)`
would emit an instruction the compiler does not.
`the_emitted_tile_count_is_the_tiling_model` states that folding and carries
`0 < T`, because `nat` subtraction truncates and the two disagree at `T = 0`.

All three modules remain byte-for-byte unchanged.

| mutation | schedule gate | correctness suites |
|---|---|---|
| `kpairs` rounds down | FAIL | 4 FAIL |
| one `kpairs` site bypasses the helper, rounds down | FAIL | 3 FAIL |
| **`tile_count` uses `T` instead of `T-1`** | **FAIL** | **all ok** |

**The third row is the clearest isolation result in this series, and the reason
is worth stating exactly.** `%mtiles` feeds *only* `malloc` sizes — `%apn = mul
%mtiles, %kps`, then a byte count, then `malloc`. Using `T` instead of `T-1`
computes `(M + 6)/6` where the model says `(M + 5)/6`: never smaller, so the
buffer is **over-allocated**. An over-allocation produces no wrong answer, no
crash, and no observable symptom at all. It is precisely a divergence between
the proof's arithmetic and the emitter's that no test of the *result* can ever
see — which is the case this whole layer exists for.

That is now the third such case (after the driver's swapped `min` operands and
the flush clamp's), and together they are the argument for the layer: the
correctness suites catch everything that changes the answer, and these catch the
class that does not.

**What is left un-extracted, named rather than left to be rediscovered.**
`a_i32_element` (`p * MR + i`, the micro-kernel's A load index) is still spelled
independently — it lives inside a fully unrolled `MR × NRV` emission loop where
the index is a Rust-side constant per iteration, not an emitted expression, so
it is a different shape of problem. And the loop nest's **structure** — which
loops exist, in what order, in which blocks, calling what — remains hand-written.
That is Phase 2.

#### Phase 1 progress, 2026-08-29 — the last named leftover, and a correction to how it was named

The previous entry closed with `a_i32_element` recorded as un-extracted and
described it as "a different shape of problem — the index is a Rust-side
constant per iteration, not an emitted expression". **That description was
wrong, and reading the emitter is what settled it.** The micro-kernel emits

    %aidx = mul i64 %p, 6
    %ai0  = add i64 %aidx, 0
    ...   through %ai5

Both instructions are emitted. What is Rust-side is one *operand* of the `add`
— exactly as `T` is a Rust-side constant operand of `tile_width_ix`, which was
extracted three entries ago without difficulty. A constant operand is not a
reason an expression cannot be extracted.

Extracted as two expressions, `a_row_base_ix` (`p * MR`) and
`a_i32_element_ix` (`base + i`), for the reason the K-split bands needed the
same split: the base is loop-invariant across the `MR`-way unroll and the
emitter **hoists** it, so the two are not one contiguous instruction sequence.
`Ix` gained a `Mul` variant to express it. All three emitted modules are
byte-for-byte unchanged.

**Why this one is worth having rather than tidy.**
`ExactGemmRegisterTile.the_i32_load_is_the_packed_pair` proves the i32 load at
element `a_i32_element p i` aliases packed slots `2i` and `2i+1` of k-pair
group `p` — the little-endian half-order being the ISA fact the whole
register-tile file rests on. That theorem says nothing whatever about the
compiler. `the_emitted_a_index_is_the_pair_element` is what says the compiler
computes that element. Same shape as `kpairs`: a theorem stated in terms of a
number nothing confirmed the emitter produced.

**The join BITES at two independent layers, checked rather than assumed.**
Byte-identity catches a change to the Rust `Ix`; and hand-editing the committed
`.v` so `a_base` reads `p * MR + 1` fails `coqc` at the theorem
(`Unable to unify "a_elem (a_base p MR) i" with "a_i32_element p i"`). A
`reflexivity` proof is not thereby paperwork — it is the statement that two
independently-reachable definitions coincide, and it stops holding the moment
one of them moves.

**MUTATION TABLE**, each `--test` target run separately from a bash script.

| mutation | schedule gate | correctness suites |
|---|---|---|
| A row stride is `NR`, not `MR` | FAIL | 6 FAIL |
| **`%aidx = mul i64 6, %p` — operands swapped by hand** | **FAIL** | **all ok** |
| hand-written `mul` with IDENTICAL text | ok | ok |

The middle row is the **fourth same-value divergence** in this series caught by
the schedule gate and by nothing else, and the count is now the argument rather
than the anecdote: the correctness suites catch every mutation that changes the
answer, and this layer catches the class that does not. The third row is the
design control — the gate checks the *property* (the proof's arithmetic is the
emitter's arithmetic), not the *plumbing* (that a helper was called), so a
hand-written identical sequence must pass and does.

**Where extraction now stands.** Every schedule *number* the emitter computes
is rendered from one description: tile width, panel index, tile count, k-pair
count, flush chunking, K-split bands, and now the packed-A element index. The
loop nest's **shape** — which loops exist, in what order, in which basic
blocks, calling what — is still hand-written `IrBuilder` and raw-string
emission. That is Phase 2, and it is a materially larger change than an
expression layer: it needs the emitter restructured around a description of the
nest rather than around the instructions, and nothing above it can be checked
by byte-identity in the same cheap way.

#### Phase 2's decisive experiment, run early on a kernel that already existed · 2026-08-29

Phase 2's "Done when" is *a second, structurally different kernel verified with
no new hand-written proof*, and its stated risk is *"if obligations don't
compose, the thing is a one-off proof rather than a compiler."* That question
was answerable now, because **`src/cpu_gemm.rs` already emits two GEMMs.**
Beside the exact `vpdpwssd` one that nine proofs are about sits
`__y_sgemm_f32_avx512` — the kernel that ships for ordinary Y programs — which
partitions the same three axes and had **no proofs at all**.

`proofs/GemmBandSplit.v` (Rocq 9.1, 8 `Print Assumptions`, no axioms) and
`tests/f32_band_split_model.rs` are that experiment. The answer is a split, and
the split is the deliverable:

| layer | transferred? |
|---|---|
| range folding (`acc_range`, `sum_range_split`) | **verbatim** — folding a contiguous range is not a property of any decomposition |
| the tiling obligation | **composed, proof did not transfer** — ~30 new lines |
| the exactness obligation | **provably does not hold**, and that is a result |

**The two kernels split K differently, which is why the proof had to be
redone.** The exact kernel gives the first `rem` bands one extra k; the f32 one
is proportional, `[t·K/n, (t+1)·K/n)`. Both tile `[0, K)`;
`the_two_splits_are_different` exhibits `K = 5, n = 3`, where the proportional
band 0 has one element and the exact one has two. So the *obligation* is the
same object and its *discharge* is not — which is exactly the distinction a
transformation IR would have to mechanise, and the first real datum about
whether it can.

**The exactness half is a result, not a gap.** f32 addition is not associative,
so per-band partials do not sum to the naive sum.
`rounding_breaks_the_proportional_split_too` refutes it at the same `f`, `K`
and thread count where the exact kernel's own refutation lives, with the same
control showing the failure belongs to the accumulate rather than to the
decomposition. The repo asserts bit-identity for the exact kernel and nowhere
for this one; that is now a stated consequence instead of an omission.

##### The finding: a redundant guard that becomes load-bearing exactly when it is needed

Both decompositions clamp their last band — `select (t+1 == n) ext hi` — under a
comment saying the last thread takes the remainder "so no row of B is dropped".
**The clamp never fires**: `(n·ext)/n` is already `ext`, and a granule count
always covers its extent. Measured exhaustively over the reachable domain
(0 of 192,000 K cases, 0 of 48,000 granule cases) and proved as `pedge_last` /
`gedge_last`.

The tempting conclusion is "dead code". **The discriminating experiment says
otherwise**, and it was run rather than reasoned about:

    R1   band edge broken to (ext/n)*t, clamp kept      correctness suites ALL PASS
    R1b  same break, clamp removed                      cpu_gemm_threaded FAILS

So the clamp is redundant *with the correct arithmetic* and is precisely what
turns a wrong band edge into a right answer. It stays, and now with a measured
reason rather than a comment. This is the mirror image of the packers' masks in
`ExactGemmPacking.v`, which are redundant *with each other* so neither is
pinned alone; here the redundancy is with an arithmetic identity, and breaking
the identity makes the guard live.

`every_edge_snaps_to_a_granule_or_the_extent` promotes a second comment to a
theorem: a band boundary inside a tile would make one thread write a partial
tile, and nothing had said it cannot happen.

##### MUTATION TABLE

Each `--test` target run separately from a bash script.

| mutation | schedule gate | band model | correctness |
|---|---|---|---|
| K edge `(ext/n)*t` — drops the remainder | FAIL | FAIL | **all ok** (the clamp saves it) |
| ...same, with the clamp removed | FAIL | FAIL | `cpu_gemm_threaded` FAIL |
| **K edge operands swapped (same value)** | **FAIL** | ok | **all ok** |
| granule edge loses its `min(·, ext)` | FAIL | FAIL | all ok |
| K edge hand-written, identical text | ok | ok | ok |
| proof re-declares `pedge` (not aliasing) | — | — | `coqc` FAIL |
| proof re-declares `pedge` identically | — | — | ok |

Rows 5 and 7 are the design controls, and they are the same control twice: the
gates check the *property* — the proof's arithmetic is the emitter's — not the
*plumbing*. Row 7's re-declaration is safe for the reason already recorded
about `mr: 6` versus `mr: VNNI_MR`: the generated `ExactGemmSchedule.v` is
byte-identity-gated against `cpu_gemm.rs`, so a copy that agrees today fails
`coqc` the moment the schedule moves.

Row 3 is the fifth same-value divergence caught by the schedule gate alone.

##### `Ix::eval`, so the model test is not a third description

The obvious way to write `f32_band_split_model.rs` is to transcribe the
arithmetic into Rust — which is the exact defect this layer exists to remove,
one file over. `Ix` gained an evaluator instead, so the test runs the **same**
expression the emitter renders to LLVM and the generator renders to Coq: one
description, three consumers. The three agree because every operand is
non-negative, where `sdiv`'s truncation and `nat` division's floor coincide;
`eval` asserts that rather than assuming it.

All four emitted modules — including the 4,752-line f32 one — are byte-for-byte
unchanged by the extraction.

##### What this does not settle

One kernel is not a compiler. Both of these are GEMMs, so the axes are the same
even though the decompositions are not; a stencil or an attention kernel would
test something this does not. And nothing here touches the f32 micro-kernel,
its packing, or its scratch reduction — this is the schedule, and only the
schedule.

#### The third kernel, and the first that is not a GEMM · 2026-08-30

The previous entry closed by naming its own limit: *"both kernels are GEMMs, so
the axes coincide even though the decompositions do not."* `src/exact_attention.rs`
is not a GEMM, and its module header makes a claim of exactly the kind this
programme exists to discharge — **"the answer does not depend on `blockDim.x`,
`gridDim.x`, `gridDim.z`, or the order the atomics land."**
`tests/gpu_attention_invariance.rs` demonstrates it on a real card at nine
launch geometries. Nine geometries is not every geometry, and *the order the
atomics land is not a geometry at all*: it is a property of a race.

`proofs/GridStrideSplit.v` (Rocq 9.1, 8 `Print Assumptions`, no axioms) and
`tests/exact_attention_schedule.rs` close both halves.

##### It needed a property neither GEMM proof did

The decomposition is different in two ways, and the second is the interesting one.

- **It is not an interval split.** Worker `w` of `n` takes
  `{ i < S : i mod n = w }` — the residue classes, interleaved. Not
  `ExactGemmKsplit`'s contiguous bands, not `GemmBandSplit`'s proportional
  edges. An index belongs to its class by arithmetic rather than by an
  accumulated offset, so `stride_classes_partition` is a third proof of the
  same obligation.
- **The partials are combined in ARBITRARY order**, by `red.shared.add.u64`
  and `red.global.add.u64`. Both GEMM proofs fold their bands in index order,
  so associativity carried the whole argument. Here the order is whatever the
  hardware chooses, so the argument needs **commutativity** —
  `atomics_may_land_in_any_order` quantifies over every permutation of the
  workers.

That is the first obligation in the programme that is genuinely *new* rather
than a re-discharge, and it is what a transformation IR would have to know to
schedule an atomic reduction at all.

Both halves are shown to be about the accumulate, and the second refutation is
a failure the GEMM kernels **cannot exhibit**:
`rounding_breaks_the_stride_split` disagrees across worker counts, as a GEMM's
K-split does; `rounding_is_order_dependent` disagrees at a *fixed* worker count
purely from the order the partials land — 1000 against 1100 — with
`exact_is_order_independent` as the control. Two fixtures were needed, because
the input that makes the worker count matter and the input that makes the order
matter are not the same one; three plausible single fixtures agreed on both
readings before this pair.

##### What the gate isolates, measured rather than asserted

The tie reads the emitted PTX and derives the dataflow rather than matching
text. The property it pins is the **precondition** the partition theorem needs:
`worker` is a mixed-radix index over `(ctaid.z, ctaid.x, tid.x)`, so `nworkers`
must be the product of exactly those three indices' extents. Drop `%nctaid.z`
and the stride is smaller than the worker count — the classes overlap, some
keys counted twice, others never.

| mutation | schedule gate | device test (GPU) | device test (no GPU) |
|---|---|---|---|
| `nworkers` drops `%nctaid.z` | **FAIL** | FAIL | **ok** |
| instructions reordered, dataflow unchanged | ok | ok | ok |
| the proof's classes become a block split | — | — | `coqc` FAIL |
| the proof's accumulator bound drifts | FAIL | — | — |
| the `[Y SEQUENCE REDUCTION]` marker removed | **FAIL** | ok | ok |

**Row 1 is the result.** With a card present the device test catches it too, so
the gate is a diagnosis by name. Without one — the ordinary CI case — the device
test prints `SKIP: no CUDA driver` and reports **ok on a broken kernel**, and
the gate is the only thing left. Verified by running it under
`CUDA_VISIBLE_DEVICES=""`, not assumed.

Row 2 is why the check derives the dataflow instead of matching a window, and
it is load-bearing rather than decorative: the two accumulating entries build
the *same* decomposition in a *different instruction order* — `attn_accum`
hoists `%ntid.x` and `%tid.x` above the shared-memory zeroing loop and
`attn_accum_naive` does not — so a literal sequence matches at most one of them.

##### Two traps in writing the tie, and both are now removed at the source

Both were mine, both are the shape this repo keeps recording, and both were
first *worked around in the test*. A workaround leaves the hazard in place for
the next reader, so both were then fixed in `src/exact_attention.rs`.

- **`attn_accum` contains TWO grid-stride loops** of identical shape: the one
  over the sequence, and one over `d` that zeroes the shared accumulators.
  Taking the first `add.s32 %i, %i, %s` picks the zeroing loop, whose stride is
  `%ntid.x` alone — which then reports the decomposition as depending on
  `tid.x` and nothing else.
  - *Worked around* by telling the loops apart by their **bound**, so the test
    only worked because the fixture happened to use `head_dim != seq_len` — the
    `lda == K` coincidence, one layer over.
  - *Fixed* by naming the loop in the artifact: `// [Y SEQUENCE REDUCTION]`,
    the device this backend already uses for `[Y PAGED DECODE ATTENTION]` and
    `[Y ZERO DRIFT]`. Both entries carry it, the test locates the reduction by
    it, and nothing depends on the shape chosen any more. The labels could not
    serve — they are `LOOP_I` in one entry and `NLOOP_I` in the other, so
    keying on those is a different hardcode, not a fix.
- **`%r9 = %r9 * %r5` was a real instruction here** — the worker count was
  built in two steps into one register — so a "last definition wins" walk
  resolves its own operand to itself.
  - *Worked around* by resolving operands at the **definition's** position
    rather than the use, which is ordinary reaching definitions and is correct
    in general.
  - *Fixed* by making the computation SSA (`%r20` for the intermediate).
    Nothing needed the reuse; it was a trap for anything reading the kernel
    back, human or tool.

**Fixing the kernel removed the only thing exercising the resolver's
reaching-definition logic, which is precisely when a capability rots.** That
path is pinned on a synthetic body instead
(`the_dependency_walk_handles_a_redefined_register`), and the transfer is
demonstrated rather than assumed: reverting the resolver to "last definition
wins" now fails **only** that test, with all five real-kernel tests passing.
Hand-written PTX elsewhere in this repo is full of non-SSA reuse, so the
capability has to survive its own kernel being cleaned up.

The register change is a pure renaming and was verified where it matters — all
nine launch geometries on the real card, `two-level == naive == reference`.

Two further mutations, on the fixes themselves: dropping `%nctaid.z` is still
caught (and still only by the gate without a card), and **removing the marker
fails the gate loudly** rather than silently selecting the wrong loop, which is
the difference between a fixed hazard and a hidden one.

##### What is NOT claimed

The accumulator ceiling (`the_bound_is_one_unit_wide`) is stated and tied to
`MAX_EXACT_SEQ_LEN`, parsed out of the `.v` rather than restated — it needs
2.7e8 keys to reach, so no device test can ever demonstrate it. The per-thread
body is not modelled: `f` is an arbitrary function of the index, so the integer
exp, the Q0.28 weight and the int8 `V` load are outside this. And the tie is
**weaker than the GEMM kernels'**: those render their schedule from an `Ix`
shared with the proof generator, so a divergence is a byte-identity failure.
This kernel is a PTX string template, and making it an `Ix` means routing
attention through `IrBuilder` — a larger change than this file.

Twelve proofs, no axioms, nothing admitted. 614 / 880 tests, both builds green.

#### The launch contract the other kernel in the same file did not have · 2026-08-30

Continuing the sweep from the schedule work found a live bug, and the tell was
that the two reductions in `src/exact_attention.rs` were **not the same kind of
loop**. `attn_accum` is grid-stride and launch-invariant — that is the module's
headline claim. `attn_scores` was **one thread per key with a bounds guard and
no loop**, so it silently required `gridDim.x * blockDim.x >= S`.

Measured on the card at `S = 512`, before touching anything:

    grid.x 1 block 128 -> 128 threads:  768 of 1024 score slots stale, max 34647
    grid.x 1 block  32 ->  32 threads:  960 of 1024 stale,             max 33794
    grid.x 4 block 128 -> 512 threads:    0 stale,                     max 34752

**The stale scores are the lesser half.** `attn_accum` subtracts the maximum
from every score, so a wrong maximum moves *every* softmax weight — a silently
wrong answer for the whole batch row, not a partially-filled buffer.

Two kernels in one file with opposite launch contracts, under a header that
advertises launch invariance, and nothing stating either. The same class as the
paged-decode kernel's fixed launch contract that this file already records, and
worse in one respect: there the contract is at least written down.

**Removed rather than documented.** `attn_scores` is a grid-stride loop now —
the same residue-class partition `proofs/GridStrideSplit.v` already proves, so
the theorem covers it with no new proof and the contract across the family is
uniform. `&Q[b][0]` moved inside the loop because the inner dot-product walks
that pointer.

##### What caught it, and what could not

| | before the fix |
|---|---|
| `gpu_attention_invariance` (new test) | **FAIL** |
| `exact_attention_schedule` | **FAIL** |
| `exact_attention_bounds` | ok |
| `ptx_portability` (assembles at every arch) | ok |

The portability gate runs real `ptxas` at five architectures and is untroubled,
which is this repo's own recorded limit — *an assemble gate cannot see a missing
instruction*, and here the missing instruction is a whole loop.

The existing invariance test could not see it either, and the reason is worth
keeping: it sweeps the geometry of `attn_accum` while launching `attn_scores`
at **one fixed, correct** geometry. **A test that sweeps one kernel's launch
geometry is not testing the other kernel's.** The new test poisons `Scores`
with `0xAB` rather than zeroing it — zeroing would hide a skipped key behind a
plausible value — and asserts every short geometry agrees with a covering one.

##### The gate had to stop hardcoding a rule that held for two of three kernels

Adding `attn_scores` to the schedule gate broke it, correctly. The check
asserted that `nworkers` is the product of **three** extents; `attn_scores`
mixes only two hardware dimensions. The rule is now *derived from each kernel*:
take the hardware indices `worker` actually depends on, map each to its extent
(`%ctaid.x` → `%nctaid.x`), and require `nworkers` to be the product of exactly
those — no fewer, or the classes overlap; no more, or they stop covering.

**And a structural pattern matched more than one thing for the third time in
this file.** "The stride update is `add.s32 %X, %X, %rY` after the marker" also
matches `add.s32 %r12, %r12, %r4` — an ordinary `b * S + i` flat-index
computation — so the counter came out as address arithmetic. It anchors on the
loop's **back-edge** now: the increment is the last such instruction before the
branch that closes the loop, which is a property of loops rather than a shape
that happens to be unique today. First the marker, now the back-edge: both
times the fix was to anchor on something the kernel actually *says*.

##### The last precondition is removed rather than documented, and the test that
##### should have covered it was annihilated by a shared constant

`red.global.max` needs no order argument — max is associative, commutative and
idempotent, so `GridStrideSplit.v` covers the order. Its **identity** is a
different question, and it used to come from the host: a signed max wants
`i32::MIN`, which is not a uniform byte pattern, so `M` could not take the
`memset(0)` that L, O and P all take. A caller who used one anyway got a wrong
answer **only when a row's scores were all negative** — with `max(0, s) = 0` the
softmax subtracts a maximum no key attains, and in Q0.28 the weights quantise
toward zero instead of cancelling as they would in exact arithmetic.

**That is the worst shape a precondition can have: it cannot fire on random
int8 data**, where the max over hundreds of keys is positive with overwhelming
probability. Every test in the file passed either way.

`x ^ 0x80000000` is the order-preserving signed→unsigned bijection, so a max
over the biased values is the max over the originals and **unsigned max has
identity 0** — a zeroed buffer now *means* `i32::MIN`, and the module's contract
is uniform: everything it writes starts at zero. The two accumulating entries
undo the bias when they load `M[b]`.

**The interesting half is the mutation that survived.** Deleting the undo from
*one* of the two accumulating entries passed everything, including the
accum-vs-naive differential on `p` — which is precisely the comparison that
should catch a one-sided change. The cause is arithmetic and worth stating: the
exp argument is `((m - s) * KFix + 2^15) >> 16` computed in 64 bits and then
**truncated to u32**, so a missing undo shifts it by `2^15 * KFix`, and every
test passed `KFix = 8 << 16 = 2^19`, for which that shift is exactly `2^34` —
**zero modulo 2^32**. The truncation annihilated the bug at the one temperature
every arm of the differential shared. `feedback-differential-arms-share-constants`,
in a place nobody thought to look: not a scale factor both sides multiply by,
but a constant that makes a *wrong* value truncate to the right one.

A real model's softmax scale is `q_scale * k_scale / sqrt(d)` and does not
oblige. `the_accumulating_entries_undo_the_bias` uses `2^19 + 1` — the same
temperature to within 2e-6, odd, so the shift is `2^15` and a missing undo
drives almost every weight to zero — and checks `p` against
`fixed_exp::exp2_neg_q16_16` on the host over the device's own scores. That is
the oracle `p` never had: the existing test says `p` is "a per-element function
of `s`, so it is not what is in question here", true of the *reduction* claim,
and the undo lives exactly there. **When a docstring explains why something is
out of scope, check what has since moved into it.**

Six mutations, all caught, `gpu_attention_invariance` the only suite that fires
for any of them: full revert to a signed max; drop the bias in `attn_scores`
only; drop the undo in each accumulating entry separately (the survivor, now
caught); bias with `0x40000000`, which is not order-preserving; and keep the
bias but reduce with a *signed* max, which puts the identity back where it was.

##### And the sweep for the same class found the saturate

The lesson generalises to "what other constant here can make a wrong value land
back on a right one", and the answer was one line down:

```text
shr.s64      %rd50, %rd50, 16;
min.s64      %rd50, %rd50, 1073741824;   // <- this
cvt.u32.u64  %r16,  %rd50;
```

Without the `min`, a large score delta wraps modulo `2^32` at the narrowing and
a **far** key comes back with a **small** argument — a weight near `2^28`, the
weight of the best key. Not precision loss: attention paid to the wrong token.

**Deleting it from the emitter passes `gpu_attention_invariance` on a real
card.** At `head_dim = 32` and `C = 2^-13` the argument never approaches `2^32`,
so no device fixture can reach it. The parameters that *do* reach it are ones
this repo already sweeps — `C = 3.0e-2` is the largest scale in
`the_temperature_multiplier_carries_two_to_the_thirty_two`'s own list, and
`head_dim = 128` is a shape `the_shapes_a_model_uses_still_generate` generates.
At those, a key `2,184,534` below the maximum comes back at `0.986 × 2^28`.

What was missing was the **necessity**, not the presence. The literal was pinned
by a substring assertion, so removing or moving it already failed — but a check
that a line exists says nothing about what happens without it, and the host
replica in the temperature test is `((ds * kfix + 32768) >> 16) as f64`, with no
clamp and **no narrowing**. Both conventions it exists to separate live above
the point where the guard acts, so it is right about the multiplier and silent
about everything else. *Nothing anywhere modelled the u32.*

`the_saturate_is_what_stops_a_far_key_wrapping_into_a_large_weight` models the
kernel's arithmetic once with the guard as a **parameter**, so the two readings
cannot drift; *searches* for the wrapping delta rather than hardcoding it;
carries the control that stops `min(t, 0)` passing (in range the guard must be a
no-op); shows the value is the exp's own saturation point; and ties the model to
the emitter. Four mutations, all caught — remove the guard, clamp at `2^31`,
clamp after the narrowing, and drop the clamp from *both* arms of the model,
which is the vacuous-necessity case.

#### Phase 2's research question, answered from what was already here

Phase 2's stated risk is *"if obligations don't compose, the thing is a one-off
proof rather than a compiler."* Three files were proving the same theorem three
times — `ExactGemmKsplit` (contiguous uneven bands), `GemmBandSplit`
(proportional edges, plus a granule-snapped family), `GridStrideSplit`
(interleaved residue classes) — each with its own edge facts, prefix lemma,
tiling theorem, rounding refutation and control. That question is answerable
with no new kernel, no IR and no hardware.

`proofs/Decomposition.v` is the schema. Every existing theorem keeps its exact
statement; its *proof* becomes an instantiation.

**It is TWO theorems, and which one a kernel gets is decided by whether its
parts are contiguous.** `contiguous_exact` takes an edge function with three
properties and spends **associativity only** — consecutive parts are a
re-bracketing, not a reordering. `decomposition_exact` takes an arbitrary
`owner` map, so the terms genuinely change order, and needs **commutativity**.
Neither is derived from the other: the general form would give the contiguous
one only by inverting the edge function, and would then demand commutativity
the contiguous case does not need — a *weaker* result presented as a simpler
one. `the_interleaved_split_is_not_contiguous` stops "just use the general one
everywhere" from looking free. That distinction was prose scattered across two
files and is now two theorems.

**The honest headline is that it does not save lines — the total went up.**
Across the *five* files now instantiating it:

| file | before | after | |
|---|---|---|---|
| `ExactGemmKsplit` | 30 | 32 | gained a bridge |
| `GemmBandSplit` | 47 | 52 | gained a bridge **and a theorem that did not exist** |
| `GridStrideSplit` | 36 | 36 | two bridges in, three inductions out |
| `ExactGemmMicro` | 71 | **68** | the clamped-edge prefix was the most expensive of the five |
| kernels | **184** | **188** | plus 42 in `Decomposition.v` |

Each kernel gains a *bridge* saying its own fold IS the schema's fold — the
price of leaving every statement unchanged — and the burden was already thin
because the informal reuse was already happening: `GemmBandSplit` reused
`acc_range` and `sum_range_split` by name and copied the *shape* of the rest.

**(An earlier version of this section published 255 → 269 plus 76. Those were
wrong: my line counter treated `Proof. … Qed.` written on one line as an
unterminated proof and counted the rest of the file. Instantiating the schema
introduces exactly that idiom, so the bug was created by the change it was
measuring. The conclusion — the total goes up — survives; the numbers did not.)**

What it buys instead:

- **A theorem that did not exist.** The f32 kernel's M/N granule split had four
  theorems about its *edges* and no exactness theorem, because writing the
  reduction out a third time was not worth it for a family used to cut rows and
  columns. Under the schema it costs one application and the three edge facts
  already proved. Measured against a straw-man written the old way, in the same
  file, both compiling and proving the same thing: **10 tactic lines standalone,
  6 through the schema** — so the marginal cost of a *new* decomposition is a
  bridge plus three facts, with **no new induction and no new reasoning about
  ranges**.
- **The reuse becomes a checked dependency instead of a convention.** Same move
  as `ExactGemmSchedule.v` made for constants, with the same honest caveat that
  file's header carries: nothing had actually drifted. Six mutations, all caught
  by `proofs_are_checked` — drop either theorem's key hypothesis, declare the
  interleaved split contiguous, make `part` ignore its owner, instantiate the
  granule family with the *proportional* edge function.
- **A fifth instantiation whose purpose is not dividing work.** The flush
  chunking in `ExactGemmMicro` cuts the k-pair range into intervals of `Fl`
  because `vpdpwssd` accumulates in int32 and must be widened before it wraps —
  an *overflow budget*, not a work split, and this file's own notes say so while
  warning against confusing the two. The schema does not care: a decomposition
  of a range is a decomposition of a range, whatever it was chosen for. Its
  edges are **clamped** (`t ↦ min(t·Fl, n)`), which is why its bridge is a case
  split rather than `reflexivity` — `cw` clamps only its right end, so a chunk
  entirely past `n` has width `min(coff (S t), n) − coff t`, truncating to 0 in
  `nat`, where the schema's is `n − n`. Both zero; saying so is the work. It is
  the only one of the five that got **smaller**.
- **The finding: two obligations turned out to be one decomposition.** The
  int32 flush interval is an *overflow budget*; the output tiling is a *memory
  partition*; this document's own notes warn against confusing them. They are
  right about the purpose and wrong about the shape — **both are
  `t ↦ min(t·X, ext)`**, and each proof file developed the three-regime case
  split from scratch. The emitter hides it by computing them differently:
  `chunk_end_ix` emits `min(iv + T, ext)` — an END — and `tile_width_ix` emits
  `min(ext - iv, T)` — a WIDTH. Two instruction sequences, one function.
  `Decomposition.clamped` is the family; `clamped_width` reconciles the two
  spellings once; and
  `the_flush_interval_and_the_output_tile_are_the_same_family` states the
  agreement in the file whose job is cross-file agreement. That is a claim
  about the *schedule*, not the code generator — the emitter still emits two
  sequences, and should, since one site has the offset in hand and the other
  has the end.
- **A second obligation family now shares the schema.** `acc_parts` folds
  *values* over `Z`; an output tiling has nothing to fold and asks only that
  the part *widths* add up with no gap and no double count. `width_sum` over
  `nat` is that, and it needs no algebra at all — the sum telescopes. Same
  decomposition, different consequence. `ExactGemmTiling`'s 16-line
  three-regime `covered_closed` is **3 lines**, and the file is 6 smaller.
  `ExactGemmMicro` went 3 *bigger*, because `cw` clamps only its right end and
  so needs a width reconciliation *and* an offset one.
- **The standing limit, stated rather than gated:** nothing *forces* a kernel to
  instantiate the schema. The straw-man above compiles perfectly. What is
  guaranteed is that the seven files which do instantiate it cannot drift from
  each other — a change to `contiguous_exact` or `clamped_width` breaks all of
  them at once.
- **One mutation of five survived, in a control written minutes earlier.** The
  non-vacuity companion to the agreement theorem evaluated both spellings at a
  ragged piece — and moving it to a *full* piece, where the agreement is
  trivial, left everything green. It states the raggedness as a proposition now
  (`tw 5 3 1 < 3`), which the move makes false. **A control that merely
  exhibits the interesting case has not said the case is interesting.**

#### The second schema: positional indices

`quot_rem_unique` had **two independent copies** — `ExactGemmTiling` and
`ExactGemmPacking`, neither file requiring the other. `ExactGemmPacking`'s
carried a comment defending it:

> Proved locally rather than imported: it is six lines, and a lemma cannot
> drift into being wrong the way a duplicated CONSTANT can — it is re-proved
> wherever it is stated.

**That argument is right, and it is right about the wrong thing.** A duplicated
*lemma* is checked by `coqc` at every copy, unlike a duplicated constant, which
is exactly why `ExactGemmSchedule.v` is generated. What it does not cover is the
package around it: uniqueness is six lines, but **onto** was eleven lines in
`ExactGemmTiling`, seven in `ExactGemmRegisterTile` and four in
`ExactGemmPacking` — the same argument three times — and the two-digit peel in
`pack_b_slot_bijective` was sixteen. *The thing that recurs is the bijection,
not the lemma inside it.*

`proofs/MixedRadix.v` is `pack B q r = q·B + r` with its two legs, its
two-sided inverse, in-range, onto, and a two-digit form. Every index in this
compiler is one of these at a different radix: a tile offset `t·T + f`, a packed
A slot `2i + h`, an accumulator column `16v + l`, a linear address `r·ldc + c`,
and the packed B slot `(j/16)·32 + (j mod 16)·2 + h` — three digits, radix 4,
16, 2.

**This is the first factoring where the kernel files got smaller in aggregate**:
222 → 200 tactic lines, plus 30 in the shared file.

| | before | after |
|---|---|---|
| `Tiling.quot_rem_unique` | 7 | **1** |
| `Packing.quot_rem_unique` | 7 | **1** |
| `Tiling.tile_index_surjective` | 11 | 6 |
| `Packing.pack_b_slot_bijective` | 16 | 13 |
| `RegisterTile.tile_position_surjective` | 7 | 5 |

**Six mutations, four caught, two survivors of different kinds** — and sorting
them is the point. Restoring `ExactGemmPacking`'s *identical* hand-written copy
passes, and should: the gate checks the property, not the plumbing, exactly as
the `Ix` gate's design control does. The other survivor was a **real hole in a
control written minutes earlier**, and it is the same shape as the previous
increment's: `without_the_digit_bound_it_is_not_injective` exhibited a collision
(`0·2 + 2 = 1·2 + 0`), and moving the fixture to a *legal* digit pair left the
statement true, provable and green. It refutes the **weakened theorem** now —
`~ (forall …, q₁·B + r₁ = q₂·B + r₂ → q₁ = q₂ ∧ r₁ = r₂)` — which no choice of
witness can satisfy vacuously. **Twice in two increments: a control has to state
what makes the case interesting as a proposition, not merely exhibit it.**

#### Phase 2's first slice: a loop's ITERATION SPACE, not just an expression

The `Ix` layer removed the second description of a schedule *number*. The loop
nest's *shape* was still hand-written — so `ExactGemmTiling.toff` and `ntiles`
were a **model** of what the driver's row-panel loop does, with nothing saying
the loop does it.

`IrBuilder::loop_begin(tag, start, end, step)` was already a counted loop, so
the skeleton was factored; what was not was the tie to the proofs.
`cpu_gemm::CountedLoop` names those three as `Ix` values and renders them **to
LLVM and to Coq**, exactly as the expression layer does. All three emitted
modules are byte-for-byte unchanged.

```coq
Definition row_panel_visit (M k : nat) : nat := (0 + (k * 6)).
Definition row_panel_trips (M : nat) : nat := (((M - 0) + (6 - 1)) / 6).
```

and in `ExactGemmTiling`, the payoff — the first tie in this development
between a proof and the **shape** of an emitted loop rather than the value of an
emitted expression:

```coq
the_emitted_row_loop_enumerates_the_tiles  : row_panel_visit M k = toff MR k
the_emitted_row_loop_runs_once_per_tile    : row_panel_trips M   = ntiles M MR
```

**`trips_ix` is rendered to Coq only, and that asymmetry is deliberate.** The
emitter never computes a trip count — the loop tests `iv < end` — so it is a
fact *about* the loop rather than an expression it emits. Which is why
**mutating it to round down is caught by the schedule gate and by nothing
else**: every correctness suite passes, because the emitted code is unchanged
and only the proof's account of it moved. That is the divergence class this
whole layer exists for.

**My own gate had a gap, and the mutation sweep found it.** Handing the
*column* loop the row description reaches Coq unchanged — the `.v` still renders
both loops correctly — so the first version of the gate passed while the driver
opened a 6-step loop over `%M` where the description says 64 over `%N`. Three
correctness suites caught it; the schedule did not, which is the weaker
guarantee. `the_driver_opens_the_loop_it_was_described_with` now recovers
`(start, end, step)` from the emitted LLVM per tag by following the loop's own
dataflow — the store that seeds the induction variable, the `icmp slt` in its
`.cond` block, and the `add` written back to the same slot — and compares them
against the description. **A description that reaches the proof is only half a
tie; something has to check the emitter uses it at the right site.**

| mutation | schedule gate | 3 correctness suites |
|---|---|---|
| row loop steps by `MR-1` | caught | caught |
| **column loop given the row description** | **caught** *(was missed)* | caught |
| row loop starts at 1 | caught | caught |
| `trips_ix` rounds down | **caught** | **all ok** |
| A-pack loop hand-written *identically* | ok | ok — the design control |

**What is still hand-written:** which loops exist, in what order, in which
blocks, and what they call. And only loops whose bounds are *operands* can be
described this way — the fold-back loops walk builder-computed registers, and
describing those means emitting their bounds through `Ix`, which changes the
instruction stream. Both stated in the code rather than left to be rediscovered.

619 / 885 tests, both builds green. **Fourteen proofs, no axioms.**

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

**Inventory correction, measured 2026-08-31.** "XOR bank swizzling" reads as a
transformation that exists and needs a proof. It does not exist in any path a
kernel can take. `src/bank_conflict.rs` is real — the type checker searches for
a conflict-free swizzle when a program declares an `SmemLayout` and prints
`[Optimization] Auto-swizzling SharedMemoryTile RxC ...` — and **that result
reaches no backend**, because every way of getting an indexable tile is either
a syntax error, refused, or (until it was fixed) emitted PTX `ptxas` rejects.
`docs/y_language_documentation.md` §21 documents the type as an API; the
section now opens with a status note saying otherwise.

The shared-memory surface that ships is `shared_alloc_u32` / `shared_load_v4` /
`shared_store_v4` / `barrier_sync`, and **it has no swizzle at all** — the
BN254 kernels apply one in generated source
(`tools/gen_bn254_kernels.py::swizzle`), where it is a bijection because bits 3
and up pass through unchanged, argued in a comment and checked by nothing.

So Phase 3's swizzle obligation is not "prove the existing pass correct". It is
"there is a hand-written swizzle in a generator whose bijectivity is prose, and
a compiler pass that computes swizzles nothing applies". Naming which of those
to build on is the first decision, and it was not visible before this.

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

**FIRST THEOREM LANDED, 2026-09-01.** `proofs/SoftmaxErrorBound.v` bounds the
exact-attention kernel's fixed-point softmax against the ideal one — `7/100`
on a `+-127` output range at 65,536 keys, with the argument reduction's
*multiplicative* error and the exp table's *additive* ulp composed, and the max
subtraction carried through as the floor that makes the second negligible. See
the dated entry at the foot of this document. The generalisable result is that
**the `Decomposition` schema needed no approximate variant, because the error
is introduced BEFORE the decomposition rather than by it** — which is exactly
the property that does *not* hold for an f32 softmax, and therefore the shape
of the addressable set: pipelines whose approximation is per-element.

**Extended the same day**: the softmax's *temperature* is itself quantized
(`KFix = round(C * 2^32)`), which the first pass named as unpriced. Priced now,
and it is a head_dim question rather than a `KFix` one — under `1/2048` on any
weight at head_dim 128. Two multipliers are refused rather than bounded, and
both preconditions had lived only as a bare `continue` in a Python script.

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

#### The certificate is EMITTED now, and that is what the programme is named after · 2026-08-30

`proofs/` is checked once, at build time, against the shipped schedule
constants. Its theorems are universally quantified over `M`, `N`, `K` and
`nthr`, so every shape is covered — and a user who compiled their own
`@ZeroDrift` nest still got a fast kernel and **no artifact**. "Proof-carrying"
described the repository, not the output. That was the largest gap between what
exists and what the programme is called, and it was mostly plumbing, because
the generator already existed.

`src/exact_gemm_certificate.rs` renders a `.v` beside the `.ll` whenever the
exact `vpdpwssd` kernel is substituted. It **instantiates**
`ExactGemmWhole.the_threaded_gemm_holds_the_source_dot_products` at this
compilation's flush interval and operand bound rather than re-proving anything:

```coq
Definition Fl : nat := 64.
Definition m  : Z   := 1024.

Theorem the_licence_holds :
  2 * Z.of_nat Fl * m * m <= ExactGemmMicro.I32MAX.

Theorem this_kernel_computes_the_source_dot_products :
  forall A B M N K nthr r c,
    (forall i k, Z.abs (A i k) <= m) -> (forall k j, Z.abs (B k j) <= m) ->
    (0 < nthr)%nat -> (r < M)%nat -> (c < N)%nat ->
    W.thread_sum A B M N K Fl nthr r c nthr = PK.sum_k (fun k => A r k * B k c) K.
```

`--emit-llvm` writes it, names it in the report, and prints the command that
checks it. `Y_NO_CERTIFICATE=1` suppresses it, loudly.

##### Why this is not paperwork: two derivations of one obligation, by two tools

The only hypothesis that depends on the program is the LICENCE. Y decides it in
**floating point** — a `sqrt` and a `floor` on `f64` in
`VnniExact::max_operand_magnitude`. The certificate states it over `Z` and
hands it to `coqc`, which has no floats. **So emitting the certificate makes the
compiler's own floating-point reasoning checkable by an independent integer
tool.**

Verified at the edge before anything was wired in: at the default interval the
certificate is **accepted at `m = 4095` and refused at `m = 4096`** — the same
one-unit boundary `tests/exact_gemm_licence_obligations.rs` finds by exhausting
the int16 domain, reproduced by a different tool from a different derivation.
`the_certificate_refuses_exactly_the_bounds_the_compiler_refuses` runs that
comparison at four intervals, at the limit and one above it, and asserts the two
answers agree.

The rounding direction is the other place the two could diverge, and the
argument is short enough to write down. `m` plays two roles at once — the
licence gets **harder** as `m` grows, the data hypothesis `|A[i,k]| <= m` gets
**easier** — so `ceil` is conservative for both, and `floor` would be wrong
twice. It cannot break a licence Y granted: Y licenses exactly when `m <= L` for
an integer `L`, and `m <= L` gives `ceil(m) <= L`. Checked over every interval
the scheme admits rather than argued.

##### TWO SURVIVORS, AND BOTH ARE THE SAME LESSON ABOUT PROOF ARTIFACTS

Eight mutations. Six caught immediately; two survived a fresh, all-green test
file, and they have a common cause worth more than the fixes.

| mutation | outcome |
|---|---|
| `integer_bound` floors instead of ceiling | caught (unit + the fractional-bound fixture) |
| renderer hardcodes `m = 1024` | caught |
| renderer hardcodes `Fl = 64` | caught (the licence sweep renders at four intervals) |
| **licence statement drops one factor of `m`** | **SURVIVED** |
| certificate never recorded at the substitution site | caught (4 tests) |
| `Y_NO_CERTIFICATE` ignored | caught |
| file name not sanitised to a Coq identifier | caught |
| **non-vacuity theorem replaced by `True`** | **SURVIVED** |

**Coq compares propositions up to CONVERSION.** At a licensed numeral,
`2*Fl*m <= I32MAX` and `2*Fl*m*m <= I32MAX` both reduce to `Lt <> Gt`, so a
certificate stating the wrong obligation type-checks, and its use site — which
still demands the real hypothesis — accepts it by conversion as well. The
mutated certificate therefore refuses **exactly** the magnitudes the correct one
refuses, and no `coqc` run can tell them apart. What it changes is the claim a
human auditing the artifact reads, which for a certificate is the entire point
of the artifact.

`Theorem the_certificate_is_not_vacuous : True.` is the same shape: it compiles,
it reports `Closed under the global context`, and it passes a count of how many
such reports appeared.

So **`coqc` accepting a generated proof is necessary and not sufficient**, and
the guard has to be on the statement's TEXT. That is
`every_proof_has_a_content_control` in `tests/proofs_are_checked.rs` — which
guards the *committed* proofs against precisely this and had no counterpart for
the *generated* one. Both survivors are closed by asserting the certificate
states the obligation it is about and evaluates the model it claims to.

The sanitiser row is small and worth stating because it is the one place the
artifact could be self-inconsistent: `coqc` derives the module's logical name
from the FILE name, so sanitising inside the renderer alone would leave a
certificate whose own "check with" line names a module that does not exist.
`4-bit gemm.ysu` is an ordinary source name and an illegal Coq identifier.

##### The controls, because "always write a certificate" would pass the rest

A nest with no operand `@bounds` is still **exact** — scalar lowering honours
`@ZeroDrift` — it is simply not licensed for the fast kernel, and then there is
nothing to certify: the emitted code *is* the naive nest. No certificate is
written for it, nor for `Y_NO_GEMM_RECOGNISER=1`, and both are asserted.

##### What the certificate does not claim

Exactly what the library files exclude, repeated in the artifact's own header
because a certificate that overstates its scope is worse than none: `vpdpwssd`'s
semantics and the little-endian i32 half-order are **definitions** pinned on
hardware by `tests/cpu_gemm_vnni_micro.rs`; the loop nest is partly extracted
and partly modelled; and nothing here is a statement about LLVM, `clang`, or the
machine code either produces.


#### The other kind of loop: a bound that is not an operand · 2026-08-31

The previous slice extracted three counted loops and named its own limit:
*only loops whose bounds are OPERANDS can be described this way; the fold-back
loops walk builder-computed registers.* That limit is now removed, and the loops
it excluded turn out to be the ones where the bound carries the obligation.

The micro-kernel writes a full `MR x NR` tile into scratch whatever the extents
are. The fold-back copies only the **live rectangle** into C — so its trip count
is the clamped tile width, and a fold-back over `MR` is an out-of-bounds
**write**. That is what `ExactGemmTiling.unclamped_tail_writes_out_of_bounds`
refutes about the model, and it is the same class as the three `ldc` sites in
`emit_vnni_threaded_module` that writing that file found in the first place.

`IrBuilder::loop_begin_over` takes the register the driver has **already**
computed plus the `Ix` that produced it. It deliberately does not re-emit the
expression: the driver computes the tile width once and uses it for several
things, and emitting it again at the loop would add instructions the compiler
does not have. **All four emitted modules are byte-for-byte unchanged**, which
is the evidence the description is faithful.

```coq
Definition fold_row_trips (ext iv T : nat) : nat := (((Nat.min (ext - iv) T - 0) + (1 - 1)) / 1).

Theorem the_emitted_fold_back_runs_the_tile_width : forall ext T t,
  SCH.fold_row_trips ext (toff T t) T = tw ext T t.

Theorem the_fold_back_stays_inside_the_live_rectangle : forall ext T t k,
  (k < SCH.fold_row_trips ext (toff T t) T)%nat ->
  (toff T t + SCH.fold_row_visit ext (toff T t) T k < ext)%nat.
```

The second is the join — `tile_index_in_range`, which was a property of the
model only, now applied to the loop that actually performs the copy.

##### Naming the register is not enough; the gate checks it was PRODUCED by the expression

A loop opening on *some* register says nothing. The gate recovers the bound
register from the emitted LLVM, finds its **reaching definition** — searching
backwards from the loop's own `cond` block, because register names are
per-function and this module has three — and requires the instructions defining
it to be the rendered `tile_width_ix`. Taking the first textual match instead
compared the driver's fold-back against `pack_a`'s address arithmetic, which is
how that was found.

##### MUTATION TABLE — and F1 is the row that matters

| mutation | schedule gate | `proofs_are_checked` | `tiling_model` | `thread_invariance` | `exact_threaded` |
|---|---|---|---|---|---|
| **F1 fold-back bound unclamped (`MR`)** | **FAIL** | ok | **ok** | **ok** | FAIL |
| F2 row fold-back given the *column* description | **FAIL** | ok | ok | ok | ok |
| F3 description claims the bound is `M` | **FAIL** | ok | ok | ok | ok |
| F5 same-value `min` swap, `.v` regenerated | **FAIL** | FAIL | ok | ok | ok |

**F1 is a live out-of-bounds write, and two of the four correctness suites do
not observe it** — including `exact_gemm_thread_invariance`, which runs
`M = 53` against the 6-row tile and is the most ragged shape in the repo. The
overrun lands in memory the process owns and the answer stays correct. That is
the same "invisible out-of-bounds" signature this document already records for
the `ldc` sites, and it is the strongest single argument for a gate that reads
the loop rather than the answer.

F2 and F3 are caught by the schedule gate **alone** — the emitted instructions
are unchanged in both, so there is nothing for a correctness suite to see. F5 is
the same-value divergence this layer keeps producing: identical value, different
instructions, and this time it also breaks `coqc`, because `lia` does not know
`Nat.min` commutes.

##### What is still hand-written

Which loops exist, in what order, in which blocks, and what they call. Every
loop in the exact-GEMM driver is now described; the *nest* is not.


#### The scratch tile starts at zero, and nothing had said so · 2026-08-31

`ExactGemmWhole.gemm_position` models the driver as accumulating into
`fun _ _ => 0` — a scratch tile that starts at zero — because the emitted
micro-kernel **accumulates** rather than assigns. That is a hypothesis about the
emitter hidden inside a definition, and nothing connected it to anything the
compiler does. It is exactly the shape of the `N <= ldc` hypothesis whose
absence turned out to be a live heap overflow: a precondition the theorem needs,
stated nowhere.

It is not decorative. `%Ctile` is allocated once and reused for every `(i, j)`
tile, so a zeroing loop one trip short carries the **previous tile's**
accumulator into this one.

`vg.z` is now a described `CountedLoop`, and the tie is an **equality**:

```coq
Theorem the_zeroing_loop_covers_exactly_the_tile :
  SCH.zero_tile_trips = (RT.MR * RT.NR)%nat.

Theorem the_scratch_is_zeroed_wherever_the_fold_back_reads :
  forall fi fj, (fi < RT.MR)%nat -> (fj < RT.NR)%nat ->
    (fi * RT.NR + fj < SCH.zero_tile_trips)%nat.
```

The inequality alone would be satisfied by any bound at least as large, and "at
least as large" is unsafe in **both** directions here — one trip short leaves a
stale slot, one trip long writes past a buffer sized `MR * NR`. The mutation
table is what established that the equality is doing work rather than tidying.

##### MUTATION TABLE — Z2 is the row that matters

| mutation | schedule gate | `proofs_are_checked` | `packing_schedule` | `thread_inv` | `exact_thr` |
|---|---|---|---|---|---|
| Z1 zeroing one trip **short** (description + emitter) | ok | **FAIL** | ok | FAIL | FAIL |
| **Z2 zeroing one trip long** (description + emitter) | ok | **FAIL** | **ok** | **ok** | **ok** |
| Z3 emitter diverges from the description | **FAIL** | ok | ok | FAIL | FAIL |
| N1 A packed inside the column loop | **FAIL** | ok | FAIL | ok | ok |

**Z2 is a write past the end of the scratch buffer that no correctness suite
observes, caught only by the Coq equality.** That is the second invisible
out-of-bounds write this week, and it is the case that justifies stating the
theorem as `=` rather than `<=`.

Z1 and Z2 also show the standing limit of "one description, two consumers"
honestly: when the description itself moves, both consumers move with it and the
byte-identity gate is silent. **What catches a coherent-but-wrong description is
a theorem tying it to something else** — here, the tile geometry `MR * NR` the
register-tile file independently fixes.

##### The nest, as a gate rather than an extraction

`the_driver_nests_its_loops_the_way_it_claims_to` reads the emitted LLVM and
checks each described loop lies inside the loop it is supposed to: `vg.i` inside
`vg.j`, `vg.z` and `vg.fi` inside `vg.i`, `vg.fj` inside `vg.fi`, and the
A-packing sweep inside **nothing** — packing A once for the whole matrix rather
than once per column panel is what makes packing asymptotically free, and
nesting it would be correct and `N/NR` times slower.

It is a gate and not an extraction, deliberately. The nest's shape lives in the
imperative order of `loop_begin`/`loop_end` calls, and there is no second
description to remove; writing one down in Rust and asserting the emitter agrees
would *create* the duplication this layer exists to delete. So it checks the
property.

**It diagnoses rather than isolates, and the table says so:** N1 is caught by
`exact_gemm_packing_schedule` as well, which excises the callees and records the
call order. What the nest gate adds is a name for the failure instead of a call
sequence that is off by a factor of `N/NR`.


#### The fourth kernel, and the first whose parts do not FOLD · 2026-08-31

`Decomposition.v` states its own limit and this document repeats it: five
files instantiate the schema and every one of them is a **reduction**. The
parts fold into a value, the theorem is "the partials combine to the naive
fold", and the two schema theorems differ only in whether that fold spends
associativity or commutativity as well. Two of the kernels are GEMMs and the
third is a softmax, so the axes coincide. Phase 2's stated risk — *if
obligations don't compose, the thing is a one-off proof rather than a
compiler* — is only half answered by five members of one family.

The test did not need a new kernel either. **The MSM binning is a parallel
counting sort**, and its parts do not fold: they TILE a destination array.
Per-chunk histograms, an exclusive prefix over buckets, and an unsynchronised
scatter through a private cursor per (writer, bucket). Real, shipped,
performance-critical code — `bin` is still the largest host phase of the GPU
MSM — and pure host arithmetic, so unlike `GridStrideSplit` the tie needs no
GPU and never skips.

`proofs/CountingSort.v`, 24 `Print Assumptions`, no axioms, nothing admitted.
15 proofs; 638 default / 904 zk tests, both builds green. No `src/` change, so
the emitted modules are byte-identical by construction.

##### The result is POSITIVE and the honest form of it is narrow

Three things could have gone wrong and none did:

- **The edges are DATA-DEPENDENT.** Every edge function instantiated before
  this — `blen`/`boff`, the proportional and granule-snapped families, residue
  classes, `Decomposition.clamped` — is a static function of the extents.
  These come out of a histogram of the input. The schema's `edge : nat -> nat`
  turns out to be general enough, and `the_bucket_edges_are_a_decomposition_
  edge_function` hands its three hypotheses over unchanged.
- **The direction is inverted.** Every decomposition so far supplies an `edge`
  and derives its widths; a counting sort *measures* its widths and derives
  the edges. That is one four-line `Fixpoint` and a bridge
  (`widths_of_an_edge_recover_it`) through `Decomposition.width_sum_closed`.
- **`the_reduction_theorem_still_applies`.** The same edges that place the
  entries reproduce a naive fold over them. Nothing in the MSM uses this — a
  counting sort has no values to add — and it is in the file because the
  alternative claim, *a placement decomposition is a different family*, would
  be wrong.

**What the schema did not have is the CONSEQUENCE.**
`Decomposition.widths_cover_the_extent` says the part widths add up, with no
gap and no double count *in aggregate*. That is strictly weaker than
exactly-once placement: a decomposition that writes one slot twice and another
never has exactly the right total width. What is needed is a bijection —
`slot_injective`, `slot_onto`, `slot_in_range` — and it is not derivable from
the fold theorems. About 90 lines.

**Composing two levels then costs one hypothesis.** `Idx` is cut by bucket and
each bucket's slice is cut again by scatter group; `dest_injective` is two
applications of the one-level bijection and nothing else. The single
hypothesis is `groups_exhaust_the_buckets`, and that is not an assumption
invented for the proof — see below.

##### `MixedRadix` does not cover it, and that is a property of the decomposition

The obvious reading of a two-level index is `MixedRadix.pack B q r`. It does
not apply: a positional index needs one radix per digit, and here the inner
extent is the bucket's own width, which differs between buckets *because it
counts data*. Refuted over a bucket set of widths 1 and 2
(`no_uniform_radix_describes_this`) rather than asserted.

##### The finding: `scatter`'s three `assert_eq!` are the proof's hypotheses

Two of them turn out to be exactly what the theorems need, which was not
arranged:

| runtime assertion | proof |
|---|---|
| `"scatter grouping does not tile the input"` | `the_group_widths_exhaust_every_bucket` |
| `"bucket offsets disagree with the entry total"` | `edge tot nb = total` |
| `"scatter group {} of bucket {} wrote {} entries too {}"` | **`the_chained_runs_tile_the_bucket`** |

The third is the one worth reading. Its comment says

> Post-condition, and it is a PROOF rather than a spot check. … If every group
> stopped exactly where the next one STARTED — and the last stopped at
> `off[b+1]` — then the groups' runs tile `off[b]..off[b+1]` exactly: every
> slot of `idx` was written, by exactly one thread, in bounds.

That is a **sufficiency claim about an observable**, argued in a comment and
checked by nothing. It is now three theorems. The route is that the observed
starts ARE the prefix edges of the (unknown) written counts
(`chained_starts_are_prefix_edges`), at which point the bijection applies —
and crucially it assumes **nothing about the histogram**, which is what makes
running the check on every call worth its 0.2–1.0 ms. Both directions of
dropping it are refuted as weakened theorems, not exhibited as witnesses:
`without_the_chaining_a_slot_can_go_unwritten` (a hole, which in `Idx` reads
back as the perfectly legitimate point index 0) and
`without_the_chaining_two_groups_can_write_one_slot`.

The empty-group case is stated separately (`an_empty_group_still_chains`)
because it is the case that check got wrong the first time it was written —
for an empty group, "where the next one started" and "where the next one
stopped" differ.

##### A third member of `clamped`, arrived at from a counting sort

A scatter group is a contiguous run of `grp` histogram chunks, clamped at
`nchunk`. That is `Decomposition.clamped`, whose two previous members are the
int32 flush interval (an overflow budget) and the output tile width (a memory
partition) — obligations this document already records as looking different
and being one family. This is the third, and it was reached from a different
kernel entirely.

##### The tie, and the oracle that did not exist

`zk_gpu_msm.rs`'s `binning_does_not_depend_on_the_thread_count` compares the
binner **against itself at one writer**. That is a strong check on the grouping
arithmetic and blind by construction to anything wrong at every thread count.
The MSM tests above it check the final curve *sum*, and this document already
records that an entry in the wrong bucket still yields a valid curve point.
**Nothing compared `Idx` against an independent placement.**

`tests/msm_counting_sort_model.rs` does, in two ways. A plain sequential
stable counting sort is the specification — the same relationship the naive
triple loop has to the tiled GEMM — and it deliberately shares `window_digit`
and the `base[w] + d` numbering, which decide which bucket an entry belongs to
rather than where it is placed and are checked against arkworks elsewhere.
Then `the_destination_map_is_the_one_the_proof_describes` evaluates
`CountingSort.dest` on real data: it recomputes both levels of edges from the
histogram and asserts every entry sits at the slot the map names. That is
stronger than the sequence comparison, which would accept two different cursor
tables that happen to yield the same order.

`scatter` records `(ngroup, span)` rather than the test recomputing them —
re-deriving `ceil(nchunk / group)` in the harness is a second copy of the
schedule, which is the defect `ExactGemmSchedule.v` exists to remove, in the
harness instead of the emitter.

##### MUTATION TABLE

| mutation | `counting_sort_model` | `zk_gpu_msm` | `proofs_are_checked` |
|---|---|---|---|
| **C1 `scatter_threads` always 1** | **FAIL** | ok | ok |
| C2 bucket numbering off by one, histogram AND scatter | FAIL | FAIL | ok |
| C3 post-condition deleted **and** all groups seeded at `off[b]` | FAIL | FAIL | ok |
| C4 post-condition deleted only | ok | ok | ok |
| C5 group base index `gi*chunk` not `gi*chunk*group` | FAIL | FAIL | ok |
| **C6 recorded `ngroup` wrong** | **FAIL** | ok | ok |
| P1 rank bound relaxed to `r <= w t` | ok | ok | FAIL |
| P2 `groups_exhaust_the_buckets` relaxed to `<=` | ok | ok | FAIL |
| P3 `dest` reads bucket 0's inner edges | ok | ok | FAIL |
| P4 span hypothesis dropped | ok | ok | FAIL |
| P5 `edge`'s base case `S O` | ok | ok | FAIL |
| P6 empty-group control moved to a non-empty group | ok | ok | FAIL |
| **T1 the oracle replaced by the implementation** | **FAIL** (after the fix below) | ok | ok |

**C1 is the row that matters.** With `ngroup` forced to 1 the second level of
the decomposition silently collapses, the scatter stops being parallel, the
runtime post-condition passes, and
`binning_does_not_depend_on_the_thread_count` passes *trivially* — every arm
becomes identical to its own one-writer reference.
`feedback-null-metrics-pass-dead-components`, in the one place a
self-comparison cannot help. C6 is the same shape aimed at the harness.

C2, C3 and C5 are caught by both suites: what the new file adds there is a
diagnosis by name (*"entry (point 13750, bucket 5881) is not at
dest = off[5881] + edge(gw[5881])[1] + 0"*) instead of a wrong curve point.
**Run the obvious mutations against the other suites before claiming
isolation** — three of six do not isolate.

C4 is a **confirmation**: deleting a redundant runtime check changes no
answer, and nothing should catch it. A "did you call my function" gate was
considered and rejected for the reason the `Ix` gate records.

##### The survivor was a real hole, and the fix had to be structural

**T1 replaces `reference_bins`' body with a call to `bin_by_digit`.** Every
comparison in the file then compares the binner against itself, and every
control still passes — they are computed from the (now identical) output.

It is not fixable by testing harder, and that generalises to every
differential in this repo. **A specification and a correct implementation
agree exactly**, so no behavioural check can distinguish "the oracle is right"
from "the oracle IS the implementation" — precisely when the implementation is
right, which is the property under test. What separates them is structural, so
`the_reference_does_not_call_the_implementation_it_is_the_oracle_for` reads
this test's own source and requires the reference not to reach into the
parallel binner.

**Writing that gate produced its own bug, twice.** Anchoring on
`src.find("fn reference_bins")` matched the **string literal on the gate's own
line**, so the extraction returned the gate's body and it then failed for the
wrong reason. Anchored on `"\nfn reference_bins("` — the definition at column
0 — it fires with the right diagnosis, verified by three probes: the function
renamed out from under it, the body no longer placing into buckets, and T1
itself. Two of my first three probes were *mis-aimed* rather than
informative: a global `sed` rename is a consistent rename and correctly passes,
and `inline_entries(scalars, g)` contains the substring `entries(scalars, g)`.

##### Found on the way: the shipped crate has a stale fork of the binner

`crates/y-gpu/src/msm.rs` — the consumer-facing library, where arkworks is a
real dependency — carries its own `bin_by_digit`, and it is the
**pre-optimisation** one: a serial `O(nchunk * nb)` cursor build (the 23.6 ms
phase this document records being cut to 3.3), one cursor row per histogram
chunk with no scatter grouping (the 2.2x this document records at nw = 20),
`cur0.to_vec()` per thread per call (the 1.34 ms this document records
deleting), and **no post-condition at all**. The measured binning work landed
in `tests/common/msm.rs` and never reached the crate that ships.

The theorems cover it regardless, and that is worth stating: it is the same
two-level map at `group = 1`, so `CountingSort.v` describes both. **A
decomposition is not a loop.** What it does not have is the runtime check or
any tie. Unifying the two is a refactor of a shipped crate and is recorded
here rather than done.

##### Found by running the suite: the `.ptx` race, in three more binaries

Verifying this increment made the zk suite fail once, in
`ptx_expression_coverage::a_v4_lane_is_still_a_member_access_that_works`, with
*"the v4 kernel emitted no loads"* — after many clean runs, which is this
repository's documented signature for the `.ptx` race. It did not reproduce in
three runs at HEAD or in three with the change, so it is **latent and
pre-existing**, not caused by the new work; adding a test binary shifts the
scheduling that makes it fire.

The mechanism is unambiguous. `--emit-ptx` writes next to its source, and
**three test binaries compile `tests/bn254_fr_mul_fast.ysu` in place** —
`ptx_expression_coverage`, `zk_gpu_field`, and `zk_gpu_groth16` through
`common/qap.rs`. `cargo test` runs them in parallel, so one truncates the file
while another reads it. The `ptx_for` helpers hold a mutex, and **a mutex is
per process**; there are five independent copies of that helper and none of
them can see the other binaries.

Fixed the way `committed_ptx_artifacts.rs` already does it: compile a copy in a
per-process temp directory, keeping `current_dir` at the repo so
`.ysu_hw_profile` is still found. Enumerating the cross-binary collisions first
showed there was exactly one — `bn254_fr_mul_fast` — so the fix is three sites
rather than the whole surface. Four consecutive zk runs and two default runs
clean afterwards, and no committed artifact is rewritten by a test run any
more.

CLAUDE.md already recorded the general rule — *"a gate that emits must run on a
COPY, and mine did not"* — and only the two artifact gates were fixed then. The
same defect was sitting in the kernel harnesses.

##### What is NOT claimed

`CountingSort.v` is facts about `nat`. Nothing in it is about memory, a `u32`
or a thread — the u32 width obligation is `scatter`'s own
`n * nw < u32::MAX` assertion, discharged there. Nothing says the histogram is
*right*, only that whatever it counted is placed exactly once; the bucket
numbering is checked against arkworks by the MSM tests. And the tie is by
running the real binner and comparing, not by byte-identity through `Ix`: this
is host Rust, not emitted code, so it is the `GridStrideSplit` grade of tie
rather than the exact GEMM's.


#### Pulling the same thread: the library ships kernels the repository never tested · 2026-08-31

The counting-sort work ended with a note that `crates/y-gpu` — the
consumer-facing crate, the one that ships — carries a stale fork of the host
binner. Following that led somewhere worse: **`cargo test -p y-gpu` was red,
and nothing in the documented build commands runs it.**

`cargo test` at a workspace root with a root package builds *that package
only*. The crate's own `ptx_is_not_stale.rs` needs `-p y-gpu` or `--workspace`,
and neither appears in CLAUDE.md's build section. So the gate that exists
precisely to stop the embedded kernels going stale had been failing, unseen.

**Four of the five embedded kernels were stale by the whole carry-flag
intrinsic series.** `bn254_fr_mul_fast` shipped at 2,112 lines with **zero**
`add.cc`, against 1,255 now; `bn254_msm_bucket` at 34,343 against 18,222. The
GPU ZK library — the MSM, the NTT, the Groth16 QAP — was running the
pre-carry-chain kernels, which this document measures at 1.06x slower and 48
registers instead of 36.

##### The gate could not have passed anyway, and that is the more interesting half

It compared the embedded copy against a compile **on the build machine**. The
whole point of these artifacts is portability: they are committed at
`.target sm_80`, and a fresh compile here probes an sm_89 card and says so.
Those are contradictory requirements, and freshness is the one that has to
give — *a compiler that probes the local machine bakes that machine into its
output* is the finding the sm_80 regeneration existed to fix. It also produced
a **false positive**: `bn254_permute` was reported stale and was not, its body
being byte-identical.

So the replacement does not choose a target. It **asks the artifact which
target it claims** and pins a `.ysu_hw_profile` to that before recompiling,
the way `ptx_portability::emitted_module_for` already does. Whether the claimed
target is legitimate is a different question, and a different gate answers it.

##### Three gates, and each of six mutations is caught by exactly one

| mutation | freshness (root) | portability (root) | wiring (crate) |
|---|---|---|---|
| **Y1 one kernel reverted to the stale version** | **FAIL** | ok | ok |
| **Y2 header re-targeted at the build machine's card** | ok | **FAIL** | ok |
| **Y3 `include_str!` mis-wired to another kernel** | ok | ok | **FAIL** |
| Y4 the `ALL` pairing swapped (right name, wrong module) | ok | — | **FAIL** |
| Y5 the freshness gate compiles in the repo, not the pinned dir | **FAIL** | — | — |
| Y6 the sweep finds nothing | **FAIL** | — | — |

Y1 is the failure that was live for a week. Y5 is the one that matters for the
design: compiling in the repo makes the gate fail *here*, which is exactly the
state the crate's version was in — so the pin is load-bearing rather than
tidy. Y2 passing the freshness gate is deliberate and not a hole: the gate
reproduces the target the artifact names, so re-targeting changes both sides.
Portability owns that question.

Freshness now lives in `tests/committed_ptx_artifacts.rs`, under the plain
`cargo test`. The crate keeps what only it can see — that the strings compiled
into the binary really are those files, filed under entry names the driver will
be asked for. A mis-wired `include_str!` embeds a kernel that is fresh,
portable, assembles, and is the wrong one.

##### The regeneration is confirmed by the prover, not by the diff

The sm_80 and sm_89 bodies are **byte-identical** — only the two header lines
differ — so what the GPU tests exercise at sm_89 is instruction-for-instruction
what ships at sm_80. And `cargo test -p y-gpu` now runs
`gpu_proof_matches_arkworks_sparse` and `_dense` on the newly embedded kernels:
real Groth16 proofs, checked against arkworks, plus a tampered-statement
rejection.

639 default / 905 zk / 8 y-gpu tests, all green. The stale *binner* fork is
still there; it is a refactor of a shipped crate and remains recorded rather
than done.


#### The fork itself, and the tests that dispatch had disarmed · 2026-08-31

The previous two entries fixed what the library *embeds*. This one fixes what
it *runs*, and the finding underneath it is worse than a stale copy.

`crates/y-gpu/src/msm.rs` and `tests/common/msm.rs` were a **whole-module
fork**, not a stale binner: 15 items byte-identical, 10 diverged, and the crate
had nothing the tests did not. Every measured MSM improvement had landed in the
test copy and none had reached the crate that ships — the parallel prefix
(23.6 → 3.3 ms), the grouped scatter and its thread count (2.2x at nw = 20),
the removal of a per-thread `to_vec` (1.34 ms), the exactly-once
post-condition, and the 128-thread block.

**That last one is a live performance regression in the shipped path**, and it
is measured here rather than quoted: `crates/y-gpu` launched the bucket kernel
through a generic helper with a hardcoded 256-thread block, and on the merged
code at n = 2^20, kernel time, 256 → 128 gives nw=20 **42.25 → 29.46 ms
(1.43x)**, nw=22 48.80 → 32.73 (1.49x), nw=25 53.86 → 40.80 (1.32x).

##### It also made `CountingSort.v` a proof about code that does not ship

`tests/msm_counting_sort_model.rs` tied the counting-sort theorems to the test
copy. The library ran something else. That is the same gap as "proof-carrying
described the repository, not the output", one layer further out, and it is the
reason unifying is proof work rather than tidying.

So there is one implementation now and it is the shipped one:
`tests/common/msm.rs` went 966 → 466 lines and re-exports the host layer from
`y_gpu::msm`. What stays behind is the harness a library must not have —
locating and running the `Y` binary, compiling a `.ysu` to PTX, and the device
layer's panic-on-error contract. The root package gains a **dev-dependency
cycle** on `y-gpu`, which cargo supports because the cycle is only through
dev-dependencies.

##### The decisive experiment, and it is the one that justifies the whole change

Put a cursor bug in the *shipped* binner — every scatter group seeded at
`off[b]`, so the groups write over each other and `Idx` is corrupted — and run
everything:

| | `zk_gpu_msm` | `counting_sort_model` | `zk_gpu_groth16` | `cargo test -p y-gpu` |
|---|---|---|---|---|
| **before unification** | ok | ok | ok | **ok** |
| **after unification** | FAIL | FAIL | — | FAIL |

Before, that bug was invisible to **every test in the repository**. Two
independent reasons, and the second was a surprise: the root's tests used the
other copy, *and* the crate's own prover tests never reach the GPU at all.

##### Dispatch had silently disarmed the crate's tests

`crates/y-gpu/tests/prove.rs` runs 2,048 and 4,096 constraints. Both are far
below `MSM_GPU_MIN_STAGED = 40,000`, so `gpu_is_worth_it` sends every MSM to
the CPU and **the tests would have passed with the entire GPU MSM path
deleted**. This document already records that exact trap for the root suite —
*"at 4,096 constraints everything routed to the CPU and the GPU tests would
have kept passing with the kernel deleted. They pass `force_gpu` and assert
they got it"* — and the root suite was fixed for it while the crate was not.

`prepare_forcing_gpu` is the same fix, and it is load-bearing rather than
tidy: with it, the cursor bug fails both prover tests; revert it alone, leaving
the bug in place, and `cargo test -p y-gpu` is **completely green**. The test
also asserts at least three of the four queries actually went to the device, so
it cannot quietly become a CPU test again.

##### A performance property needs a performance guard

Reverting the launch to 256 threads is caught by nothing — it computes the
right answer, slowly. So the launch decision is extracted as a pure function,
`bucket_launch_geometry`, and `the_bucket_kernel_is_launched_at_the_tuned_block`
asserts it without needing a device: one thread per bucket, blocks of
`bucket_block()`, grid × block exactly `nb`. Its control pins that the default
is still the measured 128 and that the override still reaches the launch —
without which a `bucket_block()` that had drifted back to 256 would satisfy
every other assertion in the test.

640 default / 906 zk / 8 y-gpu tests, all green.


### The weakest tie, and the mutation that survived it · 2026-08-31

`GridStrideSplit.v` proves that the exact-attention kernel's grid-stride
reduction visits every key exactly once, in any order, at any worker count.
It proves it for an **abstract** worker count — and nothing said the kernel's
count was one of them. The two expressions it rests on, which thread is which
worker and how many workers there are, were hand-written in a PTX template;
the proof quoted them in a **comment**; and `tests/exact_attention_schedule.rs`
matched the two back up by parsing emitted PTX with a reaching-definition
walker. This file said so in as many words: *"this tie is weaker than the GEMM
kernels' and is named as such"*.

The quoted comment had already gone stale. It showed
`mul.lo.s32 %r9, %r9, %r5` — the two-writes-to-one-register form the kernel
stopped using when that reuse was found to be a trap for anything reading the
PTX back. A proof's account of the code drifted from the code, silently, which
is the whole argument for not keeping accounts in prose.

**`Ix` now serves two backends, and that is the increment rather than a side
effect.** The schedule is `sched_scores` / `sched_accum` in
`src/exact_attention.rs`, rendered to PTX by `Ix::render_ptx` and to Coq by the
generator in the test file. An extraction layer that reaches exactly one code
generator is a one-off; pointing it at a target with a different instruction
set, a different register discipline (special registers must be `mov`'d before
use) and a fused multiply-add is what answers "does this generalise".

- **The extraction reproduced all three hand-written sequences instruction for
  instruction** — register numbering, `mad` fusion and lazy `mov` placement
  included. Only comments moved. That was not arranged: a renderer that
  materialises a special register at first use and fuses `a*b + c` produces
  exactly what someone writing PTX by hand produces, which is why the refactor
  is checked by the artifact instead of by reading it.
- **What it buys is the theorem `GridStrideSplit.v` could not state.**
  `proofs/AttentionSchedule.v` is the emitter's own `nworkers`, and
  `the_emitted_launch_geometry_visits_every_key_exactly_once` instantiates the
  partition at it. The obligation is the **pairing**: `worker` mixes three
  hardware indices with two radices and `nworkers` must be the product of
  exactly those three indices' extents. Drop a factor and two threads share a
  residue class, so their keys are accumulated twice; add one and some class is
  claimed by nobody, so its keys are dropped. Neither is a crash and neither is
  reliably a wrong-looking number on random data.
- That statement is a **bijection** from the launch geometry's coordinate box
  onto `[0, nworkers)` — a mixed-radix positional index, so `MixedRadix.v`
  discharges the injectivity with no new reasoning. **Third consumer of that
  schema, and the first reached from a GPU launch geometry rather than a GEMM
  tile.**

**THE SURVIVING MUTATION IS THE FINDING, AND IT IS A HOLE IN THE GENERATOR'S
OWN DESIGN.** Swap two radices in the emitter *and regenerate* — so the
byte-identity gate passes because both sides moved — and the whole sweep stayed
green. The cause: the generated parameter list is **derived** from the
expression's free names, so a swap renames the parameters in step, and every
theorem applies `worker_accum` **positionally**. The definition remains the
same function under different labels, and no proof can see a relabelling.

- Deriving the list is right and stays: a schedule that gains or loses an index
  must change the signature. What was missing is anything that pins **which
  index plays which role**.
- The fix is a theorem whose **binders are generated and whose right-hand side
  is fixed text**: `worker_accum <derived binders> = ctaid_z * (nctaid_x *
  ntid_x) + ctaid_x * ntid_x + tid_x`. Swap the radices and the binder list
  moves while the equation does not, so `ring` fails and `coqc` rejects the
  file. Drop an extent and it is caught harder: the binder disappears while the
  right-hand side still names it, so the reference is unbound.
- **That mutation is then caught by `coqc` and by nothing else** — the
  byte-identity gate correctly passes, because the description and the proof
  agree with each other perfectly. It is the two-layer property demonstrated
  rather than asserted, and it is the same standing limit already recorded for
  the exact-GEMM schedule: *when the description moves, both consumers move
  with it and byte-identity is silent.*

**The dataflow walker is KEPT, and a mutation says why.** Swapping the last two
result registers of `attn_accum`'s worker count leaves the loop striding by the
**partial** product `nctaid.z * nctaid.x`. The rendered block is still correct
and still present; what is wrong is which register the loop reads. Caught by
`the_worker_index_and_the_worker_count_use_paired_registers` and by nothing
else. Rendering says what the kernel computes; the walker says the loop
actually reads it, which is the decorative-codegen question one layer over.

**Eleven mutations, all caught, and the split is the useful part.** A radix
swap with the instruction count unchanged (**schedule gate alone**); the same
swap regenerated (**`coqc` alone**); the stride-register swap (**walker
alone**); a generator neutered to echo the committed file (**the
function-of-its-argument control alone**); the join corollary deleted from
`GridStrideSplit.v` (**content control alone**). Dropping `nctaid.z`, removing
`mad` fusion, giving the naive entry a different schedule and hand-editing the
committed proof are each caught by two to four suites — diagnosis by name
rather than isolation, and the entry says which is which.

643 default / 909 zk / 8 y-gpu tests, all green.


#### PHASE 4'S FIRST THEOREM: a PROVED error bound for the fixed-point softmax · 2026-09-01

Phase 4's Done-when is "a softmax or a normalization layer carries a proven
error bound rather than an empirical one", and it is the phase that decides
whether the addressable set is wider than pure reductions. **Every theorem in
`proofs/` before this one is an EXACTNESS or a COVERAGE claim** — which is not
a stylistic preference, it is §0 of the process doc, the precondition that
makes the relationship between kernel and specification an equality.

`proofs/SoftmaxErrorBound.v`, 15 `Print Assumptions`, no axioms, nothing
admitted. **653 default / 940 zk / 8 y-gpu, all green** (up 6, the new tie
file). No `src/` behaviour change, so every emitted module is byte-identical.

##### The ingredients all shipped, and the COMPOSITION was bounded by nothing

Each piece was pinned individually and well:

- `fixed_exp::exp2_neg_q16_16` — `it_is_sub_ulp_accurate_everywhere` is
  EXHAUSTIVE over `0 .. 31<<16` against `f64::exp2`, worst 0.908 ulp.
- `MAX_EXACT_SEQ_LEN` is *derived* (`2^63 / ((2^28-1)*127)`), so the int64
  accumulator provably cannot wrap.
- the `min.s64 ..., 1073741824` saturate has its own necessity test.

What existed for the OUTPUT was `tests/attention_quantization_error.rs`, whose
own stated bar is comparative and empirical — *"is it more wrong than the f32
online softmax that production flash attention already ships?"* — on synthetic
score distributions. That sentence is what this replaces.

**The interesting part is not the exp.** Exhaustion over a finite domain is
stronger than a proof about the table and the series would be, so the exp
enters as a HYPOTHESIS. The chain around it is what nothing covered:

    m - s -> ((m - s)*KFix + 2^15) >> 16 -> min(.., 2^30) -> Q0.28 -> exact
             int64 accumulate -> divide

##### Three joints, and the second is the one that existed nowhere

- **The argument reduction rounds an EXPONENT**, so its error is
  *multiplicative* in the weight while the exp's ulp is *additive*. They
  compose as `EPS*w + 1`, not as a single number, and that shape is what makes
  the max subtraction load-bearing below.
- **The saturate is outside the swept domain.** The exhaustive sweep stops at
  `31<<16 = 2,031,616`; the emitted `min.s64` admits arguments to `2^30`, **528
  times further out**. `the_swept_domain_covers_the_admitted_one` is the
  two-line argument that closes it — above the table the implementation returns
  0 and the ideal weight is below a quarter of an ulp, because `31*2^32` is
  past thirty halvings of `2^28`. So the "0.908 ulp everywhere" headline does
  hold on the whole domain the kernel can reach; it simply had never been
  established. *A number in a comment is not a check*, in the one place the
  number was not even in a comment.
- **The max subtraction had to be carried through.** It is usually described as
  an overflow guard. It is also what makes the table's ABSOLUTE error
  relatively negligible: `delta` is zero at the argmax, so one weight is
  exactly `2^28` and `Wtot >= 2^28` — the additive term is `n` ulps against a
  total of at least `2^28` of them.
  `a_total_weight_below_one_ulp_admits_a_zero_denominator` is the refutation,
  and it is not hypothetical: it is the F=16 attention-sink failure
  `attention_quantization_error.rs` already records, one layer up — every
  non-sink weight rounded to zero and the whole tail disappeared.

##### THE ANSWER TO THE PHASE 2 QUESTION: bounds compose here for a stateable reason

The question was whether the `Decomposition` schema extends to approximate
arithmetic. **It does, and narrowly: the error is introduced BEFORE the
decomposition, not by it.** `L` and `O` are exact integer reductions, so
`GridStrideSplit.grid_stride_exact` applies verbatim and the per-element error
enters the fold as *data*.
`the_bound_holds_at_every_launch_geometry` is that join — the bound is a
property of the kernel rather than of one launch.

An f32 softmax has no such split: its error is produced BY the reduction, so
the bound would have to be re-derived per decomposition. **That contrast is the
actual content of "bounds compose differently", and it says the addressable set
widens to pipelines whose approximation is PER-ELEMENT, not to approximate
reductions in general.**

##### The bound as a number, and how tight it is

`2*VMAX*(EPS*Wtot + n) / L`, with `EPS = 2^-16` the argument reduction's
relative term and `n` the exp table's absolute one. At **65,536 keys** with
int8 `V` the expression evaluates to `4318/65519 = 0.0659`, so the corollary
states `7/100` — the round number above it, not a fitted constant. At `n = 2^20`
the same expression is `0.99998`, i.e. still under one unit of `V` but only
just, which is a real statement about where Q0.28 runs out.

Measured tightness, from the tie: the **per-element** bracket is tight — worst
observed error is **0.49 of the allowance** over 3,540 points. The end-to-end
output bound is 180–10,000x loose on random data, because per-element errors
cancel in the sum. Both numbers are reported; a bound quoted without its
slack is half a claim.

##### No Reals, and that forced a real design decision

`tests/proofs_are_checked.rs` requires every `Print Assumptions` to report
`Closed under the global context`. Coq's `R` is axiomatized, so one
`Require Import Reals` puts seventeen axioms under the file. Everything is `Z`
and `Q`.

**The obvious interface for the ideal weight is INCONSISTENT over Q.** Exact
homogeneity `W(u+v)*2^28 = W(u)*W(v)` plus one exact halving forces
`W(2^31)^2 = 2^55`, and no rational squares to it —
`exact_homogeneity_is_unsatisfiable_over_Q`, on top of a from-scratch
`no_rational_squares_to_two` (prime 2 divides both numerator and denominator of
a fraction in lowest terms). An inconsistent hypothesis set proves everything,
so this is not pedantry: the file would have been worthless.

So the interface is a two-sided rational **bracket**, and both directions of
*check your premise is satisfiable* are run: `the_interface_is_satisfiable`
exhibits the model `2^28 * (1 - 2^-32)^u`, and
`the_per_unit_factor_is_what_one_unit_of_log2_costs` proves the rational
inequality `(1 - 2^-32)^(2^32) <= 1/2` that fixes the constant — the
real-analytic fact `2^(-2^-32) >= 1 - 2^-32`, made checkable without ever
mentioning a real.

##### A mechanical trap worth carrying: `ring` NORMALISES, and a big power is a landmine

`ring` (and `auto`) on a goal mentioning `qpow BETA (Z.to_nat 4294967296)` asks
for four billion multiplications: **`Stack overflow`, after 85 seconds, at
`Qed` rather than at the tactic**. `remember` does not help — it leaves a
let-binding `ring` zeta-reduces through. The fix is to prove the algebra over
ABSTRACT rationals and `apply` it, so the big term only ever meets first-order
unification. `lra`/`nra` are safe (they abstract atoms) and so is the kernel:
the theorem that STATES the big term checks in milliseconds. Diagnosed by
bisection, not guessed.

##### MUTATION TABLE — nine mutations, each `--test` target run separately

| mutation | `proofs_are_checked` | `softmax_error_bound` | `exact_attention_bounds` | `attention_quantization_error` | `__lib__` |
|---|---|---|---|---|---|
| M1 proof's `HALF` halved | **FAIL** | **FAIL** | ok | ok | ok |
| M2 twenty halvings, not thirty | **FAIL** | ok | ok | ok | ok |
| M3 `EPS` too tight for `ALPHA` | **FAIL** | ok | ok | ok | ok |
| M4 `VMAXQ` dropped from the bound | **FAIL** | ok | ok | ok | ok |
| M5 emitter loses the saturate | ok | **FAIL** | **FAIL** | ok | ok |
| M6 emitter loses round-to-nearest | ok | **FAIL** | **ok** | ok | ok |
| M7 the swept domain narrows | ok | **FAIL** | ok | ok | **FAIL** |
| M8 exp loses its early return | ok | **FAIL** | ok | **FAIL** | **FAIL** |
| M9 the zero-denominator refutation deleted | **ok → FAIL** | ok | ok | ok | ok |

**M2, M3 and M4 are caught by `coqc` alone**, which is the taxonomy working:
the emitted code is unchanged, so there is nothing for a behavioural suite to
see. **M6 is caught by the new file alone** — `exact_attention_bounds` pins the
`>> 16` and the `min.s64` and *not* the `+ 32768`, so the round-to-nearest
addend was covered by nothing until the proof's `HALF` had to equal it.

##### M9 WAS A REAL SURVIVOR, AND THE HOLE WAS IN THE GATE THAT GUARDS ALL 17 PROOFS

`each_proof_still_proves_the_thing_it_exists_for` was a bare
`src.contains(needle)`. **Every proof file names its own theorems in its header
doc comment** (`[the_name]`), so deleting a theorem *together with its
`Print Assumptions` line* — the line has to go too, or `coqc` catches the
dangling reference — left the whole gate green. Measured, not reasoned about:
all four tests passed on a `SoftmaxErrorBound.v` with the refutation removed.

`names_something_real` requires a declaration site now (`Theorem`/`Lemma`/… at
the start of a line, name ending there), or the literal `Print Assumptions`.
All 17 files pass unchanged, so no existing entry was relying on the hole, and
the fix was confirmed to generalise by deleting a header-mentioned theorem from
`GridStrideSplit.v` and watching it fire.

**It is a pre-existing hole found by mutating a new file.** That is the
argument for running the mutation sweep against the *gate* and not only against
the subject.

##### What this does NOT claim

- **It is not a statement about `f64::exp2` or the real exponential.** `W` is
  abstract, constrained by four properties the true function has; nothing here
  proves it has them. That is a TCB item and it sits beside `vpdpwssd`'s
  semantics rather than above them.
- **The exp's accuracy is a hypothesis**, discharged by exhaustion in Rust over
  the *swept* domain. The extension to the admitted domain is proved here.
- **`KFix` is itself a rounded temperature** and this file does not price that;
  it is a uniform reparameterisation of the exponent, and
  `the_temperature_multiplier_carries_two_to_the_thirty_two` is where it lives.
- **Nothing here is about int8 quantization of Q, K or V.**
- **The tie is the `GridStrideSplit` grade** — `tests/softmax_error_bound.rs`
  runs the real `exp2_neg_q16_16` and the real emitted arithmetic and reads the
  proof's constants out of the `.v` — not the exact GEMM's byte-identity.
- **`exp2_neg_q16_16` has FOUR transcriptions** (this Rust one, the PTX in
  `ptx_device_function`, and two Python ones). A theorem about one of them is a
  theorem about all four **only** because `EXP2_DOMAIN_DIGEST` pins them
  together over the whole domain. Said here because it is easy to assume and
  wrong if the digest ever goes.

653 default / 940 zk / 8 y-gpu tests, all green.


#### The temperature was quantized too, and its two preconditions lived in a Python script · 2026-09-01

`proofs/SoftmaxErrorBound.v` landed earlier the same day with a **"what this
does NOT claim"** section, and one of its four items was reachable:

> **`KFix` is itself a rounded temperature** and this file does not price
> that. `KFix = round(C * 2^32)` for a real `C = q_scale*k_scale/sqrt(d)`, so
> the kernel computes the exact-for-KFix softmax of a slightly different
> temperature.

It is priced now — Part 6b, seven new theorems, still no axioms and nothing
admitted — and the increment is worth reading for what pricing it *turned up*
rather than for the bound.

##### The shape of the answer is a head_dim question, not a KFix one

Rounding the multiplier moves the exponent by `delta * |KFix - C*2^32|`, so at
most `(delta + 1)/2` in units of `2^-32` log2. That error is proportional to
the **score delta** and **independent of `KFix`** — which is not what you would
guess, and it means a bound on the delta is the whole bound. The delta is
bounded by the head_dim: `|s| <= 127^2 * hd` for an int8 pipeline, so
`delta = m - s_i <= 2 * 127^2 * hd`, a compile-time function of one parameter.

At head_dim 128 that is a factor under **1/2048** on any ideal weight
(`at_head_dim_128_the_temperature_moves_a_weight_by_under_a_two_thousandth`).
Measured: the exponent bound is attained **exactly** — 1.000 of its half-delta
allowance — and the weight bound is **1.47x** slack, the slack coming from
Bernoulli's linearisation of `BETA^n` and from rounding `n` up to `(d+1)/2`.
Report the slack, not just the bound.

##### THE FINDING: `round(C * 2^32)` had SEVEN transcriptions and was CHECKED IN ONE

Every test that needed a multiplier wrote `(c * 2f64.powi(32)).round()` itself
— four sites in `tests/softmax_error_bound.rs`, two in
`tests/exact_attention_bounds.rs` — and `src/` had none. The only place the
result was validated at all was `tools/ptx_bridge.py`:

```python
kfix = int(round(c * 2.0 ** 32))
if kfix <= 0 or kfix >= 2 ** 31:
    continue
```

A bare `continue`, so a temperature outside the representable range **silently
removed a case from a measurement** instead of failing it —
`feedback-conditional-gates-skip-silently`, in the script whose whole job is to
validate the kernel against a real model. Skips are counted and printed now.

Both bounds are real, and neither was derived anywhere:

* **`KFix == 0`**, reachable whenever `C < 2^-33`. Then `t = (0 + 2^15) >> 16`
  is `0` for **every** key, so every weight is exactly `2^28` and the softmax
  is **uniform** — it carries no information about the scores at all. That is
  the same symptom `ptx_bridge.py`'s own finding 06 records from the opposite
  cause (the bridge passed `C * 2^16`, a multiplier `2^16` too small), and its
  differential could not see it because both arms replicate the kernel's
  formula and agree bit for bit on a uniform answer.
* **`KFix >= 2^31`**. The parameter is declared `.param .u32 q6` and consumed
  by `mul.wide.s32` — a **signed** wide multiply. The two readings are
  different numbers, and not by a rounding: at `KFix = 2^31` and one unit of
  delta the unsigned reading gives argument `32768`, a weight of
  `2^28 * 2^(-1/2)`, while the signed one gives `-32768`, which the
  `cvt.u32.u64` below the saturate wraps to `4294934528` — far above the
  table, so the weight is **zero**. The key vanishes. Note the saturate cannot
  help: `min.s64` bounds from *above* and the product is negative.

Both are `exact_attention::temperature_fixed_point` now, which refuses them by
name and says why, and both are theorems
(`a_zero_multiplier_gives_every_key_the_same_weight`,
`the_two_readings_of_the_multiplier_disagree_above_two_to_the_thirty_one`,
`the_signed_reading_zeroes_a_weight_the_unsigned_one_keeps` — the last
discharged from the exp's own early-return hypothesis, so it is the *proof* that
says the key vanishes, not an observation).

**The quantization error itself is NOT refused**, and saying so is the point of
having the bound: it is bounded rather than catastrophic, so a compiler that
refused it would be refusing every temperature.

##### A `nat` literal is UNARY, and that is the same landmine as the `ring` one

`at_head_dim_128_...` needs `n = 2,064,513`. Writing that as a `nat` literal
builds **two million constructors**, and the first version of the proof hung —
`simpl`, `Z.of_nat` and `lia` all normalise through it. `Z.to_nat 2064513` is a
*term*, and `Z2Nat.id` recovers the integer; the file goes from not finishing
to **13.9 s**.

This is the second instance of one lesson in two sessions. The first was `ring`
reflexively normalising `qpow BETA (Z.to_nat 4294967296)`. The rule to carry:
**in a proof about fixed-point arithmetic, the constants are large, and any
tactic that normalises will try to evaluate them.** Keep every large number in
`Z`, and let `nat` see only `Z.to_nat` of it.

##### Mutation table — 10 mutations, each `--test` target run separately

| mutation | proofs_are_checked | softmax_error_bound | exact_attention_bounds |
|---|---|---|---|
| T1 exponent bound loses its `+1` | **FAIL** | ok | ok |
| T2 the proof's score span coefficient | **FAIL** | **FAIL** | ok |
| T3 the head_dim-128 corollary claims 1/8192 | **FAIL** | ok | ok |
| T4 the bracket lemma's hypothesis weakened | **FAIL** | ok | ok |
| T5 compiler stops refusing `KFix = 0` | ok | **FAIL** | ok |
| T6 compiler stops refusing the signed limit | ok | **FAIL** | ok |
| T7 `score_delta_span` drops its factor of two | ok | **FAIL** | **FAIL** |
| T8 compiler's signed limit disagrees with the proof's | ok | **FAIL** | ok |
| T9 *control* — refuse every temperature | ok | **FAIL** | **FAIL** |
| T10 the worst-case temperature leaves the sweep | ok | **FAIL** (after the fix below) | ok |

`attention_quantization_error`, `exact_attention_schedule` and the lib/bin unit
tests are `ok` for all ten, which is the taxonomy working: T1–T4 change no
emitted code, so no behavioural suite can see them.

**T10 SURVIVED, and the hole it exposed is one this repository already has a
name for.** The sweep's non-vacuity floor asserted the worst *weight* factor
reached a quarter of the allowance — and the realistic temperatures alone reach
**57%** of it, so removing the deliberately-worst-case `C` left the floor
passing. What the worst case actually buys is the **exponent** bound being
*attained*, and that number was `println!`ed rather than asserted. Both this
document and the proof's header claim the bound is tight; *a number in a
comment is not a check*, and a number in a `println!` is not either. The test
now asserts `worst_exp > 0.99`, and T10 fails with the right diagnosis (`worst
was 0.837 of it`). 10/10.

##### Where this leaves the "what this does not claim" list

Three of the four items stand and are structural rather than reachable: the
exp's accuracy is a hypothesis discharged by exhaustion in Rust (which is
*stronger* than a proof about the table would be); nothing here is about int8
quantization of Q, K or V; and the tie is the `GridStrideSplit` grade — the
proof is read against the running code, not rendered with it from one `Ix`.
The fourth is gone.

`657 / 944 / 8` tests, both builds, no committed artifact changed.

#### Nine compiler warnings, and two of them were the soundness core · 2026-09-01

Both builds carried warnings — 9 in the default configuration and 11 under
`--features zk` — and had done for long enough that nobody read them. They are
zero now, and the sweep is worth recording because the class this repository
already tracks is exactly what a `never used` warning reports: **dead code is
where the bugs hide**, which is why `run_all_optimization_passes`,
`c_emitter.rs` and the `SmemLayout` surface each turned out to be findings
rather than tidying.

Seven were mechanical, and two carried a claim.

##### `VnniExact::licenses` was a SECOND, WEAKER licence predicate

Phase 0's soundness core is `VnniExact::license` — the rule that says a tiled,
threaded, K-split GEMM may use `vpdpwssd` and still be bit-identical to the
naive nest. A `bool`-returning twin sat beside it, `pub`, reading as the cheap
form of the same question:

```rust
pub fn licenses(&self, m: f64) -> bool {
    m.is_finite() && m >= 0.0 && m <= self.max_operand_magnitude()
}
```

That is `license` **without the `m < 1` refusal** — the fix for the
unsoundness this document records one entry above, where `@bounds(-0.001,
0.001)` was licensed at magnitude 0.001 and every such operand stages to int16
*zero*, so the kernel computes the zero matrix under a certificate claiming
exactness. Measured before deleting it:

```
m=0.001  licenses()=true   license()=Err(not representable as int16)
m=0.5    licenses()=true   license()=Err(not representable as int16)
m=4095   licenses()=true   license()=Ok
m=4096   licenses()=false  license()=Err(can overflow the int32 accumulator)
```

It had no *callers* — but it had **unit tests**, four of them, asserting its
behaviour. So it read as maintained code, and the only thing saying otherwise
was a warning nobody looked at. Deleted; its tests now drive `license`, which
strengthens them, since one of the four is the tie between the licence and
`ExactGemmMicro`'s proof hypothesis and it now compares the *shipping*
predicate against the proof.

**The gate is `the_magnitude_rule_lives_in_exactly_one_place_and_gives_a_reason`,
and its first version was wrong in an instructive way.** It asserted there was
exactly ONE licence-named function and failed immediately on
`license_vnni_exact` — which is legitimate: it is the free function
`cpu_gemm` calls, and it *delegates* the magnitude question rather than
re-deciding it. A wrapper is not a second implementation. So the property is
"delegates, or is the original", not "is unique" — **the gate's statement was
wrong, not the code**, and running it is what said so. It also requires the
signature to return `Result<_, String>`: a caller handed `false` cannot tell an
unrepresentable magnitude from an overflowing one, and those have different
repairs.

##### The refusal computed its reasons and threw them away

`select_repr` walks the four candidate representations and records why each
lost — "not exact: floating-point addition is not associative", "insufficient
range or resolution (holds |x| < 3.277e4 at 16 fractional bits)". On success it
returned them in `Decision::rejected`; on failure it returned `None` and
**dropped the list three lines after building it**. So a user hitting

> `@ZeroDrift on 'acc: F32' cannot be honoured. No exact representation holds
> that range at that resolution`

could not be told which four things were tried or why each failed. The struct's
own doc comment said the field existed "so the compiler can report a real
reason rather than an advisory", and no backend read it — `rejected` and
`measured` were both reported as never-read by `cargo build`.

`select_repr` returns `Result<Decision, Vec<(DriftRepr, String)>>` now, both
backends print the reasons on refusal, and the success path names how many
candidates lost (`Y_DRIFT_EXPLAIN=1` for the list). `measured` was a `bool`
field set in lockstep with `cost_ps` — two fields encoding one fact, which is
how "two fallbacks disagreeing, with the exposed one unsafe" starts — and is a
derived method. The advisory line itself is one shared function, because the
two backends each carried their own copy of the phrasing.

##### The mechanical seven, and why each was correct to be a warning

Two are *confirmations that a fix is in place*, which is worth knowing before
reaching for `#[allow]`: `ptx_emitter`'s barrier-hoisting scan no longer needs
`let mut j`, because the fix was to **stop** at the first statement it cannot
hoist rather than `j += 1` past it — the warning is that unsoundness being
visible from outside. `zk_emitter`'s `Stmt::While` arm binds `condition`,
`body` and `max_iterations` and uses none of them, because the
`@max_iterations` masked-unroll lowering was **withdrawn** for computing the
wrong function. Likewise `cpu_emitter`'s `@ZeroDrift` arm, which refuses. Those
are `..` patterns now, with the reason beside them. The rest: `DriftRepr::c_type`
was a leftover of the deleted C backend; `sentinel`'s two `unsafe` blocks around
`__cpuid` are redundant in this Rust; `cpu_gemm`'s `take` closure no longer
mutates; and `main`'s `counts()` is genuinely unreachable in a default build
because every caller is on the `--features zk` timing path — the one case that
takes an `#[allow]`, with that sentence next to it.

##### AND MY OWN SWEEP HARNESS HAD THE NULL-METRIC BUG

Fixing the last `licenses` caller left `tests/exact_gemm_micro_model.rs`
**failing to compile** — and the per-target sweep script reported a completely
green run, twice. A target that does not build emits no `test result` line at
all, and the script's `[ -z "$LINE" ] && LINE="test result: NONE"` did not match
its `*FAILED*` case. *Null metrics pass dead components*, in the harness written
to apply that rule. It counts a missing result line as a failure now, and the
aggregate `cargo test` is what caught it.

##### Mutation table — 5 mutations, each `--test` target run separately

| mutation | licence gate | micro_model | backend_agreement | lib unit |
|---|---|---|---|---|
| L1 the `bool` twin returns verbatim | **FAIL** | ok | ok | ok |
| L2 a `Result` twin that does not delegate | **FAIL** | ok | ok | ok |
| L3 the refusal drops the reasons again | ok | ok | **FAIL** | ok |
| L4 the report stops naming what lost | ok | ok | ok | **FAIL** |
| L5 *control* — a legitimate delegating wrapper | ok | ok | ok | ok |

Each isolated to exactly one gate, and the control is what stops the licence
gate degenerating into "no wrappers allowed".

`661 / 948 / 8` tests, both builds **warning-free**, no committed artifact
changed. (Those first two counts are INFLATED — the entry below shows 134 and 170
unit tests were being run twice, because `main.rs` compiled the whole compiler
a second time. The real figures at this commit are `527 / 778 / 8`.)

#### The dead-code census was 90% noise, because the compiler was compiled twice · 2026-09-01

The previous increment read the build's nine warnings and found two of them
were the soundness core. Twenty-six modules opt out of that census entirely
with a crate-level `#![allow(dead_code)]` — every emitter, `type_checker.rs`,
`zk_emitter.rs`, `cpu_gemm.rs`, `parser.rs`, `ast.rs` — and `src/zero_drift.rs`,
where the finding surfaced, is one of the few soundness-critical files without
one. Stripping all twenty-six surfaces **69 warnings** in the default build and
**85** under `--features zk`.

**Almost none of them mean what they appear to mean, and that is the increment.**

##### `src/main.rs` re-declared thirty modules, so the compiler was compiled twice

```
warning: `Y-compiler` (lib) generated  6 warnings
warning: `Y-compiler` (bin "Y") generated 62 warnings
warning: `Y-compiler` (bin "ysu_gpu_probe") generated 1 warning
```

`src/lib.rs` declares the modules `pub mod`; `src/main.rs` declared the same
thirty files again as **private** `mod`s of the `Y` binary. Two crates from one
set of sources. In the lib a `pub` item is reachable from outside and is not
flagged; in the bin it is dead unless `main.rs` itself calls it. So 62 of the
69 warnings said *"main.rs does not call this"* — a far weaker claim than
*"nothing uses this"* — and they are what made the blanket allows look
necessary. `CpuShapeDispatcher`, `OperatorFusionPass`, the fusion enum's four
variants, `pack_a_slot`, `ksplit_bands`, `CountedLoop::coq`, `render_ptx`: all
of them are used, by the lib's other modules or by the integration tests, and
every one appeared in that list.

`main.rs` uses `use y::{..}` now. The bin's dead-code count went **62 → 0**, and
its import list shrank from thirty modules to fourteen plus five under `zk` —
the other eleven had never been named by `main.rs` at all; they existed only so
that *other* modules of the duplicate crate could reach each other. The two
module lists had already drifted (`lib.rs` carries `fixed_exp`,
`exact_attention`, `c_api` and `zk_fuzz`; `main.rs` did not), which is the
ordinary end state of two descriptions of one thing.

**`PtxEnv is never constructed` was listed as a question rather than a finding,
and chasing it is what explains the rest.** `exact_attention::render_sched`
visibly calls `PtxEnv::default()`. rustc's note says why: *"has a derived impl
for the trait `Default`, but this is intentionally ignored during dead code
analysis."* A derived-trait construction does not count. Two of the 69 are that.

##### What the real census contains: nine items

With the duplicate compilation gone and all twenty-six allows removed, the two
builds report **8 warnings each, 9 distinct items** across both feature sets.
Sorted by last session's taxonomy:

| item | kind |
|---|---|
| `lexer::Lexer::matches_next` | 1 — leftover; every call site spells `peek()` + `advance()` |
| `LlvmEmitter::w` | 1 — the no-newline twin of `wln`; nothing emits a partial line |
| `cpu_gemm::b_sub1` | 1 — `x - 1` as raw IR, superseded by `IrBuilder::sub`, which is what `Ix::Sub` renders through |
| `circom_lower::resolve_signal_wire` | 1 — an unused wrapper over `resolve_signal_slot` |
| `QuantizationPass::{reg_f16, alloc_f16}` | 1 — a *scalar* f16 register class the packed form replaced |
| `bank_conflict::ThreadAccess::{thread_id, linear_byte_address}` | 1 — and evidence of a doc-vs-code gap, below |
| `ysu_gpu_probe`'s `cuMemcpyDtoH` | 1 — bound, never called; and the thread that led to the category-4 find |
| `PtxEmitter::unsupported_witness_op` | **3** — every caller is in `#[cfg(feature = "zk")] emit_witness_generator_ptx` |
| the probe's driver binding | **4** — a rule that already had one implementation |

`unsupported_witness_op` takes `#[cfg(feature = "zk")]` rather than
`#[allow(dead_code)]`: a `WitnessOp` does not exist without the ZK front end, so
the cfg is the truer statement *and* it stops the default build compiling code
no default build can reach.

##### Category 4: a second binding of the CUDA driver, following the opposite convention

The probe's unused `cuMemcpyDtoH` is one field of a **complete second copy of
the CUDA Driver API binding**, separate from `src/cuda_runtime.rs`. The two do
not agree. `cuda_runtime.rs` resolves the `_v2` symbol for `cuCtxCreate`,
`cuCtxDestroy`, `cuMemAlloc`, `cuMemFree`, `cuMemcpyHtoD`, `cuMemcpyDtoH` and
`cuEventDestroy`, with the reason written beside the macro:

> the unsuffixed `cuMemAlloc` symbol is the *legacy* 32-bit-size form, and
> calling it with a `usize` byte count silently truncates above 4GB

`ysu_gpu_probe.rs` resolved the **legacy** names for all six it uses. Confirmed
rather than assumed: `nm -D /usr/lib/libcuda.so.1` shows `cuMemAlloc` at
`0x39afb0` and `cuMemAlloc_v2` at `0x36a240` — two different functions. C code
never meets this because `cuda.h` `#define`s the plain names to the `_v2` ones;
a hand-written `dlsym` binding does.

It has never produced a wrong answer: the largest thing the probe allocates is
a 64MB array, four orders of magnitude below where the truncation begins. So
this is the `VnniExact::licenses` shape exactly — **a rule with one written-down
implementation, implemented differently a second time, latent because the live
inputs never reach the boundary** — and it surfaced from the same place, a
`never used` warning a blanket allow was hiding.

Both bindings now use the same `resolve_v2!`. `tests/cuda_driver_abi_versions.rs`
asserts they AGREE rather than re-deriving the table (a third copy is the bug,
not the fix), with the limit of an agreement assertion stated: flattening *both*
to the legacy form satisfies it, which is why a per-file floor sits beside it.

##### The bank-conflict prover's doc comment claimed a check it does not perform

`ThreadAccess` recorded `thread_id` and `linear_byte_address` and read neither.
That is not tidiness — the address is exactly what separates a hardware bank
**conflict** (two threads, one bank, *different* addresses — serialised) from a
**broadcast** (same address — free), and nothing looked at it. Measured on the
unswizzled F16 tile the pass is handed:

| `cols` | max threads/bank | distinct addresses in that bank | verdict |
|---|---|---|---|
| 16 | 4 | **4** | `Ok` |
| 32 | 8 | 8 | `Err` |
| 64 | 16 | 16 | `Err` |
| 128 | 16 | 16 | `Err` |

So at `cols = 16` a genuine 4-way serialisation is returned as proved
conflict-free, by a function whose doc comment said it "checks if any two
threads within the warp hit the same bank simultaneously". A second gap in the
same twenty lines: one thread is charged ONE bank where an `ldmatrix` row-fetch
is 16 bytes spanning FOUR — which the sibling `prove_ldmatrix_m8n8_x4` and
`ptx_emitter::max_bank_hits` both model and this one does not.

**Recorded, not repaired.** Its only consumer is the type checker's auto-swizzle
advisory, and every backend refuses the `SmemLayout` surface that advisory is
about, so tightening the predicate would change a message nothing acts on;
`ptx_emitter`'s own copy is the one that runs. What was fixed is the comment,
which now states what is decided and what is not.

##### The gate, and what it can and cannot see

`cargo build`'s warnings are not observable from inside a test binary, so
`tests/build_is_warning_free.rs` shells out — three configurations (default,
`--features zk`, and `-p y-gpu`, which the documented `cargo test` does not
build), each with **its own** `CARGO_TARGET_DIR` under `target/warning-gate`.
Separate directories are not tidiness: two feature sets of one package sharing a
target dir invalidate each other's units, which turns every run into two full
rebuilds — **12.3s against 0.11s** measured. A gate that is expensive gets
`#[ignore]`d and then never runs, which this repository has already paid for.
Cold cost is 18s; warm is 0.11s.

It sees every warning rustc emits for the root package's lib and four binaries,
in both feature sets, plus `crates/y-gpu`. It does **not** see warnings in
`tests/*.rs` (those targets are not built here — deliberate; a harness is
allowed scaffolding), anything under an item-level `#[allow]`, or a lint that is
off by default. Two source-level gates sit beside it for the parts it cannot
reach: no crate-level `#![allow(dead_code)]` anywhere in `src/`, and `main.rs`
may not re-declare a module `lib.rs` owns. Item-level `#[allow(dead_code)]`
stays legal — category 3 needs it, and `main`'s `counts()` is one.

##### The test count was inflated too, by exactly the same mechanism

`661 / 948` was the recorded headline. It is now `532 / 783`, and **nothing was
lost**: every module carrying a `#[cfg(test)] mod tests` was compiled into two
crates, so its unit tests were built and executed **twice** — once from the lib
and once from the `Y` binary. Measured against a `git worktree` at HEAD:

```
HEAD   lib 141 passed  +  bin "Y" 134 passed        (default)
HEAD   lib 177 passed  +  bin "Y" 170 passed        (--features zk)
now    lib 141 passed  +  bin "Y"   0 passed
now    lib 177 passed  +  bin "Y"   0 passed
```

and the arithmetic closes exactly: `532 - 5 (new gates) + 134 = 661`, and
`783 - 5 + 170 = 948`. A duplicated test run is not coverage.

(The worktree's own run **aborted at target 24** on `cargo test` — it had no
`target/release/Y`, and `cpu_gemm_end_to_end` says so by name. A reminder of why
the per-target sweep exists beside the aggregate, delivered by the baseline
measurement rather than by the subject.)

##### Mutation table — 8 probes, each `--test` target run separately

| probe | warning-free | census | binary-uses-lib | cuda-abi |
|---|---|---|---|---|
| G1 a crate-level `#![allow(dead_code)]` returns | ok | **FAIL** | ok | ok |
| G2 `main.rs` reverted to `mod` declarations | **FAIL** | ok | **FAIL** | ok |
| G3 a dead private `fn` added to the lib | **FAIL** | ok | ok | ok |
| G4a a build configuration that cannot run | **FAIL** (by name) | ok | ok | ok |
| G4b the same, with the non-vacuity controls removed | **ok** — the hazard | ok | ok | ok |
| G5 `unsupported_witness_op`'s `#[cfg]` removed | **FAIL** | ok | ok | ok |
| G6 the probe reverted to the legacy symbols | ok | ok | ok | **FAIL** (both) |
| G7 *control* — both bindings flattened to `resolve!` | ok | ok | ok | **FAIL** (floor only) |

**G7 is the standing limit of an agreement assertion, demonstrated rather than
asserted:** flattened together, the two bindings agree, and
`the_two_driver_bindings_agree...` passes. Only the per-file floor fails. A
consistent wrong choice for one symbol in both files remains uncaught, and
closing that needs a third copy of the table — which is the bug, not the fix.

**G1 is the most informative row, and it is a confirmation rather than a
catch.** Putting an allow back in `type_checker.rs` no longer resurrects a
single warning — because the module it silenced is no longer compiled a second
time. The twenty-six attributes were load-bearing only against noise that the
`main.rs` fix deleted.

**G4b is the null-metric hazard demonstrated.** With the `status.success()` and
`Finished`-line assertions removed, a configuration that never runs passes the
gate perfectly: a metric that counts bad things is passed by a component that
does nothing.

**Two probes were mis-aimed, and both misattributions were instructive.** G3
first used `fn __probe_dead_helper`, which produced no warning at all —
rustc's `dead_code` lint deliberately exempts identifiers beginning with `_`, so
an underscore-prefixed name is invisible to the census and the probe looked like
a survivor. And G6/G7 first failed the *warning* gate as well as the ABI one,
which read as coverage it does not have: the regex that reverted the calls left
an unused `macro_rules! resolve_v2` behind, and that has its own warning.
Removing the macro too gives the isolated rows above. **A mutation's side effects
are not the mutation.**

##### And the restore left cargo replaying the mutation

`tar xzf` restores the archive's **mtimes**, which are older than the mutation's,
so cargo decided nothing had changed and replayed the mutated build's
diagnostics — one probe appeared to fail for the right reason when it was
reading a stale unit. The stale-binary trap this repository already records,
wearing a timestamp instead of a build product. `touch` after every restore, and
the standing rule holds in both directions: verify the mutation is in the built
artifact, *and* that the restore is.

##### A separate observation, not acted on

The GPU probe's own output moved 22.66 → 4.29 cycles for FMA latency across the
change, which reads exactly like a regression. It is not: five consecutive runs
of the **same** binary gave 4.29, 43.89, 4.30, 4.09, 4.11 — a 10x run-to-run
spread with no warm-up discipline, and `.ysu_hw_profile` caches whatever the
first run happened to say. The clock-ramp lesson this repository already
records, in the tool that measures the machine. These numbers feed the analytic
cost-model fallback, so a wrong one is a suboptimal tile rather than an error.

**532 / 783 / 8** tests, zero failures across 123 per-target binaries in both
feature sets plus the aggregate; both builds warning-free; all four emitted LLVM
modules and every emitted attention PTX byte-for-byte unchanged; no committed
artifact touched.

#### The work queue itself had rotted: seven false items on the programme's own "not proved" lists · 2026-09-01

Step 12 of the checklist says to read each proof's "what this does NOT prove"
section back as the next increment's queue. It has produced two increments —
the temperature quantization came off the softmax bound's list, and the licence
predicate off the warning census. So the lists are load-bearing. **Audited all
seventeen: seven items were false, across four files, and every one of them
understated what is proved.**

| file | claim | why it is false |
|---|---|---|
| `ExactGemmChain.v` | "The third is not proved anywhere: the **kc-panel loop**" | **No such loop exists.** `emit_vnni_gemm_driver` passes the full `K` to both packers and `kpairs = (K+1)/2` to the micro-kernel; the only cut of the K axis is the thread K-split, so `kc` in every file of the series *is* K |
| `ExactGemmChain.v` | the tiling and band layers "are not joined to this one" | `ExactGemmWhole.the_threaded_gemm_holds_the_source_dot_products` joins them |
| `ExactGemmKsplit.v` | "THE MICRO-KERNEL IS NOT VERIFIED. Packing, the 2-D register tile, the masked tails and the periodic int32 → int64 flush are all assumed" | all four are proved — `ExactGemmPacking.v`, `ExactGemmRegisterTile.v`, `ExactGemmMicro.v`, composed in `ExactGemmChain.v` |
| `ExactGemmKsplit.v` | "The M and N partitioning … is not modelled" | `ExactGemmTiling.c_written_exactly_once` |
| `ExactGemmMicro.v` | the register tile and the masked tails are "ISA facts pinned by `tests/cpu_gemm_vnni_micro.rs`" | they are **proved**; only `vpdpwssd`'s semantics and the i32 half-order are TCB. This one overstates the trusted base by three proof files |
| `GridStrideSplit.v` | "this one's schedule is a PTX string template rather than an emitted expression … weaker than the byte-identity the `Ix` layer gives" | the schedule is an `Ix` (`sched_scores` / `sched_accum`), rendered to PTX and to Coq — contradicted by the file's own `Require AttentionSchedule` |
| `GridStrideSplit.v` | "The per-thread body — the integer exp, the Q0.28 weight, the int8 `V` load — is not modelled" | `SoftmaxErrorBound.v` sits **above** this file and bounds the first two |

**The kc-panel item had already been diagnosed and the diagnosis filed
elsewhere.** The session that wrote `ExactGemmWhole.v` read the emitter,
established there is no such loop, and recorded it in *that* file's header —
leaving this copy of the claim intact. So the queue has already cost one
session the time to re-derive a phantom.

##### The cause is structural, and the fix is derived rather than chosen

Each file states a **global** negative — a claim about what the programme
leaves open — and nothing updates a sibling's prose when a later file
discharges what it names.

The files whose lists stayed true are exactly the **dependency roots**:
`ExactGemmWhole.v`, `SoftmaxErrorBound.v`, `CountingSort.v`, `GemmBandSplit.v`,
`ZkControlFlow.v` — the five proofs nothing else `Require`s. That is not a
coincidence and it is not a choice: a capstone is rewritten every time the
chain below it closes, and a file with something above it *cannot* know what
the programme leaves open, because the file above it may have closed it.
`SoftmaxErrorBound.v`'s own list even carries "It was NOT priced when this file
was first written, and it is now."

So `only_the_capstone_states_a_global_negative` **derives** the capstones from
the `Require` graph rather than listing them — a hardcoded list would be a
third copy of the dependency graph, which is the mistake `ExactGemmSchedule.v`
exists to avoid. Outside a root, a negative must carry a scoping marker
("not modelled **HERE**; `X.v` proves it"), which makes the claim owned by the
file making it.

##### The gate found the seventh item, and I had dismissed it

`GridStrideSplit.v`'s per-thread-body claim was in the gate's first offender
list and I classified it a false positive — it reads like an honest global
statement about something no file covers. It is not: `SoftmaxErrorBound.v`
`Require`s that file and bounds the integer exp and the Q0.28 weight. Only the
int8 `V` load is genuinely unmodelled. **Checking before dismissing is the
whole of that finding**, and it is the same move as the licence gate's first
version, where the gate's *statement* was wrong and running it said so — here
the gate was right and my reading was wrong. Both directions happen; the
measurement settles it either way.

##### `"anywhere"` contains `"here"`

The first version of the gate exempted any line containing `here`, as a scoping
marker. `"not proved anywhere"` contains it. So the gate exempted **the exact
phrase it exists to catch**, and the mutation that restored the kc-panel
paragraph came back green — a survivor that was a real hole in a gate written
minutes earlier. The markers are delimited now (`" here "`, `" here,"`,
`" here."`, `"HERE"`), and P5 in the table below pins it. *A scoping word that
occurs inside the global claim is not a scoping word.*

##### The second gate, and the one live find it had

`every_path_a_proof_or_a_gate_cites_exists` resolves every repo-relative path
named in `proofs/*.v`, `tests/*.rs` and `src/*.rs` — **327 citations, one
stale**: `tests/zk_integer_ops.rs` pointed at `zk_divmod_soundness.rs`, renamed
to `zk_gadget_soundness.rs` when it grew past division. Cheap, and the class has
three prior instances here (three verification scripts named in `docs/` that were
never in the tree; `self_hosted/` rotting because it was named only in a doc; a
`zk_fuzz.rs` docstring citing a test file that did not exist).

**Anchor the start of the path.** The first sweep did not, so
`crates/y-gpu/tests/ptx_is_not_stale.rs` matched as `tests/ptx_is_not_stale.rs`
and was reported missing while existing — two of that sweep's three findings
were that. And the gate then fired on its own docstring, which cited the two
stale paths as examples: the self-reference trap already recorded for
`src.find("fn reference_bins")`, hit again the same afternoon.

##### Mutation table — 8 probes, each run against the whole gate file

| probe | capstone gate | path gate | the four pre-existing checks |
|---|---|---|---|
| P1 `ExactGemmChain.v`'s kc-panel claim restored | **FAIL** | ok | ok |
| P2 `GridStrideSplit.v`'s per-thread-body claim restored | **FAIL** | ok | ok |
| P3 `ExactGemmKsplit.v`'s "NOT VERIFIED" restored | **FAIL** | ok | ok |
| P4 the `Require` parse broken, so every file looks like a capstone | **FAIL** (the derived-capstone floor) | ok | ok |
| P5 the scoping markers un-delimited, with P1's claim restored | **ok** — the substring hole | ok | ok |
| P6 a citation to a file that does not exist | ok | **FAIL** | ok |
| P7 the path sweep resolves nothing | ok | **FAIL** (the floor) | ok |
| P8 *control* — `ExactGemmTiling.v`'s legitimate "not modelled **here**" | ok | ok | ok |

`coqc` stays green throughout, correctly: every edit is inside a comment.

##### What these gates cannot see

They catch the **form** of the failure — a proof with something above it making
a claim about the whole programme, and a citation to a file that no longer
exists. They cannot see a wrong claim phrased in words they do not know. Prose
staleness is not mechanically decidable, which is precisely why
`ExactGemmSchedule.v` is *generated* rather than checked; this is the reachable
part, and the unreachable part is stated in the gate rather than implied.

##### And I deleted nineteen tracked files

The mutation harness needed a `restore2.sh` and a `state2.tgz`. The shell here
is fish, and defining a function named `g` collided with an alias — a **parse**
error, which aborts the whole block before executing anything, so neither the
tarball nor the restore script was created. The next command then ran
`rm -rf proofs tests/proofs_are_checked.rs tests/zk_integer_ops.rs` and
extracted a backup that did not exist.

Recovered with `git checkout` — all nineteen were tracked and clean at HEAD, so
only the day's uncommitted edits were lost and they were re-applied from the
transcript. The restore script **verifies the archive before deleting anything**
now, and the check is a real one: the failed `tar czf` had left a 45-byte empty
archive behind, so a size or entry-count check catches exactly this. Two
process rules, both already in this file in other forms, and both violated
within an hour of each other: *verify the restore took effect*, and now *verify
the backup exists before you rely on it*.

**534 / 785 / 8** tests, zero failures across 122 per-target binaries in both
feature sets plus the aggregates; both builds warning-free; `coqc` clean over
all seventeen proofs with no axioms and nothing admitted; no `src/` change, so
every emitted module is byte-identical by construction.

## 5. End goal

> **STATUS, 2026-08-31.** The end goal below is reached for **one kernel on one
> backend**: `Y gemm.ysu --emit-llvm` substitutes the exact `vpdpwssd` GEMM and
> writes a `.v` beside the `.ll` that an auditor can check with `coqc` and the
> proofs, without trusting the compiler. What is *not* reached is the scope —
> one kernel family, one backend, and three things the certificate's own header
> declares outside it: `vpdpwssd`'s semantics (a definition pinned on hardware),
> the loop nest's structure (partly extracted, partly gated), and anything at
> all about LLVM or machine code. The repeatable method is in
> [The process](verified_kernel_process.md).

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

## 7. The first move — DONE, 2026-08-12

> **This section is kept as written because the measurement it asked for is the
> reason everything after it exists.** The four lines are deleted, the packed
> kernel accumulates exactly, and the number that "decides whether any of the
> rest is worth doing" came back **1.88x in favour of exactness** — the exact
> `vpdpwssd` GEMM is faster than the f32 one it replaces, so exactness trades
> range rather than speed. Read the rest of this section as the question, and
> §"Phase 0 status" above as the answer.

Delete four lines and find out what happens.

`src/cpu_gemm.rs:289` refuses a `@ZeroDrift` accumulator on the grounds that
substituting the packed kernel would discard the exactness guarantee. That
reasoning is correct *today*, because the packed kernel accumulates in `f32`.

Make the packed kernel accumulate exactly instead, and the refusal becomes
unnecessary. Then measure what it costs — **that number decides whether any of
the rest is worth doing.**
