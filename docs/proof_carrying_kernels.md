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
