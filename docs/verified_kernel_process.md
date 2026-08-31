# The process: taking a kernel from *fast* to *verified*

This is the method, written down after sixteen proof files and about twenty
gates. It is not a plan — every step here has been executed, and the parts that
did not work are recorded as such.

The subject is `src/cpu_gemm.rs`'s exact `vpdpwssd` GEMM: a tiled, packed,
K-split, multi-threaded kernel that is **bit-identical** to the naive triple
loop it replaces. Two other kernels have been through parts of it (the f32 GEMM,
the exact int8 attention PTX), and the differences are noted where they matter.

Current state: **sixteen `.v` files, ~265 theorems, no axioms, nothing
admitted**, all checked by `cargo test`. The counts are approximate on purpose:
an exact one goes stale every session, and a gate on it would fail on every
proof added.

---

## 0. The precondition, and it is not negotiable

**The kernel must be exact.**

Everything downstream rests on this. Floating-point addition is not associative,
so a tiled reduction provably does *not* equal the naive one — the best that can
be said is that they are close, and "close" needs an error model, an error
budget, and a proof about both. Integer and fixed-point addition **are**
associative and commutative, so the relationship between the optimized kernel
and the specification is an **equality**, and an equality is what a proof
assistant is good at.

This is stated as a theorem rather than believed:

```coq
(* GemmBandSplit.v *)
Theorem rounding_breaks_the_proportional_split_too : ...
Theorem exact_survives_the_same_split : ...
```

Same `f`, same `K = 201`, same `nthr = 2`, differing only in the accumulate. The
rounded one answers **1100 against its own reference's 1000** — it does not lose
precision, it *disagrees*. The exact one answers 1200 either way.

Finding that counterexample took a brute-force search: three plausible rounding
models agreed on every small input tried. **"Obviously non-associative" is not
enough** — the counterexample needs a large term the small ones vanish against.

The corollary is that the f32 GEMM which ships for ordinary Y programs
**provably cannot** get this treatment, and saying so is part of the process.

The cost of exactness was measured before anything was built: exact VNNI is
**1.88× faster** than the f32 path. Exactness here trades *range*, not speed.

---

## 1. Prove the SCHEDULE, not the kernel

The instinct is to verify the arithmetic. That is the expensive half and it is
not where the bugs are.

A kernel like this is a **decomposition plus a body**. The body — one
`vpdpwssd`, one broadcast, one flush — is a handful of instructions pinned on
real hardware by a differential test. The decomposition is where correctness
actually lives:

- which K values each thread takes,
- which rectangle of C each tile owns,
- which slot of the packed panel holds which source element,
- which accumulator lane reads it,
- how many k-pairs may accumulate in int32 before widening.

Every bug this programme has found was in that layer. From the project's own
history: *"twelve address computations in the CPU GEMM were correct only because
`lda == K` made stride and extent the same number."*

**Practical rule: make every stride differ from its extent in any GEMM test.**
Equality is a coincidence that hides address bugs, and it hid a live heap
overflow here until a proof forced the hypothesis `N <= ldc` to be *stated*.

### The obligation families, and the split is by CONSEQUENCE as much as by shape

```coq
(* Decomposition.v *)
Theorem contiguous_exact    (* consecutive parts: needs ASSOCIATIVITY only *)
Theorem decomposition_exact (* arbitrary owner map: needs COMMUTATIVITY *)

(* CountingSort.v *)
Theorem slot_injective      (* the parts TILE a buffer rather than folding *)
Theorem slot_onto
```

Seven proof files instantiate the first two. A new decomposition costs a
**bridge lemma plus three edge facts** — measured at 6 tactic lines against a
hand-written straw-man's 10, with the straw-man compiled in the same file so
the comparison was real rather than estimated. (It was first published as
7-against-11, from a line counter that mis-parsed `Proof. … Qed.` on one line
— the idiom the change itself introduced. **When a measuring instrument is
found broken, re-derive every number it produced.**)

**The honest headline is that the schema does not save lines.** Total tactic
lines went *up* (184 → 188 plus 42 shared) because each kernel gained a bridge
saying its own fold is the schema's fold. What it buys is that the seven files
cannot drift from each other, and that the marginal cost of the *next*
decomposition is small.

A third kind arrived later, and the difference is not the shape of the
decomposition but what is *wanted* from it. The five reduction instantiations
fold their parts into a value. A counting sort's parts tile a destination
array, and the question is whether every slot is written exactly once.
`Decomposition`'s `widths_cover_the_extent` says the widths add up **in
aggregate**, which is strictly weaker: a decomposition writing one slot twice
and another never has exactly the right total width. The bijection is ~90
lines and is not derivable from the fold theorems — but the *edges* are the
same schema, hypotheses handed over unchanged, including data-dependent ones
built from a histogram. Composing two levels of decomposition then cost one
hypothesis and nothing else.

If a fourth kind appears, expect it to reuse the edges and to need its own
consequence.

Three findings worth transferring:

- **The two fold theorems are not derivable from each other.** The general form
  would give the contiguous one only by inverting the edge function, and would
  then demand commutativity the contiguous case does not need — a *weaker*
  result presented as a simpler one.
- **Two obligations that looked different turned out to be the same family.**
  The int32 flush interval is an *overflow budget*; the output tiling is a
  *memory partition*; the code's own comments warn against confusing them. They
  are right about the purpose and wrong about the shape — both are
  `t |-> min(t*X, ext)`, and the emitter hides it by computing one as an END
  and the other as a WIDTH. A third member turned up later, from a counting
  sort rather than a GEMM: a scatter thread owns a clamped run of histogram
  chunks.
- **Two `assert_eq!` in a shipped scatter turned out to BE the proof's
  hypotheses**, and a third's *sufficiency* was a sentence in a code comment
  ("this is a PROOF rather than a spot check") that nothing checked. Read the
  runtime assertions before inventing hypotheses: a defensive check somebody
  wrote is often the precondition, already stated in the only place it was
  ever going to be.

---

## 2. Remove the second description, do not prove a translator correct

The naive way to connect a proof to a compiler is to give the IR a semantics and
prove the emitter refines it. That is a multi-year project on its own and it
would eat this one.

The cheap alternative: **there is only one description, and both the emitter and
the proof are rendered from it.**

```rust
pub enum Ix { Val(&'static str), Lit(usize), Sub(..), Add(..), Mul(..),
              Min(..), Div(..), Mod(..), SelLt(..) }

impl Ix {
    pub fn emit(&self, b: &mut IrBuilder, env: &..) -> String  // → LLVM
    pub fn coq(&self, bind: &..) -> String                     // → Coq
    pub fn eval(&self, bind: &..) -> i64                       // → values, for tests
}
```

`CountedLoop` does the same for a loop's iteration space, rendering `_visit` and
`_trips` definitions. `proofs/ExactGemmSchedule.v` is **generated** from
`src/cpu_gemm.rs`'s own constants and committed; a byte-identity gate fails the
build on any divergence.

### The generator must LINK, not parse

A `tools/` script recovering `VNNI_MR` with a regex would be a *fourth* copy of
the value, living in the generator. That is the bug, not the fix. The input here
is our own Rust, so it can be `use`d. There is no `build.rs`, and a fifth
`[[bin]]` is a permanent user surface bought for a file regenerated twice a
year — so the generator is a `#[test]` with `Y_REWRITE_SCHEDULE_PROOF=1`.

(`tools/extract_poseidon.py` parses, because *its* input is foreign circomlib.
That is the distinction.)

### One implementation, and it must be the one that ships

Removing the second description is not finished when both copies agree — it is
finished when there is one. A GPU MSM module here was forked between the crate
that ships and a copy under `tests/`, and **every measured improvement landed
in the test copy**: a parallel prefix worth 23.6 → 3.3 ms, a scatter thread
count worth 2.2x, a launch geometry worth 1.43x, and the post-condition that
proves the scatter writes each slot once. The shipped one kept the version from
before all of them.

It also quietly invalidated a proof. The counting-sort theorems were tied to
the test copy by a behavioural test, so the tie covered code the library does
not run — the same gap as a certificate that describes the repository rather
than the output, one layer further out. **Check which copy your tie is
attached to.**

### Byte-identity is the evidence a refactor is faithful

Capture the emitted modules **before** touching the emitter. Every extraction
step in this programme has left all four emitted LLVM modules byte-for-byte
unchanged, and that is the whole argument that the description is what the
compiler was already doing.

The rule survives a **second backend**, which is the evidence the layer is not
shaped around one code generator. The same `Ix` renders to PTX for the
exact-attention kernel — a target with a different instruction set, a different
register discipline (special registers must be `mov`'d into ordinary ones
before use) and a fused multiply-add — and reproduced three hand-written
sequences **instruction for instruction**, register numbering and `mov`
placement included. Only comments moved.

That was not arranged. A renderer that materialises a special register at
**first use** and fuses `a*b + c` into one instruction emits what a person
writing PTX by hand emits, because both are following the target's own grain.
**If an extracted expression does not reproduce the hand-written sequence, the
first hypothesis is that the renderer is fighting the target, not that the old
code was arbitrary.**

Where the emitted text must genuinely change — a comment moving, say — say so
and fall back to comparing the **instruction stream** with comments and blank
lines stripped. Claiming byte-identity you do not have is worse than the weaker
check honestly stated.

It also constrains the design. When a loop's bound is a value the driver has
already computed, `loop_begin_over` names the existing register rather than
re-emitting the expression — re-emitting would add instructions the compiler
does not have, and there is then nothing to check the refactor against.

### The standing limit, stated because it is real

A shared description removes drift **between consumers**. It does not make the
description right. Move the emitter away from the description and the gate
fires; move the *description* and both consumers move with it, silently.

What catches a coherent-but-wrong description is a theorem tying it to something
**independently fixed**. The scratch-zeroing loop is the worked example: one
trip short and one trip long both regenerate a self-consistent `.v`, and what
caught them was

```coq
Theorem the_zeroing_loop_covers_exactly_the_tile :
  SCH.zero_tile_trips = (RT.MR * RT.NR)%nat.
```

against the tile geometry another file fixes. **Prefer `=` to `<=` whenever both
directions are unsafe.**

### A GENERATED SIGNATURE ABSORBS A RELABELLING

Sharpest instance of the limit above, and it survived a green sweep.

The attention schedule's Coq definitions have their **parameter list derived**
from the emitted expression's free names — deliberately, so a schedule that
gains or loses an index changes the signature and the hand-written theorems
stop compiling. Swapping two radices in the emitter and regenerating then left
everything green: the parameters were renamed in step, every theorem applies
the definition **positionally**, and the result is the same function under
different labels. No proof can see a relabelling.

The fix is a theorem that is **half generated and half fixed** — binders from
the expression, right-hand side from the author:

```coq
Theorem the_worker_index_is_a_mixed_radix_number :
  forall <derived binders> : nat,
    worker_accum <derived binders>
      = ctaid_z * (nctaid_x * ntid_x) + ctaid_x * ntid_x + tid_x.
Proof. intros. unfold worker_accum. ring. Qed.
```

Swap the radices and the binder list moves while the equation does not, so
`ring` fails. Drop an extent and it is caught harder: the binder disappears
while the right-hand side still names it, so the reference is unbound.

**The general rule: if a generated artifact's *names* carry meaning, something
must pin the names, because a prover compares terms and not labels.** And note
which layer catches it — the byte-identity gate passes, correctly, because
description and proof agree perfectly. `coqc` is the only thing left.

---

## 3. Tie the proof to the running code, and know what each tie can see

Five kinds of check, each blind to something the next one catches. The
taxonomy matters more than any individual gate.

| check | what it sees | what it is blind to |
|---|---|---|
| **Correctness suite** (run it, compare against an independent reference) | anything that changes the answer | schedules; out-of-bounds writes into memory the process owns; over-allocation |
| **`coqc` + `Print Assumptions`** | unsound proofs, axioms | a theorem stating something other than what it claims; `Theorem x : True` |
| **Byte-identity of emitted modules** | the emitter drifting from the description | the description itself moving |
| **Schedule gate** (read the emitted IR; compare against the rendered expression) | same-value divergences — different instructions, identical result | anything outside the described expressions |
| **Behavioural schedule gate** (excise callees, record the call sequence) | wrong order, wrong count, re-packing | anything inside a callee |

Three results from this repository, each caught by exactly one row:

- **An over-allocation.** `tile_count` computed with `T` instead of `T-1`
  allocates a buffer that is never smaller than needed. No wrong answer, no
  crash, no observable symptom. Caught by the schedule gate alone.
- **A same-value operand swap.** `min(a,b)` emitted as `min(b,a)` — identical
  result, different instructions. Five correctness suites pass. Caught by the
  schedule gate alone; five separate instances so far.
- **A write past the end of a scratch buffer.** The zeroing loop one trip long.
  Every correctness suite passes. Caught by the Coq equality alone.
- **A relabelled definition.** Two radices swapped in the emitter *and*
  regenerated, so the description and the proof agree perfectly and the
  byte-identity row passes. Caught by `coqc` alone, and only once the role
  theorem above existed.
- **A stride register holding a partial product.** The rendered block is
  correct and present; the loop reads the wrong register out of it. Nothing in
  the rendering can see that. Caught by the dataflow gate alone.

And the one that should be read twice: an **unclamped fold-back loop**, a
genuine out-of-bounds write past the last row of C, is **not observed by two of
the four correctness suites** — including the one running `M = 53` against a
6-row tile, the most ragged shape in the repo. The overrun lands in memory the
process already owns and the answer stays correct.

### Rendering says what the code computes; a dataflow gate says the code reads it

Once a schedule is rendered from one description, it is tempting to delete the
text-reading gate it replaced. Do not, without running the mutation. Rendering
establishes that the emitted block is the right arithmetic; it establishes
nothing about whether the loop **consumes** it. Swapping two result registers
in the attention kernel's worker count leaves the block correct and present
while the loop strides by the partial product — a real launch-invariance bug,
invisible to the renderer and to the proof, caught by the walker alone.

The two are complementary, not redundant: one is the decorative-codegen
question (*is this block used?*) and the other is the drift question (*is this
block right?*).

### Where a tie needs an anchor, anchor on what the artifact SAYS

Recovering a loop from emitted text by pattern-matching its shape matched the
wrong loop twice. Both times the fix was to anchor on something the kernel
states about itself:

- a **named marker comment** (`// [Y SEQUENCE REDUCTION]`), the device the PTX
  backend already uses for `[Y PAGED DECODE ATTENTION]`;
- the loop's own **back-edge**, which is a property of loops rather than a shape
  that happens to be unique today.

And follow **reaching** definitions, not first textual matches: register names
are per-function, so the first `%g12` in a module may belong to a different
function entirely. That error compared a driver's fold-back against a packer's
address arithmetic.

The same rule governs the *build configuration*, and getting it wrong makes a
gate unpassable rather than wrong. `crates/y-gpu` embeds five compiled kernels
with `include_str!`, and its freshness test recompiled each one **on the build
machine** and demanded byte identity. But those artifacts are deliberately
portable — committed at `.target sm_80` — while a local compile probes whatever
card is present and says `sm_89`. Two gates encoding contradictory
requirements, and freshness is the one that has to give: a compiler that probes
the local machine bakes that machine into its output.

So the gate asks the artifact **which target it claims** and pins a
`.ysu_hw_profile` to that before recompiling. Whether the claimed target is
legitimate is a different question, answered by a different gate. Splitting
them that way is what let each of six mutations be caught by exactly one:
a stale kernel by freshness, a machine-specific target by portability, a
mis-wired `include_str!` by the crate's own wiring check.

**A dispatcher disarms tests silently, and the check is not "did it pass" but
"which path did it take".** A GPU MSM library's own prover tests ran 2,048 and
4,096 constraints, both below the size at which dispatch chooses the GPU, so
every one of them exercised the CPU fallback — they would have passed with the
accelerated path deleted. Combined with a forked implementation, that made a
corrupted-output bug in the shipped binner invisible to *every* test in the
repository. Force the path under test and **assert you got it**; a test that
can silently become a test of the fallback is a test of the fallback.

**And a red test nobody runs is not a gate.** That freshness test had been
failing for a week. `cargo test` at a workspace root with a root package builds
*that package only*; the crate's tests need `-p y-gpu` or `--workspace`, and
neither was in the documented commands. Four of the five embedded kernels were
stale by an entire optimisation series — `bn254_fr_mul_fast` shipped at 2,112
lines with zero carry-chain instructions against the current 1,255. Put the
check where the documented command will run it, even if that means checking the
files from outside the crate that embeds them.

---

## 4. Discharge the finite obligations by exhaustion, the unbounded ones with a solver

The licence for the exact GEMM is `2 · Fl · m² ≤ i32::MAX`, where `m` is the
operand magnitude. `m` lives in int16, so its **entire domain is 32,768 values**
— calling the real function over every input it can receive is *complete*, not
sampled, and closes the model-versus-code gap a solver would leave open.

Z3 earns its place only where an axis is genuinely unbounded — here `K`, which
gets stated integer arithmetic and a written-down ceiling (~5.5e11) that existed
nowhere before and **moves if the flush interval or either width changes**.

A proof over `Z` plus an exhaustive check over the finite domain is stronger
than either alone, and only because both are present: the proof says nothing
about range or reduction, and the exhaustion says nothing about `K`.

**The boundary is one unit wide.** At the default interval, `m = 4095` fits with
1,048,447 to spare and `m = 4096` exceeds `i32::MAX` by exactly one. An
off-by-one in the `floor(sqrt(..))` derivation is invisible to any sampled test.

### DECOMPOSE THE QUERY: THE SOLVER LIMIT WAS A TIMEOUT, NOT AN UNDECIDABILITY

`CLAUDE.md` recorded Z3 *failing* on exactly this shape: "is there any
satisfying assignment with `out != (a < b)`" for the 32-bit comparison gadget
"did **not** finish, because 254-bit modular arithmetic over `Int` defeats the
solver". That is easy to read as "Z3 does not scale to this backend", and that
reading is wrong. Measured, same question, same solver, in
`tests/zk_gadget_soundness.rs`:

| posing                                            | result | time      |
|---------------------------------------------------|--------|-----------|
| whole gadget in the field, nothing bounded        | unsat  | 560,498ms |
| bounded, range checks as separately proved facts  | unsat  |      15ms |

**37,000x — and the slow one terminates.** So the earlier failure was a
timeout. The fix is not a better solver, it is *decomposition*: prove the bound
first, then state the property over the bounded domain.

> A field question is intractable in practice; the same question restricted by
> the range checks the circuit already enforces is small. Ask the bounded one,
> and **make the bound the first thing you prove** — that is what keeps the
> bounded posing faithful rather than merely weaker.

The same file demonstrates it on itself twice over. "Is there a bit assignment
with `(mod (sum 2^i b_i) p) = p - 1`" — the documented claim that a negative
operand is *unprovable rather than answered* — does not return in 120 seconds.
Split into "the sum is bounded" (5ms) and "a bounded value is not `p - 1`"
(4ms), it is **13,000x** faster. A 254-bit modulus over a symbolic sum is the
thing to decompose away; a modulus alone is not the problem, and a `mod` whose
argument is already bounded simplifies for free.

Two consequences worth carrying:

- **A model is not a tie, and the mutation table has to say which is which.**
  The comparison, bitwise and shift queries *restate* their gadget's arithmetic,
  so they check that the arithmetic is right and not that the emitter uses it:
  an off-by-one in the shift index map, and swapped `|`/`^` scale factors, both
  pass that file and fail the behavioural `zk_integer_ops`. What closes the gap
  cheaply is a **structural count** — `emit_num2bits(x, k)` is `k + 1`
  constraints, so the model predicts 101, 99 and 34 exactly, and a dropped
  operand range check or a difference decomposed one bit short fails *with the
  derivation attached* instead of being re-pinned as a cost change.
- **Sharing a constant with the code under test is correct when the solver is
  the independent oracle.** `the_dividend_bound_is_exactly_the_supremum` reads
  the emitter's own `max_representable_dividend()`. Moving that constant *up*
  makes the achievability query unsat; moving it *down* makes the supremum query
  sat. Both directions fail, so this is a two-sided pin rather than the
  both-sides-move-together silence that a generated description can produce.
- **Ask whether two bounds are independent, because they usually are not.** The
  fold path for `/` grew two guards — a quotient-range check and a dividend
  bound. Z3 says the first *subsumes* the second whenever both operands are
  constant, and that the converse fails. So the dividend bound's only live case
  is a constant dividend with a **variable** divisor, and the test that exercises
  it has to be written for that case specifically. A conjunction of plausible
  bounds is not a set of independent obligations.

`unknown` and a timeout are treated as **failures** in that file, not skips: the
whole claim being made is that these queries are decidable, so a solver that
cannot decide one has refuted the claim.

---

## 5. Emit the certificate

`proofs/` is checked once, at build time, against the shipped schedule. Its
theorems quantify over `M`, `N`, `K` and `nthr`, so every shape is covered — but
a user compiling their own kernel gets a fast binary and no artifact.
"Proof-carrying" then describes the repository, not the output.

`--emit-llvm` now writes `<stem>_certificate.v` beside the `.ll` whenever the
exact kernel is **substituted**, instantiating the capstone theorem at this
compilation's flush interval and operand bound:

```coq
Definition Fl : nat := 64.
Definition m  : Z   := 1024.

Theorem the_licence_holds : 2 * Z.of_nat Fl * m * m <= ExactGemmMicro.I32MAX.

Theorem this_kernel_computes_the_source_dot_products :
  forall A B M N K nthr r c,
    (forall i k, Z.abs (A i k) <= m) -> (forall k j, Z.abs (B k j) <= m) ->
    (0 < nthr)%nat -> (r < M)%nat -> (c < N)%nat ->
    W.thread_sum A B M N K Fl nthr r c nthr = PK.sum_k (fun k => A r k * B k c) K.
```

### It is not paperwork, and the reason is worth copying

The only hypothesis that depends on the program is the licence, and the
**compiler decides it in floating point** — a `sqrt` and a `floor` on `f64`. The
certificate states it over `Z` and hands it to `coqc`, which has no floats. Two
derivations of one obligation, by two tools, and a gate that checks they agree
at four intervals, at the limit and one above it.

### `coqc` accepting a generated proof is necessary and NOT sufficient

Two mutations survived a fresh, all-green eight-test file:

- **The licence with one factor of `m` dropped.** Coq compares propositions up
  to *conversion*: at a licensed numeral both `2·Fl·m ≤ I32MAX` and
  `2·Fl·m² ≤ I32MAX` reduce to `Lt <> Gt`. The wrong statement type-checks, its
  use site accepts it by conversion too, and the certificate refuses **exactly**
  the magnitudes the correct one refuses. No `coqc` run can separate them. What
  changes is the claim a human auditing the artifact reads — which, for a
  certificate, is the point of the artifact.
- **`Theorem the_certificate_is_not_vacuous : True.`** It compiles, reports
  `Closed under the global context`, and satisfies a *count* of how many such
  reports appeared.

So gate the statement's **text**: assert the certificate states the obligation
it is about and evaluates the model it claims to. The committed proofs already
had `every_proof_has_a_content_control` for exactly this; the generated one had
no counterpart.

---

## 6. Mutation-test everything, and sort the survivors

Every gate is expected to fail when the mechanism it guards is removed. Run each
`--test` target **separately** — `cargo test` aborts the remaining binaries
after one fails, which has silently left the important target unmeasured.

Survivors sort into three kinds, and the sorting is where the learning is:

| kind | example | action |
|---|---|---|
| **Real test hole** | dropping the bias-undo in one of two entries passed, because the shared temperature made the error `≡ 0 mod 2^32` | fix the test, at a parameter that separates them |
| **Mis-aimed mutation** | `s.replace(old, new, 1)` hit a *docstring* forty lines above the code; a mutation whose `cargo build` failed left the previous binary in place and the driver reported a clean run | re-aim; **assert the build succeeded and grep the generated artifact before reading any result** |
| **Confirmation of a design claim** | a hand-written *identical* copy of an extracted expression passes — and should, because the gate checks the property, not the plumbing | keep, and say so in the test |

There is one survivor that is a real hole and **cannot be closed by testing
harder**: replacing a differential's oracle with the implementation it is the
oracle for. Every comparison then passes, and every control passes with it,
because they are computed from the now-identical output. No behavioural check
can separate "the oracle is right" from "the oracle IS the implementation" —
the two agree exactly when the implementation is correct, which is the property
under test. Every differential in this repository has that hole. What
distinguishes them is structural, so gate it structurally: read the test's own
source and require the reference not to call the thing it is checking.

Writing that gate has its own trap. Anchoring on `src.find("fn reference_bins")`
matches the **string literal on the gate's own line**, so the extraction
returns the gate's body and it then fails for the wrong reason. Anchor on the
definition (`"\nfn name("`), and probe it with the mutation it exists to catch
before believing it.

A second survivor of that kind, found the same way: **a proof cannot see a
relabelling**, because a prover compares terms and not names. If a generated
definition's parameter list is derived, swapping two roles renames the
parameters in step and every positionally-applied theorem stays true. The fix
is above in §2; the point here is that it was found by a mutation that
regenerated *both* sides on purpose. **Mutate the description, not only the
emitter** — the mutations that move one side are the easy ones.

Two recurring traps:

- **A mutation table row says which tests went red, not which tests detected the
  thing.** One row here read "caught by 6 model suites"; the real story was that
  seven test harnesses each carried their own `#define MR 6`, so moving the
  constant made each test disagree with *itself*. Reading that one row properly
  was worth a register-file constraint nobody had written down.
- **Controls must state the interesting case as a proposition, not merely
  exhibit it.** Twice in two increments, a control that displayed a ragged tile
  or a digit collision stayed green when the fixture moved to a boring value.
  Refute the *weakened theorem* instead — no choice of witness can satisfy that
  vacuously.

---

## 7. Say what is not covered, in the artifact itself

Every proof file in this repository carries its own "what this does not claim"
section, and the emitted certificate repeats it in its header. A certificate
that overstates its scope is worse than none.

For the exact GEMM, three things are outside:

- **`vpdpwssd`'s semantics and the little-endian order of an i32's two halves
  are definitions.** No proof over `Z` can supply them; `cpu_gemm_vnni_micro.rs`
  pins them on the real instruction. This is the TCB boundary and it is
  deliberate.
- **The loop nest is partly extracted and partly modelled.** Every loop in the
  exact-GEMM driver now has a description rendered to both consumers; *which*
  loops exist, in what order, and what they call is still hand-written and is
  covered by gates rather than by extraction.
- **Nothing here is a statement about LLVM, `clang`, or machine code.** There is
  no IR semantics in this repository and there is not going to be one.

---

## Checklist

1. **Is the kernel exact?** If not, stop — state that it cannot be verified this
   way and why.
2. **Measure the cost of exactness** before building anything on it.
3. **Identify the decomposition**, separately from the body. Which obligation
   family is it — contiguous, or arbitrary-owner?
4. **Instantiate the schema**; write the bridge lemma from the kernel's own fold.
5. **Extract every schedule number** into one description rendered to both the
   emitter and the proof. Capture byte-identity of the emitted modules first.
6. **Find the hypotheses hiding inside definitions.** `fun _ _ => 0`,
   `N <= ldc`, `0 < nthr` — each is a claim about the emitter, and each has been
   wrong at least once here.
7. **Tie each one to something independently fixed**, with `=` where both
   directions are unsafe.
8. **Exhaust the finite obligations**; use a solver only on the unbounded axes,
   and write the ceiling down.
9. **Emit the certificate**, and gate its *text*, not just that `coqc` accepts it.
10. **Mutation-test**, one target at a time, and sort every survivor.
11. **Write down what is not covered**, in the artifact.

---

## Further reading

- [Proof-carrying kernels](proof_carrying_kernels.md) — the roadmap and the
  chronological log, including the measurements that decided each step
- `proofs/` — the sixteen files, each with its own header stating scope
- `tests/proofs_are_checked.rs` — the gate that runs them, with content controls
- `tests/exact_gemm_schedule_proof.rs` — the generator and the schedule ties
- `tests/exact_gemm_certificate.rs` — the emitted certificate
- `proofs/CountingSort.v` and `tests/msm_counting_sort_model.rs` — the
  placement obligation, and the fourth kernel
- `proofs/AttentionSchedule.v` and `tests/exact_attention_schedule.rs` — the
  same extraction layer rendering to a second backend, and the relabelling
  hole it exposed
- `tests/zk_gadget_soundness.rs` — a solver question this repo had recorded as
  defeating Z3, decided in 15ms by decomposing it; plus decomposition
  uniqueness, which is the obligation the bitwise and shift gadgets rest on
