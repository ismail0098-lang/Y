# Bit-Identical Decode

**Deterministic inference — status report, 2026-08-18**

Model: Qwen2.5-0.5B-Instruct · GPU: RTX 4070 Ti SUPER · Stack: PyTorch + Triton
prototype (not the compiler's own PTX kernels)

A quantized transformer whose output does not depend on who else is in the
batch — now measured against its accuracy cost, its speed cost, and the things
it still cannot claim.

---

## The short answer

Every reduction whose order changes with batch shape has been made integer, and
integer addition is associative — so tile shape, K-split, atomic completion
order and batch size cannot change the answer. That is the whole mechanism.

| | result | |
|---|---|---|
| **Determinism** | **0 / 16** batch compositions changed the output | stock changed on 16/16 |
| **Accuracy vs bf16** | **+0.12%** wikitext-2 perplexity | task accuracy is a statistical null |
| **Decode speed** | **1.09x** slower than the bf16 baseline | both arms compiled, batch 32 |

---

## Finding 01 — determinism survives production-shaped batching

The easy version of this test decodes one prompt repeated *N* times, so every
sequence has the same length. Production never looks like that: continuous
batching mixes prompt lengths, pads to the longest, and reshuffles as sequences
join and finish. Three real dependencies on batch shape had to survive it — the
softmax weight is split into a different number of digits when the padded key
length changes, the GEMM tiles differently at a different row count, and padded
key positions must contribute exactly zero.

| test | what varies | stock bf16 | exact int8 |
|---|---|---:|---:|
| Uniform batch | size 1 / 8 / 32 | 3 / 3 | **0 / 3** |
| Ragged batch | size, neighbour length, position | 16 / 16 | **0 / 16** |
| Static KV cache | cache implementation | 3 / 3 | **0 / 3** |
| GEMM schedule | 11 tile / K-split configs | — | **0 / 11** |

**The control fires.** A test both arms pass is measuring the harness. The stock
column is the control, and it had to be earned: at 24 generated tokens stock was
*invariant too*, because a bf16 reduction-order delta needs room to flip a greedy
argmax. The test runs 160 tokens on prompts chosen for a small top-2 margin.

---

## Finding 02 — the accuracy cost went from 6.1% to 0.12%, and the last step reversed the roadmap

Perplexity is the sensitive metric here; multiple-choice accuracy was a null
throughout (net +40 of 3000 items, read as unchanged).

| configuration | perplexity | vs stock |
|---|---:|---:|
| fp32 reference | 13.863 | −0.06% |
| Stock bf16 + SDPA | 13.871 | — |
| Exact, first working version | 14.714 | +6.1% |
| Exact, wider K cache | 14.183 | +2.2% |
| **Exact, wider activations** | **13.887** | **+0.12%** |
| 4-bit control | 40.035 | +189% |

*wikitext-2, 250,000 tokens, sliding window*

**The textbook lever was the wrong lever.** The remaining 2.2% was labelled "the
linears", and the standard fix for that is group-wise weight scales. Measuring
the ceiling before building it — fake quantization in fp32, no kernel, accuracy
only — showed weight quantization costs **−0.12%**, i.e. nothing at all. The
entire gap was the *activations*. Group-wise weight scales recover **6.2%** of
it; one extra bit of activation width recovers **102%**.

**And the fix that won does not fight the invariant.** Group scales would have
split each reduction into partial sums recombined with *float* weights —
order-dependent again, the exact thing the project exists to prevent. Widening
the activation needs no such rescue: carry it as two int8 digits, run both
against the same weight tile, combine in int32. Two matmul instructions, one
weight load, integer end to end. It costs 5.9% of decode throughput.

---

## Finding 03 — on the GPU it is already faster than bf16; the remaining gap is launch overhead

Splitting decode into device time and the wall-clock gap around it separates
arithmetic cost from framework cost, and they point in opposite directions. Int8
tensor cores move half the weight bytes at twice the rate, so the exact path
wins on device time. It then gives that back by issuing 1.45x the kernel
launches.

| measurement | exact / stock | reading |
|---|---:|---|
| **Device time** | **0.97x** | faster on the GPU |
| Wall clock, narrow activations | 1.04x | launch overhead |
| Wall clock, wide activations | 1.09x | + the second matmul |
| vs production vLLM | 1.54x | entirely CUDA graphs |

**The baseline was validated, not conceded.** The obvious criticism — "you
benchmarked against HuggingFace" — was tested. vLLM with CUDA graphs disabled
measures 241.2 tok/s/seq against compiled HuggingFace's 241.9. They are the same
number. So the baseline was not weak, and the entire 1.54x gap to production is
CUDA graphs — not continuous batching, not paged attention, both of which buy
nothing at this batch size and model size.

---

## Finding 04 — the exactness argument is machine-checked, and checking it found two bugs

The whole claim reduces to one sentence: every partial sum stays below the point
where its float type starts rounding. That was discharged by five hand-derived
bounds across two files. Each is trivial alone, which is exactly why the
conjunction needed a solver — two adaptive width mechanisms interacting across
five bounds and three tunable widths.

- **One bound subsumes another.** Reaching the digit-split path is *itself* a
  proof that the float64 recombination is exact — the digit bound is strictly 2x
  tighter. I predicted a gap there and there is none.
- **An unwritten context ceiling.** The exact path is limited to 264,208 tokens
  at the shipped width, scaling as `2^25 / V`. It was in no document. It is now
  refused rather than computed.
- **A silent wrap in shipping code.** The plain int8 path's int32 accumulator was
  never bounded and wraps at `K >= 133,153`. Unreachable in any model that
  exists — a 70B MLP is 28,672 — but a wrap is a wrong answer, not an imprecise
  one. It refuses at construction now.
- **Two "obviously redundant" lines are load-bearing.** A floor cap cannot meet a
  strict bound; the power-of-two rounding and the minimum-width floor are each
  insufficient alone and sufficient together. Both looked like defensive padding.

**Mutation-verified 7 / 7.** A checker that catches nothing is worthless, so
seven deliberate bugs were introduced into the code it checks. The first round
caught six — and investigating the miss showed that mutation was *not a bug*,
for a stateable reason. It was replaced by one that is, and by a differential
against exact integer arithmetic, which caught a lowering that satisfied every
bound and still computed the wrong product.

---

## Finding 05 — the kernel the compiler emits was not checked against its own bounds

Finding 04 machine-checked the *prototype's* bounds. The compiler's own PTX
generator, `src/exact_attention.rs`, is a separate implementation and was
checked against nothing.

* **The sequence ceiling was recorded in a test comment and enforced nowhere.**
  `attention_ptx` pasted `head_dim` and `seq_len` into the template unexamined.
  Past `2^63 / ((2^28 - 1) * 127)` the 64-bit accumulator wraps — a wrong
  answer, not an imprecise one, and the same failure mode as the prototype's
  `K >= 133,153` int32 wrap. It is derived and refused now, and the test
  re-derives it so a change to the weight scale or V's width has to move it.
  Worth stating plainly: this ceiling is **270,549,122 keys**, not the
  prototype's 264,208 tokens. That number comes from recombining in float64;
  this kernel recombines in integers, so the two limits are unrelated and
  quoting the smaller one for the compiler path would be wrong.
* **A dead knob that documented the opposite of the design.** `attention_ptx`
  took a `c_hex` softmax temperature whose doc-comment said it "must stay a
  power of two, because the kernel's argument conversion is a shift". The
  kernel had since been changed to take the temperature as a *runtime*
  parameter — the whole point being that a real `q_scale * k_scale / sqrt(d)`
  is not a power of two — and `$C` had disappeared from the template. The
  argument was accepted, substituted into nothing, and discarded.
* **`--emit-attention-ptx` did not exist**, though the module header and
  `tools/ptx_bridge.py` both used it. It fell through to the source-file path
  and tried to open `64.ysu`. Loud rather than silent (the bridge checks the
  exit code), but the bridge could not have worked.
* **`fixed_exp`'s saturation guard was unpinned**, and it is a guard: without
  the `n >= 30` early return the shift width runs past 64 bits. The existing
  saturation case passed with it deleted. The accuracy and monotonicity claims
  are also exhaustive now rather than sampled — the headline 0.908 ulp was
  measured over one fractional value in sixteen above `n = 0`, and the full
  domain costs 4 ms.

The pattern across all five: **a number that lives in a comment is not a
check.** Each of these was correctly derived and written down somewhere, and
none of them was consulted by the code that needed it.

---

## Finding 06 — the bridge that proves "the demo's numbers are the kernel's" was comparing the kernel to itself

`tools/ptx_bridge.py` is the artifact behind the claim above: it loads the
compiler's own PTX through the CUDA driver, runs it on real post-RoPE Qwen
activations, and reports **12/12 bit-identical** against the torch path. The
number is true and it was not evidence.

* **The temperature was 65536x too small.** The kernel takes `KFix` and forms
  the exp's Q16.16 argument as `t = ((m - s) * KFix + 2^15) >> 16`, so one
  factor of `2^16` is consumed by the shift and `KFix` must be `C * 2^32`,
  where `C = q_scale * k_scale * softmax_scale * log2(e)`. The torch demo has
  no shift — it builds the Q16.16 logit directly — so *its* multiplier is
  `C * 2^16`. Two conventions, one variable name. The bridge computed the
  demo's and handed it to the kernel.
* **The consequence is not a small numeric error, it is a different function.**
  At Qwen's per-tensor scales that puts every exponent `t >> 16` below 0.014,
  so every softmax weight lands within 1% of `2^28`: attention with **uniform
  weights**. A computation with no temperature in it at all.
* **The differential could not see it, by construction.** Both arms replicate
  the kernel's formula — which is the right design, because `KFix` is a rounded
  integer and only a replica can be compared bit for bit — so they agree
  perfectly on the uniform answer too. The check would have passed with the
  temperature deleted from the kernel. This is the failure the repo already has
  a name for: *two wrong things agreeing*.
* **What was missing was a control, and it costs one line.** `max(p)/mean(p)`
  is exactly 1.0 for a uniform weight vector; the bridge now tracks it and
  fails below 2.0. Every other claim in this document has such a control — the
  f32 baseline's error must be non-zero, the divergence probe must actually
  diverge — and this one did not.
* **Two CPU-side tests pin the convention** so it cannot drift again unseen:
  one asserts the kernel's integer path reproduces the demo's Q16.16 argument
  (with the tolerance the rounded `KFix` forces, `1 + ds/2^17`) and that the
  demo's own multiplier does *not*; the other asserts the emitted PTX still has
  the multiply-and-shift shape, and that the saturate precedes the narrowing.
  Both are mutation-verified. `tests/attention_quantization_error.rs` now also
  ties its three private constants — the Q0.28 width, the temperature and the
  sequence bound — to the ones the compiler actually emits, instead of
  transcribing them.

**Re-run on the GPU, 2026-08-22 — the fix holds and the counterfactual is
measured, not inferred.** With the corrected `KFix`: 12/12 bit-identical and
`max(p)/mean(p) = 110.9`, i.e. a genuinely peaked softmax. Reverting only the
`kfix` line and re-running: still **12/12 bit-identical, 0 mismatched** — and
`max(p)/mean(p) = 1.0` exactly, with the new control failing the run. So the
original result really was a perfect agreement about uniform attention, on the
real activations, exactly as the algebra predicted.

Finding 05's lesson generalises: a number in a comment is not a check, and a
**differential whose two arms share a constant is not a check of that
constant.**

---

## Finding 07 — two more checks that did not reach the thing they name

Finding 06 was a comparison that could not see a wrong shared constant. The
same sweep, continued through the Python side, found two more of the same
family — a check standing next to the property it claims, not on it.

* **The Python transcriptions of the fixed-point exp were checked against
  Python.** There are four copies of that recipe: `src/fixed_exp.rs`, the PTX
  in `ptx_device_function`, a pure-Python one in
  `tools/attention_real_activations.py`, and a vectorised torch one in
  `tools/batch_invariance_demo.py` — the last being what the demo,
  `exact_accuracy.py` and `ptx_bridge.py`'s reference arm all call. Only the
  PTX one was tied to Rust. The two Python ones were compared against float64
  transcriptions of *themselves*, asserting the result is within 1 ulp of
  `2^28 · 2^-t` and monotonic. Those are **properties, not identity**: two
  implementations can both be within 1 ulp of the truth and differ from each
  other by 1, and 1 ulp is a bit. The comment claimed it made "a divergence
  visible rather than assumed away"; it made exactly that divergence invisible.
  `EXP2_DOMAIN_DIGEST` — FNV-1a over all 2,031,616 arguments — is asserted on
  both sides now. A one-LSB error in a table entry, which is how a hand-copied
  table actually goes wrong and has already happened once in this repo's
  vendored circomlib, passes the old check and fails the digest. The three
  implementations do agree; that is now a fact rather than a hope. (The ulp
  sweep also covered `t < 2^16` only — one of thirty-one unit intervals — at
  stride 7. It is exhaustive now and independently reproduces 0.9076 ulp.)
* **`exact_bounds_check.py`'s score-budget check was testing its own copy of
  the rule.** That file closes with "do the asserts actually refuse at the
  limits Z3 found?", which is the part that ties the model to the running code.
  The 2^24 fp32 score budget was a bare assert inside `exact_attention`,
  unreachable without building a model, so the checker re-derived
  `d * Q_LEVELS * k_lv < 2^24` locally and asserted that instead. Neuter the
  real assert and the checker still prints `[ok] head_dim 1041 (past the score
  budget) is refused` and passes everything else. It is a callable now
  (`assert_score_budget`) and the checker exhausts the real one.

The second is the sharper finding, because **the file already knew the rule.**
`digit_width`'s docstring, three hundred lines up, says a checker that
re-implements what it checks "only ever proves I can copy a line twice" — and
was split out of `exact_pv` for exactly that reason. The discipline was
correct, written down, and applied at one of the two sites that needed it.

Everything in this finding is CPU-only. `exact_bounds_check.py` runs green
under z3 5.0.0, including the 264,208-token context ceiling at V=127.

---

## Finding 08 — the sweepable knobs were the ones nothing swept

`exact_kv.py` exposes `Y_EXACT_K_LEVELS` and `Y_EXACT_V_LEVELS` "so the
accuracy/throughput trade can be swept without edits". Both were read with a
bare `int()` and used unchecked, and the exhaustive checker swept head_dim only,
at their defaults.

* **`Y_EXACT_V_LEVELS=0` zeroed the V cache, silently.** Measured on
  `quantize_rows([1, -2, 3, 4], lv)`: `levels=0` gives `q = [0,0,0,0]` with
  `scale = inf`, and `levels=-5` gives `q = [-5,-5,-5,-5]` with a negative
  scale. No error, no NaN — a model computing something else. `quantize_rows`'s
  docstring already cites the design-rule table for whether `levels` was
  *supplied*; the rule was never applied to its *value*.
* **A K request below 127 was silently widened to 127**, because 127 is also
  `k_levels_for`'s fail-closed sentinel — the value the score assert is
  guaranteed to refuse. A sentinel and a request cannot share a value. Sweeping
  that axis would have produced a flat left tail reading as "narrowing K costs
  nothing", which is the shape of measurement error this project has recorded
  three times already.
* **The off-axis sweep found nothing, and that is the result.** 7 `K_LEVELS`
  x 8 `q_levels` x 4096 head_dims = 229,376 configurations: every width is a
  `2^n - 1`, and every case where `d·q·k >= 2^24` is the documented 127
  fail-closed floor. The selector is correct on a domain 229,376 times larger
  than the one that was checked. Recorded because a negative result on a widened
  axis is worth as much as a positive one — it is what lets the next person
  stop looking here.
* **The checker's search for the first over-budget head_dim was an unbounded
  `while`.** It terminates only because of the 127 floor; delete the floor and
  it runs forever rather than failing. Bounded now — the same weakness
  `tests/llvm_control_flow.rs` grew a deadline for, and the mutation that
  exposed it is now caught in bounded time by two checks.

---

## Finding 09 — the acceptance harness exited 0 without a GPU, and tested one key length

`tools/exact_selftest.py` is what stands behind "the fast path computes the same
function". Its header says every check has a control, because a check that
cannot fail is not evidence. Its `main` opened with `if not
torch.cuda.is_available(): print("SKIP: no CUDA"); return 0`.

* **Ten of the thirteen checks need only torch** — including
  `check_batch_invariance`, the property the whole project is about. They run on
  CPU now, and the four that cannot are named: three launch Triton, and
  `check_pad_is_load_bearing` asserts a CUDA-specific `_int_mm` refusal that does
  not hold on CPU. `9/9` on CPU, `13/13` on CUDA.
* **`check_batch_invariance` ran at one key length, `T = 96`.** Two batch-shape
  dependencies live inside the exact path and it reached neither properly: the
  GEMM tiles by row count, and `exact_pv` picks its digit width from the
  **padded** key length — so a sequence's own answer depends on what the rest of
  the batch made `T`. The boundaries are `T = 519, 1041, 2097, 4262`. It now
  sweeps four lengths across those and adds the production-shaped case: the same
  400-key sequence answered alone (width 8) and inside a batch padded to 600
  (width 7).
* **The mutation run found the two masking sites hide each other.** Including
  padded keys in the softmax max is caught by two checks. But *not* forcing
  blocked positions to `tq = 2^30`, and deleting the `p = where(blocked, 0, p)`
  that follows, are each invisible alone — the exp already returns 0 for
  `2^30`, and the zeroing already covers a missing clamp. Remove **both** and
  the new ragged case fails on all three pad lengths (deltas 0.15–0.39) while
  the identical-rows case stays green. Defence in depth, and the harness cannot
  separate the two layers.
* **And the new case is deliberately not the digit-width gate.** Forcing a
  too-wide digit on the padded side only leaves row 0 bit-identical at widths 7,
  10, 12 and 14; the output first moves at 16, by 7.5e-9 on 7 of 896 elements.
  The bound is worst-case and a real softmax is nowhere near it except at the
  argmax, so an off-by-one in `digit_width` is invisible end to end. `check_pv`
  and `check_pv_bound_is_load_bearing` are the gate, with uniform random `p` up
  to the inclusive bound. Moving that coverage here would be a weaker test
  wearing the same name — worth stating, because "make the end-to-end test
  cover it too" is the obvious and wrong instinct.
* **The same measurement retracts a claim in `exact_ragged_batch.py`.** That
  harness names three implementation-level dependencies on batch composition and
  said of the first, the digit width, that "it is safe by argument, and this test
  is what turns the argument into evidence". Forcing a wrong width on the padded
  side at the key lengths it actually uses (`t_true` 173–500, `t_pad` 519–744,
  true width 7):

  | width | 7 | 8 | 10 | 12 | 14 | 16 | 20 |
  |---|---|---|---|---|---|---|---|
  | row 0 | = | = | = | = | = | **moved** | **moved** |

  Double the correct width, bit-identical output, at every shape tried. So that
  test does *not* evidence dependency #1. It evidences #3 strongly — the masking
  mutation above fails there with deltas of 0.15–0.39 — and #2 through batch
  size. A previous pass had already corrected this file once, for claiming a
  digit boundary was crossed when it was not; the crossing is real now, and the
  crossing has no teeth. **"The path is exercised" and "the test can fail on it"
  are different properties, and only the second one is coverage.**
  None of this moves the 0/16 determinism result, which is measured directly —
  it corrects what that result is attributed to.

---

## Finding 10 — the accuracy harness validated the path that is not the headline

`exact_task_accuracy.py` produces the two accuracy numbers in this document, and
its header states the discipline plainly: *"a harness that cannot detect damage
proves nothing"*, with `crude4` — the same weights on a 4-bit grid — as the
control, and *"assert the control fails"*.

* **It printed the verdict and returned 0.** A run whose control did not fire
  printed `*** the harness cannot see a deliberately damaged model ***` and
  exited **successfully**, so anything reading the status saw a pass. The same
  shape as the fuzz target that reported findings with `eprintln!` and never
  panicked. Its two sibling harnesses, `exact_accuracy.py` and
  `exact_ragged_batch.py`, both `return 1` on a failed control — this was the
  one file out of step with a convention the directory otherwise keeps.
* **The control checked perplexity, and the headline is the multiple-choice
  paired net.** Those are separate code paths. If the MC scorer collapsed —
  always picking option 0, or the length normalisation going flat — every arm
  would score alike, *"net +40 of 3,000 items"* would read as a clean null, and
  the perplexity control would fire anyway, because perplexity is computed
  somewhere else entirely. `crude4` must now be visible in **both**, and in the
  MC case through the same paired statistic the conclusion is drawn from.
* **The control's own logic is now checkable without a GPU.** The real harness
  needs three 0.5B models, so the thing that decides whether a run is believable
  could not be exercised on an ordinary machine. `--check-control` runs five
  synthetic scenarios in milliseconds:

  ```
  ok    crude4 clearly worse in both         exit 0 (want 0)
  ok    multiple-choice scorer collapsed     exit 1 (want 1)
  ok    perplexity path blind                exit 1 (want 1)
  ok    both blind                           exit 1 (want 1)
  ok    control arms not run                 exit 0 (want 0)
  ```

  Mutation-verified: reverting to the perplexity-only control turns the second
  line into `WRONG ... exit 0 (want 1)` and fails the run.

Neither fix moves the +0.12% or the +40-of-3,000. They change what those numbers
would have survived.

---

## Where the remaining cost is, and one lead that is not there

Counting work rather than timing it, so this holds on a contended machine.

`exact_pv` is the structural multiplier in the attention path: it splits the
Q0.28 softmax weight into base-`2^dbits` digits and runs one fp32 matmul per
digit, where a float path runs one matmul total.

| T | digit width | `p@V` matmuls |
|---|---|---|
| ≤ 518 | 8 | 4 |
| ≤ 1040 | 7 | 5 |
| ≤ 2096 | 6 | 5 |
| ≤ 4261 | 5 | 6 |
| > 4261 | 4 | **8** |

**The obvious lead is that the width is conservative — and it is not
exploitable.** Finding 09 measured widths up to 14 giving bit-identical output
at T = 600, which would be 3 matmuls instead of 5. Checked before proposing it:
the bound `T · (2^dbits − 1) · V_LEVELS < 2^24` is **tight in the worst case**.
The low digit's sum is `Σ (p_i mod 2^w)`, on which the denominator `l` gives no
bound at all; and the high digits are bounded by `l / 2^s`, which in the worst
case (`l = T · 2^28`, every score equal) reduces to exactly the same inequality.
So the slack is a property of real softmax weights and there is no sound static
version of it. A data-dependent width would also make the *number of matmuls*
depend on the data, which is a different kind of variability from the one this
project removes but not one worth introducing.

What that leaves is the cost already attributed: device time is **0.97x** stock
and wall clock is **1.09x**, so the deficit is launch count (1.45x), not
arithmetic — and the fix for launch count is CUDA graphs, which is blocked on a
library bug rather than on anything here.

---

## Finding 11 — the missing control was built, and it changes what the claim is

The control this document has named since its first draft: **a float kernel
whose reduction order is pinned should be deterministic without quantizing
anything.** If that is true and cheap, exactness is unnecessary and the pitch is
wrong. `tools/fixed_order_float.py` is that arm — fp32 throughout, with both
reductions (`q·k` over head_dim, `Σ p·v` over the key axis) done in fixed-size
chunks accumulated in a fixed sequence, and the key-axis chunks anchored at the
**end** so left padding cannot regroup a row's real keys. The linear layers are
pinned the same way, because the exact arm converts them too.

It faces the same 16 batch compositions as the other arms, in the same harness.

| arm | compositions changed |
|---|---|
| stock bf16 | 16 / 16 |
| exact int8 | **0 / 16** |
| **fixed-order fp32** | **0 / 16** |

**So determinism does not require exactness.** A pinned reduction order gives
batch invariance too, and that has to be said plainly because this document
spent several drafts implying otherwise.

What it costs is the other half, measured on the same box, interleaved, best of
3, spreads 0.2–0.8%:

| arm | tok/s/seq, no compile | tok/s/seq, `torch.compile` | vs stock |
|---|---:|---:|---:|
| stock bf16 + SDPA | 174.3 | **246.3** | — |
| exact int8 | 54.7 | **238.5** | 1.03x slower |
| fixed-order fp32 | 37.7 | **36.0** | **6.85x slower** |

**The interesting column is the compiler's.** `torch.compile` takes the exact
arm from 54.7 to 238.5 tok/s/seq — a 4.4x gain — and takes the fixed-order arm
from 37.7 to 36.0, i.e. nothing. That is not an accident of this
implementation's shape; it is the mechanism. **Pinning a reduction order is
precisely a constraint on how the reduction may be executed**, so the fusions
and kernel choices that make the float path fast are exactly what the pinning
forbids. Integer accumulation needs no pinning, so the compiler stays free.

The honest restatement of what this work buys:

> Exactness does not buy determinism. Exactness buys determinism **while
> leaving the compiler and the kernel autotuner free**, which is why it costs
> 1.03x where pinning the order costs 6.85x.

**What is not proven.** This arm pins the order with chunked loops in Python,
which is the naive construction. A hand-written fixed-order CUDA kernel would be
far faster than 6.85x slower, and nothing here rules out a fast one existing. So
`0/16` is a solid result and the 6.85x is an upper bound on *this* construction,
not a lower bound on the idea. What does generalise is the compiler column: any
fixed-order implementation forbids the same class of optimisation, whoever
writes it.

---

## Finding 12 — most of the cross-architecture claim does not need a second GPU

Finding 11 left an obvious challenge open: if a *fixed-order float* path is
batch-invariant too, why not write a fast fixed-order float kernel and skip the
quantisation? The answer is portability — a float softmax still needs a
transcendental, and `ex2.approx.f32` is specified by a **tolerance, not a
value**, so which result inside that tolerance you get belongs to the SM
generation.

That argument can be made almost entirely on one card, because it is a property
of the *instruction stream*: if every instruction is exactly specified by the
ISA, the result is determined. Assembling both probes from `tests/ptx_fixed_exp.rs`
and disassembling the cubin:

| kernel | transcendental unit | floating-point ops | instructions |
|---|---|---|---|
| `hw_exp_probe` (`ex2.approx.f32`) | **`MUFU` × 1** | FMUL×4, FSETP, FSEL, F2I | 32 |
| `fixed_exp_probe` (integer) | **none** | **none** | 72 — IMAD, SHF, IADD3, ISETP |

**The integer path contains nothing two architectures are permitted to disagree
about.** Integer multiply-add, shift, add and compare are exactly specified;
`MUFU` is not. That is now a test — `the_integer_exp_compiles_to_no_architecture_defined_instruction`
— and it needs `ptxas` and `cuobjdump` but **no GPU**, so it runs on an ordinary
machine.

Its control is the second row: the float probe must show a `MUFU`, or the
detector is blind and a clean integer result means nothing. Mutation-verified
both ways — blinding the detector fails it, and removing the `ex2.approx.f32`
from the float probe fails it.

**What this is and is not.** It does not measure two architectures agreeing;
that still needs a second GPU and remains the one headline claim without a
measurement. It proves the weaker and more useful thing: the integer path has no
instruction whose result the ISA leaves open, which is the premise the
cross-architecture claim rests on. The float path has exactly one, and it is
unavoidable in a softmax.

---

### Finding 12, resolved: the exp IS measured across two architectures now

Written while a background job held the card, run once it was free. The device
test was a ~100k structured sample; the reachable domain is 1,966,080 arguments
and 8 MB in each direction, so there was no reason to sample it. Exhaustively:

    device == host on all 1,966,085 arguments, bit for bit
    846,328 distinct results, so the agreement is not over a constant

**x86-64 against sm_89 is a wider architectural gap than two NVIDIA cards.** For
this kernel the cross-architecture claim is measured rather than argued, and the
degeneracy control rides along in the same line — the 846,328 figure
independently matches what `exact_bounds_check.py` computes for the same domain
in Python, which is two implementations agreeing on the shape of the function as
well as its values.

The same run prices the alternative: the device's `ex2.approx.f32` against the
host's `exp2` differs on **46,301 of 100,427** arguments, worst gap 32 ulp of
Q0.28, and is 32.88 ulp from correctly rounded. That is the instruction a float
softmax would have to use.

Mutation-verified: flipping a single host value out of 1,966,085 fails it.

What is still unmeasured is the **pipeline**, not the exp — the whole decode path
producing identical tokens on two different cards. That needs a second GPU and
stays the headline claim without a measurement.

## Finding 13 — `0/16` is also what a broken arm scores

Finding 11's fixed-order-float arm read **12/16** on its first run. Two of my own
errors, both caught before reporting — but the second is the interesting one: the
arm returned `[b, nh, q, d]` where `eager_attention_forward` returns
`[b, q, nh, d]`.

**It did not crash.** The model ran end to end, produced fluent-looking text in the
wrong script (`' الساد الساد ...'`), and the invariance harness reported a number
for it. I caught it by generating text and reading it, which is not a check.

The direction that matters is not the one I hit. A broken arm that is broken
*consistently* — the same wrong answer regardless of who else is in the batch —
scores **0/16**, and 0/16 is the passing number. Every harness here would have
declared it the winner.

Three sites, and the ones I had not hit were worse than the one I had:

| harness | what it shows a human | so a broken arm |
|---|---|---|
| `exact_ragged_batch.py` | text only when a case **differs** | prints nothing, scores 0/16, passes |
| `batch_invariance_demo.py` | text only when a prompt **diverges** | prints nothing, scores 0/8, **is the headline** |
| `exact_throughput.py` | never decodes a token at all | reports a throughput — and several ways of being broken are **faster** |

That last row is where the 1.03x of finding 11 comes from. A cost figure for a
model that is not computing the right thing is not a cost.

### The gate

`sanity_verdict(canary, ref_tokens)` runs per arm before any score is counted, on
two checks that are independent on purpose:

* **a canary** — the arm must continue `"The capital of France is"` with something
  containing `"Paris"`. Deliberately *not* one of the measured prompts: those were
  chosen for a narrow top-2 margin, which is the worst possible way to ask whether
  an arm computes the model's function at all.
* **degeneracy** — the arm's own 160-token reference generation must not collapse
  to fewer than 8 distinct tokens. The canary is 10 tokens on a different prompt,
  so it cannot see an arm that is right for a while and then falls over.

It lives in `batch_invariance_demo.py` and the other two import it, because
duplicating a rule across three files is how the constant-folding family in
`zk_emitter` happened.

### Verified on CPU, which is the point

`--check-gate` exercises the verdict on five synthetic outputs with no GPU and no
model, and **scenarios 3 and 4 are the ones that earn their place**: "fluent but
wrong function" passes degeneracy and fails the canary; "canary ok but generation
collapsed" does the reverse. Neither check alone catches both.

Mutation-verified, three mutations, all caught:

| mutation | caught by |
|---|---|
| canary check deleted | scenario 3 only |
| degeneracy check deleted | scenario 4 only |
| `MIN_DISTINCT` 8 → 200 (threshold too tight) | scenario 1 — a false failure stops a 20-minute run |

**Deleting the canary flips only one of the five scenarios, not two** — scenario 2,
the real layout bug, is caught by *either* check. That is defence in depth rather
than a hole, and it is exactly why scenario 3 had to be written: without an arm
that is fluent *and* wrong, the canary's own test would have passed with the canary
deleted.

The canary constant itself is the one thing that could turn this gate into a wall,
so it was run against the real model — on CPU, fp32 eager, the configuration both
the exact and fixed-order arms are built from:

    CANARY -> ' Paris. It was founded in 789'

Not yet run against the bf16+SDPA stock arm or the int8 exact arm; both need the
card.

## Finding 14 — the bound checks had the same shape, and one of them had never checked anything

Finding 13's lesson is not about invariance. It is about **any metric whose
passing value is zero**, and `exact_bounds_check.py` is full of them: every
exhaustive sweep reports "0 violations over N configurations", and 0 is what a
predicate that cannot fire reports too.

Three things came out of applying it there.

### 1. A bound check that a dead function passes

`exhaust_exp_range` verified `p <= 2^28` over the whole 2M-value domain. Stubbing
`exp2_neg_q16_16` to return **zero everywhere**:

    [ok ] p <= 2^28 over the entire reachable domain
          max p = 0 at t=-1 (2^28 = 268,435,456), ...

Green, with `max p = 0` printed in its own detail line and gating nothing. This
is finding 06's uniform-softmax bug in a different file — a bound is satisfied
most comfortably by a function that computes nothing.

The check now requires the bound to be **attained**, and three more things that
each kill a different mutation:

| requirement | kills |
|---|---|
| `worst == 2^P_BITS` | exp2 returning 0 everywhere |
| exact dyadic anchors, `exp2(-k) == 2^(28-k)` for k in 0..28 | exp2 returning the constant 2^28, which passes the row above |
| ≥ 2^16 distinct outputs (real figure: **846,328**) | a table collapsed to a handful of values that hits every anchor and is wrong between them |
| `min == 0` reached inside the domain | the decay being clipped |

### 2. A guard that was mathematically incapable of firing

`exhaust_digit_loop` had two branches. The second, described as *"the shift loop
must cover all 29 bits of p"*, was

    ceil(29 / dbits) * dbits < 29

`ceil(a/b)*b >= a` is an identity. **It is false for every `dbits >= 1`, so it
had never fired and never could** — one of the two things this file claimed to
check about the digit loop had never been checked. Extracting the predicate so a
self-test could feed it a violation is what found it: there was no violation to
feed.

It was also asking the wrong question. Coverage is a property of `exact_pv`'s
loop *condition*, not of a formula, and recomputing the formula here compares the
transcription against itself. Deleted, with the real guard named:
`differential_exact_pv`, which drives `p` to exactly `2^P_BITS` and compares
against an integer reference. **Verified rather than asserted** — mutating the
loop to stop one bit early:

    [ok ] digit width keeps every fp32 matmul exact (4131 values of T)   <- blind
    [BAD] exact_pv == integer p@v ... mismatches: [(519, 7, 30), (1024, 7, 30), ...]

### 3. `while shift < 29` was a transcription of a module constant

Found on the way. `exact_pv`'s digit loop was written against a literal `29`
while the constant it means is `P_BITS + 1`. Nothing links them, so raising
`P_BITS` would leave the one loop whose job is to cover the top bit silently
dropping it — with every bound in this file still holding, because a narrower `p`
satisfies every bound. It reads `P_BITS + 1` now. Same disease as
`feedback_constants_encode_old_structure`, and the same as the mirrored score
budget that finding 08 removed.

### The predicates are extracted, and the width rule was written twice

`width_violation` and `digit_violation` are pure functions now.
`--check-sweeps` feeds them 15 cases with no z3, no torch and no GPU — one per
reason they can return, plus the near-misses that must still be accepted.

The extraction also collapsed a **duplicate implementation**: the width rule was
written out once in `exhaust_k_levels_for` and again inside the off-axis sweep,
which is the shape that produced the `zk_emitter` constant-folding family.

Mutation-verified, six mutations, all caught:

| mutation | cases that go wrong |
|---|---|
| `2^n-1` branch deleted | 1 |
| budget branch deleted | 3 |
| non-positive-width branch deleted | 2 |
| the 127-floor exemption dropped | 1 |
| digit budget branch deleted | 2 |
| `dbits < 1` branch deleted | 1 |

**Two of my own expectations were wrong on the first run, and the predicate was
right** — `2^24-1` *is* a `2^n-1`, and `2·(2^23-1)` is one short of the budget.
That is the direction you want a self-test to fail in, and the corrected cases
now straddle the boundary by exactly one.

After all of it: bounds check **22/22**, exact self-test **13/13**, Rust suite
**615 passed / 0 failed / 30 ignored** across 69 binaries.

## Where the launches go, attributed

748 kernels/step, 3.201 ms device, batch 32, one decode step, `torch.compile` on
(`tools/exact_kernel_census.py`). A kernel COUNT is invariant under contention,
unlike every timing here, so the left column is the trustworthy one.

| kernel | launches/step | us/step | share |
|---|---|---|---|
| `_w8a8_gemm` | 145 | 1238 | 38.7% |
| `_w8a8_gemm_splitk` | 24 | 356 | 11.1% |
| cuBLAS `gemmSN_NN` | 96 | 258 | 8.1% |
| cuBLAS `gemmSN_TN` | 24 | 110 | 3.4% |
| ~18 elementwise Triton kernels | ~435 | ~1240 | 38% |

**The two cuBLAS rows are `exact_pv` and QK^T, and that is arithmetic rather
than a guess.** At every key length this workload reaches, `digit_width(T) = 8`,
so the digit loop runs `ceil(29/8) = 4` matmuls; 4 x 24 layers = **exactly 96**,
and the remaining 24 TN calls are one QK^T per layer.

### The lever

`exact_pv` issues **four separate cuBLAS matmuls per layer** over the same `p`
and `V`, round-tripping partial sums through global memory between them. One
kernel running the digit loop internally deletes **72 of 96 launches** (12.8% of
every launch in the step) and keeps the accumulator in registers.

It is also the natural next kernel for Y to emit, which is why this sits at the
junction of the launch-count work and the get-off-the-prototype work: `p` is
integer, `V` is int8, the accumulation is exact by construction, and the digit
loop is a compile-time-bounded `for` over `ceil(29/dbits)` steps.

`_w8a8_gemm`'s 145 launches are ~6 per layer, i.e. one per linear. That is not
redundancy, it is the work; its 8.54 us each is arithmetic, not overhead.

## The first kernel taken off the prototype: `exact_pv`

The census said `exact_pv`'s digit loop was 96 of 748 launches per decode step
and the plan was to fuse four matmuls into one. **The better answer is that they
should not exist.** The split is a workaround for the *accumulator* -- a Q0.28
weight times an int8 activation is 35 bits and an fp32 mantissa holds 24 -- and
a kernel accumulating in int64 has nothing to split.

`tests/exact_pv.ysu`: 21 registers, zero spill, zero local. Measured against the
digit split at batch 32 x 14 KV heads = 448 rows, head_dim 64, best of 30-40
interleaved rounds (`tools/exact_pv_bridge.py`), **bit-identical at every key
length**:

| T | dbits | matmuls | kernel alone | with the cast charged |
|---|---|---|---|---|
| 67 | 8 | 4 | 3.9–5.1x | ~3.0x |
| 1024 | 7 | 5 | 4.0–6.4x | 2.3–2.7x |
| 4096 | 5 | 6 | **4.66x** | **2.50x** |

T = 4096 reproduced to three digits across three runs; the small-T rows vary
because the kernels are 0.03–0.3 ms and this machine's absolute timings drift
between windows. Read the interleaved ratios, not the milliseconds.

### Two columns, because the first version of this benchmark was biased both ways

`quantize_rows` returns `vi` as **float32 carrying integers in [-127, 127]**, so
the digit split consumes it with no conversion. The first version of the tool
built `v` as int64 and converted *inside the timed digit arm* -- charging it for
work the model never does -- while letting the Y arm skip the float32 -> int8
cast it genuinely owes. That inflated the result to 4.2–6.7x. Both are priced
now, separately, the way the MSM numbers separate cold from fixed-base from
kernel.

**The cast is not a rounding error: 0.945 ms against the kernel's 1.095 at
T = 4096, i.e. 86% of the kernel again.** Which points at something bigger than
this kernel -- `vi` is a tensor of small integers stored at four bytes each, so
the KV cache moves four times the bytes it needs to. A genuinely int8 cache
deletes the cast *and* three quarters of the read traffic. That is a change to
the cache, not to this comparison, and it is now the ranked next item.

### And it moves a documented ceiling

The fp32 route needs `T * 2^28 * V_LEVELS < 2^53` overall, which is 264k tokens
at `V_LEVELS = 127`. Accumulating in int64 needs `< 2^63`: about **2.7e8
tokens**. The hard context ceiling recorded in
[[feedback-check-bound-conjunctions-with-z3]] was a property of the accumulator,
not of the method.

### Wired in: the int8 V cache, and what the census predicted

`Y_EXACT_V_INT8=1` stores V as real int8 and routes `exact_pv` to the compiler's
kernel. **The dtype is the switch** -- `exact_pv` dispatches on what it is
handed, so storage and consumer cannot disagree, and the int8 branch RAISES if
the kernel will not load rather than casting back to float32. A fallback there
would silently reinstate the whole-cache cast and report a number for a path
nobody chose.

Correct first, at every level:

* the two routes are **bit-identical** on four shapes including head_dim 128
  and Q = 1;
* `exact_selftest.py` is 13/13 in both modes;
* the model emits **identical tokens**, 0 of 64 differing, and coherent text --
  checked by reading it, after finding 13.

End to end at batch 32, 64 tokens, arms interleaved across rounds because the
flag is read at import and they cannot share a process:

| | best s | tok/s/seq |
|---|---|---|
| float32 V, digit split | 1.836 | 34.9 |
| int8 V, Y kernel | **1.719** | **37.2** |

**1.068x, and the census predicted 1.064x.** `exact_pv`'s digit matmuls were
258 us of a 3,201 us step (8.1%); a 5x kernel removes four fifths of that, i.e.
6.4%. Measured 6.8%, the remainder plausibly the 72 deleted launches. A phase
split that predicts the end-to-end number to within half a point is the
strongest evidence that the attribution was right -- and it is also the reason
not to oversell the kernel's own 5.04x: Amdahl caps it at 8.1%.

#### And in the configuration anyone would ship, the first version LOST 6.2x

Measured only after the eager number was already written down:

| compiled | s | tok/s/seq |
|---|---|---|
| float32 V, digit split | 0.266 | **240.3** |
| int8 V, Y kernel (first version) | 1.662 | **38.5** |

`torch.compile` gave the digit split 6.3x and the kernel path **nothing** --
38.5 compiled against 37.2 eager. A ctypes launch inside `forward` is opaque to
dynamo, so the model went from **1 graph / 0 breaks to 18 graphs / 17 breaks**,
and shattering the graph costs far more than the kernel saves.

`feedback_stateful_optimisations_lose_to_compilers` recorded this exact shape
already, for an O(1) KV cache that measured 550x slower under compile. **A
kernel-level win measured in eager is not a result**, and the eager number
above was true and would have been published as though it meant something.

The fix is a `torch.library.custom_op` with a registered fake: dynamo sees one
opaque node of known shape and dtype and traces straight past it. Back to **1
graph, 0 breaks**.

**That recovers most of the loss and does not close it.** Interleaved
round-robin, compiled, with the custom op in place: float32 V 244.4 tok/s/seq
against int8 V + Y kernel 171.3, i.e. **0.70x** where the eager measurement said
1.068x. So the graph breaks were roughly 4.4x of the 6.2x and something else is
the remaining 1.43x -- most likely that the digit split's fp32 matmuls are
themselves fused into the compiled graph, while an opaque custom op cannot be.
**These two numbers were taken while another process held the GPU**, so read the
ratio and not the absolute figures, and re-measure on an idle card before
quoting either. The conclusion that survives contention is the sign, and the
sign is negative: on this stack the kernel path is not currently worth shipping,
whatever the census predicted from launch counts alone.

## What this does not yet claim

Written at the same length as the findings, because a status report that buries
this is worth nothing to the person reading it.

### The alternative has no baseline — never measured

Every comparison here is exact-versus-stock. **Nobody has measured exact versus
fixed-order float** — keeping bf16 and pinning the reduction schedule, which is
what a competitor shipping "batch-invariant kernels" would do. That is the
control that actually answers whether exactness is necessary. My expectation is
that it wins on accuracy and loses on throughput, because pinning the order
forfeits autotuning and split-K; but this project's record on predictions is one
2x forecast that measured 1.06x. It is the single strongest experiment left.

### One model, one GPU, one stack — narrow evidence

Qwen2.5-0.5B-Instruct on an RTX 4070 Ti SUPER, in a PyTorch prototype rather
than the compiler's own PTX kernels. Head dimension 64 here; at 128 and 256 the
exactness budget forces narrower K, which is derived correctly but untested on a
real model. The cross-GPU claim — the strongest thing exactness buys over
pinning the order — has never been run on a second GPU.

### CUDA graphs, worth up to 1.47x — blocked

The prerequisites are cheap (a static cache costs 5–8%, and improves the ratio
to 1.01x) and the property survives them. Capture segfaults on *both* arms — an
in-place mutation inside the captured region, in this transformers/torch pair. A
library bug, not an exactness problem, and its proper home is a serving
integration rather than this prototype.

### "Deterministic" is the claim, not "exact" — scope

RoPE's sine and cosine, the residual adds, and the softmax exponential are
approximate — deterministic, but not exact. Exactness is the mechanism applied
to the reductions whose order moves with batch shape; it was never applied
everywhere and does not need to be. Switching cache implementation moves the
output by 5.7e-6 once and is stable thereafter — batch invariance survives it,
but cache-implementation invariance is a different claim and was never made.

---

## Use cases, ranked by sharpness of pain

**01 · Reinforcement learning teams — a rollout you cannot reproduce is a bug you
cannot bisect.** Training against a generating model means the reward signal
depends on sampled text. When the same prompt yields different tokens because the
batch happened to be fuller, a regression in the policy and a regression in the
serving stack are indistinguishable. This is the first conversation to have,
because the pain is concrete and the tolerance for a 9% decode cost is high.

**02 · Audited and regulated inference — "same input, same output" as a property
rather than a policy.** Where a decision has to be defensible after the fact,
reproducing it exactly is the difference between an audit trail and an anecdote.
Integer accumulation makes that a property of the arithmetic instead of a promise
about configuration management.

**03 · Serving CI — regression tests that do not flake.** Output-level tests
against a non-deterministic decoder are either loose enough to miss real
regressions or tight enough to fail on load. Bit-identity collapses that trade:
the assertion is equality, and it holds under any batch composition the test
harness happens to produce.

**04 · Compiler autotuning, as a side effect — the tuner may pick freely because
every choice computes the same function.** This is the underrated one. Eleven
GEMM configurations — different tiles, different K-splits, atomics whose
completion order is genuinely nondeterministic — produce identical bits, and that
is a tested property rather than an argument. Pinning a float reduction order
instead would make the schedule part of the model's definition, which forfeits
per-device tuning and cross-hardware reproducibility together.

---

## Method — how the numbers were arrived at

This is a sequence, so it is numbered. Most of the work was not building things —
it was finding out that measurements and tests were lying, usually in the
flattering direction.

1. **Measure the ceiling before building anything.** Fake-quantize in fp32 to
   price an accuracy change: no kernel, no int8, hours instead of days. It is
   what showed group-wise weight scales recover 6% before a line of the kernel
   existed. The same habit killed a radix-partitioned scatter and an incremental
   KV cache after they measured worse than what they replaced.

2. **Every check gets a control that must fail.** A test both arms pass measures
   the harness. The 4-bit arm exists to prove the accuracy harness can see
   damage; the stock arm exists to prove the invariance harness can see
   divergence. Both caught real problems — the invariance control silently passed
   at 24 tokens and needed 160.

3. **Mutate the tests, not just the code.** Introduce the bug deliberately and
   confirm something goes red. Three of eight mutations survived a fresh
   all-green test file at one point, each for a different reason. And when a
   mutation survives, decide whether it is a hole or a *confirmation* — one here
   turned out not to be a bug at all.

4. **Attribute; do not rank.** A ranking says which option is best among those
   tried. An attribution says where the cost actually is, and it needs the
   single-factor arms. Without a weights-only and an activations-only arm, the
   grouped arms' failure reads as a tuning problem instead of the wrong lever
   entirely.

5. **Write predictions down, then check them.** Carry-flag intrinsics were
   predicted at 2x and measured at 1.06x. A cache-regime hypothesis did not
   survive contact with the real model. The digit split was predicted "nearly
   free" and cost 5.9%. Logging the prediction is what converts a wrong guess
   into a transferable lesson rather than a quietly revised story.

6. **Refuse rather than approximate.** Where a bound cannot be met, the compiler
   stops with the reason and the limit. Silently doing something plausible
   instead is the single failure mode this codebase has found in itself the most
   times — a clean build, a confident artifact, and a different function than the
   source describes.

---

## What I would do with the following week

- **Build the fixed-order-float arm.** It is the missing control and it could
  invalidate the framing. If it wins, the honest conclusion is that this work
  bought portability and tuning freedom rather than determinism — which is still
  a real product, but a different pitch.
- **Run a second GPU.** Cross-hardware bit-identity is the strongest thing
  exactness buys and the only headline claim with no evidence behind it.
- **Take it to an RL team before optimizing further.** The remaining engineering
  is a serving integration, and integrating against a hypothetical customer is
  how the last two levers ended up being framework costs rather than arithmetic
  ones.

---

*All figures from the repository's own harnesses. Perplexity and multiple-choice
scores are self-computed and comparable between arms only — never against
published leaderboards. Run-to-run throughput spread is 2–3%, so ratios are
quoted to two figures.*
