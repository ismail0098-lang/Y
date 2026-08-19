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

**The bridge has not been re-run: that needs the GPU.** The fix is arithmetic
and the arithmetic is checked on the CPU, but the 12/12 line above should be
read as unverified until it runs again with a non-uniform softmax underneath
it.

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
