# Translation validation for `ptxas`

Y's proofs stop at the IR it emits. `proofs/` establishes that the exact GEMM's
schedule computes the source dot products; `tests/ptx_portability.rs` establishes
that the emitted PTX assembles at six architectures. Neither says anything about
the machine code, and the repository has said so in the one place a user reads:
the trust boundary printed into every emitted certificate names

> Everything below the LLVM IR this compilation emitted: `clang`, its optimiser,
> the assembler and the linker.

as `NOT CHECKED`, and names the remedy in the same breath — *"translation
validation — checking THIS object against THIS IR per compilation, which is not
performed."*

`tools/ptxas_tval/` performs it, on the GPU side. It symbolically executes a
kernel's PTX and the SASS that `ptxas` produced from that exact file, and asks an
SMT solver whether the two can ever store different values to memory. A proof
covers one compilation of one kernel, which is the point: it removes `ptxas` from
the trusted base for that artifact without anyone having to trust `ptxas`, model
its passes, or read its source.

**The CPU trust item stays open.** This is `ptxas`, not `clang`; the technique
transfers, the result does not.

---

## What is validated today

Measured on 2026-09-07, on this machine (RTX 4070 Ti SUPER, sm_89, CUDA 13.3,
z3 5.0.0), reproduced from a clean `tools/ptxas_tval/` by `./regress.sh`, which
ASSERTS every row below in the direction it reads and exits non-zero if any of
them moves — the two UNPROVED rows included.

| kernel | verdict | obligations | time | what makes it interesting |
|---|---|---|---|---|
| `fma/rn` | **VALIDATED** | 9 | 0.0 s | float, with contraction forbidden by `.rn` |
| `fma/plain` | UNPROVED | 10 | 0.0 s | **the negative control** — `store 0: sat` |
| `neg/folded` | **VALIDATED** | 9 | 0.0 s | a `neg.f32` folded into an `FFMA` modifier |
| `neg/sub` | **VALIDATED** | 7 | 0.0 s | a plain float subtract |
| `neg/unfoldable` | UNPROVED | 9 | 0.0 s | **a second control** — the *other* lowering of one opcode |
| `max/relu` | **VALIDATED** | 5 | 0.0 s | the **shipped ReLU** shape — needs FMAX commutativity |
| `max/general` | **VALIDATED** | 7 | 0.0 s | the same opcode with the operand order *preserved* |
| `max/min` | **VALIDATED** | 7 | 0.0 s | the other polarity of the same SASS instruction |
| `bn254_permute` | **VALIDATED** | 30 | 0.2 s | branching `ptxas` invented |
| `bn254_sub_vec` | **VALIDATED** | 88 | 12.6 s | |
| `ptx_carry_chain` | **VALIDATED** | 123 | 33.4 s | 24 predicated instructions |
| `ptx_subword_ops` | **VALIDATED** | 31 | 0.3 s | **sub-word stores**, and a load that can read one back — measured 2026-09-15 |
| `ptx_integer_ops` | **VALIDATED** | 65 | 23 s | **u32 `div`/`rem` through the float unit**; 3 stores proved over Int — measured 2026-09-19 |
| `exact_pv` @ `-O1` | **VALIDATED** | 14 | 1.1 s | across a **loop**; 1 multiplier identity assumed |
| `smem_roundtrip` | **VALIDATED** | 18 | 0.2 s | **shared memory**, 1 barrier |
| `naive_gemm_f32` @ `-O1` | **VALIDATED** | 9 | 0.2 s | **a shipped GEMM** — the emitter says `fma.rn.f32` |
| `naive_gemm_f32_muladd` @ `-O1` | UNPROVED | 7 | 0.2 s | the form Y used to ship — `store 0 value: sat` |
| `naive_gemm_f32_rn` @ `-O1` | **VALIDATED** | 9 | 0.2 s | the contraction *forbidden*, at a different SASS |

Fifteen kernels validated, **457 obligations**, and three UNPROVED rows that
are results rather than gaps. `bn254_fr_mul_fast` and `bn254_ntt4_fused` are
UNPROVED and are discussed under *The wall* below — neither produced a `sat`.

The last three rows are one experiment: one kernel, three PTX spellings.
Y **used to** emit `mul.f32` then `add.f32` — two roundings — and `ptxas`
contracts them into a single `FFMA`, which rounds once. The validator refuted
that, and *where* it refuted is the informative part: `BASE`, `STEP`,
`LOOPCOND` and `ENTRY` all proved, so the loop schedule corresponded exactly
and it was the accumulated **value** that could not be shown equal.

Both repairs validate, and they are not equally good:

* `mul.rn.f32` + `add.rn.f32` **forbids** the fusion. Costs 0 to +7.1%
  instructions, and leaves the kernel with two roundings where the hardware
  does one. It is also the arm that needs `FADD` commutativity, because
  `ptxas` sorts the addends.
* `fma.rn.f32` **states** it. Emits a **byte-identical instruction stream** to
  the form it replaces, is more accurate, and validates.

**The compiler emits the second one now** (`try_emit_fma` in
`src/ptx_emitter.rs`, gated by `tests/fma_contraction.rs`): a source-level
`a*b + c` over F32 becomes one `fma.rn.f32`, so the shipped artifact and the
validated one are the same file. `a*b − c` and `c − a*b` do too, with the sign
on the operand each shape requires — `fma(a, b, −c)` and `fma(−a, b, c)` — and
that `neg.f32` replaces the `mul.f32` the `fma` absorbs, so it is free in
instructions as well as in SASS. The refutation is kept as `_muladd`, derived
from the shipped kernel by splitting the instruction back into two — a corpus
containing nothing the validator refutes cannot be told apart from a validator
that always says VALIDATED.

### The validator's effect model had six blind spots

Every row above is a claim about **stores** — where, under what guard, what
value — and so is every obligation. The model of *when* an effect happens and
*what* a load reads was never itself checked. Asked, it could not see an effect
in six places, and **each was demonstrated by a translation built by hand from
`ptxas`'s own output that the unmodified validator VALIDATED**. All six are
closed and all are asserted by `regress.sh` from `tools/ptxas_tval/mem/`, whose
`.ptx` headers say what each fixture is and what it used to get.

| blind spot | wrong translation that VALIDATED before | now |
|---|---|---|
| a global load always reads the *initial* memory | `las_sass` (the SASS moves a store above a load that can read it back); `las_ptx` (the SASS hoists a load above the PTX store it could read) | REFUTED (`store 1: sat`), each side on its own row — REFUSED until the store-ordered model below |
| stores paired by address may land in any order | `swap_alias` (two output pointers, swapped); `swap_off3` (+0 and +3, swapped); `loop_swap_wrong` (the same in a loop's epilogue) | UNPROVED — one disjointness obligation per reordered pair, `sat` |
| `loopval` compares the epilogue's stores only | `pstore_wrong` (a store before the loop writes a different value) | REFUSED |
| an `EXIT` never crosses a region boundary | `loop_swap_exit` (the zero-trip guard `EXIT`s past both epilogue stores); `loop_body_exit` (the body `EXIT`s instead of iterating); `loop_ret_wrong` (the SASS drops the body's early exit) | REFUSED |
| a PTX `ret` is a no-op | `pret_wrong` (the SASS drops a predicated early exit) — and the **correct** `pret` came back UNPROVED | `pret` **VALIDATED**, `pret_wrong` UNPROVED |
| a loop kernel has something to prove | `loop_nostore` ("0 stores") | REFUSED |

Ten distinct wrong translations. `batch.validate` — the third validator that
pairs global stores, reached through `smemval.py` — VALIDATED three of them as
well (11 obligations each), so its two new checks are driven by rows of their
own rather than inherited from the other two.

**All sixteen standing rows are unchanged**, verdict and obligation count, 361
in total — diffed before and after each change rather than assumed. That is not
coverage, and the reason is measured: no standing row has a global load after a
store on either side, every store pairing in every row is the identity, and the
four loop rows store nothing outside the epilogue. **The blind spots were
latent**, which is the argument for closing them while nothing depends on them.

**`ptxexec` stated the first one as a fact about kernels** — *"a kernel never
reads back what it wrote to global memory in the same launch"* — and nothing
checked it. As an ordering property it is false of kernels in the corpus — a
global load after a store in program order, or both inside one loop body —
`y_cpu_matmul` among them, and identically at `-O1` and `-O3`. What was true is
narrower: no *validated* kernel had one. That is checked now instead of stated.
(No count is given here on purpose: it was measured by a one-off scan, and a
figure in this document that no gate re-derives is how a count goes stale.)

**The PTX early return inverted a verdict pair, not merely weakened one.**
`ptxexec` read `ret` as `pass`, so the specification side made every store after
a predicated return unconditional: the correct translation was refuted and the
wrong one proved. It is modelled now the way `sassexec` already modelled `EXIT`
— an `alive` term conjoined into the guard at the one place a guard is
computed, and left untouched while it is true, so a kernel with no early return
builds exactly the terms it did before.

> **Paid, 2026-09-15.** `lsls` VALIDATES and both wrong twins are refuted with a
> counterexample: a global load reads memory as updated by the stores before it,
> byte by byte. See *The memory model reads through stores* below. `pstore` and
> `loop_ls` do **not** move, and measuring why is part of that section — neither
> was ever waiting on the memory model alone. The paragraph is kept as written.

**The refusals have a price, and it is stated as standing rows rather than a
footnote.** `lsls` is `ptxas`'s *correct* output for `las_ptx`'s PTX — it kept
the store above the load it could not prove unaliased — and it is REFUSED,
because a model that reads every load from the initial array cannot tell it from
the wrong one. Before this, it validated both. A store-ordered memory model is
what would turn `lsls` green while `las_ptx` stays red. `loop_ls` and `pstore`
are the same price paid in the loop validator.

**The order is recorded by the list type, not by each executor arm.** A load or
store appended at an arm that forgot to log its position would make the
read-back check vacuous for exactly that opcode — the guard-consulted-at-one-site
bug — so `memorder.py` replaces both lists with a type that records on `append`
and refuses every other mutation, and self-checks at import.

### The memory model reads through stores, and one corpus kernel needed it

A global load reads the memory **as updated by every earlier store on its own
side** (`memorder.read_through`), byte by byte, and a store carries its width as
its value's width. So a load hoisted above a store it could read back builds a
*different* term from the load below it, and the store value it feeds is
**refuted** rather than refused. `mem/lsls` — `ptxas`'s correct output, and the
price the refusal paid — **VALIDATES**; `las_ptx` and `las_sass` come back
`store 1: sat`. The straight-line validators no longer refuse a read-back.
`loopval` still refuses one that crosses a region boundary, because it executes
regions separately and a region's store trace starts empty.

**Every standing row builds byte-identical terms**, and that is measured, not
argued. The executors' terms were fingerprinted per row, one process per row,
against a `git archive` of the previous commit: **31 of 34 rows identical**, and
the 3 that differ are exactly the read-back fixtures whose terms are supposed to
change. With nothing stored yet `read_through` returns its base unchanged and
builds no z3 node, and every standing row loads before its first store.

**Getting there found a hazard the fingerprints could see and no verdict can.**
The first version moved **nine** standing rows' fingerprints without one
executor line changing. The cause was `memorder`'s import-time self-check: it
built and freed more z3 nodes, z3 reuses a freed node's number, and the multiply
primitive and the float functions order their operands by that number — so a
self-check edit reorders commutative operands in unrelated kernels' terms. With
the self-check removed on both sides, 31 of 34 matched. The check's historical
main-context allocation is kept as it was and everything added since runs in a
**private z3 `Context`**, so a future self-check edit cannot perturb a proof term.
The same mechanism as *Two censuses in one interpreter changed a verdict*, reached
through an import rather than a preamble.

**The first working version was 300x slower on the one row it was for.** A load
after a store is an ITE over byte-address differences, and with each side's own
address terms the solver re-proves every address equality inside it: **8–11 s**
for `lsls`'s one store value, against the 15 s refinement budget. The pairing
phase had already proved each SASS load and store address equal to its PTX
partner's, so the SASS side's read-through is built from those addresses
(`abstract_addr`): **0.03 s**. `regress.sh` asserts `lsls` at **exactly 10
obligations** — the eleventh is the direct-multiply refinement, which runs only
when the abstraction cannot discharge the read-through, so the count is the
structural form of that performance property.

#### The one corpus kernel it reaches, and what that needed

Priced before building. None of the eight straight-line kernels the frontier
calls clear has a read-back on either side — measured by running the executors —
so **the store-ordered model alone validates no corpus kernel.** The loop kernels
with a read-back also need the store-in-body lift. The one straight-line kernel
whose need it *is* is `ptx_subword_ops`: its PTX loads `A8` again after a byte
store to `OBack8`, and it was blocked on three opcodes — `cvt.u8.u32`,
`st.global.s8`, `STG.E.S8` — that the width-less model could not make sound. With
widths it can, and three more things were needed:

* **Two device facts, refereed by `subword_abi.py`.** `STG.E.{U,S}{8,16}` write
  exactly their low bytes, little-endian, and nothing else: 192 stores (two
  probes, 32 lanes, 3 stores each) into poisoned words at byte offsets inside a
  lane, 0 disagreements. `cvt.u8.u32`
  **truncates**: 32 vectors, 12 of them separating truncation from saturation,
  0 disagreeing. That PTX opcode has no executable definition other than running
  it, so for it a `ptxas` bug is in the trusted base, as for `max.f32`.
* **Loads at one proved address share one base symbol.** The kernel reads `A8[i]`
  twice, and the pairing refused `load 0 address matched 2`. That refusal was
  right when a pairing chose which initial-memory symbol a load got; it is not a
  choice now, because the base stands for the initial memory *at an address* —
  one value however many loads read it — and the read-through supplies whatever
  was stored in between (`memorder.pair_by_address`).
* **Stores of different widths are refused as a pair**, rather than crashing on a
  z3 sort mismatch.

**It VALIDATES, 31 obligations**, and its two wrong twins — built by hand from
`ptxas`'s own SASS — are asserted in their own directions: `mem/subword_hoist`
(the read-back load hoisted above the byte store) is **refuted, `store 6: sat`**,
and `mem/subword_widen` (the byte store widened to a word) is **UNPROVED on the
store width**. `regress.sh` grows to 42 rows — the forty-second is below.

#### A device fact the model does not describe

A 16-bit store at an odd byte offset **faulted** in the first referee probe, so
the referee now opens with an alignment census, one kernel and one process per
case because a fault is sticky in a CUDA context:

```
st.global.u8   offset 0..3:    ok    ok    ok    ok
st.global.u16  offset 0..3:    ok FAULT    ok FAULT
st.global.u32  offset 0..3:    ok FAULT FAULT FAULT
ld.global.u8   offset 0..3:    ok    ok    ok    ok
ld.global.u16  offset 0..3:    ok FAULT    ok FAULT
ld.global.u32  offset 0..3:    ok FAULT FAULT FAULT
fault messages: ['misaligned address']
```

An access of width *w* runs iff its address is a multiple of *w*, on sm_89 —
asserted, not printed. **The executors model no fault**, for any width, and
never did. So `mem/off3` and `mem/swap_off3`, whose headers described two word
stores at `+0` and `+3` ending with different bytes, describe programs that
**fault** on the device: they pin the reorder obligation's arithmetic, not a run.
Their headers say so now. Mixed widths can overlap legally — a byte store inside
an aligned word — and the model states that case too.

#### A pairing order nothing pinned

Loads in an equal-address group share one base, so the order their members pair
in does not matter. **Stores are different**: pairing two stores at one address
in reverse compares the second store's value with the first's, so a correct
translation goes UNPROVED — soundness holds (the reorder obligation still refuses
a reversed same-address pair), completeness does not. No row had two stores at
one address, and the mutation that reverses a group's pairing **survived the
first table**. `mem/sls_alias` closes it: store to `P2[i]`, load `P3[i]`, store to
`P2[i]` again. `P3` may alias `P2`, so `ptxas` can neither forward the store into
the load nor delete the first store as dead, and its genuine output keeps both.
It VALIDATES, and with the pairing reversed it is `UNPROVED — stores 0 and 1 are
REORDERED and may overlap`.

#### What the queue had conflated

The queue said the store-ordered model was "what would make `lsls` / `loop_ls` /
`pstore` validate". Measured: it turns **`lsls`** green. `pstore` has **no load**
at all — its header says so — so no memory model can be what it needs; it needs
prologue stores *compared* rather than refused. `loop_ls` is refused on a
read-back that crosses an iteration, which is the store-in-body lift.

#### A latent dropped operand, found on the way

Every one of the corpus's 2,879 `LOP3.LUT` carries a sixth operand, `!PT`, and
`sassexec` read nothing past the fifth. Its meaning has never been exercised — the
exact condition under which a dropped operand is invisible — so any other value
is refused by name now, pinned at import because no fixture can reach it: the
`BRA P1, label` shape, one opcode over. (The 61 seven-operand hits are
`PLOP3.LUT`, a different, unmodelled opcode.)

### The three `neg` rows are one opcode in two lowerings

`neg/folded` and `neg/unfoldable` contain the same PTX instruction. One
validates and one does not, and the difference is entirely what `ptxas` does
with it.

`ptxas` has no bare float-negate instruction. Where the negation feeds another
float operation it disappears into an **operand modifier** — `FFMA Rd, Ra, Rb,
-Rx` — and that is what happens in all 23 occurrences in the committed corpus.
Where it cannot (`neg/unfoldable` sends the result straight to a store) `ptxas`
materialises it as `FADD Rd, -Rx, -RZ`, which is **arithmetic**.

The two are not the same function, and the device says so. Measured on sm_89
over denormals of both signs, `±0.0`, quiet NaNs carrying payloads, a
signalling NaN, both infinities and the all-ones pattern: the modifier is a
bit-exact sign flip on **32/32**; the un-foldable lowering agrees on every
finite input and returns the canonical quiet NaN `0x7fffffff` for **every** NaN
— discarding the payload *and* the sign — on 10 of 32 vectors. So the UNPROVED
row is a fact about the machine rather than a limitation of the model, and it is
a genuinely different refutation from `fma/plain`: not a contraction `ptxas` is
free to make, but one PTX opcode with two lowerings that compute different
functions.

`neg/sub` is the third of the set because a plain `sub.f32` lowers to `FADD Rd,
Ra, -Rb` — the same modifier. It is the shape eleven corpus kernels contain.

### The three `max` rows, and why two of them are not a spare

`max/relu` is the shape the shipped bias+ReLU epilogue emits:
`max.f32 r, r, 0f00000000`. `ptxas` folds the literal into `RZ` and puts it in
the **first** operand slot — `FMNMX d, RZ, x, !PT` — so the two sides build
`FMAX(x, +0.0)` and `FMAX(+0.0, x)`, and without a commutativity fact they are
two terms that never meet.

`max/general` is the same opcode on two runtime values, and it is **not** a
redundant control. Measured: with the canonicalisation removed, `relu` goes
UNPROVED and `general` stays **VALIDATED**, because `ptxas` preserved its
operand order. So the commutativity measurement is load-bearing for exactly the
shipped shape and for nothing else in the corpus — a general-max fixture alone
would have validated and left the need invisible.

`max/min` is the other polarity of the same SASS instruction. `FMNMX` carries
min-vs-max in its fourth operand, so a polarity error is a whole-kernel wrong
answer rather than a modelling gap; the pair is what pins that the executor
reads that operand instead of assuming.

These validate the shipped **lowering**, not the shipped kernel: the six
`gemm_f16_bias_relu_*` around it are 11 PTX and 14 SASS opcodes away, and the
four paged-decode kernels 5 and 24–26.

### The control is the row that makes the table mean something

`fma/plain` and `fma/rn` are the same source. `plain` writes `mul.f32` followed by
`add.f32`; `rn` writes `mul.rn.f32` and `add.rn.f32`. `ptxas` contracts the first
pair into a single `FFMA`, which rounds once where PTX rounds twice, and the
validator answers `sat` with a counterexample. That is not a bug in `ptxas` —
contraction is a freedom the PTX ISA grants unless the program forbids it — and it
is exactly what a validator that always says VALIDATED would also report as fine.

Every result above is worth what that row is worth. A validator with the float
guard removed reports `fma/rn` VALIDATED for the wrong reason; `./gmut.sh` is the
table that shows it.

---

## Method

Two symbolic executors over one shared z3 vocabulary:

- `ptxexec.py` — PTX. Virtual registers, predicates, `.shared`/`.global` address
  spaces, the guard under which each memory access happens.
- `sassexec.py` — SASS as `nvdisasm` prints it. Physical registers, uniform
  registers, predicate registers, `ULDC` constant-bank loads, `LEA`/`IMAD`
  addressing, `.X16` address scaling.

Each side yields a list of *effects*: `(guard, address, value)` per store, plus
the loads it performed. The obligations are then

1. the two sides perform the same number of stores and of loads;
2. for each store, the guards are equivalent, and **where the guard holds** the
   addresses and values are equal;
3. loads at equal addresses under equal guards return equal values (this is what
   lets an uninterpreted memory be shared rather than axiomatised);
4. with shared memory: barrier counts pair, the shared array entering each
   barrier is equal on both sides, the array at exit is equal, and every access
   is 4-byte aligned.

Point 2 is stated more carefully than it looks. A guarded store is
`if g then M[a] := v`, so equal effect means equal guards plus equal `(a, v)`
*where `g` holds* — not equal addresses unconditionally. Matching addresses
unconditionally was an over-strong obligation this repository shipped for weeks;
it went unnoticed until a kernel had a predicated access whose address is
computed after a path merge, where `ptxas` turns `@%p2 st.global` into an early
`@P0 EXIT` and every later register carries `If(P0, stale, new)`. The two
addresses then provably differ on the branch where *neither side stores*.

### An unmodelled opcode is a hard error

Never a guess, never an identity, never a nearest neighbour. This is the same
design rule the compiler itself enforces (`CLAUDE.md`'s table of `_ =>` arms that
silently substituted something plausible), applied to the validator: a symbolic
executor that guesses at an opcode it does not know produces a proof about a
program nobody wrote.

The cost is visible in the census below — most of the corpus is refused, by name,
at a specific opcode. That is the intended shape.

### The representation ladder

A 64-bit multiply is what the solver chokes on, so the multiplier has three
settings, and which one a query needs is itself a result:

- `uf` — an uninterpreted function. Cheapest, weakest.
- `wide` — one shared uninterpreted `MUL64`, so both sides agree on products
  without the solver reasoning about multiplication.
- `direct` — a concrete bitvector multiply. Strongest, and often intractable.

`exact_pv` validates under `wide` in 1.0 s and is UNPROVED under `direct` at
121 s. The abstraction is what makes it go — the opposite of the usual complaint
that an abstraction is too weak.

---

## The wall, and what is actually blocking this

Every scope census here counted *opcodes*, which silently assumes the executors
are what gate the corpus. The standing table says otherwise: `ptx_carry_chain`
has 29 multiplies and validates in 24 s; `bn254_fr_mul_fast` has 65 and is
UNPROVED with **no `sat`**: at default budgets its first sweep closes 17 of 276
partial sums in 16,237 s.

> **This paragraph used to say "UNPROVED after 9,705 s with 261 of 276 cut points
> closed", and that does not reproduce.** Re-measured 2026-09-19 with today's
> `tval.py` (first sweep: 17 of 276 in 16,237 s) and with the `tval.py` of
> `b427333`, the commit that published it, on byte-identical PTX and SASS: still
> in its first sweep at 9,810 s. The two builds' direct-mode terms are
> byte-identical; their `wide`-mode terms differ in let-binding numbering with sizes
> within 5 bytes of 43 MB -- consistent with commutative operand order, not checked
> term for term -- so this is not a regression of the validator; the budgets or artifact of
> the original run were not recorded. The VERDICT stands, and it is the one
> `wall.py` uses -- and this row does not set a threshold anyway: the smallest
> region measured `unknown` is `bn254_ntt4_fused` barrier 1, at 49.

> **"A solver wall between 29 and 65 multiplies per query" was a KERNEL-level
> reading and it was re-quoted as current long after the region-level measurement
> under it said otherwise.** `bn254_ntt4_fused` has barrier regions of 33, 49, 193,
> 193, 193 and 225 PTX multiplies; its barrier 0 PROVED and its barrier 1 was
> `unknown` at a 600 s budget — re-measured 2026-09-17, 731.6 s, identical. Every
> ground-truth multiply is register × register; the GEMMs' integer multiplies are
> almost all index arithmetic by an immediate, which is linear for a solver.

The measured bracket is a solver wall between **33** and **49** symbolic integer
multiplies per barrier region — derived in `wall.py` from named ground truth
rather than written down, and gated by `docgate.py`.

`tractable.py` asks the counterfactual — *if every opcode were modelled, how many
kernels could the solver close?* — using barriers as cut points. It is
three-valued now, because a region between the largest measured PROVED and the
smallest measured UNKNOWN is on neither side of anything measured:

**25 UNDER, 29 UNDECIDED, 10 PAST, 2 REFUSED** (two entry points).

> It used to read **"51 of 66 fall under the wall. 15 are over it."**, against a
> transcribed `WALL = 65` and every multiply counted. `fma.` was not counted at
> all (11 kernels under-counted, optimistically), and the two split paged-decode
> modules were counted as one program across both entry points.

> **The paragraph below is RETRACTED as a claim about the solver.** The GEMMs'
> 39–61 are almost all multiplies by an immediate; nothing of that shape has been
> measured at the wall in either direction, so every one of them is UNDECIDED, not
> tractable. What survives is the field-kernel half: all ten PAST kernels are
> field arithmetic.

And the ranking inverts. The 23 FP16 tensor-core GEMM kernels look like the
deepest bucket — five unmodelled features each — and are the **tractable** one:
8–12 barrier regions, worst region 39–61 multiplies. The field-arithmetic kernels
look shallow (`bn254_ntt4_fused` needs shared memory and nothing else) and are
the intractable ones: 244–717 multiplies per region, 3.8× to 11× over.

That measurement cancelled the feature it was taken to justify. "33 kernels are
behind shared memory" was false — shared memory alone unlocks exactly **one**,
a 64-instruction test fixture. It was still the right thing to build, for a
different reason: it is the prerequisite for the 23 tractable GEMMs *and* the
thing that creates the cut points that put them under the wall.

#### ...and this wall IS the solver, not the theory

The division tail was `unknown` over bitvectors on six posings and `unsat` in
0.2 s over Int, so the obvious question is whether the field wall is the same
artifact. `intwall.py` asks it on tval's own partial-sum obligations: each of the
first N proposed pairs, in sweep-1 conditions, of the bitvector engine and of the
exact Int translation, at one budget.

    bn254_fr_mul_fast, 40 pairs, 60 s each      (intwall.py, 2026-09-19)
      both unsat           23
      both unknown         14
      bitvector unsat,
        Int unknown         3
    Int closes 0 pairs the bitvector engine does not

**It is not.** The division tail reasons about the bounds of a few products; a
CIOS carry chain is many products whose low and high words must be matched
exactly, which is bit-level reasoning in either theory. So the Int rung stays
where it is -- first for estimate obligations, last on `unknown` for stores --
and is not added to the partial-sum sweep, where it would cost 60 s per hard pair
and buy nothing measured.

The Int rung's answers are **run-dependent**: the exploratory run and
`intwall.py` both lost three pairs, but not the same three -- pair 7 was `unknown`
in the first and `unsat` in the second, pair 12 the other way round -- on the same
obligations. The two processes had built different terms first, and this directory has already
recorded that z3 node ids -- which order commutative operands and key the
encoder's memo -- depend on construction history. A three-pair "Int loses" count
is therefore not a stable figure; the zero in the last line is the claim.

**And that zero is a null metric until the Int engine is shown live**: it is also
what the tool reports if the Int query fell back to the bitvector one, or if
`intenc` refused every formula (it answers `unknown` then). So `intwall.py` counts
the Int queries it made and requires them to equal the pairs asked, and requires
the Int engine to prove at least one pair by itself. `iwmut.sh`, control row
first and BASE at both ends: the Int query replaced by the bitvector one, the Int
engine answering `unknown` to everything, tval's anchor moved (it must refuse
rather than run the unpatched validator), and the pair rows never recorded each
fail by name; **the compound that removes the proved-something check and silences
the engine is green**, which is that check's justification.

### `sat` and `unknown` are not the same result

`sat` means the two programs provably can differ — a refutation, and a finding.
`unknown` means the solver ran out of time and says nothing whatever about the
kernel. The validator printed one message for both until this was noticed, and
the first thing it did afterwards was change what `bn254_ntt4_fused` means:

    barrier 0   PROVED EQUAL          12.9 s
    barrier 1   unknown              600.8 s

That is a wall. Under the old reporting it would have been recorded as *"shared
memory differs entering barrier 1"* — that is, as `ptxas` miscompiling a shipping
kernel.

---

## Where the corpus stands

`scope2.py`, re-run 2026-09-04 over all 66 kernels the repository ships PTX for:

```
PAST BOTH EXECUTORS: 8 / 66

   23  cp.async.cg.shared.global      the FP16 tensor-core GEMMs
   12  bra                            multi-block control flow
    8  (past both executors)
    6  crash: invalid literal for int()   a residual PARSER defect, not a gap
    5  more than one .shared array
    4  I2F.U32.RP                     32-bit integer division, via the float unit
    3  cvt.rn.f32.s32
    2  IABS
    3  cvt.u8.u32 / cvt.f64.f32 / ld.global.ca.f32
```

The six-kernel `invalid literal for int()` bucket is a **crash**, not a
refusal, and it is listed as a defect rather than as a blocker. A crash in a
census reads exactly like a missing feature, which is how two earlier crash
classes hid: `'NoneType' object has no attribute 'group'` was five unguarded
inline address parsers, and an `IndexError` was one `LEA` arity of two. Closing
those took the corpus from 7 past both executors to 8 — a census taken before the
fix under-counts, and this one supersedes it.

### A first-refusal census names one opcode and says nothing about depth

`depth.py` lists each kernel's whole opcode alphabet against what a passing
kernel uses, and it overturned the obvious read:

- `rope_64/128/256` look one transliteration away and are the **deepest** float
  kernels in the corpus — 16–18 unknown PTX ops each, three MUFU identifications
  (none device-validated) and f16 pack/unpack. FFMA contraction used to be on
  that list and no longer is: the rotation states both halves of its own
  fusion, so those kernels are now exactly as deep as their opcodes.
- **Built 2026-09-15, and it VALIDATES** — see *The memory model reads through
  stores*. Kept as written:
  `ptx_subword_ops` is the cheapest kernel left: 8 unknown PTX ops, all integer,
  no float, no branch, no loop, no shared memory. **Both halves of that are
  wrong, and it is measured below** — the dynamic gap is *three* opcodes, not
  eight, and the SASS side does branch. It is also the wrong kernel to build.
- `ptx_integer_ops` yields a **finding** rather than a kernel: `ptxas` implements
  32-bit `div.u32`/`rem.u32` through the *float* unit — `I2F.U32.RP`, `MUFU.RCP`,
  `F2I.TRUNC`. An integer PTX operation lowered as a floating-point macro-op.
  **Refereed on the device since 2026-09-18, and VALIDATED since 2026-09-19** —
  see *…and it was past the BITVECTOR solver*. The 2026-09-18 note read "its
  obligation is past the solver"; it was past z3's bitvector engine only.

### The real gap, measured by running the executor

`depth.py` over-states **by construction** — its own docstring says so. It marks
an opcode unknown if no *passing* kernel uses it, so `add.u32`, `mad.lo.u32` and
`IMAD.WIDE.U32` all appear in its list although they are modelled. `gap.py` gives
the other number: it runs the executor and, on an unmodelled opcode, records it
and **skips**, so what it reports is what the validator genuinely refuses.
Skipping is unsound for validation — the state afterwards is not the kernel's —
so the set is a *lower bound*, and the operand errors it drags behind it are
contaminated by the skipping and reported in a separate column, never mixed in.

Three measures of the same kernel, `gemm_f16_256`:

```
depth.py   static, over-states        27 PTX ops   29 SASS ops
gap.py     dynamic, the real set       9 PTX ops   13 SASS ops
"five unmodelled features"             5 PTX families
```

They are different measures, not disagreements — and the smallest is the one
easiest to quote. Across all 23 FP16 tensor-core GEMMs the gap is **21–27
opcodes each**, and every one of the 23 needs all of:

```
PTX    cp.async.cg.shared.global / .commit_group / .wait_group     23 kernels
       ldmatrix .m8n8.x2.trans / .m8n8.x4                          23
       mma.sync.m16n8k16 + wmma.store.d                            23
       bra          -- every one of these kernels has a loop       23
       st.global.v4.f32                                            19
SASS   LDGSTS.E.BYPASS.128 / LDGDEPBAR / DEPBAR.LE                 23
       HMMA.16816.F32                                              23
       I2F.U32.RP / MUFU.RCP / F2I.FTZ.U32.TRUNC.NTZ               23
       S2UR / UIMAD          -- the uniform datapath               23
       IMAD.MOV / IMNMX.U32 / WARPSYNC / CS2R                   20-23
```

So **`cp.async` is a prerequisite, not the gate.** It is three opcodes of nine on
the PTX side; and on the SASS side none of the 23 kernels ever reached it — they
refused earlier, on a const-bank operand. Earlier notes recorded it as "the next
feature in front of the tractable bucket, where all 23 now refuse". That is true
of the PTX side and of first-refusal reporting, and it reads as *one feature
away*, which is wrong by an order of magnitude.

### The front of the queue is a naming assumption — and it was measured, then cancelled

The PTX executor makes two lexical assumptions that have nothing to do with
semantics:

- predicates must be named `%p<digits>` — `re.match(r'^@(!?%p\d+)')`. The
  coprocessor kernels use `%qp0` and `%rt_p0`, so the whole predicated
  instruction is read as an unknown opcode.
- registers must be named `%r<n>` / `%rd<n>` / `%f<n>` — the register file is
  keyed by `int(name[2:])`. A named virtual register like `rt_A_ptr` gives
  `int('A_ptr')`, which is the `invalid literal for int()` crash class above.

Both fail closed, and neither is a missing semantics. They are one repair, not
two: the register file has to become **name-keyed**, with the file taken from
the `.reg` declarations rather than from the spelling — a kernel may write
`.reg .b32 %rt_r<100>;` or `.reg .b64 rt_A_ptr;`, the second with no `%` at
all, and `ptxas` accepts it.

**It was built as a probe and the measurement cancelled it.** Keying by name and
resolving the file from the declarations keeps every existing proof term
byte-identical (the undef symbol is the name minus `%`, so `%r5` is still
`ptx_undef_r5`) and all seven standing results reproduce unchanged. What it buys
is the question:

```
hello.coprocessor      clean on both sides -- and it stores NOTHING
coprocessor_test       still needs `bra` (a loop); SASS has its own int() crash
coprocessor_attention  needs a loop, F2FP.F16.F32.PACK_AB, shared address forms
```

`hello.coprocessor` was the kernel with a zero opcode gap on both sides, and it
turned out to be the **empty-artifact** kernel this repository already documents:
`--emit-coprocessor` with 0 RT and 0 Tensor nodes emitting a `ret;` body under a
fixed parameter list. It was closest to passing **because it does nothing** —
`tractable.py` counted it under the solver wall for the same reason, at zero
multiplies. Validating it would have been the `fma/plain` lesson in reverse: a
result a validator that always says VALIDATED reports identically.

So the refactor was reverted rather than landed. It also does not come free —
`loopval.py` recovers a region's live-ins by parsing the undef symbol *name*, and
with arbitrary names that encoding is ambiguous (`ptx_undef_rt_rd6` reads as
kind `r`), so doing it properly means threading the declaration map through
`run_lines` and the selector machinery. Paying that to reach a kernel with no
stores is the wrong trade. **Recorded, with what it costs and what it buys, so
the next reader does not re-derive it.**

### The probe condemned a committed artifact, and the gate for that had a blind spot

`tests/hello.coprocessor.ptx` was checked in, and the compiler **refuses to emit
it**: *"this source has no RT Core work and no Tensor Core work, so there is
nothing to fuse."* That is the third staleness class this repository names — an
artifact no run of the compiler can reproduce — and there is a gate for exactly
it, `every_committed_artifact_still_has_a_source_that_compiles`.

The gate pairs an artifact with its source by `with_extension("ysu")`. For
`hello.coprocessor.ptx` that asks for `hello.coprocessor.ysu`, which does not
exist, so it was skipped — **and so were all seven `*.coprocessor.ptx`**, because
they are emitted by a different backend and name their source `<stem>.ysu`. A
gate written for the stale-artifact class, with a blind spot created by a
filename convention. Extended to strip `.coprocessor` and to pass
`--emit-coprocessor`; it then failed on exactly one of the seven, naming the
source, the flag and the backend's own reason. The other six compile.

`build_corpus.sh` had the same shape one layer down: it wrote without cleaning,
so a source deleted from `tests/` left its artifact behind and the corpus still
reported 67. It removes `corpus/` and `o1/` first now — found by deleting the
file and watching the count not move.

**What moved, and what did not.** The corpus is 66; the 66 survivors rebuild
**byte-identically**. `8 / 66` past both executors (the numerator does not move —
`hello.coprocessor` was never in it, it crashed on the naming). `51 of 66` under
the wall, from 52 of 67. The `invalid literal for int()` bucket is 6, from 7.

Two published figures were also re-derived rather than transcribed. The corpus
instruction total is **89,400**, from 89,416 — and the recomputation reproduces
89,416 on the old corpus *exactly*, which is what says the counting is right. The
form count does **not** reproduce: 129 forms / 66 base opcodes was published and
every convention tried gives **127 / 64**, on both corpora. The offset is exactly
two either way, and `nvdisasm` emits header lines (`ET_EXEC`, `STO_CUDA_ENTRY`)
that a pattern not anchored on the `/*addr*/` prefix reads as opcodes. The
instruction total is far more sensitive to a regex difference than a set size is,
so the total agreeing and the set not is evidence the anchored pattern is the
right one.

`tval.py` crashed with an `IndexError` on a kernel that stores nothing. It
reports `REFUSED ... this kernel stores nothing` now — a crash in a validator
reads exactly like a missing feature, which has happened twice here already.

### A const-bank operand was masking the gap, and the ABI fact needed a device run

All 23 GEMMs refused on the SASS side at `unmodelled const bank slot 0xc` before
reaching any opcode. `batch.mk` already defined `nctaid_x`; only the map from
constant-bank offset to symbol was missing the launch-geometry block, so the
validator refused an operand its own vocabulary could name. Two lines — and the
offsets are a **driver ABI fact**, which is exactly the kind of thing that must
not be guessed.

Reading them out of `ptxas` output alone would use the translator under test to
license a fact used to validate that translator. `cbank_abi.py` does it in two
independent steps, and the launch is what breaks the circle:

```
(a) ptxas reads offset X for %<reg>          -- from the disassembly
(b) %<reg> returns <extent> on the device    -- from a real launch
```

The six extents are **distinct** (11, 13, 2, 3, 5, 7), so if (a) were wrong the
launch in (b) would return another axis's value. The script asserts that
distinctness before anything else — with two extents equal, a swap between them
is invisible and every check below it passes while asserting nothing. It reads
the map back out of `batch.py` rather than restating it, so a wrong entry there
is what fails. Measured: `0x00/04/08 = ntid.{x,y,z}`, `0x0c/10/14 =
nctaid.{x,y,z}` on sm_89, agreeing with `ptxas` and with the device.

This unblocks no kernel — the 23 GEMMs now refuse one instruction later, on
`S2UR` — and that is the point of landing it: the census stops reporting a
spurious operand refusal in place of the real gap.

**Mutation table**, six probes; the control is the row to read first.

| probe | `cbank_abi.py` | `gap.py` census | `regress.sh` |
|---|---|---|---|
| C5 CONTROL: reorder the two map assignments | ok | ok | ok |
| C1: the block removed (the original state) | FAIL, by name | reverts to the masking operand | ok |
| **C2: `ntid` and `nctaid` swapped** | **FAIL** | ok | ok |
| C3: `nctaid` off by one axis | FAIL | reverts | ok |
| C4: the census stops at the first refusal | ok | **gap 9/13 → 0/0** | ok |
| C6: the launch extents not distinct | FAIL, non-vacuity | ok | ok |

**C2 is the row that justifies committing the device probe**: a swapped mapping
is a *wrong semantics*, and it is invisible to the census and to every standing
result — no currently-passing kernel reads those slots, which is why the omission
survived. C4 was mis-aimed on its first run (the mutation's anchor did not match,
so it never applied, and then the column I measured could not have seen it
anyway). Both halves of that are the standing rule: confirm the mutation is in
the artifact that ran, and say what defect the mutated program has before
recording a survivor.

---

## Floats: two questions, different answers

Measured rather than assumed, and neither is the gate.

**Contraction** — **0 kernels**, MEASURED by `contract.py` as
`FMA(plain) − FMA(.rn)`: forbidding the fusion cannot remove a fused
instruction the PTX asked for, so the difference *is* the number of fusions.
`mul.f32` + `add.f32` becoming `FFMA` is a permitted freedom, and the emitter
states it everywhere it occurs — `naive_gemm_f32` and `y_cpu_matmul` through
expression lowering, and the hand-written `rmsnorm_residual_4096`,
`int8_gemm_scaled` and `rope_*` bodies directly. For **all five** the `.rn`
rewrite is a **complete no-op**: forbidding the fusion changes not one byte of
their SASS, because there is no longer a fusion to forbid. Every artifact this
repository ships states every rounding the hardware performs.

**An empty set is also what a measurement computing nothing returns**, so the
positive control is no longer a shipped kernel. `contract.measurement_is_live`
perturbs one — splitting a shipped `fma.rn.f32` back into the `mul.f32` +
`add.f32` it replaced — and requires the metric to flag it; `fpgate.py` fails
if it does not. Same device as keeping `naive_gemm_f32_muladd` in the corpus.

**Six** things this paragraph used to say were wrong, and each is the same shape.

* **The last of them was the price of finishing it.** The `mul` + `sub` half of
  the rope rotation was recorded here as costing "a `neg.f32` the hardware does
  not pay", and that is a claim about an instruction PTX **does not gain**: the
  `neg` REPLACES the `mul.f32` the `fma` absorbs. Measured — `mul,mul,sub` (3)
  becomes `mul,neg,fma` (3) in the rope body and `mul,sub` (2) becomes
  `neg,fma` (2) at the source level, with the register pool unchanged and the
  **SASS byte-identical** in every case, because `ptxas` folds the negation
  into the FFMA's own operand modifier (`FFMA R7, R4, R5, -R7`). Counting the
  PTX instruction the repair adds without counting the one it removes is the
  same indirect reading as the four below.

* **The count was a hardcoded 9** in `fpgate.py`, and it disagreed with
  `contract.py`'s own measurement *in both directions* — six
  `gemm_f16_bias_relu_*` kernels where `ptxas` contracts nothing, and ten
  contracting kernels omitted. Two lists of one thing drift.
* **Then the count was a DERIVED 16, and derived is not measured.**
  `FFMA(sass) − fma(ptx) > 0` cannot tell a contraction from an **unrolling**:
  one `fma.rn.f32` in a loop body becomes N `FFMA`, which is where
  `paged_decode_attention_*`'s +32 to +56 came from, and `gemm_fp8_*`'s +11 is
  `FFMA` `ptxas` synthesised for something else. It under-reported too, by
  counting `FFMA` only — the four `gemm_f16_swiglu_*` fuse at *half*
  precision. `contract.py` forbids the fusion with `.rn` and re-assembles now:
  if the SASS moves, `ptxas` was fusing. **Wrong in both directions again, one
  layer down, and it was found only because this change moved the numerator.**
* **And "the SASS moves" is not "ptxas fused" either — that reading gave 9 and
  the answer is 5.** The `.rn` modifier restricts `ptxas` in ways beyond
  contraction, so any effect of it registers as one. The four
  `gemm_f16_swiglu_*` move by **`FSEL` 4 → 8 and `IMAD` 202 → 206 with `FFMA`
  0 → 0** — a scheduling difference and no fusion at all. Worse, the
  half-precision explanation published for them here was a **hypothesis
  asserted as fact**: `HFMA2` appears in neither build. The predicate is
  `FMA(plain) − FMA(.rn) > 0` now, which is the quantity itself rather than a
  proxy for it. **Four readings of one number, and every wrong one was
  indirect.**
* **"Behind loop invariants"** was true when written. `loopval.py` has since
  provided them. `fpgate.py` asks the validator now instead of consulting a
  table that models its answer.
* **The repair named was the expensive one.** Forbidding the contraction with
  `mul.rn.f32`/`add.rn.f32` costs 0 to +7.1% instructions *and* gives the
  kernel two roundings where the hardware does one. Saying `fma.rn.f32` emits
  the same instructions, is more accurate, and validates. **The repair is to
  say what the machine does, not to forbid it.**

**Macro-op expansion** — 17 kernels. One PTX instruction that `ptxas` implements
as a multi-instruction refinement. No source-level token fixes it. Measured per
opcode against a mov-only baseline of 24 instructions:

| class | cost | opcodes |
|---|---|---|
| transliteration | +0 | `sin/cos.approx`, `cvt.rn.f32.s32`, `cvt.f32.f16` |
| small | +8 | `ex2 lg2 rcp rsqrt div sqrt .approx` — the same as an ordinary multiply |
| expanded | +40 … +120 | `sqrt.rn.f32`, `rcp.rn.f32`, `div.rn.f32`, `div.rn.f64` |

Every *expanded* op emits `BSSY`/`BSYNC` and `CALL.REL.NOINC` — an out-of-line
subroutine — so it is behind branch and call support anyway, and validating it
bit-exactly would be a proof about an IEEE division algorithm. **Refused by name.**

### The guard, and why it is not optional

The cheap way to "support floats" is to model a PTX macro-op and the MUFU that
`ptxas` seeds it with as the *same* uninterpreted function. Both arms then share
one float factory, and every such kernel validates for the wrong reason.

It is also false, measured on the device: `rcp.approx` differs from `rcp.rn` on
**13.23%** of inputs and `div.approx` from `div.rn` on **27.30%**, while `div.rn`
agrees with a correctly-rounded double quotient on 100.00%. `fpmode.py` routes
every such identification through a table carrying a `validated` flag, refuses an
*expanded* op by name, and self-checks at import.

### Seven float facts refereed against silicon

None can be read off a mnemonic. `fpsem_abi.py` runs each on the device.

**`FSEL` is a bit-exact select, not an arithmetic operation.** That is the
guess that mattered — the operand order is visible in the disassembly, but an
arithmetic instruction is entitled to flush a denormal, canonicalise a NaN
payload or normalise a signed zero, and modelling a flushing instruction as a
pure select would be invisible on ordinary data. Run over denormals at both
ends of the range, `+0.0` against `-0.0`, a quiet NaN carrying a payload, a
signalling NaN and both infinities: `p ? s0 : s1`, bit for bit, 32/32. The
probe also stores each operand unchanged, because if the load/store path were
itself lossy on those patterns a difference at the output would be blamed on
`FSEL`.

**An f32 add is bit-exactly commutative.** `fpmode.py` used to record this as
deliberately open: *"IEEE addition is commutative, so canonicalising by operand
id would be sound and would hide a real question — whether `ptxas` preserves
operand order — so it is left out until something needs it."* Something needed
it, and the answer to the question it was protecting is **no**: `ptxas` sorts
the addends by register number, so `add.rn.f32 d, acc, prod` comes back as
`FADD d, prod, acc`. "Sound in IEEE" is still not enough on its own, because
the claim is about stored **bits** and IEEE leaves a NaN result's payload
implementation-defined — a hardware returning the *first* operand's payload
would break this on exactly the inputs no ordinary test uses. Measured on
sm_89: 32/32 agree, including two quiet NaNs with different payloads.

Two things `ptxas` does had to be designed around, and each would have produced
a confident wrong answer:

* It **CSEs** `a+b` with `b+a` inside one kernel, so the obvious probe is
  answered by the translator under test rather than by the device. `B` is
  passed through two pointers carrying the same values instead.
* It then **sorts** the addends anyway, so both orders cannot be had from one
  kernel. The load order is varied between two otherwise identical kernels, and
  the checker traces each `FADD` source back through its load to the
  **parameter** — comparing register *names* would pass vacuously when two
  cubins differ in numbering while putting the same value in the same slot.

`FMUL` is deliberately **not** canonicalised: nothing has needed it, so nothing
has measured it, and `fpmode._self_check` now pins both halves — `FADD` must
commute, `FMUL` must not — so neither can drift.

**A `-R` operand modifier on a float source is a bit-exact sign flip.** Every
consumer of one is an `FADD` or an `FFMA`, which is entitled to canonicalise
its *result*, so the modifier's own behaviour is invisible through them. It is
observed through `FSEL` instead — the one float-shaped instruction already
refereed as bit-exact — by folding a `neg.f32` into a `selp.f32` source, which
`ptxas` obligingly does. 32/32, with the false arm of the select exercised so
the probe cannot be answered by one source alone.

**The un-foldable lowering of `neg.f32` is *not* one.** This is a refutation,
and it is asserted as one: the probe fails if *nothing* separates the lowering
from a sign flip (then the standing UNPROVED row would be a modelling gap), and
it also fails if nothing *agrees* (then it is not measuring a negation at all).
Every disagreement must be a NaN and must be the canonical `0x7fffffff`, which
is a much stronger statement than "they differ".

**`a - b` is bit-for-bit `a + (-b)`.** This licenses the `FSUB(a,b) ==
FADD(a, FNEG(b))` identification, without which the two sides of every float
subtract build terms that cannot meet. It cannot be asked of `ptxas`, because
`sub.f32` has exactly one lowering; it is asked of the device as two PTX
programs, the second negating `b` with an integer XOR. **The mask has to be
loaded from memory**: given a literal `0x80000000`, `ptxas` recognises the xor
as a negation, folds it back into a modifier and CSEs the two arms into one
instruction — so the probe compares a kernel against itself and answers
perfectly. That was measured, not supposed, and the checker asserts two
*distinct* `FADD`s and a materialised `LOP3` before believing the run.

**`max.f32` computes the PTX rule bit for bit.** Not an ordering: with one
NaN operand the result is the *other* operand, with two it is a canonical NaN,
and `+0.0` ranks above `-0.0`. 32 vectors, **0 disagreements**, and the probe
is live rather than empty — 3 of them are a denormal passing through
**unflushed**, so it distinguishes a flushing implementation from a bit-exact
one, and the run fails if none does. The load/store echo is checked in the same
pass, so a lossy plumbing path cannot be blamed on the instruction.

The model was written **before** the run and was not adjusted afterwards, which
is the difference between evidence and curve-fitting: the NaN clause comes from
the PTX ISA and the signed-zero clause was a guess, and it reported 0
disagreements on the first execution. Had it disagreed, the honest report would
have been the device's answer and a corrected model — not a clean run.

**And `max.f32` is bit-exactly commutative.** This is needed, and needed for
exactly one shape: the shipped ReLU epilogue writes `max.f32 r, r, 0f00000000`
and `ptxas` folds the literal into `RZ` in the **first** operand slot. "IEEE
max is commutative" is not enough for the same reason it was not enough for
`FADD` — the claim is about stored **bits**, and a hardware returning the first
operand's NaN payload would break it on precisely the inputs no ordinary test
uses. 16 **distinct** pairs in both orders, 0 asymmetric; two of them are quiet
NaNs with different payloads, and the run refuses if any pair has identical
operands, because a swap of those is not observable.

`FMIN` is deliberately **not** canonicalised. No committed artifact contains a
`min.f32` and no corpus SASS contains an `FMNMX ..., PT` — measured, 0 of 66 —
so nothing has needed it and nothing has measured it. Same treatment as `FMUL`
commutativity, and `fpmode._self_check` pins both halves.

---

### The third currency, and an integer reader on a float operand

Two things were wrong here and they arrived together.

**The reader was a guess.** `sassexec.rd` handled a `-R` prefix once,
generically, at the top — `return -self.rd(o[1:])`, the **two's complement of
the bit pattern** — and every float arm inherited it. That is a different
32-bit value for every input but zero and the sign bit alone. `-RZ` was worse
in the same place: it collapses to `+0.0` where the operand is `-0.0`, and
`FADD Rd, -Rx, -RZ` is precisely how `ptxas` lowers an un-foldable `neg.f32`,
so the one construct that needed it is the one that got it wrong.

It was **latent rather than live** — no standing result has a negated float
source in its SASS, checked rather than assumed — and the direction it failed
in on the one case measured was a *false UNPROVED*. That is the safe direction
and it is not a licence: nothing said the guess was safe in general, and under
the concretising rungs of the ladder two wrong values can agree. This tool's
rule is that an unmodelled operand **form** is a hard error, not a guess in a
convenient direction. Reach: **eleven** corpus kernels (three RoPE, four
`gemm_fp8`, four paged-decode attention), because a plain `sub.f32` produces
one. `sassexec.frd` is the float reader now, and `|R|` — an absolute value,
a second bit operation nobody has refereed — is refused by name rather than
silently ignored.

**And the emitter had just grown the gap.** Repairing the RoPE rotation to say
`fma.rn.f32` was measured free in PTX instructions (42/62/102 before and after)
and free in SASS bytes (byte-identical). It replaced a `sub.f32`, which
`ptxexec` models, with a `neg.f32`, which it did not — so the PTX opcode gap of
those three kernels each grew by exactly one while their SASS gap did not move,
and `neg.f32`'s reach across the corpus went from 4 kernels to 7. By
`gap.py --rank`'s own reach metric that makes it the **second-highest-reach PTX
opcode in the corpus**.

Nothing measured that. The commit carried a 13-row mutation table over nine
checks and not one of them reads the validator. *A cost stated in one currency
is not a cost until it is checked in the currency that ships* — and there was a
third currency. `fpgate.py` counts it now, with the opcodes read off the
**artifacts** rather than listed, so a list of what the emitter emits cannot
drift from the emitter.

#### …and the first version of that gate counted 5 of 30

It matched `(mul|add|sub|neg|fma)\.f32` — *"the family a change to the fusion
path moves within"*. That scope is right about fusions and it is **not** the
scope of the defect: an emitter change can hand the validator an opcode it
refuses in any family. Measured, the emitter writes **30** float-semantic
opcodes and the regex counted **five**. Two of the uncounted ones have census
reach at or above the 7 of `neg.f32`, the opcode the gate exists for —
`max.f32` at **10** and `cvt.rn.f16.f32` at 7.

(A first pass said *four*, from an ad-hoc `grep` counting kernels that
**contain** an opcode. `gap.py --rank` counts kernels it **blocks**, which is
the comparable measure and is smaller: `cvt.f32.f16` is 5 there, not 8, and
`ex2.approx.f32` does not appear at all, because a macro-op is classified by
`fpmode` and surfaces as a contaminated error rather than an opcode gap. The
same pass said 29 opcodes where the gate measures **30**, having truncated
`cvt.rn.f32.s32` to `cvt.rn.f32`. Two counting conventions, and only the gate's
is reproducible.)

The rule is total now: every float-semantic opcode in a committed artifact is
either **modelled** or in a **named family with a written reason**, and a
thirty-first is in neither and fails. Today that is 6 modelled, 11 conversions,
9 macro-ops (a family *derived* from `fpmode.MACRO_OPS`, not listed again), and
4 f64. `ld.global.v4.f32` and friends are deliberately out: they move a bit
pattern and are unmodelled for a vector-width reason, not a floating-point one.

An all-clear is also what a broken classification reports, so the gate carries
a **positive control** that runs two synthetic opcodes through the *same*
`classify()` the census uses — `abs.f32`, which must be reported, and
`setp.lt.f64`, which must not be called modelled.

That second one is a defect the first version of the widened gate actually had,
and it is the design rule inside the gate written to apply it. `setp.lt.f64`
refuses on its **operand** — `%fd1`, a 64-bit float register the executor has
no sort for — rather than on its opcode, and I had exempted a non-opcode
refusal on the reading that the sample line's operands were at fault. It was
reported as MODELLED. It was caught by reading the output, not by a test: an
unmodelled opcode in the modelled column is obvious once printed. Any refusal
means not modelled.

---

## Shared memory, and what a barrier means

Shared memory is a z3 `Array(BitVec32 → BitVec32)`, word-indexed, threaded
through both executors. A `bar.sync` applies an **uninterpreted** function `H_k`
to it — the same `H_k` on both sides, at the same barrier index.

That is not a conservative approximation, it is what makes the obligation
provable: congruence gives equal-writes ⇒ equal-reads with no axiom about what
the *other* threads in the block did. A per-thread equivalence proof does not need
to know; it needs only that both sides see the same unknown transformation.

**Which mechanism catches what was corrected by mutation.** The design was
written believing `H_k` catches a store `ptxas` moved across the barrier. It does
not — the *snapshot* obligation does, and catches it with `H_k` replaced by the
identity, so that mutation is no evidence for the barrier model at all. What
`H_k` alone catches is a **load** moved across the barrier: the arrays entering
are identical, so no snapshot can see it, and the two sides differ only in reading
`Select(A, i)` against `Select(H_k(A), i)`. Demonstrated both ways — `S8` is
caught, and `S8b`, the same illegal program with a no-op barrier, **validates a
kernel reading pre-barrier data**. Reading stale shared memory is a race, not a
rounding difference.

### Two bugs found by building it, both in code that was already there

- **The shared window is 32-bit and PTX addresses it with 64-bit registers**, so
  the truncation must happen *before* the word shift. `smem_roundtrip` computes
  `(255 - tid) << 4` as `cvt.u64.u32` + `shl.b64`, which zero-extends a near-2³²
  value and shifts it in sixty-four bits with no wrap, while SASS's `[R0.X16]`
  scales in 32 and wraps. Shift-then-truncate makes those two different numbers
  (17179868140 against 1073740780) and reports a **correct** kernel as a
  mismatch. Found from a counterexample at `tid_x = 516`, not from a manual —
  both readings are plausible and they agree everywhere in-block.
- **The store-address obligation was over-strong**, as described under *Method*.

### Two traps in the declaration syntax

`.shared .align N .bW name[K]` is static; `.extern .shared .align 16 .b8 name[]`
is **dynamic**, sized at launch, and a regex requiring `[<digits>]` misses it —
which is 23 of the 43 kernels in this corpus that declare shared memory at all.
The census then reports those as an unmodelled *operand*, i.e. as a missing
opcode rather than a missing declaration form, which is why that bucket read as
a bigger gap than it was.

A shared symbol is also read into a **32-bit** register (`mov.u32 %r26,
smem_pipeline_...`), because the window is 32-bit addressed. Wiring only the
64-bit operand reader covered exactly the one kernel written the other way.

And `.X16` is an address **scale**, not a suffix: `[R2.X16]` is `R2 * 16`.
Dropping it does not fail loudly — it yields a different well-formed address and
reports a real kernel as a mismatch.

---

## Loops

`loopval.py` validates across a loop by a **simulation relation** at the header
rather than by unrolling. Five obligation classes: `BASE` (the relation holds on
entry), `ENTRY` (both sides make the same zero-trip decision), `STEP` (the
relation is preserved — a fixpoint, since the candidate pairs are discovered
rather than declared), `LOOPCOND` (same trip count), `STORES` (same effects,
under a permutation).

`STORES` compares the **epilogue's** stores and nothing else, and that is now
enforced rather than assumed: a store in the prologue, in the PTX loop header or
in the body is refused by name; so is an `EXIT` or `ret` in the prologue or the
body — a path on which the epilogue never runs, which the region split does not
follow — and so is a loop kernel that stores nothing on either side. Before
that, each of those shapes VALIDATED a hand-built wrong translation; see
*The validator's effect model had six blind spots* above.

`exact_pv` — the one kernel here that also carries a Rocq proof — validates at `-O1` with
14 obligations, 3 relation pairs and **1 multiplier identity assumed**. It does
*not* validate at `-O2`/`-O3`, where `ptxas` unrolls the loop ×4. The
optimisation-level differential is what relates the two, and it is sampled
evidence rather than a proof; reaching `-O2` needs peel-and-remainder unroll
matching, which is not built.

**A correction, and what closed it.** This line used to read *"the kernel that
carries three Rocq files"*. That was false: `exact_pv` carried **none**. Today `exact_pv` carries one Rocq file and the
prose above credited it with three. The
three files meant — `AttentionSchedule.v`, `GridStrideSplit.v`,
`SoftmaxErrorBound.v` — are about a *different* kernel, the one
`--emit-attention-ptx` emits, whose entry points are `attn_scores`,
`attn_accum` and `attn_accum_naive`; that kernel is not in this corpus, and
`src/exact_attention_certificate.rs` is correct to record `ptxas` as **trusted
and not validated** for it. The claim was fixed in both directions: the prose
above no longer overstates, and `proofs/ExactPvExact.v` now proves what
`exact_pv`'s PTX computes, so the sentence is true of one file rather than
three. `tests/exact_pv_proof.rs` gates it — the overlap between the proved set
and the validated set is asserted rather than described.

### Nested loops: `nestval.py`, and the lift's one candidate validates

`loopval.py` validates one loop. `nestval.py` validates a **tree** of them, and it
reuses `loopval`'s relation machinery — live-ins, proposal by simulation, the
phasing between a PTX top test and a SASS bottom test — rather than restating
it. `loopval` is untouched, so every standing row it produces is byte-identical.

| kernel | verdict | obligations | what it shows |
|---|---|---|---|
| `y_cpu_matmul` @ `-O1` | **VALIDATED** | 17 | **three nested loops**, a store in the middle one |
| `…_w1_store_stride` | UNPROVED | 10 | the store's row stride is K — refuted at the iteration store **address** |
| `…_w2_acc_add` | UNPROVED | 10 | the accumulator adds — refuted at the iteration store **value** |
| `…_w3_acc_init` | UNPROVED | 10 | the inner accumulator starts at 1 — its pair is never even proposed |
| `…_w7_acc_init_k0` | UNPROVED | 11 | the accumulator is 1.0 **exactly when K = 0** — only the child's **BASE** can see it |
| `…_w4_top_guard` | UNPROVED | 17 | the `EXIT` guard tests N — **ENTRY**, top level |
| `…_w5_inner_guard` | UNPROVED | 8 | the inner zero-trip guard tests N — **ENTRY**, child |
| `…_w6_inner_backedge` | UNPROVED | 7 | the inner back edge tests N — **LOOPCOND** |
| `mem/nest_accum` | **VALIDATED** | 7 | `b[0] += a[i]`: a load reading back the **previous iteration's** store |
| `mem/nest_accum_stale` | UNPROVED | 5 | the same, with `b[0]` loaded once before the loop |

The twins are derived by `build_corpus.sh` from `ptxas`'s genuine output, one
asserted instruction substitution each, so a `ptxas` whose output moves fails the
build instead of silently testing a different program. `regress.sh` asserts every
row in its own direction.

**How a nest is validated.** Each loop gets `loopval`'s obligations — `BASE`,
`ENTRY`, `STEP`, `LOOPCOND` — plus `STORES` for **one iteration**, and it gets them
*in the state its parent has reached*: when a parent's iteration arrives at a
child, the child is proved right there, both sides at once, and then replaced by
its **effect**. Every relation pair it proved becomes one fresh symbol shared by
the two sides, every other slot it writes a fresh symbol of its own side, and if
it stores, memory becomes one fresh shared array. That is sound for a stated
reason: the pairs hold at the child's exit whatever its trip count (`BASE` covers
zero trips, `STEP` the rest, `LOOPCOND` makes the counts equal), and memory is
equal at exit because it was equal at entry and every iteration made the same
stores. Shared symbols forget *how* the outputs depend on the inputs, which costs
completeness and never soundness.

**Memory is carried, not refused.** Every iteration of a loop that stores starts
from one fresh memory array shared by both sides — the induction hypothesis
"memory is equal at the header" — and `STORES` discharges the step. So a load in
iteration k+1 that reads back iteration k's store is *modelled*. `nest_accum` is
the case that needs it, and `loopval` refuses it; `nest_accum_stale` is the case
that shows the model is doing the work: its first iteration agrees with the PTX,
and a validator reading the loop-entry memory in every iteration validates it.

**The pricing named four pieces and there were five.** It said a nested relation,
a store-in-body obligation and an order-aware memory model, and the level census
misattributed two in-body branches — which are the child loops' zero-trip guards,
because `ptxas` rotates every loop to a bottom test and places the guard
immediately before the header, inside the parent. All of that held. What no
census counted: **the SASS nest is guarded by `@!P0 EXIT`**, not by a branch — the
program ends if the outer loop would run zero times. `loopval` refuses a prologue
that can end the program, by name, and that refusal was added one increment ago.
`nestval` accepts it at the top level only, poses `ENTRY` on it, and requires that
nothing after the nest stores; `loop_swap_exit` is the row where something does.

**Two defects were found by building it, and neither was in the new code.**

* **Commutative operands are ordered by z3 node id at construction** — `FADD` and
  `FMAX` in `fpmode`, all three multiply factories in `mulmode` — and node ids
  reuse freed numbers, so the order depends on everything the process built
  before. Measured: `o1/naive_gemm_f32_rn` was UNPROVED on the first validation
  in a process and VALIDATED on the second, on identical inputs, because a PTX
  and a SASS product with equal *values* but different load-guard *shapes* landed
  on opposite sides of the accumulator. `nestval` adds `f(a,b) == f(b,a)` ground
  instances to every obligation, each licensed exactly as the canonicalisation it
  neutralises (`FMAX`'s flag is consulted). No term is built differently, so no
  other validator moves; twenty validations in one process now agree with every
  fresh-process run.
* **`sassexec.gaddr` indexed the register dict** for a 64-bit address's two
  halves, so an address pair computed *outside* the region raised `KeyError`
  instead of becoming a live-in. `nest_accum_stale` — whose address is hoisted
  into the prologue — crashed the validator rather than being refuted. It reads
  through `rd` now, which is what every other register read does; where both
  halves are defined nothing changes, and all 42 earlier `regress.sh` rows are
  identical.

**What it does not do.** `y_cpu_matmul` is validated at `-O1`, not at the level
the corpus ships: at `-O3` it has a two-opcode SASS gap and `ptxas` unrolls from
`-O2`. The frontier census reported it one blocker away for a whole increment
after that, because its structural column asked `loopval` alone; it asks the
suite now and the kernel is clear at `-O1`.
Refused by name: more than one loop at the top level (so every `SEQUENTIAL` and
`MIXED` kernel), a store in an iteration followed by a child loop or a later load
in the same iteration, a SASS loop with no zero-trip guard, a child whose
loop-carried symbols also occur in the state it is entered from, and a PTX carry
flag live inside the nest. **Those last refusals are reached by no fixture**, so
they are untested guards, stated as such.

## The one unbroken chain

For `exact_pv`, and for no other kernel in the repository, both steps are
covered:

```
Y source (tests/exact_pv.ysu)
  |  proofs/ExactPvExact.v :: the_emitted_exact_pv_holds_the_source_dot_product
emitted PTX (tests/exact_pv.ptx)
  |  tools/ptxas_tval/loopval.py @ -O1 :: VALIDATED, 14 obligations
SASS the GPU runs
```

Both seams are named rather than glossed. The first is a **transcription plus a
gate** — `ptx_emitter.rs` does not go through the `Ix` extraction layer, so the
proof is tied to the emitted text by assertions in `tests/exact_pv_proof.rs`
rather than rendered with it from one description the way `exact_attention.rs`
is. The second carries this document's own assumptions: one multiplier identity
ASSUMED, a single thread's view, one optimisation level, one architecture. The
chain is not a proof about `ptxas`.

---

## What this does not claim

- **It is not a proof about `ptxas`.** Each result covers one compilation of one
  kernel at one architecture and optimisation level. That is the whole idea, and
  it is also the whole limit.
- **It is per-thread.** Barriers are modelled soundly for a single thread's view;
  nothing here is a statement about a race between two threads of a block, and
  the `H_k` device is precisely what lets the proof avoid saying anything about
  them.
- **An `unsat` is relative to the modelled semantics.** It says no input can make
  the two sides differ *given this model of the ISA*. That the model is right
  about the silicon is a separate claim, supported by device probes (the MUFU
  identifications, the carry chain, the 64-bit MAC) and sampled rather than
  proved. Neither substitutes for the other.
- **Global memory is store-ordered and byte-faithful, and it describes no
  fault.** A load reads through every earlier store on its side, and two stores a
  translation reorders must be provably disjoint. On the device a 16- or 32-bit
  global access at a misaligned address **faults** (`subword_abi.py`), and the
  model says nothing about a program that faults. `loopval` executes regions
  separately, so a load that could read back a store made in an *earlier region*
  is still REFUSED. Initial memory is a word per byte address, which admits more
  memories than the machine has — sound for equivalence, never complete.
- **`vpdpwssd`, Rocq's kernel and the processor executing its own ISA remain in
  the trusted base**, as `src/exact_gemm_certificate.rs` says.
- **No result here is CI-gated.** It needs the CUDA toolkit, `z3`, and minutes to
  hours per kernel. It is a research tool, run by hand, in the same category as
  the rest of `tools/`.

---

## Running it

Needs `python3` with `z3-solver`, and `ptxas` + `nvdisasm` from the CUDA toolkit.

```sh
cd tools/ptxas_tval
./build_corpus.sh          # tests/*.ptx -> corpus/ and o1/, via ptxas + nvdisasm
./regress.sh               # ALL standing results: 17 kernel rows + 25 memory-model rows, ~RUNTIME
python3 subword_abi.py     # device referee: alignment census, sub-word stores, u8 conversion
python3 fpgate.py          # every float opcode a committed artifact carries
python3 docgate.py         # the doc figures that describe a measurement
python3 gap.py --rank      # cost per kernel, reach per opcode      (~15 min)
python3 frontier.py        # sufficiency: what is SOLE blocker of what  (~25 min)
python3 frontier.py --o1   # the same question with every kernel at -O1
```

The two `frontier.py` runs are minutes and cache to `.frontier_cache*.json`,
keyed on a digest of every corpus artifact **and** every executor source — so a
changed emitter or a changed model invalidates the cache by name rather than by
somebody remembering to pass a flag. The cache is derived and is not committed,
for the reason `.ysu_hw_profile` is not.

`regress.sh` used to cover the straight-line cases only, and the loop and
shared-memory results were three commands the README asked a reader to type. A
documented command nothing runs is how a result goes stale, so it runs all of
them — including the `naive_gemm_f32_muladd` **UNPROVED** row, because a run in
which that turns green is a regression just as much as one where a VALIDATED row
turns red.

`build_corpus.sh` takes each kernel's architecture from its own `.target` line,
never from the local card — compiling at the build machine's architecture is the
bug `tests/ptx_portability.rs` exists to prevent, and here it would silently
change which SASS is under test. All 66 kernels rebuilt **byte-identically** to
the ones the table above was measured on, `-O1` included, so the corpus is
reproducible rather than shipped.

### Reach 10 bought nothing, and that is why `max.f32` was modelled anyway

The reach ranking says `max.f32` blocks **10** kernels — six
`gemm_f16_bias_relu_*` and four paged-decode attention — which is more than the
7 of `neg.f32`. Reach is not why it was modelled, and the cross-check is the
whole point of having both columns: in every one of those ten it is **1 of 11**
PTX opcodes (the GEMMs) or **1 of 5** (attention), with SASS gaps of 14 to 26.
**Necessary for ten, sufficient for none** — the same verdict the one-back-edge
lift got.

What it buys is that the shipped ReLU **lowering** becomes a standing result,
and that required a fact IEEE does not give: `ptxas` swaps the operands, so the
two sides only meet under bit-exact commutativity over NaN payloads. That fact
is measured now. The GEMM around it is still eleven opcodes away.

Two residue items were also settled by measurement rather than by building
them. `selp.f32` was recorded as *"the obvious next unary"*; its corpus reach is
**0** — it appears in no kernel, and served only the FSEL probe's own PTX.
`selp.u32` (reach 25) was already modelled.

And a confirmation worth recording because it looked like a finding: `ptxas`
**rematerialises** three `FMNMX` in each paged-decode split kernel — 35 in the
PTX against 38 in the SASS — at `-O1` and above, and 35 against 35 at `-O0`.
An instruction-count delta between the two currencies cannot tell
rematerialisation from a semantic difference, which is exactly the confound
`contract.py` was fixed for one opcode over. Reach counts *kernels*, so it is
unaffected.

### The queue was ordered by cost, and nobody had computed reach

Every ranking here so far answers "what would it take to validate *this*
kernel". None answers "how many kernels would *this opcode* unblock". `gap.py
--rank` prints both, and the two disagree about what to do next.

`ptx_subword_ops` was recorded as the cheapest kernel left, and re-measuring
made it cheaper still: the real dynamic gap is **2 PTX opcodes and 1 SASS
opcode** — `cvt.u8.u32`, `st.global.s8`, `STG.E.S8` — with **zero** contaminated
errors, the only kernel in the corpus in that state. The eight was a `depth.py`
figure quoted where the dynamic one belonged.

Then reach:

```
cvt.u8.u32      blocks 1 kernel
st.global.s8    blocks 1 kernel
STG.E.S8        blocks 1 kernel
```

One kernel of 66, and no leverage: `ptx_subword_ops` is the **only** corpus
kernel that uses a sub-word store at all. Nor does it exercise machinery
nothing else reaches — its SASS branch (`BSSY`/`BRA`/`BSYNC`) is already covered
by `bn254_permute`, which passes. So it is the cheapest kernel *and* worth
almost nothing, and cheap-to-build was never the ranking a roadmap wanted.

It is also not free. Stores are recorded as `(addr, value, guard)` with **no
width**, so modelling a sub-word store forces a decision the tool has never had
to make: compare the truncated value (faithful to the `Array(BV64 -> BV32)`
memory model, which cannot distinguish an 8-bit from a 32-bit store at the same
address) or the full register (conservative, but a false `UNPROVED` on any
kernel where one side masks in a register and the other in the store). Doing it
properly means a width field through six unpack sites in `loopval.py`,
`batch.py` and `muls.py`. **Not built** — the measurement said not to, and
refusing sub-word stores today leaves the tool sound rather than leaving a hole.

> **Built 2026-09-15, because the decision stopped being a trade-off.** A
> byte-faithful memory model gives a store its width, so a sub-word store is
> compared as exactly the bytes it writes — neither the truncated-value nor the
> full-register compromise. `ptx_subword_ops` VALIDATES and its two wrong twins
> are refuted. "Worth nothing" was right about **reach** and wrong as a verdict:
> it is the one corpus kernel whose validation needs a read-back to be modelled,
> which is what made it the store-ordered model's only corpus reach.

### The loop kernels are gated twice, and only one gate had been counted

`gap.py` measures opcodes. The validator refuses on loop **structure**, and the
two are independent — closing every opcode gap would leave a kernel refused for a
reason nobody had counted. `loopgap.py` is that census. It runs the validator
over every kernel with PTX control flow and aggregates the refusal, which is
possible only because the validator refuses by name and never guesses.

**It asks the SUITE, and that is not what it used to do.** `loopval` handles one
loop and `nestval` one nest; the census reported `loopval`'s answer alone, which
is a FIRST-REFUSAL reading of the structural column — the exact defect `gap.py`
exists to avoid on the opcode column, arrived at here because there used to be
only one loop validator. `suite_validate` asks `nestval` exactly where
`loopval`'s refusal is the one it exists to lift, a back-edge **count**; every
other `loopval` refusal stands.

**It takes none of them:**

```
48 kernels with PTX control flow; 0 validated
   answered by: loopval 8, nestval 40

 24  PTX: more than one loop at one level, SEQUENTIAL depth 1
  6  PTX back edge is predicated; the recognised shape tests at the TOP
  4  PTX loop has 3 own branches; only its exit test is allowed
  3  SASS branch form this CFG cannot place: '@P BRA P1, `(.L)'
  2  SASS branch form this CFG cannot place: 'BRA.DIV ~URZ, `(.L)'
  2  PTX module holds more than one entry point
  2  PTX branch outside the loop nest
  1  PTX: more than one loop at one level, MIXED depth 2
  1  SASS back edge is unconditional
  1  PTX loop body has more than one branch
  1  PTX: loop finder found NO back edge
  1  SASS: loop finder found NO back edge
```

**25 of 48 refuse for one reason: more than one loop at one level.** That
includes all 23 FP16 tensor-core GEMMs, which have three loops. So the recorded
"21–27 opcodes each" understates them — they are behind an opcode gap *and*
behind a structural one, and only the first had been measured.

> **This block read `38` under one bucket and THREE DIFFERENT BLOCKERS were
> sitting behind it.** `loopval` refuses on the back-edge count before it looks
> at anything else, so every kernel with more than one loop was filed under a
> blocker `nestval` lifts. Asked the suite instead: the four `gemm_fp8` carry
> three branches of their own inside the loop, `int8_gemm` and `int8_gemm_scaled`
> branch outside the nest, and `y_cpu_matmul`'s SASS holds the `@!P0 BRA P1`
> form — which is **not a new blocker at all**, it is one already counted for
> `naive_gemm_f32` and `exact_pv`, so its reach was understated by one. The
> `NESTED depth 3` bucket is gone entirely; every kernel in it had a real
> blocker underneath.
>
> **`answered by` is part of the published census for the same reason the rows
> are.** A dispatch that stopped reaching `nestval` would leave every row above
> plausible while the census was wrong about what the validator can do, so
> `docgate.py` asserts the split and `loopgap.py --selftest` asserts both legs —
> on `y_cpu_matmul` at `-O1`, the one kernel in the tree where the two members
> disagree about the **verdict** rather than about the words.

> **This block read `24 / 9` and the 9 was a DETECTION failure, not a capacity
> one.** `loopcfg`'s PTX branch pattern hardcoded `%p(\d+)`, which is a lexical
> assumption about the emitter's register naming rather than a fact about PTX:
> `ptx_emitter` writes `%rt_p0` and `%qp0` in the coprocessor kernels, so a
> branch guarded by one of those **was not a branch to this CFG**. Six kernels
> have two back edges each that were invisible for exactly that reason. Two
> more — the split paged-decode kernels — were invisible because the scanner
> stopped at the first `}` and read only entry 1; they now refuse by name.
> **8 of the 9 were the defect and 1 (`test_drift`) was a correct answer**, so
> the bucket is 1 and `SEQUENTIAL depth 1` is 30. See the subject section below.

> **This block used to fold that 32 into ONE bucket, and the bucket held three
> shapes needing three different validators.** It also carried
> `2 SASS prologue branches to .L_x_0` and `2 SASS: more than one back edge`
> where it now names an unplaceable branch form; both moves are measured below.
> The figures before the split were right for what they counted.

> **This block was stale, and `python3 docgate.py` is what stops it happening
> again.** It read `30` and carried a `2 the SASS zero-trip guard is not the
> last prologue instruction` bucket. Those two kernels are `int8_gemm` and
> `int8_gemm_scaled`, and they moved because an EMITTER change moved them: the
> increment that grid-strided the int8 output tiles in x and y gave that kernel
> two more back edges, taking it from one bucket to another without re-running
> this census. The arithmetic closes exactly — 30 + 2 = 32, and the zero-trip
> bucket is now empty — so the old figures were right when they were written.
> A doc census with no gate is a measurement that decays whenever the thing it
> measures is edited by someone reading a different file.

#### …and it is not the largest lever, because it unblocks nothing alone

This section used to end "supporting more than one back edge is the single
largest lever in the corpus, and it needs no new opcode semantics". The second
clause is true and the first does not follow from it, so it was crossed against
the opcode census: **of the kernels in that bucket, 0 would validate after
the lift.** Every one also has an opcode gap. The smallest are `bn254_fr_mul`
at 2 (`CALL.REL.NOINC`, `IMAD.MOV`) and `y_cpu_matmul` at 3; the 23 GEMMs are
at 20–26.

Two corrections came out of running that cross:

* **`bra` inflates every loop kernel's gap by one.** `gap.py` drives the
  *straight-line* executor and `ptxexec` models no `bra` at all — control flow
  is `loopval`'s layer. `sassexec` does model `BRA` (it refuses a backward one
  by name, which is what hands the loop over), so only the PTX column needs the
  correction.
* **A structural refusal moves with the optimisation level** — `naive_gemm_f32`
  refuses on the back-edge shape at `-O0`, on the prologue shape at `-O2`, and
  at `-O1` is past every structural gate. Asking the question at one level
  answers it for that level. That is how the GEMM above was reached.

  > **The second half of this bullet used to read "and an opcode gap does
  > not", and that is FALSE.** An opcode gap moves with `-O` as readily as a
  > structural refusal does, because a lower level emits a smaller instruction
  > vocabulary. Measured, committed corpus (`-O3`) against `-O1`, same `.ptx`,
  > by running the real census at both levels: `y_cpu_matmul` **2 → 0**
  > (`PLOP3.LUT`, `UIADD3`, both gone), `exact_pv` **2 → 0**,
  > `naive_gemm_f32` **2 → 0**. Two more move without changing size, which is
  > the sharper form of the same point: `bn254_fr_mul` **1 → 1**
  > (`CALL.REL.NOINC` out, `CS2R` in) — **the count is not the thing that
  > moves** — and `int8_gemm_scaled` **3 → 4** (`CS2R` in). These two read
  > `2 → 2` and `4 → 4` until 2026-09-19, when `SHF.L.U32` was modelled: it had
  > been one of each kernel's `-O3` blockers, and `int8_gemm_scaled`'s `-O1`
  > build does not contain it.
  > The old claim was written from one kernel where it happens to hold, and
  > generalised.
  >
  > The `int8_gemm_scaled` figure was published here as `4 → 3` and was wrong:
  > it came from a cheap text scan asking only whether each `-O3` gap opcode
  > still *occurs* at `-O1`, which cannot see a NEW opcode arriving. `CS2R`
  > arrives in both of those kernels. `docgate.py` caught it on its first run,
  > because that check re-censuses rather than re-greps.

(`exact_pv` is refused here because the corpus is built at `-O3`. Its standing
result is at `-O1`, where it still validates — 14 obligations — and the refusal
at `-O3` is the unroll-matching gap already recorded.)

Two normalisations, and the second one is the point: back-edge counts are folded
because the count is a property of the kernel, but **zero** back edges is kept
apart from more-than-one although `loopval` phrases both as "has N back edges".
They are opposite problems — the loop finder coming up empty on a kernel that
demonstrably branches, versus capacity — and the first aggregation written here
merged them and hid nine kernels behind thirty.

#### The bucket held three shapes, and the sufficiency case is behind the dearest

The paragraph above ends "*two normalisations, and the second one is the point*"
— back-edge counts folded, zero kept apart from more-than-one. There is a third,
and leaving it out ranked the cheapest lift first.

**"More than one back edge" is not one lift.** `loopcfg.nest_shape` classifies
the back edges as intervals, and the corpus splits:

| shape | n | what a lift has to do |
|---|---|---|
| `SEQUENTIAL` depth 1 | 24 | the same relation, proved once per loop, composed at the join |
| `MIXED` depth 2 | 5 | both of the others |
| `NESTED` depth 3 | 3 | an inner loop cannot be executed straight-line, so it must be **summarised** by its own proved relation and the induction runs over the nest |

`loopgap.py`'s own docstring already records the general form of this — "*a
census key that merges two causes reports the larger one*" — for the split
between zero back edges and more than one. This is the same observation one
level in, on the bucket that split left behind.

**The ranking inverts.** Reach puts `SEQUENTIAL` first: 24 kernels, and the
cheap lift. Sufficiency puts it last — those 24 are the **furthest kernels in
the corpus**, 21–23 opcodes short each, so the cheap structural lift would buy
nothing at all. The one kernel the lift was measured to be sufficient for,
`y_cpu_matmul`, is `NESTED` depth 3 — behind the dearest of the three.

**And it links the two roadmap items rather than leaving them independent.**
`int8_gemm` — the tensor-core kernel item 1 is about — is `NESTED` depth 3 on
both sides too. Its recorded pricing, "3 PTX / 4 SASS short", is its *opcode*
gap; it also needs the nested lift, which nobody had said. Items 1 and 2 share
a blocker.

#### …and the lift is sufficient for nothing, which IS measurable without building it

The section above says of the first-refusal problem: "*what `loopval` would say
after that is not measurable without building it*". **That is wrong, and
`liftgap.py` is the measurement.** `loopcfg` refuses on the back-edge count
*before* it looks at anything else, so the checks behind that one have simply
never been asked. Ask them: decompose the nest, and run the remaining
structural predicates at every level a lift would produce.

```
38 multi-back-edge kernels, 125 PTX loop levels
 30  SEQUENTIAL depth 1
  5  MIXED depth 2
  3  NESTED depth 3
  0  left with no named structural refusal
```

**Zero.** Every one of the 38 is still refused by a NAMED check `loopcfg` never
reached.

> **SCOPE, since `nestval` exists now.** This census runs the structural
> predicates of a HYPOTHETICAL multi-loop `loopval`; it is not a validator and
> it does not report the suite's answer. `nestval` **is** that lift for the
> `NESTED` shape, and it validates `y_cpu_matmul` at `-O1` — so for those three
> kernels the real answer is `nestval`'s and this block is a statement about a
> validator nobody built. For the 35 `SEQUENTIAL` and `MIXED` kernels nothing is
> built, so it still answers, and the ranking it produced still stands: the
> cheap shape is the furthest kernels in the corpus.

> **This read `0 of 32` and then read `6 of 38`, and the 6 were an OPTIMISTIC
> answer — the one direction this file's own docstring says it cannot give.**
> When the branch pattern stopped hardcoding `%pN`, six coprocessor kernels'
> loops became visible, and this census called all six *clear* because it
> counted stores and branches and nothing else. They are **do-whiles guarded by
> a named predicate register**: `ptx_regions` refuses a predicated back edge
> (the recognised shape tests at the TOP) and refuses a guard `ptx_pred_index`
> cannot resolve (the predicate file downstream is keyed by NUMBER, so a
> `%rt_p0` branch is visible but not executable). Both are now counted, the
> answer is 0 again over the larger bucket, and `--selftest` carries a control
> that perturbs a real artifact — rename every `%rt_p` to `%p` and the
> named-predicate blocker must go while the bottom-test one stays, so the two
> checks are shown independent rather than assumed to be.

`y_cpu_matmul` — the sufficiency case — has a **store in the body** of
one PTX level and one SASS level, plus two SASS levels that branch inside the
body. `loopval` compares the stores *after* the loop, so a store in the body is
refused; and a store in an outer level's body is the ordinary shape of a tiled
kernel, not a corner case. So **the multi-back-edge lift is sufficient for
nothing at either optimisation level**, and the `-O1` result recorded above is
an artifact of the refusal ORDER.

What it does *not* say is that any kernel would validate: past these predicates
lie the opcode gap, the relation proposal and the obligations, none of which is
decidable by reading. A level reported clear is a level with **no named
structural refusal**, and nothing more.

> **The first version of this census was OPTIMISTIC — the one direction its own
> docstring claims it cannot be — and it reported `23 of 32` left with nothing.**
> The store scan anchored on `^st\.global`, and every store that matters is
> PREDICATED (`@%p11 st.global.f32 [%rd11], %f0`), so it matched none of them.
> Caught by reading the report against a kernel whose store had already been
> read by eye. `liftgap.py --selftest` now requires a predicated store to be
> recognised, a non-store not to be, and the count to FALL when the artifact
> loses its stores.

#### The cache key was missing three modules, and the control was a hand list

`frontier.py`'s cache refuses to serve an answer for an older tree, keyed on a
digest of every corpus artifact and every module whose content decides the
answer. That list named `loopval.py` and **not `loopcfg.py`** — the file that
finds the back edges and raises the refusal the structural census folds into a
key. Found by changing it: the first run after the shape split would have served
the pre-split answer straight back.

Its own docstring records that exact hole, one entry earlier, about this file
itself. **The control that was supposed to stop it was a hardcoded list of four
module names** — the defect this directory keeps finding, sitting inside the
check written to prevent it, and it passed with `loopcfg.py` absent.

The list is **derived from the import closure** now, and deriving it found two
more nobody had noticed: `conc.py` and `mac64.py`, both reached through
`loopval`'s own import. **12 modules → 14.**

> **And the first closure walk under-reported, because `import a, b, c` names
> three modules and the obvious regex captures one.** It missed `sassexec.py`,
> reached through exactly that shape. So the control has a positive control of
> its own: a synthetic root importing two known modules must resolve to a
> closure containing them, through the same call.

**The figures do not move**, which is the prediction worth stating: the key
decides whether the cache is *used*, not what the census *computes*. Re-measured
after the change, `-O3` is 66 kernels / 109 distinct blockers / 8 clear and `-O1`
is 66 / 113 / 10 with `y_cpu_matmul` the only kernel one blocker away —
identical to the run before it.

#### The PTX scanner read a fraction of the kernel

`if s == '}': break` stops at the first **inner** brace, not at the end of the
entry — and the coprocessor kernels carry `{ … }` blocks that
`quantization_pass` emits. So the scanner read a prefix and stopped, silently.
Measured, instructions actually read against instructions present:

| kernel | read | present | |
|---|---|---|---|
| `coprocessor_large.coprocessor` | 30 | 84 | 36% |
| `coprocessor_collision.coprocessor` | 26 | 66 | 39% |
| `coprocessor_combined.coprocessor` | 26 | 66 | 39% |
| `coprocessor_attention.coprocessor` | 30 | 70 | 43% |
| `coprocessor_db_index.coprocessor` | 30 | 70 | 43% |
| `paged_decode_attention_split_…_16_8` | 223 | 1035 | 22% |
| `paged_decode_attention_split_…_4_8` | 91 | 903 | **10%** |

**`ptxexec.run_ptx` has the identical `break`**, so it was *executing* 91 of 903
instructions and reporting a symbolic state for it — and `gap.py` drives that
executor, so the opcode gap published for those kernels was measured over the
fraction above. A truncated program is not a smaller program: the back edge
inside a truncated block is a loop, and one of these blocks ends with
``@%qp0 bra $QUANT_HALF2_PACK_0;``.

Brace *depth* now, on both sides. There are no `.func` bodies in this corpus
(measured: 0), so for a single-entry module with no inner block the two
scanners agree instruction for instruction — verified over all 64 such kernels,
0 differing.

#### The two sides were reading different functions

A module with more than one function has no defined SUBJECT, and this validator
did not notice: **the PTX side and the SASS side picked different ones.**

`ptxexec.run_ptx` and `loopcfg.ptx_back_edges` both scanned until the first `}`,
so they read **entry 1**. `sassexec` and `loopcfg.sass_back_edges` read the whole
disassembly. The corpus's two split paged-decode kernels declare
`..._reduce` first in the PTX and the **main kernel** first in the SASS, so the
PTX side was executing `_reduce` while the SASS side read the main kernel — plus
two `ptxas`-synthesised helper functions concatenated after it.

And the addressing makes it worse rather than merely inconsistent. Each `.text.`
section **restarts at `/*0000*/`**, so two of them in one file put two functions
in ONE address space. Measured: those two kernels carry **376 and 248 duplicated
addresses**. `lab` maps a label to an address naming two instructions, and the
`target <= addr` back-edge test compares addresses across functions — so every
back edge reported for them was an artifact.

Both sides refuse by name now. The reach is small and the shape is not:

| | kernels |
|---|---|
| PTX modules with two `.entry` points | 2 |
| SASS files with two `.text.` sections | 2 |
| SASS files carrying a `ptxas` helper function (`$__internal_…`) | 7 |

**Latent, not live, and checked rather than assumed.** In all 8 affected kernels
`loopcfg` reported **0** back edges, so the arity check (`!= 1`) refused first —
fail-closed *by accident of a different check*, not because the branch was
refused. The sweep that matters is the other one: no corpus kernel is in the
state where `loopcfg` sees exactly one back edge while another is hidden, which
is the state that would hand `loopval` a wrong CFG and let it proceed. **All 16
standing results are byte-identical**, verdicts and obligation counts alike.

The helper functions are the same phenomenon one step down: `CALL.REL.NOINC`, a
standing blocker for `bn254_fr_mul` and `bn254_msm_bucket`, is the call into one
of them.

#### A branch form the CFG could not place, found while checking that

`loopcfg.SASS_BRA` recognises exactly one branch form and used `fullmatch`, so
anything else in the branch family was silently **not a branch** — fail-open, in
a CFG. Swept over the corpus, 13 instructions in 8 kernels:
``@!P0 BRA P1, `(.L_x_1)`` and ``BRA.DIV ~URZ, `(.L_x_3)``.

Two defects, and they fail in opposite directions:

* **`loopcfg` could not see them at all.** A *backward* one would be a loop
  invisible to `sass_regions`, which would then hand `loopval` a "body" that
  actually loops.
* **`sassexec` could see them and dropped an operand.** Its `BRA` arm did
  `re.search` for a label anywhere in the instruction, so `BRA P1, target` was
  taken to be governed by its `@` guard alone and the `P1` was discarded — a
  guess, in the file that is otherwise scrupulous about refusing.

**Latent, not live, and checked rather than assumed:** all 13 are FORWARD, and
every kernel holding one is refused earlier for an opcode, so none has ever
executed. That is the reason to close it now rather than after. Both are
refusals by name now, and **all 16 standing results are unchanged** — which is
what says the refusal is not an over-refusal.

It also retracts a smaller claim: the two `paged_decode_attention` kernels were
reported as `SASS: more than one back edge, IRREDUCIBLE`, and that shape was
computed while **ignoring two branches the CFG could not place**. An
irreducible classification taken from an incomplete branch set is not a
classification.

> The first attempt at the `sassexec` half checked the wrong string — `body`
> carries the mnemonic, so ``BRA `(.L_x_0)`` was refused too and **nine standing
> results moved**. The probe that was supposed to confirm the arm had been
> stopped by an unrelated region check one layer up, and its message read as a
> pass. `regress.sh` is what caught it.

### The sufficiency census, and the frontier is empty at the level we ship

The cross above was done by hand, in prose, for one bucket. `frontier.py` is it
corpus-wide and re-derivable. It answers the question a roadmap asks and that
neither existing column does:

* **COST** (`gap.py --rank`) is how many opcodes *this kernel* is short.
* **REACH** is how many kernels *this opcode* blocks. It reads like a ranking
  and it has been wrong twice here in the same direction — `max.f32` blocks ten
  and unblocks none; the tensor-core staging blocks twenty-three and unblocks
  none.
* **SUFFICIENCY** is how many kernels an item is the *sole* blocker of. Nothing
  had computed it.

A kernel has more than one kind of blocker, gated by different layers, so the
census crosses four measurements — and imports all four rather than restating
any, because a second implementation of an aggregation agrees with the thing it
is checking while both are wrong:

* the **opcode** gap from `gap.census`, with `bra` discounted on the PTX side
  (it is not an unmodelled opcode; it is the straight-line executor being handed
  a loop, and counting it dresses a structural blocker up as a feature gap);
* the **structural** refusal from `loopgap`, for any kernel with control flow;
* a **setup** failure, which executes nothing and so reports an *empty* opcode
  gap — that is not a gap of zero and is its own blocker;
* the **unroll** layer from `unroll.py`, because `loopval`'s relation holds at
  the loop header and so needs the two loops in lockstep;
* the **wall** layer from `wall.py`: a kernel whose worst barrier region has at
  least as many symbolic integer multiplies as the smallest region measured
  `unknown`. Crossed in last; the section after the unroll one is about it.

**In the committed corpus exactly one blocker has a non-zero sole-count, and it is
the solver wall.** 66 kernels, **103** distinct blockers; at `-O3` the kernels one
blocker away are `bn254_fr_mul_fast`, `bn254_g1_add`, `bn254_g1_dbl` and
`bn254_ntt4_fused` — each has no unmodelled opcode, no structural refusal and no
unroll blocker, and each is past the wall. No opcode, no staging set and no lift
is the sole blocker of anything: `cvt.rn.f16.f32` is sole blocker of nothing; so
is the whole `cp.async`/`ldmatrix`/`HMMA` staging set; so is the back-edge lift.
That is the honest state of a corpus where **6 kernels are clear, the median is 14
blockers and 13 of 66 are 21 or more**, and it is why "reach" kept naming work
that buys nothing.

> **Until 2026-09-19 this read 106 distinct, 5 clear, median 15 and 31 at 21+.**
> Modelling the u32 division estimate took `ptx_integer_ops` to clear (it
> validates) and removed three blockers corpus-wide (`I2F.U32.RP`, `IMAD.MOV`,
> `SHF.L.U32`). Adding each kernel's newly modelled opcodes back to its set
> reproduces the old figures exactly, so the move is attributable to that and to
> nothing else. The tail fell from 31 to 13 because many GEMMs sat at 21-22 and
> carried two of the three.

> **Before the wall was crossed this read "every blocker has a sole-count of
> zero" and "9 kernels are clear", at 105 distinct.** The four kernels above were
> the difference: clear to four layers, never validated, and past the solver.

#### The fourth layer, and why the distance was a lower bound for eight increments

Everything above crosses three layers and reports the result as a **ranking**.
It is not one. `loopval`'s simulation relation holds at the loop *header*, so it
needs the two loops to run in **lockstep**; if `ptxas` unrolled the SASS loop by
k then the PTX body has to be composed k times, with a peeled prologue and a
remainder cascade. `unroll.py` has measured that layer since before the frontier
existed and **the frontier crossed it not at all**, so every distance printed
above was a lower bound. This file's own residue has said so for eight
increments.

**The structural reason it was never crossed is not that anybody judged it hard:
`unroll.py` had no `if __name__ == '__main__'` guard.** Every line of it was
module-level code that ran a whole-corpus census on import, so `import unroll`
printed a table and no tool could ask it a question. The layer was not
un-crossed because it was expensive; it was un-crossed because the file was a
script wearing a module's name.

**THE PROXY, and its premise stated so it can be checked rather than believed.**
Count the *observable* operations in each loop level's own body on each side —
global loads and stores, and the async global-to-shared copies. `ptxas` cannot
**invent** a global access, so if the SASS body holds k times as many as the PTX
body, the SASS loop is running the PTX body k times. The converse premise — that
it cannot **delete** one — is **false and measurably so**: hoisting a
loop-invariant load out of a body is legal and common, and four corpus kernels
show a level *shrinking*. So a shrink is a refusal, not a ratio.

**GROUND TRUTH EXISTS FOR THREE KERNELS AND THE PROXY AGREES WITH IT SIX TIMES
OUT OF SIX.** `exact_pv`, `naive_gemm_f32` and `y_cpu_matmul` are standing
VALIDATED results at `-O1` — `loopval` has *proved* their PTX and SASS
equivalent, which it can only do if the loops run in lockstep — and all three are
refused at the `-O3` the corpus ships. The proxy reports `MATCHED` for all three
at `-O1` and `UNROLLED` for all three at `-O3`. That is a check in **both**
directions against a verdict reached by a completely different route, and it is
what makes the layer worth crossing into a published ranking. `y_cpu_matmul`'s
levels read `2↔2, 1↔1, 0↔0` at `-O1` and `2↔8, 1↔9, 0↔0` at `-O3`; the eight
extra loads at level 1 are the **peeled remainder of the inner loop hoisted into
its parent**, which is peel-and-remainder made countable.

**WHAT IT REFUSES RATHER THAN GUESSES**, and each of the three was a number this
file used to print. It took PTX back edge 0 against SASS back edge 0 whatever the
two nests looked like, which is how `bn254_fr_mul` reported an unroll factor of
**55** from *six* PTX loops against *one* SASS loop; the two sides must now agree
on `(kind, count, depth)` **and** on which loop is inside which before any level
is paired. A level that shrank is refused. So is a PTX body with nothing
observable in it against a SASS body with something, because `ptxas` cannot
invent one and that is a wrong pairing rather than an infinite factor.

A level with **no** observable operation on **either** side is *vacuous*, not
undecidable — `y_cpu_matmul`'s outermost loop is five instructions of counter and
nothing else. Reading it as undecidable makes the whole kernel undecidable, which
is what the first version of this rewrite did and what the ground-truth check
caught.

The census, at `-O3`:

| verdict | n |
|---|---|
| `MATCHED` | **24** |
| `UNROLLED` | **3** |
| `REFUSED` | **18** |
| `no loop` | **21** |

The 18 refusals are 12 unpairable nests, 4 levels that shrank and 2 subjects the
back-edge finder will not define. The previous proxy answered for 17 of 66 and
called 23 of them `nan` — that bucket was not "unknown", it was the proxy being
blind to `cp.async` staging on both sides at once (`\bLDG\b` does not match
`LDGSTS`, and `cp.async…global` is not `ld.global`), so the ratio was 0/0.
Counting the async family takes those 23 from `nan` to a verdict.

**What crossing it changes.** At `-O3` the three kernels the frontier ranked
*nearest* — `exact_pv`, `naive_gemm_f32` and `y_cpu_matmul`, each at distance 3
and each blocked by the *same* three items — move to distance **4**, and the
corpus goes to **105** distinct blockers. At `-O1` **nothing moves**: all three
are `MATCHED` there, so no blocker is added and the corpus stays at 108 distinct
and 12 clear. That asymmetry is the layer doing exactly what ground truth says it
should.

**A blocker is added only where the proxy DECIDES.** A refusal is the instrument
failing to answer, and recording it as a blocker would be a guess in the
pessimistic direction — so those kernels are named instead, under a
`STILL A LOWER BOUND` heading that `frontier.py` prints: **18 of 66 at `-O3` and
10 at `-O1`**. The distance for those is still a lower bound. The difference from
before is that it is *named* rather than silent.

> **A control fired at `-O1` and caught a bug in this very integration.** A
> kernel the frontier calls *clear* must not be UNROLLED or REFUSED by the
> layer, or the ranking claims a kernel is zero features away while an uncounted
> layer blocks it. The first wiring re-derived the unroll verdict inside
> `report`/`controls` — which run *after* the `-O1` census has chdir'd back out
> of its temporary `-O1` corpus — so it compared an `-O1` clear set against an
> `-O3` unroll measurement and reported all three as contradictions. The
> verdicts are recorded by `measure` **in the directory it read**, and asking
> for one that was never measured now raises rather than being answered from
> whatever corpus happens to be underfoot. This increment's own subject, in the
> control written to catch it.

#### The fifth layer: the solver wall

**"Clear" was an upper bound, and the item it was about to justify was the one to
rank wrong.** The queue named a joint-sufficiency measure as the largest open
item: `exact_pv`, `naive_gemm_f32` and `y_cpu_matmul` were recorded as three
blockers away and sharing all three. Re-derived before building, they are **four**
away (the fourth is the unroll layer's peel-and-remainder), and the best joint set
the frontier then offered was elsewhere: the integer-division lowering, six or
seven blockers clearing **six** kernels. Five of those six are field kernels with
83–259 symbolic multiplies in their worst region.

The calibration that says so is the frontier's own clear set. At `-O3` it called
**nine** kernels clear and **five** of them are standing VALIDATED results; the
other four never validated, and the split between them is exact on one variable —
worst barrier region **≤ 29** multiplies for all five that validate, **≥ 65** for
all four that do not.

`wall.py` makes that a layer. Three-valued, because the honest answer between the
largest region measured PROVED and the smallest measured `unknown` is *nobody has
measured it*:

| verdict | kernels | meaning |
|---|---|---|
| UNDER | 25 | worst region ≤ 33 multiplies |
| PAST | **10** | worst region has ≥ 49 **symbolic integer** multiplies — every one of them field arithmetic |
| UNDECIDED | 29 | between the thresholds, or past them only on work the ground truth never measured |
| REFUSED | 2 | two entry points, which a region count would read as one program |

A frontier blocker is added only for PAST; the other 31 are printed by
`frontier.py` under `STILL A LOWER BOUND ... (wall)`. The layer answers identically
at `-O1` and `-O3` because it reads the PTX.

**Three ways the obvious version would have been wrong, each caught before
publishing.**

* **Counting every multiply put 15 GEMMs PAST.** Their integer multiplies are
  index arithmetic by an immediate — `mul.lo.u32 R, R, 4`, `mad.lo.u32 R, R, 136,
  R` — and every ground-truth multiply is register × register. A float multiply is
  excluded for the same reason: no float region has been measured at the wall.
* **`barregion.muls` did not count `fma.`**, so 11 kernels were under-counted —
  the optimistic direction for a wall.
* **The two-entry-point check matched nothing**, because `.visible .entry` has a
  second dot before `entry`; the split paged-decode modules read PAST. The
  synthetic two-entry control in `wall.py --selftest` is what said so.

**`barregion.py` ran its CLI at import**, so any tool importing it with arguments
of its own crashed trying to open `corpus/--selftest.ptx` — the `unroll.py`
defect, one module over, found before the frontier crossed it.

**The calibration is ground truth reached by another route.** `wall.py` reads the
kernels `regress.sh` asserts VALIDATED — eight, including the three `-O1` loop
kernels — and fails if any is not UNDER; `frontier.py`'s controls repeat that
through the census, and hold the biconditional *blocker iff PAST* non-vacuously at
both levels.

#### The integer-division lowering: refereed three layers deep, and the obligation is past the solver

**The band measurement the wall layer asked for cannot be taken yet.** The corpus
has exactly three regions between the two thresholds — region 0 of
`bn254_ntt4_fused_high{2,3,4}`, each at 48 symbolic multiplies — and `smemval`
refuses all three in 0.3 s on `UNMODELLED SASS OPCODE 'I2F.U32.RP'`. Priced both
ways before anything was built: if that region PROVES, `UNDER_AT` rises to 48 and
ten kernels leave UNDECIDED; if it answers `unknown`, nothing moves, because the
only regions at 48 belong to kernels already PAST. Either way it sits behind the
same blocker set as `ptx_integer_ops` — the lowering of a u32 `div`/`rem`:

    e  = F2I.FTZ.U32.TRUNC.NTZ( MUFU.RCP( I2F.U32.RP(d) ) + 0x0ffffffe )
    e2 = e + HI(e * (-d*e))              # IMAD.HI.U32's addend is the 64-bit PAIR
    q0 = HI(e2 * n); r0 = n - d*q0;  two conditional corrections  ->  (q, r)

The float half has no bitvector model, and matching the sequence as `UDiv` would
be a hole: it assumes the lowering correct, which is the thing under validation.
The alternative is to measure, over the whole finite domain, exactly the facts a
validator would have to assume. `divlow_abi.py` runs three device probes and
asserts every figure below; each probe's `est()` is checked instruction for
instruction against `corpus/ptx_integer_ops.sass`.

| fact | domain | result |
|---|---|---|
| the window: `e ≤ I = ⌊2³²/d⌋` and `(I−e)² ≤ I` | all 2³²−1 divisors | zero exceptions; max `(I−e)²/I` is **exactly 1**, max slack **512**, both attained |
| Lemma A: `I−1 ≤ e2 ≤ I` | all 2³²−1 divisors | zero exceptions; the deficit of **1 is attained**, so the one-unit bound is tight |
| Lemma B: the corrections are exact at **both** estimates Lemma A admits | every `d` × 10 structured `n`, plus 12 `d` × every `n` | 0 failures in 1.85 × 10¹¹ checks |

**Exhaust as far up the chain as the domain stays finite.** Measuring only the
window leaves a solver two composed 32×32 multiplies before the tail begins;
measuring through the Newton step leaves it one. Lemma A depends on `d` alone, so
it can be exhausted, and a validator can assume it the way it assumes `FSEL` is a
bit-exact select.

**The second correction exists for `e2 = I−1` and nothing else.** With it removed
(`divmut.sh` D3), `e2 = I` never fails and `e2 = I−1` fails 4,295,021,506 times.

> **Superseded 2026-09-19:** past the *bitvector* engine only; over Int the same
> obligation is `unsat` in 0.1–0.2 s. See the next section. Kept as measured.

**The tail is past the solver.** With `d` symbolic, z3 answered `unknown`, and
never `sat`, on every posing tried (2026-09-18, one machine):

| posing | budget | answer |
|---|---|---|
| window assumed, goal via `UDiv`/`URem` | 1200 s | unknown |
| window assumed, goal as `q·d + r = n ∧ r < d` | 1200 s | unknown |
| Lemma A assumed, 96-bit `I` | 600 s | unknown |
| Lemma A assumed, `I` eliminated (`e2·d ≤ 2³² < (e2+2)·d`), division-free goal | 900 s | unknown |
| split: `q0 ≤ n/d` from `e2·d ≤ 2³²` | 900 s | unknown |
| split: the correction step given `q−2 ≤ q0 ≤ q` | 300 s | unknown |

The last row was expected to be instant and is not: `r0 = n − d·q0` multiplies two
symbolic variables, so once `d` is symbolic no piece of the tail is linear. With
`d` **concrete**, Lemma A is `unsat` (proved) in 0.0–87.5 s for five of six values
tried. The difficulty is the symbolic divisor, not the lemma. Lemma B is true on
every case measured, so this is a solver limit rather than a false lemma. That is
also why the executor side is **not built**: modelling the four float opcodes as
one windowed estimate would turn `ptx_integer_ops` from REFUSED into UNPROVED and
change nothing else.

**This is a measured counterexample to the wall proxy.** `wall.py` calls
`ptx_integer_ops` UNDER, with 23 symbolic multiplies in its worst region, and its
obligation is past the solver. Two reasons, and only the first is a counting fix:

* `barregion.muls` counts `mul.`/`mad.`/`fma.` and **no integer `div.`/`rem.`**.
  33 of 66 corpus kernels contain one. Counting each as one multiply moves nine
  GEMMs from UNDECIDED to PAST and moves `ptx_integer_ops` only from 23 to 25.
  Whether one is the right weight is unmeasured; zero is the optimistic direction,
  the same defect as the uncounted `fma.` above. **Recorded, not fixed.**
* The hard part is not an instruction count at all. It is a division on the PTX
  side against a multiply chain on the SASS side, a composition that no per-region
  count can see. The proxy's premise that two regions with equal counts can differ
  in cost now has a measured instance, and it is decisive here.

**Four defects of mine on the way, each a false result, three in the direction
that reads as a refutation.** An unbounded 96-bit `I` wrapped, so the query
answered `sat` with slack −2,622,498,317. A window squared in 64 bits wrapped
`(2³²)²` to 0 and admitted `e = 0`. The window referee overflowed u64 and reported
368,450,712 violations at `C = 2` against 767 at `C = 1`; a looser bound cannot
fail more often, and that is what caught it. The first probe emitted
`F2I.U32.TRUNC.NTZ` where the corpus has the `.FTZ` form, and the fix was to change
the instruction rather than argue that denormals cannot reach it. **In a
bitvector query, every intermediate needs the width of its largest value, and a
width error most often shows up as a counterexample.**

#### …and it was past the BITVECTOR solver, not past the solver: `ptx_integer_ops` validates

> **SUPERSEDES the section above on two claims.** "The tail is past the solver" is
> false: it was past z3's *bitvector* engine. Posed over integers with the u32
> wraps written out, the same obligation is `unsat` in 0.1–0.2 s. And the "measured
> counterexample to the wall proxy" goes with it. `wall.py` calls
> `ptx_integer_ops` UNDER, and it now VALIDATES. The tables above are kept as
> measured: every bitvector posing did answer `unknown`.

**The measurement came first, and it inverted the previous increment's
recommendation.** The residue named "Int arithmetic rather than bitvectors for the
correction step" as unmeasured. Measured (z3 `QF_NIA`, `d` symbolic, u32 wraps
explicit as `r = x − k·2³², 0 ≤ r < 2³²`):

| posing over Int | answer |
|---|---|
| Lemma B at `e2 = I` | `unsat`, 0.0 s |
| Lemma B at `e2 = I−1` | `unsat`, 0.1 s |
| premises alone (control) | `sat`, 0.0 s |
| `e2 = I−2` (control: Lemma A's width is needed) | `sat`, 0.1 s |
| `e2 = I+1` (control) | `sat`, 0.0 s |
| second correction removed at `e2 = I−1` (control) | `sat`, 0.0 s |
| window ⇒ Lemma A (Newton step derived) | `unknown`, 300 s |
| window + Lemma A on the Newton term ⇒ tail | `unsat`, 0.2 s |

So the wall was the THEORY. The bitvector engine bit-blasts two composed 32×32
products, where the integer engine reasons about their bounds. What remains
underivable is Lemma A from the window, in either theory. That is fine, because
Lemma A is measured exhaustively; it is exactly the part a device can settle and a
solver cannot.

**What was built, and why each piece is sound:**

* **`divest.py`: the estimate is ONE fresh value per chain, carrying the measured
  facts.** `I2F.U32.RP → MUFU.RCP → IADD3 +0x0ffffffe → F2I.FTZ.U32.TRUNC.NTZ`, each
  link consuming the previous one's TAGGED result. The last link yields a fresh
  `e`. A tagged intermediate read by anything else, a predicated link, or
  a bias other than `0x0ffffffe` is a refusal by name, because the facts were
  measured for that chain and nothing else. The sequence is **not matched as
  `UDiv`**: the Newton step and the tail are executed as the SASS spells them, and
  the obligation still has to relate them to the PTX.
* **Lemma A is the ONE assumption, and it is stated on the SASS's own Newton
  node after a proof.** The fact is about `newton(e, d)`; ptxas computes that value
  its own way (a 65-bit sum with the pair `{e:0}`, `-d` from an `IMAD.MOV`). When
  `IMAD.HI.U32` writes a value, a bitvector query proves it equal to
  `newton(e, d)`, and only then is Lemma A (conditional on `d ≠ 0`) recorded about
  that node. With the fact stated only on its own spelling, the div store was
  `unknown` at 120 s over Int. The window is measured and NOT assumed. A first
  version recorded it, with Lemma A on its own spelling beside it, and removing
  both left every mutation-table row unchanged, so they were deleted. A fact that
  looks load-bearing and is not is a claim nobody checks.
* **`intenc.py`: an EXACT bitvector→Int translation, used as the last rung.**
  A w-bit value is an integer in `[0, 2^w)`, and every operation that can leave
  the range is reduced by a fresh quotient. The integer formula is satisfiable
  exactly when the bitvector one is, so its `unsat` is a proof. Anything it
  cannot translate (a variable shift, a bitwise op other than a low-bit mask) is
  refused, so the rung answers `unknown`. It runs only on `unknown`, and FIRST for
  an obligation about a division estimate.
* **`ptxexec.py`: division by zero is UNSPECIFIED.** The PTX ISA, for `div` and for
  `rem`: *"Division by zero yields an unspecified, machine-specific value."* The
  executor had inherited z3's convention (`x/0` = all ones, `x%0` = `x`). That is a
  stronger specification than the one ptxas is held to, and it refuted ptxas's
  correct `rem.u32` at `d = 0`, where the silicon returns `0xffffffff`. Now each
  quotient is `If(d = 0, u, x)` with `u` fresh. tval splits a store that mentions
  one: where every divisor is non-zero the stores must agree as always; where one
  is zero, the PTX store must BE `u` (then any SASS value meets the spec), and
  every store of the same `u` must store the same SASS value, because unspecified
  is one value, not a different one per store.
* **`SHF.L.U32` without `.HI`, refereed (`shf_abi.py`)**: the low word of the funnel
  equals `Ra << n` at all 32 amounts below 32, 384 cases, 0 disagreements. At
  `n ≥ 32` the silicon returns 0 (clamp); the model leaves that a fresh unknown,
  sound and complete wherever the amount is provably below 32, as it is after the
  `& 0x1f` ptxas emits. The first probe masked the amount in C, and nvvm folded the
  mask into the shift and emitted `SHF.L.W.U32`, the wrap-mode form, which is a
  different instruction. The shape check refused it, which is what it is for.

**The result.** `corpus/ptx_integer_ops`: **VALIDATED, 65 obligations, 17 stores,
three of them discharged over Int** (the quotient, the remainder, and a 64-bit
carry-out that was `unknown` over bitvectors). Its twins are derived by
`build_corpus.sh`, each by one asserted substitution, and asserted in
`regress.sh` by the store that fails:

| fixture | what changed | verdict |
|---|---|---|
| `idiv/no_second_corr` | the quotient's second correction removed | UNPROVED, `store 3: sat` |
| `idiv/rem_wrong` | the remainder's second correction subtracts 0 | UNPROVED, `store 4: sat` |
| `idiv/rem_any_at_d0` | the `d = 0` select unpredicated, so the rem differs ONLY at `d = 0` | **VALIDATED** (the positive control for "unspecified") |
| `idiv/bias_moved` | the bias one larger | REFUSED by name |
| `udiv/twice` | one quotient stored twice | VALIDATED |
| `udiv/twice_split` | the second store differs from the first only at `d = 0` | UNPROVED, by the consistency check alone |
| `udiv/and1` | only the quotient's low bit stored | UNPROVED, by name: a store computed FROM an unspecified value is refused |
| `udiv/and1_two` | ...made to store 2 at `d = 0` | UNPROVED, the same refusal |

The `and1` pair is the refusal's only fixture, and it does not isolate it. At
`d = 0` the spec allows 0 or 1 there and nothing else; tval cannot express "any
value in the image of `& 1`", so it refuses. With the refusal removed, the twin is
still UNPROVED, because `(n/d) & 1` does not prove over Int at `d ≠ 0` either. The
rows assert the refusal FIRES; nothing yet shows it is what stands between the
twin and a false VALIDATED.

**The encoder needed four repairs before the real kernel went through, and the
first was a soundness bug.** The memo keyed on z3 node ids of terms it did not keep
alive. z3 reuses a freed node's id, so a term could inherit another term's
encoding. The div store answered **`sat` in 0.0 s** in one run and `unknown` in
the next. A spurious `sat` is the safe direction, but the same bug could as easily
produce a spurious `unsat`, which is a false proof. Every keyed node is retained
now. The other three were completeness, each found by bisecting the SASS spelling
against a hand-written posing that proved. (1) Upper-bound tracking, so a 32×32
product carried in 64 bits is not wrapped. (2) **Deferred reduction**: a wrap
feeding a same-width operation is consumed unreduced, since `(x mod 2^w)·y ≡ x·y`,
so `d·(0−q0)` stays a product of the program's own values. (3) Constants at or
above `2^(w−1)` enter arithmetic as `c − 2^w`: z3 spells `−q0` as
`bvmul #xffffffff q0`, and `(2³²−1)·q0·d` needs a quotient the size of `q0·d` to
reduce. Plus one quotient per `(dividend, divisor)`, shared by `UDiv` and `URem`,
and a proved store equality kept as a fact for the stores after it. Before the last
two the remainder stayed `unknown`. **The integer engine is sensitive to spelling
in a way the bitvector engine is not**: one change took the hand posing from
`unsat` in 8.5 s to `unknown`. That fragility is recorded, not solved; the twin
`no_second_corr` shows it (its unrelated carry-out store answered `unknown` there
while the genuine kernel proves it).

**Mutation table (`idmut.sh`): 19 rows, 16 probes, one verdict letter per
fixture** (`ptx_integer_ops`, `no_second_corr`, `rem_wrong`, `rem_any_at_d0`,
`bias_moved`, `twice`, `twice_split`, `and1`, `and1_two`). BASE and the control
read `VUUVRVUUU` at the top and again at the bottom.

| probe | letters | caught by |
|---|---|---|
| M1 the estimate chain unmodelled (the original state) | `RRRRRRRRR` | every fixture refuses |
| M3 Lemma A not transferred onto the SASS node | `UUUURUUUU` | the genuine kernel and every passing twin |
| M4 Lemma A one unit wider | `UUUURUUUU` | the same |
| M5 the encoder's subtraction reversed | import FAIL | the encoder self-check |
| M6 deferred reduction off / M7 unsigned constants | `UUUURUUUU` | the genuine kernel |
| M8 separate quotients for `UDiv` and `URem` | `UUUURVUUU` | the remainder |
| M9 division by zero back to z3's convention | `UUUVRVUUU` | the genuine kernel's rem at `d = 0` |
| M10 the consistency check removed | `VUUVRVVUU` | **`twice_split` alone** |
| M12 no fact relevant / M13 no proved-store facts | `UUUURUUUU` / `UUUURVUUU` | the genuine kernel |
| M14 the bias not checked | `VUUVVVUUU` | **`bias_moved` alone**: it VALIDATES with the wrong chain's facts assumed |
| M15 `SHF.L.U32` as a right shift | `UUUURVUUU` | the genuine kernel |

Four survivors, each sorted rather than counted as coverage:

* **M5b**, the subtraction reversed with the self-check off, is green, and that is a
  confirmation. After `simplify`, `a − b` is `a + 0xffffffff·b`, so the encoder's
  `BSUB` and `BNEG` arms are reached ONLY by the self-check. The first run of M5
  was green for that reason, which is why the self-check now also encodes
  unrewritten terms.
* **M2**, the facts recorded on their own spelling, was green on the first run.
  Only the TRANSFERRED Lemma A is ever used by a proof, so the unused facts were
  deleted and the row retired.
* **M11**, the stores-the-value-itself refusal removed, is green: see the `and1`
  pair above. The refusal is reached but not isolated.
* **M16**, a tagged intermediate readable outside the chain, is green because no
  fixture reads one. Without the check a tagged value reaches z3 arithmetic and
  CRASHES, so the direction is fail-closed either way.

**A fact that does not concern an obligation is withheld from it.** The 64-bit
carry-out store is `unsat` over Int in 0.6 s alone and `unknown` at 60 s with the
division facts beside it. So an obligation is given only the facts about an
estimate it mentions. That is sound either way: an omitted fact only makes an
obligation harder.

**What this does NOT cover.** The SIGNED lowering (`I2F.RP`, `IABS`), which is what
the NTT kernels use, stays REFUSED. The unrolled field kernels are PAST the wall
regardless. `loopval`, `nestval` and `batch` ignore the estimate facts, which is
sound but incomplete. And the Int rung's success depends on formula shape, as
above.

#### …and at `-O1` the kernel that was one blocker away is now CLEAR

Crossing sufficiency with the optimisation level is what the corrected bullet
above makes necessary, and it changed the answer twice. **`y_cpu_matmul` at
`-O1` has an empty opcode gap on both sides** — measured by the real census, not
by a text scan: its `-O3` `PLOP3.LUT`/`UIADD3` are gone, no new opcode replaces
them, and its PTX gap is `bra` alone. Its only remaining blocker was the
multi-back-edge limit, three on each side — and `nestval` lifts exactly that, so
once the census asks the **suite** instead of `loopval` alone the kernel has no
blocker left at all.

Measured over the whole corpus at that level, at `-O1` the kernels one blocker
away are `bn254_fr_mul_fast`, `bn254_g1_add`, `bn254_g1_dbl` and
`bn254_ntt4_fused` — the same four as at `-O3`, and for the same one blocker, the
solver wall, which is measured on the PTX and so does not move with the level. At
`-O1` the corpus is 66 kernels, **106** distinct blockers and **9** clear
(`exact_pv`, `naive_gemm_f32` and `y_cpu_matmul` join the six — the first two
by the `-O` effect the corrected bullet above measures, the third because the
census stopped reporting one member's refusal as the suite's).

> **This read "no kernel is one blocker away" and 108 / 12, and "the frontier is
> empty at distance 1 at both levels".** Both were true of a census crossing four
> layers. With the wall crossed the frontier is NOT empty at distance 1 — its
> distance-1 set is exactly the four field kernels no opcode work can reach.

> **This section read "at `-O1` exactly one kernel is one blocker away:
> `y_cpu_matmul`", and 106 / 11.** Both were true of a census that asked
> `loopval`; neither was true of the validator. The figures moved 102 → 104 and
> 106 → 108 because the structural keys changed with the dispatch — the
> `NESTED depth 3` bucket vanished and the three blockers behind it are named
> now — and the clear counts moved 9 / 11 → 9 / **12**.

> **Those two counts were published and gated by NOTHING until this
> increment.** `check_doc`'s `-O1` branch asserted the sole-blocker SET and
> nothing else, so `112` and `10` could decay freely — a gate that checks the
> route and never the number, which is the certificate-count defect one
> increment later, in the gate written to stop published figures going stale.
> Both are asserted now. The `-O1` figures moved 112 → 113 for the same reason
> the `-O3` ones moved 108 → 109: the census keys for the kernels whose loops
> the scanner could not see.

> **And that "one blocker away" is an artifact of the refusal ORDER — see the
> second-refusal census above.** `loopcfg` refuses on the back-edge count before
> it looks at anything else, so `y_cpu_matmul`'s store-in-body is a refusal the
> frontier never reached. Lift the back edges and it is still refused. The
> frontier is empty at distance 1 at **both** levels; this row was the last one
> standing and it does not survive being asked what is behind it.

> The distinct-blocker counts moved **106 → 108** at `-O3` when the
> more-than-one-back-edge bucket split into its three shapes (+2 strings) —
> `python3 frontier.py` is what reported the stale figure, by name, on the run
> that made it stale.

> **They moved again, 109 → 101 at `-O3` and 113 → 105 at `-O1`, and only 3 of
> the 8 belong to the increment that noticed.** `ptx_subword_ops` went 3 → 0 —
> `cvt.u8.u32`, `st.global.s8` and `STG.E.S8` blocked it and nothing else — so it
> is the ninth clear kernel at `-O3` and the eleventh at `-O1`. The other net 5
> came from `a1b524d`, which rewrote two fixtures for unrelated reasons:
> `test_drift` fell 17 → 6, losing seven float64 blockers no other kernel
> carries, and `deterministic_reduce` rose 9 → 12, gaining two. **So 109 and 113
> were right when `8b7122a` published them and stale from `a1b524d` on, through
> two more commits.** The cache key was not at fault — it hashes the corpus as well
> as the modules — and nothing ran the census: its cache is dated an hour before
> `8b7122a`, and every gate run since was `frontier.py --selftest`, which checks
> the controls over two kernels and never reaches `check_doc`. A figure asserted
> only by a minutes-long command is asserted only when somebody pays for it.

> **And they moved once more, 101 → 102 and 105 → 106, because the opcode
> census's own PTX scanner still ended at the first `}`.** The brace-depth fix
> had reached `ptxexec` and `loopcfg` and not `gap.ptx_insns`; it reads
> `loopcfg`'s scanner now. Old against new over the corpus: 59 kernels identical
> instruction for instruction, and the 7 that changed are the 7 recorded — five
> coprocessor kernels each gain the two `ldmatrix` forms and an f16 `mma.sync` the
> truncated scan never reached (4 → 7 blockers), and the two split paged-decode
> modules become a named two-entry-point refusal instead of an opcode list read
> from whichever entry came first (29 → 27). Clear counts, median and the tail are
> unchanged. **These figures cannot go stale silently any more**: the census
> writes `frontier_stamp.json` — the tree it measured and what it found — and
> `docgate.py`, which runs in seconds, fails when that stamp is for a different
> tree and otherwise checks this paragraph against it through the census's own
> comparison.

> **SUPERSEDED — re-measured at the level this claim is about, and the one
> sufficiency case does not survive.** The paragraph below says the lift is
> "the one item in the corpus with a sufficiency case". The second-refusal
> census above was taken on the `-O3` corpus only; run over every kernel
> re-assembled at `-O1`, the level this claim is made at, it answers the same:
> no multi-back-edge kernel is left clear, and `y_cpu_matmul` still has a store
> in the body of one PTX and one SASS level. Behind that is a blocker neither
> census counts: the middle loop's store is followed by the next iteration's
> inner loads, a read-back the memory model cannot represent (see *The
> validator's effect model had six blind spots*). So the lift is necessary and
> sufficient for nothing at **either** level, and the honest price of its one
> candidate is a nested lift, a store-in-body relation **and** an order-aware
> memory model. The two "SASS levels branch inside the body" the census also
> reports for it are the child loops' zero-trip guards, which sit in the parent's
> own body — a misattribution by the level decomposition, in the pessimistic
> direction, recorded and not fixed. The paragraph is kept below as written.

> **SUPERSEDED AGAIN — built, and the candidate validates.** `nestval.py`
> validates `y_cpu_matmul` at `-O1` with 17 obligations: all three loops, the
> middle loop's store compared in every iteration, and the next iteration's inner
> loads reading memory carried across. The pieces priced above were real, and they
> were all of it but one — the SASS nest is guarded by an `EXIT`, which `loopval`
> refuses and no census counted. **SUPERSEDED IN TURN:** the frontier listed the
> kernel with one blocker left for one increment after that, because its
> structural column asked `loopval` alone. It asks the suite now — see *The loop
> kernels are gated twice* — and `y_cpu_matmul` is clear at `-O1`.

So the lift is not "sufficient for nothing". It is the one item in the corpus
with a sufficiency case, and paying for it buys a **new standing result** rather
than a smaller number in a census. Two honest limits go with that:

* **A refusal census reports the FIRST structural refusal**, so lifting the
  PTX back-edge check can expose another — exactly the first-refusal problem
  `gap.py` exists to solve, one layer over. `loopcfg` refuses the two sides
  independently, and both say three, so the lift has to cover both; what
  `loopval` would say after that is not measurable without building it.
* **`-O1` is a weaker subject than the shipped build.** That is already the
  standing position for `exact_pv` and `naive_gemm_f32`, and the thing that
  relates a lower level to the shipped one is a `-O0..-O3` output differential.
  For those two it is measured; **for `y_cpu_matmul` it is not**, and claiming
  the kernel rather than the `-O1` build of it would need that first.

#### Two censuses in one interpreter changed a verdict

The first version of `frontier.py` called `loopgap.census([k])` inline, right
after `gap.census(k)`, in one process. Over a full corpus that reported
**`exact_pv` at `-O1` as `store 0 value: sat`** — a *refutation* of a kernel
that is a standing VALIDATED result, on a `.sass` byte-identical to the
committed one, and which `loopgap` alone validates at budget 20 **and** 60.

It is not a budget artifact and it is not the kernel. Bisected on the number of
kernels censused before it in the same interpreter:

```
preamble  0 kernels  ->  exact_pv: []                        (validates)
preamble  5 kernels  ->  exact_pv: loop:store 0 value: sat
preamble 12 kernels  ->  exact_pv: loop:store 0 value: sat
```

The mechanism is one `sassexec.run_insns` already warns about in its seeding
comment: **the multiply primitive canonicalises its operands by z3 node id at
construction**, and node ids come from a global counter, so unrelated work
earlier in the same interpreter can leave two terms with their operands in
opposite orders and congruence closure will not relate them. A single preceding
kernel is not enough to move it; a corpus of them is.

Two things make this worth writing down rather than fixing quietly. The
observable was a **false refutation** — the direction that reads as *the kernel
is wrong*, not as *the tool gave up*. And the hazard was already documented, one
file away, for regions **inside** one validator run; nobody had asked what it
does **across tools sharing an interpreter**. `frontier.structural` runs that
census in a separate process now, which also makes its column equal to the
doc's by construction rather than by coincidence.

**It is an ordering effect, not a monotone one, and the control for it went
vacuous saying so.** `frontier.py` carries a positive control asserting *both*
that the in-process census is still contaminated and that the isolated one is
not. Run in a fresh interpreter it fires; run at the end of a full 66-kernel
census — in the same process, after far *more* unrelated work — the in-process
census validated `exact_pv` again and the control reported itself vacuous rather
than passing quietly. So more volume is not more contamination, and **a control
that sets up its own experiment cannot inherit whatever the process has already
done**: it runs in a child now, every time.

**It went vacuous a second time, in a fresh child, and that is what retired a
fixed recipe.** The child ran exactly 60 rounds of unrelated census work, and
after the memory model's self-check moved into a private z3 `Context` — a change
that alters only which nodes are built and freed at import — the in-process census
validated `exact_pv` again. Sweeping the preamble, one fresh child per size, at the
previous commit and on the change:

```
preamble        0   5   12   30   60   120   240
at e509590      V   U   U    V    U    V     U
after it        V   U   U    V    V    U     U
```

Non-monotone at **both** commits, and 60 and 120 swapped. A contaminating count
is a fact about the current construction history, not about the hazard, so the
control tries 5, 12, 60, 120 and 240 in order, uses the first that contaminates,
and reports itself vacuous only if none does.

> And the cache hid the fix. `frontier.py`'s digest keyed on the corpus and on
> the *executor* sources and not on `frontier.py` itself — where `PTX_DISCOUNT`
> and the blocker taxonomy live — so the first run made after the repair served
> the contaminated answer straight back out of the cache. It is in the key now.
> A cache built to refuse a stale tree, one entry short of the file that
> decides the answer.

### The staging bring-up would WIDEN this gap, and that is the pricing

The programme's largest standing item is the tensor-core gap: 923 of the 925
`mma.sync` this compiler emits are floating point and carry no proof of the
value at all, and the one kernel that could carry the full argument is `int8`,
which is a stub with no shared-memory staging and sits at 0.40x cuBLASLt. The
obvious reading is that building the staging is simultaneously the performance
increment and the route to a real tensor-core kernel under the validator. It is
not. Measured before writing any of it:

```
kernel            PTX gap                        SASS gap
int8_gemm         3   bra, mma.sync…s8,          4   S2UR, CS2R,
                      red.global.add.s32             IMMA.16832.S8.S8,
                                                     RED.E.ADD.S32.STRONG.GPU
gemm_f16_1024     9   + cp.async.cg.shared…,    13   + LDGSTS.E.BYPASS.128,
                      cp.async.commit_group,         LDGDEPBAR, DEPBAR.LE,
                      cp.async.wait_group,           WARPSYNC, HMMA.16816.F32 …
                      ldmatrix ×2 …
```

Staging is exactly the difference between those two rows. Adding it to the int8
kernel imports the whole `cp.async` / `ldmatrix` family on the PTX side and the
`LDGSTS` / `DEPBAR` / `WARPSYNC` family on the SASS side — roughly **3 → 8 and
4 → 8** — and it adds back edges to a kernel that is already refused for having
three. **So the perf increment and the validation increment point in opposite
directions for this kernel**, and the staging should be ranked and justified as
a throughput item on its own terms rather than as a step toward a validated
tensor-core GEMM.

What the pricing says to build instead, if the goal is validation: the int8
kernel's opcode gap is small and *its tensor-core instruction is the only one in
this corpus whose semantics fit the existing bitvector model* — an int8 `mma` is
a wrapping sum of 32 int8 products into int32, where every f16 `mma` needs a
float theory and an unspecified internal summation order. That, plus the
multi-back-edge lift, is the route; neither is the staging.

### Two ways the opcode census under-reports, both measured

`bra` appears in `gap.py`'s gap for 37 kernels where a textual scan finds 48.
The eleven are two distinct causes, and the split was measured rather than
assumed:

- **6 hidden by predication.** A predicated instruction whose predicate name the
  executor does not recognise is attributed to the *predicate*, not the opcode
  behind it, so `@%rt_p0 bra $L;` counts as `@%rt_p0`. That is the six
  coprocessor kernels, and it is why the earlier note "`coprocessor_test` still
  needs `bra`" is right while the census appears to disagree.
- **5 hidden by setup failure.** When the census cannot build the initial state
  it executes no instruction and reports an *empty* opcode gap — which reads as
  "nothing unmodelled" and sorts to the top of a cost ranking. Four `gemm_fp8_*`
  and `rmsnorm_residual_4096` are in that state (`2 .shared arrays`). `--rank`
  flags them.

Removing predication alone from the textual scan gives **42, not 37**, which is
how the 6/5 split was separated.

Measurement scripts, which is most of what the numbers above came from:

```sh
python3 scope2.py      # what blocks each kernel, cross-tabbed with control flow
python3 depth.py       # each kernel's whole opcode alphabet, not its first refusal
python3 gap.py         # what the executor GENUINELY refuses -- the dynamic gap
python3 gap.py --rank  # the same census, ranked by COST and by REACH
python3 loopgap.py     # why loopval refuses each kernel that has a loop
python3 tractable.py   # if every opcode were modelled, what could the solver close?
python3 smemdepth.py   # what ELSE each shared-memory kernel needs
python3 barregion.py   # multiplies per barrier region, against the wall
python3 fpclass.py     # contraction vs macro-op, per kernel
python3 cbank_abi.py   # referee the const-bank ABI against ptxas AND the device
python3 fpsem_abi.py   # referee the seven float facts against the device
python3 fpgate.py      # which contraction kernels a repair unlocks, by asking
python3 unroll.py      # did ptxas unroll?  (it did, x4, at -O2 and above)
python3 docgate.py     # the two doc figures that describe a MEASUREMENT
```

Mutation tables — each carries a **control row, which is read first**, because a
table where every row fails the same way is reporting the state of the tree
rather than the mutations:

```sh
./gmut.sh    # the float macro-op guard
./lmut.sh    # the loop validator
./smut.sh    # shared memory and barriers, 12 probes
./rmut.sh    # the two rankings: reach, and the loop-structure census
```

These rewrite the `.py` files in place and restore from a tarball made by
`mkbase.sh`, which refuses an archive with too few entries — a failed `tar czf`
leaves an empty one behind, and restoring from that repairs nothing, so every
later probe runs against the previous probe's mutation and the whole table is
wrong. That has happened here.
