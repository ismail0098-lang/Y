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
| `exact_pv` @ `-O1` | **VALIDATED** | 14 | 1.1 s | across a **loop**; 1 multiplier identity assumed |
| `smem_roundtrip` | **VALIDATED** | 18 | 0.2 s | **shared memory**, 1 barrier |
| `naive_gemm_f32` @ `-O1` | **VALIDATED** | 9 | 0.2 s | **a shipped GEMM** — the emitter says `fma.rn.f32` |
| `naive_gemm_f32_muladd` @ `-O1` | UNPROVED | 7 | 0.2 s | the form Y used to ship — `store 0 value: sat` |
| `naive_gemm_f32_rn` @ `-O1` | **VALIDATED** | 9 | 0.2 s | the contraction *forbidden*, at a different SASS |

Thirteen kernels validated, **361 obligations**, and three UNPROVED rows that
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
UNPROVED after 9,705 s with 261 of 276 cut points closed and **no `sat`**. There
is a solver wall between 29 and 65 multiplies per query.

`tractable.py` asks the counterfactual — *if every opcode were modelled, how many
kernels could the solver close?* — using barriers as cut points:

**51 of 66 fall under the wall. 15 are over it.**

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
- `ptx_subword_ops` is the cheapest kernel left: 8 unknown PTX ops, all integer,
  no float, no branch, no loop, no shared memory. **Both halves of that are
  wrong, and it is measured below** — the dynamic gap is *three* opcodes, not
  eight, and the SASS side does branch. It is also the wrong kernel to build.
- `ptx_integer_ops` yields a **finding** rather than a kernel: `ptxas` implements
  32-bit `div.u32`/`rem.u32` through the *float* unit — `I2F.U32.RP`, `MUFU.RCP`,
  `F2I.TRUNC`. An integer PTX operation lowered as a floating-point macro-op.

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
./regress.sh               # ALL sixteen standing results, ~50 s
```

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

### The loop kernels are gated twice, and only one gate had been counted

`gap.py` measures opcodes. `loopval` refuses on loop **structure**, and the two
are independent — closing every opcode gap would leave a kernel refused for a
reason nobody had counted. `loopgap.py` is that census. It runs `loopval` over
every kernel with PTX control flow and aggregates the refusal, which is possible
only because `loopval` refuses by name and never guesses.

**It takes none of them:**

```
48 kernels with PTX control flow; 0 validated

 30  PTX: more than one back edge (this validator handles exactly one)
  9  PTX: loop finder found NO back edge
  2  SASS prologue branches to .L_x_0 rather than the loop exit
  2  the SASS zero-trip guard is not the last prologue instruction
  2  SASS: more than one back edge
  1  SASS back edge is unconditional
  1  PTX loop body has more than one branch
  1  SASS: loop finder found NO back edge
```

**32 of 48 refuse for one reason: more than one back edge.** That includes all
23 FP16 tensor-core GEMMs, which have three. So the recorded "21–27 opcodes
each" understates them — they are behind an opcode gap *and* behind a structural
one, and only the first had been measured.

#### …and it is not the largest lever, because it unblocks nothing alone

This section used to end "supporting more than one back edge is the single
largest lever in the corpus, and it needs no new opcode semantics". The second
clause is true and the first does not follow from it, so it was crossed against
the opcode census: **of the 30 kernels in that bucket, 0 would validate after
the lift.** Every one also has an opcode gap. The smallest are `bn254_fr_mul`
at 2 (`CALL.REL.NOINC`, `IMAD.MOV`) and `y_cpu_matmul` at 3; the 23 GEMMs are
at 20–26.

Two corrections came out of running that cross:

* **`bra` inflates every loop kernel's gap by one.** `gap.py` drives the
  *straight-line* executor and `ptxexec` models no `bra` at all — control flow
  is `loopval`'s layer. `sassexec` does model `BRA` (it refuses a backward one
  by name, which is what hands the loop over), so only the PTX column needs the
  correction.
* **A structural refusal moves with the optimisation level and an opcode gap
  does not.** `naive_gemm_f32` refuses on the back-edge shape at `-O0`, on the
  prologue shape at `-O2`, and at `-O1` is past every structural gate with one
  unmodelled opcode. Asking the question at one level answers it for that
  level. That is how the GEMM above was reached.

(`exact_pv` is refused here because the corpus is built at `-O3`. Its standing
result is at `-O1`, where it still validates — 14 obligations — and the refusal
at `-O3` is the unroll-matching gap already recorded.)

Two normalisations, and the second one is the point: back-edge counts are folded
because the count is a property of the kernel, but **zero** back edges is kept
apart from more-than-one although `loopval` phrases both as "has N back edges".
They are opposite problems — the loop finder coming up empty on a kernel that
demonstrably branches, versus capacity — and the first aggregation written here
merged them and hid nine kernels behind thirty.

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
