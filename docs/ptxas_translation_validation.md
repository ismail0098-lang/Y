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

Measured on 2026-09-04, on this machine (RTX 4070 Ti SUPER, sm_89, CUDA 13.3,
z3 5.0.0), reproduced from a clean `tools/ptxas_tval/` by the commands in the
last section.

| kernel | verdict | obligations | time | what makes it interesting |
|---|---|---|---|---|
| `fma/rn` | **VALIDATED** | 9 | 0.0 s | float, with contraction forbidden by `.rn` |
| `fma/plain` | UNPROVED | 10 | 0.0 s | **the negative control** — `store 0: sat` |
| `bn254_permute` | **VALIDATED** | 30 | 0.2 s | branching `ptxas` invented |
| `bn254_sub_vec` | **VALIDATED** | 88 | 9.4 s | |
| `ptx_carry_chain` | **VALIDATED** | 123 | 24.6 s | 24 predicated instructions |
| `exact_pv` @ `-O1` | **VALIDATED** | 14 | 1.0 s | across a **loop**; 1 multiplier identity assumed |
| `smem_roundtrip` | **VALIDATED** | 18 | 0.2 s | **shared memory**, 1 barrier |

Six kernels, **282 obligations**. `bn254_fr_mul_fast` and `bn254_ntt4_fused` are
UNPROVED and are discussed under *The wall* below — neither produced a `sat`.

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

**52 of 67 fall under the wall. 15 are over it.**

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

`scope2.py`, re-run 2026-09-04 over all 67 kernels the repository ships PTX for:

```
PAST BOTH EXECUTORS: 8 / 67

   23  cp.async.cg.shared.global      the FP16 tensor-core GEMMs
   12  bra                            multi-block control flow
    8  (past both executors)
    7  crash: invalid literal for int()   a residual PARSER defect, not a gap
    5  more than one .shared array
    4  I2F.U32.RP                     32-bit integer division, via the float unit
    3  cvt.rn.f32.s32
    2  IABS
    3  cvt.u8.u32 / cvt.f64.f32 / ld.global.ca.f32
```

The seven-kernel `invalid literal for int()` bucket is a **crash**, not a
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
  (none device-validated), f16 pack/unpack, and FFMA contraction on top.
- `ptx_subword_ops` is the cheapest kernel left: 8 unknown PTX ops, all integer,
  no float, no branch, no loop, no shared memory.
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

### The front of the queue is a naming assumption, not a feature

The kernel closest to passing is not a GEMM. `hello.coprocessor` has a **zero**
opcode gap on both sides and still does not run, and the coprocessor family
behind it is blocked by two lexical assumptions in the PTX executor:

- predicates must be named `%p<digits>` — `re.match(r'^@(!?%p\d+)')`. These
  kernels use `%qp0` and `%rt_p0`, so the whole predicated instruction is read
  as an unknown opcode.
- registers must be named `%r<n>` / `%rd<n>` / `%f<n>` — the register file is
  keyed by `int(name[2:])`. A named virtual register like `rt_A_ptr` gives
  `int('A_ptr')`, which is the `invalid literal for int()` crash class above.

Both fail closed, and neither is a missing semantics. Broadening the predicate
regex alone takes their PTX opcode gap 2 → 0 and lands them on the second one, so
they are one repair, not two: the register file has to become name-keyed. That
ripples into `run_lines`' `(kind, i)` region seeding, which the loop validator
depends on — a refactor with its own mutation table, not a drive-by. It is
recorded here rather than done.

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

**Contraction** — 9 kernels. `mul.f32` + `add.f32` becoming `FFMA` is a permitted
freedom. The repair is to emit `.rn`, costing between 0 and +7.1% instructions.
It unlocks nothing today: all 9 are behind loop invariants and 6 also behind
shared memory. So it is *scheduled*, not deferred — make the change on the kernel
that needs it, when the loop work reaches it.

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

`exact_pv` — the kernel that carries three Rocq files — validates at `-O1` with
14 obligations, 3 relation pairs and **1 multiplier identity assumed**. It does
*not* validate at `-O2`/`-O3`, where `ptxas` unrolls the loop ×4. The
optimisation-level differential is what relates the two, and it is sampled
evidence rather than a proof; reaching `-O2` needs peel-and-remainder unroll
matching, which is not built.

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
./regress.sh               # the standing straight-line results, ~35 s
python3 loopval.py o1/exact_pv.ptx o1/exact_pv.sass 60 wide
python3 smemval.py smut/smem_roundtrip.ptx smut/smem_roundtrip.sass 60 wide
```

`build_corpus.sh` takes each kernel's architecture from its own `.target` line,
never from the local card — compiling at the build machine's architecture is the
bug `tests/ptx_portability.rs` exists to prevent, and here it would silently
change which SASS is under test. All 67 kernels rebuilt **byte-identically** to
the ones the table above was measured on, `-O1` included, so the corpus is
reproducible rather than shipped.

Measurement scripts, which is most of what the numbers above came from:

```sh
python3 scope2.py      # what blocks each kernel, cross-tabbed with control flow
python3 depth.py       # each kernel's whole opcode alphabet, not its first refusal
python3 gap.py         # what the executor GENUINELY refuses -- the dynamic gap
python3 tractable.py   # if every opcode were modelled, what could the solver close?
python3 smemdepth.py   # what ELSE each shared-memory kernel needs
python3 barregion.py   # multiplies per barrier region, against the wall
python3 fpclass.py     # contraction vs macro-op, per kernel
python3 cbank_abi.py   # referee the const-bank ABI against ptxas AND the device
python3 unroll.py      # did ptxas unroll?  (it did, x4, at -O2 and above)
```

Mutation tables — each carries a **control row, which is read first**, because a
table where every row fails the same way is reporting the state of the tree
rather than the mutations:

```sh
./gmut.sh    # the float macro-op guard
./lmut.sh    # the loop validator
./smut.sh    # shared memory and barriers, 12 probes
```

These rewrite the `.py` files in place and restore from a tarball made by
`mkbase.sh`, which refuses an archive with too few entries — a failed `tar czf`
leaves an empty one behind, and restoring from that repairs nothing, so every
later probe runs against the previous probe's mutation and the whole table is
wrong. That has happened here.
