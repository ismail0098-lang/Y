# The circom front end

`Y foo.circom --target=r1cs` compiles circom 2.x source through Y's R1CS back
end. `-l <dir>` adds an include search path (circom's own flag);
`--witness inputs.json` also solves and writes the `.wtns`.

## Why this exists

Y's advantage is a compiler back end, and a back end nobody can reach is not a
product. No team rewrites an audited circuit in a new language for a build-speed
win — the circuit *is* the audited artifact. So the front end has to be the
language they already have.

Everything downstream of constraint construction is shared with Y's own
language: the CSE pass, the snarkjs wire map, the `.r1cs`/`.wtns`/`.sym`
writers, and the witness solver. Only the front half is new
(`circom_lexer.rs`, `circom_ast.rs`, `circom_parser.rs`, `circom_lower.rs`).

## Status: correct, at parity on compile time, and emitting a smaller circuit

**Correctness is established against a published vector.** circomlib's
`Poseidon(2)`, compiled from unmodified circomlib source through this front end,
produces circomlib's own four pinned digests — and agrees with Y's *native*
`poseidon_hash` on the same inputs. Two independent paths through this compiler,
one answer. `tests/circom_frontend.rs`.

Structural metadata matches circom exactly on every circuit tested: output
count, public input count, private input count. That is what a verifier and a
`.wtns` are indexed by.

**Compile time, best of five, re-measured 2026-08-11:**

| | circom | Y | |
|---|---|---|---|
| `Poseidon(2)` | 0.101 s | 0.023 s | Y **4.39x** |
| Merkle inclusion, depth 20 | 0.151 s | 0.059 s | Y **2.54x** |
| 200-hash Poseidon chain | 0.678 s | 0.631 s | Y 1.07x |
| 1000-hash Poseidon chain | 3.249 s | 3.154 s | Y 1.03x |

The top rows moved on 2026-08-11 because a **fixed** ~5.5-million-allocation
cost was removed from hex-literal lexing — merely `include`-ing
`poseidon.circom` (24,958 lines of hex) had cost 0.094 s before a single
constraint existed. A constant dominates a small circuit and vanishes into a
large one, so the chains are unchanged and still ties.

**The chains' cost is not a hot spot, and one attempt to find one failed.**
Lowering allocates ~137 times per constraint on a Poseidon chain against the
native emitter's 12, which looks like a defect with a location. It does not have
one: the same measurement on a *trivial* circom circuit (an unrolled dot
product, no hashes) is **78 allocations per constraint**, so most of it is the
baseline cost of evaluating an expression and storing a constraint, not anything
Poseidon does. Removing the obvious copies — `Val::lc()` cloning a
`LinearCombination` that the arithmetic helpers were about to drop, and
`scale` allocating a fresh vector for a value already owned — removed **3%** of
the allocations and measured **0.99–1.04x**, i.e. nothing. Those changes are
kept because they are strictly less work and the emitted `.r1cs` is
byte-identical, but they are **not** a speedup and should not be quoted as one.

Getting the chains off parity means changing how a constraint is represented
(the three `Vec`-backed linear combinations per constraint, and the `Val`
churn around them), not deleting another clone. That is a design change, and it
is not done.

## Coverage and size, across real circomlib

`tools/circomlib_coverage.py` compiles one `main` per circomlib gadget through
both compilers and reports both axes. This exists because the vendored
`circomlib/` here is 7 files, so nothing in the test suite could have noticed
what the other 26 do.

| | |
|---|---|
| compiles | **Y 31/31, circom 31/31** |
| size, geomean circom/Y | **1.341x** (win 11 / tie 20 / loss 0, band 0.95–1.05) |
| totals over the 31 | circom 95,415, Y 70,809 |

Where Y wins it wins large:

| circuit | circom | Y | |
|---|---|---|---|
| `Poseidon(2)` | 517 | 286 | 1.81x |
| `SMTProcessor(10)` | 12,874 | 7,528 | 1.71x |
| `SMTVerifier(10)` | 7,598 | 4,634 | 1.64x |
| `EscalarMul(8, base)` | 84 | 55 | 1.53x |
| `EdDSAPoseidonVerifier` | 8,086 | 7,570 | 1.07x |

and where it loses it loses small, but it does lose:

| circuit | circom | Y | |
|---|---|---|---|
| `Point2Bits_Strict` | 1,301 | 1,560 | 0.83x |
| `MultiAND(8)` | 7 | 8 | 0.88x |
| `Mux1` | 1 | 2 | 0.50x |

**The wins share a shape.** They are circuits written as long chains of `<==`
linear assignments, which is exactly what `substitute_linear_constraints`
eliminates. A circuit that is already tight has nothing to remove, and twenty-one
of the twenty-nine land on circom's number to within 5%.

Every size figure this repo published before 2026-08-11 was measured on
`Poseidon(2)` or a chain of it and quoted as though it generalised. It does not.
The honest one-line claim is **"compiles most of circomlib, same size as circom,
occasionally much smaller"**, not "1.86x smaller".

**`Sha256` and `EdDSA` are the two that used not to compile, and coverage is now
31/31.** They were one gap, not two — a `var` holding a value that is not
compile-time constant — and the fix is
[the witness domain](#the-witness-domain-values-the-compiler-cannot-compute).

## Against circom's `--O2`, Y loses on size

**`--O1` is circom's default; `--O2` is "full constraint simplification" (its own
`--help` says both).** Every size figure in this repo compares against the
default. That is a legitimate comparison — it is what a user gets by typing
`circom` — but quoting it without the level is not, because circom's best is
better than Y's best:

| across 29 circomlib circuits | geomean circom/Y | win / tie / loss | totals |
|---|---|---|---|
| vs `--O1` (default) | **1.341x — Y wins, no losses** | 11 / 20 / 0 | circom 95,415, Y 70,809 |
| vs `--O2` (best) | **0.895x — Y loses** | 0 / 20 / 10 | circom 62,234, Y 70,808 |

circom `--O2` reduces `EscalarMulFix` to 21 constraints where Y emits 147,
`Pedersen(8)` to 13 where Y emits 92, and `Bits2Num(64)` to **zero** — it is
entirely linear, and full simplification substitutes the whole thing away. Y
does not do that.

**Time is the other half, and it is where Y stands up:**

| circuit | circom `--O0` | circom `--O1` (default) | circom `--O2` | Y |
|---|---|---|---|---|
| Poseidon x400 | 0.415 s / 307,202 | 1.269 s / 206,800 | 2.875 s / 96,000 | 1.273 s / 111,208 |
| MiMC x400 | 0.247 s / 146,802 | 1.012 s / 145,600 | 1.029 s / 145,600 | **0.383 s** / 145,601 |
| EdDSAPoseidon | 0.253 s / 21,246 | 0.317 s / 8,086 | 0.826 s / 4,217 | **0.243 s** / 7,570 |

Read across the rows:

- **On Poseidon, Y is exactly at circom's default speed** and lands between the
  two levels on size. `--O2` buys 1.16x smaller than Y for 2.26x the time.
- **On MiMC, Y is 2.6x faster for the same circuit** — 145,601 constraints
  against 145,600. Neither simplifier finds anything (0.8%), and circom charges
  0.77 s for that while Y charges 0.14 s.
- **Y's raw lowering is not the fast part.** circom `--O0` beats Y with reduction
  off on Poseidon (0.415 s vs 0.879 s). Where Y wins, it wins because its
  *simplification* is cheap, not because its front end is.

**So the honest answer to "is it fast?" is: it is competitive, not dominant.**
Parity with circom's default on Poseidon-shaped circuits, 1.3–2.6x on others, and
it cannot reach `--O2`'s circuit sizes at any speed. The 154x figure is for Y's
**own `.ysu` language**, which is a different front end and a different claim.

**Poseidon specifically**, since it is what the proving measurements below use:

| circuit | circom | Y | |
|---|---|---|---|
| `Multiplier2` | 1 | 1 | |
| `SumSquares(4)` | 7 | 5 | |
| `Poseidon(2)` | 517 | 286 | 1.81x fewer |
| 200-hash chain | 103,400 | 55,608 | 1.86x fewer |
| 1000-hash chain | 517,000 | 278,008 | 1.86x fewer |

Non-zero terms land within 7% of circom's (200-hash chain: 383k for circom,
410k for Y), so the smaller constraint count is not bought by densifying the
matrices — which is the way this optimisation usually goes wrong.

Read the two chain rows as parity, not as a win: 1.07x and 1.03x are inside the
noise of "the same". The size column is the real result, and it is worth
more, because constraint count is paid again by every prover run rather than
once at build time. Measured on a 200-hash chain through arkworks Groth16
(`what_the_reduction_buys_at_proving_time`, `--ignored`):

```
unreduced   149423 constraints   149426 wires   518072 nnz   setup 0.522 s   prove 0.617 s
reduced      55608 constraints    55611 wires   357425 nnz   setup 0.228 s   prove 0.288 s
```

2.29x on setup and 2.14x on prove, against 2.69x on the constraint count. The
shortfall is structural: Groth16's prover splits between terms that scale with
the WIRE count (the `a`/`b`/`c` query MSMs) and terms that scale with the
evaluation DOMAIN, which the constraint count fixes and no amount of wire
reduction touches (the QAP FFTs, the `h` query MSM).

### Wire compaction

The middle column used to read `153605` in both rows. Both reduction passes
abandon wires — linear substitution deletes the constraint that defined an
intermediate, CSE abandons the loser of a merged pair — and neither renumbers,
because both index scratch arrays by wire id for the duration of a fixpoint
round. So Y carried **1.49x more wires than circom** on this circuit while
emitting 1.86x fewer constraints.

`compact_wires` runs once, after the fixpoint, and drops what no surviving
constraint mentions: 153,605 → 55,611, i.e. 1.86x *fewer* than circom on both
axes. It costs ~1.6% of compile time, so Y stays at compile-time parity.

What it buys, measured separately (`what_compaction_buys_at_proving_time`):

```
uncompacted   55608 constraints   153605 wires   setup 0.281 s   prove 0.363 s   pk 25.4 MB
compacted     55608 constraints    55611 wires   setup 0.251 s   prove 0.306 s   pk 10.5 MB
```

1.17x on prove, 1.11x on setup — and **2.42x on the proving key**, which is the
part that is exact rather than a timing ratio, since Groth16 stores a G1 element
per wire. For a shipped circuit that is distribution size and prover memory.

Two things the pass has to get right, both of which produce a *valid proof of the
wrong statement* if missed rather than a crash:

- **Every consumer of a wire id must be renumbered together** — the constraints,
  the witness recipes (keys *and* the linear combinations inside them), the name
  table, the three boundary lists, `unconstrained_hint_vars`, `next_var_id`. This
  is the failure that made `optimize_circuit` produce unprovable circuits once
  already (CLAUDE.md gotcha #8).
- **The renumbering must be stable and the boundary must be live outright.**
  Inputs are consumed positionally in wire order, so a permuted map shuffles a
  circuit's arguments; and an input that is never read appears in no constraint
  at all, so liveness-by-constraint-scan would collect it and silently change the
  circuit's interface.

`tests/zk_wire_compaction.rs` covers both, over *both* front ends — which is not
redundant. circom declares outputs at the top of a template, so its boundary is a
live low-numbered prefix that compaction maps to itself, and a version that
forgot to renumber the boundary lists passes every circom case. Y binds its
return value last. Seven mutations of the pass were checked; the boundary one is
caught by the Y cases only.

### How it got here

The first version of this front end emitted **765 constraints for `Poseidon(2)`
against circom's 517**, and ran a 200-hash chain at **0.59x** — slower. circom's
default `--O2` substitutes linear constraints away and Y's `optimize_circuit`
only did common-subexpression elimination on identical `(A, B)` pairs, so every
`out <== in` in the source survived as a constraint of its own.

`substitute_linear_constraints` closes that. Four things mattered, in descending
order of size, and only the first was the algorithm:

1. **`Fr::inv` in the inner loop.** The pivot's coefficient is inverted to solve
   `k * L = c_w * w` for `w`. Inversion is Fermat's little theorem — a 254-bit
   exponentiation, ~380 Montgomery multiplies, ~6 µs — and it was called once per
   candidate constraint. That was **0.45 s of the pass's 0.55 s**, to divide by a
   coefficient that `<==` always leaves as exactly 1.
2. **Chained definitions need back-substitution, not deferral.** The first
   version refused to eliminate a wire that an earlier expression already read,
   and left it for the next iteration of the fixpoint. A Poseidon permutation is
   a chain of linear layers, so that peeled one level per round and hit the
   10-round safety cap having swept every constraint and every witness recipe ten
   times. Pushing the new definition back into the expressions that depend on it
   (`uses[w]`) finishes the whole chain in one round.
3. **circomlib rebuilds its constant tables per instantiation.** `POSEIDON_C(t)`
   returns a 195-element array of literals and `Poseidon(2)` calls it once per
   instantiation; a 200-hash chain evaluated it 200 times, and `call_function`
   deep-cloned the function's body AST each time to do it. circom `function`s
   cannot touch signals, so they are pure over compile-time values and their
   results cache. (This is the same disease `emit_poseidon` had on the native
   path — see CLAUDE.md gotcha #8 — found again in a different place.)
4. **`LinearCombination::simplify` built a `HashMap` per call**, and the CSE pass
   allocated a `Vec` per distinct product to hold its hash bucket. Both are gone;
   `simplify` sorts and merges in place, and the CSE table is open-addressed
   `u32` indices.

Y's headline numbers against circom (3.45x on a 1000-hash Poseidon chain, 261x on the
polynomial circuit at 1M, and `>345x` at 10M where circom did not finish inside
an hour) were measured on Y's **own** `.ysu` front end, where
`emit_poseidon` folds linear combinations as it builds them. They still should
not be quoted for circom input — the numbers for that are the tables above.

The pass costs Y's own front end 1.5% (0.905 s → 0.919 s on a 1000-hash native
chain) and finds nothing there, which is the expected result and is why its
scratch arrays are allocated on the first elimination rather than up front.

## Supported subset

Templates and template parameters; `function`s including ones returning constant
arrays; `component` declarations, component arrays, and subcomponent signal
access (`c.out`, `c.out[i]`); `signal input`/`output`/intermediate, including
multi-dimensional arrays; `var` with arrays and inline array literals; `for`,
`while`, `if`/`else` over compile-time values; `include` with search paths and
cycle-safe single parsing; all of `<==`, `==>`, `<--`, `-->`, `===`, `=` and the
compound assignments (`\=` included); `**`, `\` (integer division) and `/`
(field division) kept distinct; `assert` and `log`.

**circom 2.1 anonymous components, tuples and whole-array signal assignment**
are supported: `o <== T(targs)(in1, in2)`, `(a, b) <== T()(x)`,
`signal (a, b[n]) <== T()(x)`,
`T()(name <== v)` binding by name, `_` to discard an output, `T()(x);` as a
bare statement when the template has no outputs, and `c.in <== [a, b]` /
`c.in <== other` driving a whole signal array at once.

They are implemented as a **desugaring**, and that is what the test asserts
rather than a constraint count: `o <== T()(i)` emits byte-for-byte the same
`.r1cs` as `component c = T(); c.a <== i; o <== c.out;`. circom satisfies the
same property, which is what made the choice a decision rather than a guess.
The positions circom refuses are refused here too, with its error code recorded
beside each case — an anonymous component inside arithmetic (`TAC01`), under
`<--` (`TAC01`), a bare statement for a template that *has* outputs (`TAC02`),
and every arity and input-name mismatch.

## Refused, by name

The rule from CLAUDE.md applies with extra force here: **a front end that
quietly ignores a construct emits a circuit with fewer constraints than the
source describes, which still proves — just something weaker than the author
wrote.** Nothing downstream records the difference.

- **Non-quadratic expressions** (`a * b * c`, `a / b` with a signal divisor, a
  signal raised above degree 2) — refused with circom's own word for it.
- **Branching on a signal's value** (`if (a)`, `a ? b : c` with a signal
  condition) — the branch decides which constraints exist. The message points at
  the multiplexer idiom.
- **Comparison, boolean and bitwise operators over signals** — these are gadgets,
  not operators. The message names `comparators.circom` / `bitify.circom`.

  > **It is NOT applied to `<--` right-hand sides — that was a bug, fixed
  > 2026-08-11, and it had made circomlib's own `bitify.circom`
  > uncompilable.** `Num2Bits` computes its witness with
  > `out[i] <-- (in >> i) & 1` and then constrains it with
  > `out[i] * (out[i] - 1) === 0` and the recomposition `lc1 === in`. The shift
  > is never an R1CS expression; `<--` exists precisely to compute a value the
  > constraints check afterwards, so refusing it for having "no R1CS form"
  > applied the constraint value model where the witness model belongs. The
  > diagnostic compounded it by naming `bitify.circom`, which was the file that
  > had just failed to compile.
  >
  > `<--` right-hand sides now go through `witness_only_recipe`, which
  > recognises `(e >> k) & 1` and `e & 1` (with `k` compile-time) as
  > `WitnessOp::BitOfLc` before falling back to the ordinary value model — so an
  > unrecognised construct is still refused by name. `Num2Bits`,
  > `comparators.circom` and `aliascheck.circom` compile from unmodified
  > circomlib: a 200-wide `Num2Bits(64)` range check is 13,013 constraints
  > against circom's 13,200, and 300 `LessThan(32)` comparisons are 10,220
  > against 11,100.
  >
  > Guarded by three tests in `tests/circom_frontend.rs` that assert **values,
  > not just satisfiability**. That distinction matters here: `Num2Bits`'s own
  > recomposition constraint already makes a wrong bit unsatisfiable, so a
  > satisfiability-only test would pass on a decomposition that is
  > big-endian — every bit individually valid, the whole thing in the wrong
  > order. `num2bits_bits_are_lsb_first` reads bit 3 back and compares it to
  > `(x >> 3) & 1`.
- **Signal-dependent array indices** — needs an explicit multiplexer.
- **`bus` declarations** (circom 2.1.5+) — flattening one silently would change
  the signal layout a verifier expects.
- **Signal tags** (`signal input {binary} x`) — a tag is a claim other templates
  may rely on to skip a check, so ignoring it can drop a real constraint.
- **`custom` templates** — PLONKish custom gates; Y emits R1CS only, and
  compiling one as an ordinary template would drop the gate it stands for.
- **`assert` over signal values** — it is a constraint on the witness; write it
  as `===` so it is explicit.

## Three bugs this work exposed in the existing back end

- **`WitnessIRGraph::topological_order` was the identity** — `(0..num_signals)` —
  while being named as if it were a dependency order, and the witness solver
  ignored it and walked by wire index instead. That is correct only when a
  recipe never references a higher-numbered wire, which is a property of Y's own
  emitter (it allocates a wire at the moment it defines it) and **not** of
  circom, where `signal output out;` is conventionally declared at the top of a
  template and assigned at the bottom. The forward pass read those wires as
  zero and the witness silently failed to satisfy its own circuit. It is a real
  Kahn sort now, with the positional input assignment split into its own pass
  so reordering cannot disturb it.
- **New `WitnessOp` variants must be added to `solve_r1cs_witness`'s
  "already solved" set**, exactly as CLAUDE.md warns. `MulAddLc` and `DivLc`
  were added for this front end (`a*b + c` fused into one constraint, and the
  field division that `<--` exists for), and omitting them there makes
  back-propagation refuse to fire on anything referencing them.

- **`build_witness_ir`'s `mul_by_output` scan ignored coefficients.** It matches
  a constraint with one term on each side and reconstructs the wire as
  `WitnessOp::Mul(a, b)` — a product of two *wires*, with nowhere to carry a
  scale factor. So `2a * b = t` was reconstructed as `a * b`. It was survivable
  only by accident: the wrong value fails the forward pass's satisfiability
  check, which forfeits the fast exit and sends the whole circuit through the
  back-propagation sweep to rediscover it. All three coefficients must now be 1,
  and anything else falls through to `lc_by_output`, which keeps them.

## The witness domain: values the compiler cannot compute

circom `var`s are **not** restricted to compile-time constants. A template may
write `var y2 = out[1] * out[1];` over a signal, a `function` may be called on
signal arrays, and the result may be shifted, xored and compared — none of which
has an R1CS form. Y refused all of it, and that refusal cost exactly the two
most-used circuits in the ecosystem, `Sha256` and `EdDSA`. They were one gap, not
two.

Such a value is now `Val::Opaque`, carrying the position where it became
unknown. **The rule that makes it safe is that an opaque value may only ever
reach a `<--`.** `<--` emits no constraint, so the `.r1cs` is exactly what a
compiler that *could* evaluate the expression would emit; a `<==`, a `===`, an
array index, an array dimension, a template argument or an `assert` refuses, and
names the origin rather than the use site:

```
5:5: non-quadratic: a `<==` constraint cannot use a value the compiler was unable
to compute — it became unknown at 4:15: `BitAnd` over signals, which has no R1CS
form […]. Only a `<--` may take one, because it emits no constraint.
```

Two mechanisms carry it.

**Signal arrays reach `function`s.** `sha256compression(hin, inp)` hands a
function 256 and 512 input signals; evaluating that as a scalar is what produced
`hin is an array; expected a single value`. A signal slot now maps to a value
slot of one-term linear combinations, shape preserved.

**A condition the compiler cannot evaluate havocs instead of failing.** Neither
branch runs and every variable a branch might assign becomes opaque — the sound
over-approximation, since "unknown" is true whichever way the branch would have
gone. `sqrt` in `pointbits.circom` is five branches and two `while` loops over
the value being rooted, so nothing less would have worked. A `return` inside a
skipped body makes the enclosing function return opaque *immediately*: it may
have returned there with an unknown value or fallen through to return something
else, and opaque is the join.

**The guard that makes havoc sound is `constraint_effect`.** Before skipping a
body, Y checks it cannot emit a constraint, declare a signal or instantiate a
component; if it can, the program is refused with circom's own reasoning, since
which constraints a circuit emits may not depend on a witness value. It is
matched exhaustively with no `_ =>` arm — without the check this would be
precisely the silent-approximation bug CLAUDE.md's design rule exists to
prevent, and worse than usual, because the result still proves.

### What it costs

Only the witness, and only sometimes. **Y still solves a `<--` signal whenever
the author's own constraints determine it** — `Sha256` is exactly that case, all
256 digest bits are `<--` advice and every one is pinned by an
`out[31-k] === fsum[0].out[k]` below it. Where the constraints do *not* determine
it (a square root), satisfiability comes back false and `--witness` refuses to
write a `.wtns` rather than passing off the solver's default zero. A `[Warning]`
naming the count and the origin is printed at compile time either way.

### What checks it

`tests/circom_witness_domain.rs`, in two halves. **Does it work**: `Sha256(64)`
compiled from unmodified circomlib produces the real SHA-256 digest for two
messages, through Y's own witness solver and its own satisfiability check — a
published vector, the same standard the Poseidon interop test sets, and worth
more than a constraint count because a self-consistently wrong circuit has one
of those too. **Is it sound**: six refusal sites, three constraint-emitting
branches that may not be skipped, and the fail-closed witness behaviour.

Verified by mutation, 8 mutations of the lowering, all caught — and **three of
them were not caught by the first version of the tests**, which is the reason to
run it:

- Deleting the template-argument check changed nothing, because every fixture
  was *also* refused by a later check. The cases now assert the phrase belonging
  to their own site, not merely that something failed.
- Ignoring a `return` inside a skipped branch changed nothing, because the
  fixture's fall-through returned an unknown too. The case that separates them
  has both arms returning **constants**, so falling through produces a confident
  `7` for a function that could equally have returned `5` — a wrong circuit that
  compiles, not a refusal.
- Reversing a signal array passed to a function changed nothing — and that one is
  a *confirmation*, not just a hole: the digest is produced by the constraint
  circuit, so the opaque path genuinely cannot affect the `.r1cs`, exactly as
  claimed. Pinning it needed a separate circuit where a signal array reaches a
  function and comes back into a real constraint, weighted by position because
  an unweighted sum is commutative.

### Where Y lands on the two new circuits

Both near parity with circom's default, so they lower the geomean rather than
raise it — 1.352x → 1.341x vs `--O1` — which is the honest outcome:

| circuit | circom `--O1` | circom `--O2` | Y |
|---|---|---|---|
| `Sha256(64)` | 31,264 | 29,014 | 30,729 |
| `EdDSAVerifier(80)` | 17,467 | 7,554 | 12,565 |

`Sha256` is a tie on every axis (wires 31,017 vs 30,482; non-zero terms 174,226
vs 179,454 — 3% denser, the usual two-axis trade). `EdDSA` is 1.39x smaller than
the default and 0.60x against `--O2`, its worst remaining loss.

circomlib's `sha256/` is vendored (upstream `35e54ea`) so the digest test is
hermetic; `EdDSA`'s include chain is not, and is covered by
`tools/circomlib_coverage.py` plus small fixtures for the `sqrt` shape.

## Known gaps

- ~~Input JSON does not accept arrays~~ / ~~input names carry the `main`
  prefix~~ — **both fixed.** `--witness` reads circom's own `input.json`
  verbatim (bare names, arrays, nested arrays) and also accepts the
  fully-qualified `main.a` spelling that appears in the `.sym`. The witness it
  produces agrees with circom's own calculator element for element, and
  `snarkjs wtns check` accepts the pair. Public inputs are bound too — they
  were being solved at zero, which is the only one of the three that was not
  fail-closed.
- `Y_ZK_COMPACT=off` disables wire compaction. It is a differential baseline, not
  a safety valve — with it off, Y carries 1.49x *more* wires than circom on a
  200-hash Poseidon chain.
- `include` resolution is path-based only; there is no package/`node_modules`
  lookup.
- ~~A signal-dependent array index is refused~~ — **fixed**, and zk-email's
  `RSAVerifier65537`, `Sha256Bytes` and `Sha256BytesPartial` compile. An unknown
  index yields an unknown value in the witness domain (`out <-- t[u]`,
  `out <-- in[u]`, `t[u] = 9`) and is still refused wherever the index would
  decide which wire a constraint mentions (`out[u] <== 1`, `c[u].x <== 3`) or
  wherever the value reaches a `<==`. That boundary is circom's, probed case by
  case rather than assumed.
