# PTXAS arithmetic-cut continuation — 2026-10-08

The intermediate-equality cuts in `tval.py` now remain usable after executor
simplification, including consumers that read a complemented carry predicate.
`IMAD.HI.U32` and unsigned `IMAD.WIDE.U32` also honor the selected product model. These changes
improve proof reuse; they add no device ISA or ABI assumptions.

## Recorded verification result

The final strict selected PTXAS stage **passed** with **38 Rust tests, 162 Python
tests and 111 retained artifact cases**, and no failures or skips. The artifact
verdicts are 46 VALIDATED, 41 expected UNPROVED, 22 named REFUSED and two
ASSEMBLY_REFUSED. An expected rejection is a passing regression test; these
111 cases are not 111 validated kernels. The pipeline suite has 29 tests and
the integer abstraction and executor suites have 27 tests each.

The verified toolchain was CUDA/PTXAS/nvdisasm 13.4.92, Python 3.14.7 and Z3
5.0.0. The retained
[verification summary](../../target/verification/ptxas-arithmetic-cuts-2026-10-08-verified/summary.md),
[machine-readable results](../../target/verification/ptxas-arithmetic-cuts-2026-10-08-verified/results.json)
and [input inventory](../../target/verification/ptxas-arithmetic-cuts-2026-10-08-verified/inputs.json)
bind that completed implementation run. The evidence directory also contains
command logs, Python test records and each pipeline case's PTX, cubin, SASS,
hashes and validation transcript. `target/` is generated output; preserve it
when archiving the local evidence. This documentation update follows that run
and does not change its source, tests or retained evidence.

## Carry predicates must match in both polarities

The existing unfinished executor change normalized each recorded arithmetic
value and carry with `simplify`, matching the arithmetic retained by register
writes. A raw cut root that disappears during a register write cannot replace
later consumers, even after its equality has been proved.

Activating these cuts exposed a completeness regression in freshly assembled
add and subtract controls. The genuine add control proved its value and carry
equal, then reported `store 1: sat` after substitution. PTX retained a consumer
`If(carry, 1, 0)`, while SASS's carry was `Not(q)` and its consumer simplified
to `If(q, 0, 1)`. Replacing only `Not(q)` missed that SASS consumer and cut the
PTX consumer alone. This SAT result described the weakened proof formula, not
a wrong machine translation.

`boolean_cuts` records both the carry and its simplified complement, plus
their one-bit encodings, mapped to a fresh Boolean and corresponding values.
The one-bit encodings preserve masks whose low bit simplifies to a raw bit
extraction and no longer contains a Boolean predicate. Once the carry equality
is proved, an actual execution extends to these new symbols by assigning the
fresh Boolean its actual carry value. The substitutions retain that execution. Literal
true/false carries keep their known values and need no substitution. Guarded
register writes retain their conditional merge and their previous value on
disabled paths; cuts replace arithmetic children rather than discarding guards.

The fresh add/subtract controls now validate. Their deliberately dropped-carry
mutations remain unproved with explicit SAT diagnostics. Executor regressions
also check exact recorded roots, guarded writes, complemented consumers,
one-bit encodings, masks and a changed carry polarity that must remain
distinguishable.

## Failed cut queries must still check the original stores

The standing `ptx_carry_chain` and `ptx_integer_ops` controls exposed another
completeness failure. Simplification can flatten an intermediate addition out
of one side's expression while the other side retains a matching root. The cut
then loses a relation between its fresh value and the remaining arithmetic.
Even the cut formula's concrete multiplication rung can report SAT in this
situation; its fresh intermediate values remain an overapproximation.

When a cut store query fails and its expressions differ from the originals,
the validator retries the untouched store expressions under the same guards,
launch domain and refinement ladder. Acceptance still requires an UNSAT proof.
This restores the standing controls and avoids treating a missing structural
correlation as a final failure. A fresh O3 carry-probe test requires this retry
to prove store 11; zeroing that stored high word must still report SAT.

## High-word accumulation must use the selected product

`mul_hi_wide` formerly constructed a concrete multiplication even in `uf` and
`wide` mode. A SASS high-word accumulation and the corresponding PTX operations
therefore used different product vocabularies, forcing concrete refinement.
A shared `full_product` helper now concatenates the selected abstract halves,
then `mul_hi_wide` zero-extends that 64-bit product before adding the 64-bit
register pair. The 65th bit remains the carry out. Concrete modes retain one
full 64-bit multiplication rather than splitting its high and low halves into
different-width multiplications. `IMAD.WIDE.U32` uses the same helper. This
avoids an extra low-32/high-64 multiplier identity in the existing integer-ops
control; its high-word store can now prove directly. Signed wide multiplication
keeps its existing sign-extension semantics.

Solver controls compare this result with the split low-word/high-word PTX
calculation in both abstract modes, compare direct/default execution with exact
65-bit arithmetic, and check literal boundaries against Python integers.
Changed low addends and dropped overflow carries remain SAT controls.

A new pipeline test freshly assembles PTX that lowers to `IMAD.HI.U32` with a
register-pair addend and stores both the high word and carry. The genuine case
validates with twelve obligations: eight access obligations, one value cut,
one carry cut and two stores, with no concrete refinement. Zeroing a product
operand or discarding the carry output produces separate expected SAT results.
Mutations are labeled SASS test inputs; their retained cubin is the genuine
assembler output, not a machine artifact implementing the mutation.

## Verification and remaining scope

The existing Rust gate and focused workflow already register both integer
suites and the fresh pipeline suite. Run all PTXAS checks from the repository
root; the workflow chooses a new evidence directory automatically:

```sh
PATH=/opt/cuda/bin:$PATH Y_TVAL_PYTHON=venv/bin/python \
  venv/bin/python tools/verify.py --stage ptxas
```

The controls require Z3 and CUDA assembly/disassembly tools; they require no
GPU. They do not replay device ABI probes or establish GPU execution. Proof
mode continues to require matching licensed `sm_89` targets.

The existing standing `bn254_permute`, `bn254_sub_vec`, `ptx_carry_chain` and
`ptx_integer_ops` artifacts all validate in fresh interpreters at
`NS=12, B1=3, B2=15`. This is a control check on those artifacts; the focused
workflow separately rebuilds and binds its pipeline subjects to fresh cubins
and disassembly.

| Standing subject | Verdict | Obligations | Observed seconds |
| --- | --- | ---: | ---: |
| `bn254_permute` | VALIDATED | 30 | 0.3 |
| `bn254_sub_vec` | VALIDATED | 62 | 0.6 |
| `ptx_carry_chain` | VALIDATED | 100 | 1.7 |
| `ptx_integer_ops` | VALIDATED | 66 | 1.2 |

The observations are individual proof runs, not a controlled performance
comparison with the historical September table. The `ptx_integer_ops` result
uses two Int proofs; its wide-product high word no longer needs the extra
mixed-width multiplication identity. Standing-control scratch logs remain in
`/tmp/y-ptxas-standing-controls-2026-10-08/`, which is ephemeral.

These changes do not establish validation of `bn254_fr_mul_fast` or remove the
historical solver wall. Simplification can flatten intermediate sums so their
recorded roots are absent from later expressions, and field-kernel obligations
can still time out. A bounded run that proposes intermediate correspondences
without finishing the store obligations supplies no validation verdict.

The final field-kernel trial used `NS=4, B1=1, B2=1` and a 120-second process
deadline. It ended with timeout exit 124 after discharging 24 access obligations
and proposing 276/276 pairs, without completing a sweep or returning a validator
verdict. Its scratch log is `/tmp/y-ptxas-bn254-final.log`; an earlier
240-second trial also timed out. No counterexample or new solver-wall threshold
was established by these partial runs.

The next investigation is partial-sum query cost and cut reachability in that
field kernel. Keep explicit solver budgets, a process deadline, positive
controls and mutations with concrete rejection diagnostics. A `flat=False`
experiment increased store-DAG size about 15% without improving SASS cut
reachability, so it was not adopted as a global change.
