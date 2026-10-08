# PTXAS verification extension

This document preserves the October 1 extension review and its historical suite
and case counts. The [October 8 arithmetic-cut review](ptxas_arithmetic_cuts_2026-10-08.md)
records the current focused PTXAS result: 38 Rust tests and 162 Python tests
passed with no skips; 111 fresh cases produced 46 `VALIDATED`, 41 `UNPROVED`,
22 `REFUSED` and 2 `ASSEMBLY_REFUSED` results. The pipeline suite now has 29 tests,
and both integer suites have 27. Validation remains licensed only for `sm_89`;
the run establishes neither full-workspace verification nor GPU execution.
The NVIDIA driver was inaccessible, and the final `bn254_fr_mul_fast` trial
reached 24 access obligations and 276 value candidates before timing out at
120 seconds without a validation verdict.

The focused workflow now includes a PTXAS stage. Run it on its own with:

```bash
python3 tools/verify.py --stage ptxas
```

The stage assembles emitted PTX across six architectures, rebuilds translation
controls in isolated directories and retains their PTX, cubins, SASS, tool
versions and validator results. It checks hashes and requires evidence for
every passing pipeline test. The [workflow guide](workflow.md) describes the
commands, dependencies, report formats and remaining execution assumptions.

The extension found and corrected these errors:

| Trigger | Previous behavior | Corrected behavior |
| --- | --- | --- |
| `cp_async` with a 4- or 8-byte transfer | Compiler emitted `.cg`; actual PTXAS assembly failed | Emits `.ca` for 4/8 bytes and preserves `.cg` for 16 bytes; all three widths assemble |
| A uniform SASS parameter load changed to `@!PT` | A wrong translation still validated with 18 obligations | Uniform writes honor predicates; the mutant is unproved while the genuine output validates |
| A counted barrier treated as a full-block barrier | Removing or adding the count could still validate | Counted forms are explicitly refused until their arrival semantics are modeled |
| `ULDC.64` reading an unmapped constant-bank half | Missing words were invented as zero and could validate | Each half uses the strict constant-bank reader; unmapped words are refused |
| Semantically equal float operands with different Z3 node IDs | Rounded GEMM lost its accumulator relation despite unchanged PTX/cubin/SASS bytes | Licensed FADD/FMAX operands are ordered by their bit-pattern values, preserving equality under aliases and substitution |
| `IMAD.WIDE.U32` changed to signed `IMAD.WIDE` | A wrong high product word still validated | Signed operands are sign-extended and unsigned operands zero-extended before the wide product; opposite-signedness mutations are unproved |
| Genuine global 64/128-bit and shared scalar/64/128-bit zero stores using `RZ` | Register-number parsing raised `ValueError` | A shared vector source reader preserves every zero word; genuine outputs validate and nonzero mutations are refuted |
| Shared PTX symbol or register plus a literal byte offset | Valid addresses such as `[slot+4]` crashed in the operand reader | Offsets preserve register width and shared-window wrap; unsupported expressions refuse by name |
| A carry reader with no writer on its executing path | PTX invented an entry CC value of zero, allowing a forced-zero machine carry to validate | Every carry read must prove CC was initialized under its guard; SAT/UNKNOWN dominance checks refuse |
| `R0` used where a SASS predicate operand is required | The reader treated it as `P0`, and a wrong register file could validate | Predicate readers and destinations require their supported register-file spelling |
| Variable or hexadecimal 64-bit PTX shift amount | Decimal-only parsing raised `ValueError` | PTX amounts use the 32-bit operand reader; unsupported machine wide shifts still refuse explicitly |
| Semantically equal integer multiplier operands after aliases or substitution | Z3 allocation IDs selected different UF operand orders, losing true equality | Unsigned bit-pattern ordering preserves both product halves and changed products remain distinguishable |
| Same-named bitvectors, functions or arrays with distinct declarations/sorts | The exact integer encoder merged independent inputs and could turn concrete SAT into false UNSAT | Source declaration identity and every input/output sort remain distinct in the integer encoding |
| `UNPROVED` evidence expecting `unknown` or another broad pattern | Matching uncertainty could satisfy an expected rejection | The reader independently requires SAT or unequal counts/widths and matches only those concrete diagnostics |
| A retained `python -c` validator command with substituted program text | Correct input paths and fabricated matching output could pass command binding | Producer and reader share and compare the exact fresh-process validator program |
| Extra operands in `ISETP`, `IMAD.X` and other fixed instruction forms | Trailing operands were silently ignored and could still validate | Supported forms consume the whole operand list; unsupported counts and comparison modifiers refuse by name |
| Genuine global loads such as `ld.global.u32 %r0, [%rd0+4]` | The 64-bit register reader treated the address expression as a register and crashed | Global loads and stores share an explicit 64-bit address reader with signed literal offsets; unsupported bases refuse |
| Changed global address or store guard | The validator discarded SAT versus UNKNOWN in its diagnostic | Address failures report SAT only when every candidate comparison in the failed row is SAT; guard failures retain the solver result |
| Unrewritten quantified/bound terms and lambda-array reads in the integer encoder | Unsupported terms raised a Z3 exception rather than returning unknown | Sort/application guards preserve the explicit unsupported boundary; a non-Boolean root cannot become a Boolean input |
| A substituted validator executable or an extra successful command | Matching command paths and output could still pass | Operations match their successful probes and the workflow's discovered tool paths; only known commands in execution order are accepted |
| Floating-point or Boolean obligation count in validator stdout | Python equality made `2.0 == 2` and `True == 1` pass transcript comparison | Both JSON records independently require an integer obligation count |
| An inventory left behind with no completed case reports | Optional evidence reading bypassed the inventory | Every present inventory is checked, including empty or incomplete runs |
| A scalar global load widened to 128 bits with only its first word consumed | Extra reads disappeared under value abstraction and the mutant validated | Every load atomically records its full width; paired loads require matching widths before abstraction |
| Global loads unused by a loop's stored result | Loop proof obligations could ignore their count, address, guard and width | Loads participate in live-in discovery and are compared in each executed region; repeated PTX header loads refuse |
| A shared 128-bit access at `tid.x * 16 + 4` | Four-byte alignment discharged and the translation validated | Natural alignment covers the entire 4/8/16-byte access wherever its guard holds; the misaligned control is unproved with SAT |
| Scalar SASS `R0.64` or load destination `P0` | Width suffixes were ignored, or the other register file aliased R0 | Operand spelling and the entire supported regular-register span are checked; unsupported forms refuse |
| A scalar global load writing RZ | Destination integer parsing crashed | The load still records its access and discards the result; a changed observable value is refuted |
| A PC-tagged SASS store without a trailing semicolon | The parser silently skipped the added effect and could validate | Malformed instruction rows produce a named refusal |
| Missing, extra, repeated destination or empty PTX vector components | The executor consumed the list without checking the declared vector width | Vector widths and distinct load destinations are checked; memory operands consume the entire list |

The floating-point change adds no reassociation, multiplication commutativity
or FMA contraction. The existing FADD/FMAX commutativity assumptions remain
explicitly gated. Tests preserve distinctions between different operands,
FMUL/FMIN operand orders and fused versus separately rounded arithmetic.

At the October 1 snapshot, the fresh pipeline had twenty-three tests producing
eighty-two cases: thirty-eight
validations, thirty-two expected unproved results, ten named refusals and two
assembler refusals. One validation is an equivalent multiplier operand swap,
recorded as mutated SASS rather than a genuine disassembly.
Additional cases cover O2/O3 integer predicates and computed addresses, signed
and unsigned comparisons and narrow loads, store guards, access widths, wide
products, variable right shifts, carry chains, vector zero stores, global
word/subword/vector/float loads with positive and negative offsets, and instruction
operand counts. The O0
control explicitly refuses unmodeled `LDC.64`;
the run still passes because that refusal is the declared test expectation.
The validator suite had sixteen tests and included genuine assembled controls;
separate integer abstraction and executor suites check solver congruence, exact
SAT/UNSAT agreement, guarded CC initialization and unsupported operands. The
integer abstraction suite had twenty-two tests and the executor suite fourteen.
The arithmetic controls cover nary addition/multiplication, modular wrapping,
masks, shift boundaries, sign extension and wide-product extracts. Quantifiers,
bound variables and unsupported array forms return unknown without a proof claim.
The emitted assembly
gates contained twenty-eight tests and covered `sm_80`, `sm_86`, `sm_89`, `sm_90`,
`sm_100` and `sm_120` with the installed CUDA 13.4 toolchain.

Negative cases assert their specific diagnostic, so a timeout cannot satisfy a
refutation control. A floating-point `sat` result describes the abstraction;
device divergence needs additional evidence. Mutated SASS is a test input and
is identified separately from the genuine disassembly of its retained cubin.
These checks require no GPU and do not establish GPU execution or replay the
device ABI probes. The existing subtraction carry convention is preserved;
this review does not independently establish SASS carry or funnel-shift semantics
on hardware. Unmasked `SHF.L.U32` amounts at least 32 remain conservative, and
variable wide SASS shifts remain refused.

The global load-width check is deliberately conservative: PTXAS can legally
narrow a vector load when only one component is used, and that genuine lowering
now remains unproved. The widening controls retain scalar source loads, where
the added vector words are unused, and the narrowing controls consume all four
source words. Shared vector alignment follows the total access size specified
by the [PTX address rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands).
Misalignment is outside the shared model's supported semantics; its SAT licence
failure is not a GPU execution counterexample. Loop access checks compare regions
and do not establish general instruction motion across loop boundaries.

The evidence reader also rejects changed artifact hashes, path escapes,
unreported test IDs, missing case evidence and disassembly failures mislabeled
as expected assembler rejection. Scratch profiles select fixture targets;
compiler failures cannot inherit stale PTX from an earlier invocation.

The continued review reproduced acceptance of foreign command inputs, mismatched
optimization/target flags, non-ELF cubins, missing mutation inputs and contradictory
validator output. The reader now binds the assembly and disassembly commands to
their retained files, checks SASS output and target identity, and compares validator
JSON and stdout with the case metadata. An atomic case inventory also makes removal
of one subcase fail even if its parent test has other retained cases. Eight new
reporting regressions exercise these failures, including rehashed corrupt files
and a solver timeout presented as a refutation. Timed-out or signal-terminated
assembler processes cannot count as expected syntax or target rejections.

At that snapshot, the command-reporting suite had thirty-six tests. It included
matching but invalid uncertainty and equal-structure diagnostics, specific accepted SAT and
structural failures, substituted validator code and executables, command ordering,
independent transcript types and abandoned inventories. A registration check
compares every Rust-invoked Python suite with both workflow plans. The additional integer
controls are registered in both the focused PTXAS stage and full-workspace plan.
All source and documentation edits in that review preceded its final verification
snapshot.
