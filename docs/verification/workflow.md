# Running Y's verification workflow

Run from the repository root:

```bash
python3 tools/verify.py
python3 tools/verify.py --full
python3 tools/verify.py --stage ptxas
```

The default command runs seven verification stages with the `zk` feature. The
`--full` command also runs every workspace test target. Both use Cargo's offline,
locked mode, so their dependencies must already be available locally.
`--stage` selects individual stages and can be repeated. Selected runs identify
their smaller scope in the report. Selecting the workspace requires `--full`.

| Stage | Checks | Required tools |
| --- | --- | --- |
| `smt` | Machine arithmetic, signed runtime loop bounds and empty GEMM dimensions, reference aliases, safe invariants, reference types, scalar rules, bounds and linear tracking | Cargo, Rust, Z3, clang, `ptxas` |
| `proofs` | Every registered Coq proof and proof rejection probes | Cargo, Rust, `coqc` |
| `translation` | PTX translation validator soundness regressions | Cargo, Rust, Python with `z3-solver` |
| `ptxas` | Emitted PTX assembly, portability, fresh cubin/SASS controls and validator mutations | Cargo, Rust, Z3, clang, Python with `z3-solver`, `ptxas`, `nvdisasm` |
| `artifacts` | Exact-PV artifact binding, build isolation, assembly and Rust/Python shape/tensor contracts | Cargo, Rust, Python with `z3-solver`, `ptxas`, `nvdisasm` |
| `launch` | Recording CUDA driver checks of checked exact-PV geometry, typed ABI, ownership, limits and error propagation | Cargo, Rust |
| `workflow` | Audit input binding, strict prerequisite handling, reporting and runner regressions | Cargo, Rust, Python |

The proof rejection probes compile an independent grid-stride specimen. They
check that the production proof-evidence gate rejects admitted proofs and
axioms, and that Coq rejects unfinished proofs, a wrong stride and removal of
the worker-count licence. Positive controls compile and expose closed proofs.

The PTXAS stage assembles emitted intrinsic, coprocessor, control-flow and ZK
witness kernels. Its portability tests cover `sm_80`, `sm_86`, `sm_89`, `sm_90`,
`sm_100` and `sm_120`; strict execution requires a toolkit supporting these
targets. Scratch hardware profiles select each fixture's target without a GPU
probe or the developer's hardware configuration.
Async-copy checks assemble all three supported transfer widths (4, 8 and 16
bytes), including the compiler's cache-modifier choice, and reject illegal widths.

Fresh translation cases rebuild PTX into cubins and disassemble those exact
cubins. They check rounded arithmetic against contraction at O1 and O3, the
GEMM FMA/rounded/contracted triple, exact PV, shared barriers and writes, global
load hoisting, target preservation, invalid syntax and incompatible targets.
Integer cases cover O2/O3 comparisons, computed addresses, predicated stores,
8/16-bit signed and unsigned loads, access widths, signed/unsigned wide products,
variable signed/unsigned right shifts and addition/subtraction carry chains.
Global 64/128-bit and shared scalar/64/128-bit zero stores include changed-value
controls; shared symbol-plus-offset addresses are exercised by genuine PTXAS
coalescing into a 64-bit store. An O0 case requires an explicit `LDC.64` refusal
until that instruction is modeled. Variable 64-bit left shifts parse correctly
on the PTX side but refuse the unsupported SASS `USHF.L.U64.HI` lowering.
Global loads and stores share a 64-bit address reader for register/literal bases
and signed literal byte offsets. Fresh word, signed-halfword, vector and float
load cases cover these offsets; changed addresses and store guards require their
specific SAT diagnostic. SAT address diagnostics require every candidate for
that failed access to have checked SAT, so uncertain pairing is not a refutation.
Refutation controls require their expected solver `sat` diagnostic or named
structural mismatch, so a timeout does not satisfy them. Floating-point `sat`
can reflect the abstraction's limits rather than device divergence. Additional executor
regressions cover predicated uniform-register writes and conservative refusals
of counted barriers, uniform predicates and unmapped constant-bank reads.
Wide multiplication must preserve operand signedness in both product words;
equivalent register operand swaps still validate while changed products are SAT.
Two additional Python suites test integer abstraction congruence and exact
encoding, including aliases, substitution and distinct same-named inputs,
functions and arrays. Exact encoding retains declaration identity and all sorts,
so independent inputs cannot become a false UNSAT proof. Carry readers require
proof that their executing paths have initialized CC; solver uncertainty refuses.
Predicate operands from other register files refuse instead of aliasing predicates.
Supported SASS forms consume their entire operand list; trailing operands and
unsupported comparison modifiers cannot disappear from the model. Additional
integer controls cover modular arithmetic and refuse unsupported quantifiers,
bound variables and array forms without crashing or issuing an UNSAT proof.
Global loads record full access widths even when downstream code discards words;
width changes are checked before abstraction in straight-line and loop validation.
Loop loads also preserve regional counts, addresses and guards. Global loads in
the PTX loop header refuse until their repeated accesses have a modeled pairing.
The strict width check conservatively leaves legal vector narrowing unproved.
Shared vector accesses require natural 8/16-byte alignment under their execution
guards, following the [PTX address rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands).
The memory regression suite checks alignment, atomic load-width logs, malformed
vector operands and read-through memory shapes. Regular SASS register spans stay
within the supported R0..R254 range; scalar `.64` operands and other register-file
destinations refuse. Scalar load destinations using RZ discard the result; vector
RZ destinations remain explicitly unmodeled. PC-tagged instructions with a missing
semicolon cannot disappear from the SASS program.
Validation uses the existing ISA/ABI models
and floating-point assumptions; device ABI referee checks remain separate.
Public verdicts require matching unqualified `sm_89` targets. sm80/sm86
subjects continue to assemble in portability controls and receive explicit
architecture-license refusals in the translation validator. Fresh sm89 O3
counterparts preserve meaningful positive and numeric mutation checks.
PTX directive, brace and label lines cannot hide executable payloads; unknown
directives and text outside the recognized body refuse before symbolic execution.
The dedicated parser controls preserve fresh genuine loop and nested positives.
Nested lexical scopes refuse because the register model is flat; this is
separate from supported nested loops expressed with labels and branches.

The focused stages need the CUDA assembly tools but do not need a GPU. A focused
pass covers the listed checks; it does not establish device execution or a
passing workspace. The workspace contains additional conditional tools and GPU
tests. Their skip notices and ignored-test counts appear in the report.
`tests/exact_pv_checked_gpu.py` is an optional independent Python-integer
hardware oracle using the checked cubin path. It covers full U32/I8 values,
outputs beyond 2^53, odd channel counts, dense offset views, aliases and
invalid lazy tensor views. `Y_EXACT_PV_ARTIFACT` selects an existing bundle;
otherwise it builds a private validated bundle. Missing hardware is reported
as skipped, and `Y_VERIFICATION_STRICT=1` makes that suite fail on skips.

## Strict and partial runs

Strict mode is the default. Missing prerequisites, ignored tests and reported
runtime skips prevent a complete pass. The selected Rust gates fail on missing
prerequisites under `Y_VERIFICATION_STRICT=1`; the Python verification suites
also fail when they skip a check.

For a development machine with missing tools:

```bash
python3 tools/verify.py --allow-skips
```

This permits an exit code of zero for a partial run with at least one executed
stage. The report still says `incomplete` and records the missing tools and skip
reasons. Assertion failures, invalid evidence, timeouts and changed inputs
remain failures. A run that skips every stage returns 2 even with this option.

| Exit code | Meaning |
| --- | --- |
| `0` | Passed, or explicitly permitted incomplete execution with `--allow-skips` |
| `1` | A check failed, evidence was invalid, execution failed, or inputs changed |
| `2` | Incomplete execution without permission to accept it, or no executed stages |

Unset `Y_ALLOW_UNVERIFIED_INVARIANTS` before running. The compiler treats the
variable's presence as a bypass even when its value is empty or `0`, so the
workflow rejects every value.

Tool discovery records executable paths and version probes. `Y_Z3_PATH` and
`Y_TVAL_PYTHON` select explicit executables; an invalid override is reported
rather than replaced. Python discovery otherwise tries the repository's
`venv`, `.venv`, the running interpreter and `python3`. Translation checks
require an interpreter that can import `z3`.

## Retained evidence

Each run creates a new directory under `target/verification/` containing:

- `summary.md`: stage outcomes, counts, skipped checks and links to logs.
- `results.json`: the `y-verification-workflow-v1` report with commands, tool
  probes, durations, exit codes, input changes and individual Python results.
- `inputs.json`: SHA-256 hashes of the source, test, proof, native, Python,
  configuration and documentation inputs present before execution, including
  untracked files.
- One log per executed stage and Python result sidecars for the relevant suites.
- `ptxas-ptxas/`: fresh assembly cases with source PTX, cubin, disassembly,
  tool versions, commands, validator output and `case.json` identities. Full
  workspace execution retains a separate `workspace-ptxas/` directory.

PTXAS case reports use `y-ptxas-case-v1`. Every passing pipeline test must have
retained case evidence. An atomic `inventory.json` registers every started
subcase by name, parent test and directory; all registered subcases must remain
in the final evidence, even when another subcase from the same test passed.
Every present inventory is checked, including one with no completed cases.
The runner rechecks file hashes, command exits,
expected verdicts and positive obligation counts before accepting it. Assembly
commands record their working directory and must name the retained PTX and
cubin with the reported optimization and architecture. The cubin must be ELF;
the SASS must match retained disassembler output and its target declaration.
Operation executables must match the successful PTXAS, disassembler and Python/Z3
probes and the workflow's discovered tool paths. Extra commands, unknown command
fields and out-of-order operations fail evidence acceptance.
The validator command must use the exact shared fresh-process child program and
the recorded source and SASS input, and its
retained JSON and stdout must agree with the case verdict, diagnostics and
obligation count. Both JSON records independently require integer obligation
counts; Boolean and floating-point lookalikes are rejected. Negative results
must match their specific diagnostic.
For `UNPROVED`, the reader independently recognizes an explicit SAT result or
unequal counts/widths and applies the expected pattern only to those diagnostics.
Matching timeout, unknown, equal counts or uncertain address/guard messages cannot
satisfy a negative control.
Assembler timeouts and signal termination fail rather than satisfy a rejection.
Genuine translations, deliberately mutated disassembly and assembler rejections are
identified separately in the summary. Mutated SASS is a validator test input;
it must be retained separately and differ from the genuine disassembly. The
retained original cubin still contains the assembler's genuine output.
These records are local test evidence, not loader receipts or certificates.

The workflow compares the initial and final input snapshots. Created, removed
or changed inputs fail the run. Build caches, generated Coq products and the
run's evidence directory are excluded from those snapshots. This comparison
detects final differences; it does not lock the working tree against concurrent
edits or detect an edit restored before the final snapshot.

Python reports use `y-verification-unittest-v1`. They preserve each test's
outcome and skip reason, including tests prevented by class or module setup.
The workflow requires one complete report per expected suite, consistent
counts and exit status, and the requested strictness. A successful Cargo
process without executed Rust tests or required Python evidence fails.

Results are written after failures as well as successes. Existing evidence
directories are never reused. To choose a location or inspect the commands:

```bash
python3 tools/verify.py --output /tmp/y-verification-run
python3 tools/verify.py --list
python3 tools/verify.py --full --list
python3 tools/verify.py --timeout 900
```

`--timeout` sets the deadline for each stage in seconds (default 1800). A timed
out stage fails and its process group, including descendant solvers and
compilers, is terminated. Completed logs and the final report are retained.

Individual gates remain runnable through Cargo. For example:

```bash
Y_VERIFICATION_STRICT=1 Y_Z3_PATH="$PWD/venv/bin/z3" \
  cargo test --offline --locked --features zk --test smt_reference_aliases
```

The earlier review and its existing workspace failures are recorded in
[the verification review](verification_workflow_2026-10-01.md).
