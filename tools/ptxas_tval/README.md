# `ptxas` translation validator

Symbolically execute a kernel's PTX and the SASS `ptxas` produced from that exact
file; ask an SMT solver whether the two can ever store different values.

**An opcode or operand form neither executor models is a HARD ERROR, never a
guess.** A symbolic executor that guesses at an instruction it does not know
produces a proof about a program nobody wrote.

The write-up — method, results, findings, what is *not* claimed — is
[`docs/ptxas_translation_validation.md`](../../docs/ptxas_translation_validation.md).

The [October 8 arithmetic-cut review](../../docs/verification/ptxas_arithmetic_cuts_2026-10-08.md)
records the latest strict PTXAS stage: 38 Rust tests and 162 Python tests passed,
with no skips, and 111 fresh cases produced 46 `VALIDATED`, 41 `UNPROVED`,
22 `REFUSED` and 2 `ASSEMBLY_REFUSED` results. The pipeline suite has 29 tests;
the integer abstraction and integer semantics suites have 27 each. Validation
remains licensed only for `sm_89`. This focused run does not establish full
workspace or GPU execution: the NVIDIA driver was inaccessible. The final
`bn254_fr_mul_fast` trial reached 24 access obligations and 276 value candidates
before timing out at 120 seconds without a validation verdict.

## Requirements

`python3` with `z3-solver`; `ptxas` and `nvdisasm` from the CUDA toolkit.
The full corpus needs a GPU toolchain and takes minutes to hours per kernel.
`cargo test --test exact_pv_artifact_binding` runs the artifact-binding tests;
its real `exact_pv` compilation/validation arm requires these tools and Z3.

## Run

The repository verification workflow rebuilds isolated PTXAS controls and
mutations, retains PTX/cubin/SASS with hashes and command logs, and requires
the expected validation or refusal for each case:

```sh
python3 tools/verify.py    # from the repository root; no GPU required
python3 tools/verify.py --stage ptxas
```

It also assembles emitted PTX across six architectures and checks strict tool
availability. See the [workflow guide](../../docs/verification/workflow.md).
To run only the fresh pipeline checks with retained evidence:

```sh
Y_VERIFICATION_STRICT=1 Y_PTXAS_EVIDENCE_DIR=/tmp/y-ptxas-evidence \
  venv/bin/python tests/ptxas_pipeline_regressions.py
```

Cases identify genuine assembler output, deliberate SASS mutations and
assembler refusals separately. These are test records, not deployable receipts.

Every global load records its full access width, including words unused by
later instructions. Paired loads must preserve that width before their values
are abstracted; legal compiler narrowing of a partially consumed vector can
therefore remain unproved. Shared accesses must prove their natural 4/8/16-byte
alignment under their execution guards. Unsupported vector and register spans,
scalar `.64` operands and malformed PC-tagged instruction lines refuse by name.
Loop validation checks global loads by region and refuses header loads whose
repeated execution has no modeled SASS pairing.
Nested validation checks complete load address/guard/width traces in every
region. Both loop paths refuse uncomposed header value effects and cross-thread
effects, and classify self-branches as removable traps only when unreachable.
Every validation path requires matching explicit `sm_89` target declarations
and uses the complete legal modeled CUDA x launch domain. Other matching
targets refuse: ABI/ISA and empirical float assumptions are currently licensed
only on sm89. Cross-architecture assembly tests remain assembly evidence.
Straight-line `-O2`/`-O3`
output can validate; general unrolled loop correspondence remains unsupported.
Arithmetic cut records use the same simplified terms as register writes.
Proved carry cuts cover both Boolean polarities and one-bit encodings,
preserving consumers whose conditions Z3 rewrites while simplifying. Failed
cut queries retry the original store expressions when simplification has lost
their correlations. `IMAD.HI.U32` uses the selected
multiplier's shared product halves for its full 65-bit sum and carry, just as
the other modeled multiply instructions do. The
[arithmetic-cut review](../../docs/verification/ptxas_arithmetic_cuts_2026-10-08.md)
describes the fresh controls and remaining field-kernel limit.
`ptxsource.read` checks a full-line grammar before all specification scanners:
executable text sharing directive/brace/label lines, unsupported directives and
text outside the recognized body refuse. Pure repeated declarations are read
individually; `.reg`/`.loc` cannot conceal stores or control updates.
Nested lexical scopes refuse until register shadowing is modeled; label-based
nested loops remain supported.

```sh
./build_corpus.sh    # tests/*.ptx -> corpus/ and o1/, via ptxas + nvdisasm
./regress.sh         # standing positive/negative controls; needs generated corpus
python3 fpsem_abi.py # referee the seven float facts against the device
```

For verified `exact_pv` execution, run from the repository root:

```sh
python3 tools/ptxas_tval/exact_pv_artifact.py build tests/exact_pv.ptx /tmp/exact_pv_verified
python3 tools/exact_pv_bridge.py --verified-artifact /tmp/exact_pv_verified
```

The destination must be new. The builder preserves PTX -> `ptxas -O1` cubin ->
SASS -> `loopval.validate`, then publishes the artifacts with SHA-256 identities
only after successful validation. Both verified loaders check the pinned
receipt and files before loading the checked cubin bytes directly. They refuse
changed or missing artifacts and devices other than `sm_89`; no PTX compilation
or fallback occurs at load. Bundle format v1 supports CUDA ELF ABI 8 only.
The builder and both loaders also require the fixed complete PTX subject
reviewed against `ExactPvExact.v`, via `exact_pv_subject.sha256`. Comments and
whitespace can change; instructions, operands and declarations cannot silently
change the reviewed algorithm. This identity check is a trusted transcription
boundary, not a formal proof of PTX lowering. Updating the pin requires reviewing
the entire subject and rerunning source emission, proof and device gates.

Checked execution uses Rust `CudaContext::load_checked_exact_pv` and
`launch_checked_exact_pv`, or Python `exact_pv_launch.CheckedExactPv` (used by
`tools/exact_pv_bridge.py`). It derives grid `(Q,B,1)` and block `(D,1,1)` and
checks positive signed-I32 dimensions, all three element products <= I32MAX,
the full U32/I8 accumulator limit `T <= 16777216`, device/function limits,
buffer context, extents, alignment and output disjointness. Inputs may overlap
each other. Both APIs synchronize before and after execution. `T=0` refuses
because the present theorem requires positive T. Python also rejects lazy
negative/conjugate and noncontiguous tensors, and queries live allocation
ranges; unsupported pool/allocation queries refuse. CUDA metadata and ordinary
allocations without externally remapped physical aliases remain trusted.
The earlier raw module loader checks artifact identity only; raw launches
do not inherit these runtime contract checks.
`Y_TVAL_PYTHON` selects the interpreter used by the Cargo gate and Rust device
test. The [artifact handoff documentation](../../docs/ptxas_translation_validation.md#the-one-unbroken-chain)
states the remaining proof and execution assumptions.

`regress.sh` includes an **UNPROVED** row on purpose. `o1/naive_gemm_f32` is a
shipped GEMM and it VALIDATES, because the emitter says `fma.rn.f32`;
`o1/naive_gemm_f32_muladd` is the same kernel in the form Y used to emit -
`mul.f32` then `add.f32`, two roundings, which `ptxas` contracts into one
`FFMA` on a byte-identical instruction stream - and it is refuted. A run in
which that row turns green is a regression, because a corpus containing
nothing the validator refutes cannot be told apart from a validator that
always says VALIDATED. `regress.sh` ASSERTS its standing results in the
direction each reads and exits non-zero if any of them moves; `fpgate.py`
asserts the same pair independently, and checks the doc's contraction count
against the measurement.

The three `max/` rows are the shipped ReLU shape and its two controls.
`max/relu` needs FMAX commutativity because `ptxas` swaps the operands when it
folds the literal into `RZ`; `max/general`, whose order it preserves, validates
WITHOUT that fact — measured — so the pair is what shows the measurement is
load-bearing for exactly one shape. `max/min` is the other polarity of the same
`FMNMX`.

`neg/unfoldable` is the SECOND refutation and it is a different kind:
`neg/folded` holds the SAME PTX opcode and VALIDATES, so the pair says the
refusal is about the LOWERING rather than about the opcode. `ptxas` has no bare
float-negate instruction -- where it can it folds the negation into an operand
modifier (a bit-exact sign flip, measured), and where it cannot it emits
`FADD Rd, -Rx, -RZ`, which is arithmetic and canonicalises every NaN.

`corpus/` and `o1/` are generated and not committed: a `.cubin` is a machine-specific ELF
and a `.sass` is a disassembly of one. The September write-up recorded 66
byte-identical rebuilds with its original toolchain. The October continuation
did not repeat that complete-corpus identity measurement.

## Layout

| | |
|---|---|
| `ptxexec.py` `sassexec.py` | the two symbolic executors |
| `smem.py` | shared memory as a z3 array; barriers as an uninterpreted `H_k` |
| `fpmode.py` | float macro-op table, with a `validated` flag per identification |
| `fpsem_abi.py` `fpsem_abi.c` | referee seven float facts against the device: `FSEL`, f32-add commutativity, the `-R` sign flip, the un-foldable `neg.f32` lowering, `a-b == a+(-b)`, the `max.f32` PTX rule, and `max` commutativity |
| `mulmode.py` `conc.py` | the multiplier ladder (`uf` / `wide` / `direct`) and concretisation |
| `intenc.py` | exact bitvector-to-integer encoding; preserves source declarations and all input/output sorts |
| `validation_child.py` | fresh-process validator program shared with strict command evidence checks |
| `batch.py` | the obligations, and `same_if` — guard-relative address matching |
| `tval.py` `loopval.py` `smemval.py` | drivers: straight-line, loop, shared memory |
| `exact_pv_artifact.py` | retains the validated `exact_pv` cubin, binds its SHA-256 identity, and loads its checked bytes |
| `cfg.py` `loopcfg.py` `params.py` | control-flow and signature parsing |
| `scope2.py` `depth.py` `tractable.py` `smemdepth.py` `barregion.py` | measurement |
| `gap.py` | the dynamic gap — what the executor genuinely refuses, not its first refusal; `--rank` adds cost **and reach** |
| `loopgap.py` | why `loopval` refuses each kernel that has a loop — the structural gate `gap.py` cannot see |
| `cbank_abi.py` `cbank_abi.c` | referee the const-bank ABI against `ptxas` **and** the device |
| `fpclass.py` `expand.py` `contract.py` `unroll.py` `olevel.py` `muls.py` | measurement |
| `gmut.sh` `lmut.sh` `smut.sh` `rmut.sh` + `muts/` | mutation tables; **the control row is first** |
| `mkbase.sh` `restore.sh` | baseline snapshot/restore for the mutation harnesses |
| `fma/ div/ loop/ smut/ synth/` | small hand-built fixtures, including the negative controls and `synth/nostore` (a kernel that stores nothing) |

## The row that makes the table mean something

`fma/plain` and `fma/rn` are the same kernel; `plain` lets `ptxas` contract
`mul.f32`+`add.f32` into one `FFMA`, and the validator answers `sat` with a
counterexample. A validator that always says VALIDATED would report every other
row identically. Keep that control passing — i.e. failing.
