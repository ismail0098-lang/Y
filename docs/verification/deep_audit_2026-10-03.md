# Source audit and correctness fixes, 2026-10-03

This audit inspected the current local source, Rocq developments, symbolic
executors, tests, kernel generators, retained measurements, and benchmark
drivers. It did not use Git history. The README was checked only after the
implementation. The workspace already contained substantial development;
the conclusions apply to that local tree, not to an upstream release.

The main conclusion is that Y has useful conditional kernel proofs and a
restricted translation validator. It does not have a generally verified
compiler or an end-to-end hardware correctness theorem. The most valuable
overlap is `exact_pv`: mathematical value reasoning, emitted PTX, fresh offline
assembly, validation, and an explicit checked-cubin loader. That overlap still
depends on trusted transcription, validator implementation and ISA assumptions.
The follow-up checked launch APIs enforce explicit runtime contracts. Successful assembly, a schedule proof, and a
value proof are different evidence.

The audit reproduced several wrong translations accepted by the pre-fix
validator. They were deliberately modified PTX/disassembly controls, not
evidence that NVIDIA's assembler produced those errors. The integrated changes
reject those cases and retain genuine positive controls. They strengthen the
evidence; they do not make the Python validator formally verified.

## 1. What Y can claim today

| Subject | Evidence that actually reaches it | Boundary |
| --- | --- | --- |
| CPU exact I16/I64 GEMM | Composed packing, VNNI lane routing, flush arithmetic, tiled reduction, and output ownership model; compilation-specific Rocq certificates; generated schedule/IR agreement checks; native differential execution | Rust-to-LLVM transcription and structural checks, finite address arithmetic, libc/thread ordering, LLVM optimization and machine code, ISA/hardware |
| GPU `exact_pv` | Conditional source-dot-product theorem over wrapped/masked arithmetic; launch-domain composition; fixed complete reviewed PTX subject; fresh `ptxas -O1` translation validation; hash-bound cubin and checked Rust/Python launch | Trusted model-to-PTX review, checker implementation and metadata, Python/Z3 and multiplier identity, disassembly/ISA/driver/hardware |
| GPU exact int8 GEMM | Fragment layout, grid-stride/reindexing and integer arithmetic proofs; emitter bound refusal and source/PTX regression gates | Actual MMA semantics and complete composition of wrapping chunked/split schedule; no SASS validation |
| Exact attention | Reduction/schedule proofs; shared `Ix` generation of PTX and Coq expressions; compilation-specific accumulator bound; conditional softmax approximation theorem | Exponential accuracy hypotheses, full emitted arithmetic, atomics/shared-memory execution, `ptxas`; no SASS validation |
| F16/FP8/SwiGLU tensor kernels | Schedule/output partition facts and dynamic correctness checks | Numerical algorithm and tensor instruction values are not proved; no SASS validation |
| `y_cpu_matmul` emitted as PTX | Nested-loop PTX/SASS relation at `-O1`, including iteration memory readback | No corresponding algorithm theorem; not the CPU AVX-512 correctness chain |
| Other validator controls | Fresh integer/float/memory translations and explicit refutations/refusals | Per-subject equivalence under the supported model; no application algorithm proof |

The CPU composition is technically significant because
[`ExactGemmWhole.the_threaded_gemm_holds_the_source_dot_products`](../../proofs/ExactGemmWhole.v)
connects packing and a real register-tile/reduction organization to every
source dot product. The row-band ownership theorems in that file address
overlapping or omitted output work. These establish more than commutativity
of a sum. They still quantify over a mathematical model.

[`ExactPvExact.the_emitted_exact_pv_holds_the_source_dot_product`](../../proofs/ExactPvExact.v)
proves equality for a mathematical prefix `n <= Tx` between a wrapping
accumulator and the source sum when
indices stay within signed 32-bit range, intended input accesses are in
range, loads obey their operand domains, and
`n * 4294967295 * 128 <= I64MAX`. It is not an operational proof that a CUDA
launch terminates and writes every output. The theorem named
`every_output_element_is_written_by_one_thread` proves injectivity of an
index map under coordinate bounds; its statement does not prove that an
actual launch covers all output elements.

In [`Int8GemmExact.v`](../../proofs/Int8GemmExact.v), compare the actual
32-element `step`, generic `wclass`/`wcombine` wrapping split result, scalar
`wcombined_is_the_flat_sum`, and final capstone. The final theorem combines a
mathematical split schedule with a wrapping flat sum; a complete derivation
of the actual chunked, wrapping split execution deserves a direct capstone.
MMA operand routing is valuable, but defining MMA's mathematical result does
not prove the hardware instruction implements it. The current emitter's
`K <= 131040` bound handles the magnitude of signed int8 `-128`; the README's
older `133120` threshold was inaccurate.
The split combine also requires a zero-initialized destination. The emitter
leaves initialization to callers;
`the_combine_needs_a_zeroed_destination` in `Int8GemmExact.v` demonstrates why
this launch prerequisite matters.

[`SoftmaxErrorBound.v`](../../proofs/SoftmaxErrorBound.v) explicitly assumes
properties of an ideal weight function and bounds for the implementation's
integer exponential. `the_attention_output_is_within_the_bound` is conditional
on those hypotheses. The generated attention certificate in
[`exact_attention_certificate.rs`](../../src/exact_attention_certificate.rs)
imports grid-stride and attention schedule results and discharges a concrete
accumulator bound; it does not instantiate the complete softmax output-error
capstone for every emitted instruction. Counting-sort/MSM scatter proofs and
[`ZkControlFlow.v`](../../proofs/ZkControlFlow.v) similarly cover stated models,
not every GPU elliptic-curve operation or adversarial R1CS soundness.

## 2. Exact verification and trust boundaries

The checked Rocq developments compile. The proof gates request and inspect
`Print Assumptions` results; no active `Admitted`, `admit`, added `Axiom`,
unfinished proof, or disabled guard/positivity shortcut was found in the
inspected proof sources. This is supported by
[`proofs_are_checked.rs`](../../tests/proofs_are_checked.rs),
[`common/coq.rs`](../../tests/common/coq.rs), and actual compiler runs.
[`proof_mutation_checks.rs`](../../tests/proof_mutation_checks.rs) checks that
admitted and axiomatized partitions are refused even when Rocq accepts their
syntax. Closed theorems still have premises and chosen semantic definitions;
the absence of axioms does not validate those definitions against machines.

The GPU chain is:

```
algorithm in mathematics
  -> Rocq theorem about a chosen model                       checked by Rocq
  -> Y source / Rust emitter / complete reviewed PTX         trusted transcription + gates
  -> ptxas cubin / nvdisasm SASS                             inspected per compilation
  -> symbolic PTX/SASS relation                             trusted Python + Z3 + ISA assumptions
  -> retained cubin handed directly to CUDA                 hashes and owned bytes checked
  -> checked launch contracts                               trusted checkers + CUDA allocation metadata
  -> faithful hardware execution                            driver/ISA/hardware assumptions
  -> independent application reference                      dynamic evidence
```

The new [`exact_pv_subject.py`](../../tools/ptxas_tval/exact_pv_subject.py) and
fixed digest bind every instruction, operand, and declaration of the reviewed
subject, ignoring comments and whitespace. Both
[`exact_pv_artifact.py`](../../tools/ptxas_tval/exact_pv_artifact.py) and
[`verified_exact_pv.rs`](../../src/verified_exact_pv.rs) check it. A digest is
an identity check, not a Rocq proof of the PTX decoder or emitter. Updating it
requires reviewing the entire semantic change. The freshly emitted source is
checked against that fixed subject in
[`exact_pv_artifact_binding.rs`](../../tests/exact_pv_artifact_binding.rs).

The loaders preserve a valuable separate boundary: they hash PTX/SASS/cubin,
pin the receipt, check ELF entry/architecture and device capability, and pass
the already checked owned cubin bytes to CUDA. They do not re-open a pathname
after checking it and do not fall back to PTX JIT. The receipt's producer and
initial storage remain trusted; the receipt is not an authenticated solver
certificate. The recording-driver tests check this handoff without needing a
GPU; they do not prove hardware execution.

[`CudaContext::launch`](../../src/cuda_runtime.rs) still accepts raw parameter
pointers and arbitrary launch geometry. It does not enforce the theorem's
index/accumulator bounds, input/output lengths, disjointness, correct grid
coverage, or `block.y = block.z = grid.z = 1`. Additional coordinates can
duplicate outputs because this kernel does not distinguish all of them.
Validation of a byte-identical cubin does not establish validity of its launch.

The follow-up introduces [`ExactPvShape`](../../src/verified_exact_pv.rs),
[`CudaContext::load_checked_exact_pv/launch_checked_exact_pv`](../../src/cuda_runtime.rs)
and [`CheckedExactPv.load/launch`](../../tools/ptxas_tval/exact_pv_launch.py).
They derive precisely `(Q,B,1)/(D,1,1)`, reject nonpositive dimensions including
T=0, bound NP/NV/NO by I32MAX, and enforce the full-domain accumulator limit
T<=16777216. The launch checks cover device/function limits, context ownership,
buffer byte extents, alignment, nonwrapping ranges and output/input separation;
read-only P/V aliasing is allowed. Both synchronize before and after the launch.
Python requires ordinary dense CUDA tensor storage and rejects lazy negation/
conjugation; actual CUDA allocation/context queries must succeed. Raw APIs
remain unverified at launch and cannot inherit this claim.

[`ExactPvLaunchContract.checked_shape_instantiates_the_source_dot_product`](../../proofs/ExactPvLaunchContract.v)
derives the existing theorem's index, range and accumulator premises from the
positive shape conditions at n=T. Its coverage/uniqueness results now establish
the complete mathematical output rectangle, while the operational loop/store
connection remains trusted. The proof does not verify the Rust/Python checker,
driver metadata, external physical aliases or concurrent external mutation.

For CPU GEMM, [`cpu_gemm.rs`](../../src/cpu_gemm.rs) shares `Ix` and counted-loop
descriptions with generated Coq expressions and emitted LLVM. Schedule and
mutation gates check that these descriptions remain connected. They do not
define or prove operational LLVM semantics: loop-label/shape checks cannot by
themselves establish every branch, load, and store. In
[`exact_gemm_certificate.rs::TRUST_BOUNDARY`](../../src/exact_gemm_certificate.rs),
clang optimization, assembly, and linking are explicitly unchecked. That is
where the remaining compiler trust certainly begins; Rust emission and the
unproved parts of its model correspondence are already trusted above it.

Operand bounds are also an input contract. For a load with no inferred interval,
`type_checker.rs` accepts a declared `@bounds` interval without proving or
checking the loaded data against it. The generated GEMM certificate quantifies
over inputs respecting that declaration; certifying the flush inequality does
not establish that actual input buffers respect it. A verified application must
prove that property upstream or check it. The range/alias dispatch added here
does not check data values.

The CPU mathematical accumulators in
[`ExactGemmMicro.v`](../../proofs/ExactGemmMicro.v) are `Z`; the emitted vector
accumulator uses finite i64 operations. Per-flush safety is proved, but a
general machine-width theorem for total accumulation, allocation sizes,
pointer offsets, and all accepted extents is not the capstone. ThreadSanitizer
and native runs support libc/thread ordering; they do not prove every schedule.
Compiler-side certificate generation is not automatic per-build `coqc`
checking. `Y_NO_CERTIFICATE` and certificate write failures can leave compiled
output without a checked certificate. These are reported behaviors, but a
mandatory verified build profile should refuse them.
In [`main.rs`](../../src/main.rs), `write_exact_gemm_certificates` returns no
failure status after a write error; `--emit-llvm` calls it after publishing IR.
The default LLVM executable path does not call it at all, so a recognized
exact GEMM binary can omit the certificate even without suppression. Attention
PTX is printed before its optional certificate write. This follow-up does not
add the needed staged certificate-check build mode.

## 3. Validator coverage, reproduced holes, and remaining holes

The principal executors are [`ptxexec.py`](../../tools/ptxas_tval/ptxexec.py)
and [`sassexec.py`](../../tools/ptxas_tval/sassexec.py). Straight-line drivers
[`tval.py`](../../tools/ptxas_tval/tval.py) and
[`batch.py`](../../tools/ptxas_tval/batch.py) compare execution guards,
addresses, access extents, and stored values. Unknown instructions and operand
forms normally refuse. Inconclusive SMT results are not acceptance.

[`memorder.py::read_through`](../../tools/ptxas_tval/memorder.py) implements
little-endian readback through preceding stores, with widths and aliasing.
`reorder_obligations` checks whether paired stores can be reordered when their
byte footprints overlap. This matters because abstracting all loads as initial
memory would validate hoisting across a possibly aliased write. The model is
one thread's memory view. It is neither a GPU race-freedom proof nor a general
memory-safety theorem. Initial-memory abstractions are conservative in some
overlapping access shapes and can leave legitimate transformations unproved or
refused: independent initial 32-bit reads are not a single global byte array.

[`loopval.py`](../../tools/ptxas_tval/loopval.py) proposes a relation from
samples, then proves base, step, continuation, entry and exit obligations;
samples choose candidates, they are not the proof. Supported loops have a
restricted top-test PTX/bottom-test SASS shape. New header guards refuse
uncomposed value effects and, in the single-loop validator, predicate uses in
body/epilogue. The nested validator composes supported predicate-only headers
into the body. Neither generates a termination/ranking proof even for accepted
loops; exit comparisons are conditional on termination.

[`nestval.py`](../../tools/ptxas_tval/nestval.py) composes restricted child-loop
relations using opaque summaries, with explicit region-order restrictions.
Iteration memory and load traces now participate in the relation. Sequential,
mixed, irreducible, branching-body, and unsupported child/store arrangements
remain refusals. Shared memory and synchronization inside loop validation are
now conservatively refused throughout the subject.

[`smem.py`](../../tools/ptxas_tval/smem.py) and
[`smemval.py`](../../tools/ptxas_tval/smemval.py) support a narrower straight-line
shared-memory model: arrays, supported aligned word/vector accesses, complete
unpredicated block barriers, equal entering/exiting shared states and a shared
uninterpreted barrier transformation. This can establish equivalence assuming
that abstraction. It does not establish CUDA's concurrent memory model, race
freedom, correct participating threads, or general barrier safety. Counted,
predicated and unmodeled barriers refuse.

Integer operations use fixed-width bitvectors and, in selected cases, a
checked integer encoding. Carry initialization, widths, signed comparisons,
shifts and operand forms have dedicated regressions. Float primitives use
uninterpreted functions to avoid identifying contraction with separate
roundings. That is sound only when each cross-side identification matches the
real ISA. [`fpmode.py`](../../tools/ptxas_tval/fpmode.py) licenses selected float
commutativity/negation facts using device measurements; these are empirical
assumptions, not all-bit hardware proofs. Float conversions, approximate
macro-operations and many flags remain unsupported. The division lowering
uses its own estimated-reciprocal assumption plus checked tail arithmetic.

[`mac64.py::instances`](../../tools/ptxas_tval/mac64.py) is a particularly clear
external assumption: a 32-bit multiplier identity used in `exact_pv` is
assumed. Its reduced 8-bit form is solver-proved, and recorded device samples
support the full-width case. Neither makes the 32-bit identity a formal proof.
No Rocq axiom is needed for an unsound assumption in a separate Python checker.

Unsupported areas include atomics/reductions, tensor MMA/HMMA, async copies,
warp communication, calls, general reconvergence with cross-thread effects,
many conversion/float64 forms, and arbitrary optimized control flow. For
alignment, even elementary signed integer `max` produced `IMNMX` and was
refused by a fresh DNA-like score test. This is a concrete missing dependency,
not merely an absent high-level alignment theorem.

The following pre-fix failures were reproduced and have targeted fixes:

| Case | Why acceptance was wrong | Integrated guard/evidence |
| --- | --- | --- |
| Add a register update before a PTX loop's exit test, keep SASS unchanged | Header execution was discarded; zero and positive trips change values | `loopcfg.require_ptx_guard_only`; loop and nested header tests |
| Header `and.pred p1,p1,p0` changes a carried predicate on the terminating visit; SASS retains the prior value for an epilogue store | Nested exit summary equated last-body predicate values despite the additional final header execution | Refuse header-written predicates read before definition in that header; retain ordered predicate-only calculations |
| Remove `exact_pv`'s zero-trip SASS branch | Body can run when PTX has no iterations | Entry obligation even when no SASS guard exists; constant-positive-trip control still accepted |
| Reachable self-branch before output | Every self-branch was treated as a harmless trailing trap | Conservative CFG reachability; only unreachable traps omitted |
| Unused nested global load, changed width/address/guard | Access effects disappeared if their values were dead | Full load traces in prologue, nested iterations, epilogue |
| Add loop shared store/barrier/reconvergence | Region composition ignored cross-thread state | Whole-subject conservative refusal |
| Malformed PC-tagged instruction | Parser skipped it before unsupported-opcode checks | Common strict instruction parser and malformed-prefix controls |
| Put a PTX store after `.reg` or `.loc` on the same line, then drop its SASS store | Specification scanner skipped the entire directive line, leaving one fewer source effect | `ptxsource.require_directive_lines` uses a full-line grammar; mixed or unknown payloads refuse before execution/loop discovery |
| Shadow an outer PTX register in a nested lexical scope, change the later outer-register SASS result to the inner value | PTX register state was flat across scopes, so the checker modeled the wrong source value | Lexical brace depth >1 refuses; ordinary label-based nested loops remain supported |
| Mask stored `ctaid.x` to 24 bits | Hidden solver assumption `ctaid.x < 2^24` excluded a legal counterexample | Full legal modeled x-domain in `domain.launch_preconditions`; large-grid regression |
| Change `P[prow+t]` to `P[prow]`, assemble and validate the resulting wrong algorithm | Translation equivalence proves the wrong PTX faithfully; opcode-presence proof gates survived | Fixed full-subject pin; builder and both loaders refuse even with recomputed receipt hashes |

The large-grid bound follows the documented CUDA maximum x grid dimension
`2^31-1`, so a launched block index can reach `2^31-2`.
[NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf).

The regression files are
[`ptxas_loop_control_regressions.py`](../../tests/ptxas_loop_control_regressions.py),
[`ptxas_nested_effect_regressions.py`](../../tests/ptxas_nested_effect_regressions.py),
[`ptxas_domain_regressions.py`](../../tests/ptxas_domain_regressions.py), and
[`verified_exact_pv_artifact.py`](../../tests/verified_exact_pv_artifact.py).
Other meaningful adversarial coverage includes aliased hoisting, overlapping
store reorderings, early returns, dead vector lanes, carry initialization,
multiplier congruence and SAT/UNSAT controls, and float rounding distinctions
in the existing memory/integer/pipeline regression files. Positive controls
are essential: changing every expected result to refusal would hide loss of
coverage. Hardcoded obligation totals were replaced where they prevented
legitimate additional obligations from passing, without relaxing verdicts.

Matching targets initially passed only a syntactic check. The follow-up
[`domain.require_licensed_target`](../../tools/ptxas_tval/domain.py) now licenses
only unqualified sm_89 on every public verdict-producing path, including
batch refinement and shared validation. Matching sm_999/sm_90a/sm_80/sm_86
refuse; assembly portability remains separate. Fresh sm89 O3 positive and
numeric mutation controls preserve meaningful coverage in the licensed domain.
This closes cross-target assumption reuse; it does not prove the assumed sm89
ISA/ABI or empirical float identifications.
Remaining plausible failures deserve focused adversarial work: missing
machine-width/address and launch hypotheses; unsupported effects hidden in
metadata/directives or region boundaries; nontermination compared only through
inductive state relations; and corruption of the trusted receipt producer.
The fixed digest closes one artifact identity gap; the checked launch path
enforces its declared runtime premises while manual model correspondence
remains exposed. Ordinary JIT loaders and
unvalidated kernel emission remain available; they must not inherit the
verified-loader claim.

## 4. What the README had not caught up with

`-O2`/`-O3` are supported for appropriate straight-line output. Fresh pipeline
tests assemble real optimized integer/memory/float controls and validate them;
optimization level by itself is not a refusal criterion. Unrolled loop
translations are still not generally supported. Fresh `exact_pv` and naive
GEMM `-O2`/`-O3` output was refused on optimized branch forms; merely adding
those branch spellings would not prove the unrolled iteration correspondence.
[`unroll.py`](../../tools/ptxas_tval/unroll.py) measures factors and shapes; it
is not a proof procedure for an unrolled loop. A fully unrolled SASS body also
does not make looping PTX executable by the straight-line validator.

The current tree already had store-ordered global readback, alias-sensitive
store reorder checks, subword/vector widths, alignment checks, nested-loop
readback, strict tool discovery, artifact retention, and a cubin handoff.
These supersede older blanket statements about initial-memory-only loads and
the absence of supported access alignment/extent checks. Those checks cover specific supported
accesses; they do not justify a blanket memory-safety claim. After this
integration, signed runtime loop bounds replace the README's nonnegative-bound
workaround, and proof-subject identity is enforced at artifact build/load.

## 5. Strongest parts and performance evidence

The strongest parts are proof composition around concrete schedules, explicit
limits on claims, and adversarial controls that distinguish correct output,
refutation and refusal. Shared `Ix` generation is technically significant
because it reduces independent copies of index arithmetic; mutation checks
can then expose a wrong stride instead of allowing proof and code to drift
independently. Complete artifact identity and direct cubin loading eliminate
the separate driver-JIT translation from the claimed validated path.

Proof checking and SMT validation happen at build/verification time; they do
not run per arithmetic operation or per kernel launch. Hash and ELF checks
occur when opening/loading bundles. Packing, flushing, scratch work and a
restricted optimization level are runtime costs of the implementation.
Verification elapsed time must not be presented as runtime overhead, and
speed of an unvalidated tensor kernel does not establish speed of a validated
one.

The new CPU range/alias dispatch adds a per-call runtime cost. Archived
benchmark measurements do not include that cost and have not been refreshed.

There is insufficient evidence to claim that the proved-and-SASS-validated
kernel class is broadly competitive today. [`exact_pv_bridge.py`](../../tools/exact_pv_bridge.py)
now times a retained validated `-O1` artifact, but older reported ratios lack
the same retained, hash-bound paired measurement. Its arm ordering/minimum
selection, synchronization and wrapper work constrain conclusions. The full
model demonstration's [`batch_invariance_demo.py`](../../tools/batch_invariance_demo.py)
uses a PTX-JIT path, so its application throughput is not evidence for the
validated-cubin chain.

[`tools/exact_gemm_bench/run.py`](../../tools/exact_gemm_bench/run.py) requires
an existing release compiler without verifying its freshness, takes minima
of a small number of runs, times arms in fixed order, and does not turn a full
reference result into a benchmark correctness gate. Comparing exact integer
accumulation to floating-point BLAS also compares different numerical
contracts. Those numbers can motivate work; they are weak evidence for small
performance differences.

[`benchmark_gemm_vendor.py`](../../tools/benchmark_gemm_vendor.py) is stronger:
alternating paired rounds, multi-second warmup, raw records, source/artifact
hashes and independent reference checks. The retained
[`2026-09-25 summary`](../benchmarks/gemm_scheduling_2026-09-25/summary.json)
shows vendor elapsed time about 5.28% and 6.45% lower at 2048 and 4096 cubed
across the recorded processes. Those are tensor-kernel results outside the
proved-and-validated class. Unlocked clocks, one GPU, external load and reused
cache state still limit their scope. No GPU correctness/performance was rerun
in this audit: the local driver could not communicate with the NVIDIA device.
CPU AVX-512 native correctness execution was available.

## 6. Five highest-leverage next steps

1. **Unify and justify the validator core.** Use one strict decoder/control-flow
   subject, explicit region effect/state interfaces, non-vacuous launch domains,
   and adversarial controls at every composition boundary. Mechanize a small
   soundness result or emit independently checkable SMT proofs. Complete the
   full-width multiply identity proof. This directly protects existing claims.
2. **Replace trusted algorithm transcription with a compositional connection.**
   Generate a typed small kernel IR, its operational semantics and its PTX
   subject from one representation. Prove lowering for the integer subset and
   tie certificates to the entire artifact. The digest is useful containment,
   not that proof. Complete the actual int8 chunked/split capstone.
3. **Finish execution-domain and build enforcement.** The checked exact-PV
   launch now covers its positive shape, live ranges and geometry. Extend that
   approach to other kernels and CPU finite-width/data bounds. A staged
   verified build must require successful certificate checking and TVAL.
   Unsupported or unproved cases must not return an executable labeled verified.
   Extend CPU finite-width and allocation/alias contracts as part of this work.
4. **Prove one narrow DNA alignment recurrence.** Specify exact gap conventions,
   initialization, score width/overflow, empty inputs, ambiguity symbols and
   ties. Prove the dynamic-programming dependencies and implementation loop
   invariants, then add only the integer instructions required by that kernel
   (including signed maximum). This adds a new algorithm rather than another
   reduction theorem.
5. **Support the chosen optimized shape and measure that exact artifact.**
   Prove a bounded unroll correspondence including remainder/zero-trip paths,
   then run independent full-output reference comparisons and artifact-bound
   paired benchmarks. Keep validation wall time, assembly time and execution
   time separate. Broad opcode coverage is less valuable than finishing this
   one chain.

## 7. A defensible biological workload in four months

A meaningful narrow end-to-end workload looks achievable: exact score-only,
integer affine-gap pairwise alignment for batched DNA pairs, with explicit
maximum lengths and score bounds, one target architecture and a validated
optimization shape. This is a feasibility assessment, not a schedule guarantee.
Production throughput competitive with mature alignment libraries, full
traceback/CIGAR, heuristic search, multi-GPU execution and universal ISA
coverage would be a much larger claim.

Start with independent pairs, private global-memory workspaces, and explicit
kernel phase boundaries if necessary. This avoids making concurrent shared
memory and atomics prerequisites for the first algorithm chain. It may cost
performance; measure that cost after correctness. A banded kernel proves the
banded problem unless there is a separate argument that the unrestricted
optimum lies in the band. Traceback adds path reconstruction and tie-breaking
obligations beyond score correctness.

An independent reference should implement the same mathematical contract
without sharing the generated kernel's arithmetic/indexing code. Parasail is
a practical comparison candidate because its APIs distinguish local/global/
semiglobal and score/trace results and expose integer widths/saturation.
Its convention charges opening alone when a gap first opens; resolve this
choice explicitly in Y's specification.
[Parasail source and API documentation](https://github.com/jeffdaily/parasail).
Also keep a simple wide-integer reference for small exhaustive cases so that
the GPU and vendor library cannot agree through the same overflow or convention.

A realistic order is: weeks 1–3 finish soundness gates and launch contracts;
finish month 1 with the alignment specification and finite bounds; month 2
prove the recurrence and simple kernel; month 3 complete source/IR/PTX/TVAL
composition and artifact loading; month 4 run independent exhaustive,
adversarial, randomized and paired performance checks on the actual target.
The final claim should name the algorithm variant, input domain, artifact,
architecture, optimization level and remaining ISA/hardware assumptions.

## Integration and evidence

The integrated fixes additionally restore canonical runtime M/N/K source
compilation by aligning generic loop comparisons and explicit header width
conversions across SMT, LLVM and PTX. Signed bound proofs retain positive
stable steps, immutable headers, representable arithmetic and non-overflowing
latches. Empty signed GEMM dimensions are handled before packing/allocation.
Automatic CPU substitution now checks live buffer ranges, strides and range
arithmetic overflow before selecting a packed helper; overlapping or unsafe
ranges execute the original scalar AST. This closes a concrete source-to-code
gap: zeroing/packing C before reading an aliased A or B changes scalar source
behavior. Raw low-level helpers still require their documented storage
contracts; a runtime dispatch is not a formal ownership proof.
[`smt_runtime_loop_bounds.rs`](../../tests/smt_runtime_loop_bounds.rs) includes
native C-reference comparisons, overlapping backing-buffer controls and PTX
assembly checks. [`exact_gemm_signed_dimensions.rs`](../../tests/exact_gemm_signed_dimensions.rs)
checks both raw and assigning helper entries, padded C, inactive null pointers,
and that empty ranges do not allocate or spawn threads. These are backend
correctness repairs; they are not proofs of every scalar program operation.

Verification is rerun sequentially with `--features zk` to avoid races in the
shared `target/debug/Y` executable. [`tools/verify.py`](../../tools/verify.py)
now includes the new domain, loop-control and nested-effect suites in its
strict plan, retains fresh PTXAS evidence, and hashes the verification inputs.
The audit's original evidence is in `/tmp/y-deep-audit-20261003`; deliberately
broken translations are in `/tmp/y-audit-fresh-translation` and the large-grid
and proof-subject reproductions in `/tmp/y-audit-grid-bound` and
`/tmp/y-audit-proof-seam-yll31i1j`. These pre-fix records must not be treated as
current accepted results. The final strict integration snapshot, including the
late header-predicate guard, is retained in `/tmp/y-integrated-final2-20261003`
with stage logs, source hashes, Python reports and fresh PTXAS artifacts. Native
CPU and Rust-loader checks are additionally recorded in
`/tmp/y-integrated-cpu.log`, `/tmp/y-integrated-runtime-final.log` and
`/tmp/y-integrated-artifact-rust.log`.

The checked-launch follow-up adds runtime shape/buffer/driver-query and
synchronization costs; it does not turn formal verification into per-element
runtime instrumentation. `exact_pv_bridge.py` labels the measured Y arm as a
checked call including allocation and synchronization, with offline validation
and loading outside timing. Its float64 reference comparison requires
`T*2^P_BITS*128 <= 2^53`; the separate Python-integer hardware oracle covers
full U32/I8 values and larger exact outputs. No GPU timing or execution was
available during this follow-up.

New gates are [`exact_pv_launch_contract.rs`](../../tests/exact_pv_launch_contract.rs),
[`exact_pv_launch_contract.py`](../../tests/exact_pv_launch_contract.py), CUDA
recording-driver unit checks in [`cuda_runtime.rs`](../../src/cuda_runtime.rs),
and [`ptxas_architecture_regressions.py`](../../tests/ptxas_architecture_regressions.py).
[`ptxas_directive_regressions.py`](../../tests/ptxas_directive_regressions.py)
covers directive/structural tails and preserves real straight-line, loop and
nested positive controls. Whole-file readers refuse executable text sharing
ignored directive, brace or label lines, unknown directives and text outside
the recognized body; pure repeated declarations are modeled individually.
Nested lexical scopes refuse because the register model does not implement
shadowing. A genuine outer/inner register-shadow example and a SASS value
mutation reproduced the additional pre-fix false accept; this restriction
does not exclude label-based nested loops. The reproductions are retained in
`/tmp/y-ptx-directive-probe` and `/tmp/y-ptx-nested-scope-probe`.
Memory driver bindings also require confirmed `_v2` byte/pointer ABIs; a
legacy allocator cannot truncate a >4GiB request while recording the larger
buffer extent as checked. Driver and allocator implementations remain trusted.
The proof sweep checks the new launch-contract composition and closed
assumption reports. [`exact_pv_checked_gpu.py`](../../tests/exact_pv_checked_gpu.py)
adds an independent Python-integer oracle for extreme bits, odd D, offset
views, read-only aliases and invalid tensor views; hardware prerequisites are
explicit skips and strict mode rejects those skips. It is outside the mandatory
CPU-only plan. The follow-up strict workflow command writes its input-bound
evidence to `/tmp/y-checked-launch-final-20261003`.

A separate fresh O1 shared-memory probe found a fail-closed sampling limitation:
`tval` raises on a Z3 Store expression in `conc._ev` for `smem_roundtrip`, while
`batch` and `smemval` validate that same subject. This is not an accepted wrong
translation and does not expand shared-memory concurrency claims; the repro is
retained in `/tmp/y-shared-o1-sampler-probe/results.json`.
