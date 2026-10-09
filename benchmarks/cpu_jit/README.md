# CPU JIT comparison with C#

These benchmarks compare equivalent algorithms on one machine. Controlled
optimized Y runs use measured branch profiles (PGO); C# uses .NET 8 with tiering
and dynamic PGO disabled. Runtime workloads include allocation, growth, scans
and reclamation costs, and use different container representations. The results
do not isolate code generation or establish a general language speed ranking.

The [training-tier report](../../docs/cpu_jit_benchmarks_training_tier.md)
publishes the controlled temporary IR O3→O1 comparison. Final/native O3,
per-pass verification and atomic counters stay fixed. Instrumented compilation
falls about 20%; full preparation medians fall 2.83% and cold preparation
6.60%, with every nine/five pair improving. Final IR/profiles match; eight
native medians and cold final compilation's separate medians regress. All
losses remain. Rust's training override is explicit and defaults to inheritance.
Measured training is about 62.15% of full preparation; final O3 optimization
remains the largest compilation phase. Diagnostics are scoped separately from
the audited full comparison.

The [ORC codegen report](../../docs/cpu_jit_benchmarks_codegen.md) publishes the
2026-10-08 O3→O2 machine-code comparison with IR O3 and per-pass verification
fixed. Cold preparation medians regress 1.90%; warm preparation loses five of
nine pairs despite a small decrease in separate medians. Seven workload medians
regress. The inherited O3 default remains unchanged. Object-event measurements
place about 99.83% of cold profiled materialization before object handoff, with
about 0.17 ms afterward. The report retains every pair and loss, actual emitted
object byte counts, all hashes and scope limits.

The [verification-policy report](../../docs/cpu_jit_benchmarks_verification.md)
publishes the 2026-10-08 comparison. Only per-pass LLVM verification changes;
both Y settings keep mandatory checks before and after attempted pipelines.
Cold preparation medians decrease 21.42% and full-size preparation 10.77%,
with every preparation pair improving. Per-pass verification remains the
default; this is an explicit Rust option. Cold materialization, explicit
verification and eight native workload medians regress. Raw records, signed
paired costs, independent audits and source/binary snapshots retain all losses.
The policy difference does not isolate exclusive verifier work.

The [helper-effects report](../../docs/cpu_jit_benchmarks_helper_effects.md)
publishes the final fifteen-workload run from 2026-10-07: nine process triples,
five cold triples, independent audits and frozen source evidence. Helper String
and Vec improve 128.26x and 77.78x over the same compiler with helper analysis
disabled, each in 9/9 pairs. Small-input cold preparation regresses 12.03%; small
control-workload losses are retained. The measured compilation bottlenecks are
LLVM optimization/VerifyEach and eager ORC materialization. Direct/helper ratios
are descriptive because direct batches precede helper batches within workers.

In the [copy report](../../docs/cpu_jit_benchmarks_runtime_copies.md), next Y
improves over previous Y by 5.65x for dynamic byte vectors, 8.20x for I64 vectors
and 1.60x for bulk strings. Headline ratios divide the per-engine medians;
they are not medians of paired ratios. Against C#, the byte result is 1.46x
with Y faster in 6/9 pairs and a paired range of 0.60–2.55x. I64 and bulk strings
favor Y in 9/9 pairs. Y uses a flat byte string; C# uses UTF-16 `StringBuilder`,
and Y `Vec` and C# `List` have different layouts, growth policies and allocators.

The C# I64-vector batches record `GC.CollectionCount` deltas of `[1,1,1]` for
Gen0/Gen1/Gen2 in each of nine timed batches, with a median 8,398,592 allocated bytes per
32-call batch. Y pays explicit frees inside its timers; C# reclamation deferred
until after a batch is uncharged. These runtime costs are part of the comparison.
Copy cold preparation increased 2.6%. The
[adapter report](../../docs/cpu_jit_benchmarks_adapters.md) shows 15.8% lower
preparation only for its five small-input cold samples, and retains 11–22%
slower native medians for several numeric kernels. Their cause is unproven;
identical saved assembly does not prove identical live JIT code placement.
Y cold preparation starts with source, while C# starts with prebuilt IL.

Run from the repository root, with LLVM 17+ as a shared library, cached Cargo
dependencies, Python 3 and a .NET 8 SDK:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet
```

The runner builds both workers in Release mode without fetching packages. If a
system `dotnet` installation contains only the runtime, supply the executable
from an SDK installation with `--dotnet`. No global SDK installation is needed.
Select a logical CPU with `--cpu N`; otherwise the first allowed CPU is used.
The Y JIT searches for a shared LLVM library; use `Y_LLVM_LIBRARY=/path/to/libLLVM.so`
to select one explicitly.

Every run creates a new directory under `build_artifacts/` containing a report,
individual worker stdout/stderr, raw timings and outputs, tool versions, source
and binary hashes, correctness references and the optimized Y LLVM IR. Use
`--output /path/to/new-directory` to choose a location. `--skip-build` reuses
the workers in `target/release/examples/` and `build_artifacts/cpu_jit_csharp/`.

The default expanded suite has eight kernels. Use `--suite original` to run
the original four and compile their preserved source in `kernels_original.ysu`.
All eight definitions in `kernels.ysu` remain unchanged. The historical
[four-kernel report](../../docs/cpu_jit_benchmarks.md) and its evidence remain intact.

For a controlled optimization comparison, run:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet --compare-optimizations
```

This interleaves three fresh worker processes per repeat: baseline Y with rotate
recognition disabled and no branch profiles, optimized Y with rotate recognition
and measured branch profiles, and the fixed C# configuration. Optimized Y first
compiles instrumented code, runs twelve explicit training calls per kernel,
snapshots observed true/false counts, and recompiles with those profiles. Both Y
variants then perform the same standard warmup and timed batches. Training results
are independently checked as well as final results. `--profile-warmups N` changes
training volume; `--y-mode baseline` runs only the baseline Y/C# pair. No training
is hidden in compilation, and instrumented code is excluded from timed batches.

The controlled report separates instrumentation compilation, collection,
snapshot, optimized recompilation and total preparation. Collection timing
covers native training calls and output storage; total preparation also includes
training-buffer setup, function lookup and profile/result formatting. Raw counts,
profile fingerprints, applied branch counts, training outputs and both final IR
variants are preserved. The same input distribution trains and measures this
synthetic benchmark; results include that explicit requirement.

The default pair measurement uses nine independent process pairs, alternating which
language runs first. Each worker prepares the selected methods, performs twelve
warmup calls per method, then times a batch of thirty-two calls. Each method's
output varies with runtime input. Timed batches include the host calling loop
and output storage; input/output buffer initialization, formatting and validation
occur outside those timers. Runtime-object kernels also allocate, grow and free
their containers inside the timer. All integer outputs must match an independent Python
implementation exactly. Every float output must match within relative and
absolute tolerances of `1e-12`. A hash over the entire final memory array checks
the memory writes as well as the returned sums. Any mismatch aborts the run.

With `--compare-optimizations`, nine interleaved triples use rotating execution
orders. The compiler source, benchmark algorithms and runtime inputs match
between baseline and optimized Y; only the requested optimization settings and
observed branch profiles differ.

The runtime-object extension adds two workloads from `kernels_runtime.ysu`:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet \
  --suite runtime --optimization-stage runtime --compare-optimizations
```

This compares previous Y (rotate recognition, measured profiles including loop
controls, runtime callbacks) with next Y (the same rotate recognition, measured
profiles excluding natural-loop headers/latches, proven-local runtime queries
lowered into guarded header/data reads). Both variants train explicitly. The
next variant also lowers byte-to-ASCII conversion directly. Both variants use
the final compiler and common semantic fixes; the previous settings do not
replay a historical compiler binary. The loop policy and runtime lowering change together, so their separate performance
effects are not isolated. The C# configuration remains fixed. The default
`--optimization-stage rotate-profile` retains the earlier controlled comparison.

`string_scan` grows a local ASCII string one character at a time and scans it
eight times; C# uses `StringBuilder`. `vec_scan_append` appends the same byte
values to a local vector and scans it eight times; C# uses `List<byte>`. Every
128 positions each scan deliberately reads a negative index and index length,
which return zero in both programs. The independent Python oracle builds a
bytearray and checks weighted checksums. Default lengths are 16,384, configurable
with `--string-n` and `--vec-n`. ASCII equates Y bytes and C# UTF-16 character
values here; it does not equate the two string representations generally.

Each new native call includes allocation, growth, append callbacks, scans,
bounds guards and explicit Y frees in its timer. C# uses managed reclamation;
collections during a batch are included, while deferred reclamation afterwards
is not charged. Per-batch allocated bytes and generation collection counts are
saved. The containers have different layouts, growth rules and allocators. The
original eight kernels still allocate their input/output arrays outside timers.

The next controlled stage isolates proven-local append lowering:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet \
  --suite runtime --optimization-stage runtime-append --compare-optimizations
```

Both Y variants enable the same local queries, direct ASCII conversion, rotate
recognition and measured profiles excluding loop controls. Previous Y appends
through callbacks; next Y writes directly while a proven local object has spare
capacity. Growth and free remain callbacks. The ten source definitions and all
runtime inputs are unchanged. The runner records and checks the mutation option
in each worker result. Earlier `runtime` and `rotate-profile` stages explicitly
disable mutation lowering, preserving their intended comparisons.

The `copies` suite preserves those ten definitions and adds three workloads
from `kernels_copies.ysu`: a byte vector constructed with a runtime element-size
argument of one, an I64 vector constructed with a runtime argument of eight,
and bulk appends of a freshly assembled 32-character ASCII string. Both vectors
append from initialized scalar locals, scan eight times and include their final
length in the checksum. The I64 values populate high bits; Python checks signed
64-bit wrapping arithmetic independently. The bulk string appends its chunk 512
times by default and performs eight guarded scans. C# uses `List<byte>`,
`List<long>` and `StringBuilder.Append(StringBuilder)` for the same values.
Container layouts, growth rules, byte versus UTF-16 representation and managed
versus explicit reclamation still differ. These are matched-width input cases,
not a claim that all dynamic element sizes can use the scalar copy fast path.

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet \
  --suite copies --optimization-stage runtime-copies --compare-optimizations

python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet \
  --suite copies --optimization-stage adapters --compare-optimizations
```

`runtime-copies` fixes rotate, query, scalar append and compact adapter settings
on, excludes loop-control weights, and changes only dynamic/bulk copy lowering.
`adapters` fixes all runtime optimizations on and changes only whether checked
call adapters can duplicate source bodies through inlining. The adapter stage
also accepts the ten-kernel `runtime` suite. Historical `rotate-profile`,
`runtime` and `runtime-append` stages explicitly disable both copy lowering and
compact adapters. Every worker result records all options, and the runner
checks the requested configuration before accepting a sample.

The `helpers` suite preserves all thirteen definitions and adds two ordinary
helper variants in `kernels_helpers.ysu`. `string_scan_helper` and
`vec_scan_append_helper` reproduce the direct String/Vec workloads' allocation,
append, eight guarded scans and frees. Only ASCII selection and checksum weight
move into source helpers taking and returning I64/bool scalars. They use the
same `--string-n`/`--vec-n` dimensions and independent bytearray oracle as the
direct kernels. C# uses equivalent ordinary scalar helpers with its default
inlining policy; kernel entrypoints retain `NoInlining`.

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet \
  --suite helpers --optimization-stage helper-effects --compare-optimizations
```

This stage fixes rotate recognition, local query/append/copy lowering and
compact adapters on, and excludes loop-control profile weights. Previous Y
discards local runtime proofs around source calls. Next Y preserves proofs
across statically checked scalar helpers without pointer/object effects.
Only `optimize_helper_effects` changes; both variants collect actual profiles.
All earlier stages explicitly disable this new option. The direct/helper pairs
show whether introducing ordinary helpers changes the runtime optimization,
without assuming a performance improvement. Allocators, representations,
managed reclamation and the measured-Y-PGO/fixed-C# configuration still differ.
The report includes helper/direct ratios of median times and raw paired ratios.
Direct kernel batches precede helper batches in a fixed within-worker order;
allocator, cache and frequency effects can contribute to that comparison.

Every Y record now includes a `compilations` array, with one `original` entry
for an unprofiled compile or separate `instrumented` and `profiled` entries.
Each records integer nanoseconds for parse, checks, lowering, LLVM setup, IR
parse, profile setup, verification, optimization, IR capture, symbol resolution,
materialization and other work, plus total. The intervals are disjoint and the
runner checks their sum exactly. Verification aggregates mandatory full-module
checks at input and pipeline boundaries. Optimization includes LLVM
`VerifyEach` when requested and excludes those explicit checks. Each compilation
also records `optimization_timings` with integer `pipeline_ns`,
`profile_selection_ns`, `other_ns` and `total_ns`, plus successful
`verification_checks` (two without profile selection, three when attempted).
The nested parts sum exactly to the ordinary optimization phase; do not add
them again to the compilation total. Selection includes an attempted optional
pass lookup when unavailable. Each entry also records `materialization_timings`
with `submission_ns`, `first_lookup_ns`, `remaining_function_lookups_ns`,
`profile_lookup_ns`, `other_ns` and `total_ns`. Those five disjoint parts sum
exactly to total and the primary materialization phase. Optional
`first_lookup_before_object_ns` / `first_lookup_after_object_ns` children split
first lookup when exactly one object is observed inside it; otherwise both are
null. The callback leaves the object buffer unchanged. Entries retain
`object_observer_available`, `object_count`, `object_bytes`,
`function_lookup_count` and `profile_lookup_count`.

Before-object work includes lookup dispatch, IR-layer work, native object
emission and preceding ORC overhead; after-object work includes observer
bookkeeping, linking and lookup completion. Neither is an exclusive codegen or
linker timer. Children already belong to first lookup and must not be added
again. Object bytes include object-file metadata and do not measure executable
memory alone. Other is residual time. Reports retain warm/cold phase
distributions and show separate medians,
which need not sum to the median total. Profile collection and standardized
kernel warmups remain outside these compilation intervals.

The controlled `verify-each` comparison uses the fifteen-workload `helpers`
suite and `--compare-optimizations`. Both arms enable query/mutation/copy paths,
compact adapters, scalar helper effects and rotate recognition at O3; both
exclude natural-loop control weights and collect measured profiles. Previous
Y sets `verify_each_pass=true`; next sets it false. Historical stages explicitly
retain true. This stage requires the fixed .NET 8.0.31 worker runtime.

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite helpers --optimization-stage verify-each --compare-optimizations \
  --repeats 9 --cold-repeats 5 --calls 32 --warmup 12 --profile-warmups 12 \
  --cpu 0 --output build_artifacts/new-verify-each-run
```

Freeze measured sources and build without `--skip-build`; avoid concurrent
heavy jobs. The runner checks matching profiles and retained outputs and saves
byte-identical first-pair instrumented/final IR. Its summary includes nested
timing distributions and matched-repeat signed savings and ratios for pipeline,
selection, explicit verification, materialization, compilation and preparation.
Undefined zero-denominator ratios are null. Medians need not add; signed paired
median savings differ from subtracting per-arm medians. Pipeline timings include
pass-manager work and requested verification, not exclusive per-pass costs.
Saved assembly is reconstructed using `llc`, not captured from ORC.

The controlled `codegen` stage also requires `helpers` and
`--compare-optimizations`. Both arms use IR O3, per-pass verification enabled,
all runtime/helper/adapter/rotate optimizations and the same measured training
profiles. Previous Y inherits ORC O3; next requests ORC O2 through the Rust
`codegen_opt_level` option. Historical stages retain inheritance. Records
include effective `codegen_opt_level` and nullable requested `codegen_override`.

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite helpers --optimization-stage codegen --compare-optimizations \
  --repeats 9 --cold-repeats 5 --calls 32 --warmup 12 --profile-warmups 12 \
  --cpu 0 --output build_artifacts/new-codegen-run
```

Use a fresh directory and freeze measured sources before a fresh build. After
timing, independently audit complete outputs/training, flat/nested accounting,
settings and source/binary/runtime hashes. Saved first-pair IR matches between
arms; live machine code can differ. The recorded `llc_commands` reconstruct
O3/O2 assembly with PIC/native CPU/small code model; the live ORC template
requests JITDefault. Reconstructions are not captured executed code. Actual
object bytes are counted but not archived or hashed. Timing summaries preserve
raw values and signed paired savings, including nullable children and losses.
The published O2 result supports no overall preparation improvement or default
change. Default IR optimization remains the largest phase; within materialization,
investigate work before object handoff. Smaller smoke runs are correctness checks.

The controlled `training-tier` stage requires `helpers` and
`--compare-optimizations`. Previous training IR inherits O3; next requests O1
through `JitOptions.training_opt_level`. Both final profiled IR and native
codegen remain O3, with per-pass verification enabled and all lowering/profile
policies fixed. This changes a temporary training tier; training/recompilation
remain explicit and eager. Existing API defaults retain inheritance.

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite helpers --optimization-stage training-tier --compare-optimizations \
  --repeats 9 --cold-repeats 5 --calls 32 --warmup 12 --profile-warmups 12 \
  --cpu 0 --output build_artifacts/new-training-tier-run
```

Each Y record includes effective `training_opt_level` and nullable requested
`training_override`. Each compilation adds `compilation_settings` recording its
actual IR pipeline/analysis-target tier, native tier/override, requested training
override and verification policy. The runner requires exact profiles, training,
first-call and timed results across arms. Saved final IR must match; temporary
instrumented IR may differ for this stage. Both assembly reconstructions use
native O3. Audit preparation including all measured training work: lower IR
tiers may trade compilation savings for slower training. Native execution rows
still execute final O3 and must not be presented as a changed final IR policy.

The adapter stage supplements native kernel timings with Y-only measurements
of checked `.call` versus direct function pointers. It uses `integer_branch`
with one step for a tiny call (20,000 invocations, configurable through
`--checked-tiny-calls`) and the normal integer dimension for a large call.
Native and checked calls receive the same seeds. The worker checks the entire
two output streams for exact equality; Python independently checks stream hashes
and saved first/last values. Frame packing, argument validation, allocation and
return handling are charged to checked timers; output buffers and hashing are
outside them. Native batches run before checked batches in each process, so
cache and frequency effects may contribute to this descriptive API comparison.
There is no corresponding C# reflection benchmark or claim of equivalent API
overhead. Native C# comparisons remain separate. Cold source compilation uses
the same selected compilation unit in both Y configurations, exposing the
preparation cost of duplicated versus compact adapters.

The eight kernels are:

| Kernel | Work per call | Features exercised |
| --- | --- | --- |
| `integer_branch` | 250,000 PRNG steps and data-dependent branches | I64 arithmetic, remainder, bitwise AND, loops, branches |
| `recursive_fib` | Fibonacci at 25 or 26 | Recursive calls, returns, branches |
| `float_recurrence` | 1,000,000 recurrence steps | F64 multiplication/addition, comparisons, loops |
| `indexed_memory` | 262,144 strided read/modify/write operations on 65,536 I64 elements | Pointer indexing, loads, stores, loops, bitwise AND |
| `unsigned_mix` | 250,000 xorshift/multiply/rotate rounds | U64 high-bit values, wrapping arithmetic, logical shifts, rotate patterns |
| `float_dot` | 262,157 products/reductions over two 65,536-element F64 arrays | F64 pointer loads, multiplication, ordered reduction |
| `short_circuit` | 250,000 conditional rounds with three side-effect counters | Lazy `&&`/`||`, helper calls, conditional stores |
| `binary_search` | 20,000 queries over 65,536 sorted I64 values | Nested loops, comparisons, indexed loads, `break`, present/missing keys |

The short-circuit counters verify that skipped RHS calls have no side effects.
Unsigned arithmetic is checked against explicit modulo-2^64 Python operations.
Dot inputs use exact binary fractions. Binary search is checked against Python's
independent `bisect` implementation.

The Y worker uses the public `CpuJit` API with LLVM O3 and the host CPU target.
C# uses Release builds on .NET 8, with `DOTNET_TieredCompilation=0` and
`DOTNET_ReadyToRun=0` (and their `COMPlus_` equivalents) to produce optimized JIT
code immediately, without tiering or dynamic PGO. Controlled profiled Y variants
collect measured profiles explicitly; C# receives no corresponding profile training.
C# kernel entries use `NoInlining` so the host call boundaries match
the native function-pointer calls into Y. Memory kernels use unchecked
pointer indexing over a valid, pinned host array.

Cold measurements use five independent process pairs, or five triples in a
controlled comparison, with small inputs. Each Y compilation starts at source
parsing and ends after all public functions and adapters have native entrypoints.
Profiled Y's full preparation timer also includes instrumented compilation,
training, snapshotting, recompilation, profile/result formatting and trainer
disposal. C# starts with already-built IL, then calls
`RuntimeHelpers.PrepareMethod` on the selected kernel methods and helpers. Source-to-IL build
time is outside that timer. The two preparation times therefore start at
different stages and should not be read as equivalent compiler speed tests.
Process launch to JSON result includes runtime startup and output formatting.

The generated report shows per-kernel medians and all raw samples. `C# / Y`
divides the C# median time by the Y median time; above one favors Y by that
summary statistic. Paired ratios divide times from the same repeat and can
favor different engines across repeats. The paired ratio range is a
descriptive min/max, not a confidence interval. CPU frequency, host activity
and compiler-specific transformations can affect the result. Synthetic
workloads on one machine do not establish a general language performance ranking.

A quick end-to-end correctness smoke run is:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet \
  --repeats 1 --cold-repeats 1 --calls 2 --warmup 1 \
  --integer-n 100 --fib-n 10 --float-n 100 --memory-n 100 \
  --unsigned-n 100 --dot-n 100 --logical-n 100 --search-n 100
```
