# General-purpose CPU JIT

Y can now compile ordinary host functions to machine code in the current
process. This is an LLVM ORC LLJIT backend using Y's existing parser, type
checker, hardware requirement checks, and LLVM emitter. It optimizes for the
running CPU and materializes all public functions before compilation returns.
Function calls execute native code directly.

The initial platform is **Linux x86-64**. Install a shared LLVM **17 or newer**;
LLVM 23.1.1 is tested here. There are no new Cargo dependencies or build-time
LLVM linkage requirements. `Y_LLVM_LIBRARY=/path/to/libLLVM.so` overrides library
discovery. Other platforms return an explicit unsupported-platform error.
Discovery tries the available libraries until both their LLVM version and
required APIs are compatible. An explicit `Y_LLVM_LIBRARY` path is strict:
an incompatible override reports an error instead of selecting another library.

## Run a Y program

```sh
cargo build --release --bin Y
target/release/Y examples/cpu_jit.ysu --jit
target/release/Y examples/cpu_jit.ysu --target=jit -O2
```

`--jit` accepts `fn main()` or `fn main() -> I32`. An I32 result becomes the
process exit status. `-O0` through `-O3` select LLVM IR and machine-code
optimization; the JIT default is O3. Compilation time is printed to stderr.
The CLI resolves imports using the existing `-I` search paths. It writes no
executable, intermediate IR file, or GPU probe cache.

## Embed from Rust

```rust
use y::cpu_jit::CpuJit;

let jit = CpuJit::compile(
    "fn square(x: I64) -> I64 { return x * x; }"
)?;
let address = jit.function_address("square")?;
let square: unsafe extern "C" fn(i64) -> i64 =
    unsafe { std::mem::transmute(address) };
assert_eq!(unsafe { square(123456) }, 15241383936);
```

Keep `jit` alive until all calls finish. The function signature must match the
lowered C ABI exactly. I32/I64 map to i32/i64, F32/F64 to f32/f64, and
`GlobalMemory<T>` to a pointer to live elements. Array parameters are passed as
pointers and copied into callee-owned storage; struct results use the LLVM
host ABI. Callers must provide enough memory for raw pointer indexing.

`compile_with_options(source, JitOptions { opt_level: 0, ..Default::default() })` selects an
optimization level. `compile_duration()` includes parsing, semantic checks,
optimization, and executable code generation. `optimized_ir()` exposes the
optimized module for inspection. `functions()` lists public entrypoints;
`main` aliases the emitted `ysu_main` symbol.

`JitOptions.codegen_opt_level: Option<u8>` defaults to `None`, inheriting
`opt_level` for ORC machine-code generation. Rust callers can override that
stage with a level from 0 through 3, for example
`JitOptions { opt_level: 3, codegen_opt_level: Some(2), ..Default::default() }`.
The IR optimization pipeline and its independent analysis target still use
`opt_level`; this override changes the ORC target builder only. Invalid levels
are rejected. Compilation remains eager. Python's `codegen_opt_level=` and
the versioned C options API expose the same override; existing C entrypoints
and default Python calls retain inheritance.

`JitOptions.training_opt_level: Option<u8>` defaults to `None`, inheriting
`opt_level` for instrumented training IR. A Rust caller can set `Some(1)` to
use the O1 IR pipeline and analysis target only in `compile_instrumented`.
Ordinary compilation and profile-use recompilation still use `opt_level`.
Machine-code optimization continues to follow `codegen_opt_level` or the base
`opt_level`, independently of the training IR override. Invalid training levels
are rejected even when the selected compilation mode would ignore the override.

For example, `JitOptions { training_opt_level: Some(1), ..Default::default() }`
keeps final IR/native O3 and per-pass verification enabled while reducing the
temporary training IR tier. Profiles remain tied to original lowered IR and
site identities before optimization. Recompilation validates those identities;
changing a lowering option may invalidate a profile. Atomic instrumentation and
eager compilation are preserved. Python's `training_opt_level=` and the
versioned C options API expose this policy. Existing C entrypoints and default
Python calls retain inheritance. Training and recompilation remain explicit
and eager; both costs are charged to preparation.

`JitOptions.verify_each_pass` defaults to `true` and enables LLVM verification
after individual optimization passes. Rust callers can explicitly select
`JitOptions { verify_each_pass: false, ..Default::default() }`. Both policies
retain full-module verification before optimization, after the ordinary
pipeline, and after the optional profile-selection pipeline when attempted.
LLVM verification checks IR validity; it does not prove that an optimization
preserves program semantics. Existing C/Python compile entrypoints retain the
default policy.

`compile_timings()` exposes disjoint `Duration` measurements for parsing,
semantic checks, lowering, LLVM setup, IR parsing, profile setup, explicit
verification, optimization, IR capture, symbol resolution and materialization.
`other` accounts for remaining metadata, cleanup and timer overhead; all phases
sum to `total`, which equals `compile_duration()`. The AST API reports zero
parsing time. `verification` aggregates the mandatory full-module boundary
checks. `optimization` includes requested per-pass verification and excludes
those explicit boundary checks.
Materialization includes eager native code generation, linking and address
lookup for all public functions and checked-call adapters. These intervals do
not include training or execution, and cache hits retain the original session's
measurements. `compile_timings().to_json()` returns integer nanoseconds with
`*_ns` keys.

`optimization_timings()` returns `&JitOptimizationTimings` separately
partitioning the primary `optimization` interval into the ordinary O-level
`pipeline`, optional `profile_selection`, residual `other` and `total`.
`accounted_duration()` equals `total`, which equals
`compile_timings().optimization`. `to_json()` uses integer nanoseconds under
`pipeline_ns`, `profile_selection_ns`, `other_ns` and `total_ns`. These nested
intervals are already included in compilation accounting and must not be added
a second time. The Rust-only `compile_timings().verification_checks` records
successful explicit module checks. Neither the nested details nor the check
count changes the flat compilation JSON schema.

`materialization_timings()` returns `&JitMaterializationTimings` for the eager
ORC interval. Its six parent duration fields are `submission`, `first_lookup`,
`remaining_function_lookups`, `profile_lookup`, `other` and `total`.
`accounted_duration()` sums the first five, equaling both `total` and
`compile_timings().materialization`. `to_json()` uses the corresponding integer
nanosecond keys ending in `_ns`. These nested fields and metadata leave the
flat thirteen-key compilation JSON unchanged and must not be added again to
its total. Later function lookups, training, execution and timing getters do
not update the compilation snapshot.

Submission covers creation and addition of the thread-safe module. First
lookup includes CString construction and ORC lookup until the first exported
function or adapter address is ready. Remaining function lookups and the
instrumented branch-counter lookup have separate intervals; `other` covers
residual bookkeeping and timer overhead. This still eagerly resolves every
exported entrypoint and adapter before compilation returns.

When optional LLVM object-observer hooks are available, one uniquely observed
callback whose entry timestamp falls inside the first lookup permits two
children: `first_lookup_before_object_ns` and `first_lookup_after_object_ns`.
They sum exactly to `first_lookup_ns`, and are excluded from the parent sum.
Both children are JSON null when hooks are unavailable, when multiple objects
are observed, or when the event cannot be localized to that lookup. The before
interval includes native object emission and ORC work; the after interval
includes observer overhead, linking and lookup. Neither is an exclusive
code-generator or linker timer.

Metadata records `object_observer_available` as a boolean and `object_count`,
`object_bytes`, `function_lookup_count` and `profile_lookup_count` as nonnegative
integers. Object bytes measure observed object-file buffers, not executable
memory or total allocator usage. The observer returns LLVM's buffer unchanged
without taking ownership. Its callback state is boxed inside the Engine, with
a stable address kept alive through LLJIT disposal; it exposes no Python
callback or buffer ownership.

The source-string API accepts one resolved compilation unit. It refuses
unresolved imports and module declarations. `compile_program` accepts an AST
whose imports the embedding application has resolved, and still runs semantic
checks. A JIT instance stays on its creating thread; callers arranging native
calls from other threads must synchronize access to their memory and retain the
JIT until those calls finish.

## Checked dynamic calls and compilation reuse

The compiler also exposes source-derived function signatures and generates
adapters for scalar and pointer functions. Callers can pass tagged values
instead of constructing a native function pointer themselves:

```rust
use y::cpu_jit::{CpuJit, JitValue};

let jit = CpuJit::compile("fn square(x: I64) -> I64 { return x * x; }")?;
assert_eq!(jit.function_signature("square")?.parameters.len(), 1);
assert_eq!(unsafe { jit.call("square", &[JitValue::I64(12)])? }, JitValue::I64(144));
```

Argument count and types are checked before execution. Float bits, signed
integer widths, booleans, pointers, and void results are preserved. `char`
values use unsigned bytes from 0 to 255, matching the Y runtime. Dynamic calls
require trusted source and valid pointer storage just like direct calls.
Aggregate signatures can be inspected but use `function_address` for invocation.
Generated adapters are hidden from `functions()`.

Checked-call adapters keep a call to the shared native entrypoint instead of
duplicating its optimized body. Source functions can still inline into each
other. This reduces repeated optimization and code generation, with a possible
extra call cost for small functions. `JitOptions { optimize_call_adapters:
false, ..Default::default() }` restores unrestricted adapter inlining for
controlled measurements.

`CpuJitCache::new(capacity)` provides bounded process-local reuse of identical
source and optimization options:

```rust
use y::cpu_jit::{CpuJitCache, JitOptions};

let mut cache = CpuJitCache::new(8);
let jit = cache.compile("fn answer() -> I32 { return 42; }", JitOptions::default())?;
```

A cache hit returns an `Rc<CpuJit>` without recompiling. Eviction releases the
cache's ownership; callers retaining an Rc keep their code alive. Capacity zero
disables retention. Failed compilations are not cached. This cache belongs to
the creating thread and shares any native globals when a compilation is reused.
Cache identity v8 includes all `JitOptions`, including `verify_each_pass` and
the requested `codegen_opt_level` and `training_opt_level` overrides. `None`
and an explicit inherited
level have distinct identities even when their effective levels match.
Different settings retain separate compilations and timing snapshots.

## Rotate recognition and measured branch profiles

Rotate recognition is enabled by default in JIT and AOT LLVM lowering. Exact
unsigned idioms such as `(x << 13) | (x >> 51)` become LLVM rotate intrinsics,
preserving a single multiply followed by a rotate in the unsigned benchmark.
Recognition is conservative: the repeated value must be a stable local
identifier, and complementary constant shifts must match the actual promoted
integer width. Calls, pointer reads, signed arithmetic shifts, dynamic counts,
and invalid shifts retain their existing lowering. For controlled comparisons,
`JitOptions { recognize_rotates: false, ..Default::default() }` disables it.
The compilation cache includes this option in its identity.

The CPU JIT also inlines String/Vec length and bounded indexed reads for local
allocations whose ownership, native layout, and lifetime are proven. These
reads avoid the runtime callback and allocation-registry lock. Allocation,
growth, and freeing continue through the runtime; null and invalid
indices preserve the runtime's zero/null results. Aliases, escapes, opaque
calls, and user-defined runtime overrides retain callback lowering. The older
AOT runtime uses a different layout and does not enable these reads.
`JitOptions { optimize_runtime: false, ..Default::default() }` disables them.
The same option lowers the registered runtime's byte-to-integer conversion to
an unsigned LLVM extension, allowing optimization through character reads.
The cache includes all lowering and profile-policy options in its identity.

`optimize_runtime_mutations` enables a separate capacity-checked append path for
proven local String/Vec objects. Appends write existing storage directly when
the compiler can prove the source extent and available capacity. Vector pushes
with unproved pointer sources retain callbacks. Allocation, buffer growth,
freeing, and failed-capacity cases continue through the ordinary runtime.
`JitOptions { optimize_runtime_mutations: false, ..Default::default() }`
disables this path independently of query lowering.

`optimize_runtime_copies` extends that append path to bulk String copies and
vectors constructed with a runtime element size. Bulk String appends snapshot
both lengths, check destination capacity, and use overlap-safe copying, including
self-appends. Dynamic vector pushes copy directly only when the live element
size exactly matches the proven scalar source extent; other sizes retain the
ordinary callback. Null objects, insufficient capacity, aliases without proven
ownership, and runtime overrides keep callback behavior. This option requires
`optimize_runtime_mutations`; disabling it retains the earlier single-element
and constant-size append paths. All three runtime options are enabled by
default in the CPU JIT and disabled in the AOT emitter.

`optimize_helper_effects` preserves these local fast paths across source helpers
proved to use only scalar arguments, results, locals and expressions. The
analysis follows lexical scopes and grows from scalar leaves to callers of
already-proved helpers. Helpers with memory access, references, objects,
globals, runtime/unknown calls, inline assembly, special backend directives or
unresolved recursion remain barriers. Names dispatched as load/block-pointer
intrinsics are excluded even when a source declaration has a scalar signature.
Actual call arguments are still inspected for aliases, escapes and side effects;
ordinary source calls retain their evaluation order.
This analysis does not optimize through pointer-taking helpers or assert LLVM
memory attributes. It is enabled by default in the CPU JIT, disabled in the AOT
emitter and independently disabled with `JitOptions { optimize_helper_effects:
false, ..Default::default() }`. The cache key includes the setting.

Some existing intrinsic semantics remain limited: `load` can evaluate its
argument twice, and block-pointer intrinsic dispatch can take precedence over
a source declaration. The helper proof excludes these dispatch names; it does
not repair their override behavior. The runtime/emitter alias `Vec_get` is also
currently rejected by the type checker; use the supported `yvec_get` spelling.

The Rust API supports explicit branch training and recompilation:

```rust
use y::cpu_jit::{CpuJit, JitOptions, JitValue};

let source = "fn choose(x: I64) -> I64 { if x < 0 { return -1; } return x; }";
let options = JitOptions::default();
let training = CpuJit::compile_instrumented(source, options)?;
for value in 0..1000 {
    unsafe { training.call("choose", &[JitValue::I64(value)])?; }
}
let profile = training.branch_profile()?;
let optimized = CpuJit::compile_with_profile(source, options, &profile)?;
assert_eq!(unsafe { optimized.call("choose", &[JitValue::I64(-3)])? }, JitValue::I64(-1));
```

Compilation never executes source to generate a profile. Training uses atomic
counters for the original conditional branches; the host controls which calls
run and their side effects. `branch_profile().sites()` exposes each function,
basic block, and observed true/false counts. A snapshot owns its counts and
survives dropping the training session. Counters wrap after 2^64 observations
of an individual outcome; concurrent calls can advance different counters
during the snapshot.

The profile is tied to the SHA-256 of the original lowered IR and branch-site
identities. Changed lowering is refused. Unexecuted sites receive no guessed
weights. Observed counts guide LLVM's normal O-level pipeline and, where
available, its targeted `select-optimize` pass. Older LLVM builds without that
optional pass still use the measured branch weights.
By default, profile use omits proven natural-loop header/latch weights while
retaining measured conditional work and breaks. Iteration counts alone do not
describe the fallback paths that LLVM may introduce when transforming loops.
All original counts remain available in the snapshot.
`JitOptions { profile_loop_controls: true, ..Default::default() }` explicitly
includes loop-control weights. This policy does not guarantee faster code.
`profiled_branches()` reports the number of weighted original branches;
`profile_selection_optimization()` reports whether the targeted pass ran.

The optimized session contains no profiling counters or callbacks and preserves
IEEE floating-point operations. Profile weights are optimization hints: unseen
inputs and unobserved edges retain their behavior. Old native addresses remain
valid while their original session lives; recompilation returns a separate
session. AST callers can use `compile_program_instrumented` and
`compile_program_with_profile` after resolving imports.

## Embed from Python

```sh
cargo build --release --lib
PYTHONPATH=python python3 python/examples/cpu_jit.py
```

```python
from y_lang import CPUJit

with CPUJit("fn square(x: I64) -> I64 { return x * x; }") as jit:
    print(jit.call("square", 123456))   # 15241383936
    square = jit.function("square")
    assert square(12) == 144
    print(jit.signature("square"))
    print(jit.compile_timings())
    print(jit.optimization_timings())
    print(jit.materialization_timings())
```

The wrapper reads the compiler's signature metadata, validates integer ranges
and types, and preserves F32/F64 bits through the tagged ABI. Pointer parameters
accept ctypes arrays, pointers, `byref` objects, `c_void_p`, integer addresses,
or None. Pointer results are `c_void_p`. Keep native storage alive during calls.
Use the context manager or `close()` on the creating thread. Function callables
retain their session; calls after explicit close report an error. CPU-only
imports need neither torch nor cupy. `library_path=` selects a particular
`liby.so`; otherwise the existing package library discovery is used.

Python callers can explicitly train branches and create a separate optimized
session:

```python
source = "fn choose(x: I64) -> I64 { if x < 0 { return -1; } return x; }"
with CPUJit(source, instrument=True, training_opt_level=1) as training:
    for value in range(1000):
        training("choose", value)
    print(training.branch_profile())  # fingerprint and observed edge counts
    with training.recompile_profiled() as optimized:
        assert optimized("choose", -3) == -1
```

Recompilation preserves the training session and its callables. Each session
owns its executable memory and can be closed independently. Training calls have
their ordinary program side effects; profile collection is an explicit choice.

`training_opt_level=` selects instrumented IR only; `codegen_opt_level=` selects
native code generation for every mode. Both accept 0..3 or `None` to inherit
the base `opt_level`. The defaults remain inherited, with per-pass verification
enabled. Choosing O1 training leaves final IR/native O3 when other settings are
default. These controls expose existing policies; their availability does not
establish a performance improvement for an application's workload.

`recompile_profiled()` preserves all requested tiers. Keyword overrides select
final settings independently, for example
`training.recompile_profiled(opt_level=3, codegen_opt_level=2)`.
Passing `codegen_opt_level=None` or `training_opt_level=None` resets that
override to inheritance; omitting it preserves the training session's policy.
`opt_level=None` preserves the base level. Training IR policy has no effect on
final compilation. Overrides do not modify the original session, and invalid
levels are refused before compilation. Libraries predating the options API can
still use inherited settings; requesting an override requires a rebuilt library.

## Embed from C

Include [`c_src/y_cpu_jit.h`](../c_src/y_cpu_jit.h) for the complete interface.
The existing `liby.so` exports:

```c
void *y_cpu_jit_compile(const char *source, unsigned opt_level, char **error);
void *y_cpu_jit_compile_instrumented(const char *source, unsigned opt_level, char **error);
void *y_cpu_jit_compile_profiled(const char *source, unsigned opt_level,
                               void *training_jit, char **error);
char *y_cpu_jit_branch_profile(void *training_jit, char **error);
char *y_cpu_jit_compile_timings(void *jit, char **error);
char *y_cpu_jit_optimization_timings(void *jit, char **error);
char *y_cpu_jit_materialization_timings(void *jit, char **error);
void *y_cpu_jit_function(void *jit, const char *name, char **error);
char *y_cpu_jit_signature(void *jit, const char *name, char **error);
typedef struct { uint32_t kind, reserved; uint64_t bits; } YCpuJitValue;
int32_t y_cpu_jit_call(void *jit, const char *name,
                      const YCpuJitValue *arguments, size_t count,
                      YCpuJitValue *result, char **error);
void y_cpu_jit_free(void *jit);
void y_free_string(char *error);
```

Compilation and lookup return null on failure and an allocated error when an
error output is supplied. Release errors with `y_free_string`, and release the
JIT once with `y_cpu_jit_free`. Keep handle operations on their creating thread.
Native function addresses expire when the JIT is freed. The FFI contains Rust
compiler panics instead of unwinding through C.

The additive versioned options API exposes the training and codegen tiers:

```c
YCpuJitOptions options;
char *error = NULL;
if (y_cpu_jit_options_init(&options, sizeof(options), &error) != 0) {
    /* Handle error, then y_free_string(error). */
    return;
}
options.training_opt_level = 1;  /* Instrumented IR O1; final IR/native O3. */
void *training = y_cpu_jit_compile_instrumented_with_options(source, &options, &error);
/* Check training, explicitly call its functions, and collect observations. */
void *optimized = y_cpu_jit_compile_profiled_with_options(source, &options, training, &error);
/* Check optimized; close each independently with y_cpu_jit_free. */
```

`YCpuJitOptions` contains `abi_version`, `struct_size`, `opt_level` (u32), and
`training_opt_level`/`codegen_opt_level` (i32). Initialization sets ABI version 1,
the exact size, base O3, and both optional tiers to
`Y_CPU_JIT_OPT_LEVEL_INHERIT` (-1). Base levels accept 0..3; optional tiers accept
-1 or 0..3. The compiler checks version and exact size before reading remaining
fields, and validates every tier in every mode, including inactive training
overrides. Initialization failure preserves the destination storage.

`y_cpu_jit_compile_with_options`,
`y_cpu_jit_compile_instrumented_with_options`, and
`y_cpu_jit_compile_profiled_with_options` accept a null options pointer for
unchanged defaults. Options are copied during the call. C callers supply final
options explicitly; the library does not recover them from the training handle.
All legacy compile entrypoints retain their original signatures and inherited
tiers. Verification, atomic counter semantics and eager compilation are unchanged.

`y_cpu_jit_signature` returns JSON freed with `y_free_string`.
`y_cpu_jit_compile_timings` returns the phase measurements described above as
owned JSON, also freed with `y_free_string`. Python exposes the same object
through `jit.compile_timings()`; every value is an integer number of nanoseconds.
`y_cpu_jit_optimization_timings` returns the separate optimization breakdown as
owned JSON released with `y_free_string`; Python exposes it through
`jit.optimization_timings()`. `y_cpu_jit_materialization_timings` returns the
materialization parents, nullable children and metadata described above as
owned JSON, also released with `y_free_string`; Python exposes it through
`jit.materialization_timings()`. All three getters return independent
snapshots, exclude later lookup, training and execution costs, and execute no
source code. Python requires the creating thread and a live session.
`y_cpu_jit_call` returns 0 on success and -1 on error; failure preserves the
result buffer. The reserved field must be zero. Tags are void=0, I8=1, U8=2,
I16=3, U16=4, I32=5, U32=6, I64=7, U64=8, usize=9, bool=10, F32=11, F64=12,
pointer=13. Signed integers carry their low-width two's-complement bits;
floats carry IEEE bits. Invalid tags and noncanonical bits are refused.

`y_cpu_jit_branch_profile` returns observed counts as JSON freed with
`y_free_string`. `y_cpu_jit_compile_profiled` snapshots a live instrumented
handle and returns a separate handle using those counts. The original handle
remains valid. The source must lower to the identical profiled IR; mismatches
return an error. Compilation does not call the program to train branches.

## Language and runtime coverage

The JIT reuses the LLVM host language: integer and floating-point arithmetic,
branches, loops, recursive and ordinary calls, scalar arrays, structs, and
references. Tests exercise these at O0 and O3, including widths and value
semantics. The shared emitter also now preserves F64 literal bits, implements
unsigned division/remainder/comparison/shifts, and follows IEEE behavior for
floating-point inequality with NaN.
`&&` and `||` now short-circuit from left to right in both JIT and AOT host
code, including operands with memory side effects and array bounds checks.
Division by zero, signed division overflow, and shifts beyond the operand
width retain the existing LLVM host semantics.

Native pointer-width runtime callbacks implement strings, vectors, printing,
and Unix file operations. This runtime is separate from the older C runtime,
whose 32-bit pointer handles and special low-address stack cannot be used by
ordinary in-process callers. User-defined functions take precedence over
runtime aliases. Runtime allocations have the language's existing explicit
free/lifetime behavior; dropping a JIT releases executable memory.

Unsupported LLVM host constructs retain the emitter's explicit diagnostics.
Live calls to unavailable runtime symbols fail compilation with the symbol's
name. GPU intrinsics, GUI runtime functions, nested arrays, array returns, and
other existing host-lowering gaps are not added by this JIT. Source executes
with the permissions of its embedding process; this is the same trusted-code
model as calling a compiled native Y library.

## Executable differential checks

`cargo test --test cpu_jit_differential` compiles one deterministic generated
unit in sixteen configurations: IR O0/O3, custom lowering transforms disabled
or enabled, per-pass verification disabled or enabled, and machine-code
optimization inherited or overridden to O2. Both verification policies retain
mandatory full-module checks at pipeline boundaries. Sixteen
scalar program variants execute on 128 seeded inputs per
configuration, including full-width signed and unsigned boundary values.
Independent Rust oracles check rotations, division/reconstruction, mixed ABI
values, complete memory arrays and aliasing, short-circuit side effects, owned
String/Vec contents and freed slots, and exact/IEEE floating-point results.
Failures include the seed, configuration and source. The bounded templates
avoid undefined shifts, invalid memory and signed overflow; this is executable
regression coverage for those scenarios, not a proof for arbitrary programs.

`cargo test --test cpu_jit_verification --test cpu_jit_timings` checks the two
verification policies, pipeline-boundary check counts, profile identity,
ordinary/instrumented/profiled results and exact primary/nested accounting.
Matching final IR on the tested fixtures remains bounded regression evidence.
`cargo test --test cpu_jit_codegen` checks the independent codegen option,
unchanged IR/profile settings, materialization accounting, nullable
object-event children and full-result native calls. Focused cache tests cover
requested override identity. Observer tests cover unchanged buffers, unique
events and multiple/outside-event null fallbacks; API tests conditionally
check hook availability and preserve session ownership.

## Compare with C#

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet \
  --suite runtime --optimization-stage runtime --compare-optimizations
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet \
  --suite runtime --optimization-stage runtime-append --compare-optimizations
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet \
  --suite copies --optimization-stage runtime-copies --compare-optimizations
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet \
  --suite copies --optimization-stage adapters --compare-optimizations
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/dotnet \
  --suite helpers --optimization-stage helper-effects --compare-optimizations
```

Use a .NET 8 SDK. The runner builds both workers offline, validates all outputs
against independent Python algorithms, alternates separate processes on one
pinned CPU, and saves repeated warm measurements separately from JIT preparation
and process startup. See [the benchmark guide](../benchmarks/cpu_jit/README.md)
and [the controlled optimization comparison](cpu_jit_benchmarks_optimized.md).
The `rotate-profile` stage with `--compare-optimizations` uses baseline Y with
rotate recognition and branch profiles disabled, Y that enables both, and C#
in interleaved processes. Later stages compare previous and next Y settings as
described below. Training, recompilation, and steady-state execution are
measured separately.
The [expanded eight-workload results](cpu_jit_benchmarks_expanded.md) preserve
the measurements taken before rotate recognition and branch-profile support.
The [original four-workload report](cpu_jit_benchmarks.md) retains its historical
measurements and provenance from the initial implementation.

The runtime suite adds String and Vec allocation, append, and indexed scans.
Its [ten-workload report](cpu_jit_benchmarks_runtime.md) compares previous
optimization settings with the new runtime lowering and conservative loop
profile policy in the same compiler. Both Y variants train explicitly. These
two policy changes are measured together; their individual effects are not
isolated. ASCII input equates character values while the containers and memory
management differ: Y frees each object inside the timed call, and C# uses
managed reclamation with allocation/collection statistics recorded.
The [append comparison](cpu_jit_benchmarks_runtime_append.md) keeps query,
rotate, and profile settings fixed while toggling only capacity-checked append
lowering on the same ten workloads.

The thirteen-workload `copies` suite adds byte and I64 vectors whose element
sizes come from host arguments, plus bulk String appends. Its two controlled
stages change only copy lowering or adapter inlining, respectively. The adapter
stage also compares native and checked Y calls on tiny and full-size inputs;
those supplementary timings include Rust argument validation and have no C#
reflection counterpart. Native kernel comparisons with C# remain separate.
The [copy results](cpu_jit_benchmarks_runtime_copies.md) show all three added
workloads improving over callback lowering. On a Ryzen 9 9950X with LLVM 23.1.1
and .NET 8.0.31, the nine-repeat results are:

| Workload | Next Y median ms/call | C# median ms/call | Previous Y / next Y | C# / next Y | Y wins vs C# |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dynamic byte Vec | 0.06336 | 0.09230 | 5.65x | 1.46x | 6/9 |
| Dynamic I64 Vec | 0.04001 | 0.11243 | 8.20x | 2.81x | 9/9 |
| Bulk String append | 0.02910 | 0.26557 | 1.60x | 9.13x | 9/9 |

Ratios divide median times; the paired byte-vector C# / Y range is 0.60–2.55,
and is descriptive rather than a confidence interval. The other ten workloads
and all measured losses remain in the full report. The optimization improves
all nine previous/next Y pairs for each added workload, while small-input cold
preparation increases 2.6%.

These results compare equivalent algorithms using different runtime objects.
Y uses native byte storage and measured branch profiles. C# uses UTF-16
`StringBuilder`, `List<byte>` and `List<long>` with tiering and PGO disabled.
Its I64-vector batches include garbage collection; Y includes explicit frees.
Deferred C# reclamation outside a batch is excluded. Layouts, growth policies
and allocators differ, so the String ratio cannot be attributed entirely to
code generation and these measurements do not establish a general language
ranking. Profile training and recompilation use the measured input distribution
and are excluded from steady-state timers; the reports retain their full costs.

The [adapter results](cpu_jit_benchmarks_adapters.md) show small-input cold total
preparation decreasing from 346.905 to 292.234 ms (15.8%) across five cold
samples per setting. Tiny checked calls stay near parity. Several native
numeric medians are 11–22% slower during that run. Saved native IR and generated
assembly match after normalization; those artifacts do not prove identical
executed code placement or establish the cause of the timing differences.
The report preserves all samples. Y preparation starts from source text, and
C# preparation starts from prebuilt IL, so their preparation times cover
different compilation stages.

The [final fifteen-workload helper comparison](cpu_jit_benchmarks_helper_effects.md)
changes only `optimize_helper_effects`, with the other runtime/adapter/rotate
options and measured-profile policy fixed. Nine process triples and five cold
triples restore complete scalar-helper String/Vec workloads to approximately
their direct forms' cost: 128.26x and 77.78x improvements over helper analysis
disabled, both in 9/9 pairs. The benchmark reports all other workloads, paired
ratios, the 0.02–0.99% control median losses and the 12.03% cold preparation
regression. Earlier small smokes are not validated performance evidence.

Next cold profiled compilation spends median per-sample shares of 55.51% in
LLVM optimization, including VerifyEach, and 41.82% in eager ORC
generation/linking/lookup; parsing, checks and lowering together take about
1.41 ms of a 221.637 ms compile. These phase measurements point first to pass
cost/verification profiling, then evaluation of explicit tiers and lazy
materialization. They do not isolate verification overhead or demonstrate a
benefit from changing the pipeline. Full-size training/preparation costs,
same-distribution PGO, runtime representation/GC differences, compilation scope
and all raw/hash evidence remain in the report.

The subsequent [verification-policy comparison](cpu_jit_benchmarks_verification.md)
uses nine full-size process triples and five cold triples with only
`verify_each_pass` changing. Both settings keep the new mandatory pipeline
boundary checks. Opting out reduces cold preparation medians from 419.106 to
329.331 ms (21.42%) and full-size preparation from 862.352 to 769.482 ms
(10.77%). Every preparation pair improves. The default remains `true`;
existing C/Python compilation retains that policy.

Nested optimization measurements locate most savings in the default O3
pipeline, with optional profile selection about 1.06% of cold profiled total
in the opt-in arm. Eager ORC materialization occupies 51.76% and becomes that
arm's largest compilation phase. It still combines generation, linking and
lookup for the whole module; fewer symbol lookups alone would not establish
lazy generation. Actual module/function partitioning, lower-cost initial tiers
and adapter sharing require measured amortization and full-result checks.
With the default policy, optimization remains the largest profiled phase.

Cold materialization and explicit verification become slower; eight native
workload medians regress 0.016–0.814%. The report retains signed paired costs,
all samples, matching saved instrumented/final IR, profiles and 33-source/
two-worker hashes. These are verification-policy wall-clock differences, not
exclusive verifier timings or proof that intermediate IR remains valid.
The final gate passed 319 Rust and 17 Python CPU JIT tests against rebuilt
`liby`; independent accounting/provenance and output audits passed. Historical
reports and `CODEX_SESSION_HANDOFF.md` remain unchanged. PGO, runtime layouts,
GC/free charging and source-versus-IL preparation still differ from C#.

The subsequent [ORC codegen comparison](cpu_jit_benchmarks_codegen.md) changes
only inherited machine-code O3 to explicit O2, preserving IR O3 and per-pass
verification. Nine warm process triples and five cold triples establish no
dependable preparation win: cold medians rise 1.90%, from 425.218 to
433.309 ms. Warm separate medians fall 0.41%, but the paired median is a
0.661 ms loss and five of nine pairs regress. Seven native workload medians
regress; float recurrence rises 7.03% and helper Vec 5.59%. Dynamic-byte Vec
improves 4.54% and in all nine pairs. All observations and losses are retained;
the default remains inherited O3.

The new object-event split places about 99.83% of cold profiled materialization
before object handoff, with about 0.17 ms afterward. Its intervals include ORC
and observer overhead rather than exclusive backend/linker costs. Default
profiled compilation spends median per-sample shares of 55.11% in optimization
and 42.12% in materialization. Investigate the default IR pipeline for overall
cost and native emission/preceding ORC work within materialization. Removing a
few symbol lookups does not avoid current eager whole-module compilation.
Partitioning, initial tiers and pass-level diagnostics still need their own
full-result and amortization gates.

The final gate passed 322 selected Rust and 18 Python tests after rebuilding
`liby`; independent audits passed 137,416 accounting/provenance and 19,493 scalar
output checks. Durable evidence includes the 109-file measured source archive,
exact worker/library binaries, profiles and every raw pair. Optional-observer
compatibility was reviewed against LLVM 17 headers, with execution tested on
LLVM 23.1.1. Missing APIs, real concurrent compilation and injected
post-submission failures were not fully injected. Assembly is an `llc`
reconstruction with small code model, while live ORC requests JITDefault;
actual emitted object bytes are counted but not retained or hashed. Historical
reports and the session handoff remain unchanged. Training overlaps measured
seeds; PGO, runtime representations, GC/free charging and source-to-native
versus prebuilt-IL preparation continue to limit the C# comparison.

The [training-tier comparison](cpu_jit_benchmarks_training_tier.md) uses an
explicit O1 override only for temporary instrumented IR, keeping final IR and
native codegen O3. Both settings retain per-pass verification, atomic counters,
all lowering optimizations and the same source/profile policy. Exact profiles
and retained outputs match; saved final IR and reconstructed final assembly
are identical. Temporary instrumented IR differs as expected.

Nine warm process triples and five cold triples reduce preparation medians
from 826.790 to 803.418 ms (2.83%) and 330.061 to 308.283 ms (6.60%). Every
preparation pair improves; paired median savings are 24.314 ms full and
24.532 ms cold. Instrumented compilation falls about 20%. Training execution
stays near 499 ms and remains about 62.15% of full preparation. Final O3
optimization remains the largest compilation phase. Separate external opt
diagnostics prioritize loop unrolling, InstCombine and vectorization for
future investigation; their settings/accounting differ from actual JIT timers.

Eight native medians regress 0.001–0.184%; cold final compilation's separate
medians become slower despite a small paired median saving. These losses are
retained and no final native-code improvement is claimed. Lower training tiers
may slow other workloads, so the override stays explicit with inherited
defaults. The 325 Rust/18 Python gate includes full-result cross-tier tests and
concurrent-native-call atomic checks under inherited/O1 training. Independent
audits verify settings, timing/output/profile accounting and frozen provenance;
durable raw evidence includes corrected initial checker scope failures. All
historical evidence and the session handoff remain unchanged. PGO, layouts,
GC/free charging and source-versus-IL preparation still limit the C# comparison.
