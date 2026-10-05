# Adaptive GPU JIT

`y::adaptive_jit::AdaptiveGemm` is an opt-in runtime for repeated GPU GEMMs,
with Rust, C, and Python interfaces.
It compiles and caches baseline kernels, tracks shapes that become hot, and uses
measured performance to decide whether to replace them. Ordinary `Y` compilation
is unchanged.

**Tuning is deferred by default.** An ordinary launch does not start a seconds-long
candidate search. The application explicitly calls `tune_hot` at a suitable
warmup or maintenance point. This is a change from the first prototype's default;
select `TuningPolicy::OnLaunch` to retain its automatic synchronous behavior.

```sh
cargo run --offline --release --example adaptive_gemm
# Optional dimensions (M N K):
cargo run --offline --release --example adaptive_gemm -- 512 512 512
```

The example compiles during warmup, issues three baseline GEMMs without tuning,
explicitly tunes the pending shape, and launches the selected kernel. It checks
64 outputs against a CPU reference and reports candidate count, tuning cost, and
estimated calls needed to repay a promotion. NVIDIA sm_80+ and a working CUDA
driver are required.

## Runtime lifecycle

1. `prepare(shape)` compiles and caches the analytic baseline without requiring
   application buffers or increasing its launch count. This can move first-use
   compilation into warmup. A launch also prepares a missing shape automatically.
2. `launch(shape, a, b, c)` enqueues `C = A * B` and increments the shape's count.
   By default, 32 successful enqueues make it eligible for tuning. Repeated calls
   reuse the compiled module even after the threshold.
3. `pending_shapes()` returns hot resident shapes with no previous tuning attempt,
   ordered by launch count, then oldest use. It cannot grow beyond the kernel cache.
4. `tune_hot(max_shapes)` attempts up to that many pending shapes synchronously.
   Zero is a no-op. Each returned `TuningReport` contains the shape and its stats,
   including a per-shape failure. The method does not count scratch measurements
   as application launches. The job limit is not a wall-time deadline.
5. Later launches reuse the selected module. The cache holds 16 shapes by default
   and evicts the least recently used. Successful preparation also updates LRU.
   Eviction removes pending work and forgets that shape's counters/tuning result;
   revisiting it starts cold unless optional disk persistence restores a decision.
   Eviction and replacement synchronize before unload.

| `TuningPolicy` | Behavior |
|---|---|
| `Deferred` (default) | Launches track hotness; only `tune_hot` measures candidates. |
| `OnLaunch` | The next launch after the threshold can synchronously tune; explicit `tune_hot` is also available. |
| `Disabled` | Compile/cache baselines; no pending work, empirical tuning, or disk-cache access. |

```rust,ignore
use y::adaptive_jit::{AdaptiveGemm, AdaptiveJitConfig, GemmShape};
use y::cuda_runtime::CudaContext;

let ctx = CudaContext::new().expect("CUDA device");
let mut jit = AdaptiveGemm::new(&ctx, AdaptiveJitConfig::default())?;
let shape = GemmShape { m: 256, n: 256, k: 256 };
jit.prepare(shape)?; // Optional warmup: no application buffers needed.
// Allocate/upload A and B and allocate C through ctx; see the runnable example.
unsafe { jit.launch(shape, a_device_ptr, b_device_ptr, c_device_ptr)?; }
ctx.synchronize()?;
// After enough calls, choose an appropriate maintenance point:
let reports = jit.tune_hot(1)?;
```

The raw-pointer launch is unsafe: supply correctly sized, aligned, contiguous
allocations in the borrowed context, keep them alive until completion, and prevent
overlapping output/input storage and conflicting accesses. Keep that context
current while using and dropping the runtime and its buffers. Preparation,
launch, and maintenance validate the exact current context.

## C and Python embedding

Build the shared library and run the Python example (NumPy and CUDA required):

```sh
cargo build --offline --release --lib
python3 examples/adaptive_gemm.py
# Run twice to observe persistence from separate Python processes:
python3 examples/adaptive_gemm.py --cache-dir target/python-jit-cache 256 256 256
```

The C declarations are in `c_src/y_adaptive_jit.h`; the stdlib `ctypes` wrapper is
`tools/adaptive_jit.py`. Both expose preparation, raw device-pointer launch,
explicit synchronization, pending shapes, bounded maintenance, and statistics.
The Python example owns its CUDA allocations and context, checks the GEMM against
NumPy, and closes the adaptive runtime before releasing those resources.

An embedding host initializes CUDA and makes its chosen context current first.
`y_adaptive_jit_create_current` borrows that **exact context**; it does not create
another context, select device zero, switch contexts, or take ownership of the
host's context. Keep it alive through handle destruction. The Rust equivalent is
`unsafe { CudaContext::borrow_current() }` followed by
`AdaptiveGemm::from_owned_context(wrapper, config)`, which owns only the wrapper.

Every handle call belongs on the creating OS thread, with the original context
current. Calls from another thread or context return an error. Do not concurrently
access or destroy a handle. `y_adaptive_jit_destroy` synchronizes before unloading
modules and leaves the handle live on a context/thread/synchronization error;
restore the original thread/context and retry. Successful destruction invalidates
the pointer and leaves the host context usable. Python uses explicit `close()` or
a context manager, with no GPU cleanup from a garbage-collector finalizer.

Launches use CUDA's default stream. Callers must finish input writes from other
streams before launching, retain buffers through completion, and synchronize
before reading or freeing output. For example, after the host creates its context
and allocates correctly sized device buffers:

```python
from tools.adaptive_jit import AdaptiveGemm

with AdaptiveGemm(hot_threshold=32) as jit:
    jit.prepare(m, n, k)
    jit.launch(m, n, k, a_device_address, b_device_address, c_device_address)
    jit.synchronize()
    # Call during a suitable maintenance period after enough launches:
    reports = jit.tune_hot(1)
    stats = jit.stats(m, n, k)
```

The wrapper takes raw device addresses; it
cannot check allocation ownership, size, strides, or overlapping storage. It does
not manage tensor lifetimes or framework streams. All shape, dtype, alignment,
and nonoverlap requirements of the Rust runtime still apply.

Initialize C options with `y_adaptive_jit_config_init` and `sizeof(config)`, or pass
null options for defaults. The version and exact structure size are checked before
reading the remaining fields. An optional cache directory is a UTF-8 C string,
copied during creation. Status calls return zero on success and -1 on failure;
create/JSON calls return null on failure. An optional `char **error_out` is cleared
on entry and receives an allocated error string. Free all returned JSON and error
strings with `y_free_string`; free a previous error before reusing its output slot.
The Python wrapper handles these string allocations automatically.

Statistics and maintenance reports are JSON objects with the same runtime fields;
latencies use microseconds and `tuning_time_seconds` uses seconds. An absent resident
shape is a stats error. Per-shape tuning failures appear inside returned reports,
so callers should inspect `tier` and `tuning_error` even after successful maintenance.
Rust panics are caught at these new C boundaries; a panic during a handle operation
poisons the handle, which must then be closed. Invalid pointers remain a caller
contract and cannot be made safe by catching a panic.

## Selection and failure handling

The default search budget is eight **distinct generated kernels**, including the
baseline. Requested pipeline depths can clamp to identical code. Candidates are
deduplicated by generated PTX instructions/directives plus launch geometry, ignoring
line comments. A duplicate does not consume a budget slot. Selection preserves
analytic ranking and stops once the distinct-kernel budget is filled; cold baseline
compilation still stops at the first loadable candidate.

Adaptive tuning uses private scratch buffers and checks numerical results before
believing any timing. A short, interleaved screening pass first compares the
baseline and candidates. If even a candidate's best screen timing does not beat
the baseline's worst screen timing by the requested improvement, the runtime
retains the baseline immediately. This avoids the full warmup and final search
when the sampled candidates show no plausible benefit. If only baseline timing
spread makes a challenger look promising, one additional short screen can resolve
that uncertainty; a clear first-screen improvement still goes directly to full
confirmation. Invalid timings are errors, not evidence that a search succeeded.

A promising candidate still triggers the full three-second clock ramp, a new
screen at warmed clocks, and the final interleaved measurement pass. The baseline
is always included in that final comparison. Promotion still requires a latency
reduction exceeding both 5% by default and the observed noise band. The short
screen can only retain the checked baseline; it cannot promote a challenger.
The standalone empirical autotuner keeps its full measurement path.

A quick negative result is a bounded search decision, not proof that the baseline
is globally optimal. Short measurements can miss improvements under different
GPU states. Successful early retention can be cached just like a full search
that retained the baseline; its baseline/selected timings are coarse screen
observations and are equal. Use a new decision directory to request another search.

If no candidate clears that bar, the baseline stays. Codegen, resource, timing,
or replacement-load failures keep the baseline and record `TuningFailed`; there
is no retry while the shape is resident. A baseline that positively fails the
numerical check becomes `RejectedBaseline`: subsequent dispatches refuse it.

`stats(shape)` reports launches, tier, tuning attempts/time, candidates measured,
measured baseline/selected latency, and any error. `cache_hit` identifies a restored
decision, and `cache_error` reports a nonfatal persistence problem separately from
a tuning failure. `candidates_measured` counts
unique candidates that passed the numerical gate and were timed; it can be below
the configured budget. `estimated_break_even_launches()` is available after a
promotion and estimates `ceil(tuning_time / time_saved_per_call)`.

## Reuse across runs

Persistence is **off by default**. Set `AdaptiveJitConfig::cache_dir` to an
application-controlled directory to save successful decisions, including a
measurement that retained the baseline. Failed tuning and rejected baselines are
never saved. `max_disk_cache_entries` defaults to 128; successful writes evict
oldest-written owned records above the cap, leaving other filenames alone.

Run the same command twice:

```sh
cargo run --offline --release --example adaptive_gemm -- --cache-dir target/adaptive-jit-cache 512 512 512
```

The first run measures and saves a decision. A matching later run reports
`cache_hit=true`, regenerates/loads the selected kernel, and performs no empirical
tuning. Its historical baseline/selected timings remain available, but launches,
tuning attempts, tuning time, and measured-candidate count start at zero. Hot
launches do not queue a new search for that restored resident decision.

Each record contains candidate settings, launch geometry, code fingerprints,
and timing diagnostics; it contains no PTX, binary, or GPU handles. Reuse checks:

- The actual GPU UUID, reported CUDA API compatibility version, Linux NVIDIA
  kernel-driver build text, and the runtime's hardware profile.
- Relevant compiler/codegen, search, measurement, and cache source fingerprints,
  plus locked dependencies and build mode. Debug and release builds use different
  namespaces. Scheduling thresholds and directory names do not change a decision's
  identity; shape, search budget, and minimum improvement do.
- The locally regenerated baseline and selected PTX/launch fingerprints. Candidate
  settings must belong to the current search space before they reach codegen.

The driver identity uses `/proc/driver/nvidia/version`; it does not hash all
userspace driver/JIT libraries. Missing build metadata or unavailable identity
queries disable optional persistence and leave ordinary compilation/tuning usable.
This implementation therefore requires the Linux metadata path for persistence,
even though the underlying runtime has other platform support.

Records are versioned, checksummed, size-limited, and replaced atomically. A bad,
old, oversized, unreadable, or unloadable decision falls back to the normal baseline
and records `cache_error`; it does not suppress future tuning. Failed writes also
leave the successful runtime choice usable. Concurrent writers may temporarily
exceed the entry cap while publishing; pruning follows each successful write.
The checksum detects accidental corruption and is not authentication against a
writer who controls the directory.

To force remeasurement, use a fresh cache directory or remove its `.yjit` records
before creating a new runtime. A valid cached choice is not rebenchmarked on load;
it need not stay optimal as clocks, contention, or workload cache residency change.

## Scope and costs

- Row-major F16 inputs, F32 accumulation/output, `C = A * B`. M is 1..16384;
  N and K are multiples of 16 in 16..16384. No transposes, strides, alpha/beta,
  fused epilogues, FP8, exact/ZeroDrift, or verified cubin path. `Y_SMEM_PAD` must
  be unset or equal to 8 when emitting/tuning kernels.
- Deferred mode removes empirical tuning from dispatch; it is not a real-time
  latency guarantee. Cold compilation, CUDA calls, and eviction synchronization
  can still block. Prewarm the needed shapes and size the cache accordingly.
- Measurement still takes seconds and can use substantial scratch memory.
  `max_candidates` bounds the distinct search, and `tune_hot` bounds shape count;
  neither is a time or memory limit. The clock ramp has not been shortened.
- Hotness counts past use rather than predicting future savings. A five-second
  tune saving one microsecond needs about five million more calls to break even.
  Synthetic timing and weight-buffer rotation may differ from application cache
  residency. A faster kernel is not guaranteed, and no project-wide speedup is
  claimed. Numerical checking is sampled testing, not a proof of equivalence;
  floating-point rounding can change between tiles.
- Live modules and launch statistics remain in-process and tied to one CUDA
  context and operation. Optional disk persistence stores successful decisions.
  There is no background worker, workload-drift detection, or general CPU JIT.

## Benchmark total workload cost

The benchmark compares the analytic baseline (`TuningPolicy::Disabled`), a fresh
adaptive decision, and reuse of that decision in a separate process. It measures
1,000, 10,000, and 100,000 application calls, including runtime setup, preparation,
explicit tuning, dispatch, and synchronization. A common GPU warmup and input
allocation are excluded from the measured interval; the report documents these exclusions.

```sh
python3 tools/benchmark_adaptive_jit.py --output target/adaptive-jit-benchmark-new
```

This builds a release Rust worker and runs three trials each for 256³, 512³,
1024³, and 1×4096×4096 GEMMs. The decode case rotates six weight buffers; the square
cases reuse one. Output directories must be new; existing results are never
replaced. Optional `--shape M,N,K,WEIGHT_COPIES`, `--repeats`, and `--calls` narrow
the run. Use the normal Cargo target directory for this harness.

The baseline and selected kernels use the same runtime dispatch path. These
measurements isolate the value and cost of adaptive selection; they do not measure
its dispatch overhead relative to older direct CUDA launches. Fresh decisions are
measured on a warmed GPU with the driver's code cache already exercised. Break-even
projections require consistently positive savings in separate, interleaved timing
rounds. A retained baseline does not earn back tuning through a faster kernel.

The [initial comparison](benchmarks/adaptive_jit_2026-09-24/report.md) and
[staged-tuning comparison](benchmarks/adaptive_jit_screening_2026-09-24/report.md)
include raw observations, source fingerprints, hardware metadata, correctness
checks, and the conditions under which the results apply.

## Resident dispatch overhead

Prepared resident launches use one shape-cache lookup and one exact current-context
check unless `OnLaunch` needs to tune. Three-pointer CUDA launches keep their
argument copies on the stack. Failed enqueues still leave hotness and LRU unchanged;
cold compilation and tuning keep their subsequent context check.

The [dispatch comparison](benchmarks/adaptive_jit_dispatch_2026-09-24/report.md)
compares preserved release executables with the same kernel and benchmark sources.
It records complete-batch and enqueue times separately, plus an untimed Rust
allocation audit. It uses the default deferred policy without calling maintenance;
this measures reuse overhead, not the benefit of adaptive kernel selection.

The measured allocation count fell from two to zero per launch, and paired median
enqueue times improved by about 3–4%. Completion times were mixed, including a
slower smallest case; the report preserves that result and instrumentation limits.

To repeat, build `cargo build --release --example adaptive_jit_dispatch_bench`
and preserve the worker executable and source hashes before and after a change.
Then supply those binaries and sibling `before_sources.json`/`after_sources.json`
metadata to the runner (the measured copies remain under `target/`):

```sh
python3 tools/benchmark_adaptive_jit_dispatch.py \
  --before target/adaptive-jit-dispatch-2026-09-24/before \
  --after target/adaptive-jit-dispatch-2026-09-24/after \
  --output target/adaptive-jit-dispatch-repeat
```

Use repeated `--shape M,N,K` arguments to compare other dimensions with those
same preserved binaries, for example `--shape 2048,2048,2048 --shape 4096,4096,4096`.
By default each worker completes 41 timing batches of 1,000 calls plus an
allocation audit and warmup. Workers built from the updated example also accept
`--calls` and `--rounds` through the runner; use the same settings in both versions.
For example, `--calls 100 --rounds 15` bounds the duration of larger GEMMs.
`--trials` controls the number of alternating before/after process pairs per shape.
The older preserved workers support only the default batch sizes.

The [larger-shape comparison](benchmarks/adaptive_jit_large_dispatch_2026-09-24/report.md)
uses 2048³ and 4096³ with six process pairs per shape and shorter identical batches.
Neither showed a clear completion-time speedup: median paired changes were 0.62%
slower and 0.12% faster, respectively, with mixed gains/regressions across pairs.
The report retains the earlier long-batch pilot and explains source/binary checks.

The [kernel scheduling and matched cuBLAS comparison](benchmarks/gemm_scheduling_2026-09-25/report.md)
tests two native operand-order changes with both kernels in one process. Neither
produced a repeatable completed-time improvement, so both were reverted. A matched
FP16-input/FP32-output cuBLAS reference took about 5.3% less time at 2048³ and 6.4%
less at 4096³ on this GPU. It is a benchmark reference, not an integrated backend.
The report includes exact PTX exports, raw paired timings, rejected patches, and
commands for the new comparison tools.

## Validation

```sh
cargo test --offline --lib
cargo test --offline --test adaptive_jit_gpu --test adaptive_jit_ffi_gpu --test adaptive_jit_screening_gpu --test gemm_compute_schedule_gpu -- --ignored --test-threads=1
python3 tests/adaptive_jit_python.py
```

GPU tests are explicitly ignored by ordinary test runs because they require
hardware and measurement time. The explicit command fails if CUDA is unavailable.
It checks numerical results, preparation/LRU, pending work, deferred/automatic/
disabled policies, context identity, tuning failure fallback, persistence reuse,
corruption, policy mismatches, and disk failures. Run serially
without competing GPU measurements.

The first prototype was validated on 2026-09-24 (RTX 4070 Ti SUPER, sm_89):
157 library tests and five GPU tests passed. Its 256x256x256 release example
matched CPU samples at relative L2 error 3.79e-7. Tuning took 3.36 seconds and
retained the baseline (about 3.09 microseconds in that harness); no kernel speedup
was established. Per-phase host times include different compilation,
synchronization, and clock-warmup costs and are not kernel speedup comparisons.

After the deferred-mode and candidate-budget changes, local validation on the
same GPU passed **164 library tests and eight GPU tests**. Both release examples
kept the first three launches on the baseline with zero tuning attempts; only
the explicit maintenance call ran measurements. Eight distinct kernels were
measured for each shape.

| Shape | Baseline kernel | Selected kernel | Outcome | Tuning cost |
|---|---:|---:|---|---:|
| 256x256x256 | 3.115 us | 3.115 us | Baseline retained | 3.171 s |
| 512x512x512 | 7.440 us | 7.059 us | Promoted | 3.479 s |

These are single local runs using the harness's best-round statistic. The
512-shape latency reduction was about 5.1%; its estimated tuning break-even was
9.1 million additional calls. CPU-reference sample errors were 3.79e-7 and
8.87e-7 respectively. These observations demonstrate a working promotion and
deferred scheduling, not a general or independently reproduced speedup claim.

Persistent-cache validation on the same GPU passed **181 library tests and ten
GPU tests**. A separate two-process 512x512x512 example run retained its baseline:

| Process | Cache hit | Runtime setup | Preparation | New tuning |
|---|---|---:|---:|---:|
| First | false | 0.703 ms | 0.481 ms | 3.369 s, 8 kernels |
| Second | true | 0.696 ms | 0.300 ms | none, 0 kernels |

Both processes matched the CPU samples at relative L2 error 8.87e-7. The second
process processed zero maintenance jobs and reported the saved 7.422 us timing as
historical. This demonstrates avoided repeated tuning, not a newly measured kernel
speedup. CUDA's own code cache may contribute to module-load timings; the adaptive
cache stores only the tuning decision.

C/Python embedding validation on the same GPU passed **189 library tests, 13
GPU tests, and eight Python boundary tests**. C and C++ clients compiled and linked
against `liby.so`, checked the config layout/defaults and error-string ownership,
and confirmed that creation fails when no host context is current. GPU coverage
checks full numerical output, deferred tuning, rejection of another thread or
context, close retry after restoring the context, and continued use of host
allocations after the adaptive handle closes.

Two separate Python 256x256x256 runs matched 64 CPU-reference samples at relative
L2 error 4.898e-7. The first measured eight candidates in 3.176 seconds and retained
the baseline; the second restored the decision with zero candidates measured and
zero maintenance jobs. This validates the shared-library integration and avoided
remeasurement; it establishes no additional kernel speedup.
