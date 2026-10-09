# Y versus RyuJIT — every existing matched CPU workload

This comparison covers all **15 existing matched CPU JIT workloads**. The original, expanded, runtime and copies suites are subsets of the helpers suite; rows are counted once.

These are reanalysed sealed measurements from 2026-10-08T18:08:22.299912+00:00, not a new timed run. AMD Ryzen 9 9950X 16-Core Processor; logical CPU0; LLVM23.1.1; .NET8.0.31 x64 RyuJIT. Nine rotating warm triples, five cold triples, 32 timed calls, 12 standard warmups and 12 Y training calls. Host frequency/activity remain uncontrolled.

**Y normal** uses normal final-loop tuning; **Y outer-suppressed** disables unrolling only for eligible original outer loops. Both retain final IR/native O3, temporary training IR O1, VerifyEach and eager compilation. Neither chooses a different policy per workload. Both production unrolling defaults remain true.

**RyuJIT** is the existing Release .NET8 C# worker with tiering and ReadyToRun disabled: immediate optimized non-tiered compilation, with no dynamic PGO. These results do not measure the default tiered/dynamic-PGO configuration or a newer .NET runtime. [RyuJIT overview](https://github.com/dotnet/runtime/blob/main/docs/design/coreclr/jit/ryujit-overview.md), [runtime compilation settings](https://learn.microsoft.com/en-us/dotnet/core/runtime-config/compilation).

## Native execution

Warm inputs: integer, unsigned and short-circuit kernels run 250,000 iterations; Fibonacci alternates n=25/26; float recurrence runs 1,000,000 iterations; indexed memory performs 262,144 accesses; float dot uses 262,157 terms; binary search performs 20,000 queries. Memory/dot/search arrays contain 65,536 elements. String/Vec workloads use 16,384 elements and eight scans; bulk String uses 512 appends of 32 characters and eight scans.

Times are separate medians in milliseconds per call. RyuJIT/Y ratios above one favor Y; ratios below one favor RyuJIT. Small differences describe near parity; no confidence interval is claimed.

| Workload | Y normal ms | Y outer-suppressed ms | RyuJIT ms | RyuJIT / Y normal | RyuJIT / Y outer-suppressed |
| --- | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.708031 | 0.772575 | 1.307292 | 1.846× | 1.692× |
| recursive_fib | 0.154563 | 0.151020 | 0.383958 | 2.484× | 2.542× |
| float_recurrence | 1.425498 | 1.396642 | 1.439791 | 1.010× | 1.031× |
| indexed_memory | 0.109822 | 0.109779 | 0.172652 | 1.572× | 1.573× |
| unsigned_mix | 0.571519 | 0.552825 | 0.570414 | 0.998× | 1.032× |
| float_dot | 0.098231 | 0.123359 | 0.096230 | 0.980× | 0.780× |
| short_circuit | 2.246555 | 2.233316 | 2.462689 | 1.096× | 1.103× |
| binary_search | 1.860434 | 1.842879 | 1.795827 | 0.965× | 0.974× |
| string_scan | 0.038327 | 0.037399 | 0.353015 | 9.211× | 9.439× |
| vec_scan_append | 0.064214 | 0.064149 | 0.094949 | 1.479× | 1.480× |
| vec_dynamic_byte | 0.064140 | 0.063488 | 0.103651 | 1.616× | 1.633× |
| vec_dynamic_i64 | 0.040775 | 0.041365 | 0.111461 | 2.734× | 2.695× |
| string_bulk_append | 0.029492 | 0.029442 | 0.360473 | 12.223× | 12.244× |
| string_scan_helper | 0.036789 | 0.037388 | 0.357248 | 9.711× | 9.555× |
| vec_scan_append_helper | 0.061916 | 0.061090 | 0.095615 | 1.544× | 1.565× |

Separate medians favor Y normal in **12/15** workloads and Y outer-suppressed in **13/15**. These counts do not weight workloads by application frequency and do not establish a general language ranking.

## Paired native ratios

Each ratio first divides matching-process RyuJIT time by Y time, then takes its median. Ranges retain all nine pairs, including losses. Pair medians and ratios of separate medians can disagree.

| Workload | Normal paired ratio | Range | Y-normal gains | Outer-suppressed paired ratio | Range | Y-outer gains |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 1.783× | 1.369–2.046 | 9/9 | 1.727× | 1.468–1.974 | 9/9 |
| recursive_fib | 2.159× | 1.239–2.611 | 9/9 | 1.962× | 1.613–2.670 | 9/9 |
| float_recurrence | 1.017× | 0.763–1.454 | 5/9 | 1.031× | 0.873–1.567 | 5/9 |
| indexed_memory | 1.564× | 0.794–2.813 | 8/9 | 1.564× | 1.226–2.885 | 9/9 |
| unsigned_mix | 1.009× | 0.823–1.237 | 6/9 | 1.055× | 0.796–1.289 | 6/9 |
| float_dot | 0.985× | 0.459–1.029 | 2/9 | 0.981× | 0.427–1.553 | 2/9 |
| short_circuit | 1.080× | 0.957–1.136 | 7/9 | 1.074× | 1.051–1.133 | 9/9 |
| binary_search | 0.976× | 0.817–1.078 | 3/9 | 0.974× | 0.836–1.033 | 2/9 |
| string_scan | 8.881× | 1.598–11.956 | 9/9 | 9.503× | 1.768–11.411 | 9/9 |
| vec_scan_append | 1.447× | 0.930–2.983 | 8/9 | 1.428× | 1.062–2.990 | 9/9 |
| vec_dynamic_byte | 1.605× | 0.571–3.239 | 7/9 | 1.628× | 0.607–3.387 | 8/9 |
| vec_dynamic_i64 | 2.724× | 2.605–5.317 | 9/9 | 2.518× | 0.954–5.562 | 8/9 |
| string_bulk_append | 11.743× | 3.817–14.427 | 9/9 | 12.158× | 2.126–14.408 | 9/9 |
| string_scan_helper | 9.658× | 3.932–10.297 | 9/9 | 9.454× | 2.167–10.070 | 9/9 |
| vec_scan_append_helper | 1.544× | 0.610–3.213 | 7/9 | 1.565× | 1.487–3.134 | 9/9 |

## Preparation and startup

All rows are milliseconds and medians. Y prepares from source with instrumented compilation, explicit profile collection/snapshot and final recompilation. RyuJIT prepares selected methods from prebuilt IL using PrepareMethod; source-to-IL build is excluded. The kernel methods and three scalar helpers are selected explicitly (18 methods for 15 workloads). PrepareMethod does not recursively prepare arbitrary callees; other host/runtime work can occur on first invocation. [Installed-version implementation](https://github.com/dotnet/runtime/blob/v8.0.31/src/coreclr/vm/reflectioninvocation.cpp#L1191-L1207).

| Measurement | Y normal ms | Y outer-suppressed ms | RyuJIT ms |
| --- | ---: | ---: | ---: |
| warm: final compilation / selected IL preparation | 311.287615 | 139.708205 | 3.087464 |
| warm: first-call set | 7.142869 | 9.625227 | 9.444554 |
| warm: launch to JSON | 1436.343822 | 1265.519614 | 795.836315 |
| warm: full Y preparation | 1080.992430 | 915.366656 | Different scope above |
| cold: final compilation / selected IL preparation | 331.769910 | 155.763253 | 6.570653 |
| cold: first-call set | 0.022950 | 0.017410 | 1.401365 |
| cold: launch to JSON | 477.677274 | 304.205665 | 326.315609 |
| cold: full Y preparation | 470.913258 | 301.271602 | Different scope above |

Cold workers use n=64, Fibonacci n=10 and two 32-character bulk appends. Cold records contain first-call-set totals, not individual cold-workload throughput. Launch-to-JSON includes host/startup/formatting and is not an exclusive JIT timer.

## Scope and interpretation

Y uses explicit measured PGO whose training inputs overlap measurement seeds; RyuJIT here has dynamic PGO disabled. Y invokes native function pointers. C# uses NoInlining entry methods, with delegate dispatch for the four direct/helper String/Vec scan workloads and direct calls for the others; scalar helpers retain normal inlining. Integer arithmetic widths, algorithms and observable results match, while String/Vec object layouts, growth/allocators and reclamation differ. Y Strings store bytes; C# StringBuilder uses UTF-16. ASCII inputs make these character values match. Y allocation/free and C# in-batch collections are charged, while deferred managed reclamation is excluded. Raw GC/allocation records remain in the source evidence.

The two Y checked-API microbenchmarks (tiny/large integer calls) have no matching C# checked-call benchmark. CUDA/adaptive-dispatch, GPU/Triton/cuBLAS/attention, CPU AOT/OpenBLAS GEMM and ZK benches likewise have no matching RyuJIT implementation in this corpus. This report covers every currently matched CPU-JIT workload, with no claim of covering every public benchmark suite.

All outputs, complete memory hashes, side-effect counters and profiles were checked in the original sealed run. Its 342 Rust/18 Python gates and independent structural/scalar/copy audits are retained. Bounded gates do not prove arbitrary-program equivalence. The derived JSON retains all absolute samples, paired ratios, gains/losses and both aggregation definitions.

[Sealed measurement report](cpu_jit_benchmarks_outer_unroll.md) · [Original raw evidence](benchmark_data/cpu_jit_outer_unroll/README.md) · [Derived data and provenance](benchmark_data/cpu_jit_ryujit_all_workloads_2026-10-08/README.md)
