# Y CPU JIT append optimization comparison with C#

Measured 2026-10-06T17:30:08.029396+00:00 on AMD Ryzen 9 9950X 16-Core Processor; pinned to [0].
LLVM 23.1.1, .NET 8.0.31. 9 independent interleaved triples; 32 calls per timed batch after 12 standardized warmups per kernel.

Both Y variants use the final compiler and common semantic fixes, rotate recognition, proven-local header/byte query lowering, direct byte-to-ASCII conversion and profiles that omit structural natural-loop headers/latches. Previous Y appends through callbacks. Next Y emits inline appends when a proven local object has spare capacity; growth and free remain callbacks. Both compile instrumented code, execute 12 training calls per kernel, then recompile with their actual profiles. Timed code contains no probes. This controlled comparison changes the append optimization setting on the same compiler/source and input distribution.

The string workload improves by 8.34x against the previous query-only settings; next Y is 6.78x faster than C# in the median. The byte vector workload improves by 5.58x against the previous query-only settings; next Y is 1.55x faster than C# in the median. The unchanged eight numeric workloads should be assessed separately, with small changes read as near parity. These are descriptive results on this machine, and the C# container representations differ.

| Kernel | Previous Y ms/call | Next Y ms/call | C# ms/call | Previous / next Y | C# / next Y | Paired C# / Y range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.5915 | 0.5910 | 1.0345 | 1.00x | 1.75x | 1.72–1.77x |
| recursive_fib | 0.1487 | 0.1501 | 0.2918 | 0.99x | 1.94x | 1.84–2.02x |
| float_recurrence | 1.0929 | 1.0896 | 1.0973 | 1.00x | 1.01x | 1.00–1.01x |
| indexed_memory | 0.1085 | 0.1086 | 0.1693 | 1.00x | 1.56x | 1.40–1.63x |
| unsigned_mix | 0.4537 | 0.4550 | 0.4542 | 1.00x | 1.00x | 1.00–1.01x |
| float_dot | 0.0958 | 0.0957 | 0.0955 | 1.00x | 1.00x | 0.98–1.11x |
| short_circuit | 1.8018 | 1.8027 | 2.0444 | 1.00x | 1.13x | 1.13–1.14x |
| binary_search | 1.4429 | 1.4434 | 1.4620 | 1.00x | 1.01x | 1.01–1.02x |
| string_scan | 0.3157 | 0.0378 | 0.2564 | 8.34x | 6.78x | 6.66–7.33x |
| vec_scan_append | 0.3546 | 0.0635 | 0.0986 | 5.58x | 1.55x | 1.33–1.73x |

Ratios above 1 favor next Y. Paired ranges describe observed min/max values, not confidence intervals. Small differences should be read as near parity.

| Preparation stage | Previous Y median ms (small cold inputs) | Next Y median ms (small cold inputs) |
| --- | ---: | ---: |
| Instrumented source compilation | 96.285 | 99.673 |
| Measured profile collection | 0.491 | 0.476 |
| Profile snapshot | 0.003 | 0.003 |
| Optimized recompilation | 125.750 | 130.348 |
| Total preparation | 224.319 | 231.353 |

C# prebuilt IL preparation: 1.605 ms. Its source-to-IL build is excluded. Y starts with source text; these preparation costs have different starting points. Small cold inputs use n=64 and Fibonacci n=10.

Profile training and recompilation are explicit work excluded from steady-state kernel timers. Full-size training costs are retained in every warm-run record. Profiles use the same input distribution as the timed calls; branch frequencies do not prove a branch unreachable.

The original eight definitions remain unchanged. The two new workloads allocate an empty local object, append runtime-selected ASCII A/z values, scan eight times, and return a weighted checksum. Every 128 characters they deliberately read index -1 and index length; both languages return zero for these guarded reads. Y uses mutable byte String/Vec objects; C# uses StringBuilder and List<byte>. ASCII makes the character values equal despite different byte versus UTF-16 representations. Container layouts, growth rules and allocators differ. This measures equivalent algorithms and observable results, not identical runtime representations.

Allocation, growth, append operations and callback slow paths, scans, index guards and explicit Y frees occur inside each new kernel's timer. C# uses ordinary managed reclamation: collections occurring within a batch are included, and allocated bytes and collection counts are saved per new kernel. Deferred reclamation after the batch is not charged. Native input arrays, output buffers, hashing and JSON formatting remain outside timers for the original eight workloads.

All previous, next and instrumented-training outputs match independent Python implementations. Full-array hashes verify native memory writes and all short-circuit counters are checked exactly. Raw branch counts, profile fingerprints, applied-site counts, both LLVM IRs and LLVM-generated native assembly are preserved beside this report.

C# retains Release .NET 8 with tiering and ReadyToRun disabled, so this remains a fixed non-PGO comparison. Kernel methods use NoInlining; the two object workloads use one indirect host call per invocation in each language. The other eight use the same host boundaries as previous reports. Workers run sequentially with rotating orders. CPU affinity reduces migration; other host activity and CPU frequency can still affect results. These synthetic workloads do not establish a general language speed ranking.

Inputs: integer=250000, Fibonacci=25/26, recurrence=1000000, memory=262144, unsigned=250000, dot=262157, logical=250000, search=20000, string=16384, vector=16384.

Full-size preparation costs

| Stage | Previous Y median ms | Next Y median ms |
| --- | ---: | ---: |
| Instrumented compilation | 81.597 | 83.435 |
| Profile collection | 330.747 | 327.522 |
| Profile snapshot | 0.008 | 0.008 |
| Profiled recompilation | 104.396 | 107.292 |
| Total preparation | 516.606 | 519.953 |

Measured profile evidence: previous workers record 89,365,606 branch outcomes across 37 sites and apply weights to 25 sites. Next workers record 90,152,038 branch outcomes across 43 sites and apply weights to 31 sites. All raw outcomes remain available even when structural loop-control sites receive no weights.

The full-size table uses twelve training calls per kernel with the measured dimensions. Total preparation also includes function lookup, training-buffer construction, profile/result formatting and trainer disposal; the individual stages do not sum exactly to total preparation.

Saved final IR confirms both variants lower length/indexed-byte queries and byte-to-ASCII conversions directly. The previous variant appends through callbacks; the next emits inline data/header stores for spare-capacity appends and retains callbacks for growth. Allocation and free remain callbacks. The checked negative and out-of-range reads continue to return zero. LLVM-generated native assembly from both saved IRs is retained as code-generation evidence.

C# managed allocation evidence

| Kernel | Median allocated bytes per 32-call batch | Gen0/Gen1/Gen2 collections across nine batches |
| --- | ---: | --- |
| string_scan | 1575936 | 0/0/0 |
| vec_scan_append | 1059584 | 0/0/0 |

These counts cover the timed batches only; ordinary managed object reclamation can occur later. Allocation is charged in the batch, while Y also pays its explicit free callbacks there.

Reproduce this run with:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite runtime --optimization-stage runtime-append --compare-optimizations \
  --output build_artifacts/new-runtime-run
```

The [prior runtime-query report](cpu_jit_benchmarks_runtime.md) and its evidence remain intact. New durable evidence is in [benchmark_data/cpu_jit_runtime_append](benchmark_data/cpu_jit_runtime_append/metadata.json), including raw samples, cold records, complete output arrays, independent references, source/binary hashes, tool versions, two IRs, native assembly and process output.
