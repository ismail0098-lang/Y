# Y CPU JIT runtime optimization comparison with C#

Measured 2026-10-06T17:14:17.410144+00:00 on AMD Ryzen 9 9950X 16-Core Processor; pinned to [0].
LLVM 23.1.1, .NET 8.0.31. 9 independent interleaved triples; 32 calls per timed batch after 12 standardized warmups per kernel.

Both Y variants use the final compiler and common semantic fixes; previous settings do not replay a historical compiler binary. Previous Y enables rotate recognition and applies all measured branch profiles, with runtime queries using the existing callbacks. Next Y also exposes the headers and guarded byte reads of proven local String/Vec objects to LLVM, lowers the byte-to-ASCII conversion directly, and omits branch weights on structural natural-loop headers/latches. Both variants compile instrumented code, execute 12 training calls per kernel, then recompile with their actual profiles. The final timed code contains no profiling probes. This comparison changes both runtime lowering and loop-profile policy; it does not isolate their individual performance effects.

The two new kernels improve by 16.16x (string) and 13.18x (byte vector) against the previous settings. C# remains about 1.20x faster on the string workload and 4.02x faster on the byte-vector workload. Across the unchanged eight kernels, previous-to-next median changes are small (about 0.7% slower through 3.4% faster); this run does not establish a blanket loop-policy speedup. Some paired ranges are wide or cross parity, so the median comparisons are descriptive results from this machine.

| Kernel | Previous Y ms/call | Next Y ms/call | C# ms/call | Previous / next Y | C# / next Y | Paired C# / Y range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.6061 | 0.6075 | 1.0606 | 1.00x | 1.75x | 1.26–2.02x |
| recursive_fib | 0.1569 | 0.1517 | 0.3302 | 1.03x | 2.18x | 1.73–2.56x |
| float_recurrence | 1.1058 | 1.0942 | 1.1014 | 1.01x | 1.01x | 0.82–1.25x |
| indexed_memory | 0.1086 | 0.1094 | 0.1711 | 0.99x | 1.56x | 0.76–2.46x |
| unsigned_mix | 0.4639 | 0.4631 | 0.4659 | 1.00x | 1.01x | 0.89–1.22x |
| float_dot | 0.0959 | 0.0962 | 0.0977 | 1.00x | 1.02x | 1.00–1.91x |
| short_circuit | 1.8408 | 1.8393 | 2.0792 | 1.00x | 1.13x | 1.12–1.37x |
| binary_search | 1.5262 | 1.4978 | 1.5270 | 1.02x | 1.02x | 0.95–1.20x |
| string_scan | 5.1293 | 0.3175 | 0.2647 | 16.16x | 0.83x | 0.80–1.11x |
| vec_scan_append | 5.1745 | 0.3927 | 0.0976 | 13.18x | 0.25x | 0.21–0.43x |

Ratios above 1 favor next Y. Paired ranges describe observed min/max values, not confidence intervals. Small differences should be read as near parity.

| Preparation stage | Previous Y median ms (small cold inputs) | Next Y median ms (small cold inputs) |
| --- | ---: | ---: |
| Instrumented source compilation | 83.443 | 91.937 |
| Measured profile collection | 0.749 | 0.511 |
| Profile snapshot | 0.002 | 0.003 |
| Optimized recompilation | 84.687 | 114.273 |
| Total preparation | 169.677 | 207.407 |

C# prebuilt IL preparation: 1.603 ms. Its source-to-IL build is excluded. Y starts with source text; these preparation costs have different starting points. Small cold inputs use n=64 and Fibonacci n=10.

Profile training and recompilation are explicit work excluded from steady-state kernel timers. Full-size training costs are retained in every warm-run record. Profiles use the same input distribution as the timed calls; branch frequencies do not prove a branch unreachable.

The original eight definitions remain unchanged. The two new workloads allocate an empty local object, append runtime-selected ASCII A/z values, scan eight times, and return a weighted checksum. Every 128 characters they deliberately read index -1 and index length; both languages return zero for these guarded reads. Y uses mutable byte String/Vec objects; C# uses StringBuilder and List<byte>. ASCII makes the character values equal despite different byte versus UTF-16 representations. Container layouts, growth rules and allocators differ. This measures equivalent algorithms and observable results, not identical runtime representations.

Allocation, growth, append callbacks, scans, index guards and explicit Y frees occur inside each new kernel's timer. C# uses ordinary managed reclamation: collections occurring within a batch are included, and allocated bytes and collection counts are saved per new kernel. Deferred reclamation after the batch is not charged. Native input arrays, output buffers, hashing and JSON formatting remain outside timers for the original eight workloads.

All previous, next and instrumented-training outputs match independent Python implementations. Full-array hashes verify native memory writes and all short-circuit counters are checked exactly. Raw branch counts, profile fingerprints, applied-site counts, both LLVM IRs and LLVM-generated native assembly are preserved beside this report.

C# retains Release .NET 8 with tiering and ReadyToRun disabled, so this remains a fixed non-PGO comparison. Kernel methods use NoInlining; the two object workloads use one indirect host call per invocation in each language. The other eight use the same host boundaries as previous reports. Workers run sequentially with rotating orders. CPU affinity reduces migration; other host activity and CPU frequency can still affect results. These synthetic workloads do not establish a general language speed ranking.

Inputs: integer=250000, Fibonacci=25/26, recurrence=1000000, memory=262144, unsigned=250000, dot=262157, logical=250000, search=20000, string=16384, vector=16384.

Full-size preparation costs

| Stage | Previous Y median ms | Next Y median ms |
| --- | ---: | ---: |
| Instrumented compilation | 82.265 | 87.643 |
| Profile collection | 409.999 | 338.568 |
| Profile snapshot | 0.006 | 0.007 |
| Profiled recompilation | 82.078 | 112.943 |
| Total preparation | 581.189 | 539.858 |

Each full-size previous worker records 79,903,654 branch outcomes across 29 sites and applies all 29. Each next worker records 89,365,606 outcomes across 37 sites, including added inline access guards, and applies weights to 25 sites. The remaining structural loop-control sites keep their raw observations without receiving weights.

The full-size table uses twelve training calls per kernel with the measured dimensions. Total preparation also includes function lookup, training-buffer construction, profile/result formatting and trainer disposal; the individual stages do not sum exactly to total preparation.

Saved final IR confirms all String/Vec length and indexed-byte query callbacks and ychar_to_ascii conversion callbacks disappear from the next versions of both new kernels. Allocation, push and free callbacks remain. The checked negative and out-of-range reads continue to return zero. LLVM-generated native assembly from both saved IRs is retained as code-generation evidence.

C# managed allocation evidence

| Kernel | Median allocated bytes per 32-call batch | Gen0/Gen1/Gen2 collections across nine batches |
| --- | ---: | --- |
| string_scan | 1575936 | 0/0/0 |
| vec_scan_append | 1059584 | 0/0/0 |

These counts cover the timed batches only; ordinary managed object reclamation can occur later. Allocation is charged in the batch, while Y also pays its explicit free callbacks there.

Reproduce this run with:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite runtime --optimization-stage runtime --compare-optimizations \
  --output build_artifacts/new-runtime-run
```

The [prior eight-kernel optimization report](cpu_jit_benchmarks_optimized.md) and its evidence remain intact. New durable evidence is in [benchmark_data/cpu_jit_runtime](benchmark_data/cpu_jit_runtime/metadata.json), including raw samples, cold records, complete output arrays, independent references, source/binary hashes, tool versions, two IRs, native assembly and process output.
