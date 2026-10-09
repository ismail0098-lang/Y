# Y CPU JIT copy optimization comparison with C#

Measured 2026-10-06T19:02:26.073235+00:00 on AMD Ryzen 9 9950X 16-Core Processor; pinned to [0].
LLVM 23.1.1, .NET 8.0.31. 9 independent interleaved triples; 32 calls per timed batch after 12 standardized warmups per kernel.

This compares Y with measured branch profiles (PGO) against C# with tiering and dynamic PGO disabled. The runtime results include allocation, growth, scans and reclamation costs. Y uses flat byte strings and Vec; C# uses UTF-16 `StringBuilder` and `List<byte>`/`List<long>`, with different layouts, growth policies and allocators. ASCII makes the values agree here. These measurements do not isolate code generation or establish a general language speed ranking.

Both Y variants use rotate recognition, proven-local queries and byte-to-ASCII conversion, spare-capacity append lowering and measured profiles excluding structural loop controls. Both use compact checked-call adapters; previous Y disables dynamic-width/bulk copy lowering and next Y enables it. Both compile instrumented code, execute 12 training calls per kernel, then recompile with their actual profiles. Timed code contains no probes. Only copy lowering changes on the same compiler/source/input distribution.

Next Y improves over previous Y by 5.65x for dynamic byte vectors, 8.20x for I64 vectors and 1.60x for bulk strings. Each improves in all nine previous/next pairs. These headline ratios divide per-engine median times, rather than taking the median of paired ratios. Against C#, the byte result is 1.46x with Y faster in 6/9 pairs and a paired range of 0.60–2.55x; it does not establish a consistent advantage. I64 and bulk strings favor Y in 9/9 pairs, with ratios of medians of 2.81x and 9.13x respectively.

The C# I64 result records `GC.CollectionCount` deltas of `[1,1,1]` for Gen0/Gen1/Gen2 in each of nine timed batches, and a median 8,398,592 allocated bytes per 32-call batch. Y pays explicit frees inside its timers; C# reclamation deferred until after a batch is uncharged. These differences are part of the reported runtime result.

The preserved ten workloads remain broadly stable; the Fibonacci median is 2.9% slower and binary search is 1.0% slower in next Y. Small differences and wide float-dot ranges should be read cautiously. Small-input cold preparation increases from 255.752 to 262.357 ms, about 2.6%; this run shows no preparation benefit.

| Kernel | Previous Y ms/call | Next Y ms/call | C# ms/call | Previous / next Y | C# / next Y | Paired C# / Y range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.5959 | 0.5950 | 1.0553 | 1.00x | 1.77x | 1.54–1.80x |
| recursive_fib | 0.1489 | 0.1533 | 0.2901 | 0.97x | 1.89x | 1.29–2.35x |
| float_recurrence | 1.0928 | 1.0986 | 1.0994 | 0.99x | 1.00x | 0.88–1.02x |
| indexed_memory | 0.1113 | 0.1112 | 0.1703 | 1.00x | 1.53x | 0.88–1.67x |
| unsigned_mix | 0.4574 | 0.4588 | 0.4582 | 1.00x | 1.00x | 0.86–1.01x |
| float_dot | 0.0954 | 0.0958 | 0.0956 | 1.00x | 1.00x | 0.54–1.96x |
| short_circuit | 1.8034 | 1.8063 | 1.8017 | 1.00x | 1.00x | 0.85–1.01x |
| binary_search | 1.4452 | 1.4597 | 1.4377 | 0.99x | 0.98x | 0.81–1.00x |
| string_scan | 0.0378 | 0.0380 | 0.2722 | 1.00x | 7.17x | 6.86–10.78x |
| vec_scan_append | 0.0633 | 0.0633 | 0.0944 | 1.00x | 1.49x | 1.07–2.25x |
| vec_dynamic_byte | 0.3583 | 0.0634 | 0.0923 | 5.65x | 1.46x | 0.60–2.55x |
| vec_dynamic_i64 | 0.3283 | 0.0400 | 0.1124 | 8.20x | 2.81x | 2.77–3.59x |
| string_bulk_append | 0.0467 | 0.0291 | 0.2656 | 1.60x | 9.13x | 6.90–15.00x |

Headline ratios above 1 favor next Y by the ratio of medians. Paired ranges describe observed min/max ratios from the same repeat, not confidence intervals. Small differences should be read as near parity.

| Preparation stage | Previous Y median ms (small cold inputs) | Next Y median ms (small cold inputs) |
| --- | ---: | ---: |
| Instrumented source compilation | 91.502 | 94.785 |
| Measured profile collection | 0.920 | 0.911 |
| Profile snapshot | 0.004 | 0.005 |
| Optimized recompilation | 162.683 | 165.839 |
| Total preparation | 255.752 | 262.357 |

C# prebuilt IL preparation: 2.449 ms. Its source-to-IL build is excluded. Y starts with source text; these preparation costs have different starting points. Both Y cold variants compile the same source unit eagerly: thirteen kernels, the bump_if helper and all supported checked-call adapters. Cold inputs use n=64 and Fibonacci n=10; the bulk-string kernel performs two 32-character appends.

Profile training and recompilation are explicit work excluded from steady-state kernel timers. Full-size training costs are retained in every warm-run record. Profiles use the same input distribution as the timed calls; branch frequencies do not prove a branch unreachable.

The original eight definitions remain unchanged. The two earlier runtime-object workloads allocate an empty local object, append runtime-selected ASCII A/z values, scan eight times, and return a weighted checksum. Every 128 characters they deliberately read index -1 and index length; both languages return zero for these guarded reads. Y uses mutable byte String/Vec objects; C# uses `StringBuilder` and `List<byte>`. ASCII makes the character values equal despite different byte versus UTF-16 representations. Container layouts, growth rules and allocators differ. This measures equivalent algorithms and observable results, not identical runtime representations.

Allocation, growth, append callbacks, scans, index guards and explicit Y frees occur inside each new kernel's timer. C# uses ordinary managed reclamation: collections occurring within a batch are included, and allocated bytes and collection counts are saved per new kernel. Deferred reclamation after the batch is not charged. Native input arrays, output buffers, hashing and JSON formatting remain outside timers for the original eight workloads.

All previous, next and instrumented-training outputs match independent Python implementations. Full-array hashes verify native memory writes and all short-circuit counters are checked exactly. Raw branch counts, profile fingerprints, applied-site counts, both LLVM IRs and LLVM-generated native assembly are preserved beside this report.

C# retains Release .NET 8 with tiering and ReadyToRun disabled, so this remains a fixed non-PGO comparison. Kernel methods use NoInlining; the two earlier runtime-object workloads use one indirect host call per invocation in each language. The other eight use the same host boundaries as previous reports. Workers run sequentially with rotating orders. CPU affinity reduces migration; other host activity and CPU frequency can still affect results. These synthetic workloads do not establish a general language speed ranking.

Inputs: integer=250000, Fibonacci=25/26, recurrence=1000000, memory=262144, unsigned=250000, dot=262157, logical=250000, search=20000, string=16384, vector=16384.

Three copy workloads extend the preserved ten definitions. The dynamic byte and I64 vectors receive element_size as a runtime argument (1 or 8 respectively), append from an initialized scalar local, scan eight times and include final length in their checksums. C# uses `List<byte>` and `List<long>` for the same values; this comparison samples matching scalar widths. The I64 values populate high bits and the independent Python oracle models modulo-2^64 signed arithmetic. The bulk string creates a 32-character ASCII chunk from its seed, appends it repeatedly, scans eight times with the same invalid-index guards, includes final length and frees both objects. C# uses `StringBuilder.Append(StringBuilder)`. Allocation, growth and copies occur inside these kernel timers; representations and reclamation still differ.

Copy inputs: byte vector=16384, I64 vector=16384, bulk string=512 appends of 32 characters.

Full-size preparation costs

| Stage | Previous Y median ms | Next Y median ms |
| --- | ---: | ---: |
| Instrumented compilation | 92.536 | 95.027 |
| Profile collection | 439.927 | 437.000 |
| Profile snapshot | 0.014 | 0.014 |
| Profiled recompilation | 167.647 | 173.499 |
| Total preparation | 700.496 | 709.266 |

The full-size table uses twelve training calls per kernel with the measured dimensions. Total preparation includes function lookup, training-buffer construction, result formatting and trainer disposal, so the individual stages do not sum exactly to total preparation. These preparation costs remain separate from native and checked steady-state timers.

Measured profile evidence: previous workers record 119,062,078 outcomes across 79 raw sites and apply weights to 60 sites. next workers record 119,860,798 outcomes across 85 raw sites and apply weights to 66 sites. Raw loop-control counts remain available even though their weights are omitted.

Saved final IR confirms dynamic vector appends read the actual runtime element size and require it to match the scalar storage width (1 or 8). Spare-capacity paths emit one exact-width store and a length update; fallback paths retain Vec_push for growth or mismatched widths. Bulk string appends emit guarded llvm.memmove, a length update and trailing NUL, with String_push_str retained for growth. Constructor and free callbacks remain. Query and ASCII conversion callbacks are absent in both variants. These checks are generic local-object lowering, and do not match kernel names.

C# managed allocation evidence

| Kernel | Median allocated bytes per 32-call batch | Gen0/Gen1/Gen2 collections across nine batches |
| --- | ---: | --- |
| string_scan | 1575936 | 0/0/0 |
| vec_scan_append | 1059584 | 0/0/0 |
| vec_dynamic_byte | 1059584 | 0/0/0 |
| vec_dynamic_i64 | 8398592 | 9/9/9 |
| string_bulk_append | 1582592 | 0/0/0 |

Allocation is charged in timed batches; managed reclamation can occur later, while Y pays explicit frees in its timers. This remains an equivalent-algorithm comparison with different container layouts, growth policies, character representations and allocators.

Reproduce this run with:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite copies --optimization-stage runtime-copies --compare-optimizations --cpu 0 \
  --output build_artifacts/new-runtime-copies-run
```

The [prior append report](cpu_jit_benchmarks_runtime_append.md) and all earlier evidence remain intact. New durable evidence is in [benchmark_data/cpu_jit_runtime_copies](benchmark_data/cpu_jit_runtime_copies/metadata.json), including all 27 warm records, 15 cold records, independent references, full outputs and memory hashes, source/binary provenance, actual profiles, both final IRs, generated native assembly and process output. All 31 measured source hashes matched after both runs, and saved summaries were recomputed exactly from raw records.
