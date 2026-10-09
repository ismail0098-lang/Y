# Y CPU JIT checked-call adapter optimization comparison with C#

Measured 2026-10-06T19:04:58.271377+00:00 on AMD Ryzen 9 9950X 16-Core Processor; pinned to [0].
LLVM 23.1.1, .NET 8.0.31. 9 independent interleaved triples; 32 calls per timed batch after 12 standardized warmups per kernel.

This compares Y with measured branch profiles (PGO) against C# with tiering and dynamic PGO disabled. The runtime results include allocation, growth, scans and reclamation costs. Y uses flat byte strings and Vec; C# uses UTF-16 `StringBuilder` and `List<byte>`/`List<long>`, with different layouts, growth policies and allocators. These measurements do not isolate code generation or establish a general language speed ranking.

Both Y variants use rotate recognition, proven-local queries and byte-to-ASCII conversion, spare-capacity append lowering and measured profiles excluding structural loop controls. Both enable all runtime copy lowering; previous Y permits source bodies to inline into checked-call adapters, while next Y retains calls to the shared native entrypoints. Both compile instrumented code, execute 12 training calls per kernel, then recompile with their actual profiles. Timed code contains no probes. Only the adapter setting changes on the same compiler/source/input distribution.

Compact adapters reduce preparation from 346.905 to 292.234 ms (15.8% lower) in these five small-input cold samples of the same Y source unit. This figure describes that cold setup; full-size preparation is separately reported below at 786.910 versus 759.585 ms. Y starts from source and performs training/recompilation, while C# starts from prebuilt IL, so their cold costs do not compare equivalent compiler stages. Tiny checked calls stay near parity: 34.16 versus 34.57 ns/call. The run shows reduced compilation work and duplicated code, without a checked-call throughput improvement.

The raw native medians below retain measured slowdowns: next Y integer branching is 11.1% slower, float recurrence 21.9%, unsigned mixing 16.4%, short-circuit branching 16.9%, and binary search 16.1%. A broad timing episode appears around samples 3–7 across both Y variants and C#, while normalized native IR and LLVM-generated assembly are identical between Y settings for all fourteen source functions. This suggests timing variation; saved llc assembly does not prove identical ORC addresses or code placement, or establish an external cause. These samples do not establish a native runtime benefit or regression caused by the adapter setting. The samples remain unfiltered. Headline ratios divide per-engine median times, rather than taking the median of paired ratios.

The C# I64 result records `GC.CollectionCount` deltas of `[1,1,1]` for Gen0/Gen1/Gen2 in each of nine timed batches, and a median 8,398,592 allocated bytes per 32-call batch. Y pays explicit frees inside its timers; C# reclamation deferred until after a batch is uncharged. Runtime ratios include these costs and representation differences.

| Kernel | Previous Y ms/call | Next Y ms/call | C# ms/call | Previous / next Y | C# / next Y | Paired C# / Y range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.5984 | 0.6647 | 1.0641 | 0.90x | 1.60x | 1.47–1.80x |
| recursive_fib | 0.1511 | 0.1500 | 0.2935 | 1.01x | 1.96x | 1.29–2.66x |
| float_recurrence | 1.0974 | 1.3382 | 1.0963 | 0.82x | 0.82x | 0.76–1.03x |
| indexed_memory | 0.1082 | 0.1086 | 0.1699 | 1.00x | 1.56x | 1.52–1.69x |
| unsigned_mix | 0.4577 | 0.5328 | 0.4578 | 0.86x | 0.86x | 0.85–1.00x |
| float_dot | 0.0966 | 0.0964 | 0.0964 | 1.00x | 1.00x | 0.97–1.03x |
| short_circuit | 1.8228 | 2.1315 | 1.8336 | 0.86x | 0.86x | 0.84–1.04x |
| binary_search | 1.4869 | 1.7256 | 1.4784 | 0.86x | 0.86x | 0.80–1.00x |
| string_scan | 0.0380 | 0.0382 | 0.3119 | 0.99x | 8.16x | 6.87–9.49x |
| vec_scan_append | 0.0641 | 0.0638 | 0.0911 | 1.00x | 1.43x | 1.08–2.24x |
| vec_dynamic_byte | 0.0650 | 0.0645 | 0.0930 | 1.01x | 1.44x | 1.05–2.01x |
| vec_dynamic_i64 | 0.0401 | 0.0399 | 0.1135 | 1.01x | 2.85x | 2.77–5.26x |
| string_bulk_append | 0.0291 | 0.0293 | 0.2669 | 0.99x | 9.12x | 8.50–12.93x |

Headline ratios above 1 favor next Y by the ratio of medians. Paired ranges describe observed min/max ratios from the same repeat, not confidence intervals. Small differences should be read as near parity.

| Preparation stage | Previous Y median ms (small cold inputs) | Next Y median ms (small cold inputs) |
| --- | ---: | ---: |
| Instrumented source compilation | 131.050 | 103.621 |
| Measured profile collection | 0.917 | 0.914 |
| Profile snapshot | 0.005 | 0.005 |
| Optimized recompilation | 214.528 | 187.880 |
| Total preparation | 346.905 | 292.234 |

C# prebuilt IL preparation: 2.807 ms. Its source-to-IL build is excluded. Y starts with source text; these preparation costs have different starting points. Both Y cold variants compile the same source unit eagerly: thirteen kernels, the bump_if helper and all supported checked-call adapters. Cold inputs use n=64 and Fibonacci n=10; the bulk-string kernel performs two 32-character appends.

Profile training and recompilation are explicit work excluded from steady-state kernel timers. Full-size training costs are retained in every warm-run record. Profiles use the same input distribution as the timed calls; branch frequencies do not prove a branch unreachable.

The original eight definitions remain unchanged. The two earlier runtime-object workloads allocate an empty local object, append runtime-selected ASCII A/z values, scan eight times, and return a weighted checksum. Every 128 characters they deliberately read index -1 and index length; both languages return zero for these guarded reads. Y uses mutable byte String/Vec objects; C# uses `StringBuilder` and `List<byte>`. ASCII makes the character values equal despite different byte versus UTF-16 representations. Container layouts, growth rules and allocators differ. This measures equivalent algorithms and observable results, not identical runtime representations.

Allocation, growth, append callbacks, scans, index guards and explicit Y frees occur inside each new kernel's timer. C# uses ordinary managed reclamation: collections occurring within a batch are included, and allocated bytes and collection counts are saved per new kernel. Deferred reclamation after the batch is not charged. Native input arrays, output buffers, hashing and JSON formatting remain outside timers for the original eight workloads.

All previous, next and instrumented-training outputs match independent Python implementations. Full-array hashes verify native memory writes and all short-circuit counters are checked exactly. Raw branch counts, profile fingerprints, applied-site counts, both LLVM IRs and LLVM-generated native assembly are preserved beside this report.

C# retains Release .NET 8 with tiering and ReadyToRun disabled, so this remains a fixed non-PGO comparison. Kernel methods use NoInlining; the two earlier runtime-object workloads use one indirect host call per invocation in each language. The other eight use the same host boundaries as previous reports. Workers run sequentially with rotating orders. CPU affinity reduces migration; other host activity and CPU frequency can still affect results. These synthetic workloads do not establish a general language speed ranking.

Inputs: integer=250000, Fibonacci=25/26, recurrence=1000000, memory=262144, unsigned=250000, dot=262157, logical=250000, search=20000, string=16384, vector=16384.

Three copy workloads extend the preserved ten definitions. The dynamic byte and I64 vectors receive element_size as a runtime argument (1 or 8 respectively), append from an initialized scalar local, scan eight times and include final length in their checksums. C# uses `List<byte>` and `List<long>` for the same values; this comparison samples matching scalar widths. The I64 values populate high bits and the independent Python oracle models modulo-2^64 signed arithmetic. The bulk string creates a 32-character ASCII chunk from its seed, appends it repeatedly, scans eight times with the same invalid-index guards, includes final length and frees both objects. C# uses `StringBuilder.Append(StringBuilder)`. Allocation, growth and copies occur inside these kernel timers; representations and reclamation still differ.

Copy inputs: byte vector=16384, I64 vector=16384, bulk string=512 appends of 32 characters.

| Checked API workload | Previous native ns/call | Previous checked ns/call | Next native ns/call | Next checked ns/call |
| --- | ---: | ---: | ---: | ---: |
| integer_branch_tiny | 1.1 | 34.2 | 1.1 | 34.6 |
| integer_branch_large | 599881.9 | 597548.8 | 720654.5 | 633817.0 |

Tiny checked calls use integer_branch n=1 with 20000 calls and 64 warmups; large calls use the standard integer input/call/warmup counts. Native and checked outputs are compared exactly in the worker; independent Python references check both full-stream hashes and saved first/last output excerpts. Timers include native dispatch or checked .call validation, frame packing, allocation and return handling. Output buffers and hashing are outside timers. Native batches run first and checked batches second in each process, so cache/frequency effects can contribute to the API comparison. These supplemental Y-only measurements are separate from C# native entrypoint comparisons.

Full-size preparation costs

| Stage | Previous Y median ms | Next Y median ms |
| --- | ---: | ---: |
| Instrumented compilation | 130.722 | 110.107 |
| Profile collection | 444.368 | 456.566 |
| Profile snapshot | 0.016 | 0.015 |
| Profiled recompilation | 213.689 | 194.555 |
| Total preparation | 786.910 | 759.585 |

The full-size table uses twelve training calls per kernel with the measured dimensions. Total preparation includes function lookup, training-buffer construction, result formatting and trainer disposal, so the individual stages do not sum exactly to total preparation. These preparation costs remain separate from native and checked steady-state timers.

Measured profile evidence: previous workers record 119,860,798 outcomes across 85 raw sites and apply weights to 66 sites. next workers record 119,860,798 outcomes across 85 raw sites and apply weights to 66 sites. Raw loop-control counts remain available even though their weights are omitted.

Saved final IR confirms all fourteen compact adapters retain one call to their source entrypoint and no branches. The recurrence adapter contains no duplicated floating multiply/add instructions. All fourteen source functions have identical normalized IR and generated assembly between settings.

| Code-generation artifact | Previous bytes | Next bytes |
| --- | ---: | ---: |
| Optimized IR text | 577672 | 520274 |
| Generated assembly text | 316310 | 281655 |

These text sizes describe retained compilation artifacts, not the allocated executable memory size. The checked API includes argument validation, lookup, bit-frame allocation/packing and return handling. A tiny extra native call is small relative to that work; this run does not demonstrate an API throughput improvement. Large checked/native timings share the timing episode and fixed native-first ordering, so they should not be read as a precise overhead estimate.

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
  --suite copies --optimization-stage adapters --compare-optimizations --cpu 0 \
  --output build_artifacts/new-adapters-run
```

The [prior append report](cpu_jit_benchmarks_runtime_append.md) and all earlier evidence remain intact. New durable evidence is in [benchmark_data/cpu_jit_adapters](benchmark_data/cpu_jit_adapters/metadata.json), including all 27 warm records, 15 cold records, independent references, full outputs and memory hashes, source/binary provenance, actual profiles, both final IRs, generated native assembly and process output. All 31 measured source hashes matched after both runs, and saved summaries were recomputed exactly from raw records.
