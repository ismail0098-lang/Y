# CPU JIT scalar-helper effect results — 2026-10-07

Preserving local runtime proofs across ordinary scalar helpers improves the complete helper String workload from **4.691276 to 0.036576 ms/call (128.26×)** and the helper Vec workload from **4.751707 to 0.061095 ms/call (77.78×)**. Both improve in all nine previous/next pairs. These are repeated, full-size results from the final verified sources; the earlier small smoke is not used as performance evidence.

There is a preparation cost: small-input cold preparation increases from **306.054 to 342.875 ms (+12.03%)**. Nine of the thirteen existing workload medians also become slightly slower (0.02–0.99%); all losses and samples are retained below. The helper result restores the local runtime paths already available to direct code. It does not establish that helper calls inherently improve speed or that Y generally beats C#.

Measured run started `2026-10-07T20:35:46.754974+00:00` on AMD Ryzen 9 9950X 16-Core Processor, Linux x86-64, logical CPU 0. LLVM 23.1.1; .NET SDK 8.0.425/runtime 8.0.31. Nine interleaved process triples and five cold triples ran sequentially, with no concurrent agent builds, tests or CPU audits during timing. Each warm worker uses twelve training calls per Y workload, twelve standardized warmup calls and thirty-two timed calls per workload. Host activity and CPU frequency remain uncontrolled beyond affinity.

## Native execution

Times are medians in ms/call. Previous/next and C#/next divide the per-engine medians; values above one favor next Y. The final column counts paired samples in which next Y is faster than C#. Each timed batch includes native/managed host dispatch and output storage.

| Workload | Previous Y | Next Y | C# | Previous / next | C# / next | Next Y wins vs C# |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.596563 | 0.594422 | 1.065266 | 1.00360× | 1.79211× | 9/9 |
| recursive_fib | 0.155449 | 0.152046 | 0.294265 | 1.02238× | 1.93537× | 9/9 |
| float_recurrence | 1.095332 | 1.097783 | 1.096408 | 0.99777× | 0.99875× | 4/9 |
| indexed_memory | 0.109109 | 0.109020 | 0.169990 | 1.00082× | 1.55926× | 9/9 |
| unsigned_mix | 0.457299 | 0.457855 | 0.457155 | 0.99879× | 0.99847× | 4/9 |
| float_dot | 0.095981 | 0.096432 | 0.096585 | 0.99532× | 1.00159× | 7/9 |
| short_circuit | 1.803611 | 1.807788 | 1.943218 | 0.99769× | 1.07491× | 9/9 |
| binary_search | 1.442580 | 1.442832 | 1.472394 | 0.99983× | 1.02049× | 2/9 |
| string_scan | 0.038170 | 0.038549 | 0.262212 | 0.99016× | 6.80196× | 9/9 |
| vec_scan_append | 0.063829 | 0.063841 | 0.091573 | 0.99981× | 1.43438× | 9/9 |
| vec_dynamic_byte | 0.063719 | 0.063848 | 0.101891 | 0.99798× | 1.59584× | 9/9 |
| vec_dynamic_i64 | 0.040027 | 0.040024 | 0.111418 | 1.00009× | 2.78382× | 9/9 |
| string_bulk_append | 0.029236 | 0.029286 | 0.260468 | 0.99828× | 8.89388× | 9/9 |
| string_scan_helper | 4.691276 | 0.036576 | 0.264349 | 128.26215× | 7.22746× | 9/9 |
| vec_scan_append_helper | 4.751707 | 0.061095 | 0.092458 | 77.77571× | 1.51335× | 9/9 |

Paired ratios retain repeat identity. Their min/max ranges describe these observations; they are not confidence intervals. No samples were removed.

| Workload | Previous / next paired median | Paired range | Next improves | C# / next paired median | Paired range |
| --- | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 1.00110× | 0.97052–1.01296 | 5/9 | 1.77098× | 1.75027–1.82149 |
| recursive_fib | 1.00895× | 0.90637–1.17280 | 5/9 | 1.93506× | 1.84680–2.23406 |
| float_recurrence | 0.99749× | 0.97734–1.00585 | 3/9 | 0.99997× | 0.97920–1.02532 |
| indexed_memory | 0.99349× | 0.87884–1.11794 | 3/9 | 1.56159× | 1.39524–1.75915 |
| unsigned_mix | 0.99920× | 0.96587–1.01685 | 3/9 | 0.99989× | 0.96842–1.00504 |
| float_dot | 0.99765× | 0.98134–1.00763 | 3/9 | 1.00246× | 0.98094–1.08268 |
| short_circuit | 0.99016× | 0.96705–1.01181 | 1/9 | 1.06484× | 1.06078–1.09273 |
| binary_search | 0.99971× | 0.96122–1.03373 | 4/9 | 0.99337× | 0.95957–1.03369 |
| string_scan | 0.99153× | 0.65071–1.04604 | 3/9 | 6.83248× | 5.41192–8.61688 |
| vec_scan_append | 1.00182× | 0.98159–1.05272 | 5/9 | 1.43126× | 1.17076–1.66306 |
| vec_dynamic_byte | 1.00157× | 0.97560–1.37011 | 5/9 | 1.58440× | 1.54276–1.75457 |
| vec_dynamic_i64 | 1.00038× | 0.76161–1.21340 | 5/9 | 2.74904× | 2.02724–3.60421 |
| string_bulk_append | 1.00477× | 0.98442–1.55190 | 5/9 | 8.89388× | 6.58975–11.34018 |
| string_scan_helper | 128.99474× | 109.71005–142.05189 | 9/9 | 7.19298× | 6.79694–8.55288 |
| vec_scan_append_helper | 78.04249× | 57.02506–85.83003 | 9/9 | 1.51423× | 1.02872–2.18824 |

Measured next-Y median regressions: `float_recurrence` +0.224%; `unsigned_mix` +0.122%; `float_dot` +0.470%; `short_circuit` +0.232%; `binary_search` +0.017%; `string_scan` +0.994%; `vec_scan_append` +0.019%; `vec_dynamic_byte` +0.203%; `string_bulk_append` +0.172%.

`float_recurrence`, `unsigned_mix`, `float_dot` and `binary_search` are near parity with C#. For binary search the C#/next ratio of medians is 1.02049, but the paired median is 0.99337 and next Y wins only 2/9 pairs. The two summaries answer different questions. The small control losses are measured observations, with no proven cause or statistical significance claim. Saved IR/assembly does not establish identical live code placement.

## Direct/helper comparison

Each ratio divides helper time by its matching direct time; above one means the helper form is slower. The algorithms include the same complete allocation, append, eight-pass guarded scan and reclamation policy within each engine.

| Pair | Engine | Helper/direct ratio of medians | Paired median | Paired range |
| --- | --- | ---: | ---: | ---: |
| string_scan_helper / string_scan | Previous Y | 122.90450 | 125.10864 | 115.38845–135.44719 |
| string_scan_helper / string_scan | Next Y | 0.94880 | 0.95096 | 0.62046–1.20565 |
| string_scan_helper / string_scan | C# | 1.00815 | 0.98438 | 0.86225–1.09347 |
| vec_scan_append_helper / vec_scan_append | Previous Y | 74.44458 | 74.72225 | 63.61324–80.50856 |
| vec_scan_append_helper / vec_scan_append | Next Y | 0.95699 | 0.95592 | 0.79047–1.40260 |
| vec_scan_append_helper / vec_scan_append | C# | 1.00967 | 1.00812 | 0.87084–1.47744 |

Next Y restores helpers to approximately the direct workload cost. Direct batches precede helper batches in every process. Cache, allocator, frequency and GC state can contribute to these descriptive ratios; the paired ranges cross one for both next-Y pairs. No inherent helper advantage is established.

## Preparation and compilation phases

Five small-input cold processes per engine use n=64, Fibonacci n=10 and two bulk appends of a 32-character chunk. Both Y settings compile the identical unit: fifteen kernels plus three source helpers, with eighteen checked adapters. Historical thirteen-kernel units contained fourteen source functions; their cold totals are not directly comparable.

| Cold preparation stage | Previous Y ms | Next Y ms | C# ms |
| --- | ---: | ---: | ---: |
| Instrumented compilation | 111.035 | 119.045 | — |
| Profile collection | 1.481 | 1.206 | — |
| Snapshot | 0.005 | 0.006 | — |
| Optimized recompilation (outer timer) | 190.761 | 221.704 | — |
| Final Y compilation / C# IL preparation | 190.699 | 221.637 | 2.999 |
| Full preparation | 306.054 | 342.875 | — |
| First-call set | 0.056 | 0.018 | 1.360 |
| Process launch to JSON | 308.367 | 345.189 | 237.968 |

C# prebuilt-IL method preparation is 2.999 ms; its source-to-IL build is outside that timer. Y starts with source parsing. These preparation measurements cover different compilation stages; the launch-to-JSON timer additionally includes runtime startup and formatting. A first-call set contains one call to every selected kernel.

Full-size preparation medians across the nine warm processes:

| Stage | Previous Y ms | Next Y ms |
| --- | ---: | ---: |
| Instrumented compilation | 111.615 | 120.294 |
| Profile collection | 576.182 | 514.995 |
| Snapshot | 0.017 | 0.019 |
| Optimized recompilation (outer timer) | 194.678 | 223.483 |
| Full preparation | 880.202 | 855.907 |

Full-size preparation decreases 2.76% despite slower compilation, alongside lower aggregate training time. Training uses the same distribution as measurement and the first twelve of its thirty-two seeds; there is no held-out evaluation. Collection includes native calls and storing their outputs. Full preparation also charges training-buffer setup, profile/result formatting, lookups and trainer disposal. These costs are excluded from steady-state batches but preserved here.

Each compilation has disjoint integer-nanosecond intervals that sum exactly to its `total_ns` and agree with `compile_ns` (or `instrumented_compile_ns`). The outer `optimized_recompile_ns` also includes call-return and source-AST cleanup after the inner timer. Separate phase/stage medians need not sum to the median total.

### Warm compilation phase medians (ms)

| Phase | Previous instrumented | Next instrumented | Previous profiled | Next profiled |
| --- | ---: | ---: | ---: | ---: |
| parse | 0.374 | 0.369 | 0.419 | 0.427 |
| checks | 0.243 | 0.240 | 0.227 | 0.225 |
| lowering | 0.856 | 0.873 | 0.846 | 0.854 |
| llvm_setup | 4.490 | 4.247 | 0.471 | 0.464 |
| ir_parse | 0.759 | 0.808 | 0.719 | 0.775 |
| profile_setup | 0.189 | 0.202 | 0.126 | 0.133 |
| verification | 0.148 | 0.164 | 0.136 | 0.145 |
| optimization | 60.714 | 65.175 | 107.001 | 123.450 |
| ir_capture | 0.969 | 1.069 | 2.445 | 2.890 |
| symbol_resolution | 0.030 | 0.030 | 0.028 | 0.027 |
| materialization | 42.322 | 44.219 | 82.028 | 93.848 |
| other | 0.146 | 0.150 | 0.157 | 0.163 |
| total | 111.615 | 120.294 | 194.612 | 223.413 |

### Cold compilation phase medians (ms)

| Phase | Previous instrumented | Next instrumented | Previous profiled | Next profiled |
| --- | ---: | ---: | ---: | ---: |
| parse | 0.373 | 0.391 | 0.334 | 0.340 |
| checks | 0.302 | 0.304 | 0.219 | 0.223 |
| lowering | 0.847 | 0.885 | 0.834 | 0.848 |
| llvm_setup | 4.337 | 4.436 | 0.442 | 0.439 |
| ir_parse | 0.784 | 0.805 | 0.695 | 0.713 |
| profile_setup | 0.188 | 0.199 | 0.124 | 0.128 |
| verification | 0.150 | 0.152 | 0.127 | 0.132 |
| optimization | 60.749 | 66.293 | 105.460 | 123.426 |
| ir_capture | 0.946 | 1.070 | 2.430 | 2.860 |
| symbol_resolution | 0.030 | 0.030 | 0.027 | 0.026 |
| materialization | 41.801 | 44.315 | 79.871 | 92.180 |
| other | 0.145 | 0.146 | 0.154 | 0.154 |
| total | 111.035 | 119.045 | 190.699 | 221.637 |

The next cold profiled compile spends median per-sample shares of **55.51% in LLVM optimization** and **41.82% in ORC materialization**. Their phase medians are 123.426 and 92.180 ms of a 221.637 ms total. Parsing, semantic checks and lowering together have a median of about 1.41 ms. The measured next bottleneck is the LLVM pipeline, followed by eager native materialization. Scalar helper analysis is included in lowering and is not separately timed.

Optimization includes LLVM `VerifyEach` and profile-selection work; the current data cannot separate their contributions. Materialization combines code generation, linking and lookup of every public function and adapter. Next work should first measure pass costs and verification separately, then evaluate an explicit lower-cost initial tier and lazy entrypoint/adapter materialization against full-result gates and measured amortization. This run does not demonstrate that removing verification or adding tiering would preserve correctness or improve total application latency.

## Optimization and profile audit

Both Y arms use the final compiler, LLVM O3 and the native host CPU/features. Rotate recognition, local query/ASCII/append/copy lowering and compact adapters are enabled; natural-loop control weights are excluded. **Only `optimize_helper_effects` changes.** Previous Y discards local-object proofs around source calls; next Y preserves them only for conservatively proved scalar helpers. No LLVM purity/memory attributes are asserted. Both arms compile instrumented code, collect actual profiles and recompile. They are not historical compiler binaries.

Warm previous workers record 130,060,270 outcomes across 97 raw sites and apply weights to 72 sites; next records 140,308,654 outcomes across 111 sites and applies 88. Cold counts are 288,407 and 328,727. Counts/fingerprints repeat deterministically within each arm. Different lowered control flow explains differing site sets; raw loop-control counts remain recorded even when their weights are excluded.

Saved final IR has no instrumentation probes. LLVM inlines the ordinary scalar helpers in both Y arms. Previous helper kernels retain String/Vec query callbacks and opaque append paths; next uses guarded native reads and spare-capacity writes, as the direct kernels already do. Allocation, growth-fallback and free callbacks remain. [IR audit](benchmark_data/cpu_jit_helper_effects/verification/ir-audit.json) records call inventories and structural checks. The saved `.s` files are `llc` reconstructions from final IR, not dumped ORC machine code.

The final review also found that a scalar source definition using a block-pointer intrinsic name could be certified while call dispatch actually emitted an external memory store. The proof now excludes six special load/block-pointer names. Executable O0/O3 and AST collision regressions failed before the guard and pass after it. No private-object corruption was demonstrated. Existing `load` duplicate argument evaluation and block-pointer override behavior remain documented limitations; this guard conservatively rejects their effect summaries.

## Runtime representation and GC

| C# workload | Median allocated bytes per 32-call batch | Gen0/Gen1/Gen2 count deltas summed across nine batches |
| --- | ---: | ---: |
| string_scan | 1575936 | 0/0/0 |
| vec_scan_append | 1059584 | 0/0/0 |
| vec_dynamic_byte | 1059584 | 0/0/0 |
| vec_dynamic_i64 | 8398592 | 9/9/9 |
| string_bulk_append | 1582592 | 0/0/0 |
| string_scan_helper | 1575936 | 0/0/0 |
| vec_scan_append_helper | 1059584 | 0/0/0 |

Every I64-vector batch reports `[1,1,1]`. These are generation-count deltas: a collection covering all generations contributes to all three counters; they do not mean three independent collections. Allocation and collections inside a batch are charged, while C# reclamation deferred past that batch is uncharged. Y allocates, grows and explicitly frees its containers inside every timed call.

Y Strings store flat bytes; C# `StringBuilder` uses UTF-16. Y Vec, `List<byte>` and `List<long>` have different layouts, growth policies and allocators. ASCII equates the tested character values, not arbitrary Unicode semantics. `List<byte>` stores bytes, not UTF-16. The C# comparison uses Release .NET 8, `NoInlining` kernel entries and ordinary scalar helpers under default inlining policy. Tiering and ReadyToRun are explicitly disabled using both DOTNET/COMPlus settings; dynamic PGO is disabled with tiering. Y receives explicit measured PGO. These are equivalent-algorithm comparisons with different runtime and optimization policies, not isolated code-generation measurements. .NET 10/dynamic-PGO and matched raw-buffer controls remain future work.

## Inputs, verification and durable evidence

Full-size inputs: integer/unsigned/short-circuit 250,000 rounds; Fibonacci 25/26; recurrence 1,000,000 steps; indexed memory 262,144 operations on 65,536 I64 elements; dot 262,157 products on two 65,536-element F64 arrays; binary search 20,000 queries on 65,536 sorted elements; String/Vec/dynamic vectors 16,384 elements; bulk String 512 appends of 32 characters. Object workloads use eight scans; String and byte-Vec workloads deliberately read invalid indices returning zero, while the I64 vector scans valid indices. Host input-array/output-buffer initialization, hashing, validation and JSON formatting are outside batch timers. Object construction, growth and reclamation occur inside each kernel timer.

The final gate passed **318 selected Rust tests**, then rebuilt `liby` and passed **all 16 Python CPU JIT tests**. [Verification logs](benchmark_data/cpu_jit_helper_effects/verification/verification.json) and before/after intrinsic regression logs are retained. The deterministic differential gate uses independent full-result oracles at O0/O3 across four configurations; its bounded valid templates are regression coverage, not a proof for arbitrary programs.

The independent [measurement audit](benchmark_data/cpu_jit_helper_effects/audit.json) passed 107,015 checks; the separate [oracle audit](benchmark_data/cpu_jit_helper_effects/verification/oracle-audit.json) passed 19,493 scalar comparisons. Neither imports the runner. They check all 27 warm and 15 cold records, stdout/individual/aggregate consistency, every retained first/timed/training output, memory checksums, side-effect counters, actual profiles, optimization settings, integer phase accounting and recomputed summaries. There are 480 timed outputs per warm worker and 180 training outputs per Y worker.

All 33 runner-listed sources, two worker binaries, 107 frozen source files and archive members, 187 runtime identities and 114 historical evidence/report files match their hashes. The rebuilt Python library was explicitly identified and hashed. [Frozen sources](benchmark_data/cpu_jit_helper_effects/verification/frozen-sources.tar.gz), [final hashes/configuration](benchmark_data/cpu_jit_helper_effects/verification/final-hashes.json), the .NET runtimeconfig/deps, all samples, training references/profiles, process outputs, build logs and both IR/assembly files are durable under [benchmark_data/cpu_jit_helper_effects](benchmark_data/cpu_jit_helper_effects/metadata.json). The broader snapshot covers the local source tree and included resources; it is not a hermetic image of the complete system/SDK/Cargo cache. Exact worker library mappings were not sampled during timing; the LLVM identity was probed with Python CPUJit under the same discovery environment.

A first complete full-size run finished before the intrinsic-name review concluded. It is retained with its pre-guard source snapshot under [superseded evidence](benchmark_data/cpu_jit_helper_effects/verification/superseded/superseded.json). This report uses only the new final-source run, rebuilt without `--skip-build`. Selection of the new run followed a correctness guard change, not timing selection. All historical benchmark reports/numbers and `CODEX_SESSION_HANDOFF.md` remain unchanged.

Benchmark checksums can collide and do not expose complete String/Vec contents or lifetimes. Float comparison uses finite values and absolute/relative tolerances `1e-12`, rather than bitwise equality. Side-effect records are totals, not full traces. Discarded standardized warmup results cannot be independently audited. Differential tests separately check complete buffers, object contents/freeing and IEEE results for bounded scenarios. A synthetic single-machine suite with these controls cannot establish a general language ranking.

Reproduce with a fresh output directory:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite helpers --optimization-stage helper-effects --compare-optimizations \
  --repeats 9 --cold-repeats 5 --calls 32 --warmup 12 --profile-warmups 12 \
  --cpu 0 --output build_artifacts/new-helper-effects-run
```

Rerun the retained independent audits after timing finishes (against the retained run and current matching binaries/sources):

```sh
python3 docs/benchmark_data/cpu_jit_helper_effects/verification/audit_measurement.py \
  docs/benchmark_data/cpu_jit_helper_effects --output /tmp/y-helper-measurement-audit.json
python3 docs/benchmark_data/cpu_jit_helper_effects/verification/audit_oracles.py \
  docs/benchmark_data/cpu_jit_helper_effects --output /tmp/y-helper-oracle-audit.json
```

The [benchmark guide](../benchmarks/cpu_jit/README.md), [copy report](cpu_jit_benchmarks_runtime_copies.md) and [adapter report](cpu_jit_benchmarks_adapters.md) retain their original controls and historical evidence.
