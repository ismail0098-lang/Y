# CPU JIT training-tier results — 2026-10-08

**Explicit O1 training reduces median full preparation 2.83% and cold preparation 6.60%, with all nine warm and five cold pairs improving. Instrumented compilation falls about 20%.** Final execution remains O3.

This comparison changes only the temporary instrumented IR tier from inherited O3 to explicit O1. Final profiled IR and machine-code optimization remain O3; per-pass verification and atomic counters remain enabled. All retained training outputs, exact profiles and saved final IR match between arms. The Rust option is explicit; **default training still inherits the requested IR level**.

## Measured preparation

Times are medians in milliseconds. Signed paired saving is training O3 minus training O1; positive favors O1. Differences of separate medians are distinct from median paired differences; medians need not sum. Ranges are observed sample ranges, not confidence intervals.

| Scope / preparation stage | Training O3 ms | Training O1 ms | Paired saved median ms | Paired range ms | O1 gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented_compile | 113.130 | 90.504 | 22.419 | 21.557–24.617 | 9/9 |
| warm / profile_collection | 499.328 | 499.266 | 0.290 | -2.210–1.785 | 7/9 |
| warm / profile_snapshot | 0.013 | 0.014 | 0.002 | -0.005–0.005 | 7/9 |
| warm / optimized_recompile | 213.160 | 212.780 | 0.447 | -3.127–2.369 | 6/9 |
| warm / prepare | 826.790 | 803.418 | 24.314 | 16.206–26.868 | 9/9 |
| cold / instrumented_compile | 114.817 | 91.911 | 23.029 | 22.617–25.472 | 5/5 |
| cold / profile_collection | 1.194 | 1.196 | -0.001 | -0.003–0.007 | 2/5 |
| cold / profile_snapshot | 0.006 | 0.005 | 0.001 | -0.001–0.002 | 3/5 |
| cold / optimized_recompile | 212.999 | 214.536 | 0.067 | -1.633–2.238 | 3/5 |
| cold / prepare | 330.061 | 308.283 | 24.532 | 21.302–25.285 | 5/5 |

Warm preparation changes **-2.83%** by separate medians (826.790→803.418 ms). Its paired median saving is **24.314 ms**, with 9/9 pairs improving.

Cold preparation changes **-6.60%** by separate medians (330.061→308.283 ms). Its paired median saving is **24.532 ms**, with 5/5 pairs improving.

These are measured policy differences for this synthetic suite on one machine. Changing the temporary tier does not establish a native execution improvement. Lower training tiers can slow training on other programs enough to erase compilation savings; preserve full preparation, not only compile time.

## Controls and implementation

Run start: `2026-10-07T22:35:00.827107+00:00`; AMD Ryzen 9 9950X 16-Core Processor, Linux x86-64, logical CPU 0; LLVM 23.1.1; .NET SDK 8.0.425/runtime 8.0.31. Nine rotating warm process triples and five cold triples execute sequentially, with no concurrent agent builds/tests/CPU audits during timing. Each warm worker makes 12 standardized warmups and 32 timed calls per workload; Y additionally makes 12 measured training calls per workload. Other host activity/frequency remain uncontrolled.

`JitOptions.training_opt_level: Option<u8>` defaults to `None`; explicit 0–3 overrides only the instrumented module’s IR-analysis target and default IR pipeline. Ordinary and profile-use modes retain `opt_level`. ORC codegen still follows the separate codegen override or base `opt_level`, even for training. Invalid overrides fail before semantic checks/LLVM in every mode. Cache v9 includes presence/value. The current C/Python compilation entrypoints retain inherited policy.

Both measured arms use base IR O3, native codegen O3, VerifyEach true, mandatory full-module checks and eager whole-module materialization. Both enable rotate, runtime query/mutation/copy, compact adapters and scalar helper effects, and exclude natural-loop control weights. Per-entry `compilation_settings` records actual IR target/pipeline, native settings and verification policy. Previous training override is null; next is 1. Final options are otherwise identical.

Fingerprint/site identities come from original lowered IR before optimization. The source lowering is unchanged. All original conditional branches retain aligned monotonic atomic outcome increments; contradictory effect attributes are stripped as before. Lower IR tier does not waive profile compatibility, verification, thread safety or unseen edges. Final profile-use sessions have no training counters. Training and recompilation remain explicit and eager.

## Compilation phases and next bottleneck

| Scope / kind / phase | Training O3 ms | Training O1 ms | Paired saved median ms | Paired range ms | O1 gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented / pipeline | 62.536 | 41.718 | 20.934 | 20.680–22.338 | 9/9 |
| warm / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/9 |
| warm / instrumented / verification | 0.265 | 0.252 | 0.014 | 0.000–0.018 | 9/9 |
| warm / instrumented / materialization | 42.509 | 40.883 | 1.627 | 0.741–1.973 | 9/9 |
| warm / instrumented / total | 113.130 | 90.504 | 22.419 | 21.557–24.617 | 9/9 |
| warm / profiled / pipeline | 115.842 | 115.332 | 0.411 | -1.217–1.942 | 7/9 |
| warm / profiled / profile_selection | 2.852 | 2.848 | 0.001 | -0.025–0.030 | 5/9 |
| warm / profiled / verification | 0.825 | 0.817 | 0.007 | 0.001–0.013 | 9/9 |
| warm / profiled / materialization | 87.833 | 87.993 | -0.098 | -1.822–0.479 | 3/9 |
| warm / profiled / total | 213.103 | 212.721 | 0.450 | -3.127–2.373 | 6/9 |
| cold / instrumented / pipeline | 62.938 | 41.709 | 21.315 | 20.880–21.695 | 5/5 |
| cold / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/5 |
| cold / instrumented / verification | 0.269 | 0.253 | 0.017 | 0.007–0.022 | 5/5 |
| cold / instrumented / materialization | 42.888 | 40.937 | 2.273 | 1.949–2.930 | 5/5 |
| cold / instrumented / total | 114.817 | 91.911 | 23.029 | 22.617–25.472 | 5/5 |
| cold / profiled / pipeline | 115.382 | 115.765 | 0.217 | -0.405–1.303 | 3/5 |
| cold / profiled / profile_selection | 2.863 | 2.886 | -0.011 | -0.095–0.012 | 1/5 |
| cold / profiled / verification | 0.816 | 0.813 | 0.002 | -0.016–0.013 | 3/5 |
| cold / profiled / materialization | 88.441 | 89.310 | 0.018 | -1.288–1.892 | 3/5 |
| cold / profiled / total | 212.932 | 214.472 | 0.057 | -1.632–2.239 | 3/5 |

Flat phases sum exactly to total. Nested optimization/materialization parents sum exactly to their primary phase; optional object-ready children partition first lookup and are already included. Before-object work includes native emission and preceding ORC work; after-object includes observer overhead/linking/lookup. Neither is an exclusive backend or linker timer. Timing snapshots exclude later execution/getters; object bytes include file metadata, not executable memory alone.

With O1 training, cold final profiled compilation still spends 55.36% median per-sample share in optimization and 41.64% in materialization. These final phases remain the next overall targets. Training contains millions of atomic observations; lowering the training IR tier does not remove their runtime cost.

A separate copied-source diagnostic retained pre-pipeline module snapshots and external `opt -time-passes` reports. Its profiled O3 diagnostic assigns about 29.8% of pass wall time to loop unrolling, 18.3% to InstCombine and 13.2% to loop vectorization. These prioritize future investigation; they are not timings from the measured JIT. The diagnostic worker adds IR capture I/O, links the preceding library and runs fewer calls/repeats; external opt is unpinned and has its own pass-manager/verification accounting. Its smoke timings are not validated performance claims. No pass is removed from final O3 here.

## Native execution against C# — all fifteen workloads

Both Y columns execute final O3. Times are median milliseconds per call; C#/Y ratios divide per-arm medians. Pair gains/ranges are retained even when separate medians disagree. Identical saved final IR does not establish identical native addresses, placement, allocator/cache/frequency state or live object contents. Do not attribute these native variations to a changed final IR policy.

| Workload | O3-trained Y | O1-trained Y | C# | O1-trained time change | C# / O1-trained Y | O1-training pair gains |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.585310 | 0.584976 | 1.035527 | -0.057% | 1.77020× | 4/9 |
| recursive_fib | 0.147608 | 0.147657 | 0.287542 | +0.033% | 1.94736× | 3/9 |
| float_recurrence | 1.079422 | 1.078850 | 1.080136 | -0.053% | 1.00119× | 5/9 |
| indexed_memory | 0.106692 | 0.106648 | 0.168084 | -0.041% | 1.57607× | 3/9 |
| unsigned_mix | 0.449761 | 0.449606 | 0.450014 | -0.034% | 1.00091× | 4/9 |
| float_dot | 0.094643 | 0.094727 | 0.094841 | +0.089% | 1.00120× | 4/9 |
| short_circuit | 1.772103 | 1.764734 | 1.872388 | -0.416% | 1.06100× | 9/9 |
| binary_search | 1.418720 | 1.418729 | 1.415279 | +0.001% | 0.99757× | 4/9 |
| string_scan | 0.036967 | 0.037031 | 0.255393 | +0.173% | 6.89679× | 4/9 |
| vec_scan_append | 0.062866 | 0.062951 | 0.089949 | +0.134% | 1.42887× | 3/9 |
| vec_dynamic_byte | 0.062865 | 0.062863 | 0.096849 | -0.004% | 1.54063× | 6/9 |
| vec_dynamic_i64 | 0.039226 | 0.039299 | 0.107691 | +0.184% | 2.74033× | 4/9 |
| string_bulk_append | 0.028725 | 0.028737 | 0.254507 | +0.040% | 8.85654× | 4/9 |
| string_scan_helper | 0.035733 | 0.035648 | 0.256345 | -0.236% | 7.19090× | 6/9 |
| vec_scan_append_helper | 0.060158 | 0.060164 | 0.091773 | +0.010% | 1.52540× | 5/9 |

| Workload | O3-trained / O1-trained paired median | Range | C# / O1-trained paired median | Range |
| --- | ---: | ---: | ---: | ---: |
| integer_branch | 0.99990× | 0.99814–1.00578 | 1.77020× | 1.76624–1.77168 |
| recursive_fib | 0.99962× | 0.99733–1.01130 | 1.94634× | 1.93912–1.95557 |
| float_recurrence | 1.00041× | 0.99368–1.00407 | 1.00121× | 0.99400–1.00311 |
| indexed_memory | 0.99866× | 0.99082–1.01723 | 1.57355× | 1.55568–1.58972 |
| unsigned_mix | 0.99988× | 0.99882–1.00186 | 1.00053× | 0.99805–1.00234 |
| float_dot | 0.99934× | 0.99105–1.00758 | 0.99907× | 0.99549–1.00710 |
| short_circuit | 1.00356× | 1.00085–1.00662 | 1.06100× | 1.05326–1.08638 |
| binary_search | 0.99986× | 0.98686–1.00755 | 0.99805× | 0.98484–1.00161 |
| string_scan | 0.99757× | 0.97197–1.02202 | 6.89408× | 6.65773–6.96699 |
| vec_scan_append | 0.99929× | 0.99252–1.01067 | 1.42361× | 1.41595–1.45639 |
| vec_dynamic_byte | 1.00095× | 0.99544–1.01028 | 1.53592× | 1.50800–1.60398 |
| vec_dynamic_i64 | 0.99994× | 0.99147–1.01451 | 2.74033× | 2.68072–2.82538 |
| string_bulk_append | 0.99941× | 0.99632–1.01295 | 8.86716× | 8.73280–9.15324 |
| string_scan_helper | 1.00597× | 0.99384–1.01723 | 7.15983× | 6.88490–7.50701 |
| vec_scan_append_helper | 1.00047× | 0.99791–1.01583 | 1.52411× | 1.49220–1.60817 |

Retained native median regressions: `recursive_fib` +0.033%; `float_dot` +0.089%; `binary_search` +0.001%; `string_scan` +0.173%; `vec_scan_append` +0.134%; `vec_dynamic_i64` +0.184%; `string_bulk_append` +0.040%; `vec_scan_append_helper` +0.010%.

## PGO, runtime representation and preparation scope

Y uses explicit measured PGO. Training overlaps the first 12 of 32 measurement seeds, so evaluation is same-distribution rather than held out. Warm profiles record 140,308,654 outcomes across 111 original sites, applying 88; cold records 328,727. Both arms retain exact profiles and full training output streams. The raw original lowered IR is not archived for independent fingerprint regeneration.

C# Release .NET 8 has tiering/dynamic PGO and ReadyToRun disabled. Kernel entries use NoInlining; ordinary scalar helpers keep normal inlining. Y byte Strings and Vec differ from UTF-16 StringBuilder, List<byte> and List<long> in representation, growth and allocator policy. ASCII matches these values, not arbitrary Unicode. Y allocation/growth/free are timed. C# in-batch GC is timed; deferred reclamation is excluded. Generation-count deltas are not independent collection counts. The raw C# allocation/GC metadata remains in every batch record.

| Cold stage | O3-trained Y ms | O1-trained Y ms | C# ms |
| --- | ---: | ---: | ---: |
| Final compilation / IL preparation | 212.932 | 214.472 | 2.921 |
| First-call set | 0.018 | 0.018 | 1.333 |
| Launch to JSON | 332.341 | 310.446 | 228.924 |

Y preparation starts from source and charges instrumented compilation, measured training/return storage, snapshots/profile formatting, final recompilation, input setup and trainer disposal. The saved first warm pair also copies instrumented IR within preparation; file writes follow. C# starts from prebuilt IL and uses PrepareMethod; source-to-IL build is outside that timer. Runtime startup/formatting belong launch-to-JSON. Neither comparison establishes a general language ranking.

Full input dimensions and native dispatch/output storage are unchanged from the [preceding codegen report](cpu_jit_benchmarks_codegen.md). Host input construction, reset/hashing/validation/JSON formatting are outside batch timers. Checksums can collide; finite floating outputs use absolute/relative tolerance 1e-12. Aggregate counters do not reveal chronological effects; discarded standardized warmups cannot be audited. Bounded differential oracles do not prove arbitrary programs.

## Verification and durable evidence

**325 selected Rust tests and 18 Python CPU JIT tests passed after rebuilding liby.** The new cross-tier gate covers training levels 0–3 with inherited native O3 and explicit native O1, ordinary/instrumented/profiled modes, complete memory/object contents, exact profiles, independent branch-count/recursion/IEEE oracles, unseen inputs and final IR identity. The existing concurrent-native-call gate now covers inherited and O1 training, checking every atomic observation. The generated O0/O3 differential gate remains sixteen transform/verification/codegen configurations.

Independent [measurement audit](benchmark_data/cpu_jit_training_tier/audit.json): **139,423 checks passed**; independent [oracle audit](benchmark_data/cpu_jit_training_tier/verification/oracle-audit.json): **19,493 scalar checks passed**. Neither imports the runner. The durable-copy audit passes 139,725 checks. All 33 runner/110 frozen-source members, two workers, rebuilt liby, 187 runtime identities, 841 historical report/data files and the session handoff match. All raw pairs/losses, settings, output/training streams, source archive, exact binaries, reviews and logs are retained under [durable evidence](benchmark_data/cpu_jit_training_tier/README.md). The evidence manifest seals every retained file except itself; publication.json hashes docs.

Runtime identity was probed from fresh Python CPUJit under matching discovery environment; exact live worker mappings were not sampled during timing. The snapshot is not a hermetic SDK/system/Cargo image. Saved final IR equality is checked on the first warm pair, not every module. Instrumented IR is expected to differ. Assembly is an llc reconstruction at native O3/PIC/native CPU/small code model, while live ORC requests JITDefault. Actual emitted object bytes are counted but not retained or hashed. Historical reports/evidence and `CODEX_SESSION_HANDOFF.md` remain unchanged; no commit was made. The first oracle audit passed all scalar outputs but retained the preceding stage name in one metadata guard. Its failure, script and correction are retained; no timed run was replaced.

Reproduce in a fresh directory with no concurrent heavy jobs:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite helpers --optimization-stage training-tier --compare-optimizations \
  --repeats 9 --cold-repeats 5 --calls 32 --warmup 12 --profile-warmups 12 \
  --cpu 0 --output build_artifacts/new-training-tier-run
```

Rerun retained auditors only with matching frozen sources/binaries/runtime manifests; original metadata paths identify the actual working run.
