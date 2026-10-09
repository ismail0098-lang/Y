# CPU JIT final loop-unrolling tradeoff — 2026-10-08

**Disabling final loop unrolling reduces median warm full preparation 17.36% and changes cold preparation -45.26%. The native fifteen-workload bundle rises 1.60%.** This is an opt-in compilation-latency tradeoff; the production default remains true.

This comparison changes only requested `JitOptions.final_loop_unrolling` from true to false. True leaves LLVM’s existing default tuning untouched; explicit false requests `LLVMPassBuilderOptionsSetLoopUnrolling(..., 0)` in ordinary/profile-use mode. Instrument mode retains inherited loop unrolling in both arms. Both arms use explicit training IR O1, final IR O3, native O3, VerifyEach enabled and eager compilation. Instrumented IR and reconstructed instrumented assembly match byte for byte; final IR and reconstructed final assembly change. Exact profiles, full returned training/results and checked memory contents match. Native optimization stays O3, while changed final IR produces changed code. No general execution-speed improvement is claimed.

## Measured preparation

Times are medians in milliseconds. Positive signed paired saving is default unrolling minus disabled final unrolling, so positive favors disabled unrolling. Differences of separate medians and median paired differences are distinct; medians need not sum. Ranges are observed sample ranges, not confidence intervals.

| Scope / preparation stage | Default unroll ms | Disabled final unroll ms | Paired saved median ms | Paired range ms | Disabled gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented_compile | 93.156 | 93.189 | -0.388 | -1.766–1.240 | 4/9 |
| warm / profile_collection | 505.473 | 505.739 | 0.165 | -1.314–1.845 | 5/9 |
| warm / profile_snapshot | 0.017 | 0.017 | -0.001 | -0.002–0.002 | 3/9 |
| warm / optimized_recompile | 221.704 | 78.535 | 142.885 | 141.684–144.990 | 9/9 |
| warm / prepare | 820.972 | 678.492 | 142.735 | 139.086–145.992 | 9/9 |
| cold / instrumented_compile | 94.411 | 93.983 | 0.576 | -1.563–1.806 | 3/5 |
| cold / profile_collection | 1.209 | 1.217 | 0.000 | -0.012–0.005 | 3/5 |
| cold / profile_snapshot | 0.006 | 0.006 | -0.000 | -0.000–0.001 | 2/5 |
| cold / optimized_recompile | 220.672 | 78.039 | 142.704 | 141.958–144.447 | 5/5 |
| cold / prepare | 316.900 | 173.461 | 143.306 | 140.917–144.964 | 5/5 |

Warm preparation changes **-17.355%** by separate medians (820.971728→678.491906 ms). Its paired median saving is **142.734747 ms**, with 9/9 pairs improving.

Cold preparation changes **-45.263%** by separate medians (316.899973→173.460989 ms). Its paired median saving is **143.305622 ms**, with 5/5 pairs improving.

Warm profile execution changes +0.053% by separate medians, with 5/9 pairs improving and signed paired median saving 0.165409 ms. Its median per-sample share of optimized full preparation is 74.57%. Training policy and saved training code are identical; this comparison does not claim a profiling-execution improvement. Every loss remains in raw records and tables.

## Controls and implementation

Run start: `2026-10-08T16:48:08.426870+00:00`; AMD Ryzen 9 9950X 16-Core Processor, Linux x86-64, logical CPU 0; LLVM 23.1.1; fixed .NET 8.0.31 baseline. 9 rotating warm process triples and 5 cold triples execute sequentially without concurrent agent builds/tests/CPU audits during timing. Each warm worker makes 12 standardized warmups and 32 timed calls per workload; Y additionally makes 12 measured training calls. Other host activity and frequency remain uncontrolled. Versions, discovery environment and actual commands are retained.

`final_loop_unrolling: bool` defaults true, including existing C/Python entrypoints. False is a LLVM PassBuilder tuning request before the default IR pipeline, not custom pass removal or a promise that vectorization and every other loop transform disappear. Cache v12 includes the requested flag. Instrument compilation always inherits loop unrolling; ordinary/profile-use compilation honors the requested setting. The new requested flag is true/false in every compilation/worker record, while actual `ir_loop_unrolling` is true for both Instrument arms and follows requested policy for final IR.

Both requested edge-counter flags are false in both measured arms. Base/final IR O3, explicit training O1, native O3, VerifyEach, mandatory full-module checks and eager whole-module materialization remain enabled. Rotate, runtime query/mutation/copy, compact adapters and scalar helper effects remain enabled; natural-loop control weights are excluded. Per-entry settings retain requested and effective IR/native/unrolling policy.

Profiling still uses separate atomic true/false outcomes for each original lowered-IR site, with original fingerprint/site identities and wraparound after 2^64 observations. Owned relaxed snapshots remain atomic per counter; concurrent snapshots need not represent one instant across the array. Final sessions have no training counters. Training, snapshots and recompilation remain explicit, with eager public/adapter lookups.

## Compilation phases and object evidence

| Scope / kind / phase | Default unroll ms | Disabled final unroll ms | Paired saved median ms | Paired range ms | Disabled gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented / pipeline | 42.708 | 42.719 | 0.032 | -0.804–0.825 | 5/9 |
| warm / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/9 |
| warm / instrumented / verification | 0.257 | 0.259 | -0.002 | -0.035–0.017 | 4/9 |
| warm / instrumented / materialization | 42.218 | 42.352 | -0.275 | -1.642–0.391 | 4/9 |
| warm / instrumented / total | 93.156 | 93.189 | -0.388 | -1.766–1.240 | 4/9 |
| warm / profiled / pipeline | 119.301 | 49.832 | 69.552 | 68.462–70.713 | 9/9 |
| warm / profiled / profile_selection | 2.920 | 0.591 | 2.334 | 2.292–2.355 | 9/9 |
| warm / profiled / verification | 0.838 | 0.284 | 0.550 | 0.534–0.569 | 9/9 |
| warm / profiled / materialization | 92.407 | 24.265 | 68.036 | 67.539–69.815 | 9/9 |
| warm / profiled / total | 221.631 | 78.470 | 142.880 | 141.687–144.981 | 9/9 |
| cold / instrumented / pipeline | 42.816 | 42.723 | 0.163 | -1.335–0.451 | 3/5 |
| cold / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/5 |
| cold / instrumented / verification | 0.261 | 0.263 | -0.001 | -0.008–0.002 | 2/5 |
| cold / instrumented / materialization | 42.197 | 42.035 | 0.341 | -0.007–0.474 | 4/5 |
| cold / instrumented / total | 94.411 | 93.983 | 0.576 | -1.563–1.806 | 3/5 |
| cold / profiled / pipeline | 118.630 | 49.539 | 69.318 | 68.946–70.160 | 5/5 |
| cold / profiled / profile_selection | 2.922 | 0.594 | 2.328 | 2.320–2.357 | 5/5 |
| cold / profiled / verification | 0.828 | 0.275 | 0.553 | 0.548–0.560 | 5/5 |
| cold / profiled / materialization | 92.011 | 24.274 | 67.935 | 67.737–68.918 | 5/5 |
| cold / profiled / total | 220.603 | 77.975 | 142.703 | 141.948–144.438 | 5/5 |

Retained preparation/compilation metrics with separate-median or paired-median losses (22 metrics): `warm/preparation/instrumented_compile` separate +0.036%, paired saved -0.388205 ms; `warm/preparation/profile_collection` separate +0.053%, paired saved +0.165409 ms; `warm/preparation/profile_snapshot` separate +2.719%, paired saved -0.000820 ms; `warm/instrumented/pipeline` separate +0.026%, paired saved +0.032040 ms; `warm/instrumented/optimization_other` separate -9.718%, paired saved -0.000070 ms; `warm/instrumented/optimization` separate +0.025%, paired saved +0.031460 ms; `warm/instrumented/verification` separate +0.822%, paired saved -0.001969 ms; `warm/instrumented/materialization` separate +0.317%, paired saved -0.275373 ms; `warm/instrumented/total` separate +0.036%, paired saved -0.388205 ms; `warm/instrumented/first_lookup` separate +0.320%, paired saved -0.273964 ms; `warm/instrumented/remaining_function_lookups` separate +0.823%, paired saved +0.000019 ms; `warm/instrumented/materialization_other` separate -5.286%, paired saved -0.000130 ms; `warm/instrumented/first_lookup_before_object` separate +0.301%, paired saved -0.289914 ms; `warm/instrumented/first_lookup_after_object` separate +2.783%, paired saved +0.002301 ms; `cold/preparation/profile_collection` separate +0.698%, paired saved +0.000140 ms; `cold/preparation/profile_snapshot` separate -3.339%, paired saved -0.000071 ms; `cold/instrumented/verification` separate +0.586%, paired saved -0.001270 ms; `cold/instrumented/remaining_function_lookups` separate +1.194%, paired saved -0.000630 ms; `cold/instrumented/profile_lookup` separate +1.961%, paired saved -0.000010 ms; `cold/instrumented/first_lookup_after_object` separate +0.522%, paired saved -0.000890 ms; `cold/profiled/optimization_other` separate +32.298%, paired saved -0.000210 ms; `cold/profiled/materialization_other` separate +3.782%, paired saved -0.000430 ms.

Warm final total changes -64.594% (221.631410→78.470447 ms), with paired median saving 142.879585 ms.

Warm final pipeline changes -58.230% (119.301243→49.832220 ms), with paired median saving 69.552046 ms.

Warm final materialization changes -73.741% (92.407272→24.265228 ms), with paired median saving 68.035556 ms.

The measured default-pipeline reduction includes VerifyEach and changed downstream IR work. It is not an exclusive timing of LLVM’s unroll pass. Flat phases sum exactly to total; nested optimization/materialization parents sum to their primary phase. Optional object-ready children partition first lookup and are already included. Before-object work includes native emission and preceding ORC work; after-object includes observer overhead, linking and lookup. Neither is an exclusive backend or linker timer. Timing snapshots exclude later execution/getters.

With disabled final unrolling, cold final compilation spends 64.28% median per-sample share in optimization and 31.06% in materialization.

| Scope / compilation | Default observed object bytes | Disabled final unroll observed object bytes |
| --- | ---: | ---: |
| warm / instrumented | 21,216 | 21,216 |
| warm / profiled | 44,424 | 13,040 |
| cold / instrumented | 21,216 | 21,216 |
| cold / profiled | 45,000 | 13,040 |

All 56 measured compilations observe one object and an eligible first-lookup split, totaling 56 callbacks, 2,016 public/adapter lookups and 28 profile-global lookups. These are actual live ORC object-file bytes, including file metadata, rather than executable-memory allocation. Object contents, placement and mappings are not retained or hashed. Smaller byte counts do not prove particular object contents.

| Saved/reconstructed artifact | Default bytes | Disabled final unroll bytes |
| --- | ---: | ---: |
| Instrumented IR | 206,077 | 206,077 |
| Final IR | 687,420 | 101,819 |
| Reconstructed instrumented assembly | 126,470 | 126,470 |
| Reconstructed final assembly | 376,345 | 74,430 |

Artifact identities refer to the saved first warm pair; every full/smoke pair retains exact profiles and output streams. Reconstructed assembly uses external llc O3/PIC/native CPU/small code model, while live ORC requests JITDefault. Its text sizes/static instruction counts are scoped diagnostics, not actual live JIT-code bytes or pass timings. Final IR and reconstructed assembly differ globally; some functions remain identical. Vectorized loop metadata remains in both final arms. No pass-exclusive or dynamic instruction-count cause is inferred from these artifacts.

## Native execution against C# — all fifteen workloads

Both Y arms execute native O3 generated from their differing final IR. Times are median milliseconds per call; C#/Y ratios divide separate medians. Pair gains/ranges retain process variability. Changed code and code size do not establish general native speed improvement; even unchanged functions can vary with placement/cache/frequency state.

| Workload | Default-unroll Y | Disabled-final-unroll Y | C# | Y time change | C# / disabled Y | Disabled pair gains |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.590449 | 0.589912 | 1.044688 | -0.091% | 1.77092× | 4/9 |
| recursive_fib | 0.149092 | 0.156256 | 0.290600 | +4.805% | 1.85977× | 0/9 |
| float_recurrence | 1.088593 | 1.088501 | 1.091245 | -0.008% | 1.00252× | 5/9 |
| indexed_memory | 0.107935 | 0.131017 | 0.169145 | +21.386% | 1.29101× | 0/9 |
| unsigned_mix | 0.453535 | 0.453586 | 0.453672 | +0.011% | 1.00019× | 5/9 |
| float_dot | 0.095890 | 0.095378 | 0.095731 | -0.534% | 1.00370× | 7/9 |
| short_circuit | 1.786096 | 1.778795 | 1.896965 | -0.409% | 1.06643× | 5/9 |
| binary_search | 1.434288 | 1.399916 | 1.425314 | -2.396% | 1.01814× | 9/9 |
| string_scan | 0.038018 | 0.057557 | 0.258948 | +51.395% | 4.49897× | 0/9 |
| vec_scan_append | 0.063548 | 0.070733 | 0.091163 | +11.306% | 1.28883× | 0/9 |
| vec_dynamic_byte | 0.063635 | 0.075311 | 0.099485 | +18.347% | 1.32100× | 0/9 |
| vec_dynamic_i64 | 0.039905 | 0.054973 | 0.109005 | +37.761% | 1.98288× | 0/9 |
| string_bulk_append | 0.029181 | 0.049844 | 0.257373 | +70.810% | 5.16357× | 0/9 |
| string_scan_helper | 0.036130 | 0.056939 | 0.253840 | +57.596% | 4.45814× | 0/9 |
| vec_scan_append_helper | 0.060972 | 0.070676 | 0.092865 | +15.915% | 1.31396× | 0/9 |

| Workload | Default / disabled paired median | Range | C# / disabled paired median | Range |
| --- | ---: | ---: | ---: | ---: |
| integer_branch | 0.99969× | 0.99101–1.00684 | 1.76680× | 1.75118–1.78441 |
| recursive_fib | 0.95417× | 0.95173–0.97015 | 1.86011× | 1.85479–2.17387 |
| float_recurrence | 1.00008× | 0.99879–1.00263 | 1.00377× | 0.99923–1.00523 |
| indexed_memory | 0.82274× | 0.79329–0.82652 | 1.28942× | 1.24324–1.29995 |
| unsigned_mix | 1.00061× | 0.99709–1.00202 | 0.99950× | 0.99813–1.00337 |
| float_dot | 1.00187× | 0.99955–1.01367 | 1.00564× | 0.99634–1.00698 |
| short_circuit | 1.00152× | 0.99822–1.01802 | 1.06911× | 1.05629–1.07723 |
| binary_search | 1.02555× | 1.01852–1.03497 | 1.01612× | 1.01057–1.03119 |
| string_scan | 0.65773× | 0.57478–0.91709 | 4.50017× | 3.88602–4.53737 |
| vec_scan_append | 0.90085× | 0.84266–0.91362 | 1.28783× | 1.20256–1.31585 |
| vec_dynamic_byte | 0.84565× | 0.83587–0.84668 | 1.32206× | 1.30093–1.34012 |
| vec_dynamic_i64 | 0.72691× | 0.64658–0.73473 | 1.98601× | 1.74653–2.02617 |
| string_bulk_append | 0.58479× | 0.56833–0.59259 | 5.15575× | 5.03064–5.20907 |
| string_scan_helper | 0.63588× | 0.56266–0.64369 | 4.46294× | 3.95985–4.68671 |
| vec_scan_append_helper | 0.86260× | 0.85157–0.90752 | 1.30426× | 1.28342–1.34032 |

Retained native median regressions (10 workloads): `recursive_fib` +4.805%; `indexed_memory` +21.386%; `unsigned_mix` +0.011%; `string_scan` +51.395%; `vec_scan_append` +11.306%; `vec_dynamic_byte` +18.347%; `vec_dynamic_i64` +37.761%; `string_bulk_append` +70.810%; `string_scan_helper` +57.596%; `vec_scan_append_helper` +15.915%.

One call to each workload costs median 6.039853→6.136379 ms (+1.598%). The paired median saving is -0.093931094 ms; 9/9 bundles are slower.

## Paired preparation-plus-use model

This arithmetic model charges one full warm preparation and N bundles, where one bundle is one call to each of the fifteen heterogeneous synthetic workloads. For every process pair, bundle cost is the sum of its per-workload batch-average ns/call. Each arm’s modeled cost is `prepare_ns + N * sum(ns_per_call)`. The auditor independently reconstructs every pair, signed saving, loss count and crossover from raw worker rows; it does not import the runner. These are extrapolations of measured costs, not additional timed N-bundle runs.

| Bundles N | Default modeled ms | Disabled modeled ms | Paired saved median ms | Paired range ms | Disabled gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 827.027 | 684.628 | 142.630 | 138.978–145.903 | 9/9 |
| 12 | 893.603 | 752.115 | 141.474 | 137.795–144.929 | 9/9 |
| 32 | 1014.330 | 874.549 | 139.733 | 135.644–143.157 | 9/9 |
| 100 | 1425.349 | 1292.130 | 133.472 | 128.330–137.603 | 9/9 |
| 1000 | 6861.075 | 6814.871 | 48.808 | 31.045–76.132 | 9/9 |

A positive applicable crossover requires both an initial preparation saving and a native bundle slowdown. Signed crossover is `preparation_saved_ns / native_slowdown_ns`; negative ratios remain retained, and zero native difference has no finite crossover. The table keeps every signed pair and its applicability rather than filtering losses.

| Repeat | Preparation saved ms | Native slowdown ms/bundle | Signed crossover bundles | Applicable |
| --- | ---: | ---: | ---: | ---: |
| 0 | 144.950854 | 0.110663781 | 1309.831025 | true |
| 1 | 142.137454 | 0.067543406 | 2104.386822 | true |
| 2 | 144.433085 | 0.068301469 | 2114.640983 | true |
| 3 | 142.739275 | 0.093931094 | 1519.616873 | true |
| 4 | 141.235121 | 0.110189781 | 1281.744272 | true |
| 5 | 142.734747 | 0.105049719 | 1358.735166 | true |
| 6 | 139.085618 | 0.107556625 | 1293.138549 | true |
| 7 | 142.301139 | 0.088292969 | 1611.692766 | true |
| 8 | 145.991517 | 0.088570406 | 1648.310346 | true |

All 9 positive applicable pair crossovers have median **1519.616873 bundles**, range **1281.744272–2114.640983**. The median of individual crossover ratios differs from dividing median paired preparation saving by median paired native loss. Every requested modeled N through 1000 still favors disabled unrolling in all nine pairs.

The model holds the current 32-call batch means and input distribution fixed. It excludes first-call and standardized warmup costs, and host setup/reset/hash/format work outside preparation/batch timers. Thermal/cache/frequency/input-state effects can change in longer real runs. Equal weighting of fifteen synthetic workloads is not an application call mixture. Different programs and call mixtures can favor either policy; the model is not a universal break-even recommendation or a general language ranking.

## PGO, representation and preparation scope

Training overlaps the first 12 of 32 measurement seeds, so evaluation is same-distribution rather than held out. Both arms retain exact original profiles and complete training streams; raw original lowered IR is not archived for independent fingerprint regeneration.

Warm profiles record 140,308,654 outcomes across 111 original sites, applying 88.

Cold profiles record 328,727 outcomes across 111 original sites, applying 88.

C# Release .NET 8 has tiering/dynamic PGO and ReadyToRun disabled; kernel entries use NoInlining and scalar helpers keep normal inlining. Y byte String/Vec layout, growth and allocation/free policy differ from C# UTF-16 StringBuilder/List/GC. ASCII matches these values, not arbitrary Unicode. Y allocation/growth/free and C# in-batch GC are timed; deferred managed reclamation is excluded. Generation deltas are not independent collection counts. Raw allocation/GC fields are retained.

| Cold stage | Default-unroll Y ms | Disabled-final-unroll Y ms | C# ms |
| --- | ---: | ---: | ---: |
| Final compilation / IL preparation | 220.603 | 77.975 | 2.947 |
| First-call set | 0.019 | 0.016 | 1.349 |
| Launch to JSON | 319.258 | 175.654 | 233.596 |

Y preparation starts from source and charges instrumented compilation, measured training/return storage, snapshots/profile formatting, final recompilation, input setup and trainer disposal. The saved first warm pair also copies instrumented IR within preparation; file writes follow. C# starts from prebuilt IL with PrepareMethod; source-to-IL build is outside that timer. Startup/formatting belong to launch-to-JSON. Input dimensions and output storage follow the [preceding loop-edge report](cpu_jit_benchmarks_profile_loop_edges.md). Host construction, reset/hashing/validation/JSON formatting are outside batch timers. Checksums can collide; finite float outputs use 1e-12 absolute/relative tolerance. Aggregate counters cannot reveal chronology; discarded standardized warmups cannot be audited. Bounded oracles do not prove arbitrary-program equivalence.

## Verification and durable evidence

**332 selected Rust tests and 18 Python CPU JIT tests passed after rebuilding liby.** Final IR O0–O3 with inherited native optimization and explicit native O1 cover ordinary/Instrument/profile-use modes and both flag arms. Source-intent oracles check complete aliased/disjoint buffers and canaries, String/Vec contents/freed slots, IEEE payloads/rounding/order, recursion, short-circuit effects and unseen inputs. A nonlinear counted-loop gate checks final tuning reaches ordinary/profile-use pipelines. Instrumented IR/profiles match, mandatory boundaries and eager lookups remain verified, and owned snapshots survive trainer disposal. Four-thread gates cover training 0–3 × native inherited O3/explicit O1 × both final flags, with selected-pointer and loop-edge training policies, exact algebraic outcomes and paused/bounded concurrent snapshots. The initial new-test helper type failure and pre-fix source are retained with the passing retry log.

Independent [measurement audit](benchmark_data/cpu_jit_final_unroll/audit.json): **143,463 checks passed**. Independent [oracle audit](benchmark_data/cpu_jit_final_unroll/verification/oracle-audit.json): **19,493 scalar checks passed**, with 97 complete memory hashes and 97 short-circuit counters across 42 raw worker files. Neither imports the runner. The durable-copy audit passes **143,765 checks**. Independent [raw statistical review](benchmark_data/cpu_jit_final_unroll/verification/review-results.md) passes 113,904 checks; [artifact review](benchmark_data/cpu_jit_final_unroll/verification/review-artifacts.md) passes 944 checks across two smoke and fourteen full pairs.

Verified 33 runner/112 frozen-source members, 2 workers plus rebuilt liby, 187 runtime identities and 1,969 historical files including both handoffs. All raw pairs/losses/settings/streams, frozen sources, exact binaries, root workspace-before archive, independent reviews and logs are retained under [durable evidence](benchmark_data/cpu_jit_final_unroll/README.md). The final manifest seals every evidence file except itself; publication.json hashes this report and README.

Runtime identity was probed from fresh Python CPUJit under the matching discovery environment; exact live worker mappings were not sampled during timing. The archive is not a hermetic SDK/system/Cargo image. Frozen-source auditors can use retained matching source/binary bytes if current files later change; runtime identity requires matching external files. Historical evidence and both handoffs remain unchanged; no commit was made.

Reproduce in a fresh directory with no concurrent heavy jobs:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite helpers --optimization-stage final-unroll --compare-optimizations \
  --repeats 9 --cold-repeats 5 --calls 32 --warmup 12 --profile-warmups 12 \
  --cpu 0 --output build_artifacts/new-final-unroll-run
```
