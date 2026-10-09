# CPU JIT outer loop-unrolling tradeoff — 2026-10-08

**Suppressing eligible outer-loop unrolling reduces median warm full preparation 15.32% and changes cold preparation -36.02%. The median per-process sum of fifteen native workload costs falls 2.65% (5/9 pair gains); the sum of separate workload medians changes +0.085413%.** This is an opt-in compilation-latency tradeoff; both production unrolling defaults remain true.

This comparison changes only requested `JitOptions.final_unroll_outer_loops` from true to false. Both arms retain global `final_loop_unrolling=true`. Explicit outer false adds `llvm.loop.unroll.disable` to eligible original natural loops containing a strictly nested natural loop in the same function. Innermost loops retain default tuning. The selector skips an entire chosen loop when existing `llvm.loop` metadata on any selected latch or ambiguous/unsupported latches prevent a shared valid LoopID. Instrument mode ignores this policy in both arms. Both arms use explicit training IR O1, final IR O3, native O3, VerifyEach and eager compilation. Instrumented IR and reconstructed instrumented assembly match byte for byte. Saved whole-module final IR and reconstructed assembly artifacts differ; only the eight eligible source kernels change function bodies. Exact profiles, returned training/results and checked memory contents match. Native optimization stays O3 while changed final IR produces changed code. These are measured policy tradeoffs, not a general execution-speed claim.

## Measured preparation

Times are medians in milliseconds. Positive signed paired saving is default unrolling minus suppressed eligible outer unrolling, so positive favors outer suppression. Differences of separate medians and median paired differences are distinct; medians need not sum. Ranges are observed sample ranges, not confidence intervals.

| Scope / preparation stage | Default outer tuning ms | Outer suppression ms | Paired saved median ms | Paired range ms | Outer-suppression gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented_compile | 128.884 | 132.753 | -0.310 | -44.113–20.448 | 3/9 |
| warm / profile_collection | 650.869 | 637.386 | 15.103 | -145.165–43.300 | 5/9 |
| warm / profile_snapshot | 0.018 | 0.018 | 0.000 | -0.124–0.002 | 5/9 |
| warm / optimized_recompile | 311.354 | 139.778 | 173.264 | 148.949–202.034 | 9/9 |
| warm / prepare | 1080.992 | 915.367 | 167.013 | 12.720–241.599 | 9/9 |
| cold / instrumented_compile | 135.613 | 140.898 | 2.064 | -11.136–20.748 | 3/5 |
| cold / profile_collection | 1.226 | 1.213 | 0.013 | -0.210–4.220 | 3/5 |
| cold / profile_snapshot | 0.006 | 0.007 | -0.000 | -0.073–0.001 | 2/5 |
| cold / optimized_recompile | 332.183 | 155.833 | 186.251 | 162.962–199.166 | 5/5 |
| cold / prepare | 470.913 | 301.272 | 188.016 | 169.204–217.853 | 5/5 |

Warm preparation changes **-15.322%** by separate medians (1080.992430→915.366656 ms). Its paired median saving is **167.012507 ms**, with 9/9 pairs improving.

Cold preparation changes **-36.024%** by separate medians (470.913258→301.271602 ms). Its paired median saving is **188.016130 ms**, with 5/5 pairs improving.

Warm profile execution changes -2.072% by separate medians, with 5/9 pairs improving and signed paired median saving 15.103147 ms. Its median per-sample share of optimized full preparation is 70.28%. Training policy and saved training code are identical; this comparison does not claim a profiling-execution improvement. Every loss remains in raw records and tables.

## Controls and implementation

Run start: `2026-10-08T18:08:22.299912+00:00`; AMD Ryzen 9 9950X 16-Core Processor, Linux x86-64, logical CPU 0; LLVM 23.1.1; fixed .NET 8.0.31 baseline. 9 rotating warm process triples and 5 cold triples execute sequentially without concurrent agent builds/tests/CPU audits during timing. Each warm worker makes 12 standardized warmups and 32 timed calls per workload; Y additionally makes 12 measured training calls. Other host activity and frequency remain uncontrolled. Versions, discovery environment and actual commands are retained.

`final_unroll_outer_loops: bool` defaults true, leaving original loop tuning untouched. False is active only in ordinary/profile-use mode when global `final_loop_unrolling` is true; Instrument and globally disabled-unrolling compilations bypass the selector. Cache v13 records the requested flag. An outer loop is an original natural loop with a distinct strictly nested natural loop in the same lowered function; helper calls and loops introduced later by LLVM do not count as original nesting. Every selected loop latch shares one self-referential `llvm.loop` node containing unroll-disable. Existing `llvm.loop` metadata on any selected loop latch, unreachable/irreducible regions and ambiguous latch ownership retain existing treatment. This metadata also suppresses unroll-and-jam; O3 vectorization and interleaving remain enabled, but downstream decisions and generated code can change. [LLVM metadata semantics](https://llvm.org/docs/LangRef.html#llvm-loop-unroll-and-jam).

Requested outer policy is true/false in every settings/worker record. `outer_unroll_policy_active` is false for both Instrument arms and false/true for final compilation. `outer_unroll_annotations` counts original loops actually annotated, not transformations performed: it is zero for Instrument/inactive compilations and positive for this suite’s optimized final compilation. Both requested edge-counter flags are false throughout. Base/final IR O3, explicit training O1, native O3, VerifyEach, mandatory full-module checks and eager materialization remain enabled. Runtime query/mutation/copy, rotates, compact adapters and scalar helper effects remain enabled; natural-loop control weights are excluded. Existing C/Python defaults stay true.

Profiling still uses separate atomic true/false outcomes for each original lowered-IR site, with original fingerprint/site identities and wraparound after 2^64 observations. Owned relaxed snapshots remain atomic per counter; concurrent snapshots need not represent one instant across the array. Final sessions have no training counters. Training, snapshots and recompilation remain explicit, with eager public/adapter lookups.

## Outer-policy annotation records

Counts are original natural loops receiving a shared disable node, not pass transformation counts. The active flag records selector policy; existing latch `llvm.loop` metadata/ambiguous cases may still be skipped. A separate retained static review recovers eight eligible outers from historical preopt selected-address Instrument IR with the same original fingerprint and no CFG splits. This is a scoped historical CFG cross-check, not current raw preopt regeneration or proof of exact original text.

| Scope / compilation | Default active | Default annotations | Suppressed active | Suppressed annotations |
| --- | --- | ---: | --- | ---: |
| warm / instrumented | false | 0 | false | 0 |
| warm / profiled | false | 0 | true | 8 |
| cold / instrumented | false | 0 | false | 0 |
| cold / profiled | false | 0 | true | 8 |

## Compilation phases and object evidence

| Scope / kind / phase | Default outer tuning ms | Outer suppression ms | Paired saved median ms | Paired range ms | Outer-suppression gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented / pipeline | 58.264 | 60.210 | -3.285 | -19.224–7.723 | 4/9 |
| warm / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/9 |
| warm / instrumented / verification | 0.264 | 0.264 | 0.000 | -0.010–1.712 | 5/9 |
| warm / instrumented / materialization | 59.754 | 61.819 | -2.929 | -24.959–8.425 | 3/9 |
| warm / instrumented / total | 128.884 | 132.753 | -0.310 | -44.113–20.448 | 3/9 |
| warm / profiled / pipeline | 167.244 | 80.552 | 88.357 | 72.655–96.558 | 9/9 |
| warm / profiled / profile_selection | 3.124 | 1.012 | 2.077 | 0.507–8.886 | 9/9 |
| warm / profiled / verification | 0.864 | 0.394 | 0.467 | -0.373–1.543 | 8/9 |
| warm / profiled / materialization | 134.140 | 52.369 | 81.484 | 67.834–93.828 | 9/9 |
| warm / profiled / total | 311.288 | 139.708 | 173.269 | 148.950–202.029 | 9/9 |
| cold / instrumented / pipeline | 62.060 | 63.571 | -1.241 | -6.050–15.551 | 1/5 |
| cold / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/5 |
| cold / instrumented / verification | 0.265 | 0.274 | -0.015 | -0.024–0.012 | 2/5 |
| cold / instrumented / materialization | 62.192 | 64.294 | -1.520 | -3.882–12.279 | 2/5 |
| cold / instrumented / total | 135.613 | 140.898 | 2.064 | -11.136–20.748 | 3/5 |
| cold / profiled / pipeline | 178.865 | 92.117 | 89.004 | 82.146–95.522 | 5/5 |
| cold / profiled / profile_selection | 7.121 | 1.017 | 6.105 | 1.849–8.803 | 5/5 |
| cold / profiled / verification | 0.886 | 0.398 | 0.484 | 0.395–0.689 | 5/5 |
| cold / profiled / materialization | 138.799 | 56.307 | 82.138 | 75.218–107.300 | 5/5 |
| cold / profiled / total | 331.770 | 155.763 | 185.911 | 162.962–199.163 | 5/5 |

Retained preparation/compilation metrics with separate-median or paired-median losses (32 metrics): `warm/preparation/instrumented_compile` separate +3.002%, paired saved -0.310014 ms; `warm/instrumented/pipeline` separate +3.341%, paired saved -3.285106 ms; `warm/instrumented/optimization` separate +3.341%, paired saved -3.284406 ms; `warm/instrumented/verification` separate +0.140%, paired saved +0.000179 ms; `warm/instrumented/materialization` separate +3.455%, paired saved -2.929483 ms; `warm/instrumented/total` separate +3.002%, paired saved -0.310014 ms; `warm/instrumented/submission` separate +1.506%, paired saved -0.000430 ms; `warm/instrumented/first_lookup` separate +3.460%, paired saved -2.931432 ms; `warm/instrumented/profile_lookup` separate +4.839%, paired saved -0.000030 ms; `warm/instrumented/materialization_other` separate +8.669%, paired saved -0.000390 ms; `warm/instrumented/first_lookup_before_object` separate +3.488%, paired saved -2.936132 ms; `warm/instrumented/first_lookup_after_object` separate +2.768%, paired saved +0.000170 ms; `warm/profiled/optimization_other` separate -5.668%, paired saved -0.000190 ms; `warm/profiled/remaining_function_lookups` separate +1.243%, paired saved -0.000140 ms; `warm/profiled/materialization_other` separate +1.528%, paired saved -0.000189 ms; `cold/preparation/instrumented_compile` separate +3.897%, paired saved +2.064063 ms; `cold/preparation/profile_snapshot` separate +7.595%, paired saved -0.000470 ms; `cold/instrumented/pipeline` separate +2.434%, paired saved -1.240514 ms; `cold/instrumented/optimization_other` separate +18.634%, paired saved -0.000580 ms; `cold/instrumented/optimization` separate +2.436%, paired saved -1.241374 ms; `cold/instrumented/verification` separate +3.631%, paired saved -0.014890 ms; `cold/instrumented/materialization` separate +3.380%, paired saved -1.520498 ms; `cold/instrumented/total` separate +3.897%, paired saved +2.064063 ms; `cold/instrumented/submission` separate +8.579%, paired saved -0.001390 ms; `cold/instrumented/first_lookup` separate +3.380%, paired saved -1.518687 ms; `cold/instrumented/remaining_function_lookups` separate +5.823%, paired saved -0.000570 ms; `cold/instrumented/profile_lookup` separate -6.203%, paired saved -0.000050 ms; `cold/instrumented/materialization_other` separate +1.148%, paired saved +0.000270 ms; `cold/instrumented/first_lookup_before_object` separate +3.370%, paired saved -1.522877 ms; `cold/instrumented/first_lookup_after_object` separate +0.651%, paired saved +0.003890 ms; `cold/profiled/optimization_other` separate +24.453%, paired saved -0.001520 ms; `cold/profiled/remaining_function_lookups` separate -2.910%, paired saved -0.000380 ms.

Warm final total changes -55.119% (311.287615→139.708205 ms), with paired median saving 173.268848 ms.

Warm final pipeline changes -51.836% (167.244171→80.552142 ms), with paired median saving 88.357027 ms.

Warm final materialization changes -60.959% (134.140274→52.369210 ms), with paired median saving 81.483533 ms.

Original CFG classification and metadata construction are charged to the flat `profile_setup` phase and full preparation; their raw intervals are retained. No separate selector or unroll-pass timer is claimed.

The measured default-pipeline timing difference includes VerifyEach and changed downstream IR work. It is not an exclusive timing of LLVM’s outer-loop unroll work. Flat phases sum exactly to total; nested optimization/materialization parents sum to their primary phase. Optional object-ready children partition first lookup and are already included. Before-object work includes native emission and preceding ORC work; after-object includes observer overhead, linking and lookup. Neither is an exclusive backend or linker timer. Timing snapshots exclude later execution/getters.

With eligible outer suppression, cold final compilation spends 61.08% median per-sample share in optimization and 36.02% in materialization.

| Scope / compilation | Default outer observed object bytes | Outer suppression observed object bytes |
| --- | ---: | ---: |
| warm / instrumented | 21,216 | 21,216 |
| warm / profiled | 44,424 | 18,176 |
| cold / instrumented | 21,216 | 21,216 |
| cold / profiled | 45,000 | 18,240 |

All 56 measured compilations observe one object and an eligible first-lookup split, totaling 56 callbacks, 2,016 public/adapter lookups and 28 profile-global lookups. These are actual live ORC object-file bytes, including file metadata, rather than executable-memory allocation. Object contents, placement and mappings are not retained or hashed. Smaller byte counts do not prove particular object contents.

| Saved/reconstructed artifact | Default outer bytes | Outer suppression bytes |
| --- | ---: | ---: |
| Instrumented IR | 206,077 | 206,077 |
| Final IR | 687,420 | 185,602 |
| Reconstructed instrumented assembly | 126,470 | 126,470 |
| Reconstructed final assembly | 376,345 | 123,816 |

Independent static review counts final IR instructions 8,769→2,373 and reconstructed final assembly instructions 8,910→2,609. Instrumented IR has 2,457 instructions and reconstructed assembly 3,091 in both arms. Only the eight eligible outer-loop functions change normalized final IR/assembly bodies; other function bodies match. These whole-pipeline static counts include operandless assembly instructions and are diagnostic.

Artifact identities refer to the saved first warm pair; every full/smoke pair retains exact profiles and output streams. Reconstructed assembly uses external llc O3/PIC/native CPU/small code model, while live ORC requests JITDefault. Its text sizes/static instruction counts are scoped diagnostics, not actual live JIT-code bytes or pass timings. Saved whole-module final IR and reconstructed assembly artifacts differ; only the eight eligible source kernels change function bodies. Vectorized loop metadata remains in both final arms. No pass-exclusive or dynamic instruction-count cause is inferred from these artifacts.

## Native execution against C# — all fifteen workloads

Both Y arms execute native O3 generated from their differing final IR. Times are median milliseconds per call; C#/Y ratios divide separate medians. Pair gains/ranges retain process variability. Changed code and code size do not establish general native speed improvement; even unchanged functions can vary with placement/cache/frequency state.

| Workload | Default-outer Y | Outer-suppressed Y | C# | Y time change | C# / outer-suppressed Y | Outer-suppression pair gains |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.708031 | 0.772575 | 1.307292 | +9.116% | 1.69212× | 5/9 |
| recursive_fib | 0.154563 | 0.151020 | 0.383958 | -2.292% | 2.54243× | 7/9 |
| float_recurrence | 1.425498 | 1.396642 | 1.439791 | -2.024% | 1.03089× | 8/9 |
| indexed_memory | 0.109822 | 0.109779 | 0.172652 | -0.039% | 1.57273× | 5/9 |
| unsigned_mix | 0.571519 | 0.552825 | 0.570414 | -3.271% | 1.03182× | 5/9 |
| float_dot | 0.098231 | 0.123359 | 0.096230 | +25.581% | 0.78008× | 3/9 |
| short_circuit | 2.246555 | 2.233316 | 2.462689 | -0.589% | 1.10271× | 4/9 |
| binary_search | 1.860434 | 1.842879 | 1.795827 | -0.944% | 0.97447× | 5/9 |
| string_scan | 0.038327 | 0.037399 | 0.353015 | -2.421% | 9.43913× | 7/9 |
| vec_scan_append | 0.064214 | 0.064149 | 0.094949 | -0.101% | 1.48012× | 5/9 |
| vec_dynamic_byte | 0.064140 | 0.063488 | 0.103651 | -1.017% | 1.63261× | 6/9 |
| vec_dynamic_i64 | 0.040775 | 0.041365 | 0.111461 | +1.447% | 2.69455× | 2/9 |
| string_bulk_append | 0.029492 | 0.029442 | 0.360473 | -0.168% | 12.24356× | 3/9 |
| string_scan_helper | 0.036789 | 0.037388 | 0.357248 | +1.628% | 9.55518× | 2/9 |
| vec_scan_append_helper | 0.061916 | 0.061090 | 0.095615 | -1.333% | 1.56514× | 6/9 |

| Workload | Default / outer-suppressed paired median | Range | C# / outer-suppressed paired median | Range |
| --- | ---: | ---: | ---: | ---: |
| integer_branch | 1.01158× | 0.87951–1.20928 | 1.72699× | 1.46795–1.97405 |
| recursive_fib | 1.00080× | 0.61781–1.58360 | 1.96164× | 1.61288–2.66976 |
| float_recurrence | 1.02530× | 0.94532–1.14300 | 1.03139× | 0.87266–1.56695 |
| indexed_memory | 1.00041× | 0.48662–2.04527 | 1.56352× | 1.22566–2.88464 |
| unsigned_mix | 1.00846× | 0.79396–1.36220 | 1.05536× | 0.79582–1.28886 |
| float_dot | 0.99639× | 0.48660–2.11090 | 0.98053× | 0.42695–1.55339 |
| short_circuit | 0.99211× | 0.97836–1.10561 | 1.07392× | 1.05133–1.13295 |
| binary_search | 1.00106× | 0.88061–1.11052 | 0.97434× | 0.83608–1.03301 |
| string_scan | 1.02920× | 0.95445–3.81355 | 9.50329× | 1.76776–11.41141 |
| vec_scan_append | 1.00237× | 0.95622–1.14199 | 1.42756× | 1.06203–2.99002 |
| vec_dynamic_byte | 1.03816× | 0.64042–2.97475 | 1.62768× | 0.60709–3.38694 |
| vec_dynamic_i64 | 0.96649× | 0.28699–1.04596 | 2.51786× | 0.95448–5.56152 |
| string_bulk_append | 0.99561× | 0.33115–1.13242 | 12.15787× | 2.12557–14.40793 |
| string_scan_helper | 0.97795× | 0.22063–2.46945 | 9.45368× | 2.16716–10.07024 |
| vec_scan_append_helper | 1.01362× | 0.96758–2.66884 | 1.56531× | 1.48740–3.13448 |

Retained native median regressions (4 workloads): `integer_branch` +9.116%; `float_dot` +25.581%; `vec_dynamic_i64` +1.447%; `string_scan_helper` +1.628%.

Retained native paired-median regressions (5 workloads): `float_dot` default/suppressed paired median 0.99639×; `short_circuit` default/suppressed paired median 0.99211×; `vec_dynamic_i64` default/suppressed paired median 0.96649×; `string_bulk_append` default/suppressed paired median 0.99561×; `string_scan_helper` default/suppressed paired median 0.97795×.

One call to each workload costs median 7.835239→7.627933 ms (-2.646%). The paired median saving is 0.072214875 ms; 5/9 bundles improve and 4/9 are slower.

Summing the fifteen separate workload medians instead gives **7.510304000→7.516718812 ms (+0.085413%)**. This aggregation differs from the median of per-process bundle sums above; the sum of separate medians is near parity with a small loss in this run. Native pair consistency is limited, and neither aggregation establishes a general native speedup.

## Paired preparation-plus-use model

This arithmetic model charges one full warm preparation and N bundles, where one bundle is one call to each of the fifteen heterogeneous synthetic workloads. For every process pair, bundle cost is the sum of its per-workload batch-average ns/call. Each arm’s modeled cost is `prepare_ns + N * sum(ns_per_call)`. The auditor independently reconstructs every pair, signed saving, loss count and crossover from raw worker rows; it does not import the runner. These are extrapolations of measured costs, not additional timed N-bundle runs.

| Bundles N | Default-outer modeled ms | Outer-suppressed modeled ms | Paired saved median ms | Paired range ms | Outer-suppression gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1088.828 | 922.882 | 166.957 | 13.040–241.978 | 9/9 |
| 12 | 1175.015 | 1008.669 | 166.347 | 16.565–246.153 | 9/9 |
| 32 | 1331.720 | 1162.170 | 165.237 | 22.972–253.744 | 9/9 |
| 100 | 1864.516 | 1676.782 | 164.345 | 44.759–279.551 | 9/9 |
| 1000 | 8916.231 | 8532.560 | 229.338 | 6.389–660.453 | 9/9 |

A positive applicable crossover requires both an initial preparation saving and a native bundle slowdown. Signed crossover is `preparation_saved_ns / native_slowdown_ns`; negative ratios remain retained, and zero native difference has no finite crossover. The table keeps every signed pair and its applicability rather than filtering losses.

| Repeat | Preparation saved ms | Native slowdown ms/bundle | Signed crossover bundles | Applicable |
| --- | ---: | ---: | ---: | ---: |
| 0 | 12.719910 | -0.320392562 | -39.701015 | false |
| 1 | 214.794946 | 0.017149844 | 12524.600756 | true |
| 2 | 113.205974 | 0.076426188 | 1481.245862 | true |
| 3 | 167.012507 | 0.055484687 | 3010.064840 | true |
| 4 | 194.365771 | -0.466086813 | -417.016242 | false |
| 5 | 118.041897 | -0.211010844 | -559.411521 | false |
| 6 | 241.598752 | -0.379527281 | -636.578091 | false |
| 7 | 157.123448 | -0.072214875 | -2175.776777 | false |
| 8 | 229.083173 | 0.222694625 | 1028.687482 | true |

The 4/9 positive applicable pair crossovers have median **2245.655351 bundles**, range **1028.687482–12524.600756**. The median of individual crossover ratios differs from dividing median paired preparation saving by median paired native loss. Every signed pair, including non-applicable cases, remains in the raw summary.

Modeled N=1: 9/9 gains, 0/9 losses and 0 ties.

Modeled N=12: 9/9 gains, 0/9 losses and 0 ties.

Modeled N=32: 9/9 gains, 0/9 losses and 0 ties.

Modeled N=100: 9/9 gains, 0/9 losses and 0 ties.

Modeled N=1000: 9/9 gains, 0/9 losses and 0 ties.

The model holds the current 32-call batch means and input distribution fixed. It excludes first-call and standardized warmup costs, and host setup/reset/hash/format work outside preparation/batch timers. Thermal/cache/frequency/input-state effects can change in longer real runs. Equal weighting of fifteen synthetic workloads is not an application call mixture. Different programs and call mixtures can favor either policy; the model is not a universal break-even recommendation or a general language ranking.

## PGO, representation and preparation scope

Training overlaps the first 12 of 32 measurement seeds, so evaluation is same-distribution rather than held out. Both arms retain exact original profiles and complete training streams; raw original lowered IR is not archived for independent fingerprint regeneration.

Warm profiles record 140,308,654 outcomes across 111 original sites, applying 88.

Cold profiles record 328,727 outcomes across 111 original sites, applying 88.

C# Release .NET 8 has tiering/dynamic PGO and ReadyToRun disabled; kernel entries use NoInlining and scalar helpers keep normal inlining. Y byte String/Vec layout, growth and allocation/free policy differ from C# UTF-16 StringBuilder/List/GC. ASCII matches these values, not arbitrary Unicode. Y allocation/growth/free and C# in-batch GC are timed; deferred managed reclamation is excluded. Generation deltas are not independent collection counts. Raw allocation/GC fields are retained.

| Cold stage | Default-outer Y ms | Outer-suppressed Y ms | C# ms |
| --- | ---: | ---: | ---: |
| Final compilation / IL preparation | 331.770 | 155.763 | 6.571 |
| First-call set | 0.023 | 0.017 | 1.401 |
| Launch to JSON | 477.677 | 304.206 | 326.316 |

Y preparation starts from source and charges instrumented compilation, measured training/return storage, snapshots/profile formatting, final recompilation, input setup and trainer disposal. The saved first warm pair also copies instrumented IR within preparation; file writes follow. C# starts from prebuilt IL with PrepareMethod; source-to-IL build is outside that timer. Startup/formatting belong to launch-to-JSON. Input dimensions and output storage follow the [preceding final-unroll report](cpu_jit_benchmarks_final_unroll.md). Host construction, reset/hashing/validation/JSON formatting are outside batch timers. Checksums can collide; finite float outputs use 1e-12 absolute/relative tolerance. Aggregate counters cannot reveal chronology; discarded standardized warmups cannot be audited. Bounded oracles do not prove arbitrary-program equivalence.

## Verification and durable evidence

**342 selected Rust tests and 18 Python CPU JIT tests passed after rebuilding liby.** The retained verification metadata/logs describe the selected full-result, IEEE, memory/String/Vec, alias/short-circuit/recursion, profile, snapshot, eager-lookup and concurrency gates. Low-level CFG fixtures verify original nested-loop eligibility, shared self-referential latch metadata and whole-loop skip behavior. Ordinary/profile-use behavior, untouched Instrument code and global-policy precedence are checked independently of benchmark arithmetic. Bounded gates do not prove arbitrary-program equivalence. Initial failures and pre-fix sources, if any, remain retained.

Independent [measurement audit](benchmark_data/cpu_jit_outer_unroll/audit.json): **144,781 checks passed**. Independent [oracle audit](benchmark_data/cpu_jit_outer_unroll/verification/oracle-audit.json): **19,493 scalar checks passed**, with 97 complete memory hashes and 97 short-circuit counters across 42 raw worker files. Neither imports the runner. The durable-copy audit passes **145,083 checks**. Independent [raw statistical review](benchmark_data/cpu_jit_outer_unroll/verification/review-results.md) passes 114,655 checks; [artifact review](benchmark_data/cpu_jit_outer_unroll/verification/review-artifacts.md) passes 1,189 checks across two smoke and fourteen full pairs.

The independent [historical CFG provenance review](benchmark_data/cpu_jit_outer_unroll/verification/review-provenance.json) passes 67 checks: retained diagnostic command/stdout/CFG fingerprint and all 111 conditional site identities match all 32 current smoke/full Y worker records. This strengthens the historical CFG linkage without asserting exact current original-IR byte reconstruction.

Verified 34 runner/114 frozen-source members, 2 workers plus rebuilt liby, 187 runtime identities and 2,233 historical files including both handoffs. All raw pairs/losses/settings/streams, frozen sources, exact binaries, root workspace-before archive, independent reviews and logs are retained under [durable evidence](benchmark_data/cpu_jit_outer_unroll/README.md). The final manifest seals every evidence file except itself; publication.json hashes this report and README.

Runtime identity was probed from fresh Python CPUJit under the matching discovery environment; exact live worker mappings were not sampled during timing. The archive is not a hermetic SDK/system/Cargo image. Frozen-source auditors can use retained matching source/binary bytes if current files later change; runtime identity requires matching external files. Historical evidence and both handoffs remain unchanged; no commit was made.

Reproduce in a fresh directory with no concurrent heavy jobs:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite helpers --optimization-stage outer-unroll --compare-optimizations \
  --repeats 9 --cold-repeats 5 --calls 32 --warmup 12 --profile-warmups 12 \
  --cpu 0 --output build_artifacts/new-outer-unroll-run
```
