# CPU JIT atomic profile-edge results — 2026-10-08

**Fixed-address atomic edge counters reduce median full preparation 0.74%; cold preparation falls 2.35%. Full-size training execution rises 0.32%.** Final execution remains O3.

This comparison changes only the requested `JitOptions.profile_edge_counters` flag from false to true. Both arms use explicit O1 training IR, final IR/native O3, per-pass verification and eager compilation. Eligible branches use fixed-address atomic increments on the selected edge; PHI-successor branches retain the preceding selected-address form. Every retained profile and output stream matches, and saved final IR and reconstructed final assembly are byte-identical. The new Rust option defaults to false; existing C/Python compilation policy is unchanged.

## Measured preparation

Times are medians in milliseconds. Signed paired saving is selected-address minus edge instrumentation; positive favors edges. Differences of separate medians are distinct from median paired differences; medians need not sum. Ranges are observed sample ranges, not confidence intervals.

| Scope / preparation stage | Selected address ms | Fixed edge ms | Paired saved median ms | Paired range ms | Edge gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented_compile | 92.825 | 85.091 | 7.393 | 6.549–8.975 | 9/9 |
| warm / profile_collection | 504.457 | 506.080 | -1.522 | -3.917–2.112 | 2/9 |
| warm / profile_snapshot | 0.018 | 0.019 | -0.002 | -0.003–0.001 | 3/9 |
| warm / optimized_recompile | 220.754 | 220.133 | 1.130 | -2.009–8.850 | 6/9 |
| warm / prepare | 818.201 | 812.156 | 5.813 | 0.975–19.911 | 9/9 |
| cold / instrumented_compile | 93.477 | 85.692 | 7.925 | 5.470–9.189 | 5/5 |
| cold / profile_collection | 1.203 | 1.205 | -0.005 | -0.008–-0.000 | 0/5 |
| cold / profile_snapshot | 0.006 | 0.006 | -0.000 | -0.001–0.000 | 2/5 |
| cold / optimized_recompile | 219.308 | 219.865 | -0.557 | -3.207–2.151 | 2/5 |
| cold / prepare | 314.609 | 307.215 | 8.618 | 2.253–10.849 | 5/5 |

Warm preparation changes **-0.74%** by separate medians (818.201→812.156 ms). Its paired median saving is **5.813 ms**, with 9/9 pairs improving.

Cold preparation changes **-2.35%** by separate medians (314.609→307.215 ms). Its paired median saving is **8.618 ms**, with 5/5 pairs improving.

Measured full-size profile execution changes +0.322% by separate medians, with 2/9 pairs improving and a signed paired median saving of -1.522 ms. The preparation saving comes from compilation; no profiling-execution improvement or final native-execution win is claimed. Profile execution remains 62.31% of median per-sample full preparation and remains the next measured bottleneck.

These are measured policy differences for this synthetic suite on one machine. All paired losses remain in the raw data and tables. Changing instrumentation does not establish a final native execution improvement or a general language ranking.

## Controls and implementation

Run start: `2026-10-08T15:33:24.353429+00:00`; AMD Ryzen 9 9950X 16-Core Processor, Linux x86-64, logical CPU 0; LLVM 23.1.1; fixed .NET 8.0.31 baseline. 9 rotating warm process triples and 5 cold triples execute sequentially, with no concurrent agent builds/tests/CPU audits during timing. Each warm worker makes 12 standardized warmups and 32 timed calls per workload; Y additionally makes 12 measured training calls. Other host activity/frequency remain uncontrolled. SDK/compiler versions and commands are retained in metadata.

`JitOptions.profile_edge_counters: bool` defaults false. In Instrument mode, true adds two edge blocks to an eligible original conditional branch, with one aligned monotonic atomic add to a fixed outcome counter in the selected block. A branch whose original successor begins with a PHI keeps the selected-address atomic increment; PHI incoming edges and SSA are preserved. The original terminator and metadata remain intact. Ordinary and profile-use modes ignore this requested flag. Cache v10 includes it. Existing C/Python entrypoints keep the false default.

Both measured arms use base IR O3, explicit training O1, native O3, VerifyEach true, mandatory full-module checks and eager whole-module materialization. Rotate, runtime query/mutation/copy, compact adapters and scalar helper effects are enabled; natural-loop control weights are excluded. Requested `profile_edge_counters` is false/true in every settings record, including final compilation where it has no effect. Per-entry `compilation_settings` records the actual IR/native tiers and requested verification/instrumentation policy.

Fingerprint/site identities come from original lowered IR before instrumentation. The counter array still has separate true/false outcomes per original site. Each outcome wraps after 2^64 observations. Snapshot reads remain atomic and owned; a concurrent snapshot need not represent one instant across all counters. Final profile-use sessions have no training counters. Training, snapshots and recompilation remain explicit; no sampling, counter batching, lazy compilation or background tiering is introduced.

## Compilation phases

| Scope / kind / phase | Selected address ms | Fixed edge ms | Paired saved median ms | Paired range ms | Edge gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented / pipeline | 42.520 | 41.142 | 1.404 | 0.774–1.610 | 9/9 |
| warm / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/9 |
| warm / instrumented / verification | 0.259 | 0.264 | -0.006 | -0.011–-0.001 | 0/9 |
| warm / instrumented / materialization | 41.898 | 35.650 | 6.267 | 5.455–7.657 | 9/9 |
| warm / instrumented / total | 92.825 | 85.091 | 7.393 | 6.549–8.975 | 9/9 |
| warm / profiled / pipeline | 119.077 | 118.451 | 0.397 | -1.223–7.454 | 7/9 |
| warm / profiled / profile_selection | 2.903 | 2.896 | 0.007 | -0.052–0.030 | 5/9 |
| warm / profiled / verification | 0.831 | 0.833 | -0.002 | -0.018–0.024 | 3/9 |
| warm / profiled / materialization | 92.335 | 91.856 | -0.052 | -1.736–2.317 | 4/9 |
| warm / profiled / total | 220.686 | 220.059 | 1.131 | -2.009–8.850 | 6/9 |
| cold / instrumented / pipeline | 42.451 | 42.142 | 1.494 | -0.218–2.030 | 4/5 |
| cold / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/5 |
| cold / instrumented / verification | 0.258 | 0.264 | -0.006 | -0.010–0.001 | 2/5 |
| cold / instrumented / materialization | 42.098 | 35.371 | 6.612 | 6.057–7.741 | 5/5 |
| cold / instrumented / total | 93.477 | 85.692 | 7.925 | 5.470–9.189 | 5/5 |
| cold / profiled / pipeline | 117.865 | 118.921 | 0.075 | -3.330–2.670 | 3/5 |
| cold / profiled / profile_selection | 2.890 | 2.898 | -0.024 | -0.038–0.019 | 2/5 |
| cold / profiled / verification | 0.819 | 0.819 | -0.001 | -0.017–0.014 | 2/5 |
| cold / profiled / materialization | 91.210 | 91.431 | -0.082 | -0.886–0.232 | 2/5 |
| cold / profiled / total | 219.240 | 219.790 | -0.551 | -3.214–2.153 | 2/5 |

Retained separate-median compilation regressions: `warm/instrumented/verification` +2.266%; `warm/profiled/verification` +0.207%; `cold/instrumented/verification` +2.216%; `cold/profiled/pipeline` +0.896%; `cold/profiled/profile_selection` +0.284%; `cold/profiled/materialization` +0.242%; `cold/profiled/total` +0.251%.

Flat phases sum exactly to total. Nested optimization/materialization parents sum exactly to their primary phase; optional object-ready children partition first lookup and are already included. Before-object work includes native emission and preceding ORC work; after-object includes observer overhead/linking/lookup. Neither is an exclusive backend or linker timer. Timing snapshots exclude later execution/getters; object bytes include file metadata, not executable memory alone.

With edge training, cold final profiled compilation spends 55.40% median per-sample share in optimization and 41.58% in materialization. The final O3 policy is unchanged. Earlier external opt diagnostics prioritize investigation but remain separate from measured JIT pass timings.

The independent [static artifact review](benchmark_data/cpu_jit_profile_edges/verification/review-artifacts.md) checks 626 retained-output/structural facts across the smoke and full runs. Saved post-O1 smoke artifacts have more blocks/static atomic instructions and fewer selected-address operations/external llc instructions. These are scoped static diagnostics, not dynamic instruction counts or actual JIT pass timings; they do not identify a cause for the measured compilation saving or establish an execution improvement.

| Scope / compilation | Selected-address observed object bytes | Edge observed object bytes |
| --- | ---: | ---: |
| warm / instrumented | 21,216 | 20,552 |
| warm / profiled | 44,424 | 44,424 |
| cold / instrumented | 21,216 | 20,552 |
| cold / profiled | 45,000 | 45,000 |

All 56 measured compilations observe one object and an eligible first-lookup split, totaling 56 callbacks, 2,016 public/adapter function lookups and 28 profile-global lookups. The table reports median live ORC object-file bytes, including file metadata. Actual object contents are not retained or hashed; equal counts or sizes do not establish native content identity.

## Native execution against C# — all fifteen workloads

Both Y columns execute final O3. Times are median milliseconds per call; C#/Y ratios divide per-arm medians. Pair gains/ranges are retained even when separate medians disagree. Identical saved final IR and reconstructed assembly do not establish identical live object contents, addresses, placement, allocator/cache/frequency state. These native variations cannot be attributed to a changed final IR policy.

| Workload | Selected-address Y | Edge-trained Y | C# | Y time change | C# / edge-trained Y | Edge-training pair gains |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.589627 | 0.590541 | 1.044977 | +0.155% | 1.76952× | 1/9 |
| recursive_fib | 0.148743 | 0.149054 | 0.289823 | +0.209% | 1.94442× | 2/9 |
| float_recurrence | 1.088473 | 1.088333 | 1.089617 | -0.013% | 1.00118× | 4/9 |
| indexed_memory | 0.107978 | 0.107791 | 0.169360 | -0.173% | 1.57119× | 2/9 |
| unsigned_mix | 0.453497 | 0.453373 | 0.453803 | -0.027% | 1.00095× | 5/9 |
| float_dot | 0.095671 | 0.096024 | 0.095508 | +0.369% | 0.99462× | 2/9 |
| short_circuit | 1.783065 | 1.784155 | 1.899226 | +0.061% | 1.06450× | 3/9 |
| binary_search | 1.430802 | 1.433138 | 1.425938 | +0.163% | 0.99498× | 2/9 |
| string_scan | 0.037814 | 0.037962 | 0.257975 | +0.390% | 6.79562× | 5/9 |
| vec_scan_append | 0.063486 | 0.063548 | 0.090880 | +0.098% | 1.43009× | 4/9 |
| vec_dynamic_byte | 0.063478 | 0.063394 | 0.099092 | -0.133% | 1.56312× | 5/9 |
| vec_dynamic_i64 | 0.039834 | 0.039844 | 0.108542 | +0.026% | 2.72419× | 4/9 |
| string_bulk_append | 0.029051 | 0.029014 | 0.255147 | -0.129% | 8.79400× | 5/9 |
| string_scan_helper | 0.036133 | 0.036065 | 0.258091 | -0.188% | 7.15620× | 6/9 |
| vec_scan_append_helper | 0.060753 | 0.060808 | 0.091906 | +0.089% | 1.51142× | 2/9 |

| Workload | Selected / edge-trained paired median | Range | C# / edge-trained paired median | Range |
| --- | ---: | ---: | ---: | ---: |
| integer_branch | 0.99758× | 0.98641–1.00362 | 1.76839× | 1.74787–1.77278 |
| recursive_fib | 0.99870× | 0.97962–1.00174 | 1.94774× | 1.91337–1.94898 |
| float_recurrence | 0.99962× | 0.99788–1.00709 | 1.00086× | 0.99812–1.00347 |
| indexed_memory | 0.99624× | 0.98758–1.00745 | 1.57375× | 1.54406–1.58390 |
| unsigned_mix | 1.00017× | 0.99638–1.00437 | 1.00049× | 0.99833–1.01090 |
| float_dot | 0.99655× | 0.98840–1.00917 | 0.99513× | 0.98950–1.00396 |
| short_circuit | 0.99897× | 0.98997–1.00549 | 1.06593× | 1.05060–1.07861 |
| binary_search | 0.99670× | 0.99068–1.00285 | 0.99553× | 0.98727–1.00111 |
| string_scan | 1.00786× | 0.64223–1.01958 | 6.79937× | 4.41921–6.89523 |
| vec_scan_append | 0.99902× | 0.99396–1.00293 | 1.42832× | 1.41847–1.46150 |
| vec_dynamic_byte | 1.00185× | 0.99625–1.00445 | 1.56445× | 1.53731–1.58337 |
| vec_dynamic_i64 | 0.99904× | 0.98936–1.00457 | 2.71607× | 2.68862–2.84631 |
| string_bulk_append | 1.00087× | 0.99150–1.01132 | 8.79400× | 8.69607–8.94642 |
| string_scan_helper | 1.00623× | 0.98848–1.02580 | 7.15620× | 6.89401–7.21938 |
| vec_scan_append_helper | 0.99897× | 0.98377–1.00151 | 1.51277× | 1.45999–1.78431 |

Retained native median regressions: `integer_branch` +0.155%; `recursive_fib` +0.209%; `float_dot` +0.369%; `short_circuit` +0.061%; `binary_search` +0.163%; `string_scan` +0.390%; `vec_scan_append` +0.098%; `vec_dynamic_i64` +0.026%; `vec_scan_append_helper` +0.089%.

## PGO, runtime representation and preparation scope

Y uses explicit measured PGO. Training overlaps the first 12 of 32 measurement seeds, so evaluation is same-distribution rather than held out. Both arms retain exact profiles and complete returned training streams. The raw original lowered IR is not archived for independent fingerprint regeneration.

Warm profiles record 140,308,654 outcomes across 111 original sites, applying 88.

Cold profiles record 328,727 outcomes across 111 original sites, applying 88.

C# Release .NET 8 has tiering/dynamic PGO and ReadyToRun disabled. Kernel entries use NoInlining; ordinary scalar helpers keep normal inlining. Y byte Strings/Vecs differ from UTF-16 StringBuilder/List in representation, growth and allocator policy. ASCII matches these values, not arbitrary Unicode. Y allocation/growth/free are timed. C# in-batch GC is timed; deferred reclamation is excluded. Generation-count deltas are not independent collection counts. Raw allocation/GC fields are retained.

| Cold stage | Selected-address Y ms | Edge-trained Y ms | C# ms |
| --- | ---: | ---: | ---: |
| Final compilation / IL preparation | 219.240 | 219.790 | 2.958 |
| First-call set | 0.019 | 0.019 | 1.350 |
| Launch to JSON | 316.862 | 309.633 | 228.226 |

Y preparation starts from source and charges instrumented compilation, measured training/return storage, snapshots/profile formatting, final recompilation, input setup and trainer disposal. The saved first warm pair also copies instrumented IR within preparation; file writes follow. C# starts from prebuilt IL and uses PrepareMethod; source-to-IL build is outside that timer. Runtime startup/formatting belong launch-to-JSON. Neither comparison establishes a general language ranking.

Input dimensions and dispatch/output storage are unchanged from the [preceding training-tier report](cpu_jit_benchmarks_training_tier.md). Host input construction, reset/hashing/validation/JSON formatting are outside batch timers. Checksums can collide; finite float outputs use absolute/relative tolerance 1e-12. Aggregate counters do not reveal chronology; discarded standardized warmups cannot be audited. Bounded differential oracles do not prove arbitrary programs.

## Verification and durable evidence

**327 selected Rust tests and 18 Python CPU JIT tests passed after rebuilding liby.** The retained gates cover complete results, memory/object contents, IEEE behavior, exact profiles, final IR identity and concurrent native atomic observations. Detailed commands, tested configurations, reviews and actual log results are in verification.json and retained logs.

Independent [measurement audit](benchmark_data/cpu_jit_profile_edges/audit.json): **140,907 checks passed**; independent [oracle audit](benchmark_data/cpu_jit_profile_edges/verification/oracle-audit.json): **19,493 scalar checks passed**. Neither imports the runner. The durable-copy audit passes 141,209 checks. Verified 33 runner/111 frozen-source members, 2 workers, rebuilt liby, 187 runtime identities and 1452 historical files plus the session handoff. All raw pairs/losses, settings, output/training streams, source archive, exact binaries, reviews and logs are retained under [durable evidence](benchmark_data/cpu_jit_profile_edges/README.md). The evidence manifest seals every retained file except itself; publication.json hashes the new report and evidence README.

Runtime identity was probed from fresh Python CPUJit under matching discovery environment; exact live worker mappings were not sampled during timing. The snapshot is not a hermetic SDK/system/Cargo image. Saved final IR/assembly equality is checked on the first warm pair, not every module. Instrumented IR differs. Assembly is an llc O3/PIC/native CPU/small code-model reconstruction while live ORC requests JITDefault. Actual emitted object bytes are counted but not retained or hashed. The new auditor reads matching frozen sources and can validate archived workers if current binaries later change. Historical reports/evidence and both handoff files remain unchanged; no commit was made.

Reproduce in a fresh directory with no concurrent heavy jobs:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite helpers --optimization-stage profile-edges --compare-optimizations \
  --repeats 9 --cold-repeats 5 --calls 32 --warmup 12 --profile-warmups 12 \
  --cpu 0 --output build_artifacts/new-profile-edges-run
```

Rerun retained auditors only with matching frozen sources/binaries/runtime manifests; original metadata paths identify the actual working run.
