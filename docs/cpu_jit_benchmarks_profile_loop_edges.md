# CPU JIT atomic loop-control edge results — 2026-10-08

**Loop-control atomic edge counters reduce median full preparation 0.28%; cold preparation falls 1.23%. Full-size training execution rises 0.23%.** Final execution remains O3.

This comparison changes only the requested `JitOptions.profile_loop_edge_counters` flag from false to true. Both arms use explicit O1 training IR, final IR/native O3, per-pass verification and eager compilation. Eligible natural-loop-control branches use fixed-address atomic increments on the selected edge; other branches and PHI-successor branches retain the preceding selected-address form. Every retained profile and output stream matches, and saved final IR and reconstructed final assembly are byte-identical. The new Rust option defaults to false; existing C/Python compilation policy is unchanged.

## Measured preparation

Times are medians in milliseconds. Signed paired saving is selected-address minus loop-edge instrumentation; positive favors loop edges. Differences of separate medians are distinct from median paired differences; medians need not sum. Ranges are observed sample ranges, not confidence intervals.

| Scope / preparation stage | Selected address ms | Fixed loop edge ms | Paired saved median ms | Paired range ms | Loop-edge gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented_compile | 92.577 | 88.197 | 4.381 | 1.944–5.416 | 9/9 |
| warm / profile_collection | 505.060 | 506.209 | -0.704 | -4.274–1.253 | 4/9 |
| warm / profile_snapshot | 0.018 | 0.019 | -0.001 | -0.002–0.002 | 3/9 |
| warm / optimized_recompile | 220.072 | 221.317 | -1.171 | -4.961–3.932 | 2/9 |
| warm / prepare | 818.535 | 816.234 | 3.554 | -4.421–8.832 | 6/9 |
| cold / instrumented_compile | 93.084 | 89.153 | 2.988 | 2.435–5.477 | 5/5 |
| cold / profile_collection | 1.203 | 1.206 | 0.000 | -0.009–0.004 | 3/5 |
| cold / profile_snapshot | 0.006 | 0.006 | 0.000 | -0.001–0.001 | 3/5 |
| cold / optimized_recompile | 218.288 | 218.846 | -0.214 | -2.168–2.223 | 2/5 |
| cold / prepare | 312.723 | 308.884 | 3.002 | 0.507–7.655 | 5/5 |

Warm preparation changes **-0.28%** by separate medians (818.535→816.234 ms). Its paired median saving is **3.554 ms**, with 6/9 pairs improving.

Warm preparation includes 3 paired regressions; its median change does not imply consistent gains across processes.

Cold preparation changes **-1.23%** by separate medians (312.723→308.884 ms). Its paired median saving is **3.002 ms**, with 5/5 pairs improving.

Measured full-size profile execution changes +0.227% by separate medians, with 4/9 pairs improving and a signed paired median saving of -0.704 ms. Training execution and all compilations are charged to preparation; no profiling-execution improvement is claimed. Final native-execution improvement is not claimed. Profile execution remains 61.99% of median per-sample full preparation in this comparison.

These are measured policy differences for this synthetic suite on one machine. All paired losses remain in the raw data and tables. Changing instrumentation does not establish a final native execution improvement or a general language ranking.

## Controls and implementation

Run start: `2026-10-08T16:08:38.531210+00:00`; AMD Ryzen 9 9950X 16-Core Processor, Linux x86-64, logical CPU 0; LLVM 23.1.1; fixed .NET 8.0.31 baseline. 9 rotating warm process triples and 5 cold triples execute sequentially, with no concurrent agent builds/tests/CPU audits during timing. Each warm worker makes 12 standardized warmups and 32 timed calls per workload; Y additionally makes 12 measured training calls. Other host activity/frequency remain uncontrolled. SDK/compiler versions and commands are retained in metadata.

`JitOptions.profile_loop_edge_counters: bool` defaults false. In Instrument mode, true adds two edge blocks to an eligible original natural-loop-control conditional branch, with one aligned monotonic atomic add to a fixed outcome counter in the selected block. A branch that is not classified as loop control, or whose original successor begins with a PHI, keeps the selected-address atomic increment; PHI incoming edges and SSA are preserved. The original terminator and metadata remain intact. Ordinary and profile-use modes ignore this requested flag. The existing `profile_edge_counters` flag can still request edges at all eligible branches; it is false in both measured arms. Cache v11 includes both flags. Existing C/Python entrypoints keep their false defaults.

Both measured arms use base IR O3, explicit training O1, native O3, VerifyEach true, mandatory full-module checks and eager whole-module materialization. Rotate, runtime query/mutation/copy, compact adapters and scalar helper effects are enabled; natural-loop control weights are excluded. Requested `profile_loop_edge_counters` is false/true in every settings record, including final compilation where it has no effect; requested `profile_edge_counters` is false throughout. Per-entry `compilation_settings` records the actual IR/native tiers and requested verification/instrumentation policy.

Fingerprint/site identities come from original lowered IR before instrumentation. The counter array still has separate true/false outcomes per original site. Each outcome wraps after 2^64 observations. Snapshot reads remain atomic and owned; a concurrent snapshot need not represent one instant across all counters. Final profile-use sessions have no training counters. Training, snapshots and recompilation remain explicit; no sampling, counter batching, lazy compilation or background tiering is introduced.

## Compilation phases

| Scope / kind / phase | Selected address ms | Fixed loop edge ms | Paired saved median ms | Paired range ms | Loop-edge gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented / pipeline | 42.448 | 41.197 | 1.110 | 0.506–1.890 | 9/9 |
| warm / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/9 |
| warm / instrumented / verification | 0.259 | 0.255 | 0.002 | -0.002–0.010 | 6/9 |
| warm / instrumented / materialization | 41.962 | 38.933 | 2.876 | 1.461–3.511 | 9/9 |
| warm / instrumented / total | 92.577 | 88.197 | 4.381 | 1.944–5.416 | 9/9 |
| warm / profiled / pipeline | 118.365 | 118.758 | -0.362 | -3.075–3.462 | 2/9 |
| warm / profiled / profile_selection | 2.900 | 2.902 | 0.001 | -0.027–0.015 | 5/9 |
| warm / profiled / verification | 0.832 | 0.834 | 0.002 | -0.029–0.015 | 5/9 |
| warm / profiled / materialization | 91.910 | 92.238 | -0.098 | -2.413–0.445 | 2/9 |
| warm / profiled / total | 220.004 | 221.251 | -1.169 | -4.963–3.936 | 2/9 |
| cold / instrumented / pipeline | 42.456 | 41.168 | 1.288 | -0.244–1.795 | 4/5 |
| cold / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/5 |
| cold / instrumented / verification | 0.256 | 0.254 | -0.001 | -0.007–0.014 | 2/5 |
| cold / instrumented / materialization | 41.872 | 39.018 | 2.973 | 2.069–3.384 | 5/5 |
| cold / instrumented / total | 93.084 | 89.153 | 2.988 | 2.435–5.477 | 5/5 |
| cold / profiled / pipeline | 117.871 | 117.961 | -0.238 | -1.063–1.084 | 1/5 |
| cold / profiled / profile_selection | 2.906 | 2.900 | 0.013 | -0.043–0.029 | 3/5 |
| cold / profiled / verification | 0.820 | 0.826 | -0.004 | -0.018–0.008 | 1/5 |
| cold / profiled / materialization | 90.983 | 91.287 | 0.171 | -1.582–1.045 | 3/5 |
| cold / profiled / total | 218.222 | 218.782 | -0.217 | -2.170–2.225 | 2/5 |

Retained separate-median compilation regressions: `warm/profiled/pipeline` +0.332%; `warm/profiled/profile_selection` +0.052%; `warm/profiled/verification` +0.334%; `warm/profiled/materialization` +0.357%; `warm/profiled/total` +0.567%; `cold/profiled/pipeline` +0.077%; `cold/profiled/verification` +0.781%; `cold/profiled/materialization` +0.334%; `cold/profiled/total` +0.257%.

Flat phases sum exactly to total. Nested optimization/materialization parents sum exactly to their primary phase; optional object-ready children partition first lookup and are already included. Before-object work includes native emission and preceding ORC work; after-object includes observer overhead/linking/lookup. Neither is an exclusive backend or linker timer. Timing snapshots exclude later execution/getters; object bytes include file metadata, not executable memory alone.

With loop-edge training, cold final profiled compilation spends 55.28% median per-sample share in optimization and 41.73% in materialization. The final O3 policy is unchanged. Earlier external opt diagnostics prioritize investigation but remain separate from measured JIT pass timings.

Scoped static artifact observations, where retained in verification, are diagnostics rather than dynamic instruction counts or measured JIT pass timings. They do not identify pass-level causes for wall-time changes.

| Scope / compilation | Selected-address observed object bytes | Loop-edge observed object bytes |
| --- | ---: | ---: |
| warm / instrumented | 21,216 | 20,360 |
| warm / profiled | 44,424 | 44,424 |
| cold / instrumented | 21,216 | 20,360 |
| cold / profiled | 45,000 | 45,000 |

All 56 measured compilations observe one object and an eligible first-lookup split, totaling 56 callbacks, 2,016 public/adapter function lookups and 28 profile-global lookups. The table reports median live ORC object-file bytes, including file metadata. Actual object contents are not retained or hashed; equal counts or sizes do not establish native content identity.

## Native execution against C# — all fifteen workloads

Both Y columns execute final O3. Times are median milliseconds per call; C#/Y ratios divide per-arm medians. Pair gains/ranges are retained even when separate medians disagree. Identical saved final IR and reconstructed assembly do not establish identical live object contents, addresses, placement, allocator/cache/frequency state. These native variations cannot be attributed to a changed final IR policy.

| Workload | Selected-address Y | Loop-edge-trained Y | C# | Y time change | C# / loop-edge-trained Y | Loop-edge-training pair gains |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.589685 | 0.590894 | 1.045030 | +0.205% | 1.76856× | 1/9 |
| recursive_fib | 0.148885 | 0.149176 | 0.290054 | +0.195% | 1.94438× | 4/9 |
| float_recurrence | 1.088618 | 1.089774 | 1.090416 | +0.106% | 1.00059× | 4/9 |
| indexed_memory | 0.107822 | 0.108178 | 0.169248 | +0.331% | 1.56453× | 3/9 |
| unsigned_mix | 0.454021 | 0.453712 | 0.454120 | -0.068% | 1.00090× | 3/9 |
| float_dot | 0.095871 | 0.095861 | 0.095962 | -0.011% | 1.00105× | 2/9 |
| short_circuit | 1.790008 | 1.789697 | 1.901656 | -0.017% | 1.06256× | 3/9 |
| binary_search | 1.432869 | 1.434213 | 1.429254 | +0.094% | 0.99654× | 3/9 |
| string_scan | 0.037845 | 0.037766 | 0.259020 | -0.208% | 6.85854× | 5/9 |
| vec_scan_append | 0.063433 | 0.063494 | 0.091041 | +0.096% | 1.43385× | 5/9 |
| vec_dynamic_byte | 0.063327 | 0.063481 | 0.099850 | +0.243% | 1.57290× | 2/9 |
| vec_dynamic_i64 | 0.039784 | 0.039956 | 0.107254 | +0.432% | 2.68430× | 3/9 |
| string_bulk_append | 0.029080 | 0.029106 | 0.256126 | +0.090% | 8.79971× | 5/9 |
| string_scan_helper | 0.035991 | 0.035993 | 0.260588 | +0.006% | 7.23986× | 5/9 |
| vec_scan_append_helper | 0.060741 | 0.060792 | 0.092258 | +0.083% | 1.51760× | 3/9 |

| Workload | Selected / loop-edge-trained paired median | Range | C# / loop-edge-trained paired median | Range |
| --- | ---: | ---: | ---: | ---: |
| integer_branch | 0.99793× | 0.99540–1.00206 | 1.77086× | 1.76028–1.79303 |
| recursive_fib | 0.99850× | 0.95714–1.00636 | 1.94516× | 1.87372–1.95393 |
| float_recurrence | 0.99949× | 0.99692–1.00959 | 1.00052× | 0.99807–1.00295 |
| indexed_memory | 0.99754× | 0.96160–1.02739 | 1.56016× | 1.52107–1.59263 |
| unsigned_mix | 0.99956× | 0.99662–1.00544 | 1.00040× | 0.99572–1.00313 |
| float_dot | 0.99714× | 0.99364–1.00743 | 1.00096× | 0.99202–1.00539 |
| short_circuit | 0.99816× | 0.99397–1.01599 | 1.06256× | 1.05341–1.07427 |
| binary_search | 0.99989× | 0.99405–1.00948 | 0.99594× | 0.98194–1.01147 |
| string_scan | 1.00461× | 0.91349–1.03200 | 6.84932× | 6.44740–6.97964 |
| vec_scan_append | 1.00017× | 0.98317–1.00378 | 1.43270× | 1.40611–1.71006 |
| vec_dynamic_byte | 0.99803× | 0.99014–1.00399 | 1.56702× | 1.54520–2.58057 |
| vec_dynamic_i64 | 0.99976× | 0.98790–1.02710 | 2.70288× | 2.65253–2.83320 |
| string_bulk_append | 1.00159× | 0.98812–1.00584 | 8.80706× | 8.73167–8.87523 |
| string_scan_helper | 1.00139× | 0.97685–1.00965 | 7.24035× | 6.83698–7.43966 |
| vec_scan_append_helper | 0.99862× | 0.95757–1.00186 | 1.51738× | 1.44475–1.74628 |

Retained native median regressions: `integer_branch` +0.205%; `recursive_fib` +0.195%; `float_recurrence` +0.106%; `indexed_memory` +0.331%; `binary_search` +0.094%; `vec_scan_append` +0.096%; `vec_dynamic_byte` +0.243%; `vec_dynamic_i64` +0.432%; `string_bulk_append` +0.090%; `string_scan_helper` +0.006%; `vec_scan_append_helper` +0.083%.

## PGO, runtime representation and preparation scope

Y uses explicit measured PGO. Training overlaps the first 12 of 32 measurement seeds, so evaluation is same-distribution rather than held out. Both arms retain exact profiles and complete returned training streams. The raw original lowered IR is not archived for independent fingerprint regeneration.

Warm profiles record 140,308,654 outcomes across 111 original sites, applying 88.

Cold profiles record 328,727 outcomes across 111 original sites, applying 88.

C# Release .NET 8 has tiering/dynamic PGO and ReadyToRun disabled. Kernel entries use NoInlining; ordinary scalar helpers keep normal inlining. Y byte Strings/Vecs differ from UTF-16 StringBuilder/List in representation, growth and allocator policy. ASCII matches these values, not arbitrary Unicode. Y allocation/growth/free are timed. C# in-batch GC is timed; deferred reclamation is excluded. Generation-count deltas are not independent collection counts. Raw allocation/GC fields are retained.

| Cold stage | Selected-address Y ms | Loop-edge-trained Y ms | C# ms |
| --- | ---: | ---: | ---: |
| Final compilation / IL preparation | 218.222 | 218.782 | 2.952 |
| First-call set | 0.018 | 0.018 | 1.346 |
| Launch to JSON | 314.842 | 311.144 | 228.439 |

Y preparation starts from source and charges instrumented compilation, measured training/return storage, snapshots/profile formatting, final recompilation, input setup and trainer disposal. The saved first warm pair also copies instrumented IR within preparation; file writes follow. C# starts from prebuilt IL and uses PrepareMethod; source-to-IL build is outside that timer. Runtime startup/formatting belong launch-to-JSON. Neither comparison establishes a general language ranking.

Input dimensions and dispatch/output storage are unchanged from the [preceding all-edge report](cpu_jit_benchmarks_profile_edges.md). Host input construction, reset/hashing/validation/JSON formatting are outside batch timers. Checksums can collide; finite float outputs use absolute/relative tolerance 1e-12. Aggregate counters do not reveal chronology; discarded standardized warmups cannot be audited. Bounded differential oracles do not prove arbitrary programs.

## Verification and durable evidence

**329 selected Rust tests and 18 Python CPU JIT tests passed after rebuilding liby.** The loop-edge gate covers training IR levels 0–3 with inherited native O3 and explicit native O1. Independent oracles check every aliased buffer word, complete String/Vec contents, IEEE bits/rounding, side-effecting short circuits, recursion and every original profile site/outcome. Ordinary/final IR identity and owned snapshot accumulation/disposal are checked. CFG fixtures exercise loop headers/self-loops, entry/interior-work fallback, PHI fallback and all-edge precedence. Four-thread gates use actual loop-control and interior-work sites with algebraic exact counts, paused snapshots and bounded concurrent snapshots for all training/native configurations. Detailed commands, reviews and actual log results are in verification.json and retained logs.

Independent [measurement audit](benchmark_data/cpu_jit_profile_loop_edges/audit.json): **141,666 checks passed**; independent [oracle audit](benchmark_data/cpu_jit_profile_loop_edges/verification/oracle-audit.json): **19,493 scalar checks passed**. Neither imports the runner. The durable-copy audit passes 141,968 checks. Verified 33 runner/111 frozen-source members, 2 workers, rebuilt liby, 187 runtime identities and 1705 historical files plus the session handoff. All raw pairs/losses, settings, output/training streams, source archive, exact binaries, reviews and logs are retained under [durable evidence](benchmark_data/cpu_jit_profile_loop_edges/README.md). The evidence manifest seals every retained file except itself; publication.json hashes the new report and evidence README.

Runtime identity was probed from fresh Python CPUJit under matching discovery environment; exact live worker mappings were not sampled during timing. The snapshot is not a hermetic SDK/system/Cargo image. Saved final IR/assembly equality is checked on the first warm pair, not every module. Instrumented IR differs. Assembly is an llc O3/PIC/native CPU/small code-model reconstruction while live ORC requests JITDefault. Actual emitted object bytes are counted but not retained or hashed. The new auditor reads matching frozen sources and can validate archived workers if current binaries later change. Historical reports/evidence and both handoff files remain unchanged; no commit was made.

Reproduce in a fresh directory with no concurrent heavy jobs:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite helpers --optimization-stage profile-loop-edges --compare-optimizations \
  --repeats 9 --cold-repeats 5 --calls 32 --warmup 12 --profile-warmups 12 \
  --cpu 0 --output build_artifacts/new-profile-loop-edges-run
```

Rerun retained auditors only with matching frozen sources/binaries/runtime manifests; original metadata paths identify the actual working run.
