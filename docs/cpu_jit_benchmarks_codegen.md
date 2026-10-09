# CPU JIT materialization and code-generation results — 2026-10-08

Lowering only ORC machine-code optimization from O3 to O2 **does not establish a dependable preparation benefit** in this fifteen-workload comparison. Cold preparation medians increase from **425.218 to 433.309 ms (+1.90%)**. Warm medians suggest a 0.41% decrease (1025.134 to 1020.923 ms), but the paired median is a **0.661 ms loss**, with only four of nine pairs improving. Seven native workload medians regress, led by float recurrence (+7.03%) and helper Vec (+5.59%). **The default remains inherited O3 with per-pass verification enabled.**

New unchanged-buffer ORC observations narrow the materialization bottleneck: about **99.83% of cold profiled materialization occurs before the compiled-object handoff**. The post-object interval is about 0.17 ms, including observer work, linking and lookup completion. This supports investigating native object emission and preceding ORC work; it does not provide exclusive backend-pass or linker timings.

The run began `2026-10-07T21:46:49.839478+00:00` (October 8 in Europe/Istanbul) on AMD Ryzen 9 9950X 16-Core Processor, Linux x86-64, logical CPU 0; LLVM 23.1.1; .NET SDK 8.0.425/runtime 8.0.31. Nine rotating warm process triples and five cold triples ran sequentially with no concurrent agent builds, tests or CPU audits during timing. Each warm worker has twelve standardized warmups and thirty-two timed calls per workload; each Y worker additionally makes twelve measured training calls per workload. Host activity and frequency remain uncontrolled beyond affinity. Small smoke results are excluded from performance conclusions.

## Controlled implementation and observation scope

Both Y arms use the same frozen compiler, source, O3 IR pipeline and separate O3 IR-analysis target, native CPU/features, PIC relocation and requested JITDefault code model. Runtime queries/mutations/copies, scalar helper effects, rotate recognition and compact adapters are enabled; natural-loop profile-control weights are excluded. Both retain per-pass verification and mandatory full-module pipeline-boundary checks. Both eagerly materialize all entrypoints/adapters and report failures during compilation. **Only the ORC target template changes from inherited level 3 to explicit level 2.** Both settings use measured PGO.

`JitOptions.codegen_opt_level: Option<u8>` defaults to `None`, following the IR level. Rust callers can explicitly choose 0–3; invalid overrides fail before LLVM setup. The existing C/Python compilation entrypoints retain inheritance. Cache v8 includes presence and value, distinguishing explicit requests from inheritance. The C bridge/installed disassembly review confirms that the target template level transfers into ORC’s builder; the separate IR pipeline/target is unchanged. See the [core review](benchmark_data/cpu_jit_codegen/verification/review-core.md) and [LLVM/API review](benchmark_data/cpu_jit_codegen/verification/review-orc-timing.md).

The fresh default LLJIT object transform is identity. An optional three-symbol C API bundle installs a callback which records entry time and buffer size/count, returns success, and leaves the object pointer and contents unchanged. It does not copy or retain bytes or re-enter ORC. IR/initialization transforms are preserved. Its stable Box context remains owned by Engine until after LLJIT disposal, including early errors. Fixed-size bookkeeping is mutex protected and recovers poisoned locks. Missing APIs or a null layer preserve unsplit first-lookup timing. Multiple or out-of-first object events make the children null rather than manufacturing zero measurements.

The existing flat compilation JSON remains thirteen keys. `materialization_timings()` in Rust/C/Python partitions materialization into submission, first function lookup, remaining function lookups, profile-global lookup and residual other; their sum equals the nested total and primary materialization phase exactly. Optional before/after-object children partition only first lookup and must not be added again. Lookup intervals include CString construction and successful-address checks. Before-object time includes lookup dispatch, IR-layer work, native emission and preceding ORC overhead. After-object time includes callback bookkeeping, linking and lookup completion. Neither is exclusive codegen/linker time. Observer installation belongs LLVM setup; its execution is charged within materialization in both arms. Timing snapshots exclude later source execution/getters.

All 56 successful compilations observe one object, with a valid first-lookup split and 36 function lookups (18 source functions plus 18 checked adapters). Twenty-eight instrumented compilations also look up the profile global. The count of 2,016 function lookups is retained; general API programs may have different object/lookup counts. Object-file bytes include headers, sections, symbols and relocations, and do not measure allocated executable memory or instructions alone.

## Preparation and compilation tradeoffs

Signed paired saving means O3 minus O2; positive favors O2. Separate medians need not sum, and their difference is not the median paired difference. Observed ranges are not confidence intervals. Every sample and loss is retained.

| Scope / preparation stage | O3 ms | O2 ms | Paired saved median ms | Paired range ms | O2 gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented_compile | 145.546 | 142.838 | 0.069 | -8.049–16.873 | 5/9 |
| warm / profile_collection | 605.953 | 605.051 | 0.902 | -98.047–93.005 | 5/9 |
| warm / profile_snapshot | 0.017 | 0.017 | -0.001 | -1.052–0.002 | 4/9 |
| warm / optimized_recompile | 273.134 | 279.549 | 2.903 | -41.779–49.011 | 6/9 |
| warm / prepare | 1025.134 | 1020.923 | -0.661 | -132.260–146.936 | 4/9 |
| cold / instrumented_compile | 144.998 | 150.029 | -5.430 | -20.040–4.577 | 1/5 |
| cold / profile_collection | 1.204 | 1.207 | -0.001 | -0.007–3.355 | 2/5 |
| cold / profile_snapshot | 0.006 | 0.006 | -0.000 | -0.001–0.002 | 1/5 |
| cold / optimized_recompile | 277.951 | 279.430 | 1.941 | -7.645–3.994 | 3/5 |
| cold / prepare | 425.218 | 433.309 | -3.155 | -26.134–1.676 | 2/5 |

Warm full preparation's ratio of per-arm medians is 1.00412, but its paired median ratio is 0.99928. Cold full preparation's paired median loss is 3.155 ms; only two of five pairs improve. The conflicting warm summaries and wide paired ranges prevent presenting the small ratio-of-medians decrease as a dependable gain.

| Cold stage | O3 Y ms | O2 Y ms | C# ms |
| --- | ---: | ---: | ---: |
| Final source compilation / C# IL preparation | 277.882 | 279.359 | 2.974 |
| First-call set | 0.019 | 0.018 | 1.370 |
| Process launch to JSON | 427.662 | 435.834 | 291.046 |

Cold input sizes are n=64, Fibonacci n=10 and two bulk appends, but both Y arms compile the complete identical eighteen-function unit. Full Y preparation charges instrumented compilation, training/return storage, snapshots, profile formatting, recompilation, input setup and trainer disposal. The first warm pair copies instrumented IR within preparation; IR file writes occur afterward. C# starts with prebuilt IL and uses PrepareMethod; its source-to-IL build is outside the preparation timer. Launch-to-JSON includes runtime startup and formatting. These compilation scopes differ.

| Scope / compilation / phase | O3 ms | O2 ms | Paired saved median ms | Paired range ms | O2 gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented / pipeline | 78.486 | 79.984 | -0.546 | -6.801–9.754 | 3/9 |
| warm / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/9 |
| warm / instrumented / verification | 0.270 | 0.282 | -0.012 | -0.095–1.432 | 1/9 |
| warm / instrumented / materialization | 53.652 | 54.355 | -1.722 | -5.201–5.051 | 3/9 |
| warm / instrumented / total | 145.546 | 142.838 | 0.069 | -8.049–16.873 | 5/9 |
| warm / profiled / pipeline | 148.538 | 148.392 | 2.171 | -23.412–29.186 | 5/9 |
| warm / profiled / profile_selection | 2.991 | 4.162 | -0.345 | -3.013–0.107 | 3/9 |
| warm / profiled / verification | 0.912 | 0.862 | -0.000 | -0.742–0.209 | 4/9 |
| warm / profiled / materialization | 117.875 | 117.038 | 2.232 | -17.577–21.659 | 6/9 |
| warm / profiled / total | 273.067 | 279.480 | 2.902 | -41.780–49.011 | 6/9 |
| cold / instrumented / pipeline | 77.409 | 83.239 | -6.743 | -11.096–2.783 | 1/5 |
| cold / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/5 |
| cold / instrumented / verification | 0.272 | 0.273 | 0.001 | -0.017–0.018 | 3/5 |
| cold / instrumented / materialization | 53.345 | 55.521 | -0.602 | -11.983–3.422 | 1/5 |
| cold / instrumented / total | 144.998 | 150.029 | -5.430 | -20.040–4.577 | 1/5 |
| cold / profiled / pipeline | 149.069 | 149.999 | -0.224 | -9.323–2.931 | 2/5 |
| cold / profiled / profile_selection | 2.929 | 2.947 | -0.015 | -0.060–0.505 | 2/5 |
| cold / profiled / verification | 0.840 | 0.838 | 0.014 | -0.040–0.058 | 3/5 |
| cold / profiled / materialization | 117.051 | 116.900 | 0.151 | -6.408–6.389 | 4/5 |
| cold / profiled / total | 277.882 | 279.359 | 1.952 | -7.642–3.996 | 3/5 |

Instrumented materialization is slower in six of nine warm pairs and four of five cold pairs. Cold profiled materialization has a small paired median saving of 0.151 ms, with a range from −6.408 to +6.389 ms. Warm profiled total’s separate medians regress while its paired median suggests a small gain. Identical saved IR does not establish identical cache, allocator, frequency or execution placement. No cause or statistical significance is assigned to these variations.

## Materialization breakdown and next bottleneck

These medians are milliseconds. Parent and child rows are displayed together; the two first-lookup children are already included in their parent.

| Scope / kind | Submit | First lookup | Before object (child) | After object (child) | Remaining lookups | Profile lookup | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented / ORC O3 | 0.012 | 53.618 | 53.447 | 0.176 | 0.015 | 0.001 | 0.006 | 53.652 |
| warm / instrumented / ORC O2 | 0.012 | 54.321 | 54.153 | 0.175 | 0.015 | 0.001 | 0.007 | 54.355 |
| warm / profiled / ORC O3 | 0.012 | 117.842 | 117.668 | 0.173 | 0.015 | 0.000 | 0.006 | 117.875 |
| warm / profiled / ORC O2 | 0.011 | 117.006 | 116.837 | 0.170 | 0.015 | 0.000 | 0.006 | 117.038 |
| cold / instrumented / ORC O3 | 0.012 | 53.313 | 53.130 | 0.177 | 0.014 | 0.001 | 0.007 | 53.345 |
| cold / instrumented / ORC O2 | 0.012 | 55.315 | 54.379 | 0.174 | 0.016 | 0.001 | 0.006 | 55.521 |
| cold / profiled / ORC O3 | 0.011 | 117.017 | 116.846 | 0.168 | 0.016 | 0.000 | 0.006 | 117.051 |
| cold / profiled / ORC O2 | 0.011 | 116.868 | 116.705 | 0.169 | 0.015 | 0.000 | 0.006 | 116.900 |

| Scope / kind | O3 object bytes | O2 object bytes | O3 before-object / materialization | O2 before-object / materialization |
| --- | ---: | ---: | ---: | ---: |
| warm / instrumented | 23456 | 23456 | 99.6085% | 99.6195% |
| warm / profiled | 44424 | 41096 | 99.8243% | 99.8217% |
| cold / instrumented | 23456 | 23456 | 99.6194% | 99.6141% |
| cold / profiled | 45000 | 41928 | 99.8262% | 99.8272% |

In the default O3 cold profiled arm, optimization remains the largest total phase at 55.11% median per-sample share, versus 42.12% materialization. The new split places nearly all materialization before object handoff. Later function lookups and post-object work are small, so removing a few lookups is not a material code-generation saving for the current whole-module LLJIT. Compact adapters accounted for about 1.1% of the preceding run’s saved IR/llc instructions; that is not a bound on their compile cost. Sharing them would add an indirect checked-call boundary which the native workload rows do not measure.

Next investigate the default IR pipeline for overall compilation, and native emission plus pre-object ORC work within materialization. A separate backend-pass diagnostic probe, actual module/function partitioning or an explicit lower-cost initial tier needs its own full-result and amortization gates. The O2 result here does not justify a default change or demonstrate a tiering/lazy benefit. The prior VerifyEach opt-out comparison remains a separate opt-in policy; both arms here keep verification enabled. Cross-run times are not controlled policy comparisons.

## Native execution: all workloads and losses

Times are medians in ms/call; ratios divide per-engine medians. Above one favors O2 Y. Batches include host native/managed dispatch and output storage. Native code can change with codegen level even when optimized IR matches.

| Workload | O3 Y | O2 Y | C# | O3 / O2 | C# / O2 | O2 wins vs C# |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.691629 | 0.686767 | 1.221599 | 1.00708× | 1.77877× | 9/9 |
| recursive_fib | 0.152192 | 0.151961 | 0.344066 | 1.00152× | 2.26417× | 9/9 |
| float_recurrence | 1.253217 | 1.341271 | 1.269166 | 0.93435× | 0.94624× | 3/9 |
| indexed_memory | 0.108792 | 0.109372 | 0.170674 | 0.99470× | 1.56049× | 9/9 |
| unsigned_mix | 0.538986 | 0.533273 | 0.542190 | 1.01071× | 1.01672× | 4/9 |
| float_dot | 0.097378 | 0.096823 | 0.096912 | 1.00573× | 1.00092× | 5/9 |
| short_circuit | 2.124430 | 2.178012 | 2.262541 | 0.97540× | 1.03881× | 9/9 |
| binary_search | 1.728671 | 1.702036 | 1.675117 | 1.01565× | 0.98418× | 4/9 |
| string_scan | 0.038538 | 0.037725 | 0.340584 | 1.02156× | 9.02811× | 9/9 |
| vec_scan_append | 0.063737 | 0.063955 | 0.092508 | 0.99659× | 1.44644× | 9/9 |
| vec_dynamic_byte | 0.063944 | 0.061038 | 0.101429 | 1.04761× | 1.66173× | 9/9 |
| vec_dynamic_i64 | 0.040823 | 0.040105 | 0.136654 | 1.01791× | 3.40739× | 9/9 |
| string_bulk_append | 0.029198 | 0.029514 | 0.269445 | 0.98930× | 9.12938× | 9/9 |
| string_scan_helper | 0.036625 | 0.037924 | 0.333774 | 0.96573× | 8.80102× | 9/9 |
| vec_scan_append_helper | 0.061149 | 0.064566 | 0.092787 | 0.94709× | 1.43709× | 7/9 |

| Workload | O3 / O2 paired median | Range | O2 gains | C# / O2 paired median | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 1.05398× | 0.86260–1.29025 | 6/9 | 1.76955× | 1.59073–2.00581 |
| recursive_fib | 1.00579× | 0.61821–1.45031 | 5/9 | 2.01675× | 1.59153–2.55399 |
| float_recurrence | 0.96070× | 0.86101–1.25055 | 4/9 | 0.99263× | 0.91887–1.14035 |
| indexed_memory | 1.00501× | 0.53343–1.77736 | 5/9 | 1.55541× | 1.28719–2.40472 |
| unsigned_mix | 1.00599× | 0.82298–1.18764 | 6/9 | 0.99842× | 0.97750–1.19881 |
| float_dot | 1.00858× | 0.52341–2.02862 | 5/9 | 1.00408× | 0.53342–1.81673 |
| short_circuit | 1.01502× | 0.82097–1.16025 | 5/9 | 1.07664× | 1.00439–1.30872 |
| binary_search | 1.06665× | 0.82689–1.23200 | 6/9 | 0.99799× | 0.96350–1.15061 |
| string_scan | 1.02810× | 0.33602–1.45553 | 6/9 | 8.52592× | 3.08437–9.36027 |
| vec_scan_append | 0.99549× | 0.93925–1.01266 | 2/9 | 1.44146× | 1.36051–1.57878 |
| vec_dynamic_byte | 1.04659× | 1.02704–1.92927 | 9/9 | 1.64844× | 1.62126–3.41989 |
| vec_dynamic_i64 | 1.00961× | 0.99860–1.88329 | 7/9 | 3.40347× | 2.73387–5.94160 |
| string_bulk_append | 0.99844× | 0.23360–1.00359 | 4/9 | 8.79892× | 2.43606–12.66951 |
| string_scan_helper | 0.96462× | 0.52341–3.40401 | 4/9 | 8.60112× | 4.92069–9.69155 |
| vec_scan_append_helper | 0.94909× | 0.40074–1.66993 | 1/9 | 1.44416× | 0.60341–1.75806 |

O2 median regressions: `float_recurrence` +7.026%; `indexed_memory` +0.533%; `short_circuit` +2.522%; `vec_scan_append` +0.342%; `string_bulk_append` +1.082%; `string_scan_helper` +3.549%; `vec_scan_append_helper` +5.587%.

Helper Vec loses eight of nine O3/O2 pairs. Dynamic-byte Vec improves 4.54% by separate medians and in all nine pairs; retain that favorable workload observation alongside the losses. This is a workload-specific tradeoff, not a general compiler improvement. All helper/direct ratios also remain in the independently recomputed summary; their fixed within-worker order permits allocator/cache/frequency effects.

## PGO, representation, GC and compilation limits

| C# workload | Median allocated bytes / 32 calls | Gen0/Gen1/Gen2 count deltas summed over nine batches |
| --- | ---: | ---: |
| string_scan | 1575936 | 0/0/0 |
| vec_scan_append | 1059584 | 0/0/0 |
| vec_dynamic_byte | 1059584 | 0/0/0 |
| vec_dynamic_i64 | 8398592 | 9/9/9 |
| string_bulk_append | 1582592 | 0/0/0 |
| string_scan_helper | 1575936 | 0/0/0 |
| vec_scan_append_helper | 1059584 | 0/0/0 |

Y receives explicit measured PGO. C# Release .NET 8 has tiering/dynamic PGO and ReadyToRun disabled; kernel entries use NoInlining, scalar helpers ordinary inlining policy. Training overlaps the first twelve of thirty-two measured seeds and is same-distribution evaluation. Both Y arms have exactly matching profiles/training and saved instrumented/final IR. Warm training records 140,308,654 outcomes over 111 raw sites and applies 88; cold records 328,727. Original unoptimized IR is not retained to independently regenerate fingerprints.

Y byte Strings differ from UTF-16 StringBuilder. Y Vec, List<byte> and List<long> have different layouts, growth and allocators; List<byte> stores bytes. ASCII matches the tested values, not arbitrary Unicode semantics. Y allocation/growth/free occur inside calls. C# GC inside batches is charged; deferred reclamation is uncharged. [1,1,1] means generation-count deltas, not three independent collections. The source-to-native Y and prebuilt-IL C# preparation scopes differ. .NET 10/dynamic-PGO and matched raw-buffer arms remain future controls; this suite cannot establish a general language ranking.

Full inputs: integer/unsigned/logical 250,000 rounds; Fibonacci n=25/26; recurrence 1,000,000 steps; memory 262,144 operations over 65,536 I64s; dot 262,157 products over two 65,536-element arrays; search 20,000 queries over 65,536 sorted values; String/Vec/dynamic vectors 16,384 elements; bulk String 512 appends of 32 characters. Object scans make eight passes; String/byte-vector scans include invalid indices returning zero, while I64-vector indices are valid. Host arrays, resets/hashes, validation and JSON formatting are outside batch timers. Object allocation, growth and Y reclamation/in-batch managed GC are inside. Benchmark checksums can collide, floating outputs use finite values with absolute/relative tolerance 1e-12, and counters are aggregate side effects. Discarded standardized warmups cannot be audited. Separate bounded differential tests check full buffers/objects and exact IEEE cases, not arbitrary program equivalence.

## Verification and durable evidence

**322 selected Rust tests and 18 Python CPU JIT tests passed** after rebuilding `liby`. The generated differential gate covers sixteen IR O0/O3 × custom-transform × per-pass-verification × inherited/O2-codegen configurations. Additional tests exercise codegen levels 0–3 with ordinary/instrumented/profiled full-memory, recursion, IEEE payload and FMA-sensitive oracles, exact IR/profile identity and cache isolation. The callback gate checks unchanged buffer identity, poisoned-lock recovery and nonunique/outside-first observations. Missing APIs, real concurrent compilation and injected post-submission failures were reviewed rather than fully injected.

The independent [measurement audit](benchmark_data/cpu_jit_codegen/audit.json) passed **137,416 checks** and the independent [oracle audit](benchmark_data/cpu_jit_codegen/verification/oracle-audit.json) passed **19,493 scalar checks**. Neither imports the runner. All 27 warm / 15 cold records, first/timed/training streams, hash/counter results, raw/aggregate/stdout agreement, flags, profiles, flat/nested accounting, null-aware signed paired summaries, actual llc commands and final hashes pass. All 33 runner sources, 109 frozen source/archive members, two workers, rebuilt Python library, 187 runtime identities and 624 historical report/data files match.

[Raw evidence](benchmark_data/cpu_jit_codegen/metadata.json), [verification logs](benchmark_data/cpu_jit_codegen/verification/verification.json), [frozen sources](benchmark_data/cpu_jit_codegen/verification/frozen-sources.tar.gz), [measured worker/library binaries](benchmark_data/cpu_jit_codegen/verification/measured-binaries.tar.gz) and [final hashes](benchmark_data/cpu_jit_codegen/verification/final-hashes.json) are retained under `docs/benchmark_data/cpu_jit_codegen/`. Source/runtime snapshots are not a hermetic system/SDK/Cargo image. LLVM identity was probed with fresh Python CPUJit under the same discovery environment; exact live worker mappings were not sampled during timing.

The `.s` files are llc reconstructions with recorded O3/O2 levels, native CPU, PIC and small code model, whereas live target templates request JITDefault. Their settings and scope differ; they are not ORC dumps. Actual emitted object bytes are counted but not retained, hashed or inspected as live executable memory. IR equality is checked on the saved first warm pair, not every module. All historical reports/raw data and `CODEX_SESSION_HANDOFF.md` remain unchanged. Initial fixture-count, audit-regex and copied-path audit failures are retained with their corrections; neither caused replacement of the timed run. No commit was made.

Reproduce in a fresh directory with no concurrent heavy jobs:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite helpers --optimization-stage codegen --compare-optimizations \
  --repeats 9 --cold-repeats 5 --calls 32 --warmup 12 --profile-warmups 12 \
  --cpu 0 --output build_artifacts/new-codegen-run
```

Rerun retained `verification/audit_measurement.py` and `verification/audit_oracles.py` after timing against matching frozen sources, workers and runtime manifests. The original structural audit and durable-copy rerun are both retained; the latter passed **137,719 checks**, including byte/SHA256 equality of 151 original/copied measured artifacts. Metadata retains the actual original run paths. The original auditor is retained alongside the copy-aware version; the correction did not alter measured inputs or outputs.
