# CPU JIT verification policy and compilation costs — 2026-10-08

Opting out of LLVM verification after every individual pass reduces small-input cold preparation from **419.106 to 329.331 ms (21.42% lower)** and full-size preparation from **862.352 to 769.482 ms (10.77% lower)**. All five cold and nine warm pairs improve. Both settings keep mandatory full-module checks before optimization and after each attempted pipeline. **Per-pass verification remains enabled by default**; this is an explicit Rust option and controlled comparison, with existing C/Python compilation using the default.

The largest saving occurs in the default LLVM O3 pipeline. Profile selection is a small part of total compilation. Eager ORC materialization becomes the largest measured compilation phase in the boundary-only arm. Explicit boundary verification and cold materialization become slower; all losses are retained. The comparison measures the incremental wall-clock effect of the verification policy, not an exclusive timer for LLVM verifier work.

The run began `2026-10-07T21:00:54.394675+00:00` (October 8 in Europe/Istanbul) on AMD Ryzen 9 9950X 16-Core Processor, Linux x86-64, logical CPU 0. LLVM 23.1.1; .NET SDK 8.0.425/runtime 8.0.31. Nine rotating process triples and five cold triples ran sequentially, with no concurrent agent builds, tests or CPU audits during timing. Each warm Y worker uses twelve measured training calls per workload; every engine uses twelve standardized warmups and thirty-two timed calls per workload. Host activity and frequency remain uncontrolled beyond affinity. Small smoke results are excluded from these performance claims.

## Policy, correctness and timing scope

Both Y arms use the same frozen compiler and source unit, native host CPU/features, O3 IR and machine-code settings, rotate recognition, local query/mutation/copy paths, scalar helper effects and compact adapters. Natural-loop profile-control weights remain disabled. **Only `verify_each_pass` changes, true to false.** Both collect actual profiles and recompile. Successful instrumented compilations execute two explicit module checks; profiled compilations execute three because profile selection is attempted.

The input check precedes the default pipeline; its output is checked before optional profile selection; selection output is checked before IR capture or ORC, including the unknown-optional-pass fallback. Checks use LLVM return-status mode and recoverable failures dispose pass options. Disabling per-pass diagnosis loses checking of intermediate IR between passes. Module verification checks IR validity, not preservation of source semantics. LLVM fatal errors remain outside Rust panic recovery. Invalid post-pass IR and an LLVM build missing the optional pass were reviewed but not injected in tests.

Both arms retain exactly matching profiles, training/first/timed outputs, and byte-identical saved instrumented and final IR. Warm training records 140,308,654 outcomes over 111 raw sites and applies weights to 88; cold training records 328,727 over the same sites. Training repeats deterministically and overlaps the first twelve measurement seeds. Saved IR covers the first warm pair, not every compilation. The original unoptimized IR was not saved for independent fingerprint regeneration.

Primary compilation phases remain disjoint and sum exactly to total. Explicit full-module checks aggregate under `verification`; requested per-pass checks remain within `optimization`. The new `optimization_timings()` Rust/C/Python snapshot partitions optimization into default `pipeline`, attempted `profile_selection`, residual `other` and `total`. The nested total equals the primary optimization interval and must not be added again. Pipeline timers include LLVMRunPasses work, its pass-manager overhead and requested verification. Selection time can include an unsuccessful optional-pass lookup on compatible LLVM builds. No individual-pass timer is exposed by the currently used C pass-builder API.

## Preparation

Cold workers compile the same fifteen kernels plus three scalar source helpers and eighteen checked adapters. Small inputs are n=64, Fibonacci n=10 and two bulk appends. Full-size training uses the timed workload dimensions. Separate medians need not add to median total.

| Scope / stage | Per-pass ms | Boundary ms | Median paired saving ms | Paired range ms | Improves |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented_compile | 116.955 | 84.198 | 32.881 | 31.321–40.855 | 9/9 |
| warm / profile_collection | 507.573 | 506.718 | 1.149 | -35.879–12.487 | 6/9 |
| warm / profile_snapshot | 0.017 | 0.018 | 0.000 | -0.008–0.002 | 5/9 |
| warm / optimized_recompile | 230.738 | 177.637 | 44.402 | 12.378–61.872 | 9/9 |
| warm / prepare | 862.352 | 769.482 | 81.121 | 8.807–98.938 | 9/9 |
| cold / instrumented_compile | 140.956 | 103.787 | 36.148 | 34.976–42.899 | 5/5 |
| cold / profile_collection | 1.228 | 1.208 | 0.007 | -1.762–2.560 | 3/5 |
| cold / profile_snapshot | 0.007 | 0.006 | 0.001 | 0.001–0.001 | 5/5 |
| cold / optimized_recompile | 273.711 | 223.216 | 50.495 | 32.831–59.420 | 5/5 |
| cold / prepare | 419.106 | 329.331 | 90.200 | 71.597–95.147 | 5/5 |

Positive signed savings favor the boundary-only arm; negative values retain losses. Paired saving medians differ from subtracting per-arm medians. All ranges describe observed samples, not confidence intervals. No sample was removed.

| Cold stage | Per-pass ms | Boundary ms | C# ms |
| --- | ---: | ---: | ---: |
| Final compilation / C# IL preparation | 273.646 | 223.146 | 2.952 |
| First-call set | 0.019 | 0.020 | 1.499 |
| Process launch to JSON | 421.350 | 331.710 | 279.108 |

C# prepares prebuilt IL; source-to-IL build is outside that preparation timer. Y starts with source parsing. Launch-to-JSON includes runtime startup, formatting and other process work. Full Y preparation includes measured training, snapshots, input setup, profile/record formatting, lookups and trainer disposal. The first warm pair also copies instrumented IR within preparation; IR file writes occur afterward. Training, standardized warmup and first-call costs are excluded from steady-state batches. These compilation starting stages differ.

## Compilation costs and next bottleneck

| Scope / compilation / interval | Per-pass ms | Boundary ms | Median paired saving ms | Paired range ms | Improves |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm / instrumented / pipeline | 63.992 | 32.093 | 32.152 | 31.542–40.027 | 9/9 |
| warm / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/9 |
| warm / instrumented / optimization_other | 0.003 | 0.003 | 0.000 | -0.001–0.001 | 6/9 |
| warm / instrumented / optimization | 63.996 | 32.095 | 32.153 | 31.541–40.028 | 9/9 |
| warm / instrumented / verification | 0.276 | 0.381 | -0.106 | -0.112–-0.096 | 0/9 |
| warm / instrumented / materialization | 43.946 | 43.642 | 0.191 | -0.295–1.136 | 5/9 |
| warm / instrumented / total | 116.955 | 84.198 | 32.881 | 31.321–40.855 | 9/9 |
| warm / profiled / pipeline | 124.284 | 76.504 | 43.025 | 34.059–62.315 | 9/9 |
| warm / profiled / profile_selection | 2.920 | 2.241 | 0.668 | -2.358–2.869 | 8/9 |
| warm / profiled / optimization_other | 0.002 | 0.002 | 0.000 | -0.000–0.001 | 6/9 |
| warm / profiled / optimization | 127.622 | 78.740 | 43.693 | 31.701–62.987 | 9/9 |
| warm / profiled / verification | 0.838 | 1.061 | -0.225 | -0.311–-0.182 | 0/9 |
| warm / profiled / materialization | 92.219 | 91.709 | -0.143 | -16.984–4.382 | 4/9 |
| warm / profiled / total | 230.670 | 177.563 | 44.397 | 12.384–61.875 | 9/9 |
| cold / instrumented / pipeline | 78.509 | 38.745 | 39.406 | 35.750–41.142 | 5/5 |
| cold / instrumented / profile_selection | 0.000 | 0.000 | 0.000 | 0.000–0.000 | 0/5 |
| cold / instrumented / optimization_other | 0.003 | 0.003 | 0.000 | -0.001–0.001 | 4/5 |
| cold / instrumented / optimization | 78.512 | 38.747 | 39.407 | 35.750–41.142 | 5/5 |
| cold / instrumented / verification | 0.278 | 0.399 | -0.118 | -0.937–-0.099 | 0/5 |
| cold / instrumented / materialization | 53.404 | 53.329 | 0.891 | -1.435–2.579 | 3/5 |
| cold / instrumented / total | 140.956 | 103.787 | 36.148 | 34.976–42.899 | 5/5 |
| cold / profiled / pipeline | 146.879 | 94.190 | 53.482 | 40.473–60.418 | 5/5 |
| cold / profiled / profile_selection | 2.954 | 2.330 | 0.615 | 0.552–3.177 | 5/5 |
| cold / profiled / optimization_other | 0.003 | 0.003 | -0.000 | -0.007–0.000 | 2/5 |
| cold / profiled / optimization | 150.000 | 96.765 | 54.164 | 43.650–61.033 | 5/5 |
| cold / profiled / verification | 0.857 | 2.319 | -1.439 | -3.060–-0.259 | 0/5 |
| cold / profiled / materialization | 113.046 | 117.101 | -4.055 | -8.296–2.320 | 2/5 |
| cold / profiled / total | 273.646 | 223.146 | 50.500 | 32.401–59.421 | 5/5 |

Cold profiled default-pipeline medians fall from 146.879 to 94.190 ms; selection falls from 2.954 to 2.330 ms. Explicit verification rises from 0.857 to 2.319 ms and materialization from 113.046 to 117.101 ms. Their causes are not established; identical saved IR does not establish identical allocation, cache state, frequency or live code placement. Warm profiled total's ratio of per-arm medians is 1.29908, while its paired median ratio is 1.24905. Warm materialization medians suggest a small gain, while its paired median is a loss. All explicit verification pairs regress; five of nine warm profiled materialization pairs and three of five cold pairs regress.

In the boundary-only cold profiled arm, median per-sample shares are **51.76% materialization**, **41.87% default pipeline** and **1.06% selection**. The full-size profiled shares are 51.67%, 43.09% and 1.26%, respectively. Materialization combines eager whole-module native generation, linking and lookup of every public entrypoint and adapter. Dropping some lookups would not establish a code-generation saving for this whole-module LLJIT design.

The next measured target is eager native materialization, followed by the default O3 pipeline. Evaluate a lower-cost initial tier or actual ORC module/function partitioning against native-call amortization and full-result gates. Sharing adapters by ABI shape is another bounded experiment. Selection is too small here to be the first target. Per-pass diagnosis remains the default; deciding to change it requires an explicit diagnostics policy. This run does not establish lazy/tiered correctness or end-to-end application benefit. Cross-report cold differences cannot isolate policy cost because the old helper report predates the additional boundary checks and machine state differs.

## Native execution and retained regressions

Times below are medians in ms/call. Ratios divide per-engine medians; above one favors the boundary arm. Batches include native/managed dispatch and output storage. This diagnostic-policy increment is not intended to change executed code.

| Workload | Per-pass Y | Boundary Y | C# | Per-pass / boundary | C# / boundary | Y wins vs C# |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.593408 | 0.593506 | 1.058167 | 0.99984× | 1.78291× | 9/9 |
| recursive_fib | 0.149844 | 0.150215 | 0.291318 | 0.99753× | 1.93933× | 9/9 |
| float_recurrence | 1.093253 | 1.093199 | 1.094743 | 1.00005× | 1.00141× | 5/9 |
| indexed_memory | 0.109132 | 0.108955 | 0.170249 | 1.00163× | 1.56256× | 8/9 |
| unsigned_mix | 0.455258 | 0.455084 | 0.456522 | 1.00038× | 1.00316× | 4/9 |
| float_dot | 0.096055 | 0.095819 | 0.096340 | 1.00246× | 1.00544× | 4/9 |
| short_circuit | 1.806054 | 1.800603 | 1.924902 | 1.00303× | 1.06903× | 9/9 |
| binary_search | 1.437821 | 1.438972 | 1.436440 | 0.99920× | 0.99824× | 2/9 |
| string_scan | 0.038001 | 0.038311 | 0.259842 | 0.99193× | 6.78250× | 9/9 |
| vec_scan_append | 0.063475 | 0.063935 | 0.091522 | 0.99281× | 1.43149× | 9/9 |
| vec_dynamic_byte | 0.063540 | 0.063720 | 0.101425 | 0.99717× | 1.59173× | 9/9 |
| vec_dynamic_i64 | 0.039859 | 0.040063 | 0.109996 | 0.99490× | 2.74555× | 9/9 |
| string_bulk_append | 0.029147 | 0.029129 | 0.258544 | 1.00059× | 8.87573× | 9/9 |
| string_scan_helper | 0.036092 | 0.036245 | 0.259423 | 0.99579× | 7.15752× | 9/9 |
| vec_scan_append_helper | 0.061258 | 0.061022 | 0.093842 | 1.00388× | 1.53784× | 9/9 |

| Workload | Per-pass / boundary paired median | Range | Y improves | C# / boundary paired median | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.99939× | 0.79752–1.01785 | 4/9 | 1.77253× | 1.64040–1.80258 |
| recursive_fib | 0.99113× | 0.92889–1.53571 | 3/9 | 1.93811× | 1.52449–2.12301 |
| float_recurrence | 0.99915× | 0.90146–1.02187 | 3/9 | 1.00009× | 0.94921–1.04134 |
| indexed_memory | 1.00204× | 0.68050–1.02441 | 6/9 | 1.57227× | 0.96030–2.26019 |
| unsigned_mix | 1.00128× | 0.89279–1.02972 | 5/9 | 0.99880× | 0.97360–1.02023 |
| float_dot | 0.99963× | 0.56029–1.74495 | 4/9 | 0.99961× | 0.56596–1.76719 |
| short_circuit | 1.00279× | 0.89348–1.02842 | 6/9 | 1.06317× | 1.04626–1.11965 |
| binary_search | 0.99950× | 0.88980–1.03378 | 4/9 | 0.99414× | 0.94315–1.00265 |
| string_scan | 1.00025× | 0.32985–1.04209 | 5/9 | 6.78683× | 2.25565–8.52886 |
| vec_scan_append | 0.99435× | 0.98007–1.02397 | 1/9 | 1.43734× | 1.39911–3.54525 |
| vec_dynamic_byte | 0.99881× | 0.94467–1.01097 | 2/9 | 1.58130× | 1.49452–1.62924 |
| vec_dynamic_i64 | 0.99516× | 0.39179–1.02161 | 3/9 | 2.75005× | 1.05987–5.14068 |
| string_bulk_append | 1.00063× | 0.96948–1.06757 | 5/9 | 8.80867× | 8.54855–11.19288 |
| string_scan_helper | 0.99567× | 0.97229–1.01818 | 2/9 | 7.08175× | 6.86946–9.08167 |
| vec_scan_append_helper | 1.00180× | 0.98902–2.16951 | 6/9 | 1.53045× | 1.47583–2.66764 |

Measured boundary-arm native median regressions: `integer_branch` +0.016%; `recursive_fib` +0.248%; `binary_search` +0.080%; `string_scan` +0.814%; `vec_scan_append` +0.725%; `vec_dynamic_byte` +0.284%; `vec_dynamic_i64` +0.513%; `string_scan_helper` +0.423%.

These small changes have no demonstrated cause or statistical significance. All paired ratios and helper/direct distributions remain in the audited summary. Direct kernels precede helpers in fixed within-process order; allocator, GC, cache and frequency state can affect their ratios. Byte-identical saved IR and matching reconstructed assembly do not expose actual ORC addresses/code placement; `.s` files were produced by `llc`, not dumped from the running JIT.

## Runtime and comparison limitations

| C# workload | Median allocated bytes / 32 calls | Gen0/Gen1/Gen2 count deltas summed over nine batches |
| --- | ---: | ---: |
| string_scan | 1575936 | 0/0/0 |
| vec_scan_append | 1059584 | 0/0/0 |
| vec_dynamic_byte | 1059584 | 0/0/0 |
| vec_dynamic_i64 | 8398592 | 9/9/9 |
| string_bulk_append | 1582592 | 0/0/0 |
| string_scan_helper | 1575936 | 0/0/0 |
| vec_scan_append_helper | 1059584 | 0/0/0 |

Y uses explicit measured PGO; C# Release .NET 8 has tiering, dynamic PGO and ReadyToRun disabled, with NoInlining kernel entries and ordinary scalar helper inlining policy. Y byte Strings and C# UTF-16 StringBuilder differ in representation, growth and allocation. Y Vec, List<byte> and List<long> also differ; List<byte> stores bytes. ASCII equates tested values, not arbitrary Unicode semantics. Y allocation/growth/free are inside every call. C# GC inside batches is charged, deferred reclamation is uncharged. `[1,1,1]` is a generation-count delta: a collection covering all generations contributes to all counters, rather than three independent collections. These are equivalent-algorithm results with different optimization/runtime policies. .NET 10/dynamic-PGO and matched raw-buffer controls remain future work.

Full inputs: integer/unsigned/logical 250,000 rounds; Fibonacci 25/26; recurrence 1,000,000 steps; indexed memory 262,144 operations on 65,536 I64s; dot 262,157 products over two 65,536-element arrays; binary search 20,000 queries on 65,536 sorted elements; String/Vec/dynamic vectors 16,384 elements; bulk String 512 appends of 32 characters. Object scans make eight passes. String/byte-vector scans include deliberately invalid indices returning zero; I64-vector scans use valid indices. Host arrays, reset/hash, validation and JSON formatting are outside batch timers. Worker outputs use checksums rather than complete buffers/objects, so collisions remain possible. Floating outputs use finite values with absolute/relative tolerance 1e-12. Side effects are aggregate counts. Discarded standardized warmup outputs cannot be audited; training overlaps measurement seeds and is not held-out validation. The suite cannot establish a general language ranking or arbitrary-program equivalence.

## Verification and durable evidence

**319 selected Rust tests and 17 Python CPU JIT tests passed** after rebuilding `liby`. The differential gate now covers eight O0/O3 × custom-transform × verification configurations against independent full-result oracles. Additional ordinary, instrumented and profiled policy checks compare full buffers, object/free scenarios, exact profiles/IR and phase accounting. Timing getters test C ownership, null handles, session/thread rules and snapshot independence. These bounded gates are not optimizer proofs.

The independent [measurement audit](benchmark_data/cpu_jit_verification/audit.json) passed **129,945 checks**; the separate [oracle audit](benchmark_data/cpu_jit_verification/verification/oracle-audit.json) passed **19,493 scalar checks**. Neither imports the runner. They validate all 27 warm and 15 cold processes, every retained first/timed/training result, raw/aggregate/stdout consistency, profiles, optimization flags, nested and primary timing partitions, paired summaries and source/binary/runtime hashes. All 33 runner sources, 108 frozen source/archive entries, two workers, 187 runtime identities and 439 historical report/data files match. The Python library was explicitly identified and hashed.

[Raw evidence](benchmark_data/cpu_jit_verification/metadata.json), [verification logs/reviews](benchmark_data/cpu_jit_verification/verification/verification.json), [frozen sources](benchmark_data/cpu_jit_verification/verification/frozen-sources.tar.gz), [compiled worker/library snapshot](benchmark_data/cpu_jit_verification/verification/measured-binaries.tar.gz) and [final hashes](benchmark_data/cpu_jit_verification/verification/final-hashes.json) are retained under `docs/benchmark_data/cpu_jit_verification/`. The source/runtime snapshot is broader than the runner list but is not a hermetic system/SDK/Cargo image. Exact worker library mappings were not sampled during timing; LLVM identity was probed through Python CPUJit under the same discovery environment. All historical reports/data and `CODEX_SESSION_HANDOFF.md` remain unchanged. No commit was made.

Reproduce with a fresh output directory and no concurrent heavy jobs:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /path/to/.NET8-SDK/dotnet \
  --suite helpers --optimization-stage verify-each --compare-optimizations \
  --repeats 9 --cold-repeats 5 --calls 32 --warmup 12 --profile-warmups 12 \
  --cpu 0 --output build_artifacts/new-verify-each-run
```

Rerun the retained `verification/audit_measurement.py` and `verification/audit_oracles.py` after timing, against matching frozen sources/binaries and manifests. The original run audit and its durable-copy rerun are both preserved.
