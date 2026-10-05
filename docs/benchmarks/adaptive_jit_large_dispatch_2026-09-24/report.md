# Resident adaptive GEMM dispatch comparison

**Neither larger shape showed a clear completion-time speedup.** The median
paired change was 0.62% slower for 2048³ and 0.12% faster for 4096³; observed pair
ranges include both gains and regressions. The optimized worker's median
completion times were about 0.273 ms and 2.129 ms, respectively. The dispatch
change removes host allocations but leaves the GPU kernel unchanged.

Measured on NVIDIA GeForce RTX 4070 Ti SUPER; 6 separate-process pairs per shape, using deferred policy. Started 2026-09-24T20:57:37.374460+00:00.

Each worker prepares exactly one resident shape and reuses its inputs and baseline GEMM kernel. Compilation, tuning, buffer setup, correctness readback, and three seconds of warmup are outside timing. No tuning maintenance or persistent decision cache is used. This measures resident dispatch changes; it does not establish a better kernel, an adaptive tuning benefit, or application-wide speedup.

The preserved worker, PTX-emitter, empirical-tuner, and Cargo.lock hashes match between binaries. Among captured source hashes, changes are limited to the runtime files `src/adaptive_jit.rs` and `src/cuda_runtime.rs`; source snapshots and actual binary hashes are checked before measurement.

Each process records 15 batches of 100 launches. Tables aggregate process medians; the 15 rounds are not treated as independent process trials. Before/after order reverses on alternate trials and shape order rotates. Each pair's gain is 100 × (before − after) / before, using that pair's process medians. The reported minimum and maximum gains describe observed variation, not a confidence interval.

Every worker is pinned to logical CPU 0 using taskset. The GPU is not reserved exclusively and clocks are not locked. Existing desktop/application processes remain running; counterbalanced order cannot remove contention, scheduling, or clock noise. A positive gain range in this small synthetic sample still does not guarantee the same result in another workload.

Completed time includes final CUDA synchronization. Enqueue time ends after submitting the batch and may include driver queue blocking; it is not isolated CPU overhead. Both binaries have the same allocator wrapper, with counting disabled during timing. The separate untimed audit counts Rust allocations/reallocations and excludes CUDA driver internal allocations. Even disabled, the allocator wrapper adds a flag load per allocation in the old path; the measured enqueue gain can include that instrumentation cost and is not an exact production saving.

| Shape M×N×K | Before completed µs/call | After completed µs/call | Median paired gain | Pair gain range |
|---|---:|---:|---:|---:|
| 2048×2048×2048 | 274.3089 | 273.1770 | -0.62% | -15.84% to +5.70% |
| 4096×4096×4096 | 2130.6624 | 2128.5130 | +0.12% | -0.57% to +3.59% |

| Shape | Before enqueue µs/call | After enqueue µs/call | Median enqueue gain (range) | Rust allocations/call before → after |
|---|---:|---:|---:|---:|
| 2048×2048×2048 | 1.3319 | 1.3070 | +1.56% (-6.39% to +6.65%) | 2 → 0 |
| 4096×4096×4096 | 1.4228 | 1.3689 | +3.54% (+2.91% to +6.90%) | 2 → 0 |

Every worker passed its initial and final CPU-reference sample checks; maximum relative L2 error was 6.752561e-06 (limit 0.002). Workers check 64 distinct sampled output positions for each of these larger shapes. This is sampled correctness, not exhaustive validation. Workers also verify launch-count increments and the absence of tuning/cache activity.

`raw_records.json` preserves every round, audit, process order, and worker command. `summary.json` preserves every paired process median. `metadata.json` records binary hashes, the preserved source-hash snapshots, runner hash, environment, and system/GPU information. Worker stdout and stderr are retained in `logs/`. Both binary hashes are checked again after the run. Reproduce with the recorded arguments and preserved binaries from those source states; rebuilding modified sources does not recreate the original baseline.

## Measurement duration and preserved evidence

The original fixed-size worker was first tried with 41 timing rounds of 1,000
calls. A 4096³ worker took 93–95 seconds, making the planned comparison unnecessarily
long. That run was stopped to shorten both workers equally; all five completed
observations are retained in `long_batch_pilot/`, including the unpaired final
observation. Its completed first pair showed 2048³ at 271.57 → 276.60 µs and
4096³ at 2118.31 → 2195.18 µs (both slower after the change). Those observations
are not pooled with the main comparison because batch sizes differ.

For the main comparison, exact before/after runtime source copies were recovered
and their hashes verified against the previously preserved executable metadata.
The same updated worker, adding optional call/round counts, was built against both
versions in one isolated build directory. Swapped runtime inputs were touched to
force Cargo to recompile rather than reuse artifacts based on copied timestamps.
The two runtime source files are the only captured source differences between
these new binaries. Original source hashes and new binary hashes are in metadata.
The allocation audit additionally confirms two Rust allocations per call before
and zero after in all workers.

The main comparison used six paired processes per shape, 15 rounds of 100 calls,
and the same three-second warmup as before. The old fixed-batch results and this
new comparison have separate metadata. `runner_used.py` and `worker_used.rs` retain
the exact main-run sources; the pilot runner is also preserved. Runtime copies
and executables remain under `target/adaptive-jit-large-dispatch-2026-09-24/`.

The runner passed 13 CPU-only protocol/argument/report checks (log included).
Both worker binaries also rejected four invalid call/round bounds before CUDA
initialization, for eight additional checks. No production runtime code changed
in this measurement pass. All 24 main workers passed initial and final sampled
numerical checks, launch accounting, and tuning/cache inactivity checks.
