# Resident adaptive GEMM dispatch comparison

The confirmed change is **two Rust heap allocations per repeated GEMM launch reduced
to zero**. Paired median enqueue times improved by 2.9–4.1%, with positive enqueue
gains in all 18 process pairs. These are modest host submission savings in this
instrumented benchmark. **Completed GEMMs did not consistently get faster**: the
smallest shape had an observed median paired regression of 8.33%. The cause of
completion-time variation was not isolated, so neither dismiss that result nor
interpret these measurements as an application speedup or evidence for tuning.

Measured on NVIDIA GeForce RTX 4070 Ti SUPER; 6 separate-process pairs per shape, using deferred policy. Started 2026-09-24T20:35:38.822687+00:00.

Each worker prepares exactly one resident shape and reuses its inputs and baseline GEMM kernel. Compilation, tuning, buffer setup, correctness readback, and three seconds of warmup are outside timing. No tuning maintenance or persistent decision cache is used. This measures resident dispatch changes; it does not establish a better kernel, an adaptive tuning benefit, or application-wide speedup.

The preserved worker, PTX-emitter, empirical-tuner, and Cargo.lock hashes match between binaries. Among captured source hashes, changes are limited to the runtime files `src/adaptive_jit.rs` and `src/cuda_runtime.rs`; source snapshots and actual binary hashes are checked before measurement.

Each process records 41 batches of 1,000 launches. Tables aggregate process medians; the 41 rounds are not treated as independent process trials. Before/after order reverses on alternate trials and shape order rotates. Each pair's gain is 100 × (before − after) / before, using that pair's process medians. The reported minimum and maximum gains describe observed variation, not a confidence interval.

Every worker is pinned to logical CPU 0 using taskset. The GPU is not reserved exclusively and clocks are not locked. Existing desktop/application processes remain running; counterbalanced order cannot remove contention, scheduling, or clock noise. A positive gain range in this small synthetic sample still does not guarantee the same result in another workload.

Completed time includes final CUDA synchronization. Enqueue time ends after submitting the batch and may include driver queue blocking; it is not isolated CPU overhead. Both binaries have the same allocator wrapper, with counting disabled during timing. The separate untimed audit counts Rust allocations/reallocations and excludes CUDA driver internal allocations. Even with counting disabled, the allocator wrapper performs a flag load on each allocation. The old path executes those extra loads; the allocation-free path avoids them. Timing gains can therefore include instrumentation overhead and are not an exact production saving.

| Shape M×N×K | Before completed µs/call | After completed µs/call | Median paired gain | Pair gain range |
|---|---:|---:|---:|---:|
| 1×16×16 | 2.4484 | 2.5872 | -8.33% | -18.65% to +14.81% |
| 256×256×256 | 3.4207 | 3.3923 | +1.84% | -9.19% to +5.29% |
| 1024×1024×1024 | 39.6078 | 39.9255 | -0.03% | -5.21% to +7.08% |

| Shape | Before enqueue µs/call | After enqueue µs/call | Median enqueue gain (range) | Rust allocations/call before → after |
|---|---:|---:|---:|---:|
| 1×16×16 | 1.1485 | 1.1142 | +2.91% (+1.37% to +7.65%) | 2 → 0 |
| 256×256×256 | 1.1351 | 1.1007 | +2.93% (+1.23% to +3.81%) | 2 → 0 |
| 1024×1024×1024 | 1.1449 | 1.1000 | +4.07% (+2.59% to +5.76%) | 2 → 0 |

Every worker passed its initial and final CPU-reference sample checks; maximum relative L2 error was 2.0171241e-06 (limit 0.002). Workers sample 64 output positions; positions repeat for the smallest shape. This is sampled correctness, not exhaustive validation. Workers also verify launch-count increments and the absence of tuning/cache activity.

`raw_records.json` preserves every round, audit, process order, and worker command. `summary.json` preserves every paired process median. `metadata.json` records binary hashes, the preserved source-hash snapshots, runner hash, environment, and system/GPU information. Worker stdout and stderr are retained in `logs/`. Both binary hashes are checked again after the run. Reproduce with the recorded arguments and preserved binaries from those source states; rebuilding modified sources does not recreate the original baseline.

Validation for this change: **198 library tests and 14 serial GPU tests passed**.
The GPU tests include a resident kernel rejecting a switched CUDA context and
resuming correctly afterward; CPU driver-spy tests cover three and other argument
counts, mutable argument copies, and driver errors. Test logs are included here.
