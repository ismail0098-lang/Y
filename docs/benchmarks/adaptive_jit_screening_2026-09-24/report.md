# Adaptive GPU JIT benchmark

Measured on NVIDIA GeForce RTX 4070 Ti SUPER. Started 2026-09-24T19:53:38.126309+00:00.

These are synthetic FP16-input/F32-output GEMMs through the Rust adaptive runtime. They compare the analytic baseline with explicit tuning and persisted decision reuse; they do not establish an application-wide speedup or compare against cuBLAS.

## Effect of staged tuning

The fix reduces wasted tuning work by screening for a plausible improvement before
paying for the long clock ramp and final comparison. A short screen can only retain
a numerically checked baseline. Dispersion-only ambiguity gets one bounded retry;
a promising or unresolved challenger still receives full confirmation. The final
5% and noise thresholds are unchanged. The standalone full autotuner is unchanged.

Median tuning-maintenance wall time over three fresh processes per shape:

| Shape | Before | After | Reduction |
|---|---:|---:|---:|
| 256×256×256 | 3.176 s | 3.248 s | 2.3% slower |
| 512×512×512 | 3.340 s | 0.101 s | 97.0% |
| 1024×1024×1024 | 3.838 s | 0.101 s | 97.4% |
| 1×4096×4096 | 3.877 s | 0.098 s | 97.5% |

The short path was taken in eight of twelve fresh trials, inferred from maintenance
finishing below the mandatory three-second ramp. All three 256³ trials and one
512³ trial still performed full confirmation. This is not a hard tuning-time cap:
promising or noisy cases can still take seconds, and screening adds a small amount
of work when full confirmation is needed. The 256³ tuning cost did not improve.

All twelve fresh searches in this repeated comparison retained the baseline.
The improvement is reduced search overhead, not faster generated GEMM instructions.
The initial comparison promoted 512³ once in three trials; no repeatable kernel
speedup or reliable break-even was established here. Short screening can miss
marginal or GPU-state-dependent opportunities; it is a bounded selection policy,
not proof of optimality. Cache reuse also reuses a quick negative decision.

Un-tuned baseline execution was slower in this pass than in the initial pass
(for example, 256³ took 0.397 s versus 0.321 s for 100,000 calls). GPU clocks and
desktop activity were not controlled exclusively. Consequently, do not attribute
between-pass kernel execution differences to this change. The table above compares
measured maintenance cost; the remaining tables compare modes within the new pass.

The runtime passed 196 library tests and 14 serial GPU tests. All 36 benchmark
workers passed sampled numerical checks and all 12 persisted runs hit the cache.
The measured source hashes remained unchanged throughout this benchmark.

Reproduce with a fresh directory:

```sh
python3 tools/benchmark_adaptive_jit.py --output target/adaptive-jit-screening-new
```

[Initial benchmark](../adaptive_jit_2026-09-24/report.md) · [Before/after values](comparison.json)

## Method

Each measurement runs in a separate process. Baseline uses Disabled policy; first-run adaptive uses Deferred policy with a fresh decision-cache directory; cached reuses that trial's successful decision. All use the same inputs, eight-candidate budget, and 32-call tuning threshold. The three trial orders are baseline/adaptive/cached, adaptive/cached/baseline, and baseline/adaptive/cached. This partially balances the baseline position; cached follows its fresh tuning process.

Every process performs three seconds of identical baseline GPU warmup before measurement. Context creation, allocations, input upload, this common warmup, correctness readback, and later paired timing are excluded. Lifecycle totals start before runtime construction and include preparation, dispatch, synchronization at checkpoints, and the maintenance call after 32 launches. Any clock ramp and final search performed by adaptive tuning remain included. Maintenance wall time includes decision-cache writing.

The GPU and NVIDIA driver code cache are warm; only the application runtime and adaptive decision cache are fresh. CUDA compilation-cache behavior follows the recorded environment. Clocks are not locked and the GPU is not reserved exclusively; existing desktop and application GPU processes remain running. Partially alternated process order and paired measurements reduce drift but do not remove contention or clock noise.

Square cases reuse one weight matrix. Decode rotates six weight allocations by default (192 MiB total). Cache residency depends on the target GPU's L2 size. These are distinct memory workloads. The tuner uses its own synthetic weight rotation, which can differ from shared-weight application execution.

After the lifecycle measurement, a second Disabled runtime provides the reference for 15 paired rounds of 1,000 launches. Baseline/selected order alternates; each batch ends with synchronization. These current host microseconds per call include dispatch overhead. Statistics below aggregate per-process medians, not historical cached timing estimates. Paired gain uses within-round differences; separately aggregated latency columns need not subtract to that gain. The displayed saving range runs from the lowest trial p10 to the highest trial p90; it is descriptive dispersion, not a confidence interval.

## Measured lifecycle totals

Values are medians across fresh processes, in milliseconds. Each column executes the same number of application calls; tuning scratch launches are additional work.

| Shape M×N×K | Calls | Baseline ms | First adaptive ms | Cached ms |
|---|---:|---:|---:|---:|
| 256×256×256 | 1,000 | 3.468 | 3252.231 | 4.302 |
| 256×256×256 | 10,000 | 41.069 | 3286.614 | 38.521 |
| 256×256×256 | 100,000 | 396.757 | 3638.313 | 398.930 |
| 512×512×512 | 1,000 | 9.701 | 111.640 | 9.312 |
| 512×512×512 | 10,000 | 93.026 | 192.901 | 93.277 |
| 512×512×512 | 100,000 | 933.630 | 1015.978 | 929.550 |
| 1024×1024×1024 | 1,000 | 40.048 | 140.828 | 40.419 |
| 1024×1024×1024 | 10,000 | 403.885 | 500.738 | 422.860 |
| 1024×1024×1024 | 100,000 | 4014.706 | 4234.574 | 4116.543 |
| 1×4096×4096 | 1,000 | 75.102 | 170.987 | 71.582 |
| 1×4096×4096 | 10,000 | 725.130 | 792.261 | 712.437 |
| 1×4096×4096 | 100,000 | 7400.487 | 7189.287 | 6960.784 |

## Current steady execution

| Shape | Decision | Baseline µs | Selected µs | Paired gain | Saving p10–p90 range µs |
|---|---|---:|---:|---:|---:|
| 256×256×256 | adaptive | 3.540 | 3.689 | -0.07% | -5.490 to 2.265 |
| 256×256×256 | cached | 3.651 | 4.888 | -11.52% | -2.849 to 2.368 |
| 512×512×512 | adaptive | 9.592 | 9.686 | 0.16% | -3.005 to 2.778 |
| 512×512×512 | cached | 9.634 | 9.487 | 0.32% | -3.356 to 3.507 |
| 1024×1024×1024 | adaptive | 40.729 | 39.956 | 2.16% | -2.995 to 3.527 |
| 1024×1024×1024 | cached | 39.620 | 40.313 | -0.82% | -2.999 to 3.041 |
| 1×4096×4096 | adaptive | 69.419 | 69.374 | 0.12% | -2.569 to 2.163 |
| 1×4096×4096 | cached | 70.441 | 70.210 | -0.30% | -3.365 to 2.495 |

## Tuning cost and payoff

| Shape | Weight copies / MiB | Fresh decisions | Maintenance ms | Cache hits | Projected break-even total calls |
|---|---:|---|---:|---:|---|
| 256×256×256 | 1 / 0.1 | RetainedBaseline: 3 | 3247.790 | 3/3 | No kernel improvement to repay tuning |
| 512×512×512 | 1 / 0.5 | RetainedBaseline: 3 | 101.345 | 3/3 | No kernel improvement to repay tuning |
| 1024×1024×1024 | 1 / 2.0 | RetainedBaseline: 3 | 101.153 | 3/3 | No kernel improvement to repay tuning |
| 1×4096×4096 | 6 / 192.0 | RetainedBaseline: 3 | 97.768 | 3/3 | No kernel improvement to repay tuning |

Break-even is an extrapolation, not an observed crossover. Starting at the largest measured call count N, let D be the median first-adaptive lifecycle time minus the median baseline time, and s the median paired saving in seconds per call. The projection is N + max(0, ceil(D/s)). It is shown only with at least three trials, promotion in every trial, positive p10 paired savings in every trial, and at least 5% aggregate measured gain. Retaining the baseline offers no kernel improvement to repay tuning; small timing differences for that case are noise. Mixed promotion decisions and unresolved gains receive no numeric projection.

Persisted runs avoid empirical tuning but still regenerate/load code and perform cache validation. Their total cost is measured directly above; a cache hit alone does not establish faster execution than the analytic baseline.

## Validation and artifacts

- 256×256×256: maximum sampled CPU-reference relative L2 error 3.794e-07; fresh candidate counts [8, 8, 8].
- 512×512×512: maximum sampled CPU-reference relative L2 error 8.870e-07; fresh candidate counts [8, 8, 8].
- 1024×1024×1024: maximum sampled CPU-reference relative L2 error 2.017e-06; fresh candidate counts [8, 8, 8].
- 1×4096×4096: maximum sampled CPU-reference relative L2 error 6.483e-06; fresh candidate counts [8, 8, 8].

Correctness checks sample outputs; they are not exhaustive equivalence proofs. All worker observations and paired timings are in [raw_records.json](raw_records.json), aggregates are in [summary.json](summary.json), and hardware details and source fingerprints are in [metadata.json](metadata.json). Per-worker logs, the build log, and temporary cache records remain in the ignored `target/adaptive-jit-screening-2026-09-24/` directory.
