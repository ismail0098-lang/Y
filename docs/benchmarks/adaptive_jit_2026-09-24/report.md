# Adaptive GPU JIT benchmark

Measured on NVIDIA GeForce RTX 4070 Ti SUPER. Started 2026-09-24T19:15:55.431429+00:00.

These are synthetic FP16-input/F32-output GEMMs through the Rust adaptive runtime. They compare the analytic baseline with explicit tuning and persisted decision reuse; they do not establish an application-wide speedup or compare against cuBLAS.

The current adaptive search did not demonstrate a reliable overall payoff on these
four workloads. At 100,000 calls, first-run adaptive execution was slower in all
12 fresh trials; cached execution was approximately level with the analytic
baseline in the aggregate. Tuning retained the baseline in 11 of 12 trials. The
512³ case promoted once, but that result was not consistent across independent
trials. No workload qualified for a numeric break-even projection.

This supports keeping tuning explicit and opt-in for these workloads. It does
not establish that adaptive JIT is unhelpful for every shape or application.
There were no runtime, emitter, tuning-policy, or binding changes in this
benchmark task. The baseline uses the same adaptive runtime with tuning disabled,
so the comparison isolates adaptive selection rather than measuring dispatch
overhead against the older direct CUDA launch path.

Host: AMD Ryzen 9 9950X 16-Core Processor; Linux 7.2.7-1-cachyos; NVIDIA driver
615.71.09. Three trials per mode and workload; 36 separate worker processes in
all. All numerical checks passed (largest sampled relative L2: 6.483e-6), and
nine CPU tests of the benchmark's interpretation logic passed.

Reproduce with a new output directory:

```sh
python3 tools/benchmark_adaptive_jit.py --output target/adaptive-jit-benchmark-new
```

## Method

Each measurement runs in a separate process. Baseline uses Disabled policy; first-run adaptive uses Deferred policy with a fresh decision-cache directory; cached reuses that trial's successful decision. All use the same inputs, eight-candidate budget, and 32-call tuning threshold. The three trial orders are baseline/adaptive/cached, adaptive/cached/baseline, and baseline/adaptive/cached. This partially balances the baseline position; cached follows its fresh tuning process.

Every process performs three seconds of identical baseline GPU warmup before measurement. Context creation, allocations, input upload, this common warmup, correctness readback, and later paired timing are excluded. Lifecycle totals start before runtime construction and include preparation, dispatch, synchronization at checkpoints, and the maintenance call after 32 launches. First-run tuning's own three-second ramp remains included. Maintenance wall time includes decision-cache writing.

The GPU and NVIDIA driver code cache are warm; only the application runtime and adaptive decision cache are fresh. CUDA compilation-cache behavior follows the recorded environment. Clocks are not locked and the GPU is not reserved exclusively; existing desktop and application GPU processes remain running. Partially alternated process order and paired measurements reduce drift but do not remove contention or clock noise.

Square cases reuse one weight matrix. Decode rotates six weight allocations by default (192 MiB total). Cache residency depends on the target GPU's L2 size. These are distinct memory workloads. The tuner uses its own synthetic weight rotation, which can differ from shared-weight application execution.

After the lifecycle measurement, a second Disabled runtime provides the reference for 15 paired rounds of 1,000 launches. Baseline/selected order alternates; each batch ends with synchronization. These current host microseconds per call include dispatch overhead. Statistics below aggregate per-process medians, not historical cached timing estimates. Paired gain uses the median of within-round time differences; separately aggregated baseline and selected latency columns need not subtract to that gain. The displayed saving range runs from the lowest trial p10 to the highest trial p90; it is descriptive dispersion, not a confidence interval.

## Measured lifecycle totals

Values are medians across fresh processes, in milliseconds. Each column executes the same number of application calls; tuning scratch launches are additional work.

| Shape M×N×K | Calls | Baseline ms | First adaptive ms | Cached ms |
|---|---:|---:|---:|---:|
| 256×256×256 | 1,000 | 3.357 | 3180.045 | 4.200 |
| 256×256×256 | 10,000 | 32.346 | 3208.795 | 33.071 |
| 256×256×256 | 100,000 | 320.846 | 3497.268 | 322.123 |
| 512×512×512 | 1,000 | 8.069 | 3348.482 | 8.445 |
| 512×512×512 | 10,000 | 77.689 | 3418.444 | 78.450 |
| 512×512×512 | 100,000 | 775.316 | 4114.210 | 774.189 |
| 1024×1024×1024 | 1,000 | 33.946 | 3871.897 | 34.642 |
| 1024×1024×1024 | 10,000 | 335.055 | 4174.609 | 336.009 |
| 1024×1024×1024 | 100,000 | 3352.771 | 7209.134 | 3351.910 |
| 1×4096×4096 | 1,000 | 59.661 | 3937.522 | 59.830 |
| 1×4096×4096 | 10,000 | 585.491 | 4465.776 | 586.788 |
| 1×4096×4096 | 100,000 | 5852.269 | 9835.265 | 5852.667 |

## Current steady execution

| Shape | Decision | Baseline µs | Selected µs | Paired gain | Saving p10–p90 range µs |
|---|---|---:|---:|---:|---:|
| 256×256×256 | adaptive | 3.151 | 3.145 | 0.06% | -0.324 to 0.305 |
| 256×256×256 | cached | 3.245 | 3.144 | 0.10% | -0.230 to 0.289 |
| 512×512×512 | adaptive | 7.753 | 7.715 | -0.66% | -0.460 to 0.779 |
| 512×512×512 | cached | 7.793 | 7.727 | 0.76% | -0.409 to 1.120 |
| 1024×1024×1024 | adaptive | 33.433 | 33.501 | -0.22% | -0.562 to 0.424 |
| 1024×1024×1024 | cached | 33.515 | 33.508 | 0.05% | -0.337 to 0.574 |
| 1×4096×4096 | adaptive | 58.889 | 59.312 | -0.25% | -1.172 to 0.959 |
| 1×4096×4096 | cached | 58.658 | 58.450 | 0.08% | -0.512 to 0.912 |

## Tuning cost and payoff

| Shape | Weight copies / MiB | Fresh decisions | Maintenance ms | Cache hits | Projected break-even total calls |
|---|---:|---|---:|---:|---|
| 256×256×256 | 1 / 0.1 | RetainedBaseline: 3 | 3175.920 | 3/3 | No kernel improvement to repay tuning |
| 512×512×512 | 1 / 0.5 | RetainedBaseline: 2, Tuned: 1 | 3339.841 | 3/3 | Not resolved: inconsistent promotion or insufficient measured gain |
| 1024×1024×1024 | 1 / 2.0 | RetainedBaseline: 3 | 3837.795 | 3/3 | No kernel improvement to repay tuning |
| 1×4096×4096 | 6 / 192.0 | RetainedBaseline: 3 | 3876.506 | 3/3 | No kernel improvement to repay tuning |

Break-even is an extrapolation, not an observed crossover. Starting at the largest measured call count N, let D be the median first-adaptive lifecycle time minus the median baseline time, and s the median paired saving in seconds per call. The projection is N + max(0, ceil(D/s)). It is shown only with at least three trials, promotion in every trial, positive p10 paired savings in every trial, and at least 5% aggregate measured gain. Retaining the baseline offers no kernel improvement to repay tuning; small timing differences for that case are noise. Mixed promotion decisions and unresolved gains receive no numeric projection.

Persisted runs avoid empirical tuning but still regenerate/load code and perform cache validation. Their total cost is measured directly above; a cache hit alone does not establish faster execution than the analytic baseline.

## Validation and artifacts

- 256×256×256: maximum sampled CPU-reference relative L2 error 3.794e-07; fresh candidate counts [8, 8, 8].
- 512×512×512: maximum sampled CPU-reference relative L2 error 8.870e-07; fresh candidate counts [8, 8, 8].
- 1024×1024×1024: maximum sampled CPU-reference relative L2 error 2.017e-06; fresh candidate counts [8, 8, 8].
- 1×4096×4096: maximum sampled CPU-reference relative L2 error 6.483e-06; fresh candidate counts [8, 8, 8].

Correctness checks sample outputs; they are not exhaustive equivalence proofs. Complete worker observations, including every paired timing, are in [raw_records.json](raw_records.json); aggregate calculations are in [summary.json](summary.json), and measured-source fingerprints and machine details are in [metadata.json](metadata.json). Per-worker stdout/stderr, the build log, and temporary decision-cache records remain in the ignored `target/adaptive-jit-benchmark-2026-09-24/` directory. No compiled kernels or machine-specific decision files are published with this report.
