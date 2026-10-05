# Current 4096³ GEMM profiler observations

One Nsight Compute capture of the current FP16-input, FP32-output baseline GEMM
on the RTX 4070 Ti SUPER. This identifies optimization experiments; it does not
measure a before/after improvement. GPU clocks and caches were left uncontrolled,
and the GPU was shared with existing applications. Profiling changes execution:
its duration and the instrumented worker timings must not be compared with the
previous unprofiled benchmarks as evidence of a speedup.

| Observation | Value |
|---|---:|
| Threads per block | 128 |
| Grid | 32 × 32 |
| Registers per thread | 220 |
| Dynamic shared memory per block | 37,888 bytes |
| Additional driver shared memory per block | 1,024 bytes |
| Block limit from registers / shared memory | 2 / 2 |
| Theoretical / achieved occupancy | 16.67% / 16.24% |
| SM throughput | 44.96% |
| L2 throughput | 60.62% |
| DRAM throughput | 21.61% |
| L2 hit rate | 95.12% |
| Reported local/shared spill requests | 0 / 0 |

Nsight's rule output attributes about 63.8% of average warp cycles between issued
instructions to execution-pipeline availability stalls. That is a signal to inspect
instruction scheduling and the instruction mix, not proof of one specific pipe
being the sole limiter. Registers and shared memory jointly limit residency;
lowering only registers need not increase the number of resident blocks. Higher
occupancy also does not guarantee higher GEMM performance because smaller warp
tiles can reduce operand reuse.

The implementation already has Tensor Core MMA, operand reuse, vectorized async
copies, multistage buffering, padding, and L2 block swizzling. Historical changes
already addressed excessive CTA barrier overhead; that old diagnosis should not
be presented as an unfixed current issue.

The next useful experiment is a compact shared-memory layout and/or different
load/MMA scheduling, measured against the same baseline with numerical checks.
A matched cuBLASLt reference (same FP16 inputs, FP32 output/accumulation, layout,
working set, and synchronization) should establish practical performance
headroom first. An offline sweep of all supported candidates can test whether
the bounded adaptive search misses a winner before expanding its runtime cost.

`profile.csv` contains the raw metric and rule output, `metadata.json` records
the exact command, tool version, binary/source hashes, and extracted metrics,
and `worker.json` records the instrumented worker output. Its initial/final
64-sample numerical checks passed at relative L2 approximately 6.75e-6.

Metric interpretation reference: [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
GEMM reference API: [NVIDIA cuBLASLt documentation](https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api).
