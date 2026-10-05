# GEMM scheduling experiments and matched vendor reference

Both native scheduling experiments were rejected: neither produced a repeatable completed-time improvement. `src/ptx_emitter.rs` was restored byte-for-byte to the starting source, and the release library, compiler, and comparison example were rebuilt. **This pass ships no kernel speedup.** It retains comparison tools, two GPU correctness tests, and the evidence below.

The matched cuBLAS reference took **5.28% less time at 2048³ and 6.45% less at 4096³**. This is a benchmark result on one GPU, not an integrated backend or an application speedup.

## Environment and method

RTX 4070 Ti SUPER (sm_89), driver 615.71.09, desktop applications active, clocks unlocked. GPU experiments ran serially. Archive dates use local date 2026-09-25; UTC timestamps start on September 24. No clock settings were changed.

`examples/gemm_kernel_compare.rs` exports the actual first loadable candidate from the default eight-candidate list, recording PTX/source hashes, architecture, tile, and launch geometry. Comparisons verify matching tile/launch parameters and architecture, and check PTX hashes. These experiments do not invoke adaptive tuning or override tiles.

Each process loads both kernels into one context with identical deterministic FP16 A/B inputs and separate FP32 outputs. After three seconds of alternating warmup, 16 rounds alternate which kernel runs first. Each side has a separate synchronized host-wall batch and CUDA-event batch. Compilation, allocation, transfers, and validation are outside timing. Main suites use four fresh processes per shape; batches contain 100 calls at 512³/1024³, 50 at 2048³, and 20 at 4096³.

For native comparisons, compute each round's elapsed-time reduction as `100 * (1 - after / before)`, take its median within each process, then the median across four processes. Ranges span those four process medians; they are not confidence intervals. Positive means less time, negative means slower. Rounds inside a process are not independent trials.

Repeated buffers measure steady-state reuse. Event batches can include GPU idle time from host submission, especially for small kernels. Wall timing includes submission and completion. Neither metric includes application setup or transfers. Shared desktop activity and unlocked clocks limit conclusions about small changes.

## Experiment 1: load A first and consume B as it arrives

The original compute block loads all B fragments before looping over A fragments and matrix instructions. The experiment loads A first, then loads and immediately uses B on the first A tile, reusing B for later A tiles. Addresses, precision, instruction counts, and each accumulator's K order are preserved.

| Shape | Event reduction | Process range | Wall reduction | Process range |
|---|---:|---:|---:|---:|
| 512³ | +0.03% | +0.00 to +0.04% | -0.00% | -0.04 to +0.00% |
| 1024³ | +0.17% | +0.09 to +0.98% | -0.72% | -0.90 to -0.09% |
| 2048³ | +0.24% | -0.49 to +0.38% | -0.05% | -0.27 to +0.24% |
| 4096³ | -0.05% | -0.09 to +0.08% | +0.05% | -0.18 to +0.33% |

This did not meet the keep criterion of repeatable completion-time improvement without material regression. See [raw data](comparison_raw.json) and [configuration](comparison_metadata.json). The [initial 4096 run](initial-4096.json) is exploratory and retained separately.

## Experiment 2: cache A and stream B

The second experiment caches all A fragments for one K step, then loads one B fragment at a time and consumes it across A tiles. This reduces the conceptual live operand count for some tiles but did not reduce allocated registers in the inspected assembly.

An exploratory 1024³ run appeared 3.55% faster by CUDA events but only 0.23% faster by wall time, with visible noise. Four new processes each at 1024³ and 4096³ did not confirm it:

| Shape | Event reduction | Process range | Wall reduction | Process range |
|---|---:|---:|---:|---:|
| 1024³ | +0.15% | -1.62 to +1.19% | -0.93% | -1.78 to +3.56% |
| 4096³ | -0.15% | -0.39 to +0.02% | -0.46% | -1.11 to +0.86% |

The [confirmation suite](candidate2_confirm_raw.json) is separate from the four exploratory `candidate2-screen-*.json` files; they were not pooled. This experiment was also reverted.

## Instruction evidence

One additional 4096³ Nsight Compute capture attributes 179,535 of 179,617 sampled math-pipeline-throttle stalls to `HMMA.16816.F32` Tensor Core instructions. Tensor pipeline activity is about 45% of elapsed cycles, compared with 3.3% for ALU and 2.1% for FMA. These are diagnostic counters, not a speedup prediction or percentages of application runtime. See [raw counters](before-raw.csv) and [instruction samples](before-sass.csv).

Offline `ptxas` 13.4.92 assembly for sm_89 helps explain the result:

- At 4096³, all three variants use 220 registers, 960 instructions, and no spills. Starting at the first shared-memory matrix load, the original and experimental instruction streams, including encoded scheduling control, are identical. Some earlier setup instructions differ.
- At 1024³, experiment 2 and the original use 146 registers, 688 instructions, and no spills. Their instruction streams are identical after swapping two address-register names; scheduling-control words also match.

This is offline assembler evidence, not proof that the CUDA driver's separate PTX JIT generated identical binaries. It is consistent with the measured absence of improvement. Detailed comparisons: [experiment 1](offline/comparison.json), [experiment 2](offline2/comparison.json), with resource logs and instruction diffs alongside them.

## Matched cuBLAS reference

`tools/benchmark_gemm_vendor.py` uses `cublasGemmEx`, the default heuristic, FP16 inputs, FP32 output and computation, alpha 1, beta 0, and reduced-precision reduction disabled. Row-major multiplication is mapped through swapped column-major operands. Both implementations use the same preallocated input/output buffers and default stream. This is **not an exhaustive cuBLASLt search**.

Each process warms both implementations for three seconds, then runs ten alternating paired CUDA-event rounds with 20 calls per batch. There are four processes per shape. For each process, take the median paired ratio `r = Y_time / cuBLAS_time`, then compute `100 * (1 - 1/r)` as elapsed-time reduction. The table summarizes those four reductions.

| Shape | cuBLAS elapsed-time reduction | Process range |
|---|---:|---:|
| 2048³ | 5.28% | 4.26 to 8.35% |
| 4096³ | 6.45% | 6.13 to 6.68% |

Every run had zero full-output relative L2 difference between Y and cuBLAS and passed 128 independent CPU FP64 reference samples (maximum sampled relative L2 below 7.01e-6). This establishes agreement for these inputs, not every possible input. The eight `vendor-*-trial*.json` files contain all pairs, versions, flags, hashes, validation, and GPU snapshots. cuBLAS reported version 130800.

The Python/ctypes host loop can contribute submission gaps; these are large, repeatedly reused matrices on one GPU. This pass adds only an offline benchmark, with no embedding API or production vendor dependency.

## Correctness and final state

Native comparisons check every output for bitwise equality and nonfinite values before and after timing. Independent FP64 CPU references cover all outputs up to 4,096 outputs, otherwise 128 samples. All comparisons passed, with maximum sampled relative L2 below 6.47e-6. Experiment 1 also passed 1×16×16, 31×48×80, and 65×96×128 checks. The last case uses sampled CPU validation but full before/after equality.

The shared compute helper also serves fused SwiGLU. Two new ignored GPU tests check full CPU references at 128×256×32 and 256×128×128, covering synchronous and pipelined paths and poisoning outputs before both launches. Both passed on experiment 1 and on the restored original emitter. Unsupported SwiGLU ragged shapes remain outside this test's scope.

Final restored-source validation:

- 198 library tests passed: [log](final-lib.log).
- 16 GPU tests passed, including both new SwiGLU cases: [log](final-gpu.log).
- Release library, `Y`, and comparison example built: [log](final-build.log).

Original emitter SHA-256: `bb219378c40c2ef25141b9091835d3d224bc69ebc70d3733b24ad7286f3b145e`.
[Final provenance](final_provenance.json) records source/binary hashes and exact restoration. [Summary JSON](summary.json) preserves unrounded statistics. The rejected [first patch](candidate.patch) and [second patch](candidate2.patch) are not applied to working source.

## Reproducing

The archived `before/`, `candidate/`, and `candidate2/` directories contain the exact PTX and manifests. Run from the repository root:

```sh
cargo build --offline --release --example gemm_kernel_compare
target/release/examples/gemm_kernel_compare compare docs/benchmarks/gemm_scheduling_2026-09-25/before docs/benchmarks/gemm_scheduling_2026-09-25/candidate 4096 4096 4096 20 16
python3 tools/benchmark_gemm_vendor.py docs/benchmarks/gemm_scheduling_2026-09-25/before/4096_4096_4096.manifest --rounds 10 --iterations 20 --output target/vendor-repeat.json
```

The vendor tool requires NumPy, a CUDA driver, and cuBLAS already installed. Run GPU measurements serially on matching hardware. To export current kernels, including for a different architecture:

```sh
target/release/examples/gemm_kernel_compare export target/gemm-export 4096 4096 4096
```

Further native optimization needs to change effective machine instructions, resource usage, or memory traffic and then demonstrate a repeatable benefit. These results do not justify more adaptive-selection machinery on their own.
