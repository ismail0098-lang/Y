# Y CPU JIT optimization comparison with C#

Measured 2026-10-06T16:03:09.799419+00:00 on AMD Ryzen 9 9950X 16-Core Processor; pinned to [0].
LLVM 23.1.1, .NET 8.0.31. 9 independent interleaved triples; 32 calls per timed batch after 12 standardized warmups per kernel.

The targeted losses closed to parity with C#: F64 recurrence improved 1.90x versus baseline Y and unsigned mixing improved 1.10x. Optimized Y retained larger advantages on integer branching, recursion and indexed memory. Small differences in dot, short-circuit logic and binary search remain near parity. Profiling also produced modest regressions versus baseline Y: indexed memory slowed about 2.2%, and binary search about 2.6%.

Baseline Y uses the final compiler with rotate recognition disabled and no branch profiles. Optimized Y enables rotate recognition, compiles instrumented code, executes 12 training calls per kernel, snapshots actual branch counts, then recompiles the same source with those measured profiles. Its timed code contains no profiling probes.

| Kernel | Baseline Y ms/call | Optimized Y ms/call | C# ms/call | Y improvement | C# / optimized Y | Paired C# / Y range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| integer_branch | 0.5893 | 0.5890 | 1.0279 | 1.00x | 1.75x | 1.73–1.75x |
| recursive_fib | 0.1553 | 0.1557 | 0.2901 | 1.00x | 1.86x | 1.84–2.07x |
| float_recurrence | 2.0681 | 1.0874 | 1.0871 | 1.90x | 1.00x | 1.00–1.00x |
| indexed_memory | 0.1054 | 0.1077 | 0.1686 | 0.98x | 1.56x | 1.31–1.57x |
| unsigned_mix | 0.4979 | 0.4529 | 0.4537 | 1.10x | 1.00x | 0.99–1.01x |
| float_dot | 0.0952 | 0.0951 | 0.0953 | 1.00x | 1.00x | 0.97–1.01x |
| short_circuit | 1.7854 | 1.7826 | 1.8239 | 1.00x | 1.02x | 1.02–1.03x |
| binary_search | 1.4303 | 1.4681 | 1.4526 | 0.97x | 0.99x | 0.98–1.01x |

Ratios above 1 favor optimized Y. Paired ranges are descriptive min/max values, not confidence intervals. Small differences should be read as near parity.

| Optimized Y preparation stage | Median ms (small cold inputs) |
| --- | ---: |
| Instrumented source compilation | 65.390 |
| Measured profile collection | 0.178 |
| Profile snapshot | 0.002 |
| Optimized recompilation | 58.321 |
| Total preparation | 124.587 |

Baseline Y source-to-native compilation: 63.730 ms. C# prebuilt IL preparation: 0.958 ms. These have different starting points; C# source-to-IL building is excluded.

Training is explicit work and its cost is excluded from steady-state kernel timers. Every warm-run record also retains its full-size training cost; the cold table uses small inputs. Training uses the same input distribution as this benchmark. Profiles are observed branch frequencies, which guide code layout and selection without proving branches unreachable.

All baseline, instrumented-training and optimized outputs match independent Python implementations. Memory writes are checked with a full-array hash; all three short-circuit counters are checked exactly. Raw branch counts, fingerprints and the number of applied profile sites are preserved. Both baseline and optimized LLVM IR are saved.

C# uses Release .NET 8 code with tiering and ReadyToRun disabled, preserving its optimized non-PGO configuration. Kernel entry methods use NoInlining. Native memory indexing is unchecked in both languages. Timers include host calls and result storage; allocations, hashing and formatting are outside timed batches. Each optimized process starts fresh without the optional compilation cache.

All three variants are interleaved with balanced rotating orders and run sequentially. The standardized warmup and timed inputs match across variants. Other host activity and CPU frequency can affect measurements. These synthetic kernels do not establish a general language speed ranking.

Inputs: integer=250000, Fibonacci=25/26, recurrence=1000000, memory=262144, unsigned=250000, dot=262157, logical=250000, search=20000.

Full-size training measured **69,704,182 branch outcomes** in each optimized process, with **17 applied branch sites**. Its median collection time was **251.37 ms**, and total preparation was **375.83 ms**, including instrumented compilation and optimized recompilation. Those costs are separate from the steady-state timings. Total preparation also includes training-buffer setup, function lookup, profile/result formatting and disposal of the training JIT.

The saved native assembly explains the changes. In `unsigned_mix`, baseline code uses two integer multiplies and shift/OR per round; rotate recognition emits one multiply plus `rorx` (a rotate). In `float_recurrence`, baseline code evaluates the subtraction each round before selecting its result; measured profiling restores a conditional branch so that work executes only when needed. Profiles remain optimization hints: separate core tests cover unseen branch outcomes, infinities and NaN without changing results.

C# remains the fixed, optimized non-PGO .NET 8 configuration from the prior comparison. This experiment compares it with explicitly trained Y; it does not compare both runtimes under equivalent PGO policies. The training workload matches the measured input distribution, and its cost is real additional work.

Reproduce on this machine:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /tmp/y-csharp-dotnet-sdk/dotnet --compare-optimizations
```

For another machine, provide its .NET 8 SDK executable. The [benchmark guide](../benchmarks/cpu_jit/README.md) documents the variants, inputs and prerequisites. SDK 8.0.425 built the C# worker without NuGet packages. The [original four-kernel report](cpu_jit_benchmarks.md) and [expanded unprofiled report](cpu_jit_benchmarks_expanded.md) remain preserved separately.

Durable evidence: [summary](benchmark_data/cpu_jit_optimized/summary.json), [metadata and hashes](benchmark_data/cpu_jit_optimized/metadata.json), [all 27 timed worker records](benchmark_data/cpu_jit_optimized/samples.json), [15 cold worker records](benchmark_data/cpu_jit_optimized/cold.json), [independent references](benchmark_data/cpu_jit_optimized/references.json), [training references](benchmark_data/cpu_jit_optimized/training-references.json), [cold training references](benchmark_data/cpu_jit_optimized/cold-training-references.json), and [raw process output](benchmark_data/cpu_jit_optimized/process-output.json). Profile counts, fingerprints and applied site counts are in each optimized record.

Code artifacts: [baseline IR](benchmark_data/cpu_jit_optimized/baseline-kernels.ll), [optimized IR](benchmark_data/cpu_jit_optimized/optimized-kernels.ll), [baseline native assembly](benchmark_data/cpu_jit_optimized/baseline-kernels.s), and [optimized native assembly](benchmark_data/cpu_jit_optimized/optimized-kernels.s). Assembly was generated by LLVM 23.1.1 `llc -O3 -mcpu=native` on the measured znver5 host. LLVM version details, compilation logs and assembly logs are retained in the same directory.
