# Y CPU JIT versus C#

Measured 2026-10-06T15:05:14.813000+00:00 on AMD Ryzen 9 9950X 16-Core Processor (Linux-7.2.8-2-cachyos-x86_64-with-glibc2.44).
One CPU pinned: [0]. 9 independent process pairs, 32 calls per timed batch, 12 warmup calls per kernel.

LLVM: 23.1.1. C# runtime: .NET 8.0.31.

Y was faster on integer branching (1.97x), recursive Fibonacci (1.86x) and indexed memory (1.59x). C# was 1.90x faster on the floating-point recurrence.

| Kernel | Y median ms/call | C# median ms/call | C# / Y | Paired ratio range |
| --- | ---: | ---: | ---: | ---: |
| integer_branch | 0.5848 | 1.1498 | 1.97x | 1.95–1.97x |
| recursive_fib | 0.1548 | 0.2881 | 1.86x | 1.86–1.87x |
| float_recurrence | 2.0567 | 1.0813 | 0.53x | 0.52–0.53x |
| indexed_memory | 0.1055 | 0.1678 | 1.59x | 1.56–1.62x |

A ratio above 1 means Y was faster for that kernel. The range is the minimum and maximum ratio across paired runs; it is descriptive, not a confidence interval.

| Cold measurement | Y median ms | C# median ms |
| --- | ---: | ---: |
| JIT preparation | 23.670 | 0.730 |
| First calls (small inputs) | 0.001 | 0.001 |
| Process launch to JSON result | 25.241 | 196.638 |

Y preparation includes source parsing, type checking, LLVM optimization and materialization of all four functions. C# preparation uses RuntimeHelpers.PrepareMethod on the four already-built IL methods; its source-to-IL build is excluded. These preparation times have different starting points. Launch-to-result includes runtime startup and JSON formatting, and uses small kernel inputs.

The Y compiler uses LLVM O3 and the host CPU target. C# uses Release compilation on .NET 8, with tiering disabled so the measured methods receive optimized JIT code immediately. ReadyToRun is disabled. Kernel inputs are passed at runtime; all outputs are saved and checked against independent Python implementations. The memory array is reset before each batch, and the entire final array is checked with a 64-bit hash. Float outputs use relative and absolute tolerances of 1e-12.

The memory comparison uses unchecked pointer indexing in both languages. Allocations, array initialization, memory hashing and JSON formatting occur outside timed kernel batches. The timed batches include the host loop, output stores and function-call boundaries. C# kernel methods use NoInlining to preserve those boundaries. Recursive algorithms execute their own recursion and may be optimized differently by each JIT.

Inputs: integer n=250000; Fibonacci n=25 or n+1; float n=1000000; indexed memory n=262144 over 65536 I64 elements.

Each pair alternates which engine runs first. Kernels run in the same order in each process. No benchmark processes run concurrently. CPU affinity reduces migration, but other host activity and frequency changes can still affect results. These four synthetic workloads establish performance on this machine; they do not establish a general language speed ranking.

Raw process output, individual durations, full result arrays, correctness references, source SHA-256 hashes, binary hashes and tool versions are preserved beside this report.

Reproduce on this machine:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /tmp/y-csharp-dotnet-sdk/dotnet
```

For another machine, provide its .NET 8 SDK executable. The [benchmark guide](../benchmarks/cpu_jit/README.md) describes all inputs, flags and prerequisites. The C# build used SDK 8.0.425 without NuGet packages.

Durable evidence: [summary](benchmark_data/cpu_jit/summary.json), [metadata and hashes](benchmark_data/cpu_jit/metadata.json), [all measured batches](benchmark_data/cpu_jit/samples.json), [cold samples](benchmark_data/cpu_jit/cold.json), [independent references](benchmark_data/cpu_jit/references.json), [raw process output](benchmark_data/cpu_jit/process-output.json), and [optimized LLVM IR](benchmark_data/cpu_jit/optimized-kernels.ll). Release build logs are saved in the same directory.
