# Y CPU JIT versus C# — expanded CPU workloads

Measured 2026-10-06T15:29:11.160603+00:00 on AMD Ryzen 9 9950X 16-Core Processor (Linux-7.2.8-2-cachyos-x86_64-with-glibc2.44).
One CPU pinned: [0]. 9 independent process pairs, 32 calls per timed batch, 12 warmup calls per kernel.

LLVM: 23.1.1. C# runtime: .NET 8.0.31.

Y was faster on integer branching (1.75x), recursive Fibonacci (1.86x) and indexed memory (1.60x). C# was faster on the floating-point recurrence (1.90x) and unsigned mixing (1.10x). F64 dot reduction was effectively tied; short-circuit logic and binary search had small Y advantages of 2.3% and 1.4%, respectively. Treat those small differences as near parity.

| Kernel | Y median ms/call | C# median ms/call | C# / Y | Paired ratio range |
| --- | ---: | ---: | ---: | ---: |
| integer_branch | 0.5872 | 1.0253 | 1.75x | 1.73–1.76x |
| recursive_fib | 0.1549 | 0.2887 | 1.86x | 1.86–1.93x |
| float_recurrence | 2.0636 | 1.0847 | 0.53x | 0.52–0.53x |
| indexed_memory | 0.1054 | 0.1681 | 1.60x | 1.57–1.64x |
| unsigned_mix | 0.4972 | 0.4516 | 0.91x | 0.90–0.91x |
| float_dot | 0.0950 | 0.0949 | 1.00x | 0.99–1.00x |
| short_circuit | 1.7767 | 1.8180 | 1.02x | 1.02–1.03x |
| binary_search | 1.4265 | 1.4464 | 1.01x | 1.01–1.02x |

A ratio above 1 means Y was faster for that kernel. The range is the minimum and maximum ratio across paired runs; it is descriptive, not a confidence interval.

| Cold measurement | Y median ms | C# median ms |
| --- | ---: | ---: |
| JIT preparation | 63.179 | 0.955 |
| First calls (small inputs) | 0.008 | 0.008 |
| Process launch to JSON result | 65.306 | 222.125 |

Y preparation includes source parsing, type checking, LLVM optimization and materialization of the selected functions and their helpers. C# preparation uses RuntimeHelpers.PrepareMethod on the corresponding already-built IL methods; its source-to-IL build is excluded. These preparation times have different starting points. Launch-to-result includes runtime startup and JSON formatting, and uses small kernel inputs.

The Y compiler uses LLVM O3 and the host CPU target. C# uses Release compilation on .NET 8, with tiering disabled so the measured methods receive optimized JIT code immediately. ReadyToRun is disabled. Kernel inputs are passed at runtime; all outputs are saved and checked against independent Python implementations. The memory array is reset before each batch, and the entire final array is checked with a 64-bit hash. The expanded short-circuit workload checks all three side-effect counters exactly. Unsigned outputs use explicit modulo-2^64 Python arithmetic and include values above I64::MAX. Float outputs use relative and absolute tolerances of 1e-12.

The memory comparison uses unchecked pointer indexing in both languages. Allocations, array initialization, memory hashing and JSON formatting occur outside timed kernel batches. The timed batches include the host loop, output stores and function-call boundaries. C# kernel methods use NoInlining to preserve those boundaries. Recursive algorithms execute their own recursion and may be optimized differently by each JIT.

Inputs: integer n=250000; Fibonacci n=25 or n+1; float n=1000000; indexed memory n=262144 over 65536 I64 elements.

Each pair alternates which engine runs first. Kernels run in the same order in each process. No benchmark processes run concurrently. CPU affinity reduces migration, but other host activity and frequency changes can still affect results. These synthetic workloads establish performance on this machine; they do not establish a general language speed ranking.

Raw process output, individual durations, full result arrays, correctness references, source SHA-256 hashes, binary hashes and tool versions are preserved beside this report.

Expanded inputs: unsigned mix n=250000; F64 dot reduction n=262157 over two 65536-element arrays; short-circuit n=250000; binary search n=20000 queries over 65536 sorted I64 elements.

The dot inputs use exact binary fractions, and reduction order is the same in Y and C#. The unsigned kernel deliberately wraps arithmetic and uses only legal constant shifts. Short-circuit RHS helpers increment counters, so eager evaluation is detected. Binary search includes present and missing values; its independent oracle uses Python's bisect.

The current Y API also compiles checked scalar/pointer invocation wrappers. These are included in source preparation, while the timed batches call native entrypoints directly. This comparison does not measure Python interop or dynamic argument marshalling, and does not use the optional compilation cache. Each worker process compiles its source afresh.

Reproduce this expanded suite on this machine:

```sh
python3 tools/benchmark_cpu_jit.py --dotnet /tmp/y-csharp-dotnet-sdk/dotnet --suite expanded
```

For another machine, provide its .NET 8 SDK executable. The [benchmark guide](../benchmarks/cpu_jit/README.md) describes all inputs, flags and prerequisites. The C# build used SDK 8.0.425 without NuGet packages. `--suite original` runs the four preserved original source definitions with the current compiler. The [original measured report](cpu_jit_benchmarks.md) and its data are preserved separately; these new measurements used a larger suite and the expanded JIT implementation.

Durable evidence: [summary](benchmark_data/cpu_jit_expanded/summary.json), [metadata and hashes](benchmark_data/cpu_jit_expanded/metadata.json), [all measured batches](benchmark_data/cpu_jit_expanded/samples.json), [cold samples](benchmark_data/cpu_jit_expanded/cold.json), [independent references](benchmark_data/cpu_jit_expanded/references.json), [raw process output](benchmark_data/cpu_jit_expanded/process-output.json), and [optimized LLVM IR](benchmark_data/cpu_jit_expanded/optimized-kernels.ll). Release build logs are saved in the same directory.
