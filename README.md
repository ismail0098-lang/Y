Y -----  A Systems Language and Compiler for GPU/CPU Hardware-Aware Code Generation

Y is a compiler and systems language for writing hardware-aware code across CPU (x86/AVX-512) and GPU (NVIDIA PTX) targets. It also includes a zero-knowledge circuit compiler (R1CS constraint generation) and a dual-accelerator co-processor pipeline that automatically fuses RT Core and Tensor Core workloads.

The project is under active, single-developer, ongoing development.


What this project does

Probes the actual hardware it's running on: cache latencies, AVX-512 throughput, GPU warp/tensor-core timings, and uses those measurements to make codegen decisions (e.g. choosing IMAD.WIDE over IMAD based on measured cycle cost).
Enforces compile-time safety guarantees on marked code blocks: initialized-variable checks, loop invariants, bounds declarations, and a numerical-drift check for fixed-point accumulation.
Compiles to five backends: LLVM IR (→ native binary via clang), NVIDIA PTX, portable C, direct x86-64, and a standalone ELF emitter.
Includes an R1CS constraint generator for zero-knowledge circuits, benchmarked against Circom, Noir, and Leo.
Runs a Hardware-Sentient Dual-Accelerator Scheduler: automatically fuses RT Core traversal and Tensor Core MMA pipelines, inserting sync barriers, vectorized FP32→FP16 quantization, and bank-conflict-free swizzled SMEM layouts — from a high-level description of the workload.
Is partially self-hosting: most compiler phases (lexer, parser, type checker, LLVM emitter) have been rewritten in Y itself, alongside the original Rust implementation.


Status

Bootstrap compiler (src/, Rust): stable; this is what actually runs today.
Self-hosted compiler (self_hosted/, written in Y): in progress, not yet the default build path.
Author-built with LLM assistance for implementation; architecture and design decisions are the author's own.
There is currently a backlog of automated pull requests from a connected AI coding agent (Jules) that have not yet been reviewed or merged, due to a personal medical situation. They do not reflect the current state of main.


Project Layout

src/                       Rust bootstrap compiler
  main.rs                  CLI entry point, pipeline orchestration
  lexer.rs                 Tokenizer — @-directives, GPU intrinsics
  parser.rs                Recursive-descent parser, arena-allocated AST
  ast.rs                   AST node definitions
  type_checker.rs          Semantic analysis, safety-block enforcement, linear tracker
  bank_conflict.rs         Shared-memory bank-conflict prover
  linear_tracker.rs        Tracks that async memory tokens are consumed exactly once
  sentinel.rs              Hardware probe (CPU + GPU microbenchmarks)
  avx_wrapper.rs           AVX/AVX-512 intrinsic wrappers
  llvm_emitter.rs          LLVM IR emission
  ptx_emitter.rs           NVIDIA PTX emission
  c_emitter.rs             C transpiler backend
  cpu_emitter.rs           Direct x86-64 emission
  native_emitter.rs        ELF binary emission (no external toolchain)
  ypm.rs                   Package manager
  ysu_gpu_probe.rs         External GPU microbenchmark binary
  ir_grapher.rs            IR dependency graph for RT/Tensor Core node analysis
  coprocessor_scheduler.rs Hardware-Sentient co-processor scheduler (sync barriers, SMEM budget)
  quantization_pass.rs     Vectorized FP32→FP16 quantization pass (cvt.rn.f16x2.f32)
  rt_core_emitter.rs       RT Core PTX emitter with unified coprocessor_smem offset mapping

self_hosted/          Y compiler components rewritten in Y (.ysu)
tests/                Test programs and co-processor workloads (.ysu, .coprocessor.ptx, .wrapped.ptx)
algorithms/           Reference algorithm implementations (Y + C)
c_src/                C/C++ host bindings, CUDA wrappers
docs/                 Language specification and design notes
scripts/              Build automation


Compiler Pipeline

source (.ysu)
  → lexer.rs        tokenize
  → parser.rs       build AST
  → type_checker.rs safety-block checks, invariant/bounds verification, drift checks
  → backend select  based on hardware profile + source annotations
       → llvm_emitter.rs          → LLVM IR → clang → native binary
       → ptx_emitter.rs           → NVIDIA PTX
       → c_emitter.rs             → portable C
       → cpu_emitter.rs           → x86-64 machine code
       → native_emitter.rs        → ELF binary
       → coprocessor_scheduler.rs → fused RT+Tensor Core PTX (--emit-coprocessor)


Hardware Probing

On first run, the compiler measures the host machine and caches the result to .ysu_hw_profile:

CPU: AVX/AVX-512 support, L1/L2/L3/RAM latency (via pointer-chasing cache sweep), AVX-512 throughput, thread-handoff cost.
GPU (via external CUDA probe binary): FMA/IMAD/transcendental latencies, shared-memory bank-conflict cycles, tensor-core latencies (F16/TF32), warp-shuffle cost, global memory latency at multiple strides, RT Core traversal latency.

Example profile output:

AVX = true
AVX512 = true
L1_CYCLES = 4
L2_CYCLES = 12
L3_CYCLES = 40
MEM_CYCLES = 120
GPU_NAME = NVIDIA GeForce RTX 4070 Ti SUPER
FMA_LATENCY = 4.54
SMEM_LATENCY = 28.03
TENSOR_F16_LATENCY = 42.14
WARP_SIZE = 32

Subsequent runs load the cached profile instead of re-probing.


Safety Directives

Code inside @safe { } blocks must initialize all variables, cannot dereference raw pointers, and every loop requires an @invariant. @unsafe { } opts back into raw pointer access. chisel { } allows direct register/memory-bus access.

fn main() {
    @safe {
        let x: I32 = 10;

        @invariant(x >= 0)
        while x > 0 {
            x = x - 1;
        }
    }
}

Other directives: @bounds(min, max) for static index range checks, @ZeroDrift for verified drift-free fixed-point accumulation, @divergence(uniform) to assert non-divergent warp branches, @tile(M, N, K) to schedule tensor-core tile operations.


Dual-Accelerator Co-Processing Pipeline

The compiler includes a Hardware-Sentient Scheduler (--emit-coprocessor) that automatically fuses RT Core and Tensor Core workloads within a single GPU kernel.

The problem it solves

On modern NVIDIA GPUs (Ampere, Ada Lovelace), RT Cores and Tensor Cores are useful together but hard to combine by hand. They are:

- Asymmetric in timing: RT traversal is asynchronous and non-deterministic (latency depends on BVH depth); Tensor Core ops are synchronous, lock-step warp instructions.
- Mismatched in precision: RT Cores output FP32; Tensor Cores need packed FP16/BF16 fragments.
- Costly to hand off between: staging through shared memory requires manually placed bar.sync fences, bank-conflict-aware swizzle layouts, and vectorized cvt.rn.f16x2.f32 packing.

What Y does automatically

- Builds an IR dependency graph (ir_grapher.rs) identifying RT Core and Tensor Core nodes, cross-pipeline data edges, and the critical path through the kernel.
- Schedules the co-processor timeline (coprocessor_scheduler.rs): allocates a single unified coprocessor_smem shared-memory budget, places sync barriers at minimum-cost cut points, and overlaps RT traversal latency with independent scalar instructions.
- Injects a vectorized quantization pass (quantization_pass.rs): emits cvt.rn.f16x2.f32 loops that pack FP32 RT outputs into half2 Tensor Core inputs, using bank-conflict-free swizzled address layouts.
- Emits fused PTX (rt_core_emitter.rs): all RT scratch and output writes are aliased directly to the scheduler's coprocessor_smem offset, eliminating the double-allocation bug that causes CUDA_ERROR_INVALID_PTX at large dimensions.

Writing a co-processor workload in Y

The developer writes a high-level description. The compiler handles the rest:

# tests/coprocessor_attention.ysu  — RT-routed sparse attention
@unsafe
fn main() {
    # RT Core: BVH-accelerated K-Nearest Neighbor (128D, k=8)
    let nns_res: I32 = rt_nearest_neighbor(128, 8);

    # Tensor Core: MMA projection on routed vectors
    # sync barrier, FP32->FP16 quantization, and swizzled ldmatrix are injected automatically
    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
    let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(nns_res);
    acc = mma_sync(frag_A, frag_B, frag_C);
}

The equivalent CUDA C++ kernel requires 160+ lines: manual OptixRayQuery traversal, shared-memory staging, bar.sync fences, explicit cvt.rn.f16x2.f32 packing, and wmma:: fragment loads.

Compile with:

cargo run -- tests/coprocessor_attention.ysu --emit-coprocessor


Building

Requires: Rust toolchain, clang, optionally nvcc for the GPU probe.

cargo build --release
./target/release/Y

# Compile a Y program
cargo run -- tests/hello.ysu           # LLVM backend (default)
cargo run -- tests/train_spec.ysu --llvm
cargo run -- tests/hello.ysu --c
cargo run -- tests/test_drift.ysu      # PTX for kernel files

# Compile a co-processor kernel
cargo run -- tests/coprocessor_attention.ysu --emit-coprocessor
cargo run -- tests/coprocessor_db_index.ysu --emit-coprocessor


Benchmarks

All benchmarks were run on a single development machine (AMD Ryzen 9 9950X, NVIDIA RTX 4070 Ti SUPER, 48GB DDR5-6000). They have not been independently reproduced on other hardware. Verification scripts (verify_r1cs.py, verify_heavy.py, verify_dot_product.py) are included so results can be checked against the generated circuit files.

---

GPU kernel: Y-emitted PTX vs. PyTorch

1024-step F32 accumulation kernel, 1000 launches averaged (tests/benchmark.py):

| Implementation            | Avg time/launch |
| :--- | :--- |
| PyTorch Eager             | 2579.23 µs |
| PyTorch Compiled (Triton) | 13.40 µs |
| Y-emitted PTX             | 1.98 µs |

---

Dual-Accelerator Co-Processor: Y vs. Naive CUDA C++ (10,000 iterations, RTX 4070 Ti SUPER)

The co-processor scheduler automatically overlaps RT Core traversal with Tensor Core MMA, inserts vectorized quantization, and eliminates shared-memory bank conflicts. All results are physically measured on device via CuPy JIT.

| Workload | RT/Tensor Topology | Naive CUDA C++ | Y Co-Processor | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Sparse Attention Router (128D, k=8) | 1 RT + 5 TC + 1 barrier | 4.2175 µs | 2.3818 µs | **1.77x** |
| Large MMA Pipeline (128D, k=8, 7 TC nodes) | 1 RT + 7 TC + 1 barrier | 2.4501 µs | 1.8515 µs | **1.32x** |
| DB Index FRNN Search (256D, k=16) | 1 RT + 5 TC + 1 barrier | 10.6026 µs | 5.9137 µs | **1.79x** |

Static scheduling summary (--emit-coprocessor output):

| Kernel | Parallel Cycles | Overlap Savings | SMEM Budget |
| :--- | :---: | :---: | :---: |
| coprocessor_attention.ysu | 215 cycles | 133 cycles | 8,704 bytes |
| coprocessor_large.ysu | 287 cycles | 145 cycles | 8,704 bytes |
| coprocessor_db_index.ysu | 215 cycles | 133 cycles | 33,280 bytes |

Note: the attention and db_index kernels share an identical IR node topology (1 RT node, 5 Tensor nodes, 1 barrier), so the static scheduler produces identical cycle estimates. Their physical latencies differ substantially (2.38 µs vs. 5.91 µs) because the RT traversal cost scales with search dimensionality and neighbor count (128D/k=8 vs. 256D/k=16).

Note on db_index recall: index construction and recall@k tradeoffs are workload-specific. This benchmark demonstrates traversal speedup via hardware BVH mapping, not index quality or search accuracy.

---

CPU lock-free queue: Y vs. C++

20M push/pop ops, SPSC ring buffer, capacity 1024:

| Implementation | Time | Throughput |
| :--- | :---: | :---: |
| Mutex std::queue (baseline) | 1.460s | 13.70 MOps/s |
| C++ SPSC, unaligned | 0.089s | 225.22 MOps/s |
| C++ SPSC, cache-line aligned | 0.062s | 321.37 MOps/s |
| Y-compiled SPSC | 0.066s | 301.39 MOps/s |

Y comes within 6% of hand-tuned, cache-line-aligned C++ without manual alignment tuning — the compiler derived the correct alignment from the measured L2 cache line size and the source's @align/@atomic annotations.

---

R1CS constraint generation: Y vs. Circom, Noir, Leo

1,000,000 constraints (heavy_circuit):

| Compiler | Time | Peak memory |
| :--- | :---: | :---: |
| Y | 1.67s | 1.07 GB |
| Noir (Nargo) | 11.36s | 1.25 GB |
| Leo | 41.52s | 10.81 GB |
| Circom | 259.25s | 2.39 GB |

100,000 constraints (dot_product):

| Compiler | Time | Peak memory |
| :--- | :---: | :---: |
| Noir (Nargo) | 2.31s | 393.74 MB |
| Y | 3.66s | 154.24 MB |
| Leo | 13.83s | 3.08 GB |
| Circom | 14.51s | 1.05 GB |

Noir compiles faster on this flatter constraint graph; Y uses less memory across the board.

31,000,000 constraints (heavy_31m.ysu):

| Compiler | Result |
| :--- | :--- |
| Y | 105.28s, 30.65 GB peak RSS |
| Noir | Estimated ~39 GB required |
| Leo | Estimated ~335 GB required |
| Circom | Estimated ~74 GB, ~2.2 hours |

Noir, Leo, and Circom figures at this scale are estimated from their memory-scaling behavior at smaller sizes, not measured directly, since none completed on the test machine.

Why Y uses less memory at scale: in-place accumulator updates avoid O(N) vector copies on loop-scoped reassignment, linear-combination addition is checked in O(1) when inputs are already flat, and constraint deduplication uses an order-independent hash map.


Self-Hosting

Most compiler phases are duplicated in native Y under self_hosted/, alongside their Rust originals in src/. The Rust implementation is the stable reference; the Y implementation is the long-term target once it can compile itself end-to-end.


Author: Umut Korkmaz (YSU)
