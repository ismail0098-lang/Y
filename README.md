Y — A Systems Language and Compiler for GPU/CPU Hardware-Aware Code Generation

Y is a compiler and systems language for writing hardware-aware code across CPU (x86/AVX-512) and GPU (NVIDIA PTX) targets. It also includes a zero-knowledge circuit compiler (R1CS constraint generation) as one of its backends.

The project is under active, single-developer, ongoing development.


What this project does


Probes the actual hardware it's running on — cache latencies, AVX-512 throughput, GPU warp/tensor-core timings — and uses those measurements to make codegen decisions (e.g. choosing IMAD.WIDE over IMAD based on measured cycle cost).
Enforces compile-time safety guarantees on marked code blocks: initialized-variable checks, loop invariants, bounds declarations, and a numerical-drift check for fixed-point accumulation.
Compiles to five backends: LLVM IR (→ native binary via clang), NVIDIA PTX, portable C, direct x86-64, and a standalone ELF emitter.
Includes an R1CS constraint generator for zero-knowledge circuits, benchmarked against Circom, Noir, and Leo.
Is partially self-hosting: most compiler phases (lexer, parser, type checker, LLVM emitter) have been rewritten in Y itself, alongside the original Rust implementation.



Status


Bootstrap compiler (src/, Rust): stable, this is what actually runs today.
Self-hosted compiler (self_hosted/, written in Y): in progress, not yet the default build path.
Author-built with LLM assistance for implementation; architecture and design decisions are the author's own.
There is currently a backlog of automated pull requests from a connected AI coding agent (Jules) that have not yet been reviewed or merged, due to a personal medical situation. They do not reflect the current state of main.



Project Layout

src/                  Rust bootstrap compiler
  main.rs             CLI entry point, pipeline orchestration
  lexer.rs            Tokenizer — @-directives, GPU intrinsics
  parser.rs           Recursive-descent parser, arena-allocated AST
  ast.rs              AST node definitions
  type_checker.rs     Semantic analysis, safety-block enforcement, linear tracker
  bank_conflict.rs    Shared-memory bank-conflict prover
  linear_tracker.rs   Tracks that async memory tokens are consumed exactly once
  sentinel.rs         Hardware probe (CPU + GPU microbenchmarks)
  avx_wrapper.rs      AVX/AVX-512 intrinsic wrappers
  llvm_emitter.rs     LLVM IR emission
  ptx_emitter.rs      NVIDIA PTX emission
  c_emitter.rs        C transpiler backend
  cpu_emitter.rs      Direct x86-64 emission
  native_emitter.rs   ELF binary emission (no external toolchain)
  ypm.rs              Package manager
  ysu_gpu_probe.rs    External GPU microbenchmark binary

self_hosted/          Y compiler components rewritten in Y (.ysu)
tests/                Test programs
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
       → llvm_emitter.rs   → LLVM IR → clang → native binary
       → ptx_emitter.rs    → NVIDIA PTX
       → c_emitter.rs      → portable C
       → cpu_emitter.rs    → x86-64 machine code
       → native_emitter.rs → ELF binary


Hardware Probing

On first run, the compiler measures the host machine and caches the result to .ysu_hw_profile:


CPU: AVX/AVX-512 support, L1/L2/L3/RAM latency (via pointer-chasing cache sweep), AVX-512 throughput, thread-handoff cost.
GPU (via external CUDA probe binary): FMA/IMAD/transcendental latencies, shared-memory bank-conflict cycles, tensor-core latencies, warp-shuffle cost, global memory latency at multiple strides.


Example profile output:

AVX = true
AVX512 = true
L1_CYCLES = 4
L2_CYCLES = 12
L3_CYCLES = 40
MEM_CYCLES = 120
GPU_NAME = NVIDIA GeForce RTX 4070 Ti SUPER
FMA_LATENCY = 4.0
SMEM_LATENCY = 28.0
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


Building

Requires: Rust toolchain, clang, optionally nvcc for the GPU probe.

bashcargo build --release
./target/release/Y

# Compile a Y program
cargo run -- tests/hello.ysu           # LLVM backend (default)
cargo run -- tests/train_spec.ysu --llvm
cargo run -- tests/hello.ysu --c
cargo run -- tests/test_drift.ysu      # PTX for kernel files


Benchmarks

All benchmarks were run on a single development machine (AMD Ryzen 9 9950X, NVIDIA RTX 4070 Ti SUPER, 48GB DDR5-6000). They have not been independently reproduced on other hardware. Verification scripts (verify_r1cs.py, verify_heavy.py, verify_dot_product.py) are included so results can be checked against the generated circuit files.

GPU kernel: Y-emitted PTX vs. PyTorch

1024-step F32 accumulation kernel, 1000 launches averaged (tests/benchmark.py):

ImplementationAvg time/launchPyTorch Eager2579.23 µsPyTorch Compiled (Triton)13.40 µsY-emitted PTX1.98 µs

CPU lock-free queue: Y vs. C++

20M push/pop ops, SPSC ring buffer, capacity 1024:

ImplementationTimeThroughputMutex std::queue (baseline)1.460s13.70 MOps/sC++ SPSC, unaligned0.089s225.22 MOps/sC++ SPSC, cache-line aligned0.062s321.37 MOps/sY-compiled SPSC0.066s301.39 MOps/s

Y comes within 6% of hand-tuned, cache-line-aligned C++ without manual alignment tuning — the compiler derived the correct alignment from the measured L2 cache line size and the source's @align/@atomic annotations.

R1CS constraint generation: Y vs. Circom, Noir, Leo

1,000,000 constraints (heavy_circuit):

CompilerTimePeak memoryY1.67s1.07 GBNoir (Nargo)11.36s1.25 GBLeo41.52s10.81 GBCircom259.25s2.39 GB

100,000 constraints (dot_product):

CompilerTimePeak memoryNoir (Nargo)2.31s393.74 MBY3.66s154.24 MBLeo13.83s3.08 GBCircom14.51s1.05 GB

Noir compiles faster on this flatter constraint graph; Y uses less memory across the board.

31,000,000 constraints (heavy_31m.ysu):

CompilerResultY105.28s, 30.65 GB peak RSSNoirEstimated ~39 GB requiredLeoEstimated ~335 GB requiredCircomEstimated ~74 GB, ~2.2 hours

Noir, Leo, and Circom figures at this scale are estimated from their memory-scaling behavior at smaller sizes, not measured directly, since none completed on the test machine.

Why Y uses less memory at scale: in-place accumulator updates avoid O(N) vector copies on loop-scoped reassignment, linear-combination addition is checked in O(1) when inputs are already flat, and constraint deduplication uses an order-independent hash map.


Self-Hosting

Most compiler phases are duplicated in native Y under self_hosted/, alongside their Rust originals in src/. The Rust implementation is the stable reference; the Y implementation is the long-term target once it can compile itself end-to-end.


Author: Umut Korkmaz (YSU)
