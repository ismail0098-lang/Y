# Y  The Hardware-Sentient Systems Language

**Y** is a systems programming language and self-hosting compiler infrastructure built for high-performance computing (HPC), AI kernel development, GPU programming, and formally verified OS engineering. It powers the **YSU-Engine** and bridges the gap between mathematically rigorous compile-time verification and bare-metal, cycle-precise hardware control.

---

## Table of Contents

- [Core Philosophies](#core-philosophies)
- [The Self-Hosting Loop](#the-self-hosting-loop)
- [Project Structure](#project-structure)
- [Compiler Pipeline](#compiler-pipeline)
- [Backend Emitters](#backend-emitters)
- [Language Reference](#language-reference)
- [The SS Safe Subset](#the-ss-safe-subset)
- [Sentinel Hardware Probe](#sentinel-hardware-probe)
- [Real-World Applications](#real-world-applications)
- [Building & Running](#building--running)
- [Installation](#installation)
- [Benchmark Results](#benchmark-results)

---

## Core Philosophies

Y abandons traditional compiler design in favor of three founding principles:

### 1. Hardware-Sentience

Unlike compilers that use static target triples (e.g., `x86_64-pc-windows-msvc`), Y features a **Sentinel Probe** (`src/sentinel.rs`) that dynamically measures the host silicon on first boot and caches a full `HardwareProfile` to `.ysu_hw_profile`. The compiler inherently knows:

| Category | Measured Properties |
|---|---|
| **CPU** | AVX/AVX-512 support, L1/L2/L3/Mem latency cycles, AVX-512 throughput, thread scheduling cost |
| **GPU** | FMA/IMAD/MUFU latencies, SMEM/L1/L2/VRAM latencies, Tensor Core F16/TF32 latencies, warp size, register limits |
| **Branch** | Uniform vs divergent cycle counts, divergence penalty |
| **SMEM** | No-conflict, 2-way, and 4-way bank conflict cycles; broadcast latency |
| **SFU** | EX2/SIN/RSQ/LG2 latencies, HFMA2, BF16×2 FMA, LOP3.LUT |

The emitter uses this profile to make cycle-aware decisions — e.g., dynamically choosing `IMAD.WIDE` (2.59 cycles) over `IMAD` (4.53 cycles) on an RTX 4070 Ti SUPER.

### 2. Formal Verification  The Safety Cage

Y moves beyond runtime safety (Java GC) and ownership (Rust borrow checker) to enforce **mathematically rigorous correctness at compile-time**. The compiler distinguishes three safety levels:

| Block | Semantics |
|---|---|
| `@safe { ... }` | Strict verification zone. Variables must be explicitly initialized. Raw pointer deref (`*ptr`) is forbidden. All loops require `@invariant`. |
| `@unsafe { ... }` | Opt-in for raw pointer manipulation, unverified math, and direct hardware interaction. |
| `chisel { ... }` | Direct register/memory-bus level access for silicon-level work. |

**Compile-time directives:**

- **`@invariant(expr)`** — Required on all `while`/`for` loops inside `@safe` blocks. Proves loop bounds to the compiler, ensuring zero memory violations.
- **`@bounds(min, max)`** — Statically declares safe index ranges, eliminating runtime checks.
- **`@ZeroDrift`** — Guarantees zero numerical drift. If the fast-math path of the target GPU introduces precision loss (e.g., F16), the compiler emits a performance advisory and routes to a verified precision path.
- **`@divergence(uniform)`** — Asserts a warp branch is provably non-divergent.
- **`@tile(M, N, K)`** — Hints the emitter to schedule WMMA tile operations.

### 3. High-Precision Numerical Stability

Y is engineered for environments where a single floating-point rounding error is unacceptable — aerospace, high-frequency trading, AI training loops. The `@ZeroDrift` directive is checked against the hardware profile at compile time. If `Q32.32` fixed-point is available in `DRIFT_FREE_TYPES`, the accumulator is verified drift-free; otherwise a warning is emitted and no silent fallback occurs.

---

## The Self-Hosting Loop

The ultimate goal of this project is a fully self-compiling Y compiler.

```
Bootstrap Phase:  Rust compiler (src/) compiles .ysu files
Self-Host Phase:  self_hosted/ compiler (written in .ysu) compiles .ysu files
```

The `self_hosted/` directory already contains native Y rewrites of every major compiler component:

| File | Rust Equivalent |
|---|---|
| `self_hosted/lexer.ysu` | `src/lexer.rs` |
| `self_hosted/parser.ysu` | `src/parser.rs` |
| `self_hosted/compiler.ysu` | `src/type_checker.rs` + orchestration |
| `self_hosted/llvm_emitter.ysu` | `src/llvm_emitter.rs` |
| `self_hosted/c_emitter.ysu` | `src/c_emitter.rs` |
| `self_hosted/native_emitter.ysu` | `src/native_emitter.rs` |
| `self_hosted/type_checker.ysu` | `src/type_checker.rs` |
| `self_hosted/yls.ysu` | Y Language Server (LSP) |

The `compiler.ysu` file in the root is the current self-hosted compiler stage being actively developed (~200 KB, ~177 KB in `self_hosted/`). The stage-1 LLVM IR output (`compiler_stage1.ll`) is ~766 KB.

---

## Project Structure

```
Y_lang/
├── src/                    # Rust bootstrap compiler
│   ├── main.rs             # CLI entry point, pipeline orchestration
│   ├── lexer.rs            # Tokenizer (37 KB) — handles @-directives, GPU intrinsics
│   ├── parser.rs           # Recursive descent parser (76 KB), arena-allocated AST
│   ├── ast.rs              # Data-oriented AST node definitions (14 KB)
│   ├── type_checker.rs     # Semantic analysis, Safety Cage, Linear Tracker (86 KB)
│   ├── bank_conflict.rs    # SMEM bank conflict mathematical prover (5 KB)
│   ├── linear_tracker.rs   # Transfer obligation (pipe) consumption tracker (6 KB)
│   ├── sentinel.rs         # Hardware probe — CPU + GPU microbenchmarks (47 KB)
│   ├── avx_wrapper.rs      # AVX/AVX-512 intrinsic wrappers (18 KB)
│   ├── llvm_emitter.rs     # LLVM IR emitter → clang → native binary (136 KB)
│   ├── ptx_emitter.rs      # Bare-metal NVIDIA PTX emitter (31 KB)
│   ├── c_emitter.rs        # C transpiler backend (50 KB)
│   ├── cpu_emitter.rs      # Direct x86-64 code emitter (15 KB)
│   ├── native_emitter.rs   # ELF/native binary assembler (10 KB)
│   ├── ypm.rs              # Y Package Manager (14 KB)
│   └── ysu_gpu_probe.rs    # External GPU microbenchmark binary (33 KB)
│
├── self_hosted/            # Y compiler rewritten in native Y (.ysu)
│   ├── compiler.ysu        # Full compiler in Y (177 KB)
│   ├── parser.ysu          # Y parser in Y (62 KB)
│   ├── type_checker.ysu    # Type checker in Y (64 KB)
│   ├── llvm_emitter.ysu    # LLVM emitter in Y (27 KB)
│   ├── lexer.ysu           # Lexer in Y (21 KB)
│   ├── yls.ysu             # Y Language Server in Y (32 KB)
│   └── std/                # Y standard library
│
├── tests/                  # Y test programs (.ysu)
│   ├── hello.ysu           # Functions, recursion, control flow
│   ├── ring_buffer.ysu     # SPSC lock-free queue with @atomic fields
│   ├── train_spec.ysu      # GPU kernel with @safe, @ZeroDrift, @tile
│   ├── test_drift.ysu      # @ZeroDrift + @require(sm >= 89) directives
│   ├── math.ysu            # Arithmetic and floating-point
│   ├── bounds_test.ysu     # @bounds directive verification
│   └── ...                 # 15+ additional test programs
│
├── algorithms/
│   ├── matching.ysu        # Stroke matching algorithm (spatial + directional)
│   └── matching.c          # Reference C implementation
│
├── c_src/                  # C/C++ host bindings and CUDA wrappers
├── docs/                   # Language specifications and architecture docs
├── scripts/                # Build automation and helper scripts
├── z3/                     # Z3 SMT solver integration
├── z3_benchmarks/          # SMT benchmark suite
├── linux-cachyos/          # CachyOS kernel formal verification harnesses
├── NC-Compiler-Project/    # NC compiler sub-project
├── build_artifacts/        # Emitted binaries, LLVM IR, object files
├── compiler.ysu            # Current self-hosted compiler stage (200 KB)
├── compiler_stage1.ll      # Stage-1 LLVM IR output (766 KB)
├── yc_bootstrap            # Bootstrap compiler binary
├── yc_bootstrap.c          # Bootstrap compiler C source (229 KB)
├── Cargo.toml              # Rust package — builds Y, ypm, ysu_gpu_probe, ysu_vmm
├── install.sh              # Global installer for Y ShadowPlay
├── package.sh              # Packages ShadowPlay into a portable .tar.gz
└── .ysu_hw_profile         # Cached hardware profile (generated on first run)
```

---

## Compiler Pipeline

```
Source (.ysu)
     │
     ▼
  lexer.rs          ← Tokenizes @-directives, GPU intrinsics (cp_async, ldmatrix, etc.)
     │
     ▼
  parser.rs         ← Recursive descent → Arena-allocated AST (ast.rs)
     │
     ▼
  type_checker.rs   ← Safety Cage enforcement
     │               ├── Linear Tracker (pipe obligations)
     │               ├── Bank Conflict Prover (SMEM matrix layouts)
     │               ├── @invariant / @bounds verification
     │               └── @ZeroDrift type checking
     │
     ▼
  [Backend selection based on HardwareProfile + source annotations]
     │
     ├── llvm_emitter.rs   → LLVM IR → clang → native binary
     ├── ptx_emitter.rs    → NVIDIA PTX assembly
     ├── c_emitter.rs      → Portable C source
     ├── cpu_emitter.rs    → Direct x86-64 machine code
     └── native_emitter.rs → ELF binary (no external toolchain)
```

### Key Semantic Analysis Components

**Linear Tracker** (`src/linear_tracker.rs`): Enforces that asynchronous memory pipeline tokens (`cp_async` return values) are consumed exactly once via `pipe.wait(token)`. Prevents resource leaks in GPU pipelines.

**Bank Conflict Prover** (`src/bank_conflict.rs`): Mathematically analyzes `SmemLayout` matrix tile configurations and proves 0-bank-conflict access patterns. Uses measured SMEM conflict penalty cycles from the hardware profile to emit optimal padding.

---

## Backend Emitters

| Backend | Flag / Annotation | Output | Use Case |
|---|---|---|---|
| LLVM IR | `--llvm` | `.ll` → native binary via clang | General-purpose optimized native code |
| PTX | `kernel` keyword | `.ptx` | NVIDIA GPU kernel bare-metal |
| C | `--c` | `.c` | Portability, FFI, debugging |
| x86-64 | `--cpu` | native binary | Direct machine code, no LLVM |
| ELF Native | `--native` | ELF binary | Self-contained, no external toolchain |

The PTX emitter generates real NVIDIA PTX instructions including `cp.async`, `ldmatrix`, `wmma`, barrier synchronization, and warp shuffle primitives, informed by the GPU latency profile.

---

## Language Reference

### Type System

```ysu
// Primitive types
let a: I32 = 42;
let b: I64 = 1000000;
let c: F32 = 3.14;
let d: F16 = 1.0;       // Half precision — GPU accelerated
let e: Q32.32 = 1.0;    // Fixed-point — ZeroDrift guaranteed
let f: bool = true;

// Structs with hardware layout annotations
struct SpscBuffer {
    @align(64) @atomic(acq_rel) head: I64,
    _pad1: I64, _pad2: I64, _pad3: I64,  // Cache-line padding
    @align(64) @atomic(acq_rel) tail: I64,
    buffer: [I64; 1024],
}

// Pointers and references
let ptr: &Point2D = addr;
let val: F32 = (*ptr).x;
let s: &mut SpscBuffer = ...;
```

### Functions and Kernels

```ysu
// Regular function
fn add(a: I32, b: I32) -> I32 {
    return a + b;
}

// GPU kernel — emits PTX, requires hardware annotations
@require(sm >= 89)          // Require Ada Lovelace or newer
@require(tensor_cores >= 4)
kernel matmul(A: GlobalMemory<F16>, B: GlobalMemory<F16>) {
    // kernel body
}

// Unsafe function — raw pointer access allowed
@unsafe
fn match_stroke(user_path: ptr, ref_path: ptr, N: I32) -> F32 {
    let base: I64 = user_path;
    let pt: &Point2D = base + (i * 8);
    let dx: F32 = (*pt).x - (*ref_pt).x;
    // ...
}
```

### Safety Directives

```ysu
fn main() {
    @safe {
        let x: I32 = 10;

        // Every loop in @safe MUST have @invariant
        @invariant(x >= 0)
        while x > 0 {
            x = x - 1;
        }

        @invariant(i >= 0)
        for i in 0..10 {
            @bounds(0, 10)
            let idx: I32 = i;
            print_int(idx);
        }
    }
}
```

### GPU-Specific Directives

```ysu
// Cache hints
@cache_policy(L2_PERSIST, reuse_count=4)
let persistent_var = a;

// Asynchronous pipelining (Linear Obligation)
let pipe = Pipeline::init();
let token = cp_async(smem_buffer, global_mem);  // Obligation created
pipe.wait(token);                                 // Obligation consumed exactly once

// Warp-level operations
@divergence(uniform)
if condition { ... }                 // Asserts all threads take same path

// Tile scheduling for Tensor Core WMMA
@tile(16, 16, 8)
for i in 0..1024 step 1 { ... }

// ZeroDrift numerical accumulator
@ZeroDrift
let acc: Q32.32 = Fragment::zero(); // Verified drift-free

// Barrier synchronization
barrier::sync();
```

### Memory Hierarchy

```ysu
// GPU memory spaces
kernel example(weights: GlobalMemory<F32>) {
    let w: F32 = GlobalMemory::load(weights);  // VRAM → register
    // SharedMemory, L1, L2 addressed via @cache_policy
}
```

### Standard Library Primitives

```ysu
print_int(value: I32)
println(msg: str)
sqrtf(x: F32) -> F32
Fragment::zero() -> T
Pipeline::init() -> Pipeline
barrier::sync()
```

---



## Sentinel Hardware Probe

On first run the compiler executes a full hardware characterization and writes `.ysu_hw_profile`. On subsequent runs the profile is loaded instantly, skipping re-probing.

### CPU Probe (`src/sentinel.rs`)

- **CPUID** (EAX=1, EAX=7, EAX=0x80000006) — detects AVX, AVX-512, L2 cache line size
- **Cache Latency Sweep** — pointer-chasing with Satolo random permutation at 16 KB, 256 KB, 4 MB, 64 MB to measure L1/L2/L3/RAM latency in TSC cycles
- **AVX-512 Throughput** — 10× `vpaddd zmm` loop with 1M iterations → cycles per op
- **Thread Scheduling Cost** — atomic ping-pong handoff between two threads × 999 rounds

### GPU Probe (`src/ysu_gpu_probe.rs`)

An external binary (`ysu_gpu_probe`) launched as a subprocess. Measures 25+ GPU hardware properties via CUDA microbenchmarks and writes them to stdout in `KEY=VALUE` format, which the sentinel parser ingests. Includes:

- FMA, IMAD, MUFU (RCP/EX2/SIN/RSQ/LG2) latencies
- SMEM no-conflict, 2-way, 4-way bank conflict, and broadcast cycles
- Tensor Core HMMA F16 and TF32 latencies
- Warp shuffle vs SMEM exchange latency
- CP.ASYNC global→shared latency
- Global atomic add (F32, I32) latency
- Strided memory access (stride 1×–32×)
- Thermal correction at 40°C / 60°C / 80°C

Profile example (`.ysu_hw_profile`):
```
AVX = true
AVX512 = true
L2_LINE = 64
L1_CYCLES = 4
L2_CYCLES = 12
L3_CYCLES = 40
MEM_CYCLES = 120
GPU_NAME = NVIDIA GeForce RTX 4070 Ti SUPER
FMA_LATENCY = 4.0
SMEM_LATENCY = 28.0
HMMA_F16_LATENCY = 42.0
TF32_LATENCY = 66.0
WARP_SIZE = 32
TOTAL_GLOBAL_MEM_MB = 16376
DRIFT_FREE_TYPES = Q32.32, FP64
```

---



## Building & Running

### Prerequisites

- Rust toolchain (`cargo`)
- `clang` (for LLVM IR → binary compilation)
- `nvcc` (optional, for GPU probe compilation)
- `qemu-system-x86_64` (optional, for Y OS kernel testing)

### Build

```bash
# Build all binaries (Y compiler, ypm, ysu_gpu_probe, ysu_vmm)
cargo build --release

# The compiler binary is at:
./target/release/Y
```

### Compile a Y Program

```bash
# Compile to native binary (default: LLVM backend)
cargo run -- tests/hello.ysu

# Compile with explicit backend
cargo run -- tests/train_spec.ysu --llvm   # → LLVM IR + native
cargo run -- tests/hello.ysu --c           # → C source
cargo run -- tests/test_drift.ysu          # → PTX (kernel files)

# Compile self-hosted compiler
cargo run -- compiler.ysu
```

On first run you will see the Sentinel Probe execute and report your hardware profile. Subsequent runs load it instantly:

```
[*] Found existing .ysu_hw_profile, skipping Sentinel Probe.
    -> Loaded AVX: true
    -> Loaded AVX-512: true
    -> Loaded L2 Cache Line Size: 64 bytes
    -> Loaded CPU Memory Latency Sweep (L1/L2/L3/Mem): 4 / 12 / 40 / 120 cycles
    -> Loaded GPU Name: NVIDIA GeForce RTX 4070 Ti SUPER
    -> GPU Memory Latencies (SMEM/L1/L2/VRAM): 28 / 33 / 90 / 300
    -> GPU Tensor Core Latencies (F16/TF32): 42 / 66
```

---

## Benchmark Results

All benchmarks measured live on this machine: **NVIDIA RTX 4070 Ti SUPER (16 GB, CUDA 13.0)** · AVX-512 CPU · L2 cache line 64 B.

---

### Benchmark 1 — GPU Kernel: Y Native PTX vs PyTorch

**Source:** `tests/train_spec.ysu` compiled to PTX by Y, vs PyTorch Eager and `torch.compile` (Triton)  
**Task:** 1024-step F32 accumulation kernel · 1000 launches averaged · `tests/benchmark.py`

| Implementation | Avg Time / Launch | vs Y |
|---|---|---|
| PyTorch Eager Mode | 2579.23 µs | **1301× slower** |
| PyTorch Compiled (Triton) | 13.40 µs | **6.76× slower** |
| **Y Native PTX** (`train_spec.ysu`) | **1.98 µs** | — |

**Y is 1301× faster than PyTorch Eager and 6.76× faster than Triton.**

The Y compiler emits bare PTX with `.maxnreg 32` (pinned for 100% SM occupancy — 6 active blocks per SM on the RTX 4070 Ti SUPER), a direct `ld.global.ca.f32` cache-hinted global load, and a tight scalar loop. No Python runtime overhead, no JIT compilation latency, no kernel launch abstraction layers.

```ptx
// Y-emitted PTX (train_spec.ysu → ptx_emitter.rs)
// [WARP REGISTER ALLOCATOR] 100.00% occupancy (6 blocks/SM)
.maxnreg 32
ld.global.ca.f32 %f0, [%rd0];   // L1/L2 cache-hinted load
$LOOP_START_0:
    add.f32 %f2, %f1, %f0;      // accumulate
    add.u32 %r1, %r1, 1;
    bra $LOOP_START_0;
```

---

### Benchmark 2 — CPU Lock-Free Queue: Y vs C++

**Source:** `tests/ring_buffer.ysu` compiled to native object · vs `tests/benchmark.cpp`  
**Task:** 20 million push/pop ops · SPSC ring buffer capacity = 1024

| # | Implementation | Time | Throughput | vs Y |
|---|---|---|---|---|
| 1 | Mutex `std::queue` *(baseline)* | 1.460 s | 13.70 MOps/s | 22.0× slower |
| 2 | C++ SPSC — Unaligned `[acq/rel]` | 0.089 s | 225.22 MOps/s | 1.34× slower |
| 3 | C++ SPSC — Aligned CL64 `[acq/rel]` | 0.062 s | 321.37 MOps/s | 1.07× faster |
| **4** | **Y-compiled SPSC** (`ring_buffer.ysu`) | **0.066 s** | **301.39 MOps/s** | — |

**Y is 22× faster than mutex, 1.34× faster than unaligned C++, and within 6% of the best hand-tuned cache-line-aligned C++ SPSC** — with zero manual tuning. The hardware profile (`L2_LINE = 64` measured by Sentinel) drove the emitter to produce `load atomic i64 acquire, align 64` / `store atomic i64 release, align 64` automatically from the `.ysu` source annotations:

```ysu
struct SpscBuffer {
    @align(64) @atomic(acq_rel) head: I64,   // → align 64 acquire load
    _pad1: I64, _pad2: I64, _pad3: I64, _pad4: I64, _pad5: I64, _pad6: I64, _pad7: I64,
    @align(64) @atomic(acq_rel) tail: I64,   // → align 64 release store
    buffer: [I64; 1024],
}
```

---

### Benchmark 3 — ZK R1CS Compiler: Y vs. Circom

**Task:** Compile large-scale Rank-1 Constraint Systems (R1CS) under unrolled loops of non-linear constraints and conditionals. 
**Environment:** Measured live on the same host (NVIDIA RTX 4070 Ti SUPER, Intel AVX-512 CPU).

#### **A. 1,000,000 Constraints (Polynomial Loop - `heavy_circuit.ysu`)**
- **Y-lang Compiler**: **`2.20 seconds`** (`2,195.72 ms`)
- **Circom Compiler**: `292.05 seconds` (`292,045.16 ms` / ~4.87 minutes)
- **Speedup**: **133.01x faster compilation**

#### **B. 100,000 Constraints (Iterative Dot Product - `dot_product.ysu`)**
- **Y-lang Compiler**: **`3.63 seconds`** (`3,634.07 ms`)
- **Circom Compiler**: `17.08 seconds` (`17,077.58 ms`)
- **Speedup**: **4.70x faster compilation**

#### **C. Baseline Equivalence & Optimization (`test_circuit.ysu`)**
- **Y-lang Compiler**: **5 constraints, 8 wires** (Fully optimized linear combination folding)
- **Circom Compiler**: **5 constraints, 10 wires** (Using `IsEqual` comparator sub-template)

#### **Why is Y-lang so much faster?**
1. **In-Place Accumulator Updates**: Scope reassignments inside loops (e.g., `temp = temp * y`) mutate linear combinations in-place, eliminating $O(N)$ vector copying.
2. **Simplified State Propagation**: Linear addition checks bypass the $O(N)$ simplifier when inputs are already flat and disjoint, reducing constraint addition to $O(1)$.
3. **Rust-Native Deduplication**: Constraint hashing uses a high-performance order-independent associative map, reducing sorting and hashing complexity.

---

## Status

> This project is under active development. The self-hosting loop is progressing;  all major compiler phases are written in native Y. The bootstrap Rust compiler (`src/`) is the stable reference implementation; `self_hosted/` is the production target.

**Author:** Umut Korkmaz (YSU) (ismail0098@gmail.com)

LLMs were used to build this project. Every architectural decision is mine.

нещо
Driski
