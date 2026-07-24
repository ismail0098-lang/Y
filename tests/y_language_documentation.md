# Y Language: The Definitive Specification & Programmer's Reference Manual
Version 1.0 (Operational Systems Programming Language)

Y is a hardware-sentient, low-level systems programming language designed for high-performance computing, lock-free concurrency, and hardware-aware GPGPU/CPU acceleration. It couples structural type checking with hardware profiles (gathered via Sentinel Probes) to enforce optimal performance traits, cache alignments, memory layouts, and register usage directly at compile time.

---

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Step 1: Build the Compiler](#step-1-build-the-compiler)
  - [Step 2: Run the Hardware Sentinel Probe](#step-2-run-the-hardware-sentinel-probe)
  - [Step 3: Write Your First Y Program](#step-3-write-your-first-y-program)
  - [Step 4: Write a Safe Block with Loop Invariant](#step-4-write-a-safe-block-with-loop-invariant)
  - [Step 5: Write a GPU Kernel (PTX backend)](#step-5-write-a-gpu-kernel-ptx-backend)
  - [Step 6: Use the Dual-Accelerator Co-Processor Pipeline](#step-6-use-the-dual-accelerator-co-processor-pipeline)
  - [Quick Reference: Compiler Flags](#quick-reference-compiler-flags)
- [§1 — Introduction & Design Philosophy](#1-introduction--design-philosophy)
- [§2 — Compiler Pipeline Architecture](#2-compiler-pipeline-architecture)
- [§3 — Formal EBNF Grammar Specification](#3-formal-ebnf-grammar-specification)
- [§4 — Generics and Monomorphization](#4-generics-and-monomorphization)
- [§5 — Type System & Memory Spaces](#5-type-system--memory-spaces)
- [§6 — Compile-Time Verification & Analysis](#6-compile-time-verification--analysis)
- [§7 — Emitter Lowering & Code Generation](#7-emitter-lowering--code-generation)
- [§8 — Standard Library Reference](#8-standard-library-reference)
- [§9 — Exhaustive Reference: Language Attributes (Decorators)](#9-exhaustive-reference-language-attributes-decorators)
- [§10 — Complete Code Examples](#10-complete-code-examples)
- [§11 — Hardware-Sentient Dual-Accelerator Co-Processing Pipeline](#11-hardware-sentient-dual-accelerator-co-processing-pipeline)
- [§12 — Zero-Knowledge Circuit Backend (R1CS)](#12-zero-knowledge-circuit-backend-r1cs)
- [§13 — CUDA-to-Y Migration Guide](#13-cuda-to-y-migration-guide)
- [§14 — Performance Tuning Guide](#14-performance-tuning-guide)
- [§15 — Fragment & MMA Type Reference](#15-fragment--mma-type-reference)
- [§16 — `chisel {}` Block Reference](#16-chisel--block-reference)
- [§17 — Frequently Asked Questions](#17-frequently-asked-questions)
- [§18 — Known Limitations](#18-known-limitations)
- [§19 — `ypm` Package Manager](#19-ypm--y-package-manager)
- [§20 — Numeric Types Reference](#20-numeric-types-reference)
- [§21 — `SmemLayout` & `Pipeline` API Reference](#21-smemlayout--pipeline-api-reference)
- [§22 — Operator Precedence](#22-operator-precedence)
- [§23 — Error Code Reference](#23-error-code-reference)

---

## Getting Started

This section gets you from zero to a running Y program in a few steps. If you want the full language spec, skip ahead to Section 1.

### Prerequisites

You need the following tools installed:

| Tool | Required | Purpose |
| :--- | :---: | :--- |
| Rust toolchain (`rustup`) | | Compile the Y bootstrap compiler |
| `clang` (LLVM) |  | Link LLVM IR output into native binaries |
| `nvcc` (CUDA toolkit) | Optional | GPU hardware probe and PTX kernel execution |
| Python 3 + CuPy | Optional | Run co-processor benchmarks on GPU |

### Step 1: Build the Compiler

```bash
git clone https://github.com/ismail0098-lang/Y.git
cd Y
cargo build --release
```

The compiled binary is at `./target/release/Y`.

### Step 2: Run the Hardware Sentinel Probe

The first time you compile any `.ysu` file, Y automatically runs the hardware probe and saves the result to `.ysu_hw_profile`. Subsequent runs load the cache instantly — no re-probing.

You can also trigger the probe explicitly (e.g. after upgrading hardware, or on a new machine):

```bash
./target/release/Y --probe
```

Either way, you'll see output like:
```
[*] Running Sentinel Hardware Probe...
    -> AVX: true | AVX-512: true
    -> L1/L2/L3/Mem latency: 4 / 12 / 42 / 335 cycles
    -> GPU Name: RTX 4070 Ti SUPER
    -> GPU FMA latency: 4.54 cy | SMEM: 28.03 cy | Tensor F16: 42.14 cy
    -> Branch divergence penalty: 4.53 cy
[*] Profile saved to .ysu_hw_profile
```

### Step 3: Write Your First Y Program

Create a file `hello.ysu`:

```ysu
fn main() -> I32 {
    return 42;
}
```

Compile and run it:

```bash
cargo run -- hello.ysu          # LLVM backend (default) -> native binary
./hello                         # Run it
```

### Step 4: Write a Safe Block with Loop Invariant

Y's `@safe` blocks enforce compile-time correctness guarantees. Variables must be initialized, and loops need `@invariant` annotations:

```ysu
fn countdown(n: I32) -> I32 {
    @safe {
        let x: I32 = n;

        @invariant(x >= 0)
        while x > 0 {
            x = x - 1;
        }

        return x;
    }
}

fn main() -> I32 {
    return countdown(10);
}
```

```bash
cargo run -- countdown.ysu
```

### Step 5: Write a GPU Kernel (PTX backend)

Y compiles directly to NVIDIA PTX for GPU execution. Here is a simple F32 accumulation kernel:

```ysu
// train_spec.ysu — GPU kernel: accumulate 1024 F32 values
kernel accumulate(data: GlobalMemory<F32>, result: GlobalMemory<F32>, N: I32) {
    let acc: F32 = 0.0;
    for i in 0..N {
        acc += data[i];
    }
    result[0] = acc;
}
```

```bash
# Compile to native binary via LLVM backend:
cargo run -- tests/train_spec.ysu --llvm

# Or emit raw PTX directly:
cargo run -- tests/train_spec.ysu --ptx
```

Benchmarked against PyTorch on RTX 4070 Ti SUPER:

| Implementation | Avg latency |
| :--- | :---: |
| PyTorch Eager | 2,579 µs |
| PyTorch + Triton | 13.40 µs |
| **Y-emitted PTX** | **1.98 µs** |

### Step 6: Use the Dual-Accelerator Co-Processor Pipeline

Y can automatically fuse **RT Core** (BVH traversal / nearest neighbor search) with **Tensor Core** (matrix multiply-accumulate) workloads in a single kernel — inserting sync barriers, FP32→FP16 quantization, and bank-conflict-free shared memory layouts at compile time.

```ysu
// coprocessor_attention.ysu
// RT Core routes token queries; Tensor Core projects the results.
// The compiler handles: bar.sync, quantization, swizzled ldmatrix.

@unsafe
fn main() {
    // RT Core: hardware BVH nearest-neighbor search (128D query, k=8)
    let nns_res: I32 = rt_nearest_neighbor(128, 8);

    // Tensor Core: MMA projection on RT Core output
    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
    let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(nns_res);
    acc = mma_sync(frag_A, frag_B, frag_C);
}
```

```bash
cargo run -- tests/coprocessor_attention.ysu --emit-coprocessor
```

The compiler prints its scheduling report and writes `coprocessor_attention.coprocessor.ptx` and `.wrapped.ptx`. You can then benchmark it:

```bash
./venv/bin/python tests/benchmark_coprocessor_physical.py
```

Result on RTX 4070 Ti SUPER (10,000 iterations):
- Naive CUDA C++ (manual OptiX + WMMA): **4.22 µs**
- Y Co-Processor (auto-scheduled): **2.38 µs** → **1.77x speedup**

See **Section 11** for the full co-processor pipeline reference and more examples.

### Quick Reference: Compiler Flags

| Flag | Effect |
| :--- | :--- |
| *(none)* | Compile with LLVM backend → native binary via clang |
| `--llvm` | Explicit LLVM IR emission |
| `--c` | Emit portable C |
| `--ptx` | Emit raw NVIDIA PTX |
| `--emit-coprocessor` | Fused RT Core + Tensor Core co-processor PTX |
| `--probe` | Run hardware Sentinel probe and save `.ysu_hw_profile` |

### Where to Go Next

| Topic | Section |
| :--- | :--- |
| Language design philosophy | §1 |
| Compiler pipeline internals | §2 |
| Full grammar (EBNF) | §3 |
| Type system & memory spaces | §5 |
| Safety directives (`@safe`, `@bounds`, `@ZeroDrift`) | §6 |
| All attributes & decorators reference | §9 |
| Complete code examples (21 examples) | §10 |
| RT Core + Tensor Core co-processor pipeline | §11 |

---

## 1. Introduction & Design Philosophy

Traditional programming languages abstract away the underlying microarchitecture, resulting in suboptimal memory access patterns, cache thrashing, branch divergence, and thread serialization. Y flips this paradigm: **the compiler is co-designed with the hardware profile**.

### Core Pillars of Y:
1. **Hardware Sentience**: The compiler queries a local hardware profile (`.ysu_hw_profile`) generated by a Sentinel hardware probe. This profile reports features like L2 cache line size, SIMD vector lane sizes, thread scheduling costs, warp sizes, memory latency, and GPU execution latencies (Tensor Core, FMA, Shared Memory, etc.).
2. **Linear Memory Obligations**: The type-checker tracks the lifetime of asynchronous transactions (such as global-to-shared transfers) to prevent data races and ensure synchronization boundaries are met before values are consumed.
3. **Zero Bank Conflicts**: The compiler statically analyzes shared memory layouts and warp-level access index strides to predict and prevent bank conflicts.
4. **Explicit Hardware Mapping**: Variable allocations, layout qualifiers, and concurrency operations map directly to hardware mechanisms (such as C11 standard atomics/alignments, and LLVM IR volatile accesses, non-temporal cache bypasses, and inline assembly).

---

## 2. Compiler Pipeline Architecture

The Y compiler is designed as a multi-stage, high-throughput toolchain. Below is a detailed view of its execution flow:

```mermaid
graph TD
    A[Source Code .ysu] --> B[Lexical Analyzer]
    B -->|Token Stream| C[AST Parser]
    C -->|Abstract Syntax Tree| D[Semantic Type-Checker]
    D -->|Verified AST| E[Hardware Sentinel Probe Resolver]
    E -->|Lowered Intermediate Representation| F[Backend Emitters]
    F -->|LLVM Backend| G[LLVM IR -> Clang -> Native ELF]
    F -->|PTX Backend| H[NVIDIA PTX Assembly]
    F -->|AVX Backend| I[AVX-512 Vector Host Code]
```

### Compiler Phases:
1. **Lexical Analysis (`lexer.rs`)**: Tokenizes the raw source input into a flat token stream. Identifies keywords, datatypes, operators, and metadata decorators.
2. **Syntax Parsing (`parser.rs`)**: Consumes tokens and constructs an Abstract Syntax Tree (AST). Resolves module dependencies (`import`) recursively.
3. **Semantic Type Checking (`type_checker.rs`)**: Validates type safety, verifies structural alignments, ensures bank-conflict-free access patterns, and tracks linear memory obligations.
4. **Hardware Sentinel Resolver (`sentinel.rs`)**: Matches hardware constraints specified by `@require` decorators against physical microarchitectural capabilities.
5. **Backend Emission**: Translates verified code to native backends:
   * `llvm_emitter.rs`: Outputs target-specific LLVM IR with cache hints and atomic constraints.
   * `ptx_emitter.rs`: Emits highly optimized GPU PTX assembly.
   * `cpu_emitter.rs`: Generates target-specific AVX-512 vector code.

---

## 3. Formal EBNF Grammar Specification

Below is the formal Extended Backus-Naur Form (EBNF) grammar representing the syntax of the Y programming language:

```ebnf
Program         = { Item } ;
Item            = ImportDecl | StructDecl | EnumDecl | ImplBlock | FuncDecl | KernelDecl ;

ImportDecl      = "import" , Ident , { "::" , Ident } , ";" ;
StructDecl      = { Attr } , "struct" , Ident , [ Generics ] , "{" , { FieldDecl } , "}" ;
FieldDecl       = { Attr } , Ident , ":" , Type , "," ;

EnumDecl        = { Attr } , "enum" , Ident , "{" , { EnumVariant } , "}" ;
EnumVariant     = Ident , [ "(" , Type , ")" ] , "," ;

ImplBlock       = "impl" , [ Generics ] , Ident , [ Generics ] , "{" , { FuncDecl } , "}" ;

FuncDecl        = { Attr } , "fn" , Ident , [ Generics ] , ParameterList , [ "->" , Type ] , Block ;
KernelDecl      = { Attr } , "kernel" , Ident , ParameterList , Block ;

Generics        = "<" , Ident , { "," , Ident } , ">" ;
ParameterList   = "(" , [ Parameter , { "," , Parameter } ] , ")" ;
Parameter       = [ "mut" ] , Ident , ":" , Type ;

Type            = PrimitiveType | ArrayType | PointerType | ReferenceType | UserType ;
PrimitiveType   = "F16" | "BF16" | "TF32" | "F32" | "F64" | "I8" | "I16" | "I32" | "I64" | "U8" | "U16" | "U32" | "U64" | "bool" | "String" ;
ArrayType       = "[" , Type , ";" , Expr , "]" ;
PointerType     = "ptr" ;
ReferenceType   = "&" , [ "mut" ] , Type ;
UserType        = Ident , [ Generics ] ;

Block           = "{" , { Stmt } , "}" ;
Stmt            = LetStmt | AssignStmt | IfStmt | ForStmt | WhileStmt | ReturnStmt | ChiselStmt | ExprStmt ;

LetStmt         = { Attr } , "let" , [ "mut" ] , Ident , [ ":" , Type ] , [ "=" , Expr ] , ";" ;
AssignStmt      = Expr , "=" , Expr , ";" ;
IfStmt          = "if" , Expr , Block , [ "else" , Block ] ;
ForStmt         = "for" , Ident , "in" , Expr , ".." , Expr , [ "step" , Expr ] , Block ;
WhileStmt       = "while" , Expr , Block ;
ReturnStmt      = "return" , [ Expr ] , ";" ;
ChiselStmt      = "chisel" , "{" , { StringLiteral } , "}" ;
ExprStmt        = Expr , ";" ;

Expr            = BinaryExpr | UnaryExpr | PrimaryExpr ;
BinaryExpr      = Expr , BinaryOp , Expr ;
UnaryExpr       = UnaryOp , Expr ;
PrimaryExpr     = Ident | Literal | CallExpr | IndexExpr | MemberExpr | StructInit | ZeroInit | ParenExpr ;

CallExpr        = Expr , "(" , [ Expr , { "," , Expr } ] , ")" ;
IndexExpr       = Expr , "[" , Expr , "]" ;
MemberExpr      = Expr , "." , Ident ;
StructInit      = Ident , "{" , [ FieldInit , { "," , FieldInit } ] , "}" ;
FieldInit       = Ident , ":" , Expr ;
ZeroInit        = "{" , "}" ;
ParenExpr       = "(" , Expr , ")" ;

Attr            = "@" , Ident , [ "(" , [ Expr , { "," , Expr } ] , ")" ] ;
```

---

## 4. Generics and Monomorphization

Y supports parametric polymorphism (generics) for structs, implementations, and functions. Because execution depends on hardware layout boundaries, Y uses **static monomorphization**:

1. **AST Duplication**: When a generic type is instantiated (e.g. `RingBuffer<I32, 1024>`), the compiler duplicates the AST representation of the struct and its implementations.
2. **Generic Parameter Substitution**: Placeholders `T` and `SIZE` are replaced with concrete types and compile-time evaluated integer values.
3. **Type-Checking Specialized Code**: The monomorphized code undergoes semantic checks (including alignment constraints and hardware checks) unique to the specialized parameters. For example, a larger `SIZE` might trigger cache line warnings or GPU shared memory block layout shifts.

---

## 5. Type System & Memory Spaces

Y divides data types into two main categories: primitive scalar types and hardware-aware layout types.

### 5.1 Scalar and Compound Types
* **Floating-Point**: `F16` (half precision), `BF16` (bfloat16), `TF32` (TensorFloat-32), `F32` (single float), `F64` (double float).
* **Integers**: `I8` through `I64` (signed), `U8` through `U64` (unsigned) (e.g., standard sizing from 8-bit to 64-bit).
* **Fixed-Point**: `QFixed` types represent values using fixed fractional scaling. For example, `Q32.32` reserves 32 bits for the integer part and 32 bits for the fraction.
* **References**: `&T` represents an immutable reference; `&mut T` represents a mutable reference.
* **Arrays**: Array declarations use syntax like `[T; Size]` (e.g. `[I32; 1024]`).

### 5.2 Hardware Memory Spaces
To optimize data transfers, variables and buffers must reside in designated hardware memory spaces:

#### 1. GlobalMemory`<T>`
Global VRAM or System RAM. Large-capacity, high-latency. Must be accessed via cache policies or explicit streaming.
```ysu
let vram_ptr: GlobalMemory<F32> = ...;
```

#### 2. L2Memory`<T>`
Bypasses level 1 caches to interface directly with L2 caches. Useful for data shared across multiple thread blocks.
```ysu
let cache_ptr: L2Memory<I32> = ...;
```

#### 3. SharedMemory
GPU-resident Local SRAM. Shared across threads in a warp or block. Subject to bank conflicts.
```ysu
let smem_buffer = SharedMemory::alloc<SmemLayout<F32, rows=8, cols=32, swizzle=0>>();
```

#### 4. RegisterFile
Variables declared inside local function blocks map to registers. This is the fastest, lowest-latency storage area.
```ysu
let local_accumulator: F32 = 0.0;
```

---

## 6. Compile-Time Verification & Analysis

### 6.1 Structural Type Checking
Y utilizes structural type equivalence for layouts and complex variables. Types match if their memory footprint, byte offset boundaries, and layout alignments are identical, ensuring safe reinterpret casts.

**Example — type mismatch caught at compile time:**
```ysu
fn add(a: I32, b: F32) -> I32 {
    return a + b;  // error: cannot add I32 and F32 without explicit cast
}
```
```
error[E0308]: mismatched types
  --> add.ysu:2:14
   |
 2 |     return a + b;
   |              ^ expected `I32`, found `F32`
hint: use `b as I32` or promote `a` to `F32` before the operation
```

### 6.2 Linear Memory Obligations
Linear tracking enforces that resources (like asynchronous memory transfers) cannot be left in indeterminate states:

```
[Start cp_async] ---> State: InFlight (Transfer Obligation created)
                            |
                     [Attempt read] ---> Compile Error: Data race hazard!
                            |
                     [Pipeline::wait] -> State: Completed
                            |
                     [Attempt read] ---> State: Safe (Obligation resolved)
```

Every `Transfer` token returned by `cp_async` must be statically consumed by a corresponding `Pipeline::wait()` instruction before any read operations from that buffer are permitted.

**Example — unconsumed transfer token caught at compile time:**
```ysu
kernel bad_copy(A: GlobalMemory<F32>, buf: SharedMemory<F32>) {
    let tx: Transfer<Global, Shared, Async<1>, 128> = cp_async(A[0], buf);
    // error: Transfer token `tx` was never consumed via Pipeline::wait()
    let val: F32 = buf[0];  // read before wait -> data race
}
```
```
error[L0001]: linear obligation unresolved
  --> bad_copy.ysu:3:20
   |
 3 |     let val: F32 = buf[0];
   |                    ^^^^^^ buffer read before Transfer `tx` was awaited
hint: insert `pipe.wait(tx); barrier::sync();` before reading `buf`
```

### 6.3 Shared Memory Bank Conflict Solver
GPU shared memory consists of 32 parallel banks. The compiler maps variable read strides to bank indices:

$$\text{Bank} = \left(\frac{\text{Byte Offset}}{4}\right) \pmod{32}$$

If two threads within a 32-thread warp access indices with the same $\text{Bank}$ index in the same cycle, the compiler generates a bank conflict warning and advises layout swizzling (e.g., using `swizzle=330`).

**Example — bank conflict detected at compile time:**
```ysu
kernel conflicting_load(data: GlobalMemory<F32>) {
    // Stride of 32 floats = 128 bytes -> all 32 threads hit bank 0
    type BadLayout = SmemLayout<F32, rows=32, cols=32, swizzle=0>;
    let smem = SharedMemory::alloc<BadLayout>();
    let frag = ldmatrix(smem);  // 32-way bank conflict on every warp!
}
```
```
warning[B0001]: shared memory bank conflict detected
  --> conflicting_load.ysu:5:17
   |
 5 |     let frag = ldmatrix(smem);
   |                 ^^^^^^^ stride 128B causes 32-way conflict on bank 0
hint: use `swizzle=330` in SmemLayout to eliminate conflicts
```

### 6.4 Uninitialized Variable Check (inside `@safe` blocks)
Inside `@safe` scopes, all variables must be initialized before use. Uninitialized reads are a compile error, not a runtime crash.

**Example:**
```ysu
@safe
fn bad_read() -> I32 {
    let x: I32;       // declared but not initialized
    return x + 1;     // error: `x` used before initialization
}
```
```
error[S0002]: use of possibly uninitialized variable
  --> bad_read.ysu:3:12
   |
 2 |     let x: I32;
   |         - declared here, never assigned
 3 |     return x + 1;
   |            ^ `x` is uninitialized at this point
```

### 6.5 Static Bounds Violation
`@bounds(min, max)` annotations are checked at compile time for constant indices and as runtime assertions for dynamic indices.

**Example — constant index caught at compile time:**
```ysu
fn oob_access(data: [I32; 64]) -> I32 {
    @bounds(0 <= 99 < 64)   // error: 99 is outside [0, 64)
    return data[99];
}
```
```
error[B0002]: static bounds violation
  --> oob_access.ysu:2:14
   |
 2 |     @bounds(0 <= 99 < 64)
   |              ^^ index 99 is out of range [0, 64)
   |                       -- declared bound
```

---

## 7. Emitter Lowering & Code Generation

The following table details how high-level Y structures map directly to target LLVM IR code during lowering:

| Y Syntax / Decorator | LLVM IR Representation |
|---|---|
| `@atomic` (Field Load) | `%val = load atomic i32, ptr %ptr seq_cst, align 4` |
| `@atomic` (Field Store) | `store atomic i32 %val, ptr %ptr seq_cst, align 4` |
| `@align(N)` | `load i32, ptr %ptr, align N` or `store i32 %val, ptr %ptr, align N` |
| `@gpu_uncached` | `%val = load volatile ptr, ptr %ptr, !nontemporal !0` |
| Zero Initialization `{}` | `call void @llvm.memset.p0.i64(ptr %var, i8 0, i64 %size, i1 false)` |
| Array Indexing `arr[idx]` | `%ptr = getelementptr i32, ptr %arr, i32 %idx` |
| `@inline` | `attributes #0 = { alwaysinline }` |
| `@noinline` | `attributes #0 = { noinline }` |

---

## 8. Standard Library Reference

Y provides a built-in runtime library for environment interaction, system allocations, file access, and vector manipulation:

### 8.1 File I/O
* **`yfile_read_to_string(path: ptr) -> ptr`**: Reads file contents to a string pointer.
* **`yfile_write(path: ptr, content: ptr)`**: Writes string content to disk.

### 8.2 String Manipulation
* **`ystr_new(cstr: ptr) -> ptr`**: Creates a new Y-string from a C-string literal.
* **`ystr_push(s: ptr, ch: i8)`**: Appends a character.
* **`ystr_push_str(s: ptr, append: ptr)`**: Appends a string.
* **`ystr_eq_cstr(s: ptr, cstr: ptr) -> bool`**: Compares string contents.
* **`ystr_len(s: ptr) -> i64`**: Returns string byte count.
* **`ystr_char_at(s: ptr, idx: i64) -> i8`**: Returns character index.
* **`ystr_clone(s: ptr) -> ptr`**: Clones string allocation.

### 8.3 Vector Types
* **`yvec_new(capacity: i64) -> ptr`**: Allocates a new vector.
* **`yvec_push(vec: ptr, item: ptr)`**: Appends an item.
* **`yvec_get(vec: ptr, idx: i64) -> ptr`**: Retrieves element pointer.
* **`yvec_len(vec: ptr) -> i64`**: Returns vector element count.

### 8.4 General Utilities
* **`printf(fmt: ptr, ...)`**: Direct format printer.
* **`malloc(size: i64) -> ptr`**: Dynamic memory allocation.
* **`free(ptr: ptr)`**: Dynamic memory reclamation.
* **`exit(code: i32)`**: Aborts execution.
* **`println(s: ptr)`**: Prints a string to stdout with a trailing newline.
* **`print_int(val: i64)`**: Prints an integer.

---

## 9. Exhaustive Reference: Language Attributes (Decorators)

Decorators instruct the parser, type-checker, and backend emitters on how to handle specific variables, functions, and memory structures.

### 9.1 `@require`
* **Syntax**: `@require(hardware_feature_condition)`
* **Usage**: Placed above `kernel` or `fn` definitions.
* **Function**: Checks the user's `.ysu_hw_profile` at compile time. If the system does not support the requested features, compilation terminates.
* **Example**:
```ysu
@require(avx512 >= 1)
fn vector_add_avx512(A: &mut [F32; 16], B: &[F32; 16]) {
    // LLVM backend lowers this to AVX-512 register instructions
}
```

### 9.2 `@cache_policy`
* **Syntax**: `@cache_policy(PolicyType, [options])`
* **Usage**: Decorator for let-bindings that load from memory.
* **Function**: Tells the compiler which memory load instructions to emit to optimize cache usage.
* **Policies**:
  * `L2_PERSIST`: Flags memory pages to remain resident in L2 cache.
  * `L2_EVICT_FIRST`: Evicts the loaded cache line as soon as possible to free up cache space.
  * `L2_EVICT_LAST`: Prevents early eviction of this data.
  * `L2_STREAM`: Streams data directly to registers, bypassing the cache entirely.
* **Example**:
```ysu
// Keep weights in L2 Cache for reuse
@cache_policy(L2_PERSIST, reuse_count=16)
let weight_val: F32 = load(global_weights);
```

### 9.3 `@atomic`
* **Syntax**: `@atomic`
* **Usage**: Attribute on struct field declarations or variable bindings.
* **Function**:
  * **C Backend**: Translates to `_Atomic`.
  * **LLVM Backend**: Translates to atomic instructions (`load atomic ... seq_cst`, `store atomic ... seq_cst`).
* **Example**:
```ysu
struct Lock {
    @atomic state: I32,
}

fn lock_acquire(l: &mut Lock) {
    while l.state == 1 {
        // Spin lock
    }
    l.state = 1; // Atomic write
}
```

### 9.4 `@align`
* **Syntax**: `@align(ByteBoundary)`
* **Usage**: Applied to struct fields or variables.
* **Function**: Prevents cache-line false sharing in multi-threaded environments.
  * **C Backend**: Lowers to C11 `_Alignas(ByteBoundary)`.
  * **LLVM Backend**: Generates `, align ByteBoundary` qualifiers on loads and stores.
* **Example**:
```ysu
struct ThreadState {
    @align(64) @atomic head: I32, // Cache line size alignment (64 bytes)
    @align(64) @atomic tail: I32, // Resides in a separate cache line to prevent false-sharing
}
```

### 9.5 `@gpu_uncached`
* **Syntax**: `@gpu_uncached`
* **Usage**: Applied to memory buffers or struct fields.
* **Function**: Bypasses GPU caches entirely. Useful for ring buffers, shared state, and GPU-to-CPU status flags.
  * **C Backend**: Lowers to `volatile` qualifier.
  * **LLVM Backend**: Emits `volatile` loads/stores along with `!nontemporal !0` metadata to instruct clang to bypass the cache hierarchy.
* **Example**:
```ysu
struct IPCChannel {
    @gpu_uncached status: I32, // Bypasses cache to guarantee immediate visibility
}
```

### 9.6 `@ZeroDrift`
* **Syntax**: `@ZeroDrift`
* **Usage**: Let-bindings for float accumulations.
* **Function**: Inserts software compensation blocks (Kahan summation algorithm) to eliminate precision loss caused by floating-point rounding errors.
* **Example**:
```ysu
@ZeroDrift
let sum: F32 = 0.0;
for i in 0..1000000 {
    sum += data[i]; // Compensated mathematically at runtime
}
```

### 9.7 `@ptx_emit`, `@avx_emit`, and `@hdl_emit`
* **Syntax**: `@ptx_emit`, `@avx_emit`, `@hdl_emit`
* **Usage**: Annotations on functions or kernels.
* **Function**: Forces the compiler to lower the decorated function to a specific backend assembly format (NVIDIA PTX, CPU AVX, or Verilog/HDL).
* **Example**:
```ysu
@ptx_emit
fn gpu_special_op(a: F32) -> F32 {
    // Lowers directly to PTX instructions
}
```

### 9.8 `@inline` and `@noinline`
* **Syntax**: `@inline`, `@noinline`
* **Usage**: Function annotations.
* **Function**: Controls inlining optimizations.
* **Example**:
```ysu
@noinline
fn complex_slow_path() {
    // Prevents code size inflation
}
```

### 9.9 `@safe` and `@unsafe`
* **Syntax**: `@safe`, `@unsafe`
* **Usage**: Blocks or function annotations.
* **Function**: Toggles compile-time memory safety checks (e.g. pointer arithmetic and out-of-bounds array indexing).
* **Example**:
```ysu
@unsafe {
    let raw_addr: ptr = malloc(1024);
    // Arbitrary pointer arithmetic allowed
}
```

### 9.10 `@bounds`
* **Syntax**: `@bounds(condition)`
* **Usage**: Variable declarations or loops.
* **Function**: Asserts index limits to skip runtime bounds checks.
* **Example**:
```ysu
@bounds(0 <= index < 256)
let element: I32 = my_array[index]; // Skips safety bounds checks
```

### 9.11 `@invariant`
* **Syntax**: `@invariant(condition)`
* **Usage**: Iteration/loop loops scopes.
* **Function**: Declares a condition that is statically proven to hold true before and after each loop iteration.
* **Example**:
```ysu
for i in 0..100 {
    @invariant(accumulator >= 0)
    accumulator = accumulator + data[i];
}
```

### 9.12 `@tile`
* **Syntax**: `@tile(dimension_size, stride_step)`
* **Usage**: Nested loops or parallel iterations.
* **Function**: Optimizes layout caches by breaking 2D/3D operations into smaller block execution units.
* **Example**:
```ysu
for i in 0..1024 {
    @tile(16, 4)
    do_compute(i);
}
```

### 9.13 `@ghost`
* **Syntax**: `@ghost`
* **Usage**: Variable declarations or code scopes.
* **Function**: Declares variables that only exist for static compile-time constraint validation and assertions. These are completely stripped out by codegen and cost zero execution cycles.
* **Example**:
```ysu
@ghost let verification_step: I32 = 0;
for i in 0..10 {
    @ghost {
        verification_step = verification_step + 1;
    }
}
```

### 9.14 `@prefetch_stride`
* **Syntax**: `@prefetch_stride(byte_width)`
* **Usage**: Loop structures.
* **Function**: Inserts cache lines prefetch instructions (`_mm_prefetch` in AVX, `prefetch` in PTX) targeting memory boundaries offset by the stride width.
* **Example**:
```ysu
for i in 0..512 {
    @prefetch_stride(64) // Prefetch next cache line (64 bytes ahead)
    process(data[i]);
}
```

### 9.15 `@clock_domain`
* **Syntax**: `@clock_domain(name_string)`
* **Usage**: Fields, variables, or function blocks compiled with `@hdl_emit`.
* **Function**: Assigns signal registers to specific hardware clock domains, forcing the synthesis engine to insert synchronizers at crosses.
* **Example**:
```ysu
@clock_domain("clk_125mhz")
struct Receiver {
    data: I32,
    ready: bool,
}
```

### 9.16 `@divergence`
* **Syntax**: `@divergence(uniform | branchy)`
* **Usage**: Loop structures or branch decisions.
* **Function**: Instructs GPGPU emitters on whether warp threads will trace parallel divergent paths, allowing register file and barrier optimizations.
* **Example**:
```ysu
@divergence(uniform)
if global_thread_id < warp_limit {
    execute_warp_op();
}
```

---

## 10. Complete Code Examples

### Example 1: Lock-Free Single-Producer Single-Consumer (SPSC) Ring Buffer
```ysu
// Lock-Free Single-Producer Single-Consumer Ring Buffer
struct RingBuffer {
    @align(64) @atomic head: I32,
    @align(64) @atomic tail: I32,
    @gpu_uncached buffer: [I32; 1024],
}

fn try_enqueue(rb: &mut RingBuffer, item: I32) -> bool {
    let h: I32 = rb.head;
    let t: I32 = rb.tail;
    
    // Check if buffer is full
    let next_head: I32 = (h + 1) % 1024;
    if next_head == t {
        return false;
    }

    // Write item directly to uncached memory
    rb.buffer[h] = item;

    // Atomic release store update
    rb.head = next_head;
    return true;
}

fn try_dequeue(rb: &mut RingBuffer, out_item: &mut I32) -> bool {
    let h: I32 = rb.head;
    let t: I32 = rb.tail;

    // Check if buffer is empty
    if h == t {
        return false;
    }

    // Read item directly from uncached memory
    *out_item = rb.buffer[t];

    // Atomic release store update
    rb.tail = (t + 1) % 1024;
    return true;
}

fn main() -> I32 {
    let rb: RingBuffer = {};
    let success: bool = try_enqueue(&mut rb, 1337);
    
    let val: I32 = 0;
    let dequeued: bool = try_dequeue(&mut rb, &mut val);
    
    return val;
}
```

---

### Example 2: Matrix Multiplication (GEMM) Kernel
```ysu
@require(avx512 >= 1)
kernel matmul(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {
    // 16x64 Swizzled Shared Memory Layout
    type ATile = SmemLayout<F16, rows=16, cols=64, swizzle=330>;
    let smem_A = SharedMemory::alloc<ATile>();

    // Load inputs with persisting L2 cache policy
    @cache_policy(L2_PERSIST, reuse_count=8)
    let weights: F16 = load(A);

    // Load dynamic inputs with evict first policy
    @cache_policy(L2_EVICT_FIRST)
    let act: F16 = load(B);
    
    // Fragment registers for Tensor Core MMA
    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
    let pipe: Pipeline<stages=2, layout=ATile> = Pipeline::init();

    for k in 0..1024 step 16 {
        // Asynchronous transfer from global memory to swizzled shared memory
        let tx_A: Transfer<Global, Shared, Async<1>, 128> = cp_async(A[k], smem_A);
        
        // Wait for pipeline stages
        pipe.wait(tx_A);
        
        // Synchronize warp thread accesses
        barrier::sync();
        
        // Load data from shared memory into register fragments
        let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(smem_A);
        let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(smem_A);
        let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(smem_A);
        
        // Low level assembly injection
        chisel {
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [smem_ptr];";
        }

        // Perform Warp-Level Matrix Multiply Accumulate (MMA)
        acc = mma_sync(frag_A, frag_B, frag_C); 
    }

    // Write final output register fragments back to global memory
    store(acc, C);
}
```

---

### Example 3: Stream Vector Addition with L2 Bypass
```ysu
kernel stream_vector_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, N: I32) {
    for i in 0..N {
        // Stream data to bypass L1/L2 caches
        @cache_policy(L2_STREAM)
        let a_val: F32 = A[i];

        @cache_policy(L2_STREAM)
        let b_val: F32 = B[i];

        @cache_policy(L2_STREAM)
        C[i] = a_val + b_val;
    }
}
```

---

### Example 4: Fixed-Point Signal Filtering Algorithm
```ysu
struct FilterState {
    coefficient: Q32.32,
    prev_value: Q32.32,
}

fn apply_filter(state: &mut FilterState, signal: Q32.32) -> Q32.32 {
    // Multiply signal with coefficient in fixed-point math
    let filtered: Q32.32 = (signal * state.coefficient) + (state.prev_value * (1.0 - state.coefficient));
    state.prev_value = filtered;
    return filtered;
}

fn main() -> I32 {
    let filter = FilterState {
        coefficient: 0.25,
        prev_value: 0.0,
    };
    
    let raw_val: Q32.32 = 42.0;
    let filtered_val = apply_filter(&mut filter, raw_val);
    return 0;
}
```

---

### Example 5: Multi-Stage Pipeline Overlapping
```ysu
type BufferLayout = SmemLayout<F32, rows=8, cols=32, swizzle=0>;

kernel pipelined_copy(A: GlobalMemory<F32>, B: GlobalMemory<F32>, N: I32) {
    let buf0 = SharedMemory::alloc<BufferLayout>();
    let buf1 = SharedMemory::alloc<BufferLayout>();
    
    let pipe: Pipeline<stages=2, layout=BufferLayout> = Pipeline::init();

    // Stage 0: Prefetch first block
    let tx0 = cp_async(A[0], buf0);
    pipe.wait(tx0);
    barrier::sync();

    for i in 1..N {
        // Overlap loading next block with processing current block
        let tx_next = if i % 2 == 1 {
            cp_async(A[i * 256], buf1)
        } else {
            cp_async(A[i * 256], buf0)
        };

        // Process current block
        if i % 2 == 1 {
            process_data(buf0);
        } else {
            process_data(buf1);
        }

        pipe.wait(tx_next);
        barrier::sync();
    }
}

fn process_data(buf: &mut BufferLayout) {
    // Local memory processing logic
}
```

---

### Example 6: Compiler Verification & Safety Asserts
```ysu
@safe
fn verify_computation(data: &mut [I32; 100]) -> I32 {
    let sum: I32 = 0;
    
    for i in 0..100 {
        @invariant(sum >= 0)
        @bounds(0 <= i < 100)
        
        let val: I32 = data[i];
        if val > 0 {
            sum += val;
        }
    }
    
    return sum;
}

fn main() -> I32 {
    let data: [I32; 100] = {};
    let sum = verify_computation(&mut data);
    return 0;
}
```

---

### Example 7: Custom Memory Vector Allocation
```ysu
struct FloatVector {
    data: &mut F32,
    size: I64,
    capacity: I64,
}

@unsafe
fn vector_init(capacity: I64) -> FloatVector {
    let raw_mem: ptr = malloc(capacity * 4); // float is 4 bytes
    return FloatVector {
        data: raw_mem,
        size: 0,
        capacity: capacity,
    };
}

@unsafe
fn vector_push(v: &mut FloatVector, val: F32) {
    if v.size >= v.capacity {
        let new_capacity: I64 = v.capacity * 2;
        let new_mem: ptr = malloc(new_capacity * 4);
        
        // Copy elements manually
        for i in 0..v.size {
            new_mem[i] = v.data[i];
        }
        
        free(v.data);
        v.data = new_mem;
        v.capacity = new_capacity;
    }
    
    v.data[v.size] = val;
    v.size += 1;
}

@unsafe
fn vector_free(v: &mut FloatVector) {
    free(v.data);
    v.size = 0;
    v.capacity = 0;
}

fn main() -> I32 {
    @unsafe {
        let v: FloatVector = vector_init(4);
        vector_push(&mut v, 10.0);
        vector_push(&mut v, 20.0);
        vector_free(&mut v);
    }
    return 0;
}
```

---

### Example 8: Multi-Threaded Cache Warming & Prefetching
```ysu
@require(avx512 >= 1)
fn cache_warm_process(data: GlobalMemory<F32>, result: GlobalMemory<F32>, N: I32) {
    // Warm L2 lines via structured stride prefetching
    for i in 0..N step 16 {
        @prefetch_stride(64)
        @cache_policy(L2_PERSIST)
        let block: F32 = data[i];

        let acc: F32 = 0.0;
        for j in 0..16 {
            acc += block[j] * 2.5;
        }

        result[i] = acc;
    }
}
```

---

### Example 9: Multi-Clock Domain Signal Crosser
```ysu
struct CrossDomainRegister {
    @clock_domain("clk_fast") fast_val: I32,
    @clock_domain("clk_slow") slow_val: I32,
    @atomic handshake: bool,
}

@hdl_emit
fn cross_signal(cdr: &mut CrossDomainRegister) {
    @clock_domain("clk_fast") {
        if cdr.handshake == false {
            cdr.fast_val = 42;
            cdr.handshake = true;
        }
    }

    @clock_domain("clk_slow") {
        if cdr.handshake == true {
            cdr.slow_val = cdr.fast_val;
            cdr.handshake = false;
        }
    }
}
```

---

### Example 10: Metastability Verification Ghost State
```ysu
struct SyncState {
    value: I32,
    ready: bool,
    @ghost verification_sync_count: I32,
}

fn step_synchronization(state: &mut SyncState, signal: I32) {
    if state.ready == false {
        state.value = signal;
        state.ready = true;
        
        @ghost {
            state.verification_sync_count += 1;
            @bounds(state.verification_sync_count < 2)
        }
    } else {
        state.ready = false;
    }
}
```

---

### Example 11: Real-Time Audio DSP Biquad Filter
```ysu
struct BiquadFilter {
    // Coeffs aligned to L2 cache lines to prevent eviction during DSP loops
    @align(64) @cache_policy(L2_PERSIST) b0: F32,
    @align(64) @cache_policy(L2_PERSIST) b1: F32,
    @align(64) @cache_policy(L2_PERSIST) b2: F32,
    @align(64) @cache_policy(L2_PERSIST) a1: F32,
    @align(64) @cache_policy(L2_PERSIST) a2: F32,
    
    // History states
    x1: F32,
    x2: F32,
    y1: F32,
    y2: F32,
}

fn process_audio_frame(filter: &mut BiquadFilter, input: GlobalMemory<F32>, output: GlobalMemory<F32>, len: I32) {
    for i in 0..len {
        @prefetch_stride(64)
        let sample: F32 = input[i];
        
        // Biquad difference equation
        let out_sample: F32 = (filter.b0 * sample) 
                            + (filter.b1 * filter.x1) 
                            + (filter.b2 * filter.x2) 
                            - (filter.a1 * filter.y1) 
                            - (filter.a2 * filter.y2);
                            
        // Update history
        filter.x2 = filter.x1;
        filter.x1 = sample;
        filter.y2 = filter.y1;
        filter.y1 = out_sample;
        
        @cache_policy(L2_STREAM)
        output[i] = out_sample;
    }
}
```

---

### Example 12: Lock-Free Hazard Pointers
```ysu
struct HazardPointer {
    @atomic active_ptr: ptr,
    @align(64) owner_thread_id: I32,
}

struct HazardRegistry {
    pointers: [HazardPointer; 64],
}

@unsafe
fn acquire_hazard_ptr(registry: &mut HazardRegistry, thread_id: I32, target: ptr) -> bool {
    for i in 0..64 {
        if registry.pointers[i].owner_thread_id == thread_id {
            // Atomic store hazard pointer
            registry.pointers[i].active_ptr = target;
            return true;
        }
    }
    return false;
}

@unsafe
fn release_hazard_ptr(registry: &mut HazardRegistry, thread_id: I32) {
    for i in 0..64 {
        if registry.pointers[i].owner_thread_id == thread_id {
            registry.pointers[i].active_ptr = null;
            break;
        }
    }
}
```

---

### Example 13: Zero-Copy CUDA IPC Inter-Process Shared State
```ysu
struct SharedIPCChannel {
    @align(64) @atomic head: U64,
    @align(64) @atomic tail: U64,
    @gpu_uncached buffer_ready: bool,
    @gpu_uncached data_payload: [I32; 512],
}

@unsafe
fn send_ipc_payload(channel: &mut SharedIPCChannel, src: ptr, size: I32) -> bool {
    if channel.buffer_ready == true {
        return false; // Receiver has not consumed the last frame
    }
    
    @bounds(0 <= size <= 512)
    for i in 0..size {
        channel.data_payload[i] = src[i];
    }
    
    // Memory fence via atomic release state flag
    channel.buffer_ready = true;
    channel.head = channel.head + 1;
    return true;
}
```

---

### Example 14: Parallel Monte Carlo Option Pricer
```ysu
@require(avx512 >= 1)
fn monte_carlo_step(paths: GlobalMemory<F32>, strikes: GlobalMemory<F32>, results: GlobalMemory<F32>, size: I32) {
    // Processes 16 elements simultaneously using AVX-512 vector lanes
    for i in 0..size step 16 {
        @prefetch_stride(64)
        let path_vector: F32 = paths[i];
        let strike_vector: F32 = strikes[i];
        
        let payoff: F32 = 0.0;
        for lane in 0..16 {
            let diff: F32 = path_vector[lane] - strike_vector[lane];
            if diff > 0.0 {
                payoff[lane] = diff;
            } else {
                payoff[lane] = 0.0;
            }
        }
        
        @cache_policy(L2_STREAM)
        results[i] = payoff;
    }
}
```

---

### Example 15: Multi-Producer Multi-Consumer (MPMC) Queue
```ysu
struct QueueNode<T> {
    @atomic sequence: U64,
    data: T,
}

struct MpmcQueue<T> {
    @align(64) @atomic enqueue_pos: U64,
    @align(64) @atomic dequeue_pos: U64,
    buffer: [QueueNode<T>; 1024],
}

fn try_enqueue_mpmc<T>(q: &mut MpmcQueue<T>, item: T) -> bool {
    let pos: U64 = q.enqueue_pos;
    
    // Spin lock or CAS on queue position
    let node_idx: U64 = pos % 1024;
    let seq: U64 = q.buffer[node_idx].sequence;
    let diff: I64 = seq - pos;
    
    if diff == 0 {
        // CAS increment pos
        q.enqueue_pos = pos + 1;
        q.buffer[node_idx].data = item;
        q.buffer[node_idx].sequence = pos + 1;
        return true;
    }
    
    return false;
}

fn try_dequeue_mpmc<T>(q: &mut MpmcQueue<T>, out_item: &mut T) -> bool {
    let pos: U64 = q.dequeue_pos;
    let node_idx: U64 = pos % 1024;
    let seq: U64 = q.buffer[node_idx].sequence;
    let diff: I64 = seq - (pos + 1);
    
    if diff == 0 {
        q.dequeue_pos = pos + 1;
        *out_item = q.buffer[node_idx].data;
        q.buffer[node_idx].sequence = pos + 1024;
        return true;
    }
    
    return false;
}
```

---

### Example 16: Fast Fourier Transform (FFT) Shared Memory Butterfly
```ysu
type FFTLayout = SmemLayout<F32, rows=8, cols=32, swizzle=330>;

kernel fft_radix2_butterfly(data: GlobalMemory<F32>, N: I32) {
    let smem_real = SharedMemory::alloc<FFTLayout>();
    let smem_imag = SharedMemory::alloc<FFTLayout>();
    
    // Load data from global memory into swizzled shared memory
    for i in 0..N {
        smem_real[i] = data[i * 2];
        smem_imag[i] = data[i * 2 + 1];
    }
    
    barrier::sync();
    
    // Radix-2 Butterfly stages
    let half_n: I32 = N / 2;
    for stage in 1..half_n {
        let span: I32 = 1 << stage;
        for j in 0..half_n {
            let idx_a: I32 = j * 2;
            let idx_b: I32 = idx_a + span;
            
            // Radix-2 butterfly arithmetic
            let r_a: F32 = smem_real[idx_a];
            let r_b: F32 = smem_real[idx_b];
            
            smem_real[idx_a] = r_a + r_b;
            smem_real[idx_b] = r_a - r_b;
        }
        barrier::sync();
    }
    
    // Write results back
    for i in 0..N {
        data[i * 2] = smem_real[i];
        data[i * 2 + 1] = smem_imag[i];
    }
}
```

---

### Example 17: GPU Thermal and Power-Aware Execution
```ysu
struct DeviceState {
    temperature: I32,
    throttling_active: bool,
}

fn adjust_execution_profile(state: &mut DeviceState) {
    // Read local GPU hardware stats
    if state.temperature > 85 {
        state.throttling_active = true;
        
        // Emits power-save sleep delay inside execution threads
        chisel {
            "nanosleep.u32 1000;";
        }
    } else {
        state.throttling_active = false;
    }
}
```

---

### Example 18: Concurrent Hopscotch Hash Map
```ysu
struct HashBucket<K, V> {
    @atomic hop_info: U32,
    key: K,
    value: V,
    @atomic state: I32, // 0=Empty, 1=Busy, 2=Occupied
}

struct HopscotchMap<K, V> {
    buckets: [HashBucket<K, V>; 4096],
}

fn insert_map<K, V>(map: &mut HopscotchMap<K, V>, key: K, value: V) -> bool {
    let hash: U32 = calculate_hash(key);
    let bucket_idx: U32 = hash % 4096;
    
    // Acquire bucket atomically
    let expected: I32 = 0;
    if compare_and_swap(&mut map.buckets[bucket_idx].state, &mut expected, 1) {
        map.buckets[bucket_idx].key = key;
        map.buckets[bucket_idx].value = value;
        map.buckets[bucket_idx].state = 2; // Set occupied
        return true;
    }
    return false;
}
```

---

### Example 19: Bounding Volume Hierarchy (BVH) Traverse Kernel
```ysu
struct BvhNode {
    min_bounds: [F32; 3],
    max_bounds: [F32; 3],
    left_child: I32,
    right_child: I32,
}

kernel traverse_bvh(nodes: GlobalMemory<BvhNode>, ray_origin: [F32; 3], ray_dir: [F32; 3], hit_node: &mut I32) {
    let stack: [I32; 32] = {};
    let stack_ptr: I32 = 0;
    stack[0] = 0; // Root node index
    
    while stack_ptr >= 0 {
        let curr_idx: I32 = stack[stack_ptr];
        stack_ptr = stack_ptr - 1;
        
        // Persistently load BvhNodes to bypass dynamic cache eviction during recursive lookups
        @cache_policy(L2_PERSIST)
        let node: BvhNode = nodes[curr_idx];
        
        if ray_intersects_box(ray_origin, ray_dir, node.min_bounds, node.max_bounds) {
            if node.left_child == -1 {
                // Leaf node intersection detected
                *hit_node = curr_idx;
                return;
            } else {
                stack_ptr = stack_ptr + 1;
                stack[stack_ptr] = node.left_child;
                stack_ptr = stack_ptr + 1;
                stack[stack_ptr] = node.right_child;
            }
        }
    }
}
```

---

### Example 20: Half-Precision Deep Learning Adam Optimizer Kernel
```ysu
kernel adam_optimizer(
    weights: GlobalMemory<TF32>,
    grads: GlobalMemory<TF32>,
    m: GlobalMemory<TF32>,
    v: GlobalMemory<TF32>,
    beta1: TF32,
    beta2: TF32,
    epsilon: TF32,
    lr: TF32,
    size: I32
) {
    for i in 0..size {
        let g: TF32 = grads[i];
        
        // Update biased first moment estimate
        let m_new: TF32 = (beta1 * m[i]) + ((1.0 - beta1) * g);
        m[i] = m_new;
        
        // Update biased second raw moment estimate
        let v_new: TF32 = (beta2 * v[i]) + ((1.0 - beta2) * (g * g));
        v[i] = v_new;
        
        // Compute weight step
        let step: TF32 = lr * m_new / (sqrt(v_new) + epsilon);
        
        // Non-temporal write update weights bypassing cache
        @cache_policy(L2_STREAM)
        weights[i] = weights[i] - step;
    }
}
```

---

### Example 21: Block-Level `@safe` and `@unsafe` Safety Scope Boundaries
```ysu
struct RawBuffer {
    data_ptr: ptr,
    element_count: I32,
}

fn process_buffer_safety(buf: &mut RawBuffer, index: I32, value: I32) -> bool {
    // Transition to unsafe pointer manipulation
    @unsafe {
        let base: ptr = buf.data_ptr;
        let target_address: ptr = base + (index * 4); // Raw pointer math
        
        // Write value to raw address
        *target_address = value;
    }
    
    // Nest a safe verification scope
    @safe {
        // Enforced strict structural limits
        if index >= 0 && index < buf.element_count {
            return true;
        }
    }
    
    return false;
}

fn main() -> I32 {
    let buf = RawBuffer {
        data_ptr: malloc(1024),
        element_count: 256,
    };
    
    let ok: bool = process_buffer_safety(&mut buf, 10, 42);
    free(buf.data_ptr);
    return 0;
}
```

---

## 11. Hardware-Sentient Dual-Accelerator Co-Processing Pipeline

Y's co-processor backend (`--emit-coprocessor`) automatically fuses **RT Core** (ray tracing / BVH traversal) and **Tensor Core** (matrix multiply-accumulate) workloads within a single GPU kernel. The developer writes a high-level description of the compute intent; the compiler generates the full fused PTX including sync barriers, quantization passes, and bank-conflict-free shared memory layouts.

---

### 11.1 The Problem: Manual Fusion is Hard

On modern NVIDIA architectures (Ampere, Ada Lovelace, Blackwell), RT Cores and Tensor Cores are useful together but extremely difficult to combine by hand due to three fundamental mismatches:

| Mismatch | RT Core | Tensor Core |
| :--- | :--- | :--- |
| **Timing** | Asynchronous, non-deterministic (depends on BVH depth) | Synchronous, lock-step warp instructions |
| **Precision** | Outputs FP32 hit distances and indices | Requires packed FP16/BF16 input fragments |
| **Memory handoff** | Writes to shared memory (FP32) | Reads swizzled shared memory (FP16, bank-conflict-free) |

Bridging these by hand requires: manual `bar.sync` fence placement, explicit `cvt.rn.f16x2.f32` packing loops, hand-computed swizzle address offsets, and careful shared memory budget management to avoid `CUDA_ERROR_INVALID_PTX` from exceeding the 48 KB SM limit.

Y eliminates all of this at compile time.

---

### 11.2 Compiler Architecture

The co-processor pipeline adds four new modules to the compiler:

#### `ir_grapher.rs` — IR Dependency Graph
Builds a directed acyclic graph of all RT Core and Tensor Core nodes in the kernel, resolves their data dependencies (cross-pipeline edges), and computes the critical path through the mixed execution graph.

Output:
```
RT Core nodes:     1
Tensor Core nodes: 5
Cross-pipe edges:  3
Critical path:     250 cycles
```

#### `coprocessor_scheduler.rs` — Hardware-Sentient Scheduler
Uses the hardware profile (RT traversal latency, Tensor Core MMA latency, SMEM access cost) to schedule the fused kernel timeline. It:
- Allocates a single unified `coprocessor_smem` shared memory buffer covering both RT scratch and Tensor Core fragment staging.
- Places `bar.sync` barriers at minimum-cost cut points on the data dependency graph.
- Overlaps RT Core traversal latency with independent scalar instructions (register initialization, fragment zeroing) to hide stall cycles.

Output:
```
SMEM budget:       33280 bytes
Sync barriers:     1
Est. parallel cy:  215
Overlap savings:   133 cycles
Barrier 0: FP32 -> FP16 quantization (16384 bytes)
```

#### `quantization_pass.rs` — Vectorized FP32→FP16 Pass
Detects precision boundaries in the data flow (RT Core outputs FP32; `ldmatrix` requires FP16) and emits a vectorized conversion loop using `cvt.rn.f16x2.f32` — packing two FP32 values into one `half2` register per instruction. The destination layout is swizzled to be bank-conflict-free for subsequent `ldmatrix.sync.aligned` loads.

#### `rt_core_emitter.rs` — Unified SMEM Emitter
Emits the RT Core traversal PTX. Crucially, all RT scratch and output writes are **aliased directly to `coprocessor_smem` at the scheduler-provided offset** — eliminating the double-allocation bug (separate `rt_scratch` + global `coprocessor_smem`) that causes static shared memory to overflow the 48 KB hardware limit at high dimensions.

```ptx
// Y emits this — no separate .shared declaration:
mov.u64 %rt_scratch_base, coprocessor_smem;
add.u64 %rt_scratch_base, %rt_scratch_base, <scheduler_offset>;
st.shared.f32 [%rt_scratch_base + %offset], %dist_reg;
```

---

### 11.3 Compiler Pipeline (Co-Processor Path)

```
source (.ysu)
  → lexer / parser / type_checker   (standard pipeline)
  → ir_grapher.rs                   build RT+Tensor dependency graph
  → coprocessor_scheduler.rs        schedule timeline, allocate coprocessor_smem, place barriers
  → quantization_pass.rs            insert vectorized FP32→FP16 conversion loop
  → rt_core_emitter.rs              emit RT traversal PTX (aliased to coprocessor_smem)
  → ptx_emitter.rs                  emit Tensor Core MMA PTX
  → wrap_ptx()                      produce final .wrapped.ptx with correct .visible .entry and .shared declarations
  → CuPy / nvcc JIT                 load onto GPU driver
```

Invoked with:
```bash
cargo run -- tests/coprocessor_attention.ysu --emit-coprocessor
```

---

### 11.4 RT Core Intrinsic: `rt_nearest_neighbor`

```
rt_nearest_neighbor(dims: I32, k: I32) -> I32
```

**Parameters:**
- `dims` — dimensionality of the embedding space (e.g. `128` or `256`). The compiler maps this down to 3D/4D BVH leaf spheres using the hardware-optimal projection.
- `k` — number of nearest neighbors to retrieve.

**Returns:** An `I32` handle (`nns_res`) consumed downstream by `ldmatrix` to load the RT Core outputs as Tensor Core input fragments.

**Compiler actions:**
1. Emits a BVH traversal loop over `k` nearest-neighbor slots, computing ray-AABB intersection distances.
2. Writes all `k` FP32 distances and indices to `coprocessor_smem[scheduler_offset .. scheduler_offset + k*4]`.
3. Registers a cross-pipeline data edge to all downstream `ldmatrix(nns_res)` consumers, so the scheduler knows to place a `bar.sync` before quantization.

**Directives accepted on `rt_nearest_neighbor`:**

| Directive | Effect |
| :--- | :--- |
| `@divergence(uniform)` | Assert all warp threads query the same BVH depth (skips divergence penalty in cycle model) |
| `@cache_policy(L2_PERSIST)` | Mark BVH node loads as L2-persistent to reduce repeated eviction during deep traversals |

---

### 11.5 Tensor Core Intrinsics (Co-Processor Context)

When used after an `rt_nearest_neighbor` call, the standard Tensor Core intrinsics gain additional compiler-managed behavior:

#### `ldmatrix(nns_res)`

Loads a matrix fragment from the quantized RT Core output region in `coprocessor_smem`. The compiler:
- Computes the bank-conflict-free swizzled source address automatically from the scheduler's SMEM layout.
- Emits `ldmatrix.sync.aligned.m8n8.x4.shared.b16` for A fragments and `.x2` for B fragments.

#### `Fragment::zero()`

Initializes accumulator registers to zero. In co-processor context, the compiler overlaps this initialization with the RT Core traversal to hide latency.

#### `mma_sync(frag_A, frag_B, frag_C)`

Emits `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. In co-processor context, the compiler validates that `bar.sync` was correctly placed before this instruction's SMEM reads.

---

### 11.6 Shared Memory Model

All co-processor kernels use a single unified `coprocessor_smem` buffer. The scheduler partitions it into regions:

```
coprocessor_smem layout (example: 256D k=16 FRNN):
┌─────────────────────────────────┐  offset 0
│  RT Core scratch (FP32)         │  16384 bytes  (k=16 * dims_proj * 4)
├─────────────────────────────────┤  offset 16384
│  Quantized FP16 staging         │  8192  bytes  (half2 packed)
├─────────────────────────────────┤  offset 24576
│  Tensor Core fragment tiles     │  8704  bytes  (5 MMA tiles)
└─────────────────────────────────┘  total: 33280 bytes
```

The compiler enforces that the total never exceeds the SM shared memory limit (48 KB for Ada Lovelace). If it would, a compile-time error is emitted:
```
error: coprocessor_smem budget (49920 bytes) exceeds SM limit (49152 bytes)
hint: reduce k or dims, or split into multiple barrier stages
```

---

### 11.7 Scheduling Statistics Output

When `--emit-coprocessor` is passed, the compiler prints a static scheduling report:

```
-> Phase A: IR Dependency Graphing...
   RT Core nodes:     1
   Tensor Core nodes: 5
   Cross-pipe edges:  3
   Critical path:     250 cycles
-> Phase B: Co-Processor Scheduling...
   SMEM budget:       33280 bytes
   Sync barriers:     1
   Est. parallel cy:  215
   Overlap savings:   133 cycles
   Barrier 0: FP32 -> FP16 quantization (16384 bytes)
-> Phase C: Fused PTX Emission...
-> Written to: tests/coprocessor_db_index.coprocessor.ptx
   Dual-accelerator PTX generated successfully!
```

**Est. parallel cy** is the compiler's static estimate of total kernel execution cycles, accounting for RT/Tensor overlap. **Overlap savings** is the number of cycles hidden by scheduling independent instructions during RT traversal.

---

### 11.8 Complete Examples

---

#### Example A: RT-Routed Sparse Attention (`coprocessor_attention.ysu`)

**Use case:** Use hardware BVH to route token keys/values for sparse self-attention in a transformer, then run Tensor Core MMA to project the routed vectors.

```ysu
// Y Dual-Accelerator Co-Processing: RT-Routed Sparse Attention
// RT Core performs KNN routing; Tensor Core projects the attended vectors.
// Compiler inserts: bar.sync, FP32->FP16 quantization, swizzled ldmatrix.

@unsafe
fn main() {
    // Step 1: RT Core — Hardware BVH K-Nearest Neighbor search
    // Query space: 128-dimensional token embeddings, k=8 neighbors
    let nns_res: I32 = rt_nearest_neighbor(128, 8);

    // Step 2: Tensor Core — MMA projection on routed vectors
    // Compiler automatically:
    //   - Places bar.sync after RT traversal completes
    //   - Inserts vectorized cvt.rn.f16x2.f32 quantization loop
    //   - Emits bank-conflict-free ldmatrix.sync.aligned reads
    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
    let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(nns_res);

    acc = mma_sync(frag_A, frag_B, frag_C);
}
```

**Compiler scheduling output:**
```
SMEM budget:    8704 bytes
Sync barriers:  1
Parallel cy:    215   (overlap savings: 133 cycles)
```

**Physical result (RTX 4070 Ti SUPER, 10,000 iterations):**

| Implementation | Latency | Speedup |
| :--- | :---: | :---: |
| Naive CUDA C++ (manual OptiX + WMMA) | 4.2175 µs | 1.0x |
| Y Co-Processor | **2.3818 µs** | **1.77x** |

---

#### Example B: Database Index Fixed-Radius Nearest Neighbor (`coprocessor_db_index.ysu`)

**Use case:** Treat a vector database index as a geometric map. Cluster embeddings become AABB leaf spheres in a BVH. Query rays find nearest neighbors; Tensor Core computes distance projections.

> **Note on index quality:** index construction and recall@k tradeoffs are workload-specific. This demonstrates traversal speedup, not search accuracy.

```ysu
// Y Dual-Accelerator Co-Processing: Database Index as Geometric Map (FRNN)
// 256-dimensional cluster embeddings -> 3D/4D BVH leaf spheres.
// RT Core fires query rays; Tensor Core projects neighbor representations.

@unsafe
fn main() {
    // Step 1: RT Core — Fixed-Radius Nearest Neighbor Search
    // dims=256 embeddings mapped to BVH leaf spheres, searching k=16 neighbors
    let nns_res: I32 = rt_nearest_neighbor(256, 16);

    // Step 2: Tensor Core — MMA distance/projection computation
    // Compiler inserts: bar.sync, vectorized FP32->FP16 quantization (16384 bytes),
    // and bank-conflict-free swizzled fragment loads.
    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
    let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(nns_res);

    acc = mma_sync(frag_A, frag_B, frag_C);
}
```

**CUDA C++ equivalent requires ~65 lines, notably including:**
- A per-thread stack traversal (`stack_depth = 32`) with natural data-dependent ray-bounding sphere checks.
- A manual `if (diff * diff < 4.0f)` distance-check branch.
- A manual FP32→FP16 quantization loop prone to shared memory bank conflicts.
- Manual WMMA fragment loads with non-swizzled (conflict-prone) striding.

**Compiler scheduling output:**
```
SMEM budget:    33280 bytes
Sync barriers:  1
Parallel cy:    215   (overlap savings: 133 cycles)
Barrier 0: FP32 -> FP16 quantization (16384 bytes)
```

Note: static cycle estimates match Example A because both kernels share the same IR node topology (1 RT node, 5 Tensor nodes, 1 barrier). Physical latencies differ significantly due to the larger search space (256D/k=16 vs. 128D/k=8).

**Physical result (RTX 4070 Ti SUPER, 10,000 iterations):**

| Implementation | Latency | Speedup |
| :--- | :---: | :---: |
| Naive CUDA C++ (divergent BVH + manual quant) | 10.6026 µs | 1.0x |
| Y Co-Processor | **5.9137 µs** | **1.79x** |

---

#### Example C: Large Multi-MMA Attention Pipeline (`coprocessor_large.ysu`)

**Use case:** Larger attention pipeline with 7 sequential Tensor Core MMA nodes consuming a single RT Core routing result — demonstrating that the scheduler scales to deeper Tensor pipelines while keeping a single barrier.

```ysu
// Y Dual-Accelerator Co-Processing: Large Multi-MMA Pipeline
// 1 RT Core routing node feeds 7 Tensor Core MMA operations.
// Compiler deduplicates barriers across all 7 consumers.

@unsafe
fn main() {
    // RT Core: KNN routing (128D, k=8)
    let nns_res: I32 = rt_nearest_neighbor(128, 8);

    // Tensor Core pipeline: 7 MMA nodes
    // Single bar.sync and single quantization pass inserted by compiler
    // (deduplicated across all 7 ldmatrix consumers of nns_res)
    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();

    let frag_A0: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B0: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    acc = mma_sync(frag_A0, frag_B0, acc);

    let frag_A1: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B1: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    acc = mma_sync(frag_A1, frag_B1, acc);

    let frag_A2: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B2: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    acc = mma_sync(frag_A2, frag_B2, acc);

    let frag_A3: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B3: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    acc = mma_sync(frag_A3, frag_B3, acc);
}
```

**Compiler scheduling output:**
```
SMEM budget:    8704 bytes
Sync barriers:  1         <- deduplicated from naive 7 down to 1
Parallel cy:    287   (overlap savings: 145 cycles)
```

**Physical result (RTX 4070 Ti SUPER, 10,000 iterations):**

| Implementation | Latency | Speedup |
| :--- | :---: | :---: |
| Naive CUDA C++ (3 barriers, manual quant) | 2.4501 µs | 1.0x |
| Y Co-Processor (1 barrier, deduplicated) | **1.8515 µs** | **1.32x** |

---

### 11.9 Key Optimizations Summary

| Optimization | What Y Does | What CUDA Requires |
| :--- | :--- | :--- |
| **Sync barrier deduplication** | Inserts a single `bar.sync` at the optimal cut point regardless of how many Tensor nodes consume the RT output | Developer inserts one barrier per consumer, or risks races |
| **Vectorized FP32→FP16 quantization** | Emits `cvt.rn.f16x2.f32` packing 2 values per instruction | Manual loop with `__float2half` or inline PTX |
| **Bank-conflict-free SMEM layout** | Computes swizzle offsets for all `ldmatrix` reads automatically | Manual stride/swizzle math per tile size |
| **Unified SMEM budget** | All RT + Tensor scratch aliased to a single `coprocessor_smem` buffer | Separate `__shared__` declarations risk exceeding 48 KB SM limit |
| **RT latency hiding** | Overlaps RT traversal with register init and fragment zeroing | Not done without explicit software pipelining |
| **PTX address safety** | Uses `mov.u64` state-space offsets, not `cvta.shared` | `cvta.shared` can trigger driver-level JIT failures |

---

## 12. Zero-Knowledge Circuit Backend (R1CS)

Y includes a production-grade, standalone ZK circuit compiler backend that directly compiles annotated Y programs into Rank-1 Constraint Systems (R1CS). Unlike domain-specific ZK languages that require learning separate DSL syntax, Y allows writing zero-knowledge circuits using standard Y code with field types (`Field`, `F32`), module-level `@zk_target` annotations, and explicit loop/recursion bounds.

### 12.1 Mathematical Formulation of R1CS Systems in Y

An R1CS system over a finite scalar field $\mathbb{F}_p$ represents computation as a system of quadratic constraints on a witness vector $w = (1, x_1, x_2, \dots, x_m)^T \in \mathbb{F}_p^{m+1}$:

$$\forall k \in \{1, \dots, n\}, \quad \left( \sum_{j=0}^m A_{k,j} w_j \right) \cdot \left( \sum_{j=0}^m B_{k,j} w_j \right) = \sum_{j=0}^m C_{k,j} w_j$$

In vector-matrix notation, this is expressed as:
$$(A \cdot w) \circ (B \cdot w) = C \cdot w$$

where $\circ$ denotes the Hadamard (entrywise) product, and $A, B, C \in \mathbb{F}_p^{n \times (m+1)}$ are sparse coefficient matrices.

#### Scalar Field Support
Y supports configurable prime fields defined via the `@zk_target` module directive:
1. **BN254 (alt_bn128)**: 
   $$p_{\text{bn254}} = 21888242871839275222246405745257275088548364400416034343698204186575808495617$$
   Default field for Ethereum ZK-SNARKs (Groth16, PLONK, snarkjs).
2. **BLS12-381**:
   $$p_{\text{bls12\_381}} = 52435875175126190479447740508185965837690552500527637822603658699938569566224130796352495802057096969914837853159$$
   Standard field for Zcash, Filecoin, and Eth2 BLS signatures.

---

### 12.2 ZK Directives, Attributes, and Annotations

| Directive / Attribute | Target Scope | Description |
| :--- | :--- | :--- |
| `@zk_target(field = "bn254", scheme = "r1cs", opt_level = 1)` | Module | Configures scalar field ($p$), proof scheme, and optimization pipeline. |
| `@unsafe` | Function | Enables mutable variable reassignments and in-place state tracking inside ZK circuit blocks. |
| `@max_iterations(N)` | `while` Loop | Enforces compile-time finite unrolling bound $N$ for dynamic or static `while` loops. |
| `@max_depth(N)` | Function | Enforces compile-time recursion depth bound $N$ for monomorphized recursive function calls. |
| `@bounds(min, max)` | Parameter / Var | Emits active range-check constraints (bit-decomposition) verifying $w_i \in [\text{min}, \text{max}]$. |
| `@invariant(expr)` | Loop / Block | Verifies logic assertions statically or generates equality constraints inside loops. |

---

### 12.3 SSA Linear-Combination Folding & Single-Pass Optimization

Conventional ZK compilers (such as Circom) emit separate wires for every linear addition (e.g. `x + y + z`), generating large systems of linear equations that require slow, superlinear post-processing optimization passes (e.g., iterative Gaussian elimination under `--O2`).

Y-lang eliminates post-processing optimization penalties by performing **single-pass Static Single Assignment (SSA) linear combination folding** directly on the AST during constraint generation:

1. **Linear Accumulation**: Pure additions and scalar multiplications (e.g., `3 * a + 2 * b + 5`) are maintained as unconstrained `LinearCombination` instances in memory ($0$ R1CS constraints).
2. **Multiplication Constraint Emission**: R1CS multiplication constraints $(A \cdot w) \cdot (B \cdot w) = (C \cdot w)$ are emitted **only** when two non-constant linear combinations are multiplied together (`lc1 * lc2`).
3. **Automatic Wire Recycling & Sub-expression Deduplication**: Identical linear terms and intermediate multiplication results are deduplicated in $O(1)$ time using order-independent field hash maps.

**Example Trace (`y = a * b + c * d + e`)**:
* `a * b` $\rightarrow$ Emits Constraint 0: $(1 \cdot a) \cdot (1 \cdot b) = (1 \cdot w_{\text{mul0}})$
* `c * d` $\rightarrow$ Emits Constraint 1: $(1 \cdot c) \cdot (1 \cdot d) = (1 \cdot w_{\text{mul1}})$
* `y = w_{\text{mul0}} + w_{\text{mul1}} + e` $\rightarrow$ Folded into linear combination binding for `y` with **0 extra constraints**!

---

### 12.4 Bounded `while` Loops & SSA Active-Mask State Multiplexing

To support control flow without incurring dynamic unrolling security vulnerabilities, Y requires all `while` loops in ZK target mode to specify an explicit `@max_iterations(N)` decorator:

```ysu
@max_iterations(100)
while cond_expr {
    // loop body
}
```

Y automatically selects one of two optimization paths based on condition nature:

#### 1. Static Index Fast Path (`bounded_while_static`)
When the loop condition (`i < 100000`) evaluates to a compile-time constant, Y bypasses active-mask SSA multiplexing entirely:
* Loops are unrolled directly in compiler memory.
* **Compilation Speed**: Compiles 100,000 iterations in **`0.143s`** (**2.7x faster than Circom**, **141x faster than Noir**).

#### 2. Dynamic Active-Mask SSA Path (`bounded_while_dynamic`)
When loop conditions depend on dynamic witness inputs (`val < witness`), Y emits **active-mask state transition constraints** and **gated SSA $\Phi$-node multiplexers**:
* **Active-Mask State Wire**: $\text{active}_{i+1} = \text{active}_i \cdot \text{cond}_{i+1}$
* **Gated SSA $\Phi$-Node**: $\text{active}_i \cdot (\text{var}_{\text{body}} - \text{var}_{\text{before}}) = w_{\text{mux}} - \text{var}_{\text{before}}$
* **Constraints Emitted**: Exactly 2 R1CS constraints per iteration (1 active mask + 1 SSA multiplexer).
* **Compilation Speed**: Compiles 200,000 active-mask R1CS constraints in **`2.374s`** (**8.3x faster than Noir's SSA pass**).

---

### 12.5 Monomorphized Static Recursion

Y supports compile-time finite recursion for ZK targets via monomorphization:

```ysu
@max_depth(100)
fn recursive_pow(x: Field, n: u32) -> Field {
    if n == 0 {
        return 1;
    }
    return x * recursive_pow(x, n - 1);
}
```

* **Call Stack Depth Verification**: The compiler tracks call stack depth during monomorphization. If recursion exceeds `@max_depth(N)`, compilation aborts with a static error (`error[Z0011]: recursion depth limit exceeded`).
* **Unrolled Call Graph**: Recursive calls are expanded into flat call graphs with zero runtime stack overhead.
* **Performance**: Compiles a 100-depth recursion tree in **`0.005s`** (**1.9x faster than Circom**, **22x faster than Noir**).

---

### 12.6 R1CS Binary & Symbol File Formats

When compiling with `--target=r1cs`, Y emits three output artifacts:

| Artifact | Format | Description |
| :--- | :--- | :--- |
| `<name>.r1cs` | Binary R1CS v1 | Standard binary format containing field header, wire counts, and sparse $A, B, C$ constraint vectors. Compatible with `snarkjs`, `bellman`, `arkworks`, and `rapidsnark`. |
| `<name>.sym` | UTF-8 Text | Symbol table mapping 1-based wire indices to high-level Y program variable names. |
| `<name>.r1cs.txt` | UTF-8 Text | Human-readable linear combination printout for circuit auditing and verification. |

---

### 12.7 Benchmark & Performance Comparison Suite

Every benchmark compiler was evaluated in its fastest official optimization mode on an x86_64 host (AMD Ryzen 9, 64 GB RSS capacity).

#### 1. Structural Parity & Constraint Count Matrix

| Benchmark Circuit | Y-lang Constraints | Circom (`--O2`) Constraints | Noir ACIR Opcodes | Leo Program Size / Inst. |
| :--- | :---: | :---: | :---: | :---: |
| `test_circuit` | **5 R1CS** | 7 R1CS | 8 ACIR Opcodes | 12 Aleo Inst. |
| `dot_product` (100k) | **100,000 R1CS** | 100,000 R1CS | 100,002 ACIR Opcodes | N/A (Bytecode Limit Exceeded) |
| `heavy_circuit` (1M) | **1,000,000 R1CS** | 1,000,000 R1CS | 1,000,002 ACIR Opcodes | N/A (Bytecode Limit Exceeded) |
| `bounded_while_static` (100k Static) | **100,000 R1CS** | 100,000 R1CS | 100,002 ACIR Opcodes | N/A (Execution Timeout) |
| `bounded_while_dynamic` (100k Witness) | **200,000 R1CS** | 200,000 R1CS**†** | 200,002 ACIR Opcodes | N/A (Execution Timeout) |
| `static_rec` (100-depth) | **100 R1CS** | 100 R1CS | 102 ACIR Opcodes | N/A (Recursion Unsupported) |
| `heavy_31m` (31M) | **31,000,000 R1CS** | N/A (Terminated ~2h Timeout) | N/A (OOM / Crash) | N/A (OOM) |

#### 2. Compiler Compilation Speed Comparison (BN254 Field)

| Benchmark Circuit | Y-lang Time (s) | Circom (`--O2`) Time (s) | Noir (Aggressive) Time (s) | Leo (BLS12-377)* Time (s) | Y Speedup vs Circom / Peer |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `test_circuit` | **0.005s** | 0.016s | 0.112s | 0.066s | **3.2x faster vs Circom** |
| `dot_product` (100k) | **0.285s** | 15.280s | 2.261s | Exceeds 512KB Limit (13.7MB Bytecode) | **53.6x faster vs Circom** |
| `heavy_circuit` (1M) | **1.706s** | 253.936s | 13.069s | Exceeds 512KB Limit (31.3MB Bytecode) | **148.8x faster vs Circom** |
| `bounded_while_static` (100k Static) | **0.143s** | 0.388s | 20.243s | Execution Timeout (>600s) | **2.7x faster vs Circom** (Static Fast Path) |
| `bounded_while_dynamic` (100k Witness) | **2.374s** | 0.382s**†** | 19.782s | Execution Timeout (>600s) | **8.3x faster vs Noir** (Active-Mask SSA) |
| `static_rec` (100-depth) | **0.005s** | 0.010s | 0.112s | Syntax Error (Recursion Unsupported) | **1.9x faster vs Circom** |
| `heavy_31m` (31M) | **113.025s** | Terminated (~2h Timeout) | 31M Limit Exceeded (OOM) | 31M Limit Exceeded (OOM) | **>100x vs Circom Timeout** |

*\*Note: Leo benchmarks reflect native BLS12-377 execution since Leo does not support targeting BN254 directly.*  
**†Note on Circom's Dynamic Control Flow Limit**: Circom strictly prohibits witness signals (`signal input`) in control flow conditions (`Error: Non-constant condition in if statement`). Circom's `0.382s` time reflects C++ compile-time `var` macro unrolling (emitting 0 active-mask circuit constraints). For true witness-dependent dynamic loops, the direct peer comparison is **Y-lang vs. Noir**, where Y-lang compiles in **`2.374s`** vs Noir's **`19.782s`** (**8.3x faster**).

#### 3. Scalar Field Comparison: BN254 vs. BLS12-381 (Kernel VmHWM Isolated RAM)

| Benchmark Circuit | Compiler | BN254 Time (s) | BLS12-381 Time (s) | BN254 RAM (MB) | BLS12-381 RAM (MB) | Time Delta Overhead |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `test_circuit` | Y-lang | 0.005s | 0.005s | **3.4 MB** | **3.4 MB** | N/A (<50ms, Noise-Dominated) |
| `test_circuit` | Circom (--O2) | 0.016s | 0.016s | **11.5 MB** | **11.9 MB** | N/A (<50ms, Noise-Dominated) |
| `dot_product` | Y-lang | 0.285s | 0.275s | **152.8 MB** | **153.6 MB** | -3.4% |
| `dot_product` | Circom (--O2) | 15.280s | 15.614s | **1178.1 MB** | **1178.6 MB** | +2.2% |
| `heavy_circuit` | Y-lang | 1.706s | 1.620s | **1038.5 MB** | **1073.1 MB** | -5.0% |
| `heavy_circuit` | Circom (--O2) | 253.936s | 248.705s | **3073.1 MB** | **3073.1 MB** | -2.1% |
| `bounded_while_static` | Y-lang | 0.143s | 0.128s | **7.4 MB** | **3.3 MB** | -10.6% |
| `bounded_while_static` | Circom (--O2) | 0.388s | 0.378s | **10.6 MB** | **14.9 MB** | -2.4% |
| `bounded_while_dynamic` | Y-lang | 2.374s | 2.286s | **233.9 MB** | **234.6 MB** | -3.7% |
| `bounded_while_dynamic` | Circom (--O2) | 0.382s | 0.393s | **9.6 MB** | **9.0 MB** | -3.8% |
| `static_rec` | Y-lang | 0.005s | 0.005s | **3.4 MB** | **3.4 MB** | N/A (<50ms, Noise-Dominated) |
| `static_rec` | Circom (--O2) | 0.010s | 0.010s | **13.1 MB** | **13.0 MB** | N/A (<50ms, Noise-Dominated) |
| `heavy_31m` | Y-lang | 113.025s | 114.341s | **29301.9 MB** | **31365.8 MB** | +1.2% |

---

### 12.8 Complete Example Circuits

#### 1. Dot Product Circuit (`dot_product.ysu`)
```ysu
@zk_target(field = "bn254", scheme = "r1cs", opt_level = 1)
module DotProduct {
    @unsafe
    fn main(a: Field, b: Field) -> Field {
        let mut acc: Field = 0;
        let mut i: Field = 0;

        @max_iterations(100000)
        while i < 100000 {
            acc = acc + (a * b);
            i = i + 1;
        }
        return acc;
    }
}
```

#### 2. Dynamic Witness Bounded Loop (`bounded_while_dynamic.ysu`)
```ysu
@zk_target(field = "bn254", scheme = "r1cs", opt_level = 1)
module BoundedWhileDynamic {
    @unsafe
    fn main(x: Field, cond_flag: Field) -> Field {
        let mut acc: Field = x;

        @max_iterations(100000)
        while cond_flag {
            acc = acc + 1;
        }
        return acc;
    }
}
```

#### 3. Monomorphized Recursion Circuit (`static_rec.ysu`)
```ysu
@zk_target(field = "bn254", scheme = "r1cs", opt_level = 1)
module StaticRec {
    @max_depth(100)
    fn fib(n: Field) -> Field {
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }
        return fib(n - 1) + fib(n - 2);
    }

    @unsafe
    fn main(x: Field) -> Field {
        return fib(100) + x;
    }
}
```


### 12.6 Verifying Generated Circuits

Verification scripts are included:
```bash
python verify_r1cs.py      # dot_product circuit
python verify_heavy.py     # heavy_circuit (1M constraints)
python verify_benchmarks.js  # snarkjs-based verification
```

### 12.7 Benchmark Methodology & Transparency

To ensure scientific rigor, transparency, and reproducibility, the benchmark comparisons are conducted under the following conditions:

* **Hardware Configuration**: All benchmarks were executed on a single-tenant host featuring an AMD Ryzen 9 9950X CPU and an NVIDIA GeForce RTX 4070 Ti SUPER GPU, operating with fixed clock rates to eliminate power-management and boost-frequency scaling variance.
* **Warm-up Iterations**: GPU benchmarks execute 50 un-timed warm-up iterations to load all code into the instruction cache, allocate GPU memories, and stabilize hardware power draws before timing begins.
* **Replication and Run-to-Run Variance**: 
  - GPU co-processor benchmarks report the average execution latency over 10,000 test iterations. Iterative standard deviation was observed to be less than 1.5% across all runs.
  - ZK compiler benchmarks for baseline logic, 100k dot product, bounded loops, and static recursion are executed 3 consecutive times from a clean state (purging build caches prior to invocation). The tables report the sample mean ± standard deviation ($\pm 1.5\%$).
  - Large-scale benchmarks (`linear_heavy`, `heavy_circuit`, `heavy_31m`) represent **single-run executions** given long execution durations; confidence bounds and variance are inherently higher at these scales.
* **Target IR Architecture Model Disclosures**:
  - Noir compiles to ACIR (Abstract Circuit Intermediate Representation) opcodes and Leo compiles to Aleo Bytecode/instructions, whereas Y and Circom directly target R1CS (Rank-1 Constraint Systems). Loop unrolling and compilation latencies across Noir and Leo reflect frontend AST parsing and domain-specific IR lowering overheads rather than identical R1CS constraint generation passes.
* **No Unmeasured Extrapolations**:
  - All reported resource and timing figures reflect real empirical measurements. Unmeasured OOM projections are strictly omitted; scale limits are documented with exact empirical failure modes (e.g., Aleo's mandatory 512KB deployment bytecode limit `Error ECLI0377055`).
* **Circuit Topology Sensitivity & Simplification Best-Case Context**:
  - The **148.8x speedup** on `heavy_circuit` (1M constraints) represents a **best-case scenario** for Y's single-pass SSA linear combination folding and a **worst-case scenario** for Circom's `--O2` pass. Because `heavy_circuit` consists of 1M non-linear multiplications (`temp[i] * y`) with 0 linear constraints, Circom's `--O2` pass scans for simplifications that do not exist, incurring high overhead. Speedups vary dynamically based on circuit topology.
* **Memory Measurement Methodology**: Peak memory usage is measured as peak Resident Set Size (RSS) using the POSIX system call `getrusage(RUSAGE_CHILDREN)` immediately after the child compiler process terminates. This captures the maximum physical RAM allocated to the compilation process (in Megabytes), avoiding measurements of virtual memory layouts or shared page cache.
* **Measurement Boundaries**: GPU kernel timings measure the execution latency on the device using CUDA events (e.g., `cudaEventRecord`), excluding host-to-device memory copy and compilation overheads. ZK compiler benchmarks measure constraint generation time from the initial compiler invocation to R1CS emission, excluding proof generation time.
* **Asymptotic Scaling Curve Analysis**:
  - At 100,000 constraints (`dot_product`), Y-lang compiles in **`0.285s`** (Peak RSS: **`152.8 MB`**) while Circom (native C++ `--c --O2`) compiles in **`15.280s`** (Peak RSS: **`1178.1 MB`**), achieving a **53.6x speedup** and **7.71x memory reduction**.
  - At 1,000,000 constraints (`heavy_circuit`), Y-lang compiles in **`1.706s`** (Peak RSS: **`1038.5 MB`**) while Circom compiles in **`253.936s`** (Peak RSS: **`3073.1 MB`**), achieving a **148.8x speedup** and **2.96x memory reduction**.
  - The speedup growth from **53.6x** to **148.8x** as constraints scale up demonstrates Y's superior asymptotic scaling, validating the elimination of super-linear global simplification passes (such as Circom's `--O2` rounds) in favor of Y's localized single-pass constraint deduplication and flat in-place SSA updates.
* **Simplification Pass and Front-End Overhead Analysis**:
  - **The Role of `--O2` Simplification**: In the 100k constraint `dot_product` benchmark, compiling Circom with default `--O1` output includes 100,000 non-linear constraints, 300,000 linear constraints, and 400,003 wires. Specifying `--O2` triggers Circom's iterative Gaussian elimination pass to solve and substitute these linear relations, successfully reducing the circuit to 100,000 non-linear constraints, 0 linear constraints, and 100,003 wires (matching Y-lang's direct output of 100,001 constraints and 100,004 wires). However, this reduction incurs a compile-time penalty.
  - **Inherent Compiler Speed Advantage**: In the 1M constraint `heavy_circuit` benchmark, every loop constraint is a non-linear multiplication of two variables (`temp[i] * y`), leaving 0 linear constraints to solve. Running Circom under `--O1` yields the same constraint count as `--O2` (1M non-linear constraints, 1M+3 wires) but takes **247.3s**, while `--O2` takes **244.7s**. This proves that Circom's compilation latency is dominated by front-end parsing, template execution, symbol lookup, and file writing rather than just simplification time, showing that Y's 148.8x speedup (1.706s) is a native compiler architecture win.
  - **Superlinear Scaling Limits of Gaussian Elimination**: In the 1M constraint `linear_heavy` benchmark (which contains 5,000,000 linear relations), Circom with `--O2` did not complete within the 2-hour cutoff limit. Per Circom's official documentation, the `--O2` optimizer applies Gaussian elimination repeatedly in "rounds" until no further linear constraints containing private signals can be found. In circuits with large numbers of interconnected linear signals, this iterative substitution solver can scale superlinearly (approaching $O(N^3)$ complexity), leading to CPU/RAM bottlenecks. In contrast, Y-lang's single-pass SSA tracker performs linear folding on the fly during AST compilation, directly outputting the optimized 1,000,001 constraints circuit in **140.05s** (1.66 GB RSS).
  - **Direct Optimization via SSA**: Y-lang's parser and single-pass SSA tracker automatically perform linear-combination folding on the fly. Y directly emits the optimized constraint size without requiring a separate post-processing simplification phase, delivering both fast compilation and minimal proving size.

---

## 13. CUDA-to-Y Migration Guide

This section maps common CUDA C++ patterns directly to their Y equivalents. It is intended for GPU developers who know CUDA and want to understand Y's model quickly.

### 13.1 Memory Declarations

| CUDA C++ | Y Equivalent | Notes |
| :--- | :--- | :--- |
| `__shared__ float buf[4096];` | `let buf = SharedMemory::alloc<SmemLayout<F32, ...>>();` | Y requires an explicit layout type |
| `__global__ float* ptr` | `data: GlobalMemory<F32>` | Global memory is a first-class type |
| `__device__ float val;` | `let val: F32 = ...;` | Regular variable in kernel scope |
| `alignas(128) float x;` | `@align(128) let x: F32 = ...;` | `alignas` sets allocation alignment; `@align(N)` in Y sets LLVM `align N` on load/store instructions (access alignment hint), not allocation alignment. Effect is equivalent for most GPU memory patterns. |
| `volatile float* ptr;` | `@gpu_uncached let ptr: F32 = ...;` | Non-temporal / bypass cache |

### 13.2 Synchronization

| CUDA C++ | Y Equivalent | Notes |
| :--- | :--- | :--- |
| `__syncthreads();` | `barrier::sync();` | Block-level sync |
| `asm volatile("bar.sync 0;");` | Injected automatically by co-processor scheduler | Manual only if using `chisel {}` |
| `__syncwarp();` | `warp::sync();` | Warp-level sync |
| `cp_async_wait_all();` | `pipe.wait(tx);` | Wait for specific transfer token |

### 13.3 Tensor Core (WMMA API → Y Fragments)

| CUDA C++ (WMMA) | Y Equivalent |
| :--- | :--- |
| `wmma::fragment<wmma::matrix_a, 16,16,16, half, col_major>` | `Fragment<MMA_m16n8k16, A, F16>` |
| `wmma::fragment<wmma::matrix_b, 16,16,16, half, row_major>` | `Fragment<MMA_m16n8k16, B, F16>` |
| `wmma::fragment<wmma::accumulator, 16,16,16, float>` | `Fragment<MMA_m16n8k16, D, F32>` |
| `wmma::fill_fragment(frag_c, 0.0f);` | `Fragment::zero()` |
| `wmma::load_matrix_sync(frag_a, ptr, stride);` | `ldmatrix(smem_buf)` — stride and swizzle computed automatically |
| `wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);` | `acc = mma_sync(frag_A, frag_B, frag_C);` |
| `wmma::store_matrix_sync(ptr, frag_c, stride, layout);` | `store(acc, C);` |

### 13.4 Async Memory Transfers (cp.async)

```cpp
// CUDA C++
__pipeline_memcpy_async(smem_ptr, global_ptr, 128);
__pipeline_commit();
__pipeline_wait_prior(0);
```

```ysu
// Y equivalent
let tx: Transfer<Global, Shared, Async<1>, 128> = cp_async(A[k], smem_A);
pipe.wait(tx);
barrier::sync();
```

The Y version enforces at compile time that `pipe.wait(tx)` is called before `smem_A` is read — the CUDA version relies on developer discipline.

### 13.5 Safety Scopes

| CUDA C++ | Y Equivalent | Difference |
| :--- | :--- | :--- |
| No equivalent | `@safe { }` | Enforces initialization, bounds, invariants at compile time |
| No equivalent | `@unsafe { }` | Explicit opt-out of safety checks, required for raw pointer math |
| `assert(cond)` (runtime) | `@bounds(min <= i < max)` | Static for constant indices, runtime assertion for dynamic |
| No equivalent | `@invariant(expr)` | Loop invariant verified at every iteration by type checker |

### 13.6 Cache Policies

| CUDA C++ | Y Equivalent |
| :--- | :--- |
| `ld.global.ca` (L1/L2 cache) | Default load (no decorator) |
| `ld.global.cg` (L2 only) | `@cache_policy(L2_EVICT_FIRST)` |
| `ld.global.cs` (streaming, evict-first) | `@cache_policy(L2_STREAM)` |
| `ld.global.lu` (last-use, invalidate) | `@cache_policy(L2_EVICT_LAST)` |
| `ld.global.nc` (non-coherent / read-only) | `@cache_policy(L2_PERSIST)` |

### 13.7 Inline PTX

```cpp
// CUDA C++
asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(hi), "f"(lo));
```

```ysu
// Y: injected automatically by quantization pass when a Fragment<..., F16>
// consumes an FP32 RT Core result. Manual equivalent:
chisel {
    "cvt.rn.f16x2.f32 %packed, %hi, %lo;";
}
```

### 13.8 Full Side-by-Side: Fused RT+Tensor Kernel

**CUDA C++ (manual, ~65 lines):**
```cpp
extern "C" __global__ void cuda_rt_tensor_kernel(
    const float* query, float* out
) {
    __shared__ float rt_scratch[4096];
    __shared__ half  quant_scratch[4096];
    int tid = threadIdx.x;

    // Manual BVH traversal
    int stack_depth = 32;
    float dist = 0.0f;
    for (int i = 0; i < stack_depth; ++i) {
        float diff = query[tid % 16] - 0.5f;
        if (diff * diff < 4.0f) dist += __fsqrt_rn(diff * diff + 0.1f);
        else dist += 0.01f;
    }
    rt_scratch[tid] = dist;

    asm volatile("bar.sync 0;");

    // Manual FP32->FP16 quantization
    if (tid < 2048) {
        uint32_t packed;
        asm("cvt.rn.f16x2.f32 %0, %1, %2;"
            : "=r"(packed)
            : "f"(rt_scratch[2*tid+1]), "f"(rt_scratch[2*tid]));
        ((uint32_t*)quant_scratch)[tid] = packed;
    }
    asm volatile("bar.sync 0;");

    // WMMA Tensor Core
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::col_major> fa;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb;
    wmma::fragment<wmma::accumulator,16,16,16,float> fc;
    wmma::fill_fragment(fc, 0.0f);
    wmma::load_matrix_sync(fa, &quant_scratch[0], 16);
    wmma::load_matrix_sync(fb, &quant_scratch[256], 16);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(out, fc, 16, wmma::mem_row_major);
}
```

**Y (12 lines — compiler generates everything above):**
```ysu
@unsafe
fn main() {
    let nns_res: I32 = rt_nearest_neighbor(128, 8);

    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
    let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(nns_res);
    acc = mma_sync(frag_A, frag_B, frag_C);
}
```

**Physical result (RTX 4070 Ti SUPER, 10,000 iterations): 4.22 µs → 2.38 µs (1.77x speedup)**

---

## 14. Performance Tuning Guide

This section explains how to use Y's hardware-aware features to get the best performance out of specific workloads.

### 14.1 Reading the Hardware Profile

After the Sentinel probe runs, `.ysu_hw_profile` contains cycle-accurate measurements specific to your machine. The compiler uses these directly in its cost model. Key values and what they mean for your code:

| Profile Key | What it measures | How it affects codegen |
| :--- | :--- | :--- |
| `SMEM_LATENCY` | Shared memory access latency in cycles | Determines when `barrier::sync()` insertion is worth the cost vs. re-computing |
| `TENSOR_F16_LATENCY` | Tensor Core MMA latency (F16) | Used to schedule overlap between RT traversal and register initialization |
| `BRANCH_DIVERGENCE_PENALTY` | Extra cycles per divergent warp branch | If high, use `@divergence(uniform)` to assert non-divergent paths |
| `WARP_SHUFFLE_LATENCY` | Cycles for `__shfl_sync` | If lower than `SMEM_LATENCY`, compiler prefers warp-shuffle reductions |
| `L2_CACHE_LINE` | L2 cache line size in bytes | Used to compute `@align(N)` for struct fields in SPSC buffers |
| `FMA_LATENCY` | FP32 FMA latency | Determines IMAD.WIDE vs IMAD selection for integer multiply-accumulate |

### 14.2 Choosing a Cache Policy

```ysu
// Use L2_PERSIST for data accessed repeatedly across loop iterations
// (e.g. weight matrices in attention, BVH node data in deep traversals)
@cache_policy(L2_PERSIST, reuse_count=8)
let weights: F16 = load(W);

// Use L2_STREAM for data written once and never re-read
// (e.g. output tiles, streaming reductions)
@cache_policy(L2_STREAM)
output[i] = result;

// Use L2_EVICT_FIRST for inputs that should not pollute L2
// (e.g. large activation tensors in single-pass inference)
@cache_policy(L2_EVICT_FIRST)
let act: F16 = load(A);
```

**Rule of thumb:**
- Weights / BVH nodes repeatedly accessed → `L2_PERSIST`
- Large one-shot reads → `L2_EVICT_FIRST`
- Write-only outputs → `L2_STREAM`

### 14.3 Eliminating Shared Memory Bank Conflicts

GPU shared memory is divided into 32 banks (4 bytes each). If 32 threads in a warp all access addresses that map to the same bank, the accesses are serialized into 32 sequential steps instead of 1.

**How to diagnose:** The compiler will warn `B0001: shared memory bank conflict detected` with the stride that caused it.

**How to fix — use layout swizzling:**
```ysu
// Bad: stride 32 floats = 128 bytes -> all threads hit bank 0
type BadLayout  = SmemLayout<F32, rows=32, cols=32, swizzle=0>;

// Good: swizzle=330 XORs the row index into the column address
type GoodLayout = SmemLayout<F32, rows=32, cols=32, swizzle=330>;
```

The swizzle value `330` is a Y-specific compact integer encoding representing the swizzling parameters `xor_bits=3`, `base_shift=3`, `offset=0` (corresponding to a layout designed to match typical CUDA/CUTLASS swizzle patterns for 32-column tiles). The Y compiler validates the conflict-free property statically.

### 14.4 When to Use `--emit-coprocessor` vs Raw PTX

| Scenario | Use |
| :--- | :--- |
| You have RT Core traversal feeding Tensor Core MMA | `--emit-coprocessor` |
| Pure Tensor Core kernel (no BVH/ray queries) | `--llvm` or `--ptx` |
| Pure compute kernel (reductions, FFTs, sorting) | `--llvm` |
| ZK circuit generation | `--emit-r1cs` |
| CPU-side lock-free data structures | `--llvm` (uses AVX-512 backend) |

### 14.5 Overlapping RT Core Latency

RT Core traversal is the highest-latency operation in a co-processor kernel (~200 cycles on Ada Lovelace). The compiler automatically overlaps it with:
- `Fragment::zero()` accumulator initialization (~4 cycles)
- Register-resident scalar setup
- Independent global-memory parameter loads

If you want to maximize this overlap manually inside `chisel {}` blocks, place any register-only computation (no shared memory reads) between the `rt_nearest_neighbor` call and the first `ldmatrix`:

```ysu
@unsafe
fn main() {
    let nns_res: I32 = rt_nearest_neighbor(256, 16); // <-- RT starts here

    // These are pure register ops -- run during RT traversal:
    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
    let scale: F32 = 1.0 / 256.0;  // precompute normalization

    // bar.sync (auto-inserted) -- RT must complete before here
    let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(nns_res);
    acc = mma_sync(frag_A, frag_B, frag_C);
}
```

The compiler's scheduling report will show `Overlap savings: 133 cycles` when this pattern is detected.

### 14.6 Loop Invariant Hoisting (`@invariant`)

Inside `@safe` blocks, `@invariant` annotations do more than verify correctness — they inform the optimizer that a value is stable across iterations, enabling hoisting:

```ysu
@safe
fn process(data: [F32; 1024], scale: F32) -> F32 {
    let sum: F32 = 0.0;

    @invariant(sum >= 0.0)
    @invariant(scale > 0.0)   // <- tells compiler scale never changes in loop
    for i in 0..1024 {
        sum += data[i] * scale;
    }
    return sum;
}
```

Without `@invariant(scale > 0.0)`, the compiler conservatively reloads `scale` each iteration. With it, the compiler can hoist the load to a register before the loop begins.

### 14.7 Register Pressure in Tensor Core Kernels

Each `Fragment<MMA_m16n8k16, ...>` occupies a fixed number of 32-bit registers:

| Fragment Role | Registers Used |
| :---: | :---: |
| A (F16) | 4 |
| B (F16) | 2 |
| C / D (F32) | 4 |

The RTX 4070 Ti SUPER has **255 registers per thread**. If you chain more than ~20 MMA operations without storing intermediate accumulators, you will exhaust the register file and spill to local memory (measured at ~125 cycles/access vs ~4 cycles for registers).

**Rule:** Keep the number of live `Fragment` variables below 30 at any given point in the kernel. Interleave `store(acc, C)` calls to free registers between MMA pipeline stages.

### 14.8 ZeroDrift Fixed-Point Accumulation

For long-running accumulation loops (e.g. computing dot products over 1M+ elements), standard F32 accumulation accumulates floating-point rounding error that compounds with iteration count. Y's `@ZeroDrift` annotation enforces drift-free accumulation using Q32.32 fixed-point arithmetic:

```ysu
fn safe_sum(data: [Q32.32; 1000000]) -> Q32.32 {
    @ZeroDrift
    let acc: Q32.32 = 0.0;

    for i in 0..1000000 {
        acc += data[i];
    }
    return acc;
}
```

The compiler verifies that `Q32.32` never overflows the 64-bit fixed-point range given the bounds of `data`. If it might, a compile-time error is emitted:
```
error[D0001]: ZeroDrift accumulator may overflow Q32.32 range
hint: reduce iteration count or use Q16.48 for higher fractional precision
```

---

## 15. Fragment & MMA Type Reference

This section documents all valid `Fragment<Shape, Role, Precision>` combinations, their register footprints, and the PTX MMA instructions they map to.

### 15.1 Fragment Type Syntax

```ysu
let frag: Fragment<Shape, Role, Precision> = ...;
```

| Parameter | Options | Notes |
| :--- | :--- | :--- |
| `Shape` | `MMA_m16n8k16`, `MMA_m16n8k8`, `MMA_m16n16k16` | Tile dimensions: M×N output, K reduction depth |
| `Role` | `A`, `B`, `C`, `D` | Matrix role in the MMA operation |
| `Precision` | `F16`, `BF16`, `TF32`, `F32` | Numeric type stored in the fragment |

### 15.2 Valid Fragment Combinations

| Shape | Role A | Role B | Role C / D | PTX Instruction |
| :--- | :---: | :---: | :---: | :--- |
| `MMA_m16n8k16` | `F16` | `F16` | `F32` | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |
| `MMA_m16n8k16` | `F16` | `F16` | `F16` | `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` |
| `MMA_m16n8k8` | `TF32` | `TF32` | `F32` | `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` |
| `MMA_m16n8k16` | `BF16` | `BF16` | `F32` | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` |
| `MMA_m16n16k16` | `F16` | `F16` | `F32` | `mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32` (Ampere+) |

> **Hardware compatibility:** `MMA_m16n8k16` with F16 is supported on all Volta+ GPUs (SM 7.0+). `BF16` requires Ampere+ (SM 8.0+). `TF32` requires Ampere+ (SM 8.0+). `MMA_m16n16k16` requires Ampere+ (SM 8.0+).

### 15.3 Register Usage Per Fragment

Each `Fragment` occupies a fixed number of 32-bit registers per thread in a warp:

| Shape | Role | Precision | Registers per Thread |
| :--- | :---: | :---: | :---: |
| `MMA_m16n8k16` | A | F16 | 4 |
| `MMA_m16n8k16` | B | F16 | 2 |
| `MMA_m16n8k16` | C / D | F32 | 4 |
| `MMA_m16n8k16` | C / D | F16 | 2 |
| `MMA_m16n8k8` | A | TF32 | 4 |
| `MMA_m16n8k8` | B | TF32 | 4 |
| `MMA_m16n8k8` | C / D | F32 | 4 |
| `MMA_m16n16k16` | A | F16 | 8 |
| `MMA_m16n16k16` | B | F16 | 8 |
| `MMA_m16n16k16` | C / D | F32 | 8 |

### 15.4 `ldmatrix` Variants

The `ldmatrix(src)` intrinsic selects the correct PTX `ldmatrix` variant based on the fragment role:

| Fragment Role | PTX Emitted |
| :---: | :--- |
| A (4 registers) | `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [ptr];` |
| B (2 registers) | `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {r0,r1}, [ptr];` |
| C / D (F32, 4 regs) | Loaded via standard `ld.shared.f32` sequence — no `ldmatrix` |

### 15.5 Accumulator Initialization & Store

```ysu
// Initialize to zero (emits mov.f32 %rN, 0f00000000 for each register)
let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();

// Store result back to global memory
store(acc, C);
// Emits: wmma-equivalent store or direct st.global.f32 per accumulator register
```

### 15.6 Full MMA Example (F16 → F32 accumulation)

```ysu
kernel matmul_mma(
    A: GlobalMemory<F16>,
    B: GlobalMemory<F16>,
    C: GlobalMemory<F32>
) {
    type TileA = SmemLayout<F16, rows=16, cols=16, swizzle=330>;
    type TileB = SmemLayout<F16, rows=16, cols=8,  swizzle=330>;

    let smem_A = SharedMemory::alloc<TileA>();
    let smem_B = SharedMemory::alloc<TileB>();

    let pipe: Pipeline<stages=2, layout=TileA> = Pipeline::init();

    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();

    for k in 0..1024 step 16 {
        let tx_A = cp_async(A[k], smem_A);
        let tx_B = cp_async(B[k], smem_B);
        pipe.wait(tx_A);
        pipe.wait(tx_B);
        barrier::sync();

        let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(smem_A);
        let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(smem_B);
        let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(smem_A);

        acc = mma_sync(frag_A, frag_B, frag_C);
    }

    store(acc, C);
}
```

---

## 16. `chisel {}` Block Reference

`chisel {}` blocks allow direct inline PTX assembly injection anywhere in a Y program. They are the escape hatch for operations the compiler does not yet emit natively.

### 16.1 Syntax

```ysu
chisel {
    "ptx_instruction operands;";
    "another_instruction;";
}
```

Each line is a string literal containing a single PTX statement, terminated with a semicolon inside the string. The block is emitted verbatim into the current PTX function body at the point of the `chisel` call.

### 16.2 Variable Scoping Inside `chisel`

Y variables declared before the `chisel` block are accessible using their PTX register names. The naming convention follows the Y compiler's register allocator:

| Y Declaration | PTX Register Name |
| :--- | :--- |
| `let x: F32 = ...;` | `%x` (F32 scalar) |
| `let v: I32 = ...;` | `%v` (I32 scalar) |
| `let ptr: ptr = ...;` | `%ptr` (u64 address register) |
| Struct fields | Not directly accessible — load to a local variable first |

```ysu
let val: F32 = 1.0;
let result: F32 = 0.0;

chisel {
    // %val and %result refer to the Y variables above
    "mul.f32 %result, %val, %val;";   // result = val * val
}
```

### 16.3 Safety Implications

- `chisel {}` blocks **bypass all `@safe` guarantees**. Uninitialized reads, out-of-bounds pointer arithmetic, and data races are possible.
- `chisel {}` is implicitly `@unsafe` — it can be used inside `@safe` blocks but the compiler does not verify the inline PTX.
- Bank conflicts and register pressure from `chisel` instructions are **not tracked** by the compiler's static analyzers.

### 16.4 Common Use Cases

**Vectorized FP32 → FP16 packing** (auto-emitted by `--emit-coprocessor`, but available manually):
```ysu
let hi: F32 = src[2 * i + 1];
let lo: F32 = src[2 * i];
let packed: I32 = 0;

chisel {
    "cvt.rn.f16x2.f32 %packed, %hi, %lo;";
}
```

**Warp shuffle reduction:**
```ysu
let val: F32 = thread_value;

chisel {
    "shfl.sync.down.b32 %val, %val, 16, 31, 0xffffffff;";
    "shfl.sync.down.b32 %val, %val, 8,  31, 0xffffffff;";
    "shfl.sync.down.b32 %val, %val, 4,  31, 0xffffffff;";
    "shfl.sync.down.b32 %val, %val, 2,  31, 0xffffffff;";
    "shfl.sync.down.b32 %val, %val, 1,  31, 0xffffffff;";
}
// val in lane 0 now holds the warp-wide sum
```

**Explicit bar.sync (rarely needed — co-processor scheduler inserts automatically):**
```ysu
chisel {
    "bar.sync 0;";
}
```

**GPU clock read for microbenchmarking:**
```ysu
let clock: I64 = 0;
chisel {
    "mov.u64 %clock, %globaltimer;";
}
```

### 16.5 Restrictions

- PTX must be valid for the target `sm_XX` architecture. Invalid PTX will cause `CUDA_ERROR_INVALID_PTX` at JIT load time.
- Do not declare new `.reg` or `.shared` variables inside `chisel` — all registers must come from Y declarations. New PTX declarations will conflict with the compiler's symbol table.
- `chisel {}` cannot span multiple Y scopes (no jumping into or out of `if`/`for`/`while` blocks).

---

## 17. Frequently Asked Questions

**Do I need CUDA drivers and a GPU to use Y?**
Not for all backends. The LLVM (`--llvm`), C (`--c`), x86-64 (`--cpu`), and ZK (`--emit-r1cs`) backends work entirely on CPU with no GPU required. The PTX (`--ptx`) and co-processor (`--emit-coprocessor`) backends require an NVIDIA GPU and CUDA toolkit installed. The hardware Sentinel probe also requires CUDA for the GPU portion, but will skip it gracefully and probe CPU-only if no GPU is detected.

---

**Why does the ZK backend require `@unsafe`?**
The current ZK emitter implements SSA-style constraint generation. Mutable variable reassignment (e.g. `result = result + x` inside a loop) requires the type checker to track multiple versions of the same wire. This is currently gated behind `@unsafe` while the SSA-aware ZK emitter is being finalized. In a future release, `@safe` ZK circuits will be possible for pure functional programs.

---

**What is the difference between `@safe` and `@unsafe`?**

| | `@safe` | `@unsafe` |
| :--- | :--- | :--- |
| Raw pointer access | No Forbidden | Yes Allowed |
| Uninitialized variable reads | No Compile error | Yes Allowed |
| `@invariant` required on loops | Yes Yes | No No |
| `@bounds` enforced statically | Yes Yes | No Not enforced |
| `chisel {}` PTX injection | Yes Allowed (but bypasses checks) | Yes Allowed |

---

**Can I use Y on macOS or Windows?**
Currently Linux only. The native ELF emitter (`native_emitter.rs`) targets Linux ELF64. The LLVM backend can in principle target other platforms via `clang`, but this has not been tested. macOS/Windows support is not on the current roadmap.

---

**Is Y production-ready?**
Y is a research-grade single-developer compiler under active development. The bootstrap compiler (`src/`, Rust) is stable for its documented feature set. The self-hosted compiler (`self_hosted/`, written in Y) is in progress and not the default build path. Do not use Y for production systems without thorough testing. Benchmarks are real and measured, but the toolchain has not been audited for security or hardened for adversarial inputs.

---

**Why is `Fragment::zero()` overlapped with RT Core traversal?**
`Fragment::zero()` expands to a sequence of `mov.f32 %rN, 0f00000000` register moves — pure register operations with no memory accesses. The co-processor scheduler identifies these as independent of the RT Core traversal data path and schedules them to execute during the RT Core's traversal latency (~200 cycles), hiding that cost at zero additional clock budget.

---

**Can I mix `@safe` and `@unsafe` in the same file?**
Yes. `@safe` and `@unsafe` are block-level annotations and can be nested. An `@unsafe` block inside an `@safe` function opts out of the safety checks for that specific block only. `@safe` inside an `@unsafe` function re-enables checks for that inner scope.

---

**What happens if my co-processor SMEM budget exceeds 48 KB?**
The compiler emits a compile-time error before any PTX is generated:
```
error: coprocessor_smem budget (N bytes) exceeds SM limit (49152 bytes)
hint: reduce k or dims, or split into multiple barrier stages
```
The limit is read from the hardware profile and varies by SM generation (48 KB on Turing/Ampere/Ada Lovelace by default, configurable up to 96 KB on Ampere+ with runtime `cudaFuncSetAttribute`).

---

## 18. Known Limitations

| Area | Limitation |
| :--- | :--- |
| **Self-hosted compiler** | `self_hosted/` (written in Y) is in progress. The Rust bootstrap compiler (`src/`) is the only stable build path today. |
| **ZK backend mutability** | Mutable variable reassignment in ZK circuits requires `@unsafe`. Immutable/functional-style circuits work in `@safe`. |
| **`chisel {}` analysis** | The compiler does not track register pressure, bank conflicts, or data races introduced by inline PTX in `chisel` blocks. |
| **Windows / macOS** | Not supported. Native emitter targets Linux ELF64 only. |
| **BF16 / TF32 fragments** | Supported in PTX emission, but the co-processor scheduler does not yet model BF16/TF32 quantization passes — only F16. Manual `chisel` PTX required for BF16 quantization. |
| **`rt_nearest_neighbor` dims** | High-dimensional embeddings (>512D) require careful SMEM budget management. The compiler enforces this statically, but very large `k` values at high dimensions may require splitting into multiple kernel launches. |
| **ZK field** | Configurable via `@zk_target(field = "bn254" | "bls12_381" | "pallas" | "vesta", scheme = "r1cs")`. Native modular arithmetic and Poseidon MDS matrix generation implemented for BN254, BLS12-381, Pallas, and Vesta fields. |

---

## 19. `ypm` — Y Package Manager

`ypm` is Y's built-in package manager for managing Y library dependencies and distributing reusable `.ysu` modules.

### 19.1 Basic Commands

```bash
# Initialize a new Y package in the current directory
./target/release/Y ypm init

# Add a dependency
./target/release/Y ypm add <package-name>

# Install all dependencies listed in Y.toml
./target/release/Y ypm install

# Build the current package
./target/release/Y ypm build

# Run the package entry point
./target/release/Y ypm run
```

### 19.2 Package Manifest (`Y.toml`)

Each Y package is described by a `Y.toml` manifest:

```toml
[package]
name    = "my_kernel"
version = "0.1.0"
author  = "YSU"
entry   = "src/main.ysu"

[dependencies]
# Local path dependency
y_dsp = { path = "../y_dsp" }

# Future: registry-based dependency (planned)
# y_linalg = "0.2.1"
```

### 19.3 Package Structure

A standard `ypm` package layout:

```
my_kernel/
  Y.toml          Package manifest
  src/
    main.ysu      Entry point (fn main)
    kernels.ysu   GPU kernel definitions
    lib.ysu       Shared utilities
  tests/
    test_kernels.ysu
  algorithms/     Reference implementations (C or Y)
```

### 19.4 Importing Modules

```ysu
// Import from a local module file
import kernels::matmul_mma;
import lib::ring_buffer;

fn main() -> I32 {
    // Use imported symbols
    ring_buffer::try_enqueue(&mut rb, 42);
    return 0;
}
```

### 19.5 Current Status

`ypm` is implemented in `src/ypm.rs` and handles local path-based dependencies. Registry-based package distribution (a central package index) is planned but not yet implemented. The current use case is organizing multi-file Y projects and sharing `.ysu` utility modules between kernels in the same workspace.

---

## 20. Numeric Types Reference

Y supports the following primitive numeric types. GPU types are only valid in kernel/PTX contexts; CPU types are valid everywhere.

### 20.1 Integer Types

| Type | Bits | Signed | Range | Valid Context |
| :---: | :---: | :---: | :--- | :---: |
| `I8` | 8 | Yes | −128 to 127 | CPU + GPU |
| `I16` | 16 | Yes | −32,768 to 32,767 | CPU + GPU |
| `I32` | 32 | Yes | −2,147,483,648 to 2,147,483,647 | CPU + GPU |
| `I64` | 64 | Yes | −9.2×10¹⁸ to 9.2×10¹⁸ | CPU + GPU |
| `U8` | 8 | No | 0 to 255 | CPU + GPU |
| `U16` | 16 | No | 0 to 65,535 | CPU + GPU |
| `U32` | 32 | No | 0 to 4,294,967,295 | CPU + GPU |
| `U64` | 64 | No | 0 to 1.8×10¹⁹ | CPU + GPU |

### 20.2 Floating-Point Types

| Type | Bits | Exponent | Mantissa | Hardware | Notes |
| :---: | :---: | :---: | :---: | :--- | :--- |
| `F16` | 16 | 5 | 10 | Volta+ (SM 7.0+) | Half precision. Tensor Core input type. |
| `BF16` | 16 | 8 | 7 | Ampere+ (SM 8.0+) | Brain float. Same range as F32, less precision. |
| `TF32` | 19 | 8 | 10 | Ampere+ (SM 8.0+) | TensorFloat-32. Tensor Core accumulator intermediate. |
| `F32` | 32 | 8 | 23 | All | Single precision. Default GPU accumulator type. |
| `F64` | 64 | 11 | 52 | All | Double precision. High latency on consumer GPUs. |

### 20.3 Fixed-Point Types

| Type | Total Bits | Integer Bits | Fractional Bits | Valid Context |
| :---: | :---: | :---: | :---: | :---: |
| `Q32.32` | 64 | 32 | 32 | CPU only |
| `Q16.48` | 64 | 16 | 48 | CPU only |

Fixed-point types are used with `@ZeroDrift` for verified drift-free accumulation. They are not supported in GPU kernels.

### 20.4 Type Casting

Explicit casts use the `as` keyword:

```ysu
let x: I32 = 42;
let y: F32 = x as F32;    // I32 -> F32 widening
let z: I16 = x as I16;    // I32 -> I16 narrowing (may truncate)
let h: F16 = y as F16;    // F32 -> F16 (precision loss, no error)
```

Implicit coercion **does not happen** in Y. Mixing types in expressions is a compile error:

```
error[E0308]: mismatched types — use explicit `as` cast
```

### 20.5 GPU Type Usage Rules

| Context | Allowed Types |
| :--- | :--- |
| Fragment A / B (Tensor Core input) | `F16`, `BF16`, `TF32` |
| Fragment C / D (Tensor Core accumulator) | `F32`, `F16` |
| RT Core outputs (`rt_nearest_neighbor`) | `I32` (neighbor indices) + implicit `F32` distances in SMEM |
| `@ZeroDrift` accumulator | `Q32.32`, `Q16.48` |
| General kernel variables | `F32`, `I32`, `U32`, `F64`, `I64`, `U64` |
| `@atomic` fields | `I32`, `U32`, `U64`, `bool` |

---

## 21. `SmemLayout` & `Pipeline` API Reference

### 21.1 `SmemLayout<T, rows, cols, swizzle>`

`SmemLayout` defines the physical layout of a tile in GPU shared memory, including optional bank-conflict-eliminating swizzle.

**Syntax:**
```ysu
type MyTile = SmemLayout<ElementType, rows=R, cols=C, swizzle=S>;
let buf = SharedMemory::alloc<MyTile>();
```

**Parameters:**

| Parameter | Type | Description |
| :--- | :---: | :--- |
| `ElementType` | Type | Scalar element type stored per cell (`F16`, `F32`, etc.) |
| `rows` | const I32 | Number of rows in the tile |
| `cols` | const I32 | Number of columns in the tile |
| `swizzle` | const I32 | XOR swizzle mask applied to column index to eliminate bank conflicts. `0` = no swizzle. |

**How swizzle works:**

GPU shared memory has 32 banks of 4 bytes each. A read stride equal to the bank count causes all 32 threads in a warp to hit the same bank (32-way serialization). The swizzle parameter XORs row bits into the column address:

```
physical_col = logical_col XOR ((row >> shift) & swizzle_mask)
```

For a 32-column F32 tile (128 bytes/row = exactly 32 banks), `swizzle=330` (binary `101001010`) eliminates conflicts. The Y compiler validates the conflict-free property statically.

**Common swizzle values:**

| Tile width | Element | Swizzle | Conflict-free? |
| :---: | :---: | :---: | :---: |
| 8 cols | F32 | `0` |  (< 1 bank row) |
| 16 cols | F32 | `8` |  |
| 32 cols | F32 | `330` |  |
| 16 cols | F16 | `0` |  |
| 32 cols | F16 | `8` |  |
| 64 cols | F16 | `330` |  |

**Methods:**

```ysu
// Allocate in shared memory (static allocation, size known at compile time)
let smem_A = SharedMemory::alloc<TileA>();

// Index directly
smem_A[row * cols + col] = val;
```

---

### 21.2 `Pipeline<stages, layout>`

`Pipeline` manages multi-stage asynchronous memory transfer pipelines, overlapping global→shared memory loads with compute.

**Syntax:**
```ysu
let pipe: Pipeline<stages=N, layout=MyLayout> = Pipeline::init();
```

**Parameters:**

| Parameter | Type | Description |
| :--- | :---: | :--- |
| `stages` | const I32 | Number of pipeline stages (typically 2 for double-buffering, 3+ for deeper overlap) |
| `layout` | SmemLayout | The shared memory layout type used by all transfers in this pipeline |

**Methods:**

| Method | Description |
| :--- | :--- |
| `Pipeline::init()` | Initializes the pipeline and allocates internal stage tracking state |
| `pipe.wait(tx)` | Waits for a specific `Transfer` token to complete. Statically consumes the linear obligation. |

**Transfer type:**
```ysu
let tx: Transfer<Src, Dst, Async<stage_idx>, bytes> = cp_async(src_ptr, dst_buf);
```

| Field | Options | Description |
| :--- | :--- | :--- |
| `Src` | `Global` | Source memory space |
| `Dst` | `Shared` | Destination memory space |
| `Async<N>` | Stage index 0..stages-1 | Which pipeline stage this transfer belongs to |
| `bytes` | const I32 | Number of bytes to copy (must be 4, 8, or 16 for hardware cp.async) |

**Full double-buffered pipeline example:**
```ysu
type TileA = SmemLayout<F16, rows=16, cols=16, swizzle=330>;

kernel double_buffered(A: GlobalMemory<F16>, out: GlobalMemory<F32>, N: I32) {
    let smem_A0 = SharedMemory::alloc<TileA>();  // Stage 0 buffer
    let smem_A1 = SharedMemory::alloc<TileA>();  // Stage 1 buffer

    let pipe: Pipeline<stages=2, layout=TileA> = Pipeline::init();

    // Prefetch first block into stage 0
    let tx0 = cp_async(A[0], smem_A0);
    pipe.wait(tx0);
    barrier::sync();

    for k in 1..N {
        // Prefetch next block while processing current
        let tx_next = if k % 2 == 1 {
            cp_async(A[k * 256], smem_A1)
        } else {
            cp_async(A[k * 256], smem_A0)
        };

        // Process current block
        let current_buf = if k % 2 == 1 { smem_A0 } else { smem_A1 };
        // ... compute on current_buf ...

        pipe.wait(tx_next);
        barrier::sync();
    }
}
```

**`pipe.wait(tx)` vs `barrier::sync()`:**

| | `pipe.wait(tx)` | `barrier::sync()` |
| :--- | :--- | :--- |
| Scope | Waits for one specific transfer | Synchronizes all threads in the block |
| PTX emitted | `cp.async.wait_group N` | `bar.sync 0` |
| Required before | Reading the destination buffer of `tx` | Any shared memory access after a write |
| Linear obligation | Consumes the `Transfer` token (required by type checker) | No token involved |

---

## 22. Operator Precedence

Operators are listed from **highest** (evaluated first) to **lowest** (evaluated last). Operators at the same level associate left-to-right unless noted.

| Precedence | Operator(s) | Description | Associativity |
| :---: | :--- | :--- | :---: |
| 1 (highest) | `()` `[]` `::` `.` | Grouping, indexing, path, field access | Left |
| 2 | `-` `!` `~` `*` `&` | Unary negation, logical NOT, bitwise NOT, deref, address-of | Right |
| 3 | `as` | Type cast | Left |
| 4 | `*` `/` `%` | Multiply, divide, modulo | Left |
| 5 | `+` `-` | Add, subtract | Left |
| 6 | `<<` `>>` | Bitwise left/right shift | Left |
| 7 | `&` | Bitwise AND | Left |
| 8 | `^` | Bitwise XOR | Left |
| 9 | `\|` | Bitwise OR | Left |
| 10 | `==` `!=` `<` `>` `<=` `>=` | Comparison | Left |
| 11 | `&&` | Logical AND | Left |
| 12 | `\|\|` | Logical OR | Left |
| 13 (lowest) | `=` `+=` `-=` `*=` `/=` | Assignment, compound assignment | Right |

**Examples:**

```ysu
// Precedence 4 before 5: multiplication binds tighter than addition
let a: I32 = 2 + 3 * 4;      // = 2 + 12 = 14  (not 20)

// Precedence 6 before 7: shift binds tighter than bitwise AND
let b: I32 = 1 & 3 << 2;     // = 1 & 12 = 0   (not 4)

// Use parentheses to override:
let c: I32 = (2 + 3) * 4;    // = 20

// as (precedence 3) binds tighter than arithmetic:
let d: F32 = 1 + x as F32;   // = 1 + (x as F32), not (1 + x) as F32
```

> **Note:** The `*` symbol appears at precedence levels 2 (unary dereference) and 4 (binary multiply). The parser distinguishes them by context — unary `*` requires a prefix position.

---

## 23. Error Code Reference

All compiler error codes, their meanings, and the section where they are demonstrated.

### Compile-Time Errors

| Code | Category | Meaning | See |
| :--- | :--- | :--- | :---: |
| `E0308` | Type mismatch | Expression operands have incompatible types; explicit `as` cast required | §6.1 |
| `L0001` | Linear obligation | A `Transfer` token was not consumed by `pipe.wait()` before its buffer was read (data race) | §6.2 |
| `B0001` | Bank conflict | A `SmemLayout` or `ldmatrix` access stride causes a warp-wide shared memory bank conflict | §6.3 |
| `S0002` | Safety violation | A variable was declared but never assigned before being read inside a `@safe` block | §6.4 |
| `B0002` | Bounds violation | A `@bounds(min, max)` annotation was violated by a constant index at compile time | §6.5 |
| `D0001` | Drift overflow | A `@ZeroDrift` accumulator may overflow the fixed-point range given the loop bounds | §14.8 |
| `R0001` | Require unsatisfied | A `@require(feature >= N)` hardware requirement is not met by the current `.ysu_hw_profile` | §9 |

### Co-Processor Errors

| Code | Meaning | See |
| :--- | :--- | :---: |
| `SMEM_OVERFLOW` | `coprocessor_smem` budget exceeds SM limit (48 KB default) | §11.6, §17 |
| `CUDA_ERROR_INVALID_PTX` | Invalid PTX JIT compilation — usually from incorrect `.shared` declarations or non-ASCII characters | §11.2 |

### Runtime / JIT Errors

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `CUDA_ERROR_INVALID_PTX` | Generated PTX violates SM architecture constraints | Check `chisel {}` PTX for target SM compatibility; rebuild with correct `--emit-coprocessor` |  
| `CUDA_ERROR_ILLEGAL_ADDRESS` | Shared memory access out of allocated range | Verify `SmemLayout` dimensions and `coprocessor_smem` offsets |  
| `CUDA_ERROR_LAUNCH_FAILED` | Kernel launch with wrong grid/block dimensions | Check thread count assumptions in co-processor benchmarks |  

### Warnings

| Code | Meaning | Action |
| :--- | :--- | :--- |
| `B0001` | Bank conflict detected (warning, not error) | Add `swizzle=330` to `SmemLayout` or restructure access pattern |
| `W0001` | `chisel {}` block detected inside `@safe` scope | Review inline PTX manually — safety guarantees do not apply |
| `W0002` | Fragment register count exceeds 30 live variables | Interleave `store()` calls to reduce register pressure |

---

## 24. Configurable ZK Scalar Fields & Proof Schemes

Y-Lang supports target-agnostic cryptographic circuit compilation, parameterizing modular field arithmetic, cryptographic hash intrinsics, and bit-decomposition constraints based on the chosen ZK target.

### 24.1 Target Specification (`@zk_target`)

A module can declare its ZK target using the `@zk_target` attribute:

```ysu
@zk_target(field = "pallas", scheme = "r1cs", opt_level = 1)
module MyCircuit {
    fn main(x: I32, y: I32) -> I32 {
        let hash = poseidon_hash(x, y);
        return hash;
    }
}
```

The attribute accepts the following parameters:

| Parameter | Supported Values | Description |
| :--- | :--- | :--- |
| `field` | `"bn254"`, `"bls12_381"`, `"pallas"`, `"vesta"` | The scalar field order prime defining modular operations. |
| `scheme` | `"r1cs"`, `"plonkish"` | The cryptographic constraint format emitted by the backend. |
| `opt_level` | `0`, `1`, `2` | Compiler optimization level (dead-wire elimination, merge terms). |

### 24.2 Dynamic Modular Arithmetic

All operations on variables in the ZK backend dynamically adapt to the scalar field modulus of the selected target:

| Target Field | Modulus ($r$) | Capacity (Bits) | Capacity (Bytes) |
| :--- | :--- | :---: | :---: |
| **BN254** | `21888242871839275222246405745257275088548364400416034343698204186575808495617` | 254 | 32 (253-bit scalar bound) |
| **BLS12-381** | `52435875175126190479447740508185965837690552500527637822603658699938581184513` | 255 | 32 |
| **Pallas** | `28948022309329048855892746252171976963363056481941560715954676764349967630337` | 255 | 32 (254-bit scalar bound) |
| **Vesta** | `28948022309329048855892746252171976963363056481941600134020817490249052636161` | 255 | 32 (254-bit scalar bound) |

### 24.3 Poseidon Hash Intrinsics

The compiler provides `poseidon_hash(...)` as a built-in cryptographic intrinsic. The S-box and MDS matrix constants adapt dynamically based on the active field's generator. If all arguments are compile-time constants, the intrinsic evaluates to a constant; otherwise, it registers modular constraints $x^5$ across the Poseidon permutation rounds:

$$x^2 = x \cdot x$$
$$x^4 = x^2 \cdot x^2$$
$$x^5 = x^4 \cdot x$$

### 24.4 Automatic Bounds Validation

When a variable is annotated with `@bounds(min=..., max=...)`, the compiler generates binary bit-decomposition constraints for range enforcement:

1. Decompose the variable $x$ into $N$ bits:
   $$x = \sum_{i=0}^{N-1} b_i \cdot 2^i \quad \text{where } b_i \cdot (b_i - 1) = 0$$
   where $N = \text{bit\_len}(\text{max})$.
2. Also decompose the difference $(\text{max} - x)$ into $N$ bits to enforce $x \leq \text{max}$.

---

*made by YSU-SSS research*
