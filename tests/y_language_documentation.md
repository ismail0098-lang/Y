# Y Language: The Definitive Specification & Programmer's Reference Manual
Version 1.0 (Operational Systems Programming Language)

Y is a hardware-sentient, low-level systems programming language designed for high-performance computing, lock-free concurrency, and hardware-aware GPGPU/CPU acceleration. It couples structural type checking with hardware profiles (gathered via Sentinel Probes) to enforce optimal performance traits, cache alignments, memory layouts, and register usage directly at compile time.

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

LetStmt         = { Attr } , "let" , Ident , [ ":" , Type ] , "=" , Expr , ";" ;
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
* **Integers**: `I8` through `I64` (signed), `U8` through `U64` (unsigned), and `u3` (3-bit register values for GEP indexing).
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
    unsafe {
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
- A per-thread variable-depth stack traversal (`stack_depth = (tid % 4) == 0 ? 16 : ...`) producing severe warp branch divergence.
- A manual `if (diff * diff < 4.0f)` distance-check branch — divergent across warp lanes.
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

Y includes a standalone ZK circuit compiler backend that generates R1CS (Rank-1 Constraint Systems) directly from annotated Y programs. It does not use an intermediate circuit DSL — the same Y syntax compiles to either native binaries or ZK constraint systems depending on the backend flag.

### 12.1 What is R1CS?

An R1CS system encodes a computation as a set of quadratic constraints of the form:

$$\mathbf{a} \cdot \mathbf{w} \;\times\; \mathbf{b} \cdot \mathbf{w} = \mathbf{c} \cdot \mathbf{w}$$

where **w** is a witness vector. A satisfying assignment to **w** proves that the computation was performed correctly without revealing the inputs. R1CS is the standard format consumed by ZK proving systems like Groth16, PLONK, and Nova.

### 12.2 Writing a ZK Circuit in Y

ZK programs use the same Y syntax with two constraints:
- Use `@unsafe` at the function level (mutable variable reassignment requires it in the current ZK emitter).
- Field arithmetic operates in the BLS12-381 scalar field by default (254-bit prime).

```ysu
// Simple dot product circuit: proves knowledge of vectors a, b s.t. a·b = result
@unsafe
fn dot_product(a: [F32; 4], b: [F32; 4]) -> F32 {
    let result: F32 = 0.0;
    for i in 0..4 {
        result = result + (a[i] * b[i]);
    }
    return result;
}
```

Compile to R1CS:
```bash
cargo run -- dot_product.ysu --emit-r1cs
# Outputs: dot_product.r1cs, dot_product.sym
```

### 12.3 Compiler Output

The ZK backend emits three files:

| File | Contents |
| :--- | :--- |
| `.r1cs` | Binary R1CS constraint system (compatible with snarkjs, bellman) |
| `.sym` | Symbol table mapping wire indices to variable names |
| `.r1cs.txt` | Human-readable constraint listing for debugging |

Example `.r1cs.txt` output for a 4-element dot product:
```
Constraint 0: (1 * a[0]) * (1 * b[0]) = (1 * _mul_0)
Constraint 1: (1 * a[1]) * (1 * b[1]) = (1 * _mul_1)
Constraint 2: (1 * a[2]) * (1 * b[2]) = (1 * _mul_2)
Constraint 3: (1 * a[3]) * (1 * b[3]) = (1 * _mul_3)
Constraint 4: (1 * _mul_0 + 1 * _mul_1 + 1 * _mul_2 + 1 * _mul_3) * (1 * 1) = (1 * result)
```

### 12.4 ZK-Specific Directives

| Directive | Effect |
| :--- | :--- |
| `@ZeroDrift` | Verifies that fixed-point accumulation does not drift beyond the field modulus over iteration |
| `@safe` on loops | Requires `@invariant` — used by the ZK emitter to unroll bounded loops into flat constraint sequences |
| `@bounds(min, max)` | Used by the constraint generator to statically verify index ranges, preventing out-of-range witness accesses |

### 12.5 Performance vs Other ZK Compilers

Benchmarks run on AMD Ryzen 9 9950X, 48 GB DDR5-6000. Results are physically measured (not estimated) except where noted.

**1,000,000 constraints (`heavy_circuit`):**

| Compiler | Time | Peak Memory |
| :--- | :---: | :---: |
| **Y** | **1.67s** | **1.07 GB** |
| Noir (Nargo) | 11.36s | 1.25 GB |
| Leo | 41.52s | 10.81 GB |
| Circom | 259.25s | 2.39 GB |

**31,000,000 constraints (`heavy_31m.ysu`):**

| Compiler | Result |
| :--- | :--- |
| **Y** | **105.28s, 30.65 GB peak RSS** |
| Noir | Estimated ~39 GB required (did not complete) |
| Leo | Estimated ~335 GB required (did not complete) |
| Circom | Estimated ~74 GB, ~2.2 hours (did not complete) |

**Why Y uses less memory at scale:**
- In-place accumulator updates avoid O(N) vector copies on loop-scoped reassignment.
- Linear-combination addition is O(1) when inputs are already flat.
- Constraint deduplication uses an order-independent hash map, not a sorted list.

### 12.6 Verifying Generated Circuits

Verification scripts are included:
```bash
python verify_r1cs.py      # dot_product circuit
python verify_heavy.py     # heavy_circuit (1M constraints)
python verify_benchmarks.js  # snarkjs-based verification
```

---

## 13. CUDA-to-Y Migration Guide

This section maps common CUDA C++ patterns directly to their Y equivalents. It is intended for GPU developers who know CUDA and want to understand Y's model quickly.

### 13.1 Memory Declarations

| CUDA C++ | Y Equivalent | Notes |
| :--- | :--- | :--- |
| `__shared__ float buf[4096];` | `let buf = SharedMemory::alloc<SmemLayout<F32, ...>>();` | Y requires an explicit layout type |
| `__global__ float* ptr` | `data: GlobalMemory<F32>` | Global memory is a first-class type |
| `__device__ float val;` | `let val: F32 = ...;` | Regular variable in kernel scope |
| `alignas(128) float x;` | `@align(128) let x: F32 = ...;` | Alignment as a decorator |
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

    // Manual BVH traversal with branch divergence
    int stack_depth = (tid % 4) == 0 ? 16 : 32;
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

The swizzle value `330` comes from the standard CuTe/CUTLASS swizzle formula for 32-column F32 tiles. The Y compiler validates the conflict-free property statically.

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

*made by YSU-SSS research*
