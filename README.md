# Y-Lang: The Hardware-Sentient Systems Language

Y-Lang (or simply **Y**) is a cutting-edge systems programming language and self-hosting compiler infrastructure designed to push the boundaries of high-performance computing (HPC), AI kernel development, and secure OS engineering. Initially conceived to power the YSU-Engine, Y-Lang bridges the gap between mathematically formal verification and bare-metal, cycle-precise hardware control.

The ultimate achievement of this project is the **Y-Lang Bootstrap & Self-Hosting Loop**. The compiler is initially written in Rust (the bootstrap compiler) but is rapidly transitioning to being entirely written in native Y-Lang (`.ysu` files), establishing a truly independent, self-compiling logic ecosystem.

---

##  Core Architectural Philosophies

Y-Lang abandons traditional compiler design paradigms in favor of **hardware-sentience** and **formal logic**. 

### 1. Hardware-Aware "Sentient" Compilation
Unlike traditional compilers that use static target triples (e.g., `x86_64-pc-windows-msvc`), Y-Lang features a **Sentinel Probe** that dynamically queries the host silicon. The compiler inherently understands:
- L1/L2 Cache Line Sizes
- GPU Memory Latencies (SMEM vs. VRAM vs. L2)
- Warp Shuffle execution speeds
- Tensor Core capabilities and latencies (F16, TF32)
- Branch Divergence Penalties

Using this `HardwareProfile`, Y-Lang's emitter dynamically selects the optimal instructions (e.g., dynamically choosing between `IMAD.WIDE` vs `IMAD` on NVIDIA GPUs based on known clock-cycle latencies).

### 2. The Formal Verification "Safety Cage"
Y-Lang moves beyond runtime safety (Java) and ownership paradigms (Rust) by enforcing mathematically rigorous **Formal Verification** at compile-time using a "Cage" methodology.

By default, the compiler enforces strict rules within `@safe` blocks, while providing developers the freedom to directly manipulate silicon in `@unsafe` or `chisel` blocks.

**Key Safety Directives:**
- **`@safe { ... }`**: A strict verification block. Inside this block, variables *must* be explicitly initialized, and raw pointer dereferencing (`*ptr`) is completely forbidden.
- **`@invariant(expr)`**: Required on all loops (`while` and `for`) within safe blocks. This mathematically proves to the compiler that loop bounds will not violate memory access rules.
- **`@bounds(min, max)`**: Explicitly declares safe index bounds, ensuring zero-cost out-of-bounds safety without requiring runtime checks.
- **`@unsafe { ... }`**: Permits direct hardware interaction, raw pointer dereferencing, and unverified data operations for maximum performance.

### 3. High-Precision Numerical Stability
Y-Lang is designed for environments where a single floating-point rounding error is unacceptable (e.g., aerospace, high-frequency trading).
- **`@ZeroDrift`**: A compiler directive that guarantees zero numerical drift. If a target architecture's fast-math path causes precision loss, the compiler will emit a performance advisory and reroute to a perfectly precise, mathematically verified instruction set.

---

##  Project Structure & Modules

The repository is organized to clearly separate the Rust-based bootstrap infrastructure from the self-hosted Y-native code.

### Directory Layout
*   **`src/`**: The Rust bootstrap compiler source code.
*   **`c_src/`**: Interoperability code, raw C/C++ host bindings, and CUDA (`.cu`) wrappers.
*   **`tests/`**: Y-Lang testing suites (`.ysu` files) and unit test modules.
*   **`self_hosted/`**: The holy grail of the project. Contains the Y-Lang compiler completely rewritten in native Y-Lang.
*   **`scripts/`**: Automation, build scripts, and Python/PowerShell helper patches.
*   **`docs/`**: Language specifications and architecture blueprints.
*   **`build_artifacts/`**: Emitted binaries, LLVM IR outputs, object files, and compiled executables.

### Core Compiler Pipeline (`src/`)

1. **`lexer.rs` (Lexical Analysis)**: Converts raw Y-Lang source strings into an exhaustive `TokenKind` stream. Understands complex systems tokens like `cp_async`, `ldmatrix`, and advanced attributes (`@atomic`, `@cache_policy`).
2. **`parser.rs` (Recursive Descent)**: Parses the token stream into a Data-Oriented AST (`ast.rs`). Uses Arena allocation for blazingly fast memory mapping rather than standard pointer hierarchies.
3. **`type_checker.rs` (Semantic Brain)**: 
   - Evaluates the "Safety Cage" properties.
   - **Linear Tracker**: Prevents resource leaks by ensuring "Transfer Obligations" (like asynchronous memory pipes) are perfectly consumed (`pipe.wait()`) exactly once.
   - **Bank Conflict Prover**: A mathematical prover that evaluates Shared Memory matrix layouts (`SmemLayout`) to guarantee **0-Bank-Conflict** loads on GPU architectures.
4. **Backend Emitters**:
   - **`llvm_emitter.rs`**: Directly translates the AST into highly-optimized LLVM IR.
   - **`c_emitter.rs`**: Emits equivalent, highly-portable standard C code.
   - **`ptx_emitter.rs`**: Emits bare-metal NVIDIA PTX assembly for ultimate GPU kernel generation.

---

##  Directives & Advanced Syntax

Y-Lang provides an expressive grammar specifically tailored for hardware layout manipulation:

### Hardware Requirements
Rather than passing `-target` flags via command line, Y-Lang relies on source-code-level hardware constraints:
```ysu
@require(sm >= 89)          // Require Ada Lovelace or newer
@require(tensor_cores >= 4) // Require minimal acceleration
kernel matmul(A: GlobalMemory<F16>, B: GlobalMemory<F16>) { ... }
```

### Advanced Memory Management
```ysu
// Cache hints directly on variable initialization
let a: I32 = 0;
@cache_policy(L2_PERSIST, reuse_count=4)
let persistent_var = a;
```

### Asynchronous Pipelining
```ysu
let pipe = Pipeline::init();
let token = cp_async(smem_buffer, global_mem); // Creates a Linear Obligation
pipe.wait(token); // Safely discharges the obligation
```

---

##  Building & Running

The Y-Lang compiler is currently bootstrapped via Rust. You can compile the compiler itself, and then use it to compile Y-Lang source code (`.ysu`).

```bash
# 1. Build the bootstrap compiler
cargo build --release

# 2. Run the compiler against a Y-Lang source file
cargo run -- tests/math.ysu
```

During execution, you will see the compiler probe your host hardware, parse the source code, run the mathematical semantic type-checkers (including the Bank Conflict Prover), and emit native binaries or LLVM/PTX blobs.

---