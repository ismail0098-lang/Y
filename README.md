# Y Compiler

This is just pure pain and agony honestly instant-ngp all over again

SELF HOSTEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEED

Y is a systems programming language and self-hosted compiler infrastructure. Initially designed to support the YSU-Engine (a secure OS and GPU kernel engine), Y-Lang is evolving into a full-featured, general-purpose systems language.

## Features

- **Multi-Backend Emitters:**
  - **LLVM IR (`llvm_emitter.rs`):** Direct translation of the Y-Lang AST into SSA-form LLVM IR.
  - **C Backend (`c_emitter.rs`):** Generates C code for broad platform compatibility. (currently a work-in-progress)
  - **Dynamic Hardware Requirements:** Expressive hardware constraints using `@require(condition)` to gate code execution based on probed hardware capabilities (SM version, core counts, specialized intrinsics).
- **Sentient Backend:** Dynamically tunes instruction selection and thermal unrolling based on a real-time hardware profile (`.ysu_hw_profile`).
- **Data-Oriented Design:** The parser and AST are built with performance and memory safety in mind using arena allocation.
- **Self-Hosting Objective:** The ultimate goal of the project is to rewrite the Rust-based compiler modules into Y-Lang itself (`.ysu` files) to close the compilation loop.

## Dynamic Hardware Requirements

Y-Lang does not use static target flags. Instead, kernels declare their hardware needs using the `@require` attribute. This allows the same source code to target multiple hardware generations while ensuring the backend only executes on compatible silicon.

```ylang
@require(sm >= 89)          // Require Ada Lovelace or newer for Tensor Core ops
@require(tensor_cores >= 4) // Require minimum hardware acceleration
kernel matmul_16x16(...) {
    // ...
}
```

## Project Structure

- `lexer.rs` & `parser.rs`: Source code tokenization and AST generation.
- `ast.rs`: Abstract Syntax Tree definitions, handling everything from structs and enums to hardware-specific cache policies.
- `type_checker.rs`: Semantic analysis, variable scoping, and type inference.
- `llvm_emitter.rs`: The LLVM Intermediate Representation backend.
- `*.ysu`: Y-Lang source files (e.g., testing self-hosted parser logic).

## Building

To compile the Y-Lang bootstrap compiler using Rust:

```bash
# If using Cargo
cargo build

# If compiling directly via rustc
rustc main.rs --edition 2021
```

