# Y Compiler Foundations

This plan addresses the foundational steps for building the new "Y" compiler, combining the parallel tasks assigned to Subagent A and Subagent C.

## Proposed Changes

### Compiler Backend & Architecture (Subagent A)
Research indicated that Rust's `std::arch::x86_64` provides native SIMD intrinsics. We will draft a specialize wrapper for AVX/AVX2 utilizing `__m256` arrays.
#### [NEW] `avx_wrapper.rs` (file:///c:/YSU-engine-main/YSU-engine-main/src/y/avx_wrapper.rs)
A Rust module implementing `Y256Register`—a safe wrapper around `std::arch::x86_64::__m256` and `__m256i` supporting add/sub/mul and type conversions.

### Front-End Lexer (Subagent C)
Based on scanning `Y_Specification.docx`, we extracted keywords such as `kernel`, memory spaces (`GlobalMemory`, `SharedMemory`), layout types, and attributes (`@require`).
#### [NEW] `keywords.json` (file:///c:/YSU-engine-main/YSU-engine-main/src/y/keywords.json)
JSON listing all tokens and keywords from the Y spec.
#### [NEW] `lexer.rs` (file:///c:/YSU-engine-main/YSU-engine-main/src/y/lexer.rs)
A baseline token scanning engine written in Rust that supports Y's unique attributes, keywords, types, and generic parameters (`<`, `>`).

## Verification Plan

### Automated Tests
- Run `rustc --edition 2021 --crate-type lib avx_wrapper.rs` to verify SIMD wrapper compilation.
- Run `rustc --edition 2021 --crate-type lib lexer.rs` to verify lexer logic format.

