# Archived Claude instructions

These historical snapshots are not active instructions. Both originals are preserved
byte for byte; `manifest.json` records sizes and SHA-256 hashes. Old queues and
measurements may be superseded by current sources. Do not import these files.

Search for the affected symbol or topic, then read only the matching line range:

```sh
rg -n -i 'symbol_or_topic' docs/claude-history/*-before-2026-10-10.md
sed -n 'START,ENDp' docs/claude-history/project-before-2026-10-10.md
```

Replace the search term and `START,END` before running. Paths are relative to `Y/`.

## [Y/CLAUDE.md](project-before-2026-10-10.md)

1,122,171 bytes; 2,453 lines. Snapshot line numbers:

- Line 3: Overview
- Line 14: Build & Test Commands
- Line 16: Rust Bootstrap Compiler
- Line 39: Running the Y Compiler CLI
- Line 84: Python Package & Benchmarks
- Line 127: Architecture & Compiler Modules (`src/`)
- Line 157: Gotchas & Development Trapdoors
- Line 442: Design Rule: unhandled AST nodes in soundness-critical passes MUST reject
- Line 1433: Security, Safety & Verification Constraints

Subsystem notes in the project snapshot:

- Line 159: 1. Shared-Memory Bank Conflict Swizzling (`bank_conflict.rs` & `ptx_emitter.rs`):
- Line 161: 2. Co-Processor Shared Memory Offset Aliasing:
- Line 163: 3. Autotuner Result Cache (also `.ysu_hw_profile`):
- Line 168: 4. Hardware Profile Cache (`.ysu_hw_profile`):
- Line 170: 5. Paged Decode Attention Kernel Naming & Launch Contract (`ptx_emitter.rs`):
- Line 182: 6. Fused SwiGLU Tile Is Register-Constrained, Not Under-Tuned (`emit_gemm_swiglu_kernel`):
- Line 186: 7. The PTX Integer Datapath (`ptx_emitter.rs`):
- Line 248: 8. PTX Tests Must Assemble, Not Just String-Match:
- Line 301: 9. R1CS / BN254 Scalar Field Constraints (`zk_emitter.rs`, `zk_witness.rs`):
- Line 403: 10. The Generative Differential Fuzzer (`src/zk_fuzz.rs`, `tests/zk_fuzz_differential.rs`):
- Line 415: 11. The Parser's Struct-Literal Ambiguity (`parser.rs`):
- Line 422: 12. Self-Hosted Compiler Parity (`self_hosted/`):
- Line 428: 13. Y ShadowPlay, the repo's only end-user application (`shadowplay/shadowplay.ysu`, `c_src/shadowplay_gui.h`):

Latest investigations in the project snapshot (historical, not a task queue):

- Line 2284: THREE LOCAL COMMITS AND THE DEBUGGER REACHED `origin/main` BY REBASE, AND THE REBASE FOUND TWO THINGS THE COMMITS HAD WRONG ON IT
- Line 2290: LEXICAL SCOPES: A Y BINDING EXISTS WHERE THE LANGUAGE SAYS IT DOES - IN THE PROGRAM FIRST, THEN IN THE DEBUGGER
- Line 2298: Y VALUES PRINT AS Y AND Y STACKS SHOW AS Y, THROUGH AN EXTENSION THE BINARY CARRIES
- Line 2307: AN OPTIMISED BUILD IS DEBUGGABLE: `-O0` .. `-O3` REACH THE CLANG STEP, AND THE DEBUG INFORMATION SAYS IT DESCRIBES OPTIMISED CODE
- Line 2323: WHICH PTX AND SASS EACH Y LINE BECAME: `--emit-ptx --lineinfo` AND `tools/ydb/ymap.py`
- Line 2351: AN ELEMENT REACHED THROUGH A POINTER IS AS WIDE AS ITS TYPE: THE LLVM BACKEND INDEXED EVERY POINTER IN 8-BYTE SLOTS
- Line 2371: `ydb`: Y-AWARE COMMANDS OVER gdb, NOT A SECOND DEBUGGER
- Line 2387: AN ARRAY REACHED THROUGH A REFERENCE IS BOUNDS-CHECKED LIKE THE ARRAY: THE STRICT-MODE ARRAY RULE MATCHED A PLAIN ARRAY TYPE ONLY
- Line 2398: `ydb verify`: WHAT COVERS A LINE - THE COMPILER'S OWN GUARANTEES, AND THE EVIDENCE ABOUT THE CODE IT BECAME
- Line 2422: CODEX'S UNCOMMITTED WORK, INTEGRATED - AND AN ATTRIBUTION RUN IS WHAT SEPARATED A MERGE BUG FROM ONE THAT WAS ALREADY THERE
- Line 2439: HALF OF THE FIELD WALL'S FIRST SWEEP WAS THE CUTS, NOT THE SOLVER
- Line 2445: CODEX'S CPU JIT INTEGRATED, AND A NESTED `let` IS A NEW BINDING ON THE NATIVE, PTX AND CPU BACKENDS TOO

## [CLAUDE.md](parent-before-2026-10-10.md)

1,027,724 bytes; 2,239 lines. Snapshot line numbers:

- Line 3: Overview
- Line 14: Build & Test Commands
- Line 16: Rust Bootstrap Compiler
- Line 39: Running the Y Compiler CLI
- Line 83: Python Package & Benchmarks
- Line 125: Architecture & Compiler Modules (`src/`)
- Line 155: Gotchas & Development Trapdoors
- Line 440: Design Rule: unhandled AST nodes in soundness-critical passes MUST reject
- Line 1434: Security, Safety & Verification Constraints

