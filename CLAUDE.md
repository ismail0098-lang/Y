# Y compiler: working instructions

Y is a Rust systems-language compiler with LLVM CPU, NVIDIA PTX, CPU JIT,
and optional R1CS backends. This directory is the active Cargo project.

## Working rules

- Read the relevant source and tests before changing behavior. Preserve user
  changes and keep edits scoped to the requested task.
- Correctness claims must fail closed: unsupported AST nodes, instructions,
  types, or solver failures must produce an explicit error, never a guessed
  value, silent no-op, or weaker proof. Sound over-approximation is acceptable.
- Preserve `@safe` initialization, pointer, bounds, and invariant checks;
  numerical interval soundness; and exactly-once async token consumption on
  every execution path. Do not bypass failures with
  `Y_ALLOW_UNVERIFIED_INVARIANTS=1`. Verify low-level `chisel` changes explicitly.
- Test observable behavior, including positive and negative controls. Execute
  generated host code; assemble PTX and check semantics when changing codegen.
  A substring match or successful assembly alone does not establish correctness.
- Use the compiler built from the current sources. Tests invoking the CLI should
  use Cargo's test binary and `tests/common/pinned.rs` for isolated, deterministic
  fixtures. Avoid overwriting committed artifacts or relying on local GPU/cache
  state. Keep aggregate and per-target test runs sequential; do not edit files
  while a mutation harness is restoring them.
- Benchmark correctness first. Retain raw inputs, outputs, toolchain/hardware
  identities, paired measurements, and regressions. Distinguish measured results
  from models and proofs; do not publish stale counts or unsupported speedups.

## Build and checks

Run commands here. Select checks relevant to the changed subsystem.

```sh
cargo build
cargo build --release
cargo test                         # root package; excludes y-gpu's own tests
cargo test -p y-gpu                # GPU library's separate tests
cargo test --features zk           # optional R1CS backend
cargo test ptx_emitter             # PTX/swizzle changes
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py'
PYTHONPATH=python python3 -m unittest discover -s python/y_lang/tests -p 'test_*.py'
```

`cargo test --workspace` covers both packages. Python tests must use a freshly
built `target/release/liby.so`; FFI signatures must match `src/c_api.rs`.
Tests using Z3, clang, gdb, CUDA, or ptxas may depend on those tools; report skips.
For translation-validator changes, read `tools/ptxas_tval/README.md` and run the
applicable regression, self-check, documentation, and mutation gates.

## Compiler entrypoints

```sh
cargo run --bin Y -- tests/hello.ysu
cargo run --bin Y -- examples/cpu_jit.ysu --jit
cargo run --bin Y -- tests/hello.ysu -g -o /tmp/y-hello
cargo run --bin Y -- tests/paged_decode_attention_128_32_8_16.ysu --emit-ptx -o /tmp/y-attention.ptx
```

- LLVM is the default. `-g`/`--debug` are LLVM-only; the Y entry symbol is
  `ysu_main`. `--emit-ptx --lineinfo` adds GPU source maps. `tools/ydb/` provides
  Y-aware gdb commands, PTX/SASS mapping, and line-level guarantee reports.
- `--emit-cpu` prints Rust/AVX source; it does not compile it. `--emit-native`
  supports a restricted straight-line integer subset; reject unsupported forms.
  The C backend and `src/c_emitter.rs` have been removed.
- R1CS requires `cargo build --release --features zk` and `--target=r1cs`.
  A build without `zk` can exit successfully without emitting a circuit; check
  artifacts. Large loops may need `Y_ZK_MAX_UNROLL` above its 10,000 default.
  Outputs land beside the input; use copies of fuzz-corpus fixtures.
- PTX requires a suitable target/profile. Set `SM_VERSION` in an isolated
  `.ysu_hw_profile` when testing GPU-less compilation of hardware requirements.

## Source map

- Front end: `src/{lexer,parser,ast,type_checker,lexical_scope}.rs`;
  safety/resources: `linear_tracker.rs`, `guarantees.rs`, `zero_drift.rs`.
- Emitters: `llvm_emitter.rs`, `ptx_emitter.rs`, `cpu_emitter.rs`,
  `native_emitter.rs`, `rt_core_emitter.rs`, `zk_emitter.rs`, `zk_witness.rs`.
- Hardware/tuning: `sentinel.rs`, `ysu_gpu_probe.rs`, `autotuner.rs`,
  `empirical_autotune.rs`, `bank_conflict.rs`, `coprocessor_scheduler.rs`.
- CPU JIT: `src/cpu_jit.rs`, `src/cpu_jit/`, `src/cpu_jit_ffi.rs`;
  APIs: `src/c_api.rs`, `python/y_lang/`; GPU library: `crates/y-gpu/`.
- Self-hosted compiler: `self_hosted/`; runtime: `c_src/`; workloads: `tests/`.

## Subsystem contracts

- PTX swizzling, `ldmatrix`, and `cp.async` addresses must agree. Co-processor
  emitters share a unified SMEM layout. Verify target/version and actual ptxas
  assembly; do not assume Hopper TMA/WGMMA support exists.
- Attention specialization depends on kernel name and signature. Verify the
  emitted specialization and host launch contract. Split attention requires
  both main and reduction launches; partial states remain unnormalized f32.
- `.ysu_hw_profile` holds both hardware probes and measured autotuning. Reprobe
  after hardware/driver changes; use `--autotune-force` after GEMM codegen changes.
  `--no-autotune` avoids measured tile selection. `Y_CTA_OVERRIDE` can contaminate
  regenerated PTX. Never commit machine-specific profiles or generated caches.
- R1CS uses BN254 scalar-field semantics, not ordinary integer arithmetic.
  Constraint/wire rewrites must also update witnesses, recipes, and metadata.
  Use independent witness/proof oracles for changes to these paths.
- `@ZeroDrift` needs exact integer/fixed-point accumulation; float precision
  alone does not establish order independence.

## Known defects and limitations

Keep unresolved issues visible; archive a report only after recording its status.
These notes were checked against source, not freshly reproduced. Read
[known issues](docs/known_issues.md) before related changes:

- LLVM Q formats outside `@ZeroDrift` are scaled integers that trap on overflow
  (`fixed.rs`); Q arrays, `match` and built-in arguments are refused.
  Enum payloads are scalar only (at most 8 fields); others are refused.
- SMT integer-width checking still rejects unsigned variables in invariant proofs.
- Z3's project-venv lookup depends on the working directory; invariant checks fail
  without a solver. The PTX translation validator licenses sm89 only and does not
  establish general unrolled-loop correspondence.

Other unresolved reports are indexed in
[the reported-issue backlog](docs/claude-history/UNRESOLVED.md). Check relevant
entries before changes; archived status does not mean a bug was fixed. Some older
reports are superseded, so confirm source/tests before treating one as current.

## Read more only when relevant

- Language/backend reference: [manual](docs/y_language_documentation.md).
- CPU JIT APIs/policy: [CPU JIT](docs/cpu_jit.md); measurement procedure:
  [benchmark guide](benchmarks/cpu_jit/README.md).
- GPU runtime: [adaptive JIT](docs/adaptive_jit.md); validator:
  [validator guide](tools/ptxas_tval/README.md).
- Historical investigations: [archive index](docs/claude-history/INDEX.md).
  Both original instruction files are preserved verbatim there. Search for the
  relevant symbol/topic and read a small line range; do not load the whole archive.
  Historical tasks and test counts are context, not current instructions/status.

Keep this file under 8 KB. Put session logs, benchmark tables, mutation matrices,
and detailed investigations in `docs/`; link them here only when useful. Avoid
automatic imports of history or duplicating the parent instruction file.
