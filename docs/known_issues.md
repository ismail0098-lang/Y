# Known issues and limits

Reviewed against the working tree on 2026-10-10 after shortening `CLAUDE.md`.
The findings below are source reviews, not new behavioral reproductions or an
exhaustive bug audit. No compiler behavior was changed by this documentation work.

## Issues still visible in current source

| Issue | Evidence and consequence | Historical project snapshot lines |
| --- | --- | --- |
| `y_inductor` keeps a multiply apart from the add it feeds | `python/y_lang/inductor.py`, `_fusable`: the rule existed because ptxas contracted the pair. The PTX emitter now rounds a `let`-bound product first, and the backend writes one `let` per node, so a fused pair would match eager; the rule now costs kernels for nothing. Relaxing it changes the partitioning, so measure it first. | - |
| `--emit-cpu` writes untyped `let`s | `src/cpu_emitter.rs`, `Stmt::Let`: `let x: U8 = 200;` is written `let mut x = 200;`, so the Rust type of a local comes from inference, not from its declaration, and a narrow or unsigned local can compute in another width than the LLVM backend's. Typing it also needs typed assignments (the emitter keeps no variable types). Loop variables and every scalar parameter are typed. | - |

## Fixed since this review

| Issue | Fix and regression gate |
| --- | --- |
| Python GPU PTX cache omitted compiler identity | The key holds a SHA-256 of the loaded `liby.so`, its path, the target, the hardware profile's hash and the `Y_`/`YSU_` environment. A rebuilt or re-pointed library refuses cache hits until Python restarts. The loaded mapping is matched by path and inode, not device: btrfs reports different devices to `stat()` and `/proc/self/maps`, and comparing devices refused every real library. `python/tests/test_ptx_cache.py`. |
| LLVM data-carrying enums | Constructors store each field in its own slot with its declared type, `match` tests the tag and binds typed payloads per arm, and `x.data.Variant._N` reads and writes a field. Payloads are scalars or references, at most 8; aggregate, generic and Q payloads and ambiguous constructor names are refused by name. `tests/llvm_enum_payloads.rs`. |
| Ordinary LLVM Q formats were computed as integers | `let x: Q16.16 = 1.5` stored 1 and `x > 1.0` was false. A Q value is `value * 2^frac` in 8/16/32/64-bit storage; literals, `*` and `/` round to nearest with ties away from zero; overflow and division by zero trap; struct fields keep the scale; arrays of a Q format, `match` on a Q value and Q arguments to built-ins are refused by name. Manual §20.3; `tests/llvm_fixed_point.rs` (independent oracle, four formats, -O0 and -O2). |
| C runtime widths disagreed with the LLVM declarations | Not only `ystr_len`: seven functions were declared to return `ptr`/`i64` and defined to return `int32_t` (the caller read a register half the callee never set), and four `i64` arguments were received as `int32_t`, so `print_int(5000000000)` printed 705032704. `c_src/runtime.c` now matches the declarations and the JIT's runtime. `tests/runtime_abi_agreement.rs` compares every declaration with clang's type of the definition; handle arguments stay `int32_t` (the pool is `MAP_32BIT`). |
| A name defined twice was not refused | Two `fn f`, two `Type::m`, or a function colliding with an enum constructor: the type checker kept the second; the default build failed inside clang with no reason and `--emit-llvm`/`--emit-cpu` wrote output that does not compile. The front end refuses it by name on every backend. `tests/duplicate_definitions.rs`. |
| `@cache_policy(..., reuse_count=N)` was silently dropped | No backend lowers a reuse count (`createpolicy` takes a fraction of the lines); it is refused by name now. `tests/ptx_cache_policy.rs`. |
| PTX let-bound multiply/add contraction (archived report, project lines 2143/2151) | Confirmed live: `let p: F32 = a * b; p + c` emitted `mul.f32` + `add.f32`, which `ptxas` assembled to ONE `FFMA` under PTX stating two roundings. An unfused float multiply is `mul.rn` now, which `ptxas` does not contract. All 66 corpus kernels emit byte-identical PTX before and after (no shipped kernel takes that path). `tests/ptx_let_bound_rounding.rs` checks the PTX, the SASS and the value on the device (0 rounded twice, 2^-24 fused). |
| Solver discovery depended on the working directory | The project-venv candidates (`venv/bin/z3`, `.venv/bin/z3`, `z3/build/z3`) resolved only against the working directory, so `Y` run elsewhere - or `liby.so` in a Python process started elsewhere - refused every invariant although the repository's own solver existed. They are also looked for beside the compiler's own file (found from the process's mappings, since `current_exe()` names the interpreter inside Python), in its directory and up to three parents. Missing solvers still fail invariant verification. `tests/safe_invariant_enforcement.rs` (a copy of `Y` at the deepest Cargo layout finds a stub solver; one level deeper it does not), `python/tests/test_solver_lookup.py`; tests that need no solver run a copy of the compiler with nothing beside it. |
| `for` headers were evaluated differently by each backend | `for i in 0..20 step k { k = k + 1; }` ran 5 iterations on the LLVM backend and the JIT (the step was re-read after every body) and 20 on the GPU. `--emit-cpu` replaced every non-literal step with 1, re-read the bound every iteration, and let Rust type the variable from a `U32` bound (3e9 ran 3e9 times where every other backend compares its bits as an I32 and runs none); `U8`/`U16`/`U32`/`I64`/`F64`/`bool` parameters were written under their Y names and did not compile. `step 0` was refused only by PTX. Now every backend evaluates `start`, `end`, `step` once, in that order, with an I32 variable compared in 32 bits; `step 0` is refused by the front end. Manual §13.10; `tests/for_header_semantics.rs` (LLVM, JIT, `--emit-cpu` through rustc, and a kernel on the device). |
| Unsigned integers were refused in invariant proofs | Every unsigned variable under an `@invariant`, and so every loop with a `U32` bound, failed with "does not have a supported signed integer type", from when the backends disagreed on unsigned operators (archive lines 2437, 2453). They agree now (`llvm_unsigned_ops`, `llvm_integer_widths`, `ptx_integer_datapath` on the GPU, `for_header_semantics`), and `src/type_checker.rs` models U8..U64 exactly in `0..=2^n-1` with an unsigned quotient. Where backends could read operands differently it is conservative: a literal beside an unsigned value must lie in its range, mixed signedness or unsigned widths are proved within `0..=2^31-1`-style agreement ranges (a `U32` 4294967295 and an `I32` -1 compare equal on the machine), negation of an unsigned value is refused, and a `U32` loop bound is read as the I32 its bits spell. I8/I16/U8/U16/U64 locals are now tracked by the loop proofs. `tests/smt_unsigned.rs`. |
| `@bounds` the compiler could not prove was taken on trust | `@bounds(0, 3) let i: I32 = get(1000000);` compiled, every proof using `i`'s range rested on the annotation, and the index it fed skipped its run-time check: `arr[i] = 42` wrote a million elements past a four-element array under `@safe` and exited 0. An exact-GEMM operand outside its `@bounds` overflowed the int32 accumulator into a wrong answer under a certificate claiming exactness. Now an unprovable range is checked when the program runs: LLVM and the JIT print the value and exit 1, PTX traps, `--emit-cpu` panics, `--emit-native` refuses; the exact GEMM scans both operands before computing; `ydb verify` reports `run-time` and no proof rests on the bound. `tests/bounds_runtime_check.rs`. |
| Void `ysu_main` exit status | The CLI's AOT builds give a `main` with no return type an `i32` return of 0; at `-O0` it exited with the last callee's `eax` (43 in the test). Library and JIT callers keep the source's `void`. `tests/llvm_void_main_status.rs`. |

## Verification limits

`tools/ptxas_tval/README.md` states that licensed validation requires explicit
matching sm89 targets and that general unrolled-loop correspondence remains
unsupported. Assembly on another architecture is assembly evidence. Preserve
`UNPROVED`/refusal results; do not turn incomplete coverage into a correctness claim.


## Earlier reports that must not be blindly restored as active

Source inspection found later implementations addressing several old reports:

- AVX/AVX-512 execution support uses OS-aware `host_has_avx`/`host_has_avx512`
  helpers on fresh and cached hardware paths in `src/sentinel.rs`.
- `String_new` recognizes and clones a Y string handle in `c_src/runtime.c`.
- LLVM's `str_to_i64` function table now uses an i64 return.
- LLVM float inequality uses `fcmp une`; lexical binding renaming is present.
- `src/c_emitter.rs` has been removed, despite older notes calling it an orphan.

These are source observations, not a declaration that every associated defect
is closed. Use the existing regression tests when working on these areas.

## Remaining reported issues

[UNRESOLVED.md](claude-history/UNRESOLVED.md) indexes every line explicitly marked
unfixed in both original files, including duplicate and retracted reports. Each
entry retains its original snapshot line number and investigation context.
The [archive index](claude-history/INDEX.md) covers other findings that did not
use those markers. Original text is preserved verbatim with recorded hashes.

Unreviewed historical issues remain a backlog for verification; omission from
the short startup list is not a resolution. Before related changes, search the
backlog and source, reproduce when necessary, and record the result here. Keep
the startup list concise and current; retain detailed evidence in documentation.
