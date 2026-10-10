# Known issues and limits

Reviewed against the working tree on 2026-10-10 after shortening `CLAUDE.md`.
The findings below are source reviews, not new behavioral reproductions or an
exhaustive bug audit. No compiler behavior was changed by this documentation work.

## Issues still visible in current source

| Issue | Evidence and consequence | Historical project snapshot lines |
| --- | --- | --- |
| Unsigned SMT integer proofs remain refused | `src/type_checker.rs`, `smt_integer_width`: identifier types accepted are i8/i16/i32/i64; unsigned identifiers fail with a supported-signed-type error. LLVM's unsigned operators have since been implemented; lifting the proof restriction remains separate work. | 2437, 2453 |

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
