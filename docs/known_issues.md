# Known issues and limits

Reviewed against the working tree on 2026-10-10 after shortening `CLAUDE.md`.
The findings below are source reviews, not new behavioral reproductions or an
exhaustive bug audit. No compiler behavior was changed by this documentation work.

## Issues still visible in current source

| Issue | Evidence and consequence | Historical project snapshot lines |
| --- | --- | --- |
| Ordinary LLVM Q formats have integer fallback lowering | `src/llvm_emitter.rs`, `emit_type`: primitive types absent from `primitive_llvm_type` default to `i32`; the nearby comment explicitly identifies `Q16.16 = 1.5` truncation outside `@ZeroDrift`. | 2266 |
| LLVM/C `ystr_len` signatures disagree | `src/llvm_emitter.rs` declares `i64 @ystr_len(ptr)` while `c_src/runtime.c` defines `int32_t ystr_len(int32_t s)`. Signature disagreement remains even though the adjacent historical `str_to_i64` return-width bug has changed. | 2305 |
| Unsigned SMT integer proofs remain refused | `src/type_checker.rs`, `smt_integer_width`: identifier types accepted are i8/i16/i32/i64; unsigned identifiers fail with a supported-signed-type error. LLVM's unsigned operators have since been implemented; lifting the proof restriction remains separate work. | 2437, 2453 |
| `reuse_count` cache hint is not lowered | `src/parser.rs` stores `CachePolicy.reuse_count`, defined in `src/ast.rs`; a search of `src/` finds no consumer outside those files. | 2130, 2151 |
| Solver discovery depends on process working directory | `src/type_checker.rs`, `z3_candidates`: project-venv paths are relative. `Y_Z3_PATH`, PATH, and the home-local candidate can still provide a solver. Missing solvers fail invariant verification; do not assume all solver-dependent tests skip. | 2214 |

## Fixed since this review

| Issue | Fix and regression gate |
| --- | --- |
| Python GPU PTX cache omitted compiler identity | The key holds a SHA-256 of the loaded `liby.so`, its path, the target, the hardware profile's hash and the `Y_`/`YSU_` environment. A rebuilt or re-pointed library refuses cache hits until Python restarts. The loaded mapping is matched by path and inode, not device: btrfs reports different devices to `stat()` and `/proc/self/maps`, and comparing devices refused every real library. `python/tests/test_ptx_cache.py`. |
| LLVM data-carrying enums | Constructors store each field in its own slot with its declared type, `match` tests the tag and binds typed payloads per arm, and `x.data.Variant._N` reads and writes a field. Payloads are scalars or references, at most 8; aggregate, generic and Q payloads and ambiguous constructor names are refused by name. `tests/llvm_enum_payloads.rs`. |
| Void `ysu_main` exit status | The CLI's AOT builds give a `main` with no return type an `i32` return of 0; at `-O0` it exited with the last callee's `eax` (43 in the test). Library and JIT callers keep the source's `void`. `tests/llvm_void_main_status.rs`. |

## Verification limits

`tools/ptxas_tval/README.md` states that licensed validation requires explicit
matching sm89 targets and that general unrolled-loop correspondence remains
unsupported. Assembly on another architecture is assembly evidence. Preserve
`UNPROVED`/refusal results; do not turn incomplete coverage into a correctness claim.

The archived PTX let-bound multiply/add contraction report (project lines
2143/2151) also needs checking before changes to rounding semantics. The current
`try_emit_fma` recognizes expression-tree patterns; a clean committed-corpus
contraction census alone does not cover every user-written binding pattern.

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
