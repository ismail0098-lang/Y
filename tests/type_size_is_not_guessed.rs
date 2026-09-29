// ============================================================
//  A length is part of a TYPE, so guessing one is not lossy --
//  it produces a different type that still compiles.
//
//  Three sites guessed:
//
//    type_checker::resolve_type   `BlockTile<T, N>` with a non-literal or
//                                 missing size -> 128, at TWO separate arms.
//                                 `SemanticType` derives `PartialEq` and
//                                 `types_are_compatible` starts with
//                                 `t1 == t2`, so both tiles resolved to 128
//                                 and COMPARED EQUAL -- an assignment between
//                                 differently-sized tiles was legal.
//    cpu_emitter::emit_type       an array length it cannot evaluate -> 0,
//                                 and a `BlockTile` size -> 128. A zero-length
//                                 array is not a conservative default; it is a
//                                 type whose every element access is out of
//                                 bounds, and Rust accepts `[f32; 0]`.
//
//  Each case asserts on ITS OWN diagnostic, so a fixture stopped by an earlier
//  pass fails instead of passing green.
// ============================================================

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static N: AtomicUsize = AtomicUsize::new(0);

fn bin() -> std::path::PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("Y")
}

/// Compiles `src` and returns combined stdout+stderr.
fn compile(src: &str, flag: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "y_typesize_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("t.ysu");
    std::fs::write(&f, src).unwrap();
    let out = Command::new(bin())
        .arg(&f)
        .arg(flag)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let _ = std::fs::remove_dir_all(&dir);
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn a_block_tile_size_the_compiler_cannot_evaluate_is_refused() {
    let out = compile(
        "fn main() -> I32 {\n    let t: BlockTile<F32, SIZE> = ZeroInit;\n    return 0;\n}\n",
        "--emit-cpu",
    );
    assert!(
        out.contains("BlockTile") && out.contains("literal"),
        "a non-literal tile size was not refused by name:\n{out}"
    );
}

#[test]
fn a_literal_block_tile_size_still_compiles() {
    // The control. Refusing every size is sound and useless -- the same shape
    // as `ordinary_loop_bodies_still_verify` guarding the SMT encoding, and as
    // `thirty_two_bit_programs_still_compile_and_run` guarding the native
    // emitter's width refusal.
    let out = compile(
        "fn main() -> I32 {\n    let t: BlockTile<F32, 64> = ZeroInit;\n    return 0;\n}\n",
        "--emit-cpu",
    );
    assert!(
        !out.contains("BlockTile size") && !out.contains("needs a literal size"),
        "a perfectly ordinary literal tile size was refused:\n{out}"
    );
}

#[test]
fn a_computed_size_is_a_parse_error_which_is_why_the_emitter_arms_are_unreachable() {
    // CONFIRMATION, not a guard -- and the distinction was established by
    // mutation, not by reading. The first version of this test asserted that
    // `--emit-cpu` never prints `[f32; 0]` for `Array<F32, 4 * 4>`, and it
    // passed with the emitter's guess RESTORED. The reason is here: a generic
    // size argument may only be a literal or a bare identifier, so the `_`
    // arms in `cpu_emitter::emit_type` cannot be reached from source at all.
    //
    // The emitter still refuses instead of guessing, because "unreachable
    // today" is a property of the parser and not of the emitter -- exactly the
    // reasoning CLAUDE.md records for `break` in the PTX backend. But the
    // assertion that has teeth is this one.
    for ty in ["Array<F32, 4 * 4>", "BlockTile<F32, 4 * 4>"] {
        let src = format!("fn main() -> I32 {{\n    let a: {ty} = ZeroInit;\n    return 0;\n}}\n");
        let out = compile(&src, "--emit-cpu");
        assert!(
            out.contains("Syntax Error"),
            "`{ty}` parsed, so the emitter's unevaluable-size arm is now \
             REACHABLE and needs a real test rather than this confirmation:\n{out}"
        );
    }
}

#[test]
fn a_named_array_length_is_passed_through_to_the_host() {
    // The other half of the parser's surface: an identifier is emitted as-is,
    // which is the right answer for `--emit-cpu` (it prints host source for
    // the user to paste, so the const is theirs to supply). Pinned so that
    // "refuse anything not a literal" cannot be over-applied here.
    let out = compile(
        "fn main() -> I32 {\n    let a: Array<F32, SIZE> = ZeroInit;\n    return 0;\n}\n",
        "--emit-cpu",
    );
    assert!(
        !out.contains("Syntax Error") && !out.contains("[CPU Backend]"),
        "a named array length was refused:\n{out}"
    );
}
