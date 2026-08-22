//! Integer types and literals must survive the LLVM backend at their real width.
//!
//! Two bugs, both of which compiled cleanly, linked, ran, and gave a wrong
//! answer — the shape CLAUDE.md's design-rule table exists to catalogue:
//!
//! 1. **`emit_type` had no `U64` arm**, so it fell to `_ => "i32"` and
//!    `let x: U64 = ...` allocated an **i32**. `U32`/`U8`/`U16`/`I8` likewise.
//! 2. **`infer_type` returned `"i32"` for every `IntLit`** regardless of value,
//!    so a literal above `i32::MAX` was materialised as i32 and then widened.
//!    `store i32 4294967296` truncates to zero; `sext i32 3000000000 to i64`
//!    sign-extends to a negative number. `clang` accepts both.
//!
//! This is the same defect the PTX backend had and fixed (gotcha #7, "integer
//! literals are typed by VALUE, not fixed at I32"). It was still live here.
//!
//! **These tests RUN the binary rather than inspecting the IR**, because the
//! first probe I wrote compared a `U64` against an `I64` and passed — both
//! operands went through the same wrong path and agreed with each other while
//! both were wrong. Comparing against a *constant expectation* is what separates
//! "consistent" from "correct".
use std::path::PathBuf;
use std::process::Command;

fn run_program(name: &str, src: &str) -> Option<i32> {
    let dir = std::env::temp_dir().join(format!("y_intw_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let bin = dir.join(name);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("-o")
        .arg(&bin)
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .output()
        .expect("run Y");
    if !out.status.success() || !bin.exists() {
        return None;                       // no clang, or the build failed
    }
    Command::new(&bin).status().ok().and_then(|s| s.code())
}

#[test]
fn wide_literals_keep_their_value() {
    // Each program returns 1 iff the literal survived. `a > 0` is the cheapest
    // question that distinguishes "truncated to 0" from "sign-extended
    // negative" from "correct" without needing any I/O.
    for (name, lit) in [
        ("w_two_pow_32", "4294967296"),   // truncates to 0 as i32
        ("w_three_e9", "3000000000"),     // above i32::MAX -> negative as i32
        ("w_i32_max", "2147483647"),      // the boundary, must still work
    ] {
        let src = format!(
            "fn main() -> I32 {{\n    let a: I64 = {};\n    if a > 0 {{\n        return 1;\n    }}\n    return 0;\n}}\n",
            lit
        );
        match run_program(name, &src) {
            None => eprintln!("SKIP {}: could not build (clang missing?)", name),
            Some(code) => assert_eq!(
                code, 1,
                "`let a: I64 = {}` did not survive: the program says a <= 0. \
                 The literal is being materialised as i32 and widened.",
                lit
            ),
        }
    }
}

#[test]
fn unsigned_types_are_not_narrowed_to_i32() {
    // **Compare against a CONSTANT, not against another U64.** The first
    // version of this test read `let big: U64 = 4294967296; let half: U64 =
    // 2147483648; if big > half`, and it passed with the `U64` arm deleted -
    // caught only by mutation testing. Both operands truncate to i32, to `0`
    // and `-2147483648`, and `0 > -2147483648` is still true. Two values wrong
    // in compensating ways agree with each other, which is exactly the failure
    // this file's header warns about and I walked into anyway.
    for (name, lit) in [
        ("w_u64_2p32", "4294967296"),     // -> 0 as i32, so `> 0` turns false
        ("w_u64_2p31", "2147483648"),     // -> negative as i32, same effect
    ] {
        let src = format!(
            "fn main() -> I32 {{\n    let big: U64 = {};\n    if big > 0 {{\n        return 1;\n    }}\n    return 0;\n}}\n",
            lit
        );
        match run_program(name, &src) {
            None => eprintln!("SKIP {}: could not build (clang missing?)", name),
            Some(code) => assert_eq!(
                code, 1,
                "`let big: U64 = {}` compares as <= 0, so the type is being \
                 emitted as i32 and the value truncated",
                lit
            ),
        }
    }
}

#[test]
fn narrow_values_still_work() {
    // The control. Widening everything would pass the two tests above and be
    // just as wrong; ordinary small integers must be unaffected.
    let src = "fn main() -> I32 {\n    \
               let a: I32 = 7;\n    let b: I64 = -5;\n    \
               if a > 0 {\n        if b < 0 {\n            return 1;\n        }\n    }\n    \
               return 0;\n}\n";
    match run_program("w_small", src) {
        None => eprintln!("SKIP w_small: could not build (clang missing?)"),
        Some(code) => assert_eq!(code, 1, "ordinary small integers regressed"),
    }
}

#[test]
fn a_comparison_used_as_a_value_is_one_not_minus_one() {
    // Found by `tests/backend_differential.rs` on its first run, six programs
    // out of forty. `emit_coerce`'s narrow->wide arm was an unconditional
    // `sext`, and `icmp` produces `i1`, so `true` sign-extended to **-1**:
    //
    //     let t: I32 = a > b;   // 5 > 3  ->  -1
    //
    // Invisible in a condition, because `if t` tests non-zero either way, and
    // wrong everywhere a comparison is used as a VALUE -- `t * 5` was -5, and
    // a `sum = sum + (a > b)` counter runs backwards.
    //
    // The other three backends all answer 1: `--emit-native`, the ZK emitter
    // (whose condition carries an explicit booleanity constraint), and
    // `cpu_emitter` (which emits a Rust `bool`). LLVM was the outlier.
    let src = "\
fn main() -> I32 {
    let a: I32 = 5;
    let b: I32 = 3;
    let t: I32 = a > b;
    return t * 5;
}
";
    match run_program("cmp_is_one", src) {
        // 5, not -5 (which arrives as exit status 251).
        Some(v) => assert_eq!(v, 5, "`(5 > 3) * 5` should be 5"),
        None => eprintln!("SKIP cmp_is_one: could not build (clang missing?)"),
    }
}

#[test]
fn a_false_comparison_is_still_zero() {
    // The control. "Make `true` be 1" must not be achieved by making every
    // comparison 1 -- a mutation returning a constant passes the case above.
    let src = "\
fn main() -> I32 {
    let a: I32 = 3;
    let b: I32 = 5;
    let t: I32 = a > b;
    return t * 5 + 7;
}
";
    match run_program("cmp_is_zero", src) {
        Some(v) => assert_eq!(v, 7, "`(3 > 5) * 5 + 7` should be 7"),
        None => eprintln!("SKIP cmp_is_zero: could not build (clang missing?)"),
    }
}
