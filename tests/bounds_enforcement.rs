//! `@bounds` and static index checking, tested adversarially.
//!
//! Unlike the SMT encoding, `eval_interval` was written fail-closed from the
//! start: every unhandled operator and expression returns `None`, division by
//! an interval spanning zero returns `None`, and `None` means "this index has
//! no statically provable range", which is an error. Most of the battery below
//! therefore passed the first time it was run.
//!
//! One case did not. When a loop's bounds could not be evaluated the checker
//! fabricated `Interval { min: 0, max: 999999 }` for the loop variable - two
//! facts asserted rather than derived. The `max` half was harmless in practice,
//! since 999999 trips the overflow check for any normal array, but the `min`
//! half claimed the index was non-negative with nothing to back it. Over an
//! array bigger than 999999 both halves slipped through at once, and
//! `for i in n..3` with `n` an unconstrained parameter compiled clean.
//!
//! Run with:  cargo test --test bounds_enforcement

use std::path::PathBuf;
use std::process::Command;

/// Compiles `src` and returns the compiler's combined output.
///
/// Runs with a solver when one can be found, so that `@invariant` obligations
/// inside these programs do not fail for an unrelated reason and mask the
/// bounds behaviour under test.
fn compile(name: &str, src: &str) -> String {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("y_bounds_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&path).current_dir(&repo);
    cmd.env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS");
    let z3 = ["venv/bin/z3", ".venv/bin/z3", "z3/build/z3"]
        .iter()
        .map(|p| repo.join(p))
        .find(|p| p.exists());
    if let Some(z3) = z3 {
        cmd.env("Y_Z3_PATH", z3);
    }
    let out = cmd.output().expect("run Y");
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn assert_rejected(name: &str, src: &str, why: &str) {
    let out = compile(name, src);
    assert!(
        !out.contains("Compilation Successful"),
        "{} was accepted but must be rejected ({}).\n{}",
        name,
        why,
        out
    );
}

/// A constant index past the end.
#[test]
fn constant_out_of_range_index_is_rejected() {
    assert_rejected(
        "const_oob",
        "fn main() {\n    @safe {\n        let arr: [I32; 5] = {};\n        let v: I32 = arr[10];\n    }\n}\n",
        "10 is past the end of a 5-element array",
    );
}

/// A loop that walks past the end.
#[test]
fn loop_index_past_the_end_is_rejected() {
    assert_rejected(
        "loop_oob",
        "fn main() {\n    @safe {\n        let arr: [I32; 5] = {};\n        @invariant(i >= 0)\n        for i in 0..10 {\n            let v: I32 = arr[i];\n        }\n    }\n}\n",
        "i reaches 9 on a 5-element array",
    );
}

/// A negative index, reached by arithmetic rather than written literally.
#[test]
fn negative_index_is_rejected() {
    assert_rejected(
        "neg_index",
        "fn main() {\n    @safe {\n        let arr: [I32; 5] = {};\n        let k: I32 = 0;\n        k = k - 10;\n        let v: I32 = arr[k];\n    }\n}\n",
        "k is -10",
    );
}

/// `@bounds` must actually constrain the initialiser.
#[test]
fn initialiser_outside_declared_bounds_is_rejected() {
    assert_rejected(
        "bounds_violated",
        "fn main() {\n    @safe {\n        @bounds(min=0, max=15)\n        let x: I32 = 100;\n    }\n}\n",
        "100 is outside [0, 15]",
    );
}

/// An index built with an operator interval arithmetic does not model must be
/// refused, not assumed in range.
///
/// `%` has no arm in `eval_interval`, so the index has no provable range. The
/// correct answer is to say so.
#[test]
fn index_through_an_unmodelled_operator_is_refused() {
    let out = compile(
        "unmodelled_index",
        "fn main() {\n    @safe {\n        let arr: [I32; 4] = {};\n        @invariant(i >= 0)\n        for i in 0..4 {\n            let v: I32 = arr[i % 100];\n        }\n    }\n}\n",
    );
    assert!(
        !out.contains("Compilation Successful"),
        "an index the checker cannot bound must not be accepted:\n{}",
        out
    );
    assert!(
        out.contains("no statically provable bounds"),
        "the error should say the index could not be bounded:\n{}",
        out
    );
}

/// The regression: a loop whose bounds are unknown must not get a made-up range.
///
/// `n` is an unconstrained parameter, so nothing establishes that `i` is
/// non-negative. The array is deliberately larger than the old fabricated
/// `max` of 999999 - that is what let both halves of the invented interval slip
/// past at once, and this program used to report "Compilation Successful!".
#[test]
fn unknown_loop_bounds_do_not_get_a_fabricated_interval() {
    assert_rejected(
        "fabricated_interval",
        "fn f(n: I32) {\n    @safe {\n        let arr: [I32; 2000000] = {};\n        @invariant(i >= 0)\n        for i in n..3 {\n            let v: I32 = arr[i];\n        }\n    }\n}\n",
        "n is unconstrained, so i has no proven lower bound",
    );

    // Same shape on a small array, where the old fabricated `max` happened to
    // catch it for the wrong reason. It must still be rejected.
    assert_rejected(
        "fabricated_interval_small",
        "fn f(n: I32) {\n    @safe {\n        let arr: [I32; 5] = {};\n        @invariant(i >= 0)\n        for i in n..3 {\n            let v: I32 = arr[i];\n        }\n    }\n}\n",
        "n is unconstrained",
    );
}

/// And the other half of the requirement: a provably safe access still compiles.
///
/// Rejecting everything is trivially sound and useless. This is also the case
/// that caught an over-eager version of the `@invariant` fail-closed rule - the
/// `arr[i]` in the body is not expressible in the SMT encoding, and refusing it
/// there rather than treating the value as unknown made valid loops unbuildable.
#[test]
fn provably_safe_access_still_compiles() {
    let out = compile(
        "safe_access",
        "fn main() {\n    @safe {\n        let arr: [I32; 8] = {};\n        @invariant(i >= 0)\n        for i in 0..8 {\n            let v: I32 = arr[i];\n        }\n    }\n}\n",
    );
    assert!(
        out.contains("Compilation Successful"),
        "an in-range access over a statically known loop must compile:\n{}",
        out
    );
}
