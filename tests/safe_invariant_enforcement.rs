//! `@invariant` inside `@safe` is a proof obligation, and an unverified one
//! must fail the build.
//!
//! It did not used to. When the SMT solver could not be spawned, the type
//! checker printed `[Warning] SMT Solver execution failed` to stdout and
//! continued, so on any machine without z3 - the default - every invariant in
//! every `@safe` block was accepted unchecked. This compiled cleanly and
//! reported "Compilation Successful!":
//!
//! ```text
//! @safe {
//!     @invariant(i > 1000)     // never true for i in 0..10
//!     for i in 0..10 { ... }
//! }
//! ```
//!
//! Two things made it hard to notice. The message went to stdout among a lot of
//! other build chatter, and the search path for the solver was three entries
//! long - so a machine with a perfectly good `venv/bin/z3` (this repo has one)
//! reported it as missing and waved everything through.
//!
//! These tests drive the real `Y` binary as a subprocess rather than calling
//! `TypeChecker` directly, because the behaviour under test depends on the
//! environment and mutating `std::env` from a test races every other test in
//! the process.
//!
//! Run with:  cargo test --test safe_invariant_enforcement

use std::path::PathBuf;
use std::process::Command;

/// A `@safe` loop whose invariant is false: `i` ranges over `0..10`.
const FALSE_INVARIANT: &str = r#"
fn main() {
    @safe {
        let x: I32 = 0;
        @invariant(i > 1000)
        for i in 0..10 {
            print_int(i);
        }
    }
}
"#;

/// The same shape, with an invariant that actually holds.
const TRUE_INVARIANT: &str = r#"
fn main() {
    @safe {
        let x: I32 = 0;
        @invariant(i >= 0)
        for i in 0..10 {
            print_int(i);
        }
    }
}
"#;

fn write_source(name: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("y_safe_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, body).expect("write source");
    path
}

/// Runs the compiler on `src`. `solver_visible` false strips the environment
/// the solver could be found through.
fn compile(src: &PathBuf, solver_visible: bool, allow_unverified: bool) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(src);
    cmd.env_remove("Y_Z3_PATH");
    cmd.env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS");
    if !solver_visible {
        // No `z3` on PATH, and no `$HOME/.local/bin/z3`. The relative
        // candidates resolve against the working directory, so run somewhere
        // that has no `venv/` or `z3/` in it.
        cmd.env("PATH", "/nonexistent-path");
        cmd.env("HOME", "/nonexistent-home");
        cmd.current_dir(src.parent().expect("temp dir"));
    }
    if allow_unverified {
        cmd.env("Y_ALLOW_UNVERIFIED_INVARIANTS", "1");
    }
    let out = cmd.output().expect("run Y");
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Without a solver, an invariant cannot be discharged - so the build fails.
#[test]
fn unverifiable_invariant_fails_the_build() {
    let src = write_source("false_inv", FALSE_INVARIANT);
    let out = compile(&src, false, false);
    assert!(
        out.contains("Could not verify invariant"),
        "a @safe invariant that could not be checked must be an error, not a warning.\n{}",
        out
    );
    assert!(
        !out.contains("Compilation Successful"),
        "compilation must not succeed with an undischarged proof obligation.\n{}",
        out
    );
    // The message has to tell the user how to fix it, both ways.
    assert!(out.contains("Y_Z3_PATH"), "error should mention Y_Z3_PATH:\n{}", out);
    assert!(
        out.contains("Y_ALLOW_UNVERIFIED_INVARIANTS"),
        "error should name the opt-out:\n{}",
        out
    );
}

/// The escape hatch restores the old behaviour, loudly.
#[test]
fn opt_out_downgrades_to_a_warning() {
    let src = write_source("optout", FALSE_INVARIANT);
    let out = compile(&src, false, true);
    assert!(
        out.contains("was NOT verified"),
        "opt-out should still say plainly that nothing was proved.\n{}",
        out
    );
    assert!(
        !out.contains("Could not verify invariant"),
        "opt-out should not also raise the error.\n{}",
        out
    );
}

/// With a solver available, the checking is real in both directions.
///
/// Skipped when no z3 can be found, in the same spirit as the `ptxas` gate on
/// the PTX emitter - but note this test is the only one that proves the
/// verification does anything at all, so a CI without z3 is CI that is not
/// testing `@safe`'s central claim.
#[test]
fn with_a_solver_false_invariants_are_caught_and_true_ones_pass() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let z3 = ["venv/bin/z3", ".venv/bin/z3", "z3/build/z3"]
        .iter()
        .map(|p| repo.join(p))
        .find(|p| p.exists())
        .or_else(|| {
            std::env::var("PATH").ok().and_then(|paths| {
                std::env::split_paths(&paths)
                    .map(|d| d.join("z3"))
                    .find(|p| p.exists())
            })
        });
    let Some(z3) = z3 else {
        eprintln!("skipping: no z3 binary found");
        return;
    };

    let bad = write_source("smt_false", FALSE_INVARIANT);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&bad)
        .env("Y_Z3_PATH", &z3)
        .env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS")
        .output()
        .expect("run Y");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.contains("SMT Safety Verification Failed"),
        "z3 must reject `i > 1000` on a 0..10 loop.\n{}",
        out
    );

    let good = write_source("smt_true", TRUE_INVARIANT);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&good)
        .env("Y_Z3_PATH", &z3)
        .env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS")
        .output()
        .expect("run Y");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.contains("SMT Safety Verification Failed"),
        "`i >= 0` holds on a 0..10 loop and must be accepted.\n{}",
        out
    );
}
