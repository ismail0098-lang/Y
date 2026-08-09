//! `linear_tracker.rs`, tested adversarially. It previously had no tests at all.
//!
//! The claim under test is the one in the module header: an async transfer
//! obligation is consumed *exactly once*. That is a statement about executions,
//! and the tracker was checking source lines - it set a `consumed` flag the
//! first time the walk passed a `pipe.wait(t)`, without asking whether that
//! statement runs, or how many times. Two shapes went through clean:
//!
//! ```text
//! let t = cp_async(A, B, 16);
//! if n { pipe.wait(t); }           // awaited on one path out of two
//!
//! let t = cp_async(A, B, 16);
//! for i in 0..4 { pipe.wait(t); }  // one copy, four awaits
//! ```
//!
//! Both are exactly what the tracker exists to reject, and both are invisible
//! to a flag: the first is the missing-await race the whole mechanism is for,
//! the second is the double-consume it *does* reject when the two waits are
//! written consecutively.
//!
//! Half of this file is therefore negative cases and half is positive ones. The
//! positive half is not filler - a tracker that rejects every program satisfies
//! "never consumed twice" perfectly and is useless, and the conditional fix in
//! particular is one over-approximation away from banning `pipe.wait` inside
//! any loop, including the correct copy-and-await-per-iteration shape that real
//! pipelined kernels are built from.
//!
//! Driven through the real binary rather than the `LinearTracker` API, because
//! the tracker being right is not the property that matters - the property that
//! matters is that the type checker drives it correctly, and an earlier version
//! of this bug was entirely in the wiring.
//!
//! Run with:  cargo test --test linear_tracker_enforcement

use std::path::PathBuf;
use std::process::Command;

fn compile(name: &str, body: &str) -> String {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("y_lin_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(
        &path,
        format!(
            "kernel {}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, N: I32) {{\n{}\n}}\n\nfn main() {{\n}}\n",
            name, body
        ),
    )
    .expect("write source");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&path).arg("--emit-ptx").current_dir(&repo);
    // Keep a solver in reach so `@invariant` obligations in the loop cases do
    // not fail for an unrelated reason and mask what is under test.
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

fn assert_rejected(name: &str, body: &str, why: &str) {
    let out = compile(name, body);
    assert!(
        !out.contains("Compilation Successful"),
        "{} was accepted but must be rejected ({}).\n{}",
        name,
        why,
        out
    );
    assert!(
        out.contains("Linear Type Error"),
        "{} was rejected, but not by the linear tracker - so the guarantee under \
         test is not what stopped it ({}).\n{}",
        name,
        why,
        out
    );
}

fn assert_accepted(name: &str, body: &str, why: &str) {
    let out = compile(name, body);
    assert!(
        out.contains("Compilation Successful"),
        "{} must compile ({}).\n{}",
        name,
        why,
        out
    );
}

/// A transfer that is never awaited.
#[test]
fn unconsumed_obligation_is_rejected() {
    assert_rejected(
        "lt_unconsumed",
        "    let tok: AsyncToken = cp_async(A, B, 16);",
        "the copy is never awaited, so its destination is read in flight",
    );
}

/// A transfer awaited twice.
#[test]
fn double_consume_is_rejected() {
    assert_rejected(
        "lt_double",
        "    let tok: AsyncToken = cp_async(A, B, 16);\n    pipe.wait(tok);\n    pipe.wait(tok);",
        "the second wait awaits a group that was already retired",
    );
}

/// A transfer dropped as a bare expression statement, never bound.
#[test]
fn unbound_obligation_is_rejected() {
    let out = compile("lt_unbound", "    cp_async(A, B, 16);");
    assert!(
        !out.contains("Compilation Successful"),
        "an unbound transfer obligation cannot ever be awaited:\n{}",
        out
    );
}

/// Rebinding over a pending obligation.
#[test]
fn shadowing_a_pending_obligation_is_rejected() {
    assert_rejected(
        "lt_reassign",
        "    let tok: AsyncToken = cp_async(A, B, 16);\n    let tok: AsyncToken = cp_async(A, B, 16);\n    pipe.wait(tok);",
        "the first transfer becomes unreachable and can never be awaited",
    );
}

/// An obligation created inside a block and left pending at the block's end.
#[test]
fn obligation_pending_at_scope_exit_is_rejected() {
    assert_rejected(
        "lt_scope",
        "    if N {\n        let tok: AsyncToken = cp_async(A, B, 16);\n    }",
        "the obligation goes out of scope still pending",
    );
}

/// The regression: an await reachable on only some paths.
///
/// This compiled clean. `if` did not change the tracker's state, so a wait
/// inside the branch looked identical to a wait after it. On the paths where
/// `N` is zero the `cp.async` is never awaited and the destination is read
/// while the copy is still in flight - the exact race the tracker is for,
/// admitted by the tracker.
#[test]
fn conditionally_awaited_transfer_is_rejected() {
    assert_rejected(
        "lt_cond",
        "    let tok: AsyncToken = cp_async(A, B, 16);\n    if N {\n        pipe.wait(tok);\n    }",
        "the branch may not be taken, leaving the transfer unawaited",
    );

    // An `else` arm does not rescue it: awaiting on both arms is two consumes,
    // and awaiting on one is still conditional.
    assert_rejected(
        "lt_cond_else",
        "    let tok: AsyncToken = cp_async(A, B, 16);\n    if N {\n        pipe.wait(tok);\n    } else {\n        pipe.wait(tok);\n    }",
        "one obligation cannot be consumed by two branch bodies",
    );
}

/// The other regression: one transfer, N awaits.
///
/// Written flat this is the double-consume case above, which the tracker has
/// always caught. Wrapped in a loop header it compiled clean, because the
/// tracker saw one `pipe.wait` statement and set one flag.
#[test]
fn transfer_awaited_once_per_loop_iteration_is_rejected() {
    assert_rejected(
        "lt_loop",
        "    let tok: AsyncToken = cp_async(A, B, 16);\n    @invariant(i >= 0)\n    for i in 0..4 {\n        pipe.wait(tok);\n    }",
        "one cp.async would be awaited four times",
    );
}

/// The plain correct shape must still compile.
#[test]
fn copy_then_await_compiles() {
    assert_accepted(
        "lt_ok_flat",
        "    let tok: AsyncToken = cp_async(A, B, 16);\n    pipe.wait(tok);",
        "this is the shape the whole mechanism exists to permit",
    );
}

/// And the pipelined shape: copy and await in the same iteration.
///
/// This is the case that makes the conditional and loop rules worth stating
/// carefully rather than banning `pipe.wait` under a loop outright. Here the
/// obligation is created *and* consumed at the same nesting depth, so each
/// iteration's copy is awaited exactly once - which is precisely how a
/// multi-stage `cp.async` pipeline is written.
#[test]
fn copy_and_await_within_the_same_iteration_compiles() {
    assert_accepted(
        "lt_ok_loop",
        "    @invariant(i >= 0)\n    for i in 0..4 {\n        let tok: AsyncToken = cp_async(A, B, 16);\n        pipe.wait(tok);\n    }",
        "each iteration issues its own transfer and awaits that one",
    );
}

/// Likewise inside a branch, when the copy is in the same branch.
#[test]
fn copy_and_await_within_the_same_branch_compiles() {
    assert_accepted(
        "lt_ok_branch",
        "    if N {\n        let tok: AsyncToken = cp_async(A, B, 16);\n        pipe.wait(tok);\n    }",
        "on the paths where the copy happens, the await happens too",
    );
}
