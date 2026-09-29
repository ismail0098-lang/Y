//! `while` through the LLVM backend, RUN rather than read.
//!
//! `while` had been in the lexer, the parser, the AST and four emitters since
//! the language existed, and until now no `while` loop in safe Y could reach a
//! backend at all: `check_stmt` cleared the interval of every body-assigned
//! variable before verifying the loop, and the INITIATION obligation is about
//! the state on entry — so every `@invariant` naming such a variable failed,
//! and a `while`'s induction variable always is one. An `@invariant` is
//! mandatory outside `unsafe`.
//!
//! So this lowering has never been exercised by a passing compile. It turned
//! out to be correct, which is worth pinning rather than assuming: the PTX
//! backend's `while` arm, checked in the same hour, did not exist and emitted
//! nothing at all (`tests/ptx_control_flow.rs`).
//!
//! Every case RUNS the produced binary. Reading the IR would pass on a loop
//! that never terminates or never executes.
//!
//! Run with:  cargo test --release --test llvm_control_flow

use std::path::PathBuf;
use std::process::Command;

/// Compile with the default (LLVM) backend and run.
///
/// `None` means only "no clang on this machine". A compiler REFUSAL panics
/// instead of skipping: a harness that reports "could not build" for both is
/// a test that cannot fail, and the first version of this file skipped a case
/// the type checker was rejecting outright.
fn run_program(name: &str, src: &str) -> Option<i32> {
    let dir = std::env::temp_dir().join(format!("y_llcf_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let bin = dir.join(name);
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&path).arg("-o").arg(&bin).current_dir(&repo);
    // Every loop below carries an `@invariant`; without a solver they would
    // fail for a reason that is not what is under test.
    if let Some(z3) = ["venv/bin/z3", ".venv/bin/z3", "z3/build/z3"]
        .iter()
        .map(|p| repo.join(p))
        .find(|p| p.exists())
    {
        cmd.env("Y_Z3_PATH", z3);
    }
    let out = cmd.output().expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if text.contains("semantic errors") || text.contains("[Error]") {
        panic!("`{}` was REFUSED by the compiler, not merely unbuildable:\n{}", name, text);
    }
    if !bin.exists() {
        // Distinguish "this machine has no clang" from "clang REJECTED the
        // IR". Both used to return `None` and skip, so a mutation that
        // removed a basic block's terminator - invalid IR - made all three
        // cases pass vacuously. Found by mutation, not by review.
        if Command::new("clang").arg("--version").output().is_ok() {
            panic!(
                "`{}` produced no binary although clang is installed, so the \
                 emitted IR was rejected:\n{}",
                name, text
            );
        }
        return None;
    }
    // A dropped loop body means the induction variable never advances, so the
    // program runs forever. Without a deadline that is a hung test rather than
    // a failing one - also found by mutation.
    let mut child = Command::new(&bin).spawn().expect("spawn the built binary");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match child.try_wait().expect("wait on the built binary") {
            Some(status) => return status.code(),
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                panic!("`{}` did not terminate within 10s - the loop never exits", name);
            }
            None => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    }
}

#[test]
fn a_while_loop_runs_the_right_number_of_times() {
    // 5 iterations of `acc += 3`.
    let src = "fn main() -> I32 {\n    \
               let i: I32 = 0;\n    let acc: I32 = 0;\n    \
               @invariant(acc >= 0)\n    \
               while i < 5 {\n        acc = acc + 3;\n        i = i + 1;\n    }\n    \
               return acc;\n}\n";
    match run_program("llcf_while", src) {
        None => eprintln!("SKIP llcf_while: could not build (clang missing?)"),
        Some(code) => assert_eq!(
            code, 15,
            "a `while` loop did not run 5 times. 0 means the body was dropped; \
             any other value means the exit test is wrong."
        ),
    }
}

#[test]
fn a_while_whose_condition_is_false_runs_zero_times() {
    // The control for the one above: a loop that always runs its body once
    // would pass a "did the body execute" test and be a `do`-loop.
    let src = "fn main() -> I32 {\n    \
               let i: I32 = 9;\n    let acc: I32 = 7;\n    \
               @invariant(acc >= 0)\n    \
               while i < 5 {\n        acc = acc + 3;\n        i = i + 1;\n    }\n    \
               return acc;\n}\n";
    match run_program("llcf_while_zero", src) {
        None => eprintln!("SKIP llcf_while_zero: could not build (clang missing?)"),
        Some(code) => assert_eq!(
            code, 7,
            "the condition is false on entry, so the body must not run at all"
        ),
    }
}

/// A `while` inside a `for`, to check the label numbering does not collide and
/// the inner loop is re-entered each outer pass.
///
/// The invariants here are each about their own loop's variable, deliberately.
/// An invariant on the shared accumulator is refused, and correctly so: the
/// verifier does not assume the OUTER invariant while checking the body, so
/// `acc` is unconstrained at the inner loop's entry — and the outer
/// preservation check havocs whatever a nested loop assigns. Both are
/// pre-existing over-approximations, documented in CLAUDE.md, and neither is
/// what this file is about. The first version of this test used that fixture
/// and the harness reported it as "could not build".
#[test]
fn a_while_nested_in_a_for_still_terminates() {
    // 3 outer iterations, each running the inner loop twice.
    let src = "fn main() -> I32 {\n    \
               let acc: I32 = 0;\n    \
               @invariant(k >= 0)\n    \
               for k in 0..3 {\n        \
               let j: I32 = 0;\n        \
               @invariant(j >= 0)\n        \
               while j < 2 {\n            acc = acc + 1;\n            j = j + 1;\n        }\n    }\n    \
               return acc;\n}\n";
    match run_program("llcf_nested", src) {
        None => eprintln!("SKIP llcf_nested: could not build (clang missing?)"),
        Some(code) => assert_eq!(code, 6, "3 outer iterations x 2 inner must give 6"),
    }
}
