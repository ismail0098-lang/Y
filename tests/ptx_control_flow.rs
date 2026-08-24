//! `while`, `break`, `match` and `return` in a PTX kernel.
//!
//! `emit_stmt` handled fourteen `Stmt` variants and ended in `_ => {}`, so a
//! `while` loop emitted NOTHING. Measured on the real binary:
//!
//! ```text
//! kernel k(A: GlobalMemory<F32>, N: I32) {
//!     let i: I32 = 0;
//!     @invariant(i >= 0)
//!     while i < N { i = i + 1; }
//!     store(A, 0, i);
//! }
//! ```
//!
//! emitted `mov %r1, 0; mov %r2, 0; st.global.u32 [%rd0], %r2;` — the loop was
//! gone and the kernel stored 0 whatever `N` was. **No `ptxas` gate can catch
//! that**: the module is simply a shorter kernel and assembles perfectly, which
//! is the limit gotcha #8 states. So this file asserts on the PRESENCE and
//! ORDER of the loop's instructions as well as on assembly.
//!
//! It went unnoticed because it was unreachable: `check_stmt` clears the
//! interval of every variable a loop body assigns before verifying the loop's
//! invariant, and the INITIATION obligation is about the state on entry — so
//! every `@invariant` naming a body-assigned variable failed, an `@invariant`
//! is mandatory outside `unsafe`, and a `while`'s induction variable is always
//! body-assigned. No `while` loop in safe Y could reach a backend at all.
//! `tests/math.ysu` and `tests/safe_test.ysu` were both refused by it.
//!
//! Run with:  cargo test --release --test ptx_control_flow

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("Y")
}

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Compile `src` with `--emit-ptx`. `Ok(ptx)` or `Err(diagnostic)`.
fn emit(name: &str, src: &str) -> Result<String, String> {
    let dir = std::env::temp_dir().join(format!("y_ptxcf_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let mut cmd = Command::new(bin());
    cmd.arg(&path).arg("--emit-ptx").current_dir(repo());
    // Keep a solver in reach: every loop here carries an `@invariant`, and a
    // missing z3 would fail these for a reason that is not what they test.
    if let Some(z3) = ["venv/bin/z3", ".venv/bin/z3", "z3/build/z3"]
        .iter()
        .map(|p| repo().join(p))
        .find(|p| p.exists())
    {
        cmd.env("Y_Z3_PATH", z3);
    }
    let out = cmd.output().expect("failed to run the Y binary");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !out.status.success() {
        return Err(text);
    }
    Ok(std::fs::read_to_string(path.with_extension("ptx")).expect("no .ptx was written"))
}

const WHILE_KERNEL: &str = "kernel k(A: GlobalMemory<F32>, N: I32) {\n\
    \x20   let i: I32 = 0;\n\
    \x20   @invariant(i >= 0)\n\
    \x20   while i < N {\n\
    \x20       i = i + 1;\n\
    \x20   }\n\
    \x20   store(A, 0, i);\n\
}\n";

/// The loop must actually be in the output, in the right order.
///
/// Assembly alone is satisfied by the empty lowering, so these assertions are
/// what make the test mean anything — the same reasoning as
/// `async_copy_is_committed_and_awaited`.
#[test]
fn a_while_loop_emits_a_loop() {
    let ptx = emit("ptxcf_while", WHILE_KERNEL).expect("the while kernel must compile");
    let head = ptx
        .find("$WHILE_START")
        .unwrap_or_else(|| panic!("no loop header label was emitted:\n{}", ptx));
    let back = ptx
        .rfind("bra $WHILE_START")
        .unwrap_or_else(|| panic!("no backward branch - the body cannot repeat:\n{}", ptx));
    let exit = ptx
        .find("bra $WHILE_END")
        .unwrap_or_else(|| panic!("no exit branch - the loop cannot terminate:\n{}", ptx));
    assert!(
        head < exit && exit < back,
        "the loop's parts are out of order (header {}, exit test {}, back-edge {})",
        head,
        exit,
        back
    );
    assert!(
        ptx.contains("add.s32"),
        "the loop BODY was dropped; only its skeleton was emitted:\n{}",
        ptx
    );
}

/// ...and the result must be a legal module. A hand-built branch is exactly
/// the kind of thing that assembles or does not.
#[test]
fn the_emitted_loop_assembles() {
    let ptx = emit("ptxcf_while_asm", WHILE_KERNEL).expect("the while kernel must compile");
    let dir = std::env::temp_dir().join(format!("y_ptxcf_asm_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("k.ptx");
    std::fs::write(&f, &ptx).unwrap();
    let out = match Command::new("ptxas").arg("-arch=sm_89").arg(&f).arg("-o").arg(dir.join("k.o")).output() {
        Ok(o) => o,
        Err(_) => {
            eprintln!("skipping: no ptxas on this machine");
            return;
        }
    };
    assert!(
        out.status.success(),
        "ptxas rejected the emitted loop:\n{}\n--- ptx ---\n{}",
        String::from_utf8_lossy(&out.stderr),
        ptx
    );
}

/// Statements with no lowering here are refused BY NAME, each asserting on its
/// own phrase so a fixture stopped by an earlier pass fails instead of passing.
#[test]
fn statements_without_a_lowering_are_refused() {
    let cases = [
        (
            "ptxcf_match",
            "kernel k(A: GlobalMemory<F32>, N: I32) {\n    match N {\n        _ => N\n    }\n    store(A, 0, 1.0);\n}\n",
            "`match`",
        ),
        (
            // The PTX backend refuses this too ("no return value"), but the
            // TYPE CHECKER gets there first now: a kernel declares no return
            // type, so `return N` is the same error as `fn f() { return x; }`
            // and is reported against the source line rather than by a
            // backend. That makes the backend's arm a CONFIRMATION rather than
            // a guard - the same standing as `break` below - and this case
            // asserts on the error that actually fires, per this file's own
            // rule that a fixture stopped by an earlier pass must fail rather
            // than pass.
            "ptxcf_ret_value",
            "kernel k(A: GlobalMemory<F32>, N: I32) {\n    store(A, 0, 1.0);\n    return N;\n}\n",
            "declares no return type",
        ),
    ];
    for (name, src, phrase) in cases {
        match emit(name, src) {
            Ok(ptx) => panic!(
                "`{}` compiled. It emits nothing, so the kernel silently computes \
                 a different program:\n{}",
                name, ptx
            ),
            Err(diag) => assert!(
                diag.contains(phrase),
                "`{}` was refused, but not for the reason under test. Wanted {:?}, got:\n{}",
                name,
                phrase,
                diag
            ),
        }
    }
}

/// `break` is refused too, but by an EARLIER pass, so the `Stmt::Break` arm in
/// the emitter is a confirmation rather than a guard.
///
/// `@invariant` is mandatory on any loop outside `unsafe`, and the invariant
/// verifier refuses a body containing a `break` because it does not model one.
/// So a `break` cannot reach the backend at all today. The arm is kept anyway
/// -- this backend must not be correct only because a different pass happens
/// to stop something -- and this test asserts the end-to-end property (a
/// `break` never silently compiles) rather than pretending to pin the arm.
///
/// Checked while writing it: the diagnostic used to read "a `unsupported`
/// statement", because `stmt_kind` ended in `_ => "unsupported"` and named
/// neither the construct nor a reason. It is exhaustive now.
#[test]
fn a_break_never_silently_compiles() {
    let src = "kernel k(A: GlobalMemory<F32>, N: I32) {\n    @invariant(i >= 0)\n    for i in 0..4 {\n        break;\n    }\n    store(A, 0, 1.0);\n}\n";
    match emit("ptxcf_break", src) {
        Ok(ptx) => panic!("a `break` compiled; the emitter drops it:\n{}", ptx),
        Err(diag) => assert!(
            diag.contains("break"),
            "a `break` was refused without naming itself, which leaves a user \
             nothing to act on:\n{}",
            diag
        ),
    }
}

/// The control. Refusing every kernel would satisfy the cases above and delete
/// the backend, so the ordinary shapes must still compile — including a bare
/// `return;`, which is a legal early exit and now emits `ret;` instead of
/// nothing.
#[test]
fn ordinary_kernels_still_compile() {
    let plain = "kernel k(A: GlobalMemory<F32>, N: I32) {\n    @invariant(i >= 0)\n    for i in 0..4 {\n        store(A, i, 1.0);\n    }\n}\n";
    assert!(emit("ptxcf_ok_for", plain).is_ok(), "a plain `for` kernel must still compile");

    let bare_ret = "kernel k(A: GlobalMemory<F32>, N: I32) {\n    store(A, 0, 1.0);\n    return;\n}\n";
    let ptx = emit("ptxcf_ok_ret", bare_ret).expect("a bare `return;` must still compile");
    assert!(
        ptx.contains("ret;"),
        "a bare `return;` is an early exit and emitted nothing at all:\n{}",
        ptx
    );
}
