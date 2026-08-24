//! Two type rules the checker did not have, found by compiling Y through
//! `--emit-cpu` and handing the printed Rust to `rustc`.
//!
//! `rustc` is a type checker for the language Y transcribes into, so anything
//! Y accepts that becomes ill-typed Rust is a hole in Y — unless the emitter
//! mistranslated, which is the other thing that gate catches. Two of the three
//! failures it turned up were Y's, not the emitter's, and both reached a
//! backend:
//!
//!   * `fn f() { let x = 5; return x; }` — a value returned from a function
//!     declaring no return type. The LLVM backend emitted
//!     `sext i32 %t to void` and `ret void %t`, neither of which is legal
//!     LLVM. `--emit-llvm` wrote that file and exited 0; the default path
//!     failed inside `clang`, pointing at a line in generated IR rather than
//!     at the user's source.
//!
//!   * `if 1 { ... }` — Y has a boolean type and no implicit truthiness, so an
//!     integer condition is a type error. `tests/test_drift.ysu` shipped one.
//!
//! The second could not be caught at first: `Expr::IntLit` fell into
//! `check_expr`'s `_ => SemanticType::Unknown`, so the condition had no type
//! to check. Literals are typed now, and typed POLYMORPHICALLY — the first
//! attempt pinned them to `I32`/`F32` and made `let x: F16 = 0.0` a mismatch.
//!
//! Run with:  cargo test --test type_checker_scalar_rules

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static SALT: AtomicUsize = AtomicUsize::new(0);

fn compile(src: &str) -> (bool, String) {
    let n = SALT.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("y_tcs_{}_{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let profile = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".ysu_hw_profile");
    if profile.exists() {
        let _ = std::fs::copy(&profile, dir.join(".ysu_hw_profile"));
    }
    let path = dir.join("case.ysu");
    std::fs::write(&path, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-llvm")
        .arg("-o")
        .arg(dir.join("case.ll"))
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), text)
}

fn reject(label: &str, src: &str, phrase: &str) {
    let (ok, text) = compile(src);
    assert!(!ok, "{}: compiled clean\n{}", label, text);
    assert!(
        text.contains(phrase),
        "{}: rejected, but not for the reason under test (wanted {:?})\n{}",
        label,
        phrase,
        text
    );
}

fn accept(label: &str, src: &str) {
    let (ok, text) = compile(src);
    assert!(ok, "{}: refused a legal program\n{}", label, text);
}

#[test]
fn returning_a_value_from_a_function_with_no_return_type_is_refused() {
    reject(
        "bare return value",
        "fn f() { let x: I32 = 5; return x; }\nfn main() -> I32 { return 0; }\n",
        "declares no return type",
    );
}

#[test]
fn a_declared_return_type_still_works() {
    // The control. "Reject every `return`" passes the test above.
    accept(
        "declared return type",
        "fn f() -> I32 { let x: I32 = 5; return x; }\nfn main() -> I32 { return f(); }\n",
    );
    accept(
        "a bare return in a void function",
        "fn f() { return; }\nfn main() -> I32 { f(); return 0; }\n",
    );
}

#[test]
fn a_numeric_branch_condition_is_refused() {
    reject(
        "if on an integer",
        "fn main() -> I32 {\n    if 1 {\n        return 2;\n    }\n    return 3;\n}\n",
        "`if` condition has type",
    );
    // The same rule at the other branching site. Wiring `if` and leaving
    // `while` is the "guard consulted at one site" shape, and `while` was in
    // fact missed on the first pass.
    reject(
        "while on an integer",
        "@unsafe\nfn f() -> I32 {\n    let mut i: I32 = 0;\n    while 1 {\n        i = i + 1;\n    }\n    return i;\n}\nfn main() -> I32 { return 0; }\n",
        "`while` condition has type",
    );
}

#[test]
fn boolean_conditions_still_work() {
    // The control, in three shapes: a literal, a comparison, and a variable.
    accept(
        "if true",
        "fn main() -> I32 {\n    if true {\n        return 2;\n    }\n    return 3;\n}\n",
    );
    accept(
        "if a comparison",
        "fn main(n: I32) -> I32 {\n    if n > 0 {\n        return 2;\n    }\n    return 3;\n}\n",
    );
    accept(
        "while a comparison",
        "@unsafe\nfn f(n: I32) -> I32 {\n    let mut i: I32 = 0;\n    while i < n {\n        i = i + 1;\n    }\n    return i;\n}\nfn main() -> I32 { return f(3); }\n",
    );
}

/// Literals must stay polymorphic.
///
/// Typing `IntLit` as `I32` and `FloatLit` as `F32` outright is what makes the
/// condition rule enforceable, and it immediately made three legal bindings
/// into mismatches. A literal takes the expected type where there is one.
#[test]
fn a_literal_adopts_the_type_it_is_assigned_to() {
    accept(
        "narrower float",
        "fn main() -> I32 {\n    let x: F16 = 0.0;\n    return 0;\n}\n",
    );
    // A consistency case, not a guard: mutation shows pinning `IntLit` to
    // `I32` passes, because the `let` mismatch check is lenient between
    // integer widths. Labelled rather than deleted, so nobody reads it as
    // covering the int arm.
    accept(
        "wider int (consistency, not currently load-bearing)",
        "fn main() -> I32 {\n    let n: U64 = 1;\n    return 0;\n}\n",
    );
    accept(
        "fixed point",
        "fn main() -> I32 {\n    @ZeroDrift\n    let acc: Q32.32 = 0.0;\n    acc += 1.0;\n    return 0;\n}\n",
    );
}
