//! References in the front end: `*x` on a non-reference, and the `mut` in
//! `&mut x`.
//!
//! Two bugs, found together because fixing the first exposed the second.
//!
//! 1. **`*x` on anything that is not a reference was accepted.**
//!    `type_checker`'s `UnaryOp::Deref` arm returned `Unknown` for every
//!    non-`Reference` operand, and `Unknown` is what the `let` arm reads as
//!    "adopt the annotation" and the assignment arm exempts from the mismatch
//!    check outright. So `let mut p: I32 = 0; *p = 0;` type-checked, and the
//!    LLVM backend lowered it to `store i32 0, ptr %_t1` with `%_t1` an `i32` -
//!    invalid IR, under "Compilation Successful!" and exit 0.
//!
//!    Y has no raw pointer type, only `&T`, so there is no pointee type an
//!    `inttoptr` lowering could take a width from. Guessing one is the
//!    substitution the design rule forbids, so this is refused by name.
//!
//! 2. **The parser threw away the `mut` in `&mut x`.** It matched the token
//!    into a binding called `_mutable` and dropped it, so `&mut x` and `&x`
//!    were the same AST node. That was wrong in BOTH directions, which is why
//!    it survived: `type_checker` typed every borrow as immutable, so
//!    `let r: &mut I32 = &mut x;` - correct code - was REJECTED as a type
//!    mismatch, while `let r: &mut I32 = &x;` was accepted; and `cpu_emitter`
//!    emitted `(&x)` for both, so rustc rejected any call whose callee wanted
//!    `&mut i32`.
//!
//!    `UnaryOp::Ref` carries `mutable` now, alongside the `mutable` that
//!    `Type::Reference` and `SemanticType::Reference` already had.
//!
//! Bug 2 was found by bug 1's fix: rewriting `tests/math.ysu` to the legitimate
//! `&mut I32` spelling made `no_emitted_blob_is_invalid_rust` fail, with rustc
//! naming the mismatch. That test's allowlist is empty as a result.
use std::path::PathBuf;
use std::process::Command;

/// Compile `src` through the real binary; return (exit-ok, combined output).
fn compile(name: &str, src: &str, flag: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("y_reftyp_{}_{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg(flag)
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

/// `@unsafe` does not license it. There is no pointee type to derive a width
/// from, so there is nothing for `unsafe` to make the programmer responsible
/// for - it is a type error, not an unchecked operation.
#[test]
fn dereferencing_an_integer_is_refused_even_in_unsafe() {
    let (ok, text) = compile(
        "derefint",
        "@unsafe\nfn f() {\n    let mut p: I32 = 0;\n    *p = 0;\n}\n\nfn main() -> I32 {\n    f();\n    return 0;\n}\n",
        "--emit-llvm",
    );
    assert!(!ok, "dereferencing an I32 compiled and exited 0:\n{}", text);
    assert!(
        text.contains("Cannot dereference") && text.contains("I32"),
        "refused, but without naming the operator and its type:\n{}",
        text
    );
}

/// The control that stops "refuse every deref" from passing. Without it the
/// test above is satisfied by deleting the feature - the same shape as
/// `ordinary_loop_bodies_still_verify`.
#[test]
fn dereferencing_a_real_reference_is_still_allowed() {
    let (ok, text) = compile(
        "derefref",
        "@unsafe\nfn g(r: &mut I32) {\n    *r = 7;\n}\n\nfn main() -> I32 {\n    let mut x: I32 = 1;\n    g(&mut x);\n    return x;\n}\n",
        "--emit-llvm",
    );
    assert!(ok, "dereferencing a `&mut I32` was refused:\n{}", text);
}

/// `&mut x` must satisfy a `&mut T` annotation. This is the direction that was
/// refusing CORRECT programs, so it is the one that proves the parser is not
/// simply dropping `mut` again.
#[test]
fn a_mut_borrow_satisfies_a_mut_reference_annotation() {
    let (ok, text) = compile(
        "mutok",
        "fn main() -> I32 {\n    let mut x: I32 = 1;\n    let r: &mut I32 = &mut x;\n    let s: &I32 = &x;\n    return 0;\n}\n",
        "--emit-llvm",
    );
    assert!(
        ok,
        "`let r: &mut I32 = &mut x;` was rejected as a type mismatch:\n{}",
        text
    );
}

/// And the other direction: a shared borrow must NOT satisfy `&mut T`. A fix
/// that hardcodes `mutable: true` passes the test above and fails this one.
#[test]
fn a_shared_borrow_does_not_satisfy_a_mut_reference_annotation() {
    let (ok, text) = compile(
        "mutbad",
        "fn main() -> I32 {\n    let mut x: I32 = 1;\n    let r: &mut I32 = &x;\n    return 0;\n}\n",
        "--emit-llvm",
    );
    assert!(
        !ok,
        "`let r: &mut I32 = &x;` compiled and exited 0:\n{}",
        text
    );
    assert!(
        text.contains("Type mismatch"),
        "refused, but not as a type mismatch:\n{}",
        text
    );
}

/// The backend half. `cpu_emitter` transcribes to Rust, where the distinction
/// is load-bearing: `(&x)` passed to a `&mut i32` parameter is `error[E0308]`.
/// Both spellings are asserted, so emitting `&mut` unconditionally fails too.
#[test]
fn the_cpu_backend_keeps_both_borrow_spellings() {
    let (ok, text) = compile(
        "cpuborrow",
        "@unsafe\nfn setit(r: &mut I32) {\n    *r = 7;\n}\n\n@unsafe\nfn readit(r: &I32) -> I32 {\n    return *r;\n}\n\nfn main() -> I32 {\n    let mut x: I32 = 1;\n    setit(&mut x);\n    return readit(&x);\n}\n",
        "--emit-cpu",
    );
    assert!(ok, "the probe did not emit:\n{}", text);
    assert!(
        text.contains("setit((&mut x))"),
        "`&mut x` did not reach the CPU backend as a mutable borrow:\n{}",
        text
    );
    assert!(
        text.contains("readit((&x))"),
        "`&x` was not emitted as a shared borrow:\n{}",
        text
    );
}
