//! A `let` in a nested block is a NEW binding, on every backend.
//!
//! The type checker gives every block its own scope, so in
//! `let a = 1; @safe { let a = 2; } return a;` the answer is 1. Each backend
//! keyed its storage by NAME for the whole function, so the inner `let`
//! took over the outer binding's storage and every later read saw the inner
//! value. Measured before the fix, each one compiling and running cleanly:
//!
//! ```text
//!     --emit-native   the ELF exited 2                    (want 1)
//!     --emit-ptx      `Out[0] = a` stored the inner `a`'s register
//!     --emit-cpu      `let mut a = 2;` emitted with no braces, so Rust's own
//!                     shadowing leaked it: the function returned 2
//! ```
//!
//! The LLVM backend was fixed first (`src/lexical_scope.rs`,
//! `tests/lexical_scope.rs`); the native and PTX backends now take the same
//! renaming, and the CPU backend gives the Y block a Rust block. The ZK
//! backend restores its scope after a block and was already right; it is in
//! the cross-backend case as a control.
//!
//! Every case RUNS what the backend produced, except the PTX one, which has no
//! device in a GPU-less test run: there the oracle is the SAME kernel with the
//! inner binding renamed by hand, which must compile to byte-identical PTX.
#[path = "common/pinned.rs"]
mod pinned;

use std::path::Path;
use std::process::Command;

fn y(dir: &Path, src_name: &str, src: &str, args: &[&str]) -> (bool, String) {
    let path = dir.join(src_name);
    std::fs::write(&path, src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .args(args)
        .current_dir(dir)
        .output()
        .expect("run Y");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

/// Builds `src` with `--emit-native` and runs it; the exit status.
fn native(tag: &str, src: &str) -> i32 {
    let d = pinned::pinned_scratch(&format!("nest_native_{tag}"), pinned::SM_PINNED);
    let bin = d.join("p.bin");
    let (ok, out) = y(&d, "p.ysu", src, &["--emit-native", &format!("--output={}", bin.display())]);
    assert!(ok, "--emit-native refused a program it supports:\n{out}");
    let st = Command::new(&bin).status().expect("run the ELF");
    let _ = std::fs::remove_dir_all(&d);
    st.code().expect("the ELF exited normally")
}

const NATIVE_OUTER: &str = "fn main() -> I32 {\n    let a: I32 = 1;\n    @safe {\n        let a: I32 = 2;\n    }\n    return a;\n}\n";
const NATIVE_INNER: &str = "fn main() -> I32 {\n    let a: I32 = 1;\n    @safe {\n        let a: I32 = 2;\n        return a + 40;\n    }\n    return 0;\n}\n";
const NATIVE_INIT: &str = "fn main() -> I32 {\n    let a: I32 = 5;\n    @safe {\n        let a: I32 = a + 1;\n        return a * 10 + 3;\n    }\n    return 0;\n}\n";

#[test]
fn native_binds_a_nested_let_in_its_own_scope() {
    assert_eq!(native("outer", NATIVE_OUTER), 1, "after the block `a` is the outer binding");
    // Inside the block it is the inner one - the control that stops "never
    // let an inner binding be read" passing.
    assert_eq!(native("inner", NATIVE_INNER), 42, "inside the block `a` is the inner binding");
    // An initialiser reads the binding it shadows.
    assert_eq!(native("init", NATIVE_INIT), 63, "`let a = a + 1` reads the outer `a`");
}

fn ptx(tag: &str, src: &str) -> Result<String, String> {
    let d = pinned::pinned_scratch(&format!("nest_ptx_{tag}"), pinned::SM_PINNED);
    let o = d.join("k.ptx");
    let (ok, out) = y(&d, "k.ysu", src, &["--emit-ptx", "-o", &o.display().to_string()]);
    let r = if ok {
        Ok(std::fs::read_to_string(&o).expect("read PTX"))
    } else {
        Err(out)
    };
    let _ = std::fs::remove_dir_all(&d);
    r
}

/// PTX with comments removed: the emitter writes variable names in comments,
/// and those legitimately differ between the two spellings.
fn code(ptx: &str) -> String {
    ptx.lines()
        .map(|l| l.split("//").next().unwrap_or("").trim_end())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

const PTX_SHADOWED: &str = "kernel k(Out: GlobalMemory<I32>, N: I32) {\n    let a: I32 = 1;\n    if N > 0 {\n        let a: I32 = 2;\n        Out[1] = a;\n    }\n    Out[0] = a;\n}\nfn main() {}\n";
const PTX_RENAMED: &str = "kernel k(Out: GlobalMemory<I32>, N: I32) {\n    let a: I32 = 1;\n    if N > 0 {\n        let b: I32 = 2;\n        Out[1] = b;\n    }\n    Out[0] = a;\n}\nfn main() {}\n";
/// What the shadowed kernel computed before the fix: `Out[0]` takes the inner
/// binding's value. (Out of scope in Y, so written as the value itself.)
const PTX_WRONG: &str = "kernel k(Out: GlobalMemory<I32>, N: I32) {\n    let a: I32 = 1;\n    if N > 0 {\n        let b: I32 = 2;\n        Out[1] = b;\n    }\n    Out[0] = 2;\n}\nfn main() {}\n";

#[test]
fn ptx_binds_a_nested_let_in_its_own_scope() {
    let shadowed = code(&ptx("shadowed", PTX_SHADOWED).expect("the shadowed kernel compiles"));
    let renamed = code(&ptx("renamed", PTX_RENAMED).expect("the renamed kernel compiles"));
    assert_eq!(
        shadowed, renamed,
        "a nested `let a` must compile exactly as a differently-named binding would"
    );
    // Non-vacuity: the two readings of `Out[0] = a` are different PTX, so the
    // equality above is not two programs that compile alike whatever `a` is.
    let wrong = code(&ptx("wrong", PTX_WRONG).expect("the control compiles"));
    assert_ne!(renamed, wrong, "the control does not separate the two readings");
}

#[test]
fn a_chisel_block_naming_a_shadowed_variable_is_refused() {
    let shadowed = "kernel k(Out: GlobalMemory<F32>, N: I32) {\n    let a: F32 = 1.0;\n    if N > 0 {\n        let a: F32 = 2.0;\n        Out[1] = a;\n    }\n    chisel {\n        \"add.f32 %a, %a, %a;\";\n    }\n    Out[0] = a;\n}\nfn main() {}\n";
    let err = ptx("chisel_shadowed", shadowed).expect_err("a `%a` with two bindings must be refused");
    assert!(
        err.contains("binds `a` more than once") && err.contains("chisel"),
        "refused, but not for the shadowing:\n{err}"
    );
    // Control: with one binding the same block resolves `%a` to its register.
    let single = "kernel k(Out: GlobalMemory<F32>, N: I32) {\n    let a: F32 = 1.0;\n    chisel {\n        \"add.f32 %a, %a, %a;\";\n    }\n    Out[0] = a;\n}\nfn main() {}\n";
    let out = ptx("chisel_single", single).expect("one binding resolves");
    assert!(
        out.contains("add.f32 %f") && !out.contains("%a"),
        "the single-binding `chisel` block did not resolve `%a`:\n{out}"
    );
}

/// The `--emit-cpu` blob, compiled with rustc and run with `driver` as its
/// `main`. `None` when there is no working rustc here.
fn cpu(tag: &str, src: &str, driver: &str) -> Option<i32> {
    let d = pinned::pinned_scratch(&format!("nest_cpu_{tag}"), pinned::SM_PINNED);
    let blob = d.join("blob.rs");
    let (ok, out) = y(&d, "p.ysu", src, &["--emit-cpu", "-o", &blob.display().to_string()]);
    assert!(ok, "--emit-cpu refused:\n{out}");
    let body = std::fs::read_to_string(&blob).expect("read the blob");
    let rs = d.join("run.rs");
    std::fs::write(&rs, format!("#![allow(unused, non_snake_case)]\n{body}\n{driver}\n")).unwrap();
    let bin = d.join("run");
    let built = Command::new("rustc")
        .args(["--edition", "2021", "-o"])
        .arg(&bin)
        .arg(&rs)
        .output()
        .ok()?;
    if !built.status.success() {
        let err = String::from_utf8_lossy(&built.stderr);
        if err.contains("error: linker") {
            return None;
        }
        panic!("the emitted Rust does not compile:\n{err}\n--- blob ---\n{body}");
    }
    let code = Command::new(&bin).status().ok()?.code();
    let _ = std::fs::remove_dir_all(&d);
    code
}

#[test]
fn cpu_binds_a_nested_let_in_its_own_scope() {
    let src = "fn f(n: I32) -> I32 {\n    let a: I32 = 1;\n    @safe {\n        let a: I32 = 2;\n    }\n    @ghost {\n        let a: I32 = 7;\n    }\n    if n > 0 {\n        let a: I32 = 3;\n    }\n    return a;\n}\n";
    match cpu("outer", src, "fn main() { std::process::exit(f(5)); }") {
        Some(code) => assert_eq!(code, 1, "after the blocks `a` is the outer binding"),
        None => eprintln!("SKIP: no working rustc"),
    }
    // Control: inside the block the inner binding is read.
    let src = "fn f(n: I32) -> I32 {\n    let a: I32 = 1;\n    @safe {\n        let a: I32 = 2;\n        return a + 40;\n    }\n    return 0;\n}\n";
    match cpu("inner", src, "fn main() { std::process::exit(f(5)); }") {
        Some(code) => assert_eq!(code, 42, "inside the block `a` is the inner binding"),
        None => eprintln!("SKIP: no working rustc"),
    }
}
