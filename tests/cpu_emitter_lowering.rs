//! The CPU backend emitted Rust that never compiled, and dropped whole
//! statements while doing it.
//!
//! `--emit-cpu` prints a scalar host Rust blob. Three defects, all of them
//! the design-rule shape CLAUDE.md catalogues - a missing case answered with
//! something plausible rather than refused:
//!
//! 1. **`emit_expr`'s `_ => "0 // Fallback"`.** `Expr::BinaryOp` was not
//!    handled, so `a + b` became the **constant 0**. `FloatLit`, `BoolLit`,
//!    `StringLit`, `CharLit` and `UnaryOp` likewise. And the trailing `//`
//!    commented out the rest of the emitted line, *including the statement's
//!    own terminator*.
//! 2. **`emit_block`'s `_ => {}`.** `if`, `while`, `+=` and `break` were
//!    silently DROPPED, so a loop body vanished and the function returned its
//!    initial value.
//! 3. **`Stmt::Let` never wrote a `;`** on the path that emits a real
//!    expression - only the two placeholder arms did, which is the tell that
//!    those were the only paths ever looked at.
//!
//! **These tests COMPILE AND RUN the emitted blob**, because the two axes fail
//! independently: a blob that compiles can still be missing a statement (bug 2
//! produced perfectly valid Rust), and asserting on substrings alone cannot see
//! a wrong *value*. Same argument as `async_copy_is_committed_and_awaited` -
//! assembling proves nothing about what was left out.
use std::path::PathBuf;
use std::process::Command;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("y_cpuemit_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// Compiles `src` with `--emit-cpu` and returns the generated Rust blob.
fn emit_cpu(name: &str, src: &str) -> String {
    let dir = scratch(name);
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-cpu")
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .output()
        .expect("run Y");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "Y refused to emit `{}`:\n{}\n{}",
        name,
        stdout,
        String::from_utf8_lossy(&out.stderr)
    );
    // Match on WHOLE LINES. The blob's own first line is a `//` banner of 59
    // `=`, so a substring search for the closing rule finds that instead and
    // returns a one-line "blob" that every assertion below then reports as a
    // dropped statement.
    let mut lines = stdout.lines();
    for l in lines.by_ref() {
        if l.contains("GENERATED RUST BLOB") {
            break;
        }
    }
    let mut blob = Vec::new();
    for l in lines {
        if l.trim_start_matches('=').is_empty() && l.len() > 20 && !l.starts_with("//") {
            return blob.join("\n");
        }
        blob.push(l);
    }
    panic!("no terminated blob in output:\n{}", stdout)
}

/// Compiles the blob plus a driver `main` with rustc and runs it, returning the
/// exit code. `None` means rustc is unavailable.
fn build_and_run(name: &str, blob: &str, driver: &str) -> Option<i32> {
    // The blob is used VERBATIM. This used to filter out `use crate::…`,
    // which is how that import survived being unresolvable for every reader
    // outside the Y crate: two separate harnesses removed it before checking,
    // so neither could see that the artifact as emitted does not compile.
    // `no_emitted_blob_imports_a_path_only_the_compiler_can_resolve` in
    // `cpu_emitter_output_compiles.rs` now refuses the whole class.
    let body: String = blob.to_string();
    let dir = scratch(name);
    let rs = dir.join(format!("{}.rs", name));
    std::fs::write(&rs, format!("#![allow(unused, non_snake_case)]\n{}\n{}\n", body, driver))
        .expect("write rs");
    let bin = dir.join(name);
    let out = Command::new("rustc")
        .args(["--edition", "2021", "-O", "-o"])
        .arg(&bin)
        .arg(&rs)
        .output()
        .ok()?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("error: linker") || err.contains("not found") && !err.contains("error[") {
            return None; // no working toolchain here
        }
        panic!(
            "the emitted Rust does not compile:\n{}\n--- blob ---\n{}",
            err, body
        );
    }
    Command::new(&bin).status().ok().and_then(|s| s.code())
}

#[test]
fn arithmetic_is_not_replaced_by_zero() {
    // `a + b` used to emit the literal `0`, and the `let` had no `;`, so this
    // function was neither correct nor syntactically valid Rust.
    let blob = emit_cpu(
        "arith",
        "fn addmul(a: I32, b: I32) -> I32 {\n    \
         let s: I32 = a + b;\n    \
         let p: I32 = s * 3;\n    \
         return p - 1;\n}\n",
    );
    assert!(
        blob.contains("(a + b)"),
        "the addition is missing from the blob:\n{}",
        blob
    );
    match build_and_run(
        "arith",
        &blob,
        "fn main() { std::process::exit(unsafe { addmul(4, 5) }); }",
    ) {
        None => eprintln!("SKIP arith: rustc unavailable"),
        // (4 + 5) * 3 - 1 = 26. Compare against a CONSTANT, never against a
        // second value that went through the same suspect path.
        Some(code) => assert_eq!(code, 26, "addmul(4, 5) should be 26"),
    }
}

#[test]
fn control_flow_statements_are_not_dropped() {
    // `if`, `else` and `+=` all hit `_ => {}`, so the loop body was emitted
    // EMPTY and this returned its initial value. The blob compiled fine - which
    // is exactly why a compile-only gate is not enough.
    let blob = emit_cpu(
        "ctl",
        "fn classify() -> I32 {\n    \
         let mut acc: I32 = 0;\n    \
         @invariant(i >= 0)\n    \
         for i in 0..8 {\n        \
         if i > 2 {\n            acc += i;\n        } else {\n            acc -= 1;\n        }\n    }\n    \
         return acc;\n}\n",
    );
    for needed in ["if (i > 2)", "} else {", "acc += i;", "acc -= 1;"] {
        assert!(
            blob.contains(needed),
            "`{}` was dropped from the blob:\n{}",
            needed,
            blob
        );
    }
    match build_and_run(
        "ctl",
        &blob,
        "fn main() { std::process::exit(unsafe { classify() }); }",
    ) {
        None => eprintln!("SKIP ctl: rustc unavailable"),
        // i = 0,1,2 -> -1 each; i = 3..7 -> +3+4+5+6+7 = 25. 25 - 3 = 22.
        // With the bug the loop body was empty and this was 0.
        Some(code) => assert_eq!(code, 22, "the loop body is not being executed"),
    }
}

#[test]
fn literals_keep_their_type_and_value() {
    // FloatLit/BoolLit both became `0 // Fallback`. Floats also have to print
    // as `1.5`/`2.0` rather than `2`, or rustc re-types them as integers.
    let blob = emit_cpu(
        "lits",
        "fn f() -> F32 {\n    \
         let a: F32 = 2.0;\n    let b: F32 = 1.5;\n    \
         return a + b;\n}\n",
    );
    assert!(blob.contains("2.0"), "float literal lost its point:\n{}", blob);
    match build_and_run(
        "lits",
        &blob,
        "fn main() { std::process::exit(if (unsafe { f() } - 3.5f32).abs() < 1e-6 { 0 } else { 1 }); }",
    ) {
        None => eprintln!("SKIP lits: rustc unavailable"),
        Some(code) => assert_eq!(code, 0, "2.0 + 1.5 did not come out as 3.5"),
    }
}

#[test]
fn unlowerable_constructs_are_refused_not_guessed() {
    // The control for the three tests above: a backend that emits SOMETHING for
    // everything passes all of them and is exactly the bug. A construct this
    // backend cannot lower must fail the build, naming the construct and its
    // position.
    //
    // **The first version of this test used a `match`, and it was vacuous** -
    // the parser rejects `match` outright, so the assertion `!success` held for
    // a reason that has nothing to do with the backend, and deleting
    // `unsupported_stmt` entirely would have left it green. Both fixtures below
    // were checked to reach the emitter, and each asserts on ITS OWN message
    // rather than on "something failed".
    for (name, src, phrase) in [
        (
            "refuse_path",
            "fn f() -> I32 {\n    let t: I32 = Widget::spin(1);\n    return t;\n}\n",
            "`Widget::spin`",
        ),
        (
            "refuse_chisel",
            "fn f() -> I32 {\n    chisel {\n        let r: I32 = 1;\n    }\n    return 0;\n}\n",
            "a `chisel` block",
        ),
    ] {
        let dir = scratch(name);
        let path = dir.join(format!("{}.ysu", name));
        std::fs::write(&path, src).expect("write");
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&path)
            .arg("--emit-cpu")
            .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
            .output()
            .expect("run Y");
        let all = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.status.success(),
            "{}: a construct the CPU backend cannot lower produced a blob and \
             exit 0:\n{}",
            name,
            all
        );
        assert!(
            all.contains("[CPU Backend]") && all.contains(phrase),
            "{}: the build failed, but not with the CPU backend's own refusal \
             for {} - so this fixture is being rejected by an EARLIER pass and \
             proves nothing about the emitter:\n{}",
            name,
            phrase,
            all
        );
    }
}
