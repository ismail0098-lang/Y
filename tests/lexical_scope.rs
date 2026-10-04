//! Y is lexically scoped, and the default backend did not compile it that way.
//!
//! The type checker gives every block, every `for` loop and every `match` arm
//! a scope of its own: an inner `let` SHADOWS an outer binding of the same
//! name until its block ends. The LLVM backend kept one stack slot per NAME
//! per function, so before `src/lexical_scope.rs` - each under "Compilation
//! Successful!":
//!
//! ```text
//!     let a = 1; @safe { let a = 2; } return a;     exit 2, want 1
//!     let x: I32 = 1; let x: F64 = 2.5;  x > 2.2    false (x held fptosi 2.5)
//!     for i in 0..3 { } for i in 0..4 { }            invalid IR: two `%i` allocas
//!     fn f(n: I32) { let n: F64 = 2.5; .. }          the F64 written into the I32 slot
//! ```
//!
//! Every binding gets a name, and so a slot, of its own now. These tests RUN
//! the program and compare its answer against a constant; a program that does
//! not build is a FAILURE when clang is installed, not a skip.
//! `tests/debug_info.rs` checks the same scopes from the debugger's side.
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

fn clang_available() -> bool {
    Command::new("clang").arg("--version").output().is_ok()
}

/// Build `src` with the default backend in a pinned scratch directory, run
/// it, and return its exit status and standard output. `None` only when
/// clang is absent.
fn run(name: &str, src: &str) -> Option<(i32, String)> {
    let dir = pinned::pinned_scratch(&format!("lexscope_{}", name), pinned::SM_PINNED);
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let bin = dir.join(name);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("-o")
        .arg(&bin)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    if !out.status.success() || !bin.exists() {
        if !clang_available() {
            eprintln!("SKIP {}: no clang on this machine, so this test checked NOTHING", name);
            return None;
        }
        panic!(
            "`{}` did not build:\n{}\n{}{}",
            name,
            src,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let ran = Command::new(&bin).output().expect("run the program");
    let code = ran.status.code().unwrap_or_else(|| panic!("`{}` crashed:\n{}", name, src));
    Some((code, String::from_utf8_lossy(&ran.stdout).into_owned()))
}

fn expect(name: &str, src: &str, want_exit: i32, want_stdout: &str) {
    if let Some((code, stdout)) = run(name, src) {
        assert_eq!(
            (code, stdout.as_str()),
            (want_exit, want_stdout),
            "`{}`: exit status and output differ from what the source says:\n{}",
            name,
            src
        );
    }
}

/// The case the backend got wrong in every block form: the inner binding is
/// the one the block sees, and the outer one is back after it.
#[test]
fn an_inner_let_shadows_the_outer_binding_until_its_block_ends() {
    let forms = [
        ("if_block", "if a > 0 {\n        let a: I32 = 2;\n        print_int(a);\n    }"),
        ("else_block", "if a > 5 {\n        print_int(9);\n    } else {\n        let a: I32 = 2;\n        print_int(a);\n    }"),
        ("safe_block", "@safe {\n        let a: I32 = 2;\n        print_int(a);\n    }"),
        ("ghost_block", "@ghost {\n        let a: I32 = 2;\n        print_int(a);\n    }"),
        ("for_body", "for k in 0..1 {\n        let a: I32 = 2;\n        print_int(a);\n    }"),
        ("while_body", "let mut once: I32 = 0;\n    while once < 1 {\n        let a: I32 = 2;\n        print_int(a);\n        once = once + 1;\n    }"),
    ];
    for (name, block) in forms {
        let src = format!("@unsafe\nfn main() -> I32 {{\n    let a: I32 = 1;\n    {}\n    return a;\n}}\n", block);
        expect(name, &src, 1, "2");
    }
}

/// `let x = x + 1;` reads the binding it is about to shadow: the new one does
/// not exist until its initialiser has run.
#[test]
fn an_initialiser_reads_the_binding_it_shadows() {
    expect(
        "init_reads_outer",
        "fn main() -> I32 {\n    let x: I32 = 5;\n    let x: I32 = x + 1;\n    return x;\n}\n",
        6,
        "",
    );
}

/// A repeated `let` may give the name another type: it is another binding.
#[test]
fn a_repeated_let_may_change_the_type() {
    expect(
        "retyped",
        "fn main() -> I32 {\n    let x: I32 = 1;\n    let x: F64 = 2.5;\n    if x > 2.2 {\n        return 1;\n    }\n    return 0;\n}\n",
        1,
        "",
    );
}

/// The loop variable is scoped to its loop, so a second loop may use the name.
#[test]
fn two_loops_may_use_the_same_variable_name() {
    expect(
        "two_loops",
        "@unsafe\nfn main() -> I32 {\n    let mut s: I32 = 0;\n    for i in 0..3 {\n        s = s + i;\n    }\n    \
         for i in 0..4 {\n        s = s + i;\n    }\n    return s;\n}\n",
        9,
        "",
    );
}

/// A `let` shadowing a parameter gets its own slot rather than writing the
/// parameter's, which has the parameter's type.
#[test]
fn a_let_may_shadow_a_parameter() {
    expect(
        "shadow_param",
        "fn f(n: I32) -> I32 {\n    let n: F64 = 2.5;\n    if n > 2.2 {\n        return 7;\n    }\n    return 3;\n}\n\n\
         fn main() -> I32 {\n    return f(100);\n}\n",
        7,
        "",
    );
}

/// Sibling blocks reuse a name freely, and so do nested ones three deep.
#[test]
fn sibling_and_nested_scopes_each_see_their_own_binding() {
    expect(
        "siblings",
        "@unsafe\nfn main() -> I32 {\n    let v: I32 = 1;\n    if v > 0 {\n        let v: I32 = 2;\n        if v > 1 {\n            \
         let v: I32 = 3;\n            print_int(v);\n        }\n        print_int(v);\n    }\n    if v > 0 {\n        \
         let v: I32 = 4;\n        print_int(v);\n    }\n    print_int(v);\n    return v;\n}\n",
        1,
        // `print_int` writes no newline: 3, then 2, then 4, then 1.
        "3241",
    );
}

/// A loop variable shadows an outer binding of its name inside the loop only:
/// after the loop the name is the outer binding again, with its own value.
/// (Bound in the enclosing scope instead, `return i` would read the loop's
/// last value - and no other test here reads a name after a loop that
/// shadowed it.)
#[test]
fn a_loop_variable_shadows_only_inside_its_loop() {
    expect(
        "loop_shadow",
        "@unsafe\nfn main() -> I32 {\n    let i: I32 = 7;\n    let mut s: I32 = 0;\n    for i in 0..3 {\n        \
         s = s + i;\n    }\n    return i * 10 + s;\n}\n",
        73,
        "",
    );
}
