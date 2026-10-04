//! An element reached through a pointer is as wide as its element type.
//!
//! The LLVM backend indexed every pointer in 8-byte `i64` slots, whatever it
//! pointed at. Measured before the fix, each under "Compilation Successful!":
//!
//! ```text
//!     kernel add1(Out: GlobalMemory<I32>, ..) called with a [I32; 4]
//!         plain build: wrong answer; -g build: SIGILL (return address overwritten)
//!     fn bump(a: &mut [I16; 4]) { a[1] = a[1] + 5; }     v[1] is 0, want 5
//!     "hello"[1]                                          5 (the length field)
//! ```
//!
//! and `-g` described `Out` as a pointer to `I32`, so the debugger read the
//! memory the code had just written 8 bytes at a time 4 bytes at a time.
//!
//! The element type comes from the declared `GlobalMemory<T>` or the
//! referenced array; a pointer whose element type the backend cannot see is
//! refused by name rather than read 8 bytes at a time. These tests RUN the
//! program: the buffer is a struct field with a guard array right after it,
//! so an over-wide write lands in memory the program then checks,
//! deterministically, rather than somewhere on the stack.
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok()
}

fn text(out: &std::process::Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// Build `src` (with `flags`) in a pinned scratch directory. `None` only when
/// clang is absent.
fn build(name: &str, src: &str, flags: &[&str]) -> Option<(PathBuf, PathBuf)> {
    let dir = pinned::pinned_scratch(&format!("ptrelem_{}", name), pinned::SM_PINNED);
    let path = dir.join(format!("{}.ysu", name));
    fs::write(&path, src).expect("write source");
    let bin = dir.join(name);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .args(flags)
        .arg("-o")
        .arg(&bin)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    if !out.status.success() || !bin.exists() {
        if !have("clang") {
            eprintln!("SKIP {}: no clang on this machine, so this test checked NOTHING", name);
            return None;
        }
        panic!("`{}` did not build:\n{}\n{}", name, src, text(&out));
    }
    Some((dir, bin))
}

/// Build and run: the exit status. A crash is a failure, not a status.
fn run(name: &str, src: &str) -> Option<i32> {
    let (dir, bin) = build(name, src, &[])?;
    let ran = Command::new(&bin).output().expect("run the program");
    let code = ran.status.code().unwrap_or_else(|| panic!("`{}` crashed ({:?}):\n{}", name, ran.status, src));
    let _ = fs::remove_dir_all(&dir);
    Some(code)
}

/// A kernel adds 1 to each of 4 elements of `T` in a struct whose next field
/// is a guard; main returns 1 only if every element became 1 and the guard
/// is untouched.
fn guarded(t: &str, one: &str) -> String {
    format!(
        "struct Guarded {{\n    data: [{t}; 4],\n    guard: [I32; 8],\n}}\n\n\
         kernel add1(Out: GlobalMemory<{t}>, N: I32) {{\n    @invariant(i >= 0)\n    for i in 0..N {{\n        \
         Out[i] = Out[i] + {one};\n    }}\n}}\n\n\
         @unsafe\nfn main() -> I32 {{\n    let mut g: Guarded = Guarded {{ data: {{}}, guard: {{}} }};\n    \
         for k in 0..8 {{\n        g.guard[k] = 7;\n    }}\n    add1(g.data, 4);\n    let mut ok: I32 = 1;\n    \
         for k in 0..4 {{\n        if g.data[k] != {one} {{\n            ok = 0;\n        }}\n    }}\n    \
         for k in 0..8 {{\n        if g.guard[k] != 7 {{\n            ok = 0;\n        }}\n    }}\n    return ok;\n}}\n"
    )
}

#[test]
fn every_element_type_steps_and_stores_its_own_width() {
    for (t, one) in [
        ("I8", "1"),
        ("U8", "1"),
        ("I16", "1"),
        ("U16", "1"),
        ("I32", "1"),
        ("U32", "1"),
        ("I64", "1"),
        ("U64", "1"),
        ("F32", "1.0"),
        ("F64", "1.0"),
    ] {
        if let Some(code) = run(&format!("add1_{}", t.to_lowercase()), &guarded(t, one)) {
            assert_eq!(
                code, 1,
                "GlobalMemory<{}>: an element was missed, or the guard after the buffer was overwritten",
                t
            );
        }
    }
}

/// The same through a reference to an array, where the element type comes
/// from the array rather than a `GlobalMemory<T>`. The kernel before it has
/// a `GlobalMemory<I64>` parameter of the SAME NAME, so an element type left
/// over from the previous definition would index `bump`'s `a` 8 bytes at a
/// time.
#[test]
fn a_reference_to_an_array_is_indexed_at_its_element_width() {
    let src = "\
struct Guarded {
    data: [I16; 4],
    guard: [I16; 4],
}

kernel widen(a: GlobalMemory<I64>, N: I32) {
    @invariant(i >= 0)
    for i in 0..N {
        a[i] = a[i] + 1;
    }
}

fn bump(a: &mut [I16; 4]) {
    a[1] = a[1] + 5;
    a[3] = a[3] + 9;
}

@unsafe
fn main() -> I32 {
    let mut g: Guarded = Guarded { data: {}, guard: {} };
    bump(&mut g.data);
    let mut guard_ok: I32 = 1;
    for k in 0..4 {
        if g.guard[k] != 0 {
            guard_ok = 0;
        }
    }
    return g.data[1] * 10 + g.data[3] + guard_ok * 100;
}
";
    if let Some(code) = run("refarr", src) {
        assert_eq!(code, 159, "want data[1] = 5, data[3] = 9 and the guard untouched");
    }
}

/// A pointer whose element type the backend cannot see is refused by name,
/// not indexed 8 bytes at a time.
#[test]
fn an_unknown_element_type_is_refused_by_name() {
    let dir = pinned::pinned_scratch("ptrelem_refuse", pinned::SM_PINNED);
    for (name, src) in [
        ("qfmt", "kernel k(Out: GlobalMemory<Q16.16>) {\n    Out[0] = Out[1];\n}\n\nfn main() {}\n"),
        ("boolbuf", "kernel k(Out: GlobalMemory<bool>) {\n    Out[0] = true;\n}\n\nfn main() {}\n"),
        ("string", "fn main() -> I32 {\n    let s: String = \"hello\";\n    return s[1];\n}\n"),
    ] {
        let path = dir.join(format!("{}.ysu", name));
        fs::write(&path, src).expect("write");
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&path)
            .arg("--emit-llvm")
            .arg("-o")
            .arg(dir.join(format!("{}.ll", name)))
            .current_dir(&dir)
            .output()
            .expect("run Y");
        assert!(!out.status.success(), "{}: an element of unknown width was indexed:\n{}", name, text(&out));
        assert!(
            text(&out).contains("is indexed through a pointer whose element type this backend does not know"),
            "{}: refused for another reason:\n{}",
            name,
            text(&out)
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The debugger and the program agree about the memory: `-g` describes
/// `Out` as a pointer to `I32`, and the kernel now writes 4-byte elements, so
/// gdb reads back exactly what the kernel wrote.
#[test]
fn the_debugger_reads_what_the_kernel_wrote() {
    if !have("gdb") {
        eprintln!("SKIP the_debugger_reads_what_the_kernel_wrote: no gdb on this machine, so this test checked NOTHING");
        return;
    }
    let src = "\
kernel add1(Out: GlobalMemory<I32>, N: I32) {
    @invariant(i >= 0)
    for i in 0..N {
        Out[i] = Out[i] + i + 1;
    }
    let done: I32 = 1; // L:done
}

@unsafe
fn main() -> I32 {
    let mut buf: [I32; 4] = {};
    add1(buf, 4);
    return buf[3];
}
";
    let line = src.lines().position(|l| l.contains("// L:done")).expect("marker") + 1;
    let Some((dir, bin)) = build("gdbview", src, &["-g"]) else { return };
    let out = Command::new("gdb")
        .args(["-batch", "-nx", "-iex"])
        .arg(format!("add-auto-load-safe-path {}", bin.display()))
        .arg("-ex")
        .arg(format!("break gdbview.ysu:{}", line))
        .args(["-ex", "run", "-ex", "print *Out@4", "-ex", "kill"])
        .arg(&bin)
        .current_dir(&dir)
        .output()
        .expect("run gdb");
    let t = text(&out);
    assert!(t.contains("{1, 2, 3, 4}"), "gdb's view of Out after the loop:\n{}", t);
    let _ = fs::remove_dir_all(&dir);
}


/// The emitter's catch-all turned an expression it did not handle into the
/// constant 0 (`add i32 0, 0 ; unhandled expr`). The one variant that reached
/// it, a block expression, is one the parser never builds, so this builds it
/// by hand: it is refused by name now, and nothing in the module is a zero
/// standing in for it.
#[test]
fn an_unhandled_expression_is_refused_not_evaluated_as_zero() {
    use y::ast::{Block, Expr, Item, Stmt};
    let src = "fn main() -> I32 {\n    return 7;\n}\n";
    let tokens = y::lexer::Lexer::new(src).tokenize();
    let mut prog = y::parser::Parser::new(tokens).parse_program().expect("parse");
    let Some(Item::Func(f)) = prog.items.iter_mut().find(|i| matches!(i, Item::Func(_))) else {
        panic!("no function parsed");
    };
    let Some(Stmt::Return(Some(value), span)) = f.body.stmts.first_mut() else {
        panic!("the body is not `return <expr>`");
    };
    *value = Expr::BlockExpr(Block { stmts: vec![], span: span.clone() }, span.clone());
    let mut e = y::llvm_emitter::LlvmEmitter::new();
    let ir = e.emit_program(&prog, &y::sentinel::HardwareProfile::default());
    assert!(
        e.emit_errors.iter().any(|m| m.contains("a block used as an expression has no lowering")),
        "a block expression was not refused: {:?}",
        e.emit_errors
    );
    assert!(!ir.contains("unhandled expr"), "a constant stood in for the expression:\n{}", ir);
}
