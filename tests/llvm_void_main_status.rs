//! A `fn main()` with no return type exits 0.
//!
//! The C runtime's `main` calls `ysu_main` and hands `%eax` to `exit`
//! (`c_src/runtime.c`), so the entry point has to return an `i32`. The LLVM
//! backend emitted `define void @ysu_main()`, and the exit status was whatever
//! the last call had left in `%eax`. `VOID_MAIN` below exited **43** at `-O0`
//! (the helper's return value) and 0 at `-O2`, where the dead call is deleted:
//! a pass at the default level was luck, not a property.
//!
//! The CLI's AOT builds now give such a `main` an `i32` return of 0
//! (`LlvmEmitter::set_aot_entry_status`). An embedding or JIT caller calls
//! the function itself, so it keeps the source's `void` ABI.
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const VOID_MAIN: &str = "fn helper(a: I32) -> I32 {\n    return a * 6 + 1;\n}\n\n\
    fn main() {\n    let x: I32 = helper(7);\n}\n";

fn clang_available() -> bool {
    Command::new("clang").arg("--version").output().is_ok()
}

/// Build `src` with the default backend at `level`, run it, and return its
/// exit status. `None` only when clang is absent.
fn exit_status(name: &str, src: &str, level: &str) -> Option<i32> {
    let dir = pinned::pinned_scratch(&format!("voidmain_{}{}", name, level), pinned::SM_PINNED);
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let bin = dir.join(name);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("-o")
        .arg(&bin)
        .arg(level)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    if !out.status.success() || !bin.exists() {
        if !clang_available() {
            eprintln!("SKIP {}: no clang on this machine, so this test checked NOTHING", name);
            return None;
        }
        panic!(
            "`{}` did not build at {}:\n{}\n{}{}",
            name,
            level,
            src,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let ran = Command::new(&bin).output().expect("run the program");
    Some(ran.status.code().unwrap_or_else(|| panic!("`{}` crashed at {}", name, level)))
}

#[test]
fn a_void_main_exits_zero_at_every_level() {
    for level in ["-O0", "-O1", "-O2", "-O3"] {
        if let Some(code) = exit_status("void_main", VOID_MAIN, level) {
            assert_eq!(code, 0, "a void main exited {} at {}", code, level);
        }
    }
}

#[test]
fn a_bare_return_in_a_void_main_exits_zero() {
    let src = "fn helper(a: I32) -> I32 {\n    return a * 6 + 1;\n}\n\n\
        fn main() {\n    let x: I32 = helper(7);\n    if x > 1 {\n        return;\n    }\n    \
        let y: I32 = helper(1);\n}\n";
    if let Some(code) = exit_status("bare_return", src, "-O0") {
        assert_eq!(code, 0);
    }
}

/// The control: the exit status still carries a declared return value, so the
/// two tests above are not passing because every exit is 0.
#[test]
fn a_main_that_returns_a_value_still_exits_with_it() {
    let src = "fn helper(a: I32) -> I32 {\n    return a * 6 + 1;\n}\n\n\
        fn main() -> I32 {\n    return helper(7);\n}\n";
    if let Some(code) = exit_status("returns_value", src, "-O0") {
        assert_eq!(code, 43);
    }
}

/// The AOT entry is the CLI's choice; the library emitter keeps the source's
/// ABI for the callers that call `main` themselves.
#[test]
fn only_the_aot_entry_point_changes_its_return_type() {
    let program = y::parser::Parser::new(y::lexer::Lexer::new(VOID_MAIN).tokenize())
        .parse_program()
        .expect("parse");
    let hw = y::sentinel::HardwareProfile::default();
    let mut library = y::llvm_emitter::LlvmEmitter::new();
    let ir = library.emit_program(&program, &hw);
    assert!(ir.contains("define void @ysu_main()"), "{}", ir);
    let mut aot = y::llvm_emitter::LlvmEmitter::new();
    aot.set_aot_entry_status(true);
    let ir = aot.emit_program(&program, &hw);
    assert!(ir.contains("define i32 @ysu_main()"), "{}", ir);
    assert!(ir.contains("ret i32 0"), "{}", ir);

    let dir = pinned::pinned_scratch("voidmain_emit_llvm", pinned::SM_PINNED);
    let path = dir.join("v.ysu");
    std::fs::write(&path, VOID_MAIN).unwrap();
    let ll = dir.join("v.ll");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-llvm")
        .arg("-o")
        .arg(&ll)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = std::fs::read_to_string(&ll).expect("--emit-llvm wrote no file");
    assert!(text.contains("define i32 @ysu_main()"), "{}", text);
}
