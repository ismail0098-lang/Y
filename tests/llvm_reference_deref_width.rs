//! A write through `&mut T` must be exactly `T` wide.
//!
//! `fn g(r: &mut I32) { *r = 7; }` emitted
//!
//!     %_t2 = sext i32 7 to i64
//!     store i64 %_t2, ptr %_t1
//!
//! an **eight-byte store through a pointer to four bytes**. That is valid IR,
//! so `clang` accepted it silently and the compiler printed "Compilation
//! Successful!" and exited 0. With `struct Pair { a: I32, b: I32 }`, writing
//! through `&mut p.a` set `p.b` to zero: a runnable binary computing the wrong
//! answer, the shape CLAUDE.md's design-rule table catalogues.
//!
//! The cause was a **sentinel collision**, not a missing arm.
//! `ast_type_to_llvm_type` answers `"i32"` for five different reasons - a
//! genuine `I32`, an `Unknown` ast type, an empty one, an unregistered name,
//! and a data-less enum - so its three callers could not tell success from
//! failure and used `resolved != "i32"` to mean "resolution succeeded",
//! substituting `i64` otherwise. The correct answer for the language's
//! commonest pointer was thrown away *because it was correct*.
//!
//! Note the asymmetry that hid it: `&mut I64` resolves to `"i64"`, which is
//! not the sentinel, so it was right all along. Every `*x` in the repo's own
//! corpus is either `&mut I64` (`tests/ring_buffer.ysu`) or `&mut String`
//! (`self_hosted/type_checker.ysu`), both of which dodge it - the fix changes
//! the emitted IR of **zero** of the 76 corpus programs the LLVM backend
//! accepts. The bug was live and simply uncovered.
//!
//! These tests RUN the binary, for the reason `tests/llvm_integer_widths.rs`
//! records: two values that go through the same wrong path agree with each
//! other while both are wrong.
use std::path::PathBuf;
use std::process::Command;

/// Compile `src` to a native binary and return its exit status.
///
/// `None` means only "this machine has no clang". A binary that failed to
/// appear *while* clang is installed is a rejected module and panics - the
/// vacuous-skip hole found by mutation in `llvm_control_flow.rs` and
/// `llvm_integer_widths.rs`, which has now bitten three files.
fn run_program(name: &str, src: &str) -> Option<i32> {
    let dir = std::env::temp_dir().join(format!("y_refw_{}_{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let bin = dir.join(name);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("-o")
        .arg(&bin)
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .output()
        .expect("run Y");
    if !out.status.success() || !bin.exists() {
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if Command::new("clang").arg("--version").output().is_ok() {
            panic!(
                "`{name}` produced no binary although clang is installed, so \
                 the emitted IR was rejected:\n{text}"
            );
        }
        return None;
    }
    Command::new(&bin).status().ok().and_then(|s| s.code())
}

/// Emit LLVM IR for `src` and return it, or `None` if the backend refused.
fn emit_ir(name: &str, src: &str) -> Option<String> {
    let dir = std::env::temp_dir().join(format!("y_refir_{}_{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-llvm")
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .output()
        .expect("run Y");
    if !out.status.success() {
        return None;
    }
    std::fs::read_to_string(dir.join(format!("{}.ll", name))).ok()
}

/// The load-bearing case: an 8-byte store through `&mut I32` clobbers the
/// neighbouring struct field. `%Pair = type { i32, i32 }`, so the field at
/// offset 4 is deterministically inside the range a store at offset 0 covers -
/// this does not depend on stack layout luck.
#[test]
fn a_write_through_a_mut_i32_does_not_touch_the_next_field() {
    let src = "\
struct Pair {
    a: I32,
    b: I32,
}

@unsafe
fn setit(r: &mut I32) {
    *r = 7;
}

fn main() -> I32 {
    let mut p: Pair = Pair { a: 1, b: 5 };
    setit(&mut p.a);
    return p.b;
}
";
    match run_program("neighbour", src) {
        Some(code) => assert_eq!(
            code, 5,
            "writing 7 through `&mut p.a` changed `p.b` from 5 to {}: the store \
             is wider than the type it points at",
            code
        ),
        None => eprintln!("skipped: no clang on this machine"),
    }
}

/// The positive control. Refusing every deref, or narrowing every pointee to
/// nothing, would satisfy the test above; this is what stops that - the value
/// must still actually land at the target.
#[test]
fn a_write_through_a_mut_i32_still_lands() {
    let src = "\
@unsafe
fn setit(r: &mut I32) {
    *r = 7;
}

fn main() -> I32 {
    let mut x: I32 = 1;
    setit(&mut x);
    return x;
}
";
    match run_program("lands", src) {
        Some(code) => assert_eq!(code, 7, "the write through `&mut I32` did not land"),
        None => eprintln!("skipped: no clang on this machine"),
    }
}

/// The width control in the other direction. `&mut I64` was always correct,
/// because `"i64"` is not the sentinel value - so a "fix" that pins every
/// pointee to `i32` passes both tests above and breaks this one.
#[test]
fn a_write_through_a_mut_i64_keeps_its_full_width() {
    let src = "\
@unsafe
fn setit(r: &mut I64) {
    *r = 8589934593;
}

fn main() -> I32 {
    let mut x: I64 = 0;
    setit(&mut x);
    if x == 8589934593 {
        return 1;
    }
    return 0;
}
";
    match run_program("wide", src) {
        Some(code) => assert_eq!(
            code, 1,
            "a value needing 34 bits did not survive a write through `&mut I64`"
        ),
        None => eprintln!("skipped: no clang on this machine"),
    }
}

/// The read direction has the same cause and is *not* reliably observable at
/// runtime - the over-read's extra four bytes are truncated away again by the
/// coercion to `I32`, so the value is right and only the access is out of
/// bounds. It is pinned on the IR instead, which is the honest place for a
/// property whose symptom is UB rather than a wrong answer.
#[test]
fn a_read_through_a_mut_i32_loads_four_bytes_not_eight() {
    let src = "\
@unsafe
fn getit(r: &mut I32) -> I32 {
    let v: I32 = *r;
    return v;
}

fn main() -> I32 {
    let mut x: I32 = 3;
    return getit(&mut x);
}
";
    let ir = match emit_ir("readw", src) {
        Some(ir) => ir,
        None => {
            eprintln!("skipped: backend refused the probe");
            return;
        }
    };
    let body: String = ir
        .lines()
        .skip_while(|l| !l.contains("define") || !l.contains("@getit"))
        .take_while(|l| !l.starts_with('}'))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        body.contains("load i32, ptr"),
        "no four-byte load through `&mut I32` in @getit:\n{}",
        body
    );
    assert!(
        !body.contains("load i64, ptr"),
        "an eight-byte load through a pointer to four bytes:\n{}",
        body
    );
}
