//! A local array had NO STORAGE in the LLVM backend - the default one.
//!
//! `emit_type` answers `ptr` for every array, which is right for a parameter
//! (it arrives as a pointer) and was used for a `let` as well: the local got a
//! single pointer slot, `= {}` zeroed it, and every element access loaded that
//! null pointer and indexed from it in 8-byte `i64` slots, whatever the
//! element type. Measured on `6ffabe4`, each under "Compilation Successful!":
//!
//! ```text
//!     v[0] = 4; v[2] = 6; return v[0] + v[2];     exit 0, want 10 (clang deleted the
//!                                                  stores through null as UB)
//!     return v[1];  (after `= {}`)                 SEGFAULT, want 0
//!     let w: [I32; 3] = v;  /  w = v;              SEGFAULT
//!     f[0] = 1.5 in an [F32; 2]                    stored fptosi 1.5 = 1
//!     sum3(v)  with  fn sum3(a: [I32; 3])          invalid IR: `call @sum3(%[I32] ..)`
//!     half(a)  with  fn half(x: U32)               invalid IR: `call @half(%U32 ..)`
//! ```
//!
//! No test that RUNS an LLVM-compiled program used a local array, which is how
//! all of it survived; arrays inside a struct worked throughout, because a
//! struct field gets real `[N x T]` storage.
//!
//! The model now is C's array decay with Rust's value semantics: a local array
//! is `[N x T]` storage, an array expression evaluates to its address, `let`
//! and assignment copy, and a parameter is copied into storage of its own on
//! entry, so a callee's write does not reach its caller.
//!
//! **Every case that runs is compared against a constant**, and a case clang
//! refuses is a FAILURE, not a skip, when clang is installed - the lesson
//! `tests/llvm_integer_widths.rs` records about harnesses that skip vacuously
//! on invalid IR.
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Two tests building the same fixture name must not share a directory.
static SALT: AtomicUsize = AtomicUsize::new(0);

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "y_llarr_{}_{}_{}",
        std::process::id(),
        SALT.fetch_add(1, Ordering::SeqCst),
        name
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn clang_available() -> bool {
    Command::new("clang").arg("--version").output().is_ok()
}

enum Built {
    /// The binary ran and exited with this status.
    Ran(i32),
    /// Y refused the program, or clang refused the IR: the combined output.
    NotBuilt(String),
    /// The binary was built and did not exit normally (a signal).
    Crashed,
}

/// Compiles `src` with the default LLVM backend and runs it.
fn build_and_run(name: &str, src: &str) -> Built {
    let dir = scratch(name);
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
        return Built::NotBuilt(format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    match Command::new(&bin).status().ok().and_then(|s| s.code()) {
        Some(code) => Built::Ran(code),
        None => Built::Crashed,
    }
}

/// Runs `src` and requires exit status `want`. Skips only when clang is absent.
fn expect_exit(name: &str, src: &str, want: i32) {
    match build_and_run(name, src) {
        Built::Ran(code) => assert_eq!(code, want, "`{name}` exited {code}, want {want}:\n{src}"),
        Built::Crashed => panic!("`{name}` crashed (a signal), want exit {want}:\n{src}"),
        Built::NotBuilt(text) => {
            if clang_available() {
                panic!("`{name}` did not build although clang is installed:\n{src}\n{text}");
            }
            eprintln!("SKIP {name}: no clang on this machine");
        }
    }
}

/// `--emit-llvm` must refuse `src`, name `phrase`, and write no module.
fn expect_refused(name: &str, src: &str, phrase: &str) {
    let dir = scratch(name);
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-llvm")
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "`{name}` was accepted:\n{src}\n{text}");
    assert!(
        text.contains("[LLVM host backend]") && text.contains(phrase),
        "`{name}` was refused, but not by the LLVM backend for {phrase:?} - so this case \
         is stopped by something else and tests nothing here:\n{text}"
    );
    assert!(
        !dir.join(format!("{}.ll", name)).exists(),
        "`{name}` was refused but still wrote a module"
    );
}

#[test]
fn a_local_array_holds_what_is_written() {
    // exit 0 on 6ffabe4: the stores went through a null pointer.
    expect_exit(
        "arr_rw",
        "fn main() -> I32 {\n    let mut v: [I32; 3] = {};\n    v[0] = 4;\n    v[2] = 6;\n    \
         return v[0] + v[2];\n}\n",
        10,
    );
    // Two arrays in one frame must not share storage.
    expect_exit(
        "arr_two",
        "fn main() -> I32 {\n    let mut v: [I32; 3] = {};\n    let mut w: [I32; 3] = {};\n    \
         v[0] = 1;\n    w[0] = 2;\n    return v[0] * 10 + w[0];\n}\n",
        12,
    );
    // Written and read through loop indices.
    expect_exit(
        "arr_loop",
        "fn main() -> I32 {\n    let mut v: [I32; 4] = {};\n    let mut s: I32 = 0;\n    \
         @invariant(i >= 0)\n    for i in 0..4 {\n        v[i] = i * 2;\n    }\n    \
         @invariant(j >= 0)\n    for j in 0..4 {\n        s = s + v[j];\n    }\n    return s;\n}\n",
        12,
    );
}

#[test]
fn a_zero_initialised_element_reads_zero() {
    // SEGFAULT on 6ffabe4: the read loaded through the zeroed pointer slot.
    expect_exit(
        "arr_zero",
        "fn main() -> I32 {\n    let mut v: [I32; 3] = {};\n    return v[1];\n}\n",
        0,
    );
}

#[test]
fn elements_keep_their_declared_type() {
    // Elements were 8-byte i64 slots whatever the type, so a float went
    // through `fptosi`: 1.5 was stored as 1. Each of these answers 0 then.
    for (name, src) in [
        (
            "arr_f32_sum",
            "fn main() -> bool {\n    let mut f: [F32; 2] = {};\n    f[0] = 1.5;\n    \
             f[1] = 2.0;\n    return f[0] + f[1] > 3.25;\n}\n",
        ),
        (
            "arr_f64_prod",
            "fn main() -> bool {\n    let mut d: [F64; 2] = {};\n    d[0] = 1.25;\n    \
             d[1] = 2.5;\n    return d[0] * d[1] * 4.0 > 12.25;\n}\n",
        ),
        (
            "arr_bool",
            "fn main() -> bool {\n    let mut b: [bool; 2] = {};\n    b[1] = 3 > 2;\n    return b[1];\n}\n",
        ),
    ] {
        expect_exit(name, src, 1);
    }
    expect_exit(
        "arr_u8",
        "fn main() -> I32 {\n    let mut v: [U8; 4] = {};\n    v[3] = 200;\n    v[0] = 100;\n    \
         return v[3] - v[0];\n}\n",
        100,
    );
    // An element wider than 32 bits keeps its high half.
    expect_exit(
        "arr_i64",
        "fn main() -> I64 {\n    let mut v: [I64; 2] = {};\n    v[1] = 5000000000;\n    \
         return v[1] / 1000000000;\n}\n",
        5,
    );
}

#[test]
fn let_and_assignment_copy_the_array() {
    // Both SEGFAULTED on 6ffabe4. Each source is changed AFTER the copy, so a
    // copy that aliased its source would read the new value.
    expect_exit(
        "arr_assign",
        "fn main() -> I32 {\n    let mut v: [I32; 3] = {};\n    let mut w: [I32; 3] = {};\n    \
         v[1] = 8;\n    w = v;\n    v[1] = 1;\n    return w[1];\n}\n",
        8,
    );
    expect_exit(
        "arr_let_copy",
        "fn main() -> I32 {\n    let mut v: [I32; 3] = {};\n    v[2] = 9;\n    \
         let w: [I32; 3] = v;\n    v[2] = 1;\n    return w[2];\n}\n",
        9,
    );
}

#[test]
fn an_array_argument_is_passed_by_value() {
    // Invalid IR on 6ffabe4 (`call @f(%[I32] ..)`).
    expect_exit(
        "arr_param_read",
        "fn get1(a: [I32; 3]) -> I32 {\n    return a[1];\n}\n\n\
         fn main() -> I32 {\n    let mut v: [I32; 3] = {};\n    v[1] = 42;\n    return get1(v);\n}\n",
        42,
    );
    // The callee writes its parameter; the caller's array must not change.
    // By reference this would answer 99 * 100 + 99 = 9999.
    expect_exit(
        "arr_param_write",
        "fn poke(a: [I32; 3]) -> I32 {\n    a[0] = 99;\n    return a[0];\n}\n\n\
         fn main() -> I32 {\n    let mut v: [I32; 3] = {};\n    v[0] = 1;\n    \
         let r: I32 = poke(v);\n    return v[0] * 100 + r;\n}\n",
        199,
    );
}

/// The control: an array inside a struct was the one array shape that worked
/// on `6ffabe4` - written through the field, and returned inside a struct (the
/// replacement the refusal of an array return recommends). Both must keep
/// working. Field arrays do not go through local-array storage, so this is
/// what guards them; refusing every LOCAL array is caught by the tests above.
#[test]
fn struct_field_arrays_still_work() {
    expect_exit(
        "fld_plain",
        "struct S {\n    buf: [I32; 4],\n    tag: I32,\n}\n\nfn main() -> I32 {\n    \
         let mut s: S = {};\n    s.buf[1] = 7;\n    s.buf[3] = 9;\n    s.tag = 5;\n    \
         return s.buf[1] + s.buf[3] + s.tag;\n}\n",
        21,
    );
    expect_exit(
        "fld_returned",
        "struct Box3 {\n    v: [I32; 3],\n}\n\nfn mk() -> Box3 {\n    let mut b: Box3 = {};\n    \
         b.v[1] = 17;\n    return b;\n}\n\nfn main() -> I32 {\n    let b: Box3 = mk();\n    \
         return b.v[1];\n}\n",
        17,
    );
}

/// Where a struct field meets the other array shapes: passed as an array
/// argument (invalid IR on `6ffabe4`: `call @sum3(%[I32] ..)`), and a struct
/// literal initialised from a local array (which had no storage). The local
/// is changed after the literal is built, so an aliasing copy would read 1.
#[test]
fn struct_field_arrays_meet_local_arrays() {
    expect_exit(
        "fld_as_arg",
        "struct S {\n    buf: [I32; 3],\n}\n\nfn sum3(a: [I32; 3]) -> I32 {\n    \
         return a[0] + a[1] + a[2];\n}\n\nfn main() -> I32 {\n    let mut s: S = {};\n    \
         s.buf[0] = 10;\n    s.buf[1] = 20;\n    s.buf[2] = 12;\n    return sum3(s.buf);\n}\n",
        42,
    );
    expect_exit(
        "fld_from_local",
        "struct S {\n    buf: [I32; 3],\n    tag: I32,\n}\n\nfn main() -> I32 {\n    \
         let mut v: [I32; 3] = {};\n    v[2] = 40;\n    let s: S = S { buf: v, tag: 2 };\n    \
         v[2] = 1;\n    return s.buf[2] + s.tag;\n}\n",
        42,
    );
}

/// A call site derived each argument's type a second time, through a private
/// table that did not know `U32`: `call i32 @half(%U32 %x)`, which clang
/// refuses, after "Compilation Successful!".
#[test]
fn a_u32_argument_is_declared_as_its_definition_declares_it() {
    expect_exit(
        "u32_arg",
        "fn half(x: U32) -> U32 {\n    return x / 2;\n}\n\n\
         fn main() -> U32 {\n    let a: U32 = 20;\n    return half(a);\n}\n",
        10,
    );
}

/// A `Q16.16` value is lowered as an `i32` by `emit_type`'s default - a wrong
/// answer this change does not fix. Its call site used to name `%Q16.16`,
/// which clang refuses; taking the definition's `i32` instead would turn that
/// loud failure into a silently wrong program. Whatever the backend does with
/// this program, it must not run it to the wrong answer (`2 * 1.5 > 2.5` is
/// true; as `i32` it is `2 > 2.5`).
#[test]
fn a_q_format_parameter_never_runs_to_a_wrong_answer() {
    let src = "fn dbl(x: Q16.16) -> Q16.16 {\n    return x + x;\n}\n\n\
               fn main() -> bool {\n    let y: Q16.16 = dbl(1.5);\n    return y > 2.5;\n}\n";
    match build_and_run("q_param", src) {
        Built::Ran(code) => assert_eq!(code, 1, "a Q16.16 argument ran to a wrong answer"),
        Built::Crashed => panic!("a Q16.16 argument crashed"),
        Built::NotBuilt(_) => {}
    }
}

#[test]
fn arrays_this_backend_cannot_hold_are_refused_by_name() {
    expect_refused(
        "ref_return",
        "fn mk() -> [I32; 3] {\n    let mut v: [I32; 3] = {};\n    v[0] = 11;\n    return v;\n}\n\n\
         fn main() -> I32 {\n    let w: [I32; 3] = mk();\n    return w[0];\n}\n",
        "returns an array",
    );
    expect_refused(
        "ref_nested",
        "fn main() -> I32 {\n    let mut m: [[I32; 2]; 2] = {};\n    m[1][0] = 5;\n    return m[1][0];\n}\n",
        "its elements are `[I32]`",
    );
    expect_refused(
        "ref_structs",
        "struct P {\n    x: I32,\n    y: I32,\n}\n\nfn main() -> I32 {\n    let mut ps: [P; 2] = {};\n    \
         ps[1].y = 6;\n    return ps[1].y;\n}\n",
        "its elements are `P`",
    );
    expect_refused(
        "ref_strings",
        "fn main() -> I32 {\n    let mut s: [String; 2] = {};\n    return 0;\n}\n",
        "its elements are `String`",
    );
    // Under `@unsafe` the type checker does not bounds-check the indices, so
    // these reach the backend instead of being refused there first.
    expect_refused(
        "ref_zero_len",
        "@unsafe\nfn main() -> I32 {\n    let mut v: [I32; 0] = {};\n    return 3;\n}\n",
        "its length is 0",
    );
    expect_refused(
        "ref_const_len",
        "const N: I32 = 4;\n\n@unsafe\nfn main() -> I32 {\n    let mut v: [I32; N] = {};\n    return 3;\n}\n",
        "its length is not an integer literal",
    );
    expect_refused(
        "ref_param_nested",
        "fn f(m: [[I32; 2]; 2]) -> I32 {\n    return 0;\n}\n\nfn main() -> I32 {\n    return 0;\n}\n",
        "a parameter of type `[[I32]]`",
    );
}

/// A second `let` of an array name is another binding with storage of its
/// own. This was REFUSED while the backend kept one slot per name per
/// function - the second declaration would have written eight elements into
/// the first one's three - and every binding has a slot of its own now
/// (`src/lexical_scope.rs`).
#[test]
fn a_redeclared_array_gets_storage_of_its_own() {
    expect_exit(
        "redeclared",
        "fn main() -> I32 {\n    let mut v: [I32; 3] = {};\n    v[0] = 1;\n    let mut v: [I32; 8] = {};\n    \
         v[7] = 2;\n    return v[7] + v[0];\n}\n",
        2,
    );
}

/// Every aggregate copy in this backend is a `memcpy` claiming `align 8` on
/// both pointers, and an `[N x i32]` alloca is only 4-aligned by default. The
/// claim is what LLVM may use to widen the copy, so the storage has to honour
/// it - which no answer can show, hence the look at the module itself.
#[test]
fn array_storage_honours_the_alignment_its_copies_claim() {
    let dir = scratch("align");
    let path = dir.join("align.ysu");
    std::fs::write(
        &path,
        "fn get1(a: [I32; 3]) -> I32 {\n    return a[1];\n}\n\n\
         fn main() -> I32 {\n    let mut v: [I32; 3] = {};\n    let w: [I32; 3] = v;\n    return get1(w);\n}\n",
    )
    .expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-llvm")
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .output()
        .expect("run Y");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stdout));
    let ir = std::fs::read_to_string(dir.join("align.ll")).expect("read the module");
    let arrays: Vec<&str> = ir
        .lines()
        .filter(|l| l.contains("= alloca [") && !l.contains(".y_oob_sink"))
        .collect();
    assert_eq!(arrays.len(), 3, "expected storage for v, w and the parameter copy:\n{ir}");
    for l in arrays {
        assert!(l.trim_end().ends_with("align 8"), "array storage without `align 8`: {l}");
    }
}
