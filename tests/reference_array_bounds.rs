//! An array reached through a REFERENCE is bounds-checked like the array.
//!
//! Strict mode - the default, which `@unsafe` turns off - refuses an array
//! index it cannot prove in bounds. The rule matched a plain array type only,
//! so through `a: &mut [I16; 4]` an index was neither proved nor checked at run
//! time: `a[k] = 1` with an unconstrained `k` compiled clean, and a call with
//! `k = 9` wrote past the array and exited 0. Member access already looked
//! through one reference (`s.buffer[k]` through `s: &mut S` is checked); the
//! index site did not.
//!
//! These tests RUN the programs: the run-time check under `@unsafe` is only
//! observable by executing it, and an in-bounds control stops "always fail"
//! from satisfying the out-of-bounds case.
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "y_refidx_{}_{}_{}",
        std::process::id(),
        tag,
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn clang_available() -> bool {
    Command::new("clang").arg("--version").output().is_ok()
}

/// What happened to `src`: Y's verdict, and if it built, the run.
struct Outcome {
    built: bool,
    compiler: String,
    /// `(exit status, stdout)`; `None` when not built or killed by a signal.
    run: Option<(i32, String)>,
}

/// Compile `src` with the default LLVM backend and, if it builds, run it.
fn build_and_run(tag: &str, src: &str) -> Outcome {
    let dir = scratch(tag);
    let path = dir.join("prog.ysu");
    std::fs::write(&path, src).expect("write source");
    let bin = dir.join("prog");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("-o")
        .arg(&bin)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run Y");
    let compiler = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let built = out.status.success() && bin.exists();
    let run = if built {
        let r = Command::new(&bin).output().expect("run the program");
        r.status.code().map(|c| (c, String::from_utf8_lossy(&r.stdout).into_owned()))
    } else {
        None
    };
    let _ = std::fs::remove_dir_all(&dir);
    Outcome { built, compiler, run }
}

/// `true` when clang is missing, after saying so: a build that needs clang
/// cannot have been tested without it.
fn skip_without_clang(o: &Outcome, test: &str) -> bool {
    if !o.built && !clang_available() {
        eprintln!("SKIP {}: no clang on this machine, so this test checked NOTHING", test);
        return true;
    }
    false
}

/// The finding. This compiled clean and wrote past the array.
#[test]
fn an_unprovable_index_through_a_reference_is_refused() {
    let src = "\
fn poke(a: &mut [I16; 4], k: I32) -> I32 {
    a[k] = 1;
    return 0;
}

fn main() -> I32 {
    let v: [I16; 4] = {};
    return poke(&mut v, 9);
}
";
    let o = build_and_run("unprovable", src);
    assert!(!o.built, "an index through a reference compiled with no proof:\n{}", o.compiler);
    assert!(
        o.compiler.contains("Line 2: [Strict Safety] Array access is unsafe: index has no statically provable bounds"),
        "refused for the wrong reason:\n{}",
        o.compiler
    );
}

/// The bound is the REFERENCED array's: one past the end is refused with that
/// array's size. An index the check never looked at could not name it.
#[test]
fn the_bound_is_the_referenced_arrays_size() {
    for (mutable, tag) in [("&mut ", "mut"), ("&", "shared")] {
        let src = format!(
            "\
fn peek(a: {mutable}[I16; 4]) -> I32 {{
    return a[4];
}}

fn main() -> I32 {{
    let v: [I16; 4] = {{}};
    return peek({mutable}v);
}}
"
        );
        let o = build_and_run(tag, &src);
        assert!(!o.built, "`a[4]` through `{}[I16; 4]` compiled:\n{}", mutable, o.compiler);
        assert!(
            o.compiler.contains("possible overflow index access (inferred max: 4 >= array size 4)"),
            "`{}`: refused for the wrong reason:\n{}",
            mutable,
            o.compiler
        );
    }
}

/// The control: an index the compiler CAN prove, through a reference, still
/// compiles and computes the right answer - through `&mut` and through `&`.
#[test]
fn a_provable_index_through_a_reference_still_works() {
    let src = "\
fn fill(a: &mut [I16; 4]) -> I32 {
    @invariant(i >= 0)
    for i in 0..4 {
        a[i] = i * 10;
    }
    return 0;
}

fn total(a: &[I16; 4]) -> I32 {
    let s: I32 = 0;
    @invariant(i >= 0)
    for i in 0..4 {
        s = s + a[i];
    }
    return s;
}

fn main() -> I32 {
    let v: [I16; 4] = {};
    let r: I32 = fill(&mut v);
    return total(&v) + r;
}
";
    let o = build_and_run("provable", src);
    if skip_without_clang(&o, "a_provable_index_through_a_reference_still_works") {
        return;
    }
    assert!(o.built, "a provable index through a reference was refused:\n{}", o.compiler);
    assert_eq!(o.run.as_ref().map(|r| r.0), Some(60), "0 + 10 + 20 + 30:\n{}", o.compiler);
}

/// What lets the loops above verify is narrow: indexing through a reference
/// hands no reference to anyone only when the array's ELEMENTS carry none. An
/// element that carries `&mut I32` is still a reference passed to the call,
/// and the invariant is still refused - otherwise `bump_link(a[0])` could
/// change a tracked variable behind the verifier's back.
#[test]
fn an_element_that_carries_a_reference_still_counts() {
    let src = "\
struct Link { p: &mut I32 }
@unsafe fn bump_link(link: Link) { *link.p = -1; }
fn touch(a: &mut [Link; 1]) -> I32 {
    let x: I32 = 0;
    @invariant(x >= 0)
    for i in 0..1 {
        bump_link(a[0]);
    }
    return x;
}
fn main() -> I32 { return 0; }
";
    let o = build_and_run("element_reference", src);
    assert!(!o.built, "an element carrying a reference was not counted:\n{}", o.compiler);
    assert!(o.compiler.contains("passes a reference to a call"), "{}", o.compiler);
}

/// Under `@unsafe` an unprovable index is not refused, and it is checked when
/// the program runs: index 9 into 4 elements stops the program. Index 3 is the
/// control - the check must not fire in bounds.
#[test]
fn under_unsafe_it_is_checked_at_run_time() {
    let prog = |k: i32| {
        format!(
            "\
@unsafe
fn poke(a: &mut [I16; 4], k: I32) -> I32 {{
    a[k] = 7;
    return a[k];
}}

fn main() -> I32 {{
    let v: [I16; 4] = {{}};
    return poke(&mut v, {k});
}}
"
        )
    };
    let o = build_and_run("unsafe_out", &prog(9));
    if skip_without_clang(&o, "under_unsafe_it_is_checked_at_run_time") {
        return;
    }
    assert!(o.built, "{}", o.compiler);
    let (code, out) = o.run.expect("the program was killed by a signal");
    assert_eq!(code, 1, "index 9 into 4 elements did not stop the program:\n{}", out);
    assert!(out.contains("Index out of bounds panic: index 9, array size 4"), "{}", out);

    let o = build_and_run("unsafe_in", &prog(3));
    assert!(o.built, "{}", o.compiler);
    let (code, out) = o.run.expect("the program was killed by a signal");
    assert_eq!(code, 7, "index 3 is in bounds:\n{}", out);
    assert!(!out.contains("out of bounds"), "{}", out);
}
