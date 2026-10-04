//! `tools/ydb/ydb`: debugging a Y program as Y, over gdb.
//!
//! ydb builds the program with `-g` and starts gdb with the Y commands of
//! `tools/ydb/ydb_gdb.py`: `break` on Y locations (`kernel:42`), `locals`
//! with Y types, `tensor` for a buffer, and `asm` for the code a line became
//! - this process's machine code, or, through `Y --emit-ptx --lineinfo`, the
//! PTX and SASS of a kernel's line. These tests drive it in batch mode and
//! read what gdb printed.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok()
}

fn skip(test: &str, tools: &str) {
    eprintln!("SKIP {}: no {} on this machine, so this test checked NOTHING", test, tools);
}

/// The line `// L:<tag>` marks, 1-based.
fn line_of(src: &str, tag: &str) -> usize {
    let mark = format!("// L:{}", tag);
    let hits: Vec<usize> = src.lines().enumerate().filter(|(_, l)| l.contains(&mark)).map(|(i, _)| i + 1).collect();
    assert_eq!(hits.len(), 1, "marker {} must occur once", mark);
    hits[0]
}

const PROG: &str = "\
struct Point {
    x: I32,
    y: F32,
}

fn scale(p: Point, k: I32) -> I32 {
    let s: I32 = p.x * k; // L:scale
    return s; // L:ret
}

kernel bump(Out: GlobalMemory<F32>, N: I32) {
    @invariant(i >= 0)
    for i in 0..N {
        Out[i] = Out[i] * 2.0 + 1.0; // L:store
    }
    let done: I32 = 1; // L:done
}

@unsafe
fn main() -> I32 {
    let v: [F32; 6] = {};
    v[1] = 1.5;
    v[4] = -2.0;
    let w: [U8; 4] = {};
    w[0] = 200;
    let name: String = String_new(\"hello\");
    let p: Point = Point { x: 5, y: 1.25 };
    let a: I32 = 7;
    let r: I32 = scale(p, a); // L:call
    if r > 0 {
        let a: I32 = 9;
        print_int(a); // L:inner
    }
    bump(v, 6);
    return 0;
}
";

/// A pinned directory holding `prog.ysu`.
fn setup(tag: &str, src: &str) -> (PathBuf, PathBuf) {
    let dir = pinned::pinned_scratch(&format!("ydb_{}", tag), pinned::SM_PINNED);
    let path = dir.join("prog.ysu");
    fs::write(&path, src).expect("write source");
    (dir, path)
}

/// `ydb prog.ysu --batch -ex ...`: everything it printed.
fn ydb(dir: &Path, program: &Path, cmds: &[String]) -> String {
    let mut c = Command::new("python3");
    c.arg(pinned::repo().join("tools/ydb/ydb"))
        .arg(program)
        .arg("--y")
        .arg(env!("CARGO_BIN_EXE_Y"))
        .args(["--nx", "--batch"])
        .current_dir(dir);
    for cmd in cmds {
        c.arg("-ex").arg(cmd);
    }
    let out = c.output().expect("run ydb");
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn ready(test: &str, gpu: bool) -> bool {
    let mut missing = Vec::new();
    for t in ["python3", "gdb", "clang"] {
        if !have(t) {
            missing.push(t);
        }
    }
    if gpu {
        for t in ["ptxas", "nvdisasm"] {
            if !have(t) {
                missing.push(t);
            }
        }
    }
    if missing.is_empty() {
        return true;
    }
    skip(test, &missing.join("/"));
    false
}

/// `break NAME:LINE` takes a Y location: NAME is a file's stem or a
/// function, and a condition after `if` is kept.
///
/// Plain gdb reads `scale:8` as the FUNCTION `scale` and drops the 8: the
/// breakpoint lands on the function's first line, silently. And `main:N` also
/// lands in the C runtime's `main`. So the lines below are deliberately NOT a
/// function's first line - on that one the two readings agree.
#[test]
fn break_takes_a_y_location() {
    if !ready("break_takes_a_y_location", false) {
        return;
    }
    let (dir, prog) = setup("break", PROG);
    let inner = line_of(PROG, "inner");
    let ret = line_of(PROG, "ret");
    let out = ydb(
        &dir,
        &prog,
        &[
            format!("break prog:{}", inner),
            format!("break scale:{}", ret),
            format!("break main:{}", inner),
            "run".into(),
            "kill".into(),
        ],
    );
    assert!(out.contains(&format!("Breakpoint 1 at ")) && out.contains(&format!("file prog.ysu, line {}.", inner)), "`break prog:N` is the file prog.ysu:\n{}", out);
    assert!(out.contains(&format!("file prog.ysu, line {}.", ret)), "`break scale:N` is line N of the file scale is in, not scale's first line:\n{}", out);
    assert!(!out.contains("(2 locations)"), "`break main:N` reached the C runtime's main too:\n{}", out);
    assert!(out.contains(&format!("scale (p=..., k=7) at prog.ysu:{}", ret)), "{}", out);
    // gdb never stops to ask about distribution debug information.
    assert!(!out.contains("Enable debuginfod"), "{}", out);
    let shown = ydb(&dir, &prog, &["show prompt".into()]);
    assert!(shown.contains("\"(ydb) \""), "the prompt: {}", shown);
    // A condition is kept: k is 7, so `k == 8` never stops.
    let out = ydb(&dir, &prog, &["break scale if k == 8".into(), "run".into()]);
    assert!(!out.contains("Breakpoint 1, scale"), "the condition was dropped:\n{}", out);
    assert!(out.contains("exited normally") || out.contains("exited with code"), "{}", out);
    let _ = fs::remove_dir_all(&dir);
}

/// `locals` lists the arguments and the bindings in scope with their Y
/// types, and not a binding an inner `let` shadows.
#[test]
fn locals_shows_what_is_in_scope_with_y_types() {
    if !ready("locals_shows_what_is_in_scope_with_y_types", false) {
        return;
    }
    let (dir, prog) = setup("locals", PROG);
    let out = ydb(
        &dir,
        &prog,
        &[
            "break scale".into(),
            format!("break prog.ysu:{}", line_of(PROG, "inner")),
            "run".into(),
            "echo @@scale\\n".into(),
            "locals".into(),
            "continue".into(),
            "echo @@inner\\n".into(),
            "locals".into(),
            "kill".into(),
        ],
    );
    let scale = out.split("@@scale").nth(1).and_then(|s| s.split("@@inner").next()).unwrap_or("");
    assert!(scale.contains("(arg) k: I32 = 7"), "{}", out);
    assert!(scale.contains("(arg) p: Point = {x = 5, y = 1.25}"), "{}", out);
    assert!(!scale.contains("s: I32"), "`s` is listed before its `let` ran:\n{}", out);
    let inner = out.split("@@inner").nth(1).unwrap_or("");
    assert!(inner.contains("a: I32 = 9"), "{}", out);
    assert!(!inner.contains("a: I32 = 7"), "the shadowed `a` is listed:\n{}", out);
    assert!(inner.contains("r: I32 = 35"), "{}", out);
    assert!(inner.contains("v: [F32; 6] = {0, 1.5, 0, 0, -2, 0}"), "{}", out);
    // Through the extension the binary embeds - which runs only because ydb
    // puts the program in gdb's auto-load safe path.
    assert!(inner.contains("name: String = \"hello\""), "{}", out);
    let _ = fs::remove_dir_all(&dir);
}

/// `tensor` prints a buffer's elements and statistics: an array's length is
/// its own, a pointer's must be given.
#[test]
fn tensor_prints_elements_and_statistics() {
    if !ready("tensor_prints_elements_and_statistics", false) {
        return;
    }
    let (dir, prog) = setup("tensor", PROG);
    let out = ydb(
        &dir,
        &prog,
        &[
            format!("break prog.ysu:{}", line_of(PROG, "inner")),
            format!("break prog.ysu:{}", line_of(PROG, "done")),
            "run".into(),
            "tensor v".into(),
            "tensor w".into(),
            "continue".into(),
            "tensor Out".into(),
            "tensor Out 6".into(),
            "kill".into(),
        ],
    );
    assert!(out.contains("v: 6 x F32"), "{}", out);
    assert!(out.contains("[1] 1.5") && out.contains("[4] -2"), "{}", out);
    assert!(out.contains("min -2, max 1.5") && out.contains("zeros 4"), "{}", out);
    assert!(out.contains("w: 4 x U8") && out.contains("[0] 200"), "a U8 read as unsigned:\n{}", out);
    assert!(out.contains("Out is a pointer, so its length is not known here: tensor Out COUNT"), "{}", out);
    // After `bump`: 2x + 1 of {0, 1.5, 0, 0, -2, 0}.
    assert!(out.contains("Out: 6 x F32"), "{}", out);
    assert!(out.contains("[0] 1  [1] 4  [2] 1  [3] 1  [4] -3  [5] 1"), "the kernel's buffer read back:\n{}", out);
    let _ = fs::remove_dir_all(&dir);
}

/// `asm` is the machine code of the current line, and `asm --ptx` /
/// `asm --sass` the GPU code a kernel's line became.
#[test]
fn asm_shows_the_code_a_line_became() {
    if !ready("asm_shows_the_code_a_line_became", true) {
        return;
    }
    let (dir, prog) = setup("asm", PROG);
    let store = line_of(PROG, "store");
    let out = ydb(
        &dir,
        &prog,
        &[
            "break scale".into(),
            "run".into(),
            "echo @@host\\n".into(),
            "asm".into(),
            "echo @@ptx\\n".into(),
            format!("asm --ptx {}", store),
            "echo @@sass\\n".into(),
            format!("asm --sass {}", store),
            "echo @@hostline\\n".into(),
            format!("asm --ptx {}", line_of(PROG, "call")),
            "kill".into(),
        ],
    );
    let part = |a: &str, b: &str| out.split(a).nth(1).and_then(|s| s.split(b).next()).unwrap_or("").to_string();
    let host = part("@@host\n", "@@ptx");
    assert!(host.contains("Dump of assembler code") && host.contains("imul"), "the line `p.x * k`:\n{}", out);
    let ptx = part("@@ptx\n", "@@sass");
    assert!(ptx.contains("kernel bump") && ptx.contains("st.global.f32") && ptx.contains("fma.rn.f32"), "{}", out);
    let sass = part("@@sass\n", "@@hostline");
    assert!(sass.contains("STG.E") && sass.contains("FFMA"), "{}", out);
    assert!(out.contains(&format!("prog.ysu:{} is not in a kernel's code: it becomes no PTX", line_of(PROG, "call"))), "{}", out);
    let _ = fs::remove_dir_all(&dir);
}

/// A program whose kernel uses a GPU intrinsic does not build for the host,
/// so there is no process; ydb says so and still answers the GPU questions.
#[test]
fn a_gpu_only_program_still_has_its_gpu_commands() {
    if !ready("a_gpu_only_program_still_has_its_gpu_commands", true) {
        return;
    }
    let src = "\
kernel saxpy(X: GlobalMemory<F32>, Yv: GlobalMemory<F32>, N: I32) {
    let i: I32 = thread_idx_x(); // L:tid
    if i < N {
        Yv[i] = 2.0 * X[i] + Yv[i];
    }
}

fn main() {}
";
    let (dir, prog) = setup("gpuonly", src);
    let out = ydb(&dir, &prog, &[format!("asm --sass {}", line_of(src, "tid"))]);
    assert!(out.contains("does not build for the host, so there is no process to debug"), "{}", out);
    assert!(out.contains("S2R") && out.contains("SR_TID.X"), "the SASS of `thread_idx_x()`:\n{}", out);
    let _ = fs::remove_dir_all(&dir);
}
