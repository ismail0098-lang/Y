//! `-g`: DWARF debug information from the LLVM backend, and `--debug`.
//!
//! A debugger that shows the wrong value is worse than none - it is believed.
//! So these tests do not stop at "the module carries metadata": they build a
//! program with `-g` and drive real gdb against it, and every assertion is
//! about what gdb SHOWS - the line it stops on, the value it prints - checked
//! against what the source says.
//!
//! What each part pins:
//!
//! * Without `-g` nothing changes: no marker, no metadata. (The corpus was
//!   also compared byte for byte against the parent commit's compiler when
//!   this was written - 55 of 55 modules identical; that comparison needs a
//!   second compiler and is recorded in CLAUDE.md rather than run here.)
//! * With `-g` every corpus program the backend accepts still verifies under
//!   `opt`, and no marker line survives.
//! * gdb stops on Y lines, prints Y values with their Y types, steps into a
//!   call with its arguments already stored, visits each statement of a loop
//!   once per iteration, finds an imported function in its own file, and
//!   `--debug` starts on the first line of `fn main`.
//! * A `-g` build behaves like a plain one.
//! * Every backend that produces no debug information refuses `-g` by name.
//!
//! **The tool-dependent tests SKIP, loudly, when the tool is absent** (`gdb`,
//! `opt`, `llvm-dwarfdump`, `clang`). Where the tool is present they run in
//! full; nothing here passes vacuously because a fixture failed to build -
//! a build failure is a FAILURE.
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[path = "common/pinned.rs"]
mod pinned;

/// Names files written inside one directory, so two of them never collide.
static SALT: AtomicUsize = AtomicUsize::new(0);

/// A fresh directory holding a pinned hardware profile. Every compile here
/// runs in one, so it neither reads this machine's `.ysu_hw_profile` nor
/// writes one into the repository - and a `@ZeroDrift` fixture gets the same
/// representation on every machine, which a measured profile does not promise
/// (`common/pinned.rs`). Unique per call, whatever the tag.
fn scratch(tag: &str) -> PathBuf {
    pinned::pinned_scratch(&format!("dbginfo_{}", tag), pinned::SM_PINNED)
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok()
}

fn skip(test: &str, tool: &str) {
    eprintln!("SKIP {}: `{}` is not installed, so this test checked NOTHING", test, tool);
}

/// The compiler, run in `dir` - a [`scratch`] directory, so the profile it
/// reads is the pinned one.
fn y(dir: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_Y"));
    c.current_dir(dir);
    c
}

/// Run `cmd` with stdout and stderr captured to files, killing it after
/// `secs`. A debugger waiting for input it will never get would otherwise
/// hang the suite rather than fail it.
fn run_with_deadline(cmd: &mut Command, dir: &Path, secs: u64) -> (Option<i32>, String) {
    let out_path = dir.join(format!("out_{}.txt", SALT.fetch_add(1, Ordering::SeqCst)));
    let out = fs::File::create(&out_path).expect("output file");
    let err = out.try_clone().expect("output file");
    let mut child = cmd.stdout(out).stderr(err).spawn().expect("spawn");
    let deadline = Instant::now() + Duration::from_secs(secs);
    let status = loop {
        match child.try_wait().expect("wait") {
            Some(s) => break s.code(),
            None if Instant::now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "deadline of {}s passed; output so far:\n{}",
                    secs,
                    fs::read_to_string(&out_path).unwrap_or_default()
                );
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    };
    (status, fs::read_to_string(&out_path).unwrap_or_default())
}

fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

/// Writes `src` as `<dir>/<name>.ysu` and compiles it to `<dir>/<name>`.
fn build(dir: &Path, name: &str, src: &str, g: bool) -> PathBuf {
    let path = dir.join(format!("{}.ysu", name));
    fs::write(&path, src).expect("write source");
    let bin = dir.join(if g { format!("{}_g", name) } else { name.to_string() });
    let mut c = y(dir);
    c.arg(&path).arg("-o").arg(&bin);
    if g {
        c.arg("-g");
    }
    let out = c.output().expect("run Y");
    assert!(
        out.status.success() && bin.exists(),
        "building {} {} failed:\n{}",
        name,
        if g { "with -g" } else { "without -g" },
        text(&out)
    );
    bin
}

/// `--emit-llvm`, with or without `-g`; the module text.
fn emit_ll(dir: &Path, name: &str, src: &str, g: bool) -> String {
    let path = dir.join(format!("{}.ysu", name));
    fs::write(&path, src).expect("write source");
    let ll = dir.join(format!("{}{}.ll", name, if g { "_g" } else { "" }));
    let mut c = y(dir);
    c.arg(&path).arg("--emit-llvm").arg("-o").arg(&ll);
    if g {
        c.arg("-g");
    }
    let out = c.output().expect("run Y");
    assert!(out.status.success(), "emitting {} failed:\n{}", name, text(&out));
    fs::read_to_string(&ll).expect("read module")
}

/// gdb in batch mode, reading no user configuration, on a script. The Y
/// extension embedded in the program is NOT loaded - the program is outside
/// gdb's auto-load safe path - so this is the plain DWARF view.
fn gdb(dir: &Path, bin: &Path, script: &str) -> String {
    gdb_session(dir, bin, script, false)
}

/// The same with the program in gdb's auto-load safe path, so gdb runs the
/// Y extension embedded in it, as `Y prog.ysu --debug` arranges.
fn gdb_y(dir: &Path, bin: &Path, script: &str) -> String {
    gdb_session(dir, bin, script, true)
}

fn gdb_session(dir: &Path, bin: &Path, script: &str, extension: bool) -> String {
    let file = dir.join(format!("script_{}.gdb", SALT.fetch_add(1, Ordering::SeqCst)));
    fs::write(&file, format!("set debuginfod enabled off\nset pagination off\n{}", script))
        .expect("write script");
    let mut c = Command::new("gdb");
    c.args(["-nx", "-q", "-batch"]);
    if extension {
        let abs = fs::canonicalize(bin).expect("canonical program path");
        c.arg("-iex").arg(format!("add-auto-load-safe-path {}", abs.display()));
    }
    c.arg("-x").arg(&file).arg(bin).stdin(Stdio::null()).current_dir(dir);
    run_with_deadline(&mut c, dir, 120).1
}

/// The line carrying `// L:<tag>` in `src`, 1-based.
fn line_of(src: &str, tag: &str) -> usize {
    let needle = format!("// L:{}", tag);
    let hits: Vec<usize> = src
        .lines()
        .enumerate()
        .filter(|(_, l)| l.trim_end().ends_with(&needle))
        .map(|(i, _)| i + 1)
        .collect();
    assert_eq!(hits.len(), 1, "fixture tag {} must mark exactly one line", tag);
    hits[0]
}

/// Lines gdb reported stopping on, in order: the `N\t<source>` lines it
/// prints after a stop or a step.
fn stop_lines(out: &str) -> Vec<usize> {
    out.lines()
        .filter_map(|l| {
            let (n, rest) = l.split_once('\t')?;
            if rest.is_empty() {
                return None;
            }
            n.trim().parse::<usize>().ok()
        })
        .collect()
}

const FIXTURE: &str = r#"// Every line a test refers to carries an `L:` tag.
struct Point {
    x: I32,
    y: F64,
    flag: bool,
}

enum Color {
    Red,
    Green,
    Blue,
}

fn scale(p: Point, k: I32) -> I32 {
    let s: I32 = p.x * k; // L:scale_body
    return s;
}

fn twice(x: I32) -> I32 {
    return x * 2;
}

fn note(x: I32) {
    let unused: I32 = x + 1; // L:note_last
}

kernel fill(n: I32) {
    let t: I32 = n * 3; // L:kernel_last
}

@unsafe
fn main() -> I32 {
    let a: I32 = 7; // L:first
    let big: I64 = 5000000000;
    let u: U32 = 4000000000;
    let f: F64 = 2.5;
    let ok: bool = a > 3;
    let ch: char = 'A';
    let c: Color = Color::Blue;
    let p: Point = Point { x: 5, y: 1.25, flag: true };
    let v: [I32; 3] = {};
    v[0] = 10;
    v[1] = 20;
    let name: String = "hello";
    let w = 9;
    let greet = String_new("x");
    let total: I32 = 0; // L:values
    let k: I32 = 0;
    while k < 3 { // L:while
        if k == 1 { // L:if
            total = total + 10; // L:then
        } else {
            total = total + 1; // L:else
        }
        k = k + 1; // L:inc
    }
    let sum: I32 = 0; // L:after_while
    for i in 0..3 { // L:for
        sum = sum + v[i]; // L:for_a
        sum = sum + 1; // L:for_b
    }
    let r: I32 = scale(p, a); // L:call
    let r2: I32 = twice(r);
    note(r2); // L:note_call
    fill(3); // L:after_note
    print_int(r2); // L:after_fill
    return total + sum + r; // L:ret
}
"#;

// ── The module ───────────────────────────────────────────────

#[test]
fn without_g_the_module_carries_no_debug_information() {
    let dir = scratch("plain");
    let ll = emit_ll(&dir, "fixture", FIXTURE, false);
    for needle in ["!dbg", "llvm.dbg", "DICompileUnit", "DILocalVariable", ";@y.dbg"] {
        assert!(!ll.contains(needle), "a module built without -g contains `{}`", needle);
    }
}

#[test]
fn with_g_every_function_and_variable_is_described() {
    let dir = scratch("described");
    let ll = emit_ll(&dir, "fixture", FIXTURE, true);
    assert!(!ll.contains(";@y.dbg"), "a location or variable marker survived into the module");
    assert!(ll.contains("!llvm.dbg.cu"), "no compile unit");
    for f in ["main", "scale", "twice", "note", "fill"] {
        assert!(
            ll.contains(&format!("DISubprogram(name: \"{}\"", f)),
            "no subprogram named {}",
            f
        );
    }
    for v in [
        "a", "big", "u", "f", "ok", "ch", "c", "p", "v", "name", "w", "greet", "total", "k", "sum", "i",
        "r", "s", "unused", "t",
    ] {
        assert!(
            ll.contains(&format!("DILocalVariable(name: \"{}\"", v)),
            "variable {} is not described",
            v
        );
    }
    // `x` is the first parameter of the SECOND function with parameters:
    // numbering restarts per function.
    for (v, arg) in [("p", 1), ("k", 2), ("x", 1), ("n", 1)] {
        assert!(
            ll.contains(&format!("DILocalVariable(name: \"{}\", arg: {},", v, arg)),
            "parameter {} is not argument {}",
            v,
            arg
        );
    }
    // `fn main` is the symbol `ysu_main`: the C runtime owns the process's
    // `main`. The debugger is told it is `main`.
    assert!(ll.contains("define i32 @ysu_main() #0 !dbg !"), "ysu_main has no subprogram attached");
    if have("opt") {
        let path = dir.join("fixture_g.ll");
        let out = Command::new("opt")
            .args(["-passes=verify", "-disable-output"])
            .arg(&path)
            .output()
            .expect("run opt");
        assert!(out.status.success(), "the LLVM verifier rejects the -g module:\n{}", text(&out));
    } else {
        skip("with_g_every_function_and_variable_is_described (verifier half)", "opt");
    }
}

/// Every program in `tests/` the LLVM backend accepts must still be accepted
/// with `-g`, and its module must pass the LLVM verifier - which checks the
/// debug metadata as well as the code. Compiles COPIES: `--emit-llvm` writes
/// beside its input.
#[test]
fn every_corpus_program_the_backend_accepts_still_verifies_with_g() {
    if !have("opt") {
        return skip("every_corpus_program_the_backend_accepts_still_verifies_with_g", "opt");
    }
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut sources: Vec<PathBuf> = fs::read_dir(&tests)
        .expect("tests/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "ysu").unwrap_or(false))
        .collect();
    sources.sort();
    let dir = scratch("corpus");
    let mut checked = 0;
    let mut failures = Vec::new();
    for src in &sources {
        let name = src.file_stem().unwrap().to_string_lossy().into_owned();
        let copy = dir.join(format!("{}.ysu", name));
        fs::copy(src, &copy).expect("copy source");
        let plain = dir.join(format!("{}.ll", name));
        let accepted = y(&dir).arg(&copy).arg("--emit-llvm").arg("-o").arg(&plain).output().expect("run Y");
        if !accepted.status.success() {
            continue;
        }
        let with_g = dir.join(format!("{}_g.ll", name));
        let out = y(&dir).arg(&copy).arg("--emit-llvm").arg("-g").arg("-o").arg(&with_g).output().expect("run Y");
        if !out.status.success() {
            failures.push(format!("{}: refused with -g only:\n{}", name, text(&out)));
            continue;
        }
        let module = fs::read_to_string(&with_g).expect("read module");
        if module.lines().any(|l| l.starts_with(";@y.dbg")) {
            failures.push(format!("{}: a marker survived", name));
        }
        if !module.contains("!llvm.dbg.cu") {
            failures.push(format!("{}: no debug information", name));
        }
        let v = Command::new("opt")
            .args(["-passes=verify", "-disable-output"])
            .arg(&with_g)
            .output()
            .expect("run opt");
        if !v.status.success() {
            failures.push(format!("{}: verifier: {}", name, text(&v)));
        }
        checked += 1;
    }
    assert!(failures.is_empty(), "{} of {} failed:\n{}", failures.len(), checked, failures.join("\n"));
    // A sweep that compiled nothing would report no failures perfectly.
    assert!(checked >= 40, "only {} corpus programs were accepted - the sweep is not testing the corpus", checked);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_line_table_names_the_y_source_and_its_lines() {
    if !have("llvm-dwarfdump") {
        return skip("the_line_table_names_the_y_source_and_its_lines", "llvm-dwarfdump");
    }
    let dir = scratch("lines");
    let bin = build(&dir, "fixture", FIXTURE, true);
    let out = Command::new("llvm-dwarfdump").arg("--debug-line").arg(&bin).output().expect("dwarfdump");
    let table = text(&out);
    assert!(table.contains("fixture.ysu"), "the line table does not name fixture.ysu:\n{}", table);
    let rows: Vec<usize> = table
        .lines()
        .filter(|l| l.starts_with("0x"))
        .filter_map(|l| l.split_whitespace().nth(1)?.parse().ok())
        .collect();
    for tag in ["scale_body", "first", "while", "if", "then", "else", "inc", "for", "for_a", "for_b", "call", "ret"] {
        let line = line_of(FIXTURE, tag);
        assert!(rows.contains(&line), "no line-table row for line {} (`{}`)", line, tag);
    }
}

// ── gdb ──────────────────────────────────────────────────────

#[test]
fn gdb_prints_y_values_with_y_types() {
    if !have("gdb") {
        return skip("gdb_prints_y_values_with_y_types", "gdb");
    }
    let dir = scratch("values");
    let bin = build(&dir, "fixture", FIXTURE, true);
    let out = gdb(
        &dir,
        &bin,
        &format!(
            "break fixture.ysu:{}\nrun\ninfo locals\nprint *name\nwhatis u\nptype p\nprint v[1]\nprint p.y * 2\n\
             whatis w\nwhatis greet\nkill\n",
            line_of(FIXTURE, "values")
        ),
    );
    let at = format!(", main () at fixture.ysu:{}", line_of(FIXTURE, "values"));
    assert!(out.contains(&at), "did not stop at `{}`:\n{}", at, out);
    for want in [
        "a = 7",
        // above 2^32: the slot is described at its real 64-bit width
        "big = 5000000000",
        // above 2^31: printed unsigned, because the type says U32
        "u = 4000000000",
        "f = 2.5",
        "ok = true",
        "ch = 65 'A'",
        "c = Blue",
        "p = {x = 5, y = 1.25, flag = true}",
        "v = {10, 20, 0}",
        "data = ",
        "\"hello\", len = 5",
        "type = U32",
        "F64 y;",
        // Unannotated: the emitter infers `i64` for the literal and stores it
        // in an `i32` slot. Described at the inferred width, gdb would read
        // four bytes past the slot.
        "w = 9",
        "type = I32",
        // Unannotated: inferred from the call, which only says `ptr`.
        "type = String",
        "= 20",
        "= 2.5",
    ] {
        assert!(out.contains(want), "gdb did not show `{}`:\n{}", want, out);
    }
}

#[test]
fn stepping_into_a_call_finds_its_arguments_already_stored() {
    if !have("gdb") {
        return skip("stepping_into_a_call_finds_its_arguments_already_stored", "gdb");
    }
    let dir = scratch("step");
    let bin = build(&dir, "fixture", FIXTURE, true);
    let call = line_of(FIXTURE, "call");
    let body = line_of(FIXTURE, "scale_body");
    let out = gdb(
        &dir,
        &bin,
        &format!("break fixture.ysu:{}\nrun\nstep\ninfo args\nbt\nfinish\nkill\n", call),
    );
    // `step` must land on the callee's first statement, not its `fn` line: a
    // stop before the parameter spills shows every argument as whatever its
    // slot held before. That is what a producer gdb does not recognise as
    // LLVM gets (see `debug_info::DebugInfo::finish`).
    let callee = format!("scale (p=..., k=7) at fixture.ysu:{}", body);
    assert!(out.contains(&callee), "step did not stop at `{}`:\n{}", callee, out);
    assert!(out.contains("p = {x = 5, y = 1.25, flag = true}"), "argument p:\n{}", out);
    assert!(out.contains("k = 7"), "argument k:\n{}", out);
    assert!(out.contains(&format!("in main () at fixture.ysu:{}", call)), "caller frame:\n{}", out);
    assert!(out.contains("Value returned is $") && out.contains(" = 35"), "finish:\n{}", out);
}

#[test]
fn a_function_breakpoint_stops_after_the_parameters_are_stored() {
    if !have("gdb") {
        return skip("a_function_breakpoint_stops_after_the_parameters_are_stored", "gdb");
    }
    let dir = scratch("fnbreak");
    let bin = build(&dir, "fixture", FIXTURE, true);
    let out = gdb(&dir, &bin, "break scale\nrun\ninfo args\nkill\n");
    let at = format!("scale (p=..., k=7) at fixture.ysu:{}", line_of(FIXTURE, "scale_body"));
    assert!(out.contains(&at), "break scale did not stop at `{}`:\n{}", at, out);
    assert!(out.contains("k = 7") && out.contains("p = {x = 5"), "arguments:\n{}", out);
}

/// One stop per statement per iteration. Two things used to add stops: the
/// jump out of a `then` branch attributed to the `if` (a second stop on the
/// `if` line), and a `while` back edge attributed to the `while` (a second
/// stop on the loop line every iteration).
#[test]
fn next_visits_each_statement_of_a_loop_once_per_iteration() {
    if !have("gdb") {
        return skip("next_visits_each_statement_of_a_loop_once_per_iteration", "gdb");
    }
    let dir = scratch("loop");
    let bin = build(&dir, "fixture", FIXTURE, true);
    let l = |t| line_of(FIXTURE, t);
    let out = gdb(
        &dir,
        &bin,
        &format!("break fixture.ysu:{}\nrun\n{}kill\n", l("while"), "next\n".repeat(13)),
    );
    let want = vec![
        l("while"), // the breakpoint
        l("if"), l("else"), l("inc"), l("while"),
        l("if"), l("then"), l("inc"), l("while"),
        l("if"), l("else"), l("inc"), l("while"),
        l("after_while"),
    ];
    assert_eq!(stop_lines(&out), want, "gdb output:\n{}", out);
}

/// A `for` loop's increment and back edge are its header's code, so they
/// belong to the `for` line: the stop after the last body statement is there.
#[test]
fn a_for_loop_increment_belongs_to_the_for_line() {
    if !have("gdb") {
        return skip("a_for_loop_increment_belongs_to_the_for_line", "gdb");
    }
    let dir = scratch("for");
    let bin = build(&dir, "fixture", FIXTURE, true);
    let l = |t| line_of(FIXTURE, t);
    let out = gdb(
        &dir,
        &bin,
        &format!("break fixture.ysu:{}\nrun\nnext\nnext\nnext\nprint sum\nprint i\nkill\n", l("for_b")),
    );
    assert_eq!(stop_lines(&out), vec![l("for_b"), l("for"), l("for_a"), l("for_b")], "gdb output:\n{}", out);
    // Second iteration, after `sum = sum + v[0]; sum = sum + 1; sum = sum + v[1];`.
    assert!(out.contains("= 31") && out.contains("= 1"), "loop state:\n{}", out);
}

/// Stepping cannot see where the increment is attributed - the condition
/// block is on the `for` line either way, so `next` stops there - but the line
/// table can, and so can anything that reads it (a profiler charging the
/// increment to the last body line, `info line`, a disassembly view). Every
/// store to the loop variable must carry the `for` line.
#[test]
fn a_for_loop_increment_carries_the_for_line_in_the_module() {
    let dir = scratch("for_ir");
    let ll = emit_ll(&dir, "fixture", FIXTURE, true);
    let for_line = line_of(FIXTURE, "for");
    let locs: std::collections::HashMap<String, usize> = ll
        .lines()
        .filter_map(|l| {
            let (id, rest) = l.split_once(" = !DILocation(line: ")?;
            Some((id.to_string(), rest.split(',').next()?.parse().ok()?))
        })
        .collect();
    let stores: Vec<&str> = ll
        .lines()
        .filter(|l| l.trim_start().starts_with("store ") && l.contains(", ptr %i, !dbg !"))
        .collect();
    // The initial store and the increment's.
    assert_eq!(stores.len(), 2, "stores to the loop variable:\n{}", stores.join("\n"));
    for st in stores {
        let id = st.rsplit("!dbg ").next().unwrap().trim();
        assert_eq!(locs.get(id), Some(&for_line), "`{}` is not attributed to the `for` line {}", st, for_line);
    }
}

/// The implicit return at the end of a function without one belongs to its
/// last statement, so `next` from there goes straight back to the caller.
/// Attributed to the function's own line, it was one more stop, on the `fn`
/// header, on the way out.
#[test]
fn next_from_a_functions_last_line_returns_to_the_caller() {
    if !have("gdb") {
        return skip("next_from_a_functions_last_line_returns_to_the_caller", "gdb");
    }
    let dir = scratch("epilogue");
    let bin = build(&dir, "fixture", FIXTURE, true);
    let l = |t| line_of(FIXTURE, t);
    let out = gdb(
        &dir,
        &bin,
        &format!(
            "break fixture.ysu:{}\nbreak fixture.ysu:{}\nrun\nnext\ncontinue\nnext\nbt 1\nkill\n",
            l("note_last"),
            l("kernel_last")
        ),
    );
    // A function, then a kernel: each `next` from the last line lands in main.
    assert_eq!(
        stop_lines(&out),
        vec![l("note_last"), l("after_note"), l("kernel_last"), l("after_fill")],
        "gdb output:\n{}",
        out
    );
    assert!(out.contains("#0  main () at fixture.ysu:"), "did not return to main:\n{}", out);
}

#[test]
fn a_zero_drift_accumulator_is_shown_as_its_raw_representation() {
    if !have("gdb") {
        return skip("a_zero_drift_accumulator_is_shown_as_its_raw_representation", "gdb");
    }
    let src = "@unsafe\nfn main() -> I32 {\n    @ZeroDrift @bounds(-1000, 1000)\n    let acc: F32 = 0.0;\n    acc += 1.5;\n    acc += 2.25;\n    let done: I32 = 1; // L:after\n    return done;\n}\n";
    let dir = scratch("drift");
    let bin = build(&dir, "drift", src, true);
    let out = gdb(&dir, &bin, &format!("break drift.ysu:{}\nrun\nwhatis acc\nkill\n", line_of(src, "after")));
    // The slot holds the exact representation the compiler chose, not the
    // value: a Q format is the value times 2^frac, and the type says which.
    let ty = out
        .lines()
        .find_map(|l| l.strip_prefix("type = "))
        .unwrap_or_else(|| panic!("no type for acc:\n{}", out))
        .to_string();
    let frac: u32 = ty
        .strip_suffix("_raw")
        .and_then(|q| q.strip_prefix('Q'))
        .and_then(|q| q.split_once('.'))
        .and_then(|(_, f)| f.parse().ok())
        .unwrap_or_else(|| panic!("acc's type `{}` does not name a Q format", ty));
    let out = gdb(
        &dir,
        &bin,
        &format!("break drift.ysu:{}\nrun\nprint acc / {}.0\nkill\n", line_of(src, "after"), 1u64 << frac),
    );
    assert!(out.contains("= 3.75"), "acc / 2^{} is not 3.75:\n{}", frac, out);
}

#[test]
fn an_imported_function_is_reported_in_its_own_file() {
    if !have("gdb") {
        return skip("an_imported_function_is_reported_in_its_own_file", "gdb");
    }
    let dir = scratch("import");
    let util = "// helper module\n\nfn helper(x: I32) -> I32 {\n    let y: I32 = x + 100; // L:helper\n    return y;\n}\n";
    fs::write(dir.join("util.ysu"), util).expect("write util");
    let app = "import util;\n\n@unsafe\nfn main() -> I32 {\n    let a: I32 = 5;\n    let b: I32 = helper(a); // L:call\n    return b;\n}\n";
    let bin = build(&dir, "app", app, true);
    let out = gdb(&dir, &bin, "break helper\nrun\nbt\nkill\n");
    // The imported function's spans are lines of util.ysu. Attributed to the
    // file being compiled, they would point at app.ysu's lines instead.
    let at = format!("helper (x=5) at util.ysu:{}", line_of(util, "helper"));
    assert!(out.contains(&at), "did not stop at `{}`:\n{}", at, out);
    let caller = format!("in main () at app.ysu:{}", line_of(app, "call"));
    assert!(out.contains(&caller), "caller frame `{}` missing:\n{}", caller, out);
}

/// `Y prog.ysu --debug` builds with `-g` and starts gdb on the first line of
/// `fn main`. gdb reads the commands piped to it, so this drives a real
/// session. HOME points at the scratch directory so no user gdbinit applies.
#[test]
fn debug_starts_gdb_on_the_first_line_of_main() {
    if !have("gdb") {
        return skip("debug_starts_gdb_on_the_first_line_of_main", "gdb");
    }
    let dir = scratch("launcher");
    let path = dir.join("fixture.ysu");
    fs::write(&path, FIXTURE).expect("write source");
    let commands = dir.join("commands.txt");
    fs::write(&commands, "info auto-load python-scripts\nnext\nprint a\ncontinue\n").expect("write commands");
    let mut c = y(&dir);
    c.arg(&path)
        .arg("--debug")
        .arg("-o")
        .arg(dir.join("fixture_dbg"))
        .env("HOME", &dir)
        .env("XDG_CONFIG_HOME", &dir)
        .stdin(fs::File::open(&commands).expect("commands"));
    let (status, out) = run_with_deadline(&mut c, &dir, 180);
    let first = line_of(FIXTURE, "first");
    // `, main ()`, not `main ()`: the second is a substring of `ysu_main ()`.
    let at = format!(", main () at fixture.ysu:{}", first);
    assert!(out.contains(&at), "--debug did not stop at `{}`:\n{}", at, out);
    assert!(out.contains("$1 = 7"), "`print a` after one `next`:\n{}", out);
    // The launcher put the program in gdb's safe path, so the Y extension
    // embedded in it is running in the session.
    assert!(
        out.lines().any(|l| l.contains("Yes") && l.contains("ysu-gdb-extension")),
        "--debug did not load the Y extension:\n{}",
        out
    );
    assert!(!out.contains("auto-loading has been declined"), "{}", out);
    assert!(out.contains("exited with code 0120"), "the program did not run to its end (exit 80):\n{}", out);
    assert_eq!(status, Some(0), "gdb's exit status:\n{}", out);
}

// ── Behaviour ────────────────────────────────────────────────

/// `-g` compiles at -O0 and adds metadata; the program must compute the same
/// thing. (A void `fn main` is not used: its exit status is whatever the
/// return register held, at any optimisation level - see CLAUDE.md.)
#[test]
fn a_g_build_behaves_like_a_plain_build() {
    if !have("clang") {
        return skip("a_g_build_behaves_like_a_plain_build", "clang");
    }
    let dir = scratch("behaviour");
    let plain = build(&dir, "fixture", FIXTURE, false);
    let debug = build(&dir, "fixture", FIXTURE, true);
    let a = Command::new(&plain).output().expect("run plain");
    let b = Command::new(&debug).output().expect("run -g");
    assert_eq!(a.status.code(), Some(80), "plain build:\n{}", text(&a));
    assert_eq!(b.status.code(), a.status.code(), "the -g build exits differently");
    assert_eq!(b.stdout, a.stdout, "the -g build prints differently");
    assert_eq!(String::from_utf8_lossy(&a.stdout).trim(), "70", "print_int(twice(35))");
}

// ── The flags ────────────────────────────────────────────────

#[test]
fn every_backend_without_debug_information_refuses_g_by_name() {
    let dir = scratch("refuse");
    let path = dir.join("fixture.ysu");
    fs::write(&path, FIXTURE).expect("write source");
    for backend in [
        "--emit-ptx",
        "--emit-cpu",
        "--emit-native",
        "--emit-coprocessor",
        "--target=r1cs",
        "--emit-zk-ptx",
    ] {
        for flag in ["-g", "--debug"] {
            let out = y(&dir).arg(&path).arg(backend).arg(flag).output().expect("run Y");
            let t = text(&out);
            assert!(!out.status.success(), "{} {} was accepted", backend, flag);
            assert!(
                t.contains(&format!("{} cannot be combined with {}", flag, backend)),
                "{} {} was not refused by name:\n{}",
                backend,
                flag,
                t
            );
        }
    }
    let out = y(&dir).arg(&path).arg("--emit-llvm").arg("--debug").output().expect("run Y");
    assert!(!out.status.success(), "--debug --emit-llvm was accepted");
    assert!(text(&out).contains("use -g --emit-llvm"), "{}", text(&out));
    // circom input produces R1CS, whatever backend flag accompanies it.
    let circom = dir.join("c.circom");
    fs::write(&circom, "template T() { signal input a; }\ncomponent main = T();\n").expect("write circom");
    let out = y(&dir).arg(&circom).arg("-g").output().expect("run Y");
    assert!(!out.status.success(), "-g on circom input was accepted");
    assert!(text(&out).contains("-g applies to Y source"), "{}", text(&out));
    // Debug information names the file it describes; the built-in harness
    // (no input file) has none.
    let out = y(&dir).arg("-g").output().expect("run Y");
    assert!(!out.status.success(), "-g without a source file was accepted");
    assert!(text(&out).contains("-g needs a source file"), "{}", text(&out));
    // The control: the LLVM backend accepts both spellings.
    let out = y(&dir).arg(&path).arg("--emit-llvm").arg("-g").arg("-o").arg(dir.join("ok.ll")).output().expect("run Y");
    assert!(out.status.success(), "-g --emit-llvm was refused:\n{}", text(&out));
}

/// Lexical scopes, seen from the debugger. A variable is visible from the
/// statement after its `let` to the end of its block, a shadowing binding
/// hides the outer one inside its scope only, and a loop variable exists in
/// its loop and not after it. Before this, every variable was visible in the
/// whole function (garbage before its `let`), and the shadowing `a` WAS the
/// outer one - one slot per name - so `a` read 2 after the block.
const SCOPES: &str = "\
@unsafe
fn main() -> I32 {
    let a: I32 = 1;
    let pre_v: I32 = 10; // L:pre
    if a > 0 {
        let a: I32 = 2;
        let inner_v: I32 = a + pre_v;
        print_int(inner_v); // L:in_block
    }
    let b_v: I32 = a + 5; // L:b
    for ki in 0..2 {
        let kt: I32 = ki * 2;
        print_int(kt); // L:in_loop
    }
    let after_v: I32 = b_v; // L:after
    return after_v; // L:ret
}
";

/// What `name` evaluates to at each stop, keyed by the `@@<stop>` sections
/// the script prints: `Ok(value)` or `Err(gdb's message)`. Asked through
/// Python so that a name gdb cannot see does not abort the batch script.
fn scope_views(out: &str) -> HashMap<String, HashMap<String, Result<String, String>>> {
    let mut views: HashMap<String, HashMap<String, Result<String, String>>> = HashMap::new();
    let mut stop = String::new();
    for l in out.lines() {
        if let Some(s) = l.strip_prefix("@@") {
            stop = s.trim().to_string();
        } else if let Some(rest) = l.strip_prefix("Y=") {
            let (name, v) = rest.split_once('=').expect("Y=name=value");
            views.entry(stop.clone()).or_default().insert(name.into(), Ok(v.into()));
        } else if let Some(rest) = l.strip_prefix("Y!") {
            let (name, e) = rest.split_once('!').expect("Y!name!error");
            views.entry(stop.clone()).or_default().insert(name.into(), Err(e.into()));
        }
    }
    views
}

#[test]
fn a_variable_is_visible_exactly_where_its_scope_is() {
    if !have("gdb") {
        return skip("a_variable_is_visible_exactly_where_its_scope_is", "gdb");
    }
    let dir = scratch("scopes");
    let bin = build(&dir, "scopes", SCOPES, true);
    let mut script = String::from(
        "python\n\
         def y_show(name):\n\
         \x20   try:\n\
         \x20       print('Y=%s=%s' % (name, gdb.parse_and_eval(name)))\n\
         \x20   except gdb.error as e:\n\
         \x20       print('Y!%s!%s' % (name, e))\n\
         end\n",
    );
    for tag in ["pre", "in_block", "b", "in_loop", "after", "ret"] {
        script.push_str(&format!("break scopes.ysu:{}\n", line_of(SCOPES, tag)));
    }
    script.push_str("run\n");
    let stops: [(&str, &[&str]); 7] = [
        ("pre", &["a", "pre_v"]),
        ("in_block", &["a", "pre_v", "inner_v", "b_v"]),
        ("b", &["a", "inner_v", "b_v"]),
        ("loop0", &["ki", "kt"]),
        ("loop1", &["ki", "kt"]),
        ("after", &["ki", "kt", "b_v", "after_v"]),
        ("ret", &["after_v"]),
    ];
    for (stop, names) in stops {
        script.push_str(&format!("echo @@{}\\n\n", stop));
        for n in names {
            script.push_str(&format!("python y_show('{}')\n", n));
        }
        script.push_str("continue\n");
    }
    let out = gdb(&dir, &bin, &script);
    let views = scope_views(&out);
    let val = |stop: &str, name: &str| -> Result<String, String> {
        views
            .get(stop)
            .and_then(|v| v.get(name))
            .cloned()
            .unwrap_or_else(|| panic!("no answer for `{}` at `{}`:\n{}", name, stop, out))
    };
    let unknown = |stop: &str, name: &str| {
        let got = val(stop, name);
        assert!(
            matches!(&got, Err(e) if e.contains("No symbol")),
            "`{}` must not be visible at `{}`, gdb gave {:?}:\n{}",
            name,
            stop,
            got,
            out
        );
    };
    let is = |stop: &str, name: &str, want: &str| {
        assert_eq!(val(stop, name), Ok(want.to_string()), "`{}` at `{}`:\n{}", name, stop, out);
    };
    // Before its `let` has run, a name does not exist yet.
    is("pre", "a", "1");
    unknown("pre", "pre_v");
    // Inside the block the inner `a` is the one in scope.
    is("in_block", "a", "2");
    is("in_block", "inner_v", "12");
    unknown("in_block", "b_v");
    // After the block the outer `a` is back, with ITS value.
    is("b", "a", "1");
    unknown("b", "inner_v");
    unknown("b", "b_v");
    is("loop0", "ki", "0");
    is("loop0", "kt", "0");
    is("loop1", "ki", "1");
    is("loop1", "kt", "2");
    // The loop's variables end with the loop.
    unknown("after", "ki");
    unknown("after", "kt");
    is("after", "b_v", "6");
    unknown("after", "after_v");
    is("ret", "after_v", "6");
    // The program computed what the source says: the inner `a` did not
    // overwrite the outer one, so `b_v` is 1 + 5.
    assert!(out.contains("exited with code 06"), "did not exit 6:\n{}", out);
}

/// A `let` the emitter never reaches - here after a `return` - has no extent
/// the program can stop in, so it is not described at all. Described anyway,
/// it would be visible in the whole function, holding whatever its slot held.
#[test]
fn an_unreachable_binding_is_not_described() {
    if !have("gdb") {
        return skip("an_unreachable_binding_is_not_described", "gdb");
    }
    let src = "fn main() -> I32 {\n    let a: I32 = 3;\n    if a > 1 {\n        return a; // L:ret\n    }\n    \
               return 0;\n    let dead_v: I32 = 7;\n}\n";
    let dir = scratch("dead");
    let bin = build(&dir, "dead", src, true);
    let out = gdb(
        &dir,
        &bin,
        &format!("break dead.ysu:{}\nrun\ninfo locals\nprint dead_v\nkill\n", line_of(src, "ret")),
    );
    assert!(out.contains("a = 3"), "the reachable binding is missing:\n{}", out);
    assert!(out.contains("No symbol \"dead_v\" in current context."), "the dead binding is described:\n{}", out);
}

/// A program for the gdb extension: every value it formats, and a method in
/// the call stack.
const PRINTERS: &str = "\
enum Color {
    Red,
    Green,
    Blue,
}

struct Point {
    x: I32,
    y: I32,
}

impl Point {
    fn sum(p: Point) -> I32 {
        let s: I32 = p.x + p.y; // L:sum
        return s;
    }
}

fn scale(p: Point, k: I32) -> I32 {
    let r: I32 = Point::sum(p) * k; // L:scale
    return r;
}

@unsafe
fn main() -> I32 {
    let name: String = \"hello\";
    let copy: String = String_new(\"hi\");
    let c: Color = Color::Blue;
    let small: I8 = -5;
    let byte: U8 = 200;
    @ZeroDrift @bounds(-1000, 1000)
    let acc: F32 = 0.0;
    acc += 1.5;
    acc += 2.25;
    @ZeroDrift @bounds(-1000, 1000)
    let neg: F32 = 0.0;
    neg -= 2.5;
    @ZeroDrift @bounds(-1000, 1000)
    let tenth: F32 = 0.0;
    tenth += 0.1;
    println(copy); // L:println
    let p: Point = Point { x: 3, y: 4 };
    let r: I32 = scale(p, 2); // L:call
    return r;
}
";

/// `raw / 2^frac` written out exactly - the oracle for the extension's
/// fixed-point printer, computed here independently of it.
fn exact_decimal(raw: i128, frac: u32) -> String {
    let sign = if raw < 0 { "-" } else { "" };
    let a = raw.unsigned_abs();
    let whole = a >> frac;
    let rest = a & ((1u128 << frac) - 1);
    if rest == 0 {
        return format!("{}{}", sign, whole);
    }
    let digits = format!("{:0>width$}", rest * 5u128.pow(frac), width = frac as usize);
    format!("{}{}.{}", sign, whole, digits.trim_end_matches('0'))
}

/// With the extension loaded, a Y value prints as Y: a `String` as its text,
/// an enum as `Enum::Variant`, an `I8`/`U8` as a number, and a `@ZeroDrift`
/// accumulator as its exact value - while `print/r` still shows the raw
/// representation the DWARF describes.
#[test]
fn y_values_print_as_y() {
    if !have("gdb") {
        return skip("y_values_print_as_y", "gdb");
    }
    let dir = scratch("printers");
    let bin = build(&dir, "printers", PRINTERS, true);
    let out = gdb_y(
        &dir,
        &bin,
        &format!(
            "info auto-load python-scripts\nbreak printers.ysu:{}\nrun\n\
             print name\nprint copy\nprint c\nprint small\nprint byte\n\
             whatis acc\nprint acc\nprint/r acc\nprint neg\nprint/r neg\nprint tenth\nprint/r tenth\n\
             print *name\nkill\n",
            line_of(PRINTERS, "call")
        ),
    );
    assert!(
        out.lines().any(|l| l.starts_with("Yes") && l.contains("ysu-gdb-extension")),
        "the embedded extension did not load:\n{}",
        out
    );
    for want in [
        "$1 = \"hello\"",
        "$2 = \"hi\"",
        "$3 = Color::Blue",
        "$4 = -5",
        "$5 = 200",
    ] {
        assert!(out.lines().any(|l| l == want), "missing `{}`:\n{}", want, out);
    }
    // The fixed-point values against an oracle computed from the raw words.
    let ty = out.lines().find_map(|l| l.strip_prefix("type = ")).expect("acc's type").to_string();
    let frac: u32 = ty
        .strip_suffix("_raw")
        .and_then(|q| q.split_once('.'))
        .and_then(|(_, f)| f.parse().ok())
        .unwrap_or_else(|| panic!("acc's type `{}` is not a Q format", ty));
    let repr = ty.strip_suffix("_raw").unwrap();
    let raw_of = |n: usize| -> i128 {
        let key = format!("${} = ", n);
        out.lines()
            .find_map(|l| l.strip_prefix(key.as_str()))
            .unwrap_or_else(|| panic!("no ${}:\n{}", n, out))
            .parse()
            .unwrap_or_else(|_| panic!("${} is not an integer:\n{}", n, out))
    };
    for (shown, raw, value) in [(6, 7, Some("3.75")), (8, 9, Some("-2.5")), (10, 11, None)] {
        let r = raw_of(raw);
        let exact = exact_decimal(r, frac);
        if let Some(v) = value {
            assert_eq!(exact, v, "the raw word ${} is not {}", raw, v);
        }
        let want = format!("${} = {} ({})", shown, exact, repr);
        assert!(out.lines().any(|l| l == want), "missing `{}`:\n{}", want, out);
    }
    // The raw structure is still there to look at.
    assert!(out.contains("len = 5, cap = 6"), "`print *name` lost the YStr view:\n{}", out);
}

/// A backtrace shows Y frames: a method by its Y name, no return addresses,
/// and none of the C runtime's frames below `fn main`. `bt -no-filters` is
/// gdb's own view, and still has them.
#[test]
fn a_backtrace_shows_the_y_frames() {
    if !have("gdb") {
        return skip("a_backtrace_shows_the_y_frames", "gdb");
    }
    let dir = scratch("bt");
    let bin = build(&dir, "printers", PRINTERS, true);
    let out = gdb_y(
        &dir,
        &bin,
        &format!(
            "break printers.ysu:{}\nrun\necho @@bt\\n\nbt\necho @@raw\\n\nbt -no-filters\nkill\n",
            line_of(PRINTERS, "sum")
        ),
    );
    let bt: Vec<&str> = out
        .split("@@bt")
        .nth(1)
        .and_then(|r| r.split("@@raw").next())
        .expect("the filtered backtrace")
        .lines()
        .filter(|l| l.starts_with('#'))
        .collect();
    let raw: Vec<&str> = out
        .split("@@raw")
        .nth(1)
        .expect("the unfiltered backtrace")
        .lines()
        .filter(|l| l.starts_with('#'))
        .collect();
    let want = [
        format!("#0  Point::sum (p=...) at printers.ysu:{}", line_of(PRINTERS, "sum")),
        format!("#1  scale (p=..., k=2) at printers.ysu:{}", line_of(PRINTERS, "scale")),
        format!("#2  main () at printers.ysu:{}", line_of(PRINTERS, "call")),
    ];
    assert_eq!(bt, want, "the Y backtrace:\n{}", out);
    // gdb's own view: the symbol name, return addresses, and the runtime's
    // `main` below the Y one.
    assert!(raw.len() > bt.len(), "bt -no-filters shows no more than bt:\n{}", out);
    assert!(raw[0].contains("Point_sum"), "{}", out);
    assert!(raw.iter().any(|l| l.contains(" in main ()") && !l.contains(" at ")), "no runtime main:\n{}", out);
}

/// What Y code CALLED stays in the backtrace even without Y source - stopped
/// inside the runtime's `println`, that frame is where the program is. Only
/// the frames BELOW `fn main` are the runtime's own business.
#[test]
fn a_runtime_frame_above_y_code_is_kept() {
    if !have("gdb") {
        return skip("a_runtime_frame_above_y_code_is_kept", "gdb");
    }
    let dir = scratch("bt_runtime");
    let bin = build(&dir, "printers", PRINTERS, true);
    let out = gdb_y(&dir, &bin, "break println\nrun\necho @@bt\\n\nbt\nkill\n");
    let bt: Vec<&str> = out
        .split("@@bt")
        .nth(1)
        .expect("the backtrace")
        .lines()
        .filter(|l| l.starts_with('#'))
        .collect();
    assert_eq!(bt.len(), 2, "want println's frame and main's:\n{}", out);
    assert!(bt[0].starts_with("#0") && bt[0].contains("println"), "{}", out);
    assert_eq!(bt[1], format!("#1  main () at printers.ysu:{}", line_of(PRINTERS, "println")), "{}", out);
}

/// Without the program in gdb's safe path, gdb declines the extension - and
/// the values are the raw DWARF view. That is the control: it is the
/// extension, not a change to the debug information, that formats them.
#[test]
fn outside_the_safe_path_gdb_shows_the_raw_view() {
    if !have("gdb") {
        return skip("outside_the_safe_path_gdb_shows_the_raw_view", "gdb");
    }
    let dir = scratch("declined");
    let bin = build(&dir, "printers", PRINTERS, true);
    let out = gdb(
        &dir,
        &bin,
        &format!("break printers.ysu:{}\nrun\nprint c\nprint small\nkill\n", line_of(PRINTERS, "call")),
    );
    assert!(out.contains("auto-loading has been declined"), "gdb loaded it anyway:\n{}", out);
    assert!(out.lines().any(|l| l == "$1 = Blue"), "{}", out);
    assert!(out.lines().any(|l| l.starts_with("$2 = -5 '")), "{}", out);
}

/// The section gdb reads is the compiler's extension, byte for byte, after
/// the program's own facts.
#[test]
fn the_embedded_extension_is_the_compilers_file() {
    if !have("objcopy") {
        return skip("the_embedded_extension_is_the_compilers_file", "objcopy");
    }
    let dir = scratch("section");
    let bin = build(&dir, "printers", PRINTERS, true);
    let dump = dir.join("scripts.bin");
    let st = Command::new("objcopy")
        .arg("--dump-section")
        .arg(format!(".debug_gdb_scripts={}", dump.display()))
        .arg(&bin)
        .arg(dir.join("stripped"))
        .status()
        .expect("run objcopy");
    assert!(st.success(), "objcopy found no .debug_gdb_scripts section");
    let bytes = fs::read(&dump).expect("read the section");
    let ext = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/debug_info_gdb.py"))
        .expect("read src/debug_info_gdb.py");
    let head = "\u{4}ysu-gdb-extension\n";
    let text = String::from_utf8(bytes).expect("the section is text");
    assert!(text.starts_with(head), "not an inline Python script entry: {:?}", &text[..text.len().min(40)]);
    let body = text[head.len()..].strip_suffix('\0').expect("NUL-terminated");
    let (facts, rest) = body.split_once('\n').expect("the program's facts, then the extension");
    assert!(facts.starts_with("Y_PROGRAM = {"), "{}", facts);
    assert!(facts.contains("\"Color\": False"), "{}", facts);
    assert!(facts.contains("\"Point_sum\": \"Point::sum\""), "{}", facts);
    assert_eq!(rest, ext, "the embedded extension differs from src/debug_info_gdb.py");
}

/// `String_new` of a literal holds the literal's text. A literal reaches it
/// as a string HANDLE, which it read as text: `String_new("hi")` printed `(`.
#[test]
fn string_new_holds_the_text_it_was_given() {
    let dir = scratch("string_new");
    let bin = build(
        &dir,
        "sn",
        "fn main() -> I32 {\n    let s: String = String_new(\"hi\");\n    println(s);\n    \
         let t: String = String_new(s);\n    println(t);\n    return 0;\n}\n",
        false,
    );
    let out = Command::new(&bin).output().expect("run");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hi\nhi\n");
}
