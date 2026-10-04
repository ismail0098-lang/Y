//! `--emit-ptx --lineinfo`: which PTX - and, through `ptxas -lineinfo`, which
//! SASS - each Y line became.
//!
//! The PTX backend writes a line table: `.file` directives naming the sources,
//! and a `.loc` before the instructions of every statement, with the
//! enclosing statement's line said again after a nested one so a `for` loop's
//! increment and back edge are the `for`'s. `tools/ydb/ymap.py` reads it back
//! through the assembler and the disassembler.
//!
//! The emitter had a field documented as doing exactly this - `pub
//! debug_info: bool`, "emits .file and .loc directives for NCU profiling and
//! debugging" - that nothing set and nothing read.
//!
//! What these tests hold it to:
//!
//! * **The line table changes nothing else.** Over the whole corpus, the PTX
//!   with `--lineinfo`, its `.file`/`.loc` lines removed, is byte-for-byte the
//!   PTX without it. And `ptxas -lineinfo` emits the same SASS instructions as
//!   plain `ptxas`, so the SASS the map shows is the SASS the kernel runs.
//! * **Every instruction has a line**, and the lines are the statements'.
//! * **It is refused where it would do nothing**, and `-o` is honoured or
//!   refused by every backend rather than ignored.
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[path = "common/pinned.rs"]
mod pinned;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok()
}

fn skip(test: &str, tool: &str) {
    eprintln!("SKIP {}: no {} on this machine, so this test checked NOTHING", test, tool);
}

fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn y(dir: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_Y"));
    c.current_dir(dir);
    c
}

/// `src` compiled with `--emit-ptx` (and `extra`) in a pinned directory. The
/// source is named by a RELATIVE path, as a user types it, so a `.file` that
/// is not made canonical shows.
fn ptx(dir: &Path, name: &str, src: &str, extra: &[&str]) -> String {
    let path = dir.join(format!("{}.ysu", name));
    fs::write(&path, src).expect("write source");
    let out_path = dir.join(format!("{}{}.ptx", name, if extra.is_empty() { "" } else { "_li" }));
    let out = y(dir)
        .arg(format!("{}.ysu", name))
        .arg("--emit-ptx")
        .args(extra)
        .arg("-o")
        .arg(&out_path)
        .output()
        .expect("run Y");
    assert!(out.status.success(), "{} {:?} did not compile:\n{}", name, extra, text(&out));
    fs::read_to_string(&out_path).expect("read PTX")
}

/// The line `// L:<tag>` marks, 1-based.
fn line_of(src: &str, tag: &str) -> usize {
    let mark = format!("// L:{}", tag);
    let hits: Vec<usize> = src.lines().enumerate().filter(|(_, l)| l.contains(&mark)).map(|(i, _)| i + 1).collect();
    assert_eq!(hits.len(), 1, "marker {} must occur once", mark);
    hits[0]
}

fn is_line_table(l: &str) -> bool {
    let t = l.trim_start();
    t.starts_with(".loc ") || t.starts_with(".file ")
}

/// For every entry body: each instruction with the (file, line) of the `.loc`
/// in effect, and the instructions before the first `.loc`.
///
/// An instruction ends at its `;`, and may span lines: an operand group in
/// braces (`mma.sync .. {%f0, ..}`, a `wmma.store` whose operands continue on
/// the next lines). A SCOPE brace is a line of its own.
fn attributed(ptx: &str) -> BTreeMap<String, (Vec<((usize, usize), String)>, Vec<String>)> {
    let mut out = BTreeMap::new();
    let mut current: Option<String> = None;
    let mut depth = 0i32;
    let mut loc: Option<(usize, usize)> = None;
    let mut pending = String::new();
    for raw in ptx.lines() {
        let t = raw.split("//").next().unwrap_or("").trim();
        if current.is_none() {
            if let Some(rest) = t.strip_prefix(".visible .entry ") {
                let name = rest.split('(').next().unwrap_or("").trim().to_string();
                out.insert(name.clone(), (Vec::new(), Vec::new()));
                current = Some(name);
                loc = None;
                depth = 0;
            }
            continue;
        }
        let name = current.clone().expect("in an entry");
        if !pending.is_empty() {
            pending.push(' ');
            pending.push_str(t);
        } else if t == "{" {
            depth += 1;
            continue;
        } else if t == "}" {
            depth -= 1;
            if depth == 0 {
                current = None;
            }
            continue;
        } else if depth == 0 {
            continue; // the parameter list and `.maxnreg`
        } else if let Some(rest) = t.strip_prefix(".loc ") {
            let f: Vec<usize> = rest.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            loc = Some((f[0], f[1]));
            continue;
        } else if t.is_empty() || (t.ends_with(':') && !t.contains(';')) || t.starts_with(".reg") || t.starts_with(".shared") || t.starts_with(".local") {
            continue;
        } else {
            pending.push_str(t);
        }
        if pending.contains(';') {
            let entry = out.get_mut(&name).expect("entry recorded");
            let insn = std::mem::take(&mut pending);
            match loc {
                Some(l) => entry.0.push((l, insn)),
                None => entry.1.push(insn),
            }
        }
    }
    out
}

fn corpus() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(pinned::repo().join("tests"))
        .expect("tests/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "ysu").unwrap_or(false))
        .collect();
    v.sort();
    v
}

/// THE property: the line table is the only difference. Over every corpus
/// program the PTX backend compiles.
#[test]
fn the_line_table_changes_nothing_else() {
    let dir = pinned::pinned_scratch("lineinfo_corpus", pinned::SM_PINNED);
    let (mut compared, mut located, mut failures) = (0, 0, Vec::new());
    for src in corpus() {
        let name = src.file_stem().unwrap().to_string_lossy().into_owned();
        let copy = dir.join(format!("{}.ysu", name));
        fs::copy(&src, &copy).expect("copy fixture");
        let plain_path = dir.join(format!("{}.ptx", name));
        let plain = y(&dir).arg(&copy).arg("--emit-ptx").arg("-o").arg(&plain_path).output().expect("run Y");
        if !plain.status.success() {
            continue;
        }
        let li_path = dir.join(format!("{}_li.ptx", name));
        let li = y(&dir).arg(&copy).arg("--emit-ptx").arg("--lineinfo").arg("-o").arg(&li_path).output().expect("run Y");
        if !li.status.success() {
            failures.push(format!("{}: compiles without --lineinfo and not with it:\n{}", name, text(&li)));
            continue;
        }
        let a = fs::read_to_string(&plain_path).expect("plain PTX");
        let b = fs::read_to_string(&li_path).expect("line-table PTX");
        assert!(!a.lines().any(is_line_table), "{}: a .loc/.file without --lineinfo", name);
        let stripped: Vec<&str> = b.lines().filter(|l| !is_line_table(l)).collect();
        let original: Vec<&str> = a.lines().collect();
        if stripped != original {
            failures.push(format!("{}: --lineinfo changed more than the line table", name));
        }
        if b.lines().any(|l| l.trim_start().starts_with(".loc ")) {
            located += 1;
        }
        compared += 1;
    }
    assert!(compared >= 50, "only {} corpus programs compiled; the sweep is not testing anything", compared);
    assert_eq!(located, compared, "a compiled module with no .loc at all");
    assert!(failures.is_empty(), "{} of {}:\n{}", failures.len(), compared, failures.join("\n"));
    let _ = fs::remove_dir_all(&dir);
}

/// Every instruction of every corpus kernel has a line, and every `.loc`
/// names a `.file` the module declares.
#[test]
fn every_instruction_has_a_line() {
    let dir = pinned::pinned_scratch("lineinfo_every", pinned::SM_PINNED);
    let mut kernels = 0;
    for src in corpus() {
        let name = src.file_stem().unwrap().to_string_lossy().into_owned();
        let copy = dir.join(format!("{}.ysu", name));
        fs::copy(&src, &copy).expect("copy fixture");
        let out_path = dir.join(format!("{}.ptx", name));
        let out = y(&dir).arg(&copy).arg("--emit-ptx").arg("--lineinfo").arg("-o").arg(&out_path).output().expect("run Y");
        if !out.status.success() {
            continue;
        }
        let p = fs::read_to_string(&out_path).expect("PTX");
        let files: Vec<usize> = p
            .lines()
            .filter_map(|l| l.strip_prefix(".file "))
            .filter_map(|r| r.split_whitespace().next().and_then(|n| n.parse().ok()))
            .collect();
        // A `.loc` with nothing after it is replaced, not followed.
        let lines: Vec<&str> = p.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
        for w in lines.windows(2) {
            assert!(
                !(w[0].starts_with(".loc ") && w[1].starts_with(".loc ")),
                "{}: two .loc lines in a row:\n{}\n{}",
                name,
                w[0],
                w[1]
            );
        }
        for (k, (insns, before)) in attributed(&p) {
            assert!(before.is_empty(), "{}::{}: instructions before any .loc: {:?}", name, k, before);
            for ((f, _), i) in &insns {
                assert!(files.contains(f), "{}::{}: `{}` is under .file {}, which is not declared", name, k, i, f);
            }
            kernels += 1;
        }
    }
    assert!(kernels >= 50, "only {} kernels examined", kernels);
    let _ = fs::remove_dir_all(&dir);
}

const LINES: &str = "\
kernel saxpy(X: GlobalMemory<F32>, Yv: GlobalMemory<F32>, N: I32) { // L:kernel
    let i: I32 = thread_idx_x(); // L:tid
    let a: F32 = 2.0;
    if i < N { // L:if
        let x: F32 = X[i]; // L:load
        Yv[i] = a * x + Yv[i]; // L:store
    }
}

kernel sum(Out: GlobalMemory<I32>) { // L:sum
    let mut s: I32 = 0;
    @invariant(s >= 0) // L:invariant
    for k in 0..8 { // L:for
        s = s + k; // L:body
    }
    Out[0] = s; // L:out
}

fn main() {}
";

fn under(insns: &[((usize, usize), String)], line: usize) -> Vec<&str> {
    insns.iter().filter(|((_, l), _)| *l == line).map(|(_, i)| i.as_str()).collect()
}

/// The lines are the statements': each statement's instructions follow its
/// own `.loc`, a loop's increment and back edge are the loop's, and a
/// statement with an attribute is on its keyword's line, not the attribute's.
#[test]
fn each_instruction_is_on_its_statements_line() {
    let dir = pinned::pinned_scratch("lineinfo_lines", pinned::SM_PINNED);
    let p = ptx(&dir, "lines", LINES, &["--lineinfo"]);
    let file = fs::canonicalize(dir.join("lines.ysu")).expect("canonical path");
    assert!(
        p.contains(&format!(".file 1 \"{}\"", file.display())),
        "the module must name its source by its canonical path:\n{}",
        p
    );
    let map = attributed(&p);
    let (saxpy, _) = &map["saxpy"];
    let l = |tag: &str| line_of(LINES, tag);
    assert!(under(saxpy, l("kernel")).iter().all(|i| i.starts_with("ld.param")), "{:?}", under(saxpy, l("kernel")));
    assert!(under(saxpy, l("tid")).iter().any(|i| i.contains("%tid.x")), "{:?}", saxpy);
    assert!(under(saxpy, l("if")).iter().any(|i| i.starts_with("setp")), "{:?}", saxpy);
    assert!(under(saxpy, l("load")).iter().any(|i| i.starts_with("ld.global")), "{:?}", saxpy);
    let store = under(saxpy, l("store"));
    assert!(store.iter().any(|i| i.starts_with("st.global")), "{:?}", store);
    assert!(store.iter().any(|i| i.starts_with("fma.rn.f32")), "the multiply-add is the store line's: {:?}", store);
    assert!(!under(saxpy, l("load")).iter().any(|i| i.starts_with("st.global")), "{:?}", saxpy);

    let (sum, _) = &map["sum"];
    assert!(under(sum, l("invariant")).is_empty(), "the attribute's line has code: {:?}", sum);
    let for_line = under(sum, l("for"));
    assert!(for_line.iter().any(|i| i.starts_with("setp")), "the loop test is the for line's: {:?}", for_line);
    let body_at = sum.iter().position(|((_, line), _)| *line == l("body")).expect("the body has code");
    let after: Vec<&str> = sum[body_at..]
        .iter()
        .skip_while(|((_, line), _)| *line == l("body"))
        .take_while(|((_, line), _)| *line == l("for"))
        .map(|(_, i)| i.as_str())
        .collect();
    assert!(
        after.iter().any(|i| i.starts_with("add")) && after.iter().any(|i| i.starts_with("bra")),
        "the increment and back edge after the body must be the for line's, got {:?} in\n{:?}",
        after,
        sum
    );
    assert!(under(sum, l("out")).iter().any(|i| i.starts_with("st.global")), "{:?}", sum);
    let _ = fs::remove_dir_all(&dir);
}

/// A kernel an `import` brought in is described in its own file.
#[test]
fn an_imported_kernel_names_its_own_file() {
    let dir = pinned::pinned_scratch("lineinfo_import", pinned::SM_PINNED);
    let util = "kernel zero(Out: GlobalMemory<I32>) {\n    Out[0] = 0; // L:util\n}\n";
    fs::write(dir.join("util.ysu"), util).expect("write util");
    let main = "import util;\n\nkernel one(Out: GlobalMemory<I32>) {\n    Out[0] = 1; // L:main\n}\n\nfn main() {}\n";
    let p = ptx(&dir, "app", main, &["--lineinfo"]);
    let app = fs::canonicalize(dir.join("app.ysu")).expect("canonical");
    let utilp = fs::canonicalize(dir.join("util.ysu")).expect("canonical");
    assert!(p.contains(&format!(".file 1 \"{}\"", app.display())), "{}", p);
    assert!(p.contains(&format!(".file 2 \"{}\"", utilp.display())), "{}", p);
    let map = attributed(&p);
    assert!(map["one"].0.iter().all(|((f, _), _)| *f == 1), "{:?}", map["one"]);
    assert!(map["zero"].0.iter().all(|((f, _), _)| *f == 2), "{:?}", map["zero"]);
    assert!(!under(&map["zero"].0, line_of(util, "util")).is_empty(), "{:?}", map["zero"]);
    assert!(!under(&map["one"].0, line_of(main, "main")).is_empty(), "{:?}", map["one"]);
    let _ = fs::remove_dir_all(&dir);
}

fn ptxas_arch(p: &str) -> String {
    p.lines()
        .find_map(|l| l.strip_prefix(".target "))
        .map(|t| t.split(',').next().unwrap_or("").trim().to_string())
        .expect(".target")
}

/// SASS instruction text of a cubin, one per line, padding included.
fn sass_text(dir: &Path, cubin: &Path, g: bool) -> String {
    let mut c = Command::new("nvdisasm");
    if g {
        c.arg("-g");
    }
    let out = c.arg("-c").arg(cubin).current_dir(dir).output().expect("nvdisasm");
    assert!(out.status.success(), "nvdisasm:\n{}", text(&out));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn instructions(sass: &str) -> Vec<String> {
    sass.lines()
        .filter(|l| l.trim_start().starts_with("/*"))
        .map(|l| l.trim().to_string())
        .collect()
}

fn assemble(dir: &Path, ptx: &str, name: &str, lineinfo: bool) -> PathBuf {
    let src = dir.join(format!("{}.ptx", name));
    fs::write(&src, ptx).expect("write PTX");
    let cubin = dir.join(format!("{}.cubin", name));
    let mut c = Command::new("ptxas");
    c.arg(format!("-arch={}", ptxas_arch(ptx)));
    if lineinfo {
        c.arg("-lineinfo");
    }
    let out = c.arg("-o").arg(&cubin).arg(&src).current_dir(dir).output().expect("ptxas");
    assert!(out.status.success(), "ptxas rejected {}:\n{}", name, text(&out));
    cubin
}

/// `ptxas -lineinfo` carries the Y lines into the SASS, and changes no
/// instruction: plain PTX through plain `ptxas`, the line-table PTX through
/// plain `ptxas`, and the line-table PTX through `ptxas -lineinfo` all
/// disassemble to the same instructions.
#[test]
fn the_sass_carries_the_lines_and_is_unchanged() {
    if !have("ptxas") || !have("nvdisasm") {
        return skip("the_sass_carries_the_lines_and_is_unchanged", "ptxas/nvdisasm");
    }
    let dir = pinned::pinned_scratch("lineinfo_sass", pinned::SM_PINNED);
    let plain = ptx(&dir, "lines", LINES, &[]);
    let li = ptx(&dir, "lines", LINES, &["--lineinfo"]);
    let a = instructions(&sass_text(&dir, &assemble(&dir, &plain, "plain", false), false));
    let b = instructions(&sass_text(&dir, &assemble(&dir, &li, "li_plain", false), false));
    let c_cubin = assemble(&dir, &li, "li", true);
    let c = instructions(&sass_text(&dir, &c_cubin, false));
    assert!(a.len() > 10, "too few instructions to compare: {:?}", a);
    assert_eq!(a, b, "the .loc lines changed the SASS");
    assert_eq!(a, c, "-lineinfo changed the SASS");
    // ptxas interleaves lines, so a line's SASS is every run under its tag.
    let with_lines = sass_text(&dir, &c_cubin, true);
    let file = fs::canonicalize(dir.join("lines.ysu")).expect("canonical");
    let tag = format!("//## File \"{}\", line {}", file.display(), line_of(LINES, "store"));
    let mut store = Vec::new();
    let mut inside = false;
    for l in with_lines.lines() {
        if l.contains("//## File") {
            inside = l.trim_end().ends_with(&tag);
        } else if inside && l.trim_start().starts_with("/*") {
            store.push(l.trim().to_string());
        }
    }
    assert!(store.iter().any(|l| l.contains("STG")), "the store line's SASS has no STG: {:?}\n{}", store, with_lines);
    let _ = fs::remove_dir_all(&dir);
}

/// The same over the whole corpus: for every program the PTX backend
/// compiles, `ptxas -lineinfo` on the line-table PTX emits exactly the SASS
/// instructions plain `ptxas` emits for the plain PTX. Without this, the map
/// would describe a build that does not ship.
#[test]
fn the_sass_is_unchanged_over_the_corpus() {
    if !have("ptxas") || !have("nvdisasm") {
        return skip("the_sass_is_unchanged_over_the_corpus", "ptxas/nvdisasm");
    }
    let dir = pinned::pinned_scratch("lineinfo_sass_corpus", pinned::SM_PINNED);
    let (mut compared, mut failures) = (0, Vec::new());
    for src in corpus() {
        let name = src.file_stem().unwrap().to_string_lossy().into_owned();
        let copy = dir.join(format!("{}.ysu", name));
        fs::copy(&src, &copy).expect("copy fixture");
        let plain_path = dir.join(format!("{}.plain.ptx", name));
        let li_path = dir.join(format!("{}.li.ptx", name));
        let plain = y(&dir).arg(&copy).arg("--emit-ptx").arg("-o").arg(&plain_path).output().expect("run Y");
        if !plain.status.success() {
            continue;
        }
        let li = y(&dir).arg(&copy).arg("--emit-ptx").arg("--lineinfo").arg("-o").arg(&li_path).output().expect("run Y");
        assert!(li.status.success(), "{}:\n{}", name, text(&li));
        let a = fs::read_to_string(&plain_path).expect("plain PTX");
        let b = fs::read_to_string(&li_path).expect("line-table PTX");
        let sa = instructions(&sass_text(&dir, &assemble(&dir, &a, &format!("{}_a", name), false), false));
        let sb = instructions(&sass_text(&dir, &assemble(&dir, &b, &format!("{}_b", name), true), false));
        if sa.is_empty() || sa != sb {
            failures.push(name);
        }
        compared += 1;
    }
    assert!(compared >= 50, "only {} corpus programs assembled; the sweep is not testing anything", compared);
    assert!(failures.is_empty(), "-lineinfo changed the SASS of {} of {}: {:?}", failures.len(), compared, failures);
    let _ = fs::remove_dir_all(&dir);
}

/// `tools/ydb/ymap.py` reports each line's PTX and SASS - including a line
/// whose PTX ptxas removed, which is the kind of thing the map is for.
#[test]
fn ymap_reports_each_lines_ptx_and_sass() {
    if !have("python3") || !have("ptxas") || !have("nvdisasm") {
        return skip("ymap_reports_each_lines_ptx_and_sass", "python3/ptxas/nvdisasm");
    }
    let dir = pinned::pinned_scratch("lineinfo_ymap", pinned::SM_PINNED);
    let src = "\
kernel dead(Out: GlobalMemory<I32>, N: I32) {
    let t: I32 = thread_idx_x(); // L:t
    let unused: I32 = t * 7 + N; // L:dead
    Out[t] = t + 1; // L:store
}

fn main() {}
";
    let path = dir.join("dead.ysu");
    fs::write(&path, src).expect("write");
    let out = Command::new("python3")
        .arg(pinned::repo().join("tools/ydb/ymap.py"))
        .arg(&path)
        .arg("--json")
        .arg("--y")
        .arg(env!("CARGO_BIN_EXE_Y"))
        .current_dir(&dir)
        .output()
        .expect("run ymap");
    assert!(out.status.success(), "ymap failed:\n{}", text(&out));
    let json = String::from_utf8_lossy(&out.stdout).into_owned();
    // The JSON is small and flat enough to check by its rows.
    let row = |line: usize| -> String {
        let key = format!("\"line\": {},", line);
        let at = json.find(&key).unwrap_or_else(|| panic!("no row for line {}:\n{}", line, json));
        let end = json[at..].find("\n  }").map(|e| at + e).unwrap_or(json.len());
        json[at..end].to_string()
    };
    let dead = row(line_of(src, "dead"));
    assert!(dead.contains("mul.lo") || dead.contains("mad.lo"), "the dead line's PTX: {}", dead);
    assert!(dead.contains("\"sass\": []"), "ptxas removed the dead line, so it has no SASS: {}", dead);
    let store = row(line_of(src, "store"));
    assert!(store.contains("st.global"), "{}", store);
    assert!(store.contains("STG"), "{}", store);
    assert!(json.contains("\"sass_padding\""), "{}", json);
    assert!(!json.contains("  NOP"), "padding NOPs belong to no line: {}", json);
    // One line, as text.
    let one = Command::new("python3")
        .arg(pinned::repo().join("tools/ydb/ymap.py"))
        .arg(&path)
        .arg("--line")
        .arg(line_of(src, "store").to_string())
        .arg("--y")
        .arg(env!("CARGO_BIN_EXE_Y"))
        .current_dir(&dir)
        .output()
        .expect("run ymap");
    let t = text(&one);
    assert!(one.status.success() && t.contains("Out[t] = t + 1;") && t.contains("STG"), "{}", t);
    assert!(!t.contains("let unused"), "--line shows one line: {}", t);
    let _ = fs::remove_dir_all(&dir);
}

/// `--lineinfo` belongs to the PTX backend, and is refused - by name - where
/// it would do nothing. `-g` with `--emit-ptx` points at it.
#[test]
fn lineinfo_is_refused_where_it_would_do_nothing() {
    let dir = pinned::pinned_scratch("lineinfo_refuse", pinned::SM_PINNED);
    let path = dir.join("k.ysu");
    fs::write(&path, LINES).expect("write");
    for backend in ["--emit-cpu", "--emit-native", "--emit-coprocessor", "--target=r1cs"] {
        let out = y(&dir).arg(&path).arg(backend).arg("--lineinfo").output().expect("run Y");
        assert!(!out.status.success(), "--lineinfo {} was accepted", backend);
        assert!(text(&out).contains(&format!("--lineinfo cannot be combined with {}", backend)), "{}", text(&out));
    }
    for llvm in [vec![], vec!["--emit-llvm"]] {
        let out = y(&dir).arg(&path).args(&llvm).arg("--lineinfo").output().expect("run Y");
        assert!(!out.status.success(), "--lineinfo {:?} was accepted", llvm);
        assert!(text(&out).contains("use -g"), "{}", text(&out));
    }
    let out = y(&dir).args(["--emit-attention-ptx", "64", "128", "--lineinfo"]).output().expect("run Y");
    assert!(!out.status.success() && text(&out).contains("--lineinfo cannot be combined with --emit-attention-ptx"), "{}", text(&out));
    let out = y(&dir).arg(&path).arg("--emit-ptx").arg("-g").output().expect("run Y");
    assert!(!out.status.success() && text(&out).contains("use --lineinfo"), "{}", text(&out));
    // The control: the PTX backend takes it.
    let out = y(&dir).arg(&path).arg("--emit-ptx").arg("--lineinfo").arg("-o").arg(dir.join("ok.ptx")).output().expect("run Y");
    assert!(out.status.success(), "{}", text(&out));
    let _ = fs::remove_dir_all(&dir);
}

/// `-o` used to be ignored by `--emit-ptx`, `--emit-coprocessor`,
/// `--emit-zk-ptx` and `--emit-cpu`, which wrote `<source>.ptx` and so on - or
/// printed - whatever it said. Every backend that writes a file writes it to
/// `-o` now, and `--emit-attention-ptx`, which writes to standard output,
/// refuses it.
#[test]
fn minus_o_names_the_output_of_every_backend() {
    let dir = pinned::pinned_scratch("lineinfo_minus_o", pinned::SM_PINNED);
    let k = dir.join("k.ysu");
    fs::write(&k, LINES).expect("write");
    let host = dir.join("h.ysu");
    fs::write(&host, "fn main() -> I32 {\n    return 3;\n}\n").expect("write");
    let co = pinned::copy_fixture(&dir, "tests/coprocessor_attention.ysu");
    let mut cases: Vec<(&Path, Vec<&str>, &str)> = vec![
        (&k, vec!["--emit-ptx"], "ptx_out.txt"),
        (&host, vec!["--emit-cpu"], "cpu_out.txt"),
        (&co, vec!["--emit-coprocessor"], "co_out.txt"),
        (&host, vec!["--emit-llvm"], "ll_out.txt"),
    ];
    if have("clang") {
        cases.push((&host, vec![], "bin_out"));
    }
    for (src, flags, name) in &cases {
        let target = dir.join(name);
        let _ = fs::remove_file(&target);
        let out = y(&dir).arg(src).args(flags).arg("-o").arg(&target).output().expect("run Y");
        assert!(out.status.success(), "{:?} -o {}:\n{}", flags, name, text(&out));
        assert!(target.exists(), "{:?} -o {} did not write {}", flags, name, name);
        assert!(fs::metadata(&target).expect("meta").len() > 0, "{:?}: {} is empty", flags, name);
    }
    // --emit-cpu -o writes the source, not the console banner.
    let cpu = fs::read_to_string(dir.join("cpu_out.txt")).expect("cpu source");
    assert!(!cpu.contains("Y Compiler") && !cpu.contains("GENERATED RUST BLOB"), "{}", cpu);
    // --emit-ptx -o writes there and nowhere else.
    assert!(!dir.join("k.ptx").exists(), "--emit-ptx -o also wrote k.ptx");
    let out = y(&dir).args(["--emit-attention-ptx", "64", "128", "-o"]).arg(dir.join("a.ptx")).output().expect("run Y");
    assert!(!out.status.success() && text(&out).contains("takes no other argument"), "{}", text(&out));
    assert!(!dir.join("a.ptx").exists());
    let _ = fs::remove_dir_all(&dir);
}

/// `--emit-zk-ptx -o` writes there too (a `--features zk` build).
#[cfg(feature = "zk")]
#[test]
fn minus_o_names_the_zk_witness_kernel() {
    let dir = pinned::pinned_scratch("lineinfo_minus_o_zk", pinned::SM_PINNED);
    let src = dir.join("c.ysu");
    fs::write(&src, "fn main(a: I32, b: I32) -> I32 { return a * b; }").expect("write");
    let target = dir.join("w_out.ptx");
    let out = y(&dir).arg(&src).arg("--emit-zk-ptx").arg("-o").arg(&target).output().expect("run Y");
    assert!(out.status.success(), "{}", text(&out));
    assert!(target.exists(), "--emit-zk-ptx -o did not write it:\n{}", text(&out));
    assert!(!dir.join("c.witness.ptx").exists(), "--emit-zk-ptx -o also wrote c.witness.ptx");
    let _ = fs::remove_dir_all(&dir);
}
