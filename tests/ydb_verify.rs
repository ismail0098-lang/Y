//! `verify`: what covers a line of a Y program.
//!
//! The compiler records what it proved, checked or took on trust about each
//! line (`src/guarantees.rs`): a `-g` program carries the table, and
//! `Y --emit-guarantees` writes it. `tools/ydb/yverify.py` reports it for a
//! line - each fact with how it is established and the assumptions a proof
//! used - and adds the repository's evidence about the code a kernel becomes:
//! the proofs of the lowering it was given, and `tools/ptxas_tval`'s standing
//! results, credited only when this compile's kernel is the committed one they
//! are about, instruction for instruction.
//!
//! The controls are what make these claims worth anything: a kernel with the
//! committed one's NAME and one instruction changed must not be credited, a
//! proof from a trusted range must say so, and a proof whose solver did not
//! run must not read as proved.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

/// The report with every run of whitespace collapsed: it wraps at 100
/// columns, so a phrase can break across lines.
fn flat(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok()
}

/// The line `// L:<tag>` marks, 1-based.
fn line_of(src: &str, tag: &str) -> usize {
    let mark = format!("// L:{}", tag);
    let hits: Vec<usize> = src.lines().enumerate().filter(|(_, l)| l.contains(&mark)).map(|(i, _)| i + 1).collect();
    assert_eq!(hits.len(), 1, "marker {} must occur once", mark);
    hits[0]
}

const PROG: &str = "\
fn total(n: I32) -> I32 {
    @bounds(0, 6) let m: I32 = n; // L:trusted
    let v: [I32; 8] = {};
    let s: I32 = 0;
    @invariant(i >= 0)
    for i in 0..m { // L:loop
        s = s + v[i]; // L:vi
    }
    return s + v[2]; // L:v2
}

@unsafe
fn raw(k: I32) -> I32 { // L:raw
    let w: [I32; 4] = {};
    @bounds(0, 3) let c: I32 = 2; // L:checked
    @safe { // L:safeblock
        let t: I32 = w[c]; // L:wc
    }
    @invariant(j >= 0)
    for j in 0..2 { // L:unsafeloop
        let u: I32 = j;
    }
    return w[k]; // L:wk
}

kernel bump(Out: GlobalMemory<F32>, N: I32) { // L:bump
    @invariant(i >= 0)
    for i in 0..N { // L:kloop
        Out[i] = Out[i] * 2.0 + 1.0; // L:out
    }
}

@require(sm >= 80) // L:require
kernel tiny(Out: GlobalMemory<F32>) {
    let x: F32 = 1.0;
}

fn main() -> I32 {
    return total(3) + raw(1);
}
";

/// A pinned directory holding `prog.ysu`.
fn setup(tag: &str, src: &str, sm: &str) -> (PathBuf, PathBuf) {
    let dir = pinned::pinned_scratch(&format!("yv_{}", tag), sm);
    let path = dir.join("prog.ysu");
    fs::write(&path, src).expect("write source");
    (dir, path)
}

/// `Y prog --emit-guarantees -o g.json`: (the table, or None, and the output).
fn emit(dir: &Path, program: &Path, env: &[(&str, &str)]) -> (Option<String>, String) {
    let out_path = dir.join("g.json");
    let _ = fs::remove_file(&out_path);
    let mut c = Command::new(env!("CARGO_BIN_EXE_Y"));
    c.arg(program).arg("--emit-guarantees").arg("-o").arg(&out_path).current_dir(dir);
    for (k, v) in env {
        c.env(k, v);
    }
    let out = c.output().expect("run Y");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let table = if out.status.success() { fs::read_to_string(&out_path).ok() } else { None };
    (table, text)
}

/// Without a solver an invariant cannot be checked and the program is
/// refused, so a test that needs one says so instead of failing.
fn no_solver(text: &str, test: &str) -> bool {
    if text.contains("SMT solver could not be run") {
        eprintln!("SKIP {}: no z3 on this machine, so this test checked NOTHING", test);
        return true;
    }
    false
}

/// The fact objects of a table, as text: `{"item": ...}` up to the next one.
fn facts(table: &str) -> Vec<String> {
    let start = table.find("\"facts\": [").expect("a facts list") + "\"facts\": [".len();
    table[start..].split("{\"item\": ").skip(1).map(|s| s.to_string()).collect()
}

/// The one fact of `kind` on `line` about `what`.
fn fact(table: &str, kind: &str, line: usize, what: &str) -> String {
    let hits: Vec<String> = facts(table)
        .into_iter()
        .filter(|f| {
            f.contains(&format!("\"kind\": \"{}\"", kind))
                && f.contains(&format!("\"line\": {},", line))
                && f.contains(&format!("\"what\": \"{}\"", what.replace('"', "\\\"")))
        })
        .collect();
    assert!(!hits.is_empty(), "no {} fact about `{}` on line {} in:\n{}", kind, what, line, table);
    hits[0].clone()
}

fn status(f: &str) -> &str {
    let i = f.find("\"status\": \"").expect("a status") + "\"status\": \"".len();
    &f[i..i + f[i..].find('"').unwrap()]
}

/// Every fact says how it is established, and a proof from a range the
/// compiler took on trust names it.
#[test]
fn the_table_says_how_each_fact_is_established() {
    let (dir, prog) = setup("table", PROG, pinned::SM_PINNED);
    let (table, text) = emit(&dir, &prog, &[]);
    if no_solver(&text, "the_table_says_how_each_fact_is_established") {
        return;
    }
    let table = table.unwrap_or_else(|| panic!("--emit-guarantees failed:\n{}", text));
    let l = |t| line_of(PROG, t);

    let vi = fact(&table, "index", l("vi"), "v[i]");
    assert_eq!(status(&vi), "proved", "{}", vi);
    assert!(vi.contains("the index lies in [0, 5] and there are 8 elements"), "{}", vi);
    assert!(vi.contains("@bounds(0, 6) on `m`"), "the proof rests on m's trusted range and must say so:\n{}", vi);
    // `v[2]` needs no assumption, and must not borrow one.
    let v2 = fact(&table, "index", l("v2"), "v[2]");
    assert_eq!(status(&v2), "proved");
    assert!(v2.contains("\"rests_on\": []"), "{}", v2);

    let trusted = fact(&table, "bounds", l("trusted"), "@bounds(0, 6) on `m`");
    assert_eq!(status(&trusted), "trusted", "nothing bounds `n`: {}", trusted);
    let checked = fact(&table, "bounds", l("checked"), "@bounds(0, 3) on `c`");
    assert_eq!(status(&checked), "checked", "{}", checked);
    assert!(checked.contains("[2, 2]"), "{}", checked);

    let inv = fact(&table, "invariant", l("loop"), "@invariant(i >= 0)");
    assert_eq!(status(&inv), "proved");
    assert!(inv.contains("@bounds(0, 6) on `m`"), "z3 was given m's trusted range:\n{}", inv);
    assert!(inv.contains(&format!("\"end\": {},", l("vi"))), "the invariant covers its body:\n{}", inv);
    let kinv = fact(&table, "invariant", l("kloop"), "@invariant(i >= 0)");
    assert!(kinv.contains("\"rests_on\": []"), "{}", kinv);
    let uinv = fact(&table, "invariant", l("unsafeloop"), "@invariant(j >= 0)");
    assert_eq!(status(&uinv), "not-checked", "an @unsafe loop's invariant is not verified:\n{}", uinv);

    let wk = fact(&table, "index", l("wk"), "w[k]");
    assert_eq!(status(&wk), "run-time", "{}", wk);
    assert!(wk.contains("checked against the 4 elements"), "{}", wk);
    let wc = fact(&table, "index", l("wc"), "w[c]");
    assert_eq!(status(&wc), "proved", "inside @safe, from a checked range:\n{}", wc);
    let out = fact(&table, "index", l("out"), "Out[i]");
    assert_eq!(status(&out), "not-checked", "{}", out);
    assert!(out.contains("`Out` is not a fixed-size array"), "{}", out);

    let raw = fact(&table, "safe", l("raw"), "fn raw");
    assert_eq!(status(&raw), "not-checked", "{}", raw);
    let block = fact(&table, "safe", l("safeblock"), "@safe { }");
    assert_eq!(status(&block), "checked", "{}", block);
    assert!(block.contains(&format!("\"end\": {},", l("wc"))), "{}", block);
    let bump = fact(&table, "safe", l("bump"), "kernel bump");
    assert_eq!(status(&bump), "checked", "a kernel is always strict:\n{}", bump);
    let req = fact(&table, "require", l("require"), "@require(sm >= 80)");
    assert!(req.contains("satisfied: `sm` is 80"), "{}", req);
    assert!(table.contains("{\"name\": \"bump\", \"kind\": \"kernel\""), "{}", table);
    let _ = fs::remove_dir_all(&dir);
}

/// A `-g` program carries the same table `--emit-guarantees` writes: `verify`
/// reads the facts the binary was built with.
#[test]
fn the_binary_carries_the_table_it_was_built_with() {
    if !have("gdb") || !have("clang") || !have("python3") {
        eprintln!("SKIP the_binary_carries_the_table_it_was_built_with: no gdb, clang or python3, so this test checked NOTHING");
        return;
    }
    let (dir, prog) = setup("carried", PROG, pinned::SM_PINNED);
    let (table, text) = emit(&dir, &prog, &[]);
    if no_solver(&text, "the_binary_carries_the_table_it_was_built_with") {
        return;
    }
    assert!(table.is_some(), "{}", text);
    let canon = |src: &str| {
        let out = Command::new("python3")
            .args(["-c", "import json,sys; print(json.dumps(json.load(open(sys.argv[1])), sort_keys=True))"])
            .arg(src)
            .output()
            .expect("python3");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let written = canon(dir.join("g.json").to_str().unwrap());
    let out = Command::new("python3")
        .arg(pinned::repo().join("tools/ydb/ydb"))
        .arg(&prog)
        .args(["--y", env!("CARGO_BIN_EXE_Y"), "--nx", "--batch", "-ex"])
        .arg("python import __main__, json; print('@@' + json.dumps(__main__.Y_PROGRAM['guarantees'], sort_keys=True))")
        .current_dir(&dir)
        .output()
        .expect("run ydb");
    let text = String::from_utf8_lossy(&out.stdout);
    let carried = text.lines().find_map(|l| l.strip_prefix("@@")).unwrap_or_else(|| panic!("no table:\n{}", text));
    assert!(!written.is_empty() && written != "null", "{}", written);
    assert_eq!(carried, written, "the binary carries a different table from the one --emit-guarantees writes");
    let _ = fs::remove_dir_all(&dir);
}

/// `verify` at a stop: the proofs on the line, what they assume, and that the
/// machine code below the IR is trusted. On a kernel's line, the GPU section.
#[test]
fn verify_reports_proofs_and_what_they_assume() {
    if !have("gdb") || !have("clang") || !have("python3") {
        eprintln!("SKIP verify_reports_proofs_and_what_they_assume: no gdb, clang or python3, so this test checked NOTHING");
        return;
    }
    let (dir, prog) = setup("live", PROG, pinned::SM_PINNED);
    let (_, text) = emit(&dir, &prog, &[]);
    if no_solver(&text, "verify_reports_proofs_and_what_they_assume") {
        return;
    }
    let out = Command::new("python3")
        .arg(pinned::repo().join("tools/ydb/ydb"))
        .arg(&prog)
        .args(["--y", env!("CARGO_BIN_EXE_Y"), "--nx", "--batch"])
        .args(["-ex", &format!("break prog:{}", line_of(PROG, "vi")), "-ex", "run"])
        .args(["-ex", "echo @@here\\n", "-ex", "verify"])
        .args(["-ex", "echo @@kernel\\n", "-ex", &format!("verify {}", line_of(PROG, "out")), "-ex", "kill"])
        .current_dir(&dir)
        .output()
        .expect("run ydb");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let here = flat(text.split("@@here").nth(1).and_then(|s| s.split("@@kernel").next()).unwrap_or(""));
    assert!(here.contains("(the facts this binary carries)"), "{}", text);
    assert!(here.contains("PROVED v[i]: in bounds: the index lies in [0, 5]"), "{}", text);
    assert!(here.contains("assumes, without checking: @bounds(0, 6) on `m` (prog.ysu:"), "{}", text);
    assert!(here.contains("The code this process runs (clang -O0, from the LLVM IR):"), "{}", text);
    assert!(!here.contains("The code the GPU runs"), "a host function has no GPU section:\n{}", text);
    let kernel = flat(text.split("@@kernel").nth(1).unwrap_or(""));
    assert!(kernel.contains("NOT CHECKED Out[i] (2 accesses on this line)"), "{}", text);
    assert!(kernel.contains("The code the GPU runs (Y --emit-ptx for sm_80, then ptxas):"), "{}", text);
    assert!(kernel.contains("ptxas: the SASS is not checked against the PTX for this kernel"), "{}", text);
    let _ = fs::remove_dir_all(&dir);
}

/// `yverify.py PROGRAM LINE`: the report without gdb.
fn yverify(dir: &Path, program: &Path, line: usize) -> String {
    let out = Command::new("python3")
        .arg(pinned::repo().join("tools/ydb/yverify.py"))
        .arg(program)
        .arg(line.to_string())
        .args(["--y", env!("CARGO_BIN_EXE_Y")])
        .current_dir(dir)
        .output()
        .expect("run yverify.py");
    flat(&format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

/// The repository's proof and validation of `exact_pv` are credited to a
/// kernel that IS `tests/exact_pv.ptx`'s, and not to one with the same name
/// and one instruction changed.
#[test]
fn a_committed_kernel_is_credited_only_when_its_ptx_is_the_same() {
    if !have("python3") {
        eprintln!("SKIP a_committed_kernel_is_credited_only_when_its_ptx_is_the_same: no python3, so this test checked NOTHING");
        return;
    }
    // The artifact's own target: the comparison includes `.target`.
    let dir = pinned::pinned_scratch("yv_exact_pv", pinned::SM_FP8);
    let prog = pinned::copy_fixture(&dir, "tests/exact_pv.ysu");
    let src = fs::read_to_string(&prog).unwrap();
    let acc = src.lines().position(|l| l.contains("acc = acc + pv * vv;")).expect("the accumulation") + 1;
    let text = yverify(&dir, &prog, acc);
    if no_solver(&text, "a_committed_kernel_is_credited_only_when_its_ptx_is_the_same") {
        return;
    }
    assert!(text.contains("SAME PTX this compile's kernel is tests/exact_pv.ptx's"), "{}", text);
    assert!(text.contains("PROVED proofs/ExactPvExact.v"), "{}", text);
    assert!(text.contains("Hacc: the accumulator licence"), "the proof's own hypotheses are quoted:\n{}", text);
    assert!(text.contains("nothing checks them at launch"), "{}", text);
    assert!(text.contains("row o1/exact_pv"), "{}", text);
    assert!(text.contains("VALIDATED the SASS ptxas -O1 makes for sm_89: loopval.py"), "{}", text);
    // There is no binary here, so nothing is said about one.
    assert!(!text.contains("The code this process runs"), "{}", text);

    // The control: same kernel name, the multiply's operands swapped.
    fs::write(&prog, src.replace("acc = acc + pv * vv;", "acc = acc + vv * pv;")).unwrap();
    let text = yverify(&dir, &prog, acc);
    assert!(text.contains("NOT COVERED tests/exact_pv.ptx's proofs and validation are about the kernel"), "{}", text);
    assert!(!text.contains("VALIDATED"), "a different kernel was credited with exact_pv's validation:\n{}", text);
    assert!(!text.contains("proofs/ExactPvExact.v"), "a different kernel was credited with exact_pv's proof:\n{}", text);
    assert!(text.contains("ptxas: the SASS is not checked against the PTX for this kernel"), "{}", text);
    let _ = fs::remove_dir_all(&dir);
}

/// A kernel the PTX backend replaces wholesale is credited with the proofs of
/// that lowering - whatever its shape - and its body's lines become no PTX.
#[test]
fn a_lowering_is_credited_with_the_proofs_about_it() {
    if !have("python3") {
        eprintln!("SKIP a_lowering_is_credited_with_the_proofs_about_it: no python3, so this test checked NOTHING");
        return;
    }
    let dir = pinned::pinned_scratch("yv_int8", pinned::SM_FP8);
    let prog = pinned::copy_fixture(&dir, "tests/int8_gemm.ysu");
    let src = fs::read_to_string(&prog).unwrap();
    let k = src.lines().position(|l| l.starts_with("kernel int8_gemm")).expect("the kernel") + 1;
    let text = yverify(&dir, &prog, k);
    if no_solver(&text, "a_lowering_is_credited_with_the_proofs_about_it") {
        return;
    }
    assert!(text.contains("REPLACED kernel int8_gemm is replaced wholesale by Y's int8 tensor-core GEMM"), "{}", text);
    assert!(text.contains("PROVED proofs/Int8GemmSchedule.v"), "{}", text);
    assert!(text.contains("PROVED proofs/Int8GemmExact.v"), "{}", text);
    assert!(text.contains("for K up to 133,120"), "{}", text);
    // The control: an ordinary kernel is replaced by nothing and proved by nothing.
    let (dir2, prog2) = setup("plainkernel", PROG, pinned::SM_PINNED);
    let text = yverify(&dir2, &prog2, line_of(PROG, "out"));
    assert!(!text.contains("REPLACED") && !text.contains("proofs/"), "{}", text);
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&dir2);
}

/// `Y_ALLOW_UNVERIFIED_INVARIANTS` lets a program through with an invariant
/// the verifier could not check; the table must say UNVERIFIED, not proved.
#[test]
fn an_unverified_invariant_is_not_reported_as_proved() {
    let src = "\
@unsafe
fn bump(x: &mut I32) -> I32 {
    *x = *x + 1;
    return 0;
}

fn main() -> I32 {
    let s: I32 = 0;
    @invariant(i >= 0)
    for i in 0..3 { // L:loop
        let r: I32 = bump(&mut s);
    }
    return s;
}
";
    let (dir, prog) = setup("unverified", src, pinned::SM_PINNED);
    let (table, text) = emit(&dir, &prog, &[]);
    assert!(table.is_none(), "an invariant the verifier cannot model is refused:\n{}", text);
    let (table, text) = emit(&dir, &prog, &[("Y_ALLOW_UNVERIFIED_INVARIANTS", "1")]);
    let table = table.unwrap_or_else(|| panic!("{}", text));
    let inv = fact(&table, "invariant", line_of(src, "loop"), "@invariant(i >= 0)");
    assert_eq!(status(&inv), "unverified", "{}", inv);
    assert!(inv.contains("passes a reference to a call"), "{}", inv);
    let _ = fs::remove_dir_all(&dir);
}

/// The flag writes a table and compiles nothing, so a backend or debug flag
/// beside it is refused rather than ignored.
#[test]
fn emit_guarantees_refuses_what_it_would_ignore() {
    let (dir, prog) = setup("refuse", PROG, pinned::SM_PINNED);
    for flag in ["-g", "--emit-ptx", "--emit-llvm", "-O2"] {
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&prog)
            .args(["--emit-guarantees", flag])
            .current_dir(&dir)
            .output()
            .expect("run Y");
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(!out.status.success(), "`{}` was accepted beside --emit-guarantees:\n{}", flag, text);
        assert!(text.contains(&format!("so {} cannot be combined with it", flag)), "{}", text);
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The host backend's GEMM substitution is a fact about what RUNS: the exact
/// kernel is proved (and rests on the operands' trusted `@bounds`), the f32
/// one is tested. Each fact appears exactly when the backend substituted.
#[test]
fn a_substituted_kernel_is_a_fact_about_the_code() {
    let exact = "\
kernel exact_matmul(A: GlobalMemory<I16>, B: GlobalMemory<I16>, C: GlobalMemory<I64>, M: I32, N: I32, K: I32) {
    @invariant(i >= 0)
    for i in 0..M step 1 {
        @invariant(j >= 0)
        for j in 0..N step 1 {
            @ZeroDrift
            let mut sum: I64 = 0;
            @invariant(k >= 0)
            for k in 0..K step 1 {
                @bounds(min=-1024, max=1024)
                let a_val: I64 = block_ptr2d_load(A, i, k, K, M, K);
                @bounds(min=-1024, max=1024)
                let b_val: I64 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a_val * b_val;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}

kernel f32_matmul(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, M: I32, N: I32, K: I32) {
    @invariant(i >= 0)
    for i in 0..M step 1 {
        @invariant(j >= 0)
        for j in 0..N step 1 {
            let mut sum: F32 = 0.0;
            @invariant(k >= 0)
            for k in 0..K step 1 {
                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);
                let b_val: F32 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a_val * b_val;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}

fn main() {
}
";
    let (dir, prog) = setup("gemm", exact, pinned::SM_PINNED);
    let ll = dir.join("prog.ll");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&prog)
        .args(["-g", "--emit-llvm", "-o"])
        .arg(&ll)
        .current_dir(&dir)
        .env("Y_NO_CERTIFICATE", "1")
        .output()
        .expect("run Y");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if no_solver(&text, "a_substituted_kernel_is_a_fact_about_the_code") {
        return;
    }
    assert!(out.status.success(), "{}", text);
    let ll_text = fs::read_to_string(&ll).unwrap();
    let table = carried_table(&ll_text);
    let gemm_of = |kernel: &str| -> Vec<String> {
        facts(&table)
            .into_iter()
            .filter(|f| f.contains("\"kind\": \"gemm\"") && f.contains(&format!("\"what\": \"kernel {}\"", kernel)))
            .collect()
    };
    // The packed f32 kernel: TESTED, not proved, exactly when it was substituted.
    let f32_body = ll_text.split("define void @f32_matmul").nth(1).and_then(|s| s.split("\n}").next()).unwrap_or("");
    let f32_gemm = gemm_of("f32_matmul");
    if f32_body.contains("[Y CPU GEMM]") {
        assert_eq!(f32_gemm.len(), 1, "{}", table);
        assert_eq!(status(&f32_gemm[0]), "tested", "{}", f32_gemm[0]);
        assert!(f32_gemm[0].contains("NOT bit-identical"), "{}", f32_gemm[0]);
    } else {
        assert!(f32_gemm.is_empty(), "{}", table);
    }
    let substituted = text.contains("EXACT vpdpwssd kernel substituted");
    let gemm = gemm_of("exact_matmul");
    if substituted {
        assert_eq!(gemm.len(), 1, "{}", table);
        assert_eq!(status(&gemm[0]), "proved", "{}", gemm[0]);
        assert!(gemm[0].contains("@bounds(-1024, 1024) on `a_val`") && gemm[0].contains("@bounds(-1024, 1024) on `b_val`"),
                "the exactness claim rests on the operands' trusted ranges:\n{}", gemm[0]);
    } else {
        eprintln!("NOTE: the exact kernel was not substituted on this machine (no AVX-512 VNNI), so only its absence is checked");
        assert!(gemm.is_empty(), "a substitution that did not happen is reported:\n{}", table);
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The table a `.ll` built with `-g` carries: the JSON inside
/// `__import__("json").loads("...")` in `.debug_gdb_scripts`.
fn carried_table(ll: &str) -> String {
    let line = ll.lines().find(|l| l.starts_with("@__y_debug_gdb_scripts")).expect("the gdb script");
    let start = line.find("c\"").unwrap() + 2;
    let end = line.rfind("\\00\"").unwrap();
    let raw = line[start..end].as_bytes();
    let mut bytes = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'\\' {
            bytes.push(u8::from_str_radix(std::str::from_utf8(&raw[i + 1..i + 3]).unwrap(), 16).unwrap());
            i += 3;
        } else {
            bytes.push(raw[i]);
            i += 1;
        }
    }
    let script = String::from_utf8(bytes).unwrap();
    let s = script.find("loads(\"").expect("the guarantees") + "loads(\"".len();
    // A Python string literal: a backslash quotes the next character.
    let mut out = String::new();
    let mut chars = script[s..].chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => out.push(chars.next().unwrap()),
            '"' => break,
            c => out.push(c),
        }
    }
    out
}

/// The tables `verify` reads by - the lowerings and their proofs, the fixture
/// proofs, the rows of `regress.sh` - checked against the repository, each
/// with a control.
#[test]
fn the_evidence_tables_match_the_repository() {
    if !have("python3") {
        eprintln!("SKIP the_evidence_tables_match_the_repository: no python3, so this test checked NOTHING");
        return;
    }
    let out = Command::new("python3")
        .arg(pinned::repo().join("tools/ydb/yverify.py"))
        .arg("--selftest")
        .output()
        .expect("run yverify.py --selftest");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && text.contains("selftest: ok"), "{}", text);
}
