//! A `for` header is evaluated once, before the first iteration, and the loop
//! variable is an I32 compared and stepped in 32 bits - on every backend.
//!
//! Measured before the fix, on the programs below:
//!
//! * **The step.** `for i in 0..20 step k { k = k + 1; }` ran 5 iterations on
//!   the LLVM backend (and the JIT, which compiles its IR), which re-read the
//!   step after every body, and 20 on the GPU, which reads it once.
//! * **`--emit-cpu`** wrote `let mut i = start; while i < end { ... i += 1; }`:
//!   every step that was not a literal became 1 (`step k` with `k = 2` ran 10
//!   iterations, not 5), `end` was re-read each iteration (a body shrinking it
//!   ran 5, not 10), and Rust inferred `i: u32` from a `U32` bound, so a bound
//!   of 3e9 ran 3e9 times where every other backend compares the bound's bits
//!   as an I32 and runs none. A `U8` or `U16` parameter did not compile at all:
//!   it was written under its Y name.
//! * **`step 0`** was refused by the PTX backend and compiled by the LLVM and
//!   CPU backends into a loop that never ends.
//!
//! Each host path runs the same program against constants; the GPU runs a
//! kernel whose step is a register, which is the path whose evaluation moved.
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const PROGRAM: &str = r#"
@unsafe
fn varstep(k: I32) -> I32 {
    let c: I32 = 0;
    for i in 0..10 step k { c = c + 1; }
    return c;
}

@unsafe
fn shrink() -> I32 {
    let n: I32 = 10;
    let c: I32 = 0;
    for i in 0..n { n = n - 1; c = c + 1; }
    return c;
}

@unsafe
fn grow() -> I32 {
    let k: I32 = 1;
    let c: I32 = 0;
    for i in 0..20 step k { k = k + 1; c = c + 1; }
    return c;
}

@unsafe
fn below8(n: U8) -> I32 {
    let c: I32 = 0;
    for i in 0..n { c = c + 1; }
    return c;
}

@unsafe
fn below16(n: U16) -> I32 {
    let c: I32 = 0;
    for i in 0..n { c = c + 1; }
    return c;
}

@unsafe
fn below32(n: U32) -> I32 {
    let c: I32 = 0;
    for i in 0..n { c = c + 1; }
    return c;
}

fn main() -> I32 {
    print_int(varstep(2) * 10000 + shrink() * 100 + grow());
    println("");
    print_int(below8(200) * 1000000 + below16(60000));
    println("");
    print_int(below32(3000000000));
    println("");
    return 0;
}
"#;

/// `varstep(2) = 5`, `shrink() = 10`, `grow() = 20`; `below8(200) = 200`,
/// `below16(60000) = 60000`; `below32(3e9) = 0` (its bits read as I32 are
/// negative).
const EXPECTED: [i64; 3] = [5 * 10000 + 10 * 100 + 20, 200 * 1000000 + 60000, 0];

/// The lines of `out` that are exactly one integer: the program's output
/// among the compiler's.
fn printed(out: &str) -> Vec<i64> {
    out.lines().filter_map(|l| l.trim().parse().ok()).collect()
}

fn source(tag: &str, src: &str) -> std::path::PathBuf {
    let dir = pinned::pinned_scratch(&format!("forhdr_{tag}"), pinned::SM_PINNED);
    let path = dir.join(format!("{tag}.ysu"));
    std::fs::write(&path, src).unwrap();
    path
}

fn y(path: &std::path::Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(path)
        .args(args)
        .current_dir(path.parent().unwrap())
        .output()
        .expect("run Y");
    (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

#[test]
fn the_llvm_backend_reads_the_header_once() {
    let path = source("llvm", PROGRAM);
    let bin = path.with_extension("bin");
    let (ok, out) = y(&path, &["-o", bin.to_str().unwrap()]);
    assert!(ok, "{out}");
    let run = Command::new(&bin).output().expect("run the program");
    assert_eq!(printed(&String::from_utf8_lossy(&run.stdout)), EXPECTED);
}

#[test]
fn the_jit_reads_the_header_once() {
    let path = source("jit", PROGRAM);
    let (ok, out) = y(&path, &["--jit"]);
    assert!(ok, "{out}");
    assert_eq!(printed(&out), EXPECTED, "{out}");
}

#[test]
fn emit_cpu_reads_the_header_once_and_compares_in_32_bits() {
    let path = source("cpu", PROGRAM);
    let (ok, out) = y(&path, &["--emit-cpu"]);
    assert!(ok, "{out}");
    let mut lines = out.lines().skip_while(|l| !l.contains("GENERATED RUST BLOB")).skip(1);
    let blob: Vec<&str> = lines
        .by_ref()
        .take_while(|l| !(l.trim_start_matches('=').is_empty() && l.len() > 20 && !l.starts_with("//")))
        .collect();
    let blob = blob.join("\n").replace("pub fn main() -> i32", "pub fn y_main() -> i32");
    assert!(blob.contains("pub fn y_main() -> i32"), "no `main` to drive:\n{blob}");
    let dir = path.parent().unwrap();
    let rs = dir.join("blob.rs");
    std::fs::write(
        &rs,
        format!("#![allow(unused, non_snake_case)]\n{blob}\nfn main() {{ std::process::exit(y_main()); }}\n"),
    )
    .unwrap();
    let bin = dir.join("blob");
    let Ok(built) = Command::new("rustc").args(["--edition", "2021", "-O", "-o"]).arg(&bin).arg(&rs).output() else {
        eprintln!("SKIP: no rustc, so the emitted Rust was not run");
        return;
    };
    assert!(built.status.success(), "the emitted Rust does not compile:\n{}\n{blob}", String::from_utf8_lossy(&built.stderr));
    let run = Command::new(&bin).output().expect("run the blob");
    assert_eq!(printed(&String::from_utf8_lossy(&run.stdout)), EXPECTED, "{blob}");
}

/// `step 0` is refused by name on every backend instead of becoming a loop
/// that never ends on two of them.
#[test]
fn a_zero_step_is_refused_everywhere() {
    let host = "fn main() -> I32 {\n    let c: I32 = 0;\n    @invariant(c >= 0)\n    for i in 0..10 step 0 { c = 1; }\n    return c;\n}\n";
    let kernel = "kernel k(Out: GlobalMemory<I32>) {\n    let c: I32 = 0;\n    @invariant(c >= 0)\n    for i in 0..10 step 0 { c = 1; }\n    Out[0] = c;\n}\n\nfn main() {}\n";
    for (tag, src, args) in [
        ("zero_llvm", host, vec!["--emit-llvm"]),
        ("zero_cpu", host, vec!["--emit-cpu"]),
        ("zero_jit", host, vec!["--jit"]),
        ("zero_ptx", kernel, vec!["--emit-ptx"]),
    ] {
        let path = source(tag, src);
        let (ok, out) = y(&path, &args);
        assert!(!ok, "{tag}: `step 0` was accepted:\n{out}");
        assert!(out.contains("`step 0` never advances"), "{tag}: not refused by name:\n{out}");
    }
}

/// On the device: a step held in a register, the path whose evaluation
/// moved after `start` and `end`, still steps by its value.
#[test]
fn a_register_step_on_the_device() {
    let src = "kernel k(Out: GlobalMemory<I32>) {\n    let k2: I32 = 2;\n    let c: I32 = 0;\n    \
               @invariant(c >= 0 && c <= i)\n    for i in 0..10 step k2 { c = c + 1; }\n    Out[0] = c;\n}\n\nfn main() {}\n";
    let path = source("device", src);
    let ptx_path = path.with_extension("ptx");
    let (ok, out) = y(&path, &["--emit-ptx", "-o", ptx_path.to_str().unwrap()]);
    assert!(ok, "{out}");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap();
    let Some(ctx) = y::cuda_runtime::CudaContext::new() else {
        eprintln!("SKIP: no CUDA device, so the kernel was not run");
        return;
    };
    let module = ctx.load_ptx(&ptx, "k").expect("load");
    let buf = ctx.alloc(4).unwrap();
    ctx.memcpy_htod_at(&buf, 0, &(-1i32).to_le_bytes()).unwrap();
    ctx.launch(&module, (1, 1, 1), (1, 1, 1), 0, &[buf.device_ptr()]).expect("launch");
    ctx.synchronize().unwrap();
    let mut got = [0u8; 4];
    ctx.memcpy_dtoh_at(&mut got, &buf, 0).unwrap();
    assert_eq!(i32::from_le_bytes(got), 5, "0..10 step 2 is five iterations");
}
