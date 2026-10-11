//! A `@bounds` the compiler cannot prove is checked when the program runs.
//!
//! It used to be taken on trust. `@bounds(min=0, max=3) let i: I32 =
//! get(1000000);` compiled, every proof using `i`'s range rested on the
//! annotation, and the index it fed skipped its own run-time check: `arr[i] =
//! 42` wrote a million elements past a four-element array under `@safe` and
//! the program exited 0. An exact-GEMM operand outside its `@bounds` overflowed
//! the int32 accumulator into a wrong answer under a certificate claiming
//! exactness.
//!
//! Now every backend that lowers such a `let` tests the stored value: the LLVM
//! backend and the JIT print it and exit 1, PTX traps, `--emit-cpu`'s Rust
//! panics, and `--emit-native` - which emits no branches - refuses the `let`.
//! The exact GEMM never runs its operand `let`s, so it scans both operands
//! before computing anything. Each case has an in-range control: a check that
//! stopped everything would pass the out-of-range half on its own.
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const ARRAY: &str = "@unsafe
fn get(n: I32) -> I32 {
    return n;
}

fn main() -> I32 {
    let r: I32 = 0;
    @safe {
        let arr: [I32; 4] = {};
        @bounds(min=0, max=3)
        let i: I32 = get(VALUE);
        arr[i] = 42;
        r = arr[i] - 40;
    }
    return r;
}
";

fn program(template: &str, value: &str) -> String {
    template.replace("VALUE", value)
}

fn source(tag: &str, src: &str) -> PathBuf {
    let dir = pinned::pinned_scratch(&format!("bndrt_{tag}"), pinned::SM_PINNED);
    let path = dir.join(format!("{tag}.ysu"));
    std::fs::write(&path, src).unwrap();
    path
}

fn y(path: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(path)
        .args(args)
        .current_dir(path.parent().unwrap())
        .output()
        .expect("run Y");
    (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

/// Build with the default backend and run: (exit code, output).
fn run_llvm(tag: &str, src: &str) -> (Option<i32>, String) {
    let path = source(tag, src);
    let bin = path.with_extension("bin");
    let (ok, out) = y(&path, &["-o", bin.to_str().unwrap()]);
    assert!(ok, "{tag}: {out}");
    let run = Command::new(&bin).output().expect("run the program");
    (run.status.code(), format!("{}{}", String::from_utf8_lossy(&run.stdout), String::from_utf8_lossy(&run.stderr)))
}

#[test]
fn the_llvm_backend_stops_a_value_outside_its_bounds() {
    let (code, out) = run_llvm("llvm_out", &program(ARRAY, "1000000"));
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("Y: line 11: 1000000 lies outside @bounds(0, 3); stopping"), "{out}");
    let (code, out) = run_llvm("llvm_in", &program(ARRAY, "2"));
    assert_eq!(code, Some(2), "the in-range value runs through: {out}");
}

#[test]
fn the_jit_stops_it_too() {
    for (value, code, says) in [("1000000", Some(1), true), ("2", Some(2), false)] {
        let path = source(&format!("jit_{value}"), &program(ARRAY, value));
        let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&path).arg("--jit").current_dir(path.parent().unwrap()).output().unwrap();
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert_eq!(out.status.code(), code, "{text}");
        assert_eq!(text.contains("lies outside @bounds(0, 3)"), says, "{text}");
    }
}

/// Unsigned values compare unsigned (3e9 is not a negative I32 here), a float
/// compares unordered (a NaN is outside), and a Q format at its scale.
#[test]
fn every_scalar_kind_compares_as_its_own_type() {
    let cases = [
        ("u32_out", "@unsafe\nfn get(n: U32) -> U32 {\n    return n;\n}\n\nfn main() -> I32 {\n    @bounds(0, 100)\n    let u: U32 = get(3000000000);\n    return 7;\n}\n", Some(1)),
        ("u32_in", "@unsafe\nfn get(n: U32) -> U32 {\n    return n;\n}\n\nfn main() -> I32 {\n    @bounds(0, 100)\n    let u: U32 = get(50);\n    return 7;\n}\n", Some(7)),
        // Read as an I32 (sign-extended) 3e9 is negative and would stop here.
        ("u32_high_in", "@unsafe\nfn get(n: U32) -> U32 {\n    return n;\n}\n\nfn main() -> I32 {\n    @bounds(0, 4000000000)\n    let u: U32 = get(3000000000);\n    return 7;\n}\n", Some(7)),
        ("f32_nan", "@unsafe\nfn get(n: F32) -> F32 {\n    return n;\n}\n\nfn main() -> I32 {\n    let z: F32 = 0.0;\n    @bounds(0, 1)\n    let x: F32 = get(z / z);\n    return 9;\n}\n", Some(1)),
        ("f32_in", "@unsafe\nfn get(n: F32) -> F32 {\n    return n;\n}\n\nfn main() -> I32 {\n    @bounds(0, 1)\n    let x: F32 = get(0.5);\n    return 9;\n}\n", Some(9)),
        ("q_out", "@unsafe\nfn get(n: Q16.16) -> Q16.16 {\n    return n;\n}\n\nfn main() -> I32 {\n    @bounds(0, 10)\n    let q: Q16.16 = get(12.5);\n    return 5;\n}\n", Some(1)),
        ("q_in", "@unsafe\nfn get(n: Q16.16) -> Q16.16 {\n    return n;\n}\n\nfn main() -> I32 {\n    @bounds(0, 10)\n    let q: Q16.16 = get(9.5);\n    return 5;\n}\n", Some(5)),
    ];
    for (tag, src, code) in cases {
        let (got, out) = run_llvm(tag, src);
        assert_eq!(got, code, "{tag}: {out}");
        assert_eq!(out.contains("lies outside @bounds"), code == Some(1), "{tag}: {out}");
    }
}

/// A bound the checker proves costs nothing: no test is emitted.
#[test]
fn a_proved_bound_emits_no_check() {
    let proved = "fn main() -> I32 {\n    @bounds(0, 3)\n    let i: I32 = 2;\n    return i;\n}\n";
    let path = source("proved", proved);
    let (ok, out) = y(&path, &["--emit-llvm"]);
    assert!(ok, "{out}");
    let ir = std::fs::read_to_string(path.with_extension("ll")).unwrap();
    assert!(!ir.contains("bounds.stop") && !ir.contains("@.y_bounds_"), "{ir}");
    let path = source("unproved", &program(ARRAY, "2"));
    let (ok, out) = y(&path, &["--emit-llvm"]);
    assert!(ok, "{out}");
    let ir = std::fs::read_to_string(path.with_extension("ll")).unwrap();
    assert!(ir.contains("bounds.stop"), "the control: an unproved bound IS checked:\n{ir}");
}

#[test]
fn emit_cpu_panics_and_native_refuses() {
    let template = "@unsafe\nfn get(n: I32) -> I32 {\n    return n;\n}\n\nfn main() -> I32 {\n    @bounds(min=0, max=3)\n    let i: I32 = get(VALUE);\n    return i;\n}\n";
    for (value, code) in [("1000000", Some(101)), ("2", Some(2))] {
        let path = source(&format!("cpu_{value}"), &program(template, value));
        let (ok, out) = y(&path, &["--emit-cpu"]);
        assert!(ok, "{out}");
        let mut lines = out.lines().skip_while(|l| !l.contains("GENERATED RUST BLOB")).skip(1);
        let blob: Vec<&str> = lines
            .by_ref()
            .take_while(|l| !(l.trim_start_matches('=').is_empty() && l.len() > 20 && !l.starts_with("//")))
            .collect();
        let blob = blob.join("\n").replace("pub fn main() -> i32", "pub fn y_main() -> i32");
        assert!(blob.contains("y_within(i, 0, 3, 8);"), "{blob}");
        let dir = path.parent().unwrap();
        let rs = dir.join("blob.rs");
        std::fs::write(&rs, format!("#![allow(unused, non_snake_case)]\n{blob}\nfn main() {{ std::process::exit(y_main()); }}\n")).unwrap();
        let bin = dir.join("blob");
        let Ok(built) = Command::new("rustc").args(["--edition", "2021", "-O", "-o"]).arg(&bin).arg(&rs).output() else {
            eprintln!("SKIP: no rustc, so the emitted Rust was not run");
            return;
        };
        assert!(built.status.success(), "{}\n{blob}", String::from_utf8_lossy(&built.stderr));
        let run = Command::new(&bin).output().unwrap();
        assert_eq!(run.status.code(), code, "{}", String::from_utf8_lossy(&run.stderr));
        assert_eq!(
            String::from_utf8_lossy(&run.stderr).contains("Y: line 8: 1000000 lies outside @bounds(0, 3); stopping"),
            code == Some(101)
        );
    }
    let path = source("native", &program(template, "2"));
    let (ok, out) = y(&path, &["--emit-native", "-o", path.with_extension("elf").to_str().unwrap()]);
    assert!(!ok && out.contains("a `@bounds` checked when the program runs"), "{out}");
}

/// On the device: a value loaded from memory, bounded by `@bounds`, traps
/// outside the range; inside it the kernel computes. Each launch runs in a
/// child process (this test, re-run with `Y_BOUNDS_GPU_CASE`): a trap leaves
/// the device unusable for the rest of its process, so even a fresh context
/// cannot be created after one.
#[test]
fn the_gpu_traps_a_value_outside_its_bounds() {
    if let Ok(case) = std::env::var("Y_BOUNDS_GPU_CASE") {
        let (ptx_path, input) = case.split_once('|').unwrap();
        let ptx = std::fs::read_to_string(ptx_path).unwrap();
        let ctx = y::cuda_runtime::CudaContext::new().expect("a context");
        let module = ctx.load_ptx(&ptx, "k").expect("load");
        let (a, o) = (ctx.alloc(4).unwrap(), ctx.alloc(4).unwrap());
        ctx.memcpy_htod_at(&a, 0, &input.parse::<u32>().unwrap().to_le_bytes()).unwrap();
        ctx.memcpy_htod_at(&o, 0, &(-1i32).to_le_bytes()).unwrap();
        match ctx.launch(&module, (1, 1, 1), (1, 1, 1), 0, &[a.device_ptr(), o.device_ptr()]).and_then(|_| ctx.synchronize()) {
            Ok(()) => {
                let mut got = [0u8; 4];
                ctx.memcpy_dtoh_at(&mut got, &o, 0).unwrap();
                println!("RESULT {}", i32::from_le_bytes(got));
            }
            Err(e) => println!("TRAPPED {e}"),
        }
        return;
    }
    if y::cuda_runtime::CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA device, so no kernel was run");
        return;
    }
    let kernels = [
        ("i32", "kernel k(A: GlobalMemory<I32>, Out: GlobalMemory<I32>) {\n    @bounds(min=0, max=3)\n    let i: I32 = A[0];\n    Out[0] = i + 1;\n}\n\nfn main() {}\n"),
        ("u32", "kernel k(A: GlobalMemory<U32>, Out: GlobalMemory<I32>) {\n    @bounds(min=0, max=100)\n    let u: U32 = A[0];\n    Out[0] = 7;\n}\n\nfn main() {}\n"),
        ("f32", "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<I32>) {\n    @bounds(min=0, max=1)\n    let x: F32 = A[0];\n    Out[0] = 9;\n}\n\nfn main() {}\n"),
    ];
    let cases: [(&str, u32, Option<i32>); 6] = [
        ("i32", 2, Some(3)),
        ("i32", 1_000_000, None),
        ("u32", 50, Some(7)),
        ("u32", 3_000_000_000, None),
        ("f32", 0.5f32.to_bits(), Some(9)),
        ("f32", f32::NAN.to_bits(), None),
    ];
    for (ty, input, want) in cases {
        let src = kernels.iter().find(|(t, _)| *t == ty).unwrap().1;
        let path = source(&format!("gpu_{ty}_{input}"), src);
        let ptx_path = path.with_extension("ptx");
        let (ok, out) = y(&path, &["--emit-ptx", "-o", ptx_path.to_str().unwrap()]);
        assert!(ok, "{out}");
        assert!(std::fs::read_to_string(&ptx_path).unwrap().contains("trap;"));
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "the_gpu_traps_a_value_outside_its_bounds", "--nocapture"])
            .env("Y_BOUNDS_GPU_CASE", format!("{}|{input}", ptx_path.display()))
            .output()
            .expect("run the launch in a child");
        let text = String::from_utf8_lossy(&child.stdout).into_owned();
        match want {
            Some(v) => assert!(text.contains(&format!("RESULT {v}\n")), "{ty} {input}: {text}"),
            None => assert!(text.contains("TRAPPED"), "{ty} {input}: a value outside its @bounds ran to completion: {text}"),
        }
    }
}

const GEMM: &str = "kernel y_matmul(A: GlobalMemory<I16>, B: GlobalMemory<I16>, C: GlobalMemory<I64>, M: I32, N: I32, K: I32) {
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

fn main() {
}
";

/// `big` puts 32767 in a row of A and all of B: enough to overflow the int32
/// accumulator, which without the scan gave a wrong answer and exit 0.
/// `alias` makes B overlap A, so the body runs as written and its operand
/// `let` stops at the 5000 planted in A.
const GEMM_DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
void y_matmul(const int16_t *A, const int16_t *B, int64_t *C, int M, int N, int K);
int main(int argc, char **argv) {
    int M = 53, N = 71, K = 299;
    const char *mode = argc > 1 ? argv[1] : "";
    int16_t *buf = calloc((size_t)(M*K + K*N + 64), 2);
    int16_t *A = buf, *B = strcmp(mode, "alias") ? malloc((size_t)K*N*2) : buf + 3;
    int64_t *C = malloc((size_t)M*N*8), *R = malloc((size_t)M*N*8);
    for (long i = 0; i < (long)M*K; ++i) A[i] = (int16_t)((i * 37) % 2049 - 1024);
    if (strcmp(mode, "alias")) for (long i = 0; i < (long)K*N; ++i) B[i] = (int16_t)((i * 53) % 2049 - 1024);
    if (!strcmp(mode, "big")) { for (int k = 0; k < K; ++k) A[5L*K + k] = 32767; for (long i = 0; i < (long)K*N; ++i) B[i] = 32767; }
    if (!strcmp(mode, "alias")) A[5L*K + 7] = 5000;
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        int64_t a = 0;
        for (int k = 0; k < K; ++k) a += (int64_t)A[(long)i*K+k] * (int64_t)B[(long)k*N+j];
        R[(long)i*N+j] = a;
    }
    y_matmul(A, B, C, M, N, K);
    printf("reference %s\n", memcmp(C, R, (size_t)M*N*8) ? "MISMATCH" : "OK");
    return 0;
}
"#;

#[test]
fn the_exact_gemm_scans_its_operands_before_computing() {
    let have_clang = Command::new("clang").arg("--version").output().is_ok_and(|o| o.status.success());
    if !have_clang {
        eprintln!("SKIP: no clang, so the exact GEMM was not linked and run");
        return;
    }
    let path = source("gemm", GEMM);
    let (ok, out) = y(&path, &["--emit-llvm"]);
    assert!(ok, "{out}");
    let dir = path.parent().unwrap();
    let ll = path.with_extension("ll");
    let ir = std::fs::read_to_string(&ll).unwrap();
    let driver = dir.join("drv.c");
    std::fs::write(&driver, GEMM_DRIVER).unwrap();
    let exe = dir.join("run");
    let cc = Command::new("clang")
        .args([ll.to_str().unwrap(), driver.to_str().unwrap(), "-O2", "-lm", "-lpthread", "-o", exe.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(cc.status.success(), "{}", String::from_utf8_lossy(&cc.stderr));
    let run = |mode: &str| {
        let r = Command::new(&exe).arg(mode).output().unwrap();
        (r.status.code(), format!("{}{}", String::from_utf8_lossy(&r.stdout), String::from_utf8_lossy(&r.stderr)))
    };
    let (code, text) = run("in");
    assert_eq!((code, text.trim()), (Some(0), "reference OK"), "operands inside their bounds");
    // The body as written, when the buffers overlap: its operand `let` stops.
    let (code, text) = run("alias");
    assert_eq!(code, Some(1), "{text}");
    assert!(text.contains("5000 lies outside @bounds(-1024, 1024); stopping"), "{text}");
    if !out.contains("EXACT vpdpwssd kernel substituted") {
        eprintln!("NOTE: no AVX-512 VNNI here, so the exact kernel was not substituted and only the body's checks ran");
        return;
    }
    assert!(ir.contains("call void @__y_operand_bounds"), "{ir}");
    let (code, text) = run("big");
    assert_eq!(code, Some(1), "{text}");
    assert!(
        text.contains("operand A[5][0] is 32767, outside its @bounds(-1024, 1024); stopping before computing anything"),
        "{text}"
    );
}
