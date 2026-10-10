//! A product bound with `let` is rounded before it is added.
//!
//! `let p: F32 = a * b; let s: F32 = p + c;` binds a rounded F32 and adds
//! it, which is what the LLVM and CPU backends compute. The PTX backend
//! emitted `mul.f32` and `add.f32` with no rounding modifier, and ptxas is
//! free to contract exactly that pair: it emitted ONE `FFMA` (one rounding)
//! under PTX text stating two. Only the syntactic `a * b + c` is meant to be
//! fused - the emitter writes `fma.rn` for it (manual §20, `try_emit_fma`).
//!
//! A multiply the emitter does not fuse is `mul.rn` now, and ptxas does not
//! contract an instruction with an explicit rounding modifier. Checked three
//! ways: the PTX text, the SASS ptxas makes of it, and the value on the
//! device for inputs where one and two roundings differ:
//! `a = b = 1 + 2^-12`, `c = -(1 + 2^-11)` gives `0` rounded twice and
//! `2^-24` fused.
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;
#[path = "common/ptxas.rs"]
mod ptxas;

const LET_BOUND: &str = "kernel k(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, Out: GlobalMemory<F32>) {\n    \
    let a: F32 = A[0];\n    let b: F32 = B[0];\n    let c: F32 = C[0];\n    \
    let p: F32 = a * b;\n    let s: F32 = p + c;\n    Out[0] = s;\n}\n\nfn main() {}\n";

const SYNTACTIC: &str = "kernel k(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, Out: GlobalMemory<F32>) {\n    \
    let a: F32 = A[0];\n    let b: F32 = B[0];\n    let c: F32 = C[0];\n    \
    Out[0] = a * b + c;\n}\n\nfn main() {}\n";

fn ptx(tag: &str, src: &str) -> String {
    let dir = pinned::pinned_scratch(&format!("letround_{tag}"), pinned::SM_PINNED);
    let path = dir.join("k.ysu");
    std::fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-ptx")
        .arg("-o")
        .arg(dir.join("k.ptx"))
        .current_dir(&dir)
        .output()
        .expect("run Y");
    assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    std::fs::read_to_string(dir.join("k.ptx")).expect("no .ptx written")
}

/// The instruction lines of a module, comments dropped.
fn code(ptx: &str) -> Vec<String> {
    ptx.lines()
        .map(|l| l.split("//").next().unwrap().trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// ptxas's SASS for `ptx`, or `None` without the CUDA tools.
fn sass(tag: &str, ptx: &str) -> Option<String> {
    let ptxas = ptxas::ptxas()?;
    // Beside ptxas, or on PATH when ptxas was found there by bare name.
    let cuobjdump = ptxas.with_file_name("cuobjdump");
    if !Command::new(&cuobjdump).arg("--version").output().is_ok_and(|o| o.status.success()) {
        return None;
    }
    let dir = pinned::scratch(&format!("letround_sass_{tag}"));
    std::fs::write(dir.join("k.ptx"), ptx).unwrap();
    let built = Command::new(&ptxas)
        .args(["-arch=sm_80", "-o"])
        .arg(dir.join("k.cubin"))
        .arg(dir.join("k.ptx"))
        .output()
        .unwrap();
    assert!(built.status.success(), "ptxas: {}", String::from_utf8_lossy(&built.stderr));
    let out = Command::new(&cuobjdump).arg("-sass").arg(dir.join("k.cubin")).output().unwrap();
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn a_let_bound_product_is_rounded_before_the_add() {
    let text = ptx("let", LET_BOUND);
    let lines = code(&text);
    assert!(lines.iter().any(|l| l.starts_with("mul.rn.f32")), "{text}");
    assert!(!lines.iter().any(|l| l.starts_with("mul.f32") || l.starts_with("fma")), "{text}");
    match sass("let", &text) {
        Some(s) => {
            assert!(s.contains("FMUL") && s.contains("FADD"), "{s}");
            assert!(!s.contains("FFMA"), "ptxas contracted a product the PTX rounds:\n{s}");
        }
        None => eprintln!("SKIP SASS: no ptxas/cuobjdump, so the contraction was not checked"),
    }
}

/// The control: the syntactic form is still ONE `fma.rn` - fused by design.
#[test]
fn the_syntactic_multiply_add_is_still_one_fma() {
    let text = ptx("syntactic", SYNTACTIC);
    let lines = code(&text);
    assert!(lines.iter().any(|l| l.starts_with("fma.rn.f32")), "{text}");
    assert!(!lines.iter().any(|l| l.starts_with("mul.")), "{text}");
    match sass("syntactic", &text) {
        Some(s) => assert!(s.contains("FFMA"), "{s}"),
        None => eprintln!("SKIP SASS: no ptxas/cuobjdump, so the fusion was not checked"),
    }
}

/// On the device: twice rounded is exactly 0, fused is exactly 2^-24.
#[test]
fn the_device_computes_what_the_ptx_states() {
    let Some(ctx) = y::cuda_runtime::CudaContext::new() else {
        eprintln!("SKIP: no CUDA device, so the device result was not checked");
        return;
    };
    let a = 1.0f32 + f32::powi(2.0, -12);
    let c = -(1.0f32 + f32::powi(2.0, -11));
    // The oracle, on the host: the same inputs, rounded each way.
    assert_eq!(a * a + c, 0.0, "two roundings on the host");
    assert_eq!(a.mul_add(a, c), f32::powi(2.0, -24), "one rounding on the host");
    for (tag, src, want) in [("let", LET_BOUND, 0.0f32), ("syntactic", SYNTACTIC, f32::powi(2.0, -24))] {
        let text = ptx(&format!("dev_{tag}"), src);
        let module = ctx.load_ptx(&text, "k").expect("load");
        let bufs: Vec<_> = (0..4).map(|_| ctx.alloc(4).unwrap()).collect();
        for (buf, v) in bufs.iter().zip([a, a, c]) {
            ctx.memcpy_htod_at(buf, 0, &v.to_le_bytes()).unwrap();
        }
        let args: Vec<u64> = bufs.iter().map(|b| b.device_ptr()).collect();
        ctx.launch(&module, (1, 1, 1), (1, 1, 1), 0, &args).expect("launch");
        ctx.synchronize().unwrap();
        let mut out = [0u8; 4];
        ctx.memcpy_dtoh_at(&mut out, &bufs[3], 0).unwrap();
        assert_eq!(f32::from_le_bytes(out), want, "{tag}: the device rounded otherwise");
    }
}
