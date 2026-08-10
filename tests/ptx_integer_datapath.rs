//! The PTX backend has no integer datapath, and used to pretend otherwise.
//!
//! A kernel declared `GlobalMemory<U32>` with `let s: U32 = a + b;` compiled
//! clean, assembled clean under `ptxas -arch=sm_89`, printed
//! "PTX Assembly generated successfully!", exited 0 — and emitted
//! `ld.global.f32`, `add.f32`, `st.global.f32`. `U64`, `I32` and `I64` behaved
//! identically. Every value above 2^24 is silently rounded; a 254-bit field
//! limb is destroyed outright.
//!
//! That is a wrong answer, not a missing feature, and it is the same shape as
//! the `tma_load` / `wgmma_async` intrinsics deleted in CLAUDE.md gotcha #7 —
//! except worse, because `ptxas` *accepts* this one. Nothing downstream can
//! catch it: the PTX is valid, the kernel launches, and the numbers are wrong.
//!
//! Until typed integer loads and arithmetic exist, the backend refuses. **This
//! is the blocker for GPU field arithmetic** (BN254 Montgomery multiply for NTT
//! and MSM): that kernel is 8×u32 limbs with `mul.wide.u32` accumulation, and
//! not one of those operations can be expressed today.
//!
//! Driven through the real binary because the behaviour is a build failure.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("Y")
}

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn compile(src: &str, name: &str) -> (bool, String, PathBuf) {
    let dir = std::env::temp_dir().join("y_ptx_int_datapath");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).unwrap();
    let ptx = path.with_extension("ptx");
    let _ = std::fs::remove_file(&ptx);
    let out = Command::new(bin())
        .arg(&path)
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .expect("failed to run the Y binary");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text, ptx)
}

const INT_KERNEL: &str = r#"
kernel int_probe(A: GlobalMemory<TY>, B: GlobalMemory<TY>, C: GlobalMemory<TY>, N: I32) {
    let i: I32 = block_idx_x() * 256 + thread_idx_x();
    let a: TY = block_ptr2d_load(A, 0, i, N, 1, N);
    let b: TY = block_ptr2d_load(B, 0, i, N, 1, N);
    let s: TY = a + b;
    block_ptr2d_store(C, 0, i, N, 1, N, s);
}
fn main() {}
"#;

#[test]
fn integer_element_types_are_refused_not_silently_floated() {
    for ty in ["U32", "U64", "I64", "U8", "I16"] {
        let (ok, out, ptx) = compile(&INT_KERNEL.replace("TY", ty), &format!("int_{}", ty));
        assert!(!ok, "GlobalMemory<{}> compiled successfully; it must be refused", ty);
        assert!(
            out.contains("no integer datapath"),
            "GlobalMemory<{}> failed without naming the reason:\n{}",
            ty,
            out
        );
        assert!(
            !ptx.exists(),
            "GlobalMemory<{}> was refused but a .ptx was written anyway",
            ty
        );
    }
}

/// The control, and it matters as much as the test above: refusing everything
/// would satisfy that assertion and break the compiler.
#[test]
fn float_element_types_still_compile() {
    let (ok, out, ptx) = compile(&INT_KERNEL.replace("TY", "F32"), "flt_probe");
    assert!(ok, "GlobalMemory<F32> must still compile:\n{}", out);
    assert!(ptx.exists(), "no .ptx written for the F32 kernel");
    let text = std::fs::read_to_string(&ptx).unwrap();
    assert!(
        text.contains("ld.global.f32") && text.contains("add.f32"),
        "F32 kernel no longer emits the float datapath"
    );
}

/// If the integer datapath is ever implemented, this test is the thing that
/// tells you to delete the refusal — and it pins what "implemented" has to
/// mean, so a partial job cannot quietly claim it.
#[test]
#[ignore]
fn what_the_integer_datapath_must_emit() {
    let (ok, _, ptx) = compile(&INT_KERNEL.replace("TY", "U32"), "int_future");
    assert!(ok, "still refused — remove the #[ignore] once it lowers");
    let text = std::fs::read_to_string(&ptx).unwrap();
    for want in ["ld.global.u32", "add.u32", "st.global.u32"] {
        assert!(text.contains(want), "integer kernel does not emit `{}`", want);
    }
    assert!(
        !text.contains("add.f32"),
        "integer kernel is still routing arithmetic through the float datapath"
    );
}
