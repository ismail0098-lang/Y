//! The GPU witness generator must refuse what it cannot lower.
//!
//! `--emit-zk-ptx` wrote a `<name>.witness.ptx`, printed "GPU PTX Witness
//! Generator Kernel compiled successfully", and exited 0 for every program.
//! The README already flagged it — "nothing in the test suite runs it or checks
//! what it computes" — but the gap was wider than unverified. Of `WitnessOp`'s
//! seventeen variants the emitter lowered **five**, and the other twelve did
//! not fail:
//!
//!   * `Inv` / `Div` emitted `mov s_out, s_a` under a comment reading
//!     "256-bit Field Inversion / Division Hint" — the IDENTITY, so `1/x`
//!     computed `x`;
//!   * everything else fell to `_ => mov 0`, writing ZERO into the witness
//!     slot.
//!
//! Both assemble. `ptxas` cannot distinguish a wrong `mov` from a right one,
//! which is the standing limit of an assemble-only gate (gotcha #8) and the
//! reason this file asserts on refusals instead.
//!
//! WHICH ops fell into the zero arm is what makes it severe rather than
//! partial: `BitOfLc` is how every comparison, bitwise operator, shift,
//! integer division and range check gets its witness; `IsZeroLc` /
//! `InvOrZeroLc` are `==` and `!=`; `MulAddLc` is the most common statement a
//! circom program lowers to. And `MulLc` — which `build_witness_ir` assigns to
//! any single-term-output constraint, i.e. the binding of every linear
//! expression — means **`return a + b;` was zeroed too.**
//!
//! Run with:  cargo test --features zk --test zk_ptx_witness_refuses

#![cfg(feature = "zk")]

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Per-case directory: two tests sharing one path is the `.ptx` race this repo
/// has now hit in five files.
static SALT: AtomicUsize = AtomicUsize::new(0);

struct Run {
    ok: bool,
    text: String,
    wrote_ptx: bool,
}

fn run(src: &str) -> Run {
    let n = SALT.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("y_zkptx_{}_{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let profile = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".ysu_hw_profile");
    if profile.exists() {
        let _ = std::fs::copy(&profile, dir.join(".ysu_hw_profile"));
    }

    let path = dir.join("case.ysu");
    std::fs::write(&path, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-zk-ptx")
        .current_dir(&dir)
        .output()
        .expect("failed to run Y");

    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let wrote_ptx = dir.join("case.witness.ptx").exists();
    let ok = out.status.success();
    let _ = std::fs::remove_dir_all(&dir);
    Run { ok, text, wrote_ptx }
}

/// Every construct outside the lowered subset must be named, not zeroed.
#[test]
fn an_unlowerable_witness_op_is_refused_by_name() {
    let cases: [(&str, &str, &str); 6] = [
        (
            "a linear binding",
            "fn main(a: I32, b: I32) -> I32 { return a + b; }",
            "MulLc",
        ),
        (
            "equality",
            "fn main(a: I32, b: I32) -> I32 { if a == b { return 1; } return 0; }",
            "IsZeroLc",
        ),
        (
            "a comparison",
            "fn main(a: I32, b: I32) -> I32 { if a < b { return 1; } return 0; }",
            "BitOfLc",
        ),
        (
            "bitwise and",
            "fn main(a: I32, b: I32) -> I32 { return a & b; }",
            "BitOfLc",
        ),
        (
            "integer division",
            "fn main(a: I32, b: I32) -> I32 { return a / b; }",
            "BitOfLc",
        ),
        (
            "a shift",
            "fn main(a: I32) -> I32 { return a << 2; }",
            "BitOfLc",
        ),
    ];

    for (label, src, op) in cases {
        let r = run(src);
        assert!(
            !r.ok,
            "{}: exited 0 for a program the backend cannot lower\n{}",
            label, r.text
        );
        assert!(
            r.text.contains("[PTX witness generator]"),
            "{}: failed without naming the witness generator\n{}",
            label, r.text
        );
        assert!(
            r.text.contains(op),
            "{}: expected the refusal to name `{}`\n{}",
            label, op, r.text
        );
        // The artifact must not exist. A file on disk beside a red error is
        // still a file someone will pick up.
        assert!(
            !r.wrote_ptx,
            "{}: refused, and wrote case.witness.ptx anyway\n{}",
            label, r.text
        );
    }
}

/// The control: refusing everything is sound and useless.
///
/// Without this, deleting the five arms that do lower would pass the test
/// above — the same shape as `ordinary_loop_bodies_still_verify` guarding the
/// invariant checker, and as the identity-conversion case in
/// `quantization_pass_refuses`.
#[test]
fn the_lowered_subset_still_emits() {
    for (label, src) in [
        ("a bare input", "fn main(a: I32) -> I32 { return a; }"),
        ("a product", "fn main(a: I32, b: I32) -> I32 { return a * b; }"),
    ] {
        let r = run(src);
        assert!(r.ok, "{}: refused a program it can lower\n{}", label, r.text);
        assert!(
            r.text.contains("compiled successfully"),
            "{}: succeeded without saying so\n{}",
            label,
            r.text
        );
    }
}

/// ...and what it emits must assemble at the target it names.
#[test]
fn the_emitted_witness_kernel_assembles() {
    let Some(ptxas) = which_ptxas() else {
        eprintln!("skipping: no ptxas on PATH");
        return;
    };

    let dir = std::env::temp_dir().join(format!("y_zkptx_asm_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let profile = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".ysu_hw_profile");
    if profile.exists() {
        let _ = std::fs::copy(&profile, dir.join(".ysu_hw_profile"));
    }
    let path = dir.join("case.ysu");
    std::fs::write(&path, "fn main(a: I32, b: I32) -> I32 { return a * b; }").unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-zk-ptx")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stdout));

    let ptx_path = dir.join("case.witness.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).expect("no witness.ptx written");
    let target = ptx
        .lines()
        .find(|l| l.trim_start().starts_with(".target"))
        .and_then(|l| l.split_whitespace().nth(1))
        .expect("no .target line")
        .to_string();

    let asm = Command::new(&ptxas)
        .arg(format!("-arch={}", target))
        .arg(&ptx_path)
        .arg("-o")
        .arg(dir.join("out.cubin"))
        .output()
        .expect("run ptxas");
    let stderr = String::from_utf8_lossy(&asm.stderr).to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        asm.status.success(),
        "the witness kernel does not assemble at its own target {}:\n{}",
        target,
        stderr
    );
}

fn which_ptxas() -> Option<PathBuf> {
    for p in ["/opt/cuda/bin/ptxas", "/usr/local/cuda/bin/ptxas", "/usr/bin/ptxas"] {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join("ptxas"))
            .find(|c| c.exists())
    })
}
