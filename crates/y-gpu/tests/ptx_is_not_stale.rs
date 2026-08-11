//! The embedded PTX must match the `.ysu` it was compiled from.
//!
//! Embedding the compiler's output is what frees this crate from a runtime
//! dependency on the Y binary — but it introduces the one failure mode that
//! trade always has: the `.ysu` changes, the `.ptx` does not, and the library
//! silently keeps running last month's kernel. Nothing else in the test suite
//! can see that, because every other test exercises the embedded copy.
//!
//! Skips when the `Y` binary is not built, like the `ptxas` gates elsewhere.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    // crates/y-gpu -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

#[test]
fn embedded_ptx_matches_a_fresh_compile() {
    let root = repo_root();
    let bin = root.join("target/release/Y");
    if !bin.exists() {
        eprintln!("SKIP: target/release/Y not built — cannot check PTX freshness.");
        return;
    }

    let mut checked = 0;
    for (entry, embedded) in y_gpu::kernels::ALL {
        let ysu = root.join(format!("tests/{entry}.ysu"));
        assert!(ysu.exists(), "{entry}.ysu is missing; the embedded PTX has no source");

        let out = Command::new(&bin)
            .arg(&ysu)
            .arg("--emit-ptx")
            .current_dir(&root)
            .output()
            .expect("failed to run the Y compiler");
        assert!(
            out.status.success(),
            "{entry}.ysu no longer compiles:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let fresh = std::fs::read_to_string(root.join(format!("tests/{entry}.ptx")))
            .expect("no .ptx written");
        assert_eq!(
            fresh.trim(),
            embedded.trim(),
            "crates/y-gpu/ptx/{entry}.ptx is STALE.\n\
             Regenerate with:  ./target/release/Y tests/{entry}.ysu --emit-ptx \\\n\
             && cp tests/{entry}.ptx crates/y-gpu/ptx/{entry}.ptx"
        );
        checked += 1;
    }
    assert!(checked >= 5, "only {checked} kernels checked; the list shrank unexpectedly");
}

/// Every embedded kernel must actually assemble for the target arch. This is
/// the `ptxas` gate the rest of the repo uses, applied to what ships.
#[test]
fn embedded_ptx_assembles() {
    if Command::new("ptxas").arg("--version").output().is_err() {
        eprintln!("SKIP: ptxas not found.");
        return;
    }
    for (entry, ptx) in y_gpu::kernels::ALL {
        let dir = std::env::temp_dir().join(format!("y_gpu_ptx_{entry}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("k.ptx");
        std::fs::write(&path, ptx).unwrap();
        let out = Command::new("ptxas")
            .arg("-arch=sm_89")
            .arg("-o")
            .arg(dir.join("k.cubin"))
            .arg(&path)
            .output()
            .expect("ptxas");
        assert!(
            out.status.success(),
            "embedded {entry} does not assemble:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
