//! Run the dependency-free artifact regressions through the ordinary Cargo
//! test gate. Real ptxas/nvdisasm/Z3 validation additionally runs when present.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/validator.rs"]
mod validator;

#[path = "common/verification.rs"]
mod verification;

#[path = "common/pinned.rs"]
mod pinned;

struct TemporaryArtifacts(PathBuf);

#[test]
fn exact_pv_checked_launch_enforces_the_tensor_and_hardware_contract() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new(validator::validator_python(root))
        .arg(root.join("tests/exact_pv_launch_contract.py"))
        .current_dir(root)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("run checked exact_pv launch contract regressions");
    assert!(output.status.success(), "{}{}",
        String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
}

impl Drop for TemporaryArtifacts {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn freshly_emitted_ptx_matches_the_complete_reviewed_proof_subject() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // A PINNED sm_89 profile, the target the reviewed subject declares. With
    // the repository as working directory the emitted `.target` was whatever
    // this machine's profile said - an unprobed worktree's says `sm_00` - and
    // the subject check failed for a reason unrelated to the subject.
    let temporary = TemporaryArtifacts(pinned::pinned_scratch("pv_subject", pinned::SM_FP8));
    let source = pinned::copy_fixture(&temporary.0, "tests/exact_pv.ysu");
    let emitted = Command::new(env!("CARGO_BIN_EXE_Y"))
        .args([source.as_os_str(), std::ffi::OsStr::new("--emit-ptx")])
        .env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS")
        .current_dir(&temporary.0)
        .output()
        .unwrap();
    assert!(
        emitted.status.success(),
        "source emission failed: {}{}",
        String::from_utf8_lossy(&emitted.stdout),
        String::from_utf8_lossy(&emitted.stderr)
    );
    let checked = Command::new(validator::validator_python(root))
        .args(["-c", "import sys; sys.path.insert(0,sys.argv[1]); from exact_pv_subject import require_proved_subject; from pathlib import Path; require_proved_subject(Path(sys.argv[2]).read_bytes())"])
        .arg(root.join("tools/ptxas_tval")).arg(source.with_extension("ptx"))
        .env("PYTHONDONTWRITEBYTECODE", "1").output().unwrap();
    assert!(
        checked.status.success(),
        "fresh PTX left the reviewed proof subject: {}{}",
        String::from_utf8_lossy(&checked.stdout),
        String::from_utf8_lossy(&checked.stderr)
    );
}

#[test]
fn verified_exact_pv_loads_only_the_validated_cubin() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = validator::validator_python(root);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temporary = TemporaryArtifacts(
        std::env::temp_dir().join(format!("y-artifact-interop-{}-{nonce}", std::process::id())),
    );
    std::fs::create_dir(&temporary.0).expect("create artifact interoperability directory");
    let bundle = temporary.0.join("validated");
    let output = Command::new(&python)
        .arg(root.join("tests/verified_exact_pv_artifact.py"))
        .current_dir(root)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("Y_TVAL_TEST_BUNDLE", &bundle)
        .output()
        .unwrap_or_else(|error| panic!("cannot run artifact regressions with {python:?}: {error}"));
    assert!(
        output.status.success(),
        "exact_pv artifact regressions failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    // Keep optional-toolchain skip reasons visible with `cargo test -- --nocapture`.
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    if bundle.exists() {
        let ptx = std::fs::read(root.join("tests/exact_pv.ptx")).unwrap();
        let expected_ptx = format!("{:x}", Sha256::digest(ptx));
        let artifact = y::verified_exact_pv::ValidatedExactPv::open(&bundle, Some(&expected_ptx))
            .expect("Rust must accept the bundle produced by real Python translation validation");
        let cubin = std::fs::read(bundle.join("exact_pv.cubin")).unwrap();
        assert_eq!(
            artifact.cubin_sha256(),
            format!("{:x}", Sha256::digest(cubin)),
            "Rust and Python must identify the same validated cubin"
        );
    } else {
        verification::prerequisite_available(
            false,
            "a real ptxas/nvdisasm/Z3 validated exact_pv bundle for Rust/Python interoperability",
        );
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("real translation-validation integration requires"),
            "a real build must be exported unless its integration tests explicitly skipped"
        );
    }
}
