//! Exercise the Python validator's counterexamples through the Cargo test gate.
//! Honor Y_TVAL_PYTHON; otherwise prefer a repository virtual environment
//! with Z3, then system Python. Parser tests also run when Z3 is unavailable
//! locally; `Y_VERIFICATION_STRICT=1` requires the Python solver.

use std::path::Path;
use std::process::Command;

#[path = "common/validator.rs"]
mod validator;

#[test]
fn translation_validator_rejects_unsound_proofs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = validator::validator_python(root);
    let output = Command::new(&python)
        .arg(root.join("tests/translation_validator_soundness.py"))
        .current_dir(root)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .unwrap_or_else(|error| {
            panic!("cannot run validator regressions with {python:?}: {error}")
        });
    assert!(
        output.status.success(),
        "translation-validator regressions failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
}
