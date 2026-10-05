//! Keep audit evidence bound to the inputs that the workflow actually tested.

use std::path::Path;
use std::process::Command;

#[test]
fn regression_audit_rejects_changing_inputs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = std::env::var_os("Y_TVAL_PYTHON").unwrap_or_else(|| "python3".into());
    let output = Command::new(&python)
        .arg(root.join("tests/verification_workflow_regressions.py"))
        .current_dir(root)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .unwrap_or_else(|error| panic!("cannot run verification workflow checks with {python:?}: {error}"));
    assert!(
        output.status.success(),
        "verification workflow regressions failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
