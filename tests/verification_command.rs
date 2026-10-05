//! Exercise the verification command's evidence and failure handling.
use std::path::Path;
use std::process::Command;

#[test]
fn verification_command_rejects_incomplete_or_unbound_evidence() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = std::env::var_os("Y_TVAL_PYTHON").unwrap_or_else(|| "python3".into());
    for suite in [
        "verification_command_tests.py",
        "verification_unittest_regressions.py",
    ] {
        let output = Command::new(&python)
            .arg(root.join("tests").join(suite))
            .current_dir(root)
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .output()
            .expect("Python is required to verify the workflow command");
        assert!(
            output.status.success(),
            "{suite} failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
}
