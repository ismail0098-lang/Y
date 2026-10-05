//! Choose the same Python interpreter for every translation-validation gate.

use std::ffi::OsString;
use std::path::Path;
use std::process::Command;

#[path = "verification.rs"]
mod verification;

fn enforce_python_solver(python: &OsString) {
    if !verification::strict() {
        return;
    }
    let available = Command::new(python)
        .args(["-c", "import z3"])
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .is_ok_and(|output| output.status.success());
    verification::prerequisite_available(
        available,
        &format!("Python interpreter {python:?} with z3-solver for translation validation"),
    );
}

pub fn validator_python(root: &Path) -> OsString {
    if let Some(python) = std::env::var_os("Y_TVAL_PYTHON") {
        enforce_python_solver(&python);
        return python;
    }

    let mut candidates: Vec<OsString> = ["venv/bin/python", ".venv/bin/python"]
        .into_iter()
        .map(|path| root.join(path))
        .filter(|path| path.is_file())
        .map(|path| path.into_os_string())
        .collect();
    candidates.push("python3".into());

    let mut fallback = None;
    for python in candidates {
        let Ok(output) = Command::new(&python)
            .args([
                "-c",
                "import sys\ntry:\n import z3\nexcept ImportError:\n sys.exit(3)",
            ])
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .output()
        else {
            continue;
        };
        if output.status.success() {
            return python;
        }
        if output.status.code() == Some(3) && fallback.is_none() {
            fallback = Some(python);
        }
    }
    // Keep dependency-free regressions active when Z3 is unavailable; the
    // Python suites explicitly report why their SMT integration is skipped.
    let python = fallback.unwrap_or_else(|| "python3".into());
    enforce_python_solver(&python);
    python
}
