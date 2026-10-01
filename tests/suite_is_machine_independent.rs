//! The suite's verdict must not depend on the machine it runs on, or on what
//! was built before it ran.
//!
//! Until 2026-10-01 the suite had only ever run on the developer's own machine
//! - an sm_89 card, a probed `.ysu_hw_profile` in the repo root, a CUDA
//! toolkit, LLVM 21+, and a release build left over from the last session.
//! The first run anywhere else (a fresh clone in a GPU-less container) failed
//! 25 tests, and five of the causes were properties of the SUITE rather than
//! of the compiler. This file holds the source-level gates for them; the
//! behaviour is fixed in the files they name.

use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.rs` file a test binary is built from: `tests/`, `tests/common/`,
/// and the `y-gpu` crate's tests.
fn test_sources() -> Vec<PathBuf> {
    let mut v = Vec::new();
    for dir in ["tests", "tests/common", "crates/y-gpu/tests"] {
        let Ok(rd) = std::fs::read_dir(repo().join(dir)) else { continue };
        for e in rd {
            let p = e.expect("entry").path();
            if p.extension().and_then(|x| x.to_str()) == Some("rs") {
                v.push(p);
            }
        }
    }
    v.sort();
    v
}

/// The code part of a line: everything before a `//` comment.
fn code(line: &str) -> &str {
    line.split("//").next().unwrap_or("")
}

fn rel(p: &Path) -> String {
    p.strip_prefix(repo()).unwrap_or(p).display().to_string()
}

/// **Eight test files ran `target/release/Y`, a binary `cargo test` never
/// builds.** So after a source edit, the documented `cargo test` checked those
/// eight against whatever release build happened to exist - or failed with
/// "build the compiler first" in a clean checkout. Demonstrated rather than
/// argued: with `chisel {}`'s original register-naming defect restored in
/// `resolve_chisel_registers`, `chisel_register_scope` passed 8/8 against the
/// release binary built before the edit, and a mutation table for the `sm_00`
/// fix found that file's column green in every row - including the over-fix
/// that four other suites caught. `struct_field_array_bounds` was worse still:
/// it PREFERRED a stale release binary over the fresh debug one.
///
/// `env!("CARGO_BIN_EXE_Y")` is the binary `cargo test` builds from the source
/// under test, under any profile and any `CARGO_TARGET_DIR`. A hardcoded
/// `target/debug/Y` is refused too: it is wrong under `cargo test --release`
/// and under a relocated target directory.
#[test]
fn no_test_runs_a_compiler_binary_cargo_test_does_not_build() {
    let me = file!().rsplit('/').next().unwrap_or("").to_string();
    let mut offenders = Vec::new();
    let mut scanned = 0;
    for p in test_sources() {
        if p.file_name().and_then(|n| n.to_str()) == Some(me.as_str()) {
            continue;
        }
        scanned += 1;
        let text = std::fs::read_to_string(&p).expect("read test source");
        for (i, line) in text.lines().enumerate() {
            let c = code(line);
            for bad in [
                "join(\"target/release/Y\")",
                "join(\"target/debug/Y\")",
                "Command::new(\"target/release/Y\")",
                "Command::new(\"./target/release/Y\")",
                "Command::new(\"target/debug/Y\")",
            ] {
                if c.contains(bad) {
                    offenders.push(format!("{}:{}: {}", rel(&p), i + 1, line.trim()));
                }
            }
        }
    }
    assert!(scanned > 100, "only {scanned} test sources scanned; the walk is not reading tests/");
    assert!(
        offenders.is_empty(),
        "these tests run a compiler binary `cargo test` does not build from the source \
         under test - use `env!(\"CARGO_BIN_EXE_Y\")`:\n  {}",
        offenders.join("\n  ")
    );
}
