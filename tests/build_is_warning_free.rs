//! The build's warnings are a free census of the defect class this programme is
//! built around, and they are only readable if there are none to start with.
//!
//! Both builds are warning-free. That state was reached by two changes, and
//! this file guards the mechanism of each:
//!
//! 1. **`src/main.rs` re-declared thirty modules with `mod`,** so every one of
//!    those files was compiled a SECOND time as a private module of the `Y`
//!    binary — two crates from one set of sources. In that second crate a `pub`
//!    item is only reachable if `main.rs` itself calls it, so the dead-code
//!    census read "main.rs does not use this", which is a far weaker claim than
//!    "nothing uses this". Measured: 69 warnings, of which **62 were the bin's**
//!    and 6 the lib's. The 62 were noise; the 6 were the real census.
//! 2. **Twenty-six modules carried a crate-level `#![allow(dead_code)]`,** which
//!    is what made the noise tolerable — and buried the signal with it. The
//!    class those attributes suppress is exactly the one this repository keeps
//!    finding bugs in: `run_all_optimization_passes`, `c_emitter.rs`, the
//!    `SmemLayout` surface, and `VnniExact::licenses` — a second, weaker copy of
//!    Phase 0's soundness predicate that read as maintained code because it had
//!    unit tests, and whose only accuser was a warning nobody looked at.
//!
//! ## What these gates can and cannot see
//!
//! `the_build_reports_no_warnings` shells out, because a `cargo build` warning
//! is not observable from inside a test binary — the test is compiled by a
//! different invocation than the one whose diagnostics we care about. It uses a
//! separate `CARGO_TARGET_DIR` so it cannot deadlock against the `cargo test`
//! that is running it, and that directory persists, so the usual cost is one
//! `Finished` line.
//!
//! It sees: every warning `rustc` emits for the root package's lib and four
//! binaries, in both feature sets, plus `crates/y-gpu` — which the documented
//! `cargo test` does NOT build, and which is where a stale check once sat red
//! for a week.
//!
//! It does NOT see: warnings in `tests/*.rs` (those targets are not built here),
//! anything suppressed by an item-level `#[allow]`, or a lint that is off by
//! default. The first is deliberate — a test harness is allowed scaffolding —
//! and the second is why `no_module_suppresses_the_dead_code_census` exists
//! beside it. Item-level `#[allow(dead_code)]` stays legal: category 3 of the
//! taxonomy (a genuinely configuration-dependent path) needs it, and `main`'s
//! `counts()` is one. A whole MODULE opting out is what is banned.

use std::path::Path;
use std::process::Command;

/// A build configuration, and the label a failure names it by.
/// A build configuration: the label a failure names it by, its own target
/// subdirectory, and the arguments.
///
/// The subdirectories are SEPARATE on purpose. Two feature sets of one package
/// sharing a target dir invalidate each other's units, so a single directory
/// turns every run of this gate into two full rebuilds - 12.3s against 0.6s
/// measured. A gate that is expensive gets `#[ignore]`d and then never runs,
/// which this repository has already paid for once.
const CONFIGS: &[(&str, &str, &[&str])] = &[
    ("cargo build --release", "default", &["build", "--release"]),
    (
        "cargo build --release --features zk",
        "zk",
        &["build", "--release", "--features", "zk"],
    ),
    (
        "cargo build --release -p y-gpu",
        "ygpu",
        &["build", "--release", "-p", "y-gpu"],
    ),
];

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn the_build_reports_no_warnings() {
    // Siblings of the main target dir: different paths, so different locks,
    // and persistent, so these are incremental rather than cold builds.
    let root = repo_root().join("target").join("warning-gate");

    let mut ran = 0usize;
    let mut complaints = Vec::new();

    for (label, slug, args) in CONFIGS {
        let out = Command::new(env!("CARGO"))
            .args(*args)
            .current_dir(repo_root())
            .env("CARGO_TARGET_DIR", root.join(slug))
            // Colour codes would defeat the `starts_with` below.
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap_or_else(|e| panic!("could not run `{label}`: {e}"));

        let err = String::from_utf8_lossy(&out.stderr);

        // NON-VACUITY. A command that failed to run, or that cargo refused,
        // emits no `warning:` line either - which is a perfect pass for a gate
        // that only counts bad things. Require the evidence that a build
        // actually happened before believing its silence.
        assert!(
            out.status.success(),
            "`{label}` did not succeed; a gate counting warnings cannot \
             distinguish a clean build from one that never ran.\n{err}"
        );
        assert!(
            err.lines().any(|l| l.trim_start().starts_with("Finished")),
            "`{label}` produced no `Finished` line, so cargo did not complete a \
             build and the absence of warnings means nothing.\n{err}"
        );
        ran += 1;

        for line in err.lines() {
            let l = line.trim_start();
            if l.starts_with("warning:") {
                complaints.push(format!("{label}: {l}"));
            }
        }
    }

    assert_eq!(ran, CONFIGS.len(), "not every configuration was built");
    assert!(
        complaints.is_empty(),
        "the build is no longer warning-free ({} warning line(s)). A `never used` \
         warning is a free census of the class this repository keeps finding bugs \
         in - read it and sort it (leftover / confirmation-of-a-fix / \
         configuration-dependent / a second implementation of an existing rule) \
         rather than reaching for `#[allow]`:\n  {}",
        complaints.len(),
        complaints.join("\n  ")
    );
}

#[test]
fn no_module_suppresses_the_dead_code_census() {
    let src = repo_root().join("src");
    let mut scanned = 0usize;
    let mut offenders = Vec::new();

    let mut stack = vec![src.clone()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).expect("src/ is readable") {
            let p = e.expect("readable entry").path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                scanned += 1;
                let text = std::fs::read_to_string(&p).expect("source is utf-8");
                // Crate/module-level (`#!`) only. Item-level (`#[`) is allowed:
                // it names one item, and category 3 of the taxonomy needs it.
                if text
                    .lines()
                    .any(|l| l.trim().starts_with("#![") && l.contains("allow(dead_code)"))
                {
                    offenders.push(
                        p.strip_prefix(repo_root())
                            .unwrap_or(&p)
                            .display()
                            .to_string(),
                    );
                }
            }
        }
    }

    // Non-vacuity: a sweep that walked nothing reports "no offenders" perfectly.
    assert!(
        scanned > 20,
        "only {scanned} source files were scanned; the sweep is not looking where it thinks"
    );
    offenders.sort();
    assert!(
        offenders.is_empty(),
        "a crate-level `#![allow(dead_code)]` opts a whole module out of the \
         dead-code census, which is how 69 warnings - including a second, weaker \
         copy of the exact-GEMM licence predicate - stayed invisible. Suppress \
         one ITEM with `#[allow(dead_code)]` and write the configuration that \
         makes it unreachable beside it. Offending files:\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn the_binary_uses_the_library_rather_than_recompiling_it() {
    let lib = std::fs::read_to_string(repo_root().join("src").join("lib.rs")).unwrap();
    let main = std::fs::read_to_string(repo_root().join("src").join("main.rs")).unwrap();

    let lib_mods: Vec<String> = lib
        .lines()
        .filter_map(|l| l.trim().strip_prefix("pub mod "))
        .filter_map(|r| r.strip_suffix(';'))
        .map(str::to_string)
        .collect();

    // Non-vacuity: an empty module list makes the check below unfailable.
    assert!(
        lib_mods.len() > 20,
        "src/lib.rs declares only {} modules; the comparison below would be \
         nearly vacuous",
        lib_mods.len()
    );

    let redeclared: Vec<&String> = lib_mods
        .iter()
        .filter(|m| {
            main.lines()
                .any(|l| l.trim() == format!("mod {m};") || l.trim() == format!("pub mod {m};"))
        })
        .collect();

    assert!(
        redeclared.is_empty(),
        "src/main.rs re-declares {} module(s) that src/lib.rs already owns, so \
         each is compiled a second time as a private module of the `Y` binary. \
         That doubles the build AND makes the dead-code census meaningless - in \
         a private module a `pub` item is dead unless main.rs itself calls it, \
         so the warnings become \"main.rs does not use this\" (62 of 69, when \
         this was measured) and drown the ones that mean \"nothing uses this\". \
         Use `use y::{{..}}` instead. Re-declared: {:?}",
        redeclared.len(),
        redeclared
    );
}
