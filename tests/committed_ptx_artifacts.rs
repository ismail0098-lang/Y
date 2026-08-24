//! The committed `.ptx` files are artifacts, and an artifact drifts from the
//! compiler that produced it in silence.
//!
//! Two gates, both hermetic (no GPU, no `ptxas`, no recompilation), because
//! both failures are invisible to every gate this repo already had:
//!
//! * An EMPTY module assembles perfectly, so no `ptxas` sweep can see it.
//!   `tests/backends_refuse_empty_artifacts.rs` stops the backend PRODUCING
//!   one; this stops one being COMMITTED. 24 were, which is exactly the
//!   "24 of 85" the empty-artifact sweep measured - they were that bug's
//!   output, checked in.
//!
//! * An OVER-STATED `.version` assembles perfectly too, on the machine that
//!   wrote it. It fails at load time on a merely-older driver with
//!   `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, i.e. on a machine nobody tested.
//!   55 of the 59 committed artifacts at `.version 8.4` contained no FP8
//!   instruction at all - they demanded CUDA 12.4 to run two `mov`s.
//!
//! CLAUDE.md already recorded the second one happening ("two committed
//! artifacts were also still at 8.4 - the previous regeneration pass missed
//! them, and no test could tell"). The regeneration pass missed 32, not two.
//! A regeneration is a one-time act and this is the gate that makes it stick.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.ptx` tracked in the repo, found on disk rather than through git so
/// the test needs no subprocess.
fn committed_ptx() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![repo_root()];
    while let Some(dir) = stack.pop() {
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            let p = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if p.is_dir() {
                // `target/` holds build output and `.git/` holds objects;
                // neither is a committed source artifact.
                if name == "target" || name == ".git" || name == "node_modules" {
                    continue;
                }
                stack.push(p);
            } else if name.ends_with(".ptx") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// A line that is not a directive, a comment, a brace or a parameter - i.e.
/// something the GPU would actually execute.
///
/// This is the same "assert the EFFECT, not the vocabulary" measure that
/// `tests/quantization_pass_refuses.rs` uses: a module full of comments and
/// `.reg` declarations assembles and does nothing.
fn instruction_lines(src: &str) -> usize {
    src.lines()
        .map(str::trim)
        .filter(|l| {
            !l.is_empty()
                && !l.starts_with("//")
                && !l.starts_with('.')
                && !l.starts_with('{')
                && !l.starts_with('}')
                && !l.starts_with(')')
        })
        .count()
}

fn header_value(src: &str, directive: &str) -> Option<String> {
    src.lines()
        .map(str::trim)
        .find(|l| l.starts_with(directive))
        .map(|l| l[directive.len()..].trim().to_string())
}

fn rel(p: &Path) -> String {
    p.strip_prefix(repo_root())
        .unwrap_or(p)
        .to_string_lossy()
        .into_owned()
}

#[test]
fn no_committed_ptx_module_is_empty() {
    let files = committed_ptx();
    let mut empty: Vec<String> = Vec::new();
    for p in &files {
        let src = std::fs::read_to_string(p).unwrap();
        if instruction_lines(&src) == 0 {
            empty.push(rel(p));
        }
    }
    assert!(
        empty.is_empty(),
        "these committed .ptx modules contain no instructions at all - they are \
         a `.version`/`.target` header and nothing else, which is what the PTX \
         backend used to emit for a source containing no `kernel`. `ptxas` \
         accepts them, so nothing else in this repo can see them. Delete them, \
         or regenerate from a source the backend accepts:\n  {}",
        empty.join("\n  ")
    );
}

/// The control for the test above: if the sweep found no files, "none of them
/// is empty" is vacuously true. This is the `feedback-null-metrics-pass-dead-
/// components` shape - a count of bad things being zero is also what a broken
/// harness reports.
#[test]
fn the_artifact_sweep_actually_finds_artifacts() {
    let files = committed_ptx();
    assert!(
        files.len() >= 40,
        "expected the repo to carry a substantial body of committed .ptx; found {}. \
         If they have genuinely been removed, lower this floor deliberately - do not \
         let the two gates in this file start passing vacuously.",
        files.len()
    );
    // And they must be real modules, not a directory of stubs.
    let total: usize = files
        .iter()
        .map(|p| instruction_lines(&std::fs::read_to_string(p).unwrap()))
        .sum();
    assert!(
        total > 1000,
        "the committed .ptx corpus holds only {total} instruction lines in total"
    );
}

#[test]
fn every_committed_ptx_declares_the_version_floor_for_its_target() {
    let mut wrong: Vec<String> = Vec::new();
    for p in committed_ptx() {
        let src = std::fs::read_to_string(&p).unwrap();
        let (target, version) = match (header_value(&src, ".target"), header_value(&src, ".version"))
        {
            (Some(t), Some(v)) => (t, v),
            // A file with no header is not a module this gate can judge; the
            // emptiness gate above is what covers a degenerate file.
            _ => continue,
        };

        // The compiler's own table is the reference. Re-deriving it here would
        // make a third copy, which CLAUDE.md records as the bug rather than the
        // fix (`the_coprocessor_backend_declares_the_same_floor_as_the_ptx_backend`
        // asserts the producers AGREE for the same reason).
        let floor = y::ptx_emitter::ptx_version_for_sm(&target)
            .trim_start_matches(".version")
            .trim()
            .to_string();

        // The one legitimate reason to declare MORE than the floor: FP8
        // `mma.sync` really does need ISA 8.4, even on the arch that has the
        // hardware. That is a per-INSTRUCTION requirement, so it is detected
        // per-instruction rather than waved through by architecture.
        let needs_fp8 = src.contains("e4m3") || src.contains("e5m2");
        let expected = if needs_fp8 && floor.as_str() < "8.4" {
            "8.4".to_string()
        } else {
            floor
        };

        if version != expected {
            wrong.push(format!(
                "{}: .target {target} declares .version {version}, expected {expected}{}",
                rel(&p),
                if needs_fp8 { " (FP8 present)" } else { "" }
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "a `.version` above the floor its instructions require is a DRIVER \
         requirement nothing in the artifact needs. It assembles perfectly here \
         and fails to load on an older driver with \
         CUDA_ERROR_UNSUPPORTED_PTX_VERSION. Regenerate these:\n  {}",
        wrong.join("\n  ")
    );
}
