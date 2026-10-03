//! Compile without touching the repository, against a hardware profile the
//! test chose.
//!
//! Two things a test that runs the compiler must not do, both found by the
//! first runs of this suite outside the developer's machine (2026-10-01):
//!
//! * **Compile a committed fixture in place.** `--emit-ptx`, `--emit-llvm`,
//!   `--emit-coprocessor` and `--target=r1cs` write next to their SOURCE, so
//!   compiling `tests/k.ysu` rewrites the committed `tests/k.ptx` - with
//!   whatever target and measured latencies this machine's profile holds -
//!   while `committed_ptx_artifacts.rs` reads it. Every run of the default
//!   suite rewrote four or five committed artifacts, under every profile.
//!   Copy the fixture into a scratch directory with [`copy_fixture`].
//! * **Let a verdict depend on this machine's `.ysu_hw_profile`.** The
//!   compiler reads the profile in its WORKING directory (and probes and
//!   writes one there if there is none), so `current_dir(repo())` compiles for
//!   whatever card this machine has. Thirteen tests assembled that output at
//!   a hardcoded `-arch=sm_89` and failed on a machine whose profile names
//!   sm_90. Every test pins the profile with [`pin`] - a test that launches
//!   the kernel too, since a module declaring the floor loads on any card.
//!
//! And one thing moving a compile out of the repository must not do:
//!
//! * **Hide a solver the repository can see.** The compiler also looks for
//!   z3 at RELATIVE paths (`venv/bin/z3`, ...), resolved against its working
//!   directory. [`pin`] mirrors the ones the repository has - see
//!   [`mirror_solver`].
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The compute capability a pinned compile targets: the floor, `PTX_FLOOR`
/// (sm_80). Every architecture the compiler supports assembles a module that
/// declares it and every supported card loads one, so the same pinned compile
/// serves an assembler check at a fixed arch AND a launch on whatever card is
/// present - "guess down", the rule the emitter itself follows.
pub const SM_PINNED: &str = "8.0";

/// For a test that needs FP8 or `@require(sm >= 89)`: Ada, the lowest
/// architecture with FP8, and the target the committed artifacts declare.
pub const SM_FP8: &str = "8.9";

/// The GPU name a pinned profile records. Every per-GPU cache in the profile
/// (`DRIFT_ACC_*`, `AUTOTUNE_*`) is keyed by it.
pub const GPU_NAME: &str = "PinnedByTest";

/// Recorded `@ZeroDrift` accumulate costs, in ps per accumulate, keyed by
/// [`GPU_NAME`].
///
/// **Without them a pinned compile is NOT the same on every machine.** Every
/// `--emit-ptx` and `--emit-llvm` compile calls `load_or_measure_drift_costs`,
/// and a profile holding no costs for its GPU name makes it time a probe
/// kernel on the device - seconds of GPU time - and append the result. So on a
/// machine with a GPU each pinned compile measured, and a representation choice
/// followed the measurement: `Q32.32` and `I64` emit byte-identical probe PTX,
/// so for an accumulator both can hold, which one won was timing noise. Where
/// there is no GPU nothing is measured and the selector takes the narrowest
/// sufficient representation instead.
///
/// The figures are the ones CLAUDE.md records for an RTX 4070 Ti SUPER. Their
/// ORDER is the no-measurement fallback's (narrowest first, a float last), so a
/// pinned compile selects what a GPU-less one would.
pub const DRIFT_COSTS: &[(&str, &str)] =
    &[("Q16.16", "1790.000"), ("Q32.32", "1922.000"), ("I64", "2106.000"), ("F64", "17726.000")];

/// The text of a pinned profile: the target, the GPU name, and costs for every
/// per-GPU measurement the compiler would otherwise take on this machine. Every
/// other field falls back to a fixed default. `SM_VERSION` is the field that
/// decides the PTX target.
pub fn profile_text(sm: &str) -> String {
    let mut t = format!("SM_VERSION={sm}\nCOMPUTE_CAPABILITY={sm}\nGPU_NAME={GPU_NAME}\nSM_COUNT=66\n");
    for (repr, ps) in DRIFT_COSTS {
        t.push_str(&format!("DRIFT_ACC_{repr}_{GPU_NAME}={ps}\n"));
    }
    t
}

/// Write a pinned `.ysu_hw_profile` into `dir`, and give `dir` the solver
/// candidates the repository has ([`mirror_solver`]). Run the compiler with
/// `current_dir(dir)` so that it is the profile read.
pub fn pin(dir: &Path, sm: &str) {
    std::fs::write(dir.join(".ysu_hw_profile"), profile_text(sm)).expect("pin the profile");
    mirror_solver(&repo(), dir);
}

/// Make the SMT solver the compiler finds from `dir` the one it finds from
/// `root`: every z3 candidate that is a RELATIVE path and exists under `root`
/// is linked at the same relative path under `dir`.
///
/// `type_checker::z3_candidates` tries `Y_Z3_PATH`, `z3` on `PATH`, then
/// `venv/bin/z3`, `.venv/bin/z3` and `z3/build/z3` - and those three are
/// resolved against the compiler's WORKING directory. So moving a compile from
/// the repository into a scratch directory hid a solver installed in a
/// repo-local venv, and every program carrying an `@invariant` was refused
/// with "the SMT solver could not be run". Measured with z3 only at
/// `<repo>/venv/bin/z3`: 21 tests in 7 files failed that pass when their
/// compile runs in the repository. Whether z3 is installed decides verdicts
/// by design - an invariant that cannot be checked fails the build - but
/// WHERE it is installed must not.
///
/// The candidate list is the compiler's own, not a copy, so a candidate added
/// there is mirrored here. Only candidates that exist under `root` are linked,
/// so `dir` resolves exactly what `root` resolves, in the same order. A bare
/// name is a `PATH` lookup and an absolute path resolves the same from
/// anywhere; only a relative path with a directory part depends on the
/// working directory.
pub fn mirror_solver(root: &Path, dir: &Path) {
    for cand in y::type_checker::z3_candidates() {
        let rel = Path::new(&cand);
        if rel.is_absolute() || rel.components().count() < 2 {
            continue;
        }
        let (src, dst) = (root.join(rel), dir.join(rel));
        // `./venv/bin/z3` and `venv/bin/z3` name one file: link it once.
        if !src.exists() || dst.symlink_metadata().is_ok() {
            continue;
        }
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).expect("solver mirror dir");
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(&src, &dst).expect("link the solver");
        #[cfg(not(unix))]
        std::fs::copy(&src, &dst).map(|_| ()).expect("copy the solver");
    }
}

/// A fresh, empty directory. The tag is for whoever finds one left behind;
/// the process id and a counter are what make it unique, because two callers
/// passing the same tag is the shared-temp-dir race this repository has hit
/// eight times.
pub fn scratch(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "y_pinned_{}_{}_{}",
        tag,
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// A fresh scratch directory holding a pinned profile.
pub fn pinned_scratch(tag: &str, sm: &str) -> PathBuf {
    let dir = scratch(tag);
    pin(&dir, sm);
    dir
}

/// Copy the repository file `rel` (e.g. `tests/k.ysu`) into `dir` and return
/// the copy's path. The compiler writes its output next to the copy.
pub fn copy_fixture(dir: &Path, rel: &str) -> PathBuf {
    let from = repo().join(rel);
    let to = dir.join(from.file_name().expect("fixture file name"));
    std::fs::copy(&from, &to).unwrap_or_else(|e| panic!("copy {rel}: {e}"));
    to
}
