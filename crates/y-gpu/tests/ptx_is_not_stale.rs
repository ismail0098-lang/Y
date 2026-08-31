//! What the library EMBEDS, checked from inside the library.
//!
//! Embedding the compiler's output with `include_str!` is what frees this
//! crate from a runtime dependency on the Y binary — but it introduces the one
//! failure mode that trade always has: the `.ysu` changes, the `.ptx` does
//! not, and the library silently keeps running last month's kernel.
//!
//! **This file used to gate that, and it had been red for a week without
//! anyone seeing it.** `cargo test` at the workspace root builds the root
//! package only, so nothing in the documented build commands runs
//! `cargo test -p y-gpu`. Four of the five embedded kernels were stale by the
//! whole carry-flag intrinsic series.
//!
//! **And the test could not have passed on this machine anyway.** It compared
//! the embedded copy against a compile on the BUILD MACHINE, while the whole
//! point of these artifacts is portability: they are committed at
//! `.target sm_80` and a fresh local compile probes whatever card is present.
//! Those are contradictory requirements and freshness is the one that has to
//! give, because a compiler that probes the local machine bakes that machine
//! into its output. It also reported `bn254_permute` stale when its body was
//! identical — a false positive of the same cause.
//!
//! Freshness therefore lives in `tests/committed_ptx_artifacts.rs`
//! (`the_shipped_gpu_kernels_match_their_sources`), which pins a
//! `.ysu_hw_profile` to the artifact's OWN declared target and runs under the
//! plain `cargo test`. Portability lives in `tests/ptx_portability.rs`
//! (`every_shipped_kernel_loads_on_every_supported_card`), which requires the
//! sm_80 floor and assembles at six architectures.
//!
//! What is left here is the claim only this crate can make: that the strings
//! it compiles into the binary really are those files, filed under the right
//! entry names. A mis-wired `include_str!` embeds a valid, fresh, portable
//! kernel under the wrong name and fails at load — on a machine with a GPU,
//! which is not this one in CI.

use std::path::{Path, PathBuf};
use std::process::Command;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A temp directory tagged with the calling test AND the process, because this
/// repo has been bitten five times by two runs sharing one path.
fn work_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("y_gpu_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("temp dir");
    d
}

/// The name a PTX module exports, i.e. what `cuModuleGetFunction` will be
/// asked for.
fn declared_entry(ptx: &str) -> Option<String> {
    ptx.lines()
        .find(|l| l.trim_start().starts_with(".visible .entry"))
        .and_then(|l| l.split(".entry").nth(1))
        .map(|rest| rest.trim().trim_end_matches('(').trim().to_string())
}

/// Each embedded constant is the file it is named after. Catches a hand-edited
/// `ptx/` file and a mis-pointed `include_str!` — the second of which produces
/// a module that is fresh, portable, assembles, and is the wrong kernel.
#[test]
fn every_embedded_kernel_is_the_file_it_names() {
    let mut checked = 0usize;
    for (entry, embedded) in y_gpu::kernels::ALL {
        let path = crate_root().join(format!("ptx/{entry}.ptx"));
        let on_disk = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            on_disk.trim(),
            embedded.trim(),
            "the constant filed under `{entry}` is not the contents of ptx/{entry}.ptx"
        );
        checked += 1;
    }
    assert!(checked >= 5, "only {checked} embedded kernels found; the list shrank");
}

/// ...and each one exports the entry point it is filed under. `load()` asks
/// the driver for `entry` by name, so a pair that disagrees is a load failure
/// that only appears on a machine with a GPU.
#[test]
fn every_embedded_kernel_declares_the_entry_it_is_filed_under() {
    let mut checked = 0usize;
    for (entry, ptx) in y_gpu::kernels::ALL {
        let declared = declared_entry(ptx)
            .unwrap_or_else(|| panic!("{entry}: the embedded module declares no .visible .entry"));
        assert_eq!(
            &declared, entry,
            "the module filed under `{entry}` exports `{declared}`, so loading it by \
             name will fail at run time"
        );
        checked += 1;
    }
    assert!(checked >= 5, "only {checked} embedded kernels found; the list shrank");
}

/// Every embedded kernel assembles at the architecture it declares.
///
/// Deliberately the DECLARED target and not a hardcoded one: an assemble gate
/// pinned to the build machine's card cannot find a portability bug, which is
/// the lesson `coprocessor_ptx_assembles` produced when it hardcoded sm_89 —
/// the one arch where the version it was checking happened to be legal. The
/// authoritative sweep is `tests/ptx_portability.rs` from the workspace root,
/// which requires the sm_80 floor and assembles at six architectures; this is
/// the in-crate floor, so `cargo test -p y-gpu` is not vacuous on its own.
#[test]
fn embedded_ptx_assembles() {
    if Command::new("ptxas").arg("--version").output().is_err() {
        eprintln!("SKIP: ptxas not found.");
        return;
    }
    let dir = work_dir("assembles");
    for (entry, ptx) in y_gpu::kernels::ALL {
        let arch = ptx
            .lines()
            .find(|l| l.trim_start().starts_with(".target"))
            .map(|l| l.trim().trim_start_matches(".target").trim().to_string())
            .unwrap_or_else(|| panic!("{entry}: the embedded module declares no .target"));
        let path = dir.join(format!("{entry}.ptx"));
        std::fs::write(&path, ptx).unwrap();
        let out = Command::new("ptxas")
            .arg(format!("-arch={arch}"))
            .arg("-o")
            .arg(dir.join(format!("{entry}.cubin")))
            .arg(&path)
            .output()
            .expect("ptxas");
        assert!(
            out.status.success(),
            "embedded {entry} does not assemble at its own .target {arch}:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The sources still exist. Not a freshness check — that is
/// `the_shipped_gpu_kernels_match_their_sources` at the root — but a deleted
/// `.ysu` would leave a kernel nobody can regenerate, and this crate is where
/// the pairing is declared.
#[test]
fn every_embedded_kernel_still_has_a_source() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (entry, _) in y_gpu::kernels::ALL {
        let ysu = root.join(format!("tests/{entry}.ysu"));
        assert!(
            ysu.exists(),
            "{entry} ships in this library and tests/{entry}.ysu is gone, so the \
             embedded PTX can never be regenerated"
        );
    }
}
