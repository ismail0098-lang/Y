//! "Compilation Successful!" must appear if and only if the compiler succeeded.
//!
//! The banner used to be printed BEFORE the backend dispatch ran, so every
//! backend failure was announced under it:
//!
//! ```text
//! Compilation Successful!
//!
//! [4/4] Emitting Native x86-64 ELF Binary...
//! [!] [Native x86-64 Backend] `if` (this backend emits no branches) ...
//! ```
//!
//! The exit code was always right (1) and no artifact was ever written, so
//! this is not a correctness bug. It is the *reporting* half of the failure
//! shape this repo keeps finding - `wgmma_async` printing success and writing
//! PTX that `ptxas` rejects, `--target=r1cs` without the `zk` feature exiting
//! 0 as a silent no-op, `@zk_target(scheme="plonkish")` agreeing with a user
//! about a scheme it did not use. A green banner above a red error costs a
//! reader the assumption that the tool means what it says.
//!
//! The property asserted here is the biconditional, not just the negative
//! direction: "never print the banner" would satisfy half of it and is
//! useless, the same way `ordinary_loop_bodies_still_verify` guards the
//! invariant checker against refusing everything.
//!
//! Run with:  cargo test --test success_banner_means_success

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

const BANNER: &str = "Compilation Successful!";

/// Each case gets its own directory: two tests writing `case.ysu` into one
/// shared path is the `.ptx` race this repo has now hit in three files.
static SALT: AtomicUsize = AtomicUsize::new(0);

struct Run {
    ok: bool,
    text: String,
}

fn run(src: &str, flags: &[&str]) -> Run {
    let n = SALT.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("y_banner_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();

    // Copy the cached hardware profile in so the probe does not re-run on the
    // GPU once per case.
    let profile = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".ysu_hw_profile");
    if profile.exists() {
        let _ = std::fs::copy(&profile, dir.join(".ysu_hw_profile"));
    }

    let path = dir.join("case.ysu");
    std::fs::write(&path, src).unwrap();

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .args(flags)
        .current_dir(&dir)
        .output()
        .expect("failed to run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
    Run { ok: out.status.success(), text }
}

fn check(label: &str, src: &str, flags: &[&str], want_ok: bool) {
    let r = run(src, flags);
    assert_eq!(
        r.ok, want_ok,
        "{}: expected exit {} but got {}\n{}",
        label,
        if want_ok { "0" } else { "non-zero" },
        if r.ok { "0" } else { "non-zero" },
        r.text
    );
    let said = r.text.contains(BANNER);
    assert_eq!(
        said, want_ok,
        "{}: exit was {} but the banner was {}printed\n{}",
        label,
        if r.ok { "0" } else { "non-zero" },
        if said { "" } else { "NOT " },
        r.text
    );
}

/// A backend refusal must not be announced as a success.
#[test]
fn a_backend_refusal_does_not_print_the_banner() {
    // `--emit-native` is a straight-line integer subset and refuses branches
    // by name. Chosen because it needs no external toolchain.
    check(
        "native refuses `if`",
        "fn main() -> I32 { let a: I32 = 1; if a > 0 { return 2; } return 3; }",
        &["--emit-native"],
        false,
    );
    check(
        "native refuses 64-bit types",
        "fn main() -> I64 { let a: I64 = 5; return a; }",
        &["--emit-native"],
        false,
    );
}

/// The control: a compile that really works must still say so.
///
/// Without this, deleting the banner entirely passes the test above.
#[test]
fn a_real_success_still_prints_the_banner() {
    check(
        "native accepts its own subset",
        "fn main() -> I32 { let a: I32 = 9; let b: I32 = 2; return a - b; }",
        &["--emit-native"],
        true,
    );
}

/// The same biconditional on the ZK backend, where the artifact is a proof.
#[cfg(feature = "zk")]
#[test]
fn the_zk_backend_obeys_the_same_rule() {
    check(
        "r1cs refuses an out-of-range comparison operand",
        "fn main(a: I32) -> I32 { if 4294967296 < a { return 1; } return 0; }",
        &["--target=r1cs"],
        false,
    );
    check(
        "r1cs accepts an in-range one",
        "fn main(a: I32) -> I32 { if 5 < a { return 1; } return 0; }",
        &["--target=r1cs"],
        true,
    );
}
