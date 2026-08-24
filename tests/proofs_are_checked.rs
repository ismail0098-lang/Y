//! The Coq proof is part of the product, so it has to be part of the suite.
//!
//! `proofs/ZkControlFlow.v` is the one machine-checked artifact in this repo,
//! and until now it was run by a command in CLAUDE.md and by **nothing else** -
//! no test, no build script, no CI. That is the shape this repo already
//! catalogues under other names: `@ZeroDrift` before it was implemented, the
//! `scheme = "plonkish"` flag, the `wgmma...s4` that existed on no hardware.
//! An unrun proof is the paperwork of a proof.
//!
//! It matters more here than for an ordinary test because the proof is about a
//! MODEL of the ZK control-flow lowering rather than about `zk_emitter.rs`
//! itself. The model can drift from the code silently; the least this gate can
//! do is guarantee the model still type-checks and still rests on nothing.
//!
//! Three claims, and the third is the one that rots quietly:
//!
//! 1. `coqc` accepts the file.
//! 2. Every `Print Assumptions` reports `Closed under the global context` -
//!    i.e. no axioms. An `Axioms:` section appearing later would mean a proof
//!    now depends on an assumption nobody stated.
//! 3. Nothing is `Admitted`. `coqc` accepts an admitted lemma happily and
//!    prints a warning that no exit code reflects.
//!
//! Skipped with a printed notice when `coqc` is absent, like the `ptxas`,
//! `solcjs` and `z3` gates. That is a real hole - CI without Rocq is not
//! checking this - and it is stated rather than hidden.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn proof_source() -> PathBuf {
    repo().join("proofs/ZkControlFlow.v")
}

fn have_coqc() -> bool {
    Command::new("coqc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Compile the proof in a private directory.
///
/// Not in `proofs/` on purpose: `coqc` writes `.vo`/`.glob`/`.vok`/`.vos`
/// beside the source, so compiling in place would race a developer running
/// `coqc` by hand and would litter a tracked directory. The `.ptx` race in the
/// GPU harness was exactly this, and it presented as an intermittent failure.
fn compile_proof() -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("y_coq_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let dst = dir.join("ZkControlFlow.v");
    std::fs::copy(proof_source(), &dst).expect("copy the proof");

    let out = Command::new("coqc")
        .arg("ZkControlFlow.v")
        .current_dir(&dir)
        .output()
        .expect("run coqc");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), text)
}

#[test]
fn the_coq_proof_still_checks_and_rests_on_no_axioms() {
    if !have_coqc() {
        eprintln!("skipping: no coqc on PATH - the proof is NOT being checked");
        return;
    }
    let (ok, output) = compile_proof();
    assert!(
        ok,
        "`coqc proofs/ZkControlFlow.v` failed. This proof is a documented \
         command and the repo's only machine-checked artifact.\n{output}"
    );

    // `Print Assumptions` prints either `Closed under the global context` or an
    // `Axioms:` section listing what the proof leans on. The file asks for two.
    let closed = output.matches("Closed under the global context").count();
    assert!(
        closed >= 2,
        "expected every `Print Assumptions` to report `Closed under the global \
         context`; saw {closed}. Output:\n{output}"
    );
    assert!(
        !output.contains("Axioms:"),
        "the proof now depends on an axiom. `low_correct` is supposed to hold \
         unconditionally - an assumption here means the theorem is weaker than \
         its statement reads.\nOutput:\n{output}"
    );
}

/// `coqc` accepts an admitted lemma and exits 0. Only the source says.
#[test]
fn nothing_in_the_proof_is_admitted() {
    let src = std::fs::read_to_string(proof_source()).expect("read the proof");
    let mut offenders: Vec<String> = Vec::new();
    for (i, line) in src.lines().enumerate() {
        let code = line.split("(*").next().unwrap_or("").trim();
        // `Admitted.` closes a proof with a hole; `admit.` leaves one mid-proof.
        if code == "Admitted." || code.starts_with("admit") || code.contains(" admit.") {
            offenders.push(format!("line {}: {}", i + 1, line.trim()));
        }
    }
    assert!(
        offenders.is_empty(),
        "an admitted proof is the paperwork of a proof without the proof:\n  {}",
        offenders.join("\n  ")
    );
}

/// The control. Both tests above pass perfectly against a file containing no
/// theorems at all - "no axioms" and "nothing admitted" are properties an empty
/// file has. Same shape as `the_artifact_sweep_actually_finds_artifacts` and
/// `ordinary_loop_bodies_still_verify`.
#[test]
fn the_proof_actually_proves_the_lowering_correct() {
    let src = std::fs::read_to_string(proof_source()).expect("read the proof");
    // The load-bearing theorem: the shipped lowering agrees with the semantics
    // on every program and environment. If it is renamed, this gate must be
    // pointed at the new name deliberately rather than silently passing.
    assert!(
        src.contains("low_correct"),
        "`low_correct` is the theorem that makes this file worth checking - the \
         shipped lowering agrees with the operational semantics. It is gone or \
         renamed; point this gate at whatever replaced it."
    );
    assert!(
        src.contains("Print Assumptions low_correct"),
        "`low_correct` must be under `Print Assumptions`, or the axiom check \
         above says nothing about it."
    );
    // And the counterexample theorems, which are what pin WHICH change fixed
    // what - the flat-vs-nested distinction that Z3 and a fresh test file both
    // missed.
    assert!(
        src.contains("low_tail_is_wrong_when_nested"),
        "the machine-checked refutation of the old lowering is missing; without \
         it the file proves the new lowering right but not the old one wrong."
    );
}
