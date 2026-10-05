//! Execute Coq against valid and deliberately broken schedule proofs.
//!
//! These are compiler probes, not token-scanner fixtures: Coq must accept the
//! valid partition and its concrete counterexamples, reject incorrect or
//! incomplete proofs, and expose admitted or axiomatized claims to the same
//! assumption checker used for the production proofs.

#[path = "common/coq.rs"]
mod coq;
#[path = "common/verification.rs"]
mod verification;

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_PROBE: AtomicUsize = AtomicUsize::new(0);

const SCHEDULE: &str = r#"From Stdlib Require Import Arith Lia.
Definition visit (workers index : nat) : nat :=
  workers * (index / workers) + index mod workers.
"#;

// Quotient/remainder is the grid-stride ownership map. The positive worker
// count bounds the owner; the reconstruction prevents a skipped index.
const PARTITION: &str = r#"Theorem schedule_partition : forall workers index,
  0 < workers -> index mod workers < workers /\ visit workers index = index.
"#;

const PROOF: &str = r#"Proof.
  intros workers index Hworkers. split.
  - apply Nat.mod_upper_bound. lia.
  - unfold visit. symmetry. apply Nat.div_mod_eq.
Qed.
"#;

struct PrivateCoqDir(PathBuf);

impl PrivateCoqDir {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before UNIX epoch")
            .as_nanos();
        let sequence = NEXT_PROBE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "y_proof_mutation_{}_{}_{}",
            std::process::id(),
            nonce,
            sequence
        ));
        std::fs::create_dir(&path).expect("create isolated Coq probe directory");
        Self(path)
    }
}

impl Drop for PrivateCoqDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn coqc_available() -> bool {
    let available = Command::new("coqc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    verification::prerequisite_available(available, "coqc for formal mutation probes")
}

fn compile(source: &str) -> (bool, String) {
    let directory = PrivateCoqDir::new();
    std::fs::write(directory.0.join("Probe.v"), source).expect("write Coq probe");
    let output = Command::new("coqc")
        .arg("Probe.v")
        .current_dir(&directory.0)
        .output()
        .expect("execute Coq probe");
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), text)
}

fn partition_source(schedule: &str, statement: &str, proof: &str) -> String {
    format!("{schedule}{statement}{proof}Print Assumptions schedule_partition.\n")
}

fn assert_accepted(source: &str, reports: usize) {
    let (success, output) = compile(source);
    assert!(
        coq::check_coq_output(success, &output, reports).is_ok(),
        "valid Coq control was rejected:\n{output}"
    );
}

fn assert_rejected(source: &str, compilation_should_succeed: bool) -> String {
    let (success, output) = compile(source);
    assert_eq!(
        success, compilation_should_succeed,
        "unexpected Coq result for mutation:\n{output}"
    );
    assert!(
        coq::check_coq_output(success, &output, 1).is_err(),
        "proof gate accepted an invalid mutation:\n{output}"
    );
    output
}

#[test]
fn valid_grid_stride_partition_is_accepted() {
    if !coqc_available() {
        return;
    }
    assert_accepted(&partition_source(SCHEDULE, PARTITION, PROOF), 1);
}

#[test]
fn unfinished_proof_is_rejected_by_coq() {
    if !coqc_available() {
        return;
    }
    let unfinished = "Proof. intros workers index Hworkers. Qed.\n";
    let output = assert_rejected(&partition_source(SCHEDULE, PARTITION, unfinished), false);
    assert!(output.contains("incomplete proof"), "{output}");
}

#[test]
fn admitted_partition_compiles_but_is_rejected_by_the_proof_gate() {
    if !coqc_available() {
        return;
    }
    let output = assert_rejected(&partition_source(SCHEDULE, PARTITION, "Admitted.\n"), true);
    assert!(output.contains("Axioms:"), "{output}");
    assert!(output.contains("schedule_partition"), "{output}");
}

#[test]
fn axiomatized_partition_compiles_but_is_rejected_by_the_proof_gate() {
    if !coqc_available() {
        return;
    }
    let oracle = PARTITION.replace("Theorem schedule_partition", "Axiom schedule_oracle");
    let source = format!(
        "{SCHEDULE}{oracle}{PARTITION}Proof. exact schedule_oracle. Qed.\n\
         Print Assumptions schedule_partition.\n"
    );
    let output = assert_rejected(&source, true);
    assert!(output.contains("Axioms:"), "{output}");
    assert!(output.contains("schedule_oracle"), "{output}");
}

#[test]
fn changed_stride_is_rejected_and_its_wrong_index_is_proved() {
    if !coqc_available() {
        return;
    }
    let changed = SCHEDULE.replace(
        "workers * (index / workers)",
        "(workers + 1) * (index / workers)",
    );
    assert_rejected(&partition_source(&changed, PARTITION, PROOF), false);
    // At two workers, changing the stride maps source index 2 to index 3.
    // Accepting this counterexample rules out a broken Coq invocation that
    // simply rejects every source, including the negative mutation.
    let counterexample = format!(
        "{changed}Example changed_stride_skips_index : visit 2 2 = 3 /\\ visit 2 2 <> 2.\n\
         Proof. split; [reflexivity | discriminate]. Qed.\n\
         Print Assumptions changed_stride_skips_index.\n"
    );
    assert_accepted(&counterexample, 1);
}

#[test]
fn removing_worker_bound_is_rejected_and_zero_worker_counterexample_checks() {
    if !coqc_available() {
        return;
    }
    let unconditional = PARTITION.replace("0 < workers -> ", "");
    let without_bound = PROOF.replace("intros workers index Hworkers.", "intros workers index.");
    assert_rejected(
        &partition_source(SCHEDULE, &unconditional, &without_bound),
        false,
    );
    let counterexample = format!(
        "{SCHEDULE}Example zero_workers_cannot_own_an_index : ~ (1 mod 0 < 0).\n\
         Proof. cbn. lia. Qed.\n\
         Print Assumptions zero_workers_cannot_own_an_index.\n"
    );
    assert_accepted(&counterexample, 1);
}
