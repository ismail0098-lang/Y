//! Build fresh PTXAS artifacts and check the validator against controls and mutations.
use std::path::Path;
use std::process::Command;

#[path = "common/validator.rs"]
mod validator;

fn run_suite(name: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = validator::validator_python(root);
    let output = Command::new(&python)
        .arg(root.join("tests").join(name))
        .current_dir(root)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .unwrap_or_else(|error| panic!("cannot run PTXAS verification with {python:?}: {error}"));
    assert!(
        output.status.success(),
        "{name} failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
}

#[test]
fn fresh_ptxas_artifacts_preserve_the_expected_translation_results() {
    run_suite("ptxas_pipeline_regressions.py");
}

#[test]
fn ptxas_validator_rejects_unmodeled_or_changed_machine_behavior() {
    run_suite("ptxas_validator_regressions.py");
}

#[test]
fn integer_abstractions_preserve_congruence_and_exact_satisfiability() {
    run_suite("ptxas_integer_abstraction_regressions.py");
}

#[test]
fn integer_executors_require_defined_carry_and_supported_operands() {
    run_suite("ptxas_integer_semantics_regressions.py");
}

#[test]
fn memory_accesses_preserve_width_and_natural_alignment() {
    run_suite("ptxas_memory_regressions.py");
}

#[test]
fn validation_covers_legal_launch_domains_and_matching_targets() {
    run_suite("ptxas_domain_regressions.py");
}

#[test]
fn validator_verdicts_require_a_licensed_architecture() {
    run_suite("ptxas_architecture_regressions.py");
}

#[test]
fn ptx_directive_and_structural_lines_cannot_hide_executable_effects() {
    run_suite("ptxas_directive_regressions.py");
}

#[test]
fn loop_entry_control_flow_and_header_effects_are_checked() {
    run_suite("ptxas_loop_control_regressions.py");
}

#[test]
fn nested_loops_preserve_all_memory_accesses_and_effects() {
    run_suite("ptxas_nested_effect_regressions.py");
}
