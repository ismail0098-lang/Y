//! Validate Rocq's evidence as well as its process status.

pub fn check_coq_output(
    success: bool,
    output: &str,
    expected_assumptions: usize,
) -> Result<(), String> {
    if !success {
        return Err(format!("coqc failed:\n{output}"));
    }
    if expected_assumptions == 0 {
        return Err(
            "no Print Assumptions requests; nothing checks the proof's dependencies".into(),
        );
    }
    if output.contains("Axioms:") {
        return Err(format!("the proof now depends on an axiom:\n{output}"));
    }
    let closed = output.matches("Closed under the global context").count();
    if closed != expected_assumptions {
        return Err(format!(
            "expected {expected_assumptions} Print Assumptions reports, got {closed} closed reports:\n{output}"
        ));
    }
    Ok(())
}
