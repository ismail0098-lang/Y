//! Shared policy for optional local checks and required verification runs.

pub fn strict() -> bool {
    std::env::var_os("Y_VERIFICATION_STRICT").is_some_and(|value| value == "1")
}

/// Missing dependencies may skip in ordinary local tests. The verification
/// runner sets `Y_VERIFICATION_STRICT=1` so a skip cannot certify an unchecked
/// proof, solver obligation, or executable artifact.
pub fn prerequisite_available(available: bool, requirement: &str) -> bool {
    if available {
        return true;
    }
    assert!(
        !strict(),
        "Y_VERIFICATION_STRICT=1 requires {requirement}; this verification check cannot be skipped"
    );
    eprintln!("SKIP: {requirement} is unavailable");
    false
}
