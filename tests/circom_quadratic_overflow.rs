//! Arithmetic that exceeds one R1CS constraint is WITNESS-domain, not an error.
//!
//! `circom-ecdsa`'s `BigMult` is the motivating circuit and it states the case
//! plainly:
//!
//! ```circom
//! for (var i = 0; i < ka; i++) {
//!     for (var j = 0; j < kb; j++) {
//!         prod_val[i + j] += a[i] * b[j];      // <- a var, over SIGNALS
//!     }
//! }
//! for (var i = 0; i < ka + kb - 1; i++) {
//!     out[i] <-- prod_val[i];                  // <- advice, no constraint
//! }
//! ```
//!
//! `prod_val` accumulates a polynomial product: far more than the `a*b + c`
//! that one R1CS constraint holds. But it never appears IN a constraint - it
//! reaches a `<--`, and the author binds `out` with a separate polynomial
//! identity below. Y refused the ARITHMETIC, which refused a legal circuit.
//!
//! The refusal belongs at the point of USE, where `require_known` already
//! names both the site and the cause. That is the same correction `assert` over
//! signals needed (`tests/circom_assert_is_witness_domain.rs`) and rests on the
//! same property: an opaque value may only reach somewhere that emits no
//! constraint.
//!
//! **This is the change in this family with the most room to go wrong** - a
//! relaxation that let constraints go missing would produce a circuit that
//! still proves, just something weaker. So the tests below check counts against
//! circom EXACTLY, check the witness solves, check the value, and check that
//! every constraint-domain use is still refused.
//!
//! circom-ecdsa's suite went 7/17 -> 12/17 on this change.
//!
//! Run with:  cargo test --features zk --test circom_quadratic_overflow

#![cfg(feature = "zk")]

use std::path::{Path, PathBuf};
use y::circom_lower::compile_file;
use y::zk_field::Fr;
use y::zk_witness::solve_r1cs_witness;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/circom").join(name)
}

fn compile(name: &str) -> Result<y::zk_emitter::ZkEmitter, String> {
    compile_file(&fixture(name), &[root().join("circomlib/circuits")])
}

// ────────────────────────────────────────────────────────
// It compiles, and it is still fully constrained
// ────────────────────────────────────────────────────────

/// The advice pattern compiles, matches circom's counts EXACTLY, and solves.
///
/// Counts are asserted against circom 2.2.3's own output rather than against a
/// number of Y's own choosing: a relaxation that dropped a constraint would
/// still compile and still prove, and only a comparison against the reference
/// can see it. The witness solve is the second half - it proves `out` is
/// determined by the author's `===`, i.e. that the circuit did not merely get
/// smaller.
#[test]
fn quadratic_overflow_as_advice_matches_circom_exactly() {
    let e = compile("quadratic_overflow_advice.circom")
        .unwrap_or_else(|e| panic!("the advice pattern must compile: {}", e));
    let circuit = e.build_circuit();

    // circom 2.2.3: non-linear 2 + linear 1 = 3 constraints, 8 wires.
    assert_eq!(
        circuit.constraints.len(),
        3,
        "constraint count must match circom's 3 - fewer means the relaxation \
         dropped something the source asked for"
    );
    assert_eq!(circuit.num_variables, 8, "wire count must match circom's 8");

    // a=2, b=3, c=4, d=5  ->  out = 2*3 + 4*5 = 26.
    let privs = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(4), Fr::from_u64(5)];
    let ir = e.build_witness_ir();
    let (w, sat) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);

    assert!(
        sat,
        "the witness does not satisfy the circuit - `out` is `<--` from an opaque \
         value and must be pinned by the author's `===`"
    );
    assert_eq!(circuit.outputs.len(), 1);
    assert_eq!(
        w[circuit.outputs[0]].to_u64(),
        Some(26),
        "out should be 2*3 + 4*5 = 26"
    );
}

// ────────────────────────────────────────────────────────
// ...and every constraint-domain use is still refused
// ────────────────────────────────────────────────────────

/// The controls. Without these, "make all arithmetic opaque" passes the file.
///
/// Each case asserts the phrase belonging to its own SITE *and* that the
/// message names the CAUSE, because moving a refusal from the arithmetic to the
/// point of use is only an improvement if the user is still told why.
#[test]
fn a_constraint_may_not_use_an_over_quadratic_value() {
    for (name, site, cause) in [
        (
            "quadratic_overflow_into_constraint.circom",
            "`<==` constraint",
            "sum of two quadratic expressions",
        ),
        (
            "quadratic_overflow_into_equality.circom",
            "`===` constraint",
            "sum of two quadratic expressions",
        ),
        (
            "degree_three_into_constraint.circom",
            "`<==` constraint",
            "degree greater than 2",
        ),
    ] {
        let e = compile(name)
            .err()
            .unwrap_or_else(|| panic!("{}: an over-quadratic value reached a {} and was ACCEPTED - \
                                       circom refuses this program", name, site));
        assert!(
            e.contains(site),
            "{}: refused somewhere other than the site under test ({}): {}",
            name,
            site,
            e
        );
        assert!(
            e.contains(cause),
            "{}: refused without naming WHY the value was unknown ({}), so moving the \
             refusal to the point of use lost the diagnosis: {}",
            name,
            cause,
            e
        );
    }
}
