//! Indexing an UNKNOWN yields an unknown - it is not "too many indices".
//!
//! A circom `function` whose branch depends on a signal is havoc'd, and a
//! `return` inside a skipped branch makes the function return `Val::Opaque` -
//! a SCALAR. Where the caller declared an array, every later `v[i]` then looked
//! like a mis-indexed scalar and was refused.
//!
//! `circom-ecdsa`'s `secp256k1_addunequal_func` is the motivating case: it runs
//! `mod_inv` over signal-derived values, which cannot be evaluated, so the whole
//! function comes back opaque and the `out[0][i] <-- tmp[0][i]` advice below it
//! could not be written.
//!
//! Opaque already propagates through `add_vals` and `mul_vals`; this is the
//! same rule for indexing, and it is sound for the same reason as the rest of
//! the model - the value may only reach a site that emits no constraint, and
//! `require_known` refuses it everywhere else.
//!
//! With this, **all 17 of circom-ecdsa's test circuits compile** (from 0 before
//! the signed-comparison fix). Constraint counts land at 0.97-0.98x circom's
//! across the suite, which is Y's reduction passes - a relaxation that dropped
//! constraints would show a far larger gap, since the advice signals number in
//! the hundreds.
//!
//! Run with:  cargo test --features zk --test circom_opaque_array_index

#![cfg(feature = "zk")]

use std::path::{Path, PathBuf};
use y::circom_lower::compile_file;
use y::zk_field::Fr;
use y::zk_witness::solve_r1cs_witness;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn compile(name: &str) -> Result<y::zk_emitter::ZkEmitter, String> {
    compile_file(
        &PathBuf::from(root()).join("tests/circom").join(name),
        &[root().join("circomlib/circuits")],
    )
}

/// The advice pattern compiles, matches circom exactly, and still solves.
#[test]
fn an_opaque_array_element_may_be_advice() {
    let e = compile("opaque_array_index_advice.circom")
        .unwrap_or_else(|e| panic!("indexing an opaque value must not be an error: {}", e));
    let circuit = e.build_circuit();

    // circom 2.2.3: 1 non-linear + 0 linear = 1 constraint, 3 wires.
    assert_eq!(circuit.constraints.len(), 1, "constraint count must match circom's 1");
    assert_eq!(circuit.num_variables, 3, "wire count must match circom's 3");

    // `out` is `<--` from an opaque value and bound by `out === a * a`.
    // a = 5  ->  out = 25, derived from the constraint, not the advice.
    let ir = e.build_witness_ir();
    let (w, sat) = solve_r1cs_witness(
        &circuit.constraints,
        &ir,
        circuit.num_variables,
        &[],
        &[Fr::from_u64(5)],
    );
    assert!(sat, "the witness must satisfy the circuit; `out` is pinned by the `===`");
    assert_eq!(
        w[circuit.outputs[0]].to_u64(),
        Some(25),
        "out should be a*a = 25, derived from the author's constraint"
    );
}

/// The controls. Without these, "never report too many indices" passes.
#[test]
fn the_refusals_that_must_survive() {
    // (a) an opaque element reaching a constraint is still refused, by the
    //     opaque check rather than by the index check.
    let e = compile("opaque_index_into_constraint.circom")
        .err()
        .expect("an opaque value reaching `<==` must be refused");
    assert!(
        e.contains("`<==` constraint"),
        "refused somewhere other than the `<==`: {}",
        e
    );

    // (b) a genuine mis-index of a KNOWN scalar is still an error. The
    //     relaxation is only for values the compiler cannot compute.
    let e = compile("too_many_indices_is_still_an_error.circom")
        .err()
        .expect("indexing a known scalar must still be refused");
    assert!(
        e.contains("too many indices"),
        "refused, but not by the index check - so this control proves nothing: {}",
        e
    );
}
