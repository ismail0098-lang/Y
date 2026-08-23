//! circom 2.1 anonymous components, tuples, and whole-array signal assignment.
//!
//! These three arrive together because they are one feature in circom's own
//! release, and because the circuits that use them use all three at once.
//! semaphore's body is four lines of it:
//!
//! ```text
//!     isLessThan.in <== [secret, l];              // array assignment
//!     (Ax, Ay) = BabyPbk()(secret);               // tuple + anonymous
//!     var c = Poseidon(2)([Ax, Ay]);              // anonymous in a value position
//!     RangeCheck(LIMIT_BIT_SIZE)(messageId, ...); // anonymous as a bare statement
//! ```
//!
//! **The design claim is that this is a DESUGARING, and that is what is
//! asserted here rather than a constraint count.** `o <== T()(i)` must emit
//! byte-for-byte the same `.r1cs` as
//! `component c = T(); c.a <== i; o <== c.out;`. circom satisfies the same
//! property on the simple case, checked with circom 2.2.3 before any of this
//! was written; on the larger fixture pair circom agrees on every reported
//! count (1 non-linear / 3 linear / 7 wires both ways) and differs only in
//! wire numbering inside the file, so the byte claim is made of Y's output
//! against Y's own.
//!
//! **Every refusal here is a refusal circom also makes**, with its error code
//! recorded beside the case. That matters more than usual in this front end:
//! a permissive lowering of `T()(a <== x, b <== y)` that ignored the names
//! would compile, solve, and prove -- a different circuit from the one the
//! author wrote, which is the shape of the signed-comparison bug.
//!
//! Run with:  cargo test --features zk --test circom_anonymous_components

#![cfg(feature = "zk")]

use std::path::{Path, PathBuf};
use y::circom_lower::compile_file;
use y::zk_emitter::ZkEmitter;
use y::zk_field::Fr;
use y::zk_witness::solve_r1cs_witness;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/circom").join(name)
}

fn compile(name: &str) -> Result<ZkEmitter, String> {
    compile_file(&fixture(name), &[root().join("circomlib/circuits")])
}

fn compile_ok(name: &str) -> ZkEmitter {
    compile(name).unwrap_or_else(|e| panic!("{} failed to compile: {}", name, e))
}

/// Serialize the circuit and return the bytes. Comparing ARTIFACTS rather than
/// counts is the point: two different circuits can share a constraint count.
fn r1cs_bytes(name: &str, emitter: &ZkEmitter) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!(
        "y_anon_{}_{}",
        std::process::id(),
        name.replace('.', "_")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("c.r1cs");
    emitter
        .write_r1cs_binary(path.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: could not write r1cs: {}", name, e));
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

/// Solve the circuit for the given private inputs and return its single output.
fn solve_one(name: &str, privs: &[Fr]) -> (bool, Option<u64>) {
    let emitter = compile_ok(name);
    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();
    let (w, sat) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], privs);
    assert_eq!(circuit.outputs.len(), 1, "{}: expected one output", name);
    (sat, w[circuit.outputs[0]].to_u64())
}

// ────────────────────────────────────────────────────────
// The claim: it is a desugaring
// ────────────────────────────────────────────────────────

/// The anonymous form and the hand-written explicit form must be the SAME
/// artifact.
///
/// If this ever fails, the desugaring has acquired a meaning of its own and
/// the difference has to be explained before it is accepted -- an anonymous
/// component that emits something the explicit form does not is a circuit the
/// author did not write.
#[test]
fn the_anonymous_form_is_exactly_the_explicit_one() {
    let anon = compile_ok("anon_component.circom");
    let expl = compile_ok("anon_component_explicit.circom");

    let a = r1cs_bytes("anon", &anon);
    let b = r1cs_bytes("explicit", &expl);

    assert_eq!(
        a.len(),
        b.len(),
        "the anonymous and explicit forms produced .r1cs files of different SIZE \
         ({} vs {} bytes)",
        a.len(),
        b.len()
    );
    assert!(
        a == b,
        "the anonymous form is not the desugaring it claims to be: its .r1cs \
         differs from the hand-written explicit component"
    );
}

/// ...and the circuit computes the right thing.
///
/// Byte-identity to a hand-written circuit is satisfied by two circuits that
/// are equally wrong, so the value is pinned separately. `AnonTwo` gives
/// `x = a*b`, `y = a + 2b`; `AnonSum(3)` sums its input. The fixture computes
/// `o = y + (i + j + x)` = `2i + 3j + i*j`, which at i=7, j=3 is 44.
#[test]
fn the_desugared_circuit_computes_what_the_source_says() {
    let (sat, out) = solve_one("anon_component.circom", &[Fr::from_u64(7), Fr::from_u64(3)]);
    assert!(sat, "the solved witness does not satisfy its own circuit");
    assert_eq!(out, Some(44), "o should be 2*7 + 3*3 + 7*3 = 44");
}

// ────────────────────────────────────────────────────────
// Named inputs bind by NAME
// ────────────────────────────────────────────────────────

/// `T()(b <== j, a <== i)` must be `T()(i, j)`, not `T()(j, i)`.
///
/// This is the test that a positional-zipping lowering fails. `AnonTwo` is
/// asymmetric in `b` on purpose: without that, both spellings would produce
/// the same circuit and the whole case would be vacuous.
#[test]
fn named_inputs_bind_by_name_and_not_by_position() {
    let named = r1cs_bytes("named", &compile_ok("anon_named_inputs.circom"));
    let positional = r1cs_bytes("positional", &compile_ok("anon_named_positional.circom"));
    let swapped = r1cs_bytes("swapped", &compile_ok("anon_named_swapped.circom"));

    assert!(
        named == positional,
        "`T()(b <== j, a <== i)` did not produce the same circuit as `T()(i, j)`"
    );
    assert!(
        named != swapped,
        "`T()(b <== j, a <== i)` produced the same circuit as `T()(j, i)` -- the \
         input NAMES are being ignored and the arguments zipped positionally, \
         which silently computes a different function"
    );
}

/// The values, so "different bytes" is backed by "different answers".
#[test]
fn the_two_orderings_really_do_compute_different_functions() {
    let ins = [Fr::from_u64(7), Fr::from_u64(3)];
    // a=i=7, b=j=3: x = 21, y = 7 + 6 = 13, o = 34.
    assert_eq!(solve_one("anon_named_inputs.circom", &ins), (true, Some(34)));
    assert_eq!(solve_one("anon_named_positional.circom", &ins), (true, Some(34)));
    // a=j=3, b=i=7: x = 21, y = 3 + 14 = 17, o = 38.
    assert_eq!(solve_one("anon_named_swapped.circom", &ins), (true, Some(38)));
}

// ────────────────────────────────────────────────────────
// The other two spellings the real circuits use
// ────────────────────────────────────────────────────────

/// An anonymous component with no outputs, written as a whole statement.
/// circom-rln's `RangeCheck(LIMIT_BIT_SIZE)(messageId, userMessageLimit);`.
///
/// The constraint it exists for must actually be emitted, which is checked by
/// giving it a value that violates it: `AnonBool` says `a*(a-1) === 0`, so
/// i = 5 must NOT satisfy the circuit. A lowering that parsed the statement
/// and dropped it would leave a satisfiable circuit here.
#[test]
fn a_bare_anonymous_statement_still_emits_its_constraints() {
    let (sat_ok, out) = solve_one("anon_bare_statement.circom", &[Fr::from_u64(1)]);
    assert!(sat_ok, "i = 1 satisfies a*(a-1) = 0 and should solve");
    assert_eq!(out, Some(2), "o should be i + 1");

    let (sat_bad, _) = solve_one("anon_bare_statement.circom", &[Fr::from_u64(5)]);
    assert!(
        !sat_bad,
        "i = 5 violates `a*(a-1) === 0`, so the circuit must be unsatisfiable -- \
         the bare anonymous statement emitted no constraint"
    );
}

/// `_` discards an output. The component is still instantiated and its inputs
/// are still constrained; only the binding is dropped.
#[test]
fn an_underscore_discards_an_output_without_discarding_the_component() {
    let (sat, out) = solve_one("anon_discarded_output.circom", &[Fr::from_u64(6)]);
    assert!(sat, "the solved witness does not satisfy its own circuit");
    // (keep, _) <== AnonTwo()(i, i)  ->  keep = i*i = 36.
    assert_eq!(out, Some(36), "keep should be 6*6 = 36");
}

// ────────────────────────────────────────────────────────
// Refusals. Every one of these is refused by circom 2.2.3 too.
// ────────────────────────────────────────────────────────

fn refused(name: &str, needle: &str, circom_code: &str) {
    match compile(name) {
        Ok(_) => panic!(
            "{} compiled, but circom refuses it with {}. Accepting it here means Y \
             and circom give the same source two different meanings.",
            name, circom_code
        ),
        Err(e) => assert!(
            e.contains(needle),
            "{}: refused for the wrong reason.\n  expected to mention: {}\n  got: {}",
            name,
            needle,
            e
        ),
    }
}

#[test]
fn an_anonymous_component_inside_arithmetic_is_refused() {
    // circom error[TAC01]: "This is the anonymous component whose use is not
    // allowed". Y's own reason names the same restriction.
    refused(
        "anon_in_arithmetic.circom",
        "may only be the entire right-hand side",
        "error[TAC01]",
    );
}

#[test]
fn an_anonymous_component_under_a_witness_arrow_is_refused() {
    // circom error[TAC01]: "Anonymous components only admit the use of the
    // operator <==". Accepting `<--` would constrain the component's INPUTS
    // while leaving its outputs free, which no author can have meant.
    refused(
        "anon_with_witness_arrow.circom",
        "may only be used with `<==`",
        "error[TAC01]",
    );
}

#[test]
fn taking_two_outputs_into_one_target_is_refused() {
    // circom error[TAC02]: "Left-side of the statement is not a tuple".
    refused(
        "anon_two_outputs_one_target.circom",
        "output signal(s) but 1 target(s)",
        "error[TAC02]",
    );
}

#[test]
fn a_tuple_of_the_wrong_length_is_refused() {
    // circom error[TAC02]: "The number of elements in both tuples does not
    // coincide".
    refused(
        "anon_tuple_arity.circom",
        "output signal(s) but 3 target(s)",
        "error[TAC02]",
    );
}

#[test]
fn a_bare_statement_is_refused_when_the_template_has_outputs() {
    // circom error[TAC02]: "This expression must be a tuple or an anonymous
    // component". The bare form is legal ONLY for a template with no outputs.
    refused(
        "anon_bare_with_outputs.circom",
        "output signal(s) but 0 target(s)",
        "error[TAC02]",
    );
}

#[test]
fn the_wrong_number_of_inputs_is_refused() {
    // circom error[TAC01]: "The number of template input signals must coincide
    // with the number of input parameters".
    refused(
        "anon_input_arity.circom",
        "must coincide with the number of",
        "error[TAC01]",
    );
}

#[test]
fn a_repeated_input_name_is_refused() {
    // circom error[TAC01] (as a count mismatch). Naming one input twice would
    // otherwise leave the other one silently unconstrained.
    refused(
        "anon_duplicate_named_input.circom",
        "is given twice",
        "error[TAC01]",
    );
}

#[test]
fn an_unknown_input_name_is_refused() {
    refused(
        "anon_unknown_named_input.circom",
        "has no input signal `z`",
        "error[TAC01]",
    );
}

#[test]
fn mixing_named_and_positional_inputs_is_refused() {
    // circom error[P1012]: it does not parse at all.
    refused(
        "anon_mixed_named_positional.circom",
        "all positional or all named",
        "error[P1012]",
    );
}

// ────────────────────────────────────────────────────────
// Controls
// ────────────────────────────────────────────────────────

/// Refusing every anonymous component would satisfy every refusal test above
/// and delete the whole feature. This is what stops that.
#[test]
fn ordinary_anonymous_components_still_compile() {
    for name in [
        "anon_component.circom",
        "anon_named_inputs.circom",
        "anon_bare_statement.circom",
        "anon_discarded_output.circom",
    ] {
        let emitter = compile_ok(name);
        assert!(
            !emitter.build_circuit().constraints.is_empty(),
            "{} compiled to a circuit with no constraints at all",
            name
        );
    }
}
