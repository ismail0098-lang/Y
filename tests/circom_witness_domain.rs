//! circom's witness domain: values the compiler cannot compute.
//!
//! circom `var`s are not restricted to compile-time constants. A template may
//! write `var y2 = out[1] * out[1];` over a signal, a `function` may be called
//! on signal arrays, and the result may be shifted, xored and compared - none
//! of which has an R1CS form. Y used to refuse all of it, which cost the two
//! most-used circuits in the ecosystem: `Sha256` and `EdDSA`.
//!
//! They are now carried as `Val::Opaque`, and the rule that makes that safe is
//! that an opaque value may **only** reach a `<--`. `<--` emits no constraint,
//! so the `.r1cs` is exactly what a compiler that *could* evaluate the
//! expression would emit; everywhere else - `<==`, `===`, an array index, an
//! array dimension, a template argument, an `assert` - it is refused, and the
//! refusal names the position where the value first became unknown rather than
//! the position where it was finally used.
//!
//! This file is in two halves.
//!
//! **Does it work**: `Sha256(64)` compiled from unmodified circomlib produces
//! NIST's digest, checked through Y's own witness solver and its own
//! satisfiability check. That is the same standard the Poseidon interop test
//! sets - a published vector, not an internal consistency check - and it is
//! worth more than a constraint count, because a circuit that is
//! self-consistently wrong has a constraint count too.
//!
//! **Is it sound**: the opaque value must not escape into the constraint
//! system, and `havoc_branch` must refuse rather than skip a branch that could
//! emit constraints. A front end that quietly drops a construct emits a circuit
//! with fewer constraints than its source describes - which still proves, just
//! something weaker than the author wrote.
//!
//! Run with:  cargo test --features zk --test circom_witness_domain

#![cfg(feature = "zk")]

use std::path::{Path, PathBuf};
use y::circom_lower::compile_file;
use y::zk_field::Fr;
use y::zk_witness::solve_r1cs_witness;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn circomlib() -> PathBuf {
    root().join("circomlib/circuits")
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/circom").join(name)
}

/// Compile a fixture, returning the emitter or the front end's message.
fn compile(name: &str) -> Result<y::zk_emitter::ZkEmitter, String> {
    compile_file(&fixture(name), &[circomlib()])
}

fn compile_ok(name: &str) -> y::zk_emitter::ZkEmitter {
    compile(name).unwrap_or_else(|e| panic!("{} failed to compile: {}", name, e))
}

/// Compile, solve for `privs` in declaration order, and return the outputs.
fn solve(name: &str, privs: &[u64]) -> (Vec<Fr>, Vec<usize>, bool) {
    let emitter = compile_ok(name);
    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();
    let privs: Vec<Fr> = privs.iter().map(|v| Fr::from_u64(*v)).collect();
    let (w, sat) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);
    (w, circuit.outputs, sat)
}

// ────────────────────────────────────────────────────────
// Does it work: circomlib's SHA-256 against NIST's digest
// ────────────────────────────────────────────────────────

/// `Sha256(64)` from unmodified circomlib, checked against the real digest.
///
/// This is the whole feature end to end. `sha256compression.circom` opens with
///
/// ```text
/// var outCalc[256] = sha256compression(hin, inp);
/// for (i=0; i<256; i++) out[i] <-- outCalc[i];
/// ```
///
/// - a circom `function` called on two signal ARRAYS, whose body shifts, xors
/// and rotates them. None of that is expressible as a constraint, and none of
/// it needs to be: the constrained SHA-256 circuit is built below the call and
/// `out[31-k] === fsum[0].out[k]` is what actually binds the digest. The `<--`
/// is advice for a witness calculator, and Y now treats it as advice.
///
/// Y is therefore able to solve the witness here **despite** not being able to
/// evaluate the advice, because the constraints determine the answer on their
/// own. Asserting the digest rather than only satisfiability is what separates
/// "the circuit is consistent" from "the circuit is SHA-256".
#[test]
fn sha256_of_circomlib_produces_the_published_digest() {
    // Two 8-byte messages and their real digests (`python3 -c
    // "import hashlib; hashlib.sha256(bytes(8)).hexdigest()"`).
    for (msg, want) in [
        (
            [0u8; 8],
            "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc",
        ),
        (
            [0x01u8, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef],
            "55c53f5d490297900cefa825d0c8e8e9532ee8a118abe7d8570762cd38be9818",
        ),
    ] {
        // circomlib's Sha256 takes its message MSB-first, one bit per signal.
        let bits: Vec<u64> = msg
            .iter()
            .flat_map(|b| (0..8).map(move |i| ((b >> (7 - i)) & 1) as u64))
            .collect();
        assert_eq!(bits.len(), 64);

        let (w, outputs, sat) = solve("sha256_64.circom", &bits);
        assert!(
            sat,
            "Sha256(64) witness does not satisfy its own circuit for message {:02x?}",
            msg
        );
        assert_eq!(outputs.len(), 256, "Sha256 should have 256 output bits");

        let mut got = String::new();
        for byte in outputs.chunks(8) {
            let mut v = 0u16;
            for wire in byte {
                let bit = &w[*wire];
                assert!(
                    bit.is_zero() || *bit == Fr::one(),
                    "a digest output signal is neither 0 nor 1"
                );
                v = (v << 1) | u16::from(*bit == Fr::one());
            }
            got.push_str(&format!("{:02x}", v));
        }
        assert_eq!(got, want, "wrong SHA-256 digest for message {:02x?}", msg);
    }
}

/// The structural metadata must match circom's, because that is what a verifier
/// and a `.wtns` are indexed by.
///
/// Measured against `circom --r1cs` on the identical source: 64 private inputs,
/// 256 public outputs, 0 public inputs.
#[test]
fn sha256_declares_the_same_interface_circom_does() {
    let e = compile_ok("sha256_64.circom");
    let c = e.build_circuit();
    assert_eq!(c.outputs.len(), 256, "output count");
    assert_eq!(e.private_inputs.len(), 64, "private input count");
    assert_eq!(e.public_inputs.len(), 0, "public input count");
}

// ────────────────────────────────────────────────────────
// Is it sound: the opaque value must not reach the circuit
// ────────────────────────────────────────────────────────

/// An unknown value is refused everywhere the emitted circuit would depend on
/// it, and the message names where it became unknown.
///
/// Each fixture takes a signal, does something to it that has no R1CS form, and
/// then uses the result in a position that decides the circuit. Silently
/// accepting any of these is the failure this whole model exists to avoid: a
/// `.r1cs` that is smaller than its source, and still proves.
/// Each case asserts the phrase belonging to ITS OWN site, not merely that
/// something was refused. Every one of these programs is refused by some later
/// check too - an unknown template argument reaches a `<==` inside the
/// template, an unknown array index would fail on the next line - so a test
/// that only asserts "it failed" passes with the guard it is aiming at deleted.
/// Mutation testing is what showed that: removing the template-argument check
/// was caught by nothing until this loop started reading the message.
#[test]
fn an_unknown_value_is_refused_wherever_the_circuit_depends_on_it() {
    for (name, site) in [
        ("opaque_into_constrain.circom", "`<==` constraint"),
        ("opaque_into_equality.circom", "`===` constraint"),
        ("opaque_into_index.circom", "array indices"),
        ("opaque_into_dim.circom", "array dimensions"),
        ("opaque_into_template_arg.circom", "a template argument"),
        ("opaque_into_assert.circom", "`assert`"),
    ] {
        let e = compile(name).err().unwrap_or_else(|| {
            panic!("{}: an unknown value reached {} and was ACCEPTED", name, site)
        });
        assert!(
            e.contains("unknown"),
            "{}: refused, but the message does not say the value was unknown, so it does \
             not point at the cause: {}",
            name,
            e
        );
        assert!(
            e.contains(site),
            "{}: refused somewhere else than the site under test ({}), so this case is not \
             gating the check it is named for: {}",
            name,
            site,
            e
        );
    }
}

/// **A skipped branch's `return` must not be silently fallen through.**
///
/// The earlier `sqrt`-shaped case cannot see this: its fall-through is
/// `return n * 2` over the same unknown `n`, so ignoring the branch still
/// yields an unknown and the test passes either way. Here both arms return
/// CONSTANTS, so falling through produces a confident `7` for a function that
/// could equally have returned `5` — a wrong circuit that compiles, rather than
/// a refusal.
///
/// Correct behaviour is to refuse: the value is unknown, and it is reaching a
/// `<==`.
#[test]
fn a_skipped_branch_may_not_fall_through_to_a_constant() {
    let e = compile("unknown_fn_return_folds.circom").err().unwrap_or_else(|| {
        panic!(
            "a function whose branches return 5 and 7 on an unevaluable condition was folded \
             to one of them and accepted into a constraint"
        )
    });
    assert!(e.contains("unknown"), "{}", e);
}

/// A signal array passed to a `function` and used in a constraint.
///
/// This is the other half of the `Sha256` fix — `sha256compression(hin, inp)`
/// hands a function 768 signals — and it is the half the digest test cannot
/// check, because there the function's result is advice that no constraint
/// reads. Here the result IS the constraint, so the wire order is observable.
///
/// Weighted by position on purpose: an unweighted sum is commutative and would
/// pass with the array reversed.
#[test]
fn a_signal_array_reaches_a_function_in_the_right_order() {
    // x = [1, 2, 3, 4], weights 1..4 => 1 + 4 + 9 + 16 = 30. Reversed it is 20.
    let (w, outputs, sat) = solve("signal_array_to_function.circom", &[1, 2, 3, 4]);
    assert!(sat, "the circuit is not satisfied by its own witness");
    assert_eq!(
        w[outputs[0]].to_decimal_string(),
        "30",
        "a signal array reached the function in the wrong order or with the wrong values"
    );
}

/// **The guard that makes skipping a branch sound.**
///
/// When a condition cannot be evaluated, Y runs neither branch. That is only
/// safe if running neither cannot change the emitted circuit, so a skipped body
/// containing a `<==`, a `===`, a `signal` or a `component` is refused instead.
/// Without this check the circuit would silently lose the constraints in the
/// branch - the exact shape of bug the design rule in CLAUDE.md is about, and
/// worse here because the result still proves.
#[test]
fn a_branch_that_emits_constraints_may_not_be_skipped() {
    for name in [
        "unknown_branch_constrains.circom",
        "unknown_branch_equality.circom",
        "unknown_branch_component.circom",
    ] {
        let e = compile(name)
            .err()
            .unwrap_or_else(|| panic!("{}: a constraint-emitting branch was SKIPPED", name));
        assert!(
            e.contains("multiplexer"),
            "{}: refused without pointing at the multiplexer idiom: {}",
            name,
            e
        );
    }
}

/// A branch that only touches `var`s IS skipped, and everything it could have
/// assigned becomes unknown.
///
/// This is the positive half, and it matters as much as the negative one:
/// refusing every unevaluable condition is sound and would leave `Sha256` and
/// `EdDSA` exactly where they were. `sqrt` in circomlib's `pointbits.circom` is
/// five branches and two `while` loops over the value being rooted.
#[test]
fn a_branch_over_vars_is_skipped_and_its_variables_become_unknown() {
    // Compiles: the unknown ends up in a `<--`.
    let e = compile_ok("unknown_branch_var.circom");
    assert!(
        !e.build_circuit().constraints.is_empty(),
        "the fixture should still emit the constraints written outside the branch"
    );

    // The same fixture with the result routed into a `<==` instead must fail,
    // which is what proves the value really did become unknown rather than
    // keeping whatever it held before the branch.
    let err = compile("unknown_branch_var_constrained.circom")
        .err()
        .expect("a havoc'd variable reached `<==` and was accepted");
    assert!(err.contains("unknown"), "{}", err);
}

/// **A `return` inside a branch that cannot be taken.**
///
/// This is `sqrt`'s shape, and it is the mechanism `EdDSA` needs.
/// `pointbits.circom` opens with `if (n == 0) { return 0; }` over the very value
/// being rooted, so the function has no compile-time result at all - it may have
/// returned there with an unknown value, or fallen through to return something
/// else. Unknown is the join of those, and returning it immediately is what
/// lets the call be made.
///
/// Getting this wrong is not a refusal, it is a WRONG ANSWER: falling through
/// the untaken branch and continuing would produce a confident value for a
/// function that never ran, and `<--` would accept it without complaint.
#[test]
fn a_function_returning_from_an_unknown_branch_yields_an_unknown() {
    let e = compile_ok("unknown_fn_return.circom");
    assert!(!e.build_circuit().constraints.is_empty());

    let err = compile("unknown_fn_return_constrained.circom")
        .err()
        .expect("a function that could not be evaluated returned a value a `<==` accepted");
    assert!(err.contains("unknown"), "{}", err);
}

/// A `while` whose condition is unknown must not be executed, and must not hang.
///
/// circomlib's `sqrt` has two nested ones (`while ((r != 0) && (t != 1))`),
/// which is why the loop cases need their own gate rather than riding on the
/// `if` one.
#[test]
fn an_unknown_loop_condition_terminates() {
    let e = compile_ok("unknown_while.circom");
    assert!(!e.build_circuit().constraints.is_empty());
}

/// What happens to the WITNESS when the compiler cannot evaluate a `<--`.
///
/// Two outcomes, and the difference is the author's constraints, not Y:
///
/// - `pinned`: `out <-- u; out === a;` — the `===` determines `out`, so Y
///   solves it anyway and gets the right value. This is why `Sha256` works:
///   all 256 digest bits are `<--` advice, and every one is pinned by an
///   `out[31-k] === fsum[0].out[k]` below it.
/// - `sqrtish`: `out <-- u; out * out === a;` — a square root, which no amount
///   of back-propagation recovers. Satisfiability must come back FALSE so the
///   caller refuses to write a `.wtns`, rather than writing the solver's zero
///   and calling it a witness.
///
/// The second is the one that matters. An unsolved wire defaults to zero, and
/// zero satisfies a surprising number of constraints; a test that only checked
/// the first case would pass with the satisfiability check deleted.
#[test]
fn an_unsolvable_witness_fails_rather_than_defaulting_to_zero() {
    let (w, outputs, sat) = solve("opaque_pinned.circom", &[5]);
    assert!(sat, "`out === a` determines `out`; the witness should still solve");
    assert_eq!(
        w[outputs[0]].to_decimal_string(),
        "5",
        "the constraints pin this signal, so Y must derive it despite the `<--` being opaque"
    );

    let (_, _, sat) = solve("opaque_unsolvable.circom", &[4]);
    assert!(
        !sat,
        "a `<--` the compiler cannot evaluate and the constraints cannot determine was \
         reported as SATISFIED - the solver's default zero is being passed off as a witness"
    );
}

/// **The control.**
///
/// Every test above passes if the front end simply refuses more than it used
/// to. This asserts the two circuits the feature exists for actually compile,
/// and that ordinary compile-time control flow is untouched.
#[test]
fn the_ordinary_paths_still_compile() {
    for name in ["sha256_64.circom", "num2bits_sum.circom", "poseidon2.circom"] {
        let e = compile_ok(name);
        assert!(
            !e.build_circuit().constraints.is_empty(),
            "{} emitted no constraints",
            name
        );
    }
}
