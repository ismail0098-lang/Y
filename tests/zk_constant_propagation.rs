//! Constant propagation through the constraint system, and the cycle it can
//! create.
//!
//! `substitute_linear_constraints` used to accept exactly one shape:
//! `k * lin = c_w * w`, a linear constraint whose `C` is a single wire. That is
//! the `<==` circom emits for every assignment, and it is not the only way a
//! wire gets defined.
//!
//! The one it missed is `k * lin = c0` with `c0` **constant**, which PINS a
//! wire rather than defining it in terms of others. Skipping it left Y at
//! circom's `--O1` on the entire `EscalarMulFix` family, because that shape is
//! the *seed* of constant propagation and without it the cascade never starts:
//! circomlib passes a fixed base point in as a signal, `MontgomeryDouble` and
//! `MontgomeryAdd` derive `2B..8B` from it using `<--` (witness-domain, so
//! invisible to any folding at lowering time), and the `===` that pins each
//! result back down is exactly this shape. Pin the first and every later stage
//! becomes constant in turn. Measured on `EscalarMulFix(8)`: **147 -> 26
//! constraints**, against circom `--O2`'s 21.
//!
//! Deleting constraints is the one optimisation that can silently weaken a
//! proof, so this file does not test "did it get smaller" - it tests that the
//! reduced circuit still binds its output, that a contradiction stays
//! unsatisfiable, and that the witness the reduced circuit produces satisfies
//! the UNREDUCED one. There is a control asserting the pass removes anything at
//! all, without which a pass that did nothing would satisfy every other test
//! here.
//!
//! Run with:  cargo test --features zk --test zk_constant_propagation

#![cfg(feature = "zk")]

use std::path::{Path, PathBuf};
use y::circom_lower::compile_file;
use y::zk_emitter::{set_linsub_budget, set_wire_compaction, Constraint};
use y::zk_field::Fr;
use y::zk_witness::{check_r1cs_satisfiability, solve_r1cs_witness};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/circom").join(name)
}

fn circomlib() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("circomlib/circuits")
}

struct Run {
    constraints: Vec<Constraint>,
    witness: Vec<Fr>,
    satisfied: bool,
    out: usize,
}

/// Compile and solve with the pass set to `budget` on this thread.
///
/// Thread-local rather than an env var, so flipping it cannot race the rest of
/// the binary. Wire compaction is off on both arms for the reason given in
/// `zk_linear_substitution.rs`: every assertion below is stated in wire
/// numbers, and compaction renumbers by a different amount on each arm because
/// they leave different wires dead.
fn run(name: &str, budget: Option<usize>, inputs: &[u64]) -> Run {
    set_linsub_budget(budget);
    set_wire_compaction(false);
    let emitter = compile_file(&fixture(name), &[circomlib()])
        .unwrap_or_else(|e| panic!("{} failed to compile: {}", name, e));
    set_linsub_budget(Some(16));
    set_wire_compaction(true);

    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();
    let privs: Vec<Fr> = inputs.iter().map(|v| Fr::from_u64(*v)).collect();
    let (witness, satisfied) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);
    let out = *circuit.outputs.first().expect("circuit has no output signal");
    Run { constraints: circuit.constraints, witness, satisfied, out }
}

const OFF: Option<usize> = None;
const ON: Option<usize> = Some(16);

/// The value must not move. `a = 4`, `b = 16`, `c = 20`, so `out = 20 + x`.
#[test]
fn constant_propagation_preserves_the_computed_value() {
    for x in [0u64, 1, 7, 1000] {
        let on = run("const_chain.circom", ON, &[x]);
        let off = run("const_chain.circom", OFF, &[x]);
        assert!(on.satisfied, "reduced circuit has no witness at x = {}", x);
        assert!(off.satisfied, "unreduced circuit has no witness at x = {}", x);
        assert_eq!(
            on.witness[on.out].to_decimal_string(),
            (20 + x).to_string(),
            "reduced circuit computed the wrong value at x = {}",
            x
        );
        assert_eq!(
            on.witness[on.out].to_decimal_string(),
            off.witness[off.out].to_decimal_string(),
            "reduction changed the computed value at x = {}",
            x
        );
    }
}

/// **The control.** A pass that eliminated nothing would satisfy every other
/// test in this file.
#[test]
fn constant_propagation_actually_removes_constraints() {
    let off = run("const_chain.circom", OFF, &[5]).constraints.len();
    let on = run("const_chain.circom", ON, &[5]).constraints.len();
    assert!(
        on < off,
        "the pass removed nothing ({} constraints either way)",
        off
    );
    eprintln!("const_chain.circom {} -> {} constraints", off, on);
}

/// The reduced circuit must still force its output.
///
/// The failure mode that matters: a pass that deletes the constraint binding
/// `out` leaves a circuit that compiles, proves, and proves nothing about the
/// output.
#[test]
fn the_reduced_circuit_still_binds_its_output() {
    for (name, inputs) in [
        ("const_chain.circom", &[7u64][..]),
        ("bits_roundtrip.circom", &[12345][..]),
    ] {
        let r = run(name, ON, inputs);
        assert!(r.satisfied, "{}: no witness", name);
        check_r1cs_satisfiability(&r.constraints, &r.witness)
            .unwrap_or_else(|e| panic!("{}: honest witness rejected: {}", name, e));

        let mut tampered = r.witness.clone();
        tampered[r.out] = tampered[r.out].add(&Fr::one());
        assert!(
            check_r1cs_satisfiability(&r.constraints, &tampered).is_err(),
            "{}: the output wire is unconstrained after reduction - changing it \
             still satisfies every constraint",
            name
        );
    }
}

/// An over-determined circuit must stay impossible.
///
/// `3*a === 12` and `5*a === 15` disagree. The pass pins `a` from the first and
/// the second must collapse to a non-zero constant identity, which
/// `constraint_is_vacuous` keeps on purpose. Dropping it as "trivial" would
/// turn an unsatisfiable statement into a provable one.
#[test]
fn a_contradiction_survives_constant_propagation() {
    let r = run("const_chain_contradiction.circom", ON, &[1]);
    assert!(
        !r.satisfied || check_r1cs_satisfiability(&r.constraints, &r.witness).is_err(),
        "an unsatisfiable circuit became satisfiable under constant propagation"
    );
}

/// The strongest available check: the witness produced for the REDUCED circuit
/// must satisfy the UNREDUCED one.
///
/// Wire numbering is identical between the two - the pass deletes constraints
/// and rewrites terms, it never allocates or renumbers - so the witnesses are
/// directly comparable. A pin with a wrong coefficient, or a constraint deleted
/// that was not really a definition, fails here even when the output still
/// happens to come out right.
#[test]
fn the_reduced_witness_satisfies_the_original_circuit() {
    for (name, inputs) in [
        ("const_chain.circom", &[9u64][..]),
        ("bits_roundtrip.circom", &[65535][..]),
    ] {
        let off = run(name, OFF, inputs);
        let on = run(name, ON, inputs);
        assert_eq!(
            on.witness.len(),
            off.witness.len(),
            "{}: the pass changed the wire count, so the witnesses are not comparable",
            name
        );
        check_r1cs_satisfiability(&off.constraints, &on.witness).unwrap_or_else(|e| {
            panic!("{}: witness from the reduced circuit fails the original: {}", name, e)
        });
    }
}

/// **The cycle regression, and the reason this shape may only pin CONSTANTS.**
///
/// `Num2Bits`' recomposition is `1 * (sum 2^i b_i - in) = 0`: a linear equation
/// with an empty `C`. Read as a definition it says `in` is the sum of the bits
/// - true, and useless, because every bit's `BitOfLc` recipe decomposes `in`.
/// Substituting it rewrites those recipes to derive each bit from a value
/// derived from the bits themselves.
///
/// The circuit stayed satisfiable and no witness could be found, which reads as
/// "this circuit is unprovable" rather than as a compiler bug - the same
/// signature as the `optimize_circuit` gadget-wire bug in `CLAUDE.md`. The fix
/// is that a pivot taken from anywhere other than `C`'s own single term must
/// resolve to a **constant**, which references no wire and so cannot close a
/// cycle.
#[test]
fn bit_decomposition_still_solves() {
    for x in [0u64, 1, 8, 12345, 65535] {
        let r = run("bits_roundtrip.circom", ON, &[x]);
        assert!(
            r.satisfied,
            "Num2Bits(16) recomposition has no witness at x = {} - a wire was \
             pinned from a constraint its own witness recipe depends on",
            x
        );
        assert_eq!(
            r.witness[r.out].to_decimal_string(),
            x.to_string(),
            "bit decomposition did not recompose {}",
            x
        );
    }
}

/// **A convergence gate, not a correctness one.**
///
/// The other tests here all pass if the pass merely produces a *correct*
/// circuit, and reading each constraint as stored rather than resolving it
/// through the eliminations already made in the same round does exactly that -
/// it just converges one chain link per round and hits the safety cap with most
/// of the circuit still symbolic. Mutation testing is what showed that: neutering
/// `resolve_lc` was caught by nothing until this test existed.
///
/// Eight stages, each quadratic until the previous is pinned. Measured: **1
/// constraint** with in-round resolution, **10** without.
#[test]
fn a_deep_constant_chain_collapses_completely() {
    let r = run("const_chain_deep.circom", ON, &[3]);
    assert!(r.satisfied, "deep chain has no witness");
    assert!(
        r.constraints.len() <= 3,
        "the constant chain did not fully collapse: {} constraints left. The \
         pass is converging one chain link per round - see `resolve_lc`.",
        r.constraints.len()
    );
    // a[8] = 4 * 4^8 = 262144, so out = 262144 + x.
    assert_eq!(
        r.witness[r.out].to_decimal_string(),
        (262144u64 + 3).to_string(),
        "deep chain computed the wrong value"
    );
}
