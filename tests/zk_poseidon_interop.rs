//! Y's Poseidon must agree with circomlib's, digit for digit.
//!
//! This is an interop test, not a smoke test, and the distinction is the whole
//! point. A hash function is only useful if someone else computes the same
//! value: a Merkle tree built by circomlib has to verify inside a Y circuit and
//! vice versa. Every internally-consistent hash passes a "does it run" test.
//! Only the published constants pass this one.
//!
//! The expected values below come from circomlib 2.0.5's own `Poseidon(2)`
//! template, compiled with `circom --wasm` and evaluated through its witness
//! calculator. `tools/extract_poseidon.py` regenerates the constant table from
//! the same source. `poseidon(1,2)` is additionally the test vector published
//! in circomlibjs, so it can be checked against a third party by eye.
//!
//! Run with:  cargo test --features zk --test zk_poseidon_interop
#![cfg(feature = "zk")]

use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{Fr, ZkEmitter};
use y::zk_witness::{check_r1cs_satisfiability, solve_r1cs_witness};

/// Compiles `poseidon_hash(a, b)` and returns (digest, constraint count).
fn poseidon_of(a: u64, b: u64) -> (String, usize) {
    let src = "@unsafe\nfn main(x: I32, y: I32) -> I32 {\n    return poseidon_hash(x, y);\n}\n";

    let tokens = Lexer::new(src).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    TypeChecker::new().check_program(&program);

    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&program).expect("zk lowering");
    let circuit = emitter.build_circuit();
    let witness_ir = emitter.build_witness_ir();

    let (witness, satisfied) = solve_r1cs_witness(
        &circuit.constraints,
        &witness_ir,
        circuit.num_variables,
        &[],
        &[Fr::from_u64(a), Fr::from_u64(b)],
    );
    assert!(satisfied, "poseidon witness does not satisfy its own circuit");
    check_r1cs_satisfiability(&circuit.constraints, &witness).expect("satisfiability");

    let out_wire = *circuit.outputs.first().expect("circuit has an output");
    (witness[out_wire].to_decimal_string(), circuit.constraints.len())
}

/// The interop gate.
///
/// If this fails after a change to `zk_poseidon_constants.rs` or
/// `emit_poseidon`, the hash has silently forked from circomlib's. Do not
/// "update the expected values" - regenerate them from circomlib and find out
/// why they moved.
#[test]
fn poseidon_matches_circomlib() {
    // circomlib 2.0.5, Poseidon(2), via its wasm witness calculator.
    let vectors: [(u64, u64, &str); 4] = [
        (1, 2, "7853200120776062878684798364095072458815029376092732009249414926327459813530"),
        (3, 4, "14763215145315200506921711489642608356394854266165572616578112107564877678998"),
        (0, 0, "14744269619966411208579211824598458697587494354926760081771325075741142829156"),
        (7, 0, "10402197090275139279073177788985849389816807868761640028215734431067655199248"),
    ];

    for (a, b, expected) in vectors {
        let (got, _) = poseidon_of(a, b);
        assert_eq!(
            got, expected,
            "poseidon({}, {}) disagrees with circomlib.\n  Y:        {}\n  circomlib: {}",
            a, b, got, expected
        );
    }
}

/// One Poseidon costs 241 constraints, and the arithmetic is worth spelling
/// out because it is the check that no wire is being wasted.
///
/// 81 S-boxes (`t*R_F + R_P = 3*8 + 57`) at 3 constraints each is 243 - which
/// is exactly what `circom --r1cs` reports as non-linear for `Poseidon(2)`.
/// Y then differs by two:
///
///   -3  the capacity lane starts at the constant 0, so `0 + C[0]` is constant
///       through the first round's S-box and folds at compile time.
///   +1  the emitter binds `main`'s return value to an output wire.
///
/// Y emits no linear constraints at all, where circom reports 274 of them,
/// because every mix here is folded directly into the `A`/`B` operands of the
/// next multiplication instead of being given its own signal. A jump in this
/// number means an `allocate_var_from_lc` crept back into the round function -
/// the digest would still be correct, so `poseidon_matches_circomlib` would
/// not catch it.
#[test]
fn poseidon_costs_241_constraints() {
    let (_, n) = poseidon_of(1, 2);
    assert_eq!(n, 241, "expected 241 constraints for one Poseidon, got {}", n);
}

/// Unsupported arities must be refused, not approximated.
///
/// The previous implementation accepted any number of inputs and padded its
/// constant table with a hardcoded filler value, producing a confident,
/// deterministic, entirely non-standard hash. Failing closed is the only safe
/// behaviour: a wrong digest cannot be detected downstream, because every
/// digest looks like noise.
#[test]
fn poseidon_rejects_unsupported_arity() {
    for src in [
        "@unsafe\nfn main(x: I32) -> I32 {\n    return poseidon_hash(x);\n}\n",
        "@unsafe\nfn main(x: I32, y: I32, z: I32) -> I32 {\n    return poseidon_hash(x, y, z);\n}\n",
    ] {
        let tokens = Lexer::new(src).tokenize();
        let program = Parser::new(tokens).parse_program().expect("parse");
        TypeChecker::new().check_program(&program);
        let mut emitter = ZkEmitter::new();
        let err = emitter
            .emit_program(&program)
            .expect_err("non-t=3 arity must be rejected, not silently hashed");
        assert!(
            err.contains("poseidon_hash"),
            "error should name the offending call, got: {}",
            err
        );
    }
}
