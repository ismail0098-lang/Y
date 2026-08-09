//! End-to-end proof test: Y source -> R1CS -> witness -> Groth16 -> verify.
//!
//! Before this existed, the ZK backend stopped at constraint emission. It could
//! write a `.r1cs` file, but nothing in the repo ever proved anything with one,
//! and no independent implementation had ever read the constraints back. "R1CS
//! backend" therefore meant "emits a file shaped like an R1CS" - the emitter's
//! own hand-rolled BN254 arithmetic was both the producer and the only judge.
//!
//! These tests close that loop with arkworks as an INDEPENDENT oracle (a
//! dev-dependency only - the shipping `Y` binary still links nothing). If Y's
//! field arithmetic, wire numbering, constant-wire convention, or constraint
//! layout were wrong in a way its own checker shares, `Groth16::verify` would
//! reject, because arkworks reimplements all of it separately.
//!
//! Run with:  cargo test --features zk --test zk_groth16_end_to_end
#![cfg(feature = "zk")]

use ark_bn254::{Bn254, Fr as ArkFr};
use ark_ff::PrimeField;
use ark_groth16::Groth16;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystemRef, LinearCombination as ArkLc, SynthesisError, Variable,
};
use ark_snark::SNARK;
use ark_std::rand::{rngs::StdRng, SeedableRng};

use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{Circuit, Fr, LinearCombination, ZkEmitter};
use y::zk_witness::{check_r1cs_satisfiability, solve_r1cs_witness};

/// Y's `Fr` -> arkworks' `Fr`.
///
/// `BigUint` stores base-2^32 digits little-endian, so the little-endian byte
/// serialisation is just each digit's LE bytes in order. Going through bytes
/// rather than a decimal string keeps this independent of Y's formatting code -
/// a bug in `to_decimal_string` must not be able to hide a bug in the field.
fn to_ark(v: &Fr) -> ArkFr {
    ArkFr::from_le_bytes_mod_order(&v.to_bytes_le(32))
}

fn ark_lc(lc: &LinearCombination, vars: &[Variable]) -> ArkLc<ArkFr> {
    let mut out = ArkLc::zero();
    for (wire, coeff) in &lc.terms {
        out = out + (to_ark(coeff), vars[*wire]);
    }
    out
}

/// Compiles Y source to a circuit plus a satisfying witness.
fn compile(source: &str, pub_in: &[u64], priv_in: &[u64]) -> (Circuit, Vec<Fr>) {
    let tokens = Lexer::new(source).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    TypeChecker::new().check_program(&program);

    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&program).expect("zk lowering");
    let circuit = emitter.build_circuit();
    let witness_ir = emitter.build_witness_ir();

    let pubs: Vec<Fr> = pub_in.iter().map(|v| Fr::from_u64(*v)).collect();
    let privs: Vec<Fr> = priv_in.iter().map(|v| Fr::from_u64(*v)).collect();

    let (witness, satisfied) = solve_r1cs_witness(
        &circuit.constraints,
        &witness_ir,
        circuit.num_variables,
        &pubs,
        &privs,
    );
    assert!(satisfied, "solver did not produce a satisfying witness");
    check_r1cs_satisfiability(&circuit.constraints, &witness).expect("native satisfiability");
    (circuit, witness)
}

/// Y's circuit, replayed into an arkworks constraint system.
///
/// The public wires are the circuit's declared outputs plus its declared public
/// inputs - that is the statement worth proving for a Y `fn main`: "I know
/// private inputs that drive this circuit to the published output." Every other
/// wire, including the private inputs and all intermediates, is a witness.
#[derive(Clone)]
struct YCircuit {
    circuit: Circuit,
    witness: Vec<Fr>,
    public_wires: Vec<usize>,
}

impl YCircuit {
    fn new(circuit: Circuit, witness: Vec<Fr>) -> Self {
        let mut public_wires = circuit.public_inputs.clone();
        for o in &circuit.outputs {
            if !public_wires.contains(o) {
                public_wires.push(*o);
            }
        }
        public_wires.retain(|w| *w != 0);
        Self { circuit, witness, public_wires }
    }

    /// The public input vector, in the same order the wires are allocated -
    /// which is the order `Groth16::verify` expects.
    fn public_values(&self) -> Vec<ArkFr> {
        self.public_wires.iter().map(|w| to_ark(&self.witness[*w])).collect()
    }
}

impl ConstraintSynthesizer<ArkFr> for YCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<ArkFr>) -> Result<(), SynthesisError> {
        let n = self.circuit.num_variables;
        let mut vars = vec![Variable::One; n];

        // Public first, so allocation order matches `public_values()`.
        for &wire in &self.public_wires {
            let v = self.witness[wire].clone();
            vars[wire] = cs.new_input_variable(|| Ok(to_ark(&v)))?;
        }
        for wire in 1..n {
            if self.public_wires.contains(&wire) {
                continue;
            }
            let v = self.witness[wire].clone();
            vars[wire] = cs.new_witness_variable(|| Ok(to_ark(&v)))?;
        }
        // vars[0] stays Variable::One - R1CS wire 0 is the constant 1.

        for c in &self.circuit.constraints {
            cs.enforce_constraint(
                ark_lc(&c.a, &vars),
                ark_lc(&c.b, &vars),
                ark_lc(&c.c, &vars),
            )?;
        }
        Ok(())
    }
}

const POLY_SRC: &str = r#"
@unsafe
fn main(x: I32, y: I32) -> I32 {
    let mut temp = x;
    for i in 0..8 {
        temp = temp * y;
    }
    return temp;
}
"#;

/// Y's hand-rolled BN254 scalar modulus must be the real one. If this drifts,
/// every proof below would still verify against Y's own arithmetic while being
/// worthless against any real verifier.
#[test]
fn y_field_modulus_matches_bn254() {
    assert_eq!(
        Fr::modulus().to_decimal_string(),
        ArkFr::MODULUS.to_string(),
        "Y's ACTIVE_MODULUS is not the BN254 scalar field modulus"
    );
}

/// The whole loop: compile, witness, prove, verify.
#[test]
fn groth16_proof_of_y_circuit_verifies() {
    // x=3, y=2 -> 3 * 2^8 = 768. Deliberately not all-ones: an all-ones witness
    // satisfies almost any multiplicative circuit and would hide real errors.
    let (circuit, witness) = compile(POLY_SRC, &[], &[3, 2]);
    assert!(!circuit.constraints.is_empty(), "circuit emitted no constraints");

    let c = YCircuit::new(circuit, witness);
    let public = c.public_values();
    assert!(!public.is_empty(), "circuit exposes nothing public to prove about");

    // The circuit must compute the RIGHT function, not merely a self-consistent
    // one. A satisfiable R1CS whose output wire holds the wrong value would pass
    // every check above and still be a broken compiler: x*y^8 = 3*2^8 = 768.
    assert_eq!(
        *public.last().unwrap(),
        ArkFr::from(768u64),
        "circuit output wire is not x*y^8; the R1CS is satisfiable but computes the wrong function"
    );

    let mut rng = StdRng::seed_from_u64(0);
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).expect("setup");
    let proof = Groth16::<Bn254>::prove(&pk, c, &mut rng).expect("prove");

    assert!(
        Groth16::<Bn254>::verify(&vk, &public, &proof).expect("verify"),
        "an honest Groth16 proof over Y's own R1CS failed to verify"
    );
}

/// Soundness direction: the proof must be bound to the public output. Verifying
/// the same proof against a different claimed output has to fail, or the
/// circuit proves nothing.
#[test]
fn groth16_proof_rejects_tampered_public_input() {
    let (circuit, witness) = compile(POLY_SRC, &[], &[3, 2]);
    let c = YCircuit::new(circuit, witness);
    let public = c.public_values();

    let mut rng = StdRng::seed_from_u64(0);
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).expect("setup");
    let proof = Groth16::<Bn254>::prove(&pk, c, &mut rng).expect("prove");

    let mut tampered = public.clone();
    tampered[0] += ArkFr::from(1u64);
    assert!(
        !Groth16::<Bn254>::verify(&vk, &tampered, &proof).expect("verify"),
        "a proof verified against a public output it does not correspond to"
    );
}

/// Ordering comparisons must compute the RIGHT function.
///
/// Regression guard for a soundness hole: `BinaryOp::Lt | Le | Gt | Ge` used to
/// re-emit the expression as `BinaryOp::NotEq` whenever either operand was not
/// a compile-time constant. So `x < y` lowered to `x != y` - `5 <= 5` was
/// false, and `5 > 3` and `3 > 5` were both true. Groth16 proves a wrong
/// statement exactly as readily as a right one, so this was not a precision
/// issue. The tell was that `<`, `>`, `<=`, `==` and `!=` all emitted an
/// identical 3 constraints; a real ordering comparison needs bit decomposition
/// and now costs 101.
#[test]
fn comparison_operators_compute_correct_values() {
    let src = |op: &str| format!("@unsafe\nfn main(x: I32, y: I32) -> I32 {{ return x {op} y; }}\n");
    // (op, x, y, expected)
    let cases: &[(&str, u64, u64, u64)] = &[
        ("<", 5, 3, 0), ("<", 3, 5, 1), ("<", 4, 4, 0),
        (">", 5, 3, 1), (">", 3, 5, 0), (">", 4, 4, 0),
        ("<=", 5, 4, 0), ("<=", 4, 4, 1), ("<=", 3, 4, 1),
        (">=", 3, 4, 0), (">=", 4, 4, 1), (">=", 5, 4, 1),
        ("==", 4, 4, 1), ("==", 4, 5, 0),
        ("!=", 5, 3, 1), ("!=", 4, 4, 0),
    ];
    for (op, x, y, want) in cases {
        let (circuit, witness) = compile(&src(op), &[], &[*x, *y]);
        let out = witness[*circuit.outputs.last().unwrap()].clone();
        assert_eq!(
            to_ark(&out),
            ArkFr::from(*want),
            "{} {} {} produced the wrong value",
            x, op, y
        );
    }
}

/// The comparison gadget must survive a real prover, not just the native
/// satisfiability check - it is the piece with bit decomposition, booleanity
/// constraints and a range check, so it is where a wire-numbering or
/// coefficient error would actually bite.
#[test]
fn groth16_proves_a_comparison_circuit() {
    let src = "@unsafe\nfn main(x: I32, y: I32) -> I32 { return x < y; }\n";
    let (circuit, witness) = compile(src, &[], &[3, 5]);
    let c = YCircuit::new(circuit, witness);
    let public = c.public_values();
    assert_eq!(*public.last().unwrap(), ArkFr::from(1u64), "3 < 5 should be 1");

    let mut rng = StdRng::seed_from_u64(0);
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).expect("setup");
    let proof = Groth16::<Bn254>::prove(&pk, c, &mut rng).expect("prove");
    assert!(
        Groth16::<Bn254>::verify(&vk, &public, &proof).expect("verify"),
        "Groth16 could not verify a proof over the comparison gadget"
    );

    // And it must still be bound to the claimed answer.
    let mut tampered = public.clone();
    tampered[0] += ArkFr::from(1u64);
    assert!(
        !Groth16::<Bn254>::verify(&vk, &tampered, &proof).expect("verify"),
        "comparison proof verified against a public output it does not match"
    );
}

/// A real proof over the integer and bitwise gadgets.
///
/// `tests/zk_integer_ops.rs` checks those operators against Rust's semantics
/// using Y's own witness solver, which shares Y's field arithmetic and wire
/// numbering with the emitter. This closes that loop the same way the rest of
/// this file does: arkworks re-derives the constraint system independently, so
/// a gadget that is self-consistently wrong still fails here.
#[test]
fn groth16_proves_an_integer_gadget_circuit() {
    // 123 % 10 = 3, 123 / 10 = 12, (12 & 6) = 4, 4 | 3 = 7.
    let src = "@unsafe\nfn main(x: I32, y: I32) -> I32 { return ((x / y) & 6) | (x % y); }\n";
    let (circuit, witness) = compile(src, &[], &[123, 10]);
    let c = YCircuit::new(circuit, witness);
    let public = c.public_values();
    assert_eq!(
        *public.last().unwrap(),
        ArkFr::from(7u64),
        "((123/10) & 6) | (123%10) should be 7"
    );

    let mut rng = StdRng::seed_from_u64(0);
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).expect("setup");
    let proof = Groth16::<Bn254>::prove(&pk, c, &mut rng).expect("prove");
    assert!(
        Groth16::<Bn254>::verify(&vk, &public, &proof).expect("verify"),
        "Groth16 could not verify a proof over the integer/bitwise gadgets"
    );

    let mut tampered = public.clone();
    tampered[0] += ArkFr::from(1u64);
    assert!(
        !Groth16::<Bn254>::verify(&vk, &tampered, &proof).expect("verify"),
        "integer gadget proof verified against a public output it does not match"
    );
}

/// A witness that does not satisfy the constraints must be caught before it
/// ever reaches a prover - and must not be provable into a verifying proof.
#[test]
fn corrupted_witness_fails_satisfiability() {
    let (circuit, mut witness) = compile(POLY_SRC, &[], &[3, 2]);
    // Perturb an intermediate wire, not wire 0 (the constant) or an input.
    let victim = circuit.num_variables / 2;
    witness[victim] = witness[victim].add(&Fr::one());
    assert!(
        check_r1cs_satisfiability(&circuit.constraints, &witness).is_err(),
        "a perturbed witness still satisfied every constraint - the circuit is under-constrained"
    );
}
