//! End-to-end cost of PROVING a large Y circuit, not just emitting it.
//!
//! `#[ignore]`d: this is a benchmark, not a correctness gate, and at 1M
//! constraints it needs minutes and several GB. Run explicitly:
//!
//! ```bash
//! cargo test --release --features zk --test zk_groth16_scale -- --ignored --nocapture
//! ```
//!
//! Why it exists: `docs/heavy_circuit_speed_test.md` measures COMPILE time, and
//! Y wins that decisively. But compile time is only the first of three costs,
//! and it is the smallest. Anyone comparing Y against a zkVM (SP1, RISC Zero)
//! is really asking about the total, so the total is what this reports.
//! Note that Y itself has no prover - the setup/prove numbers here are
//! arkworks', reached through Y's R1CS.
#![cfg(feature = "zk")]

use ark_bn254::{Bn254, Fr as ArkFr};
use ark_ff::PrimeField;
use ark_groth16::Groth16;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystemRef, LinearCombination as ArkLc, SynthesisError, Variable,
};
use ark_snark::SNARK;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;

use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{Circuit, Fr, LinearCombination, ZkEmitter};
use y::zk_witness::{solve_r1cs_witness, execute_host_witness_ir, check_r1cs_satisfiability};

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

#[derive(Clone)]
struct YCircuit {
    circuit: Circuit,
    witness: Vec<Fr>,
    public_wires: Vec<usize>,
}

impl ConstraintSynthesizer<ArkFr> for YCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<ArkFr>) -> Result<(), SynthesisError> {
        let n = self.circuit.num_variables;
        let mut vars = vec![Variable::One; n];
        for &wire in &self.public_wires {
            let v = self.witness[wire].clone();
            vars[wire] = cs.new_input_variable(|| Ok(to_ark(&v)))?;
        }
        let is_public = |w: usize| self.public_wires.contains(&w);
        for wire in 1..n {
            if is_public(wire) {
                continue;
            }
            let v = self.witness[wire].clone();
            vars[wire] = cs.new_witness_variable(|| Ok(to_ark(&v)))?;
        }
        for c in &self.circuit.constraints {
            cs.enforce_constraint(ark_lc(&c.a, &vars), ark_lc(&c.b, &vars), ark_lc(&c.c, &vars))?;
        }
        Ok(())
    }
}

fn run(n: u32) {
    // The emitter reads this at emit time; 10_000 is the default soft guard.
    std::env::set_var("Y_ZK_MAX_UNROLL", "40000000");

    let src = format!(
        "@unsafe\nfn main(x: I32, y: I32) -> I32 {{\n    let mut temp = x;\n    for i in 0..{} {{\n        temp = temp * y;\n    }}\n    return temp;\n}}\n",
        n
    );

    let t = Instant::now();
    let tokens = Lexer::new(&src).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    TypeChecker::new().check_program(&program);
    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&program).expect("lower");
    let circuit = emitter.build_circuit();
    let witness_ir = emitter.build_witness_ir();
    let emit_s = t.elapsed().as_secs_f64();

    // split the two halves so we can see which one costs
    let t = Instant::now();
    let fwd = execute_host_witness_ir(&witness_ir, &[], &[Fr::from_u64(3), Fr::from_u64(2)]).unwrap();
    let fwd_s = t.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let fwd_ok = check_r1cs_satisfiability(&circuit.constraints, &fwd).is_ok();
    let check_s = t2.elapsed().as_secs_f64();
    println!("  SPLIT forward={:.2}s satisfiability_check={:.2}s forward_already_satisfies={}", fwd_s, check_s, fwd_ok);

    let t = Instant::now();
    let (witness, ok) = solve_r1cs_witness(
        &circuit.constraints,
        &witness_ir,
        circuit.num_variables,
        &[],
        &[Fr::from_u64(3), Fr::from_u64(2)],
    );
    let witness_s = t.elapsed().as_secs_f64();
    assert!(ok, "witness does not satisfy the circuit");

    let n_con = circuit.constraints.len();
    let mut public_wires = circuit.public_inputs.clone();
    for o in &circuit.outputs {
        if !public_wires.contains(o) {
            public_wires.push(*o);
        }
    }
    public_wires.retain(|w| *w != 0);
    let c = YCircuit { circuit, witness, public_wires };
    let public: Vec<ArkFr> = c.public_wires.iter().map(|w| to_ark(&c.witness[*w])).collect();

    let mut rng = StdRng::seed_from_u64(0);
    let t = Instant::now();
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).expect("setup");
    let setup_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let proof = Groth16::<Bn254>::prove(&pk, c, &mut rng).expect("prove");
    let prove_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    assert!(Groth16::<Bn254>::verify(&vk, &public, &proof).expect("verify"));
    let verify_s = t.elapsed().as_secs_f64();

    println!(
        "RESULT n={} constraints={} emit={:.2}s witness={:.2}s setup={:.2}s prove={:.2}s verify={:.4}s total_prove_path={:.2}s",
        n, n_con, emit_s, witness_s, setup_s, prove_s, verify_s,
        emit_s + witness_s + setup_s + prove_s
    );
}

#[test]
#[ignore]
fn groth16_scale_10k() {
    run(10_000);
}

#[test]
#[ignore]
fn groth16_scale_25k() {
    run(25_000);
}

#[test]
#[ignore]
fn groth16_scale_50k() {
    run(50_000);
}

#[test]
#[ignore]
fn groth16_scale_100k() {
    run(100_000);
}

#[test]
#[ignore]
fn groth16_scale_1m() {
    run(1_000_000);
}
