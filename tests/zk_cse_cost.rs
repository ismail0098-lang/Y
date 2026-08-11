//! What common-subexpression elimination costs, and what it buys.
//!
//! The two are wildly different shapes and neither was measurable before. On a
//! Poseidon chain `optimize_circuit` is **33–40% of compile time** and removes
//! **1.26% of the constraints**; on the sparse dot-product circuit it removes
//! nothing at all and still costs. Stated that way it sounds like a bad trade.
//!
//! It is not, and the reason is the workflow rather than the numbers: a ZK
//! circuit is compiled **once** and proved **many** times, so a permanent 1.26%
//! off every future proof overtakes a one-off 38% of a single compile after
//! enough proofs. This test measures where that crossover actually is instead of
//! asserting it — `Y_ZK_CSE=off` / `set_cse_enabled` exist so the question can
//! be asked at all.
//!
//! ```bash
//! cargo test --release --features zk --test zk_cse_cost -- --ignored --nocapture
//! ```
#![cfg(feature = "zk")]

use std::time::Instant;

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
use y::zk_emitter::{set_cse_enabled, Circuit, Fr, LinearCombination, ZkEmitter};
use y::zk_witness::solve_r1cs_witness;

const HASHES: usize = 100;

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
struct C {
    circuit: Circuit,
    witness: Vec<Fr>,
    public: Vec<usize>,
}

impl ConstraintSynthesizer<ArkFr> for C {
    fn generate_constraints(self, cs: ConstraintSystemRef<ArkFr>) -> Result<(), SynthesisError> {
        let n = self.circuit.num_variables;
        let mut vars = vec![Variable::One; n];
        for &w in &self.public {
            let v = self.witness[w];
            vars[w] = cs.new_input_variable(|| Ok(to_ark(&v)))?;
        }
        for w in 1..n {
            if self.public.contains(&w) {
                continue;
            }
            let v = self.witness[w];
            vars[w] = cs.new_witness_variable(|| Ok(to_ark(&v)))?;
        }
        for c in &self.circuit.constraints {
            cs.enforce_constraint(ark_lc(&c.a, &vars), ark_lc(&c.b, &vars), ark_lc(&c.c, &vars))?;
        }
        Ok(())
    }
}

fn compile(cse: bool) -> (f64, Circuit, Vec<Fr>) {
    let src = format!(
        "@unsafe\nfn main(x: I32, y: I32) -> I32 {{\n    let mut h = x;\n\
         \x20   for i in 0..{} {{ h = poseidon_hash(h, y); }}\n    return h;\n}}\n",
        HASHES
    );
    std::env::set_var("Y_ZK_MAX_UNROLL", "40000000");
    let t = Instant::now();
    let tokens = Lexer::new(&src).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    TypeChecker::new().check_program(&program);
    set_cse_enabled(cse);
    let mut emitter = ZkEmitter::new();
    // `ZkEmitter::new` re-reads `Y_ZK_CSE`, so set the thread-local after it.
    set_cse_enabled(cse);
    emitter.emit_program(&program).expect("lower");
    let circuit = emitter.build_circuit();
    let compile_s = t.elapsed().as_secs_f64();
    set_cse_enabled(true);

    let ir = emitter.build_witness_ir();
    let (witness, sat) = solve_r1cs_witness(
        &circuit.constraints,
        &ir,
        circuit.num_variables,
        &[],
        &[Fr::from_u64(3), Fr::from_u64(7)],
    );
    assert!(sat, "cse={}: witness does not satisfy the circuit", cse);
    (compile_s, circuit, witness)
}

/// Both circuits must prove, and the deduplicated one must genuinely be smaller.
///
/// This is the correctness half and runs by default: CSE merges constraints that
/// are already identical, so switching it off must change the constraint count
/// and nothing else observable.
#[test]
fn cse_off_still_produces_a_provable_circuit() {
    let (_, on, on_w) = compile(true);
    let (_, off, off_w) = compile(false);
    assert!(
        off.constraints.len() > on.constraints.len(),
        "CSE removed nothing on a circuit it should reduce: {} vs {}",
        off.constraints.len(),
        on.constraints.len()
    );

    // Same interface either way - CSE must not touch the boundary.
    //
    // Compared by NAME, not by wire id. The ids stopped being comparable when
    // wire compaction landed: CSE abandons the loser of every merged pair, so
    // the two settings leave different wires dead and the renumbering that
    // follows differs by exactly that count. Names are allocated during
    // emission, before either pass runs, so they are the stable identity here -
    // and the values below are what the assertion was really about anyway.
    let names = |c: &Circuit, ws: &[usize]| -> Vec<String> {
        ws.iter().map(|w| c.variables[*w].clone()).collect()
    };
    assert_eq!(names(&on, &on.public_inputs), names(&off, &off.public_inputs));
    assert_eq!(names(&on, &on.private_inputs), names(&off, &off.private_inputs));
    assert_eq!(names(&on, &on.outputs), names(&off, &off.outputs));

    // And the boundary must carry the same values, which is the property the
    // wire-id comparison was standing in for.
    for (i, (&a, &b)) in on.outputs.iter().zip(off.outputs.iter()).enumerate() {
        assert_eq!(
            on_w[a].to_decimal_string(),
            off_w[b].to_decimal_string(),
            "CSE changed output {}",
            i
        );
    }
}

#[test]
#[ignore]
fn what_cse_costs_and_what_it_buys() {
    let mut rows = Vec::new();
    for (label, cse) in [("CSE off", false), ("CSE on ", true)] {
        let (compile_s, circuit, witness) = compile(cse);
        let n_con = circuit.constraints.len();
        let public = circuit.outputs.clone();
        let c = C { circuit, witness: witness.clone(), public: public.clone() };

        let mut rng = StdRng::seed_from_u64(7);
        let t = Instant::now();
        let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).unwrap();
        let setup_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let proof = Groth16::<Bn254>::prove(&pk, c, &mut rng).unwrap();
        let prove_s = t.elapsed().as_secs_f64();
        let inputs: Vec<ArkFr> = public.iter().map(|w| to_ark(&witness[*w])).collect();
        assert!(Groth16::<Bn254>::verify(&vk, &inputs, &proof).unwrap());

        eprintln!(
            "{}  {:>7} constraints   compile {:>6.3} s   setup {:>6.3} s   prove {:>6.3} s",
            label, n_con, compile_s, setup_s, prove_s
        );
        rows.push((compile_s, prove_s, n_con));
    }

    let (off_compile, off_prove, off_n) = rows[0];
    let (on_compile, on_prove, on_n) = rows[1];
    let compile_cost = on_compile - off_compile;
    let prove_saving = off_prove - on_prove;
    eprintln!(
        "\nCSE costs {:.3} s of compile and removes {} constraints ({:.2}%), \
         saving {:.4} s per proof.",
        compile_cost,
        off_n - on_n,
        (off_n - on_n) as f64 / off_n as f64 * 100.0,
        prove_saving
    );
    if prove_saving > 0.0 {
        eprintln!(
            "Break-even: {:.0} proofs. Compile once, prove many -- so it stays on by default.",
            compile_cost / prove_saving
        );
    } else {
        eprintln!(
            "Prove time did not improve measurably at this size; the crossover is \
             further out than one run can resolve."
        );
    }
}
