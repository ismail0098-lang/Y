//! End-to-end through the crate's PUBLIC API.
//!
//! The acceptance criterion is the strongest one available: the proof must be
//! identical, element for element, to the one arkworks' own prover produces
//! from the same `r` and `s`. Groth16 is deterministic given its randomness,
//! so `verify` returning true is a weaker statement — it can accept a proof
//! whose MSMs are wrong in compensating ways. This cannot.

use ark_bn254::{Bn254, Fr};
use ark_ff::UniformRand;
use ark_groth16::Groth16;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, OptimizationGoal,
    SynthesisError, Variable,
};
use ark_snark::SNARK;
use ark_std::rand::{rngs::StdRng, SeedableRng};

use y_gpu::groth16::GpuProver;

/// `out = x * y^n`, plus a widening term so the matrices are not all one term
/// wide — the sparse-circuit trap that made an earlier prover benchmark
/// unrepresentative.
#[derive(Clone)]
struct Chain {
    x: Fr,
    y: Fr,
    n: usize,
    dense: bool,
}

impl ConstraintSynthesizer<Fr> for Chain {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        let mut acc = self.x;
        let xv = cs.new_input_variable(|| Ok(self.x))?;
        let yv = cs.new_witness_variable(|| Ok(self.y))?;
        let mut cur = xv;
        let mut hist = vec![xv];
        let mut vals = vec![self.x];
        for _ in 0..self.n {
            let next_val = acc * self.y;
            let next = cs.new_witness_variable(|| Ok(next_val))?;
            if self.dense {
                // A wide `A` row: sum of every previous wire times the current
                // one. Keeps rows growing so the matrices are genuinely dense.
                let mut lc = ark_relations::lc!();
                for h in hist.iter().rev().take(8) {
                    lc = lc + (Fr::from(1u64), *h);
                }
                // (sum of the last 8 wires) * y = t. The value has to be the
                // real sum of those wires, not the latest one repeated.
                let t_val: Fr = vals.iter().rev().take(8).copied().sum::<Fr>() * self.y;
                let t = cs.new_witness_variable(|| Ok(t_val))?;
                cs.enforce_constraint(lc, ark_relations::lc!() + yv, ark_relations::lc!() + t)?;
            }
            cs.enforce_constraint(
                ark_relations::lc!() + cur,
                ark_relations::lc!() + yv,
                ark_relations::lc!() + next,
            )?;
            hist.push(next);
            vals.push(next_val);
            cur = next;
            acc = next_val;
        }
        Ok(())
    }
}

fn run(n: usize, dense: bool) {
    let Some(prover) = GpuProver::new().expect("prover construction failed") else {
        eprintln!("SKIP: no CUDA device.");
        return;
    };

    let circuit = Chain { x: Fr::from(3u64), y: Fr::from(2u64), n, dense };
    let mut rng = StdRng::seed_from_u64(0x9E3779B9);
    let (pk, vk) =
        Groth16::<Bn254>::circuit_specific_setup(circuit.clone(), &mut rng).expect("setup");
    let (r, s) = (Fr::rand(&mut rng), Fr::rand(&mut rng));

    // Matrices + assignment, the way a caller would obtain them.
    let cs = ConstraintSystem::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);
    circuit.clone().generate_constraints(cs.clone()).expect("synthesis");
    assert!(cs.is_satisfied().unwrap(), "circuit is not satisfied");
    cs.finalize();
    let matrices = cs.to_matrices().expect("matrices");
    let full: Vec<Fr> = {
        let p = cs.borrow().unwrap();
        [p.instance_assignment.clone(), p.witness_assignment.clone()].concat()
    };
    let public: Vec<Fr> = {
        let p = cs.borrow().unwrap();
        p.instance_assignment[1..].to_vec()
    };

    // FORCED onto the GPU. These circuits are 2,048 and 4,096 constraints,
    // both far below `MSM_GPU_MIN_STAGED`, so plain `prepare` sends every MSM
    // to the CPU — and this test would then pass with the entire GPU MSM path
    // deleted. It did, for as long as this crate has existed: the binner and
    // the kernels it ships had drifted several optimisations behind the ones
    // under test in the root suite, and nothing here could see it because
    // nothing here ran them.
    let key = prover.prepare_forcing_gpu(&pk, &matrices).expect("prepare");
    let on_gpu = key.gpu_queries();
    assert!(
        on_gpu.len() >= 3,
        "n={n} dense={dense}: only {on_gpu:?} went to the GPU, so this test is \
         mostly checking the CPU fallback"
    );

    let proof = prover
        .prove(&pk, &key, &matrices, &full, r, s)
        .expect("prove");

    assert!(
        Groth16::<Bn254>::verify(&vk, &public, &proof).expect("verify"),
        "n={n} dense={dense}: the GPU proof does not verify"
    );

    let want = Groth16::<Bn254>::create_proof_with_reduction(circuit, &pk, r, s)
        .expect("reference prove");
    assert_eq!(proof.a, want.a, "n={n} dense={dense}: proof element A differs");
    assert_eq!(proof.b, want.b, "n={n} dense={dense}: proof element B differs");
    assert_eq!(proof.c, want.c, "n={n} dense={dense}: proof element C differs");
}

#[test]
fn gpu_proof_matches_arkworks_sparse() {
    run(4096, false);
}

/// The dense case matters on its own: wide constraint rows change the sparse
/// matvec, the number of live bases, and which MSMs clear the dispatch
/// threshold.
#[test]
fn gpu_proof_matches_arkworks_dense() {
    run(2048, true);
}

#[test]
fn a_tampered_statement_is_rejected() {
    let Some(prover) = GpuProver::new().expect("prover construction failed") else {
        eprintln!("SKIP: no CUDA device.");
        return;
    };
    let circuit = Chain { x: Fr::from(5u64), y: Fr::from(7u64), n: 2048, dense: false };
    let mut rng = StdRng::seed_from_u64(11);
    let (pk, vk) =
        Groth16::<Bn254>::circuit_specific_setup(circuit.clone(), &mut rng).expect("setup");
    let (r, s) = (Fr::rand(&mut rng), Fr::rand(&mut rng));

    let cs = ConstraintSystem::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);
    circuit.generate_constraints(cs.clone()).unwrap();
    cs.finalize();
    let matrices = cs.to_matrices().unwrap();
    let full: Vec<Fr> = {
        let p = cs.borrow().unwrap();
        [p.instance_assignment.clone(), p.witness_assignment.clone()].concat()
    };
    let mut public: Vec<Fr> = {
        let p = cs.borrow().unwrap();
        p.instance_assignment[1..].to_vec()
    };

    let key = prover.prepare(&pk, &matrices).unwrap();
    let proof = prover.prove(&pk, &key, &matrices, &full, r, s).unwrap();
    assert!(Groth16::<Bn254>::verify(&vk, &public, &proof).unwrap());
    public[0] += Fr::from(1u64);
    assert!(
        !Groth16::<Bn254>::verify(&vk, &public, &proof).unwrap(),
        "a tampered public input was accepted"
    );
}

/// A bad assignment length must be an ERROR, not a panic and not a wrong
/// proof. This is the difference between a library and test code.
#[test]
fn a_wrong_assignment_length_is_an_error_not_a_panic() {
    let Some(prover) = GpuProver::new().expect("prover construction failed") else {
        eprintln!("SKIP: no CUDA device.");
        return;
    };
    let circuit = Chain { x: Fr::from(3u64), y: Fr::from(2u64), n: 512, dense: false };
    let mut rng = StdRng::seed_from_u64(3);
    let (pk, _) = Groth16::<Bn254>::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
    let cs = ConstraintSystem::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);
    circuit.generate_constraints(cs.clone()).unwrap();
    cs.finalize();
    let matrices = cs.to_matrices().unwrap();
    let key = prover.prepare(&pk, &matrices).unwrap();

    let err = prover
        .prove(&pk, &key, &matrices, &[Fr::from(1u64); 3], Fr::from(1u64), Fr::from(2u64))
        .expect_err("a short assignment should be rejected");
    assert!(
        matches!(err, y_gpu::Error::Invalid(_)),
        "expected Error::Invalid, got {err:?}"
    );
}
