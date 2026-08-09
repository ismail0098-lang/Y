//! The linear-substitution pass: `optimize_circuit`'s second reduction.
//!
//! A constraint of the form `k * L = c_w * w`, with `k` constant and `w` an
//! intermediate wire, *defines* `w`. Substituting `w` away and deleting the
//! constraint is equisatisfiable, because `w` is neither an input nor an output
//! and no verifier ever sees it. Every `out <== in` circom writes is of exactly
//! that shape, which is why Y's circom front end emitted 765 constraints for
//! `Poseidon(2)` where circom emits 517.
//!
//! **Deleting constraints is the one kind of optimisation that can silently
//! weaken a proof.** A pass that removes too much still produces an `.r1cs`,
//! still admits a witness, still proves - just a weaker statement than the
//! source describes, with nothing downstream recording the difference. So the
//! tests here are not "did it get smaller"; they are:
//!
//! 1. the reduced circuit computes the same values (pinned against circomlib),
//! 2. the reduced circuit still *binds* its output,
//! 3. an unsatisfiable circuit stays unsatisfiable,
//! 4. the witness the reduced circuit produces satisfies the UNREDUCED one,
//! 5. and a control, so that "delete nothing" cannot pass the file.
//!
//! Run with:  cargo test --features zk --test zk_linear_substitution

#![cfg(feature = "zk")]

use std::path::{Path, PathBuf};
use y::circom_lower::compile_file;
use y::zk_emitter::{set_linsub_budget, Constraint};
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
/// `set_linsub_budget` is thread-local precisely so this can flip between the
/// two settings without an `env::set_var` that would race every other test in
/// the binary.
fn run(name: &str, budget: Option<usize>, inputs: &[u64]) -> Run {
    set_linsub_budget(budget);
    let emitter = compile_file(&fixture(name), &[circomlib()])
        .unwrap_or_else(|e| panic!("{} failed to compile: {}", name, e));
    set_linsub_budget(Some(16));

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

/// The pass must not change what the circuit computes, and the reduced circuit
/// must still admit a witness for the same inputs.
#[test]
fn reduction_preserves_the_computed_value() {
    for (name, inputs) in [
        ("multiplier.circom", &[3u64, 5][..]),
        ("sum_squares.circom", &[1, 2, 3, 4][..]),
        ("linear_chain.circom", &[6][..]),
        ("poseidon2.circom", &[1, 2][..]),
    ] {
        let off = run(name, OFF, inputs);
        let on = run(name, ON, inputs);
        assert!(off.satisfied, "{}: unreduced circuit has no witness", name);
        assert!(on.satisfied, "{}: reduced circuit has no witness", name);
        assert_eq!(
            on.witness[on.out].to_decimal_string(),
            off.witness[off.out].to_decimal_string(),
            "{}: the pass changed the circuit's output",
            name
        );
    }
}

/// **The control.** Without this, a pass that eliminated nothing would satisfy
/// every other test in this file.
#[test]
fn the_pass_actually_removes_constraints() {
    let cases = [
        ("sum_squares.circom", &[1u64, 2, 3, 4][..]),
        ("linear_chain.circom", &[6][..]),
        ("poseidon2.circom", &[1, 2][..]),
    ];
    for (name, inputs) in cases {
        let off = run(name, OFF, inputs).constraints.len();
        let on = run(name, ON, inputs).constraints.len();
        assert!(
            on < off,
            "{}: the pass removed nothing ({} constraints either way)",
            name,
            off
        );
        eprintln!("{:24} {:>6} -> {:>6} constraints", name, off, on);
    }
}

/// The reduced circuit must still force its output.
///
/// This is the failure mode that matters: a pass that deletes the constraint
/// binding `out` leaves a circuit that compiles, proves, and proves nothing
/// about the output. Perturbing the honest witness's output wire has to break
/// satisfiability - if it does not, the output is unconstrained.
#[test]
fn the_reduced_circuit_still_binds_its_output() {
    for (name, inputs) in [
        ("multiplier.circom", &[3u64, 5][..]),
        ("sum_squares.circom", &[1, 2, 3, 4][..]),
        ("linear_chain.circom", &[6][..]),
        ("poseidon2.circom", &[1, 2][..]),
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

/// An impossible circuit must stay impossible.
///
/// `a <== x + 1; a === x + 2;` has no solution. The pass deletes the constraint
/// that defines `a` and substitutes it into the `===`, which collapses to the
/// constant identity `-1 = 0`. Dropping *that* as "trivial" would turn an
/// unsatisfiable statement into a provable one, so `constraint_is_vacuous` keeps
/// non-zero constant identities on purpose.
#[test]
fn a_contradiction_survives_substitution() {
    for budget in [OFF, ON] {
        let r = run("contradiction.circom", budget, &[9]);
        let live = check_r1cs_satisfiability(&r.constraints, &r.witness);
        assert!(
            !r.satisfied || live.is_err(),
            "an unsatisfiable circuit reported a satisfying witness (budget {:?})",
            budget
        );
        assert!(
            !r.constraints.is_empty(),
            "every constraint was optimised away, so nothing is left to fail (budget {:?})",
            budget
        );
    }
}

/// The strongest available check: the witness produced for the REDUCED circuit
/// must satisfy the UNREDUCED one.
///
/// Wire numbering is identical between the two - the pass deletes constraints
/// and rewrites terms, it never allocates or renumbers - so the two witnesses
/// are directly comparable. Each eliminated wire keeps a recipe holding exactly
/// the expression its deleted constraint asserted, which is what makes this hold
/// and is why that recipe is written even though nothing in the reduced circuit
/// reads it.
///
/// A substitution with a wrong coefficient, or a constraint deleted that was not
/// really a definition, fails here even when the output still happens to come
/// out right.
#[test]
fn the_reduced_witness_satisfies_the_original_circuit() {
    for (name, inputs) in [
        ("multiplier.circom", &[3u64, 5][..]),
        ("sum_squares.circom", &[1, 2, 3, 4][..]),
        ("linear_chain.circom", &[6][..]),
        ("poseidon2.circom", &[1, 2][..]),
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

/// `Poseidon(2)` from unmodified circomlib, reduced, still hashes correctly.
///
/// The digests are circomlib's own published vectors, pinned identically in
/// `circom_frontend.rs` and `zk_poseidon_interop.rs`. **Do not update them.**
#[test]
fn reduced_poseidon_still_matches_circomlib() {
    let vectors: &[(u64, u64, &str)] = &[
        (1, 2, "7853200120776062878684798364095072458815029376092732009249414926327459813530"),
        (3, 4, "14763215145315200506921711489642608356394854266165572616578112107564877678998"),
        (0, 0, "14744269619966411208579211824598458697587494354926760081771325075741142829156"),
        (7, 0, "10402197090275139279073177788985849389816807868761640028215734431067655199248"),
    ];
    for (a, b, expected) in vectors {
        let r = run("poseidon2.circom", ON, &[*a, *b]);
        assert!(r.satisfied, "Poseidon({}, {}): no witness", a, b);
        check_r1cs_satisfiability(&r.constraints, &r.witness)
            .unwrap_or_else(|e| panic!("Poseidon({}, {}): {}", a, b, e));
        assert_eq!(
            r.witness[r.out].to_decimal_string(),
            *expected,
            "the reduced Poseidon(2) disagrees with circomlib"
        );
    }
}

/// What the reduction is worth downstream, measured rather than assumed.
///
/// Halving the constraint count does not automatically halve proving time:
/// Groth16's cost also scales with the WIRE count and with the number of
/// non-zero terms, and this pass leaves the eliminated wires in the variable
/// table rather than renumbering every wire in the circuit. So the honest way
/// to state the benefit is to run a real prover over both circuits.
///
/// `#[ignore]`d - it is a benchmark, not a gate. Run with:
///
/// ```bash
/// cargo test --release --features zk --test zk_linear_substitution \
///     -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn what_the_reduction_buys_at_proving_time() {
    use ark_bn254::{Bn254, Fr as ArkFr};
    use ark_ff::PrimeField;
    use ark_groth16::Groth16;
    use ark_relations::r1cs::{
        ConstraintSynthesizer, ConstraintSystemRef, LinearCombination as ArkLc, SynthesisError,
        Variable,
    };
    use ark_snark::SNARK;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use std::time::Instant;
    use y::zk_emitter::{Circuit, LinearCombination};

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
                cs.enforce_constraint(
                    ark_lc(&c.a, &vars),
                    ark_lc(&c.b, &vars),
                    ark_lc(&c.c, &vars),
                )?;
            }
            Ok(())
        }
    }

    for (label, budget) in [("unreduced", OFF), ("reduced  ", ON)] {
        set_linsub_budget(budget);
        let emitter = compile_file(&fixture("poseidon_chain20.circom"), &[circomlib()]).unwrap();
        set_linsub_budget(Some(16));
        let circuit = emitter.build_circuit();
        let ir = emitter.build_witness_ir();
        let (witness, sat) = solve_r1cs_witness(
            &circuit.constraints,
            &ir,
            circuit.num_variables,
            &[],
            &[Fr::from_u64(3), Fr::from_u64(7)],
        );
        assert!(sat, "{}: no witness", label);
        let nnz: usize = circuit
            .constraints
            .iter()
            .map(|c| c.a.terms.len() + c.b.terms.len() + c.c.terms.len())
            .sum();
        let public = circuit.outputs.clone();
        let c = C { circuit: circuit.clone(), witness: witness.clone(), public: public.clone() };

        let mut rng = StdRng::seed_from_u64(7);
        let t = Instant::now();
        let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).unwrap();
        let setup = t.elapsed();
        let t = Instant::now();
        let proof = Groth16::<Bn254>::prove(&pk, c, &mut rng).unwrap();
        let prove = t.elapsed();
        let inputs: Vec<ArkFr> = public.iter().map(|w| to_ark(&witness[*w])).collect();
        assert!(
            Groth16::<Bn254>::verify(&vk, &inputs, &proof).unwrap(),
            "{}: proof does not verify",
            label
        );

        eprintln!(
            "{}  {:>7} constraints  {:>7} wires  {:>8} nnz   setup {:>6.3} s   prove {:>6.3} s",
            label,
            circuit.constraints.len(),
            circuit.num_variables,
            nnz,
            setup.as_secs_f64(),
            prove.as_secs_f64()
        );
    }
}
