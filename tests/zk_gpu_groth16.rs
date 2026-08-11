//! A Groth16 prover whose G1 multi-scalar multiplications run on the GPU,
//! over a circuit Y itself compiled.
//!
//! This is the step that connects the two halves of the repo. Until now the ZK
//! backend emitted R1CS and arkworks proved it entirely on the CPU, while the
//! GPU kernels in `tools/gen_bn254_kernels.py` were checked against arkworks in
//! isolation. Here Y's emitter produces the circuit, Y's witness solver fills
//! it in, and Y's GPU MSM computes four of the five multi-scalar
//! multiplications a Groth16 proof is made of.
//!
//! **The acceptance criterion is arkworks' own verifier, and one thing
//! stronger.** `Groth16::verify` accepting the proof would already be hard to
//! fake — a wrong MSM gives a valid-looking curve point that fails the pairing
//! check. But the proof is also compared ELEMENT FOR ELEMENT against the one
//! arkworks' prover produces from the same `r` and `s`, which is exact:
//! Groth16 is deterministic given its randomness, so `A`, `B` and `C` must
//! match bit for bit. That distinguishes "the MSMs are right" from "the MSMs
//! are wrong in a way that happens to still verify", which is not a
//! distinction `verify` alone can make.
//!
//! `B` in G2 stays on the CPU. G2 coordinates are `Fq2`, a quadratic extension
//! this kernel series has no arithmetic for yet, and claiming otherwise by
//! quietly leaving it out is exactly the failure mode `CLAUDE.md` is full of.
//! It is named here, measured separately below, and it is the reason the
//! end-to-end prover speedup is smaller than the MSM speedup.

#[path = "common/msm.rs"]
mod msm;
use msm::*;

use ark_bn254::{Bn254, Fr as ArkFr, G1Affine, G1Projective};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, Zero};
use ark_groth16::{r1cs_to_qap::{LibsnarkReduction, R1CSToQAP}, Groth16, Proof, ProvingKey};
use ark_poly::GeneralEvaluationDomain;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef,
    LinearCombination as ArkLc, OptimizationGoal, SynthesisError, Variable,
};
use ark_snark::SNARK;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use ark_std::UniformRand;

use y::cuda_runtime::{CudaContext, KernelModule};
use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{Circuit, Fr, LinearCombination, ZkEmitter};
use y::zk_witness::solve_r1cs_witness;

// ---------------------------------------------------------------------------
// Y's circuit, replayed into arkworks (same shape as zk_groth16_end_to_end.rs)
// ---------------------------------------------------------------------------

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
    (circuit, witness)
}

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

    fn public_values(&self) -> Vec<ArkFr> {
        self.public_wires.iter().map(|w| to_ark(&self.witness[*w])).collect()
    }
}

impl ConstraintSynthesizer<ArkFr> for YCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<ArkFr>) -> Result<(), SynthesisError> {
        let n = self.circuit.num_variables;
        let mut vars = vec![Variable::One; n];
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

// ---------------------------------------------------------------------------
// The GPU MSM, adapted to a proving key
// ---------------------------------------------------------------------------

/// `add-2007-bl` cannot represent the point at infinity, and a proving key is
/// FULL of them: `a_query[i]` is `A_i(tau) * G`, and `A_i` is the zero
/// polynomial for every variable that never appears in the `A` matrix — which
/// in any real circuit is most of them.
///
/// Dropping those pairs is exact rather than approximate: a term whose base is
/// the identity contributes nothing to the sum whatever its scalar is. The
/// same goes for a zero scalar, and zero scalars are the common case in a
/// sparse witness. Both filters shrink the MSM as well as making it legal.
///
/// This is a PRECONDITION of the kernel being made checkable, not a
/// workaround: `gpu_msm` is documented to require non-identity points, and
/// feeding it one produces a wrong bucket rather than an error.
fn filter_msm_terms(bases: &[G1Affine], scalars: &[ArkFr]) -> (Vec<G1Projective>, Vec<ArkFr>) {
    let n = bases.len().min(scalars.len());
    let mut p = Vec::with_capacity(n);
    let mut s = Vec::with_capacity(n);
    for i in 0..n {
        if bases[i].is_zero() || scalars[i].is_zero() {
            continue;
        }
        p.push(bases[i].into_group());
        s.push(scalars[i]);
    }
    (p, s)
}

/// One MSM on the GPU, falling back to nothing — an empty term list is the
/// identity, which is the correct answer and not a special case.
fn msm_gpu(
    ctx: &CudaContext,
    module: &KernelModule,
    g: &Geom,
    bases: &[G1Affine],
    scalars: &[ArkFr],
) -> G1Projective {
    let (p, s) = filter_msm_terms(bases, scalars);
    if p.is_empty() {
        return G1Projective::zero();
    }
    gpu_msm(ctx, module, &p, &s, g).0
}

/// `Groth16::calculate_coeff`, with the MSM on the GPU.
///
/// `query[0]` is the coefficient of the constant wire, whose assignment is
/// always 1, so it is added rather than multiplied — the MSM covers
/// `query[1..]`.
fn calculate_coeff_gpu(
    ctx: &CudaContext,
    module: &KernelModule,
    g: &Geom,
    initial: G1Projective,
    query: &[G1Affine],
    vk_param: G1Affine,
    assignment: &[ArkFr],
) -> G1Projective {
    let acc = msm_gpu(ctx, module, g, &query[1..], assignment);
    let mut res = initial;
    res += query[0].into_group();
    res += acc;
    res += vk_param.into_group();
    res
}

#[derive(Default)]
struct ProveTiming {
    g1_msms: f64,
    g2_msm: f64,
    total: f64,
}

/// Groth16 `create_proof_with_assignment`, with all four G1 MSMs on the GPU.
///
/// Deliberately a transcription of arkworks' own prover rather than a
/// reformulation: the point is to substitute ONE component and hold everything
/// else identical, so that an element-for-element comparison against arkworks
/// is a test of that component and nothing else.
#[allow(clippy::too_many_arguments)]
fn gpu_prove(
    ctx: &CudaContext,
    module: &KernelModule,
    g: &Geom,
    pk: &ProvingKey<Bn254>,
    r: ArkFr,
    s: ArkFr,
    h: &[ArkFr],
    input_assignment: &[ArkFr],
    aux_assignment: &[ArkFr],
    tm: &mut ProveTiming,
) -> Proof<Bn254> {
    use std::time::Instant;
    let t0 = Instant::now();

    let t = Instant::now();
    let h_acc = msm_gpu(ctx, module, g, &pk.h_query, h);
    let l_aux_acc = msm_gpu(ctx, module, g, &pk.l_query, aux_assignment);

    let assignment: Vec<ArkFr> =
        [input_assignment, aux_assignment].concat();

    let r_s_delta_g1 = pk.delta_g1.into_group() * (r * s);

    let g_a = calculate_coeff_gpu(
        ctx, module, g,
        pk.delta_g1.into_group() * r,
        &pk.a_query,
        pk.vk.alpha_g1,
        &assignment,
    );
    let s_g_a = g_a * s;

    let g1_b = if !r.is_zero() {
        calculate_coeff_gpu(
            ctx, module, g,
            pk.delta_g1.into_group() * s,
            &pk.b_g1_query,
            pk.beta_g1,
            &assignment,
        )
    } else {
        G1Projective::zero()
    };
    tm.g1_msms = t.elapsed().as_secs_f64();

    // B in G2 stays on the CPU: Fq2 arithmetic is not in this kernel series.
    let t = Instant::now();
    let assignment_bi: Vec<_> = assignment.iter().map(|x| x.into_bigint()).collect();
    let g2_b = {
        let acc = <Bn254 as Pairing>::G2::msm_bigint(&pk.b_g2_query[1..], &assignment_bi);
        let mut res = pk.vk.delta_g2.into_group() * s;
        res += pk.b_g2_query[0].into_group();
        res += acc;
        res += pk.vk.beta_g2.into_group();
        res
    };
    tm.g2_msm = t.elapsed().as_secs_f64();

    let r_g1_b = g1_b * r;

    let mut g_c = s_g_a;
    g_c += r_g1_b;
    g_c -= r_s_delta_g1;
    g_c += l_aux_acc;
    g_c += h_acc;

    tm.total = t0.elapsed().as_secs_f64();
    Proof {
        a: g_a.into_affine(),
        b: g2_b.into_affine(),
        c: g_c.into_affine(),
    }
}

/// Runs the circuit through the constraint system exactly as arkworks' prover
/// does, and hands back the pieces its `create_proof_with_assignment` takes.
///
/// `check` runs `is_satisfied`, which arkworks' prover only does under
/// `debug_assert`. It is off on the measured path: including a check the
/// baseline does not run would bias the comparison the other way, and the
/// point of measuring is to be able to believe the number afterwards.
fn synthesize(circuit: YCircuit, check: bool) -> (Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>) {
    synthesize_timed(circuit, check).0
}

/// Same, reporting how the pre-MSM time splits between replaying the circuit
/// into arkworks' constraint system and the QAP witness map. The split decides
/// what is worth accelerating next: the witness map is an FFT over the
/// constraint matrices, which is what this repo's GPU NTT exists for, whereas
/// constraint replay is an artifact of handing Y's R1CS to arkworks at all.
#[allow(clippy::type_complexity)]
fn synthesize_timed(
    circuit: YCircuit,
    check: bool,
) -> ((Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>), f64, f64) {
    use std::time::Instant;
    let t = Instant::now();
    let cs = ConstraintSystem::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);
    circuit.generate_constraints(cs.clone()).expect("synthesis");
    if check {
        assert!(cs.is_satisfied().unwrap(), "the circuit is not satisfied");
    }
    cs.finalize();
    let replay = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let h = LibsnarkReduction::witness_map::<ArkFr, GeneralEvaluationDomain<ArkFr>>(cs.clone())
        .expect("QAP witness map");
    let qap = t.elapsed().as_secs_f64();
    let prover = cs.borrow().unwrap();
    (
        (
            h,
            prover.instance_assignment[1..].to_vec(),
            prover.witness_assignment.clone(),
        ),
        replay,
        qap,
    )
}

/// A Y circuit with `a * b` multiplications, so the MSMs are big enough to
/// matter. Nested rather than one flat loop because the emitter's unroll guard
/// is 10,000 iterations PER LOOP and is env-var-only — and mutating the
/// environment from a test races the rest of the suite.
fn poly_src(a: usize, b: usize) -> String {
    format!(
        r#"
@unsafe
fn main(x: I32, y: I32) -> I32 {{
    let mut temp = x;
    for i in 0..{} {{
        for j in 0..{} {{
            temp = temp * y;
        }}
    }}
    return temp;
}}
"#,
        a, b
    )
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

/// The integration, end to end: Y compiles the circuit, Y solves the witness,
/// Y's GPU MSM computes the G1 proof elements, and arkworks verifies.
#[test]
fn gpu_groth16_proof_verifies_and_matches_arkworks() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the GPU Groth16 prover was not executed.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_msm_bucket");
    let g = Geom::new(28);

    let (circuit, witness) = compile(&poly_src(64, 64), &[], &[3, 2]);
    let c = YCircuit::new(circuit, witness);
    let public = c.public_values();

    let mut rng = StdRng::seed_from_u64(0xF00D);
    let (pk, vk) =
        Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).expect("setup");

    // Fixed r and s so the two provers are comparable at all. Groth16 is
    // deterministic given them.
    let mut rng2 = StdRng::seed_from_u64(0xC0DE);
    let (r, s) = (ArkFr::rand(&mut rng2), ArkFr::rand(&mut rng2));

    let (h, input_assignment, aux_assignment) = synthesize(c.clone(), true);
    let mut tm = ProveTiming::default();
    let proof = gpu_prove(
        &ctx, &module, &g, &pk, r, s, &h, &input_assignment, &aux_assignment, &mut tm,
    );

    assert!(
        Groth16::<Bn254>::verify(&vk, &public, &proof).expect("verify"),
        "the GPU-produced Groth16 proof does not verify"
    );

    // The stronger statement: identical to arkworks' own proof, element for
    // element. `verify` can accept a proof whose MSMs are wrong in compensating
    // ways; this cannot.
    let reference =
        Groth16::<Bn254>::create_proof_with_reduction(c, &pk, r, s).expect("reference prove");
    assert_eq!(proof.a, reference.a, "proof element A differs from arkworks'");
    assert_eq!(proof.b, reference.b, "proof element B differs from arkworks'");
    assert_eq!(proof.c, reference.c, "proof element C differs from arkworks'");
}

/// A tampered public input must be rejected. Without this, a prover that
/// produced some fixed valid-looking proof would pass the test above.
#[test]
fn gpu_groth16_rejects_a_tampered_statement() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_msm_bucket");
    let g = Geom::new(28);

    let (circuit, witness) = compile(&poly_src(16, 16), &[], &[3, 2]);
    let c = YCircuit::new(circuit, witness);
    let public = c.public_values();
    assert!(!public.is_empty(), "circuit exposes nothing public");

    let mut rng = StdRng::seed_from_u64(7);
    let (pk, vk) =
        Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).expect("setup");
    let (r, s) = (ArkFr::rand(&mut rng), ArkFr::rand(&mut rng));
    let (h, input_assignment, aux_assignment) = synthesize(c, true);
    let mut tm = ProveTiming::default();
    let proof = gpu_prove(
        &ctx, &module, &g, &pk, r, s, &h, &input_assignment, &aux_assignment, &mut tm,
    );

    assert!(Groth16::<Bn254>::verify(&vk, &public, &proof).unwrap());
    let mut tampered = public.clone();
    *tampered.last_mut().unwrap() += ArkFr::from(1u64);
    assert!(
        !Groth16::<Bn254>::verify(&vk, &tampered, &proof).unwrap(),
        "a tampered public input was accepted"
    );
}

/// `cargo test --release --features zk --test zk_gpu_groth16 -- --ignored --nocapture`
///
/// What the substitution is worth on a real proof, and — as importantly —
/// what it is NOT worth, because the G2 MSM and the QAP step do not move.
#[test]
#[ignore]
fn what_the_gpu_prover_costs() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_msm_bucket");
    let g = Geom::new(28);

    for (a, b) in [(256usize, 256usize), (512, 512)] {
        let (circuit, witness) = compile(&poly_src(a, b), &[], &[3, 2]);
        let nc = circuit.constraints.len();
        let c = YCircuit::new(circuit, witness);
        let public = c.public_values();

        let mut rng = StdRng::seed_from_u64(0xF00D);
        let (pk, vk) =
            Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).expect("setup");
        let (r, s) = (ArkFr::rand(&mut rng), ArkFr::rand(&mut rng));
        // Warm up the GPU clock and the JIT before timing anything.
        {
            let (h, ia, aa) = synthesize(c.clone(), true);
            let mut tm = ProveTiming::default();
            let _ = gpu_prove(&ctx, &module, &g, &pk, r, s, &h, &ia, &aa, &mut tm);
        }

        // Synthesis + the QAP witness map are part of proving and arkworks'
        // `create_proof_with_reduction` does them inside the call being timed,
        // so they are inside the GPU prover's number too. Timing only the MSMs
        // against a baseline that also does the QAP reported 2.80x and 5.94x;
        // those figures were wrong and are what this restructuring exists to
        // correct.
        let mut best = f64::MAX;
        let mut best_tm = ProveTiming::default();
        let mut best_synth = 0.0;
        let (mut best_replay, mut best_qap) = (0.0, 0.0);
        for _ in 0..3 {
            let ((h, ia, aa), replay, qap) = synthesize_timed(c.clone(), false);
            let synth = replay + qap;
            let mut tm = ProveTiming::default();
            let p = gpu_prove(&ctx, &module, &g, &pk, r, s, &h, &ia, &aa, &mut tm);
            assert!(Groth16::<Bn254>::verify(&vk, &public, &p).unwrap());
            if synth + tm.total < best {
                best = synth + tm.total;
                best_synth = synth;
                best_replay = replay;
                best_qap = qap;
                best_tm = tm;
            }
        }

        let mut cpu = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let p = Groth16::<Bn254>::create_proof_with_reduction(c.clone(), &pk, r, s)
                .expect("reference prove");
            let e = t.elapsed().as_secs_f64();
            assert!(Groth16::<Bn254>::verify(&vk, &public, &p).unwrap());
            cpu = cpu.min(e);
        }

        println!("\n=== {} constraints ({} a_query terms) ===", nc, pk.a_query.len());
        println!("  R1CS replay          = {:8.1} ms   (identical work on both sides)", best_replay * 1e3);
        println!("  QAP witness map      = {:8.1} ms   (an FFT -- the GPU NTT's target)", best_qap * 1e3);
        println!("  G1 MSMs, on the GPU  = {:8.1} ms", best_tm.g1_msms * 1e3);
        println!("  G2 MSM, on the CPU   = {:8.1} ms   (no Fq2 kernel yet)", best_tm.g2_msm * 1e3);
        println!("  GPU prover, TOTAL    = {:8.1} ms", best * 1e3);
        println!("  arkworks prove       = {:8.1} ms", cpu * 1e3);
        println!("  speedup on the prove = {:8.2}x", cpu / best);
        println!(
            "  where the GPU prover's time goes: {:.0}% synthesis+QAP, {:.0}% G1 MSM, {:.0}% G2 MSM",
            100.0 * best_synth / best,
            100.0 * best_tm.g1_msms / best,
            100.0 * best_tm.g2_msm / best
        );
    }
}
