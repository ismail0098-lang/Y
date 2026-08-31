//! A Groth16 prover with the G1 MSMs and the QAP witness map on the GPU.
//!
//! Usage is two-phase, mirroring how a prover is actually deployed: a
//! `PreparedKey` is built once per proving key and reused for every proof.
//! That split is not cosmetic — in Groth16 the G1 bases ARE the proving key
//! and the scalars are the witness, so staging the bases across PCIe and
//! building the QAP twiddle tables is key-load work. Paying it per proof would
//! roughly double the cost of a large proof.
//!
//! ```ignore
//! let prover = GpuProver::new()?.expect("no CUDA device");
//! let key = prover.prepare(&pk, &matrices)?;          // once per proving key
//! let proof = prover.prove(&pk, &key, &matrices, &full_assignment, r, s)?;
//! ```

use ark_bn254::{Bn254, Fr, G1Affine, G1Projective};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{FftField, PrimeField, Zero};
use ark_groth16::{Proof, ProvingKey};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::ConstraintMatrices;

use y::cuda_runtime::{CudaContext, KernelModule};

use crate::error::{Error, Result};
use crate::msm::{gpu_is_worth_it, gpu_msm_staged, stage_bases, DeviceBases, Geom};
use crate::qap::GpuQap;

/// Window geometry for a proof's MSMs. See `msm::Geom`.
const MSM_WINDOWS: usize = 28;

/// One proving-key query, staged on the device if an MSM of its size is worth
/// putting there.
pub struct StagedQuery<'a> {
    bases: &'a [G1Affine],
    len: usize,
    dev: Option<DeviceBases>,
    keep: Vec<u32>,
}

impl<'a> StagedQuery<'a> {
    fn new(
        ctx: &CudaContext,
        bases: &'a [G1Affine],
        len: usize,
        force_gpu: bool,
    ) -> Result<Self> {
        // `add-2007-bl` cannot represent the point at infinity, and a proving
        // key is full of them: `a_query[i]` is `A_i(tau) * G`, and `A_i` is the
        // zero polynomial for every variable absent from the `A` matrix.
        // Dropping those pairs is exact, not an approximation — an identity
        // base contributes nothing whatever its scalar is.
        //
        // Only the BASES are filtered. Filtering zero scalars too would shrink
        // the MSM but make the surviving point set witness-dependent, which
        // would forbid staging the bases once per key; and a zero scalar
        // already costs nothing, since all its window digits are 0 and digit 0
        // lands in no bucket.
        let keep: Vec<u32> = (0..bases.len().min(len))
            .filter(|&i| !bases[i].is_zero())
            .map(|i| i as u32)
            .collect();
        let len = len.min(bases.len());
        // `force_gpu` exists for the correctness tests and for nothing else.
        // Dispatch silently disarms them otherwise: this crate's own prover
        // tests run 2,048 and 4,096 constraints, both far below
        // `MSM_GPU_MIN_STAGED`, so every MSM routed to the CPU and the tests
        // would have passed with the GPU path DELETED. It was worse than that
        // in practice — the shipped binner was several optimisations behind
        // the one under test and nothing could see it, because nothing ran it.
        // The same trap is recorded for the root suite, which was fixed for it
        // and this crate was not.
        let dev = if force_gpu || gpu_is_worth_it(keep.len(), true) {
            let pts: Vec<G1Projective> =
                keep.iter().map(|&i| bases[i as usize].into_group()).collect();
            Some(stage_bases(ctx, &pts)?)
        } else {
            None
        };
        Ok(StagedQuery { bases, len, dev, keep })
    }

    /// Whether this query's MSM will run on the GPU.
    pub fn on_gpu(&self) -> bool {
        self.dev.is_some()
    }

    /// Number of live (non-identity) bases.
    pub fn len(&self) -> usize {
        self.keep.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keep.is_empty()
    }
}

/// Everything about a proving key and circuit that does not change per proof.
pub struct PreparedKey<'a> {
    h: StagedQuery<'a>,
    l: StagedQuery<'a>,
    a: StagedQuery<'a>,
    b_g1: StagedQuery<'a>,
    qap: GpuQap,
    domain_size: usize,
}

impl PreparedKey<'_> {
    /// Which of the four G1 queries will run on the GPU, for logging.
    pub fn gpu_queries(&self) -> Vec<&'static str> {
        [("h", &self.h), ("l", &self.l), ("a", &self.a), ("b_g1", &self.b_g1)]
            .iter()
            .filter(|(_, q)| q.on_gpu())
            .map(|(n, _)| *n)
            .collect()
    }

    pub fn domain_size(&self) -> usize {
        self.domain_size
    }
}

/// A GPU prover. Holds the CUDA context and the loaded kernels.
pub struct GpuProver {
    ctx: CudaContext,
    msm_kernel: KernelModule,
    geom: Geom,
}

impl GpuProver {
    /// `Ok(None)` when there is no CUDA driver or device — that is an ordinary
    /// deployment condition, not an error, and the caller should fall back to
    /// a CPU prover.
    pub fn new() -> Result<Option<Self>> {
        let Some(ctx) = CudaContext::new() else {
            return Ok(None);
        };
        let msm_kernel =
            crate::kernels::load(&ctx, "bn254_msm_bucket", crate::kernels::MSM_BUCKET)?;
        Ok(Some(GpuProver { ctx, msm_kernel, geom: Geom::new(MSM_WINDOWS) }))
    }

    /// Stage a proving key onto the device. Do this once per key.
    pub fn prepare<'a>(
        &self,
        pk: &'a ProvingKey<Bn254>,
        matrices: &ConstraintMatrices<Fr>,
    ) -> Result<PreparedKey<'a>> {
        self.prepare_inner(pk, matrices, false)
    }

    /// `prepare`, but every G1 query is staged on the device whatever its
    /// size. For tests that must exercise the GPU path at a circuit size the
    /// dispatcher would send to the CPU — see the note in `StagedQuery::new`.
    /// A caller who wants the fast path should use `prepare` and let the
    /// measured thresholds decide.
    pub fn prepare_forcing_gpu<'a>(
        &self,
        pk: &'a ProvingKey<Bn254>,
        matrices: &ConstraintMatrices<Fr>,
    ) -> Result<PreparedKey<'a>> {
        self.prepare_inner(pk, matrices, true)
    }

    fn prepare_inner<'a>(
        &self,
        pk: &'a ProvingKey<Bn254>,
        matrices: &ConstraintMatrices<Fr>,
        force_gpu: bool,
    ) -> Result<PreparedKey<'a>> {
        let ni = matrices.num_instance_variables;
        let nc = matrices.num_constraints;
        let domain = GeneralEvaluationDomain::<Fr>::new(nc + ni).ok_or_else(|| {
            Error::Unsupported(format!("no evaluation domain for {} constraints", nc))
        })?;
        let domain_size = domain.size();
        let aux_len = matrices.num_witness_variables;
        let assign_len = ni + aux_len - 1;

        Ok(PreparedKey {
            h: StagedQuery::new(&self.ctx, &pk.h_query, domain_size - 1, force_gpu)?,
            l: StagedQuery::new(&self.ctx, &pk.l_query, aux_len, force_gpu)?,
            a: StagedQuery::new(&self.ctx, &pk.a_query[1..], assign_len, force_gpu)?,
            b_g1: StagedQuery::new(&self.ctx, &pk.b_g1_query[1..], assign_len, force_gpu)?,
            qap: GpuQap::new(&self.ctx, domain_size)?,
            domain_size,
        })
    }

    /// The three constraint-evaluation vectors, zero-padded to the domain.
    /// Parallel over constraints; ~28 terms per row on a hash circuit.
    fn qap_inputs(
        &self,
        m: &ConstraintMatrices<Fr>,
        full: &[Fr],
        domain_size: usize,
    ) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>) {
        let nc = m.num_constraints;
        let ni = m.num_instance_variables;
        let zero = Fr::zero();
        let dot = |row: &[(Fr, usize)]| -> Fr { row.iter().map(|(c, i)| *c * full[*i]).sum() };
        let (mut a, mut b, mut c) =
            (vec![zero; domain_size], vec![zero; domain_size], vec![zero; domain_size]);
        let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1);
        let chunk = nc.div_ceil(threads).max(1);
        std::thread::scope(|s| {
            for (t, ((ca, cb), cc)) in a[..nc]
                .chunks_mut(chunk)
                .zip(b[..nc].chunks_mut(chunk))
                .zip(c[..nc].chunks_mut(chunk))
                .enumerate()
            {
                let dot = &dot;
                s.spawn(move || {
                    for j in 0..ca.len() {
                        let i = t * chunk + j;
                        ca[j] = dot(&m.a[i]);
                        cb[j] = dot(&m.b[i]);
                        cc[j] = dot(&m.c[i]);
                    }
                });
            }
        });
        a[nc..nc + ni].clone_from_slice(&full[..ni]);
        (a, b, c)
    }

    fn msm(&self, q: &StagedQuery, scalars: &[Fr]) -> Result<G1Projective> {
        if q.keep.is_empty() {
            return Ok(G1Projective::zero());
        }
        match &q.dev {
            Some(dev) => {
                let s: Vec<Fr> = q.keep.iter().map(|&i| scalars[i as usize]).collect();
                Ok(gpu_msm_staged(&self.ctx, &self.msm_kernel, dev, &s, &self.geom)?.0)
            }
            None => {
                // The CPU path uses the ORIGINAL bases, points at infinity and
                // all: arkworks handles those, the filtering is for the kernel.
                let n = q.len.min(scalars.len());
                let bi: Vec<_> = scalars[..n].iter().map(|x| x.into_bigint()).collect();
                Ok(G1Projective::msm_bigint(&q.bases[..n], &bi))
            }
        }
    }

    fn coeff(
        &self,
        initial: G1Projective,
        query: &[G1Affine],
        staged: &StagedQuery,
        vk_param: G1Affine,
        assignment: &[Fr],
    ) -> Result<G1Projective> {
        // `query[0]` is the constant wire's coefficient, whose assignment is
        // always 1, so it is added rather than multiplied.
        let acc = self.msm(staged, assignment)?;
        let mut res = initial;
        res += query[0].into_group();
        res += acc;
        res += vk_param.into_group();
        Ok(res)
    }

    /// Produce a proof. `full_assignment` is `[1, public inputs.., witness..]`,
    /// the layout `ConstraintMatrices` indexes into.
    pub fn prove(
        &self,
        pk: &ProvingKey<Bn254>,
        key: &PreparedKey,
        matrices: &ConstraintMatrices<Fr>,
        full_assignment: &[Fr],
        r: Fr,
        s: Fr,
    ) -> Result<Proof<Bn254>> {
        let ni = matrices.num_instance_variables;
        if full_assignment.len() != ni + matrices.num_witness_variables {
            return Err(Error::Invalid(format!(
                "assignment has {} entries, expected {}",
                full_assignment.len(),
                ni + matrices.num_witness_variables
            )));
        }

        let (a, b, c) = self.qap_inputs(matrices, full_assignment, key.domain_size);
        let h = key.qap.h_from_abc(&self.ctx, &a, &b, &c)?;

        let input_assignment = &full_assignment[1..ni];
        let aux_assignment = &full_assignment[ni..];
        let assignment: Vec<Fr> = [input_assignment, aux_assignment].concat();

        let h_acc = self.msm(&key.h, &h)?;
        let l_aux_acc = self.msm(&key.l, aux_assignment)?;
        let r_s_delta_g1 = pk.delta_g1.into_group() * (r * s);

        let g_a = self.coeff(
            pk.delta_g1.into_group() * r,
            &pk.a_query,
            &key.a,
            pk.vk.alpha_g1,
            &assignment,
        )?;
        let s_g_a = g_a * s;

        let g1_b = if r.is_zero() {
            G1Projective::zero()
        } else {
            self.coeff(
                pk.delta_g1.into_group() * s,
                &pk.b_g1_query,
                &key.b_g1,
                pk.beta_g1,
                &assignment,
            )?
        };

        // B in G2 stays on the CPU: Fq2 arithmetic is not in this kernel set.
        let assignment_bi: Vec<_> = assignment.iter().map(|x| x.into_bigint()).collect();
        let g2_b = {
            let acc = <Bn254 as Pairing>::G2::msm_bigint(&pk.b_g2_query[1..], &assignment_bi);
            let mut res = pk.vk.delta_g2.into_group() * s;
            res += pk.b_g2_query[0].into_group();
            res += acc;
            res += pk.vk.beta_g2.into_group();
            res
        };

        let mut g_c = s_g_a;
        g_c += g1_b * r;
        g_c -= r_s_delta_g1;
        g_c += l_aux_acc;
        g_c += h_acc;

        Ok(Proof {
            a: g_a.into_affine(),
            b: g2_b.into_affine(),
            c: g_c.into_affine(),
        })
    }

    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }
}

/// Unused import guard: `FftField` is needed by `qap`, re-exported for callers
/// building their own domains.
#[allow(unused)]
fn _generator() -> Fr {
    Fr::GENERATOR
}
