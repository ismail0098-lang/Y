//! The Groth16 QAP witness map, with its seven transforms on the GPU.
//!
//! `LibsnarkReduction::witness_map_from_matrices` is 3 iFFTs, 3 coset FFTs and
//! one coset iFFT over the evaluation domain, plus three sparse
//! matrix-vector products. Measured at 1,048,577 constraints (domain 2^21),
//! the transforms are **89% of that phase** — 672 ms of 752 ms — which is what
//! makes moving them worth the machinery below.
//!
//! **Everything stays on the device between transforms.** A single transform
//! could be permuted and staged on the host for free, since the data is being
//! uploaded anyway, but seven back-to-back round trips of 2^21 elements is
//! ~900 MB over PCIe and would cost more than the transforms save. So the
//! chain uploads `a`, `b`, `c` once and downloads `h` once.
//!
//! **Every scalar constant is folded into a table**, which is why the only
//! pointwise kernels needed are a multiply and a subtract:
//!
//!   - an iFFT is `(1/n) * DFT_{w^-1}`, and a coset FFT pre-multiplies by
//!     `g^i`. Back to back they become one table `T1[i] = g^i / n`.
//!   - the final coset iFFT post-multiplies by `g^-i`, and the vanishing
//!     polynomial's inverse is a constant applied just before it, so
//!     `T2[i] = g^-i * Z(g)^-1 / n`.
//!
//! The DFT convention is the one `tests/zk_gpu_ntt.rs` already pins against a
//! naive O(N^2) DFT: `bn254_ntt_stage` is decimation-in-time, so it consumes
//! bit-reversed input and produces natural order. That permutation is a table
//! rather than in-kernel bit twiddling because it is identical for all seven
//! transforms.


use ark_bn254::Fr as ArkFr;
use ark_ff::{FftField, Field, PrimeField};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

use y::cuda_runtime::{CudaContext, DeviceBuffer, KernelModule};

use crate::device::{alloc, launch, read_u32, sync, upload_u32};
use crate::error::Result;

/// `R mod p` for the SCALAR field. The kernels consume and produce Montgomery
/// form; arkworks' own Montgomery limbs are not public, so the factor is
/// applied through the public API exactly as the MSM does it.
pub fn r_mod_r() -> ArkFr {
    let mut r = ArkFr::from(1u64);
    for _ in 0..256 {
        r = r + r;
    }
    r
}

fn limbs32(x: &ArkFr) -> [u32; 8] {
    let l = x.into_bigint().0;
    let mut o = [0u32; 8];
    for i in 0..4 {
        o[2 * i] = l[i] as u32;
        o[2 * i + 1] = (l[i] >> 32) as u32;
    }
    o
}

fn from_limbs32(w: &[u32]) -> ArkFr {
    let mut b = [0u64; 4];
    for i in 0..4 {
        b[i] = w[2 * i] as u64 | ((w[2 * i + 1] as u64) << 32);
    }
    ArkFr::from_bigint(ark_ff::BigInt(b)).expect("limbs are not a canonical Fr element")
}

/// Planar arrays of `uint4`, already in Montgomery form.
fn to_planar_mont(v: &[ArkFr], n: usize, r: ArkFr) -> Vec<u32> {
    let mut out = vec![0u32; n * 8];
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1);
    let chunk = n.div_ceil(threads).max(1);
    let (p0, p1) = out.split_at_mut(4 * n);
    std::thread::scope(|s| {
        for ((c0, c1), part) in p0
            .chunks_mut(4 * chunk)
            .zip(p1.chunks_mut(4 * chunk))
            .zip(v.chunks(chunk))
        {
            s.spawn(move || {
                for (i, x) in part.iter().enumerate() {
                    let l = limbs32(&(*x * r));
                    c0[i * 4..i * 4 + 4].copy_from_slice(&l[0..4]);
                    c1[i * 4..i * 4 + 4].copy_from_slice(&l[4..8]);
                }
            });
        }
    });
    out
}

fn from_planar_mont(raw: &[u32], n: usize, r_inv: ArkFr) -> Vec<ArkFr> {
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1);
    let chunk = n.div_ceil(threads).max(1);
    let mut out = vec![ArkFr::from(0u64); n];
    std::thread::scope(|s| {
        for (t, part) in out.chunks_mut(chunk).enumerate() {
            let raw = &raw;
            s.spawn(move || {
                for (j, o) in part.iter_mut().enumerate() {
                    let i = t * chunk + j;
                    let w: [u32; 8] = std::array::from_fn(|k| raw[(k / 4) * 4 * n + i * 4 + (k % 4)]);
                    *o = from_limbs32(&w) * r_inv;
                }
            });
        }
    });
    out
}

fn bit_reverse_table(n: usize, log_n: u32) -> Vec<u32> {
    (0..n as u32).map(|i| i.reverse_bits() >> (32 - log_n)).collect()
}

/// A resident QAP evaluator: twiddles, permutation and scratch all live on the
/// device, built once per domain size and reused across proofs.
pub struct GpuQap {
    n: usize,
    log_n: u32,
    r: ArkFr,
    r_inv: ArkFr,
    stage: KernelModule,
    permute: KernelModule,
    mul: KernelModule,
    sub: KernelModule,
    /// Forward twiddles per stage `s = 1..=log_n`, index `s-1`.
    tw_fwd: Vec<DeviceBuffer>,
    tw_inv: Vec<DeviceBuffer>,
    d_perm: DeviceBuffer,
    /// `g^i / n` and `g^-i * Z(g)^-1 / n`.
    d_t1: DeviceBuffer,
    d_t2: DeviceBuffer,
    /// Working buffers.
    d_a: DeviceBuffer,
    d_b: DeviceBuffer,
    d_c: DeviceBuffer,
    d_scratch: DeviceBuffer,
}

impl GpuQap {
    pub fn new(ctx: &CudaContext, domain_size: usize) -> Result<Self> {
        let n = domain_size;
        let log_n = n.trailing_zeros();
        if !n.is_power_of_two() {
            return Err(crate::error::Error::Invalid(format!(
                "the evaluation domain must be a power of two, got {n}"
            )));
        }
        let r = r_mod_r();
        let r_inv = r.inverse().unwrap();

        let domain = GeneralEvaluationDomain::<ArkFr>::new(n).expect("domain");
        assert_eq!(domain.size(), n, "arkworks chose a different domain size");
        let w = domain.group_gen();
        let w_inv = domain.group_gen_inv();

        let upload = |v: &[ArkFr]| -> Result<DeviceBuffer> {
            upload_u32(ctx, &to_planar_mont(v, v.len(), r))
        };

        // Stage `s` needs `w_m^pos` for pos in 0..2^(s-1), where w_m = w^(n/m).
        let build_tw = |root: ArkFr| -> Result<Vec<DeviceBuffer>> {
            (1..=log_n)
                .map(|s| {
                    let m = 1usize << s;
                    let half = m / 2;
                    let step = root.pow([(n / m) as u64]);
                    let mut v = Vec::with_capacity(half);
                    let mut cur = ArkFr::from(1u64);
                    for _ in 0..half {
                        v.push(cur);
                        cur *= step;
                    }
                    upload(&v)
                })
                .collect()
        };

        let n_inv = ArkFr::from(n as u64).inverse().unwrap();
        let g = ArkFr::GENERATOR;
        let g_inv = g.inverse().unwrap();
        let vanishing = domain
            .evaluate_vanishing_polynomial(g)
            .inverse()
            .unwrap();

        let pow_table = |base: ArkFr, scale: ArkFr| -> Vec<ArkFr> {
            let mut v = Vec::with_capacity(n);
            let mut cur = scale;
            for _ in 0..n {
                v.push(cur);
                cur *= base;
            }
            v
        };

        let perm = bit_reverse_table(n, log_n);
        let d_perm = upload_u32(ctx, &perm)?;

        let bytes = n * 8 * 4;
        Ok(GpuQap {
            n,
            log_n,
            r,
            r_inv,
            stage: crate::kernels::load(ctx, "bn254_ntt_stage", crate::kernels::NTT_STAGE)?,
            permute: crate::kernels::load(ctx, "bn254_permute", crate::kernels::PERMUTE)?,
            mul: crate::kernels::load(ctx, "bn254_fr_mul_fast", crate::kernels::FR_MUL)?,
            sub: crate::kernels::load(ctx, "bn254_sub_vec", crate::kernels::SUB_VEC)?,
            tw_fwd: build_tw(w)?,
            tw_inv: build_tw(w_inv)?,
            d_perm,
            d_t1: upload(&pow_table(g, n_inv))?,
            d_t2: upload(&pow_table(g_inv, n_inv * vanishing))?,
            d_a: alloc(ctx, bytes)?,
            d_b: alloc(ctx, bytes)?,
            d_c: alloc(ctx, bytes)?,
            d_scratch: alloc(ctx, bytes)?,
        })
    }

    fn launch(&self, ctx: &CudaContext, m: &KernelModule, threads: usize, args: &[u64]) -> Result<()> {
        launch(ctx, m, threads, args)
    }

    /// `dst <- DFT(src)` with the given twiddle set. `src` is consumed in
    /// natural order; the permutation into bit-reversed order happens on the
    /// device, into `dst`.
    fn transform(&self, ctx: &CudaContext, src: &DeviceBuffer, dst: &DeviceBuffer, inv: bool) -> Result<()> {
        self.launch(
            ctx,
            &self.permute,
            self.n,
            &[src.device_ptr(), dst.device_ptr(), self.d_perm.device_ptr(), self.n as u64],
        )?;
        let tws = if inv { &self.tw_inv } else { &self.tw_fwd };
        for s in 1..=self.log_n {
            let half = 1usize << (s - 1);
            self.launch(
                ctx,
                &self.stage,
                self.n / 2,
                &[
                    dst.device_ptr(),
                    tws[(s - 1) as usize].device_ptr(),
                    half as u64,
                    (self.n / 2) as u64,
                    self.n as u64,
                ],
            )?;
        }
        Ok(())
    }

    fn pointwise(&self, ctx: &CudaContext, m: &KernelModule, a: &DeviceBuffer, b: &DeviceBuffer, out: &DeviceBuffer) -> Result<()> {
        self.launch(
            ctx,
            m,
            self.n,
            &[a.device_ptr(), b.device_ptr(), out.device_ptr(), self.n as u64],
        )
    }

    /// `coset_fft(ifft(x))`, in place on `x`, using `scratch`.
    fn ifft_then_coset_fft(&self, ctx: &CudaContext, x: &DeviceBuffer) -> Result<()> {
        self.transform(ctx, x, &self.d_scratch, true)?;
        // T1 = g^i / n: the iFFT's 1/n and the coset offset in one table.
        self.pointwise(ctx, &self.mul, &self.d_scratch, &self.d_t1, &self.d_scratch)?;
        self.transform(ctx, &self.d_scratch, x, false)
    }

    /// The whole witness map, given the three constraint evaluations in
    /// natural coefficient order. Returns `h`.
    pub fn h_from_abc(&self, ctx: &CudaContext, a: &[ArkFr], b: &[ArkFr], c: &[ArkFr]) -> Result<Vec<ArkFr>> {
        for (name, v) in [("a", a), ("b", b), ("c", c)] {
            if v.len() != self.n {
                return Err(crate::error::Error::Invalid(format!(
                    "`{name}` has {} elements, expected the domain size {}",
                    v.len(),
                    self.n
                )));
            }
        }
        let up = |v: &[ArkFr], d: &DeviceBuffer| -> Result<()> {
            crate::device::write_u32(ctx, d, &to_planar_mont(v, self.n, self.r))
        };
        up(a, &self.d_a)?;
        up(b, &self.d_b)?;
        up(c, &self.d_c)?;

        self.ifft_then_coset_fft(ctx, &self.d_a)?;
        self.ifft_then_coset_fft(ctx, &self.d_b)?;
        self.ifft_then_coset_fft(ctx, &self.d_c)?;

        // ab <- a * b, then ab <- ab - c.
        self.pointwise(ctx, &self.mul, &self.d_a, &self.d_b, &self.d_a)?;
        self.pointwise(ctx, &self.sub, &self.d_a, &self.d_c, &self.d_a)?;

        // coset_ifft, with Z(g)^-1 folded into T2 alongside 1/n and g^-i.
        self.transform(ctx, &self.d_a, &self.d_scratch, true)?;
        self.pointwise(ctx, &self.mul, &self.d_scratch, &self.d_t2, &self.d_scratch)?;
        sync(ctx)?;

        let w = read_u32(ctx, &self.d_scratch, self.n * 8)?;
        Ok(from_planar_mont(&w, self.n, self.r_inv))
    }
}
