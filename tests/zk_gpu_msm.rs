//! BN254 G1 point addition on the GPU, written in Y.
//!
//! This is the atom of MSM, and the first thing in this series that runs over
//! the BASE field Fq rather than the scalar field Fr. They are different
//! 254-bit primes and are not interchangeable: a witness lives in Fr, a
//! point's coordinates live in Fq. `tools/gen_bn254_kernels.py` emits the same
//! CIOS multiply against either.
//!
//! The oracle is arkworks, which is this repo's designated independent check
//! and already a dev-dependency. Crucially the comparison is by point
//! EQUIVALENCE, not by limbs: Jacobian coordinates are not unique, `(X, Y, Z)`
//! and `(λ²X, λ³Y, λZ)` are the same point, and arkworks' addition formula is
//! not the one in the kernel. Comparing raw limbs would fail on correct
//! output — and, worse, could pass on incorrect output that happened to match
//! a different representative.
//!
//! What this does NOT yet do is a full MSM. Pippenger's bucket accumulation is
//! scheduling on top of this operation; getting the operation right first is
//! the part that cannot be skipped.

use std::path::{Path, PathBuf};
use std::process::Command;

use ark_bn254::{Fq, G1Projective};
use ark_ec::{AdditiveGroup, PrimeGroup};
use ark_ff::{BigInteger, Field, PrimeField, UniformRand, Zero};
use ark_std::rand::SeedableRng;

use y::cuda_runtime::{CudaContext, KernelModule};

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("Y")
}

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Compiling the same `.ysu` from several test threads at once races on the
/// `.ptx` output path: one thread reads the file while another is still
/// writing it, and the JIT rejects the torn result with "Can't load this
/// binary kind". Compile each kernel once per process, under a lock.
fn ptx_for(entry: &str) -> String {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<std::collections::HashMap<String, String>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = guard.get(entry) {
        return p.clone();
    }
    let out = Command::new(bin())
        .arg(repo().join(format!("tests/{}.ysu", entry)))
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .expect("failed to run the Y binary");
    assert!(
        out.status.success(),
        "{}.ysu did not compile:\n{}{}",
        entry,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(repo().join(format!("tests/{}.ptx", entry)))
        .expect("no .ptx written");
    guard.insert(entry.to_string(), ptx.clone());
    ptx
}

fn load_kernel(ctx: &CudaContext, entry: &str) -> KernelModule {
    let ptx = ptx_for(entry);
    // A single float instruction in a field kernel means some path is
    // still hardcoded and the limbs are being rounded.
    assert!(
        !ptx.contains("ld.global.f32") && !ptx.contains("add.f32"),
        "{} is routing field limbs through the float datapath",
        entry
    );
    ctx.load_ptx(&ptx, entry)
        .unwrap_or_else(|e| panic!("{} did not load: {}", entry, e))
}

/// `R mod q` for Fq, via the public API: 2^256 as a field element. The kernel
/// consumes and produces Montgomery form, and arkworks' own Montgomery limbs
/// are not part of its public surface.
fn r_mod_q() -> Fq {
    let mut r = Fq::from(1u64);
    for _ in 0..256 {
        r = r + r;
    }
    r
}

fn limbs32(x: &Fq) -> [u32; 8] {
    let l = x.into_bigint().0;
    let mut o = [0u32; 8];
    for i in 0..4 {
        o[2 * i] = l[i] as u32;
        o[2 * i + 1] = (l[i] >> 32) as u32;
    }
    o
}

fn from_limbs32(w: &[u32]) -> Fq {
    let mut b = [0u64; 4];
    for i in 0..4 {
        b[i] = w[2 * i] as u64 | ((w[2 * i + 1] as u64) << 32);
    }
    Fq::from_bigint(ark_ff::BigInt(b)).expect("limbs are not a canonical Fq element")
}

/// Planar arrays of `uint4`, the layout every kernel in this series reads.
fn to_planar(v: &[Fq], n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n * 8];
    for (i, x) in v.iter().enumerate() {
        let l = limbs32(x);
        for j in 0..8 {
            out[(j / 4) * 4 * n + i * 4 + (j % 4)] = l[j];
        }
    }
    out
}

fn from_planar(raw: &[u32], n: usize) -> Vec<Fq> {
    (0..n)
        .map(|i| {
            let w: Vec<u32> = (0..8)
                .map(|j| raw[(j / 4) * 4 * n + i * 4 + (j % 4)])
                .collect();
            from_limbs32(&w)
        })
        .collect()
}

fn as_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

#[test]
fn gpu_g1_addition_agrees_with_arkworks() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — bn254_g1_add.ysu was not executed.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_g1_add");

    const N: usize = 1024;
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
    let r = r_mod_q();

    // Independent random points. add-2007-bl is not a COMPLETE formula: it
    // assumes both inputs are non-zero and P != +-Q, which is the precondition
    // Pippenger's bucket accumulation is arranged to satisfy. Feeding it a
    // doubling would be testing a case it does not claim to handle.
    let ps: Vec<G1Projective> = (0..N).map(|_| G1Projective::rand(&mut rng)).collect();
    let qs: Vec<G1Projective> = (0..N).map(|_| G1Projective::rand(&mut rng)).collect();

    let mont = |v: &[G1Projective], f: fn(&G1Projective) -> Fq| -> Vec<u32> {
        let c: Vec<Fq> = v.iter().map(|p| f(p) * r).collect();
        to_planar(&c, v.len())
    };
    let px = mont(&ps, |p| p.x);
    let py = mont(&ps, |p| p.y);
    let pz = mont(&ps, |p| p.z);
    let qx = mont(&qs, |p| p.x);
    let qy = mont(&qs, |p| p.y);
    let qz = mont(&qs, |p| p.z);

    let up = |v: &[u32]| {
        let d = ctx.alloc(v.len() * 4).unwrap();
        ctx.memcpy_htod_at(&d, 0, as_bytes(v)).unwrap();
        d
    };
    let (dpx, dpy, dpz) = (up(&px), up(&py), up(&pz));
    let (dqx, dqy, dqz) = (up(&qx), up(&qy), up(&qz));
    let (drx, dry, drz) = (
        ctx.alloc(N * 8 * 4).unwrap(),
        ctx.alloc(N * 8 * 4).unwrap(),
        ctx.alloc(N * 8 * 4).unwrap(),
    );
    for d in [&drx, &dry, &drz] {
        ctx.memset_u8(d, 0xA5).unwrap();
    }

    let args = vec![
        dpx.device_ptr(), dpy.device_ptr(), dpz.device_ptr(),
        dqx.device_ptr(), dqy.device_ptr(), dqz.device_ptr(),
        drx.device_ptr(), dry.device_ptr(), drz.device_ptr(),
        N as u64,
    ];
    ctx.launch(&module, ((N / 256) as u32, 1, 1), (256, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");

    let read = |d: &y::cuda_runtime::DeviceBuffer| {
        let mut raw = vec![0u8; N * 8 * 4];
        ctx.memcpy_dtoh_at(&mut raw, d, 0).unwrap();
        let w: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        from_planar(&w, N)
    };
    // Results come back in Montgomery form; divide R back out.
    let r_inv = r.inverse().unwrap();
    let (rx, ry, rz) = (read(&drx), read(&dry), read(&drz));

    for i in 0..N {
        let got = G1Projective::new_unchecked(rx[i] * r_inv, ry[i] * r_inv, rz[i] * r_inv);
        let want = ps[i] + qs[i];
        // Point equivalence, not limb equality: Jacobian coordinates are not
        // unique and arkworks does not use add-2007-bl.
        assert_eq!(got, want, "GPU G1 add disagrees with arkworks at {}", i);
        assert!(
            got.into_affine_unchecked_is_on_curve(),
            "GPU result at {} is not on the curve",
            i
        );
    }
}


/// Doubling needs its own formula, and this is not a nicety: `add-2007-bl`
/// computes `H = U2 - U1`, which is ZERO when P == Q, and then builds the
/// result out of it. Feeding a doubling to the add kernel does not give a
/// wrong point, it gives the point at infinity - silently. Pippenger's bucket
/// reduction doubles constantly, so both formulas have to exist and both have
/// to be right.
#[test]
fn gpu_g1_doubling_agrees_with_arkworks() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — bn254_g1_dbl.ysu was not executed.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_g1_dbl");

    const N: usize = 1024;
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0xD00B1E);
    let r = r_mod_q();
    let ps: Vec<G1Projective> = (0..N).map(|_| G1Projective::rand(&mut rng)).collect();

    let up = |v: &[u32]| {
        let d = ctx.alloc(v.len() * 4).unwrap();
        ctx.memcpy_htod_at(&d, 0, as_bytes(v)).unwrap();
        d
    };
    let mont = |f: fn(&G1Projective) -> Fq| -> Vec<u32> {
        to_planar(&ps.iter().map(|p| f(p) * r).collect::<Vec<Fq>>(), N)
    };
    let (dx, dy, dz) = (up(&mont(|p| p.x)), up(&mont(|p| p.y)), up(&mont(|p| p.z)));
    let (drx, dry, drz) = (
        ctx.alloc(N * 8 * 4).unwrap(),
        ctx.alloc(N * 8 * 4).unwrap(),
        ctx.alloc(N * 8 * 4).unwrap(),
    );
    for d in [&drx, &dry, &drz] {
        ctx.memset_u8(d, 0xA5).unwrap();
    }
    let args = vec![
        dx.device_ptr(), dy.device_ptr(), dz.device_ptr(),
        drx.device_ptr(), dry.device_ptr(), drz.device_ptr(),
        N as u64,
    ];
    ctx.launch(&module, ((N / 256) as u32, 1, 1), (256, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");

    let read = |d: &y::cuda_runtime::DeviceBuffer| {
        let mut raw = vec![0u8; N * 8 * 4];
        ctx.memcpy_dtoh_at(&mut raw, d, 0).unwrap();
        let w: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        from_planar(&w, N)
    };
    let r_inv = r.inverse().unwrap();
    let (rx, ry, rz) = (read(&drx), read(&dry), read(&drz));
    for i in 0..N {
        let got = G1Projective::new_unchecked(rx[i] * r_inv, ry[i] * r_inv, rz[i] * r_inv);
        assert_eq!(got, ps[i].double(), "GPU G1 double disagrees with arkworks at {}", i);
        assert!(got.into_affine_unchecked_is_on_curve(), "result {} is not on the curve", i);
    }
}

/// The reason the two kernels cannot be collapsed into one, stated as a test:
/// the ADD kernel, given P and P, must NOT produce 2P. If a future change made
/// add-2007-bl appear to handle doubling, that would mean H stopped being
/// computed correctly.
#[test]
fn the_add_formula_really_is_incomplete() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_g1_add");
    const N: usize = 256;
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(7);
    let r = r_mod_q();
    let ps: Vec<G1Projective> = (0..N).map(|_| G1Projective::rand(&mut rng)).collect();
    let up = |v: &[u32]| {
        let d = ctx.alloc(v.len() * 4).unwrap();
        ctx.memcpy_htod_at(&d, 0, as_bytes(v)).unwrap();
        d
    };
    let mont = |f: fn(&G1Projective) -> Fq| -> Vec<u32> {
        to_planar(&ps.iter().map(|p| f(p) * r).collect::<Vec<Fq>>(), N)
    };
    let (dx, dy, dz) = (up(&mont(|p| p.x)), up(&mont(|p| p.y)), up(&mont(|p| p.z)));
    let (drx, dry, drz) = (
        ctx.alloc(N * 8 * 4).unwrap(),
        ctx.alloc(N * 8 * 4).unwrap(),
        ctx.alloc(N * 8 * 4).unwrap(),
    );
    // Both operands are the same buffer: P + P.
    let args = vec![
        dx.device_ptr(), dy.device_ptr(), dz.device_ptr(),
        dx.device_ptr(), dy.device_ptr(), dz.device_ptr(),
        drx.device_ptr(), dry.device_ptr(), drz.device_ptr(),
        N as u64,
    ];
    ctx.launch(&module, ((N / 256).max(1) as u32, 1, 1), (256, 1, 1), 0, &args).unwrap();
    ctx.synchronize().unwrap();
    let mut raw = vec![0u8; N * 8 * 4];
    ctx.memcpy_dtoh_at(&mut raw, &drz, 0).unwrap();
    let w: Vec<u32> = raw.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let rz = from_planar(&w, N);
    // H = 0 makes Z3 = 0: the point at infinity, not 2P. Caller beware.
    for (i, z) in rz.iter().enumerate() {
        assert!(z.is_zero(), "add(P, P) at {} did not degenerate to Z=0 as documented", i);
    }
}

trait OnCurve {
    fn into_affine_unchecked_is_on_curve(&self) -> bool;
}
impl OnCurve for G1Projective {
    /// A point that satisfies the curve equation is a far stronger statement
    /// than one that matches an expected value, and it is the check that
    /// catches a formula which is self-consistently wrong.
    fn into_affine_unchecked_is_on_curve(&self) -> bool {
        use ark_ec::CurveGroup;
        let a = self.into_affine();
        a.is_on_curve()
    }
}

/// The control. Every assertion above compares against arkworks, so this
/// checks arkworks is being driven correctly at all — that the generator is a
/// real point, that addition is not the identity, and that the Montgomery
/// staging round-trips.
#[test]
fn the_arkworks_oracle_is_not_vacuous() {
    let r = r_mod_q();
    let r_inv = r.inverse().unwrap();
    let x = Fq::from(123456789u64);
    assert_eq!((x * r) * r_inv, x, "Montgomery staging does not round-trip");
    assert_ne!(x * r, x, "R is 1; the staging is doing nothing");

    let g = G1Projective::generator();
    assert_ne!(g + g, g, "point addition is behaving as the identity");
    use ark_ec::CurveGroup;
    assert!(g.into_affine().is_on_curve());
}
