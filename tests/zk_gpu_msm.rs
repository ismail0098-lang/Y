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
//! On top of those two operations sits a full Pippenger MSM
//! (`gpu_msm_matches_arkworks`): the host bins point indices by scalar window
//! digit into a CSR pair and the GPU accumulates every bucket of every window
//! in one launch.

use std::path::{Path, PathBuf};
use std::process::Command;

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::{AdditiveGroup, CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_ff::{Field, PrimeField, UniformRand, Zero};
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
    assert!(g.into_affine().is_on_curve());
}

// ---------------------------------------------------------------------------
// Pippenger MSM
// ---------------------------------------------------------------------------

/// Bucket geometry: how the 254 bits of a scalar are cut into windows.
///
/// Parameterised by the NUMBER of windows rather than by a window width, and
/// that is the whole point. The obvious formulation — every window `c` bits
/// wide, `ceil(254/c)` of them — leaves the top window holding whatever bits
/// are left over, and 254 is not a multiple of anything convenient. At `c = 9`
/// the top window is 2 bits wide, so its three live buckets take a QUARTER of
/// all points each, while a typical bucket in a full window takes `n / 512`.
/// One thread then runs 262,144 serial point additions at `n = 2^20` and the
/// other 14,847 wait for it.
///
/// Measured, before this was fixed (`n = 2^20`, kernel time only): c=8 421 ms,
/// c=9 6,380 ms, c=10 1,730 ms, c=12 6,196 ms, c=13 231 ms. The ranking has
/// nothing to do with total work — which falls monotonically as `c` grows —
/// and everything to do with `2^(c - top window bits)`.
///
/// So the windows are made as EVEN as possible instead: `254 = nw*q + r`
/// gives `r` windows of `q+1` bits and `nw-r` of `q`, and the worst imbalance
/// is 2x rather than 128x. This is entirely host-side; the kernel never knew
/// how the buckets were cut.
///
/// The remaining tuning tension is genuine:
///   - GPU work is `n * nw` point additions, so FEWER windows is less work.
///   - GPU parallelism is one thread per bucket, `sum 2^width`, which grows as
///     windows get wider.
///   - The host-side reduction is `O(total buckets)` curve operations and does
///     not depend on `n`, so wider windows grow the part that does not scale.
#[derive(Clone)]
struct Geom {
    widths: Vec<usize>,
    shifts: Vec<usize>,
    /// Bucket index at which each window starts; `base[nw]` is the total.
    base: Vec<usize>,
    nw: usize,
    nb: usize,
}

impl Geom {
    fn new(nw: usize) -> Self {
        let q = 254 / nw;
        let r = 254 % nw;
        assert!(q >= 8, "windows narrower than 8 bits break the 256-thread block");
        // The wider windows go first; it makes no difference to correctness,
        // only to which buckets are which.
        let widths: Vec<usize> = (0..nw).map(|w| if w < r { q + 1 } else { q }).collect();
        let mut shifts = Vec::with_capacity(nw);
        let mut base = Vec::with_capacity(nw + 1);
        let (mut bit, mut b) = (0usize, 0usize);
        for &w in &widths {
            shifts.push(bit);
            base.push(b);
            bit += w;
            b += 1usize << w;
        }
        base.push(b);
        assert_eq!(bit, 254);
        assert_eq!(b % 256, 0, "bucket count must fill whole blocks");
        Geom { widths, shifts, base, nw, nb: b }
    }

    fn avg_width(&self) -> f64 {
        254.0 / self.nw as f64
    }
}

/// The `width`-bit field of a little-endian 4-limb scalar starting at `shift`.
/// Written over limbs rather than over a bit vector because the bit vector is
/// 254 pushes per scalar and this is on the measured path.
fn window_digit(l: &[u64; 4], shift: usize, width: usize) -> u32 {
    let word = shift / 64;
    if word >= 4 {
        return 0;
    }
    let off = shift % 64;
    let mut v = l[word] >> off;
    if off + width > 64 && word + 1 < 4 {
        v |= l[word + 1] << (64 - off);
    }
    (v & ((1u64 << width) - 1)) as u32
}

/// The counting sort that makes bucket accumulation tractable on a GPU.
///
/// The natural formulation of Pippenger scatters: many scalars land in one
/// bucket and the accumulation into it is serial, so a thread per POINT needs
/// atomics over a non-commutative-looking sequence of curve additions. Binning
/// on the host inverts it into a gather — thread `b` owns bucket `b` and walks
/// a contiguous slice `Idx[Off[b]..Off[b+1]]` with no synchronisation at all.
/// The sort is integer work linear in `n * windows`.
///
/// Digit 0 contributes nothing to the sum and is dropped here rather than
/// accumulated and discarded, so each window's bucket 0 is always empty.
///
/// Parallel over points, in the standard three phases: per-thread histograms,
/// a prefix sum that hands each thread a private write cursor per bucket, then
/// a scatter with no synchronisation. It has to be parallel to be measured
/// honestly — the baseline it is compared against uses every core, and a
/// single-threaded host phase was 110 ms of a 285 ms total at `n = 2^20`,
/// larger than the GPU kernel it was feeding.
///
/// Points land in a bucket in a different ORDER than the serial version would
/// put them. That is safe here and not by luck: curve addition is associative
/// and commutative, so a bucket's sum is the same point whatever order it was
/// accumulated in. The Jacobian representative differs; the point does not.
fn bin_by_digit(scalars: &[Fr], g: &Geom) -> (Vec<u32>, Vec<u32>) {
    let n = scalars.len();
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(n.max(1));
    let chunk = n.div_ceil(threads.max(1)).max(1);
    let nchunk = n.div_ceil(chunk);

    // Phase 1: a private histogram per chunk.
    let counts: Vec<Vec<u32>> = std::thread::scope(|s| {
        let handles: Vec<_> = scalars
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    let mut c = vec![0u32; g.nb];
                    for sc in part {
                        let l = sc.into_bigint().0;
                        for w in 0..g.nw {
                            let d = window_digit(&l, g.shifts[w], g.widths[w]);
                            if d != 0 {
                                c[g.base[w] + d as usize] += 1;
                            }
                        }
                    }
                    c
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Phase 2: bucket offsets, then each chunk's private cursor inside them.
    let mut off = vec![0u32; g.nb + 1];
    let mut cursor = vec![0u32; nchunk * g.nb];
    let mut running = 0u32;
    for b in 0..g.nb {
        off[b] = running;
        for (t, c) in counts.iter().enumerate() {
            cursor[t * g.nb + b] = running;
            running += c[b];
        }
    }
    off[g.nb] = running;

    // Phase 3: scatter. Every write target is private to one thread.
    let mut idx = vec![0u32; running as usize];
    {
        // Hand each thread a disjoint set of slots by index, which the borrow
        // checker cannot see; the cursors above prove it.
        struct Shared(*mut u32);
        unsafe impl Sync for Shared {}
        let shared = Shared(idx.as_mut_ptr());
        std::thread::scope(|s| {
            for (t, part) in scalars.chunks(chunk).enumerate() {
                let cur0 = &cursor[t * g.nb..(t + 1) * g.nb];
                let shared = &shared;
                let base_i = t * chunk;
                s.spawn(move || {
                    let mut cur = cur0.to_vec();
                    for (j, sc) in part.iter().enumerate() {
                        let l = sc.into_bigint().0;
                        for w in 0..g.nw {
                            let d = window_digit(&l, g.shifts[w], g.widths[w]);
                            if d != 0 {
                                let b = g.base[w] + d as usize;
                                unsafe { *shared.0.add(cur[b] as usize) = (base_i + j) as u32 };
                                cur[b] += 1;
                            }
                        }
                    }
                });
            }
        });
    }
    (idx, off)
}

/// Where the wall clock went. Reported separately because the lesson from the
/// NTT applies verbatim here: the first end-to-end figure measured there was
/// 96% marshalling, and a single number would have hidden it.
#[derive(Default, Clone, Copy)]
struct MsmTiming {
    bin: f64,
    stage: f64,
    h2d: f64,
    kernel: f64,
    d2h: f64,
    reduce: f64,
    nidx: usize,
}

impl MsmTiming {
    fn total(&self) -> f64 {
        self.bin + self.stage + self.h2d + self.kernel + self.d2h + self.reduce
    }

    /// What a SECOND MSM over the same bases costs.
    ///
    /// This is the number a prover actually pays, and it is not an accounting
    /// trick: in Groth16 the G1 bases are the proving key, fixed for the
    /// lifetime of the circuit, while the scalars are the witness and change
    /// every proof. Staging the bases into Montgomery form and pushing them
    /// across PCIe therefore happens once at key-load, not per proof. Reported
    /// alongside the cold total, never instead of it.
    fn steady_state(&self) -> f64 {
        self.bin + self.kernel + self.d2h + self.reduce
    }
}

/// A full MSM: `sum_i scalars[i] * points[i]`, with the O(n) bucket
/// accumulation on the GPU.
///
/// The two reductions that follow are on the host. Their cost is
/// `O(windows * 2^c)` curve operations and does not depend on `n` at all, so
/// they are a fixed overhead that a large enough MSM amortises — but at
/// `c = 16` that fixed cost is a million curve additions, which is why the
/// benchmark sweeps `c` rather than assuming one. Moving them to the GPU is a
/// scheduling change, not a correctness one.
fn gpu_msm(
    ctx: &CudaContext,
    module: &KernelModule,
    points: &[G1Projective],
    scalars: &[Fr],
    g: &Geom,
) -> (G1Projective, MsmTiming) {
    use std::time::Instant;
    let (buckets, _off, mut tm) = bucket_pass(ctx, module, points, scalars, g);

    // Each window's `sum_d d * B[d]` is independent, so they run in parallel
    // and only the final combination is serial.
    let t = Instant::now();
    let sums: Vec<G1Projective> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..g.nw)
            .map(|w| {
                let buckets = &buckets;
                s.spawn(move || {
                    // sum_d d * B[d], by a running sum from the top: each B[d]
                    // is added once for every step down to 1, i.e. d times.
                    let (mut running, mut acc) =
                        (G1Projective::zero(), G1Projective::zero());
                    for d in (1..(1usize << g.widths[w])).rev() {
                        running += buckets[g.base[w] + d];
                        acc += running;
                    }
                    acc
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut total = G1Projective::zero();
    for w in (0..g.nw).rev() {
        // Bring the accumulated higher windows down to this window's scale.
        for _ in 0..g.widths[w] {
            total = total.double();
        }
        total += sums[w];
    }
    tm.reduce = t.elapsed().as_secs_f64();
    (total, tm)
}

/// Montgomery-stage one coordinate into the planar `uint4` layout, in
/// parallel. Chunks of points map to disjoint contiguous runs in both planes.
fn stage_planar(points: &[G1Projective], f: fn(&G1Projective) -> Fq, r: Fq) -> Vec<u32> {
    let n = points.len();
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(n.max(1));
    let chunk = n.div_ceil(threads.max(1)).max(1);
    let mut out = vec![0u32; n * 8];
    let (plane0, plane1) = out.split_at_mut(4 * n);
    std::thread::scope(|s| {
        for ((c0, c1), pts) in plane0
            .chunks_mut(4 * chunk)
            .zip(plane1.chunks_mut(4 * chunk))
            .zip(points.chunks(chunk))
        {
            s.spawn(move || {
                for (i, p) in pts.iter().enumerate() {
                    let l = limbs32(&(f(p) * r));
                    c0[i * 4..i * 4 + 4].copy_from_slice(&l[0..4]);
                    c1[i * 4..i * 4 + 4].copy_from_slice(&l[4..8]);
                }
            });
        }
    });
    out
}

/// The GPU half on its own: bin, stage, launch, read back. Returns one point
/// per bucket (the identity for empty ones) plus the offsets, so a test can
/// look at the buckets themselves rather than only at the reduced total.
fn bucket_pass(
    ctx: &CudaContext,
    module: &KernelModule,
    points: &[G1Projective],
    scalars: &[Fr],
    g: &Geom,
) -> (Vec<G1Projective>, Vec<u32>, MsmTiming) {
    use std::time::Instant;
    let mut tm = MsmTiming::default();
    let n = points.len();
    let r = r_mod_q();

    let t = Instant::now();
    let (idx, off) = bin_by_digit(scalars, g);
    tm.bin = t.elapsed().as_secs_f64();
    tm.nidx = idx.len();

    // Montgomery staging of the point coordinates into the planar layout.
    // This is `to_limbs()`-shaped work and is exactly what made the NTT's
    // first end-to-end number meaningless, so it gets its own column.
    let t = Instant::now();
    let (hx, hy, hz) = (
        stage_planar(points, |p| p.x, r),
        stage_planar(points, |p| p.y, r),
        stage_planar(points, |p| p.z, r),
    );
    tm.stage = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let up = |v: &[u32]| {
        let d = ctx.alloc(v.len().max(1) * 4).unwrap();
        ctx.memcpy_htod_at(&d, 0, as_bytes(v)).unwrap();
        d
    };
    let (dpx, dpy, dpz) = (up(&hx), up(&hy), up(&hz));
    let (didx, doff) = (up(&idx), up(&off));
    let (drx, dry, drz) = (
        ctx.alloc(g.nb * 8 * 4).unwrap(),
        ctx.alloc(g.nb * 8 * 4).unwrap(),
        ctx.alloc(g.nb * 8 * 4).unwrap(),
    );
    // Poison the output so an unwritten bucket cannot pass as a correct one.
    for d in [&drx, &dry, &drz] {
        ctx.memset_u8(d, 0xA5).unwrap();
    }
    ctx.synchronize().unwrap();
    tm.h2d = t.elapsed().as_secs_f64();

    let args = vec![
        dpx.device_ptr(), dpy.device_ptr(), dpz.device_ptr(),
        didx.device_ptr(), doff.device_ptr(),
        drx.device_ptr(), dry.device_ptr(), drz.device_ptr(),
        g.nb as u64, n as u64, tm.nidx as u64,
    ];
    let t = Instant::now();
    ctx.launch(module, ((g.nb / 256) as u32, 1, 1), (256, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().expect("bucket kernel did not complete");
    tm.kernel = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let read = |d: &y::cuda_runtime::DeviceBuffer| {
        let mut raw = vec![0u8; g.nb * 8 * 4];
        ctx.memcpy_dtoh_at(&mut raw, d, 0).unwrap();
        let w: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        from_planar(&w, g.nb)
    };
    let r_inv = r.inverse().unwrap();
    let (bx, by, bz) = (read(&drx), read(&dry), read(&drz));
    tm.d2h = t.elapsed().as_secs_f64();

    // Bucket `b` is the identity exactly when its slice is empty. The kernel
    // seeds its accumulator with the first point of the slice because
    // add-2007-bl has no zero element, so an empty bucket's output is
    // meaningless — the host knows which those are and never reads them.
    let buckets: Vec<G1Projective> = (0..g.nb)
        .map(|b| {
            if off[b] == off[b + 1] {
                G1Projective::zero()
            } else {
                G1Projective::new_unchecked(bx[b] * r_inv, by[b] * r_inv, bz[b] * r_inv)
            }
        })
        .collect();
    (buckets, off, tm)
}

/// The whole point of the series: an MSM whose per-point work runs on the GPU,
/// checked against arkworks' own `VariableBaseMSM`. Run at several window
/// widths, because `c` changes the bucket geometry, the digit extraction and
/// the number of windows all at once — a bug in any of those can hide at one
/// width and not another.
#[test]
fn gpu_msm_matches_arkworks() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — bn254_msm_bucket.ysu was not executed.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_msm_bucket");

    const N: usize = 4096;
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0x5A1AD);
    let points: Vec<G1Projective> = (0..N).map(|_| G1Projective::rand(&mut rng)).collect();
    let scalars: Vec<Fr> = (0..N).map(|_| Fr::rand(&mut rng)).collect();
    let affine: Vec<G1Affine> = points.iter().map(|p| p.into_affine()).collect();
    let want = G1Projective::msm(&affine, &scalars).expect("arkworks MSM failed");

    for nw in [20usize, 24, 28, 31] {
        let g = Geom::new(nw);
        let (got, tm) = gpu_msm(&ctx, &module, &points, &scalars, &g);
        assert_eq!(got, want, "GPU MSM disagrees with arkworks at nw={}", nw);

        // Guard against a vacuous pass: if every bucket held at most one point
        // the kernel's accumulation loop would never run and this would be
        // checking the load/store path only.
        let (_, off) = bin_by_digit(&scalars, &g);
        let deepest = (0..g.nb).map(|b| off[b + 1] - off[b]).max().unwrap();
        assert!(
            deepest > 1,
            "nw={}: no bucket held more than one point; the accumulation loop never ran",
            nw
        );
        assert!(tm.nidx > N, "nw={}: binning produced fewer entries than points", nw);
    }
}

/// Every non-empty bucket must come back ON THE CURVE. This is the check that
/// catches a formula which is self-consistently wrong: matching an expected
/// value can be an accident of one input, but satisfying `y^2 = x^3 + 3` after
/// a few thousand chained additions cannot be.
#[test]
fn every_bucket_lands_on_the_curve() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_msm_bucket");
    let g = Geom::new(28);

    const N: usize = 2048;
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0x0C_1234);
    let points: Vec<G1Projective> = (0..N).map(|_| G1Projective::rand(&mut rng)).collect();
    let scalars: Vec<Fr> = (0..N).map(|_| Fr::rand(&mut rng)).collect();
    let (buckets, off, _) = bucket_pass(&ctx, &module, &points, &scalars, &g);

    let mut checked = 0usize;
    for b in 0..g.nb {
        if off[b] == off[b + 1] {
            continue;
        }
        assert!(
            buckets[b].into_affine_unchecked_is_on_curve(),
            "bucket {} ({} points) came back off the curve",
            b,
            off[b + 1] - off[b]
        );
        checked += 1;
    }
    assert!(checked > g.nb / 2, "only {} buckets were non-empty", checked);
}

/// The control for the test above. `assert_eq!` against arkworks is only
/// meaningful if a wrong answer would actually differ, and an MSM has a
/// failure mode where it does not: get the window SHIFT wrong and every
/// scalar is still consumed, every bucket is still accumulated, and the result
/// is a perfectly valid point that is not the right one. Perturbing one scalar
/// bit must change the result.
#[test]
fn the_msm_oracle_is_not_vacuous() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_msm_bucket");

    const N: usize = 512;
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0xBEEF11);
    let points: Vec<G1Projective> = (0..N).map(|_| G1Projective::rand(&mut rng)).collect();
    let mut scalars: Vec<Fr> = (0..N).map(|_| Fr::rand(&mut rng)).collect();

    let g = Geom::new(28);
    let (a, _) = gpu_msm(&ctx, &module, &points, &scalars, &g);
    scalars[N / 2] += Fr::from(1u64);
    let (b, _) = gpu_msm(&ctx, &module, &points, &scalars, &g);

    assert_ne!(a, b, "changing a scalar did not change the MSM");
    assert_eq!(b - a, points[N / 2], "the difference is not the perturbed point");
}

// ---------------------------------------------------------------------------
// What it costs
// ---------------------------------------------------------------------------

/// The CPU baseline, using every core.
///
/// arkworks' `parallel` feature is deliberately NOT enabled in this repo's
/// dev-dependencies — turning it on would also speed up the Groth16 setup and
/// prove timings that `CLAUDE.md` quotes, silently invalidating them. So the
/// parallelism is done here instead: an MSM splits over points exactly, and
/// the partial sums add.
///
/// This is a slightly PESSIMISTIC baseline and the direction matters. Each
/// chunk runs its own Pippenger, and Pippenger's cost per point falls as `n`
/// grows, so 16 chunks of `n/16` do more total work than one pass over `n`.
/// The honest reading is that the true whole-CPU number sits somewhere between
/// this and `threads *` the single-core number — so this is the baseline to
/// beat, not one to hide behind.
fn cpu_msm_all_cores(points: &[G1Affine], scalars: &[Fr], threads: usize) -> G1Projective {
    let chunk = points.len().div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = points
            .chunks(chunk)
            .zip(scalars.chunks(chunk))
            .map(|(p, k)| s.spawn(move || G1Projective::msm(p, k).expect("chunk MSM failed")))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

/// `cargo test --release --features zk --test zk_gpu_msm -- --ignored --nocapture`
///
/// Reports the window-width sweep and the phase split, never a single number.
/// The NTT in this series taught that lesson expensively: its first end-to-end
/// figure was 2.1x and 96% of it was host marshalling.
#[test]
#[ignore]
fn what_the_gpu_msm_costs() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_msm_bucket");
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1);

    for log_n in [16usize, 18, 20] {
        let n = 1usize << log_n;
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0x11115 + log_n as u64);
        let points: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let affine: Vec<G1Affine> = points.iter().map(|p| p.into_affine()).collect();

        println!("\n=== n = 2^{} ({} points) ===", log_n, n);
        println!(
            "{:>3} {:>6}  {:>8} {:>8} {:>8} {:>9} {:>8} {:>9} {:>9}  {:>9}",
            "nw", "bits", "bin", "stage", "h2d", "kernel", "d2h", "reduce", "TOTAL", "buckets"
        );

        let mut best = (f64::MAX, 0usize, MsmTiming::default());
        for nw in [16usize, 18, 20, 22, 25, 28, 31] {
            let g = Geom::new(nw);
            // Warm up: the first launch pays PTX JIT and a cold GPU clock, and
            // a cold-clock reading is what made an earlier NTT measurement 17x
            // too slow in one direction and 8x too fast in another.
            let (warm, _) = gpu_msm(&ctx, &module, &points, &scalars, &g);
            let mut tm = MsmTiming::default();
            let mut total = f64::MAX;
            for _ in 0..3 {
                let (got, t) = gpu_msm(&ctx, &module, &points, &scalars, &g);
                assert_eq!(got, warm, "nw={} is not deterministic", nw);
                if t.steady_state() < total {
                    total = t.steady_state();
                    tm = t;
                }
            }
            println!(
                "{:>3} {:>6.2}  {:>7.1}m {:>7.1}m {:>7.1}m {:>8.1}m {:>7.1}m {:>8.1}m {:>8.1}m  {:>9}",
                nw, g.avg_width(),
                tm.bin * 1e3, tm.stage * 1e3, tm.h2d * 1e3, tm.kernel * 1e3,
                tm.d2h * 1e3, tm.reduce * 1e3, tm.total() * 1e3, g.nb
            );
            if total < best.0 {
                best = (total, nw, tm);
            }
        }

        let t = std::time::Instant::now();
        let cpu1 = G1Projective::msm(&affine, &scalars).unwrap();
        let one_core = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let cpu_n = cpu_msm_all_cores(&affine, &scalars, threads);
        let all_cores = t.elapsed().as_secs_f64();
        assert_eq!(cpu1, cpu_n, "the parallel CPU baseline disagrees with the serial one");

        let (_, nw, tm) = best;
        let total = tm.total();
        println!("\n  best nw           = {}  (chosen by the fixed-base cost)", nw);
        println!("  arkworks 1 core   = {:.1} ms", one_core * 1e3);
        println!("  arkworks {:>2} cores = {:.1} ms", threads, all_cores * 1e3);
        println!(
            "  cold end-to-end   = {:.1} ms -> {:.2}x one core, {:.2}x {} cores",
            total * 1e3, one_core / total, all_cores / total, threads
        );
        println!(
            "  fixed bases       = {:.1} ms -> {:.2}x one core, {:.2}x {} cores   <- what a prover pays per proof",
            tm.steady_state() * 1e3,
            one_core / tm.steady_state(),
            all_cores / tm.steady_state(),
            threads
        );
        println!(
            "  kernel only       = {:.1} ms -> {:.2}x one core, {:.2}x {} cores",
            tm.kernel * 1e3, one_core / tm.kernel, all_cores / tm.kernel, threads
        );
    }
}
