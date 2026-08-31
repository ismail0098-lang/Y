//! Shared GPU MSM driver for the BN254 kernels in `tools/gen_bn254_kernels.py`.
//!
//! Included by `tests/zk_gpu_msm.rs` (which checks it against arkworks) and by
//! `tests/zk_gpu_groth16.rs` (which runs a real Groth16 prover on top of it).
//! It lives in a module rather than in `src/` because every type in its
//! signature is an arkworks type, and arkworks is deliberately a DEV-only
//! dependency of this repo — nothing in the shipping `Y` binary links it.
#![allow(dead_code)]
use std::path::{Path, PathBuf};
use std::process::Command;

use ark_bn254::{Fq, Fr, G1Projective};
use ark_ec::{AdditiveGroup, CurveGroup};
use ark_ff::{Field, PrimeField, Zero};

use y::cuda_runtime::{CudaContext, KernelModule};

pub fn bin() -> PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("Y")
}

pub fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Compiling the same `.ysu` from several test threads at once races on the
/// `.ptx` output path: one thread reads the file while another is still
/// writing it, and the JIT rejects the torn result with "Can't load this
/// binary kind". Compile each kernel once per process, under a lock.
pub fn ptx_for(entry: &str) -> String {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<std::collections::HashMap<String, String>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = guard.get(entry) {
        return p.clone();
    }
    // Compile a COPY in a per-process temp directory. The mutex above makes
    // this safe WITHIN a process and cannot help ACROSS them: `cargo test`
    // runs the test binaries in parallel, and three of them compile
    // `bn254_fr_mul_fast.ysu` — this file, `zk_gpu_field.rs`, and
    // `zk_gpu_groth16.rs` through `common/qap.rs`. `--emit-ptx` writes next
    // to its source, so in-place compilation had them truncating and reading
    // one repo path at once. Observed as
    // `the_v4_kernel_emitted_no_loads` after several clean runs, which is the
    // documented signature of this race. Same fix `committed_ptx_artifacts.rs`
    // already uses; `current_dir` stays the repo so `.ysu_hw_profile` is still
    // found.
    let dir = std::env::temp_dir().join(format!("y_ptx_{}_{}", std::process::id(), entry));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir for the emitted PTX");
    let src = dir.join(format!("{}.ysu", entry));
    std::fs::copy(repo().join(format!("tests/{}.ysu", entry)), &src)
        .expect("copy the kernel source");
    let out = Command::new(bin())
        .arg(&src)
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
    let ptx = std::fs::read_to_string(dir.join(format!("{}.ptx", entry)))
        .expect("no .ptx written");
    let _ = std::fs::remove_dir_all(&dir);
    guard.insert(entry.to_string(), ptx.clone());
    ptx
}

pub fn load_kernel(ctx: &CudaContext, entry: &str) -> KernelModule {
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
pub fn r_mod_q() -> Fq {
    let mut r = Fq::from(1u64);
    for _ in 0..256 {
        r = r + r;
    }
    r
}

pub fn limbs32(x: &Fq) -> [u32; 8] {
    let l = x.into_bigint().0;
    let mut o = [0u32; 8];
    for i in 0..4 {
        o[2 * i] = l[i] as u32;
        o[2 * i + 1] = (l[i] >> 32) as u32;
    }
    o
}

pub fn from_limbs32(w: &[u32]) -> Fq {
    let mut b = [0u64; 4];
    for i in 0..4 {
        b[i] = w[2 * i] as u64 | ((w[2 * i + 1] as u64) << 32);
    }
    Fq::from_bigint(ark_ff::BigInt(b)).expect("limbs are not a canonical Fq element")
}

/// Planar arrays of `uint4`, the layout every kernel in this series reads.
pub fn to_planar(v: &[Fq], n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n * 8];
    for (i, x) in v.iter().enumerate() {
        let l = limbs32(x);
        for j in 0..8 {
            out[(j / 4) * 4 * n + i * 4 + (j % 4)] = l[j];
        }
    }
    out
}

pub fn from_planar(raw: &[u32], n: usize) -> Vec<Fq> {
    (0..n)
        .map(|i| {
            let w: Vec<u32> = (0..8)
                .map(|j| raw[(j / 4) * 4 * n + i * 4 + (j % 4)])
                .collect();
            from_limbs32(&w)
        })
        .collect()
}

pub fn as_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
pub trait OnCurve {
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

// ---------------------------------------------------------------------------
// Pippenger MSM
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Pippenger MSM: the host half lives in the crate that SHIPS
// ---------------------------------------------------------------------------

// `Geom`, the window geometry, the digit extraction and the whole parallel
// counting sort used to be duplicated here, and the copy in this file was the
// one that received every measured improvement: the parallel prefix
// (23.6 -> 3.3 ms), the grouped scatter and its thread count (2.2x at
// nw = 20), the removal of a per-thread `to_vec`, the 128-thread block, and
// the exactly-once post-condition. `crates/y-gpu` — the crate a caller
// actually links — still had the version from before all of it.
//
// That is the defect this repository exists to remove, one layer out from the
// emitter: a second description drifts, and it drifts silently because no test
// compares the two. It also made `proofs/CountingSort.v` a proof about code
// that does not ship — `tests/msm_counting_sort_model.rs` tied the theorems to
// THIS file, and the library ran something else.
//
// So there is one implementation now and it is the shipped one. What stays
// here is the harness the crate cannot have: locating and running the `Y`
// binary, compiling a `.ysu` to PTX, and the device layer's panic-on-error
// contract, which a library must not adopt.
pub use y_gpu::msm::{
    bin_by_digit, bucket_block, bucket_launch_geometry, gpu_is_worth_it, last_bin_trace, last_scatter_shape, pick_windows,
    set_bucket_block, set_scatter_threads, window_digit, BinTrace, Geom, MSM_GPU_MIN_COLD,
    MSM_GPU_MIN_STAGED,
};

/// Where the wall clock went. Reported separately because the lesson from the
/// NTT applies verbatim here: the first end-to-end figure measured there was
/// 96% marshalling, and a single number would have hidden it.
#[derive(Default, Clone, Copy)]
pub struct MsmTiming {
    pub bin: f64,
    pub stage: f64,
    pub h2d: f64,
    pub kernel: f64,
    pub d2h: f64,
    pub reduce: f64,
    pub nidx: usize,
}

impl MsmTiming {
    pub fn total(&self) -> f64 {
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
    pub fn steady_state(&self) -> f64 {
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
pub fn gpu_msm(
    ctx: &CudaContext,
    module: &KernelModule,
    points: &[G1Projective],
    scalars: &[Fr],
    g: &Geom,
) -> (G1Projective, MsmTiming) {
    let (buckets, _off, tm) = bucket_pass(ctx, module, points, scalars, g);
    reduce_buckets(&buckets, g, tm)
}

/// The same MSM over bases that already live on the device.
pub fn gpu_msm_staged(
    ctx: &CudaContext,
    module: &KernelModule,
    bases: &DeviceBases,
    scalars: &[Fr],
    g: &Geom,
) -> (G1Projective, MsmTiming) {
    let (buckets, _off, tm) = bucket_pass_staged(ctx, module, bases, scalars, g);
    reduce_buckets(&buckets, g, tm)
}

/// `sum_b d(b) * B[b]`, then the window combination.
pub fn reduce_buckets(
    buckets: &[G1Projective],
    g: &Geom,
    mut tm: MsmTiming,
) -> (G1Projective, MsmTiming) {
    use std::time::Instant;
    // Each window's `sum_d d * B[d]` is independent, so they run in parallel
    // and only the final combination is serial.
    let t = Instant::now();
    let sums: Vec<G1Projective> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..g.nw)
            .map(|w| {
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
pub fn stage_planar(points: &[G1Projective], f: fn(&G1Projective) -> Fq, r: Fq) -> Vec<u32> {
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
pub fn bucket_pass(
    ctx: &CudaContext,
    module: &KernelModule,
    points: &[G1Projective],
    scalars: &[Fr],
    g: &Geom,
) -> (Vec<G1Projective>, Vec<u32>, MsmTiming) {
    let bases = stage_bases(ctx, points);
    let (b, o, mut tm) = bucket_pass_staged(ctx, module, &bases, scalars, g);
    tm.stage = bases.stage;
    tm.h2d += bases.h2d;
    (b, o, tm)
}

/// Point bases that already live on the device, in the planar `uint4` layout.
///
/// A Groth16 proving key is fixed for the lifetime of a circuit while the
/// scalars change every proof, so Montgomery-staging the bases and pushing
/// them across PCIe is key-load work. Holding them here is what makes the
/// `steady_state` column of the MSM benchmark reachable in practice rather
/// than merely arithmetic.
pub struct DeviceBases {
    pub dpx: y::cuda_runtime::DeviceBuffer,
    pub dpy: y::cuda_runtime::DeviceBuffer,
    pub dpz: y::cuda_runtime::DeviceBuffer,
    pub n: usize,
    pub stage: f64,
    pub h2d: f64,
}

pub fn stage_bases(ctx: &CudaContext, points: &[G1Projective]) -> DeviceBases {
    use std::time::Instant;
    let r = r_mod_q();
    let t = Instant::now();
    let (hx, hy, hz) = (
        stage_planar(points, |p| p.x, r),
        stage_planar(points, |p| p.y, r),
        stage_planar(points, |p| p.z, r),
    );
    let stage = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let up = |v: &[u32]| {
        let d = ctx.alloc(v.len().max(1) * 4).unwrap();
        ctx.memcpy_htod_at(&d, 0, as_bytes(v)).unwrap();
        d
    };
    let (dpx, dpy, dpz) = (up(&hx), up(&hy), up(&hz));
    ctx.synchronize().unwrap();
    DeviceBases { dpx, dpy, dpz, n: points.len(), stage, h2d: t.elapsed().as_secs_f64() }
}

/// The bucket pass over bases already on the device: bin, upload the CSR pair,
/// launch, read back.
pub fn bucket_pass_staged(
    ctx: &CudaContext,
    module: &KernelModule,
    bases: &DeviceBases,
    scalars: &[Fr],
    g: &Geom,
) -> (Vec<G1Projective>, Vec<u32>, MsmTiming) {
    use std::time::Instant;
    let mut tm = MsmTiming::default();
    let n = bases.n;
    let r = r_mod_q();

    let t = Instant::now();
    let (idx, off) = bin_by_digit(scalars, g);
    tm.bin = t.elapsed().as_secs_f64();
    tm.nidx = idx.len();

    let t = Instant::now();
    let up = |v: &[u32]| {
        let d = ctx.alloc(v.len().max(1) * 4).unwrap();
        ctx.memcpy_htod_at(&d, 0, as_bytes(v)).unwrap();
        d
    };
    let (dpx, dpy, dpz) = (&bases.dpx, &bases.dpy, &bases.dpz);
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
    ctx.launch(
        module,
        ((g.nb / bucket_block()) as u32, 1, 1),
        (bucket_block() as u32, 1, 1),
        0,
        &args,
    )
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

