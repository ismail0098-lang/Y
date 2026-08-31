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
pub struct Geom {
    pub widths: Vec<usize>,
    pub shifts: Vec<usize>,
    /// Bucket index at which each window starts; `base[nw]` is the total.
    pub base: Vec<usize>,
    pub nw: usize,
    pub nb: usize,
}

/// Threads per CTA for `bn254_msm_bucket`, and the granularity the bucket
/// count must be a multiple of.
///
/// **This is an occupancy knob, not a tuning preference.** NCU says the kernel
/// is latency-bound — 70% of stalls are `wait` (a dependent ALU result), DRAM
/// under 9%, SM throughput under 27% — and it uses 136 registers per thread.
/// A 256-thread CTA therefore needs 34,816 registers and only ONE block fits
/// in an SM's 65,536, capping occupancy at 256/1536 = 16.7%. Halving the block
/// doubles the blocks per SM and the occupancy with it.
///
/// `Y_MSM_BLOCK` sets it for a whole process; `set_bucket_block` sets it for
/// the calling thread only, which is what `what_the_msm_block_size_costs`
/// uses — mutating the environment from a test races every other test in the
/// binary, and this one shares a process with the correctness suite.
pub fn bucket_block() -> usize {
    OVERRIDE.with(|o| o.get()).unwrap_or_else(|| {
        std::env::var("Y_MSM_BLOCK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|b| legal_block(*b))
            .unwrap_or(BUCKET_BLOCK_DEFAULT)
    })
}

/// Set the block size for this thread. `None` restores the default.
pub fn set_bucket_block(b: Option<usize>) {
    if let Some(v) = b {
        assert!(legal_block(v), "illegal MSM block size {}", v);
    }
    OVERRIDE.with(|o| o.set(b));
}

fn legal_block(b: usize) -> bool {
    (32..=1024).contains(&b) && b % 32 == 0
}

thread_local! {
    static OVERRIDE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// 128, not 256, and the difference is measured rather than assumed: at 136
/// registers a 256-thread CTA needs 34,816 registers and only ONE block fits
/// in an SM's 65,536, capping occupancy at 16.7%. Halving it doubles the
/// blocks per SM.
///
/// **The size of the win depends entirely on the window geometry, and the
/// first version of this measurement missed that.** Swept at n = 2^20
/// (`what_the_msm_block_size_costs`), kernel ms at 256 -> 128:
///
/// | nw | buckets | threads/SM | 256 | 128 | |
/// |---|---|---|---|---|---|
/// | 20 | 139,264 | 2110 | 36.6 | 25.6 | 1.43x |
/// | 22 |  69,632 | 1055 | 45.0 | 29.5 | 1.53x |
/// | 25 |  29,696 |  450 | 53.8 | 42.1 | 1.28x |
/// | 28 |  15,360 |  233 | 61.6 | 60.0 | 1.03x |
/// | 31 |   9,472 |  144 | 119.8| 114.3| 1.05x |
///
/// Below ~450 threads per SM the kernel is THREAD-STARVED — there are not
/// enough buckets to fill the machine — and no block size helps. Above it the
/// kernel is REGISTER-limited and a smaller block buys occupancy directly.
/// Measuring only `nw = 28` reported a flat 1.00x while the end-to-end run
/// showed 1.18x; sweep the geometry too.
///
/// Going below 128 is flat to within noise at every geometry measured, so 128
/// is chosen as the largest size that gets the whole effect.
const BUCKET_BLOCK_DEFAULT: usize = 128;

impl Geom {
    pub fn new(nw: usize) -> Self {
        let q = 254 / nw;
        let r = 254 % nw;
        assert!(q >= 8, "windows narrower than 8 bits break the bucket blocking");
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
        assert_eq!(b % bucket_block(), 0, "bucket count must fill whole blocks");
        Geom { widths, shifts, base, nw, nb: b }
    }

    pub fn avg_width(&self) -> f64 {
        254.0 / self.nw as f64
    }
}

/// Below these sizes the GPU MSM is the SLOWER choice and a dispatcher should
/// not use it. Measured by `where_the_gpu_starts_winning`, which prints the
/// table these are read off: cold crosses 1.00x between 32,768 (0.81x) and
/// 65,536 (1.07x); staged crosses between 32,768 (0.88x) and 65,536 (1.30x).
///
/// Two thresholds rather than one because staging the bases is a per-call cost
/// the warm path does not pay, so a single number would be wrong for one of
/// them — and wrong in the direction of using the GPU when it loses.
///
/// **These encode this machine.** An RTX 4070 Ti SUPER against a 32-thread
/// CPU; a faster GPU or a smaller CPU moves them. `Y_MSM_GPU_MIN` overrides,
/// and `the_dispatch_thresholds_are_still_true` fails if the hardware has
/// drifted far enough to invalidate them — which is the only thing separating
/// a measured constant from a magic number.
pub const MSM_GPU_MIN_COLD: usize = 56_000;
pub const MSM_GPU_MIN_STAGED: usize = 40_000;

/// Is the GPU worth using for an MSM of `n` terms?
pub fn gpu_is_worth_it(n: usize, staged: bool) -> bool {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    let ov = OVERRIDE.get_or_init(|| {
        std::env::var("Y_MSM_GPU_MIN").ok().and_then(|v| v.parse().ok())
    });
    let min = ov.unwrap_or(if staged { MSM_GPU_MIN_STAGED } else { MSM_GPU_MIN_COLD });
    n >= min
}

/// The window count to use at a given problem size.
///
/// Fewer windows means less GPU work but more buckets, and the bucket count is
/// what the host pays for (binning's cursor table, the readback, the
/// reduction). At small `n` those fixed costs dominate, so narrow windows win;
/// at large `n` the kernel dominates and wide ones do. Measured from the sweep
/// in `what_the_gpu_msm_costs`, which prints the whole table it is fitted to.
pub fn pick_windows(n: usize) -> usize {
    match n {
        0..=65_536 => 31,
        65_537..=262_144 => 28,
        _ => 25,
    }
}

/// The `width`-bit field of a little-endian 4-limb scalar starting at `shift`.
/// Written over limbs rather than over a bit vector because the bit vector is
/// 254 pushes per scalar and this is on the measured path.
pub fn window_digit(l: &[u64; 4], shift: usize, width: usize) -> u32 {
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
/// Per-phase wall time for `bin_by_digit`. Binning is the largest host phase
/// and the largest single obstacle to a wider window geometry, and "it is the
/// cursor table" was an attribution nobody had measured — so measure it.
#[derive(Default, Clone, Copy, Debug)]
pub struct BinTrace {
    pub histogram: f64,
    /// Global bucket offsets only. The per-group cursor table the scatter
    /// needs is charged to the scatter, since it exists solely for it.
    pub prefix: f64,
    pub scatter: f64,
}

/// The shape of the most recent scatter: `(ngroup, span)` — how many writers
/// shared the destination, and how many POINTS each of them owned.
///
/// Recorded by `scatter` itself rather than recomputed by the caller. Both
/// values are already derived there and re-deriving them anywhere else would
/// be a second copy of `ceil(nchunk / group)` — the defect
/// `proofs/ExactGemmSchedule.v` exists to remove, in the harness instead of
/// the emitter. `tests/msm_counting_sort_model.rs` needs them to evaluate the
/// two-level destination map on real data, and to assert it is genuinely two
/// levels rather than an `ngroup == 1` degenerate case.
thread_local! {
    static LAST_SHAPE: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

/// `(ngroup, span)` for the most recent `bin_by_digit` on this thread.
pub fn last_scatter_shape() -> (usize, usize) {
    LAST_SHAPE.with(|c| c.get())
}

thread_local! {
    static LAST_TRACE: std::cell::Cell<BinTrace> =
        const { std::cell::Cell::new(BinTrace {
            histogram: 0.0, prefix: 0.0, scatter: 0.0,
        }) };
    static SCATTER_THREADS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// The phase split of the most recent `bin_by_digit` call on this thread.
pub fn last_bin_trace() -> BinTrace {
    LAST_TRACE.with(|t| t.get())
}

/// Force the scatter's thread count for this thread. `None` restores the
/// measured default. Used by `what_the_scatter_is_actually_bound_by`, which is
/// what the default was read off in the first place.
pub fn set_scatter_threads(v: Option<usize>) {
    SCATTER_THREADS.with(|c| c.set(v));
}


pub fn bin_by_digit(scalars: &[Fr], g: &Geom) -> (Vec<u32>, Vec<u32>) {
    let n = scalars.len();
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(n.max(1));
    let chunk = n.div_ceil(threads.max(1)).max(1);
    let nchunk = n.div_ceil(chunk);

    let t_hist = std::time::Instant::now();
    // Phase 1: a private histogram per chunk. This phase DOES want every
    // thread — its destination is private, so none of the sharing that limits
    // the scatter applies, and it is 3-4 ms at any geometry.
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

    let d_hist = t_hist.elapsed().as_secs_f64();
    let t_prefix = std::time::Instant::now();

    // Phase 2: global bucket offsets. Parallel per-bucket totals, then one
    // serial exclusive prefix that is only `nb` long.
    let mut off = vec![0u32; g.nb + 1];
    let mut tot = vec![0u32; g.nb];
    let bchunk = g.nb.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ti, part) in tot.chunks_mut(bchunk).enumerate() {
            let counts = &counts;
            let b0 = ti * bchunk;
            s.spawn(move || {
                for (j, slot) in part.iter_mut().enumerate() {
                    let b = b0 + j;
                    *slot = counts.iter().map(|c| c[b]).sum();
                }
            });
        }
    });
    let mut running = 0u32;
    for b in 0..g.nb {
        off[b] = running;
        running += tot[b];
    }
    off[g.nb] = running;

    let d_prefix = t_prefix.elapsed().as_secs_f64();
    let t_scatter = std::time::Instant::now();

    // Phase 3: scatter, over GROUPS of the histogram's chunks — see
    // `scatter_threads`.
    let group = nchunk.div_ceil(scatter_threads(g, nchunk));
    let idx = scatter(scalars, g, &counts, &off, chunk, group, running as usize);

    LAST_TRACE.with(|t| {
        t.set(BinTrace {
            histogram: d_hist,
            prefix: d_prefix,
            scatter: t_scatter.elapsed().as_secs_f64(),
        })
    });
    (idx, off)
}

/// How many threads the scatter runs on — deliberately FEWER than the machine
/// has, which is the opposite of what every other phase here wants.
///
/// Measured by `what_the_scatter_is_actually_bound_by`, which sweeps the thread
/// count at three sizes and four geometries. Scatter ms, n = 2^22:
///
/// | nw | buckets | 32 | 16 | 12 | 8 | 6 | 4 |
/// |---|---|---|---|---|---|---|---|
/// | 20 | 139,264 | 185.4 | 132.1 | 103.3 | 84.1 | **82.8** | 83.4 |
/// | 22 |  69,632 | 137.8 |  66.4 | **55.1** | 58.2 | 67.8 | 80.9 |
/// | 25 |  29,696 |  68.6 | **47.8** | 51.5 | 56.4 | 68.2 | 80.1 |
/// | 31 |   9,472 |  66.2 | **54.7** | 59.1 | 63.5 | 79.1 | 95.5 |
///
/// **32 threads is never the optimum, and at nw = 20 it is 2.2x off it.** The
/// per-entry cost falls monotonically as threads are removed — 70.7 ns at 32
/// threads down to 4.0 ns at 4 — so the scatter loses more to running wide than
/// it gains from it, until there are simply not enough cores left.
///
/// **The cause is sharing, not capacity, and the thread sweep is what
/// separates them.** The destination working set is `nb` cache lines however
/// many threads walk it, so a capacity story predicts a flat per-entry cost
/// across a row; it collapses by 17x instead. What changes with the thread
/// count is the length of a thread's contiguous run inside one bucket — 75
/// bytes at 32 threads and nw = 20, so two cores write the same 64-byte line,
/// against 402 bytes at 6, where they do not.
///
/// The optimum tracks `nb` and not `n` — the two sizes swept agree to within
/// one step at every geometry — which is why the table below is keyed on the
/// bucket count. It is absolute thread counts for a 32-thread machine, clamped
/// to what is available.
///
/// **A radix-partitioned scatter was built and measured before this**, on the
/// theory that the destination working set was the problem: pass A streams
/// `(bucket, point)` pairs into a few hundred per-thread buffers, pass B gives
/// each partition to one thread so no two threads share a line at all. It
/// LOSES at every geometry — 0.94x at nw = 20 down to 0.38x at nw = 31, and
/// 1.11x at its best partition size against 32-thread one-pass, which is still
/// well behind simply using 6 threads. It moves three times the bytes to avoid
/// misses that mostly were not happening, because `nb * 64` is 8.9 MB at worst
/// and this machine has 64 MB of L3. Do not rebuild it.
fn scatter_threads(g: &Geom, nchunk: usize) -> usize {
    let want = SCATTER_THREADS
        .with(|c| c.get())
        .or_else(|| {
            std::env::var("Y_MSM_SCATTER_THREADS").ok().and_then(|v| v.parse().ok())
        })
        .unwrap_or(match g.nb {
            0..=16_384 => 16,
            16_385..=49_152 => 12,
            49_153..=98_304 => 10,
            _ => 7,
        });
    want.min(nchunk).max(1)
}

/// Scatter into `idx`, one write per (point, window).
///
/// `group` consecutive histogram chunks are handled by one thread, so the
/// number of writers per bucket is `nchunk / group` rather than `nchunk`. That
/// is the whole tuning surface; see `scatter_threads` for why it is a knob at
/// all.
///
/// Swapping the loops to window-outer looks like the obvious locality fix — a
/// point-outer pass writes into every window's slice of `idx` at once,
/// spanning 84 MB at n = 2^20 and nw = 20, where a window-outer pass would
/// confine itself to one window's ~4 MB. **It measured WORSE** — 22.1 -> 33.1
/// ms at nw = 25 — and flat across geometries instead of growing with them,
/// because it marches every thread onto the same cache lines at the same time
/// instead of letting them drift across windows independently. Reducing the
/// writer count is the fix that works; reordering is not.
fn scatter(
    scalars: &[Fr],
    g: &Geom,
    counts: &[Vec<u32>],
    off: &[u32],
    chunk: usize,
    group: usize,
    total: usize,
) -> Vec<u32> {
    let nchunk = counts.len();
    let ngroup = nchunk.div_ceil(group);
    LAST_SHAPE.with(|c| c.set((ngroup, chunk * group)));

    // Both writes below are through a raw pointer shared across threads, so
    // the preconditions that make them disjoint and in bounds are asserted
    // rather than argued. Neither can fire at any size this repo runs, which
    // is exactly why they would otherwise go unnoticed if a caller changed.
    //
    // (a) The scatter zips `scalars.chunks(chunk * group)` against
    //     `cursor.chunks_mut(nb)`. Those lengths are equal by
    //     `ceil(ceil(n/c)/g) == ceil(n/(c*g))` — but `zip` TRUNCATES, so if a
    //     future change to how `chunk` is derived broke that identity, the
    //     tail of the input would be silently dropped: no crash, no panic,
    //     just a bucket set missing points and an MSM that is quietly wrong.
    assert_eq!(
        scalars.len().div_ceil(chunk * group),
        ngroup,
        "scatter grouping does not tile the input"
    );
    // (b) `off` is `u32`, so a circuit with more than 4.29e9 (point, window)
    //     entries wraps. `total` would then under-allocate `idx` while the
    //     cursors still point past its end — an out-of-bounds WRITE, not a
    //     wrong answer. Unreachable below n ~ 1.4e8 points, and unchecked
    //     until now.
    assert_eq!(
        total,
        off[g.nb] as usize,
        "bucket offsets disagree with the entry total"
    );
    assert!(
        (scalars.len() as u64) * (g.nw as u64) < u32::MAX as u64,
        "entry count overflows the u32 bucket offsets"
    );

    // Where each GROUP's slice of each bucket begins. One row per scatter
    // thread, not one per histogram chunk, so this table shrinks with `group`
    // too — 17.8 MB at nw = 20 with 32 rows, 3.3 MB with 6.
    // A second copy is written here rather than cloned afterwards: the
    // scatter advances `cursor` in place, so the starts have to be kept to
    // check the ends against, and a `clone()` of it measured 1.34 ms at
    // nw = 20 — almost all of it first-touch page faults on a fresh 3.9 MB
    // allocation. Filling it inside this loop distributes that over the same
    // threads for free.
    let mut cursor = vec![0u32; ngroup * g.nb];
    let mut starts = vec![0u32; ngroup * g.nb];
    {
        struct Shared(*mut u32, *mut u32);
        unsafe impl Sync for Shared {}
        let shared = Shared(cursor.as_mut_ptr(), starts.as_mut_ptr());
        let nb = g.nb;
        let bchunk = g.nb.div_ceil(nchunk.max(1)).max(1);
        std::thread::scope(|s| {
            for ti in 0..g.nb.div_ceil(bchunk) {
                let shared = &shared;
                s.spawn(move || {
                    for b in (ti * bchunk)..((ti + 1) * bchunk).min(nb) {
                        let mut run = off[b];
                        for (gi, rows) in counts.chunks(group).enumerate() {
                            unsafe {
                                *shared.0.add(gi * nb + b) = run;
                                *shared.1.add(gi * nb + b) = run;
                            }
                            run += rows.iter().map(|c| c[b]).sum::<u32>();
                        }
                    }
                });
            }
        });
    }

    let mut idx = vec![0u32; total];
    struct Shared(*mut u32);
    unsafe impl Sync for Shared {}
    let shared = Shared(idx.as_mut_ptr());
    std::thread::scope(|s| {
        for ((gi, part), cur) in scalars
            .chunks(chunk * group)
            .enumerate()
            .zip(cursor.chunks_mut(g.nb))
        {
            let shared = &shared;
            let base_i = gi * chunk * group;
            s.spawn(move || {
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

    // Post-condition, and it is a PROOF rather than a spot check.
    //
    // Each thread advanced `cursor[gi*nb + b]` once per entry it wrote, so
    // that slot now holds where group `gi` STOPPED. If every group stopped
    // exactly where the next one STARTED — and the last stopped at
    // `off[b+1]` — then the groups' runs tile `off[b]..off[b+1]` exactly:
    // every slot of `idx` was written, by exactly one thread, in bounds. That
    // is precisely what the `unsafe impl Sync` above assumes, and it is
    // otherwise only an argument about the histogram matching the scatter.
    //
    // Note this must compare against the STARTS, not against the next group's
    // post-scatter cursor — for an empty group those differ, and the first
    // version of this check failed on bucket 1 for exactly that reason.
    //
    // Costs `ngroup * nb` compares, run over the same threads and GROUP-outer
    // so both arrays are walked sequentially. The bucket-outer form strides by
    // `nb` and measured 1.45 ms against 0.2.
    {
        let nb = g.nb;
        std::thread::scope(|s| {
            for (gi, ends) in cursor.chunks(nb).enumerate() {
                let starts = &starts;
                s.spawn(move || {
                    let next =
                        if gi + 1 == ngroup { &off[1..nb + 1] } else { &starts[(gi + 1) * nb..(gi + 2) * nb] };
                    for (b, (&end, &want)) in ends.iter().zip(next).enumerate() {
                        assert_eq!(
                            end, want,
                            "scatter group {} of bucket {} wrote {} entries too {}",
                            gi,
                            b,
                            (end as i64 - want as i64).abs(),
                            if end > want { "many" } else { "few" }
                        );
                    }
                });
            }
        });
    }
    idx
}

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

