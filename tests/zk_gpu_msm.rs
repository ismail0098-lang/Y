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

#[path = "common/msm.rs"]
mod msm;
use msm::*;

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::{AdditiveGroup, CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_ff::{Field, UniformRand, Zero};
use ark_std::rand::SeedableRng;

use y::cuda_runtime::CudaContext;

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
/// The best CPU MSM available, and what it cost.
///
/// Takes the better of the monolithic call and the chunked one, because
/// NEITHER dominates: chunking wins slightly at 2^20 (279 vs 342 ms) and loses
/// badly at small n, where it splits into 32 Pippengers of a few hundred
/// points each. Timing against only the chunked version at n = 14,000 reported
/// 9.72 ms where the monolithic call takes ~5.5 ms -- enough to make the GPU
/// look like it wins at a size where it loses, which is how
/// `the_dispatch_thresholds_are_still_true` first failed.
fn cpu_msm_best(points: &[G1Affine], scalars: &[Fr], threads: usize) -> (G1Projective, f64) {
    use std::time::Instant;
    let t = Instant::now();
    let a = G1Projective::msm(points, scalars).expect("msm");
    let mono = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let b = cpu_msm_all_cores(points, scalars, threads);
    let chunked = t.elapsed().as_secs_f64();
    assert_eq!(a, b, "the two CPU baselines disagree");
    (a, mono.min(chunked))
}

/// Refuse to compare a CPU against a GPU in a debug build.
///
/// `cargo test` is a DEBUG build, and a debug build unoptimises exactly one side
/// of this comparison. The GPU work is PTX that `ptxas` compiled and the device
/// executes: it runs at full speed no matter how the host was built. arkworks'
/// MSM is Rust that rustc just compiled with `-O0`, and it is roughly an order
/// of magnitude slower than it should be. So the measured crossover moves by
/// that factor and the GPU appears to win at sizes where it loses badly.
///
/// This is not a stale constant, which is what it looked like:
/// `the_dispatch_thresholds_are_still_true` reported the GPU faster at n=14,000
/// (42 ms vs 56 ms) and suggested `MSM_GPU_MIN_COLD` was too high, while the same
/// test passes in release and the release sweep puts the crossover at ~70,000 -
/// right where the constant says. Skipping is the honest outcome: the machine
/// cannot answer this question in this build mode. Same shape as the `ptxas` and
/// `solc` skips elsewhere in the suite.
fn require_release_build(what: &str) -> bool {
    if cfg!(debug_assertions) {
        eprintln!(
            "SKIP: {} compares a CPU against a GPU, and a debug build unoptimises \
             only the CPU side. Re-run with --release.",
            what
        );
        return false;
    }
    true
}

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
    if !require_release_build("what_the_gpu_msm_costs") {
        return;
    }
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_msm_bucket");
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1);
    let mut summary: Vec<(usize, f64, f64, f64, f64)> = Vec::new();

    for log_n in [14usize, 16, 18, 20, 22] {
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
        // NOT one core: arkworks' msm is parallel here via feature unification
        // (~12.5 cores at n = 2^20). See cpu_msm_all_cores' docstring.
        println!("  arkworks monolithic = {:.1} ms  (internally parallel, ~12.5 cores)", one_core * 1e3);
        println!("  arkworks chunked {:>2}x = {:.1} ms", threads, all_cores * 1e3);
        let cpu_best = one_core.min(all_cores);
        println!(
            "  cold end-to-end     = {:.1} ms -> {:.2}x the best CPU",
            total * 1e3, cpu_best / total
        );
        println!(
            "  fixed bases         = {:.1} ms -> {:.2}x the best CPU   <- what a prover pays per proof",
            tm.steady_state() * 1e3, cpu_best / tm.steady_state()
        );
        println!(
            "  kernel only         = {:.1} ms -> {:.2}x the best CPU",
            tm.kernel * 1e3, cpu_best / tm.kernel
        );
        summary.push((log_n, cpu_best, total, tm.steady_state(), tm.kernel));
    }

    println!("\n=== how the MSM speedup scales ===");
    println!("{:>5} {:>10} {:>10} {:>9} {:>9} {:>9}", "n", "cpu ms", "gpu ms", "cold", "fixed", "kernel");
    for (l, cpu, cold, fixed, kern) in &summary {
        println!(
            "2^{:<3} {:10.1} {:10.1} {:8.2}x {:8.2}x {:8.2}x",
            l, cpu * 1e3, cold * 1e3, cpu / cold, cpu / fixed, cpu / kern
        );
    }
    println!("\nA ratio that stops growing means both sides have reached their\nasymptotic cost per point; a ratio that falls means the GPU has hit a\nlimit (bandwidth or memory) the CPU has not.");
}

/// `cargo test --release --features zk --test zk_gpu_msm where_the_gpu_starts_winning -- --ignored --nocapture`
///
/// Locates the crossover finely, for both the cold path and the path where the
/// bases already live on the device. They are different thresholds because
/// staging is a per-call cost the second one does not pay, and a single
/// threshold would be wrong for one of them.
#[test]
#[ignore]
fn where_the_gpu_starts_winning() {
    if !require_release_build("where_the_gpu_starts_winning") {
        return;
    }
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_msm_bucket");
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1);

    println!(
        "\n{:>8} {:>6} {:>10} {:>10} {:>9} {:>10} {:>9}",
        "n", "nw", "cpu ms", "gpu cold", "cold x", "gpu warm", "warm x"
    );
    for log_n in 11..=18usize {
        let n = 1usize << log_n;
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0x5EED + log_n as u64);
        let points: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let affine: Vec<G1Affine> = points.iter().map(|p| p.into_affine()).collect();

        // Pick the geometry the dispatcher would pick, not the best one in
        // hindsight -- a threshold tuned against an oracle is not a threshold.
        let g = Geom::new(pick_windows(n));

        let _ = gpu_msm(&ctx, &module, &points, &scalars, &g);
        let bases = stage_bases(&ctx, &points);

        let (mut cold, mut warm, mut cpu) = (f64::MAX, f64::MAX, f64::MAX);
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let _ = gpu_msm(&ctx, &module, &points, &scalars, &g);
            cold = cold.min(t.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            let _ = gpu_msm_staged(&ctx, &module, &bases, &scalars, &g);
            warm = warm.min(t.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            let a = G1Projective::msm(&affine, &scalars).unwrap();
            let e1 = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let b = cpu_msm_all_cores(&affine, &scalars, threads);
            let e2 = t.elapsed().as_secs_f64();
            assert_eq!(a, b);
            cpu = cpu.min(e1.min(e2));
        }
        println!(
            "{:>8} {:>6} {:10.2} {:10.2} {:8.2}x {:10.2} {:8.2}x",
            n, g.nw, cpu * 1e3, cold * 1e3, cpu / cold, warm * 1e3, cpu / warm
        );
    }
    println!("\n  The crossover is where a column reaches 1.00x. Below it the GPU\n  is the slower choice and a dispatcher should not use it.");
}

/// The dispatch thresholds are measured constants, and a measured constant
/// with no test is a magic number waiting to go stale. This asserts the
/// direction of the inequality still holds on whatever machine is running it:
/// comfortably below the threshold the CPU must win, comfortably above it the
/// GPU must.
///
/// The 2x margins are deliberate. Asserting the crossover sits at exactly
/// 56,000 would fail on any hardware change and teach nothing; asserting the
/// SHAPE is right is what catches a threshold that has become wrong.
#[test]
fn the_dispatch_thresholds_are_still_true() {
    if !require_release_build("the_dispatch_thresholds_are_still_true") {
        return;
    }
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_msm_bucket");
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1);

    let time_both = |n: usize| -> (f64, f64) {
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0xD15 + n as u64);
        let points: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let affine: Vec<G1Affine> = points.iter().map(|p| p.into_affine()).collect();
        let g = Geom::new(pick_windows(n));
        let _ = gpu_msm(&ctx, &module, &points, &scalars, &g); // warm the clock
        let (mut gpu, mut cpu) = (f64::MAX, f64::MAX);
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let _ = gpu_msm(&ctx, &module, &points, &scalars, &g);
            gpu = gpu.min(t.elapsed().as_secs_f64());
            // The BEST CPU, not just the chunked one -- chunking is a poor
            // baseline at small n and that is exactly where this test looks.
            let (_, e) = cpu_msm_best(&affine, &scalars, threads);
            cpu = cpu.min(e);
        }
        (cpu, gpu)
    };

    let small = MSM_GPU_MIN_COLD / 4;
    let (cpu_s, gpu_s) = time_both(small);
    assert!(
        cpu_s < gpu_s,
        "at n={} (well under the {} threshold) the GPU was FASTER ({:.2} ms vs CPU {:.2} ms) \
         -- MSM_GPU_MIN_COLD is too high on this machine and the dispatcher is leaving \
         performance on the table",
        small, MSM_GPU_MIN_COLD, gpu_s * 1e3, cpu_s * 1e3
    );

    let large = MSM_GPU_MIN_COLD * 4;
    let (cpu_l, gpu_l) = time_both(large);
    assert!(
        gpu_l < cpu_l,
        "at n={} (well over the {} threshold) the CPU was FASTER ({:.2} ms vs GPU {:.2} ms) \
         -- MSM_GPU_MIN_COLD is too low on this machine and the dispatcher is choosing \
         the slower backend",
        large, MSM_GPU_MIN_COLD, cpu_l * 1e3, gpu_l * 1e3
    );

    assert!(
        !gpu_is_worth_it(small, false) && gpu_is_worth_it(large, false),
        "gpu_is_worth_it disagrees with the constants it is built from"
    );
}
