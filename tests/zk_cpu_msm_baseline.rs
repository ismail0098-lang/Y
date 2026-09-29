//! Is the arkworks MSM baseline actually using every core?
//!
//! This exists because a GPU-vs-CPU ratio is worthless if the CPU side is
//! secretly single-threaded, and this repo has already been burned by that
//! exact mistake once (the NTT's "60.6x" was against one core). Wall time
//! alone cannot tell: a 300 ms MSM looks the same whether it used one core for
//! 300 ms or 32 cores for 9 ms each.
//!
//! So it reports CPU time / wall time, read from the kernel. ~1.0 means one
//! core. ~32 means the whole machine.

use std::time::Instant;

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::UniformRand;
use ark_std::rand::SeedableRng;

/// Total CPU time (user + system) this process has burned, across all threads.
fn cpu_seconds() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").expect("no /proc/self/stat");
    // Fields after the (comm) field, which may itself contain spaces.
    let tail = &s[s.rfind(')').expect("malformed stat") + 2..];
    let f: Vec<&str> = tail.split_whitespace().collect();
    // utime and stime are fields 14 and 15 (1-based, whole line); after the
    // comm split they are indices 11 and 12.
    let ticks: f64 = f[11].parse::<f64>().unwrap() + f[12].parse::<f64>().unwrap();
    ticks / 100.0 // USER_HZ is 100 on Linux
}

#[test]
#[ignore]
fn how_many_cores_does_the_arkworks_msm_use() {
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1);
    println!("\nmachine reports {} hardware threads", threads);

    for log_n in [18usize, 20] {
        let n = 1usize << log_n;
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(1);
        let pts: Vec<G1Affine> = (0..n)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        // Warm up, then measure a single monolithic MSM.
        let _ = G1Projective::msm(&pts, &scalars).unwrap();
        let (c0, w0) = (cpu_seconds(), Instant::now());
        let _ = G1Projective::msm(&pts, &scalars).unwrap();
        let wall = w0.elapsed().as_secs_f64();
        let cpu = cpu_seconds() - c0;

        println!(
            "  n = 2^{:<2}  wall {:7.1} ms   cpu {:7.1} ms   => {:5.2} cores in use",
            log_n,
            wall * 1e3,
            cpu * 1e3,
            cpu / wall
        );
    }
    println!(
        "\n  A ratio near 1.0 means the baseline is ONE CORE and any GPU speedup\n  quoted against it is a per-core number, not a per-machine one."
    );
}
