//! The multi-threaded CPU GEMM paths must compute the right answer, and must
//! terminate.
//!
//! `cpu_gemm_end_to_end.rs` never sets `Y_NUM_THREADS`, and its largest shape
//! is `193x65x257` — 3.2M multiply-adds against a `WORK_PER_THREAD` of 4M. So
//! every shape it runs takes the single-threaded path in `emit_entry`, and
//! **the thread pool, the partition, the barrier and the K-split reduction had
//! no test coverage at all.** That is the same gap `linear_tracker` had: a
//! mechanism whose guarantee had never been checked in either direction.
//!
//! Two failure modes are specific to the threaded paths and invisible to the
//! single-threaded test:
//!
//!   1. **A wrong partition is a wrong answer, not a slow one.** Under the
//!      K-split each thread accumulates a private `M x N` panel over its own
//!      band of K and the panels are summed afterwards; drop a band, overlap
//!      two, or reduce the wrong columns and the result is silently wrong.
//!
//!   2. **A miscounted barrier hangs.** `POOL_BARRIER` is sized to the active
//!      thread count, and exactly the threads with work must arrive at it. The
//!      subtle case is a thread whose *reduction* band is empty but whose
//!      *accumulation* band is not: it still has to arrive. `N = 33` on six
//!      threads is that case — the column bands are snapped to 16, so four of
//!      the six reduce nothing — and a driver that returned early there would
//!      wait forever. Hence the shapes with tiny N, and hence the timeout: a
//!      deadlocked test that merely hangs reports nothing.
//!
//! What this file does NOT prove is that the K-split is the path being taken —
//! the N-split computes the same numbers, just slower. The structural check on
//! the emitted IR below is what ties the test to the intended path; the
//! measured effect lives in `docs/cpu_gemm_tuning.md`.
//!
//! Requires `clang`; skipped with a notice otherwise, like the `ptxas` and
//! `solc` gates elsewhere in the suite.
//!
//! Run with:  cargo test --release --test cpu_gemm_threaded

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const Y_SOURCE: &str = r#"
kernel y_matmul(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, M: I32, N: I32, K: I32) {
    @invariant(i >= 0)
    for i in 0..M step 1 {
        @invariant(j >= 0)
        for j in 0..N step 1 {
            let mut sum: F32 = 0.0;
            @invariant(k >= 0)
            for k in 0..K step 1 {
                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);
                let b_val: F32 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a_val * b_val;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}

fn main() {
}
"#;

/// Shapes big enough that `__y_gemm_threads` asks for more than one thread.
///
/// The comment on each says which property it is there for; none of them is a
/// round number by accident.
const SHAPES: &[(usize, usize, usize, &str)] = &[
    // --- small-M path, K-split (M <= SM_MAX_M and K >= KSPLIT_MIN_K) ---
    (8, 1024, 8192, "decode-like, asks for all 16 threads"),
    (5, 1000, 9001, "ragged M, N and K; last thread takes the K remainder"),
    (1, 4096, 4096, "GEMV through the K-split"),
    (3, 1027, 6000, "N ragged mod 16, so the masked tail runs in both the \
                     kernel and the reduction"),
    (8, 33, 100000, "N smaller than nthr*16: most threads reduce NOTHING but \
                     must still reach the barrier"),
    (8, 15, 200000, "N < 16: the full-width j loop is empty, everything goes \
                     through the masked tail"),
    // --- small-M path, no K-split: K is too shallow, so the N-split runs ---
    (8, 4096, 300, "K < KSPLIT_MIN_K, so this must NOT take the K-split"),
    (2, 8192, 400, "same, narrow M"),
    // --- packed path, multi-threaded: unchanged by this work, but the
    //     partition code it shares was edited, so it is re-checked here ---
    (512, 512, 512, "packed path, M-split"),
    (37, 2048, 1024, "packed path, ragged M"),
    (1024, 96, 512, "packed path, N narrow enough to force the M cut"),
];

/// Thread counts to run every shape under. 3 and 7 are there because an
/// off-by-one in a partition tends to hide behind powers of two.
const THREAD_COUNTS: &[usize] = &[1, 2, 3, 7, 16];

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void y_matmul(const float *A, const float *B, float *C, int M, int N, int K);

static float gen(unsigned long i, unsigned long salt) {
    unsigned long h = i * 2654435761UL + salt * 40503UL;
    h ^= h >> 13;
    return (float)((h & 0xffff) / 65535.0 - 0.5);
}

int main(void) {
    static const int shapes[][3] = { SHAPES_HERE };
    int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));
    int bad = 0;

    for (int s = 0; s < ns; ++s) {
        int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
        float *A = malloc((size_t)M * K * sizeof(float));
        float *B = malloc((size_t)K * N * sizeof(float));
        float *C = malloc((size_t)M * N * sizeof(float));
        float *R = malloc((size_t)M * N * sizeof(float));
        double *Rd = malloc((size_t)M * N * sizeof(double));
        if (!A || !B || !C || !R || !Rd) { printf("ALLOC FAIL\n"); return 2; }

        for (long i = 0; i < (long)M * K; ++i) A[i] = gen(i, 1);
        for (long i = 0; i < (long)K * N; ++i) B[i] = gen(i, 2);
        /* Poison C rather than zeroing it: the kernel is required to WRITE
           every element, and a zeroed buffer would let a skipped tile pass
           whenever the true answer happened to be near zero. */
        for (long i = 0; i < (long)M * N; ++i) C[i] = 12345.0f;

        /* k outer, j inner, accumulating in double.
           The obvious `for i, for j, for k` form strides B by N on its
           innermost index, which is a cache miss per element: it took 85
           seconds per run here and made the test ~100x slower than the kernel
           it is checking. Accumulating in double keeps this an independent
           check despite sharing a summation order with the small-M kernel --
           what the comparison is testing is which products were included, and
           at these sizes double is exact enough that the order cannot hide a
           dropped band. */
        for (long i = 0; i < (long)M * N; ++i) Rd[i] = 0.0;
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                double a = (double)A[(long)i * K + k];
                const float *brow = B + (long)k * N;
                double *rrow = Rd + (long)i * N;
                for (int j = 0; j < N; ++j) rrow[j] += a * (double)brow[j];
            }
        for (long i = 0; i < (long)M * N; ++i) R[i] = (float)Rd[i];

        /* Twice, into the same buffer. The K-split accumulates into per-thread
           scratch that is reused across calls, so a missing zero-fill is
           correct on the first call and wrong on the second. */
        y_matmul(A, B, C, M, N, K);
        y_matmul(A, B, C, M, N, K);

        double num = 0.0, den = 0.0;
        for (long i = 0; i < (long)M * N; ++i) {
            double d = (double)C[i] - (double)R[i];
            num += d * d;
            den += (double)R[i] * (double)R[i];
        }
        double rel = sqrt(num) / (sqrt(den) + 1e-30);
        if (!(rel < 1e-5)) {
            printf("MISMATCH %dx%dx%d relL2=%g\n", M, N, K, rel);
            bad = 1;
        }
        free(A); free(B); free(C); free(R); free(Rd);
    }
    if (!bad) printf("ALL_SHAPES_OK\n");
    return bad;
}
"#;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run `exe` with `Y_NUM_THREADS=nthr`, failing if it has not finished within
/// `limit`. A barrier sized for the wrong number of threads does not crash and
/// does not produce a wrong answer — it hangs, and a hanging test that is
/// merely slow reports nothing at all.
fn run_with_timeout(exe: &PathBuf, nthr: usize, limit: Duration) -> (bool, String) {
    let mut child = Command::new(exe)
        .env("Y_NUM_THREADS", nthr.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the compiled program");

    let start = Instant::now();
    loop {
        match child.try_wait().expect("try_wait") {
            Some(_) => break,
            None => {
                if start.elapsed() > limit {
                    let _ = child.kill();
                    let _ = child.wait();
                    return (
                        false,
                        format!(
                            "TIMED OUT after {:?} at Y_NUM_THREADS={} — the pool \
                             deadlocked (most likely the barrier width and the set \
                             of threads that reach it disagree)",
                            limit, nthr
                        ),
                    );
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
    let out = child.wait_with_output().expect("collect output");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success() && text.contains("ALL_SHAPES_OK"), text)
}

#[test]
fn threaded_cpu_gemm_is_correct_and_terminates() {
    if !have("clang") {
        eprintln!("note: clang not found, skipping threaded CPU GEMM test");
        return;
    }

    let dir = std::env::temp_dir().join(format!("y_cpu_gemm_mt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let src = dir.join("mm.ysu");
    std::fs::write(&src, Y_SOURCE).expect("write Y source");

    let y_bin = repo_root().join("target/release/Y");
    assert!(
        y_bin.exists(),
        "build the compiler first: cargo build --release"
    );

    let out = Command::new(&y_bin)
        .arg(&src)
        .arg("--emit-llvm")
        .current_dir(repo_root())
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "Y failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let ll = dir.join("mm.ll");
    let ir = std::fs::read_to_string(&ll).expect("read emitted IR");

    // Tie the test to the path it is meant to cover. Correctness alone cannot
    // do this: the N-split computes the same numbers, so if the K-split stopped
    // being emitted every assertion below would still pass and the only symptom
    // would be a 4x slowdown on decode shapes.
    assert!(
        ir.contains("define internal void @__y_gemm_small_reduce("),
        "the K-split reduction is not being emitted; the threaded small-M path \
         has silently reverted to the N-split"
    );
    assert!(
        ir.contains("call void @__y_gemm_small_reduce("),
        "__y_gemm_small_reduce is emitted but never called"
    );
    assert!(
        ir.contains(&format!("{}", y::cpu_gemm::KSPLIT_MIN_K)),
        "the K-split depth guard is not in the emitted driver"
    );

    let shapes = SHAPES
        .iter()
        .map(|(m, n, k, _)| format!("{{{},{},{}}}", m, n, k))
        .collect::<Vec<_>>()
        .join(", ");
    let driver = dir.join("driver.c");
    std::fs::write(&driver, DRIVER.replace("SHAPES_HERE", &shapes)).expect("write driver");

    let exe = dir.join("mm_mt_test");
    let cc = Command::new("clang")
        .args(["-O2", "-o"])
        .arg(&exe)
        .arg(&driver)
        .arg(&ll)
        .arg("-lm")
        .arg("-lpthread")
        .output()
        .expect("run clang");
    assert!(
        cc.status.success(),
        "clang failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    // Generous: the reference is an O(M*N*K) scalar triple loop in the driver
    // and dominates. Tight enough that a deadlock is reported as one.
    let limit = Duration::from_secs(180);
    let mut failures = Vec::new();
    for &nthr in THREAD_COUNTS {
        let (ok, text) = run_with_timeout(&exe, nthr, limit);
        if !ok {
            failures.push(format!("Y_NUM_THREADS={}:\n{}", nthr, text));
        }
    }
    assert!(
        failures.is_empty(),
        "threaded CPU GEMM failed:\n{}",
        failures.join("\n")
    );

    let _ = std::fs::remove_dir_all(&dir);
}
