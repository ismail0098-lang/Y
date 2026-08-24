//! The exact GEMM must give the SAME BITS at every thread count.
//!
//! This is Phase 0's stated "done when" in `docs/proof_carrying_kernels.md`,
//! and it is the property the whole proof-carrying-kernel programme rests on:
//! integer addition is associative, so partitioning K across workers and
//! summing the partial sums cannot change the total. An f32 kernel cannot make
//! that claim - `cpu_gemm_exact_threaded.rs` keeps a control showing the f32
//! formulation genuinely drifts under the same two axes.
//!
//! ## The trap this file is built around
//!
//! "Every thread count gives the same answer" is **also** what a threading path
//! that never spawns a thread reports. A no-op is perfectly invariant. So the
//! bit-identity assertion is paired with a liveness canary that counts actual
//! `pthread_create` calls, via `-Wl,--wrap=pthread_create` - deterministic,
//! unlike a timing check, and it fails if the K-split silently collapses to one
//! worker. See `feedback-null-metrics-pass-dead-components`.
//!
//! K is 4099 on purpose: the band floor is `KSPLIT_MIN_BAND` (128), so the
//! thread count is capped at `K/128`. At the K values the correctness tests use
//! (~300) the cap is 2, and a sweep to 16 threads would silently be a sweep to
//! 2 - `feedback-exercised-is-not-covered`.
//!
//! Requires `clang`; skipped with a notice otherwise.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

const SOURCE: &str = r#"
kernel y_matmul(A: GlobalMemory<I16>, B: GlobalMemory<I16>, C: GlobalMemory<I64>, M: I32, N: I32, K: I32) {
    @invariant(i >= 0)
    for i in 0..M step 1 {
        @invariant(j >= 0)
        for j in 0..N step 1 {
            @ZeroDrift
            let mut sum: I64 = 0;
            @invariant(k >= 0)
            for k in 0..K step 1 {
                @bounds(min=-1024, max=1024)
                let a_val: I64 = block_ptr2d_load(A, i, k, K, M, K);
                @bounds(min=-1024, max=1024)
                let b_val: I64 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a_val * b_val;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}

fn main() {
}
"#;

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

void y_matmul(const int16_t *A, const int16_t *B, int64_t *C, int M, int N, int K);

/* The liveness canary. Counts real spawns, so a K-split that quietly collapses
   to a single worker fails instead of passing perfectly. */
static long spawned = 0;
int __real_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int __wrap_pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*f)(void *), void *p) {
    spawned++;
    return __real_pthread_create(t, a, f, p);
}

static int16_t gen(long i, long salt) {
    unsigned long h = i * 2654435761UL + salt * 40503UL;
    h ^= h >> 13;
    return (int16_t)((long)(h % 2049) - 1024);
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    /* M and N are deliberately not multiples of MR=6 or NR=64, and K is odd,
       so the k-pair packing has a ragged tail on every axis. */
    int M = 53, N = 71, K = 4099;
    int16_t *A = malloc((size_t)M*K*2), *B = malloc((size_t)K*N*2);
    int64_t *C = malloc((size_t)M*N*8), *R = malloc((size_t)M*N*8);
    for (long i = 0; i < (long)M*K; ++i) A[i] = gen(i, 1);
    for (long i = 0; i < (long)K*N; ++i) B[i] = gen(i, 2);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            int64_t a = 0;
            for (int k = 0; k < K; ++k)
                a += (int64_t)A[(long)i*K+k] * (int64_t)B[(long)k*N+j];
            R[(long)i*N+j] = a;
        }
    memset(C, 0xAB, (size_t)M*N*8);
    y_matmul(A, B, C, M, N, K);
    FILE *f = fopen(argv[1], "wb");
    if (!f) return 2;
    fwrite(C, sizeof(int64_t), (size_t)M*N, f);
    fclose(f);
    printf("spawned %ld\n", spawned);
    printf("reference %s\n", memcmp(C, R, (size_t)M*N*8) ? "MISMATCH" : "OK");
    printf("DONE\n");
    return 0;
}
"#;

#[test]
fn the_exact_gemm_gives_the_same_bits_at_every_thread_count() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_exact_thr_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let src = dir.join("m.ysu");
    std::fs::write(&src, SOURCE).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-llvm")
        .current_dir(repo())
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "the exact nest must compile:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ll = dir.join("m.ll");
    let ir = std::fs::read_to_string(&ll).expect("emitted IR");
    assert!(
        ir.contains("__y_gemm_exact_vnni_threaded"),
        "the nest must be routed through the threaded entry, or this file is \
         measuring a single-threaded kernel"
    );

    let driver = dir.join("drv.c");
    std::fs::write(&driver, DRIVER).expect("write driver");
    let exe = dir.join("run");
    let cc = Command::new("clang")
        .args([
            ll.to_str().unwrap(),
            driver.to_str().unwrap(),
            "-O2",
            "-lm",
            "-lpthread",
            "-Wl,--wrap=pthread_create",
            "-o",
            exe.to_str().unwrap(),
        ])
        .output()
        .expect("run clang");
    assert!(
        cc.status.success(),
        "linking failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    let counts = [1usize, 2, 3, 5, 8, 16];
    let mut results: Vec<(usize, Vec<u8>, usize)> = Vec::new();
    for n in counts {
        let dump = dir.join(format!("c{n}.bin"));
        let run = Command::new(&exe)
            .arg(&dump)
            .env("Y_NUM_THREADS", n.to_string())
            .output()
            .expect("run");
        let text = String::from_utf8_lossy(&run.stdout).into_owned();
        assert!(
            run.status.success() && text.contains("DONE"),
            "the run at {n} threads did not finish:\n{text}"
        );
        assert!(
            text.contains("reference OK"),
            "at {n} threads the result disagrees with the integer reference:\n{text}"
        );
        let spawned: usize = text
            .lines()
            .find_map(|l| l.strip_prefix("spawned "))
            .and_then(|v| v.trim().parse().ok())
            .expect("spawn count");
        results.push((n, std::fs::read(&dump).expect("read dump"), spawned));
    }

    // THE LIVENESS CANARY. Without it, a threading path that never forks
    // passes every assertion below perfectly.
    let one = results[0].2;
    assert_eq!(
        one, 0,
        "at Y_NUM_THREADS=1 the wrapper must take its direct single-threaded \
         path and spawn nothing; it spawned {one}"
    );
    for (n, _, spawned) in &results[1..] {
        assert_eq!(
            spawned, n,
            "at Y_NUM_THREADS={n} the K-split spawned {spawned} workers. If this \
             is 0 the split collapsed to one thread and the invariance below is \
             vacuous - K is 4099 here precisely so the band floor \
             (KSPLIT_MIN_BAND) does not cap the count."
        );
    }

    // THE CLAIM. Same bits, every thread count, over a ragged K.
    let (_, first, _) = &results[0];
    for (n, bytes, _) in &results {
        assert_eq!(
            bytes,
            first,
            "the exact GEMM produced different bytes at {n} threads than at 1. \
             Integer addition is associative, so partitioning K and summing the \
             partial sums cannot change the total - a difference here means the \
             K-split is not a partition, or a partial was dropped or \
             double-counted."
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
