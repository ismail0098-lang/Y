//! The CPU GEMM path must compute the right answer — proven by running it.
//!
//! This backend had two silent wrong-answer bugs at once, and neither was
//! visible to a test that inspected the emitted IR:
//!
//!   1. `block_ptr2d_load`/`_store` were lowered as calls to an external
//!      runtime helper declared `i32 (...)`, so the 64-bit base pointer was
//!      truncated by `ptrtoint ptr %p to i32`. The old benchmark worked around
//!      this by allocating with `mmap(MAP_32BIT)` — which is exactly the shape
//!      of workaround that hides a miscompile.
//!   2. The same helper returned a float's *bit pattern* as `i32`, and the
//!      declared `F32` result was then produced with `sitofp` — an integer
//!      conversion of a value that was already a float. Every element was
//!      garbage.
//!
//! So this compiles a Y matmul with `clang`, links it against a driver that
//! uses ordinary 64-bit `malloc`, runs it, and checks the numbers against a
//! reference computed in the driver. A wrong pointer width now segfaults or
//! mismatches instead of quietly working.
//!
//! The shapes matter: they cross the micro-kernel's `MR`/`NR` blocking and the
//! `SM_MAX_M` dispatch boundary, so both the packed path and the small-M
//! k-outer path are exercised, including their ragged tails. A GEMM that is
//! only ever tested on multiples of the tile size cannot catch a dropped
//! remainder — the same class of bug the PTX backend shipped as a truncated
//! `k_tiles`.
//!
//! Requires `clang`; skipped with a notice otherwise, like the `ptxas` and
//! `solc` gates elsewhere in the suite.
//!
//! Run with:  cargo test --release --test cpu_gemm_end_to_end

use std::path::PathBuf;
use std::process::Command;

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

/// Shapes chosen to straddle every blocking boundary the kernel has:
/// `MR = 12`, `NR = 32`, `KC = 256`, `MC = 192`, and the `SM_MAX_M = 8`
/// dispatch. Several are prime or otherwise ragged in all three dimensions.
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),        // degenerate
    (1, 129, 257),    // GEMV, ragged N and K -> small-M path
    (8, 64, 64),      // exactly at the small-M dispatch boundary
    (9, 64, 64),      // one past it -> packed path, M < MR
    (12, 32, 256),    // exactly one micro-tile
    (13, 33, 257),    // one past every tile bound
    (17, 40, 300),    // ragged everywhere
    (64, 64, 64),     // several tiles
    (100, 100, 100),  // ragged, multiple MC/NR blocks
    (193, 65, 257),   // M just past MC, N and K ragged
];

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void y_matmul(const float *A, const float *B, float *C, int M, int N, int K);

/* Deterministic, and spread over a few orders of magnitude so a dropped
   tail shows up rather than averaging away. */
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
        /* Ordinary 64-bit heap. If the emitter still truncated pointers to
           32 bits this would fault rather than silently work. */
        float *A = malloc((size_t)M * K * sizeof(float));
        float *B = malloc((size_t)K * N * sizeof(float));
        float *C = malloc((size_t)M * N * sizeof(float));
        float *R = malloc((size_t)M * N * sizeof(float));
        if (!A || !B || !C || !R) { printf("ALLOC FAIL\n"); return 2; }

        for (long i = 0; i < (long)M * K; ++i) A[i] = gen(i, 1);
        for (long i = 0; i < (long)K * N; ++i) B[i] = gen(i, 2);
        memset(C, 0, (size_t)M * N * sizeof(float));

        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k)
                    acc += (double)A[(long)i * K + k] * (double)B[(long)k * N + j];
                R[(long)i * N + j] = (float)acc;
            }

        y_matmul(A, B, C, M, N, K);

        /* Relative L2 over the whole matrix: at large K the kernel sums in a
           different order than the reference, so a per-element tolerance would
           produce false alarms without catching anything a norm misses. */
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
        free(A); free(B); free(C); free(R);
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

#[test]
fn emitted_cpu_gemm_matches_a_reference_on_ragged_shapes() {
    if !have("clang") {
        eprintln!("note: clang not found, skipping CPU GEMM end-to-end test");
        return;
    }

    let dir = std::env::temp_dir().join(format!("y_cpu_gemm_{}", std::process::id()));
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
    assert!(ll.exists(), "Y did not write {}", ll.display());

    // The fast path must actually be the one under test. Without this the
    // test would still pass on the scalar fallback, which is correct but is
    // not what this file exists to check.
    let ir = std::fs::read_to_string(&ll).expect("read emitted IR");
    assert!(
        ir.contains("[Y CPU GEMM]"),
        "the GEMM recogniser did not fire; this test would be measuring the \
         scalar fallback instead of the packed kernel"
    );
    assert!(
        !ir.contains("ptrtoint ptr"),
        "a pointer is still being narrowed through an integer in the kernel"
    );

    let shapes = SHAPES
        .iter()
        .map(|(m, n, k)| format!("{{{},{},{}}}", m, n, k))
        .collect::<Vec<_>>()
        .join(", ");
    let driver = dir.join("driver.c");
    std::fs::write(&driver, DRIVER.replace("SHAPES_HERE", &shapes)).expect("write driver");

    let exe = dir.join("mm_test");
    let cc = Command::new("clang")
        .args(["-O2", "-o"])
        .arg(&exe)
        .arg(&driver)
        .arg(&ll)
        .arg("-lm")
        .output()
        .expect("run clang");
    assert!(
        cc.status.success(),
        "clang failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    let run = Command::new(&exe).output().expect("run the compiled program");
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success() && stdout.contains("ALL_SHAPES_OK"),
        "compiled GEMM produced wrong results:\n{}\n{}",
        stdout,
        String::from_utf8_lossy(&run.stderr)
    );

    let _ = std::fs::remove_dir_all(&dir);
}
