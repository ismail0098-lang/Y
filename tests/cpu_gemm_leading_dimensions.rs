//! A submatrix is a legal GEMM input, and it must be both CORRECT and FAST.
//!
//! Until this landed, `recognize_gemm` required each operand's row stride to
//! be the matching loop extent (`lda == K`, `ldb == N`, `ldc == N`). Anything
//! else fell through to scalar lowering — correct, but roughly 100x slower,
//! with no diagnostic. That restriction is why the kernel could not implement
//! `sgemm`: the BLAS API is *defined* in terms of leading dimensions, so a
//! packed-only kernel cannot be put behind it without copying every operand.
//!
//! What makes this worth its own file rather than another shape in
//! `cpu_gemm_end_to_end.rs`: when `lda == K` the stride and the extent are the
//! same number, so every one of the ~12 address computations threaded through
//! this backend is correct *by coincidence*. Only a stride that differs from
//! its extent can tell them apart, and a site that still uses the extent is
//! then either reading the wrong element (caught by the norm below) or running
//! off the end of a row (caught by the padding sentinel).
//!
//! The padding sentinel is the load-bearing check. A relative-L2 comparison
//! alone cannot see a kernel that writes *past* `N` into C's padding — every
//! element it was asked to produce would still be right. That is exactly the
//! failure a `ldc`-for-`n` mix-up produces in the K-split reduction, where the
//! per-thread panels are `n`-strided and C is `ldc`-strided in the same loop.
//!
//! Requires `clang`; skipped with a notice otherwise, like the sibling gates.
//!
//! Run with:  cargo test --release --test cpu_gemm_leading_dimensions

use std::path::PathBuf;
use std::process::Command;

/// The same canonical nest, but with the three strides as their own
/// parameters. The recogniser must accept this and record `LDA`/`LDB`/`LDC`
/// rather than refusing it for not matching `K`/`N`/`N`.
const Y_SOURCE: &str = r#"
kernel y_matmul_ld(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, M: I32, N: I32, K: I32, LDA: I32, LDB: I32, LDC: I32) {
    @invariant(i >= 0)
    for i in 0..M step 1 {
        @invariant(j >= 0)
        for j in 0..N step 1 {
            let mut sum: F32 = 0.0;
            @invariant(k >= 0)
            for k in 0..K step 1 {
                let a_val: F32 = block_ptr2d_load(A, i, k, LDA, M, K);
                let b_val: F32 = block_ptr2d_load(B, k, j, LDB, K, N);
                sum = sum + a_val * b_val;
            }
            block_ptr2d_store(C, i, j, LDC, M, N, sum);
        }
    }
}

fn main() {
}
"#;

/// `(M, N, K, pad_a, pad_b, pad_c)` — the pads are added to K, N and N
/// respectively to give `lda`, `ldb`, `ldc`.
///
/// Every dispatch path gets both a padded and an unpadded case, because the
/// three paths address memory with entirely separate code: `__y_gemm_tiny`
/// (copy-free, all three operands in place), `__y_gemm_small_m` (copy-free A
/// and B, C written directly) and the packed path (A and B absorbed by
/// `pack_a`/`pack_b`, C written by the micro-kernel, and under a K-split
/// written by `small_reduce` instead).
///
/// Pads are deliberately not multiples of 16: a pad that keeps every row
/// 64-byte aligned would hide a masked-tail bug in the same way a
/// tile-multiple shape hides a dropped remainder.
const SHAPES: &[(usize, usize, usize, usize, usize, usize)] = &[
    // Zero pad on all three: the packed case must keep working, and this is
    // the control that says a failure below is about strides and not about
    // the change in general.
    (64, 64, 64, 0, 0, 0),
    (100, 100, 100, 0, 0, 0),
    // One stride at a time, so a single wrong site is attributable.
    (64, 64, 64, 7, 0, 0),
    (64, 64, 64, 0, 5, 0),
    (64, 64, 64, 0, 0, 3),
    // All three at once, ragged in every dimension.
    (17, 40, 300, 11, 13, 9),
    (13, 33, 257, 1, 1, 1),
    (193, 65, 257, 23, 7, 19),
    // GEMV and decode -> `__y_gemm_small_m`, the shapes the commercial case
    // rests on and the ones most likely to be called on a submatrix (a row
    // block of a weight matrix is exactly this).
    (1, 129, 257, 6, 15, 2),
    (4, 256, 512, 33, 9, 5),
    (8, 64, 64, 3, 3, 3),
    // Just past the small-M boundary -> packed path with M < MR.
    (9, 64, 64, 4, 4, 4),
    // `__y_gemm_tiny`: M > SM_MAX_M, N <= TINY_MAX_N, work <= TINY_MAX_WORK.
    // One per accumulator body, each with a ragged M.
    (48, 48, 48, 5, 5, 5),
    (25, 16, 40, 2, 17, 6),
    (25, 17, 40, 8, 1, 4),
    (20, 49, 33, 3, 6, 11),
    // Large enough to thread, so the 2-D grid, the shared packed-B panel and
    // the K-split reduction all run with a padded C.
    (512, 512, 512, 9, 6, 12),
    (1024, 256, 256, 4, 8, 16),
    // K LARGER THAN `kc`, so the `pc` loop runs more than once. Without one of
    // these, B's row offset (`pc * ldb`) is only ever evaluated at `pc = 0`,
    // where every stride gives 0 — so a wrong stride there is invisible.
    // Found by mutation: every shape above passed with that offset computed
    // from the extent instead of the leading dimension. `kc` is chosen at
    // runtime and its default is 1024, hence K well past that.
    (32, 48, 3000, 7, 11, 5),
    (64, 64, 2500, 0, 9, 0),
    (200, 96, 1500, 13, 5, 21),
];

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void y_matmul_ld(const float *A, const float *B, float *C,
                 int M, int N, int K, int lda, int ldb, int ldc);

static float gen(unsigned long i, unsigned long salt) {
    unsigned long h = i * 2654435761UL + salt * 40503UL;
    h ^= h >> 13;
    return (float)((h & 0xffff) / 65535.0 - 0.5);
}

/* A value that is not plausibly a result, so a kernel that reads the padding
   as data produces an obviously wrong answer rather than a subtly wrong one,
   and one that writes the padding is caught exactly. */
#define PAD_SENTINEL (-987654.0f)

int main(void) {
    static const int shapes[][6] = { SHAPES_HERE };
    int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));
    int bad = 0;

    for (int s = 0; s < ns; ++s) {
        int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
        int lda = K + shapes[s][3], ldb = N + shapes[s][4], ldc = N + shapes[s][5];

        float *A = malloc((size_t)M * lda * sizeof(float));
        float *B = malloc((size_t)K * ldb * sizeof(float));
        float *C = malloc((size_t)M * ldc * sizeof(float));
        float *R = malloc((size_t)M * N * sizeof(float));
        if (!A || !B || !C || !R) { printf("ALLOC FAIL\n"); return 2; }

        /* Live region gets data; padding gets the sentinel. A kernel that
           strides by the extent instead of the leading dimension walks
           diagonally into the padding and reads these. */
        for (long i = 0; i < (long)M * lda; ++i) A[i] = PAD_SENTINEL;
        for (long i = 0; i < (long)K * ldb; ++i) B[i] = PAD_SENTINEL;
        for (long i = 0; i < (long)M * ldc; ++i) C[i] = PAD_SENTINEL;
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) A[(long)i * lda + k] = gen((long)i * K + k, 1);
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j) B[(long)k * ldb + j] = gen((long)k * N + j, 2);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) C[(long)i * ldc + j] = 0.0f;

        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k)
                    acc += (double)A[(long)i * lda + k] * (double)B[(long)k * ldb + j];
                R[(long)i * N + j] = (float)acc;
            }

        /* Called in a TIGHT LOOP, not once, and that is required rather than
           defensive. The thread-count heuristic treats a call more than
           HOT_WINDOW_NS (100us) after the previous one as cold and runs it on
           one thread; the reference loop above takes far longer than that, so
           a single call per shape is always single-threaded. With one call
           this test never reached __y_gemm_small_reduce at all -- confirmed by
           mutation: breaking that function's C stride left every assertion
           green. The kernel overwrites C rather than accumulating into it, so
           repeating the call is idempotent and needs no re-zeroing. */
        for (int rep = 0; rep < 24; ++rep)
            y_matmul_ld(A, B, C, M, N, K, lda, ldb, ldc);

        double num = 0.0, den = 0.0;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double d = (double)C[(long)i * ldc + j] - (double)R[(long)i * N + j];
                num += d * d;
                den += (double)R[(long)i * N + j] * (double)R[(long)i * N + j];
            }
        double rel = sqrt(num) / (sqrt(den) + 1e-30);
        if (!(rel < 1e-5)) {
            printf("MISMATCH %dx%dx%d lda=%d ldb=%d ldc=%d relL2=%g\n",
                   M, N, K, lda, ldb, ldc, rel);
            bad = 1;
        }

        /* C's padding must be byte-for-byte untouched. This is the check a
           norm over the live region cannot make. */
        long touched = 0;
        for (int i = 0; i < M; ++i)
            for (int j = N; j < ldc; ++j)
                if (C[(long)i * ldc + j] != PAD_SENTINEL) touched++;
        if (touched) {
            printf("PAD WRITTEN %dx%dx%d ldc=%d: %ld elements\n", M, N, K, ldc, touched);
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

/// Compile `Y_SOURCE`, link it against `DRIVER`, run it under `threads`.
fn run_with_threads(threads: &str) {
    let dir = std::env::temp_dir().join(format!(
        "y_cpu_gemm_ld_{}_{}",
        std::process::id(),
        threads
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let src = dir.join("mm_ld.ysu");
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

    let ll = dir.join("mm_ld.ll");
    assert!(ll.exists(), "Y did not write {}", ll.display());
    let ir = std::fs::read_to_string(&ll).expect("read emitted IR");

    // Without this the whole file would pass on the scalar fallback — which is
    // correct, and is precisely the behaviour this change exists to replace.
    // A strided nest that merely computes the right answer slowly is a
    // regression, not a pass.
    assert!(
        ir.contains("[Y CPU GEMM]"),
        "the GEMM recogniser did not fire on a nest with explicit leading \
         dimensions; this test would be measuring the scalar fallback"
    );
    // The strides must reach the kernel as their own arguments. If the emitter
    // passed the extents instead, every shape here would still be dispatched
    // to the fast path and would then read the wrong elements.
    assert!(
        ir.contains("%LDA") && ir.contains("%LDB") && ir.contains("%LDC"),
        "the leading-dimension parameters never appear in the emitted IR"
    );

    let shapes = SHAPES
        .iter()
        .map(|(m, n, k, pa, pb, pc)| format!("{{{},{},{},{},{},{}}}", m, n, k, pa, pb, pc))
        .collect::<Vec<_>>()
        .join(", ");
    let driver = dir.join("driver.c");
    std::fs::write(&driver, DRIVER.replace("SHAPES_HERE", &shapes)).expect("write driver");

    let obj = dir.join("mm_ld.o");
    let cc = Command::new("clang")
        .args(["-O3", "-c"])
        .arg(&ll)
        .arg("-o")
        .arg(&obj)
        .output()
        .expect("run clang on the emitted IR");
    assert!(
        cc.status.success(),
        "clang failed on emitted IR:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    let exe = dir.join("mm_ld_test");
    let cc = Command::new("clang")
        .args(["-O2", "-o"])
        .arg(&exe)
        .arg(&driver)
        .arg(&obj)
        .args(["-lm", "-lpthread"])
        .output()
        .expect("link driver");
    assert!(
        cc.status.success(),
        "link failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    let run = Command::new(&exe)
        .env("Y_NUM_THREADS", threads)
        .output()
        .expect("run the test binary");
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success() && stdout.contains("ALL_SHAPES_OK"),
        "strided GEMM failed at Y_NUM_THREADS={}:\n{}",
        threads,
        stdout
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// One thread: isolates the kernels from the partition. A failure here is an
/// address computation; a failure only in the threaded test below is the
/// partition or the K-split reduction.
#[test]
fn submatrix_operands_are_correct_single_threaded() {
    if !have("clang") {
        eprintln!("note: clang not found, skipping strided CPU GEMM test");
        return;
    }
    run_with_threads("1");
}

/// Threaded, which is the only way to reach `__y_gemm_small_reduce` — the one
/// site where an `n`-strided buffer (the per-thread scratch panels) and an
/// `ldc`-strided one (the caller's C) are indexed in the same loop. Passing
/// `ldc` to the panel there would stride past a thread's scratch slot into the
/// next thread's, which is a wrong answer AND a data race.
#[test]
fn submatrix_operands_are_correct_threaded() {
    if !have("clang") {
        eprintln!("note: clang not found, skipping strided CPU GEMM test");
        return;
    }
    run_with_threads("16");
}
