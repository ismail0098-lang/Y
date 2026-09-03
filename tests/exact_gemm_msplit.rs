//! The exact GEMM cuts M when it can, and K when it must - and the two axes
//! must give the same bits.
//!
//! ## Why there are two axes at all
//!
//! `ExactGemmKsplit.v` proves the contraction split correct, and it is the one
//! the wrapper shipped. Reading its own performance back showed it is the
//! EXPENSIVE one: cutting K is a reduction, so every thread needs a private
//! `M x N` C to sum out of, and that bookkeeping is `O(T * M * N)` against
//! `O(M * N * K)` of work. Measured at 1024x1024x1024 on 8 threads, by calling
//! the unthreaded kernel directly to decompose the shipped path:
//!
//! ```text
//!   pure compute, one core, whole K      7.344 ms
//!   K-split buffer + reduce overhead     9.311 ms   (memset 4.804, reduce 2.393)
//!   shipped threaded path               10.774 ms
//!   ideal if it parallelised             0.918 ms
//! ```
//!
//! The bookkeeping cost more than the entire GEMM, and eight threads ran
//! SLOWER than one. Cutting M instead is a PARTITION - each thread owns a
//! disjoint row band of C - so there is no private buffer, no zero-fill and
//! nothing to reduce. Measured after: 1.575 ms at 8 threads, **6.4x**.
//!
//! ## What this file pins, and why bit-identity is the only acceptable evidence
//!
//! Changing which axis a kernel is cut on, under a certificate that claims the
//! result is bit-identical to the naive nest, is exactly the change that must
//! not be justified by a benchmark. So:
//!
//! * `the_two_axes_give_the_same_bits` compiles ONE program and drives it at
//!   shapes and thread counts that select each axis, comparing whole output
//!   buffers - and checks each against an independent integer reference, so two
//!   schedules cannot be wrong together.
//! * `the_emitted_dispatch_is_the_axis_the_model_predicts` ties
//!   `cpu_gemm::split_axis` to an OBSERVABLE - the `--wrap=pthread_create`
//!   count, which differs between the axes at a shape chosen to separate them.
//!   Without it the model is a comment.
//! * `the_row_bands_write_the_callers_c_at_the_callers_stride` reads the
//!   emitted IR. The M path stores `%ldc` in job slot 8 where the K path
//!   stores `%N`, and that is not a detail: the K path's destination is a
//!   compact private buffer, so storing `ldc` there was a heap overflow this
//!   repository has already had; the M path's destination is the caller's own
//!   C, so storing `N` there is a wrong answer on a padded C.
//! * `the_m_path_allocates_no_private_c_and_reduces_nothing` is the structural
//!   claim that makes the whole change worth making.
//! * `a_short_m_still_takes_the_contraction_axis` is the control. "Always split
//!   M" satisfies every assertion above and is 2.7x slower on the shapes the
//!   K-split exists for.
//!
//! Requires `clang`; skipped with a notice otherwise.

use std::path::PathBuf;
use std::process::Command;
use y::cpu_gemm::{split_axis, SplitAxis, KSPLIT_MIN_BAND, MSPLIT_MIN_ROWS, VNNI_MR};

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
fn main() {}
"#;

/// Takes M, N, K on the command line, writes the result buffer to a file, and
/// prints the real spawn count. The reference is computed in plain int64 so
/// neither schedule is its own oracle.
const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

void y_matmul(const int16_t *A, const int16_t *B, int64_t *C, int M, int N, int K);

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
    if (argc < 5) return 2;
    int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]);
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
    /* Poisoned, not zeroed: the wrapper owns the zeroing of C, and a zeroed
       buffer would hide a band that was never written. */
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

/// Compile the exact nest and link it against the counting driver.
///
/// `tag` keeps each test in its own directory: the tests in this file share a
/// pid, so naming the directory after the pid alone is the race this
/// repository has hit five times.
fn build(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("y_msplit_{tag}_{}", std::process::id()));
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
        "the nest must be routed through the threaded entry"
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
    (dir, exe)
}

struct Run {
    bytes: Vec<u8>,
    spawned: usize,
}

fn run(exe: &PathBuf, dir: &PathBuf, threads: usize, m: usize, n: usize, k: usize) -> Run {
    let out_path = dir.join(format!("out_{threads}_{m}_{n}_{k}.bin"));
    let out = Command::new(exe)
        .env("Y_NUM_THREADS", threads.to_string())
        .args([
            out_path.to_str().unwrap(),
            &m.to_string(),
            &n.to_string(),
            &k.to_string(),
        ])
        .output()
        .expect("run the kernel");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success() && text.contains("DONE"),
        "the kernel did not finish at {threads} threads, {m}x{n}x{k}:\n{text}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("reference OK"),
        "the kernel disagrees with an independent integer reference at \
         {threads} threads, {m}x{n}x{k} - so the two schedules cannot be \
         checked against each other:\n{text}"
    );
    let spawned = text
        .lines()
        .find_map(|l| l.strip_prefix("spawned "))
        .and_then(|v| v.trim().parse().ok())
        .expect("the driver prints its spawn count");
    Run {
        bytes: std::fs::read(&out_path).expect("result buffer"),
        spawned,
    }
}

// ── The acceptance criterion ────────────────────────────────────────────

#[test]
fn the_two_axes_give_the_same_bits() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let (dir, exe) = build("bits");

    // Both axes, and both the ragged and the aligned case on each. M = 53 and
    // 197 are not multiples of MR = 6; N = 71 is not a multiple of NR = 64;
    // K = 4099 is odd, so the k-pair packing has a ragged tail too.
    for &(m, n, k) in &[
        (197usize, 71usize, 4099usize), // M-split: 197/12 = 16 >= 16
        (53, 71, 4099),                 // K-split: 53/12 = 4 < 16
        (384, 64, 512),                 // M-split, aligned M
        (64, 64, 2048),                 // K-split, short M
    ] {
        let mut baseline: Option<Vec<u8>> = None;
        for &t in &[1usize, 2, 3, 5, 8, 16] {
            let r = run(&exe, &dir, t, m, n, k);
            match &baseline {
                None => baseline = Some(r.bytes),
                Some(b) => assert_eq!(
                    b, &r.bytes,
                    "{m}x{n}x{k} at {t} threads differs from the 1-thread result. \
                     The axis a schedule is cut on must not be observable in the \
                     answer - that is what the certificate claims."
                ),
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ── The model tie: the axis choice must be OBSERVABLE ───────────────────

#[test]
fn the_emitted_dispatch_is_the_axis_the_model_predicts() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let (dir, exe) = build("axis");

    // Shapes chosen so the two axes predict DIFFERENT spawn counts. Without
    // that the assertion holds whichever axis ran, which is the trap this
    // repository calls a null metric.
    //
    //   (a) M = 384, K = 512 at 16: M gives 16 (384/12 = 32), K gives 4.
    //   (b) M = 64,  K = 1024 at 16: M gives 5 < 16 so K is taken, K gives 8.
    //   (c) M = 192 = MSPLIT_MIN_ROWS * 16 exactly: the LAST M that saturates
    //       16 threads. K = 1024 gives the K-split 8, so the counts differ.
    //   (d) M = 191, one row below: the axis flips to K and the count to 8.
    //
    // (c) and (d) are what make the FLOOR load-bearing. The first version of
    // this test had only (a) and (b), and a mutation halving the emitted floor
    // to 6 - so the emitter and `msplit_threads` disagreed - PASSED: at both
    // shapes the two floors happen to give the same answer. A model tie needs
    // a shape whose axis MOVES when the constant does.
    for &(m, n, k, req) in &[
        (384usize, 64usize, 512usize, 16usize),
        (64, 64, 1024, 16),
        (MSPLIT_MIN_ROWS * 16, 64, 1024, 16),
        (MSPLIT_MIN_ROWS * 16 - 1, 64, 1024, 16),
    ] {
        let (axis, threads) = split_axis(req, m, k);
        let r = run(&exe, &dir, req, m, n, k);
        assert_eq!(
            r.spawned, threads,
            "at {m}x{n}x{k} with {req} requested the model says {axis:?} on \
             {threads} threads; the kernel spawned {}. The spawn count is the \
             only thing the axis choice reveals in a run.",
            r.spawned
        );
    }

    // ...and the two really do disagree at (a), so the assertion above is not
    // satisfied by either answer.
    let (axis_a, ta) = split_axis(16, 384, 512);
    assert_eq!(axis_a, SplitAxis::Rows);
    assert_eq!(ta, 16);
    let (axis_b, tb) = split_axis(16, 64, 1024);
    assert_eq!(axis_b, SplitAxis::Contraction);
    assert_eq!(tb, 8);
    assert_ne!(ta, tb, "the two shapes must predict different spawn counts");

    // The floor's boundary is one row wide, and both sides are asserted: a
    // gate that only checks the winning side cannot see a floor that moved
    // down.
    let (axis_at, tat) = split_axis(16, MSPLIT_MIN_ROWS * 16, 1024);
    let (axis_below, tbelow) = split_axis(16, MSPLIT_MIN_ROWS * 16 - 1, 1024);
    assert_eq!(
        (axis_at, tat),
        (SplitAxis::Rows, 16),
        "exactly MSPLIT_MIN_ROWS rows per thread must still saturate"
    );
    assert_eq!(
        (axis_below, tbelow),
        (SplitAxis::Contraction, 8),
        "one row short must fall back to the contraction axis"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ── The control ─────────────────────────────────────────────────────────

#[test]
fn a_short_m_still_takes_the_contraction_axis() {
    // "Always split M" passes every other assertion in this file and is 2.7x
    // slower than one thread on the shapes the K-split exists for: at 16
    // threads and K = 16384, M = 96 gives 8 M-threads and loses 0.61x, M = 128
    // gives 10 and loses 0.88x. The rule is that M is taken only when it
    // SATURATES the request.
    assert_eq!(split_axis(16, 96, 16384).0, SplitAxis::Contraction);
    assert_eq!(split_axis(16, 128, 16384).0, SplitAxis::Contraction);
    assert_eq!(split_axis(16, 192, 16384).0, SplitAxis::Rows);

    // A single thread cuts nothing.
    assert_eq!(split_axis(1, 4096, 4096), (SplitAxis::None, 1));

    // ...and a K too short for even one band still runs, on one thread.
    assert_eq!(split_axis(16, 8, KSPLIT_MIN_BAND - 1), (SplitAxis::None, 1));

    // The floor is derived from the register tile, not chosen: two `MR`-row
    // micro-kernel tiles per band. 192/16 is exactly that, which is where the
    // measured crossover sits.
    assert_eq!(MSPLIT_MIN_ROWS, 2 * VNNI_MR);
}

// ── The structural claims about the emitted M path ──────────────────────

/// `tag` is in the SIGNATURE, not left to the caller's memory. The first
/// version of this helper was pid-only with two callers, and the two tests
/// raced on one directory - caught by the mutation table's CONTROL row coming
/// back red, which is the second thing a control is for.
fn emitted_ir(tag: &str) -> String {
    let dir = std::env::temp_dir().join(format!("y_msplit_ir_{tag}_{}", std::process::id()));
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
    assert!(out.status.success(), "the exact nest must compile");
    let ir = std::fs::read_to_string(dir.join("m.ll")).expect("emitted IR");
    let _ = std::fs::remove_dir_all(&dir);
    ir
}

/// The region between two labels, so an assertion about the M path cannot be
/// satisfied by a line in the K path.
fn region<'a>(ir: &'a str, from: &str, to: &str) -> &'a str {
    let i = ir.find(from).unwrap_or_else(|| panic!("no `{from}` block"));
    let j = ir[i..]
        .find(to)
        .unwrap_or_else(|| panic!("no `{to}` after `{from}`"));
    &ir[i..i + j]
}

#[test]
fn the_row_bands_write_the_callers_c_at_the_callers_stride() {
    let ir = emitted_ir("stride");
    let m = region(&ir, "mspawn.body:", "mjoin.head:");

    // The destination is the caller's C, offset by this band's first row.
    assert!(
        m.contains("%mcoffi = mul i64 %moff, %ldc"),
        "the M band's C offset must be its first row times the caller's stride"
    );
    assert!(
        m.contains("getelementptr i64, ptr %C, i64 %mcoffi"),
        "the M band must write into the CALLER's C, not a private buffer"
    );
    // Slot 8 is the worker's output stride. The K path stores `%N` because its
    // destination is a compact private buffer - storing `ldc` there was a heap
    // overflow. The M path's destination is the caller's C, so `N` here would
    // be a wrong answer on a padded C. The two are opposite for a reason.
    assert!(
        m.contains("store i64 %ldc, ptr %m8"),
        "the M band's output stride must be the caller's `ldc`"
    );
    let k = region(&ir, "spawn.body:", "joinloop.head:");
    assert!(
        k.contains("store i64 %N, ptr %s8"),
        "the K band's output stride must stay `%N` - its destination is compact"
    );
    // A is offset by rows, and B is shared whole.
    assert!(
        m.contains("%maoffi = mul i64 %moff, %lda"),
        "the M band's A offset must be its first row times `lda`"
    );
    assert!(
        m.contains("store ptr %B, ptr %m1"),
        "every M band reads all of B"
    );
}

#[test]
fn the_m_path_allocates_no_private_c_and_reduces_nothing() {
    let ir = emitted_ir("noprivc");
    let m = region(&ir, "mspawn.pre:", "mcleanup:");

    // The three per-thread scratch buffers, and nothing the size of C.
    let allocs = m.matches("call ptr @__y_gemm_exact_alloc").count();
    assert_eq!(
        allocs, 6,
        "the M path should allocate exactly jobs, tids, live and the three \
         per-band scratch buffers - found {allocs}"
    );
    assert!(
        !m.contains("%cb"),
        "the M path must not allocate an `M * N * 8` private C: that buffer, \
         its zero-fill and the reduction that reads it back are the whole cost \
         the row partition removes"
    );
    // No reduction: the join loop is followed by frees, not by an accumulate.
    assert!(
        !m.contains("reduce"),
        "the M path must not reduce - each element of C was written by exactly \
         one band, which is what `ExactGemmMSplit.owner_unique` proves"
    );
    // ...while the K path still does both, so this is a contrast and not an
    // assertion that the reduction was deleted everywhere.
    let k = region(&ir, "many:", "mspawn.pre:");
    assert!(
        k.contains("%cpt = call ptr @__y_gemm_exact_alloc(i64 %cb)"),
        "the K path still needs its private C"
    );
    assert!(
        ir.contains("reduce.inner.body:"),
        "the K path still needs its reduction"
    );
}
