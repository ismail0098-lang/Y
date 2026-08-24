//! The compiler substitutes a packed GEMM for a naive loop nest. This runs
//! BOTH readings of the same source and compares them.
//!
//! `recognize_gemm` matches `sum += A[i,k] * B[k,j]` and replaces it with a
//! tiled, packed, threaded kernel. That substitution is the claim
//! `docs/proof_carrying_kernels.md` exists to eventually PROVE: the optimized
//! kernel computes what the naive nest computes. Until there is a proof, the
//! cheapest honest form of the claim is to run both and compare.
//!
//! ## Why the oracle is the compiler's own lowering
//!
//! Every other GEMM test in this repo checks the substituted kernel against a
//! reference typed into the test in C. That is a good oracle and it has one
//! weakness: it is a second implementation of the language's semantics, written
//! by hand, and it can drift from what the source actually means. Here the
//! oracle is the SAME source lowered by the SAME compiler with the recogniser
//! switched off (`Y_NO_GEMM_RECOGNISER=1`), so it cannot disagree with the
//! language - it IS the language.
//!
//! It also closes a gap nothing else covers: **the naive lowering is executed
//! by no other test**. Every existing GEMM test runs the substituted path, so a
//! bug in the ordinary loop/index/load lowering would be invisible to all of
//! them.
//!
//! ## Three-way, not two-way
//!
//! Two implementations that agree can be wrong together - the differential
//! would report a clean run. So each arm is ALSO checked against a
//! double-precision reference computed in the driver. Three sources, two
//! independent of each other:
//!
//!   naive lowering  vs  packed kernel   - the substitution claim
//!   naive lowering  vs  f64 reference   - is the spec itself lowered right
//!   packed kernel   vs  f64 reference   - the existing tests' question
//!
//! ## What the comparison may assert, and why it is not bit-identity
//!
//! Not yet. f32 addition is not associative, so a tiled reduction provably
//! does NOT equal the naive one bit for bit - that non-associativity is the
//! whole reason kernel verification is considered impractical, and the reason
//! exact accumulation is the programme's foundation. So today the two arms are
//! compared by relative L2 over the whole matrix.
//!
//! **When an exact accumulator is substituted, this must become bit-identity.**
//! `plan_exact_gemm` currently reports "LICENSED but not yet implemented" and
//! falls back to scalar lowering, so no `@ZeroDrift` nest is substituted and
//! that arm cannot be written yet. `the_exact_path_is_still_unsubstituted`
//! below pins that state, so whoever wires Phase 0's kernel is told - by a
//! failing test - that the tolerance here has to be replaced by equality.
//!
//! Requires `clang`; skipped with a notice otherwise, like the `ptxas`, `solc`
//! and `coqc` gates.

use std::path::{Path, PathBuf};
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

/// Straddling the blocking boundaries: `MR = 12`, `NR = 32`, the `SM_MAX_M = 8`
/// dispatch, and the copy-free tiny path. Deliberately fewer shapes than
/// `cpu_gemm_end_to_end.rs` uses - the naive arm is an interpreterless scalar
/// loop nest and is slow, so K stays modest.
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),      // degenerate
    (1, 65, 129),   // GEMV, ragged -> small-M path
    (8, 64, 64),    // exactly at the small-M dispatch boundary
    (9, 64, 64),    // one past it -> packed path
    (13, 33, 129),  // one past every tile bound
    (48, 48, 48),   // the copy-free tiny path
    (25, 17, 40),   // tiny path, both dims ragged
    (100, 100, 96), // several MC/NR blocks
];

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

int main(int argc, char **argv) {
    if (argc < 2) { printf("need an output path\n"); return 2; }
    FILE *out = fopen(argv[1], "wb");
    if (!out) { printf("cannot open %s\n", argv[1]); return 2; }

    static const int shapes[][3] = { SHAPES_HERE };
    int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));

    for (int s = 0; s < ns; ++s) {
        int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
        float *A = malloc((size_t)M * K * sizeof(float));
        float *B = malloc((size_t)K * N * sizeof(float));
        float *C = malloc((size_t)M * N * sizeof(float));
        if (!A || !B || !C) { printf("ALLOC FAIL\n"); return 2; }

        for (long i = 0; i < (long)M * K; ++i) A[i] = gen(i, 1);
        for (long i = 0; i < (long)K * N; ++i) B[i] = gen(i, 2);
        memset(C, 0, (size_t)M * N * sizeof(float));

        y_matmul(A, B, C, M, N, K);

        /* The independent oracle: f64, so it is not merely a re-ordering of
           the same f32 arithmetic. */
        double num = 0.0, den = 0.0;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k)
                    acc += (double)A[(long)i * K + k] * (double)B[(long)k * N + j];
                double d = (double)C[(long)i * N + j] - acc;
                num += d * d;
                den += acc * acc;
            }
        double rel = (den > 0.0) ? sqrt(num / den) : sqrt(num);
        printf("shape %d %d %d rel_l2 %.9g\n", M, N, K, rel);

        /* The raw result, for the arm-versus-arm comparison. */
        fwrite(C, sizeof(float), (size_t)M * N, out);

        free(A); free(B); free(C);
    }
    fclose(out);
    printf("DONE\n");
    return 0;
}
"#;

struct Arm {
    /// Every shape's C, concatenated, as the driver wrote it.
    results: Vec<f32>,
    /// Relative L2 against the f64 reference, per shape.
    rel_l2: Vec<f64>,
    /// Whether the packed kernel was actually substituted.
    substituted: bool,
}

/// Compile `Y_SOURCE` with the recogniser on or off, link the driver, run it.
fn build_and_run(dir: &Path, recognise: bool) -> Arm {
    let tag = if recognise { "packed" } else { "naive" };
    let src = dir.join(format!("{tag}.ysu"));
    std::fs::write(&src, Y_SOURCE).expect("write source");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&src).arg("--emit-llvm").current_dir(repo());
    if !recognise {
        cmd.env("Y_NO_GEMM_RECOGNISER", "1");
    }
    let out = cmd.output().expect("run Y");
    assert!(
        out.status.success(),
        "compiling the {tag} arm failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let ll = dir.join(format!("{tag}.ll"));
    let ir = std::fs::read_to_string(&ll).expect("the emitter should have written .ll");
    let substituted = ir.contains("__y_gemm");

    let shapes = SHAPES
        .iter()
        .map(|(m, n, k)| format!("{{{m},{n},{k}}}"))
        .collect::<Vec<_>>()
        .join(",");
    let driver = dir.join(format!("{tag}_driver.c"));
    std::fs::write(&driver, DRIVER.replace("SHAPES_HERE", &shapes)).expect("write driver");

    let exe = dir.join(tag);
    let cc = Command::new("clang")
        .args([
            ll.to_str().unwrap(),
            driver.to_str().unwrap(),
            "-O2",
            "-lm",
            "-lpthread",
            "-o",
            exe.to_str().unwrap(),
        ])
        .output()
        .expect("run clang");
    assert!(
        cc.status.success(),
        "linking the {tag} arm failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    let dump = dir.join(format!("{tag}.bin"));
    let run = Command::new(&exe)
        .arg(&dump)
        .output()
        .expect("run the driver");
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    assert!(
        run.status.success() && stdout.contains("DONE"),
        "the {tag} arm did not finish:\n{stdout}{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let rel_l2 = stdout
        .lines()
        .filter_map(|l| l.rsplit_once("rel_l2 ").map(|(_, v)| v.trim().parse().unwrap()))
        .collect::<Vec<f64>>();

    let bytes = std::fs::read(&dump).expect("read the dump");
    let results = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Arm {
        results,
        rel_l2,
        substituted,
    }
}

#[test]
fn the_packed_kernel_computes_what_the_naive_nest_computes() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_gemm_diff_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let naive = build_and_run(&dir, false);
    let packed = build_and_run(&dir, true);

    // The control that stops the whole file being vacuous: if the recogniser
    // did not fire, both arms are the same program and agreeing proves
    // nothing. This is `feedback-null-metrics-pass-dead-components` - a
    // difference count of zero is also what a dead harness reports.
    assert!(
        packed.substituted,
        "the packed arm was not substituted, so this test compared a program \
         against itself"
    );
    assert!(
        !naive.substituted,
        "Y_NO_GEMM_RECOGNISER did not disable the recogniser, so both arms are \
         the packed kernel"
    );

    assert_eq!(
        naive.results.len(),
        packed.results.len(),
        "the two arms produced different amounts of output"
    );
    assert_eq!(naive.rel_l2.len(), SHAPES.len(), "a shape did not report");

    // Each arm against the INDEPENDENT f64 oracle. Two implementations that
    // agree can be wrong together; this is what rules that out.
    for (i, (m, n, k)) in SHAPES.iter().enumerate() {
        assert!(
            naive.rel_l2[i] < 1e-5,
            "the NAIVE lowering is wrong at {m}x{n}x{k}: relative L2 {} against \
             the f64 reference. No other test in this repo executes this path - \
             every GEMM test runs the substituted kernel.",
            naive.rel_l2[i]
        );
        assert!(
            packed.rel_l2[i] < 1e-5,
            "the PACKED kernel is wrong at {m}x{n}x{k}: relative L2 {}",
            packed.rel_l2[i]
        );
    }

    // The substitution claim itself, arm against arm.
    //
    // Relative L2 over each shape rather than per element: the packed kernel
    // sums in a different order, so a per-element tolerance produces false
    // alarms at large K without catching anything a norm misses.
    let mut off = 0usize;
    for (m, n, k) in SHAPES {
        let len = m * n;
        let (a, b) = (&naive.results[off..off + len], &packed.results[off..off + len]);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = f64::from(*x) - f64::from(*y);
            num += d * d;
            den += f64::from(*x) * f64::from(*x);
        }
        let rel = if den > 0.0 { (num / den).sqrt() } else { num.sqrt() };
        assert!(
            rel < 1e-5,
            "the packed kernel and the naive nest disagree at {m}x{n}x{k}: \
             relative L2 {rel}. The substitution is supposed to preserve what \
             the source computes."
        );
        off += len;
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// The forcing function for Phase 0.
///
/// While no exact accumulator is substituted, the differential above can only
/// assert a TOLERANCE - f32 addition is not associative, so a tiled reduction
/// genuinely does not equal the naive one bit for bit.
///
/// The moment `plan_exact_gemm`'s kernel is wired in, that stops being true:
/// integer addition IS associative, so an exact substituted kernel must be
/// BIT-IDENTICAL to the naive nest, and a tolerance would silently accept a
/// kernel that is merely close. This test fails when that day comes, which is
/// the point - it is how the person wiring it is told to come back here.
#[test]
fn the_exact_path_is_still_unsubstituted_so_the_tolerance_above_is_still_right() {
    use y::cpu_gemm::{plan_exact_gemm, DriftAccumulator, ExactGemmPlan};
    let licensed = DriftAccumulator {
        ty: "F32".into(),
        bounds: None,
        a_bounds: Some((-1024.0, 1024.0)),
        b_bounds: Some((-1024.0, 1024.0)),
    };
    // The licence is granted...
    assert!(
        matches!(plan_exact_gemm(&licensed), ExactGemmPlan::Vnni { .. }),
        "a nest with bounded integral operands must still be licensed"
    );
    // ...and the emitter still does not act on it. `try_emit_gemm_kernel`
    // reports the licence and returns None, so a `@ZeroDrift` nest lowers
    // scalar-exact rather than being substituted.
    let ir = {
        let dir = std::env::temp_dir().join(format!("y_gemm_exact_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("exact.ysu");
        let body = Y_SOURCE.replace(
            "            let mut sum: F32 = 0.0;",
            "            @ZeroDrift\n            @bounds(-1024.0, 1024.0)\n            let mut sum: F32 = 0.0;",
        );
        std::fs::write(&src, body).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&src)
            .arg("--emit-llvm")
            .current_dir(repo())
            .output()
            .expect("run Y");
        assert!(
            out.status.success(),
            "the @ZeroDrift nest must still compile:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let ir = std::fs::read_to_string(dir.join("exact.ll")).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        ir
    };
    assert!(
        !ir.contains("__y_gemm_exact_vnni"),
        "THE EXACT KERNEL IS NOW SUBSTITUTED. The differential in this file \
         compares the two arms by a TOLERANCE, which was correct only while the \
         substituted kernel accumulated in f32. An exact kernel accumulates in \
         integers, integer addition is associative, and the tiled K-split \
         result must therefore be BIT-IDENTICAL to the naive nest. Replace the \
         relative-L2 comparison with equality for the exact arm - a tolerance \
         would accept a kernel that is merely close, which is exactly the claim \
         `docs/proof_carrying_kernels.md` is selling against."
    );
}
