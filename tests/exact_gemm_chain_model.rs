//! The chain of `proofs/ExactGemmChain.v`, on the real packers and the real
//! `vpdpwssd` micro-kernel.
//!
//! The proof's headline is
//! `the_emitted_lane_computes_the_source_dot_product`: run the emitted flush
//! schedule, accumulating each chunk in int32 and adding it into an int64
//! running sum, over the panels the emitted packers produce, through the
//! emitted routing - and lane `l` of vector `v` for row `i` holds exactly
//! `sum_k A[i][k] * B[k][16v+l]`, the dot product of the SOURCE matrices.
//!
//! **The reference here is over the SOURCE, which is what makes this test the
//! chain rather than one of its links.** `exact_gemm_register_tile_model.rs`
//! checks the micro-kernel against a reference over the PANELS (packers taken
//! out of the loop, deliberately); `exact_gemm_panel_model.rs` checks the
//! packers alone. Only a source-level reference run through both can catch a
//! packing error and a routing error that are inverse to each other - and the
//! register-tile file demonstrates that such a pair exists and passes every
//! full-GEMM suite in the repo.
//!
//! **It also crosses more than one flush chunk**, which none of the sibling
//! model tests do: `kc = 133` gives 67 k-pairs against a 64-pair interval, so
//! the clamped final chunk is a real case rather than a corner one. That is the
//! part of the chain `ExactGemmMicro.flush_exact` supplies and the register
//! tile knows nothing about.
//!
//! Every stride differs from its extent (`lda != kc`, `ldb != ncols`), per the
//! rule the tiling work established.
//!
//! **What it catches, and what it does NOT - measured, not claimed.** Five
//! mutations of the emitter across every exact-GEMM suite:
//!
//! | mutation | packing | panel | regtile | micro | thr-inv | exact-thr | chain |
//! |---|---|---|---|---|---|---|---|
//! | flush OVERWRITES C instead of accumulating | FAIL | ok | ok | ok | FAIL | FAIL | **FAIL** |
//! | pack_b vector stride 32 -> 16 | FAIL | FAIL | ok | ok | FAIL | FAIL | **FAIL** |
//! | pack_b zero-fill dropped | ok | FAIL | ok | ok | ok | ok | **FAIL** |
//! | compensating pair (pack_b v^1 AND store column v^1) | ok | FAIL | FAIL | ok | ok | ok | **ok** |
//!
//! The last row is the honest one: **this file does not subsume
//! `exact_gemm_register_tile_model.rs`, and cannot.** It composes the packers
//! with the routing, so a packing error and a routing error that are inverse
//! to each other cancel here exactly as they do in a full GEMM. Isolating that
//! pair needs panels stated by the test (the register tile) or a panel checked
//! slot by slot (the panel model). Adding a test does not retire the tests it
//! resembles.
//!
//! What it does add over the full-GEMM suites is a SOURCE-level reference with
//! the outer tiling driver removed, so a disagreement points at the packers or
//! the micro-kernel rather than anywhere in the pipeline - and the licence
//! boundary reached from source operands rather than from hand-built panels.

use std::process::Command;

use y::cpu_gemm::{VNNI_MR, VNNI_NR};
use y::zero_drift::VnniExact;

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn host_has_vnni() -> bool {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.contains("avx512_vnni"))
        .unwrap_or(false)
}

const WRAPPERS: &str = r#"
define void @y_test_pack_a(ptr noalias %s, i64 %lda, i64 %m, i64 %kc, ptr noalias %d) {
  call void @__y_gemm_vnni_pack_a(ptr %s, i64 %lda, i64 %m, i64 %kc, ptr %d)
  ret void
}
define void @y_test_pack_b(ptr noalias %s, i64 %ldb, i64 %kc, i64 %n, ptr noalias %d) {
  call void @__y_gemm_vnni_pack_b(ptr %s, i64 %ldb, i64 %kc, i64 %n, ptr %d)
  ret void
}
"#;

/// The schedule constants, as C `#define`s taken from `cpu_gemm.rs` itself.
///
/// **This driver used to hardcode `#define MR 6` / `#define NR 64`.** That is a
/// second copy of the tile shape, in the half of the harness that allocates the
/// buffers the emitted kernel writes into - so a change to `VNNI_MR` did not
/// make this test report a schedule mismatch, it made the test disagree with
/// itself and crash or mis-size a panel. Found by diagnosing exactly that: with
/// `VNNI_MR = 8`, `exact_gemm_thread_invariance` (which checks the ANSWER)
/// passes, while seven harnesses carrying their own `6` fail.
///
/// Same defect `proofs/ExactGemmSchedule.v` exists to remove, one layer down.
fn schedule_defines() -> String {
    format!("#define MR {VNNI_MR}\n#define NR {VNNI_NR}\n")
}

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


void y_test_pack_a(int16_t *src, long lda, long mrows, long kc, int16_t *dst);
void y_test_pack_b(int16_t *src, long ldb, long kc, long ncols, int16_t *dst);
void __y_gemm_micro_vnni(const int16_t *Ap, const int16_t *Bp, int64_t *C,
                         long ldc, long kpairs);

int main(int argc, char **argv) {
    long mrows = atol(argv[1]);
    long kc    = atol(argv[2]);
    long ncols = atol(argv[3]);
    long lda   = atol(argv[4]);
    long ldb   = atol(argv[5]);
    long mag   = atol(argv[6]);   /* 0 = position-encoded, else a constant */
    long c0on  = atol(argv[7]);   /* 1 = pre-load C, to exercise the ACCUMULATE */
    long kp    = (kc + 1) / 2;

    int16_t *A = malloc(sizeof(int16_t) * (size_t)(MR + 2) * lda);
    int16_t *B = malloc(sizeof(int16_t) * (size_t)(kc + 2) * ldb);
    for (long i = 0; i < MR + 2; i++)
        for (long k = 0; k < lda; k++)
            A[i * lda + k] = (int16_t)(mag ? mag : (1 + 3 * i + 5 * k));
    for (long k = 0; k < kc + 2; k++)
        for (long j = 0; j < ldb; j++)
            B[k * ldb + j] = (int16_t)(mag ? mag : (2 + 7 * k + 3 * j));

    int16_t *Ap = aligned_alloc(64, (size_t)kp * MR * 2 * 2);
    int16_t *Bp = aligned_alloc(64, (size_t)kp * NR * 2 * 2);
    int64_t *C  = aligned_alloc(64, (size_t)MR * NR * 8);

    y_test_pack_a(A, lda, mrows, kc, Ap);
    y_test_pack_b(B, ldb, kc, ncols, Bp);

    /* The kernel ACCUMULATES into C, so the caller owns the initial value.
       With `c0on` the tile is pre-loaded, which is the only way to tell an
       accumulate from an assignment at a single flush chunk - and the only way
       to check that a DEAD position is left exactly as it was rather than
       merely left at zero. `proofs/ExactGemmChain.v` states both as
       `C0 i j + ...` and `a_dead_position_leaves_c_untouched`. */
    for (long i = 0; i < MR; ++i)
        for (long j = 0; j < NR; ++j)
            C[i * NR + j] = c0on ? (int64_t)(1000 + 100 * i + j) : 0;
    __y_gemm_micro_vnni(Ap, Bp, C, NR, kp);

    long wrong = 0;
    long long first_got = 0, first_want = 0;
    for (long i = 0; i < MR; i++)
        for (long j = 0; j < NR; j++) {
            /* The reference is over the SOURCE matrices, masked exactly as a
               ragged tile is: a dead row or column contributes nothing. */
            int64_t want = c0on ? (int64_t)(1000 + 100 * i + j) : 0;
            if (i < mrows && j < ncols)
                for (long k = 0; k < kc; k++)
                    want += (int64_t)A[i * lda + k] * (int64_t)B[k * ldb + j];
            if (C[i * NR + j] != want) {
                if (wrong == 0) { first_got = C[i * NR + j]; first_want = want; }
                wrong++;
            }
        }
    /* The (0,0) reference, printed unconditionally: `first_want` is only set
       when something disagrees, so it says nothing on a passing run. */
    int64_t ref00 = c0on ? 1000 : 0;
    if (0 < mrows && 0 < ncols)
        for (long k = 0; k < kc; k++)
            ref00 += (int64_t)A[k] * (int64_t)B[k * ldb];

    printf("wrong %ld\n", wrong);
    printf("first_got %lld\n", first_got);
    printf("first_want %lld\n", first_want);
    printf("ref00 %lld\n", ref00);
    printf("got00 %lld\n", (long long)C[0]);
    printf("DONE\n");
    return 0;
}
"#;

/// **`tag` is per-test, not decoration.** Both tests here call this, and with a
/// shared directory name one `remove_dir_all` lands while the other is writing
/// its IR - the intermittent race this repo has now hit three times (the GPU
/// `.ptx` harness, `tests/backend_differential.rs`, here). It presented as
/// `write IR` failing with the module perfectly valid.
fn build(tag: &str) -> Option<std::path::PathBuf> {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return None;
    }
    if !host_has_vnni() {
        eprintln!("skipping: host CPU has no avx512_vnni");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("y_chain_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let ll = dir.join("m.ll");
    let mut module = y::cpu_gemm::emit_vnni_gemm_module(VnniExact::DEFAULT_FLUSH_K_PAIRS);
    module.push_str(WRAPPERS);
    std::fs::write(&ll, module).expect("write IR");
    let drv = dir.join("drv.c");
    std::fs::write(&drv, schedule_defines() + DRIVER).expect("write driver");
    let exe = dir.join("run");
    let cc = Command::new("clang")
        .args(["-O2", "-x", "ir"])
        .arg(&ll)
        .args(["-x", "c"])
        .arg(&drv)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("clang");
    assert!(
        cc.status.success(),
        "link failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );
    Some(exe)
}

/// `(wrong, got00, ref00)` - the (0,0) cell is reported unconditionally so the
/// licence test can assert on a value that exists on a PASSING run too.
fn run(
    exe: &std::path::Path,
    mrows: usize,
    kc: usize,
    ncols: usize,
    mag: i64,
    c0: bool,
) -> (i64, i64, i64) {
    let out = Command::new(exe)
        .args([
            mrows.to_string(),
            kc.to_string(),
            ncols.to_string(),
            (kc + 5).to_string(),
            (ncols + 9).to_string(),
            mag.to_string(),
            (c0 as i32).to_string(),
        ])
        .output()
        .expect("run");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success() && text.contains("DONE"),
        "driver did not finish:\n{text}"
    );
    let field = |tag: &str| -> i64 {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("{tag} ")))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or_else(|| panic!("no `{tag}` line in:\n{text}"))
    };
    if field("wrong") > 0 {
        eprintln!(
            "  first disagreement: got {} want {}",
            field("first_got"),
            field("first_want")
        );
    }
    (field("wrong"), field("got00"), field("ref00"))
}

/// Shapes. The last two cross more than one flush chunk: `kc = 133` is 67
/// k-pairs against the 64-pair interval, so the clamped final chunk carries
/// three k-pairs.
const SHAPES: &[(usize, usize, usize)] = &[
    (VNNI_MR, 8, VNNI_NR),
    (VNNI_MR, 7, VNNI_NR),
    (4, 8, VNNI_NR),
    (VNNI_MR, 8, 53),
    (4, 7, 53),
    (VNNI_MR, 133, VNNI_NR),
    (4, 133, 53),
];

/// **The chain: packers, then the real `vpdpwssd` kernel, against a
/// SOURCE-level reference.**
#[test]
fn the_kernel_computes_the_source_dot_product() {
    let Some(exe) = build("dot") else { return };

    // Both initial states, because a kernel that ASSIGNS rather than
    // accumulates is indistinguishable from a correct one when C starts zeroed
    // and there is only one flush chunk.
    for &c0 in &[false, true] {
    for &(mrows, kc, ncols) in SHAPES {
        let (wrong, got00, ref00) = run(&exe, mrows, kc, ncols, 0, c0);
        assert_eq!(
            wrong, 0,
            "shape ({mrows}, {kc}, {ncols}), C pre-loaded = {c0}: {wrong} of {} \
             tile positions disagree with the source dot product",
            VNNI_MR * VNNI_NR
        );
        // Non-vacuity: a kernel that wrote nothing would leave the zeroed C and
        // agree with a reference that was also zero.
        assert!(
            ref00 > 0 && got00 == ref00,
            "shape ({mrows}, {kc}, {ncols}): the (0,0) reference is {ref00}, so \
             this shape proves nothing about a kernel that never writes"
        );
    }
    }
}

/// **The licence is load-bearing for the whole chain, and the boundary is one
/// unit wide - the same pair `proofs/ExactGemmChain.v` refutes symbolically.**
///
/// `VnniExact::max_operand_magnitude` licenses `m` exactly when
/// `2 * Fl * m^2 <= i32::MAX`; at the shipped `Fl = 64` that is `m <= 4095`.
/// With constant operands - the worst case random data never reaches - 4095 is
/// exact and 4096 is not merely imprecise, it has the wrong SIGN.
///
/// `exact_gemm_micro_model.rs` observes the same boundary with HAND-BUILT
/// panels; this one reaches it from source matrices through the real packers,
/// so it is the licence's obligation stated over the operands a user actually
/// supplies.
#[test]
fn the_licence_boundary_is_one_unit_wide_through_the_whole_chain() {
    let Some(exe) = build("licence") else { return };
    let scheme = VnniExact::new(VnniExact::DEFAULT_FLUSH_K_PAIRS)
        .expect("the shipping flush interval must be valid");
    let m = scheme.max_operand_magnitude() as i64;
    assert_eq!(m, 4095, "the licensed magnitude moved; this test's numbers are stale");

    // kc = 128 is exactly one flush interval of 64 k-pairs.
    let (wrong_ok, got_ok, want_ok) = run(&exe, VNNI_MR, 128, VNNI_NR, m, false);
    assert_eq!(
        wrong_ok, 0,
        "the largest LICENSED operand magnitude ({m}) already disagrees, so the \
         refutation below would not be about the licence"
    );
    assert_eq!(want_ok, 128 * m * m, "the licensed fixture's reference moved");
    assert_eq!(got_ok, want_ok);

    let (wrong_bad, got_bad, want_bad) = run(&exe, VNNI_MR, 128, VNNI_NR, m + 1, false);
    let _ = want_bad;
    assert!(
        wrong_bad > 0,
        "operand magnitude {} exceeds the licence by exactly one and the kernel \
         still agreed - either the flush interval moved or nothing is \
         accumulating in int32",
        m + 1
    );
    assert_eq!(
        128 * (m + 1) * (m + 1),
        2_147_483_648,
        "the fixture no longer sits on the edge"
    );
    assert_eq!(
        got_bad, -2_147_483_648,
        "expected the int32 accumulator to wrap to i32::MIN, which is the value \
         `violating_the_licence_breaks_the_chain` computes symbolically"
    );
}
