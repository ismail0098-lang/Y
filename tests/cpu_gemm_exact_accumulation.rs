//! A `@ZeroDrift` matmul must never be lowered to the f32 packed kernel.
//!
//! `recognize_gemm` used to refuse a nest whose accumulator carried
//! `@ZeroDrift`, so the request could not reach the emitter at all. It records
//! the request now — that is the plumbing Phase 0 of
//! `docs/proof_carrying_kernels.md` needs — and recording it opens a seam that
//! did not previously exist: the recogniser says "this is a GEMM", and the
//! obvious next step is to emit the packed kernel for it.
//!
//! **That would be a silently wrong answer.** The packed kernel accumulates in
//! f32, which is not exact and not order-independent, so it does not compute
//! what `@ZeroDrift` asked for. The program would still run, still produce
//! plausible numbers, and quietly fail the one guarantee it requested — the
//! failure mode CLAUDE.md's design rule exists to prevent, and the reason the
//! original refusal was correct even though it was in the wrong place.
//!
//! Until an exact kernel exists, `try_emit_gemm_kernel` returns `None` for a
//! drifted nest and it falls through to scalar lowering, which honours
//! `@ZeroDrift` properly. Slow and right rather than fast and wrong.
//!
//! When the exact kernel lands, the first assertion here changes to name that
//! kernel instead — it must never simply be deleted.
//!
//! Run with:  cargo test --release --test cpu_gemm_exact_accumulation

use std::path::PathBuf;
use std::process::Command;

const PLAIN: &str = r#"
kernel mm(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, M: I32, N: I32, K: I32) {
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

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Compile `src` through the real binary and return the emitted LLVM IR.
fn emit_ir(tag: &str, src: &str) -> String {
    emit_ir_and_log(tag, src).0
}

/// As [`emit_ir`], but also returns the compiler's stdout.
///
/// The `@ZeroDrift` advisories (`drift_report`) are printed there, not written
/// into the IR, so a test about what the compiler *told the user* has to read
/// the log. That distinction matters here: the advisory is the only signal a
/// user gets that their matmul is on the slow path, and it is the thing most
/// likely to be silently dropped by a refactor.
fn emit_ir_and_log(tag: &str, src: &str) -> (String, String) {
    let dir = std::env::temp_dir().join(format!("y_exact_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("mm.ysu");
    std::fs::write(&path, src).expect("write Y source");

    let y_bin = repo_root().join("target/release/Y");
    assert!(
        y_bin.exists(),
        "build the compiler first: cargo build --release"
    );
    let out = Command::new(&y_bin)
        .arg(&path)
        .arg("--emit-llvm")
        .current_dir(repo_root())
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "Y failed on the {} case:\n{}\n{}",
        tag,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ir = std::fs::read_to_string(dir.join("mm.ll")).expect("read emitted IR");
    let _ = std::fs::remove_dir_all(&dir);
    (ir, String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn a_zero_drift_matmul_is_not_lowered_to_the_f32_kernel() {
    let drifted = PLAIN.replace(
        "            let mut sum: F32 = 0.0;",
        "            @bounds(min=-1000, max=1000)\n            @ZeroDrift\n            let mut sum: F32 = 0.0;",
    );
    let ir = emit_ir("drift", &drifted);

    assert!(
        !ir.contains(&format!("call void @{}(", y::cpu_gemm::KERNEL_NAME)),
        "a @ZeroDrift accumulator was lowered to the f32 packed kernel, which \
         accumulates in f32 and is therefore neither exact nor \
         order-independent — the guarantee the source asked for is gone, and \
         the program still runs and still looks right"
    );

    // The exactness machinery must actually be doing something, or the first
    // assertion passes for the wrong reason (e.g. the directive silently
    // ignored). @ZeroDrift stores the accumulator as an integer, so the
    // emitted IR carries an integer alloca for it rather than a float one.
    assert!(
        ir.contains("[Y ZERO-DRIFT]") || ir.contains("%sum = alloca i"),
        "the @ZeroDrift accumulator does not appear to have been lowered as an \
         exact fixed-point value at all"
    );
}

/// An unlicensed nest is told WHY it is on the slow path, and told that it is
/// still exact.
///
/// Both halves matter. Without the reason the user has no way to reach the fast
/// path — the fix is tighter `@bounds` on the operands, which is not guessable.
/// Without the reassurance that scalar lowering is still exact, the message
/// reads as "your guarantee was dropped", which is the opposite of the truth
/// and would send someone hunting a bug that does not exist.
#[test]
fn an_unlicensed_matmul_says_why_the_fast_path_was_skipped() {
    let drifted = PLAIN.replace(
        "            let mut sum: F32 = 0.0;",
        "            @bounds(min=-1000, max=1000)\n            @ZeroDrift\n            let mut sum: F32 = 0.0;",
    );
    let (_, log) = emit_ir_and_log("unlicensed", &drifted);

    assert!(
        log.contains("still EXACT"),
        "the advisory must say the program is still exact, or it reads as a \
         dropped guarantee:\n{log}"
    );
    assert!(
        log.contains("cancel"),
        "the advisory must explain that the accumulator's bound does not imply \
         an operand bound, which is the whole reason this nest is unlicensed:\n{log}"
    );
}

/// A licensed nest reports the licence and the terms it was granted on.
///
/// This is the control for the test above: without it, "the advisory said the
/// fast path was skipped" would also pass if the licence NEVER succeeded — a
/// permanently-unlicensed compiler produces exactly the same message on every
/// input, and the reason text would still match.
#[test]
fn a_licensed_matmul_reports_the_licence_and_its_terms() {
    let licensed = PLAIN
        .replace(
            "            let mut sum: F32 = 0.0;",
            "            @bounds(min=-1000, max=1000)\n            @ZeroDrift\n            let mut sum: F32 = 0.0;",
        )
        .replace(
            "                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);",
            "                @bounds(min=-1024, max=1024)\n                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);",
        )
        .replace(
            "                let b_val: F32 = block_ptr2d_load(B, k, j, N, K, N);",
            "                @bounds(min=-1024, max=1024)\n                let b_val: F32 = block_ptr2d_load(B, k, j, N, K, N);",
        );
    let (ir, log) = emit_ir_and_log("licensed", &licensed);

    assert!(
        log.contains("LICENSED"),
        "operands bounded by 1024 must be licensed at the default flush \
         interval:\n{log}"
    );
    // The terms have to appear, not just the verdict: a licence whose interval
    // is not stated cannot be checked against the kernel that eventually
    // consumes it.
    assert!(
        log.contains("1024") && log.contains("64 k-pairs"),
        "the advisory must state the operand bound and the flush interval it \
         was granted against:\n{log}"
    );

    // And being licensed must NOT yet cause the f32 packed kernel to be
    // emitted — the licence is for an exact kernel that does not exist. This is
    // the assertion that would catch a future change wiring the licence to the
    // wrong emitter.
    assert!(
        !ir.contains(&format!("call void @{}(", y::cpu_gemm::KERNEL_NAME)),
        "a licensed @ZeroDrift nest was lowered to the f32 packed kernel, which \
         is not exact"
    );
}

/// The control. Without this, "the f32 kernel was not used" would also pass if
/// the GEMM recogniser had stopped firing altogether — which is the same
/// symptom for a completely different reason, and would hide a real regression
/// in the fast path.
#[test]
fn a_plain_matmul_still_gets_the_f32_kernel() {
    let ir = emit_ir("plain", PLAIN);
    assert!(
        ir.contains(&format!("call void @{}(", y::cpu_gemm::KERNEL_NAME)),
        "the plain matmul stopped being recognised — the previous test would \
         now pass vacuously"
    );
}
