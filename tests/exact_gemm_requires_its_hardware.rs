//! `plan_exact_gemm` consulted no hardware at all.
//!
//! It licensed the exact `vpdpwssd` GEMM on operand magnitudes alone, so Y
//! emitted `vpdpwssd` on a machine that has none - an illegal instruction, on
//! whatever machine the compile happened to run on. This repository's own rule
//! is that **the one genuine hardware requirement must REFUSE, not emit**; that
//! is `require_fp8_hardware` for `e4m3` tensor cores, and there was no
//! counterpart on the CPU side.
//!
//! **The refusal is CHEAPER here than the FP8 one, and saying why is the
//! point.** `require_fp8_hardware` must refuse the whole kernel: the
//! instruction is absent and there is no other way to compute it. This refuses
//! only the FAST path. The scalar lowering still honours `@ZeroDrift` and is
//! still bit-for-bit exact, so the cost of refusing is time, not an error - and
//! `the_refusal_does_not_change_the_answer` is what turns that from a claim
//! into a measurement. Measured at 512x512x2048: **7.5 ms substituted, 510 ms
//! scalar**, same 64-bit checksum to the digit.
//!
//! **The check must be a BICONDITIONAL.** "Refuse always" satisfies every
//! assertion about the refusal and silently deletes a working path on the
//! hardware that has it - the same trap the FP8 gate records. So
//! `the_fast_path_is_still_taken_on_hardware_that_has_it` runs on this machine
//! with nothing forced and asserts the substitution DID happen.
//!
//! **The escape hatch can only go DOWN.** `Y_NO_AVX512_VNNI=1` forces the
//! refusal; there is deliberately no variable that claims hardware. An override
//! in that direction would let a caller produce a binary that faults, so
//! `the_override_cannot_claim_hardware_the_machine_lacks` reads the source and
//! pins that the predicate's only early return is `false`.
//!
//! **And the licence must stay host-independent.** `VnniExact::license` is a
//! statement about operand magnitudes and int32 overflow; it is exhausted over
//! the whole int16 domain by `tests/exact_gemm_licence_obligations.rs` and
//! `exact_gemm_certificate` instantiates it in the emitted Coq. A licence that
//! consulted the host would make the CERTIFICATE consult the host, which is the
//! one thing it must not do - so the gate lives in `plan_exact_gemm`, before
//! the licence, and `the_licence_does_not_consult_the_host` pins that
//! `zero_drift.rs` never reaches for the machine.

use std::path::PathBuf;
use std::process::Command;

const SOURCE: &str = r#"
kernel exact_matmul(A: GlobalMemory<I16>, B: GlobalMemory<I16>, C: GlobalMemory<I64>, M: I32, N: I32, K: I32) {
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

/// Small enough that the 68x-slower scalar arm still finishes promptly, large
/// enough that K crosses several flush intervals on the substituted arm.
const DRIVER: &str = r#"
#include <stdint.h>
#include <stdio.h>
#define M 37
#define N 23
#define K 601
static int16_t A[M*K], B[K*N];
static int64_t C[M*N];
void exact_matmul(const int16_t*, const int16_t*, int64_t*, int32_t, int32_t, int32_t);
static int16_t gen(long i, long s) {
    unsigned long h = i * 2654435761UL + s * 40503UL; h ^= h >> 13;
    return (int16_t)((long)(h % 2049) - 1024);
}
int main(void) {
    for (long i = 0; i < M*K; i++) A[i] = gen(i, 1);
    for (long i = 0; i < K*N; i++) B[i] = gen(i, 2);
    for (long i = 0; i < M*N; i++) C[i] = 0;
    exact_matmul(A, B, C, M, N, K);
    // A checksum alone can agree by cancellation; mix in the position.
    long long s = 0, w = 0;
    for (long i = 0; i < M*N; i++) { s += C[i]; w += C[i] * (i + 1); }
    printf("sum %lld weighted %lld\n", s, w);
    return 0;
}
"#;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Does this machine have VNNI, asked WITHOUT going through the function under
/// test?
///
/// The first version of this file guarded the biconditional with
/// `sentinel::host_has_avx512_vnni()` - so a mutation making that always answer
/// `false` (an over-refusal, the exact failure the biconditional exists to
/// catch) made the test SKIP ITSELF and report ok while four other suites went
/// red. `feedback-conditional-gates-skip-silently`, in the test written to
/// avoid it. The guard now reads `/proc/cpuinfo`, a different mechanism
/// entirely, and falls back to the std macro off Linux.
fn machine_really_has_vnni() -> bool {
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        return info.contains("avx512_vnni");
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// The tag is in the signature rather than in a comment asking the caller to
/// remember: two tests sharing a temp-dir name is a race this repository has
/// hit five times.
///
/// `force_off` sets `Y_NO_AVX512_VNNI` in the CHILD, because mutating this
/// process's environment races the rest of the suite.
fn emit(tag: &str, force_off: bool) -> (PathBuf, String, String) {
    let dir = std::env::temp_dir().join(format!("y_exact_hw_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let src = dir.join("m.ysu");
    std::fs::write(&src, SOURCE).expect("write source");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&src).arg("--emit-llvm").current_dir(repo());
    if force_off {
        cmd.env("Y_NO_AVX512_VNNI", "1");
    } else {
        cmd.env_remove("Y_NO_AVX512_VNNI");
    }
    let out = cmd.output().expect("run Y");
    assert!(
        out.status.success(),
        "the exact nest must compile either way:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ir = std::fs::read_to_string(dir.join("m.ll")).expect("emitted IR");
    (dir, ir, log)
}

#[test]
fn without_the_hardware_no_vpdpwssd_is_emitted() {
    let (dir, ir, log) = emit("refuse", true);
    assert_eq!(
        ir.matches("vpdpwssd").count(),
        0,
        "a machine without AVX-512 VNNI must not be handed `vpdpwssd`"
    );
    assert!(
        !ir.contains("__y_gemm_exact_vnni"),
        "the exact kernel module must not be emitted at all"
    );
    assert!(
        log.contains("AVX-512 VNNI"),
        "the advisory must name the missing hardware, not just decline: {log}"
    );
    assert!(
        log.to_lowercase().contains("still exact"),
        "and must say the scalar fallback is still exact, or a user reads this \
         as losing the @ZeroDrift guarantee: {log}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The other half of the biconditional. Without it, "refuse always" passes
/// every assertion above while deleting a working path.
#[test]
fn the_fast_path_is_still_taken_on_hardware_that_has_it() {
    if !machine_really_has_vnni() {
        eprintln!("SKIP: this machine has no AVX-512 VNNI, so there is no fast path to take");
        return;
    }
    let (dir, ir, log) = emit("accept", false);
    assert!(
        ir.matches("vpdpwssd").count() > 0,
        "on a VNNI machine the exact kernel must still be substituted"
    );
    assert!(
        log.contains("EXACT vpdpwssd kernel substituted"),
        "and must say so: {log}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The load-bearing one: refusing the fast path costs time, not exactness.
///
/// Both arms are compiled and RUN, and the two 64-bit results must agree to the
/// digit. A plain sum can agree by cancellation, so the driver also reports a
/// position-weighted sum.
#[test]
fn the_refusal_does_not_change_the_answer() {
    if !have("clang") {
        eprintln!("SKIP: clang not found");
        return;
    }
    if !machine_really_has_vnni() {
        eprintln!("SKIP: both arms would take the scalar path on this machine");
        return;
    }
    let mut answers = Vec::new();
    for (tag, off) in [("run_on", false), ("run_off", true)] {
        let (dir, ir, _) = emit(tag, off);
        assert_eq!(
            ir.matches("vpdpwssd").count() > 0,
            !off,
            "the arms must actually differ in whether the kernel was substituted"
        );
        let drv = dir.join("d.c");
        std::fs::write(&drv, DRIVER).expect("write driver");
        let exe = dir.join("a.out");
        let build = Command::new("clang")
            .args(["-O2", "-o"])
            .arg(&exe)
            .arg(dir.join("m.ll"))
            .arg(&drv)
            .args(["-lpthread", "-lm"])
            .output()
            .expect("clang");
        assert!(
            build.status.success(),
            "the {tag} arm must link:\n{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let run = Command::new(&exe).output().expect("run");
        assert!(run.status.success(), "the {tag} arm must run");
        answers.push((
            tag,
            String::from_utf8_lossy(&run.stdout).trim().to_string(),
            dir,
        ));
    }
    assert_eq!(
        answers[0].1, answers[1].1,
        "the scalar fallback must be bit-identical to the substituted kernel - \
         that is what `@ZeroDrift` promises, and it is the whole reason refusing \
         the fast path is safe. substituted: {} / scalar: {}",
        answers[0].1, answers[1].1
    );
    // Non-vacuity: a driver that printed nothing would pass the comparison.
    assert!(
        answers[0].1.contains("sum ") && answers[0].1.contains("weighted "),
        "the driver must report both figures: {}",
        answers[0].1
    );
    assert!(
        !answers[0].1.contains("sum 0 weighted 0"),
        "a zero result would compare equal whatever the kernel did: {}",
        answers[0].1
    );
    for (_, _, d) in answers {
        let _ = std::fs::remove_dir_all(&d);
    }
}

/// The escape hatch may only make the compiler more conservative.
///
/// A variable that let a caller CLAIM hardware would let it produce a binary
/// that faults. Asserted at the source, because there is no way to test the
/// absence of a variable by running the compiler.
#[test]
fn the_override_cannot_claim_hardware_the_machine_lacks() {
    let src = std::fs::read_to_string(repo().join("src/sentinel.rs")).expect("sentinel.rs");
    let start = src
        .find("\npub fn host_has_avx512_vnni()")
        .expect("the predicate must exist");
    let body = &src[start..];
    let end = body.find("\n}\n").expect("function end") + 3;
    let body = &body[..end];

    let reads: Vec<&str> = body
        .lines()
        .filter(|l| l.contains("env::var"))
        .map(|l| l.trim())
        .collect();
    assert_eq!(
        reads.len(),
        1,
        "the predicate must read exactly one variable: {reads:?}"
    );
    assert!(
        reads[0].contains("Y_NO_AVX512_VNNI"),
        "and it must be the one that turns the feature OFF: {reads:?}"
    );
    // The only early return guarded by that read must be `false`.
    let after = body.split("Y_NO_AVX512_VNNI").nth(1).expect("split");
    let ret = after
        .lines()
        .find(|l| l.contains("return"))
        .expect("the override must return something");
    assert!(
        ret.contains("false"),
        "the override must be able to say only `false`; an override that could \
         return `true` would let a caller claim hardware and emit a faulting \
         binary: {ret}"
    );
}

/// The gate must not leak into the licence, or the emitted certificate starts
/// depending on the machine that compiled it.
#[test]
fn the_licence_does_not_consult_the_host() {
    let src = std::fs::read_to_string(repo().join("src/zero_drift.rs")).expect("zero_drift.rs");
    for needle in ["sentinel", "host_has_", "is_x86_feature_detected"] {
        assert!(
            !src.contains(needle),
            "`zero_drift.rs` must not consult the host ({needle}). `VnniExact::license` \
             is exhausted over the int16 domain and instantiated by the emitted Coq \
             certificate; making it host-dependent makes the certificate \
             host-dependent. The hardware gate belongs in `plan_exact_gemm`."
        );
    }
    // Non-vacuity: the file must actually be the licence.
    assert!(
        src.contains("fn license"),
        "zero_drift.rs must still define the licence, or this asserts nothing"
    );
}

/// The predicate must agree with the machine, and this test cannot skip itself.
///
/// It is what catches an over-refusal - a `host_has_avx512_vnni` that always
/// answers `false` - which every other assertion here would tolerate: the
/// refusal tests pass trivially and the biconditional would skip. Both readings
/// come from different mechanisms (`/proc/cpuinfo` against the std macro's
/// CPUID + XGETBV), so they can only agree by both being right.
#[test]
fn the_predicate_agrees_with_the_machine() {
    // Guard against the env override leaking in from a parent process; the
    // child-process tests set it deliberately, this one must see the machine.
    assert!(
        std::env::var("Y_NO_AVX512_VNNI").is_err(),
        "this test reads the real machine, so the override must not be set"
    );
    assert_eq!(
        y::sentinel::host_has_avx512_vnni(),
        machine_really_has_vnni(),
        "`host_has_avx512_vnni` disagrees with /proc/cpuinfo. Answering `false` \
         on a machine that HAS the feature silently deletes the fast path; \
         answering `true` on one that lacks it emits an instruction that faults."
    );
}
