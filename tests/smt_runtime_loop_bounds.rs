//! Runtime I32 loop bounds use the same signed comparison in SMT, LLVM and PTX.
//! Negative dimensions describe empty ascending loops; they are not assumed
//! positive to make an invariant proof succeed.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use y::type_checker::{z3_candidates, TypeChecker};

#[path = "common/verification.rs"]
mod verification;

#[path = "common/pinned.rs"]
mod pinned;

fn solver_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let found = z3_candidates().iter().any(|candidate| {
            Command::new(candidate)
                .arg("-version")
                .output()
                .is_ok_and(|output| output.status.success())
        });
        verification::prerequisite_available(found, "Z3 for runtime loop-bound proofs")
    })
}

fn check(source: &str) -> Vec<String> {
    let program = y::parser::Parser::new(y::lexer::Lexer::new(source).tokenize())
        .parse_program()
        .expect("parse runtime loop-bound regression");
    check_program(&program)
}

fn check_program(program: &y::ast::Program) -> Vec<String> {
    assert!(
        std::env::var_os("Y_ALLOW_UNVERIFIED_INVARIANTS").is_none(),
        "runtime-bound regressions require checked invariants"
    );
    let mut checker = TypeChecker::new();
    checker.check_program(program);
    assert!(
        checker
            .errors
            .iter()
            .all(|error| !error.contains("SMT solver could not be run")),
        "a discovered solver must run: {:?}",
        checker.errors
    );
    checker.errors
}

fn accept(source: &str) {
    let errors = check(source);
    assert!(errors.is_empty(), "{source}\n{errors:?}");
}

fn reject(source: &str) {
    let errors = check(source);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("SMT Safety Verification Failed")
                || error.contains("Cannot verify invariant")),
        "expected a proof refusal, got {errors:?}\n{source}"
    );
}

/// A pinned scratch directory, which is also the compiler's working directory:
/// the repository's would make every `--emit-ptx` here compile for this
/// machine's card (`suite_is_machine_independent.rs`).
fn scratch(tag: &str) -> PathBuf {
    pinned::pinned_scratch(&format!("runtime_bounds_{tag}"), pinned::SM_PINNED)
}

fn compile_source(tag: &str, source: &str, option: &str) -> (PathBuf, String) {
    let dir = scratch(tag);
    let path = dir.join("input.ysu");
    std::fs::write(&path, source).expect("write source");
    let output = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg(option)
        .env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS")
        .env_remove("Y_NO_GEMM_RECOGNISER")
        .env_remove("Y_NO_CERTIFICATE")
        .current_dir(&dir)
        .output()
        .expect("compile source with the tested Y binary");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "{tag} source was refused:\n{log}");
    assert!(
        !log.contains("UNVERIFIED"),
        "{tag} bypassed its proof:\n{log}"
    );
    (dir, log)
}

#[test]
fn signed_runtime_endpoints_do_not_need_a_nonnegative_assumption() {
    if !solver_available() {
        return;
    }
    accept(
        r#"
fn zero_start(end: I32) {
    @invariant(i >= 0)
    for i in 0..end { }
}
fn negative_start(end: I32) {
    let begin: I32 = -7;
    @invariant(i >= -7)
    for i in begin..end { }
}
fn runtime_start(begin: I32, end: I32) {
    @invariant(i >= begin)
    for i in begin..end { }
}
"#,
    );
}

#[test]
fn wide_or_overflowing_endpoints_are_still_refused() {
    if !solver_available() {
        return;
    }
    for source in [
        "fn wide(end: I64) { @invariant(i >= 0) for i in 0..end { } }",
        "fn wide_start(begin: I64) { @invariant(i >= begin) for i in begin..7 { } }",
        "fn unsigned_threshold() { @invariant(i >= 0) for i in 0..2147483648 { } }",
        "fn narrowed() { @invariant(i >= 0) for i in 0..4294967297 { } }",
    ] {
        reject(source);
    }
    // The range parser currently accepts only primary bounds. Exercise the
    // checker's arithmetic obligation directly without changing that grammar.
    let source = "fn overflow(end: I32) { @invariant(i >= 0) for i in 0..end { } }";
    let mut program = y::parser::Parser::new(y::lexer::Lexer::new(source).tokenize())
        .parse_program()
        .unwrap();
    let y::ast::Item::Func(function) = &mut program.items[0] else {
        panic!("function fixture")
    };
    let y::ast::Stmt::For { end, .. } = &mut function.body.stmts[0] else {
        panic!("loop fixture")
    };
    let span = end.span();
    *end = y::ast::Expr::BinaryOp {
        left: Box::new(end.clone()),
        op: y::ast::BinaryOp::Add,
        right: Box::new(y::ast::Expr::IntLit(1, span.clone())),
        span,
    };
    let errors = check_program(&program);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("Loop condition arithmetic is not provably representable")),
        "an overflowing runtime endpoint must fail its arithmetic proof: {errors:?}"
    );
}

#[test]
fn unsafe_latches_and_nonpositive_steps_are_still_refused() {
    if !solver_available() {
        return;
    }
    for source in [
        // end may be I32::MAX and i may be I32::MAX - 1: adding two wraps.
        "fn wrap(end: I32) { @invariant(i >= 0) for i in 0..end step 2 { } }",
        "fn zero() { @invariant(i >= 0) for i in 0..7 step 0 { } }",
        "fn negative() { let stride: I32 = -1; @invariant(i >= 0) for i in 0..7 step stride { } }",
        "fn unknown(stride: I32) { @invariant(i >= 0) for i in 0..7 step stride { } }",
        "fn changed(end: I32) { @invariant(i >= 0) for i in 0..end { end = end - 1; } }",
    ] {
        reject(source);
    }
}

#[test]
fn generic_ptx_uses_the_signed_i32_loop_guard() {
    if !solver_available() {
        return;
    }
    let (dir, _) = compile_source(
        "ptx_signed",
        r#"
kernel signed_range(Out: GlobalMemory<I32>, begin: I32, end: I32) {
    let result: I32 = begin;
    @invariant(i >= begin)
    for i in begin..end { result = i; }
    block_ptr2d_store(Out, 0, 0, 1, 1, 1, result);
}
"#,
        "--emit-ptx",
    );
    let ptx = std::fs::read_to_string(dir.join("input.ptx")).expect("emitted PTX");
    assert_eq!(ptx.matches("setp.ge.s32").count(), 1, "{ptx}");
    assert!(!ptx.contains("setp.ge.u32"), "unsigned loop guard:\n{ptx}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn the_current_exact_pv_source_emits_checked_ptx() {
    if !solver_available() {
        return;
    }
    let (dir, _) = compile_source("exact_pv", include_str!("exact_pv.ysu"), "--emit-ptx");
    let ptx = std::fs::read_to_string(dir.join("input.ptx")).expect("emitted exact_pv PTX");
    assert!(ptx.contains(".entry exact_pv("), "{ptx}");
    assert!(ptx.contains("setp.ge.s32"), "{ptx}");
    assert!(
        ptx.contains("mul.lo.s64") && ptx.contains("add.s64"),
        "{ptx}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn canonical_exact_cpu_gemm_with_runtime_m_n_k_still_substitutes() {
    if !solver_available() {
        return;
    }
    let (dir, log) = compile_source(
        "exact_cpu",
        r#"
kernel y_matmul(A: GlobalMemory<I16>, B: GlobalMemory<I16>, C: GlobalMemory<I64>, M: I32, N: I32, K: I32) {
    @invariant(i >= 0)
    for i in 0..M step 1 {
        @invariant(j >= 0)
        for j in 0..N step 1 {
            @ZeroDrift
            let sum: I64 = 0;
            @invariant(k >= 0)
            for k in 0..K step 1 {
                @bounds(min=-1024, max=1024)
                let a: I64 = block_ptr2d_load(A, i, k, K, M, K);
                @bounds(min=-1024, max=1024)
                let b: I64 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a * b;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}
fn main() { }
"#,
        "--emit-llvm",
    );
    let ir = std::fs::read_to_string(dir.join("input.ll")).expect("emitted exact GEMM LLVM");
    assert!(
        ir.contains("__y_gemm_exact_vnni"),
        "substitution stopped:\n{log}"
    );
    assert!(
        dir.join("input_certificate.v").exists(),
        "missing certificate:\n{log}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn signed_runtime_loops_execute_like_an_independent_reference() {
    if !solver_available() {
        return;
    }
    let have_clang = Command::new("clang")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    if !verification::prerequisite_available(have_clang, "clang to execute signed runtime loops") {
        return;
    }
    let source = r#"
fn last(begin: I32, end: I32) -> I32 {
    let result: I32 = begin;
    @invariant(i >= begin)
    for i in begin..end { result = i; }
    return result;
}
"#;
    accept(source);
    let (dir, _) = compile_source("native_signed", source, "--emit-llvm");
    let ir = std::fs::read_to_string(dir.join("input.ll")).expect("emitted LLVM");
    assert!(ir.contains("icmp slt i32"), "{ir}");
    let harness = dir.join("reference.c");
    std::fs::write(
        &harness,
        r#"
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
extern int32_t last(int32_t begin, int32_t end);
static int32_t reference(int32_t begin, int32_t end) {
    int32_t result = begin;
    for (int64_t i = begin; i < (int64_t)end; ++i) result = (int32_t)i;
    return result;
}
int main(void) {
    static const int32_t endpoints[][2] = {
        {0, -7}, {0, INT32_MIN}, {0, 0}, {0, 7}, {-7, -3},
        {-7, 0}, {-7, 3}, {4, -3}, {4, 4}, {4, 7},
        {INT32_MIN, INT32_MIN + 3}, {INT32_MAX - 3, INT32_MAX},
        {INT32_MAX, INT32_MIN}, {INT32_MAX, INT32_MAX}
    };
    for (unsigned n = 0; n < sizeof(endpoints) / sizeof(endpoints[0]); ++n) {
        int32_t begin = endpoints[n][0], end = endpoints[n][1];
        int32_t got = last(begin, end), want = reference(begin, end);
        if (got != want) {
            fprintf(stderr, "%d..%d: got %d, expected %d\n", begin, end, got, want);
            return 1;
        }
    }
    return 0;
}
"#,
    )
    .unwrap();
    for optimization in ["-O0", "-O3"] {
        let exe = dir.join(format!("run{}", &optimization[2..]));
        let cc = Command::new("clang")
            .args([optimization, "-Wno-override-module"])
            .arg(dir.join("input.ll"))
            .arg(&harness)
            .arg("-o")
            .arg(&exe)
            .output()
            .expect("compile independent reference");
        assert!(
            cc.status.success(),
            "clang: {}",
            String::from_utf8_lossy(&cc.stderr)
        );
        let run = Command::new(&exe)
            .output()
            .expect("run signed loop comparison");
        assert!(
            run.status.success(),
            "{optimization}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn proved_i64_headers_are_converted_before_execution_and_assembly() {
    if !solver_available() {
        return;
    }
    for tool in ["clang", "ptxas"] {
        let available = Command::new(tool)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success());
        if !verification::prerequisite_available(
            available,
            &format!("{tool} for proved-I64 loop-header lowering"),
        ) {
            return;
        }
    }
    // Bounds are explicit source preconditions. Every value the independent
    // harness supplies satisfies them, including negative start/end values.
    let body = r#"
    @bounds(min=-7, max=7) let begin: I64 = begin_arg;
    @bounds(min=-7, max=7) let end: I64 = end_arg;
    let stride: I64 = 2;
    let result: I32 = begin;
    @invariant(i >= begin)
    for i in begin..end step stride { result = i; }
"#;
    let native_source =
        format!("fn bounded(begin_arg: I64, end_arg: I64) -> I32 {{\n{body}\nreturn result;\n}}");
    accept(&native_source);
    let (native_dir, _) = compile_source("i64_native", &native_source, "--emit-llvm");
    let ir = std::fs::read_to_string(native_dir.join("input.ll")).unwrap();
    assert!(ir.matches("trunc i64").count() >= 3, "{ir}");
    let harness = native_dir.join("reference.c");
    std::fs::write(
        &harness,
        r#"
#include <stdint.h>
#include <stdio.h>
extern int32_t bounded(int64_t begin, int64_t end);
int main(void) {
    static const int64_t endpoints[] = {-7, -3, 0, 3, 7};
    for (unsigned a = 0; a < 5; ++a) for (unsigned b = 0; b < 5; ++b) {
        int64_t begin = endpoints[a], end = endpoints[b];
        int32_t want = (int32_t)begin;
        for (int64_t i = begin; i < end; i += 2) want = (int32_t)i;
        int32_t got = bounded(begin, end);
        if (got != want) {
            fprintf(stderr, "%lld..%lld: got %d, expected %d\n",
                    (long long)begin, (long long)end, got, want);
            return 1;
        }
    }
    return 0;
}
"#,
    )
    .unwrap();
    let exe = native_dir.join("run");
    let cc = Command::new("clang")
        .args(["-O2", "-Wno-override-module"])
        .arg(native_dir.join("input.ll"))
        .arg(&harness)
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        cc.status.success(),
        "{}",
        String::from_utf8_lossy(&cc.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let gpu_source = format!(
        "kernel bounded_gpu(Out: GlobalMemory<I32>, begin_arg: I64, end_arg: I64) {{\n\
         {body}\nblock_ptr2d_store(Out, 0, 0, 1, 1, 1, result);\n}}"
    );
    let (ptx_dir, _) = compile_source("i64_ptx", &gpu_source, "--emit-ptx");
    let ptx = std::fs::read_to_string(ptx_dir.join("input.ptx")).unwrap();
    assert!(ptx.matches("cvt.u32.u64").count() >= 3, "{ptx}");
    assert!(ptx.contains("setp.ge.s32"), "{ptx}");
    let assembly = Command::new("ptxas")
        .args(["-O1", "-arch=sm_89"])
        .arg(ptx_dir.join("input.ptx"))
        .arg("-o")
        .arg(ptx_dir.join("input.cubin"))
        .output()
        .unwrap();
    assert!(
        assembly.status.success(),
        "{}",
        String::from_utf8_lossy(&assembly.stderr)
    );
    std::fs::remove_dir_all(native_dir).unwrap();
    std::fs::remove_dir_all(ptx_dir).unwrap();
}

#[test]
fn substituted_float_gemm_preserves_empty_signed_source_loops() {
    if !solver_available() {
        return;
    }
    let have_clang = Command::new("clang")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    if !verification::prerequisite_available(have_clang, "clang for signed GEMM source execution") {
        return;
    }
    let (dir, _) = compile_source(
        "float_signed_dims",
        r#"
kernel signed_gemm(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, M: I32, N: I32, K: I32) {
    @invariant(i >= 0)
    for i in 0..M {
        @invariant(j >= 0)
        for j in 0..N {
            let sum: F32 = 0.0;
            @invariant(k >= 0)
            for k in 0..K {
                let a: F32 = block_ptr2d_load(A, i, k, K, M, K);
                let b: F32 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a * b;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}
"#,
        "--emit-llvm",
    );
    let ir = std::fs::read_to_string(dir.join("input.ll")).unwrap();
    assert!(ir.contains("call void @__y_sgemm_f32_avx512"), "{ir}");
    let harness = dir.join("reference.c");
    std::fs::write(
        &harness,
        r#"
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
extern void signed_gemm(const float *, const float *, float *, int, int, int);
int main(int argc, char **argv) {
    static const int dimensions[][3] = {
        {0, 3, 2}, {-3, 3, 2}, {3, 0, 2}, {3, -3, 2}, {-3, -3, -2},
        {INT_MIN, INT_MAX, INT_MAX}, {INT_MAX, INT_MIN, INT_MAX},
        {3, 3, 0}, {3, 3, -1}, {3, 3, INT_MIN}
    };
    for (unsigned n = 0; n < sizeof(dimensions) / sizeof(dimensions[0]); ++n) {
        int M = dimensions[n][0], N = dimensions[n][1], K = dimensions[n][2];
        float got[32], want[32];
        for (unsigned p = 0; p < 32; ++p) got[p] = want[p] = 19.0f;
        if (M > 0 && N > 0)
            for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) want[i*N+j] = 0.0f;
        signed_gemm(NULL, NULL, got, M, N, K);
        if (memcmp(got, want, sizeof(got))) {
            fprintf(stderr, "empty %d x %d x %d did not match the source\n", M, N, K);
            return 1;
        }
        if (M <= 0 || N <= 0) signed_gemm(NULL, NULL, NULL, M, N, K);
    }
    if (argc > 1) {
        const float A[] = {-1, 2, 3, -4}, B[] = {1, -2, 3, 4, -5, 6};
        float got[6], want[6];
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 2; ++k) sum += A[i*2+k] * B[k*3+j];
            want[i*3+j] = sum;
            got[i*3+j] = 19.0f;
        }
        signed_gemm(A, B, got, 2, 3, 2);
        if (memcmp(got, want, sizeof(got))) return 1;
    }
    return 0;
}
"#,
    )
    .unwrap();
    let exe = dir.join("run");
    let cc = Command::new("clang")
        .args(["-O2", "-Wno-override-module"])
        .arg(dir.join("input.ll"))
        .arg(&harness)
        .args(["-lm", "-lpthread", "-o"])
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        cc.status.success(),
        "{}",
        String::from_utf8_lossy(&cc.stderr)
    );
    let mut run = Command::new(&exe);
    run.env("Y_NUM_THREADS", "1");
    if y::sentinel::host_has_avx512() {
        run.arg("positive-control");
    }
    let output = run.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn overlapping_gemm_buffers_execute_the_original_sequential_source() {
    if !solver_available() {
        return;
    }
    let have_clang = Command::new("clang")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    if !verification::prerequisite_available(have_clang, "clang for aliasing GEMM source execution")
    {
        return;
    }
    let mut source = String::new();
    for (name, input, output, operand, drift, bounds) in [
        ("alias_f32", "F32", "F32", "F32", "", ""),
        (
            "alias_i16",
            "I16",
            "I64",
            "I64",
            "@ZeroDrift",
            "@bounds(min=-1024, max=1024)",
        ),
    ] {
        source.push_str(&format!(r#"
kernel {name}(A: GlobalMemory<{input}>, B: GlobalMemory<{input}>, C: GlobalMemory<{output}>, M: I32, N: I32, K: I32) {{
    @invariant(i >= 0)
    for i in 0..M {{
        @invariant(j >= 0)
        for j in 0..N {{
            {drift}
            let sum: {operand} = 0;
            @invariant(k >= 0)
            for k in 0..K {{
                {bounds}
                let a: {operand} = block_ptr2d_load(A, i, k, K, M, K);
                {bounds}
                let b: {operand} = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a * b;
            }}
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }}
    }}
}}
"#));
    }
    let (dir, _) = compile_source("aliases", &source, "--emit-llvm");
    let ir = std::fs::read_to_string(dir.join("input.ll")).unwrap();
    assert!(ir.contains("call void @__y_sgemm_f32_avx512"), "{ir}");
    if y::sentinel::host_has_avx512_vnni() {
        assert!(
            ir.contains("call void @__y_gemm_exact_vnni_threaded"),
            "{ir}"
        );
    }
    // The checked range calls are executable dispatch instructions, not merely
    // declarations of intrinsics beside an unconditional fast call.
    assert!(
        ir.contains("call { i64, i1 } @llvm.umul.with.overflow.i64"),
        "{ir}"
    );
    assert!(
        ir.contains("call { i64, i1 } @llvm.uadd.with.overflow.i64"),
        "{ir}"
    );
    let harness = dir.join("reference.c");
    std::fs::write(
        &harness,
        r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void alias_f32(const float *, const float *, float *, int, int, int);
extern void alias_i16(const int16_t *, const int16_t *, int64_t *, int, int, int);
static float read_f32(const unsigned char *p) { float v; memcpy(&v, p, 4); return v; }
static int16_t read_i16(const unsigned char *p) { int16_t v; memcpy(&v, p, 2); return v; }
static void reference(unsigned char *bytes, int ao, int bo, int co, int exact) {
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        if (exact) {
            int64_t sum = 0;
            for (int k = 0; k < 2; ++k)
                sum += (int64_t)read_i16(bytes+ao+2*(i*2+k)) * read_i16(bytes+bo+2*(k*2+j));
            memcpy(bytes+co+8*(i*2+j), &sum, 8);
        } else {
            float sum = 0;
            for (int k = 0; k < 2; ++k)
                sum += read_f32(bytes+ao+4*(i*2+k)) * read_f32(bytes+bo+4*(k*2+j));
            memcpy(bytes+co+4*(i*2+j), &sum, 4);
        }
    }
}
int main(int argc, char **argv) {
    // C aliases A, C aliases B, read-only A/B overlap, and disjoint control.
    static const int offsets[][3] = {{16,64,16}, {16,64,64}, {16,16,112}, {16,64,112}};
    for (int exact = 0; exact < 2; ++exact) for (unsigned test = 0; test < 4; ++test) {
        // Disjoint F32 can execute the fast AVX512 helper; omit that single
        // control on unsupported hosts. Overlaps always take scalar lowering.
        if (!exact && test == 3 && argc == 1) continue;
        _Alignas(64) unsigned char got[160], want[160];
        memset(got, 0x5a, sizeof(got));
        int ao = offsets[test][0], bo = offsets[test][1], co = offsets[test][2];
        for (int p = 0; p < 4; ++p) {
            if (exact) {
                int16_t a = (int16_t)(p+1), b = (int16_t)(4-p);
                memcpy(got+ao+2*p, &a, 2); memcpy(got+bo+2*p, &b, 2);
            } else {
                float a = (float)(p+1), b = (float)(4-p);
                memcpy(got+ao+4*p, &a, 4); memcpy(got+bo+4*p, &b, 4);
            }
        }
        memcpy(want, got, sizeof(got));
        reference(want, ao, bo, co, exact);
        if (exact)
            alias_i16((const int16_t *)(got+ao), (const int16_t *)(got+bo), (int64_t *)(got+co), 2, 2, 2);
        else
            alias_f32((const float *)(got+ao), (const float *)(got+bo), (float *)(got+co), 2, 2, 2);
        // Includes input bytes and untouched canaries, not just output values.
        if (memcmp(got, want, sizeof(got))) {
            fprintf(stderr, "alias case %u exact=%d disagrees with sequential source\n", test, exact);
            return 1;
        }
    }
    return 0;
}
"#,
    )
    .unwrap();
    let exe = dir.join("run");
    let cc = Command::new("clang")
        .args(["-O2", "-Wno-override-module"])
        .arg(dir.join("input.ll"))
        .arg(&harness)
        .args(["-lm", "-lpthread", "-o"])
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        cc.status.success(),
        "{}",
        String::from_utf8_lossy(&cc.stderr)
    );
    let mut run = Command::new(&exe);
    run.env("Y_NUM_THREADS", "1");
    if y::sentinel::host_has_avx512() {
        run.arg("positive-control");
    }
    let output = run.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(dir).unwrap();
}
