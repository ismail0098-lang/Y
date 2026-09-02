//! The exact threaded GEMM under ThreadSanitizer - and the reason a TSan run
//! over an emitted module reports nothing at all unless you make it.
//!
//! ** The item this is for.
//!
//! `exact_gemm_certificate::TRUST_BOUNDARY` carried, as `Check::Unchecked`,
//! the claim that a worker's stores are visible to the reduction that reads
//! and then frees its buffers. `ExactGemmThreading.v` proves WHICH threads are
//! joined; the ORDER a join imposes is not arithmetic and no proof in this
//! repository speaks about it. TSan does: it reasons about happens-before
//! rather than about observed interleavings, so one execution is enough to
//! find a missing edge.
//!
//! It is a DYNAMIC check and this file says so rather than implying more. It
//! covers the schedules these runs explore, at these shapes; it is not a
//! statement about every schedule, which would need a memory model.
//!
//! ** THE TRAP, MEASURED.
//!
//! `clang -fsanitize=thread` over a `.ll` file instruments almost nothing.
//! TSan's memory-access instrumentation is gated on the `sanitize_thread`
//! FUNCTION attribute, which the C frontend adds and which a hand-written
//! module does not carry. What you get instead is `__tsan_func_entry` and
//! `__tsan_func_exit` - the call-stack bookkeeping - plus the malloc/free and
//! pthread INTERCEPTORS, which live in the runtime and fire regardless.
//!
//! Measured on the exact kernel: without the attribute
//! `__y_gemm_exact_vnni_threaded` carries **zero** `__tsan_read`/`__tsan_write`
//! checks, and with it 33. What it does carry either way is
//! `__tsan_func_entry`/`_exit` and, since the fix below, `__tsan_atomic64_*` -
//! atomics go through a different lowering. So an ordinary race on a global is
//! completely invisible, while a use-after-free is still caught by the
//! malloc/free interceptor - which is exactly the combination that makes the
//! silence look like coverage.
//!
//! That is `feedback-null-metrics-pass-dead-components` wearing a sanitizer:
//! a tool that reports zero findings perfectly when it is not looking. The
//! rewrite below is what makes the silence mean something, and
//! [the_attribute_rewrite_is_what_instruments_the_kernel] is what stops the
//! rewrite silently ceasing to work.
//!
//! ** What it found.
//!
//! `@__y_gemm_exact_nthreads`, the memoised thread count, was a plain
//! load/store on a mutable global. Two application threads calling the exact
//! GEMM at once both find it unset and both write it - measured at 4 to 8 of 8
//! concurrent callers entering that path on every run. Every writer stores the
//! same value and on x86 an aligned `i64` cannot tear, so it never produced a
//! wrong answer; it is undefined behaviour all the same, in a kernel that
//! ships a certificate claiming exactness. It is a relaxed atomic now, which
//! costs a plain `mov`.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// TSan needs the runtime as well as the flag, and neither is guaranteed.
/// Compile and RUN a trivial program rather than asking `clang --version`.
fn tsan_available(dir: &PathBuf) -> bool {
    let c = dir.join("probe.c");
    if std::fs::write(&c, "int main(void){return 0;}\n").is_err() {
        return false;
    }
    let exe = dir.join("probe");
    let ok = Command::new("clang")
        .args([
            c.to_str().unwrap(),
            "-fsanitize=thread",
            "-o",
            exe.to_str().unwrap(),
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    ok && Command::new(&exe)
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

/// One caller, ragged in all three axes against the 6x64 tile.
const DRIVER_ONE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
void y_matmul(const int16_t *A, const int16_t *B, int64_t *C, int M, int N, int K);
#define MM 53
#define NN 71
#define KK 4099
static int16_t A[MM * KK];
static int16_t B[KK * NN];
static int64_t C[MM * NN];
static int16_t gen(long i, long salt) {
    unsigned long h = i * 2654435761UL + salt * 40503UL; h ^= h >> 13;
    return (int16_t)((long)(h % 2049) - 1024);
}
int main(void) {
    for (long i = 0; i < MM * KK; i++) A[i] = gen(i, 1);
    for (long i = 0; i < KK * NN; i++) B[i] = gen(i, 2);
    y_matmul(A, B, C, MM, NN, KK);
    long long sum = 0;
    for (long i = 0; i < MM * NN; i++) sum += C[i];
    printf("checksum %lld\n", sum);
    return 0;
}
"#;

/// Eight application threads entering the exact GEMM together, released by a
/// barrier so they contend for the memoised thread count.
///
/// `Y_TS_CANARY` adds a deliberate unsynchronised global write. Without it
/// this whole file would pass against a build where TSan is present, linked,
/// and looking at nothing - which is precisely the state a `.ll` compiled with
/// `-fsanitize=thread` is in.
const DRIVER_MANY: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
void y_matmul(const int16_t *A, const int16_t *B, int64_t *C, int M, int N, int K);
#define MM 1
#define NN 64
#define KK 4096
#define NC 8
static int16_t A[MM * KK];
static int16_t B[KK * NN];
static int64_t C[NC][MM * NN];
static pthread_barrier_t bar;
long canary = 0;
static int16_t gen(long i, long salt) {
    unsigned long h = i * 2654435761UL + salt * 40503UL; h ^= h >> 13;
    return (int16_t)((long)(h % 2049) - 1024);
}
static void *call(void *p) {
    long id = (long)p;
    pthread_barrier_wait(&bar);
    if (getenv("Y_TS_CANARY")) canary = id + 1;
    y_matmul(A, B, C[id], MM, NN, KK);
    return 0;
}
int main(void) {
    for (long i = 0; i < MM * KK; i++) A[i] = gen(i, 1);
    for (long i = 0; i < KK * NN; i++) B[i] = gen(i, 2);
    pthread_barrier_init(&bar, 0, NC);
    pthread_t t[NC];
    for (long i = 0; i < NC; i++) pthread_create(&t[i], 0, call, (void *)i);
    for (long i = 0; i < NC; i++) pthread_join(t[i], 0);
    long long s = 0; int same = 1;
    for (long i = 0; i < MM * NN; i++) s += C[0][i];
    for (int c = 1; c < NC; c++) {
        long long x = 0;
        for (long i = 0; i < MM * NN; i++) x += C[c][i];
        if (x != s) same = 0;
    }
    printf("checksum %lld same %d\n", s, same);
    return 0;
}
"#;

/// The tag is in the signature rather than in a comment asking the caller to
/// remember: two tests sharing a temp-dir name is a race this repository has
/// hit five times.
fn emit(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("y_exact_tsan_{tag}_{}", std::process::id()));
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
        "the nest must route through the exact threaded entry"
    );
    (dir, ll)
}

/// Give every `define` in the module the `sanitize_thread` attribute.
///
/// `#77` rather than a name because an attribute-group reference must be
/// numeric, and rather than a low number because this module already defines
/// `#0` twice - once per emitted GEMM - and reusing an id would silently
/// rewrite one of them.
fn instrument(ll: &PathBuf) -> PathBuf {
    let text = std::fs::read_to_string(ll).expect("emitted IR");
    let mut out = String::with_capacity(text.len() + 4096);
    let mut touched = 0usize;
    for line in text.lines() {
        if line.starts_with("define ") && line.ends_with(" {") && !line.contains(" #77 ") {
            out.push_str(&line[..line.len() - 2]);
            out.push_str(" #77 {\n");
            touched += 1;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str("attributes #77 = { sanitize_thread }\n");
    assert!(
        touched > 5,
        "the attribute rewrite matched {touched} function definitions, so the \
         module's shape changed and every assertion below would be vacuous"
    );
    let path = ll.with_file_name("m_tsan.ll");
    std::fs::write(&path, out).expect("write instrumented IR");
    path
}

/// `__tsan_read*` / `__tsan_write*` inside one function of a post-optimisation
/// module - the checks on ORDINARY loads and stores, which are exactly the
/// class the attribute gates.
///
/// Three other families are deliberately not counted, and each would make this
/// check pass against a kernel TSan is not watching. `__tsan_func_entry` and
/// `__tsan_func_exit` are call-stack bookkeeping and appear either way.
/// `__tsan_atomic*` appears either way too - atomics are lowered by a
/// different mechanism, which is why the memoised thread count's own accesses
/// show up in the un-rewritten module and its plain neighbours do not.
fn memory_instrumentation(module: &str, func: &str) -> usize {
    // Find the DEFINITION, not the first mention: a call site of the same
    // name appears earlier, and walking back from it lands in whichever
    // function happens to make the call.
    let needle = format!("@{func}(");
    let mut body: Option<&str> = None;
    for (i, _) in module.match_indices("\ndefine ") {
        let header_end = module[i + 1..]
            .find('\n')
            .map(|e| i + 1 + e)
            .unwrap_or(module.len());
        if !module[i..header_end].contains(&needle) {
            continue;
        }
        let rest = &module[i..];
        let end = rest.find("\n}").map(|e| e + 2).unwrap_or(rest.len());
        body = Some(&rest[..end]);
        break;
    }
    let Some(body) = body else { return 0 };
    body.matches("@__tsan_read").count()
        + body.matches("@__tsan_write").count()
        + body.matches("@__tsan_unaligned").count()
}

fn optimised_ir(ll: &PathBuf, tag: &str) -> String {
    let out = ll.with_file_name(format!("opt_{tag}.ll"));
    let cc = Command::new("clang")
        .args([
            "-S",
            "-emit-llvm",
            "-fsanitize=thread",
            "-O1",
            ll.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run clang");
    assert!(
        cc.status.success(),
        "clang failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );
    std::fs::read_to_string(&out).expect("optimised IR")
}

fn build(dir: &PathBuf, ll: &PathBuf, driver: &str, name: &str) -> PathBuf {
    let c = dir.join(format!("{name}.c"));
    std::fs::write(&c, driver).expect("write driver");
    let exe = dir.join(name);
    let cc = Command::new("clang")
        .args([
            ll.to_str().unwrap(),
            c.to_str().unwrap(),
            "-O1",
            "-g",
            "-fsanitize=thread",
            "-lm",
            "-lpthread",
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
    exe
}

struct Report {
    warnings: usize,
    text: String,
    ok: bool,
}

fn run(exe: &PathBuf, threads: &str, canary: bool) -> Report {
    let mut cmd = Command::new(exe);
    cmd.env("Y_NUM_THREADS", threads);
    if canary {
        cmd.env("Y_TS_CANARY", "1");
    }
    let out = cmd.output().expect("run the kernel");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    Report {
        warnings: text.matches("WARNING: ThreadSanitizer").count(),
        ok: out.status.success(),
        text,
    }
}

/// The attribute rewrite is load-bearing, and this is the measurement that
/// says so.
///
/// If a future clang starts instrumenting a `.ll` under the flag alone, this
/// fails - and that is a useful failure: it says the rewrite can go.
#[test]
fn the_attribute_rewrite_is_what_instruments_the_kernel() {
    let (dir, ll) = emit("attr");
    if !tsan_available(&dir) {
        eprintln!("skipping: no working ThreadSanitizer");
        let _ = std::fs::remove_dir_all(dir);
        return;
    }
    let plain = optimised_ir(&ll, "plain");
    let rewritten = optimised_ir(&instrument(&ll), "rewritten");

    for f in [
        "__y_gemm_exact_vnni_threaded",
        "__y_gemm_exact_worker",
        "__y_gemm_micro_vnni",
    ] {
        let before = memory_instrumentation(&plain, f);
        let after = memory_instrumentation(&rewritten, f);
        assert_eq!(
            before, 0,
            "`{f}` already has {before} memory-access checks under the flag \
             alone; the rewrite may no longer be needed"
        );
        assert!(
            after > 0,
            "`{f}` has no memory-access instrumentation even after the \
             rewrite, so every race assertion in this file is vacuous"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// The runtime canary. Without it a build where TSan is linked and looking at
/// nothing passes every assertion below.
#[test]
fn the_sanitizer_reports_a_race_when_there_is_one() {
    let (dir, ll) = emit("canary");
    if !tsan_available(&dir) {
        eprintln!("skipping: no working ThreadSanitizer");
        let _ = std::fs::remove_dir_all(dir);
        return;
    }
    let exe = build(&dir, &instrument(&ll), DRIVER_MANY, "many");

    let clean = run(&exe, "4", false);
    assert_eq!(
        clean.warnings, 0,
        "the unperturbed run reported a race:\n{}",
        clean.text
    );
    let canary = run(&exe, "4", true);
    assert!(
        canary.warnings > 0,
        "ThreadSanitizer did not report a deliberate unsynchronised global \
         write, so its silence everywhere else means nothing:\n{}",
        canary.text
    );
    assert!(
        canary.text.contains("data race"),
        "the canary's report is not a data race:\n{}",
        canary.text
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// One caller, several thread counts: the reduction must not read or free a
/// worker's buffers without the join that orders them.
#[test]
fn the_threaded_kernel_is_race_free_at_every_thread_count() {
    let (dir, ll) = emit("counts");
    if !tsan_available(&dir) {
        eprintln!("skipping: no working ThreadSanitizer");
        let _ = std::fs::remove_dir_all(dir);
        return;
    }
    let exe = build(&dir, &instrument(&ll), DRIVER_ONE, "one");

    for threads in ["1", "2", "4", "8"] {
        let r = run(&exe, threads, false);
        assert!(
            r.ok,
            "the kernel did not exit cleanly at {threads} threads:\n{}",
            r.text
        );
        assert_eq!(
            r.warnings, 0,
            "ThreadSanitizer reported {} finding(s) at {threads} threads:\n{}",
            r.warnings, r.text
        );
        assert!(
            r.text.contains("checksum 1079268997"),
            "the instrumented kernel disagrees with the reference at \
             {threads} threads: {}",
            r.text
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// Eight application threads entering the exact GEMM at once.
///
/// This is the shape that found the memoised thread count's race, and the
/// answer alone could not have: every writer stored the same value.
#[test]
fn concurrent_callers_do_not_race_on_the_memoised_thread_count() {
    let (dir, ll) = emit("concurrent");
    if !tsan_available(&dir) {
        eprintln!("skipping: no working ThreadSanitizer");
        let _ = std::fs::remove_dir_all(dir);
        return;
    }
    let exe = build(&dir, &instrument(&ll), DRIVER_MANY, "many");

    for threads in ["2", "4"] {
        let r = run(&exe, threads, false);
        assert!(r.ok, "the run did not exit cleanly:\n{}", r.text);
        assert_eq!(
            r.warnings, 0,
            "ThreadSanitizer reported {} finding(s) with eight concurrent \
             callers at {threads} threads each:\n{}",
            r.warnings, r.text
        );
        assert!(
            r.text.contains("same 1"),
            "the eight callers disagreed with each other: {}",
            r.text
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// The memo is a relaxed atomic, not a plain global.
///
/// A structural companion to the runs above: they are dynamic and cover the
/// schedules they happen to explore, while this cannot be got past by timing.
#[test]
fn the_memoised_thread_count_is_accessed_atomically() {
    let (dir, ll) = emit("atomic");
    let ir = std::fs::read_to_string(&ll).expect("emitted IR");

    assert!(
        ir.contains("load atomic i64, ptr @__y_gemm_exact_nthreads monotonic"),
        "the memoised thread count is read non-atomically again"
    );
    assert!(
        ir.contains("store atomic i64 %r2, ptr @__y_gemm_exact_nthreads monotonic"),
        "the memoised thread count is written non-atomically again"
    );
    // The control: `monotonic` must not have been reached by making every
    // access to this global atomic and calling it done, nor by widening the
    // change beyond the two lines it is. Counting atomics module-wide would
    // be the wrong assertion - the f32 kernel's thread pool has its own.
    let touching = ir
        .lines()
        .filter(|l| l.contains("@__y_gemm_exact_nthreads") && !l.starts_with('@'))
        .count();
    assert_eq!(
        touching, 2,
        "the memoised thread count is accessed at {touching} sites; the fix \
         covers a load and a store"
    );
    let _ = std::fs::remove_dir_all(dir);
}
