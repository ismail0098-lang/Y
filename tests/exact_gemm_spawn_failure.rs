//! The two spawn paths nothing in this repository had ever executed: a
//! `pthread_create` that fails, and a thread whose `pthread_t` is zero.
//!
//! ** What was wrong, and it was not a wrong number.
//!
//! The join loop decided whether a worker had started by testing its recorded
//! thread id against zero. `pthread_t` is opaque and POSIX says nothing about
//! its representation; glibc's happens to be a pointer to the thread
//! descriptor, so zero never occurs there and the sentinel never misfired.
//! On a library where a live thread's id can be zero, the join is SKIPPED -
//! and the reduction then reads that worker's private C buffer and `free`s
//! the four buffers it is still writing into.
//!
//! Measured with `-Wl,--wrap=pthread_create` handing out a zero id for a
//! thread that really runs, at `M=53 N=71 K=4099` on four threads: the join
//! count fell 4 -> 3 -> 2 as more workers were aliased, and with all four
//! aliased the process **segfaulted on six runs out of six**.
//!
//! ** Why the assertion is the JOIN COUNT and not the answer.
//!
//! At one or two aliased workers the answer came back right every time - the
//! race did not fire. A race that does not fire is not a test, and this
//! repository has the same lesson recorded about a missing `bar.sync`. The
//! join count is deterministic: it must equal the number of `pthread_create`
//! calls that returned success, whatever ids the library handed out.
//!
//! ** The other path: `pthread_create` failing.
//!
//! The wrapper runs that band inline instead. That arm ships, and until this
//! file existed nothing had run it - `pthread_create` does not fail on a test
//! machine. It is correct: bit-identical answers at every failure mask, which
//! is a confirmation rather than a finding and is worth having either way.

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

/// Ragged in all three axes against the 6x64 tile, and K large enough that a
/// four-way split gives every worker real work to be racing over.
const M: usize = 53;
const N: usize = 71;
const K: usize = 4099;

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

/// `argv[1]` is a bit mask of `pthread_create` calls that must FAIL;
/// `argv[2]` a mask of calls that succeed but report a zero id.
///
/// The zero-id arm aliases the real id so that a join of 0 still reaches the
/// right thread - deliberately, because the question is whether the kernel
/// JOINS, not whether joining zero happens to work. Anything the kernel does
/// not join is joined here before the checksum, so the process never leaves a
/// worker running past `main`.
const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <errno.h>

void y_matmul(const int16_t *A, const int16_t *B, int64_t *C, int M, int N, int K);

static long creates = 0, failed = 0, zeroid = 0, joins = 0;
static long fail_mask = 0, zero_mask = 0;
static pthread_t alias[64];
static int alias_head = 0, alias_tail = 0;

int __real_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int __real_pthread_join(pthread_t, void **);

int __wrap_pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*f)(void *), void *arg) {
    long idx = creates++;
    if (idx < 62 && ((fail_mask >> idx) & 1L)) { failed++; return EAGAIN; }
    if (idx < 62 && ((zero_mask >> idx) & 1L)) {
        pthread_t real;
        int rc = __real_pthread_create(&real, a, f, arg);
        if (rc) return rc;
        alias[alias_tail++] = real;
        zeroid++;
        *t = 0;
        return 0;
    }
    return __real_pthread_create(t, a, f, arg);
}

int __wrap_pthread_join(pthread_t t, void **r) {
    joins++;
    if (t == 0 && alias_head < alias_tail) return __real_pthread_join(alias[alias_head++], r);
    return __real_pthread_join(t, r);
}

#define MM 53
#define NN 71
#define KK 4099
static int16_t A[MM * KK];
static int16_t B[KK * NN];
static int64_t C[MM * NN];

static int16_t gen(long i, long salt) {
    unsigned long h = i * 2654435761UL + salt * 40503UL;
    h ^= h >> 13;
    return (int16_t)((long)(h % 2049) - 1024);
}

int main(int argc, char **argv) {
    fail_mask = (argc > 1) ? atol(argv[1]) : 0;
    zero_mask = (argc > 2) ? atol(argv[2]) : 0;
    for (long i = 0; i < MM * KK; i++) A[i] = gen(i, 1);
    for (long i = 0; i < KK * NN; i++) B[i] = gen(i, 2);

    y_matmul(A, B, C, MM, NN, KK);

    while (alias_head < alias_tail) __real_pthread_join(alias[alias_head++], 0);

    long long sum = 0;
    for (long i = 0; i < MM * NN; i++) sum += C[i];
    printf("checksum %lld creates %ld failed %ld zeroid %ld joins %ld\n",
           sum, creates, failed, zeroid, joins);
    return 0;
}
"#;

/// The tag is in the signature rather than in a comment asking the caller to
/// remember: two tests sharing a temp-dir name is a race this repository has
/// hit five times.
fn build(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("y_exact_spawn_{tag}_{}", std::process::id()));
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
        "the nest must route through the exact threaded entry, or this file is \
         testing a kernel that is not the subject"
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
            "-Wl,--wrap=pthread_join",
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

fn gen(i: i64, salt: i64) -> i64 {
    let h = (i as u64)
        .wrapping_mul(2654435761)
        .wrapping_add((salt as u64).wrapping_mul(40503));
    let h = h ^ (h >> 13);
    (h % 2049) as i64 - 1024
}

fn reference_checksum() -> i64 {
    let a: Vec<i64> = (0..(M * K) as i64).map(|i| gen(i, 1)).collect();
    let b: Vec<i64> = (0..(K * N) as i64).map(|i| gen(i, 2)).collect();
    let mut total = 0i64;
    for i in 0..M {
        for j in 0..N {
            let mut s = 0i64;
            for k in 0..K {
                s += a[i * K + k] * b[k * N + j];
            }
            total += s;
        }
    }
    total
}

struct Run {
    code: Option<i32>,
    checksum: Option<i64>,
    creates: i64,
    failed: i64,
    zeroid: i64,
    joins: i64,
}

fn field(text: &str, name: &str) -> i64 {
    text.split(&format!("{name} "))
        .nth(1)
        .and_then(|t| t.split_whitespace().next())
        .and_then(|t| t.parse().ok())
        .unwrap_or(-1)
}

fn run(exe: &PathBuf, fail_mask: i64, zero_mask: i64, threads: &str) -> Run {
    let out = Command::new(exe)
        .arg(fail_mask.to_string())
        .arg(zero_mask.to_string())
        .env("Y_NUM_THREADS", threads)
        .output()
        .expect("run the kernel");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    Run {
        code: out.status.code(),
        checksum: text
            .split("checksum ")
            .nth(1)
            .and_then(|t| t.split_whitespace().next())
            .and_then(|t| t.parse().ok()),
        creates: field(&text, "creates"),
        failed: field(&text, "failed"),
        zeroid: field(&text, "zeroid"),
        joins: field(&text, "joins"),
    }
}

/// The control. Without it every sweep below could pass against a kernel that
/// spawns nothing: no creates, no joins, and the single-threaded arm's answer.
#[test]
fn the_wrappers_are_live_and_the_kernel_spawns_without_them() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let (dir, exe) = build("control");
    let r = run(&exe, 0, 0, "4");
    assert_eq!(r.code, Some(0), "the unperturbed run must succeed");
    assert_eq!(
        r.checksum,
        Some(reference_checksum()),
        "the kernel disagrees with an independent reference"
    );
    assert!(
        r.creates > 1,
        "the wrapped counter saw {} `pthread_create` calls, so `--wrap` did \
         not take and every mask below would be a no-op",
        r.creates
    );
    assert_eq!(
        r.joins, r.creates,
        "every spawned thread must be joined even with nothing injected"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A worker whose `pthread_t` is zero is still joined.
///
/// The JOIN COUNT is the assertion, because the answer is not reliably wrong
/// when a join is skipped - at one or two aliased workers it came back right
/// every time. The exit code is asserted too: with all four aliased, the
/// sentinel version segfaulted on every run.
#[test]
fn a_worker_whose_thread_id_is_zero_is_still_joined() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let (dir, exe) = build("zeroid");
    let want = reference_checksum();

    for mask in [1i64, 3, 6, 15] {
        let r = run(&exe, 0, mask, "4");
        assert_eq!(
            r.code,
            Some(0),
            "zero-id mask {mask} did not exit cleanly (code {:?}); skipping a \
             join lets the reduction free buffers a worker is still writing",
            r.code
        );
        assert!(
            r.zeroid > 0,
            "zero-id mask {mask} aliased no thread, so it asserts nothing"
        );
        assert_eq!(
            r.joins, r.creates,
            "zero-id mask {mask}: {} threads started and {} were joined - the \
             join loop is deciding from the thread id again",
            r.creates, r.joins
        );
        assert_eq!(
            r.checksum,
            Some(want),
            "zero-id mask {mask} changed the answer"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// A `pthread_create` that fails runs its band inline, and the answer does not
/// move. Nothing had ever executed this arm.
#[test]
fn a_failed_spawn_runs_its_band_inline_and_the_answer_does_not_move() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let (dir, exe) = build("failspawn");
    let want = reference_checksum();

    for mask in [1i64, 2, 6, 15] {
        let r = run(&exe, mask, 0, "4");
        assert_eq!(r.code, Some(0), "fail mask {mask} did not exit cleanly");
        assert!(
            r.failed > 0,
            "fail mask {mask} failed no spawn, so it asserts nothing"
        );
        assert_eq!(
            r.joins,
            r.creates - r.failed,
            "fail mask {mask}: the join loop must visit the spawns that \
             SUCCEEDED - {} created, {} failed, {} joined",
            r.creates,
            r.failed,
            r.joins
        );
        assert_eq!(
            r.checksum,
            Some(want),
            "fail mask {mask} changed the answer, so the inline-fallback arm \
             does not compute the band it replaces"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}
