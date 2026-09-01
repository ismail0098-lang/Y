//! Running out of memory must be a DEFINED failure, not a null dereference.
//!
//! ** What was wrong.
//!
//! `emit_vnni_threaded_module` made nine `malloc` calls and checked none of
//! them - the returned pointer went straight into a `memset`. An out-of-memory
//! condition was therefore undefined behaviour, and a null `memset` is a wild
//! write rather than reliably a clean segfault.
//!
//! It is not a remote hazard. The per-thread private C copy is `M * N * 8`
//! bytes **per thread**, so a 4096x4096 GEMM on 16 threads asks for 2.1 GB in
//! that one allocation. And the f32 kernel *in the same emitter* has had a
//! malloc-failure fallback (`@__y_gemm_fallback`, guarded by
//! `icmp ne ptr %g5, null`) the whole time - so this was an asymmetry between
//! two kernels in one file, not an oversight nobody had thought about.
//!
//! ** What it does now, and why it is an exit rather than a fallback.
//!
//! Every allocation goes through `@__y_gemm_exact_alloc`, which prints the
//! byte count and exits 1. It cannot do what the f32 path does: that kernel
//! blocks by `kc`/`mc`/`nc` panels so a fixed static reserve covers it, while
//! this one packs the WHOLE A matrix once - the property that makes its
//! packing asymptotically free - so its panel size is unbounded in M and K.
//! A blocked fallback variant is a feature to build, not a branch to add.
//!
//! Until it exists the choice is between a defined failure and an undefined
//! one, and the repository's design rule settles it: a wrong answer under a
//! certificate claiming exactness is the failure this programme exists to
//! prevent. The residue stays named on the certificate's trust boundary.
//!
//! ** What these tests can and cannot see.
//!
//! `-Wl,--wrap=malloc` redirects the kernel's allocations to a counter that
//! returns NULL on the n-th call. Every n is swept, in both the threaded and
//! the single-threaded arm, and each must exit **1** with the diagnostic on
//! stdout - `status.code() == Some(1)`, which a signal death does not satisfy.
//!
//! They cannot see an allocation made inside libc (`printf`'s buffers are not
//! wrapped, which is deliberate - wrapping them would make the diagnostic
//! itself unprintable), and they sweep the allocations that SURVIVE to
//! runtime: `clang -O2` promotes a non-escaping `malloc`/`free` pair to an
//! `alloca`, which is why the observed counts are 2 single-threaded and 14 at
//! four threads rather than 3 and 18. An `alloca` cannot fail, so nothing is
//! lost - but the sweep is over what the binary does, not over what the IR
//! says, and those differ here. And the control is what stops the whole file passing
//! against a kernel that allocates nothing: at `fail_at = 0` the run must
//! succeed, produce the right answer, and report a NON-ZERO number of wrapped
//! allocations.

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

const M: usize = 24;
const N: usize = 64;
const K: usize = 512;

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

/// The driver allocates NOTHING on the heap: every buffer is static, so the
/// wrapped counter sees only the kernel's own allocations and `fail_at` means
/// what it says.
const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void y_matmul(const int16_t *A, const int16_t *B, int64_t *C, int M, int N, int K);

static long calls = 0;
static long fail_at = 0;

void *__real_malloc(size_t);
void *__wrap_malloc(size_t n) {
    calls++;
    if (fail_at > 0 && calls == fail_at) return 0;
    return __real_malloc(n);
}

#define MM 24
#define NN 64
#define KK 512
static int16_t A[MM * KK];
static int16_t B[KK * NN];
static int64_t C[MM * NN];

static int16_t gen(long i, long salt) {
    unsigned long h = i * 2654435761UL + salt * 40503UL;
    h ^= h >> 13;
    return (int16_t)((long)(h % 2049) - 1024);
}

int main(int argc, char **argv) {
    fail_at = (argc > 1) ? atol(argv[1]) : 0;
    for (long i = 0; i < MM * KK; i++) A[i] = gen(i, 1);
    for (long i = 0; i < KK * NN; i++) B[i] = gen(i, 2);

    y_matmul(A, B, C, MM, NN, KK);

    long long sum = 0;
    for (long i = 0; i < MM * NN; i++) sum += C[i];
    printf("checksum %lld allocs %ld\n", sum, calls);
    return 0;
}
"#;

/// Build the kernel and link it against the counting driver.
///
/// The tag is in the signature rather than in a comment asking the caller to
/// remember: two tests sharing a temp-dir name, one removing it while the
/// other writes, is a race this repository has hit five times.
fn build(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("y_exact_oom_{tag}_{}", std::process::id()));
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
    assert!(
        !ir.contains("  %ap1 = call ptr @malloc("),
        "the exact driver is calling `malloc` directly again. Every allocation \
         on this path must go through `@__y_gemm_exact_alloc`, or an \
         out-of-memory condition is a null dereference."
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
            "-Wl,--wrap=malloc",
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

/// The independent answer, so the control arm checks correctness rather than
/// merely exit status.
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

fn run(exe: &PathBuf, fail_at: i64, threads: &str) -> std::process::Output {
    Command::new(exe)
        .arg(fail_at.to_string())
        .env("Y_NUM_THREADS", threads)
        .output()
        .expect("run the kernel")
}

/// The control, and it is what stops every assertion below being vacuous.
///
/// With no injected failure the kernel must produce the right answer AND the
/// wrapper must report a non-zero count. A `--wrap` that silently did not
/// apply, or a kernel that stopped allocating, would make every sweep below
/// pass by never reaching the path it is about.
#[test]
fn the_wrapper_is_live_and_the_kernel_is_correct_without_an_injected_failure() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let (dir, exe) = build("control");
    let out = run(&exe, 0, "4");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(
        out.status.code(),
        Some(0),
        "the unperturbed run must succeed:\n{text}{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let want = reference_checksum();
    assert!(
        text.contains(&format!("checksum {want} ")),
        "the kernel disagrees with an independent reference. got: {text}"
    );

    let allocs: i64 = text
        .split("allocs ")
        .nth(1)
        .and_then(|t| t.trim().parse().ok())
        .expect("the driver reports its allocation count");
    assert!(
        allocs > 0,
        "the wrapped counter saw {allocs} allocations, so `-Wl,--wrap=malloc` \
         did not take and every injected failure below would be a no-op"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Every allocation the kernel makes, failed one at a time, in both arms.
///
/// The assertion is `status.code() == Some(1)`: a process killed by SIGSEGV
/// has no exit code at all, so this distinguishes the defined failure from
/// the undefined one it replaced rather than merely observing that the run
/// did not succeed.
#[test]
fn every_allocation_failure_exits_with_a_diagnosis_rather_than_a_null_dereference() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let (dir, exe) = build("sweep");

    for threads in ["1", "4"] {
        let baseline = run(&exe, 0, threads);
        let text = String::from_utf8_lossy(&baseline.stdout).into_owned();
        let allocs: i64 = text
            .split("allocs ")
            .nth(1)
            .and_then(|t| t.trim().parse().ok())
            .expect("allocation count");
        assert!(
            allocs > 0,
            "no allocations observed at Y_NUM_THREADS={threads}; nothing to fail"
        );

        for n in 1..=allocs {
            let out = run(&exe, n, threads);
            let so = String::from_utf8_lossy(&out.stdout).into_owned();
            assert_eq!(
                out.status.code(),
                Some(1),
                "Y_NUM_THREADS={threads}, failing allocation {n} of {allocs}: expected a \
                 clean exit(1), got {:?}. `None` means the process died by signal - which \
                 is the null dereference this test exists to rule out.\nstdout: {so}",
                out.status.code()
            );
            assert!(
                so.contains("could not allocate"),
                "Y_NUM_THREADS={threads}, allocation {n}: exited 1 without saying why. \
                 An exit code alone is not a diagnosis.\nstdout: {so}"
            );
            assert!(
                !so.contains("checksum "),
                "Y_NUM_THREADS={threads}, allocation {n}: the kernel printed a result \
                 after failing to allocate. A partial answer under an exactness \
                 certificate is worse than no answer.\nstdout: {so}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}
