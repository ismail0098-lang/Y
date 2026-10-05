//! The exact helper's public signed extents preserve empty source-loop ranges.
//! Compare the emitted native entries with an independent signed-loop oracle,
//! including a padded destination, surrounding canaries, inaccessible inputs
//! for empty ranges, and allocation/thread observations.

use std::process::Command;
use y::cpu_gemm::{emit_vnni_gemm_module, emit_vnni_threaded_module, VNNI_MR, VNNI_NR};

const DECLS: &str = r#"
declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @printf(ptr, ...)
declare void @exit(i32)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)
"#;

const DRIVER: &str = r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

void __y_gemm_exact_vnni_threaded(const int16_t*, const int16_t*, int64_t*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void __y_gemm_exact_vnni(const int16_t*, const int16_t*, int64_t*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int16_t*, int16_t*, int64_t*);

static unsigned allocations, spawns;
void *__real_malloc(size_t);
void *__wrap_malloc(size_t n) { ++allocations; return __real_malloc(n); }
int __real_pthread_create(pthread_t*, const pthread_attr_t*, void*(*)(void*), void*);
int __wrap_pthread_create(pthread_t *t, const pthread_attr_t *a,
                         void *(*f)(void*), void *p) {
    ++spawns; return __real_pthread_create(t, a, f, p);
}

enum { LDA = 11, LDB = 13, LDC = 17, GUARD = 8, CWORDS = GUARD + 4 * LDC + GUARD };
static int16_t A[4 * LDA], B[8 * LDB];
static int16_t AP[MR * 2 * 8], BP[NR * 2 * 8];
static int64_t CT[MR * NR];
static int64_t got[CWORDS], want[CWORDS];

/* Both entries use signed ranges. The wrapper assigns C; the lower entry
   accumulates into C, and so an empty contraction leaves its destination as-is. */
static void reference(int assign, int64_t m, int64_t n, int64_t k) {
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j) {
            int64_t sum = 0;
            for (int64_t q = 0; q < k; ++q)
                sum += (int64_t)A[i * LDA + q] * B[q * LDB + j];
            if (assign) want[GUARD + i * LDC + j] = sum;
            else want[GUARD + i * LDC + j] += sum;
        }
}

static int check(int assign, int64_t m, int64_t n, int64_t k) {
    for (unsigned q = 0; q < CWORDS; ++q)
        got[q] = want[q] = INT64_C(0x1234000000000000) + q * 19;
    memset(AP, 0x5a, sizeof(AP));
    memset(BP, 0x6b, sizeof(BP));
    memset(CT, 0x7c, sizeof(CT));
    reference(assign, m, n, k);
    int active = m > 0 && n > 0 && k > 0;
    const int16_t *a = active ? A : NULL, *b = active ? B : NULL;
    allocations = spawns = 0;
    if (assign)
        __y_gemm_exact_vnni_threaded(a, b, got + GUARD, m, n, k, LDA, LDB, LDC);
    else
        __y_gemm_exact_vnni(a, b, got + GUARD, m, n, k, LDA, LDB, LDC,
                           active ? AP : NULL, active ? BP : NULL, active ? CT : NULL);
    if (memcmp(got, want, sizeof(got))) {
        fprintf(stderr, "%s mismatch M=%lld N=%lld K=%lld\n", assign ? "assign" : "accumulate",
                (long long)m, (long long)n, (long long)k);
        return 1;
    }
    if (!active && (allocations || spawns)) {
        fprintf(stderr, "empty range allocated or spawned: %u/%u\n", allocations, spawns);
        return 2;
    }
    if (active && assign && !allocations) {
        fprintf(stderr, "positive control did not exercise allocation instrumentation\n");
        return 3;
    }
    return 0;
}

int main(int argc, char **argv) {
    int positive = argc == 1 || strcmp(argv[1], "inactive-only");
    setenv("Y_NUM_THREADS", "4", 1);
    for (unsigned q = 0; q < sizeof(A)/sizeof(A[0]); ++q) A[q] = (int16_t)((q * 7) % 101 - 50);
    for (unsigned q = 0; q < sizeof(B)/sizeof(B[0]); ++q) B[q] = (int16_t)((q * 13) % 103 - 51);
    const int64_t ms[] = { INT64_MIN, -7, 0, 3 };
    const int64_t ns[] = { INT64_MIN, -9, 0, 5 };
    const int64_t ks[] = { INT64_MIN, -11, 0, 7 };
    unsigned cases = 0;
    for (unsigned i = 0; i < 4; ++i)
        for (unsigned j = 0; j < 4; ++j)
            for (unsigned q = 0; q < 4; ++q) {
                if (!positive && ms[i] > 0 && ns[j] > 0 && ks[q] > 0) continue;
                for (int assign = 0; assign < 2; ++assign) {
                    int r = check(assign, ms[i], ns[j], ks[q]);
                    if (r) return r;
                    ++cases;
                }
            }
    /* A vacuous output range permits even C to be inaccessible. Use extreme
       other dimensions to ensure their derived sizes cannot be evaluated first. */
    __y_gemm_exact_vnni_threaded(NULL, NULL, NULL, 0, INT64_MAX, INT64_MAX, 0, 0, 0);
    __y_gemm_exact_vnni_threaded(NULL, NULL, NULL, INT64_MAX, INT64_MIN, INT64_MAX, 0, 0, 0);
    __y_gemm_exact_vnni(NULL, NULL, NULL, INT64_MAX, INT64_MAX, INT64_MIN, 0, 0, 0, NULL, NULL, NULL);
    printf("signed dimensions: %u native differential cases passed\n", cases);
    return 0;
}
"#;

#[test]
fn exact_native_helpers_preserve_signed_empty_ranges() {
    let clang = Command::new("clang").arg("--version").output();
    if !clang.map(|o| o.status.success()).unwrap_or(false) {
        if std::env::var_os("Y_VERIFICATION_STRICT").is_some() {
            panic!("clang required for native signed-dimension verification");
        }
        eprintln!("skipping native signed-dimension verification: clang unavailable");
        return;
    }
    let dir =
        std::env::temp_dir().join(format!("y-exact-signed-dimensions-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ir = format!(
        "{DECLS}\n{}\n{}",
        emit_vnni_gemm_module(64),
        emit_vnni_threaded_module(true)
    );
    std::fs::write(dir.join("helpers.ll"), ir).unwrap();
    std::fs::write(
        dir.join("driver.c"),
        format!("#define MR {VNNI_MR}\n#define NR {VNNI_NR}\n{DRIVER}"),
    )
    .unwrap();
    let exe = dir.join("driver");
    let compile = Command::new("clang")
        .arg("-O2")
        .arg(dir.join("helpers.ll"))
        .arg(dir.join("driver.c"))
        .args([
            "-lpthread",
            "-Wl,--wrap=malloc",
            "-Wl,--wrap=pthread_create",
            "-o",
        ])
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "clang failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let vnni = std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.contains("avx512_vnni"))
        .unwrap_or(false);
    let mut run = Command::new(&exe);
    if !vnni {
        run.arg("inactive-only");
    }
    let result = run.output().unwrap();
    assert!(
        result.status.success(),
        "native differential failed: {}\n{}; artifacts: {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr),
        dir.display()
    );
    eprint!("{}", String::from_utf8_lossy(&result.stdout));
    if !vnni && std::env::var_os("Y_VERIFICATION_STRICT").is_some() {
        panic!("inactive cases passed, but AVX512-VNNI is required for the positive control");
    }
    if !vnni {
        eprintln!("AVX512-VNNI unavailable; positive native control skipped");
    }
    std::fs::remove_dir_all(dir).unwrap();
}
