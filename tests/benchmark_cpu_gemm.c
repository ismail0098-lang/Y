// ============================================================
//  Y  —  CPU GEMM benchmark: Y's emitted kernel vs OpenBLAS
//  tests/benchmark_cpu_gemm.c
//
//  Build (see docs/cpu_gemm_tuning.md):
//    Y tests/y_cpu_matmul.ysu --emit-llvm
//    clang -O3 -c tests/y_cpu_matmul.ll -o /tmp/y_cpu_matmul.o
//    clang -O2 -o tests/benchmark_cpu_gemm tests/benchmark_cpu_gemm.c \
//          /tmp/y_cpu_matmul.o -L<openblas> -lopenblas -lm -lpthread
//
//  Run ONE library per process, thread count from the ENVIRONMENT:
//    Y_NUM_THREADS=16 OMP_NUM_THREADS=1  OMP_WAIT_POLICY=passive \
//        ./benchmark_cpu_gemm 16 7 y
//    OMP_NUM_THREADS=16 ./benchmark_cpu_gemm 16 7 ob
//
//  ---------------------------------------------------------------
//  Five measurement biases this file has had, all of which inflated
//  Y's reported advantage. Do not reintroduce any of them.
//  ---------------------------------------------------------------
//
//  1. TIMING BOTH LIBRARIES IN ONE PROCESS.
//     OpenBLAS's idle threads spin before parking, so whichever side ran
//     second was measured against a busy machine. Hence one library per
//     process, and `OMP_WAIT_POLICY=passive` in Y's process so that even the
//     single-threaded reference GEMM cannot leave a spinning team behind.
//
//  2. CALLING openblas_set_num_threads() ON AN OpenMP BUILD.
//     This build is USE_OPENMP (libgomp). Toggling the count through the
//     OpenBLAS API measured 1621 GFLOPS at 1024^3 where the same build driven
//     purely by OMP_NUM_THREADS measured 1825 — a 12% handicap applied to the
//     baseline only. Thread count now comes from the environment and this file
//     never calls the setter. The `threads` argv is used for LABELLING and for
//     a consistency check, nothing else.
//
//  3. SIZING THE TIMED BATCH FROM AN ASSUMED 100 GFLOPS.
//     `iters = 20ms / (flops / 100e9)` yields THREE iterations at 1024^3 on a
//     machine that does 2000+ GFLOPS — a 4 ms batch, not the 20 ms the comment
//     claimed, with the thread-pool wake amortised over almost nothing. The
//     count is now calibrated from a measured single call.
//
//  4. WARMING AT ONE THREAD AND THEN TIMING AT SIXTEEN.
//     The warm-up ran before the thread count was raised, so the first timed
//     round paid for team creation. Warm-up now runs in the final config.
//
//  5. RAMPING THE CLOCK ON ONE CORE.
//     A single-threaded spin loop leaves the other 15 cores at the 624 MHz
//     idle floor, so a 16-thread kernel starts against cold clocks. The ramp
//     is now all-core, and the achieved all-core frequency is reported so a
//     run at the wrong clock is visible rather than silent.
//
//  Retained discipline: rank by the MINIMUM across rounds (contamination is
//  one-sided), and gate every shape on relative L2 against OpenBLAS — a fast
//  wrong kernel is the failure mode this exists to catch. The spread between
//  the minimum and the median is printed so a ranking that sits inside the
//  run's own dispersion can be recognised as a tie.
// ============================================================

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

// Declared by hand: OpenBLAS's cblas.h pulls in common.h, which does not
// survive a modern clang's stdatomic.h.
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
void cblas_sgemm(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE,
                 int, int, int, float, const float *, int,
                 const float *, int, float, float *, int);
int openblas_get_num_threads(void);
char *openblas_get_config(void);
char *openblas_get_corename(void);

// Emitted by the Y compiler from tests/y_cpu_matmul.ysu.
void y_cpu_matmul_kernel(const float *A, const float *B, float *C,
                         int M, int N, int K);

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct { int m, n, k; const char *tag; } Shape;

static const Shape SHAPES[] = {
    {  256,  256,  256, "nice    256^3"},
    {  512,  512,  512, "nice    512^3"},
    { 1024, 1024, 1024, "nice   1024^3"},
    { 2048, 2048, 2048, "nice   2048^3"},
    {  250,  250,  250, "ragged  250^3"},
    { 1000, 1000, 1000, "ragged 1000^3"},
    { 1021, 1021, 1021, "ragged 1021^3 prime"},
    {  137,  391, 1013, "ragged 137x391x1013"},
    {  333,  777,   64, "ragged 333x777x64"},
    { 4096, 4096,    8, "flatK  4096x4096x8"},
    {   64,   64,32768, "deepK  64x64x32768"},
    {   33, 4096, 4096, "skinny 33x4096x4096"},
    {   17, 4096, 4096, "skinny 17x4096x4096"},
    {    8, 4096, 4096, "decode  8x4096x4096"},
    {    4, 4096, 4096, "decode  4x4096x4096"},
    {    1, 4096, 4096, "gemv    1x4096x4096"},
    {    1, 8192, 8192, "gemv    1x8192x8192"},
    {   48,   48,   48, "tiny    48^3"},
    /* Appended, so indices 0-17 keep the meaning the docs and the harnesses
       quote.  The benchmark set jumps straight from 48^3 (110 KMAC) to 250^3
       (15.6 MMAC), which is a 140x gap with nothing in it -- and the crossover
       between the copy-free path and the packed one lives inside that gap, so
       it could not be located with the shapes that were here.  N stays <= 64
       on the first three because that is the copy-free path's structural
       limit; the fourth is past it and is the control. */
    {   64,   64,   64, "small   64^3"},
    {  128,   64,  128, "small   128x64x128"},
    {  256,   64,  256, "small   256x64x256"},
    {  128,  128,  128, "small   128^3"},
};
#define NSHAPES ((int)(sizeof(SHAPES) / sizeof(SHAPES[0])))

// ---- all-core clock ramp -------------------------------------------------
// A one-core spin loop leaves the rest of the package at its idle floor. This
// box idles at 624 MHz and boosts to ~5.7 GHz single-core / ~5.1 all-core, so
// starting a 16-thread kernel against cold siblings is a multiple-x bias, not
// noise -- and it lands on whichever library is threaded at the time.

static volatile int g_ramp_stop;

static void *ramp_worker(void *arg) {
    volatile double x = 1.0;
    (void)arg;
    while (!g_ramp_stop)
        for (int i = 0; i < 10000; ++i) x = x * 1.0000001 + 1e-9;
    (void)x;
    return NULL;
}

/// Mean of `scaling_cur_freq` across all online cores, in GHz.
///
/// Read from sysfs rather than timed from an FMA chain: the obvious
/// "time a dependent chain of known length" probe is CONSTANT-FOLDABLE, and
/// `clang -O2` duly deleted it -- the first version of this function reported
/// 4,000,406 GHz. That is the same trap the GPU drift probe hit (see
/// `measure_accumulate_costs` in the project docs); a probe that reports an
/// absurd number is lucky, a probe that reports a plausible wrong one is not.
static double mean_core_ghz(int nthr) {
    double sum = 0; int n = 0;
    for (int c = 0; c < nthr; ++c) {
        char p[128];
        snprintf(p, sizeof p,
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", c);
        FILE *f = fopen(p, "r");
        if (!f) continue;
        long khz = 0;
        if (fscanf(f, "%ld", &khz) == 1 && khz > 0) { sum += khz / 1e6; n++; }
        fclose(f);
    }
    return n ? sum / n : 0.0;
}

/// Spin every core for `seconds`, then report the all-core frequency actually
/// reached while still under load.
static double ramp_all_cores(double seconds, int nthr) {
    pthread_t th[64];
    int n = nthr > 64 ? 64 : nthr;
    g_ramp_stop = 0;
    for (int i = 0; i < n; ++i) pthread_create(&th[i], NULL, ramp_worker, NULL);

    double t0 = now_s();
    volatile double x = 1.0;
    while (now_s() - t0 < seconds)
        for (int i = 0; i < 10000; ++i) x = x * 1.0000001 + 1e-9;

    double ghz = mean_core_ghz(n);   // sampled while the ramp threads still run
    g_ramp_stop = 1;
    for (int i = 0; i < n; ++i) pthread_join(th[i], NULL);
    (void)x;
    return ghz;
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 1;
    int rounds  = argc > 2 ? atoi(argv[2]) : 7;
    const char *mode = argc > 3 ? argv[3] : "y";
    int time_y = (mode[0] == 'y');

    // Thread count is environment-driven for BOTH sides. Never call
    // openblas_set_num_threads here: on this USE_OPENMP build it costs the
    // baseline 12% (see bias #2 at the top of this file).
    const char *ynt = getenv("Y_NUM_THREADS");
    const char *ont = getenv("OMP_NUM_THREADS");
    int ob_thr = openblas_get_num_threads();

    printf("# %s  |  %d rounds, ranked by minimum\n",
           time_y ? "Y" : "OpenBLAS", rounds);
    printf("# openblas: %s / core=%s\n", openblas_get_config(), openblas_get_corename());
    printf("# env: Y_NUM_THREADS=%s OMP_NUM_THREADS=%s OMP_WAIT_POLICY=%s\n",
           ynt ? ynt : "(unset)", ont ? ont : "(unset)",
           getenv("OMP_WAIT_POLICY") ? getenv("OMP_WAIT_POLICY") : "(unset)");
    printf("# openblas_get_num_threads()=%d, expected %d\n", ob_thr, threads);

    // Fail loudly rather than silently benchmarking a 1-thread baseline
    // against a 16-thread kernel.
    if (!time_y && ob_thr != threads) {
        fprintf(stderr,
                "FATAL: OpenBLAS reports %d threads but %d were requested. "
                "Set OMP_NUM_THREADS=%d.\n", ob_thr, threads, threads);
        return 3;
    }
    if (time_y && ob_thr != 1) {
        fprintf(stderr,
                "FATAL: in 'y' mode OpenBLAS must be single-threaded (it only "
                "computes the reference). Set OMP_NUM_THREADS=1.\n");
        return 3;
    }
    if (time_y && (!ynt || atoi(ynt) != threads)) {
        fprintf(stderr, "FATAL: set Y_NUM_THREADS=%d.\n", threads);
        return 3;
    }

    printf("ramping all %d cores...\n", threads);
    double ghz = ramp_all_cores(3.0, threads);
    printf("# achieved clock during all-core load: %.2f GHz\n", ghz);

    printf("%-22s %12s %12s %8s  %s\n", "shape", "us", "GFLOPS", "med/min", "relL2");

    int failures = 0;

    // Index filter, so a single shape can be re-measured in a fresh process.
    // Not a convenience: running the full sweep in one process was found to
    // depress the later shapes, so isolating one is how that gets checked.
    //
    // Deliberately an INDEX and not a substring. The substring version silently
    // reported 2048^3's numbers under the "tiny 48^3" label, because "48^3" is
    // a substring of "2048^3" -- a filter that matches the wrong row does not
    // fail, it relabels.
    const char *sel = getenv("SHAPE_INDEX");
    int only = sel ? atoi(sel) : -1;
    if (sel && (only < 0 || only >= NSHAPES)) {
        fprintf(stderr, "FATAL: SHAPE_INDEX=%s out of range 0..%d\n",
                sel, NSHAPES - 1);
        return 3;
    }

    for (int s = 0; s < NSHAPES; ++s) {
        if (only >= 0 && s != only) continue;
        int M = SHAPES[s].m, N = SHAPES[s].n, K = SHAPES[s].k;
        size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
        float *A  = aligned_alloc(64, (na * 4 + 63) & ~63UL);
        float *B  = aligned_alloc(64, (nb * 4 + 63) & ~63UL);
        float *Cy = aligned_alloc(64, (nc * 4 + 63) & ~63UL);
        float *Cb = aligned_alloc(64, (nc * 4 + 63) & ~63UL);
        if (!A || !B || !Cy || !Cb) { fprintf(stderr, "alloc failed\n"); return 2; }

        for (size_t i = 0; i < na; ++i) A[i] = (float)((i * 2654435761u >> 8) & 0xff) / 255.0f - 0.5f;
        for (size_t i = 0; i < nb; ++i) B[i] = (float)((i * 40503u >> 4) & 0xff) / 255.0f - 0.5f;
        memset(Cy, 0, nc * 4);
        memset(Cb, 0, nc * 4);

        // Correctness gate. Relative L2 over the whole matrix, not per-element
        // tolerance: at large K the summation order differs and per-element
        // rtol/atol produces false alarms without catching anything a norm
        // misses. In 'ob' mode OpenBLAS is its own reference, so this is
        // trivially 0 and the gate carries no information -- it is the 'y'
        // run that has to pass it.
        double rel = 0.0;
        if (time_y) {
            y_cpu_matmul_kernel(A, B, Cy, M, N, K);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                        1.0f, A, K, B, N, 0.0f, Cb, N);
            double num = 0, den = 0;
            for (size_t i = 0; i < nc; ++i) {
                double d = (double)Cy[i] - (double)Cb[i];
                num += d * d;
                den += (double)Cb[i] * (double)Cb[i];
            }
            rel = sqrt(num) / (sqrt(den) + 1e-30);
        }
        int ok = rel < 1e-5;
        if (!ok) failures++;

        double flops = 2.0 * M * N * K;

        #define CALL()                                                        \
            do { if (time_y) y_cpu_matmul_kernel(A, B, Cy, M, N, K);          \
                 else cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,  \
                                  M, N, K, 1.0f, A, K, B, N, 0.0f, Cb, N);    \
            } while (0)

        // Warm up IN THE FINAL THREADING CONFIG, then calibrate the batch size
        // from a measured call rather than an assumed 100 GFLOPS.
        for (int w = 0; w < 5; ++w) CALL();
        double probe0 = now_s();
        CALL();
        double per = now_s() - probe0;
        if (per <= 0) per = 1e-9;
        long iters = (long)(50e-3 / per);          // target a 50 ms batch
        if (iters < 5) iters = 5;
        if (iters > 200000) iters = 200000;

        double times[64];
        if (rounds > 64) rounds = 64;
        for (int r = 0; r < rounds; ++r) {
            double t0 = now_s();
            for (long i = 0; i < iters; ++i) CALL();
            times[r] = (now_s() - t0) / (double)iters;
        }
        #undef CALL

        // min and median: if med/min is far from 1.0 the run was contended and
        // the ranking should not be trusted to a few percent.
        for (int i = 1; i < rounds; ++i)
            for (int j = i; j > 0 && times[j] < times[j-1]; --j) {
                double tmp = times[j]; times[j] = times[j-1]; times[j-1] = tmp;
            }
        double tmin = times[0], tmed = times[rounds / 2];

        printf("%-22s %12.3f %12.2f %8.3f  %8.1e%s\n",
               SHAPES[s].tag, tmin * 1e6, flops / tmin / 1e9, tmed / tmin, rel,
               ok ? "" : "   *** MISMATCH ***");
        fflush(stdout);

        free(A); free(B); free(Cy); free(Cb);
    }

    if (failures) {
        printf("%d shape(s) FAILED the correctness gate\n", failures);
        return 1;
    }
    return 0;
}
