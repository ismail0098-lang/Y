/* Per-call latency when the thread pool is COLD, which the throughput harness
 * cannot see.
 *
 * `tests/benchmark_cpu_gemm.c` times a hot loop: calls are back-to-back, so the
 * pool's workers are still inside their spin window (`POOL_SPIN`) and a
 * dispatch is a release store plus a spin. Every threading decision in this
 * backend -- above all `WORK_PER_THREAD`, which decides how many threads a
 * shape gets at all -- was measured in exactly that regime.
 *
 * A caller who issues ONE small GEMM and then goes and does something else is
 * in the other regime: the workers have exhausted their spin and are blocked in
 * `pthread_cond_wait`, so each dispatch costs a futex wake and a scheduler
 * round trip per thread. If `WORK_PER_THREAD` is too small, that cost is paid
 * on shapes too small to amortise it, and the throughput benchmark reports the
 * opposite of what the user experiences.
 *
 * So: sleep long enough between calls to guarantee the pool has parked, then
 * time a SINGLE call. Report the median over many isolated calls, and the min.
 * Compare `Y_NUM_THREADS=1` against `Y_NUM_THREADS=16` on identical protocol --
 * the 1-thread arm never touches the pool, so it is the control.
 *
 * Build:
 *   clang -O2 -o tests/cold_call_cpu_gemm tests/cold_call_cpu_gemm.c \
 *         /tmp/y_cpu_matmul.o -lm -lpthread
 * Run:
 *   Y_NUM_THREADS=16 ./tests/cold_call_cpu_gemm 128 128 128 200
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void y_cpu_matmul_kernel(const float *A, const float *B, float *C,
                         int M, int N, int K);

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static float gen(long i, long s) {
    unsigned long h = (unsigned long)i * 2654435761UL + (unsigned long)s * 40503UL;
    h ^= h >> 13;
    return (float)((h & 0xffff) / 65535.0 - 0.5);
}

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 128;
    int N = argc > 2 ? atoi(argv[2]) : 128;
    int K = argc > 3 ? atoi(argv[3]) : 128;
    int reps = argc > 4 ? atoi(argv[4]) : 200;
    /* Gap between calls. Must exceed the pool's spin window by a wide margin,
       or the workers are still spinning and this measures the hot case again. */
    long gap_us = argc > 5 ? atol(argv[5]) : 2000;

    float *A = malloc((size_t)M * K * 4);
    float *B = malloc((size_t)K * N * 4);
    float *C = malloc((size_t)M * N * 4);
    if (!A || !B || !C) { printf("ALLOC FAIL\n"); return 2; }
    for (long i = 0; i < (long)M * K; i++) A[i] = gen(i, 1);
    for (long i = 0; i < (long)K * N; i++) B[i] = gen(i, 2);

    /* Warm: fault every page, start the pool, resolve the thread count, and let
       the clock ramp. None of that is what this probe is about. */
    for (int i = 0; i < 50; i++) y_cpu_matmul_kernel(A, B, C, M, N, K);
    double ramp = now_ns();
    while (now_ns() - ramp < 3e8) y_cpu_matmul_kernel(A, B, C, M, N, K);

    double *s = malloc(sizeof(double) * reps);
    struct timespec gap = {0, gap_us * 1000};
    for (int r = 0; r < reps; r++) {
        nanosleep(&gap, NULL);          /* workers exhaust their spin and park */
        double t0 = now_ns();
        y_cpu_matmul_kernel(A, B, C, M, N, K);
        s[r] = now_ns() - t0;
    }
    qsort(s, reps, sizeof(double), cmp_d);

    double med = s[reps / 2], lo = s[0], p90 = s[(int)(reps * 0.9)];
    double flops = 2.0 * M * N * K;
    printf("%4dx%4dx%4d  median %8.2f us  min %8.2f us  p90 %8.2f us   "
           "%7.1f GF (median)\n",
           M, N, K, med / 1e3, lo / 1e3, p90 / 1e3, flops / med);
    return 0;
}
