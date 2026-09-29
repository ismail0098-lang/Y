/* Times ONE shape, in its OWN process, for ONE arm.
 *
 * One shape per process per arm is not fussiness: the repo's own tuning
 * document records that timing two GEMM libraries in one process measured
 * 512^3 16T scaling at 1.95x interleaved against 7.35x standalone, because
 * OpenBLAS's idle threads spin before parking and whichever ran second was
 * measured against a busy machine.
 *
 * Reports the MINIMUM over rounds. The checksum goes to stderr so nothing can
 * be dead-code-eliminated, and A/B share a PRNG seed so the exact and f32 arms
 * see the same matrices - the f32 checksum should equal the integer one at
 * these magnitudes, which is a free cross-check that both arms compute the
 * same GEMM.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#if ARM_EXACT
void y_matmul(const int16_t *, const int16_t *, int64_t *, int, int, int);
#elif ARM_YF32
void f32_matmul(const float *, const float *, float *, int, int, int);
#else /* ARM_OPENBLAS */
typedef long long bi;
void CBLAS_SGEMM(int, int, int, bi, bi, bi, float, const float *, bi,
                 const float *, bi, float, float *, bi);
void CBLAS_SET_THREADS(int);
char *CBLAS_CORENAME(void);
#endif

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: M N K reps rounds [threads]\n"); return 2; }
    int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
    int reps = atoi(argv[4]), rounds = atoi(argv[5]);
    int thr = argc > 6 ? atoi(argv[6]) : 16;
    (void)thr;
    srand(1);
    double best = 1e30;

#if ARM_EXACT
    int16_t *A = malloc((size_t)M * K * 2), *B = malloc((size_t)K * N * 2);
    int64_t *C = malloc((size_t)M * N * 8);
    if (!A || !B || !C) return 3;
    for (long i = 0; i < (long)M * K; i++) A[i] = (int16_t)((rand() % 2001) - 1000);
    for (long i = 0; i < (long)K * N; i++) B[i] = (int16_t)((rand() % 2001) - 1000);
    y_matmul(A, B, C, M, N, K);                       /* warm */
    for (int r = 0; r < rounds; r++) {
        double t = now();
        for (int i = 0; i < reps; i++) y_matmul(A, B, C, M, N, K);
        double d = (now() - t) / reps;
        if (d < best) best = d;
    }
    fprintf(stderr, "chk %lld\n", (long long)C[0]);
#elif ARM_YF32
    float *A = malloc((size_t)M * K * 4), *B = malloc((size_t)K * N * 4), *C = malloc((size_t)M * N * 4);
    if (!A || !B || !C) return 3;
    for (long i = 0; i < (long)M * K; i++) A[i] = (float)((rand() % 2001) - 1000);
    for (long i = 0; i < (long)K * N; i++) B[i] = (float)((rand() % 2001) - 1000);
    f32_matmul(A, B, C, M, N, K);
    for (int r = 0; r < rounds; r++) {
        double t = now();
        for (int i = 0; i < reps; i++) f32_matmul(A, B, C, M, N, K);
        double d = (now() - t) / reps;
        if (d < best) best = d;
    }
    fprintf(stderr, "chk %.0f\n", C[0]);
#else
    CBLAS_SET_THREADS(thr);
    if (getenv("Y_SHOW_CORE")) fprintf(stderr, "openblas kernel: %s\n", CBLAS_CORENAME());
    float *A = malloc((size_t)M * K * 4), *B = malloc((size_t)K * N * 4), *C = malloc((size_t)M * N * 4);
    if (!A || !B || !C) return 3;
    for (long i = 0; i < (long)M * K; i++) A[i] = (float)((rand() % 2001) - 1000);
    for (long i = 0; i < (long)K * N; i++) B[i] = (float)((rand() % 2001) - 1000);
    /* CblasRowMajor=101, CblasNoTrans=111 */
    CBLAS_SGEMM(101, 111, 111, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    for (int r = 0; r < rounds; r++) {
        double t = now();
        for (int i = 0; i < reps; i++)
            CBLAS_SGEMM(101, 111, 111, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        double d = (now() - t) / reps;
        if (d < best) best = d;
    }
    fprintf(stderr, "chk %.0f\n", C[0]);
#endif
    printf("%.9f\n", best);
    return 0;
}
