/* Phase 0, follow-up: does AVX512-VNNI make exact accumulation cheaper than
 * float, rather than 2.56x more expensive?
 *
 * Baselines carried over from exact_probe.c on the same box, one core:
 *   f32 FMA          166.9 G MAC/s  (95% of this core's ~176 peak)
 *   exact int64      65.2  G MAC/s  (2.56x slower)
 *
 * Kernels live in vnni_kernels.c and are linked WITHOUT LTO. Every throughput
 * figure is checked against the hardware issue ceiling before it is believed;
 * two earlier versions of the int64 probe reported physically impossible
 * numbers, the second because integer addition is associative and the compiler
 * batched the repeating panels.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MR 6
#define NR 64
#define KPAIRS 128        /* 256 k-steps, matching the int64 probe's KC */

void micro_vnni_raw(const int32_t *Ap, const int16_t *Bp, int32_t *C, long ldc, long kpairs);
void micro_vnni_exact(const int32_t *Ap, const int16_t *Bp, int64_t *C, long ldc, long kpairs);

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* Scalar reference over the same packed layout, so "exact" means "equals the
 * obvious serial answer", not "equals the other vector kernel". */
static void reference(const int32_t *Ap, const int16_t *Bp, int64_t *C, long ldc, long kpairs) {
    for (long p = 0; p < kpairs; ++p)
        for (int i = 0; i < MR; ++i) {
            int32_t packed = Ap[p * MR + i];
            int16_t a0 = (int16_t)(packed & 0xFFFF), a1 = (int16_t)(packed >> 16);
            for (int j = 0; j < NR; ++j) {
                int16_t b0 = Bp[p * NR * 2 + j * 2 + 0];
                int16_t b1 = Bp[p * NR * 2 + j * 2 + 1];
                C[i * ldc + j] += (int64_t)a0 * b0 + (int64_t)a1 * b1;
            }
        }
}

int main(void) {
    long ldc = 128;
    long PANELS = 8;
    int32_t *A  = aligned_alloc(64, PANELS * KPAIRS * MR * sizeof(int32_t));
    int16_t *B  = aligned_alloc(64, KPAIRS * NR * 2 * sizeof(int16_t));
    int32_t *C32 = aligned_alloc(64, MR * ldc * sizeof(int32_t));
    int64_t *C64 = aligned_alloc(64, MR * ldc * sizeof(int64_t));
    int64_t *Cref = aligned_alloc(64, MR * ldc * sizeof(int64_t));

    /* Magnitudes chosen so the int32 accumulator cannot overflow between
       flushes: |a|,|b| <= 1024 gives a product <= 2^20, and FLUSH_T=64 k-pairs
       is 128 products <= 2^27, comfortably inside int32. This is exactly the
       obligation @bounds(min,max) exists to discharge. */
    for (long i = 0; i < PANELS * KPAIRS * MR; ++i) {
        int16_t lo = (int16_t)(((i * 37) % 2048) - 1024);
        int16_t hi = (int16_t)(((i * 53) % 2048) - 1024);
        A[i] = ((int32_t)(uint16_t)hi << 16) | (uint16_t)lo;
    }
    for (long i = 0; i < KPAIRS * NR * 2; ++i) B[i] = (int16_t)(((i * 29) % 2048) - 1024);

    /* Exactness: the vector kernel must equal the scalar reference exactly. */
    memset(C64, 0, MR * ldc * sizeof(int64_t));
    memset(Cref, 0, MR * ldc * sizeof(int64_t));
    micro_vnni_exact(A, B, C64, ldc, KPAIRS);
    reference(A, B, Cref, ldc, KPAIRS);
    int mismatch = 0;
    for (int i = 0; i < MR; ++i)
        for (int j = 0; j < NR; ++j)
            if (C64[i * ldc + j] != Cref[i * ldc + j]) mismatch++;
    printf("Exactness vs scalar reference: %s (%d/%d elements differ)\n",
           mismatch ? "FAIL" : "EXACT", mismatch, MR * NR);
    if (mismatch) return 1;

    double ramp = now_s();
    while (now_s() - ramp < 3.0) micro_vnni_raw(A, B, C32, ldc, KPAIRS);

    const long REPS = 200000;
    double best_raw = 1e30, best_exact = 1e30;
    for (int r = 0; r < 7; ++r) {
        double t0 = now_s();
        for (long k = 0; k < REPS; ++k)
            micro_vnni_raw(A + (k % PANELS) * KPAIRS * MR, B, C32, ldc, KPAIRS);
        double t1 = now_s();
        for (long k = 0; k < REPS; ++k)
            micro_vnni_exact(A + (k % PANELS) * KPAIRS * MR, B, C64, ldc, KPAIRS);
        double t2 = now_s();
        if (t1 - t0 < best_raw)   best_raw   = t1 - t0;
        if (t2 - t1 < best_exact) best_exact = t2 - t1;
    }

    /* Each k-pair does 2 MACs per (row, column). */
    double macs = (double)REPS * KPAIRS * MR * NR * 2.0;
    double g_raw = macs / best_raw / 1e9, g_ex = macs / best_exact / 1e9;

    const double F32 = 166.9, INT64 = 65.2;      /* measured, same box, same probe */
    double ghz = 5.5, fma_peak = 32.0 * ghz;     /* 2 x 512-bit FMA, f32 */
    double vnni_peak = 64.0 * ghz;               /* 2 x vpdpwssd, 32 MAC each */

    printf("\nThroughput (G MAC/s, best of 7 interleaved rounds, one core):\n");
    printf("  f32 FMA            %8.1f   (carried over)\n", F32);
    printf("  exact int64        %8.1f   (carried over)\n", INT64);
    printf("  VNNI raw           %8.1f   no flush -- upper bound, overflows\n", g_raw);
    printf("  VNNI exact         %8.1f   flushed to int64, verified exact\n", g_ex);
    printf("\n  VNNI exact / f32   %8.3fx\n", g_ex / F32);
    printf("  VNNI exact / int64 %8.3fx\n", g_ex / INT64);
    printf("  flush overhead     %8.1f%%\n", 100.0 * (g_raw - g_ex) / g_raw);
    printf("\n  ceilings: f32 FMA ~%.0f, vpdpwssd ~%.0f G MAC/s at %.1f GHz\n",
           fma_peak, vnni_peak, ghz);
    if (g_raw > vnni_peak || g_ex > vnni_peak)
        printf("  *** ABOVE PEAK -- the measurement is wrong, not the kernel ***\n");
    printf("  sink %lld %lld\n", (long long)C32[0], (long long)C64[0]);
    return 0;
}
