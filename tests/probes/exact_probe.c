/* Phase 0 probe: what does EXACT accumulation cost in a register-blocked
 * GEMM micro-kernel?
 *
 * This is deliberately standalone rather than wired into cpu_gemm.rs. The
 * number it produces decides whether the whole proof-carrying-kernels
 * programme is worth starting, and threading DriftRepr through a 132 KB
 * emitter to find that out would be the expensive way to learn it.
 *
 * Both kernels use the SAME register budget (24 zmm accumulators), which is
 * the fair comparison — the f32 tile is Y's shipped one (mr=6, nr=64 as
 * 4 x <16 x float> per row) and the exact tile is what fits in the same 24
 * registers once each holds 8 int64 lanes instead of 16 float lanes:
 * mr=6, nr=32. That halving is not a tuning choice, it is forced by the
 * representation, and it costs arithmetic intensity on top of the
 * instruction mix.
 *
 * Fixed-point format: inputs are Q16.16 int32. The product of two Q16.16
 * values is Q32.32 in an int64, and accumulation is exact until the int64
 * overflows -- which for bounded inputs is what @bounds(min,max) exists to
 * establish. Integer addition is associative, so any reordering of the
 * K loop gives a bit-identical result. That is the property the whole
 * programme rests on and `exactness_holds` below checks it directly.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

#define MR 6
#define NR_F32 64          /* 4 x <16 x float>  per row -> 24 zmm */
#define NR_FIX 32          /* 4 x <8  x int64>  per row -> 24 zmm */
#define KC 256

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* The kernels live in exact_kernels.c and are linked WITHOUT LTO, on purpose.
 * Second measurement bug: with them inline, the exact kernel read 540 G MAC/s,
 * ~6x what vpmuldq can issue. INTEGER ADDITION IS ASSOCIATIVE, so the compiler
 * may hoist each of the 8 repeating A panels' contribution and multiply by the
 * repeat count -- it cannot do that to the f32 arm because float addition is
 * not associative. The property that makes exact accumulation PROVABLE is the
 * same property that lets a compiler eat its own benchmark. */
void micro_f32(const float *Ap, const float *Bp, float *C, long ldc, long kc);
void micro_fix(const int32_t *Ap, const int64_t *Bp, int64_t *C, long ldc, long kc);

/* ---- the property the programme rests on -------------------------------- */
static int exactness_holds(void) {
    const long n = 4001;
    int32_t *a = aligned_alloc(64, n * sizeof(int32_t));
    float   *f = aligned_alloc(64, n * sizeof(float));
    for (long i = 0; i < n; ++i) {
        double v = (i % 7 ? 1.0 / (i + 3) : (double)(i % 13));
        f[i] = (float)v;
        a[i] = (int32_t)llround(v * 65536.0);      /* Q16.16 */
    }
    int64_t fwd = 0, rev = 0;
    for (long i = 0; i < n; ++i)      fwd += (int64_t)a[i] * 65536;
    for (long i = n - 1; i >= 0; --i) rev += (int64_t)a[i] * 65536;

    float ffwd = 0.f, frev = 0.f;
    for (long i = 0; i < n; ++i)      ffwd += f[i];
    for (long i = n - 1; i >= 0; --i) frev += f[i];

    int exact_ok  = (fwd == rev);
    int f32_drifts = (ffwd != frev);   /* the control: without this the test is vacuous */
    printf("  exact  fwd==rev : %s\n", exact_ok ? "YES (bit-identical)" : "NO");
    printf("  f32    fwd==rev : %s\n",
           f32_drifts ? "NO  (drifts, as expected)" : "yes -- CONTROL FAILED, test is vacuous");
    free(a); free(f);
    return exact_ok && f32_drifts;
}

int main(void) {
    printf("Phase 0 probe: cost of exact accumulation in a GEMM micro-kernel\n");
    printf("MR=%d  f32 NR=%d  exact NR=%d  KC=%d  (both 24 zmm accumulators)\n\n",
           MR, NR_F32, NR_FIX, KC);

    printf("Exactness property:\n");
    int ok = exactness_holds();
    printf("\n");

    long ldc = 128;
    float   *Af = aligned_alloc(64, 8 * KC * MR * sizeof(float));
    float   *Bf = aligned_alloc(64, KC * NR_F32 * sizeof(float));
    float   *Cf = aligned_alloc(64, MR * ldc * sizeof(float));
    int32_t *Ax = aligned_alloc(64, 8 * KC * MR * sizeof(int32_t));
    int64_t *Bx = aligned_alloc(64, KC * NR_FIX * sizeof(int64_t));
    int64_t *Cx = aligned_alloc(64, MR * ldc * sizeof(int64_t));

    for (long i = 0; i < 8 * KC * MR; ++i) { Af[i] = (float)((i % 17) - 8) * 0.03f; Ax[i] = (int32_t)(Af[i] * 65536.f); }
    for (long i = 0; i < KC * NR_F32; ++i)   Bf[i] = (float)((i % 23) - 11) * 0.02f;
    for (long i = 0; i < KC * NR_FIX; ++i)   Bx[i] = (int64_t)(((float)((i % 23) - 11) * 0.02f) * 65536.f);
    memset(Cf, 0, MR * ldc * sizeof(float));
    memset(Cx, 0, MR * ldc * sizeof(int64_t));

    /* Ramp the clock -- this box idles at 624 MHz and needs load to boost. */
    double ramp = now_s();
    while (now_s() - ramp < 3.0) micro_f32(Af, Bf, Cf, ldc, KC);

    /* Each call reads a DIFFERENT A panel and accumulates into C, so the
       result changes every iteration and nothing can be hoisted. The first
       version of this probe timed 698 G MAC/s -- four times this core's f32
       FMA peak -- because the call was pure and the compiler lifted it out
       of the loop entirely. Same failure the @ZeroDrift GPU probe had. */
    const long REPS = 200000;
    const long PANELS = 8;
    double best_f32 = 1e30, best_fix = 1e30;

    for (int round = 0; round < 7; ++round) {
        double t0 = now_s();
        for (long r = 0; r < REPS; ++r) micro_f32(Af + (r % PANELS) * KC * MR, Bf, Cf, ldc, KC);
        double t1 = now_s();
        for (long r = 0; r < REPS; ++r) micro_fix(Ax + (r % PANELS) * KC * MR, Bx, Cx, ldc, KC);
        double t2 = now_s();
        if (t1 - t0 < best_f32) best_f32 = t1 - t0;
        if (t2 - t1 < best_fix) best_fix = t2 - t1;
    }

    double macs_f32 = (double)REPS * KC * MR * NR_F32;
    double macs_fix = (double)REPS * KC * MR * NR_FIX;
    double g_f32 = macs_f32 / best_f32 / 1e9;
    double g_fix = macs_fix / best_fix / 1e9;

    printf("Throughput (G MAC/s, best of 7 interleaved rounds):\n");
    printf("  f32 FMA          %8.2f   (%d cols/tile)\n", g_f32, NR_F32);
    printf("  exact Q16.16     %8.2f   (%d cols/tile)\n", g_fix, NR_FIX);
    printf("\n  exact / f32      %8.3fx\n", g_fix / g_f32);
    printf("  slowdown         %8.2fx\n", g_f32 / g_fix);

    double ghz = 5.5, peak = 32.0 * ghz;   /* 2x512-bit FMA = 32 f32 MAC/cycle */
    printf("\n  f32 vs this core's FMA peak (~%.0f G MAC/s at %.1f GHz): %.0f%%\n",
           peak, ghz, 100.0 * g_f32 / peak);
    if (g_f32 > peak) printf("  *** ABOVE PEAK -- the measurement is wrong, not the kernel ***\n");
    printf("  sink %g %lld\n", (double)Cf[0], (long long)Cx[0]);
    return ok ? 0 : 1;
}
