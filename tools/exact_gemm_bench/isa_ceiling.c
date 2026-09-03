/* What the two datapaths can actually issue, so a GEMM throughput can be read
 * as a fraction of ITS OWN ceiling rather than compared across ISAs.
 *
 * `vpdpwssd` retires 32 int16 multiply-accumulates; `vfmadd132ps` retires 16
 * f32 ones. If the two issue at the same rate then the exact datapath's MAC
 * ceiling is DOUBLE the f32 one, and "exact is 0.93x OpenBLAS in wall clock"
 * and "exact is at half OpenBLAS's ISA efficiency" are both true at once.
 * That is not a fact to assume from a manual - it is measured here.
 *
 * Sixteen independent accumulators so neither chain is latency-bound, and the
 * loop is verified non-foldable two ways: the disassembly must contain the
 * instruction (`objdump -d isa_ceiling | grep -c vpdpwssd`), and doubling
 * ITERS must double the time.
 */
#include <stdio.h>
#include <immintrin.h>
#include <time.h>

#define ACCS 16

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

__attribute__((target("avx512f,avx512vnni"), noinline))
static double vnni_secs(long iters) {
    __m512i a = _mm512_set1_epi16(3), b = _mm512_set1_epi16(5), c[ACCS];
    for (int i = 0; i < ACCS; i++) c[i] = _mm512_setzero_si512();
    double t = now();
    for (long i = 0; i < iters; i++)
        for (int j = 0; j < ACCS; j++) c[j] = _mm512_dpwssd_epi32(c[j], a, b);
    double d = now() - t;
    int s = 0;
    for (int i = 0; i < ACCS; i++) s += _mm512_reduce_add_epi32(c[i]);
    return d + 0.0 * s;
}

__attribute__((target("avx512f"), noinline))
static double fma_secs(long iters) {
    __m512 a = _mm512_set1_ps(3), b = _mm512_set1_ps(5), c[ACCS];
    for (int i = 0; i < ACCS; i++) c[i] = _mm512_setzero_ps();
    double t = now();
    for (long i = 0; i < iters; i++)
        for (int j = 0; j < ACCS; j++) c[j] = _mm512_fmadd_ps(a, b, c[j]);
    double d = now() - t;
    float s = 0;
    for (int i = 0; i < ACCS; i++) s += _mm512_reduce_add_ps(c[i]);
    return d + 0.0 * s;
}

int main(int argc, char **argv) {
    long iters = argc > 1 ? atol(argv[1]) : 20000000L;
    double bv = 1e30, bf = 1e30;
    for (int r = 0; r < 5; r++) {
        double v = vnni_secs(iters), f = fma_secs(iters);
        if (v < bv) bv = v;
        if (f < bf) bf = f;
    }
    double instr = (double)iters * ACCS;
    printf("vpdpwssd        %8.3f s   %6.2f G instr/s   %7.1f G MAC/s  (1 core)\n",
           bv, instr / bv / 1e9, instr * 32 / bv / 1e9);
    printf("vfmadd132ps     %8.3f s   %6.2f G instr/s   %7.1f G MAC/s  (1 core)\n",
           bf, instr / bf / 1e9, instr * 16 / bf / 1e9);
    printf("instruction-rate ratio (vnni / fma): %.2f\n", (instr / bv) / (instr / bf));
    printf("MAC-ceiling ratio  (exact / f32):    %.2f\n",
           (instr * 32 / bv) / (instr * 16 / bf));
    return 0;
}
