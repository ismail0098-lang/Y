/* AVX512-VNNI exact-accumulation micro-kernels.
 *
 * Separate translation unit, linked without LTO -- see exact_probe.c for why
 * that is mandatory rather than tidy.
 *
 * vpdpwssd computes, per int32 lane: acc += a[2i]*b[2i] + a[2i+1]*b[2i+1].
 * So one instruction is 16 lanes x 2 = 32 MACs, against an f32 FMA's 16, AND
 * an int32 accumulator holds 16 lanes like a float does -- so unlike the
 * int64 design the register tile does NOT have to halve. Both of the costs
 * measured in the int64 probe are removed at once.
 *
 * What it costs instead is RANGE. int32 overflows, so exactness needs the
 * accumulator flushed into int64 before it can. With int16 inputs of
 * magnitude <= M, a product is <= M^2 and T accumulations need T*M^2 < 2^31.
 * `micro_vnni_exact` flushes every FLUSH_T k-pairs, which is the design that
 * is actually exact; `micro_vnni_raw` omits the flush and exists only as the
 * upper bound, to separate the instruction's throughput from the flush cost.
 */
#include <stdint.h>
#include <immintrin.h>

#define MR 6
#define NR 64          /* 4 x <16 x int32> per row -> 24 zmm, same as f32 */
#define FLUSH_T 64     /* k-pairs between int32 -> int64 flushes */

/* Upper bound: no flush, so this overflows on real data. Throughput only. */
void micro_vnni_raw(const int32_t *restrict Ap, const int16_t *restrict Bp,
                    int32_t *restrict C, long ldc, long kpairs) {
    __m512i acc[MR][4];
    for (int i = 0; i < MR; ++i)
        for (int v = 0; v < 4; ++v)
            acc[i][v] = _mm512_loadu_si512((const void *)(C + i * ldc + v * 16));

    for (long p = 0; p < kpairs; ++p) {
        const __m512i *b = (const __m512i *)(Bp + p * NR * 2);
        __m512i b0 = _mm512_loadu_si512(b + 0);
        __m512i b1 = _mm512_loadu_si512(b + 1);
        __m512i b2 = _mm512_loadu_si512(b + 2);
        __m512i b3 = _mm512_loadu_si512(b + 3);
        for (int i = 0; i < MR; ++i) {
            __m512i a = _mm512_set1_epi32(Ap[p * MR + i]);
            acc[i][0] = _mm512_dpwssd_epi32(acc[i][0], a, b0);
            acc[i][1] = _mm512_dpwssd_epi32(acc[i][1], a, b1);
            acc[i][2] = _mm512_dpwssd_epi32(acc[i][2], a, b2);
            acc[i][3] = _mm512_dpwssd_epi32(acc[i][3], a, b3);
        }
    }
    for (int i = 0; i < MR; ++i)
        for (int v = 0; v < 4; ++v)
            _mm512_storeu_si512((void *)(C + i * ldc + v * 16), acc[i][v]);
}

/* The exact design: int32 in registers, widened into int64 every FLUSH_T
 * k-pairs so the narrow accumulator can never overflow. */
void micro_vnni_exact(const int32_t *restrict Ap, const int16_t *restrict Bp,
                      int64_t *restrict C, long ldc, long kpairs) {
    __m512i acc[MR][4];
    for (int i = 0; i < MR; ++i)
        for (int v = 0; v < 4; ++v) acc[i][v] = _mm512_setzero_si512();

    for (long p = 0; p < kpairs; ++p) {
        const __m512i *b = (const __m512i *)(Bp + p * NR * 2);
        __m512i b0 = _mm512_loadu_si512(b + 0);
        __m512i b1 = _mm512_loadu_si512(b + 1);
        __m512i b2 = _mm512_loadu_si512(b + 2);
        __m512i b3 = _mm512_loadu_si512(b + 3);
        for (int i = 0; i < MR; ++i) {
            __m512i a = _mm512_set1_epi32(Ap[p * MR + i]);
            acc[i][0] = _mm512_dpwssd_epi32(acc[i][0], a, b0);
            acc[i][1] = _mm512_dpwssd_epi32(acc[i][1], a, b1);
            acc[i][2] = _mm512_dpwssd_epi32(acc[i][2], a, b2);
            acc[i][3] = _mm512_dpwssd_epi32(acc[i][3], a, b3);
        }
        if ((p & (FLUSH_T - 1)) == (FLUSH_T - 1)) {
            for (int i = 0; i < MR; ++i)
                for (int v = 0; v < 4; ++v) {
                    __m256i lo = _mm512_extracti64x4_epi64(acc[i][v], 0);
                    __m256i hi = _mm512_extracti64x4_epi64(acc[i][v], 1);
                    int64_t *dst = C + i * ldc + v * 16;
                    _mm512_storeu_si512((void *)dst,
                        _mm512_add_epi64(_mm512_loadu_si512((const void *)dst),
                                         _mm512_cvtepi32_epi64(lo)));
                    _mm512_storeu_si512((void *)(dst + 8),
                        _mm512_add_epi64(_mm512_loadu_si512((const void *)(dst + 8)),
                                         _mm512_cvtepi32_epi64(hi)));
                    acc[i][v] = _mm512_setzero_si512();
                }
        }
    }
    for (int i = 0; i < MR; ++i)
        for (int v = 0; v < 4; ++v) {
            __m256i lo = _mm512_extracti64x4_epi64(acc[i][v], 0);
            __m256i hi = _mm512_extracti64x4_epi64(acc[i][v], 1);
            int64_t *dst = C + i * ldc + v * 16;
            _mm512_storeu_si512((void *)dst,
                _mm512_add_epi64(_mm512_loadu_si512((const void *)dst),
                                 _mm512_cvtepi32_epi64(lo)));
            _mm512_storeu_si512((void *)(dst + 8),
                _mm512_add_epi64(_mm512_loadu_si512((const void *)(dst + 8)),
                                 _mm512_cvtepi32_epi64(hi)));
        }
}
