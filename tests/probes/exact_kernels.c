#include <stdint.h>
#include <immintrin.h>
#define MR 6
#define NR_F32 64
#define NR_FIX 32

/* ---- f32 micro-kernel: Y's shipped shape -------------------------------- */
void micro_f32(const float *restrict Ap, const float *restrict Bp,
                      float *restrict C, long ldc, long kc) {
    __m512 acc[MR][4];
    for (int i = 0; i < MR; ++i)
        for (int v = 0; v < 4; ++v) acc[i][v] = _mm512_loadu_ps(C + i * ldc + v * 16);

    for (long p = 0; p < kc; ++p) {
        __m512 b0 = _mm512_loadu_ps(Bp + p * NR_F32 +  0);
        __m512 b1 = _mm512_loadu_ps(Bp + p * NR_F32 + 16);
        __m512 b2 = _mm512_loadu_ps(Bp + p * NR_F32 + 32);
        __m512 b3 = _mm512_loadu_ps(Bp + p * NR_F32 + 48);
        for (int i = 0; i < MR; ++i) {
            __m512 a = _mm512_set1_ps(Ap[p * MR + i]);
            acc[i][0] = _mm512_fmadd_ps(a, b0, acc[i][0]);
            acc[i][1] = _mm512_fmadd_ps(a, b1, acc[i][1]);
            acc[i][2] = _mm512_fmadd_ps(a, b2, acc[i][2]);
            acc[i][3] = _mm512_fmadd_ps(a, b3, acc[i][3]);
        }
    }
    for (int i = 0; i < MR; ++i)
        for (int v = 0; v < 4; ++v)
            _mm512_storeu_ps(C + i * ldc + v * 16, acc[i][v]);
}

/* ---- exact micro-kernel: Q16.16 in, Q32.32 int64 accumulators ------------
 * B is packed as int64 with the value in the low 32 bits, so _mm512_mul_epi32
 * (signed 32x32 -> 64 on the even 32-bit elements) consumes it directly with
 * no widening in the loop. Packing already exists in the real kernel, so
 * choosing this layout is free there too.
 */
void micro_fix(const int32_t *restrict Ap, const int64_t *restrict Bp,
                      int64_t *restrict C, long ldc, long kc) {
    __m512i acc[MR][4];
    for (int i = 0; i < MR; ++i)
        for (int v = 0; v < 4; ++v) acc[i][v] = _mm512_loadu_si512(C + i * ldc + v * 8);

    for (long p = 0; p < kc; ++p) {
        __m512i b0 = _mm512_loadu_si512(Bp + p * NR_FIX +  0);
        __m512i b1 = _mm512_loadu_si512(Bp + p * NR_FIX +  8);
        __m512i b2 = _mm512_loadu_si512(Bp + p * NR_FIX + 16);
        __m512i b3 = _mm512_loadu_si512(Bp + p * NR_FIX + 24);
        for (int i = 0; i < MR; ++i) {
            __m512i a = _mm512_set1_epi64((int64_t)Ap[p * MR + i]);
            acc[i][0] = _mm512_add_epi64(acc[i][0], _mm512_mul_epi32(a, b0));
            acc[i][1] = _mm512_add_epi64(acc[i][1], _mm512_mul_epi32(a, b1));
            acc[i][2] = _mm512_add_epi64(acc[i][2], _mm512_mul_epi32(a, b2));
            acc[i][3] = _mm512_add_epi64(acc[i][3], _mm512_mul_epi32(a, b3));
        }
    }
    for (int i = 0; i < MR; ++i)
        for (int v = 0; v < 4; ++v)
            _mm512_storeu_si512(C + i * ldc + v * 8, acc[i][v]);
}

