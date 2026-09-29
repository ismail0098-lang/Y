/* EXHAUST Lemma A over every u32 divisor.

   `rcpwin_abi.cu` establishes the WINDOW the float estimate lands in.  The
   validator's argument then rests on a second claim -- that the Newton step
   the lowering performs turns that window into a bound ONE UNIT wide:

       e2 = e + HI(e * (-d))      should satisfy   I - B <= e2 <= I,  I = floor(2^32/d)

   Nothing had checked whether that is TRUE.  Asking z3 to prove a false lemma
   is the most expensive way to discover it is false, and the domain is finite,
   so it is exhausted here first.  B is MEASURED (the max deficit) rather than
   assumed, and the count of e2 ABOVE I is reported separately because that is
   the direction that would kill the model outright. */
#include <cstdio>
#include <cstdint>
__device__ unsigned long long g_maxdef, g_above, g_seen, g_e2max;

__device__ __forceinline__ unsigned est(unsigned d) {
    float f, r; unsigned b, e;
    asm("cvt.rp.f32.u32 %0, %1;"   : "=f"(f) : "r"(d));
    asm("rcp.approx.f32 %0, %1;"   : "=f"(r) : "f"(f));
    asm("mov.b32 %0, %1;"          : "=r"(b) : "f"(r));
    b += 0x0ffffffeu;
    asm("mov.b32 %0, %1;"          : "=f"(f) : "r"(b));
    asm("cvt.rzi.ftz.u32.f32 %0, %1;" : "=r"(e) : "f"(f));
    return e;
}
__global__ void sweep(unsigned base, unsigned count) {
    unsigned long long md = 0, ab = 0, sn = 0;
    for (unsigned long long i = blockIdx.x*(unsigned long long)blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long)gridDim.x*blockDim.x) {
        unsigned d = base + (unsigned)i;
        if (d == 0) continue;
        unsigned e = est(d);
        /* exactly the emitted sequence, in u32 with wrap, as the SASS performs it */
        unsigned t  = (0u - d) * e;                       /* IMAD.MOV / IMAD  */
        unsigned hi = (unsigned)(((unsigned long long)e * t) >> 32);  /* IMAD.HI.U32 */
        unsigned e2 = e + hi;                             /* the +Rc of the 64-bit pair */
        unsigned long long I = (1ULL << 32) / d;
        sn++;
        if ((unsigned long long)e2 > I) { ab++; continue; }
        unsigned long long def = I - e2;
        if (def > md) md = def;
    }
    if (md) atomicMax(&g_maxdef, md);
    if (ab) atomicAdd(&g_above, ab);
    if (sn) atomicAdd(&g_seen, sn);
}
int main() {
    unsigned long long z = 0;
    cudaMemcpyToSymbol(g_maxdef,&z,8); cudaMemcpyToSymbol(g_above,&z,8); cudaMemcpyToSymbol(g_seen,&z,8);
    const unsigned long long TOTAL = 1ULL<<32; const unsigned CHUNK = 1u<<28;
    for (unsigned long long b = 0; b < TOTAL; b += CHUNK) {
        unsigned n = (unsigned)((TOTAL-b < CHUNK) ? (TOTAL-b) : CHUNK);
        sweep<<<2048,256>>>((unsigned)b, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("FAIL launch: %s\n", cudaGetErrorString(e)); return 1; }
    }
    unsigned long long md, ab, sn;
    cudaMemcpyFromSymbol(&md,g_maxdef,8); cudaMemcpyFromSymbol(&ab,g_above,8); cudaMemcpyFromSymbol(&sn,g_seen,8);
    printf("seen %llu  maxdeficit(I-e2) %llu  above(e2>I) %llu\n", sn, md, ab);
    return 0;
}
