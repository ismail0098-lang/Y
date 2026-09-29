/* Is the TAIL true for every estimate Lemma A admits?

   `newton_abi.cu` measures that the real e2 satisfies I-1 <= e2 <= I.  The
   validator will assume ONLY that -- it will not know which of the two e2 is --
   so the obligation it hands z3 quantifies over BOTH.  Nothing had checked
   whether the two conditional corrections are enough at e2 = I-1, and a
   solver timing out on a FALSE lemma is the most expensive way to find out.
   The domain (n,d,e2) is 2^64 and cannot be exhausted, so this does two
   things a sample cannot do alone:
     (a) EVERY d, at both admitted e2, over a structured set of n chosen where
         a quotient correction is decided -- the multiples of d and the ends;
     (b) a small set of d, at both admitted e2, over EVERY n.
   Failures are counted separately per e2 so "I-1 is not enough" is
   distinguishable from "the lowering is wrong". */
#include <cstdio>
#include <cstdint>
__device__ unsigned long long g_bad[2], g_seen[2], g_worst;

__device__ __forceinline__ bool check(unsigned n, unsigned d, unsigned e2) {
    unsigned q0 = (unsigned)(((unsigned long long)e2 * n) >> 32);
    unsigned r0 = d * (0u - q0) + n;
    bool c1 = r0 >= d; unsigned q1 = c1 ? q0+1 : q0, r1 = c1 ? r0-d : r0;
    bool c2 = r1 >= d; unsigned q2 = c2 ? q1+1 : q1, r2 = c2 ? r1-d : r1;
    return q2 == n/d && r2 == n%d;
}
/* (a) all d, structured n */
__global__ void sweepD(unsigned base, unsigned count) {
    unsigned long long b0=0,b1=0,s0=0,s1=0;
    for (unsigned long long i = blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
         i < count; i += (unsigned long long)gridDim.x*blockDim.x) {
        unsigned d = base + (unsigned)i;
        if (d == 0) continue;
        unsigned long long I = (1ULL<<32)/d;
        for (int w = 0; w < 2; w++) {
            unsigned long long Iw = I - w;               /* e2 = I, then I-1 */
            if (Iw > 0xFFFFFFFFULL) continue;            /* d==1, e2=I is not a u32 */
            unsigned e2 = (unsigned)Iw;
            unsigned ns[10];
            ns[0]=0u; ns[1]=1u; ns[2]=d-1u; ns[3]=d; ns[4]=d+1u;
            ns[5]=0xFFFFFFFFu; ns[6]=0xFFFFFFFFu-d; ns[7]=0x80000000u;
            ns[8]=(unsigned)(I*(unsigned long long)d);   /* the largest multiple of d */
            ns[9]=ns[8]-1u;
            for (int j = 0; j < 10; j++) {
                if (w) s1++; else s0++;
                if (!check(ns[j], d, e2)) { if (w) b1++; else b0++; }
            }
        }
    }
    if (b0) atomicAdd(&g_bad[0], b0);  if (b1) atomicAdd(&g_bad[1], b1);
    if (s0) atomicAdd(&g_seen[0], s0); if (s1) atomicAdd(&g_seen[1], s1);
}
/* (b) chosen d, EVERY n */
__global__ void sweepN(unsigned d, unsigned e2, int w, unsigned base, unsigned count) {
    unsigned long long b=0,s=0;
    for (unsigned long long i = blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
         i < count; i += (unsigned long long)gridDim.x*blockDim.x) {
        unsigned n = base + (unsigned)i;
        s++;
        if (!check(n, d, e2)) b++;
    }
    if (b) atomicAdd(&g_bad[w], b);
    if (s) atomicAdd(&g_seen[w], s);
}
int main() {
    unsigned long long z[2]={0,0};
    cudaMemcpyToSymbol(g_bad,z,16); cudaMemcpyToSymbol(g_seen,z,16);
    const unsigned long long TOTAL=1ULL<<32; const unsigned CHUNK=1u<<28;
    for (unsigned long long b=0;b<TOTAL;b+=CHUNK) {
        unsigned n=(unsigned)((TOTAL-b<CHUNK)?(TOTAL-b):CHUNK);
        sweepD<<<2048,256>>>((unsigned)b,n);
        if (cudaDeviceSynchronize()!=cudaSuccess) { printf("FAIL a\n"); return 1; }
    }
    unsigned long long bad[2],seen[2];
    cudaMemcpyFromSymbol(bad,g_bad,16); cudaMemcpyFromSymbol(seen,g_seen,16);
    printf("(a) all d, structured n:  e2=I   seen %llu bad %llu   |   e2=I-1 seen %llu bad %llu\n",
           seen[0],bad[0],seen[1],bad[1]);
    cudaMemcpyToSymbol(g_bad,z,16); cudaMemcpyToSymbol(g_seen,z,16);
    unsigned ds[12]={1u,2u,3u,7u,10u,1000u,65535u,65537u,1u<<25,0x7FFFFFFFu,0xd2470f5du,0xFFFFFFFFu};
    for (int k=0;k<12;k++) {
        unsigned d=ds[k]; unsigned long long I=(1ULL<<32)/d;
        for (int w=0;w<2;w++) {
            unsigned long long Iw=I-w; if (Iw>0xFFFFFFFFULL) continue;
            for (unsigned long long b=0;b<TOTAL;b+=CHUNK) {
                unsigned n=(unsigned)((TOTAL-b<CHUNK)?(TOTAL-b):CHUNK);
                sweepN<<<2048,256>>>(d,(unsigned)Iw,w,(unsigned)b,n);
                if (cudaDeviceSynchronize()!=cudaSuccess) { printf("FAIL b\n"); return 1; }
            }
        }
    }
    cudaMemcpyFromSymbol(bad,g_bad,16); cudaMemcpyFromSymbol(seen,g_seen,16);
    printf("(b) 12 chosen d, EVERY n: e2=I   seen %llu bad %llu   |   e2=I-1 seen %llu bad %llu\n",
           seen[0],bad[0],seen[1],bad[1]);
    return 0;
}
