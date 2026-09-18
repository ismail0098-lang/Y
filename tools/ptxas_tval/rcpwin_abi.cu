/* EXHAUST the reciprocal-estimate window over EVERY u32 divisor.

   The integer-division lowering ptxas emits starts with a float estimate

       E(d) = F2I.FTZ.U32.TRUNC.NTZ( MUFU.RCP( I2F.U32.RP(d) ) + 0x0ffffffe )

   and then corrects it with exact integer arithmetic.  `tools/ptxas_tval` can
   validate that tail without any float semantics AT ALL provided E(d) lies in
   a window -- so the window is the whole obligation, and it is a claim about
   every one of 2^32 divisors rather than about a sample.  A GPU can check all
   of them, so this EXHAUSTS rather than samples: the recorded rule is that a
   finite domain is exhausted, not modelled.

   The two inequalities are stated MULTIPLICATIVELY, because that is the form
   the validator will assume and a division in the assumption is a division z3
   has to reason about:

       E(d) * d <= 2^32                 (E never exceeds floor(2^32/d))
       (E(d) + 1 + K) * d >  2^32       (E is at most K below it)

   The max slack is also reported directly, which needs the division, so that
   K is MEASURED rather than guessed and a later tightening is checkable.

   The inline asm is the point of the file: the four instructions must be the
   ones the corpus kernel contains, and `rcpwin_abi.py` asserts that against
   the SASS rather than trusting that the compiler chose them. */
#include <cstdio>
#include <cstdint>

__device__ unsigned long long g_maxslack;   /* max over d of floor(2^32/d) - E(d) */
__device__ unsigned long long g_above;      /* count of d with E(d) > floor(2^32/d) */
__device__ unsigned long long g_seen;       /* liveness floor: divisors examined   */
__device__ unsigned long long g_badmul;     /* count failing the RELATIVE form      */
__device__ unsigned long long g_badslack;   /* count failing (I-e)^2 <= C*I         */
__device__ double g_maxsratio;              /* max of (I-e)^2 / I                   */
__device__ double g_maxratio;               /* max of rem^2 / (2^32 * d)            */     /* count failing the multiplicative form */

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

__global__ void sweep(unsigned base, unsigned count, unsigned K) {
    unsigned long long mx = 0, ab = 0, sn = 0, bm = 0, bs = 0;
    double mr = 0.0, ms = 0.0;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned d = base + (unsigned)i;
        if (d == 0) continue;                       /* div by zero is its own SASS path */
        unsigned long long e     = est(d);
        unsigned long long ideal = (1ULL << 32) / d;
        sn++;
        if (e > ideal) { ab++; continue; }          /* the case that would kill the model */
        unsigned long long slack = ideal - e;
        if (slack > mx) mx = slack;
        /* THE RELATIVE FORM.  An ABSOLUTE bound is the wrong shape: z3 refutes it at
           d = 2^25, where an estimate 96 below ideal is well inside 512 and still
           three quarters of the way down, and the Newton step cannot recover.  The
           Newton step squares the relative error, so the condition is slack^2 <~ ideal,
           which clears of division as  rem^2 <= 2^32 * d  with rem = 2^32 - e*d. */
        unsigned long long rem = (1ULL << 32) - e * d;
        /* 128-BIT ON BOTH SIDES.  rem can reach ~2^41 and 2^32*d ~2^64, so a u64
           comparison overflows -- and it does so in the LOUD direction: at C=2 the
           first version reported 368,450,712 violations where C=1 reported 767, and
           a looser bound cannot fail more often.  That is what caught it. */
        unsigned __int128 lhs = (unsigned __int128)rem * rem;
        unsigned __int128 rhs = (unsigned __int128)K * (1ULL << 32) * (unsigned long long)d;
        if (lhs > rhs) bm++;
        /* THE PREDICATE THE VALIDATOR ASSUMES, measured in the SAME form it is
           stated in.  The multiplicative twin above reaches 2^82 and needs 128
           bits; this one is the same claim divided through by d^2 and stays in
           64, which is what z3 has to reason about.  Measuring one form and
           assuming the other is the drift this directory exists to remove. */
        if (slack * slack > (unsigned long long)K * ideal) bs++;
        double r2 = (double)slack * (double)slack / (double)(ideal ? ideal : 1);
        if (r2 > ms) ms = r2;
        double r = (double)rem * (double)rem / (4294967296.0 * (double)d);
        if (r > mr) mr = r;
    }
    if (mx) atomicMax(&g_maxslack, mx);
    if (ab) atomicAdd(&g_above, ab);
    if (sn) atomicAdd(&g_seen, sn);
    if (bm) atomicAdd(&g_badmul, bm);
    if (bs) atomicAdd(&g_badslack, bs);
    if (ms > 0.0) atomicMax((unsigned long long*)&g_maxsratio, __double_as_longlong(ms));
    if (mr > 0.0) atomicMax((unsigned long long*)&g_maxratio, __double_as_longlong(mr));
}

int main(int argc, char** argv) {
    unsigned K = argc > 1 ? (unsigned)strtoul(argv[1], 0, 0) : 65536u;
    unsigned long long z = 0;
    cudaMemcpyToSymbol(g_maxslack, &z, 8); cudaMemcpyToSymbol(g_above, &z, 8);
    cudaMemcpyToSymbol(g_seen, &z, 8);     cudaMemcpyToSymbol(g_badmul, &z, 8);
    cudaMemcpyToSymbol(g_maxratio, &z, 8);
    cudaMemcpyToSymbol(g_badslack, &z, 8);
    cudaMemcpyToSymbol(g_maxsratio, &z, 8);
    const unsigned long long TOTAL = 1ULL << 32;
    const unsigned CHUNK = 1u << 28;
    for (unsigned long long b = 0; b < TOTAL; b += CHUNK) {
        unsigned n = (unsigned)((TOTAL - b < CHUNK) ? (TOTAL - b) : CHUNK);
        sweep<<<2048, 256>>>((unsigned)b, n, K);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("FAIL launch: %s\n", cudaGetErrorString(e)); return 1; }
    }
    unsigned long long mx, ab, sn, bm;
    cudaMemcpyFromSymbol(&mx, g_maxslack, 8); cudaMemcpyFromSymbol(&ab, g_above, 8);
    cudaMemcpyFromSymbol(&sn, g_seen, 8);     cudaMemcpyFromSymbol(&bm, g_badmul, 8);
    double mr; cudaMemcpyFromSymbol(&mr, g_maxratio, 8);
    unsigned long long bs; cudaMemcpyFromSymbol(&bs, g_badslack, 8);
    double ms; cudaMemcpyFromSymbol(&ms, g_maxsratio, 8);
    printf("seen %llu maxslack %llu above %llu badrel %llu maxratio %.9f badslack %llu maxsratio %.9f C %u\n", sn, mx, ab, bm, mr, bs, ms, K);
    return 0;
}
