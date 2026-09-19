/* SHF.L.U32 without .HI, the form ptxas lowers `shl.b32` by a masked amount to.
   sassexec models it as the LOW word of the funnel {Rc:Ra} << n for n < 32 and
   leaves n >= 32 unmodelled.  This asks the silicon about every n < 32 against
   a host shift; shf_abi.py checks the probe's SASS is that form. */
#include <cstdio>
__global__ void k(const unsigned *x, const unsigned *n, unsigned *o, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    /* the RAW amount, unmasked.  Masking with an immediate -- in C or in the
       asm -- lets the compiler prove n < 32 and emit SHF.L.W.U32, the wrap-mode
       form: another instruction, and caught by shf_abi.py's shape check.  The
       corpus masks with a constant held in a REGISTER and gets the clamp form. */
    if (i < N) { unsigned r; asm("shl.b32 %0, %1, %2;" : "=r"(r) : "r"(x[i]), "r"(n[i])); o[i] = r; }
}
int main() {
    const unsigned xs[] = {0u, 1u, 2u, 3u, 0x80000000u, 0x7fffffffu, 0xffffffffu, 0xdeadbeefu,
                           0x12345678u, 0x0f0f0f0fu, 0xaaaaaaaau, 0x00010001u};
    const int NX = sizeof(xs)/sizeof(xs[0]), N = NX * 64;
    unsigned hx[N], hn[N], ho[N];
    for (int i = 0; i < N; i++) { hx[i] = xs[i % NX]; hn[i] = (unsigned)(i / NX);          /* 0..63 */ }
    unsigned *dx, *dn, *dout; cudaMalloc(&dx, N*4); cudaMalloc(&dn, N*4); cudaMalloc(&dout, N*4);
    cudaMemcpy(dx, hx, N*4, cudaMemcpyHostToDevice); cudaMemcpy(dn, hn, N*4, cudaMemcpyHostToDevice);
    cudaMemset(dout, 0xab, N*4);
    k<<<(N+127)/128, 128>>>(dx, dn, dout, N);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("FAIL launch\n"); return 1; }
    cudaMemcpy(ho, dout, N*4, cudaMemcpyDeviceToHost);
    int bad = 0, amounts[32] = {0}, zero_above = 0, above = 0;
    for (int i = 0; i < N; i++) {
        unsigned m = hn[i];
        if (m < 32) { amounts[m] = 1; if (ho[i] != (hx[i] << m)) bad++; }
        else { above++; if (ho[i] == 0) zero_above++; }   /* reported, not modelled */
    }
    int cov = 0; for (int m = 0; m < 32; m++) cov += amounts[m];
    printf("seen %d amounts %d bad %d   (n>=32: %d of %d zero)\n", N - above, cov, bad, zero_above, above);
    return 0;
}
