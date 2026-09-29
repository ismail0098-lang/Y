/* Launch a 6-buffer probe and print the three output words per lane.

   Generic on purpose: fpsem_abi.py drives TWO different probes through it
   (FSEL's semantics and FADD's commutativity), and a driver per question is a
   second copy of the launch plumbing.  Buffers 0..2 are filled from stdin,
   buffers 3..5 are poisoned with 0xAB and read back, so a lane the kernel
   never writes is visible rather than reading back as a plausible zero. */
#include <stdio.h>
#include <cuda.h>
#define C(x) do{CUresult r=(x); if(r){const char*s;cuGetErrorString(r,&s);\
  printf("FAIL %s: %s\n",#x,s); return 1;}}while(0)
#define N 32
int main(int argc, char** argv){
  if(argc<2){ printf("usage: fpsem_abi <cubin>\n"); return 2; }
  unsigned in[3][N];
  for(int i=0;i<N;i++)
    if(scanf("%u %u %u",&in[0][i],&in[1][i],&in[2][i])!=3){
      printf("FAIL: short input at lane %d\n", i); return 2; }
  CUdevice d; CUcontext c; CUmodule m; CUfunction f; CUdeviceptr b[6];
  C(cuInit(0)); C(cuDeviceGet(&d,0)); C(cuCtxCreate(&c,NULL,0,d));
  C(cuModuleLoad(&m,argv[1])); C(cuModuleGetFunction(&f,m,"probe"));
  for(int k=0;k<6;k++){
    C(cuMemAlloc(&b[k],N*4));
    if(k<3) C(cuMemcpyHtoD(b[k],in[k],N*4)); else C(cuMemsetD8(b[k],0xAB,N*4));
  }
  void* a[6]; for(int k=0;k<6;k++) a[k]=&b[k];
  C(cuLaunchKernel(f, 1,1,1, N,1,1, 0,0,a,0));
  C(cuCtxSynchronize());
  unsigned out[3][N];
  for(int k=0;k<3;k++) C(cuMemcpyDtoH(out[k],b[k+3],N*4));
  for(int i=0;i<N;i++) printf("%u %u %u\n", out[0][i], out[1][i], out[2][i]);
  return 0;
}
