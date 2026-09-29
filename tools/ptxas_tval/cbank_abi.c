/* Launch a kernel with six DISTINCT extents and read back what each special
   register saw.  Built and driven by cbank_abi.py, which supplies the .cubin
   and checks the answers against batch.py's own const-bank map. */
#include <stdio.h>
#include <cuda.h>
#define C(x) do{CUresult r=(x); if(r){const char*s;cuGetErrorString(r,&s);\
  printf("FAIL %s: %s\n",#x,s); return 1;}}while(0)
int main(int argc, char** argv){
  if(argc<2){ printf("usage: cbank_abi <cubin>\n"); return 2; }
  CUdevice d; CUcontext c; CUmodule m; CUfunction f; CUdeviceptr o;
  C(cuInit(0)); C(cuDeviceGet(&d,0)); C(cuCtxCreate(&c,NULL,0,d));
  C(cuModuleLoad(&m,argv[1])); C(cuModuleGetFunction(&f,m,"probe"));
  C(cuMemAlloc(&o,6*4)); C(cuMemsetD8(o,0xAB,6*4));
  void*a[]={&o};
  C(cuLaunchKernel(f, 3,5,7, 11,13,2, 0,0,a,0));
  C(cuCtxSynchronize());
  unsigned h[6]; C(cuMemcpyDtoH(h,o,sizeof h));
  for(int i=0;i<6;i++) printf("%u\n", h[i]);
  return 0;
}
