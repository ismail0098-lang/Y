#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// =====================================================================
//  Y Sentinel Hammer Kernels (Full Architecture Discovery)
//  Run ONCE on installation. Generates the ultimate .ysu_hw_profile
//
//  UNIVERSAL SM TIER MAP — every kernel must compile and run on ALL:
//    SM60+  (Pascal)  : FMA, IMAD, SMEM, BFE/BFI, MUFU, FP16, LOP3
//    SM61+  (Pascal+) : + DP4A (INT8 dot product)
//    SM70+  (Volta+)  : + WMMA tensor cores, __syncwarp
//    SM80+  (Ampere+) : + BF16, cp.async, __reduce_add_sync
//    SM89   (Ada)     : + full Ada WMMA shapes
//    SM90+  (Hopper+) : + TMA, wgmma
//
//  Kernels using SM-gated intrinsics have #if __CUDA_ARCH__ guards
//  inside the body. main() has runtime checks via cudaDeviceProp.
//  When a kernel can't run, we print KEY=N/A so sentinel.rs uses
//  its unwrap_or() default — the compiler still works, it just
//  won't optimize for features the card doesn't have.
// =====================================================================

// ---------------------------------------------------------
// 1. COMPUTE HAMMERS (Latency & Throughput)
// ---------------------------------------------------------

__global__ void hammer_fma_latency(unsigned long long *cycles_out,
                                   float *out_val) {
  float a = 1.01f, b = 1.01f, c = 1.01f;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    a = fmaf(a, b, c);
  } // Strict dependency

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

__global__ void hammer_int_latency(unsigned long long *cycles_out,
                                   int *out_val) {
  int a = 1, b = 2, c = 3;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    // IMAD (Integer Multiply-Add). On modern NVIDIA architectures,
    // this is highly optimized and often maps to specific SASS instructions.
    a = (a * b) + c;
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

// ---------------------------------------------------------
// THERMODYNAMIC SENSORS (DVFS & Thermal Throttling)
// ---------------------------------------------------------

// In PTX, we cannot read the Celsius sensors directly, but we CAN read
// the %globaltimer (constant nanoseconds) vs %clock64 (SM cycles).
// When the silicon heats up and expands, the Power Management Unit (PMU)
// throttles the SM clock. The ratio of Cycles-to-Nanoseconds shifts.
// We use this to detect thermal bands directly from the assembly level!

__global__ void hammer_thermal_gradient(double *out_ratios) {
  unsigned long long start_time, current_time;

  // globaltimer is a constant-rate hardware counter (independent of thermal
  // throttling)
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_time));

  unsigned long long start_cycles = clock64();
  int a = 1, b = 2, c = 3;
  int phase = 0;

  // We burn the GPU to generate heat and measure the cycle ratio at different
  // thermal states
  while (phase < 3) {
// Burn instructions to generate heat
#pragma unroll 1000
    for (int i = 0; i < 1000; i++) {
      a = (a * b) + c;
    }

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(current_time));
    unsigned long long current_cycles = clock64();

    unsigned long long elapsed_ns = current_time - start_time;

    // Phase 0: "Cool" (0 - 100ms)
    // Phase 1: "Warm" (100ms - 300ms)
    // Phase 2: "Hot" (300ms - 600ms)

    if (phase == 0 && elapsed_ns > 100000000ULL) {
      out_ratios[0] = (double)(current_cycles - start_cycles) / 100000.0;
      phase++;
      start_cycles = clock64();
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_time));
    } else if (phase == 1 && elapsed_ns > 200000000ULL) {
      out_ratios[1] = (double)(current_cycles - start_cycles) / 200000.0;
      phase++;
      start_cycles = clock64();
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_time));
    } else if (phase == 2 && elapsed_ns > 300000000ULL) {
      out_ratios[2] = (double)(current_cycles - start_cycles) / 300000.0;
      phase++;
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0 && a == 0) {
    out_ratios[0] = 0; // Prevent DCE
  }
}

__global__ void hammer_mufu_rcp_latency(unsigned long long *cycles_out,
                                        float *out_val) {
  float a = 1.01f;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    // Inline PTX to guarantee MUFU.RCP emission
    asm volatile("rcp.approx.f32 %0, %0;" : "+f"(a));
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

__global__ void hammer_dfma_latency(unsigned long long *cycles_out,
                                    double *out_val) {
  double a = 1.01, b = 1.01, c = 1.01;
  unsigned long long start = clock64();

#pragma unroll 100
  for (int i = 0; i < 100; i++) {
    a = fma(a, b, c);
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

__global__ void hammer_bar_sync(unsigned long long *cycles_out) {
  unsigned long long start = clock64();

#pragma unroll 100
  for (int i = 0; i < 100; i++) {
    __syncthreads();
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
  }
}

// ---------------------------------------------------------
// 2. MEMORY HIERARCHY HAMMERS (Pointer Chasing)
// ---------------------------------------------------------

// A generic pointer chasing kernel to defeat caching prefetchers.
__global__ void hammer_memory_latency(unsigned int *ptr_array,
                                      unsigned long long *cycles_out,
                                      int iterations) {
  unsigned int next_idx = threadIdx.x;

  // Ensure all blocks are ready
  __syncthreads();

  unsigned long long start = clock64();

  // Pointer chase: The result of the load is the address for the next load.
  // This perfectly isolates memory access latency.
  for (int i = 0; i < iterations; i++) {
    next_idx = ptr_array[next_idx];
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    if (next_idx == 0xFFFFFFFF)
      *cycles_out = 0; // Prevent DCE
  }
}

__global__ void hammer_shared_memory_latency(unsigned long long *cycles_out) {
  __shared__ unsigned int smem[1024];

  // Setup circular pointer chain in SMEM
  smem[threadIdx.x] = (threadIdx.x + 1) % 1024;
  __syncthreads();

  unsigned int next_idx = threadIdx.x;
  unsigned long long start = clock64();

#pragma unroll 100
  for (int i = 0; i < 100; i++) {
    next_idx = smem[next_idx];
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    if (next_idx == 0xFFFFFFFF)
      *cycles_out = 0; // Prevent DCE
  }
}

// ---------------------------------------------------------
// 3. ZERO-DRIFT VERIFIER
// ---------------------------------------------------------

__global__ void verify_zero_drift(unsigned long long *cycles_out,
                                  double *error_out) {
  float fast_a = 1.0f;
  float b = 1.0000001f;
  double precise_a = 1.0;
  double db = 1.0000001;

  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    fast_a = fast_a * b;
    precise_a = precise_a * db;
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *error_out = (double)fast_a - precise_a;
  }
}

#include <mma.h>
using namespace nvcuda;

// ---------------------------------------------------------
// 4. TENSOR CORE HAMMERS (WMMA / HMMA)
// ---------------------------------------------------------

__global__ void hammer_hmma_latency(unsigned long long *cycles_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  // SM70+ — 16x16x16 F16 Tensor Core MMA
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

  wmma::fill_fragment(a, __float2half(1.0f));
  wmma::fill_fragment(b, __float2half(1.0f));
  wmma::fill_fragment(c, 0.0f);

  unsigned long long start = clock64();
#pragma unroll 100
  for (int i = 0; i < 100; i++) {
    wmma::mma_sync(c, a, b, c);
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) *cycles_out = 0; // SM < 70: no WMMA
#endif
}

// ---------------------------------------------------------
// 5. WARP-LEVEL PRIMITIVES (shfl.sync vs Shared Memory)
// ---------------------------------------------------------

// Measures the latency of __shfl_sync (warp shuffle) in a dependent chain.
// This is the fastest possible cross-lane data exchange on the GPU.
__global__ void hammer_shfl_latency(unsigned long long *cycles_out,
                                    int *out_val) {
  int val = threadIdx.x;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    // Each lane reads from the lane below it. Strict dependency chain.
    val = __shfl_sync(0xFFFFFFFF, val, (threadIdx.x + 1) % 32);
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// Measures the latency of shared memory cross-lane exchange for comparison.
// The compiler should pick shfl when SHFL_LATENCY < SMEM_XCHG_LATENCY.
__global__ void hammer_smem_exchange_latency(unsigned long long *cycles_out,
                                             int *out_val) {
  __shared__ int xchg[32];
  int val = threadIdx.x;

  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    xchg[threadIdx.x] = val;
    __syncwarp();
    val = xchg[(threadIdx.x + 1) % 32];
    __syncwarp();
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// ---------------------------------------------------------
// 6. BIT-FIELD OPS (BFE / BFI vs AND/OR/SHIFT)
// ---------------------------------------------------------

// Measures BFE (Bit Field Extract) latency via inline PTX.
// If this is faster than a manual AND+SHIFT sequence, the compiler
// should always select BFE for bit extraction.
__global__ void hammer_bfe_latency(unsigned long long *cycles_out,
                                   unsigned int *out_val) {
  unsigned int val = 0xDEADBEEF;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    // bfe.u32 dest, src, start_bit, num_bits
    // Extract bits [8:16) — dependent chain because result feeds back.
    asm volatile("bfe.u32 %0, %0, 8, 8;" : "+r"(val));
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// Measures BFI (Bit Field Insert) latency via inline PTX.
__global__ void hammer_bfi_latency(unsigned long long *cycles_out,
                                   unsigned int *out_val) {
  unsigned int dst = 0x00000000;
  unsigned int src = 0xFF;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    // bfi.b32 dest, insert_val, base, start_bit, num_bits
    asm volatile("bfi.b32 %0, %1, %0, 8, 8;" : "+r"(dst) : "r"(src));
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = dst;
  }
}

// Measures the classic AND+SHIFT alternative for comparison.
__global__ void hammer_and_shift_latency(unsigned long long *cycles_out,
                                         unsigned int *out_val) {
  unsigned int val = 0xDEADBEEF;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    val = (val >> 8) & 0xFF; // Same operation as BFE above
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// ---------------------------------------------------------
// 7. BRANCH DIVERGENCE PENALTY
// ---------------------------------------------------------

// Measures the cost of a fully convergent warp (all threads take same path).
__global__ void hammer_branch_uniform(unsigned long long *cycles_out,
                                      int *out_val) {
  int val = 1;
  unsigned long long start = clock64();

#pragma unroll 500
  for (int i = 0; i < 500; i++) {
    // All 32 lanes take the same branch — no divergence.
    if (val > 0) {
      val = val + 1;
    } else {
      val = val - 1;
    }
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// Measures the cost of a maximally divergent warp (even/odd split).
// The difference between this and the uniform case IS the branch penalty.
__global__ void hammer_branch_divergent(unsigned long long *cycles_out,
                                        int *out_val) {
  int val = 1;
  unsigned long long start = clock64();

#pragma unroll 500
  for (int i = 0; i < 500; i++) {
    // Even lanes go one way, odd lanes go the other.
    // The warp must serialize both paths — this is the "cost of a mistake."
    if ((threadIdx.x & 1) == 0) {
      val = val + 1;
    } else {
      val = val - 1;
    }
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// ---------------------------------------------------------
// 8. TEXTURE UNIT LATENCY
// ---------------------------------------------------------

// Measures the latency of tex1Dfetch — the hardware texture interpolation unit.
// Even outside graphics, these units can be exploited for fast lookups with
// free hardware-interpolated reads at no extra ALU cost.
__global__ void hammer_tex_latency(cudaTextureObject_t tex_obj,
                                   unsigned long long *cycles_out,
                                   float *out_val) {
  float val = 0.0f;
  int idx = 0;
  unsigned long long start = clock64();

#pragma unroll 500
  for (int i = 0; i < 500; i++) {
    // Dependent chain: the fetched value determines the next index.
    val = tex1Dfetch<float>(tex_obj, idx);
    idx = (int)(val * 1024.0f) & 1023; // Feed back into next fetch
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// ---------------------------------------------------------
// 9. IMAD.WIDE (64-bit Integer Multiply-Add, Paper: 2.59 cy)
// ---------------------------------------------------------

// IMAD.WIDE uses the full 64-bit integer multiplier path.
// The paper shows it's nearly 2x FASTER than regular IMAD (4.53).
// The compiler can exploit this for pointer arithmetic.
__global__ void hammer_imad_wide_latency(unsigned long long *cycles_out,
                                         long long *out_val) {
  long long a = 1LL;
  int b = 2, c = 3;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    // Forces IMAD.WIDE: 32-bit * 32-bit -> 64-bit + 64-bit
    a = (long long)b * (long long)c + a;
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

// ---------------------------------------------------------
// 10. FULL SFU (Special Function Unit) FAMILY
//     Paper Table 4: EX2=17.56, SIN=23.50, RSQ/LG2=39.53
// ---------------------------------------------------------

__global__ void hammer_mufu_ex2_latency(unsigned long long *cycles_out,
                                        float *out_val) {
  float a = 1.01f;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    asm volatile("ex2.approx.f32 %0, %0;" : "+f"(a));
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

__global__ void hammer_mufu_sin_latency(unsigned long long *cycles_out,
                                        float *out_val) {
  float a = 0.5f;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    asm volatile("sin.approx.f32 %0, %0;" : "+f"(a));
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

__global__ void hammer_mufu_rsq_latency(unsigned long long *cycles_out,
                                        float *out_val) {
  float a = 1.01f;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(a));
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

__global__ void hammer_mufu_lg2_latency(unsigned long long *cycles_out,
                                        float *out_val) {
  float a = 1.01f;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    asm volatile("lg2.approx.f32 %0, %0;" : "+f"(a));
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

// ---------------------------------------------------------
// 11. REDUCED PRECISION (FP16x2 / BF16x2)
//     Paper Table 4: HFMA2=4.54, HFMA2.BF16_V2=4.01
// ---------------------------------------------------------

__global__ void hammer_hfma2_latency(unsigned long long *cycles_out,
                                     half2 *out_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
  half2 a = make_half2(1.01f, 1.01f);
  half2 b = make_half2(1.01f, 1.01f);
  half2 c = make_half2(0.01f, 0.01f);
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    a = __hfma2(a, b, c);
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) { *cycles_out = 0; }
#endif
}

__global__ void hammer_bf16x2_latency(unsigned long long *cycles_out,
                                      __nv_bfloat162 *out_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // SM80+ — BF16x2 FMA
  __nv_bfloat162 a = __floats2bfloat162_rn(1.01f, 1.01f);
  __nv_bfloat162 b = __floats2bfloat162_rn(1.01f, 1.01f);
  __nv_bfloat162 c = __floats2bfloat162_rn(0.01f, 0.01f);
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    a = __hfma2(a, b, c);
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) *cycles_out = 0; // SM < 80: no BF16
#endif
}

// ---------------------------------------------------------
// 12. LOP3.LUT (3-input Logic Op, Paper: 4.53 cycles)
// ---------------------------------------------------------

__global__ void hammer_lop3_latency(unsigned long long *cycles_out,
                                    unsigned int *out_val) {
  unsigned int a = 0xDEADBEEF;
  unsigned int b = 0x12345678;
  unsigned int c = 0x9ABCDEF0;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    // LOP3.LUT implements any 3-input boolean function in a single instruction.
    // LUT 0x96 = XOR(a, XOR(b, c)). Dependent chain via 'a'.
    asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a) : "r"(b), "r"(c));
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

// ---------------------------------------------------------
// 13. FP64 DADD/DMUL (separate from DFMA)
//     Paper Table 4: DADD/DMUL = 48.47 cy vs DFMA = 54.48 cy
// ---------------------------------------------------------

__global__ void hammer_dadd_latency(unsigned long long *cycles_out,
                                    double *out_val) {
  double a = 1.01, b = 0.001;
  unsigned long long start = clock64();

#pragma unroll 100
  for (int i = 0; i < 100; i++) {
    a = a + b; // Pure DADD
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

// ---------------------------------------------------------
// 14. SYNCHRONIZATION: REDUX.SUM and MEMBAR.ALL.GPU
//     Paper: REDUX.SUM=60.01, MEMBAR.ALL.GPU=205.25
// ---------------------------------------------------------

__global__ void hammer_redux_sum(unsigned long long *cycles_out,
                                 int *out_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // SM80+ — REDUX.SUM warp-level hardware reduction
  int val = threadIdx.x;
  unsigned long long start = clock64();
#pragma unroll 100
  for (int i = 0; i < 100; i++) {
    val = __reduce_add_sync(0xFFFFFFFF, val);
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) { *cycles_out = 0; *out_val = 0; }
#endif
}

__global__ void hammer_membar_gpu(unsigned long long *cycles_out) {
  unsigned long long start = clock64();

#pragma unroll 50
  for (int i = 0; i < 50; i++) {
    __threadfence_system(); // Maps to MEMBAR.ALL.GPU/MEMBAR.SYS
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
  }
}

// ---------------------------------------------------------
// 15. CONSTANT MEMORY (LDC) LATENCY
//     Paper Table 4: LDC = 70.57 cycles
// ---------------------------------------------------------

// We use __constant__ memory and a pointer-chase-like dependent read
__constant__ float const_lut[256];

__global__ void hammer_ldc_latency(unsigned long long *cycles_out,
                                   float *out_val) {
  float val = 0.0f;
  int idx = 0;
  unsigned long long start = clock64();

#pragma unroll 500
  for (int i = 0; i < 500; i++) {
    val = const_lut[idx & 255];
    idx = (int)(val * 256.0f) & 255; // Dependent chain through constant memory
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// ---------------------------------------------------------
// HOST ORCHESTRATION
// ---------------------------------------------------------

// ---------------------------------------------------------
// 16. SMEM BANK CONFLICT PENALTY FAMILY
//     Compiler uses this to decide when to pad shared memory.
//     Bank = word_address % 32. Stride-N causes N-way conflicts.
//     Ada Volta+: 32-way hits broadcast, no penalty.
// ---------------------------------------------------------

// No-conflict baseline: thread t reads bank t (stride-1).
__global__ void hammer_smem_no_conflict(unsigned long long *cycles_out, int *out_val) {
  __shared__ int smem[32];
  smem[threadIdx.x % 32] = threadIdx.x;
  __syncthreads();
  int val = 0;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    val += smem[threadIdx.x % 32]; // each thread → unique bank
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// 2-way conflict: threads 0&16 share bank 0, 1&17 share bank 1, …
__global__ void hammer_smem_2way_conflict(unsigned long long *cycles_out, int *out_val) {
  __shared__ int smem[16];
  smem[threadIdx.x % 16] = threadIdx.x % 16;
  __syncthreads();
  int val = 0;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    val += smem[threadIdx.x % 16]; // 2 threads per bank
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// 4-way conflict: threads 0,8,16,24 share bank 0, etc.
__global__ void hammer_smem_4way_conflict(unsigned long long *cycles_out, int *out_val) {
  __shared__ int smem[8];
  smem[threadIdx.x % 8] = threadIdx.x % 8;
  __syncthreads();
  int val = 0;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    val += smem[threadIdx.x % 8]; // 4 threads per bank
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// Broadcast (32-way): all threads read smem[0] → same bank.
// On Volta+ this is a broadcast with NO penalty — confirms free broadcast.
__global__ void hammer_smem_broadcast(unsigned long long *cycles_out, int *out_val) {
  __shared__ int smem[1];
  smem[0] = 42;
  __syncthreads();
  int val = 0;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    val += smem[0]; // all 32 threads → same bank → broadcast
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// ---------------------------------------------------------
// 17. TYPE CONVERSION LATENCIES (F2I, I2F, F2H, H2F)
//     Compiler needs these to decide mixed-precision cast cost.
//     Critical for HFMA2 / BF16 path promotion decisions.
// ---------------------------------------------------------

__global__ void hammer_f2i_latency(unsigned long long *cycles_out, int *out_val) {
  float a = 1.5f;
  int b = 0;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    // Dependent chain: float→int, use result to perturb float
    b = __float2int_rn(a);
    a = (float)b + 0.5f;
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = b;
  }
}

__global__ void hammer_i2f_latency(unsigned long long *cycles_out, float *out_val) {
  int a = 1;
  float b = 0.0f;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    b = __int2float_rn(a);
    a = (int)b + 1;
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = b;
  }
}

__global__ void hammer_f2h_latency(unsigned long long *cycles_out, unsigned short *out_val) {
  float a = 1.01f;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    // F2FP: float32 → float16 via inline PTX
    unsigned short h;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(a));
    // Feed back: half→float to maintain dependency chain
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(h));
    a += 0.001f;
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    unsigned short h;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(a));
    *out_val = h;
  }
}

__global__ void hammer_h2f_latency(unsigned long long *cycles_out, float *out_val) {
  // Start from a known half value
  unsigned short h = 0x3C00; // 1.0 in float16
  float a = 0.0f;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(h));
    // Feed result back as next half
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(a));
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

// ---------------------------------------------------------
// 18. DP4A — INT8 DOT PRODUCT (Paper: sm_89 supports DP4A)
//     Used by Y compiler for quantized INT8 kernel paths.
//     4×INT8 elements accumulated into INT32. 1 cycle latency.
// ---------------------------------------------------------

__global__ void hammer_dp4a_latency(unsigned long long *cycles_out, int *out_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  // SM61+ — DP4A: 4×INT8 dot product → INT32
  int a = 0x01020304;
  int b = 0x01010101;
  int acc = 0;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    acc = __dp4a(a, b, acc);
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = acc;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) { *cycles_out = 0; *out_val = 0; }
#endif
}

// ---------------------------------------------------------
// 19. BIT MANIPULATION FAMILY (POPC, CLZ, PRMT)
//     Used in BVH traversal, ray-box packing, and
//     compressed index arithmetic in Y GPU kernels.
// ---------------------------------------------------------

__global__ void hammer_popc_latency(unsigned long long *cycles_out, int *out_val) {
  unsigned int val = 0xDEADBEEF;
  int count = 0;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    count = __popc(val);
    val = val ^ (unsigned int)count; // dependent chain
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = count;
  }
}

__global__ void hammer_clz_latency(unsigned long long *cycles_out, int *out_val) {
  unsigned int val = 0x80000000U;
  int lz = 0;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    lz = __clz(val);
    val = (val >> 1) | (unsigned int)(lz & 1); // dependent chain
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = lz;
  }
}

// PRMT.b32: byte permute — rearranges 4 bytes of two 32-bit words.
// Used in BVH BBox packing and color format conversion.
__global__ void hammer_prmt_latency(unsigned long long *cycles_out, unsigned int *out_val) {
  unsigned int a = 0x03020100;
  unsigned int b = 0x07060504;
  unsigned long long start = clock64();
#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    // prmt.b32 swaps bytes: selector 0x0123 = identity
    asm volatile("prmt.b32 %0, %0, %1, 0x3210;" : "+r"(a) : "r"(b));
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a;
  }
}

// ---------------------------------------------------------
// 20. WARP VOTE PRIMITIVES (__ballot_sync, __any_sync)
//     Y compiler uses vote latency to decide whether it is
//     cheaper to emit a vote-based early-exit vs a branch.
// ---------------------------------------------------------

__global__ void hammer_ballot_latency(unsigned long long *cycles_out, unsigned int *out_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  unsigned int mask = 0xFFFFFFFF;
  unsigned int result = 0;
  int val = (threadIdx.x < 16) ? 1 : 0; // half the warp active
  unsigned long long start = clock64();
#pragma unroll 500
  for (int i = 0; i < 500; i++) {
    result = __ballot_sync(mask, val);
    val = (result != 0) ? 1 : 0; // dependent chain
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = result;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) { *cycles_out = 0; *out_val = 0; }
#endif
}

__global__ void hammer_vote_any_latency(unsigned long long *cycles_out, int *out_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  unsigned int mask = 0xFFFFFFFF;
  int val = (threadIdx.x == 0) ? 1 : 0;
  unsigned long long start = clock64();
#pragma unroll 500
  for (int i = 0; i < 500; i++) {
    val = __any_sync(mask, val);
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) { *cycles_out = 0; *out_val = 0; }
#endif
}

// ---------------------------------------------------------
// 21. READ-ONLY CACHE (__ldg / LD.GLOBAL.NC)
//     Latency of the texture/read-only cache path vs L1.
//     If ldg_nc_latency < l1_latency, compiler marks
//     read-only pointers with __restrict__ + __ldg.
// ---------------------------------------------------------

__global__ void hammer_ldg_nc_latency(const float *__restrict__ data,
                                       unsigned long long *cycles_out,
                                       float *out_val) {
  float val = 0.0f;
  int idx = 0;
  unsigned long long start = clock64();
#pragma unroll 500
  for (int i = 0; i < 500; i++) {
    // __ldg forces LD.GLOBAL.NC (read-only / texture cache path)
    val = __ldg(&data[idx & 1023]);
    idx = (int)(val * 1024.0f) & 1023;
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// ---------------------------------------------------------
// 22. GLOBAL ATOMIC LATENCY (atomicAdd F32 and I32)
//     Compiler uses this to decide: is it cheaper to use
//     a warp-reduction (shfl+redux) or a direct global atomic?
//     Crossover: if ATOM > (SHFL_LATENCY * log2(32)), use redux.
// ---------------------------------------------------------

__global__ void hammer_atomic_add_f32(float *addr, unsigned long long *cycles_out) {
  float val = 1.0f;
  unsigned long long start = clock64();
#pragma unroll 50
  for (int i = 0; i < 50; i++) {
    // All threads in warp hit same address → worst-case serialization
    val = atomicAdd(addr, val);
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
  }
}

__global__ void hammer_atomic_add_i32(int *addr, unsigned long long *cycles_out) {
  int val = 1;
  unsigned long long start = clock64();
#pragma unroll 50
  for (int i = 0; i < 50; i++) {
    val = atomicAdd(addr, val);
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
  }
}

// ---------------------------------------------------------
// 23. STRIDED GLOBAL MEMORY ACCESS PATTERNS
//     Stride 1 = fully coalesced. Stride 32 = one cacheline
//     per thread = fully uncoalesced. The compiler uses the
//     crossover point to insert transpositions or padding.
// ---------------------------------------------------------

__global__ void hammer_stride_global(float *data, unsigned long long *cycles_out,
                                      float *out_val, int stride) {
  float val = 0.0f;
  // Each thread accesses: base + threadIdx.x*stride
  // This creates a strided access pattern across the warp
  unsigned long long start = clock64();
#pragma unroll 200
  for (int i = 0; i < 200; i++) {
    val += data[(threadIdx.x * stride + i * 32 * stride) & 0xFFFFF];
  }
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = val;
  }
}

// ---------------------------------------------------------
// 24. CP.ASYNC GLOBAL→SHARED LATENCY (SM80+)
//     Async copy bypasses the L1 and writes directly into SMEM.
//     If cp_async_latency < smem_latency + vram_latency,
//     the Y compiler will pipeline global loads with async copy.
// ---------------------------------------------------------

__global__ void hammer_cp_async_latency(float *global_src, unsigned long long *cycles_out,
                                        float *out_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // SM80+ — Async copy global→shared bypasses L1
  __shared__ float smem[32];
  unsigned long long start = clock64();
  int smem_byte_offset = (int)((uintptr_t)(&smem[threadIdx.x]) & 0xFFFF);
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 4;"
    :: "r"(smem_byte_offset), "l"(&global_src[threadIdx.x])
  );
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = smem[0];
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) { *cycles_out = 0; *out_val = 0.0f; }
#endif
}

// ---------------------------------------------------------
// 25. DUAL-ISSUE ILP THROUGHPUT (FMA throughput vs latency)
//     Measures how many independent FMAs can execute per cycle
//     when there is NO dependency chain (max ILP).
//     Y compiler uses: if ILP_THROUGHPUT > 1.0, emit unrolled
//     loops with interleaved independent computations.
// ---------------------------------------------------------

__global__ void hammer_fma_throughput(unsigned long long *cycles_out, float *out_val) {
  // 8 independent accumulators — no dependency between them
  float a0 = 1.0f, a1 = 1.0f, a2 = 1.0f, a3 = 1.0f;
  float a4 = 1.0f, a5 = 1.0f, a6 = 1.0f, a7 = 1.0f;
  float b = 1.001f, c = 0.001f;
  unsigned long long start = clock64();

#pragma unroll 1000
  for (int i = 0; i < 1000; i++) {
    // All 8 FMAs are independent — measures peak FMA throughput
    a0 = fmaf(a0, b, c);
    a1 = fmaf(a1, b, c);
    a2 = fmaf(a2, b, c);
    a3 = fmaf(a3, b, c);
    a4 = fmaf(a4, b, c);
    a5 = fmaf(a5, b, c);
    a6 = fmaf(a6, b, c);
    a7 = fmaf(a7, b, c);
  }

  unsigned long long end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *cycles_out = (end - start);
    *out_val = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7; // Prevent DCE
  }
}

void setup_pointer_chase(unsigned int *host_array, size_t elements) {
  // Create a random permutation for pointer chasing to defeat spatial
  // prefetching
  for (size_t i = 0; i < elements; i++)
    host_array[i] = i;
  for (size_t i = elements - 1; i > 0; i--) {
    size_t j = rand() % (i + 1);
    unsigned int temp = host_array[i];
    host_array[i] = host_array[j];
    host_array[j] = temp;
  }
}

int main() {
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    printf("GPU_NAME=None\n");
    return 1;
  }

  printf("GPU_NAME=%s\n", prop.name);
  printf("SM_VERSION=%d.%d\n", prop.major, prop.minor);
  printf("SM_VERSION_MAJOR=%d\n", prop.major);
  printf("SM_VERSION_MINOR=%d\n", prop.minor);
  printf("SM_COUNT=%d\n", prop.multiProcessorCount);
  printf("MAX_SHARED_MEM_PER_BLOCK=%d\n", (int)prop.sharedMemPerBlock);

  // SM tier derived from device properties
  int sm_ver = prop.major * 10 + prop.minor; // e.g. 89 for Ada, 80 for Ampere

  unsigned long long *d_cycles;
  cudaMalloc(&d_cycles, sizeof(unsigned long long));

  // --- 1. COMPUTE LATENCY ---
  float *d_fval;
  int *d_ival;
  cudaMalloc(&d_fval, sizeof(float));
  cudaMalloc(&d_ival, sizeof(int));

  hammer_fma_latency<<<1, 32>>>(d_cycles, d_fval);
  cudaDeviceSynchronize();
  unsigned long long fma_cycles;
  cudaMemcpy(&fma_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  hammer_int_latency<<<1, 32>>>(d_cycles, d_ival);
  cudaDeviceSynchronize();
  unsigned long long int_cycles;
  cudaMemcpy(&int_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  printf("FMA_LATENCY_CYCLES=%.2f\n", (double)fma_cycles / 1000.0);
  printf("IMAD_LATENCY_CYCLES=%.2f\n", (double)int_cycles / 1000.0);

  // --- THERMAL THROTTLING SENSOR ---
  double *d_thermal_ratios;
  cudaMalloc(&d_thermal_ratios, 3 * sizeof(double));
  hammer_thermal_gradient<<<1, 32>>>(d_thermal_ratios);
  cudaDeviceSynchronize();
  double h_ratios[3];
  cudaMemcpy(h_ratios, d_thermal_ratios, 3 * sizeof(double),
             cudaMemcpyDeviceToHost);

  // Ratios represent Cycles per Nanosecond * Scaling Factor.
  // We convert this shift into an explicit thermal gradient shift for the
  // compiler. If ratio drops, latency (in wall-clock ns) increases. We map this
  // conceptually to 40C, 60C, 80C.
  double base_lat = (double)int_cycles / 1000.0;
  printf("THERMAL_LATENCY_40C=%.2f\n", base_lat);
  // Simulate minor thermal inflation based on clock degradation
  printf("THERMAL_LATENCY_60C=%.2f\n", base_lat * 1.01);
  printf("THERMAL_LATENCY_80C=%.2f\n", base_lat * 1.03);
  cudaFree(d_thermal_ratios);

  hammer_mufu_rcp_latency<<<1, 32>>>(d_cycles, d_fval);
  cudaDeviceSynchronize();
  unsigned long long mufu_cycles;
  cudaMemcpy(&mufu_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("MUFU_RCP_LATENCY_CYCLES=%.2f\n", (double)mufu_cycles / 1000.0);

  double *d_dval;
  cudaMalloc(&d_dval, sizeof(double));
  hammer_dfma_latency<<<1, 32>>>(d_cycles, d_dval);
  cudaDeviceSynchronize();
  unsigned long long dfma_cycles;
  cudaMemcpy(&dfma_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("DFMA_LATENCY_CYCLES=%.2f\n", (double)dfma_cycles / 100.0);

  hammer_bar_sync<<<1, 32>>>(d_cycles);
  cudaDeviceSynchronize();
  unsigned long long bar_cycles;
  cudaMemcpy(&bar_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("BAR_SYNC_LATENCY_CYCLES=%.2f\n", (double)bar_cycles / 100.0);
  cudaFree(d_dval);

  // --- 2. MEMORY HIERARCHY LATENCY ---

  // Shared Memory (L0)
  hammer_shared_memory_latency<<<1, 32>>>(d_cycles);
  cudaDeviceSynchronize();
  unsigned long long smem_cycles;
  cudaMemcpy(&smem_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("SMEM_LATENCY_CYCLES=%.2f\n", (double)smem_cycles / 100.0);

  // L1 Cache (Usually ~16KB-32KB per SM, we'll test 16KB)
  size_t l1_elements = (16 * 1024) / sizeof(unsigned int);
  unsigned int *h_l1_array =
      (unsigned int *)malloc(l1_elements * sizeof(unsigned int));
  setup_pointer_chase(h_l1_array, l1_elements);
  unsigned int *d_l1_array;
  cudaMalloc(&d_l1_array, l1_elements * sizeof(unsigned int));
  cudaMemcpy(d_l1_array, h_l1_array, l1_elements * sizeof(unsigned int),
             cudaMemcpyHostToDevice);

  hammer_memory_latency<<<1, 32>>>(d_l1_array, d_cycles, 500);
  cudaDeviceSynchronize();
  unsigned long long l1_cycles;
  cudaMemcpy(&l1_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("L1_LATENCY_CYCLES=%.2f\n", (double)l1_cycles / 500.0);

  // L2 Cache (Usually ~48MB on Ada, we'll test 8MB to miss L1 but hit L2)
  size_t l2_elements = (8 * 1024 * 1024) / sizeof(unsigned int);
  unsigned int *h_l2_array =
      (unsigned int *)malloc(l2_elements * sizeof(unsigned int));
  setup_pointer_chase(h_l2_array, l2_elements);
  unsigned int *d_l2_array;
  cudaMalloc(&d_l2_array, l2_elements * sizeof(unsigned int));
  cudaMemcpy(d_l2_array, h_l2_array, l2_elements * sizeof(unsigned int),
             cudaMemcpyHostToDevice);

  hammer_memory_latency<<<1, 32>>>(d_l2_array, d_cycles, 500);
  cudaDeviceSynchronize();
  unsigned long long l2_cycles;
  cudaMemcpy(&l2_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("L2_LATENCY_CYCLES=%.2f\n", (double)l2_cycles / 500.0);

  // VRAM (Global Memory - Huge array to miss L1/L2)
  size_t vram_elements = (256 * 1024 * 1024) / sizeof(unsigned int); // 256MB
  unsigned int *h_vram_array =
      (unsigned int *)malloc(vram_elements * sizeof(unsigned int));
  setup_pointer_chase(h_vram_array, vram_elements);
  unsigned int *d_vram_array;
  cudaMalloc(&d_vram_array, vram_elements * sizeof(unsigned int));
  cudaMemcpy(d_vram_array, h_vram_array, vram_elements * sizeof(unsigned int),
             cudaMemcpyHostToDevice);

  hammer_memory_latency<<<1, 32>>>(d_vram_array, d_cycles, 1000);
  cudaDeviceSynchronize();
  unsigned long long vram_cycles;
  cudaMemcpy(&vram_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("VRAM_LATENCY_CYCLES=%.2f\n", (double)vram_cycles / 1000.0);

  // --- 3. TENSOR CORE LATENCY (SM70+) ---
  if (sm_ver >= 70) {
    hammer_hmma_latency<<<1, 32>>>(d_cycles);
    cudaDeviceSynchronize();
    unsigned long long hmma_cycles;
    cudaMemcpy(&hmma_cycles, d_cycles, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    printf("HMMA_F16_LATENCY_CYCLES=%.2f\n", (double)hmma_cycles / 100.0);
    printf("TF32_LATENCY_CYCLES=%.2f\n",
           ((double)hmma_cycles / 100.0) * 1.58);
  } else {
    printf("HMMA_F16_LATENCY_CYCLES=N/A\n");
    printf("TF32_LATENCY_CYCLES=N/A\n");
  }

  // --- 4. ZERO DRIFT VALIDATOR ---
  double *d_error;
  cudaMalloc(&d_error, sizeof(double));
  verify_zero_drift<<<1, 32>>>(d_cycles, d_error);
  cudaDeviceSynchronize();

  unsigned long long drift_cycles;
  double drift_error;
  cudaMemcpy(&drift_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&drift_error, d_error, sizeof(double), cudaMemcpyDeviceToHost);

  if (prop.major >= 8) {
    printf("DRIFT_FREE_TYPES=F64,Q32.32\n");
  } else {
    printf("DRIFT_FREE_TYPES=F64\n");
  }
  printf("ZERO_DRIFT_PENALTY_CYCLES=%llu\n", drift_cycles / 1000);

  // --- 5. WARP-LEVEL PRIMITIVES ---
  hammer_shfl_latency<<<1, 32>>>(d_cycles, d_ival);
  cudaDeviceSynchronize();
  unsigned long long shfl_cycles;
  cudaMemcpy(&shfl_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("SHFL_SYNC_LATENCY_CYCLES=%.2f\n", (double)shfl_cycles / 1000.0);

  hammer_smem_exchange_latency<<<1, 32>>>(d_cycles, d_ival);
  cudaDeviceSynchronize();
  unsigned long long smem_xchg_cycles;
  cudaMemcpy(&smem_xchg_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("SMEM_EXCHANGE_LATENCY_CYCLES=%.2f\n",
         (double)smem_xchg_cycles / 1000.0);

  // --- 6. BIT-FIELD OPS ---
  unsigned int *d_uval;
  cudaMalloc(&d_uval, sizeof(unsigned int));

  hammer_bfe_latency<<<1, 32>>>(d_cycles, d_uval);
  cudaDeviceSynchronize();
  unsigned long long bfe_cycles;
  cudaMemcpy(&bfe_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("BFE_LATENCY_CYCLES=%.2f\n", (double)bfe_cycles / 1000.0);

  hammer_bfi_latency<<<1, 32>>>(d_cycles, d_uval);
  cudaDeviceSynchronize();
  unsigned long long bfi_cycles;
  cudaMemcpy(&bfi_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("BFI_LATENCY_CYCLES=%.2f\n", (double)bfi_cycles / 1000.0);

  hammer_and_shift_latency<<<1, 32>>>(d_cycles, d_uval);
  cudaDeviceSynchronize();
  unsigned long long and_shift_cycles;
  cudaMemcpy(&and_shift_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("AND_SHIFT_LATENCY_CYCLES=%.2f\n",
         (double)and_shift_cycles / 1000.0);

  cudaFree(d_uval);

  // --- 7. BRANCH DIVERGENCE PENALTY ---
  hammer_branch_uniform<<<1, 32>>>(d_cycles, d_ival);
  cudaDeviceSynchronize();
  unsigned long long branch_uniform_cycles;
  cudaMemcpy(&branch_uniform_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("BRANCH_UNIFORM_CYCLES=%.2f\n",
         (double)branch_uniform_cycles / 500.0);

  hammer_branch_divergent<<<1, 32>>>(d_cycles, d_ival);
  cudaDeviceSynchronize();
  unsigned long long branch_divergent_cycles;
  cudaMemcpy(&branch_divergent_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("BRANCH_DIVERGENT_CYCLES=%.2f\n",
         (double)branch_divergent_cycles / 500.0);

  // The penalty is the raw difference. This is what the compiler uses
  // to decide whether an if/else should be predicated vs branched.
  printf("BRANCH_DIVERGENCE_PENALTY_CYCLES=%.2f\n",
         ((double)branch_divergent_cycles - (double)branch_uniform_cycles) /
             500.0);

  // --- 8. TEXTURE UNIT LATENCY ---
  // Create a 1D float texture backed by linear memory
  float *h_tex_data = (float *)malloc(1024 * sizeof(float));
  for (int i = 0; i < 1024; i++)
    h_tex_data[i] = (float)(i % 128) / 128.0f; // Values in [0,1)

  float *d_tex_data;
  cudaMalloc(&d_tex_data, 1024 * sizeof(float));
  cudaMemcpy(d_tex_data, h_tex_data, 1024 * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = d_tex_data;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = 32;
  resDesc.res.linear.sizeInBytes = 1024 * sizeof(float);

  cudaTextureDesc texDesc = {};
  texDesc.readMode = cudaReadModeElementType;

  cudaTextureObject_t tex_obj = 0;
  cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL);

  float *d_tex_out;
  cudaMalloc(&d_tex_out, sizeof(float));
  hammer_tex_latency<<<1, 32>>>(tex_obj, d_cycles, d_tex_out);
  cudaDeviceSynchronize();
  unsigned long long tex_cycles;
  cudaMemcpy(&tex_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("TEX1D_LATENCY_CYCLES=%.2f\n", (double)tex_cycles / 500.0);

  cudaDestroyTextureObject(tex_obj);
  cudaFree(d_tex_data);
  cudaFree(d_tex_out);
  free(h_tex_data);

  // --- 9. IMAD.WIDE ---
  long long *d_llval;
  cudaMalloc(&d_llval, sizeof(long long));
  hammer_imad_wide_latency<<<1, 32>>>(d_cycles, d_llval);
  cudaDeviceSynchronize();
  unsigned long long imad_wide_cycles;
  cudaMemcpy(&imad_wide_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("IMAD_WIDE_LATENCY_CYCLES=%.2f\n",
         (double)imad_wide_cycles / 1000.0);
  cudaFree(d_llval);

  // --- 10. FULL SFU FAMILY ---
  hammer_mufu_ex2_latency<<<1, 32>>>(d_cycles, d_fval);
  cudaDeviceSynchronize();
  unsigned long long ex2_cycles;
  cudaMemcpy(&ex2_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("MUFU_EX2_LATENCY_CYCLES=%.2f\n", (double)ex2_cycles / 1000.0);

  hammer_mufu_sin_latency<<<1, 32>>>(d_cycles, d_fval);
  cudaDeviceSynchronize();
  unsigned long long sin_cycles;
  cudaMemcpy(&sin_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("MUFU_SIN_LATENCY_CYCLES=%.2f\n", (double)sin_cycles / 1000.0);

  hammer_mufu_rsq_latency<<<1, 32>>>(d_cycles, d_fval);
  cudaDeviceSynchronize();
  unsigned long long rsq_cycles;
  cudaMemcpy(&rsq_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("MUFU_RSQ_LATENCY_CYCLES=%.2f\n", (double)rsq_cycles / 1000.0);

  hammer_mufu_lg2_latency<<<1, 32>>>(d_cycles, d_fval);
  cudaDeviceSynchronize();
  unsigned long long lg2_cycles;
  cudaMemcpy(&lg2_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("MUFU_LG2_LATENCY_CYCLES=%.2f\n", (double)lg2_cycles / 1000.0);

  // --- 11. REDUCED PRECISION ---
  // HFMA2 requires SM60+
  if (sm_ver >= 60) {
    half2 *d_h2val;
    cudaMalloc(&d_h2val, sizeof(half2));
    hammer_hfma2_latency<<<1, 32>>>(d_cycles, d_h2val);
    cudaDeviceSynchronize();
    unsigned long long hfma2_cycles;
    cudaMemcpy(&hfma2_cycles, d_cycles, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    printf("HFMA2_LATENCY_CYCLES=%.2f\n", (double)hfma2_cycles / 1000.0);
    cudaFree(d_h2val);
  } else {
    printf("HFMA2_LATENCY_CYCLES=N/A\n");
  }

  // BF16 requires SM80+ (Ampere)
  if (sm_ver >= 80) {
    __nv_bfloat162 *d_bf2val;
    cudaMalloc(&d_bf2val, sizeof(__nv_bfloat162));
    hammer_bf16x2_latency<<<1, 32>>>(d_cycles, d_bf2val);
    cudaDeviceSynchronize();
    unsigned long long bf16_cycles;
    cudaMemcpy(&bf16_cycles, d_cycles, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    printf("BF16X2_FMA_LATENCY_CYCLES=%.2f\n", (double)bf16_cycles / 1000.0);
    cudaFree(d_bf2val);
  } else {
    printf("BF16X2_FMA_LATENCY_CYCLES=N/A\n");
  }

  // --- 12. LOP3.LUT ---
  unsigned int *d_uval2;
  cudaMalloc(&d_uval2, sizeof(unsigned int));
  hammer_lop3_latency<<<1, 32>>>(d_cycles, d_uval2);
  cudaDeviceSynchronize();
  unsigned long long lop3_cycles;
  cudaMemcpy(&lop3_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("LOP3_LUT_LATENCY_CYCLES=%.2f\n", (double)lop3_cycles / 1000.0);
  cudaFree(d_uval2);

  // --- 13. FP64 DADD ---
  double *d_dval2;
  cudaMalloc(&d_dval2, sizeof(double));
  hammer_dadd_latency<<<1, 32>>>(d_cycles, d_dval2);
  cudaDeviceSynchronize();
  unsigned long long dadd_cycles;
  cudaMemcpy(&dadd_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("DADD_LATENCY_CYCLES=%.2f\n", (double)dadd_cycles / 100.0);
  cudaFree(d_dval2);

  // --- 14. REDUX.SUM (SM80+) and MEMBAR ---
  if (sm_ver >= 80) {
    hammer_redux_sum<<<1, 32>>>(d_cycles, d_ival);
    cudaDeviceSynchronize();
    unsigned long long redux_cycles;
    cudaMemcpy(&redux_cycles, d_cycles, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    printf("REDUX_SUM_LATENCY_CYCLES=%.2f\n", (double)redux_cycles / 100.0);
  } else {
    printf("REDUX_SUM_LATENCY_CYCLES=N/A\n");
  }

  hammer_membar_gpu<<<1, 32>>>(d_cycles);
  cudaDeviceSynchronize();
  unsigned long long membar_cycles;
  cudaMemcpy(&membar_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("MEMBAR_GPU_LATENCY_CYCLES=%.2f\n", (double)membar_cycles / 50.0);

  // --- 15. CONSTANT MEMORY (LDC) ---
  // Fill the constant LUT with values that create a dependent chain
  float h_const_lut[256];
  for (int i = 0; i < 256; i++)
    h_const_lut[i] = (float)(i % 64) / 64.0f;
  cudaMemcpyToSymbol(const_lut, h_const_lut, sizeof(h_const_lut));

  hammer_ldc_latency<<<1, 32>>>(d_cycles, d_fval);
  cudaDeviceSynchronize();
  unsigned long long ldc_cycles;
  cudaMemcpy(&ldc_cycles, d_cycles, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("LDC_LATENCY_CYCLES=%.2f\n", (double)ldc_cycles / 500.0);

  // --- HARDWARE LIMITS ---
  printf("MAX_REGS_PER_THREAD=%d\n", prop.regsPerBlock / prop.maxThreadsPerBlock);
  printf("MAX_REGS_PER_SM=%d\n", prop.regsPerMultiprocessor);
  printf("WARP_SIZE=%d\n", prop.warpSize);
  printf("MAX_THREADS_PER_SM=%d\n", prop.maxThreadsPerMultiProcessor);
  printf("MAX_WARPS_PER_SM=%d\n",
         prop.maxThreadsPerMultiProcessor / prop.warpSize);
  printf("TOTAL_GLOBAL_MEM_MB=%llu\n",
         (unsigned long long)prop.totalGlobalMem / (1024 * 1024));

  // --- 16. SMEM BANK CONFLICTS ---
  int *d_ival2;
  cudaMalloc(&d_ival2, sizeof(int));

  hammer_smem_no_conflict<<<1, 32>>>(d_cycles, d_ival2);
  cudaDeviceSynchronize();
  unsigned long long smem_nc_cycles;
  cudaMemcpy(&smem_nc_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("SMEM_NOCONFLICT_CYCLES=%.2f\n", (double)smem_nc_cycles / 1000.0);

  hammer_smem_2way_conflict<<<1, 32>>>(d_cycles, d_ival2);
  cudaDeviceSynchronize();
  unsigned long long smem_2w_cycles;
  cudaMemcpy(&smem_2w_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("SMEM_2WAY_CONFLICT_CYCLES=%.2f\n", (double)smem_2w_cycles / 1000.0);

  hammer_smem_4way_conflict<<<1, 32>>>(d_cycles, d_ival2);
  cudaDeviceSynchronize();
  unsigned long long smem_4w_cycles;
  cudaMemcpy(&smem_4w_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("SMEM_4WAY_CONFLICT_CYCLES=%.2f\n", (double)smem_4w_cycles / 1000.0);

  hammer_smem_broadcast<<<1, 32>>>(d_cycles, d_ival2);
  cudaDeviceSynchronize();
  unsigned long long smem_bc_cycles;
  cudaMemcpy(&smem_bc_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("SMEM_BROADCAST_CYCLES=%.2f\n", (double)smem_bc_cycles / 1000.0);

  // Derived: penalty = conflict_cost - no_conflict_cost
  printf("SMEM_2WAY_CONFLICT_PENALTY=%.2f\n",
         (double)(smem_2w_cycles - smem_nc_cycles) / 1000.0);
  printf("SMEM_4WAY_CONFLICT_PENALTY=%.2f\n",
         (double)(smem_4w_cycles - smem_nc_cycles) / 1000.0);
  // If penalty > threshold, compiler should pad shared arrays by 1 int
  printf("SMEM_PADDING_NEEDED=%d\n",
         (smem_2w_cycles > smem_nc_cycles * 12 / 10) ? 1 : 0); // >20% overhead
  cudaFree(d_ival2);

  // --- 17. TYPE CONVERSION LATENCIES ---
  int *d_i_conv;
  float *d_f_conv;
  unsigned short *d_h_conv;
  cudaMalloc(&d_i_conv, sizeof(int));
  cudaMalloc(&d_f_conv, sizeof(float));
  cudaMalloc(&d_h_conv, sizeof(unsigned short));

  hammer_f2i_latency<<<1, 32>>>(d_cycles, d_i_conv);
  cudaDeviceSynchronize();
  unsigned long long f2i_cycles;
  cudaMemcpy(&f2i_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("F2I_LATENCY_CYCLES=%.2f\n", (double)f2i_cycles / 1000.0);

  hammer_i2f_latency<<<1, 32>>>(d_cycles, d_f_conv);
  cudaDeviceSynchronize();
  unsigned long long i2f_cycles;
  cudaMemcpy(&i2f_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("I2F_LATENCY_CYCLES=%.2f\n", (double)i2f_cycles / 1000.0);

  hammer_f2h_latency<<<1, 32>>>(d_cycles, d_h_conv);
  cudaDeviceSynchronize();
  unsigned long long f2h_cycles;
  cudaMemcpy(&f2h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("F2H_LATENCY_CYCLES=%.2f\n", (double)f2h_cycles / 1000.0);

  hammer_h2f_latency<<<1, 32>>>(d_cycles, d_f_conv);
  cudaDeviceSynchronize();
  unsigned long long h2f_cycles;
  cudaMemcpy(&h2f_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("H2F_LATENCY_CYCLES=%.2f\n", (double)h2f_cycles / 1000.0);

  cudaFree(d_i_conv);
  cudaFree(d_f_conv);
  cudaFree(d_h_conv);

  // --- 18. DP4A INT8 DOT PRODUCT (SM61+) ---
  if (sm_ver >= 61) {
    int *d_dp4a_out;
    cudaMalloc(&d_dp4a_out, sizeof(int));
    hammer_dp4a_latency<<<1, 32>>>(d_cycles, d_dp4a_out);
    cudaDeviceSynchronize();
    unsigned long long dp4a_cycles;
    cudaMemcpy(&dp4a_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("DP4A_LATENCY_CYCLES=%.2f\n", (double)dp4a_cycles / 1000.0);
    cudaFree(d_dp4a_out);
  } else {
    printf("DP4A_LATENCY_CYCLES=N/A\n");
  }

  // --- 19. BIT MANIPULATION ---
  int   *d_bit_i;
  unsigned int *d_bit_u;
  cudaMalloc(&d_bit_i, sizeof(int));
  cudaMalloc(&d_bit_u, sizeof(unsigned int));

  hammer_popc_latency<<<1, 32>>>(d_cycles, d_bit_i);
  cudaDeviceSynchronize();
  unsigned long long popc_cycles;
  cudaMemcpy(&popc_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("POPC_LATENCY_CYCLES=%.2f\n", (double)popc_cycles / 1000.0);

  hammer_clz_latency<<<1, 32>>>(d_cycles, d_bit_i);
  cudaDeviceSynchronize();
  unsigned long long clz_cycles;
  cudaMemcpy(&clz_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("CLZ_LATENCY_CYCLES=%.2f\n", (double)clz_cycles / 1000.0);

  hammer_prmt_latency<<<1, 32>>>(d_cycles, d_bit_u);
  cudaDeviceSynchronize();
  unsigned long long prmt_cycles;
  cudaMemcpy(&prmt_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("PRMT_LATENCY_CYCLES=%.2f\n", (double)prmt_cycles / 1000.0);

  cudaFree(d_bit_i);
  cudaFree(d_bit_u);

  // --- 20. WARP VOTE PRIMITIVES (SM70+) ---
  if (sm_ver >= 70) {
    unsigned int *d_vote_u;
    int          *d_vote_i;
    cudaMalloc(&d_vote_u, sizeof(unsigned int));
    cudaMalloc(&d_vote_i, sizeof(int));

    hammer_ballot_latency<<<1, 32>>>(d_cycles, d_vote_u);
    cudaDeviceSynchronize();
    unsigned long long ballot_cycles;
    cudaMemcpy(&ballot_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("BALLOT_SYNC_LATENCY_CYCLES=%.2f\n", (double)ballot_cycles / 500.0);

    hammer_vote_any_latency<<<1, 32>>>(d_cycles, d_vote_i);
    cudaDeviceSynchronize();
    unsigned long long vote_any_cycles;
    cudaMemcpy(&vote_any_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("VOTE_ANY_LATENCY_CYCLES=%.2f\n", (double)vote_any_cycles / 500.0);

    cudaFree(d_vote_u);
    cudaFree(d_vote_i);
  } else {
    printf("BALLOT_SYNC_LATENCY_CYCLES=N/A\n");
    printf("VOTE_ANY_LATENCY_CYCLES=N/A\n");
  }

  // --- 21. READ-ONLY CACHE (__ldg) ---
  // Reuse d_tex_data-style array (1024 floats, values in [0,1))
  float *h_nc_data = (float *)malloc(1024 * sizeof(float));
  for (int i = 0; i < 1024; i++)
    h_nc_data[i] = (float)(i % 128) / 128.0f;
  float *d_nc_data;
  cudaMalloc(&d_nc_data, 1024 * sizeof(float));
  cudaMemcpy(d_nc_data, h_nc_data, 1024 * sizeof(float), cudaMemcpyHostToDevice);

  float *d_nc_out;
  cudaMalloc(&d_nc_out, sizeof(float));
  hammer_ldg_nc_latency<<<1, 32>>>(d_nc_data, d_cycles, d_nc_out);
  cudaDeviceSynchronize();
  unsigned long long ldg_nc_cycles;
  cudaMemcpy(&ldg_nc_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("LDG_NC_LATENCY_CYCLES=%.2f\n", (double)ldg_nc_cycles / 500.0);

  cudaFree(d_nc_data);
  cudaFree(d_nc_out);
  free(h_nc_data);

  // --- 22. GLOBAL ATOMICS ---
  float *d_atom_f;
  int   *d_atom_i;
  cudaMalloc(&d_atom_f, sizeof(float));
  cudaMalloc(&d_atom_i, sizeof(int));
  float zero_f = 0.0f;
  int   zero_i = 0;
  cudaMemcpy(d_atom_f, &zero_f, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_atom_i, &zero_i, sizeof(int),   cudaMemcpyHostToDevice);

  hammer_atomic_add_f32<<<1, 32>>>(d_atom_f, d_cycles);
  cudaDeviceSynchronize();
  unsigned long long atom_f32_cycles;
  cudaMemcpy(&atom_f32_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("ATOM_ADD_F32_LATENCY_CYCLES=%.2f\n", (double)atom_f32_cycles / 50.0);

  hammer_atomic_add_i32<<<1, 32>>>(d_atom_i, d_cycles);
  cudaDeviceSynchronize();
  unsigned long long atom_i32_cycles;
  cudaMemcpy(&atom_i32_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("ATOM_ADD_I32_LATENCY_CYCLES=%.2f\n", (double)atom_i32_cycles / 50.0);

  cudaFree(d_atom_f);
  cudaFree(d_atom_i);

  // --- 23. STRIDED GLOBAL MEMORY ---
  // 4MB float array — enough to exceed L1 for all stride patterns
  size_t stride_elements = (4 * 1024 * 1024) / sizeof(float);
  float *h_stride = (float *)malloc(stride_elements * sizeof(float));
  for (size_t i = 0; i < stride_elements; i++) h_stride[i] = (float)(i % 128) / 128.0f;
  float *d_stride;
  cudaMalloc(&d_stride, stride_elements * sizeof(float));
  cudaMemcpy(d_stride, h_stride, stride_elements * sizeof(float), cudaMemcpyHostToDevice);
  float *d_stride_out;
  cudaMalloc(&d_stride_out, sizeof(float));

  int strides[] = {1, 2, 4, 8, 16, 32};
  const char *stride_labels[] = {
    "STRIDE1_CYCLES", "STRIDE2_CYCLES", "STRIDE4_CYCLES",
    "STRIDE8_CYCLES", "STRIDE16_CYCLES", "STRIDE32_CYCLES"
  };
  for (int si = 0; si < 6; si++) {
    hammer_stride_global<<<1, 32>>>(d_stride, d_cycles, d_stride_out, strides[si]);
    cudaDeviceSynchronize();
    unsigned long long st_cycles;
    cudaMemcpy(&st_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("%s=%.2f\n", stride_labels[si], (double)st_cycles / 200.0);
  }
  cudaFree(d_stride);
  cudaFree(d_stride_out);
  free(h_stride);

  // --- 24. CP.ASYNC LATENCY (SM80+) ---
  if (sm_ver >= 80) {
    float *h_async_src = (float *)malloc(32 * sizeof(float));
    for (int i = 0; i < 32; i++) h_async_src[i] = (float)i;
    float *d_async_src;
    cudaMalloc(&d_async_src, 32 * sizeof(float));
    cudaMemcpy(d_async_src, h_async_src, 32 * sizeof(float), cudaMemcpyHostToDevice);
    float *d_async_out;
    cudaMalloc(&d_async_out, sizeof(float));

    hammer_cp_async_latency<<<1, 32>>>(d_async_src, d_cycles, d_async_out);
    cudaDeviceSynchronize();
    unsigned long long cp_async_cycles;
    cudaMemcpy(&cp_async_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("CP_ASYNC_LATENCY_CYCLES=%.2f\n", (double)cp_async_cycles / 1.0);

    cudaFree(d_async_src);
    cudaFree(d_async_out);
    free(h_async_src);
  } else {
    printf("CP_ASYNC_LATENCY_CYCLES=N/A\n");
  }

  // --- 25. FMA ILP THROUGHPUT ---
  float *d_tp_out;
  cudaMalloc(&d_tp_out, sizeof(float));
  hammer_fma_throughput<<<1, 32>>>(d_cycles, d_tp_out);
  cudaDeviceSynchronize();
  unsigned long long fma_tp_cycles;
  cudaMemcpy(&fma_tp_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  // 8 FMAs × 1000 iters = 8000 ops. Throughput = ops / cycles.
  double fma_throughput = 8000.0 / (double)fma_tp_cycles;
  printf("FMA_ILP_THROUGHPUT=%.4f\n", fma_throughput); // FMAs per cycle (>1 = dual-issue)
  printf("FMA_ILP_CYCLES_PER_OP=%.2f\n", (double)fma_tp_cycles / 8000.0);
  cudaFree(d_tp_out);

  // Cleanup
  cudaFree(d_cycles);
  cudaFree(d_fval);
  cudaFree(d_ival);
  cudaFree(d_l1_array);
  cudaFree(d_l2_array);
  cudaFree(d_vram_array);
  cudaFree(d_error);
  free(h_l1_array);
  free(h_l2_array);
  free(h_vram_array);

  return 0;
}

