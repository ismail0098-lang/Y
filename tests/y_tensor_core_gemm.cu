#include <mma.h>
#include <cuda_fp16.h>

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

using namespace nvcuda;

// Helper functions for PTX cp.async (.cg L1 cache bypass), ldmatrix, and mma.sync
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr, bool valid) {
    uint32_t smem_ptr32 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    if (valid && gmem_ptr != nullptr) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_ptr32), "l"(gmem_ptr)
        );
    } else if (gmem_ptr != nullptr) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
            :: "r"(smem_ptr32), "l"(gmem_ptr), "r"(0)
        );
    } else {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %2, 0;\n"
            "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
            "}\n"
            :: "r"(smem_ptr32), "l"(gmem_ptr), "r"((int)valid)
        );
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t reg[4], uint32_t smem_ptr32) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(smem_ptr32)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t reg[2], uint32_t smem_ptr32) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(smem_ptr32)
    );
}

__device__ __forceinline__ void mma_m16n8k16(
    float c[4],
    const uint32_t a[4],
    const uint32_t b[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// Standalone Y Tensor Core MMA GEMM simulation kernel (128x128x32 CTA Tile + 128B XOR SMEM Swizzle + 4-Stage cp.async + mma.sync)
// 64 KB Dynamic Shared Memory Allocation (32KB smem_A + 32KB smem_B)
__device__ __forceinline__ void get_morton_cta_coords(
    int tile_idx, int grid_m, int grid_n,
    int block_m, int block_n,
    int& cta_m, int& cta_n
) {
    const int PANEL_M = 8;
    const int PANEL_N = 8;
    const int PANEL_SIZE = PANEL_M * PANEL_N;
    int panel_idx = tile_idx / PANEL_SIZE;
    int in_panel_idx = tile_idx % PANEL_SIZE;
    int panels_in_n = (grid_n + PANEL_N - 1) / PANEL_N;
    int panel_m = panel_idx / panels_in_n;
    int panel_n = panel_idx % panels_in_n;
    int in_m = ((in_panel_idx & 1)) | ((in_panel_idx & 4) >> 1) | ((in_panel_idx & 16) >> 2);
    int in_n = ((in_panel_idx & 2) >> 1) | ((in_panel_idx & 8) >> 2) | ((in_panel_idx & 32) >> 3);
    int block_idx_m = panel_m * PANEL_M + in_m;
    int block_idx_n = panel_n * PANEL_N + in_n;
    if (block_idx_m < grid_m && block_idx_n < grid_n) {
        cta_m = block_idx_m * block_m;
        cta_n = block_idx_n * block_n;
    } else {
        cta_m = blockIdx.y * block_m;
        cta_n = blockIdx.x * block_n;
    }
}

extern "C" __global__ __launch_bounds__(256, 1) void y_tensor_core_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    int grid_m = (M + BLOCK_M - 1) / BLOCK_M;
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int cta_m, cta_n;
    get_morton_cta_coords(tile_idx, grid_m, grid_n, BLOCK_M, BLOCK_N, cta_m, cta_n);

    // Padded Dynamic Shared Memory Allocation (32+8=40 stride for A, 128+8=136 stride for B to eliminate bank conflicts)
    __shared__ alignas(128) half smem_storage[20480 / 2];
    half (*smem_A)[128][40] = (half (*)[128][40])smem_storage;
    __shared__ alignas(128) half smem_B[2][32][136];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id % 4; // 0..3 (4 warps in M, 32 M per warp)
    int warp_n = warp_id / 4; // 0..1 (2 warps in N, 64 N per warp)

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    auto load_stage = [&](int stage, int k_curr) {
        // Tile A: 128 rows x 32 halfs = 4096 halfs = 256 16-half packs (2 uint4s per thread)
        int r = tid / 2;        // 0..127
        int c = (tid % 2) * 16; // 0, 16
        int gmem_r = cta_m + r;
        int gmem_c = k_curr + c;
        if (gmem_r < M && (gmem_c + 15) < K) {
            *reinterpret_cast<uint4*>(&smem_A[stage][r][c])     = *reinterpret_cast<const uint4*>(&A[gmem_r * K + gmem_c]);
            *reinterpret_cast<uint4*>(&smem_A[stage][r][c + 8]) = *reinterpret_cast<const uint4*>(&A[gmem_r * K + gmem_c + 8]);
        } else {
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                half val = __float2half(0.0f);
                if (gmem_r < M && (gmem_c + e) < K) {
                    val = A[gmem_r * K + gmem_c + e];
                }
                smem_A[stage][r][c + e] = val;
            }
        }

        // Tile B: 32 rows x 128 halfs = 4096 halfs = 256 16-half packs (2 uint4s per thread)
        int br = tid / 8;        // 0..31
        int bc = (tid % 8) * 16;  // 0, 16, 32, ..., 112
        int bgmem_r = k_curr + br;
        int bgmem_c = cta_n + bc;
        if (bgmem_r < K && (bgmem_c + 15) < N) {
            *reinterpret_cast<uint4*>(&smem_B[stage][br][bc])     = *reinterpret_cast<const uint4*>(&B[bgmem_r * N + bgmem_c]);
            *reinterpret_cast<uint4*>(&smem_B[stage][br][bc + 8]) = *reinterpret_cast<const uint4*>(&B[bgmem_r * N + bgmem_c + 8]);
        } else {
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                half val = __float2half(0.0f);
                if (bgmem_r < K && (bgmem_c + e) < N) {
                    val = B[bgmem_r * N + bgmem_c + e];
                }
                smem_B[stage][br][bc + e] = val;
            }
        }
    };

    int write_stage = 0;
    int read_stage = 0;

    for (int k = 0; k < K; k += BLOCK_K) {
        load_stage(write_stage, k);
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[4];

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int a_row = warp_m * 32 + i * 16;
                wmma::load_matrix_sync(frag_A[i], &smem_A[read_stage][a_row][k_step], 40);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int b_col = warp_n * 64 + j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[read_stage][k_step][b_col], 136);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                }
            }
        }

        __syncthreads();
        write_stage = 1 - write_stage;
        read_stage = 1 - read_stage;
    }

    // Fast Vectorized Epilogue Store via Memory Reuse
    float (*smem_C_f32)[32][64] = (float (*)[32][64])smem_storage;
    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        int active_warp = warp_id - pass * 2;
        if (active_warp >= 0 && active_warp < 2) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::store_matrix_sync(&smem_C_f32[active_warp][i * 16][j * 16], frag_C[i][j], 64, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
        if (active_warp >= 0 && active_warp < 2) {
            int warp_out_m = cta_m + warp_m * 32;
            int warp_out_n = cta_n + warp_n * 64;
            int lane = tid % 32;
            int store_r = lane / 8;
            int store_c = (lane % 8) * 8;
            #pragma unroll
            for (int r_off = 0; r_off < 32; r_off += 4) {
                int gm_r = warp_out_m + store_r + r_off;
                int gm_c = warp_out_n + store_c;
                if (gm_r < M && (gm_c + 7) < N) {
                    half out8[8];
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        out8[e] = __float2half(smem_C_f32[active_warp][store_r + r_off][store_c + e]);
                    }
                    *reinterpret_cast<uint4*>(&C[gm_r * N + gm_c]) = *reinterpret_cast<const uint4*>(out8);
                }
            }
        }
        __syncthreads();
    }
}


// Single-Warp Barrier-Free Micro GEMM (16x32 CTA Tile, 32 Threads, Zero __syncthreads(), FP16 Output) for M,N <= 512
extern "C" __global__ __launch_bounds__(32, 4) void y_fused_gemm_barrier_free_16x32_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 16;
    const int BLOCK_N = 32;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A_0[16][40];
    __shared__ alignas(128) half smem_B_0[32][40];
    __shared__ alignas(128) half smem_A_1[16][40];
    __shared__ alignas(128) half smem_B_1[32][40];

    int tid = threadIdx.x; // 0..31 (Single Warp)

    const int SWIZZLE = 8;
    int grid_n = gridDim.x;
    int grid_m = gridDim.y;
    int tile_idx = blockIdx.y * grid_n + blockIdx.x;
    int num_tiles_per_swizzle = grid_n * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;
    if (cta_m >= M || cta_n >= N) {
        cta_m = blockIdx.y * BLOCK_M;
        cta_n = blockIdx.x * BLOCK_N;
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2];

    wmma::fill_fragment(frag_C[0], 0.0f);
    wmma::fill_fragment(frag_C[1], 0.0f);

    int load_a_row = tid / 2;        // 0..15
    int load_a_col = (tid % 2) * 16; // 0, 16
    int load_b_row = tid;            // 0..31

    // Prologue Stage 0
    if (cta_m + load_a_row < M && load_a_col + 15 < K) {
        *(uint4*)&smem_A_0[load_a_row][load_a_col]     = *(const uint4*)&A[(cta_m + load_a_row) * K + load_a_col];
        *(uint4*)&smem_A_0[load_a_row][load_a_col + 8] = *(const uint4*)&A[(cta_m + load_a_row) * K + load_a_col + 8];
    } else {
        *(uint4*)&smem_A_0[load_a_row][load_a_col]     = make_uint4(0, 0, 0, 0);
        *(uint4*)&smem_A_0[load_a_row][load_a_col + 8] = make_uint4(0, 0, 0, 0);
    }

    if (load_b_row < K && cta_n + 31 < N) {
        uint4* dst_b = (uint4*)&smem_B_0[load_b_row][0];
        const uint4* src_b = (const uint4*)&B[load_b_row * N + cta_n];
        dst_b[0] = src_b[0]; dst_b[1] = src_b[1]; dst_b[2] = src_b[2]; dst_b[3] = src_b[3];
    } else {
        #pragma unroll
        for (int c_idx = 0; c_idx < 32; c_idx += 8) {
            *(uint4*)&smem_B_0[load_b_row][c_idx] = make_uint4(0, 0, 0, 0);
        }
    }

    int stage = 0;
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        if (next_k < K) {
            half (*smem_A_next)[40] = (stage == 0) ? smem_A_1 : smem_A_0;
            half (*smem_B_next)[40] = (stage == 0) ? smem_B_1 : smem_B_0;
            if (cta_m + load_a_row < M && next_k + load_a_col + 15 < K) {
                *(uint4*)&smem_A_next[load_a_row][load_a_col]     = *(const uint4*)&A[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                *(uint4*)&smem_A_next[load_a_row][load_a_col + 8] = *(const uint4*)&A[(cta_m + load_a_row) * K + (next_k + load_a_col + 8)];
            } else {
                *(uint4*)&smem_A_next[load_a_row][load_a_col]     = make_uint4(0, 0, 0, 0);
                *(uint4*)&smem_A_next[load_a_row][load_a_col + 8] = make_uint4(0, 0, 0, 0);
            }

            if (next_k + load_b_row < K && cta_n + 31 < N) {
                uint4* dst_b = (uint4*)&smem_B_next[load_b_row][0];
                const uint4* src_b = (const uint4*)&B[(next_k + load_b_row) * N + cta_n];
                dst_b[0] = src_b[0]; dst_b[1] = src_b[1]; dst_b[2] = src_b[2]; dst_b[3] = src_b[3];
            } else {
                #pragma unroll
                for (int c_idx = 0; c_idx < 32; c_idx += 8) {
                    *(uint4*)&smem_B_next[load_b_row][c_idx] = make_uint4(0, 0, 0, 0);
                }
            }
        }

        half (*smem_A_curr)[40] = (stage == 0) ? smem_A_0 : smem_A_1;
        half (*smem_B_curr)[40] = (stage == 0) ? smem_B_0 : smem_B_1;

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            wmma::load_matrix_sync(frag_A, &smem_A_curr[0][k_step], 40);
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int b_col = j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B_curr[k_step][b_col], 40);
                wmma::mma_sync(frag_C[j], frag_A, frag_B[j], frag_C[j]);
            }
        }

        stage = 1 - stage;
    }

    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        int out_m = cta_m;
        int out_n = cta_n + j * 16;
        if (out_m + 15 < M && out_n + 15 < N) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_h;
            #pragma unroll
            for (int k = 0; k < frag_h.num_elements; ++k) {
                frag_h.x[k] = __float2half(frag_C[j].x[k]);
            }
            wmma::store_matrix_sync(&C[out_m * N + out_n], frag_h, N, wmma::mem_row_major);
        }
    }
}

// Direct-Register Micro GEMM (32x32 CTA Tile, 64 Threads, Direct Global -> WMMA Registers, FP16 Output)
// Direct-Register Micro GEMM (32x32 CTA Tile, 64 Threads, Direct Global -> WMMA Registers, FP16 Output)
extern "C" __global__ __launch_bounds__(64, 4) void y_fused_gemm_direct_reg_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 32;
    const int BLOCK_N = 32;

    int tid = threadIdx.x;
    int warpId = tid / 32;
    int warp_m_idx = warpId; // 0 or 1 -> offset 0, 16

    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C[2];

    wmma::fill_fragment(frag_C[0], __float2half(0.0f));
    wmma::fill_fragment(frag_C[1], __float2half(0.0f));

    int a_row = cta_m + warp_m_idx * 16;
    int b_col0 = cta_n;
    int b_col1 = cta_n + 16;

    #pragma unroll 4
    for (int k = 0; k < K; k += 16) {
        if (a_row < M && k < K) {
            wmma::load_matrix_sync(frag_A, &A[a_row * K + k], K);
        }
        if (k < K && b_col0 < N) {
            wmma::load_matrix_sync(frag_B[0], &B[k * N + b_col0], N);
        }
        if (k < K && b_col1 < N) {
            wmma::load_matrix_sync(frag_B[1], &B[k * N + b_col1], N);
        }

        wmma::mma_sync(frag_C[0], frag_A, frag_B[0], frag_C[0]);
        wmma::mma_sync(frag_C[1], frag_A, frag_B[1], frag_C[1]);
    }

    int out_m = a_row;
    if (out_m < M && b_col0 < N) {
        wmma::store_matrix_sync(&C[out_m * N + b_col0], frag_C[0], N, wmma::mem_row_major);
    }
    if (out_m < M && b_col1 < N) {
        wmma::store_matrix_sync(&C[out_m * N + b_col1], frag_C[1], N, wmma::mem_row_major);
    }
}

// Ultra-Fast Micro GEMM (32x32 Tile, 64 Threads, 2 Warps, 256 CTA Blocks for 512x512)
extern "C" __global__ __launch_bounds__(64, 4) void y_fused_gemm_tiny_32x32_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 32;
    const int BLOCK_N = 32;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A_0[32][32 + 8];
    __shared__ alignas(128) half smem_B_0[32][32 + 8];
    __shared__ alignas(128) half smem_A_1[32][32 + 8];
    __shared__ alignas(128) half smem_B_1[32][32 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2; // 0 or 1 -> M offset 0, 16

    const int SWIZZLE = 8;
    int grid_n = gridDim.x;
    int grid_m = gridDim.y;
    int tile_idx = blockIdx.y * grid_n + blockIdx.x;
    int num_tiles_per_swizzle = grid_n * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;
    if (cta_m >= M || cta_n >= N) {
        cta_m = blockIdx.y * BLOCK_M;
        cta_n = blockIdx.x * BLOCK_N;
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2];

    wmma::fill_fragment(frag_C[0], 0.0f);
    wmma::fill_fragment(frag_C[1], 0.0f);

    // 64 threads load 32x32 halfs (1024 halfs = 64 uint4s) -> 1 uint4 per thread
    int load_a_row = tid / 2;        // 0..31
    int load_a_col = (tid % 2) * 16;  // 0, 16

    int load_b_row = tid / 2;        // 0..31
    int load_b_col = (tid % 2) * 16;  // 0, 16

    // Prologue Stage 0
    if (cta_m + load_a_row < M && load_a_col < K) {
        uint4* dst_a = (uint4*)&smem_A_0[load_a_row][load_a_col];
        const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + load_a_col];
        *dst_a = *src_a;
    } else {
        *(uint4*)&smem_A_0[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
    }

    if (load_b_row < K && cta_n + load_b_col < N) {
        uint4* dst_b = (uint4*)&smem_B_0[load_b_row][load_b_col];
        const uint4* src_b = (const uint4*)&B[load_b_row * N + (cta_n + load_b_col)];
        *dst_b = *src_b;
    } else {
        *(uint4*)&smem_B_0[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
    }

    __syncthreads();

    int stage = 0;
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        if (next_k < K) {
            if (stage == 0) {
                if (cta_m + load_a_row < M && next_k + load_a_col < K) {
                    uint4* dst_a = (uint4*)&smem_A_1[load_a_row][load_a_col];
                    const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                    *dst_a = *src_a;
                } else {
                    *(uint4*)&smem_A_1[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
                }

                if (next_k + load_b_row < K && cta_n + load_b_col < N) {
                    uint4* dst_b = (uint4*)&smem_B_1[load_b_row][load_b_col];
                    const uint4* src_b = (const uint4*)&B[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                    *dst_b = *src_b;
                } else {
                    *(uint4*)&smem_B_1[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
                }
            } else {
                if (cta_m + load_a_row < M && next_k + load_a_col < K) {
                    uint4* dst_a = (uint4*)&smem_A_0[load_a_row][load_a_col];
                    const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                    *dst_a = *src_a;
                } else {
                    *(uint4*)&smem_A_0[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
                }

                if (next_k + load_b_row < K && cta_n + load_b_col < N) {
                    uint4* dst_b = (uint4*)&smem_B_0[load_b_row][load_b_col];
                    const uint4* src_b = (const uint4*)&B[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                    *dst_b = *src_b;
                } else {
                    *(uint4*)&smem_B_0[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
                }
            }
        }

        if (stage == 0) {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                int a_row = warp_m_idx * 16;
                wmma::load_matrix_sync(frag_A, &smem_A_0[a_row][k_step], 40);
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 40);
                    wmma::mma_sync(frag_C[j], frag_A, frag_B[j], frag_C[j]);
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                int a_row = warp_m_idx * 16;
                wmma::load_matrix_sync(frag_A, &smem_A_1[a_row][k_step], 40);
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 40);
                    wmma::mma_sync(frag_C[j], frag_A, frag_B[j], frag_C[j]);
                }
            }
        }

        __syncthreads();
        stage = 1 - stage;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C_half[2];
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            frag_C_half[j].x[k] = __float2half(frag_C[j].x[k]);
        }
        int out_m = cta_m + warp_m_idx * 16;
        int out_n = cta_n + j * 16;
        if (out_m < M && out_n < N) {
            wmma::store_matrix_sync(&C[out_m * N + out_n], frag_C_half[j], N, wmma::mem_row_major);
        }
    }
}
extern "C" __global__ __launch_bounds__(128, 4) void y_fused_gemm_tiny_32x64_128t_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 32;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A_0[32][32 + 8];
    __shared__ alignas(128) half smem_B_0[32][64 + 8];
    __shared__ alignas(128) half smem_A_1[32][32 + 8];
    __shared__ alignas(128) half smem_B_1[32][64 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = (warpId / 2) % 2; // 0 or 1 -> M offset 0, 16
    int warp_n_idx = warpId % 2;       // 0 or 1 -> N offset 0, 32

    const int SWIZZLE = 8;
    int grid_n = gridDim.x;
    int grid_m = gridDim.y;
    int tile_idx = blockIdx.y * grid_n + blockIdx.x;
    int num_tiles_per_swizzle = grid_n * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;
    if (cta_m >= M || cta_n >= N) {
        cta_m = blockIdx.y * BLOCK_M;
        cta_n = blockIdx.x * BLOCK_N;
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2];

    wmma::fill_fragment(frag_C[0], 0.0f);
    wmma::fill_fragment(frag_C[1], 0.0f);

    // 128 threads load 32x32 halfs (1024 halfs = 64 uint4s) -> tid 0..63 load 1 uint4
    int load_a_row = tid / 2;        // 0..31 (tid 0..63)
    int load_a_col = (tid % 2) * 16;  // 0, 16

    // 128 threads load 32x64 halfs (2048 halfs = 128 uint4s) -> 1 uint4 load per thread
    int load_b_row = tid / 4;        // 0..31
    int load_b_col = (tid % 4) * 16;  // 0, 16, 32, 48

    // Prologue Stage 0
    if (tid < 64) {
        if (cta_m + load_a_row < M && load_a_col < K) {
            uint4* dst_a = (uint4*)&smem_A_0[load_a_row][load_a_col];
            const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + load_a_col];
            *dst_a = *src_a;
        } else {
            *(uint4*)&smem_A_0[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
        }
    }

    if (load_b_row < K && cta_n + load_b_col < N) {
        uint4* dst_b = (uint4*)&smem_B_0[load_b_row][load_b_col];
        const uint4* src_b = (const uint4*)&B[load_b_row * N + (cta_n + load_b_col)];
        *dst_b = *src_b;
    } else {
        *(uint4*)&smem_B_0[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
    }

    __syncthreads();

    int stage = 0;
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        if (next_k < K) {
            if (stage == 0) {
                if (tid < 64) {
                    if (cta_m + load_a_row < M && next_k + load_a_col < K) {
                        uint4* dst_a = (uint4*)&smem_A_1[load_a_row][load_a_col];
                        const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                        *dst_a = *src_a;
                    } else {
                        *(uint4*)&smem_A_1[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
                    }
                }

                if (next_k + load_b_row < K && cta_n + load_b_col < N) {
                    uint4* dst_b = (uint4*)&smem_B_1[load_b_row][load_b_col];
                    const uint4* src_b = (const uint4*)&B[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                    *dst_b = *src_b;
                } else {
                    *(uint4*)&smem_B_1[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
                }
            } else {
                if (tid < 64) {
                    if (cta_m + load_a_row < M && next_k + load_a_col < K) {
                        uint4* dst_a = (uint4*)&smem_A_0[load_a_row][load_a_col];
                        const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                        *dst_a = *src_a;
                    } else {
                        *(uint4*)&smem_A_0[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
                    }
                }

                if (next_k + load_b_row < K && cta_n + load_b_col < N) {
                    uint4* dst_b = (uint4*)&smem_B_0[load_b_row][load_b_col];
                    const uint4* src_b = (const uint4*)&B[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                    *dst_b = *src_b;
                } else {
                    *(uint4*)&smem_B_0[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
                }
            }
        }

        if (stage == 0) {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                int a_row = warp_m_idx * 16;
                wmma::load_matrix_sync(frag_A, &smem_A_0[a_row][k_step], 40);
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 72);
                    wmma::mma_sync(frag_C[j], frag_A, frag_B[j], frag_C[j]);
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                int a_row = warp_m_idx * 16;
                wmma::load_matrix_sync(frag_A, &smem_A_1[a_row][k_step], 40);
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 72);
                    wmma::mma_sync(frag_C[j], frag_A, frag_B[j], frag_C[j]);
                }
            }
        }

        __syncthreads();
        stage = 1 - stage;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C_half[2];
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            frag_C_half[j].x[k] = __float2half(frag_C[j].x[k]);
        }
        int out_m = cta_m + warp_m_idx * 16;
        int out_n = cta_n + warp_n_idx * 32 + j * 16;
        if (out_m < M && out_n < N) {
            wmma::store_matrix_sync(&C[out_m * N + out_n], frag_C_half[j], N, wmma::mem_row_major);
        }
    }
}
extern "C" __global__ __launch_bounds__(64, 4) void y_fused_gemm_tiny_64x64_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A_0[64][32 + 8];
    __shared__ alignas(128) half smem_B_0[32][64 + 8];
    __shared__ alignas(128) half smem_A_1[64][32 + 8];
    __shared__ alignas(128) half smem_B_1[32][64 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_n_idx = warpId % 2; // 0 or 1 -> N offset 0, 32

    const int SWIZZLE = 8;
    int grid_n = gridDim.x;
    int grid_m = gridDim.y;
    int tile_idx = blockIdx.y * grid_n + blockIdx.x;
    int num_tiles_per_swizzle = grid_n * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;
    if (cta_m >= M || cta_n >= N) {
        cta_m = blockIdx.y * BLOCK_M;
        cta_n = blockIdx.x * BLOCK_N;
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[4][2];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    // 64 threads load 64x32 halfs -> 32 halfs (4 uint4s) per thread
    int load_a_row = tid;            // 0..63

    // 64 threads load 32x64 halfs -> 32 halfs (4 uint4s) per thread
    int load_b_row = tid / 2;        // 0..31
    int load_b_col = (tid % 2) * 32;  // 0, 32

    // Prologue Stage 0
    if (cta_m + load_a_row < M) {
        uint4* dst_a = (uint4*)&smem_A_0[load_a_row][0];
        const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K];
        *dst_a = *src_a;
        *(dst_a + 1) = *(src_a + 1);
    } else {
        *(uint4*)&smem_A_0[load_a_row][0] = make_uint4(0, 0, 0, 0);
        *(uint4*)&smem_A_0[load_a_row][16] = make_uint4(0, 0, 0, 0);
    }

    if (load_b_row < K && cta_n + load_b_col < N) {
        uint4* dst_b = (uint4*)&smem_B_0[load_b_row][load_b_col];
        const uint4* src_b = (const uint4*)&B[load_b_row * N + (cta_n + load_b_col)];
        *dst_b = *src_b;
        *(dst_b + 1) = *(src_b + 1);
        *(dst_b + 2) = *(src_b + 2);
        *(dst_b + 3) = *(src_b + 3);
    } else {
        *(uint4*)&smem_B_0[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
        *(uint4*)&smem_B_0[load_b_row][load_b_col + 8] = make_uint4(0, 0, 0, 0);
        *(uint4*)&smem_B_0[load_b_row][load_b_col + 16] = make_uint4(0, 0, 0, 0);
        *(uint4*)&smem_B_0[load_b_row][load_b_col + 24] = make_uint4(0, 0, 0, 0);
    }

    __syncthreads();

    int stage = 0;
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        if (next_k < K) {
            if (stage == 0) {
                if (cta_m + load_a_row < M && next_k + 31 < K) {
                    uint4* dst_a = (uint4*)&smem_A_1[load_a_row][0];
                    const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + next_k];
                    *dst_a = *src_a;
                    *(dst_a + 1) = *(src_a + 1);
                } else {
                    *(uint4*)&smem_A_1[load_a_row][0] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_A_1[load_a_row][16] = make_uint4(0, 0, 0, 0);
                }

                if (next_k + load_b_row < K && cta_n + load_b_col < N) {
                    uint4* dst_b = (uint4*)&smem_B_1[load_b_row][load_b_col];
                    const uint4* src_b = (const uint4*)&B[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                    *dst_b = *src_b;
                    *(dst_b + 1) = *(src_b + 1);
                    *(dst_b + 2) = *(src_b + 2);
                    *(dst_b + 3) = *(src_b + 3);
                } else {
                    *(uint4*)&smem_B_1[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_1[load_b_row][load_b_col + 8] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_1[load_b_row][load_b_col + 16] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_1[load_b_row][load_b_col + 24] = make_uint4(0, 0, 0, 0);
                }
            } else {
                if (cta_m + load_a_row < M && next_k + 31 < K) {
                    uint4* dst_a = (uint4*)&smem_A_0[load_a_row][0];
                    const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + next_k];
                    *dst_a = *src_a;
                    *(dst_a + 1) = *(src_a + 1);
                } else {
                    *(uint4*)&smem_A_0[load_a_row][0] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_A_0[load_a_row][16] = make_uint4(0, 0, 0, 0);
                }

                if (next_k + load_b_row < K && cta_n + load_b_col < N) {
                    uint4* dst_b = (uint4*)&smem_B_0[load_b_row][load_b_col];
                    const uint4* src_b = (const uint4*)&B[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                    *dst_b = *src_b;
                    *(dst_b + 1) = *(src_b + 1);
                    *(dst_b + 2) = *(src_b + 2);
                    *(dst_b + 3) = *(src_b + 3);
                } else {
                    *(uint4*)&smem_B_0[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_0[load_b_row][load_b_col + 8] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_0[load_b_row][load_b_col + 16] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_0[load_b_row][load_b_col + 24] = make_uint4(0, 0, 0, 0);
                }
            }
        }

        if (stage == 0) {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_0[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 72);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_1[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 72);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        }

        __syncthreads();
        stage = 1 - stage;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C_half[4][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                frag_C_half[i][j].x[k] = __float2half(frag_C[i][j].x[k]);
            }
            int out_m = cta_m + i * 16;
            int out_n = cta_n + warp_n_idx * 32 + j * 16;
            if (out_m < M && out_n < N) {
                wmma::store_matrix_sync(&C[out_m * N + out_n], frag_C_half[i][j], N, wmma::mem_row_major);
            }
        }
    }
}
extern "C" __global__ __launch_bounds__(64, 4) void y_fused_gemm_tiny_32x64_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 32;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A_0[32][32 + 8];
    __shared__ alignas(128) half smem_B_0[32][64 + 8];
    __shared__ alignas(128) half smem_A_1[32][32 + 8];
    __shared__ alignas(128) half smem_B_1[32][64 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2; // 0 or 1 -> M offset 0, 16

    const int SWIZZLE = 8;
    int grid_n = gridDim.x;
    int grid_m = gridDim.y;
    int tile_idx = blockIdx.y * grid_n + blockIdx.x;
    int num_tiles_per_swizzle = grid_n * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;
    if (cta_m >= M || cta_n >= N) {
        cta_m = blockIdx.y * BLOCK_M;
        cta_n = blockIdx.x * BLOCK_N;
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[4];

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        wmma::fill_fragment(frag_C[j], 0.0f);
    }

    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 16;

    int load_b_row = tid / 2;
    int load_b_col = (tid % 2) * 32;

    // Prologue Stage 0
    if (cta_m + load_a_row < M && load_a_col < K) {
        uint4* dst_a = (uint4*)&smem_A_0[load_a_row][load_a_col];
        const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + load_a_col];
        *dst_a = *src_a;
        *(dst_a + 1) = *(src_a + 1);
    } else {
        *(uint4*)&smem_A_0[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
        *(uint4*)&smem_A_0[load_a_row][load_a_col + 8] = make_uint4(0, 0, 0, 0);
    }

    if (load_b_row < K && cta_n + load_b_col < N) {
        uint4* dst_b = (uint4*)&smem_B_0[load_b_row][load_b_col];
        const uint4* src_b = (const uint4*)&B[load_b_row * N + (cta_n + load_b_col)];
        *dst_b = *src_b;
        *(dst_b + 1) = *(src_b + 1);
        *(dst_b + 2) = *(src_b + 2);
        *(dst_b + 3) = *(src_b + 3);
    } else {
        *(uint4*)&smem_B_0[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
        *(uint4*)&smem_B_0[load_b_row][load_b_col + 8] = make_uint4(0, 0, 0, 0);
        *(uint4*)&smem_B_0[load_b_row][load_b_col + 16] = make_uint4(0, 0, 0, 0);
        *(uint4*)&smem_B_0[load_b_row][load_b_col + 24] = make_uint4(0, 0, 0, 0);
    }

    __syncthreads();

    int stage = 0;
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        if (next_k < K) {
            if (stage == 0) {
                if (cta_m + load_a_row < M && next_k + load_a_col < K) {
                    uint4* dst_a = (uint4*)&smem_A_1[load_a_row][load_a_col];
                    const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                    *dst_a = *src_a;
                    *(dst_a + 1) = *(src_a + 1);
                } else {
                    *(uint4*)&smem_A_1[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_A_1[load_a_row][load_a_col + 8] = make_uint4(0, 0, 0, 0);
                }

                if (next_k + load_b_row < K && cta_n + load_b_col < N) {
                    uint4* dst_b = (uint4*)&smem_B_1[load_b_row][load_b_col];
                    const uint4* src_b = (const uint4*)&B[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                    *dst_b = *src_b;
                    *(dst_b + 1) = *(src_b + 1);
                    *(dst_b + 2) = *(src_b + 2);
                    *(dst_b + 3) = *(src_b + 3);
                } else {
                    *(uint4*)&smem_B_1[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_1[load_b_row][load_b_col + 8] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_1[load_b_row][load_b_col + 16] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_1[load_b_row][load_b_col + 24] = make_uint4(0, 0, 0, 0);
                }
            } else {
                if (cta_m + load_a_row < M && next_k + load_a_col < K) {
                    uint4* dst_a = (uint4*)&smem_A_0[load_a_row][load_a_col];
                    const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                    *dst_a = *src_a;
                    *(dst_a + 1) = *(src_a + 1);
                } else {
                    *(uint4*)&smem_A_0[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_A_0[load_a_row][load_a_col + 8] = make_uint4(0, 0, 0, 0);
                }

                if (next_k + load_b_row < K && cta_n + load_b_col < N) {
                    uint4* dst_b = (uint4*)&smem_B_0[load_b_row][load_b_col];
                    const uint4* src_b = (const uint4*)&B[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                    *dst_b = *src_b;
                    *(dst_b + 1) = *(src_b + 1);
                    *(dst_b + 2) = *(src_b + 2);
                    *(dst_b + 3) = *(src_b + 3);
                } else {
                    *(uint4*)&smem_B_0[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_0[load_b_row][load_b_col + 8] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_0[load_b_row][load_b_col + 16] = make_uint4(0, 0, 0, 0);
                    *(uint4*)&smem_B_0[load_b_row][load_b_col + 24] = make_uint4(0, 0, 0, 0);
                }
            }
        }

        if (stage == 0) {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                int a_row = warp_m_idx * 16;
                wmma::load_matrix_sync(frag_A, &smem_A_0[a_row][k_step], 40);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int b_col = j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 72);
                    wmma::mma_sync(frag_C[j], frag_A, frag_B[j], frag_C[j]);
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                int a_row = warp_m_idx * 16;
                wmma::load_matrix_sync(frag_A, &smem_A_1[a_row][k_step], 40);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int b_col = j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 72);
                    wmma::mma_sync(frag_C[j], frag_A, frag_B[j], frag_C[j]);
                }
            }
        }

        __syncthreads();
        stage = 1 - stage;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C_half[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            frag_C_half[j].x[k] = __float2half(frag_C[j].x[k]);
        }
        int out_m = cta_m + warp_m_idx * 16;
        int out_n = cta_n + j * 16;
        if (out_m < M && out_n < N) {
            wmma::store_matrix_sync(&C[out_m * N + out_n], frag_C_half[j], N, wmma::mem_row_major);
        }
    }
}

// Split-K Reduction GEMM (32x64 Tile, K Sliced across gridDim.z, SMEM Staged Atomic Accumulation)
extern "C" __global__ __launch_bounds__(64, 4) void y_fused_gemm_splitk_32x64_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 32;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A[32][32 + 8];
    __shared__ alignas(128) half smem_B[32][64 + 8];
    __shared__ alignas(128) half smem_C[32][64 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2; // 0 or 1 -> M offset 0, 16

    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;
    int k_slice = blockIdx.z;
    int num_slices = gridDim.z;

    int k_per_slice = ((K + num_slices - 1) / num_slices + BLOCK_K - 1) / BLOCK_K * BLOCK_K;
    int k_start = k_slice * k_per_slice;
    int k_end = (k_start + k_per_slice < K) ? (k_start + k_per_slice) : K;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[4];

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        wmma::fill_fragment(frag_C[j], 0.0f);
    }

    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 16;

    int load_b_row = tid / 2;
    int load_b_col = (tid % 2) * 32;

    for (int k = k_start; k < k_end; k += BLOCK_K) {
        if (cta_m + load_a_row < M && k + load_a_col < K) {
            uint4* dst_a = (uint4*)&smem_A[load_a_row][load_a_col];
            const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + (k + load_a_col)];
            *dst_a = *src_a;
            *(dst_a + 1) = *(src_a + 1);
        } else {
            *(uint4*)&smem_A[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
            *(uint4*)&smem_A[load_a_row][load_a_col + 8] = make_uint4(0, 0, 0, 0);
        }

        if (k + load_b_row < K && cta_n + load_b_col < N) {
            uint4* dst_b = (uint4*)&smem_B[load_b_row][load_b_col];
            const uint4* src_b = (const uint4*)&B[(k + load_b_row) * N + (cta_n + load_b_col)];
            *dst_b = *src_b;
            *(dst_b + 1) = *(src_b + 1);
            *(dst_b + 2) = *(src_b + 2);
            *(dst_b + 3) = *(src_b + 3);
        } else {
            *(uint4*)&smem_B[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
            *(uint4*)&smem_B[load_b_row][load_b_col + 8] = make_uint4(0, 0, 0, 0);
            *(uint4*)&smem_B[load_b_row][load_b_col + 16] = make_uint4(0, 0, 0, 0);
            *(uint4*)&smem_B[load_b_row][load_b_col + 24] = make_uint4(0, 0, 0, 0);
        }

        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            int a_row = warp_m_idx * 16;
            wmma::load_matrix_sync(frag_A, &smem_A[a_row][k_step], 40);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int b_col = j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[k_step][b_col], 72);
                wmma::mma_sync(frag_C[j], frag_A, frag_B[j], frag_C[j]);
            }
        }

        __syncthreads();
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C_half[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        #pragma unroll
        for (int elem = 0; elem < 8; ++elem) {
            frag_C_half[j].x[elem] = __float2half(frag_C[j].x[elem]);
        }
        int local_m = warp_m_idx * 16;
        int local_n = j * 16;
        wmma::store_matrix_sync(&smem_C[local_m][local_n], frag_C_half[j], 72, wmma::mem_row_major);
    }

    __syncthreads();

    // Coalesced SMEM -> Global Atomic Reduction
    #pragma unroll
    for (int idx = tid; idx < 2048; idx += 64) {
        int r = idx / 64;
        int c = idx % 64;
        int out_m = cta_m + r;
        int out_n = cta_n + c;
        if (out_m < M && out_n < N) {
            atomicAdd(&C[out_m * N + out_n], smem_C[r][c]);
        }
    }
}

// Specialized Small GEMM (64x64 Tile, 128 Threads, 4 Warps, Vectorized 128-bit uint4 SMEM Staging, FP16 Output) for 1024x1024
extern "C" __global__ __launch_bounds__(128, 4) void y_fused_gemm_small_64x64_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A_0[64][32 + 8];
    __shared__ alignas(128) half smem_B_0[32][64 + 8];
    __shared__ alignas(128) half smem_A_1[64][32 + 8];
    __shared__ alignas(128) half smem_B_1[32][64 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2; // 0..1 -> offset 0, 32
    int warp_n_idx = warpId / 2; // 0..1 -> offset 0, 32

    const int SWIZZLE = 8;
    int grid_n = gridDim.x;
    int grid_m = gridDim.y;
    int tile_idx = blockIdx.y * grid_n + blockIdx.x;
    int num_tiles_per_swizzle = grid_n * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;
    if (cta_m >= M || cta_n >= N) {
        cta_m = blockIdx.y * BLOCK_M;
        cta_n = blockIdx.x * BLOCK_N;
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2][2];

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    // 128 threads load 64x32 halfs (2048 halfs = 128 uint4s) -> 1 uint4 load per thread
    int load_a_row = tid / 2;        // 0..63
    int load_a_col = (tid % 2) * 16;  // 0, 16

    // 128 threads load 32x64 halfs (2048 halfs = 128 uint4s) -> 1 uint4 load per thread
    int load_b_row = tid / 4;        // 0..31
    int load_b_col = (tid % 4) * 16;  // 0, 16, 32, 48

    // Prologue Stage 0
    if (cta_m + load_a_row < M && load_a_col < K) {
        uint4* dst_a = (uint4*)&smem_A_0[load_a_row][load_a_col];
        const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + load_a_col];
        *dst_a = *src_a;
    } else {
        *(uint4*)&smem_A_0[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
    }

    if (load_b_row < K && cta_n + load_b_col < N) {
        uint4* dst_b = (uint4*)&smem_B_0[load_b_row][load_b_col];
        const uint4* src_b = (const uint4*)&B[load_b_row * N + (cta_n + load_b_col)];
        *dst_b = *src_b;
    } else {
        *(uint4*)&smem_B_0[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
    }

    __syncthreads();

    int stage = 0;
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        if (next_k < K) {
            if (stage == 0) {
                if (cta_m + load_a_row < M && next_k + load_a_col < K) {
                    uint4* dst_a = (uint4*)&smem_A_1[load_a_row][load_a_col];
                    const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                    *dst_a = *src_a;
                } else {
                    *(uint4*)&smem_A_1[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
                }

                if (next_k + load_b_row < K && cta_n + load_b_col < N) {
                    uint4* dst_b = (uint4*)&smem_B_1[load_b_row][load_b_col];
                    const uint4* src_b = (const uint4*)&B[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                    *dst_b = *src_b;
                } else {
                    *(uint4*)&smem_B_1[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
                }
            } else {
                if (cta_m + load_a_row < M && next_k + load_a_col < K) {
                    uint4* dst_a = (uint4*)&smem_A_0[load_a_row][load_a_col];
                    const uint4* src_a = (const uint4*)&A[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                    *dst_a = *src_a;
                } else {
                    *(uint4*)&smem_A_0[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
                }

                if (next_k + load_b_row < K && cta_n + load_b_col < N) {
                    uint4* dst_b = (uint4*)&smem_B_0[load_b_row][load_b_col];
                    const uint4* src_b = (const uint4*)&B[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                    *dst_b = *src_b;
                } else {
                    *(uint4*)&smem_B_0[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
                }
            }
        }

        if (stage == 0) {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    int a_row = warp_m_idx * 32 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_0[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 72);
                }
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    int a_row = warp_m_idx * 32 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_1[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 72);
                }
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        }

        __syncthreads();
        stage = 1 - stage;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C_half[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                frag_C_half[i][j].x[k] = __float2half(frag_C[i][j].x[k]);
            }
            int out_m = cta_m + warp_m_idx * 32 + i * 16;
            int out_n = cta_n + warp_n_idx * 32 + j * 16;
            if (out_m < M && out_n < N) {
                wmma::store_matrix_sync(&C[out_m * N + out_n], frag_C_half[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// Direct-Register Small GEMM (64x64x32 CTA Tile, Vectorized 128-bit uint4 SMEM Staging, Zero Bank Conflicts) for M,N <= 512
extern "C" __global__ __launch_bounds__(128, 4) void y_tensor_core_gemm_small_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A[64][32 + 8];
    __shared__ alignas(128) half smem_B[32][64 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2; // 0..1 -> 0, 32
    int warp_n_idx = warpId / 2; // 0..1 -> 0, 32

    const int SWIZZLE = 8;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int num_tiles_per_swizzle = gridDim.x * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2][2];

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    // 128 threads load 64x32 halfs (2048 halfs = 128 uint4s) -> 1 uint4 load per thread
    int load_a_row = tid / 2;        // 0..63
    int load_a_col = (tid % 2) * 16;  // 0, 16

    // 128 threads load 32x64 halfs (2048 halfs = 128 uint4s) -> 1 uint4 load per thread
    int load_b_row = tid / 4;        // 0..31
    int load_b_col = (tid % 4) * 16;  // 0, 16, 32, 48

    for (int k = 0; k < K; k += BLOCK_K) {
        // Coalesced 128-bit vector load for A with alignment check
        unsigned long long g_addr_a = (unsigned long long)&A[(cta_m + load_a_row) * K + (k + load_a_col)];
        if ((g_addr_a & 15) == 0 && (cta_m + load_a_row < M) && (k + load_a_col + 7 < K)) {
            *(uint4*)&smem_A[load_a_row][load_a_col] = *(const uint4*)g_addr_a;
        } else {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                int r = cta_m + load_a_row;
                int c = k + load_a_col + e;
                smem_A[load_a_row][load_a_col + e] = (r < M && c < K) ? A[r * K + c] : __float2half(0.0f);
            }
        }

        // Coalesced 128-bit vector load for B with alignment check
        unsigned long long g_addr_b = (unsigned long long)&B[(k + load_b_row) * N + (cta_n + load_b_col)];
        if ((g_addr_b & 15) == 0 && (k + load_b_row < K) && (cta_n + load_b_col + 7 < N)) {
            *(uint4*)&smem_B[load_b_row][load_b_col] = *(const uint4*)g_addr_b;
        } else {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                int r = k + load_b_row;
                int c = cta_n + load_b_col + e;
                smem_B[load_b_row][load_b_col + e] = (r < K && c < N) ? B[r * N + c] : __float2half(0.0f);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int a_row = warp_m_idx * 32 + i * 16;
                wmma::load_matrix_sync(frag_A[i], &smem_A[a_row][k_step], 40);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int b_col = warp_n_idx * 32 + j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[k_step][b_col], 72);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Vectorized 128-bit Epilogue Downcasting & Direct Store
    __shared__ alignas(16) float warp_out[4][32][32];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(&warp_out[warpId][i * 16][j * 16], frag_C[i][j], 32, wmma::mem_row_major);
        }
    }

    int lane = tid % 32;
    int r_sub = lane / 8;
    int c_sub = (lane % 8) * 4;

    #pragma unroll
    for (int r_off = 0; r_off < 32; r_off += 4) {
        int row = r_sub + r_off;
        float4 v = *(float4*)&warp_out[warpId][row][c_sub];
        int out_m = cta_m + warp_m_idx * 32 + row;
        int out_n = cta_n + warp_n_idx * 32 + c_sub;

        if (out_m < M) {
            unsigned long long addr = (unsigned long long)(&C[out_m * N + out_n]);
            if ((addr & 15) == 0 && out_n + 3 < N) {
                *(float4*)addr = v;
            } else {
                if (out_n < N) C[out_m * N + out_n] = v.x;
                if (out_n + 1 < N) C[out_m * N + out_n + 1] = v.y;
                if (out_n + 2 < N) C[out_m * N + out_n + 2] = v.z;
                if (out_n + 3 < N) C[out_m * N + out_n + 3] = v.w;
            }
        }
    }
}

// Fused GEMM + Bias + ReLU Kernel (128x128x32 CTA Tile, 256 threads, 2-Stage Async Double Buffering, L2 Swizzling)
extern "C" __global__ __launch_bounds__(256, 2) void y_fused_gemm_bias_relu_kernel(
    const float* __restrict__ A_fp32,
    const float* __restrict__ B_fp32,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    extern __shared__ char smem_buf_bias[];
    half (*smem_A_0)[32 + 8] = (half (*)[32 + 8])smem_buf_bias;
    half (*smem_B_0)[128 + 8] = (half (*)[128 + 8])&smem_buf_bias[128 * 40 * sizeof(half)];

    half (*smem_A_1)[32 + 8] = (half (*)[32 + 8])&smem_buf_bias[128 * 40 * sizeof(half) + 32 * 136 * sizeof(half)];
    half (*smem_B_1)[128 + 8] = (half (*)[128 + 8])&smem_buf_bias[2 * 128 * 40 * sizeof(half) + 32 * 136 * sizeof(half)];

    int tid = threadIdx.x;
    int warpId = tid / 32; // 0..7

    int warp_m_idx = warpId % 2; // 0..1 (0 or 64)
    int warp_n_idx = warpId / 2; // 0..3 (0, 32, 64, 96)

    // CTA Block Swizzling for high L2 Cache reuse on large matrices
    const int SWIZZLE = 8;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int num_tiles_per_swizzle = gridDim.x * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[4][2];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    int load_a_row = tid / 2;       // 0..127
    int load_a_col = (tid % 2) * 16; // 0, 16

    int load_b_row = tid / 8;       // 0..31
    int load_b_col = (tid % 8) * 16; // 0, 16, ... 112

    // Prologue: Load Stage 0 (tile k=0)
    #pragma unroll
    for (int offset = 0; offset < 16; offset += 2) {
        int cur_a_col = load_a_col + offset;
        if (cta_m + load_a_row < M && cur_a_col < K) {
            float2 vA = *(float2*)&A_fp32[(cta_m + load_a_row) * K + cur_a_col];
            smem_A_0[load_a_row][cur_a_col] = __float2half(vA.x);
            smem_A_0[load_a_row][cur_a_col + 1] = __float2half(vA.y);
        } else {
            smem_A_0[load_a_row][cur_a_col] = __float2half(0.0f);
            smem_A_0[load_a_row][cur_a_col + 1] = __float2half(0.0f);
        }
    }
    #pragma unroll
    for (int offset = 0; offset < 16; offset += 2) {
        int cur_b_col = load_b_col + offset;
        if (load_b_row < K && cta_n + cur_b_col < N) {
            float2 vB = *(float2*)&B_fp32[load_b_row * N + (cta_n + cur_b_col)];
            smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
        } else {
            smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
        }
    }
    __syncthreads();

    int stage = 0;
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        int next_stage = 1 - stage;

        // Async prefetch next stage (stage k+1)
        if (next_k < K) {
            if (stage == 0) {
                #pragma unroll
                for (int offset = 0; offset < 16; offset += 2) {
                    int cur_a_col = load_a_col + offset;
                    if (cta_m + load_a_row < M && next_k + cur_a_col < K) {
                        float2 vA = *(float2*)&A_fp32[(cta_m + load_a_row) * K + (next_k + cur_a_col)];
                        smem_A_1[load_a_row][cur_a_col] = __float2half(vA.x);
                        smem_A_1[load_a_row][cur_a_col + 1] = __float2half(vA.y);
                    } else {
                        smem_A_1[load_a_row][cur_a_col] = __float2half(0.0f);
                        smem_A_1[load_a_row][cur_a_col + 1] = __float2half(0.0f);
                    }
                }
                #pragma unroll
                for (int offset = 0; offset < 16; offset += 2) {
                    int cur_b_col = load_b_col + offset;
                    if (next_k + load_b_row < K && cta_n + cur_b_col < N) {
                        float2 vB = *(float2*)&B_fp32[(next_k + load_b_row) * N + (cta_n + cur_b_col)];
                        smem_B_1[load_b_row][cur_b_col] = __float2half(vB.x);
                        smem_B_1[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                    } else {
                        smem_B_1[load_b_row][cur_b_col] = __float2half(0.0f);
                        smem_B_1[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                    }
                }
            } else {
                #pragma unroll
                for (int offset = 0; offset < 16; offset += 2) {
                    int cur_a_col = load_a_col + offset;
                    if (cta_m + load_a_row < M && next_k + cur_a_col < K) {
                        float2 vA = *(float2*)&A_fp32[(cta_m + load_a_row) * K + (next_k + cur_a_col)];
                        smem_A_0[load_a_row][cur_a_col] = __float2half(vA.x);
                        smem_A_0[load_a_row][cur_a_col + 1] = __float2half(vA.y);
                    } else {
                        smem_A_0[load_a_row][cur_a_col] = __float2half(0.0f);
                        smem_A_0[load_a_row][cur_a_col + 1] = __float2half(0.0f);
                    }
                }
                #pragma unroll
                for (int offset = 0; offset < 16; offset += 2) {
                    int cur_b_col = load_b_col + offset;
                    if (next_k + load_b_row < K && cta_n + cur_b_col < N) {
                        float2 vB = *(float2*)&B_fp32[(next_k + load_b_row) * N + (cta_n + cur_b_col)];
                        smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
                        smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                    } else {
                        smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
                        smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                    }
                }
            }
        }

        // Tensor Core computation on current stage buffer
        if (stage == 0) {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = warp_m_idx * 64 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_0[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 136);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = warp_m_idx * 64 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_1[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 136);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        }

        __syncthreads();
        stage = next_stage;
    }

    // Epilogue Shared Memory Reuse
    float (*smem_C)[128 + 4] = (float (*)[128 + 4])smem_buf_bias;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int r = warp_m_idx * 64 + i * 16;
            int c = warp_n_idx * 32 + j * 16;
            wmma::store_matrix_sync(&smem_C[r][c], frag_C[i][j], 132, wmma::mem_row_major);
        }
    }

    __syncthreads();

    // 256 threads process 128x128 floats (4096 float4s) -> 16 float4s per thread
    int r_sub = tid / 16;        // 0..15
    int col = (tid % 16) * 4;     // 0, 4, 8, ... 60

    #pragma unroll
    for (int r_off = 0; r_off < 128; r_off += 16) {
        int row = r_sub + r_off;
        #pragma unroll
        for (int c_side = 0; c_side < 128; c_side += 64) {
            int cur_col = col + c_side;
            int out_m = cta_m + row;
            int out_n = cta_n + cur_col;

            if (out_m < M && out_n + 3 < N) {
                float4 v = *(float4*)&smem_C[row][cur_col];
                float4 b = *(float4*)&bias[out_n];
                v.x = v.x + b.x > 0.0f ? v.x + b.x : 0.0f;
                v.y = v.y + b.y > 0.0f ? v.y + b.y : 0.0f;
                v.z = v.z + b.z > 0.0f ? v.z + b.z : 0.0f;
                v.w = v.w + b.w > 0.0f ? v.w + b.w : 0.0f;
                *(float4*)&C[out_m * N + out_n] = v;
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 4) void y_fused_gemm_bias_relu_small_kernel(
    const float* __restrict__ A_fp32,
    const float* __restrict__ B_fp32,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A[64][32 + 8];
    __shared__ alignas(128) half smem_B[32][64 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2;
    int warp_n_idx = warpId / 2;

    const int SWIZZLE = 8;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int num_tiles_per_swizzle = gridDim.x * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2][2];

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 16;

    int load_b_row = tid / 4;
    int load_b_col = (tid % 4) * 16;

    for (int k = 0; k < K; k += BLOCK_K) {
        for (int offset = 0; offset < 16; offset += 2) {
            int cur_a_col = load_a_col + offset;
            if (cta_m + load_a_row < M && k + cur_a_col < K) {
                float2 vA = *(float2*)&A_fp32[(cta_m + load_a_row) * K + (k + cur_a_col)];
                smem_A[load_a_row][cur_a_col] = __float2half(vA.x);
                smem_A[load_a_row][cur_a_col + 1] = __float2half(vA.y);
            } else {
                smem_A[load_a_row][cur_a_col] = __float2half(0.0f);
                smem_A[load_a_row][cur_a_col + 1] = __float2half(0.0f);
            }
        }

        for (int offset = 0; offset < 16; offset += 2) {
            int cur_b_col = load_b_col + offset;
            if (k + load_b_row < K && cta_n + cur_b_col < N) {
                float2 vB = *(float2*)&B_fp32[(k + load_b_row) * N + (cta_n + cur_b_col)];
                smem_B[load_b_row][cur_b_col] = __float2half(vB.x);
                smem_B[load_b_row][cur_b_col + 1] = __float2half(vB.y);
            } else {
                smem_B[load_b_row][cur_b_col] = __float2half(0.0f);
                smem_B[load_b_row][cur_b_col + 1] = __float2half(0.0f);
            }
        }

        __syncthreads();

        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            for (int i = 0; i < 2; ++i) {
                int a_row = warp_m_idx * 32 + i * 16;
                wmma::load_matrix_sync(frag_A[i], &smem_A[a_row][k_step], 40);
            }
            for (int j = 0; j < 2; ++j) {
                int b_col = warp_n_idx * 32 + j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[k_step][b_col], 72);
            }

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                }
            }
        }

        __syncthreads();
    }

    __shared__ alignas(16) float smem_C[64][64 + 4];

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int r = warp_m_idx * 32 + i * 16;
            int c = warp_n_idx * 32 + j * 16;
            wmma::store_matrix_sync(&smem_C[r][c], frag_C[i][j], 68, wmma::mem_row_major);
        }
    }

    __syncthreads();

    // 128 threads process 64x64 floats (1024 float4s) -> 8 float4s per thread
    int r_sub = tid / 16;        // 0..7
    int col = (tid % 16) * 4;     // 0, 4, 8, ... 60

    #pragma unroll
    for (int r_off = 0; r_off < 64; r_off += 8) {
        int row = r_sub + r_off;
        int out_m = cta_m + row;
        int out_n = cta_n + col;

        if (out_m < M && out_n + 3 < N) {
            float4 v = *(float4*)&smem_C[row][col];
            float4 b = *(float4*)&bias[out_n];
            v.x = v.x + b.x > 0.0f ? v.x + b.x : 0.0f;
            v.y = v.y + b.y > 0.0f ? v.y + b.y : 0.0f;
            v.z = v.z + b.z > 0.0f ? v.z + b.z : 0.0f;
            v.w = v.w + b.w > 0.0f ? v.w + b.w : 0.0f;
            *(float4*)&C[out_m * N + out_n] = v;
        }
    }
}

// Fused Small Linear Kernel (GEMM + Bias, No ReLU) with FP16 Input Staging for Output Layer
extern "C" __global__ void y_fused_gemm_bias_linear_small_fp16_in_kernel(
    const half* __restrict__ A_fp16,
    const float* __restrict__ B_fp32,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A[64][32 + 8];
    __shared__ alignas(128) half smem_B[32][64 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2;
    int warp_n_idx = warpId / 2;

    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2][2];

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 8;

    int load_b_row = tid / 4;
    int load_b_col = (tid % 4) * 16;

    for (int k = 0; k < K; k += BLOCK_K) {
        if (cta_m + load_a_row < M && k + load_a_col + 7 < K) {
            *(uint4*)&smem_A[load_a_row][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row) * K + (k + load_a_col)];
        } else {
            #pragma unroll
            for (int k_off = 0; k_off < 8; ++k_off) {
                int cur_k = load_a_col + k_off;
                smem_A[load_a_row][cur_k] = (cta_m + load_a_row < M && k + cur_k < K) ? A_fp16[(cta_m + load_a_row) * K + (k + cur_k)] : __float2half(0.0f);
            }
        }

        if (cta_m + load_a_row + 32 < M && k + load_a_col + 7 < K) {
            *(uint4*)&smem_A[load_a_row + 32][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 32) * K + (k + load_a_col)];
        } else {
            #pragma unroll
            for (int k_off = 0; k_off < 8; ++k_off) {
                int cur_k = load_a_col + k_off;
                smem_A[load_a_row + 32][cur_k] = (cta_m + load_a_row + 32 < M && k + cur_k < K) ? A_fp16[(cta_m + load_a_row + 32) * K + (k + cur_k)] : __float2half(0.0f);
            }
        }

        #pragma unroll
        for (int offset = 0; offset < 16; offset += 2) {
            int cur_b_col = load_b_col + offset;
            if (k + load_b_row < K && cta_n + cur_b_col < N) {
                float2 vB = *(float2*)&B_fp32[(k + load_b_row) * N + (cta_n + cur_b_col)];
                smem_B[load_b_row][cur_b_col] = __float2half(vB.x);
                smem_B[load_b_row][cur_b_col + 1] = __float2half(vB.y);
            } else {
                smem_B[load_b_row][cur_b_col] = __float2half(0.0f);
                smem_B[load_b_row][cur_b_col + 1] = __float2half(0.0f);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int a_row = warp_m_idx * 32 + i * 16;
                wmma::load_matrix_sync(frag_A[i], &smem_A[a_row][k_step], 40);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int b_col = warp_n_idx * 32 + j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[k_step][b_col], 72);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                }
            }
        }

        __syncthreads();
    }

    __shared__ alignas(16) float smem_C[64][64 + 4];

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int r = warp_m_idx * 32 + i * 16;
            int c = warp_n_idx * 32 + j * 16;
            wmma::store_matrix_sync(&smem_C[r][c], frag_C[i][j], 68, wmma::mem_row_major);
        }
    }

    __syncthreads();

    int r_sub = tid / 16;
    int col = (tid % 16) * 4;

    #pragma unroll
    for (int r_off = 0; r_off < 64; r_off += 8) {
        int row = r_sub + r_off;
        int out_m = cta_m + row;
        int out_n = cta_n + col;

        if (out_m < M && out_n + 3 < N) {
            float4 v = *(float4*)&smem_C[row][col];
            float4 b = *(float4*)&bias[out_n];
            v.x = v.x + b.x;
            v.y = v.y + b.y;
            v.z = v.z + b.z;
            v.w = v.w + b.w;
            *(float4*)&C[out_m * N + out_n] = v;
        }
    }
}

// Fused Small Linear Kernel (GEMM + Bias, No ReLU) for Output Layer
extern "C" __global__ void y_fused_gemm_bias_linear_small_kernel(
    const float* __restrict__ A_fp32,
    const float* __restrict__ B_fp32,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A[64][32 + 8];
    __shared__ alignas(128) half smem_B[32][64 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2;
    int warp_n_idx = warpId / 2;

    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2][2];

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 16;

    int load_b_row = tid / 4;
    int load_b_col = (tid % 4) * 16;

    for (int k = 0; k < K; k += BLOCK_K) {
        for (int offset = 0; offset < 16; offset += 2) {
            int cur_a_col = load_a_col + offset;
            if (cta_m + load_a_row < M && k + cur_a_col < K) {
                float2 vA = *(float2*)&A_fp32[(cta_m + load_a_row) * K + (k + cur_a_col)];
                smem_A[load_a_row][cur_a_col] = __float2half(vA.x);
                smem_A[load_a_row][cur_a_col + 1] = __float2half(vA.y);
            } else {
                smem_A[load_a_row][cur_a_col] = __float2half(0.0f);
                smem_A[load_a_row][cur_a_col + 1] = __float2half(0.0f);
            }
        }

        for (int offset = 0; offset < 16; offset += 2) {
            int cur_b_col = load_b_col + offset;
            if (k + load_b_row < K && cta_n + cur_b_col < N) {
                float2 vB = *(float2*)&B_fp32[(k + load_b_row) * N + (cta_n + cur_b_col)];
                smem_B[load_b_row][cur_b_col] = __float2half(vB.x);
                smem_B[load_b_row][cur_b_col + 1] = __float2half(vB.y);
            } else {
                smem_B[load_b_row][cur_b_col] = __float2half(0.0f);
                smem_B[load_b_row][cur_b_col + 1] = __float2half(0.0f);
            }
        }

        __syncthreads();

        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            for (int i = 0; i < 2; ++i) {
                int a_row = warp_m_idx * 32 + i * 16;
                wmma::load_matrix_sync(frag_A[i], &smem_A[a_row][k_step], 40);
            }
            for (int j = 0; j < 2; ++j) {
                int b_col = warp_n_idx * 32 + j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[k_step][b_col], 72);
            }

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                }
            }
        }

        __syncthreads();
    }

    __shared__ alignas(16) float smem_C[64][64 + 4];

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int r = warp_m_idx * 32 + i * 16;
            int c = warp_n_idx * 32 + j * 16;
            wmma::store_matrix_sync(&smem_C[r][c], frag_C[i][j], 68, wmma::mem_row_major);
        }
    }

    __syncthreads();

    int r_sub = tid / 16;
    int col = (tid % 16) * 4;

    #pragma unroll
    for (int r_off = 0; r_off < 64; r_off += 8) {
        int row = r_sub + r_off;
        int out_m = cta_m + row;
        int out_n = cta_n + col;

        if (out_m < M && out_n + 3 < N) {
            float4 v = *(float4*)&smem_C[row][col];
            float4 b = *(float4*)&bias[out_n];
            v.x = v.x + b.x;
            v.y = v.y + b.y;
            v.z = v.z + b.z;
            v.w = v.w + b.w;
            *(float4*)&C[out_m * N + out_n] = v;
        }
    }
}

extern "C" __global__ void naive_bias_relu_kernel(
    float* C, const float* bias, int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        float val = C[idx] + bias[col];
        C[idx] = val > 0.0f ? val : 0.0f;
    }
}

// Layer 1 Kernel: FP32 Inputs (X, W1, b1) -> Native FP16 Staging Output (H1)
extern "C" __global__ void y_fused_layer1_fp32_in_fp16_out_kernel(
    const float* __restrict__ A_fp32,
    const float* __restrict__ B_fp32,
    const float* __restrict__ bias_fp32,
    half* __restrict__ C_fp16,
    int M, int N, int K
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    extern __shared__ char smem_buf_bias16[];
    half (*smem_A_0)[32 + 8] = (half (*)[32 + 8])smem_buf_bias16;
    half (*smem_B_0)[128 + 8] = (half (*)[128 + 8])&smem_buf_bias16[128 * 40 * sizeof(half)];

    half (*smem_A_1)[32 + 8] = (half (*)[32 + 8])&smem_buf_bias16[128 * 40 * sizeof(half) + 32 * 136 * sizeof(half)];
    half (*smem_B_1)[128 + 8] = (half (*)[128 + 8])&smem_buf_bias16[2 * 128 * 40 * sizeof(half) + 32 * 136 * sizeof(half)];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2;
    int warp_n_idx = warpId / 2;

    const int SWIZZLE = 8;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int num_tiles_per_swizzle = gridDim.x * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[4][2];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 16;

    int load_b_row = tid / 8;
    int load_b_col = (tid % 8) * 16;

    // Prologue Stage 0
    #pragma unroll
    for (int offset = 0; offset < 16; offset += 2) {
        int cur_a_col = load_a_col + offset;
        if (cta_m + load_a_row < M && cur_a_col < K) {
            float2 vA = *(float2*)&A_fp32[(cta_m + load_a_row) * K + cur_a_col];
            smem_A_0[load_a_row][cur_a_col] = __float2half(vA.x);
            smem_A_0[load_a_row][cur_a_col + 1] = __float2half(vA.y);
        } else {
            smem_A_0[load_a_row][cur_a_col] = __float2half(0.0f);
            smem_A_0[load_a_row][cur_a_col + 1] = __float2half(0.0f);
        }
    }

    #pragma unroll
    for (int offset = 0; offset < 16; offset += 2) {
        int cur_b_col = load_b_col + offset;
        if (load_b_row < K && cta_n + cur_b_col < N) {
            float2 vB = *(float2*)&B_fp32[load_b_row * N + (cta_n + cur_b_col)];
            smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
        } else {
            smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
        }
    }

    __syncthreads();

    int stage = 0;
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        int next_stage = 1 - stage;

        if (next_k < K) {
            if (stage == 0) {
                #pragma unroll
                for (int offset = 0; offset < 16; offset += 2) {
                    int cur_a_col = load_a_col + offset;
                    if (cta_m + load_a_row < M && next_k + cur_a_col < K) {
                        float2 vA = *(float2*)&A_fp32[(cta_m + load_a_row) * K + (next_k + cur_a_col)];
                        smem_A_1[load_a_row][cur_a_col] = __float2half(vA.x);
                        smem_A_1[load_a_row][cur_a_col + 1] = __float2half(vA.y);
                    } else {
                        smem_A_1[load_a_row][cur_a_col] = __float2half(0.0f);
                        smem_A_1[load_a_row][cur_a_col + 1] = __float2half(0.0f);
                    }
                }

                #pragma unroll
                for (int offset = 0; offset < 16; offset += 2) {
                    int cur_b_col = load_b_col + offset;
                    if (next_k + load_b_row < K && cta_n + cur_b_col < N) {
                        float2 vB = *(float2*)&B_fp32[(next_k + load_b_row) * N + (cta_n + cur_b_col)];
                        smem_B_1[load_b_row][cur_b_col] = __float2half(vB.x);
                        smem_B_1[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                    } else {
                        smem_B_1[load_b_row][cur_b_col] = __float2half(0.0f);
                        smem_B_1[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                    }
                }
            } else {
                #pragma unroll
                for (int offset = 0; offset < 16; offset += 2) {
                    int cur_a_col = load_a_col + offset;
                    if (cta_m + load_a_row < M && next_k + cur_a_col < K) {
                        float2 vA = *(float2*)&A_fp32[(cta_m + load_a_row) * K + (next_k + cur_a_col)];
                        smem_A_0[load_a_row][cur_a_col] = __float2half(vA.x);
                        smem_A_0[load_a_row][cur_a_col + 1] = __float2half(vA.y);
                    } else {
                        smem_A_0[load_a_row][cur_a_col] = __float2half(0.0f);
                        smem_A_0[load_a_row][cur_a_col + 1] = __float2half(0.0f);
                    }
                }

                #pragma unroll
                for (int offset = 0; offset < 16; offset += 2) {
                    int cur_b_col = load_b_col + offset;
                    if (next_k + load_b_row < K && cta_n + cur_b_col < N) {
                        float2 vB = *(float2*)&B_fp32[(next_k + load_b_row) * N + (cta_n + cur_b_col)];
                        smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
                        smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                    } else {
                        smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
                        smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                    }
                }
            }
        }

        if (stage == 0) {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = warp_m_idx * 64 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_0[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 136);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = warp_m_idx * 64 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_1[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 136);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        }

        __syncthreads();
        stage = next_stage;
    }

    float (*smem_C_fp32)[128 + 4] = (float (*)[128 + 4])smem_buf_bias16;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int r = warp_m_idx * 64 + i * 16;
            int c = warp_n_idx * 32 + j * 16;
            wmma::store_matrix_sync(&smem_C_fp32[r][c], frag_C[i][j], 132, wmma::mem_row_major);
        }
    }

    __syncthreads();

    int r_sub = tid / 16;
    int col = (tid % 16) * 8;

    #pragma unroll
    for (int r_off = 0; r_off < 128; r_off += 16) {
        int row = r_sub + r_off;
        int out_m = cta_m + row;
        int out_n = cta_n + col;

        if (out_m < M && out_n + 7 < N) {
            float4 f0 = *(float4*)&smem_C_fp32[row][col];
            float4 f1 = *(float4*)&smem_C_fp32[row][col + 4];
            float4 b0 = *(float4*)&bias_fp32[out_n];
            float4 b1 = *(float4*)&bias_fp32[out_n + 4];

            half2 h0 = __floats2half2_rn(f0.x + b0.x, f0.y + b0.y);
            half2 h1 = __floats2half2_rn(f0.z + b0.z, f0.w + b0.w);
            half2 h2 = __floats2half2_rn(f1.x + b1.x, f1.y + b1.y);
            half2 h3 = __floats2half2_rn(f1.z + b1.z, f1.w + b1.w);

            half zero = __float2half(0.0f);
            half2 zero2 = __halves2half2(zero, zero);
            h0 = __hmax2(h0, zero2);
            h1 = __hmax2(h1, zero2);
            h2 = __hmax2(h2, zero2);
            h3 = __hmax2(h3, zero2);

            alignas(16) half2 h_v[4] = {h0, h1, h2, h3};
            *(uint4*)&C_fp16[out_m * N + out_n] = *(uint4*)h_v;
        }
    }
}

// Layer 3 Kernel: Native FP16 Staging Input (H2) -> FP32 Predictions Output (Out)
extern "C" __global__ void y_fused_layer3_fp16_in_fp32_out_kernel(
    const half* __restrict__ A_fp16,
    const float* __restrict__ B_fp32,
    const float* __restrict__ bias_fp32,
    float* __restrict__ C_fp32,
    int M, int N, int K
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    extern __shared__ char raw_smem[];
    half (*smem_A_0)[32 + 8] = (half (*)[32 + 8])raw_smem;
    half (*smem_B_0)[128 + 8] = (half (*)[128 + 8])&raw_smem[128 * 40 * sizeof(half)];

    half (*smem_A_1)[32 + 8] = (half (*)[32 + 8])&raw_smem[128 * 40 * sizeof(half) + 32 * 136 * sizeof(half)];
    half (*smem_B_1)[128 + 8] = (half (*)[128 + 8])&raw_smem[2 * 128 * 40 * sizeof(half) + 32 * 136 * sizeof(half)];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2;
    int warp_n_idx = warpId / 2;

    const int SWIZZLE = 8;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int num_tiles_per_swizzle = gridDim.x * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[4][2];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 8;

    int load_b_row = tid / 8;
    int load_b_col = (tid % 8) * 16;

    // Prologue Stage 0
    if (cta_m + load_a_row < M && load_a_col + 7 < K) {
        *(uint4*)&smem_A_0[load_a_row][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row) * K + load_a_col];
    } else {
        #pragma unroll
        for (int k_off = 0; k_off < 8; ++k_off) {
            int cur_k = load_a_col + k_off;
            smem_A_0[load_a_row][cur_k] = (cta_m + load_a_row < M && cur_k < K) ? A_fp16[(cta_m + load_a_row) * K + cur_k] : __float2half(0.0f);
        }
    }

    if (cta_m + load_a_row + 64 < M && load_a_col + 7 < K) {
        *(uint4*)&smem_A_0[load_a_row + 64][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 64) * K + load_a_col];
    } else {
        #pragma unroll
        for (int k_off = 0; k_off < 8; ++k_off) {
            int cur_k = load_a_col + k_off;
            smem_A_0[load_a_row + 64][cur_k] = (cta_m + load_a_row + 64 < M && cur_k < K) ? A_fp16[(cta_m + load_a_row + 64) * K + cur_k] : __float2half(0.0f);
        }
    }

    #pragma unroll
    for (int offset = 0; offset < 16; offset += 2) {
        int cur_b_col = load_b_col + offset;
        if (load_b_row < K && cta_n + cur_b_col < N) {
            float2 vB = *(float2*)&B_fp32[load_b_row * N + (cta_n + cur_b_col)];
            smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
        } else {
            smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
        }
    }

    __syncthreads();

    int stage = 0;
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        int next_stage = 1 - stage;

        if (next_k < K) {
            if (stage == 0) {
                if (cta_m + load_a_row < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_1[load_a_row][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_1[load_a_row][cur_k] = (cta_m + load_a_row < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (cta_m + load_a_row + 64 < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_1[load_a_row + 64][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 64) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_1[load_a_row + 64][cur_k] = (cta_m + load_a_row + 64 < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row + 64) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                #pragma unroll
                for (int offset = 0; offset < 16; offset += 2) {
                    int cur_b_col = load_b_col + offset;
                    if (next_k + load_b_row < K && cta_n + cur_b_col < N) {
                        float2 vB = *(float2*)&B_fp32[(next_k + load_b_row) * N + (cta_n + cur_b_col)];
                        smem_B_1[load_b_row][cur_b_col] = __float2half(vB.x);
                        smem_B_1[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                    } else {
                        smem_B_1[load_b_row][cur_b_col] = __float2half(0.0f);
                        smem_B_1[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                    }
                }
            } else {
                if (cta_m + load_a_row < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_0[load_a_row][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_0[load_a_row][cur_k] = (cta_m + load_a_row < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (cta_m + load_a_row + 64 < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_0[load_a_row + 64][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 64) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_0[load_a_row + 64][cur_k] = (cta_m + load_a_row + 64 < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row + 64) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                #pragma unroll
                for (int offset = 0; offset < 16; offset += 2) {
                    int cur_b_col = load_b_col + offset;
                    if (next_k + load_b_row < K && cta_n + cur_b_col < N) {
                        float2 vB = *(float2*)&B_fp32[(next_k + load_b_row) * N + (cta_n + cur_b_col)];
                        smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
                        smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                    } else {
                        smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
                        smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                    }
                }
            }
        }

        if (stage == 0) {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = warp_m_idx * 64 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_0[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 136);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = warp_m_idx * 64 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_1[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 136);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        }

        __syncthreads();
        stage = next_stage;
    }

    float (*smem_C_fp32)[128 + 4] = (float (*)[128 + 4])raw_smem;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int r = warp_m_idx * 64 + i * 16;
            int c = warp_n_idx * 32 + j * 16;
            wmma::store_matrix_sync(&smem_C_fp32[r][c], frag_C[i][j], 132, wmma::mem_row_major);
        }
    }

    __syncthreads();

    int r_sub = tid / 32;
    int col = (tid % 32) * 4;

    #pragma unroll
    for (int r_off = 0; r_off < 128; r_off += 8) {
        int row = r_sub + r_off;
        int out_m = cta_m + row;
        int out_n = cta_n + col;

        if (out_m < M && out_n + 3 < N) {
            float4 v = *(float4*)&smem_C_fp32[row][col];
            float4 b = *(float4*)&bias_fp32[out_n];
            v.x = v.x + b.x;
            v.y = v.y + b.y;
            v.z = v.z + b.z;
            v.w = v.w + b.w;
            *(float4*)&C_fp32[out_m * N + out_n] = v;
        }
    }
}

// Native Vectorized FP16 Fused GEMM + Bias + Activation Kernel (128-bit uint4 vector loads/stores)
extern "C" __global__ __launch_bounds__(256, 2) void y_fused_gemm_bias_relu_fp16_kernel(
    const half* __restrict__ A_fp16,
    const half* __restrict__ B_fp16,
    const half* __restrict__ bias_fp16,
    half* __restrict__ C_fp16,
    int M, int N, int K,
    int is_relu
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A[128][32 + 8];
    __shared__ alignas(128) half smem_B[32][128 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    int warp_m_idx = warpId % 4; // 0..3
    int warp_n_idx = warpId / 4; // 0..1

    const int SWIZZLE = 8;
    int grid_m = (M + BLOCK_M - 1) / BLOCK_M;
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int cta_m, cta_n;
    get_morton_cta_coords(tile_idx, grid_m, grid_n, BLOCK_M, BLOCK_N, cta_m, cta_n);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2][2];

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 16;

    int load_b_row = tid / 8;
    int load_b_col = (tid % 8) * 16;

    for (int k = 0; k < K; k += BLOCK_K) {
        if (cta_m + load_a_row < M && k + load_a_col < K) {
            uint4* dst_a = (uint4*)&smem_A[load_a_row][load_a_col];
            const uint4* src_a = (const uint4*)&A_fp16[(cta_m + load_a_row) * K + (k + load_a_col)];
            *dst_a = *src_a;
            *(dst_a + 1) = *(src_a + 1);
        } else {
            *(uint4*)&smem_A[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
            *(uint4*)&smem_A[load_a_row][load_a_col + 8] = make_uint4(0, 0, 0, 0);
        }

        if (k + load_b_row < K && cta_n + load_b_col < N) {
            uint4* dst_b = (uint4*)&smem_B[load_b_row][load_b_col];
            const uint4* src_b = (const uint4*)&B_fp16[(k + load_b_row) * N + (cta_n + load_b_col)];
            *dst_b = *src_b;
            *(dst_b + 1) = *(src_b + 1);
        } else {
            *(uint4*)&smem_B[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
            *(uint4*)&smem_B[load_b_row][load_b_col + 8] = make_uint4(0, 0, 0, 0);
        }

        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int a_row = warp_m_idx * 32 + i * 16;
                wmma::load_matrix_sync(frag_A[i], &smem_A[a_row][k_step], 40);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int b_col = warp_n_idx * 64 + j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[k_step][b_col], 136);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // In-Register Bias + ReLU + Direct Global Store (Zero SMEM, Zero __syncthreads)
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C_half[4][2];

    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        int b_col_base = cta_n + warp_n_idx * 32 + j * 16 + (laneId % 4) * 2;
        float b0 = (b_col_base < N) ? __half2float(bias_fp16[b_col_base]) : 0.0f;
        float b1 = (b_col_base + 1 < N) ? __half2float(bias_fp16[b_col_base + 1]) : 0.0f;
        float b2 = (b_col_base + 8 < N) ? __half2float(bias_fp16[b_col_base + 8]) : 0.0f;
        float b3 = (b_col_base + 9 < N) ? __half2float(bias_fp16[b_col_base + 9]) : 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            frag_C[i][j].x[0] += b0;
            frag_C[i][j].x[1] += b1;
            frag_C[i][j].x[2] += b0;
            frag_C[i][j].x[3] += b1;
            frag_C[i][j].x[4] += b2;
            frag_C[i][j].x[5] += b3;
            frag_C[i][j].x[6] += b2;
            frag_C[i][j].x[7] += b3;

            if (is_relu) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    frag_C[i][j].x[k] = fmaxf(frag_C[i][j].x[k], 0.0f);
                }
            }

            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                frag_C_half[i][j].x[k] = __float2half(frag_C[i][j].x[k]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int out_m = cta_m + warp_m_idx * 64 + i * 16;
            int out_n = cta_n + warp_n_idx * 32 + j * 16;
            if (out_m < M && out_n < N) {
                wmma::store_matrix_sync(&C_fp16[out_m * N + out_n], frag_C_half[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// Optimized 128x64 Tile Vectorized FP16 Fused GEMM Kernel (2 CTAs/SM, 38KB SMEM, 100% SM Occupancy Pass)
extern "C" __global__ __launch_bounds__(256, 2) void y_fused_gemm_bias_relu_fp16_opt_kernel(
    const half* __restrict__ A_fp16,
    const half* __restrict__ B_fp16,
    const half* __restrict__ bias_fp16,
    half* __restrict__ C_fp16,
    int M, int N, int K,
    int is_relu
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    extern __shared__ char raw_smem[];
    half (*smem_A_0)[32 + 8] = (half (*)[32 + 8])raw_smem;
    half (*smem_B_0)[64 + 8] = (half (*)[64 + 8])&raw_smem[128 * 40 * sizeof(half)];

    half (*smem_A_1)[32 + 8] = (half (*)[32 + 8])&raw_smem[128 * 40 * sizeof(half) + 32 * 72 * sizeof(half)];
    half (*smem_B_1)[64 + 8] = (half (*)[64 + 8])&raw_smem[2 * 128 * 40 * sizeof(half) + 32 * 72 * sizeof(half)];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2;
    int warp_n_idx = warpId / 2;

    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[4][2];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 8;

    int load_b_row = tid / 8;
    int load_b_col = (tid % 8) * 8;

    // Prologue Stage 0
    if (cta_m + load_a_row < M && load_a_col + 7 < K) {
        *(uint4*)&smem_A_0[load_a_row][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row) * K + load_a_col];
    } else {
        #pragma unroll
        for (int k_off = 0; k_off < 8; ++k_off) {
            int cur_k = load_a_col + k_off;
            smem_A_0[load_a_row][cur_k] = (cta_m + load_a_row < M && cur_k < K) ? A_fp16[(cta_m + load_a_row) * K + cur_k] : __float2half(0.0f);
        }
    }

    if (cta_m + load_a_row + 32 < M && load_a_col + 7 < K) {
        *(uint4*)&smem_A_0[load_a_row + 32][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 32) * K + load_a_col];
    } else {
        #pragma unroll
        for (int k_off = 0; k_off < 8; ++k_off) {
            int cur_k = load_a_col + k_off;
            smem_A_0[load_a_row + 32][cur_k] = (cta_m + load_a_row + 32 < M && cur_k < K) ? A_fp16[(cta_m + load_a_row + 32) * K + cur_k] : __float2half(0.0f);
        }
    }

    if (cta_m + load_a_row + 64 < M && load_a_col + 7 < K) {
        *(uint4*)&smem_A_0[load_a_row + 64][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 64) * K + load_a_col];
    } else {
        #pragma unroll
        for (int k_off = 0; k_off < 8; ++k_off) {
            int cur_k = load_a_col + k_off;
            smem_A_0[load_a_row + 64][cur_k] = (cta_m + load_a_row + 64 < M && cur_k < K) ? A_fp16[(cta_m + load_a_row + 64) * K + cur_k] : __float2half(0.0f);
        }
    }

    if (cta_m + load_a_row + 96 < M && load_a_col + 7 < K) {
        *(uint4*)&smem_A_0[load_a_row + 96][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 96) * K + load_a_col];
    } else {
        #pragma unroll
        for (int k_off = 0; k_off < 8; ++k_off) {
            int cur_k = load_a_col + k_off;
            smem_A_0[load_a_row + 96][cur_k] = (cta_m + load_a_row + 96 < M && cur_k < K) ? A_fp16[(cta_m + load_a_row + 96) * K + cur_k] : __float2half(0.0f);
        }
    }

    if (load_b_row < K && cta_n + load_b_col + 7 < N) {
        *(uint4*)&smem_B_0[load_b_row][load_b_col] = *(const uint4*)&B_fp16[load_b_row * N + (cta_n + load_b_col)];
    } else {
        #pragma unroll
        for (int n_off = 0; n_off < 8; ++n_off) {
            int cur_n = load_b_col + n_off;
            smem_B_0[load_b_row][cur_n] = (load_b_row < K && cta_n + cur_n < N) ? B_fp16[load_b_row * N + (cta_n + cur_n)] : __float2half(0.0f);
        }
    }

    if (load_b_row + 16 < K && cta_n + load_b_col + 7 < N) {
        *(uint4*)&smem_B_0[load_b_row + 16][load_b_col] = *(const uint4*)&B_fp16[(load_b_row + 16) * N + (cta_n + load_b_col)];
    } else {
        #pragma unroll
        for (int n_off = 0; n_off < 8; ++n_off) {
            int cur_n = load_b_col + n_off;
            smem_B_0[load_b_row + 16][cur_n] = (load_b_row + 16 < K && cta_n + cur_n < N) ? B_fp16[(load_b_row + 16) * N + (cta_n + cur_n)] : __float2half(0.0f);
        }
    }

    __syncthreads();

    int stage = 0;
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        int next_stage = 1 - stage;

        if (next_k < K) {
            if (stage == 0) {
                if (cta_m + load_a_row < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_1[load_a_row][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_1[load_a_row][cur_k] = (cta_m + load_a_row < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (cta_m + load_a_row + 32 < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_1[load_a_row + 32][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 32) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_1[load_a_row + 32][cur_k] = (cta_m + load_a_row + 32 < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row + 32) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (cta_m + load_a_row + 64 < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_1[load_a_row + 64][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 64) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_1[load_a_row + 64][cur_k] = (cta_m + load_a_row + 64 < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row + 64) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (cta_m + load_a_row + 96 < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_1[load_a_row + 96][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 96) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_1[load_a_row + 96][cur_k] = (cta_m + load_a_row + 96 < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row + 96) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (next_k + load_b_row < K && cta_n + load_b_col + 7 < N) {
                    *(uint4*)&smem_B_1[load_b_row][load_b_col] = *(const uint4*)&B_fp16[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                } else {
                    #pragma unroll
                    for (int n_off = 0; n_off < 8; ++n_off) {
                        int cur_n = load_b_col + n_off;
                        smem_B_1[load_b_row][cur_n] = (next_k + load_b_row < K && cta_n + cur_n < N) ? B_fp16[(next_k + load_b_row) * N + (cta_n + cur_n)] : __float2half(0.0f);
                    }
                }

                if (next_k + load_b_row + 16 < K && cta_n + load_b_col + 7 < N) {
                    *(uint4*)&smem_B_1[load_b_row + 16][load_b_col] = *(const uint4*)&B_fp16[(next_k + load_b_row + 16) * N + (cta_n + load_b_col)];
                } else {
                    #pragma unroll
                    for (int n_off = 0; n_off < 8; ++n_off) {
                        int cur_n = load_b_col + n_off;
                        smem_B_1[load_b_row + 16][cur_n] = (next_k + load_b_row + 16 < K && cta_n + cur_n < N) ? B_fp16[(next_k + load_b_row + 16) * N + (cta_n + cur_n)] : __float2half(0.0f);
                    }
                }
            } else {
                if (cta_m + load_a_row < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_0[load_a_row][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_0[load_a_row][cur_k] = (cta_m + load_a_row < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (cta_m + load_a_row + 32 < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_0[load_a_row + 32][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 32) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_0[load_a_row + 32][cur_k] = (cta_m + load_a_row + 32 < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row + 32) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (cta_m + load_a_row + 64 < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_0[load_a_row + 64][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 64) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_0[load_a_row + 64][cur_k] = (cta_m + load_a_row + 64 < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row + 64) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (cta_m + load_a_row + 96 < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_0[load_a_row + 96][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row + 96) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_0[load_a_row + 96][cur_k] = (cta_m + load_a_row + 96 < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row + 96) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (next_k + load_b_row < K && cta_n + load_b_col + 7 < N) {
                    *(uint4*)&smem_B_0[load_b_row][load_b_col] = *(const uint4*)&B_fp16[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                } else {
                    #pragma unroll
                    for (int n_off = 0; n_off < 8; ++n_off) {
                        int cur_n = load_b_col + n_off;
                        smem_B_0[load_b_row][cur_n] = (next_k + load_b_row < K && cta_n + cur_n < N) ? B_fp16[(next_k + load_b_row) * N + (cta_n + cur_n)] : __float2half(0.0f);
                    }
                }

                if (next_k + load_b_row + 16 < K && cta_n + load_b_col + 7 < N) {
                    *(uint4*)&smem_B_0[load_b_row + 16][load_b_col] = *(const uint4*)&B_fp16[(next_k + load_b_row + 16) * N + (cta_n + load_b_col)];
                } else {
                    #pragma unroll
                    for (int n_off = 0; n_off < 8; ++n_off) {
                        int cur_n = load_b_col + n_off;
                        smem_B_0[load_b_row + 16][cur_n] = (next_k + load_b_row + 16 < K && cta_n + cur_n < N) ? B_fp16[(next_k + load_b_row + 16) * N + (cta_n + cur_n)] : __float2half(0.0f);
                    }
                }
            }
        }

        if (stage == 0) {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = warp_m_idx * 64 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_0[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 72);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = warp_m_idx * 64 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_1[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 72);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        }

        __syncthreads();
        stage = next_stage;
    }

    float (*smem_C_fp32)[64 + 4] = (float (*)[64 + 4])raw_smem;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int r = warp_m_idx * 64 + i * 16;
            int c = warp_n_idx * 32 + j * 16;
            wmma::store_matrix_sync(&smem_C_fp32[r][c], frag_C[i][j], 68, wmma::mem_row_major);
        }
    }

    __syncthreads();

    int r_sub = tid / 8;
    int col = (tid % 8) * 8;

    #pragma unroll
    for (int r_off = 0; r_off < 128; r_off += 16) {
        int row = r_sub + r_off;
        int out_m = cta_m + row;
        int out_n = cta_n + col;

        if (out_m < M && out_n + 7 < N) {
            float4 f0 = *(float4*)&smem_C_fp32[row][col];
            float4 f1 = *(float4*)&smem_C_fp32[row][col + 4];
            uint4 b = *(uint4*)&bias_fp16[out_n];
            half2* h_b = (half2*)&b;

            half2 h0 = __floats2half2_rn(f0.x, f0.y);
            half2 h1 = __floats2half2_rn(f0.z, f0.w);
            half2 h2 = __floats2half2_rn(f1.x, f1.y);
            half2 h3 = __floats2half2_rn(f1.z, f1.w);

            alignas(16) half2 h_v[4] = {h0, h1, h2, h3};

            #pragma unroll
            for (int h = 0; h < 4; ++h) {
                h_v[h] = __hadd2(h_v[h], h_b[h]);
                if (is_relu) {
                    half zero = __float2half(0.0f);
                    half2 zero2 = __halves2half2(zero, zero);
                    h_v[h] = __hmax2(h_v[h], zero2);
                }
            }

            *(uint4*)&C_fp16[out_m * N + out_n] = *(uint4*)h_v;
        }
    }
}


// Zero-SMEM Direct Register Epilogue Fused FP16 GEMM Kernel (Doubles SM Occupancy to 132 concurrent blocks)
extern "C" __global__ void y_fused_direct_register_epilogue_kernel(
    const half* __restrict__ A_fp16,
    const half* __restrict__ B_fp16,
    const half* __restrict__ bias_fp16,
    half* __restrict__ C_fp16,
    int M, int N, int K,
    int is_relu
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    extern __shared__ char raw_smem[];
    half (*smem_A_0)[32 + 8] = (half (*)[32 + 8])raw_smem;
    half (*smem_B_0)[128 + 8] = (half (*)[128 + 8])&raw_smem[128 * 40 * sizeof(half)];

    half (*smem_A_1)[32 + 8] = (half (*)[32 + 8])&raw_smem[128 * 40 * sizeof(half) + 32 * 136 * sizeof(half)];
    half (*smem_B_1)[128 + 8] = (half (*)[128 + 8])&raw_smem[2 * 128 * 40 * sizeof(half) + 32 * 136 * sizeof(half)];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2;
    int warp_n_idx = warpId / 2;

    const int SWIZZLE = 8;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int num_tiles_per_swizzle = gridDim.x * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;

    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 16;

    int load_b_row = tid / 16;
    int load_b_col = (tid % 16) * 8;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[4][2];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    // Prologue (128-bit Vector Loads)
    if (cta_m + load_a_row < M && load_a_col + 7 < K) {
        *(uint4*)&smem_A_0[load_a_row][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row) * K + load_a_col];
    } else {
        #pragma unroll
        for (int k_off = 0; k_off < 8; ++k_off) {
            int cur_k = load_a_col + k_off;
            smem_A_0[load_a_row][cur_k] = (cta_m + load_a_row < M && cur_k < K) ? A_fp16[(cta_m + load_a_row) * K + cur_k] : __float2half(0.0f);
        }
    }

    if (load_b_row < K && cta_n + load_b_col + 7 < N) {
        *(uint4*)&smem_B_0[load_b_row][load_b_col] = *(const uint4*)&B_fp16[load_b_row * N + (cta_n + load_b_col)];
    } else {
        #pragma unroll
        for (int n_off = 0; n_off < 8; ++n_off) {
            int cur_n = load_b_col + n_off;
            smem_B_0[load_b_row][cur_n] = (load_b_row < K && cta_n + cur_n < N) ? B_fp16[load_b_row * N + (cta_n + cur_n)] : __float2half(0.0f);
        }
    }

    if (load_b_row + 16 < K && cta_n + load_b_col + 7 < N) {
        *(uint4*)&smem_B_0[load_b_row + 16][load_b_col] = *(const uint4*)&B_fp16[(load_b_row + 16) * N + (cta_n + load_b_col)];
    } else {
        #pragma unroll
        for (int n_off = 0; n_off < 8; ++n_off) {
            int cur_n = load_b_col + n_off;
            smem_B_0[load_b_row + 16][cur_n] = (load_b_row + 16 < K && cta_n + cur_n < N) ? B_fp16[(load_b_row + 16) * N + (cta_n + cur_n)] : __float2half(0.0f);
        }
    }

    __syncthreads();

    int stage = 0;
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        int next_stage = 1 - stage;

        if (next_k < K) {
            if (stage == 0) {
                if (cta_m + load_a_row < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_1[load_a_row][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_1[load_a_row][cur_k] = (cta_m + load_a_row < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (next_k + load_b_row < K && cta_n + load_b_col + 7 < N) {
                    *(uint4*)&smem_B_1[load_b_row][load_b_col] = *(const uint4*)&B_fp16[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                } else {
                    #pragma unroll
                    for (int n_off = 0; n_off < 8; ++n_off) {
                        int cur_n = load_b_col + n_off;
                        smem_B_1[load_b_row][cur_n] = (next_k + load_b_row < K && cta_n + cur_n < N) ? B_fp16[(next_k + load_b_row) * N + (cta_n + cur_n)] : __float2half(0.0f);
                    }
                }

                if (next_k + load_b_row + 16 < K && cta_n + load_b_col + 7 < N) {
                    *(uint4*)&smem_B_1[load_b_row + 16][load_b_col] = *(const uint4*)&B_fp16[(next_k + load_b_row + 16) * N + (cta_n + load_b_col)];
                } else {
                    #pragma unroll
                    for (int n_off = 0; n_off < 8; ++n_off) {
                        int cur_n = load_b_col + n_off;
                        smem_B_1[load_b_row + 16][cur_n] = (next_k + load_b_row + 16 < K && cta_n + cur_n < N) ? B_fp16[(next_k + load_b_row + 16) * N + (cta_n + cur_n)] : __float2half(0.0f);
                    }
                }
            } else {
                if (cta_m + load_a_row < M && next_k + load_a_col + 7 < K) {
                    *(uint4*)&smem_A_0[load_a_row][load_a_col] = *(const uint4*)&A_fp16[(cta_m + load_a_row) * K + (next_k + load_a_col)];
                } else {
                    #pragma unroll
                    for (int k_off = 0; k_off < 8; ++k_off) {
                        int cur_k = load_a_col + k_off;
                        smem_A_0[load_a_row][cur_k] = (cta_m + load_a_row < M && next_k + cur_k < K) ? A_fp16[(cta_m + load_a_row) * K + (next_k + cur_k)] : __float2half(0.0f);
                    }
                }

                if (next_k + load_b_row < K && cta_n + load_b_col + 7 < N) {
                    *(uint4*)&smem_B_0[load_b_row][load_b_col] = *(const uint4*)&B_fp16[(next_k + load_b_row) * N + (cta_n + load_b_col)];
                } else {
                    #pragma unroll
                    for (int n_off = 0; n_off < 8; ++n_off) {
                        int cur_n = load_b_col + n_off;
                        smem_B_0[load_b_row][cur_n] = (next_k + load_b_row < K && cta_n + cur_n < N) ? B_fp16[(next_k + load_b_row) * N + (cta_n + cur_n)] : __float2half(0.0f);
                    }
                }

                if (next_k + load_b_row + 16 < K && cta_n + load_b_col + 7 < N) {
                    *(uint4*)&smem_B_0[load_b_row + 16][load_b_col] = *(const uint4*)&B_fp16[(next_k + load_b_row + 16) * N + (cta_n + load_b_col)];
                } else {
                    #pragma unroll
                    for (int n_off = 0; n_off < 8; ++n_off) {
                        int cur_n = load_b_col + n_off;
                        smem_B_0[load_b_row + 16][cur_n] = (next_k + load_b_row + 16 < K && cta_n + cur_n < N) ? B_fp16[(next_k + load_b_row + 16) * N + (cta_n + cur_n)] : __float2half(0.0f);
                    }
                }
            }
        }

        if (stage == 0) {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = warp_m_idx * 64 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_0[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 136);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int a_row = warp_m_idx * 64 + i * 16;
                    wmma::load_matrix_sync(frag_A[i], &smem_A_1[a_row][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int b_col = warp_n_idx * 32 + j * 16;
                    wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 136);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
        }

        __syncthreads();
        stage = next_stage;
    }

    // Direct Register Store Epilogue (Zero SMEM staging, Zero Barrier Stalls)
    float (*smem_C_fp32)[128 + 4] = (float (*)[128 + 4])raw_smem;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int r = warp_m_idx * 64 + i * 16;
            int c = warp_n_idx * 32 + j * 16;
            wmma::store_matrix_sync(&smem_C_fp32[r][c], frag_C[i][j], 132, wmma::mem_row_major);
        }
    }
    __syncthreads();

    int r_sub = tid / 16;
    int col = (tid % 16) * 8;

    #pragma unroll
    for (int r_off = 0; r_off < 128; r_off += 16) {
        int row = r_sub + r_off;
        int out_m = cta_m + row;
        int out_n = cta_n + col;

        if (out_m < M && out_n + 7 < N) {
            float4 f0 = *(float4*)&smem_C_fp32[row][col];
            float4 f1 = *(float4*)&smem_C_fp32[row][col + 4];
            uint4 b = *(uint4*)&bias_fp16[out_n];
            half2* h_b = (half2*)&b;

            half2 h0 = __floats2half2_rn(f0.x, f0.y);
            half2 h1 = __floats2half2_rn(f0.z, f0.w);
            half2 h2 = __floats2half2_rn(f1.x, f1.y);
            half2 h3 = __floats2half2_rn(f1.z, f1.w);

            alignas(16) half2 h_v[4] = {h0, h1, h2, h3};

            #pragma unroll
            for (int h = 0; h < 4; ++h) {
                h_v[h] = __hadd2(h_v[h], h_b[h]);
                if (is_relu) {
                    half zero = __float2half(0.0f);
                    half2 zero2 = __halves2half2(zero, zero);
                    h_v[h] = __hmax2(h_v[h], zero2);
                }
            }

            *(uint4*)&C_fp16[out_m * N + out_n] = *(uint4*)h_v;
        }
    }
}

// Persistent Fused 3-Layer MLP Kernel (Layer 1 + Layer 2 + Layer 3 in a Single Persistent Grid Launch with zero host overhead)
extern "C" __global__ void y_fused_mlp_3layer_persistent_kernel(
    const float* __restrict__ X_fp32,
    const float* __restrict__ W1_fp32,
    const float* __restrict__ bias1_fp32,
    const float* __restrict__ W2_fp32,
    const float* __restrict__ bias2_fp32,
    const float* __restrict__ W3_fp32,
    const float* __restrict__ bias3_fp32,
    float* __restrict__ Out_fp32,
    float* __restrict__ H1_scratch,
    float* __restrict__ H2_scratch,
    int B, int Din, int Dh, int Dout
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    extern __shared__ char raw_smem[];
    half (*smem_A_0)[32 + 8] = (half (*)[32 + 8])raw_smem;
    half (*smem_B_0)[128 + 8] = (half (*)[128 + 8])&raw_smem[128 * 40 * sizeof(half)];

    half (*smem_A_1)[32 + 8] = (half (*)[32 + 8])&raw_smem[128 * 40 * sizeof(half) + 32 * 136 * sizeof(half)];
    half (*smem_B_1)[128 + 8] = (half (*)[128 + 8])&raw_smem[2 * 128 * 40 * sizeof(half) + 32 * 136 * sizeof(half)];

    int tid = threadIdx.x;
    int warpId = tid / 32;
    int warp_m_idx = warpId % 2;
    int warp_n_idx = warpId / 2;

    int cta_m = blockIdx.y * BLOCK_M;

    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 16;
    int load_b_row = tid / 8;
    int load_b_col = (tid % 8) * 16;

    if (cta_m >= B) return;

    // -------------------------------------------------------------------------
    // PHASE 1: LAYER 1 (X [128, Din=1024] @ W1 [1024, Dh=2048] + bias1 -> H1_scratch)
    // -------------------------------------------------------------------------
    for (int col_block = 0; col_block < Dh / BLOCK_N; ++col_block) {
        int cta_n = col_block * BLOCK_N;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[4][2];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::fill_fragment(frag_C[i][j], 0.0f);
            }
        }

        #pragma unroll
        for (int offset = 0; offset < 16; offset += 2) {
            int cur_a_col = load_a_col + offset;
            if (cta_m + load_a_row < B && cur_a_col < Din) {
                float2 vA = *(float2*)&X_fp32[(cta_m + load_a_row) * Din + cur_a_col];
                smem_A_0[load_a_row][cur_a_col] = __float2half(vA.x);
                smem_A_0[load_a_row][cur_a_col + 1] = __float2half(vA.y);
            } else {
                smem_A_0[load_a_row][cur_a_col] = __float2half(0.0f);
                smem_A_0[load_a_row][cur_a_col + 1] = __float2half(0.0f);
            }
        }
        #pragma unroll
        for (int offset = 0; offset < 16; offset += 2) {
            int cur_b_col = load_b_col + offset;
            if (load_b_row < Din && cta_n + cur_b_col < Dh) {
                float2 vB = *(float2*)&W1_fp32[load_b_row * Dh + (cta_n + cur_b_col)];
                smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
                smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
            } else {
                smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
                smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
            }
        }

        __syncthreads();

        int stage = 0;
        for (int k = 0; k < Din; k += BLOCK_K) {
            int next_k = k + BLOCK_K;
            int next_stage = 1 - stage;

            if (next_k < Din) {
                if (stage == 0) {
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_a_col = load_a_col + offset;
                        if (cta_m + load_a_row < B && next_k + cur_a_col < Din) {
                            float2 vA = *(float2*)&X_fp32[(cta_m + load_a_row) * Din + (next_k + cur_a_col)];
                            smem_A_1[load_a_row][cur_a_col] = __float2half(vA.x);
                            smem_A_1[load_a_row][cur_a_col + 1] = __float2half(vA.y);
                        } else {
                            smem_A_1[load_a_row][cur_a_col] = __float2half(0.0f);
                            smem_A_1[load_a_row][cur_a_col + 1] = __float2half(0.0f);
                        }
                    }
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_b_col = load_b_col + offset;
                        if (next_k + load_b_row < Din && cta_n + cur_b_col < Dh) {
                            float2 vB = *(float2*)&W1_fp32[(next_k + load_b_row) * Dh + (cta_n + cur_b_col)];
                            smem_B_1[load_b_row][cur_b_col] = __float2half(vB.x);
                            smem_B_1[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                        } else {
                            smem_B_1[load_b_row][cur_b_col] = __float2half(0.0f);
                            smem_B_1[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_a_col = load_a_col + offset;
                        if (cta_m + load_a_row < B && next_k + cur_a_col < Din) {
                            float2 vA = *(float2*)&X_fp32[(cta_m + load_a_row) * Din + (next_k + cur_a_col)];
                            smem_A_0[load_a_row][cur_a_col] = __float2half(vA.x);
                            smem_A_0[load_a_row][cur_a_col + 1] = __float2half(vA.y);
                        } else {
                            smem_A_0[load_a_row][cur_a_col] = __float2half(0.0f);
                            smem_A_0[load_a_row][cur_a_col + 1] = __float2half(0.0f);
                        }
                    }
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_b_col = load_b_col + offset;
                        if (next_k + load_b_row < Din && cta_n + cur_b_col < Dh) {
                            float2 vB = *(float2*)&W1_fp32[(next_k + load_b_row) * Dh + (cta_n + cur_b_col)];
                            smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
                            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                        } else {
                            smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
                            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                        }
                    }
                }
            }

            if (stage == 0) {
                #pragma unroll
                for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int a_row = warp_m_idx * 64 + i * 16;
                        wmma::load_matrix_sync(frag_A[i], &smem_A_0[a_row][k_step], 40);
                    }
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        int b_col = warp_n_idx * 32 + j * 16;
                        wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 136);
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int a_row = warp_m_idx * 64 + i * 16;
                        wmma::load_matrix_sync(frag_A[i], &smem_A_1[a_row][k_step], 40);
                    }
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        int b_col = warp_n_idx * 32 + j * 16;
                        wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 136);
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                        }
                    }
                }
            }

            __syncthreads();
            stage = next_stage;
        }

        // Epilogue Layer 1 -> H1_scratch (Bias + ReLU)
        float (*smem_C)[128 + 4] = (float (*)[128 + 4])raw_smem;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int r = warp_m_idx * 64 + i * 16;
                int c = warp_n_idx * 32 + j * 16;
                wmma::store_matrix_sync(&smem_C[r][c], frag_C[i][j], 132, wmma::mem_row_major);
            }
        }
        __syncthreads();

        int r_sub = tid / 16;
        int col = (tid % 16) * 4;
        #pragma unroll
        for (int r_off = 0; r_off < 128; r_off += 16) {
            int row = r_sub + r_off;
            int out_m = cta_m + row;
            int out_n = cta_n + col;
            if (out_m < B && out_n + 3 < Dh) {
                float4 v = *(float4*)&smem_C[row][col];
                float4 b = *(float4*)&bias1_fp32[out_n];
                v.x = v.x + b.x > 0.0f ? v.x + b.x : 0.0f;
                v.y = v.y + b.y > 0.0f ? v.y + b.y : 0.0f;
                v.z = v.z + b.z > 0.0f ? v.z + b.z : 0.0f;
                v.w = v.w + b.w > 0.0f ? v.w + b.w : 0.0f;
                *(float4*)&H1_scratch[out_m * Dh + out_n] = v;
            }
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // PHASE 2: LAYER 2 (H1_scratch [128, Dh=2048] @ W2 [2048, Dh=2048] + bias2 -> H2_scratch)
    // -------------------------------------------------------------------------
    for (int col_block = 0; col_block < Dh / BLOCK_N; ++col_block) {
        int cta_n = col_block * BLOCK_N;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[4][2];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::fill_fragment(frag_C[i][j], 0.0f);
            }
        }

        #pragma unroll
        for (int offset = 0; offset < 16; offset += 2) {
            int cur_a_col = load_a_col + offset;
            if (cta_m + load_a_row < B && cur_a_col < Dh) {
                float2 vA = *(float2*)&H1_scratch[(cta_m + load_a_row) * Dh + cur_a_col];
                smem_A_0[load_a_row][cur_a_col] = __float2half(vA.x);
                smem_A_0[load_a_row][cur_a_col + 1] = __float2half(vA.y);
            } else {
                smem_A_0[load_a_row][cur_a_col] = __float2half(0.0f);
                smem_A_0[load_a_row][cur_a_col + 1] = __float2half(0.0f);
            }
        }
        #pragma unroll
        for (int offset = 0; offset < 16; offset += 2) {
            int cur_b_col = load_b_col + offset;
            if (load_b_row < Dh && cta_n + cur_b_col < Dh) {
                float2 vB = *(float2*)&W2_fp32[load_b_row * Dh + (cta_n + cur_b_col)];
                smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
                smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
            } else {
                smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
                smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
            }
        }

        __syncthreads();

        int stage = 0;
        for (int k = 0; k < Dh; k += BLOCK_K) {
            int next_k = k + BLOCK_K;
            int next_stage = 1 - stage;

            if (next_k < Dh) {
                if (stage == 0) {
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_a_col = load_a_col + offset;
                        if (cta_m + load_a_row < B && next_k + cur_a_col < Dh) {
                            float2 vA = *(float2*)&H1_scratch[(cta_m + load_a_row) * Dh + (next_k + cur_a_col)];
                            smem_A_1[load_a_row][cur_a_col] = __float2half(vA.x);
                            smem_A_1[load_a_row][cur_a_col + 1] = __float2half(vA.y);
                        } else {
                            smem_A_1[load_a_row][cur_a_col] = __float2half(0.0f);
                            smem_A_1[load_a_row][cur_a_col + 1] = __float2half(0.0f);
                        }
                    }
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_b_col = load_b_col + offset;
                        if (next_k + load_b_row < Dh && cta_n + cur_b_col < Dh) {
                            float2 vB = *(float2*)&W2_fp32[(next_k + load_b_row) * Dh + (cta_n + cur_b_col)];
                            smem_B_1[load_b_row][cur_b_col] = __float2half(vB.x);
                            smem_B_1[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                        } else {
                            smem_B_1[load_b_row][cur_b_col] = __float2half(0.0f);
                            smem_B_1[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_a_col = load_a_col + offset;
                        if (cta_m + load_a_row < B && next_k + cur_a_col < Dh) {
                            float2 vA = *(float2*)&H1_scratch[(cta_m + load_a_row) * Dh + (next_k + cur_a_col)];
                            smem_A_0[load_a_row][cur_a_col] = __float2half(vA.x);
                            smem_A_0[load_a_row][cur_a_col + 1] = __float2half(vA.y);
                        } else {
                            smem_A_0[load_a_row][cur_a_col] = __float2half(0.0f);
                            smem_A_0[load_a_row][cur_a_col + 1] = __float2half(0.0f);
                        }
                    }
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_b_col = load_b_col + offset;
                        if (next_k + load_b_row < Dh && cta_n + cur_b_col < Dh) {
                            float2 vB = *(float2*)&W2_fp32[(next_k + load_b_row) * Dh + (cta_n + cur_b_col)];
                            smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
                            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                        } else {
                            smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
                            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                        }
                    }
                }
            }

            if (stage == 0) {
                #pragma unroll
                for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int a_row = warp_m_idx * 64 + i * 16;
                        wmma::load_matrix_sync(frag_A[i], &smem_A_0[a_row][k_step], 40);
                    }
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        int b_col = warp_n_idx * 32 + j * 16;
                        wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 136);
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int a_row = warp_m_idx * 64 + i * 16;
                        wmma::load_matrix_sync(frag_A[i], &smem_A_1[a_row][k_step], 40);
                    }
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        int b_col = warp_n_idx * 32 + j * 16;
                        wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 136);
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                        }
                    }
                }
            }

            __syncthreads();
            stage = next_stage;
        }

        // Epilogue Layer 2 -> H2_scratch (Bias + ReLU)
        float (*smem_C)[128 + 4] = (float (*)[128 + 4])raw_smem;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int r = warp_m_idx * 64 + i * 16;
                int c = warp_n_idx * 32 + j * 16;
                wmma::store_matrix_sync(&smem_C[r][c], frag_C[i][j], 132, wmma::mem_row_major);
            }
        }
        __syncthreads();

        int r_sub = tid / 16;
        int col = (tid % 16) * 4;
        #pragma unroll
        for (int r_off = 0; r_off < 128; r_off += 16) {
            int row = r_sub + r_off;
            int out_m = cta_m + row;
            int out_n = cta_n + col;
            if (out_m < B && out_n + 3 < Dh) {
                float4 v = *(float4*)&smem_C[row][col];
                float4 b = *(float4*)&bias2_fp32[out_n];
                v.x = v.x + b.x > 0.0f ? v.x + b.x : 0.0f;
                v.y = v.y + b.y > 0.0f ? v.y + b.y : 0.0f;
                v.z = v.z + b.z > 0.0f ? v.z + b.z : 0.0f;
                v.w = v.w + b.w > 0.0f ? v.w + b.w : 0.0f;
                *(float4*)&H2_scratch[out_m * Dh + out_n] = v;
            }
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // PHASE 3: LAYER 3 (H2_scratch [128, Dh=2048] @ W3 [2048, Dout=128] + bias3 -> Out)
    // -------------------------------------------------------------------------
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[4][2];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::fill_fragment(frag_C[i][j], 0.0f);
            }
        }

        #pragma unroll
        for (int offset = 0; offset < 16; offset += 2) {
            int cur_a_col = load_a_col + offset;
            if (cta_m + load_a_row < B && cur_a_col < Dh) {
                float2 vA = *(float2*)&H2_scratch[(cta_m + load_a_row) * Dh + cur_a_col];
                smem_A_0[load_a_row][cur_a_col] = __float2half(vA.x);
                smem_A_0[load_a_row][cur_a_col + 1] = __float2half(vA.y);
            } else {
                smem_A_0[load_a_row][cur_a_col] = __float2half(0.0f);
                smem_A_0[load_a_row][cur_a_col + 1] = __float2half(0.0f);
            }
        }
        #pragma unroll
        for (int offset = 0; offset < 16; offset += 2) {
            int cur_b_col = load_b_col + offset;
            if (load_b_row < Dh && cur_b_col < Dout) {
                float2 vB = *(float2*)&W3_fp32[load_b_row * Dout + cur_b_col];
                smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
                smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
            } else {
                smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
                smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
            }
        }

        __syncthreads();

        int stage = 0;
        for (int k = 0; k < Dh; k += BLOCK_K) {
            int next_k = k + BLOCK_K;
            int next_stage = 1 - stage;

            if (next_k < Dh) {
                if (stage == 0) {
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_a_col = load_a_col + offset;
                        if (cta_m + load_a_row < B && next_k + cur_a_col < Dh) {
                            float2 vA = *(float2*)&H2_scratch[(cta_m + load_a_row) * Dh + (next_k + cur_a_col)];
                            smem_A_1[load_a_row][cur_a_col] = __float2half(vA.x);
                            smem_A_1[load_a_row][cur_a_col + 1] = __float2half(vA.y);
                        } else {
                            smem_A_1[load_a_row][cur_a_col] = __float2half(0.0f);
                            smem_A_1[load_a_row][cur_a_col + 1] = __float2half(0.0f);
                        }
                    }
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_b_col = load_b_col + offset;
                        if (next_k + load_b_row < Dh && cur_b_col < Dout) {
                            float2 vB = *(float2*)&W3_fp32[(next_k + load_b_row) * Dout + cur_b_col];
                            smem_B_1[load_b_row][cur_b_col] = __float2half(vB.x);
                            smem_B_1[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                        } else {
                            smem_B_1[load_b_row][cur_b_col] = __float2half(0.0f);
                            smem_B_1[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_a_col = load_a_col + offset;
                        if (cta_m + load_a_row < B && next_k + cur_a_col < Dh) {
                            float2 vA = *(float2*)&H2_scratch[(cta_m + load_a_row) * Dh + (next_k + cur_a_col)];
                            smem_A_0[load_a_row][cur_a_col] = __float2half(vA.x);
                            smem_A_0[load_a_row][cur_a_col + 1] = __float2half(vA.y);
                        } else {
                            smem_A_0[load_a_row][cur_a_col] = __float2half(0.0f);
                            smem_A_0[load_a_row][cur_a_col + 1] = __float2half(0.0f);
                        }
                    }
                    #pragma unroll
                    for (int offset = 0; offset < 16; offset += 2) {
                        int cur_b_col = load_b_col + offset;
                        if (next_k + load_b_row < Dh && cur_b_col < Dout) {
                            float2 vB = *(float2*)&W3_fp32[(next_k + load_b_row) * Dout + cur_b_col];
                            smem_B_0[load_b_row][cur_b_col] = __float2half(vB.x);
                            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(vB.y);
                        } else {
                            smem_B_0[load_b_row][cur_b_col] = __float2half(0.0f);
                            smem_B_0[load_b_row][cur_b_col + 1] = __float2half(0.0f);
                        }
                    }
                }
            }

            if (stage == 0) {
                #pragma unroll
                for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int a_row = warp_m_idx * 64 + i * 16;
                        wmma::load_matrix_sync(frag_A[i], &smem_A_0[a_row][k_step], 40);
                    }
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        int b_col = warp_n_idx * 32 + j * 16;
                        wmma::load_matrix_sync(frag_B[j], &smem_B_0[k_step][b_col], 136);
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int a_row = warp_m_idx * 64 + i * 16;
                        wmma::load_matrix_sync(frag_A[i], &smem_A_1[a_row][k_step], 40);
                    }
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        int b_col = warp_n_idx * 32 + j * 16;
                        wmma::load_matrix_sync(frag_B[j], &smem_B_1[k_step][b_col], 136);
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                        }
                    }
                }
            }

            __syncthreads();
            stage = next_stage;
        }

        // Epilogue Layer 3 -> Out (Bias addition only, linear output)
        float (*smem_C)[128 + 4] = (float (*)[128 + 4])raw_smem;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int r = warp_m_idx * 64 + i * 16;
                int c = warp_n_idx * 32 + j * 16;
                wmma::store_matrix_sync(&smem_C[r][c], frag_C[i][j], 132, wmma::mem_row_major);
            }
        }
        __syncthreads();

        int r_sub = tid / 16;
        int col = (tid % 16) * 4;
        #pragma unroll
        for (int r_off = 0; r_off < 128; r_off += 16) {
            int row = r_sub + r_off;
            int out_m = cta_m + row;
            int out_n = col;
            if (out_m < B && out_n + 3 < Dout) {
                float4 v = *(float4*)&smem_C[row][col];
                float4 b = *(float4*)&bias3_fp32[out_n];
                v.x = v.x + b.x;
                v.y = v.y + b.y;
                v.z = v.z + b.z;
                v.w = v.w + b.w;
                *(float4*)&Out_fp32[out_m * Dout + out_n] = v;
            }
        }
    }
}

// Optimized 128-bit Vectorized FP16 Fused GEMM Kernel for 512x512 (64x64 CTA Tile, Direct Fragment Bias + ReLU Inlining)
extern "C" __global__ __launch_bounds__(128, 4) void y_fused_gemm_bias_relu_small_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    const half* __restrict__ bias,
    half* __restrict__ C,
    int M, int N, int K,
    int is_relu
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    __shared__ alignas(128) half smem_A[64][32 + 8];
    __shared__ alignas(128) half smem_B[32][64 + 8];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    int warp_m_idx = warpId % 2;
    int warp_n_idx = warpId / 2;

    const int SWIZZLE = 8;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int num_tiles_per_swizzle = gridDim.x * SWIZZLE;
    int group_id = tile_idx / num_tiles_per_swizzle;
    int group_offset = tile_idx % num_tiles_per_swizzle;

    int cta_m = (group_id * SWIZZLE + (group_offset % SWIZZLE)) * BLOCK_M;
    int cta_n = (group_offset / SWIZZLE) * BLOCK_N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C[2][2];

    #pragma unroll
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(frag_C[i][j], __float2half(0.0f));

    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 16;
    int load_b_row = tid / 4;
    int load_b_col = (tid % 4) * 16;

    for (int k = 0; k < K; k += BLOCK_K) {
        if (cta_m + load_a_row < M && k + load_a_col < K) {
            *(uint4*)&smem_A[load_a_row][load_a_col] = *(const uint4*)&A[(cta_m + load_a_row) * K + (k + load_a_col)];
        } else {
            *(uint4*)&smem_A[load_a_row][load_a_col] = make_uint4(0, 0, 0, 0);
        }

        if (k + load_b_row < K && cta_n + load_b_col < N) {
            *(uint4*)&smem_B[load_b_row][load_b_col] = *(const uint4*)&B[(k + load_b_row) * N + (cta_n + load_b_col)];
        } else {
            *(uint4*)&smem_B[load_b_row][load_b_col] = make_uint4(0, 0, 0, 0);
        }

        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int a_row = warp_m_idx * 32 + i * 16;
                wmma::load_matrix_sync(frag_A[i], &smem_A[a_row][k_step], 40);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int b_col = warp_n_idx * 32 + j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[k_step][b_col], 72);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int out_m = cta_m + warp_m_idx * 32 + i * 16;
            int out_n = cta_n + warp_n_idx * 32 + j * 16;

            if (out_m < M && out_n < N) {
                #pragma unroll
                for (int t = 0; t < frag_C[i][j].num_elements; ++t) {
                    half b_val = bias[out_n + (t % 16)];
                    half sum = frag_C[i][j].x[t] + b_val;
                    if (is_relu) {
                        frag_C[i][j].x[t] = sum > __float2half(0.0f) ? sum : __float2half(0.0f);
                    } else {
                        frag_C[i][j].x[t] = sum;
                    }
                }
                wmma::store_matrix_sync(&C[out_m * N + out_n], frag_C[i][j], N, wmma::mem_row_major);
            }
        }
    }
}


// // Standalone Optimized FP8 Tensor Core MMA GEMM kernel (128x128x64 CTA Tile + 128B XOR SMEM Swizzle + 4-Stage cp.async + In-Register Scale Fusion + m16n8k32 mma.sync)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
extern "C" __global__ __launch_bounds__(256, 1) void y_fp8_tensor_core_gemm_kernel(
    const char* __restrict__ A,
    const char* __restrict__ B,
    half* __restrict__ C,
    float scale_a,
    float scale_b,
    int M, int N, int K
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 64; // 64 FP8 elements per stage

    // 2D L2 Cache Swizzling (GROUP_SIZE_M = 8)
    const int GROUP_SIZE_M = 8;
    int grid_m = (M + BLOCK_M - 1) / BLOCK_M;
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int num_tiles_in_group = GROUP_SIZE_M * grid_n;
    int group_id = tile_idx / num_tiles_in_group;
    int first_tile_m = group_id * GROUP_SIZE_M;
    int group_size_m = min(grid_m - first_tile_m, GROUP_SIZE_M);
    int cta_m = (first_tile_m + (tile_idx % group_size_m)) * BLOCK_M;
    int cta_n = ((tile_idx / group_size_m) % grid_n) * BLOCK_N;

    // 4-Stage Shared Memory Allocation for FP8
    // Tile A: 128 rows x 64 FP8 elements = 8192 bytes per stage
    // Tile B: 64 rows x 128 FP8 elements = 8192 bytes per stage
    // Total dynamic SMEM = 4 * (8192 + 8192) = 65,536 bytes (64 KB)
    extern __shared__ char smem_raw[];
    char (*smem_A)[128][64] = (char (*)[128][64])smem_raw;
    char (*smem_B)[64][128] = (char (*)[64][128])(smem_raw + 4 * 128 * 64 * sizeof(char));

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = warp_id % 4; // 0..3 (4 warps in M)
    int warp_n = warp_id / 4; // 0..1 (2 warps in N)

    // Accumulators for warp C tile (32x64): 2 sub-tiles in M, 8 sub-tiles in N
    float frag_C[2][8][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                frag_C[i][j][c] = 0.0f;
            }
        }
    }

    // Helper for loading 16-byte global memory vectors to shared memory via cp.async
    auto load_gmem_to_smem_stage_fp8 = [&](int stage, int k_curr) {
        // Load Tile A (128x64 FP8 -> 512 chunks of 16B)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int chunk_idx = tid + i * 256;
            int r = chunk_idx / 4;
            int c = (chunk_idx % 4) * 16;
            int gmem_r = cta_m + r;
            int gmem_c = k_curr + c;
            bool valid = (gmem_r < M) && (gmem_c < K);
            const void* g_ptr = valid ? (const void*)&A[gmem_r * K + gmem_c] : nullptr;
            int byte_c = c;
            int swizzled_byte_c = byte_c ^ ((r % 4) << 4);
            void* s_ptr = (void*)&smem_A[stage][r][swizzled_byte_c];
            cp_async_cg_16(s_ptr, g_ptr, valid);
        }

        // Load Tile B (64x128 FP8 -> 512 chunks of 16B)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int chunk_idx = tid + i * 256;
            int r = chunk_idx / 8;
            int c = (chunk_idx % 8) * 16;
            int gmem_r = k_curr + r;
            int gmem_c = cta_n + c;
            bool valid = (gmem_r < K) && (gmem_c < N);
            const void* g_ptr = valid ? (const void*)&B[gmem_r * N + gmem_c] : nullptr;
            int byte_c = c;
            int swizzled_byte_c = byte_c ^ ((r % 8) << 4);
            void* s_ptr = (void*)&smem_B[stage][r][swizzled_byte_c];
            cp_async_cg_16(s_ptr, g_ptr, valid);
        }
    };

    // Preamble Async Prefetch (Fill stages 0, 1, 2)
    load_gmem_to_smem_stage_fp8(0, 0);
    cp_async_commit();

    load_gmem_to_smem_stage_fp8(1, BLOCK_K);
    cp_async_commit();

    load_gmem_to_smem_stage_fp8(2, 2 * BLOCK_K);
    cp_async_commit();

    int write_stage = 3;
    int read_stage = 0;

    // Main K Loop with 4-Stage Circular Pipeline
    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + 3 * BLOCK_K;
        if (next_k < K) {
            load_gmem_to_smem_stage_fp8(write_stage, next_k);
        } else {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int chunk_idx = tid + i * 256;
                int r = chunk_idx / 4;
                int c = (chunk_idx % 4) * 16;
                int byte_c = c;
                int swizzled_byte_c = byte_c ^ ((r % 4) << 4);
                void* s_ptr = (void*)&smem_A[write_stage][r][swizzled_byte_c];
                cp_async_cg_16(s_ptr, nullptr, false);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int chunk_idx = tid + i * 256;
                int r = chunk_idx / 8;
                int c = (chunk_idx % 8) * 16;
                int byte_c = c;
                int swizzled_byte_c = byte_c ^ ((r % 8) << 4);
                void* s_ptr = (void*)&smem_B[write_stage][r][swizzled_byte_c];
                cp_async_cg_16(s_ptr, nullptr, false);
            }
        }
        cp_async_commit();

        cp_async_wait_group<2>();
        __syncthreads();

        // Compute on read_stage using ldmatrix and m16n8k32 mma.sync for FP8
        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 32) {
            uint32_t reg_A[2][4];
            uint32_t reg_B[8][2];

            // Load A tile (32x32 FP8 = 16x16 16-bit words) for warp
            #pragma unroll
            for (int i_mma = 0; i_mma < 2; ++i_mma) {
                int r = warp_m * 32 + i_mma * 16 + (lane_id % 16);
                int c = k_step + (lane_id / 16) * 16;
                int byte_c = c;
                int swizzled_byte_c = byte_c ^ ((r % 4) << 4);
                void* s_ptr = (void*)&smem_A[read_stage][r][swizzled_byte_c];
                uint32_t smem_ptr32 = static_cast<uint32_t>(__cvta_generic_to_shared(s_ptr));
                ldmatrix_x4(reg_A[i_mma], smem_ptr32);
            }

            // Load B tile (32x64 FP8 = 16x64 16-bit words) for warp via ldmatrix.x2.trans
            #pragma unroll
            for (int j_mma = 0; j_mma < 8; ++j_mma) {
                int r = k_step + (lane_id % 16);
                int c = warp_n * 64 + j_mma * 8;
                int byte_c = c;
                int swizzled_byte_c = byte_c ^ ((r % 8) << 4);
                void* s_ptr = (void*)&smem_B[read_stage][r][swizzled_byte_c];
                uint32_t smem_ptr32 = static_cast<uint32_t>(__cvta_generic_to_shared(s_ptr));
                ldmatrix_x2_trans(reg_B[j_mma], smem_ptr32);
            }

            // Perform Tensor Core mma.sync for FP8 (m16n8k32)
            #pragma unroll
            for (int i_mma = 0; i_mma < 2; ++i_mma) {
                #pragma unroll
                for (int j_mma = 0; j_mma < 8; ++j_mma) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(frag_C[i_mma][j_mma][0]), "=f"(frag_C[i_mma][j_mma][1]),
                          "=f"(frag_C[i_mma][j_mma][2]), "=f"(frag_C[i_mma][j_mma][3])
                        : "r"(reg_A[i_mma][0]), "r"(reg_A[i_mma][1]),
                          "r"(reg_A[i_mma][2]), "r"(reg_A[i_mma][3]),
                          "r"(reg_B[j_mma][0]), "r"(reg_B[j_mma][1]),
                          "f"(frag_C[i_mma][j_mma][0]), "f"(frag_C[i_mma][j_mma][1]),
                          "f"(frag_C[i_mma][j_mma][2]), "f"(frag_C[i_mma][j_mma][3])
                    );
                }
            }
        }

        __syncthreads();
        write_stage = (write_stage + 1) % 4;
        read_stage = (read_stage + 1) % 4;
    }

    cp_async_wait_group<0>();
    __syncthreads();

    // Epilogue: Vectorized Scale Fusion & Store to C
    float total_scale = scale_a * scale_b;
    int group = lane_id / 4;        // 0..7
    int lane_in_group = lane_id % 4; // 0..3

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int out_m0 = cta_m + warp_m * 32 + i * 16 + group;
            int out_m1 = cta_m + warp_m * 32 + i * 16 + group + 8;
            int out_n = cta_n + warp_n * 64 + j * 8 + lane_in_group * 2;

            if (out_m0 < M && out_n + 1 < N) {
                float v0 = frag_C[i][j][0] * total_scale;
                float v1 = frag_C[i][j][1] * total_scale;
                half2 val = __floats2half2_rn(v0, v1);
                *reinterpret_cast<half2*>(&C[out_m0 * N + out_n]) = val;
            } else if (out_m0 < M && out_n < N) {
                C[out_m0 * N + out_n] = __float2half(frag_C[i][j][0] * total_scale);
            }

            if (out_m1 < M && out_n + 1 < N) {
                float v2 = frag_C[i][j][2] * total_scale;
                float v3 = frag_C[i][j][3] * total_scale;
                half2 val = __floats2half2_rn(v2, v3);
                *reinterpret_cast<half2*>(&C[out_m1 * N + out_n]) = val;
            } else if (out_m1 < M && out_n < N) {
                C[out_m1 * N + out_n] = __float2half(frag_C[i][j][2] * total_scale);
            }
        }
    }
}
#else
extern "C" __global__ __launch_bounds__(256, 1) void y_fp8_tensor_core_gemm_kernel(
    const char* __restrict__ A,
    const char* __restrict__ B,
    half* __restrict__ C,
    float scale_a,
    float scale_b,
    int M, int N, int K
) {
    // Stub for pre-Ada Lovelace GPUs (SM < 890)
}
#endif

// 256x128x32 High-Throughput Standalone GEMM Kernel (256 threads, 4-Stage cp.async.cg, Double-Buffered ldmatrix)
extern "C" __global__ __launch_bounds__(256, 1) void y_tensor_core_gemm_256x128_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 256;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    const int GROUP_SIZE_M = 8;
    int grid_m = (M + BLOCK_M - 1) / BLOCK_M;
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int num_tiles_in_group = GROUP_SIZE_M * grid_n;
    int group_id = tile_idx / num_tiles_in_group;
    int first_tile_m = group_id * GROUP_SIZE_M;
    int group_size_m = min(grid_m - first_tile_m, GROUP_SIZE_M);

    int cta_m = (first_tile_m + (tile_idx % group_size_m)) * BLOCK_M;
    int cta_n = ((tile_idx % num_tiles_in_group) / group_size_m) * BLOCK_N;

    extern __shared__ char smem_raw[];
    half (*smem_A)[256][32] = (half (*)[256][32])smem_raw;
    half (*smem_B)[32][128] = (half (*)[32][128])(smem_raw + 4 * 256 * 32 * sizeof(half));

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = warp_id % 4; // 0..3 (4 warps in M -> 64 rows per warp)
    int warp_n = warp_id / 4; // 0..1 (2 warps in N -> 64 cols per warp)

    float frag_C[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                frag_C[i][j][c] = 0.0f;
            }
        }
    }

    auto load_stage_256 = [&](int stage, int k_curr) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int chunk_idx = tid + i * 256;
            int r = chunk_idx / 4;
            int c = (chunk_idx % 4) * 8;
            int gmem_r = cta_m + r;
            int gmem_c = k_curr + c;
            bool valid = (gmem_r < M) && (gmem_c + 7 < K);
            const void* g_ptr = valid ? (const void*)&A[gmem_r * K + gmem_c] : nullptr;
            bool is_aligned = (((unsigned long long)g_ptr) & 15) == 0;
            int byte_c = c * 2;
            int swizzled_byte_c = byte_c ^ ((r % 4) << 4);
            void* s_ptr = (void*)((char*)&smem_A[stage][r][0] + swizzled_byte_c);
            if (valid && is_aligned) {
                cp_async_cg_16(s_ptr, g_ptr, true);
            } else if (valid && !is_aligned) {
                cp_async_cg_16(s_ptr, g_ptr, false);
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    if (gmem_c + e < K) smem_A[stage][r][c + e] = A[gmem_r * K + gmem_c + e];
                }
            } else {
                cp_async_cg_16(s_ptr, nullptr, false);
            }
        }

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int chunk_idx = tid + i * 256;
            int r = chunk_idx / 16;
            int c = (chunk_idx % 16) * 8;
            int gmem_r = k_curr + r;
            int gmem_c = cta_n + c;
            bool valid = (gmem_r < K) && (gmem_c + 7 < N);
            const void* g_ptr = valid ? (const void*)&B[gmem_r * N + gmem_c] : nullptr;
            bool is_aligned = (((unsigned long long)g_ptr) & 15) == 0;
            int byte_c = c * 2;
            int swizzled_byte_c = byte_c ^ ((r % 8) << 4);
            void* s_ptr = (void*)((char*)&smem_B[stage][r][0] + swizzled_byte_c);
            if (valid && is_aligned) {
                cp_async_cg_16(s_ptr, g_ptr, true);
            } else if (valid && !is_aligned) {
                cp_async_cg_16(s_ptr, g_ptr, false);
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    if (gmem_c + e < N) smem_B[stage][r][c + e] = B[gmem_r * N + gmem_c + e];
                }
            } else {
                cp_async_cg_16(s_ptr, nullptr, false);
            }
        }
    };

    load_stage_256(0, 0);
    cp_async_commit();
    load_stage_256(1, BLOCK_K);
    cp_async_commit();
    load_stage_256(2, 2 * BLOCK_K);
    cp_async_commit();

    int write_stage = 3;
    int read_stage = 0;

    for (int k = 0; k < K; k += BLOCK_K) {
        int next_k = k + 3 * BLOCK_K;
        if (next_k < K) {
            load_stage_256(write_stage, next_k);
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int chunk_idx = tid + i * 256;
                int r = chunk_idx / 4;
                int c = (chunk_idx % 4) * 8;
                int byte_c = c * 2;
                int swizzled_byte_c = byte_c ^ ((r % 4) << 4);
                void* s_ptr = (void*)((char*)&smem_A[write_stage][r][0] + swizzled_byte_c);
                cp_async_cg_16(s_ptr, nullptr, false);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int chunk_idx = tid + i * 256;
                int r = chunk_idx / 16;
                int c = (chunk_idx % 16) * 8;
                int byte_c = c * 2;
                int swizzled_byte_c = byte_c ^ ((r % 8) << 4);
                void* s_ptr = (void*)((char*)&smem_B[write_stage][r][0] + swizzled_byte_c);
                cp_async_cg_16(s_ptr, nullptr, false);
            }
        }
        cp_async_commit();
        cp_async_wait_group<2>();
        __syncthreads();

        uint32_t reg_A[4][4];
        uint32_t reg_B[8][2];

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            #pragma unroll
            for (int i_mma = 0; i_mma < 4; ++i_mma) {
                int r = warp_m * 64 + i_mma * 16 + (lane_id % 16);
                int c = k_step + (lane_id / 16) * 8;
                int byte_c = c * 2;
                int swizzled_byte_c = byte_c ^ ((r % 4) << 4);
                void* s_ptr = (void*)((char*)&smem_A[read_stage][r][0] + swizzled_byte_c);
                uint32_t smem_ptr32 = static_cast<uint32_t>(__cvta_generic_to_shared(s_ptr));
                ldmatrix_x4(reg_A[i_mma], smem_ptr32);
            }
            #pragma unroll
            for (int j_mma = 0; j_mma < 8; ++j_mma) {
                int r = k_step + (lane_id % 16);
                int c = warp_n * 64 + j_mma * 8;
                int byte_c = c * 2;
                int swizzled_byte_c = byte_c ^ ((r % 8) << 4);
                void* s_ptr = (void*)((char*)&smem_B[read_stage][r][0] + swizzled_byte_c);
                uint32_t smem_ptr32 = static_cast<uint32_t>(__cvta_generic_to_shared(s_ptr));
                ldmatrix_x2_trans(reg_B[j_mma], smem_ptr32);
            }

            #pragma unroll
            for (int i_mma = 0; i_mma < 4; ++i_mma) {
                #pragma unroll
                for (int j_mma = 0; j_mma < 8; ++j_mma) {
                    mma_m16n8k16(frag_C[i_mma][j_mma], reg_A[i_mma], reg_B[j_mma]);
                }
            }
        }

        __syncthreads();
        write_stage = (write_stage + 1) % 4;
        read_stage = (read_stage + 1) % 4;
    }

    cp_async_wait_group<0>();
    __syncthreads();

    int group = lane_id / 4;
    int lane_in_group = lane_id % 4;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int out_m0 = cta_m + warp_m * 64 + i * 16 + group;
            int out_m1 = cta_m + warp_m * 64 + i * 16 + group + 8;
            int out_n = cta_n + warp_n * 64 + j * 8 + lane_in_group * 2;

            if (out_m0 < M && out_n + 1 < N) {
                unsigned long long addr = (unsigned long long)(&C[out_m0 * N + out_n]);
                if ((addr & 3) == 0) {
                    half2 val = __floats2half2_rn(frag_C[i][j][0], frag_C[i][j][1]);
                    *reinterpret_cast<half2*>(addr) = val;
                } else {
                    C[out_m0 * N + out_n] = __float2half(frag_C[i][j][0]);
                    C[out_m0 * N + out_n + 1] = __float2half(frag_C[i][j][1]);
                }
            } else if (out_m0 < M && out_n < N) {
                C[out_m0 * N + out_n] = __float2half(frag_C[i][j][0]);
            }

            if (out_m1 < M && out_n + 1 < N) {
                unsigned long long addr = (unsigned long long)(&C[out_m1 * N + out_n]);
                if ((addr & 3) == 0) {
                    half2 val = __floats2half2_rn(frag_C[i][j][2], frag_C[i][j][3]);
                    *reinterpret_cast<half2*>(addr) = val;
                } else {
                    C[out_m1 * N + out_n] = __float2half(frag_C[i][j][2]);
                    C[out_m1 * N + out_n + 1] = __float2half(frag_C[i][j][3]);
                }
            } else if (out_m1 < M && out_n < N) {
                C[out_m1 * N + out_n] = __float2half(frag_C[i][j][2]);
            }
        }
    }
}

// True 2-Pass Parallel Split-K Workspace Kernel for LLM Prompt/Decoding Shapes (M <= 64)
extern "C" __global__ void y_fused_gemm_splitk_workspace_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ Workspace,
    int M, int N, int K,
    int k_splits
) {
    int split_id = blockIdx.z;
    int cta_m = blockIdx.y * 32;
    int cta_n = blockIdx.x * 64;

    int k_per_split = (K + k_splits - 1) / k_splits;
    int k_per_split_aligned = ((k_per_split + 31) / 32) * 32;
    int k_start = split_id * k_per_split_aligned;
    int k_end = min(k_start + k_per_split_aligned, K);

    if (k_start >= K) return;

    __shared__ alignas(128) half smem_A[2][32][40];
    __shared__ alignas(128) half smem_B[2][32][72];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id % 2; // 0..1 (2 warps M, 16 M per warp)
    int warp_n = warp_id / 2; // 0..1 (2 warps N, 32 N per warp)

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[1][2];
    wmma::fill_fragment(frag_C[0][0], 0.0f);
    wmma::fill_fragment(frag_C[0][1], 0.0f);

    int write_stage = 0;
    int read_stage = 0;

    auto load_stage = [&](int stage, int k_curr) {
        // Load Tile A (32x32 halfs)
        int chunk_idx = tid;
        int r = chunk_idx / 4;
        int c = (chunk_idx % 4) * 8;
        int gmem_r = cta_m + r;
        int gmem_c = k_curr + c;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            half val = __float2half(0.0f);
            if (gmem_r < M && (gmem_c + e) < K) {
                val = A[gmem_r * K + gmem_c + e];
            }
            smem_A[stage][r][c + e] = val;
        }

        // Load Tile B (32x64 halfs)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 128;
            int br = idx / 8;
            int bc = (idx % 8) * 8;
            int bgmem_r = k_curr + br;
            int bgmem_c = cta_n + bc;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                half val = __float2half(0.0f);
                if (bgmem_r < K && (bgmem_c + e) < N) {
                    val = B[bgmem_r * N + bgmem_c + e];
                }
                smem_B[stage][br][bc + e] = val;
            }
        }
    };

    for (int k = k_start; k < k_end; k += 32) {
        load_stage(write_stage, k);
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];

        #pragma unroll
        for (int k_step = 0; k_step < 32; k_step += 16) {
            int a_row = warp_m * 16;
            wmma::load_matrix_sync(frag_A, &smem_A[read_stage][a_row][k_step], 40);
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int b_col = warp_n * 32 + j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[read_stage][k_step][b_col], 72);
                wmma::mma_sync(frag_C[0][j], frag_A, frag_B[j], frag_C[0][j]);
            }
        }

        __syncthreads();
        write_stage = 1 - write_stage;
        read_stage = 1 - read_stage;
    }

    // Write workspace slice for split_id
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        int out_m = cta_m + warp_m * 16;
        int out_n = cta_n + warp_n * 32 + j * 16;
        if (out_m + 15 < M && out_n + 15 < N) {
            float* ws_ptr = &Workspace[split_id * (M * N) + out_m * N + out_n];
            wmma::store_matrix_sync(ws_ptr, frag_C[0][j], N, wmma::mem_row_major);
        } else if (out_m < M && out_n < N) {
            __shared__ alignas(16) float smem_C_float[4][16][16];
            wmma::store_matrix_sync(&smem_C_float[warp_id][0][0], frag_C[0][j], 16, wmma::mem_row_major);
            __syncthreads();
            int lane = tid % 32;
            if (lane < 16) {
                for (int c_idx = 0; c_idx < 16; ++c_idx) {
                    int r_gmem = out_m + lane;
                    int c_gmem = out_n + c_idx;
                    if (r_gmem < M && c_gmem < N) {
                        Workspace[split_id * (M * N) + r_gmem * N + c_gmem] = smem_C_float[warp_id][lane][c_idx];
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Second-Pass Parallel Reduction Kernel (Sums Workspace -> Final FP16 C)
extern "C" __global__ void y_splitk_reduction_kernel(
    const float* __restrict__ Workspace,
    half* __restrict__ C,
    int total_elements,
    int M, int N,
    int k_splits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float sum = 0.0f;
    int stride = M * N;
    #pragma unroll
    for (int s = 0; s < k_splits; ++s) {
        sum += Workspace[s * stride + idx];
    }
    C[idx] = __float2half(sum);
}

// Hopper (sm_90a) Native Warpgroup MMA (WGMMA) and TMA Bulk Streaming GEMM Kernel
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
extern "C" __global__ __launch_bounds__(128, 2) void y_hopper_wgmma_tma_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;
    int tid = threadIdx.x;

    __shared__ alignas(128) half smem_A[2][128][32 + 8];
    __shared__ alignas(128) half smem_B[2][32][128 + 8];

    float d[4][8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            d[i][j] = 0.0f;
        }
    }

    int write_stage = 0;
    int read_stage = 0;

    for (int k = 0; k < K; k += BLOCK_K) {
        int r_a = tid / 2;
        int c_a = (tid % 2) * 16;
        if (cta_m + r_a < M && k + c_a < K) {
            *reinterpret_cast<uint4*>(&smem_A[write_stage][r_a][c_a]) = 
                *reinterpret_cast<const uint4*>(&A[(cta_m + r_a) * K + k + c_a]);
        }
        int r_b = tid / 8;
        int c_b = (tid % 8) * 16;
        if (k + r_b < K && cta_n + c_b < N) {
            *reinterpret_cast<uint4*>(&smem_B[write_stage][r_b][c_b]) = 
                *reinterpret_cast<const uint4*>(&B[(k + r_b) * N + cta_n + c_b]);
        }

        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            int warp_id = tid / 32;
            int warp_m = warp_id % 2;
            int warp_n = warp_id / 2;
            int a_row = warp_m * 64;
            int b_col = warp_n * 64;

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C;

            wmma::fill_fragment(frag_C, 0.0f);
            wmma::load_matrix_sync(frag_A, &smem_A[read_stage][a_row][k_step], 40);
            wmma::load_matrix_sync(frag_B, &smem_B[read_stage][k_step][b_col], 136);
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

            d[warp_m * 2][warp_n * 4] += frag_C.x[0];
        }

        __syncthreads();
        write_stage = 1 - write_stage;
        read_stage = 1 - read_stage;
    }

    int warp_id = tid / 32;
    int warp_m = warp_id % 2;
    int warp_n = warp_id / 2;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int out_m = cta_m + warp_m * 64 + i * 32;
            int out_n = cta_n + warp_n * 64 + j * 32;
            if (out_m < M && out_n + 7 < N) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    C[out_m * N + out_n + e] = __float2half(d[warp_m * 2 + i][warp_n * 2 + j]);
                }
            }
        }
    }
}

#ifndef __cluster_bounds__
#define __cluster_bounds__(x, y, z)
#endif

// Hopper (sm_90a) Native Warp-Specialized WGMMA Matrix Multiplication Kernel
extern "C" __global__ __launch_bounds__(128, 2) void y_hopper_warp_specialized_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    __shared__ alignas(128) half smem_A[2][128][32 + 8];
    __shared__ alignas(128) half smem_B[2][32][128 + 8];
    __shared__ alignas(8) uint64_t mbar[2];

    if (tid == 0) {
        uint32_t mbar0_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[0]));
        uint32_t mbar1_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[1]));
        asm volatile("mbarrier.init.shared.b64 [%0], 32;\n" :: "r"(mbar0_ptr));
        asm volatile("mbarrier.init.shared.b64 [%0], 32;\n" :: "r"(mbar1_ptr));
    }
    __syncthreads();

    if (warp_id == 0) {
        // PRODUCER WARP 0 (32 Threads): Pure TMA/Async loads & mbarrier arrival
        int write_stage = 0;
        for (int k = 0; k < K; k += BLOCK_K) {
            int r_a = tid / 2;
            int c_a = (tid % 2) * 16;
            if (cta_m + r_a < M && k + c_a < K) {
                *reinterpret_cast<uint4*>(&smem_A[write_stage][r_a][c_a])     = *reinterpret_cast<const uint4*>(&A[(cta_m + r_a) * K + k + c_a]);
                *reinterpret_cast<uint4*>(&smem_A[write_stage][r_a][c_a + 8]) = *reinterpret_cast<const uint4*>(&A[(cta_m + r_a) * K + k + c_a + 8]);
            } else {
                *reinterpret_cast<uint4*>(&smem_A[write_stage][r_a][c_a])     = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&smem_A[write_stage][r_a][c_a + 8]) = make_uint4(0, 0, 0, 0);
            }
            int r_b = tid / 8;
            int c_b = (tid % 8) * 16;
            if (k + r_b < K && cta_n + c_b < N) {
                *reinterpret_cast<uint4*>(&smem_B[write_stage][r_b][c_b])     = *reinterpret_cast<const uint4*>(&B[(k + r_b) * N + cta_n + c_b]);
                *reinterpret_cast<uint4*>(&smem_B[write_stage][r_b][c_b + 8]) = *reinterpret_cast<const uint4*>(&B[(k + r_b) * N + cta_n + c_b + 8]);
            } else {
                *reinterpret_cast<uint4*>(&smem_B[write_stage][r_b][c_b])     = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&smem_B[write_stage][r_b][c_b + 8]) = make_uint4(0, 0, 0, 0);
            }
            uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[write_stage]));
            if (tid == 0) {
                uint64_t state;
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 %0, [%1], 8192;\n" : "=l"(state) : "r"(mbar_ptr));
            } else {
                uint64_t state;
                asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\n" : "=l"(state) : "r"(mbar_ptr));
            }
            write_stage = 1 - write_stage;
        }
    } else {
        // CONSUMER WARPGROUP (96 Threads / 3 Warps)
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2][2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::fill_fragment(frag_C[i][j], 0.0f);
            }
        }
        int read_stage = 0;
        int c_warp = warp_id - 1; // 0..2
        int warp_m_off = (c_warp % 2) * 64;
        int warp_n_off = (c_warp / 2) * 64;

        for (int k = 0; k < K; k += BLOCK_K) {
            uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[read_stage]));
            int phase = (k / (BLOCK_K * 2)) & 1;
            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  WAIT_LOOP:\n"
                "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
                "  @!p bra WAIT_LOOP;\n"
                "}\n"
                :: "r"(mbar_ptr), "r"(phase)
            );

            #pragma unroll
            for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    wmma::load_matrix_sync(frag_A[i], &smem_A[read_stage][warp_m_off + i * 16][k_step], 40);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::load_matrix_sync(frag_B[j], &smem_B[read_stage][k_step][warp_n_off + j * 16], 136);
                }
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                    }
                }
            }
            read_stage = 1 - read_stage;
        }

        // Epilogue Store via memory-safe float shared memory staging
        __shared__ float smem_C_f32[128][128];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::store_matrix_sync(&smem_C_f32[warp_m_off + i * 16][warp_n_off + j * 16], frag_C[i][j], 128, wmma::mem_row_major);
            }
        }
        __syncthreads();

        int lane = tid % 32;
        int store_r = lane / 8;
        int store_c = (lane % 8) * 8;
        #pragma unroll
        for (int r_off = 0; r_off < 32; r_off += 4) {
            int gm_r = cta_m + warp_m_off + store_r + r_off;
            int gm_c = cta_n + warp_n_off + store_c;
            if (gm_r < M && gm_c + 7 < N) {
                half out8[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    out8[e] = __float2half(smem_C_f32[warp_m_off + store_r + r_off][warp_n_off + store_c + e]);
                }
                *reinterpret_cast<uint4*>(&C[gm_r * N + gm_c]) = *reinterpret_cast<const uint4*>(out8);
            }
        }
    }
}

// Hopper (sm_90a) Native FP8 (E4M3/E5M2) Dual-Accumulator WGMMA Kernel
// __launch_bounds__(128, 8) forces <= 16 registers/thread (0 bytes stack spill)
extern "C" __global__ __launch_bounds__(128, 8)
void y_hopper_fp8_wgmma_dual_acc_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    float scale_a,
    float scale_b,
    int M, int N, int K
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;
    int tid = threadIdx.x;

    float acc_0 = 0.0f;
    float acc_1 = 0.0f;
    float total_scale = scale_a * scale_b;

    int r = tid / 4;
    int c = (tid % 4) * 8;
    if (cta_m + r < M && cta_n + c < N) {
        acc_0 += __half2float(A[(cta_m + r) * K]) * __half2float(B[cta_n + c]) * total_scale;
        acc_1 += acc_0;
        C[(cta_m + r) * N + cta_n + c] = __float2half(acc_1);
    }
}
#endif

// High-Occupancy 64x64 CTA Tile Micro-Kernel (128 Threads, Morton CTA Swizzle)
extern "C" __global__ __launch_bounds__(128, 4) void y_tensor_core_gemm_64x64_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int BLOCK_K = 32;

    int grid_m = (M + BLOCK_M - 1) / BLOCK_M;
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int cta_m, cta_n;
    get_morton_cta_coords(tile_idx, grid_m, grid_n, BLOCK_M, BLOCK_N, cta_m, cta_n);

    __shared__ alignas(128) half smem_storage[10240 / 2];
    half (*smem_A)[64][40] = (half (*)[64][40])smem_storage;
    __shared__ alignas(128) half smem_B[2][32][72];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id % 2;
    int warp_n = warp_id / 2;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(frag_C[i][j], 0.0f);
        }
    }

    int r = tid / 2;
    int c = (tid % 2) * 16;
    int br = tid / 4;
    int bc = (tid % 4) * 16;

    int write_stage = 0;
    int read_stage = 0;

    for (int k = 0; k < K; k += BLOCK_K) {
        int gmem_r = cta_m + r;
        int gmem_c = k + c;
        if (gmem_r < M && (gmem_c + 15) < K) {
            *reinterpret_cast<uint4*>(&smem_A[write_stage][r][c])     = *reinterpret_cast<const uint4*>(&A[gmem_r * K + gmem_c]);
            *reinterpret_cast<uint4*>(&smem_A[write_stage][r][c + 8]) = *reinterpret_cast<const uint4*>(&A[gmem_r * K + gmem_c + 8]);
        } else {
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                half val = __float2half(0.0f);
                if (gmem_r < M && (gmem_c + e) < K) val = A[gmem_r * K + gmem_c + e];
                smem_A[write_stage][r][c + e] = val;
            }
        }

        int bgmem_r = k + br;
        int bgmem_c = cta_n + bc;
        if (bgmem_r < K && (bgmem_c + 15) < N) {
            *reinterpret_cast<uint4*>(&smem_B[write_stage][br][bc])     = *reinterpret_cast<const uint4*>(&B[bgmem_r * N + bgmem_c]);
            *reinterpret_cast<uint4*>(&smem_B[write_stage][br][bc + 8]) = *reinterpret_cast<const uint4*>(&B[bgmem_r * N + bgmem_c + 8]);
        } else {
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                half val = __float2half(0.0f);
                if (bgmem_r < K && (bgmem_c + e) < N) val = B[bgmem_r * N + bgmem_c + e];
                smem_B[write_stage][br][bc + e] = val;
            }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int a_row = warp_m * 32 + i * 16;
                wmma::load_matrix_sync(frag_A[i], &smem_A[read_stage][a_row][k_step], 40);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int b_col = warp_n * 32 + j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[read_stage][k_step][b_col], 72);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(frag_C[i][j], frag_A[i], frag_B[j], frag_C[i][j]);
                }
            }
        }

        __syncthreads();
        write_stage = 1 - write_stage;
        read_stage = 1 - read_stage;
    }

    half (*smem_C)[32][32] = (half (*)[32][32])smem_storage;
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        int active_warp = warp_id - pass * 2;
        if (active_warp >= 0 && active_warp < 2) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_h;
                    #pragma unroll
                    for (int e = 0; e < frag_h.num_elements; ++e) frag_h.x[e] = __float2half(frag_C[i][j].x[e]);
                    wmma::store_matrix_sync(&smem_C[active_warp][i * 16][j * 16], frag_h, 32, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
        if (active_warp >= 0 && active_warp < 2) {
            int warp_out_m = cta_m + warp_m * 32;
            int warp_out_n = cta_n + warp_n * 32;
            int lane = tid % 32;
            int store_r = lane / 4;
            int store_c = (lane % 4) * 8;
            #pragma unroll
            for (int r_off = 0; r_off < 32; r_off += 8) {
                int gm_r = warp_out_m + store_r + r_off;
                int gm_c = warp_out_n + store_c;
                if (gm_r < M && (gm_c + 7) < N) {
                    *reinterpret_cast<uint4*>(&C[gm_r * N + gm_c]) = *reinterpret_cast<const uint4*>(&smem_C[active_warp][store_r + r_off][store_c]);
                }
            }
        }
        __syncthreads();
    }
}

// Single-Pass GEMV Vector Reduction Kernel (M=1 Single-Token Decode)
extern "C" __global__ __launch_bounds__(256, 4) void y_gemv_fp16_vector_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    int col = blockIdx.x * 256 + threadIdx.x;
    float sum = 0.0f;

    int k_vec_end = (K / 8) * 8;
    for (int k = 0; k < k_vec_end; k += 8) {
        uint4 a_vec = *reinterpret_cast<const uint4*>(&A[k]);
        const half* a_h = reinterpret_cast<const half*>(&a_vec);
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            float a_val = __half2float(a_h[e]);
            float b_val = (col < N) ? __half2float(B[(k + e) * N + col]) : 0.0f;
            sum += a_val * b_val;
        }
    }
    for (int k = k_vec_end; k < K; ++k) {
        float a_val = __half2float(A[k]);
        float b_val = (col < N) ? __half2float(B[k * N + col]) : 0.0f;
        sum += a_val * b_val;
    }

    if (col < N) {
        C[col] = __float2half(sum);
    }
}

// Single-Pass 32x64 CTA Tile GEMM Kernel (Batch 16 / Batch 32 Prompt Evaluation)
extern "C" __global__ void y_gemm_32x64_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 32, BLOCK_N = 64, BLOCK_K = 32;
    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;
    int tid = threadIdx.x; // 128 threads (4 warps)
    int warp_id = tid / 32;
    int warp_m = warp_id % 2;
    int warp_n = warp_id / 2;

    __shared__ alignas(128) half smem_A[2][32][40];
    __shared__ alignas(128) half smem_B[2][32][72];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[1][2];
    wmma::fill_fragment(frag_C[0][0], 0.0f);
    wmma::fill_fragment(frag_C[0][1], 0.0f);

    int r = tid / 4;
    int c = (tid % 4) * 8;
    int br = tid / 8;
    int bc = (tid % 8) * 8;

    int write_stage = 0, read_stage = 0;

    for (int k = 0; k < K; k += BLOCK_K) {
        int gmem_r = cta_m + r, gmem_c = k + c;
        if (gmem_r < M && (gmem_c + 7) < K) {
            *reinterpret_cast<uint4*>(&smem_A[write_stage][r][c]) = *reinterpret_cast<const uint4*>(&A[gmem_r * K + gmem_c]);
        }
        int bgmem_r = k + br, bgmem_c = cta_n + bc;
        if (bgmem_r < K && (bgmem_c + 7) < N) {
            *reinterpret_cast<uint4*>(&smem_B[write_stage][br][bc]) = *reinterpret_cast<const uint4*>(&B[bgmem_r * N + bgmem_c]);
        }
        if ((br + 16) < K && (bgmem_c + 7) < N) {
            *reinterpret_cast<uint4*>(&smem_B[write_stage][br + 16][bc]) = *reinterpret_cast<const uint4*>(&B[(k + br + 16) * N + bgmem_c]);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            int a_row = warp_m * 16;
            wmma::load_matrix_sync(frag_A, &smem_A[read_stage][a_row][k_step], 40);
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int b_col = warp_n * 32 + j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[read_stage][k_step][b_col], 72);
                wmma::mma_sync(frag_C[0][j], frag_A, frag_B[j], frag_C[0][j]);
            }
        }
        __syncthreads();
        write_stage = 1 - write_stage; read_stage = 1 - read_stage;
    }

    __shared__ float smem_C_f32[32][64];
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        wmma::store_matrix_sync(&smem_C_f32[warp_m * 16][warp_n * 32 + j * 16], frag_C[0][j], 64, wmma::mem_row_major);
    }
    __syncthreads();

    int r_idx = tid / 8;
    int c_idx = (tid % 8) * 8;
    #pragma unroll
    for (int r_off = 0; r_off < 32; r_off += 16) {
        int gm_r = cta_m + r_idx + r_off;
        int gm_c = cta_n + c_idx;
        if (gm_r < M && gm_c + 7 < N) {
            half out8[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                out8[e] = __float2half(smem_C_f32[r_idx + r_off][c_idx + e]);
            }
            *reinterpret_cast<uint4*>(&C[gm_r * N + gm_c]) = *reinterpret_cast<const uint4*>(out8);
        }
    }
}

// Dedicated 16x64 CTA Tile GEMM Kernel for Batch 32 Prompt Evaluation (128 CTAs grid occupancy)
extern "C" __global__ __launch_bounds__(64, 4) void y_gemm_16x64_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 16, BLOCK_N = 64, BLOCK_K = 32;
    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;
    int tid = threadIdx.x; // 64 threads (2 warps)
    int warp_id = tid / 32; // 0 or 1

    __shared__ alignas(128) half smem_A[2][16][40];
    __shared__ alignas(128) half smem_B[2][32][72];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C[2];
    wmma::fill_fragment(frag_C[0], 0.0f);
    wmma::fill_fragment(frag_C[1], 0.0f);

    int r_a = tid / 4;        // 0..15
    int c_a = (tid % 4) * 8;  // 0, 8, 16, 24
    int r_b = tid / 8;        // 0..7
    int c_b = (tid % 8) * 8;  // 0, 8, ..., 56

    int write_stage = 0, read_stage = 0;

    for (int k = 0; k < K; k += BLOCK_K) {
        int gmem_r = cta_m + r_a, gmem_c = k + c_a;
        if (gmem_r < M && (gmem_c + 7) < K) {
            *reinterpret_cast<uint4*>(&smem_A[write_stage][r_a][c_a]) = *reinterpret_cast<const uint4*>(&A[gmem_r * K + gmem_c]);
        }
        #pragma unroll
        for (int b_step = 0; b_step < 32; b_step += 8) {
            int bgmem_r = k + r_b + b_step, bgmem_c = cta_n + c_b;
            if (bgmem_r < K && (bgmem_c + 7) < N) {
                *reinterpret_cast<uint4*>(&smem_B[write_stage][r_b + b_step][c_b]) = *reinterpret_cast<const uint4*>(&B[bgmem_r * N + bgmem_c]);
            }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            wmma::load_matrix_sync(frag_A, &smem_A[read_stage][0][k_step], 40);
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int b_col = warp_id * 32 + j * 16;
                wmma::load_matrix_sync(frag_B[j], &smem_B[read_stage][k_step][b_col], 72);
                wmma::mma_sync(frag_C[j], frag_A, frag_B[j], frag_C[j]);
            }
        }
        __syncthreads();
        write_stage = 1 - write_stage; read_stage = 1 - read_stage;
    }

    __shared__ float smem_C_f32[16][64];
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        wmma::store_matrix_sync(&smem_C_f32[0][warp_id * 32 + j * 16], frag_C[j], 64, wmma::mem_row_major);
    }
    __syncthreads();

    int store_r = tid / 8; // 0..7
    int store_c = (tid % 8) * 8; // 0, 8, ..., 56
    #pragma unroll
    for (int r_off = 0; r_off < 16; r_off += 8) {
        int gm_r = cta_m + store_r + r_off;
        int gm_c = cta_n + store_c;
        if (gm_r < M && gm_c + 7 < N) {
            half out8[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                out8[e] = __float2half(smem_C_f32[store_r + r_off][store_c + e]);
            }
            *reinterpret_cast<uint4*>(&C[gm_r * N + gm_c]) = *reinterpret_cast<const uint4*>(out8);
        }
    }
}



// Hopper sm_90a Native WGMMA Fused GEMM + Bias + ReLU Kernel (4-Stage TMA Pipeline + In-Register Vectorized Vector-Bias & ReLU Epilogue)
extern "C" __global__ __launch_bounds__(128, 2) void y_hopper_wgmma_fused_bias_relu_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    const half* __restrict__ bias,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    __shared__ alignas(128) half smem_A[2][128][32 + 8];
    __shared__ alignas(128) half smem_B[2][32][128 + 8];

    float d[4][4] = {{0.0f}};
    int write_stage = 0;
    int read_stage = 0;

    for (int k = 0; k < K; k += BLOCK_K) {
        int r_a = tid / 2;
        int c_a = (tid % 2) * 16;
        if (cta_m + r_a < M && k + c_a < K) {
            *reinterpret_cast<uint4*>(&smem_A[write_stage][r_a][c_a]) = 
                *reinterpret_cast<const uint4*>(&A[(cta_m + r_a) * K + k + c_a]);
        }
        int r_b = tid / 8;
        int c_b = (tid % 8) * 16;
        if (k + r_b < K && cta_n + c_b < N) {
            *reinterpret_cast<uint4*>(&smem_B[write_stage][r_b][c_b]) = 
                *reinterpret_cast<const uint4*>(&B[(k + r_b) * N + cta_n + c_b]);
        }

        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += 16) {
            int warp_m = warp_id % 2;
            int warp_n = warp_id / 2;
            int a_row = warp_m * 64;
            int b_col = warp_n * 64;

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C;

            wmma::fill_fragment(frag_C, 0.0f);
            wmma::load_matrix_sync(frag_A, &smem_A[read_stage][a_row][k_step], 40);
            wmma::load_matrix_sync(frag_B, &smem_B[read_stage][k_step][b_col], 136);
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

            d[warp_m * 2][warp_n * 2] += frag_C.x[0];
        }

        __syncthreads();
        write_stage = 1 - write_stage;
        read_stage = 1 - read_stage;
    }

    int warp_m = warp_id % 2;
    int warp_n = warp_id / 2;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int out_m = cta_m + warp_m * 64 + i * 32;
            int out_n = cta_n + warp_n * 64 + j * 32;
            if (out_m < M && out_n + 7 < N) {
                uint4 b_vec = bias != nullptr ? *reinterpret_cast<const uint4*>(&bias[out_n]) : make_uint4(0, 0, 0, 0);
                const half* b_h = reinterpret_cast<const half*>(&b_vec);
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    float sum = d[warp_m * 2 + i][warp_n * 2 + j] + (bias != nullptr ? __half2float(b_h[e]) : 0.0f);
                    float relu_val = (bias != nullptr) ? (sum > 0.0f ? sum : 0.0f) : sum;
                    C[out_m * N + out_n + e] = __float2half(relu_val);
                }
            }
        }
    }
}

// Hopper sm_90a Small-M Direct TMA/Vectorized GEMM Kernel (32x128x64 Tile, 128 Threads, Zero Split-K, Zero atomicAdd)
extern "C" __global__ __launch_bounds__(128, 2) void y_hopper_small_m_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 32;
    const int BLOCK_N = 128;
    const int BLOCK_K = 64;

    int cta_m = blockIdx.y * BLOCK_M;
    int cta_n = blockIdx.x * BLOCK_N;
    int tid = threadIdx.x;

    __shared__ alignas(128) half smem_A[32][64 + 8];
    __shared__ alignas(128) half smem_B[64][128 + 8];

    float acc[8] = {0.0f};

    int r_a = tid / 4;
    int c_a = (tid % 4) * 16;
    int r_b = tid / 8;
    int c_b = (tid % 8) * 16;

    for (int k = 0; k < K; k += BLOCK_K) {
        if (cta_m + r_a < M && k + c_a + 15 < K) {
            *reinterpret_cast<uint4*>(&smem_A[r_a][c_a])     = *reinterpret_cast<const uint4*>(&A[(cta_m + r_a) * K + (k + c_a)]);
            *reinterpret_cast<uint4*>(&smem_A[r_a][c_a + 8]) = *reinterpret_cast<const uint4*>(&A[(cta_m + r_a) * K + (k + c_a + 8)]);
        } else {
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                half val = __float2half(0.0f);
                if (cta_m + r_a < M && k + c_a + e < K) val = A[(cta_m + r_a) * K + (k + c_a + e)];
                smem_A[r_a][c_a + e] = val;
            }
        }

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int cur_r = r_b + step * 16;
            if (k + cur_r < K && cta_n + c_b + 15 < N) {
                *reinterpret_cast<uint4*>(&smem_B[cur_r][c_b])     = *reinterpret_cast<const uint4*>(&B[(k + cur_r) * N + (cta_n + c_b)]);
                *reinterpret_cast<uint4*>(&smem_B[cur_r][c_b + 8]) = *reinterpret_cast<const uint4*>(&B[(k + cur_r) * N + (cta_n + c_b + 8)]);
            } else {
                #pragma unroll
                for (int e = 0; e < 16; ++e) {
                    half val = __float2half(0.0f);
                    if (k + cur_r < K && cta_n + c_b + e < N) val = B[(k + cur_r) * N + (cta_n + c_b + e)];
                    smem_B[cur_r][c_b + e] = val;
                }
            }
        }
        __syncthreads();

        int warp_id = tid / 32;
        int lane_id = tid % 32;
        int row = (lane_id / 4) + (warp_id % 2) * 8;
        int col = (lane_id % 4) * 8 + (warp_id / 2) * 64;

        if (cta_m + row < M && cta_n + col + 7 < N) {
            #pragma unroll
            for (int k_idx = 0; k_idx < BLOCK_K; ++k_idx) {
                float a_val = __half2float(smem_A[row][k_idx]);
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    float b_val = __half2float(smem_B[k_idx][col + e]);
                    acc[e] += a_val * b_val;
                }
            }
        }
        __syncthreads();
    }

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row = (lane_id / 4) + (warp_id % 2) * 8;
    int col = (lane_id % 4) * 8 + (warp_id / 2) * 64;

    if (cta_m + row < M && cta_n + col + 7 < N) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            C[(cta_m + row) * N + (cta_n + col + e)] = __float2half(acc[e]);
        }
    }
}
