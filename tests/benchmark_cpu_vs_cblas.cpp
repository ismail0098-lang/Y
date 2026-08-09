// ============================================================
//  Y  —  REAL OPENBLAS HARDWARE BENCHMARK: Y vs REAL OpenBLAS
//  tests/benchmark_cpu_vs_cblas.cpp
//
//  Executes physical benchmarks comparing Y against real compiled
//  OpenBLAS ZEN library across both 1-Thread and Multi-Thread (16 Cores) modes.
// ============================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <random>
#include <immintrin.h>
#include <omp.h>

extern "C" {
    #include "/tmp/openblas_build/cblas.h"
    void openblas_set_num_threads(int num_threads);
}

inline void do_not_optimize(void* ptr) {
    asm volatile("" : : "g"(ptr) : "memory");
}

// ------------------------------------------------------------
// 1. Y Specialized CPU Kernels
// ------------------------------------------------------------

void y_small_direct_gemm(const float* __restrict A, const float* __restrict B, float* __restrict C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void y_decode_gemv(const float* __restrict A, const float* __restrict B, float* __restrict C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        float* __restrict C_row = C + i * N;
        std::memset(C_row, 0, N * sizeof(float));

        for (int k = 0; k < K; ++k) {
            float a_val = A[i * K + k];
            const float* __restrict B_row = B + k * N;
            #pragma omp simd
            for (int j = 0; j < N; ++j) {
                C_row[j] += a_val * B_row[j];
            }
        }
    }
}

void y_deep_k_gemm(const float* __restrict A, const float* __restrict B, float* __restrict C, int M, int N, int K) {
    std::memset(C, 0, M * N * sizeof(float));

    #pragma omp parallel
    {
        std::vector<float> C_local(M * N, 0.0f);
        #pragma omp for nowait
        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < M; ++i) {
                float a_val = A[i * K + k];
                const float* __restrict B_row = B + k * N;
                #pragma omp simd
                for (int j = 0; j < N; ++j) {
                    C_local[i * N + j] += a_val * B_row[j];
                }
            }
        }

        #pragma omp critical
        {
            for (int idx = 0; idx < M * N; ++idx) {
                C[idx] += C_local[idx];
            }
        }
    }
}

void y_irregular_masked_gemm(const float* __restrict A, const float* __restrict B, float* __restrict C, int M, int N, int K) {
    constexpr int BLOCK_K = 64;
    std::memset(C, 0, M * N * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int bk = 0; bk < K; bk += BLOCK_K) {
            int k_end = std::min(bk + BLOCK_K, K);
            float* __restrict C_row = C + i * N;
            for (int k = bk; k < k_end; ++k) {
                float a_val = A[i * K + k];
                const float* __restrict B_row = B + k * N;
                #pragma omp simd
                for (int j = 0; j < N; ++j) {
                    #pragma omp atomic update
                    C_row[j] += a_val * B_row[j];
                }
            }
        }
    }
}

void y_nice_square_gemm(const float* __restrict A, const float* __restrict B, float* __restrict C, int M, int N, int K) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 64;

    std::memset(C, 0, M * N * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < M; bi += BLOCK_M) {
        for (int bj = 0; bj < N; bj += BLOCK_N) {
            int i_end = std::min(bi + BLOCK_M, M);
            int j_end = std::min(bj + BLOCK_N, N);

            for (int bk = 0; bk < K; bk += BLOCK_K) {
                int k_end = std::min(bk + BLOCK_K, K);

                for (int i = bi; i < i_end; ++i) {
                    for (int k = bk; k < k_end; ++k) {
                        float a_val = A[i * K + k];
                        #pragma omp simd
                        for (int j = bj; j < j_end; ++j) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------
// 2. Real OpenBLAS Invocation
// ------------------------------------------------------------
void openblas_reference_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K,
                B, N,
                0.0f, C, N);
}

// ------------------------------------------------------------
// 3. Benchmark Runner Utility
// ------------------------------------------------------------
struct BenchmarkResult {
    std::string regime_name;
    int M, N, K;
    double y_time_us;
    double openblas_time_us;
    double y_gflops;
    double openblas_gflops;
    double ratio;
    bool output_matched;
};

BenchmarkResult run_benchmark(
    const std::string& name,
    int M, int N, int K,
    void (*y_kernel)(const float*, const float*, float*, int, int, int),
    int warmup_iters = 3,
    int bench_iters = 10
) {
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_y(M * N, 0.0f);
    std::vector<float> C_openblas(M * N, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& val : A) val = dist(rng);
    for (auto& val : B) val = dist(rng);

    y_kernel(A.data(), B.data(), C_y.data(), M, N, K);
    openblas_reference_gemm(A.data(), B.data(), C_openblas.data(), M, N, K);
    
    double max_diff = 0.0;
    for (size_t idx = 0; idx < C_y.size(); ++idx) {
        max_diff = std::max(max_diff, (double)std::abs(C_y[idx] - C_openblas[idx]));
    }
    bool matched = max_diff < 1e-2;

    for (int i = 0; i < warmup_iters; ++i) {
        y_kernel(A.data(), B.data(), C_y.data(), M, N, K);
        do_not_optimize(C_y.data());
    }

    auto start_y = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        y_kernel(A.data(), B.data(), C_y.data(), M, N, K);
        do_not_optimize(C_y.data());
    }
    auto end_y = std::chrono::high_resolution_clock::now();
    double y_us = std::chrono::duration<double, std::micro>(end_y - start_y).count() / bench_iters;

    for (int i = 0; i < warmup_iters; ++i) {
        openblas_reference_gemm(A.data(), B.data(), C_openblas.data(), M, N, K);
        do_not_optimize(C_openblas.data());
    }

    auto start_openblas = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        openblas_reference_gemm(A.data(), B.data(), C_openblas.data(), M, N, K);
        do_not_optimize(C_openblas.data());
    }
    auto end_openblas = std::chrono::high_resolution_clock::now();
    double openblas_us = std::chrono::duration<double, std::micro>(end_openblas - start_openblas).count() / bench_iters;

    double total_flops = 2.0 * M * N * K;
    double y_gflops = (total_flops / (y_us * 1e-6)) / 1e9;
    double openblas_gflops = (total_flops / (openblas_us * 1e-6)) / 1e9;
    double ratio = openblas_us / y_us;

    return { name, M, N, K, y_us, openblas_us, y_gflops, openblas_gflops, ratio, matched };
}

void print_suite(const std::string& title, int threads) {
    omp_set_num_threads(threads);
    openblas_set_num_threads(threads);

    std::cout << "\n===============================================================" << std::endl;
    std::cout << " " << title << " (" << threads << " THREADS)" << std::endl;
    std::cout << "===============================================================" << std::endl;

    std::vector<BenchmarkResult> results;
    results.push_back(run_benchmark("SmallDirect (L1 Cache)", 16, 16, 16, y_small_direct_gemm, 10, 100));
    results.push_back(run_benchmark("DecodeGEMV (LLM Token)", 1, 4096, 4096, y_decode_gemv, 3, 10));
    results.push_back(run_benchmark("DeepK (Reduction Heavy)", 64, 64, 32768, y_deep_k_gemm, 3, 10));
    results.push_back(run_benchmark("IrregularMasked (Odd Shape)", 137, 391, 1013, y_irregular_masked_gemm, 3, 10));
    results.push_back(run_benchmark("NiceSquare (Large Square)", 1024, 1024, 1024, y_nice_square_gemm, 2, 5));

    std::cout << std::left 
              << std::setw(28) << "Regime / Shape"
              << std::setw(16) << "Dimensions"
              << std::setw(14) << "Y Time (us)"
              << std::setw(18) << "OpenBLAS Time (us)"
              << std::setw(12) << "Y GFLOPS"
              << std::setw(16) << "OpenBLAS GFLOPS"
              << std::setw(10) << "Match?"
              << std::setw(14) << "Ratio (OB/Y)"
              << std::endl;
    std::cout << std::string(128, '-') << std::endl;

    for (const auto& r : results) {
        std::string dim_str = std::to_string(r.M) + "x" + std::to_string(r.N) + "x" + std::to_string(r.K);
        std::cout << std::left
                  << std::setw(28) << r.regime_name
                  << std::setw(16) << dim_str
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.y_time_us
                  << std::setw(18) << std::fixed << std::setprecision(2) << r.openblas_time_us
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.y_gflops
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.openblas_gflops
                  << std::setw(10) << (r.output_matched ? "YES" : "NO")
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.ratio << "x"
                  << std::endl;
    }
    std::cout << "===============================================================" << std::endl;
}

int main() {
    print_suite("SINGLE-THREADED BENCHMARK", 1);
    print_suite("MULTI-THREADED BENCHMARK", 16);
    return 0;
}
