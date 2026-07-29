# Blackwell / Next-Gen GPU Benchmark Report
- **GPU Hardware**: `NVIDIA GeForce RTX 4070 Ti SUPER`
- **Compute Capability**: `sm_89`
- **VRAM**: `16.71 GB`
- **PyTorch / CUDA**: `2.12.0+cu130` / `13.0`

## 1. Standalone Dense GEMM Benchmarks (M=N=K)
| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |
|---|:---:|:---:|:---:|:---:|:---:|
| 512x512x512 | 9.99 | 23.20 | 11.57 | 0.43x | PASSED |
| 1024x1024x1024 | 61.44 | 54.48 | 39.42 | 1.13x | PASSED |
| 2048x2048x2048 | 618.35 | 1050.09 | 16.36 | 0.59x | PASSED |
| 4096x4096x4096 | 6137.23 | 2949.09 | 46.60 | 2.08x | PASSED |
| 8192x8192x8192 | 13078.53 | 22227.56 | 49.47 | 0.59x | PASSED |
| 16384x16384x16384 | 101878.10 | 213629.36 | 41.17 | 0.48x | WARN |

## 2. Real-World LLM Inference & Prompt Decoding Shapes
| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1x4096x4096 | LLaMA 7B Single-Token Decode | 66.52 | 38.28 | 877.0 GB/s | 1.74x | PASSED |
| 1x11008x4096 | LLaMA 7B SwiGLU FFN Gate/Up | 188.14 | 169.49 | 532.2 GB/s | 1.11x | WARN |
| 16x4096x4096 | Batch 16 Prompt Evaluation | 50.67 | 40.20 | 841.2 GB/s | 1.26x | PASSED |
| 32x4096x4096 | Batch 32 Prompt Evaluation | 24.06 | 42.21 | 807.4 GB/s | 0.57x | PASSED |

## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)
| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |
|---|:---:|:---:|:---:|
| 512x512x512 | 10.24 | 14.60 | 0.70x |
| 1024x1024x1024 | 35.01 | 29.78 | 1.18x |
| 2048x2048x2048 | 231.28 | 156.30 | 1.48x |
| 4096x4096x4096 | 1834.19 | 1217.26 | 1.51x |