# Blackwell / Next-Gen GPU Benchmark Report
- **GPU Hardware**: `NVIDIA GeForce RTX 4070 Ti SUPER`
- **Compute Capability**: `sm_89`
- **VRAM**: `16.71 GB`
- **PyTorch / CUDA**: `2.12.0+cu130` / `13.0`

## 1. Standalone Dense GEMM Benchmarks (M=N=K)
| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |
|---|:---:|:---:|:---:|:---:|:---:|
| 512x512x512 | 10.53 | 41.94 | 6.40 | 0.25x | WARN |
| 1024x1024x1024 | 62.87 | 101.50 | 21.16 | 0.62x | WARN |
| 2048x2048x2048 | 433.44 | 1072.09 | 16.02 | 0.40x | WARN |
| 4096x4096x4096 | 4117.18 | 2895.94 | 47.46 | 1.42x | WARN |
| 8192x8192x8192 | 12763.34 | 21227.94 | 51.80 | 0.60x | WARN |
| 16384x16384x16384 | 99796.16 | 207717.96 | 42.35 | 0.48x | WARN |

## 2. Real-World LLM Inference & Prompt Decoding Shapes
| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1x4096x4096 | LLaMA 7B Single-Token Decode | 26.91 | 45.02 | 745.8 GB/s | 0.60x | PASSED |
| 1x11008x4096 | LLaMA 7B SwiGLU FFN Gate/Up | 175.65 | 186.51 | 483.7 GB/s | 0.94x | PASSED |
| 16x4096x4096 | Batch 16 Prompt Evaluation | 50.68 | 50.28 | 672.6 GB/s | 1.01x | PASSED |
| 32x4096x4096 | Batch 32 Prompt Evaluation | 24.21 | 57.38 | 593.9 GB/s | 0.42x | PASSED |

## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)
| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |
|---|:---:|:---:|:---:|
| 512x512x512 | 10.30 | 14.40 | 0.72x |
| 1024x1024x1024 | 35.04 | 29.35 | 1.19x |
| 2048x2048x2048 | 224.32 | 160.15 | 1.40x |
| 4096x4096x4096 | 1768.92 | 1187.64 | 1.49x |