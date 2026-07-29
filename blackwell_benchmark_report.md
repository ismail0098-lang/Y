# Blackwell / Next-Gen GPU Benchmark Report
- **GPU Hardware**: `NVIDIA GeForce RTX 4070 Ti SUPER`
- **Compute Capability**: `sm_89`
- **VRAM**: `16.71 GB`
- **PyTorch / CUDA**: `2.12.0+cu130` / `13.0`

## 1. Standalone Dense GEMM Benchmarks (M=N=K)
| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |
|---|:---:|:---:|:---:|:---:|:---:|
| 512x512x512 | 7.21 | 15.73 | 17.07 | 0.46x | WARN |
| 1024x1024x1024 | 33.67 | 38.36 | 55.98 | 0.88x | WARN |
| 2048x2048x2048 | 377.65 | 595.46 | 28.85 | 0.63x | WARN |
| 4096x4096x4096 | 6447.92 | 2882.36 | 47.68 | 2.24x | WARN |
| 8192x8192x8192 | 12994.94 | 22452.90 | 48.97 | 0.58x | WARN |
| 16384x16384x16384 | 102440.86 | 213541.38 | 41.19 | 0.48x | WARN |

## 2. Real-World LLM Inference & Prompt Decoding Shapes
| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1x4096x4096 | LLaMA 7B Single-Token Decode | 67.24 | 38.03 | 882.6 GB/s | 1.77x | PASSED |
| 1x11008x4096 | LLaMA 7B SwiGLU FFN Gate/Up | 160.09 | 495.51 | 182.1 GB/s | 0.32x | PASSED |
| 16x4096x4096 | Batch 16 Prompt Evaluation | 50.65 | 39.38 | 858.7 GB/s | 1.29x | PASSED |
| 32x4096x4096 | Batch 32 Prompt Evaluation | 24.07 | 41.23 | 826.6 GB/s | 0.58x | PASSED |

## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)
| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |
|---|:---:|:---:|:---:|
| 512x512x512 | 10.25 | 14.60 | 0.70x |
| 1024x1024x1024 | 35.00 | 29.61 | 1.18x |
| 2048x2048x2048 | 235.92 | 158.33 | 1.49x |
| 4096x4096x4096 | 1851.53 | 1219.99 | 1.52x |