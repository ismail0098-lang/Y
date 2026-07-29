# Blackwell / Next-Gen GPU Benchmark Report
- **GPU Hardware**: `NVIDIA GeForce RTX 4070 Ti SUPER`
- **Compute Capability**: `sm_89`
- **VRAM**: `16.71 GB`
- **PyTorch / CUDA**: `2.12.0+cu130` / `13.0`

## 1. Standalone Dense GEMM Benchmarks (M=N=K)
| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |
|---|:---:|:---:|:---:|:---:|:---:|
| 512x512x512 | 26.09 | 64.35 | 4.17 | 0.41x | PASSED |
| 1024x1024x1024 | 263.43 | 836.87 | 2.57 | 0.31x | PASSED |
| 2048x2048x2048 | 954.06 | 368.87 | 46.57 | 2.59x | PASSED |
| 4096x4096x4096 | 1752.01 | 2935.82 | 46.81 | 0.60x | PASSED |
| 8192x8192x8192 | 13034.91 | 22593.79 | 48.66 | 0.58x | PASSED |
| 16384x16384x16384 | 102032.39 | 213552.64 | 41.19 | 0.48x | WARN |

## 2. Real-World LLM Inference & Prompt Decoding Shapes
| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1x4096x4096 | LLaMA 7B Single-Token Decode | 26.85 | 52.41 | 640.6 GB/s | 0.51x | PASSED |
| 1x11008x4096 | LLaMA 7B SwiGLU FFN Gate/Up | 191.80 | 184.40 | 489.2 GB/s | 1.04x | PASSED |
| 16x4096x4096 | Batch 16 Prompt Evaluation | 56.28 | 50.40 | 670.9 GB/s | 1.12x | PASSED |
| 32x4096x4096 | Batch 32 Prompt Evaluation | 25.03 | 58.06 | 586.9 GB/s | 0.43x | PASSED |

## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)
| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |
|---|:---:|:---:|:---:|
| 512x512x512 | 10.28 | 14.58 | 0.71x |
| 1024x1024x1024 | 34.98 | 29.76 | 1.18x |
| 2048x2048x2048 | 223.15 | 162.84 | 1.37x |
| 4096x4096x4096 | 1824.90 | 1208.18 | 1.51x |