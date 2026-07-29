# Blackwell / Next-Gen GPU Benchmark Report
- **GPU Hardware**: `NVIDIA GeForce RTX 4070 Ti SUPER`
- **Compute Capability**: `sm_89`
- **VRAM**: `16.71 GB`
- **PyTorch / CUDA**: `2.12.0+cu130` / `13.0`

## 1. Standalone Dense GEMM Benchmarks (M=N=K)
| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |
|---|:---:|:---:|:---:|:---:|:---:|
| 512x512x512 | 24.72 | 287.91 | 0.93 | 0.09x | PASSED |
| 1024x1024x1024 | 284.65 | 555.93 | 3.86 | 0.51x | PASSED |
| 2048x2048x2048 | 1389.98 | 522.94 | 32.85 | 2.66x | PASSED |
| 4096x4096x4096 | 1750.97 | 3297.46 | 41.68 | 0.53x | PASSED |
| 8192x8192x8192 | 13079.70 | 23829.95 | 46.14 | 0.55x | PASSED |
| 16384x16384x16384 | 101803.31 | 220808.65 | 39.84 | 0.46x | WARN |

## 2. Real-World LLM Inference & Prompt Decoding Shapes
| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1x4096x4096 | LLaMA 7B Single-Token Decode | 27.01 | 44.97 | 746.4 GB/s | 0.60x | PASSED |
| 1x11008x4096 | LLaMA 7B SwiGLU FFN Gate/Up | 175.63 | 191.90 | 470.1 GB/s | 0.92x | PASSED |
| 16x4096x4096 | Batch 16 Prompt Evaluation | 50.69 | 59.74 | 566.1 GB/s | 0.85x | PASSED |
| 32x4096x4096 | Batch 32 Prompt Evaluation | 24.04 | 58.12 | 586.4 GB/s | 0.41x | PASSED |

## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)
| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |
|---|:---:|:---:|:---:|
| 512x512x512 | 10.28 | 14.38 | 0.72x |
| 1024x1024x1024 | 35.06 | 29.49 | 1.19x |
| 2048x2048x2048 | 225.30 | 154.09 | 1.46x |
| 4096x4096x4096 | 1833.37 | 1219.33 | 1.50x |