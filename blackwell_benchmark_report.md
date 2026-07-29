# Blackwell / Next-Gen GPU Benchmark Report
- **GPU Hardware**: `NVIDIA GeForce RTX 4070 Ti SUPER`
- **Compute Capability**: `sm_89`
- **VRAM**: `16.71 GB`
- **PyTorch / CUDA**: `2.12.0+cu130` / `13.0`

## 1. Standalone Dense GEMM Benchmarks (M=N=K)
| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |
|---|:---:|:---:|:---:|:---:|:---:|
| 512x512x512 | 25.83 | 62.57 | 4.29 | 0.41x | PASSED |
| 1024x1024x1024 | 332.94 | 679.94 | 3.16 | 0.49x | PASSED |
| 2048x2048x2048 | 1833.55 | 367.86 | 46.70 | 4.98x | PASSED |
| 4096x4096x4096 | 1693.83 | 2903.76 | 47.33 | 0.58x | PASSED |
| 8192x8192x8192 | 13028.35 | 22758.20 | 48.31 | 0.57x | PASSED |
| 16384x16384x16384 | 102728.17 | 212273.67 | 41.44 | 0.48x | WARN |

## 2. Real-World LLM Inference & Prompt Decoding Shapes
| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1x4096x4096 | LLaMA 7B Single-Token Decode | 28.30 | 45.01 | 745.8 GB/s | 0.63x | PASSED |
| 1x11008x4096 | LLaMA 7B SwiGLU FFN Gate/Up | 187.11 | 187.04 | 482.3 GB/s | 1.00x | PASSED |
| 16x4096x4096 | Batch 16 Prompt Evaluation | 50.67 | 49.62 | 681.5 GB/s | 1.02x | PASSED |
| 32x4096x4096 | Batch 32 Prompt Evaluation | 24.12 | 57.26 | 595.1 GB/s | 0.42x | PASSED |

## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)
| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |
|---|:---:|:---:|:---:|
| 512x512x512 | 10.12 | 14.38 | 0.70x |
| 1024x1024x1024 | 34.96 | 29.47 | 1.19x |
| 2048x2048x2048 | 232.10 | 153.70 | 1.51x |
| 4096x4096x4096 | 1820.88 | 1189.27 | 1.53x |