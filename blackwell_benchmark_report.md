# Blackwell / Next-Gen GPU Benchmark Report
- **GPU Hardware**: `NVIDIA GeForce RTX 4070 Ti SUPER`
- **Compute Capability**: `sm_89`
- **VRAM**: `16.71 GB`
- **PyTorch / CUDA**: `2.12.0+cu130` / `13.0`

## 1. Standalone Dense GEMM Benchmarks (M=N=K)
| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |
|---|:---:|:---:|:---:|:---:|:---:|
| 512x512x512 | 63.63 | 460.92 | 0.58 | 0.14x | PASSED |
| 1024x1024x1024 | 394.85 | 674.73 | 3.18 | 0.59x | PASSED |
| 2048x2048x2048 | 1994.12 | 533.28 | 32.22 | 3.74x | PASSED |
| 4096x4096x4096 | 1640.99 | 3918.15 | 35.08 | 0.42x | PASSED |
| 8192x8192x8192 | 12617.98 | 30771.67 | 35.73 | 0.41x | PASSED |
| 16384x16384x16384 | 101243.90 | 265544.22 | 33.12 | 0.38x | WARN |

## 2. Real-World LLM Inference & Prompt Decoding Shapes
| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1x4096x4096 | LLaMA 7B Single-Token Decode | 36.89 | 45.28 | 741.4 GB/s | 0.81x | PASSED |
| 1x11008x4096 | LLaMA 7B SwiGLU FFN Gate/Up | 176.37 | 185.65 | 485.9 GB/s | 0.95x | PASSED |
| 16x4096x4096 | Batch 16 Prompt Evaluation | 50.73 | 49.70 | 680.3 GB/s | 1.02x | PASSED |
| 32x4096x4096 | Batch 32 Prompt Evaluation | 24.35 | 57.54 | 592.3 GB/s | 0.42x | PASSED |

## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)
| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |
|---|:---:|:---:|:---:|
| 512x512x512 | 10.36 | 14.37 | 0.72x |
| 1024x1024x1024 | 35.12 | 29.49 | 1.19x |
| 2048x2048x2048 | 224.91 | 156.85 | 1.43x |
| 4096x4096x4096 | 1764.90 | 1186.61 | 1.49x |