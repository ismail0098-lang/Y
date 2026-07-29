# Blackwell / Next-Gen GPU Benchmark Report
- **GPU Hardware**: `NVIDIA GeForce RTX 4070 Ti SUPER`
- **Compute Capability**: `sm_89`
- **VRAM**: `16.71 GB`
- **PyTorch / CUDA**: `2.12.0+cu130` / `13.0`

## 1. Standalone Dense GEMM Benchmarks (M=N=K)
| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |
|---|:---:|:---:|:---:|:---:|:---:|
| 512x512x512 | 13.84 | 56.38 | 4.76 | 0.25x | WARN |
| 1024x1024x1024 | 83.78 | 201.05 | 10.68 | 0.42x | WARN |
| 2048x2048x2048 | 1028.12 | 1375.89 | 12.49 | 0.75x | WARN |
| 4096x4096x4096 | 1741.95 | 2897.16 | 47.44 | 0.60x | WARN |
| 8192x8192x8192 | 13129.73 | 21849.08 | 50.32 | 0.60x | WARN |
| 16384x16384x16384 | 103009.28 | 211424.42 | 41.60 | 0.49x | WARN |

## 2. Real-World LLM Inference & Prompt Decoding Shapes
| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1x4096x4096 | LLaMA 7B Single-Token Decode | 26.93 | 50.11 | 669.9 GB/s | 0.54x | PASSED |
| 1x11008x4096 | LLaMA 7B SwiGLU FFN Gate/Up | 188.31 | 188.02 | 479.8 GB/s | 1.00x | PASSED |
| 16x4096x4096 | Batch 16 Prompt Evaluation | 50.81 | 49.74 | 679.8 GB/s | 1.02x | PASSED |
| 32x4096x4096 | Batch 32 Prompt Evaluation | 24.94 | 58.42 | 583.4 GB/s | 0.43x | PASSED |

## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)
| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |
|---|:---:|:---:|:---:|
| 512x512x512 | 10.30 | 14.38 | 0.72x |
| 1024x1024x1024 | 35.07 | 29.37 | 1.19x |
| 2048x2048x2048 | 220.78 | 165.87 | 1.33x |
| 4096x4096x4096 | 1841.04 | 1234.19 | 1.49x |