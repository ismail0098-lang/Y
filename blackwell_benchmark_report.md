# Blackwell / Next-Gen GPU Benchmark Report
- **GPU Hardware**: `NVIDIA GeForce RTX 4070 Ti SUPER`
- **Compute Capability**: `sm_89`
- **VRAM**: `16.71 GB`
- **PyTorch / CUDA**: `2.12.0+cu130` / `13.0`

## 1. Standalone Dense GEMM Benchmarks (M=N=K)
| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |
|---|:---:|:---:|:---:|:---:|:---:|
| 512x512x512 | 23.55 | 162.26 | 1.65 | 0.15x | WARN |
| 1024x1024x1024 | 225.20 | 511.06 | 4.20 | 0.44x | WARN |
| 2048x2048x2048 | 1524.88 | 1642.99 | 10.46 | 0.93x | WARN |
| 4096x4096x4096 | 1762.22 | 2860.23 | 48.05 | 0.62x | WARN |
| 8192x8192x8192 | 13112.93 | 22089.31 | 49.78 | 0.59x | WARN |
| 16384x16384x16384 | 101605.38 | 210845.00 | 41.72 | 0.48x | WARN |

## 2. Real-World LLM Inference & Prompt Decoding Shapes
| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1x4096x4096 | LLaMA 7B Single-Token Decode | 26.91 | 51.02 | 658.0 GB/s | 0.53x | PASSED |
| 1x11008x4096 | LLaMA 7B SwiGLU FFN Gate/Up | 174.18 | 192.55 | 468.5 GB/s | 0.90x | PASSED |
| 16x4096x4096 | Batch 16 Prompt Evaluation | 50.50 | 49.52 | 682.9 GB/s | 1.02x | PASSED |
| 32x4096x4096 | Batch 32 Prompt Evaluation | 24.04 | 57.16 | 596.2 GB/s | 0.42x | PASSED |

## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)
| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |
|---|:---:|:---:|:---:|
| 512x512x512 | 10.96 | 14.32 | 0.77x |
| 1024x1024x1024 | 35.55 | 29.33 | 1.21x |
| 2048x2048x2048 | 238.19 | 152.68 | 1.56x |
| 4096x4096x4096 | 1772.27 | 1212.89 | 1.46x |