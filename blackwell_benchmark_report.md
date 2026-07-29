# Blackwell / Next-Gen GPU Benchmark Report
- **GPU Hardware**: `NVIDIA GeForce RTX 4070 Ti SUPER`
- **Compute Capability**: `sm_89`
- **VRAM**: `16.71 GB`
- **PyTorch / CUDA**: `2.12.0+cu130` / `13.0`

## 1. Standalone Dense GEMM Benchmarks (M=N=K)
| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |
|---|:---:|:---:|:---:|:---:|:---:|
| 512x512x512 | 7.99 | 19.74 | 13.60 | 0.40x | PASSED |
| 1024x1024x1024 | 42.93 | 213.98 | 10.04 | 0.20x | PASSED |
| 2048x2048x2048 | 461.12 | 616.55 | 27.86 | 0.75x | PASSED |
| 4096x4096x4096 | 1735.75 | 2914.70 | 47.15 | 0.60x | PASSED |
| 8192x8192x8192 | 13230.08 | 22491.75 | 48.89 | 0.59x | WARN |
| 16384x16384x16384 | 102179.33 | 213240.69 | 41.25 | 0.48x | WARN |

## 2. Real-World LLM Inference & Prompt Decoding Shapes
| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1x4096x4096 | LLaMA 7B Single-Token Decode | 68.44 | 45.01 | 745.8 GB/s | 1.52x | PASSED |
| 1x11008x4096 | LLaMA 7B SwiGLU FFN Gate/Up | 171.46 | 187.66 | 480.7 GB/s | 0.91x | PASSED |
| 16x4096x4096 | Batch 16 Prompt Evaluation | 55.07 | 50.25 | 672.9 GB/s | 1.10x | PASSED |
| 32x4096x4096 | Batch 32 Prompt Evaluation | 24.04 | 57.26 | 595.1 GB/s | 0.42x | PASSED |

## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)
| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |
|---|:---:|:---:|:---:|
| 512x512x512 | 10.19 | 14.36 | 0.71x |
| 1024x1024x1024 | 34.95 | 30.11 | 1.16x |
| 2048x2048x2048 | 228.02 | 157.68 | 1.45x |
| 4096x4096x4096 | 1839.79 | 1245.05 | 1.48x |