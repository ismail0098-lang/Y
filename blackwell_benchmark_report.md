# Blackwell / Next-Gen GPU Benchmark Report
- **GPU Hardware**: `NVIDIA GeForce RTX 4070 Ti SUPER`
- **Compute Capability**: `sm_89`
- **VRAM**: `16.71 GB`
- **PyTorch / CUDA**: `2.12.0+cu130` / `13.0`

## 1. Standalone Dense GEMM Benchmarks (M=N=K)
| Matrix (M=N=K) | cuBLAS (us) | Y Compiler (us) | Y TFLOPS | Speedup vs cuBLAS | Parity |
|---|:---:|:---:|:---:|:---:|:---:|
| 512x512x512 | 15.44 | 77.64 | 3.46 | 0.20x | PASSED |
| 1024x1024x1024 | 81.47 | 195.85 | 10.96 | 0.42x | PASSED |
| 2048x2048x2048 | 1629.10 | 1369.05 | 12.55 | 1.19x | PASSED |
| 4096x4096x4096 | 1656.49 | 2970.42 | 46.27 | 0.56x | PASSED |
| 8192x8192x8192 | 13134.85 | 22661.06 | 48.52 | 0.58x | PASSED |
| 16384x16384x16384 | 103114.75 | 213626.37 | 41.18 | 0.48x | WARN |

## 2. Real-World LLM Inference & Prompt Decoding Shapes
| Shape (M x N x K) | Workload Description | cuBLAS (us) | Y Split-K (us) | Memory Bandwidth | Speedup | Parity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1x4096x4096 | LLaMA 7B Single-Token Decode | 26.91 | 44.94 | 747.0 GB/s | 0.60x | PASSED |
| 1x11008x4096 | LLaMA 7B SwiGLU FFN Gate/Up | 180.24 | 193.24 | 466.8 GB/s | 0.93x | PASSED |
| 16x4096x4096 | Batch 16 Prompt Evaluation | 50.71 | 49.58 | 682.0 GB/s | 1.02x | PASSED |
| 32x4096x4096 | Batch 32 Prompt Evaluation | 24.15 | 57.34 | 594.3 GB/s | 0.42x | PASSED |

## 3. Fused Neural Network Layers (Y Compiler vs PyTorch Fused GEMM+Bias+ReLU)
| Matrix (M=N=K) | PyTorch Fused (us) | Y Compiler Fused (us) | Speedup vs PyTorch |
|---|:---:|:---:|:---:|
| 512x512x512 | 10.28 | 14.38 | 0.72x |
| 1024x1024x1024 | 34.98 | 29.37 | 1.19x |
| 2048x2048x2048 | 235.38 | 168.76 | 1.39x |
| 4096x4096x4096 | 1784.15 | 1218.80 | 1.46x |