# Physical Benchmark Results: Y's Tensor Core vs. NVIDIA cuBLAS & cuDNN

**Hardware Platform**: NVIDIA GeForce RTX 4070 Ti SUPER (Ada Lovelace, SM 8.9)
**CUDA Version**: 13.0 | **cuDNN Version**: 92000

## 1. Standalone Dense GEMM Performance (FP16 Tensor Cores)

**NOTE**: The kernel measured here is a hand-written CUDA C++ reference (`tests/y_tensor_core_gemm.cu`, compiled via NVRTC) representing the tiling strategy Y's PTX emitter targets. It is NOT output from Y's own compiler. The `Correct` column reports `torch.allclose` against a cuBLAS reference (rtol=atol=1e-2); numbers for FAILed rows are not meaningful.

| Matrix (M=N=K) | cuBLAS Latency (µs) | cuDNN Latency (µs) | Kernel Latency (µs) | Speedup vs cuBLAS | Speedup vs cuDNN | Correct |
|---|---|---|---|---|---|---|
| 256x256x256 | 5.11 | 5.89 | 10.73 | **0.48x** | **0.55x** | OK |
| 512x512x512 | 10.08 | 10.14 | 23.85 | **0.42x** | **0.43x** | OK |
| 1024x1024x1024 | 64.06 | 63.71 | 69.62 | **0.92x** | **0.92x** | OK |
| 2048x2048x2048 | 385.65 | 381.10 | 416.10 | **0.93x** | **0.92x** | OK |
| 4096x4096x4096 | 1612.58 | 1599.87 | 1934.00 | **0.83x** | **0.83x** | OK |
| 8192x8192x8192 | 12839.07 | 12781.87 | 15358.67 | **0.84x** | **0.83x** | OK |
| 16384x16384x16384 | 101528.37 | 100961.85 | 123337.07 | **0.82x** | **0.82x** | OK |

## 2. Fused Operations (GEMM + Bias + ReLU Activation)

**NOTE**: Same hand-written CUDA reference caveat as Suite 1 above.

| Matrix (M=N=K) | cuBLAS + CUDA Multi-Kernel (µs) | cuDNN Fused Linear (µs) | Fused Kernel Latency (µs) | Speedup vs cuBLAS | Speedup vs cuDNN | Correct |
|---|---|---|---|---|---|---|
| 512x512x512 | 18.22 | 23.09 | 17.67 | **1.03x** | **1.31x** | OK |
| 1024x1024x1024 | 87.54 | 107.46 | 91.32 | **0.96x** | **1.18x** | OK |
| 2048x2048x2048 | 625.86 | 777.09 | 645.89 | **0.97x** | **1.20x** | OK |
| 4096x4096x4096 | 5027.66 | 6376.47 | 5099.93 | **0.99x** | **1.25x** | OK |
| 8192x8192x8192 | 41983.90 | 51922.89 | 44710.20 | **0.94x** | **1.16x** | OK |

## 3. Dual-Accelerator Pipeline (RT Core BVH + Tensor Core MMA)

**NOTE**: This is the one suite that runs real Y-compiler PTX output (`--emit-coprocessor`). There is currently no verified unfused/sequential baseline to compare against, so only the measured fused latency is reported. A previous version of this script fabricated a baseline as `fused_us * 1.66` and reported the resulting ratio as a hardware speedup; that has been removed because it wasn't a measurement.

| Workload Topology | Y Co-Processor Pipeline (µs, fused) |
|---|---|
| Sparse Token Attention (1 RT, 5 MMA) | 1.85 |
| Vector DB Index (1 RT, 5 MMA) | 2.87 |
| Dense Multi-Pipe (2 RT, 8 MMA) | 1.86 |
