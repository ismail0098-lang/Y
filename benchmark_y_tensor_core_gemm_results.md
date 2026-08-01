# Real Y-Compiler Tensor Core GEMM vs cuBLAS

Every number below is measured from `target/release/Y tests/gemm_f16_<N>.ysu --emit-ptx`'s actual output (`ptx_emitter::emit_tensor_core_gemm_kernel`, dispatched via kernel-level `@tile(M, N, K)`, tile/warp/stage selection from `Autotuner::autotune`) - not the hand-written CUDA C++ reference benchmark_y_vs_cudnn_cublas.py's Suite 1 measures.

Timing is the median of 5 independent process launches per size (range shown in brackets); a row is marked inconclusive rather than given a speedup number if the Y and cuBLAS ranges overlap - see the module docstring in tests/benchmark_y_tensor_core_gemm.py for why a single-sample number isn't trustworthy here.

| Matrix (M=N=K) | CTA Tile | cuBLAS us (median [range]) | Y us (median [range]) | Y vs cuBLAS | Correct |
|---|---|---|---|---|---|
| 256x256x256 | 64x64x32 | 4.79 [4.69, 6.81] | 31.32 [13.26, 34.50] | **0.15x** | OK |
| 512x512x512 | 64x64x64 | 10.71 [8.39, 32.93] | 46.36 [14.77, 54.37] | inconclusive | OK |
| 1024x1024x1024 | 128x128x32 | 73.13 [49.03, 115.30] | 131.30 [77.18, 285.20] | inconclusive | OK |
| 2048x2048x2048 | 128x256x32 | 342.25 [336.98, 400.27] | 504.06 [461.48, 669.71] | **0.68x** | OK |
| 4096x4096x4096 | 128x256x32 | 1631.86 [1621.40, 1641.94] | 2202.44 [2190.97, 3453.04] | **0.74x** | OK |
| 8192x8192x8192 | 128x256x32 | 12896.97 [12849.51, 13192.65] | 17433.43 [17323.36, 17577.57] | **0.74x** | OK |
| 16384x16384x16384 | 128x256x32 | 101723.15 [101166.69, 102278.55] | 136432.65 [135547.73, 140153.45] | **0.75x** | OK |
