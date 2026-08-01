# Real Y-Compiler Fused GEMM+Bias+ReLU vs cuDNN Fused Linear

Every number below is measured from `target/release/Y tests/gemm_f16_bias_relu_<N>.ysu --emit-ptx`'s actual output (`ptx_emitter::emit_gemm_bias_relu_epilogue`, dispatched via the 4-param kernel-level `@tile(M, N, K)` shape) - not the hand-written CUDA C++ reference benchmark_y_vs_cudnn_cublas.py Suite 2 measures.

Timing is the median of 5 independent process launches per size (range shown in brackets); a row is marked inconclusive rather than given a speedup number if the Y and cuDNN ranges overlap.

Two cuDNN columns, deliberately: **FP32** matches benchmark_y_vs_cudnn_cublas.py Suite 2's exact convention (FP32 `torch.nn.Linear`, the same baseline that file's 1.16x-1.31x number came from) - but FP32 cuBLAS/cuDNN gemm is intrinsically much slower per-FLOP than FP16 tensor-core gemm regardless of fusion quality, so a large margin there conflates "the fused epilogue is good" with "FP16 beats FP32". **FP16** feeds `torch.nn.Linear` the same FP16 operands the Y kernel itself consumes - the apples-to-apples precision-matched comparison.

| Matrix (M=N=K) | CTA Tile | cuDNN FP32 us (median [range]) | cuDNN FP16 us (median [range]) | Y Fused us (median [range]) | Y vs cuDNN FP32 | Y vs cuDNN FP16 | Correct |
|---|---|---|---|---|---|---|---|
| 512x512x512 | 64x64x64 | 45.31 [41.12, 96.10] | 14.66 [14.04, 34.20] | 33.15 [23.38, 70.82] | inconclusive | inconclusive | OK |
| 1024x1024x1024 | 128x128x64 | 127.04 [117.20, 266.54] | 39.90 [36.13, 87.89] | 65.42 [50.23, 147.15] | inconclusive | inconclusive | OK |
| 2048x2048x2048 | 128x256x64 | 838.86 [823.85, 847.60] | 264.56 [260.03, 268.34] | 363.24 [360.58, 368.58] | **2.31x** | **0.73x** | OK |
| 4096x4096x4096 | 128x256x64 | 6818.55 [6794.40, 6878.88] | 2044.35 [2026.54, 2097.42] | 3200.39 [3178.51, 3260.89] | **2.13x** | **0.64x** | OK |
| 8192x8192x8192 | 128x256x64 | 53894.50 [51918.44, 54420.79] | 16049.10 [15413.40, 16123.80] | 27487.30 [26627.65, 27770.01] | **1.96x** | **0.58x** | OK |
