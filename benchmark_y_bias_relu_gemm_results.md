# Real Y-Compiler Fused GEMM+Bias+ReLU vs cuDNN Fused Linear

Every number below is measured from `target/release/Y tests/gemm_f16_bias_relu_<N>.ysu --emit-ptx`'s actual output (`ptx_emitter::emit_gemm_bias_relu_epilogue`, dispatched via the 4-param kernel-level `@tile(M, N, K)` shape) - not the hand-written CUDA C++ reference benchmark_y_vs_cudnn_cublas.py Suite 2 measures.

Timing is the median of 5 independent process launches per size (range shown in brackets); a row is marked inconclusive rather than given a speedup number if the Y and cuDNN ranges overlap.

Two cuDNN columns, deliberately: **FP32** matches benchmark_y_vs_cudnn_cublas.py Suite 2's exact convention (FP32 `torch.nn.Linear`, the same baseline that file's 1.16x-1.31x number came from) - but FP32 cuBLAS/cuDNN gemm is intrinsically much slower per-FLOP than FP16 tensor-core gemm regardless of fusion quality, so a large margin there conflates "the fused epilogue is good" with "FP16 beats FP32". **FP16** feeds `torch.nn.Linear` the same FP16 operands the Y kernel itself consumes - the apples-to-apples precision-matched comparison.

| Matrix (M=N=K) | CTA Tile | cuDNN FP32 us (median [range]) | cuDNN FP16 us (median [range]) | Y Fused us (median [range]) | Y vs cuDNN FP32 | Y vs cuDNN FP16 | Correct |
|---|---|---|---|---|---|---|---|
| 512x512x512 | 64x64x64 | 22.41 [21.28, 23.10] | 8.42 [8.35, 8.96] | 7.84 [7.65, 8.17] | **2.86x** | **1.07x** | OK |
| 1024x1024x1024 | 64x128x32 | 101.29 [100.47, 102.84] | 33.38 [32.56, 33.63] | 32.36 [32.09, 32.63] | **3.13x** | inconclusive | OK |
| 2048x2048x2048 | 128x128x32 | 730.85 [720.76, 738.92] | 246.44 [245.77, 251.68] | 249.82 [245.16, 252.64] | **2.93x** | inconclusive | OK |
| 4096x4096x4096 | 128x128x32 | 6006.58 [5953.95, 6129.13] | 1984.52 [1945.70, 1991.36] | 1965.67 [1906.69, 1984.12] | **3.06x** | inconclusive | OK |
| 8192x8192x8192 | 128x128x32 | 47518.38 [46991.32, 48265.98] | 15188.42 [14859.01, 15404.54] | 14082.05 [13989.63, 14791.68] | **3.37x** | **1.08x** | OK |
