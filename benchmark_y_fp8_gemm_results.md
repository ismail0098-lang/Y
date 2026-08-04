# Y FP8 (e4m3) Tensor Core GEMM vs torch._scaled_mm

Every number below is measured from the REAL Y compiler CLI's output (`cargo run --bin Y -- tests/gemm_fp8_<N>.ysu --emit-ptx`, `ptx_emitter::emit_fp8_gemm_kernel`, multi-warp CTA tiling with two-tier size selection - 64x64x64/2x2 warps/128 threads for M<=512||N<=512, else 128x128x64/4x2 warps/256 threads - fused on-the-fly FP32->e4m3 quantization (session 5: vectorized 128-bit ld.global.v4.f32 staging), software double-buffered K-loop pipelining), loaded via `cp.RawModule(path=...)` - not a hand-written probe. Reference: `torch._scaled_mm` (`torch==2.13.0+cu130`), `out_dtype=torch.float32`. GPU-side timing via CUDA events, median of 7 alternating rounds (20 launches/round) after 15 rounds of shared alternating warmup - see this file's own docstring for why alternating (not sequential full-A-then-full-B) timing matters on this project's hardware.

Correctness metric is relative L2 error (`||Y-ref||/||ref||`), the standard way to validate reduced-precision GEMM kernels - NOT a per-element max/rtol check, which is statistically inappropriate at M=N=4096 (16M+ compared elements: order statistics alone push the observed MAX per-element deviation up even when the underlying error distribution is small and well-behaved - confirmed this session, see investigation_fp8_gemm_findings.md).

| Matrix (M=N=K) | Y us (median [range]) | torch._scaled_mm us (median [range]) | Y vs torch | Correct (rel L2) |
|---|---|---|---|---|
| 256x256x256 | 24.01 [23.92, 26.57] | 6.14 [5.89, 17.31] | **0.256x** | OK (0.018%) |
| 512x512x512 | 44.13 [44.03, 44.24] | 6.96 [6.90, 7.53] | **0.158x** | OK (0.032%) |
| 1024x1024x1024 | 123.55 [123.49, 126.05] | 25.24 [25.09, 31.28] | **0.204x** | OK (0.052%) |
| 4096x4096x4096 | 3726.49 [3725.36, 3866.33] | 821.25 [815.63, 860.36] | **0.220x** | OK (0.131%) |
