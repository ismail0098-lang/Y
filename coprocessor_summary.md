Y Co-Processor vs. NVIDIA CUDA: Dual-Accelerator Workload & Cycle Analysis

Target Hardware: NVIDIA Ada Lovelace Architecture (RTX 4070 Ti SUPER, SM 8.9)


1. Executive Summary

On modern NVIDIA GPU architectures (Ampere, Ada Lovelace, Blackwell), workloads that combine RT Cores (BVH spatial routing, ray tracing, nearest-neighbor queries) and Tensor Cores (matrix multiplication, MLP projections, low-precision GEMM) are hard to schedule well by hand. The two accelerators are:


Asymmetric in timing — RT Core queries are asynchronous and non-deterministic (latency depends on BVH depth, spatial clustering, coherence), while Tensor Core ops are synchronous, lock-step warp instructions.
Mismatched in precision — RT Cores output high-precision FP32 values; Tensor Cores need packed low-precision fragments (FP16/BF16/FP8).
Costly to hand off between — passing RT output to Tensor input requires staging through Shared Memory, which introduces bank conflicts and manual bar.sync fences.


The Y compiler's Hardware-Sentient Scheduler handles all of this at compile time: it tracks dependency paths, allocates register pools, maps shared-memory tiles, swizzles addresses to eliminate bank conflicts, and injects vectorized quantization (cvt.rn.f16x2.f32) — all while overlapping RT Core latency with independent scalar math.

All results below are physically measured on an RTX 4070 Ti SUPER, not simulated.


2. Workloads

Workload A — RT-Routed Sparse Attention

Use RT Core BVH-accelerated K-Nearest Neighbor search (128D query space, k=8) to route token keys/values for sparse self-attention, then run Tensor Core MMA to project routed vectors.

Y (tests/coprocessor_attention.ysu) — 19 lines

rust@unsafe
fn main() {
    // 1. RT Core: Hardware BVH-accelerated K-Nearest Neighbor (128D query, k=8)
    let nns_res: I32 = rt_nearest_neighbor(128, 8);
    // 2. Tensor Core: MMA calculation (consumes routing output)
    // Automated sync barriers, swizzling, and FP32 -> FP16 packing are injected here
    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
    let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(nns_res);

    acc = mma_sync(frag_A, frag_B, frag_C);
}

CUDA C++ / OptiX equivalent — 160+ lines, requiring:


Manual OptixRayQuery traversal setup and loop
Manual shared-memory staging of RT output (rt_scratch)
Manual bar.sync fences before/after quantization
Manual FP32→FP16 packing via inline PTX (cvt.rn.f16x2.f32)
Manual wmma:: fragment loads, fill, and MMA calls



Workload B — Database Index Geometric Map (Vector Search / ANN)

Use RT Core BVH-accelerated Fixed-Radius Nearest Neighbor search to accelerate vector index lookups: 256-dimensional cluster embeddings are mapped to 3D/4D leaf spheres and searched for k=16 neighbors, then Tensor Cores compute distances/projections on the results.

Y (tests/coprocessor_db_index.ysu) — 12 lines

rust@unsafe
fn main() {
    // 1. RT Core: Hardware BVH-accelerated Fixed-Radius Nearest Neighbor Search
    // Map 256-dimensional cluster embeddings down to 3D/4D leaf spheres, search k=16 neighbors.
    let nns_res: I32 = rt_nearest_neighbor(256, 16);

    // 2. Tensor Core: Process nearest neighbor representations / compute distances/projections.
    // Compiler automatically swizzles and quantizes FP32 search outputs to FP16 matrix fragments.
    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
    let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(nns_res);

    acc = mma_sync(frag_A, frag_B, frag_C);
}

CUDA C++ equivalent — divergent BVH traversal, ~65 lines, notably including:


A stack-depth search pattern that varies per thread (tid % 4), simulating real branch divergence in manual index traversal — the exact pattern RT core hardware is built to avoid
Manual divergent distance-check branching (if (diff * diff < 4.0f))
Manual FP32→FP16 quantization loop, prone to shared-memory bank conflicts
Manual wmma:: fragment loads with standard (non-swizzled) striding



Note on index quality: index construction and recall@k tradeoffs are workload-specific. This benchmark demonstrates traversal speedup via hardware BVH mapping — not index quality or search accuracy. Teams evaluating this for production ANN search should validate recall on their own embedding distributions.




3. Compiler Scheduling Statistics (RTX 4070 Ti SUPER)

Compiled via ./target/release/Y <file> --emit-coprocessor:

Kernel FileRT NodesTensor NodesSync BarriersParallel CyclesOverlap Savingscoprocessor_combined.ysu151235133coprocessor_test.ysu30053016coprocessor_attention.ysu151215133coprocessor_large.ysu171287145coprocessor_db_index.ysu151215133


Note: the static scheduling estimates for coprocessor_attention.ysu and coprocessor_db_index.ysu are identical (215 parallel cycles, 133 overlap savings) because both kernels share the same IR node topology — 1 RT node, 5 Tensor nodes, 1 quantization sync barrier. The scheduler's cost model operates on instruction-graph shape, not on the underlying data being searched, so topology-identical kernels produce identical static schedules even though their physical latencies differ substantially (2.38 µs vs. 5.91 µs) due to the actual RT traversal cost of a 256D/k=16 search vs. a 128D/k=8 search.




4. Physical GPU Execution Results (RTX 4070 Ti SUPER)

Co-Processor Benchmark vs. Native CUDA C++

Benchmark FileNaive CUDA C++ LatencyY Co-Processor LatencyMeasured SpeedupTime Reductioncoprocessor_attention.ysu4.2175 µs2.3818 µs1.77x43.5%coprocessor_large.ysu2.4501 µs1.8515 µs1.32x24.4%coprocessor_db_index.ysu10.6026 µs5.9137 µs1.79x44.2%

(Averaged over 10,000 iterations per benchmark.)

Host Handoff Benchmark vs. PyTorch & Triton

(Separate benchmark — general kernel launch overhead, not RT/Tensor co-scheduling.)

Runtime EnvironmentAverage LatencyY SpeedupPyTorch Eager Mode2,496.06 µs1,253.89xPyTorch Compiled (Triton)13.31 µs6.69xY Native PTX Kernel1.99 µs—


5. Key Performance Gains & Explanations


Deduplicated sync barriers. The optimizer eliminates redundant synchronization and quantization passes across cross-pipeline edges from the same producer, cutting barrier count from 3 to 1 in the attention pipeline and roughly halving Y's own latency (4.74 µs → 2.38 µs).
Overlapping traversal latency. The scheduler interleaves register initialization, memory loads, and preprocessing while the RT Core processes traversal requests, saving 16–145 cycles depending on kernel topology.
Elimination of shared-memory bank conflicts. The bank-conflict optimizer computes conflict-free swizzled layouts for Tensor Core ldmatrix reads automatically. Manual CUDA requires hand-derived striding/swizzle math.
ptxas unrolling optimization. Index-based offset calculations (rather than global pointer striding) eliminate loop-carried dependencies, letting ptxas unroll maximally.
PTX state-space address safety. Shared-memory buffers are loaded via state-space-specific offsets (mov.u64) rather than generic pointer conversion (cvta.shared), avoiding a class of driver-level JIT compilation failures.



6. Why This Matters

Divergent, branch-heavy BVH traversal (Workload B) is a standard pattern in production vector search and database indexing — and the exact case RT Core hardware is designed to accelerate but almost no general-purpose toolchain exposes for non-graphics use. Y's scheduler automates the mechanics (sync, swizzling, quantization) that make hand-written CUDA implementations of this pattern slow and error-prone to get right.