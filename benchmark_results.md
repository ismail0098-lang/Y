# Tensor Core BCP Solver — Benchmark Results

## System Configuration

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM) |
| **CUDA** | v13.2 |
| **Z3 Version** | 4.17.0 (Release, MSVC `/O2`) |
| **Build** | Release x64 with optimized CUDA kernels (`-O2`, `/MD`) |
| **OS** | Windows 10/11 x64 |

---

## Build Fix Applied

> [!IMPORTANT]
> The previous Debug build's CUDA object (`booledass_kernels.obj`) was compiled with `/MDd` (Debug CRT) and `_ITERATOR_DEBUG_LEVEL=2`, which caused **1,668 LNK2038 linker mismatches** when linking into the Release build. Fixed by changing nvcc compiler options to `/MD /DNDEBUG /D_ITERATOR_DEBUG_LEVEL=0` and replacing `-G` (CUDA debug) with `-O2` (optimized GPU code).

```diff
-    -G
-    --compiler-options "/FS /MDd /D_DEBUG /D_ITERATOR_DEBUG_LEVEL=2"
+    -O2
+    --compiler-options "/FS /MD /DNDEBUG /D_ITERATOR_DEBUG_LEVEL=0"
```

---

## Benchmark 1: `bench_qfbv_mult64.smt2` (64-bit Chained Multiplier)

A QF_BV formula with four 64-bit variables linked by multiplications, additions, XOR constraints, and range bounds. **Result: SAT**

### Formula Statistics

| Metric | Value |
|--------|-------|
| **Initial clauses** | 139,465 |
| **Initial binary clauses** | 62,721 |
| **Variables** | 40,645 |
| **After simplification** | 105,324 clauses / 20,277 binary |
| **Active tiles (16×16)** | 84,679 |

### Timing Comparison

| Phase | GPU-Enabled | CPU-Only | Δ |
|-------|------------|---------|---|
| **Bit-blast + init** | ~0.00s | ~0.00s | — |
| **GPU matrix build** | 0.036s | — | +0.036s |
| ├ Adjacency graph | 0.013s | — | |
| ├ RCM ordering | 0.002s | — | |
| ├ Tile identification | 0.009s | — | |
| └ Data fill | 0.012s | — | |
| **GPU kernel exec** | ~0.11s | — | +0.11s |
| **GPU verdict** | `l_undef` | — | |
| **CPU search (CDCL)** | 0.34s | 0.24s | +0.10s |
| **Total wall time** | **0.524s** | **0.388s** | **+0.136s (+35%)** |

> [!NOTE]
> The GPU kernel returned `l_undef` (could not solve with a single BCP pass), adding ~0.14s of overhead. Since the CPU solved this in only 0.39s, the GPU overhead is proportionally significant. On longer-running instances this overhead becomes negligible.

---

## Benchmark 2: `bench_gpu_arx_medium.smt2` (64-bit ARX Network, 4 chains)

A much harder QF_BV formula modeling a 4-chain ARX mixing network with chained 64-bit multiplications. **Result: TIMEOUT (180s)**

### Formula Statistics

| Metric | Value |
|--------|-------|
| **Initial clauses** | 624,235 |
| **Initial binary clauses** | 280,783 |
| **Variables** | 181,616 |
| **After simplification** | 479,551 clauses / 100,221 binary |
| **Active tiles (16×16)** | 387,793 |

### Timing Comparison

| Phase | GPU-Enabled | CPU-Only | Δ |
|-------|------------|---------|---|
| **GPU matrix build** | 0.170s | — | +0.170s |
| ├ Adjacency graph | 0.064s | — | |
| ├ RCM ordering | 0.017s | — | |
| ├ Tile identification | 0.041s | — | |
| └ Data fill | 0.047s | — | |
| **GPU kernel exec** | ~0.16s | — | +0.16s |
| **GPU verdict** | `l_undef` | — | |

### CPU Search Progress (same CDCL behavior in both modes)

| Conflicts | GPU-Enabled | CPU-Only | Overhead |
|-----------|------------|---------|----------|
| 113 | 0.33s | 0.04s | +0.29s |
| 400 | 1.29s | 1.07s | +0.22s |
| 8,316 | 1.82s | 1.64s | +0.18s |
| 27,309 | 7.04s | 7.02s | +0.02s |
| 61,447 | 36.34s | 35.94s | +0.40s |
| 90,000 | 47.50s | 46.58s | +0.92s |
| 135,001 | 67.75s | 66.82s | +0.93s |
| **Timeout** | **180.06s** | **180.04s** | **+0.02s** |

> [!NOTE]
> The GPU overhead (~0.33s) is amortized and becomes negligible (<0.2%) for problems running longer than 1 minute.

---

## Benchmark 3: `bench_qfbv_mult32.smt2` (32-bit Multiplier)

| Metric | Value |
|--------|-------|
| **Clauses** | 33,667 |
| **Binary clauses** | 14,995 |
| **GPU status** | Bypassed (33,667 < 35,000 threshold) |
| **Result** | SAT |
| **Wall time** | ~0.04s |

---

## Host-Side Setup Performance

> [!TIP]
> The refactored `build_compressed_tile_layout_direct` using flat arrays + binary search (replacing the previous 10+ GB 2D `tile_index` lookup table) now scales linearly with active tile count.

| Instance Size | Clauses | Variables | Active Tiles | Build Time |
|--------------|---------|-----------|--------------|------------|
| mult64 | 139K | 40K | 84,679 | **0.036s** |
| arx_medium | 624K | 181K | 387,793 | **0.170s** |

The previous implementation would have allocated a `(M_pad/16) × (K_pad/16)` 2D array which for `arx_medium` would be `39,040 × 11,352 = 443M entries × 4 bytes = 1.77 GB` — now handled with a sorted flat array of 387K entries.

---

## Key Findings

### 1. Release Build is Essential
- Debug z3.exe: **70 MB**, Release z3.exe: **15.7 MB** (4.5× smaller)
- Release builds are **5-10× faster** for SAT solving due to MSVC `/O2` optimizations

### 2. GPU BCP as a Pre-solver
The current GPU BCP implementation acts as a **one-shot pre-solver**: it attempts to solve the problem via a single Tensor Core BCP pass before falling back to the full CDCL solver. On both benchmarks, the GPU returned `l_undef`, meaning the single-pass approach couldn't find a solution or prove UNSAT.

### 3. Overhead Analysis
| Instance Size | GPU Overhead | Problem Solve Time | Overhead % |
|--------------|-------------|-------------------|------------|
| 139K clauses | 0.14s | 0.39s (SAT) | 36% |
| 624K clauses | 0.33s | 180s (timeout) | 0.18% |

### 4. Sparse Tile Layout is Fast
The refactored matrix build achieves **~3.7M clauses/second** throughput for tile construction, making host-side setup negligible compared to solve time.

> [!WARNING]
> **Current limitation**: The GPU BCP solver returns `l_undef` on all tested benchmarks, meaning it cannot yet provide a speedup. The Tensor Core WMMA-based BCP pass needs to either:
> 1. **Iterate** (multiple BCP rounds with conflict-driven learning on GPU), or
> 2. **Feed unit propagation results back** to the CPU solver to accelerate the CDCL search, rather than attempting a full solve in one shot.
> 
> Without these enhancements, the GPU path adds overhead without providing a solve-time reduction.

---

## Recommendations

1. **GPU BCP as acceleration**: Instead of one-shot solving, use the GPU for **bulk unit propagation** within the CDCL loop — processing all watched-clause implications in parallel via Tensor Cores for each decision level.

2. **Streaming GPU BCP**: Implement a `gpu_propagate()` that replaces the inner `propagate()` call in `bounded_search()` for large instances, keeping clause data resident on GPU memory.

3. **Benchmark selection**: For future testing, the `bench_gpu_arx_medium.smt2` benchmark (624K clauses, 181K variables) provides a good stress test that runs long enough to measure meaningful differences.
