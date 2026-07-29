# Y ZK Compiler Backend: Benchmark Suite

This directory contains the benchmark suite used to validate the performance, structural correctness, and Circom-equivalence of the Y-lang Zero-Knowledge (ZK) Rank-1 Constraint System (R1CS) compiler backend.

---

## Benchmark Registry

### 1. Baseline Verification (`test_circuit`)
* **Objective**: Validate the compilation of conditional control flow, loops, and basic arithmetic.
* **Concepts Tested**: Static loop unrolling, type checker safety, SSA variable reassignments, and conditional branch isolation via multiplexing.
* **Y-lang Source (`test_circuit.ysu`)**:
  ```rust
  @unsafe
  fn main(x: I32, y: I32) -> I32 {
      let product = x * y;
      let mut loop_sum = 0;
      for i in 0..5 {
          loop_sum = loop_sum + x * i;
      }
      let mut cond_val = 0;
      if x == y {
          cond_val = 100;
      } else {
          cond_val = 200;
      }
      return product + loop_sum + cond_val;
  }
  ```
* **Circom Source (`test_circuit.circom`)**:
  ```circom
  pragma circom 2.0.0;
  include "circomlib/circuits/comparators.circom";

  template TestCircuit() {
      signal input x;
      signal input y;
      signal output result;

      signal product <== x * y;
      signal loop_sum <== 10 * x;

      component eq = IsEqual();
      eq.in[0] <== x;
      eq.in[1] <== y;

      signal cond_val <== eq.out * 100 + (1 - eq.out) * 200;
      result <== product + loop_sum + cond_val;
  }
  component main {public [x, y]} = TestCircuit();
  ```
* **Constraints & Wires**:
  * **Y-lang**: **5 constraints, 8 wires** (Natively optimized linear combinations & multiplexer)
  * **Circom**: **7 constraints, 10 wires** (Using `IsEqual` comparator sub-template)

---

### 2. Large-Scale Polynomial Loop (`heavy_circuit`)
* **Objective**: Stress-test compiler performance, memory consumption, and optimization passes over a massive constraint budget ($1,000,000+$ constraints).
* **Concepts Tested**: In-place scope mutations, single-term linear combination shortcutting, and deduplication memory bounds.
* **Y-lang Source (`heavy_circuit.ysu`)**:
  ```rust
  @unsafe
  fn main(x: I32, y: I32) -> I32 {
      let mut temp = x;
      for i in 0..1000000 {
          temp = temp * y;
      }
      return temp;
  }
  ```
* **Circom Source (`heavy_circuit.circom`)**:
  ```circom
  pragma circom 2.0.0;

  template HeavyCircuit() {
      signal input x;
      signal input y;
      signal output out;

      signal temps[1000001];
      temps[0] <== x;
      for (var i = 0; i < 1000000; i++) {
          temps[i+1] <== temps[i] * y;
      }
      out <== temps[1000000];
  }
  component main {public [x, y]} = HeavyCircuit();
  ```
* **Constraints**: **1,000,000 constraints**.
* **Compilation Resources (Constraint Generation, Release Binary)**:
  * **Y-lang**: **`1.706 seconds`** | Peak Memory: **`1.04 GB`** (RSS) (148.8x speedup)
  * **Noir**: **`13.069 seconds`** | Peak Memory: **`1.25 GB`** (RSS) (19.4x speedup)
  * **Leo**: **`41.52 seconds`** | Peak Memory: **`10.81 GB`** (RSS) (6.2x speedup)
  * **Circom**: **`253.936 seconds`** | Peak Memory: **`2.39 GB`** (RSS)

---

### 3. Iterative Dot Product (`dot_product`)
* **Objective**: Evaluate loop unrolling overhead when mutating multiple registers per iteration.
* **Concepts Tested**: Linear combination addition optimizations (`is_simplified` state propagation).
* **Y-lang Source (`dot_product.ysu`)**:
  ```rust
  @unsafe
  fn main(x: I32, y: I32) -> I32 {
      let mut sum = 0;
      let mut a = x;
      let mut b = y;
      for i in 0..100000 {
          a = a + 1;
          b = b + 1;
          sum = sum + a * b;
      }
      return sum;
  }
  ```
* **Circom Source (`dot_product.circom`)**:
  ```circom
  pragma circom 2.0.0;

  template DotProduct() {
      signal input x;
      signal input y;
      signal output out;

      var a = x;
      var b = y;
      signal products[100000];
      var running_sum = 0;

      for (var i = 0; i < 100000; i++) {
          a = a + 1;
          b = b + 1;
          products[i] <== a * b;
          running_sum = running_sum + products[i];
      }
      out <== running_sum;
  }
  component main {public [x, y]} = DotProduct();
  ```
* **Constraints**: **100,000 constraints**.
* **Compilation Resources (Constraint Generation, Release Binary)**:
  * **Y-lang**: **`0.285 seconds`** | Peak Memory: **`152.8 MB`** (RSS) (53.6x speedup vs Circom)
  * **Noir**: **`2.261 seconds`** | Peak Memory: **`393.74 MB`** (RSS) (6.3x speedup)
  * **Leo**: **`13.83 seconds`** | Peak Memory: **`3.08 GB`** (RSS) (1.05x speedup)
  * **Circom**: **`15.28 seconds`** | Peak Memory: **`1.05 GB`** (RSS)

---

### 4. Poseidon Hash Function (`poseidon_benchmark`)
* **Objective**: Verify compiler compatibility with standard library sub-templates, parameter evaluation, and multi-stage cryptographic loops.
* **Concepts Tested**: Sub-template initialization, array constraints mapping, functions in circom templates, and local `circomlib` inclusion paths.
* **Circom Source (`poseidon_benchmark.circom`)**:
  ```circom
  pragma circom 2.0.0;
  include "circomlib/circuits/poseidon.circom";

  template PoseidonBenchmark() {
      signal input x;
      signal input y;
      signal output out;

      component pos = Poseidon(2);
      pos.inputs[0] <== x;
      pos.inputs[1] <== y;
      out <== pos.out;
  }
  component main {public [x, y]} = PoseidonBenchmark();
  ```

---

### 5. 31 Million Constraints Extreme Loop (`heavy_31m`)
* **Objective**: Evaluate compiler memory usage and compilation speed when unrolling large-scale programs that typically cause out-of-memory (OOM) failures or significant swap delays.
* **Concepts Tested**: Linear combination allocations, garbage collection overhead, AST nodes pruning, and R1CS generation scaling.
* **Y-lang Source (`heavy_31m.ysu`)**:
  ```rust
  @unsafe
  fn main(x: I32, y: I32) -> I32 {
      let mut temp = x;
      for i in 0..31000000 {
          temp = temp * y;
      }
      return temp;
  }
  ```
* **Constraints**: **31,000,000 constraints**.
* **Compilation Resources (Constraint Generation)**:
  * **Y-lang**: **`105.28 seconds`** | Peak Memory: **`30.65 GB`** (RSS) — *Generates a valid 3.7 GB `.r1cs` binary file under 2 minutes.*
  * **Circom**: **OOM** (Estimated **~74 GB of RAM** and over **2.2 hours**).
  * **Leo**: **OOM** (Estimated **~335 GB of RAM**).
  * **Noir**: **OOM / Swap Latency** (Estimated **~39 GB of RAM**).

---

## Executing the Benchmarks

### 1. Compile Y-lang Circuits
Ensure the Y compiler is compiled with the `zk` feature enabled, then run:
```bash
# Compile to R1CS
cargo run --features zk --bin Y -- <circuit_name>.ysu --target=r1cs
```

### 2. Compile Circom Circuits
Pass the local `-l .` flag to resolve the `circomlib` templates directory:
```bash
# Compile with Circom
circom <circuit_name>.circom -l . --r1cs --wasm --sym
```

### 3. Verification Scripts
Validate constraint count and structure using offline Python harnesses:
```bash
# Verify test_circuit
python verify_r1cs.py test_circuit.r1cs

# Verify heavy_circuit
python verify_heavy.py heavy_circuit.r1cs
```

### 4. Automated Speed & Equivalence Test Runs
Run the JavaScript scripts to measure and compare speed metrics:
```bash
# Run 1M polynomial speed benchmark
node run_speed_test.js

# Run 100K dot product speed benchmark
node run_dot_product_benchmark.js

# Run end-to-end equivalence checks
node verify_benchmarks.js
```

---

## 6. GPU Backend Physical Hardware Benchmarks (Y vs OpenAI Triton, cuBLAS & cuDNN)

* **Target Device**: NVIDIA GeForce RTX 4070 Ti SUPER (Ada Lovelace, SM 8.9, 16 GB VRAM)
* **Full Documentation**: [docs/y_gpu_benchmark_results.md](file:///home/yumin/NVME%20files/YSU-engine-main/YSU-engine-main/src/Y_lang/docs/y_gpu_benchmark_results.md)

### Standalone FP16 GEMM & Extreme 64K Matrix Performance

| Matrix Shape ($M=N=K$) | cuBLAS ($\mu s$) | OpenAI Triton 3.7.0 ($\mu s$) | Y Compiler Autotuned ($\mu s$) | TFLOPS | Speedup vs cuBLAS | Speedup vs Triton |
|---|---|---|---|---|---|---|
| **512x512** | 71.67 | 12.12 | **8.18** | 32.0 TFLOPS | **8.76x** | **1.48x** |
| **1024x1024** | 87.56 | 31.15 | **30.28** | 71.0 TFLOPS | **2.89x** | **1.03x** |
| **2048x2048** | 222.87 | 210.20 | **149.53** | 114.9 TFLOPS | **1.49x** | **1.41x** |
| **4096x4096** | 1627.84 | 1615.50 | **1242.87** | 110.6 TFLOPS | **1.31x** | **1.30x** |
| **8192x8192** | 12788.02 | 12741.94 | **9594.65** | 114.6 TFLOPS | **1.33x** | **1.33x** |
| **16384x16384** | 101823.28 | 105751.15 | **80061.03** | 110.0 TFLOPS | **1.27x** | **1.32x** |
| **32768x32768** | 837205.02 | 873836.55 | **644496.52** | 109.1 TFLOPS | **1.30x** | **1.36x** |
| **65536x65536 (64K)** | **6.885 s** | **6.848 s** | **5.056 s** | **111.35 TFLOPS** | **1.36x** | **1.35x** |

### Fused Operations (GEMM + Bias + ReLU Activation)

| Matrix Shape ($M=N=K$) | cuBLAS Multi-Kernel ($\mu s$) | OpenAI Triton Fused ($\mu s$) | Y Fused Tensor Core ($\mu s$) | Speedup vs cuBLAS | Speedup vs Triton |
|---|---|---|---|---|---|
| **512x512** | 21.57 | 12.22 | **7.05** | **3.06x** | **1.73x** |
| **1024x1024** | 102.69 | 30.01 | **30.00** | **3.42x** | **1.00x** |
| **2048x2048** | 642.41 | 216.96 | **171.19** | **3.75x** | **1.27x** |
| **4096x4096** | 5052.42 | 1768.22 | **1278.44** | **3.95x** | **1.38x** |
| **8192x8192** | 42978.24 | 14355.00 | **9898.04** | **4.34x** | **1.45x** |
| **16384x16384** | 113447.52 | 114697.62 | **77665.64** | **1.46x** | **1.48x** |
| **32768x32768** | 929587.71 | 939088.38 | **764916.20** | **1.22x** | **1.23x** |

---

## 7. NVIDIA RTX A4500 (Ampere SM 8.6) Live Cloud Benchmarks

* **Target Device**: NVIDIA RTX A4500 (Ampere SM 8.6, 20 GB VRAM, CUDA 12.4, PyTorch 2.4.1)
* **Environment**: Live Cloud Container Instance

### Suite 1: Standalone FP16 GEMM

| Matrix Shape ($M=N=K$) | cuBLAS ($\mu s$) | cuDNN ($\mu s$) | Y Tensor Core ($\mu s$) | Speedup vs cuBLAS | Speedup vs cuDNN |
|---|---|---|---|---|---|
| **256x256** | 16.97 | 18.77 | **6.57** | **2.58x** | **2.85x** |
| **512x512** | 15.19 | 20.75 | **14.26** | **1.07x** | **1.46x** |
| **1024x1024** | 36.88 | 43.13 | **48.01** | **0.77x** | **0.90x** |

### Suite 2: Fused Deep Learning Operations (GEMM + Bias + ReLU)

| Matrix Shape ($M=N=K$) | cuBLAS + Activation Kernel ($\mu s$) | cuDNN Fused Graph ($\mu s$) | Y Fused Tensor Core ($\mu s$) | Speedup vs cuBLAS | Speedup vs cuDNN Fused |
|---|---|---|---|---|---|
| **2048x2048** | 1518.91 | 1685.25 | **1092.95** | **1.39x** | **1.54x** |
| **4096x4096** | 9483.99 | 12210.75 | **8119.94** | **1.17x** | **1.50x** |
| **8192x8192** | 75040.95 | 97139.67 | **65208.83** | **1.15x** | **1.49x** |

### Suite 3: Dual-Accelerator Pipeline (RT Core + Tensor Core Overlap)

| Workload Topology | Sequential Baseline ($\mu s$) | Y Co-Processor Pipeline ($\mu s$) | Hardware Speedup | Latency Reduction |
|---|---|---|---|---|
| **Sparse Token Attention** (1 RT, 5 MMA) | 7.20 | **4.34** | **1.66x** | **39.8% Time Saved** |
| **Vector DB Index** (1 RT, 5 MMA) | 7.26 | **4.37** | **1.66x** | **39.8% Time Saved** |
| **Dense Multi-Pipe** (2 RT, 8 MMA) | 7.22 | **4.35** | **1.66x** | **39.8% Time Saved** |

