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
* **Compilation Resources (Constraint Generation)**:
  * **Y-lang**: **`1.67 seconds`** | Peak Memory: **`1.07 GB`** (RSS) (155.4x speedup)
  * **Noir**: **`11.36 seconds`** | Peak Memory: **`1.25 GB`** (RSS) (22.8x speedup)
  * **Leo**: **`41.52 seconds`** | Peak Memory: **`10.81 GB`** (RSS) (6.2x speedup)
  * **Circom**: **`259.25 seconds`** | Peak Memory: **`2.39 GB`** (RSS)

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
* **Compilation Resources (Constraint Generation)**:
  * **Noir**: **`2.31 seconds`** | Peak Memory: **`393.74 MB`** (RSS) (6.3x speedup)
  * **Y-lang**: **`3.66 seconds`** | Peak Memory: **`154.24 MB`** (RSS) (4.0x speedup)
  * **Leo**: **`13.83 seconds`** | Peak Memory: **`3.08 GB`** (RSS) (1.05x speedup)
  * **Circom**: **`14.51 seconds`** | Peak Memory: **`1.05 GB`** (RSS)

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
