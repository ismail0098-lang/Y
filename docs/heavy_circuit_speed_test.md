# ZK Compiler Speed Test: Y-lang vs. Circom

This document describes the compilation speed benchmark designed to compare Y-lang against Circom for heavy circuits containing unrolled loops of non-linear constraint multiplications.

---

## 1. Benchmark Circuits

### A. Polynomial Loop (`heavy_circuit.ysu`)
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

### B. Iterative Dot Product (`dot_product.ysu`)
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

---

## 2. Compilation Speed Optimization

Key bottlenecks were identified and optimized to support large-scale loop unrolling without memory exhaustion or quadratic slow-down:

1. **In-Place Accumulator Update**: In assignments like `sum = sum + a * b` or `a = a + 1`, we take the binding from the scope, mutate the linear combination in-place, and bind it back to the same lexical scope level. This completely eliminates $O(N)$ vector cloning at each iteration, dropping loop unrolling compilation time from $O(N^2)$ to $O(N)$ linear time.
2. **`is_simplified` State Propagation**: We introduced an `is_simplified` boolean flag on `LinearCombination`. In `add_linear`, if both operands are already simplified and their wire boundaries do not overlap (e.g. adding a new multiplication tmp wire to a running sum), the resulting linear combination remains simplified. This bypasses the $O(N)$ checks in `simplify()`, rendering additions $O(1)$.
3. **Single-Term Simplification Fast Path**: The simplification logic returns immediately for `terms.len() <= 1`, eliminating redundant `HashMap` allocations and sorting iterations.
4. **Commutative Hash Lookup**: Manually implemented order-independent hashing on `LinearCombination` to support constraint deduplication using a global hash map, dropping comparison counts from 20,000,000 to 20,000.

---

## 3. Performance Results

### **1,000,000 Constraints (Polynomial)**
- **Y-lang**: **`2.20 seconds`** (`2,195.72 ms`)
- **Circom**: `292.05 seconds` (`292,045.16 ms` / ~4.87 minutes)
- **Speedup**: **133.01x faster than Circom**

### **100,000 Constraints (Iterative Dot Product)**
- **Y-lang**: **`3.63 seconds`** (`3,634.07 ms`)
- **Circom**: `17.08 seconds` (`17,077.58 ms`)
- **Speedup**: **4.70x faster than Circom**

---

## 4. Running the Benchmarks

To run the speed test benchmarks:
```bash
# Run polynomial unroll benchmark
node run_speed_test.js

# Run dot product unroll benchmark
node run_dot_product_benchmark.js
```
