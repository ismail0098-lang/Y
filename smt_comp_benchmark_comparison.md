# SMT-COMP Official Benchmark Performance Comparison

To evaluate the real-world impact of the BooledASS optimizations, we conducted a side-by-side performance evaluation on a representative, official SMT-COMP QF_BV benchmark:

* **File Name**: `scrambled100030.smt2`
* **Logic category**: QF_BV (Quantifier-Free Bit-Vectors)
* **Path**: `z3_benchmarks/16887742/single_query/single_query/QF_BV/scrambled100030.smt2`
* **Outcome**: `unsat`

---

## Performance Metrics Table

| Metric | Baseline Z3 (Upstream Master) | BooledASS (Optimized CPU solver) | Delta (%) | Optimization Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Solve Time (sec)** | `4.89s` | `2.70s` | **-44.8%** | Massive speedup due to smaller SAT search space. |
| **SAT Conflicts** | `180,693` | `113,159` | **-37.4%** | Significantly fewer search tree backtracks. |
| **SAT Decisions** | `246,216` | `152,749` | **-38.0%** | More directed search paths. |
| **Variables Created** | `2,872` | `2,115` | **-26.3%** | Reduced variable footprint per adder block. |
| **Clauses (n-ary)** | `190,995` | `124,506` | **-34.8%** | Substantially reduced clause representation database. |
| **Propagations (n-ary)**| `14,202,811` | `7,619,831` | **-46.3%** | Lower unit propagation workload. |
| **Restarts** | `6,783` | `3,915` | **-42.3%** | Highly stable and converging search. |

---

## Key Optimization Analysis

The massive improvements in both memory/space overhead and SAT search efficiency are driven by the two main optimizations implemented in BooledASS:

### 1. Symmetrical Joint Full-Adder Internalization
* **Location**: `src/sat/smt/bv_internalize.cpp`
* **Mechanism**: Jointly maps the sum (`OP_XOR3`) and carry (`OP_CARRY`) of adder modules directly into a minimal 10-clause cover. 
* **Impact**: Decreases variable counts by **26.3%** and clause counts by **34.8%**, since it eliminates auxiliary intermediate variables that baseline Z3's separate internalization requires.

### 2. Shared Carry-Out via ITE Gate Reuse
* **Location**: `src/ast/rewriter/bit_blaster/bit_blaster_rewriter.cpp`
* **Mechanism**: Rewrites the carry-out logic to use the ternary operator: $Cout = \text{ITE}(\text{XOR}(A, B), C, A)$. By referencing the cached $\text{XOR}(A, B)$ computed for the sum bit, we avoid duplicate gate creation.
* **Impact**: Drops the carry-out encoding cost to only 4 clauses, dramatically reducing the solver's unit propagation overhead (evidenced by the **46.3%** drop in n-ary propagations).
