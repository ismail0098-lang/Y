# Where ZK compilation time actually goes

Measured on a chain of Poseidon hashes (`h = poseidon_hash(h, x)` repeated N
times), because the polynomial benchmark cannot see any of this: its linear
combinations are one or two terms wide and its coefficients are 0, 1 and 2, so
it barely multiplies. Y was 154x faster than circom on that circuit and
**13.5x slower on Poseidon** — this file is about the second number, and about
what it took to close it.

> **Status: fixed.** `Fr` is now a `Copy` `[u64; 4]` in Montgomery form
> (`src/zk_field.rs`), and `optimize_circuit` no longer dominates what is left.
> Emit for the 1000-hash chain went **9.35 s -> 0.91 s**, allocations
> **356 M -> 5.6 M**, dense-circuit peak RSS **4.84 -> 1.42 KB/constraint**, and
> Y is now **2.6x faster than circom** on the same circuit rather than 13.5x
> slower. The sections below are the original
> diagnosis, kept because the reasoning is what generalises; the results are at
> the bottom.

Reproduce with `Y_ZK_TIMING=1`, and allocation counts with
`cargo build --release --features zk,alloc-stats`.

## The ZK path is single-threaded and does not touch the GEMM backend

Worth stating because the CPU GEMM work was originally framed as being "for ZK
and SMT":

- No threading primitives in any `zk_*.rs` — no `thread::`, `spawn`, `rayon`,
  `par_iter`, `Mutex`, or atomics beyond the counters added for this profile.
  It cannot be hiding in a dependency either: `[dependencies]` in `Cargo.toml`
  is **empty**. arkworks is `[dev-dependencies]`, a test oracle only.
- `cpu_gemm` is referenced from exactly one place, `llvm_emitter.rs`.
  `--target=r1cs` routes to `ZkEmitter`, which lowers to BN254 field
  constraints. There is no path from a ZK compile to the GEMM kernel, and the
  kernel is f32-only while ZK is 254-bit modular arithmetic.
- `@safe` invariants shell z3 out as a subprocess; the only thread-ish code in
  `type_checker.rs` is `Command::spawn`.

**No amount of GEMM work speeds up ZK.**

## Phase breakdown

4000 hashes, ~964,000 constraints, 38 s total:

| phase | time | share |
|---|---|---|
| `emit_program` | 34.9 s | **92%** |
| `write_r1cs_binary` (incl. `.sym`) | 3.2 s | 8% |
| `write_r1cs_txt` | 0.000 s | 0% |

So the writers are not the problem this time — unlike the earlier
`to_decimal_string` regression, which made *formatting* the largest phase.

## It is linear, so this is a constant factor and not an algorithmic bug

| hashes | emit | per hash |
|---|---|---|
| 250 | 2.14 s | 8.54 ms |
| 500 | 4.35 s | 8.70 ms |
| 1000 | 8.69 s | 8.69 ms |
| 2000 | 17.47 s | 8.73 ms |

**8.7 ms per Poseidon hash**, flat. At 241 constraints per hash that is
**36 µs per constraint**.

## It is NOT field-arithmetic bound, which is the whole point of this file

The natural conclusion from "a Poseidon circuit is pure field arithmetic" is
that the fix is a faster field multiply. Measured, that is wrong.

At 1000 hashes / 241,000 constraints the emitter performs **4.12 M `Fr::mul`
and 7.81 M `Fr::add`**. Microbenchmarked on full-width 254-bit operands
(`tests/zk_field_microbench.rs`):

| | cost | allocations |
|---|---|---|
| `Fr::mul` | 256–300 ns | **22.2** |
| `Fr::add` | 76–86 ns | **6.4** |
| `Fr::clone` | — | 1.4 |

All that arithmetic is `4.12M x 280ns + 7.81M x 80ns` = **~1.8 s of the 8.6 s
emit, about 20%.** A field multiply made infinitely fast would leave 6.8 s.

> Benchmark the operands, not just the operation. `BigUint` trims leading zero
> limbs, so `Fr(3) * Fr(5)` multiplies one limb by one limb and skips the
> reduction. A microbenchmark built from small constants measures a function
> the circuit never calls; `spread()` in the test exists for that reason and
> asserts its operands are >= 7 limbs.

## What it actually is: 356 million allocations

Same circuit, `--features alloc-stats`:

```
emit allocations    356,381,447 allocs,   12.69 GB
```

**1,479 allocations per constraint.** At ~24 ns for a malloc/free pair that is
~8.5 s — essentially the entire emit. ZK compilation is **allocator-bound**.

The root cause is one line:

```rust
pub struct BigUint {
    pub digits: Vec<u32>,      // heap, base 2^32
}
```

A BN254 field element is 254 bits — four `u64`s — and it is stored as a
heap-allocated `Vec<u32>`. Consequently every field element that is created,
cloned, returned by value, or moved into a `HashMap` is a heap allocation, and
`Fr::mul` alone performs 22 of them (`BigUint::mul` allocates its product
vector, then `barrett_reduce` allocates for the `mu` multiply, the limb shifts,
the `p` multiply and the correction loop).

Splitting the 356 M: the field primitives directly explain ~141 M (40%), and the
linear-combination layer the other 60% — but the LC layer's allocations are
themselves mostly the field ops it invokes. One `scale + add_linear + simplify`
on a width-28 combination costs **1,643 allocations**, of which ~1,400 are the
28 muls and 28 adds inside it. `LinearCombination::simplify` additionally builds
a `HashMap<usize, Fr>` per call and sorts it.

Width 28 is not arbitrary: the emitted `.r1cs` is **986 MB for 964 k
constraints**, ~1 KB each, i.e. ~28 terms per constraint. The polynomial
benchmark's are 1–2.

## The fix, and what it measured

**Represent `Fr` as `[u64; 4]` on the stack with Montgomery multiplication.**
Done, in `src/zk_field.rs`. Two changes, in this order:

1. `Fr` became a `Copy` `[u64; 4]` in Montgomery form, with SOS Montgomery
   multiplication. `BigUint` stays for parsing, one-off setup and the writers.
2. circomlib's 144 Poseidon constants stopped being re-parsed from hex on every
   `poseidon_hash` call. `BigUint::from_hex_str` is a mul-and-add per digit, so
   one hash spent ~37,000 allocations rebuilding compile-time constants. They
   are cached, keyed on the active modulus.

The second one was not predicted, and is worth stating plainly: after the
representation change, emit was still 2.9 s, and the largest remaining source of
allocations was not the field or the linear-combination layer at all. **Measure
again after the fix, not just before it.**

### Primitives (`tests/zk_field_microbench.rs`)

| | before | after |
|---|---|---|
| `Fr::mul` | 256-300 ns, 22.2 allocs | **17.2 ns, 0 allocs** |
| `Fr::add` | 76-86 ns, 6.4 allocs | **9.0 ns, 0 allocs** |
| `Fr` clone | 1.4 allocs | **0** (it is `Copy`) |
| `scale + add_linear + simplify`, width 28 | 1,643 allocs | **10** |

### The 1000-hash Poseidon chain, 241,000 constraints

| | before | after `Fr` | after `optimize_circuit` |
|---|---|---|---|
| `emit_program` | 9.35 s | 1.86 s | **1.19 s** (7.8x) |
| — `emit_circuit_entry` | — | 0.60 s | 0.60 s |
| — `optimize_circuit` | — | 1.22 s | **0.59 s** |
| allocations during emit | 356 M / 12.69 GB | 7.4 M / 4.77 GB | **5.6 M / 3.31 GB** |
| `write_r1cs_binary` | 0.72 s | 0.50 s | 0.50 s |
| per hash | 8.7 ms | 1.8 ms | **1.19 ms** |

Still exactly linear: 1.10 / 1.16 / 1.19 / 1.19 ms per hash at 250 / 500 / 1000 /
2000, so it remains a constant factor and not an algorithmic bug.

### Against circom, same circuit, whole process

| hashes | circom | Y | |
|---|---|---|---|
| 250 | 0.788 s | **0.427 s** | Y **1.85x** faster |
| 1000 | 3.118 s | **1.766 s** | Y **1.77x** faster |

Whole-process wall time, best of three, same chain of `Poseidon(2)` over
circomlib's own template. Y emits 241 constraints per hash against circom's 243
non-linear + 274 linear, so the circuits are comparable and Y's is slightly
smaller.

The documented **13.5x deficit on Poseidon is gone**, and without moving a single
pinned digest. The polynomial circuit's lead was not left alone either — it went
from 154x to **261x**, because that circuit's emit dropped 1.60 s → 0.94 s for
the same reasons. Both benchmarks moved; only one of them was the target.

### The prove path at 1M constraints (polynomial circuit)

| phase | before | after |
|---|---|---|
| emit | 1.77 s | 1.34 s |
| witness | 0.57 s | **0.10 s** (5.7x) |
| Groth16 setup | 2.80 s | 2.89 s |
| prove | 2.87 s | 3.10 s |
| verify | 0.002 s | 0.002 s |
| **total** | **8.0 s** | **7.5 s** |

The polynomial circuit barely multiplies, which is why its *emit* moves only
1.3x — close to the prediction this file made. Its witness solve, which does
touch dense values, moves 5.7x. arkworks is now 80% of the prove path, so what
remains is a prover Y does not have.

## Where the time is now

`emit_program` for the 1000-hash chain splits evenly:

| | after `Fr` | now |
|---|---|---|
| `emit_circuit_entry` (build the constraints) | 0.60 s | 0.60 s |
| `optimize_circuit` (the CSE / dedup pass) | 1.22 s | **0.59 s** |

`optimize_circuit` was two thirds of emit and is now half of a much smaller
number. Three things were wrong with it, none of them the algorithm:

- Its hash buckets stored **clones of `A` and `B`** — two ~1 KB `Vec`s per
  constraint, 480,000 allocations on this circuit — solely so a hash collision
  could be resolved against data already sitting in `self.constraints`. They hold
  indices now.
- It bucketed with `DefaultHasher` (SipHash), hashing ~2.2 KB of term data per
  constraint through a keyed permutation to compute a table index whose every hit
  is verified by full equality anyway. A ten-line multiply-xor mixer does the
  same job.
- The dominant cost was neither: **applying the substitution**. It ran a
  `HashMap` probe per term — ~7 M per iteration, four iterations deep, almost all
  misses — and rebuilt the whole constraint vector each time. Wire ids are dense,
  so the substitution is an array index; removal is `retain` in place.

The fixpoint itself is doing real work and was not the problem: it finds 999
replacements on each of four passes, because in `h = poseidon_hash(h, y)` the
part of every hash that depends only on `y` is genuinely identical across all
1000 iterations, and merging one layer exposes the next.

At 5.6 M allocations (23 per constraint) the allocator is no longer the
constraint anywhere. The remaining even split means there is no single next
target; `Y_ZK_TIMING=1` reports both sub-phases.

## Memory, which turned out to be the more important number

Time is not what bounds this backend. Cost per constraint is linear, so **RAM
sets a hard ceiling on circuit size**, and that ceiling is the whole basis of the
"circuits too big for circom" position.

The figure in `heavy_circuit_speed_test.md` was **0.96 GB per million
constraints**, implying ~45M on a 46 GB box. That is the *polynomial* circuit's
number. Its linear combinations are one or two terms wide; a real circuit's are
~28, and it costs proportionally more:

| | polynomial (1-2 term LCs) | Poseidon chain (~28 term LCs) |
|---|---|---|
| before | 0.77 KB/constraint | **4.84 KB/constraint** |
| after | **0.44** | **1.42** |
| 46 GB box ceiling | ~105M constraints | 9.5M -> **~32M** |

So the claim was running out of memory at roughly the circuit sizes it was about.
Four fixes, none of which changed a byte of the emitted `.r1cs`:

- `write_r1cs_binary` called `build_circuit()`, which **deep-clones every
  `Constraint`**, to hand a read-only view to the encoder. `CircuitView` borrows
  instead. The same clone was on the `--witness` path in `main.rs`.
- `encode_to_stream` **buffered the entire constraints section** in a `Vec<u8>`
  before writing a byte. The section length the format wants up front is
  arithmetic — `3*4 + terms*36` — so it streams now, with a check that what was
  written matches what was declared (a mismatch would be a structurally corrupt
  file rather than a rejected one).
- It allocated a **32-byte `Vec` per term** for each coefficient — 6.6 M of them
  on a 237 k constraint circuit, the largest remaining allocation source in the
  pipeline. `Fr::write_bytes_le` fills a reused buffer.
- **`emit_mul_lc` stored a verbatim second copy of every constraint's `A` and
  `B`** in `witness_recipes` — 1096 B/constraint against the constraints' own
  1135, nearly doubling live memory on *every* compile including those that never
  generate a witness. Its docstring's justification had expired: the
  `lc_by_output` second-chance scan in `build_witness_ir` reconstructs exactly
  that shape, and its `all(|(w, _)| *w < out)` guard holds by construction
  because the wire is freshly allocated. Verified the witness path still takes
  the fast forward pass rather than falling back to back-propagation.

That last one is the reason `Y_ZK_COMPOSITION=1` exists. A peak-RSS total says
nothing about what to fix; the per-owner breakdown made a 1:1 duplicate obvious
in one line of output.

## Where the time is now (after the memory work)

| | time |
|---|---|
| `emit_circuit_entry` | 0.57 s |
| `optimize_circuit` | 0.34 s |
| `emit_program` total | **0.91 s** |
| `write_r1cs_binary` | 0.30 s |

Against circom on the same Poseidon chain, whole process, best of three:

| hashes | circom | Y | |
|---|---|---|---|
| 250 | 0.814 s | **0.294 s** | Y **2.77x** faster |
| 1000 | 3.161 s | **1.236 s** | Y **2.56x** faster |


## Three bugs this work exposed

None is about performance. All three were found because a representation change
forces every assumption to be re-checked.

- **CSE silently made valid circuits unprovable.** `optimize_circuit` merges two
  constraints with identical `A` and `B`, rewriting the loser's wire id
  throughout the constraint system — but not in `witness_recipes`, whose
  `LinearCombination`s are captured at emit time. Every gadget recipe
  (`emit_num2bits` keeps the value it decomposes, `emit_int_div_mod` its dividend
  and divisor, the is-zero gadget its difference) therefore evaluated a wire the
  pass had just deleted, read it as zero, and produced a witness that did not
  satisfy the circuit. `let a = x * y; let b = x * y; return a == b;` — two
  provably equal values — came back **unprovable**, and so did `<`, `<=`, `>`,
  `>=`, `!=`, `&`, `|`, `^`, `/` and `%` whenever an operand happened to be a
  common subexpression. It presented as `satisfied = false`, which reads as
  "this circuit is unsatisfiable", not as a compiler bug. `remap_witness_op` now
  rewrites them, matched exhaustively with no `_ =>` arm so a new variant is a
  compile error rather than a silent omission. Pinned by
  `tests/zk_cse_gadget_wires.rs`, whose control asserts the duplicated form is
  still one constraint smaller — otherwise "stop eliminating anything" would pass
  every other test in the file.

- **`ScalarField::Vesta`'s modulus was composite.** It read
  `...941600134020817490249052636161`, a corrupted transcription of Vesta's base
  field sharing only the leading `0x40000000000000000000000000000000224698fc09`
  with the real value. An R1CS over a composite modulus is not a proof system:
  inverses stop existing and no gadget that needs one is sound. It survived
  because the old `Fr::inv` was extended Euclid, which inverts anything coprime
  to the modulus and so returned plausible values. `FieldParams::new` now runs
  Miller-Rabin over twelve bases and **refuses** a composite modulus.
- **A dropped carry in the Montgomery reduction, caught by widening a test.**
  Koc's SOS reduction step is `ADD(t[i+s], C)`, and ADD *propagates*; stopping
  after one word loses the carry for operand pairs that happen to overflow it.
  Under BN254 every test passed. Under Pallas, `inv(2)` and `inv(3)` were wrong
  and `inv(5)` was right — a data-dependent failure that a differential test
  pinned to the default field cannot see. The field tests now run over all four
  moduli and over small elements specifically, because `(p+1)/2` and `(2p+1)/3`
  are the inverses with the most carries.

## Not the lever

Parallelising emission divides the same work across cores rather than removing
it, needs Y's first runtime dependency or a hand-rolled pool, and would have
made allocator contention worse before it made it better. Fixing the allocations
first was right, and it is still right for `optimize_circuit`.

## Instrumentation

- `Y_ZK_TIMING=1` — per-phase timings, the `emit_circuit_entry` /
  `optimize_circuit` split, `Fr` operation counts, and **resident and peak RSS
  per phase**. Always available.
- `Y_ZK_COMPOSITION=1` — the emitter's live memory broken down by owner
  (constraint terms, `Constraint` structs, `witness_recipes`, variable names).
  One pass over the constraints, only when set. This is what found the
  `emit_mul_lc` duplication.
- `--features alloc-stats` — counting global allocator. **Opt-in because it
  costs ~3.4 ns per allocation.** Now that allocations are 48x rarer its
  overhead is correspondingly smaller, but the rule stands: ratios, not absolute
  timings.
- `tests/zk_field_microbench.rs` — `--ignored`. Measures per-op cost and
  **asserts** the field primitives allocate zero, which is the property the whole
  change exists to establish.
