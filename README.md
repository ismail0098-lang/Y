Y
-----  A Systems Language and Compiler for GPU/CPU Hardware-Aware Code Generation

Y is a compiler and systems language for writing hardware-aware code across CPU
(x86-64 / AVX-512) and GPU (NVIDIA PTX) targets, with a zero-knowledge circuit
backend that emits R1CS and interoperates with the existing circom/snarkjs
toolchain.

The project is under active, single-developer development. It is a research
compiler, not a production toolchain.

---

## How to read the numbers in this file

Every benchmark here was run on one machine — AMD Ryzen 9 9950X, NVIDIA RTX 4070
Ti SUPER (Ada Lovelace, sm_89), 48 GB DDR5-6000 — and **none of it has been
independently reproduced on other hardware.** Where a result is a tie, it says
tie. Where Y loses, the loss is in the table rather than in a footnote.

Two conventions worth stating up front, because both were learned the hard way
here:

- **A ratio between 0.9 and 1.1 is a tie**, not a win. This box's run-to-run
  spread is ±7–8% on 16-thread CPU GEMM and a few percent on GPU kernels, so
  anything inside that band is reported as parity. The CPU figure is *measured*,
  not assumed: running two behaviourally identical binaries against each other
  as if they were an A/B gives 0.92–1.07, and that is the instrument's floor.
- **The GPU clock idles at ~210 MHz and needs ~3 s of load to reach ~2670 MHz.**
  Timing one implementation fully and then the other gives the second one a
  hotter clock — a systematic bias, not noise. GPU comparisons here ramp the
  clock first and then A/B-interleave.

Provenance is given per section. A number with a date attached was measured on
that date and has not been re-run since.

---

## What is real, and what is not

This section exists because an earlier version of this README claimed things the
repository's own investigation documents contradict.

**Real, measured, and reproducible from this repo:**

- R1CS / zero-knowledge circuit compilation, including a circom front end that
  compiles unmodified circomlib.
- FP16 tensor-core GEMM and fused GEMM+bias+ReLU on NVIDIA PTX.
- Fused RMSNorm+residual and RoPE kernels.
- A multi-threaded AVX-512 CPU GEMM, partitioned over a 2-D thread grid, with a
  copy-free path for shapes too small to amortise packing.
- `@safe` blocks with Z3-discharged loop invariants, `@ZeroDrift` exact
  accumulation, and linear tracking of async memory tokens.
- BN254 field arithmetic, NTT and MSM as compiler-emitted PTX, checked on the
  device against arkworks. The NTT is **ahead of icicle** at both sizes measured;
  the MSM is still behind it.
- A deterministic-inference path whose output does not depend on batch
  composition — 0/16 against a stock bf16 control that changes on 16/16 — at
  +0.12% perplexity.
- A C-callable shared library: the crate builds as `cdylib` as well as `rlib`
  (`src/c_api.rs`), so the compiler can be embedded rather than shelled out to.
- **Machine-checked proofs**: twenty Rocq files, 416 theorems and lemmas under
  260 `Print Assumptions`, no axioms and nothing admitted, all run by
  `cargo test`. They cover the ZK backend's control-flow lowering and — the bulk
  of them — the exact AVX-512 GEMM's schedule end to end, from the source dot
  product to the threaded, tiled, row- or K-split kernel. A compilation that
  substitutes that kernel now **emits its own certificate** beside the `.ll`.
  What the verified kernel *costs* is measured separately and is
  [in its own section](#what-the-verified-kernel-costs) — it is not free.
- **Translation validation against `ptxas`**: six GPU kernels — including one
  across a loop and one using shared memory and a barrier — proved to store
  exactly what their PTX stores, by symbolically executing the PTX and the SASS
  `ptxas` emitted from it and discharging 282 obligations in z3. It is a
  by-hand research tool (`tools/ptxas_tval/`), not a CI gate, and it covers one
  compilation of one kernel at a time. The negative control is in the table:
  the same kernel *without* `.rn` is refuted, because `ptxas` contracts to
  `FFMA`. [Details](docs/ptxas_translation_validation.md).
- **Zero runtime dependencies.** `[dependencies]` in `Cargo.toml` is empty; the
  compiler ships its own BN254 field arithmetic and its own JSON reader. The
  arkworks crates are `[dev-dependencies]` and are used as an *independent
  oracle* in tests — nothing in the `Y` binary links them.

**Not real, and previously presented as if it were:**

- **The "Dual-Accelerator RT + Tensor Core Co-Processor" is a scheduling
  simulation, not a capability.** The dependency graph, the slot assignment and
  the cost model are real code with real tests — but the per-node cycle costs
  are hardcoded constants, not measurements, and there is no public PTX
  instruction for BVH/ray-tracing hardware, so `rt_core_emitter.rs` cannot
  invoke an RT Core by construction. Disassembling the compiled SASS shows the
  whole kernel reduces to nine instructions: write a constant to shared memory
  twice, pack it to FP16, exit. **Zero RT instructions, zero `HMMA`, zero global
  memory traffic.** The "1.66x / 39.8% latency saved" figures this README used
  to headline were produced by comparing that against a CUDA busy-loop tuned to
  cost about what the RT trace was estimated to cost. Full write-up, including
  how it was confirmed on hardware:
  [investigation_rt_tensor_coprocessor_findings.md](investigation_rt_tensor_coprocessor_findings.md).
  The scheduler is kept as a design artifact; do not read its output as a
  measurement.
- **There are not five working backends.** LLVM IR (the default) and NVIDIA PTX
  are the two real ones. The C transpiler was removed — `--emit-c` says so and
  exits. `--emit-native` writes a runnable x86-64 ELF but covers only a
  **straight-line integer subset**: `let`, `return`, calls of up to six integer
  arguments, and the sixteen integer binary operators. It has no branches, so
  `if`/`while`/`for`/assignment, floats, strings, indexing and field access are
  refused by name with a line number, as are 64-bit types. Before that it
  emitted an ELF for all of them and computed the wrong answer under a success
  banner — `9 / 2` returned **9**.
- **Leo did not compile the ZK benchmark circuits.** Earlier tables reported
  timings for Leo at 100k and 1M constraints. Leo 4.2.0 refuses both: the
  compiled program exceeds its 512,000-byte limit (`leo build` on
  `leo/dot_product` errors at 14,322,372 bytes). Those rows have been removed
  rather than corrected.
- **The PTX backend compiled integers as floats.** *(Fixed since; kept here
  because this README asserted the opposite for a while.)* A kernel declaring
  `GlobalMemory<U32>` with `let s: U32 = a + b;` compiled clean, assembled clean
  under `ptxas -arch=sm_89`, reported success, and emitted `ld.global.f32` /
  `add.f32` / `st.global.f32` — silently rounding every value above 2^24. There
  is a real integer datapath now (typed loads and strides, signed-vs-unsigned
  arithmetic, `mul.wide.u32`, carry-flag chains, 128-bit vector loads), gated by
  `tests/ptx_integer_datapath.rs`, which runs 16 operations over 4,096
  full-range `u32` pairs on the device. That is what unblocked the BN254 kernels
  below. **Sub-word widths (`U8`/`U16`/`I8`/`I16`) are still refused**, because
  an element type is also a stride and there is no byte width threaded through
  the address math.
- **There is no AMD/ROCm backend.** `src/rocm_emitter.rs` was 173 lines
  compiled into the library and called by **nothing** — no CLI flag, no test,
  no caller anywhere in the tree — and has been deleted, along with
  `auto_vectorize.rs`, which was dead in the same way. The header of this file
  says "CPU (x86-64 / AVX-512) and GPU (NVIDIA PTX)" and that is the complete
  list.
- **`ypm`, the package manager, IS built — this entry used to say the
  opposite.** `src/ypm.rs` is declared by no `mod`, which is what the previous
  version of this paragraph checked, but it is a `[[bin]]` target in
  `Cargo.toml`: `cargo build --release` produces a working `target/release/ypm`
  and `ypm new / init / build / run / test / clean` all do something.
  **`mod` is not the same question as "compiled"**, and
  `tests/source_surface.rs` now asks both. What IS wrong is the documentation:
  `docs/y_language_documentation.md` §19 spelled every invocation `Y ypm ...`,
  which runs the compiler and reports `Failed to read file`, and documented
  `add` and `install`, which answer `Unknown command`. Corrected there; the
  registry side really is unimplemented.
- **Hopper TMA / WGMMA support never existed.** Sixteen of nineteen `emit_*`
  methods in that family produced PTX that `ptxas` rejects at their own target
  architecture. They were deleted rather than fixed. `mma.sync` — the path the
  working GEMM kernels use — was never affected.

---

## Zero-Knowledge Backend (R1CS → Groth16)

This is the most complete part of the project.

**Not in a default build.** Without `--features zk` the binary prints `The ZK
Circuit Backend is not compiled into this binary` and **exits 0** — a silent
no-op that reads as a fast successful run.

```bash
cargo build --release --features zk
Y circuit.ysu   --target=r1cs --witness input.json   # Y's own language
Y circuit.circom --target=r1cs -l path/to/circomlib  # circom 2.x
```

### It plugs into the existing toolchain

Y emits iden3-format `.r1cs` and `.wtns`, verified end to end against snarkjs:

```bash
Y circuit.ysu --target=r1cs --witness input.json
snarkjs groth16 setup circuit.r1cs pot_final.ptau circuit.zkey
snarkjs groth16 prove circuit.zkey circuit.wtns proof.json public.json
snarkjs groth16 verify verification_key.json public.json proof.json
# snarkJS: OK!
```

Circuit inputs are matched **by name** against `fn main`'s parameters — a file
listing values in the wrong order would otherwise produce a valid proof of the
wrong statement.

### circom front end

`Y foo.circom --target=r1cs` compiles circom 2.x through the same back end.
Everything downstream of constraint construction is shared with Y's own
language.

Its acceptance test is not "does it parse": circomlib's `Poseidon(2)`, compiled
from unmodified circomlib source, produces circomlib's own four published
digests **and** agrees with Y's native `poseidon_hash` on the same inputs — two
independent paths through the compiler, one answer. Output, public-input and
private-input counts match circom exactly on every circuit tested.

Measured 2026-08-11, best of three, same circom source through both compilers:

| | circom 2.2.3 | Y | |
|---|---|---|---|
| `Poseidon(2)` | 0.101 s | 0.023 s | **4.39x** |
| 200-hash Poseidon chain | 0.678 s | 0.631 s | **1.07x — a tie** |
| 1000-hash Poseidon chain | 3.249 s | 3.154 s | **1.03x — a tie** |

**The first row is not a general speedup, and the shape of this table is the
point.** A fixed ~5.5-million-allocation cost in hex-literal parsing was removed
on 2026-08-11 (`Poseidon(2)` went 0.076 s → 0.023 s), and merely `include`-ing
`poseidon.circom` — 24,958 lines of hex constants — used to cost 0.094 s before
a single constraint existed. That is a constant, so it dominates a small circuit
and vanishes into a large one. The chains are still ties because their cost is
**per-hash lowering**, at ~135 allocations per constraint against Y's own native
emitter's 12. Until that is fixed, expect small circom circuits to be fast and
large ones to be a tie.

**Coverage and size, measured across real circomlib** — 31 gadgets, not the 7
vendored here. Reproduce with `python3 tools/circomlib_coverage.py`:

| | |
|---|---|
| compiles | **Y 31/31, circom 31/31** |
| size vs circom `--O1` (its default) | **1.341x — Y wins** (win 11 / tie 20 / **loss 0**) |
| size vs circom `--O2` (its best) | **0.895x — Y loses** (win 0 / tie 20 / loss 10) |

`Sha256` and `EdDSA` were the two that did not compile. They were one gap — a
circom `var` holding a value that is not compile-time constant — and it is closed;
see [the witness domain](docs/circom_frontend.md#the-witness-domain-values-the-compiler-cannot-compute).
Both land near parity with circom's default, so they *lower* the geomean (1.352x →
1.341x) rather than raise it.

**`--O1` is circom's default and `--O2` is "full constraint simplification".**
Every size claim in this repo is against the default — what a user gets by typing
`circom` — and Y still **cannot reach `--O2`'s circuit sizes**: it reduces
`Bits2Num(64)` to zero, which Y has no equivalent for. What remains of that gap is
mostly a deliberate trade, not a missing pass: closing it costs 3.8x matrix
density and 1.60x proving time (see `docs/circom_frontend.md`). Y sits between the
two levels, closer on time to `--O1`:

| circuit | circom `--O1` | circom `--O2` | Y |
|---|---|---|---|
| Poseidon x400 | 1.269 s / 206,800 | 2.875 s / 96,000 | 1.273 s / 111,208 |
| MiMC x400 | 1.012 s / 145,600 | 1.029 s / 145,600 | **0.383 s** / 145,601 |
| EdDSAPoseidon | 0.317 s / 8,086 | 0.826 s / 4,217 | **0.243 s** / 7,570 |

Y is at circom's default speed on Poseidon and 1.3–2.6x faster elsewhere,
because its *simplification* is cheap — not its front end. circom `--O0` lowers
Poseidon faster than Y does with reduction off (0.415 s vs 0.879 s).

(That paragraph used to end "Still refused: `Sha256` and `EdDSA`", which had
been true and stopped being true two sections earlier in the same file. Both
compile.)

**Read the geomean, not the best row.** Where Y wins it wins large — `Poseidon(2)`
1.81x, `SMTProcessor` 1.71x, `SMTVerifier` 1.64x, `EscalarMul` 1.53x — but on
twenty-one of the twenty-nine it lands on circom's number, and on three it is
worse (`Point2Bits` 1,301 vs 1,560, `Mux1` 1 vs 2). The wins share a shape: circuits written as long
chains of `<==` linear assignments, which is exactly what
`substitute_linear_constraints` eliminates. Circuits that are already tight have
nothing to remove.

Every size figure this repo published before 2026-08-11 was measured on Poseidon
and quoted as a general result. It is not one.

| Poseidon specifically | circom | Y | |
|---|---|---|---|
| `Poseidon(2)` | 517 | 286 | 1.81x fewer constraints |
| 200-hash chain | 103,400 | 55,608 | 1.86x fewer |
| 1000-hash chain | 517,000 | 278,008 | 1.86x fewer |

Non-zero matrix terms land within 7% of circom's, so the smaller constraint
count is not bought by densifying the matrices — which is how this optimisation
usually goes wrong.

What that is worth downstream, measured through arkworks Groth16 on a 200-hash
chain rather than assumed:

```
unreduced  149423 constraints  149426 wires  518072 nnz   setup 0.522s   prove 0.617s
reduced     55608 constraints   55611 wires  357425 nnz   setup 0.228s   prove 0.288s
```

**2.29x on setup, 2.14x on prove**, against 2.69x on the constraint count. The
shortfall is structural rather than a missing optimisation: Groth16's cost splits
between terms that scale with the wire count and terms that scale with the
evaluation domain, which the constraint count fixes — so reducing wires moves
some of the prover and none of the FFTs.

Wires are compacted (`compact_wires`): the reduction passes abandon wires and
neither renumbers, which left Y at 153,605 wires against circom's 103,403 on this
circuit even while emitting 1.86x fewer constraints. Compacting takes that to
55,611 — 1.86x fewer than circom on both axes — for ~1.6% of compile time, and
shrinks the Groth16 proving key from **25.4 MB to 10.5 MB**.

**Coverage on circuits people actually deploy, not just circomlib.** circomlib
is a gadget library; the circuits above it are where a front end either works or
does not:

| suite | before | now |
|---|---|---|
| circomlib (31 gadgets) | 31/31 | 31/31 |
| [circom-ecdsa](https://github.com/0xPARC/circom-ecdsa) (17 circuits) | **0/17** | **17/17** |
| zk-email `RSAVerifier65537(121, 17)` | did not compile | 189,271 constraints vs circom's 190,945 (**0.991x**) |
| zk-email `Sha256Bytes(64)` | did not compile | 32,334 vs 33,136 |
| semaphore, rln, zk-kit, withdraw | did not compile | compile; sizes at or below circom `--O1` |

**Almost every one of those was Y refusing a WITNESS-domain construct because
the CONSTRAINT domain could not hold it** — `assert` over signals, over-quadratic
arithmetic accumulated in a `var`, an unknown array index. None of them emits a
constraint, so refusing them was over-strict rather than safe. The boundary is
at the USE (`o <-- t[u]` is fine, `o <== t[u]` is not), and it was established by
probing circom itself rather than read off the documentation.

Two were not refusals at all. **circom compares `var`s as SIGNED and Y compared
them canonically**, so `var a = 0 - 1; if (a < 1)` produced a circuit computing
`111` under circom and `222` under Y — both valid, both provable, differing
silently, and cross-compiler. The same bug's other face is that
`for (var j = k-1; j >= 0; j--)` never terminates canonically, which is the shape
`circom-ecdsa` and `zk-email/rsa` are built on. And **circom 2.1's anonymous
components** (`Poseidon(2)([a,b])`, `(x, y) = BabyPbk()(s)`, `_` to discard) were
genuinely missing; they are a desugaring, and the acceptance test is that
`o <== T()(i)` emits **byte-identical** `.r1cs` to the explicit form — a property
circom itself satisfies, checked before anything was written.

**`--witness` reads circom's own `input.json`,** and the witness it produces
agrees with circom's calculator element for element — verified further through
`snarkjs wtns check` and a full `groth16 setup/prove/verify` on Y's artifacts.
Three separate bugs had to be fixed for that, and only one was fail-closed: the
public inputs were never bound, so every signal in a `{public [...]}` list was
solved at **zero**. Y's own language has no `public` keyword, which is why
nothing internal could see it.

Detail, subset, and the constructs that are refused by name:
[docs/circom_frontend.md](docs/circom_frontend.md).

### Compile speed on Y's own language

The two benchmark circuits are an unrolled polynomial (`temp = temp * y`) and an
iterative dot product (`sum += a * b`). The harness **aborts the comparison
unless both compilers report the same non-linear constraint count**, because
comparing compile speed across tools that built different circuits is
meaningless.

Polynomial rows measured 2026-08-09, minimum of three
(`tests/benchmark_zk_vs_circom.py`); the 10,000,000 row and **all three dot
product rows** 2026-08-10 on an idle box, minimum of five below 1M. The two 1M
figures are single runs of circom (978.6 s and 16 minutes apiece) against a
minimum of three for Y; a control of two byte-identical Y binaries reads 1.01x,
so the noise floor is ~3% and none of these ratios is near it:

| circuit | N | Y | circom | speedup |
|---|---|---|---|---|
| polynomial | 10,000 | 0.008 s | 0.132 s | 16.5x |
| polynomial | 100,000 | 0.087 s | 3.53 s | 40.5x |
| polynomial | 1,000,000 | 0.895 s | 249.04 s | **278x** |
| polynomial | 10,000,000 | 10.47 s | **did not finish in 1 h** | **>345x** |
| dot product | 10,000 | 0.011 s | 0.517 s | 46x |
| dot product | 100,000 | 0.122 s | 14.23 s | 117x |
| dot product | 1,000,000 | 1.27 s | 978.6 s | **773x** |

**Read the two circuits separately, and read the dot-product row with the
constraint counts in front of you — the ratio is real and it is also the least
honest number on this page.**

The polynomial circuit's linear combinations are one or two terms wide, and both
tools emit the same thing: circom reports 1,000,000 non-linear and **0 linear**
constraints against Y's 1,000,001. Same artifact, and Y is 278x faster building
it. That is the clean comparison.

**At 10M the 10x row is a bound, not a number, because circom did not finish.**
It was given a one-hour wall clock under `systemd-run -p MemoryMax=40G` on an
otherwise idle box and killed at 3,612 s, still running, at 3.53 GB. Y compiles
the same circuit in **10.47 s** (minimum of three) at 3.60 GB, writing a 1.19 GB
`.r1cs`. So the row says `>345x`, which is what was observed; the true figure is
larger. Circom's own scaling says how much larger — 3.50 s at 100k, 26.77 s at
316k, 245.69 s at 1M is **O(N^1.9)**, which projects ~5.4 hours at 10M — but that
is a projection and is not what the table reports. Y is linear across the same
range (0.087 → 0.895 → 10.47 s), so the gap widens with size rather than
converging. Two caveats worth stating: the harness's fairness gate could not be
applied at 10M, since circom emitted no constraint count to compare (both
circuits are the same template, scaled), and this is the *narrow* circuit — the
dot product below is linear in Y too now, but there the two tools build
different-sized artifacts, so its ratio is not comparable to this one.

The dot product accumulates a dense linear combination, and **Y used to be the
super-linear one on it** — 476 seconds at 1M, roughly O(N²·²), against circom's
983. That was a real defect and it is fixed: `sum = sum + a * b` appends one
freshly allocated wire to `sum` per iteration and then called
`LinearCombination::simplify`, whose "already sorted" fast path still scans every
term to conclude it has nothing to do. An accumulator holding `i` terms at
iteration `i` therefore cost N²/2 term visits. Wire ids are allocated ascending,
so appending a fresh one cannot break sortedness; deciding that from the boundary
term is O(1). Measured on the same box, minimum of runs: **424.8 s → 1.27 s at
1M (336x), 3.14 → 0.12 s at 100k**, and the curve is linear now. Details and the
attribution in
[docs/heavy_circuit_speed_test.md](docs/heavy_circuit_speed_test.md).

**But 773x is not a like-for-like number, and the reason has nothing to do with
that fix.** The two tools emit different artifacts here: circom emits 1,000,000
non-linear **plus 3,000,000 linear** constraints, Y emits 1,000,001 total,
because Y folds the linear operations into the combinations instead of
allocating a signal per intermediate. So Y is building roughly a quarter of the
constraints, and the harness's fairness gate compares *non-linear* counts only,
which is why this pair passes it. A same-total-constraints comparison would be
markedly less favourable. Y's R1CS is genuinely smaller and correspondingly
cheaper to prove — 0.22 GB against circom's 0.48 GB, at 0.51 GB peak RSS against
circom's 10.9 GB — but "773x faster" and "773x more work per second" are not the
same claim and only the first is measured.

**Quote the polynomial row, not this one, if you want one number.** The
polynomial circuit is the strictly like-for-like comparison: same non-linear
count, zero linear constraints on either side, same artifact.

### What it looks like on circuits people actually build

**Neither 278x nor 773x survives contact with a real circuit, and the honest
range is 1.0–4.4x through the circom front end and 3.5–23x through Y's own
language.** Both benchmark circuits are single unrolled loops a million
iterations long, which is the shape circom is worst at; real circuits are built
from hash and range-check gadgets. Measured 2026-08-11 on an idle box, minimum
of five, **circom source compiled by both tools** so the front end is not a
variable:

| circuit (circom input) | Y | circom | speedup | Y cons. | circom cons. |
|---|---|---|---|---|---|
| `Poseidon(2)` | 0.023 s | 0.101 s | **4.39x** | 286 | 517 |
| Merkle inclusion, depth 20 | 0.059 s | 0.151 s | **2.54x** | 5,685 | 10,400 |
| Poseidon chain, 200 hashes | 0.631 s | 0.678 s | **1.07x** | 55,608 | 103,400 |
| Poseidon chain, 1000 hashes | 3.154 s | 3.249 s | **1.03x** | 278,008 | 517,000 |

**On Y's own `.ysu` front end the same circuits are 3.5–23x**, because
`poseidon_hash` folds its linear layers as it builds them instead of allocating
a signal per intermediate:

| circuit (Y's own language) | Y | circom | speedup | Y cons. | circom cons. |
|---|---|---|---|---|---|
| `Poseidon(2)` | 0.0045 s | 0.101 s | 22.8x | 241 | 517 |
| Merkle inclusion, depth 20 | 0.0118 s | 0.151 s | 12.8x | 4,841 | 10,400 |
| Poseidon chain, 200 hashes | 0.169 s | 0.678 s | **4.02x** | 47,404 | 103,400 |
| Poseidon chain, 1000 hashes | 0.942 s | 3.249 s | **3.45x** | 237,004 | 517,000 |

**The ratio falls as the circuit grows (22.8x → 3.45x), and that is circom
amortising a fixed cost, not Y degrading.** Y's own scaling was checked
directly rather than inferred: across 125 → 2,000 hashes every phase is linear
(emit 0.035 → 0.442 s, optimize 0.040 → 0.730 s, write 0.058 → 0.656 s, all ~2x
per 2x) and the total is 13.7x for 16x the work — *sub*-linear, because Y's own
fixed cost is amortising too. At the margin Y is a flat ~3.4x per hash at both
200 and 1,000.

**Quote 3.45x, not 22.8x.** Circom carries a fixed startup of roughly 0.09 s —
mostly parsing `poseidon_constants.circom` — so on a 517-constraint circuit that
constant *is* the measurement, and the two small rows mostly report it. At
517,000 constraints it has amortised away and 3.45x is what remains. Note this
is not a like-for-like circuit comparison either: Y emits **2.15–2.18x fewer
constraints** for the same computation, and the hash is verified identical
against circomlib's published digests, so read it as "same computation, smaller
circuit, less time" rather than as raw compiler throughput.

One thing that scaling check did turn up: **`optimize_circuit` is 33–40% of
compile time on these circuits and removes 1.26% of the constraints**, and its
share grows with N. That reads like a bad trade and is not one, because a
circuit is compiled once and proved many times — measured with
`Y_ZK_CSE=off` and arkworks Groth16 on a 100-hash chain, it costs 0.015 s of
compile and saves 0.0035 s per proof, so it **pays for itself after 4 proofs**
(`tests/zk_cse_cost.rs`). It is off-able for measurement, and on by default.

**Read the chain rows as ties, not wins**, and read the spread down each table
as the real shape of it: the small circuits are fast because a fixed
~5.5-million-allocation cost in hex-literal parsing was removed (2026-08-11),
and the chain barely moves because its cost is per-hash lowering, which is
untouched — ~135 allocations per constraint against the native emitter's 12.
Expect the chain row to improve when that is fixed and not before.

The durable result on circom input is not speed at all — it is that Y emits
**1.81–1.86x fewer constraints** for the same circuit (2.15–2.18x through Y's
own language), which is worth roughly 1.4x at Groth16 proving time, not the
1.8x the count suggests: the substitution pass deletes constraints but does not
compact the wires they used, and Groth16 pays for wires too.

**Where the 773x actually goes**, since it is still 34x larger than any row here:
circom emits 4x the constraints on the dot product, and costs **244.6 µs per
constraint** at that size against Y's 1.27 µs. Circom's own per-constraint cost
is not a constant — it is 7.0 µs on the Poseidon chain and 244.6 µs at 1M, a 35x
spread, because circom is superlinear and Y is linear. So the benchmark's *size*
produces most of the ratio, and 4 × (244.6 / 1.27) = 771x accounts for
essentially all of it.

**Coverage, and a limit that was just removed:** until 2026-08-11 Y's circom
front end could not compile circomlib's `bitify.circom`, because `Num2Bits`
computes its witness with `out[i] <-- (in >> i) & 1` and Y applied its
*constraint* value model to the `<--` right-hand side — refusing a shift that
never becomes a constraint. That ruled out `Num2Bits`, `comparators.circom`,
`aliascheck.circom` and everything built on them: range checks and comparisons,
i.e. most of a real circuit. Fixed: a 200-wide `Num2Bits(64)` range check is
13,013 constraints against circom's 13,200, and 300 `LessThan(32)` comparisons
are 10,220 against 11,100, both from unmodified circomlib.

**Memory is the binding constraint on this backend, not time.** Peak RSS,
polynomial circuit, measured 2026-08-09:

| Constraints | time | peak RSS | `.r1cs` on disk |
|---|---|---|---|
| 1,000,000 | 0.90 s | 0.37 GB | 0.13 GB |
| 10,000,000 | 10.47 s | 3.60 GB | 1.19 GB |
| 31,000,000 | 36.1 s | 11.4 GB | 3.97 GB |

Cost is linear in both. **But per-constraint memory is a property of the
circuit, not of the compiler**: this circuit's linear combinations are 1–2 terms
wide (0.37 KB/constraint), where a Poseidon chain's are ~28 (1.42 KB/constraint).
On a 48 GB box that is ~105M constraints of the former and ~32M of the latter.
Quote the dense number. Circom, Noir and Leo were not run at 31M — earlier
tables here reported estimates for them at that size as if measured, and those
have been removed. Circom *was* run at 10M and did not finish in an hour; at the
point it was killed it held 3.53 GB against Y's 3.60 GB peak for the completed
job, and its memory is linear at ~2.3 KB/constraint, so finishing would have
cost it roughly 23 GB.

### Proving cost, end to end

`cargo test --release --features zk --test zk_groth16_scale -- --ignored`,
measured 2026-08-09; the 10M and 31M rows 2026-08-10. Y has no prover of its own
and performs no trusted setup — setup and prove are arkworks, reached through
Y's R1CS.

| Constraints | emit | witness | setup | prove | verify | total | peak RSS |
|---|---|---|---|---|---|---|---|
| 10,000 | 0.01 s | 0.00 s | 0.04 s | 0.05 s | 0.002 s | **0.11 s** | |
| 100,000 | 0.12 s | 0.01 s | 0.40 s | 0.36 s | 0.002 s | **0.89 s** | |
| 1,000,000 | 1.25 s | 0.09 s | 2.83 s | 2.99 s | 0.002 s | **7.17 s** | |
| 10,000,000 | 11.70 s | 0.92 s | 30.73 s | 39.90 s | 0.002 s | **83.25 s** | 31.8 GB |
| 31,000,000 | 40.35 s | 2.83 s | **OOM** | — | — | — | >40 GB |

arkworks is **81%** of the 1M total and **85%** of the 10M one — its share grows
with size. The remaining headroom is in a prover Y does not have, so "Y compiles
circuits 278x faster than circom" and "Y produces proofs faster" are different
claims and only the first is supported.

**31M constraints cannot be proved on this machine, and the row says so rather
than estimating it.** Under `systemd-run -p MemoryMax=40G -p MemorySwapMax=0`,
31M reaches the 40 GB cap and is OOM-killed 56 s in, still inside
`circuit_specific_setup`. Emit and witness — the half Y owns — finish in 43 s at
30.2 GB. (That is higher than the 11.4 GB the memory table above reports for the
same size because this harness holds the circuit, the witness IR and the solved
witness live at once, where `--target=r1cs` streams the constraints to disk and
drops them. Same circuit, different question.) The wall is Groth16's proving
key, which holds `num_variables` G1 points for each of `a`, `b_g1` and `l`, as
many G2 points for `b_g2`, and a domain-sized `h`; at ~3 GB per million
constraints on this circuit, 31M needs on the order of 100 GB. That is a
property of the prover and the curve, not of Y's emitter, which is why the
constraint-emission ceiling in the memory table above (105M on this circuit) is
so much higher than the *proving* ceiling (~10M).

### Correctness

`tests/zk_groth16_end_to_end.rs` proves Y's circuits through arkworks as an
independent oracle: an honest proof must verify, a tampered public input must be
rejected, a perturbed witness must fail satisfiability, and Y's modulus must
equal the true BN254 one. String-matching an emitted `.r1cs` cannot catch a
wrong field or a mis-numbered wire; this can.

- **Poseidon is circomlib's**, pinned against digests taken from circomlib's own
  `Poseidon(2)` — 241 constraints against circom's 243 non-linear + 274 linear.
  Only t=3 (two inputs) is supported; other arities and non-BN254 fields are
  refused rather than approximated.
- **Comparisons cost ~101 constraints, by necessity.** A field has no order, so
  an ordering claim without a range proof is vacuous. `<`, `<=`, `>`, `>=`
  range-check both operands and decompose the difference.
- **Bitwise, shift, `/` and `%`** are supported: `&`/`|`/`^` cost 98, `<<`/`>>`
  33 (constant shift amounts only — a variable shift is a 32-way multiplexer, so
  it is refused rather than approximated), `/` and `%` 135. `/` is integer
  division. Values are unsigned 32-bit; a negative operand fails its range check
  and is unprovable rather than wrong.
- **`@zk_target(scheme = "plonkish")` is refused.** It used to parse, print
  `Proof Scheme: Plonkish` into the artifact header, and emit R1CS anyway.

### On-chain verifier

```bash
Y --emit-verifier verification_key.json -o Verifier.sol --name MyVerifier
```

Generates a Groth16 verifier calling the BN254 precompiles at
`0x06`/`0x07`/`0x08`, with the same
`verifyProof(uint[2], uint[2][2], uint[2], uint[N])` signature snarkjs emits, so
its exported calldata and existing front-ends work unchanged.
`tests/zk_solidity_verifier.rs` compiles the contract with `solc` and executes it
on a real EVM via `revm`. (The failure mode that matters — G2 coordinates in
library order rather than the EVM precompile's reversed order — produces a
contract that compiles, deploys, burns the full gas of a pairing check and
rejects every valid proof. No string-matching test can catch that.)

---

## GPU: NVIDIA PTX backend

Hardware: RTX 4070 Ti SUPER, 66 SMs. Theoretical dense FP16 tensor-core peak at
the 2.61 GHz boost clock is **88.1 TFLOPS**.

### Square FP16 GEMM vs cuBLAS

Measured from `target/release/Y tests/gemm_f16_<N>.ysu --emit-ptx` — the
compiler's own output, not a hand-written CUDA reference. Interleaved A/B,
ranked by minimum, correctness-checked every run.

| M=N=K | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| **Y vs cuBLAS** | 1.12x | 0.84x | 0.89x | 0.93x | 0.93x | 0.94x |
| **Y TFLOPS** | 11.1 | 38.2 | 69.8 | 79.5 | 79.1 | 81.6 |

**Y is behind cuBLAS at every size above 256, by 6–16%.** 256 reads 1.12–1.23x
across runs and 4096 reads 0.93–0.94x; both are at the edge of what a
3 µs kernel reproduces to, but the ordering is stable. This is a large
improvement on the 0.61–0.94x the same benchmark measured before the mainloop
work (`ptxas` could not unroll the `cp.async` staging loops because their trip
count depended on `%tid.x`, costing ~22 SASS instructions per 16-byte copy), and
it is still a loss.

### Fused GEMM + bias + ReLU vs cuDNN

Precision-matched — the cuDNN baseline is fed the same FP16 operands the Y
kernel consumes. (An FP32 `torch.nn.Linear` baseline shows 2.9–3.4x, but that
conflates "the fused epilogue is good" with "FP16 beats FP32".)

| M=N=K | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| **Y vs cuDNN FP16** | 1.07x | tie | tie | tie | 1.08x |

Epilogue fusion is where fusion pays. The general rule this established, after
measuring it the other way: **fusions that only remove work win; mainloop
fusions that add accumulator pressure force a worse tile and tend to net zero.**
Fused SwiGLU measures **1.00x against Y's own unfused two-GEMM path** — its two
FP32 accumulator arrays cost 8 registers each per tile element, so a 128x128 tile
over 2x2 warps would need 256 registers/thread against ptxas's 255 cap, pinning
it to the 4x4 split that measures 52.5 against 75.8 TFLOPS in the plain GEMM.

### RMSNorm and RoPE vs FlashInfer

Compared against FlashInfer's hand-tuned production kernels (what vLLM and
SGLang actually use), same math and same convention — not against eager PyTorch.

| kernel | rows | result |
|---|---|---|
| Fused Add+RMSNorm (hidden=4096) | 128 / 1024 / 8192 | parity at every size |
| Fused RoPE (head_dim=128) | 128 | 1.85x |
| Fused RoPE (head_dim=128) | 1024 | 1.92x |
| Fused RoPE (head_dim=128) | 8192 | tie |

RoPE wins in 8 of 9 (head_dim, rows) combinations across head_dim 64/128/256.

### Paged decode attention vs FlashInfer

head_dim 128, 32 query heads, 8 KV heads (GQA 4:1), page_size 16, against
FlashInfer's `BatchDecodeWithPagedKVCacheWrapper` on the same shuffled page
table and the same NHD KV layout. Ramped clocks, A/B interleaved, minimum of 7
rounds. `python3 tests/benchmark_y_paged_decode_attention.py`.

| case | FlashInfer | Y single-pass | Y split-K | split-K vs FI |
|---|---|---|---|---|
| batch 1, ctx 1024 | 8.4 µs | 12.6 µs | **8.0 µs** | **1.06x** |
| batch 1, ctx 4096 | 13.7 µs | 43.0 µs | 16.9 µs | 0.81x |
| batch 8, ctx 1024 | 22.1 µs | 48.1 µs | 29.2 µs | 0.76x |
| batch 8, ctx 4096 | 189.5 µs | 226.6 µs | 201.9 µs | 0.94x |
| batch 32, ctx 1024 | 197.1 µs | 246.1 µs | 211.6 µs | 0.93x |
| batch 32, ctx 4096 | 886.3 µs | 995.9 µs | 861.5 µs | tie (1.03x) |

One run, not a best-of. Y's columns reproduce to ~1% across runs; FlashInfer's
batch-32/ctx-4096 figure moved between 825 and 886 µs over three runs, which is
the whole of that row's 0.97–1.03x — read it as a tie, not as a win.

**This was 1.3–3.7x slower and is now 0.76–1.06x**, i.e. between 1.3x slower and
6% faster depending on the shape. The two causes this README used to name were
both real, and neither could be fixed alone:

* **The GQA re-read.** The single-pass grid is `num_q_heads x num_seqs`, so the
  four CTAs sharing a KV head each streamed that head's whole cache — four
  times the necessary DRAM traffic. The split-K kernel's grid is
  `num_kv_heads x num_seqs x splits` and one CTA carries all four query heads,
  so a token's K and V are loaded once and feed four scores.
* **No split-K.** With `num_q_heads x num_seqs` CTAs, batch 1 is 32 CTAs against
  66 SMs whatever the kernel does internally. `%ctaid.z` now partitions the KV
  sequence, so the CTA count stops depending on batch size.

Merging the GQA heads *without* split-K would have made batch 1 worse, not
better — 8 CTAs instead of 32. That is why the fix is one kernel shape rather
than two independent changes, and why the split kernel is a second entry point
(`..._reduce`, a max-rescale-and-sum over the per-split partial softmax states)
rather than an option on the existing one: it needs two f32 scratch buffers and
a device-wide barrier that does not exist inside a kernel. Both launches are
inside the timed region above.

Two things this does not claim. The split count is compile-time (16 for batch 1,
4 for batch-many — `paged_decode_attention_split_128_32_8_16_<splits>_<warps>`)
and its optimum depends on batch size, which is a runtime value; a single
compiled binary shipping only the 16-split kernel measures **0.72–1.06x**
instead of 0.76–1.06x. And at batch 32 / ctx 4096 the kernel moves 623 GB/s of
the KV bytes it is obliged to read, against this card's 672 GB/s DRAM
ceiling — 93% of the roofline, with FlashInfer between 90% and 97% across runs.
There is nothing left to win there; the remaining gap is entirely in the
L2-resident shapes.

The single-pass kernel got 1.12–1.43x faster on the way, and **not** for the
reason that sounds likely. Hoisting the V load above the QK warp reduction is
worth 1.04–1.38x on its own: `exp` and five dependent `shfl` sat between that
load's issue and its use, so its latency was exposed rather than overlapped.
Restricting the cross-warp merge to warp 0 — instead of having all 32 warps
compute an answer 31 of them discard — measured **0.83–1.10x in isolation**,
a wash: it removes work but lengthens the live ranges around the branch, and
`ptxas`'s register count went *up* (54 → 64 at 32 warps, 34 → 52 at 8). It is
kept because it is what makes the multi-head epilogue affordable, not because
it was a speedup.

### BN254 field kernels, NTT and MSM

Once the integer datapath existed, the ZK backend's arithmetic could be
expressed as Y source and compiled to PTX. These are the compiler's own kernels,
checked on the device against `src/zk_field.rs` and against arkworks.

| kernel | result | baseline |
|---|---|---|
| Fr Montgomery multiply | **0.160 ns/mul, 6.23 G mul/s** | 89% of the card's DRAM peak — memory-bound, not compute-bound |
| NTT, N = 2^20 | **0.56 ms** | icicle 0.608 ms |
| NTT, N = 2^22 | **2.47 ms** | icicle 2.834 ms |
| MSM, n = 2^22, kernel | 6.10x a 32-core CPU | **still 2.8–4.8x behind icicle** |
| Groth16 prove, 2^20 constraints | **~5–6x** arkworks on 32 cores | sparse *and* dense circuits, ±15% run to run |

**The NTT result is the halved pass count, not the arithmetic.** Radix-4 plus
shared-memory stage fusion takes 11 passes to 3 at N = 2^22 and is what puts Y
ahead of icicle; the fused kernel is no longer DRAM-bound (39% of peak, 71% SM)
so the bottleneck moved rather than shrank. **Carry-flag intrinsics were
predicted at 2x and measured at 1.06x** — `IMAD.WIDE` already produces both
halves of a 32x32 product in one instruction, so the two-pass carry form doubles
the multiply count exactly as it collapses the bookkeeping. Count the work an
instruction does, not the instructions.

**The Groth16 figure is honest about which circuit it was measured on.** A
sparse polynomial benchmark circuit reported 3.61x where a dense Poseidon chain
reported 1.76x, and the phase split *inverted* — matrix build 73% instead of
45%, MSM 10% instead of 39%. Both are ~5–6x after the O(k²) and the
single-threaded matvecs that difference exposed were fixed. On the dense circuit
the largest remaining phase is the **G2 MSM at 47%**, which needs `Fq2`
arithmetic that does not exist in this kernel series — a roadmap drawn from the
sparse circuit alone would have ranked it last.

**Do not multiply these against the circom compile-speed numbers.** Those
measure *emitting* R1CS; circom does not prove at all. Different stage,
different baseline.

### Where the GPU backend loses

| workload | baseline | result |
|---|---|---|
| **FP8 (e4m3) GEMM** | `torch._scaled_mm` | **0.16–0.26x — 4–6x slower** |
| Paged decode attention, L2-resident | FlashInfer | 0.76–0.81x |
| Decode-shaped GEMM (M=4–8) | cuBLAS | at the DRAM roofline; tied |

FP8 is the largest gap and is not being chased: this is Ada, not Hopper, and the
kernel is instruction-bound in its quantize-and-stage step. Paged decode
attention is now at the DRAM roofline when the KV cache does not fit in L2, and
loses by up to 1.3x when it does. What binds it there has not been measured; the
suspect is the per-token instruction count — one 32-lane butterfly reduction per
query head, so four per token, which the GQA merge amortized the *loads* over
but not the reductions.

### Memory bandwidth

Y's elementwise and normalization kernels emit 128-bit vector loads and stores
(`ld.global.v4` / `st.global.v4`), measuring **663 GB/s against this card's
672 GB/s theoretical GDDR6X ceiling — 98.7%.** PyTorch's unvectorized 32-bit
access pattern on the same kernels measures 520 GB/s (77.3%).

*Measured earlier in the project's history; not re-run for this revision.*

### Cold compilation latency

Y emits PTX directly from Rust and loads it through the CUDA Driver API:
**0.078 ms**, against ~50 ms for Triton / PyTorch Inductor, which parse Python,
generate a C++ wrapper and shell out to `nvcc`/`ptxas`. This is a compile-time
comparison, not a kernel-speed one — it matters for dynamic LLM prompt shapes
where a new kernel is needed per shape, and for nothing else.

*Measured earlier in the project's history; not re-run for this revision.*

### PTX is gated on assembling, not on string matching

`tests/ptx_intrinsics_assemble.rs` compiles a probe `.ysu` per intrinsic through
the real binary and runs `ptxas`. Adding that gate is what found the sixteen dead
Hopper intrinsics, a `.maxnreg 0` bug that made **every** kernel compiled without
a probed hardware profile structurally invalid, and two live user-callable
intrinsics (`tma_load`, `wgmma_async`) that printed "PTX Assembly generated
successfully!", exited 0, and wrote a file `ptxas` rejects with five distinct
errors. Both now refuse and fail the build. Prefer extending this gate to adding
another substring assertion.

### Portability: the artifact must not name the build machine

A compiler that probes the local machine will bake that machine into its output
unless something stops it. Y did, in seven places, and the failure is not subtle:
a `.target` above the running device is a **hard load failure**
(`CUDA_ERROR_NO_BINARY_FOR_GPU`) — the kernel does not run, on a machine nobody
tested. PTX is forward compatible and never backward, so the correct target is
the *lowest* architecture the instructions require. Guess down.

**You can check this without owning the cards.** `ptxas -arch=sm_80|sm_86|sm_89|
sm_120` assembles for architectures that are not plugged in, so "will this run on
a 3060" is answerable locally — which is the whole reason `tests/ptx_portability.rs`
exists. **93 of 97 committed `.ptx` files assembled at sm_80 unchanged**, so the
over-specification was ~96% gratuitous.

**`.version` is a separate requirement from `.target`: it is what the DRIVER must
support**, and over-stating it makes a kernel unloadable on a machine whose driver
is merely older (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`) — invisible to any assemble
gate, because `ptxas` is perfectly happy to assemble an over-stated version. The
floors are measured by bisecting `.version` under `ptxas -arch=<a>`, not read off
a release table:

```
sm_75 6.3   sm_80 7.0   sm_86 7.1   sm_87 7.5   sm_89 7.8   sm_90 8.0   sm_120 8.7
```

Four of the seven were over-stated. sm_89 was **8.4** — a whole CUDA major — for
every kernel, because FP8 `mma.sync` needs 8.4 *on that arch*. That is a
**per-instruction** requirement and now lives on the module: a plain sm_89 kernel
declares 7.8, an FP8 one raises the floor to 8.4 through `require_ptx_version`.
sm_90 keeps 8.0 although `ptxas` 13.3 accepts 7.8, because sm_90 arrived in CUDA
12.0 and this assembler is merely being lenient — guess down, but not below spec.

**Fixing the artifacts is not fixing the compiler, and this took three passes to
learn.** The first corrected the committed `.ptx` files; the emitter still
over-stated. The second fixed the emitter; `--emit-coprocessor`, which builds its
own module header rather than going through `emit_program`, still wrote
`.version 8.0` by hand — over-stated on Ada, and **rejected outright by `ptxas`
on Blackwell** while the backend printed "generated successfully!" and exited 0.
The gate that should have caught it hardcoded `-arch=sm_89`, the one architecture
where 8.0 is legal.

There are therefore two source-level gates as well as the artifact ones: no source
file may hardcode a `.target` above the floor, and none may hardcode a `.version`
above it. A literal in a Rust format string cannot be prevented from coming back
by assembling the current output.

**The CPU side had the same bug and it was worse — a SIGILL rather than a load
failure.** `CpuHardwareProfile::default()` guessed *up* (AVX-512 masking, 16
floats) and `--emit-cpu` used that default as its only profile, so it emitted
AVX-512 dispatch on every machine; most consumer Intel since Alder Lake has none.
Meanwhile the real CPUID probe existed and **had zero callers**. Two more
portability failures had nothing to do with hardware at all: `c_src/runtime.c` was
a CWD-relative path, so the LLVM backend linked only from inside its own source
tree, and `-lX11` was linked unconditionally, so a headless machine could not
build any Y program.

The one genuine hardware requirement **refuses** rather than emitting: FP8
`e4m3` really is Ada and later, so below sm_89 it fails by name and points at the
int8/f16 path. Asserted as a biconditional — "refuse FP8 always" satisfies half of
it and deletes a working path on the hardware that has it.

---

## CPU: AVX-512 GEMM

A multi-threaded f32 GEMM emitted through the LLVM backend, against OpenBLAS
built for the same ISA (`TARGET=SKYLAKEX`, 39,476 `zmm`). All-core clock
5.09 GHz, so AVX-512 peak is 5212 GF. Every shape gated on relative L2 error
< 1e-5. One shape per process per library, arms interleaved and rotated, four
launches, OpenBLAS measured in the same session.

**Geomean 1.52 on 16 threads, Y ahead on 13 of 18 shapes. But read it by class,
because the mean hides where the gains came from:**

| class | shapes | Y vs OpenBLAS |
|---|---|---|
| GEMV | `1×4096×4096`, `1×8192×8192` | **2.1–4.9x** |
| Decode (M=4–8) | `4×4096×4096`, `8×4096×4096` | **1.8–3.1x** |
| Small / ragged square | `250³`, `256³`, `333×777×64` | **2.4–2.8x** |
| Rank-k / deep-k | `4096×4096×8`, `64×64×32768` | **1.1–1.6x** |
| Tiny | `48³` | **1.00x** (was 0.43) |
| Large square | `512³`, `1024³` | 1.08–1.13x |
| Large square | `1021³`, `1000³`, `2048³` | **0.81–0.95x** |

**Most of the 16-thread gain is Y no longer under-threading its own kernel**, not
Y beating OpenBLAS at dense arithmetic. The thread-count constant had been
calibrated against a redundant-packing cost that a 2-D `ntm × ntn` partition
removed, and it was 64x too large; fixing it is worth 2.2–3.3x on the small and
ragged shapes and nothing at all on the large ones. Large square GEMM is still
the weak class — `2048³` is 0.81x — and a reader who cares about that should
read those rows and ignore the geomean, which is sensitive to how many small
shapes the set happens to contain.

Single-threaded, Y is at 1.50 geomean — but that column is a control rather than
a headline. At one thread the pool and the partition are bypassed entirely, so
every shape that does not route to the copy-free kernel must be *unchanged* by
this work, and all seventeen are: **0.976–1.020, at spreads of 0.5–2%**. That is
what says the micro-kernel was not touched, and at 16 threads a ±7% instrument
could not have told you. (The 1.50 was measured one revision before the final
row-block change, which moves `48³` and nothing else, and moves it upward.)

> **The throughput harness measures one regime and cannot see the other.** It
> calls the kernel in a tight loop, so the thread pool never parks and a
> dispatch is nearly free. Timing a *single* call after a gap instead, 16
> threads against 1: `56³` measured **0.11x** — a fixed ~20 µs dispatch cost,
> flat from `nt=2` to `nt=16`. The thread count is chosen from the caller's call
> frequency now, which recovers it to 0.92–1.00x with throughput unchanged.
> Spinning longer does not fix it (fast while the caller's gap fits the spin
> window, 175 µs just past it) and asking the pool whether workers are parked
> *latches*.

> **Seven independent biases were found in the harness that produced the first
> version of these numbers, and all seven favoured Y.** Both libraries timed in
> one process (OpenBLAS's idle threads spin before parking, so whichever ran
> second was measured against a busy machine); thread count driven through
> `openblas_set_num_threads()` on a libgomp build; all 18 shapes in one process,
> depressing later ones; a substring shape filter under which `"48^3"` matched
> `"2048^3"` and silently reported the wrong row. The reported figures are after
> those fixes. The run-to-run spread on this box is **±7%, measured** by running
> two behaviourally identical binaries against each other — anything inside that
> band is reported as a tie. Detail:
> [docs/cpu_gemm_tuning.md](docs/cpu_gemm_tuning.md).

### CPU lock-free queue vs C++

20M push/pop, SPSC ring buffer, capacity 1024 (measured earlier in the project's
history and not re-run since):

| Implementation | Time | Throughput |
|---|---|---|
| Mutex `std::queue` | 1.460 s | 13.70 MOps/s |
| C++ SPSC, unaligned | 0.089 s | 225.22 MOps/s |
| C++ SPSC, cache-line aligned | 0.062 s | 321.37 MOps/s |
| Y-compiled SPSC | 0.066 s | 301.39 MOps/s |

Within 6% of hand-tuned aligned C++ without manual alignment tuning — the
compiler derived the alignment from the measured L2 cache line size.

---

## Deterministic inference

A quantized transformer whose output does not depend on who else is in the
batch. Every reduction whose order changes with batch shape is made integer, and
integer addition is associative — so tile shape, K-split, atomic completion
order and batch size cannot change the answer. That is the whole mechanism.

**And exactness is not what buys the determinism — it is what makes the
determinism affordable.** The control that settles this was built and measured:
a float32 path with the reduction *order* pinned instead of the arithmetic made
exact is **also 0/16**. It just costs **6.85x**, where the exact path costs
1.03x. The reason is the one that generalises past this implementation:
`torch.compile` takes the exact arm from 54.7 to 238.5 tok/s/seq and the
fixed-order arm from 37.7 to 36.0 — pinning an order *is* a constraint on how
the reduction may execute, so it forbids exactly the fusions that make float
fast. Integer accumulation needs no pinning, so the compiler stays free.

**This is unfinished work, and it is here because the finished parts are
measured.** What exists is a PyTorch + Triton **prototype** carrying the whole
pipeline, plus one kernel the compiler itself emits (exact attention) and its
integer `exp2`. The determinism and accuracy numbers below are the prototype's.
Replacing the rest of the prototype with compiler output is the remaining work,
not a detail.

**`0/16` is also what a broken arm scores**, and that is not hypothetical: the
fixed-order control's first version returned its tensor in the wrong layout, did
not crash, emitted fluent-looking garbage, and got a number reported for it. An
arm broken *consistently* would have scored the winning 0/16. Every arm in every
harness now passes a sanity gate before it is counted — a factual-prompt canary
plus a degeneracy check on its own generation — and the gate's logic is
mutation-verified on CPU (`python3 tools/exact_ragged_batch.py --check-gate`).
The throughput tool is the sharpest case: it never decodes a token, so a broken
arm was invisible there, and several ways of being broken are *faster*.

Model: Qwen2.5-0.5B-Instruct · RTX 4070 Ti SUPER · measured 2026-08-18.

| | result | control |
|---|---|---|
| **Determinism** | **0 / 16** batch compositions changed the output | stock bf16 changed on **16 / 16** |
| **Accuracy vs bf16** | **+0.12%** wikitext-2 perplexity | 4-bit arm: +189% |
| **Task accuracy** | statistical null | net +40 of 3,000 items |
| **Decode speed** | 1.09x slower wall clock | **0.97x device time** — the gap is launch overhead |

**The control had to be earned.** At 24 generated tokens the stock arm was
invariant *too*, because a bf16 reduction-order delta needs room to flip a greedy
argmax. The test runs 160 tokens on prompts chosen for a small top-2 margin. A
test both arms pass is measuring the harness.

**The textbook lever was the wrong lever.** The remaining 2.2% perplexity gap was
labelled "the linears", and the standard fix is group-wise weight scales.
Measuring the ceiling first — fake quantization in fp32, no kernel — showed
weight quantization costs **−0.12%**, i.e. nothing. The whole gap was
*activations*: group scales recover 6.2% of it, one extra bit of activation width
recovers 102%. And the winning fix does not fight the invariant, where group
scales would have — they recombine partial sums with *float* weights, which is
the exact thing being prevented.

**You can check most of this without a GPU.** `python3 tools/exact_selftest.py`
runs 9 of its 13 checks on CPU — including batch invariance itself — and names
the four it skips (three launch Triton, one asserts a CUDA-specific `_int_mm`
refusal). It used to print `SKIP: no CUDA` and exit 0 having checked nothing.
`python3 tools/exact_bounds_check.py` needs only z3 and runs all 22 of its
checks on CPU.

**The exactness argument is machine-checked.** `tools/exact_bounds_check.py`
puts every bound to Z3 and then exhausts the real selectors rather than
transcriptions of them. Checking the conjunction rather than each bound alone
showed one bound *subsumes* another and surfaced an unwritten hard context
ceiling (264,208 tokens at V=127). It found two live bugs doing it.

**The compiler's kernel is checked against the torch path on real
activations** — `tools/ptx_bridge.py` loads the emitted PTX through the CUDA
driver and runs it on post-RoPE Q/K/V captured from the model: 12/12 layer/head
pairs bit-identical, at `max(p)/mean(p) = 110.9` where a uniform softmax would
be 1.0. That second number is the control, and it is not decorative — an earlier
run of this bridge passed 12/12 on a temperature 65,536x too small, which is to
say on uniform attention, and the control reproduces that failure exactly when
the bug is put back.

**The compiler emits the attention kernel itself** — `Y --emit-attention-ptx
<head_dim> <seq_len>`, in `src/exact_attention.rs`, with an architecture-
independent integer `exp2` (`src/fixed_exp.rs`, 0.908 ulp proved exhaustively)
so the result is identical across GPU *architectures*, not merely across launches
of one.

### Scope: "deterministic", not "exact"

RoPE's sine and cosine, the residual adds and the softmax exponential are
approximate — deterministic, but not exact. Exactness is applied to the
reductions whose order moves with batch shape, and nowhere else, because nowhere
else needs it. Switching cache implementation moves the output by 5.7e-6 once
and is stable after: batch invariance survives that, but *cache-implementation*
invariance is a different claim and is not made.

### What this does not yet claim

- **One model, one GPU, one stack — but the exp is now measured across two
  architectures.** Cross-hardware bit-identity is the strongest thing exactness
  buys. For the integer `exp2` it is no longer an argument:

      device == host on all 1,966,085 arguments, bit for bit
      846,328 distinct results, so the agreement is not over a constant

  That is every argument the function can be reached with, x86-64 against
  sm_89 — two instruction sets from different vendors, which is a wider gap
  than two NVIDIA cards. The control is in the same line: a degenerate kernel
  agreeing over one value would report 1 distinct result, not 846,328. And the
  same run measures why the float path cannot make the claim: the device's
  `ex2.approx.f32` disagrees with the host's `exp2` on **46,301 of 100,427**
  arguments, worst gap 32 ulp.

  The premise behind it is checked statically too: disassembling the emitted
  cubin, the integer exp compiles to **no `MUFU` and no floating-point
  instruction** — 72 instructions of IMAD/SHF/IADD3/ISETP, all exactly specified
  by the ISA — while the `ex2.approx.f32` probe beside it compiles to a `MUFU`,
  which the ISA specifies by tolerance rather than by value.

  **This covers one kernel, not the pipeline.** The end-to-end claim — the whole
  decode path producing identical tokens on different hardware — still needs a
  second card, and remains the headline with no measurement behind it.
- **CUDA graphs are worth up to 1.47x and are blocked.** The prerequisites are
  cheap — a static cache costs 5–8% and improves the exact/stock ratio to 1.01x
  — and the property survives them. Capture then segfaults on *both* arms, from
  an in-place mutation inside the captured region in this transformers/torch
  pair. A library bug rather than an exactness problem, and its proper home is a
  serving integration rather than this prototype.

### What is being worked on now

Ranked, and the first item is the one that could invalidate the framing rather
than extend it:

1. **Run a second GPU.** Cross-hardware bit-identity is the strongest thing
   exactness buys and the only headline claim with no evidence behind it.
3. **Move the pipeline off the prototype** onto compiler-emitted kernels, and
   into a serving integration — which is also where CUDA graphs stop being
   blocked.

Findings 05–08 in the write-up are the result of turning this same process on
the tooling: four rounds that found the emitted kernel enforced none of its own
bounds, a differential whose two arms shared a wrong constant, two replicas
checked against themselves, and a checker asserting a copy of the rule it was
checking. None of those changed a number above; all four changed what the
numbers are worth.

**A note on how the caveats above were found.** Five of the nine findings in the
write-up are not about the kernels at all — they are about the tooling that
measures them: a differential whose two arms shared a wrong constant and so
agreed perfectly on a uniform softmax; two implementations checked against
transcriptions of themselves rather than against the compiler; a checker
asserting a copy of the rule it was checking; an acceptance harness that exited
0 on any machine without a GPU; and a coverage claim that turned out to rest on
a code path the test could not fail on. None of them moved a number. All of them
changed what the numbers are worth, which is the reason they are written down at
the same length as the results.

Full write-ups: [bit-identical decode](docs/bit_identical_decode.md) (findings,
controls, and the bugs found by turning this process on the tooling itself)
and [deterministic inference](docs/deterministic_inference.md) (the design).

---

## Safety and verification

These are the features the compiler refuses to compile without, and each one is
here because the previous version of it silently passed.

### `@safe` blocks and Z3-discharged invariants

Code inside `@safe { }` must initialize all variables, cannot dereference raw
pointers, and requires an `@invariant` on every loop.

```
fn main() {
    @safe {
        let x: I32 = 10;

        @invariant(x >= 0)
        while x > 0 {
            x = x - 1;
        }
    }
}
```

Every `@invariant` is discharged by Z3. **An invariant that cannot be checked
now fails the build.** This is a deliberate reversal: the four solver call sites
used to print a warning and continue, so on any machine without z3 — the default
— every invariant was accepted unchecked, and `@invariant(i > 1000)` on a `0..10`
loop compiled cleanly and printed "Compilation Successful!".
`Y_ALLOW_UNVERIFIED_INVARIANTS=1` restores the old behaviour loudly. The solver
is looked for at `Y_Z3_PATH`, on `PATH`, and at `venv/bin/z3`, `.venv/bin/z3`,
`z3/build/z3`, `$HOME/.local/bin/z3`.

The SMT encoding itself was unsound until recently. `trace_body_statements`
ended in `_ => {}`, so a statement it did not model was skipped — and dropping a
body's effects makes the preservation obligation strictly *easier*. The
identical violation was rejected when written plainly (`i = i - 100;`) and
**accepted** when wrapped in a trivially-true `if`. Branches are havoc'd now, and
unmodellable constructs are refused by name.

### `@ZeroDrift` — exact, order-independent accumulation

Floating-point addition is not associative, so a reduction's result depends on
the order the hardware happened to combine it in. On a GPU that order is decided
by launch geometry, which means retuning a tile size can change the answer.

```
@bounds(min=0, max=1000)
@ZeroDrift
let acc: F32 = 0.0;

acc += x;          // scaled once, then accumulated with exact integer adds
```

**Only integer and fixed-point arithmetic is drift-free.** `f64` is the same
non-associative arithmetic with a longer mantissa — it drifts less and still
drifts — so it is never selected, however fast it measures.

**Which representation is chosen is measured, not assumed.** The compiler times a
serially dependent accumulate chain per candidate on the GPU actually present:

```
      ->  KahanF32:      1455 ps/acc  not exact (never selected)
      ->    Q16.16:      1790 ps/acc  exact
      ->    Q32.32:      1922 ps/acc  exact
      ->       I64:      2106 ps/acc  exact
      ->       F64:     17726 ps/acc  not exact (never selected)
      -> @ZeroDrift acc: F32 -> Q32.32 (measured 1922 ps/acc)
```

Exact fixed-point is ~9x cheaper here than reaching for higher precision, and the
GeForce FP64 penalty lands exactly where it should.

`tests/zero_drift_end_to_end.rs` compiles a program with clang and **runs it**,
summing the same 4001 terms in opposite orders and requiring bit-identical
results — alongside a control asserting that sequence genuinely disagrees with
itself in `f32`, so the result cannot be vacuous.

`@ZeroDrift` used to do literally nothing: it lexed, parsed, was counted, was
printed as an advisory, and was read by no backend. Output was byte-identical
with and without it.

### Linear tracking of async tokens

Async memory tokens must be consumed exactly once. "Exactly once" is a claim
about *executions*, and the tracker used to check source lines — so
`if n { pipe.wait(t); }` (awaited on one path of two) and
`for i in 0..4 { pipe.wait(t); }` (one copy, four awaits) both compiled clean.
The tracker now records conditional and loop nesting depth at creation and
compares it at consumption. `tests/linear_tracker_enforcement.rs` is 10 tests,
6 negative and 4 positive — rejecting every `pipe.wait` under a loop would be
sound and would also ban the shape every real pipelined kernel is built from.

### Proof-carrying kernels: the exact GEMM, verified end to end

`src/cpu_gemm.rs` emits a tiled, packed, K-split, multi-threaded AVX-512
`vpdpwssd` GEMM. It is **bit-identical** to the naive triple loop it replaces,
and that is a theorem rather than a test result:

```coq
(* proofs/ExactGemmWhole.v *)
Theorem the_threaded_gemm_holds_the_source_dot_products :
  forall A B M N K Fl m nthr r c,
    (0 < Fl)%nat -> 0 <= m ->
    2 * Z.of_nat Fl * m * m <= ExactGemmMicro.I32MAX ->
    (forall idx k, Z.abs (A idx k) <= m) ->
    (forall k idx, Z.abs (B k idx) <= m) ->
    (0 < nthr)%nat -> (r < M)%nat -> (c < N)%nat ->
    thread_sum A B M N K Fl nthr r c nthr
    = PK.sum_k (fun k => A r k * B k c) K.
```

Sum the partials of `nthr` threads, each handed a K band, each running the
emitted driver — packing, the register tile's routing, the k-pair loop, the
int32 flush, the output tiling and the fold-back — and every position of C is
exactly the source matrices' dot product. **No hypothesis that `MR` divides
`M`, that `NR` divides `N`, that `nthr` divides `K`, or that `K` is even.**

**This is only possible because the kernel is exact.** Floating-point addition
is not associative, so a tiled f32 reduction provably does *not* equal the naive
one — `GemmBandSplit.v` refutes it at `K = 201`, `nthr = 2`, where the rounded
version answers 1100 against its own reference's 1000. Integer addition *is*
associative, so the relationship is an equality, which is what a proof assistant
is good at. What exactness costs is measured below, and it is not free.

Twenty files, 416 theorems and lemmas, **no axioms, nothing admitted** — and
`tests/proofs_are_checked.rs` runs `coqc` over all of them in `cargo test`,
with a content control per file so that "it compiles" and "no axioms" (both
properties an *empty* file has) are not the whole check.

#### What the verified kernel costs

The proofs say the kernel is bit-identical to the naive nest. They say nothing
about whether it is *fast*, so that is a separate measurement — three arms, one
shape per process per arm, interleaved, minimum over rounds:

```bash
python3 tools/exact_gemm_bench/run.py            # the table below
python3 tools/exact_gemm_bench/run.py --scaling  # thread scaling
python3 tools/exact_gemm_bench/run.py --isa      # what each datapath can issue
```

**The two datapaths do not have the same ceiling, and that has to be measured
before any ratio means anything.** `vpdpwssd` retires 32 int16 multiply-accumulates
where `vfmadd132ps` retires 16 f32 ones. On one core they issue at the *same*
rate — **11.18 against 11.21 G instructions/s, ratio 1.00** — so the exact
datapath's MAC ceiling is exactly **double** the f32 one: 357.9 against 179.3
G MAC/s per core, and 5212 against 2606 across sixteen at the 5.09 GHz all-core
clock. The probe is verified non-foldable two ways: the disassembly must contain
the instruction, and doubling the iteration count must double the time.

| 16 threads | Y exact | Y f32 | OpenBLAS f32 | exact / Y f32 | exact / OpenBLAS |
|---|---|---|---|---|---|
| 512³ | 370 | 763 | 780 | 0.48x | 0.47x |
| 1024³ | 708 | 1105 | 1319 | 0.64x | 0.54x |
| 2048³ | 1057 | 982 | 1618 | 1.08x | 0.65x |
| 4096³ | **1570** | 1072 | 1648 | **1.47x** | **0.95x** |

G MAC/s, the unit the two datapaths share. OpenBLAS is the `scipy-openblas`
0.3.33 numpy ships — `DYNAMIC_ARCH`, dispatching to its `SkylakeX` AVX-512
kernel here, which the harness prints, because this repository has previously
used a distro OpenBLAS with **zero `zmm` instructions** as a baseline. Both
arms are seeded identically and the f32 checksum equals the exact integer one
at these magnitudes, so the two are demonstrably computing the same GEMM.

**Read the spread before reading a row.** Over three runs the exact column holds
to ±5% and OpenBLAS to ±10%, but **Y's own f32 arm swings by up to 1.7x at 512³
and 1024³** — that kernel takes its thread count from the caller's call
*frequency*, so it is regime-sensitive in exactly the way documented under
[CPU: AVX-512 GEMM](#cpu-avx-512-gemm). Read the `exact / Y f32` column at those
two shapes as "well below 1", not as a number.

**Two things are true at once, and the second says where the work is.** In wall
clock the exact kernel reaches **0.95x OpenBLAS f32 at 4096³** and 0.47–0.65x
below that. In ISA efficiency it is at **30% of its own ceiling** where OpenBLAS
is at **63%** of its — so the near-parity at 4096³ is bought with an instruction
that is twice as strong, and roughly 2x of headroom is still on the table.

**Exactness is not free the way an earlier version of this README said.** That
claim — "exact VNNI is 1.88× faster than the f32 path" — came from a *single-core
micro-kernel* probe, and `docs/proof_carrying_kernels.md` had already corrected
it to 1.10–1.15x for the micro-kernel without the correction reaching here.
Against Y's own f32 GEMM through the same emitter the exact kernel does not
cross over until about 2048³. Exactness trades range *and*, below a few
thousand, speed.

#### Cutting the right axis was worth more than any of it

The wrapper used to split **K**. That is a reduction, so every thread needs a
private `M × N` copy of C to sum out of, and the bookkeeping is `O(T·M·N)`
against `O(M·N·K)` of work — measured at 1024³ on 8 threads, **9.311 ms of
buffer-and-reduce against 7.344 ms of compute**, so eight threads ran *slower
than one*. Splitting **M** is a partition instead: disjoint row bands, no private
buffer, no zero-fill, nothing to reduce. `ExactGemmTiling.c_written_exactly_once`
had already proved the output tiling is one, so the fix needed no new proof —
**the kernel had been cutting the one axis that needs an arithmetic property,
while the axis that needs none was already covered.**

| threads | 1024³ | 4096³ |
|---|---|---|
| 1 | 152 G MAC/s | 158 G MAC/s |
| 2 | 269 (1.77x) | 355 (2.25x) |
| 4 | 458 (3.02x) | 649 (4.12x) |
| 8 | 650 (4.28x) | 1069 (6.79x) |
| 16 | **711 (4.68x)** | **1570 (9.97x)** |
| 32 | 649 (4.28x) | 1296 (8.23x) |

Bit-identical at every row and every thread count, which is the only acceptable
evidence for changing a schedule under a certificate that claims exactness
(`tests/exact_gemm_msplit.rs` compares whole output buffers *and* checks each
against an independent integer reference, so two schedules cannot be wrong
together). Thirty-two threads is never best: this is a 16-core part and the
kernel gets nothing from SMT.

**`ExactGemmMSplit.msplit_needs_no_algebra` is the part that generalises.** The
row split is exact for an *arbitrary* accumulate — not associative, not
commutative, not exact — because a row's accumulator is never split, so there is
no re-bracketing to justify. Exact accumulation is what makes a *reduction*-shaped
parallelisation provable; a *partition*-shaped one is provable without it.
**Exactness buys the axes you could not otherwise cut; it is not the price of
cutting anything.**

#### The proof is tied to the emitted code by removing the second description

The usual way to connect a proof to a compiler is to give the IR a semantics and
prove the emitter refines it. That is a multi-year project of its own. Instead,
**there is one description and both consumers are rendered from it** — an `Ix`
expression tree and a `CountedLoop` iteration space that render to LLVM, to Coq,
and to values for tests. `proofs/ExactGemmSchedule.v` is *generated* from
`src/cpu_gemm.rs`'s own constants, and a byte-identity gate fails the build on
any divergence.

Every extraction step left all four emitted LLVM modules **byte-for-byte
unchanged**, which is the argument that the description is what the compiler was
already doing.

#### A compilation emits its own certificate

```
$ Y gemm.ysu --emit-llvm
  -> @ZeroDrift matmul MxN: EXACT vpdpwssd kernel substituted (|x| <= 1024, flush every 64 k-pairs)
  -> Written to: gemm.ll
  -> Certificate: gemm_certificate.v (check with `coqc -Q proofs "" gemm_certificate.v`)
```

The `.v` instantiates the theorem above at *this* compilation's flush interval
and operand bound. It is not paperwork: the one hypothesis that depends on the
program is the overflow licence `2·Fl·m² ≤ i32::MAX`, and the compiler decides
it in **floating point** — a `sqrt` and a `floor` on `f64`. The certificate
states it over `Z` and hands it to `coqc`, which has no floats. Two derivations
of one obligation, by two tools, gated against each other at the boundary, which
is one unit wide: accepted at `m = 4095`, refused at `m = 4096`.

#### What the proofs find that running the code does not

Every kind of check here is blind to something:

| bug | correctness suites | schedule gate | Coq |
|---|---|---|---|
| tile count over-allocates a buffer | **pass** | **FAIL** | pass |
| `min(a,b)` emitted as `min(b,a)` — same value | **pass** | **FAIL** | pass |
| scratch-zeroing loop one trip long | **pass** | pass | **FAIL** |
| fold-back loop unclamped (writes past C) | 2 of 4 **pass** | **FAIL** | pass |

The last row is the one to read twice. An unclamped fold-back is a genuine
out-of-bounds write past the last row of C, and it is **not observed** by two of
the four correctness suites — including the one running `M = 53` against a
6-row tile, the most ragged shape in the repo. The overrun lands in memory the
process already owns and the answer stays correct.

Writing the proofs also found a **live heap overflow**. Formalising the output
tiling forced the hypothesis `N <= ldc` to be *stated*; nothing in the compiler
stated it, every caller happened to pass `ldc = N`, and calling the kernel with
a padded C corrupted the heap from three separate sites. The proof did not
discharge the bug — it surfaced the unwritten precondition, and the test written
from that precondition found it.

The method is written up in
**[The process: taking a kernel from *fast* to *verified*](docs/verified_kernel_process.md)**.

### Translation validation: removing `ptxas` from the trusted base

Those proofs stop at the IR. The trust boundary printed into every emitted
certificate says so, and names the remedy in the same breath — *"translation
validation — checking THIS object against THIS IR per compilation, which is not
performed."*

`tools/ptxas_tval/` performs it, on the GPU side. It symbolically executes a
kernel's PTX and the SASS `ptxas` produced from that exact file, and asks z3
whether the two can ever store different values. **An opcode neither executor
models is a hard error, never a guess.**

| kernel | verdict | obligations | time | |
|---|---|---|---|---|
| `fma/rn` | **VALIDATED** | 9 | 0.0 s | float, contraction forbidden by `.rn` |
| `fma/plain` | UNPROVED | 10 | 0.0 s | **the control** — `store 0: sat` |
| `bn254_permute` | **VALIDATED** | 30 | 0.2 s | branching `ptxas` invented |
| `bn254_sub_vec` | **VALIDATED** | 88 | 9.4 s | |
| `ptx_carry_chain` | **VALIDATED** | 123 | 24.6 s | 24 predicated instructions |
| `exact_pv` @ `-O1` | **VALIDATED** | 14 | 1.0 s | across a **loop** |
| `smem_roundtrip` | **VALIDATED** | 18 | 0.2 s | **shared memory**, 1 barrier |

Six kernels, 282 obligations. The `plain` row is what the rest are worth: it is
the same kernel as `rn` without the `.rn` suffixes, `ptxas` contracts
`mul.f32`+`add.f32` into one `FFMA` that rounds once where PTX rounds twice, and
the validator answers `sat` with a counterexample. That is a freedom the ISA
grants, not a `ptxas` bug — and a validator that always said VALIDATED would
report every other row identically.

**The binding constraint is the solver, not opcode coverage**, and measuring that
cancelled the feature the measurement was taken to justify. `ptx_carry_chain`
validates with 29 multiplies in 24 s; `bn254_fr_mul_fast` has 65 and is UNPROVED
after 9,705 s — with 261 of 276 cut points closed and **no `sat`**. Asking the
counterfactual (*if every opcode were modelled, what could the solver close?*)
puts **52 of 67 kernels under that wall** using barriers as cut points — and
inverts the ranking: the 23 tensor-core GEMMs look deepest and are tractable at
39–61 multiplies per barrier region, while the field kernels look shallow and run
244–717. "33 kernels are behind shared memory" was false; shared memory alone
unlocks exactly one, a test fixture. It was still right to build, because it is
what *creates* the cut points the GEMMs need.

Also measured, and reported separately because they are different claims:
`rcp.approx` differs from `rcp.rn` on **13.23%** of inputs on the device and
`div.approx` from `div.rn` on **27.30%** — so identifying a PTX macro-op with the
MUFU that seeds it, the cheap way to "support floats", would validate 17 kernels
for the wrong reason. And `ptxas` implements 32-bit `div.u32`/`rem.u32` through
the *float* unit (`I2F.U32.RP`, `MUFU.RCP`, `F2I.TRUNC`).

Nothing here is CI-gated: it needs the CUDA toolkit, z3 and minutes to hours per
kernel. It covers one compilation of one kernel at one architecture and
optimisation level — which is the point of the technique and also its limit. The
CPU-side trust item (`clang`, its optimiser, the assembler and the linker) stays
open. Full write-up, including what is *not* claimed:
**[Translation validation for `ptxas`](docs/ptxas_translation_validation.md)**.

### A machine-checked proof of the ZK control-flow lowering

`proofs/ZkControlFlow.v` formalises the `return` / `if` / sequencing fragment of
the ZK backend: an operational semantics for the language, three candidate
lowerings, and the theorem that the shipped one agrees with the semantics **on
every program and every environment**.

```
Theorem low_correct : forall s e, bools e s -> agrees e s.
```

Reproduce with `coqc proofs/ZkControlFlow.v && coqchk -o proofs/ZkControlFlow.vo`
— Rocq 9.1.1, 0.3 s to compile, and the kernel checker reports:

```
* Axioms: <none>
* Constants/Inductives relying on type-in-type: <none>
* Constants/Inductives relying on unsafe (co)fixpoints: <none>
* Inductives whose positivity is assumed: <none>
```

Nothing admitted, nothing assumed. **Writing it down is what found the last
bug** — a one-sided `return` inside a *nested* `if` reported its own empty
tail's zero, so `if c { if d { return 1; } } return 7;` emitted a circuit
computing 0. Z3 on the emitted artifact had missed it, a fresh test file had
missed it, and so had the fix that was supposed to close the family. The flat
case is the one everyone tests.

The proof also states what it does **not** cover, in the file: it is a model
rather than `zk_emitter.rs` (no extraction, no refinement proof — the tests are
what tie them together); the field is a commutative ring taken as `Z`, so
nothing about modular range transfers; and constraint emission is modelled as
evaluation, so it says nothing about constraint *count* or about what an
adversarial prover could satisfy.

### Generative differential fuzzing

`src/zk_fuzz.rs` generates well-formed programs from a grammar — 100% parse rate
by construction — and checks each against three oracles. It found two bugs in
two different subsystems on its first run.

The eight `cargo fuzz` targets in `fuzz/` are a useful counter-example and are
kept as one: none has ever been run here (`cargo-fuzz` is not installed and
there is no nightly toolchain), and none *could* have found these bugs. Two feed
raw bytes to the lexer, so the chance of forming a program with an `if` and a
`return` in it is negligible; the "differential" one compared nothing, merely
asserting an output string was non-empty; and the soundness one reported every
finding with `eprintln!` and never panicked, so libFuzzer could not see a
failure — a week-long run over a backend computing the wrong function would have
exited 0.

| oracle | what it is | why it is there |
|---|---|---|
| Independent interpreter | written against the fuzzer's own IR, sharing no code with the compiler path | walking the parser's AST would have hidden the parser bug it found |
| **Metamorphic fold** | the same program rendered twice — inputs as parameters, then as literals — must agree | needs no reference implementation, so it **cannot be wrong in the same direction as the model** |
| Parse-failure grading | parse failures counted separately from semantic refusals | collapsing them hid an over-refusal, and reverting the parser fix passed the whole sweep |

400 programs per `cargo test` run, 20,000 in the extended sweep
(`extended_sweep --ignored`). The corpus is checked for **coverage** rather than
assumed to have it — `the_generated_corpus_is_not_vacuous` asserts how many
generated programs actually contain an `if`, a loop, an ordering comparison and
a compound assignment, because a generator restriction made to simplify an
oracle once silently deleted two bug classes. Counterexamples are minimised by
delta debugging **over the IR**, not the text, so every candidate stays a
well-formed program; that took a 40-line counterexample down to four lines and
turned a guess into an attribution.

### Mutation testing, as a standing practice

Every gate in this repository is expected to fail when the mechanism it guards
is removed, and the ones that did not are recorded rather than quietly fixed.
Three of eight mutations survived one fresh, all-green test file; a device test
for a shared-memory barrier passed with `bar.sync` deleted, because the race
never fired; a `u32`-to-`u32` differential passed with the conversion replaced by
the identity. Mutations that survive are then sorted into **test holes** (fix the
test) and **confirmations** (the guard is genuinely redundant with a rule
enforced earlier — keep it, and say so), which is a distinction that only shows
up if you look.

### A design rule the repository enforces

**In any pass whose output is a correctness claim, an unhandled AST node is a
hard error — never a silent identity, no-op, or "close enough" substitution.**
This is written down because the same bug has now been found **39 times** — a
`_ =>` arm that guessed instead of refusing, or, in the later cases, a correct
guard consulted at a subset of the sites where its property has to hold. A pass
that silently approximates produces the paperwork of a proof without the proof,
and the build goes green. The full table — site, silent fallback, consequence —
is maintained in the project's engineering notes, which live outside this
repository.

The instances span every layer: a comparison lowered to the wrong operator in the
ZK backend (`5 <= 5` was false, and Groth16 proved it anyway); an SMT encoder
that translated `x & y` as `x + y`; a `while` loop that emitted no PTX at all;
an ELF emitter in which every identifier read the first local; and a Coq proof
of the ZK control-flow lowering that found a bug three rounds of testing had
missed. Reading it as a list of past mistakes is the wrong reading — it is a
list of **shapes to grep for in the next pass you write.**

---

## Building

Requires: Rust toolchain, clang.

```bash
cargo build --release
cargo build --release --features zk     # ZK backend is NOT in a default build

cargo test --release                    # ~605 tests
cargo test --release --features zk      # ~855 tests, ZK included
cargo test --release -p y-gpu           # the sibling crate; a bare `cargo test`
                                        # builds the root package ONLY and does
                                        # not run these 8
```

**Four gates are conditional on an external tool, and a missing tool makes them
SKIP AND REPORT `ok`.** A green run is therefore not by itself evidence that
they ran — read this list before trusting one:

| tool | gate | what it is the only check for |
|---|---|---|
| `z3` | `safe_invariant_enforcement` | that `@safe`'s `@invariant` is discharged at all rather than assumed |
| `ptxas` | `ptx_portability`, `ptx_intrinsics_assemble`, `coprocessor_ptx_assembles` | that emitted PTX is legal, at architectures this machine does not have |
| `solc` + Node (`npm install solc`) | `zk_solidity_verifier` | that the generated Groth16 verifier accepts a real proof on a real EVM |
| `circom` | `circom_frontend`, `tools/circomlib_coverage.py` | that Y agrees with the reference compiler |
| `rustc` | `cpu_emitter_output_compiles` | that the Rust `--emit-cpu` prints is Rust |

That list is here because the third one had been skipping. Installing `solc`
made it run — and with the G2 coordinate order reverted to `(c0, c1)`, the bug
the module comment in `src/zk_solidity.rs` exists to prevent, it **failed
immediately**. Before the install, the same mutation passed the whole suite in
0.00 s under a green `ok`. `nvcc` is optional and gates only the GPU probe, not
a correctness claim.

### The whole command-line surface

Every flag the binary accepts, so that none of it has to be discovered by
reading `main.rs`. `--target=<x>` is accepted as a synonym of `--emit-<x>`
throughout.

| flag | what it does | state |
|---|---|---|
| *(none)* | LLVM IR → native binary via `clang` | the default backend |
| `--emit-llvm` | LLVM IR | real |
| `--emit-ptx` | NVIDIA PTX | real |
| `--emit-native` | standalone x86-64 ELF | **straight-line integer subset only**; refuses the rest by name |
| `--emit-cpu` | prints **scalar host Rust** source **for you to paste** — Y never compiles it | real, but not a build step; gated on `rustc` accepting what it prints, verbatim. **It emits no SIMD**: measured, 0 of 46 corpus blobs contain a vector intrinsic, vector type or `target_feature` |
| `--emit-attention-ptx <head_dim> <seq_len>` | the exact-attention kernel, to stdout | real; both positional arguments are required and refused by name if absent |
| `--emit-coprocessor` | RT + Tensor Core fused schedule | **a scheduling simulation** — see "What is real". It writes a complete module whose `.version`/`.target` are gated against the PTX backend's |
| `--emit-c`, `--c`, `--target=c` | removed; reports so and exits 1 | gone |
| `--target=r1cs` / `--emit-r1cs` | R1CS `.r1cs` / `.sym` / `.r1cs.txt` | real, needs `--features zk` |
| `--witness <in.json>` | also solve and write `.wtns` (iden3 format) | real |
| `--emit-verifier <vkey.json>` | Groth16 Solidity verifier | real; `--name <N>` sets the contract name |
| `--emit-zk-ptx` | GPU witness-generator PTX | **lowers 5 of `WitnessOp`'s 17 variants and refuses the rest by name** — the reachable subset is tiny (see below) |
| `-l`, `--link <dir>` | circom include path | real |
| `-o`, `--output <path>` | output path | real |
| `--autotune` / `--autotune-force` / `--no-autotune` | GEMM tile selection: measure / re-measure / analytic model only | real |
| `--portable` | clears the probed AVX / AVX-512 feature bits | real |

**`--emit-zk-ptx` is the one line in this table to read sceptically**, and the
previous edition of this paragraph — "nothing in the test suite runs it or checks
what it computes, treat it as unverified" — understated it. Checking found the
backend lowered **5 of `WitnessOp`'s 17 variants** and did not fail on the other
twelve:

- `Inv` / `Div` emitted `mov s_out, s_a` under a comment reading "256-bit Field
  Inversion / Division Hint" — the **identity**, so `1/x` computed `x`;
- everything else hit `_ => mov 0`, writing **zero** into the witness slot.

Both assemble, which is why the existing gate could not see either: `ptxas`
cannot tell a wrong `mov` from a right one. Which operations landed in the zero
arm is what makes it severe rather than partial — `BitOfLc` is how every
comparison, bitwise operator, shift, integer division and range check gets its
witness; `IsZeroLc`/`InvOrZeroLc` are `==` and `!=`; `MulAddLc` is the most
common statement a circom program lowers to. And `MulLc` covers the binding of
any linear expression, so **`return a + b;` was zeroed too** — the flag printed
"compiled successfully" for essentially every circuit while filling most of the
witness with zeros.

It now **refuses by name and writes no file**, and `tests/zk_ptx_witness_refuses.rs`
gates that in both directions: six constructs must be refused naming the
`WitnessOp`, the two that do lower must still emit, and what they emit must
assemble at the target it declares. Use `--target=r1cs --witness <in.json>` —
the CPU solver handles all seventeen and is checked against circom's own witness
calculator element for element.

Empirical GEMM autotuning for `@tile`d kernels measures candidates on the real
GPU and caches per (M, N, K, precision, GPU) in `.ysu_hw_profile`. A cold shape
costs ~4 s (~100 s at M=N=K=16384). The cache **cannot** detect that codegen
itself changed — re-tune with `--autotune-force` after editing a kernel or the
compiler will keep emitting a tile chosen for the old one.

---

## Hardware probing

On first run the compiler measures the host and caches to `.ysu_hw_profile`:
CPU cache latencies via pointer-chasing, AVX-512 throughput, thread-handoff
cost; and via an external CUDA probe, FMA/IMAD/transcendental latencies,
shared-memory bank-conflict cycles, tensor-core latencies, warp-shuffle cost and
global memory latency at several strides. Delete the file to force a re-probe
after a driver, GPU or CPU-governor change — note that this also discards
autotuning measured on the old configuration, which is the intent.

---

## Project layout

```
src/                       Rust bootstrap compiler
  lexer.rs parser.rs ast.rs        front end
  type_checker.rs                  safety blocks, Z3 invariants, interval arithmetic
  linear_tracker.rs                async token single-consumption
  sentinel.rs ysu_gpu_probe.rs     hardware probe
  autotuner.rs empirical_autotune.rs cuda_runtime.rs
  bank_conflict.rs                 shared-memory swizzle solver
  llvm_emitter.rs                  LLVM IR (default backend)
  ptx_emitter.rs                   NVIDIA PTX
  cpu_emitter.rs cpu_gemm.rs       x86-64 / AVX-512 GEMM
  native_emitter.rs                standalone ELF
  zero_drift.rs                    @ZeroDrift representation selection
  exact_gemm_certificate.rs        the .v a compilation emits with its kernel
  exact_attention.rs fixed_exp.rs  exact int8 attention PTX + integer exp2
  zk_field.rs                      BN254 Fr, Montgomery form
  zk_emitter.rs zk_witness.rs      R1CS emission and witness solving
  zk_poseidon_constants.rs         circomlib parameters (GENERATED — do not edit)
  zk_solidity.rs                   Groth16 on-chain verifier
  circom_{lexer,ast,parser,lower}.rs   circom 2.x front end
  quantization_pass.rs             FP32 -> FP16 staging conversions
  cpu_specializer.rs               CPU-side rewrites
  zk_fuzz.rs                       generative differential fuzzer (grammar + 3 oracles)
  c_api.rs                         C ABI — the crate also builds as a cdylib
  ir_grapher.rs coprocessor_scheduler.rs rt_core_emitter.rs
                                   scheduling simulation — see "What is real"
  ypm.rs ysu_gpu_probe.rs          separate `[[bin]]` targets, not modules —
                                   they build as their own executables

self_hosted/    compiler phases rewritten in Y (.ysu); not the default build path
proofs/         Rocq proofs — ExactGemmSchedule.v is GENERATED
tests/          test programs, benchmarks, PTX assembly gates
tools/          measurement and analysis harnesses (Python), run by hand
  ptxas_tval/     PTX-vs-SASS translation validator — see docs/
  exact_gemm_bench/  the three-arm GEMM benchmark this README quotes
circomlib/      vendored circomlib (upstream 2.0.5)
docs/           language spec and design notes
```

---

## Status

The Rust bootstrap compiler in `src/` is the stable reference and is what runs
today. The self-hosted compiler in `self_hosted/` is in progress and is not the
default build path.

Author-built with LLM assistance for implementation; architecture and design
decisions are the author's own.

Further reading:

- [Y Language Specification & Reference Manual](docs/y_language_documentation.md)
- [ZK compile-speed detail and measurement traps](docs/heavy_circuit_speed_test.md)
- [circom front end](docs/circom_frontend.md)
- [ZK emit profiling](docs/zk_emit_profile.md)
- [CPU GEMM tuning, the harness biases, and the two regimes a loop benchmark
  cannot distinguish](docs/cpu_gemm_tuning.md)
- [The process: taking a kernel from *fast* to *verified*](docs/verified_kernel_process.md)
- [The ZK control-flow lowering, proved in Rocq](proofs/ZkControlFlow.v)
- [Deterministic / bit-identical decode](docs/bit_identical_decode.md)
- [Deterministic inference design notes](docs/deterministic_inference.md)
- [Proof-carrying kernels](docs/proof_carrying_kernels.md)
- [RT/Tensor co-processor: why it is scaffolding](investigation_rt_tensor_coprocessor_findings.md)
- [Benchmarks index](README_BENCHMARKS.md)

Author: Umut Korkmaz (YSU)
