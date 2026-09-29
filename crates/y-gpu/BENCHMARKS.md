# Where y-gpu actually stands

Measured on one machine: RTX 4070 Ti SUPER, 32-thread CPU, CUDA 13.3.

## Against icicle, BN254 NTT — measured 2026-08-12

The NTT had **never been compared to anything**; only the MSM had. It is Y's
stronger primitive by a wide margin.

| n | Y radix-4 | Y **fused** | icicle kNN | |
|---|---|---|---|---|
| 2^20 | 0.61 ms | **0.56 ms** | 0.608 ms | **Y 1.09x faster** |
| 2^22 | 5.20 ms | **2.47 ms** | 2.834 ms | **Y 1.15x faster** |

**Y is ahead at both sizes, from 1.25–1.9x behind — against 2.8–4.8x on the
MSM.** icicle was re-measured on the same day and reproduced its earlier run to
within noise (0.606 → 0.608, 2.817 → 2.834), so the movement in the ratio is
Y's. Run-to-run spread is ±0.02 ms over six runs; read these as 0.56 and 2.5.

**Read the ordering carefully or this comparison is meaningless.** icicle's
`kNN` (natural in, natural out) is its *fast* path — `kRN` measures 7.05 ms at
2^22 against `kNN`'s 2.82, i.e. asking for bit-reversed input makes it
**slower**, because a well-built NTT pairs DIF with DIT and needs no separate
reversal at all. Y's kernel is decimation-in-time and requires bit-reversed
input, and its benchmark does that permutation **on the host, outside the timed
region**. So the table above already flatters Y: icicle's figure is a complete
natural-to-natural transform, Y's is the transform only. **The "parity at 2^20"
row is therefore not a claim of parity end to end** — it says the butterflies
cost the same and Y still owes a permutation icicle does not.

### The gap was pass count, and NCU said so

This is the diagnosis that produced the fused kernel below; it describes
`bn254_ntt4_stage`, which is still what runs the stages the fusion cannot
absorb. Three profiled launches at 2^22, all in agreement:

| metric | value |
|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | **90.5–91.0%** |
| `dram__bytes.sum.per_second` | 593–595 GB/s |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 44–46% |
| `sm__warps_active` | 28–30% (register-limited, 2 blocks/SM) |

**The kernel is DRAM-bound at 91% of peak, so there is nothing left to win
inside it.** Occupancy is a red herring here and always has been — forcing
registers down was measured strictly slower.

The traffic arithmetic locates the difference exactly. A 2^22 array is 134.2 MB;
radix-4 needs 11 passes, each reading and writing it:

- **Y**: 11 passes = 2.95 GB in 5.33 ms = 554 GB/s.
- **icicle**: 2.817 ms at the same achieved 593 GB/s implies 1.67 GB, i.e.
  **~6.2 equivalent passes.**

So icicle fuses roughly **twice as many butterfly stages per kernel launch**,
keeping intermediates in shared memory instead of returning them to DRAM. That
is the whole gap. It is structural, not tuning, and `docs/` has predicted it
since the radix-4 work landed: *"Fusing them in shared memory is the next step,
not more tuning."*

### The fused kernels, and where the bottleneck went

`tests/bn254_ntt4_fused.ysu` runs the five lowest radix-4 stages inside one
launch. A CTA owns 1024 contiguous elements (32 KB, eight u32 each), stages
them into shared memory once, runs `q = 1, 4, 16, 64, 256` there, and writes
back once. 87 registers, zero spill, one barrier per stage.

The stages above it need a **strided** view, because their quarter-stride
exceeds a contiguous slice. `bn254_ntt4_fused_high{h}` takes the same 1024
elements as `4^h` *j*-values across `U = 1024/4^h` consecutive *base* values:

```
CTA owns   sb*Q0*4^h + p0 + u + j*Q0     u in [0,U),  j in [0,4^h)
```

In *j*-space a stage of quarter `Q0 * 4^t` is an ordinary radix-4 butterfly of
quarter `4^t`, so `t = 0..h-1` fuses `h` of them. The `u` are what keeps it
coalesced — `U` consecutive threads share a `j` and so read `U` *consecutive*
elements. Without them a warp would ask for 32 separate 16-byte pieces 16 KB
apart and use half of every sector fetched.

**Depths 2, 3 and 4 are generated, and that is what takes 11 passes to 3 rather
than to 4.** `log4(N) - 5` is not a multiple of 4, so a single depth always
leaves a remainder running one launch per stage. Depth 1 is deliberately *not*
generated: fusing one stage is the unfused kernel plus a shared-memory round
trip, i.e. strictly worse.

Device time, launches only, min of 12 interleaved rounds, after a global clock
ramp (`what_stage_fusion_buys`, `--ignored`):

| n | unfused | low only | **all** | | passes | schedule |
|---|---|---|---|---|---|---|
| 2^20 | 0.61 ms | 0.56 ms | **0.56 ms** | 1.09x | 10 → 3 | low+4+1×1 |
| 2^22 | 5.20 ms | 3.91 ms | **2.47 ms** | 2.11x | 11 → 3 | low+4+2 |

The strided group is worth nothing at 2^20 and everything at 2^22, and the
reason is that a 2^20 array is 33.5 MB against this card's 48 MB L2 — it was
never DRAM-bound, so removing passes removes nothing.

**The pass ratio predicted 3.7x at 2^22 and 1.96x arrived, because two of the
three kernels stopped being DRAM-bound.** Where each one now sits, profiled at
2^22:

| kernel | stages | time | DRAM | SM | bank conflicts |
|---|---|---|---|---|---|
| `bn254_ntt4_fused` | 5 | 1.05 ms | 39% | **71%** | 0.95% |
| `bn254_ntt4_fused_high4` | 4 | 1.03 ms | 41% | **68%** | 1.75% |
| `bn254_ntt4_fused_high2` | 2 | 0.65 ms | **92%** | 55% | 2.25% |

The XOR swizzle in `gen_bn254_kernels.py::swizzle` is doing its job in all
three. The first two move one pass of traffic in ~1 ms where bandwidth alone
says 0.45 — removing the memory bound simply exposed the arithmetic that had
been hiding behind it. That is the honest reading of 1.96x rather than 3.7x:
the fusion did exactly what it claimed to the traffic.

The SASS says what that arithmetic is:

| SASS op | count | |
|---|---|---|
| IADD3 / IADD3.X | 4,557 | carry propagation |
| **IMAD.WIDE.U32** | **2,400** | the actual limb products |
| IMAD.MOV.U32 / MOV | 2,638 | register moves |
| IMAD.X | 1,710 | carry propagation |
| LOP3.LUT | 1,308 | limb extraction |

**The multiplies look like 18% of the instruction stream.** The rest is
Montgomery's carry chain done the long way: `mul_wide_u32(a, b) + t + c` becomes
an `IMAD.WIDE` plus a 64-bit add plus a shift-and-truncate.

That reading is what motivated the carry-flag work below — and it is also what
made its payoff look bigger than it is, because an `IMAD.WIDE` is worth two
ordinary multiplies. See "predicted 2x, measured 1.06x".

One arithmetic saving is already taken: **stage 0 has a single twiddle
position and `w^0 = 1`**, so three of its four CIOS multiplies were computing
`x * R * R^-1`. Dropping them is exact, and worth 1.18 → 1.06 ms and 108 → 87
registers. Note that the same saving in the *unfused* kernel would have bought
nothing at all — that kernel is DRAM-bound at 91%. **It became worth doing
because the bottleneck moved.**

### Carry-flag intrinsics: predicted 2x, measured 1.06x

`add.cc` / `addc.cc` / `mad.lo.cc` / `madc.hi.cc` are in the backend now
(`carry_op` in `ptx_emitter.rs`, gated by `tests/ptx_carry_chain.rs`), and the
whole BN254 kernel family is generated on top of them — `CARRY` in
`tools/gen_bn254_kernels.py` switches between the two formulations so the
comparison is the same kernel measured both ways.

**It is worth 1.06x on the NTT at 2^22 (2.63 → 2.47 ms) and about 1.02x on the
MSM (kernel-only 279.6 → 273.8 ms at 2^22). The 2x this section previously
predicted was wrong, and the SASS says exactly why.**

| SASS op | u64 accumulate | carry chains |
|---|---|---|
| `IMAD.WIDE.U32` | 2,400 | 0 |
| `IMAD` + `IMAD.HI.U32` | 0 | **4,508** |
| `IADD3` / `IADD3.X` | 4,557 | 5,203 |
| `IMAD.X` | 1,710 | 274 |
| `IMAD.MOV.U32` / `MOV` | 2,638 | 274 |
| `LOP3.LUT` | 1,308 | 484 |
| **total** | **11,728** | **11,184** |

The carry bookkeeping did collapse — moves 2,638 → 274, `LOP3` 1,308 → 484,
`IMAD.X` 1,710 → 274. But **`IMAD.WIDE` produces both halves of a 32×32 product
in one instruction, and the two-pass carry form cannot**: it needs a separate
`IMAD` for the low half and an `IMAD.HI` for the high one. So the multiply
count doubles just as the bookkeeping falls, and the totals nearly cancel —
4.6% fewer instructions, 6% less time.

The reasoning that produced the 2x was "the multiplies are only 18% of the
stream, so the other 82% is waste." The flaw is that the 18% was counted in
`IMAD.WIDE`s, which are worth two ordinary multiplies each. **Count the work an
instruction does, not the instructions.**

It is kept because 1.06x is real and it is not the only thing it buys: the
generated `.ysu` shrinks by a third (the fused NTT 12,412 → 8,122 lines), ptxas
gets faster, and registers fall — `bn254_fr_mul_fast` 48 → 36, the fused NTT
87 → 83. And unlike the fusions, it applies to every field kernel in the repo.

### What is left

The 2.47 ms at 2^22 splits ~1.0 / ~1.0 / ~0.5 across the three kernels.

1. **Generate the top stages' twiddles instead of reading them.** The depth-2
   kernel is DRAM-bound at 92%, and a third of what it moves is not data:

   | kernel | data | twiddles | twiddle share |
   |---|---|---|---|
   | `fused` | 268.4 MB | 0.0 MB | 0% |
   | `fused_high4` | 268.4 MB | 8.4 MB | 3% |
   | `fused_high2` | 268.4 MB | **125.8 MB** | **32%** |

   A stage of quarter `q` carries `3q` twiddle words, so the top stages
   approach one twiddle table per data array — `q = N/4` alone is three
   quarters of the array. A two-level table (`w^i = w^(i_hi·2^k) · w^(i_lo)`)
   turns 126 MB into kilobytes for two extra Montgomery multiplies per
   twiddle, which is the right trade in a kernel at 92% of DRAM peak and 55%
   of SM.

2. **The host-side digit-reversal permutation**, which is still outside the
   timed region and is the reason the icicle comparison is not yet end to end.

Neither was visible before the fusions landed. Re-rank after every structural
change — and note that the lever this section ranked first last time turned out
to be worth 6%.

### The shared-memory surface this needed

None of the above was expressible before. `SharedMemory::alloc` emitted a
hardcoded 8 KB `.b8` blob with no way to index it, and **`barrier_sync()`
emitted no instruction at all** — while `emit_block`'s barrier-hoisting pass
recognised it and moved arithmetic across it. `shared_alloc_u32`,
`shared_load_v4`, `shared_store_v4` and a real `bar.sync` are gated by
`tests/ptx_shared_memory.rs`, which assembles the PTX *and* runs a cross-thread
exchange on the device.

Making the barrier real immediately broke the fused kernel, which is how the
hoisting pass turned out to be unsound: it scanned past statements it could
not hoist instead of stopping at them, and it checked no dependencies at all,
so it pulled a Montgomery multiply out from under the shared loads defining its
operands. It now hoists only a contiguous run of arithmetic whose every operand
is already bound at the barrier.

### How the fused kernels are checked

The naive O(N^2) DFT, at a size where it is affordable — and making that
possible is why `Q0` is a runtime parameter rather than a generated constant.
A depth-4 group needs `N >= 4^9` at its production `Q0 = 1024`, and a DFT at
2^18 is ~7e10 field multiplies. Passed in, the identical kernel runs at
`N = 4096` with `Q0 = 4^(5-h)`, where the oracle costs 16.7M multiplies and
every line of the index derivation is exercised.

`Q0` cannot simply be 4 for every depth — a depth-`h` CTA needs `Q0 >= 1024/4^h`
for `p0 = pg*U` to enumerate anything, and the first version of the test that
assumed otherwise was rejected by `cuLaunchKernel` rather than passing quietly.

Three tests, and the split between them is real rather than belt-and-braces —
verified by mutation:

| mutation | `high_fusion` (depth 4) | `every_strided_depth` | `every_fusion` |
|---|---|---|---|
| depth-4 twiddle index drops `jpos` | **fail** | **fail** | **fail** |
| depth-4 wrong table offset | **fail** | **fail** | **fail** |
| depth-4 slot drops `u` | **fail** | **fail** | **fail** |
| depth-2 wrong table offset | pass | **fail** | **fail** |
| depth-3 twiddle index drops `jpos` | pass | **fail** | pass |
| swizzle → identity | pass | pass | pass |

The depth-3 row is the one that justifies `every_strided_depth` existing: no
size in the schedule tests uses depth 3, so nothing else would catch it. The
last row is also correct — the swizzle is a bijection applied on both store and
load, so neutering it is a *performance* change, and its guard is the bank
conflict measurement above, not a test.

### The benchmark was under-reporting its own kernel by 38x

`what_the_gpu_ntt_costs` had **no clock warmup**, while its sibling
`is_the_gpu_actually_winning` in the same file did, complete with a comment
describing this exact failure. Measured cold it reported **28.92 ms and an 18.0x
speedup** at 2^20; warm it is **0.76 ms and 393x**, which matches the 0.75 ms
the docs had claimed all along. The documentation was right and the checked-in
benchmark was wrong. Fixed by warming before the timed region.

## Against icicle (Ingonyama), BN254 G1 MSM

icicle v1.3.0, built from source with `CUDAARCHS=89`. Same GPU, same `n`, same
curve, single MSM (batch 1). Both figures are host-to-host for one call, best
of 5 after a warm-up. "staged" means the bases already live on the device,
which is what a prover with a loaded proving key has.

| n | y-gpu cold | icicle cold | y-gpu staged | icicle staged | icicle is |
|---|---|---|---|---|---|
| 2^18 |  37.8 ms | **13.6 ms** |  29.5 ms | **12.3 ms** | 2.8x faster |
| 2^20 | 111.5 ms | **28.9 ms** |  83.3 ms | **21.4 ms** | 3.9x faster |
| 2^22 | 419.1 ms | **88.2 ms** | 309.3 ms | **63.8 ms** | 4.8x faster |

**icicle was 2.8–4.8x faster; after the block-size fix below the staged gap at
2^22 is ~4.2x.** The table above predates that fix — re-measure it before
quoting the ratio.

Against the same CPU baseline that gives y-gpu 2.6–3.5x, icicle is ~10–14x.

### Where the gap is — now profiled

NCU on `bn254_msm_bucket`, three launches of different grid sizes:

| metric | value |
|---|---|
| `dram__throughput` | **0.8–8.7%** |
| `sm__throughput` | 24–27% |
| `sm__warps_active` | **11–14%** |
| `smsp__…stalled_wait_per_issue_active` | **0.70** |
| `…stalled_long_scoreboard…` | 0.05–0.15 |

**Latency-bound, not bandwidth- or throughput-bound.** 70% of stalls are
`wait` — a dependent ALU result — and memory barely registers. A kernel in
that state wants more warps in flight, and at 136 registers per thread a
256-thread CTA needs 34,816 of an SM's 65,536: one block per SM, 16.7%
occupancy ceiling.

**Fixed: the block size is 128 now, and the win depends entirely on the window
geometry.** Swept at n = 2^20 by `what_the_msm_block_size_costs`:

| nw | buckets | threads/SM | b=256 | b=128 | |
|---|---|---|---|---|---|
| 20 | 139,264 | 2110 | 36.6 ms | 25.6 ms | **1.43x** |
| 22 | 69,632 | 1055 | 45.0 ms | 29.5 ms | **1.53x** |
| 25 | 29,696 | 450 | 53.8 ms | 42.1 ms | 1.28x |
| 28 | 15,360 | 233 | 61.6 ms | 60.0 ms | 1.03x |
| 31 | 9,472 | 144 | 119.8 ms | 114.3 ms | 1.05x |

Below ~450 threads per SM the kernel is **thread-starved** — there are not
enough buckets to fill the machine — and no block size helps. Above it it is
**register-limited** and a smaller block buys occupancy directly. Measuring
only `nw = 28` reported a flat 1.00x while the end-to-end run showed 1.18x;
sweep the geometry too. End to end this took the kernel from **4.47x to 6.10x**
a 32-core CPU at n = 2^22, and cold from 2.18x to 2.57x.

### What actually forces the bad geometry, and it is NOT the reduction

This section used to say the host-side bucket reduction was the reason wider
windows do not pay. **That is wrong.** The phase split at n = 2^20 (block 128):

| nw | buckets | bin | kernel | d2h | reduce | TOTAL |
|---|---|---|---|---|---|---|
| 20 | 139,264 | **68.6** | 36.5 | 14.0 | 5.5 | 160.4 |
| 22 | 69,632 | 49.1 | **33.8** | 7.1 | 3.0 | 130.4 |
| 25 | 29,696 | 32.0 | 43.5 | 3.4 | 1.7 | **118.6** |
| 28 | 15,360 | 28.8 | 61.9 | 1.6 | 1.0 | 133.4 |
| 31 | 9,472 | 27.5 | 115.6 | 1.0 | 0.7 | 184.1 |

`reduce` is 5.5 ms at `nw = 20` — it is not what stops anything. The phase that
grows with the bucket count is **`bin`, the host-side binning: 68.6 ms against
32.0**, and it is the largest single phase at *every* geometry. It scales that
way because its cost is dominated by the `O(threads × buckets)` cursor table,
not by the `O(n × windows)` counting sort — 32 × 139,264 entries against
32 × 29,696.

So the kernel is fastest at `nw = 22` (33.8 ms) and the total is fastest at
`nw = 25` (118.6 ms), and the 10 ms of kernel time in between is being paid to
save 17 ms of binning.

### Inside binning, and what a phase split settles

`where_binning_spends_its_time` (`--ignored`) breaks it into its three phases.
At n = 2^20, before any of the fixes below:

| nw | buckets | cursor MB | histogram | prefix | scatter | TOTAL |
|---|---|---|---|---|---|---|
| 20 | 139,264 | 17.8 | 3.8 | **23.6** | **39.3** | 67.1 |
| 22 | 69,632 | 8.9 | 3.3 | 11.2 | 32.2 | 47.0 |
| 25 | 29,696 | 3.8 | 3.5 | 4.8 | 22.7 | 31.3 |
| 28 | 15,360 | 2.0 | 4.2 | 2.6 | 19.6 | 26.8 |
| 31 | 9,472 | 1.2 | 4.5 | 0.3 | 20.9 | 26.1 |

The cursor table *is* a real cost and it *does* track the bucket count — 0.3 ms
at `nw = 31`, 23.6 ms at `nw = 20`. But it is not the largest phase at any
geometry; the **scatter** is, and it grows too. Two fixes landed:

- **The prefix pass was one serial loop of `nchunk × nb` iterations**, each
  touching `counts[t][b]` and `cursor[t·nb + b]` at stride `nb` — two cache
  misses per iteration over a 17.8 MB table. Split into a parallel per-bucket
  total, a serial exclusive prefix that is only `nb` long, and a parallel
  cursor fill: **23.6 → 3.3 ms**.
- **Phase 3 copied its cursor row out** (`cur0.to_vec()`) before using it —
  an `nb`-sized allocation and copy per thread per call, 17.8 MB of churn for
  a row nothing else reads. It mutates `cursor.chunks_mut(nb)` in place now.

Binning at n = 2^20: **67.1 → 45.9 ms** at `nw = 20`, 47.0 → 34.8 at `nw = 22`,
31.3 → 26.8 at `nw = 25`. The optimum geometry moved with it — `nw = 22` is now
the best total (113.1 ms) where `nw = 25` was (118.6), which is the wider-window
shift the ranking predicted.

**The obvious next fix was tried and it lost.** Point-outer scatter writes into
every window's slice of `idx` at once, spanning 84 MB at `nw = 20`, where
window-outer would confine a pass to one window's ~4 MB. Swapping the loops
measured **worse** — 22.1 → 33.1 ms at `nw = 25` — and went flat across
geometries. The cause is false sharing: a thread's slice of a bucket is
`n / (threads · buckets_per_window)` entries, about **four u32** at `nw = 20`,
finer than a cache line, and consecutive threads' slices of the same bucket are
adjacent in `idx`. Point-outer lets threads drift across windows independently;
window-outer marches all 32 onto the same lines at once. Reverted, and the
reasoning is written into the code so it is not re-attempted.

### The scatter: the fix was to use FEWER cores, and the first two attempts lost

The scatter was then the largest host phase at every geometry, and it has an
odd signature: it gets **slower as the work falls**. At n = 2^20 it writes
32.5M entries at `nw = 31` in 19.9 ms and 21.0M entries at `nw = 20` in 34.3 ms.
It tracks the bucket count, not the entry count.

**Attempt 1, a radix-partitioned scatter, LOST.** Pass A streams
`(bucket, point)` pairs into a few hundred per-thread buffers; pass B gives each
partition to one thread, so no two threads ever share a cache line. Measured
0.94x at `nw = 20` down to **0.38x** at `nw = 31`, and 1.11x at its best
partition size (swept over 2^7…2^15) against the 32-thread original. It moves
three times the bytes to avoid misses that mostly were not happening — `nb · 64`
is 8.9 MB at worst and this machine has 64 MB of L3. It was deleted, not kept
behind a flag.

**The diagnosis that mattered came from sweeping the THREAD count**, which is
the one axis that separates the two candidate explanations. Capacity says the
destination working set is `nb` cache lines however many threads walk it, so
per-entry cost should be flat across a thread sweep. Sharing says it should
fall, because halving the threads doubles each thread's contiguous run inside a
bucket. Scatter ms at n = 2^22:

| nw | buckets | 32 | 16 | 12 | 8 | 6 | 4 | ns/entry, 32 → 4 |
|---|---|---|---|---|---|---|---|---|
| 20 | 139,264 | 185.4 | 132.1 | 103.3 | 84.1 | **82.8** | 83.4 | 70.7 → 4.0 |
| 22 | 69,632 | 137.8 | 66.4 | **55.1** | 58.2 | 67.8 | 80.9 | 47.8 → 3.5 |
| 25 | 29,696 | 68.6 | **47.8** | 51.5 | 56.4 | 68.2 | 80.1 | 20.9 → 3.1 |
| 31 | 9,472 | 66.2 | **54.7** | 59.1 | 63.5 | 79.1 | 95.5 | 16.3 → 2.9 |

Per-entry cost collapses by 17x. **It is sharing, and 32 threads is never the
optimum** — at `nw = 20` it is 2.2x off it. The scatter now runs on a group of
histogram chunks per thread, with the group size chosen from `nb` alone (the
optimum agrees to within one step between n = 2^20 and 2^22, so it does not
depend on `n`). Binning at n = 2^20: **41.9 → 28.5 ms** at `nw = 20`, 36.5 →
24.1 at `nw = 22`, 26.4 → 19.7 at `nw = 25`.

End to end, same session, A/B by `Y_MSM_SCATTER_THREADS`, three runs:

| n | cold | fixed bases | best nw |
|---|---|---|---|
| 2^18 | 2.51 → 2.55x (**no effect**) | 3.30 → 3.44x | 28 |
| 2^20 | 2.47 → **2.93x** | 3.61 → **4.64x** | 25 → **22** |
| 2^22 | 2.49 → **2.99x** | 3.67 → **4.79x** | 25 → **22** |

A single earlier run read 2^18 as 2.22 → 2.77x. **It did not reproduce** — two
further A/Bs put it at 2.51 → 2.55x, i.e. flat, which is what the phase split
predicts (binning is a small share at that size). Sizes at or below 2^16 swing
2x run to run because they sit at the dispatch crossover; do not read them at
all.

**The `kernel only` column also moved (6.38 → 9.13x at 2^20) and that is NOT a
kernel change.** The kernel is byte-identical and its input is byte-identical —
`binning_does_not_depend_on_the_thread_count` pins the latter. What happened is
that cheaper binning moved the chosen geometry from `nw = 25` to `nw = 22`, and
`nw = 22`'s kernel is faster. At *matched* geometry the kernel times agree
(122.3 vs 118.1 ms at `nw = 22`, n = 2^22). A host-side change can move an
apparently unrelated device column through the tuning decision it feeds; check
the geometry before attributing it.

### The scatter proves its own memory safety on every run

The scatter writes through a raw pointer shared across threads
(`unsafe impl Sync`), so "no two threads touch the same slot" is load-bearing
and was only an *argument* — the histogram counts match the scatter's writes,
therefore the per-group regions tile each bucket exactly. Regrouping the
threads is precisely the kind of change that breaks such an argument quietly:
an overlap silently corrupts one point index, which still yields a valid curve
point and an MSM that is merely wrong.

It is now checked rather than argued. Each thread advances its cursor once per
write, so afterwards the cursor holds where that group *stopped*; if every
group stopped exactly where the next one *started*, and the last stopped at
`off[b+1]`, then the runs tile `off[b]..off[b+1]` exactly — every slot written,
by one thread, in bounds. Two unstated preconditions are asserted alongside it:
that the group chunking tiles the input (`zip` truncates, so a mismatch would
silently drop the tail) and that `n · nw` fits the `u32` offsets (a wrap would
under-allocate `idx` while the cursors still ran past its end — an out-of-bounds
*write*, not a wrong answer).

Costs ~0.2–1.0 ms against a 14–23 ms scatter. Two details were needed to get it
there, and both were measured rather than assumed: the starts array is filled
inside the cursor build rather than `clone()`d afterwards (the clone was 1.34 ms,
almost all first-touch page faults), and the comparison runs group-outer, since
the bucket-outer form strides by `nb` and cost 1.45 ms against 0.2. The check is
mutation-verified: making two groups' regions overlap fails it in every test
that touches binning.

**The first version of the check was itself wrong** — after the scope *every*
cursor holds an end, so comparing group `gi` against `cursor[gi+1]` compares two
ends. Empty groups make the two differ, and it failed on bucket 1 immediately.
Worth noting because a post-condition that is subtly wrong in the *safe*
direction would have passed quietly and proved nothing.

### Ranked, after all of the above

1. **`bin` is still the largest host phase**, now 79–90 ms at n = 2^22 against
   `stage` 61 and `h2d` 60. Its scatter is at the floor for this layout — the
   thread sweep bottoms out at ~3 ns/entry and the optimum is already there —
   so the next real move is doing the binning **on the device**, which also
   deletes the 84 MB `Idx` upload that `h2d` is mostly made of.
2. **`d2h` + `reduce` on the device.** 4.9 ms at `nw = 25` but 20.4 ms at
   `nw = 20`, so it compounds with (1) rather than standing alone.
3. **No signed-digit recoding.** NAF-style recoding halves the bucket count
   for the same window width; y-gpu does not do it.
4. **Jacobian + Jacobian addition.** Mixed (affine + Jacobian) addition is
   cheaper and is what a bases-are-affine MSM should use.

## Against arkworks (CPU), whole Groth16 prove

arkworks is parallel here (~12.5 cores at n=2^20 — via feature unification, not
by declaration). Best of 3.

| circuit | ~250k constraints | ~1M |
|---|---|---|
| sparse (multiplication chain) | ~4.6–6.0x | ~5.1x |
| dense (Poseidon chain) | ~5.7x | ~6.2x |

Run-to-run variance is ~15%.

**Read these two tables together.** The prover speedup over a CPU is real, but
the MSM inside it is ~4x off the state of the art, so the same prover built on
icicle's MSM would be substantially faster than this one. The honest summary
is: y-gpu beats a parallel CPU prover and does not currently compete with a
mature GPU MSM library.

## Reproducing the icicle comparison

```sh
curl -sL https://codeload.github.com/ingonyama-zk/icicle/tar.gz/refs/tags/v1.3.0 | tar xz
# icicle hardcodes /usr/local/cuda; on Arch it is /opt/cuda
sed -i 's|/usr/local/cuda|/opt/cuda|g' icicle-1.3.0/wrappers/rust/icicle-cuda-runtime/build.rs
# CUDA 13 dropped sm_50, which is CMake's default compiler probe
CUDAARCHS=89 cargo build --release
```
