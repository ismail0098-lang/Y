"""An exact W8A8 linear for DECODE shapes, written because NCU said to.

## What the profiler found

Profiling two steady-state decode steps (batch 32) with `ncu --set basic`:

| | exact arm | stock arm |
|---|---|---|
| GEMM total | 4803 us | 4957 us |
| GEMM **DRAM % of peak** | **36.1%** | **69.5%** |

The int8 path moves *half* the weight bytes of bf16 and should therefore be
about twice as fast. It is not: it runs at half the bandwidth efficiency, and
the two effects cancel almost exactly. `torch._int_mm` dispatches a CUTLASS
tensor-core GEMM built for large M; at decode M is the batch size, so most of
the tile is wasted. **The int8 weight advantage was being thrown away.**

The same profile showed the surrounding elementwise work costing another ~29%
in kernels that are neither bandwidth-bound nor compute-bound - the activation
quantiser ran at 55.9 GB/s while executing 12 instructions per thread, i.e.
pure latency stall on a trivial amount of work.

## What this does about it

One kernel that reads the int8 weights once and does everything else on the
way past: int32 accumulation via `tl.dot` (tensor cores, and integer addition
so any tile or split-K order gives the same bits), then the dequantise and bias
in the epilogue. The activation quantiser is a second small kernel rather than
a third pass.

Exactness is unchanged and is the acceptance criterion: `K * 127^2` is at most
78.5M for this model, far inside int32, so the accumulator never wraps and the
result is the same integer whatever order the tiles complete in.
"""
import os

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

# The split-K kernel below is now one entry in `_CANDIDATES` rather than a
# separately-flagged path. Its history is worth keeping: it was written to fix
# `down_proj` (2x on that kernel, 39.3 -> 20.0 us) and gated by a hand-written
# rule that at M=32 fired on `down_proj` and nothing else - 24 of the 169 GEMMs
# in a decode step - so it moved the model 1.07x -> 1.08x, i.e. not at all. The
# guard also keyed on K when the starvation it existed to fix is caused by N.
# Measured against the model it does win on that shape (2.6% of decode device
# time), so the kernel was right and its GATING was wrong; a candidate list does
# not need to be right, only complete, and the measurement picks.


@triton.jit
def _quant_rows(X, X8, SX, M, K, stride_xm, BLOCK_K: tl.constexpr):
    """Symmetric per-row int8 quantisation. One program per row."""
    row = tl.program_id(0)
    amax = 0.0
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + row * stride_xm + k, mask=k < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    s = amax / 127.0
    s = tl.where(s == 0.0, 1.0, s)
    tl.store(SX + row, s)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + row * stride_xm + k, mask=k < K, other=0.0).to(tl.float32)
        # `x / s` MUST be IEEE correctly-rounded. Triton's default float divide
        # is the approximate one, and at 458,752 elements it produced exactly
        # -22.5 where `div.rn` gives -22.5000019 - so round-half-to-even
        # answered -22 against torch's -23. One element in 458,752, invisible at
        # decode shapes, and it is the same class of bug as `ex2.approx.f32`:
        # an *unspecified* operation, which is precisely what a determinism
        # project cannot contain. Quantisation is a step function, so a 1-ulp
        # divide error becomes a whole int8 step.
        q = tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(x, s))
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(X8 + row * K + k, q.to(tl.int8), mask=k < K)


# Autotuning is SAFE HERE FOR A STATEABLE REASON, and would not be for a float
# GEMM: the accumulator is int32, integer addition is associative and
# commutative, so every tile shape and K-split produces the identical bits. The
# tuner may therefore pick per shape without the output depending on what it
# picked - `check_all_gemm_configs_agree` asserts exactly that.
#
# **`triton.autotune` is NOT used, because it tunes in the wrong cache regime.**
# It benchmarks each candidate by calling it repeatedly on the same buffers, so
# the weight matrix is resident in this card's 48 MB L2 for every timing it
# takes. A decode step reads ~493 MB of distinct weights, so in the model every
# weight is cold. The two regimes disagree about the answer, not just about the
# magnitude: with L2 hot the big tile wins on reuse, and from DRAM the small
# tile wins because it launches enough CTAs to fill the machine.
#
# The tell was `gate/up` measuring **108% of DRAM peak** in the first version of
# `tools/w8a8_shape_sweep.py`. A number above the roofline is not a good result,
# it is a wrong question; it is what sent me looking at what the benchmark was
# keeping in cache, and `triton.autotune` turned out to have the same flaw.
#
# **BUT THE CACHE REGIME IS NOT WHY THIS IS FASTER, and the microbenchmark that
# said it was does not reproduce.** That sweep put cold tuning at 1.20x on the
# GEMM; re-run, two of its six shapes move by 2x between runs (`gate/up`: 6.0,
# 8.9, 17.6 us), so its per-shape table cannot carry a conclusion. Measured on
# the model, where stock held 3.355-3.385 ms across six runs, decode device
# time for the exact arm:
#
#   triton.autotune tiles   3.513 / 3.515 ms   (exact/stock 1.04-1.05x)
#   this tuner, cold        3.278 ms           (0.97x)   <- ships
#   this tuner, hot         3.231 ms           (0.96x)
#
# So the rework is worth **1.07x on decode device time**, and cold-vs-hot is a
# **wash** - 1.5% apart, inside the exact arm's own ~3% run-to-run spread. What
# separates this tuner from `triton.autotune` is the CANDIDATE LIST (split-K is
# in it, worth 2.6% on `down_proj` alone) and interleaved measurement.
#
# Cold tuning stays because a decode step genuinely is cold and the reporting
# bugs it fixed were real - not because it was measured to pay. Do not quote the
# 1.20x.
@triton.jit
def _w8a8_gemm(X8, W8, SX, SW, Y, M, N, K,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """acc = x8 @ w8 in int32, then y = acc * sx * sw (+ bias).

    W8 is [K, N] contiguous - pre-transposed at construction, so the inner loop
    walks it in its fastest-varying dimension and every warp's load is a
    contiguous run.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(X8 + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0)
        b = tl.load(W8 + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0)
        acc += tl.dot(a, b, out_dtype=tl.int32)
    sx = tl.load(SX + offs_m, mask=offs_m < M, other=0.0)
    sw = tl.load(SW + offs_n, mask=offs_n < N, other=0.0)
    # The bias add deliberately does NOT happen here. `acc * sx * sw + bias`
    # gets contracted into an FMA, which rounds once where the reference rounds
    # twice - 25% of elements differed by 1 ulp, and `libdevice` exposes no
    # `fadd_rn`/`fmul_rn` to pin it. Two multiplies with no add cannot contract,
    # so this expression is safe; the caller adds the bias in torch. Only q/k/v
    # carry a bias in this model and their outputs are tiny, so the extra pass
    # is free. Same category as the `div_rn` fix above: an exactness project
    # has to pin every float operation to a *specified* one.
    y = acc.to(tl.float32) * sx[:, None] * sw[None, :]
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], y,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _w8a8_gemm_splitk(X8, W8, ACC, M, N, K,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """Same product, with the K loop cut `SPLIT_K` ways and summed by atomics.

    For a K-heavy skinny shape the ordinary kernel runs out of blocks, not
    bandwidth: `down_proj` is K=4864, N=896, so at BLOCK_N=128 it launches
    **7 CTAs onto 66 SMs** and reaches 111 GB/s on a 4.36 MB weight matrix.
    Splitting K multiplies the block count by SPLIT_K.

    **`tl.atomic_add` on int32 is exact and order-independent**, so this is
    legal here in a way it would not be for a float accumulator: integer
    addition is associative and commutative, so whatever order the CTAs happen
    to land in, the sum is the same. The partial sums cannot wrap either -
    `K * 127^2` is 78.5M against int32's 2.1e9.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(X8 + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0)
        b = tl.load(W8 + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0)
        acc += tl.dot(a, b, out_dtype=tl.int32)
    tl.atomic_add(ACC + offs_m[:, None] * N + offs_n[None, :], acc,
                  mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _w8a8_gemm_hilo(XHI, XLO, W8, SX, SW, Y, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
    """`(xhi*128 + xlo) @ w8`, for activations wider than int8, still exact.

    `tools/exact_quant_attribution.py` measured the whole remaining perplexity
    cost of this scheme to be the ACTIVATIONS - weights alone cost -0.12%, i.e.
    nothing, and one extra bit of activation width recovers 102% of the gap.
    The obvious lever (group-wise weight scales) recovers 6%.

    A wider activation cannot go through `tl.dot`, which wants int8 operands. So
    the value is carried as two int8 digits, `q = hi*128 + lo` with `hi` in
    [-4, 3] and `lo` in [0, 127] at 9 bits, and the product is reassembled as
    `128*(hi@w) + (lo@w)`. **Every step is integer**, so the accumulation is
    still associative and commutative and the batch-invariance argument is
    untouched - which is exactly what the group-scale alternative could not say,
    since its `G` partials have to be recombined with float scales.

    The two dots share ONE load of `b`, and that is the whole economic case: at
    decode M the GEMM is DRAM-bound on the weights, so a second `tl.dot` over a
    tile already in registers adds arithmetic to a kernel that is not
    arithmetic-bound. Measured cost is in `docs/deterministic_inference.md`;
    predicting it as free would be the mistake this repo has made before.

    The `*128` is exact in int32 by the same budget check as everything else:
    `K * a_levels * 127` is 3.16e8 at K=4864 and a_levels=511, against 2.1e9.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        am = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        ahi = tl.load(XHI + offs_m[:, None] * K + offs_k[None, :], mask=am, other=0)
        alo = tl.load(XLO + offs_m[:, None] * K + offs_k[None, :], mask=am, other=0)
        b = tl.load(W8 + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0)
        acc += tl.dot(ahi, b, out_dtype=tl.int32) * 128
        acc += tl.dot(alo, b, out_dtype=tl.int32)
    sx = tl.load(SX + offs_m, mask=offs_m < M, other=0.0)
    sw = tl.load(SW + offs_n, mask=offs_n < N, other=0.0)
    # Two multiplies and no add, so nothing can contract into an FMA - see the
    # note in `_w8a8_gemm`. The bias is still added by the caller in torch.
    y = acc.to(tl.float32) * sx[:, None] * sw[None, :]
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], y,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _w8a8_gemm_hilo_splitk(XHI, XLO, W8, ACC, M, N, K,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """The hi/lo product with the K loop split, summed by int32 atomics.

    This variant exists so a hi/lo measurement is not secretly a measurement of
    losing split-K. `_pick` chooses SPLIT_K=4 for `down_proj` and that is worth
    2.6% there, so comparing a non-split hi/lo kernel against a split int8 one
    would charge the digit split for a tile change it did not make.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        am = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        ahi = tl.load(XHI + offs_m[:, None] * K + offs_k[None, :], mask=am, other=0)
        alo = tl.load(XLO + offs_m[:, None] * K + offs_k[None, :], mask=am, other=0)
        b = tl.load(W8 + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0)
        acc += tl.dot(ahi, b, out_dtype=tl.int32) * 128
        acc += tl.dot(alo, b, out_dtype=tl.int32)
    tl.atomic_add(ACC + offs_m[:, None] * N + offs_n[None, :], acc,
                  mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# (BLOCK_N, BLOCK_K, num_warps, num_stages, SPLIT_K). SPLIT_K=1 is the ordinary
# kernel; >1 uses the atomic one. Both are in ONE search space on purpose: the
# previous code chose between them with a hand-written guard
# (`blocks >= sms or k < 2048`), which at M=32 fired on `down_proj` and nothing
# else - 24 of 169 GEMMs in a step - which is why split-K measured 2x on that
# kernel and nothing at all end to end. Worse, the guard keyed on K when the
# starvation it was there to fix is caused by N. Cold measurement then showed a
# plain 64x128 tile beating every K-split on that very shape, so the heuristic
# was both misaimed and unnecessary. A candidate list does not need to be right,
# only complete; the measurement picks.
_CANDIDATES = (
    (32, 64, 2, 3, 1), (32, 128, 2, 3, 1), (64, 64, 4, 3, 1),
    (64, 128, 4, 3, 1), (128, 32, 4, 3, 1), (128, 64, 4, 4, 1),
    (128, 128, 4, 3, 1), (256, 32, 8, 3, 1),
    (64, 128, 4, 3, 2), (64, 128, 4, 3, 4), (64, 128, 4, 3, 8),
)
_PICKED = {}
_TUNE = os.environ.get("Y_EXACT_GEMM_TUNE", "cold")     # cold | hot | first | <n>x<n>
# L2 the tuning has to defeat. Queried once, because a wrong value here silently
# turns the cold benchmark back into a hot one.
try:
    L2_MB = torch.cuda.get_device_properties(0).L2_cache_size / 1e6
except Exception:
    L2_MB = 48.0


def _bench_all(m, n, k, cfgs, device, rounds=9, inner=20, cold=True):
    """Time every candidate for one shape, INTERLEAVED, on cold weights.

    Returns {cfg: us}, the minimum over rounds.

    Interleaving is the reason this is one function over a list rather than a
    loop over a per-config timer. Running each candidate to completion in turn
    let slow drift land entirely on whichever candidate was running, and the
    winner changed between consecutive tunings of the same shape - `gate/up`
    came out 128x64, then 64x64, then 64x64 on three trials. Round-robin puts
    the same drift into every candidate, which is the discipline this project
    already applies to its model-level A/Bs and had not applied here.

    `torch.cuda.Event` around a loop of these would measure CPU dispatch, not
    the kernel: at these sizes the issue rate is ~13 us/call while the kernel is
    ~4, so the events span the gaps. CUPTI reports occupancy of the device,
    which is the quantity the roofline argument is about.

    `cold=False` reuses ONE buffer, which is what `triton.autotune` does and
    what this code used to do. It exists as a CONTROL: `Y_EXACT_GEMM_TUNE=hot`
    reproduces the old regime so the two can be A/B'd in one window, rather than
    the cold result having to be believed on its own.
    """
    # Size the ring in BYTES, not in buffers, and WALK IT CONTINUOUSLY. Two
    # separate ways this benchmark stayed hot while claiming to be cold:
    #   * a 48-buffer cap, which for a 0.80 MB weight is 38 MB - inside this
    #     card's 48 MB L2;
    #   * the round loop restarting the buffer index at 0, so 5 rounds x 20
    #     calls touched buffers 0..19 (16 MB) five times instead of 100 distinct
    #     ones, leaving rounds 2-5 hot. Taking the MINIMUM then selected the
    #     most cache-contaminated round of the five.
    # Together they read `q_proj` at 3.43 us against 4.4 in an interleaved
    # sweep - the same error being corrected in `triton.autotune`, made twice
    # more inside the correction itself. The ring targets 2x L2 so a buffer is
    # certainly evicted before the walk returns to it.
    one_mb = k * n / 1e6
    copies = max(2, min(4096, int(L2_MB * 2 / max(one_mb, 1e-9)) + 1)) if cold else 1
    bufs = [torch.randint(-127, 128, (k, n), dtype=torch.int8, device=device)
            for _ in range(copies)]
    x8 = torch.randint(-127, 128, (m, k), dtype=torch.int8, device=device)
    sx = torch.ones(m, device=device)
    sw = torch.ones(n, device=device)
    bm = min(64, max(16, triton.next_power_of_2(m)))
    y = torch.empty((m, n), dtype=torch.float32, device=device)
    acc = torch.zeros((m, n), dtype=torch.int32, device=device)
    step = [0]

    def once(cfg):
        bn, bk, w, s, sp = cfg
        wt = bufs[step[0] % copies]
        step[0] += 1
        if sp == 1:
            _w8a8_gemm[(triton.cdiv(n, bn), triton.cdiv(m, bm))](
                x8, wt, sx, sw, y, m, n, k, BLOCK_M=bm, BLOCK_N=bn,
                BLOCK_K=bk, num_warps=w, num_stages=s)
        else:
            # The zero-fill IS timed; the epilogue is NOT, and both halves of
            # that were settled against the model rather than argued.
            #
            # Timing neither made split-K look 1.19x better than plain 64x128 on
            # `down_proj`; timing both made it look 8% WORSE (18.9 vs 17.5).
            # In the model split-K wins by 2.6% of decode device time
            # (3.332 -> 3.245 ms, stock steady at 3.36), so the second version
            # was over-corrected. The evidence is the launch count: turning
            # split-K on for that shape adds **24** kernels per step, one per
            # layer, not 48 - so inductor fuses `acc * sx * sw` into neighbouring
            # elementwise work in the model, while an isolated benchmark has no
            # neighbour and must pay for a whole kernel.
            #
            # A residual bias toward split-K remains (a fused epilogue is not
            # free, it enlarges its host kernel), so this is the closer of two
            # imperfect models and not a correct one. **An isolated benchmark
            # cannot price an epilogue whose cost depends on the graph around
            # it**; the arbiter is `exact_launch_audit.py`.
            acc.zero_()
            _w8a8_gemm_splitk[(triton.cdiv(n, bn), triton.cdiv(m, bm), sp)](
                x8, wt, acc, m, n, k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                SPLIT_K=sp, num_warps=w, num_stages=s)

    for c in cfgs:                                       # compile, not timed
        once(c)
    torch.cuda.synchronize()
    best = {c: float("inf") for c in cfgs}
    for _ in range(rounds):
        for c in cfgs:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(inner):
                    once(c)
                torch.cuda.synchronize()
            us = sum(e.self_device_time_total for e in prof.events()
                     if e.device_type == torch.autograd.DeviceType.CUDA)
            best[c] = min(best[c], us / inner)
    del bufs
    return best


_DEFAULT = (64, 128, 4, 3, 1)
_TUNE_M = 32            # M the tuning runs at; see `_pick` for why it is fixed
_PROFILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        ".ysu_exact_gemm")


def _gpu_tag():
    try:
        return torch.cuda.get_device_name(0).replace(" ", "_")
    except Exception:
        return "unknown"


def _load_profile():
    """Measured tile choices from a previous run, keyed on (GPU, N, K).

    **The cache is what makes the choice stable, and tightening the tuner was
    not.** Several candidates for a shape sit inside the 5% tie band, so which
    ones are *inside it* moves with the noise even when the band does its job -
    raising the round count from 5 to 9 simply moved the flip from `gate/up` to
    `q_proj`. Measuring once per machine and reusing the answer is the fix, and
    it is the convention this repo already uses for GEMM autotuning
    (`.ysu_hw_profile`, CLAUDE.md gotcha #3).

    Same trapdoor as that one, for the same reason: **the cache cannot detect
    that the kernel itself changed.** After editing `_w8a8_gemm`, re-tune with
    `Y_EXACT_GEMM_TUNE=force` or every run will keep using a tile chosen for the
    old kernel. Nothing here can notice that for you.
    """
    out = {}
    try:
        with open(_PROFILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                parts = key.split("_")
                if parts[0] != "GEMM" or "x" not in parts[-1]:
                    continue
                if "_".join(parts[1:-1]) != _gpu_tag():
                    continue                      # another GPU's measurement
                n, k = (int(v) for v in parts[-1].split("x"))
                out[(n, k)] = tuple(int(v) for v in val.split(","))
    except FileNotFoundError:
        pass
    return out


def _save_profile(n, k, cfg):
    with open(_PROFILE, "a") as f:
        f.write(f"GEMM_{_gpu_tag()}_{n}x{k}={','.join(str(v) for v in cfg)}\n")


def _pick(n, k, device=None):
    """Config for this weight shape, measured once on cold weights and cached.

    **Keyed on (N, K) and not on M, deliberately.** In the decode regime the
    block count is `cdiv(N, BLOCK_N) * cdiv(M, BLOCK_M)` and `BLOCK_M` is
    `min(64, next_pow2(M))`, so the second factor is 1 for every M <= 64 and the
    tile choice cannot depend on M. Prefill (M=2048) fills the machine on its
    own and is not what these tiles are for. Keying on M as well would also make
    the key symbolic under `torch.compile`, which is how the first version of
    this failed - dynamo traced into the tuner and tried to benchmark a kernel
    with a SymInt grid.

    **Never benchmarks while tracing.** A cache miss under `torch.compile`
    returns a safe default rather than measuring: the tuner allocates ~96 MB and
    synchronises, neither of which belongs in a traced region. `prime()` is
    called from `exact_model.convert()` so the miss does not happen in practice.

    `Y_EXACT_GEMM_TUNE=first` takes the first candidate without measuring (for
    tests, which care only about values - and every candidate agrees on those);
    `Y_EXACT_GEMM_TUNE=64x128` pins one.
    """
    key = (n, k)
    if key in _PICKED:
        return _PICKED[key]
    if _TUNE == "first":
        _PICKED[key] = _CANDIDATES[0]
    elif "x" in _TUNE:
        bn, bk = (int(v) for v in _TUNE.split("x"))
        _PICKED[key] = next((c for c in _CANDIDATES if c[0] == bn and c[1] == bk),
                            _DEFAULT)
    elif device is None or torch.compiler.is_compiling():
        return _DEFAULT                       # do NOT cache a guess
    else:
        viable = [c for c in _CANDIDATES if c[4] == 1 or k // c[4] >= 256]
        times = _bench_all(_TUNE_M, n, k, viable, device,
                           cold=(_TUNE != "hot"))
        _PICKED[key] = _resolve(times, viable)
        if _TUNE in ("cold", "force"):
            _save_profile(n, k, _PICKED[key])
    return _PICKED[key]


def _resolve(times, order, band=0.05):
    """Best config, with ties broken STRUCTURALLY rather than by noise.

    Two consecutive tunings of `lm_head` picked 256x32 and then 128x128, which
    an interleaved sweep measures **0.6% apart** - the tuner was deciding a coin
    flip and the model's speed then varied run to run.

    The band is 5% because that is what min-over-5-rounds actually disperses by
    here, measured: 0.2-9.6% across shape/config pairs, median ~2-5%. Do not
    take that from an earlier reading of 0.1-0.9% - that one was taken while the
    benchmark was accidentally hot, and a hot benchmark is repeatable precisely
    because it is not measuring the memory system. Among tied candidates the
    first in `_CANDIDATES` wins, so an un-split kernel beats a split one on a
    tie and does not pay a zero-fill launch for nothing.

    This does NOT affect what the model computes - every candidate is bit-equal
    by construction, which is the whole reason tuning is allowed here. It
    affects only whether the same machine gives the same speed twice.
    """
    best = min(times.values())
    tied = [c for c in order if times[c] <= best * (1.0 + band)]
    return tied[0]


def prime(shapes, device):
    """Tune every (N, K) the model will use, before anything is compiled.

    Called from `exact_model.convert()`. On the first run for a given GPU this
    measures each distinct shape (five for Qwen2.5-0.5B, a few seconds total,
    against a `torch.compile` warmup measured in minutes) and appends the result
    to `.ysu_exact_gemm`; afterwards it is a file read.

    `Y_EXACT_GEMM_TUNE=force` re-measures and appends fresh entries, which
    override the older ones because the loader takes the last line for a key.
    """
    if _TUNE != "force":
        _PICKED.update(_load_profile())
    for n, k in sorted(set(shapes)):
        _pick(n, k, device)
    return dict(_PICKED)


def w8a8_matmul(x8, sx, w8t, sw, bias):
    """The GEMM ONLY. `x8`/`sx` come from the caller, deliberately.

    The first version quantised inside this function with `_quant_rows`, and
    NCU showed why that loses: taking the linear out of inductor's graph turned
    a *fused* norm+quantise kernel (830 us) into a bare norm (409 us) plus a
    standalone quantiser (1449 us). The GEMM itself got 658 us faster and the
    lost fusion gave back 1028 us, so end-to-end moved 1.43x -> 1.42x, i.e. not
    at all. **A faster kernel that breaks a fusion can be a net loss**; leave
    the elementwise work where the compiler can attach it to its neighbour.
    """
    m, k = x8.shape
    n = w8t.shape[1]
    y = torch.empty((m, n), dtype=torch.float32, device=x8.device)
    # BLOCK_M is a TILE, not the whole M. Letting it track M directly (which the
    # first version did) asks for BLOCK_M*BLOCK_K*num_stages bytes of shared
    # memory and dies at M=256 with `OutOfResources: Required 122880, limit
    # 101376`. 64 caps it; `tl.dot` wants at least 16 rows.
    block_m = min(64, max(16, triton.next_power_of_2(m)))
    # Tile AND K-split come from one cold-weight measurement, cached per shape.
    bn, bk, warps, stages, split_k = _pick(n, k)
    if split_k > 1:
        # K-heavy and block-starved: cut K and let int32 atomics sum the parts.
        acc = torch.zeros((m, n), dtype=torch.int32, device=x8.device)
        _w8a8_gemm_splitk[(triton.cdiv(n, bn), triton.cdiv(m, block_m), split_k)](
            x8, w8t, acc, m, n, k, BLOCK_M=block_m, BLOCK_N=bn, BLOCK_K=bk,
            SPLIT_K=split_k, num_warps=warps, num_stages=stages,
        )
        y = acc.to(torch.float32) * sx[:, None] * sw[None, :]
    else:
        _w8a8_gemm[(triton.cdiv(n, bn), triton.cdiv(m, block_m))](
            x8, w8t, sx, sw, y, m, n, k, BLOCK_M=block_m, BLOCK_N=bn,
            BLOCK_K=bk, num_warps=warps, num_stages=stages,
        )
    if bias is not None:
        y = y + bias                    # in torch, so it cannot become an FMA
    return y


def w8a8_matmul_hilo(xhi, xlo, sx, w8t, sw, bias):
    """`(xhi*128 + xlo) @ w8t`, scaled. Same contract as `w8a8_matmul`.

    Shares `_pick`'s cached tile with the int8 path deliberately: the shape,
    the weight traffic and the block count are identical, and the only
    difference is one extra `tl.dot` over a tile already in registers. Re-tuning
    would give the digit split its own cache entry keyed on the same (N, K),
    which is how `Y_CTA_OVERRIDE`-style poisoning happens.
    """
    m, k = xhi.shape
    n = w8t.shape[1]
    y = torch.empty((m, n), dtype=torch.float32, device=xhi.device)
    block_m = min(64, max(16, triton.next_power_of_2(m)))
    bn, bk, warps, stages, split_k = _pick(n, k)
    if split_k > 1:
        acc = torch.zeros((m, n), dtype=torch.int32, device=xhi.device)
        _w8a8_gemm_hilo_splitk[(triton.cdiv(n, bn), triton.cdiv(m, block_m), split_k)](
            xhi, xlo, w8t, acc, m, n, k, BLOCK_M=block_m, BLOCK_N=bn,
            BLOCK_K=bk, SPLIT_K=split_k, num_warps=warps, num_stages=stages,
        )
        y = acc.to(torch.float32) * sx[:, None] * sw[None, :]
    else:
        _w8a8_gemm_hilo[(triton.cdiv(n, bn), triton.cdiv(m, block_m))](
            xhi, xlo, w8t, sx, sw, y, m, n, k, BLOCK_M=block_m, BLOCK_N=bn,
            BLOCK_K=bk, num_warps=warps, num_stages=stages,
        )
    if bias is not None:
        y = y + bias
    return y


def w8a8_linear(x, w8t, sw, bias):
    """Quantise + GEMM, for standalone benchmarking and the self-test.

    The model path does NOT use this - it quantises with torch ops so inductor
    can fuse them into the preceding norm, then calls `w8a8_matmul`.
    """
    m, k = x.shape
    x8 = torch.empty((m, k), dtype=torch.int8, device=x.device)
    sx = torch.empty((m,), dtype=torch.float32, device=x.device)
    _quant_rows[(m,)](x, x8, sx, m, k, x.stride(0),
                      BLOCK_K=min(1024, triton.next_power_of_2(k)))
    return w8a8_matmul(x8, sx, w8t, sw, bias)
