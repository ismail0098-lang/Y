"""Per-shape efficiency of the W8A8 GEMM, against the bandwidth it could reach.

    python3 tools/w8a8_shape_sweep.py

`exact_kernel_census.py` says `_w8a8_gemm` is 52% of the exact path's decode
device time, at roughly 43% of this card's DRAM peak. That is an aggregate, and
an aggregate cannot say WHICH shape is slow or WHY. This sweeps the seven
shapes a Qwen2.5-0.5B decode step actually runs and reports, per shape:

  * blocks   - CTAs the chosen tile launches, against 66 SMs. At M=32 a decode
               GEMM is weight-streaming, so a shape that cannot fill the
               machine cannot reach peak bandwidth no matter how good the tile.
  * %peak    - achieved bandwidth over weight bytes, against 672 GB/s.
  * split-K  - the same product with the K loop cut, summed by int32 atomics.

**The point of the %peak column is that it is an absolute ceiling, not a
comparison.** A GEMM at 90% of peak is finished whatever a rival implementation
does; one at 20% has 4x in it and no amount of tile tuning inside the same
grid shape will find it.

Measurement discipline (learned the hard way in this project, twice):
interleave the variants rather than running each to completion, report the
MINIMUM, and print the clock so a throttled or contended run is visible in the
output instead of silently becoming a conclusion.
"""
import _nospace

_nospace.guard()

import argparse  # noqa: E402
import subprocess  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import w8a8_gemv as G  # noqa: E402

PEAK_GBS = 672.0        # RTX 4070 Ti SUPER: 256-bit GDDR6X @ 21 Gbps
SMS = 66

# (name, N, K) for one Qwen2.5-0.5B layer, plus the head. M is the batch.
SHAPES = [
    ("q_proj",   896,    896),
    ("k/v_proj", 128,    896),
    ("o_proj",   896,    896),
    ("gate/up",  4864,   896),
    ("down_proj", 896,   4864),
    ("lm_head",  151936, 896),
]


def clock():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        cur, mx = (int(v) for v in out.split(",")[:2])
        return cur, mx
    except Exception:
        return 0, 0


def time_min(fn, reps, inner=20):
    """DEVICE time per call, in us, from CUPTI - not wall clock.

    The first version of this bracketed `inner` back-to-back calls with CUDA
    events, and produced a flat 13.5-13.8 us for shapes whose weight bytes
    differ by 40x. A flat column across a swept axis means the axis is not what
    is being measured: at these sizes the CPU cannot issue faster than ~13 us
    per call (triton's autotune wrapper plus dispatch), so the events were
    spanning the GAPS between kernels rather than the kernels.

    CUPTI reports how long each kernel actually occupied the GPU, which is the
    quantity the roofline argument is about and the one that carries over to a
    model where the launches are already in flight.
    """
    best = float("inf")
    for _ in range(reps):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(inner):
                fn()
            torch.cuda.synchronize()
        us = sum(e.self_device_time_total for e in prof.events()
                 if e.device_type == torch.autograd.DeviceType.CUDA)
        best = min(best, us / inner)
    return best


class ColdWeights:
    """A ring of distinct weight tensors big enough to blow past L2.

    Benchmarking one GEMM in a loop leaves its weights resident in this card's
    48 MB L2, and the numbers that come out are cache bandwidth: `gate/up`
    measured **108% of DRAM peak**, which is not a thing. A decode step reads
    ~493 MB of distinct weights, so every weight is cold. Rotating over enough
    copies to exceed L2 restores the regime the model is actually in.

    This is not only a reporting fix. `triton.autotune` benchmarks exactly the
    way the broken version did, so its choices were made in the hot regime too
    - which is the hypothesis this file exists to test.
    """

    def __init__(self, k, n, device, budget_mb=96):
        one = k * n / 1e6
        self.n_copies = max(2, min(64, int(budget_mb / max(one, 1e-9)) + 1))
        self.bufs = [torch.randint(-127, 128, (k, n), dtype=torch.int8,
                                   device=device) for _ in range(self.n_copies)]
        self.i = 0

    def next(self):
        b = self.bufs[self.i]
        self.i = (self.i + 1) % self.n_copies
        return b


def run_plain(x8, w8t, sx, sw, m, n, k, cfg=None):
    y = torch.empty((m, n), dtype=torch.float32, device=x8.device)
    block_m = min(64, max(16, triton.next_power_of_2(m)))
    if cfg is None:
        bn, bk, w, s, _ = G._pick(m, n, k, x8.device)
        G._w8a8_gemm[(triton.cdiv(n, bn), triton.cdiv(m, block_m))](
            x8, w8t, sx, sw, y, m, n, k, BLOCK_M=block_m, BLOCK_N=bn,
            BLOCK_K=bk, num_warps=w, num_stages=s)
    else:
        bn, bk, w, s = cfg
        G._w8a8_gemm[(triton.cdiv(n, bn), triton.cdiv(m, block_m))](
            x8, w8t, sx, sw, y, m, n, k, BLOCK_M=block_m, BLOCK_N=bn,
            BLOCK_K=bk, num_warps=w, num_stages=s)
    return y


def run_split(x8, w8t, sx, sw, m, n, k, split, bn=64, bk=128):
    block_m = min(64, max(16, triton.next_power_of_2(m)))
    acc = torch.zeros((m, n), dtype=torch.int32, device=x8.device)
    G._w8a8_gemm_splitk[(triton.cdiv(n, bn), triton.cdiv(m, block_m), split)](
        x8, w8t, acc, m, n, k, BLOCK_M=block_m, BLOCK_N=bn, BLOCK_K=bk,
        SPLIT_K=split, num_warps=4, num_stages=3)
    return acc.to(torch.float32) * sx[:, None] * sw[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(0)
    # Ramp the clock BEFORE reading it. A cold GPU sits at ~240 MHz of 3120 and
    # the first version of this reported that number beside its own results,
    # warning about a condition it had created by not warming up.
    warm = torch.randn(4096, 4096, device=a.device)
    for _ in range(60):
        warm = warm @ warm.T * 1e-4
    torch.cuda.synchronize()
    cur, mx = clock()
    print(f"\nW8A8 decode GEMMs at M={a.m}   SM clock {cur}/{mx} MHz   "
          f"peak {PEAK_GBS:.0f} GB/s over {SMS} SMs")
    if mx and cur < 0.5 * mx:
        print("  NOTE: clock is far below max - warm the GPU or check for "
              "another tenant before quoting these.")
    print("  weights are COLD (rotated over >L2 of distinct copies), which is "
          "the regime a decode step is in")
    print(f"\n{'shape':<11}{'N':>7}{'K':>6}{'MB':>7}{'hot-tuned':>11}{'us':>8}"
          f"{'%pk':>5}{'blk':>5}   {'best cold':>11}{'us':>8}{'%pk':>5}{'blk':>5}"
          f"{'gain':>7}")

    tot_auto = tot_best = 0.0
    rows = []
    for name, n, k in SHAPES:
        m = a.m
        x8 = torch.randint(-127, 128, (m, k), dtype=torch.int8, device=a.device)
        sx = torch.rand(m, device=a.device) * 0.01 + 0.001
        sw = torch.rand(n, device=a.device) * 0.01 + 0.001
        mb = k * n / 1e6
        cw = ColdWeights(k, n, a.device)
        bm = min(64, max(16, triton.next_power_of_2(m)))
        nblk = lambda bn, sp=1: triton.cdiv(n, bn) * triton.cdiv(m, bm) * sp  # noqa: E731

        # The CONTROL: tune this shape twice, once in each cache regime, and
        # measure both choices under the cold conditions the model runs in.
        # `hot` is what `triton.autotune` did and what shipped until now.
        viable = [c for c in G._CANDIDATES if c[4] == 1 or k // c[4] >= 256]
        hot_cfg = G._resolve(G._bench_all(m, n, k, viable, a.device, cold=False),
                             viable)
        cold_cfg = G._resolve(G._bench_all(m, n, k, viable, a.device, cold=True),
                              viable)
        acfg = hot_cfg[:4]

        cands = {}
        for bn, bk, w, s in ((32, 64, 2, 3), (32, 128, 2, 3), (64, 64, 4, 3),
                             (64, 128, 4, 3), (128, 32, 4, 3), (128, 64, 4, 4),
                             (128, 128, 4, 3), (256, 32, 8, 3)):
            cands[f"{bn}x{bk}"] = ((lambda c: lambda: run_plain(
                x8, cw.next(), sx, sw, m, n, k, c))((bn, bk, w, s)), nblk(bn))
        for sp in (2, 4, 8, 16):
            if k // sp >= 256:
                cands[f"split{sp}"] = ((lambda s: lambda: run_split(
                    x8, cw.next(), sx, sw, m, n, k, s))(sp), nblk(64, sp))
        for cfg in (hot_cfg, cold_cfg):
            nm = f"{cfg[0]}x{cfg[1]}" if cfg[4] == 1 else f"split{cfg[4]}"
            if nm not in cands:
                cands[nm] = ((lambda c: lambda: run_plain(
                    x8, cw.next(), sx, sw, m, n, k, c[:4]))(cfg), nblk(cfg[0]))
        akey = f"{hot_cfg[0]}x{hot_cfg[1]}" if hot_cfg[4] == 1 else f"split{hot_cfg[4]}"

        best = {nm: float("inf") for nm in cands}
        for _ in range(a.reps):                      # interleaved, then min
            for nm, (fn, _) in cands.items():
                best[nm] = min(best[nm], time_min(fn, 1))

        # Exactness is the product. Every tile and every split must be bit-equal.
        ref = run_plain(x8, cw.bufs[0], sx, sw, m, n, k, (128, 64, 4, 4))
        for bn, bk, w, s in ((32, 64, 2, 3), (64, 128, 4, 3), (256, 32, 8, 3)):
            assert torch.equal(run_plain(x8, cw.bufs[0], sx, sw, m, n, k,
                                         (bn, bk, w, s)), ref), \
                f"{name}: tile {bn}x{bk} disagrees - int32 should make these equal"
        for sp in (2, 4, 8, 16):
            if k // sp >= 256:
                assert torch.equal(run_split(x8, cw.bufs[0], sx, sw, m, n, k, sp),
                                   ref), f"{name}: SPLIT_K={sp} disagrees"

        a_us = best[akey]
        bnm = min(best, key=lambda nm: best[nm])
        b_us = best[bnm]
        pk = lambda us: 100 * (mb / 1e3) / (us / 1e6) / PEAK_GBS  # noqa: E731
        print(f"{name:<11}{n:>7}{k:>6}{mb:>7.2f}{akey:>11}{a_us:>8.1f}"
              f"{pk(a_us):>4.0f}%{cands[akey][1]:>5}   {bnm:>11}{b_us:>8.1f}"
              f"{pk(b_us):>4.0f}%{cands[bnm][1]:>5}{a_us / b_us:>6.2f}x")
        mult = 24 * (2 if name in ("k/v_proj", "gate/up") else 1)
        if name == "lm_head":
            mult = 1
        tot_auto += a_us * mult
        tot_best += b_us * mult
        rows.append((name, akey, bnm, a_us / b_us))

    print(f"\n  per decode step: hot-tuned {tot_auto:.0f} us, "
          f"cold-tuned {tot_best:.0f} us  ->  {tot_auto / tot_best:.2f}x on the "
          f"GEMM alone")
    print("  both columns are measured COLD; only the TUNING regime differs, "
          "so this is the A/B for the whole finding")
    moved = [r for r in rows if r[1] != r[2]]
    if moved:
        print("  the cold choice differs from the hot one on: "
              + ", ".join(f"{n} ({x}->{y}, {g:.2f}x)" for n, x, y, g in moved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
