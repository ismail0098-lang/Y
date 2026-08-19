"""Prove the FAST exact path equals the SLOW exact path, bit for bit.

    python3 tools/exact_selftest.py

`tools/exact_throughput.py` reports a speedup. A speedup is only interesting if
the fast path computes the same function, and here "the same function" is not
approximate: both paths are exact integer arithmetic, so anything other than
bitwise equality is a bug, not a tolerance question. This file is what lets the
optimisation be believed.

The reference implementations below are deliberate transcriptions of the code
as it stood *before* the optimisation - float64 everywhere, K/V widened with
`repeat_interleave`, one matmul for `p @ V`. They are slow on purpose.

## Every check has a control

The repo's own history is full of tests that passed with the mechanism deleted
(the shared-memory barrier whose race never fired; the fuzzer that could not
fail; the circom-only test that passed with the boundary remap removed). So
each property here is paired with a mutation that must BREAK it. A check that
cannot fail is not evidence.
"""
import _nospace

_nospace.guard()

import math  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import batch_invariance_demo as D  # noqa: E402
import exact_kv  # noqa: E402
import exact_model as EM  # noqa: E402

try:
    import triton  # noqa: E402
    import w8a8_gemv as w8a8  # noqa: E402
except Exception:  # noqa: BLE001
    w8a8 = triton = None

LN2_Q30 = 744261118
RECIP6_Q32 = 715827883


# ---------------------------------------------------------------- references
def ref_table(dev):
    return torch.tensor([round((1 << 30) * 2 ** (-i / 64)) for i in range(64)],
                        dtype=torch.float64, device=dev)


def ref_exp2(t, table):
    """The float64 formulation, kept verbatim as the oracle for the int64 one."""
    t = t.clamp(min=0)
    n = torch.div(t, 65536, rounding_mode="floor")
    f = t - n * 65536
    idx = torch.div(f, 1024, rounding_mode="floor").clamp(0, 63)
    base = table[idx.long()]
    d = f - idx * 1024
    y = torch.div(d * LN2_Q30, 65536, rounding_mode="floor")
    y2 = torch.div(y * y, 1 << 30, rounding_mode="floor")
    y3 = torch.div(y2 * y, 1 << 30, rounding_mode="floor")
    y3o6 = torch.div(y3 * RECIP6_Q32, 1 << 32, rounding_mode="floor")
    corr = (1 << 30) - y + torch.div(y2, 2, rounding_mode="floor") - y3o6
    g = torch.div(base * corr, 1 << 30, rounding_mode="floor")
    sh = (2 + n).clamp(max=62)
    two = torch.tensor(2.0, dtype=torch.float64, device=t.device)
    out = torch.div(g + torch.pow(two, sh - 1), torch.pow(two, sh),
                    rounding_mode="floor")
    return torch.where(n >= 30, torch.zeros_like(out), out)


def ref_attention(query, key, value, mask, scaling=None):
    """Pre-optimisation exact attention: float64 throughout, K/V expanded."""
    dev = query.device
    table = ref_table(dev)
    b, nh, q_len, d = query.shape
    nkv = key.shape[1]
    rep = nh // nkv
    sc = scaling if scaling is not None else 1.0 / math.sqrt(d)
    # Quantisation happens in float32 (see exact_kv.quantize_rows); only the
    # logit/exp arithmetic below is carried in double.
    k = key.repeat_interleave(rep, dim=1)
    v = value.repeat_interleave(rep, dim=1)
    q = query

    def qt(x, dims, levels):
        # `levels` tracks the shipping widths: the reference's job is "the same
        # function, computed in float64", so hardcoding 127 here would make the
        # test pass only at the default width and report a *width change* as a
        # correctness failure.
        s = x.abs().amax(dim=dims, keepdim=True) / float(levels)
        s = torch.where(s == 0, torch.ones_like(s), s)
        return torch.round(x / s).clamp(-levels, levels), s

    # A per-CHANNEL helper used to live here, from the pre-per-token scheme. It
    # was dead, and it was worse than dead: a per-channel scale sits inside the
    # dot product (`sum_c q_c k_c s_c`) instead of factoring out of it, so the
    # reduction becomes float-weighted and order-dependent. Per-channel is the
    # one family of quantisation this approach cannot use, so a per-channel
    # quantiser sitting beside the reference is an invitation to a wrong fix.

    qi, sq = qt(q, (-1,), D.Q_LEVELS)              # activations stay int8
    # Derived from head_dim, exactly as the shipping path does - a constant
    # here would make the reference a different function at head_dim 256.
    ki, sk = qt(k, (-1,), exact_kv.k_levels_for(d, D.Q_LEVELS))
    vi, sv = qt(v, (-1,), exact_kv.V_LEVELS)
    sq, sk, sv = sq.double(), sk.double(), sv.double()
    s_int = torch.matmul(qi, ki.transpose(-1, -2)).double()
    qi, ki, vi = qi.double(), ki.double(), vi.double()
    if mask is not None:
        blocked = mask[:, :, :, : k.shape[-2]] < -1.0
    else:
        blocked = torch.zeros_like(s_int, dtype=torch.bool)
    neg = torch.finfo(torch.float64).min
    logit = s_int * (sq * sc * math.log2(math.e) * 65536.0) * sk.transpose(-1, -2)
    m = torch.where(blocked, torch.full_like(logit, neg), logit).amax(dim=-1, keepdim=True)
    tq = torch.round(m - logit)
    tq = torch.where(blocked, torch.full_like(tq, 1 << 30), tq).clamp(0, 1 << 30)
    p = ref_exp2(tq, table)
    p = torch.where(blocked, torch.zeros_like(p), p)
    l = p.sum(dim=-1, keepdim=True)
    w = p * sv.transpose(-1, -2)
    wmax = w.amax(dim=-1, keepdim=True)
    safe = torch.where(wmax > 0, wmax, torch.ones_like(wmax))
    W = torch.round(w * (float(1 << D.P_BITS) / safe)).clamp(0, 1 << D.P_BITS)
    acc = torch.matmul(W, vi)
    out = (acc * (safe / float(1 << D.P_BITS))) / l.clamp(min=1.0)
    return out.to(query.dtype).transpose(1, 2).contiguous()


def ref_linear(lin, x):
    """Pre-optimisation ExactLinear: float64 matmul, no padding, no int_mm."""
    w = lin.weight.data
    sw = w.abs().amax(-1, keepdim=True) / 127.0
    sw = torch.where(sw == 0, torch.ones_like(sw), sw)
    w8 = torch.round(w / sw).clamp(-127, 127).to(torch.int8)
    xf = x.reshape(-1, x.shape[-1])
    s = xf.abs().amax(-1, keepdim=True) / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    x8 = torch.round(xf / s).clamp(-127, 127).to(torch.int8)
    acc = torch.matmul(x8.double(), w8.double().t()).float()
    y = acc * s.float() * sw.squeeze(-1).float().unsqueeze(0)
    if lin.bias is not None:
        y = y + lin.bias.data
    return y.reshape(*x.shape[:-1], lin.out_features).to(x.dtype)


# ---------------------------------------------------------------- the checks
SHAPES = [(32, 14, 2, 1, 80, 64), (1, 14, 2, 1, 17, 64), (8, 14, 2, 1, 257, 64),
          (4, 14, 2, 13, 13, 64), (32, 14, 2, 1, 1, 64), (2, 16, 16, 1, 64, 64),
          (3, 14, 2, 5, 300, 64), (2, 14, 2, 1, 1100, 64)]


def make(b, nh, nkv, q_len, T, d, dev):
    q = torch.randn(b, nh, q_len, d, device=dev)
    k = torch.randn(b, nkv, T, d, device=dev)
    v = torch.randn(b, nkv, T, d, device=dev)
    mask = None
    if q_len > 1:
        neg = float(torch.finfo(torch.float32).min)
        mask = torch.triu(torch.full((q_len, T), neg, device=dev),
                          diagonal=T - q_len + 1).expand(b, 1, q_len, T).contiguous()
    return q, k, v, mask


def check_attention(dev, fail):
    for shape in SHAPES:
        q, k, v, mask = make(*shape, dev)
        a = ref_attention(q, k, v, mask)
        b_ = D.exact_attention(None, q, k, v, mask, scaling=None)[0]
        if not torch.equal(a, b_):
            fail(f"attention {shape}: max delta {(a - b_).abs().max():.3e}")
    return f"attention == float64 reference on {len(SHAPES)} shapes"


def check_linear(dev, fail):
    n = 0
    for (I, O, bias) in [(896, 4864, False), (896, 896, True), (4864, 896, False),
                         (896, 151936, False)]:
        lin = nn.Linear(I, O, bias=bias).to(dev)
        ex = EM.ExactLinear(lin).to(dev)
        for M in (1, 2, 7, 8, 16, 31, 32, 33, 64, 96, 129):
            x = torch.randn(M, I, device=dev)
            if not torch.equal(ref_linear(lin, x), ex(x)):
                fail(f"linear I={I} O={O} M={M}")
            n += 1
    return f"ExactLinear == float64 reference on {n} shape/size combinations"


def check_exp2(dev, fail):
    t = torch.cat([torch.arange(0, 1 << 21, 37, device=dev),
                   torch.tensor([0, 1, 65535, 65536, 1 << 30], device=dev)]).double()
    a = ref_exp2(t, ref_table(dev))
    b_ = D.exp2_neg_q16_16(t, D._table(dev))
    if not torch.equal(a.long(), b_):
        fail("int64 exp2 disagrees with the float64 recipe")
    if int(b_[t == 0][0]) != 1 << 28:
        fail("exp2(0) is not exactly 2^28")

    # Both sides above are transcriptions of the same recipe in this same
    # file's language: this compares Python with Python and says nothing about
    # the compiler's exp, which is what the PTX kernel actually calls. The
    # digest is the whole domain, and `src/fixed_exp.rs` asserts the same
    # constant, so a drift on either side breaks one of the two.
    if D.exp2_domain_digest(dev) != D.EXP2_DOMAIN_DIGEST:
        fail("the torch exp2 no longer matches src/fixed_exp.rs over the "
             "whole domain -- the demo has stopped computing what the kernel "
             "computes")
    return (f"integer exp2 == float64 recipe on {t.numel()} arguments, and "
            f"== src/fixed_exp.rs over all {31 << 16} of them")


def check_pv(dev, fail):
    for T in (1, 63, 80, 512, 1039, 1040, 4097):
        p = torch.randint(0, (1 << 28) + 1, (3, 2, 5, T), device=dev, dtype=torch.int64)
        p[0, 0, 0, 0] = 1 << 28                       # the exact upper bound
        v = torch.randint(-127, 128, (3, 2, T, 64), device=dev, dtype=torch.int64)
        want = torch.matmul(p.double(), v.double())
        if not torch.equal(want, D.exact_pv(p, v)):
            fail(f"exact_pv T={T} disagrees with the float64 matmul")
    return "exact_pv == float64 matmul at T in {1..4097}, p up to 2^28 inclusive"


def check_pv_bound_is_load_bearing(dev, fail):
    """The control: a digit too wide must actually corrupt the answer.

    `exact_pv` picks `dbits` so each digit matmul stays inside f32's exact
    range. If that choice were decorative - if f32 happened to be accurate
    enough anyway - then the check above would pass with the derivation
    removed, and would be proving nothing. Force a too-wide digit and confirm
    the result really does break.
    """
    T = 4097
    p = torch.randint(0, 1 << 28, (2, 1, 4, T), device=dev, dtype=torch.int64)
    v = torch.randint(-127, 128, (2, 1, T, 64), device=dev, dtype=torch.int64)
    want = torch.matmul(p.double(), v.double())
    vf = v.to(torch.float32)
    acc, shift, dbits = None, 0, 14           # deliberately past the 2^24 bound
    while shift < 29:
        digit = ((p >> shift) & ((1 << dbits) - 1)).to(torch.float32)
        r = torch.matmul(digit, vf).to(torch.float64) * float(1 << shift)
        acc = r if acc is None else acc + r
        shift += dbits
    if torch.equal(want, acc):
        fail("a 14-bit digit did NOT corrupt p@V - the dbits derivation is "
             "not what makes exact_pv exact, so the check above is vacuous")
    return "control: a too-wide digit does corrupt p@V (the bound is real)"


def check_pad_is_load_bearing(dev, fail):
    """The control for ExactLinear: _int_mm must really refuse the raw shape.

    The padding exists because `torch._int_mm` accepts only M divisible by 32.
    If it silently accepted M=1 on some future torch, the padding would be dead
    code and the measured speedup would have a different cause than the one
    written down.
    """
    w = torch.randint(-127, 127, (896, 4864), device=dev, dtype=torch.int8).contiguous()
    x = torch.randint(-127, 127, (1, 896), device=dev, dtype=torch.int8)
    try:
        torch._int_mm(x, w)
    except Exception:
        return "control: torch._int_mm still refuses M=1, so the padding is load-bearing"
    fail("torch._int_mm now accepts M=1; the padding path is dead code and the "
         "comment explaining the speedup is stale")


def check_quantiser_at_volume(dev, fail):
    """The Triton quantiser must match `torch.round(x/s)` over MILLIONS of rows.

    This is a volume test on purpose. The Triton kernel originally used the
    default float divide, which is the *approximate* one, and it disagreed with
    IEEE `div.rn` on **one element in 458,752**: `x/s` came out as exactly
    -22.5 where `div.rn` gives -22.5000019, so round-half-to-even answered -22
    against torch's -23. `check_linear` above never sees it - its largest case
    is 129x896 - so a decode-shaped test suite would have shipped it.

    The reason a 1-ulp divide error matters at all is that quantisation is a
    STEP function: it does not stay 1 ulp, it becomes a whole int8 level, and
    from there a different token. `tl.fdiv(..., ieee_rounding=True)` does NOT
    fix it (verified); `libdevice.div_rn` does.
    """
    if w8a8 is None:
        return "SKIP: triton kernel not importable"
    total = 0
    for (M, K) in ((512, 896), (2048, 896), (2048, 4864)):
        x = torch.randn(M, K, device=dev)
        s = x.abs().amax(-1, keepdim=True) / 127.0
        s = torch.where(s == 0, torch.ones_like(s), s)
        want = torch.round(x / s).clamp(-127, 127).to(torch.int8)
        got = torch.empty((M, K), dtype=torch.int8, device=dev)
        sx = torch.empty((M,), dtype=torch.float32, device=dev)
        w8a8._quant_rows[(M,)](x, got, sx, M, K, x.stride(0),
                               BLOCK_K=min(1024, triton.next_power_of_2(K)))
        n = int((got != want).sum())
        if n:
            fail(f"quantiser differs on {n}/{M * K} elements at M={M} K={K}")
        if not torch.equal(sx, s.squeeze(-1)):
            fail(f"row scales differ at M={M} K={K}")
        total += M * K
    return f"Triton quantiser == torch.round(x/s) on {total:,} elements"


def check_bias_is_not_fused(dev, fail):
    """A biased layer must round TWICE, not once.

    `acc * sx * sw + bias` inside the Triton kernel was contracted into an FMA,
    which rounds once where the reference rounds twice - 25% of elements off by
    1 ulp. `libdevice` exposes no `fadd_rn`, so the add is done in torch
    instead. Only a layer WITH a bias can catch this: with no add there is
    nothing to contract, which is why every no-bias shape passed while the
    q/k/v projections were wrong.
    """
    torch.manual_seed(3)
    for (K, N, M) in ((896, 896, 1), (896, 896, 32), (896, 4864, 32)):
        lin = nn.Linear(K, N, bias=True).to(dev)
        ex = EM.ExactLinear(lin).to(dev)
        x = torch.randn(M, K, device=dev)
        a, b_ = ref_linear(lin, x), ex(x)
        if not torch.equal(a, b_):
            fail(f"biased layer K={K} N={N} M={M}: "
                 f"{int((a != b_).sum())}/{a.numel()} elements differ")
    return "biased layers round twice (no FMA contraction) at 3 shapes"


def check_all_gemm_configs_agree(dev, fail):
    """Every tuner candidate must produce the IDENTICAL bits.

    This is what makes tuning the W8A8 GEMM legitimate, and it would not be
    legitimate for a float GEMM. The accumulator is int32 and integer addition
    is associative and commutative, so tile shape, K-split and CTA completion
    order cannot change the answer. Without this the tuner would be free to pick
    a different *function* on a different machine, which is precisely the
    non-determinism the project exists to remove - a model whose output depends
    on what a benchmark happened to measure at startup.

    The candidate list now contains SPLIT_K entries as well as tiles, so this
    covers the atomic kernel too - and the atomic path is the one where the
    claim has teeth, because `tl.atomic_add` genuinely does complete in a
    nondeterministic ORDER. int32 is what makes that order not matter; the same
    test over a float accumulator would be expected to fail.
    """
    if w8a8 is None:
        return "SKIP: triton kernel not importable"
    torch.manual_seed(5)
    runs = 0
    for (K, N) in ((896, 896), (896, 128), (896, 4864), (4864, 896)):
        for M in (1, 32):
            x8 = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8)
            w8t = torch.randint(-127, 127, (K, N), device=dev,
                                dtype=torch.int8).contiguous()
            sx = torch.rand(M, device=dev)
            sw = torch.rand(N, device=dev)
            ref = None
            bm = min(64, max(16, triton.next_power_of_2(M)))
            for bn, bk, nw, ns, sp in w8a8._CANDIDATES:
                if sp > 1 and K // sp < 256:
                    continue
                if sp == 1:
                    y = torch.empty((M, N), dtype=torch.float32, device=dev)
                    w8a8._w8a8_gemm[(triton.cdiv(N, bn), triton.cdiv(M, bm))](
                        x8, w8t, sx, sw, y, M, N, K, BLOCK_M=bm, BLOCK_N=bn,
                        BLOCK_K=bk, num_warps=nw, num_stages=ns)
                else:
                    acc = torch.zeros((M, N), dtype=torch.int32, device=dev)
                    w8a8._w8a8_gemm_splitk[
                        (triton.cdiv(N, bn), triton.cdiv(M, bm), sp)](
                        x8, w8t, acc, M, N, K, BLOCK_M=bm, BLOCK_N=bn,
                        BLOCK_K=bk, SPLIT_K=sp, num_warps=nw, num_stages=ns)
                    y = acc.to(torch.float32) * sx[:, None] * sw[None, :]
                runs += 1
                if ref is None:
                    ref = y.clone()
                elif not torch.equal(ref, y):
                    fail(f"K={K} N={N} M={M} cfg={bn}x{bk}/split{sp} differs by "
                         f"{(ref - y).abs().max():.3e} - the tuner can change "
                         f"the answer, so tuning is UNSOUND here")
    return (f"all {len(w8a8._CANDIDATES)} GEMM candidates (tiles + K-splits) "
            f"agree bit-for-bit ({runs} runs)")


def check_digit_split_is_exact(dev, fail):
    """`q == hi*128 + lo` over EVERY representable q, both digits fitting int8.

    Exhaustive rather than sampled: the domain is `[-a_levels, a_levels]`, a few
    thousand values, and the risky half is the negative one - two's-complement
    shift-and-mask is "obviously" exact right up until it is not. The `U32`/`I32`
    literal bug in the PTX backend was the same species of obvious.
    """
    lv = EM.act_levels_for(4864, requested=511)   # the wide path, not the default
    q = torch.arange(-lv, lv + 1, dtype=torch.int32, device=dev)
    hi, lo = EM.split_digits(q, lv)
    rebuilt = hi.to(torch.int32) * 128 + lo.to(torch.int32)
    if not torch.equal(rebuilt, q):
        n = int((rebuilt != q).sum())
        fail(f"digit split is lossy for {n} of {q.numel()} values")
    if int(hi.min()) < -128 or int(hi.max()) > 127 or int(lo.min()) < 0:
        fail(f"digits escape int8: hi [{int(hi.min())}, {int(hi.max())}], "
             f"lo [{int(lo.min())}, {int(lo.max())}]")
    return (f"q == hi*128 + lo exactly over all {q.numel()} values of q "
            f"(a_levels {lv}, hi in [{int(hi.min())}, {int(hi.max())}])")


def check_hilo_gemm_configs_agree(dev, fail):
    """The wide-activation path must be config-invariant too, and match int32.

    Two claims, and the second is what makes the first worth anything: every
    tuner candidate agrees bit for bit, AND the shared answer is the exact
    integer product `128*(hi@w) + (lo@w)`. Agreement alone would be satisfied by
    eleven identically wrong kernels - the same reason the MSM tests carry a
    perturbation control rather than only an oracle comparison.
    """
    if w8a8 is None or not hasattr(w8a8, "_w8a8_gemm_hilo"):
        return "SKIP: triton kernel not importable"
    torch.manual_seed(11)
    lv = EM.act_levels_for(4864, requested=511)   # the wide path, not the default
    runs = 0
    for (K, N) in ((896, 896), (896, 4864), (4864, 896)):
        for M in (1, 32):
            q = torch.randint(-lv, lv + 1, (M, K), device=dev, dtype=torch.int32)
            hi, lo = EM.split_digits(q, lv)
            hi, lo = hi.contiguous(), lo.contiguous()
            w8t = torch.randint(-127, 127, (K, N), device=dev,
                                dtype=torch.int8).contiguous()
            sx = torch.rand(M, device=dev)
            sw = torch.rand(N, device=dev)
            ref_i = torch._int_mm(hi, w8t) * 128 + torch._int_mm(lo, w8t) \
                if M % 32 == 0 else None
            ref = None
            bm = min(64, max(16, triton.next_power_of_2(M)))
            for bn, bk, nw, ns, sp in w8a8._CANDIDATES:
                if sp > 1 and K // sp < 256:
                    continue
                if sp == 1:
                    y = torch.empty((M, N), dtype=torch.float32, device=dev)
                    w8a8._w8a8_gemm_hilo[(triton.cdiv(N, bn), triton.cdiv(M, bm))](
                        hi, lo, w8t, sx, sw, y, M, N, K, BLOCK_M=bm, BLOCK_N=bn,
                        BLOCK_K=bk, num_warps=nw, num_stages=ns)
                else:
                    acc = torch.zeros((M, N), dtype=torch.int32, device=dev)
                    w8a8._w8a8_gemm_hilo_splitk[
                        (triton.cdiv(N, bn), triton.cdiv(M, bm), sp)](
                        hi, lo, w8t, acc, M, N, K, BLOCK_M=bm, BLOCK_N=bn,
                        BLOCK_K=bk, SPLIT_K=sp, num_warps=nw, num_stages=ns)
                    if ref_i is not None and not torch.equal(acc, ref_i):
                        fail(f"K={K} N={N} M={M} split{sp}: hi/lo int32 product "
                             f"disagrees with torch._int_mm")
                    y = acc.to(torch.float32) * sx[:, None] * sw[None, :]
                runs += 1
                if ref is None:
                    ref = y.clone()
                elif not torch.equal(ref, y):
                    fail(f"K={K} N={N} M={M} cfg={bn}x{bk}/split{sp} differs - "
                         f"the wide-activation path is not config-invariant")
            if ref_i is not None:
                want = ref_i.to(torch.float32) * sx[:, None] * sw[None, :]
                if not torch.equal(ref, want):
                    fail(f"K={K} N={N} M={M}: every candidate agrees but they "
                         f"all disagree with the exact integer product")
    return (f"wide-activation GEMM: all candidates agree AND match the exact "
            f"int32 product ({runs} runs, a_levels {lv})")


def check_kv_cache_matches_stateless(dev, fail):
    """Appending to the quantised KV cache must equal quantising from scratch.

    This is the whole justification for `tools/exact_kv.py`: because the scales
    are per token, a row's quantisation depends on that row alone, so a cache
    appended to over 40 decode steps must be bit-identical to one built in one
    shot at step 40. `module=None` takes the stateless path, which is what makes
    the two comparable at all.

    It also pins the invalidation rules. A cache that silently kept stale rows
    when the batch shape changed, or when `T` went backwards at the start of a
    new sequence, would still produce plausible attention output.
    """
    import exact_kv
    was, exact_kv.USE_CACHE = exact_kv.USE_CACHE, True   # off by default; see exact_kv
    try:
        return _kv_cache_body(dev, fail)
    finally:
        exact_kv.USE_CACHE = was


def _kv_cache_body(dev, fail):
    torch.manual_seed(11)
    nh, nkv, d = 14, 2, 64
    for b in (1, 4):
        mod = types.SimpleNamespace()               # something to cache on
        k_all = torch.randn(b, nkv, 40, d, device=dev)
        v_all = torch.randn(b, nkv, 40, d, device=dev)
        for t in range(1, 41):
            q = torch.randn(b, nh, 1, d, device=dev)
            k, v = k_all[:, :, :t], v_all[:, :, :t]
            inc = D.exact_attention(mod, q, k, v, None)[0]
            fresh = D.exact_attention(None, q, k, v, None)[0]
            if not torch.equal(inc, fresh):
                fail(f"b={b} step {t}: cached path differs from stateless, "
                     f"max {(inc - fresh).abs().max():.3e}")
                break
        # the batch shape changed -> the cache must rebuild, not reuse
        q2 = torch.randn(b + 2, nh, 1, d, device=dev)
        k2 = torch.randn(b + 2, nkv, 9, d, device=dev)
        v2 = torch.randn(b + 2, nkv, 9, d, device=dev)
        if not torch.equal(D.exact_attention(mod, q2, k2, v2, None)[0],
                           D.exact_attention(None, q2, k2, v2, None)[0]):
            fail(f"cache did not invalidate when the batch went {b} -> {b + 2}")
        # T went backwards (a new sequence) -> must rebuild
        k3, v3 = k_all[:, :, :3], v_all[:, :, :3]
        q3 = torch.randn(b, nh, 1, d, device=dev)
        if not torch.equal(D.exact_attention(mod, q3, k3, v3, None)[0],
                           D.exact_attention(None, q3, k3, v3, None)[0]):
            fail("cache did not invalidate when T went backwards")
    return "quantised KV cache == stateless quantisation over 80 appends + 4 resets"


def check_batch_invariance(dev, fail):
    """The property the whole project is about, on the fast path specifically.

    Row 0 of a batch of identical rows must equal the batch-of-one answer, for
    every batch size. This is not implied by agreeing with the float64
    reference - the reference could be batch-dependent too.
    """
    torch.manual_seed(7)
    nh, nkv, d, T = 14, 2, 64, 96
    q1 = torch.randn(1, nh, 1, d, device=dev)
    k1 = torch.randn(1, nkv, T, d, device=dev)
    v1 = torch.randn(1, nkv, T, d, device=dev)
    base = D.exact_attention(None, q1, k1, v1, None)[0]
    for b in (2, 3, 8, 17, 32, 64):
        o = D.exact_attention(None, q1.repeat(b, 1, 1, 1), k1.repeat(b, 1, 1, 1),
                              v1.repeat(b, 1, 1, 1), None)[0]
        if not torch.equal(base[0], o[0]):
            fail(f"row 0 changed at batch {b}: {(base[0] - o[0]).abs().max():.3e}")
    return "attention row 0 is bit-identical at batch 1, 2, 3, 8, 17, 32, 64"


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    dev = "cuda"
    bad = []
    checks = [check_exp2, check_pv, check_pv_bound_is_load_bearing,
              check_linear, check_bias_is_not_fused, check_quantiser_at_volume,
              check_pad_is_load_bearing, check_all_gemm_configs_agree,
              check_digit_split_is_exact, check_hilo_gemm_configs_agree,
              check_attention,
              check_kv_cache_matches_stateless, check_batch_invariance]
    for fn in checks:
        errs = []
        try:
            msg = fn(dev, errs.append)
        except Exception as e:                       # noqa: BLE001
            errs.append(f"{type(e).__name__}: {e}")
            msg = fn.__name__
        if errs:
            bad.extend(errs)
            print(f"  FAIL  {fn.__name__}")
            for e in errs[:4]:
                print(f"        {e}")
        else:
            print(f"  ok    {msg}")
    print()
    if bad:
        print(f"{len(bad)} FAILURES - the fast path is not the same function")
        return 1
    print("The optimised exact path computes exactly the same function as the "
          "float64 reference,\nand is batch-invariant on its own account.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
