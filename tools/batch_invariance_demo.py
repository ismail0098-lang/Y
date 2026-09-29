"""M5: the demo. Same prompt, different batch size, same tokens.

Run:
    python3 tools/batch_invariance_demo.py

## What is being shown

A real model (Qwen2.5-0.5B-Instruct) decodes the same prompt twice: once alone
(batch 1) and once as row 0 of a batch of N identical copies. Mathematically
those are the same computation, so any difference is *purely* the kernel's
reduction order changing with the batch size. That is the bug this project
exists to remove, and it is why an eval score moves when the batch size does.

Two attention implementations are compared:

  * **stock** - the model's own f32 attention (the control).
  * **exact** - scores as exact integer dot products over int8 Q/K, a global
    max, softmax weights in Q0.28 via the integer `exp2` of `src/fixed_exp.rs`,
    and the p*V accumulation carried exactly.

## Why floating point here is still "exact"

torch has no batched CUDA integer matmul, so the reductions are carried in
float32 and float64. That is not a weakening, because **every value involved is
an integer**: a float type represents every integer below 2^24 (f32) or 2^53
(f64) exactly, so if the bound holds then no partial sum ever rounds, and a
computation with no rounding in it gives the identical answer in every
summation order. That is the same associativity argument as the integer
kernels, carried by a different type - and the bounds are *derived* from shapes
and dtypes rather than sampled, so they hold for every input rather than for
the tensor that happened to be measured (`assert_bound`; `Y_EXACT_CHECK=1`
audits the derivation against reality).

Where the bound does not hold it is not ignored: the `p @ V` product needs 35
bits before it is summed, so `exact_pv` splits `p` into digits narrow enough
that each partial matmul fits f32 exactly, and recombines them in f64.

The PTX kernels in `src/exact_attention.rs` are the real implementation and use
int64 directly; `tools/ptx_bridge.py` checks this path against them bit for bit
on real activations. This script exists to show the property end to end on a
real model. It is not the fast path, but it is no longer gratuitously slow -
see `tools/exact_throughput.py`.
"""
import argparse
import math
import os
import sys

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2
from transformers import AutoModelForCausalLM, AutoTokenizer

import exact_model
from exact_kv import V_LEVELS, fold_scale_into_weights, quantize_kv

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
P_BITS = 28  # Q0.28 softmax weights, derived in tests/attention_quantization_error.rs
# Query width. Named rather than written as 127 at each site because it appears
# in the quantiser AND in two exactness bounds, and a bound that disagrees with
# the computation it bounds is the failure this file has already had once
# (`quantize_rows` silently giving V the K width). Q is not cached, so its width
# is free in memory - but the score budget is shared: exactness needs
# `d * Q_LEVELS * k_lv < 2^24`, and `exact_kv.k_levels_for` spends whatever
# Q leaves - so widening Q directly narrows K. Q is not cached, so its width
# is free in memory, but it is not free in that budget.
Q_LEVELS = 127

# ---- the integer exp, same recipe as src/fixed_exp.rs ----------------------
LN2_Q30 = 744261118
RECIP6_Q32 = 715827883

# FNV-1a over the whole domain `0 .. 31 << 16`, pinned as
# `fixed_exp::EXP2_DOMAIN_DIGEST` in src/fixed_exp.rs and asserted by
# `fixed_exp::tests::the_whole_domain_digest_is_the_cross_language_anchor`.
#
# There are four transcriptions of this recipe: the Rust one, the PTX one, a
# pure-Python one in `attention_real_activations.py`, and this one -- and this
# is the one the demo, `exact_accuracy.py` and `ptx_bridge.py`'s reference arm
# all call. The PTX one was checked against Rust; the two Python ones were
# checked only against float64 transcriptions of THEMSELVES, i.e. Python
# against Python.
#
# What those checks assert is "within 1 ulp of 2^28 * 2^-t, and monotonic".
# Those are properties, not identity: two implementations can both be within
# 1 ulp of the truth and differ from each other by 1, and 1 ulp is a BIT. For
# a project whose entire claim is bit-identity that is the only failure that
# matters, and it was the one failure the check could not see.
EXP2_DOMAIN_DIGEST = 0x7170CD39B442D506


def _table(device):
    return torch.tensor(
        [round((1 << 30) * 2 ** (-i / 64)) for i in range(64)],
        dtype=torch.int64, device=device,
    )


def exp2_neg_q16_16(t, table):
    """Vectorised `2^-t` for Q16.16 `t`, returning Q0.28. Integer throughout.

    Carried in int64 rather than float64. Every intermediate here is an integer
    below 2^60, so both types represent it exactly and the two agree bit for
    bit - but float64 arithmetic runs at 1/64 rate on a GeForce part, and this
    chain is a dozen elementwise ops over the whole score tensor.
    """
    t = t.to(torch.int64).clamp(min=0)
    n = t >> 16
    f = t & 0xFFFF
    idx = (f >> 10).clamp(0, 63)
    base = table[idx]
    d = f & 0x3FF
    y = (d * LN2_Q30) >> 16
    y2 = (y * y) >> 30
    y3 = (y2 * y) >> 30
    y3o6 = (y3 * RECIP6_Q32) >> 32
    corr = (1 << 30) - y + (y2 >> 1) - y3o6
    g = (base * corr) >> 30
    sh = (2 + n).clamp(max=62)
    one = torch.ones_like(sh)
    out = (g + (one << (sh - 1))) >> sh
    return torch.where(n >= 30, torch.zeros_like(out), out)


def exp2_domain_digest(device="cpu"):
    """FNV-1a of this replica over every argument the exp accepts.

    Whole-domain, so it cannot be satisfied by a spot check, and ~0.25 s. The
    fold is sequential and the values come back to the host for it; that is
    fine, this runs once.
    """
    table = _table(device)
    h, mask = 0xCBF29CE484222325, (1 << 64) - 1
    for lo in range(0, 31 << 16, 1 << 19):
        hi = min(lo + (1 << 19), 31 << 16)
        t = torch.arange(lo, hi, dtype=torch.int64, device=device)
        for v in exp2_neg_q16_16(t, table).tolist():
            h = ((h ^ v) * 0x00000100000001B3) & mask
    return h


def digit_width(T, v_levels=None):
    """Widest base-2^dbits digit whose matmul is still exact in fp32, or 0.

    Split out of `exact_pv` so `tools/exact_bounds_check.py` can exhaust the
    REAL selector rather than a transcription of it. A checker that
    re-implements the rule it is checking only ever proves I can copy a line
    twice - the same reason the ZK oracle is written against an independent IR
    rather than against Y's own AST.
    """
    lv = V_LEVELS if v_levels is None else v_levels
    dbits = 8
    while dbits > 0 and T * ((1 << dbits) - 1) * lv >= (1 << 24):
        dbits -= 1
    return dbits


_Y_PV = {}


# Registered as a CUSTOM OP, not called directly, and the reason is measured.
#
# The first version called `_y_exact_pv` straight from `exact_pv`. It was
# correct and it destroyed `torch.compile`: a ctypes launch inside `forward` is
# opaque to dynamo, so the model went from **1 graph / 0 breaks to 18 graphs /
# 17 breaks**, and compiled throughput went from 240.3 tok/s/seq on the digit
# split to 38.5 -- i.e. compilation bought the kernel path nothing at all, and
# a 1.068x win in eager was a 6.2x LOSS in the configuration anyone would ship.
#
# `feedback_stateful_optimisations_lose_to_compilers` recorded this exact shape
# once already, for an O(1) KV cache that was 550x slower under compile. A
# custom op with a registered fake lets dynamo keep ONE graph: it sees an
# opaque node with known shape and dtype and traces straight past it.
@torch.library.custom_op("y::exact_pv", mutates_args=())
def _y_pv_op(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _y_exact_pv(p, v)


@_y_pv_op.register_fake
def _(p, v):
    return torch.empty(
        (*p.shape[:-1], v.shape[-1]), dtype=torch.float64, device=p.device
    )


def _y_exact_pv(p, v):
    """`p @ v` through tests/exact_pv.ysu -- one launch, int64 accumulation.

    `p` is [..., Q, T] and `v` is [..., T, D]; the leading dimensions flatten
    into the grid's `b`. Returns float64, matching the digit split, so callers
    and their `assert_exact_range` checks are unchanged.

    Loads the PTX once per process. If it cannot be loaded this RAISES rather
    than falling back to the digit split: a fallback would silently reinstate
    the whole-cache cast and quietly halve the throughput this path exists for,
    and a silent fallback to a slower-but-correct route is exactly the shape
    CLAUDE.md's design rule is about -- the build stays green and the claim
    stops being true.
    """
    import ptx_bridge as PB

    dev = p.device
    if "fn" not in _Y_PV:
        import subprocess
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        ybin = repo / "target" / "release" / "Y"
        if not ybin.exists():
            raise RuntimeError(
                f"Y_EXACT_V_INT8 needs the compiler at {ybin}; build it with "
                "`cargo build --release`. Not falling back to the digit split: "
                "that would cast the whole cache every step and report a "
                "number for a path nobody asked for."
            )
        torch.zeros(1, device=dev)          # torch makes its context lazily
        subprocess.run([str(ybin), str(repo / "tests" / "exact_pv.ysu"),
                        "--emit-ptx"], capture_output=True, check=True, cwd=repo)
        ptx = (repo / "tests" / "exact_pv.ptx").read_text()
        _Y_PV["mod"] = PB.Module(ptx)
        _Y_PV["fn"] = _Y_PV["mod"].fn("exact_pv")

    import ctypes
    lead = p.shape[:-2]
    B = 1
    for x in lead:
        B *= x
    Qn, T = p.shape[-2], p.shape[-1]
    Dh = v.shape[-1]
    assert v.shape[-2] == T, f"key length disagrees: p has {T}, v has {v.shape[-2]}"
    p32 = p.reshape(B, Qn, T).to(torch.int32).contiguous()
    v8 = v.reshape(B, T, Dh).contiguous()
    out = torch.empty(B * Qn * Dh, dtype=torch.int64, device=dev)
    PB.launch(_Y_PV["fn"], (Qn, B, 1), (Dh, 1, 1),
              [PB.dptr(p32), PB.dptr(v8), PB.dptr(out),
               ctypes.c_uint(T), ctypes.c_uint(Dh), ctypes.c_uint(Qn),
               ctypes.c_uint(B * Qn * T), ctypes.c_uint(B * T * Dh),
               ctypes.c_uint(B * Qn * Dh)])
    return out.view(*lead, Qn, Dh).to(torch.float64)


def exact_pv(p, v):
    """`p @ v` with p a non-negative integer < 2^29 and v an int8 value, exact.

    This is the one product in the network that does not fit fp32's 24-bit
    mantissa: a Q0.28 weight times an int8 activation is already 35 bits before
    anything is summed. The prototype therefore ran it in float64, which
    measured **949 us** against fp32's 28.6 us at batch 32 - 33x, and the
    single largest line in the profile.

    Splitting `p` into base-2^`dbits` digits fixes it without giving up a bit.
    Each digit matmul has terms bounded by `(2^dbits - 1) * V_LEVELS` and
    partial sums bounded by `T` times that; choose `dbits` so the bound clears
    2^24 and every one of those matmuls is *exactly* representable in fp32,
    hence has no rounding and no order dependence. The digits recombine in
    float64, where the total (at most `T * 2^28 * V_LEVELS`) is far below 2^53.

    **The bound is computed from `V_LEVELS`, not from the literal 127.** A wider
    V forces narrower digits and therefore MORE matmuls (8 instead of 5 at 10
    bits), which is exactly the cost of buying accuracy on the V side; writing
    the constant here instead would keep the digit count and silently start
    rounding.
    """
    # An int8 `v` means the cache is storing V as int8 (`Y_EXACT_V_INT8`), and
    # the ONLY correct route for that is the compiler's kernel: converting it
    # to float32 here would reinstate the O(T) whole-cache cast that storing
    # int8 exists to delete, and would do it silently. Refusing is the fix if
    # the kernel cannot be loaded -- see `_y_exact_pv`.
    if v.dtype == torch.int8:
        return torch.ops.y.exact_pv(p, v)
    T = v.shape[-2]
    dbits = digit_width(T)
    if dbits == 0:                      # absurdly long context; stay correct
        return torch.matmul(p.to(torch.float64), v.to(torch.float64))
    vf = v.to(torch.float32)
    acc = None
    shift = 0
    # `P_BITS + 1`, not a literal 29: `p` can be exactly 2^P_BITS, so the top
    # bit needs one more position than P_BITS names. It was written `29` -- a
    # transcription of a module constant into the one loop whose job is to
    # cover it, so raising P_BITS would have left this silently dropping the
    # top bit while every bound in exact_bounds_check still held. The guard
    # that catches a short loop is `differential_exact_pv`, which drives p to
    # exactly 2^P_BITS; the formula check that looked like it covered this
    # could not fire (see `digit_violation`).
    while shift < P_BITS + 1:
        digit = ((p >> shift) & ((1 << dbits) - 1)).to(torch.float32)
        r = torch.matmul(digit, vf).to(torch.float64)
        r = r * float(1 << shift)       # a power of two: exact in float64
        acc = r if acc is None else acc + r
        shift += dbits
    return acc


EXACT_CHECK = os.environ.get("Y_EXACT_CHECK") == "1"


def assert_bound(bound, what):
    """Check the *provable* bound on a value, from shapes and dtypes alone.

    This replaces a measured `float(x.abs().max()) < 2^53`, which was **61% of
    attention time** - three of them per call, each a device-to-host copy that
    stalls the whole pipeline (1049 us/call with them, 404 us without). It is
    also the stronger check of the two: a bound derived from `T`, `d` and the
    int8 range holds for *every* input, where a measurement only ever spoke for
    the tensor in front of it. `Y_EXACT_CHECK=1` re-enables the measured form
    beside it, so the derivation can still be audited against reality.
    """
    assert bound < 2.0 ** 53, (
        f"{what} can reach {bound:.3e}, past 2^53 where float64 stops "
        f"representing every integer exactly. The order-independence argument "
        f"no longer holds."
    )


def assert_exact_range(x, what, bound=None):
    if bound is not None:
        assert_bound(bound, what)
    if EXACT_CHECK:
        m = float(x.abs().max()) if x.numel() else 0.0
        assert m < 2.0 ** 53, f"{what} measured {m:.3e}, past 2^53"
        if bound is not None:
            assert m <= bound, (
                f"{what} measured {m:.3e} but the derivation promised at most "
                f"{bound:.3e} - the analytic bound is WRONG, not merely loose"
            )


def assert_score_budget(d, q_levels, k_lv):
    """The fp32 score budget, as a callable rather than a bare assert.

    Split out of `exact_attention` for the same reason `digit_width` was, and
    stated there: a checker that re-implements the rule it is checking only
    ever proves the line can be copied twice. `tools/exact_bounds_check.py`
    used to mirror this one -- its own comment said so -- so the check "head_dim
    past the score budget is refused" would have passed with the assert deleted.
    It exhausts this function now.

    A dot product of two int8 vectors of length `d` is bounded by
    `d * q_levels * k_lv`, and so is every partial sum of it, so fp32's 24-bit
    mantissa holds the whole computation without a single rounding -- which is
    what makes the reduction order-independent, and is the same argument the
    int64 kernel uses one level down.
    """
    assert d * q_levels * k_lv < (1 << 24), (
        f"head_dim {d} with Q at {q_levels} and K at {k_lv} levels puts the "
        f"score sum past 2^24; fp32 would start rounding and the reduction would "
        f"stop being order-independent. Max K width here is "
        f"{(1 << 24) // (d * q_levels)}; `k_levels_for` should already have "
        f"capped to it, so reaching this means head_dim is too large for exact "
        f"int8 scores in fp32."
    )


# `P_BITS` is not an independent choice: it is the exp's output scale, and
# `exp2_neg_q16_16(0)` is exactly `1 << P_BITS` by construction. The digest
# above pins the FUNCTION; this pins the constant the bound arithmetic
# downstream reads, so the two cannot drift apart.
assert int(exp2_neg_q16_16(torch.zeros(1, dtype=torch.int64),
                           _table("cpu"))[0]) == 1 << P_BITS, \
    "P_BITS disagrees with the exp's own output scale"


# ---- the exact attention path ---------------------------------------------
_orig = qwen2.eager_attention_forward
TABLE = {}


def exact_attention(module, query, key, value, attention_mask, scaling=None,
                    dropout=0.0, **kw):
    dev = query.device
    if dev not in TABLE:
        TABLE[dev] = _table(dev)
    table = TABLE[dev]

    b, nh, q_len, d = query.shape
    nkv = key.shape[1]
    rep = nh // nkv
    sc = scaling if scaling is not None else 1.0 / math.sqrt(d)

    # K and V stay at `nkv` heads. The prototype called `repeat_interleave` to
    # widen them to `nh` and then quantised the copies, which is `rep`x the
    # work (7x on this model) for tensors that are identical across the group -
    # measured at **2.0 ms per call**, the largest single line in the profile.
    # Sharing K/V across a query group is what the split PTX kernel does too;
    # see the `paged_decode_attention_split` note in CLAUDE.md.
    # NOT widened to float64. Quantisation is an elementwise divide followed by
    # a round; doing it in double cost 5.60x on this GPU (fp64 is 1/64 rate) and
    # bought one differing int8 value in 270,336. See exact_kv.quantize_rows.
    k, v, q = key, value, query

    def q8_per_tensor(x, dims):
        s = x.abs().amax(dim=dims, keepdim=True) / float(Q_LEVELS)
        s = torch.where(s == 0, torch.ones_like(s), s)
        return torch.round(x / s).clamp(-Q_LEVELS, Q_LEVELS), s

    qi, sq = q8_per_tensor(q, (-1,))          # per query vector (float32)
    sq = sq.to(torch.float64)
    # K and V use PER-TOKEN scales, so a row's quantisation depends on that row
    # alone and the cache can be appended to instead of rebuilt. Under the old
    # per-head / per-channel scales every scale could move when a token arrived,
    # which made the work O(T) per decode step - ~29% of decode, at 8.9% of DRAM
    # peak. See tools/exact_kv.py for why the monotone-max trick rescues K and
    # not V, and tools/exact_accuracy.py for what the change costs.
    # The widths come BACK from the quantiser, so the bounds below cannot
    # disagree with the operands they bound. K's is derived from head_dim.
    ki, sk, vi, sv, k_lv, v_lv = quantize_kv(module, k, v, Q_LEVELS)
    tk = ki.shape[-2]

    # Exact integer scores, in fp32 rather than fp64. A dot product of two int8
    # vectors of length d is bounded by d*127^2 = 1.03e6 here, and so is every
    # partial sum of it, so fp32's 24-bit mantissa holds the whole computation
    # without a single rounding - which is what makes it order-independent, and
    # is the same argument the int64 kernel uses one level down.
    assert_score_budget(d, Q_LEVELS, k_lv)
    # ... and the query group is folded into the row dimension instead of K
    # being broadcast into it, so this is one matmul over `nkv` heads.
    qg = qi.reshape(b, nkv, rep * q_len, d).contiguous()
    s_int = torch.matmul(qg, ki.transpose(-1, -2))
    # Back to float64 for the logit and the exp argument: those are elementwise
    # over a tensor `rep`x smaller than the KV cache, so the fp64 rate does not
    # bite, and the softmax argument is where precision actually matters.
    s_int = s_int.reshape(b, nh, q_len, tk).to(torch.float64)
    # Per-token scales arrive as [b, nkv, T, 1]; the scores want them laid out
    # against the KEY axis, i.e. [b, nh, 1, T].
    sk = sk.transpose(-1, -2).repeat_interleave(rep, dim=1).to(torch.float64)
    sv = sv.transpose(-1, -2).repeat_interleave(rep, dim=1).to(torch.float64)
    # |score| <= d * Q_LEVELS * k_lv, from the two operands' widths.
    # Provable, not sampled - and derived from the widths so it tracks them.
    assert_exact_range(s_int, "integer scores",
                       float(d) * Q_LEVELS * k_lv)

    if attention_mask is not None:
        mask = attention_mask[:, :, :, : k.shape[-2]]
        blocked = mask < -1.0
    else:
        blocked = torch.zeros_like(s_int, dtype=torch.bool)

    # The score's scale is now per KEY, so it cannot be deferred to a single
    # `kfix` multiply after the max - the logit has to be formed first. This is
    # elementwise and the max is exact, so both stay order-independent.
    logit = s_int * (sq * sc * math.log2(math.e) * 65536.0) * sk
    neg = torch.finfo(torch.float64).min
    m = torch.where(blocked, torch.full_like(logit, neg), logit).amax(dim=-1, keepdim=True)

    # Argument to the exp, in Q16.16, as an integer.
    tq = torch.round(m - logit)
    tq = torch.where(blocked, torch.full_like(tq, 1 << 30), tq).clamp(0, 1 << 30)

    p = exp2_neg_q16_16(tq, table)            # Q0.28 integers, int64
    p = torch.where(blocked, torch.zeros_like(p), p)

    l = p.sum(dim=-1, keepdim=True)           # exact: int64, the TRUE denominator
    # p <= 2^28 by construction (exp2_neg_q16_16 returns Q0.28), summed tk times.
    assert_exact_range(l, "softmax denominator", float(tk) * (1 << 28))
    # Fold the per-token V scale into the weight and requantise to one common
    # scale, so the accumulation below is still a sum of INTEGERS. Note the
    # denominator stays `l` (built from p): only the numerator carries sv.
    W, wscale = fold_scale_into_weights(p, sv, P_BITS)
    Wg = W.reshape(b, nkv, rep * q_len, tk)
    acc = exact_pv(Wg, vi).reshape(b, nh, q_len, d)
    assert_exact_range(acc, "W*V accumulator",
                       float(tk) * (1 << P_BITS) * v_lv)

    out = (acc * wscale) / l.to(torch.float64).clamp(min=1.0)
    out = out.to(query.dtype).transpose(1, 2).contiguous()
    return out, None


# A factual prompt with a one-word answer no broken arm reaches by accident.
# Deliberately NOT one of the measured prompts: the point is to ask whether the
# arm computes the model's function at all, which a prompt chosen for a narrow
# top-2 margin is the worst possible way to ask.
CANARY = "The capital of France is"
CANARY_MUST_CONTAIN = "Paris"
CANARY_TOKENS = 10
# 160 greedy tokens from this model run 60-110 distinct. A collapsed arm runs
# 1-4. The threshold is nowhere near either, because a false failure here stops
# a 20-minute run.
MIN_DISTINCT = 8


def sanity_verdict(canary, ref_tokens):
    """Is this arm computing the model's function? -> (ok, reason).

    Pure: no model, no GPU, no torch. `canary` is the decoded continuation of
    `CANARY`; `ref_tokens` is the arm's own reference generation, the one every
    later comparison is made against.

    Two checks, and they are independent on purpose:

      * **the canary** catches an arm that computes the WRONG FUNCTION - the
        layout bug in the docstring above answered this prompt in Arabic.
      * **degeneracy** catches an arm that computes the right function up to
        some length and then collapses. The canary is 10 tokens on a different
        prompt, so it cannot see that; this reads the 160-token generation the
        result is actually derived from.

    Either alone leaves a gap that `--check-gate` demonstrates rather than
    asserts: scenario 3 passes degeneracy and fails the canary, scenario 4 the
    reverse.
    """
    if CANARY_MUST_CONTAIN not in canary:
        return False, (f"canary {CANARY!r} -> {canary!r}, which does not "
                       f"contain {CANARY_MUST_CONTAIN!r}")
    n = len(set(ref_tokens))
    if len(ref_tokens) >= MIN_DISTINCT * 4 and n < MIN_DISTINCT:
        return False, (f"reference generation collapsed: {n} distinct tokens "
                       f"in {len(ref_tokens)}")
    return True, f"canary ok, {n} distinct tokens in reference"


def canary_text(model, tok, device):
    # No `padding=True`: one sequence needs none, and asking for it makes
    # this fail on a tokenizer with no `pad_token` set - which the
    # throughput tool's is, since it never batches ragged input.
    enc = tok(CANARY, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask,
            max_new_tokens=CANARY_TOKENS, do_sample=False, num_beams=1,
            use_cache=True, pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, enc.input_ids.shape[1]:])


def check_gate_logic():
    """Exercise `sanity_verdict` on synthetic outputs. No GPU, no model.

        python3 tools/exact_ragged_batch.py --check-gate
    """
    healthy = list(range(90)) * 2
    collapsed = [11, 11, 12, 11] * 40
    scenarios = [
        ("healthy arm", " Paris. It was founded in 789 AD by", healthy, True),
        ("wrong tensor layout (the real bug)",
         " \u0627\u0644\u0633\u0627\u062f \u0627\u0644\u0633\u0627\u062f \u0627\u0644\u0633\u0627\u062f", [772, 773] * 80, False),
        ("fluent, wrong function", " Berlin, a city in Germany", healthy, False),
        ("canary ok but generation collapsed", " Paris.", collapsed, False),
        ("short reference, not enough to judge", " Paris.", [5, 5], True),
    ]
    bad = 0
    for name, canary, toks, want in scenarios:
        ok, reason = sanity_verdict(canary, toks)
        bad += ok != want
        print(f"  {'ok  ' if ok == want else 'WRONG'}  {name:<38} "
              f"{'pass' if ok else 'FAIL'} (want {'pass' if want else 'FAIL'})"
              f"\n          {reason}")
    if bad:
        print(f"\n{bad} scenario(s) wrong: the gate's own logic is broken.")
        return 1
    print("\ngate logic behaves on all 5 scenarios; 3 and 4 show the two "
          "checks are\nindependent - neither alone catches both.")
    return 0


def sanity_reference(model, tok, args):
    """The arm's own batch-1 generation of PROMPTS[0], for the degeneracy half.

    Reuses the run's own token budget so the check reads the same length of
    output the verdict is drawn from, rather than a shorter proxy.
    """
    ids = tok(PROMPTS[0], return_tensors="pt").to(args.device)
    with torch.no_grad():
        g = model.generate(
            input_ids=ids.input_ids, attention_mask=ids.attention_mask,
            max_new_tokens=args.tokens, do_sample=False, num_beams=1,
            use_cache=True, pad_token_id=tok.eos_token_id,
        )
    return g[0, ids.input_ids.shape[1]:].tolist()


def decode(model, tok, prompt, batch, n_new, device, static_cache=False):
    ids = tok(prompt, return_tensors="pt").to(device)
    inp = ids.input_ids.repeat(batch, 1)
    att = ids.attention_mask.repeat(batch, 1)
    with torch.no_grad():
        out = model.generate(
            input_ids=inp, attention_mask=att, max_new_tokens=n_new,
            do_sample=False, num_beams=1, use_cache=True,
            pad_token_id=tok.eos_token_id,
            **({"cache_implementation": "static"} if static_cache else {}),
        )
    return out[0, inp.shape[1]:].tolist()


def build(device, exact):
    """Stock = bf16 + SDPA, i.e. what production actually serves.

    The earlier version of this demo ran the control in fp32, where the
    batch-dependent logit delta is ~4e-5 -- real, but too small to flip a
    greedy argmax inside a short generation, so the tokens matched and the
    demo looked like it proved nothing. In bf16 the delta is **0.34** against
    a typical top-2 margin of **0.25**: the noise exceeds the decision margin
    and the text visibly changes.
    """
    if exact:
        qwen2.eager_attention_forward = exact_attention
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        ).to(device).eval()
        nl, nn_ = exact_model.convert(m)
        print(f"    ({nl} linear layers + {nn_} norms converted to exact form)")
        return m
    qwen2.eager_attention_forward = _orig
    return AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()


PROMPTS = [
    "Write a short story about a lighthouse keeper who discovers something unusual.",
    "List five surprising facts about octopuses and explain each one.",
    "Describe, step by step, how a suspension bridge carries load.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--tokens", type=int, default=160)
    # A static cache is the prerequisite for CUDA graphs, which is the whole of
    # vLLM's advantage over this baseline (M5 step 12). Two facts, both measured:
    # it changes the output ONCE relative to a dynamic cache (~5.7e-6 per step,
    # from the float32 RoPE and residuals that this project does NOT make exact,
    # fused differently under the changed shapes) - a version change, acceptable
    # if stated - and **batch invariance still holds under it**, exact 0/3
    # against stock 3/3, which is the property being sold. So adopting a static
    # cache is compatible with the claim; cache-implementation invariance is a
    # different claim and was never made.
    ap.add_argument("--static-cache", action="store_true",
                    help="use a StaticCache (the CUDA-graph prerequisite)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)

    print(f"\n{'=' * 74}")
    print("  SAME PROMPT, SAME WEIGHTS, DIFFERENT BATCH SIZE")
    print(f"{'=' * 74}")
    print("  The prompt is replicated to fill each batch, so row 0 is the")
    print("  identical computation every time. Anything that changes is the")
    print("  kernel's reduction order following the batch size.\n")

    verdict = {}
    for name, is_exact in (("stock  (bf16 + SDPA, what production serves)", False),
                           ("exact  (int8, order-independent reductions)", True)):
        print(f"--- {name} ---")
        model = build(args.device, is_exact)
        # Before anything is counted. This loop reports `diverged`, and it
        # prints the generated text ONLY when a prompt diverges - so an arm
        # that is uniformly broken scores 0/8, prints no text at all, and is
        # declared the winner. See `sanity_verdict`.
        ok, reason = sanity_verdict(canary_text(model, tok, args.device),
                                    sanity_reference(model, tok, args))
        print(f"  sanity: {reason}")
        if not ok:
            print(f"\n  *** THIS ARM IS NOT COMPUTING THE MODEL'S FUNCTION. A "
                  f"uniformly broken arm\n      diverges on nothing and scores "
                  f"0/{len(PROMPTS)}, which is the passing number. "
                  f"Refusing to report it. ***")
            return 1
        diverged = 0
        for pi, prompt in enumerate(PROMPTS):
            ids = tok(prompt, return_tensors="pt").to(args.device)
            seqs = {}
            for b in args.batches:
                with torch.no_grad():
                    g = model.generate(
                        input_ids=ids.input_ids.repeat(b, 1),
                        attention_mask=ids.attention_mask.repeat(b, 1),
                        max_new_tokens=args.tokens, do_sample=False, num_beams=1,
                        use_cache=True, pad_token_id=tok.eos_token_id,
                        **({"cache_implementation": "static"}
                           if args.static_cache else {}),
                    )
                seqs[b] = g[0, ids.input_ids.shape[1]:].tolist()
            base = seqs[args.batches[0]]
            marks = []
            worst = None
            for b in args.batches[1:]:
                at = next((i for i, (x, y) in enumerate(zip(seqs[b], base)) if x != y), None)
                marks.append(f"b{b}: " + ("same" if at is None else f"DIVERGES@{at}"))
                if at is not None and (worst is None or at < worst[0]):
                    worst = (at, b)
            if worst:
                diverged += 1
            print(f"  prompt {pi}  {'  '.join(marks)}")
            if worst:
                at, b = worst
                a_txt = tok.decode(base[max(0, at - 6):at + 14])
                b_txt = tok.decode(seqs[b][max(0, at - 6):at + 14])
                print(f"      batch {args.batches[0]:>2}: ...{a_txt!r}")
                print(f"      batch {b:>2}: ...{b_txt!r}")
        verdict[name] = diverged
        print(f"  => {diverged}/{len(PROMPTS)} prompts produced different text\n")
        del model
        torch.cuda.empty_cache()
        qwen2.eager_attention_forward = _orig

    keys = list(verdict)
    print("=" * 74)
    print(f"  stock : {verdict[keys[0]]}/{len(PROMPTS)} prompts changed with batch size")
    print(f"  exact : {verdict[keys[1]]}/{len(PROMPTS)} prompts changed with batch size")
    print("=" * 74)
    if verdict[keys[0]] == 0:
        print("\n  CONTROL DID NOT MOVE -- this run proves nothing.")
        return 1
    return 0 if verdict[keys[1]] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
