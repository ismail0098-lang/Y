"""An append-only quantised KV cache. O(1) per decode step instead of O(T).

## The problem NCU found

Attention's quantisation was ~29% of decode time, in kernels running at 8.9% of
DRAM peak - latency-stalled on almost no work. The cause was not tuning. K and V
were re-quantised **in full on every decode step** although the cache grows by
one row, so the work was O(T) per step and O(T^2) over a generation.

## Why the old scheme could not be cached, and what changed

The scales depended on the whole cache: K used one scale per head over all
`(T, d)`, V one scale per feature channel over all `T`. Add a token and every
scale can move, so every previously quantised row has to be recomputed.

The running maximum is monotone, so one might hope to rebuild only when a scale
actually grows. That works for K (a per-head max over `T*d` values settles
quickly) and **fails for V**: with `b*nkv*d` = 4096 independent per-channel
maxima, the expected number that move at step `t` is `4096/t`, so *some*
channel moves nearly every step and a full pass is needed anyway.

So the scales are **per token** now. Row `t`'s quantisation depends on row `t`
alone, which is exactly the property that makes the cache append-only.

## Why this is still exact

A per-token V scale does not factor out of `sum_t p[t] * sv[t] * v8[t,d]`, which
is what makes the naive version of this idea break exactness. It is recovered by
folding `sv` into the weight and requantising the product to one common scale:

    w[t] = p[t] * sv[t]                        (float, elementwise)
    W[t] = round(w[t] * 2^28 / max_t w[t])     (integer, <= 2^28)
    numerator[d] = (max_t w / 2^28) * sum_t W[t] * v8[t,d]      <- exact
    denominator  = sum_t p[t]                                   <- exact

Both sums are over integers, so both are associative and commutative and the
result does not depend on tile order, split, or batch size - the same argument
as before, one level of algebra further in. `max_t w` is a maximum, which is
exact in floating point and order-independent too. `W` carries 28 bits, the
same as `p` did, so the fold costs nothing measurable in precision.

This IS a change of quantisation scheme (per-channel-over-time -> per-token) and
therefore needs an accuracy argument rather than a bit-identity one; see
`tools/exact_accuracy.py`, which measures both against an fp64 oracle.

## Cache validity

Row-local quantisation is valid only if a cached row's *input* never changes.
That holds for an append-only KV cache under greedy or sampled decode. It does
NOT hold under beam search or any cache reordering, so the state is keyed to the
module, invalidated whenever the batch/head/depth shape changes or `T` fails to
grow, and `module=None` bypasses the cache entirely - which is what the
self-test uses to check the cached and uncached paths agree.
"""
import os

import torch

# DEFAULT OFF, and the measurement below is the whole reason. The cache does
# asymptotically LESS work - O(1) per decode step instead of O(T) - and under
# `torch.compile` every way of wiring it in loses badly, because it is stateful
# and the compiler is not:
#
#   stateless, in-graph                    5.24 ms/step   <- shipped
#   cache, visible to dynamo            2882    ms/step   (recompiles per token:
#                                                          `self.t` is a Python
#                                                          int and dynamo guards
#                                                          on it, so the whole
#                                                          model recompiles)
#   cache, @torch._dynamo.disable         27.8  ms/step   (one graph break per
#                                                          attention call; 3
#                                                          frames -> 51)
#   stateless, @torch._dynamo.disable     27.3  ms/step   (the break alone)
#
# So the break costs ~5x more than the O(T) quantisation it removes, and the
# recompiles cost 550x. The code stays because it is the right structure for a
# serving path that is NOT a traced graph - the PTX kernels - where there is no
# dynamo to lose to. `Y_EXACT_KVCACHE=1` turns it on.
USE_CACHE = os.environ.get("Y_EXACT_KVCACHE", "0") != "0"


# K and V widths, as the largest representable magnitude (127 = int8).
# Separate because they cost differently: widening K is free (one matmul either
# way, bound `d*127*K_LEVELS < 2^24` allows up to 2055 at head_dim 64), while
# widening V forces `exact_pv` into narrower digits and more matmuls. Both are
# env-overridable so the accuracy/throughput trade can be swept without edits.
# `K_LEVELS` is a REQUESTED MAXIMUM, not the width used: the real width comes
# from `k_levels_for(head_dim)` because the exactness budget shrinks as head_dim
# grows. `V_LEVELS` is used as given - V is bounded by the digit split, not by
# the score sum, and the sweep says widening it buys nothing anyway.
def _levels_from_env(name, default, floor):
    """Read a width from the environment, or refuse.

    `quantize_rows` already applies CLAUDE.md's design rule to whether `levels`
    was *supplied* -- its docstring says so, and cites the `_ =>` table. It was
    not applied to the VALUE, and these are advertised as sweepable knobs.
    Measured before this check existed, on `quantize_rows([1, -2, 3, 4], lv)`:

        levels=0   ->  q = [0, 0, 0, 0],      scale = inf
        levels=-5  ->  q = [-5, -5, -5, -5],  scale = -0.8

    So `Y_EXACT_V_LEVELS=0` silently zeroed the whole V cache and a negative
    width pinned every element to the clamp. No error, no NaN, just a model
    quietly computing something else.

    The floor for K is 127 rather than 1 because `k_levels_for` returns 127 as
    its FAIL-CLOSED sentinel -- the value `exact_attention`'s score assert is
    guaranteed to refuse when even int8 will not fit. A request below 127 was
    therefore silently widened to 127 (`Y_EXACT_K_LEVELS=63` at head_dim 64 gave
    127), which would have put a flat left tail on any accuracy sweep across
    that axis and read as "narrowing K costs nothing". Refused rather than
    re-purposed: the sentinel and the request cannot share a value.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None
    if v < floor:
        raise ValueError(
            f"{name}={v} is below {floor}. A width is the largest representable "
            f"magnitude, so it must be positive"
            + ("; 127 is int8 and also `k_levels_for`'s fail-closed sentinel, so "
               "a narrower K cannot be requested through this knob"
               if floor == 127 else "")
        )
    return v


K_LEVELS = _levels_from_env("Y_EXACT_K_LEVELS", 1023, 127)
V_LEVELS = _levels_from_env("Y_EXACT_V_LEVELS", 127, 1)

# Store V as real int8 rather than float32 carrying integers.
#
# `quantize_rows`' docstring gives the reason it was float32: "every consumer
# wants floats and a per-step cast of the whole cache would put back the O(T)
# pass this file exists to remove." The first half is what changed -- V's
# consumer is `exact_pv`, and the int64 kernel in tests/exact_pv.ysu wants int8.
# The second half never applied to STORAGE: the cache appends only the new rows,
# so quantising at append is O(1) per decode step. The O(T) cast appears only if
# a float32 consumer reads an int8 cache, which is why this flag and the
# `exact_pv` dispatch are one decision and not two.
#
# Worth 4x the read traffic on V and, measured through tools/exact_pv_bridge.py,
# takes exact_pv from 2.75x to 5.04x at T=4096 by deleting the cast.
#
# **V only.** V_LEVELS is 127 and fits int8 exactly; K's width is
# `k_levels_for(head_dim, q_levels)`, which is 1023 at head_dim 64 and needs 16
# bits. Asserted below rather than left as a comment.
V_INT8 = os.environ.get("Y_EXACT_V_INT8", "0") != "0"
if V_INT8 and V_LEVELS > 127:
    raise ValueError(
        f"Y_EXACT_V_INT8 needs V_LEVELS <= 127 to be exact; it is {V_LEVELS}. "
        "Widening V past int8 is a real choice with a real cost -- it is not "
        "something to absorb with a silent clamp."
    )


def v_store_dtype():
    """The dtype the V side of the cache is held in."""
    import torch
    return torch.int8 if V_INT8 else torch.float32
SCORE_BUDGET = 1 << 24          # fp32 holds every integer below this exactly


def k_levels_for(d, q_levels):
    """Widest exact K width at this head_dim, capped at `K_LEVELS`.

    **A fixed constant is wrong here.** Exactness needs
    `d * q_levels * k_levels < 2^24`, so the ceiling falls as head_dim rises:
    2064 at d=64, 1032 at d=128, 516 at d=256. Hardcoding 1023 - which is what
    the sweep that found this measured at d=64 - fails closed at head_dim 256,
    i.e. refuses to run some real models. Deriving it gives 1023 where it fits
    and 511 where it does not.

    **Using 99% of the budget is exactly as safe as using 6%.** `|score| <=
    d * q * k` is a proof, and it bounds every partial sum as well as the total,
    so there is no margin to leave. The cap at `K_LEVELS` is about diminishing
    returns (127 -> 511 is worth 0.486 ppl, 511 -> 1023 another 0.045) and about
    K/V cache bytes, not about safety.

    Rounded down to a `2^n - 1` so the width is a whole number of bits. If the
    head_dim is so large that even int8 will not fit, this returns 127 and
    `exact_attention`'s assert refuses - fail-closed rather than silently
    rounding.
    """
    cap = min(SCORE_BUDGET // max(d * q_levels, 1), K_LEVELS)
    n = max(1, cap.bit_length())
    if (1 << n) - 1 > cap:
        n -= 1
    return max(127, (1 << n) - 1)


def quantize_rows(x, levels):
    """Symmetric per-token quantisation: one scale per row, from that row alone.

    `levels` is the largest representable magnitude (127 = int8), and it is
    **required on purpose**: it first defaulted to `K_LEVELS`, which silently
    gave V the K width at any caller that forgot to say which side it was
    quantising - the same shape as every `_ =>` arm in CLAUDE.md's design-rule
    table. A caller must state which side it is. It is a parameter because **the K/V cache costs 3.9% of perplexity where the weight
    quantisation costs 2.2%** (M5 step 10 attribution), and because the two
    sides of the cache can afford different widths:

      * **K is free to widen.** The score bound is `d * 127 * levels < 2^24`,
        which at head_dim 64 permits `levels` up to 2055, and Q*K^T is one
        matmul whatever the width. Nothing in the mainloop changes.
      * **V is not.** `exact_pv` splits the softmax weight into base-2^dbits
        digits bounded by `T * (2^dbits - 1) * levels < 2^24`, so widening V
        forces narrower digits and MORE matmuls - 8 instead of 5 at 10 bits.

    **Per-ROW is what exactness requires; per-CHANNEL would be unsound here.**
    A row scale factors straight out of the dot product (`sum_c q_c k_c * s`),
    leaving an integer accumulation. A per-channel scale sits *inside* the sum
    (`sum_c q_c k_c s_c`) and turns it into a float-weighted reduction, which is
    order-dependent again. The literature's better K scheme is per-channel; it
    is not available to us, and that is a real constraint of the approach rather
    than an oversight.

    `x` is [..., T, d] in **float32**. Returns (float32 integer-valued, float32
    scale of shape [..., T, 1]). The values come back as float32 rather than
    int8 because every consumer wants floats and a per-step cast of the whole
    cache would put back the O(T) pass this file exists to remove.

    **This used to run in float64 and that was the single largest avoidable
    cost in decode.** Widening `x` to double and dividing there is 1/64 rate on
    a GeForce part, and NCU showed the kernel at SM 61.2% with DRAM at 8.9% -
    saturating the fp64 divide pipe, not memory. Measured on the real shape
    (b=32, nkv=2, T=66, d=64; 270,336 elements): **68.71 us in float64 against
    12.27 us in float32, 5.60x**, for 2132 us of a 12.10 ms decode.

    The float64 was there to keep bit-identity with the ORIGINAL reference,
    which no longer exists - the per-token scheme replaced it. What it costs is
    measured rather than argued: the two disagree on **one int8 value in
    270,336** (0.0004%), from the same divide-boundary effect as the Triton
    `div_rn` bug, and `tools/exact_accuracy.py` re-measures the whole scheme
    against an fp64 oracle. Determinism is untouched either way: an elementwise
    divide in any single precision is reproducible; it is *reassociation* and
    *unspecified* operations that break batch invariance, not narrowness.
    """
    if not isinstance(levels, int) or levels < 1:
        raise ValueError(
            f"levels={levels!r} is not a positive integer. It is the largest "
            f"representable magnitude: 0 sends every value to zero with an "
            f"infinite scale, and a negative one pins every value to the clamp "
            f"-- both silently"
        )
    lv = float(levels)
    s = x.abs().amax(dim=-1, keepdim=True) / lv
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.round(x / s).clamp(-lv, lv)
    return q, s


class QuantKVCache:
    """Holds quantised K/V in geometrically grown buffers, appended in place."""

    __slots__ = ("shape", "t", "ki", "sk", "vi", "sv")

    def __init__(self):
        self.shape = None
        self.t = 0
        self.ki = self.sk = self.vi = self.sv = None

    def _grow(self, b, nkv, need, d, device):
        cap = self.ki.shape[2] if self.ki is not None else 0
        if cap >= need:
            return
        cap = max(need, max(64, cap * 2))
        ki = torch.empty((b, nkv, cap, d), dtype=torch.float32, device=device)
        sk = torch.empty((b, nkv, cap, 1), dtype=torch.float32, device=device)
        vi = torch.empty((b, nkv, cap, d), dtype=v_store_dtype(), device=device)
        sv = torch.empty((b, nkv, cap, 1), dtype=torch.float32, device=device)
        if self.t:
            ki[:, :, : self.t] = self.ki[:, :, : self.t]
            sk[:, :, : self.t] = self.sk[:, :, : self.t]
            vi[:, :, : self.t] = self.vi[:, :, : self.t]
            sv[:, :, : self.t] = self.sv[:, :, : self.t]
        self.ki, self.sk, self.vi, self.sv = ki, sk, vi, sv

    def get(self, k, v, k_levels):
        """k, v: [b, nkv, T, d] float32. Returns (ki, sk, vi, sv) sliced to T."""
        b, nkv, t_new, d = k.shape
        shape = (b, nkv, d)
        # Invalidate on a shape change or on T going backwards (a new sequence).
        # Both leave a cache whose rows no longer correspond to the input, and
        # attention over stale rows still produces plausible-looking output, so
        # this is checked rather than assumed - see
        # `check_kv_cache_matches_stateless`.
        if self.shape != shape or t_new < self.t:
            self.shape = shape
            self.t = 0
            self.ki = torch.empty((b, nkv, 0, d), dtype=torch.float32, device=k.device)
            self.sk = torch.empty((b, nkv, 0, 1), dtype=torch.float32, device=k.device)
            self.vi = torch.empty((b, nkv, 0, d), dtype=v_store_dtype(), device=k.device)
            self.sv = torch.empty((b, nkv, 0, 1), dtype=torch.float32, device=k.device)
        if t_new > self.t:
            self._grow(b, nkv, t_new, d, k.device)
            new = slice(self.t, t_new)
            self.ki[:, :, new], self.sk[:, :, new] = quantize_rows(
                k[:, :, new], k_levels)
            self.vi[:, :, new], self.sv[:, :, new] = quantize_rows(
                v[:, :, new], V_LEVELS)
            self.t = t_new
        return (self.ki[:, :, :t_new], self.sk[:, :, :t_new],
                self.vi[:, :, :t_new], self.sv[:, :, :t_new])


def quantize_kv(module, k, v, q_levels=127):
    """Per-token K/V, incrementally if given a module.

    **Returns the widths it used** (`..., k_lv, v_lv`) rather than leaving the
    caller to recompute them. Six separate bugs in this file's history came from
    a width being written in one place and assumed in another; handing the widths
    back with the data makes a bound that disagrees with its own operands
    impossible to write.
    """
    k_lv = k_levels_for(k.shape[-1], q_levels)
    if module is None or not USE_CACHE:
        ki, sk = quantize_rows(k, k_lv)
        vi, sv = quantize_rows(v, V_LEVELS)
        if V_INT8:
            vi = vi.to(torch.int8)
        return ki, sk, vi, sv, k_lv, V_LEVELS
    cache = getattr(module, "_y_qkv", None)
    if cache is None:
        cache = QuantKVCache()
        module._y_qkv = cache
    ki, sk, vi, sv = cache.get(k, v, k_lv)
    return ki, sk, vi, sv, k_lv, V_LEVELS


def fold_scale_into_weights(p, sv_t, p_bits=28):
    """`W = round(p * sv * 2^p_bits / max(p*sv))`, plus the common scale.

    `p` is int64 Q0.`p_bits`; `sv_t` is the per-token V scale broadcast to
    p's shape ([..., 1, T]). Returns (W int64, scale float64 [..., q, 1]).
    """
    w = p.to(torch.float64) * sv_t
    wmax = w.amax(dim=-1, keepdim=True)
    safe = torch.where(wmax > 0, wmax, torch.ones_like(wmax))
    W = torch.round(w * (float(1 << p_bits) / safe))
    W = W.clamp(0, 1 << p_bits).to(torch.int64)
    return W, safe / float(1 << p_bits)
