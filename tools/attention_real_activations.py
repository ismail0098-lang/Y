"""M5 step 1: does the exact attention path survive REAL activations?

`tests/attention_quantization_error.rs` bounded the numerics on synthetic score
distributions and said so explicitly: "no model, no calibration, no activation
outliers ... this bounds the numerics; it does not answer §6 item 1."

This captures post-RoPE Q/K/V from a real model on real text and reruns the
same comparison on it. Two references, deliberately:

  * vs f64 attention over the SAME int8 V  -> isolates the softmax
    representation, which is this project's design choice.
  * vs f64 attention over the ORIGINAL fp32 V -> the total cost, which is
    mostly int8 itself, i.e. the customer's existing decision.

The baseline in both cases is an f32 tiled online softmax, which is what
production flash attention does.
"""
import json
import math
import sys

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------------------------------------------------------------- fixed exp
# The same recipe as src/fixed_exp.rs. Reimplemented here rather than called
# through FFI, and `check_fixed_exp` below compares it against the Rust one
# BIT FOR BIT over the whole domain.
#
# It used to check only that the result is within 1 ulp of `2^28 * 2^-t` and
# monotonic, and its comment claimed that made "a divergence visible rather
# than assumed away". It did not: two implementations can both be within 1 ulp
# of the truth and differ from each other by 1, and 1 ulp is a BIT. The ulp
# sweep also covered `t < 2^16` only -- the first of thirty-one unit intervals
# -- at stride 7.
LN2_Q30 = 744261118
RECIP6_Q32 = 715827883
# FNV-1a over `0 .. 31 << 16`; `fixed_exp::EXP2_DOMAIN_DIGEST` in Rust.
EXP2_DOMAIN_DIGEST = 0x7170CD39B442D506
TABLE = [round((1 << 30) * 2 ** (-i / 64)) for i in range(64)]


def exp2_neg_q16_16(t: int) -> int:
    n = t >> 16
    if n >= 30:
        return 0
    f = t & 0xFFFF
    base = TABLE[f >> 10]
    d = f & 0x3FF
    y = (d * LN2_Q30) >> 16
    y2 = (y * y) >> 30
    y3 = (y2 * y) >> 30
    corr = (1 << 30) - y + (y2 >> 1) - ((y3 * RECIP6_Q32) >> 32)
    g = (base * corr) >> 30
    sh = 2 + n
    return (g + (1 << (sh - 1))) >> sh


def check_fixed_exp():
    assert exp2_neg_q16_16(0) == 1 << 28, "exp2(0) must be exactly 2^28"
    worst = 0.0
    prev = 1 << 30
    h, mask = 0xCBF29CE484222325, (1 << 64) - 1
    # One pass, over EVERY argument the exp accepts rather than the first
    # thirty-first of them. ~2 s in pure Python.
    for t in range(31 << 16):
        v = exp2_neg_q16_16(t)
        h = ((h ^ v) * 0x00000100000001B3) & mask
        want = (1 << 28) * 2.0 ** (-t / 65536.0)
        worst = max(worst, abs(v - want))
        assert v <= prev, f"not monotonic at t={t}"
        prev = v
    # The ulp bound and monotonicity are PROPERTIES; this is IDENTITY, and it
    # is the only one of the three that can see a one-bit divergence from the
    # compiler's own exp.
    assert h == EXP2_DOMAIN_DIGEST, (
        f"this transcription no longer matches src/fixed_exp.rs: digest "
        f"{h:#018x} against {EXP2_DOMAIN_DIGEST:#018x}"
    )
    assert worst < 1.0, f"fixed exp is {worst:.3f} ulp off"
    return worst


# ---------------------------------------------------------------- capture
CAPTURED = []
_orig = qwen2.eager_attention_forward


def spy(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    CAPTURED.append(
        (
            getattr(module, "layer_idx", -1),
            query.detach().float().cpu(),
            key.detach().float().cpu(),
            value.detach().float().cpu(),
            scaling,
        )
    )
    return _orig(module, query, key, value, attention_mask, scaling=scaling,
                 dropout=dropout, **kw)


# ---------------------------------------------------------------- the paths
def quantize_per_channel(x):
    """Symmetric per-CHANNEL int8: one scale per feature dimension.

    This is what production int8 pipelines do; per-tensor scaling is destroyed
    by a single outlier channel, which is exactly what SmoothQuant/AWQ exist to
    address. Reported beside the per-tensor number so the error is attributed
    to the SCHEME rather than to int8.
    """
    s = x.abs().amax(dim=0) / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.round(x / s).clamp(-127, 127).to(torch.int64), s


def quantize(x):
    """Symmetric per-tensor int8. `x` is a torch tensor; returns (int8, scale)."""
    s = x.abs().max().item() / 127.0
    if s == 0:
        s = 1.0
    return torch.round(x / s).clamp(-127, 127).to(torch.int64), s


def exact_path(q8, k8, v8, sq, sk, sv, d, sc):
    """Global max, Q0.28 weights via the integer exp, integer accumulation."""
    s = (q8.unsqueeze(0) * k8).sum(dim=1)          # int32-exact scores, [T]
    s = s.tolist()
    m = max(s)
    # logit(t) = (s[t]-m) * sq*sk/sqrt(d); argument to the exp is
    # (m - s[t]) * sq*sk/sqrt(d) * log2(e), taken to Q16.16 in integer.
    kfix = sq * sk * sc * math.log2(math.e) * 65536.0
    tq = [min(int(round((m - x) * kfix)), (1 << 31) - 1) for x in s]
    p = [exp2_neg_q16_16(t) for t in tq]
    l = sum(p)
    if l == 0:
        return None
    pt = torch.tensor(p, dtype=torch.int64)
    o = (pt.unsqueeze(1) * v8).sum(dim=0)          # exact int64
    return (o.double() / float(l)) * sv


def flash_f32(logits, v, tile=64):
    """The production baseline: tiled online softmax, f32 throughout."""
    m, l = float("-inf"), 0.0
    acc = torch.zeros(v.shape[1], dtype=torch.float32)
    for i in range(0, logits.shape[0], tile):
        x = logits[i:i + tile].to(torch.float32)
        tm = float(x.max())
        mn = max(m, tm)
        corr = float(2.0 ** ((m - mn) * math.log2(math.e))) if m != float("-inf") else 0.0
        l *= corr
        acc *= corr
        p = torch.exp(x - mn)
        l += float(p.sum())
        acc += (p.unsqueeze(1) * v[i:i + tile].to(torch.float32)).sum(dim=0)
        m = mn
    return (acc.double() / l)


def reference(logits, v):
    w = torch.softmax(logits.double(), dim=0)
    return (w.unsqueeze(1) * v.double()).sum(dim=0)


def main():
    ulp = check_fixed_exp()
    print(f"fixed exp agrees with the Rust recipe to {ulp:.3f} ulp\n")

    qwen2.eager_attention_forward = spy
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    # Long enough to be DECODE-shaped. The first run used 13-21 token prompts,
    # where a flash kernel barely reassociates anything and its float error is
    # therefore near zero -- the regime least favourable to exact accumulation
    # and least like the one being sold.
    body = (
        "The history of computing begins with mechanical calculation. "
        "Charles Babbage designed the Analytical Engine, and Ada Lovelace wrote "
        "the first algorithm intended for a machine. A century later, Alan Turing "
        "formalised computability, and von Neumann described the stored-program "
        "architecture that nearly every processor still follows. "
    )
    prompts = [body * 12, (body + "Transistors replaced vacuum tubes. ") * 10]
    for text in prompts:
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            model(**ids)

    print(f"captured {len(CAPTURED)} attention calls\n")

    rows = []
    # Sample across depth: early / middle / late layers behave differently.
    for layer, qs, ks, vs, scaling in CAPTURED:
        if layer % 8 != 0:
            continue
        b, nh, T, d = qs.shape
        nkv = ks.shape[1]
        rep = nh // nkv
        for h in range(0, nh, 5):
            kv = h // rep
            k = ks[0, kv]          # [T, d]
            v = vs[0, kv]
            k8, sk = quantize(k)
            v8, sv = quantize(v)
            # Decode-shaped: attend from the last position over the whole cache.
            q = qs[0, h, T - 1]
            q8, sq = quantize(q)

            sc = scaling if scaling is not None else 1.0 / math.sqrt(d)
            v8f = v8.to(torch.float32) * sv

            # The TRUE attention, for the total-cost column.
            logits_fp = (q.double() @ k.double().T) * sc
            ref_fp = reference(logits_fp, v)

            # Both candidates must see the SAME logits, or this measures Q/K
            # quantization rather than the softmax representation -- the exact
            # path derives its scores from int8 q,k, so the baseline must too.
            # Getting this wrong once already produced a 10^6x "finding".
            s_int = (q8.unsqueeze(0) * k8).sum(dim=1)
            logits_q = s_int.double() * (sq * sk * sc)
            ref_q = reference(logits_q, v8f)

            ex = exact_path(q8, k8, v8, sq, sk, sv, d, sc)
            if ex is None:
                continue
            fl = flash_f32(logits_q, v8f)

            v8pc, svpc = quantize_per_channel(v)
            ref_pc = reference(logits_fp, v8pc.to(torch.float32) * svpc)

            scale = float(ref_fp.abs().max()) + 1e-12
            rows.append(
                dict(
                    layer=layer, head=h, T=T,
                    exact_vs_q=float((ex - ref_q).abs().max()) / scale,
                    flash_vs_q=float((fl - ref_q).abs().max()) / scale,
                    exact_vs_fp=float((ex - ref_fp).abs().max()) / scale,
                    int8_cost=float((ref_q - ref_fp).abs().max()) / scale,
                    int8_pc=float((ref_pc - ref_fp).abs().max()) / scale,
                )
            )

    if not rows:
        print("no rows captured", file=sys.stderr)
        return 1

    print(f"{'layer':>5} {'head':>4} {'T':>4} {'exact/ref_q':>12} {'flash/ref_q':>12} "
          f"{'ratio':>7} {'int8 cost':>10}")
    for r in rows:
        ratio = r["exact_vs_q"] / r["flash_vs_q"] if r["flash_vs_q"] > 0 else float("inf")
        print(f"{r['layer']:>5} {r['head']:>4} {r['T']:>4} {r['exact_vs_q']:>12.3e} "
              f"{r['flash_vs_q']:>12.3e} {ratio:>6.2f}x {r['int8_cost']:>10.3e}")

    worst_ratio = max(r["exact_vs_q"] / r["flash_vs_q"] for r in rows if r["flash_vs_q"] > 0)
    med_int8 = sorted(r["int8_cost"] for r in rows)[len(rows) // 2]
    med_pc = sorted(r["int8_pc"] for r in rows)[len(rows) // 2]
    med_exact = sorted(r["exact_vs_q"] for r in rows)[len(rows) // 2]
    print()
    print(f"worst exact/flash error ratio (same int8 V): {worst_ratio:.2f}x")
    print(f"median softmax-representation error        : {med_exact:.3e}  (relative)")
    print(f"median int8-V error, per-TENSOR scale      : {med_int8:.3e}  (relative)")
    print(f"median int8-V error, per-CHANNEL scale     : {med_pc:.3e}  (relative)")
    print(f"  -> int8 V costs {med_int8 / max(med_exact, 1e-30):.1f}x what the exact "
          f"softmax path does")
    json.dump(rows, open("/tmp/attn_real_rows.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
