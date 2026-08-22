"""What did the per-token KV scale COST? Measured on real activations.

    python3 tools/exact_accuracy.py

`tools/exact_kv.py` changed K and V from per-head / per-channel scales to
per-token ones, so that a decode step's quantisation work stops being O(T). That
is a change of *scheme*, not an optimisation, so bit-identity is the wrong
acceptance test - this is the right one.

Four attention implementations are compared against an **fp64 exact softmax
oracle over the original fp32 K/V**, on post-RoPE activations captured from
Qwen2.5-0.5B:

  * `old`   - per-head K scale, per-channel V scale (what shipped before)
  * `new`   - per-token K and V scales, `sv` folded into the weights
  * `flash` - f32 tiled online softmax over the SAME int8 operands, i.e. what a
              production flash kernel does. The bar the exact path has to clear.
  * `int8`  - the fp64 softmax over int8 K/V with no fixed-point step at all,
              which isolates how much of the error is *quantisation* rather
              than anything this project does.

The last one matters most and is the reason the comparison is worth running:
if `old` and `new` both sit far below `int8`, then the scheme change is not
what limits accuracy and the O(T) fix was free.
"""
import _nospace

_nospace.guard()

import math  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
import transformers.models.qwen2.modeling_qwen2 as qwen2  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import batch_invariance_demo as D  # noqa: E402
from exact_kv import (V_LEVELS, fold_scale_into_weights,  # noqa: E402
                      k_levels_for, quantize_rows)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def oracle(q, k, v, sc):
    """Exact softmax attention in fp64 over the ORIGINAL fp32 K/V."""
    logit = (q.double() @ k.double().transpose(-1, -2)) * sc
    w = torch.softmax(logit, dim=-1)
    return w @ v.double()


def q_rows(x, levels):
    """float32, matching what ships. `path_old` stays in float64 - that is what
    the old scheme actually did, so the comparison is scheme against scheme.

    `levels` is required, like `quantize_rows`'s. With a 127 default this file
    measured int8 K/V whatever the shipping widths were, so raising
    `Y_EXACT_K_LEVELS` would have left this tool quietly reporting the error of
    a scheme that no longer ships - a harness disagreeing with its subject in
    the one direction nobody checks.
    """
    return quantize_rows(x.float(), levels)


def q_head(x):
    s = x.double().abs().amax(dim=(-2, -1), keepdim=True) / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.round(x.double() / s).clamp(-127, 127), s


def q_chan(x):
    s = x.double().abs().amax(dim=-2, keepdim=True) / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.round(x.double() / s).clamp(-127, 127), s


def path_old(q, k, v, sc, table):
    qi, sq = q_rows(q, D.Q_LEVELS)
    ki, sk = q_head(k)
    vi, sv = q_chan(v)
    s_int = qi.double() @ ki.transpose(-1, -2)
    kfix = sq * sk * sc * math.log2(math.e) * 65536.0
    m = s_int.amax(dim=-1, keepdim=True)
    tq = torch.round((m - s_int) * kfix).clamp(0, 1 << 30)
    p = D.exp2_neg_q16_16(tq, table)
    l = p.sum(dim=-1, keepdim=True)
    acc = p.double() @ vi
    return (acc / l.double().clamp(min=1.0)) * sv


def path_new(q, k, v, sc, table):
    qi, sq = q_rows(q, D.Q_LEVELS)
    ki, sk = q_rows(k, k_levels_for(k.shape[-1], D.Q_LEVELS))
    vi, sv = q_rows(v, V_LEVELS)
    s_int = qi.double() @ ki.double().transpose(-1, -2)
    logit = s_int * (sq * sc * math.log2(math.e) * 65536.0) * sk.transpose(-1, -2)
    m = logit.amax(dim=-1, keepdim=True)
    tq = torch.round(m - logit).clamp(0, 1 << 30)
    p = D.exp2_neg_q16_16(tq, table)
    l = p.sum(dim=-1, keepdim=True)
    W, wscale = fold_scale_into_weights(p, sv.transpose(-1, -2), D.P_BITS)
    acc = W.double() @ vi.double()
    return (acc * wscale) / l.double().clamp(min=1.0)


def path_int8_only(q, k, v, sc):
    """fp64 softmax over per-token int8 K/V: the quantisation cost alone."""
    qi, sq = q_rows(q, D.Q_LEVELS)
    ki, sk = q_rows(k, k_levels_for(k.shape[-1], D.Q_LEVELS))
    vi, sv = q_rows(v, V_LEVELS)
    logit = (qi.double() @ ki.double().transpose(-1, -2)) * sq * sk.transpose(-1, -2) * sc
    w = torch.softmax(logit, dim=-1)
    return w @ (vi.double() * sv)


def path_flash(q, k, v, sc, tile=64):
    """f32 tiled online softmax over the same int8 operands."""
    qi, sq = q_rows(q, D.Q_LEVELS)
    ki, sk = q_rows(k, k_levels_for(k.shape[-1], D.Q_LEVELS))
    vi, sv = q_rows(v, V_LEVELS)
    logit = ((qi.float() @ ki.float().transpose(-1, -2))
             * (sq * sk.transpose(-1, -2) * sc).float())
    vq = (vi * sv).float()
    T = logit.shape[-1]
    m = torch.full(logit.shape[:-1] + (1,), float("-inf"), device=q.device)
    l = torch.zeros_like(m)
    acc = torch.zeros(vq.shape[:-2] + (logit.shape[-2], vq.shape[-1]), device=q.device)
    for i in range(0, T, tile):
        x = logit[..., i:i + tile]
        mn = torch.maximum(m, x.amax(dim=-1, keepdim=True))
        corr = torch.where(torch.isinf(m), torch.zeros_like(m), torch.exp(m - mn))
        p = torch.exp(x - mn)
        l = l * corr + p.sum(dim=-1, keepdim=True)
        acc = acc * corr + p @ vq[..., i:i + tile, :]
        m = mn
    return (acc / l).double()


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL)
    cap = []
    orig = qwen2.eager_attention_forward

    def spy(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        cap.append((query.detach(), key.detach(), value.detach(), scaling))
        return orig(module, query, key, value, attention_mask, scaling=scaling,
                    dropout=dropout, **kw)

    qwen2.eager_attention_forward = spy
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    ).cuda().eval()
    text = ("The history of computing begins with mechanical calculation. "
            "Charles Babbage designed the Analytical Engine and Ada Lovelace "
            "wrote the first algorithm intended for a machine. ") * 10
    with torch.no_grad():
        model(**tok(text, return_tensors="pt").to("cuda"))
    qwen2.eager_attention_forward = orig
    table = D._table("cuda")

    rows = []
    for li in (0, 4, 8, 12, 16, 20, 23):
        qs, ks, vs, scaling = cap[li]
        _, nh, T, d = qs.shape
        nkv = ks.shape[1]
        sc = scaling if scaling is not None else 1.0 / math.sqrt(d)
        for h in (0, nh // 2, nh - 1):
            kv = h // (nh // nkv)
            q = qs[0, h, T - 1:T]          # decode-shaped: last query, full cache
            k, v = ks[0, kv], vs[0, kv]
            ref = oracle(q, k, v, sc)
            scale = float(ref.abs().max()) + 1e-30
            rows.append((
                float((path_old(q, k, v, sc, table) - ref).abs().max()) / scale,
                float((path_new(q, k, v, sc, table) - ref).abs().max()) / scale,
                float((path_flash(q, k, v, sc) - ref).abs().max()) / scale,
                float((path_int8_only(q, k, v, sc) - ref).abs().max()) / scale,
            ))

    # Control: the four paths must actually be four paths. They agree closely by
    # design - the error is dominated by int8 quantisation, which three of them
    # share - so "they all scored the same" has to be distinguished from "they
    # are accidentally the same code".
    qq = torch.randn(1, 64, device="cuda")
    kk = torch.randn(200, 64, device="cuda")
    vv = torch.randn(200, 64, device="cuda")
    outs = [path_old(qq, kk, vv, 0.125, table), path_new(qq, kk, vv, 0.125, table),
            path_flash(qq, kk, vv, 0.125), path_int8_only(qq, kk, vv, 0.125)]
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            if torch.equal(outs[i], outs[j]):
                print(f"  CONTROL FAILED: paths {i} and {j} are bit-identical; "
                      f"the comparison below is vacuous")
                return 1

    def med(i):
        return sorted(r[i] for r in rows)[len(rows) // 2]

    print(f"\n{len(rows)} layer/head pairs, relative max error vs an fp64 oracle "
          f"over the ORIGINAL fp32 K/V\n")
    names = ("old  (per-head K, per-channel V)", "new  (per-token K and V)",
             "flash (f32 online softmax, same int8)", "int8 only (fp64 softmax)")
    for i, n in enumerate(names):
        print(f"  {n:<40} median {med(i):.3e}   worst {max(r[i] for r in rows):.3e}")
    print()
    print(f"  new / old   = {med(1) / med(0):.2f}x")
    print(f"  new / flash = {med(1) / med(2):.2f}x")
    print(f"  new / int8-only = {med(1) / med(3):.2f}x   "
          f"<- how much the FIXED-POINT step adds on top of quantisation")
    if med(1) > 2.0 * med(3):
        print("\n  WARNING: the fixed-point path is adding materially more error "
              "than int8 quantisation itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
