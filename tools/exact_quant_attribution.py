"""What is the remaining +2.2% perplexity made of, and what would fix it?

    python3 tools/exact_quant_attribution.py

M5 step 10 split the original +6.1% into weights (+2.2%) and KV cache (+3.9%),
and closed the KV half for free by widening K alone. The weight half is still
open, and the standard lever for it is **group-wise weight scales**: instead of
one scale per output channel, one per group of `g` inputs along the reduction
axis, so a single wide input channel stops setting the scale for the whole row.

**This measures the ceiling before anything is built**, because group-wise scales
are NOT free here the way widening K was:

    y_n = sum_k x_k w_nk s_x s_w[n]              <- one scale, factors out
    y_n = sum_g s_w[n,g] * (sum_{k in g} x_k w_nk)   <- G integer partials,
                                                        combined in FLOAT

The inner sums are still exact integers, but the outer combination is a
float-weighted sum over `G` terms and is therefore **order-dependent again** -
precisely the property the whole project exists to preserve. It can be recovered
by requantising the `G` scales onto a common integer grid and folding them into
the partials, which is the same device `fold_scale_into_weights` already uses for
the per-token V scale. That is real work in the Triton kernel, so it needs a
number in front of it rather than behind it.

Everything here is **fake quantisation in fp32** - weights and activations are
rounded to the target grid and immediately scaled back, with no int8 kernel
involved. That measures accuracy and nothing else, which is the only question at
this stage; it deliberately says nothing about speed or determinism.

Arms, chosen so the answer is an attribution rather than a ranking:

  fp32          no quantisation at all - the floor
  w8            per-output-channel weights, fp32 activations
  a8            per-token activations, fp32 weights
  w8a8          both: the SHIPPED scheme, must reproduce the known +2.2%
  w8g<g>        weights grouped by `g` along K, fp32 activations
  w8g<g>a8      grouped weights AND per-token activations - the realistic target

`w8` and `a8` matter as much as the grouped arms: **if the cost is mostly in the
activations, group-wise weight scales cannot fix it**, and the honest answer is
to stop rather than to build the kernel and discover that afterwards.
"""
import _nospace

_nospace.guard()

import argparse  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from exact_task_accuracy import MODEL, perplexity  # noqa: E402


def fq_weight(w, group=None, levels=127):
    """Fake-quantise `w` [out, in] symmetrically, per row or per row-group.

    `group=None` is the shipped scheme: one scale per output channel. Otherwise
    the reduction axis is cut into blocks of `group` and each gets its own scale.
    """
    if group is None:
        s = w.abs().amax(dim=-1, keepdim=True) / levels
        s = torch.where(s == 0, torch.ones_like(s), s)
        return torch.round(w / s).clamp(-levels, levels) * s
    out, kin = w.shape
    pad = (-kin) % group
    wp = w if pad == 0 else torch.nn.functional.pad(w, (0, pad))
    wg = wp.reshape(out, -1, group)
    s = wg.abs().amax(dim=-1, keepdim=True) / levels
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.round(wg / s).clamp(-levels, levels) * s
    return q.reshape(out, -1)[:, :kin]


class FakeQuantLinear(nn.Module):
    """Quantise weights once at construction, activations per call.

    Activation quantisation stays per TOKEN (per row) whatever the weight
    grouping, because a per-group activation scale would have to be computed
    from the activation at run time and would land in the same float-combination
    problem from the other side. Grouping the two independently is not on the
    table; this arm exists to bound the weight side.
    """

    def __init__(self, lin, group=None, quant_w=True, quant_a=True, levels=127,
                 act_levels=None):
        super().__init__()
        w = lin.weight.data.float()
        self.register_buffer("w", fq_weight(w, group, levels) if quant_w else w)
        self.register_buffer(
            "bias", lin.bias.data.float().clone() if lin.bias is not None else None)
        self.quant_a = quant_a
        self.a_levels = levels if act_levels is None else act_levels

    def forward(self, x):
        if self.quant_a:
            lv = self.a_levels
            s = x.abs().amax(dim=-1, keepdim=True) / lv
            s = torch.where(s == 0, torch.ones_like(s), s)
            x = torch.round(x / s).clamp(-lv, lv) * s
        return torch.nn.functional.linear(x, self.w, self.bias)


def apply_(model, **kw):
    """Replace EVERY Linear, `lm_head` included.

    Including it is not an oversight - `exact_model.convert` includes it, which
    is why the demo reports 169 layers on a 24-layer model (24*7 + 1). Excluding
    it here would make the `w8a8` arm measure a scheme nobody ships, and that arm
    reproducing the known +2.2% is the only check this harness has on itself.
    """
    n = 0
    for mod in list(model.modules()):
        for child, sub in list(mod.named_children()):
            if isinstance(sub, nn.Linear):
                setattr(mod, child, FakeQuantLinear(sub, **kw).to(sub.weight.device))
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppl-tokens", type=int, default=250_000)
    ap.add_argument("--groups", default="128,64,32")
    ap.add_argument("--act-levels", default="255,511,2047,32767",
                    help="activation widths to sweep, as 2^n - 1")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL)
    groups = [int(g) for g in a.groups.split(",") if g]

    arms = [("fp32", None), ("w8", dict(quant_a=False)),
            ("a8", dict(quant_w=False)), ("w8a8", dict())]
    for g in groups:
        arms.append((f"w8g{g}", dict(group=g, quant_a=False)))
    for g in groups:
        arms.append((f"w8g{g}a8", dict(group=g)))
    # Activation WIDTH, the other axis. Included unconditionally because the
    # weight arms above can only ever explain part of the gap, and if the rest
    # is activations then this is the sweep that prices the alternative. The
    # widths are `2^n - 1` for the same reason `k_levels_for` rounds that way.
    for lv in [int(x) for x in a.act_levels.split(",") if x]:
        arms.append((f"w8a{lv}", dict(act_levels=lv)))

    print(f"\nWeight-grouping ceiling, {MODEL}, wikitext-2, "
          f"{a.ppl_tokens:,} tokens")
    print("  fake quantisation in fp32 - accuracy only, no kernel, no int8")
    out = {}
    for name, kw in arms:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        ).to(a.device).eval()
        n = 0 if kw is None else apply_(model, **kw)
        ppl, ntok = perplexity(model, tok, a.device, max_tokens=a.ppl_tokens)
        out[name] = ppl
        base = out.get("fp32")
        rel = f"{(ppl / base - 1) * 100:+.2f}%" if base else "reference"
        print(f"  {name:<10} ppl {ppl:7.4f}   {rel:>8}   ({n} layers)")
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 74)
    fp32, w8a8 = out["fp32"], out["w8a8"]
    gap = w8a8 / fp32 - 1
    print(f"  shipped scheme costs {gap * 100:+.2f}% over fp32")
    print(f"    of which weights alone {(out['w8'] / fp32 - 1) * 100:+.2f}%, "
          f"activations alone {(out['a8'] / fp32 - 1) * 100:+.2f}%")
    for lv in [int(x) for x in a.act_levels.split(",") if x]:
        k = f"w8a{lv}"
        if k in out:
            r = (w8a8 - out[k]) / (w8a8 - fp32) if w8a8 > fp32 else 0.0
            print(f"    act {lv:<6} recovers {r * 100:5.1f}% of the gap "
                  f"({out[k]:.4f})")
    best, recovered = None, 0.0
    for g in groups:
        k = f"w8g{g}a8"
        r = (w8a8 - out[k]) / (w8a8 - fp32) if w8a8 > fp32 else 0.0
        print(f"    group {g:<4} recovers {r * 100:5.1f}% of the gap "
              f"({out[k]:.4f})")
        if r > recovered:
            best, recovered = g, r
    print("=" * 74)
    # The decision this tool exists to make, stated rather than left to the
    # reader. The threshold is a judgement, so it is named: folding G float
    # scales back into an integer grid is a real change to the Triton kernel and
    # to the exactness argument, and under half the gap does not pay for it.
    act_best, act_rec = None, 0.0
    for lv in [int(x) for x in a.act_levels.split(",") if x]:
        k = f"w8a{lv}"
        if k in out:
            r = (w8a8 - out[k]) / (w8a8 - fp32) if w8a8 > fp32 else 0.0
            if r > act_rec:
                act_best, act_rec = lv, r
    if recovered < 0.5 and act_rec >= 0.9:
        print(f"  VERDICT: group-wise weight scales recover only "
              f"{recovered * 100:.0f}% of the gap and are the WRONG LEVER - "
              f"weight quantisation costs {(out['w8'] / fp32 - 1) * 100:+.2f}%, "
              f"i.e. nothing. The gap is activations, and widening them to "
              f"{act_best} levels recovers {act_rec * 100:.0f}% of it.")
        print(f"  That is buildable WITHOUT giving up exactness: split the wider")
        print(f"  activation into two int8 digits (q = hi*128 + lo), run both")
        print(f"  against the same weights, and combine in int32. Every step is")
        print(f"  integer, so there is no float recombination to make")
        print(f"  order-dependent - unlike the group-scale design above.")
    elif recovered < 0.5:
        print(f"  VERDICT: group-wise weight scales recover at most "
              f"{recovered * 100:.0f}% of the gap. NOT worth the exactness "
              f"machinery - the cost is elsewhere.")
    else:
        print(f"  VERDICT: group {best} recovers {recovered * 100:.0f}% of the "
              f"gap. Worth building, and the design is to requantise the G "
              f"scales onto a common integer grid (as fold_scale_into_weights "
              f"already does for V) so the combination stays integer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
