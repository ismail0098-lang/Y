"""Does exactness cost task quality? Perplexity and multiple-choice, all arms.

    python3 tools/exact_task_accuracy.py
    python3 tools/exact_task_accuracy.py --limit 200      # quick pass

Everything in `docs/deterministic_inference.md` rests on int8 being
quality-neutral, and until now that had never been measured. §6 item 1 has been
open since the project started: the throughput figures say determinism is free,
which is worth nothing if the model got worse.

## The comparison that matters is NOT exact-vs-fp32

`stock` is bf16, and bf16 is already lossy. The question a buyer asks is "what
does switching to the deterministic path cost me *against what I deploy today*",
so **`exact - stock` is the headline** and `fp32` is the yardstick that says how
big a delta is normal. If int8's distance from fp32 is comparable to bf16's,
exactness costs nothing anyone is not already paying.

## Controls, because a harness that cannot detect damage proves nothing

`crude4` is the same weights rounded to a 4-bit per-output-channel grid - a
deliberately degraded model. If it does not score clearly worse, the harness is
not measuring quality and no "no regression" result from it means anything.
This is the same discipline as `check_pv_bound_is_load_bearing`: assert the
control fails.

Two things that paragraph described and the code did not do, both fixed:

  * **It printed the verdict and returned 0.** A run whose control failed exited
    successfully, so anything reading the exit status saw a pass -- the same
    shape as the fuzz target that reported findings with `eprintln!` and never
    panicked. Its two sibling harnesses, `exact_accuracy.py` and
    `exact_ragged_batch.py`, both `return 1` on a failed control; this one was
    the odd file out.
  * **It checked perplexity only, and the headline is the multiple-choice
    paired net.** Those are separate code paths. A collapsed MC scorer would
    make every arm score alike, "net +40 of 3,000" would read as a clean null,
    and the perplexity control would fire regardless. `crude4` now has to be
    visible in both, and in the MC case through the same paired statistic the
    conclusion is drawn from.

## What these numbers are and are not

Self-computed, not `lm-evaluation-harness`. Every arm gets byte-identical
prompts, tokenisation, batching and scoring, so **differences between arms are
meaningful**; absolute values are not comparable to published leaderboards.
Multiple-choice sets are subsampled with a fixed seed (`--limit`), which is a
further reason to read columns against each other rather than against a table
somewhere else.

Accuracy needs no interleaving, unlike every timing in this project: the
computation is deterministic, so an arm measured an hour later scores the same.
Arms are therefore built and freed one at a time to keep three 0.5B models off
the GPU at once.
"""
import _nospace

_nospace.guard()

import argparse  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402

import torch  # noqa: E402
import transformers.models.qwen2.modeling_qwen2 as qwen2  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import batch_invariance_demo as D  # noqa: E402
import exact_model  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ARMS = ("fp32", "stock", "exact", "crude4")
# `q8lin` is not in the default set: it is an ATTRIBUTION arm, not a shipping
# configuration. int8 linears with untouched fp32 attention, so
# `q8lin - fp32` is what quantising the weights costs and `exact - q8lin` is
# what quantising K/V on top of that costs. Worth having because the two have
# completely different fixes.


def crude_quantize_(model, bits=4):
    """Round every Linear weight onto a per-output-channel grid, in place.

    The sensitivity control. Deliberately cruder than the real path: symmetric,
    per-channel, round-to-nearest, no error feedback and no calibration.
    """
    levels = (1 << (bits - 1)) - 1
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear):
                w = mod.weight
                s = w.abs().amax(dim=1, keepdim=True) / levels
                s = torch.where(s == 0, torch.ones_like(s), s)
                w.copy_(torch.round(w / s).clamp(-levels, levels) * s)
    return model


def build(arm, device):
    """One model. Leaves the attention hook pointing wherever this arm needs it.

    `stock` is sdpa and never routes through `eager_attention_forward`, so the
    hook only matters for the eager arms - but it is set explicitly for each so
    the function does not depend on call order.
    """
    if arm == "exact":
        qwen2.eager_attention_forward = D.exact_attention
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        ).to(device).eval()
        exact_model.convert(m)
        return m
    qwen2.eager_attention_forward = D._orig
    if arm == "stock":
        return AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()
    if arm == "crude4":
        return crude_quantize_(m)
    if arm == "q8lin":
        exact_model.convert(m, linears=True, norms=False)   # attention untouched
    return m


# ---------------------------------------------------------------- perplexity

def perplexity(model, tok, device, window=1024, stride=512, max_tokens=250_000):
    """Sliding-window perplexity on wikitext-2 raw test.

    Only the last `stride` positions of each window are scored, so every token
    is predicted with at least `window - stride` tokens of context and no token
    is counted twice. Scoring a whole window per step instead would count early
    tokens under short context and flatter every arm equally but noisily.
    """
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[:, :max_tokens].to(device)
    nll, ntok = 0.0, 0
    prev = 0
    with torch.no_grad():
        for begin in range(0, ids.shape[1] - 1, stride):
            end = min(begin + window, ids.shape[1])
            chunk = ids[:, begin:end]
            target = chunk.clone()
            keep = end - prev                      # positions not yet scored
            target[:, :-keep] = -100
            out = model(input_ids=chunk, labels=target)
            n = int((target[:, 1:] != -100).sum())
            if n:
                nll += float(out.loss) * n
                ntok += n
            prev = end
            if end == ids.shape[1]:
                break
    return math.exp(nll / ntok), ntok


# ------------------------------------------------------- multiple choice

def _mc_items(task, limit, seed=0):
    """(context, [choices], gold) triples, subsampled with a fixed seed."""
    if task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        rows = [(f"{r['activity_label']}: {r['ctx']}", r["endings"],
                 int(r["label"])) for r in ds if r["label"] != ""]
    else:
        cfg = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        rows = []
        for r in ds:
            texts, labels = r["choices"]["text"], r["choices"]["label"]
            if r["answerKey"] not in labels:
                continue
            rows.append((f"Question: {r['question']}\nAnswer:",
                         [f" {t}" for t in texts], labels.index(r["answerKey"])))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def multiple_choice(model, tok, device, task, limit):
    """Accuracy by continuation log-likelihood: raw sum and per-token mean.

    Both are reported because they disagree in a known way - the raw sum favours
    short continuations, which is why `acc_norm` exists. An arm that moves on one
    and not the other has changed the model's length preference rather than its
    knowledge, and that is worth being able to see.
    """
    items = _mc_items(task, limit)
    hit = hit_norm = 0
    per_item = []                    # for the PAIRED comparison; see below
    with torch.no_grad():
        for ctx, choices, gold in items:
            ctx_ids = tok(ctx, return_tensors="pt").input_ids[0]
            seqs, nconts = [], []
            for ch in choices:
                cont = tok(ch, add_special_tokens=False,
                           return_tensors="pt").input_ids[0]
                seqs.append(torch.cat([ctx_ids, cont]))
                nconts.append(len(cont))
            width = max(len(s) for s in seqs)
            # LEFT-pad, so every continuation ends at the same position and the
            # scored slice is the tail of the row for all choices at once.
            inp = torch.full((len(seqs), width), tok.eos_token_id or 0,
                             dtype=torch.long)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for i, s in enumerate(seqs):
                inp[i, width - len(s):] = s
                att[i, width - len(s):] = 1
            inp, att = inp.to(device), att.to(device)
            logits = model(input_ids=inp, attention_mask=att).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = inp[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            scores, norms = [], []
            for i, nc in enumerate(nconts):
                s = float(tok_lp[i, -nc:].sum())
                scores.append(s)
                norms.append(s / nc)
            ok = int(max(range(len(scores)), key=lambda i: scores[i]) == gold)
            ok_n = int(max(range(len(norms)), key=lambda i: norms[i]) == gold)
            hit += ok
            hit_norm += ok_n
            per_item.append(ok_n)
    n = len(items)
    return 100.0 * hit / n, 100.0 * hit_norm / n, n, per_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000,
                    help="multiple-choice items per task (0 = all)")
    ap.add_argument("--ppl-tokens", type=int, default=250_000)
    ap.add_argument("--skip-mc", action="store_true",
                    help="perplexity only (for attribution runs)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-control", action="store_true",
                    help="exercise the control's own logic on synthetic scores "
                         "and exit; needs no GPU and no model")
    a = ap.parse_args()
    if a.check_control:
        return check_control_logic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL)
    arms = [x for x in a.arms.split(",") if x]

    print(f"\nQwen2.5-0.5B-Instruct   task quality, {a.limit or 'all'} items per "
          f"MC task, {a.ppl_tokens:,} ppl tokens")
    print("  fp32   = unquantised reference (the yardstick)")
    print("  stock  = bf16 + SDPA, what ships and what `exact` must not lose to")
    print("  exact  = int8, order-independent, deterministic")
    print("  crude4 = 4-bit round-to-nearest CONTROL; if this does not drop, "
          "the harness is not measuring quality")
    print(f"\n{'arm':<9}{'wikitext ppl':>14}{'hellaswag':>11}{'(norm)':>9}"
          f"{'arc_easy':>10}{'(norm)':>9}{'arc_chal':>10}{'(norm)':>9}")
    res = {}
    for arm in arms:
        model = build(arm, a.device)
        ppl, ntok = perplexity(model, tok, a.device, max_tokens=a.ppl_tokens)
        row = {"ppl": ppl, "items": {}}
        tasks = () if a.skip_mc else ("hellaswag", "arc_easy", "arc_challenge")
        for task in tasks:
            acc, acc_n, n, per = multiple_choice(model, tok, a.device, task,
                                                 a.limit)
            row[task] = (acc, acc_n)
            row["items"][task] = per
        res[arm] = row
        if a.skip_mc:
            print(f"{arm:<9}{row['ppl']:>14.3f}")
        else:
            print(f"{arm:<9}{row['ppl']:>14.3f}"
                  f"{row['hellaswag'][0]:>10.1f}%{row['hellaswag'][1]:>8.1f}%"
                  f"{row['arc_easy'][0]:>9.1f}%{row['arc_easy'][1]:>8.1f}%"
                  f"{row['arc_challenge'][0]:>9.1f}%"
                  f"{row['arc_challenge'][1]:>8.1f}%")
        del model
        torch.cuda.empty_cache()
    qwen2.eager_attention_forward = D._orig

    if "q8lin" in res and "fp32" in res and "exact" in res:
        f, q, e = res["fp32"]["ppl"], res["q8lin"]["ppl"], res["exact"]["ppl"]
        print(f"\n  where the perplexity cost comes from:")
        print(f"    int8 LINEARS      {f:.3f} -> {q:.3f}  "
              f"({100 * (q / f - 1):+.1f}%)")
        print(f"    int8 K/V on top   {q:.3f} -> {e:.3f}  "
              f"({100 * (e / q - 1):+.1f}%)")
    if "exact" in res and "stock" in res and not a.skip_mc:
        e, s = res["exact"], res["stock"]
        print(f"\n  exact - stock (the number that decides this):")
        print(f"    wikitext ppl  {e['ppl'] - s['ppl']:+.3f}  "
              f"({100 * (e['ppl'] / s['ppl'] - 1):+.1f}%)")
        for t in ("hellaswag", "arc_easy", "arc_challenge"):
            print(f"    {t:<14}{e[t][0] - s[t][0]:+.1f} pts   "
                  f"norm {e[t][1] - s[t][1]:+.1f} pts")
        # PAIRED view. Both arms answer the same items, so the accuracy delta
        # hides how much moved: two arms can score identically while disagreeing
        # on a fifth of the set. `won`/`lost` are the items only one arm got
        # right; their DIFFERENCE is the accuracy delta and their SUM is how
        # much the model actually changed. A net of ~0 on a large sum means the
        # quantisation is shuffling borderline items, not destroying knowledge.
        tw = tl = tb = 0
        for t in ("hellaswag", "arc_easy", "arc_challenge"):
            ei, si = e["items"][t], s["items"][t]
            won = sum(1 for x, y in zip(ei, si) if x and not y)
            lost = sum(1 for x, y in zip(ei, si) if y and not x)
            tw += won
            tl += lost
            tb += len(ei)
            print(f"    {t:<14}paired: exact-only {won}, stock-only {lost}, "
                  f"of {len(ei)}")
        print(f"    TOTAL         net {tw - tl:+d} of {tb} items, "
              f"{tw + tl} disagreements ({100 * (tw + tl) / max(tb, 1):.1f}% churn)")
    return report_control(res, a.skip_mc)


MC_TASKS = ("hellaswag", "arc_easy", "arc_challenge")


def control_verdict(res, skip_mc):
    """Did the harness demonstrate it can see a deliberately damaged model?

    Returns (ran, ppl_ok, mc_ok, detail). Split out from the printing so the
    logic can be exercised on synthetic scores without a GPU or a model.

    **The control used to check perplexity only, and the headline is the
    multiple-choice paired net.** Those are separate code paths: if the MC
    scorer collapsed - always picking option 0, or the length normalisation
    going flat - every arm would score identically, "net +40 of 3,000" would
    read as a beautiful null, and the perplexity control would still fire,
    because perplexity is computed somewhere else entirely. So `crude4` has to
    be visible in BOTH, and in the MC case through the same paired statistic
    the conclusion is drawn from.
    """
    if "crude4" not in res or "fp32" not in res:
        return False, False, False, "control arms not run (--arms)"
    c, f = res["crude4"], res["fp32"]
    ppl_ok = c["ppl"] > f["ppl"] * 1.10
    detail = [f"ppl {c['ppl']:.3f} vs fp32 {f['ppl']:.3f}"]
    if skip_mc:
        return True, ppl_ok, True, "; ".join(detail) + "; MC skipped"
    won = lost = tot = 0
    for t in MC_TASKS:
        ci, fi = c["items"][t], f["items"][t]
        won += sum(1 for x, y in zip(ci, fi) if x and not y)
        lost += sum(1 for x, y in zip(ci, fi) if y and not x)
        tot += len(ci)
    drop = sum(f[t][0] - c[t][0] for t in MC_TASKS) / len(MC_TASKS)
    # Both halves matter. A pooled drop alone could come from one task; a
    # negative paired net alone could be a handful of items on a tiny set.
    mc_ok = drop >= 2.0 and (lost - won) >= max(20, tot // 100)
    detail.append(f"MC paired net {won - lost:+d} of {tot}, mean acc {-drop:+.1f} pts")
    return True, ppl_ok, mc_ok, "; ".join(detail)


def check_control_logic():
    """Exercise `control_verdict` on synthetic scores. No GPU, no model.

    The real harness needs a GPU and three 0.5B models, so the control that
    decides whether a run is believable could not itself be checked on the
    machine most people have. These five scenarios do it in milliseconds, and
    the second one is the whole reason the MC half exists: it is a run where the
    multiple-choice scorer has collapsed so every arm scores alike, which the
    old perplexity-only control passed.

        python3 tools/exact_task_accuracy.py --check-control
    """
    import random

    n = 1000

    def items(frac, seed):
        r = random.Random(seed)
        return {t: [r.random() < frac for _ in range(n)] for t in MC_TASKS}

    def arm(ppl, acc, it):
        d = {"ppl": ppl, "items": it}
        for t in MC_TASKS:
            d[t] = (acc, acc)
        return d

    fp32 = arm(13.86, 55.0, items(0.55, 1))
    good = arm(40.0, 30.0, items(0.30, 2))
    same = arm(40.0, 55.0, items(0.55, 1))          # MC scorer collapsed
    scenarios = [
        ("crude4 clearly worse in both", {"fp32": fp32, "crude4": good}, 0),
        ("multiple-choice scorer collapsed", {"fp32": fp32, "crude4": same}, 1),
        ("perplexity path blind",
         {"fp32": fp32, "crude4": arm(13.9, 30.0, items(0.30, 2))}, 1),
        ("both blind", {"fp32": fp32, "crude4": arm(13.9, 55.0, items(0.55, 1))}, 1),
        ("control arms not run", {"fp32": fp32}, 0),
    ]
    bad = 0
    for name, res, want in scenarios:
        ran, ppl_ok, mc_ok, _ = control_verdict(res, skip_mc=False)
        got = 0 if not ran else (0 if (ppl_ok and mc_ok) else 1)
        flag = "ok  " if got == want else "WRONG"
        bad += got != want
        print(f"  {flag}  {name:<36} exit {got} (want {want})")
    if bad:
        print(f"\n{bad} scenario(s) wrong: the control's own logic is broken.")
        return 1
    print("\ncontrol logic behaves on all 5 scenarios.")
    return 0


def report_control(res, skip_mc):
    ran, ppl_ok, mc_ok, detail = control_verdict(res, skip_mc)
    if not ran:
        print(f"\n  control: NOT RUN -- {detail}.")
        print("    This run licenses no 'no regression' conclusion; the default "
              "--arms includes crude4 for exactly this reason.")
        return 0
    ok = ppl_ok and mc_ok
    print(f"\n  control (crude4, 4-bit): {detail} -> "
          f"{'DETECTED' if ok else 'NOT DETECTED'}")
    if not ok:
        which = []
        if not ppl_ok:
            which.append("perplexity")
        if not mc_ok:
            which.append("multiple choice")
        print(f"    *** the harness cannot see a deliberately damaged model in "
              f"{' or '.join(which)}; no 'no regression' conclusion from this "
              f"run is valid ***")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
