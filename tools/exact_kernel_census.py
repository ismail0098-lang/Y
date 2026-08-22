"""Which kernels does a decode step launch, and how many of each?

    python3 tools/exact_kernel_census.py --arm exact --compile

`exact_launch_audit.py` says the exact path launches 1.41x the kernels of stock
and is nonetheless FASTER on device time - so the remaining deficit is launch
count, not arithmetic. Two ways to attack that: capture the step in a CUDA
graph (pays for every launch at once, but forces a static cache, and padding
attention to a fixed length is not free), or delete launches outright.

This tool exists to price the second option before committing to the first.
A kernel COUNT is invariant under GPU contention, unlike every timing in this
project, so this report is trustworthy on a busy machine.
"""
import _nospace

_nospace.guard()

import argparse  # noqa: E402
import collections  # noqa: E402

import torch  # noqa: E402
import transformers.models.qwen2.modeling_qwen2 as qwen2  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import batch_invariance_demo as D  # noqa: E402
import exact_model  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build(arm, device, compile_it):
    if arm == "exact":
        qwen2.eager_attention_forward = D.exact_attention
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        ).to(device).eval()
        exact_model.convert(m)
    else:
        qwen2.eager_attention_forward = D._orig
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()
    if compile_it:
        m.forward = torch.compile(m.forward, dynamic=True)
    return m


def prefill(model, tok, batch, n, device):
    ids = tok("word " * n, return_tensors="pt").to(device)
    inp = ids.input_ids[:, :n].repeat(batch, 1)
    with torch.no_grad():
        out = model(input_ids=inp, use_cache=True)
    return out.past_key_values, out.logits[:, -1:].argmax(-1)


def decode(model, past, nxt, steps):
    with torch.no_grad():
        for _ in range(steps):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
    return past, nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("exact", "stock"), default="exact")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prefill", type=int, default=64)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--top", type=int, default=22)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = build(a.arm, a.device, a.compile)

    for _ in range(3):
        past, nxt = prefill(model, tok, a.batch, a.prefill, a.device)
        past, nxt = decode(model, past, nxt, a.steps)
    torch.cuda.synchronize()
    past, nxt = prefill(model, tok, a.batch, a.prefill, a.device)
    past, nxt = decode(model, past, nxt, 2)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        decode(model, past, nxt, a.steps)
        torch.cuda.synchronize()

    cnt, tim = collections.Counter(), collections.Counter()
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        cnt[e.key] += e.count
        tim[e.key] += e.self_device_time_total
    n = sum(cnt.values()) / a.steps
    t = sum(tim.values()) / a.steps / 1e3
    print(f"\narm={a.arm}  batch {a.batch}  {n:.0f} kernels/step, {t:.3f} ms/step "
          f"device  ({n / 24:.1f} kernels per layer over 24 layers)")
    print(f"\n{'launches/step':>14}{'us/step':>10}{'us each':>10}  kernel")
    for k, c in cnt.most_common(a.top):
        per = c / a.steps
        us = tim[k] / a.steps
        print(f"{per:>14.1f}{us:>10.1f}{us / max(per, 1e-9):>10.2f}  {k[:78]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
