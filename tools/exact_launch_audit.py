"""Is decode launch-bound? Kernels per step, device time, and the gap.

    python3 tools/exact_launch_audit.py --compile

M5 step 8 concluded "what remains is launch overhead, not arithmetic" from a
kernel-mix argument: dozens of ~5 us kernels at ~13% SM throughput. That is
suggestive, not a measurement. This tool measures it directly, because the
next optimisation (CUDA graphs) is only worth building if the gap is real.

The number that matters is **wall - device**: the part of a decode step during
which the GPU is not running a kernel. If it is small, the step is
arithmetic-bound and capturing the launches changes nothing; if it is most of
the step, launch overhead is the whole remaining target.

Reported per arm, so the *difference* is visible: a launch-overhead fix helps
whichever arm launches more, and stock (SDPA, bf16, one fused attention kernel)
launches far fewer than a path that spells attention out in torch ops.

Contention note: `wall - device` is a difference of two quantities that both
inflate under load, so read it on an idle GPU. The kernel COUNT does not move
with contention and is the robust half of this report.
"""
import _nospace

_nospace.guard()

import argparse  # noqa: E402
import time  # noqa: E402

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


def one_round(model, tok, batch, prefill_n, steps, device):
    """One (wall, device, kernel count) sample for one arm.

    Wall and device come from two separate decode runs of the same length, so
    on a contended GPU they see different interference. That is tolerable only
    because the caller INTERLEAVES the arms and takes minima; measuring the
    arms one after the other - which the first version of this file did, six
    minutes apart while another process held the GPU - put all of that drift
    into the difference being reported, and it produced a device-time ratio
    16% away from the NCU figure for the same quantity.
    """
    past, nxt = prefill(model, tok, batch, prefill_n, device)
    past, nxt = decode(model, past, nxt, 2)              # settle
    torch.cuda.synchronize()

    t = time.perf_counter()
    past, nxt = decode(model, past, nxt, steps)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t) / steps * 1e3

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        decode(model, past, nxt, steps)
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    n = sum(e.count for e in ev)
    dev = sum(e.self_device_time_total for e in ev) / steps / 1e3
    return n / steps, dev, wall


def audit_interleaved(models, tok, batch, prefill_n, steps, device, reps):
    for m in models.values():                            # warm up / compile
        for _ in range(3):
            past, nxt = prefill(m, tok, batch, prefill_n, device)
            decode(m, past, nxt, steps)
    torch.cuda.synchronize()
    acc = {k: [] for k in models}
    for _ in range(reps):
        for name, m in models.items():
            acc[name].append(one_round(m, tok, batch, prefill_n, steps, device))
    out = {}
    for name, rows in acc.items():
        out[name] = (rows[0][0],                          # count: invariant
                     min(r[1] for r in rows),
                     min(r[2] for r in rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prefill", type=int, default=64)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL)

    print(f"\nQwen2.5-0.5B  batch {a.batch}  decode only  "
          f"compile={'on' if a.compile else 'off'}")
    print(f"{'arm':<10}{'kernels/step':>14}{'device ms':>11}{'wall ms':>10}"
          f"{'gap ms':>9}{'gap %':>8}")
    # Both arms are built before anything is measured so they can be
    # interleaved. Leaving the hook on the exact path afterwards is safe: the
    # stock arm is `sdpa`, which never routes through `eager_attention_forward`.
    stock = build("stock", a.device, a.compile)
    exact = build("exact", a.device, a.compile)
    out = audit_interleaved({"stock": stock, "exact": exact}, tok, a.batch,
                            a.prefill, a.steps, a.device, a.reps)
    for arm in ("stock", "exact"):
        n, dev, wall = out[arm]
        print(f"{arm:<10}{n:>14.0f}{dev:>11.3f}{wall:>10.3f}"
              f"{wall - dev:>9.3f}{100 * (wall - dev) / wall:>7.0f}%")
    qwen2.eager_attention_forward = D._orig

    ns, ds, ws = out["stock"]
    ne, de, we = out["exact"]
    print(f"\n  exact launches {ne / ns:.2f}x the kernels of stock")
    print(f"  exact / stock:  device {de / ds:.2f}x   wall {we / ws:.2f}x")
    print(f"  if every launch gap vanished, exact/stock would be "
          f"{de / ds:.2f}x  (device-time floor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
