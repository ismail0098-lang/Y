"""A fixed decode workload for `ncu` to profile. Not a benchmark.

    ncu --set full --profile-from-start off \
        -o /tmp/exact python3 tools/ncu_workload.py --arm exact

`generate()` is the wrong thing to hand a profiler: its kernel mix shifts with
sampling, cache growth and early-stopping, so two runs are not the same
population. This drives the model directly for a fixed number of decode steps
at a fixed cache length, warms up first, and only then turns the profiler on -
so every profiled launch belongs to steady-state decode.

`--profile-from-start off` is required; the torch and inductor warmup would
otherwise dominate the report with kernels that never run in steady state.
"""
import _nospace

_nospace.guard()

import argparse  # noqa: E402

import torch  # noqa: E402
import transformers.models.qwen2.modeling_qwen2 as qwen2  # noqa: E402
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


def decode_steps(model, past, nxt, steps):
    """`steps` single-token decodes from an existing cache.

    Prefill is deliberately OUTSIDE this function and outside the profiled
    window. At batch 32 / prompt 64 a prefill GEMM has M = 2048 against decode's
    M = 32, so including it makes the average `us/call` of every linear layer a
    blend of two regimes that need different fixes - the first run of this
    profile reported 76.8 us for a decode GEMM that actually costs 6.
    """
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
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = build(a.arm, a.device, a.compile)

    for _ in range(a.warmup):
        past, nxt = prefill(model, tok, a.batch, a.prefill, a.device)
        decode_steps(model, past, nxt, a.steps)
    torch.cuda.synchronize()

    past, nxt = prefill(model, tok, a.batch, a.prefill, a.device)
    past, nxt = decode_steps(model, past, nxt, 2)   # settle at steady state
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    decode_steps(model, past, nxt, a.steps)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"profiled {a.steps} DECODE steps (prefill excluded), "
          f"arm={a.arm}, batch={a.batch}")


if __name__ == "__main__":
    main()
