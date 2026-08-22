"""What does determinism actually cost? Decode throughput, both arms, same work.

    python3 tools/exact_throughput.py --batch 32 --tokens 64
    python3 tools/exact_throughput.py --batch 32 --tokens 64 --compile

The first measurement of this (M5 step 3) reported the exact path at **6.7x
slower** and attributed it to "the prototype". That was honest but useless: it
named no cause and pointed at no fix. This tool exists to replace it with a
number that has a cause attached.

## The rule this harness exists to enforce

**Both arms get the same treatment.** If `--compile` is passed, it is passed to
the stock model too. Compiling only the challenger is the same bias that was
already caught twice in this project - once when an f32 attention baseline was
charged for work the exact path did in an untimed pass, and once when a
baseline got exact fp32 logits while the candidate was quantised. A speedup
measured against a handicapped control is not a speedup.

The stock arm is bf16 + SDPA, which is what production serves, and is therefore
the *harder* comparison: it is also the arm that is allowed to be wrong.
"""
import _nospace

_nospace.guard()

import argparse  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import transformers.models.qwen2.modeling_qwen2 as qwen2  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import batch_invariance_demo as D  # noqa: E402
import exact_model  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = ("Explain, step by step and in detail, how a suspension bridge carries "
          "the load of the traffic crossing it down into the ground. ")


def build(arm, device, compile_it, graphs=False):
    if arm == "fixedfloat":
        # The control the project's thesis rests on: float32 with the reduction
        # ORDER pinned instead of the arithmetic made exact. It is
        # batch-invariant too (0/16, `exact_ragged_batch.py`), so what it costs
        # is the whole question. See tools/fixed_order_float.py.
        import fixed_order_float
        qwen2.eager_attention_forward = fixed_order_float.fixed_order_attention
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        ).to(device).eval()
        fixed_order_float.convert(m)
        return torch.compile(m) if compile_it else m
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
        # Applied identically to both arms - see the module docstring.
        # `graphs` swaps dynamic tracing for CUDA-graph capture, which needs
        # static shapes: `mode="reduce-overhead"` and `dynamic=False`, paired
        # with `cache_implementation="static"` at the generate call. This exists
        # because vLLM's ENTIRE advantage over this baseline is CUDA graphs
        # (M5 step 12: 356.4 tok/s/seq with them, 241.2 without, against compiled
        # HF's 241.9), so the question is whether the same feature is available
        # to the deterministic path.
        if graphs:
            compiled = torch.compile(m.forward, dynamic=False,
                                     mode="reduce-overhead")

            def fwd(*args, _c=compiled, **kw):
                # Required, not optional: inductor's cudagraph trees reuse the
                # capture's output buffers, and HF's static cache writes into
                # them in place (`index_copy_`), so without a step boundary the
                # run dies with "accessing tensor output of CUDAGraphs that has
                # been overwritten". This is the documented remedy.
                torch.compiler.cudagraph_mark_step_begin()
                return _c(*args, **kw)

            m.forward = fwd
        else:
            m.forward = torch.compile(m.forward, dynamic=True)
    return m


def one_run(model, inp, att, kw):
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        model.generate(input_ids=inp, attention_mask=att, **kw)
    torch.cuda.synchronize()
    return time.perf_counter() - t


def run_interleaved(models, tok, batch, tokens, device, reps, graphs=False,
                    static_cache=False):
    """Time the arms round-robin, not one after the other.

    The first version of this ran stock to completion and then exact, and a
    measurement caught it out: between the two halves of a run the stock arm
    moved 250.7 -> 223.7 tok/s/seq, 12%, because an unrelated process had taken
    the GPU. Sequential arms turn any drift - thermal, clock, another tenant -
    into a difference between the things being compared. Round-robin puts the
    same drift into both.

    The MINIMUM of the reps is reported, not the mean: a slower run had
    interference in it and the fastest is the least contaminated one.
    """
    ids = tok(PROMPT, return_tensors="pt").to(device)
    inp = ids.input_ids.repeat(batch, 1)
    att = ids.attention_mask.repeat(batch, 1)
    kw = dict(max_new_tokens=tokens, do_sample=False, num_beams=1,
              use_cache=True, pad_token_id=tok.eos_token_id)
    if graphs or static_cache:
        # A static cache is what makes the shapes constant enough to capture.
        # It also PADS attention to `max_cache_len` every step, so the win has
        # to beat that extra work - which is the cost this experiment measures
        # rather than assumes.
        kw["cache_implementation"] = "static"
    for m in models.values():                   # warm up + let inductor compile
        with torch.no_grad():
            m.generate(input_ids=inp, attention_mask=att,
                       **dict(kw, max_new_tokens=min(8, tokens)))
        one_run(m, inp, att, kw)
    times = {k: [] for k in models}
    for _ in range(reps):
        for name, m in models.items():
            times[name].append(one_run(m, inp, att, kw))
    return {k: (min(v), statistics.median(v)) for k, v in times.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--graphs", action="store_true",
                    help="CUDA graphs via static cache + reduce-overhead, BOTH arms")
    ap.add_argument("--static-cache", action="store_true",
                    help="static cache WITHOUT graphs: prices the padding alone")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL)

    print(f"\nQwen2.5-0.5B  batch {a.batch}  {a.tokens} new tokens  "
          f"compile={'on (BOTH arms)' if a.compile else 'off'}"
          f"{'  cuda-graphs=on (BOTH arms)' if a.graphs else ''}"
          f"{'  static-cache=on (BOTH arms)' if a.static_cache else ''}  "
          f"best of {a.reps}")
    print(f"{'arm':<38}{'s':>8}{'tok/s/seq':>12}{'tok/s total':>13}{'spread':>9}")
    # Both models are built BEFORE anything is timed, so the arms can be
    # interleaved. `build` rebinds the global attention hook, so the exact
    # model's modules must be constructed while the exact hook is installed -
    # they capture it at call time, which is why the hook is left pointing at
    # the exact path afterwards and restored at the end.
    arms = os.environ.get("Y_THR_ARMS", "stock,exact").split(",")
    models = {}
    for name in arms:
        models[name] = build(name, a.device, a.compile, a.graphs)
    res = run_interleaved(models, tok, a.batch,
                          a.tokens, a.device, a.reps, a.graphs, a.static_cache)
    LABELS = {"stock": "stock  bf16 + SDPA (what ships)",
              "exact": "exact  int8, order-independent",
              "fixedfloat": "fixedfloat  fp32, reduction order pinned"}
    for arm, label in ((n, LABELS.get(n, n)) for n in arms):
        best, med = res[arm]
        print(f"{label:<38}{best:>8.3f}{a.tokens / best:>12.1f}"
              f"{a.batch * a.tokens / best:>13.1f}{100 * (med - best) / best:>8.1f}%")
    qwen2.eager_attention_forward = D._orig
    for n in arms:
        if n != "stock" and "stock" in res:
            print(f"\n  {n} / stock = {res[n][0] / res['stock'][0]:.2f}x slower")
    if "exact" in res and "fixedfloat" in res:
        print(f"  fixedfloat / exact = "
              f"{res['fixedfloat'][0] / res['exact'][0]:.2f}x slower")
    worst = max(100 * (m - b) / b for b, m in res.values())
    if worst > 5.0:
        print(f"  NOTE: median is {worst:.1f}% above the minimum on one arm - the "
              f"machine was busy; re-run before quoting this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
