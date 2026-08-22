"""What does vLLM do on the same work? The baseline that decides "at parity".

Run in the ISOLATED venv, not the project one:

    /tmp/.../scratchpad/vllmenv/bin/python tools/vllm_baseline.py \
        --batch 32 --tokens 64

vLLM pins its own torch; installing it beside the exact path risks downgrading
`torch` under a working environment. It lives in its own venv for that reason
and this script imports nothing from the project.

## Why this exists

Every throughput figure in `docs/deterministic_inference.md` compares against
`stock` = bf16 + SDPA under `torch.compile`. That is a legitimate control — it
is the same framework, same model, same harness, and it is what makes the
*ratio* trustworthy. It is **not** what anyone deploys. A production serving
stack does continuous batching, paged KV, CUDA graphs and a fused sampler, none
of which HuggingFace `generate` does.

So "exact is at parity" is currently a statement about a torch prototype against
a torch baseline. This measures the third point: **how far the torch baseline
itself is from a real server**. If vLLM is 2x `stock`, then "parity" has to be
restated as "parity with compiled HF, 2x off production", which is a different
sentence and the honest one.

## What is and is not comparable

vLLM's `generate` includes its own scheduling and detokenisation, and it batches
continuously rather than in a fixed lockstep. To keep the work identical this
sends **one fixed prompt repeated `batch` times, greedy, `ignore_eos=True` with
`max_tokens` fixed**, so every sequence produces exactly the same token count as
the HF arms do and no sequence finishes early. Without `ignore_eos` a short
generation makes vLLM look faster by doing less work.

Reported as tokens/s/seq and total tokens/s, same as `exact_throughput.py`,
best of N with the spread printed - a busy machine must be visible in the
output, not silently folded into the result.
"""
import argparse
import statistics
import time

PROMPT = ("Explain, step by step and in detail, how a suspension bridge carries "
          "the load of the traffic crossing it down into the ground. ")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable CUDA graphs, for a closer-to-HF comparison")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.55,
              enforce_eager=a.enforce_eager, disable_log_stats=True,
              max_model_len=2048)
    # ignore_eos so every sequence emits exactly `tokens`, matching the HF arms.
    sp = SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True)
    prompts = [PROMPT] * a.batch

    llm.generate(prompts, sp)                                   # warm up
    times = []
    for _ in range(a.reps):
        t = time.perf_counter()
        out = llm.generate(prompts, sp)
        times.append(time.perf_counter() - t)
    produced = sum(len(o.outputs[0].token_ids) for o in out)

    best, med = min(times), statistics.median(times)
    print(f"\nvLLM {a.model}  batch {a.batch}  {a.tokens} new tokens  "
          f"cuda_graphs={'off' if a.enforce_eager else 'on'}  best of {a.reps}")
    print(f"{'arm':<38}{'s':>8}{'tok/s/seq':>12}{'tok/s total':>13}{'spread':>9}")
    print(f"{'vllm   bf16, continuous batching':<38}{best:>8.3f}"
          f"{a.tokens / best:>12.1f}{a.batch * a.tokens / best:>13.1f}"
          f"{100 * (med - best) / best:>8.1f}%")
    print(f"\n  produced {produced} tokens ({produced / a.batch:.1f} per seq; "
          f"must equal {a.tokens} or the arms are not doing equal work)")
    if produced != a.batch * a.tokens:
        print("  *** token counts differ - do NOT compare this against the HF "
              "arms until they match ***")
    if 100 * (med - best) / best > 5.0:
        print("  NOTE: median is >5% above the minimum - the machine was busy; "
              "re-run before quoting this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
