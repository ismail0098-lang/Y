"""Is a sequence's output independent of WHO ELSE is in the batch?

    python3 tools/exact_ragged_batch.py

`batch_invariance_demo.py` decodes one prompt repeated `batch` times, so every
sequence in the batch has the same length. **Production never looks like that.**
Continuous batching mixes prompts of different lengths, pads them to the longest,
and changes the batch's composition as sequences join and finish. If invariance
only holds for uniform batches then "deterministic under load" is false in
exactly the setting the whole pitch is about.

This is the cheapest experiment that can still invalidate the central claim, so
it is worth being precise about what it varies:

  * **batch SIZE** (1 .. 32) - already covered by the demo, kept as a baseline.
  * **neighbour LENGTHS** - short, long, and mixed distractors, which changes how
    far the batch is padded and therefore the key length `T` every row sees.
  * **the target's POSITION** in the batch - first, middle, last.
  * **which neighbours** - same lengths, different text.

## Three implementation-level dependencies on batch composition

The algebra is row-local, but the *implementation* genuinely reads batch shape,
and each of these is a real chance to break:

1. `exact_pv` picks its digit width from `T` (`while T * ((1<<dbits)-1) *
   V_LEVELS >= 2^24`). Pad the batch differently and `T` changes, so the softmax
   weight is **decomposed into a different number of digits**. The split is exact
   at every width, which is what makes this safe.

   **This test is NOT the evidence for that, and it used to say it was.**
   Measured by forcing a wrong width on the padded side only, at the key lengths
   this harness actually uses (`t_true` 173-500, `t_pad` 519-744, true width 7):

       width  7   8  10  12  14  16  20
       row 0  =   =   =   =   =  MOVED MOVED

   The width can be **double** the correct value and the output is still
   bit-identical, at every shape tried. The exactness bound is worst-case and a
   real softmax's weights are nowhere near `2^28` except at the argmax, so an
   off-by-one - or an off-by-seven - in `digit_width` is invisible from here.
   Crossing a digit boundary in this test is therefore weak evidence about the
   split, however good it looks in a table.

   The gate for the split is `exact_selftest.py`'s `check_pv` plus
   `check_pv_bound_is_load_bearing`, which use uniform random `p` up to the
   inclusive `2^28` bound - the worst case, where a too-wide digit corrupts the
   answer immediately. What THIS test evidences about #1 is narrower and still
   worth having: that changing the width changes nothing observable, which is
   the property a user cares about.
2. `w8a8_matmul` picks `BLOCK_M` from `M = batch * seq`, so a different batch
   size tiles the GEMM differently. int32 accumulation is what makes that
   bit-identical.
3. Padded key positions must contribute exactly zero: `p` is forced to 0 where
   the mask blocks, so `l` (the denominator) and `wmax` (the fold's scale) are
   untouched by padding. If any of those leaked, longer neighbours would shift
   the target's softmax.

## The control is the point

`stock` must FAIL this. A test that both arms pass is measuring the harness, not
the property - and bf16 attention is genuinely sensitive to padding, so a
green-across-the-board result means the comparison is broken.
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
MAX_T = []      # key lengths reached, so the dbits claim stays checkable

# Taken verbatim from `batch_invariance_demo.PROMPTS`, because those were chosen
# for a small top-2 margin - the bf16 logit delta is ~0.34 against a margin of
# ~0.25, so a flip actually happens. A prompt the model is confident about is
# invariant in bf16 too, and then the control cannot fire.
TARGET = "Describe, step by step, how a suspension bridge carries load."

# Distractors chosen for LENGTH SPREAD, not content: the shortest is a few
# tokens and the longest is several hundred, so a batch containing one pads far
# beyond a batch containing the other.
SHORT = ["Hi.", "What is 2+2?", "Name a colour.", "Yes or no?"]
LONG = [
    ("Write a detailed technical account of how a modern optimising compiler "
     "lowers a nested loop over a two-dimensional array into vector "
     "instructions, covering dependence analysis, unrolling, register "
     "allocation and the interaction with the memory hierarchy. " * 4),
    ("Describe in exhaustive detail the process by which sedimentary rock "
     "forms, is buried, is subjected to heat and pressure, and eventually "
     "returns to the surface through tectonic uplift and erosion. " * 4),
    ("Give a thorough explanation of how a four-stroke internal combustion "
     "engine converts the chemical energy in fuel into rotational mechanical "
     "work, naming every stroke and the valve timing involved. " * 4),
]
# Long enough that the padded key length crosses `exact_pv`'s digit-width
# boundary. That boundary is at T >= 2^24/(255*127) = 518, and the LONG prompts
# above top out at 169 tokens, so with 160 generated the first version of this
# test reached T ~ 329 and **never exercised the dependency its own docstring
# claimed to test**. Measured, not assumed - the tool now prints the max T it
# reached and whether the boundary was crossed.
EXTRA_LONG = [
    ("Write an exhaustive technical account of how a modern optimising compiler "
     "lowers a nested loop over a two-dimensional array into vector "
     "instructions, covering dependence analysis, loop interchange, unrolling, "
     "register allocation, spill heuristics and the interaction with each level "
     "of the memory hierarchy. " * 11),
]
MEDIUM = [
    "Summarise how a refrigerator moves heat from inside to outside.",
    "Why does bread rise when yeast is added to the dough?",
    "Explain what a database index is and when it helps.",
    "How does noise-cancelling in headphones actually work?",
]


def build(arm, device):
    if arm == "exact":
        qwen2.eager_attention_forward = D.exact_attention
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        ).to(device).eval()
        exact_model.convert(m)
        return m
    qwen2.eager_attention_forward = D._orig
    return AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()


def decode_target(model, tok, others, pos, n_new, device):
    """Generate `n_new` tokens for TARGET sitting at index `pos` among `others`.

    Left padding, because a decoder-only model must have its prompts flush to
    the right for `generate` to continue them; `position_ids` then come from the
    attention mask. Returns the target row's new tokens only.
    """
    prompts = list(others)
    prompts.insert(pos, TARGET)
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    MAX_T.append(enc.input_ids.shape[1] + n_new)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask,
            max_new_tokens=n_new, do_sample=False, num_beams=1, use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
    return out[pos, enc.input_ids.shape[1]:].tolist()


def cases():
    """(label, neighbours, position) - each a different batch composition."""
    return [
        ("alone (batch 1)", [], 0),
        ("+3 short, first", SHORT[:3], 0),
        ("+3 short, last", SHORT[:3], 3),
        ("+3 long, first", LONG[:3], 0),
        ("+3 long, middle", LONG[:3], 1),
        ("+3 long, last", LONG[:3], 3),
        ("+1 short +1 long", [SHORT[0], LONG[0]], 1),
        ("+3 medium", MEDIUM[:3], 2),
        ("mixed 8, pos 0", SHORT + LONG[:1] + MEDIUM[:3], 0),
        ("mixed 8, pos 5", SHORT + LONG[:1] + MEDIUM[:3], 5),
        ("mixed 12, pos 7", SHORT + LONG + MEDIUM + SHORT[:1], 7),
        ("mixed 12 reordered", MEDIUM + LONG + SHORT + SHORT[:1], 7),
        # Up to 32: the demo's divergence is measured at batch 1 vs 8 vs 32, so
        # the ragged test has to reach the same batch sizes to be at least as
        # demanding as the uniform one it generalises.
        ("mixed 32, pos 0", (SHORT + MEDIUM + LONG) * 3 + SHORT[:1], 0),
        ("mixed 32, pos 17", (SHORT + MEDIUM + LONG) * 3 + SHORT[:1], 17),
        ("uniform 32 (demo shape)", [TARGET] * 31, 0),
        # These two cross the digit-width boundary: padding the batch to an
        # EXTRA_LONG neighbour changes how many base-2^dbits digits the softmax
        # weight is split into, so the target's `p @ V` is computed by a
        # DIFFERENT decomposition than when it decodes alone. Exact at every
        # width is the claim; this is what tests it.
        ("+1 extra-long (new dbits)", EXTRA_LONG, 0),
        ("mixed 8 + extra-long", SHORT + EXTRA_LONG + MEDIUM[:3], 4),
    ]


def main():
    ap = argparse.ArgumentParser()
    # 160, matching the demo. At 24 tokens the CONTROL FAILED - stock was
    # invariant across all 11 compositions - because a bf16 reduction-order
    # delta needs room to flip a greedy argmax and then diverge visibly. A
    # too-short generation makes a broken comparison look like a clean pass.
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"\nRagged-batch invariance, Qwen2.5-0.5B-Instruct, {a.tokens} new "
          f"tokens, greedy")
    print("  The SAME prompt is decoded in batches of different size, different")
    print("  neighbour lengths and different positions. Its output must not move.")
    results = {}
    for arm in ("stock", "exact"):
        model = build(arm, a.device)
        label = ("stock  bf16 + SDPA" if arm == "stock"
                 else "exact  int8, order-independent")
        print(f"\n--- {label} ---")
        base = None
        n_diff = 0
        for name, others, pos in cases():
            got = decode_target(model, tok, others, pos, a.tokens, a.device)
            if base is None:
                base = got
                print(f"  {name:<22} reference")
                continue
            same = got == base
            n_diff += 0 if same else 1
            if same:
                print(f"  {name:<22} same")
            else:
                first = next((i for i, (x, y) in enumerate(zip(base, got))
                              if x != y), 0)
                print(f"  {name:<22} DIFFERS at token {first}: "
                      f"{tok.decode(base[first:first + 6])!r} -> "
                      f"{tok.decode(got[first:first + 6])!r}")
        total = len(cases()) - 1
        results[arm] = (n_diff, total)
        print(f"  => {n_diff}/{total} batch compositions changed the output")
        del model
        torch.cuda.empty_cache()
    qwen2.eager_attention_forward = D._orig

    # The digit-width boundary is a claim about what this test COVERS, so it is
    # asserted rather than described. `V_LEVELS` is read from the module so the
    # threshold tracks the shipping width.
    import exact_kv
    thr = (1 << 24) // (255 * exact_kv.V_LEVELS)
    print(f"\n  key lengths reached: T = {min(MAX_T)} .. {max(MAX_T)}; "
          f"exact_pv changes digit width at T >= {thr}")
    if max(MAX_T) < thr:
        print("  *** the digit-width path was NOT exercised - add a longer "
              "neighbour before claiming this test covers it ***")
    else:
        print("  => the target's p@V really is decomposed differently across "
              "these batches, and the output still does not move")

    print("\n" + "=" * 74)
    for arm in ("stock", "exact"):
        d, t = results[arm]
        print(f"  {arm:<6}: {d}/{t} compositions changed the target's output")
    print("=" * 74)
    if results["stock"][0] == 0:
        print("  *** CONTROL FAILED: stock is invariant too, so this test is not "
              "exercising the property. Do not read the exact result. ***")
        return 1
    if results["exact"][0] != 0:
        print("  *** exact is NOT ragged-batch invariant - the central claim does "
              "not hold under production batching. ***")
        return 1
    print("  exact is invariant to batch size, neighbour length AND position;")
    print("  stock is not. This is the claim under production-shaped batching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
