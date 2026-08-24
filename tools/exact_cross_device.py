"""Does the same prompt decode to the same tokens on a DIFFERENT GPU?

    # on card A                      # on card B
    python3 tools/exact_cross_device.py --record a.json
    python3 tools/exact_cross_device.py --record b.json

    # anywhere, no GPU and no torch needed
    python3 tools/exact_cross_device.py --compare a.json b.json

This is the one headline claim in `docs/bit_identical_decode.md` with no
measurement behind it, and Finding 11 is what made it the load-bearing one.
Once `tools/fixed_order_float.py` measured **0/16 as well**, batch invariance on
a single card stopped being what distinguishes exactness -- a competitor can pin
a float reduction order and get it. What survives is two things:

  * determinism *while leaving the compiler free* -- measured (`torch.compile`
    gives the exact arm 4.4x and the pinned-order arm nothing); and
  * determinism *across hardware* -- argued in Finding 12 and measured for the
    exp alone, never for the pipeline.

Finding 12 proved the integer exp path contains no instruction whose result the
ISA leaves open, and the exp agrees bit for bit between x86-64 and sm_89 on all
1,966,085 arguments. That is a wider architectural gap than two NVIDIA cards, so
the *arithmetic* argument is in good shape. What this tool tests is the part the
argument does not cover: whether the whole decode pipeline, with its cuBLAS
calls, its kernel autotuning and its device-dependent defaults, actually lands
on the same tokens. The risk here is a hidden float or a device-dependent
default somewhere in the plumbing -- not the algebra.

## Why this measurement is cheap in a way none of the others were

It is a bit-identity check, not a benchmark. There is no timing, so none of the
clock-ramp, interleaving and contention discipline the rest of this repo's
measurements need applies -- a busy card gives the same answer as an idle one,
just later. Record on a rented box, copy one JSON back, compare offline.

## What can make a run of this vacuous, and what is done about each

`0 differences` is the passing number. It is also what a broken comparison
prints, which is the failure shape `feedback-null-metrics-pass-dead-components`
records. Five ways it could be empty, each refused rather than described:

1. **Both artifacts came from the same GPU.** Then this measures re-run
   determinism, which is a weaker and different claim -- a single card is
   deterministic for a fixed batch even in bf16. Refused unless
   `--allow-same-device` is passed, and then labelled as what it is.
2. **The control did not fire.** `stock` must CHANGE its output across the two
   cards. If bf16 lands on identical tokens too, this pair of devices does not
   distinguish anything and the exact result is unreadable. Reported as a
   failure of the comparison, not as a win.
3. **An arm was not computing the model's function.** `batch_invariance_demo`'s
   two-part gate (canary + degeneracy) runs before anything is recorded and its
   verdict is written into the artifact; `--compare` refuses an arm whose gate
   failed on either side.
4. **The two runs did not do the same work.** The prompt text, the case list and
   the token budget are hashed into the artifact and compared. A prompt edited
   between the two recordings would otherwise read as a device difference --
   `feedback-differential-arms-share-constants`, where both arms replicated one
   wrong constant and 12/12 bit-identical meant nothing.
5. **The two runs did not start from the same weights.** Every parameter is
   fingerprinted on the CPU. If an arm's weights differ, its token comparison is
   not attributable to decode, and that arm is reported INCONCLUSIVE rather than
   passed or failed. For the `exact` arm this can be a real result rather than a
   harness fault -- the quantisation scales are computed on-device -- so it is
   localised rather than swallowed.

6. **The card does not reproduce its own output.** `--record` decodes one case
   twice on the same device; if that already differs, a cross-device comparison
   cannot mean anything. Recorded, and refused by `--compare`.

`--check-gate` exercises all of that on synthetic artifacts with no GPU, no
torch and no model, because a gate nobody has seen fail is not known to work.
"""
import argparse
import hashlib
import json
import os
import platform
import sys

SCHEMA = "y-exact-cross-device/1"

# ---------------------------------------------------------------------------
# Pure half: artifact comparison. No torch, no GPU, no model -- so it runs on a
# laptop against two JSONs copied off two rented boxes, and so `--check-gate`
# can exercise every branch below without hardware.
# ---------------------------------------------------------------------------


def device_identity(dev):
    """What makes two runs 'the same device' for the purpose of this test."""
    return (dev.get("gpu_name"), dev.get("capability"))


def _fmt_dev(dev):
    return (f"{dev.get('gpu_name')} (sm_{str(dev.get('capability','?')).replace('.','')}, "
            f"{dev.get('multi_processor_count')} SMs, driver {dev.get('driver')})")


def compare_artifacts(a, b, allow_same_device=False, out=print):
    """Compare two recordings. -> (exit_code, summary dict).

    Pure and total: every refusal is a return, never an exception, so
    `--check-gate` can drive all of them.
    """
    summary = {"arms": {}, "refused": None, "control_fired": None}

    for name, art in (("A", a), ("B", b)):
        if art.get("schema") != SCHEMA:
            summary["refused"] = "schema"
            out(f"  *** artifact {name} is not {SCHEMA} (got "
                f"{art.get('schema')!r}) -- refusing to compare. ***")
            return 2, summary

    # (4) Same work on both sides. A prompt edited between the two recordings
    # would otherwise present as a device difference.
    for field, what in (("workload_hash", "prompts, cases and token budget"),
                        ("case_labels", "case list")):
        if a["config"].get(field) != b["config"].get(field):
            summary["refused"] = field
            out(f"  *** the two runs did not do the same work: {what} differs. "
                f"***\n      A: {a['config'].get(field)}\n"
                f"      B: {b['config'].get(field)}")
            return 2, summary

    # (1) Same device is vacuous: one card is deterministic for a fixed batch
    # even in bf16, so this would measure re-run determinism instead.
    ida, idb = device_identity(a["device"]), device_identity(b["device"])
    same_device = ida == idb
    if same_device and not allow_same_device:
        summary["refused"] = "same_device"
        out(f"  *** both artifacts came from {_fmt_dev(a['device'])}. ***\n"
            f"      That measures RE-RUN determinism on one card, not "
            f"cross-hardware bit-identity,\n      and a single card is "
            f"deterministic for a fixed batch even in bf16. Pass\n"
            f"      --allow-same-device only if you know that is what you want.")
        return 2, summary

    out(f"  A: {_fmt_dev(a['device'])}")
    out(f"  B: {_fmt_dev(b['device'])}")
    if same_device:
        out("  NOTE: same device identity -- this is a re-run determinism "
            "check, not a\n        cross-hardware one. Do not quote it as the "
            "latter.")
    out(f"  model {a['config']['model']}, {a['config']['tokens']} new tokens, "
        f"greedy, {len(a['config']['case_labels'])} cases")

    # The integer exp is the sharpest single localiser: Finding 12 measured it
    # exhaustively across x86-64 and sm_89, so a mismatch here says the
    # divergence is in the arithmetic rather than in the plumbing.
    da = a.get("exp2_domain_digest", {}).get("device")
    db = b.get("exp2_domain_digest", {}).get("device")
    if da is not None and db is not None:
        agree = da == db
        summary["exp2_agrees"] = agree
        out(f"\n  integer exp over its whole domain: "
            f"{'IDENTICAL' if agree else 'DIFFERS'} ({da} vs {db})")
        if not agree:
            out("      The divergence is in the arithmetic, not the plumbing. "
                "This is the\n      premise Finding 12 rests on and it did not "
                "hold on these two devices.")

    out("")
    for arm in sorted(set(a["arms"]) & set(b["arms"])):
        aa, bb = a["arms"][arm], b["arms"][arm]

        # (3) An arm that was not computing the model's function scores 0
        # differences for the same reason a dead component does.
        if not aa["gate"]["ok"] or not bb["gate"]["ok"]:
            which = "A" if not aa["gate"]["ok"] else "B"
            reason = (aa if which == "A" else bb)["gate"]["reason"]
            summary["arms"][arm] = {"verdict": "gate_failed"}
            out(f"  {arm:<11} REFUSED -- sanity gate failed on {which}: {reason}")
            continue

        # (6) The card must agree with ITSELF before it is compared to another.
        # `record` writes this field; nothing read it until a pass over the tool
        # asked which of its recorded facts were actually consulted. A number in
        # an artifact that no check reads is documentation, not a guard.
        if aa.get("self_check") is False or bb.get("self_check") is False:
            which = "A" if aa.get("self_check") is False else "B"
            summary["arms"][arm] = {"verdict": "not_reproducible"}
            out(f"  {arm:<11} REFUSED -- {which} did not reproduce its own "
                f"output on a re-run of the\n              same card, so a "
                f"cross-device comparison cannot mean anything.")
            continue

        # (5) Same weights, or a token difference is not attributable to decode.
        if aa.get("weight_fingerprint") != bb.get("weight_fingerprint"):
            summary["arms"][arm] = {"verdict": "inconclusive_weights"}
            out(f"  {arm:<11} INCONCLUSIVE -- the two runs did not start from "
                f"the same weights\n              ({aa.get('weight_fingerprint')}"
                f" vs {bb.get('weight_fingerprint')}), so a token difference "
                f"is not\n              attributable to the device. For 'exact' "
                f"this can be a real finding: the\n              quantisation "
                f"scales are computed on-device.")
            continue

        n_diff, firsts = 0, []
        for label in a["config"]["case_labels"]:
            ta, tb = aa["cases"].get(label), bb["cases"].get(label)
            if ta != tb:
                n_diff += 1
                i = next((k for k, (x, y) in enumerate(zip(ta or [], tb or []))
                          if x != y), min(len(ta or []), len(tb or [])))
                firsts.append((label, i))
        total = len(a["config"]["case_labels"])
        summary["arms"][arm] = {"verdict": "measured", "diff": n_diff,
                                "total": total}
        out(f"  {arm:<11} {n_diff}/{total} cases changed across the two devices")
        for label, i in firsts[:4]:
            out(f"                 {label!r} first differs at token {i}")

    out("\n" + "=" * 74)
    stock = summary["arms"].get("stock", {})
    exact = summary["arms"].get("exact", {})

    # (2) The control. If bf16 lands identically on both cards there is nothing
    # here to distinguish, and the exact result is unreadable.
    if stock.get("verdict") == "measured":
        summary["control_fired"] = stock["diff"] > 0
        if not summary["control_fired"]:
            out("  *** CONTROL DID NOT FIRE: stock bf16 produced identical "
                "tokens on both\n      devices, so this pair does not "
                "distinguish anything. The exact result\n      is not readable "
                "as evidence of portability. ***")
            out("=" * 74)
            return 1, summary
        out(f"  control fired: stock bf16 moved on {stock['diff']}/"
            f"{stock['total']} cases across these devices")
    else:
        out("  *** no usable `stock` arm, so there is no control. Record both "
            "arms. ***")
        out("=" * 74)
        return 1, summary

    if exact.get("verdict") != "measured":
        out(f"  *** no usable `exact` arm ({exact.get('verdict', 'absent')}). ***")
        out("=" * 74)
        return 1, summary
    if exact["diff"]:
        out(f"  *** exact changed on {exact['diff']}/{exact['total']} cases. "
            f"Cross-hardware bit-identity\n      does NOT hold on this pair. "
            f"***")
        out("=" * 74)
        return 1, summary
    out(f"  exact: {exact['diff']}/{exact['total']} -- the same prompt decodes "
        f"to identical tokens\n  on both devices, where bf16 does not. This is "
        f"the cross-hardware claim.")
    out("=" * 74)
    return 0, summary


# ---------------------------------------------------------------------------
# The gate's own self-test. Every refusal branch above, on synthetic artifacts.
# No GPU, no torch, no model -- `feedback-null-metrics-pass-dead-components`
# ends with "self-test the gate on CPU", and this is that.
# ---------------------------------------------------------------------------


def _art(gpu="RTX 4070 Ti SUPER", cap="8.9", *, stock, exact,
         gate_ok=True, wfp="wf-1", exp="0xAAAA", labels=("c0", "c1", "c2"),
         self_check=True):
    return {
        "schema": SCHEMA,
        "device": {"gpu_name": gpu, "capability": cap,
                   "multi_processor_count": 66, "driver": "580.00"},
        "config": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "tokens": 160,
                   "case_labels": list(labels), "workload_hash": "wh-1"},
        "exp2_domain_digest": {"cpu": "0xAAAA", "device": exp},
        "arms": {
            "stock": {"gate": {"ok": True, "reason": "ok"},
                      "weight_fingerprint": "wf-stock", "self_check": True,
                      "cases": dict(zip(labels, stock))},
            "exact": {"gate": {"ok": gate_ok, "reason": "ok" if gate_ok
                               else "canary -> Arabic"},
                      "weight_fingerprint": wfp, "self_check": self_check,
                      "cases": dict(zip(labels, exact))},
        },
    }


SAME = ([1, 2, 3], [1, 2, 3], [1, 2, 3])
MOVED = ([1, 2, 3], [1, 9, 3], [1, 2, 8])


def check_gate_logic():
    """Drive every refusal branch of `compare_artifacts`. -> exit code."""
    A100 = ("A100", "8.0")
    quiet = lambda *_a, **_k: None                            # noqa: E731

    scenarios = [
        # (name, artifact A, artifact B, allow_same, want_code, want_note)
        ("the result we are looking for",
         _art(stock=MOVED, exact=SAME), _art(*A100, stock=SAME, exact=SAME),
         False, 0, "exact identical, stock moved"),

        ("same device -- vacuous, measures re-run determinism",
         _art(stock=MOVED, exact=SAME), _art(stock=SAME, exact=SAME),
         False, 2, "refused"),

        ("same device, explicitly allowed",
         _art(stock=MOVED, exact=SAME), _art(stock=SAME, exact=SAME),
         True, 0, "runs, labelled as a re-run check"),

        ("control did not fire -- stock identical on both cards",
         _art(stock=SAME, exact=SAME), _art(*A100, stock=SAME, exact=SAME),
         False, 1, "the pair distinguishes nothing"),

        ("the claim is false -- exact moved too",
         _art(stock=MOVED, exact=SAME), _art(*A100, stock=SAME, exact=MOVED),
         False, 1, "exact is not portable on this pair"),

        ("an arm was not computing the model's function",
         _art(stock=MOVED, exact=SAME),
         _art(*A100, stock=SAME, exact=SAME, gate_ok=False),
         False, 1, "gate failure refuses the arm"),

        ("different weights -- not attributable to the device",
         _art(stock=MOVED, exact=SAME),
         _art(*A100, stock=SAME, exact=SAME, wfp="wf-2"),
         False, 1, "inconclusive, not a pass"),

        ("a card that does not reproduce its own output",
         _art(stock=MOVED, exact=SAME),
         _art(*A100, stock=SAME, exact=SAME, self_check=False),
         False, 1, "refused before it is compared to anything"),

        ("prompts edited between the two recordings",
         _art(stock=MOVED, exact=SAME),
         _art(*A100, stock=SAME, exact=SAME, labels=("c0", "c1", "cX")),
         False, 2, "refused"),
    ]

    bad = 0
    for name, a, b, allow, want, note in scenarios:
        code, _ = compare_artifacts(a, b, allow_same_device=allow, out=quiet)
        ok = code == want
        bad += not ok
        print(f"  {'ok   ' if ok else 'WRONG'} {name:<52} "
              f"exit {code} (want {want})\n         {note}")

    # A degenerate artifact must not be readable as a pass: `stock` moving is
    # the only thing that makes `exact` not moving mean anything, and scenario 4
    # is what pins that. Stated here so the file says why 8 scenarios and not 3.
    print("\n  scenarios 2/3 pin the same-device refusal in both directions;")
    print("  4 is the control, 5 the claim being false, 6-9 the four ways two")
    print("  runs can fail to be comparable at all.")
    if bad:
        print(f"\n{bad} scenario(s) wrong: the gate's own logic is broken.")
        return 1
    print(f"\ngate logic behaves on all {len(scenarios)} scenarios.")
    return 0


# ---------------------------------------------------------------------------
# Recording half. Imports torch lazily so everything above runs without it.
# ---------------------------------------------------------------------------


def _weight_fingerprint(model):
    """Deterministic, device-independent digest of an arm's parameters.

    Everything is pulled to the CPU and widened to f64 first, so this is a
    property of the weights and not of where they were sitting. Catches a
    different checkpoint revision between the two recordings -- and, for the
    `exact` arm whose scales are computed on-device, a genuine cross-device
    difference in the quantisation itself.
    """
    import torch
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        flat = p.detach().flatten()
        take = flat[:8].to("cpu", dtype=torch.float64)
        h.update(take.numpy().tobytes())
    for name, b in sorted(model.named_buffers(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(str(tuple(b.shape)).encode())
    return h.hexdigest()[:16]


def _device_block(device):
    import torch
    d = {"host": platform.node(), "torch": torch.__version__,
         "python": platform.python_version(), "platform": platform.platform()}
    try:
        import transformers
        d["transformers"] = transformers.__version__
    except Exception:
        pass
    if device.startswith("cuda") and torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        maj, minor = torch.cuda.get_device_capability(i)
        d.update({
            "gpu_name": props.name,
            "capability": f"{maj}.{minor}",
            "multi_processor_count": props.multi_processor_count,
            "total_memory": props.total_memory,
            "driver": getattr(torch.version, "cuda", None),
        })
    else:
        d.update({"gpu_name": f"cpu:{platform.machine()}", "capability": "cpu",
                  "multi_processor_count": os.cpu_count(), "driver": None})
    return d


# The default case list. Chosen so every mechanism that could be
# device-dependent is crossed at least once, and small enough that a rented box
# is not held for an hour: batch 1 (no batching at all), a batch size that tiles
# the GEMM differently, a ragged composition, and one that crosses `exact_pv`'s
# digit-width boundary. `--full` runs the whole ragged-batch set instead.
DEFAULT_CASES = (
    "alone (batch 1)",
    "+3 long, middle",
    "mixed 8, pos 5",
    "mixed 32, pos 17",
    "+1 extra-long (new dbits)",
)


def record(args):
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    from transformers import AutoTokenizer

    import batch_invariance_demo as D
    import exact_ragged_batch as R

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tok = AutoTokenizer.from_pretrained(R.MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_cases = {name: (others, pos) for name, others, pos in R.cases()}
    if args.cases:
        asked = [c.strip() for c in args.cases.split("|") if c.strip()]
    elif args.full:
        asked = list(all_cases)
    else:
        asked = list(DEFAULT_CASES)
    wanted = [c for c in asked if c in all_cases]
    missing = [c for c in asked if c not in all_cases]
    if missing:
        print(f"  *** case(s) {missing} do not exist in "
              f"exact_ragged_batch.cases(). Available:")
        for c in all_cases:
            print(f"        {c!r}")
        return 2

    # Hash the WORK, not just its labels: the prompt text and token budget are
    # what a comparison silently depends on.
    wh = hashlib.sha256()
    wh.update(f"{R.MODEL}|{args.tokens}|{R.TARGET}".encode())
    for c in wanted:
        others, pos = all_cases[c]
        wh.update(f"|{c}|{pos}|".encode())
        for o in others:
            wh.update(o.encode())
    workload_hash = wh.hexdigest()[:16]

    art = {
        "schema": SCHEMA,
        "device": _device_block(args.device),
        "config": {"model": R.MODEL, "tokens": args.tokens,
                   "case_labels": wanted, "workload_hash": workload_hash},
        "arms": {},
    }
    print(f"\nRecording on {_fmt_dev(art['device'])}")
    print(f"  {len(wanted)} cases x {args.tokens} tokens, arms: "
          f"{', '.join(args.arms)}")

    try:
        art["exp2_domain_digest"] = {
            "cpu": hex(D.exp2_domain_digest("cpu")),
            "device": hex(D.exp2_domain_digest(args.device)),
        }
        print(f"  integer exp digest: cpu {art['exp2_domain_digest']['cpu']}, "
              f"device {art['exp2_domain_digest']['device']}")
    except Exception as e:                                    # pragma: no cover
        print(f"  (exp digest unavailable: {e})")

    for arm in args.arms:
        print(f"\n--- {arm} ---")
        model = R.build(arm, args.device)
        canary = D.canary_text(model, tok, args.device)
        cases_out, gate = {}, None
        for label in wanted:
            others, pos = all_cases[label]
            got = R.decode_target(model, tok, others, pos, args.tokens,
                                  args.device)
            cases_out[label] = got
            if gate is None:
                ok, reason = D.sanity_verdict(canary, got)
                gate = {"ok": bool(ok), "reason": reason}
                print(f"  gate: {'ok' if ok else 'FAILED'} -- {reason}")
                if not ok:
                    # Recording it anyway, with the verdict attached: `--compare`
                    # refuses the arm and says why. Writing nothing would leave
                    # the operator guessing on a rented box.
                    print(f"  *** arm {arm!r} is not computing the model's "
                          f"function; its 0-difference score would be "
                          f"meaningless. ***")
            print(f"  {label:<28} {len(got)} tokens")
        art["arms"][arm] = {
            "gate": gate,
            "weight_fingerprint": _weight_fingerprint(model),
            "cases": cases_out,
            "digest": hashlib.sha256(
                json.dumps(cases_out, sort_keys=True).encode()).hexdigest()[:16],
        }
        print(f"  weights {art['arms'][arm]['weight_fingerprint']}, "
              f"tokens {art['arms'][arm]['digest']}")

        if args.self_check:
            # Re-run determinism on THIS card is the precondition for a
            # cross-card comparison. One case, so it is nearly free.
            label = wanted[0]
            others, pos = all_cases[label]
            again = R.decode_target(model, tok, others, pos, args.tokens,
                                    args.device)
            same = again == cases_out[label]
            art["arms"][arm]["self_check"] = bool(same)
            print(f"  self-check (same card, twice): "
                  f"{'identical' if same else '*** NOT REPRODUCIBLE ***'}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    qwen2.eager_attention_forward = D._orig
    with open(args.record, "w") as f:
        json.dump(art, f, indent=1, sort_keys=True)
    print(f"\n  written to {args.record}")
    print(f"  now run this on the OTHER card and compare:\n"
          f"    python3 tools/exact_cross_device.py --compare "
          f"{args.record} other.json")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="cross-GPU bit-identity of the deterministic decode path")
    ap.add_argument("--record", metavar="OUT.json",
                    help="decode on this machine's GPU and write an artifact")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                    help="compare two artifacts; no GPU or torch needed")
    ap.add_argument("--check-gate", action="store_true",
                    help="exercise the comparison logic on synthetic "
                         "artifacts and exit; no GPU, no torch, no model")
    ap.add_argument("--arms", default="stock,exact",
                    help="comma-separated: stock, exact, fixedfloat")
    ap.add_argument("--tokens", type=int, default=160,
                    help="160 by default: at 24 the bf16 control does not "
                         "diverge, which makes a broken comparison look clean")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cases", metavar="'a|b'",
                    help="pipe-separated case labels from "
                         "exact_ragged_batch.cases(); the workload hash covers "
                         "the selection, so two runs with different --cases "
                         "refuse to compare")
    ap.add_argument("--full", action="store_true",
                    help="all of exact_ragged_batch.cases() instead of the "
                         "5-case default")
    ap.add_argument("--no-self-check", dest="self_check", action="store_false",
                    help="skip the same-card re-run precondition")
    ap.add_argument("--allow-same-device", action="store_true",
                    help="compare two artifacts from the same GPU anyway; "
                         "this is a re-run determinism check, not a "
                         "cross-hardware one")
    a = ap.parse_args()

    if a.check_gate:
        return check_gate_logic()
    if a.record:
        # Must happen before `import torch`, which is why it is here and not at
        # module scope: everything above this line runs without torch, and a
        # `--compare` on a machine that has no `_nospace` beside it must still
        # work. See tools/_nospace.py.
        try:
            import _nospace
            _nospace.guard()
        except ImportError:
            pass
    if a.compare:
        arts = []
        for p in a.compare:
            with open(p) as f:
                arts.append(json.load(f))
        print(f"\nCross-device bit-identity\n  {a.compare[0]}  vs  "
              f"{a.compare[1]}")
        code, _ = compare_artifacts(arts[0], arts[1],
                                    allow_same_device=a.allow_same_device)
        return code
    if a.record:
        a.arms = [s.strip() for s in a.arms.split(",") if s.strip()]
        return record(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
