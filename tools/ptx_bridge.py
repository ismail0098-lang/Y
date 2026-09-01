"""Run the Y-compiled attention kernel on a real model's tensors.

`tools/batch_invariance_demo.py` shows the determinism property end to end, but
it computes attention in torch. That leaves an honest gap: the PTX kernel that
was measured at ~3% overhead and 85% of DRAM peak is a *different* piece of
code from the one in the demo. This closes it.

The PTX comes from the compiler itself:

    Y --emit-attention-ptx <head_dim> <seq_len>

which is the same string `tests/gpu_attention_invariance.rs` tests, loaded here
through the CUDA driver API into torch's own context and launched on torch
tensors. The check is **bit-identical output against the torch path**, on real
post-RoPE Q/K/V captured from Qwen2.5-0.5B — not on synthetic data.

Run:
    python3 tools/ptx_bridge.py
"""
import ctypes
import math
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
YBIN = os.path.join(REPO, "target", "release", "Y")

# ------------------------------------------------------------------ driver
cuda = ctypes.CDLL("libcuda.so.1")


def _chk(err, what):
    if err != 0:
        buf = ctypes.c_char_p()
        cuda.cuGetErrorString(err, ctypes.byref(buf))
        raise RuntimeError(f"{what} failed: {err} {buf.value!r}")


class Module:
    def __init__(self, ptx: str):
        # torch has already created and bound a primary context; reuse it.
        ctx = ctypes.c_void_p()
        _chk(cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if not ctx.value:
            raise RuntimeError("no current CUDA context - touch a cuda tensor first")
        self.mod = ctypes.c_void_p()
        _chk(cuda.cuModuleLoadData(ctypes.byref(self.mod), ptx.encode()),
             "cuModuleLoadData")

    def fn(self, name):
        f = ctypes.c_void_p()
        _chk(cuda.cuModuleGetFunction(ctypes.byref(f), self.mod, name.encode()),
             "cuModuleGetFunction")
        return f


def launch(fn, grid, block, args):
    """`args` is a list of (ctypes value) already sized for the parameter."""
    holders = [ctypes.byref(a) for a in args]
    arr = (ctypes.c_void_p * len(holders))(
        *[ctypes.cast(h, ctypes.c_void_p) for h in holders]
    )
    _chk(cuda.cuLaunchKernel(fn, grid[0], grid[1], grid[2],
                             block[0], block[1], block[2],
                             0, None, arr, None), "cuLaunchKernel")
    _chk(cuda.cuCtxSynchronize(), "cuCtxSynchronize")


def dptr(t):
    return ctypes.c_ulonglong(t.data_ptr())


# ------------------------------------------------------------------ the kernel
def emit_ptx(head_dim, seq_len):
    out = subprocess.run([YBIN, "--emit-attention-ptx", str(head_dim), str(seq_len)],
                         capture_output=True, check=True)
    return out.stdout.decode()


def kernel_attention(q8, k8, v8, kfix, seq_len, head_dim, mod):
    """One query row over `seq_len` keys, entirely on the device."""
    dev = q8.device
    B = 1
    scores = torch.zeros(B * seq_len, dtype=torch.int32, device=dev)
    m = torch.full((B,), -2147483647, dtype=torch.int32, device=dev)
    l = torch.zeros(B, dtype=torch.int64, device=dev)
    o = torch.zeros(B * head_dim, dtype=torch.int64, device=dev)
    p = torch.zeros(B * seq_len, dtype=torch.int32, device=dev)

    launch(mod.fn("attn_scores"),
           ((seq_len + 127) // 128, B, 1), (128, 1, 1),
           [dptr(q8), dptr(k8), dptr(scores), dptr(m)])
    launch(mod.fn("attn_accum"),
           (4, B, 3), (64, 1, 1),
           [dptr(scores), dptr(v8), dptr(m), dptr(l), dptr(o), dptr(p),
            ctypes.c_uint(kfix)])
    return o, l, p


def torch_attention(q8, k8, v8, kfix, seq_len, head_dim):
    """The same arithmetic in torch, as the demo does it."""
    from batch_invariance_demo import exp2_neg_q16_16, _table
    k = k8.view(seq_len, head_dim).to(torch.float64)
    s = (q8.view(1, head_dim).to(torch.float64) * k).sum(dim=1)
    m = s.max()
    t = torch.clamp(torch.div((m - s) * kfix + 32768, 65536,
                              rounding_mode="floor"), 0, 1 << 30)
    p = exp2_neg_q16_16(t, _table(q8.device))
    l = p.sum()
    o = (p.unsqueeze(1) * v8.view(seq_len, head_dim).to(torch.float64)).sum(dim=0)
    return o, l, p


def main():
    sys.path.insert(0, HERE)
    if not os.path.exists(YBIN):
        print(f"build the compiler first: cargo build --release ({YBIN} missing)")
        return 1
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.models.qwen2.modeling_qwen2 as qwen2

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    captured = []
    orig = qwen2.eager_attention_forward

    def spy(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        captured.append((query.detach(), key.detach(), value.detach(), scaling))
        return orig(module, query, key, value, attention_mask, scaling=scaling,
                    dropout=dropout, **kw)

    qwen2.eager_attention_forward = spy
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    ).cuda().eval()
    text = ("Explain why floating point addition is not associative and what "
            "that means for reproducibility on a GPU. ") * 6
    with torch.no_grad():
        model(**tok(text, return_tensors="pt").to("cuda"))
    qwen2.eager_attention_forward = orig

    print(f"captured {len(captured)} attention calls from a real forward pass")
    checked = mismatched = 0
    skipped = []  # temperatures the kernel cannot represent; see below
    worst_spread = 0.0

    for li in (0, 8, 16, 23):
        qs, ks, vs, scaling = captured[li]
        _, nh, T, d = qs.shape
        nkv = ks.shape[1]
        ptx = emit_ptx(d, T)
        mod = Module(ptx)
        sc = scaling if scaling is not None else 1.0 / math.sqrt(d)

        for h in (0, nh // 2, nh - 1):
            kv = h // (nh // nkv)
            q = qs[0, h, T - 1].double()
            k = ks[0, kv].double()
            v = vs[0, kv].double()

            def q8(x, dims):
                s = x.abs().amax(dim=dims, keepdim=True) / 127.0
                s = torch.where(s == 0, torch.ones_like(s), s)
                return torch.round(x / s).clamp(-127, 127), s

            qi, sq = q8(q, (-1,))
            ki, sk = q8(k, (-2, -1))
            vi, sv = q8(v, (-2, -1))
            # The kernel's `KFix` and the demo's per-key scale are NOT the same
            # number, and this line used to compute the demo's.
            #
            # `C = sq * sk * sc * log2(e)` converts one unit of the integer
            # score into log2 units. The exp wants its argument in Q16.16, so
            # the demo (`batch_invariance_demo.py`) forms the Q16.16 logit
            # directly and multiplies by `C * 2^16`. The KERNEL instead takes a
            # multiplier and shifts: `t = ((m - s) * KFix + 2^15) >> 16`, so it
            # needs `KFix = C * 2^32`.
            #
            # Passing `C * 2^16` -- which is what this did -- makes every
            # exponent 65536x too small. On these activations that puts every
            # `n = t >> 16` below 0.014, i.e. every weight within 1% of 2^28:
            # a UNIFORM softmax. Both arms below replicate the kernel's
            # formula, so they still agreed bit for bit, and the check passed
            # 12/12 while validating a computation with no temperature in it.
            # That is why `spread` is tracked and asserted at the end.
            #
            # The two bounds below are the compiler's, not this script's:
            # `exact_attention::temperature_fixed_point` refuses the same two
            # multipliers and says why, and `SoftmaxErrorBound.v` proves what
            # each one does to the answer (`a_zero_multiplier_gives_every_key_
            # the_same_weight`, `the_two_readings_of_the_multiplier_disagree_
            # above_two_to_the_thirty_one`). They lived HERE, as a bare
            # `continue`, until 2026-09-01 -- so a temperature outside the
            # representable range silently removed a case from a measurement
            # instead of failing it. Skips are counted and reported now.
            c = float(sq) * float(sk) * sc * math.log2(math.e)
            kfix = int(round(c * 2.0 ** 32))
            if kfix <= 0 or kfix >= 2 ** 31:
                skipped.append((c, kfix))
                continue

            qb = qi.to(torch.int8).contiguous()
            kb = ki.to(torch.int8).contiguous()
            vb = vi.to(torch.int8).contiguous()

            o_k, l_k, p_k = kernel_attention(qb, kb, vb, kfix, T, d, mod)
            o_t, l_t, p_t = torch_attention(qb, kb, vb, kfix, T, d)

            same = (
                torch.equal(p_k.to(torch.int64), p_t.to(torch.int64))
                and int(l_k[0]) == int(l_t)
                and torch.equal(o_k, o_t.to(torch.int64))
            )
            checked += 1
            # The control. Both arms replicate the kernel's own formula, so
            # agreement alone says nothing about whether the SOFTMAX is right --
            # a temperature of zero makes every weight 2^28 and both arms agree
            # perfectly on that. `max(p)/mean(p)` is exactly 1.0 for a uniform
            # weight vector and grows with peaking, so it is the cheapest thing
            # that separates "the kernel implements this arithmetic" from "the
            # arithmetic is attention".
            worst_spread = max(
                worst_spread,
                float(p_k.max()) / max(float(p_k.double().mean()), 1.0),
            )
            if not same:
                mismatched += 1
                print(f"  MISMATCH layer {li} head {h}: "
                      f"l {int(l_k[0])} vs {int(l_t)}")

    print(f"\nlayer/head pairs checked : {checked}")
    print(f"bit-identical            : {checked - mismatched}")
    print(f"mismatched               : {mismatched}")
    print(f"worst max(p)/mean(p)     : {worst_spread:.1f}  (1.0 = uniform)")
    print(f"skipped (unrepresentable): {len(skipped)}")
    if skipped:
        # Loud, because a silent skip of an unrepresentable temperature is a
        # measurement quietly shrinking rather than failing.
        for c, kf in skipped[:5]:
            print(f"    C = {c:e} -> KFix = {kf}, outside (0, 2^31)")
    if checked > 0 and worst_spread < 2.0:
        print("\nFAIL: every softmax weight came back within a factor of two of the")
        print("mean, i.e. the weights are uniform and the temperature is doing")
        print("nothing. Both arms share the kernel's formula, so they would agree")
        print("bit for bit on that too -- this comparison would be vacuous. Check")
        print("that KFix is C * 2^32 and not the demo's C * 2^16.")
        return 1
    if mismatched == 0 and checked > 0:
        print("\nThe Y-compiled PTX kernel and the torch path agree BIT FOR BIT on")
        print("real post-RoPE activations. The demo's numbers are the kernel's.")
    return 1 if mismatched or checked == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
