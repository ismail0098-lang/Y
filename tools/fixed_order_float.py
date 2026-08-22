"""The missing control: is float batch-invariant if you pin the reduction order?

`docs/bit_identical_decode.md` has said since the first draft that this arm is
the one that could reframe the whole result. The project's claim is that
determinism needs *exact* arithmetic, because float addition is
non-associative and every reordering gives a different answer. The obvious
objection is that you do not need exactness for that -- you need a fixed
ORDER, and a float kernel with a pinned reduction order would be deterministic
without quantising anything.

If that is cheap, the quantisation is unnecessary and the pitch is wrong. So it
is worth finding out, and it is worth finding out with the SAME 16 batch
compositions the exact arm faces, which is why this is an arm inside
`exact_ragged_batch.py` rather than a harness of its own.

## What "pinned" has to mean

Decode attention has two reductions, and batch shape reaches both:

  * `q . k_t` reduces over `head_dim`. That extent does not change with the
    batch -- but `M = b * n_heads` does, so cuBLAS may pick a different kernel
    with a different internal order for the same `K`.
  * `sum_t p_t * v_t` reduces over the KEY axis, whose extent changes with how
    far the batch was padded.

Both are pinned here by doing the reduction in Python: fixed-size chunks,
accumulated one at a time in a fixed sequence, so the only library reduction
left is inside a chunk of constant shape.

**The key-axis chunks are anchored at the END, not the start.** Prompts are
LEFT-padded, so a chunking aligned to position 0 puts a row's real keys in
different groups depending on how much padding the batch happened to need --
which is exactly the dependency being removed. Anchored at the right, every
row's real keys fall in the same groups whatever the padding, and the leading
chunks contain only masked positions, which contribute exactly 0.0 and so
change nothing.

## The knobs are the measurement

`Y_FOF_TCHUNK` / `Y_FOF_DCHUNK` set the two chunk sizes. Smaller is more
tightly pinned and slower. The result of this experiment is not "float can be
made invariant" -- of course it can, at chunk size 1 -- it is **what chunk size
it takes and what that costs**, measured against the exact arm on the same
compositions.
"""
import math
import os

import torch
import torch.nn as nn

TCHUNK = int(os.environ.get("Y_FOF_TCHUNK", "64"))
DCHUNK = int(os.environ.get("Y_FOF_DCHUNK", "16"))


def fixed_order_attention(module, query, key, value, attention_mask,
                          scaling=None, dropout=0.0, **kw):
    """float32 attention whose reduction order depends on nothing but head_dim
    and the key length -- never on batch size, padding or position."""
    b, nh, q_len, d = query.shape
    nkv = key.shape[1]
    rep = nh // nkv
    sc = scaling if scaling is not None else 1.0 / math.sqrt(d)

    q = query.float()
    k = key.float().repeat_interleave(rep, dim=1)
    v = value.float().repeat_interleave(rep, dim=1)
    t = k.shape[-2]

    # --- scores: reduce over head_dim in fixed chunks, in a fixed sequence ---
    scores = torch.zeros(b, nh, q_len, t, device=q.device, dtype=torch.float32)
    for d0 in range(0, d, DCHUNK):
        d1 = min(d0 + DCHUNK, d)
        scores = scores + torch.matmul(q[..., d0:d1], k[..., d0:d1].transpose(-1, -2))
    scores = scores * sc

    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :t].float()

    p = torch.softmax(scores, dim=-1)
    if attention_mask is not None:
        p = torch.where(attention_mask[:, :, :, :t].float() < -1.0,
                        torch.zeros_like(p), p)

    # --- p @ V: reduce over the key axis in fixed chunks anchored at the END ---
    acc = torch.zeros(b, nh, q_len, d, device=q.device, dtype=torch.float32)
    hi = t
    while hi > 0:
        lo = max(hi - TCHUNK, 0)
        acc = acc + torch.matmul(p[..., lo:hi], v[..., lo:hi, :])
        hi = lo
    # [b, nh, q, d] -> [b, q, nh, d], as `eager_attention_forward` returns.
    # Getting this wrong does not crash: the model runs, emits fluent-looking
    # garbage, and the invariance harness happily reports a number for it. The
    # first run of this arm read 12/16 that way.
    return acc.to(query.dtype).transpose(1, 2).contiguous(), None


class FixedOrderLinear(nn.Module):
    """`nn.Linear` whose reduction over `in_features` is pinned the same way.

    **Pinning attention alone is not the experiment, and the first run of this
    control got that wrong.** The exact arm replaces the linear layers too, so a
    control that leaves them as ordinary cuBLAS matmuls is not doing the same
    work: `M = batch * seq` changes with the batch, cuBLAS picks a different
    kernel, and the reduction over `in_features` reorders. The measured result
    was 12/16 compositions changed -- which says nothing about fixed-order
    FLOAT, only that the layers nobody pinned are still batch-dependent.
    """

    def __init__(self, lin, chunk):
        super().__init__()
        self.register_buffer("w", lin.weight.data.float().t().contiguous())
        self.register_buffer(
            "b", None if lin.bias is None else lin.bias.data.float())
        self.chunk = chunk

    def forward(self, x):
        x = x.float()
        k = self.w.shape[0]
        out = None
        for k0 in range(0, k, self.chunk):
            k1 = min(k0 + self.chunk, k)
            part = torch.matmul(x[..., k0:k1], self.w[k0:k1])
            out = part if out is None else out + part
        if self.b is not None:
            out = out + self.b
        return out


def convert(model, chunk=None):
    """Replace every `nn.Linear` outside the embedding/head with a pinned one."""
    chunk = DCHUNK * 8 if chunk is None else chunk
    n = 0
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, name, FixedOrderLinear(child, chunk).to(child.weight.device))
                n += 1
    return n
