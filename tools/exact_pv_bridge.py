"""Y's `exact_pv` kernel against the prototype's digit split, on real shapes.

    python3 tools/exact_pv_bridge.py

`batch_invariance_demo.exact_pv` computes `p @ v` as `ceil(29/dbits)` separate
fp32 matmuls, because a Q0.28 weight times an int8 activation is 35 bits and an
fp32 mantissa holds 24. The split is a workaround for the ACCUMULATOR. The
kernel in `tests/exact_pv.ysu` accumulates in int64, so there is nothing to
split, and the launch census says that is worth 72 of a decode step's 748
kernel launches.

This tool is the honest comparison the census cannot make on its own:

  * **bit-identical output**, not "close". Both routes compute the same integer;
    if they disagree by one ULP the deterministic claim is gone.
  * **launch count**, which is the thing being optimised and is invariant under
    contention.
  * **time**, reported as a MINIMUM over interleaved rounds. The two arms are
    timed round-robin rather than one after the other, because on this machine
    an unrelated process taking the GPU between the halves of a run turns drift
    into a difference between the things being compared.

**Both arms must start from what the model actually holds, and the first
version of this tool got that wrong in both directions.** `quantize_rows`
returns `vi` as **float32** carrying integer values in [-127, 127], so the
digit split consumes it with no conversion at all -- while the first version
built `v` as int64 and converted inside the timed digit arm, charging it for
work the model never does. The Y kernel wants real int8, so it owes a
float32 -> int8 cast that the digit path does not. Both are now priced: `kernel`
is the launch alone and `+cast` includes the conversion, reported separately
the way the MSM numbers separate cold from fixed-base from kernel.

That `vi` is float32 at all is worth noticing on its own -- it is a tensor of
small integers stored at 4 bytes each, so the KV cache moves 4x the bytes it
needs to. Feeding this kernel a genuinely int8 cache would delete the cast AND
three quarters of the read traffic, but that is a change to the cache, not to
this comparison.

Shapes come from the model the rest of this directory measures: Qwen2.5-0.5B at
batch 32 is 14 KV heads x 32 sequences of head_dim 64.
"""
import _nospace

_nospace.guard()

import argparse  # noqa: E402
import ctypes  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

import batch_invariance_demo as D  # noqa: E402
import ptx_bridge as PB  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
YBIN = str(REPO / "target" / "release" / "Y")


def emit_pv_ptx():
    """Compile tests/exact_pv.ysu and read back the .ptx it writes."""
    src = REPO / "tests" / "exact_pv.ysu"
    subprocess.run([YBIN, str(src), "--emit-ptx"], capture_output=True,
                   check=True, cwd=REPO)
    return (REPO / "tests" / "exact_pv.ptx").read_text()


def kernel_pv(p_u32, v_i8, B, T, Dh, fn):
    """One launch. `p_u32` is [B, T], `v_i8` is [B, T, Dh]; returns [B, Dh] i64."""
    out = torch.empty(B * Dh, dtype=torch.int64, device=p_u32.device)
    PB.launch(fn, (B, 1, 1), (Dh, 1, 1),
              [PB.dptr(p_u32), PB.dptr(v_i8), PB.dptr(out),
               ctypes.c_uint(T), ctypes.c_uint(Dh),
               ctypes.c_uint(B * T), ctypes.c_uint(B * T * Dh),
               ctypes.c_uint(B * Dh)])
    return out.view(B, Dh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=14)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--keys", type=int, nargs="+", default=[67, 512, 1024, 4096])
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device.")
        return 0
    dev = "cuda"
    B = a.batch * a.kv_heads
    Dh = a.head_dim
    # `cuModuleLoadData` needs a current context, and torch creates one
    # lazily on first tensor use rather than in `cuda.init()`.
    torch.zeros(1, device=dev)
    mod = PB.Module(emit_pv_ptx())
    fn = mod.fn("exact_pv")

    print(f"\nexact_pv: Y kernel (one launch, int64) vs the digit split "
          f"(ceil(29/dbits) fp32 matmuls)")
    print(f"  B = {a.batch} x {a.kv_heads} heads = {B} rows, head_dim {Dh}, "
          f"best of {a.reps} interleaved\n")
    print(f"{'T':>6}{'dbits':>7}{'matmuls':>9}{'digit ms':>11}{'Y ms':>9}"
          f"{'+cast ms':>10}{'kernel':>9}{'w/cast':>9}   identical")

    g = torch.Generator(device=dev).manual_seed(20260822)
    all_ok = True
    for T in a.keys:
        p = torch.randint(0, (1 << D.P_BITS) + 1, (B, T), generator=g,
                          dtype=torch.int64, device=dev)
        p[:, 0] = 1 << D.P_BITS          # the top bit, explicitly
        p[:, 1] = 0
        v = torch.randint(-127, 128, (B, T, Dh), generator=g,
                          dtype=torch.int64, device=dev)

        # What the model holds: float32 carrying integers, straight out of
        # `quantize_rows`. This is the common input both arms start from.
        v_f32 = v.to(torch.float32).contiguous()
        p32 = p.to(torch.int32).contiguous()
        v8 = v_f32.to(torch.int8).contiguous()

        # Correctness first, before anything is timed.
        want = D.exact_pv(p.unsqueeze(1), v_f32).squeeze(1)
        got = kernel_pv(p32, v8, B, T, Dh, fn).to(torch.float64)
        torch.cuda.synchronize()
        identical = bool(torch.equal(got, want))
        all_ok &= identical

        dbits = D.digit_width(T)
        matmuls = (D.P_BITS + 1 + dbits - 1) // dbits if dbits else 0

        p1 = p.unsqueeze(1)

        def t_digit():
            D.exact_pv(p1, v_f32)         # no cast: this is what the model has

        def t_y():
            kernel_pv(p32, v8, B, T, Dh, fn)

        def t_y_cast():
            kernel_pv(p32, v_f32.to(torch.int8), B, T, Dh, fn)

        arms = (("digit", t_digit), ("y", t_y), ("ycast", t_y_cast))
        for _, f in arms:                 # warm up all three
            for _ in range(3):
                f()
        torch.cuda.synchronize()
        best = {k: 1e9 for k, _ in arms}
        for _ in range(a.reps):           # ROUND-ROBIN, not one then the other
            for name, f in arms:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                f()
                torch.cuda.synchronize()
                best[name] = min(best[name], time.perf_counter() - t0)

        print(f"{T:>6}{dbits:>7}{matmuls:>9}{best['digit'] * 1e3:>11.3f}"
              f"{best['y'] * 1e3:>9.3f}{best['ycast'] * 1e3:>10.3f}"
              f"{best['digit'] / best['y']:>8.2f}x"
              f"{best['digit'] / best['ycast']:>8.2f}x"
              f"   {'yes' if identical else 'NO'}")

    print()
    if not all_ok:
        print("  *** the two routes DISAGREE. They compute the same integer, so "
              "any difference\n      is a bug, not a tolerance. ***")
        return 1
    print("  bit-identical at every key length, and one launch instead of "
          f"{matmuls}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
