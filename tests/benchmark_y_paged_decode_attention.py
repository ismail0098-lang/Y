#!/usr/bin/env python3
"""
Correctness + performance harness for the REAL Y-compiler-emitted paged
decode-attention kernel (`ptx_emitter::emit_paged_decode_attention_kernel`,
dispatched from `tests/paged_decode_attention_*.ysu`).

Every number below comes from `target/release/Y <file>.ysu --emit-ptx`'s
actual output, loaded as-is via `cp.RawModule(path=...)` - not from a
hand-written reference kernel.

Two things are measured, and they answer different questions:

  * CORRECTNESS against a PyTorch float32 reference that walks the page table
    itself. The page table is always SHUFFLED - an identity mapping would
    hide every paging address bug - and sequence lengths are ragged,
    including partial final pages, lengths below the warp count (so some
    warps get no tokens at all) and zero.

  * PERFORMANCE against FlashInfer's `BatchDecodeWithPagedKVCacheWrapper`,
    which is a production paged decode-attention kernel solving exactly this
    problem with exactly this KV layout (`NHD`, k/v as separate
    `[num_pages, page_size, num_kv_heads, head_dim]` tensors). This is an
    apples-to-apples comparison, not a comparison against eager PyTorch.

Timing discipline (see benchmark_y_decode_gemm.py for the full reasoning):
this GPU's SM clock idles at ~210 MHz and needs ~3s of sustained load to
reach ~2670 MHz, and clocks cannot be locked here. So every measurement
ramps first, then A/B-INTERLEAVES Y and FlashInfer so both see the same
clock, and reports the MINIMUM over interleaved rounds - the contamination
is one-sided (interference only ever makes a round slower), so the fastest
round is the best estimate and the median mostly measures how disturbed the
machine was.

Usage:
    python3 tests/benchmark_y_paged_decode_attention.py            # correctness + perf
    python3 tests/benchmark_y_paged_decode_attention.py --correctness-only
"""
import argparse
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import cupy as cp

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
Y_BIN = os.path.join(REPO_ROOT, "target/release/Y")
DEV = torch.device("cuda:0")

RAMP_SECONDS = 3.0
INTERLEAVE_ROUNDS = 7
NUM_WARPS = 32  # informational; the real value is read from the emitted PTX comment

# Correctness gate: relative L2, the aggregate metric this codebase uses for
# GEMM-shaped comparisons (a per-element rtol/atol gate produces false
# failures purely from order statistics once millions of elements are
# compared). f16 storage of Q/K/V puts the floor around 2e-4.
TOL_REL_L2 = 3e-3


def import_flashinfer():
    """Imports FlashInfer, working around its JIT breaking on paths containing
    a space.

    FlashInfer builds its kernels at first use by generating a ninja file, and
    it does not quote the `-I`/`-isystem` include paths it writes there. This
    checkout lives under `/home/yumin/NVME files/...`; nvcc splits that on the
    space and dies with "A single input file is required for a non-link phase
    when an outputfile is specified". The include roots come from
    `pathlib.Path(__file__).resolve().parents[1]` in `flashinfer/jit/env.py`,
    and `.resolve()` follows symlinks, so pointing `sys.path` at a symlink
    does not help - the package files themselves have to sit somewhere without
    a space.

    So: copy `flashinfer` and `tvm_ffi` (the two packages that contribute
    include paths) into a space-free cache directory once, and import from
    there. This is a defect in FlashInfer's build-file generation, not in Y,
    and it only affects the BASELINE side of this comparison - Y's own kernel
    is unaffected either way.
    """
    try:
        import flashinfer  # noqa: F401
        if " " not in os.path.dirname(flashinfer.__file__):
            return flashinfer
    except ImportError:
        return None

    import shutil
    shim = os.path.join(os.environ.get("TMPDIR", "/tmp"), "y_flashinfer_shim")
    site = os.path.dirname(os.path.dirname(flashinfer.__file__))
    os.makedirs(shim, exist_ok=True)
    for pkg in ("flashinfer", "tvm_ffi"):
        src, dst = os.path.join(site, pkg), os.path.join(shim, pkg)
        if os.path.isdir(src) and not os.path.isdir(dst):
            print("[*] copying %s to a space-free path (%s) - see import_flashinfer()" % (pkg, shim))
            shutil.copytree(src, dst)
    for mod in [m for m in list(sys.modules) if m.split(".")[0] in ("flashinfer", "tvm_ffi")]:
        del sys.modules[mod]
    sys.path.insert(0, shim)
    import flashinfer  # noqa: F811
    return flashinfer


def sm_clock_mhz():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return 0


def ysu_path(hd, nqh, nkvh, ps, warps=None):
    stem = "paged_decode_attention_%d_%d_%d_%d" % (hd, nqh, nkvh, ps)
    if warps is not None:
        stem += "_%d" % warps
    return os.path.join(REPO_ROOT, "tests", stem + ".ysu")


def compile_kernel(hd, nqh, nkvh, ps, warps=None):
    """Invokes the real Y CLI, then reads the launch geometry back out of the
    emitter's own PTX comment so the launch cannot silently disagree with
    what the kernel was compiled for."""
    src = ysu_path(hd, nqh, nkvh, ps, warps)
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    res = subprocess.run([Y_BIN, src, "--emit-ptx"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("Y compile failed for %s:\n%s\n%s" % (src, res.stdout, res.stderr))
    ptx = src[:-4] + ".ptx"
    text = open(ptx).read()
    marker = "[Y PAGED DECODE ATTENTION]"
    if marker not in text:
        raise RuntimeError(
            "%s has no '%s' comment - the attention dispatch did not fire "
            "(the emitter falls back to generic scalar lowering silently)" % (ptx, marker))
    line = [l for l in text.splitlines() if marker in l][0]
    real_warps = int(line.split("|")[1].strip().split()[0])
    name = os.path.basename(src)[:-4]
    return ptx, name, real_warps * 32


def make_inputs(hd, nqh, nkvh, ps, seq_lens, num_pages, seed=0):
    torch.manual_seed(seed)
    ns = len(seq_lens)
    max_pages = max(1, max((int(x) + ps - 1) // ps for x in seq_lens))
    q = torch.randn(ns, nqh, hd, dtype=torch.float16, device=DEV)
    kc = torch.randn(num_pages, ps, nkvh, hd, dtype=torch.float16, device=DEV)
    vc = torch.randn(num_pages, ps, nkvh, hd, dtype=torch.float16, device=DEV)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    pt = torch.stack([torch.randperm(num_pages, generator=g)[:max_pages] for _ in range(ns)])
    pt = pt.to(torch.int32).to(DEV)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    return q, kc, vc, pt, sl, max_pages


def reference(q, kc, vc, pt, sl, hd, nqh, nkvh, ps):
    ns = q.shape[0]
    gqa = nqh // nkvh
    out = torch.zeros(ns, nqh, hd, dtype=torch.float32, device=DEV)
    scale = 1.0 / (hd ** 0.5)
    for s in range(ns):
        L = int(sl[s].item())
        if L <= 0:
            continue
        t = torch.arange(L, device=DEV)
        k_all = kc[pt[s, t // ps].long(), (t % ps).long()].float()
        v_all = vc[pt[s, t // ps].long(), (t % ps).long()].float()
        for h in range(nqh):
            k = k_all[:, h // gqa]
            v = v_all[:, h // gqa]
            p = torch.softmax((q[s, h].float() @ k.T) * scale, dim=-1)
            out[s, h] = p @ v
    return out


def make_y_runner(ptx, name, threads, q, kc, vc, pt, sl, max_pages, out):
    mod = cp.RawModule(path=ptx)
    fn = mod.get_function(name)
    nqh, ns = q.shape[1], q.shape[0]
    args = (cp.asarray(q), cp.asarray(kc), cp.asarray(vc),
            cp.asarray(pt), cp.asarray(sl), cp.asarray(out), np.int32(max_pages))
    grid = (nqh, ns, 1)
    block = (threads, 1, 1)

    def run():
        fn(grid, block, args)
    return run


def make_flashinfer_runner(flashinfer, kc, vc, pt, sl, q, hd, nqh, nkvh, ps):
    """FlashInfer wants a flat page-index list plus per-sequence page counts;
    convert this harness's [num_seqs, max_pages] table into that form."""
    ns = q.shape[0]
    lens = [int(x.item()) for x in sl]
    npages = [max(1, (L + ps - 1) // ps) for L in lens]
    indptr = torch.tensor([0] + list(np.cumsum(npages)), dtype=torch.int32, device=DEV)
    indices = torch.cat([pt[s, :npages[s]] for s in range(ns)]).to(torch.int32)
    last = torch.tensor([(L - 1) % ps + 1 if L > 0 else 0 for L in lens],
                        dtype=torch.int32, device=DEV)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(indptr, indices, last, nqh, nkvh, hd, ps,
                 q_data_type=torch.float16, kv_data_type=torch.float16)

    def run():
        wrapper.run(q, (kc, vc))
    return run


TARGET_BATCH_US = 8000.0   # per timed batch, so host launch overhead is negligible


def calibrate_iters(runner):
    """Iterations needed for one timed batch to last ~TARGET_BATCH_US.

    Both sides are driven from Python (a cupy `RawKernel.__call__` for Y, a
    FlashInfer wrapper call for the baseline) and each carries tens of
    microseconds of HOST overhead. With a short batch that overhead lands
    inside the CUDA-event window and inflates the result - it is what made an
    earlier revision of this harness report Y at 223us for a shape it runs in
    44us, and report ctx=4096 as FASTER than ctx=1024, which is impossible for
    a sequential scan over tokens. Sizing every batch to the same wall time
    fixes it for both sides symmetrically.
    """
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(3):
        runner()
    ev1.record()
    torch.cuda.synchronize()
    per = ev0.elapsed_time(ev1) / 3 * 1000.0
    if per <= 0:
        return 20
    return int(max(5, min(2000, TARGET_BATCH_US / per)))


def time_interleaved(runners):
    """A/B-interleaved timing with per-round rotation; returns per-runner
    (best, median) microseconds. See the module docstring for why the
    minimum is the ranking statistic."""
    n = len(runners)
    # Warm up BOTH sides first: cupy loads the module on first launch and
    # FlashInfer does plan/JIT work, and neither belongs in a timed round.
    for r in runners:
        for _ in range(10):
            r()
    torch.cuda.synchronize()
    iters = [calibrate_iters(r) for r in runners]

    samples = [[] for _ in range(n)]
    for rnd in range(INTERLEAVE_ROUNDS):
        for off in range(n):
            i = (off + rnd) % n
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(iters[i]):
                runners[i]()
            ev1.record()
            torch.cuda.synchronize()
            samples[i].append(ev0.elapsed_time(ev1) / iters[i] * 1000.0)
    return [(min(s), statistics.median(s)) for s in samples]


def ramp():
    x = torch.randn(4096, 4096, dtype=torch.float16, device=DEV)
    t0 = time.time()
    while time.time() - t0 < RAMP_SECONDS:
        for _ in range(20):
            _ = x @ x
        torch.cuda.synchronize()
    del x


CORRECTNESS_CASES = [
    # (label, seq_lens, num_pages)
    ("single seq, partial page",        [37],                              512),
    ("exact page multiple",             [64],                              512),
    ("one token",                       [1],                               512),
    ("len below warp count",            [3],                               512),
    ("zero length",                     [0],                               512),
    ("ragged batch incl. zero",         [0, 1, 7, 63, 64, 65, 200, 33],    512),
    ("long context 4096",               [4096],                            512),
    ("long ragged batch",               [1000, 4095, 2, 2048],             512),
]

PERF_CASES = [
    # (label, batch, seq_len)
    ("batch 1,  ctx 1024",   1, 1024),
    ("batch 1,  ctx 4096",   1, 4096),
    ("batch 8,  ctx 1024",   8, 1024),
    ("batch 8,  ctx 4096",   8, 4096),
    ("batch 32, ctx 1024",  32, 1024),
    ("batch 32, ctx 4096",  32, 4096),
]

HD, NQH, NKVH, PS = 128, 32, 8, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correctness-only", action="store_true")
    args = ap.parse_args()

    print("=" * 100)
    print("   REAL Y-COMPILER PAGED DECODE ATTENTION vs FlashInfer BatchDecodeWithPagedKVCacheWrapper")
    print("=" * 100)
    print("[*] head_dim=%d  q_heads=%d  kv_heads=%d (GQA %d:1)  page_size=%d"
          % (HD, NQH, NKVH, HD and NQH // NKVH, PS))

    ptx, name, threads = compile_kernel(HD, NQH, NKVH, PS)
    print("[*] Compiled: %s  (%d threads/CTA)" % (os.path.relpath(ptx, REPO_ROOT), threads))
    ptx8, name8, threads8 = compile_kernel(HD, NQH, NKVH, PS, 8)
    print("[*] Compiled: %s  (%d threads/CTA)" % (os.path.relpath(ptx8, REPO_ROOT), threads8))
    print()

    # ---------------- correctness ----------------
    print("Correctness (vs PyTorch f32 reference walking the same shuffled page table)")
    print("-" * 100)
    print("%-32s | %-10s | %-12s | %s" % ("case", "seqs", "rel L2", "verdict"))
    print("-" * 100)
    worst, fails = 0.0, 0
    for label, lens, npg in CORRECTNESS_CASES:
        q, kc, vc, pt, sl, mp = make_inputs(HD, NQH, NKVH, PS, lens, npg)
        out = torch.zeros(len(lens), NQH, HD, dtype=torch.float16, device=DEV)
        make_y_runner(ptx, name, threads, q, kc, vc, pt, sl, mp, out)()
        cp.cuda.Device(0).synchronize()
        ref = reference(q, kc, vc, pt, sl, HD, NQH, NKVH, PS)
        got = out.float()

        bad = None
        if not torch.isfinite(got).all():
            bad = "NaN/Inf in output"
        for s, L in enumerate(lens):
            if L <= 0 and got[s].abs().max().item() != 0.0:
                bad = "seq_len=0 row %d not zeroed" % s
        err = 0.0 if ref.norm().item() == 0 else ((got - ref).norm() / ref.norm()).item()
        worst = max(worst, err)
        ok = bad is None and err <= TOL_REL_L2
        if not ok:
            fails += 1
        print("%-32s | %-10d | %-12.2e | %s" % (label, len(lens), err, bad or ("OK" if ok else "FAIL")))
    print("-" * 100)
    print("worst relative L2 = %.2e over %d cases, %d failures\n" % (worst, len(CORRECTNESS_CASES), fails))

    if args.correctness_only:
        return 1 if fails else 0

    # ---------------- performance ----------------
    flashinfer = import_flashinfer()
    if flashinfer is None:
        print("[!] flashinfer not installed - skipping the performance comparison.")
        return 1 if fails else 0

    print("Performance vs FlashInfer (ramped clocks, A/B interleaved, min over %d rounds)"
          % INTERLEAVE_ROUNDS)
    print("-" * 100)
    ramp()
    print("[*] SM clock after ramp: %d MHz" % sm_clock_mhz())
    print("-" * 100)
    print("%-20s | %-16s | %-18s | %-18s | %-9s | %s"
          % ("case", "FlashInfer us", "Y 32w us (min)", "Y 8w us (min)", "best vs FI", "KV GB/s"))
    print("-" * 100)

    for label, batch, ctx in PERF_CASES:
        lens = [ctx] * batch
        npages_needed = batch * ((ctx + PS - 1) // PS)
        num_pages = max(512, npages_needed * 2)
        q, kc, vc, pt, sl, mp = make_inputs(HD, NQH, NKVH, PS, lens, num_pages)
        out = torch.zeros(batch, NQH, HD, dtype=torch.float16, device=DEV)
        out8 = torch.zeros(batch, NQH, HD, dtype=torch.float16, device=DEV)
        y_run = make_y_runner(ptx, name, threads, q, kc, vc, pt, sl, mp, out)
        y8_run = make_y_runner(ptx8, name8, threads8, q, kc, vc, pt, sl, mp, out8)
        fi_run = make_flashinfer_runner(flashinfer, kc, vc, pt, sl, q, HD, NQH, NKVH, PS)

        (fi_best, fi_med), (y_best, y_med), (y8_best, y8_med) = time_interleaved(
            [fi_run, y_run, y8_run])

        # Bytes the kernel actually moves: K and V for every cached token and
        # KV head, re-read once per query head sharing that KV head (GQA).
        # Reporting only the UNIQUE footprint would understate traffic by the
        # GQA ratio and make an L2-served run look like it beat DRAM peak.
        unique = batch * ctx * NKVH * HD * 2 * 2
        kv_bytes = unique * (NQH // NKVH)
        best_y = min(y_best, y8_best)
        gbs = kv_bytes / (best_y * 1e-6) / 1e9
        print("%-20s | %-16s | %-18s | %-18s | %-9s | %.0f"
              % (label,
                 "%.2f" % fi_best,
                 "%.2f (med %.1f)" % (y_best, y_med),
                 "%.2f (med %.1f)" % (y8_best, y8_med),
                 "%.2fx" % (fi_best / best_y),
                 gbs))

    print("-" * 100)
    print("[*] DRAM peak on this card is ~672 GB/s; an effective rate above that")
    print("    means the KV cache was served from L2, not memory.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
