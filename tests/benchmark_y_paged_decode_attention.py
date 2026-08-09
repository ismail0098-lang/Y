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


def ysu_path(hd, nqh, nkvh, ps, warps=None, splits=None):
    if splits is None:
        stem = "paged_decode_attention_%d_%d_%d_%d" % (hd, nqh, nkvh, ps)
    else:
        stem = "paged_decode_attention_split_%d_%d_%d_%d_%d" % (hd, nqh, nkvh, ps, splits)
    if warps is not None:
        stem += "_%d" % warps
    return os.path.join(REPO_ROOT, "tests", stem + ".ysu")


def compile_kernel(hd, nqh, nkvh, ps, warps=None, splits=None):
    """Invokes the real Y CLI, then reads the launch geometry back out of the
    emitter's own PTX comment so the launch cannot silently disagree with
    what the kernel was compiled for."""
    src = ysu_path(hd, nqh, nkvh, ps, warps, splits)
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


def make_y_split_runner(ptx, name, threads, splits, nkvh,
                        q, kc, vc, pt, sl, max_pages, out):
    """Drives the split-K shape: the partial-state kernel over
    (num_kv_heads, num_seqs, splits), then the combine over
    (num_q_heads, num_seqs).

    BOTH launches are inside the timed region, and the scratch buffers are
    allocated once outside it. That is the honest accounting: the second
    launch is a real cost of this shape (~3us of launch latency on top of a
    kernel that at batch 1 runs in ~10), and a comparison that timed only the
    first would be measuring a kernel that has not produced `Out` yet.
    """
    mod = cp.RawModule(path=ptx)
    fn = mod.get_function(name)
    fn_red = mod.get_function(name + "_reduce")
    ns, nqh = q.shape[0], q.shape[1]
    hd = q.shape[2]
    partial = cp.empty((ns, nqh, splits, hd), dtype=cp.float32)
    meta = cp.empty((ns, nqh, splits, 2), dtype=cp.float32)
    args = (cp.asarray(q), cp.asarray(kc), cp.asarray(vc),
            cp.asarray(pt), cp.asarray(sl), cp.asarray(out),
            partial, meta, np.int32(max_pages))
    grid = (nkvh, ns, splits)
    block = (threads, 1, 1)
    grid_red = (nqh, ns, 1)

    def run():
        fn(grid, block, args)
        fn_red(grid_red, (32, 1, 1), args)
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
    # The shapes the performance table below actually times. Without these the
    # correctness set and the timed set do not intersect above batch 4, and a
    # kernel can be verified on shapes nobody measures.
    ("timed shape: b8 ctx 1024",        [1024] * 8,                       1024),
    ("timed shape: b32 ctx 1024",       [1024] * 32,                      4096),
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
# Two split-K configurations, for the same reason the single-pass kernel is
# measured at both 32 and 8 warps: the CTA count that fills the GPU depends on
# the batch, and the batch is a runtime value. 16 splits is the batch-1
# configuration, 4 the batch-many one. Correctness is checked on the first.
SPLIT_WARPS = 8
SPLITS_SMALL_BATCH = 16
SPLITS_LARGE_BATCH = 4


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
    ptxs, names, threadss = compile_kernel(HD, NQH, NKVH, PS, SPLIT_WARPS, SPLITS_SMALL_BATCH)
    print("[*] Compiled: %s  (%d threads/CTA, %d splits, %d q heads/CTA)"
          % (os.path.relpath(ptxs, REPO_ROOT), threadss, SPLITS_SMALL_BATCH, NQH // NKVH))
    ptxl, namel, threadsl = compile_kernel(HD, NQH, NKVH, PS, SPLIT_WARPS, SPLITS_LARGE_BATCH)
    print("[*] Compiled: %s  (%d threads/CTA, %d splits, %d q heads/CTA)"
          % (os.path.relpath(ptxl, REPO_ROOT), threadsl, SPLITS_LARGE_BATCH, NQH // NKVH))
    print()

    # ---------------- correctness ----------------
    # Both shapes are checked against the same reference. The split kernel is
    # the one that can be wrong in new ways - a partial state that is empty,
    # a combine that rescales by the wrong max - so the ragged and short cases
    # matter more for it than for the single-pass kernel, and the split count
    # (16) deliberately exceeds several of the sequence lengths below.
    print("Correctness (vs PyTorch f32 reference walking the same shuffled page table)")
    print("-" * 100)
    print("%-32s | %-6s | %-14s | %-14s | %s"
          % ("case", "seqs", "rel L2 (1-pass)", "rel L2 (split)", "verdict"))
    print("-" * 100)
    worst, fails = 0.0, 0
    for label, lens, npg in CORRECTNESS_CASES:
        q, kc, vc, pt, sl, mp = make_inputs(HD, NQH, NKVH, PS, lens, npg)
        ref = reference(q, kc, vc, pt, sl, HD, NQH, NKVH, PS)
        errs, bad = [], None
        for tag, runner_factory in (
            ("1-pass", lambda o: make_y_runner(ptx, name, threads, q, kc, vc, pt, sl, mp, o)),
            ("split", lambda o: make_y_split_runner(ptxs, names, threadss, SPLITS_SMALL_BATCH,
                                                    NKVH, q, kc, vc, pt, sl, mp, o)),
        ):
            out = torch.zeros(len(lens), NQH, HD, dtype=torch.float16, device=DEV)
            runner_factory(out)()
            cp.cuda.Device(0).synchronize()
            got = out.float()
            if not torch.isfinite(got).all():
                bad = "%s: NaN/Inf in output" % tag
            for s, L in enumerate(lens):
                if L <= 0 and got[s].abs().max().item() != 0.0:
                    bad = "%s: seq_len=0 row %d not zeroed" % (tag, s)
            err = 0.0 if ref.norm().item() == 0 else ((got - ref).norm() / ref.norm()).item()
            errs.append(err)
            worst = max(worst, err)
        ok = bad is None and max(errs) <= TOL_REL_L2
        if not ok:
            fails += 1
        print("%-32s | %-6d | %-14.2e | %-14.2e | %s"
              % (label, len(lens), errs[0], errs[1], bad or ("OK" if ok else "FAIL")))
    print("-" * 100)
    print("worst relative L2 = %.2e over %d cases x 2 shapes, %d failures\n"
          % (worst, len(CORRECTNESS_CASES), fails))

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
    print("%-20s | %-9s | %-9s | %-9s | %-10s | %-10s | %-9s | %s"
          % ("case", "FI us", "1pass 32w", "1pass 8w",
             "split %2dx" % SPLITS_SMALL_BATCH, "split %2dx" % SPLITS_LARGE_BATCH,
             "best vs FI", "KV GB/s"))
    print("-" * 118)

    rows = []
    for label, batch, ctx in PERF_CASES:
        lens = [ctx] * batch
        npages_needed = batch * ((ctx + PS - 1) // PS)
        num_pages = max(512, npages_needed * 2)
        q, kc, vc, pt, sl, mp = make_inputs(HD, NQH, NKVH, PS, lens, num_pages)
        outs_buf = [torch.zeros(batch, NQH, HD, dtype=torch.float16, device=DEV)
                    for _ in range(4)]
        y_run = make_y_runner(ptx, name, threads, q, kc, vc, pt, sl, mp, outs_buf[0])
        y8_run = make_y_runner(ptx8, name8, threads8, q, kc, vc, pt, sl, mp, outs_buf[1])
        ys_run = make_y_split_runner(ptxs, names, threadss, SPLITS_SMALL_BATCH, NKVH,
                                     q, kc, vc, pt, sl, mp, outs_buf[2])
        yl_run = make_y_split_runner(ptxl, namel, threadsl, SPLITS_LARGE_BATCH, NKVH,
                                     q, kc, vc, pt, sl, mp, outs_buf[3])
        fi_run = make_flashinfer_runner(flashinfer, kc, vc, pt, sl, q, HD, NQH, NKVH, PS)

        (fi_best, _), (y_best, _), (y8_best, _), (ys_best, ys_med), (yl_best, _) = \
            time_interleaved([fi_run, y_run, y8_run, ys_run, yl_run])

        # The KV cache footprint the kernel MUST touch: K and V for every
        # cached token and KV head. The single-pass shape re-reads it once per
        # query head sharing a KV head (x4 here); the split-K shape does not,
        # which is the entire point of it - so the rate below is quoted on the
        # UNIQUE bytes, and a figure above DRAM peak means L2 served it.
        unique = batch * ctx * NKVH * HD * 2 * 2
        best_1pass = min(y_best, y8_best)
        best_split = min(ys_best, yl_best)
        rows.append((label, fi_best, best_1pass, best_split, ys_best))
        print("%-20s | %9.2f | %9.2f | %9.2f | %10.2f | %10.2f | %-9s | %.0f"
              % (label, fi_best, y_best, y8_best, ys_best, yl_best,
                 "%.2fx" % (fi_best / best_split),
                 unique / (best_split * 1e-6) / 1e9))

    print("-" * 118)
    print("[*] DRAM peak on this card is ~672 GB/s. The KV GB/s column is over the")
    print("    bytes the split-K kernel is obliged to read exactly once; above peak")
    print("    means the cache fit in L2.")
    print()
    print("Summary (x vs FlashInfer, >1 is Y faster). The split-K columns are the")
    print("two configurations above; batch size is a runtime value, so a deployment")
    print("picks one - `fixed 16` is what a single compiled binary would ship.")
    print("%-20s | %-12s | %-12s | %-12s | %s"
          % ("case", "1-pass best", "split fixed16", "split best", "split/1-pass"))
    for label, fi_b, one_b, sp_b, fixed_b in rows:
        print("%-20s | %-12s | %-12s | %-12s | %s"
              % (label, "%.2fx" % (fi_b / one_b), "%.2fx" % (fi_b / fixed_b),
                 "%.2fx" % (fi_b / sp_b), "%.2fx" % (one_b / sp_b)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
