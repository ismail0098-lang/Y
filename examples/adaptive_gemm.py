#!/usr/bin/env python3
"""Run after cargo build --release --lib; needs NumPy and an NVIDIA CUDA driver.

    python3 examples/adaptive_gemm.py --cache-dir target/python-jit-cache 512 512 512

This example owns a CUDA context and raw allocations. The adaptive wrapper only
borrows them, making the cleanup order explicit. Production embedders can borrow
their existing context instead. No CUDA toolkit or PyTorch is needed here.
"""

import argparse
import ctypes
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from adaptive_jit import AdaptiveGemm  # noqa: E402


class _Cuda:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        u64, pointer = ctypes.c_uint64, ctypes.c_void_p
        signatures = {
            "cuInit": [ctypes.c_uint32],
            "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            "cuCtxCreate_v2": [ctypes.POINTER(pointer), ctypes.c_uint32, ctypes.c_int],
            "cuCtxDestroy_v2": [pointer],
            "cuCtxSynchronize": [],
            "cuMemAlloc_v2": [ctypes.POINTER(u64), ctypes.c_size_t],
            "cuMemFree_v2": [u64],
            "cuMemcpyHtoD_v2": [u64, pointer, ctypes.c_size_t],
            "cuMemcpyDtoH_v2": [pointer, u64, ctypes.c_size_t],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, ctypes.c_int
        self.context = ctypes.c_void_p()
        self.allocations = []

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with CUDA status {status}")

    def __enter__(self):
        self.call("cuInit", 0)
        device = ctypes.c_int()
        self.call("cuDeviceGet", ctypes.byref(device), 0)
        self.call("cuCtxCreate_v2", ctypes.byref(self.context), 0, device.value)
        return self

    def allocate(self, array, *, upload=False):
        address = ctypes.c_uint64()
        self.call("cuMemAlloc_v2", ctypes.byref(address), array.nbytes)
        self.allocations.append(address.value)
        if upload:
            self.call("cuMemcpyHtoD_v2", address.value, array.ctypes.data, array.nbytes)
        return address.value

    def download(self, array, address):
        self.call("cuMemcpyDtoH_v2", array.ctypes.data, address, array.nbytes)

    def __exit__(self, exc_type, exc, traceback):
        errors = []
        operations = [("cuCtxSynchronize", ())]
        operations.extend(("cuMemFree_v2", (address,)) for address in reversed(self.allocations))
        operations.append(("cuCtxDestroy_v2", (self.context,)))
        for name, args in operations:
            try:
                self.call(name, *args)
            except RuntimeError as error:
                errors.append(str(error))
        if errors:
            message = "CUDA cleanup: " + "; ".join(errors)
            if exc is None:
                raise RuntimeError(message)
            if hasattr(exc, "add_note"):
                exc.add_note(message)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--library", type=Path, help="path to liby.so")
    parser.add_argument("dimensions", nargs="*", type=int, metavar="DIM")
    args = parser.parse_args()
    if len(args.dimensions) not in (0, 3):
        parser.error("provide either no dimensions or M N K")
    m, n, k = args.dimensions or (256, 256, 256)
    if any(d < 1 or d > 16384 for d in (m, n, k)) or n % 16 or k % 16:
        parser.error("M, N, K must be in 1..16384; N and K must be multiples of 16")
    rng = np.random.default_rng(47)
    a = rng.uniform(-1, 1, (m, k)).astype(np.float16)
    b = rng.uniform(-1, 1, (k, n)).astype(np.float16)
    c = np.empty((m, n), dtype=np.float32)
    with _Cuda() as cuda:
        a_ptr = cuda.allocate(a, upload=True)
        b_ptr = cuda.allocate(b, upload=True)
        c_ptr = cuda.allocate(c)
        with AdaptiveGemm(library=args.library, hot_threshold=2, cache_dir=args.cache_dir) as jit:
            start = time.perf_counter()
            jit.prepare(m, n, k)
            stats = jit.stats(m, n, k)
            print(f"{m}x{n}x{k}: prepare {(time.perf_counter() - start) * 1000:.3f} ms; cache_hit={stats['cache_hit']}")
            for call in range(1, 4):
                jit.launch(m, n, k, a_ptr, b_ptr, c_ptr)
                jit.synchronize()
                print(f"launch {call}: {jit.stats(m, n, k)['tier']}; pending={jit.pending_shapes()}")
            start = time.perf_counter()
            reports = jit.tune_hot(1)
            print(f"explicit maintenance: {time.perf_counter() - start:.3f} s; {len(reports)} shape(s)")
            jit.launch(m, n, k, a_ptr, b_ptr, c_ptr)
            jit.synchronize()
            cuda.download(c, c_ptr)
            samples = np.arange(64)
            rows, columns = samples * 37 % m, samples * 71 % n
            references = np.array([np.dot(a[row].astype(np.float64), b[:, column].astype(np.float64))
                                   for row, column in zip(rows, columns)])
            actual = c[rows, columns].astype(np.float64)
            relative_l2 = np.linalg.norm(actual - references) / max(np.linalg.norm(references), np.finfo(np.float64).tiny)
            if not np.isfinite(relative_l2) or relative_l2 > 0.002:
                raise RuntimeError(f"CPU reference comparison failed: relative L2 {relative_l2}")
            stats = jit.stats(m, n, k)
            print(f"64 CPU-reference samples: relative L2 {relative_l2:.3e}")
            print(f"{stats['tier']}; candidates measured={stats['candidates_measured']}; "
                  f"tuning time={stats['tuning_time_seconds']:.3f} s; cache_hit={stats['cache_hit']}")
            if stats["cache_error"]:
                print(f"cache unavailable: {stats['cache_error']}")
            if stats["tuning_error"]:
                print(f"tuning failed: {stats['tuning_error']}")
        # close() above unloaded Y kernels; the host context still works.
        cuda.call("cuCtxSynchronize")


if __name__ == "__main__":
    main()
