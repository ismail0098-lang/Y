"""ctypes interface to Y's adaptive FP16 GEMM runtime.

Build the library with ``cargo build --release --lib``. Create AdaptiveGemm only
while the host's CUDA context is current. The host owns that context and must
keep it alive and current until close() succeeds. Every method must run on the
thread that created the wrapper. Use a with block or explicit close(); there is
no GPU work in a finalizer.

launch(m, n, k, a, b, c) accepts raw CUDA device addresses: contiguous row-major
A[M,K] and B[K,N] contain float16, and C[M,N] contains float32. Buffers must be
16-byte aligned, allocated in the borrowed context, and large enough. C must
not overlap A or B. Keep all allocations alive until synchronize() completes.
Launches use CUDA's default stream. The caller must order writes from other
streams before launching, and order other consumers after completion. For an
embedding framework, synchronizing its producer streams before launch and
calling this wrapper's synchronize() before consumption is a simple approach.

Deferred tuning is the default: launch only queues hot shapes; tune_hot() runs
synchronous maintenance, which can take seconds. Its limit counts shapes and
is not a time budget. prepare() compiles a shape without consuming user data.
"""

from __future__ import annotations

import ctypes
import json
import math
import operator
import os
from pathlib import Path
import threading


class AdaptiveJitError(RuntimeError):
    """A Y adaptive-runtime or shared-library error."""


class _Config(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("tuning_policy", ctypes.c_uint32),
        ("max_cached_shapes", ctypes.c_uint32),
        ("max_candidates", ctypes.c_uint32),
        ("max_disk_cache_entries", ctypes.c_uint32),
        ("hot_threshold", ctypes.c_uint64),
        ("min_improvement", ctypes.c_double),
        ("cache_dir", ctypes.c_char_p),
    ]


def _uint(value, name, bits=32, minimum=0):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if not minimum <= result < 1 << bits:
        raise ValueError(f"{name} must be between {minimum} and {(1 << bits) - 1}")
    return result


def _shape(m, n, k):
    dimensions = tuple(_uint(value, name, minimum=1) for name, value in zip("MNK", (m, n, k)))
    if any(value > 16384 for value in dimensions):
        raise ValueError("M, N and K must be at most 16384")
    if dimensions[1] % 16 or dimensions[2] % 16:
        raise ValueError("N and K must be multiples of 16")
    return dimensions


def _device_pointer(value, name):
    result = _uint(value, name, bits=64, minimum=1)
    if result % 16:
        raise ValueError(f"{name} must be 16-byte aligned")
    return result


def _load_library(library):
    path = Path(library) if library is not None else Path(__file__).resolve().parents[1] / "target/release/liby.so"
    try:
        lib = ctypes.CDLL(str(path))
        error = ctypes.POINTER(ctypes.c_void_p)
        handle = ctypes.c_void_p
        u32, u64 = ctypes.c_uint32, ctypes.c_uint64
        signatures = {
            "config_init": ([ctypes.POINTER(_Config), u32, error], ctypes.c_int32),
            "create_current": ([ctypes.POINTER(_Config), error], handle),
            "destroy": ([handle, error], ctypes.c_int32),
            "prepare": ([handle, u32, u32, u32, error], ctypes.c_int32),
            "launch": ([handle, u32, u32, u32, u64, u64, u64, error], ctypes.c_int32),
            "synchronize": ([handle, error], ctypes.c_int32),
            "stats_json": ([handle, u32, u32, u32, error], ctypes.c_void_p),
            "pending_json": ([handle, error], ctypes.c_void_p),
            "tune_hot_json": ([handle, u32, error], ctypes.c_void_p),
        }
        for suffix, (args, result) in signatures.items():
            function = getattr(lib, "y_adaptive_jit_" + suffix)
            function.argtypes, function.restype = args, result
        lib.y_free_string.argtypes = [ctypes.c_void_p]
        lib.y_free_string.restype = None
    except (OSError, AttributeError) as error:
        raise AdaptiveJitError(f"Cannot load Y adaptive ABI from {path}; build it with cargo build --release --lib: {error}") from error
    return lib


class AdaptiveGemm:
    """Borrow the current CUDA context for adaptive FP16-to-FP32 GEMM.

    library selects liby.so; None uses this checkout's target/release/liby.so.
    All optional settings except tuning_policy inherit the native defaults.
    cache_dir opts into a persistent decision cache in a trusted local directory.
    This wrapper cannot validate device allocations behind raw integer pointers.
    """

    def __init__(self, *, library=None, tuning_policy="deferred", hot_threshold=None,
                 max_cached_shapes=None, max_candidates=None, min_improvement=None,
                 cache_dir=None, max_disk_cache_entries=None):
        policies = {"deferred": 0, "on_launch": 1, "disabled": 2}
        if not isinstance(tuning_policy, str) or tuning_policy not in policies:
            raise ValueError("tuning_policy must be deferred, on_launch, or disabled")
        settings = {"tuning_policy": policies[tuning_policy]}
        for name, value, bits in (
            ("hot_threshold", hot_threshold, 64),
            ("max_cached_shapes", max_cached_shapes, 32),
            ("max_candidates", max_candidates, 32),
            ("max_disk_cache_entries", max_disk_cache_entries, 32),
        ):
            if value is not None:
                settings[name] = _uint(value, name, bits, minimum=1)
        if max_candidates is not None and not 2 <= settings["max_candidates"] <= 64:
            raise ValueError("max_candidates must be between 2 and 64")
        if min_improvement is not None:
            if isinstance(min_improvement, (bool, str, bytes)):
                raise TypeError("min_improvement must be a number")
            improvement = float(min_improvement)
            if not math.isfinite(improvement) or not 0 <= improvement < 1:
                raise ValueError("min_improvement must be finite and in [0, 1)")
            settings["min_improvement"] = improvement
        cache_bytes = None
        if cache_dir is not None:
            path = os.fspath(cache_dir)
            if not isinstance(path, str):
                raise TypeError("cache_dir must be a string or text path")
            if not path or "\0" in path:
                raise ValueError("cache_dir must be nonempty and contain no NUL")
            cache_bytes = path.encode("utf-8")
        self._owner = threading.current_thread()
        self._handle = None
        self._lib = _load_library(library)
        config = _Config()
        self._status("config_init", ctypes.byref(config), ctypes.sizeof(config))
        for name, value in settings.items():
            setattr(config, name, value)
        config.cache_dir = cache_bytes
        error = ctypes.c_void_p()
        handle = self._lib.y_adaptive_jit_create_current(ctypes.byref(config), ctypes.byref(error))
        message = self._take_string(error.value) if error.value else None
        if not handle:
            raise AdaptiveJitError(message or "Y failed to borrow the current CUDA context")
        self._handle = handle

    def _take_string(self, pointer):
        try:
            return ctypes.string_at(pointer).decode("utf-8", errors="replace")
        finally:
            self._lib.y_free_string(pointer)

    def _status(self, suffix, *arguments):
        error = ctypes.c_void_p()
        status = getattr(self._lib, "y_adaptive_jit_" + suffix)(*arguments, ctypes.byref(error))
        message = self._take_string(error.value) if error.value else None
        if status != 0:
            raise AdaptiveJitError(message or f"Y {suffix} failed with status {status}")

    def _check(self):
        if threading.current_thread() is not self._owner:
            raise AdaptiveJitError("AdaptiveGemm must be used and closed on its creating thread")
        if self._handle is None:
            raise AdaptiveJitError("AdaptiveGemm is closed")
        return self._handle

    def _json(self, suffix, *arguments):
        handle = self._check()
        error = ctypes.c_void_p()
        pointer = getattr(self._lib, "y_adaptive_jit_" + suffix)(handle, *arguments, ctypes.byref(error))
        message = self._take_string(error.value) if error.value else None
        if not pointer:
            raise AdaptiveJitError(message or f"Y {suffix} failed")
        return json.loads(self._take_string(pointer))

    def prepare(self, m, n, k):
        """Compile a shape for later launches; does not count as a launch."""
        self._status("prepare", self._check(), *_shape(m, n, k))

    def launch(self, m, n, k, a, b, c):
        """Enqueue GEMM on the default stream using caller-owned device pointers."""
        self._status("launch", self._check(), *_shape(m, n, k),
                     _device_pointer(a, "a"), _device_pointer(b, "b"), _device_pointer(c, "c"))

    def synchronize(self):
        """Wait for work in the borrowed CUDA context to complete."""
        self._status("synchronize", self._check())

    def stats(self, m, n, k):
        """Return a stats dict for a resident shape; absent shapes raise an error."""
        return self._json("stats_json", *_shape(m, n, k))

    def pending_shapes(self):
        """Return hot resident shapes awaiting maintenance, as (M, N, K) tuples."""
        return [tuple(shape) for shape in self._json("pending_json")]

    def tune_hot(self, max_shapes=1):
        """Synchronously tune up to max_shapes queued shapes and return reports."""
        return self._json("tune_hot_json", _uint(max_shapes, "max_shapes"))

    def close(self):
        """Synchronize and release Y's resources; leaves the host context alive.

        On failure this object remains open, so the host can restore its current
        context and retry. Repeated successful closes are harmless.
        """
        if self._handle is not None:
            self._status("destroy", self._check())
            self._handle = None

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
