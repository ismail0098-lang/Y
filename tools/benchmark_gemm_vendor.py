#!/usr/bin/env python3
"""Compare exported Y GEMM PTX with a matched cuBLAS GemmEx reference.

Both paths use the same preallocated row-major FP16 A/B and FP32 C buffers,
FP32 accumulation, alpha=1, beta=0, and the same CUDA default stream. No matrix
transfers, compilation, allocation, or correctness work is inside event timing.
A >=3-second alternating warmup precedes alternating-order paired timing rounds.
CUDA events measure batched dispatch on the GPU timeline; for tiny kernels this
can include gaps caused by Python/ctypes dispatch, not just kernel execution.

cuBLAS uses CUBLAS_COMPUTE_32F and the CUBLAS_GEMM_DEFAULT heuristic. This is a
matched library reference, not an exhaustive search for the fastest cuBLASLt
algorithm. There is no separate explicit workspace or algorithm search. See
https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex and its compute/math
mode tables. ABI declarations below follow locally installed cublas_api.h.

Requires NumPy, an NVIDIA driver, and cuBLAS; no CuPy, Torch, or Y Python API.
Outputs JSON to stdout; --output additionally creates a NEW file without
replacing any existing result. This tool runs one process/trial; repeat it in
separate processes when drawing performance conclusions.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


# cuda.h/cublas_api.h/library_types.h ABI enum values, not guessed defaults.
CUDA_R_32F, CUDA_R_16F = 0, 2
CUBLAS_COMPUTE_32F = 68
CUBLAS_GEMM_DEFAULT = -1
CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16


def bind(library, name, arguments):
    function = getattr(library, name)
    function.argtypes, function.restype = arguments, ct.c_int
    return function


class Cuda:
    """Own one retained primary context and benchmark-only driver resources."""

    def __init__(self, ordinal):
        self.ordinal = ordinal
        self.lib = ct.CDLL("libcuda.so.1")
        pointer, u64, integer = ct.c_void_p, ct.c_uint64, ct.c_int
        signatures = {
            "cuInit": [ct.c_uint],
            "cuDriverGetVersion": [ct.POINTER(integer)],
            "cuDeviceGet": [ct.POINTER(integer), integer],
            "cuDeviceGetName": [pointer, integer, integer],
            "cuDeviceGetAttribute": [ct.POINTER(integer), integer, integer],
            "cuDevicePrimaryCtxRetain": [ct.POINTER(pointer), integer],
            "cuDevicePrimaryCtxRelease_v2": [integer],
            "cuCtxGetCurrent": [ct.POINTER(pointer)],
            "cuCtxSetCurrent": [pointer],
            "cuCtxSynchronize": [],
            "cuMemAlloc_v2": [ct.POINTER(u64), ct.c_size_t],
            "cuMemFree_v2": [u64],
            "cuMemsetD32_v2": [u64, ct.c_uint, ct.c_size_t],
            "cuMemcpyHtoD_v2": [u64, pointer, ct.c_size_t],
            "cuMemcpyDtoH_v2": [pointer, u64, ct.c_size_t],
            "cuModuleLoadData": [ct.POINTER(pointer), pointer],
            "cuModuleUnload": [pointer],
            "cuModuleGetFunction": [ct.POINTER(pointer), pointer, ct.c_char_p],
            "cuFuncSetAttribute": [pointer, integer, integer],
            "cuLaunchKernel": [pointer, ct.c_uint, ct.c_uint, ct.c_uint,
                               ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint,
                               pointer, ct.POINTER(pointer), ct.POINTER(pointer)],
            "cuEventCreate": [ct.POINTER(pointer), ct.c_uint],
            "cuEventDestroy_v2": [pointer],
            "cuEventRecord": [pointer, pointer],
            "cuEventSynchronize": [pointer],
            "cuEventElapsedTime": [ct.POINTER(ct.c_float), pointer, pointer],
            "cuGetErrorName": [integer, ct.POINTER(ct.c_char_p)],
        }
        for name, arguments in signatures.items():
            bind(self.lib, name, arguments)
        self.cleanup = ExitStack()

    def check(self, name, *arguments):
        result = getattr(self.lib, name)(*arguments)
        if result:
            label = ct.c_char_p()
            self.lib.cuGetErrorName(result, ct.byref(label))
            detail = label.value.decode() if label.value else "unknown"
            raise RuntimeError(f"{name} failed: {detail} ({result})")

    def __enter__(self):
        try:
            self.check("cuInit", 0)
            self.device = ct.c_int()
            self.check("cuDeviceGet", ct.byref(self.device), self.ordinal)
            previous = ct.c_void_p()
            self.check("cuCtxGetCurrent", ct.byref(previous))
            self.context = ct.c_void_p()
            self.check("cuDevicePrimaryCtxRetain", ct.byref(self.context), self.device)
            self.cleanup.callback(self.check, "cuDevicePrimaryCtxRelease_v2", self.device)
            self.check("cuCtxSetCurrent", self.context)
            self.cleanup.callback(self.check, "cuCtxSetCurrent", previous)
            self.start, self.end = self.event(), self.event()
            return self
        except BaseException:
            self.cleanup.close()
            raise

    def __exit__(self, *exception):
        try:
            self.check("cuCtxSynchronize")
        finally:
            self.cleanup.close()
        return False

    def event(self):
        event = ct.c_void_p()
        self.check("cuEventCreate", ct.byref(event), 0)
        self.cleanup.callback(self.check, "cuEventDestroy_v2", event)
        return event

    def allocate(self, array, *, upload=False):
        pointer = ct.c_uint64()
        self.check("cuMemAlloc_v2", ct.byref(pointer), array.nbytes)
        self.cleanup.callback(self.check, "cuMemFree_v2", pointer.value)
        if upload:
            self.check("cuMemcpyHtoD_v2", pointer.value, array.ctypes.data, array.nbytes)
        return pointer.value

    def download(self, pointer, shape):
        result = np.empty(shape, dtype=np.float32)
        self.check("cuMemcpyDtoH_v2", result.ctypes.data, pointer, result.nbytes)
        return result

    def kernel(self, metadata, ptx, a, b, c):
        module = ct.c_void_p()
        ptx_buffer = ct.create_string_buffer(ptx)
        self.check("cuModuleLoadData", ct.byref(module), ptx_buffer)
        self.cleanup.callback(self.check, "cuModuleUnload", module)
        function = ct.c_void_p()
        self.check("cuModuleGetFunction", ct.byref(function), module, metadata["kernel_name"].encode())
        shared = metadata["shared_mem_bytes"]
        if shared:
            self.check("cuFuncSetAttribute", function, 8, shared)
        pointers = [ct.c_uint64(address) for address in (a, b, c)]
        parameters = (ct.c_void_p * 3)(*(ct.addressof(pointer) for pointer in pointers))
        arguments = (function, *metadata["grid"], *metadata["block"], shared, None, parameters, None)

        def launch(_parameter_values=pointers):
            # The default argument retains the three underlying pointer values.
            result = self.lib.cuLaunchKernel(*arguments)
            if result:
                raise RuntimeError(f"cuLaunchKernel failed with CUDA status {result}")

        return launch

    def measure(self, launch, iterations):
        self.check("cuEventRecord", self.start, None)
        for _ in range(iterations):
            launch()
        self.check("cuEventRecord", self.end, None)
        self.check("cuEventSynchronize", self.end)
        elapsed = ct.c_float()
        self.check("cuEventElapsedTime", ct.byref(elapsed), self.start, self.end)
        microseconds = elapsed.value * 1000 / iterations
        if not math.isfinite(microseconds) or microseconds <= 0:
            raise RuntimeError(f"invalid CUDA event duration: {microseconds}")
        return microseconds

    def metadata(self):
        name = ct.create_string_buffer(256)
        self.check("cuDeviceGetName", name, len(name), self.device)
        version = ct.c_int()
        self.check("cuDriverGetVersion", ct.byref(version))
        compute = []
        for attribute in (75, 76):
            value = ct.c_int()
            self.check("cuDeviceGetAttribute", ct.byref(value), attribute, self.device)
            compute.append(value.value)
        return {"device_ordinal": self.ordinal, "device_name": name.value.decode(),
                "driver_api_version": version.value, "compute_capability": compute}


def load_cublas(path):
    candidates = [str(path)] if path else [
        "/opt/cuda/targets/x86_64-linux/lib/libcublas.so",
        "/usr/local/cuda/lib64/libcublas.so", "libcublas.so.13", "libcublas.so.12", "libcublas.so",
    ]
    errors = []
    for candidate in candidates:
        try:
            return ct.CDLL(candidate), candidate
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError("Cannot load cuBLAS; use --cublas PATH. " + "; ".join(errors))


class Cublas:
    def __init__(self, cuda, library, shape, a, b, c):
        self.lib, self.path = load_cublas(library)
        pointer, integer = ct.c_void_p, ct.c_int
        signatures = {
            "cublasCreate_v2": [ct.POINTER(pointer)],
            "cublasDestroy_v2": [pointer],
            "cublasGetVersion_v2": [pointer, ct.POINTER(integer)],
            "cublasSetStream_v2": [pointer, pointer],
            "cublasSetPointerMode_v2": [pointer, integer],
            "cublasSetMathMode": [pointer, integer],
            "cublasGemmEx": [pointer, integer, integer, integer, integer, integer,
                             pointer, pointer, integer, integer, pointer, integer, integer,
                             pointer, pointer, integer, integer, integer, integer],
        }
        for name, arguments in signatures.items():
            bind(self.lib, name, arguments)
        self.handle = pointer()
        self.check("cublasCreate_v2", ct.byref(self.handle))
        cuda.cleanup.callback(self.check, "cublasDestroy_v2", self.handle)
        self.check("cublasSetStream_v2", self.handle, None)
        self.check("cublasSetPointerMode_v2", self.handle, 0)  # host alpha/beta
        self.check("cublasSetMathMode", self.handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
        version = integer()
        self.check("cublasGetVersion_v2", self.handle, ct.byref(version))
        self.version = version.value
        self.alpha, self.beta = ct.c_float(1), ct.c_float(0)
        m, n, k = shape
        # cuBLAS is column-major: C_row = A_row @ B_row is equivalent to
        # C_col[N,M] = B_col[N,K] @ A_col[K,M]. No transpose/copy occurs.
        self.arguments = (self.handle, 0, 0, n, m, k, ct.byref(self.alpha),
                          b, CUDA_R_16F, n, a, CUDA_R_16F, k,
                          ct.byref(self.beta), c, CUDA_R_32F, n,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)

    def check(self, name, *arguments):
        result = getattr(self.lib, name)(*arguments)
        if result:
            raise RuntimeError(f"{name} failed with cuBLAS status {result}")

    def launch(self):
        result = self.lib.cublasGemmEx(*self.arguments)
        if result:
            raise RuntimeError(f"cublasGemmEx failed with cuBLAS status {result}")


def checked_int(value, name, minimum=1, maximum=(1 << 31) - 1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def relative_l2(actual, reference):
    error, norm = 0.0, 0.0
    actual, reference = actual.ravel(), reference.ravel()
    if actual.shape != reference.shape:
        raise ValueError("reference and actual shapes differ")
    # Bound temporary CPU memory even for large matrices.
    for offset in range(0, actual.size, 1 << 20):
        observed = actual[offset:offset + (1 << 20)].astype(np.float64)
        expected = reference[offset:offset + (1 << 20)].astype(np.float64)
        if not np.isfinite(observed).all() or not np.isfinite(expected).all():
            return float("inf")
        difference = observed - expected
        error += float(np.dot(difference, difference))
        norm += float(np.dot(expected, expected))
    return math.sqrt(error / max(norm, np.finfo(np.float64).tiny))


def correctness(cuda, launches, shape, a_host, b_host, c_pointer, tolerance):
    m, n, _ = shape
    outputs = {}
    for name, launch in launches.items():
        # NaN-fill detects partial writes, including any illegal read of C at beta=0.
        cuda.check("cuMemsetD32_v2", c_pointer, 0x7fc00000, m * n)
        launch()
        cuda.check("cuCtxSynchronize")
        outputs[name] = cuda.download(c_pointer, (m, n))
    rng = np.random.default_rng(901)
    rows = np.r_[0, m - 1, m // 2, rng.integers(0, m, 125)]
    columns = np.r_[0, n - 1, n // 2, rng.integers(0, n, 125)]
    reference = np.array([np.dot(a_host[row].astype(np.float64), b_host[:, column].astype(np.float64))
                          for row, column in zip(rows, columns)])
    errors = {"full_y_vs_cublas_relative_l2": relative_l2(outputs["y"], outputs["cublas"]),
              "cpu_fp64_samples": len(rows)}
    for name, output in outputs.items():
        errors[f"sample_{name}_vs_cpu_fp64_relative_l2"] = relative_l2(output[rows, columns], reference)
    for name, value in errors.items():
        if name.endswith("relative_l2") and (not math.isfinite(value) or value > tolerance):
            raise RuntimeError(f"correctness check failed: {name}={value}, tolerance={tolerance}")
    return errors


def gpu_snapshot():
    command = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu", "--format=csv"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run(args, selected, ptx, manifest):
    shape = selected["shape"]
    m, n, k = shape
    a_host = random_f16((m, k), args.seed_a)
    b_host = random_f16((k, n), args.seed_b)
    c_host = np.empty((m, n), dtype=np.float32)
    result = {"schema": "y-gemm-vendor-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
              "manifest": str(args.manifest.resolve()), "export": manifest, "selected": selected,
              "ptx_sha256": hashlib.sha256(ptx).hexdigest(), "input_generator": "cuda_runtime::random_f16_bits",
              "seeds": {"a": args.seed_a, "b": args.seed_b},
              "python": sys.version, "numpy_version": np.__version__, "gpu_before": gpu_snapshot(),
              "environment": {name: os.environ[name] for name in (
                  "CUDA_VISIBLE_DEVICES", "CUDA_CACHE_DISABLE", "NVIDIA_TF32_OVERRIDE",
                  "CUBLAS_WORKSPACE_CONFIG", "CUDA_LAUNCH_BLOCKING") if name in os.environ}}
    with Cuda(args.device) as cuda:
        result.update(cuda.metadata())
        a, b = cuda.allocate(a_host, upload=True), cuda.allocate(b_host, upload=True)
        c = cuda.allocate(c_host)
        y_launch = cuda.kernel(selected, ptx, a, b, c)
        vendor = Cublas(cuda, args.cublas, shape, a, b, c)
        launches = {"y": y_launch, "cublas": vendor.launch}
        result["reference"] = {"api": "cublasGemmEx", "library": vendor.path,
                               "version": vendor.version, "algorithm": "CUBLAS_GEMM_DEFAULT",
                               "compute_type": "CUBLAS_COMPUTE_32F", "input_dtype": "float16",
                               "output_dtype": "float32", "layout": "row_major",
                               "math_mode": "CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION",
                               "alpha": 1, "beta": 0, "same_input_and_output_buffers": True,
                               "stream": "CUDA default stream", "explicit_workspace_bytes": None}
        result["correctness"] = correctness(cuda, launches, shape, a_host, b_host, c, args.tolerance)
        warmup_start = time.perf_counter()
        warmup_cycles = 0
        while time.perf_counter() - warmup_start < args.warmup_seconds:
            order = ("y", "cublas") if warmup_cycles % 2 == 0 else ("cublas", "y")
            for name in order:
                for _ in range(8):
                    launches[name]()
            cuda.check("cuCtxSynchronize")
            warmup_cycles += 1
        result["warmup_seconds"] = time.perf_counter() - warmup_start
        result["warmup_cycles"] = warmup_cycles
        records = []
        for round_index in range(args.rounds):
            order = ("y", "cublas") if (round_index + args.order_offset) % 2 == 0 else ("cublas", "y")
            record = {"round": round_index, "order": list(order)}
            for name in order:
                record[f"{name}_us"] = cuda.measure(launches[name], args.iterations)
            record["y_over_cublas"] = record["y_us"] / record["cublas_us"]
            records.append(record)
        result["pairs"] = records
    result["gpu_after"] = gpu_snapshot()
    result["iterations_per_measurement"] = args.iterations
    result["summary"] = {}
    for name in ("y", "cublas"):
        values = [record[f"{name}_us"] for record in records]
        median = statistics.median(values)
        result["summary"][name] = {"median_us": median, "p10_us": quantile(values, 0.1),
                                   "p90_us": quantile(values, 0.9), "tflops": 2 * m * n * k / median / 1e6}
    result["summary"]["median_paired_y_over_cublas"] = statistics.median(record["y_over_cublas"] for record in records)
    result["limitations"] = [
        "cuBLAS default heuristic reference; no exhaustive cuBLASLt algorithm/workspace search",
        "One process and GPU; repeat independent processes before generalizing",
        "GPU clocks are warmed but not locked; external load and clock drift remain possible",
        "CUDA-event batched timings include any GPU idle gaps caused by Python/ctypes dispatch",
        "Repeated same-buffer workload keeps data hot; no weight rotation or cold-cache measurement",
        "Full Y/cuBLAS agreement plus 128 FP64 CPU samples, not exhaustive FP64 verification",
    ]
    return result


def load_export(path):
    """Read gemm_kernel_compare's version=1 export without guessing launch geometry."""
    fields = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if "=" not in line:
            raise ValueError(f"manifest line {line_number} must contain key=value")
        name, value = line.split("=", 1)
        if not name or name in fields:
            raise ValueError(f"empty or duplicate manifest key at line {line_number}")
        fields[name] = value
    if fields.get("version") != "1":
        raise ValueError("expected a gemm_kernel_compare version=1 manifest")

    def integer(name, minimum=1, maximum=(1 << 31) - 1):
        try:
            value = int(fields[name])
        except (KeyError, ValueError) as error:
            raise ValueError(f"manifest requires integer {name}") from error
        return checked_int(value, name, minimum, maximum)

    shape = [integer(name, maximum=16384) for name in ("m", "n", "k")]
    if shape[1] % 16 or shape[2] % 16:
        raise ValueError("N and K must be multiples of 16")
    entry = fields.get("entry", "")
    if not entry or not entry.isascii() or "\0" in entry:
        raise ValueError("manifest requires a nonempty ASCII entry point")
    selected = {"shape": shape, "kernel_name": entry,
                "grid": [integer("grid_x"), integer("grid_y", maximum=65535), 1],
                "block": [integer("threads", maximum=1024), 1, 1],
                "shared_mem_bytes": integer("dyn_smem_bytes", minimum=0),
                "candidate": fields.get("candidate"), "sm_version": fields.get("sm_version")}
    digest = fields.get("ptx_sha256", "")
    ptx_path = path.with_suffix(".ptx")
    ptx = ptx_path.read_bytes()
    if hashlib.sha256(ptx).hexdigest() != digest:
        raise ValueError(f"PTX digest does not match manifest: {ptx_path}")
    if not ptx or b"\0" in ptx:
        raise ValueError("PTX must be nonempty text with no embedded NUL")
    selected["ptx_path"] = str(ptx_path.resolve())
    return selected, ptx, fields


def random_f16(shape, seed):
    """Vectorized, chunked copy of cuda_runtime::random_f16_bits (splitmix64)."""
    count = math.prod(shape)
    result = np.empty(count, dtype=np.uint16)
    for start in range(0, count, 1 << 20):
        stop = min(count, start + (1 << 20))
        z = np.arange(start, stop, dtype=np.uint64)
        z = z * np.uint64(0x9E3779B97F4A7C15) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
        z ^= z >> 31
        sign = (z >> 63).astype(np.uint16)
        exponent = (12 + ((z >> 40) & 3) % 3).astype(np.uint16)
        mantissa = (z & 0x3FF).astype(np.uint16)
        result[start:stop] = (sign << 15) | (exponent << 10) | mantissa
    return result.view(np.float16).reshape(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="M_N_K.manifest from gemm_kernel_compare export")
    parser.add_argument("--cublas", type=Path, help="exact cuBLAS shared library path")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, help="new JSON result file; existing files are never replaced")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=32, help="GEMMs per CUDA-event measurement")
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--order-offset", type=int, choices=(0, 1), default=0, help="which side goes first in round zero")
    parser.add_argument("--seed-a", type=int, default=123)
    parser.add_argument("--seed-b", type=int, default=987)
    parser.add_argument("--tolerance", type=float, default=0.002)
    args = parser.parse_args()
    try:
        checked_int(args.device, "device", minimum=0)
        checked_int(args.rounds, "rounds", minimum=2)
        checked_int(args.iterations, "iterations")
        checked_int(args.seed_a, "seed_a", minimum=0, maximum=(1 << 64) - 1)
        checked_int(args.seed_b, "seed_b", minimum=0, maximum=(1 << 64) - 1)
        if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 3:
            raise ValueError("warmup-seconds must be finite and at least 3")
        if not math.isfinite(args.tolerance) or not 0 < args.tolerance <= 0.002:
            raise ValueError("tolerance must be finite and in (0, 0.002]")
        if args.output and (args.output.exists() or not args.output.parent.is_dir()):
            raise ValueError("output must be a new file inside an existing directory")
        selected, ptx, manifest = load_export(args.manifest)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    result = run(args, selected, ptx, manifest)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(encoded)
    sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
