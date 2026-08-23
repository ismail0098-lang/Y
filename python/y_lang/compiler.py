import ctypes
import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

def _find_liby() -> str:
    # 1. Environment variable override
    if "Y_LIB_PATH" in os.environ and os.path.exists(os.environ["Y_LIB_PATH"]):
        return os.environ["Y_LIB_PATH"]

    # 2. Local target builds.
    #
    # The cargo project is `Y/`, NOT this package's grandparent - so the old
    # single candidate (`<grandparent>/target/release/liby.so`) pointed at a
    # directory `cargo build --release` never writes. It did not fail loudly:
    # a stale `liby.so` from an abandoned build layout was sitting there, and
    # the whole Python package silently ran against a compiler 25 days older
    # than `src/`. Missing symbols would raise; a stale symbol just answers.
    #
    # Cargo's own root is therefore searched FIRST, and when both exist the
    # newer one wins so an old artifact cannot shadow a fresh build.
    curr_dir = Path(__file__).resolve().parent
    pkg_root = curr_dir.parent.parent

    found = []
    for root in [pkg_root / "Y", pkg_root]:
        for build_type in ["release", "debug"]:
            candidate = root / "target" / build_type / "liby.so"
            if candidate.exists():
                found.append(candidate)
    if found:
        return str(max(found, key=lambda c: c.stat().st_mtime))

    # 3. Package directory fallback
    pkg_lib = curr_dir / "liby.so"
    if pkg_lib.exists():
        return str(pkg_lib)

    raise RuntimeError(
        "Could not locate liby.so shared library. Please build Y-compiler with `cargo build --release` "
        "or set Y_LIB_PATH environment variable."
    )

class YCompilerLib:
    _instance = None

    def __init__(self):
        lib_path = _find_liby()
        self.lib_path = lib_path
        self.lib = ctypes.CDLL(lib_path)

        # void* y_compile_to_ptx(const char* source, const char* target_sm, char** error_out)
        self.lib.y_compile_to_ptx.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.y_compile_to_ptx.restype = ctypes.c_void_p

        # void* y_autotune_search_space_json(uint32_t m, uint32_t n, uint32_t k, bool is_fp8)
        #
        # `is_fp8` was MISSING from this list. ctypes only fills the registers
        # it is told about, so the callee read whatever happened to be in the
        # fourth argument register - observed flipping between True and False
        # across calls in a single process. Undefined behaviour on an ABI
        # boundary, and it decided which precision's search space came back.
        self.lib.y_autotune_search_space_json.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool
        ]
        self.lib.y_autotune_search_space_json.restype = ctypes.c_void_p

        # void* y_autotune_select_config_json(uint32_t m, uint32_t n, uint32_t k, bool is_fp8)
        self.lib.y_autotune_select_config_json.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool
        ]
        self.lib.y_autotune_select_config_json.restype = ctypes.c_void_p

        # void y_free_string(char* s)
        self.lib.y_free_string.argtypes = [ctypes.c_void_p]
        self.lib.y_free_string.restype = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

import hashlib

_JIT_CACHE: Dict[str, str] = {}
_CACHE_STATS = {
    "mem_hits": 0,
    "disk_hits": 0,
    "misses": 0,
}

def get_cache_dir() -> Path:
    if "YSU_CACHE_DIR" in os.environ:
        path = Path(os.environ["YSU_CACHE_DIR"]).resolve()
    else:
        path = Path.cwd() / ".ysu" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_cache_stats() -> Dict[str, Any]:
    cache_dir = get_cache_dir()
    cached_files = list(cache_dir.glob("*.ptx")) if cache_dir.exists() else []
    return {
        "mem_hits": _CACHE_STATS["mem_hits"],
        "disk_hits": _CACHE_STATS["disk_hits"],
        "misses": _CACHE_STATS["misses"],
        "cached_files_count": len(cached_files),
        "cache_dir": str(cache_dir),
    }

def clear_disk_cache() -> int:
    global _JIT_CACHE, _CACHE_STATS
    _JIT_CACHE.clear()
    _CACHE_STATS["mem_hits"] = 0
    _CACHE_STATS["disk_hits"] = 0
    _CACHE_STATS["misses"] = 0

    cache_dir = get_cache_dir()
    count = 0
    if cache_dir.exists():
        for ptx_file in cache_dir.glob("*.ptx"):
            try:
                ptx_file.unlink()
                count += 1
            except Exception:
                pass
    return count

def compile_to_ptx(source: str, target_sm: str = "auto") -> str:
    """Compiles Y-lang source string into NVIDIA PTX assembly string with disk caching."""
    cache_key = f"{target_sm}:{source}"
    if cache_key in _JIT_CACHE:
        _CACHE_STATS["mem_hits"] += 1
        return _JIT_CACHE[cache_key]

    key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{key_hash}.ptx"

    if cache_file.exists():
        try:
            ptx_str = cache_file.read_text(encoding="utf-8")
            _JIT_CACHE[cache_key] = ptx_str
            _CACHE_STATS["disk_hits"] += 1
            return ptx_str
        except Exception:
            pass

    _CACHE_STATS["misses"] += 1
    lib = YCompilerLib.get_instance()
    error_out = ctypes.c_char_p()

    ptx_ptr = lib.lib.y_compile_to_ptx(
        source.encode("utf-8"),
        target_sm.encode("utf-8"),
        ctypes.byref(error_out)
    )

    if error_out.value:
        err_msg = error_out.value.decode("utf-8")
        lib.lib.y_free_string(error_out)
        raise ValueError(f"Y Compilation Error:\n{err_msg}")

    if not ptx_ptr:
        raise ValueError("Y Compilation produced null PTX pointer")

    ptx_str = ctypes.cast(ptx_ptr, ctypes.c_char_p).value.decode("utf-8")
    lib.lib.y_free_string(ptx_ptr)

    # Persist to .ysu/cache disk cache
    try:
        tmp_file = cache_dir / f"{key_hash}.tmp"
        tmp_file.write_text(ptx_str, encoding="utf-8")
        tmp_file.replace(cache_file)
    except Exception:
        pass

    _JIT_CACHE[cache_key] = ptx_str
    return ptx_str

def generate_autotune_search_space(m: int, n: int, k: int, is_fp8: bool = False) -> List[Dict[str, Any]]:
    """Generates candidate CTA tile autotuning configurations for target dimensions M, N, K."""
    lib = YCompilerLib.get_instance()
    json_ptr = lib.lib.y_autotune_search_space_json(
        ctypes.c_uint32(m), ctypes.c_uint32(n), ctypes.c_uint32(k), ctypes.c_bool(is_fp8)
    )
    if not json_ptr:
        return []

    json_str = ctypes.cast(json_ptr, ctypes.c_char_p).value.decode("utf-8")
    lib.lib.y_free_string(json_ptr)
    return json.loads(json_str)

def select_autotune_config(m: int, n: int, k: int, is_fp8: bool = False) -> Dict[str, Any]:
    """The analytic model's single pick for (M, N, K) - the compiler's own answer.

    This is what `--emit-ptx` uses when it has no on-device measurement, and it
    is the right thing to fall back to when measurement cannot separate the
    candidates. Returns {} if the library call fails.
    """
    lib = YCompilerLib.get_instance()
    json_ptr = lib.lib.y_autotune_select_config_json(
        ctypes.c_uint32(m), ctypes.c_uint32(n), ctypes.c_uint32(k), ctypes.c_bool(is_fp8)
    )
    if not json_ptr:
        return {}

    json_str = ctypes.cast(json_ptr, ctypes.c_char_p).value.decode("utf-8")
    lib.lib.y_free_string(json_ptr)
    return json.loads(json_str)

def is_interpreter_enabled() -> bool:
    """Checks if CPU Interpreter / Debug Mode is enabled via Y_INTERPRETER=1 or Y_DEBUG=1 environment variables."""
    return os.environ.get("Y_INTERPRETER", "0").lower() in ("1", "true", "yes") or os.environ.get("Y_DEBUG", "0").lower() in ("1", "true", "yes")

class JITFunction:
    """Wrapper class for JIT-compiled Y functions supporting Triton-style grid launcher syntax kernel[grid](*args)."""
    def __init__(self, fn, target_sm: str = "auto"):
        self.fn = fn
        self.target_sm = target_sm
        self.source = fn() if callable(fn) and not hasattr(fn, "__code__") else fn
        if hasattr(fn, "__name__"):
            self.__name__ = fn.__name__
        else:
            self.__name__ = "jit_kernel"
        self._torch_kernel = None

    @property
    def ptx(self) -> str:
        src_str = self.source() if callable(self.source) else str(self.source)
        return compile_to_ptx(src_str, target_sm=self.target_sm)

    def __getitem__(self, grid):
        """Allows launch syntax: kernel[grid](*args, stream=stream)."""
        if not isinstance(grid, tuple):
            grid = (grid, 1, 1)
        elif len(grid) == 1:
            grid = (grid[0], 1, 1)
        elif len(grid) == 2:
            grid = (grid[0], grid[1], 1)

        def launcher(*args, block=(256, 1, 1), stream=None):
            from .torch_interop import TorchKernel
            if self._torch_kernel is None:
                src_str = self.source() if callable(self.source) else str(self.source)
                self._torch_kernel = TorchKernel(src_str, kernel_name=self.__name__, target_sm=self.target_sm)
            return self._torch_kernel.launch(grid, block, list(args), stream=stream)

        return launcher

    def __call__(self, *args, **kwargs):
        if self._torch_kernel is None:
            from .torch_interop import TorchKernel
            src_str = self.source() if callable(self.source) else str(self.source)
            self._torch_kernel = TorchKernel(src_str, kernel_name=self.__name__, target_sm=self.target_sm)
        # Default grid (1, 1, 1) launch
        return self._torch_kernel.launch((1, 1, 1), (256, 1, 1), list(args))

def jit(fn=None, target_sm="auto"):
    """Decorator to JIT-compile a Y-lang kernel into PTX assembly or CPU interpreter with launcher support."""
    def decorator(func):
        return JITFunction(func, target_sm=target_sm)

    if fn is None:
        return decorator
    return decorator(fn)


