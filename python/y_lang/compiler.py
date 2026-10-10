import ctypes
import os
import sys
import json
import hashlib
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
    #
    # Since the merge 37651fb `python/` lives inside the cargo project, so
    # `pkg_root` IS the cargo root and it is the second candidate that
    # resolves; `pkg_root / "Y"` is the old outer checkout's layout.
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

def _library_signature(path: Path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _verify_loaded_library(lib, signature, path):
    """Detect dlopen returning an older, already mapped inode after a rebuild.

    The mapping is identified by its pathname and inode, never by its device
    number: on btrfs (and overlayfs) `/proc/self/maps` shows the superblock's
    device while `stat()` shows the subvolume's, so comparing devices refused
    every freshly built library on such a filesystem.
    """
    if not sys.platform.startswith("linux"):
        return
    address = ctypes.cast(lib.y_compile_to_ptx, ctypes.c_void_p).value
    expected = os.path.realpath(path)
    with open("/proc/self/maps", encoding="utf-8") as mappings:
        for line in mappings:
            fields = line.rstrip("\n").split(None, 5)
            start, end = (int(value, 16) for value in fields[0].split("-"))
            if start <= address < end:
                mapped_path = fields[5] if len(fields) > 5 else ""
                if (mapped_path.endswith(" (deleted)") or mapped_path != expected
                        or int(fields[4]) != signature[1]):
                    raise RuntimeError(
                        "Python has an older Y compiler library loaded. "
                        "Restart Python after rebuilding liby.so."
                    )
                return
    raise RuntimeError("Could not verify the loaded Y compiler library identity")


class YCompilerLib:
    _instance = None

    def __init__(self):
        lib_path = Path(_find_liby()).resolve()
        self.lib_path = str(lib_path)
        self._signature = _library_signature(lib_path)
        # Hash once at load, never on a warm cache lookup. Freeze the identity
        # of these bytes: a subsequent rebuild must not relabel a stale handle.
        digest = hashlib.sha256()
        with lib_path.open("rb") as binary:
            for chunk in iter(lambda: binary.read(1024 * 1024), b""):
                digest.update(chunk)
        self.compiler_identity = digest.hexdigest()
        self.lib = ctypes.CDLL(str(lib_path))
        if _library_signature(lib_path) != self._signature:
            raise RuntimeError("Y compiler library changed while loading; restart Python")
        _verify_loaded_library(self.lib, self._signature, lib_path)

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
        else:
            instance = cls._instance
            try:
                selected_path = Path(_find_liby()).resolve()
                unchanged = (
                    str(selected_path) == instance.lib_path
                    and _library_signature(selected_path) == instance._signature
                )
            except (OSError, RuntimeError):
                unchanged = False
            if not unchanged:
                # dlopen may reuse an old handle for the same pathname. Refuse
                # cache hits and compilation until a fresh process loads it.
                _JIT_CACHE.clear()
                raise RuntimeError(
                    "The selected Y compiler library changed after it was loaded. "
                    "Restart Python to load the rebuilt library or new Y_LIB_PATH."
                )
        return cls._instance

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


def _ptx_cache_context():
    # The C API reads this profile even for explicit SM targets: the remaining
    # hardware/tuning fields still affect lowering. With no readable profile,
    # a live probe may pick a different GPU, so caching cannot be safe yet.
    profile = Path.cwd() / ".ysu_hw_profile"
    try:
        profile_identity = hashlib.sha256(profile.read_bytes()).hexdigest()
    except OSError:
        return None
    settings = sorted(
        (name, value) for name, value in os.environ.items()
        if name.startswith(("Y_", "YSU_"))
        and name not in ("Y_LIB_PATH", "YSU_CACHE_DIR")
    )
    return [str(profile.resolve()), profile_identity, settings]


def compile_to_ptx(source: str, target_sm: str = "auto") -> str:
    """Compile PTX, caching by loaded compiler, target, and hardware profile.

    Restart Python after rebuilding the loaded library or changing Y_LIB_PATH.
    Compilations without a readable hardware profile bypass the cache until the
    compiler's hardware probe has recorded one.
    """
    lib = YCompilerLib.get_instance()
    context = _ptx_cache_context()
    cache_key = json.dumps(
        ["y-ptx-cache-v2", lib.compiler_identity, lib.lib_path, target_sm, context, source],
        ensure_ascii=False, separators=(",", ":"),
    )
    key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    if context is not None and key_hash in _JIT_CACHE:
        _CACHE_STATS["mem_hits"] += 1
        return _JIT_CACHE[key_hash]

    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{key_hash}.ptx"

    if context is not None and cache_file.exists():
        try:
            ptx_str = cache_file.read_text(encoding="utf-8")
            _JIT_CACHE[key_hash] = ptx_str
            _CACHE_STATS["disk_hits"] += 1
            return ptx_str
        except Exception:
            pass

    _CACHE_STATS["misses"] += 1
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

    # A probe can create/update the profile during compilation. Such a result
    # must not be stored under the hardware context observed before the probe.
    if context is None or context != _ptx_cache_context():
        return ptx_str

    # Persist to .ysu/cache disk cache
    try:
        tmp_file = cache_dir / f"{key_hash}.tmp"
        tmp_file.write_text(ptx_str, encoding="utf-8")
        tmp_file.replace(cache_file)
    except Exception:
        pass

    _JIT_CACHE[key_hash] = ptx_str
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


