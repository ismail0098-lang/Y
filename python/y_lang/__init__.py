from .compiler import (
    compile_to_ptx,
    generate_autotune_search_space,
    jit,
    clear_disk_cache,
    get_cache_stats,
    is_interpreter_enabled,
)
from .torch_interop import TorchKernel
from .autotune_decorator import autotune, AutotuneConfig, Config, heuristics
from .interpreter import CPUInterpreter, debug_interpret
from .cpu_jit import CPUJit, CPUJitError


def __getattr__(name):
    # CPU compilation needs neither torch nor cupy. Load the existing tensor
    # integrations only when a caller asks for those exports.
    if name == "y_inductor":
        from .inductor import y_inductor
        globals()[name] = y_inductor
        return y_inductor
    if name == "ops":
        import importlib
        ops = importlib.import_module(".ops", __name__)
        globals()[name] = ops
        return ops
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

__version__ = "1.1.0"
__all__ = [
    "compile_to_ptx",
    "generate_autotune_search_space",
    "jit",
    "clear_disk_cache",
    "get_cache_stats",
    "is_interpreter_enabled",
    "TorchKernel",
    "autotune",
    "AutotuneConfig",
    "Config",
    "heuristics",
    "y_inductor",
    "CPUInterpreter",
    "debug_interpret",
    "CPUJit",
    "CPUJitError",
    "ops",
]


