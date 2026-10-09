"""Compile trusted Y source into CPU machine code and call it from Python.

The compiler supplies each function's signature. Scalar calls use checked
tagged values, so callers do not have to declare a ctypes function prototype.
Pointers must still address valid storage for the entire native call. Sessions
and their calls belong to the thread that created them; use a context manager
or close() there to release executable memory promptly.
"""

import ctypes
import json
import numbers
import operator
import os
import struct
import threading
import weakref
from typing import Any, Dict, Optional

from .compiler import _find_liby


class CPUJitError(RuntimeError):
    """A CPU compilation, lookup, or native-call error reported by Y."""


class _Value(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("bits", ctypes.c_uint64),
    ]


_TAGS = {
    "void": 0,
    "I8": 1,
    "U8": 2,
    "I16": 3,
    "U16": 4,
    "I32": 5,
    "U32": 6,
    "I64": 7,
    "U64": 8,
    "usize": 9,
    "bool": 10,
    "F32": 11,
    "F64": 12,
    "ptr": 13,
}
_INTEGER_TYPES = {
    "I8": (8, True),
    "U8": (8, False),
    "I16": (16, True),
    "U16": (16, False),
    "I32": (32, True),
    "U32": (32, False),
    "I64": (64, True),
    "U64": (64, False),
    "usize": (64, False),
}


def _integer(value: Any, width: int, signed: bool, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError("{} expects an integer, got bool".format(label))
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError("{} expects an integer".format(label)) from None
    minimum = -(1 << (width - 1)) if signed else 0
    maximum = (1 << (width - (1 if signed else 0))) - 1
    if not minimum <= value <= maximum:
        raise OverflowError("{} is outside [{}, {}]".format(label, minimum, maximum))
    return value & ((1 << width) - 1)


def _pointer(value: Any, label: str) -> int:
    if value is None:
        return 0
    if isinstance(value, ctypes.c_void_p):
        return value.value or 0
    if isinstance(value, int):
        return _integer(value, 64, False, label)
    # ctypes.cast also accepts Python strings and creates a temporary text
    # pointer. Require an explicit ctypes buffer/pointer for native memory.
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise TypeError("{} expects an address or ctypes pointer/array".format(label))
    try:
        return ctypes.cast(value, ctypes.c_void_p).value or 0
    except (TypeError, ctypes.ArgumentError):
        raise TypeError("{} expects an address or ctypes pointer/array".format(label)) from None


def _encode(type_name: str, value: Any, label: str) -> _Value:
    if type_name in _INTEGER_TYPES:
        bits = _integer(value, *_INTEGER_TYPES[type_name], label)
    elif type_name == "bool":
        if not isinstance(value, bool):
            raise TypeError("{} expects bool".format(label))
        bits = int(value)
    elif type_name in ("F32", "F64"):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError("{} expects a real number".format(label))
        fmt = "<f" if type_name == "F32" else "<d"
        bits = int.from_bytes(struct.pack(fmt, float(value)), "little")
    elif type_name == "ptr":
        bits = _pointer(value, label)
    else:
        raise TypeError("dynamic CPU JIT calls do not support {}".format(type_name))
    return _Value(_TAGS[type_name], 0, bits)


def _decode(type_name: str, value: _Value) -> Any:
    if value.kind != _TAGS.get(type_name) or value.reserved != 0:
        raise CPUJitError("CPU JIT returned an incompatible tagged value")
    bits = value.bits
    if type_name == "void":
        return None
    if type_name in _INTEGER_TYPES:
        width, signed = _INTEGER_TYPES[type_name]
        bits &= (1 << width) - 1
        return bits - (1 << width) if signed and bits & (1 << (width - 1)) else bits
    if type_name == "bool":
        return bool(bits)
    if type_name in ("F32", "F64"):
        width = 4 if type_name == "F32" else 8
        return struct.unpack("<f" if width == 4 else "<d", bits.to_bytes(width, "little"))[0]
    if type_name == "ptr":
        return ctypes.c_void_p(bits)
    raise CPUJitError("unsupported CPU JIT result type {}".format(type_name))


def _release(library: Any, handle: int, creator_thread: int) -> None:
    # The C API's session is thread bound. Explicit close/context management
    # handles normal ownership; finalization must not violate that contract.
    if threading.get_ident() == creator_thread:
        library.y_cpu_jit_free(handle)


class CPUJit:
    """An in-process CPU JIT session for trusted Y source.

    Example::

        with CPUJit("fn square(x: I64) -> I64 { return x * x; }") as jit:
            assert jit.call("square", 12) == 144

    Pointer arguments accept ctypes pointers, arrays, byref objects, c_void_p,
    integer addresses, or None. Pointer results are returned as c_void_p.
    Dynamic calls support scalar and pointer ABIs; aggregate results are
    reported by signature() and refused by call().

    instrument=True records outcomes during explicit calls. branch_profile()
    returns their counts, and recompile_profiled() creates a separate optimized
    session without counters. Neither compilation executes the source.
    """

    def __init__(self, source: str, opt_level: int = 3, library_path: Optional[str] = None,
                 *, instrument: bool = False):
        if not isinstance(source, str):
            raise TypeError("CPU JIT source must be a string")
        if "\0" in source:
            raise ValueError("CPU JIT source cannot contain NUL")
        if isinstance(opt_level, bool) or not isinstance(opt_level, int) or not 0 <= opt_level <= 3:
            raise ValueError("opt_level must be 0, 1, 2, or 3")
        if not isinstance(instrument, bool):
            raise TypeError("instrument must be bool")
        self._creator_thread = threading.get_ident()
        self._handle = None
        self._signatures = {}  # type: Dict[str, Dict[str, Any]]
        self._source = source
        self._opt_level = opt_level
        self.library_path = os.fspath(library_path) if library_path is not None else _find_liby()
        self._library = ctypes.CDLL(self.library_path)
        self._declare_api()
        error = ctypes.c_void_p()
        compile_function = (self._library.y_cpu_jit_compile_instrumented if instrument
                            else self._library.y_cpu_jit_compile)
        handle = compile_function(source.encode("utf-8"), opt_level, ctypes.byref(error))
        if not handle:
            self._raise(error, "CPU compilation returned a null session")
        if error.value:
            self._library.y_cpu_jit_free(handle)
            self._raise(error, "CPU compilation failed")
        self._own_handle(handle)

    def _own_handle(self, handle: int) -> None:
        try:
            self._finalizer = weakref.finalize(self, _release, self._library, handle, self._creator_thread)
        except BaseException:
            self._library.y_cpu_jit_free(handle)
            raise
        self._handle = handle

    @classmethod
    def _adopt_profiled(cls, owner: "CPUJit", handle: int) -> "CPUJit":
        """Take an already compiled handle without compiling or training again."""
        try:
            result = cls.__new__(cls)
            result._creator_thread = threading.get_ident()
            result._handle = None
            result._signatures = {}
            result._source = owner._source
            result._opt_level = owner._opt_level
            result.library_path = owner.library_path
            result._library = owner._library
        except BaseException:
            owner._library.y_cpu_jit_free(handle)
            raise
        result._own_handle(handle)
        return result

    def _declare_api(self) -> None:
        error_pointer = ctypes.POINTER(ctypes.c_void_p)
        declarations = {
            "y_cpu_jit_compile": ([ctypes.c_char_p, ctypes.c_uint32, error_pointer], ctypes.c_void_p),
            "y_cpu_jit_compile_instrumented": ([ctypes.c_char_p, ctypes.c_uint32, error_pointer], ctypes.c_void_p),
            "y_cpu_jit_compile_profiled": ([ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p,
                                             error_pointer], ctypes.c_void_p),
            "y_cpu_jit_branch_profile": ([ctypes.c_void_p, error_pointer], ctypes.c_void_p),
            "y_cpu_jit_compile_timings": ([ctypes.c_void_p, error_pointer], ctypes.c_void_p),
            "y_cpu_jit_optimization_timings": ([ctypes.c_void_p, error_pointer], ctypes.c_void_p),
            "y_cpu_jit_materialization_timings": ([ctypes.c_void_p, error_pointer], ctypes.c_void_p),
            "y_cpu_jit_signature": ([ctypes.c_void_p, ctypes.c_char_p, error_pointer], ctypes.c_void_p),
            "y_cpu_jit_call": ([ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_Value), ctypes.c_size_t,
                                 ctypes.POINTER(_Value), error_pointer], ctypes.c_int32),
            "y_cpu_jit_free": ([ctypes.c_void_p], None),
            "y_free_string": ([ctypes.c_void_p], None),
        }
        try:
            for name, (parameters, result) in declarations.items():
                function = getattr(self._library, name)
                function.argtypes = parameters
                function.restype = result
        except AttributeError as error:
            raise CPUJitError("liby is missing CPU JIT APIs; rebuild it with cargo build --release") from error

    def _raise(self, error: ctypes.c_void_p, default: str) -> None:
        if error.value:
            try:
                message = ctypes.string_at(error.value).decode("utf-8", errors="replace")
            finally:
                self._library.y_free_string(error)
            raise CPUJitError(message)
        raise CPUJitError(default)

    def _check(self) -> None:
        if threading.get_ident() != self._creator_thread:
            raise RuntimeError("CPU JIT sessions must be used on their creating thread")
        if self._handle is None:
            raise RuntimeError("CPU JIT session is closed")

    @staticmethod
    def _name(name: str) -> bytes:
        if not isinstance(name, str):
            raise TypeError("CPU JIT function name must be a string")
        if "\0" in name:
            raise ValueError("CPU JIT function name cannot contain NUL")
        return name.encode("utf-8")

    def _signature(self, name: str) -> Dict[str, Any]:
        self._check()
        encoded = self._name(name)
        if name not in self._signatures:
            error = ctypes.c_void_p()
            result = self._library.y_cpu_jit_signature(self._handle, encoded, ctypes.byref(error))
            if not result:
                self._raise(error, "CPU signature lookup returned null")
            try:
                if error.value:
                    self._raise(error, "CPU signature lookup failed")
                self._signatures[name] = json.loads(ctypes.string_at(result).decode("utf-8"))
            finally:
                self._library.y_free_string(result)
        return self._signatures[name]

    def signature(self, name: str) -> Dict[str, Any]:
        """Return source-derived ABI metadata for a compiled function."""
        signature = self._signature(name)
        return dict(signature, parameters=list(signature["parameters"]))

    def call(self, name: str, *arguments: Any) -> Any:
        """Call a function using its checked scalar/pointer signature."""
        signature = self._signature(name)
        if not signature["dynamic_call"]:
            raise TypeError("dynamic CPU JIT calls do not support the aggregate signature of {}".format(name))
        parameters = signature["parameters"]
        if len(arguments) != len(parameters):
            raise TypeError("{} expects {} arguments, got {}".format(name, len(parameters), len(arguments)))
        values = (_Value * len(arguments))(*[
            _encode(type_name, value, "{} argument {} ({})".format(name, index + 1, type_name))
            for index, (type_name, value) in enumerate(zip(parameters, arguments))
        ])
        result = _Value()
        error = ctypes.c_void_p()
        status = self._library.y_cpu_jit_call(self._handle, self._name(name), values, len(values),
                                              ctypes.byref(result), ctypes.byref(error))
        if status != 0 or error.value:
            self._raise(error, "CPU JIT native call failed")
        return _decode(signature["return_type"], result)

    __call__ = call

    def function(self, name: str) -> "CPUJitFunction":
        """Return a callable that keeps its JIT session alive."""
        self._signature(name)
        return CPUJitFunction(self, name)

    def compile_timings(self) -> Dict[str, int]:
        """Return owned compilation phase timings in integer nanoseconds.

        Phase keys end in _ns and sum to total_ns (excluding total itself).
        Training, calls and this lookup are excluded. LLVM optimization includes
        per-pass verification; materialization includes codegen, linking and lookup.
        """
        self._check()
        error = ctypes.c_void_p()
        result = self._library.y_cpu_jit_compile_timings(self._handle, ctypes.byref(error))
        if not result:
            self._raise(error, "CPU compilation timings lookup returned null")
        try:
            if error.value:
                self._raise(error, "CPU compilation timings lookup failed")
            return json.loads(ctypes.string_at(result).decode("utf-8"))
        finally:
            self._library.y_free_string(result)

    def optimization_timings(self) -> Dict[str, int]:
        """Return owned optimization subphase timings in integer nanoseconds.

        pipeline_ns, profile_selection_ns and other_ns sum exactly to total_ns,
        which equals compile_timings()["optimization_ns"]. Training, calls and
        this lookup are excluded. Pipeline time includes per-pass verification.
        """
        self._check()
        error = ctypes.c_void_p()
        result = self._library.y_cpu_jit_optimization_timings(self._handle, ctypes.byref(error))
        if not result:
            self._raise(error, "CPU optimization timings lookup returned null")
        try:
            if error.value:
                self._raise(error, "CPU optimization timings lookup failed")
            return json.loads(ctypes.string_at(result).decode("utf-8"))
        finally:
            self._library.y_free_string(result)

    def materialization_timings(self) -> Dict[str, Any]:
        """Return an owned snapshot of eager materialization timings and metadata.

        submission_ns, first_lookup_ns, remaining_function_lookups_ns,
        profile_lookup_ns and other_ns sum to total_ns, which equals
        compile_timings()["materialization_ns"]. Optional before/after-object
        children partition first_lookup_ns when one observed object event can
        be localized there; they are otherwise None. Children and object/lookup
        metadata are excluded from the primary sum. Training, execution, host
        lookups and this getter do not update this compilation snapshot.
        """
        self._check()
        error = ctypes.c_void_p()
        result = self._library.y_cpu_jit_materialization_timings(self._handle, ctypes.byref(error))
        if not result:
            self._raise(error, "CPU materialization timings lookup returned null")
        try:
            if error.value:
                self._raise(error, "CPU materialization timings lookup failed")
            return json.loads(ctypes.string_at(result).decode("utf-8"))
        finally:
            self._library.y_free_string(result)

    def branch_profile(self) -> Dict[str, Any]:
        """Return an owned snapshot of measured original branch outcomes.

        Counters wrap after 2**64 observations per outcome. Each counter is
        read atomically; concurrent calls can advance counts during a snapshot.
        """
        self._check()
        error = ctypes.c_void_p()
        result = self._library.y_cpu_jit_branch_profile(self._handle, ctypes.byref(error))
        if not result:
            self._raise(error, "CPU branch profile snapshot returned null")
        try:
            if error.value:
                self._raise(error, "CPU branch profile snapshot failed")
            return json.loads(ctypes.string_at(result).decode("utf-8"))
        finally:
            self._library.y_free_string(result)

    def recompile_profiled(self) -> "CPUJit":
        """Create an independent optimized session from this session's counts.

        This instrumented session and its callables remain live. The new
        session uses the same source, optimization level and loaded library,
        has no counters, and must be closed independently.
        """
        self._check()
        error = ctypes.c_void_p()
        handle = self._library.y_cpu_jit_compile_profiled(
            self._source.encode("utf-8"), self._opt_level, self._handle, ctypes.byref(error))
        if not handle:
            self._raise(error, "CPU profile-use compilation returned a null session")
        if error.value:
            self._library.y_cpu_jit_free(handle)
            self._raise(error, "CPU profile-use compilation failed")
        return self._adopt_profiled(self, handle)

    @property
    def closed(self) -> bool:
        return self._handle is None

    def close(self) -> None:
        """Release executable memory on the session's creating thread."""
        if threading.get_ident() != self._creator_thread:
            raise RuntimeError("CPU JIT sessions must be closed on their creating thread")
        if self._handle is not None:
            self._finalizer()
            self._handle = None
            self._signatures.clear()

    def __enter__(self) -> "CPUJit":
        self._check()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class CPUJitFunction:
    """A named dynamic callable with a strong reference to its session."""

    def __init__(self, owner: CPUJit, name: str):
        self._owner = owner
        self.name = name

    @property
    def signature(self) -> Dict[str, Any]:
        return self._owner.signature(self.name)

    def __call__(self, *arguments: Any) -> Any:
        return self._owner.call(self.name, *arguments)
