"""Checked, synchronous launch of the pinned exact_pv cubin on torch tensors.

These checks establish the arithmetic, layout and ownership preconditions of
ExactPvExact.v for a positive contraction. They do not prove the transcription
of that model into operational PTX, the validator, or NVIDIA hardware. CUDA
and torch metadata are trusted. Ordinary CUDA allocations with no physically
aliased mappings are required. Callers must not concurrently mutate inputs,
free storage, switch contexts or unload the module during this call. Unknown
allocation encodings (including pools unsupported by the range query) refuse.
"""
from dataclasses import dataclass
import ctypes

if __package__:
    from .exact_pv_artifact import ArtifactError, ValidatedExactPv
else:
    from exact_pv_artifact import ArtifactError, ValidatedExactPv


I32MAX = (1 << 31) - 1
I64MAX = (1 << 63) - 1
U64MAX = (1 << 64) - 1
TERM_BOUND = ((1 << 32) - 1) * 128


@dataclass(frozen=True)
class ExactPvShape:
    batch: int
    queries: int
    keys: int
    channels: int

    def __post_init__(self):
        values = (self.batch, self.queries, self.keys, self.channels)
        if any(type(value) is not int or not 1 <= value <= I32MAX for value in values):
            raise ArtifactError("checked exact_pv requires positive signed-i32 dimensions")
        if self.keys * TERM_BOUND > I64MAX:
            raise ArtifactError("exact_pv full-domain accumulator can overflow signed i64")
        if max(self.np, self.nv, self.no) > I32MAX:
            raise ArtifactError("exact_pv array extents exceed the signed-i32 index licence")

    @property
    def np(self):
        return self.batch * self.queries * self.keys

    @property
    def nv(self):
        return self.batch * self.keys * self.channels

    @property
    def no(self):
        return self.batch * self.queries * self.channels

    @property
    def grid(self):
        return (self.queries, self.batch, 1)

    @property
    def block(self):
        return (self.channels, 1, 1)

    @property
    def scalar_args(self):
        return (self.keys, self.channels, self.queries, self.np, self.nv, self.no)


def _check(result, operation):
    if result != 0:
        raise ArtifactError(f"{operation} failed (CUresult {result})")


def _current(cuda):
    value = ctypes.c_void_p()
    _check(cuda.cuCtxGetCurrent(ctypes.byref(value)), "cuCtxGetCurrent")
    if not value.value:
        raise ArtifactError("checked exact_pv requires a current CUDA context")
    return value.value


def _configure_driver(cuda):
    if not isinstance(cuda, ctypes.CDLL):
        return  # Recording drivers exercise the same calls without a library.
    ptr, integer, uint = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
    pptr, pint = ctypes.POINTER(ptr), ctypes.POINTER(integer)
    signatures = {
        "cuCtxGetCurrent": [pptr], "cuCtxGetDevice": [pint], "cuCtxSynchronize": [],
        "cuDeviceGet": [pint, integer], "cuDeviceGetAttribute": [pint, integer, integer],
        "cuModuleLoadData": [pptr, ptr], "cuModuleGetFunction": [pptr, ptr, ctypes.c_char_p],
        "cuModuleUnload": [ptr], "cuFuncGetAttribute": [pint, integer, ptr],
        "cuPointerGetAttribute": [ptr, integer, ctypes.c_uint64],
        "cuMemGetAddressRange_v2": [ctypes.POINTER(ctypes.c_uint64),
                                    ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint64],
        "cuLaunchKernel": [ptr] + [uint] * 7 + [ptr, pptr, pptr],
    }
    try:
        for name, arguments in signatures.items():
            function = getattr(cuda, name)
            function.argtypes, function.restype = arguments, integer
    except AttributeError as error:
        raise ArtifactError(f"checked exact_pv driver interface unavailable: {error}") from error


def _span(pointer, nbytes, alignment, allocation_base, allocation_size):
    """Check a half-open live byte range without wrapping address arithmetic."""
    if (type(pointer) is not int or not 0 < pointer <= U64MAX
            or pointer % alignment or not 0 < nbytes <= U64MAX - pointer):
        raise ArtifactError("exact_pv buffer is null, misaligned, or wraps its address range")
    if (not 0 < allocation_base <= U64MAX
            or not 0 < allocation_size <= U64MAX - allocation_base
            or pointer < allocation_base
            or pointer + nbytes > allocation_base + allocation_size):
        raise ArtifactError("exact_pv buffer exceeds its live CUDA allocation")
    return (pointer, pointer + nbytes)


def _require_disjoint_output(p, v, out):
    if any(out[0] < src[1] and src[0] < out[1] for src in (p, v)):
        raise ArtifactError("exact_pv output overlaps an input byte range")


class CheckedExactPv:
    """Retain the module identity and derive all geometry/ABI arguments.

    Construct with ``CheckedExactPv.load(artifact, cuda)``. The public launch
    accepts actual contiguous CUDA torch tensors, never raw pointers or
    caller-supplied launch geometry. P uses int32/uint32 storage interpreted
    as unsigned bits, V uses int8, and output uses int64.
    """

    def __init__(self):
        raise TypeError("use CheckedExactPv.load")

    @classmethod
    def load(cls, artifact, cuda):
        if not isinstance(artifact, ValidatedExactPv):
            raise ArtifactError("checked exact_pv requires a bound validation artifact")
        _configure_driver(cuda)
        instance = cls.__new__(cls)
        instance._cuda = cuda
        instance._context = _current(cuda)
        instance._module = artifact.load(cuda)
        try:
            instance._require_current()
            device = ctypes.c_int()
            _check(cuda.cuCtxGetDevice(ctypes.byref(device)), "cuCtxGetDevice")
            instance._device = device.value
            instance._function = ctypes.c_void_p()
            _check(cuda.cuModuleGetFunction(ctypes.byref(instance._function),
                                           instance._module, b"exact_pv"),
                   "cuModuleGetFunction(exact_pv)")
            if not instance._function.value:
                raise ArtifactError("CUDA did not resolve exact_pv")
            instance._limits = []
            # CUDA driver enum 1..7: block threads, block XYZ, grid XYZ.
            for attribute in range(1, 8):
                value = ctypes.c_int()
                _check(cuda.cuDeviceGetAttribute(ctypes.byref(value), ctypes.c_int(attribute),
                                                device), "CUDA launch limits")
                if value.value <= 0:
                    raise ArtifactError("CUDA returned an invalid launch limit")
                instance._limits.append(value.value)
            value = ctypes.c_int()
            _check(cuda.cuFuncGetAttribute(ctypes.byref(value), ctypes.c_int(0),
                                          instance._function), "exact_pv function thread limit")
            if value.value <= 0:
                raise ArtifactError("CUDA returned an invalid function thread limit")
            instance._function_threads = value.value
            # Fail closed when required ownership/allocation queries are absent.
            for name in ("cuPointerGetAttribute", "cuMemGetAddressRange_v2"):
                if not callable(getattr(cuda, name, None)):
                    raise ArtifactError(f"checked exact_pv requires {name}")
            instance._require_current()
        except Exception as error:
            # No launch has occurred. Clean up only in the owning context.
            try:
                if _current(cuda) == instance._context:
                    cuda.cuModuleUnload(instance._module)
            except Exception:
                pass
            if isinstance(error, ArtifactError):
                raise
            raise ArtifactError(f"checked exact_pv driver interface unavailable: {error}") from error
        return instance

    def _require_current(self):
        if not self._module or not self._module.value:
            raise ArtifactError("checked exact_pv module is closed")
        if _current(self._cuda) != self._context:
            raise ArtifactError("checked exact_pv CUDA context changed")

    def close(self):
        if self._module and self._module.value:
            self._require_current()
            _check(self._cuda.cuCtxSynchronize(), "exact_pv close synchronization")
            _check(self._cuda.cuModuleUnload(self._module), "cuModuleUnload(exact_pv)")
            self._module = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def check_shape(self, shape):
        if type(shape) is not ExactPvShape:
            raise ArtifactError("checked exact_pv requires an ExactPvShape")
        # Reconstruct the shape so even malicious mutation of a frozen Python
        # instance cannot bypass validation before scalar marshalling.
        shape = ExactPvShape(shape.batch, shape.queries, shape.keys, shape.channels)
        threads, bx, by, bz, gx, gy, gz = self._limits
        if (shape.channels > min(threads, bx, self._function_threads)
                or min(by, bz) < 1
                or any(want > limit for want, limit in zip(shape.grid, (gx, gy, gz)))):
            raise ArtifactError("exact_pv shape exceeds device or function launch limits")
        return shape

    def _tensor_span(self, tensor, torch, dtypes, dimensions, count, width):
        if (type(tensor) is not torch.Tensor or tensor.dtype not in dtypes
                or tensor.layout != torch.strided or tensor.is_neg() or tensor.is_conj()
                or tensor.device.type != "cuda" or not tensor.is_contiguous()
                or tuple(tensor.shape) != dimensions or tensor.numel() != count
                or tensor.element_size() != width):
            raise ArtifactError("exact_pv requires contiguous CUDA tensors with the licensed shape and dtype")
        ordinal = tensor.device.index
        if type(ordinal) is not int or ordinal < 0:
            raise ArtifactError("exact_pv tensor has no CUDA device identity")
        device = ctypes.c_int()
        _check(self._cuda.cuDeviceGet(ctypes.byref(device), ctypes.c_int(ordinal)), "tensor CUDA device")
        if device.value != self._device:
            raise ArtifactError("exact_pv tensor belongs to a different CUDA device")
        pointer = tensor.data_ptr()
        if type(pointer) is not int or not 0 < pointer <= U64MAX:
            raise ArtifactError("exact_pv tensor has an invalid device pointer")
        context = ctypes.c_void_p()
        _check(self._cuda.cuPointerGetAttribute(ctypes.byref(context), ctypes.c_int(1),
                                               ctypes.c_uint64(pointer)), "tensor CUDA context")
        if context.value != self._context:
            raise ArtifactError("exact_pv tensor belongs to a different CUDA context")
        base, size = ctypes.c_uint64(), ctypes.c_size_t()
        _check(self._cuda.cuMemGetAddressRange_v2(ctypes.byref(base), ctypes.byref(size),
                                                ctypes.c_uint64(pointer)), "tensor CUDA allocation")
        return _span(pointer, count * width, width, base.value, size.value)

    def launch(self, p, v, out, shape):
        """On success, complete one launch while keeping all tensors alive.

        Any CUDA error returns no checked result. Cleanup after an error
        relies on driver semantics, outside the formal arithmetic proof.
        """
        import torch
        shape = self.check_shape(shape)
        self._require_current()
        _check(self._cuda.cuCtxSynchronize(), "exact_pv input synchronization")
        ptypes = (torch.int32,)
        if hasattr(torch, "uint32"):
            ptypes += (torch.uint32,)
        ps = self._tensor_span(p, torch, ptypes, (shape.batch, shape.queries, shape.keys), shape.np, 4)
        vs = self._tensor_span(v, torch, (torch.int8,), (shape.batch, shape.keys, shape.channels), shape.nv, 1)
        os = self._tensor_span(out, torch, (torch.int64,), (shape.batch, shape.queries, shape.channels), shape.no, 8)
        _require_disjoint_output(ps, vs, os)
        self._require_current()
        # Kernel parameters are three u64 pointers followed by six actual i32
        # values; storing every parameter in a u64 would obscure ABI errors.
        args = [ctypes.c_uint64(span[0]) for span in (ps, vs, os)]
        args += [ctypes.c_int32(value) for value in shape.scalar_args]
        pointers = (ctypes.c_void_p * len(args))(
            *(ctypes.cast(ctypes.byref(arg), ctypes.c_void_p) for arg in args))
        _check(self._cuda.cuLaunchKernel(self._function,
                                        *(ctypes.c_uint(value) for value in shape.grid + shape.block),
                                        ctypes.c_uint(0), None, pointers, None), "checked exact_pv launch")
        _check(self._cuda.cuCtxSynchronize(), "exact_pv output synchronization")
        return out
