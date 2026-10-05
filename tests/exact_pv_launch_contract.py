"""Launch-contract regressions using tensor metadata and a recording driver.

No synthetic execution claims SASS semantics or measured GPU performance.
"""
import ctypes
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from verification_unittest import main as verification_main
from verified_exact_pv_artifact import CUBIN, RecordingCuda, write_bundle

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "ptxas_tval"))
import exact_pv_artifact as binding
import exact_pv_launch as launch


def put(destination, kind, value):
    ctypes.cast(destination, ctypes.POINTER(kind))[0] = value


def number(value):
    return getattr(value, "value", value)


class Tensor:
    def __init__(self, shape, dtype, width, pointer, device=0):
        self.shape, self.dtype, self.width, self.pointer = shape, dtype, width, pointer
        self.device = types.SimpleNamespace(type="cuda", index=device)
        self.contiguous = True
        self.layout = "strided"
        self.negative = self.conjugate = False

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugate

    def numel(self):
        import math
        return math.prod(self.shape)

    def element_size(self):
        return self.width

    def data_ptr(self):
        return self.pointer


TORCH = types.SimpleNamespace(Tensor=Tensor, int32="i32", uint32="u32", int8="i8", int64="i64", strided="strided")


class LaunchCuda(RecordingCuda):
    def __init__(self):
        super().__init__(CUBIN)
        self.context = 0x9876
        self.pointer_context = self.context
        self.limits = [1024, 1024, 1024, 64, (1 << 31) - 1, 65535, 65535]
        self.function_threads = 1024
        self.regions = {0x1000: 512, 0x2000: 512, 0x3000: 512}
        self.launches, self.unloads, self.syncs = [], [], 0
        self.launch_error = self.sync_error = self.attribute_error = 0
        self.pointer_error = self.allocation_error = self.function_error = 0
        self.allocation_callback = None

    def cuCtxGetCurrent(self, result):
        put(result, ctypes.c_void_p, self.context)
        return 0

    def cuDeviceGet(self, result, ordinal):
        put(result, ctypes.c_int, number(ordinal))
        return 0

    def cuDeviceGetAttribute(self, result, attribute, device):
        attribute = number(attribute)
        if attribute in (75, 76):
            return super().cuDeviceGetAttribute(result, attribute, device)
        put(result, ctypes.c_int, self.limits[attribute - 1])
        return self.attribute_error

    def cuModuleGetFunction(self, result, module, entry):
        self.assert_entry = entry
        put(result, ctypes.c_void_p, 0x5678)
        return 0

    def cuFuncGetAttribute(self, result, attribute, function):
        if number(attribute) != 0:
            raise AssertionError("unexpected function attribute")
        put(result, ctypes.c_int, self.function_threads)
        return self.function_error

    def cuPointerGetAttribute(self, result, attribute, pointer):
        if number(attribute) != 1:
            raise AssertionError("unexpected pointer attribute")
        put(result, ctypes.c_void_p, self.pointer_context)
        return self.pointer_error

    def cuMemGetAddressRange_v2(self, base, size, pointer):
        if self.allocation_callback:
            callback, self.allocation_callback = self.allocation_callback, None
            callback()
        pointer = number(pointer)
        region = next(((at, length) for at, length in self.regions.items()
                       if at <= pointer < at + length), None)
        if region is None:
            return 1
        put(base, ctypes.c_uint64, region[0])
        put(size, ctypes.c_size_t, region[1])
        return self.allocation_error

    def cuLaunchKernel(self, function, *args):
        geometry = tuple(number(value) for value in args[:7])
        parameters = args[8]
        pointers = tuple(ctypes.cast(parameters[i], ctypes.POINTER(ctypes.c_uint64))[0]
                         for i in range(3))
        scalars = tuple(ctypes.cast(parameters[i], ctypes.POINTER(ctypes.c_int32))[0]
                        for i in range(3, 9))
        self.launches.append((geometry, pointers, scalars))
        return self.launch_error

    def cuCtxSynchronize(self):
        self.syncs += 1
        return self.sync_error

    def cuModuleUnload(self, module):
        self.unloads.append(module.value)
        return 0


class ShapeContracts(unittest.TestCase):
    def test_positive_shape_covers_every_output_once_and_all_inputs_are_in_range(self):
        for bmax in range(1, 4):
            for qmax in range(1, 4):
                for tmax in range(1, 4):
                    for dmax in range(1, 4):
                        shape = launch.ExactPvShape(bmax, qmax, tmax, dmax)
                        outputs = []
                        for b in range(shape.grid[1]):
                            for q in range(shape.grid[0]):
                                for d in range(shape.block[0]):
                                    outputs.append((b * qmax + q) * dmax + d)
                                    for t in range(tmax):
                                        self.assertLess((b * qmax + q) * tmax + t, shape.np)
                                        self.assertLess((b * tmax + t) * dmax + d, shape.nv)
                        self.assertEqual(sorted(outputs), list(range(shape.no)))

    def test_signed_dimensions_and_checked_products(self):
        for bad in (0, -1, -(1 << 63), 1 << 31, True, 1.0):
            for position in range(4):
                values = [1] * 4
                values[position] = bad
                with self.subTest(bad=bad, position=position), self.assertRaises(binding.ArtifactError):
                    launch.ExactPvShape(*values)
        for values in ((65536, 32768, 1, 1), (65536, 1, 1, 32768), (1, 1, 65536, 32768)):
            with self.subTest(values=values), self.assertRaises(binding.ArtifactError):
                launch.ExactPvShape(*values)
        shape = launch.ExactPvShape(1, launch.I32MAX, 1, 1)
        self.assertEqual(shape.np, launch.I32MAX)

    def test_full_unsigned_probability_and_signed_byte_accumulator_limit(self):
        limit = launch.I64MAX // launch.TERM_BOUND
        self.assertEqual(limit, 16777216)
        self.assertEqual(launch.ExactPvShape(1, 1, limit, 1).keys, limit)
        with self.assertRaisesRegex(binding.ArtifactError, "accumulator"):
            launch.ExactPvShape(1, 1, limit + 1, 1)

    def test_live_ranges_are_aligned_contained_and_nonwrapping(self):
        self.assertEqual(launch._span(0x1008, 8, 8, 0x1000, 16), (0x1008, 0x1010))
        for pointer, size, alignment, base, capacity in (
                (0, 4, 4, 0x1000, 16), (0x1001, 4, 4, 0x1000, 16),
                (0x1008, 9, 8, 0x1000, 16), (0xFF8, 8, 8, 0x1000, 16),
                (launch.U64MAX - 7, 8, 8, launch.U64MAX - 7, 8),
                (0x1000, 4, 4, 0x1000, launch.U64MAX)):
            with self.subTest(pointer=pointer, size=size), self.assertRaises(binding.ArtifactError):
                launch._span(pointer, size, alignment, base, capacity)

    def test_package_import_does_not_require_z3_or_torch(self):
        script = "import sys; sys.path.insert(0,sys.argv[1]); from ptxas_tval.exact_pv_launch import ExactPvShape; assert ExactPvShape(1,2,3,4).no==8; assert 'z3' not in sys.modules; assert 'torch' not in sys.modules"
        import subprocess
        output = subprocess.run([sys.executable, "-c", script, str(REPO / "tools")], capture_output=True)
        self.assertEqual(output.returncode, 0, output.stderr.decode())


class CheckedLaunchContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="y-pv-launch-contract-")
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name) / "bundle"
        write_bundle(directory)
        self.artifact = binding.open_verified(directory)
        self.cuda = LaunchCuda()
        self.module = launch.CheckedExactPv.load(self.artifact, self.cuda)
        self.shape = launch.ExactPvShape(2, 3, 5, 4)
        self.p = Tensor((2, 3, 5), "i32", 4, 0x1000)
        self.v = Tensor((2, 5, 4), "i8", 1, 0x2000)
        self.out = Tensor((2, 3, 4), "i64", 8, 0x3000)
        patcher = mock.patch.dict(sys.modules, {"torch": TORCH})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_launch(self):
        return self.module.launch(self.p, self.v, self.out, self.shape)

    def assert_refused(self, regex=None):
        with self.assertRaisesRegex(binding.ArtifactError, regex or ".*"):
            self.run_launch()
        self.assertEqual(self.cuda.launches, [], "refusal must precede cuLaunchKernel")

    def test_launch_uses_bound_cubin_exact_geometry_and_nine_abi_arguments(self):
        self.assertIs(self.run_launch(), self.out)
        self.assertEqual(self.cuda.launches, [((3, 2, 1, 4, 1, 1, 0),
                                              (0x1000, 0x2000, 0x3000),
                                              (5, 4, 3, 30, 40, 24))])
        self.assertEqual(self.cuda.loaded, [CUBIN])
        self.assertEqual(self.cuda.jit_calls, [])
        self.assertEqual(self.cuda.syncs, 2)

    def test_uint32_storage_and_overlapping_read_only_inputs_are_allowed(self):
        self.p.dtype = "u32"
        self.v.pointer = self.p.pointer + 1
        self.run_launch()
        self.assertEqual(len(self.cuda.launches), 1)

    def test_tensor_subclasses_with_custom_logical_storage_are_refused(self):
        class LogicalTensor(Tensor):
            pass
        self.p = LogicalTensor(self.p.shape, self.p.dtype, self.p.width, self.p.pointer)
        self.assert_refused("licensed shape and dtype")

    def test_invalid_tensor_metadata_is_refused(self):
        for tensor, attribute, invalid in (
                (self.p, "dtype", "i64"), (self.v, "dtype", "u8"),
                (self.out, "dtype", "f64"), (self.p, "width", 8),
                (self.v, "negative", True), (self.out, "negative", True),
                (self.p, "conjugate", True), (self.p, "layout", "sparse"),
                (self.v, "contiguous", False), (self.out, "shape", (24,)),
                (self.p, "device", types.SimpleNamespace(type="cpu", index=0)),
                (self.p, "device", types.SimpleNamespace(type="cuda", index=1)),
                (self.p, "device", types.SimpleNamespace(type="cuda", index=None))):
            previous = getattr(tensor, attribute)
            setattr(tensor, attribute, invalid)
            with self.subTest(attribute=attribute, invalid=invalid):
                self.assert_refused()
            setattr(tensor, attribute, previous)

    def test_short_misaligned_and_overlapping_buffers_are_refused(self):
        for tensor, pointer in ((self.p, 0x1001), (self.out, 0x3004), (self.out, 0x1008),
                                (self.out, 0x2000), (self.p, 0), (self.p, 1 << 64)):
            previous = tensor.pointer
            tensor.pointer = pointer
            with self.subTest(pointer=pointer):
                self.assert_refused()
            tensor.pointer = previous
        for base, capacity in ((0x1000, 119), (0x2000, 39), (0x3000, 191)):
            previous = self.cuda.regions[base]
            self.cuda.regions[base] = capacity
            with self.subTest(base=base):
                self.assert_refused("allocation")
            self.cuda.regions[base] = previous

    def test_current_context_and_buffer_context_must_match(self):
        self.cuda.context += 1
        self.assert_refused("context changed")
        self.cuda.context -= 1
        self.cuda.pointer_context += 1
        self.assert_refused("different CUDA context")

    def test_context_change_during_allocation_queries_is_refused(self):
        self.cuda.allocation_callback = lambda: setattr(self.cuda, "context", 0x9999)
        self.assert_refused("context changed")

    def test_device_grid_and_function_limits_are_enforced_without_rounding(self):
        for limits, function_limit in (([3, 1024, 1024, 64, 100, 100, 100], 1024),
                                      ([1024, 3, 1024, 64, 100, 100, 100], 1024),
                                      ([1024, 1024, 1024, 64, 2, 100, 100], 1024),
                                      ([1024, 1024, 1024, 64, 100, 1, 100], 1024),
                                      ([1024, 1024, 1024, 64, 100, 100, 100], 3)):
            self.module._limits = limits
            self.module._function_threads = function_limit
            with self.subTest(limits=limits, function_limit=function_limit):
                self.assert_refused("launch limits")
        self.module._limits = [4, 4, 1, 1, 3, 2, 1]
        self.module._function_threads = 4
        self.run_launch()

    def test_shape_cannot_be_forged_by_mutating_frozen_instance(self):
        object.__setattr__(self.shape, "keys", 0)
        self.assert_refused("positive")

    def test_driver_failures_propagate(self):
        for field in ("pointer_error", "allocation_error", "sync_error"):
            setattr(self.cuda, field, 7)
            with self.subTest(field=field):
                self.assert_refused("CUresult 7")
            setattr(self.cuda, field, 0)
        self.cuda.launch_error = 8
        with self.assertRaisesRegex(binding.ArtifactError, "CUresult 8"):
            self.run_launch()
        self.assertEqual(len(self.cuda.launches), 1)
        self.cuda.launch_error = 0
        original = self.cuda.cuCtxSynchronize
        self.cuda.cuCtxSynchronize = lambda: 9 if len(self.cuda.launches) >= 2 else original()
        with self.assertRaisesRegex(binding.ArtifactError, "CUresult 9"):
            self.run_launch()

    def test_missing_or_failed_launch_queries_refuse_loading(self):
        for field, value in (("attribute_error", 7), ("function_error", 8),
                             ("function_threads", 0), ("cuMemGetAddressRange_v2", None),
                             ("cuPointerGetAttribute", None)):
            cuda = LaunchCuda()
            setattr(cuda, field, value)
            with self.subTest(field=field), self.assertRaises(binding.ArtifactError):
                launch.CheckedExactPv.load(self.artifact, cuda)
            self.assertEqual(cuda.launches, [])
            self.assertEqual(cuda.unloads, [0x1234])

    def test_module_close_is_synchronous_and_prevents_future_launch(self):
        self.module.close()
        self.assertEqual(self.cuda.unloads, [0x1234])
        self.module.close()
        self.assert_refused("closed")


if __name__ == "__main__":
    verification_main()
