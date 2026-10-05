"""Optional hardware differential checks for the bound exact_pv cubin.

Run with the repository Python environment on an sm_89 GPU:

    Y_EXACT_PV_ARTIFACT=/path/to/validated venv/bin/python tests/exact_pv_checked_gpu.py -v

Without Y_EXACT_PV_ARTIFACT, build a private bundle from the current, pinned
tests/exact_pv.ptx using real ptxas and the translation validator. This suite
does not rebuild Y; source emission is checked by the separate compiler gates.
An invalid supplied bundle is an error, never a reason to rebuild or JIT.

Missing prerequisites produce explicit skips. Y_VERIFICATION_STRICT=1 makes
those skips return nonzero, and the verification_unittest runner can publish a
structured report. This optional suite is outside the mandatory CPU-only gate.
Only executed hardware checks are evidence about hardware; no timing is taken.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from verification_unittest import main as verification_main


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from ptxas_tval.exact_pv_artifact import ArtifactError, build, open_verified
from ptxas_tval.exact_pv_launch import CheckedExactPv, ExactPvShape


def integer_reference(p_bits, v_values, shape):
    """The mathematical batched contraction, using only Python integers.

    P storage contains unsigned 32-bit bits even when its Torch dtype is int32.
    V is signed int8. No Torch matmul, floating-point conversion or GPU result
    participates in this calculation.
    """
    if len(p_bits) != shape.np or len(v_values) != shape.nv:
        raise ValueError("reference arrays do not match the contraction shape")
    result = []
    for batch in range(shape.batch):
        for query in range(shape.queries):
            for channel in range(shape.channels):
                result.append(sum(
                    (int(p_bits[(batch * shape.queries + query) * shape.keys + key]) & 0xffffffff)
                    * int(v_values[(batch * shape.keys + key) * shape.channels + channel])
                    for key in range(shape.keys)
                ))
    return result


def signed_p_storage(values):
    return [value if value < (1 << 31) else value - (1 << 32) for value in values]


class IntegerReferenceChecks(unittest.TestCase):
    def test_hand_calculated_batch_query_and_unsigned_bits(self):
        shape = ExactPvShape(2, 2, 3, 2)
        p = [1, 2, 3, 4, 5, 6, -1, 0, -(1 << 31), 1, 0, 1]
        v = [1, -1, 2, -2, 3, -3, -128, 127, 5, -4, 127, -128]
        self.assertEqual(integer_reference(p, v, shape),
                         [14, -14, 32, -32, -277025390464, 270582939521, -1, -1])


class ExactPvHardwareChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(f"Torch unavailable: {error}") from error
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable; no exact_pv hardware check executed")
        ordinal = int(os.environ.get("Y_EXACT_PV_CUDA_DEVICE", "0"))
        if not 0 <= ordinal < torch.cuda.device_count():
            raise ValueError("Y_EXACT_PV_CUDA_DEVICE is not an available CUDA ordinal")
        if torch.cuda.get_device_capability(ordinal) != (8, 9):
            raise unittest.SkipTest("bound exact_pv hardware checks require an sm_89 device")
        torch.cuda.set_device(ordinal)
        cls.torch = torch
        cls.device = torch.device("cuda", ordinal)
        torch.empty(1, dtype=torch.int8, device=cls.device)  # Establish the current context.
        try:
            cuda = ctypes.CDLL("libcuda.so.1")
        except OSError as error:
            raise unittest.SkipTest(f"CUDA driver library unavailable: {error}") from error
        supplied = os.environ.get("Y_EXACT_PV_ARTIFACT")
        if supplied:
            bundle = Path(supplied)
            artifact = open_verified(bundle)
        else:
            missing = [name for name in ("ptxas", "nvdisasm") if shutil.which(name) is None]
            if missing:
                raise unittest.SkipTest(f"artifact build tools unavailable: {', '.join(missing)}")
            try:
                import z3  # noqa: F401; needed only to build, never to load a retained bundle.
            except ImportError as error:
                raise unittest.SkipTest(f"artifact build requires Z3: {error}") from error
            temporary = tempfile.TemporaryDirectory(prefix="y_exact_pv_checked_gpu_")
            cls.addClassCleanup(temporary.cleanup)
            bundle = Path(temporary.name) / "validated"
            artifact = build(REPO / "tests" / "exact_pv.ptx", bundle)
        cls.kernel = CheckedExactPv.load(artifact, cuda)
        cls.addClassCleanup(cls.kernel.close)
        print(f"HARDWARE exact_pv: sm_89, cubin SHA-256 {artifact.cubin_sha256}, bundle {bundle}")

    def _padded(self, values, dtype, dimensions, prefix, suffix, canary):
        host = [canary] * prefix + list(values) + [canary] * suffix
        storage = self.torch.tensor(host, dtype=dtype, device=self.device)
        live = storage.narrow(0, prefix, len(values)).view(*dimensions)
        self.assertTrue(live.is_contiguous())
        self.assertGreater(live.storage_offset(), 0)
        return storage, live, host

    def _assert_output(self, p, v, shape, p_values, v_values):
        expected = integer_reference(p_values, v_values, shape)
        canary = -0x123456789abc
        storage, out, host = self._padded(
            [canary] * shape.no, self.torch.int64,
            (shape.batch, shape.queries, shape.channels), 3, 5, canary)
        returned = self.kernel.launch(p, v, out, shape)
        self.assertIs(returned, out)
        self.assertEqual(out.cpu().reshape(-1).tolist(), expected)
        self.assertEqual(storage.cpu().tolist(), host[:3] + expected + host[-5:])
        return expected

    def test_full_u32_and_i8_domains_with_odd_channel_counts(self):
        p_pattern = [0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0x80000001]
        v_pattern = [-128, 127, 0, 1, -1, -127, 42]
        for dimensions in ((1, 1, 1, 1), (2, 3, 7, 7), (3, 2, 33, 35)):
            shape = ExactPvShape(*dimensions)
            for unsigned_dtype in (False, True):
                with self.subTest(shape=dimensions, unsigned_dtype=unsigned_dtype):
                    if unsigned_dtype and not hasattr(self.torch, "uint32"):
                        self.skipTest("this Torch build cannot represent uint32 storage")
                    p_values = [p_pattern[(index * 5 + 3) % len(p_pattern)] for index in range(shape.np)]
                    v_values = [v_pattern[(index * 3) % len(v_pattern)] for index in range(shape.nv)]
                    p = self.torch.tensor(signed_p_storage(p_values), dtype=self.torch.int32,
                                          device=self.device).view(shape.batch, shape.queries, shape.keys)
                    if unsigned_dtype:
                        p = p.view(self.torch.uint32)
                    v = self.torch.tensor(v_values, dtype=self.torch.int8,
                                          device=self.device).view(shape.batch, shape.keys, shape.channels)
                    self._assert_output(p, v, shape, p_values, v_values)
                    self.assertEqual(p.cpu().reshape(-1).tolist(),
                                     p_values if unsigned_dtype else signed_p_storage(p_values))
                    self.assertEqual(v.cpu().reshape(-1).tolist(), v_values)

    def test_integer_output_beyond_exact_float64_integer_range(self):
        shape = ExactPvShape(1, 1, 20001, 3)
        p_values = [0xffffffff] * shape.np
        v_values = [value for _ in range(shape.keys) for value in (127, -128, 1)]
        p = self.torch.tensor(signed_p_storage(p_values), dtype=self.torch.int32,
                              device=self.device).view(1, 1, shape.keys)
        v = self.torch.tensor(v_values, dtype=self.torch.int8,
                              device=self.device).view(1, shape.keys, 3)
        expected = self._assert_output(p, v, shape, p_values, v_values)
        self.assertGreater(expected[0], 1 << 53)
        self.assertEqual(expected[0] % 2, 1)

    def test_dense_storage_offsets_preserve_inputs_and_output_canaries(self):
        shape = ExactPvShape(2, 2, 11, 5)
        p_values = signed_p_storage([0xffffffff if index % 2 else 0x80000000
                                    for index in range(shape.np)])
        v_values = [-128 if index % 3 else 127 for index in range(shape.nv)]
        ps, p, phost = self._padded(p_values, self.torch.int32,
                                   (shape.batch, shape.queries, shape.keys), 3, 7, 0x345678)
        vs, v, vhost = self._padded(v_values, self.torch.int8,
                                   (shape.batch, shape.keys, shape.channels), 5, 9, 71)
        self._assert_output(p, v, shape, p_values, v_values)
        self.assertEqual(ps.cpu().tolist(), phost)
        self.assertEqual(vs.cpu().tolist(), vhost)

    def test_readonly_input_byte_ranges_may_alias(self):
        shape = ExactPvShape(2, 2, 7, 5)
        pattern = signed_p_storage([0xffffffff, 0x80000000, 0x7fffffff, 0, 1])
        host = self.torch.tensor([pattern[index % len(pattern)] for index in range(shape.np + 8)],
                                 dtype=self.torch.int32)
        storage = host.to(self.device)
        p = storage.narrow(0, 2, shape.np).view(shape.batch, shape.queries, shape.keys)
        v = storage.view(self.torch.int8).narrow(0, 8, shape.nv).view(shape.batch, shape.keys, shape.channels)
        p_values = host.narrow(0, 2, shape.np).tolist()
        v_values = host.view(self.torch.int8).narrow(0, 8, shape.nv).tolist()
        self.assertEqual(p.data_ptr(), v.data_ptr())
        self._assert_output(p, v, shape, p_values, v_values)
        self.assertEqual(storage.cpu().tolist(), host.tolist())

    def test_overlapping_output_is_refused_without_writes(self):
        shape = ExactPvShape(1, 1, 3, 3)
        for overlapped_input in ("p", "v"):
            with self.subTest(overlapped_input=overlapped_input):
                storage = self.torch.full((16,), 0x123456, dtype=self.torch.int64, device=self.device)
                before = storage.cpu().tolist()
                out = storage.narrow(0, 1, shape.no).view(1, 1, 3)
                p = self.torch.ones((1, 1, 3), dtype=self.torch.int32, device=self.device)
                v = self.torch.ones((1, 3, 3), dtype=self.torch.int8, device=self.device)
                if overlapped_input == "p":
                    p = storage.view(self.torch.int32).narrow(0, 2, shape.np).view(1, 1, 3)
                else:
                    v = storage.view(self.torch.int8).narrow(0, 8, shape.nv).view(1, 3, 3)
                with self.assertRaisesRegex(ArtifactError, "overlaps"):
                    self.kernel.launch(p, v, out, shape)
                self.assertEqual(storage.cpu().tolist(), before)

    def test_lazy_negative_input_and_output_views_are_refused(self):
        if not hasattr(self.torch, "_neg_view"):
            self.skipTest("this Torch build does not expose lazy negative views")
        shape = ExactPvShape(1, 1, 3, 3)
        p = self.torch.ones((1, 1, 3), dtype=self.torch.int32, device=self.device)
        v = self.torch.ones((1, 3, 3), dtype=self.torch.int8, device=self.device)
        out = self.torch.full((1, 1, 3), 71, dtype=self.torch.int64, device=self.device)
        for source, destination in ((self.torch._neg_view(v), out),
                                    (v, self.torch._neg_view(out))):
            self.assertTrue(source.is_neg() or destination.is_neg())
            self.assertTrue(source.is_contiguous() and destination.is_contiguous())
            with self.assertRaisesRegex(ArtifactError, "licensed shape and dtype"):
                self.kernel.launch(p, source, destination, shape)
            self.assertEqual(out.cpu().reshape(-1).tolist(), [71] * shape.no)


if __name__ == "__main__":
    verification_main()
