"""Artifact-binding regressions; the recording CUDA driver needs no GPU.

The synthetic ELF fixtures exercise only the artifact/loading boundary.  They
make no claim about instruction semantics.  The optional toolchain test also
builds and validates the committed exact_pv PTX with the real validator.
"""
from __future__ import annotations

import ast
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from verification_unittest import main as verification_main


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "ptxas_tval"))
import exact_pv_artifact as binding


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def synthetic_cubin():
    """A structurally identified sm_89 ELF with an exact_pv code section."""
    names = b"\0.shstrtab\0.text.exact_pv\0"
    code = b"\x00\x91\x00\xa7\x00\xb3\x00\xc5" * 2
    text_offset = 64 + len(names)
    sections_offset = (text_offset + len(code) + 7) & ~7
    ident = b"\x7fELF\x02\x01\x01\x41\x08" + b"\0" * 7
    header = struct.pack(
        "<16sHHIQQQIHHHHHH", ident, 2, 190, 1, 0, 0,
        sections_offset, 0x06005904, 64, 0, 0, 64, 3, 1,
    )
    result = header + names + code
    result += b"\0" * (sections_offset - len(result))
    result += b"\0" * 64
    result += struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, 64, len(names), 0, 0, 1, 0)
    result += struct.pack(
        "<IIQQQQIIQQ", names.index(b".text.exact_pv"), 1, 6, 0,
        text_offset, len(code), 0, 0, 16, 0,
    )
    return result


PTX = (REPO / "tests" / "exact_pv.ptx").read_bytes()
CUBIN = synthetic_cubin()
SASS = b".target sm_89\n// Synthetic fixture; instruction semantics are not tested here.\n"


def write_bundle(directory, cubin=CUBIN):
    directory.mkdir()
    for name, content in (("ptx", PTX), ("sass", SASS), ("cubin", cubin)):
        (directory / f"exact_pv.{name}").write_bytes(content)
    receipt = {
        "format": "y-exact-pv-cubin-v1",
        "entry": "exact_pv",
        "target": "sm_89",
        "optimization": "1",
        "validator": "loopval",
        "verdict": "VALIDATED",
        "obligations": "14",
        "ptx_sha256": sha256(PTX),
        "sass_sha256": sha256(SASS),
        "cubin_sha256": sha256(cubin),
    }
    (directory / "receipt.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in receipt.items()), encoding="ascii"
    )


def corrupt_byte(path):
    data = bytearray(path.read_bytes())
    # Change the code payload, preserving the ELF header and all offsets.
    data[64 + len(b"\0.shstrtab\0.text.exact_pv\0")] ^= 1
    path.write_bytes(data)


class RecordingCuda:
    """Only direct binary module loading is admitted by this fake driver."""

    def __init__(self, expected, capability=(8, 9), attribute_callback=None):
        self.expected = expected
        self.capability = capability
        self.attribute_callback = attribute_callback
        self.loaded = []
        self.jit_calls = []

    def cuCtxGetDevice(self, device):
        ctypes.cast(device, ctypes.POINTER(ctypes.c_int))[0] = 0
        return 0

    def cuDeviceGetAttribute(self, value, attribute, device):
        if self.attribute_callback:
            callback, self.attribute_callback = self.attribute_callback, None
            callback()
        attribute = getattr(attribute, "value", attribute)
        if attribute not in (75, 76):
            raise AssertionError(f"unexpected device attribute {attribute}")
        ctypes.cast(value, ctypes.POINTER(ctypes.c_int))[0] = self.capability[attribute - 75]
        return 0

    def cuModuleLoadData(self, module, image):
        # An explicit byte count is essential: c_char_p/.value would truncate
        # this binary at its first embedded NUL.
        actual = ctypes.string_at(image, len(self.expected))
        if not actual.startswith(b"\x7fELF"):
            raise AssertionError("verified loader submitted PTX instead of an ELF cubin")
        if actual != self.expected:
            raise AssertionError("CUDA received bytes other than the validated artifact")
        self.loaded.append(actual)
        ctypes.cast(module, ctypes.POINTER(ctypes.c_void_p))[0] = 0x1234
        return 0

    def cuModuleLoadDataEx(self, *args):
        self.jit_calls.append(args)
        raise AssertionError("verified path invoked the PTX JIT API")

    def cuLinkCreate(self, *args):
        self.jit_calls.append(args)
        raise AssertionError("verified path invoked a JIT linker")


class ArtifactBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="y-artifact-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "bundle"
        write_bundle(self.directory)

    def assert_refused(self, operation, cuda=None):
        with self.assertRaises(binding.ArtifactError):
            operation()
        if cuda is not None:
            self.assertEqual(cuda.loaded, [], "failure must precede module loading")
            self.assertEqual(cuda.jit_calls, [])

    def test_validated_cubin_is_loaded_byte_for_byte_without_ptx_jit(self):
        artifact = binding.open_verified(self.directory, expected_ptx_sha256=sha256(PTX))
        cuda = RecordingCuda(CUBIN)
        module = artifact.load(cuda)
        self.assertEqual(module.value, 0x1234)
        self.assertEqual(cuda.loaded, [CUBIN])
        self.assertEqual(cuda.jit_calls, [])

    def test_bridge_uses_verified_loader_without_calling_legacy_ptx_constructor(self):
        # Execute the actual Module class without importing torch or loading
        # libcuda. This checks the bridge wiring as well as the artifact API.
        path = REPO / "tools" / "ptx_bridge.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        module_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Module"
        )
        cuda = RecordingCuda(CUBIN)
        namespace = {"ctypes": ctypes, "cuda": cuda}
        exec(compile(ast.Module(body=[module_class], type_ignores=[]), str(path), "exec"), namespace)
        module_type = namespace["Module"]
        artifact = binding.open_verified(self.directory)
        with mock.patch.object(
            module_type, "__init__", side_effect=AssertionError("legacy PTX constructor invoked")
        ) as legacy:
            module = module_type.from_validated_exact_pv(artifact)
        legacy.assert_not_called()
        self.assertEqual(module.mod.value, 0x1234)
        self.assertEqual(cuda.loaded, [CUBIN])
        self.assertEqual(cuda.jit_calls, [])

    def test_modifying_one_byte_invalidates_the_binding(self):
        corrupt_byte(self.directory / "exact_pv.cubin")
        self.assert_refused(lambda: binding.open_verified(self.directory))

    def test_change_between_open_and_load_is_refused(self):
        artifact = binding.open_verified(self.directory)
        corrupt_byte(self.directory / "exact_pv.cubin")
        cuda = RecordingCuda(CUBIN)
        self.assert_refused(lambda: artifact.load(cuda), cuda)

    def test_change_during_device_query_is_refused(self):
        artifact = binding.open_verified(self.directory)
        cuda = RecordingCuda(
            CUBIN, attribute_callback=lambda: corrupt_byte(self.directory / "exact_pv.cubin")
        )
        self.assert_refused(lambda: artifact.load(cuda), cuda)

    def test_driver_consumes_the_checked_snapshot_without_reopening_the_file(self):
        artifact = binding.open_verified(self.directory)
        cuda = RecordingCuda(CUBIN)
        load_binary = cuda.cuModuleLoadData

        def change_file_then_load(module, image):
            corrupt_byte(self.directory / "exact_pv.cubin")
            return load_binary(module, image)

        cuda.cuModuleLoadData = change_file_then_load
        self.assertEqual(artifact.load(cuda).value, 0x1234)
        self.assertEqual(cuda.loaded, [CUBIN])
        self.assertNotEqual((self.directory / "exact_pv.cubin").read_bytes(), CUBIN)
        # The just-loaded bytes were the correct immutable snapshot. A later
        # load sees the changed file and must refuse it.
        self.assert_refused(lambda: binding.open_verified(self.directory))

    def test_missing_artifact_or_receipt_is_refused(self):
        for filename in ("exact_pv.ptx", "exact_pv.sass", "exact_pv.cubin", "receipt.txt"):
            with self.subTest(filename=filename):
                path = self.directory / filename
                saved = path.read_bytes()
                path.unlink()
                self.assert_refused(lambda: binding.open_verified(self.directory))
                path.write_bytes(saved)

    def test_missing_receipt_after_open_is_refused(self):
        artifact = binding.open_verified(self.directory)
        (self.directory / "receipt.txt").unlink()
        cuda = RecordingCuda(CUBIN)
        self.assert_refused(lambda: artifact.load(cuda), cuda)

    def test_changed_receipt_after_open_is_refused(self):
        artifact = binding.open_verified(self.directory)
        path = self.directory / "receipt.txt"
        path.write_text(path.read_text().replace("obligations=14", "obligations=15"))
        cuda = RecordingCuda(CUBIN)
        self.assert_refused(lambda: artifact.load(cuda), cuda)

    def test_receipt_cannot_relabel_the_target_or_entry(self):
        path = self.directory / "receipt.txt"
        original = path.read_text()
        for old, new in (("target=sm_89", "target=sm_90"), ("entry=exact_pv", "entry=other")):
            with self.subTest(new=new):
                path.write_text(original.replace(old, new))
                self.assert_refused(lambda: binding.open_verified(self.directory))
        path.write_text(original)

    def test_duplicate_or_missing_identity_is_refused(self):
        path = self.directory / "receipt.txt"
        original = path.read_text()
        for bad in (
            original + f"cubin_sha256={sha256(CUBIN)}\n",
            "\n".join(line for line in original.splitlines() if not line.startswith("cubin_sha256=")),
            original.replace(sha256(CUBIN), "unidentified"),
        ):
            with self.subTest(receipt=bad):
                path.write_text(bad)
                self.assert_refused(lambda: binding.open_verified(self.directory))

    def test_unvalidated_or_empty_obligation_receipts_are_refused(self):
        path = self.directory / "receipt.txt"
        original = path.read_text()
        for old, new in (("verdict=VALIDATED", "verdict=REFUSED"), ("obligations=14", "obligations=0")):
            with self.subTest(new=new):
                path.write_text(original.replace(old, new))
                self.assert_refused(lambda: binding.open_verified(self.directory))

    def test_expected_ptx_identity_must_match(self):
        self.assert_refused(lambda: binding.open_verified(self.directory, expected_ptx_sha256="0" * 64))

    def test_rehashed_wrong_index_ptx_is_outside_the_reviewed_proof_subject(self):
        wrong = PTX.replace(b"add.s32 %r17, %r11, %r14;", b"mov.u32 %r17, %r11;", 1)
        self.assertNotEqual(wrong, PTX, "reviewed index anchor moved")
        (self.directory / "exact_pv.ptx").write_bytes(wrong)
        receipt = self.directory / "receipt.txt"
        receipt.write_text(receipt.read_text().replace(sha256(PTX), sha256(wrong)))
        # All artifact hashes and opcode-presence checks can hold for a wrong
        # algorithm. A successful local translation receipt cannot expand the
        # fixed proof subject accepted by the verified loader.
        self.assert_refused(lambda: binding.open_verified(self.directory))

    def test_comment_and_whitespace_changes_preserve_reviewed_subject_identity(self):
        changed = b"/* reviewed subject, comments only */\n" + PTX.replace(
            b"    add.s32 %r17, %r11, %r14;", b"\tadd.s32  %r17, %r11, %r14; // index\n", 1)
        (self.directory / "exact_pv.ptx").write_bytes(changed)
        receipt = self.directory / "receipt.txt"
        receipt.write_text(receipt.read_text().replace(sha256(PTX), sha256(changed)))
        self.assertEqual(binding.open_verified(self.directory).checked_image(), CUBIN)

    def test_device_target_mismatch_is_refused(self):
        artifact = binding.open_verified(self.directory)
        for capability in ((8, 6), (9, 0), (0, 0)):
            with self.subTest(capability=capability):
                cuda = RecordingCuda(CUBIN, capability=capability)
                self.assert_refused(lambda: artifact.load(cuda), cuda)

    def test_unidentifiable_device_is_refused(self):
        artifact = binding.open_verified(self.directory)
        cuda = RecordingCuda(CUBIN)
        cuda.cuCtxGetDevice = lambda device: 201
        self.assert_refused(lambda: artifact.load(cuda), cuda)

    def test_hash_correct_ptx_disguised_as_cubin_is_refused(self):
        shutil.rmtree(self.directory)
        write_bundle(self.directory, cubin=PTX)
        self.assert_refused(lambda: binding.open_verified(self.directory))

    def test_hash_correct_cubin_for_other_target_is_refused(self):
        wrong_target = bytearray(CUBIN)
        struct.pack_into("<I", wrong_target, 48, 0x06005004)
        shutil.rmtree(self.directory)
        write_bundle(self.directory, cubin=bytes(wrong_target))
        self.assert_refused(lambda: binding.open_verified(self.directory))


class BuildBindingTests(unittest.TestCase):
    """Exercise publication and validation gating without external programs."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="y-artifact-build-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "bundle"
        self.source = Path(self.temporary.name) / "input.ptx"
        self.source.write_bytes(PTX)

    def compiler(self, command, **kwargs):
        command = [str(value) for value in command]
        if Path(command[0]).name == "ptxas":
            self.assertTrue("-O1" in command or "-O" in command and "1" in command)
            self.assertTrue("-arch=sm_89" in command or "sm_89" in command)
            Path(command[command.index("-o") + 1]).write_bytes(CUBIN)
            output = b""
        elif Path(command[0]).name == "nvdisasm":
            self.assertIn("-c", command)
            self.assertEqual(Path(command[-1]).read_bytes(), CUBIN)
            output = SASS
        else:
            raise AssertionError(f"unexpected subprocess {command}")
        if kwargs.get("text") or kwargs.get("encoding"):
            output = output.decode()
        destination = kwargs.get("stdout")
        if hasattr(destination, "write"):
            destination.write(output)
            output = None
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="" if kwargs.get("text") else b"")

    def build_with_validator(self, validator):
        self.validation = mock.Mock(side_effect=validator)
        with mock.patch.object(binding, "_validate_translation", self.validation), mock.patch.object(
            binding.subprocess, "run", side_effect=self.compiler
        ):
            return binding.build(self.source, self.directory)

    def test_success_publishes_the_validated_bytes_then_loads_them(self):
        validated = []

        def validate(ptx_path, sass_path, **kwargs):
            self.assertEqual(Path(ptx_path).read_bytes(), PTX)
            self.assertEqual(Path(sass_path).read_bytes(), SASS)
            self.assertFalse((self.directory / "receipt.txt").exists())
            validated.append(True)
            return "VALIDATED", "1 stores, 3 relation pairs", 14

        artifact = self.build_with_validator(validate)
        self.validation.assert_called_once()
        self.assertEqual(validated, [True])
        self.assertEqual((self.directory / "exact_pv.cubin").read_bytes(), CUBIN)
        cuda = RecordingCuda(CUBIN)
        self.assertEqual(artifact.load(cuda).value, 0x1234)
        self.assertEqual(cuda.loaded, [CUBIN])
        self.assertEqual(cuda.jit_calls, [])

    def test_changed_algorithm_refuses_before_compilation_or_publication(self):
        wrong = PTX.replace(b"add.s32 %r17, %r11, %r14;", b"mov.u32 %r17, %r11;", 1)
        self.assertNotEqual(wrong, PTX)
        self.source.write_bytes(wrong)
        with mock.patch.object(binding.subprocess, "run") as compiler:
            with self.assertRaisesRegex(binding.ArtifactError, 'reviewed ExactPvExact proof subject'):
                binding.build(self.source, self.directory)
        compiler.assert_not_called()
        self.assertFalse(self.directory.exists())

    def test_commented_target_directives_bind_the_actual_ptx_snapshot(self):
        sources = (
            PTX.replace(b".target sm_89", b".target sm_89 // target comment", 1),
            PTX.replace(b".target sm_89", b".target/* target comment */ sm_89", 1),
            b"/*\n.target sm_90\n*/\n" + PTX,
        )
        for index, source in enumerate(sources):
            with self.subTest(source=source):
                self.source.write_bytes(source)
                self.directory = Path(self.temporary.name) / f"commented-{index}"
                artifact = self.build_with_validator(
                    lambda *args, **kwargs: ("VALIDATED", "test successful validation", 14)
                )
                self.assertEqual((self.directory / "exact_pv.ptx").read_bytes(), source)
                binding.open_verified(self.directory, expected_ptx_sha256=sha256(source))
                self.assertEqual(artifact.checked_image(), CUBIN)

    def test_target_features_and_unterminated_comments_are_refused(self):
        for source in (
            PTX.replace(b".target sm_89", b".target sm_89, texmode_independent", 1),
            PTX + b"/* unterminated comment",
        ):
            with self.subTest(source=source):
                self.source.write_bytes(source)
                with self.assertRaises(binding.ArtifactError):
                    self.build_with_validator(
                        lambda *args, **kwargs: ("VALIDATED", "test successful validation", 14)
                    )
                self.validation.assert_not_called()
                self.assertFalse(self.directory.exists())

    def test_destination_is_reserved_while_validation_runs(self):
        def validate(*args, **kwargs):
            self.assertTrue(self.directory.is_dir())
            self.assertFalse((self.directory / "receipt.txt").exists())
            with self.assertRaises(binding.ArtifactError):
                binding.build(self.source, self.directory)
            return "VALIDATED", "test successful validation", 14

        artifact = self.build_with_validator(validate)
        self.validation.assert_called_once()
        self.assertEqual(artifact.checked_image(), CUBIN)

    def test_foreign_contents_are_neither_replaced_nor_removed(self):
        marker = self.directory / "foreign.txt"

        def validate(*args, **kwargs):
            marker.write_text("belongs to another process")
            return "VALIDATED", "test successful validation", 14

        with self.assertRaises(binding.ArtifactError):
            self.build_with_validator(validate)
        self.assertEqual(marker.read_text(), "belongs to another process")
        self.assertFalse((self.directory / "receipt.txt").exists())
        self.assertEqual(list(self.directory.iterdir()), [marker])

    def test_failed_validation_never_publishes_an_executable_binding(self):
        for verdict in ("UNPROVED", "REFUSED"):
            with self.subTest(verdict=verdict):
                if self.directory.exists():
                    shutil.rmtree(self.directory)
                with self.assertRaises(binding.ArtifactError):
                    self.build_with_validator(lambda *args, **kwargs: (verdict, "test refusal", 0))
                self.validation.assert_called_once()
                self.assertFalse(self.directory.exists(), "failed build must release its reservation")
                self.assertFalse((self.directory / "receipt.txt").exists())
                cuda = RecordingCuda(CUBIN)
                with self.assertRaises(binding.ArtifactError):
                    binding.open_verified(self.directory).load(cuda)
                self.assertEqual(cuda.loaded, [])

    def test_validator_exception_never_publishes_an_executable_binding(self):
        def refuse(*args, **kwargs):
            raise RuntimeError("test unsupported program")

        with self.assertRaises(binding.ArtifactError):
            self.build_with_validator(refuse)
        self.validation.assert_called_once()
        self.assertFalse((self.directory / "receipt.txt").exists())
        with self.assertRaises(binding.ArtifactError):
            binding.open_verified(self.directory)

    def test_cubin_changed_while_validation_runs_is_never_published(self):
        def validate(ptx_path, sass_path, **kwargs):
            corrupt_byte(Path(sass_path).with_suffix(".cubin"))
            return "VALIDATED", "test successful validation", 14

        with self.assertRaises(binding.ArtifactError):
            self.build_with_validator(validate)
        self.validation.assert_called_once()
        self.assertFalse((self.directory / "receipt.txt").exists())

    def test_failed_or_malformed_child_result_never_publishes_a_bundle(self):
        for failure in ("crash", "timeout", "malformed"):
            with self.subTest(failure=failure):
                def run(command, **kwargs):
                    if command[0] != sys.executable:
                        return self.compiler(command, **kwargs)
                    if failure == "crash":
                        raise subprocess.CalledProcessError(1, command, stderr=b"validator crashed")
                    if failure == "timeout":
                        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                    return subprocess.CompletedProcess(command, 0, stdout=b"VALIDATED", stderr=b"")

                with mock.patch.object(binding.subprocess, "run", side_effect=run):
                    with self.assertRaises(binding.ArtifactError):
                        binding.build(self.source, self.directory)
                self.assertFalse(self.directory.exists())


class IsolatedValidationTests(unittest.TestCase):
    def response(self, **fields):
        result = {"format": "y-exact-pv-loopval-result-v1", "verdict": "VALIDATED",
                  "detail": "fixture validation", "obligations": 14}
        result.update(fields)
        return json.dumps(result).encode()

    def test_same_interpreter_budget_and_bounded_process_timeout(self):
        completed = subprocess.CompletedProcess([], 0, stdout=self.response(), stderr=b"")
        with mock.patch.object(binding.subprocess, "run", return_value=completed) as run:
            result = binding._validate_translation("input.ptx", "output.sass", budget=7)
        self.assertEqual(result, ("VALIDATED", "fixture validation", 14))
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[-3:], ["input.ptx", "output.sass", "7"])
        self.assertIn("mode='wide'", command[2])
        self.assertEqual(run.call_args.kwargs["timeout"], 32 * 7)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_invalid_result_schema_is_refused(self):
        malformed = [
            b"[]", b"not JSON", b"\xff", b"{}", self.response(format="unknown"),
            self.response(verdict="SUCCESS"), self.response(detail=14),
            self.response(obligations=True), self.response(obligations=-1),
            self.response(obligations=1 << 64), self.response(extra="unknown field"),
            self.response().replace(b'"obligations": 14', b'"obligations": 0, "obligations": 14'),
        ]
        for output in malformed:
            with self.subTest(output=output):
                completed = subprocess.CompletedProcess([], 0, stdout=output, stderr=b"")
                with mock.patch.object(binding.subprocess, "run", return_value=completed):
                    with self.assertRaises(binding.ArtifactError):
                        binding._validate_translation("input.ptx", "output.sass", budget=7)


class RealTranslationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [tool for tool in ("ptxas", "nvdisasm") if not shutil.which(tool)]
        if importlib.util.find_spec("z3") is None:
            missing.append("Python z3")
        if missing:
            raise unittest.SkipTest("real translation-validation integration requires " + ", ".join(missing))
        cls.temporary = tempfile.TemporaryDirectory(prefix="y-artifact-real-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name) / "bundle"
        cls.artifact = binding.build(REPO / "tests" / "exact_pv.ptx", cls.directory)
        # The Cargo wrapper opens this same real build through Rust's parser.
        # Nothing is exported when this class skips for absent dependencies.
        if destination := os.environ.get("Y_TVAL_TEST_BUNDLE"):
            shutil.copytree(cls.directory, destination)

    def test_real_validated_offline_cubin_is_the_loaded_image(self):
        cubin = (self.directory / "exact_pv.cubin").read_bytes()
        self.assertTrue(cubin.startswith(b"\x7fELF"))
        self.assertIn(b"\0", cubin[16:])
        cuda = RecordingCuda(cubin)
        reopened = binding.open_verified(self.directory, expected_ptx_sha256=sha256(PTX))
        self.assertEqual(reopened.load(cuda).value, 0x1234)
        self.assertEqual(cuda.loaded, [cubin])
        self.assertEqual(cuda.jit_calls, [])

    def test_sequential_builds_survive_prior_shared_validation(self):
        import batch
        import loopval
        import smemval
        import tval

        shared = REPO / "tools" / "ptxas_tval" / "smut" / "smem_roundtrip"
        ptx, sass = str(shared.with_suffix(".ptx")), str(shared.with_suffix(".sass"))
        self.assertEqual(smemval.validate(ptx, sass, 2)[0], "VALIDATED")
        self.assertEqual(batch.validate(ptx, sass, 2, "direct")[0], "VALIDATED")
        tval.run(ptx, sass, NS=2, B1=1, B2=2, log=lambda _: None)
        # The old inline builder inherited the parent's Z3 node ordering and
        # could reject this exact_pv compilation after the shared preamble.
        # Patching its inline entry also makes the process boundary explicit.
        with mock.patch.object(loopval, "validate", side_effect=AssertionError("inline validation")) as inline:
            first_obligations = None
            for index in range(2):
                directory = Path(self.temporary.name) / f"sequential-{index}"
                artifact = binding.build(REPO / "tests" / "exact_pv.ptx", directory, budget=5)
                receipt = binding._receipt((directory / "receipt.txt").read_bytes())
                self.assertGreater(int(receipt["obligations"]), 0)
                if first_obligations is None:
                    first_obligations = receipt["obligations"]
                self.assertEqual(receipt["obligations"], first_obligations)
                cuda = RecordingCuda((directory / "exact_pv.cubin").read_bytes())
                self.assertEqual(artifact.load(cuda).value, 0x1234)
                self.assertEqual(cuda.jit_calls, [])
        inline.assert_not_called()

    def test_one_byte_mutation_of_real_cubin_is_rejected(self):
        copied = Path(self.temporary.name) / "mutated"
        shutil.copytree(self.directory, copied)
        cubin_path = copied / "exact_pv.cubin"
        data = bytearray(cubin_path.read_bytes())
        data[-1] ^= 1
        cubin_path.write_bytes(data)
        with self.assertRaises(binding.ArtifactError):
            binding.open_verified(copied)


if __name__ == "__main__":
    verification_main(verbosity=2)
