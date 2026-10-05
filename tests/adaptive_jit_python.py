"""CPU-only wrapper boundary tests: python3 tests/adaptive_jit_python.py."""

import ctypes
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import adaptive_jit as module  # noqa: E402


class FakeLibrary:
    def __init__(self):
        self.calls = []
        self.strings = {}
        self.fail_destroy = False
        self.fail_create = False
        self.json_text = '{"launches":3,"cache_hit":false}'

    def string(self, value):
        buffer = ctypes.create_string_buffer(value.encode())
        pointer = ctypes.addressof(buffer)
        self.strings[pointer] = buffer
        return pointer

    def error(self, output, message):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = self.string(message)

    def y_free_string(self, pointer):
        del self.strings[pointer]

    def y_adaptive_jit_config_init(self, output, size, error):
        config = ctypes.cast(output, ctypes.POINTER(module._Config)).contents
        config.abi_version, config.struct_size = 1, size
        config.tuning_policy = 0
        config.hot_threshold = 32
        config.max_cached_shapes = 16
        config.max_candidates = 8
        config.max_disk_cache_entries = 128
        config.min_improvement = 0.05
        config.cache_dir = None
        return 0

    def y_adaptive_jit_create_current(self, pointer, error):
        config = ctypes.cast(pointer, ctypes.POINTER(module._Config)).contents
        self.config = {name: getattr(config, name) for name, _ in module._Config._fields_}
        if self.fail_create:
            self.error(error, "no current context")
            return None
        return 1234

    def y_adaptive_jit_destroy(self, handle, error):
        self.calls.append(("destroy", handle))
        if self.fail_destroy:
            self.error(error, "wrong current context")
            return -1
        return 0

    def y_adaptive_jit_prepare(self, *args):
        self.calls.append(("prepare", *args[:-1]))
        return 0

    def y_adaptive_jit_launch(self, *args):
        self.calls.append(("launch", *args[:-1]))
        return 0

    def y_adaptive_jit_synchronize(self, *args):
        self.calls.append(("synchronize", *args[:-1]))
        return 0

    def y_adaptive_jit_stats_json(self, *args):
        self.calls.append(("stats", *args[:-1]))
        return self.string(self.json_text)

    def y_adaptive_jit_pending_json(self, *args):
        return self.string("[[256,128,64]]")

    def y_adaptive_jit_tune_hot_json(self, *args):
        self.calls.append(("tune_hot", *args[:-1]))
        return self.string("[]")


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.lib = FakeLibrary()
        self.patch = patch.object(module, "_load_library", return_value=self.lib)
        self.loader = self.patch.start()
        self.addCleanup(self.patch.stop)

    def tearDown(self):
        self.assertEqual(self.lib.strings, {}, "all native strings should be freed")

    def test_native_defaults_and_context_manager_close(self):
        with module.AdaptiveGemm() as jit:
            self.assertEqual(self.lib.config["hot_threshold"], 32)
            self.assertEqual(self.lib.config["struct_size"], ctypes.sizeof(module._Config))
            self.assertEqual(self.lib.config["abi_version"], 1)
            jit.prepare(1, 16, 16)
            jit.launch(1, 16, 16, 16, 32, 48)
            jit.synchronize()
        jit.close()
        self.assertEqual(sum(call[0] == "destroy" for call in self.lib.calls), 1)
        with self.assertRaisesRegex(module.AdaptiveJitError, "closed"):
            jit.synchronize()

    def test_settings_are_passed_without_truncation(self):
        with module.AdaptiveGemm(tuning_policy="on_launch", hot_threshold=2**63,
                                 max_candidates=64, min_improvement=0,
                                 max_cached_shapes=23, max_disk_cache_entries=7,
                                 cache_dir=Path("cache-ü")):
            self.assertEqual(self.lib.config["tuning_policy"], 1)
            self.assertEqual(self.lib.config["hot_threshold"], 2**63)
            self.assertEqual(self.lib.config["max_cached_shapes"], 23)
            self.assertEqual(self.lib.config["max_candidates"], 64)
            self.assertEqual(self.lib.config["min_improvement"], 0)
            self.assertEqual(self.lib.config["max_disk_cache_entries"], 7)
            self.assertEqual(self.lib.config["cache_dir"], "cache-ü".encode())

    def test_invalid_settings_fail_before_loading_native_library(self):
        for settings in (
            {"hot_threshold": -1}, {"hot_threshold": 2**64}, {"hot_threshold": True},
            {"max_cached_shapes": 2**32}, {"max_cached_shapes": 1.5},
            {"max_candidates": 1}, {"max_candidates": 65},
            {"min_improvement": float("nan")}, {"min_improvement": 1},
            {"min_improvement": True}, {"max_disk_cache_entries": 0},
            {"tuning_policy": "unknown"}, {"tuning_policy": []},
            {"cache_dir": "contains\0nul"}, {"cache_dir": ""}, {"cache_dir": b"bytes"},
        ):
            with self.subTest(settings=settings), self.assertRaises((ValueError, TypeError)):
                module.AdaptiveGemm(**settings)
        self.loader.assert_not_called()

    def test_bad_shapes_and_pointers_never_reach_native_launch(self):
        with module.AdaptiveGemm() as jit:
            for shape in ((0, 16, 16), (1, 15, 16), (16385, 16, 16), (2**32, 16, 16)):
                with self.subTest(shape=shape), self.assertRaises(ValueError):
                    jit.launch(*shape, 16, 32, 48)
            for pointer in (0, -16, 2**64, 17, True, 3.5):
                with self.subTest(pointer=pointer), self.assertRaises((TypeError, ValueError)):
                    jit.launch(1, 16, 16, pointer, 32, 48)
            for count in (-1, 2**32, False):
                with self.assertRaises((TypeError, ValueError)):
                    jit.tune_hot(count)
            self.assertEqual(self.lib.calls, [])

    def test_json_outputs_are_freed_and_pending_shapes_are_tuples(self):
        with module.AdaptiveGemm() as jit:
            self.assertEqual(jit.stats(1, 16, 16), {"launches": 3, "cache_hit": False})
            self.assertEqual(jit.pending_shapes(), [(256, 128, 64)])
            self.assertEqual(jit.tune_hot(0), [])
            self.lib.json_text = "invalid JSON"
            with self.assertRaises(json.JSONDecodeError):
                jit.stats(1, 16, 16)

    def test_close_error_keeps_handle_for_retry_and_frees_error(self):
        jit = module.AdaptiveGemm()
        self.lib.fail_destroy = True
        with self.assertRaisesRegex(module.AdaptiveJitError, "wrong current context"):
            jit.close()
        self.assertEqual(jit._handle, 1234)
        self.assertEqual(self.lib.strings, {})
        self.lib.fail_destroy = False
        jit.close()
        self.assertIsNone(jit._handle)

    def test_creation_failure_frees_error(self):
        self.lib.fail_create = True
        with self.assertRaisesRegex(module.AdaptiveJitError, "no current context"):
            module.AdaptiveGemm()

    def test_other_thread_is_rejected_before_entering_native_api(self):
        with module.AdaptiveGemm() as jit:
            errors = []

            def wrong_thread():
                for operation in (jit.synchronize, jit.close, jit.pending_shapes):
                    try:
                        operation()
                    except module.AdaptiveJitError as error:
                        errors.append(str(error))

            thread = threading.Thread(target=wrong_thread)
            thread.start()
            thread.join()
            self.assertEqual(len(errors), 3)
            self.assertTrue(all("creating thread" in error for error in errors))
            self.assertEqual(self.lib.calls, [])


if __name__ == "__main__":
    unittest.main()
