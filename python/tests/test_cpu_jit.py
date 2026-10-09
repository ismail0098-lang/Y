"""CPU-only native JIT bindings; no torch, cupy, or GPU is required."""

import ctypes
import gc
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import unittest
from unittest import mock
import weakref

from y_lang import CPUJit, CPUJitError


SOURCE = """
fn id_i8(x: I8) -> I8 { return x; }
fn id_u8(x: U8) -> U8 { return x; }
fn id_char(x: char) -> char { return x; }
fn char_code(x: char) -> I32 { return ychar_to_ascii(x); }
fn high_char() -> char { return 'È'; }
fn id_i16(x: I16) -> I16 { return x; }
fn id_u16(x: U16) -> U16 { return x; }
fn id_i32(x: I32) -> I32 { return x; }
fn id_u32(x: U32) -> U32 { return x; }
fn id_i64(x: I64) -> I64 { return x; }
fn id_u64(x: U64) -> U64 { return x; }
fn id_size(x: usize) -> usize { return x; }
fn invert(x: bool) -> bool { return !x; }
fn single(x: F32) -> F32 { return x; }
fn double(x: F64) -> F64 { return x; }
fn mixed(x: I32, factor: F64, enabled: bool) -> F64 {
    if enabled && x > 0 { return factor * 3.0 + 0.5; }
    return -1.0;
}
fn empty() {}
@unsafe
fn bump(values: &mut [I32; 4]) {
    values[0] = values[0] + 10;
    values[3] = values[3] + 20;
}
fn identity(p: &mut I32) -> &mut I32 { return p; }
struct Pair { a: I32, b: I32, }
fn pair() -> Pair { return Pair { a: 3, b: 4 }; }
fn main() -> I32 { return 23; }
"""

PROFILE_SOURCE = """
fn choose(x: I64) -> I64 {
    if x < 0 { return x - 1; }
    return x + 2;
}
"""


class TestCPUJit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jit = CPUJit(SOURCE)

    @classmethod
    def tearDownClass(cls):
        cls.jit.close()

    def test_metadata_and_mixed_scalar_abi(self):
        metadata = self.jit.signature("mixed")
        self.assertEqual(metadata["parameters"], ["I32", "F64", "bool"])
        self.assertEqual(metadata["return_type"], "F64")
        self.assertTrue(metadata["dynamic_call"])
        self.assertEqual(self.jit("mixed", 3, 1.25, True), 4.25)
        self.assertEqual(self.jit.call("mixed", 3, 1.25, False), -1.0)
        self.assertIsNone(self.jit("empty"))
        self.assertEqual(self.jit("main"), 23)
        metadata["parameters"][0] = "F64"
        self.assertEqual(self.jit.signature("mixed")["parameters"][0], "I32")

    def test_integer_widths_signedness_and_high_bits(self):
        for name, minimum, maximum in [
            ("id_i8", -128, 127), ("id_u8", 0, 255), ("id_char", 0, 255),
            ("id_i16", -32768, 32767), ("id_u16", 0, 65535),
            ("id_i32", -(1 << 31), (1 << 31) - 1), ("id_u32", 0, (1 << 32) - 1),
            ("id_i64", -(1 << 63), (1 << 63) - 1), ("id_u64", 0, (1 << 64) - 1),
            ("id_size", 0, (1 << 64) - 1),
        ]:
            with self.subTest(name=name):
                self.assertEqual(self.jit(name, minimum), minimum)
                self.assertEqual(self.jit(name, maximum), maximum)
                with self.assertRaises(OverflowError):
                    self.jit(name, minimum - 1)
                with self.assertRaises(OverflowError):
                    self.jit(name, maximum + 1)
        self.assertEqual(self.jit("char_code", 200), 200)
        self.assertEqual(self.jit("high_char"), 200)

    def test_float_precision_negative_zero_and_nan(self):
        self.assertEqual(self.jit("single", 1.25), 1.25)
        exact = 1.0 + 2.0 ** -40
        self.assertEqual(self.jit("double", exact), exact)
        for name in ("single", "double"):
            with self.subTest(name=name):
                self.assertEqual(math.copysign(1.0, self.jit(name, -0.0)), -1.0)
                self.assertTrue(math.isnan(self.jit(name, float("nan"))))
                self.assertEqual(self.jit(name, float("inf")), float("inf"))

    def test_ctypes_pointer_arguments_mutate_host_storage(self):
        values = (ctypes.c_int32 * 4)(1, 2, 3, 4)
        self.assertIsNone(self.jit("bump", values))
        self.assertEqual(list(values), [11, 2, 3, 24])
        element = ctypes.c_int32(123)
        for pointer in (ctypes.byref(element), ctypes.pointer(element), ctypes.c_void_p(ctypes.addressof(element)), ctypes.addressof(element)):
            with self.subTest(pointer=pointer):
                returned = self.jit("identity", pointer)
                self.assertIsInstance(returned, ctypes.c_void_p)
                self.assertEqual(returned.value, ctypes.addressof(element))
        self.assertIsNone(self.jit("identity", None).value)

    def test_bad_argument_types_counts_and_aggregates_are_refused(self):
        for call in (
            lambda: self.jit("mixed", 1),
            lambda: self.jit("id_i32", 1.5),
            lambda: self.jit("id_i32", True),
            lambda: self.jit("invert", 1),
            lambda: self.jit("single", "1.0"),
            lambda: self.jit("identity", "a pointer"),
            lambda: self.jit("pair"),
        ):
            with self.assertRaises(TypeError):
                call()
        self.assertFalse(self.jit.signature("pair")["dynamic_call"])
        self.assertEqual(self.jit("invert", True), False)
        self.assertEqual(self.jit("invert", False), True)
        with self.assertRaises(CPUJitError) as error:
            self.jit("missing_function")
        self.assertIn("missing_function", str(error.exception))

    def test_context_close_and_callable_lifetime(self):
        with CPUJit("fn plus(x: I64) -> I64 { return x + 7; }", opt_level=0) as jit:
            function = jit.function("plus")
            self.assertEqual(function(8589934593), 8589934600)
        self.assertTrue(jit.closed)
        jit.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            function(1)
        live = CPUJit("fn plus(x: I64) -> I64 { return x + 7; }")
        reference = weakref.ref(live)
        function = live.function("plus")
        del live
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(function(8), 15)
        del function
        gc.collect()
        self.assertIsNone(reference())

    def test_session_calls_and_close_reject_other_threads(self):
        errors = []

        def worker():
            for operation in (lambda: self.jit("id_i32", 1), self.jit.close,
                              self.jit.branch_profile, self.jit.recompile_profiled,
                              self.jit.compile_timings, self.jit.optimization_timings,
                              self.jit.materialization_timings):
                try:
                    operation()
                except RuntimeError as error:
                    errors.append(str(error))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 7)
        self.assertTrue(all("creating thread" in error for error in errors))
        self.assertEqual(self.jit("id_i32", 17), 17)

    def test_compile_errors_and_input_validation(self):
        with self.assertRaises(CPUJitError) as error:
            CPUJit("fn main() -> I32 { return unknown_cpu_function(); }")
        self.assertIn("unknown_cpu_function", str(error.exception))
        with self.assertRaises(ValueError):
            CPUJit("fn main() {}\0")
        for level in (-1, 4, True, 3.5):
            with self.assertRaises(ValueError):
                CPUJit("fn main() {}", opt_level=level)
        for instrument in (1, "yes", None):
            with self.assertRaises(TypeError):
                CPUJit("fn main() {}", instrument=instrument)
        with self.assertRaises(ValueError):
            self.jit.signature("i32\0ignored")

    def test_instrumented_counts_and_profiled_recompilation_are_explicit(self):
        for opt_level in (0, 3):
            with self.subTest(opt_level=opt_level), CPUJit(
                    PROFILE_SOURCE, opt_level=opt_level, instrument=True) as training:
                empty = training.branch_profile()
                self.assertEqual(empty["total_observations"], 0)
                self.assertRegex(empty["fingerprint"], r"^[0-9a-f]{64}$")
                self.assertEqual(len(empty["sites"]), 1)
                self.assertEqual(empty["sites"][0]["function"], "choose")
                choose = training.function("choose")
                for x in (-3, -1, 0, 2, 5):
                    self.assertEqual(choose(x), x - 1 if x < 0 else x + 2)
                measured = training.branch_profile()
                self.assertEqual(measured["total_observations"], 5)
                self.assertEqual(measured["sites"][0]["true_count"], 2)
                self.assertEqual(measured["sites"][0]["false_count"], 3)
                self.assertEqual(empty["total_observations"], 0)
                # Adopting the compiled handle must not invoke the constructor
                # and accidentally perform a second compilation.
                with mock.patch.object(CPUJit, "__init__", side_effect=AssertionError("duplicate compile")):
                    optimized = training.recompile_profiled()
                self.assertIs(optimized._library, training._library)
                self.assertEqual(optimized.library_path, training.library_path)
                self.assertEqual(optimized._opt_level, opt_level)
                self.assertNotEqual(optimized._handle, training._handle)
                with optimized:
                    self.assertEqual(training.branch_profile(), measured)
                    for x in (-50, -1, 0, 17):
                        self.assertEqual(optimized("choose", x), x - 1 if x < 0 else x + 2)
                    with self.assertRaisesRegex(CPUJitError, "instrumentation"):
                        optimized.branch_profile()
                    self.assertEqual(choose(-9), -10)
                self.assertEqual(choose(10), 12, "closing optimized session preserves training code")
                measured["sites"][0]["true_count"] = 99999
                self.assertEqual(training.branch_profile()["sites"][0]["true_count"], 3)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                training.branch_profile()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                training.recompile_profiled()

    def test_profiled_local_runtime_queries_preserve_mutation_and_lifetimes(self):
        source = """
        @unsafe
        fn scan(n: I64) -> I64 {
            let mut text: String = "";
            let mut values: Vec = Vec_new(1);
            let mut byte: char = 'È';
            let mut i: I64 = 0;
            while i < n {
                String_push(&mut text, 'z');
                Vec_push(&mut values, &byte);
                i = i + 1;
            }
            let mut sum: I64 = 0;
            i = -1;
            while i <= String_len(&text) {
                sum = sum + ychar_to_ascii(String_char_at(&text, i));
                sum = sum + ychar_to_ascii(Vec_get_char(&values, i));
                i = i + 1;
            }
            String_free(&mut text);
            Vec_free(&mut values);
            return sum + String_len(&text) + Vec_len(&values)
                + ychar_to_ascii(String_char_at(&text, 0))
                + ychar_to_ascii(Vec_get_char(&values, 0));
        }
        """
        for opt_level in (0, 3):
            with self.subTest(opt_level=opt_level), CPUJit(
                    source, opt_level=opt_level, instrument=True) as training:
                for n in (-1, 0, 1, 20):
                    self.assertEqual(training("scan", n), max(n, 0) * 322)
                with training.recompile_profiled() as optimized:
                    for n in (0, 2, 100, -4):
                        self.assertEqual(optimized("scan", n), max(n, 0) * 322)

    def test_profiled_bulk_strings_and_dynamic_vectors_preserve_values_and_lifetimes(self):
        source = r"""
        @unsafe
        fn combine(n: I64, width: I32) -> I64 {
            let mut text: String = "";
            let mut piece: String = "";
            String_push(&mut piece, 'È');
            String_push(&mut piece, '\0');
            String_push(&mut piece, 'z');
            let mut values: Vec = Vec_new(width);
            let value: I64 = 200;
            let mut i: I64 = 0;
            while i < n {
                String_push_str(&mut text, &piece);
                Vec_push(&mut values, &value);
                i = i + 1;
            }
            let mut result: I64 = String_len(&text) * 1000 + Vec_len(&values) * 100;
            i = 0;
            while i < String_len(&text) {
                result = result + ychar_to_ascii(String_char_at(&text, i));
                i = i + 1;
            }
            i = 0;
            while i < Vec_len(&values) {
                result = result + ychar_to_ascii(Vec_get_char(&values, i));
                i = i + 1;
            }
            String_free(&mut text);
            String_free(&mut piece);
            Vec_free(&mut values);
            return result + String_len(&text) + Vec_len(&values);
        }
        """
        expected = lambda n, width: max(n, 0) * (3622 if width > 0 else 3322)
        for opt_level in (0, 3):
            with self.subTest(opt_level=opt_level), CPUJit(
                    source, opt_level=opt_level, instrument=True) as training:
                for n, width in ((0, 8), (1, 1), (9, 3), (20, 8), (3, 0), (-1, 8)):
                    self.assertEqual(training("combine", n, width), expected(n, width))
                with training.recompile_profiled() as optimized:
                    for n, width in ((100, 8), (2, 1), (17, 3), (5, -1), (0, 1)):
                        self.assertEqual(optimized("combine", n, width), expected(n, width))

    def test_profiled_and_original_callable_ownership_are_independent(self):
        training = CPUJit(PROFILE_SOURCE, instrument=True)
        old_function = training.function("choose")
        self.assertEqual(old_function(7), 9)
        optimized = training.recompile_profiled()
        new_function = optimized.function("choose")
        training_reference = weakref.ref(training)
        optimized_reference = weakref.ref(optimized)
        del training
        gc.collect()
        self.assertIsNotNone(training_reference())
        self.assertEqual(old_function(-7), -8)
        del old_function
        gc.collect()
        self.assertIsNone(training_reference(), "optimized session must not retain the old session")
        self.assertEqual(new_function(-9), -10, "new code survives old session finalization")
        del optimized
        gc.collect()
        self.assertIsNotNone(optimized_reference())
        self.assertEqual(new_function(19), 21)
        del new_function
        gc.collect()
        self.assertIsNone(optimized_reference())

    def test_compilation_timings_are_owned_complete_and_exclude_calls(self):
        keys = {name + "_ns" for name in (
            "parse", "checks", "lowering", "llvm_setup", "ir_parse", "profile_setup",
            "verification", "optimization", "ir_capture", "symbol_resolution",
            "materialization", "other", "total")}

        def check(session):
            timings = session.compile_timings()
            self.assertEqual(set(timings), keys)
            self.assertTrue(all(type(value) is int and value >= 0 for value in timings.values()))
            self.assertGreater(timings["total_ns"], 0)
            self.assertEqual(sum(value for key, value in timings.items() if key != "total_ns"),
                             timings["total_ns"])
            before = dict(timings)
            self.assertEqual(session("choose", -7), -8)
            self.assertEqual(session.compile_timings(), before)
            timings["total_ns"] = -1
            self.assertEqual(session.compile_timings(), before)

        with CPUJit(PROFILE_SOURCE) as original:
            check(original)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            original.compile_timings()
        with CPUJit(PROFILE_SOURCE, instrument=True) as training:
            self.assertTrue(all(site["true_count"] == site["false_count"] == 0
                                for site in training.branch_profile()["sites"]))
            check(training)
            profile = training.branch_profile()
            with training.recompile_profiled() as optimized:
                check(optimized)
                self.assertEqual(training.branch_profile(), profile)
        error = ctypes.c_void_p()
        result = self.jit._library.y_cpu_jit_compile_timings(None, ctypes.byref(error))
        self.assertFalse(result)
        with self.assertRaisesRegex(CPUJitError, "handle pointer is null"):
            self.jit._raise(error, "expected null handle refusal")

    def test_optimization_timings_are_owned_partitioned_and_session_local(self):
        keys = {"pipeline_ns", "profile_selection_ns", "other_ns", "total_ns"}

        def check(session):
            timings = session.optimization_timings()
            self.assertEqual(set(timings), keys)
            self.assertTrue(all(type(value) is int and value >= 0 for value in timings.values()))
            self.assertGreater(timings["total_ns"], 0)
            self.assertEqual(timings["pipeline_ns"] + timings["profile_selection_ns"] + timings["other_ns"],
                             timings["total_ns"])
            compile_before = session.compile_timings()
            self.assertEqual(timings["total_ns"], compile_before["optimization_ns"])
            before = dict(timings)
            self.assertEqual(session("choose", -7), -8)
            self.assertEqual(session.optimization_timings(), before)
            self.assertEqual(session.compile_timings(), compile_before)
            timings["total_ns"] = -1
            self.assertEqual(session.optimization_timings(), before)
            return before

        for opt_level in (0, 3):
            with self.subTest(opt_level=opt_level):
                with CPUJit(PROFILE_SOURCE, opt_level=opt_level) as original:
                    original_snapshot = check(original)
                    self.assertEqual(original_snapshot["profile_selection_ns"], 0)
                    error = ctypes.c_void_p()
                    owned = original._library.y_cpu_jit_optimization_timings(
                        original._handle, ctypes.byref(error))
                    self.assertTrue(owned)
                    self.assertFalse(error.value)
                try:
                    self.assertEqual(json.loads(ctypes.string_at(owned).decode("utf-8")),
                                     original_snapshot, "owned C snapshot outlives its session")
                finally:
                    original._library.y_free_string(owned)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    original.optimization_timings()
                with CPUJit(PROFILE_SOURCE, opt_level=opt_level, instrument=True) as training:
                    self.assertTrue(all(site["true_count"] == site["false_count"] == 0
                                        for site in training.branch_profile()["sites"]))
                    training_snapshot = check(training)
                    self.assertEqual(training_snapshot["profile_selection_ns"], 0)
                    profile = training.branch_profile()
                    with training.recompile_profiled() as optimized:
                        optimized_snapshot = check(optimized)
                        if opt_level == 0:
                            self.assertEqual(optimized_snapshot["profile_selection_ns"], 0)
                        self.assertEqual(training.optimization_timings(), training_snapshot)
                        self.assertEqual(training.branch_profile(), profile)
                    self.assertEqual(training.optimization_timings(), training_snapshot)

        api = self.jit._library.y_cpu_jit_optimization_timings
        self.assertEqual(api.argtypes, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)])
        self.assertIs(api.restype, ctypes.c_void_p)
        error = ctypes.c_void_p()
        self.assertFalse(api(None, ctypes.byref(error)))
        with self.assertRaisesRegex(CPUJitError, "handle pointer is null"):
            self.jit._raise(error, "expected null optimization handle refusal")
        self.assertFalse(api(None, None))

    def test_materialization_timings_are_owned_partitioned_and_session_local(self):
        parts = {"submission_ns", "first_lookup_ns", "remaining_function_lookups_ns",
                 "profile_lookup_ns", "other_ns"}
        children = {"first_lookup_before_object_ns", "first_lookup_after_object_ns"}
        counters = {"object_count", "object_bytes", "function_lookup_count", "profile_lookup_count"}
        keys = parts | children | counters | {"total_ns", "object_observer_available"}

        def check(session, instrumented=False):
            timings = session.materialization_timings()
            self.assertEqual(set(timings), keys)
            for key in parts | counters | {"total_ns"}:
                self.assertIs(type(timings[key]), int, key)
                self.assertGreaterEqual(timings[key], 0, key)
            self.assertIs(type(timings["object_observer_available"]), bool)
            self.assertGreater(timings["total_ns"], 0)
            self.assertGreater(timings["first_lookup_ns"], 0)
            self.assertGreater(timings["function_lookup_count"], 0)
            self.assertEqual(timings["profile_lookup_count"], int(instrumented))
            self.assertEqual(sum(timings[key] for key in parts), timings["total_ns"])
            compile_before = session.compile_timings()
            optimization_before = session.optimization_timings()
            self.assertEqual(timings["total_ns"], compile_before["materialization_ns"])
            before_object = timings["first_lookup_before_object_ns"]
            after_object = timings["first_lookup_after_object_ns"]
            self.assertEqual(before_object is None, after_object is None)
            if before_object is not None:
                self.assertTrue(timings["object_observer_available"])
                self.assertEqual(timings["object_count"], 1)
                for value in (before_object, after_object):
                    self.assertIs(type(value), int)
                    self.assertGreaterEqual(value, 0)
                self.assertEqual(before_object + after_object, timings["first_lookup_ns"])
            if not timings["object_observer_available"]:
                self.assertEqual(timings["object_count"], 0)
                self.assertEqual(timings["object_bytes"], 0)
                self.assertIsNone(before_object)
            elif timings["object_count"]:
                self.assertGreater(timings["object_bytes"], 0)

            before = dict(timings)
            profile_before = session.branch_profile() if instrumented else None
            self.assertEqual(session.materialization_timings(), before)
            self.assertEqual(session.signature("choose")["parameters"], ["I64"])
            choose = session.function("choose")
            self.assertEqual(session.materialization_timings(), before,
                             "cached signature/function lookups are outside compilation")
            if instrumented:
                self.assertEqual(session.branch_profile(), profile_before,
                                 "timing and signature lookups must not execute source")
            self.assertEqual(choose(-7), -8)
            self.assertEqual(choose(11), 13)
            self.assertEqual(session.materialization_timings(), before)
            self.assertEqual(session.compile_timings(), compile_before)
            self.assertEqual(session.optimization_timings(), optimization_before)
            timings["total_ns"] = -1
            timings["object_count"] = -1
            timings["first_lookup_before_object_ns"] = -1
            self.assertEqual(session.materialization_timings(), before,
                             "each getter returns an independent owned snapshot")
            return before

        for opt_level in (0, 3):
            with self.subTest(opt_level=opt_level):
                with CPUJit(PROFILE_SOURCE, opt_level=opt_level) as original:
                    original_snapshot = check(original)
                    error = ctypes.c_void_p()
                    owned = original._library.y_cpu_jit_materialization_timings(
                        original._handle, ctypes.byref(error))
                    self.assertTrue(owned)
                    self.assertFalse(error.value)
                try:
                    self.assertEqual(json.loads(ctypes.string_at(owned).decode("utf-8")),
                                     original_snapshot, "owned C snapshot outlives its session")
                finally:
                    original._library.y_free_string(owned)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    original.materialization_timings()
                with CPUJit(PROFILE_SOURCE, opt_level=opt_level, instrument=True) as training:
                    self.assertEqual(training.branch_profile()["total_observations"], 0)
                    training_snapshot = check(training, instrumented=True)
                    profile = training.branch_profile()
                    with training.recompile_profiled() as optimized:
                        check(optimized)
                        self.assertEqual(training.materialization_timings(), training_snapshot)
                        self.assertEqual(training.branch_profile(), profile)
                    self.assertEqual(training.materialization_timings(), training_snapshot,
                                     "disposing another session does not change this snapshot")

        api = self.jit._library.y_cpu_jit_materialization_timings
        self.assertEqual(api.argtypes, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)])
        self.assertIs(api.restype, ctypes.c_void_p)
        error = ctypes.c_void_p()
        self.assertFalse(api(None, ctypes.byref(error)))
        with self.assertRaisesRegex(CPUJitError, "handle pointer is null"):
            self.jit._raise(error, "expected null materialization handle refusal")
        self.assertFalse(api(None, None))

    def test_profile_boundary_rejects_changed_source_and_uninstrumented_sessions(self):
        for operation in (self.jit.branch_profile, self.jit.recompile_profiled):
            with self.assertRaisesRegex(CPUJitError, "instrumentation"):
                operation()
        with CPUJit(PROFILE_SOURCE, instrument=True) as training:
            self.assertEqual(training("choose", 4), 6)
            before = training.branch_profile()
            error = ctypes.c_void_p()
            result = training._library.y_cpu_jit_compile_profiled(
                PROFILE_SOURCE.replace("return x + 2", "return x + 7").encode("utf-8"),
                3, training._handle, ctypes.byref(error))
            self.assertFalse(result)
            with self.assertRaisesRegex(CPUJitError, "profile does not match"):
                training._raise(error, "expected stale profile refusal")
            self.assertEqual(training.branch_profile(), before)
            with training.recompile_profiled() as optimized:
                self.assertEqual(optimized("choose", -100), -101)
        with CPUJit("fn answer() -> I32 { return 42; }", instrument=True) as training:
            self.assertEqual(training.branch_profile()["sites"], [])
            with training.recompile_profiled() as optimized:
                self.assertEqual(optimized("answer"), 42)

    def test_cpu_import_does_not_require_tensor_dependencies(self):
        code = """
import sys
import importlib.abc
class RejectTensorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'cupy'):
            raise AssertionError('CPU import attempted optional dependency: ' + fullname)
sys.meta_path.insert(0, RejectTensorImports())
from y_lang import CPUJit
assert CPUJit
"""
        output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(output.returncode, 0, output.stdout + output.stderr)

    def test_declared_c_api_arities_match_rust_exports(self):
        source = (Path(__file__).resolve().parents[2] / "src" / "cpu_jit_ffi.rs")
        # python/tests is two levels below the Rust project.
        text = source.read_text()
        declared = 0
        for match in re.finditer(r'pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+(y_cpu_jit_\w+)\s*\(([^)]*)\)\s*(?:->\s*[^{]+)?\s*\{', text, re.S):
            name, arguments = match.groups()
            function = getattr(self.jit._library, name, None)
            if function is None or function.argtypes is None:
                continue
            arity = len([parameter for parameter in arguments.split(",") if parameter.strip()])
            self.assertEqual(len(function.argtypes), arity, name)
            declared += 1
        self.assertGreaterEqual(declared, 7)


if __name__ == "__main__":
    unittest.main()
