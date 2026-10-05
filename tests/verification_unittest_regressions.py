#!/usr/bin/env python3
"""Check verification sidecars and skip policy without external toolchains."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import verification_unittest as reporting


class StructuredResults(unittest.TestCase):
    def run_cases(self, cases, *, strict=False):
        runner = reporting.VerificationRunner(
            stream=io.StringIO(), suite_name="fixture", verbosity=0, strict=strict,
        )
        result = runner.run(unittest.TestSuite(cases))
        report = result.report("fixture")
        self.assertEqual(report["format"], reporting.FORMAT)
        self.assertEqual(sum(report["counts"][status] for status in reporting.STATUSES),
                         report["counts"]["tests"])
        self.assertEqual(len(report["tests"]), report["counts"]["tests"])
        return report

    def test_class_fixture_skips_expose_every_prevented_check(self):
        class Fixture(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise unittest.SkipTest("missing solver")

            def test_one(self):
                self.fail("a skipped body must not run")

            def test_two(self):
                self.fail("a skipped body must not run")

        for strict in (False, True):
            with self.subTest(strict=strict):
                report = self.run_cases(unittest.defaultTestLoader.loadTestsFromTestCase(Fixture),
                                        strict=strict)
                self.assertEqual(report["counts"]["skip"], 2)
                self.assertEqual(report["unittest_tests_run"], 0)
                self.assertTrue(report["complete"])
                self.assertEqual(report["exit_code"], int(strict))
                self.assertEqual(report["successful"], not strict)
                self.assertTrue(all(test["reason"] == "missing solver" for test in report["tests"]))
                self.assertTrue(all(test["id"].endswith(("test_one", "test_two"))
                                    for test in report["tests"]))

    def test_class_fixture_errors_are_errors_even_without_started_tests(self):
        class Fixture(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise RuntimeError("broken initialization")

            def test_one(self):
                pass

            def test_two(self):
                pass

        report = self.run_cases(unittest.defaultTestLoader.loadTestsFromTestCase(Fixture))
        self.assertEqual(report["counts"]["error"], 2)
        self.assertEqual(report["counts"]["skip"], 0)
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(report["unittest_tests_run"], 0)
        self.assertTrue(report["complete"])
        self.assertIn("RuntimeError: broken initialization", report["tests"][0]["detail"])

    def test_module_fixture_skip_exposes_checks_in_each_class(self):
        module_name = "y_verification_skipped_fixture"
        module = types.ModuleType(module_name)

        def skip_module():
            raise unittest.SkipTest("module tool missing")

        module.setUpModule = skip_module

        class First(unittest.TestCase):
            def test_one(self):
                self.fail("a skipped body must not run")

        class Second(unittest.TestCase):
            def test_two(self):
                self.fail("a skipped body must not run")

        First.__module__ = Second.__module__ = module_name
        with mock.patch.dict(sys.modules, {module_name: module}):
            report = self.run_cases([First("test_one"), Second("test_two")], strict=True)
        self.assertTrue(report["complete"])
        self.assertEqual(report["unittest_tests_run"], 0)
        self.assertEqual(report["counts"]["skip"], 2)
        self.assertEqual(report["exit_code"], 1)
        self.assertTrue(all(test["reason"] == "module tool missing" for test in report["tests"]))

    def test_subtest_error_cannot_be_hidden_by_later_failure_or_skip(self):
        class Fixture(unittest.TestCase):
            def test_subtests(self):
                with self.subTest(check="error"):
                    raise RuntimeError("bad fixture")
                with self.subTest(check="failure"):
                    self.fail("wrong answer")
                with self.subTest(check="skip"):
                    self.skipTest("missing optional check")

        report = self.run_cases([Fixture("test_subtests")], strict=True)
        self.assertEqual(report["counts"]["error"], 1)
        self.assertEqual(report["counts"]["skip"], 1)
        self.assertEqual(report["unittest_tests_run"], 1)
        self.assertIn("bad fixture", report["tests"][0]["detail"])
        self.assertIn("wrong answer", report["tests"][0]["detail"])
        self.assertEqual(report["exit_code"], 1)

    def test_cleanup_skip_preserves_the_body_failure(self):
        class Fixture(unittest.TestCase):
            def test_failure(self):
                self.fail("wrong answer")

            def tearDown(self):
                self.skipTest("cleanup unavailable")

        report = self.run_cases([Fixture("test_failure")], strict=True)
        self.assertEqual(report["counts"]["fail"], 1)
        self.assertEqual(report["counts"]["skip"], 1)
        self.assertEqual(report["exit_code"], 1)

    def test_expected_failure_and_unexpected_success_are_distinguished(self):
        class Fixture(unittest.TestCase):
            @unittest.expectedFailure
            def test_expected(self):
                self.fail("known failure")

            @unittest.expectedFailure
            def test_unexpected(self):
                pass

        expected = self.run_cases([Fixture("test_expected")])
        self.assertEqual(expected["counts"]["expected_failure"], 1)
        self.assertEqual(expected["exit_code"], 0)
        unexpected = self.run_cases([Fixture("test_unexpected")])
        self.assertEqual(unexpected["counts"]["unexpected_success"], 1)
        self.assertEqual(unexpected["exit_code"], 1)

    def test_failed_atomic_publication_keeps_the_previous_file(self):
        with tempfile.TemporaryDirectory(prefix="y-sidecar-atomic-") as directory:
            path = Path(directory) / "result.json"
            path.write_text("previous completed result")
            with mock.patch.object(reporting.os, "replace", side_effect=OSError("write refused")):
                with self.assertRaisesRegex(OSError, "write refused"):
                    reporting._publish(path, {"new": True})
            self.assertEqual(path.read_text(), "previous completed result")
            self.assertEqual(list(path.parent.iterdir()), [path])


class CommandLineResults(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="y-sidecar-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.script = self.directory / "fixture.py"
        self.script.write_text(
            "import unittest\n"
            "from verification_unittest import main\n"
            "class Fixture(unittest.TestCase):\n"
            "    def test_a_failure(self): self.fail('wrong answer')\n"
            "    def test_b_pass(self): pass\n"
            "    def test_c_skip(self): self.skipTest('missing tool')\n"
            "if __name__ == '__main__': main()\n"
        )

    def invoke(self, *arguments, environment=None):
        env = dict(os.environ)
        for key in ("Y_VERIFICATION_RESULT_FILE", "Y_VERIFICATION_RESULT_DIR", "Y_VERIFICATION_STRICT"):
            env.pop(key, None)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
        env.update(environment or {})
        return subprocess.run(
            [sys.executable, str(self.script), *arguments], env=env,
            capture_output=True, text=True,
        )

    def test_default_skip_behavior_and_selection_are_unchanged(self):
        completed = self.invoke("Fixture.test_c_skip")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("OK (skipped=1)", completed.stderr)
        self.assertEqual(list(self.directory.iterdir()), [self.script])

    def test_strict_mode_fails_skips_without_a_sidecar(self):
        completed = self.invoke("Fixture.test_c_skip", environment={"Y_VERIFICATION_STRICT": "1"})
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIn("skipped=1", completed.stderr)

    def test_result_file_precedes_directory_and_preserves_skip_reason(self):
        path = self.directory / "nested" / "explicit.json"
        result_dir = self.directory / "unused"
        completed = self.invoke("Fixture.test_c_skip", environment={
            "Y_VERIFICATION_RESULT_FILE": str(path),
            "Y_VERIFICATION_RESULT_DIR": str(result_dir),
            "Y_VERIFICATION_STRICT": "1",
        })
        self.assertEqual(completed.returncode, 1, completed.stderr)
        report = json.loads(path.read_text())
        self.assertEqual(report["suite"], "fixture")
        self.assertEqual(report["exit_code"], completed.returncode)
        self.assertEqual(report["tests"], [{
            "id": "__main__.Fixture.test_c_skip", "status": "skip", "reason": "missing tool",
        }])
        self.assertFalse(result_dir.exists())

    def test_result_directory_gives_each_process_a_distinct_sidecar(self):
        result_dir = self.directory / "results"
        for _ in range(2):
            completed = self.invoke("Fixture.test_b_pass", environment={
                "Y_VERIFICATION_RESULT_DIR": str(result_dir),
            })
            self.assertEqual(completed.returncode, 0, completed.stderr)
        paths = list(result_dir.glob("fixture-*.json"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            report = json.loads(path.read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["counts"]["tests"], 1)
            self.assertEqual(report["counts"]["pass"], 1)
            self.assertEqual(report["unittest_tests_run"], 1)

    def test_failfast_cli_exposes_incomplete_execution(self):
        path = self.directory / "failfast.json"
        completed = self.invoke("--failfast", "Fixture.test_a_failure", "Fixture.test_b_pass",
                                environment={"Y_VERIFICATION_RESULT_FILE": str(path)})
        self.assertEqual(completed.returncode, 1, completed.stderr)
        report = json.loads(path.read_text())
        self.assertFalse(report["complete"])
        self.assertEqual(report["unittest_tests_run"], 1)
        self.assertEqual(report["counts"]["fail"], 1)
        self.assertEqual(report["exit_code"], 1)


if __name__ == "__main__":
    reporting.main(verbosity=2)
