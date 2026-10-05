#!/usr/bin/env python3
"""Exercise audit input binding without invoking Cargo or hardware tools."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from verification_unittest import main as verification_main


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_review_regressions", ROOT / "tools/verify_review_regressions.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditInputs(unittest.TestCase):
    def run_audit(self, directory, change=None, profile_present=True):
        root = directory / "repo"
        (root / "tools").mkdir(parents=True)
        (root / "tests").mkdir()
        (root / "venv/bin").mkdir(parents=True)
        (root / "venv/bin/z3").write_text("test solver")
        runner = root / "tools/verify_review_regressions.py"
        shutil.copy2(ROOT / "tools/verify_review_regressions.py", runner)
        (root / "tests/audit.rs").write_text("regression fixture")
        profile = root / ".ysu_hw_profile"
        if profile_present:
            profile.write_text("SM_VERSION=8.9\n")
        output = directory / "evidence"

        def archive(command, **kwargs):
            self.assertEqual(command[:2], ["git", "archive"])
            with tarfile.open(fileobj=kwargs["stdout"], mode="w") as tree:
                tests = tarfile.TarInfo("tests")
                tests.type = tarfile.DIRTYPE
                tree.addfile(tests)
            return subprocess.CompletedProcess(command, 0)

        def run_tests(checkout, binaries, logs, env):
            if checkout.name == "original" and change:
                change(root, output / "original", runner)
            return {
                "audit::control": {"status": "pass", "log": "control.log"},
                "audit::regression": {
                    "status": "pass" if checkout == root else "fail",
                    "log": "regression.log",
                },
            }

        with mock.patch.object(AUDIT, "__file__", str(runner)), \
                mock.patch.object(AUDIT, "TARGETS", ("audit",)), \
                mock.patch.object(AUDIT, "CONTROLS", {"audit::control"}), \
                mock.patch.object(AUDIT, "source_hashes", return_value={"src/lib.rs": "unchanged"}), \
                mock.patch.object(AUDIT, "capture", return_value=""), \
                mock.patch.object(AUDIT.subprocess, "run", side_effect=archive), \
                mock.patch.object(AUDIT, "build", return_value={}), \
                mock.patch.object(AUDIT, "run_tests", side_effect=run_tests), \
                mock.patch.dict(AUDIT.os.environ, {"Y_Z3_PATH": str(root / "venv/bin/z3")}), \
                mock.patch.object(sys, "argv", ["audit", "--output", str(output)]), \
                contextlib.redirect_stdout(io.StringIO()):
            AUDIT.main()
        return root, output

    def test_stable_inputs_preserve_the_tested_hashes(self):
        with tempfile.TemporaryDirectory(prefix="y-audit-inputs-") as directory:
            root, output = self.run_audit(Path(directory))
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(report["profile_sha256"], AUDIT.sha256(root / ".ysu_hw_profile"))
            self.assertEqual(report["runner_sha256"], AUDIT.sha256(root / "tools/verify_review_regressions.py"))
            self.assertEqual(report["transitions"]["audit::control"], "pass -> pass")
            self.assertEqual(report["transitions"]["audit::regression"], "fail -> pass")

    def test_profile_changes_in_either_checkout_fail_the_audit(self):
        for checkout in ("working", "original"):
            with self.subTest(checkout=checkout), \
                    tempfile.TemporaryDirectory(prefix="y-audit-profile-") as directory:
                def change(root, original, runner):
                    path = root if checkout == "working" else original
                    (path / ".ysu_hw_profile").write_text("SM_VERSION=9.0\n")

                with self.assertRaisesRegex(RuntimeError, "Hardware profile changed"):
                    self.run_audit(Path(directory), change)
                self.assertFalse((Path(directory) / "evidence/results.json").exists())

    def test_profile_creation_or_removal_fails_the_audit(self):
        for initially_present in (False, True):
            with self.subTest(initially_present=initially_present), \
                    tempfile.TemporaryDirectory(prefix="y-audit-profile-") as directory:
                def change(root, original, runner):
                    profile = root / ".ysu_hw_profile"
                    if initially_present:
                        profile.unlink()
                    else:
                        profile.write_text("SM_VERSION=9.0\n")

                with self.assertRaisesRegex(RuntimeError, "Hardware profile changed"):
                    self.run_audit(Path(directory), change, initially_present)

    def test_runner_changes_cannot_be_reported_as_the_executed_runner(self):
        with tempfile.TemporaryDirectory(prefix="y-audit-runner-") as directory:
            def change(root, original, runner):
                with runner.open("a") as stream:
                    stream.write("\n# concurrent edit\n")

            with self.assertRaisesRegex(RuntimeError, "Audit runner changed"):
                self.run_audit(Path(directory), change)


if __name__ == "__main__":
    verification_main()
