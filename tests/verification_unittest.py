"""Optional structured results for Python verification suites.

The ordinary unittest command line remains available.  Set
Y_VERIFICATION_RESULT_FILE, or Y_VERIFICATION_RESULT_DIR for a unique filename,
to publish an atomic JSON sidecar.  Y_VERIFICATION_STRICT=1 also makes any
skipped check fail the process without relabeling it as an assertion failure.
"""

from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


FORMAT = "y-verification-unittest-v1"
STATUSES = ("pass", "fail", "error", "skip", "expected_failure", "unexpected_success")


def _cases(suite):
    """Read the suite before unittest consumes and clears its child suites."""
    if isinstance(suite, unittest.TestSuite):
        for child in suite:
            yield from _cases(child)
    else:
        yield suite


class VerificationResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity, *, planned, strict):
        super().__init__(stream, descriptions, verbosity)
        self.strict = strict
        self.planned = [
            (case.id(), case.__class__.__module__, case.__class__.__qualname__)
            for case in planned
        ]
        self.records = []
        self.active = {}

    def startTest(self, test):
        super().startTest(test)
        record = {"id": test.id(), "status": "pass"}
        self.records.append(record)
        self.active[id(test)] = record

    def stopTest(self, test):
        self.active.pop(id(test), None)
        super().stopTest(test)

    def _record(self, test, status, **details):
        record = self.active.get(id(test))
        if record is not None:
            # Cleanup can itself skip after a body failure. Preserve both the
            # real failure and the skipped check instead of concealing either.
            if status == "skip" and record["status"] in ("fail", "error"):
                self.records.append({"id": test.id() + " [cleanup skip]", "status": status, **details})
                return
            # A later failing subtest cannot conceal an earlier runtime error.
            priorities = {"pass": 0, "skip": 1, "expected_failure": 2,
                          "unexpected_success": 3, "fail": 4, "error": 5}
            if priorities[status] >= priorities[record["status"]]:
                record["status"] = status
            if "detail" in details and "detail" in record:
                details["detail"] = record["detail"] + "\n" + details["detail"]
            record.update(details)
            return

        # unittest normally reports a setUpClass/setUpModule SkipTest as one
        # synthetic fixture event, with testsRun == 0.  Record each check that
        # this fixture prevented so the verification report exposes the gap.
        holder = test.id()
        reported = Counter(record["id"] for record in self.records)
        expanded = False
        for test_id, module, class_name in self.planned:
            if holder not in (f"setUpClass ({module}.{class_name})", f"setUpModule ({module})"):
                continue
            if reported[test_id]:
                reported[test_id] -= 1
                continue
            self.records.append({"id": test_id, "status": status, **details})
            expanded = True
        if not expanded:
            self.records.append({"id": holder, "status": status, **details})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "fail", detail=self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "error", detail=self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "skip", reason=str(reason))

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record(test, "expected_failure", detail=self._exc_info_to_string(err, test))

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, "unexpected_success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            status = "fail" if issubclass(err[0], test.failureException) else "error"
            self._record(test, status, detail=self._exc_info_to_string(err, subtest))

    @property
    def complete(self):
        reported = Counter(record["id"] for record in self.records)
        planned = Counter(test_id for test_id, _, _ in self.planned)
        return all(reported[test_id] >= count for test_id, count in planned.items())

    def wasSuccessful(self):
        return super().wasSuccessful() and self.complete and not (self.strict and self.skipped)

    def report(self, suite):
        counts = {status: 0 for status in STATUSES}
        for record in self.records:
            counts[record["status"]] += 1
        counts["tests"] = len(self.records)
        successful = bool(self.wasSuccessful())
        return {
            "format": FORMAT,
            "suite": suite,
            "strict": self.strict,
            "complete": self.complete,
            "successful": successful,
            "exit_code": 0 if successful else 1,
            "unittest_tests_run": self.testsRun,
            "counts": counts,
            "tests": self.records,
        }


def _publish(path, report):
    """Replace a sidecar only after the complete JSON has reached disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class VerificationRunner(unittest.TextTestRunner):
    def __init__(self, *args, suite_name, result_file=None, strict=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.suite_name = suite_name
        self.result_file = result_file
        self.strict = strict
        self.planned = []

    def _makeResult(self):
        return VerificationResult(
            self.stream, self.descriptions, self.verbosity,
            planned=self.planned, strict=self.strict,
        )

    def run(self, test):
        self.planned = list(_cases(test))
        result = super().run(test)
        if self.result_file:
            _publish(self.result_file, result.report(self.suite_name))
        return result


def main(**kwargs):
    """Run unittest, preserving its CLI and exit behavior unless opted in."""
    result_file = os.environ.get("Y_VERIFICATION_RESULT_FILE")
    result_dir = os.environ.get("Y_VERIFICATION_RESULT_DIR")
    strict = os.environ.get("Y_VERIFICATION_STRICT") == "1"
    if not (result_file or result_dir or strict):
        return unittest.main(**kwargs)

    module = kwargs.get("module", "__main__")
    if isinstance(module, str):
        module = sys.modules.get(module)
    suite_name = Path(getattr(module, "__file__", sys.argv[0])).stem
    if not result_file and result_dir:
        result_file = Path(result_dir) / f"{suite_name}-{os.getpid()}.json"
    class ConfiguredRunner(VerificationRunner):
        def __init__(self, *runner_args, **runner_kwargs):
            super().__init__(
                *runner_args, suite_name=suite_name, result_file=result_file,
                strict=strict, **runner_kwargs,
            )

    # Pass a class so TestProgram applies parsed --failfast/--buffer/verbosity
    # options just as it does for the ordinary TextTestRunner.
    kwargs["testRunner"] = ConfiguredRunner
    return unittest.main(**kwargs)
