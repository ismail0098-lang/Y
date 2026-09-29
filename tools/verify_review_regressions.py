#!/usr/bin/env python3
"""Run identical regression tests on the reviewed revision and working tree.

Requires offline Cargo dependencies, clang, ptxas, and Z3. Keeps independent
builds, per-test logs, test hashes, and machine-readable results in /tmp.
An original-source build failure is an audit failure, not a reproduced bug.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile


BASELINE = "8b7122ab6cb78c5224e9d22c2d10aa41ca9dab50"
TARGETS = (
    "compile_time_assertions",
    "bounds_control_flow",
    "frontend_semantics",
    "integer_literal_diagnostics",
    "llvm_zero_drift_integer",
    "backend_function_state",
    "zk_field_context",
)
CONTROLS = {
    "compile_time_assertions::true_assertions_are_evaluated_and_erased",
    "frontend_semantics::valid_forward_calls_polymorphic_literals_and_struct_fields_work",
    "integer_literal_diagnostics::ordinary_decimal_floats_and_range_tokens_still_work",
    "integer_literal_diagnostics::representable_nonnegative_integers_are_preserved",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command, cwd, env=None):
    return subprocess.run(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=True,
    ).stdout


def source_hashes(root):
    names = capture(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard",
         "--", "src", "crates/y-gpu", "Cargo.toml", "Cargo.lock"], root
    ).splitlines()
    return {name: sha256(root / name) for name in sorted(set(names))
            if (root / name).is_file()}


def build(root, output, env):
    command = ["cargo", "test", "--offline", "--locked", "--features", "zk",
               "--no-run", "--message-format=json"]
    for target in TARGETS:
        command.extend(["--test", target])
    print(f"Building {root}", flush=True)
    with (output / "build.jsonl").open("w") as stdout, \
            (output / "build.log").open("w") as stderr:
        result = subprocess.run(command, cwd=root, env=env,
                                stdout=stdout, stderr=stderr)
    if result.returncode:
        raise RuntimeError(f"Build failed; inspect {output / 'build.log'}")
    binaries = {}
    for line in (output / "build.jsonl").read_text().splitlines():
        item = json.loads(line)
        if (item.get("reason") == "compiler-artifact"
                and item.get("executable")
                and item["target"]["name"] in TARGETS
                and "test" in item["target"]["kind"]):
            binaries[item["target"]["name"]] = item["executable"]
    if set(binaries) != set(TARGETS):
        raise RuntimeError(f"Missing regression binaries: {set(TARGETS) - set(binaries)}")
    return binaries


def run_tests(root, binaries, output, env):
    results = {}
    for target in TARGETS:
        listing = capture([binaries[target], "--list", "--format=terse"], root, env)
        cases = [line.removesuffix(": test") for line in listing.splitlines()
                 if line.endswith(": test")]
        if not cases:
            raise RuntimeError(f"No cases in {target}")
        for case in cases:
            command = [binaries[target], "--exact", case, "--nocapture", "--test-threads=1"]
            result = subprocess.run(command, cwd=root, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=120)
            log = output / f"{target}--{case}.log"
            log.write_text(result.stdout)
            summaries = re.findall(
                r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;",
                result.stdout,
            )
            expected = ("1", "0", "0") if result.returncode == 0 else ("0", "1", "0")
            if (summaries != [expected]
                    or "SKIP execution:" in result.stdout
                    or "SKIP assembly:" in result.stdout):
                raise RuntimeError(f"Case skipped or did not complete normally: {log}")
            status = "pass" if result.returncode == 0 else "fail"
            results[f"{target}::{case}"] = {"status": status, "log": str(log)}
            print(f"  {status.upper()} {target}::{case}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory for all audit artifacts")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="y-regression-audit-"))
    if args.output:
        output.mkdir(parents=True, exist_ok=False)
    print(f"Audit artifacts: {output}", flush=True)
    env = os.environ.copy()
    env.pop("Y_ALLOW_UNVERIFIED_INVARIANTS", None)
    env.pop("CARGO_TARGET_DIR", None)
    env["RUST_BACKTRACE"] = "0"
    z3 = env.get("Y_Z3_PATH") or str(root / "venv/bin/z3")
    if not Path(z3).is_file():
        z3 = shutil.which("z3")
    if not z3:
        raise RuntimeError("Z3 is required: set Y_Z3_PATH")
    env["Y_Z3_PATH"] = str(Path(z3).resolve())
    versions = {}
    for tool, flag in (("cargo", "--version"), ("rustc", "--version"),
                       ("clang", "--version"), ("ptxas", "--version"),
                       (env["Y_Z3_PATH"], "--version")):
        versions[tool] = capture([tool, flag], root, env).strip()
    before_hashes = source_hashes(root)
    test_hashes = {target: sha256(root / "tests" / f"{target}.rs") for target in TARGETS}
    baseline = output / "original"
    baseline.mkdir()
    archive = output / "original.tar"
    with archive.open("wb") as stream:
        subprocess.run(["git", "archive", "--format=tar", BASELINE],
                       cwd=root, stdout=stream, check=True)
    with tarfile.open(archive) as tree:
        tree.extractall(baseline, filter="data")
    for target in TARGETS:
        shutil.copy2(root / "tests" / f"{target}.rs", baseline / "tests")
        assert sha256(baseline / "tests" / f"{target}.rs") == test_hashes[target]
    profile = root / ".ysu_hw_profile"
    if profile.exists():
        shutil.copy2(profile, baseline)
    (output / "production.patch").write_text(capture(
        ["git", "diff", "--binary", BASELINE, "--", "src", "crates/y-gpu",
         "Cargo.toml", "Cargo.lock"], root,
    ))
    # Git diff omits untracked sources; preserve those separately as well.
    for name in capture(["git", "ls-files", "--others", "--exclude-standard",
                         "--", "src", "crates/y-gpu"], root).splitlines():
        destination = output / "untracked-production" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    results = {}
    for label, checkout in (("original", baseline), ("patched", root)):
        logs = output / f"{label}-logs"
        logs.mkdir()
        binaries = build(checkout, logs, env)
        results[label] = run_tests(checkout, binaries, logs, env)
    if before_hashes != source_hashes(root):
        raise RuntimeError("Compiler source changed during the audit")
    if test_hashes != {target: sha256(root / "tests" / f"{target}.rs") for target in TARGETS}:
        raise RuntimeError("Regression source changed during the audit")
    if results["original"].keys() != results["patched"].keys():
        raise RuntimeError("The two revisions ran different test cases")
    transitions = {
        case: f"{results['original'][case]['status']} -> {results['patched'][case]['status']}"
        for case in results["original"]
    }
    report = {"baseline": BASELINE, "tools": versions, "test_sha256": test_hashes,
              "runner_sha256": sha256(Path(__file__)),
              "source_sha256": before_hashes,
              "profile_sha256": sha256(profile) if profile.exists() else None,
              "results": results, "transitions": transitions}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    counts = {transition: list(transitions.values()).count(transition)
              for transition in sorted(set(transitions.values()))}
    print(json.dumps(counts, indent=2), flush=True)
    if not CONTROLS.issubset(transitions):
        raise RuntimeError("A required compatibility control is missing")
    for case, status in transitions.items():
        expected = "pass -> pass" if case in CONTROLS else "fail -> pass"
        if status != expected:
            raise RuntimeError(f"{case}: expected {expected}, observed {status}")
    print(f"Audit complete: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
