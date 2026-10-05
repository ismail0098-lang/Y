#!/usr/bin/env python3
"""Run Y's verification gates and retain complete, input-bound evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time

from ptxas_tval.validation_child import VALIDATOR_CHILD
from ptxas_tval.ptxsource import strip_comments


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "y-verification-workflow-v1"
PYTHON_FORMAT = "y-verification-unittest-v1"
PYTHON_Z3_PROBE = "import sys,z3; print(sys.version); print(z3.get_version_string())"
TEST_STATUSES = ("pass", "fail", "error", "skip", "expected_failure", "unexpected_success")
SUMMARY = re.compile(r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SKIP = re.compile(r"^(?:test \S+ \.\.\.\s*)?(?:SKIP(?:PED)?\b|skipping\b|"
                  r"note:.*\bskipping\b|ptxas not found;\s*skipping\b)", re.I)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes(root, evidence=None):
    """Include untracked inputs, excluding generated caches and build output."""
    files = [root / name for name in ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain", "rust-toolchain.toml",
                                      "README.md", ".ysu_hw_profile", ".ysu_exact_gemm")]
    for directory in ("src", "tests", "proofs", "tools", "crates", "c_src", "python", ".cargo",
                      "self_hosted", "circomlib", "scripts", "algorithms", "docs"):
        for base, directories, names in os.walk(root / directory):
            directories[:] = sorted(name for name in directories
                                    if name not in {"target", "__pycache__", ".git", ".ysu"}
                                    and (evidence is None or (Path(base) / name).resolve() != evidence.resolve()))
            files.extend(Path(base) / name for name in names
                         if not name.endswith((".pyc", ".vo", ".vos", ".vok", ".glob", ".aux")))
    return {str(path.relative_to(root)): sha256(path)
            for path in sorted(files) if path.is_file()}


def executable(candidate, root):
    if os.path.dirname(str(candidate)):
        path = Path(candidate)
        path = path if path.is_absolute() else root / path
        return str(path.absolute()) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(str(candidate))


def probe(candidates, arguments, root, env):
    errors = []
    for candidate in candidates:
        path = executable(candidate, root)
        if path is None:
            errors.append(f"{candidate}: executable unavailable")
            continue
        try:
            output = subprocess.run([path, *arguments], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"{candidate}: {error}")
            continue
        if output.returncode == 0:
            return {"available": True, "path": path,
                    "version": (output.stdout + output.stderr).strip()}
        errors.append(f"{candidate}: exit {output.returncode}")
    return {"available": False, "reason": "; ".join(errors)}


def discover(root, env):
    tools = {name: probe([name], [flag], root, env) for name, flag in (
        ("cargo", "--version"), ("rustc", "--version"), ("clang", "--version"),
        ("coqc", "--version"), ("ptxas", "--version"), ("nvdisasm", "--version"))}
    python = ([env["Y_TVAL_PYTHON"]] if "Y_TVAL_PYTHON" in env else
              ["venv/bin/python", ".venv/bin/python", sys.executable, "python3"])
    tools["python"] = probe(python, ["--version"], root, env)
    tools["python_z3"] = probe(python, ["-c", "import sys,z3; print(sys.version.split()[0], z3.get_version_string())"], root, env)
    z3 = ([env["Y_Z3_PATH"]] if "Y_Z3_PATH" in env else
          ["z3", "venv/bin/z3", ".venv/bin/z3", "z3/build/z3"])
    tools["z3"] = probe(z3, ["-version"], root, env)
    return tools


def plan(cargo, full=False):
    base = [cargo, "test", "--offline", "--locked", "--features", "zk"]
    stages = [
        ("smt", ("smt_machine_arithmetic", "smt_runtime_loop_bounds", "exact_gemm_signed_dimensions",
                 "smt_reference_aliases", "safe_invariant_enforcement",
                 "reference_types_are_checked", "type_checker_scalar_rules", "bounds_enforcement",
                 "linear_tracker_enforcement"), ("cargo", "rustc", "z3", "clang", "ptxas"), ()),
        ("proofs", ("proofs_are_checked", "proof_mutation_checks"), ("cargo", "rustc", "coqc"), ()),
        ("translation", ("translation_validator_soundness",),
         ("cargo", "rustc", "python_z3"), ("translation_validator_soundness",)),
        ("ptxas", ("ptx_portability", "ptx_intrinsics_assemble", "coprocessor_ptx_assembles",
                   "zk_ptx_witness_refuses", "ptx_control_flow", "ptxas_verification"),
         ("cargo", "rustc", "z3", "clang", "python_z3", "ptxas", "nvdisasm"),
         ("ptxas_pipeline_regressions", "ptxas_validator_regressions",
          "ptxas_integer_abstraction_regressions", "ptxas_integer_semantics_regressions",
          "ptxas_memory_regressions", "ptxas_domain_regressions", "ptxas_architecture_regressions",
          "ptxas_directive_regressions",
          "ptxas_loop_control_regressions", "ptxas_nested_effect_regressions")),
        ("artifacts", ("exact_pv_artifact_binding", "exact_pv_launch_contract"),
         ("cargo", "rustc", "python_z3", "ptxas", "nvdisasm"),
         ("verified_exact_pv_artifact", "exact_pv_launch_contract")),
        ("launch", (), ("cargo", "rustc"), ()),
        ("workflow", ("verification_workflow_regressions", "verification_strictness", "verification_command"),
         ("cargo", "rustc", "python"), ("verification_workflow_regressions", "verification_command_tests",
                                        "verification_unittest_regressions")),
    ]
    result = []
    for name, targets, requirements, suites in stages:
        command = base[:]
        if name == "launch":
            command += ["--lib", "cuda_runtime::tests"]
        for target in targets:
            command += ["--test", target]
        command += ["--no-fail-fast", "--", "--nocapture", "--test-threads=1", "--color", "never"]
        result.append({"name": name, "command": command, "requires": list(requirements),
                       "python_suites": list(suites)})
    if full:
        result.append({"name": "workspace", "command": base + ["--workspace", "--tests", "--no-fail-fast",
                       "--", "--nocapture", "--test-threads=1", "--color", "never"],
                       "requires": ["cargo", "rustc"],
                       "python_suites": ["translation_validator_soundness", "verified_exact_pv_artifact",
                                         "verification_workflow_regressions", "verification_command_tests",
                                         "verification_unittest_regressions", "ptxas_pipeline_regressions",
                                         "ptxas_validator_regressions", "ptxas_integer_abstraction_regressions",
                                         "ptxas_integer_semantics_regressions", "ptxas_memory_regressions",
                                         "ptxas_domain_regressions", "ptxas_architecture_regressions",
                                         "ptxas_directive_regressions",
                                         "exact_pv_launch_contract", "ptxas_loop_control_regressions",
                                         "ptxas_nested_effect_regressions"]})
    return result


def run_command(command, root, env, log, timeout):
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        try:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            return {"exit_code": None, "error": str(error), "duration_seconds": time.monotonic() - started}
        try:
            child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            # All descendant compilers/solvers belong to this private group.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            if isinstance(error, KeyboardInterrupt):
                raise
            return {"exit_code": child.returncode, "error": f"stage exceeded {timeout}s deadline",
                    "duration_seconds": time.monotonic() - started}
    return {"exit_code": child.returncode, "duration_seconds": time.monotonic() - started}


def rust_results(text):
    text = ANSI.sub("", text)
    summaries = SUMMARY.findall(text)
    return {"targets": len(summaries),
            "pass": sum(int(row[0]) for row in summaries),
            "fail": sum(int(row[1]) for row in summaries),
            "ignored": sum(int(row[2]) for row in summaries),
            "skip_notices": [line for line in text.splitlines() if SKIP.match(line.strip())]}


def python_results(directory, expected, strict=None):
    results = []
    for path in sorted(directory.glob("*.json")):
        def no_duplicates(pairs):
            value = dict(pairs)
            if len(value) != len(pairs):
                raise ValueError(f"duplicate JSON field in {path.name}")
            return value
        data = json.loads(path.read_text(), object_pairs_hook=no_duplicates)
        if (type(data) is not dict or data.get("format") != PYTHON_FORMAT
                or type(data.get("suite")) is not str or not data["suite"]
                or type(data.get("tests")) is not list or type(data.get("counts")) is not dict
                or type(data.get("strict")) is not bool or type(data.get("successful")) is not bool
                or type(data.get("complete")) is not bool
                or type(data.get("unittest_tests_run")) is not int or data["unittest_tests_run"] < 0
                or type(data.get("exit_code")) is not int or data["exit_code"] not in (0, 1)):
            raise ValueError(f"invalid Python result schema: {path.name}")
        if strict is not None and data["strict"] != strict:
            raise ValueError(f"Python result strictness differs from the requested mode: {path.name}")
        if data["suite"] not in expected:
            raise ValueError(f"unexpected Python verification suite: {data['suite']}")
        actual = {status: 0 for status in TEST_STATUSES}
        for test in data["tests"]:
            if (type(test) is not dict or type(test.get("id")) is not str
                    or type(test.get("status")) is not str
                    or test.get("status") not in actual
                    or ("reason" in test and type(test["reason"]) is not str)):
                raise ValueError(f"invalid Python test record: {path.name}")
            actual[test["status"]] += 1
        counts = data["counts"]
        wanted = {"tests": len(data["tests"]), **actual}
        if (counts != wanted or any(type(count) is not int for count in counts.values())):
            raise ValueError(f"inconsistent Python test counts: {path.name}")
        failure = actual["fail"] + actual["error"] + actual["unexpected_success"]
        successful = not bool(failure or not data["complete"] or data["strict"] and actual["skip"])
        if data["exit_code"] != int(not successful):
            raise ValueError(f"inconsistent Python exit status: {path.name}")
        if data["successful"] != successful:
            raise ValueError(f"inconsistent Python success status: {path.name}")
        if not data["tests"]:
            raise ValueError(f"empty Python verification suite: {path.name}")
        data["report"] = str(path)
        results.append(data)
    for suite in expected:
        if sum(result["suite"] == suite for result in results) != 1:
            raise ValueError(f"expected exactly one Python report for {suite}")
    return results


def ptxas_json(path):
    def no_duplicates(pairs):
        data = dict(pairs)
        if len(data) != len(pairs):
            raise ValueError(f"duplicate PTXAS result field: {path}")
        return data
    return json.loads(path.read_text(), object_pairs_hook=no_duplicates)


def ptxas_refutation_diagnostics(diagnostic):
    """Recognize concrete failed obligations, excluding solver uncertainty.

Address pairing and guard messages without a solver result can also arise from
timeouts. Only explicit SAT results or unequal counts/widths establish the
specific negative controls used by this workflow.
"""
    concrete = []
    sat = (
        r"store \d+(?: (?:guard|value))?: sat",
        r"(?:load|store) \d+ address: sat",
        r"load \d+ guard: sat",
        r"(?:LOOPCOND: back edge vs the next guard|ENTRY: zero-trip guards disagree): sat",
        r"(?:epilogue )?stores \d+ and \d+ are REORDERED and may overlap(?: \[sat\]|: sat)",
        r"shared memory (?:entering barrier \d+|at exit): REFUTED \(sat\)",
        r"a shared access is not provably 4-byte aligned(?: or naturally aligned for its access width)? \[sat\]",
    )
    for raw in diagnostic.splitlines():
        line = raw.strip()
        if any(re.fullmatch(pattern, line) for pattern in sat):
            concrete.append(line)
            continue
        counts = re.fullmatch(r"load/store counts (\d+)/(\d+) (\d+)/(\d+)", line)
        if counts and (int(counts[1]) != int(counts[2]) or int(counts[3]) != int(counts[4])):
            concrete.append(line)
            continue
        pair = re.fullmatch(
            r"(?:barrier counts differ: ptx (\d+), sass (\d+)|epilogue store counts (\d+) vs (\d+))", line)
        if pair:
            values = [int(value) for value in pair.groups() if value is not None]
            if values[0] != values[1]:
                concrete.append(line)
            continue
        width = re.fullmatch(r"(?:(?:prologue|body|epilogue) )?(?:load|store) \d+ width: ptx (\d+) bits, sass (\d+) bits", line)
        if width and int(width[1]) != int(width[2]):
            concrete.append(line)
    return "\n".join(concrete)


def ptxas_command_probes(path, data, assembly, tools):
    """Bind subprocess executables to the successful retained tool probes."""
    probes = {}
    for name, arguments in (("ptxas", ["--version"]), ("nvdisasm", ["--version"]),
                            ("python_z3", ["-c", PYTHON_Z3_PROBE])):
        matches = [command for command in data["commands"]
                   if command["command"][1:] == arguments
                   and (name == "python_z3" or Path(command["command"][0]).name == name)]
        if len(matches) != 1 or matches[0]["exit_code"] != 0:
            raise ValueError(f"expected exactly one successful PTXAS {name} probe: {path}")
        probe = probes[name] = matches[0]
        executable_path = probe["command"][0]
        if not Path(executable_path).is_absolute():
            raise ValueError(f"PTXAS tool probe executable is not absolute: {path}")
        if tools is not None:
            tool = tools.get(name, {})
            if (not tool.get("available") or type(tool.get("path")) is not str
                    or os.path.normpath(executable_path) != os.path.normpath(tool["path"])):
                raise ValueError(f"PTXAS {name} executable differs from the discovered tool: {path}")
    if assembly["command"][0] != probes["ptxas"]["command"][0]:
        raise ValueError(f"PTXAS assembly executable differs from its tool probe: {path}")
    return probes


def ptxas_tool_versions(path, data, tools):
    name = "tool_versions.json"
    if name not in data["files"]:
        raise ValueError(f"missing retained PTXAS tool versions: {path}")
    versions = ptxas_json(path.parent / name)
    if (type(versions) is not dict or set(versions) != {"ptxas", "nvdisasm", "python_z3"}
            or any(type(value) is not str or not value.strip() for value in versions.values())):
        raise ValueError(f"invalid retained PTXAS tool versions: {path}")
    if tools is not None:
        for name, value in versions.items():
            if name == "python_z3":
                lines = value.strip().splitlines()
                if len(lines) != 2:
                    raise ValueError(f"invalid retained Python/Z3 version output: {path}")
                value = lines[0].split()[0] + " " + lines[1].strip()
            expected = tools.get(name, {}).get("version")
            if type(expected) is not str or value.strip() != expected.strip():
                raise ValueError(f"retained PTXAS {name} version differs from its discovered tool: {path}")


def ptxas_cubin_sections(path):
    """Check CUDA ELF64 structure before using its named code sections."""
    raw = path.read_bytes()
    if not raw.startswith(b"\x7fELF"):
        raise ValueError(f"retained PTXAS cubin is not ELF: {path}")
    if len(raw) < 64 or raw[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError(f"invalid CUDA ELF64 header: {path}")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
    _, kind, machine, version, _, phoff, shoff, _, ehsize, phsize, phnum, shsize, shnum, strings_index = header
    if (kind != 2 or machine != 190 or version != 1 or ehsize != 64
            or shsize != 64 or not shnum or not 0 < strings_index < shnum
            or shoff < ehsize or shoff + shsize * shnum > len(raw)
            or phnum and (phsize != 56 or phoff < ehsize or phoff + phsize * phnum > len(raw))):
        raise ValueError(f"invalid CUDA ELF64 header or table bounds: {path}")
    sections = [struct.unpack_from("<IIQQQQIIQQ", raw, shoff + index * shsize)
                for index in range(shnum)]
    strings = sections[strings_index]
    if strings[1] != 3 or not strings[5] or strings[4] + strings[5] > len(raw):
        raise ValueError(f"invalid CUDA ELF section string table: {path}")
    names = raw[strings[4]:strings[4] + strings[5]]
    text = {}
    for section in sections:
        offset, size = section[4:6]
        if section[1] != 8 and offset + size > len(raw):
            raise ValueError(f"CUDA ELF section extends past its cubin: {path}")
        start = section[0]
        end = names.find(b"\0", start)
        if start >= len(names) or end < 0:
            raise ValueError(f"invalid CUDA ELF section name: {path}")
        try:
            name = names[start:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"invalid CUDA ELF section name: {path}") from error
        if name.startswith(".text."):
            entry = name[len(".text."):]
            if (not entry or entry in text or section[1] != 1 or not section[2] & 4 or not size):
                raise ValueError(f"invalid CUDA ELF code section: {path}")
            text[entry] = size
    if not text:
        raise ValueError(f"CUDA ELF cubin has no code sections: {path}")
    return text


def ptxas_subject(path, data, source, sass, sections):
    """The retained PTX entries, cubin sections, and genuine SASS must agree."""
    try:
        text = strip_comments(source.read_text())
    except Exception as error:
        raise ValueError(f"invalid retained PTX source: {path}: {error}") from error
    text = re.sub(r'"(?:\\.|[^"\\])*"', lambda match: " " * len(match[0]), text)
    entries = re.findall(r"\.entry\s+([\w.$]+)\s*\(", text)
    targets = re.findall(r"(?m)^\s*\.target\s+(sm_[0-9]+[af]?)(?=\s|,|$)", text)
    address_sizes = re.findall(r"(?m)^\s*\.address_size\s+(32|64)\s*$", text)
    if (not entries or len(entries) != len(set(entries)) or len(targets) != 1
            or len(address_sizes) != 1 or not set(entries) <= sections.keys()):
        raise ValueError(f"retained PTX subject differs from its cubin or has invalid declarations: {path}")
    names = re.findall(r"(?m)^\s*\.section\s+\.text\.([\w.$]+)(?=,|\s|$)", sass)
    labels = re.findall(r"(?m)^\s*\.text\.([\w.$]+):\s*$", sass)
    if (names != labels or len(names) != len(sections) or set(names) != sections.keys()
            or data["verdict"] == "VALIDATED" and (len(entries) != 1 or len(sections) != 1)):
        raise ValueError(f"retained disassembly subject differs from its PTX or cubin: {path}")
    current = None
    pcs = {name: [] for name in names}
    for line in sass.splitlines():
        section = re.match(r"\s*\.section\s+\.text\.([\w.$]+)(?=,|\s|$)", line)
        if section:
            current = section[1]
        elif re.match(r"\s*/\*", line):
            instruction = re.fullmatch(r"\s*/\*([0-9a-fA-F]+)\*/\s+.+;\s*", line)
            if instruction is None or current is None:
                raise ValueError(f"malformed retained disassembly instruction: {path}")
            pcs[current].append(int(instruction[1], 16))
    for name, offsets in pcs.items():
        if (not offsets or offsets[0] != 0 or offsets != sorted(set(offsets))
                or offsets[-1] >= sections[name]):
            raise ValueError(f"retained disassembly instruction offsets disagree with its cubin: {path}")


def ptxas_bindings(path, data, assembly, tools=None):
    """Bind command transcripts and validation output to the retained inputs."""
    def command_path(command, argument):
        value = Path(argument)
        return (value if value.is_absolute() else Path(command["cwd"]) / value).resolve()

    def artifact(name):
        if name not in data["files"]:
            raise ValueError(f"missing retained PTXAS artifact {name}: {path}")
        return path.parent / name

    source = artifact("source.ptx")
    cubin = path.parent / "kernel.cubin"
    args = assembly["command"]
    if (len(args) != 6 or args[1:3] != [f"-O{data['optimization']}", f"-arch={data['target']}"]
            or args[4] != "-o" or command_path(assembly, args[3]) != source.resolve()
            or command_path(assembly, args[5]) != cubin.resolve()):
        raise ValueError(f"PTXAS assembly command differs from its retained inputs or target: {path}")
    probes = ptxas_command_probes(path, data, assembly, tools)
    ptxas_tool_versions(path, data, tools)
    operations = [assembly]
    expected_detail = data.get("expected_detail")
    if type(expected_detail) is not str:
        raise ValueError(f"invalid PTXAS diagnostic expectation: {path}")
    if data["verdict"] != "VALIDATED" and not expected_detail:
        raise ValueError(f"negative PTXAS case has no diagnostic expectation: {path}")
    if data["role"] == "assembly_refusal":
        if cubin.exists():
            raise ValueError(f"assembler refusal retained a cubin: {path}")
        stderr = artifact("ptxas.stderr.txt").read_text(errors="replace")
        if data["detail"] != stderr:
            raise ValueError(f"assembler refusal differs from its retained diagnostic: {path}")
        diagnostic = stderr
    else:
        sections = ptxas_cubin_sections(artifact("kernel.cubin"))
        sass = artifact("kernel.sass")
        if sass.read_bytes() != artifact("nvdisasm.stdout.txt").read_bytes():
            raise ValueError(f"retained SASS differs from disassembly output: {path}")
        targets = re.findall(rb"(?m)^\s*\.target\s+(sm_[0-9]+[af]?)\s*$", sass.read_bytes())
        if targets != [data["target"].encode()]:
            raise ValueError(f"retained SASS target differs from the assembly target: {path}")
        ptxas_subject(path, data, source, sass.read_text(), sections)
        disassembly = [c for c in data["commands"] if Path(c["command"][0]).name == "nvdisasm"
                       and "-c" in c["command"]]
        if (len(disassembly) != 1 or len(disassembly[0]["command"]) != 3
                or disassembly[0]["command"][1] != "-c"
                or command_path(disassembly[0], disassembly[0]["command"][2]) != cubin.resolve()):
            raise ValueError(f"disassembly command differs from its retained cubin: {path}")
        if disassembly[0]["command"][0] != probes["nvdisasm"]["command"][0]:
            raise ValueError(f"PTXAS disassembly executable differs from its tool probe: {path}")
        operations.append(disassembly[0])
        name = data.get("validation_sass")
        if type(name) is not str or name not in data["files"] or Path(name).suffix != ".sass":
            raise ValueError(f"missing retained validator SASS input: {path}")
        validation_sass = artifact(name)
        if (data["role"] == "genuine" and validation_sass.resolve() != sass.resolve()
                or data["role"] == "mutated" and (validation_sass.resolve() == sass.resolve()
                                                  or validation_sass.read_bytes() == sass.read_bytes())):
            raise ValueError(f"validator SASS input disagrees with its case role: {path}")
        if type(data.get("validator")) is not str or data["validator"] not in {"tval", "loopval", "smemval"}:
            raise ValueError(f"invalid PTXAS validator identity: {path}")
        validators = [c for c in data["commands"] if len(c["command"]) == 7
                      and c["command"][1] == "-c" and c["command"][4] == data["validator"]]
        if (len(validators) != 1
                or validators[0]["command"][2] != VALIDATOR_CHILD
                or command_path(validators[0], validators[0]["command"][3]) != (ROOT / "tools/ptxas_tval").resolve()
                or command_path(validators[0], validators[0]["command"][5]) != source.resolve()
                or command_path(validators[0], validators[0]["command"][6]) != validation_sass.resolve()):
            raise ValueError(f"validator command differs from its retained inputs: {path}")
        if validators[0]["command"][0] != probes["python_z3"]["command"][0]:
            raise ValueError(f"PTXAS validator executable differs from its Python/Z3 probe: {path}")
        operations.append(validators[0])
        validation = ptxas_json(artifact("validation.json"))
        transcript = ptxas_json(artifact("validator.stdout.txt"))
        for record in (validation, transcript):
            if (type(record) is not dict or set(record) != {"format", "verdict", "detail", "obligations", "log"}
                    or record.get("format") != "y-ptxas-validation-v1"
                    or type(record.get("obligations")) is not int or type(record.get("log")) is not str
                    or type(record.get("detail")) is not str or type(record.get("verdict")) is not str):
                raise ValueError(f"invalid PTXAS validator result schema: {path}")
        if (validation != transcript
                or any(validation[key] != data[key] for key in ("verdict", "detail", "obligations"))
                or validation["log"] != data.get("diagnostic_log")):
            raise ValueError(f"validator result differs from its retained transcript or case verdict: {path}")
        diagnostic = validation["detail"] + "\n" + validation["log"]
        if data["verdict"] == "UNPROVED":
            diagnostic = ptxas_refutation_diagnostics(diagnostic)
            if not diagnostic:
                raise ValueError(f"PTXAS diagnostic has no concrete SAT or structural refutation: {path}")
    expected_commands = [*probes.values(), *operations]
    if (len(data["commands"]) != len(expected_commands)
            or any(command not in expected_commands for command in data["commands"])
            or [command for command in data["commands"] if command in operations] != operations):
        raise ValueError(f"unexpected or out-of-order PTXAS command evidence: {path}")
    if expected_detail:
        try:
            matches = re.search(expected_detail, diagnostic)
        except (re.error, RecursionError) as error:
            raise ValueError(f"invalid PTXAS diagnostic expression: {path}: {error}") from error
        if not matches:
            raise ValueError(f"PTXAS diagnostic does not match its expected refusal or refutation: {path}")


def ptxas_results(directory, required=False, passed_tests=(), allowed_tests=None, tools=None):
    """Recheck retained PTXAS evidence; these records are test results, not load receipts."""
    results = []
    names = set()
    verdicts = {"VALIDATED", "UNPROVED", "REFUSED", "ASSEMBLY_REFUSED"}
    for path in sorted(directory.glob("*/case.json")):
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"PTXAS case escapes its evidence root: {path}")
        data = ptxas_json(path)
        if (type(data) is not dict or data.get("format") != "y-ptxas-case-v1"
                or type(data.get("name")) is not str or not data["name"] or data["name"] in names
                or type(data.get("test_id")) is not str or not data["test_id"]
                or type(data.get("target")) is not str or re.fullmatch(r"sm_[0-9]+[af]?", data["target"]) is None
                or type(data.get("optimization")) is not str or data["optimization"] not in {"0", "1", "2", "3"}
                or type(data.get("role")) is not str or data["role"] not in {"genuine", "mutated", "assembly_refusal"}
                or type(data.get("verdict")) is not str or data["verdict"] not in verdicts
                or data.get("expected") != data["verdict"] or type(data.get("detail")) is not str
                or type(data.get("obligations")) is not int or data["obligations"] < 0
                or type(data.get("files")) is not dict or not data["files"]
                or type(data.get("commands")) is not list or not data["commands"]):
            raise ValueError(f"invalid or mismatched PTXAS evidence: {path}")
        if data["verdict"] == "VALIDATED" and data["obligations"] == 0:
            raise ValueError(f"PTXAS validation has no proof obligations: {path}")
        assembly_refusal = data["role"] == "assembly_refusal"
        if assembly_refusal != (data["verdict"] == "ASSEMBLY_REFUSED"):
            raise ValueError(f"PTXAS evidence role disagrees with verdict: {path}")
        for command in data["commands"]:
            if (type(command) is not dict or set(command) != {"command", "exit_code", "cwd"}
                    or type(command.get("command")) is not list
                    or not command["command"] or any(type(arg) is not str for arg in command["command"])
                    or type(command.get("exit_code")) is not int
                    or type(command.get("cwd")) is not str or not Path(command["cwd"]).is_absolute()):
                raise ValueError(f"invalid PTXAS command evidence: {path}")
        failures = [command for command in data["commands"] if command["exit_code"] != 0]
        if bool(failures) != assembly_refusal:
            raise ValueError(f"PTXAS command exits disagree with expected result: {path}")
        assembly = [command for command in data["commands"]
                    if Path(command["command"][0]).name == "ptxas"
                    and "-o" in command["command"] and "--version" not in command["command"]]
        if len(assembly) != 1:
            raise ValueError(f"expected exactly one PTXAS assembly command: {path}")
        if assembly_refusal and (assembly[0]["exit_code"] <= 0 or assembly[0]["exit_code"] == 124
                                 or failures != assembly):
            raise ValueError(f"assembly refusal requires the PTXAS assembly command to fail: {path}")
        if not assembly_refusal and not any(Path(command["command"][0]).name == "nvdisasm"
                                           and "-c" in command["command"] for command in data["commands"]):
            raise ValueError(f"missing successful disassembly command: {path}")
        suffixes = set()
        for filename, digest in data["files"].items():
            if (type(filename) is not str or not filename or Path(filename).is_absolute()
                    or ".." in Path(filename).parts or type(digest) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
                raise ValueError(f"invalid PTXAS artifact identity: {path}")
            artifact = path.parent / filename
            if not artifact.resolve().is_relative_to(path.parent.resolve()):
                raise ValueError(f"PTXAS artifact escapes its evidence directory: {artifact}")
            if sha256(artifact) != digest:
                raise ValueError(f"PTXAS artifact SHA-256 mismatch: {artifact}")
            suffixes.add(artifact.suffix)
        wanted = {".ptx"} if assembly_refusal else {".ptx", ".cubin", ".sass"}
        if not wanted <= suffixes:
            raise ValueError(f"missing retained PTXAS artifacts: {path}")
        ptxas_bindings(path, data, assembly[0], tools)
        names.add(data["name"])
        results.append({**data, "report": str(path)})
    if required and not results:
        raise ValueError("passing PTXAS suite has no retained artifact evidence")
    inventory_path = directory / "inventory.json"
    if results or required or inventory_path.exists() or inventory_path.is_symlink():
        if not inventory_path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("PTXAS inventory escapes its evidence root")
        inventory = ptxas_json(inventory_path)
        if (type(inventory) is not dict or set(inventory) != {"format", "cases"}
                or inventory.get("format") != "y-ptxas-inventory-v1" or type(inventory.get("cases")) is not list):
            raise ValueError("invalid PTXAS case inventory")
        declared = []
        for item in inventory["cases"]:
            if (type(item) is not dict or set(item) != {"name", "test_id", "directory"}
                    or any(type(value) is not str or not value for value in item.values())
                    or Path(item["directory"]).name != item["directory"] or item["directory"] in {".", ".."}):
                raise ValueError("invalid PTXAS case inventory entry")
            declared.append((item["name"], item["test_id"], item["directory"]))
        observed = [(case["name"], case["test_id"], Path(case["report"]).parent.name) for case in results]
        if (len({item[0] for item in declared}) != len(declared)
                or len({item[2] for item in declared}) != len(declared) or set(declared) != set(observed)):
            raise ValueError("PTXAS case inventory differs from the retained cases")
        for case in results:
            case["inventory_sha256"] = sha256(inventory_path)
    missing = set(passed_tests) - {case["test_id"] for case in results}
    if missing:
        raise ValueError("passing PTXAS tests have no retained evidence: " + ", ".join(sorted(missing)))
    if allowed_tests is None and passed_tests:
        allowed_tests = passed_tests
    if allowed_tests is not None:
        unexpected = {case["test_id"] for case in results} - set(allowed_tests)
        if unexpected:
            raise ValueError("PTXAS evidence names unreported tests: " + ", ".join(sorted(unexpected)))
    return results


def run_stage(stage, root, env, tools, output, timeout):
    result = dict(stage)
    missing = [name for name in stage["requires"] if not tools[name]["available"]]
    if missing:
        return {**result, "status": "skipped", "reason": "missing prerequisites: " + ", ".join(missing)}
    log = output / f"{stage['name']}.log"
    sidecars = output / f"{stage['name']}-python"
    sidecars.mkdir()
    child_env = dict(env, Y_VERIFICATION_RESULT_DIR=str(sidecars))
    child_env.pop("Y_VERIFICATION_RESULT_FILE", None)
    child_env.pop("Y_PTXAS_EVIDENCE_DIR", None)
    evidence = None
    if "ptxas_pipeline_regressions" in stage["python_suites"]:
        evidence = output / f"{stage['name']}-ptxas"
        evidence.mkdir()
        child_env["Y_PTXAS_EVIDENCE_DIR"] = str(evidence)
    result.update(run_command(stage["command"], root, child_env, log, timeout))
    result["log"] = str(log)
    result["rust"] = rust_results(log.read_text(errors="replace"))
    try:
        result["python"] = python_results(sidecars, stage["python_suites"],
                                          strict=child_env.get("Y_VERIFICATION_STRICT") == "1")
    except (ValueError, OSError) as error:
        result["error"] = f"{result['error']}; {error}" if result.get("error") else str(error)
    if evidence is not None:
        try:
            passed = [test["id"] for report in result.get("python", [])
                      if report["suite"] == "ptxas_pipeline_regressions"
                      for test in report["tests"] if test["status"] == "pass"]
            allowed = [test["id"] for report in result.get("python", [])
                       if report["suite"] == "ptxas_pipeline_regressions" for test in report["tests"]]
            result["ptxas"] = ptxas_results(evidence, required=bool(passed), passed_tests=passed,
                                            allowed_tests=allowed, tools=tools)
        except (ValueError, OSError) as error:
            result["error"] = f"{result['error']}; {error}" if result.get("error") else str(error)
    rust = result["rust"]
    if not rust["targets"] or not rust["pass"] + rust["fail"]:
        result.setdefault("error", "no executed Rust test results")
    failed_python = any(report["exit_code"] != 0 or not report["complete"] or report["counts"]["fail"]
                        + report["counts"]["error"] + report["counts"]["unexpected_success"]
                        for report in result.get("python", []))
    skipped_python = any(report["counts"]["skip"] + report["counts"]["expected_failure"]
                         for report in result.get("python", []))
    if result.get("error") or result["exit_code"] != 0 or rust["fail"] or failed_python:
        result["status"] = "failed"
    elif rust["skip_notices"] or rust["ignored"] or skipped_python:
        result["status"] = "incomplete"
    else:
        result["status"] = "passed"
    return result


def outcome(stages, errors, allow_skips):
    if errors or any(stage["status"] == "failed" for stage in stages):
        return "failed", 1
    if any(stage["status"] != "passed" for stage in stages) or not stages:
        ran = any(stage["status"] in ("passed", "incomplete") for stage in stages)
        return "incomplete", 0 if allow_skips and ran else 2
    return "passed", 0


def write_report(output, report):
    temporary = output / ".results.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "results.json")
    lines = ["# Y verification results", "", f"Status: **{report['status']}** (exit {report['exit_code']}).",
             "", f"Profile: `{report['profile']}`. Strict: `{report['strict']}`.", "",
             "| Stage | Status | Rust pass/fail/ignored | Evidence |", "| --- | --- | --- | --- |"]
    for stage in report["stages"]:
        rust = stage.get("rust", {})
        counts = "/".join(str(rust.get(key, 0)) for key in ("pass", "fail", "ignored"))
        evidence = f"[{stage['name']}.log]({stage['name']}.log)" if stage.get("log") else stage.get("reason", "")
        lines.append(f"| {stage['name']} | {stage['status']} | {counts} | {evidence} |")
    if report["errors"]:
        lines.extend(["", "Errors:", "", *[f"- {error}" for error in report["errors"]]])
    for stage in report["stages"]:
        skips = stage.get("rust", {}).get("skip_notices", [])[:10]
        if stage.get("error") or skips:
            lines.extend(["", f"{stage['name']}:", ""])
            lines.extend(f"- {message}" for message in ([stage["error"]] if stage.get("error") else []) + skips)
        for result in stage.get("python", []):
            counts = result["counts"]
            lines.extend(["", f"`{result['suite']}`: {counts['pass']} passed, {counts['fail']} failed, "
                          f"{counts['error']} errors, {counts['skip']} skipped."])
            lines.extend(f"- `{test['id']}`: {test.get('reason', 'no reason supplied')}"
                         for test in result["tests"] if test["status"] == "skip")
        if stage.get("ptxas"):
            lines.extend(["", f"{stage['name']} retained PTXAS cases:", "",
                          "| Case | Role | Target / optimization | Expected result | Obligations | Evidence |",
                          "| --- | --- | --- | --- | --- | --- |"])
            for case in stage["ptxas"]:
                relative = Path(case["report"]).relative_to(output)
                lines.append(f"| {case['name']} | {case['role']} | {case['target']} / O{case['optimization']} | "
                             f"{case['verdict']} | {case['obligations']} | [case.json]({relative.as_posix()}) |")
    lines.extend(["", "Full commands, tool probes, and Python test records are in [results.json](results.json).",
                  "Source hashes are in [inputs.json](inputs.json).", ""])
    (output / "summary.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also run the entire workspace with zk")
    parser.add_argument("--stage", action="append", choices=("smt", "proofs", "translation", "ptxas",
                        "artifacts", "launch", "workflow", "workspace"), help="run only this stage; repeat to select several")
    parser.add_argument("--allow-skips", action="store_true", help="permit exit 0 for explicitly incomplete runs")
    parser.add_argument("--output", type=Path, help="new directory for results and logs")
    parser.add_argument("--timeout", type=int, default=1800, help="deadline per stage in seconds (default: 1800)")
    parser.add_argument("--list", action="store_true", help="show selected stages without running them")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.stage and "workspace" in args.stage and not args.full:
        parser.error("--stage workspace requires --full")
    def selected(cargo):
        return [stage for stage in plan(cargo, args.full)
                if not args.stage or stage["name"] in args.stage]
    if args.list:
        for stage in selected("cargo"):
            print(f"{stage['name']}: {' '.join(stage['command'])}")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "target/verification" / f"{stamp}-{os.getpid()}").absolute()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", RUST_BACKTRACE="0")
    env["Y_VERIFICATION_STRICT"] = "0" if args.allow_skips else "1"
    report = {"format": FORMAT, "profile": "selected" if args.stage else "full" if args.full else "focused",
              "strict": not args.allow_skips, "started_at": datetime.now(timezone.utc).isoformat(),
              "root": str(ROOT), "output": str(output), "selected_stages": args.stage,
              "stages": [], "errors": []}
    print(f"Verification evidence: {output}", flush=True)
    before = None
    try:
        before = input_hashes(ROOT, output)
        (output / "inputs.json").write_text(json.dumps(before, indent=2) + "\n")
        report["tools"] = tools = discover(ROOT, env)
        if "Y_ALLOW_UNVERIFIED_INVARIANTS" in env:
            raise ValueError("Y_ALLOW_UNVERIFIED_INVARIANTS bypasses proofs even when empty or 0; unset it before verification")
        if tools["z3"]["available"]:
            env["Y_Z3_PATH"] = tools["z3"]["path"]
        interpreter = tools["python_z3"] if tools["python_z3"]["available"] else tools["python"]
        if interpreter["available"]:
            env["Y_TVAL_PYTHON"] = interpreter["path"]
        cargo = tools["cargo"].get("path", "cargo")
        for stage in selected(cargo):
            print(f"Running {stage['name']}...", flush=True)
            result = run_stage(stage, ROOT, env, tools, output, args.timeout)
            report["stages"].append(result)
            print(f"  {result['status'].upper()}: {stage['name']}", flush=True)
    except (OSError, ValueError, KeyboardInterrupt) as error:
        report["errors"].append(str(error) or "verification interrupted")
    if before is not None:
        try:
            after = input_hashes(ROOT, output)
            changed = [name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]
            report["changed_inputs"] = changed
            if changed:
                report["errors"].append("Verification inputs changed during the run: " + ", ".join(changed))
        except OSError as error:
            report["errors"].append(f"cannot check final input hashes: {error}")
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["status"], report["exit_code"] = outcome(report["stages"], report["errors"], args.allow_skips)
    write_report(output, report)
    print(f"{report['status'].upper()}; report: {output / 'summary.md'}", flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
