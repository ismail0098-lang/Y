"""Adversarial evidence and execution checks for tools/verify.py."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify
from verification_unittest import main as verification_main


def sidecar(suite="suite", statuses=("pass",), strict=False, complete=True):
    counts = {status: statuses.count(status) for status in verify.TEST_STATUSES}
    counts["tests"] = len(statuses)
    successful = complete and not (counts["fail"] or counts["error"] or counts["unexpected_success"]
                                   or strict and counts["skip"])
    return {"format": verify.PYTHON_FORMAT, "suite": suite, "strict": strict,
            "complete": complete, "unittest_tests_run": len(statuses),
            "successful": bool(successful), "exit_code": 0 if successful else 1,
            "counts": counts, "tests": [{"id": f"suite.case{n}", "status": status}
                                         for n, status in enumerate(statuses)]}


def cubin_fixture():
    strings = b"\0.shstrtab\0.text.control\0"
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    header = struct.pack("<16sHHIQQQIHHHHHH", ident, 2, 190, 1, 0, 0, 64, 0, 64, 0, 0, 64, 3, 1)
    string_section = struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, 256, len(strings), 0, 0, 1, 0)
    code_section = struct.pack("<IIQQQQIIQQ", 11, 1, 6, 0, 256 + len(strings), 16, 0, 0, 16, 0)
    return header + bytes(64) + string_section + code_section + strings + bytes(16)


class Reports(unittest.TestCase):
    def inventory(self, directory):
        records = []
        for path in sorted(directory.glob('*/case.json')):
            data = json.loads(path.read_text())
            records.append({"name": data["name"], "test_id": data["test_id"], "directory": path.parent.name})
        (directory / "inventory.json").write_text(json.dumps({"format": "y-ptxas-inventory-v1", "cases": records}))

    def update_evidence(self, case, data):
        data["files"] = {p.name: verify.sha256(p) for p in case.iterdir()
                         if p.is_file() and p.name != "case.json"}
        (case / "case.json").write_text(json.dumps(data))

    def evidence(self, directory):
        case = directory / "control"
        case.mkdir()
        validation = {"format": "y-ptxas-validation-v1", "verdict": "VALIDATED",
                      "detail": "all obligations proved", "obligations": 2, "log": "proved"}
        sass = b'.target sm_89\n.section .text.control,"ax",@progbits\n.text.control:\n/*0000*/ EXIT;\n'
        versions = {"ptxas": "fixture PTXAS 1", "nvdisasm": "fixture nvdisasm 1",
                    "python_z3": sys.version + "\n4.15.4\n"}
        source = b".version 7.8\n.target sm_89\n.address_size 64\n.visible .entry control()\n{\nret;\n}\n"
        for name, content in (("source.ptx", source), ("kernel.cubin", cubin_fixture()),
                              ("kernel.sass", sass), ("nvdisasm.stdout.txt", sass),
                              ("validation.json", json.dumps(validation).encode()),
                              ("validator.stdout.txt", json.dumps(validation).encode()),
                              ("tool_versions.json", json.dumps(versions).encode()),
                              ("ptxas.stderr.txt", b"")):
            (case / name).write_bytes(content)
        data = {"format": "y-ptxas-case-v1", "name": "control", "test_id": "suite.control",
                "target": "sm_89", "optimization": "1", "role": "genuine", "expected": "VALIDATED",
                "verdict": "VALIDATED", "detail": "all obligations proved", "obligations": 2,
                "expected_detail": "", "diagnostic_log": "proved", "validator": "tval",
                "validation_sass": "kernel.sass",
                "commands": [{"command": ["/fixture/bin/ptxas", "-O1", "-arch=sm_89", "source.ptx", "-o", "kernel.cubin"],
                              "exit_code": 0, "cwd": str(case)},
                             {"command": ["/fixture/bin/nvdisasm", "-c", "kernel.cubin"], "exit_code": 0, "cwd": str(case)},
                             {"command": [sys.executable, "-c", verify.VALIDATOR_CHILD, str(verify.ROOT / "tools/ptxas_tval"),
                                          "tval", "source.ptx", "kernel.sass"], "exit_code": 0, "cwd": str(case)},
                             {"command": ["/fixture/bin/ptxas", "--version"], "exit_code": 0, "cwd": str(case)},
                             {"command": ["/fixture/bin/nvdisasm", "--version"], "exit_code": 0, "cwd": str(case)},
                             {"command": [sys.executable, "-c", verify.PYTHON_Z3_PROBE], "exit_code": 0, "cwd": str(case)}],
                "files": {p.name: verify.sha256(p) for p in case.iterdir()}}
        (case / "case.json").write_text(json.dumps(data))
        self.inventory(directory)
        return case, data

    def refutation_evidence(self, directory, detail, log, anchor):
        case, data = self.evidence(directory)
        data.update(verdict="UNPROVED", expected="UNPROVED", detail=detail,
                    diagnostic_log=log, expected_detail=anchor)
        validation = {"format": "y-ptxas-validation-v1", "verdict": data["verdict"],
                      "detail": detail, "obligations": data["obligations"], "log": log}
        for name in ("validation.json", "validator.stdout.txt"):
            (case / name).write_text(json.dumps(validation))
        self.update_evidence(case, data)
        return case, data

    def test_retained_ptxas_evidence_is_bound_to_each_passing_test_and_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            case, _ = self.evidence(directory)
            parsed = verify.ptxas_results(directory, required=True, passed_tests=["suite.control"])
            self.assertEqual(parsed[0]["obligations"], 2)
            with self.assertRaisesRegex(ValueError, "no retained evidence"):
                verify.ptxas_results(directory, passed_tests=["suite.omitted"])
            (case / "kernel.cubin").write_bytes(b"replaced")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify.ptxas_results(directory, required=True)

    def test_inconsistent_ptxas_evidence_cannot_be_accepted(self):
        for corruption in ("no_cases", "no_obligations", "wrong_verdict", "command_failed", "missing_sass",
                           "traversal", "absolute", "duplicate_field", "wrong_role", "no_test_id",
                           "wrong_process_failed", "no_disassembly"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                if corruption == "no_cases":
                    with self.assertRaises(ValueError):
                        verify.ptxas_results(directory, required=True)
                    continue
                case, data = self.evidence(directory)
                if corruption == "no_obligations":
                    data["obligations"] = 0
                elif corruption == "wrong_verdict":
                    data["verdict"] = "UNPROVED"
                elif corruption == "command_failed":
                    data["commands"][0]["exit_code"] = 1
                elif corruption == "missing_sass":
                    del data["files"]["kernel.sass"]
                elif corruption == "traversal":
                    data["files"]["../foreign.sass"] = "0" * 64
                elif corruption == "absolute":
                    data["files"][str(case / "kernel.sass")] = data["files"].pop("kernel.sass")
                elif corruption == "wrong_role":
                    data["role"] = "assembly_refusal"
                elif corruption == "no_test_id":
                    del data["test_id"]
                elif corruption == "wrong_process_failed":
                    data.update(role="assembly_refusal", verdict="ASSEMBLY_REFUSED", expected="ASSEMBLY_REFUSED", obligations=0)
                    data["commands"][1]["exit_code"] = 1
                elif corruption == "no_disassembly":
                    data["commands"] = data["commands"][:1]
                text = json.dumps(data)
                if corruption == "duplicate_field":
                    text = text.replace('"obligations": 2', '"obligations": 0, "obligations": 2')
                (case / "case.json").write_text(text)
                with self.assertRaises(ValueError):
                    verify.ptxas_results(directory, required=True)

    def test_ptxas_case_symlinks_and_unreported_test_ids_cannot_enter_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "evidence"
            directory.mkdir()
            outside = Path(temporary) / "outside"
            outside.mkdir()
            case, data = self.evidence(outside)
            (directory / "foreign").symlink_to(case, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes its evidence root"):
                verify.ptxas_results(directory, required=True)
            (directory / "foreign").unlink()
            case, data = self.evidence(directory)
            other = directory / "extra"
            shutil.copytree(case, other)
            data.update(name="extra", test_id="suite.unreported")
            for command in data["commands"]:
                command["cwd"] = str(other)
            (other / "case.json").write_text(json.dumps(data))
            self.inventory(directory)
            with self.assertRaisesRegex(ValueError, "unreported tests"):
                verify.ptxas_results(directory, passed_tests=["suite.control"])

    def test_expected_assembler_rejections_remain_distinct_from_translation_proofs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            case, data = self.evidence(directory)
            data.update(role="assembly_refusal", verdict="ASSEMBLY_REFUSED", expected="ASSEMBLY_REFUSED", obligations=0)
            data.update(detail="syntax error", expected_detail="syntax")
            data["commands"][0]["exit_code"] = 1
            data["commands"] = [data["commands"][0], *data["commands"][3:]]
            (case / "kernel.cubin").unlink()
            (case / "ptxas.stderr.txt").write_text("syntax error")
            data["files"] = {name: verify.sha256(case / name) for name in
                             ("source.ptx", "ptxas.stderr.txt", "tool_versions.json")}
            (case / "case.json").write_text(json.dumps(data))
            self.assertEqual(verify.ptxas_results(directory)[0]["verdict"], "ASSEMBLY_REFUSED")

    def test_ptxas_commands_cannot_substitute_foreign_inputs_or_flags(self):
        corruptions = ("source", "cubin", "optimization", "target", "disassembly", "validator_source",
                       "validator_sass", "validator_missing", "validator_type", "validator_program", "cwd")
        for corruption in corruptions:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case, data = self.evidence(directory)
                if corruption == "source":
                    data["commands"][0]["command"][3] = "/foreign/source.ptx"
                elif corruption == "cubin":
                    data["commands"][0]["command"][5] = "/foreign/kernel.cubin"
                elif corruption == "optimization":
                    data["commands"][0]["command"][1] = "-O3"
                elif corruption == "target":
                    data["commands"][0]["command"][2] = "-arch=sm_90"
                elif corruption == "disassembly":
                    data["commands"][1]["command"][2] = "/foreign/kernel.cubin"
                elif corruption == "validator_source":
                    data["commands"][2]["command"][5] = "/foreign/source.ptx"
                elif corruption == "validator_sass":
                    data["commands"][2]["command"][6] = "/foreign/kernel.sass"
                elif corruption == "validator_missing":
                    data["commands"].pop(2)
                elif corruption == "validator_type":
                    data["validator"] = []
                elif corruption == "validator_program":
                    data["commands"][2]["command"][2] = 'print("fabricated validator output")'
                elif corruption == "cwd":
                    del data["commands"][0]["cwd"]
                self.update_evidence(case, data)
                with self.assertRaises(ValueError):
                    verify.ptxas_results(directory, required=True)

    def test_assembler_timeout_or_signal_cannot_be_an_expected_rejection(self):
        for exit_code in (124, -9, -11):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case, data = self.evidence(directory)
                data.update(role="assembly_refusal", verdict="ASSEMBLY_REFUSED", expected="ASSEMBLY_REFUSED",
                            obligations=0, detail="syntax error", expected_detail="syntax")
                data["commands"] = [data["commands"][0], *data["commands"][3:]]
                data["commands"][0]["exit_code"] = exit_code
                (case / "kernel.cubin").unlink()
                (case / "ptxas.stderr.txt").write_text("syntax error")
                self.update_evidence(case, data)
                with self.assertRaisesRegex(ValueError, "assembly command to fail"):
                    verify.ptxas_results(directory, required=True)

    def test_ptxas_command_executables_probes_and_operations_cannot_be_substituted(self):
        corruptions = ("assembler", "disassembler", "validator", "missing_probe", "duplicate_probe",
                       "probe_program", "relative_probe", "extra_command", "extra_validator",
                       "operation_order", "extra_field")
        for corruption in corruptions:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case, data = self.evidence(directory)
                commands = data["commands"]
                if corruption == "assembler":
                    commands[0]["command"][0] = "/foreign/ptxas"
                elif corruption == "disassembler":
                    commands[1]["command"][0] = "/foreign/nvdisasm"
                elif corruption == "validator":
                    commands[2]["command"][0] = "/bin/true"
                elif corruption == "missing_probe":
                    commands.pop(5)
                elif corruption == "duplicate_probe":
                    commands.append(commands[5])
                elif corruption == "probe_program":
                    commands[5]["command"][2] = 'print("pretend Python/Z3 version")'
                elif corruption == "relative_probe":
                    commands[2]["command"][0] = commands[5]["command"][0] = "python"
                elif corruption == "extra_command":
                    commands.append({"command": ["/bin/true"], "exit_code": 0, "cwd": str(case)})
                elif corruption == "extra_validator":
                    copied = json.loads(json.dumps(commands[2]))
                    copied["command"][4] = "loopval"
                    commands.append(copied)
                elif corruption == "operation_order":
                    commands[1], commands[2] = commands[2], commands[1]
                elif corruption == "extra_field":
                    commands[2]["env"] = {"PYTHONPATH": "/foreign"}
                self.update_evidence(case, data)
                with self.assertRaisesRegex(ValueError, "PTXAS"):
                    verify.ptxas_results(directory, required=True)

    def test_live_ptxas_evidence_uses_the_discovered_executables_even_if_probes_are_altered(self):
        for name, operation, probe, foreign in (("ptxas", 0, 3, "/foreign/ptxas"),
                                                ("nvdisasm", 1, 4, "/foreign/nvdisasm"),
                                                ("python_z3", 2, 5, "/bin/true")):
            with self.subTest(tool=name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case, data = self.evidence(directory)
                tools = {name: {"available": True, "path": data["commands"][index]["command"][0]}
                         for name, index in (("ptxas", 3), ("nvdisasm", 4), ("python_z3", 5))}
                versions = json.loads((case / "tool_versions.json").read_text())
                for tool, value in tools.items():
                    value["version"] = (sys.version.split()[0] + " 4.15.4"
                                        if tool == "python_z3" else versions[tool])
                self.assertEqual(verify.ptxas_results(directory, tools=tools)[0]["verdict"], "VALIDATED")
                data["commands"][operation]["command"][0] = foreign
                data["commands"][probe]["command"][0] = foreign
                self.update_evidence(case, data)
                with self.assertRaisesRegex(ValueError, "differs from the discovered tool"):
                    verify.ptxas_results(directory, tools=tools)

    def test_disassembly_identity_and_elf_are_checked_after_rehashing(self):
        for corruption, expected in (("elf", "not ELF"), ("stdout", "differs from disassembly"),
                                     ("target", "target differs")):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case, data = self.evidence(directory)
                if corruption == "elf":
                    (case / "kernel.cubin").write_bytes(b"not a cubin")
                elif corruption == "stdout":
                    (case / "nvdisasm.stdout.txt").write_bytes(b"unrelated disassembly")
                else:
                    for name in ("kernel.sass", "nvdisasm.stdout.txt"):
                        (case / name).write_bytes(b".target sm_90\n/*0000*/ EXIT;\n")
                self.update_evidence(case, data)
                with self.assertRaisesRegex(ValueError, expected):
                    verify.ptxas_results(directory, required=True)

    def test_validation_results_cannot_disagree_with_case_or_stdout(self):
        for corruption in ("verdict", "detail", "stdout", "duplicate", "boolean_obligations"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case, data = self.evidence(directory)
                validation = json.loads((case / "validation.json").read_text())
                if corruption == "verdict":
                    validation.update(verdict="REFUSED", obligations=0)
                elif corruption == "detail":
                    validation["detail"] = "unrelated result"
                elif corruption == "boolean_obligations":
                    validation["obligations"] = True
                text = json.dumps(validation)
                if corruption == "duplicate":
                    text = text.replace('"obligations": 2', '"obligations": 0, "obligations": 2')
                (case / "validation.json").write_text(text)
                if corruption != "stdout":
                    (case / "validator.stdout.txt").write_text(text)
                else:
                    (case / "validator.stdout.txt").write_text("{}")
                self.update_evidence(case, data)
                with self.assertRaises(ValueError):
                    verify.ptxas_results(directory, required=True)

    def test_validator_stdout_obligations_require_integer_types_independently(self):
        for invalid, actual in ((2.0, 2), (True, 1)):
            with self.subTest(value=invalid), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case, data = self.evidence(directory)
                data["obligations"] = actual
                validation = json.loads((case / "validation.json").read_text())
                validation["obligations"] = actual
                (case / "validation.json").write_text(json.dumps(validation))
                validation["obligations"] = invalid
                (case / "validator.stdout.txt").write_text(json.dumps(validation))
                self.update_evidence(case, data)
                with self.assertRaisesRegex(ValueError, "validator result schema"):
                    verify.ptxas_results(directory, required=True)

    def test_mutated_role_requires_the_actual_distinct_validator_input(self):
        for corruption in ("missing", "identical", "genuine_mutant", "wrong_command"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case, data = self.evidence(directory)
                data.update(role="mutated", validation_sass="mutated.sass")
                data["commands"][2]["command"][6] = "mutated.sass"
                if corruption != "missing":
                    (case / "mutated.sass").write_bytes((case / "kernel.sass").read_bytes()
                                                        if corruption == "identical" else b".target sm_89\nNOP;\n")
                if corruption == "genuine_mutant":
                    data["role"] = "genuine"
                elif corruption == "wrong_command":
                    data["commands"][2]["command"][6] = "kernel.sass"
                self.update_evidence(case, data)
                with self.assertRaises(ValueError):
                    verify.ptxas_results(directory, required=True)

    def test_timeout_cannot_satisfy_a_retained_refutation_expectation(self):
        for anchor in (r"store 0: sat", "unknown", ""):
            with self.subTest(anchor=anchor), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case, data = self.evidence(directory)
                data.update(verdict="UNPROVED", expected="UNPROVED", expected_detail=anchor,
                            detail="solver timed out", diagnostic_log="unknown")
                validation = {"format": "y-ptxas-validation-v1", "verdict": "UNPROVED",
                              "detail": data["detail"], "obligations": 2, "log": data["diagnostic_log"]}
                for name in ("validation.json", "validator.stdout.txt"):
                    (case / name).write_text(json.dumps(validation))
                self.update_evidence(case, data)
                with self.assertRaisesRegex(ValueError, "diagnostic"):
                    verify.ptxas_results(directory, required=True)

    def test_invalid_refutation_expression_is_rejected_after_a_concrete_sat_result(self):
        for anchor in ("[", "(" * 1000 + "x" + ")" * 1000):
            with self.subTest(anchor=anchor), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.refutation_evidence(directory, "1 stores, 0 loads, 0.0s", "store 0: sat", anchor)
                with self.assertRaisesRegex(ValueError, "invalid PTXAS diagnostic expression"):
                    verify.ptxas_results(directory, required=True)

    def test_matching_uncertainty_or_equal_structure_cannot_satisfy_negative_controls(self):
        diagnostics = (
            ("solver timed out", "store 0: unknown", "unknown"),
            ("store 0 guard", "", "guard"),
            ("load 0 address", "", "address"),
            ("load 0 address: unknown", "", "address: unknown"),
            ("store 0 address: unknown", "", "address: unknown"),
            ("load 0 guard: unknown", "", "guard: unknown"),
            ("shared memory at exit: solver said unknown -- a WALL, not a mismatch", "", "unknown"),
            ("load/store counts 00/0 2/2", "", "counts"),
            ("barrier counts differ: ptx 1, sass 1", "", "counts"),
            ("epilogue store counts 2 vs 2", "", "counts"),
            ("store 0 width: ptx 32 bits, sass 32 bits", "", "width"),
            ("stores 0 and 1 are REORDERED and may overlap [unknown]", "", "overlap"),
            ("", "store 0: unsat", "sat"),
            ("", "store 0: sat extra", "sat"),
            ("", "NOTE store 0: sat", "sat"),
            ("store 0 guard", "store 1: sat", "store 0 guard"),
        )
        for detail, log, anchor in diagnostics:
            with self.subTest(detail=detail, log=log), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.refutation_evidence(directory, detail, log, anchor)
                with self.assertRaisesRegex(ValueError, "diagnostic"):
                    verify.ptxas_results(directory, required=True)

    def test_specific_sat_and_unequal_structural_diagnostics_remain_accepted(self):
        diagnostics = (
            ("1 stores, 0 loads, 0.0s", "  store 0: sat\n", "store 0: sat"),
            ("store 0 value: sat", "", "store 0 value: sat"),
            ("store 0 guard: sat", "", "store 0 guard: sat"),
            ("load 0 address: sat", "", "load 0 address: sat"),
            ("store 0 address: sat", "", "store 0 address: sat"),
            ("load 0 guard: sat", "", "load 0 guard: sat"),
            ("LOOPCOND: back edge vs the next guard: sat", "", "LOOPCOND"),
            ("ENTRY: zero-trip guards disagree: sat", "", "ENTRY"),
            ("stores 0 and 1 are REORDERED and may overlap [sat]", "", "overlap [sat]"),
            ("epilogue stores 0 and 1 are REORDERED and may overlap: sat", "", "overlap: sat"),
            ("shared memory entering barrier 0: REFUTED (sat)", "", "REFUTED (sat)"),
            ("shared memory at exit: REFUTED (sat)", "", "REFUTED (sat)"),
            ("a shared access is not provably 4-byte aligned [sat]", "", "aligned [sat]"),
            ("load/store counts 0/0 2/1", "", "counts 0/0 2/1"),
            ("barrier counts differ: ptx 1, sass 0", "", "counts differ: ptx 1, sass 0"),
            ("epilogue store counts 2 vs 1", "", "counts 2 vs 1"),
            ("store 0 width: ptx 8 bits, sass 32 bits", "", "width: ptx 8 bits, sass 32 bits"),
            ("epilogue store 0 width: ptx 8 bits, sass 32 bits", "", "width: ptx 8 bits, sass 32 bits"),
        )
        for detail, log, anchor in diagnostics:
            with self.subTest(detail=detail, log=log), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.refutation_evidence(directory, detail, log, re.escape(anchor))
                parsed = verify.ptxas_results(directory, required=True)
                self.assertEqual(parsed[0]["verdict"], "UNPROVED")

    def test_each_started_subcase_must_survive_in_retained_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            case, data = self.evidence(directory)
            for number in range(3):
                extra = directory / f"extra{number}"
                shutil.copytree(case, extra)
                copied = json.loads(json.dumps(data))
                copied["name"] = extra.name
                for command in copied["commands"]:
                    command["cwd"] = str(extra)
                self.update_evidence(extra, copied)
            self.inventory(directory)
            self.assertEqual(len(verify.ptxas_results(directory, passed_tests=["suite.control"])), 4)
            shutil.rmtree(directory / "extra0")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                verify.ptxas_results(directory, passed_tests=["suite.control"])

    def test_case_inventory_rejects_duplicates_path_escapes_and_undeclared_cases(self):
        for corruption in ("duplicate", "directory", "missing", "undeclared", "type", "symlink"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "evidence"
                directory.mkdir()
                self.evidence(directory)
                path = directory / "inventory.json"
                inventory = json.loads(path.read_text())
                if corruption == "duplicate":
                    inventory["cases"].append(inventory["cases"][0])
                elif corruption == "directory":
                    inventory["cases"][0]["directory"] = "../foreign"
                elif corruption == "undeclared":
                    inventory["cases"] = []
                elif corruption == "type":
                    inventory["cases"] = {}
                if corruption == "missing":
                    path.unlink()
                elif corruption == "symlink":
                    outside = directory.parent / "inventory.json"
                    outside.write_text(json.dumps(inventory))
                    path.unlink()
                    path.symlink_to(outside)
                else:
                    path.write_text(json.dumps(inventory))
                with self.assertRaises((ValueError, OSError)):
                    verify.ptxas_results(directory, required=True)

    def test_present_inventory_is_checked_even_without_completed_cases_or_passing_tests(self):
        for corruption in ("started", "invalid", "symlink"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "evidence"
                directory.mkdir()
                case, _ = self.evidence(directory)
                shutil.rmtree(case)
                path = directory / "inventory.json"
                if corruption == "invalid":
                    path.write_text("{}")
                elif corruption == "symlink":
                    outside = directory.parent / "inventory.json"
                    path.replace(outside)
                    path.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "inventory"):
                    verify.ptxas_results(directory, required=False)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertEqual(verify.ptxas_results(directory), [])
            self.inventory(directory)
            self.assertEqual(verify.ptxas_results(directory), [])

    def test_missing_and_forged_python_evidence_cannot_pass(self):
        for corruption in ("missing", "counts", "success", "duplicate", "empty", "two_suites", "status"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                report = sidecar()
                if corruption == "counts":
                    report["counts"]["pass"] = 0
                elif corruption == "success":
                    report["successful"] = False
                elif corruption == "empty":
                    report = sidecar(statuses=())
                elif corruption == "status":
                    report["tests"][0]["status"] = []
                if corruption != "missing":
                    text = json.dumps(report)
                    if corruption == "duplicate":
                        text = text.replace('"suite": "suite"', '"suite": "other", "suite": "suite"')
                    (directory / "suite.json").write_text(text)
                if corruption == "two_suites":
                    (directory / "second.json").write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    verify.python_results(directory, ["suite"])

    def test_strict_skips_and_failfast_results_remain_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for strict, complete in ((False, True), (True, True), (False, False)):
                data = sidecar(statuses=("skip",), strict=strict, complete=complete)
                data["tests"][0]["reason"] = "solver unavailable"
                (directory / "suite.json").write_text(json.dumps(data))
                parsed = verify.python_results(directory, ["suite"])[0]
                self.assertEqual(parsed["tests"][0]["reason"], "solver unavailable")
                self.assertEqual(parsed["complete"], complete)
                self.assertEqual(parsed["exit_code"], 0 if not strict and complete else 1)

    def test_sidecar_cannot_change_the_requested_strictness_or_suite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "suite.json").write_text(json.dumps(sidecar()))
            with self.assertRaisesRegex(ValueError, "strictness"):
                verify.python_results(directory, ["suite"], strict=True)
            with self.assertRaisesRegex(ValueError, "unexpected"):
                verify.python_results(directory, ["another_suite"], strict=False)

    def test_failed_results_cannot_be_downgraded_by_allow_skips(self):
        self.assertEqual(verify.outcome([{"status": "failed"}], [], True), ("failed", 1))
        self.assertEqual(verify.outcome([{"status": "passed"}], ["inputs changed"], True), ("failed", 1))
        self.assertEqual(verify.outcome([{"status": "skipped"}], [], True), ("incomplete", 2))
        stages = [{"status": "passed"}, {"status": "skipped"}]
        self.assertEqual(verify.outcome(stages, [], False), ("incomplete", 2))
        self.assertEqual(verify.outcome(stages, [], True), ("incomplete", 0))

    def test_rust_skips_are_reported_without_confusing_hardware_probe_messages(self):
        parsed = verify.rust_results("""test valid ... ok
test no_driver ... SKIP: no CUDA driver
ok
test no_clang ... note: clang not found, skipping CPU GEMM end-to-end test
test no_circom ... note: `circom` not installed; skipping the differential metadata check
note: `circom --version` failed; skipping
note: circom could not compile probe; skipping it
ptxas not found; skipping
[*] Found existing profile, skipping Sentinel Probe
census: 10 checks; skipped: ['unsupported backend']
test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out;
""")
        self.assertEqual(parsed["pass"], 2)
        self.assertEqual(parsed["ignored"], 1)
        self.assertEqual(len(parsed["skip_notices"]), 6)
        self.assertFalse(any("Sentinel" in line or "census" in line for line in parsed["skip_notices"]))


class Execution(unittest.TestCase):
    def test_ptxas_profiles_require_every_python_suite_invoked_by_the_rust_target(self):
        rust = (verify.ROOT / "tests/ptxas_verification.rs").read_text()
        invoked = set(re.findall(r'run_suite\("([^"/]+)\.py"\)', rust))
        self.assertTrue(invoked)
        stages = {stage["name"]: stage for stage in verify.plan("cargo", full=True)}
        self.assertEqual(set(stages["ptxas"]["python_suites"]), invoked)
        self.assertTrue(invoked <= set(stages["workspace"]["python_suites"]))

    def test_stage_selection_lists_only_requested_checks_and_requires_explicit_workspace(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(verify.main(["--stage", "ptxas", "--list"]), 0)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("ptxas:"))
        self.assertIn("--test ptxas_verification", lines[0])
        with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
            verify.main(["--stage", "workspace", "--list"])
        self.assertEqual(error.exception.code, 2)

    def stage(self, script, suites=()):
        return {"name": "probe", "command": [sys.executable, "-c", script],
                "requires": [], "python_suites": list(suites)}

    def test_zero_exit_without_tests_or_sidecars_is_failure(self):
        for script, suites in (("pass", ()),
                               ("print('test result: ok. 1 passed; 0 failed; 0 ignored;')", ("suite",))):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as temporary:
                result = verify.run_stage(self.stage(script, suites), Path(temporary), {}, {}, Path(temporary), 5)
                self.assertEqual(result["status"], "failed")
                self.assertIn("error", result)

    def test_partial_and_complete_actual_processes_have_distinct_outcomes(self):
        for skip in (False, True):
            with self.subTest(skip=skip), tempfile.TemporaryDirectory() as temporary:
                script = "print('test result: ok. 1 passed; 0 failed; 0 ignored;')"
                if skip:
                    script += "; print('SKIP: optional device unavailable')"
                result = verify.run_stage(self.stage(script), Path(temporary), {}, {}, Path(temporary), 5)
                self.assertEqual(result["status"], "incomplete" if skip else "passed")

    def test_python_failure_evidence_cannot_be_hidden_by_a_zero_process_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = sidecar(statuses=("skip",), strict=True)
            script = ("import os,pathlib; "
                      f"pathlib.Path(os.environ['Y_VERIFICATION_RESULT_DIR'],'suite.json').write_text({json.dumps(data)!r}); "
                      "print('test result: ok. 1 passed; 0 failed; 0 ignored;')")
            result = verify.run_stage(self.stage(script, ["suite"]), Path(temporary),
                                      {"Y_VERIFICATION_STRICT": "1"}, {}, Path(temporary), 5)
            self.assertEqual(result["status"], "failed")

    def test_timeout_preserves_log_and_reports_failure(self):
        for suites in ((), ("suite",)):
            with self.subTest(suites=suites), tempfile.TemporaryDirectory() as temporary:
                result = verify.run_stage(self.stage("import time; print('started',flush=True); time.sleep(10)", suites),
                                          Path(temporary), {}, {}, Path(temporary), 0.1)
                self.assertEqual(result["status"], "failed")
                self.assertIn("deadline", result["error"])
                if suites:
                    self.assertIn("expected exactly one Python report", result["error"])
                self.assertIn("started", (Path(temporary) / "probe.log").read_text())

    def test_empty_explicit_tool_overrides_do_not_use_fallbacks(self):
        tools = verify.discover(verify.ROOT, dict(os.environ, Y_TVAL_PYTHON="", Y_Z3_PATH=""))
        for name in ("python", "python_z3", "z3"):
            self.assertFalse(tools[name]["available"], f"{name} silently ignored an explicit empty override")

    def test_missing_prerequisite_does_not_launch_a_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = self.stage("raise RuntimeError('must not execute')")
            stage["requires"] = ["coqc"]
            result = verify.run_stage(stage, Path(temporary), {}, {"coqc": {"available": False}},
                                      Path(temporary), 5)
            self.assertEqual(result["status"], "skipped")
            self.assertIn("coqc", result["reason"])
            self.assertFalse((Path(temporary) / "probe.log").exists())

    def test_input_mutation_invalidates_final_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            (root / "src").mkdir(parents=True)
            source = root / "src/lib.rs"
            source.write_text("before")
            output = Path(temporary) / "results"
            tools = {name: {"available": True, "path": sys.executable} for name in
                     ("cargo", "rustc", "z3", "clang", "coqc", "ptxas", "nvdisasm", "python", "python_z3")}
            def mutate(*args):
                source.write_text("after")
                return {"name": "probe", "status": "passed"}
            with mock.patch.object(verify, "ROOT", root), mock.patch.object(verify, "discover", return_value=tools), \
                    mock.patch.object(verify, "plan", return_value=[{"name": "probe"}]), \
                    mock.patch.object(verify, "run_stage", side_effect=mutate), \
                    mock.patch.dict(os.environ, {name: value for name, value in os.environ.items()
                                               if name != "Y_ALLOW_UNVERIFIED_INVARIANTS"}, clear=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(verify.main(["--allow-skips", "--output", str(output)]), 1)
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(report["changed_inputs"], ["src/lib.rs"])
            self.assertEqual(report["status"], "failed")
            self.assertTrue((output / "summary.md").exists())

    def test_existing_evidence_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "results.json"
            evidence.write_text("retained")
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                verify.main(["--output", temporary])
            self.assertEqual(evidence.read_text(), "retained")

    def test_all_unverified_invariant_override_values_are_rejected(self):
        for value in ("", "0", "1", "false"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "repo"
                root.mkdir()
                output = Path(temporary) / "results"
                with mock.patch.object(verify, "ROOT", root), \
                        mock.patch.object(verify, "discover", return_value={}), \
                        mock.patch.object(verify, "run_stage") as run, \
                        mock.patch.dict(os.environ, {"Y_ALLOW_UNVERIFIED_INVARIANTS": value}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(verify.main(["--allow-skips", "--output", str(output)]), 1)
                run.assert_not_called()
                report = json.loads((output / "results.json").read_text())
                self.assertIn("unset it", report["errors"][0])

    def test_hashes_include_native_python_and_config_inputs_but_exclude_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("c_src/native.c", "python/y_lang/api.py", ".cargo/config.toml"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            evidence = root / "docs/verification/run"
            evidence.mkdir(parents=True)
            (evidence / "results.json").write_text("generated")
            before = verify.input_hashes(root, evidence)
            self.assertEqual(set(before), {"c_src/native.c", "python/y_lang/api.py", ".cargo/config.toml"})
            (root / "c_src/native.c").write_text("mutated")
            (root / "python/y_lang/api.py").unlink()
            (root / ".cargo/config.toml").write_text("new config")
            (evidence / "new.log").write_text("generated during verification")
            after = verify.input_hashes(root, evidence)
            self.assertNotEqual(before["c_src/native.c"], after["c_src/native.c"])
            self.assertNotIn("python/y_lang/api.py", after)
            self.assertNotEqual(before[".cargo/config.toml"], after[".cargo/config.toml"])
            self.assertFalse(any(name.startswith("docs/verification/run/") for name in after))


if __name__ == "__main__":
    verification_main(verbosity=2)
