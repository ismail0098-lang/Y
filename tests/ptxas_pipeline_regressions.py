#!/usr/bin/env python3
"""Fresh PTXAS translation controls, without CUDA driver or GPU execution.

Each validator runs in its own interpreter to isolate solver and module state.
Y_PTXAS_EVIDENCE_DIR retains unique
case directories; otherwise artifacts live in a temporary directory. Evidence
describes genuine translations or explicit disassembly mutations, not a
deployable validation receipt.
"""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import verification_unittest


REPO = Path(__file__).resolve().parents[1]
TVAL = REPO / "tools" / "ptxas_tval"
INSTRUCTION = re.compile(r"(?m)^(\s*/\*[0-9a-fA-F]+\*/\s*)(.*?)(\s*;\s*)$")
sys.path.insert(0, str(REPO / "tools"))
from ptxas_tval.validation_child import VALIDATOR_CHILD


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_inventory(root, cases, *, create=False):
    """Publish the complete case list without exposing a partial JSON write."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix=".inventory-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump({"format": "y-ptxas-inventory-v1", "cases": cases}, output,
                      indent=2, sort_keys=True)
            output.write("\n")
        destination = root / "inventory.json"
        if create:
            os.link(temporary, destination)  # An existing inventory is never replaced at setup.
        else:
            os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def instruction_stream(text):
    return [" ".join(match.group(2).split()) for match in INSTRUCTION.finditer(text)]


def integer_kernel(name, parameters, instructions, *, target="sm_89"):
    return (f".version 7.8\n.target {target}\n.address_size 64\n"
            f".visible .entry {name}({parameters})\n{{\n"
            ".reg .b32 %r<12>;\n.reg .b64 %rd<8>;\n.reg .pred %p<3>;\n"
            + instructions.strip() + "\nret;\n}\n").encode()


class Evidence:
    def __init__(self, test, name, expected, *, validator="tval", role="genuine",
                 optimization=1, target="sm_89", expected_detail=""):
        self.test = test
        self.directory = Path(tempfile.mkdtemp(prefix=name + "-", dir=test.evidence_root))
        test.case_inventory.append({"name": name, "test_id": test.id(),
                                    "directory": self.directory.name})
        write_inventory(test.evidence_root, test.case_inventory)
        self.metadata = {
            "format": "y-ptxas-case-v1", "name": name, "role": role,
            "test_id": test.id(),
            "target": target, "optimization": str(optimization), "expected": expected,
            "expected_detail": expected_detail, "validator": validator,
            "verdict": "REFUSED", "detail": "case did not complete", "obligations": 0,
            "commands": list(test.version_commands), "files": {},
        }
        (self.directory / "tool_versions.json").write_text(
            json.dumps(test.tool_versions, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def run(self, command, label, *, timeout=60):
        command = [str(argument) for argument in command]
        try:
            result = subprocess.run(command, cwd=REPO, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            result = subprocess.CompletedProcess(command, 124, error.stdout or b"", error.stderr or b"")
            self.metadata["detail"] = f"{label} exceeded {timeout}s wall limit"
        self.metadata["commands"].append({"command": command, "exit_code": result.returncode,
                                           "cwd": str(REPO)})
        (self.directory / f"{label}.stdout.txt").write_bytes(result.stdout)
        (self.directory / f"{label}.stderr.txt").write_bytes(result.stderr)
        self.test.assertNotEqual(result.returncode, 124, self.metadata["detail"])
        return result

    def assemble(self, source, *, expect_error=False):
        (self.directory / "source.ptx").write_bytes(source)
        result = self.run([
            self.test.ptxas, f"-O{self.metadata['optimization']}",
            f"-arch={self.metadata['target']}", self.directory / "source.ptx",
            "-o", self.directory / "kernel.cubin",
        ], "ptxas")
        if expect_error:
            self.metadata.update(verdict="ASSEMBLY_REFUSED", detail=result.stderr.decode(errors="replace"))
            self.test.assertNotEqual(result.returncode, 0, "invalid PTX unexpectedly assembled")
            self.test.assertFalse((self.directory / "kernel.cubin").exists())
            self.check_result()
            return
        self.test.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        cubin = self.directory / "kernel.cubin"
        self.test.assertTrue(cubin.read_bytes().startswith(b"\x7fELF"), "ptxas did not produce an ELF cubin")
        disassembly = self.run([self.test.nvdisasm, "-c", cubin], "nvdisasm")
        self.test.assertEqual(disassembly.returncode, 0, disassembly.stderr.decode(errors="replace"))
        (self.directory / "kernel.sass").write_bytes(disassembly.stdout)
        targets = re.findall(rb"(?m)^\s*\.target\s+(sm_[0-9]+[af]?)\s*$", disassembly.stdout)
        self.test.assertEqual(targets, [self.metadata["target"].encode()])

    def validate(self, *, sass_name="kernel.sass"):
        self.metadata["validation_sass"] = sass_name
        result = self.run([
            sys.executable, "-c", VALIDATOR_CHILD, TVAL, self.metadata["validator"],
            self.directory / "source.ptx", self.directory / sass_name,
        ], "validator", timeout=120)
        self.test.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        data = json.loads(result.stdout)
        self.test.assertEqual(data.keys(), {"format", "verdict", "detail", "obligations", "log"})
        self.test.assertEqual(data["format"], "y-ptxas-validation-v1")
        self.test.assertIn(data["verdict"], ("VALIDATED", "UNPROVED", "REFUSED"))
        self.test.assertIs(type(data["obligations"]), int)
        self.test.assertGreaterEqual(data["obligations"], 0)
        self.test.assertIs(type(data["detail"]), str)
        self.test.assertIs(type(data["log"]), str)
        (self.directory / "validation.json").write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.metadata.update(verdict=data["verdict"], detail=data["detail"],
                             obligations=data["obligations"], diagnostic_log=data["log"])
        self.check_result()
        return data

    def check_result(self):
        self.test.assertEqual(self.metadata["verdict"], self.metadata["expected"],
                              self.metadata["detail"] + "\n" + self.metadata.get("diagnostic_log", ""))
        if self.metadata["verdict"] == "VALIDATED":
            self.test.assertGreater(self.metadata["obligations"], 0, "acceptance without proof obligations")
        if self.metadata["expected_detail"]:
            self.test.assertRegex(self.metadata["detail"] + "\n" + self.metadata.get("diagnostic_log", ""),
                                  self.metadata["expected_detail"])

    def finish(self):
        self.metadata["files"] = {
            path.name: digest(path) for path in sorted(self.directory.iterdir())
            if path.is_file() and path.name != "case.json"
        }
        (self.directory / "case.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class PtxasPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [tool for tool in ("ptxas", "nvdisasm") if not shutil.which(tool)]
        if importlib.util.find_spec("z3") is None:
            missing.append("Python z3-solver")
        if missing:
            reason = "PTXAS pipeline requires " + ", ".join(missing)
            if os.environ.get("Y_VERIFICATION_STRICT") == "1":
                raise RuntimeError(reason + "; strict verification cannot skip these checks")
            raise unittest.SkipTest(reason)
        cls.ptxas, cls.nvdisasm = shutil.which("ptxas"), shutil.which("nvdisasm")
        if destination := os.environ.get("Y_PTXAS_EVIDENCE_DIR"):
            cls.evidence_root = Path(destination).resolve()
            cls.evidence_root.mkdir(parents=True, exist_ok=True)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="y-ptxas-pipeline-")
            cls.addClassCleanup(temporary.cleanup)
            cls.evidence_root = Path(temporary.name)
        cls.case_inventory = []
        write_inventory(cls.evidence_root, cls.case_inventory, create=True)
        cls.tool_versions, cls.version_commands = {}, []
        commands = {
            "ptxas": [cls.ptxas, "--version"], "nvdisasm": [cls.nvdisasm, "--version"],
            "python_z3": [sys.executable, "-c", "import sys,z3; print(sys.version); print(z3.get_version_string())"],
        }
        for name, command in commands.items():
            result = subprocess.run(command, cwd=REPO, capture_output=True, check=True, timeout=15)
            cls.version_commands.append({"command": command, "exit_code": result.returncode,
                                         "cwd": str(REPO)})
            cls.tool_versions[name] = (result.stdout + result.stderr).decode(errors="replace")

    @contextmanager
    def case(self, name, expected="VALIDATED", **kwargs):
        evidence = Evidence(self, name, expected, **kwargs)
        try:
            yield evidence
        finally:
            evidence.finish()

    def mutate_instruction(self, case, pattern, replacement):
        """Change one asserted instruction body while retaining its PC and cubin."""
        text = (case.directory / "kernel.sass").read_text()
        candidates = [match for match in INSTRUCTION.finditer(text)
                      if re.fullmatch(pattern, match.group(2))]
        self.assertEqual(len(candidates), 1, "mutation anchor moved or became ambiguous")
        instruction = candidates[0]
        changed = re.sub(pattern, replacement, instruction.group(2))
        self.assertNotEqual(changed, instruction.group(2), "mutation did not change the instruction")
        mutated = text[:instruction.start(2)] + changed + text[instruction.end(2):]
        (case.directory / "mutated.sass").write_text(mutated)
        case.metadata["mutation"] = {"instruction": instruction.group(2), "replacement": changed}

    def test_integer_signedness_and_computed_addresses_at_o2_and_o3(self):
        # Keep real O3 counterexamples in the licensed model as well as
        # assembling the sm86 portability subjects that now refuse validation.
        for optimization, target, comparison in ((2, "sm_89", "s32"),
                (3, "sm_89", "u32"), (3, "sm_86", "u32")):
            source = integer_kernel("integer_address", ".param .u64 O, .param .u32 X, .param .u32 Y", f"""
ld.param.u64 %rd0, [O];
ld.param.u32 %r0, [X];
ld.param.u32 %r1, [Y];
setp.lt.{comparison} %p0, %r0, %r1;
selp.u32 %r2, 0x11223344, 0x55667788, %p0;
xor.b32 %r3, %r2, 0xa5a5a5a5;
mov.u32 %r4, %tid.x;
shl.b32 %r5, %r4, 1;
add.u32 %r6, %r5, 3;
cvt.u64.u32 %rd1, %r6;
shl.b64 %rd2, %rd1, 2;
add.u64 %rd3, %rd0, %rd2;
st.global.u32 [%rd3], %r3;
""", target=target)
            for mutate in (False, True):
                with self.subTest(optimization=optimization, target=target, comparison=comparison, mutate=mutate):
                    with self.case(f"integer-address-{comparison}-o{optimization}-{target}-"
                                   + ("wrong-signedness" if mutate else "genuine"),
                                   ("UNPROVED" if mutate else "VALIDATED") if target == "sm_89" else "REFUSED",
                                   optimization=optimization,
                                   target=target, role="mutated" if mutate else "genuine",
                                   expected_detail=((r"store 0: sat" if mutate else "")
                                                    if target == "sm_89" else "unsupported architecture")) as case:
                        case.assemble(source)
                        stream = instruction_stream((case.directory / "kernel.sass").read_text())
                        self.assertTrue(any(line.startswith("LEA ") for line in stream), "computed address was optimized away")
                        self.assertTrue(any(line.startswith("LOP3.LUT ") for line in stream), "integer bit operation was optimized away")
                        if mutate:
                            suffix = r"\.U32" if comparison == "u32" else ""
                            changed = "" if comparison == "u32" else ".U32"
                            self.mutate_instruction(case, r"ISETP\.(LT|GE)" + suffix + r"\.AND (.*)",
                                                    r"ISETP.\1" + changed + r".AND \2")
                        case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")

    def test_o0_constant_load_is_explicitly_refused(self):
        source = integer_kernel("unoptimized_integer", ".param .u64 O, .param .u32 X", """
ld.param.u64 %rd0, [O];
ld.param.u32 %r0, [X];
add.u32 %r1, %r0, 3;
st.global.u32 [%rd0], %r1;
""")
        with self.case("integer-o0-unmodelled-constant-load", "REFUSED", optimization=0,
                       expected_detail=r"UNMODELLED SASS OPCODE 'LDC\.64'.*refusing, not guessing") as case:
            case.assemble(source)
            stream = instruction_stream((case.directory / "kernel.sass").read_text())
            self.assertTrue(any(line.startswith("LDC.64 ") for line in stream))
            self.assertEqual(case.validate()["obligations"], 0)

    def test_instruction_predicates_preserve_conditional_stores(self):
        source = integer_kernel("conditional_store", ".param .u64 O, .param .u32 X", """
ld.param.u64 %rd0, [O];
ld.param.u32 %r0, [X];
mov.u32 %r1, %tid.x;
setp.lt.u32 %p0, %r1, 4;
@%p0 st.global.u32 [%rd0], %r0;
add.u64 %rd1, %rd0, 4;
st.global.u32 [%rd1], %r1;
""")
        for mutation in (None, "dropped", "guard"):
            with self.subTest(mutation=mutation):
                diagnostic = (r"load/store counts 0/0 2/1" if mutation == "dropped"
                              else r"store 0 guard: sat" if mutation else "")
                with self.case("conditional-store-o2-" + (mutation or "genuine"),
                               "UNPROVED" if mutation else "VALIDATED", optimization=2,
                               role="mutated" if mutation else "genuine",
                               expected_detail=diagnostic) as case:
                    case.assemble(source)
                    stream = instruction_stream((case.directory / "kernel.sass").read_text())
                    self.assertEqual(sum(bool(re.fullmatch(r"@!?P\d+ STG\.E .*", line)) for line in stream), 1)
                    if mutation == "dropped":
                        self.mutate_instruction(case, r"@!?P\d+\s+STG\.E .*", "NOP")
                    elif mutation == "guard":
                        self.mutate_instruction(case, r"@(!?P\d+)\s+(STG\.E .*)",
                                                lambda m: '@' + (m[1][1:] if m[1].startswith('!') else '!' + m[1])
                                                + ' ' + m[2])
                    case.validate(sass_name="mutated.sass" if mutation else "kernel.sass")

    def test_global_load_offsets_and_changed_addresses(self):
        for name, transfer, offset in (
                ("word-plus", "ld.global.u32 %r0, [ADDRESS];\nst.global.u32 [%rd1], %r0;", "+4"),
                ("word-minus", "ld.global.u32 %r0, [ADDRESS];\nst.global.u32 [%rd1], %r0;", "+-4"),
                ("signed-half", "ld.global.s16 %r0, [ADDRESS];\nst.global.u32 [%rd1], %r0;", "+2"),
                ("vector", "ld.global.v4.u32 {%r0,%r1,%r2,%r3}, [ADDRESS];\n"
                 "st.global.v4.u32 [%rd1], {%r0,%r1,%r2,%r3};", "+16"),
                ("float", ".reg .f32 %f<2>;\nld.global.f32 %f0, [ADDRESS];\n"
                 "st.global.f32 [%rd1], %f0;", "+4")):
            source = integer_kernel("global_offset_load", ".param .u64 I, .param .u64 O", """
ld.param.u64 %rd0, [I];
ld.param.u64 %rd1, [O];
""" + transfer.replace("ADDRESS", "%rd0" + offset))
            for mutate in (False, True):
                with self.subTest(name=name, mutate=mutate):
                    with self.case("global-offset-" + name + "-o1-" + ("changed" if mutate else "genuine"),
                                   "UNPROVED" if mutate else "VALIDATED",
                                   role="mutated" if mutate else "genuine",
                                   expected_detail=r"load 0 address: sat" if mutate else "") as case:
                        case.assemble(source)
                        if mutate:
                            self.mutate_instruction(case,
                                                    r"(LDG\.E(?:\.S16|\.128)? R\d+, \[R\d+\.64)"
                                                    r"(?:\+-?0x[0-9a-f]+)?\]", r"\1+0x20]")
                        case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")

    def test_unsupported_trailing_integer_operands_refuse(self):
        for name, instructions, pattern in (
                ("comparison", """
setp.lt.u32 %p0, %r0, %r1;
selp.u32 %r2, 0x11223344, 0x55667788, %p0;
st.global.u32 [%rd0], %r2;
""", r"(ISETP\.[^ ]+ .*)"),
                ("imadx", """
sub.cc.u32 %r2, %r0, %r1;
subc.u32 %r3, 0, 0;
st.global.v2.u32 [%rd0], {%r2,%r3};
""", r"(IMAD\.X .*)"),
                ("iadd3", """
add.cc.u32 %r2, %r0, %r1;
addc.u32 %r3, 0, 0;
st.global.v2.u32 [%rd0], {%r2,%r3};
""", r"(IADD3 R\d+, P0, .*)"),
                ("iadd3x", """
sub.cc.u32 %r2, %r0, %r1;
subc.cc.u32 %r3, %r0, %r1;
subc.u32 %r4, 0, 0;
st.global.v4.u32 [%rd0], {%r2,%r3,%r4,%r0};
""", r"(IADD3\.X .*)")):
            source = integer_kernel("integer_operand_count", ".param .u64 O, .param .u32 X, .param .u32 Y", """
ld.param.u64 %rd0, [O];
ld.param.u32 %r0, [X];
ld.param.u32 %r1, [Y];
""" + instructions)
            for mutate in (False, True):
                with self.subTest(name=name, mutate=mutate):
                    with self.case("integer-operand-count-" + name + "-o1-" + ("extra" if mutate else "genuine"),
                                   "REFUSED" if mutate else "VALIDATED",
                                   role="mutated" if mutate else "genuine",
                                   expected_detail=r"UNMODELLED SASS OPERAND COUNT" if mutate else "") as case:
                        case.assemble(source)
                        if mutate:
                            self.mutate_instruction(case, pattern, r"\1, !PT")
                        result = case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")
                        if mutate:
                            self.assertEqual(result["obligations"], 0)

    def test_subword_load_signedness_has_counterexamples(self):
        for kind, optimization, target in (("s8", 2, "sm_89"), ("u8", 2, "sm_89"),
                ("s16", 3, "sm_89"), ("u16", 3, "sm_89"),
                ("s16", 3, "sm_86"), ("u16", 3, "sm_86")):
            source = integer_kernel("subword_load", ".param .u64 I, .param .u64 O", f"""
ld.param.u64 %rd0, [I];
ld.param.u64 %rd1, [O];
ld.global.{kind} %r0, [%rd0];
st.global.u32 [%rd1], %r0;
""", target=target)
            for mutate in (False, True):
                with self.subTest(kind=kind, optimization=optimization, target=target, mutate=mutate):
                    with self.case(f"load-{kind}-o{optimization}-{target}-"
                                   + ("wrong-signedness" if mutate else "genuine"),
                                   ("UNPROVED" if mutate else "VALIDATED") if target == "sm_89" else "REFUSED",
                                   optimization=optimization,
                                   target=target, role="mutated" if mutate else "genuine",
                                   expected_detail=((r"store 0: sat" if mutate else "")
                                                    if target == "sm_89" else "unsupported architecture")) as case:
                        case.assemble(source)
                        if mutate:
                            changed = ("U" if kind.startswith("s") else "S") + kind[1:]
                            self.mutate_instruction(case, r"LDG\.E\." + kind.upper() + r" (.*)",
                                                    "LDG.E." + changed + r" \1")
                        case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")

    def test_global_loads_require_matching_access_width(self):
        for vector in (False, True):
            load = ("ld.global.v4.u32 {%r0,%r1,%r2,%r3}, [%rd0];" if vector else
                    "ld.global.u32 %r0, [%rd0];")
            store = ("st.global.v4.u32 [%rd1], {%r0,%r1,%r2,%r3};" if vector else
                     "st.global.u32 [%rd1], %r0;")
            source = integer_kernel("load_extent", ".param .u64 I, .param .u64 O", f"""
ld.param.u64 %rd0, [I];
ld.param.u64 %rd1, [O];
{load}
{store}
""")
            for validator in ("tval", "smemval"):
                for mutate in (False, True):
                    with self.subTest(vector=vector, validator=validator, mutate=mutate):
                        wp, ws = (128, 32) if vector else (32, 128)
                        with self.case(f"load-extent-{wp}-{validator}-"
                                       + ("changed-width" if mutate else "genuine"),
                                       "UNPROVED" if mutate else "VALIDATED", validator=validator,
                                       role="mutated" if mutate else "genuine",
                                       expected_detail=(rf"load 0 width: ptx {wp} bits, sass {ws} bits"
                                                        if mutate else "")) as case:
                            case.assemble(source)
                            if mutate:
                                pattern = r"LDG\.E\.128 (.*)" if vector else r"LDG\.E (.*)"
                                replacement = r"LDG.E \1" if vector else r"LDG.E.128 \1"
                                self.mutate_instruction(case, pattern, replacement)
                            case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")

    def test_subword_store_widening_is_rejected_by_width(self):
        for optimization, target in ((2, "sm_89"), (3, "sm_89"), (3, "sm_86")):
            source = integer_kernel("byte_store", ".param .u64 I, .param .u64 O", """
ld.param.u64 %rd0, [I];
ld.param.u64 %rd1, [O];
ld.global.u32 %r0, [%rd0];
st.global.u8 [%rd1], %r0;
""", target=target)
            for mutate in (False, True):
                with self.subTest(optimization=optimization, target=target, mutate=mutate):
                    with self.case(f"byte-store-o{optimization}-{target}-" + ("widened" if mutate else "genuine"),
                                   ("UNPROVED" if mutate else "VALIDATED") if target == "sm_89" else "REFUSED",
                                   optimization=optimization,
                                   target=target, role="mutated" if mutate else "genuine",
                                   expected_detail=((r"store 0 width: ptx 8 bits, sass 32 bits" if mutate else "")
                                                    if target == "sm_89" else "unsupported architecture")) as case:
                        case.assemble(source)
                        if mutate:
                            self.mutate_instruction(case, r"STG\.E\.U8 (.*)", r"STG.E \1")
                        case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")

    def test_wide_multiply_signedness_changes_high_word(self):
        for kind in ("u32", "s32"):
            source = integer_kernel("wide_product", ".param .u64 O, .param .u32 N", f"""
ld.param.u64 %rd0, [O];
ld.param.u32 %r0, [N];
mul.wide.{kind} %rd1, %r0, 2;
st.global.u64 [%rd0], %rd1;
""")
            for mutate in (False, True):
                with self.subTest(kind=kind, mutate=mutate):
                    with self.case(f"wide-product-{kind}-o1-" + ("wrong-signedness" if mutate else "genuine"),
                                   "UNPROVED" if mutate else "VALIDATED",
                                   role="mutated" if mutate else "genuine",
                                   expected_detail=r"store 1: sat" if mutate else "") as case:
                        case.assemble(source)
                        if mutate:
                            suffix = r"\.U32" if kind == "u32" else ""
                            changed = "" if kind == "u32" else ".U32"
                            self.mutate_instruction(case, r"IMAD\.WIDE" + suffix + r" (.*)",
                                                    "IMAD.WIDE" + changed + r" \1")
                        result = case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")
                        if not mutate:
                            self.assertEqual(result["detail"].split(",")[0], "2 stores")

    def test_zero_register_pair_store_and_wrong_source(self):
        source = integer_kernel("zero_wide_store", ".param .u64 O", """
ld.param.u64 %rd0, [O];
mov.u64 %rd1, 0;
st.global.u64 [%rd0], %rd1;
""")
        for mutate in (False, True):
            with self.subTest(mutate=mutate):
                with self.case("zero-wide-store-o1-" + ("wrong-source" if mutate else "genuine"),
                               "UNPROVED" if mutate else "VALIDATED",
                               role="mutated" if mutate else "genuine",
                               expected_detail=r"store 0: sat" if mutate else "") as case:
                    case.assemble(source)
                    stream = instruction_stream((case.directory / "kernel.sass").read_text())
                    self.assertEqual(sum(bool(re.fullmatch(r"STG\.E\.64 \[R\d+\.64\], RZ", line))
                                         for line in stream), 1)
                    if mutate:
                        self.mutate_instruction(case, r"STG\.E\.64 \[R(\d+)\.64\], RZ",
                                                r"STG.E.64 [R\1.64], R\1")
                    result = case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")
                    if not mutate:
                        self.assertEqual(result["detail"].split(",")[0], "2 stores")

    def test_zero_register_global_vector_store_and_source_controls(self):
        source = integer_kernel("zero_vector_store", ".param .u64 O", """
ld.param.u64 %rd0, [O];
mov.u32 %r0, 0;
st.global.v4.u32 [%rd0], {%r0,%r0,%r0,%r0};
""")
        for mutation, expected, diagnostic in (
                (None, "VALIDATED", ""),
                ("pointer", "UNPROVED", r"store 0: sat"),
                ("uniform", "REFUSED", r"UNMODELLED SASS vector store source 'URZ'")):
            with self.subTest(mutation=mutation):
                with self.case("zero-global128-o1-" + (mutation or "genuine"), expected,
                               role="mutated" if mutation else "genuine",
                               expected_detail=diagnostic) as case:
                    case.assemble(source)
                    stream = instruction_stream((case.directory / "kernel.sass").read_text())
                    self.assertEqual(sum(bool(re.fullmatch(r"STG\.E\.128 \[R\d+\.64\], RZ", line))
                                         for line in stream), 1)
                    if mutation:
                        replacement = (r"STG.E.128 [R\1.64], R\1" if mutation == "pointer"
                                       else r"STG.E.128 [R\1.64], URZ")
                        self.mutate_instruction(case, r"STG\.E\.128 \[R(\d+)\.64\], RZ", replacement)
                    case.validate(sass_name="mutated.sass" if mutation else "kernel.sass")

    def test_register_multiplier_swaps_and_changed_products(self):
        source = integer_kernel("multiply_loaded_pair",
                                ".param .u64 I, .param .u64 J, .param .u64 O", """
ld.param.u64 %rd0, [I];
ld.param.u64 %rd1, [J];
ld.param.u64 %rd2, [O];
ld.global.u32 %r0, [%rd0];
ld.global.u32 %r1, [%rd1];
mul.wide.u32 %rd3, %r0, %r1;
st.global.u64 [%rd2], %rd3;
""")
        for mutation in (None, "swapped", "changed"):
            with self.subTest(mutation=mutation):
                with self.case("multiply-registers-o1-" + (mutation or "genuine"),
                               "UNPROVED" if mutation == "changed" else "VALIDATED",
                               role="mutated" if mutation else "genuine",
                               expected_detail=r"store 0: sat" if mutation == "changed" else "") as case:
                    case.assemble(source)
                    if mutation:
                        replacement = (r"IMAD.WIDE.U32 \1, \3, \2, RZ" if mutation == "swapped"
                                       else r"IMAD.WIDE.U32 \1, \2, 0x1, RZ")
                        self.mutate_instruction(case, r"IMAD\.WIDE\.U32 (R\d+), (R\d+), (R\d+), RZ",
                                                replacement)
                    case.validate(sass_name="mutated.sass" if mutation else "kernel.sass")

    def test_variable_right_shift_signedness_controls(self):
        for kind in ("u32", "s32"):
            source = integer_kernel("variable_right_shift", ".param .u64 O, .param .u32 X, .param .u32 Y", f"""
ld.param.u64 %rd0, [O];
ld.param.u32 %r0, [X];
ld.param.u32 %r1, [Y];
shr.{kind} %r2, %r0, %r1;
st.global.u32 [%rd0], %r2;
""")
            for mutate in (False, True):
                with self.subTest(kind=kind, mutate=mutate):
                    with self.case(f"right-shift-{kind}-o1-" + ("wrong-signedness" if mutate else "genuine"),
                                   "UNPROVED" if mutate else "VALIDATED",
                                   role="mutated" if mutate else "genuine",
                                   expected_detail=r"store 0: sat" if mutate else "") as case:
                        case.assemble(source)
                        if mutate:
                            changed = "S32" if kind == "u32" else "U32"
                            self.mutate_instruction(case, r"SHF\.R\." + kind.upper() + r"\.HI (.*)",
                                                    "SHF.R." + changed + r".HI \1")
                        case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")

    def test_carry_chains_and_dropped_carry_inputs(self):
        for kind, instructions, pattern, replacement in (
                ("add", """
add.cc.u32 %r2, %r0, %r1;
addc.u32 %r3, 0, 0;
st.global.v2.u32 [%rd0], {%r2,%r3};
""", r"IADD3 (R\d+), P0, (.*)", r"IADD3 \1, PT, \2"),
                ("sub", """
sub.cc.u32 %r2, %r0, %r1;
subc.cc.u32 %r3, %r0, %r1;
subc.u32 %r4, 0, 0;
st.global.v4.u32 [%rd0], {%r2,%r3,%r4,%r0};
""", r"(IADD3\.X .*), P0, !PT", r"\1, !PT, !PT")):
            source = integer_kernel("carry_chain", ".param .u64 O, .param .u32 X, .param .u32 Y", """
ld.param.u64 %rd0, [O];
ld.param.u32 %r0, [X];
ld.param.u32 %r1, [Y];
""" + instructions)
            for mutation in ((None, "dropped", "predicate-file") if kind == "add" else (None, "dropped")):
                with self.subTest(kind=kind, mutation=mutation):
                    expected = ("REFUSED" if mutation == "predicate-file"
                                else "UNPROVED" if mutation else "VALIDATED")
                    diagnostic = (r"UNMODELLED SASS predicate operand 'R0'" if mutation == "predicate-file"
                                  else r"store 1: sat" if mutation else "")
                    with self.case(f"{kind}-carry-chain-o1-" + (mutation or "genuine"), expected,
                                   role="mutated" if mutation else "genuine",
                                   expected_detail=diagnostic) as case:
                        case.assemble(source)
                        if mutation == "predicate-file":
                            self.mutate_instruction(case, r"(SEL .*), !P0", r"\1, !R0")
                        elif mutation:
                            self.mutate_instruction(case, pattern, replacement)
                        case.validate(sass_name="mutated.sass" if mutation else "kernel.sass")

    def test_uninitialized_carry_refuses_genuine_and_forced_zero(self):
        source = integer_kernel("uninitialized_carry", ".param .u64 O, .param .u32 X, .param .u32 Y", """
ld.param.u64 %rd0, [O];
ld.param.u32 %r0, [X];
ld.param.u32 %r1, [Y];
addc.u32 %r2, %r0, %r1;
st.global.u32 [%rd0], %r2;
""")
        for mutate in (False, True):
            with self.subTest(mutate=mutate):
                with self.case("entry-carry-o1-" + ("forced-zero" if mutate else "genuine"), "REFUSED",
                               role="mutated" if mutate else "genuine",
                               expected_detail=r"UNMODELLED PTX carry read.*CC may be uninitialized") as case:
                    case.assemble(source)
                    if mutate:
                        self.mutate_instruction(case, r"(IADD3\.X .*), P0, !PT", r"\1, !PT, !PT")
                    self.assertEqual(case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")
                                     ["obligations"], 0)

    def test_variable_wide_shift_has_named_machine_refusal(self):
        source = integer_kernel("wide_variable_shift", ".param .u64 O, .param .u64 X, .param .u32 N", """
ld.param.u64 %rd0, [O];
ld.param.u64 %rd1, [X];
ld.param.u32 %r0, [N];
shl.b64 %rd2, %rd1, %r0;
st.global.u64 [%rd0], %rd2;
""")
        with self.case("wide-variable-left-shift-o1-unmodeled", "REFUSED",
                       expected_detail=r"UNMODELLED SASS OPCODE 'USHF\.L\.U64\.HI'") as case:
            case.assemble(source)
            stream = instruction_stream((case.directory / "kernel.sass").read_text())
            self.assertTrue(any(line.startswith("USHF.L.U64.HI ") for line in stream))
            self.assertEqual(case.validate()["obligations"], 0)

    def test_zero_register_shared_vectors_and_symbol_offsets(self):
        for bits in (64, 128):
            if bits == 64:
                transfers = """
st.shared.u32 [slot], 0;
st.shared.u32 [slot+4], 0;
bar.sync 0;
ld.shared.u32 %r0, [slot];
ld.shared.u32 %r1, [slot+4];
st.global.u32 [%rd0], %r0;
add.u64 %rd1, %rd0, 4;
st.global.u32 [%rd1], %r1;
"""
            else:
                transfers = """
st.shared.v4.u32 [slot], {0,0,0,0};
bar.sync 0;
ld.shared.v4.u32 {%r0,%r1,%r2,%r3}, [slot];
st.global.v4.u32 [%rd0], {%r0,%r1,%r2,%r3};
"""
            source = integer_kernel("zero_shared_vector", ".param .u64 O",
                                    f".shared .align 16 .b8 slot[{bits//8}];\n"
                                    "ld.param.u64 %rd0, [O];\n" + transfers)
            for mutate in (False, True):
                with self.subTest(bits=bits, mutate=mutate):
                    with self.case(f"zero-shared{bits}-o1-" + ("wrong-source" if mutate else "genuine"),
                                   "UNPROVED" if mutate else "VALIDATED",
                                   role="mutated" if mutate else "genuine",
                                   expected_detail=(r"shared memory entering barrier 0: REFUTED \(sat\)"
                                                    if mutate else "")) as case:
                        case.assemble(source)
                        stream = instruction_stream((case.directory / "kernel.sass").read_text())
                        anchor = rf"STS\.{bits} \[RZ\], RZ"
                        self.assertEqual(sum(bool(re.fullmatch(anchor, line)) for line in stream), 1)
                        if mutate:
                            self.mutate_instruction(case, anchor, f"STS.{bits} [RZ], R1")
                        result = case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")
                        if not mutate:
                            self.assertEqual(result["detail"].split(",")[0], f"{bits//32} stores")

    def test_shared_vector_accesses_require_natural_alignment(self):
        for misaligned in (False, True):
            source = integer_kernel("vector_alignment", ".param .u64 O", f"""
.shared .align 16 .b8 slot[16400];
ld.param.u64 %rd0, [O];
mov.u32 %r0, %tid.x;
shl.b32 %r1, %r0, 4;
add.u32 %r2, %r1, {4 if misaligned else 0};
st.shared.v4.u32 [%r2], {{0,0,0,0}};
bar.sync 0;
ld.shared.v4.u32 {{%r4,%r5,%r6,%r7}}, [%r2];
st.global.v4.u32 [%rd0], {{%r4,%r5,%r6,%r7}};
""")
            with self.subTest(misaligned=misaligned):
                with self.case("shared-vector-alignment-" + ("misaligned" if misaligned else "genuine"),
                               "UNPROVED" if misaligned else "VALIDATED",
                               expected_detail=(r"shared access is not provably.*aligned.*\[sat\]"
                                                if misaligned else "")) as case:
                    case.assemble(source)
                    stream = instruction_stream((case.directory / "kernel.sass").read_text())
                    self.assertTrue(any(line.startswith("STS.128 ") for line in stream))
                    self.assertTrue(any(line.startswith("LDS.128 ") for line in stream))
                    case.validate()

    def test_scalar_load_destination_and_modifier_spellings_refuse(self):
        source = integer_kernel("load_destination", ".param .u64 I, .param .u64 O", """
ld.param.u64 %rd0, [I];
ld.param.u64 %rd1, [O];
ld.global.u32 %r0, [%rd0];
st.global.u32 [%rd1], %r0;
""")
        for mutation in ("", "predicate-file", "scalar-width", "zero-destination"):
            with self.subTest(mutation=mutation):
                with self.case("load-destination-" + (mutation or "genuine"),
                               ("UNPROVED" if mutation == "zero-destination" else
                                "REFUSED" if mutation else "VALIDATED"),
                               role="mutated" if mutation else "genuine",
                               expected_detail=(r"store 0: sat" if mutation == "zero-destination" else
                                                r"UNMODELLED SASS.*refusing, not guessing" if mutation else "")) as case:
                    case.assemble(source)
                    if mutation == "predicate-file":
                        self.mutate_instruction(case, r"LDG\.E R(\d+), (.*)", r"LDG.E P\1, \2")
                    elif mutation == "scalar-width":
                        self.mutate_instruction(case, r"LDG\.E (R\d+), (.*)", r"LDG.E \1.64, \2")
                    elif mutation == "zero-destination":
                        self.mutate_instruction(case, r"LDG\.E R\d+, (.*)", r"LDG.E RZ, \1")
                    case.validate(sass_name="mutated.sass" if mutation else "kernel.sass")

    def test_pc_tagged_instruction_without_semicolon_is_not_ignored(self):
        source = integer_kernel("malformed_instruction", ".param .u64 O", """
ld.param.u64 %rd0, [O];
st.global.u32 [%rd0], 0;
""")
        for mutate in (False, True):
            with self.subTest(mutate=mutate):
                with self.case("malformed-instruction-" + ("inserted-effect" if mutate else "genuine"),
                               "REFUSED" if mutate else "VALIDATED",
                               role="mutated" if mutate else "genuine",
                               expected_detail=r"UNMODELLED SASS instruction line.*refusing, not guessing" if mutate else "") as case:
                    case.assemble(source)
                    if mutate:
                        text = (case.directory / "kernel.sass").read_text()
                        exit_insn = [m for m in INSTRUCTION.finditer(text) if m.group(2) == "EXIT"]
                        self.assertEqual(len(exit_insn), 1)
                        effect = "        /*ff00*/ STG.E [R0], RZ\n"
                        changed = text[:exit_insn[0].start()] + effect + text[exit_insn[0].start():]
                        (case.directory / "mutated.sass").write_text(changed)
                        case.metadata["mutation"] = {"instruction": "", "replacement": effect.strip()}
                    case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")

    def test_float_rounding_and_contraction_at_o1_and_o3(self):
        for optimization in (1, 3):
            for fixture, expected in (("rn", "VALIDATED"), ("plain", "UNPROVED")):
                with self.subTest(optimization=optimization, fixture=fixture):
                    anchor = "" if expected == "VALIDATED" else r"store 0: sat"
                    with self.case(f"float-{fixture}-o{optimization}", expected,
                                   optimization=optimization, expected_detail=anchor) as case:
                        case.assemble((TVAL / "fma" / f"{fixture}.ptx").read_bytes())
                        instructions = instruction_stream((case.directory / "kernel.sass").read_text())
                        self.assertTrue(any("FFMA" in line for line in instructions) == (fixture == "plain"))
                        case.validate()

    def test_gemm_fma_and_split_rounding_controls(self):
        source = (REPO / "tests" / "naive_gemm_f32.ptx").read_text()
        operation = re.compile(r"(?m)^(\s*)fma\.rn\.f32 (%f\d+), (%f\d+), (%f\d+), (%f\d+);$")
        matches = list(operation.finditer(source))
        self.assertEqual(len(matches), 1, "standing GEMM must contain exactly one FMA to split")
        declaration = re.search(r"\.reg \.f32 %f<(\d+)>;", source)
        self.assertIsNotNone(declaration)
        index = int(declaration.group(1))
        grown = source.replace(declaration.group(), f".reg .f32 %f<{index + 1}>;", 1)
        ind, destination, left, right, accumulator = matches[0].groups()
        streams = {}
        for variant, modifier, expected in (("fma", None, "VALIDATED"),
                                            ("rounded", ".rn", "VALIDATED"),
                                            ("contracted", "", "UNPROVED")):
            with self.subTest(variant=variant):
                text = source if modifier is None else operation.sub(
                    lambda _: f"{ind}mul{modifier}.f32 %f{index}, {left}, {right};\n"
                              f"{ind}add{modifier}.f32 {destination}, {accumulator}, %f{index};", grown)
                with self.case(f"gemm-{variant}-o1", expected, validator="loopval",
                               expected_detail="" if expected == "VALIDATED" else r"store 0 value: sat") as case:
                    case.assemble(text.encode())
                    streams[variant] = instruction_stream((case.directory / "kernel.sass").read_text())
                    case.validate()
        self.assertEqual(streams["fma"], streams["contracted"], "FMA repair must match contracted machine instructions")
        self.assertNotEqual(streams["fma"], streams["rounded"], "separate rounding must retain separate operations")

    def test_exact_pv_o1_validates_a_fresh_cubin(self):
        with self.case("exact-pv-o1", validator="loopval") as case:
            case.assemble((REPO / "tests" / "exact_pv.ptx").read_bytes())
            case.validate()  # Evidence.check_result requires a nonempty accepted proof.

    def test_shared_roundtrip_and_dropped_effects(self):
        source = (TVAL / "smut" / "smem_roundtrip.ptx").read_bytes()
        for mutation, expected, anchor in ((None, "VALIDATED", ""),
                                           ("barrier", "UNPROVED", r"barrier counts differ"),
                                           ("store", "UNPROVED", r"shared memory entering barrier 0: REFUTED \(sat\)")):
            with self.subTest(mutation=mutation):
                with self.case("shared-" + (mutation or "genuine"), expected, validator="smemval",
                               role="mutated" if mutation else "genuine", expected_detail=anchor) as case:
                    case.assemble(source)
                    if mutation:
                        text = (case.directory / "kernel.sass").read_text()
                        opcode = "BAR\\.SYNC" if mutation == "barrier" else "STS"
                        pattern = re.compile(r"(?m)^(\s*/\*[0-9a-fA-F]+\*/\s*)" + opcode + r"[^;]*;")
                        self.assertEqual(len(pattern.findall(text)), 1, "mutation anchor moved")
                        (case.directory / "mutated.sass").write_text(pattern.sub(r"\1NOP;", text))
                    case.validate(sass_name="mutated.sass" if mutation else "kernel.sass")

    def test_global_load_hoisting_across_possible_alias(self):
        source = (TVAL / "mem" / "lsls.ptx").read_bytes()
        for mutate in (False, True):
            with self.subTest(mutate=mutate):
                with self.case("global-readback-" + ("hoisted" if mutate else "genuine"),
                               "UNPROVED" if mutate else "VALIDATED", role="mutated" if mutate else "genuine",
                               expected_detail=r"store 1: sat" if mutate else "") as case:
                    case.assemble(source)
                    if mutate:
                        text = (case.directory / "kernel.sass").read_text()
                        instructions = list(INSTRUCTION.finditer(text))
                        pairs = [(first, second) for first, second in zip(instructions, instructions[1:])
                                 if first.group(2).startswith("STG.") and second.group(2).startswith("LDG.")]
                        self.assertEqual(len(pairs), 1, "read-back hoist anchor moved")
                        first, second = pairs[0]
                        text = (text[:first.start(2)] + second.group(2) + text[first.end(2):second.start(2)]
                                + first.group(2) + text[second.end(2):])
                        (case.directory / "mutated.sass").write_text(text)
                    case.validate(sass_name="mutated.sass" if mutate else "kernel.sass")

    def test_declared_sm80_target_assembles_but_is_not_licensed(self):
        source = (TVAL / "fma" / "rn.ptx").read_bytes().replace(b".target sm_89", b".target sm_80", 1)
        with self.case("float-rounded-sm80", "REFUSED", target="sm_80",
                       expected_detail="unsupported architecture") as case:
            case.assemble(source)
            case.validate()

    def test_assembler_rejects_bad_syntax_and_incompatible_target(self):
        source = (TVAL / "fma" / "rn.ptx").read_bytes()
        for name, text, anchor in (
            ("syntax", source.replace(b"mul.rn.f32", b"invalid.ptx.opcode", 1), r"(?i)(unknown|illegal|syntax|parse|modifier)"),
            ("target", source.replace(b".target sm_89", b".target sm_90", 1), r"(?i)(sm_90|sm_89|target)"),
        ):
            with self.subTest(name=name):
                with self.case("assembly-reject-" + name, "ASSEMBLY_REFUSED", role="assembly_refusal",
                               expected_detail=anchor) as case:
                    case.assemble(text, expect_error=True)


if __name__ == "__main__":
    verification_unittest.main(verbosity=2)
