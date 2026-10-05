"""Real ptxas controls and mutations for uniform reads and barrier effects.

The cubins are assembled and disassembled locally; no GPU is required. SASS
mutations test the validator's refusal and refutation paths, without claiming
that the text mutation is a separately assembled machine artifact.
"""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from verification_unittest import main as verification_main


ROOT = Path(__file__).resolve().parents[1]
TVAL = ROOT / 'tools' / 'ptxas_tval'
sys.path.insert(0, str(TVAL))

try:
    import z3
except ImportError:
    HAS_Z3 = False
else:
    HAS_Z3 = True
    import batch
    import fpmode
    import smemval
    import tval


HEADER = '''.version 7.8
.target sm_89
.address_size 64
.visible .entry probe(.param .u64 O, .param .u32 N)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd0, [O];
    ld.param.u32 %r0, [N];
'''


def ptx(body):
    return HEADER + body + '\n    ret;\n}\n'


@unittest.skipUnless(HAS_Z3, 'float abstraction checks require z3-solver')
class FloatCommutativity(unittest.TestCase):
    def setUp(self):
        self.float_op = fpmode.factory()
        self.x, self.y, self.u, self.v = z3.BitVecs('fp_alias_x fp_alias_y fp_alias_u fp_alias_v', 32)

    def check(self, claim, expected, *assumptions):
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(*assumptions, z3.Not(claim))
        self.assertEqual(solver.check(), expected, solver.reason_unknown())

    def test_licensed_swaps_survive_aliases_and_substitution(self):
        # x + 0 has the same bit pattern as x but a different AST allocation
        # ID. Allocation order must not determine semantic congruence.
        for name in ('FADD', 'FMAX'):
            with self.subTest(op=name):
                original = self.float_op(name, self.x, self.y, side='ptx')
                alias = self.float_op(name, self.x + 0, self.y, side='sass')
                self.check(original == alias, z3.unsat)
                swapped = self.float_op(name, self.u, self.v, side='sass')
                substituted = z3.substitute(swapped, (self.u, self.y), (self.v, self.x))
                self.check(original == substituted, z3.unsat)
                semantic_alias = self.float_op(name, self.v, self.u, side='sass')
                self.check(original == semantic_alias, z3.unsat,
                           self.x == self.u, self.y == self.v)

    def test_value_ordering_preserves_distinctions(self):
        for name in ('FADD', 'FMAX'):
            with self.subTest(op=name):
                original = self.float_op(name, self.x, self.y, side='ptx')
                changed = self.float_op(name, self.u, self.y, side='sass')
                self.check(original == changed, z3.sat)
        for name in ('FMUL', 'FMIN'):
            with self.subTest(op=name):
                self.check(self.float_op(name, self.x, self.y, side='ptx') ==
                           self.float_op(name, self.y, self.x, side='sass'), z3.sat)
        split = self.float_op('FADD', self.float_op('FMUL', self.x, self.y, side='ptx'),
                              self.u, side='ptx')
        fused = self.float_op('FFMA', self.x, self.y, self.u, side='sass')
        self.check(split == fused, z3.sat)

    def test_disabled_commutativity_licenses_refuse(self):
        for name in ('FADD', 'FMAX'):
            flag = f'{name}_IS_COMMUTATIVE'
            previous = fpmode.IDENTIFICATIONS[flag]
            try:
                fpmode.IDENTIFICATIONS[flag] = False
                with self.subTest(op=name), self.assertRaisesRegex(Exception, 'unvalidated'):
                    self.float_op(name, self.x, self.y, side='sass')
            finally:
                fpmode.IDENTIFICATIONS[flag] = previous


@unittest.skipUnless(HAS_Z3, 'shared address checks require z3-solver')
class SharedAddressOperands(unittest.TestCase):
    def test_offsets_preserve_register_width_and_shared_window_wrap(self):
        import ptxexec
        import smem
        state = ptxexec.Ptx({})
        state.smem_layout = {'slot': 0}
        state.r[0] = z3.BitVecVal(0xfffffffc, 32)
        state.rd[0] = z3.BitVecVal(0xfffffffffffffffc, 64)
        for operand in ('%r0+4', '%rd0+0x4', 'slot+4', 'slot+0x4', 'slot + -4'):
            with self.subTest(operand=operand):
                result = z3.simplify(smem.word(state.SA(operand)))
                expected = 0x3fffffff if operand.endswith('-4') else (1 if operand.startswith('slot') else 0)
                self.assertEqual(result.as_long(), expected)

    def test_unmodeled_address_expressions_refuse_by_name(self):
        import ptxexec
        state = ptxexec.Ptx({})
        state.smem_layout = {'slot': 0}
        for operand in ('slot+%r0', 'missing+4', '%rd0+4+4'):
            with self.subTest(operand=operand), self.assertRaisesRegex(
                    Exception, 'UNMODELLED PTX shared address.*refusing, not guessing'):
                state.SA(operand)


@unittest.skipUnless(HAS_Z3, 'operand checks require z3-solver')
class GlobalAddressAndSassOperands(unittest.TestCase):
    def test_global_offsets_and_hex_literals_preserve_64_bit_wrap(self):
        import ptxexec
        state = ptxexec.Ptx({})
        state.rd[0] = z3.BitVecVal(0xfffffffffffffffc, 64)
        expected = {'%rd0+4': 0, '%rd0+-4': 0xfffffffffffffff8,
                    '%rd0 + -0x4': 0xfffffffffffffff8,
                    '0x100000000+4': 0x100000004, '4294967296': 0x100000000,
                    '-0x4': 0xfffffffffffffffc}
        for operand, value in expected.items():
            with self.subTest(operand=operand):
                self.assertEqual(z3.simplify(state.GA(operand)).as_long(), value)

    def test_global_reader_refuses_unknown_expressions_and_address_widths(self):
        import ptxexec
        state = ptxexec.Ptx({})
        for operand in ('%r0', '%r0+4', '%rd0+%r0', '%rd0+4+4', '%rd0-4', 'slot'):
            with self.subTest(operand=operand), self.assertRaisesRegex(
                    Exception, 'UNMODELLED PTX global address.*refusing, not guessing'):
                state.GA(operand)

    def test_trailing_operands_are_not_silently_ignored(self):
        import sassexec
        bodies = ('MOV R0, RZ, P1', 'IMAD R0, RZ, RZ, RZ, P1',
                  'IMAD.WIDE.U32 R0, RZ, RZ, RZ, P1',
                  'SHF.R.U32.HI R0, RZ, 0x0, RZ, P1',
                  'USHF.L.U32 UR0, URZ, 0x0, URZ, P1',
                  'LEA.HI.X R0, RZ, RZ, RZ, 0x1, PT, P1',
                  'STG.E [R0.64], RZ, P1', 'LDG.E R0, [R2.64], P1',
                  'STS.128 [RZ], RZ, P1', 'EXIT PT', 'NOP R0')
        for body in bodies:
            with self.subTest(body=body), self.assertRaisesRegex(
                    Exception, 'UNMODELLED SASS OPERAND COUNT.*refusing, not guessing'):
                sassexec.Sass({}).step(body, 0)


class RealToolchain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [name for name in ('ptxas', 'nvdisasm') if not shutil.which(name)]
        if not HAS_Z3:
            missing.append('z3-solver')
        if missing:
            raise unittest.SkipTest('PTXAS validation requires ' + ', '.join(missing))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='y-ptxas-regression-')
        self.directory = Path(temporary.name)
        self.addCleanup(temporary.cleanup)

    def assemble(self, source, name='probe'):
        p = self.directory / f'{name}.ptx'
        c = self.directory / f'{name}.cubin'
        s = self.directory / f'{name}.sass'
        p.write_text(source)
        assembled = subprocess.run(
            ['ptxas', '-O1', '-arch=sm_89', str(p), '-o', str(c)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(assembled.returncode, 0, assembled.stdout + assembled.stderr)
        disassembled = subprocess.run(
            ['nvdisasm', '-c', str(c)], capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(disassembled.returncode, 0, disassembled.stderr)
        s.write_text(disassembled.stdout)
        return p, s, disassembled.stdout

    def mutation(self, text, pattern, replacement, name):
        mutated, count = re.subn(pattern, replacement, text)
        self.assertEqual(count, 1, 'the expected real SASS instruction was not unique')
        self.assertNotEqual(mutated, text, 'mutation did not change the disassembly')
        path = self.directory / f'{name}.sass'
        path.write_text(mutated)
        return path

    def validate(self, source, machine):
        return tval.run(str(source), str(machine), NS=2, B1=1, B2=2,
                        log=lambda _: None)

    def replace_instruction(self, text, pattern, replacement, name):
        """Replace a live instruction, updating addresses of its trailing trap."""
        instructions = re.compile(r'\s*/\*[0-9a-f]+\*/\s*(.*?);\s*$')
        lines, count, address = [], 0, 0
        for line in text.splitlines():
            instruction = instructions.fullmatch(line)
            if instruction is None:
                lines.append(line)
                continue
            original = instruction.group(1).strip()
            match = re.fullmatch(pattern, original)
            body = replacement(match) if match else [original]
            count += bool(match)
            for item in body:
                lines.append(f'        /*{address:04x}*/ {item} ;')
                address += 16
        self.assertEqual(count, 1, 'expected one live instruction to replace')
        path = self.directory / f'{name}.sass'
        path.write_text('\n'.join(lines) + '\n')
        return path

    def test_disabled_uniform_parameter_load_cannot_validate(self):
        source = (TVAL / 'smut' / 'smem_roundtrip.ptx').read_text()
        p, s, text = self.assemble(source, 'roundtrip')
        correct = smemval.validate(str(p), str(s), 2, 'direct')
        self.assertEqual(correct[0], 'VALIDATED', correct)
        # The uniform N load determines the output guard. Disabling it leaves
        # an unconstrained old register, so identical output guards cannot be
        # proved. Previously this mutation still VALIDATED with 18 obligations.
        mutant = self.mutation(
            text, r'(ULDC\s+UR\d+,\s*c\[0x0\]\[0x168\])',
            r'@!PT \1', 'disabled_uniform_load',
        )
        wrong = smemval.validate(str(p), str(mutant), 2, 'direct')
        self.assertEqual(wrong[0], 'UNPROVED', wrong)
        self.assertEqual(batch.validate(str(p), str(mutant), 2, 'direct')[0], 'UNPROVED')

    def test_unknown_64_bit_constant_bank_reads_are_refused(self):
        p, s, text = self.assemble(ptx('''    mov.u32 %r1, 0;
    st.global.u32 [%rd0], %r1;'''))
        correct = self.validate(p, s)
        self.assertEqual(correct[0], 'VALIDATED', correct)
        for slot, missing in ((0x200, 0x200), (0x168, 0x16c)):
            with self.subTest(slot=hex(slot)):
                # N occupies one word at 0x168; its neighbor is not an ABI
                # symbol. ULDC.64 previously invented zero for either half.
                mutant = self.mutation(
                    text, r'(ULDC\.64\s+UR\d+,\s*c\[0x0\])\[0x118\]',
                    rf'\1[0x{slot:x}]', f'unknown_bank_{slot:x}',
                )
                result = self.validate(p, mutant)
                self.assertEqual(result[0], 'REFUSED', result)
                self.assertIn(f'const bank slot 0x{missing:x}', result[1])

    def test_wide_multiply_signedness_is_preserved(self):
        for ty, opcode, mutant_opcode in (
                ('u32', 'IMAD.WIDE.U32', 'IMAD.WIDE'),
                ('s32', 'IMAD.WIDE', 'IMAD.WIDE.U32')):
            with self.subTest(type=ty):
                p, s, text = self.assemble(ptx(f'''    mul.wide.{ty} %rd1, %r0, 2;
    st.global.u64 [%rd0], %rd1;'''), 'wide_' + ty)
                correct = self.validate(p, s)
                self.assertEqual(correct[0], 'VALIDATED', correct)
                mutant = self.mutation(
                    text, re.escape(opcode) + r'(?=\s)', mutant_opcode,
                    'opposite_wide_' + ty,
                )
                log = []
                wrong = tval.run(str(p), str(mutant), NS=2, B1=1, B2=2, log=log.append)
                self.assertEqual(wrong[0], 'UNPROVED', wrong)
                self.assertRegex('\n'.join(log), r'(?m)^  store 1: sat$')
                # At N=-1, multiplying by 2 gives high word 1 when unsigned
                # and 0xffffffff when signed. The mutation changes stored bits.

    def test_zero_register_store_sources_validate_real_output(self):
        p, s, text = self.assemble(ptx('''    mov.u64 %rd1, 0;
    st.global.u64 [%rd0], %rd1;'''), 'zero64')
        self.assertRegex(text, r'STG\.E\.64\s+\[R\d+\.64\],\s*RZ')
        self.assertEqual(self.validate(p, s)[0], 'VALIDATED')
        pointer_bits = self.mutation(
            text, r'(STG\.E\.64\s+\[(R\d+)\.64\],\s*)RZ', r'\1\2',
            'pointer_instead_of_zero',
        )
        log = []
        wrong = tval.run(str(p), str(pointer_bits), NS=2, B1=1, B2=2, log=log.append)
        self.assertEqual(wrong[0], 'UNPROVED', wrong)
        self.assertRegex('\n'.join(log), r'(?m)^  store [01]: sat$')

        source = ptx('''    .shared .align 4 .b8 slot[4];
    st.shared.u32 [slot], 0;
    bar.sync 0;
    ld.shared.u32 %r1, [slot];
    st.global.u32 [%rd0], %r1;''')
        p, s, text = self.assemble(source, 'shared_zero')
        self.assertRegex(text, r'STS\s+\[RZ\],\s*RZ')
        self.assertEqual(self.validate(p, s)[0], 'VALIDATED')
        one = self.replace_instruction(
            text, r'STS\s+\[RZ\],\s*RZ',
            lambda _: ['MOV R62, 0x1', 'STS [RZ], R62'], 'one_instead_of_zero',
        )
        wrong = self.validate(p, one)
        self.assertEqual(wrong[0], 'UNPROVED', wrong)
        self.assertIn('REFUTED (sat)', wrong[1])

    def test_disabled_uniform_writers_preserve_the_previous_output_value(self):
        p, s, text = self.assemble(ptx('    st.global.u32 [%rd0], %r0;'))
        self.assertEqual(self.validate(p, s)[0], 'VALIDATED')
        writes = ('UMOV UR62, 0x9',
                  'ULDC UR62, c[0x0][0x118]',
                  'USHF.L.U32 UR62, URZ, 0x0, URZ',
                  'ULDC.64 UR62, c[0x0][0x118]')
        pattern = r'(?:MOV|IMAD\.MOV\.U32) (R\d+),\s*(?:RZ,\s*RZ,\s*)?c\[0x0\]\[0x168\]'
        for index, write in enumerate(writes):
            with self.subTest(write=write):
                def replacement(match, guard='@!PT '):
                    return ['UMOV UR62, c[0x0][0x168]', guard + write,
                            f'MOV {match.group(1)}, UR62']

                # A disabled write must keep N in the uniform register, so
                # the reference parameter kernel still matches its stores.
                preserved = self.replace_instruction(text, pattern, replacement,
                                                       f'preserved_uniform_{index}')
                correct = self.validate(p, preserved)
                self.assertEqual(correct[0], 'VALIDATED', correct)
                # Enabling the same write changes that value. This companion
                # prevents simply ignoring all uniform writes from passing.
                changed = self.replace_instruction(
                    text, pattern, lambda match: replacement(match, ''),
                    f'changed_uniform_{index}',
                )
                wrong = self.validate(p, changed)
                self.assertEqual(wrong[0], 'UNPROVED', wrong)

    def test_counted_barrier_cannot_be_treated_as_a_full_block_barrier(self):
        source = ptx('''    bar.sync 0;
    st.global.u32 [%rd0], %r0;''')
        p, s, text = self.assemble(source, 'full_barrier')
        correct = self.validate(p, s)
        self.assertEqual(correct[0], 'VALIDATED', correct)
        counted = self.mutation(
            text, r'(BAR\.SYNC\.DEFER_BLOCKING\s+0x0)\s*;',
            r'\1, 0x20 ;', 'counted_sass_barrier',
        )
        result = self.validate(p, counted)
        self.assertEqual(result[0], 'REFUSED', result)
        self.assertIn('full-block barriers', result[1])

        # This is valid PTX and real ptxas emits the explicit 32-thread count.
        # Both that output and an output dropping the count must be refused
        # until the validator models subgroup arrival/release semantics.
        p, s, text = self.assemble(source.replace('bar.sync 0;', 'bar.sync 0, 32;'),
                                   'counted_barrier')
        self.assertRegex(text, r'BAR\.SYNC\.DEFER_BLOCKING\s+0x0,\s*0x20')
        dropped = self.mutation(
            text, r'(BAR\.SYNC\.DEFER_BLOCKING\s+0x0),\s*0x20',
            r'\1', 'dropped_barrier_count',
        )
        for machine in (s, dropped):
            result = self.validate(p, machine)
            self.assertEqual(result[0], 'REFUSED', result)
            self.assertIn('PTX BARRIER', result[1])

    def test_uniform_predicate_reports_a_named_refusal(self):
        p, _, text = self.assemble((TVAL / 'smut' / 'smem_roundtrip.ptx').read_text(),
                                   'uniform_predicate')
        mutant = self.mutation(
            text, r'(ULDC\s+UR\d+,\s*c\[0x0\]\[0x168\])',
            r'@UP0 \1', 'unsupported_uniform_predicate',
        )
        result = self.validate(p, mutant)
        self.assertEqual(result[0], 'REFUSED', result)
        self.assertIn('uniform predicate', result[1])

    def test_address_uncertainty_never_reports_sat(self):
        source = ptx('''    ld.global.u32 %r1, [%rd0+4];
    add.u64 %rd1, %rd0, 8;
    st.global.u32 [%rd1], %r1;''')
        p, s, _ = self.assemble(source, 'load_offset_unknown')
        self.assertEqual(self.validate(p, s)[0], 'VALIDATED')
        solver = mock.Mock()
        solver.check.return_value = z3.unknown
        with mock.patch.object(tval, 'Solver', return_value=solver):
            result = self.validate(p, s)
        self.assertEqual(result[0], 'UNPROVED', result)
        self.assertIn('address', result[1])
        self.assertNotIn('sat', result[1])


if __name__ == '__main__':
    verification_main(verbosity=2)
