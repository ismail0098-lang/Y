"""Guarded carry, exact integer instruction forms, and wide shift regressions."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from verification_unittest import main as verification_main

TVAL = Path(__file__).resolve().parents[1] / 'tools' / 'ptxas_tval'
sys.path.insert(0, str(TVAL))
try:
    import z3
except ImportError:
    HAS_Z3 = False
else:
    HAS_Z3 = True
    import ptxexec
    import sassexec


@unittest.skipUnless(HAS_Z3, 'integer executor checks require z3-solver')
class IntegerSemantics(unittest.TestCase):
    def state(self):
        return ptxexec.Ptx({})

    def prove(self, claim):
        solver = z3.Solver(); solver.set(timeout=3000); solver.add(z3.Not(claim))
        self.assertEqual(solver.check(), z3.unsat, solver.reason_unknown())

    def test_scalar_sources_preserve_width_and_register_reuse_grammar(self):
        state = sassexec.Sass({})
        state.R[0], state.UR[0] = z3.BitVecs('scalar_r scalar_ur', 32)
        self.prove(z3.And(state.rd('R0.reuse') == state.R[0],
                          state.rd('-R0.reuse') == -state.R[0],
                          state.rd('UR0.reuse') == state.UR[0]))
        for operand in ('R0.64', 'UR0.64', 'R.reuse0', 'R0.reuse.reuse',
                        '0x1.reuse', 'R0.64.reuse', 'R255', 'R256'):
            with self.subTest(operand=operand):
                with self.assertRaisesRegex(Exception, 'UNMODELLED SASS.*refusing'):
                    state.rd(operand)

    def test_register_destinations_require_regular_scalar_names_and_width(self):
        state = sassexec.Sass({}); value = z3.BitVecVal(7, 32)
        for destination in ('P0', 'X0', 'UR0', 'R0.64', 'R0.reuse', 'R255'):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(Exception, 'UNMODELLED SASS.*destination'):
                    state.wr(destination, value, z3.BoolVal(True))
        with self.assertRaisesRegex(Exception, 'destination width.*expected 32 bits'):
            state.wr('R0', z3.BitVecVal(7, 64), z3.BoolVal(True))
        self.assertFalse(state.R)
        state.wr('RZ', value, z3.Bool('zero_discard_guard'))
        self.assertFalse(state.R)
        self.assertFalse(state.defs)

    def test_pair_and_store_spans_refuse_boundary_crossings(self):
        state = sassexec.Sass({})
        state.R[253], state.R[254] = z3.BitVecs('span_lo span_hi', 32)
        self.prove(state.pair('R253.reuse') == z3.Concat(state.R[254], state.R[253]))
        self.prove(state.pair('RZ') == 0)
        self.prove(z3.And(*[value == 0 for value in state.store_words('RZ', 4)]))
        for source, count in (('R254', 2), ('R252', 4), ('UR0', 2),
                              ('R0.64', 2), ('R0.reuse.reuse', 2), ('R0', 3)):
            with self.subTest(source=source, count=count):
                with self.assertRaisesRegex(Exception, 'UNMODELLED SASS.*refusing'):
                    state.store_words(source, count)
        for source in ('R254', 'R255', 'UR0', 'R0.64'):
            with self.subTest(pair=source):
                with self.assertRaisesRegex(Exception, 'UNMODELLED SASS.*refusing'):
                    state.pair(source)

    def test_load_destination_spans_preserve_scalar_zero_discard(self):
        state = sassexec.Sass({})
        self.assertEqual(state.load_destinations('RZ', 1), ['RZ'])
        self.assertEqual(state.load_destinations('R250', 4), ['R250', 'R251', 'R252', 'R253'])
        self.assertEqual(state.load_destinations('R254', 1), ['R254'])
        for destination, count in (('RZ', 4), ('R252', 4), ('R254', 2),
                                   ('P4', 4), ('UR4', 1), ('R4.64', 1)):
            with self.subTest(destination=destination, count=count):
                with self.assertRaisesRegex(Exception, 'UNMODELLED SASS.*load destination'):
                    state.load_destinations(destination, count)

    def test_malformed_load_destinations_refuse_before_register_aliasing(self):
        for opcode in ('LDG.E', 'LDG.E.128', 'LDG.E.128.CONSTANT', 'LDS', 'LDS.128'):
            address = '[R2]' if opcode.startswith('LDS') else '[R2.64]'
            for destination in ('P4', 'UR4', 'R4.64', 'R255'):
                with self.subTest(opcode=opcode, destination=destination):
                    state = sassexec.Sass({})
                    with self.assertRaisesRegex(Exception, 'UNMODELLED SASS.*load destination'):
                        state.step(f'{opcode} {destination}, {address}')
                    self.assertNotIn(4, state.R)

    def test_scalar_zero_load_retains_read_and_cannot_supply_store_value(self):
        memory = z3.Array('zero_load_memory', z3.BitVecSort(64), z3.BitVecSort(32))
        original, changed = sassexec.Sass({'mem': memory}), sassexec.Sass({'mem': memory})
        for state in (original, changed):
            state.R[2], state.R[3] = z3.BitVecVal(16, 32), z3.BitVecVal(0, 32)
        original.step('LDG.E R4, [R2.64]'); changed.step('LDG.E RZ, [R2.64]')
        self.assertEqual(len(changed.loads), 1)
        self.assertNotIn(4, changed.R)
        original.step('STG.E [R2.64], R4'); changed.step('STG.E [R2.64], R4')
        solver = z3.Solver(); solver.set(timeout=3000)
        solver.add(original.stores[0][1] != changed.stores[0][1])
        self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())

    def test_pc_tagged_malformed_instruction_lines_refuse(self):
        with tempfile.TemporaryDirectory(prefix='y-sass-parser-regression-') as directory:
            source = Path(directory) / 'probe.sass'
            for malformed in ('/*0010*/ STG.E [R2.64], R0', '/*0010*/',
                              '/*00x0*/ MOV R0, 0x1;', '/*0010*/ MOV R0, 0x1; trailing'):
                with self.subTest(malformed=malformed):
                    source.write_text('/*0000*/ MOV R0, 0x1;\n' + malformed + '\n')
                    with self.assertRaisesRegex(Exception, 'UNMODELLED SASS instruction line.*semicolon'):
                        sassexec.run_sass(str(source), {})
            source.write_text('/* generated metadata */\n/*0000*/ MOV R0, 0x1;\n/*00A0*/ EXIT;\n')
            state = sassexec.run_sass(str(source), {})
            self.assertEqual(state.count, 2)
            self.prove(state.R[0] == 1)

    def test_comparison_requires_exact_modifiers_and_operand_count(self):
        for opcode in ('ISETP.LT.AND.U32', 'ISETP.LT.U32.U32.AND',
                       'ISETP.LT.U64.AND', 'ISETP.LT.U32.AND.EX',
                       'ISETP.LT.S32.AND', 'ISETP.LT'):
            with self.subTest(opcode=opcode):
                with self.assertRaisesRegex(Exception, 'UNMODELLED SASS OPCODE'):
                    sassexec.Sass({}).step(opcode + ' P0, PT, R0, R1, PT')
        for operands in ('P0, PT, R0', 'P0, PT, R0, R1',
                         'P0, PT, R0, R1, PT, P1'):
            with self.subTest(operands=operands):
                with self.assertRaisesRegex(Exception, 'UNMODELLED SASS OPERAND COUNT'):
                    sassexec.Sass({}).step('ISETP.LT.U32.AND ' + operands)
        with self.assertRaisesRegex(Exception, 'ISETP writes a second predicate.*refusing'):
            sassexec.Sass({}).step('ISETP.EQ.U32.AND P0, P1, R0, R1, PT')

    def test_carry_forms_require_exact_operand_count(self):
        for instruction in ('IMAD.X R0, R1, R2, R3',
                            'IMAD.X R0, R1, R2, R3, P0, !PT',
                            'IMAD.HI.U32 R0, R1, R2',
                            'IMAD.HI.U32 R0, P0, R1, R2, R3, P1',
                            'IADD3 R0, R1, R2',
                            'IADD3 R0, P0, R1, R2, R3, P1',
                            'IADD3.X R0, R1, R2, R3, P0',
                            'IADD3.X R0, P0, R1, R2, R3, P1, !PT, P2'):
            with self.subTest(instruction=instruction):
                with self.assertRaisesRegex(Exception, 'UNMODELLED SASS OPERAND COUNT'):
                    sassexec.Sass({}).step(instruction)

    def test_compare_predicate_alias_preserves_input_and_guard(self):
        old, guard = z3.Bools('compare_old compare_guard')
        # These pairs distinguish unsigned -1 from signed -1 at the zero
        # boundary. The predicate destination also supplies the combine input.
        for opcode, expected in (
                ('ISETP.LT.AND', old),
                ('ISETP.LT.U32.AND', z3.BoolVal(False)),
                ('ISETP.GE.OR', old),
                ('ISETP.GE.U32.XOR', z3.Not(old))):
            with self.subTest(opcode=opcode):
                state = sassexec.Sass({})
                state.R[0], state.R[1] = z3.BitVecVal(0xffffffff, 32), z3.BitVecVal(0, 32)
                state.P[0], state.P[1] = old, guard
                state.step('@P1 ' + opcode + ' P0, PT, R0, R1, P0')
                self.prove(state.P[0] == z3.If(guard, expected, old))

    def test_iadd3x_reads_old_source_and_carry_under_aliases(self):
        x, y = z3.BitVecs('iadd_alias_x iadd_alias_y', 32)
        old, guard = z3.Bools('iadd_alias_old iadd_alias_guard')
        state = sassexec.Sass({})
        state.R[0], state.R[1] = x, y
        state.P[0], state.P[1] = old, guard
        total = z3.ZeroExt(1, x) + z3.ZeroExt(1, y) + z3.If(
            old, z3.BitVecVal(1, 33), z3.BitVecVal(0, 33))
        state.step('@P1 IADD3.X R0, P0, R0, R1, RZ, P0, !PT')
        self.prove(z3.And(
            state.R[0] == z3.If(guard, z3.Extract(31, 0, total), x),
            state.P[0] == z3.If(guard, z3.Extract(32, 32, total) == 1, old)))

    def test_imadx_reads_old_inputs_under_destination_alias(self):
        x, y = z3.BitVecs('imad_alias_x imad_alias_y', 32)
        carry, guard = z3.Bools('imad_alias_carry imad_alias_guard')
        state = sassexec.Sass({})
        state.R[0], state.R[1] = x, y
        state.P[0], state.P[1] = carry, guard
        expected = x * y + x + z3.If(carry, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32))
        state.step('@P1 IMAD.X R0, R0, R1, R0, P0')
        self.prove(z3.And(state.R[0] == z3.If(guard, expected, x), state.P[0] == carry))

    def test_changed_carry_polarity_and_comparison_signedness_are_sat(self):
        for original, changed in (
                ('IMAD.X R0, RZ, RZ, -0x1, P0', 'IMAD.X R0, RZ, RZ, -0x1, !P0'),
                ('ISETP.LT.AND P0, PT, R0, R1, PT',
                 'ISETP.LT.U32.AND P0, PT, R0, R1, PT')):
            left, right = sassexec.Sass({}), sassexec.Sass({})
            for state in (left, right):
                state.R[0], state.R[1] = z3.BitVecVal(0xffffffff, 32), z3.BitVecVal(0, 32)
                state.P[0] = z3.Bool('mutated_carry')
            left.step(original); right.step(changed)
            difference = (left.R[0] != right.R[0] if original.startswith('IMAD')
                          else left.P[0] != right.P[0])
            solver = z3.Solver(); solver.set(timeout=3000); solver.add(difference)
            self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())

    def test_first_carry_read_refuses_in_each_family(self):
        for instruction in ('addc.u32 %r0, 1, 2', 'subc.u32 %r0, 1, 2',
                            'madc.lo.cc.u32 %r0, 1, 2, 3',
                            'madc.hi.cc.u32 %r0, 1, 2, 3'):
            with self.subTest(instruction=instruction):
                with self.assertRaisesRegex(Exception, 'UNMODELLED PTX carry read.*uninitialized'):
                    self.state().step(instruction)

    def test_defined_add_and_subtract_chains_keep_values(self):
        for first, second, low, high in (
                ('add.cc.u32 %r0, 0xffffffff, 1', 'addc.u32 %r1, 0, 0', 0, 1),
                ('sub.cc.u32 %r0, 0, 1', 'subc.u32 %r1, 0, 0', 0xffffffff, 0xffffffff)):
            with self.subTest(first=first):
                state = self.state(); state.step(first); state.step(second)
                self.prove(z3.And(state.r[0] == low, state.r[1] == high))

    def test_predicated_writer_initializes_only_its_paths(self):
        guard = z3.Bool('carry_guard')
        state = self.state(); state.p[0] = guard
        state.step('@%p0 add.cc.u32 %r0, 0xffffffff, 1')
        state.step('@%p0 addc.u32 %r1, 0, 0')
        self.prove(z3.Implies(guard, state.r[1] == 1))
        with self.assertRaisesRegex(Exception, 'CC may be uninitialized'):
            state.step('@!%p0 addc.u32 %r2, 0, 0')

    def test_narrower_guards_require_and_pass_implication_proof(self):
        a, b = z3.Bools('carry_a carry_b')
        state = self.state()
        state.write_cc(z3.BoolVal(True), z3.Or(a, b))
        self.prove(z3.Implies(z3.And(a, b), state.read_cc(z3.And(a, b), 'addc.u32')))
        with self.assertRaisesRegex(Exception, 'CC may be uninitialized'):
            state.read_cc(z3.BoolVal(True), 'addc.u32')

    def test_unknown_carry_dominance_is_a_refusal(self):
        state = self.state(); state.cc_defined = z3.Bool('carry_unknown_guard')
        solver = mock.Mock(); solver.check.return_value = z3.unknown
        with mock.patch.object(ptxexec, 'Solver', return_value=solver):
            with self.assertRaisesRegex(Exception, 'CC may be uninitialized'):
                state.read_cc(z3.BoolVal(True), 'addc.u32')

    def test_non_cc_instruction_does_not_initialize_carry(self):
        state = self.state(); state.step('add.u32 %r0, 1, 2')
        with self.assertRaisesRegex(Exception, 'CC may be uninitialized'):
            state.step('addc.u32 %r1, 3, 4')

    def test_predicate_operand_register_files_are_distinct(self):
        ptx = self.state(); sass = sassexec.Sass({})
        for operand in ('%r0', '%rd0', 'P0', '%p0.extra'):
            with self.subTest(side='ptx', operand=operand):
                with self.assertRaisesRegex(Exception, 'UNMODELLED PTX predicate operand'):
                    ptx.P(operand)
        for operand in ('R0', 'UR0', 'UP0', 'P0.extra'):
            with self.subTest(side='sass', operand=operand):
                with self.assertRaisesRegex(Exception, 'UNMODELLED SASS predicate operand'):
                    sass.pr(operand)
                with self.assertRaisesRegex(Exception, 'UNMODELLED SASS predicate destination'):
                    sass.wp(operand, z3.BoolVal(True), z3.BoolVal(True))
        self.prove(sass.pr('PT'))
        self.prove(z3.Not(sass.pr('!PT')))

    def test_wide_shift_amounts_accept_registers_and_hex_literals(self):
        for opcode in ('shl.b64', 'shr.u64'):
            for amount in (0, 31, 32, 63, 64, 0xffffffff):
                with self.subTest(opcode=opcode, amount=amount):
                    state = self.state()
                    state.rd[0] = z3.BitVecVal(0xfedcba9876543210, 64)
                    state.r[0] = z3.BitVecVal(amount, 32)
                    state.step(f'{opcode} %rd1, %rd0, %r0')
                    state.step(f'{opcode} %rd2, %rd0, 0x{amount:x}')
                    expected = ((0xfedcba9876543210 << amount) & ((1 << 64) - 1)
                                if opcode == 'shl.b64' and amount < 64 else
                                (0xfedcba9876543210 >> amount if amount < 64 else 0))
                    self.prove(z3.And(state.rd[1] == expected, state.rd[2] == expected))


if __name__ == '__main__':
    verification_main()
