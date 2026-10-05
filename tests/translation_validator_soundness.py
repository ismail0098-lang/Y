"""Counterexamples and controls for the PTX translation-validation boundary."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from verification_unittest import main as verification_main

REPO = Path(__file__).resolve().parent.parent
TVAL = REPO / 'tools' / 'ptxas_tval'
sys.path.insert(0, str(TVAL))
import loopcfg
import params
import ptxsource

try:
    import z3
except ImportError:
    HAS_Z3 = False
else:
    HAS_Z3 = True
    import batch
    import loopval
    import smem
    import smemval
    import tval


HEADER = '''.version 7.8
.target sm_89
.address_size 64
.visible .entry probe(.param .u64 P0, .param .u64 P1)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [P0];
    ld.param.u64 %rd1, [P1];
'''


def ptx(body):
    return HEADER + body + '\n    ret;\n}\n'


def sass(body):
    instructions = ['MOV R2, c[0x0][0x160]', 'MOV R3, c[0x0][0x164]',
                    'MOV R4, c[0x0][0x168]', 'MOV R5, c[0x0][0x16c]']
    instructions += body + ['EXIT']
    return '.target sm_89\n.text.probe:\n' + ''.join(
        f'/*{i * 16:04x}*/ {instruction} ;\n'
        for i, instruction in enumerate(instructions))


class TemporarySources(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='y-tval-soundness-')
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def write(self, name, text):
        path = self.directory / name
        path.write_text(text)
        return str(path)


class Parsing(TemporarySources):
    def test_comments_do_not_change_loop_regions_or_parameter_layout(self):
        original = TVAL / 'mem' / 'loop_swap.ptx'
        commented = ''.join(line.rstrip('\n') + ' // instruction or directive\n'
                            for line in original.read_text().splitlines(True))
        path = self.write('commented.ptx', commented)
        self.assertEqual(loopcfg.ptx_regions(original), loopcfg.ptx_regions(path))
        self.assertEqual(params.parse(original), params.parse(path))

    def test_block_comments_do_not_invent_entries_or_parameters(self):
        source = '/*\n.visible .entry hidden(.param .u64 fake)\n*/\n' + ptx('')
        path = self.write('comments.ptx', source)
        self.assertEqual(loopcfg.ptx_entry_points(path), ['probe'])
        self.assertEqual([name for name, _ in params.parse(path)[0]], ['P0', 'P1'])

    def test_quoted_directive_strings_keep_comment_markers(self):
        source = '.file 1 "path//literal/*name*/"\n// discarded\n'
        path = self.write('quoted.ptx', source)
        self.assertIn('"path//literal/*name*/"', ptxsource.read(path))
        self.assertNotIn('discarded', ptxsource.read(path))

    def test_unterminated_comment_is_refused(self):
        path = self.write('unterminated.ptx', ptx('') + '/* missing end')
        with self.assertRaisesRegex(Exception, 'unterminated block comment'):
            loopcfg.ptx_entry_points(path)

    def test_unparsed_instruction_is_refused_by_loop_scanner(self):
        path = self.write('unparsed.ptx', ptx('    trap'))
        with self.assertRaisesRegex(Exception, 'UNMODELLED PTX source line'):
            loopcfg.ptx_back_edges(path)

    def test_parameter_array_cannot_be_parsed_as_one_scalar(self):
        for declaration in ('.param .b8 ignored[16]',
                            '.param .align 8 .b8 ignored[16]',
                            '.param .u8 ignored'):
            with self.subTest(declaration=declaration):
                source = ptx('').replace('.entry probe(',
                                          f'.entry probe({declaration}, ')
                path = self.write('array.ptx', source)
                with self.assertRaisesRegex(Exception, 'UNMODELLED PTX PARAMETER'):
                    params.parse(path)


@unittest.skipUnless(HAS_Z3, 'translation-validator soundness tests require z3-solver')
class Validation(TemporarySources):
    def validate(self, source, machine):
        p = self.write('probe.ptx', source)
        s = self.write('probe.sass', machine)
        return tval.run(p, s, NS=2, B1=1, B2=2, log=lambda _: None)[0]

    def test_later_store_is_checked_when_first_predicate_is_false(self):
        source = ptx('''    mov.u32 %r0, %tid.x;
    mov.u32 %r1, 7;
    setp.eq.u32 %p0, %r0, 0;
    @%p0 st.global.u32 [%rd0], %r1;
    st.global.u32 [%rd1], %r0;''')
        machine = ['S2R R0, SR_TID.X', 'MOV R6, 0x7',
                   'ISETP.EQ.U32.AND P0, PT, R0, RZ, PT',
                   '@P0 STG.E [R2.64], R6', 'STG.E [R4.64], R0']
        self.assertEqual(self.validate(source, sass(machine)), 'VALIDATED')
        machine[-1] = 'STG.E [R4.64], RZ'
        self.assertEqual(self.validate(source, sass(machine)), 'UNPROVED')

    def test_complementary_stores_use_their_own_predicates(self):
        source = ptx('''    mov.u32 %r0, %tid.x;
    mov.u32 %r1, 7;
    setp.eq.u32 %p0, %r0, 0;
    @%p0 st.global.u32 [%rd0], %r1;
    @!%p0 mov.u32 %r2, %r0;
    @!%p0 st.global.u32 [%rd1], %r2;''')
        machine = ['S2R R0, SR_TID.X', 'MOV R6, 0x7',
                   'ISETP.EQ.U32.AND P0, PT, R0, RZ, PT',
                   '@P0 STG.E [R2.64], R6', '@!P0 STG.E [R4.64], R0']
        self.assertEqual(self.validate(source, sass(machine)), 'VALIDATED')
        machine[-1] = '@!P0 STG.E [R4.64], RZ'
        self.assertEqual(self.validate(source, sass(machine)), 'UNPROVED')

    def test_inline_commented_store_cannot_be_dropped(self):
        for comment in ('// observable store', '/* observable store */'):
            with self.subTest(comment=comment):
                source = ptx('    mov.u32 %r0, 7;\n'
                             f'    st.global.u32 [%rd0], %r0; {comment}\n'
                             '    st.global.u32 [%rd1], %r0;')
                correct = sass(['MOV R6, 0x7', 'STG.E [R2.64], R6',
                                'STG.E [R4.64], R6'])
                wrong = sass(['MOV R6, 0x7', 'STG.E [R4.64], R6'])
                self.assertEqual(self.validate(source, correct), 'VALIDATED')
                self.assertEqual(self.validate(source, wrong), 'UNPROVED')
                self.assertEqual(batch.validate(self.write('batch.ptx', source),
                                                self.write('batch.sass', wrong),
                                                2, 'direct')[0], 'UNPROVED')

    def test_commented_loop_still_validates(self):
        original = TVAL / 'mem' / 'loop_swap.ptx'
        source = ''.join(line.rstrip('\n') + ' // retained\n'
                         for line in original.read_text().splitlines(True))
        result = loopval.validate(self.write('loop.ptx', source),
                                  str(TVAL / 'mem' / 'loop_swap.sass'),
                                  budget=2, verbose=False)
        self.assertEqual(result[0], 'VALIDATED', result)

    def test_float_predicate_in_bit_registers_is_refused(self):
        source = ptx('''    mov.u32 %r0, 0xbf800000;
    mov.u32 %r1, 0x3f800000;
    setp.lt.f32 %p0, %r0, %r1;
    selp.u32 %r2, 7, 9, %p0;
    st.global.u32 [%rd0], %r2;''')
        wrong = sass(['MOV R6, 0x9', 'STG.E [R2.64], R6'])
        self.assertEqual(self.validate(source, wrong), 'REFUSED')
        with self.assertRaisesRegex(Exception, 'UNMODELLED PTX COMPARISON'):
            batch.validate(self.write('float.ptx', source),
                           self.write('float.sass', wrong), 2, 'direct')
        # The identical register bit patterns have supported signed integer
        # semantics; rejecting every setp instruction is not an acceptable fix.
        integer = source.replace('setp.lt.f32', 'setp.lt.s32')
        correct = sass(['MOV R6, 0x7', 'STG.E [R2.64], R6'])
        self.assertEqual(self.validate(integer, correct), 'VALIDATED')
        if shutil.which('ptxas'):
            subprocess.run(['ptxas', '-O1', '-arch=sm_89',
                            self.write('valid_float.ptx', source), '-o',
                            str(self.directory / 'valid_float.cubin')],
                           check=True, capture_output=True)

    def test_unmodeled_predicate_modifiers_are_refused(self):
        source = ptx('''    mov.u32 %r0, 0;
    setp.eq.u32 %p1, %r0, 0;
    setp.eq.and.u32 %p0, %r0, 0, %p1;
    selp.u32 %r2, 7, 9, %p0;
    st.global.u32 [%rd0], %r2;''')
        self.assertEqual(self.validate(source, sass(['MOV R6, 0x9',
                                                    'STG.E [R2.64], R6'])),
                         'REFUSED')

    def test_commented_shared_declarations_are_ignored(self):
        source = '/*\n.shared .align 4 .b32 hidden[8];\n*/\n' + ptx('')
        self.assertEqual(smem.layout(self.write('shared.ptx', source)), {})

    def test_shared_float_load_updates_float_registers(self):
        for vector in (False, True):
            with self.subTest(vector=vector):
                body = '''    .shared .align 16 .b32 buf[4];
    .reg .f32 %f<8>;
    mov.u32 %r0, buf;
'''
                body += ''.join(f'    mov.f32 %f{i}, 0f3f800000;\n'
                                for i in range(4))
                body += ''.join(f'    mov.f32 %f{i}, 0f40000000;\n'
                                for i in range(4, 8))
                if vector:
                    body += '''    st.shared.v4.f32 [%r0], {%f4, %f5, %f6, %f7};
    ld.shared.v4.f32 {%f0, %f1, %f2, %f3}, [%r0];
    st.global.f32 [%rd0], %f3;'''
                    machine = ['MOV R0, RZ'] + [f'MOV R{i}, 0x40000000'
                                               for i in range(6, 10)]
                    machine += ['STS.128 [R0], R6', 'LDS.128 R10, [R0]',
                                'STG.E [R2.64], R13']
                    stale = 'MOV R13, 0x3f800000'
                else:
                    body += '''    st.shared.f32 [%r0], %f4;
    ld.shared.f32 %f0, [%r0];
    st.global.f32 [%rd0], %f0;'''
                    machine = ['MOV R0, RZ', 'MOV R6, 0x40000000',
                               'STS [R0], R6', 'LDS R7, [R0]',
                               'STG.E [R2.64], R7']
                    stale = 'MOV R7, 0x3f800000'
                source = ptx(body)
                p = self.write('float.ptx', source)
                for validator in (lambda p, s: tval.run(p, s, NS=2, B1=1, B2=2,
                                                       log=lambda _: None),
                                  lambda p, s: batch.validate(p, s, 2, 'direct'),
                                  lambda p, s: smemval.validate(p, s, 2, 'direct')):
                    correct = self.write('float.sass', sass(machine))
                    self.assertEqual(validator(p, correct)[0], 'VALIDATED')
                    wrong_machine = machine[:-2] + [stale, machine[-1]]
                    wrong = self.write('float_wrong.sass', sass(wrong_machine))
                    self.assertEqual(validator(p, wrong)[0], 'UNPROVED')

    def test_direct_validators_check_shared_effects(self):
        source = ptx('''    .shared .align 4 .b32 buf[4];
    mov.u32 %r0, buf;
    mov.u32 %r1, 7;
    st.shared.u32 [%r0], %r1;
    st.global.u32 [%rd0], %r1;''')
        machine = ['MOV R0, RZ', 'MOV R6, 0x7', 'STS [R0], R6',
                   'STG.E [R2.64], R6']
        self.assertEqual(self.validate(source, sass(machine)), 'VALIDATED')
        wrong = sass(machine[:2] + machine[3:])
        self.assertEqual(self.validate(source, wrong), 'UNPROVED')
        p, s = self.write('shared.ptx', source), self.write('shared.sass', wrong)
        self.assertEqual(batch.validate(p, s, 2, 'direct')[0], 'UNPROVED')
        barrier_source = source.replace('    st.global',
                                         '    bar.sync 0;\n    st.global')
        self.assertEqual(self.validate(barrier_source, sass(machine)), 'UNPROVED')
        p = self.write('barrier.ptx', barrier_source)
        s = self.write('barrier.sass', sass(machine))
        self.assertEqual(batch.validate(p, s, 2, 'direct')[0], 'UNPROVED')

    def test_direct_validators_do_not_round_unaligned_shared_addresses(self):
        source = ptx('''    .shared .align 4 .b32 buf[4];
    mov.u32 %r0, 1;
    mov.u32 %r1, 7;
    st.shared.u32 [%r0], %r1;
    st.global.u32 [%rd0], %r1;''')
        machine = sass(['MOV R0, RZ', 'MOV R6, 0x7', 'STS [R0], R6',
                        'STG.E [R2.64], R6'])
        self.assertEqual(self.validate(source, machine), 'UNPROVED')
        p = self.write('unaligned.ptx', source)
        s = self.write('unaligned.sass', machine)
        result = batch.validate(p, s, 2, 'direct')
        self.assertEqual(result[0], 'UNPROVED')
        self.assertIn('not provably 4-byte aligned', result[1])

    def test_array_parameter_wrong_bank_offset_is_refused(self):
        source = ptx('    mov.u32 %r0, 7;\n'
                     '    st.global.u32 [%rd0], %r0;').replace(
                         '.entry probe(', '.entry probe(.param .b8 ignored[16], ')
        # The old parser treated ignored[16] as a one-byte scalar and assigned
        # P0 to 0x168. Its actual bank offset is 0x170.
        wrong = sass(['MOV R6, 0x7', 'STG.E [R2.64], R6'])
        wrong = wrong.replace('0x168]', '0x170]').replace('0x16c]', '0x174]')
        wrong = wrong.replace('0x160]', '0x168]').replace('0x164]', '0x16c]')
        self.assertEqual(self.validate(source, wrong), 'REFUSED')

    def test_unmodeled_parameter_load_width_is_refused(self):
        source = ptx('''    ld.param.u16 %r0, [P0];
    st.global.u32 [%rd1], %r0;''')
        wrong = sass(['MOV R6, c[0x0][0x160]', 'STG.E [R4.64], R6'])
        self.assertEqual(self.validate(source, wrong), 'REFUSED')

    def test_float_instruction_cannot_write_integer_register_file(self):
        source = ptx('''    mov.u32 %r0, 7;
    mov.f32 %r0, 0f40000000;
    st.global.u32 [%rd0], %r0;''')
        wrong = sass(['MOV R6, 0x7', 'STG.E [R2.64], R6'])
        self.assertEqual(self.validate(source, wrong), 'REFUSED')


if __name__ == '__main__':
    verification_main(verbosity=2)
