"""Ignored PTX metadata must not hide executable instructions.

Fresh NVIDIA output is the positive control. The negative disassembly removes
one real store; placing that store after .reg or .loc formerly false-accepted
the mutation. Refusing mixed source lines is an explicit parser limitation.
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from verification_unittest import main as verification_main

ROOT = Path(__file__).resolve().parents[1]
TVAL = ROOT / 'tools' / 'ptxas_tval'
sys.path.insert(0, str(TVAL))
import loopcfg
import ptxsource
import exact_pv_subject

try:
    import z3
    import batch
    import loopval
    import nestval
    import ptxexec
    import smem
    import smemval
    import tval
except ImportError:
    HAS_Z3 = False
else:
    HAS_Z3 = True

SCALAR = '''.version 7.8
.target sm_89
.address_size 64
.visible .entry directive_control(.param .u64 O)
{
 .reg .b64 %rd<2>; .reg .u32 %r<2>;
 ld.param.u64 %rd0, [O];
 mov.u32 %r0, 123;
 mov.u32 %r1, 456;
 add.u64 %rd1, %rd0, 8;
 st.global.u32 [%rd0], %r0;
 st.global.u32 [%rd1], %r1;
 ret;
}
'''

SCOPED = '''.version 7.8
.target sm_89
.address_size 64
.visible .entry scope_control(.param .u64 O)
{
 .reg .b64 %rd<2>;
 .reg .u32 %r0;
 ld.param.u64 %rd0, [O];
 add.u64 %rd1, %rd0, 8;
 mov.u32 %r0, 1;
 {
  .reg .u32 %r0;
  mov.u32 %r0, 2;
  st.global.u32 [%rd0], %r0;
 }
 st.global.u32 [%rd1], %r0;
 ret;
}
'''


class DirectiveSyntax(unittest.TestCase):
    def test_pure_directives_headers_and_repeated_declarations_are_preserved(self):
        source = '''.version 7.8
.target sm_89
.address_size 64
.file 1 "metadata; st.global.u32 [%rd0], %r0; // still a string"
.visible .entry example(
 .param .u64 A, .param .u32 N)
.maxnreg 64
.reqntid 32, 1, 1
{
 .reg .b32 %r<4>; /* safe comment */ .reg .b64 %rd<2>;
 .reg .pred %p0, %p1;
 .pragma "nounroll";
 .loc 1 2 0
 ret;
}
'''
        normalized = ptxsource.require_directive_lines(ptxsource.strip_comments(source))
        self.assertIn('.reg .b32 %r<4>;\n.reg .b64 %rd<2>;', normalized)
        self.assertIn('.param .u64 A, .param .u32 N)', normalized)
        self.assertIn('st.global.u32 [%rd0], %r0; // still a string', normalized)

    def test_every_mixed_or_unknown_directive_line_refuses(self):
        for line in (
            '.reg .u32 %r<2>; st.global.u32 [%rd0], %r0;',
            '.reg .u32 %r<2>; @%p0 bra done;',
            '.reg .pred %p<2>; and.pred %p0, %p0, %p1;',
            '.loc 1 2 0 st.global.u32 [%rd0], %r0;',
            '.pragma "nounroll"; ret;',
            '.maxnreg 32 mov.u32 %r0, 1;',
            '.file 1 "metadata" st.global.u32 [%rd0], %r0;',
            '.visible .entry example(.param .u64 O) { ret; }',
            '.param .u64 O) { st.global.u32 [%rd0], %r0;',
            '.shared .align 4 .b32 cell[1]; st.shared.b32 [0], %r0;',
            '.unknown_metadata st.global.u32 [%rd0], %r0;',
            '.func helper() { ret; }',
            '.global .u32 initialized = 7;',
        ):
            with self.subTest(line=line), self.assertRaisesRegex(Exception, 'UNMODELLED PTX directive line'):
                ptxsource.require_directive_lines(line + '\n')

    def test_guard_precedes_both_entry_and_loop_scans(self):
        with tempfile.TemporaryDirectory(prefix='y-directive-scans-') as directory:
            path = Path(directory) / 'mixed.ptx'
            path.write_text(SCALAR.replace(' st.global.u32 [%rd1], %r1;',
                ' .reg .u32 %unused<1>; st.global.u32 [%rd1], %r1;'))
            for scanner in (ptxsource.read, loopcfg.ptx_entry_points, loopcfg.ptx_back_edges):
                with self.subTest(scanner=scanner.__name__), self.assertRaisesRegex(
                        Exception, 'UNMODELLED PTX directive line'):
                    scanner(path)

    def test_structural_lines_cannot_hide_inline_payloads(self):
        for source in (
            '{ mov.u32 %r0, 1;\n}\n',
            '{\n} st.global.u32 [%rd0], %r0;\n',
            ') {\n st.global.u32 [%rd0], %r0;\n}\n',
            '{\ntail: st.global.u32 [%rd0], %r0;\n}\n',
            '{\nmov.u32 %r0, 1; ret;\n}\n',
            '{\nret; }\n',
            '{\n.reg .u32 %r<2>; }\n',
        ):
            with self.subTest(source=source), self.assertRaisesRegex(Exception, 'UNMODELLED PTX'):
                ptxsource.require_directive_lines(source)

    def test_nested_lexical_scopes_refuse_before_any_scanner_loses_register_ownership(self):
        with tempfile.TemporaryDirectory(prefix='y-directive-scopes-') as directory:
            path = Path(directory) / 'scoped.ptx'
            path.write_text(SCOPED)
            for scanner in (ptxsource.read, loopcfg.ptx_entry_points, loopcfg.ptx_back_edges):
                with self.subTest(scanner=scanner.__name__), self.assertRaisesRegex(
                        Exception, 'UNMODELLED PTX nested lexical scope'):
                    scanner(path)

    def test_comment_lexer_and_proof_subject_identity_remain_independent(self):
        source = (ROOT / 'tests/exact_pv.ptx').read_text()
        exact_pv_subject.require_proved_subject(source)
        # This changes whitespace only, so the reviewed operational PTX subject
        # is identical. Its line-oriented execution syntax must still refuse.
        changed, count = re.subn(r'(\.reg \.pred [^;]+;)\s+(ld\.param)',
                                 r'\1 \2', source, count=1)
        self.assertEqual(count, 1)
        exact_pv_subject.require_proved_subject(changed)
        with self.assertRaisesRegex(Exception, 'UNMODELLED PTX directive line'):
            ptxsource.require_directive_lines(ptxsource.strip_comments(changed))

    @unittest.skipUnless(HAS_Z3, 'shared layout regression requires z3-solver')
    def test_repeated_shared_declarations_cannot_hide_a_second_symbol(self):
        with tempfile.TemporaryDirectory(prefix='y-directive-layout-') as directory:
            path = Path(directory) / 'shared.ptx'
            path.write_text('.shared .align 4 .b32 a[1]; .shared .align 4 .b32 b[1];\n')
            with self.assertRaisesRegex(Exception, '2 .shared arrays'):
                smem.layout(path)


@unittest.skipUnless(HAS_Z3, 'whole executor checks require z3-solver')
class WholeExecutorDirectiveChecks(unittest.TestCase):
    def test_whole_executor_checks_directives_before_executing_regions(self):
        with tempfile.TemporaryDirectory(prefix='y-directive-executor-') as directory:
            path = Path(directory) / 'mixed.ptx'
            path.write_text(SCALAR.replace(' st.global.u32 [%rd1], %r1;',
                ' .reg .u32 %unused<1>; st.global.u32 [%rd1], %r1;'))
            with self.assertRaisesRegex(Exception, 'UNMODELLED PTX directive line'):
                ptxexec.run_ptx(path, {})


@unittest.skipUnless(HAS_Z3, 'fresh directive controls require z3-solver')
class FreshDirectiveControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(shutil.which(tool) for tool in ('ptxas', 'nvdisasm')):
            raise unittest.SkipTest('fresh directive controls require ptxas and nvdisasm')
        temporary = tempfile.TemporaryDirectory(prefix='y-directive-controls-')
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.artifacts = {}
        for name, source, optimization in (
            ('scalar', SCALAR, 3),
            ('loop', (TVAL / 'mem/loop_swap.ptx').read_text(), 1),
            # `corpus/` is generated and gitignored; its .ptx is a byte copy of
            # the committed file, so read that and depend on no untracked state.
            ('nested', (ROOT / 'tests/y_cpu_matmul.ptx').read_text(), 1),
        ):
            cls.artifacts[name] = cls.compile(name, source, optimization)

    @classmethod
    def compile(cls, name, source, optimization):
        ptx, cubin, sass = (cls.directory / (name + suffix)
                            for suffix in ('.ptx', '.cubin', '.sass'))
        ptx.write_text(source)
        subprocess.run(['ptxas', f'-O{optimization}', '-arch=sm_89', str(ptx), '-o', str(cubin)],
                       check=True, capture_output=True, timeout=30)
        sass.write_bytes(subprocess.run(['nvdisasm', '-c', str(cubin)],
                        check=True, capture_output=True, timeout=30).stdout)
        return ptx, sass

    def validators(self):
        return {
            'tval': lambda p, s: tval.run(str(p), str(s), NS=2, B1=3, B2=3, log=lambda _: None),
            'batch': lambda p, s: batch.validate(str(p), str(s), 3, 'direct'),
            'batch.validate2': lambda p, s: batch.validate2(str(p), str(s), 3),
            'smemval': lambda p, s: smemval.validate(str(p), str(s), 3, 'direct'),
            'loopval': lambda p, s: loopval.validate(str(p), str(s), budget=3, samples=8, verbose=False),
            'nestval': lambda p, s: nestval.validate(str(p), str(s), budget=3, verbose=False),
        }

    def result(self, validate, ptx, sass):
        try:
            return validate(ptx, sass)
        except Exception as error:
            if 'UNMODELLED' not in str(error) and 'refusing, not guessing' not in str(error):
                raise
            return 'REFUSED', str(error), 0

    def test_genuine_scalar_loop_and_nested_translations_still_validate(self):
        applicable = {'scalar': ('tval', 'batch', 'batch.validate2', 'smemval'),
                      'loop': ('loopval', 'nestval'), 'nested': ('nestval',)}
        validators = self.validators()
        for subject, names in applicable.items():
            for name in names:
                with self.subTest(subject=subject, validator=name):
                    result = self.result(validators[name], *self.artifacts[subject])
                    self.assertEqual(result[0], 'VALIDATED', result)
                    self.assertGreater(result[2], 0)

    def test_deleted_store_is_unproved_and_mixed_directives_refuse_every_public_path(self):
        ptx, sass = self.artifacts['scalar']
        machine = sass.read_text()
        stores = re.findall(r'(?m)^\s*/\*[0-9a-fA-F]+\*/\s*STG[^\n]*\n', machine)
        self.assertEqual(len(stores), 2, 'fresh two-store mutation anchor moved')
        missing = self.directory / 'scalar-missing-store.sass'
        missing.write_text(machine.replace(stores[-1], ''))
        validators = self.validators()
        for name in ('tval', 'batch', 'batch.validate2', 'smemval'):
            result = self.result(validators[name], ptx, missing)
            self.assertEqual(result[0], 'UNPROVED', result)
            self.assertIn('load/store counts', result[1])
        for directive in ('reg', 'loc'):
            prefix = '.reg .u32 %unused<1>;' if directive == 'reg' else '.loc 1 1 0'
            source = SCALAR.replace(' st.global.u32 [%rd1], %r1;',
                                    ' ' + prefix + ' st.global.u32 [%rd1], %r1;')
            if directive == 'loc':
                source = source.replace('.visible .entry', '.file 1 "subject.ptx"\n.visible .entry')
            changed, genuine = self.compile('scalar-' + directive, source, 3)
            # These are legal PTX accepted by the compiler; support for their
            # execution syntax is deliberately refused on both translations.
            for name, validate in validators.items():
                for proposed in (genuine, missing):
                    with self.subTest(directive=directive, validator=name, machine=proposed.name):
                        result = self.result(validate, changed, proposed)
                        self.assertEqual(result[0], 'REFUSED', result)
                        self.assertIn('UNMODELLED PTX directive line', result[1])
                        self.assertEqual(result[2], 0)

    def test_mixed_loop_and_nested_effects_refuse_before_cfg_construction(self):
        validators = self.validators()
        for subject in ('loop', 'nested'):
            ptx, _ = self.artifacts[subject]
            for directive in ('reg', 'loc'):
                prefix = '.reg .u32 %unused<1>;' if directive == 'reg' else '.loc 1 1 0'
                source, count = re.subn(r'(?m)^(\s*)((?:@!?%[\w$]+\s+)?st\.global\.[^\n]+)$',
                                       r'\1' + prefix + r' \2', ptx.read_text(), count=1)
                self.assertEqual(count, 1)
                if directive == 'loc':
                    source = source.replace('.visible .entry', '.file 1 "subject.ptx"\n.visible .entry')
                changed, sass = self.compile(subject + '-' + directive, source, 1)
                for name in ('loopval', 'nestval'):
                    with self.subTest(subject=subject, directive=directive, validator=name):
                        result = self.result(validators[name], changed, sass)
                        self.assertEqual(result[0], 'REFUSED', result)
                        self.assertIn('UNMODELLED PTX directive line', result[1])
                        self.assertEqual(result[2], 0)

    def test_legal_inline_structure_and_instruction_lists_refuse_every_public_path(self):
        variants = {
            'open': SCALAR.replace('{\n .reg', '{ .reg', 1),
            'close': SCALAR.replace(' ret;\n}', ' ret; }', 1),
            'entry_open': SCALAR.replace('(.param .u64 O)\n{', '(.param .u64 O) {', 1),
            'parameters_open': SCALAR.replace('(.param .u64 O)\n{', '(\n .param .u64 O\n) {', 1),
            'label': SCALAR.replace(' st.global.u32 [%rd1], %r1;',
                                   ' tail: st.global.u32 [%rd1], %r1;', 1),
            'instruction_list': SCALAR.replace('st.global.u32 [%rd0], %r0;\n ',
                                               'st.global.u32 [%rd0], %r0; ', 1),
            'declaration_close': SCALAR.replace('\n}\n', '\n .reg .u32 %unused<1>; }\n', 1),
        }
        for variant, source in variants.items():
            ptx, sass = self.compile('inline-' + variant, source, 3)
            for name, validate in self.validators().items():
                with self.subTest(variant=variant, validator=name):
                    result = self.result(validate, ptx, sass)
                    self.assertEqual(result[0], 'REFUSED', result)
                    self.assertIn('UNMODELLED PTX', result[1])
                    self.assertEqual(result[2], 0)

    def test_register_shadowing_cannot_validate_an_incorrect_outer_store(self):
        ptx, sass = self.compile('scoped', SCOPED, 3)
        machine, count = re.subn(
            r'((?:MOV R\d+, |IMAD\.MOV\.U32 R\d+, RZ, RZ, ))0x1(\s*;)',
            r'\g<1>0x2\2', sass.read_text())
        self.assertEqual(count, 1, 'outer-register immediate mutation anchor moved')
        changed = self.directory / 'scoped-outer-value-changed.sass'
        changed.write_text(machine)
        # PTX writes 2, then 1. The old flat register model retained the inner
        # r0=2 and falsely validated this two-store SASS mutation (6 obligations).
        for name, validate in self.validators().items():
            for proposed in (sass, changed):
                with self.subTest(validator=name, machine=proposed.name):
                    result = self.result(validate, ptx, proposed)
                    self.assertEqual(result[0], 'REFUSED', result)
                    self.assertIn('UNMODELLED PTX nested lexical scope', result[1])
                    self.assertEqual(result[2], 0)
        # The same arithmetic with explicit distinct register names is in the
        # licensed subset. This proves that the refusal is a scoping boundary.
        flat = SCOPED.replace(' {\n  .reg .u32 %r0;\n  mov.u32 %r0, 2;\n  st.global.u32 [%rd0], %r0;\n }',
                              ' .reg .u32 %r1;\n mov.u32 %r1, 2;\n st.global.u32 [%rd0], %r1;')
        self.assertNotEqual(flat, SCOPED)
        flat_ptx, flat_sass = self.compile('scope-alpha-renamed', flat, 3)
        flat_machine, count = re.subn(
            r'((?:MOV R\d+, |IMAD\.MOV\.U32 R\d+, RZ, RZ, ))0x1(\s*;)',
            r'\g<1>0x2\2', flat_sass.read_text())
        self.assertEqual(count, 1)
        flat_changed = self.directory / 'scope-alpha-renamed-wrong-store.sass'
        flat_changed.write_text(flat_machine)
        for name in ('tval', 'batch', 'batch.validate2', 'smemval'):
            with self.subTest(flat_validator=name):
                positive = self.result(self.validators()[name], flat_ptx, flat_sass)
                self.assertEqual(positive[0], 'VALIDATED', positive)
                self.assertGreater(positive[2], 0)
                negative = self.result(self.validators()[name], flat_ptx, flat_changed)
                self.assertEqual(negative[0], 'UNPROVED', negative)


if __name__ == '__main__':
    verification_main()
