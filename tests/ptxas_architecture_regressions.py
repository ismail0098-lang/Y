"""A target declaration must not authorize unlicensed ISA/ABI assumptions.

Fresh sm89 translations remain positive controls. Relabeled declarations are
deliberate unsupported subjects, not claims about what NVIDIA tools emit.
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
import domain

try:
    import z3
    import batch
    import loopval
    import nestval
    import smemval
    import tval
except ImportError:
    HAS_Z3 = False
else:
    HAS_Z3 = True

SHARED = b'''.version 7.8
.target sm_89
.address_size 64
.visible .entry shared_control(.param .u64 O, .param .u32 X)
{
 .reg .b32 %r<4>;
 .reg .b64 %rd<2>;
 .shared .align 4 .b32 cell[1];
 ld.param.u64 %rd0, [O];
 ld.param.u32 %r0, [X];
 mov.u64 %rd1, cell;
 st.shared.b32 [%rd1], %r0;
 bar.sync 0;
 ld.shared.b32 %r1, [%rd1];
 st.global.b32 [%rd0], %r1;
 ret;
}
'''


class ArchitectureLicense(unittest.TestCase):
    def test_matching_targets_require_an_architecture_license(self):
        with tempfile.TemporaryDirectory(prefix='y-architecture-license-') as directory:
            ptx, sass = (Path(directory) / name for name in ('subject.ptx', 'subject.sass'))
            for target in ('sm_80', 'sm_86', 'sm_90', 'sm_90a', 'sm_90f', 'sm_999'):
                with self.subTest(target=target):
                    ptx.write_text('.target ' + target + '\n')
                    sass.write_text('.target ' + target + '\n')
                    with self.assertRaisesRegex(Exception, 'UNMODELLED unsupported architecture'):
                        domain.require_matching_targets(ptx, sass)
            ptx.write_text('.target sm_89\n')
            sass.write_text('.target sm_89\n')
            self.assertEqual(domain.require_matching_targets(ptx, sass), 'sm_89')

    def test_target_disagreement_is_checked_before_licensing(self):
        with tempfile.TemporaryDirectory(prefix='y-architecture-mismatch-') as directory:
            ptx, sass = (Path(directory) / name for name in ('subject.ptx', 'subject.sass'))
            ptx.write_text('.target sm_999\n')
            sass.write_text('.target sm_90a\n')
            with self.assertRaisesRegex(Exception, 'UNMODELLED target mismatch'):
                domain.require_matching_targets(ptx, sass)


@unittest.skipUnless(HAS_Z3, 'public architecture checks require z3-solver')
class PublicValidatorLicenses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(shutil.which(tool) for tool in ('ptxas', 'nvdisasm')):
            raise unittest.SkipTest('architecture controls require ptxas and nvdisasm')
        temporary = tempfile.TemporaryDirectory(prefix='y-architecture-controls-')
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.artifacts = {}
        for name, source in (('scalar', (TVAL / 'fma/rn.ptx').read_bytes()),
                             ('shared', SHARED),
                             ('loop', (TVAL / 'mem/loop_swap.ptx').read_bytes())):
            ptx, cubin, sass = (cls.directory / (name + suffix)
                                for suffix in ('.ptx', '.cubin', '.sass'))
            ptx.write_bytes(source)
            subprocess.run(['ptxas', '-O1', '-arch=sm_89', str(ptx), '-o', str(cubin)],
                           check=True, capture_output=True, timeout=30)
            sass.write_bytes(subprocess.run(['nvdisasm', '-c', str(cubin)],
                check=True, capture_output=True, timeout=30).stdout)
            cls.artifacts[name] = (ptx, sass)

    def validators(self, subject):
        if subject in ('scalar', 'shared'):
            return {
                'tval': lambda p, s: tval.run(p, s, NS=2, B1=3, B2=3, log=lambda _: None),
                'batch': lambda p, s: batch.validate(p, s, 3, 'direct'),
                'batch.validate2': lambda p, s: batch.validate2(p, s, 3),
                'smemval': lambda p, s: smemval.validate(p, s, 3, 'direct'),
            }
        return {
            'loopval': lambda p, s: loopval.validate(p, s, budget=3, verbose=False),
            'nestval': lambda p, s: nestval.validate(p, s, budget=3, verbose=False),
        }

    def result(self, validator, ptx, sass):
        try:
            return validator(str(ptx), str(sass))
        except Exception as error:
            if 'UNMODELLED' not in str(error) and 'refusing, not guessing' not in str(error):
                raise
            return 'REFUSED', str(error), 0

    def test_real_sm89_translations_validate_through_every_public_path(self):
        for subject, (ptx, sass) in self.artifacts.items():
            for name, validator in self.validators(subject).items():
                with self.subTest(subject=subject, validator=name):
                    result = self.result(validator, ptx, sass)
                    self.assertEqual(result[0], 'VALIDATED', result)
                    self.assertGreater(result[2], 0)

    def test_matching_unlicensed_targets_refuse_through_every_public_path(self):
        for subject, (ptx, sass) in self.artifacts.items():
            for target in ('sm_80', 'sm_86', 'sm_90a', 'sm_999'):
                changed_ptx, changed_sass = (self.directory / (subject + '-' + target + suffix)
                                             for suffix in ('.ptx', '.sass'))
                for source, destination in ((ptx, changed_ptx), (sass, changed_sass)):
                    text, count = re.subn(r'(?m)^(\s*\.target\s+)sm_89\s*$',
                                         r'\g<1>' + target, source.read_text())
                    self.assertEqual(count, 1, 'target declaration anchor moved')
                    destination.write_text(text)
                for name, validator in self.validators(subject).items():
                    with self.subTest(subject=subject, target=target, validator=name):
                        result = self.result(validator, changed_ptx, changed_sass)
                        self.assertEqual(result[0], 'REFUSED', result)
                        self.assertIn('unsupported architecture', result[1])
                        self.assertEqual(result[2], 0)

    def test_smemval_and_refinement_preserve_full_legal_grid_checks(self):
        ptx, cubin, sass = (self.directory / ('grid' + suffix)
                            for suffix in ('.ptx', '.cubin', '.sass'))
        ptx.write_text('''.version 7.8
.target sm_89
.address_size 64
.visible .entry grid(.param .u64 O)
{
 .reg .b32 %r<4>;
 .reg .b64 %rd<2>;
 ld.param.u64 %rd0, [O];
 mov.u32 %r0, %ctaid.x;
 st.global.u32 [%rd0], %r0;
 ret;
}
''')
        subprocess.run(['ptxas', '-O3', '-arch=sm_89', str(ptx), '-o', str(cubin)],
                       check=True, capture_output=True, timeout=30)
        machine = subprocess.run(['nvdisasm', '-c', str(cubin)],
                                 check=True, capture_output=True, timeout=30).stdout.decode()
        sass.write_text(machine)
        store = re.search(r'(?m)^(\s*/\*[0-9a-fA-F]+\*/\s+)(STG\.E \[[^\]]+\], )(R\d+)(\s*;.*)$', machine)
        self.assertIsNotNone(store, 'fresh CTA-value store shape changed')
        changed = (machine[:store.start()] +
                   f'/*fff0*/ LOP3.LUT R20, {store[3]}, 0xffffff, RZ, 0xc0, !PT ;\n' +
                   store[1] + store[2] + 'R20' + store[4] + machine[store.end():])
        pcs = iter(range(0, 1 << 20, 16))
        changed = re.sub(r'/\*[0-9a-fA-F]+\*/', lambda _: f'/*{next(pcs):04x}*/', changed)
        wrong = self.directory / 'grid-truncated.sass'
        wrong.write_text(changed)
        for name in ('batch.validate2', 'smemval'):
            validator = self.validators('scalar')[name]
            for path, expected in ((sass, 'VALIDATED'), (wrong, 'UNPROVED')):
                with self.subTest(validator=name, subject=path.name):
                    result = self.result(validator, ptx, path)
                    self.assertEqual(result[0], expected, result)


if __name__ == '__main__':
    verification_main()
