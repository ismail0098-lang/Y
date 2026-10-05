"""Translation checks cover the complete modeled legal CUDA x launch domain."""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from verification_unittest import main as verification_main

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'ptxas_tval'))
import domain

try:
    import z3
    import batch
    import tval
except ImportError:
    HAS_Z3 = False
else:
    HAS_Z3 = True


class TargetAgreement(unittest.TestCase):
    def test_missing_duplicate_featured_and_mismatched_targets_refuse(self):
        with tempfile.TemporaryDirectory(prefix='y-target-domain-') as directory:
            ptx, sass = (Path(directory) / name for name in ('source.ptx', 'machine.sass'))
            ptx.write_text('.target sm_89 // supported target\n')
            for subject in ('', '.target sm_86\n', '.target sm_89\n.target sm_89\n',
                            '.target sm_89, texmode_independent\n'):
                with self.subTest(subject=subject):
                    sass.write_text(subject)
                    with self.assertRaisesRegex(Exception, 'UNMODELLED'):
                        domain.require_matching_targets(ptx, sass)
            sass.write_text('/* .target sm_86 */\n.target sm_89\n')
            self.assertEqual(domain.require_matching_targets(ptx, sass), 'sm_89')


@unittest.skipUnless(HAS_Z3, 'launch domain checks require z3-solver')
class LaunchDomain(unittest.TestCase):
    def test_large_legal_block_indices_are_in_the_solver_domain(self):
        sym = {'tid_x': z3.BitVec('domain_tid', 32), 'ctaid_x': z3.BitVec('domain_cta', 32)}
        for block, expected in ((1 << 24, z3.sat), (0x7ffffffe, z3.sat),
                                (0x7fffffff, z3.unsat)):
            solver = z3.Solver()
            solver.add(*domain.launch_preconditions(sym), sym['ctaid_x'] == block)
            self.assertEqual(solver.check(), expected)

    def test_fresh_o3_large_block_value_and_truncation(self):
        if not all(shutil.which(tool) for tool in ('ptxas', 'nvdisasm')):
            self.skipTest('fresh launch domain check requires ptxas and nvdisasm')
        with tempfile.TemporaryDirectory(prefix='y-full-grid-domain-') as directory:
            directory = Path(directory)
            ptx, cubin, sass = (directory / name for name in ('grid.ptx', 'grid.cubin', 'grid.sass'))
            ptx.write_text(''' .version 7.8
.target sm_89
.address_size 64
.visible .entry grid(.param .u64 O)
{
.reg .b32 %r<8>;
.reg .b64 %rd<4>;
ld.param.u64 %rd0, [O];
mov.u32 %r0, %ctaid.x;
shl.b32 %r1, %r0, 1;
add.u32 %r2, %r1, 3;
cvt.u64.u32 %rd3, %r2;
shl.b64 %rd1, %rd3, 2;
add.u64 %rd2, %rd0, %rd1;
st.global.u32 [%rd2], %r0;
ret;
}
''')
            subprocess.run(['ptxas', '-O3', '-arch=sm_89', str(ptx), '-o', str(cubin)],
                           capture_output=True, check=True, timeout=30)
            machine = subprocess.run(['nvdisasm', '-c', str(cubin)],
                                     capture_output=True, check=True, timeout=30).stdout.decode()
            sass.write_text(machine)
            store = re.search(r'(?m)^(\s*/\*[0-9a-f]+\*/\s+)(STG\.E \[[^\]]+\], )(R\d+)(\s*;.*)$', machine)
            self.assertIsNotNone(store, 'fresh ptxas store shape changed')
            # Keep the address unchanged; truncate only the stored CTA value.
            altered = (machine[:store.start()] +
                       f'/*fff0*/ LOP3.LUT R20, {store[3]}, 0xffffff, RZ, 0xc0, !PT ;\n' +
                       store[1] + store[2] + 'R20' + store[4] + machine[store.end():])
            pcs = iter(range(0, 1 << 20, 16))
            altered = re.sub(r'/\*[0-9a-fA-F]+\*/', lambda _: f'/*{next(pcs):04x}*/', altered)
            wrong = directory / 'truncated.sass'
            wrong.write_text(altered)
            for validator in ('tval', 'batch'):
                for path, expected in ((sass, 'VALIDATED'), (wrong, 'UNPROVED')):
                    with self.subTest(validator=validator, subject=path.name):
                        result = (tval.run(str(ptx), str(path), NS=2, B1=3, B2=5, log=lambda _: None)
                                  if validator == 'tval' else
                                  batch.validate(str(ptx), str(path), 5, 'direct'))
                        self.assertEqual(result[0], expected, result)
            mismatch = directory / 'mismatch.sass'
            other_target, count = re.subn(r'(?m)^(\s*\.target\s+)sm_89\s*$',
                                          r'\g<1>sm_86', machine, count=1)
            self.assertEqual(count, 1, 'disassembler target anchor moved')
            mismatch.write_text(other_target)
            for validator in (lambda p, s: batch.validate(p, s, 3), tval.run):
                try:
                    result = validator(str(ptx), str(mismatch))
                except Exception as error:
                    self.assertRegex(str(error), 'target mismatch')
                else:
                    self.assertEqual(result[0], 'REFUSED', result)
                    self.assertIn('target mismatch', result[1])


if __name__ == '__main__':
    verification_main()
