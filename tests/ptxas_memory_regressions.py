"""Memory extents, shared alignment and access-trace controls without a GPU.

Fresh cubins provide PTXAS controls. Mutated disassembly and edited loop
fixtures exercise the symbolic validators; they are not assembled artifacts.
Negative controls require explicit structural refusals or SAT diagnostics.
"""

import operator
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

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
    import loopval
    import memorder
    import mulmode
    import ptxexec
    import smem
    import smemval
    import tval


def kernel(body, parameters='.param .u64 I, .param .u64 O'):
    return ('.version 7.8\n.target sm_89\n.address_size 64\n'
            f'.visible .entry memory_probe({parameters})\n{{\n'
            '.reg .b32 %r<12>;\n.reg .b64 %rd<8>;\n.reg .pred %p<3>;\n'
            + body.strip() + '\nret;\n}\n')


@unittest.skipUnless(HAS_Z3, 'memory model checks require z3-solver')
class MemoryModel(unittest.TestCase):
    def setUp(self):
        self.address = z3.BitVec('memory_model_address', 64)
        self.guard = z3.Bool('memory_model_guard')

    def state(self):
        state = SimpleNamespace()
        memorder.install(state)
        return state

    def test_load_widths_are_atomic_and_preserve_address_guard_pairs(self):
        state = self.state()
        for width in (1, 2, 4, 8, 16):
            state.loads.append((self.address, self.guard, width))
        self.assertEqual(state.load_widths, [1, 2, 4, 8, 16])
        self.assertEqual(state.loads, [(self.address, self.guard)] * 5)
        self.assertEqual(memorder.order_of([state]), 'LLLLL')
        invalid = ((self.address, self.guard), (self.address, self.guard, 3),
                   (self.address, self.guard, True), (self.address, self.guard, 4.0),
                   (z3.BitVecVal(0, 32), self.guard, 4),
                   (self.address, z3.BitVecVal(1, 1), 4))
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(memorder.Refusal):
                state.loads.append(item)
            self.assertEqual(len(state.loads), 5)
            self.assertEqual(state.load_widths, [1, 2, 4, 8, 16])
            self.assertEqual(memorder.order_of([state]), 'LLLLL')

    def test_trace_and_width_mutations_cannot_bypass_metadata(self):
        state = self.state()
        state.loads.append((self.address, self.guard, 4))
        for sequence in (state.loads, state.load_widths):
            mutations = (lambda s: operator.imul(s, 0), lambda s: operator.iadd(s, []),
                         lambda s: s.clear(), lambda s: s.pop(),
                         lambda s: operator.setitem(s, 0, None),
                         lambda s: operator.delitem(s, 0), lambda s: s.extend([]))
            for mutate in mutations:
                with self.subTest(sequence=type(sequence).__name__, mutate=mutate):
                    with self.assertRaises(memorder.Refusal):
                        mutate(sequence)
        with self.assertRaises(memorder.Refusal):
            state.load_widths.append(16)
        self.assertEqual(state.load_widths, [4])
        self.assertEqual(len(state.loads), 1)
        self.assertEqual(memorder.order_of([state]), 'L')

    def test_read_through_rejects_bad_shapes_even_without_prior_stores(self):
        base = z3.BitVecVal(0x44332211, 32)
        valid_store = (self.address, base, z3.BoolVal(True))
        for stores in ([], [valid_store]):
            for address, value, width in ((self.address, base, 3),
                                           (self.address, base, True),
                                           (self.address, base, 8),
                                           (z3.BitVecVal(0, 32), base, 4),
                                           (self.address, z3.BitVecVal(0, 16), 4)):
                with self.subTest(stores=bool(stores), width=width):
                    with self.assertRaises(memorder.Refusal):
                        memorder.read_through(stores, address, value, width)
        self.assertIs(memorder.read_through([], self.address, base), base)
        for store in (None, (), (self.address, base),
                      (self.address, z3.BitVecVal(0, 64), z3.BoolVal(True))):
            with self.subTest(store=store), self.assertRaises(memorder.Refusal):
                memorder.store_width(store)

    def test_byte_read_through_matches_modular_little_endian_oracle(self):
        mask = (1 << 64) - 1
        base = 0x44332211
        for start in (0, 0x1000, mask - 1):
            # Overlaps, last writer wins, skipped writes, and address wrap.
            source = ((start - 2, 0x88776655, 4, True),
                      (start + 1, 0x99, 1, True),
                      (start + 2, 0xABCD, 2, False),
                      (start + 3, 0xEF01, 2, True))
            stores = [(z3.BitVecVal(a & mask, 64), z3.BitVecVal(v, 8*w), z3.BoolVal(g))
                      for a, v, w, g in source]
            for width in (1, 2, 4):
                expected = base
                for k in range(width):
                    byte = (base >> (8*k)) & 0xff
                    for address, value, size, guard in source:
                        distance = ((start + k) - address) & mask
                        if guard and distance < size:
                            byte = (value >> (8*distance)) & 0xff
                    expected = (expected & ~(0xff << (8*k))) | (byte << (8*k))
                actual = memorder.read_through(stores, z3.BitVecVal(start, 64),
                                              z3.BitVecVal(base, 32), width)
                self.assertEqual(z3.simplify(actual).as_long(), expected)

    def test_shared_alignment_uses_total_width_window_and_guard(self):
        context = z3.Context()
        address = z3.BitVec('shared_alignment_address', 64, context)
        active = z3.Bool('shared_alignment_active', context)
        for width in (4, 8, 16):
            for offset in (0, 4, 8, 12, 16, (1 << 32) + 4):
                with self.subTest(width=width, offset=offset):
                    claim = smem.require_aligned(address, width, active)
                    solver = z3.Solver(ctx=context)
                    solver.set(timeout=5000)
                    solver.add(address == offset, active, z3.Not(claim))
                    expected = z3.unsat if offset % width == 0 else z3.sat
                    self.assertEqual(solver.check(), expected, solver.reason_unknown())
                    solver = z3.Solver(ctx=context)
                    solver.set(timeout=5000)
                    solver.add(address == offset, z3.Not(active), z3.Not(claim))
                    self.assertEqual(solver.check(), z3.unsat, solver.reason_unknown())
        for address, width, guard in ((z3.BitVecVal(0, 16), 4, None),
                                     (self.address, 3, None), (self.address, True, None),
                                     (self.address, 4, z3.BitVecVal(1, 1))):
            with self.subTest(width=width), self.assertRaisesRegex(Exception, 'UNMODELLED SHARED ALIGNMENT'):
                smem.require_aligned(address, width, guard)

    def test_malformed_ptx_vector_components_and_memory_arity_refuse(self):
        for count in (2, 4):
            valid = '{' + ','.join(f'%r{i}' for i in range(count)) + '}'
            malformed = ('{' + ','.join(f'%r{i}' for i in range(count - 1)) + '}',
                         '{' + ','.join(f'%r{i}' for i in range(count + 1)) + '}',
                         '{' + ','.join([''] + [f'%r{i}' for i in range(1, count)]) + '}')
            for vector in malformed:
                for space in ('global', 'shared'):
                    for operation in ('ld', 'st'):
                        operands = f'{vector}, [%rd0]' if operation == 'ld' else f'[%rd0], {vector}'
                        line = f'{operation}.{space}.v{count}.u32 {operands}'
                        # Shared v2 is outside the modeled opcode set; malformed
                        # global vectors and modeled shared v4 refuse by name.
                        with self.subTest(line=line), self.assertRaisesRegex(Exception, 'UNMODELLED.*refusing, not guessing'):
                            ptxexec.run_lines([line], batch.mk(mulmode.direct, {}))
            for operation in ('ld', 'st'):
                operands = f'{valid}, [%rd0], 7' if operation == 'ld' else f'[%rd0], {valid}, 7'
                line = f'{operation}.global.v{count}.u32 {operands}'
                with self.subTest(line=line), self.assertRaisesRegex(Exception, 'UNMODELLED PTX OPERAND COUNT'):
                    ptxexec.run_lines([line], batch.mk(mulmode.direct, {}))
        for line in ('ld.global.u32 %r0, [%rd0], 7', 'st.global.u32 [%rd0], %r0, 7',
                     'ld.shared.u32 %r0, [%r0], 7', 'st.shared.u32 [%r0], %r1, 7',
                     'ld.global.u32 %r0, [%rd0],', 'st.global.u32 [%rd0], %r0,'):
            with self.subTest(line=line), self.assertRaisesRegex(Exception, 'UNMODELLED PTX OPERAND COUNT'):
                ptxexec.run_lines([line], batch.mk(mulmode.direct, {}))
        for count in (2, 4):
            repeated = '{' + ','.join(['%r0'] * count) + '}'
            for space in ('global', 'shared') if count == 4 else ('global',):
                line = f'ld.{space}.v{count}.u32 {repeated}, [%rd0]'
                with self.subTest(line=line), self.assertRaisesRegex(Exception, 'UNMODELLED PTX vector operand .*distinct destinations'):
                    ptxexec.run_lines([line], batch.mk(mulmode.direct, {}))
            # Repeated source registers are valid and must retain all words.
            state = ptxexec.run_lines([f'st.global.v{count}.u32 [%rd0], {repeated}'],
                                     batch.mk(mulmode.direct, {}))
            self.assertEqual(len(state.stores), count)


class TemporarySources(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='y-memory-validation-')
        self.directory = Path(temporary.name)
        self.addCleanup(temporary.cleanup)

    def write(self, name, text):
        path = self.directory / name
        path.write_text(text)
        return str(path)


@unittest.skipUnless(HAS_Z3, 'memory validation controls require z3-solver')
class ValidatorMemory(TemporarySources):
    def test_loop_discarded_epilogue_load_extents_are_checked(self):
        source = (TVAL / 'mem' / 'loop_swap.ptx').read_text().replace(
            '    st.global.u32 [%rd0], %r2;',
            '    ld.global.u32 %r6, [%rd0];\n    st.global.u32 [%rd0], %r2;')
        original = (TVAL / 'mem' / 'loop_swap.sass').read_text()
        machine = original.replace('/*0100*/',
                                   '/*0100*/ LDG.E R6, [R2.64] ;\n        /*0100*/', 1)
        counter = iter(range(0, 10000, 16))
        machine = re.sub(r'/\*[0-9a-fA-F]+\*/', lambda _: f'/*{next(counter):04x}*/', machine)
        p = self.write('loop_load.ptx', source)
        genuine_fixture = self.write('loop_load.sass', machine)
        result = loopval.validate(p, genuine_fixture, budget=2, samples=8, verbose=False)
        self.assertEqual(result[0], 'VALIDATED', result)
        for label, text, diagnostic in (
                ('dropped', machine.replace('LDG.E R6, [R2.64]', 'NOP'),
                 'epilogue load counts 1 vs 0'),
                ('widened', machine.replace('LDG.E R6, [R2.64]', 'LDG.E.128 R6, [R2.64]'),
                 'epilogue load 0 width: ptx 32 bits, sass 128 bits')):
            with self.subTest(label=label):
                result = loopval.validate(p, self.write(label + '.sass', text),
                                          budget=2, samples=8, verbose=False)
                self.assertEqual(result[0], 'UNPROVED', result)
                self.assertEqual(result[1], diagnostic)

    def test_loop_header_global_access_refuses_explicitly(self):
        source = (TVAL / 'mem' / 'loop_swap.ptx').read_text().replace(
            '    $LOOP_START_0:',
            '    $LOOP_START_0:\n    ld.global.u32 %r6, [%rd0];')
        with self.assertRaisesRegex(memorder.Refusal, 'global load in the PTX loop header'):
            loopval.validate(self.write('header_load.ptx', source),
                             str(TVAL / 'mem' / 'loop_swap.sass'), budget=2, verbose=False)


@unittest.skipUnless(HAS_Z3, 'PTXAS memory controls require z3-solver')
class RealToolchain(TemporarySources):
    @classmethod
    def setUpClass(cls):
        missing = [name for name in ('ptxas', 'nvdisasm') if not shutil.which(name)]
        if missing:
            raise unittest.SkipTest('PTXAS memory controls require ' + ', '.join(missing))

    def assemble(self, source, name):
        p = self.write(name + '.ptx', source)
        cubin = self.directory / (name + '.cubin')
        result = subprocess.run(['ptxas', '-O1', '-arch=sm_89', p, '-o', str(cubin)],
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertTrue(cubin.read_bytes().startswith(b'\x7fELF'))
        result = subprocess.run(['nvdisasm', '-c', str(cubin)], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return p, self.write(name + '.sass', result.stdout), result.stdout

    def validate_both(self, p, s):
        return (tval.run(p, s, NS=2, B1=1, B2=2, log=lambda _: None),
                smemval.validate(p, s, budget=2, mode='wide'))

    def test_genuine_scalar_load_and_discarded_widening(self):
        p, s, text = self.assemble(kernel('''
ld.param.u64 %rd0, [I];
ld.param.u64 %rd1, [O];
ld.global.u32 %r0, [%rd0];
st.global.u32 [%rd1], %r0;
'''), 'scalar_load')
        for result in self.validate_both(p, s):
            self.assertEqual(result[0], 'VALIDATED', result)
        text, changes = re.subn(r'LDG\.E (R\d+, \[R\d+\.64\])', r'LDG.E.128 \1', text)
        self.assertEqual(changes, 1)
        for result in self.validate_both(p, self.write('widened.sass', text)):
            self.assertEqual(result[0], 'UNPROVED', result)
            self.assertEqual(result[1], 'load 0 width: ptx 32 bits, sass 128 bits')

    def test_genuine_shared_vector_alignment_and_undefined_offset(self):
        for offset in (0, 4):
            source = kernel(f'''
.shared .align 16 .b32 slot[4096];
ld.param.u64 %rd0, [O];
mov.u32 %r0, %tid.x;
mul.lo.u32 %r0, %r0, 16;
add.u32 %r0, %r0, {offset};
st.shared.v4.u32 [%r0], {{0,0,0,0}};
st.global.u32 [%rd0], %r0;
''', '.param .u64 O')
            p, s, text = self.assemble(source, 'vector_alignment_' + str(offset))
            self.assertIn('STS.128 ', text)
            for result in self.validate_both(p, s):
                with self.subTest(offset=offset):
                    self.assertEqual(result[0], 'UNPROVED' if offset else 'VALIDATED', result)
                    if offset:
                        self.assertRegex(result[1], r'shared access.*naturally aligned.*\[sat\]')

    def test_genuine_predicated_shared_vector_requires_alignment_only_when_active(self):
        p, s, text = self.assemble(kernel('''
.shared .align 16 .b32 slot[4];
ld.param.u64 %rd0, [O];
mov.u32 %r0, %tid.x;
setp.eq.u32 %p0, %r0, 0;
@%p0 st.shared.v4.u32 [%r0], {0,0,0,0};
st.global.u32 [%rd0], %r0;
''', '.param .u64 O'), 'guarded_vector')
        self.assertRegex(text, r'@!?P\d+\s+STS\.128 ')
        for result in self.validate_both(p, s):
            self.assertEqual(result[0], 'VALIDATED', result)


if __name__ == '__main__':
    verification_main()
