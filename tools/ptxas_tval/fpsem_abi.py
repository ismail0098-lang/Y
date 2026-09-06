"""Referee two float-semantics facts against the DEVICE, not against a manual.

Both are facts the executors NEED and neither can be read off a mnemonic.

  FSEL is a bit-exact select, not an arithmetic operation.
  An f32 add is bit-exactly commutative.

WHY EACH IS A MEASUREMENT AND NOT A READING.

  FSEL.  `naive_gemm_f32` at -O1 is past every structural gate in `loopval` and
  refuses on exactly one unmodelled opcode.  The guess that matters is not the
  operand order -- that is visible in the disassembly -- but whether FSEL is a
  SELECT or ARITHMETIC.  A float op is entitled to flush a denormal,
  canonicalise a NaN payload or normalise a signed zero; a select is not, and
  modelling a flushing instruction as a pure select would be invisible on
  ordinary data.  So the vectors are the patterns a flushing implementation
  changes.

  FADD.  `fpmode.py` used to record commutativity as deliberately NOT assumed:
  "IEEE addition is commutative, so canonicalising by operand id would be sound
  and would hide a real question -- whether ptxas preserves operand order -- so
  it is left out until something needs it."  Something needs it: with the
  contraction repaired, `naive_gemm_f32` is one operand order from a result,
  and ptxas does NOT preserve the order -- it SORTS the addends by register
  number.  "Sound in IEEE" is still not enough, because the claim is about
  stored BITS and IEEE leaves a NaN result's payload implementation defined: a
  hardware returning the FIRST operand's payload would break this on exactly
  the inputs no ordinary test uses.

TWO THINGS PTXAS DOES THAT HAD TO BE DESIGNED AROUND.

  It CSEs `a+b` with `b+a` inside one kernel, so the obvious commutativity
  probe is answered by the translator under test rather than by the device.  B
  is therefore passed through TWO pointers that carry the same values; ptxas
  cannot merge loads it cannot prove equal.

  It then SORTS the addends anyway, so both orders cannot be had from one
  kernel.  The LOAD ORDER is varied between two otherwise identical kernels,
  which changes the register numbering and so changes which slot each value
  lands in -- and the checker traces each FADD source back through its load to
  the PARAMETER it came from, because comparing register NAMES would pass
  vacuously when two cubins differ in numbering while putting the same value in
  the same slot.

Run:  python3 fpsem_abi.py        (needs ptxas, nvdisasm, gcc, a GPU)
"""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__)) or '.'
ARCH = 'sm_89'

# Patterns a float operation is entitled to change and a bit-exact one is not.
FSEL_PAIRS = [
    (0x00000001, 0x007FFFFF),   # smallest denormal     vs largest denormal
    (0x80000001, 0x807FFFFF),   # negative denormals
    (0x00000000, 0x80000000),   # +0.0                  vs -0.0
    (0x7FC0DEAD, 0x7F800001),   # quiet NaN w/ payload  vs signalling NaN
    (0x7F800000, 0xFF800000),   # +inf                  vs -inf
    (0x3F800000, 0xBF800000),   # +1.0                  vs -1.0
    (0xFFFFFFFF, 0x00800000),   # all ones              vs smallest normal
    (0x12345678, 0x9ABCDEF0),   # arbitrary
]
FADD_PAIRS = [
    (0x7FC0DEAD, 0x7FC0BEEF),   # two quiet NaNs with DIFFERENT payloads
    (0x7FC0DEAD, 0x3F800000),   # quiet NaN payload + 1.0
    (0x3F800000, 0x7FC0DEAD),   # and the other way round
    (0x7F800001, 0x7FC0BEEF),   # signalling NaN    + quiet NaN
    (0x7F800000, 0xFF800000),   # +inf + -inf  -> a NaN the hardware invents
    (0x00000000, 0x80000000),   # +0.0 + -0.0
    (0x00000001, 0x80000001),   # smallest denormals of each sign
    (0x3F800000, 0x33800000),   # 1.0 + 2^-24  (a rounding tie)
]

HEAD = """.version 7.8
.target %s
.address_size 64
.visible .entry probe(
    .param .u64 P0, .param .u64 P1, .param .u64 P2,
    .param .u64 P3, .param .u64 P4, .param .u64 P5
)
{
    .reg .b32 %%r<8>;
    .reg .f32 %%f<8>;
    .reg .b64 %%rd<32>;
    .reg .pred %%p<4>;
""" % ARCH

# The parameter's const-bank offset, for tracing a source back to a parameter.
PARAM_OFF = {0x160: 'P0', 0x168: 'P1', 0x170: 'P2',
             0x178: 'P3', 0x180: 'P4', 0x188: 'P5'}


def _addrs():
    s = ''
    for i in range(6):
        s += f'    ld.param.u64 %rd{i}, [P{i}];\n'
    s += ('    mov.u32 %r0, %tid.x;\n'
          '    cvt.u64.u32 %rd10, %r0;\n'
          '    shl.b64 %rd11, %rd10, 2;\n')
    for i in range(6):
        s += f'    add.u64 %rd{20+i}, %rd{i}, %rd11;\n'
    return s


def fsel_ptx():
    """out = p ? a : b, with a and b ALSO stored so ptxas cannot rewrite the
    select into a predicated load -- which is what it does when only the
    selected value is used, and which would make the probe observe one source."""
    return HEAD + _addrs() + """    ld.global.f32 %f0, [%rd20];
    ld.global.f32 %f1, [%rd21];
    ld.global.u32 %r1, [%rd22];
    st.global.f32 [%rd24], %f0;
    st.global.f32 [%rd25], %f1;
    setp.ne.u32 %p0, %r1, 0;
    selp.f32 %f2, %f0, %f1, %p0;
    st.global.f32 [%rd23], %f2;
}
"""


def fadd_ptx(swap):
    """X = a + b1 and Y = b2 + a, where b1 and b2 arrive through two pointers
    carrying the same values.  `swap` changes only the LOAD ORDER."""
    loads = ['    ld.global.f32 %f0, [%rd20];\n',
             '    ld.global.f32 %f1, [%rd21];\n',
             '    ld.global.f32 %f4, [%rd22];\n']
    if swap: loads = loads[1:] + loads[:1]
    return HEAD + _addrs() + ''.join(loads) + """    add.rn.f32 %f2, %f0, %f1;
    add.rn.f32 %f3, %f4, %f0;
    st.global.f32 [%rd23], %f2;
    st.global.f32 [%rd24], %f3;
}
"""


def build(d, tag, ptx):
    p, cub = f'{d}/{tag}.ptx', f'{d}/{tag}.cubin'
    open(p, 'w').write(ptx)
    subprocess.run(['ptxas', '-O1', f'-arch={ARCH}', '-o', cub, p], check=True)
    sass = subprocess.run(['nvdisasm', '-c', cub], capture_output=True,
                          text=True, check=True).stdout
    return cub, sass[sass.index('probe:'):]


def run(exe, cub, cases):
    stdin = '\n'.join(f'{a} {b} {c}' for a, b, c in cases)
    r = subprocess.run([exe, cub], input=stdin, capture_output=True, text=True)
    if r.returncode != 0 or 'FAIL' in r.stdout:
        raise SystemExit('FAIL: launch: ' + (r.stdout.strip() or r.stderr.strip()))
    return [tuple(int(x) for x in ln.split()) for ln in r.stdout.split('\n') if ln.strip()]


def fadd_operand_params(sass):
    """Which PARAMETER feeds each source slot of the first FADD.

    Register names are not enough and this is the measurement's weak point:
    two cubins can differ in numbering while putting the same VALUE in the same
    slot, and then the two runs execute the same instruction and agreeing says
    nothing at all.  So walk back: source register -> the LDG that wrote it ->
    the IADD3 that built its address -> the const-bank offset -> the parameter.
    """
    m = re.search(r'FADD (R\d+), (R\d+)(?:\.reuse)?, (R\d+)(?:\.reuse)? ;', sass)
    if not m: return None
    upto = sass[:m.start()]
    ld   = dict(re.findall(r'LDG\.E (R\d+), \[(R\d+)\.64\] ;', upto))
    base = dict(re.findall(r'IADD3 (R\d+), P\d+, R\d+, c\[0x0\]\[(0x[0-9a-f]+)\], RZ ;', upto))
    out = []
    for r in (m.group(2), m.group(3)):
        a = ld.get(r)
        out.append(PARAM_OFF.get(int(base.get(a, '-1'), 0), '?') if a else '?')
    return tuple(out)


def check_fsel(d, exe):
    cases = [(a, b, 1) for a, b in FSEL_PAIRS] + [(a, b, 0) for a, b in FSEL_PAIRS]
    cases = (cases * 2)[:32]
    if any(a == b for a, b, _ in cases):
        print('FAIL: a case has a == b -- a swapped-source model would pass'); return 1
    if len({p for _, _, p in cases}) != 2:
        print('FAIL: the predicate does not take both values'); return 1

    cub, sass = build(d, 'fsel', fsel_ptx())
    m = re.search(r'^\s*/\*[0-9a-f]+\*/\s+FSEL (R\d+), (R\d+), (R\w+), (P\d+) ;', sass, re.M)
    if not m:
        print('FAIL: no UNPREDICATED FSEL in the probe -- a predicated one never '
              'executes its false arm, so the probe would observe one source only')
        return 1
    if m.group(2) == m.group(3):
        print('FAIL: the two FSEL sources are the same register'); return 1
    print(f'  probe emits  FSEL {m.group(1)}, {m.group(2)}, {m.group(3)}, {m.group(4)}')

    bad = 0
    for (a, b, p), (o, ea, eb) in zip(cases, run(exe, cub, cases)):
        # THE CONTROL: if the load/store path were itself lossy on these
        # patterns, a difference at the output would be blamed on FSEL.
        if ea != a or eb != b:
            print(f'  CONTROL FAILED a={a:#010x} b={b:#010x}: the load/store path '
                  f'changed the bits (ea={ea:#010x} eb={eb:#010x})')
            bad += 1; continue
        want = a if p else b
        if o != want:
            print(f'  p={p} a={a:#010x} b={b:#010x} -> {o:#010x}, a bit-exact '
                  f'select gives {want:#010x}')
            bad += 1
    if bad:
        print(f'FSEL is NOT a bit-exact select: {bad}/{len(cases)} cases differ.')
        return 1
    print(f'  FSEL d, s0, s1, p  ==  p ? s0 : s1, bit for bit, on all {len(cases)} '
          f'cases (denormals, both zeros, NaN payloads, a signalling NaN, both infinities)')
    return 0


def check_fadd_commutes(d, exe):
    cases = [(a, b, b) for a, b in FADD_PAIRS]          # P1 and P2 carry the same values
    cases = (cases * 4)[:32]
    c1, s1 = build(d, 'fadd', fadd_ptx(False))
    c2, s2 = build(d, 'fadd_swap', fadd_ptx(True))
    o1, o2 = fadd_operand_params(s1), fadd_operand_params(s2)
    print(f'  probe      first FADD sources: {o1}')
    print(f'  probe_swap first FADD sources: {o2}')
    if o1 is None or o2 is None or '?' in o1 + o2:
        print('FAIL: could not trace a FADD source back to a parameter'); return 1
    # NON-VACUITY: same two operands, opposite slots.  Either half missing and
    # the comparison below is a kernel against itself.
    if o1 == o2:
        print('FAIL: both cubins use the same operand order'); return 1
    if set(o1) != set(o2):
        print(f'FAIL: the two probes add different operands ({o1} vs {o2})'); return 1

    r1, r2 = run(exe, c1, cases), run(exe, c2, cases)
    bad = 0
    for (a, b, _), (x1, _, _), (x2, _, _) in zip(cases, r1, r2):
        if x1 != x2:
            print(f'  a={a:#010x} b={b:#010x}: {x1:#010x} vs {x2:#010x}  DIFFER')
            bad += 1
    if bad:
        print(f'f32 add is NOT bit-exactly commutative here: {bad}/{len(cases)} '
              f'differ.  Canonicalising operand order would be UNSOUND.')
        return 1
    print(f'  f32 add is bit-exactly commutative on {ARCH}: all {len(cases)} cases '
          f'agree (two quiet NaNs with different payloads, a signalling NaN, '
          f'inf+(-inf), both zeros, denormals)')
    return 0


def main():
    os.chdir(HERE)
    inc = next((x for x in ('/opt/cuda/include', '/usr/local/cuda/include')
                if os.path.exists(x + '/cuda.h')), None)
    if inc is None:
        print('SKIP: no cuda.h -- neither fact can be refereed without the device')
        return 0
    d = tempfile.mkdtemp(prefix=f'fpsem_{os.getpid()}_')
    exe = f'{d}/drv'
    subprocess.run(['gcc', '-O1', '-o', exe, 'fpsem_abi.c', '-I', inc, '-lcuda'],
                   check=True)
    print('FSEL: is it a select or an arithmetic operation?')
    rc = check_fsel(d, exe)
    print('\nFADD: is it bit-exactly commutative?')
    rc |= check_fadd_commutes(d, exe)
    return rc


if __name__ == '__main__':
    sys.exit(main())
