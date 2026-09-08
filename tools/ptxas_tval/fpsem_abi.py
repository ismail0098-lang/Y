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

# The negation vectors.  A sign flip is total and exact; an ARITHMETIC
# implementation of one is entitled to canonicalise a NaN, flush a denormal or
# normalise a signed zero, so these are the patterns that separate the two.
NEG_VECS = [
    0x00000001, 0x007FFFFF, 0x80000001, 0x807FFFFF,   # denormals, both signs
    0x00000000, 0x80000000,                           # +0.0, -0.0
    0x7FC0DEAD, 0x7FC0BEEF, 0xFF800001, 0x7F800001,   # NaNs: quiet w/ payload, sNaN
    0x7F800000, 0xFF800000,                           # +inf, -inf
    0x3F800000, 0xBF800000, 0xFFFFFFFF, 0x7FFFFFFF,   # normals and all-ones
]


def is_nan(b):
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0


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


def negsel_ptx():
    """A `neg.f32` folded into an FSEL SOURCE MODIFIER.

    FSEL is the one float-shaped instruction already refereed as bit-exact, so
    this observes the `-R` modifier with nothing arithmetic in the way.  Every
    other consumer of a negated float source is an FADD/FFMA, which is entitled
    to canonicalise the result and would hide what the modifier itself did."""
    return HEAD + _addrs() + """    ld.global.f32 %f0, [%rd20];
    ld.global.f32 %f1, [%rd21];
    ld.global.u32 %r1, [%rd22];
    neg.f32 %f2, %f0;
    setp.ne.u32 %p0, %r1, 0;
    selp.f32 %f3, %f2, %f1, %p0;
    st.global.f32 [%rd23], %f3;
    st.global.f32 [%rd24], %f0;
}
"""


def negstore_ptx():
    """A `neg.f32` that CANNOT be folded: its result goes straight to a store,
    so ptxas has to materialise the negation as an instruction of its own."""
    return HEAD + _addrs() + """    ld.global.f32 %f0, [%rd20];
    neg.f32 %f1, %f0;
    st.global.f32 [%rd23], %f1;
    st.global.f32 [%rd24], %f0;
}
"""


def subneg_ptx():
    """`a - b` against `a + t`, where t is b with its sign bit flipped by an
    integer XOR against a mask LOADED FROM MEMORY.

    The mask must be opaque.  With a literal 0x80000000 ptxas recognises the xor
    as a negation, folds it back into an operand modifier and CSEs the two arms
    into ONE instruction -- so the probe would compare a kernel against itself
    and answer perfectly.  That was measured, not supposed."""
    return HEAD + _addrs() + """    ld.global.f32 %f0, [%rd20];
    ld.global.f32 %f1, [%rd21];
    ld.global.u32 %r2, [%rd21];
    ld.global.u32 %r4, [%rd22];
    sub.f32 %f2, %f0, %f1;
    xor.b32 %r3, %r2, %r4;
    mov.b32 %f3, %r3;
    add.f32 %f4, %f0, %f3;
    st.global.f32 [%rd23], %f2;
    st.global.f32 [%rd24], %f4;
    st.global.u32 [%rd25], %r3;
}
"""


MAX_PAIRS = [
    # sixteen DISTINCT pairs -- distinctness is the non-vacuity condition, since
    # a swap of two identical operands is not observable.
    (0x7FC0DEAD, 0x00000000), (0x7FC0DEAD, 0x3F800000),   # quiet NaN vs zero, vs a normal
    (0x7FC0DEAD, 0x7FC0BEEF), (0x7F800001, 0x00000000),   # two payloads; signalling NaN
    (0x80000000, 0x00000000), (0x00000001, 0x00000000),   # -0.0 vs +0.0; denormal
    (0x80000001, 0x00000000), (0x807FFFFF, 0x00000001),   # negative denormals
    (0x7F800000, 0x7FC0DEAD), (0xFF800000, 0x00000000),   # infinities against NaN and zero
    (0x7F800000, 0xFF800000), (0x3F800000, 0xBF800000),   # +inf vs -inf; 1 vs -1
    (0x007FFFFF, 0x00000001), (0xFF800001, 0x3F800000),   # denormal ends; negative sNaN
    (0x7FC0DEAD, 0xFF800000), (0xBF800000, 0x00000000),
]


def ptx_max_rule(a, b):
    """PTX ISA `max.f32`: with ONE NaN operand the result is the OTHER operand;
    with two it is a canonical NaN.  That is not an ordering, which is why the
    executor models max as an uninterpreted function rather than as If(a>b,..).
    Signed zeros are compared as an ordering with +0.0 above -0.0, which is the
    part of this that is a MEASUREMENT rather than a reading."""
    if is_nan(a) and is_nan(b): return 0x7FFFFFFF
    if is_nan(a): return b
    if is_nan(b): return a
    def val(x):
        import struct
        return struct.unpack('<f', struct.pack('<I', x))[0]
    fa, fb = val(a), val(b)
    if fa == fb:                       # only reachable for +0.0 vs -0.0
        return a if (a >> 31) == 0 else b
    return a if fa > fb else b


def maxg_ptx():
    """max(a, b) with a and b ALSO stored, so a load/store path that changed the
    bits would be blamed on the instruction rather than on the plumbing."""
    return HEAD + _addrs() + """    ld.global.f32 %f0, [%rd20];
    ld.global.f32 %f1, [%rd21];
    st.global.f32 [%rd24], %f0;
    st.global.f32 [%rd25], %f1;
    max.f32 %f2, %f0, %f1;
    st.global.f32 [%rd23], %f2;
    ret;
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


def check_neg_modifier(d, exe):
    """Is `-R` on a float source a BIT-EXACT sign flip?

    This licenses modelling it as FNEG on the SASS side and `neg.f32` as the
    same FNEG on the PTX side, which is what lets the 23 corpus occurrences --
    every one of them folded into an operand modifier -- meet their PTX."""
    cases = [(v, 0x5EEDBEEF, p) for p in (1, 0) for v in NEG_VECS]
    cases = (cases * 2)[:32]
    if len({p for _, _, p in cases}) != 2:
        print('FAIL: the predicate does not take both values -- the false arm of '
              'the select never runs and the probe observes one source'); return 1

    cub, sass = build(d, 'negsel', negsel_ptx())
    m = re.search(r'^\s*/\*[0-9a-f]+\*/\s+FSEL (R\d+), (-R\d+), (R\w+), (P\d+) ;',
                  sass, re.M)
    if not m:
        print('FAIL: the probe does not emit `FSEL Rd, -Ra, Rb, P` -- ptxas has '
              'stopped folding the negation into the select, so what this measures '
              'is no longer the operand modifier'); return 1
    print(f'  probe emits  FSEL {m.group(1)}, {m.group(2)}, {m.group(3)}, {m.group(4)}')

    bad = 0
    for (v, other, pr), (o, e, _) in zip(cases, run(exe, cub, cases)):
        if e != v:
            print(f'  CONTROL FAILED in={v:#010x}: the load/store path changed the '
                  f'bits (echo={e:#010x})'); bad += 1; continue
        want = (v ^ 0x80000000) if pr else other
        if o != want:
            print(f'  DIFFERS in={v:#010x} pred={pr} device={o:#010x} '
                  f'want={want:#010x}'); bad += 1
    print(f'  {len(cases)} vectors, {bad} disagreements'
          f"{'' if bad else '  -- the modifier is a bit-exact sign flip'}")
    return 1 if bad else 0


def check_neg_unfoldable(d, exe):
    """The un-foldable lowering is NOT a sign flip, and this is a REFUTATION.

    ptxas has no bare float-negate instruction, so a `neg.f32` whose result is
    stored comes back as `FADD Rd, -Rx, -RZ` -- arithmetic, which canonicalises.
    That is why `neg/unfoldable` is a standing UNPROVED row rather than a gap
    somebody should close, and a run in which these agree would mean the
    validator is refusing something it could prove."""
    cases = [(v, 0, 0) for v in NEG_VECS]
    cases = (cases * 3)[:32]
    cub, sass = build(d, 'negstore', negstore_ptx())
    m = re.search(r'^\s*/\*[0-9a-f]+\*/\s+FADD (R\d+), (-R\d+), (-RZ) ;', sass, re.M)
    if not m:
        print('FAIL: the probe does not emit `FADD Rd, -Rx, -RZ` -- ptxas has '
              'changed how it materialises an un-foldable negation, so the '
              'standing UNPROVED row is about something else now'); return 1
    print(f'  probe emits  FADD {m.group(1)}, {m.group(2)}, {m.group(3)}')

    diff = agree = 0
    for (v, _, _), (o, e, _) in zip(cases, run(exe, cub, cases)):
        if e != v:
            print(f'  CONTROL FAILED in={v:#010x} echo={e:#010x}'); return 1
        flip = v ^ 0x80000000
        if o == flip:
            agree += 1
            continue
        diff += 1
        if not is_nan(v):
            print(f'  UNEXPECTED: a NON-NaN input differs.  in={v:#010x} '
                  f'device={o:#010x} signflip={flip:#010x}'); return 1
        if o != 0x7FFFFFFF:
            print(f'  UNEXPECTED: a NaN came back as {o:#010x}, not the canonical '
                  f'0x7fffffff  (in={v:#010x})'); return 1
    # NON-VACUITY, and it is the whole point: a run in which nothing differs
    # says the two ARE the same function, and then the UNPROVED row is a
    # limitation of the model rather than a fact about the machine.
    if diff == 0:
        print('FAIL: no vector separates the lowering from a sign flip -- then '
              'nothing licenses `neg/unfoldable` being UNPROVED'); return 1
    if agree == 0:
        print('FAIL: no vector AGREES -- the probe is measuring something else '
              'entirely, not a negation'); return 1
    print(f'  {len(cases)} vectors: {agree} agree with a sign flip, {diff} do not; '
          f'every disagreement is a NaN and every one canonicalises to 0x7fffffff')
    return 0


def check_sub_is_add_of_neg(d, exe):
    """`sub.f32 a b` against `add.f32 a (b with its sign bit flipped)`.

    This licenses the FSUB(a,b) == FADD(a, FNEG(b)) identification in
    `fpmode.py`, without which no kernel containing a float subtract can
    validate -- eleven in the corpus, because that is what `sub.f32` lowers to."""
    A = 0x3FC00000                                    # 1.5
    OTHER = [A, A, A, A, 0x00000000, 0x80000000, A, A, 0x7F800000, 0x7F800000,
             0x7F800000, 0x7F800000, A, A, A, A]      # inf-inf and 0-0 among them
    cases = [(a, b, 0x80000000) for a, b in zip(OTHER, NEG_VECS)]
    cases = (cases * 2)[:32]

    cub, sass = build(d, 'subneg', subneg_ptx())
    adds = re.findall(r'^\s*/\*[0-9a-f]+\*/\s+FADD (R\d+), (\S+), (\S+) ;', sass, re.M)
    lop = re.search(r'LOP3\.LUT (R\d+),', sass)
    if len(adds) != 2 or lop is None:
        print(f'FAIL: the probe collapsed -- {len(adds)} FADD and '
              f'{"a" if lop else "no"} LOP3.  ptxas folds an xor-by-a-LITERAL '
              f'sign bit back into an operand modifier and then CSEs the two '
              f'arms into one instruction, and the probe compares a kernel '
              f'against itself'); return 1
    if not any(x.startswith('-') for x in adds[0][1:]):
        print('FAIL: the subtract arm has no negated source'); return 1
    if any(x.startswith('-') for x in adds[1][1:]):
        print('FAIL: the xor arm ALSO uses a modifier -- ptxas recognised the '
              'negation, so both arms are the same instruction'); return 1
    print(f'  probe emits  FADD {adds[0][0]}, {adds[0][1]}, {adds[0][2]}   and   '
          f'LOP3.LUT {lop.group(1)} ; FADD {adds[1][0]}, {adds[1][1]}, {adds[1][2]}')

    bad = 0
    for (a, b, _), (sub, addn, xb) in zip(cases, run(exe, cub, cases)):
        if xb != (b ^ 0x80000000):
            print(f'  CONTROL FAILED b={b:#010x}: the xor arm produced '
                  f'{xb:#010x}'); bad += 1; continue
        if sub != addn:
            print(f'  DIFFERS a={a:#010x} b={b:#010x}  a-b={sub:#010x}  '
                  f'a+(-b)={addn:#010x}'); bad += 1
    print(f'  {len(cases)} vectors, {bad} disagreements'
          f"{'' if bad else '  -- subtraction is addition of a negation'}")
    return 1 if bad else 0


def check_max_is_the_ptx_rule(d, exe):
    """Does `max.f32` compute the PTX rule bit for bit?

    This licenses modelling it at all.  A float instruction is entitled to
    flush a denormal or canonicalise a NaN payload, and the NaN rule is not an
    ordering -- so what the executor may assume about FMAX has to be measured,
    not read off the mnemonic.  The denormal vectors are what make the probe
    DISCRIMINATE: a flushing implementation returns +0.0 where this returns the
    denormal, so a clean run is evidence rather than an absence of evidence."""
    cases = [(a, b, 0) for a, b in MAX_PAIRS] + [(b, a, 0) for a, b in MAX_PAIRS]
    cub, sass = build(d, 'maxg', maxg_ptx())
    m = re.search(r'^\s*/\*[0-9a-f]+\*/\s+FMNMX (R\d+), (R\w+), (R\w+), (!?PT) ;',
                  sass, re.M)
    if not m:
        print('FAIL: the probe does not emit `FMNMX Rd, Ra, Rb, !PT` -- what this '
              'measures is no longer the instruction the corpus contains'); return 1
    if m.group(4) != '!PT':
        print(f'FAIL: `max.f32` lowered with polarity {m.group(4)}, not !PT -- the '
              f'executor selects FMIN/FMAX on that operand'); return 1
    print(f'  probe emits  FMNMX {m.group(1)}, {m.group(2)}, {m.group(3)}, {m.group(4)}')
    bad = flushed = 0
    for (a, b, _), (o, ea, eb) in zip(cases, run(exe, cub, cases)):
        if ea != a or eb != b:
            print(f'  CONTROL FAILED a={a:#010x} b={b:#010x}: the load/store path '
                  f'changed the bits'); bad += 1; continue
        want = ptx_max_rule(a, b)
        if o != want:
            print(f'  DIFFERS a={a:#010x} b={b:#010x} device={o:#010x} '
                  f'want={want:#010x}'); bad += 1
        elif (a & 0x7F800000) == 0 and (a & 0x7FFFFF) and o == a:
            flushed += 1      # a denormal survived, so the probe discriminates
    if not flushed:
        print('FAIL: no vector observed a denormal passing through -- this probe '
              'cannot tell a flushing implementation from a bit-exact one'); return 1
    print(f'  {len(cases)} vectors, {bad} disagreements with the PTX rule '
          f'({flushed} of them a denormal surviving unflushed)')
    return 1 if bad else 0


def check_max_commutes(d, exe):
    """Is `max.f32` bit-exactly COMMUTATIVE?

    Needed, and needed for exactly one shape.  The shipped ReLU epilogue writes
    `max.f32 r, r, 0f00000000` and ptxas folds the literal into RZ and puts it
    in the FIRST operand slot -- `FMNMX d, RZ, x` -- so the two sides build
    FMAX(x, +0.0) and FMAX(+0.0, x).  Measured: with the canonicalisation
    removed `max/relu` goes UNPROVED while `max/general`, whose operand order
    ptxas preserves, still validates.

    "IEEE max is commutative" is not enough, for the same reason it was not
    enough for FADD: the claim is about stored BITS, and a hardware returning
    the FIRST operand's NaN payload would break it on precisely the inputs no
    ordinary test uses.  Two of the pairs are quiet NaNs with DIFFERENT
    payloads for that reason.
    """
    cases = [(a, b, 0) for a, b in MAX_PAIRS] + [(b, a, 0) for a, b in MAX_PAIRS]
    if any(a == b for a, b in MAX_PAIRS):
        print('FAIL: a pair has identical operands -- a swap of those is not '
              'observable and the run would agree vacuously'); return 1
    cub, _ = build(d, 'maxg', maxg_ptx())
    res = run(exe, cub, cases)
    n = len(MAX_PAIRS)
    bad = 0
    for i, (a, b) in enumerate(MAX_PAIRS):
        fwd, rev = res[i][0], res[n + i][0]
        if fwd != rev:
            print(f'  ASYMMETRIC a={a:#010x} b={b:#010x} '
                  f'max(a,b)={fwd:#010x} max(b,a)={rev:#010x}'); bad += 1
    print(f'  {n} DISTINCT pairs in both orders, {bad} asymmetric'
          f"{'' if bad else '  -- bit-exactly commutative'}")
    return 1 if bad else 0


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
    print('\nNEG: is the `-R` operand modifier a bit-exact sign flip?')
    rc |= check_neg_modifier(d, exe)
    print('\nNEG: and is the UN-FOLDABLE lowering one too?  (it is not)')
    rc |= check_neg_unfoldable(d, exe)
    print('\nFSUB: is `a - b` the same bits as `a + (-b)`?')
    rc |= check_sub_is_add_of_neg(d, exe)
    print('\nMAX: does `max.f32` compute the PTX rule bit for bit?')
    rc |= check_max_is_the_ptx_rule(d, exe)
    print('\nMAX: and is it bit-exactly commutative?  (the shipped ReLU needs it)')
    rc |= check_max_commutes(d, exe)
    return rc


if __name__ == '__main__':
    sys.exit(main())
