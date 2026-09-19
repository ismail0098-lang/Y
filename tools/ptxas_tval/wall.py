r"""THE FIFTH LAYER: the solver wall, as a PROXY the frontier can cross.

WHY IT EXISTS.  The frontier called nine kernels at -O3 CLEAR and five of them
validate.  The other four -- `bn254_fr_mul_fast`, `bn254_ntt4_fused`,
`bn254_g1_dbl`, `bn254_g1_add` -- have no unmodelled opcode, no structural
refusal and no unroll blocker, and are past the solver: `bn254_fr_mul_fast` is
UNPROVED with no `sat` (first sweep: 17 of 276 partial sums in 16,237 s), and the
exact Int translation does not help it (`intwall.py`).  So "clear" was an UPPER bound on what
validates, and any ranking built on it -- a joint-sufficiency measure above all
-- inherited that.  Measured before building this: the best six-blocker set the
frontier offered (the integer-division lowering) clears six kernels, and five of
the six are past the wall.  Reach has ranked work that buys nothing four times;
this is the same error one layer later, caught before it was published.

WHAT IT MEASURES.  The worst BARRIER REGION's multiply count, on the PTX side
(`barregion.muls`), because a barrier is a legitimate cut point and a region is
what one solver query sees.  PTX rather than SASS because it does not move with
the optimisation level, so the layer answers identically at -O1 and -O3.

THE THRESHOLDS ARE DERIVED FROM NAMED GROUND TRUTH, NOT WRITTEN DOWN.  The old
figure, "a wall between 29 and 65 multiplies", was a KERNEL-level reading and it
is not the bracket the measurements give.  `bn254_ntt4_fused` has six barrier
regions of 33, 49, 193, 193, 193 and 225 PTX multiplies; its barrier 0 was
PROVED and barrier 1 was `unknown` -- so at region level the bracket is 33 / 49.
Transcribed it would be a fourth copy of a number; here it is the maximum over
PROVED regions and the minimum over UNKNOWN regions of `GROUND_TRUTH`, and the
selftest re-counts every one of those regions from the artifact.

WHAT IT IS NOT.  A validator, or a claim about solver behaviour in general.
Multiplies are the variable the representation ladder is about, and the one the
recorded measurements vary; two regions with equal counts can differ in cost.
So there are three answers, and the middle one is the point:

    UNDER      worst region <= the largest region measured PROVED
    PAST       worst region >= the smallest region measured UNKNOWN
    UNDECIDED  between them -- nothing measured says which side it is on

A frontier blocker is added ONLY for PAST.  UNDECIDED is the instrument not
answering, and recording it as a blocker would be a guess in the pessimistic
direction -- so it is named as a lower bound instead, as `unroll.py`'s refusals
are.  A module with two entry points is REFUSED: `barregion` would count both
entries as one program, which is a different program.

`fma.` IS COUNTED as a multiply.  It was not, which under-counted 11 corpus
kernels in the optimistic direction; see `barregion.muls`.
"""
import os, re, sys
import barregion

# (kernel, barrier region, verdict, where it was measured).  Region-level
# because that is the size of a solver query; a whole-kernel VALIDATED is a
# PROVED for every region of that kernel, so it is listed as each region.
GROUND_TRUTH = (
    ('ptx_carry_chain',   0, 'PROVED',  'regress.sh standing row, VALIDATED'),
    ('bn254_ntt4_fused',  0, 'PROVED',  'smemval, shared memory entering barrier 0 PROVED EQUAL'),
    ('bn254_ntt4_fused',  1, 'UNKNOWN', 'smemval, barrier 1 solver said unknown at a 600 s budget'),
    ('bn254_fr_mul_fast', 0, 'UNKNOWN', 'tval, UNPROVED, no sat; first sweep 17/276 in 16,237 s (2026-09-19)'),
)

WALL_KEY = ('a barrier region has at least as many multiplies as the smallest '
            'region measured unknown (solver wall)')


def region_muls(kernel, d='corpus'):
    """Multiplies per barrier region of the PTX, or None for a module with more
    than one entry point (counting both as one program would be wrong)."""
    p = os.path.join(d, kernel + '.ptx')
    # `.visible .entry` has a SECOND dot before `entry`.  The first version of this
    # pattern was `\.(?:visible\s+)?entry`, which matches neither spelling, so the
    # two split paged-decode modules were counted as one program and read PAST.
    # The synthetic two-entry control below is what said so.
    if len(re.findall(r'^\s*(?:\.visible\s+)?\.entry\b', open(p).read(), re.M)) > 1:
        return None
    return [(barregion.muls([o for o, _t in r], True), sum(1 for o, t in r if _is_sym_int_mul(o, t)))
            for r in barregion.regions(p, True, text=True)]


def _is_sym_int_mul(op, insn):
    """A SYMBOLIC INTEGER multiply: integer, and both FACTORS are registers.

    The only kind the ground truth measures.  Every multiply in every region of
    `GROUND_TRUTH` is register times register, while the f16 GEMMs' integer
    multiplies are almost all index arithmetic by an immediate (`mul.lo.u32 R,
    R, 4`, `mad.lo.u32 R, R, 136, R`), which is LINEAR for a solver.  Counting
    those put 15 GEMMs PAST on a count of work the wall was never measured on.
    A float multiply is excluded for the same reason: no float region has been
    measured at the wall in either direction.

    Factors are operands 2 and 3 (`mul d, a, b` / `mad d, a, b, c`, where `c` is
    the addend and may be an immediate without making the product linear)."""
    if not op.startswith(('mul.', 'mad.')):
        return False
    if re.search(r'\.f(?:16|32|64)\b|e4m3|e5m2', op) or \
            not re.search(r'\.(?:lo|hi|wide|[us](?:8|16|32|64))\b', op):
        return False
    ops = [o.strip() for o in insn.split(None, 1)[1].split(',')] if ' ' in insn else []
    return len(ops) >= 3 and all(o.startswith('%') for o in ops[1:3])


def _thresholds(gt=GROUND_TRUTH, d='corpus'):
    proved, unknown = [], []
    for k, i, v, _src in gt:
        m = region_muls(k, d)
        if m is None or i >= len(m):
            raise Exception(f'ground truth names {k} region {i}, which the artifact does not have')
        if m[i][0] != m[i][1]:
            raise Exception(f'ground truth {k} region {i} has a multiply that is not symbolic integer; '
                            'the thresholds would then be about a different kind of work')
        (proved if v == 'PROVED' else unknown).append(m[i][0])
    if not proved or not unknown:
        raise Exception('ground truth needs at least one PROVED and one UNKNOWN region')
    return max(proved), min(unknown)


_here = os.path.dirname(os.path.abspath(__file__))
_cwd = os.getcwd()
try:
    os.chdir(_here)
    UNDER_AT, PAST_AT = _thresholds()
finally:
    os.chdir(_cwd)
if UNDER_AT >= PAST_AT:
    # The evidence would contradict itself: a region at least as large as one the
    # solver could not decide was proved.  Then the proxy has no meaning.
    raise Exception(f'the wall ground truth is inconsistent: PROVED up to {UNDER_AT}, '
                    f'UNKNOWN from {PAST_AT}')


def verdict(kernel, d='corpus'):
    m = region_muls(kernel, d)
    if m is None:
        return 'REFUSED', 'more than one entry point; barregion would count them as one program'
    w = max((t for t, _i in m), default=0)
    wi = max((i for _t, i in m), default=0)
    if wi >= PAST_AT:
        return 'PAST', f'worst region has {wi} symbolic integer multiplies >= {PAST_AT}'
    if w <= UNDER_AT:
        return 'UNDER', f'worst region {w} <= {UNDER_AT}'
    if w >= PAST_AT:
        return 'UNDECIDED', (f'worst region {w} multiplies but only {wi} symbolic integer; '
                             'nothing of that shape has been measured at the wall')
    return 'UNDECIDED', f'worst region {w} is between {UNDER_AT} (proved) and {PAST_AT} (unknown)'


def regress_validated():
    """Every corpus kernel `regress.sh` asserts VALIDATED.

    Read from the script rather than listed, so the calibration control below
    cannot drift from the standing results.  Every `corpus/` and `smut/` row
    and `o1/` row naming a corpus kernel is asserted in the VALIDATED direction
    (the UNPROVED `o1/` twins carry suffixes no corpus kernel has, and the proxy
    reads PTX only, so the level does not matter); a `smut/` row counts only if
    its PTX is byte-identical to the corpus kernel of the same name."""
    s = open(os.path.join(_here, 'regress.sh')).read()
    out = set()
    for dirn, k in re.findall(r'\b(corpus|smut|o1)/(\w+)', s):
        c = os.path.join(_here, 'corpus', k + '.ptx')
        if not os.path.exists(c):
            continue
        if dirn == 'smut' and open(os.path.join(_here, 'smut', k + '.ptx'), 'rb').read() != open(c, 'rb').read():
            continue
        out.add(k)
    return out


def selftest():
    import tempfile, shutil
    bad = 0
    # 1. GROUND TRUTH IS CONSISTENT WITH ITSELF, re-counted from the artifacts.
    print(f'  thresholds: UNDER <= {UNDER_AT} (largest region PROVED), '
          f'PAST >= {PAST_AT} (smallest region UNKNOWN)')
    for k, i, v, src in GROUND_TRUTH:
        got = verdict(k)[0]
        if v == 'UNKNOWN' and got != 'PAST':
            print(f'FAIL: {k} has a region measured UNKNOWN ({src}) and the proxy says {got}')
            bad += 1
    # 2. CALIBRATION.  Every kernel the standing results assert VALIDATED must be
    # UNDER.  This is ground truth reached by another route and it is what makes
    # the layer worth crossing; a threshold that blocked a validated kernel is
    # simply wrong.  Non-vacuous by the floor.
    val = regress_validated()
    if len(val) < 4:
        print(f'FAIL: read only {sorted(val)} as validated from regress.sh -- the parse is not reading it')
        bad += 1
    wrong = {k: verdict(k) for k in sorted(val) if verdict(k)[0] != 'UNDER'}
    if wrong:
        print(f'FAIL: regress.sh asserts these VALIDATED and the wall calls them otherwise: {wrong}')
        bad += 1
    else:
        print(f'  control: all {len(val)} regress-validated corpus kernels are UNDER: {sorted(val)}')
    # 3. THE PROXY READS ITS INPUT: a synthetic kernel at PAST_AT multiplies is PAST,
    # the same kernel cut by a barrier into two halves is not, and one short of
    # UNDER_AT+1 is UNDER.  Perturbing the INPUT through `verdict` itself.
    d = tempfile.mkdtemp(prefix='wall_ctl_')
    try:
        def put(name, body):
            open(os.path.join(d, name + '.ptx'), 'w').write(
                '.version 7.0\n.target sm_80\n.visible .entry k(.param .u32 x) {\n'
                + ''.join('\t' + l + '\n' for l in body) + '\tret;\n}\n')
        mul = 'mul.lo.u32 %r1, %r1, %r1;'
        put('past', [mul] * PAST_AT)
        put('cut', [mul] * UNDER_AT + ['bar.sync 0;'] + [mul] * (PAST_AT - UNDER_AT))
        put('under', [mul] * UNDER_AT)
        put('fused', ['fma.rn.f32 %f1, %f1, %f1, %f1;'] * PAST_AT)
        put('fbig', ['mul.f32 %f1, %f1, %f1;'] * PAST_AT)
        put('konst', ['mad.lo.u32 %r1, %r1, 136, %r2;'] * PAST_AT)
        put('addend', ['mad.lo.u32 %r1, %r1, %r2, 48;'] * PAST_AT)
        put('two', [mul] + ['}', '.visible .entry k2(.param .u32 x) {', mul])
        want = {'past': 'PAST', 'under': 'UNDER', 'fused': 'UNDECIDED', 'fbig': 'UNDECIDED',
                'konst': 'UNDECIDED', 'addend': 'PAST',
                'two': 'REFUSED'}
        for n, w in want.items():
            if verdict(n, d)[0] != w:
                print(f'FAIL: synthetic `{n}` should be {w}, the proxy says {verdict(n, d)}')
                bad += 1
        if verdict('cut', d)[0] == 'PAST' or verdict('past', d)[0] != 'PAST':
            print(f'FAIL: a barrier cutting {PAST_AT} multiplies in two still reads PAST; '
                  'regions are not being split at bar.sync')
            bad += 1
        # `fma.` IS counted -- it is what moves a float region out of UNDER -- and
        # it is not an integer multiply.
        if region_muls('fused', d)[0] != (PAST_AT, 0):
            print(f'FAIL: `fma.` counted as {region_muls("fused", d)[0]}, want ({PAST_AT}, 0)')
            bad += 1
        if not bad:
            print('  control: synthetic past/under/float/two-entry kernels answer as built, '
                  'and a bar.sync cut takes a PAST kernel out of PAST')
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return bad


if __name__ == '__main__':
    if '--selftest' in sys.argv[1:]:
        sys.exit(selftest())
    import glob
    ks = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob('corpus/*.ptx'))
    for k in ks:
        print(f'{k:48s} {verdict(k)[0]:10s} {verdict(k)[1]}')
