"""Did ptxas UNROLL the loop? -- the FOURTH layer, and the one the frontier
could not cross.

`frontier.py` is this directory's published ranking instrument: for every
corpus kernel it crosses the opcode gap, the structural refusal and the setup
failure, and reports how many blockers stand between the kernel and a verdict.
Its own residue has recorded, for eight increments, that the distance it
reports is a LOWER BOUND -- because `loopval`'s simulation relation holds at
the loop header, so it needs the two loops to run in LOCKSTEP, and if ptxas
unrolled the SASS loop by k then the PTX body has to be composed k times with
a peeled prologue and a remainder cascade.  That layer is measured here and
the frontier crossed it not at all.

THE STRUCTURAL REASON IT WAS NEVER CROSSED, and it is not that nobody tried:
THIS FILE HAD NO `if __name__ == '__main__'` GUARD.  Every line of it was
module-level code that ran a whole-corpus census on import, so `import unroll`
printed a table and no tool could ask it a question.  The layer was not
un-crossed because it was hard; it was un-crossed because the file was a
script wearing a module's name.

THE PROXY, and its premise stated so it can be checked rather than believed.
Count the OBSERVABLE operations in a loop level's own body on each side --
global loads and stores, and the async global-to-shared copies, which are
global accesses under any reading and which the previous proxy could not see
(`\bLDG\b` does not match `LDGSTS`, and `cp.async...global` is not
`ld.global`).  ptxas cannot INVENT a global access, so if the SASS body holds
k times as many as the PTX body, the SASS loop is running the PTX body k
times.  The converse premise -- that ptxas cannot DELETE one -- is FALSE, and
measurably so: hoisting a loop-invariant load out of a body is legal and
common, and four corpus kernels show a level SHRINKING.  So a shrink is a
refusal, not a ratio.

WHAT IT REFUSES RATHER THAN GUESSES, and each of the three was a number this
file used to print:

  a nest it cannot pair   the old code took PTX back edge 0 against SASS back
                          edge 0 whatever the two nests looked like, which is
                          how `bn254_fr_mul` reported an unroll factor of 55
                          from SIX PTX loops against ONE SASS loop.  The two
                          sides must agree on (kind, count, depth) AND on the
                          containment relation before any level is paired.
  a level that shrank     see above.
  PTX 0 against SASS n    ptxas cannot invent one, so this is a wrong pairing
                          rather than an infinite unroll factor.

A level with NO observable operation on EITHER side is VACUOUS, not
undecidable -- `y_cpu_matmul`'s outermost loop is five instructions of counter
and nothing else, and 0 against 0 is consistent with any unroll factor of a
body that does nothing observable.  Reading it as undecidable makes the whole
kernel undecidable, which is what the first version of this rewrite did and
what the ground-truth check below caught.

GROUND TRUTH EXISTS FOR THREE KERNELS AND THE PROXY AGREES WITH IT SIX TIMES
OUT OF SIX.  `exact_pv`, `naive_gemm_f32` and `y_cpu_matmul` are standing
VALIDATED results at `-O1` -- `loopval` has PROVED their PTX and SASS
equivalent, which it can only do if the loops run in lockstep -- and all three
are refused at the `-O3` the corpus ships.  This proxy reports MATCHED for all
three at `-O1` and UNROLLED for all three at `-O3`.  That is a check in BOTH
directions against a verdict derived by a completely different route, and it
is what makes the layer worth crossing into a published ranking.

WHAT IT IS NOT.  It is a proxy, not a validator.  MATCHED means no observable
operation count disagrees; it is not a proof that the loops run in lockstep,
and nothing here composes a body or matches a remainder.  A kernel this file
refuses is one whose unroll status is UNKNOWN, so for that kernel the
frontier's distance remains a lower bound -- which the frontier now says
rather than silently under-counting.
"""
import collections, glob, os, re, sys

import liftgap
import loopcfg

# The observable operations.  Global accesses, and the async global-to-shared
# copy family, which is a global access on both sides and which the previous
# proxy's patterns missed on both sides at once -- consistently, which is why
# 23 kernels reported a 0/0 ratio as `nan` rather than as a disagreement.
PTX_OBS = re.compile(r'^(?:@!?%[\w$]+\s+)?(?:(?:ld|st)\.global\b|cp\.async\.[\w.]*\bglobal\b)')
SASS_OBS = re.compile(r'^(?:@!?P\d+\s+)?(?:LDG|STG|LDGSTS)\b')


def _containment(iv):
    """Which back edge lies inside which -- the relation, not just the shape.

    `nest_shape` folds a nest to (kind, count, depth), and two different nests
    can share one fold: three sequential loops and three sequential loops in
    the other order are both ('SEQUENTIAL', 3, 1).  Pairing by program order is
    only defensible when the two sides agree on this matrix as well."""
    return tuple(tuple(1 if (j != i and iv[i][0] <= iv[j][0] and iv[j][1] <= iv[i][1])
                       else 0 for j in range(len(iv)))
                 for i in range(len(iv)))


def _side(backs, items, key, text, pat):
    """Observable-operation count per level's OWN body.

    The decomposition is `liftgap`'s, imported rather than restated: a level's
    own body is its region minus its immediate children's regions, and a second
    implementation of that is exactly the drift this directory exists to
    prevent."""
    iv = [(b[0], b[1]) for b in backs]
    out = []
    for i in range(len(iv)):
        holes = [(iv[j][0], iv[j][1]) for j in liftgap.children(iv, i)]
        body = liftgap.own(items, key, iv[i][0], iv[i][1], holes)
        out.append(sum(1 for x in body if pat.match(text(x))))
    return out, iv


def levels(kernel, d='corpus'):
    """(ptx counts, ptx intervals, sass counts, sass intervals). Raises to refuse."""
    raw, _lab, pbacks = loopcfg.ptx_back_edges(f'{d}/{kernel}.ptx')
    ins, _l2, _trap, sbacks = loopcfg.sass_back_edges(f'{d}/{kernel}.sass')
    p, piv = _side(pbacks, [(i, t) for i, (k, t) in enumerate(raw) if k == 'i'],
                   lambda x: x[0], lambda x: x[1], PTX_OBS)
    s, siv = _side(sbacks, ins, lambda x: x[0], lambda x: x[1], SASS_OBS)
    return p, piv, s, siv


# The census key for the frontier.  Per-kernel ratios are a property of the
# kernel; the FORM is what a reader has to act on, so it is folded exactly as
# `loopgap.reason_key` folds a back-edge count.
UNROLL_KEY = ('the SASS loop composes the PTX body more than once '
              '(peel-and-remainder matching)')


def factor(kernel, d='corpus'):
    """(verdict, detail).  Verdict is one of:

        None        this kernel has no loop -- the layer does not apply
        MATCHED     every level with something observable in it is 1:1
        UNROLLED    some level's SASS body holds a multiple of the PTX body
        REFUSED     the proxy cannot decide, and says which of its premises failed
        UNDECIDED   nothing observable anywhere on either side
    """
    try:
        p, piv, s, siv = levels(kernel, d)
    except Exception as e:
        return 'REFUSED', str(e).split('\n')[0][:90]
    if not piv or not siv:
        return None, 'no loop'
    ps, ss = loopcfg.nest_shape(piv), loopcfg.nest_shape(siv)
    if ps != ss:
        return 'REFUSED', f'PTX nest {ps} against SASS nest {ss}; these cannot be paired'
    if _containment(piv) != _containment(siv):
        return 'REFUSED', 'the two nests agree on their shape and not on which loop is inside which'
    ratios, vacuous = [], 0
    for i, (a, b) in enumerate(zip(p, s)):
        if a == 0 and b == 0:
            vacuous += 1; ratios.append(None); continue
        if a == 0:
            return 'REFUSED', (f'level {i}: PTX body holds no observable operation and the '
                               f'SASS body holds {b}; ptxas cannot invent one, so the '
                               f'pairing is wrong')
        ratios.append(b / a)
    seen = [r for r in ratios if r is not None]
    shown = '/'.join('vac' if r is None else f'{r:g}' for r in ratios)
    if not seen:
        return 'UNDECIDED', f'all {vacuous} level(s) have no observable operation on either side'
    if any(r < 1 - 1e-9 for r in seen):
        return 'REFUSED', (f'a level SHRANK ({shown}); ptxas may hoist a loop-invariant '
                           f'access out of a body, so this proxy cannot read it as a ratio')
    if any(r > 1 + 1e-9 for r in seen):
        return 'UNROLLED', shown
    return 'MATCHED', shown + (f' ({vacuous} vacuous)' if vacuous else '')


def census(ks=None, d='corpus'):
    """Every kernel's verdict, and the FLOOR.

    A census that paired nothing reports "no kernel is unrolled" perfectly --
    the null metric this repository keeps meeting -- so an empty decided set
    raises rather than returning."""
    if ks is None:
        ks = sorted(os.path.basename(x)[:-4] for x in glob.glob(f'{d}/*.ptx')
                    if os.path.exists(f'{d}/{os.path.basename(x)[:-4]}.sass'))
    rows = {k: factor(k, d) for k in ks}
    decided = [k for k, (v, _) in rows.items() if v in ('MATCHED', 'UNROLLED')]
    if not decided:
        raise Exception(f'decided {len(decided)} of {len(rows)} kernels -- '
                        'there is nothing to report')
    return rows


# The kernels whose unroll status is known INDEPENDENTLY of this proxy, because
# `loopval` has proved them equivalent at -O1 and refuses them at -O3.  A
# simulation relation at the loop header cannot hold unless the loops run in
# lockstep, so a VALIDATED result IS a 1:1 verdict arrived at by another route.
GROUND_TRUTH = (('exact_pv', 'o1', 'MATCHED'), ('exact_pv', 'corpus', 'UNROLLED'),
                ('naive_gemm_f32', 'o1', 'MATCHED'), ('naive_gemm_f32', 'corpus', 'UNROLLED'),
                ('y_cpu_matmul', 'o1', 'MATCHED'), ('y_cpu_matmul', 'corpus', 'UNROLLED'))


def selftest():
    """Controls, and every one of them perturbs the INPUT.

    A control applied to the ANSWER shows the comparison is live and cannot see
    the measurement being subverted to read something else -- found by mutation
    in `docgate.py` and again in `frontier.py`."""
    import shutil, tempfile
    bad = []

    # 1. GROUND TRUTH, both directions.  This is the control that matters: it
    #    checks the proxy against a verdict derived with no shared code.
    for k, d, want in GROUND_TRUTH:
        got, det = factor(k, d)
        if got != want:
            bad.append(f'{k} at {d}: expected {want}, got {got} ({det})')

    # 2. A PERTURBED CORPUS.  Duplicate every observable operation in one
    #    kernel's SASS and the verdict must move MATCHED -> UNROLLED.  Through
    #    `factor`, the same call the census uses.
    tmp = tempfile.mkdtemp(prefix='unroll_ctl_')
    try:
        k = 'rmsnorm_residual_4096'
        if factor(k)[0] != 'MATCHED':
            bad.append(f'control fixture {k} is not MATCHED to start with')
        shutil.copy(f'corpus/{k}.ptx', f'{tmp}/{k}.ptx')
        out = []
        for line in open(f'corpus/{k}.sass'):
            out.append(line)
            if SASS_OBS.match(line.strip().split('*/')[-1].strip().rstrip(';').strip()):
                out.append(line)
        open(f'{tmp}/{k}.sass', 'w').writelines(out)
        v, det = factor(k, tmp)
        if v != 'UNROLLED':
            bad.append(f'a SASS body with every observable operation doubled '
                       f'reported {v} ({det}), not UNROLLED')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 3. A NEST IT CANNOT PAIR is refused rather than given a number.  The
    #    kernel that used to report an unroll factor of 55.
    v, det = factor('bn254_fr_mul')
    if v != 'REFUSED' or 'cannot be paired' not in det:
        bad.append(f'bn254_fr_mul (6 PTX loops against 1 SASS loop) reported {v} ({det})')

    # 4. THE FLOOR fires on a census that decided nothing.
    try:
        census(ks=['bn254_fr_mul'])
        bad.append('the floor did not fire on a census that decided nothing')
    except Exception as e:
        if 'nothing to report' not in str(e):
            bad.append(f'the floor fired with the wrong diagnosis: {e}')

    for b in bad:
        print('FAIL:', b)
    if not bad:
        print('  control: the proxy agrees with six standing verdicts arrived at by '
              'another route, a doubled SASS body reads as UNROLLED, an unpairable '
              'nest is refused rather than numbered, and the floor fires')
    return bad


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(1 if selftest() else 0)
    d = 'o1' if '--o1' in sys.argv else 'corpus'
    rows = census(d=d)
    tally = collections.Counter()
    for k in sorted(rows):
        v, det = rows[k]
        tally['no loop' if v is None else v] += 1
        if v is not None:
            print(f'{k:46s} {v:10s} {det[:78]}')
    print()
    for v, n in tally.most_common():
        print(f'{n:4d}  {v}')
