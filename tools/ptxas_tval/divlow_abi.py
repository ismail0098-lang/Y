#!/usr/bin/env python3
"""The integer-division lowering, refereed on the device.  One consumer for
three probes, so none of them is a referee nothing reads.

ptxas lowers a u32 `div`/`rem` to a float reciprocal estimate, one Newton step
and two conditional corrections:

    e  = F2I.FTZ.U32.TRUNC.NTZ( MUFU.RCP( I2F.U32.RP(d) ) + 0x0ffffffe )
    e2 = e + HI(e * (-d*e))          # IMAD.HI.U32 with the 64-bit PAIR addend
    q0 = HI(e2 * n); r0 = n - d*q0;  two corrections  ->  (q, r)

The float half cannot be modelled in bitvectors, and pattern-matching the whole
sequence as `UDiv` would ASSUME the lowering correct -- the thing under
validation.  What can be done instead is to measure, over the whole finite
domain, the facts a validator would have to assume, so that it assumes nothing
the silicon has not been shown to do.  Three probes, each exhaustive where the
domain allows it:

  rcpwin_abi.cu  the WINDOW the estimate lands in, all 2^32-1 divisors:
                 e <= I = floor(2^32/d), and (I-e)^2 <= I.  Both ATTAINED.
  newton_abi.cu  LEMMA A, all 2^32-1 divisors: the Newton step turns that window
                 into a bound ONE UNIT WIDE, I-1 <= e2 <= I.  The deficit of 1 is
                 ATTAINED, so B = 0 is false and B = 1 is the tight statement.
  tail_abi.cu    LEMMA B, the corrections, at BOTH estimates Lemma A admits
                 (the validator will not know which): every d over a structured
                 n set, plus twelve d over EVERY n.  The domain is 2^64 so this
                 is a strong sample, not exhaustion, and it says so.

WHY LEMMA A MATTERS MORE THAN THE WINDOW.  Measuring only the window leaves the
solver two composed 32x32 multiplies before the tail even starts; measuring
through the Newton step leaves it one.  Exhaust as far up the chain as the
domain stays finite.

WHAT THIS DOES NOT DO: discharge the tail.  With `d` symbolic, z3 answered
`unknown` -- never `sat` -- on six posings of it (with UDiv, division-free, I
eliminated, and the three split lemmas), at budgets up to 1200 s.  Lemma B is
true on every case measured, so that is a SOLVER limit, not a false lemma.  See
docs/ptxas_translation_validation.md.

Only est() touches an ISA fact; e2, q0 and the corrections are integer
arithmetic the device merely computes fast.  So the instruction-shape check is
on est(), in BOTH probes that use it, against the corpus kernel's own SASS.

Needs nvcc, cuobjdump and a CUDA device.  Exits non-zero on any disagreement.
"""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_SASS = os.path.join(HERE, 'corpus', 'ptx_integer_ops.sass')
U32_NONZERO = (1 << 32) - 1

# The four instructions of the estimate, as the corpus kernel spells them.  A
# probe emitting F2I.U32.TRUNC.NTZ (no .FTZ) was the first version's defect:
# modelling a flushing instruction as a non-flushing one is the FSEL lesson.
EST_FORMS = ('I2F.U32.RP', 'MUFU.RCP', 'F2I.FTZ.U32.TRUNC.NTZ')
EST_ADD = re.compile(r'\bIADD3\s[^;]*0xffffffe\b')


def fail(msg):
    print('FAIL:', msg)
    return 1


def build(src, out):
    r = subprocess.run(['nvcc', '-O2', '-arch=sm_89', '-o', out, os.path.join(HERE, src)],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f'FAIL: nvcc {src}\n{r.stderr}')


def sass_of(binary):
    return subprocess.run(['cuobjdump', '-sass', binary], capture_output=True,
                          text=True, check=True).stdout


def check_shape(tag, sass, corpus):
    bad = 0
    for f in EST_FORMS:
        if f not in corpus:
            bad += fail(f'the corpus kernel no longer contains {f}; the probe is refereeing '
                        f'a lowering ptxas does not emit')
        if f not in sass:
            bad += fail(f'{tag}: probe SASS lacks {f}, so it measures a different instruction')
    if not EST_ADD.search(corpus):
        bad += fail('the corpus kernel no longer adds 0x0ffffffe to the reciprocal bits')
    if not EST_ADD.search(sass):
        bad += fail(f'{tag}: probe SASS lacks the +0x0ffffffe bit-pattern add')
    if not bad:
        print(f'ok   {tag}: est() is the corpus kernel\'s four instructions')
    return bad


def run(binary, *args):
    return subprocess.run([binary, *args], capture_output=True, text=True, check=True).stdout


def fields(line, *names):
    out = {}
    for n in names:
        m = re.search(re.escape(n) + r'\s+([0-9.]+)', line)
        if not m:
            raise SystemExit(f'FAIL: cannot read {n!r} from {line!r}')
        out[n] = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
    return out


def main():
    if not os.path.exists(CORPUS_SASS):
        return fail(f'{CORPUS_SASS} missing -- run build_corpus.sh first')
    corpus = open(CORPUS_SASS).read()
    bad = 0
    with tempfile.TemporaryDirectory(prefix='divlow_') as t:
        exe = {k: os.path.join(t, k) for k in ('rcpwin', 'newton', 'tail')}
        build('rcpwin_abi.cu', exe['rcpwin'])
        build('newton_abi.cu', exe['newton'])
        build('tail_abi.cu', exe['tail'])
        bad += check_shape('rcpwin', sass_of(exe['rcpwin']), corpus)
        bad += check_shape('newton', sass_of(exe['newton']), corpus)

        # --- the window --------------------------------------------------------
        w = fields(run(exe['rcpwin'], '1'), 'seen', 'maxslack', 'above', 'badslack', 'maxsratio')
        print(f'     window: {w}')
        if w['seen'] != U32_NONZERO:
            bad += fail(f'window examined {w["seen"]} divisors, not {U32_NONZERO}')
        if w['above']:
            bad += fail(f'{w["above"]} divisors have an estimate ABOVE floor(2^32/d)')
        if w['badslack']:
            bad += fail(f'{w["badslack"]} divisors violate (I-e)^2 <= I')
        # ATTAINED, or the bound is looser than what is true and a validator
        # assuming it would be assuming less than the silicon guarantees -- and
        # a probe computing nothing would report a slack of 0 here.
        if w['maxslack'] != 512:
            bad += fail(f'max absolute slack is {w["maxslack"]}, measured 512 before')
        if abs(w['maxsratio'] - 1.0) > 1e-9:
            bad += fail(f'max (I-e)^2/I is {w["maxsratio"]}, not the attained 1.0')

        # --- Lemma A -----------------------------------------------------------
        a = fields(run(exe['newton']), 'seen', 'maxdeficit(I-e2)', 'above(e2>I)')
        print(f'     lemma A: {a}')
        if a['seen'] != U32_NONZERO:
            bad += fail(f'Lemma A examined {a["seen"]} divisors, not {U32_NONZERO}')
        if a['above(e2>I)']:
            bad += fail(f'{a["above(e2>I)"]} divisors have e2 ABOVE floor(2^32/d)')
        if a['maxdeficit(I-e2)'] != 1:
            bad += fail(f'max I-e2 is {a["maxdeficit(I-e2)"]}; Lemma A was measured ONE unit '
                        f'wide and attained (0 would mean the Newton step is exact, >1 that '
                        f'the stated bound is false)')

        # --- Lemma B -----------------------------------------------------------
        lines = run(exe['tail']).strip().splitlines()
        if len(lines) != 2:
            return fail(f'tail probe printed {len(lines)} lines: {lines}')
        for ln in lines:
            nums = [int(x) for x in re.findall(r'seen (\d+) bad (\d+)', ln) for x in x]
            if len(nums) != 4:
                bad += fail(f'cannot read the tail line {ln!r}')
                continue
            s0, b0, s1, b1 = nums
            print(f'     lemma B: {ln}')
            # A floor per estimate: the I-1 arm is the one a lowering could fail
            # on, and a probe that skipped it would report zero failures.
            if s0 < 10**9 or s1 < 10**9:
                bad += fail(f'tail examined too little: {s0} / {s1}')
            if b0 or b1:
                bad += fail(f'the corrections are WRONG at an estimate Lemma A admits: '
                            f'{b0} at e2=I, {b1} at e2=I-1')
    if not bad:
        print('ok   the division lowering\'s window, Lemma A and Lemma B hold as measured')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
