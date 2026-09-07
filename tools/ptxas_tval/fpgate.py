"""Does repairing the contraction unlock any kernel TODAY -- and which repair?

THIS FILE USED TO ANSWER "0 / 9" AND BOTH HALVES WERE WRONG.

  The 9 was a HARDCODED list, and it disagreed with `contract.py`'s own
  measurement in BOTH directions: it named six `gemm_f16_bias_relu_*` kernels
  where ptxas contracts nothing, and omitted ten kernels where it does.  Two
  lists of one thing drift; the set is derived here now.

  The 0 came from a hardcoded blocker table whose entry for every loop kernel
  was "needs loop invariants".  That was true when it was written and `loopval`
  has since provided them.  A tool that models another tool's answer instead of
  asking it goes stale silently -- so this asks.

AND THE REPAIR NAMED WAS THE EXPENSIVE ONE.  Forbidding the contraction with
`mul.rn.f32`/`add.rn.f32` costs 0 to +7.1% instructions and gives the kernel
TWO roundings where the hardware does one.  Saying `fma.rn.f32` instead -- the
fusion the machine already performs -- emits a BYTE-IDENTICAL instruction
stream, is more accurate, and validates.  The repair is to SAY what the machine
does, not to forbid it.
"""
import contextlib, io, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or '.')
import contract, gap, loopgap, loopval


def structural(k, o1=False):
    """(verdict, reason) from loopval, with its progress output suppressed.

    The VERDICT is returned separately from the reason because the pair below
    asserts an UNPROVED as well as a VALIDATED, and UNPROVED's message is the
    failing obligation ("store 0 value: sat") rather than the word.
    """
    d = 'o1' if o1 else 'corpus'
    p, s = f'{d}/{k}.ptx', f'{d}/{k}.sass'
    if not os.path.exists(s): return ('-', 'no build')
    if not loopgap.has_control_flow(p): return ('-', '(no loop)')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            v, msg, _ = loopval.validate(p, s, 20, 'wide')
    except Exception as e:
        v, msg = 'REFUSED', str(e)
    return (v, v if v == 'VALIDATED' else loopgap.reason_key(msg))


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    ks = sorted(contract.contraction_kernels())
    # ANTI-DRIFT.  The set must be what the artifacts say, not a list someone
    # keeps up to date: the list this file used to carry was wrong in BOTH
    # directions (six kernels named that contract nothing, ten omitted).  A
    # hardcoded set is invisible to every assertion below -- the pair at the
    # bottom does not depend on it -- so the agreement is checked here.
    measured = sorted(contract.contraction_kernels())
    if ks != measured:
        print(f'FAIL: fpgate is not measuring its own set -- it reports {len(ks)} '
              f'kernels where contract.py measures {len(measured)}')
        return 1
    # ...and the guard above compares one function against ITSELF, so it catches
    # a hardcoded LIST and is SILENT about a wrong MEASUREMENT.  Both sides move
    # together, which is the same silence a generated description has.  These
    # two are the control, and they are the two directions the obvious metric
    # (`FFMA(sass) - fma(ptx) > 0`) gets wrong:
    #   swiglu  fuses at HALF precision, so its FFMA count is 0 either way and
    #           the metric misses it entirely;
    #   fp8     has 11 FFMA the PTX did not ask for and forbidding the fusion
    #           changes nothing, so the metric invents a contraction.
    for k, want in (('gemm_f16_swiglu_512', True), ('gemm_fp8_512', False)):
        if not os.path.exists(f'corpus/{k}.ptx'): continue
        if (k in measured) != want:
            print(f'FAIL: {k} is {"absent from" if want else "in"} the contraction '
                  f'set. That is the metric answering by inference rather than by '
                  f'forbidding the fusion and re-assembling.')
            return 1
    print(f'{len(ks)} kernels where ptxas actually fuses a mul.f32 into an add.f32\n')
    # The doc quotes this number, and it has been wrong twice -- once as a
    # hardcoded 9, once as a derived 16.  Check it here, where the measurement
    # is, rather than leaving a third copy to go stale on its own.
    doc = os.path.join('..', '..', 'docs', 'ptxas_translation_validation.md')
    if os.path.exists(doc):
        m = re.search(r'\*\*Contraction\*\* [^\n]*?\*\*(\d+) kernels\*\*', open(doc).read())
        if not m:
            print('FAIL: the doc no longer states a contraction count for this to check')
            return 1
        if int(m.group(1)) != len(ks):
            print(f'FAIL: the doc says {m.group(1)} kernels contract; measured {len(ks)}')
            return 1
    print(f'{"kernel":38s}{"gap":>4}   what still blocks it')
    unlocked = []
    for k in ks:
        r = gap.census(k)
        pg = [o for o in r['ptx'][1] if o != 'bra']   # loopval owns `bra`
        sg = r['sass'][1]
        n = len(pg) + len(sg)
        v, why = structural(k)
        if n: why = f'{n} unmodelled opcode(s); ' + why
        print(f'{k:38s}{n:4d}   {why}')
        if n == 0 and v == 'VALIDATED': unlocked.append(k)

    # The one kernel a repaired build exists for -- and the PAIR is the result,
    # so both halves are asserted rather than just the green one.  The emitter
    # says `fma.rn.f32` now, so the SHIPPED kernel is the validated one and the
    # refutation lives on in `_muladd`, the form it used to emit.  Asserting
    # only the green half would be satisfied by a validator that never refutes.
    print()
    bad = 0
    for v, want in (('naive_gemm_f32', 'VALIDATED'), ('naive_gemm_f32_muladd', 'UNPROVED')):
        got, why = structural(v, o1=True)
        mark = 'ok' if got == want else f'CHANGED (wanted {want})'
        print(f'  o1/{v:24s} {got:10s} {mark:28s} {why if got != "VALIDATED" else ""}')
        if got != want: bad += 1
        if got == 'VALIDATED' and not v.endswith('_muladd'): unlocked.append(v)
    # A printed CHANGED is not a gate.  This pair is a STANDING RESULT in both
    # directions -- the shipped kernel must validate, and the form it replaced
    # must still be refuted, because a corpus containing nothing the validator
    # refutes cannot be told apart from a validator that always says VALIDATED.
    if bad:
        print(f'\nFAIL: {bad} of the two standing verdicts moved.')
        return 1

    print(f'\nkernels the fma.rn repair unlocks today: {len(unlocked)}'
          f'{" -- " + ", ".join(unlocked) if unlocked else ""}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
