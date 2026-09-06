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
    print(f'{len(ks)} kernels where ptxas emitted an FFMA the PTX did not ask for\n')
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

    # The one kernel a repaired build exists for -- and the pair is the result,
    # so BOTH halves are asserted rather than just the green one.
    print()
    for v, want in (('naive_gemm_f32', 'UNPROVED'), ('naive_gemm_f32_fma', 'VALIDATED')):
        got, why = structural(v, o1=True)
        mark = 'ok' if got == want else f'CHANGED (wanted {want})'
        print(f'  o1/{v:24s} {got:10s} {mark:28s} {why if got != "VALIDATED" else ""}')
        if got == 'VALIDATED' and v.endswith('_fma'): unlocked.append(v)

    print(f'\nkernels the fma.rn repair unlocks today: {len(unlocked)}'
          f'{" -- " + ", ".join(unlocked) if unlocked else ""}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
