"""Translation validation for a straight-line kernel that uses SHARED memory.

Adds three obligation classes to the straight-line ones in `tval.py`:

  BARRIERS  the two sides cross the same number of barriers.  Checked BEFORE any
            query is posed, because if they differ the `H_k` indices no longer
            pair and every later obligation is comparing the wrong things.
  SMEM@k    the shared array ENTERING barrier k is equal on both sides.  This is
            the obligation that makes the uninterpreted `H_k` do its work: equal
            arguments give equal results by congruence, so everything read after
            the barrier agrees for free -- and a store ptxas moved across the
            barrier changes the argument and is caught here rather than being
            silently absorbed.
  SMEM@end  the array after the last barrier.  Writes there are observable by
            other threads even though nothing in THIS thread reads them back, so
            they are part of the kernel's meaning and not dead code.

ALIGNMENT is a fourth, and it is a proof obligation rather than an assert
because the addresses are symbolic: the word-indexed model in `smem.py` is only
faithful if every access is 4-byte aligned, so that is discharged, not assumed.
"""
import sys, time
from z3 import *
import params, batch, mulmode, ptxexec, sassexec


def validate(ptx, sass, budget=60, mode='wide'):
    _, layout = params.parse(ptx)
    sym = batch.mk(mulmode.MODES[mode](), layout)
    P = ptxexec.run_ptx(ptx, sym)
    S = sassexec.run_sass(sass, sym)
    bar = sym['bar']

    if not bar.agree():
        return 'UNPROVED', (f'barrier counts differ: ptx {bar.ptx_count}, '
                            f'sass {bar.sass_count}'), 0
    pre = [ULT(sym['tid_x'], BitVecVal(1024, 32)),
           ULT(sym['ctaid_x'], BitVecVal(1 << 24, 32))]

    def unsat(claim, to=budget):
        """Returns (ok, verdict).

        `sat` AND `unknown` ARE NOT THE SAME RESULT, and reporting them with one
        message is the mistake this whole project is about.  `sat` means the two
        programs provably CAN differ -- a refutation, and a finding.  `unknown`
        means the solver ran out of time and says NOTHING about the kernel.  The
        first version of this file printed one string for both, which would have
        recorded a solver wall as a miscompilation.
        """
        s = Solver(); s.set('timeout', to * 1000); s.add(pre); s.add(Not(claim))
        r = str(s.check())
        return r == 'unsat', r

    n = 0
    for e in P.align_obs + S.align_obs:
        ok, r = unsat(e)
        if not ok:
            return 'UNPROVED', f'a shared access is not provably 4-byte aligned [{r}]', n
        n += 1

    for k in range(bar.ptx_count):
        ok, r = unsat(P.smem_snaps[k] == S.smem_snaps[k])
        if not ok:
            return 'UNPROVED', (f'shared memory entering barrier {k}: REFUTED (sat)' if r == 'sat'
                                else f'shared memory entering barrier {k}: solver said {r} '
                                     f'-- a WALL, not a mismatch'), n
        n += 1

    ok, r = unsat(P.smem == S.smem)
    if not ok:
        return 'UNPROVED', ('shared memory at exit: REFUTED (sat)' if r == 'sat'
                            else f'shared memory at exit: solver said {r} -- a WALL, not a mismatch'), n
    n += 1

    # the global side is unchanged; reuse the existing straight-line obligations
    v, why, m = batch.validate(ptx, sass, budget, mode)
    return v, f'{why}  [+{n} shared obligations, {bar.ptx_count} barrier(s)]', n + m


if __name__ == '__main__':
    a = sys.argv[1:]
    t0 = time.time()
    v, why, n = validate(a[0], a[1], int(a[2]) if len(a) > 2 else 60,
                         a[3] if len(a) > 3 else 'wide')
    print(f'{v}  {n} obligations  {why}  {time.time()-t0:.1f}s')
