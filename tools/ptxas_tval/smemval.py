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
faithful only when each access has its required natural alignment (4/8/16 bytes
for scalar/64/128-bit accesses), so that is discharged wherever its guard holds.
"""
import sys, time
import batch


def validate(ptx, sass, budget=60, mode='wide'):
    # batch also serves callers that bypass this wrapper. It checks shared
    # alignment, barriers and arrays through the same helper as tval.run.
    return batch.validate(ptx, sass, budget, mode)


if __name__ == '__main__':
    a = sys.argv[1:]
    t0 = time.time()
    v, why, n = validate(a[0], a[1], int(a[2]) if len(a) > 2 else 60,
                         a[3] if len(a) > 3 else 'wide')
    print(f'{v}  {n} obligations  {why}  {time.time()-t0:.1f}s')
