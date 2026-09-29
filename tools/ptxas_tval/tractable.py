"""IF EVERY OPCODE WERE MODELLED, HOW MANY KERNELS COULD THE SOLVER CLOSE?

Every scope census once counted OPCODE coverage, which silently assumes the
binding constraint is the executors.  The standing results say otherwise, and
this asks the counterfactual directly, using barriers as cut points.

IT USED TO CARRY ITS OWN COPY OF THE WALL -- `WALL = 65`, a kernel-level
reading, and a hand-listed `VALID` set that had not been updated since four
kernels validated.  Both are `wall.py`'s now: the thresholds are derived from
named region-level ground truth, the validated set is read from `regress.sh`,
and the answer is three-valued, because a region between the largest measured
PROVED and the smallest measured UNKNOWN is on neither side of anything measured.
"""
import collections, glob, os
import wall

rows = collections.defaultdict(list)
val = wall.regress_validated()
for f in sorted(glob.glob('corpus/*.ptx')):
    k = os.path.basename(f)[:-4]
    v, det = wall.verdict(k)
    rows[v].append((k, det))

n = sum(len(x) for x in rows.values())
print(f'corpus {n} kernels; wall thresholds UNDER <= {wall.UNDER_AT}, PAST >= {wall.PAST_AT} '
      f'(symbolic integer multiplies per barrier region)')
for v in ('UNDER', 'UNDECIDED', 'PAST', 'REFUSED'):
    print(f'  {v:10s} {len(rows[v])}')
print()
print('UNDER the wall and not yet validated (the real work queue):')
for k, det in rows['UNDER']:
    if k not in val:
        print(f'   {k:46s} {det}')
print()
print('PAST the wall:')
for k, det in rows['PAST']:
    print(f'   {k:46s} {det}')
