"""IF EVERY OPCODE WERE MODELLED, HOW MANY KERNELS COULD THE SOLVER CLOSE?

Every scope census so far has counted OPCODE coverage, which silently assumes
the binding constraint is the executors.  The standing table says otherwise:

    ptx_carry_chain      29 mul/mad   VALIDATED    24s
    bn254_fr_mul_fast    65 mul/mad   UNPROVED   9705s, 261/276 cut points

so the wall sits between 29 and 65, and it is a property of the SOLVER, not of
the executors.  This asks the counterfactual question directly.

Multiplies rather than instructions because that is what the representation
ladder is about: `bn254_sub_vec` is 251 instructions and closes in 9s; the
multiply count is what predicts the wall.  Barrier regions are used where they
exist, since a barrier is a legitimate cut point.
"""
import glob, os
import depth, barregion

WALL = 65
rows = []
for f in sorted(glob.glob('corpus/*.ptx')):
    k = os.path.basename(f)[:-4]
    if not os.path.exists(f'corpus/{k}.sass'): continue
    try:
        rs = barregion.regions(f, True)
    except Exception:
        continue
    per = [barregion.muls(r, True) for r in rs]
    rows.append((k, sum(per), max(per) if per else 0, len(rs)))

VALID = {'bn254_permute', 'bn254_sub_vec', 'ptx_carry_chain', 'exact_pv'}
under = [r for r in rows if r[2] <= WALL]
over  = [r for r in rows if r[2] >  WALL]
print(f'corpus {len(rows)} kernels;  worst barrier region <= {WALL} muls: '
      f'{len(under)}   over: {len(over)}')
print(f'  of the {len(under)} tractable, {len(under & VALID.__class__(VALID)) if 0 else sum(1 for r in under if r[0] in VALID)} are already VALIDATED')
print()
print('TRACTABLE but not yet validated (the real work queue):')
for k, tot, worst, n in sorted(under, key=lambda r: r[2]):
    if k in VALID: continue
    print(f'   {k:38s} {worst:5d} muls worst region ({n} region(s), {tot} total)')
print()
print('PAST THE WALL even with barrier cut points:')
for k, tot, worst, n in sorted(over, key=lambda r: -r[2])[:8]:
    print(f'   {k:38s} {worst:5d}  ({worst/WALL:.1f}x over)')
print(f'   ... {len(over)} kernels total')
