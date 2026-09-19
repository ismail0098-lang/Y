"""Is the solver wall a BITVECTOR artifact?  Ask both engines, per pair, at one budget.

The division tail was: six bitvector posings `unknown` at up to 1200 s, and the
exact Int translation (`intenc.py`) `unsat` in 0.2 s.  So the obvious next
question is whether the wall -- the field kernels, sole blocker of four at both
levels -- is the same kind of wall.  This asks it directly, on tval's OWN
partial-sum obligations: for each proposed (SASS, PTX) pair of the first N, in
sweep-1 conditions (no substitutions), the bitvector `direct` query and the Int
query on the identical formula, each at the same budget.

Measured on `bn254_fr_mul_fast`, 40 pairs at 60 s (2026-09-19):

    both unsat 23     both unknown 14     bitvector unsat / Int unknown 3

Int closes NONE of the fourteen the bitvector engine cannot, and loses three it
closes.  For carry-chain field multiplies the wall is not the theory.  The
division tail's reasoning is about BOUNDS of a few products; a CIOS carry chain
is many products whose low and high words must be matched exactly, which is
bit-level reasoning either way.

tval.py is not edited: its source is patched IN MEMORY at one asserted anchor
and executed as a module, so no standing result can move.  A moved anchor is a
hard failure, not a silent run of the unpatched validator.

    python3 intwall.py corpus/bn254_fr_mul_fast [N=40] [BUDGET=60]
"""
import collections, re, sys, types

ANCHOR = "    todo = list(prop); rnd_pass = 0\n"
PROBE = '''    for j,i in prop[:__N__]:
        rd = ask(Sd.wide[j][2], Pd.wide[i][2], __B__)
        ri = ask_int(Sd.wide[j][2], Pd.wide[i][2], __B__)
        log(f'PAIR {j} {i} direct {rd} int {ri}')
    return 'MEASURED', 'rung comparison', nobl
'''


def load(n, budget):
    src = open('tval.py').read()
    if src.count(ANCHOR) != 1:
        raise SystemExit('FAIL: tval.py anchor moved; this would run the unpatched validator')
    src = src.replace(ANCHOR, PROBE.replace('__N__', str(n)).replace('__B__', str(budget)) + ANCHOR)
    m = types.ModuleType('tval_intwall')
    m.__file__ = 'tval.py'
    src = src.replace("if __name__ == '__main__':", "if False:")
    exec(compile(src, 'tval.py', 'exec'), m.__dict__)
    return m


def measure(stem, n=40, budget=60, log=print):
    m = load(n, budget)
    # LIVENESS.  "Int closes 0" is also what this reports if the Int query fell
    # back to the bitvector one, or if intenc refused every formula (it answers
    # `unknown` then).  So count what the Int engine was actually asked and what
    # it proved by itself.
    calls = {'n': 0, 'unsat': 0}
    real = m.intenc.check
    def counted(*a, **k):
        r = real(*a, **k)
        calls['n'] += 1
        calls['unsat'] += r == 'unsat'
        return r
    m.intenc = types.SimpleNamespace(**{**vars(m.intenc), 'check': counted})
    rows = []
    def lg(s):
        mm = re.match(r'PAIR (\d+) (\d+) direct (\w+) int (\w+)', s)
        if mm:
            rows.append((int(mm[1]), int(mm[2]), mm[3], mm[4]))
            log(s)
    v, why, _ = m.run(stem + '.ptx', stem + '.sass', log=lg)
    if v != 'MEASURED':
        raise SystemExit(f'FAIL: the probe did not run ({v}: {why})')
    if calls['n'] != len(rows):
        raise SystemExit(f'FAIL: {len(rows)} pairs and {calls["n"]} Int queries -- the Int rung is not what answered')
    if not calls['unsat']:
        raise SystemExit('FAIL: the Int engine proved nothing, so "it closes nothing extra" says nothing')
    return rows


def tally(rows):
    return collections.Counter((d, i) for _j, _i, d, i in rows)


if __name__ == '__main__':
    stem = sys.argv[1] if len(sys.argv) > 1 else 'corpus/bn254_fr_mul_fast'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    b = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    rows = measure(stem, n, b)
    if not rows:
        raise SystemExit('FAIL: no pair was asked')
    t = tally(rows)
    print(f'{len(rows)} pairs at {b} s:')
    for (d, i), c in sorted(t.items(), key=lambda kv: -kv[1]):
        print(f'  {c:3d}  bitvector {d:8s} Int {i}')
    gain = t[('unknown', 'unsat')] + t[('sat', 'unsat')]
    print(f'Int closes {gain} pair(s) the bitvector engine does not')
