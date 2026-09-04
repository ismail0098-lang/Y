r"""Does a barrier partition the solver problem?

The wall in this corpus is ~65 multiplies (bn254_fr_mul_fast: 65 mul/mad,
UNPROVED at 9705s; ptx_carry_chain: 29, VALIDATED at 24s).  A `bar.sync` is a
natural cut point -- shared memory is the interface across it -- so the question
that decides whether shared memory is worth building is not "how many kernels
does it unlock" but "how many MULTIPLIES does the biggest region still have".

TWO COUNTING TRAPS, both hit while writing this:
  - `@?%?\S*\s*([a-z]...)` BACKTRACKS INTO THE MNEMONIC: greedy \S* eats
    `mul.lo.u32`, fails on `%r1`, and unwinds to the first position where the
    rest matches -- yielding `lo.u32`.  Anchor at the start of the line.
  - `IMAD` IS NOT A MULTIPLY IN SASS.  ptxas spells `mov` as `IMAD.MOV` and
    `add` as `IMAD.IADD`; counting the prefix over-states the real multiplies.
"""
import re, sys

REAL_SASS_MUL = re.compile(r'^(IMAD|IMUL|XMAD)')
FAKE_SASS_MUL = re.compile(r'^IMAD\.(MOV|IADD)')

def regions(f, isptx):
    cur, out = [], []
    for ln in open(f):
        if isptx:
            t = ln.split('//')[0].strip()
            if not t or t.startswith(('.', '{', '}')) or t.endswith(':'): continue
            if t.startswith('@'):                      # predicate prefix
                t = t.split(None, 1)[1] if ' ' in t else ''
            m = re.match(r'([a-z][a-z0-9_.]*)', t)     # ANCHORED: no backtracking
            if not m: continue
            op = m.group(1)
            if op.startswith('bar.sync'): out.append(cur); cur = []; continue
        else:
            m = re.search(r'\*/\s+(@!?\w+\s+)?([A-Z][A-Z0-9_.]*)', ln)
            if not m: continue
            op = m.group(2)
            if op.startswith('BAR.SYNC'): out.append(cur); cur = []; continue
        cur.append(op)
    out.append(cur)
    return out

def muls(r, isptx):
    if isptx:
        return sum(1 for o in r if o.startswith(('mul.', 'mad.')))
    return sum(1 for o in r if REAL_SASS_MUL.match(o) and not FAKE_SASS_MUL.match(o))

WALL = 65
for k in sys.argv[1:]:
    print(f'--- {k}')
    for tag, f, isp in (('PTX', f'corpus/{k}.ptx', True), ('SASS', f'corpus/{k}.sass', False)):
        rs = regions(f, isp)
        c = [muls(r, isp) for r in rs]
        w = max(c) if c else 0
        print(f'   {tag:4s} {len(rs):2d} regions  insn {[len(r) for r in rs]}')
        print(f'   {"":4s}    mul/region {c}')
        print(f'   {"":4s}    WORST {w}   vs wall {WALL}  -> {"OVER by %.1fx" % (w/WALL) if w > WALL else "UNDER"}')
    print()
