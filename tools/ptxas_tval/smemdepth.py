"""Is "33 kernels behind shared memory" one feature or thirty-three?

Counts, per shared-memory kernel, the OTHER features it needs, so the bucket can
be priced instead of assumed.  A first-refusal census cannot answer this: it
names one opcode and stops.
"""
import re, collections
import depth, cfg

FEATURES = {
  'smem'     : (r'^(ld|st)\.shared|^bar\.sync', r'^(LDS|STS|BAR\.SYNC)'),
  'tensor'   : (r'^(mma|wmma|ldmatrix)\.', r'^(HMMA|QMMA|IMMA|LDSM)'),
  'asynccp'  : (r'^cp\.async', r'^(LDGSTS|DEPBAR|LDGDEPBAR)'),
  'shuffle'  : (r'^shfl\.', r'^SHFL'),
  'float'    : (r'^(add|sub|mul|fma|div|rcp|sqrt|rsqrt|ex2|lg2|sin|cos)\.[a-z.]*f(32|64)|^cvt\.[a-z.0-9]*f(16|32|64)',
                r'^(F[A-Z]|MUFU|HADD|HMUL|HFMA|D[A-Z]|I2F|F2F|F2I|F2FP)'),
  'intdiv'   : (r'^(div|rem)\.u?s?(32|64)$', r'^(I2F\.U32\.RP|MUFU\.RCP)'),
  'fp8'      : (r'e4m3|e5m2', r'E4M3|E5M2'),
  'macroop'  : (r'^(div|rcp|sqrt)\.rn\.', r'^CALL\.REL\.NOINC'),
  'uniform'  : (r'^$', r'^(ULDC|USHF|UIADD|UIMAD|UMOV|ULEA|S2UR|R2UR)'),
}

def feats(k):
    p, s = depth.ptx_ops(f'corpus/{k}.ptx'), depth.sass_ops(f'corpus/{k}.sass')
    out = []
    # BRANCH COMES FROM cfg.analyse, NOT A REGEX.  Every SASS kernel ends with a
    # `.L_x_0: BRA .L_x_0` self-loop after EXIT; a regex on BRA therefore fires
    # on all 67 and reports nothing.  cfg.analyse already excludes that trap.
    n, fwd, back, rec, mem = cfg.analyse(f'corpus/{k}.sass')
    if fwd or p.get('bra'): out.append('branch')
    if back: out.append('LOOP')
    for name, (pre, sre) in FEATURES.items():
        if any(re.search(pre, o) for o in p) or any(re.search(sre, o) for o in s):
            out.append(name)
    return out, sum(p.values()), sum(s.values()), p

# DERIVE the shared-memory kernels from the corpus rather than reading a list.
# A checked-in list is a second copy of a fact the corpus already carries, and
# it goes stale the moment a kernel is added.
import glob, os
ks = sorted(os.path.basename(p)[:-4] for p in glob.glob('corpus/*.ptx')
            if '.shared' in open(p).read())
rows = []
for k in ks:
    f, np_, ns, p = feats(k)
    muls = sum(c for o, c in p.items() if o.startswith(('mul.', 'mad.', 'mul24')))
    rows.append((k, f, np_, ns, muls))

print(f'{"kernel":34s}{"ptx":>7}{"sass":>7}{"muls":>7}  features beyond shared memory')
for k, f, np_, ns, muls in rows:
    extra = [x for x in f if x not in ('smem', 'uniform')]
    print(f'{k:34s}{np_:7d}{ns:7d}{muls:7d}  {" ".join(extra) or "-- NONE --"}')

print()
g = collections.Counter(tuple(sorted(x for x in f if x not in ('smem','uniform'))) for _,f,_,_,_ in rows)
print('groups:')
for combo, n in g.most_common():
    print(f'  {n:3d}  {" ".join(combo) or "SHARED MEMORY ALONE"}')
