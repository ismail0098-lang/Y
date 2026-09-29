"""Scope census WITH per-kernel detail, cross-tabbed against control flow.

The question this answers: for each kernel refused by an executor, WHAT stopped
it, and would it be in scope if that one thing were fixed?  A count of refusals
is not a plan; the detail is.
"""
import glob,os,sys,collections,re
import sassexec, ptxexec, params, batch, mulmode, cfg

FLOATY = re.compile(r'^(F|MUFU|HMMA|HADD|HMUL|HFMA|I2F|F2F|F2I|DADD|DMUL|DFMA|IMMA|OFMA)')

def blocker(k):
    f=f'corpus/{k}.ptx'
    try:
        _, layout = params.parse(f)
        sym = batch.mk(mulmode.MODES['wide'](), layout)
        P = ptxexec.run_ptx(f, sym)
        S = sassexec.run_sass(f'corpus/{k}.sass', sym)
        return None, (len(P.stores), len(S.stores))
    except Exception as e:
        return str(e).split('\n')[0], None

rows=[]
for f in sorted(glob.glob('corpus/*.ptx')):
    k=os.path.basename(f)[:-4]
    if not os.path.exists(f'corpus/{k}.sass'): continue
    b,st = blocker(k)
    n,fwd,back,rec,mem = cfg.analyse(f'corpus/{k}.sass')
    rows.append((k,b,st,n,fwd,back,rec,mem))

print(f'{"kernel":36s}{"insn":>6}{"fwd":>4}{"loop":>5}{"shmem":>6}  blocker')
for k,b,st,n,fwd,back,rec,mem in rows:
    tag = 'PAST-EXEC' if b is None else b[:58]
    print(f'{k:36s}{n:6d}{fwd:4d}{back:5d}{mem:6d}  {tag}')

print()
past=[r for r in rows if r[1] is None]
print(f'PAST BOTH EXECUTORS: {len(past)} / {len(rows)}')

# classify the blockers
def cls(b):
    if b is None: return 'past'
    m=re.search(r"UNMODELLED[^']*'([^']+)'", b)
    op = m.group(1) if m else b[:40]
    return ('FLOAT '+op) if FLOATY.match(op.upper()) else ('OTHER '+op)
c=collections.Counter(cls(r[1]) for r in rows)
print('\nblockers:')
for kk,v in c.most_common(40): print(f'  {v:3d}  {kk}')
