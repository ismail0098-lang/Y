"""Translation validation of ptxas, with per-obligation abstraction refinement.

Three representations of a 32x32 product, in increasing strength:

  wide    ONE uninterpreted 64-bit value, lo and hi taken as its halves.
          Keeps the fact that the two halves belong to one product -- which
          is what relates the machine's fused `hi(a*b + C)` to the PTX's
          separate lo and hi steps -- while hiding the arithmetic.
  direct  the real bitvector multiply.  Complete, and bit-blasts a multiplier
          per obligation.

Each obligation is tried in the cheap representation first and refined only
if that does not discharge it.  Both are SOUND (an abstraction's `unsat` is a
proof); they differ only in completeness.
"""
import sys, time, random, collections
from z3 import *
import sassexec, ptxexec, mulmode, params, batch, conc

def build(ptxf, sassf, mode, layout, sf, inv):
    mul = mulmode.MODES[mode]()
    symP = batch.mk(mul, layout, sf)
    symS = dict(symP); symS['abstract'] = lambda j,k: sf(inv[j], k)
    return ptxexec.run_ptx(ptxf, symP), sassexec.run_sass(sassf, symS), symP

def run(ptxf, sassf, NS=8, B1=5, B2=60, log=print):
    t_start = time.time(); nobl = 0
    mul0 = mulmode.MODES['wide']()
    _, layout = params.parse(ptxf)
    sym0 = batch.mk(mul0, layout)
    P0 = ptxexec.run_ptx(ptxf, sym0); S0 = sassexec.run_sass(sassf, sym0)
    if len(P0.loads)!=len(S0.loads) or len(P0.stores)!=len(S0.stores):
        return 'UNPROVED', f'load/store counts {len(P0.loads)}/{len(S0.loads)} {len(P0.stores)}/{len(S0.stores)}', 0
    pre0=[ULT(sym0['tid_x'],BitVecVal(1024,32)), ULT(sym0['ctaid_x'],BitVecVal(1<<24,32))]
    def same(a,b,to=20):
        s=Solver(); s.set('timeout',to*1000); s.add(pre0); s.add(a!=b); return str(s.check())=='unsat'
    lperm=[]; sperm=[]
    for i in range(len(P0.loads)):
        hit=[j for j in range(len(S0.loads)) if same(P0.loads[i][0],S0.loads[j][0])]
        if len(hit)!=1: return 'UNPROVED', f'load {i} address matched {len(hit)}', nobl
        lperm.append(hit[0]); nobl+=1
    for i in range(len(P0.stores)):
        hit=[j for j in range(len(S0.stores)) if same(P0.stores[i][0],S0.stores[j][0])]
        if len(hit)!=1: return 'UNPROVED', f'store {i} address matched {len(hit)}', nobl
        sperm.append(hit[0]); nobl+=1
    for i in range(len(P0.loads)):
        if not same(P0.loads[i][1], S0.loads[lperm[i]][1]): return 'UNPROVED', f'load {i} guard', nobl
        nobl+=1
    for i in range(len(P0.stores)):
        if not same(P0.stores[i][2], S0.stores[sperm[i]][2]): return 'UNPROVED', f'store {i} guard', nobl
        nobl+=1
    log(f'  loads/addresses/guards: {nobl} obligations, load perm {lperm}, store perm {sperm}')

    pool={}
    def sf(i,k): return pool.setdefault((i,k), BitVec(f'L{i}_{k}',32))
    inv={j:i for i,j in enumerate(lperm)}
    Pw,Sw,symW = build(ptxf,sassf,'wide',layout,sf,inv)
    Pd,Sd,symD = build(ptxf,sassf,'direct',layout,sf,inv)
    # Every obligation is stated UNDER THE GUARD.  Out of range both programs
    # store nothing, and their intermediates are then free to differ -- ptxas
    # zeroes a register with SEL where the PTX predicates a mov, and neither
    # value is observable.  Asking for unconditional agreement asks for
    # something the compiler never promised, and the counterexamples are all
    # of that shape.
    pre=[ULT(symW['tid_x'],BitVecVal(1024,32)), ULT(symW['ctaid_x'],BitVecVal(1<<24,32))]
    pre.append(Sw.stores[0][2])

    # Propose the pairing by simulation (proposes; proves nothing).  Match on
    # the 32-bit VALUE, not on (value, carry): the two programs agree on every
    # partial sum's value and NOT always on its carry -- a carry nothing
    # consumes is free, and ptxas is entitled to leave it different.  Pairing
    # on the pair therefore rejects correct correspondences.
    sigP=[[] for _ in Pd.wide]; sigS=[[] for _ in Sd.wide]
    for n in range(NS):
        rnd=random.Random(4000+n)
        env={'stackptr':0,'gridc_lo':0,'gridc_hi':0,
             '__mem__':lambda a,b:0,'__salt__':n}
        for base in ('ctaid','tid','ntid','nctaid'):
            for ax in 'xyz': env[f'{base}_{ax}']=rnd.randrange(64)
        for nm in layout: env[nm+'_lo']=rnd.getrandbits(20)*4+0x10000; env[nm+'_hi']=0
        for k in list(pool): env[f'L{k[0]}_{k[1]}']=rnd.getrandbits(32)
        c=conc.Conc(env)
        for j,w in enumerate(Pd.wide): sigP[j].append(c.ev(w[2]))
        for j,w in enumerate(Sd.wide): sigS[j].append(c.ev(w[2]))
    by=collections.defaultdict(list)
    for j,v in enumerate(sigP): by[tuple(v)].append(j)
    prop=[(j, by[tuple(v)][0]) for j,v in enumerate(sigS) if tuple(v) in by]
    log(f'  partial-sum pairs proposed: {len(prop)} / {len(Sd.wide)}')

    # Every obligation is stated UNDER THE GUARD: out of range both programs
    # store nothing and their intermediates are free to differ.
    pre=[ULT(symW['tid_x'],BitVecVal(1024,32)), ULT(symW['ctaid_x'],BitVecVal(1<<24,32)),
         Sw.stores[0][2]]
    def ask(a, b, budget):
        s=Solver(); s.set('timeout',budget*1000); s.add(pre); s.add(a!=b); return str(s.check())
    subW=[[],[]]; subD=[[],[]]; nv=0; okw=0; okd=0; okc=0
    t0=time.time()
    todo = list(prop); rnd_pass = 0
    while todo and rnd_pass < 4:
        rnd_pass += 1; again=[]; progress=0
        for j,i in todo:
            sw=substitute(Sw.wide[j][2], *subW[0]) if subW[0] else Sw.wide[j][2]
            pw=substitute(Pw.wide[i][2], *subW[1]) if subW[1] else Pw.wide[i][2]
            r = ask(sw, pw, B1); nobl+=1; how='wide'
            if r!='unsat':
                sd=substitute(Sd.wide[j][2], *subD[0]) if subD[0] else Sd.wide[j][2]
                pd=substitute(Pd.wide[i][2], *subD[1]) if subD[1] else Pd.wide[i][2]
                r = ask(sd, pd, B2); nobl+=1; how='direct'
            if r!='unsat': again.append((j,i)); continue
            V=BitVec(f'V{nv}',32); nv+=1
            subW[0].append((Sw.wide[j][2],V)); subW[1].append((Pw.wide[i][2],V))
            subD[0].append((Sd.wide[j][2],V)); subD[1].append((Pd.wide[i][2],V))
            okw += how=='wide'; okd += how=='direct'; progress+=1
            # the carry is a SEPARATE obligation: prove it where it holds and
            # leave it alone where it does not -- an unconsumed carry is free
            cs=substitute(Sw.wide[j][3], *subW[0]) if subW[0] else Sw.wide[j][3]
            cp=substitute(Pw.wide[i][3], *subW[1]) if subW[1] else Pw.wide[i][3]
            rc = ask(cs, cp, B1); nobl+=1
            if rc=='unsat':
                C=Bool(f'C{nv}')
                subW[0].append((Sw.wide[j][3],C)); subW[1].append((Pw.wide[i][3],C))
                subD[0].append((Sd.wide[j][3],C)); subD[1].append((Pd.wide[i][3],C))
                okc+=1
        log(f'    sweep {rnd_pass}: {progress} values discharged, {len(again)} left, {time.time()-t0:.0f}s')
        todo=again
        if progress==0: break
    fail=len(todo)
    log(f'  partial sums: {okw} by abstraction, {okd} refined, {okc} carries, {fail} not discharged  ({time.time()-t0:.1f}s)')
    allok=True
    for k in range(len(Pd.stores)):
        sw=substitute(Sw.stores[sperm[k]][1], *subW[0]) if subW[0] else Sw.stores[sperm[k]][1]
        pw=substitute(Pw.stores[k][1],       *subW[1]) if subW[1] else Pw.stores[k][1]
        r=ask(sw,pw,B1); nobl+=1
        if r!='unsat':
            sd=substitute(Sd.stores[sperm[k]][1], *subD[0]) if subD[0] else Sd.stores[sperm[k]][1]
            pd=substitute(Pd.stores[k][1],        *subD[1]) if subD[1] else Pd.stores[k][1]
            r=ask(sd,pd,B2); nobl+=1
        if r!='unsat': allok=False; log(f'  store {k}: {r}')
    dt=time.time()-t_start
    return ('VALIDATED' if allok else 'UNPROVED'), f'{len(Pd.stores)} stores, {len(Pd.loads)} loads, {dt:.1f}s', nobl

if __name__=='__main__':
    v,msg,n = run(sys.argv[1], sys.argv[2],
                  int(sys.argv[3]) if len(sys.argv)>3 else 8,
                  int(sys.argv[4]) if len(sys.argv)>4 else 5,
                  int(sys.argv[5]) if len(sys.argv)>5 else 60)
    print(f'{v}  {n} obligations  {msg}')
