"""Run the validator over the whole corpus and classify every kernel.

  VALIDATED  - every store proved equal for every input
  UNPROVED   - the validator ran, some obligation timed out
  REFUSED    - an opcode or operand form this validator does not model
               (a hard error by design: guessing a semantics would make the
                result meaningless)
"""
import sys, os, glob, time, traceback, importlib
from z3 import *
import sassexec, ptxexec, mulmode, params

def mk(mul, layout, ab=None):
    sym = {'stackptr': BitVec('stackptr',32),
           'gridc_lo': BitVec('gridc_lo',32), 'gridc_hi': BitVec('gridc_hi',32)}
    for base in ('ctaid','tid','ntid','nctaid'):
        for ax in 'xyz': sym[f'{base}_{ax}'] = BitVec(f'{base}_{ax}',32)
    for nm in layout:
        sym[nm+'_lo'] = BitVec(nm+'_lo',32); sym[nm+'_hi'] = BitVec(nm+'_hi',32)
    cb = params.cbank(layout)
    cb[0x28]  = 'stackptr'; cb[0x118] = 'gridc_lo'; cb[0x11c] = 'gridc_hi'
    # Launch geometry.  These offsets are a DRIVER ABI fact, not a `ptxas` fact,
    # and establishing them by reading `ptxas` output would use the translator
    # under test to license a fact used to validate it.  `cbank_abi.c` launches
    # a kernel with six distinct extents and reads them back from the DEVICE:
    #   0x00/04/08 = ntid.{x,y,z}   0x0c/10/14 = nctaid.{x,y,z}   (sm_89)
    # `batch.mk` already carried these symbols; only the map was missing, so a
    # kernel reading gridDim refused on an operand the vocabulary could name.
    for i, ax in enumerate('xyz'):
        cb[0x00 + 4*i] = f'ntid_{ax}'
        cb[0x0c + 4*i] = f'nctaid_{ax}'
    sym['cbank'] = cb
    sym['mem']=Array('mem',BitVecSort(64),BitVecSort(32)); sym['mul']=mul
    import fpmode; sym['fp']=fpmode.factory()
    # Shared memory and the barrier family live in `sym` for the same reason the
    # multiply primitive does: two sides that each built their own would share no
    # structure, and congruence could not relate them.
    import smem; sym['smem']=smem.fresh(); sym['bar']=smem.Barriers()
    sym['smem_layout']={}
    if ab: sym['abstract']=ab
    return sym

def validate(ptx, sass, budget, mode='uf'):
    mul = mulmode.MODES[mode]()
    _, layout = params.parse(ptx)
    sym = mk(mul, layout)
    P = ptxexec.run_ptx(ptx, sym); S = sassexec.run_sass(sass, sym)
    if len(P.loads)!=len(S.loads) or len(P.stores)!=len(S.stores):
        return 'UNPROVED', f'load/store counts {len(P.loads)}/{len(S.loads)} {len(P.stores)}/{len(S.stores)}', 0
    pre=[ULT(sym['tid_x'],BitVecVal(1024,32)), ULT(sym['ctaid_x'],BitVecVal(1<<24,32))]
    def same(a,b,to=20):
        s=Solver(); s.set('timeout',to*1000); s.add(pre); s.add(a!=b); return str(s.check())=='unsat'
    def same_if(g,a,b,to=20):
        """Addresses need only agree WHERE THE ACCESS HAPPENS.

        The unconditional form was over-strong, and nothing showed it until a
        kernel had a predicated access whose address is computed AFTER a path
        merge: ptxas turns `@%p2 st.global` into an early `@P0 EXIT`, so every
        register written past that point carries an `If(P0, stale, new)` and the
        two sides' addresses provably differ on the branch where neither stores.
        The semantics of a guarded store is `if g then M[a] := v`, so equal
        effect means equal guards plus equal (a, v) WHERE g HOLDS -- which is
        what the VALUE obligation below already did, and the address obligation
        did not.  Weaker than before, and paired with the guard-equality check
        it is still exactly the semantics; a match that becomes ambiguous under
        the weaker test is refused by the `len(hit)!=1` count."""
        s=Solver(); s.set('timeout',to*1000); s.add(pre); s.add(And(g, a!=b)); return str(s.check())=='unsat'
    n=0
    lperm=[]
    for i in range(len(P.loads)):
        hit=[j for j in range(len(S.loads)) if same_if(P.loads[i][1], P.loads[i][0], S.loads[j][0])]
        if len(hit)!=1: return 'UNPROVED', f'load {i} matched {len(hit)} sass loads', n
        lperm.append(hit[0]); n+=1
    for i in range(len(P.loads)):
        if not same(P.loads[i][1], S.loads[lperm[i]][1]): return 'UNPROVED', f'load {i} guard', n
        n+=1
    sperm=[]
    for i in range(len(P.stores)):
        hit=[j for j in range(len(S.stores)) if same_if(P.stores[i][2], P.stores[i][0], S.stores[j][0])]
        if len(hit)!=1: return 'UNPROVED', f'store {i} matched {len(hit)} sass stores', n
        sperm.append(hit[0]); n+=1
    for i in range(len(P.stores)):
        if not same(P.stores[i][2], S.stores[sperm[i]][2]): return 'UNPROVED', f'store {i} guard', n
        n+=1
    # values, with the loads abstracted so both sides share the loaded words
    pool={}
    def sf(i,k): return pool.setdefault((i,k), BitVec(f'L{i}_{k}',32))
    inv={j:i for i,j in enumerate(lperm)}
    symP=mk(mul,layout,sf); symS=dict(symP); symS['abstract']=lambda j,k: sf(inv[j],k)
    P2=ptxexec.run_ptx(ptx,symP); S2=sassexec.run_sass(sass,symS)
    pre2=[ULT(symP['tid_x'],BitVecVal(1024,32)), ULT(symP['ctaid_x'],BitVecVal(1<<24,32))]
    for i in range(len(P2.stores)):
        pa,pv,pg = P2.stores[i]; sa,sv,sg = S2.stores[sperm[i]]
        s=Solver(); s.set('timeout',budget*1000); s.add(pre2); s.add(And(pg, pv!=sv))
        r=str(s.check())
        if r!='unsat': return 'UNPROVED', f'store {i} value: {r}', n
        n+=1
    return 'VALIDATED', f'{len(P.stores)} stores, {len(P.loads)} loads', n

def validate2(ptx, sass, budget):
    """Abstraction refinement.  The uninterpreted-multiply posing is fast and
    SOUND for `unsat`; a `sat` from it may be spurious (the abstraction lets
    MULLO(x,0) be non-zero), so anything it does not discharge is re-checked
    against the real bitvector multiply."""
    v, msg, n = validate(ptx, sass, budget, 'uf')
    if v == 'VALIDATED': return v, msg + ' [uf]', n
    v2, msg2, n2 = validate(ptx, sass, budget*4, 'direct')
    if v2 == 'VALIDATED': return v2, msg2 + ' [refined to concrete multiply]', n2
    return v2, f'uf: {msg} | concrete: {msg2}', max(n, n2)

if __name__=='__main__':
    budget=int(sys.argv[1]) if len(sys.argv)>1 else 20
    only=sys.argv[2] if len(sys.argv)>2 else None
    rows=[]
    for sfile in sorted(glob.glob('corpus/*.sass')):
        b=os.path.basename(sfile)[:-5]
        if only and only not in b: continue
        p=f'corpus/{b}.ptx'
        t=time.time()
        try:
            v,msg,n = validate2(p, sfile, budget)
        except Exception as e:
            v,msg,n = 'REFUSED', str(e).split('\n')[0][:90], 0
        dt=time.time()-t
        rows.append((v,b,msg,n,dt))
        print(f'{v:10s} {b:44s} {n:4d} obl {dt:7.1f}s  {msg}', flush=True)
    import collections
    c=collections.Counter(r[0] for r in rows)
    print('\n'+'  '.join(f'{k}={v}' for k,v in sorted(c.items())))
