import sys
from z3 import *
import sassexec, ptxexec, mulmode
def mk(mul, ab):
    sym={k:BitVec(k,32) for k in ('A_lo','A_hi','B_lo','B_hi','O_lo','O_hi','N','stackptr','gridc_lo','gridc_hi','ctaid_x','tid_x')}
    sym['mem']=Array('mem',BitVecSort(64),BitVecSort(32)); sym['mul']=mul; sym['abstract']=ab
    return sym
mul=mulmode.MODES['uf']()
pool={}
def sf(i,k): return pool.setdefault((i,k), BitVec(f'L{i}_{k}',32))
LP=[0,2,1,3]; inv={j:i for i,j in enumerate(LP)}
symP=mk(mul,sf); symS=dict(symP); symS['abstract']=lambda j,k: sf(inv[j],k)
P=ptxexec.run_ptx(sys.argv[1],symP); S=sassexec.run_sass(sys.argv[2],symS)
def collect(e, out, seen):
    st=[e]
    while st:
        x=st.pop()
        if x.get_id() in seen: continue
        seen.add(x.get_id())
        if x.decl().kind()==Z3_OP_UNINTERPRETED and x.num_args()==2 and x.decl().name() in ('MULLO','MULHI'):
            out[x.sexpr()]=x
        st.extend(x.children())
pm={}; sm={}
sp=set(); ss=set()
for _,v,_ in P.stores: collect(v,pm,sp)
for _,v,_ in S.stores: collect(v,sm,ss)
print(f'distinct multiply terms: ptx {len(pm)}  sass {len(sm)}  identical (same sexpr) {len(set(pm)&set(sm))}')
ks=sorted(pm)[:2]
for k in ks: print('  ptx  sample:', k[:200])
ks=sorted(sm)[:2]
for k in ks: print('  sass sample:', k[:200])
