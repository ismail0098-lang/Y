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
import sassexec, ptxexec, mulmode, params, batch, conc, memorder, intenc

def build(ptxf, sassf, mode, layout, sf, inv, sinv, lrep=None):
    mul = mulmode.MODES[mode]()
    # Loads at one proved address share ONE base symbol, the initial memory there.
    rep = (lambda i: i) if lrep is None else (lambda i: lrep[i])
    symP = batch.mk(mul, layout, (lambda i,k: sf(rep(i), k)) if lrep is not None else sf)
    symS = dict(symP); symS['abstract'] = lambda j,k: sf(rep(inv[j]), k)
    P = ptxexec.run_ptx(ptxf, symP)
    # A load read THROUGH a store is spelled with the PTX side's addresses on
    # both sides: the pairing above proved each SASS load and store address equal
    # to its PTX partner's.  See ptxexec.Ptx.gload for the measurement.
    symS['abstract_addr'] = lambda kind, j: (P.loads[inv[j]][0] if kind == 'L'
                                             else P.stores[sinv[j]][0])
    return P, sassexec.run_sass(sassf, symS), symP

def run(ptxf, sassf, NS=8, B1=5, B2=60, log=print):
    """`_run`, with a refusal raised INSIDE an executor reported as a refusal.

    `memorder` refuses a trace it cannot represent, and it can do so from inside
    `ptxexec`/`sassexec` while they build the state -- before any obligation is
    posed.  That escaped here as a traceback, so `regress.sh` read a crash where
    the tool meant "REFUSED", and a crash reads as a missing feature.  `batch`
    already caught it; this validator did not."""
    try:
        return _run(ptxf, sassf, NS, B1, B2, log)
    except memorder.Refusal as e:
        return 'REFUSED', f'{e}', 0
    except Exception as e:
        # An executor's NAMED refusal -- an unmodelled opcode, an estimate chain
        # the device facts were not measured for -- is a verdict.  Anything else
        # is a bug and keeps its traceback.
        if 'refusing, not guessing' in str(e) or 'UNMODELLED' in str(e):
            return 'REFUSED', str(e).splitlines()[0][:200], 0
        raise


def _run(ptxf, sassf, NS=8, B1=5, B2=60, log=print):
    t_start = time.time(); nobl = 0
    mul0 = mulmode.MODES['wide']()
    _, layout = params.parse(ptxf)
    sym0 = batch.mk(mul0, layout)
    P0 = ptxexec.run_ptx(ptxf, sym0); S0 = sassexec.run_sass(sassf, sym0)
    # A READ-BACK IS MODELLED, NOT REFUSED.  Each executor reads a global load
    # through the stores it has already made (memorder.read_through), so a load
    # hoisted above a store it could read back builds a different term from the
    # one below it and the store it feeds is refuted.  This validator VALIDATED
    # such a translation while loads read the initial array, and then refused
    # every read-back -- including ptxas's correct `mem/lsls`.
    if len(P0.loads)!=len(S0.loads) or len(P0.stores)!=len(S0.stores):
        return 'UNPROVED', f'load/store counts {len(P0.loads)}/{len(S0.loads)} {len(P0.stores)}/{len(S0.stores)}', 0
    pre0=[ULT(sym0['tid_x'],BitVecVal(1024,32)), ULT(sym0['ctaid_x'],BitVecVal(1<<24,32))]
    def same(a,b,to=20):
        s=Solver(); s.set('timeout',to*1000); s.add(pre0); s.add(a!=b); return str(s.check())=='unsat'
    # Pairing by PROVED address equality; several accesses at one address form a
    # group rather than a refusal -- see memorder.pair_by_address.  One obligation
    # counted per access, exactly as before.
    lhits=[]
    for i in range(len(P0.loads)):
        lhits.append([j for j in range(len(S0.loads)) if same(P0.loads[i][0],S0.loads[j][0])]); nobl+=1
    lperm, lrep = memorder.pair_by_address(lhits)
    if lperm is None: return 'UNPROVED', f'load {lrep} address', nobl
    shits=[]
    for i in range(len(P0.stores)):
        shits.append([j for j in range(len(S0.stores)) if same(P0.stores[i][0],S0.stores[j][0])]); nobl+=1
    sperm, _ = memorder.pair_by_address(shits)
    if sperm is None: return 'UNPROVED', f'store {_} address', nobl
    # A store's width is its value's.  Two stores paired by address with
    # different widths do not leave the same bytes, and comparing their values
    # would be a z3 sort error rather than an answer.
    for i in range(len(P0.stores)):
        wp, ws = P0.stores[i][1].size(), S0.stores[sperm[i]][1].size()
        if wp != ws: return 'UNPROVED', f'store {i} width: ptx {wp} bits, sass {ws} bits', nobl
    for i in range(len(P0.loads)):
        if not same(P0.loads[i][1], S0.loads[lperm[i]][1]): return 'UNPROVED', f'load {i} guard', nobl
        nobl+=1
    for i in range(len(P0.stores)):
        if not same(P0.stores[i][2], S0.stores[sperm[i]][2]): return 'UNPROVED', f'store {i} guard', nobl
        nobl+=1
    # The pairing above is BY ADDRESS and in any order, so two stores the SASS
    # performs in the opposite order were accepted however they overlap.  One
    # obligation per reordered pair; none for an identity pairing (memorder.py).
    try:
        reord = memorder.reorder_obligations(list(P0.stores), sperm)
    except memorder.Refusal as e:
        return 'REFUSED', str(e), nobl
    for (i, j), claim in reord:
        s=Solver(); s.set('timeout',20*1000); s.add(pre0); s.add(Not(claim)); r=str(s.check())
        nobl+=1
        if r!='unsat': return 'UNPROVED', f'stores {i} and {j} are REORDERED and may overlap [{r}]', nobl
    log(f'  loads/addresses/guards: {nobl} obligations, load perm {lperm}, store perm {sperm}')

    pool={}
    def sf(i,k): return pool.setdefault((i,k), BitVec(f'L{i}_{k}',32))
    inv={j:i for i,j in enumerate(lperm)}
    sinv={j:i for i,j in enumerate(sperm)}
    # A unique pairing passes no representative map, so a result with no shared
    # address builds exactly the calls it always did.
    lr = lrep if any(r != i for i, r in enumerate(lrep)) else None
    Pw,Sw,symW = build(ptxf,sassf,'wide',layout,sf,inv,sinv,lr)
    Pd,Sd,symD = build(ptxf,sassf,'direct',layout,sf,inv,sinv,lr)
    # Every obligation is stated UNDER THE GUARD.  Out of range both programs
    # store nothing, and their intermediates are then free to differ -- ptxas
    # zeroes a register with SEL where the PTX predicates a mov, and neither
    # value is observable.  Asking for unconditional agreement asks for
    # something the compiler never promised, and the counterexamples are all
    # of that shape.
    pre=[ULT(symW['tid_x'],BitVecVal(1024,32)), ULT(symW['ctaid_x'],BitVecVal(1<<24,32))]
    # A kernel that stores NOTHING is not a validation success, it is a kernel
    # with no obligations -- and the whole point of the `fma/plain` control is
    # that a validator which always says VALIDATED reports every row alike.
    # This used to be an IndexError one line below, and a crash in a validator
    # reads exactly like a missing feature (twice already in this repo).
    if not Sw.stores or not Pw.stores:
        return ('REFUSED',
                f'this kernel stores nothing (ptx {len(Pw.stores)}, sass '
                f'{len(Sw.stores)}) -- there is nothing to prove equal, '
                f'{time.time()-t_start:.1f}s', 0)
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
    # Measured facts the SASS executor recorded (divest.py): each build has its
    # own, because each build has its own terms.  Using them is sound -- they
    # hold of the real execution -- and IGNORING them would be sound too, only
    # incomplete; so a validator that does not pass them on is merely weaker.
    def ask(a, b, budget, facts=()):
        s=Solver(); s.set('timeout',budget*1000); s.add(pre); s.add(list(facts)); s.add(a!=b); return str(s.check())
    def relevant(facts, *terms):
        # Only the facts about an estimate the obligation MENTIONS.  An
        # irrelevant fact cannot change the answer and it can cost it: the
        # 64-bit carry-out store of ptx_integer_ops is `unsat` over Int in 0.6 s
        # alone and `unknown` at 60 s with the division facts beside it.
        def ests(t, seen, out):
            if t.get_id() in seen: return out
            seen.add(t.get_id())
            if is_const(t) and str(t).startswith('div_est_'): out.add(str(t))
            for c in t.children(): ests(c, seen, out)
            return out
        mine = set()
        for t in terms: ests(t, set(), mine)
        return [f for f in facts if ests(f, set(), set()) & mine]
    def ask_int(a, b, budget, facts=()):
        # The last rung, and only on `unknown`: the same formula, translated
        # EXACTLY into integer arithmetic (intenc.py).  An untranslatable one
        # stays `unknown`.
        return intenc.check(pre + list(facts) + [a != b], budget)
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
    allok=True; nint=[0]
    def mentions(t, u):
        seen=set(); st=[t]
        while st:
            x=st.pop()
            if x.get_id() in seen: continue
            seen.add(x.get_id())
            if x.eq(u): return True
            st.extend(x.children())
        return False
    proved=[]; first={}
    for k in range(len(Pd.stores)):
        sw=substitute(Sw.stores[sperm[k]][1], *subW[0]) if subW[0] else Sw.stores[sperm[k]][1]
        pw=substitute(Pw.stores[k][1],       *subW[1]) if subW[1] else Pw.stores[k][1]
        sd=substitute(Sd.stores[sperm[k]][1], *subD[0]) if subD[0] else Sd.stores[sperm[k]][1]
        pd=substitute(Pd.stores[k][1],        *subD[1]) if subD[1] else Pd.stores[k][1]
        # A PTX value the spec leaves UNSPECIFIED (ptxexec.Ptx.unspecified) splits
        # the obligation.  Where no unspecified value is produced, the stores must
        # agree as always.  Where one is, the spec is met by ANY SASS value --
        # PROVIDED the PTX stores that value itself: then the obligation is
        # "exists u. sass == u", which holds.  A store that does arithmetic on
        # an unspecified value first is not decided here; it is UNPROVED by name.
        un = [(u, c) for u, c in Pd.unspec if mentions(pd, u)]
        extra = [Not(Or([c for _, c in un]))] if un else []
        bad = None
        for u, c in un:
            sv=Solver(); sv.set('timeout',B1*1000); sv.add(pre); sv.add(c); sv.add(pd != u); nobl+=1
            if str(sv.check()) != 'unsat':
                bad = f'store {k}: stores a value computed FROM an unspecified division, not the value itself'
                continue
            # ONE unspecified value, however many stores carry it: the spec lets
            # it be anything, not a different thing at each store.
            if u.get_id() in first:
                s0 = first[u.get_id()]
                sv=Solver(); sv.set('timeout',B2*1000); sv.add(pre); sv.add(c); sv.add(sd != s0); nobl+=1
                if str(sv.check()) != 'unsat':
                    bad = f'store {k}: the SASS stores a different value than another store of the same unspecified division'
            else:
                first[u.get_id()] = sd
        if bad: allok=False; log('  '+bad); continue
        r=ask(sw,pw,B1,relevant(Sw.assume,sw,pw)+extra); nobl+=1
        if r!='unsat':
            facts=relevant(Sd.assume+proved,sd,pd)+extra
            # An obligation about a division estimate goes to Int FIRST: over
            # bitvectors its tail was `unknown` on six posings at up to 1200 s,
            # so the direct rung there is a timeout spent for nothing.
            if relevant(Sd.assume,sd,pd):
                r=ask_int(sd,pd,B2,facts); nobl+=1
                if r=='unsat': nint[0]+=1
            if r!='unsat':
                r=ask(sd,pd,B2,relevant(Sd.assume,sd,pd)+extra); nobl+=1
                if r=='unknown':
                    r=ask_int(sd,pd,B2,facts); nobl+=1
                    if r=='unsat': nint[0]+=1
        if r!='unsat': allok=False; log(f'  store {k}: {r}')
        else:
            # A PROVED store equality is a fact for the stores after it -- stated
            # under the same side condition it was proved under.  The remainder
            # of a division is `n - d*q` for the quotient just proved.
            proved.append(Implies(And(extra), sd == pd) if extra else sd == pd)
    dt=time.time()-t_start
    xi = f', {nint[0]} over Int' if nint[0] else ''
    return ('VALIDATED' if allok else 'UNPROVED'), f'{len(Pd.stores)} stores, {len(Pd.loads)} loads{xi}, {dt:.1f}s', nobl

if __name__=='__main__':
    v,msg,n = run(sys.argv[1], sys.argv[2],
                  int(sys.argv[3]) if len(sys.argv)>3 else 8,
                  int(sys.argv[4]) if len(sys.argv)>4 else 5,
                  int(sys.argv[5]) if len(sys.argv)>5 else 60)
    print(f'{v}  {n} obligations  {msg}')
