"""Translation validation across ONE loop, by a simulation relation at the header.

WHY A RELATION.  The two programs run an unbounded number of iterations, so
nothing can be settled by executing them.  The standard move is to guess a
relation R between the two header states, prove it holds on entry and is
preserved by one iteration, and use it to compare the stores after the loop.  R
is PROPOSED by simulating the prologue -- which computes concrete entry values
on both sides -- and PROVED by the solver.  A wrong proposal fails its own
obligation and is dropped, so nothing is assumed.

WHAT IS PROVED, for every input:
   BASE       R holds when the header is first reached
   ENTRY      both programs agree on whether to enter the loop at all
   STEP       R is preserved by one iteration          (to a fixpoint)
   LOOPCOND   they agree on whether to iterate again
   STORES     under R, the code after the loop performs the same stores
Induction on the iteration count then gives: same trip count, same stores.

THREE THINGS THAT ARE EASY TO GET WRONG, each found by a counterexample here.

1. PHASING.  The PTX tests at the TOP of the body and ptxas rewrites the test to
   the BOTTOM.  So the SASS's back-edge condition at iteration k is the PTX's
   guard at k+1, and the SASS epilogue's input is the SASS body's OUTPUT while
   the PTX epilogue's input is the PTX HEADER state.  Those line up only because
   R is preserved.  Comparing them at the same NAME rather than at the same
   POINT is the mistake.

2. INVARIANT vs CARRIED live-ins.  A pairwise relation between header states is
   not enough on its own, because ptxas may RECOMPUTE inside the body a value
   the PTX computed once in the prologue.  In `exact_pv` the PTX carries
   `(ctaid.y*Q + ctaid.x)*T` in a register across the loop while the SASS
   rebuilds it every iteration from `R0` and a parameter -- so no live-in of one
   side corresponds to a live-in of the other, and the load addresses were free
   to differ.  A slot the body never writes holds its prologue value at every
   header; rewriting it to that value expresses both sides in the kernel's
   shared parameters and the addresses coincide.

3. SUBSTITUTE, DO NOT MERELY ASSUME.  See rewrite().

REFUSED by name, never assumed: more than one back edge, a branch inside the
body, a store inside the body, an unrolled SASS loop.  ptxas unrolls x4 at -O2
and above even for a loop with no memory in it (`unroll.py`), so in practice
this applies to -O0/-O1 output; the optimization-level differential is what
relates that to the shipped build, and it is sampled evidence rather than a
proof.
"""
import re, sys, time, random
from z3 import *
import loopcfg, ptxexec, sassexec, params, batch, mulmode, conc, mac64


# ---------- selectors: a name for one 32-bit slot of a side's state ----------
def _vars(e, acc=None, seen=None):
    if acc is None: acc, seen = set(), set()
    if e.get_id() in seen: return acc
    seen.add(e.get_id())
    if e.num_args() == 0 and e.decl().kind() == Z3_OP_UNINTERPRETED:
        acc.add(e.decl().name())
    for c in e.children(): _vars(c, acc, seen)
    return acc

def ptx_live_ins(exprs):
    names = set()
    for e in exprs: names |= {d for d in _vars(e) if d.startswith('ptx_undef_')}
    sels = []
    for n in sorted(names):
        for pat, mk in ((r'ptx_undef_r(\d+)',  lambda i: [('r', i)]),
                        (r'ptx_undef_rd(\d+)', lambda i: [('rdlo', i), ('rdhi', i)]),
                        (r'ptx_undef_p(\d+)',  lambda i: [('p', i)]),
                        (r'ptx_undef_f(\d+)',  lambda i: [('f', i)])):
            m = re.fullmatch(pat, n)
            if m: sels += mk(int(m.group(1))); break
        else:
            raise Exception(f'unrecognised PTX live-in {n!r}  (refusing, not guessing)')
    return sels

def sass_live_ins(exprs):
    names = set()
    for e in exprs: names |= {d for d in _vars(e) if d.startswith('sass_undef_')}
    sels = []
    for n in sorted(names):
        m = re.fullmatch(r'sass_undef_R(\d+)', n)
        if m: sels.append(('R', int(m.group(1)))); continue
        m = re.fullmatch(r'sass_undef_P(\d+)', n)
        if m: sels.append(('P', int(m.group(1)))); continue
        raise Exception(f'unrecognised SASS live-in {n!r}  (refusing, not guessing)')
    return sels

def p_in(sel):
    k, n = sel
    if k == 'r': return BitVec(f'ptx_undef_r{n}', 32)
    if k == 'f': return BitVec(f'ptx_undef_f{n}', 32)
    if k == 'p': return Bool(f'ptx_undef_p{n}')
    d = BitVec(f'ptx_undef_rd{n}', 64)
    return Extract(31, 0, d) if k == 'rdlo' else Extract(63, 32, d)

def p_out(sel, st):
    k, n = sel
    if k == 'r': return st.r.get(n, p_in(sel))
    if k == 'f': return st.f.get(n, p_in(sel))
    if k == 'p': return st.p.get(n, p_in(sel))
    d = st.rd.get(n, BitVec(f'ptx_undef_rd{n}', 64))
    return Extract(31, 0, d) if k == 'rdlo' else Extract(63, 32, d)

def s_in(sel):
    k, n = sel
    return BitVec(f'sass_undef_R{n}', 32) if k == 'R' else Bool(f'sass_undef_P{n}')

def s_out(sel, st):
    k, n = sel
    return (st.R if k == 'R' else st.P).get(n, s_in(sel))

def kind_of(sel): return 'bool' if sel[0] in ('p', 'P') else 'bv'

def _region_exprs(st):
    """Every value a region leaves behind.  NOTE Ptx.P is a METHOD (the operand
    reader) while Sass.P is the predicate DICT -- the names collide, so each
    candidate is type-checked rather than duck-typed."""
    out = []
    for nm in ('r', 'rd', 'f', 'R', 'UR'):
        d = getattr(st, nm, None)
        if isinstance(d, dict): out += list(d.values())
    for nm in ('p', 'P'):
        d = getattr(st, nm, None)
        if isinstance(d, dict):
            out += [If(v, BitVecVal(1, 32), BitVecVal(0, 32)) for v in d.values()]
    for a, v, g in st.stores:
        out += [a, v if is_bv(v) else BitVecVal(0, 32),
                If(g, BitVecVal(1, 32), BitVecVal(0, 32))]
    return out or [BitVecVal(0, 32)]


def propose(psel, ssel, allp, alls, pp, sprol, pb, sb, sym, samples, iters=4):
    """Match PTX carried slots to SASS carried slots by simulating both sides
    THROUGH SEVERAL ITERATIONS, not just the prologue.

    Simulating the prologue alone is not discriminating: at loop entry the
    counter and both halves of the accumulator are ALL ZERO on both sides, so
    every zero slot matches every other and which partner a slot gets is decided
    by sort order.  That is how the accumulator relation went missing twice --
    once by a greedy dedup at proposal time, and again inside rewrite(), which
    has to pick ONE partner per SASS slot and picked the counter.

    Stepping the body forward separates them on the first iteration: the counter
    becomes 1 and the accumulator becomes a product.  The concrete state is fed
    back in as the next header's live-in values, which is exactly the
    interpretation the relation is about.  Where a pair still cannot be
    separated the STEP obligation drops it, so this only has to be good enough
    to make rewrite()'s choice sensible."""
    rng = random.Random(20260904)
    agree, held = {}, []

    def slots(sels, mk_out, region, side):
        """the concrete state to carry between iterations, by SYMBOL NAME"""
        out = {}
        for k, n in sels:
            if side == 'p':
                nm = {'r': f'ptx_undef_r{n}', 'f': f'ptx_undef_f{n}',
                      'p': f'ptx_undef_p{n}'}.get(k, f'ptx_undef_rd{n}')
                e = (region.rd.get(n) if k in ('rdlo', 'rdhi') else
                     mk_out((k, n), region))
                if e is None: continue
                out[nm] = e
            else:
                nm = f'sass_undef_R{n}' if k == 'R' else f'sass_undef_P{n}'
                out[nm] = mk_out((k, n), region)
        return out

    step_p = slots(psel, p_out, pb, 'p')
    step_s = slots(ssel, s_out, sb, 's')

    for t in range(samples):
        env = {'__salt__': rng.getrandbits(30),
               '__mem__': lambda arr, a: (a * 2654435761 + 12345) & 0xffffffff}
        for k, v in sym.items():
            if is_bv(v) and v.num_args() == 0:
                env[v.decl().name()] = rng.getrandbits(v.size())
        c = conc.Conc(env)
        # EVERY live-in gets its prologue value, not just the carried ones.  The
        # invariant ones do not change from iteration to iteration, but they DO
        # appear in the body's load addresses, and leaving them unbound gives
        # each side a different pseudo-value -- so the two sides' addresses
        # diverge and nothing matches.  That is what made the first stepped
        # version propose one pair where the un-stepped one proposed nine.
        seed = {}
        for a in allp:
            nm = ({'r': f'ptx_undef_r{a[1]}', 'f': f'ptx_undef_f{a[1]}',
                   'p': f'ptx_undef_p{a[1]}'}.get(a[0], f'ptx_undef_rd{a[1]}'))
            e = (pp.rd.get(a[1]) if a[0] in ('rdlo', 'rdhi') else p_out(a, pp))
            if e is not None and not e.eq(p_in(a)): seed[nm] = e
        for b in alls:
            nm = f'sass_undef_R{b[1]}' if b[0] == 'R' else f'sass_undef_P{b[1]}'
            e = s_out(b, sprol)
            if not e.eq(s_in(b)): seed[nm] = e
        held += list(seed.values())
        for nm, e in seed.items(): env[nm] = c.ev(e)

        pe = {a: p_out(a, pp) for a in psel}
        se = {b: s_out(b, sprol) for b in ssel}
        held += list(pe.values()) + list(se.values())      # keep the AST alive
        pv = {a: c.ev(e) for a, e in pe.items()}
        sv = {b: c.ev(e) for b, e in se.items()}
        for it in range(iters):
            for a in psel:
                for b in ssel:
                    if kind_of(a) != kind_of(b): continue
                    k = (a, b)
                    if t == 0 and it == 0: agree[k] = True
                    if agree.get(k) and pv[a] != sv[b]: agree[k] = False
            if it == iters - 1: break
            # feed this header's values in and take one step of each body
            e2 = dict(env)
            for a in psel:
                nm = ({'r': f'ptx_undef_r{a[1]}', 'f': f'ptx_undef_f{a[1]}',
                       'p': f'ptx_undef_p{a[1]}'}.get(a[0], f'ptx_undef_rd{a[1]}'))
                if a[0] == 'rdlo':   e2[nm] = (e2.get(nm, 0) & ~0xffffffff) | pv[a]
                elif a[0] == 'rdhi': e2[nm] = (e2.get(nm, 0) & 0xffffffff) | (pv[a] << 32)
                else:                e2[nm] = pv[a]
            for b in ssel:
                e2[f'sass_undef_R{b[1]}' if b[0] == 'R' else f'sass_undef_P{b[1]}'] = sv[b]
            c2 = conc.Conc(e2)
            held += list(step_p.values()) + list(step_s.values())
            nxt_p, nxt_s = {}, {}
            for a in psel:
                nm = ({'r': f'ptx_undef_r{a[1]}', 'f': f'ptx_undef_f{a[1]}',
                       'p': f'ptx_undef_p{a[1]}'}.get(a[0], f'ptx_undef_rd{a[1]}'))
                if nm not in step_p: nxt_p[a] = pv[a]; continue
                w = c2.ev(step_p[nm])
                nxt_p[a] = (w & 0xffffffff) if a[0] == 'rdlo' else (
                           (w >> 32) & 0xffffffff) if a[0] == 'rdhi' else w
            for b in ssel:
                nm = f'sass_undef_R{b[1]}' if b[0] == 'R' else f'sass_undef_P{b[1]}'
                nxt_s[b] = c2.ev(step_s[nm]) if nm in step_s else sv[b]
            pv, sv = nxt_p, nxt_s
    return sorted(k for k, v in agree.items() if v)


def validate(ptx_path, sass_path, budget=60, mode='wide', samples=24, verbose=True):
    P = loopcfg.ptx_regions(ptx_path)
    S = loopcfg.sass_regions(sass_path)
    _, layout = params.parse(ptx_path)
    sym = batch.mk(mulmode.MODES[mode](), layout)

    sp_ins = [i for i in S['prologue'] if 'BRA' not in i[1]]
    sp_bra = [i for i in S['prologue'] if 'BRA' in i[1]]
    if len(sp_bra) > 1:
        raise Exception('SASS prologue has more than one branch  (refusing, not guessing)')
    if sp_bra and sp_bra[0][0] != max(a for a, _ in S['prologue']):
        raise Exception('the SASS zero-trip guard is not the last prologue instruction; '
                        'instructions after it would run under a condition this '
                        'validator does not track  (refusing, not guessing)')

    pp    = ptxexec.run_lines(P['prologue'], sym)
    sprol = sassexec.run_insns(sp_ins, sym, 'sass prologue')
    pb0   = ptxexec.run_lines(P['body'], sym)
    sb0   = sassexec.run_insns(S['body'], sym, 'sass body')
    pe0   = ptxexec.run_lines(P['epilogue'], sym)
    se0   = sassexec.run_insns(S['epilogue'], sym, 'sass epilogue')
    pg0   = ptxexec.run_lines(P['pre_guard'], sym)

    if pb0.stores or sb0.stores:
        raise Exception(f'store inside the loop body (ptx {len(pb0.stores)}, sass '
                        f'{len(sb0.stores)}); this validator compares the stores after '
                        f'the loop only  (refusing, not guessing)')

    neg, pidx = P['guard_pred']
    if pg0.p.get(pidx) is None: raise Exception('the PTX guard predicate is never defined')
    _, bm = S['back']
    sneg, spid = (bm.group(1) == '!'), int(bm.group(2))

    psel = ptx_live_ins(_region_exprs(pb0) + _region_exprs(pg0) + _region_exprs(pe0))
    ssel = sass_live_ins(_region_exprs(sb0) + _region_exprs(se0))

    def untouched(sel, region, inn, out): return out(sel, region).eq(inn(sel))
    carried_p = [a for a in psel if not untouched(a, pb0, p_in, p_out)]
    carried_s = [b for b in ssel if not untouched(b, sb0, s_in, s_out)]
    if verbose:
        print(f'  live-ins  ptx {len(psel)} sass {len(ssel)}   '
              f'carried ptx {len(carried_p)} sass {len(carried_s)}')

    pairs = propose(carried_p, carried_s, psel, ssel, pp, sprol, pb0, sb0, sym, samples)
    if verbose:
        print(f'  relation proposed: {len(pairs)} pairs '
              f'({sum(1 for a, _ in pairs if a[0] in ("rdlo","rdhi"))} are halves of a '
              f'64-bit PTX register)')
    if not pairs:
        return 'UNPROVED', 'random simulation found no corresponding loop-carried values', 0

    # ---- SEED BOTH SIDES, do not rewrite afterwards -----------------------
    #
    # Substituting into a finished term is not enough, and the reason is the
    # sharpest thing this file has to say.  The multiply primitive canonicalises
    # its operands BY z3 NODE ID *at the moment the term is built*, so `MUL64(D,
    # X)` on one side and `MUL64(X', D)` on the other have their order fixed
    # before X and X' are known to be equal -- and no later rewriting reorders
    # them, so congruence closure cannot relate the two.  Measured on this
    # kernel: after substitution the word load's address was provably equal and
    # the byte load's provably UNEQUAL, differing by one in the high word, a
    # carry inherited from an operand order decided during execution.
    #
    # Running both regions in ONE vocabulary from the start fixes it: every
    # loop-invariant live-in is seeded with the value its own prologue computed,
    # and every SASS carried slot with the PTX symbol it is paired to.  Both
    # sides are then built over the kernel's shared parameters and the same
    # loop-carried symbols, the canonicalisation agrees, and the addresses,
    # guards and the accumulator all discharge.
    def seeds(pairs):
        ps = {}
        for a in psel:
            if not untouched(a, pb0, p_in, p_out): continue
            if a[0] in ('rdlo', 'rdhi'):
                if a[1] in pp.rd: ps[('rd', a[1])] = pp.rd[a[1]]
            elif a[0] == 'r' and a[1] in pp.r: ps[('r', a[1])] = pp.r[a[1]]
            elif a[0] == 'p' and a[1] in pp.p: ps[('p', a[1])] = pp.p[a[1]]
        ss, used = {}, set()
        for b in ssel:
            if b[0] == 'R' and untouched(b, sb0, s_in, s_out): ss[b[1]] = s_out(b, sprol)
        for a, b in pairs:
            if b[0] != 'R' or b[1] in used: continue
            used.add(b[1]); ss[b[1]] = p_in(a)
        return ps, ss

    pre = [ULT(sym['tid_x'], BitVecVal(1024, 32)), ULT(sym['ctaid_x'], BitVecVal(1 << 24, 32))]
    # The identity has to be instantiated over the SEEDED terms.  Instantiated
    # over the unseeded ones its operands are `ptx_undef_*` expressions that do
    # not occur in any obligation, so it sits there doing nothing -- which is
    # what the first wiring did, and it looks exactly like the axiom not being
    # strong enough rather than not being about anything.
    axioms = []

    def prove(claim, extra=(), to=budget):
        so = Solver(); so.set('timeout', to * 1000)
        so.add(pre); so.add(axioms); so.add(list(extra)); so.add(Not(claim))
        return str(so.check())

    n = 0
    # ---- BASE ------------------------------------------------------------
    bad = []
    for a, b in pairs:
        r = prove(p_out(a, pp) == s_out(b, sprol)); n += 1
        if r != 'unsat': bad.append((a, b))
    if bad:
        pairs = [x for x in pairs if x not in bad]
        if verbose: print(f'  BASE dropped {len(bad)} unproved pair(s); {len(pairs)} remain')
        if not pairs: return 'UNPROVED', 'no pair survived the base case', n
    if verbose: print(f'  BASE ok ({len(pairs)} pairs)')

    # ---- STEP, to a FIXPOINT ---------------------------------------------
    for it in range(12):
        ps, ss = seeds(pairs)
        pb = ptxexec.run_lines(P['body'], sym, ps)
        sb = sassexec.run_insns(S['body'], sym, 'sass body', ss)
        axioms = mac64.instances(_region_exprs(pb), sym['mul'])
        if verbose and it == 0:
            print(f'  multiplier identity instantiated {len(axioms)}x'
                  + (f'\n    ASSUMED: {mac64.PROVENANCE}' if axioms else ''))
        keep, drop = [], []
        for a, b in pairs:
            r = prove(p_out(a, pb) == s_out(b, sb), [ptx_cont_of(P, sym, ps)]); n += 1
            (keep if r == 'unsat' else drop).append((a, b, r))
        if not drop: break
        pairs = [(a, b) for a, b, _ in keep]
        if verbose:
            unk = sum(1 for *_, r in drop if r != 'sat')
            print(f'  STEP round {it+1}: dropped {len(drop)} '
                  f'({len(drop)-unk} refuted, {unk} unknown/timeout) -> '
                  + ', '.join(f'{a}~{b}[{r}]' for a, b, r in drop[:6]))
        if not pairs: return 'UNPROVED', 'no pair survived the step case', n
    else:
        return 'UNPROVED', 'the relation did not reach a fixpoint in 12 rounds', n
    if verbose: print(f'  STEP ok  (R preserved by one iteration, {len(pairs)} pairs)')

    ps, ss = seeds(pairs)
    pb = ptxexec.run_lines(P['body'], sym, ps)
    sb = sassexec.run_insns(S['body'], sym, 'sass body', ss)
    pe = ptxexec.run_lines(P['epilogue'], sym, ps)
    se = sassexec.run_insns(S['epilogue'], sym, 'sass epilogue', ss)
    axioms = mac64.instances(_region_exprs(pb) + _region_exprs(pe), sym['mul'])
    cont = ptx_cont_of(P, sym, ps)
    sc = sb.P.get(spid, Bool(f'sass_undef_P{spid}'))
    sass_cont = Not(sc) if sneg else sc

    # ---- LOOPCOND: the SASS back edge at k is the PTX guard at k+1 --------
    step_sub = [(p_in(a), p_out(a, pb)) for a in psel]
    r = prove(sass_cont == substitute(cont, *step_sub), [cont]); n += 1
    if r != 'unsat': return 'UNPROVED', f'LOOPCOND: back edge vs the next guard: {r}', n
    if verbose: print('  LOOPCOND ok  (same trip count)')

    # ---- ENTRY -----------------------------------------------------------
    if sp_bra:
        m = re.fullmatch(r'(?:@(!?)P(\d+)\s+)?BRA\s+`\(\.L_\w+\)', sp_bra[0][1])
        eneg, epid = (m.group(1) == '!'), int(m.group(2))
        ec = sprol.P.get(epid, Bool(f'sass_undef_P{epid}'))
        sass_skip = Not(ec) if eneg else ec
        pg_entry = ptxexec.run_lines(P['pre_guard'], sym,
                                     {('r', i): v for i, v in pp.r.items()})
        gpe = pg_entry.p[pidx]
        ptx_skip = Not(gpe) if neg else gpe
        r = prove(sass_skip == ptx_skip); n += 1
        if r != 'unsat': return 'UNPROVED', f'ENTRY: zero-trip guards disagree: {r}', n
        if verbose: print('  ENTRY ok  (same zero-trip decision)')

    # ---- STORES after the loop -------------------------------------------
    if len(pe.stores) != len(se.stores):
        return 'UNPROVED', f'epilogue store counts {len(pe.stores)} vs {len(se.stores)}', n
    perm = []
    for i, (pa, pv, pgd) in enumerate(pe.stores):
        hit = [j for j, (sa, _, _) in enumerate(se.stores) if prove(pa == sa) == 'unsat']
        n += 1
        if len(hit) != 1:
            return 'UNPROVED', f'epilogue store {i} matched {len(hit)} sass stores', n
        perm.append(hit[0])
    for i, (pa, pv, pgd) in enumerate(pe.stores):
        sa, sv, sg = se.stores[perm[i]]
        r = prove(pgd == sg); n += 1
        if r != 'unsat': return 'UNPROVED', f'store {i} guard: {r}', n
        r = prove(pv == sv, [pgd]); n += 1
        if r != 'unsat': return 'UNPROVED', f'store {i} value: {r}', n
    if verbose: print(f'  STORES ok ({len(pe.stores)} stores, perm {perm})')
    ax = f', {len(axioms)} multiplier identity instance(s) ASSUMED' if axioms else ''
    return 'VALIDATED', f'{len(pe.stores)} stores, {len(pairs)} relation pairs{ax}', n


def ptx_cont_of(P, sym, ps):
    """The loop-continue condition, computed in the seeded vocabulary."""
    neg, pidx = P['guard_pred']
    g = ptxexec.run_lines(P['pre_guard'], sym, ps).p[pidx]
    return Not(Not(g) if neg else g)


if __name__ == '__main__':
    ptx, sass = sys.argv[1], sys.argv[2]
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    mode = sys.argv[4] if len(sys.argv) > 4 else 'wide'
    t = time.time()
    try:
        v, msg, n = validate(ptx, sass, budget, mode)
    except Exception as e:
        v, msg, n = 'REFUSED', str(e).split('\n')[0], 0
    print(f'{v}  {n} obligations  {msg}, {time.time()-t:.1f}s')
