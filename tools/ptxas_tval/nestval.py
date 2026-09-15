"""Translation validation across a NEST of loops -- the lift, built.

`loopval.py` validates ONE loop by a simulation relation at its header.  This
file validates a tree of them, and it reuses `loopval`'s relation machinery
(live-ins, proposal by simulation, the phasing between a PTX top test and a SASS
bottom test) rather than restating it.  `loopval` is not touched, so every
standing result it produces is byte-identical.

WHAT IS PROVED, per loop, IN THE STATE ITS PARENT HAS REACHED:

   BASE       the relation holds when the loop is first reached
   ENTRY      both sides agree whether to enter it at all
   STEP       the relation is preserved by one iteration   (to a fixpoint)
   STORES     one iteration performs the same stores on both sides
   LOOPCOND   they agree whether to iterate again

A CHILD LOOP IS SUMMARISED BY ITS OWN PROOF.  A loop cannot be executed
straight-line, so when a parent's iteration reaches a child, the child is
validated right there -- in the parent's symbolic state, both sides at once --
and then replaced by its EFFECT: every relation pair it proved becomes one fresh
symbol SHARED by the two sides, every other slot it writes becomes a fresh symbol
of its own side, and if it stores, memory becomes one fresh SHARED array.  That
is sound for a stated reason: the pairs hold at the child's exit whatever its
trip count (BASE covers zero trips, STEP the rest, LOOPCOND makes the counts
equal), and memory is equal at exit because it was equal at entry and every
iteration performed the same stores.  Shared symbols forget HOW the outputs
depend on the inputs; that loses completeness, never soundness.

MEMORY IS CARRIED, NOT REFUSED.  Each iteration of a loop that stores starts
from one fresh memory array shared by both sides, which is the induction
hypothesis "memory is equal at the header".  STORES then discharges the step.
So a load in iteration k+1 that reads back a store from iteration k is
MODELLED: it reads the shared array, and a translation that hoisted it above the
store would read a different array.  `loopval` refuses that program.

WHY THE SASS SIDE LOOKS DIFFERENT.  `ptxas` rotates every loop to a bottom test
and puts its zero-trip guard IMMEDIATELY BEFORE the header -- inside the PARENT's
body.  A child unit on the SASS side is therefore `guard + loop`, and the guard
is evaluated in the parent's state.  The top-level guard may be `@!Pn EXIT`
rather than a branch: the program ends when the outer loop would run zero
times, which is only equivalent when nothing after the loop stores, so that is
required.

REFUSED by name, never assumed: more than one loop at the top level, a store in
an iteration followed by a child loop or a later load in the same iteration (the
store trace does not cross a segment boundary), a SASS loop with no zero-trip
guard, a predicated PTX back edge, a branch in a body that is not a child's
structure, a body that can end the program, a PTX carry flag live across a
segment, and a child whose loop-carried symbols also occur in the state it is
entered from (the two would be conflated).
"""
import itertools, re, sys, time
from z3 import *
import loopcfg, ptxexec, sassexec, params, batch, mulmode, mac64, memorder
from loopval import ptx_live_ins, sass_live_ins, p_in, p_out, s_in, s_out, \
    _region_exprs, _vars, propose

REFUSE = '  (refusing, not guessing)'


class Unproved(Exception):
    pass


def refuse(msg):
    raise memorder.Refusal(msg + REFUSE)


# ------------------------------------------------------------------ the trees
def ptx_tree(path):
    raw, lab, backs = loopcfg.ptx_back_edges(path)
    unk = loopcfg.ptx_unclassified_branches(raw)
    if unk:
        refuse(f'PTX branch form this CFG cannot place: {unk[0][1]!r}')
    if not backs:
        refuse('PTX has no loop; use tval.py')
    loops = []
    for h, e, m in backs:
        if m.group(2) is not None:
            refuse('PTX back edge is predicated; the recognised shape tests at the TOP')
        loops.append({'h': h, 'e': e})
    for L in loops:
        inside = [C for C in loops if C is not L and L['h'] < C['h'] and C['e'] < L['e']]
        L['children'] = sorted((C for C in inside
                                if not any(D is not C and D['h'] < C['h'] and C['e'] < D['e']
                                           for D in inside)), key=lambda c: c['h'])
    # innermost first, so a child's exit is known before its parent is split
    for L in sorted(loops, key=lambda l: l['e'] - l['h']):
        h, e = L['h'], L['e']
        if e + 1 >= len(raw) or raw[e + 1][0] != 'l':
            refuse('PTX loop exit label does not follow its back edge')
        L['x'] = e + 1
        spans = [(C['h'], C['x']) for C in L['children']]
        own = [i for i in range(h + 1, e) if not any(a <= i <= b for a, b in spans)]
        bras = [i for i in own if raw[i][0] == 'i' and loopcfg.PTX_BRA.fullmatch(raw[i][1])]
        if len(bras) != 1:
            refuse(f'PTX loop has {len(bras)} own branches; only its exit test is allowed')
        gi = bras[0]
        gm = loopcfg.PTX_BRA.fullmatch(raw[gi][1])
        if gm.group(2) is None or lab.get(gm.group(3).lstrip('$')) != L['x']:
            refuse('PTX exit test does not branch to the label after the back edge')
        if any(C['h'] < gi for C in L['children']):
            refuse('PTX child loop before its parent\'s exit test')
        gp = loopcfg.ptx_pred_index(gm.group(2))
        if gp is None:
            refuse(f'PTX exit test is guarded by %{gm.group(2)}; the predicate file is keyed by number')
        L['guard_pred'] = (gm.group(1) == '!', gp)
        L['pre_guard'] = [raw[i][1] for i in own if h < i < gi and raw[i][0] == 'i']
        groups, cur, i = [], [], gi + 1
        kids = {C['h']: C for C in L['children']}
        while i < e:
            if i in kids:
                groups.append(cur); cur = []; i = kids[i]['x'] + 1; continue
            if raw[i][0] == 'i':
                cur.append(raw[i][1])
            i += 1
        groups.append(cur)
        L['groups'] = groups
    roots = [L for L in loops if not any(L in D['children'] for D in loops)]
    if len(roots) != 1:
        refuse(f'PTX has {len(roots)} top-level loops; this validator handles one nest')
    R = roots[0]
    covered = lambda i: R['h'] <= i <= R['x']
    for i, (k, t) in enumerate(raw):
        if k == 'i' and not covered(i) and loopcfg.PTX_BRA.fullmatch(t):
            refuse('PTX branch outside the loop nest')
    body = [t for grp in _all_groups(R) for t in grp] + [t for L in _all(R) for t in L['pre_guard']]
    if any(re.match(r'^(?:@!?%[\w$]+\s+)?\w+\.cc\b', t) for t in body):
        refuse('PTX carry-flag instruction inside the nest; the flag is not carried across segments')
    return {'prologue': [t for i, (k, t) in enumerate(raw) if k == 'i' and i < R['h']],
            'root': R,
            'epilogue': [t for i, (k, t) in enumerate(raw) if k == 'i' and i > R['x']]}


def sass_tree(path):
    ins, lab, trap, backs = loopcfg.sass_back_edges(path)
    unk = loopcfg.sass_unclassified_branches(ins)
    if unk:
        refuse(f'SASS branch form this CFG cannot place at 0x{unk[0][0]:x}: {unk[0][1]!r}')
    if not backs:
        refuse('SASS has no loop')
    idx = {a: i for i, (a, _) in enumerate(ins)}
    loops = []
    for h, e, m in backs:
        if m.group(2) is None:
            refuse('SASS back edge is unconditional')
        x = next((a for a, _ in ins if a > e and a not in trap), None)
        i = idx[h]
        if i == 0:
            refuse('SASS loop has no zero-trip guard before its header')
        ga, gt = ins[i - 1]
        gm = loopcfg.SASS_BRA.fullmatch(gt)
        em = re.fullmatch(r'@(!?)P(\d+)\s+EXIT', gt)
        if gm and gm.group(2) is not None and lab.get(gm.group(3)) == x:
            kind, gneg, gp = 'bra', gm.group(1) == '!', int(gm.group(2))
        elif em:
            kind, gneg, gp = 'exit', em.group(1) == '!', int(em.group(2))
        else:
            refuse(f'SASS loop at 0x{h:x} has no zero-trip guard immediately before its header')
        loops.append({'g': ga, 'h': h, 'e': e, 'x': x, 'back': m,
                      'gkind': kind, 'gneg': gneg, 'gp': gp})
    for L in loops:
        inside = [C for C in loops if C is not L and L['h'] <= C['g'] and C['e'] < L['e']]
        L['children'] = sorted((C for C in inside
                                if not any(D is not C and D['h'] <= C['g'] and C['e'] < D['e']
                                           for D in inside)), key=lambda c: c['g'])
        if any(C['gkind'] == 'exit' for C in L['children']):
            refuse('SASS child loop guarded by EXIT; the program would end inside the nest')
        spans = [(C['g'], C['e']) for C in L['children']]
        groups, cur = [], []
        kids = {C['g']: C for C in L['children']}
        skip_to = -1
        for a, t in ins:
            if a < L['h'] or a >= L['e'] or a <= skip_to:
                continue
            if a in kids:
                groups.append(cur); cur = []; skip_to = kids[a]['e']; continue
            if loopcfg.SASS_BRA.fullmatch(t) or re.search(r'\bEXIT\b', t):
                refuse(f'SASS loop body branches or exits at 0x{a:x}')
            cur.append((a, t))
        groups.append(cur)
        L['groups'] = groups
    roots = [L for L in loops if not any(L in D['children'] for D in loops)]
    if len(roots) != 1:
        refuse(f'SASS has {len(roots)} top-level loops; this validator handles one nest')
    R = roots[0]
    pro = [(a, t) for a, t in ins if a < R['g']]
    epi = [(a, t) for a, t in ins if a > R['e'] and a not in trap]
    for a, t in pro + epi:
        if loopcfg.SASS_BRA.fullmatch(t):
            refuse(f'SASS branch outside the loop nest at 0x{a:x}')
    return {'prologue': pro, 'root': R, 'epilogue': epi}


def _all(L):
    yield L
    for C in L['children']:
        yield from _all(C)


def _all_groups(L):
    for M in _all(L):
        yield from M['groups']


def same_shape(P, S):
    return len(P['children']) == len(S['children']) and \
        all(same_shape(a, b) for a, b in zip(P['children'], S['children']))


# ------------------------------------------------------------------ states
def mk_p(sym, seed):
    st = ptxexec.Ptx(sym)
    for (k, i), v in seed.items():
        getattr(st, k)[i] = v
    return st


def mk_s(sym, seed):
    st = sassexec.Sass(sym)
    for k, v in seed.items():
        if isinstance(k, str): st.P[int(k[1:])] = v
        else: st.R[k] = v
    return st


def p_seed_of(st):
    return {(k, i): v for k in ('r', 'rd', 'p', 'f') for i, v in getattr(st, k).items()}


def s_seed_of(st):
    d = dict(st.R)
    d.update({f'P{i}': v for i, v in st.P.items()})
    return d


def p_undef(k, i):
    return {'r': lambda: BitVec(f'ptx_undef_r{i}', 32), 'f': lambda: BitVec(f'ptx_undef_f{i}', 32),
            'p': lambda: Bool(f'ptx_undef_p{i}'), 'rd': lambda: BitVec(f'ptx_undef_rd{i}', 64)}[k]()


def s_undef(k):
    return Bool(f'sass_undef_{k}') if isinstance(k, str) else BitVec(f'sass_undef_R{k}', 32)


def writes_p(st):
    return {key for key, v in p_seed_of(st).items() if not v.eq(p_undef(*key))}


def writes_s(st):
    return {key for key, v in s_seed_of(st).items() if not v.eq(s_undef(key))}


def stores_exprs(states):
    out = []
    for st in states:
        for a, v, g in st.stores:
            out += [a, v if is_bv(v) else BitVecVal(0, 32), If(g, BitVecVal(1, 32), BitVecVal(0, 32))]
    return out


def fresh_like(v, name):
    if is_bool(v): return Bool(name)
    return BitVec(name, v.size())


# ------------------------------------------------------------------ commutativity
def comm_instances(exprs):
    """`f(a, b) == f(b, a)` for every commutative application in `exprs`.

    WHY THIS EXISTS, AND IT IS A FINDING RATHER THAN A CONVENIENCE.  `fpmode`
    (FADD, FMAX) and `mulmode` (MUL64, MULLO, MULHI) put a commutative
    operation's operands in ONE order at construction, by z3 NODE ID.  Node ids
    come from a counter that reuses freed ids, so the order depends on every
    term the process built and freed before.  When the two sides' operands are
    equal only semantically -- a PTX product and a SASS product with the same
    value but different load-guard shapes -- they can land in opposite orders,
    and congruence closure will not relate `FADD(acc, pp)` to `FADD(ps, acc)`.
    Measured: `o1/naive_gemm_f32_rn` UNPROVED on the first validation in a
    process and VALIDATED on the second, on identical inputs.

    Ground instances make the order irrelevant to the solver without changing
    how any term is BUILT, so no other validator's terms move.  Each instance is
    licensed exactly as the canonicalisation it neutralises: FADD and FMAX by
    the device facts in `fpmode` (FMAX's flag is consulted, not assumed), and
    the products because `mulmode` already identifies `a*b` with `b*a`."""
    import fpmode
    names = {'FADD', 'MUL64', 'MULLO', 'MULHI'}
    if fpmode.IDENTIFICATIONS['FMAX_IS_COMMUTATIVE']:
        names.add('FMAX')
    out, seen, todo = [], set(), list(exprs)
    while todo:
        e = todo.pop()
        if not is_expr(e) or e.get_id() in seen:
            continue
        seen.add(e.get_id())
        if is_app(e) and e.num_args() == 2 and e.decl().name() in names:
            a, b = e.arg(0), e.arg(1)
            out.append(e == e.decl()(b, a))
        todo.extend(e.children())
    return out


# ------------------------------------------------------------------ the prover
class Nest:
    def __init__(self, sym, budget, samples=24, verbose=True):
        self.sym, self.budget, self.samples, self.verbose = sym, budget, samples, verbose
        self.n = 0
        self.uid = itertools.count()
        self.pre = [ULT(sym['tid_x'], BitVecVal(1024, 32)), ULT(sym['ctaid_x'], BitVecVal(1 << 24, 32))]
        self.stores = 0
        self.disc = {}
        self.depth = 0

    def say(self, msg):
        if self.verbose: print('  ' * (self.depth + 1) + msg)

    def prove(self, claim, extra=(), axioms=()):
        so = Solver(); so.set('timeout', self.budget * 1000)
        so.add(self.pre); so.add(list(axioms)); so.add(list(extra)); so.add(Not(claim))
        so.add(comm_instances([claim] + list(extra)))
        self.n += 1
        return str(so.check())

    def mem(self, tag):
        m = self.sym['mem']
        return Array(f'nest{next(self.uid)}_{tag}', m.domain(), m.range())

    # ---- one side's iteration, children summarised --------------------------
    def run_both(self, PL, SL, pseed, sseed, psym, ssym, prove_children):
        pst, sst, pX, sX = [], [], [], []
        ng = len(PL['groups'])
        for i in range(ng):
            if PL['groups'][i]:
                st = ptxexec.run_lines(PL['groups'][i], psym, pseed); pst.append((i, st)); pseed = p_seed_of(st)
            if SL['groups'][i]:
                st = sassexec.run_insns(SL['groups'][i], ssym, 'sass segment', sseed); sst.append((i, st)); sseed = s_seed_of(st)
            if i < len(PL['children']):
                PC, SC = PL['children'][i], SL['children'][i]
                if prove_children:
                    pseed, sseed, psym = self.prove_loop(PC, SC, pseed, sseed, psym)
                    ssym = psym
                else:
                    pseed, psym, ex = self.opaque('p', PC, SC, pseed, psym); pX += ex
                    sseed, ssym, ex = self.opaque('s', PC, SC, sseed, ssym); sX += ex
        for side, sts in (('PTX', pst), ('SASS', sst)):
            for i, st in sts:
                if not is_true(simplify(st.alive)):
                    refuse(f'{side} loop body can end the program (EXIT / ret)')
                if st.stores and (i < len(PL['children']) or any(j > i and s2.loads for j, s2 in sts)):
                    refuse(f'{side} store in an iteration before a child loop or a later load in the '
                           f'same iteration; the store trace does not cross that boundary')
        pF, sF = mk_p(psym, pseed), mk_s(ssym, sseed)
        return (pF, [s for _, s in pst], pX, psym), (sF, [s for _, s in sst], sX, ssym)

    def discover(self, PL, SL):
        """What a loop reads and writes, children opaque, no proofs."""
        key = (id(PL), id(SL))
        if key in self.disc: return self.disc[key]
        sym = self.sym
        pg0 = ptxexec.run_lines(PL['pre_guard'], sym)
        (pF, pS, pX, psym), (sF, sS, sX, ssym) = self.run_both(PL, SL, {}, {}, sym, sym, False)
        pex = _region_exprs(pF) + stores_exprs(pS) + pX + _region_exprs(pg0)
        sex = _region_exprs(sF) + stores_exprs(sS) + sX + [Bool(f'sass_undef_P{SL["gp"]}')]
        stores = any(s.stores for s in pS + sS) or psym is not sym or ssym is not sym
        d = {'pg0': pg0, 'pF': pF, 'sF': sF,
             'psel': ptx_live_ins(pex), 'ssel': sass_live_ins(sex),
             'pw': writes_p(pF) | writes_p(pg0), 'sw': writes_s(sF),
             'stores': stores}
        self.disc[key] = d
        return d

    def opaque(self, side, PC, SC, seed, sym):
        d = self.discover(PC, SC)
        uid = next(self.uid)
        if side == 'p':
            st = mk_p(sym, seed)
            ex = [p_out(a, st) for a in d['psel']]
            new = dict(seed)
            for k, i in d['pw']:
                new[(k, i)] = fresh_like(p_undef(k, i), f'opq{uid}_p_{k}{i}')
        else:
            st = mk_s(sym, seed)
            ex = [s_out(b, st) for b in d['ssel']]
            new = dict(seed)
            for k in d['sw']:
                new[k] = fresh_like(s_undef(k), f'opq{uid}_s_{k}')
        if d['stores']:
            sym = dict(sym, mem=self.mem(f'opq_{side}'))
        return new, sym, ex

    # ---- one loop, proved in the state its parent reached -------------------
    def prove_loop(self, PL, SL, pseed, sseed, sym):
        d = self.discover(PL, SL)
        psel, ssel, pF0, sF0 = d['psel'], d['ssel'], d['pF'], d['sF']
        pp, sp = mk_p(sym, pseed), mk_s(sym, sseed)
        untouched_p = lambda a: p_out(a, pF0).eq(p_in(a))
        untouched_s = lambda b: s_out(b, sF0).eq(s_in(b))
        carried_p = [a for a in psel if not untouched_p(a)]
        carried_s = [b for b in ssel if not untouched_s(b)]
        # CONFLATION: a carried slot's live-in symbol also naming a value in the
        # entry state would make "this iteration's" and "the parent's" one term.
        ctx = set()
        for v in list(pseed.values()) + list(sseed.values()):
            ctx |= _vars(v)
        names = {(f'ptx_undef_rd{a[1]}' if a[0] in ('rdlo', 'rdhi') else f'ptx_undef_{a[0]}{a[1]}')
                 for a in carried_p} | {f'sass_undef_{"R" if b[0] == "R" else "P"}{b[1]}' for b in carried_s}
        clash = names & ctx
        if clash:
            refuse(f'loop-carried symbol(s) {sorted(clash)} also occur in the state the loop is entered from')
        self.say(f'loop: live-ins ptx {len(psel)} sass {len(ssel)}, carried ptx {len(carried_p)} sass {len(carried_s)}')

        pairs = propose(carried_p, carried_s, psel, ssel, pp, sp, pF0, sF0, self.sym, self.samples)
        if not pairs:
            raise Unproved('random simulation found no corresponding loop-carried values')

        # BASE
        pairs = [(a, b) for a, b in pairs if self.prove(p_out(a, pp) == s_out(b, sp)) == 'unsat']
        if not pairs:
            raise Unproved('no pair survived the base case')

        def seeds(pairs):
            ps = {}
            for a in psel:
                if not untouched_p(a): continue
                if a[0] in ('rdlo', 'rdhi'):
                    if a[1] in pp.rd: ps[('rd', a[1])] = pp.rd[a[1]]
                elif a[1] in getattr(pp, a[0]): ps[(a[0], a[1])] = getattr(pp, a[0])[a[1]]
            ss, used = {}, set()
            for b in ssel:
                if untouched_s(b):
                    key = b[1] if b[0] == 'R' else f'P{b[1]}'
                    v = s_out(b, sp)
                    if not v.eq(s_in(b)): ss[key] = v
            for a, b in pairs:
                key = b[1] if b[0] == 'R' else f'P{b[1]}'
                if key in used: continue
                used.add(key); ss[key] = p_in(a)
            return ps, ss

        neg, pidx = PL['guard_pred']

        def cont_of(ps, psym):
            g = ptxexec.run_lines(PL['pre_guard'], psym, ps).p[pidx]
            return Not(Not(g) if neg else g)

        self.depth += 1
        try:
            for it in range(12):
                ps, ss = seeds(pairs)
                symS = dict(sym, mem=self.mem('step')) if d['stores'] else sym
                (pF, pS, _, psymE), (sF, sS, _, _) = self.run_both(PL, SL, ps, ss, symS, symS, True)
                axioms = mac64.instances(_region_exprs(pF) + stores_exprs(pS), self.sym['mul'])
                cont = cont_of(ps, symS)
                keep = [(a, b) for a, b in pairs
                        if self.prove(p_out(a, pF) == s_out(b, sF), [cont], axioms) == 'unsat']
                if len(keep) == len(pairs): break
                pairs = keep
                if not pairs:
                    raise Unproved('no pair survived the step case')
            else:
                raise Unproved('the relation did not reach a fixpoint in 12 rounds')
        finally:
            self.depth -= 1

        # STORES in one iteration
        pstores = [x for s in pS for x in s.stores]
        sstores = [x for s in sS for x in s.stores]
        self.compare_stores(pstores, sstores, [cont], axioms, 'iteration')
        self.stores += len(pstores)

        # LOOPCOND
        bm = SL['back']
        sneg, spid = bm.group(1) == '!', int(bm.group(2))
        sc = sF.P.get(spid, Bool(f'sass_undef_P{spid}'))
        sass_cont = Not(sc) if sneg else sc
        sub = [(p_in(a), p_out(a, pF)) for a in psel if not p_out(a, pF).eq(p_in(a))]
        nxt = substitute(cont, *sub) if sub else cont
        r = self.prove(sass_cont == nxt, [cont], axioms)
        if r != 'unsat':
            raise Unproved(f'LOOPCOND: back edge vs the next guard: {r}')

        # ENTRY
        ec = sp.P.get(SL['gp'], Bool(f'sass_undef_P{SL["gp"]}'))
        sass_skip = Not(ec) if SL['gneg'] else ec
        gpe = ptxexec.run_lines(PL['pre_guard'], sym, pseed).p[pidx]
        ptx_skip = Not(gpe) if neg else gpe
        r = self.prove(sass_skip == ptx_skip)
        if r != 'unsat':
            raise Unproved(f'ENTRY: zero-trip guards disagree: {r}')
        self.say(f'ok: {len(pairs)} pairs, {len(pstores)} store(s) per iteration')

        # EXIT STATE -- the summary
        uid = next(self.uid)
        shared = {}
        for a, b in pairs:
            key = b[1] if b[0] == 'R' else f'P{b[1]}'
            if key not in shared:
                shared[key] = fresh_like(s_in(b), f'nest{uid}_{b[0]}{b[1]}')
        pexit, sexit = dict(pseed), dict(sseed)
        pmap = {}
        for a, b in pairs:
            pmap.setdefault(a, shared[b[1] if b[0] == 'R' else f'P{b[1]}'])
        for k, i in d['pw']:
            if k == 'rd':
                lo = pmap.get(('rdlo', i), BitVec(f'nest{uid}_p_rdlo{i}', 32))
                hi = pmap.get(('rdhi', i), BitVec(f'nest{uid}_p_rdhi{i}', 32))
                pexit[('rd', i)] = Concat(hi, lo)
            else:
                pexit[(k, i)] = pmap.get((k, i), fresh_like(p_undef(k, i), f'nest{uid}_p_{k}{i}'))
        for k in d['sw']:
            sexit[k] = shared.get(k, fresh_like(s_undef(k), f'nest{uid}_s_{k}'))
        symE = dict(sym, mem=self.mem('exit')) if d['stores'] else sym
        return pexit, sexit, symE

    def compare_stores(self, p, s, extra, axioms, where):
        if len(p) != len(s):
            raise Unproved(f'{where} store counts {len(p)} vs {len(s)}')
        perm = []
        for i, (pa, _, _) in enumerate(p):
            hit = [j for j, (sa, _, _) in enumerate(s) if self.prove(pa == sa, extra, axioms) == 'unsat']
            if len(hit) != 1:
                raise Unproved(f'{where} store {i} address matched {len(hit)} sass stores')
            perm.append(hit[0])
        for i, (_, pv, _) in enumerate(p):
            if pv.size() != s[perm[i]][1].size():
                raise Unproved(f'{where} store {i} width')
        for (i, j), claim in memorder.reorder_obligations(list(p), perm):
            r = self.prove(claim, extra, axioms)
            if r != 'unsat':
                raise Unproved(f'{where} stores {i} and {j} are REORDERED and may overlap: {r}')
        for i, (_, pv, pg) in enumerate(p):
            _, sv, sg = s[perm[i]]
            r = self.prove(pg == sg, extra, axioms)
            if r != 'unsat': raise Unproved(f'{where} store {i} guard: {r}')
            r = self.prove(pv == sv, list(extra) + [pg], axioms)
            if r != 'unsat': raise Unproved(f'{where} store {i} value: {r}')


def validate(ptx_path, sass_path, budget=60, mode='wide', verbose=True):
    PT, ST = ptx_tree(ptx_path), sass_tree(sass_path)
    if not same_shape(PT['root'], ST['root']):
        refuse('the PTX and SASS loop nests have different shapes')
    _, layout = params.parse(ptx_path)
    sym = batch.mk(mulmode.MODES[mode](), layout)
    K = Nest(sym, budget, verbose=verbose)
    pp = ptxexec.run_lines(PT['prologue'], sym)
    sp = sassexec.run_insns(ST['prologue'], sym, 'sass prologue') if ST['prologue'] else mk_s(sym, {})
    for side, st in (('PTX', pp), ('SASS', sp)):
        if st.stores:
            refuse(f'store in the {side} prologue; this validator compares stores in the nest and after it')
        if not is_true(simplify(st.alive)):
            refuse(f'{side} prologue can end the program before the nest')
    try:
        pexit, sexit, symE = K.prove_loop(PT['root'], ST['root'], p_seed_of(pp), s_seed_of(sp), sym)
        pe = ptxexec.run_lines(PT['epilogue'], symE, pexit)
        se = sassexec.run_insns(ST['epilogue'], symE, 'sass epilogue', sexit) if ST['epilogue'] else mk_s(symE, sexit)
        if ST['root']['gkind'] == 'exit' and (pe.stores or se.stores):
            refuse('the SASS program EXITs when the nest runs zero times, and something after it stores')
        K.compare_stores(list(pe.stores), list(se.stores), [], [], 'epilogue')
        K.stores += len(pe.stores)
    except Unproved as e:
        return 'UNPROVED', str(e), K.n
    if not K.stores:
        refuse('this nest stores nothing on either side -- there is nothing to prove equal')
    return 'VALIDATED', f'{K.stores} store(s) across the nest', K.n


if __name__ == '__main__':
    ptx, sass = sys.argv[1], sys.argv[2]
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    t = time.time()
    try:
        v, msg, n = validate(ptx, sass, budget)
    except memorder.Refusal as e:
        v, msg, n = 'REFUSED', str(e).split('\n')[0], 0
    print(f'{v}  {n} obligations  {msg}, {time.time()-t:.1f}s')
