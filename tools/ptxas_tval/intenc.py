"""An EXACT translation of a bitvector formula into integer arithmetic.

Why it exists.  The u32 division lowering's tail -- `q0 = HI(e2*n)`, `r0 = n -
d*q0` and two corrections -- was posed to z3 six ways over bitvectors with the
divisor symbolic and every one came back `unknown` at budgets up to 1200 s.
Posed over Int with the u32 wraps written out, the same question is `unsat` in
0.2 s, and its controls (e2 = I-2, e2 = I+1, the second correction removed) are
`sat` in 0.1 s.  The wall was the THEORY, not the arithmetic: the bitvector
engine bit-blasts two composed 32x32 products, where the integer engine reasons
about their bounds directly.  See docs/ptxas_translation_validation.md.

The translation is EXACT, not an abstraction: a w-bit value x becomes an integer
in [0, 2^w), and every operation that can leave that range is wrapped by a
fresh quotient variable (`r = x - k*2^w, 0 <= r < 2^w`), which pins r to
`x mod 2^w` without asking the solver for `mod`.  So the integer formula is
satisfiable exactly when the bitvector one is, and an `unsat` here is a proof
of the bitvector formula.  That is what lets `tval.py` use it as one more rung
of its ladder without weakening anything.

What it does NOT translate it REFUSES (raises `Unsupported`), never
approximates: a variable shift, a bitwise operation other than a low-bit mask,
an array other than a bare `Select` of an uninterpreted array, and any operator
not listed below.  A refusal makes the rung answer `unknown`, which is the
safe answer.

Everything is built in the CONTEXT OF THE TERM IT IS GIVEN, and the import-time
self-check runs in a private one: z3 reuses freed node ids and `mulmode`/
`fpmode` order commutative operands by id, so an import that allocates in the
main context reorders operands in unrelated kernels' terms (measured once
already in this directory, through `memorder`'s self-check).
"""
from z3 import *


class Unsupported(Exception):
    pass


class Encoder:
    def __init__(self, ctx):
        self.ctx = ctx
        self.side = []          # range and wrap constraints, all must hold
        self.memo = {}
        self.vars = {}
        self.funcs = {}
        self.n = 0
        # EVERY node whose id is used as a memo key is kept alive here.  z3
        # reuses a freed node's id, so a key whose node was collected can be
        # handed to a DIFFERENT term later, which then inherits the wrong
        # encoding -- a false `sat` or, worse, a false `unsat`.  The first
        # version of this file keyed on ids of simplified formulas it did not
        # retain, and the same obligation answered `sat` in one run and
        # `unknown` in the next.
        self.keep = []
        # An UPPER BOUND per encoded term (keyed by its id; the term is kept
        # alive above).  Wrapping an operation that cannot overflow adds a
        # quotient variable the solver must prove zero -- and a 32x32 product
        # carried in 64 bits, the shape every `hi(a*b)` has, never overflows.
        # With a wrap on each of those the division tail was `unknown` at 120 s
        # where the hand-written posing without them is `unsat` in 0.2 s.
        self.ub = {}
        # DEFERRED REDUCTION.  For every wrap variable r = x mod 2^w, `pre` keeps
        # x.  An operation of the SAME width that consumes r may consume x
        # instead and wrap once at the end, because (x mod 2^w) * y, x + y, -x
        # are all congruent to their unreduced forms mod 2^w.  Without it
        # `d*(0-q0)` became `d*r` with r a wrap of -q0, and the division tail --
        # `unsat` in 0.1 s when written with `d*q0` -- was `unknown` at 60 s in
        # every spelling the SASS uses.
        self.pre = {}

    # -- helpers ---------------------------------------------------------------
    def fresh(self, tag):
        self.n += 1
        return Int(f'_ie{self.n}_{tag}', self.ctx)

    def I(self, v):
        return IntVal(v, self.ctx)

    def bound(self, x):
        if is_int_value(x):
            return max(x.as_long(), 0)
        return self.ub[x.get_id()]

    def note(self, x, b):
        self.keep.append(x)
        self.ub[x.get_id()] = b
        return x

    def wrap(self, x, w, lower_ok=True):
        """x mod 2^w, as a fresh r with r = x - k*2^w and 0 <= r < 2^w -- or x
        itself when x is known to lie in [0, 2^w) already.  `lower_ok` says
        the caller knows x >= 0 (a sum or product of in-range terms)."""
        if is_int_value(x) :
            return self.I(x.as_long() % (1 << w))
        if lower_ok and self.bound(x) < (1 << w):
            return x
        k = self.fresh('k'); r = self.fresh('r')
        self.side += [r == x - k * self.I(1 << w), r >= 0, r < self.I(1 << w)]
        self.note(r, (1 << w) - 1)
        self.pre[r.get_id()] = (x, w)
        return r

    def unreduced(self, a, w):
        """a, or the expression a is the w-bit reduction of.  A CONSTANT at or
        above 2^(w-1) comes back as its signed representative c - 2^w: z3
        spells -q0 as `bvmul #xffffffff q0`, and (2^32-1)*q0*d needs a quotient
        of size q0*d to reduce where -q0*d needs one of 0 or 1."""
        if is_int_value(a):
            v = a.as_long()
            return self.I(v - (1 << w)) if v >= (1 << (w - 1)) else a
        p = self.pre.get(a.get_id())
        return p[0] if p is not None and p[1] == w else a

    def floordiv(self, x, m, tag):
        """floor(x / m) for a positive integer constant m, as a fresh q."""
        q = self.fresh(tag)
        self.side += [q * self.I(m) <= x, x < (q + 1) * self.I(m)]
        return self.note(q, self.bound(x) // m)

    def product(self, a, b, w):
        """(a*b) mod 2^w, HASH-CONSED on the encoded operands.  Two spellings
        of one product -- `(0-d)*e` against `(-d)*e`, a 64-bit extract of a
        product against a 65-bit carry sum -- would otherwise get two unrelated
        wrap variables, and relating them is exactly the nonlinear reasoning
        the integer engine is bad at.  Keying on the ENCODED operands (after
        their own normalisation) makes one product one variable."""
        if is_int_value(a) and is_int_value(b):
            return self.I((a.as_long() * b.as_long()) % (1 << w))
        self.keep += [a, b]
        ka, kb = sorted((a.get_id(), b.get_id()))
        key = ('mul', ka, kb, w)
        r = self.memo.get(key)
        if r is None:
            ua, ub_ = self.unreduced(a, w), self.unreduced(b, w)
            if ua is a and ub_ is b:
                r = self.wrap(self.note(a * b, self.bound(a) * self.bound(b)), w)
            else:
                r = self.wrap(ua * ub_, w, lower_ok=False)
            self.memo[key] = r
        return r

    def ranged(self, name, w):
        v = self.vars.get(name)
        if v is None:
            v = Int(f'_iv_{name}', self.ctx)
            self.vars[name] = v
            self.note(v, (1 << w) - 1)
            self.side += [v >= 0, v < self.I(1 << w)]
        return v

    # -- the translation -------------------------------------------------------
    def bv(self, t):
        key = t.get_id()
        if key in self.memo:
            return self.memo[key]
        self.keep.append(t)
        r = self._bv(t)
        self.memo[key] = r
        self.keep.append(r)
        return r

    def _bv(self, t):
        w = t.size()
        M = 1 << w
        if is_bv_value(t):
            return self.I(t.as_long())
        k = t.decl().kind()
        ch = t.children()
        if k == Z3_OP_UNINTERPRETED:
            if not ch:
                return self.ranged(str(t), w)
            return self.uf(t, ch)
        if k == Z3_OP_SELECT:
            arr, idx = ch
            if arr.decl().kind() != Z3_OP_UNINTERPRETED or arr.children():
                raise Unsupported(f'select from a non-bare array {arr.sexpr()[:60]}')
            return self.uf(t, [idx], name='sel_' + str(arr))
        if k == Z3_OP_ITE:
            a, b = self.bv(ch[1]), self.bv(ch[2])
            return self.note(If(self.bool(ch[0]), a, b), max(self.bound(a), self.bound(b)))
        if k == Z3_OP_BADD:
            xs = [self.bv(c) for c in ch]
            us = [self.unreduced(x, w) for x in xs]
            if all(u is x for u, x in zip(us, xs)):
                return self.wrap(self.note(Sum(xs), sum(self.bound(x) for x in xs)), w)
            return self.wrap(Sum(us), w, lower_ok=False)
        if k == Z3_OP_BSUB:
            a, b = [self.unreduced(self.bv(c), w) for c in ch]
            return self.wrap(a - b, w, lower_ok=False)
        if k == Z3_OP_BNEG:
            return self.wrap(-self.unreduced(self.bv(ch[0]), w), w, lower_ok=False)
        if k == Z3_OP_BMUL:
            p = self.bv(ch[0])
            for c in ch[1:]:
                p = self.product(p, self.bv(c), w)
                self.keep.append(p)
            return p
        if k == Z3_OP_BNOT:
            return self.note(self.I(M - 1) - self.bv(ch[0]), M - 1)
        if k == Z3_OP_EXTRACT:
            hi, lo = t.params()
            x = self.bv(ch[0])
            if lo:
                x = self.floordiv(x, 1 << lo, 'ex')
            return self.wrap(x, hi - lo + 1) if hi - lo + 1 < ch[0].size() - lo else x
        if k == Z3_OP_CONCAT:
            acc = self.bv(ch[0])
            for c in ch[1:]:
                acc = acc * self.I(1 << c.size()) + self.bv(c)
            return self.note(acc, M - 1)
        if k == Z3_OP_ZERO_EXT:
            return self.bv(ch[0])
        if k == Z3_OP_SIGN_EXT:
            x = self.bv(ch[0]); w0 = ch[0].size()
            return self.note(If(x >= self.I(1 << (w0 - 1)), x + self.I(M - (1 << w0)), x), M - 1)
        if k in (Z3_OP_BUDIV, Z3_OP_BUDIV_I, Z3_OP_BUREM, Z3_OP_BUREM_I):
            a, b = [self.bv(c) for c in ch]
            # ONE quotient per (dividend, divisor): URem is n - d*q with the SAME
            # q as UDiv, so a proved quotient says something about the remainder.
            key = ('divq', a.get_id(), b.get_id(), w)
            q = self.memo.get(key)
            if q is None:
                self.keep += [a, b]
                q = self.note(self.fresh('q'), M - 1)
                self.side += [Implies(b > 0, And(q * b <= a, a < (q + 1) * b, q >= 0))]
                self.memo[key] = q
            if k in (Z3_OP_BUDIV, Z3_OP_BUDIV_I):
                return self.note(If(b == 0, self.I(M - 1), q), M - 1)
            return self.note(If(b == 0, a, a - b * q), M - 1)
        if k in (Z3_OP_BSHL, Z3_OP_BLSHR):
            if not is_bv_value(ch[1]):
                raise Unsupported('a variable shift amount')
            s = ch[1].as_long()
            x = self.bv(ch[0])
            if s >= w:
                return self.I(0)
            if k == Z3_OP_BSHL:
                return self.wrap(self.note(x * self.I(1 << s), self.bound(x) << s), w)
            return self.floordiv(x, 1 << s, 'sh')
        if k == Z3_OP_BAND:
            # only a mask of low bits against a constant: x & (2^j - 1)
            consts = [c for c in ch if is_bv_value(c)]
            others = [c for c in ch if not is_bv_value(c)]
            if len(consts) == 1 and len(others) == 1:
                m = consts[0].as_long()
                if m & (m + 1) == 0:
                    j = m.bit_length()
                    return self.wrap(self.bv(others[0]), j) if j < w else self.bv(others[0])
            raise Unsupported('a bitwise AND other than a low-bit mask')
        raise Unsupported(f'bitvector operator {t.decl().name()}')

    def uf(self, t, args, name=None):
        name = name or t.decl().name()
        doms = tuple(a.sort() for a in args)
        key = (name, doms)
        f = self.funcs.get(key)
        if f is None:
            isorts = []
            for s in doms:
                if s.kind() == Z3_BV_SORT: isorts.append(IntSort(self.ctx))
                elif s.kind() == Z3_BOOL_SORT: isorts.append(BoolSort(self.ctx))
                else: raise Unsupported(f'uninterpreted argument sort {s}')
            f = Function(f'_if_{name}', *isorts, IntSort(self.ctx))
            self.funcs[key] = f
        ia = [self.bv(a) if a.sort().kind() == Z3_BV_SORT else self.bool(a) for a in args]
        v = f(*ia)
        self.side += [v >= 0, v < self.I(1 << t.size())]
        return self.note(v, (1 << t.size()) - 1)

    def bool(self, t):
        key = ('b', t.get_id())
        if key in self.memo:
            return self.memo[key]
        self.keep.append(t)
        r = self._bool(t)
        self.memo[key] = r
        return r

    def _bool(self, t):
        if is_true(t): return BoolVal(True, self.ctx)
        if is_false(t): return BoolVal(False, self.ctx)
        k = t.decl().kind(); ch = t.children()
        if k == Z3_OP_AND: return And([self.bool(c) for c in ch])
        if k == Z3_OP_OR: return Or([self.bool(c) for c in ch])
        if k == Z3_OP_NOT: return Not(self.bool(ch[0]))
        if k == Z3_OP_IMPLIES: return Implies(self.bool(ch[0]), self.bool(ch[1]))
        if k == Z3_OP_XOR: return Xor(self.bool(ch[0]), self.bool(ch[1]))
        if k == Z3_OP_ITE: return If(self.bool(ch[0]), self.bool(ch[1]), self.bool(ch[2]))
        if k in (Z3_OP_EQ, Z3_OP_DISTINCT):
            if ch[0].sort().kind() == Z3_BV_SORT:
                xs = [self.bv(c) for c in ch]
            elif ch[0].sort().kind() == Z3_BOOL_SORT:
                xs = [self.bool(c) for c in ch]
            else:
                raise Unsupported(f'equality over {ch[0].sort()}')
            return (xs[0] == xs[1]) if k == Z3_OP_EQ else Distinct(*xs)
        U = {Z3_OP_ULEQ: lambda a, b: a <= b, Z3_OP_ULT: lambda a, b: a < b,
             Z3_OP_UGEQ: lambda a, b: a >= b, Z3_OP_UGT: lambda a, b: a > b}
        if k in U:
            return U[k](self.bv(ch[0]), self.bv(ch[1]))
        S = {Z3_OP_SLEQ: lambda a, b: a <= b, Z3_OP_SLT: lambda a, b: a < b,
             Z3_OP_SGEQ: lambda a, b: a >= b, Z3_OP_SGT: lambda a, b: a > b}
        if k in S:
            w = ch[0].size(); H = self.I(1 << (w - 1)); M = self.I(1 << w)
            sv = [If(self.bv(c) >= H, self.bv(c) - M, self.bv(c)) for c in ch]
            return S[k](sv[0], sv[1])
        if k == Z3_OP_UNINTERPRETED and not ch:
            return Bool(f'_ib_{t}', self.ctx)
        raise Unsupported(f'boolean operator {t.decl().name()}')


def check(formulas, budget_s, want_model=False, rewrite=True):
    """Satisfiability of the conjunction of bitvector-sorted boolean formulas,
    decided over Int.  Returns 'sat' / 'unsat' / 'unknown'; an untranslatable
    formula is 'unknown', never a guess."""
    if not formulas:
        return 'sat'
    ctx = formulas[0].ctx
    enc = Encoder(ctx)
    try:
        # z3's own rewriter first: it canonicalises `0-d` against `-d`, operand
        # order, and extract/concat of products, so equal values tend to become
        # equal NODES before the translation sees them.  It is an equivalence
        # rewrite, so the translation stays exact.
        body = [enc.bool(simplify(f) if rewrite else f) for f in formulas]
    except Unsupported:
        return 'unknown'
    s = SolverFor('QF_NIA', ctx=ctx) if not enc.funcs else SolverFor('QF_UFNIA', ctx=ctx)
    s.set('timeout', int(budget_s * 1000))
    s.add(enc.side); s.add(body)
    r = str(s.check())
    if want_model:
        # the model of every INPUT, as bitvector values, so a caller can check a
        # counterexample against the bitvector formula it came from
        m = s.model() if r == 'sat' else None
        return r, ({n: m.eval(v).as_long() for n, v in enc.vars.items()} if m else None)
    return r


def _self_check():
    """Concrete agreement with the bitvector semantics on every supported
    operator, at edge values, in a PRIVATE context.  An encoder that is wrong
    about one operator makes every obligation using it prove or fail for the
    wrong reason, and no kernel row isolates which."""
    C = Context()
    a, b = BitVecs('a b', 32, C)
    x16 = BitVec('x16', 16, C)
    terms = [a + b, a - b, -a, a * b, ~a, Extract(63, 32, ZeroExt(32, a) * ZeroExt(32, b)),
             Extract(15, 0, a), Extract(31, 8, a), Concat(x16, Extract(15, 0, b)),
             ZeroExt(8, x16), SignExt(16, x16), UDiv(a, b), URem(a, b), a << 3, LShR(a, 5),
             a & 0xff, If(ULT(a, b), a, b), If(a < b, a, b),
             # the deferred-reduction, signed-constant and shared-quotient paths
             (a - b) * b, -a * b, BitVecVal(0xffffffff, 32, C) * a * b,
             (a + b) * (a - b), BitVecVal(0x80000001, 32, C) * a + b,
             URem(a, b) + UDiv(a, b) * b, (0 - a) * (0 - b) - a]
    vals = [0, 1, 2, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff, 12345, 0xd2470f5d]
    for t in terms:
        for va in vals:
            for vb in (0, 1, 7, 0x80000000, 0xffffffff):
                sub = [(a, BitVecVal(va, 32, C)), (b, BitVecVal(vb, 32, C)),
                       (x16, BitVecVal(va & 0xffff, 16, C))]
                want = simplify(substitute(t, *sub))
                # the encoding of t must be FORCED to the concrete value
                f = And(a == va, b == vb, x16 == (va & 0xffff), t != want)
                # BOTH with and without z3's rewriter: after `simplify`, `a - b`
                # is `a + 0xffffffff*b` and `-a` is `0xffffffff*a`, so the
                # BSUB/BNEG arms are reached ONLY unrewritten.  The mutation
                # table reversed the subtraction and nothing noticed, because
                # this check only ever asked about rewritten terms.
                for rw in (True, False):
                    r = check([f], 5, rewrite=rw)
                    if r != 'unsat':
                        raise AssertionError(f'intenc disagrees with z3 on {t} at a={va:#x} '
                                             f'b={vb:#x} (rewrite={rw}): {r}')
    # and the encoding must not be vacuously unsat
    if check([a * b == BitVecVal(6, 32, C), ULT(a, 3)], 5) != 'sat':
        raise AssertionError('intenc: a satisfiable formula came back unsat')
    # an unsupported operator is REFUSED, not approximated
    if check([a << b == 1], 5) != 'unknown':
        raise AssertionError('intenc: a variable shift was translated instead of refused')


_self_check()
