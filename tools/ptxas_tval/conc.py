"""A concrete evaluator over the z3 AST our executors build.

Random simulation is how a translation validator FINDS the correspondence
between two programs' intermediate values; the solver then proves the pairs
it proposes.  Walking the DAG with a memo is linear, where model evaluation
per expression is not.

An unhandled node kind raises.  A concrete evaluator that guesses would
produce a correspondence that is silently wrong, which is worse than none.
"""
from z3 import *
K = Z3_OP_TRUE
M32 = (1<<32)-1
class Conc:
    def __init__(self, env, mulmask=M32):
        self.env = env; self.memo = {}; self._keep = []
    def ev(self, e):
        # THE MEMO KEYS ON e.get_id() AND z3 REUSES AST IDS after a node is
        # garbage-collected.  A caller that builds a TEMPORARY expression --
        # `c.ev(Extract(31,0,x))` in a loop -- frees it before the next one is
        # built, the next node lands on the same id, and the memo returns the
        # PREVIOUS answer.  Measured: three distinct expressions all evaluated
        # to the first one's value.
        #
        # Holding a reference here rather than asking every caller to hold one
        # is the fix: a caller cannot be expected to know that building a
        # temporary corrupts someone else's cache.  The executors' own callers
        # happen to be safe (they pass expressions held in `st.wide`), so this
        # is a latent hazard rather than a wrong result -- but a wrong random
        # simulation costs proposals, and a validator that silently proposes
        # nothing looks exactly like one whose kernel is too hard.
        i = e.get_id()
        if i in self.memo: return self.memo[i]
        v = self._ev(e); self.memo[i] = v; self._keep.append(e); return v
    def _ev(self, e):
        d = e.decl(); k = d.kind(); ch = e.children()
        if k == Z3_OP_BNUM:  return e.as_long()
        if k == Z3_OP_TRUE:  return True
        if k == Z3_OP_FALSE: return False
        if k == Z3_OP_UNINTERPRETED and not ch:
            n = d.name()
            if n in self.env: return self.env[n]
            # unconstrained: deterministic pseudo-value, stable across samples
            h = (hash(n) ^ self.env.get('__salt__', 0)) & M32
            self.env[n] = (h if is_bv(e) else bool(h & 1)); return self.env[n]
        w = e.size() if is_bv(e) else None
        m = (1 << w) - 1 if w else None
        a = [self.ev(c) for c in ch]
        if k == Z3_OP_BADD:  return sum(a) & m
        if k == Z3_OP_BSUB:  return (a[0] - a[1]) & m
        if k == Z3_OP_BMUL:
            r = 1
            for x in a: r *= x
            return r & m
        if k == Z3_OP_BAND:
            r = m
            for x in a: r &= x
            return r
        if k == Z3_OP_BOR:
            r = 0
            for x in a: r |= x
            return r
        if k == Z3_OP_BXOR:
            r = 0
            for x in a: r ^= x
            return r
        if k == Z3_OP_BNOT:  return (~a[0]) & m
        if k == Z3_OP_BSHL:  return (a[0] << a[1]) & m if a[1] < 512 else 0
        if k == Z3_OP_BLSHR: return (a[0] >> a[1]) if a[1] < 512 else 0
        if k == Z3_OP_BASHR:
            sgn = -(a[0] >> (w-1)) & m                    # 0 or all ones
            if a[1] >= w: return sgn
            return ((a[0] >> a[1]) | (sgn << max(0, w - a[1]))) & m
        if k == Z3_OP_SIGN_EXT:
            iw = ch[0].size()
            v = a[0] - (1 << iw) if (a[0] >> (iw-1)) & 1 else a[0]
            return v & m
        if k == Z3_OP_EXTRACT:
            hi, lo = d.params()
            return (a[0] >> lo) & ((1 << (hi-lo+1)) - 1)
        if k == Z3_OP_ZERO_EXT: return a[0]
        if k == Z3_OP_CONCAT:
            r = 0
            for c, v in zip(ch, a): r = (r << c.size()) | v
            return r
        if k == Z3_OP_ITE:   return a[1] if a[0] else a[2]
        if k == Z3_OP_NOT:   return not a[0]
        if k == Z3_OP_AND:   return all(a)
        if k == Z3_OP_OR:    return any(a)
        if k == Z3_OP_EQ:    return a[0] == a[1]
        if k == Z3_OP_DISTINCT: return a[0] != a[1]
        if k == Z3_OP_ULT:   return a[0] <  a[1]
        if k == Z3_OP_ULEQ:  return a[0] <= a[1]
        if k == Z3_OP_UGT:   return a[0] >  a[1]
        if k == Z3_OP_UGEQ:  return a[0] >= a[1]
        if k == Z3_OP_SELECT:
            arr = ch[0].decl().name()
            return self.env['__mem__'](arr, a[1])
        if k == Z3_OP_UNINTERPRETED:
            n = d.name()
            if n == 'MULLO': return (a[0]*a[1]) & M32
            if n == 'MULHI': return ((a[0]*a[1]) >> 32) & M32
            if n == 'MUL64': return (a[0]*a[1]) & ((1<<64)-1)
            raise Exception(f'unmodelled uninterpreted function {n}')
        raise Exception(f'unmodelled AST node kind {k} ({d.name()})')
