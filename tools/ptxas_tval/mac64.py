"""One multiplier identity, discharged OUTSIDE the solver and named.

    zext64(a) * sext64(b)  ==  zext64(a)*zext64(b) + ((a * sign(b)) << 32)

WHY IT IS HERE.  The PTX writes a 64-bit multiply of two extended 32-bit values
(`mul.lo.s64` on a `cvt.u64.u32` and a `cvt.s64.s32`); ptxas splits it into
`IMAD.WIDE.U32` plus one `IMAD` carrying the sign correction.  Relating the two
is this identity and nothing else.

WHY IT IS NOT LEFT TO THE SOLVER.  Measured: z3 proves it at 8 bits in 0.8 s and
returns `unknown` at 16 bits and at 32, at a 60 s budget.  So the multiply wall
this programme keeps meeting is not a matter of kernel size -- here it is a
two-line algebraic law that cannot be bit-blasted at sixteen bits.

WHY THIS IS NOT CIRCULAR, which is the question to ask of it.  Encoding ptxas's
INSTRUCTION SEQUENCE into the model of the source would make the validator
unable to see an error in that sequence -- the same trap as composing FFMA into
FADD(FMUL(..)) on the float side.  This is different in kind: it is an algebraic
law of two's-complement arithmetic, stated over abstract operands, and ptxas
happens to use the same law.  Applying a proved law to put both sides in one
normal form is what a translation validator does; copying the compiler's output
into the specification is not.

WHAT IT COSTS.  It is an ASSUMPTION at 32 bits, so it is a trust-boundary item
and `loopval` reports every instance it used rather than absorbing it silently.
Evidence, from `mac64/probe.cu` on an RTX 4070 Ti SUPER: 4,194,304 cases --
432 structured edge pairs (0, 1, 2^31-1, 2^31, -1, -2, 0xFFFF0000, ...) crossed
with three accumulators (0, all-ones, a mixed pattern), the rest random --
0 mismatches against an independent __int128 host oracle, 0 unwritten slots.
Sampled evidence for the semantics, not a proof of it; the 8-bit case IS proved.
"""
from z3 import *

PROVENANCE = ('zext64(a)*sext64(b) == zext64(a)*zext64(b) + ((a*sign(b))<<32); '
              'z3-proved at 8 bits, unknown at 16 and 32; device-validated over '
              '4,194,304 cases incl. structured edges, 0 mismatches')

def _widens(e, mk):
    """Is this 64-bit term the widening `mk` of its own low half?

    Asked SEMANTICALLY rather than by matching `Z3_OP_ZERO_EXT`.  z3's
    simplifier rewrites `ZeroExt(32,a)` to `Concat(0,a)` and `SignExt(32,b)` to
    a 33-way `Concat` of the sign bit, so a syntactic matcher fires on the term
    as written and never on the term as it reaches the solver -- which is how
    the first version of this file silently instantiated nothing at all.  The
    query is a pure bit-level one with no multiply in it, so it is cheap."""
    if e.size() != 64: return None
    lo = Extract(31, 0, e)
    s = Solver(); s.set('timeout', 4000)
    s.add(e != mk(32, lo))
    return lo if s.check() == unsat else None

def instances(exprs, mul=None):
    """Every instance of the identity that the given terms actually contain.

    Only these are handed to the solver -- a universally quantified axiom would
    be sound too and would put a quantifier in front of a bitvector problem,
    which is the thing to avoid.

    THE RIGHT-HAND SIDE IS BUILT FROM THE MODEL'S OWN MULTIPLY PRIMITIVE, which
    is Phase A's representation rule applied to an axiom.  Written with a
    concrete `*` it bridges nothing under the `wide` posing, where the SASS side
    holds `MUL64(a,b)`: the two would be different terms and the identity would
    sit there instantiated and useless -- which is what the first wiring of it
    did.  Written through `mul` it constrains `MUL64` at exactly the operand
    pair in play, which is sound (the true product is a model of the
    abstraction) and is the refinement the obligation needs."""
    seen, out, done = set(), [], set()
    def walk(e, depth=0):
        if e.get_id() in seen or depth > 400: return
        seen.add(e.get_id())
        if e.decl().kind() == Z3_OP_BMUL and e.size() == 64 and e.num_args() == 2:
            x, y = e.arg(0), e.arg(1)
            for u, v in ((x, y), (y, x)):
                a = _widens(u, ZeroExt)
                if a is None: continue
                b = _widens(v, SignExt)
                if b is None: continue
                key = (a.get_id(), b.get_id())
                if key in done: break
                done.add(key)
                sgn = If(Extract(31, 31, b) == BitVecVal(1, 1),
                         BitVecVal(0xffffffff, 32), BitVecVal(0, 32))
                if mul is None:
                    wide = ZeroExt(32, a) * ZeroExt(32, b)
                    corr = a * sgn
                else:
                    wide = Concat(mul('hi', a, b), mul('lo', a, b))
                    corr = mul('lo', a, sgn)
                out.append(e == wide + Concat(corr, BitVecVal(0, 32)))
                break
        for c in e.children(): walk(c, depth + 1)
    for e in exprs: walk(e)
    return out
