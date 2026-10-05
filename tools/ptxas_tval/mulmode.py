"""The three ways of posing the multiplier, in increasing abstraction."""
from z3 import *
W = 32
def direct(kind, a, b):
    r = a*b if kind=='lo' else Extract(2*W-1, W, ZeroExt(W,a)*ZeroExt(W,b))
    # fold when both operands are literals: `IMAD.MOV.U32 d, RZ, RZ, x` is a
    # move spelled as a multiply, and an unfolded `bvmul 0 0` term severs the
    # dataflow for every consumer downstream of it.
    return simplify(r) if (is_bv_value(a) and is_bv_value(b)) else r
def canon(kind, a, b):  # noqa
    """same, but with the operands of a commutative product in a fixed order,
       so `a*b` on one side and `b*a` on the other become one DAG node"""
    if a.get_id() > b.get_id(): a, b = b, a
    return direct(kind, a, b)

def ordered(a, b):
    """A commutative UF receives the same operand VALUES in either order.

    Sorting only expression IDs does not survive aliases or substitution:
    MUL(a,b) and MUL(b,a+0) can become different UF calls even though their
    concrete products agree. The unsigned value order is a total order of bit
    patterns, so every concrete multiplication remains an interpretation of
    this abstraction. Lexical orientation just makes syntactic swaps share a
    node before the solver runs; it imposes no arithmetic assumption.
    """
    a, b = orient(a, b)
    before = ULE(a, b)
    return If(before, a, b), If(before, b, a)


def orient(a, b):
    """A deterministic order for two operands that does not depend on how the
    terms were built - only on what they ARE.

    `hash()` is z3's STRUCTURAL hash: equal for equal structure, in any context,
    whatever the construction history, and O(1) because the node caches it.
    Comparing `sexpr()` instead - the first version - prints both operands in
    full on every commutative call; in a field kernel those are large shared
    DAGs, and the census of `bn254_g1_add` went from 44 s to more than 17
    minutes. `sexpr` survives only to break a hash TIE between different terms,
    which is rare and keeps the order total."""
    ha, hb = a.hash(), b.hash()
    if ha != hb:
        return (b, a) if ha > hb else (a, b)
    if a.eq(b):
        return a, b
    return (b, a) if a.sexpr() > b.sexpr() else (a, b)

def uf_factory():
    """multiply as an uninterpreted function: SOUND for proving equivalence
       (every concrete model is a model of the abstraction), so an `unsat`
       here is a proof; only a `sat` needs re-checking concretely."""
    LO = Function('MULLO', BitVecSort(W), BitVecSort(W), BitVecSort(W))
    HI = Function('MULHI', BitVecSort(W), BitVecSort(W), BitVecSort(W))
    def f(kind, a, b):
        # 1. both literal -> fold.  `IMAD.MOV.U32 d, RZ, RZ, x` is a MOVE
        #    spelled as 0*0+x; leaving it symbolic severs every consumer.
        if is_bv_value(a) and is_bv_value(b):
            return simplify(direct(kind, a, b))
        if is_bv_value(a): a, b = b, a
        if is_bv_value(b):
            v = b.as_long() & ((1 << W) - 1)
            # 2. literal power of two -> a SHIFT.  This is what `IMAD.SHL.U32`
            #    and `IMAD d, a, 0x100, c` really are, and a shift is also the
            #    one form z3's simplifier will not distribute.
            if v == 0: return BitVecVal(0, W)
            if v & (v - 1) == 0:
                k = v.bit_length() - 1
                if kind == 'lo':
                    return a if k == 0 else a << BitVecVal(k, W)
                return BitVecVal(0, W) if k == 0 else LShR(a, BitVecVal(W - k, W))
            # 3. any other literal (the Montgomery n-prime, the modulus limbs)
            #    stays OPAQUE.  z3's simplifier distributes a literal multiply
            #    over a sum with no option to stop it, and these literals meet
            #    130-term sums: measured 202 -> 7059 nodes in one expression.
        a, b = ordered(a, b)
        return (LO if kind=='lo' else HI)(a, b)
    return f
def wide_factory():
    """The product as ONE uninterpreted 64-bit value, with lo and hi taken as
    halves of it.

    Two independent functions MULLO and MULHI are too weak: the machine
    computes `hi(a*b + C)` in one instruction where the PTX computes
    `lo(a*b)+c` and `hi(a*b)+c'` in two, and relating them needs the fact that
    the two halves belong to ONE 64-bit product.  Independent functions lose
    exactly that.  Concrete bitvector multiplication keeps it and forces the
    solver to bit-blast a multiplier per obligation, which is the other
    extreme.  A shared 64-bit product is the representation that keeps the
    relationship and hides the arithmetic.
    """
    M = Function('MUL64', BitVecSort(W), BitVecSort(W), BitVecSort(2*W))
    def f(kind, a, b):
        if is_bv_value(a) and is_bv_value(b):
            return simplify(direct(kind, a, b))
        if is_bv_value(a): a, b = b, a
        if is_bv_value(b):
            v = b.as_long() & ((1 << W) - 1)
            if v == 0: return BitVecVal(0, W)
            if v & (v - 1) == 0:
                k = v.bit_length() - 1
                if kind == 'lo':
                    return a if k == 0 else a << BitVecVal(k, W)
                return BitVecVal(0, W) if k == 0 else LShR(a, BitVecVal(W - k, W))
        a, b = ordered(a, b)
        p = M(a, b)
        return Extract(W-1, 0, p) if kind == 'lo' else Extract(2*W-1, W, p)
    return f

MODES = {'direct': lambda: direct, 'canon': lambda: canon,
         'uf': uf_factory, 'wide': wide_factory}
