"""The u32 division ESTIMATE, as the SASS executor models it.

ptxas lowers `div.u32`/`rem.u32` to

    I2F.U32.RP  f, d                 # d as a float, rounded toward +inf
    MUFU.RCP    f, f                 # approximate reciprocal
    IADD3       f, f, 0xffffffe, RZ  # bias the BIT PATTERN
    F2I.FTZ.U32.TRUNC.NTZ e, f       # back to an integer: the estimate e

followed by ordinary integer code: a Newton step and two corrections.  The four
instructions above are float work a bitvector model cannot express, and
matching the whole lowering as `UDiv` would ASSUME it correct -- the thing
under validation.  So the executor does neither.  It models the chain as ONE
fresh 32-bit value `e` per occurrence.  The one fact it relies on is

  LEMMA A  I-1 <= e + hi(e * lo(-d*e)) <= I,    I = floor(2^32/d),  d != 0

measured EXHAUSTIVELY over all 2^32-1 non-zero divisors by `divlow_abi.py`
(`newton_abi.cu`), which also asserts that the probe's SASS carries this chain
instruction for instruction against the corpus kernel.  The Newton term is a
function of `e` alone, so the fact is a claim about the real estimate whatever
the SASS then does with it -- and it is stated about the SASS's OWN Newton
value only after a solver has PROVED that value equal to `newton(e, d)`
(`Sass.est_transfer`).  Nothing about the TAIL is assumed: `intenc.py`
discharges it (`unknown` over bitvectors on six posings, `unsat` over Int).

The WINDOW (`e <= I`, `(I-e)^2 <= I`, also measured) is NOT assumed.  An earlier
version recorded it, with Lemma A on its own spelling beside it, and the
mutation table showed neither was ever what a proof used: removing both left
every row unchanged.  A fact that looks load-bearing and is not is removed
rather than kept.  Deriving Lemma A from the window by solver is not available
-- `unknown` at 300 s over Int, as over bitvectors -- which is exactly why
Lemma A is measured rather than proved.

At d = 0 the estimate is garbage and nothing is claimed; the lowering
selects its answer by an explicit `d != 0` test, which the executor models.

The intermediates are TAGGED, not values: any instruction other than the next
link of this chain reading one is a refusal by name, because the facts were
measured for the chain and for nothing else.  A predicated link is refused for
the same reason.
"""
from z3 import *


class Tagged:
    STAGES = ('i2f', 'rcp', 'bias')

    def __init__(self, stage, d):
        assert stage in self.STAGES
        self.stage, self.d = stage, d

    def __repr__(self):
        return f'<division estimate, after {self.stage}>'


BIAS = 0x0ffffffe


def newton(e, d):
    """The measured Newton step, e + hi(e * lo(-d*e)), as a 32-bit term."""
    C = d.ctx
    t = (BitVecVal(0, 32, C) - d) * e
    return e + Extract(63, 32, ZeroExt(32, e) * ZeroExt(32, t))


def lemma_a(e2, d):
    """I-1 <= e2 <= I, I = floor(2^32/d), for d != 0 -- about ANY term e2 that
    has been PROVED equal to newton(e, d); see Sass.est_transfer."""
    C = d.ctx
    I64 = UDiv(BitVecVal(1 << 32, 64, C), ZeroExt(32, d))
    x = ZeroExt(32, e2)
    return Implies(d != BitVecVal(0, 32, C),
                   And(ULE(I64 - BitVecVal(1, 64, C), x), ULE(x, I64)))


def _self_check():
    """Lemma A must hold where it should and BITE where it should not, in a
    private context: at e = I the Newton step lands in [I-1, I]; an estimate
    far below I lands outside it."""
    C = Context()
    for dv in (1, 3, 7, 1000, 0x80000001, 0xffffffff):
        I = (1 << 32) // dv
        for ev, ok in ((min(I, 0xffffffff), True), (I // 2, False)):
            if ev == I // 2 and I < 16:
                continue      # a small I leaves no room below it to violate
            e, d = BitVecVal(ev, 32, C), BitVecVal(dv, 32, C)
            got = is_true(simplify(lemma_a(newton(e, d), d)))
            if got != ok:
                raise AssertionError(f'divest Lemma A at d={dv:#x} e={ev:#x}: {got}, want {ok}')


_self_check()
