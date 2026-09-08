"""Floating point as uninterpreted functions over the 32-bit pattern.

SOUNDNESS.  Every float operation becomes an uninterpreted function of its
operands' bit patterns.  That is an ABSTRACTION, and every real interpretation
-- IEEE 754 round-to-nearest included -- is a model of it.  So if the two
sides' terms are provably equal under it, they are equal in IEEE: `unsat` is
a proof, exactly as for the `wide` multiply rung.

A `sat` is NOT a proof of divergence.  It says the abstraction cannot show
them equal, which may mean they genuinely differ or only that it is too
coarse.  `FFMA(a,b,c)` and `FADD(FMUL(a,b),c)` are distinct terms here; they
are also distinct FUNCTIONS, but that has to be established outside the
abstraction -- by refinement, or numerically.

This suffices whenever ptxas TRANSLITERATES the float operations.  Where it
CONTRACTS them the terms differ, and reporting that is the correct answer
rather than a limitation of the model.

TWO CLASSES, AND ONLY ONE IS A CONTRACTION.  Measured over the 67 committed
kernels: 9 are pure CONTRACTION (ptxas fuses `mul.f32`+`add.f32` into FFMA -- a
permitted freedom, repaired by writing `.rn`, cost 0 to +7.1% instructions), and
17 carry a MACRO-OP -- `div`/`rcp`/`sqrt`/`rsqrt`/`ex2`/`sin`/`cos`, a single
PTX instruction with an exact IEEE meaning that ptxas implements as a MUFU seed
plus a Newton-Raphson refinement.  `div.rn.f32` alone becomes 144 SASS
instructions with 18 branches and 3 reconvergence regions.  No PTX spelling
makes ptxas transliterate it; `.rn` IS the exact one and is what produces the
sequence.  So the macro-op class is not a float-semantics gap, it is a
different verification problem -- proving an IEEE division algorithm -- and it
is REFUSED BY NAME here rather than approximated.

TWO IDENTIFICATIONS ARE IMPOSED, each licensed by a device probe and each
recorded with the flag that says so -- `FADD` commutes, and `FSUB(a,b)` is
`FADD(a, FNEG(b))`.  Both were once written here as deliberately open, and both
were closed only when something needed them, which is the order to do it in: an
identification nothing needs is an unmeasured assumption with no benefit.

Deliberately NOT assumed: `FMUL` commutativity.  Nothing has needed it, so
nothing has measured it, and `_self_check` pins BOTH halves of every
identification -- the one that must hold and the one that must not -- so a
third cannot arrive without a probe.
"""
from z3 import *
W = 32

# --- the float problem is THREE classes, measured per opcode ------------------
#
# `expand.py`: compile ONE PTX float op at sm_89 -O3 against a mov-only baseline
# of 24 instructions, and count.  The mnemonic does not predict the class; the
# rounding modifier does.
#
#   TRANSLIT   +0   sin.approx cos.approx cvt.rn.f32.s32 cvt.f32.f16
#   SMALL      +8   ex2 lg2 rcp rsqrt div sqrt .approx        <- and `mul.f32`
#                   is ALSO +8, so these cost exactly what an ordinary multiply
#                   costs: a MUFU plus a special-value guard.
#   EXPANDED  +40   sqrt.rn.f32
#             +72   rcp.rn.f32
#            +120   div.rn.f32, div.rn.f64
#
# The boundary is `.approx` vs `.rn`, and it is the boundary between "the
# hardware has this instruction" and "ptxas must SYNTHESISE a correctly-rounded
# result".  Every EXPANDED op emits BSSY/BSYNC and `CALL.REL.NOINC` -- an
# out-of-line subroutine -- so it is behind branch and call support in any case,
# and validating it bit-exactly is a proof about an IEEE division algorithm
# rather than a term match.  Refused by name.
#
# --- and the shortcut this file exists to make impossible ---------------------
#
# For a TRANSLIT/SMALL op the cheapest move is to model the PTX op and the MUFU
# ptxas lowers it to as the SAME uninterpreted function.  That is defensible --
# it is the same trade as `IMAD.HI.U32 = hi32(a*b)`, a semantics assumed and
# settled on the device -- but it is only defensible ONCE THE DEVICE HAS SETTLED
# IT, and it is FALSE if extended to an EXPANDED op.  Measured
# (div/mufu_vs_exact, RTX 4070 Ti SUPER, 2^20 finite normal inputs, outputs
# poisoned, 0 unwritten):
#
#     rcp.approx.f32 (= MUFU.RCP)  != rcp.rn.f32   on 13.23% of inputs
#     div.approx.f32               != div.rn.f32   on 27.30% of inputs
#     div.rn.f32 == correctly-rounded double quotient on 100.00%
#     worst observed MUFU.RCP error: 0.97 ulp
#
# Both arms of the differential share this factory, so an identification made
# for convenience makes every kernel using it validate for the wrong reason, and
# `_self_check` cannot see it: the UFs are perfectly distinct and the
# unsoundness is in WHICH one the executor calls.  This repository has a name
# for that shape (`feedback-differential-arms-share-constants`).
#
# So a PTX op may reach a hardware primitive only through this table, and only
# with `validated` recording that a device probe settled it.
HARDWARE_PRIMITIVES = frozenset((
    'MUFU_SIN', 'MUFU_COS', 'MUFU_EX2', 'MUFU_LG2', 'MUFU_RCP', 'MUFU_RSQ',
    'MUFU_RCP64H', 'MUFU_SQRT',
))

# ptx opcode -> (class, extra SASS instructions vs a mov-only baseline,
#                the primitive it may be identified with, device-validated?)
EXPANDED = {
    'rcp.rn.f32': 72, 'sqrt.rn.f32': 40, 'div.rn.f32': 120,
    'div.rn.f64': 120, 'rcp.rn.f64': None, 'sqrt.rn.f64': None,
    'div.rnd.f32': None,
}
TRANSLITERATED = {
    # ptx opcode          primitive       device-validated?
    'sin.approx.f32':   ('MUFU_SIN',      False),
    'cos.approx.f32':   ('MUFU_COS',      False),
    'ex2.approx.f32':   ('MUFU_EX2',      False),
    'ex2.approx.ftz.f32': ('MUFU_EX2',    False),
    'lg2.approx.f32':   ('MUFU_LG2',      False),
    'rcp.approx.f32':   ('MUFU_RCP',      False),
    'rsqrt.approx.f32': ('MUFU_RSQ',      False),
    'sqrt.approx.f32':  ('MUFU_SQRT',     False),
}
MACRO_OPS = dict.fromkeys(list(EXPANDED) + list(TRANSLITERATED))

# --- identifications licensed by a device probe --------------------------------
#
# An identification is an EQUATION imposed on the abstraction, so it is exactly
# as load-bearing as an opcode model and it drifts the same way.  Each one is
# recorded here with the flag that says a probe settled it, and the factory
# REFUSES to impose an unvalidated one -- otherwise the flag is a comment.
#
#   FSUB(a,b) == FADD(a, FNEG(b))
#     `sub.f32` has exactly one lowering -- `FADD Ra, -Rb` -- so "is FSUB the
#     same function as FADD-of-a-negation" cannot be asked of ptxas without
#     asking the translator under test.  It is asked of the DEVICE instead, as
#     two PTX programs: `sub.f32 a b` against `add.f32 a t` where `t` is `b`
#     with its sign bit flipped by an integer XOR against a mask LOADED FROM
#     MEMORY.  The mask has to be opaque: with a literal 0x80000000 ptxas
#     recognises the xor as a negation and folds it back into the modifier, so
#     both arms become the same instruction and the probe compares a kernel
#     against itself.  Measured 32/32 identical on sm_89 over denormals of both
#     signs, +0.0/-0.0, quiet NaNs carrying payloads, a signalling NaN, both
#     infinities and inf+(-inf).
IDENTIFICATIONS = {
    'FSUB_IS_FADD_OF_FNEG': True,
}


def refuse_macro_op(op):
    """Refuse a PTX macro-op BY NAME, with the class and the reason.

    A named refusal says `this is out of scope and here is why`; the generic
    unmodelled-opcode message reads like a missing case somebody should add --
    and for the EXPANDED class adding it is exactly the wrong move."""
    if op in EXPANDED:
        n = EXPANDED[op]
        size = f', which ptxas expands to +{n} SASS instructions' if n else ''
        raise Exception(
            f'EXPANDED MACRO-OP {op!r}{size}  (refusing, not guessing).  ptxas '
            f'synthesises a correctly-rounded result from a MUFU seed and a '
            f'Newton-Raphson refinement, with BSSY/BSYNC and an out-of-line '
            f'CALL.REL.NOINC.  Bit-exact validation of it is a proof about an '
            f'IEEE algorithm, not a term match.  Identifying it with its MUFU '
            f'seed is UNSOUND: they differ on 13-27% of inputs, measured on the '
            f'device.')
    prim, validated = TRANSLITERATED[op]
    raise Exception(
        f'TRANSLITERATED MACRO-OP {op!r}  (refusing, not guessing).  ptxas '
        f'lowers it 1:1 to {prim}, so modelling both as one uninterpreted '
        f'function would be sound -- but only once a DEVICE PROBE has settled '
        f'that they are the same function, and that has not been run. '
        f'Set TRANSLITERATED[{op!r}] validated=True when it has.')


def primitive_for(op):
    """The hardware primitive a PTX op may be identified with, or a refusal.
    The ONLY route by which the PTX side may reach a MUFU."""
    if op not in TRANSLITERATED:
        refuse_macro_op(op)
    prim, validated = TRANSLITERATED[op]
    if not validated:
        refuse_macro_op(op)
    return prim


def factory():
    F2 = {n: Function(n, BitVecSort(W), BitVecSort(W), BitVecSort(W))
          for n in ('FMUL', 'FADD', 'FSUB')}
    F3 = {n: Function(n, BitVecSort(W), BitVecSort(W), BitVecSort(W), BitVecSort(W))
          for n in ('FFMA',)}
    # the MUFU entries are DERIVED from HARDWARE_PRIMITIVES rather than listed
    # again: two lists of the same thing drift, which is the shape of bug this
    # whole file is about.
    F1 = {n: Function(n, BitVecSort(W), BitVecSort(W))
          for n in (tuple(sorted(HARDWARE_PRIMITIVES)) +
                    ('I2F_S32', 'I2F_U32', 'F2F_F16_F32', 'F2F_F32_F16', 'FNEG'))}
    # FSUB is deliberately NOT a symbol: `f` rewrites it below, so deleting that
    # rewrite is a HARD ERROR at the first `sub.f32` rather than a silent revert
    # to two terms that no longer meet.
    def f(name, *args, side, _via_table=False):
        # COMMUTATIVITY, for FADD only, and licensed by a measurement.
        #
        # ptxas does not preserve the operand order of an add: it SORTS the
        # addends by register number, so `add.rn.f32 d, acc, prod` comes back as
        # `FADD d, prod, acc` whenever prod is in the lower register.  That was
        # the last thing between `naive_gemm_f32` and a result, and it is the
        # question this file used to record as deliberately open.
        #
        # "IEEE addition is commutative" is not enough on its own, because the
        # claim being made is about stored BITS and IEEE leaves a NaN result's
        # payload implementation defined -- a hardware that returned the FIRST
        # operand's payload would break this on exactly the inputs no ordinary
        # test uses.  `fpsem_abi.py` runs both operand orders on the device (two
        # cubins, because ptxas CSEs the two orders inside one kernel) over two
        # quiet NaNs with different payloads, a signalling NaN, inf+(-inf), both
        # zeros and denormals, and they agree bit for bit.  Measured on sm_89.
        #
        # FMUL is NOT canonicalised: nothing has needed it, so nothing has
        # measured it, and an unmeasured identification is the guess this file
        # exists to refuse.  `_self_check` pins both halves of that.
        # SUBTRACTION IS ADDITION OF A NEGATION, and licensed by a measurement.
        #
        # The PTX side writes `sub.f32`; ptxas writes `FADD Ra, -Rb`, with the
        # negation as an OPERAND MODIFIER.  Without this the two sides build
        # different terms and no kernel containing a float subtract can ever
        # validate -- and the failure reads as a divergence in the arithmetic
        # rather than as a modelling gap, which is the worst way to be told.
        #
        # `-R` itself is modelled as FNEG on both sides; `fpsem_abi.py` measures
        # that the modifier is a BIT-EXACT sign flip, by folding a `neg.f32`
        # into an FSEL source -- FSEL being the one float-shaped instruction
        # already refereed as bit-exact, so the modifier is observed with
        # nothing arithmetic in the way.
        if name == 'FSUB':
            if len(args) != 2:
                raise Exception(f'FSUB/{len(args)} -- the identification is binary')
            if not IDENTIFICATIONS['FSUB_IS_FADD_OF_FNEG']:
                raise Exception('FSUB is identified with FADD(a, FNEG(b)) but the '
                                'device probe that settles it is marked unvalidated')
            return f('FADD', args[0], f('FNEG', args[1], side=side), side=side)
        if name == 'FADD' and len(args) == 2 and args[0].get_id() > args[1].get_id():
            args = (args[1], args[0])
        if side not in ('ptx', 'sass'):
            raise Exception(f'float op {name!r} asked for without a side'
                            f' -- the caller must say which program it is executing')
        if side == 'ptx' and name in HARDWARE_PRIMITIVES and not _via_table:
            raise Exception(
                f'the PTX side reached the hardware primitive {name!r} directly. '
                f'A PTX macro-op is not the primitive ptxas seeds it with; the '
                f'identification is a semantic claim that a device probe has to '
                f'settle, so it must go through primitive_for().')
        t = {1: F1, 2: F2, 3: F3}[len(args)]
        if name not in t:
            raise Exception(f'UNMODELLED FLOAT OP {name!r}/{len(args)}  (refusing, not guessing)')
        return t[name](*args)
    _self_check(f)
    _side_check(f)
    return f


def _self_check(f):
    """The abstraction must not collapse distinct operations, and it must not
    ignore its arguments.

    BOTH SIDES OF THE DIFFERENTIAL SHARE THIS FACTORY, so a degenerate one
    makes every float kernel validate trivially -- the two stores become the
    same term for the wrong reason.  That is not hypothetical: two mutations
    demonstrated it (every op mapped to a single UF; the result replaced by a
    constant), and both PASSED the whole kernel suite while the model said
    nothing at all.  Neither mutation is visible in any per-kernel assertion,
    because the property they break is a property of the abstraction rather
    than of any one program.
    """
    a, b, c = (BitVec(n, W) for n in ('_fpchk_a', '_fpchk_b', '_fpchk_c'))
    two = ('FMUL', 'FADD', 'FSUB')
    for i, n in enumerate(two):
        for m in two[i+1:]:
            if is_true(simplify(f(n, a, b, side='sass') == f(m, a, b, side='sass'))):
                raise Exception(f'float abstraction collapses {n} and {m} '
                                f'-- both arms share it, so this validates everything')
        if is_true(simplify(f(n, a, b, side='sass') == f(n, c, b, side='sass'))) or \
           is_true(simplify(f(n, a, b, side='sass') == f(n, a, c, side='sass'))):
            raise Exception(f'float abstraction {n} ignores an argument '
                            f'-- both arms share it, so this validates everything')
    if is_true(simplify(f('FFMA', a, b, c, side='sass') == f('FADD', f('FMUL', a, b, side='sass'), c, side='sass'))):
        raise Exception('float abstraction identifies FFMA with FADD(FMUL(..)) '
                        '-- that is exactly the contraction under test')
    # The commutativity above is exactly one identification, and both halves of
    # that have to be pinned or it drifts: FADD must commute (or the operand
    # order ptxas rewrites is not covered and no float loop kernel validates),
    # and FMUL must NOT (or an identification nothing measured has crept in).
    if not is_true(simplify(f('FADD', a, b, side='sass') == f('FADD', b, a, side='sass'))):
        raise Exception('FADD no longer commutes -- ptxas SORTS the addends, so '
                        'without this no float kernel whose add it reorders can '
                        'be validated')
    if is_true(simplify(f('FMUL', a, b, side='sass') == f('FMUL', b, a, side='sass'))):
        raise Exception('FMUL now commutes -- nothing has needed that, so nothing '
                        'has measured it; see fpsem_abi.py for what licensing one '
                        'of these costs')
    # FNEG, and both halves again.  It must be a real function -- an identity
    # FNEG makes `FADD Ra, -Rb` and `FADD Ra, Rb` the same term, so a kernel
    # whose SASS negates the wrong source validates -- and the FSUB rewrite must
    # actually fire, or the PTX and SASS sides of every float subtract build
    # terms that cannot meet and no such kernel can ever be validated.
    if is_true(simplify(f('FNEG', a, side='sass') == a)):
        raise Exception('FNEG is the identity -- a negated source is then the same '
                        'term as the source, so a kernel that negates the wrong '
                        'operand validates')
    if is_true(simplify(f('FNEG', a, side='sass') == f('FNEG', b, side='sass'))):
        raise Exception('FNEG ignores its argument -- both arms share it, so this '
                        'validates everything')
    if not is_true(simplify(f('FSUB', a, b, side='sass') ==
                            f('FADD', a, f('FNEG', b, side='sass'), side='sass'))):
        raise Exception('FSUB is no longer rewritten to FADD(a, FNEG(b)) -- ptxas '
                        'lowers `sub.f32` to `FADD Ra, -Rb`, so without the rewrite '
                        'the two sides of every float subtract build terms that '
                        'cannot meet')
    # ...and it must be a rewrite rather than a collapse: `a - b` is not `a + b`.
    if is_true(simplify(f('FSUB', a, b, side='sass') == f('FADD', a, b, side='sass'))):
        raise Exception('FSUB collapsed onto FADD -- FNEG has stopped doing anything')


def _side_check(f):
    """The side split must actually hold, and a macro-op must actually refuse.

    Without this the guard is a comment: someone removes the `side` test, every
    per-kernel assertion still passes, and 17 kernels start validating because
    both arms call one UF for two different functions.
    """
    a = BitVec('_fpchk_a', W)
    for n in HARDWARE_PRIMITIVES:
        try:
            f(n, a, side='ptx')
        except Exception:
            pass
        else:
            raise Exception(f'the PTX side reached {n!r} without going through the '
                            f'table -- the identification is no longer a claim '
                            f'anything has to settle, and every kernel using it '
                            f'now validates for the wrong reason')
        f(n, a, side='sass')          # and the SASS side must still be able to
    # an EXPANDED op must refuse whatever anyone does to the table
    for op in ('div.rn.f32', 'rcp.rn.f32', 'sqrt.rn.f32'):
        try:
            primitive_for(op)
        except Exception as e:
            if 'EXPANDED' not in str(e):
                raise Exception(f'{op} is no longer refused as EXPANDED: {e}')
        else:
            raise Exception(f'{op} is no longer refused -- it emits an out-of-line '
                            f'CALL and cannot be a term match')
    # and an unvalidated identification must refuse too, or the table is a comment
    # an identification whose probe has not been run must refuse, for the same
    # reason: the flag is a comment unless something reads it.
    # SAVE and RESTORE, never assign the flag back to True: a check that puts a
    # constant back is a check that repairs the mutation it was written to find.
    _was = IDENTIFICATIONS['FSUB_IS_FADD_OF_FNEG']
    IDENTIFICATIONS['FSUB_IS_FADD_OF_FNEG'] = False
    try:
        f('FSUB', a, a, side='sass')
    except Exception:
        pass
    else:
        raise Exception('FSUB was identified with FADD(a, FNEG(b)) although the '
                        'validated flag says no device probe has settled it')
    finally:
        IDENTIFICATIONS['FSUB_IS_FADD_OF_FNEG'] = _was
    unval = [o for o, (_, v) in TRANSLITERATED.items() if not v]
    if unval:
        try:
            primitive_for(unval[0])
        except Exception:
            pass
        else:
            raise Exception(f'{unval[0]} was identified with its primitive although '
                            f'no device probe has settled that they agree')


# Run both checks AT IMPORT, not only inside factory().  A check that fires only
# where it is called is disarmed by deleting one line that reads as incidental;
# at import there is no call site to remove, only the check itself.  Every entry
# point (tval, batch, scope) imports this module.
_probe = factory()
