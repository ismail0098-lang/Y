"""Is there ANY legal (head_dim, context, K width, V width) that breaks exactness?

    python3 tools/exact_bounds_check.py

The whole determinism claim rests on one sentence: *every partial sum stays
below the point where its float type starts rounding.* That is not a statistical
margin, it is a proof obligation, and it is currently discharged by four bounds
scattered across two files, each derived by hand:

    A  score sum, fp32:      d * Q_LEVELS * k_lv          <  2^24
    B  digit matmul, fp32:   T * (2^dbits - 1) * V_LEVELS <  2^24
    C  W*V accumulator, f64: T * 2^P_BITS * V_LEVELS      <  2^53
    D  softmax denominator:  T * 2^P_BITS                 <  2^53
    E  wide-activation acc:  K * (a_levels + 128) * 127    <  2^31
    F  digit representable:  a_levels                      <= 16383

Every one is easy to check by hand *in isolation*, and that is exactly why this
tool exists. The failure mode is not a wrong bound, it is an **interaction**: a
configuration where one bound's escape hatch fires and another's does not, so
the code takes a path it believes is exact and is not. `k_levels_for` narrows K
as head_dim grows; `exact_pv` narrows its digits as T grows and falls back to
float64 when it runs out. Two adaptive mechanisms, four bounds, three
env-settable widths - the conjunction is where a hand derivation stops being
trustworthy, and it is decidable integer arithmetic, so Z3 settles it.

## What is proved and what is only tested

Z3 answers questions about the **derivation**: given the spec each helper claims
to meet, do the bounds hold for every input? That is a proof over an infinite
domain, and it is the part hand-checking cannot do.

It says nothing about whether `k_levels_for` and `exact_pv`'s digit loop
actually meet those specs - the same model-vs-code gap `proofs/ZkControlFlow.v`
states about itself. That half is closed by **exhaustion over the real Python
functions**, which is complete rather than sampled: `k_levels_for` is a pure
function of one small integer, and `exp2_neg_q16_16`'s claimed range is checked
over its entire reachable input domain (~2M values), not spot-checked.

The `p <= 2^28` claim is the one worth singling out. Bounds C and D both rest on
it, it is asserted nowhere, and it is a property of a twelve-step integer
polynomial - the kind of thing a docstring says and nobody checks.
"""
import _nospace

_nospace.guard()

import sys  # noqa: E402

import torch  # noqa: E402
import z3  # noqa: E402

import batch_invariance_demo as D  # noqa: E402
import exact_kv  # noqa: E402

TWO24 = 1 << 24
TWO53 = 1 << 53

# Domains. Deliberately far wider than anything shipped: the point of asking a
# solver is to cover configurations nobody has run yet.
MAX_D = 4096            # head_dim; real models are 64..256
MAX_T = 1 << 22         # 4M context
MAX_LEVELS = 1 << 20    # a width nobody would set, included so the answer is
                        # about the arithmetic and not about my taste in widths

fails = []


def report(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'BAD'}] {name}" + (f"\n         {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------------------
# The violation predicates, extracted.
#
# The width rule was written out TWICE - once in `exhaust_k_levels_for` and
# again inside the off-axis sweep - which is the shape that produced the
# `zk_emitter` constant-folding family: a rule with two implementations gets
# fixed in one of them. It is one function now, and `--check-sweeps` feeds it
# values that must be rejected.
#
# The reason that matters here specifically: every sweep below reports "0
# violations over N configurations", and 0 is also what a predicate that cannot
# fire reports. Extracting them is what makes that checkable at all.
# ---------------------------------------------------------------------------

def width_violation(d, q, k, budget=TWO24, floor=127):
    """Is this (head_dim, q_levels, k_levels) triple outside the spec?

    Returns a reason string, or None if it is fine. Pure: no torch, no module
    state, so `--check-sweeps` can hand it whatever it likes.
    """
    if k < 1:
        return "non-positive width"
    if (k + 1) & k:
        return "not 2^n-1"
    if d * q * k >= budget and k != floor:
        return "over budget and not the floor"
    return None


def digit_violation(t, dbits, v, budget=TWO24, pbits=None):
    """Does `exact_pv`'s digit split stay exact at this key length?

    `dbits == 0` is the float64 fallback and is governed by bound C elsewhere,
    so it is not this predicate's business; the caller skips it.
    """
    pbits = D.P_BITS if pbits is None else pbits
    if dbits < 1:
        return "no digit width"
    if t * ((1 << dbits) - 1) * v >= budget:
        return "digit product exceeds the fp32 exact-integer budget"
    # There WAS a second branch here, `ceil(29/dbits)*dbits < 29`, described as
    # "the shift loop must cover all 29 bits of p". It is unreachable for every
    # dbits >= 1 -- `ceil(a/b)*b >= a` is an identity -- so it had never fired
    # and never could. Extracting this predicate so `--check-sweeps` could feed
    # it a violation is what surfaced that; there was no violation to feed.
    #
    # It was also checking the wrong thing. Coverage is not a property of the
    # formula, it is a property of `exact_pv`'s loop CONDITION, and recomputing
    # the formula here compares the transcription against itself. The real
    # guard is `differential_exact_pv`, which drives `p` to exactly 2^P_BITS
    # and compares against an integer reference -- a behavioural check that
    # fails if the loop stops one digit early.
    return None


def check_sweep_predicates():
    """Can the sweeps' predicates FIRE? No z3, no torch, no GPU.

        python3 tools/exact_bounds_check.py --check-sweeps

    Every exhaustive check in this file reports "0 violations over N
    configurations". `0` is also what a predicate that cannot fire reports, and
    that is not hypothetical here: `digit_violation` shipped with a branch
    (`ceil(29/dbits)*dbits < 29`) that is false for every dbits, so one of the
    two things this file said it checked about the digit loop had never been
    checked at all. Writing these cases is what found it - there was no
    violation to feed it.

    Each case below is a value the predicate MUST reject, one per reason it can
    return, plus the near-misses that must still be accepted.
    """
    cases = [
        # (d, q, k, expected reason or None)
        (128, 127, 255, None),                    # ordinary, well inside budget
        (128, 127, 256, "not 2^n-1"),             # a width that is not 2^n-1
        (128, 127, 0, "non-positive width"),
        (128, 127, -1, "non-positive width"),
        (4096, 127, 511, "over budget and not the floor"),
        (4096, 127, 127, None),                   # the floor IS allowed
        # The budget boundary, straddled. `d*q*k` of 2^24-1 is fine and 2^24 is
        # not, so these two differ by ONE in the product. Both were wrong in the
        # first version of this list -- I wrote 2^24-1 as an example of "not a
        # 2^n-1" when it is exactly one, and 2*(2^23-1) as over budget when it
        # is one short. The predicate was right and the test was wrong, which is
        # the direction you want a self-test to fail in.
        (1, 1, (1 << 24) - 1, None),                 # product = 2^24 - 1, ok
        (2, 1, (1 << 24) - 1, "over budget and not the floor"),   # 2*(2^24-1)
        (1, 3, 8388607, "over budget and not the floor"),         # 3*(2^23-1)
        (1, 1, 8388607, None),
    ]
    bad = 0
    print("\n--- can width_violation fire? ---")
    for d, q, k, want in cases:
        got = width_violation(d, q, k)
        okc = got == want
        bad += not okc
        print(f"  {'ok  ' if okc else 'WRONG'}  d={d:<5} q={q:<4} k={k:<9} "
              f"-> {got!r} (want {want!r})")

    print("\n--- can digit_violation fire? ---")
    dcases = [
        (100, 7, 127, None),
        (1 << 20, 7, 127,
         "digit product exceeds the fp32 exact-integer budget"),
        (100, 0, 127, "no digit width"),
        # The boundary: T*(2^dbits-1)*V just under and just over 2^24.
        (TWO24 // (127 * 127) - 1, 7, 127, None),
        (TWO24 // (127 * 127) + 2, 7, 127,
         "digit product exceeds the fp32 exact-integer budget"),
    ]
    for t, db, v, want in dcases:
        got = digit_violation(t, db, v)
        okc = got == want
        bad += not okc
        print(f"  {'ok  ' if okc else 'WRONG'}  T={t:<10} dbits={db:<3} v={v:<5} "
              f"-> {got!r} (want {want!r})")

    if bad:
        print(f"\n{bad} case(s) wrong: a sweep predicate does not do what the "
              f"sweeps report about it.")
        return 1
    print(f"\nboth predicates reject every violation and accept every "
          f"near-miss ({len(cases) + len(dcases)} cases).")
    print("The sweeps' '0 violations' therefore means something.")
    return 0


# ---------------------------------------------------------------------------
# Part 1: Z3 over the derivations.
# ---------------------------------------------------------------------------

def z3_derivations():
    print("\n--- Z3: are the derivations sound for every legal input? ---")

    d, q, k_lv, t, v, dbits_pow = z3.Ints("d q k_lv T V two_to_dbits")
    legal_d = z3.And(d >= 1, d <= MAX_D)
    legal_t = z3.And(t >= 1, t <= MAX_T)
    legal_v = z3.And(v >= 1, v <= MAX_LEVELS)
    legal_q = z3.And(q >= 1, q <= MAX_LEVELS)

    # A. `k_levels_for` computes `cap = 2^24 // (d*q)` and returns a width no
    # larger. Is `k_lv <= cap` on its own enough for bound A?
    #
    # **No** - and this is the check that nearly went in as a tautology. The
    # bound is STRICT (`< 2^24`) while the cap is a floor, so whenever `d*q`
    # divides 2^24 exactly, `k_lv = cap` gives `d*q*k_lv = 2^24` on the nose.
    # The rounding down to a `2^n - 1` is what saves it, and that is a
    # load-bearing line rather than a cosmetic one. Ask both forms so the
    # difference is on the record.
    s = z3.Solver()
    s.add(legal_d, legal_q, k_lv >= 1)
    s.add(k_lv <= TWO24 / (d * q))          # z3 `/` on Ints is floor division
    s.add(z3.Not(d * q * k_lv < TWO24))
    loose = s.check()
    report("A: `k_lv <= floor(2^24/(d*q))` alone is NOT enough",
           loose == z3.sat,
           "the cap is a floor and the bound is strict, so d*q | 2^24 puts the "
           "score at exactly 2^24; the `2^n - 1` rounding is what closes it")

    # ... and the rounding alone is not enough either. Z3's counterexample is
    # `k_lv = 1` at `d*q = 2^24`: 1 is `2^1 - 1`, it satisfies the cap, and the
    # product lands on 2^24 exactly. It is the ONLY violating width - `k_lv >= 3`
    # forces `d*q <= 5,592,405` and hence `d*q*k_lv <= 2^24 - 1` - so what
    # actually closes the bound is the `max(127, ...)` floor, and only because
    # 127 > 1. Both mechanisms are load-bearing and neither is sufficient alone.
    s = z3.Solver()
    s.add(legal_d, legal_q, k_lv >= 127)                          # the floor
    s.add(k_lv <= TWO24 / (d * q))
    s.add(z3.Or(*[k_lv == (1 << n) - 1 for n in range(7, 25)]))   # ... and 2^n-1
    s.add(z3.Not(d * q * k_lv < TWO24))
    report("A: rounding AND the 127 floor together ARE enough",
           s.check() == z3.unsat,
           "the only width that can hit 2^24 exactly is k_lv=1, which the floor "
           "excludes; neither mechanism suffices on its own")

    # ... but the spec has an escape hatch: `max(127, ...)`. When the budget
    # cannot even hold int8, the function returns 127 anyway rather than 0. Is
    # that reachable, and how large must head_dim be?
    s = z3.Solver()
    s.add(legal_d, q == D.Q_LEVELS)
    s.add(d * q * 127 >= TWO24)
    ok = s.check() == z3.sat
    if ok:
        s2 = z3.Optimize()
        s2.add(legal_d, d * D.Q_LEVELS * 127 >= TWO24)
        s2.minimize(d)
        s2.check()
        dmin = s2.model()[d].as_long()
        report("A': the K-width floor of 127 IS reachable (fail-closed expected)",
               True, f"first violating head_dim = {dmin}; `exact_attention`'s "
                     f"assert must refuse there, checked below")
    else:
        report("A': floor unreachable", False, "expected it to be reachable")

    # B. The digit loop exits when the bound holds or when dbits hits 0. Model
    # the exit state: either two_to_dbits >= 2 and the bound holds, or dbits==0.
    s = z3.Solver()
    s.add(legal_t, legal_v, dbits_pow >= 2)
    s.add(t * (dbits_pow - 1) * v < TWO24)          # the loop's exit condition
    s.add(z3.Not(t * (dbits_pow - 1) * v < TWO24))  # bound B
    report("B: the digit loop's exit condition IS bound B",
           s.check() == z3.unsat)

    # C is the reason this file exists, and the answer was not the one I
    # predicted. The question: is there a configuration where the DIGIT path
    # succeeds - so the code believes every matmul is exact - while the float64
    # accumulator that recombines the digits silently passes 2^53?
    #
    # **UNSAT, and that is a structural guarantee rather than luck.** The digit
    # loop needs T*(2^dbits - 1)*V < 2^24 and with dbits >= 1 that already forces
    # T*V < 2^24, while the accumulator only fails at T*V >= 2^25. So the digit
    # bound is strictly 2x tighter and *subsumes* the accumulator bound: taking
    # the digit path is by itself a proof that the recombination is exact. I had
    # expected to find a window here and to report it as a documented ceiling.
    s = z3.Solver()
    s.add(legal_t, legal_v, dbits_pow >= 2)
    s.add(t * (dbits_pow - 1) * v < TWO24)          # digit path taken
    s.add(t * (1 << D.P_BITS) * v >= TWO53)         # accumulator NOT exact
    report("C: the digit path can NEVER overflow the accumulator",
           s.check() == z3.unsat,
           "bound B is 2x tighter than bound C, so reaching the digit path "
           "proves the recombination is exact too")

    # Which leaves exactly one route into the unsafe region: the float64
    # fallback taken when dbits hits 0. Is that window real, and how wide?
    s = z3.Solver()
    s.add(legal_t, legal_v)
    s.add(t * 1 * v >= TWO24)                       # dbits driven to 0
    s.add(t * (1 << D.P_BITS) * v < TWO53)          # ... and C still holds
    live = s.check() == z3.sat
    report("C': the float64 fallback covers exactly one octave of T",
           live,
           "dbits hits 0 at T >= 2^24/V and C fails at T >= 2^25/V, so the "
           "fallback is live on a 2x window and then NOTHING is exact - that "
           "second edge is the exact path's hard context ceiling")

    # ... and it is reachable, so the ceiling must be refused rather than
    # computed. Smallest violating T at each shipped width:
    for vv in (127, 511, 1023):
        limit = TWO53 // ((1 << D.P_BITS) * vv)
        print(f"         V={vv:<5} -> exact path is limited to T <= {limit:,} tokens")

    # D. Implied by C whenever V >= 1 - state it rather than assume it.
    s = z3.Solver()
    s.add(legal_t, legal_v)
    s.add(t * (1 << D.P_BITS) * v < TWO53)
    s.add(z3.Not(t * (1 << D.P_BITS) < TWO53))
    report("D: the denominator bound is implied by the accumulator bound",
           s.check() == z3.unsat)

    # The conjunction, at the shipped constants: is there any (d, T) with
    # realistic widths where a bound fails and no assert covers it? Every assert
    # in the path is modelled, so an unsat here is the real statement.
    s = z3.Solver()
    s.add(legal_d, legal_t, q == D.Q_LEVELS, v == exact_kv.V_LEVELS)
    s.add(k_lv >= 1, k_lv * d * q <= TWO24 - 1)     # k_levels_for honoured its budget
    s.add(t * (1 << D.P_BITS) * v < TWO53)          # assert_exact_range passed
    s.add(z3.Or(z3.Not(d * q * k_lv < TWO24),
                z3.Not(t * (1 << D.P_BITS) < TWO53)))
    report("conjunction: no legal config passes every assert and still rounds",
           s.check() == z3.unsat)


# ---------------------------------------------------------------------------
# Part 2: exhaustion over the real code, closing the model-vs-code gap.
# ---------------------------------------------------------------------------

def z3_activation_width():
    """Bound E: the wide-activation GEMM's int32 accumulator.

    Added in M5 step 16. The kernel accumulates `128*(hi@w)` and `(lo@w)` into
    ONE int32, interleaved across K blocks, so the quantity that must fit is the
    sum of the two parts' bounds and not the bound on the value they reconstruct.
    Getting that wrong is loose by 128 levels and would only ever show up as a
    silently wrapped accumulator at some large K.
    """
    print("\n--- Z3: the wide-activation accumulator (bound E) ---")
    import exact_model as EM

    k, a, w = z3.Ints("K a_levels w_levels")
    legal = z3.And(k >= 1, k <= 1 << 20, a >= 1, a <= MAX_LEVELS,
                   w >= 1, w <= MAX_LEVELS)

    # The naive budget is NOT sufficient - state it, so the tightening is on the
    # record as necessary rather than as belt-and-braces.
    s = z3.Solver()
    s.add(legal)
    s.add(k * a * w < EM.ACC_BUDGET)                        # naive spec
    s.add(z3.Not(k * (a + 128) * w < EM.ACC_BUDGET))        # what must hold
    report("E: the naive `k*a*w` budget is NOT sufficient",
           s.check() == z3.sat,
           "the accumulator holds 128*(hi@w) + (lo@w) interleaved, so the sum of "
           "the parts' bounds is what must fit, not the reconstructed value's")

    # The floor cap alone is still not sufficient, for EXACTLY the reason bound
    # A was not: `floor(B/(k*w)) - 128` against a STRICT `<` lands on the budget
    # when `k*w` divides it. The `2^n - 1` rounding closes it here too, which is
    # a second independent instance of that lesson rather than a coincidence -
    # every one of these caps is a floor and every one of these bounds is strict.
    s = z3.Solver()
    s.add(legal)
    s.add(a <= EM.ACC_BUDGET / (k * w) - 128)               # cap only
    s.add(z3.Not(k * (a + 128) * w < EM.ACC_BUDGET))
    report("E: the floor cap alone is NOT sufficient", s.check() == z3.sat,
           "same shape as bound A: a floor cap cannot meet a strict bound when "
           "k*w divides the budget")

    s = z3.Solver()
    s.add(legal)
    s.add(a <= EM.ACC_BUDGET / (k * w) - 128)
    s.add(z3.Or(*[a == (1 << n) - 1 for n in range(7, 21)]))  # ... plus rounding
    s.add(z3.Not(k * (a + 128) * w < EM.ACC_BUDGET))
    report("E: cap plus the power-of-two rounding IS sufficient",
           s.check() == z3.unsat)

    # And the implementation. `act_levels_for` has the same `max(127, ...)` floor
    # as `k_levels_for`, so at absurd K it returns a width the budget cannot
    # afford - and then `split_act` is False and the PLAIN int8 path runs, whose
    # bound is `K*127^2`. So the right question is not "does the width fit" but
    # "does the bound of the path actually taken fit", and that is where this
    # found something: the plain path wraps at K >= 133,153 and nothing said so.
    bad, widths, first_refused = [], {}, None
    for kk in [1] + [1 << i for i in range(1, 21)] + [896, 4864, 11008, 28672]:
        lv = EM.act_levels_for(kk, requested=1 << 20)
        if (lv + 1) & lv:
            bad.append((kk, lv, "not 2^n-1"))
        bound = kk * (lv + 128) * 127 if lv > 127 else kk * 127 * 127
        if bound >= EM.ACC_BUDGET and first_refused is None:
            first_refused = kk
        if kk in (896, 4864, 11008, 28672):
            widths[kk] = lv
    report("act_levels_for returns a power-of-two width at every K tried",
           not bad, f"violations {bad[:4]}" if bad else f"K -> width: {widths}")

    # F. The digit-REPRESENTABILITY bound, which is not implied by E and was
    # the blind spot that let a real bug through. `q = hi*128 + lo` puts `hi` in
    # an int8, so `|q| <= 127*128 + 127 = 16383` regardless of what the
    # accumulator could afford - and at K=64 the accumulator affords 262,143.
    # `torch`'s `.to(torch.int8)` WRAPS rather than raising, so exceeding it is
    # a silent wrong answer. Two independent ceilings on one quantity; the
    # smaller has to be applied explicitly.
    over = [kk for kk in [1] + [1 << i for i in range(1, 21)]
            if EM.act_levels_for(kk, requested=1 << 20) > EM.DIGIT_MAX]
    report(f"F: no K yields a width past DIGIT_MAX ({EM.DIGIT_MAX})", not over,
           f"{len(over)} K values exceed it, first {over[:4]} - `hi` would wrap "
           f"in int8" if over else
           "not implied by E: the accumulator alone permits 262,143 at K=64")

    # ... and `split_digits` refuses rather than wrapping, so the cap failing
    # would cost a message instead of a wrong product.
    ok_f = False
    try:
        EM.split_digits(torch.zeros(1, dtype=torch.int32), EM.DIGIT_MAX * 2 + 1)
    except AssertionError:
        ok_f = True
    try:
        EM.split_digits(torch.zeros(1, dtype=torch.int32), EM.DIGIT_MAX)
    except AssertionError:
        ok_f = False
    report("F: split_digits refuses an over-wide declaration", ok_f)

    # ... and where no width fits, construction must REFUSE.
    ok = first_refused is not None
    if ok:
        lin = torch.nn.Linear(first_refused, 8, bias=False)
        try:
            EM.ExactLinear(lin)
            ok = False
        except AssertionError:
            pass
        lin_ok = torch.nn.Linear(first_refused // 2, 8, bias=False)
        try:
            EM.ExactLinear(lin_ok)
        except AssertionError:
            ok = False                  # the ceiling is stated too low
    report(f"ExactLinear refuses at K={first_refused:,} and accepts at "
           f"K={(first_refused or 2) // 2:,}", ok,
           "the plain int8 path's int32 accumulator was unbounded until this "
           "check asked; unreachable in any real model, but unstated")


def exhaust_k_levels_for():
    print("\n--- exhaustive: does k_levels_for meet the spec Z3 assumed? ---")
    bad_spec, floor_at = [], None
    for d in range(1, MAX_D + 1):
        k = exact_kv.k_levels_for(d, D.Q_LEVELS)
        why = width_violation(d, D.Q_LEVELS, k)
        if why:
            bad_spec.append((d, k, why))
        elif d * D.Q_LEVELS * k >= TWO24 and floor_at is None:
            floor_at = d           # the 127 floor, which the caller refuses
    report("k_levels_for returns a power-of-two width within budget",
           not bad_spec,
           f"{len(bad_spec)} violations, first {bad_spec[:3]}" if bad_spec
           else f"all d in 1..{MAX_D}; the 127 floor first bites at d={floor_at}")

    widths = {}
    for d in (32, 64, 80, 96, 128, 160, 256, 512):
        widths[d] = exact_kv.k_levels_for(d, D.Q_LEVELS)
    print(f"         head_dim -> K width: {widths}")

    # The sweep above fixes `q_levels` at 127 and `K_LEVELS` at its default,
    # and BOTH are documented as sweepable -- `K_LEVELS` through
    # `Y_EXACT_K_LEVELS`, `q_levels` because it is a parameter. A check pinned
    # to the default configuration is how a dropped Montgomery carry survived
    # in this repo once already: it was correct under BN254 and wrong under
    # Pallas. `k_levels_for` reads the module-level `K_LEVELS`, so the third
    # axis is swept by re-deriving with the same arithmetic and comparing
    # against the module at its own setting.
    def klv(d, q, kmax):
        cap = min(exact_kv.SCORE_BUDGET // max(d * q, 1), kmax)
        n = max(1, cap.bit_length())
        if (1 << n) - 1 > cap:
            n -= 1
        return max(127, (1 << n) - 1)

    agree = all(klv(d, D.Q_LEVELS, exact_kv.K_LEVELS)
                == exact_kv.k_levels_for(d, D.Q_LEVELS)
                for d in range(1, MAX_D + 1))
    off_axis = []
    for kmax in (127, 255, 511, 1023, 2047, 4095, 16383):
        for q in (1, 7, 15, 63, 127, 128, 255, 1023):
            for d in range(1, MAX_D + 1):
                k = klv(d, q, kmax)
                why = width_violation(d, q, k)
                if why:
                    off_axis.append((d, q, kmax, k, why))
    report("the same holds off the default axes (K_LEVELS x q_levels x head_dim)",
           agree and not off_axis,
           f"{len(off_axis)} violations, first {off_axis[:2]}" if off_axis
           else ("7 K_LEVELS x 8 q_levels x " + str(MAX_D) + " head_dims = "
                 + f"{7 * 8 * MAX_D:,} configurations, and the re-derivation "
                   "agrees with the module at its own setting"))


def exhaust_digit_loop():
    print("\n--- exhaustive: does exact_pv's digit loop meet bound B? ---")
    # The loop is monotone in T, so every distinct outcome is reached by
    # sweeping T across the boundaries. Sweep densely near each one and sparsely
    # between, plus every exact boundary.
    v = exact_kv.V_LEVELS
    probes = set()
    for dbits in range(0, 10):
        b = TWO24 // max((1 << dbits) - 1, 1) // v
        probes.update(x for x in (b - 2, b - 1, b, b + 1, b + 2) if x >= 1)
    probes.update(range(1, 4096))
    probes.update(1 << i for i in range(1, 23))
    bad = []
    for t in sorted(probes):
        dbits = D.digit_width(t)        # the REAL selector, not a transcription
        if dbits == 0:
            continue                       # float64 fallback, bound C governs
        why = digit_violation(t, dbits, v)
        if why:
            bad.append((t, dbits, why))
    report(f"digit width keeps every fp32 matmul exact ({len(probes)} values of T)",
           not bad, f"violations: {bad[:5]}" if bad else "")


def differential_exact_pv():
    """Run the real `exact_pv` against an exact integer reference.

    Everything above checks the *selector*. This checks the thing the selector
    is for: that `p @ v` computed by the digit split equals the same product
    computed in exact integer arithmetic, bit for bit, at key lengths on both
    sides of every digit-width boundary.

    It is what catches a bug the bound arithmetic cannot see - the shift loop
    covering 28 bits instead of 29, say, which drops the top bit of `p` while
    every bound still holds. `p` is driven to its extremes (0 and exactly 2^28)
    rather than sampled uniformly, because the top bit is the one at risk.
    """
    print("\n--- differential: does the digit split equal exact integer p@v? ---")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(20260818)
    v_lv = exact_kv.V_LEVELS
    bad = []
    boundaries = sorted({D.digit_width(t) and t for t in
                         (TWO24 // (((1 << b) - 1) * v_lv)
                          for b in range(1, 9))} - {0, None})
    probes = sorted({1, 2, 7, 64, 129, 512, 1024}
                    | {x + o for x in boundaries for o in (-1, 0, 1) if x + o >= 1})
    for t in probes:
        p = torch.randint(0, (1 << D.P_BITS) + 1, (2, 3, t), generator=g,
                          dtype=torch.int64, device=dev)
        p[..., 0] = 1 << D.P_BITS            # the top bit, explicitly
        p[..., -1] = 0
        v = torch.randint(-v_lv, v_lv + 1, (2, t, 5), generator=g,
                          dtype=torch.int64, device=dev)
        got = D.exact_pv(p, v.to(torch.float32))            # [B,Q,T] @ [B,T,d]
        ref = torch.matmul(p.to(torch.float64), v.to(torch.float64))
        # The reference is itself only trustworthy while bound C holds, which is
        # what every probe here satisfies by construction (T is far below the
        # ceiling). Assert it rather than assume it.
        assert t * (1 << D.P_BITS) * v_lv < TWO53
        if not torch.equal(got, ref):
            n = int((got != ref).sum())
            bad.append((t, D.digit_width(t), n))
    report(f"exact_pv == integer p@v at {len(probes)} key lengths "
           f"(digit widths {sorted({D.digit_width(t) for t in probes})})",
           not bad, f"mismatches: {bad[:5]}" if bad else "")


def exhaust_exp_range():
    """`p <= 2^28` is the load-bearing claim under bounds C and D, and nothing
    asserts it. `exp2_neg_q16_16` is a pure integer function of one int64, and
    its caller clamps the input to [0, 2^30] and zeroes everything with
    `n = t >> 16 >= 30`. So the entire reachable domain is t in [0, 30*65536),
    ~2M values - enumerable outright rather than sampled."""
    print("\n--- exhaustive: is the softmax weight really bounded by 2^28? ---")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    table = D._table(dev)
    hi = 30 * 65536
    worst, worst_at, mono_breaks = 0, -1, 0
    prev_last = None
    seen, floor_reached = set(), False
    for lo in range(0, hi, 1 << 20):
        t = torch.arange(lo, min(lo + (1 << 20), hi), dtype=torch.int64, device=dev)
        p = D.exp2_neg_q16_16(t, table)
        mx = int(p.max())
        if mx > worst:
            worst, worst_at = mx, int(t[int(p.argmax())])
        seen.update(int(x) for x in torch.unique(p).tolist())
        lo_v = int(p.min())
        floor_reached = floor_reached or lo_v == 0
        if lo_v < 0:
            report("exp2 is non-negative", False, "negative weight")
            return
        # 2^-t is decreasing, so the integer approximation should be too. A
        # non-monotone step is not itself unsound but it means the polynomial
        # is not behaving, which is worth knowing before trusting the range.
        dif = p[1:] - p[:-1]
        mono_breaks += int((dif > 0).sum())
        if prev_last is not None and int(p[0]) > prev_last:
            mono_breaks += 1
        prev_last = int(p[-1])
    # Above the domain: the caller clamps to 2^30 and the function zeroes n>=30.
    t = torch.tensor([hi, hi + 1, (1 << 30) - 1, 1 << 30], dtype=torch.int64, device=dev)
    tail = int(D.exp2_neg_q16_16(t, table).max())
    distinct = len(seen)
    # `worst <= 2^28` is ALSO what an exp2 returning zero everywhere produces,
    # and this check reported `[ok]` under exactly that mutation, with `max p =
    # 0` printed in its own detail line and gating nothing. A bound is not a
    # test; the bound must be ATTAINED and the function must be alive:
    #
    #   * `worst == 2^P_BITS` - exp2(0) is 1.0, so the maximum is not merely
    #     below the ceiling, it IS the ceiling. Kills the all-zeros mutation.
    #   * exact dyadic anchors - exp2(-k) must be exactly 2^(P_BITS-k) for every
    #     integer k in the domain. Kills a constant-2^28 mutation, which passes
    #     the first check, and pins the scale rather than just the range.
    #   * distinct outputs - a table that has collapsed to a handful of values
    #     satisfies every anchor at the anchors and is useless between them.
    #     Same degeneracy check the arm gate uses; 846,328 is the real figure.
    #   * `p.min() == 0` - it decays to zero inside the domain, so the clamp at
    #     the top is a real edge and not the whole function.
    anchors = []
    for k in range(0, D.P_BITS + 1):
        got = int(D.exp2_neg_q16_16(
            torch.tensor([k << 16], dtype=torch.int64, device=dev), table)[0])
        if got != (1 << (D.P_BITS - k)):
            anchors.append((k, got, 1 << (D.P_BITS - k)))
    alive = (worst == (1 << D.P_BITS) and not anchors
             and distinct >= 1 << 16 and floor_reached)
    report(f"p <= 2^{D.P_BITS} over the entire reachable domain, and attains it",
           alive and tail == 0,
           f"max p = {worst:,} at t={worst_at} (2^{D.P_BITS} = {1 << D.P_BITS:,}), "
           f"above-domain max = {tail}, {mono_breaks} non-monotone steps, "
           f"{distinct:,} distinct values, {len(anchors)} bad dyadic anchors"
           + (f" {anchors[:3]}" if anchors else ""))


def check_asserts_refuse():
    """The bounds Z3 found reachable must be REFUSED, not silently exceeded.

    This is the fail-closed half. A limit that the code walks past quietly is a
    wrong answer; a limit it refuses at is a documented ceiling.
    """
    print("\n--- do the asserts actually refuse at the limits Z3 found? ---")
    # Bounded, not `while True`. The search only terminates because
    # `k_levels_for` floors at 127 and so eventually exceeds the budget; delete
    # that floor and the width shrinks with `d` fast enough that the product
    # stays under budget forever. Mutation-checked, and the first version of
    # this loop HUNG on it -- a test that runs forever is worse than one that
    # fails, which is the same harness weakness `tests/llvm_control_flow.rs`
    # had to grow a deadline for.
    dmin = None
    for d in range(1, MAX_D + 1):
        if d * D.Q_LEVELS * exact_kv.k_levels_for(d, D.Q_LEVELS) >= TWO24:
            dmin = d
            break
    if dmin is None:
        report(f"some head_dim <= {MAX_D} exceeds the score budget", False,
               "k_levels_for never exceeds the budget, so its fail-closed floor "
               "is gone and nothing below can be tested")
        return
    ok = False
    try:
        D.assert_exact_range(None, "integer scores",
                             float(dmin) * D.Q_LEVELS
                             * exact_kv.k_levels_for(dmin, D.Q_LEVELS))
        # `assert_exact_range` only checks 2^53. The 2^24 score budget used to
        # be a bare assert inside `exact_attention`, and this line MIRRORED it
        # -- so deleting the real one left this check green. It is a callable
        # now (`D.assert_score_budget`) and this exhausts the real rule, the
        # same discipline `digit_width` is split out for.
        D.assert_score_budget(dmin, D.Q_LEVELS,
                              exact_kv.k_levels_for(dmin, D.Q_LEVELS))
    except AssertionError:
        ok = True
    report(f"head_dim {dmin} (past the score budget) is refused", ok)

    tmax = TWO53 // ((1 << D.P_BITS) * exact_kv.V_LEVELS)
    ok_hi = False
    try:
        D.assert_bound(float(tmax + 1) * (1 << D.P_BITS) * exact_kv.V_LEVELS,
                       "W*V accumulator")
    except AssertionError:
        ok_hi = True
    ok_lo = True
    try:
        D.assert_bound(float(tmax - 1) * (1 << D.P_BITS) * exact_kv.V_LEVELS,
                       "W*V accumulator")
    except AssertionError:
        ok_lo = False           # would mean the ceiling is stated too high
    report(f"context T={tmax + 1:,} is refused and T={tmax - 1:,} is accepted",
           ok_hi and ok_lo,
           f"so the exact path's hard context ceiling at V={exact_kv.V_LEVELS} "
           f"is {tmax:,} tokens")


def main():
    if "--check-sweeps" in sys.argv:
        return check_sweep_predicates()
    print("Exactness-bound conjunction check")
    print(f"  constants: Q_LEVELS={D.Q_LEVELS} P_BITS={D.P_BITS} "
          f"K_LEVELS={exact_kv.K_LEVELS} V_LEVELS={exact_kv.V_LEVELS}")
    print(f"  z3 {z3.get_version_string()}")
    z3_derivations()
    z3_activation_width()
    exhaust_k_levels_for()
    exhaust_digit_loop()
    differential_exact_pv()
    exhaust_exp_range()
    check_asserts_refuse()
    print("\n" + "=" * 74)
    if fails:
        print(f"  {len(fails)} CHECK(S) FAILED: {fails}")
        return 1
    print("  every bound holds on its whole domain, and every configuration")
    print("  Z3 found outside a bound is refused rather than computed.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
