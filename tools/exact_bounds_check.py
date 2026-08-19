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
        # It must be a 2^n - 1 ...
        if (k + 1) & k:
            bad_spec.append((d, k, "not 2^n-1"))
            continue
        # ... and either within budget, or the 127 floor (which the caller's
        # assert then refuses).
        if d * D.Q_LEVELS * k >= TWO24:
            if k != 127:
                bad_spec.append((d, k, "over budget and not the floor"))
            elif floor_at is None:
                floor_at = d
    report("k_levels_for returns a power-of-two width within budget",
           not bad_spec,
           f"{len(bad_spec)} violations, first {bad_spec[:3]}" if bad_spec
           else f"all d in 1..{MAX_D}; the 127 floor first bites at d={floor_at}")

    widths = {}
    for d in (32, 64, 80, 96, 128, 160, 256, 512):
        widths[d] = exact_kv.k_levels_for(d, D.Q_LEVELS)
    print(f"         head_dim -> K width: {widths}")


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
        if t * ((1 << dbits) - 1) * v >= TWO24:
            bad.append((t, dbits))
        # the shift loop must cover all 29 bits of p
        if ((29 + dbits - 1) // dbits) * dbits < 29:
            bad.append((t, dbits, "digit loop does not cover 29 bits"))
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
    for lo in range(0, hi, 1 << 20):
        t = torch.arange(lo, min(lo + (1 << 20), hi), dtype=torch.int64, device=dev)
        p = D.exp2_neg_q16_16(t, table)
        mx = int(p.max())
        if mx > worst:
            worst, worst_at = mx, int(t[int(p.argmax())])
        if int(p.min()) < 0:
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
    report(f"p <= 2^{D.P_BITS} over the entire reachable domain",
           worst <= (1 << D.P_BITS) and tail == 0,
           f"max p = {worst:,} at t={worst_at} (2^{D.P_BITS} = {1 << D.P_BITS:,}), "
           f"above-domain max = {tail}, {mono_breaks} non-monotone steps")


def check_asserts_refuse():
    """The bounds Z3 found reachable must be REFUSED, not silently exceeded.

    This is the fail-closed half. A limit that the code walks past quietly is a
    wrong answer; a limit it refuses at is a documented ceiling.
    """
    print("\n--- do the asserts actually refuse at the limits Z3 found? ---")
    dmin = 1
    while dmin * D.Q_LEVELS * exact_kv.k_levels_for(dmin, D.Q_LEVELS) < TWO24:
        dmin += 1
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
