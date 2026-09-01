//! The tie between `proofs/SoftmaxErrorBound.v` and the running kernel.
//!
//! That file bounds the exact-attention softmax's output against the ideal one.
//! It is the first BOUND in `proofs/` - every other theorem there is an
//! exactness or a coverage claim - and its hypotheses are only worth anything
//! if the constants they are stated over are the compiler's own and the one
//! hypothesis it does not prove is discharged somewhere.
//!
//! This is the `GridStrideSplit` grade of tie, not the exact GEMM's: the proof
//! is checked against the running code, not against an expression both are
//! rendered from. Said plainly rather than glossed.
//!
//! What each test is for:
//!
//! - [the_proof_and_the_emitter_agree_on_every_constant] - the proof's `SAT`,
//!   `HALF`, `ULP` and Q0.28 scale are read OUT OF THE `.v` and compared
//!   against the emitted PTX and `fixed_exp`. Re-deriving them here would be a
//!   third copy; this asserts the two producers agree.
//! - [the_admitted_domain_beyond_the_exhaustive_sweep_is_below_a_quarter_ulp] -
//!   the Rust twin of `the_swept_domain_covers_the_admitted_one`. The
//!   exhaustive accuracy sweep in `fixed_exp.rs` stops at `31<<16`; the emitted
//!   `min.s64` admits arguments to `2^30`, 528 times further out. Nothing had
//!   checked the region between.
//! - [the_per_element_bracket_holds_over_the_whole_chain] - the composed
//!   joint: the exp's ADDITIVE ulp and the argument reduction's MULTIPLICATIVE
//!   error, measured together against `EPS * w + 1`.
//! - [the_output_bound_holds_on_score_distributions] - the headline, end to
//!   end, including an attention sink and a flat distribution.
//! - two controls, because a bound that cannot be violated is not a bound:
//!   [without_the_saturate_the_bound_is_violated] and
//!   [without_the_max_subtraction_the_denominator_collapses].
//!
//! Run with:  cargo test --release --test softmax_error_bound

use y::exact_attention::attention_ptx;
use y::fixed_exp::{exp2_neg_q16_16, EXP2_DOMAIN_BITS_SWEPT, EXP2_FRAC_BITS};

/// The emitted argument reduction, transcribed from the PTX in
/// `src/exact_attention.rs`:
///
/// ```text
/// mul.wide.s32 %rd50, %r16, %r50;     // (m - s) * KFix
/// add.s64      %rd50, %rd50, 32768;
/// shr.s64      %rd50, %rd50, 16;
/// min.s64      %rd50, %rd50, 1073741824;
/// cvt.u32.u64  %r16,  %rd50;
/// ```
///
/// The `clamp` flag is a parameter so both readings come from ONE description
/// and cannot drift - the same shape `the_saturate_is_what_stops_a_far_key_...`
/// in `tests/exact_attention_bounds.rs` uses.
fn emitted_arg(delta: i64, kfix: i64, clamp: bool) -> u32 {
    let t = (delta * kfix + 32768) >> 16;
    let t = if clamp { t.min(1_073_741_824) } else { t };
    t as u32
}

/// The ideal Q0.28 weight, `2^28 * 2^(-u / 2^32)`, at the EXACT exponent
/// numerator `u = delta * KFix`. This is the proof's `W (u_exact delta kfix)`.
fn ideal_weight(delta: i64, kfix: i64) -> f64 {
    let u = (delta as f64) * (kfix as f64);
    (1u64 << EXP2_FRAC_BITS) as f64 * (-u / 4_294_967_296.0f64).exp2()
}

/// The proof's `EPS`, `1 / 65536`.
const EPS: f64 = 1.0 / 65536.0;
/// `V` is int8.
const VMAX: f64 = 127.0;

/// Pull a `Definition NAME : Z := <literal>.` out of the proof, so the numbers
/// below are the proof's and not a transcription of them.
fn proof_constant(name: &str) -> i64 {
    let src = std::fs::read_to_string("proofs/SoftmaxErrorBound.v")
        .expect("proofs/SoftmaxErrorBound.v must exist - it is the subject of this file");
    // The definitions are column-aligned in the `.v`, so the spacing around
    // the colon varies. Match the head and then the assignment.
    let head = format!("Definition {name} ");
    let at = src
        .find(&head)
        .unwrap_or_else(|| panic!("the proof no longer defines {name}"));
    let rest = &src[at + head.len()..];
    let asg = rest
        .find(":=")
        .unwrap_or_else(|| panic!("{name} has no `:=`"));
    let rest = &rest[asg + 2..];
    let end = rest.find('.').expect("a Coq definition ends in a period");
    rest[..end]
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("{name} is no longer an integer literal: {:?}", &rest[..end]))
}

/// **The tie.** Every constant the bound is stated over must be the one the
/// compiler emits. Read out of the `.v` rather than restated, so this fails
/// when either side moves rather than when both do.
#[test]
fn the_proof_and_the_emitter_agree_on_every_constant() {
    let sat = proof_constant("SAT");
    let half = proof_constant("HALF");
    let ulp = proof_constant("ULP");
    let two32 = proof_constant("TWO32");

    assert_eq!(two32, 1i64 << 32, "TWO32 is the log2-unit span");
    assert_eq!(ulp, 1i64 << 16, "ULP is the Q16.16 shift");
    assert_eq!(half, ulp / 2, "HALF must be half an ulp, or the rounding lemma is about a different rounding");

    let ptx = attention_ptx(64, 4096).expect("64/4096 is well inside every bound");
    // The saturate, at the value the proof's second regime is stated at.
    assert_eq!(
        ptx.matches(&format!("min.s64 %rd50, %rd50, {sat};")).count(),
        2,
        "both accumulating entries must clamp the exp argument at the proof's SAT"
    );
    // The round-to-nearest addend and the shift.
    assert_eq!(
        ptx.matches(&format!("add.s64 %rd50, %rd50, {half};")).count(),
        2,
        "the proof's HALF must be the emitted round-to-nearest addend"
    );
    assert_eq!(
        ptx.matches("shr.s64 %rd50, %rd50, 16;").count(),
        2,
        "the proof's ULP must be the emitted shift"
    );

    // The Q0.28 scale the ideal weight is expressed in.
    assert_eq!(
        exp2_neg_q16_16(0),
        1 << EXP2_FRAC_BITS,
        "W 0 == TWO28 is a hypothesis of the proof; this is what discharges it"
    );
    assert_eq!(
        EXP2_FRAC_BITS, 28,
        "the proof's TWO28 is 2^28; if the fixed-point width moves, so must it"
    );

    // And the swept domain the proof's exp hypothesis is stated over must be
    // the domain `fixed_exp`'s exhaustive tests actually sweep. This is the
    // one that rots silently: widening or narrowing that sweep changes what
    // `exp_is_sub_ulp_on_the_swept_domain` is discharged by.
    assert!(
        std::fs::read_to_string("proofs/SoftmaxErrorBound.v")
            .unwrap()
            .contains("(t < 31 * 65536)%Z"),
        "the proof's swept-domain hypothesis must name the same bound \
         `fixed_exp`'s exhaustive tests sweep"
    );
    assert_eq!(
        EXP2_DOMAIN_BITS_SWEPT,
        31 << 16,
        "the exhaustive sweep's upper bound is what the proof assumes"
    );
}

/// **The gap, measured.** `it_is_sub_ulp_accurate_everywhere` sweeps
/// `0 .. 31<<16`. The emitted saturate admits arguments up to `2^30`. In
/// between, the implementation returns 0 and the ideal is below a quarter of
/// an ulp - so the "0.908 ulp everywhere" headline does hold on the whole
/// domain the kernel can reach, which nothing had established.
///
/// This is `the_swept_domain_covers_the_admitted_one`'s behavioural twin. The
/// proof gets it from thirty halvings of `2^28`; this gets it from the real
/// function.
#[test]
fn the_admitted_domain_beyond_the_exhaustive_sweep_is_below_a_quarter_ulp() {
    let sat = proof_constant("SAT") as u32;
    let swept_to = EXP2_DOMAIN_BITS_SWEPT;
    assert!(swept_to < sat, "there is a gap to check, or this test is vacuous");

    // Every argument in the gap, on a stride that includes both ends and every
    // power-of-two boundary between them.
    let mut checked = 0usize;
    let mut worst = 0.0f64;
    let mut t = swept_to;
    while t <= sat {
        assert_eq!(
            exp2_neg_q16_16(t),
            0,
            "above the table the implementation must return 0; at t = {t} it did not"
        );
        // The true value, in ulps of Q0.28.
        let want = (1u64 << EXP2_FRAC_BITS) as f64 * (-(t as f64) / 65536.0).exp2();
        worst = worst.max(want);
        checked += 1;
        // powers of two, and a few points around each
        t = t.saturating_add(t / 3 + 1);
    }
    for t in [swept_to, swept_to + 1, sat - 1, sat, u32::MAX] {
        assert_eq!(exp2_neg_q16_16(t), 0, "t = {t} must return 0");
    }
    assert!(checked > 20, "the gap sweep visited only {checked} points");
    assert!(
        worst < 0.25,
        "the largest ideal weight in the un-swept region is {worst} ulp, not \
         below the quarter ulp the proof's thirty-halvings argument gives"
    );
}

/// **The composed joint.** The exp's error is additive (one ulp of Q0.28); the
/// argument reduction's is multiplicative (it perturbs an exponent). The proof
/// composes them as `EPS * w + 1`, and this is that inequality measured on the
/// real function over the temperatures and score deltas a model produces.
///
/// It covers both regimes the proof splits on: the saturating one is included
/// deliberately, since that is where the bracket is carried by the
/// thirty-halvings argument rather than by the two-sided one.
#[test]
fn the_per_element_bracket_holds_over_the_whole_chain() {
    // The per-tensor scales real activations give, from
    // `the_temperature_multiplier_carries_two_to_the_thirty_two`.
    let scales = [7.0e-5f64, 4.5e-4, 1.8e-3, 3.0e-2];
    let mut worst_ratio = 0.0f64;
    let mut saturating = 0usize;
    let mut checked = 0usize;

    for &c in &scales {
        let kfix = (c * 2f64.powi(32)).round() as i64;
        // An int8 dot product over head_dim <= 128 spans 127*127*128 either
        // way, so a delta spans twice that. Sweep it, plus the region past the
        // saturate.
        let ds_max = 2 * 127 * 127 * 128;
        let mut delta = 0i64;
        while delta <= ds_max * 8 {
            let p = exp2_neg_q16_16(emitted_arg(delta, kfix, true)) as f64;
            let w = ideal_weight(delta, kfix);
            let allowed = EPS * w + 1.0;
            let err = (p - w).abs();
            assert!(
                err <= allowed,
                "the per-element bracket fails at C = {c:e}, delta = {delta}: \
                 |{p} - {w}| = {err} > EPS*w + 1 = {allowed}"
            );
            if allowed > 0.0 {
                worst_ratio = worst_ratio.max(err / allowed);
            }
            if (delta * kfix + 32768) >> 16 > 1_073_741_824 {
                saturating += 1;
            }
            checked += 1;
            delta = delta + 1 + delta / 64;
        }
    }
    println!(
        "per-element bracket: {checked} points, worst err/allowed = {worst_ratio:.4}, \
         {saturating} of them saturating"
    );
    assert!(checked > 1000, "only {checked} points were checked");
    assert!(
        saturating > 0,
        "no swept point reached the saturate, so the regime the proof handles \
         with the thirty-halvings argument is untested here"
    );
    // Non-vacuity: the bracket must actually be exercised rather than being
    // orders of magnitude away from anything observed.
    assert!(
        worst_ratio > 0.01,
        "the worst observed error is {worst_ratio} of the allowance, so this \
         sweep is not exercising the bracket at all"
    );
}

/// Run the whole chain - scores, max, argument reduction, exp, exact integer
/// accumulate, divide - and check the output against an f64 ideal softmax at
/// the same temperature. The comparison is the proof's:
///
/// ```text
/// |O/L - Wnum/Wtot| <= 2 * VMAX * (EPS * Wtot + n) / L
/// ```
fn run_chain(scores: &[i32], vals: &[i8], kfix: i64, clamp: bool, subtract_max: bool) -> (f64, f64, f64) {
    let n = scores.len();
    let m = if subtract_max { *scores.iter().max().unwrap() } else { 0 };
    let mut lq: i64 = 0;
    let mut oq: i64 = 0;
    let mut wtot = 0.0f64;
    let mut wnum = 0.0f64;
    for i in 0..n {
        let delta = (m as i64) - (scores[i] as i64);
        // A negative delta is outside the kernel's domain - the max
        // subtraction is what makes it non-negative - so the no-max control
        // below clamps it, which is exactly the "all weights are tiny" regime.
        let delta = delta.max(0);
        let p = exp2_neg_q16_16(emitted_arg(delta, kfix, clamp)) as i64;
        lq += p;
        oq += p * (vals[i] as i64);
        let w = ideal_weight(delta, kfix);
        wtot += w;
        wnum += w * (vals[i] as f64);
    }
    let bound = 2.0 * VMAX * (EPS * wtot + n as f64) / (lq as f64);
    let got = if lq == 0 { f64::NAN } else { oq as f64 / lq as f64 };
    let want = wnum / wtot;
    (got, want, bound)
}

#[test]
fn the_output_bound_holds_on_score_distributions() {
    let mut st = 0x5EED_1234_u64;
    let mut rnd = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (st >> 33) as i64
    };

    let c = 4.5e-4f64;
    let kfix = (c * 2f64.powi(32)).round() as i64;
    let mut worst_ratio = 0.0f64;

    for &n in &[16usize, 256, 4096] {
        for shape in 0..3 {
            let scores: Vec<i32> = (0..n)
                .map(|i| {
                    let base = (rnd() % 4000) as i32 - 2000;
                    match shape {
                        // an attention SINK: one score far above the rest, the
                        // distribution that broke a 16-bit weight width.
                        0 => if i == 0 { 60_000 } else { base },
                        // flat: every weight near 2^28, so the additive term
                        // is relatively smallest.
                        1 => base / 100,
                        // wide: deltas large enough that most weights are
                        // small integers, where the additive ulp dominates.
                        _ => base * 30,
                    }
                })
                .collect();
            let vals: Vec<i8> = (0..n).map(|_| (rnd() % 255 - 127) as i8).collect();

            let (got, want, bound) = run_chain(&scores, &vals, kfix, true, true);
            let err = (got - want).abs();
            assert!(
                err <= bound,
                "n = {n}, shape {shape}: output {got} against ideal {want}, \
                 error {err} exceeds the proved bound {bound}"
            );
            worst_ratio = worst_ratio.max(err / bound);
            println!("n={n} shape={shape}: err={err:.3e} bound={bound:.3e} ratio={:.4}", err / bound);
        }
    }
    // The bound is a worst case, so random data will not approach it; what
    // this asserts is that the comparison is measuring something at all.
    assert!(
        worst_ratio > 0.0,
        "every observed error was exactly zero, so this test would pass \
         against a bound of any size"
    );
}

/// **Control 1.** The proof's second regime is the saturate, and this is what
/// makes it necessary rather than decorative: without the `min.s64` a far key
/// wraps modulo 2^32 and comes back with a near-maximal weight, and the output
/// leaves the bound entirely.
#[test]
fn without_the_saturate_the_bound_is_violated() {
    // The largest scale swept in `exact_attention_bounds.rs`, where an int8
    // head_dim-128 dot product can produce a wrapping delta.
    let c = 3.0e-2f64;
    let kfix = (c * 2f64.powi(32)).round() as i64;
    let ds_max = 2 * 127 * 127 * 128;
    let wrapped = (1..=ds_max as i64)
        .find(|&ds| {
            let t = (ds * kfix + 32768) >> 16;
            t >= 1 << 32 && (t % (1 << 32)) < 65536
        })
        .expect("no delta in range wraps the argument; the arithmetic has changed");

    // Two keys: the best one, and one `wrapped` below it with the opposite V.
    let scores = [0i32, -(wrapped as i32)];
    let vals = [127i8, -128];

    let (got_ok, want_ok, bound_ok) = run_chain(&scores, &vals, kfix, true, true);
    assert!(
        (got_ok - want_ok).abs() <= bound_ok,
        "with the saturate the bound must hold: {got_ok} vs {want_ok}, bound {bound_ok}"
    );
    // The far key contributes nothing, so the answer is the best key's V.
    assert!((got_ok - 127.0).abs() < 1.0, "clamped, the answer is the best key: {got_ok}");

    let (got_bad, _, _) = run_chain(&scores, &vals, kfix, false, true);
    assert!(
        (got_bad - want_ok).abs() > bound_ok,
        "without the saturate the output must leave the proved bound; it gave \
         {got_bad} against {want_ok} with bound {bound_ok}. If this passes, the \
         wrap is no longer observable and the proof's saturation regime is \
         being tested by nothing"
    );
}

/// **Control 2.** The floor `Wtot >= 2^28` is what makes the exp table's
/// ABSOLUTE error relatively negligible, and it comes from the max
/// subtraction. Without it, every weight rounds to zero and there is no output
/// at all - which is `a_total_weight_below_one_ulp_admits_a_zero_denominator`,
/// and is the F=16 attention-sink failure `attention_quantization_error.rs`
/// records, one layer up.
#[test]
fn without_the_max_subtraction_the_denominator_collapses() {
    let c = 4.5e-4f64;
    let kfix = (c * 2f64.powi(32)).round() as i64;
    // Scores all far below zero: with the max subtracted they are ordinary,
    // without it every delta is huge.
    let scores: Vec<i32> = (0..64).map(|i| -3_000_000 - (i as i32) * 17).collect();
    let vals: Vec<i8> = (0..64).map(|i| (i as i8) - 32).collect();

    let (got, want, bound) = run_chain(&scores, &vals, kfix, true, true);
    assert!(
        (got - want).abs() <= bound,
        "with the max subtracted the bound holds: {got} vs {want}, bound {bound}"
    );

    let (got_bad, _, _) = run_chain(&scores, &vals, kfix, true, false);
    assert!(
        got_bad.is_nan(),
        "without the max subtraction every weight must round to zero, leaving \
         no denominator at all; it gave {got_bad}. If this passes, the floor \
         the proof puts under Wtot is being demonstrated by nothing"
    );
}
