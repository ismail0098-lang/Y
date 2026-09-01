//! Architecture-independent fixed-point `exp2`, for deterministic softmax.
//!
//! # Why this exists
//!
//! Every reduction in the exact attention path is order-independent by
//! construction, so the answer does not depend on batch size, split count or
//! thread count. But the softmax weight itself is computed per element with
//! `ex2.approx.f32`, and that is a *fixed-function unit whose rounding is a
//! property of the SM generation*. Same input, same output, every launch — on
//! one architecture. Across two different GPUs, no guarantee.
//!
//! That is fine for batch invariance (the step is per-element, so it cannot
//! see the partition) and fatal for the "same answer on two different GPUs"
//! claim in `docs/deterministic_inference.md` M5. This module closes it: the
//! whole computation is integer, so it is bit-identical on any machine that
//! implements 32- and 64-bit integer arithmetic — which is all of them.
//!
//! # The algorithm
//!
//! `2^-t` for `t >= 0` in Q16.16. Split `t = n + f` with `n` the integer part
//! and `f` in `[0, 1)`, so `2^-t = 2^-f >> n`. Then split `f` again: the top
//! six bits index a 64-entry table of `round(2^30 * 2^-(i/64))`, and the low
//! ten bits are handled by a truncated series in `y = delta * ln2`:
//!
//! ```text
//! 2^-delta = 1 - y + y^2/2 - y^3/6 + ...
//! ```
//!
//! `delta < 2^-6` gives `y < 0.0108`, so the first omitted term `y^4/24` is
//! below `5.7e-10` — about **0.15 ulp** of the Q0.28 result, i.e. invisible.
//! Three terms are needed and four would be waste: `y^3/6` alone is ~56 ulp,
//! so dropping it would be a real error.
//!
//! Measured worst case over the whole domain: **0.908 ulp**, i.e. the result
//! is correctly rounded to within one unit everywhere.
//!
//! # Bit-exactness is the point, so the operations are pinned
//!
//! Every step below is chosen so a PTX transcription can be *identical*, not
//! merely equivalent: the division by six is a multiply-high by
//! `ceil(2^32 / 6)` rather than a `div`, which is exact for the range `y^3`
//! can occupy (`< 2^12`) and needs no divider. `tests/ptx_fixed_exp.rs`
//! asserts the device agrees with this function bit for bit; if you change one
//! side, that test is what tells you.

/// Fractional bits in the result. `exp2_neg_q16_16(0) == 1 << EXP2_FRAC_BITS`.
pub const EXP2_FRAC_BITS: u32 = 28;

/// The upper end of the domain the exhaustive accuracy sweep covers.
///
/// It was a bare `31 << 16` repeated in four places, and it is not an
/// arbitrary sweep width: it is what `it_is_sub_ulp_accurate_everywhere`,
/// `it_is_monotonically_non_increasing` and the whole-domain digest actually
/// walk, and therefore exactly the domain over which the sub-ulp headline is
/// established.
///
/// **The kernel admits arguments far past it.** `attn_accum` saturates to
/// `2^30` before the call - 528 times further out - so a claim stated over
/// this bound says nothing about the region between unless something closes
/// the gap. `proofs/SoftmaxErrorBound.v`'s
/// `the_swept_domain_covers_the_admitted_one` closes it (above the table the
/// result is 0 and the true weight is below a quarter of an ulp), and
/// `tests/softmax_error_bound.rs` asserts the proof's hypothesis is stated
/// over THIS number rather than over a number that used to be this one.
pub const EXP2_DOMAIN_BITS_SWEPT: u32 = 31 << 16;

/// `round(ln 2 * 2^30)`.
pub const LN2_Q30: u64 = 744_261_118;

/// `ceil(2^32 / 6)`. Exact for dividends below `2^12`, which bounds `y^3`.
pub const RECIP6_Q32: u64 = 715_827_883;

/// FNV-1a over `exp2_neg_q16_16(t)` for every `t` in the whole domain
/// `0 .. 31 << 16`, as the cross-language bit-identity anchor.
///
/// There are FOUR transcriptions of this recipe in the repo: this one, the PTX
/// in `ptx_device_function`, a pure-Python one in
/// `tools/attention_real_activations.py`, and a vectorised torch one in
/// `tools/batch_invariance_demo.py` (which is what the demo, `exact_accuracy`
/// and `tools/ptx_bridge.py`'s reference arm all call). The PTX one is checked
/// against this one by `tests/ptx_fixed_exp.rs`; the two Python ones were
/// checked against nothing.
///
/// What they DID have was a self-check asserting the result is within 1 ulp of
/// `2^28 * 2^-t` and monotonic — properties, not identity. Two implementations
/// can both be within 1 ulp of the truth and still differ from each other by 1,
/// and 1 ulp IS a bit. So the check could not see the only failure that matters
/// to a project whose claim is bit-identity, while its comment said it could.
///
/// Both replicas assert this constant now. It is a whole-domain digest, so it
/// cannot be satisfied by a spot check, and it costs ~4 ms here and ~2 s in
/// pure Python.
///
/// The three implementations DO agree today — verified before pinning, over all
/// 2,031,616 inputs. The gate is what makes that a fact rather than a hope.
pub const EXP2_DOMAIN_DIGEST: u64 = 0x7170_cd39_b442_d506;

/// `round(2^30 * 2^-(i/64))`.
pub const EXP2_TABLE: [u32; 64] = [
    1073741824, 1062175491, 1050733751, 1039415261,
    1028218693, 1017142735, 1006186087, 995347464,
    984625594, 974019220, 963527098, 953147997,
    942880699, 932724001, 922676710, 912737649,
    902905651, 893179563, 883558244, 874040567,
    864625413, 855311680, 846098274, 836984114,
    827968132, 819049271, 810226483, 801498734,
    792865000, 784324269, 775875538, 767517817,
    759250125, 751071493, 742980960, 734977579,
    727060411, 719228525, 711481005, 703816941,
    696235434, 688735596, 681316545, 673977412,
    666717336, 659535466, 652430958, 645402981,
    638450708, 631573326, 624770026, 618040012,
    611382493, 604796689, 598281827, 591837143,
    585461881, 579155293, 572916640, 566745190,
    560640218, 554601009, 548626854, 542717053,
];

/// `round(2^28 * 2^(-t / 2^16))` for `t` a non-negative Q16.16 fixed-point
/// value. Saturates to 0 once the result would round to nothing.
///
/// Pure integer: bit-identical on every architecture, by construction.
pub fn exp2_neg_q16_16(t: u32) -> u32 {
    let n = t >> 16;
    // Q0.28 cannot represent anything below 2^-29 after rounding.
    if n >= 30 {
        return 0;
    }
    let f = t & 0xFFFF;
    let base = EXP2_TABLE[(f >> 10) as usize] as u64; // Q0.30
    let d = (f & 0x3FF) as u64; // delta * 2^16

    let y = (d * LN2_Q30) >> 16; // Q0.30
    let y2 = (y * y) >> 30;
    let y3 = (y2 * y) >> 30;
    // y3 < 2^12, so the reciprocal multiply is exact division by 6.
    let y3_over_6 = (y3 * RECIP6_Q32) >> 32;
    let corr = (1u64 << 30) - y + (y2 >> 1) - y3_over_6; // Q0.30

    let g = (base * corr) >> 30; // Q0.30 = 2^30 * 2^-f
    let shift = 2 + n; // Q0.30 -> Q0.28, then >> n
    let half = 1u64 << (shift - 1);
    ((g + half) >> shift) as u32
}

/// The same function as a PTX `.const` table plus a `.func`, so a kernel can
/// call it and the two implementations cannot drift apart.
///
/// Emitted here rather than written out in a test so that
/// `exp2_neg_q16_16` above and the device code are edited together;
/// `tests/ptx_fixed_exp.rs` checks they agree bit for bit.
pub fn ptx_device_function() -> String {
    let table = EXP2_TABLE
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"
.const .align 4 .u32 y_exp2_tbl[64] = {{ {table} }};

// 2^-t for t in Q16.16, returning Q0.28. Pure integer: identical on every
// architecture, unlike `ex2.approx.f32` whose rounding belongs to the SM
// generation.
.func (.reg .u32 %ret) y_exp2_neg_q16_16 (.reg .u32 %targ)
{{
    .reg .pred %ep<4>;
    .reg .u32  %e<16>;
    .reg .u64  %ed<24>;

    mov.u32 %e1, %targ;
    shr.u32 %e2, %e1, 16;
    mov.u32 %e15, 0;
    setp.ge.u32 %ep1, %e2, 30;
    @%ep1 bra Y_EXP_DONE;

    and.b32 %e3, %e1, 65535;
    shr.u32 %e4, %e3, 10;
    mul.wide.u32 %ed1, %e4, 4;
    mov.u64 %ed2, y_exp2_tbl;
    add.s64 %ed3, %ed2, %ed1;
    ld.const.u32 %e5, [%ed3];
    cvt.u64.u32 %ed4, %e5;

    and.b32 %e6, %e3, 1023;
    cvt.u64.u32 %ed5, %e6;
    mov.u64 %ed6, {ln2};
    mul.lo.u64 %ed7, %ed5, %ed6;
    shr.u64 %ed7, %ed7, 16;

    mul.lo.u64 %ed8, %ed7, %ed7;
    shr.u64 %ed8, %ed8, 30;
    mul.lo.u64 %ed9, %ed8, %ed7;
    shr.u64 %ed9, %ed9, 30;
    mov.u64 %ed10, {r6};
    mul.lo.u64 %ed11, %ed9, %ed10;
    shr.u64 %ed11, %ed11, 32;

    mov.u64 %ed12, 1073741824;
    sub.s64 %ed13, %ed12, %ed7;
    shr.u64 %ed14, %ed8, 1;
    add.s64 %ed13, %ed13, %ed14;
    sub.s64 %ed13, %ed13, %ed11;

    mul.lo.u64 %ed15, %ed4, %ed13;
    shr.u64 %ed15, %ed15, 30;

    add.u32 %e7, %e2, 2;
    sub.u32 %e8, %e7, 1;
    mov.u64 %ed16, 1;
    shl.b64 %ed17, %ed16, %e8;
    add.s64 %ed18, %ed15, %ed17;
    shr.u64 %ed19, %ed18, %e7;
    cvt.u32.u64 %e15, %ed19;
Y_EXP_DONE:
    mov.u32 %ret, %e15;
    ret;
}}
"#,
        table = table,
        ln2 = LN2_Q30,
        r6 = RECIP6_Q32
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Correctly rounded to within one unit, over the entire domain.
    /// The anchor `tools/attention_real_activations.py` and
    /// `tools/batch_invariance_demo.py` assert against. If this moves, either
    /// the recipe changed (and both Python replicas must move with it) or
    /// something drifted.
    #[test]
    fn the_whole_domain_digest_is_the_cross_language_anchor() {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for t in 0..EXP2_DOMAIN_BITS_SWEPT {
            h ^= exp2_neg_q16_16(t) as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        assert_eq!(
            h, EXP2_DOMAIN_DIGEST,
            "the fixed-point exp changed. Two Python transcriptions and one PTX \
             one pin this same value; move all four together or the demo stops \
             computing what the kernel computes"
        );
    }

    #[test]
    fn it_is_sub_ulp_accurate_everywhere() {
        let mut worst = 0.0f64;
        let mut worst_t = 0u32;
        // EXHAUSTIVE over the whole domain, not a stride over it. This used to
        // sample one fractional value in sixteen above n = 0, which left the
        // headline "0.908 ulp" a measurement rather than a proof. The domain is
        // 31 * 2^16 values and the sweep costs about 4 ms, so the sampling
        // bought nothing.
        for t in 0..EXP2_DOMAIN_BITS_SWEPT {
            let want = (1u64 << EXP2_FRAC_BITS) as f64 * (-(t as f64) / 65536.0).exp2();
            let err = (exp2_neg_q16_16(t) as f64 - want).abs();
            if err > worst {
                worst = err;
                worst_t = t;
            }
        }
        println!("worst error {worst:.4} ulp of Q0.{EXP2_FRAC_BITS} at t = {worst_t} (exhaustive)");
        assert!(
            worst < 1.0,
            "fixed-point exp2 is {worst:.3} ulp off at t = {worst_t}; the series or \
             the table has lost precision"
        );
    }

    /// The identity that anchors the whole fixed-point scale.
    #[test]
    fn exp2_of_zero_is_exactly_one() {
        assert_eq!(exp2_neg_q16_16(0), 1 << EXP2_FRAC_BITS);
    }

    /// A softmax weight must never increase as its score falls. A table or
    /// series bug shows up here as a local inversion long before it shows up
    /// as a wrong model output.
    #[test]
    fn it_is_monotonically_non_increasing() {
        let mut prev = u32::MAX;
        // The whole domain. It used to stop at 2^20, i.e. n < 16, so the half
        // of the range where the shift dominates was unchecked - including the
        // step down to zero, which is where a cliff would be.
        for t in 0..EXP2_DOMAIN_BITS_SWEPT {
            let v = exp2_neg_q16_16(t);
            assert!(
                v <= prev,
                "exp2 increased between t = {} and t = {t} ({prev} -> {v})",
                t - 1
            );
            prev = v;
        }
    }

    /// Exact powers of two must come out exact, which pins the shift path
    /// independently of the table and the series.
    #[test]
    fn integral_arguments_are_exact_powers_of_two() {
        for n in 0..=28u32 {
            assert_eq!(
                exp2_neg_q16_16(n << 16),
                1 << (EXP2_FRAC_BITS - n),
                "2^-{n} is not exact"
            );
        }
        // And it saturates rather than wrapping.
        assert_eq!(exp2_neg_q16_16(40 << 16), 0);
    }

    /// The `n >= 30` early return is a GUARD, not an optimisation, and nothing
    /// pinned it.
    ///
    /// `shift = 2 + n` and the code computes `1u64 << (shift - 1)` and
    /// `>> shift`. Past `n = 62` those exceed a 64-bit register: a panic in a
    /// debug build, a masked shift and a garbage weight in a release one. The
    /// existing saturation case (`40 << 16`) passes with the guard deleted,
    /// because at `n = 40` the arithmetic still reaches zero on its own — so
    /// the case that pins it has to be far enough out to break the shift.
    ///
    /// This is not hypothetical for the kernel either: `attn_accum` saturates
    /// its argument to `2^30` before the call, which is `n = 16384`.
    #[test]
    fn the_saturation_guard_covers_the_whole_input_range() {
        for t in [30u32 << 16, 62 << 16, 63 << 16, 1 << 30, u32::MAX] {
            assert_eq!(
                exp2_neg_q16_16(t),
                0,
                "exp2_neg_q16_16({t}) must saturate to zero. Without the `n >= 30` \
                 early return the shift width runs past 64 bits here."
            );
        }
    }

    /// The `y^3` term is load-bearing, and the `y^4` term is not. Dropping the
    /// wrong one is a plausible "simplification", so both directions are
    /// pinned here rather than left to a comment.
    #[test]
    fn the_series_is_truncated_at_the_right_place() {
        let two_term = |t: u32| -> u32 {
            let n = t >> 16;
            let f = t & 0xFFFF;
            let base = EXP2_TABLE[(f >> 10) as usize] as u64;
            let d = (f & 0x3FF) as u64;
            let y = (d * LN2_Q30) >> 16;
            let y2 = (y * y) >> 30;
            let corr = (1u64 << 30) - y + (y2 >> 1); // y^3/6 dropped
            let g = (base * corr) >> 30;
            let shift = 2 + n;
            ((g + (1u64 << (shift - 1))) >> shift) as u32
        };
        let worst = (0..(1u32 << 16))
            .map(|t| {
                let want = (1u64 << EXP2_FRAC_BITS) as f64 * (-(t as f64) / 65536.0).exp2();
                (two_term(t) as f64 - want).abs()
            })
            .fold(0.0f64, f64::max);
        println!("dropping y^3/6 costs {worst:.1} ulp");
        assert!(
            worst > 4.0,
            "dropping the y^3 term only cost {worst:.2} ulp, so it is not carrying \
             its weight and this module is more complicated than it needs to be"
        );
    }
}
