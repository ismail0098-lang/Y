//! The exact-VNNI licence, checked as a CONJUNCTION rather than one bound at a
//! time.
//!
//! `VnniExact::license` is the predicate that says a tiled, threaded, K-split
//! GEMM may use `vpdpwssd` and still be bit-identical to the naive nest. It is
//! the soundness core of Phase 0 in `docs/proof_carrying_kernels.md`: everything
//! the proof-carrying-kernel programme wants to prove is stated on top of it.
//!
//! Each of its bounds is easy to check by hand IN ISOLATION, which is exactly
//! why they had not been checked together. The failure mode is never a wrong
//! bound, it is an interaction - the same finding `tools/exact_bounds_check.py`
//! produced for the attention path, where the interesting result was that one
//! bound SUBSUMED another and that a hard ceiling existed which was written down
//! nowhere.
//!
//! ## Why this exhausts instead of using a solver
//!
//! The operand magnitude lives in `vpdpwssd`'s int16 domain, so its whole
//! domain is 32,768 values. Exhausting it calls the REAL function over every
//! input it can ever receive - complete, not sampled - and closes the
//! model-versus-code gap that `proofs/ZkControlFlow.v` states about itself. A
//! solver would prove something about a transcription of the rule; this proves
//! it about the rule. Z3 earns its place where an axis is unbounded, which here
//! is K, handled separately below in exact integer arithmetic.

use y::zero_drift::VnniExact;

/// The interval the compiler actually ships. `flush_interval_for` exists but is
/// only ever consulted to phrase an error message - `plan_exact_gemm` passes
/// `DEFAULT_FLUSH_K_PAIRS` unconditionally - so this is the configuration whose
/// soundness matters.
const SHIPPING_T: u32 = VnniExact::DEFAULT_FLUSH_K_PAIRS;

/// The obligation, in exact integer arithmetic and stated independently of the
/// code under test.
///
/// `vpdpwssd` does TWO MACs per int32 lane per instruction, the accumulator is
/// zeroed at every flush, and each product is at most `m^2`. So one full flush
/// interval starting from zero contributes at most `2 * T * m^2`, and int32
/// holds `2^31 - 1`.
///
/// Written with `u128` on purpose: computing the obligation in the width whose
/// overflow it is about is how the check silently becomes vacuous.
fn accumulator_cannot_overflow(m: u64, t: u32) -> bool {
    let products = 2u128 * u128::from(t);
    let worst = products * u128::from(m) * u128::from(m);
    worst <= i32::MAX as u128
}

#[test]
fn the_licence_is_exactly_the_overflow_obligation_over_the_whole_int16_domain() {
    let scheme = VnniExact::new(SHIPPING_T).expect("the shipping interval must be valid");
    let mut too_loose: Vec<u64> = Vec::new();
    let mut too_tight: Vec<u64> = Vec::new();

    for m in 0..=(i16::MAX as u64) {
        let licensed = scheme.license(m as f64).is_ok();
        // The magnitude must not overflow int32 between flushes, AND must be a
        // magnitude the int16 operand domain can actually carry. Zero is
        // representable; anything in (0, 1) is not, and stages to zero.
        let should = accumulator_cannot_overflow(m, SHIPPING_T);

        if licensed && !should {
            too_loose.push(m);
        }
        if !licensed && should {
            too_tight.push(m);
        }
    }

    assert!(
        too_loose.is_empty(),
        "the licence is UNSOUND at these magnitudes - it permits `vpdpwssd` where the int32 \
         accumulator can overflow between flushes, which is a wrong answer under a certificate \
         claiming exactness: {:?}",
        &too_loose[..too_loose.len().min(8)]
    );
    assert!(
        too_tight.is_empty(),
        "the licence refuses these magnitudes although the accumulator provably cannot overflow. \
         Refusing working programs is not safe-by-default here - it silently drops them to the \
         scalar path and the fast kernel never runs: {:?}",
        &too_tight[..too_tight.len().min(8)]
    );
}

/// The boundary is one unit wide, so an off-by-one in the `floor(sqrt(..))`
/// derivation is invisible to any sampled test.
///
/// At T=64 the largest safe magnitude is 4095: `128 * 4095^2 = 2,146,435,200`,
/// which fits int32 with 1,048,447 to spare. 4096 gives `2,147,483,648` -
/// over `i32::MAX` by EXACTLY ONE.
#[test]
fn the_boundary_is_where_the_arithmetic_says_it_is() {
    let scheme = VnniExact::new(SHIPPING_T).unwrap();
    assert!(
        accumulator_cannot_overflow(4095, SHIPPING_T),
        "4095 must be safe at T=64"
    );
    assert!(
        !accumulator_cannot_overflow(4096, SHIPPING_T),
        "4096 must overflow at T=64 - by exactly 1"
    );
    assert!(scheme.license(4095.0).is_ok(), "4095 must be licensed");
    assert!(
        scheme.license(4096.0).is_err(),
        "4096 overflows int32 by one product-unit and must be refused"
    );
    assert_eq!(scheme.max_operand_magnitude(), 4095.0);
}

/// K is the one axis with no small finite domain, so it gets stated arithmetic
/// rather than exhaustion.
///
/// The int32 accumulator is flushed into an int64 one, and the K-split sums
/// several int64 partials. Every one of the K products is at most `m^2`, so the
/// whole reduction is bounded by `K * m^2` REGARDLESS of how it is tiled,
/// threaded or split - which is the same associativity that makes the kernel
/// worth proving in the first place.
///
/// This ceiling is written down nowhere else. It is astronomically safe today,
/// and it MOVES if the flush interval, the operand width or the accumulator
/// width changes - which is exactly when someone needs to be told.
#[test]
fn the_int64_accumulator_has_a_stated_k_ceiling() {
    let m: u128 = 4095; // the largest licensed magnitude at the shipping interval
    let max_k = (i64::MAX as u128) / (m * m);

    // ~5.5e11. A K that large is not reachable, and saying so is the point:
    // the int64 half of the accumulation needs no runtime check.
    assert!(
        max_k > 5.0e11 as u128,
        "the K ceiling has moved to {max_k}, which is low enough that the int64 accumulation \
         now needs a check the kernel does not perform"
    );

    // And the claim that makes the split safe: the bound does not depend on how
    // the K range is divided. Any partition of K sums to the same total, so no
    // arrangement of threads or k-chunks can exceed it.
    for parts in [1u128, 2, 7, 64, 1024] {
        let per_part = max_k / parts;
        assert!(
            per_part * parts * m * m <= i64::MAX as u128,
            "a {parts}-way K-split exceeds the int64 bound, which would make the result depend \
             on the thread count - the exact property being sold"
        );
    }
}

// ───────────────────────── the representability gap ─────────────────────────
//
// `license`'s own doc says `m` is "in the scheme's own integer domain - the
// int16 operand values actually fed to `vpdpwssd` - NOT the real-valued range
// of the source matrices", and that converting between them "is the caller's
// job precisely because getting it wrong is a wrong answer rather than a slow
// one".
//
// The caller did not do it. `DriftAccumulator::operand_bounds` takes
// `max(|lo|, |hi|)` of the user's `@bounds` - real numbers - and hands them
// straight to `license`. Those two are the same number only when staging into
// int16 is the identity.
//
// So `@bounds(-0.001, 0.001)` was LICENSED at magnitude 0.001, and every operand
// of that magnitude stages to int16 zero. The kernel would compute the zero
// matrix while the licence certified it exact. It has never shipped a wrong
// answer only because the kernel is not wired in yet - `plan_exact_gemm`'s
// result is currently a `drift_report` advisory and the emitter falls back to
// scalar exact lowering. This is the licence that would become a wrong answer
// the moment Phase 0's wiring lands.
//
// Refused rather than guessed at a scale, which is the design rule's subject.

#[test]
fn a_magnitude_below_one_is_not_representable_and_is_refused() {
    let scheme = VnniExact::new(SHIPPING_T).unwrap();
    for m in [0.001, 0.5, 0.9999] {
        let err = scheme
            .license(m)
            .expect_err(&format!("{m} stages to int16 zero and must not be licensed"));
        assert!(
            err.contains("int16") || err.contains("representable"),
            "the refusal must name the domain mismatch rather than reading as an overflow: {err}"
        );
    }
}

/// The control, and it is what stops "refuse everything small" from passing.
///
/// Zero is representable - an all-zero operand really does stage to zero
/// correctly - and 1.0 is the smallest useful magnitude. Refusing either would
/// refuse correct programs.
#[test]
fn zero_and_one_are_still_licensed() {
    let scheme = VnniExact::new(SHIPPING_T).unwrap();
    assert!(
        scheme.license(0.0).is_ok(),
        "an all-zero operand is exactly representable"
    );
    assert!(scheme.license(1.0).is_ok(), "1.0 is a legal int16 magnitude");
    assert!(
        scheme.license(1024.0).is_ok(),
        "the magnitude the unit tests licence must stay licensed"
    );
}

/// The width limit can never fire first, so it exists for the MESSAGE - and a
/// test that asserts only "an error happened" cannot tell the difference.
///
/// `max_operand_magnitude` is `floor(sqrt(i32::MAX / 2T))`, which is 32767 at
/// T=1 and smaller at every larger interval. So it is never ABOVE `i16::MAX`,
/// and any magnitude exceeding the width also exceeds the overflow bound: the
/// overflow branch would refuse everything the width branch refuses.
///
/// The first version of this test asserted `license(32768.0).is_err()` and
/// **passed with the width check deleted** - the later check masked it, which
/// is the hole `feedback-mutation-holes-hide-behind-later-checks` describes.
/// It asserts WHICH diagnosis is given now, because that is the only thing the
/// width check actually decides: "does not fit int16" is a different repair
/// (narrow the operands) from "can overflow the int32 accumulator" (shorten the
/// flush interval), and telling a user the second when the first is true sends
/// them to tune a knob that cannot help.
#[test]
fn an_over_wide_operand_is_diagnosed_as_a_width_problem_not_an_overflow_one() {
    let tight = VnniExact::new(64).unwrap();
    assert_eq!(tight.max_operand_magnitude(), 4095.0);

    let loose = VnniExact::new(1).unwrap();
    assert_eq!(
        loose.max_operand_magnitude(),
        32767.0,
        "at T=1 the overflow bound lands exactly on i16::MAX"
    );

    for scheme in [&tight, &loose] {
        let err = scheme
            .license(32768.0)
            .expect_err("32768 does not fit int16 whatever the flush interval says");
        assert!(
            err.contains("int16"),
            "an operand too wide for int16 must be diagnosed as a WIDTH problem - the overflow \
             branch would also refuse it, and would send the user to shorten a flush interval \
             that cannot fix a width: {err}"
        );
        assert!(
            !err.contains("overflow the int32"),
            "the width case must not be reported as an accumulator overflow: {err}"
        );
    }
}
