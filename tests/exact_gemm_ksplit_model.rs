//! The K-split model, checked against the theorem and against the emitter.
//!
//! `proofs/ExactGemmKsplit.v` proves that the exact GEMM's K-band
//! decomposition covers `[0, K)` exactly and that summing the per-band partials
//! equals the naive sum, for every `K` and every positive thread count. That is
//! a proof about a MODEL. `cpu_gemm::ksplit_bands` is the Rust transcription of
//! the same two definitions, and this file is what stops the two from drifting:
//!
//! - `the_bands_tile_at_every_thread_count` is the theorem's finite instance,
//!   run over the transcription rather than over the Coq definitions.
//! - `the_dropped_remainder_mutation_does_not_tile` re-runs, in Rust, the
//!   refutation the proof states - and is what makes the sweep above non-vacuous.
//!   A tiling check passes perfectly against a decomposition of nothing.
//! - `the_model_and_the_emitter_agree_on_their_constants` asserts the two
//!   AGREE rather than re-deriving the numbers here. A third copy of a constant
//!   is the bug, not the fix - the `.version` floor table taught that.
//!
//! **What this file cannot do** is prove the emitted LLVM implements the
//! transcription; there is no extraction. Two things narrow that gap and both
//! live elsewhere: `tests/exact_gemm_thread_invariance.rs` runs the real kernel
//! at six thread counts against an integer reference (a decomposition that does
//! not tile fails it - that was mutation 2 of 4), and its floor sweep asserts
//! the observed `pthread_create` count is exactly `ksplit_threads`'s answer.

use y::cpu_gemm::{
    emit_vnni_threaded_module, ksplit_bands, ksplit_threads, KSPLIT_MAX_THREADS, KSPLIT_MIN_BAND,
};

/// `bands_tile`, finitely: contiguous from 0, no gap, no overlap, ending at K.
#[test]
fn the_bands_tile_at_every_thread_count() {
    let mut checked = 0usize;
    let mut ragged = 0usize;
    for nthr in 1..=KSPLIT_MAX_THREADS {
        for k in 0..=3000usize {
            let bands = ksplit_bands(nthr, k);
            assert_eq!(bands.len(), nthr, "nthr={nthr} k={k}: wrong band count");

            let mut off = 0usize;
            for (i, (o, l)) in bands.iter().enumerate() {
                assert_eq!(
                    *o, off,
                    "nthr={nthr} k={k}: band {i} starts at {o}, but band {} ended \
                     at {off}. A gap loses terms; an overlap double-counts them.",
                    i.wrapping_sub(1)
                );
                off += l;
            }
            assert_eq!(
                off, k,
                "nthr={nthr} k={k}: the bands end at {off}. `bands_tile` says they \
                 must end at K - this is the dropped-remainder bug."
            );

            // The emitter's split is uneven by exactly one, which is what keeps a
            // band boundary from lining up with the flush interval.
            let lens: Vec<usize> = bands.iter().map(|(_, l)| *l).collect();
            let lo = *lens.iter().min().unwrap();
            let hi = *lens.iter().max().unwrap();
            assert!(
                hi - lo <= 1,
                "nthr={nthr} k={k}: band lengths span {lo}..{hi}; the decomposition \
                 is supposed to differ by at most one k"
            );
            if k % nthr != 0 {
                ragged += 1;
            }
            checked += 1;
        }
    }
    assert!(checked > 100_000, "the sweep covered only {checked} cases");
    assert!(
        ragged > 100_000,
        "only {ragged} of {checked} cases had a remainder to distribute; a sweep \
         over exact multiples cannot see the bug this exists for"
    );
}

/// `ksplit_exact`, finitely: the per-band partials sum to the whole sum.
///
/// This follows from tiling for an exact accumulate - which is the theorem's
/// point, not a redundancy. The same statement is FALSE under a rounding
/// accumulate, and `rounding_breaks_the_split` in the proof exhibits that.
#[test]
fn summing_the_band_partials_equals_the_naive_sum() {
    // A deterministic non-constant term, so a band that is skipped or counted
    // twice changes the total. A constant `f` would hide both.
    let f = |k: usize| -> i64 { (k as i64 % 97) * 31 - 1481 };
    for nthr in 1..=KSPLIT_MAX_THREADS {
        for k in [0usize, 1, 2, 63, 64, 127, 128, 129, 999, 1024, 4099] {
            let whole: i64 = (0..k).map(f).sum();
            let split: i64 = ksplit_bands(nthr, k)
                .iter()
                .map(|(o, l)| (*o..*o + *l).map(f).sum::<i64>())
                .sum();
            assert_eq!(
                split, whole,
                "nthr={nthr} k={k}: the K-split total disagrees with the naive sum"
            );
        }
    }
}

/// The mutation the proof refutes, re-run here.
///
/// Without this the tiling sweep above is not obviously testing anything: it
/// would pass against a decomposition that never has a remainder to lose.
#[test]
fn the_dropped_remainder_mutation_does_not_tile() {
    // `blen_flat` in the proof: `K / nthr` for every band, remainder discarded.
    let flat = |nthr: usize, k: usize| -> usize { nthr * (k / nthr) };

    // The concrete counterexample the proof computes: K=3, nthr=2 covers 2.
    assert_eq!(flat(2, 3), 2, "the mutation is supposed to cover only 2 of 3");
    assert_eq!(
        ksplit_bands(2, 3).iter().map(|(_, l)| *l).sum::<usize>(),
        3,
        "the shipped decomposition must cover all 3"
    );

    // ...and it is not a one-off: every ragged case loses terms.
    let mut lost = 0usize;
    for nthr in 2..=KSPLIT_MAX_THREADS {
        for k in 0..=500usize {
            if k % nthr != 0 {
                assert!(flat(nthr, k) < k);
                lost += 1;
            }
        }
    }
    assert!(lost > 1000, "only {lost} ragged cases were exercised");
}

/// The floor and the ceiling are decisions the emitted module makes; this
/// asserts the transcription AGREES with it rather than restating the numbers.
#[test]
fn the_model_and_the_emitter_agree_on_their_constants() {
    let module = emit_vnni_threaded_module(true);

    // The min-band floor: `%byk = sdiv i64 %K, <minband>`.
    let floor_line = format!("sdiv i64 %K, {KSPLIT_MIN_BAND}");
    assert!(
        module.contains(&floor_line),
        "the emitted wrapper does not divide K by {KSPLIT_MIN_BAND}; \
         `ksplit_threads` is transcribing a floor the code no longer uses"
    );

    // The thread ceiling, used twice: the compare and the select.
    let ceil_cmp = format!("icmp sgt i64 %r1, {KSPLIT_MAX_THREADS}");
    assert!(
        module.contains(&ceil_cmp),
        "the emitted wrapper does not clamp the request to {KSPLIT_MAX_THREADS} \
         threads; KSPLIT_MAX_THREADS is stale"
    );

    // The band arithmetic itself: base, remainder, and the uneven first `rem`.
    for needle in [
        "%base = sdiv i64 %K, %nthr",
        "%rem = srem i64 %K, %nthr",
        "%extra = icmp slt i64 %t, %rem",
    ] {
        assert!(
            module.contains(needle),
            "the emitted spawn loop no longer contains `{needle}`. \
             `ksplit_bands` and `proofs/ExactGemmKsplit.v` are transcriptions of \
             exactly these three lines - if the emitter's split changed, both \
             have to change with it."
        );
    }
}

/// Properties of the thread-count clamp, including the two boundaries the
/// floor sweep in `exact_gemm_thread_invariance.rs` then checks on real threads.
#[test]
fn the_thread_count_floors_at_one_and_never_hands_out_a_sliver() {
    // A request of zero (or an absurd one) is clamped, never propagated.
    assert_eq!(ksplit_threads(0, 100_000), 1);
    assert_eq!(ksplit_threads(1_000_000, 100_000), KSPLIT_MAX_THREADS);

    // Below one full band there is nothing to split.
    for k in 0..KSPLIT_MIN_BAND {
        assert_eq!(
            ksplit_threads(16, k),
            1,
            "K={k} is under one band, so the wrapper must take its direct path"
        );
    }
    // ...and the floor bites exactly at the boundary.
    assert_eq!(ksplit_threads(8, KSPLIT_MIN_BAND - 1), 1);
    assert_eq!(ksplit_threads(8, KSPLIT_MIN_BAND), 1);
    assert_eq!(ksplit_threads(8, 2 * KSPLIT_MIN_BAND), 2);
    assert_eq!(ksplit_threads(8, 8 * KSPLIT_MIN_BAND - 1), 7);
    assert_eq!(ksplit_threads(8, 8 * KSPLIT_MIN_BAND), 8);
    assert_eq!(ksplit_threads(8, 9 * KSPLIT_MIN_BAND), 8);

    // Never more workers than the request, and never a band under the floor.
    for req in 1..=KSPLIT_MAX_THREADS {
        for k in 0..=5000usize {
            let n = ksplit_threads(req, k);
            assert!(n >= 1 && n <= req);
            if n > 1 {
                let shortest = ksplit_bands(n, k).iter().map(|(_, l)| *l).min().unwrap();
                assert!(
                    shortest >= KSPLIT_MIN_BAND,
                    "req={req} k={k}: split into {n} gives a band of {shortest}, \
                     under the {KSPLIT_MIN_BAND} floor"
                );
            }
        }
    }
}
