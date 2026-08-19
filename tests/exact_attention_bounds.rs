//! What `attention_ptx` accepts, and what it must refuse.
//!
//! The module's whole claim is that the answer does not depend on how the
//! reduction was split. That holds while every partial sum fits its
//! accumulator, and the generator checked nothing at all: `head_dim` and
//! `seq_len` were pasted into a PTX template unexamined.
//!
//! The sequence bound was already known — `tests/gpu_attention_invariance.rs`
//! says in a comment "the i64 accumulator holds sequences to 2^28 keys" — and
//! enforced nowhere. Past it the sum WRAPS, which is a wrong answer and not an
//! imprecise one: the same failure the Python prototype's `K >= 133,153` int32
//! wrap turned out to be (`docs/bit_identical_decode.md` finding 04). A claim
//! recorded only in a comment is the shape this repo keeps finding.
//!
//! It also took a third argument, `c_hex`, "the softmax temperature folded with
//! log2(e)", documented as needing to stay a power of two "because the kernel's
//! argument conversion is a shift". The kernel was later changed to take the
//! temperature as a RUNTIME parameter — exactly so an arbitrary
//! `q_scale * k_scale / sqrt(d)` would work — and `$C` disappeared from the
//! template. The `.replace` for it matched nothing, so the argument was
//! accepted and discarded, and the doc-comment asserted the very restriction
//! the change had removed. Deleted rather than re-wired: the runtime parameter
//! is the design.
//!
//! Run with:  cargo test --release --test exact_attention_bounds

use y::exact_attention::{attention_ptx, MAX_EXACT_SEQ_LEN};

/// The bound is arithmetic, not a magic number: a Q0.28 weight times an int8
/// value, summed, must stay inside a signed 64-bit accumulator.
#[test]
fn the_sequence_bound_is_the_accumulator_width() {
    let max_term = ((1u128 << 28) - 1) * 127;
    let derived = ((1u128 << 63) / max_term) as usize;
    assert_eq!(
        MAX_EXACT_SEQ_LEN, derived,
        "MAX_EXACT_SEQ_LEN must be `2^63 / ((2^28 - 1) * 127)`; if the weight \
         scale or V's width changes, this constant has to move with it"
    );
    // And it must actually be safe at the limit, with a term to spare.
    assert!(
        (MAX_EXACT_SEQ_LEN as u128) * max_term < (1u128 << 63),
        "the largest accepted sequence already overflows the accumulator"
    );
}

#[test]
fn shapes_outside_the_exactness_argument_are_refused() {
    let cases: [(usize, usize, &str); 5] = [
        (0, 128, "head_dim > 0"),
        (64, 0, "seq_len > 0"),
        (64, MAX_EXACT_SEQ_LEN + 1, "accumulator"),
        (64, usize::MAX, "accumulator"),
        // `osm` is one u64 per output element: 8192 * 8 is past the 48 KB
        // static shared-memory limit, which ptxas would reject with a message
        // about the module rather than about the parameter.
        (8192, 128, "shared memory"),
    ];
    for (d, s, phrase) in cases {
        match attention_ptx(d, s) {
            Ok(_) => panic!(
                "attention_ptx({d}, {s}) produced a kernel. Outside the bound the \
                 reduction is no longer order-independent, which is the one thing \
                 this kernel exists to guarantee."
            ),
            Err(why) => assert!(
                why.contains(phrase),
                "attention_ptx({d}, {s}) was refused, but not for the reason under \
                 test. Wanted {phrase:?}, got: {why}"
            ),
        }
    }
}

/// The control, and it carries the weight: refusing everything would satisfy
/// the case above. The shapes a real model uses must still produce a kernel,
/// and the sequence length at the exact limit must be accepted rather than
/// rejected by an off-by-one.
#[test]
fn the_shapes_a_model_uses_still_generate() {
    for (d, s) in [(64usize, 128usize), (64, 4096), (128, 2048), (256, 512), (1, 1)] {
        let ptx = attention_ptx(d, s)
            .unwrap_or_else(|e| panic!("attention_ptx({d}, {s}) must generate: {e}"));
        assert!(
            ptx.contains(".visible .entry attn_scores")
                && ptx.contains(".visible .entry attn_accum")
                && ptx.contains(".visible .entry attn_accum_naive"),
            "attention_ptx({d}, {s}) lost an entry point"
        );
        // The template's placeholders must all be substituted; a leftover `$`
        // is a kernel ptxas rejects, and one that assembles is worse.
        assert!(
            !ptx.contains('$'),
            "attention_ptx({d}, {s}) left an unsubstituted template placeholder"
        );
    }
    assert!(
        attention_ptx(64, MAX_EXACT_SEQ_LEN).is_ok(),
        "the largest exactly-representable sequence must be accepted"
    );
}

/// The head dimension and the sequence length are baked into the kernel as
/// constants, so a generator that ignored them would produce the same text for
/// every shape and every launch would read the wrong strides.
#[test]
fn the_shape_reaches_the_generated_kernel() {
    let a = attention_ptx(64, 128).unwrap();
    let b = attention_ptx(128, 128).unwrap();
    let c = attention_ptx(64, 256).unwrap();
    assert_ne!(a, b, "head_dim did not reach the kernel");
    assert_ne!(a, c, "seq_len did not reach the kernel");
    assert!(a.contains("osm[512]"), "head_dim 64 must allocate 64 u64 of shared memory");
    assert!(b.contains("osm[1024]"), "head_dim 128 must allocate 128 u64 of shared memory");
}
