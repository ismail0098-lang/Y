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

// ---------------------------------------------------------------- the temperature

/// `KFix` is `C * 2^32`, and `tools/ptx_bridge.py` passed `C * 2^16`.
///
/// The kernel forms the exp's Q16.16 argument as
/// `t = ((m - s) * KFix + 2^15) >> 16`, so one factor of `2^16` is consumed by
/// the shift and `KFix` must carry `2^32`. The demo
/// (`tools/batch_invariance_demo.py`) has no shift — it builds the Q16.16 logit
/// directly — so its multiplier is `C * 2^16`. Two conventions, one name.
///
/// The bridge computed the demo's and handed it to the kernel. That makes every
/// exponent 65536x too small: on real Qwen activations no `t >> 16` reaches
/// 0.014, so every weight is within 1% of `2^28` and the softmax is uniform.
/// Its differential could not see it, because BOTH arms replicate the kernel's
/// formula and agree bit for bit on a uniform answer just as readily.
///
/// Nothing here can run the kernel — that needs a GPU — so this pins the
/// arithmetic the two sides have to share, and the emitted PTX is checked for
/// the multiply-and-shift shape rather than the `shl 3` it replaced.
#[test]
fn the_temperature_multiplier_carries_two_to_the_thirty_two() {
    // The kernel's integer path, exactly as the PTX writes it.
    let kernel_t = |ds: i64, kfix: i64| ((ds * kfix + 32768) >> 16) as f64;
    // The demo's float path: the Q16.16 argument, formed directly.
    let demo_t = |ds: i64, c: f64| (ds as f64 * c * 65536.0).round();

    // Score deltas span what an int8 dot product over head_dim <= 128 can
    // produce, and `c` spans the per-tensor scales real activations give.
    for &c in &[7.0e-5f64, 4.5e-4, 1.8e-3, 3.0e-2] {
        let kfix = (c * 2f64.powi(32)).round() as i64;
        let wrong = (c * 65536.0).round() as i64; // what the bridge passed
        for &ds in &[1i64, 97, 10_000, 100_000, 500_000, 2_064_512] {
            let want = demo_t(ds, c);
            let got = kernel_t(ds, kfix);
            // `kfix` is a rounded integer, so its error is `ds * 0.5 / 2^16`
            // in the argument. That is inherent to a fixed-point temperature
            // and is why `ptx_bridge.py` must compare the kernel against a
            // replica of the KERNEL's formula, not against the demo's.
            let tol = 1.0 + ds as f64 / 131_072.0;
            assert!(
                (got - want).abs() <= tol,
                "C={c:e} ds={ds}: kernel gave t={got}, the demo's Q16.16 \
                 argument is {want} (tolerance {tol:.2})"
            );
            // The control: the demo's own multiplier, fed to the kernel, is
            // not close — it is 2^16 out. Without this the test above would
            // pass for a `kfix` that is merely in the right ballpark.
            if ds >= 10_000 {
                let bad = kernel_t(ds, wrong);
                assert!(
                    bad < want / 1000.0,
                    "C={c:e} ds={ds}: passing the demo's C*2^16 to the kernel \
                     gave t={bad} against the correct {want}; the two \
                     conventions have stopped being distinguishable, so this \
                     test can no longer catch the bridge's bug"
                );
            }
        }
    }

    // The synthetic tests use C = 2^-13 and pass `8 << 16`. The docs claim that
    // "is exactly the old shift, so their semantics are unchanged" — which is
    // an arithmetic identity, and therefore checkable.
    let legacy = 8i64 << 16;
    for ds in 0..4096i64 {
        assert_eq!(
            (ds * legacy + 32768) >> 16,
            ds << 3,
            "KFix = 8<<16 must reproduce the kernel's old `shl 3` exactly"
        );
    }
}

/// The emitted PTX must actually be the runtime-temperature form.
#[test]
fn the_emitted_kernel_multiplies_by_a_runtime_temperature() {
    let ptx = attention_ptx(64, 4096).expect("64/4096 is well inside every bound");
    // Both entry points read the parameter and use the multiply-and-shift.
    assert_eq!(
        ptx.matches("ld.param.u32 %r50, [q6]").count(),
        2,
        "both accum entry points must load the runtime temperature"
    );
    assert_eq!(
        ptx.matches("mul.wide.s32 %rd50, %r16, %r50").count(),
        2,
        "the score delta must be multiplied by the runtime temperature, not \
         shifted by a compile-time constant"
    );
    assert_eq!(
        ptx.matches("shr.s64 %rd50, %rd50, 16").count(),
        2,
        "the >> 16 is what makes KFix a 2^32-scaled quantity; if it goes, every \
         caller's temperature is 65536x wrong"
    );
    // The saturate must come before the narrowing, or a large delta wraps into
    // a small argument and a far-away key gets a large weight.
    let sat = ptx.find("min.s64 %rd50, %rd50, 1073741824").expect("saturate");
    let narrow = ptx.find("cvt.u32.u64 %r16, %rd50").expect("narrowing");
    assert!(
        sat < narrow,
        "the argument must be clamped to 2^30 BEFORE it is narrowed to u32"
    );
}
