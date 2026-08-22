// ============================================================
//  The cross-pipeline quantization pass must CONVERT or REFUSE.
//
//  `emit_vectorized_quantization` implements 7 of the 64 ordered
//  (src, dst) precision pairs. The other 57 used to reach
//  `emit_scalar_fallback`, which wrote two PTX comments and no
//  instructions -- so the conversion silently did not happen and the
//  tensor core read the RT core's bits as a different type. It sits on
//  the `needs_quantization` path of `coprocessor_scheduler`, i.e. the
//  same `cross_pipeline_edges()` loop that emits the sync barriers.
//
//  Two properties are pinned here, and they fail independently:
//
//    * an implemented pair emits at least one INSTRUCTION, not only
//      comments -- the liveness canary that the stub would have failed;
//    * an unimplemented pair is refused by name, and the refusal
//      reaches the CLI as a non-zero exit rather than a comment block.
// ============================================================

use y::ir_grapher::Precision;
use y::quantization_pass::QuantizationPass;
use y::sentinel::HardwareProfile;

const ALL: [Precision; 8] = [
    Precision::FP32,
    Precision::FP16,
    Precision::BF16,
    Precision::TF32,
    Precision::FP8,
    Precision::FP4,
    Precision::INT8,
    Precision::INT4,
];

/// The pairs the pass implements. Anything else must be refused.
///
/// This list was SEVEN until the canary below was written. Five of the seven
/// converted nothing: FP8 and FP4/INT4 emitted `mov.u32 %qr, 0` -- a constant
/// zero -- under a comment claiming an E4M3 pack; INT8 initialised `absmax` to
/// 0.0, never computed it, and divided 127.0 by it; and four of the five never
/// emitted a store at all, so the destination kept the SOURCE's bits, which is
/// the exact reinterpretation the pass exists to prevent.
const IMPLEMENTED: [(Precision, Precision); 2] = [
    (Precision::FP32, Precision::FP16),
    (Precision::FP32, Precision::BF16),
];

fn emit(src: Precision, dst: Precision) -> Result<String, String> {
    let hw = HardwareProfile::default();
    QuantizationPass::new().emit_vectorized_quantization(src, dst, 0, 4096, &hw)
}

/// Lines that are neither blank nor a `//` comment.
fn instructions(ptx: &str) -> Vec<&str> {
    ptx.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with("//"))
        .collect()
}

#[test]
fn every_implemented_pair_emits_real_instructions() {
    // The canary. `emit_scalar_fallback` returned a well-formed, entirely
    // commented-out block -- it would pass any "did it return a string" or
    // "does it mention the precisions" assertion, and fails this one.
    for (src, dst) in IMPLEMENTED {
        let ptx = emit(src, dst).unwrap_or_else(|e| panic!("{src:?} -> {dst:?} refused: {e}"));
        let body = instructions(&ptx);
        assert!(
            body.len() >= 4,
            "{src:?} -> {dst:?} emitted {} instructions; a conversion that is \
             all comments does not convert anything:\n{ptx}",
            body.len()
        );
        assert!(
            body.iter().any(|l| l.starts_with("cvt.")),
            "{src:?} -> {dst:?} emitted no cvt instruction:\n{ptx}"
        );
        // The load-bearing half. A conversion that never STORES has converted
        // nothing: shared memory still holds the source bits when the tensor
        // core reads it. Four of the five deleted paths passed the `cvt`
        // assertion above and failed this one.
        assert!(
            body.iter().any(|l| l.starts_with("st.shared")),
            "{src:?} -> {dst:?} never stores its result, so SMEM keeps the \
             source bits:\n{ptx}"
        );
    }
}

#[test]
fn every_unimplemented_pair_is_refused_by_name() {
    let mut refused = 0;
    for src in ALL {
        for dst in ALL {
            if src == dst || IMPLEMENTED.contains(&(src, dst)) {
                continue;
            }
            let err = emit(src, dst).expect_err(&format!(
                "{src:?} -> {dst:?} is not implemented but was accepted"
            ));
            assert!(
                err.contains(&format!("{src:?}")) && err.contains(&format!("{dst:?}")),
                "refusal for {src:?} -> {dst:?} does not name the pair: {err}"
            );
            refused += 1;
        }
    }
    // 64 ordered pairs - 8 identities - 2 implemented.
    assert_eq!(refused, 54, "the pair census moved; update IMPLEMENTED");
}

#[test]
fn an_identity_conversion_is_a_no_op_not_a_refusal() {
    // Control. Refusing everything unimplemented is sound and useless if it
    // also refuses the case that legitimately needs no work -- the same shape
    // as `ordinary_loop_bodies_still_verify` guarding the SMT encoding.
    for p in ALL {
        let ptx = emit(p, p).unwrap_or_else(|e| panic!("{p:?} -> {p:?} refused: {e}"));
        assert!(
            instructions(&ptx).is_empty(),
            "{p:?} -> {p:?} emitted instructions for a conversion that is not needed:\n{ptx}"
        );
    }
}

#[test]
fn tf32_has_no_path_at_all() {
    // TF32 is a variant of `Precision` with no arm in the conversion table in
    // EITHER direction, which is what makes the refusal load-bearing rather
    // than theoretical: `output_precision` is a public field and setting it to
    // TF32 is the ordinary way to say an RT node produced TF32.
    for other in ALL {
        if other == Precision::TF32 {
            continue;
        }
        assert!(emit(Precision::TF32, other).is_err());
        assert!(emit(other, Precision::TF32).is_err());
    }
}
