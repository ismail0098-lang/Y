//! Does the `@tile` rejection message name every shape that is actually accepted?
//!
//! It did not. The message listed three shapes -- plain GEMM, GEMM+Bias+ReLU,
//! Linear+SwiGLU -- while the checker also accepted the int8 Tensor Core GEMM
//! `(A, B: I8, C: I32)` and the 5-parameter FP8 shape. So a user one parameter
//! away from the working int8 form was told
//!
//!     supports no other shape ... A (expected GlobalMemory<F16>)
//!
//! which denies a feature that exists and steers them to convert their int8
//! operands to F16. A diagnostic that is merely unhelpful costs a user minutes;
//! one that is WRONG about what the compiler supports costs them the feature.
//!
//! This is the inverse of `feedback_docs_assert_absent_optimizations`: there,
//! documentation claimed an optimisation the code did not have. Here the code
//! had a capability its own error message denied. Both are the same failure --
//! prose and behaviour drifting apart with nothing checking.
//!
//! The guard runs in both directions. Every shape the message advertises must
//! COMPILE, and every shape that compiles must be NAMED. Neither half alone is
//! enough: listing shapes that do not work is as bad as omitting ones that do.

use std::path::Path;
use std::process::Command;

/// Every probe gets its OWN directory. Keying it on `src.len()` collided
/// between tests running on different threads -- one removed the directory
/// while another was reading its output, and the failure looked like the
/// compiler rejecting a shape it accepts. A counter is enough and cannot
/// collide.
static PROBE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn compile(src: &str) -> (bool, String) {
    let n = PROBE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "y_tile_shape_{}_{}",
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("probe.ysu");
    std::fs::write(&f, src).unwrap();
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    let out = Command::new(bin.join("Y"))
        .arg(&f)
        .arg("--emit-ptx")
        .current_dir(repo)
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), text)
}

/// (a name for the failure message, the parameter list) -- one per shape the
/// checker accepts.
/// (name, `@tile` dimensions, parameter list).
///
/// The tile is per-shape because the fused SwiGLU kernel has a FIXED 128x128x32
/// CTA and no tail path, so it hard-asserts that M, N and K are multiples of
/// it. Handing every shape the same `@tile(64, 32, 128)` made this test report
/// "the Linear+SwiGLU shape does not compile", which was true and had nothing
/// to do with the shape being unsupported -- the first version of this file
/// blamed the wrong thing.
const SHAPES: &[(&str, &str, &str)] = &[
    (
        "plain GEMM",
        "64, 32, 128",
        "A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>",
    ),
    (
        "GEMM+Bias+ReLU",
        "64, 32, 128",
        "A: GlobalMemory<F16>, B: GlobalMemory<F16>, Bias: GlobalMemory<F32>, C: GlobalMemory<F32>",
    ),
    (
        "Linear+SwiGLU",
        "128, 128, 32",
        "X: GlobalMemory<F16>, Wg: GlobalMemory<F16>, Wu: GlobalMemory<F16>, Out: GlobalMemory<F32>",
    ),
    (
        "int8 Tensor Core GEMM",
        "64, 32, 128",
        "A: GlobalMemory<I8>, B: GlobalMemory<I8>, C: GlobalMemory<I32>",
    ),
];

#[test]
fn every_advertised_shape_actually_compiles() {
    for (name, tile, params) in SHAPES {
        let src = format!("@tile({tile})\nkernel k({params}) {{\n}}\n\nfn main() {{}}\n");
        let (ok, text) = compile(&src);
        assert!(
            ok,
            "the `{name}` shape is advertised by the @tile diagnostic but does \
             not compile:\n{text}"
        );
    }
}

#[test]
fn the_rejection_message_names_every_accepted_shape() {
    // A near miss: the int8 shape with one extra parameter. Before the fix this
    // produced a list that did not mention I8 at all.
    let src = "@tile(64, 32, 128)\nkernel k(A: GlobalMemory<I8>, B: GlobalMemory<I8>, \
               C: GlobalMemory<I32>, D: GlobalMemory<I32>) {\n}\n\nfn main() {}\n";
    let (ok, text) = compile(src);
    assert!(!ok, "the 4-parameter int8 shape should be rejected:\n{text}");

    for needle in [
        "GlobalMemory<I8>",       // the int8 GEMM
        "int8 Tensor Core GEMM",
        "e4m3",                   // the FP8 shape
        "scale_a",
        "SwiGLU",
        "Bias",
    ] {
        assert!(
            text.contains(needle),
            "the @tile rejection message does not mention {needle:?}, so a user \
             hitting it is not told about a shape the compiler accepts:\n{text}"
        );
    }
}

#[test]
fn the_control_a_genuinely_unsupported_shape_is_still_refused() {
    // Without this, "name every shape" could be satisfied by accepting
    // everything. Two F32 buffers is not any of the listed shapes.
    let src = "@tile(64, 32, 128)\nkernel k(A: GlobalMemory<F32>, B: GlobalMemory<F32>, \
               C: GlobalMemory<F32>) {\n}\n\nfn main() {}\n";
    let (ok, _) = compile(src);
    assert!(!ok, "an F32/F32/F32 tile GEMM is not a supported shape and must be refused");
}
