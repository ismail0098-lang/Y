//! `@zk_target` must not claim a backend Y does not have.
//!
//! `scheme = "plonkish"` parsed, was stored in `active_scheme`, was printed into
//! the `.r1cs.txt` header as `Proof Scheme: Plonkish` — and was then read by
//! nothing. The emitter has one arithmetization. A user selecting a PLONKish
//! backend got a clean compile, a success message, a header agreeing with them,
//! and an R1CS file.
//!
//! That is the same failure as `@ZeroDrift` before it was implemented (lexed,
//! counted, printed, read by no backend) and as the Hopper intrinsics that
//! assembled to nothing. For a proof system it is worse: the artifact would go
//! to a prover that cannot read it, or be assumed to carry soundness properties
//! belonging to a scheme that was never used.
//!
//! Run with:  cargo test --features zk --test zk_target_claims

#![cfg(feature = "zk")]

use y::lexer::Lexer;
use y::parser::Parser;
use y::zk_emitter::ZkEmitter;

fn compile(scheme: &str) -> Result<String, String> {
    let src = format!(
        "@zk_target(field = \"bn254\", scheme = \"{}\", opt_level = 1)\n\
         module SchemeProbe {{\n    fn main(x: I32, y: I32) -> I32 {{\n        return x * y;\n    }}\n}}\n",
        scheme
    );
    let tokens = Lexer::new(&src).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    ZkEmitter::new().emit_program(&program)
}

#[test]
fn plonkish_is_refused_not_silently_lowered_to_r1cs() {
    let err = compile("plonkish")
        .expect_err("`scheme = \"plonkish\"` must be refused; Y has no PLONKish backend");
    assert!(
        err.contains("plonkish") && err.contains("not implemented"),
        "the refusal must name the unsupported scheme and say it is unimplemented, got: {}",
        err
    );
    assert!(
        err.to_lowercase().contains("r1cs"),
        "the refusal must point at the backend that does exist, got: {}",
        err
    );
}

/// The control. Refusing everything is sound and useless.
#[test]
fn r1cs_still_compiles() {
    let out = compile("r1cs").expect("`scheme = \"r1cs\"` must still compile");
    assert!(
        out.contains("Rank-1 Constraint System"),
        "expected an R1CS report, got: {}",
        &out[..out.len().min(200)]
    );
}

/// No emitted artifact may describe a scheme that was not used.
///
/// The header line is what made the old behaviour convincing: it agreed with the
/// user. Whatever `Proof Scheme:` says has to be the thing on disk.
#[test]
fn emitted_header_never_names_an_unused_scheme() {
    let out = compile("r1cs").expect("r1cs compiles");
    assert!(
        !out.contains("Plonkish"),
        "the R1CS report names a scheme that was not emitted"
    );
    assert!(
        out.contains("Proof Scheme: R1cs"),
        "the R1CS report should state the scheme it actually emitted, got: {}",
        &out[..out.len().min(300)]
    );
}
