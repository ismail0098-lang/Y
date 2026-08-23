//! An unknown INDEX yields an unknown value -- the dual of
//! `circom_opaque_array_index.rs`, which covers an unknown BASE.
//!
//! The motivating case is zk-email's `long_div` in `bigint-func.circom`:
//!
//! ```text
//! function long_div(n, k, m, a, b){
//!     var out[2][100];
//!     while (b[k-1] == 0) { out[1][k] = 0; k--; assert(k > 0); }
//!     ...
//!     out[1][k] = 0;
//! }
//! ```
//!
//! `b` is signal-derived, so the loop is havoc'd and `k` becomes unknown -- and
//! `k` is then an array index. circom's witness calculator runs this at WITNESS
//! time with concrete values; Y evaluates symbolically at compile time and
//! cannot, so it refused the index and `RSAVerifier65537` did not compile.
//!
//! **The boundary was PROBED against circom rather than reasoned about, and it
//! is not where I would have guessed.** circom draws the line at the USE, not
//! at the array being indexed:
//!
//! | construct                       | circom          | Y  |
//! |---------------------------------|-----------------|----|
//! | `out <-- t[u]`  (var, advice)   | accepted        | accepted |
//! | `out <-- in[u]` (SIGNAL, advice)| accepted        | accepted |
//! | `t[u] = 9`      (var store)     | accepted        | accepted |
//! | `out <== t[u]`  (var, constrain)| `error[T20462]` | refused |
//! | `out <== in[u]` (signal)        | `error[T20462]` | refused |
//! | `out[u] <== 1`  (lvalue)        | `error[T20462]` | refused |
//! | `c[u].x <== 3`  (component)     | `error[T20462]` | refused |
//!
//! So reading a SIGNAL array at an unknown index is fine as long as the value
//! only reaches a `<--`. The two that are refused *at the index* are the ones
//! where the index decides which wire a constraint mentions, or which
//! subcomponent exists -- neither may depend on a witness value.
//!
//! A store at an unknown index makes the WHOLE array unknown: some element
//! changed and there is no way to say which. Over-approximating in that
//! direction is the safe one, since an opaque value is refused at every site
//! that emits a constraint.
//!
//! Run with:  cargo test --features zk --test circom_opaque_index_value

#![cfg(feature = "zk")]

use std::path::{Path, PathBuf};
use y::circom_lower::compile_file;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn compile(name: &str) -> Result<y::zk_emitter::ZkEmitter, String> {
    compile_file(
        &PathBuf::from(root()).join("tests/circom").join(name),
        &[root().join("circomlib/circuits")],
    )
}

// ────────────────────────────────────────────────────────
// The relaxation
// ────────────────────────────────────────────────────────

/// Reading and writing at an unknown index is accepted in the witness domain,
/// and the circuit that comes out has exactly the constraints the author wrote.
///
/// The count is asserted against circom's, not against a number of Y's own
/// choosing: **a relaxation that drops constraints still compiles and still
/// proves, and only the reference can see it.** circom 2.2.3 emits 2
/// non-linear and 0 linear constraints for this fixture.
#[test]
fn an_unknown_index_is_ordinary_in_the_witness_domain() {
    let emitter = compile("opaque_index_advice.circom")
        .unwrap_or_else(|e| panic!("opaque_index_advice.circom failed to compile: {}", e));
    let circuit = emitter.build_circuit();
    assert_eq!(
        circuit.constraints.len(),
        2,
        "expected circom's 2 constraints (the author's two `<==`), got {}",
        circuit.constraints.len()
    );
    assert_eq!(circuit.outputs.len(), 2);
}

// ────────────────────────────────────────────────────────
// The refusals that make it safe
// ────────────────────────────────────────────────────────

fn refused_at_the_index(name: &str) {
    match compile(name) {
        Ok(_) => panic!(
            "{} compiled. circom refuses it with error[T20462]; an index that decides \
             which wire a constraint mentions may not depend on a witness value.",
            name
        ),
        Err(e) => assert!(
            e.contains("array indices must be known at compile time"),
            "{}: refused, but not at the INDEX -- so this case is not gating the check \
             it is named for: {}",
            name,
            e
        ),
    }
}

/// An unknown index on the LEFT of a `<==` decides which wire is constrained.
#[test]
fn an_unknown_index_in_an_lvalue_is_refused_at_the_index() {
    refused_at_the_index("opaque_index_into_lvalue.circom");
}

/// An unknown COMPONENT index decides which constraints exist at all.
#[test]
fn an_unknown_component_index_is_refused_at_the_index() {
    refused_at_the_index("opaque_index_into_component.circom");
}

/// **A STORE at an unknown index makes the whole array unknown.**
///
/// `t[u] = 99` changed some element and there is no way to say which, so
/// `t[3]` is unknown afterwards even though 3 is a constant. Without that,
/// `out <== t[3]` would build a constraint from 40 -- a value the program may
/// have just overwritten. circom refuses the same program (`error[T3001]`).
///
/// This is the soundness-critical case and the first version of this file did
/// not gate it: the only store fixture fed its result to a `<--`, which emits
/// no constraint, so the whole tainting rule could be deleted with every test
/// still green. Found by mutation.
#[test]
fn a_store_at_an_unknown_index_taints_the_whole_array() {
    match compile("opaque_index_store_taints.circom") {
        Ok(_) => panic!(
            "`t[u] = 99; out <== t[3];` compiled. The store at an unknown index did \
             not make the array unknown, so the constraint was built from a value the \
             program may have overwritten."
        ),
        Err(e) => assert!(
            e.contains("`<==` constraint") && e.contains("unknown"),
            "refused, but not for the array being unknown: {}",
            e
        ),
    }
}

/// An unknown index must not absorb a genuine typo.
///
/// Also a mutation-found hole: making `indexable_base` return `true` for every
/// name left the whole file green, because nothing indexed something that does
/// not exist.
#[test]
fn an_unknown_index_does_not_absorb_an_undeclared_base() {
    match compile("opaque_index_unknown_base.circom") {
        Ok(_) => panic!("`nope[u]` compiled, with `nope` never declared"),
        Err(e) => assert!(
            e.contains("`nope` is not defined"),
            "refused, but not because the base is undeclared -- so the message points \
             at the index instead of at the typo: {}",
            e
        ),
    }
}

/// Reading at an unknown index and feeding the result to a `<==` is still
/// refused -- at the CONSTRAINT, which is where the value stops being
/// witness-domain. circom refuses the same program.
///
/// This case used to be refused at the index and is in the census of
/// `circom_witness_domain.rs` for that reason; the site it is gating moved, so
/// the assertion moved with it rather than being quietly relaxed to "it
/// failed".
#[test]
fn an_unknown_index_reaching_a_constraint_is_still_refused() {
    for name in ["opaque_into_index.circom", "opaque_index_into_lvalue.circom"] {
        assert!(compile(name).is_err(), "{} must not compile", name);
    }
    let e = match compile("opaque_into_index.circom") {
        Err(e) => e,
        Ok(_) => unreachable!("asserted above"),
    };
    assert!(
        e.contains("`<==` constraint") && e.contains("unknown"),
        "expected a refusal naming the `<==` and the unknown value, got: {}",
        e
    );
}
