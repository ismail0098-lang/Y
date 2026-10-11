//! `@bounds` in a circuit constrains both ends of its range.
//!
//! The ZK backend decomposed the value into the bit width of `max` and nothing
//! else. `min` was ignored, so `@bounds(10, 20)` admitted 0..9 - values the
//! source rules out, which a prover could use - and `@bounds(-8, 7)` refused
//! every negative value the source allows. A bound that was not a constant was
//! skipped without a word, and the bit wires carried no witness recipe. Now
//! `value - min` and `max - value` each decompose into the width of
//! `max - min`, through `emit_num2bits`, and a non-constant bound is refused.
//!
//! Run with:  cargo test --features zk --test zk_bounds_constraint

#![cfg(feature = "zk")]

use y::zk_emitter::Fr;
use y::zk_fuzz::{run_circuit, Outcome};

fn neg(v: u64) -> Fr {
    Fr::zero().sub(&Fr::from_u64(v))
}

fn check(src: &str, cases: &[(u64, Option<Fr>)]) {
    for (input, want) in cases {
        match (run_circuit(src, &[*input]), want) {
            (Outcome::Value(v), Some(w)) => assert_eq!(v, *w, "p = {input}"),
            (Outcome::Unprovable, None) => {}
            (other, _) => panic!("p = {input}: want {want:?}, got {other:?}\n{src}"),
        }
    }
}

#[test]
fn both_ends_of_the_range_are_constrained() {
    let src = "fn main(p: I32) -> I32 {\n    @bounds(min=10, max=20)\n    let b = p;\n    return b;\n}\n";
    let v = |n| Some(Fr::from_u64(n));
    // 0..9 were provable: only `max` was constrained.
    check(src, &[(10, v(10)), (15, v(15)), (20, v(20)), (9, None), (5, None), (0, None), (21, None)]);
}

#[test]
fn a_negative_range_admits_its_negative_values() {
    // Every negative value was unprovable: the value itself was decomposed.
    let src = "fn main(p: I32) -> I32 {\n    @bounds(min=-8, max=7)\n    let b = p - 10;\n    return b;\n}\n";
    check(
        src,
        &[(2, Some(neg(8))), (7, Some(neg(3))), (10, Some(Fr::zero())), (17, Some(Fr::from_u64(7))), (1, None), (18, None)],
    );
}

#[test]
fn a_bound_that_is_not_a_constant_is_refused() {
    let src = "fn main(p: I32, q: I32) -> I32 {\n    @bounds(min=0, max=q)\n    let b = p;\n    return b;\n}\n";
    assert!(matches!(run_circuit(src, &[3, 5]), Outcome::Rejected), "a bound the circuit cannot constrain was ignored");
}
