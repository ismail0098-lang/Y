//! A circuit has no control flow, so a constraint in an untaken branch binds.
//!
//! `src/zk_fuzz.rs` has cited this file since the dead-branch attribution was
//! written. **It did not exist.** The limitation it names - the one the sweep
//! blames for the overwhelming majority of its over-refusals - was pinned by
//! nothing, so neither the behaviour nor the attribution that reports it had a
//! regression test. Same shape as the three verification scripts `docs/` named
//! that were never in the repo.
//!
//! TWO KINDS OF CONSTRAINT do this, and the attribution knew only the first:
//!
//!   * a RANGE check, from `emit_num2bits` on a gadget operand;
//!   * a BOOLEANITY check, from a dynamic `if` whose condition is not a bit.
//!
//! Both are fail-closed and intended. What was not intended is filing the
//! second as "something else": in a 3.2M-program sweep 22,526 findings were
//! reported as unexplained, and extending the attribution to booleanity moved
//! it from 90.3% to 93.8% of over-refusals explained.
//!
//! The asymmetry that makes these visible at all is worth stating. The
//! interpreter models both rules - it refuses an out-of-range operand and a
//! non-bit condition - but only where control flow REACHES them. So a
//! violation on the live path makes both sides refuse and produces no finding;
//! only a dead one makes the circuit alone refuse.
//!
//! Run with:  cargo test --features zk --test zk_dead_branch_range
#![cfg(feature = "zk")]

use y::zk_fuzz::{dead_branch_violation, run_circuit, FExpr, FProgram, FStmt, Op, Outcome};

fn eval(src: &str, inputs: &[u64]) -> Outcome {
    run_circuit(src, inputs)
}

/// The case `src/zk_fuzz.rs` documents verbatim.
#[test]
fn a_range_violation_in_an_untaken_branch_makes_the_circuit_unprovable() {
    let src = "fn main(p0: I32) -> I32 {\n    if (p0 < 5) {\n        return 1;\n    } else {\n        return ((0 - p0) < 3);\n    }\n}\n";
    // p0 = 1 takes the FIRST branch, whose value is a plain constant. The
    // else arm underflows - `0 - 1` is `p - 1` - and its range check is
    // emitted anyway.
    assert_eq!(eval(src, &[1]), Outcome::Unprovable);

    // **The control.** Without it this test is satisfied by a compiler that
    // refuses the program for any reason at all. Make the dead arm's operand
    // in range and the same shape compiles and answers.
    let ok = "fn main(p0: I32) -> I32 {\n    if (p0 < 5) {\n        return 1;\n    } else {\n        return ((7 - p0) < 3);\n    }\n}\n";
    assert!(
        matches!(eval(ok, &[1]), Outcome::Value(_)),
        "an in-range dead branch must not make the circuit unprovable"
    );
}

/// The kind the attribution did not know about.
#[test]
fn a_booleanity_violation_in_an_untaken_branch_does_too() {
    // `13 <= 2` is false, so the inner `if` is never taken. The circuit emits
    // its booleanity constraint regardless, and `p0 = 2` is not a bit.
    let src = "fn main(p0: I32) -> I32 {\n    if (13 <= p0) {\n        if p0 {\n        } else {\n        }\n    }\n    return 7;\n}\n";
    assert_eq!(eval(src, &[2]), Outcome::Unprovable);

    // **Two controls, and both are needed.** The same program with a BOOLEAN
    // value compiles - so it is the value and not the shape...
    assert_eq!(eval(src, &[1]), Outcome::Value(y::zk_field::Fr::from_u64(7)));

    // ...and the same violation on the LIVE path is equally unprovable, which
    // is what says the branch being dead is not what makes it fail. That case
    // produces no fuzzer finding only because the interpreter refuses it too.
    let live = "fn main(p0: I32) -> I32 {\n    if p0 {\n    }\n    return 7;\n}\n";
    assert_eq!(eval(live, &[2]), Outcome::Unprovable);
    assert_eq!(eval(live, &[1]), Outcome::Value(y::zk_field::Fr::from_u64(7)));
}

/// The attribution must recognise both, or the sweep files known limitations
/// as unexplained findings.
#[test]
fn the_attribution_recognises_both_kinds() {
    let p0 = || Box::new(FExpr::Param(0));
    let lit = |v: u64| Box::new(FExpr::Lit(v));

    // if (13 <= p0) { if p0 { } } return 7;   with p0 = 2
    let booleanity = FProgram {
        nparams: 1,
        locals_init: vec![],
        body: vec![
            FStmt::If {
                cond: FExpr::Bin(Op::Le, lit(13), p0()),
                then_b: vec![FStmt::If {
                    cond: FExpr::Param(0),
                    then_b: vec![],
                    else_b: None,
                }],
                else_b: None,
            },
            FStmt::Return(FExpr::Lit(7)),
        ],
    };
    assert!(
        dead_branch_violation(&booleanity, &[2]),
        "a non-bit `if` condition in an untaken branch is a dead-branch \
         violation and must be attributed as one"
    );

    // **The control.** A boolean value in the same program is not a violation,
    // so the attribution is not simply answering true.
    assert!(
        !dead_branch_violation(&booleanity, &[1]),
        "attribution fires on a program with no violation at all"
    );

    // if (13 <= p0) { return (0 - p0) < 3; } return 7;   with p0 = 2
    let range = FProgram {
        nparams: 1,
        locals_init: vec![],
        body: vec![
            FStmt::If {
                cond: FExpr::Bin(Op::Le, lit(13), p0()),
                then_b: vec![FStmt::Return(FExpr::Bin(
                    Op::Lt,
                    Box::new(FExpr::Bin(Op::Sub, lit(0), p0())),
                    lit(3),
                ))],
                else_b: None,
            },
            FStmt::Return(FExpr::Lit(7)),
        ],
    };
    assert!(
        dead_branch_violation(&range, &[2]),
        "the range-check case the attribution was written for"
    );
}

/// A recipe standing on an `Unknown` dependency must not count as SOLVED.
///
/// Reduced from an unattributed over-refusal in the 3.2M sweep, and the shape
/// is the canonical search loop. A satisfiable circuit was reported unprovable
/// because the witness solver assigned 0 to the output wire, so
/// `--target=r1cs --witness` refused to write a `.wtns` for a circuit that has
/// one - the worst-shaped failure this backend can produce short of a wrong
/// answer, because it reads as "your circuit is impossible".
///
/// # The mechanism
///
/// `execute_host_witness_ir` evaluates a recipe unconditionally, and `eval_lc`
/// reads an unassigned wire as ZERO - so a recipe whose operand is `Unknown`
/// still produces a number. `solve_r1cs_witness` then marked a wire solved from
/// its node's KIND alone, which told back-propagation the wire was settled;
/// nothing revisited it even once the dependency was rediscovered.
///
/// Here the loop's predicated return builds a `ret_mux` pinned by a constraint
/// whose `C` has two terms (`-1*prev + 1*mux`), which neither scan in
/// `build_witness_ir` can turn into a recipe, so it is `Unknown`. The output
/// binding above it IS a recipe and read that zero. Back-propagation recovered
/// the mux correctly - and left the output at 0.
///
/// # Why the trigger looked so narrow
///
/// A NON-ZERO tail makes the output binding's own `C` two-term as well
/// (`-9*one + 1*out`), so the output is `Unknown` too and back-propagation owns
/// both wires. That is why `return 9;` always worked and `return 0;` did not,
/// and why the first hypothesis - that the returned value being a parameter was
/// the trigger - was wrong. Bisection settled it; the difference is which side
/// of the solver owns the wire, not the program's shape.
///
/// The narrow-looking trigger is therefore a coincidence of two arithmetic
/// facts, so the assertions below sweep the axes it was bisected along rather
/// than pinning the one reproducer.
#[test]
fn a_recipe_standing_on_an_unknown_is_not_solved() {
    let mk = |iters: u32, tail: &str| {
        format!(
            "fn main(p0: I32) -> I32 {{\n    @invariant(i0 >= 0)\n    for i0 in 0..{} {{\n        if (9 >= p0) {{\n            return p0;\n        }}\n    }}\n    return {};\n}}\n",
            iters, tail
        )
    };
    // The reproducer. Two iterations, one-sided return, zero tail.
    assert_eq!(
        eval(&mk(2, "0"), &[2]),
        Outcome::Value(y::zk_field::Fr::from_u64(2)),
        "the loop returns p0 on its first iteration, so the answer is 2"
    );
    // Both axes it was bisected along. The non-zero tails always worked - they
    // are the CONTROL that says the fix did not simply stop solving.
    for tail in ["0", "1", "9"] {
        for iters in [1u32, 2, 3, 4] {
            let src = mk(iters, tail);
            assert_eq!(
                eval(&src, &[2]),
                Outcome::Value(y::zk_field::Fr::from_u64(2)),
                "iters={} tail={}: the condition holds, so the loop returns p0",
                iters,
                tail
            );
            // And when the condition is FALSE the tail decides, which is the
            // other half of the mux and would be silently right if the solver
            // had simply started answering 0 everywhere.
            let expect: u64 = tail.parse().unwrap();
            assert_eq!(
                eval(&src, &[40]),
                Outcome::Value(y::zk_field::Fr::from_u64(expect)),
                "iters={} tail={}: 9 >= 40 is false, so the tail decides",
                iters,
                tail
            );
        }
    }
}
