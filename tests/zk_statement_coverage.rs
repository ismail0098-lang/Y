//! Every `Stmt` variant the ZK backend can meet, and what it does with it.
//!
//! `emit_stmt` ended in `_ => {}`. Ten variants were handled and the rest were
//! silently skipped, which in this backend means the emitted circuit computes
//! a different function than the program — and Groth16 proves the different
//! one just as readily. Two were live:
//!
//! ```text
//! let x: I32 = a;  x += 5;              return x;   // emitted out = a
//! let x: I32 = a;  @ghost { x = x + 5; } return x;   // emitted out = a
//! ```
//!
//! Both compiled clean, printed "Compilation Successful!", solved to a
//! satisfying witness, and wrote a `.r1cs` whose single constraint was
//! `w_1 * 1 = w_2`. The correct circuit is `(5 + w_1) * 1 = w_2`, which is
//! exactly what the same program spelled `x = x + 5;` had always emitted — so
//! the backend disagreed with itself about what the source meant.
//!
//! `emit_stmt` is exhaustive now, with no `_ =>` arm, so a new statement kind
//! is a compile error rather than a circuit that quietly omits it.
//!
//! The tests state VALUES, not constraint shapes. A differential against the
//! desugared spelling alone would be vacuous if both forms were dropped, so
//! every case also pins the arithmetic answer.
//!
//! Run with:  cargo test --features zk --test zk_statement_coverage

#![cfg(feature = "zk")]

use y::zk_emitter::ZkEmitter;
use y::zk_field::Fr;
use y::zk_witness::solve_r1cs_witness;

/// Compile `source` to R1CS and solve it for `inputs`, or `None` if the system
/// has no satisfying assignment.
fn eval(source: &str, inputs: &[u64]) -> Option<String> {
    let tokens = y::lexer::Lexer::new(source).tokenize();
    let prog = y::parser::Parser::new(tokens)
        .parse_program()
        .expect("program did not parse");
    let mut emitter = ZkEmitter::new();
    emitter
        .emit_program(&prog)
        .expect("program did not compile to R1CS");

    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();
    let privs: Vec<Fr> = inputs.iter().map(|v| Fr::from_u64(*v)).collect();
    let (witness, satisfied) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);
    if !satisfied {
        return None;
    }
    let out = *circuit.outputs.first().expect("circuit has no output");
    Some(witness[out].to_decimal_string())
}

/// The emitter's refusal message for `source`, or `None` if it compiled.
fn refusal(source: &str) -> Option<String> {
    let tokens = y::lexer::Lexer::new(source).tokenize();
    let prog = y::parser::Parser::new(tokens)
        .parse_program()
        .expect("program did not parse");
    ZkEmitter::new().emit_program(&prog).err()
}

fn body(stmts: &str) -> String {
    format!("fn main(a: I32) -> I32 {{\n    let x: I32 = a;\n{}\n    return x;\n}}\n", stmts)
}

/// `x op= e` must mean `x = x op e`, and must mean it in the circuit.
///
/// Both spellings are checked against each other AND against the arithmetic,
/// because "the compound form emits nothing" and "both forms emit nothing"
/// are indistinguishable from the differential alone.
#[test]
fn compound_assignment_is_not_dropped() {
    // (operator, right-hand side, input, expected output)
    let cases = [
        ("+=", "5", 7u64, 12u64),
        ("-=", "5", 7, 2),
        ("*=", "3", 7, 21),
        ("/=", "2", 7, 3),
    ];
    for (op, rhs, input, want) in cases {
        let compound = body(&format!("    x {} {};", op, rhs));
        let desugared = body(&format!("    x = x {} {};", op.trim_end_matches('='), rhs));

        let got = eval(&compound, &[input]);
        assert_eq!(
            got.as_deref(),
            Some(want.to_string().as_str()),
            "`x {} {};` on x = {} must give {}. It used to emit no constraint at \
             all, so the circuit computed the identity and proved it.",
            op,
            rhs,
            input,
            want
        );

        // The control. If this ever disagrees, the bug is in the desugaring
        // and not in the arm that was missing.
        assert_eq!(
            eval(&desugared, &[input]).as_deref(),
            got.as_deref(),
            "`x {} {};` and `x = x {} {};` are the same statement and must emit \
             the same function",
            op,
            rhs,
            op.trim_end_matches('='),
            rhs
        );
    }
}

/// `@safe` and `@ghost` are ordinary blocks in every other backend — PTX,
/// LLVM and the host emitter all lower them by emitting the body — and this
/// one dropped them whole.
#[test]
fn a_block_statement_body_is_not_dropped() {
    for kw in ["@safe", "@ghost"] {
        let src = body(&format!("    {} {{\n        x = x + 5;\n    }}", kw));
        assert_eq!(
            eval(&src, &[7]).as_deref(),
            Some("12"),
            "`{} {{ x = x + 5; }}` must constrain x. Dropping the block emitted \
             a circuit computing the identity.",
            kw
        );
    }
}

/// A `return` inside such a block still terminates the function, so the
/// predicated `Ret` has to come back out of it rather than being discarded —
/// the `Stmt::For` row of the design-rule table, in a different statement.
#[test]
fn a_return_inside_a_block_statement_still_returns() {
    let src = "fn main(a: I32) -> I32 {\n    @ghost {\n        return 3;\n    }\n    return 9;\n}\n";
    assert_eq!(
        eval(src, &[7]).as_deref(),
        Some("3"),
        "the `return` inside the block was dropped and the function fell \
         through to the tail"
    );
}

/// Constructs with no R1CS meaning are refused BY NAME. Each case asserts on
/// its own phrase: a fixture stopped by some earlier check would otherwise
/// pass while proving nothing about the arm under test.
#[test]
fn hardware_only_statements_are_refused_by_name() {
    let chisel = "fn main(a: I32) -> I32 {\n    let x: I32 = a;\n    chisel {\n        x = x + 1;\n    }\n    return x;\n}\n";
    let msg = refusal(chisel).unwrap_or_default();
    assert!(
        msg.contains("chisel"),
        "a `chisel` block is direct hardware access and has no R1CS form; it \
         must be refused by name, not skipped. Got: {:?}",
        msg
    );

    let clock = "fn main(a: I32) -> I32 {\n    let x: I32 = a;\n    @clock_domain(a) {\n        x = x + 1;\n    }\n    return x;\n}\n";
    let msg = refusal(clock).unwrap_or_default();
    assert!(
        msg.contains("clock_domain"),
        "`@clock_domain` is an HDL construct and has no R1CS form. Got: {:?}",
        msg
    );
}

/// The control for the two refusals above: refusing everything would satisfy
/// them and check nothing. A `type` alias and a `compile_time::assert!` emit
/// no constraint ON PURPOSE — the first is erased before any backend sees it,
/// the second is discharged by the type checker and is zero-cost by
/// definition — so the surrounding program must still compile and still be
/// correct.
#[test]
fn statements_that_correctly_emit_nothing_still_compile() {
    let src = "fn main(a: I32) -> I32 {\n    type T = I32;\n    let x: I32 = a;\n    compile_time::assert!(1 == 1, \"trivial\");\n    x += 5;\n    return x;\n}\n";
    assert_eq!(
        eval(src, &[7]).as_deref(),
        Some("12"),
        "an erased statement must not disturb the statements around it"
    );
}

/// The desugaring builds `x = x op e`, so `e` may mention `x` itself. That is
/// exactly the shape `Stmt::Assign`'s running-sum fast path guards against
/// with `expr_references_var` — it takes the target's binding OUT of scope
/// before emitting the right-hand side, so a missed self-reference would read
/// a name that is no longer bound.
///
/// Audited rather than assumed: the failure mode is fail-closed. An unbound
/// identifier is `Err("Undefined variable ...")` in `emit_expr`, never a fresh
/// wire, so a gap in that guard costs a compile error and not a wrong circuit.
/// These cases pin the values regardless, because the guard is what keeps the
/// error from happening at all.
#[test]
fn a_self_referencing_right_hand_side_is_computed_correctly() {
    let cases = [
        ("x = x + x;", 14u64),
        ("x += x;", 14),
        ("x = x + (x * 2);", 21),
        ("x += x * 2;", 21),
        ("x -= x;", 0),
        ("x *= x;", 49),
        ("x += 1; x += x;", 16),
    ];
    for (stmts, want) in cases {
        let src = body(&format!("    {}", stmts));
        assert_eq!(
            eval(&src, &[7]).as_deref(),
            Some(want.to_string().as_str()),
            "`{}` on x = 7 must give {}",
            stmts,
            want
        );
    }
}

/// Expression shapes the ZK backend cannot lower must be REFUSED, not guessed
/// at. `emit_expr`'s catch-all is an error and this pins that it stays one —
/// the whole design-rule table is instances of that arm having been a value.
#[test]
fn unlowerable_expressions_are_refused() {
    let cases = [
        ("unary minus", "fn main(a: I32) -> I32 { return -a; }"),
        ("unary not", "fn main(a: I32) -> I32 { if !a { return 1; } return 0; }"),
        ("float literal", "fn main(a: I32) -> I32 { let x: I32 = 1.5; return x; }"),
        ("member access", "fn main(a: I32) -> I32 { return a.x; }"),
    ];
    for (what, src) in cases {
        let msg = refusal(src).unwrap_or_default();
        assert!(
            msg.contains("unsupported in ZK backends"),
            "{} has no R1CS lowering and must be refused by name. Got: {:?}",
            what,
            msg
        );
    }
}
