//! Resolve names and complete signatures before any backend emits code.
use y::{lexer::Lexer, parser::Parser, type_checker::TypeChecker};

fn errors(source: &str) -> Vec<String> {
    let ast = Parser::new(Lexer::new(source).tokenize())
        .parse_program()
        .unwrap();
    let mut tc = TypeChecker::new();
    tc.check_program(&ast);
    tc.errors
}

fn rejects(source: &str, diagnostic: &str) {
    let found = errors(source);
    assert!(
        found.iter().any(|e| e.contains(diagnostic)),
        "expected {diagnostic}: {source}\n{found:?}"
    );
}

#[test]
fn undeclared_names_are_not_unknown_typed_values() {
    rejects(
        "fn main() -> I32 { return nonexistent; }",
        "Undefined variable",
    );
    rejects("fn main() { nonexistent(1); }", "Unknown function");
    rejects("fn main() { let n = 3; n(); }", "not callable");
    rejects("fn main() { for i in missing..3 {} }", "Undefined variable");
}

#[test]
fn calls_validate_arity_argument_types_and_return_types() {
    for call in ["f()", "f(1, 2)"] {
        rejects(
            &format!("fn f(x: I32) -> I32 {{ return x; }} fn main() {{ {call}; }}"),
            "argument(s)",
        );
    }
    rejects(
        "fn f(x: I32) -> I32 { return x; } fn main() { f(true); }",
        "argument 1",
    );
    rejects("fn f(x: I32) {} fn main() { f(1.5); }", "argument 1");
    rejects(
        "fn f(x: &F32) {} fn main() { let x: I32 = 1; f(&x); }",
        "argument 1",
    );
    rejects(
        "fn f() -> bool { return true; } fn main() { let n: I32 = f(); }",
        "Type mismatch",
    );
    rejects("fn f() -> I32 { return true; }", "return type");
    rejects("fn f() -> I32 { return; }", "return type");
}

#[test]
fn operators_have_types_and_boolean_conditions_are_required() {
    rejects("fn main() { if 1 + 2 {} }", "condition has type");
    rejects("fn main() { let n = 1 + true; }", "operands");
    rejects("fn main() { let n: I32 = 1; n += true; }", "operands");
    rejects("fn main() { if 1 && 2 {} }", "boolean");
    rejects("fn main() { let x: F32 = 1.0; let n = x & x; }", "integer");
    rejects("fn main() { if \"yes\" {} }", "condition has type");
}

#[test]
fn valid_forward_calls_polymorphic_literals_and_struct_fields_work() {
    for source in [
        "fn main() -> I32 { let n = twice(4); if n > 0 { return n; } return 0; } fn twice(x: I32) -> I32 { return x + x; }",
        "fn f(x: F16) -> F16 { return x + 1.0; } fn main() { let n: F16 = f(2.0); }",
        "fn f(x: F16) -> bool { return 1.0 < x; } fn main() { if f(2.0) {} }",
        "fn f() -> bool { return true; } fn main() { if f() && (1 + 2 == 3) {} }",
        "fn main() -> I32 { let predicate: I32 = 2 > 1; return (predicate == 1) * 5; }",
        "fn f(p: P) -> I32 { return p.x; } struct P { x: I32 } fn main() { let p: P = P { x: 2 }; f(p); }",
    ] {
        assert!(errors(source).is_empty(), "{source}\n{:?}", errors(source));
    }
}
