//! Numeric scanning must preserve an exact value or report a source error.
//! The AST currently carries signed i64 literals, so larger U64 magnitudes
//! are explicitly refused until unsigned literals have complete lowering.

use std::process::Command;
use y::ast::{Expr, Item, Program, Stmt};
use y::lexer::{Lexer, TokenKind};
use y::parser::Parser;

fn parse(source: &str) -> Result<Program, String> {
    Parser::new(Lexer::new(source).tokenize()).parse_program()
}

fn returned_literal(source: &str) -> Expr {
    let program = parse(source).expect("valid literal parses");
    let Item::Func(function) = &program.items[0] else {
        panic!("expected a function");
    };
    let Stmt::Return(Some(value), _) = &function.body.stmts[0] else {
        panic!("expected a return value");
    };
    value.clone()
}

#[test]
fn integer_overflow_preserves_its_spelling_and_is_refused() {
    for literal in [
        "9223372036854775808",
        "18446744073709551615",
        "18446744073709551616",
        "99999999999999999999999999999999999999999999",
    ] {
        let tokens = Lexer::new(literal).tokenize();
        assert_eq!(tokens[0].lexeme, literal);
        assert!(
            !matches!(
                tokens[0].kind,
                TokenKind::IntLit(_) | TokenKind::FloatLit(_)
            ),
            "overflow must not become a numeric value: {:?}",
            tokens[0]
        );
        let error = parse(&format!("fn value() -> I64 {{ return {literal}; }}"))
            .expect_err("out-of-range integer must be refused");
        assert!(error.contains("Invalid integer literal"), "{error}");
        assert!(error.contains(literal), "{error}");
    }
}

#[test]
fn integer_errors_name_the_literal_and_supported_range_in_every_context() {
    for source in [
        "fn main() -> U64 { return 18446744073709551615; }",
        "fn main() -> I64 { return -9223372036854775809; }",
        "fn main() -> I64 { return 0 - 9223372036854775808; }",
        "fn main() { let values: [I32; 18446744073709551615]; }",
        "fn main() { let tile: BlockTile<F32, 18446744073709551615>; }",
        "@tile(18446744073709551615, 16, 16) kernel bad() {}",
    ] {
        let error = parse(source).expect_err("out-of-range number must be refused");
        assert!(error.contains("Invalid integer literal"), "{error}");
        assert!(error.contains("signed 64-bit literal range"), "{error}");
        assert!(
            error.contains("Unsigned literals above I64::MAX"),
            "{error}"
        );
        assert!(error.contains("Line 1, column"), "{error}");
    }
}

#[test]
fn representable_nonnegative_integers_are_preserved() {
    for value in [i64::MAX, 9007199254740993, 0] {
        let expr = returned_literal(&format!("fn value() -> I64 {{ return {value}; }}"));
        assert!(matches!(expr, Expr::IntLit(actual, _) if actual == value));
    }
}

#[test]
fn signed_minimum_is_preserved_including_leading_zeroes() {
    for literal in ["-9223372036854775808", "-00009223372036854775808"] {
        let expr = returned_literal(&format!("fn value() -> I64 {{ return {literal}; }}"));
        assert!(
            matches!(expr, Expr::IntLit(i64::MIN, _)),
            "signed minimum was changed: {expr:?}"
        );
    }
}

fn assert_float_refused(literal: &str) {
    let tokens = Lexer::new(literal).tokenize();
    assert_eq!(tokens[0].lexeme, literal);
    assert!(
        !matches!(
            tokens[0].kind,
            TokenKind::IntLit(_) | TokenKind::FloatLit(_)
        ),
        "invalid float must not become a numeric value: {:?}",
        tokens[0]
    );
    let error = parse(&format!("fn value() -> F64 {{ return {literal}; }}"))
        .expect_err("invalid float must be refused");
    assert!(error.contains("Invalid floating-point literal"), "{error}");
    assert!(error.contains(literal), "{error}");
}

#[test]
fn malformed_float_is_refused_instead_of_becoming_zero() {
    assert_float_refused("1.2.3");
}

#[test]
fn nonfinite_float_is_refused() {
    assert_float_refused(&format!("{}.0", "9".repeat(400)));
}

#[test]
fn ordinary_decimal_floats_and_range_tokens_still_work() {
    let expr = returned_literal("fn value() -> F64 { return 12.25; }");
    assert!(matches!(expr, Expr::FloatLit(12.25, _)));
    let tokens = Lexer::new("0..10").tokenize();
    assert!(matches!(tokens[0].kind, TokenKind::IntLit(0)));
    assert!(matches!(tokens[1].kind, TokenKind::DotDot));
    assert!(matches!(tokens[2].kind, TokenKind::IntLit(10)));
}

#[test]
fn the_cli_refuses_a_u64_literal_it_cannot_lower_without_writing_ir() {
    let dir = std::env::temp_dir().join(format!("y_literal_diagnostics_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("overflow.ysu");
    let ir = dir.join("overflow.ll");
    if ir.exists() {
        std::fs::remove_file(&ir).unwrap();
    }
    std::fs::write(&source, "fn main() -> U64 { return 18446744073709551615; }").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&source)
        .arg("--emit-llvm")
        .arg("-o")
        .arg(&ir)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run compiler");
    let diagnostic = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.status.success(), "{diagnostic}");
    assert!(
        diagnostic.contains("Invalid integer literal"),
        "{diagnostic}"
    );
    assert!(diagnostic.contains("18446744073709551615"), "{diagnostic}");
    assert!(!ir.exists(), "invalid input produced LLVM IR");
    std::fs::remove_dir_all(dir).unwrap();
}
