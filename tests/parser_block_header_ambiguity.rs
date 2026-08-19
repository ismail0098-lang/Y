//! `{` after a block header is the block, not a struct literal — at EVERY site.
//!
//! `n { }` is a well-formed empty struct literal, so an expression parser that
//! is willing to build one will swallow the block that follows it. Y fixes this
//! the way Rust does, with a `no_struct_literal` flag set while parsing a block
//! header. The flag was set at three sites — `if`, `while` and a `match`
//! scrutinee — and a `for` loop's bounds are a fourth:
//!
//! ```text
//! for i in 0..n { }
//! //        ^^^^^^ parsed as the struct literal `n {}`, leaving the `for`
//! //               with no block
//! ```
//!
//! The diagnostic is what made it survive in both cases. It points at the token
//! AFTER the loop, so it reads as a mistake on the line below rather than as an
//! ambiguity in the line you wrote.
//!
//! Only the EMPTY body is reachable: the heuristic's other arm needs `ident :`
//! after the brace and no Y statement begins that way. That is also why neither
//! fuzzer found this one — the generator emits `for i0 in 0..1 { ... }`, with a
//! literal bound and a non-empty body, so it cannot write the program. A
//! generator restriction is a deleted bug class; see `CLAUDE.md` gotcha #11.
//!
//! This file is deliberately NOT feature-gated. The same cases live in
//! `tests/zk_gadget_range_and_parsing.rs`, where the `if` half was originally
//! found, but nothing about either bug is ZK and a `--features zk` gate means
//! the default `cargo test` does not check the parser at all.
//!
//! Run with:  cargo test --test parser_block_header_ambiguity

fn parses(src: &str) -> Result<(), String> {
    let tokens = y::lexer::Lexer::new(src).tokenize();
    y::parser::Parser::new(tokens).parse_program().map(|_| ())
}

/// Every position where a block follows an expression, with an empty body.
#[test]
fn an_empty_block_after_an_identifier_header_parses() {
    let cases = [
        ("if", "fn main(p: I32) -> I32 {\n    if p {\n    }\n    return 1;\n}\n"),
        (
            "if/else",
            "fn main(p: I32) -> I32 {\n    if p {\n    } else {\n    }\n    return 1;\n}\n",
        ),
        ("while", "fn main(p: I32) -> I32 {\n    while p {\n    }\n    return 1;\n}\n"),
        (
            "for end bound",
            "fn main(p: I32) -> I32 {\n    for i in 0..p {\n    }\n    return 1;\n}\n",
        ),
        // A CONFIRMATION, not a guard: `start` is always followed by `..`, so
        // it can never see a `{` and reverting its arm alone leaves this file
        // green. It is guarded anyway, because "the token after it is always
        // `..`" is a property of the grammar today and not of this function.
        (
            "for start bound",
            "fn main(p: I32) -> I32 {\n    for i in p..9 {\n    }\n    return 1;\n}\n",
        ),
        (
            "for step",
            "fn main(p: I32) -> I32 {\n    for i in 0..9 step p {\n    }\n    return 1;\n}\n",
        ),
        (
            "nested for",
            "fn main(p: I32) -> I32 {\n    for i in 0..p {\n        for j in 0..p {\n        }\n    }\n    return 1;\n}\n",
        ),
    ];
    for (name, src) in cases {
        assert!(
            parses(src).is_ok(),
            "`{}` with an empty body failed to parse. The header expression \
             swallowed the block as a struct literal, and the error points at \
             the statement after it: {:?}",
            name,
            parses(src).unwrap_err()
        );
    }
}

/// The control, in both directions.
///
/// Suppressing struct literals everywhere would pass every case above and
/// break the language, so a literal must still parse where it is unambiguous —
/// in a `let`, and inside the loop BODY, which is what proves the flag is
/// RESTORED after the header rather than left set for the rest of the parse.
#[test]
fn struct_literals_still_parse_where_they_are_unambiguous() {
    let src = "struct P { x: I32 }\n\
               fn main(p: I32) -> I32 {\n\
               \x20   let q: P = P { x: 3 };\n\
               \x20   for i in 0..p {\n\
               \x20       let r: P = P { x: 4 };\n\
               \x20   }\n\
               \x20   if p {\n\
               \x20       let s: P = P { x: 5 };\n\
               \x20   }\n\
               \x20   return 0;\n\
               }\n";
    assert!(
        parses(src).is_ok(),
        "a struct literal stopped parsing where it is unambiguous: {:?}",
        parses(src).unwrap_err()
    );
}

// ── Reserved words as variable names ───────────────────────────────────
//
// Not the same bug as the ones above, but the same cost: a message that
// points at the construct instead of at the word. `step` is reserved by the
// `for i in a..b step N` syntax, so `let step: I64 = ...;` reported only
// "Expected identifier after let". Y's own self-hosted type checker used
// `step` as a variable and had been unparseable ever since.

/// The refusal is correct; the diagnostic must name the word.
#[test]
fn a_reserved_word_used_as_a_variable_names_itself() {
    let err = {
        let src = "fn main() -> I32 {\n    let step: I32 = 3;\n    return step;\n}\n";
        let tokens = y::lexer::Lexer::new(src).tokenize();
        y::parser::Parser::new(tokens)
            .parse_program()
            .expect_err("`step` is reserved, so this must not parse")
    };
    assert!(
        err.contains("step"),
        "the error must name the offending word, or a user has to guess which \
         of the line's tokens is reserved. Got: {:?}",
        err
    );
    assert!(
        err.contains("reserved"),
        "the error must say WHY the name is rejected. Got: {:?}",
        err
    );
}

/// The control: an ordinary name must still work, and the message above must
/// not be produced for a genuine syntax error elsewhere.
#[test]
fn an_ordinary_name_still_parses() {
    let src = "fn main() -> I32 {\n    let stride: I32 = 3;\n    return stride;\n}\n";
    let tokens = y::lexer::Lexer::new(src).tokenize();
    assert!(
        y::parser::Parser::new(tokens).parse_program().is_ok(),
        "`stride` is not reserved and must parse"
    );
}
