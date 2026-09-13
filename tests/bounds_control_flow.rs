//! Interval proofs must describe every path reaching an array access.
use y::{lexer::Lexer, parser::Parser, type_checker::TypeChecker};

fn check(body: &str) -> Vec<String> {
    let source = format!("@safe fn f(flag: bool, n: I32) {{ let arr: [I32; 5] = {{}}; {body} }}");
    let ast = Parser::new(Lexer::new(&source).tokenize())
        .parse_program()
        .unwrap();
    let mut tc = TypeChecker::new();
    tc.check_program(&ast);
    tc.errors
}

fn rejected(body: &str) {
    let errors = check(body);
    assert!(
        errors
            .iter()
            .any(|e| e.contains("bounds") || e.contains("Bounds")),
        "accepted invalid bounds: {body}\n{errors:?}"
    );
}

#[test]
fn each_branch_starts_from_the_entry_facts_and_exits_are_joined() {
    rejected("let i: I32 = 100; if false { i = 0; } let x = arr[i];");
    rejected("let i: I32 = 0; if flag { i = 100; } else { i = 1; } let x = arr[i];");
    rejected("let i: I32 = 100; if flag { i = 0; } else { let x = arr[i]; }");
    rejected("let i: I32 = 0; if flag { i = n; } else { i = 1; } let x = arr[i];");
    assert!(
        check("let i: I32 = 100; if flag { i = 0; } else { i = 4; } let x = arr[i];").is_empty()
    );
    assert!(check("let i: I32 = 2; if flag { let i: I32 = 100; } let x = arr[i];").is_empty());
}

#[test]
fn compound_assignment_updates_or_invalidates_the_range() {
    rejected("let i: I32 = 0; i += 100; let x = arr[i];");
    rejected("let i: I32 = 0; i -= 1; let x = arr[i];");
    rejected("let i: I32 = 2; i *= 3; let x = arr[i];");
    rejected("let i: I32 = 0; i += n; let x = arr[i];");
    rejected("let i: I64 = 9223372036854775807; i += 1; i -= 9223372036854775807; let x = arr[i];");
    rejected("let i: I32 = 2147483647; i += 1; i /= 2; i -= 1073741824; let x = arr[i];");
    rejected("@bounds(min=0, max=4) let i: I32 = 0; i += 100; let x = arr[i];");
    assert!(check("let i: I32 = 0; i += 4; let x = arr[i];").is_empty());
}

#[test]
fn annotated_integer_width_controls_the_stored_range() {
    // Narrowing changes the stored value, even when its initializer is an
    // identifier whose mathematical range was known exactly.
    rejected("let wide: I64 = 4294967296; let narrow: I32 = wide; let index: I64 = narrow; index -= 4294967296; let x = arr[index];");
    // Widening before arithmetic preserves values beyond the source width.
    assert!(check("let initial: I32 = 2147483647; let wide: I64 = initial; wide += 1; wide -= 2147483648; let x = arr[wide];").is_empty());
}

#[test]
fn match_arms_do_not_overwrite_each_others_facts() {
    use y::ast::{Expr, Item, Stmt};
    // Block-valued match arms exist in the AST even though the source parser
    // currently accepts only expression arms. Exercise the analysis directly.
    let source = "@safe fn f(n: I32) { let arr: [I32; 5] = {}; let i: I32 = 100; match n { 0 => 0, _ => 0 } } fn first() { i = 0; } fn second() { let x = arr[i]; }";
    let mut ast = Parser::new(Lexer::new(source).tokenize())
        .parse_program()
        .unwrap();
    let bodies: Vec<_> = ast
        .items
        .drain(1..)
        .map(|item| match item {
            Item::Func(f) => f.body,
            _ => unreachable!(),
        })
        .collect();
    let Item::Func(f) = &mut ast.items[0] else {
        unreachable!()
    };
    let Stmt::Match { arms, .. } = &mut f.body.stmts[2] else {
        unreachable!()
    };
    for (arm, block) in arms.iter_mut().zip(bodies) {
        arm.body = Expr::BlockExpr(block, arm.span.clone());
    }
    let mut tc = TypeChecker::new();
    tc.check_program(&ast);
    assert!(
        tc.errors.iter().any(|e| e.contains("bounds")),
        "{:?}",
        tc.errors
    );
}
