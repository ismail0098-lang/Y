//! A saved reference has the same mutation effects as a reference literal.
use std::process::Command;
use y::type_checker::{z3_candidates, TypeChecker};

#[path = "common/verification.rs"]
mod verification;

fn have_solver() -> bool {
    let available = z3_candidates().iter().any(|candidate| {
        Command::new(candidate)
            .arg("-version")
            .output()
            .is_ok_and(|out| out.status.success())
    });
    verification::prerequisite_available(available, "executable Z3 for reference-alias invariant checks")
}

fn errors(source: &str) -> Vec<String> {
    assert_ne!(
        std::env::var("Y_ALLOW_UNVERIFIED_INVARIANTS").as_deref(),
        Ok("1")
    );
    let program = y::parser::Parser::new(y::lexer::Lexer::new(source).tokenize())
        .parse_program()
        .expect("reference regression parses");
    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    assert!(
        checker
            .errors
            .iter()
            .all(|e| !e.contains("SMT solver could not be run")),
        "a discovered solver must actually run: {:?}",
        checker.errors
    );
    checker.errors
}

const MUTATOR: &str = "@unsafe fn bump(p: &mut I32) -> I32 { *p = -1; return 0; }\n";

#[test]
fn saved_references_cannot_hide_mutation_in_loop_bodies() {
    if !have_solver() {
        return;
    }
    for body in [
        "bump(p);",
        "if i >= 0 { bump(p); }",
        "let result: I32 = bump(p);",
        "ignored = bump(p);",
    ] {
        for loop_source in [
            format!("for i in 0..1 {{ {body} }}"),
            format!("while i < 1 {{ {body} i = i + 1; }}"),
        ] {
            let source = format!(
                "{MUTATOR}
fn main() {{
    let x: I32 = 0;
    let i: I32 = 0;
    let ignored: I32 = 0;
    let p: &mut I32 = &mut x;
    @safe {{ @invariant(x >= 0) {loop_source} }}
}}"
            );
            let found = errors(&source);
            assert!(
                found
                    .iter()
                    .any(|e| e.contains("passes a reference to a call")),
                "a saved reference can invalidate x >= 0: {found:?}\n{source}"
            );
        }
    }
}

#[test]
fn calls_before_loop_entry_invalidate_aliased_range_facts() {
    if !have_solver() {
        return;
    }
    for argument in ["p", "&mut x"] {
        for loop_source in ["for i in 0..1 { }", "while x >= 0 { }"] {
            let source = format!(
                "{MUTATOR}
fn main() {{
    let x: I32 = 0;
    let p: &mut I32 = &mut x;
    bump({argument});
    @safe {{ @invariant(x >= 0) {loop_source} }}
}}"
            );
            let found = errors(&source);
            assert!(
                found.iter().any(|e| e.contains("initiation check failed")),
                "a call can invalidate the initializer before loop entry: {found:?}\n{source}"
            );
        }
    }
}

#[test]
fn references_carried_by_structs_cannot_hide_aliases() {
    if !have_solver() {
        return;
    }
    let source = r#"
struct Link { p: &mut I32 }
@unsafe fn bump_link(link: Link) { *link.p = -1; }
fn main() {
    let x: I32 = 0;
    let link: Link = Link { p: &mut x };
    @safe {
        @invariant(x >= 0)
        for i in 0..1 { bump_link(link); }
    }
}

"#;
    let found = errors(source);
    assert!(
        found
            .iter()
            .any(|e| e.contains("passes a reference to a call")),
        "an aggregate can carry a reference to x: {found:?}"
    );
}

#[test]
fn scalar_calls_and_reestablished_ranges_still_verify() {
    if !have_solver() {
        return;
    }
    for source in [
        "fn observe(x: I32) -> I32 { return x; }
fn main() {
    let x: I32 = 0;
    let p: &mut I32 = &mut x;
    @safe { @invariant(x == 0) for i in 0..1 { observe(x); } }
}",
        "@unsafe fn bump(p: &mut I32) { *p = -1; }
fn main() {
    let x: I32 = 0;
    let p: &mut I32 = &mut x;
    bump(p);
    x = 0;
    @safe { @invariant(x == 0) for i in 0..1 { } }
}",
    ] {
        let found = errors(source);
        assert!(
            found.is_empty(),
            "valid scalar invariant refused: {found:?}\n{source}"
        );
    }
}

#[test]
fn writes_through_saved_references_invalidate_loop_entry_facts() {
    if !have_solver() {
        return;
    }
    for store in ["*p = -1;", "*p -= 1;"] {
        let source = format!(
            "@unsafe fn main() {{
    let x: I32 = 0;
    let p: &mut I32 = &mut x;
    {store}
    @safe {{ @invariant(x >= 0) for i in 0..1 {{ }} }}
}}"
        );
        let found = errors(&source);
        assert!(
            found.iter().any(|e| e.contains("initiation check failed")),
            "an indirect store can invalidate the entry range: {found:?}\n{source}"
        );
    }
}

#[test]
fn references_carried_by_enums_are_tracked_before_and_inside_loops() {
    if !have_solver() {
        return;
    }
    for mutation in [
        "bump_link(link); @safe { @invariant(x >= 0) for i in 0..1 {} }",
        "@safe { @invariant(x >= 0) for i in 0..1 { bump_link(link); } }",
    ] {
        let source = format!(
            "{MUTATOR}
enum Link {{ Ref(&mut I32) }}
@unsafe fn bump_link(link: Link) {{ match link {{ Link::Ref(p) => bump(p) }} }}
fn main() {{
    let x: I32 = 0;
    let link: Link = Link::Ref(&mut x);
    x = 0;
    {mutation}
}}"
        );
        let found = errors(&source);
        assert!(
            found.iter().any(|e| e.contains("initiation check failed")
                || e.contains("passes a reference to a call")),
            "an enum payload can carry a reference to x: {found:?}\n{source}"
        );
    }
}

#[test]
fn dotted_calls_invalidate_saved_reference_entry_facts() {
    if !have_solver() {
        return;
    }
    let source = "@unsafe fn ns_bump(p: &mut I32) { *p = -1; }
fn main() {
    let ns: I32 = 0;
    let x: I32 = 0;
    let p: &mut I32 = &mut x;
    ns.bump(p);
    @safe { @invariant(x >= 0) for i in 0..1 {} }
}";
    let found = errors(source);
    assert!(
        found.iter().any(|e| e.contains("initiation check failed")),
        "a dotted call can write through a saved reference: {found:?}"
    );
}
