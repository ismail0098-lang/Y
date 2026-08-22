// ============================================================
//  `Type::Reference` resolved to `SemanticType::Unknown`, and
//  EVERY unary expression did too. Both had to be fixed together.
//
//  `Unknown` is not a neutral answer in this type checker: the `let`
//  arm reads it as "adopt the annotation" and the assignment arm
//  exempts it from the mismatch check outright. So a reference-typed
//  binding was unchecked on both sides --
//
//      let x: I32 = 1;
//      let r: &F32 = &x;      // compiled clean
//
//  -- and fixing only the ANNOTATION would have bought nothing, because
//  the initialiser `&x` was `Unknown` too and `Unknown` on either side
//  suppresses the check. That is the "guards consulted at one site"
//  shape from CLAUDE.md, in the two halves of a single comparison.
//
//  Every negative case is paired with the control that would pass if
//  the fix were "refuse all references": refusing everything is sound
//  and useless.
// ============================================================

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static N: AtomicUsize = AtomicUsize::new(0);

fn bin() -> std::path::PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("Y")
}

fn body_is_refused(body: &str) -> bool {
    let dir = std::env::temp_dir().join(format!(
        "y_refty_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("t.ysu");
    std::fs::write(&f, format!("fn main() -> I32 {{\n{body}\n    return 0;\n}}\n")).unwrap();
    let out = Command::new(bin())
        .arg(&f)
        .arg("--emit-cpu")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let _ = std::fs::remove_dir_all(&dir);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !text.contains("Syntax Error"),
        "fixture did not parse, so it proves nothing about the type checker:\n{text}"
    );
    text.contains("Type mismatch") || text.contains("types do not match")
}

#[test]
fn a_reference_to_the_wrong_element_type_is_refused() {
    assert!(
        body_is_refused("    let x: I32 = 1;\n    let r: &F32 = &x;"),
        "`&F32 = &x` with `x: I32` was accepted"
    );
}

#[test]
fn a_reference_to_the_right_element_type_still_compiles() {
    assert!(
        !body_is_refused("    let x: I32 = 1;\n    let r: &I32 = &x;"),
        "`&I32 = &x` with `x: I32` was refused"
    );
}

#[test]
fn a_value_where_a_reference_is_declared_is_refused() {
    // The distinction that did not exist at all before: `Unknown` could not
    // tell a reference from the thing it points at.
    assert!(
        body_is_refused("    let x: I32 = 1;\n    let r: &I32 = x;"),
        "a bare value bound to a `&I32` was accepted"
    );
}

#[test]
fn a_dereference_yields_what_the_reference_points_at() {
    let decl = "    let x: I32 = 1;\n    let r: &I32 = &x;\n";
    assert!(
        !body_is_refused(&format!("{decl}    let v: I32 = *r;")),
        "`*r` of a `&I32` did not type as `I32`"
    );
    assert!(
        body_is_refused(&format!("{decl}    let v: F32 = *r;")),
        "`*r` of a `&I32` was accepted as `F32`"
    );
}

#[test]
fn negation_keeps_its_operand_type() {
    // `-x` was `Unknown`, so ANY annotation was adopted without a check. This
    // is not about references at all, and is the half of the fix that pays off
    // in ordinary arithmetic code.
    assert!(
        body_is_refused("    let x: F32 = 1.0;\n    let y: I32 = -x;"),
        "`-x` of an F32 was accepted as I32"
    );
    assert!(
        !body_is_refused("    let x: F32 = 1.0;\n    let y: F32 = -x;"),
        "`-x` of an F32 was refused as F32"
    );
}
