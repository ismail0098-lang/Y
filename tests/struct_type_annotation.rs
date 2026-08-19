//! A `let` annotated with a declared struct type.
//!
//! `resolve_type` turned `Type::Ident(name)` into `SemanticType::Unknown` for
//! anything that was not a type alias — it consulted the variable table and
//! never the struct table, which sat beside it and was read by
//! `Expr::StructLit` alone. `types_are_compatible` treats `Unknown` as
//! compatible with nothing, so:
//!
//! ```text
//! struct P { x: I32, y: I32 }
//! let p: P = P { x: 4, y: 3 };     // "Type mismatch in let assignment."
//! let p  = P { x: 4, y: 3 };       // fine
//! ```
//!
//! The language rejected the more explicit of two spellings of one program.
//! It surfaced from the self-hosted compiler, which annotates ~340 `let`s with
//! struct types (`Token`, `Expr`, `Stmt`, `FuncDecl`, ...).
//!
//! Every case RUNS the produced binary: an annotation that type-checks but
//! lowers to the wrong thing would pass a compile-only test.
//!
//! Run with:  cargo test --release --test struct_type_annotation

use std::path::PathBuf;
use std::process::Command;

/// `Ok(exit code)`, or `Err(diagnostic)` if the compiler refused.
fn build_and_run(name: &str, src: &str) -> Result<i32, String> {
    let dir = std::env::temp_dir().join(format!("y_sta_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let bin = dir.join(name);
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("-o")
        .arg(&bin)
        .current_dir(&repo)
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if text.contains("semantic errors") || text.contains("Syntax Error") {
        return Err(text);
    }
    if !bin.exists() {
        return Err(format!("no binary (clang missing?):\n{}", text));
    }
    Command::new(&bin)
        .status()
        .ok()
        .and_then(|s| s.code())
        .ok_or_else(|| "the built binary did not run".to_string())
}

const DECL: &str = "struct P { x: I32, y: I32 }\n";

#[test]
fn an_annotated_struct_binding_compiles_and_runs() {
    let src = format!(
        "{}fn main() -> I32 {{\n    let p: P = P {{ x: 4, y: 3 }};\n    return p.x + p.y;\n}}\n",
        DECL
    );
    match build_and_run("sta_annotated", &src) {
        Ok(code) => assert_eq!(code, 7, "the annotated binding lowered to the wrong value"),
        Err(e) if e.starts_with("no binary") => eprintln!("SKIP: clang missing"),
        Err(e) => panic!("`let p: P = P {{ .. }}` must compile:\n{}", e),
    }
}

/// The same program without the annotation always worked. If this ever
/// disagrees with the case above, the two spellings have diverged again.
#[test]
fn the_unannotated_spelling_agrees() {
    let src = format!(
        "{}fn main() -> I32 {{\n    let p = P {{ x: 4, y: 3 }};\n    return p.x + p.y;\n}}\n",
        DECL
    );
    match build_and_run("sta_inferred", &src) {
        Ok(code) => assert_eq!(code, 7, "the inferred binding lowered to the wrong value"),
        Err(e) if e.starts_with("no binary") => eprintln!("SKIP: clang missing"),
        Err(e) => panic!("the unannotated spelling regressed:\n{}", e),
    }
}

/// A typo'd type name must still be a type error.
///
/// Mutation-verified as a CONFIRMATION, not as a guard on the
/// `structs.contains_key` restriction: resolving EVERY unknown identifier type
/// to itself leaves this green, because `Q` and `P` are then two different
/// `Primitive`s and `types_are_compatible` rejects the pair anyway. The
/// restriction is kept because `Unknown` is the honest answer for a name the
/// compiler has never seen, not because this test pins it. Do not read it as
/// doing so.
#[test]
fn an_undeclared_type_name_is_still_refused() {
    let src = format!(
        "{}fn main() -> I32 {{\n    let p: Q = P {{ x: 4, y: 3 }};\n    return p.x;\n}}\n",
        DECL
    );
    match build_and_run("sta_typo", &src) {
        Ok(code) => panic!(
            "`let p: Q = P {{ .. }}` compiled (exit {}). `Q` is not a declared \
             type, so this must stay a type error.",
            code
        ),
        Err(e) => assert!(
            e.contains("Type mismatch") || e.contains("semantic errors"),
            "refused, but not as a type error: {}",
            e
        ),
    }
}
