//! Every `self_hosted/*.ysu` must at least PARSE.
//!
//! Nothing in this repo built `self_hosted/` — no test, no script, no Makefile
//! target; it is named only in a documentation file. So it had rotted without
//! anyone finding out: **6 of its 10 files did not parse**, and the two causes
//! were both mechanical.
//!
//!  1. `parser.ysu` had a `while parsing_body {` header deleted (the identical
//!     loop 40 lines above is intact, same variable names) and a 98-line
//!     duplicate of `parse_stmt`'s tail left dangling after the method's
//!     closing brace. Both errors were reported far from their cause: the
//!     stray `}` closed the method, so the message was "Expected 'fn' in impl
//!     block" pointing 20 lines further down. Five files import `parser`, so
//!     one deletion accounted for five of the six failures.
//!  2. `type_checker.ysu` used `step` as a variable name, and `step` is a
//!     reserved word — the `for i in a..b step N` syntax. Same shape as the
//!     generated BN254 temporaries colliding with `U16` recorded in CLAUDE.md,
//!     and found the same way: by a name that reads as perfectly ordinary.
//!     The diagnostic said only "Expected identifier after let"; it names the
//!     word now.
//!
//! This gate asserts PARSING only, not a clean compile. The files still report
//! `[Strict Safety] Variables in safe blocks must be explicitly initialized`
//! for `lib.ysu`'s stub accessors (`fn get_Token(..) -> Token { let d: Token;
//! return d; }`), which is the compiler correctly refusing a genuinely
//! uninitialized read. Implementing those stubs is self-hosted compiler work,
//! not a compiler bug, and pinning "compiles cleanly" would pin the stubs.
//!
//! Run with:  cargo test --release --test self_hosted_parses

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("Y")
}

#[test]
fn every_self_hosted_source_parses() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = repo.join("self_hosted");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("self_hosted/ is missing")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "ysu"))
        .collect();
    sources.sort();

    // A directory that has quietly become empty would otherwise pass.
    assert!(
        sources.len() >= 10,
        "expected the self-hosted compiler's sources, found {} file(s)",
        sources.len()
    );

    let mut broken = Vec::new();
    for src in &sources {
        let out = Command::new(bin())
            .arg(src)
            .arg("-o")
            .arg(std::env::temp_dir().join("y_selfhosted_probe"))
            .current_dir(repo)
            .env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS")
            .output()
            .expect("failed to run the Y binary");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if text.contains("Syntax Error") {
            let detail = text
                .lines()
                .find(|l| l.contains("Line "))
                .unwrap_or("<no location>")
                .trim()
                .to_string();
            broken.push(format!("{}: {}", src.file_name().unwrap().to_string_lossy(), detail));
        }
    }
    assert!(
        broken.is_empty(),
        "{} of {} self-hosted sources do not parse:\n  {}",
        broken.len(),
        sources.len(),
        broken.join("\n  ")
    );
}
