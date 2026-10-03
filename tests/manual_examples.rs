//! Every complete example in chapter 10 of the language manual either
//! compiles verbatim or says, at its head, that it does not.
//!
//! Measured 2026-09-28 with the compiler at `767779a`: of the 21 examples in
//! `docs/y_language_documentation.md` chapter 10, three compiled and eighteen
//! did not - and ONE of the eighteen said so. The rest failed on syntax the
//! parser has never accepted (`@clock_domain`, `@ghost` and `@cache_policy` on
//! struct fields, generic functions, `TF32`, an `@unsafe { }` block, an `as`
//! cast, `@bounds(0 <= i < 100)`, `@require` on a `fn`) or on the `@safe`
//! checks, and each was presented as a working program. A manual whose worked
//! examples do not compile is the same defect as a doc describing an absent
//! optimisation, and it was invisible because nothing compiled them.
//!
//! The check is a biconditional, and both halves are needed:
//!
//! * an example that fails and carries no status note is an undocumented
//!   broken example - the state the chapter was in;
//! * an example that carries a "does not compile" note and compiles is a note
//!   that went stale when the compiler grew the feature - which would make the
//!   manual understate the language, and makes the notes a record nobody has
//!   to maintain by hand.
//!
//! An example is compiled with `--emit-ptx` if it declares a `kernel` and with
//! `--emit-llvm` otherwise, from a COPY in a temp directory: both backends
//! write next to the source.
//!
//! **What this does not cover.** Only chapter 10: its examples are complete
//! programs. The fragments elsewhere in the manual (the §9 directive examples,
//! §20's type examples) are not, and compiling them needs context a scraper
//! would have to invent. §20.4's cast example and §9.10's `@bounds` syntax were
//! wrong in exactly that unchecked territory, found by hand the same day.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "common/pinned.rs"]
mod pinned;

const NOTE: &str = "**Status: this example does not compile**";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A per-call directory: the tag makes a leftover legible, the counter makes
/// the path unique when two tests pass the same tag.
fn scratch(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "y_manual_examples_{tag}_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

struct Example {
    number: u32,
    code: String,
    noted: bool,
}

fn examples() -> Vec<Example> {
    let src = std::fs::read_to_string(repo().join("docs/y_language_documentation.md"))
        .expect("read the manual");
    let start = src
        .find("\n## 10. Complete Code Examples")
        .expect("chapter 10 heading");
    let end = src[start + 1..]
        .find("\n## ")
        .map(|e| start + 1 + e)
        .expect("the heading after chapter 10");
    let chapter = &src[start..end];
    let mut out = Vec::new();
    for sec in chapter.split("\n### Example ").skip(1) {
        let number: u32 = sec
            .split(':')
            .next()
            .and_then(|n| n.trim().parse().ok())
            .unwrap_or_else(|| panic!("unnumbered example: {}", &sec[..sec.len().min(60)]));
        let blocks: Vec<&str> = sec.split("```ysu\n").skip(1).collect();
        assert_eq!(
            blocks.len(),
            1,
            "Example {number} has {} ysu blocks; this gate compiles exactly one per example",
            blocks.len()
        );
        let code = blocks[0].split("\n```").next().expect("closing fence").to_string();
        // The note must sit at the head of the example, before its code - a
        // note further down is not the one a reader sees first.
        let head = sec.split("```ysu\n").next().unwrap_or("");
        out.push(Example {
            number,
            code: code + "\n",
            noted: head.contains(NOTE),
        });
    }
    out
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\u{1b}' {
            for d in it.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Compile `code` the way chapter 10 says its examples are compiled. `Err`
/// carries the first diagnostic.
fn compile(code: &str, name: &str, dir: &Path) -> Result<(), String> {
    let flag = if code.lines().any(|l| l.trim_start().starts_with("kernel ")) {
        "--emit-ptx"
    } else {
        "--emit-llvm"
    };
    let file = dir.join(format!("{name}.ysu"));
    std::fs::write(&file, code).expect("write example");
    // A PINNED profile in the scratch directory, which is the working
    // directory: `current_dir(repo)` compiled for this machine's card.
    pinned::pin(dir, pinned::SM_PINNED);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&file)
        .arg(flag)
        .current_dir(dir)
        .output()
        .expect("run Y");
    if out.status.success() {
        return Ok(());
    }
    let text = strip_ansi(&format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    ));
    Err(first_diagnostic(&text))
}

/// The first `Line <n>:` diagnostic, else the first line mentioning an error.
/// `Line ` alone is not enough: the compiler's banner prints `L2 Cache Line
/// Size`, and taking that as the diagnostic is how the first version of the
/// control below failed.
fn first_diagnostic(text: &str) -> String {
    let numbered = text.lines().find(|l| {
        l.find("Line ")
            .map(|i| l[i + 5..].starts_with(|c: char| c.is_ascii_digit()))
            .unwrap_or(false)
    });
    numbered
        .or_else(|| text.lines().find(|l| l.to_ascii_lowercase().contains("error")))
        .unwrap_or("<no diagnostic>")
        .trim()
        .to_string()
}

#[test]
fn every_example_compiles_or_says_it_does_not() {
    let exs = examples();
    assert!(
        exs.len() >= 20,
        "found {} examples in chapter 10 - the scrape has drifted and this gate is checking nothing",
        exs.len()
    );
    let dir = scratch("all");
    let mut problems = Vec::new();
    for ex in &exs {
        let name = format!("example_{}", ex.number);
        match (compile(&ex.code, &name, &dir), ex.noted) {
            (Ok(()), false) | (Err(_), true) => {}
            (Ok(()), true) => problems.push(format!(
                "Example {} says it does not compile, and it compiles verbatim - remove the note",
                ex.number
            )),
            (Err(why), false) => problems.push(format!(
                "Example {} does not compile ({why}) and says nothing - fix it or add a status note",
                ex.number
            )),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(problems.is_empty(), "chapter 10:\n  {}", problems.join("\n  "));
}

/// The biconditional above is only as good as `compile`'s ability to tell a
/// program that compiles from one that does not. A helper that always answered
/// `Ok` would fail it only while some example is noted, and one that always
/// answered `Err` only while some example is not - so it is checked here, on
/// inputs whose answers are known, rather than through the manual.
#[test]
fn the_compile_helper_can_tell_the_two_apart() {
    let dir = scratch("control");
    let good = "fn main() -> I32 {\n    let x: I32 = 1;\n    return x;\n}\n";
    // The `@bounds` form §9.10 used to document.
    let bad = "fn main() -> I32 {\n    @bounds(0 <= i < 100)\n    let i: I32 = 1;\n    return i;\n}\n";
    assert_eq!(compile(good, "good", &dir), Ok(()));
    let why = compile(bad, "bad", &dir).expect_err("the old @bounds form must not compile");
    assert!(why.contains("@bounds"), "unexpected diagnostic: {why}");
    let _ = std::fs::remove_dir_all(&dir);
}
