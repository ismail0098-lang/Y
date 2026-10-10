//! The suite's verdict must not depend on the machine it runs on, or on what
//! was built before it ran.
//!
//! Until 2026-10-01 the suite had only ever run on the developer's own machine
//! - an sm_89 card, a probed `.ysu_hw_profile` in the repo root, a CUDA
//! toolkit, LLVM 21+, and a release build left over from the last session.
//! The first run anywhere else (a fresh clone in a GPU-less container) failed
//! 25 tests, and five of the causes were properties of the SUITE rather than
//! of the compiler. This file holds the source-level gates for them; the
//! behaviour is fixed in the files they name.

use std::path::{Path, PathBuf};

#[path = "common/pinned.rs"]
mod pinned;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.rs` file a test binary is built from: `tests/`, `tests/common/`,
/// and the `y-gpu` crate's tests.
fn test_sources() -> Vec<PathBuf> {
    let mut v = Vec::new();
    for dir in ["tests", "tests/common", "crates/y-gpu/tests"] {
        let Ok(rd) = std::fs::read_dir(repo().join(dir)) else { continue };
        for e in rd {
            let p = e.expect("entry").path();
            if p.extension().and_then(|x| x.to_str()) == Some("rs") {
                v.push(p);
            }
        }
    }
    v.sort();
    v
}

/// The code part of a line: everything before a `//` comment.
fn code(line: &str) -> &str {
    line.split("//").next().unwrap_or("")
}

fn rel(p: &Path) -> String {
    p.strip_prefix(repo()).unwrap_or(p).display().to_string()
}

/// **Eight test files ran `target/release/Y`, a binary `cargo test` never
/// builds.** So after a source edit, the documented `cargo test` checked those
/// eight against whatever release build happened to exist - or failed with
/// "build the compiler first" in a clean checkout. Demonstrated rather than
/// argued: with `chisel {}`'s original register-naming defect restored in
/// `resolve_chisel_registers`, `chisel_register_scope` passed 8/8 against the
/// release binary built before the edit, and a mutation table for the `sm_00`
/// fix found that file's column green in every row - including the over-fix
/// that four other suites caught. `struct_field_array_bounds` was worse still:
/// it PREFERRED a stale release binary over the fresh debug one.
///
/// `env!("CARGO_BIN_EXE_Y")` is the binary `cargo test` builds from the source
/// under test, under any profile and any `CARGO_TARGET_DIR`. A hardcoded
/// `target/debug/Y` is refused too: it is wrong under `cargo test --release`
/// and under a relocated target directory.
#[test]
fn no_test_runs_a_compiler_binary_cargo_test_does_not_build() {
    let me = file!().rsplit('/').next().unwrap_or("").to_string();
    let mut offenders = Vec::new();
    let mut scanned = 0;
    for p in test_sources() {
        if p.file_name().and_then(|n| n.to_str()) == Some(me.as_str()) {
            continue;
        }
        scanned += 1;
        let text = std::fs::read_to_string(&p).expect("read test source");
        for (i, line) in text.lines().enumerate() {
            let c = code(line);
            for bad in [
                "join(\"target/release/Y\")",
                "join(\"target/debug/Y\")",
                "Command::new(\"target/release/Y\")",
                "Command::new(\"./target/release/Y\")",
                "Command::new(\"target/debug/Y\")",
            ] {
                if c.contains(bad) {
                    offenders.push(format!("{}:{}: {}", rel(&p), i + 1, line.trim()));
                }
            }
        }
    }
    assert!(scanned > 100, "only {scanned} test sources scanned; the walk is not reading tests/");
    assert!(
        offenders.is_empty(),
        "these tests run a compiler binary `cargo test` does not build from the source \
         under test - use `env!(\"CARGO_BIN_EXE_Y\")`:\n  {}",
        offenders.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// A tiny Rust reader: enough to find function extents and string literals in
// the test sources without being fooled by braces inside the Y programs those
// tests embed (in raw strings) or inside comments.

/// One `fn` in a test source.
#[derive(Debug)]
struct Func {
    file: PathBuf,
    name: String,
    line: usize,
    /// The function's text, comments removed, string contents kept.
    body: String,
}

/// Blank comments and (optionally) string contents, preserving every newline
/// so line numbers survive AND every byte offset, so a position found in one
/// mask indexes the same character in the other and in the source. Handles
/// `//`, nested `/* */`, `"..."` with escapes, raw strings `r#"..."#`, and char
/// literals (told apart from lifetimes).
///
/// A blanked character becomes as many spaces as it has UTF-8 bytes. It used to
/// become ONE, and `functions` slices the string-keeping mask with offsets found
/// in the string-blanking one - so every multi-byte character inside a string
/// literal (an em-dash in an assert message) shifted every later function's
/// extent. Measured when found: 178 of the 1,603 function bodies this reader
/// returned started or ended in the wrong place, across 33 files, so every gate
/// built on it was reading truncated or neighbouring code.
/// `the_reader_returns_exact_function_extents` pins it.
fn mask(src: &str, blank_strings: bool) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    fn blank(out: &mut String, c: char) {
        if c == '\n' {
            out.push('\n');
        } else {
            for _ in 0..c.len_utf8() {
                out.push(' ');
            }
        }
    }
    while i < b.len() {
        let c = b[i];
        if c == '/' && i + 1 < b.len() && b[i + 1] == '/' {
            while i < b.len() && b[i] != '\n' {
                blank(&mut out, b[i]);
                i += 1;
            }
        } else if c == '/' && i + 1 < b.len() && b[i + 1] == '*' {
            let mut depth = 0;
            while i < b.len() {
                if b[i] == '/' && i + 1 < b.len() && b[i + 1] == '*' {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if b[i] == '*' && i + 1 < b.len() && b[i + 1] == '/' {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    blank(&mut out, b[i]);
                    i += 1;
                }
            }
        } else if c == 'r'
            && i + 1 < b.len()
            && (b[i + 1] == '"' || b[i + 1] == '#')
            && (i == 0 || !(b[i - 1].is_alphanumeric() || b[i - 1] == '_'))
        {
            // raw string: r"..." or r#"..."#
            let mut j = i + 1;
            let mut hashes = 0;
            while j < b.len() && b[j] == '#' {
                hashes += 1;
                j += 1;
            }
            if j < b.len() && b[j] == '"' {
                for k in i..=j {
                    out.push(b[k]);
                }
                j += 1;
                loop {
                    if j >= b.len() {
                        break;
                    }
                    if b[j] == '"' && (0..hashes).all(|h| j + 1 + h < b.len() && b[j + 1 + h] == '#') {
                        out.push('"');
                        for _ in 0..hashes {
                            out.push('#');
                        }
                        j += 1 + hashes;
                        break;
                    }
                    if blank_strings {
                        blank(&mut out, b[j]);
                    } else {
                        out.push(b[j]);
                    }
                    j += 1;
                }
                i = j;
            } else {
                out.push(c);
                i += 1;
            }
        } else if c == '"' {
            out.push('"');
            i += 1;
            while i < b.len() && b[i] != '"' {
                if b[i] == '\\' && i + 1 < b.len() {
                    if blank_strings {
                        out.push(' ');
                        blank(&mut out, b[i + 1]);
                    } else {
                        out.push(b[i]);
                        out.push(b[i + 1]);
                    }
                    i += 2;
                } else {
                    if blank_strings {
                        blank(&mut out, b[i]);
                    } else {
                        out.push(b[i]);
                    }
                    i += 1;
                }
            }
            out.push('"');
            i += 1;
        } else if c == '\'' {
            // char literal ('x', '\n', '\'') vs lifetime ('a)
            if i + 2 < b.len() && b[i + 1] == '\\' {
                let mut j = i + 2;
                while j < b.len() && b[j] != '\'' {
                    j += 1;
                }
                for k in i..=j.min(b.len() - 1) {
                    blank(&mut out, b[k]);
                }
                i = j + 1;
            } else if i + 2 < b.len() && b[i + 2] == '\'' {
                for k in i..i + 3 {
                    blank(&mut out, b[k]);
                }
                i += 3;
            } else {
                out.push(c);
                i += 1;
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// Every `fn` in `path`, with its body.
fn functions(path: &Path) -> Vec<Func> {
    let src = std::fs::read_to_string(path).expect("read test source");
    let no_comments = mask(&src, false);
    let shape = mask(&src, true);
    let starts: Vec<usize> = {
        let mut v = vec![0];
        for (i, ch) in shape.char_indices() {
            if ch == '\n' {
                v.push(i + 1);
            }
        }
        v
    };
    let line_of = |off: usize| starts.partition_point(|&s| s <= off);
    let bytes = shape.as_bytes();
    let mut out = Vec::new();
    let mut search = 0;
    while let Some(rel) = shape[search..].find("fn ") {
        let at = search + rel;
        search = at + 3;
        // `fn` as a keyword, not the tail of an identifier.
        if at > 0 && (bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_') {
            continue;
        }
        let name: String = shape[at + 3..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        // The body starts at the first `{` after the signature; a `;` first
        // means a declaration without a body.
        let rest = &shape[at..];
        let Some(open) = rest.find(|c| c == '{' || c == ';') else { continue };
        if rest.as_bytes()[open] == b';' {
            continue;
        }
        let mut depth = 0i32;
        let mut end = at + open;
        for (k, ch) in shape[at + open..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = at + open + k + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push(Func {
            file: path.to_path_buf(),
            name,
            line: line_of(at),
            body: no_comments[at..end].to_string(),
        });
        search = at + open + 1;
    }
    out
}

/// The flags with which the compiler writes its output NEXT TO THE SOURCE.
const SOURCE_ADJACENT_FLAGS: &[&str] =
    &["\"--emit-ptx\"", "\"--emit-llvm\"", "\"--emit-coprocessor\"", "\"--target=r1cs\""];

/// The flags whose output depends on the profile's GPU architecture.
const PTX_FLAGS: &[&str] = &["\"--emit-ptx\"", "\"--emit-coprocessor\""];

/// The argument of every `.current_dir(..)` call in `text`.
fn current_dirs(text: &str) -> Vec<String> {
    call_args(text, ".current_dir(")
}

/// The argument text of every call that starts with `call` (which ends in its
/// opening parenthesis) in `text`, up to the matching close.
fn call_args(text: &str, call: &str) -> Vec<String> {
    let mut v = Vec::new();
    let mut s = 0;
    while let Some(rel) = text[s..].find(call) {
        let open = s + rel + call.len();
        let mut depth = 1;
        let mut end = open;
        for (k, ch) in text[open..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + k;
                        break;
                    }
                }
                _ => {}
            }
        }
        v.push(text[open..end].to_string());
        s = end;
    }
    v
}

/// **No test compiles a committed fixture in place.** `--emit-ptx`,
/// `--emit-llvm`, `--emit-coprocessor` and `--target=r1cs` write next to their
/// source, so passing `tests/k.ysu` itself rewrites the committed `tests/k.ptx`
/// (with this machine's target and measured latencies) while
/// `committed_ptx_artifacts.rs` reads it. Measured before the fix: every run of
/// the default suite rewrote four or five committed artifacts, under every
/// profile tried. A fixture is copied into a scratch directory first
/// (`common/pinned.rs::copy_fixture`).
#[test]
fn no_test_compiles_a_committed_fixture_in_place() {
    let mut offenders = Vec::new();
    let mut runs = 0;
    for p in test_sources() {
        for f in functions(&p) {
            if !SOURCE_ADJACENT_FLAGS.iter().any(|flag| f.body.contains(flag)) {
                continue;
            }
            // Variables bound to a path under the repository in this function.
            let repo_vars: Vec<String> = f
                .body
                .lines()
                .filter_map(|l| {
                    let t = l.trim();
                    let rest = t.strip_prefix("let ")?;
                    let (lhs, rhs) = rest.split_once('=')?;
                    let name = lhs.split(':').next()?.trim().trim_start_matches("mut ").trim();
                    let rooted = rhs.contains("repo().join(")
                        || rhs.contains("repo_root().join(")
                        || rhs.contains("repo.join(")
                        || rhs.contains("root.join(")
                        || rhs.contains("CARGO_MANIFEST_DIR");
                    // `copy(repo.join(..), &dst)` binds the COPY's result, not a
                    // repo path.
                    if rooted && !rhs.contains("copy(") {
                        Some(name.to_string())
                    } else {
                        None
                    }
                })
                .collect();
            for (k, l) in f.body.lines().enumerate() {
                let t = l.trim();
                let Some(arg) = t.strip_prefix(".arg(") else { continue };
                let arg = arg.trim_end_matches(';').trim_end_matches(')');
                runs += 1;
                let direct = (arg.contains(".join(") && arg.contains("\"tests/"))
                    || arg.contains("format!(\"tests/");
                let via_var = repo_vars
                    .iter()
                    .any(|v| arg.trim_start_matches('&') == v.as_str());
                if direct || via_var {
                    offenders.push(format!(
                        "{}:{} in `{}`: `.arg({arg})`",
                        rel(&f.file),
                        f.line + k,
                        f.name
                    ));
                }
            }
        }
    }
    assert!(runs > 100, "only {runs} `.arg(` calls examined; the reader is not finding compiler runs");
    assert!(
        offenders.is_empty(),
        "these compiler runs take a repository fixture as their source with a flag \
         that writes next to the source, so they rewrite committed artifacts - copy \
         the fixture into a scratch directory first:\n  {}",
        offenders.join("\n  ")
    );
}

/// The functions in a file that return the compiler binary: a helper with a
/// return type whose body names `CARGO_BIN_EXE_Y` or builds `<dir>.join("Y")`.
fn compiler_helpers(funcs: &[Func]) -> Vec<String> {
    funcs
        .iter()
        .filter(|f| {
            let sig = &f.body[..f.body.find('{').unwrap_or(f.body.len())];
            sig.contains("->")
                && (f.body.contains("CARGO_BIN_EXE_Y") || f.body.contains("join(\"Y\")"))
        })
        .map(|f| f.name.clone())
        .collect()
}

/// The functions in a file that return a path under the repository: a helper
/// with a return type whose body names `CARGO_MANIFEST_DIR`.
fn repo_helpers(funcs: &[Func]) -> Vec<String> {
    funcs
        .iter()
        .filter(|f| {
            let sig = &f.body[..f.body.find('{').unwrap_or(f.body.len())];
            sig.contains("->") && f.body.contains("CARGO_MANIFEST_DIR")
        })
        .map(|f| f.name.clone())
        .collect()
}

/// The right-hand side of `let [mut] name[: T] = <rhs>;` in `body`, over
/// several lines if need be - every binding of that name.
fn let_bindings<'a>(body: &'a str, name: &str) -> Vec<&'a str> {
    let mut v = Vec::new();
    let mut s = 0;
    while let Some(rel) = body[s..].find("let ") {
        let at = s + rel + 4;
        s = at;
        let rest = body[at..].trim_start().trim_start_matches("mut ").trim_start();
        let found: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if found != name {
            continue;
        }
        let Some(eq) = rest.find('=') else { continue };
        let rhs = &rest[eq + 1..];
        v.push(&rhs[..rhs.find(';').unwrap_or(rhs.len())]);
    }
    v
}

fn is_ident(e: &str) -> bool {
    !e.is_empty() && e.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Is `expr` (a working directory) the repository or a directory inside it?
/// Named (`repo()`, `CARGO_MANIFEST_DIR`, a helper returning it), joined
/// beneath (`repo().join("tests")` - where the compiler would probe and WRITE a
/// profile into the tree), or a local bound to either.
fn rooted_in_repo(expr: &str, body: &str, repo_helpers: &[String], depth: usize) -> bool {
    let e: String = expr.chars().filter(|c| !c.is_whitespace()).collect();
    let e = e.trim_start_matches('&').trim_start_matches("pinned::");
    if e.contains("CARGO_MANIFEST_DIR") {
        return true;
    }
    let base = e.split('.').next().unwrap_or("");
    if matches!(base, "repo()" | "repo" | "repo_root()" | "repo_root" | "root" | "root()") {
        return true;
    }
    if let Some(name) = base.strip_suffix("()") {
        if repo_helpers.iter().any(|h| h == name) {
            return true;
        }
    }
    depth == 0
        && is_ident(base)
        && let_bindings(body, base)
            .iter()
            .any(|rhs| rooted_in_repo(rhs, body, repo_helpers, depth + 1))
}

/// Is `expr` (the argument of a `Command::new`) the compiler? Directly, through
/// a helper that returns it, or through a local bound to either.
fn names_the_compiler(expr: &str, body: &str, helpers: &[String], depth: usize) -> bool {
    let e: String = expr.chars().filter(|c| !c.is_whitespace()).collect();
    let e = e.trim_start_matches('&');
    if e.contains("CARGO_BIN_EXE_Y") || e.ends_with(".join(\"Y\")") {
        return true;
    }
    if let Some(name) = e.strip_suffix("()") {
        if helpers.iter().any(|h| h == name) {
            return true;
        }
    }
    depth == 0
        && is_ident(e)
        && let_bindings(body, e)
            .iter()
            .any(|rhs| names_the_compiler(rhs, body, helpers, depth + 1))
}

/// Does this function run the compiler?
fn spawns_compiler(body: &str, helpers: &[String]) -> bool {
    call_args(body, "Command::new(")
        .iter()
        .any(|a| names_the_compiler(a, body, helpers, 0))
}

/// The names of a signature's string-typed parameters - the ones a flag can
/// arrive through.
fn string_params(sig: &str) -> Vec<String> {
    let Some(open) = sig.find('(') else { return Vec::new() };
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut parts = Vec::new();
    for ch in sig[open + 1..].chars() {
        match ch {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' if depth == 0 => break,
            ')' | ']' | '>' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(ch);
    }
    parts.push(cur);
    parts
        .iter()
        .filter_map(|p| {
            let (name, ty) = p.split_once(':')?;
            let ty: String = ty.chars().filter(|c| !c.is_whitespace()).collect();
            let stringy = matches!(
                ty.as_str(),
                "&str" | "String" | "&String" | "&'staticstr" | "&[&str]" | "&[String]"
                    | "&[&'staticstr]" | "Vec<String>" | "Vec<&str>"
            );
            let name = name.trim().trim_start_matches("mut ").trim();
            stringy.then(|| name.to_string())
        })
        .collect()
}

/// Is `word` an identifier occurring in `text` on its own?
fn has_word(text: &str, word: &str) -> bool {
    let b = text.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut s = 0;
    while let Some(rel) = text[s..].find(word) {
        let at = s + rel;
        let end = at + word.len();
        if (at == 0 || !ident(b[at - 1])) && (end >= b.len() || !ident(b[end])) {
            return true;
        }
        s = at + 1;
    }
    false
}

/// Does this function compile PTX against whatever `.ysu_hw_profile` this
/// machine has? It runs the compiler, with the repository as working directory
/// - named, or INHERITED: a test binary runs with the package root as its
/// working directory, so a `Command` with no `current_dir` reads the repository
/// profile too - and the PTX flag is either written in it or arrives through a
/// string parameter in a file that passes one.
fn compiles_ptx_against_this_machine(
    body: &str,
    file_passes_ptx: bool,
    helpers: &[String],
    repo_helpers: &[String],
) -> bool {
    if !spawns_compiler(body, helpers) {
        return false;
    }
    let dirs = current_dirs(body);
    if !(dirs.is_empty() || dirs.iter().any(|d| rooted_in_repo(d, body, repo_helpers, 0))) {
        return false;
    }
    if PTX_FLAGS.iter().any(|flag| body.contains(flag)) {
        return true;
    }
    let sig = &body[..body.find('{').unwrap_or(body.len())];
    let args: Vec<String> =
        call_args(body, ".arg(").into_iter().chain(call_args(body, ".args(")).collect();
    file_passes_ptx
        && string_params(sig)
            .iter()
            .any(|p| args.iter().any(|a| has_word(a, p)))
}

/// **No verdict depends on this machine's GPU.** The compiler reads
/// `.ysu_hw_profile` from its working directory, so a PTX compile run with the
/// repository as working directory targets whatever card this machine has.
/// Measured before the fix, with the repository's profile set three ways: a
/// GPU-less machine (compute capability 0.0) failed 3 tests that pass on an
/// sm_89 card, and an sm_90 profile failed 13, every one of them assembling
/// that output at a hardcoded `-arch=sm_89`, which refuses a `.target sm_90`
/// module.
///
/// The rule: every PTX compile pins its profile in a scratch directory
/// (`common/pinned.rs::pin`) - a test that LAUNCHES the kernel too, since a
/// module declaring the floor (sm_80) loads on every supported card. And no
/// test reads the repository's profile as a template: what it holds is this
/// machine's card and measurements.
#[test]
fn no_verdict_depends_on_this_machines_gpu() {
    let mut offenders = Vec::new();
    let mut ptx_compiles = 0;
    for p in test_sources() {
        // (a) the repository's profile is never read - by name, or by the
        // bare relative path, which a test binary resolves against the
        // package root.
        let text = mask(&std::fs::read_to_string(&p).expect("read"), false);
        for (i, l) in text.lines().enumerate() {
            let named = l.contains("join(\".ysu_hw_profile\")")
                && (l.contains("repo") || l.contains("CARGO_MANIFEST_DIR"));
            let relative = l
                .match_indices("(\".ysu_hw_profile\"")
                .any(|(at, _)| !l[..at].ends_with("join"));
            if named || relative {
                offenders.push(format!(
                    "{}:{}: reads the repository's profile: {}",
                    rel(&p),
                    i + 1,
                    l.trim()
                ));
            }
        }
        // (b) no PTX compile runs with the repository as working directory.
        let funcs = functions(&p);
        let helpers = compiler_helpers(&funcs);
        let roots = repo_helpers(&funcs);
        let file_passes_ptx = PTX_FLAGS.iter().any(|flag| text.contains(flag));
        for f in &funcs {
            if PTX_FLAGS.iter().any(|flag| f.body.contains(flag)) {
                ptx_compiles += 1;
            }
            if compiles_ptx_against_this_machine(&f.body, file_passes_ptx, &helpers, &roots) {
                offenders.push(format!(
                    "{}:{}: `{}` compiles PTX with the repository as working directory, \
                     i.e. for this machine's card - pin the profile in a scratch directory",
                    rel(&f.file),
                    f.line,
                    f.name
                ));
            }
        }
    }
    // Non-vacuity, both ways: the sweep must SEE the PTX compiles (dozens of
    // them), and the predicate must fire on the shape it exists to refuse - a
    // predicate matching nothing reports a clean suite perfectly.
    assert!(
        ptx_compiles > 30,
        "only {ptx_compiles} functions that compile PTX were found; the reader has stopped \
         seeing them"
    );
    let helpers = vec!["bin".to_string()];
    let roots = vec!["manifest".to_string()];
    for shape in [
        // the repository named as working directory
        "fn k() { Command::new(env!(\"CARGO_BIN_EXE_Y\")).arg(&s).arg(\"--emit-ptx\").current_dir(repo()).output(); }",
        "fn k() { Command::new(env!(\"CARGO_BIN_EXE_Y\")).arg(&s).arg(\"--emit-coprocessor\").current_dir(&repo).output(); }",
        "fn k() { Command::new(env!(\"CARGO_BIN_EXE_Y\")).arg(&s).arg(\"--emit-ptx\").current_dir(Path::new(env!(\"CARGO_MANIFEST_DIR\"))).output(); }",
        // INHERITED: no working directory at all
        "fn k() { Command::new(env!(\"CARGO_BIN_EXE_Y\")).arg(&s).arg(\"--emit-ptx\").output(); }",
        // the flag arrives as a parameter
        "fn compile(src: &Path, flag: &str) { Command::new(env!(\"CARGO_BIN_EXE_Y\")).arg(src).arg(flag).output(); }",
        // the compiler reached through a helper, and through a local
        "fn k() { Command::new(bin()).arg(&s).arg(\"--emit-ptx\").current_dir(repo()).output(); }",
        "fn k() { let y = PathBuf::from(\n env!(\"CARGO_BIN_EXE_Y\")); Command::new(&y).arg(\"--emit-ptx\").output(); }",
        // the repository through a local of any name, beneath the root, or
        // through a helper
        "fn k() { let dir = PathBuf::from(env!(\"CARGO_MANIFEST_DIR\")); Command::new(env!(\"CARGO_BIN_EXE_Y\")).arg(\"--emit-ptx\").current_dir(&dir).output(); }",
        "fn k() { Command::new(env!(\"CARGO_BIN_EXE_Y\")).arg(\"--emit-ptx\").current_dir(repo().join(\"tests\")).output(); }",
        "fn k() { Command::new(env!(\"CARGO_BIN_EXE_Y\")).arg(\"--emit-ptx\").current_dir(manifest()).output(); }",
    ] {
        assert!(
            compiles_ptx_against_this_machine(shape, true, &helpers, &roots),
            "the predicate does not recognise `{shape}`, so it would pass a machine-profile \
             compile"
        );
    }
    for shape in [
        // a pinned scratch directory: the fix
        "fn k() { Command::new(env!(\"CARGO_BIN_EXE_Y\")).arg(&s).arg(\"--emit-ptx\").current_dir(&dir).output(); }",
        "fn k() { let dir = pinned::pinned_scratch(\"t\", pinned::SM_PINNED); Command::new(env!(\"CARGO_BIN_EXE_Y\")).arg(\"--emit-ptx\").current_dir(&dir).output(); }",
        // not the compiler: other tools may inherit the working directory
        "fn k(flag: &str) { Command::new(\"clang\").arg(flag).output(); }",
    ] {
        assert!(
            !compiles_ptx_against_this_machine(shape, true, &helpers, &roots),
            "the predicate refuses `{shape}`, which compiles nothing against this machine"
        );
    }
    assert!(
        !compiles_ptx_against_this_machine(
            "fn compile(src: &Path, flag: &str) { Command::new(env!(\"CARGO_BIN_EXE_Y\")).arg(src).arg(flag).output(); }",
            false,
            &helpers,
            &roots
        ),
        "a parameterised flag in a file that never passes a PTX flag is refused"
    );
    offenders.sort();
    offenders.dedup();
    assert!(
        offenders.is_empty(),
        "these tests' verdicts depend on this machine's GPU:\n  {}",
        offenders.join("\n  ")
    );
}

/// The pinned profile is the emitter's own floor. `common/pinned.rs` claims a
/// module compiled at `SM_PINNED` assembles at every supported arch and loads
/// on every supported card - which is true of `PTX_FLOOR` and of nothing above
/// it - so the claim is checked against the emitter rather than restated.
#[test]
fn the_pinned_profile_is_the_emitters_floor() {
    let (major, minor) = pinned::SM_PINNED.split_once('.').expect("a dotted SM version");
    assert_eq!(
        format!("sm_{major}{minor}"),
        y::ptx_emitter::PTX_FLOOR,
        "SM_PINNED is not the emitter's floor, so a pinned kernel may not load on an \
         older card than the one this suite was run on"
    );
}

/// `pin` and `copy_fixture` take effect: the compiler reads the pinned profile
/// (an sm_86 pin yields `.target sm_86`, which no probe of this container and
/// no default produces), and a copied fixture lives in the scratch directory,
/// not in the repository. A `pin` that wrote nothing would pass every other
/// test on a GPU-less machine, where the probe ALSO lands on sm_80.
#[test]
fn pinning_decides_the_target_and_copies_leave_the_repo_alone() {
    let dir = pinned::pinned_scratch("selftest", "8.6");
    let src = pinned::copy_fixture(&dir, "tests/smem_roundtrip.ysu");
    assert!(src.starts_with(&dir), "copy_fixture returned {} outside its directory", src.display());
    assert!(!src.starts_with(repo()), "copy_fixture returned a path inside the repository");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    assert!(out.status.success(), "the fixture did not compile:\n{}", String::from_utf8_lossy(&out.stdout));
    let ptx = std::fs::read_to_string(src.with_extension("ptx")).expect("no .ptx beside the copy");
    assert!(
        ptx.lines().any(|l| l.trim() == ".target sm_86"),
        "an sm_86 pin did not decide the target:\n{}",
        ptx.lines().take(12).collect::<Vec<_>>().join("\n")
    );
}

/// **A pinned directory finds the solver the repository finds.** The compiler
/// resolves three of its z3 candidates against its WORKING directory, so moving
/// a compile out of the repository lost a solver installed in a repo-local
/// venv: with z3 only at `<repo>/venv/bin/z3`, 21 tests in 7 files failed that
/// pass when the compile runs in the repository (`pinned::mirror_solver`).
///
/// Checked through the real compiler on any machine: a stand-in root holds a
/// stub solver that answers `unsat` and leaves a marker, and every other way of
/// finding a solver is removed. Mirrored, the compile must run the stub. With
/// nothing to mirror it must report that no solver could be run - the control
/// that says the stub was reached through the mirror and not through some
/// other candidate. Neither directory goes through `pin`: that would mirror
/// the REAL repository first wherever it has a repo-local solver, and the stub
/// would never be linked.
///
/// That `pin` mirrors from the repository is visible behaviourally only where
/// the repository has a relative candidate, so it is checked both ways: the
/// pinned directory resolves every candidate the repository resolves, and
/// `pin`'s body passes `repo()` to `mirror_solver`.
#[cfg(unix)]
#[test]
fn a_pinned_directory_finds_the_solver_the_repository_finds() {
    use std::os::unix::fs::PermissionsExt;

    let root = pinned::scratch("solver_root");
    let marker = root.join("stub_ran");
    let stub = root.join("venv/bin/z3");
    std::fs::create_dir_all(stub.parent().unwrap()).unwrap();
    // Shell builtins only: the compile runs with PATH removed. The query is read
    // to EOF so the compiler's write never meets a closed pipe.
    std::fs::write(
        &stub,
        format!("#!/bin/sh\n: > '{}'\nwhile read -r _q; do :; done\necho unsat\n", marker.display()),
    )
    .unwrap();
    std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();

    let program = "fn main() {\n    @safe {\n        @invariant(i >= 0)\n        for i in 0..10 {\n            print_int(i);\n        }\n    }\n}\n";
    // The compiler in `target/` would also find a repo-local `venv/bin/z3`
    // beside itself; this copy has nothing beside it, so the working
    // directory is the only place a solver can come from.
    let detached = pinned::DetachedCompiler::new("solver_mirror");
    let compile = |dir: &Path| -> String {
        std::fs::write(dir.join(".ysu_hw_profile"), pinned::profile_text(pinned::SM_PINNED)).unwrap();
        let src = dir.join("inv.ysu");
        std::fs::write(&src, program).unwrap();
        let out = pinned::DetachedCompiler::output(
            std::process::Command::new(&detached.exe)
                .arg(&src)
                .current_dir(dir)
                .env_remove("Y_Z3_PATH")
                .env_remove("Y_ALLOW_UNVERIFIED_INVARIANTS")
                .env("PATH", "/nonexistent-path")
                .env("HOME", "/nonexistent-home"),
        );
        format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
    };

    let mirrored = pinned::scratch("solver_mirrored");
    pinned::mirror_solver(&root, &mirrored);
    let out = compile(&mirrored);
    assert!(marker.exists(), "the compile did not run the solver mirrored from the root:\n{out}");
    assert!(out.contains("Front-end analysis complete"), "the stub answered unsat and the front end still refused:\n{out}");

    std::fs::remove_file(&marker).unwrap();
    let bare_root = pinned::scratch("solver_none");
    let unmirrored = pinned::scratch("solver_unmirrored");
    pinned::mirror_solver(&bare_root, &unmirrored);
    assert!(
        unmirrored.join("venv").symlink_metadata().is_err(),
        "mirror_solver linked a candidate its root does not have"
    );
    let out = compile(&unmirrored);
    assert!(!marker.exists(), "the stub ran with nothing mirrored, so it was found some other way:\n{out}");
    assert!(
        out.contains("SMT solver could not be run") && !out.contains("Front-end analysis complete"),
        "with no solver anywhere the invariant must be refused:\n{out}"
    );

    let repo_root = repo();
    let pinned_dir = pinned::pinned_scratch("solver_pin", pinned::SM_PINNED);
    for cand in y::type_checker::z3_candidates() {
        let rel = Path::new(&cand);
        if rel.is_absolute() || rel.components().count() < 2 {
            continue;
        }
        assert_eq!(
            pinned_dir.join(rel).exists(),
            repo_root.join(rel).exists(),
            "a pinned directory and the repository disagree on the solver candidate `{cand}`"
        );
    }
    let pin = functions(&repo_root.join("tests/common/pinned.rs"))
        .into_iter()
        .find(|f| f.name == "pin")
        .expect("tests/common/pinned.rs defines `pin`");
    assert!(
        pin.body.contains("mirror_solver(&repo(), dir)"),
        "`pin` no longer mirrors the repository's solver candidates:\n{}",
        pin.body
    );
    for d in [root, bare_root, mirrored, unmirrored, pinned_dir] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// **A pinned compile never measures the device.** Every `--emit-ptx` and
/// `--emit-llvm` compile calls `load_or_measure_drift_costs`, and a profile with
/// no `@ZeroDrift` costs recorded for its GPU name makes it time a probe kernel
/// on the device and append the result - seconds per compile, on a machine with
/// a GPU, with the representation choice following the measurement. So the
/// pinned profile records costs ([`pinned::DRIFT_COSTS`]) and this asserts the
/// compiler TAKES them.
///
/// Two observations, because each machine can show only one failure: where
/// there is no GPU an untaken cost table reads "no device measurements", and
/// where there is one a fresh measurement appends to the profile. So the report
/// must name the pinned cost of the representation chosen, and the profile must
/// be byte-identical after the compile.
#[test]
fn a_pinned_compile_never_measures_the_device() {
    let table =
        y::zero_drift::parse_costs(&pinned::profile_text(pinned::SM_PINNED), pinned::GPU_NAME);
    for r in y::zero_drift::DriftRepr::ALL {
        if r.is_exact() {
            assert!(
                table.contains_key(&r),
                "the pinned profile records no cost the compiler reads for {} - a \
                 requirement only it satisfies would still measure the device",
                r.name()
            );
        }
    }

    // `tests/test_drift.ysu` requires sm_89 and carries two accumulators.
    let dir = pinned::pinned_scratch("drift_costs", pinned::SM_FP8);
    let before = std::fs::read(dir.join(".ysu_hw_profile")).expect("pinned profile");
    let src = pinned::copy_fixture(&dir, "tests/test_drift.ysu");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "the drift fixture did not compile:\n{text}");
    let reports: Vec<&str> = text.lines().filter(|l| l.contains("@ZeroDrift")).collect();
    assert!(
        reports.len() >= 2,
        "the fixture reported {} @ZeroDrift decisions; it no longer exercises the \
         selector:\n{text}",
        reports.len()
    );
    for l in &reports {
        // `-> @ZeroDrift acc: Q32.32 -> Q32.32 (measured 1922 ps/acc, ...)`
        let (repr, rest) = l
            .split_once("@ZeroDrift ")
            .and_then(|(_, r)| r.split_once(" -> "))
            .and_then(|(_, r)| r.split_once(" ("))
            .unwrap_or_else(|| panic!("unrecognised @ZeroDrift report `{l}`"));
        let (_, ps) = pinned::DRIFT_COSTS
            .iter()
            .find(|(name, _)| *name == repr)
            .unwrap_or_else(|| panic!("`{l}` chose {repr}, which has no pinned cost"));
        let want = format!("measured {:.0} ps/acc", ps.parse::<f64>().expect("a cost"));
        assert!(
            rest.starts_with(&want),
            "`{l}` does not report the pinned cost ({want}): the compiler did not take the \
             recorded costs, so on a machine with a GPU it measures"
        );
    }
    assert_eq!(
        std::fs::read(dir.join(".ysu_hw_profile")).expect("profile after"),
        before,
        "the compile appended to the pinned profile: it measured the device"
    );
}

/// **The reader returns exact function extents.** Every gate in this file reads
/// functions through [`functions`]; a body that starts or ends in the wrong
/// place hides the end of one function and charges the next with its tail.
/// Checked on a synthetic file with multi-byte characters in a string, a char
/// literal and a comment - the case that shifted 178 extents - and on every real
/// test source, with a floor so a reader that finds nothing cannot pass. (A
/// body keeps its string literals and blanks its comments and char literals, a
/// byte per byte.)
#[test]
fn the_reader_returns_exact_function_extents() {
    let dir = pinned::scratch("reader");
    let sample = dir.join("sample.rs");
    std::fs::write(
        &sample,
        "fn a() { let s = \"\u{2014}\u{e9}{\"; let c = '\u{e9}'; } // \u{2014} }\nfn b() { x(\"\u{2014}\") }\n",
    )
    .expect("write sample");
    let found: Vec<(String, String)> =
        functions(&sample).into_iter().map(|f| (f.name, f.body)).collect();
    assert_eq!(
        found,
        vec![
            ("a".to_string(), "fn a() { let s = \"\u{2014}\u{e9}{\"; let c =     ; }".to_string()),
            ("b".to_string(), "fn b() { x(\"\u{2014}\") }".to_string()),
        ],
        "the reader mis-sliced a file with multi-byte characters"
    );

    let mut n = 0;
    for p in test_sources() {
        let src = std::fs::read_to_string(&p).expect("read");
        for blank in [false, true] {
            assert_eq!(
                mask(&src, blank).len(),
                src.len(),
                "{}: masking changed the byte length, so offsets found in one mask \
                 no longer index the other",
                rel(&p)
            );
        }
        for f in functions(&p) {
            n += 1;
            assert!(
                f.body.starts_with("fn ") && f.body.ends_with('}'),
                "{}:{} `{}` was read as {:?}...{:?}",
                rel(&p),
                f.line,
                f.name,
                f.body.chars().take(20).collect::<String>(),
                f.body.chars().rev().take(20).collect::<String>()
            );
        }
    }
    assert!(n > 1000, "only {n} functions read from the test sources");
}
