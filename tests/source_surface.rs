//! What the source ADVERTISES against what is actually built.
//!
//! Two censuses, both of the same shape as `no_source_file_hardcodes_a_ptx_
//! version_above_the_floor`: read the source, not an artifact, because a
//! literal that has drifted out of use assembles perfectly and a token nobody
//! matches lexes perfectly.
//!
//! **`mod` is not the same question as "compiled", and getting that wrong is
//! what this file exists to stop.** `README.md` recorded `src/ypm.rs` as "not
//! compiled; no `mod` declares them" - and `ypm` is a `[[bin]]` target, so
//! `cargo build --release` produces a working 560 KB executable. A first draft
//! of this gate repeated the identical mistake. A file is compiled if it is a
//! module OR a binary target, and both have to be checked.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(p: &str) -> String {
    std::fs::read_to_string(repo().join(p)).unwrap_or_else(|e| panic!("read {p}: {e}"))
}

/// Module names declared by `mod x;` / `pub mod x;` in the crate roots.
fn declared_modules() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for f in ["src/lib.rs", "src/main.rs"] {
        for line in read(f).lines() {
            let t = line.trim();
            let t = t.strip_prefix("pub ").unwrap_or(t);
            if let Some(rest) = t.strip_prefix("mod ") {
                if let Some(name) = rest.split(';').next() {
                    let name = name.trim();
                    if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        out.insert(name.to_string());
                    }
                }
            }
        }
    }
    out
}

/// Source paths named by a `[[bin]]` target in `Cargo.toml`.
fn binary_paths() -> BTreeSet<String> {
    read("Cargo.toml")
        .lines()
        .filter_map(|l| {
            let t = l.trim();
            let rest = t.strip_prefix("path")?.trim_start().strip_prefix('=')?;
            Some(rest.trim().trim_matches('"').to_string())
        })
        .collect()
}

/// **No `.rs` under `src/` may be dead**: it is a module, or it is a binary
/// target, or it should not be in the tree.
///
/// This is what `c_emitter.rs` was - 1,301 lines the C backend left behind when
/// it was removed, declared by nothing, compiled by nothing, and therefore
/// incapable of being wrong in a way any test could report. A reader has no way
/// to tell it apart from live code.
#[test]
fn every_source_file_is_either_a_module_or_a_binary() {
    let mods = declared_modules();
    let bins = binary_paths();
    let mut files = Vec::new();
    for e in std::fs::read_dir(repo().join("src")).expect("read src/") {
        let p = e.expect("dir entry").path();
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let stem = p.file_stem().unwrap().to_str().unwrap().to_string();
        if stem == "lib" || stem == "main" {
            continue;
        }
        files.push(stem);
    }
    // Non-vacuity: a census over an empty directory reports no orphans
    // perfectly well.
    assert!(
        files.len() > 30,
        "only {} source files found - the sweep is not looking where it thinks",
        files.len()
    );
    assert!(mods.len() > 30, "only {} modules declared", mods.len());

    let orphans: Vec<&String> = files
        .iter()
        .filter(|f| !mods.contains(*f) && !bins.contains(&format!("src/{f}.rs")))
        .collect();
    assert!(
        orphans.is_empty(),
        "these files under src/ are declared by no `mod` and named by no \
         [[bin]] target, so nothing compiles them: {orphans:?}. Either wire \
         them in or delete them - an uncompiled file cannot be wrong, which \
         means no test can tell you it is."
    );
}

/// **Every `@`-directive the lexer knows must be matched by the parser.**
///
/// `@inline`, `@noinline` and `@avx_emit` had dedicated `TokenKind`s that no
/// parser arm ever matched, so the lexer advertised three features the language
/// does not have. Fail-closed - the parser says "Unexpected top-level item" -
/// but a token kind nobody consumes is a claim the source makes and cannot
/// honour, and the same shape one layer up is how `@zk_target(scheme =
/// "plonkish")` came to print a header naming an arithmetization it never used.
#[test]
fn every_lexed_directive_is_matched_by_the_parser() {
    let lexer = read("src/lexer.rs");
    let parser = read("src/parser.rs");
    let mut pairs = Vec::new();
    for line in lexer.lines() {
        let t = line.trim();
        // `"@name" => TokenKind::AtName,`
        let Some(rest) = t.strip_prefix('"') else { continue };
        let Some((spelling, tail)) = rest.split_once('"') else { continue };
        if !spelling.starts_with('@') {
            continue;
        }
        let Some(kind) = tail.split("TokenKind::").nth(1) else { continue };
        let kind: String = kind
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !kind.is_empty() {
            pairs.push((spelling.to_string(), kind));
        }
    }
    assert!(
        pairs.len() > 20,
        "only {} directives found in the lexer - the parse of lexer.rs has \
         drifted and this gate is checking nothing",
        pairs.len()
    );
    let unmatched: Vec<&(String, String)> = pairs
        .iter()
        .filter(|(_, kind)| !parser.contains(kind.as_str()))
        .collect();
    assert!(
        unmatched.is_empty(),
        "these directives lex to a TokenKind the parser never matches, so the \
         language advertises them and cannot accept them: {unmatched:?}"
    );
}

/// The control that makes deleting a directive token safe: an unknown `@`-word
/// is refused, with a line number, exactly as a known-but-unparsed one was.
///
/// Without this, "delete the three token kinds" and "delete the whole directive
/// lexing path" are indistinguishable to the gate above.
#[test]
fn an_unknown_directive_is_still_refused_with_a_line_number() {
    let dir = std::env::temp_dir().join(format!("y_surface_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    for src in ["@inline\nfn main() {\n}\n", "@definitely_not_a_directive\nfn main() {\n}\n"] {
        let f = dir.join("m.ysu");
        std::fs::write(&f, src).expect("write");
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&f)
            .arg("--emit-llvm")
            .current_dir(repo())
            .output()
            .expect("run Y");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.status.success(),
            "`{}` was ACCEPTED; a directive the parser cannot honour must be \
             refused, not ignored:\n{text}",
            src.lines().next().unwrap()
        );
        assert!(
            text.contains("Line 1"),
            "`{}` was refused without naming the line:\n{text}",
            src.lines().next().unwrap()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = Path::new("");
}

/// **Every `.rs` the README's repo-layout listing names must exist.**
///
/// Scoped to the fenced layout block on purpose. Prose may legitimately name a
/// deleted file while explaining that it was deleted - `rocm_emitter.rs` and
/// `auto_vectorize.rs` are both discussed that way, and CLAUDE.md's own rule
/// is to keep dated history with a forward pointer rather than erase it. A
/// stale name in the LAYOUT is a different thing: it tells a reader the tree
/// contains something it does not.
///
/// This found four at once - `auto_vectorize.rs`, `layout_pass.rs`,
/// `rocm_emitter.rs` and `c_emitter.rs` - three of which had been deleted in
/// earlier sessions with the listing left behind.
#[test]
fn the_readme_layout_names_only_files_that_exist() {
    let readme = read("README.md");
    let block = readme
        .split("\n```")
        .find(|b| b.contains("self_hosted/") && b.contains("tests/"))
        .expect("the repo-layout block is gone; re-point this gate at it");
    let mut named = BTreeSet::new();
    let bytes: Vec<char> = block.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_alphanumeric() || bytes[i] == '_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_alphanumeric() || bytes[i] == '_') {
                i += 1;
            }
            let word: String = bytes[start..i].iter().collect();
            if block[..].contains(&format!("{word}.rs")) && i < bytes.len() && bytes[i] == '.' {
                let tail: String = bytes[i..].iter().take(3).collect();
                if tail == ".rs" {
                    named.insert(format!("{word}.rs"));
                }
            }
        } else {
            i += 1;
        }
    }
    assert!(
        named.len() > 15,
        "only {} source files named in the layout block - the gate has lost \
         track of it and is checking nothing",
        named.len()
    );
    let missing: Vec<&String> = named
        .iter()
        .filter(|f| {
            !["src", "tests", "src/bin", "fuzz/fuzz_targets"]
                .iter()
                .any(|d| repo().join(d).join(f.as_str()).exists())
        })
        .collect();
    assert!(
        missing.is_empty(),
        "the README's repo layout names files that do not exist: {missing:?}. \
         A reader takes that listing for the current tree."
    );
}

/// **`ypm` is a separate binary and its manifest is `Ysu.toml`** - the docs
/// said `Y ypm ...` and `Y.toml`, and both are wrong in a way that stops the
/// documented flow dead.
///
/// Found by running the documented commands, which is the sweep that keeps
/// producing bugs in this repo. `Y ypm init` runs the COMPILER with `ypm` as a
/// source filename (`Failed to read file`), and `ypm build` refuses a
/// directory whose manifest is not under the name it looks for. The real flow
/// - `ypm new demo && ypm build && ypm run` - works and prints
/// `Hello, World from YPM!`, which is why the README calling this file "not
/// compiled" was the more serious error of the two.
#[test]
fn the_ypm_docs_name_the_right_binary_and_the_right_manifest() {
    let ypm = read("src/ypm.rs");
    let docs = read("docs/y_language_documentation.md");
    assert!(
        ypm.contains("\"Ysu.toml\""),
        "src/ypm.rs no longer reads `Ysu.toml`; re-point §19 of the docs at \
         whatever it reads now"
    );
    assert!(
        docs.contains("`Ysu.toml`"),
        "§19 does not name the manifest ypm actually reads"
    );
    // Only the COMMAND LINES, not the prose. §19 now explains that the old
    // spelling was wrong, and it has to quote it to do so - the same reason
    // `the_readme_layout_names_only_files_that_exist` is scoped to the layout
    // block rather than the whole file.
    let sec = {
        let a = docs.find("## 19.").expect("§19 is gone");
        let b = docs[a + 6..].find("\n## ").map(|i| a + 6 + i).unwrap_or(docs.len());
        &docs[a..b]
    };
    let bad: Vec<&str> = sec
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("./target/release/") && l.contains(" ypm "))
        .collect();
    assert!(
        bad.is_empty(),
        "§19 gives these as commands to run: {bad:?} - `ypm` is its own \
         [[bin]], so `Y ypm ...` runs the compiler with `ypm` as a source \
         filename and reports `Failed to read file`"
    );
    assert!(
        sec.lines()
            .any(|l| l.trim().starts_with("./target/release/ypm ")),
        "§19 no longer shows a runnable ypm command at all"
    );
    // The two subcommands that were documented and do not exist. Asserting on
    // the usage text rather than a hand-kept list, so adding one for real
    // retires the check by itself.
    let usage_start = ypm.find("ypm new").expect("ypm's usage text moved");
    let usage: String = ypm[usage_start..].chars().take(600).collect();
    for cmd in ["new", "init", "build", "run", "test", "clean"] {
        assert!(
            usage.contains(&format!("ypm {cmd}")),
            "`{cmd}` is documented in §19 but is not in ypm's own usage text"
        );
    }
    for absent in ["add", "install"] {
        if usage.contains(&format!("ypm {absent}")) {
            continue; // implemented since - the docs may name it again
        }
        assert!(
            !docs.contains(&format!("ypm {absent} ")),
            "§19 documents `ypm {absent}`, which answers `Unknown command`"
        );
    }
}

/// Collect `Y <file.ysu> [flags]` invocations out of the docs' fenced blocks,
/// in any of the three spellings the docs use.
fn documented_y_commands() -> Vec<(String, Vec<String>)> {
    let mut docs = vec!["README.md".to_string()];
    if let Ok(rd) = std::fs::read_dir(repo().join("docs")) {
        for e in rd.flatten() {
            if e.path().extension().and_then(|s| s.to_str()) == Some("md") {
                docs.push(format!("docs/{}", e.file_name().to_str().unwrap()));
            }
        }
    }
    docs.sort();
    let mut out = Vec::new();
    for d in docs {
        let text = read(&d);
        let mut inside = false;
        for line in text.lines() {
            if line.starts_with("```") {
                inside = !inside;
                continue;
            }
            if !inside {
                continue;
            }
            let t = line.trim();
            let t = t.split('#').next().unwrap_or(t).trim();
            let rest = if let Some(r) = t.strip_prefix("./target/release/Y ") {
                r
            } else if let Some(r) = t.strip_prefix("cargo run -- ") {
                r
            } else if let Some(r) = t.strip_prefix("Y ") {
                r
            } else {
                continue;
            };
            let parts: Vec<String> = rest.split_whitespace().map(str::to_string).collect();
            let Some(src) = parts.first() else { continue };
            if !src.ends_with(".ysu") || !repo().join(src).exists() {
                continue; // a file the reader is told to create, not a repo fixture
            }
            // Skip what this gate cannot legitimately judge:
            //  - the ZK backend needs `--features zk`, absent from a default build
            //  - `--emit-c` and friends are DOCUMENTED as failing, and do
            //  - autotuning touches the GPU
            let flags: Vec<String> = parts[1..].to_vec();
            if flags.iter().any(|f| {
                f.contains("r1cs") || f.contains("zk") || f.contains("autotune")
                    || *f == "--emit-c" || *f == "--c" || f.contains("target=c")
            }) {
                continue;
            }
            out.push((src.clone(), flags));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// **Every documented `Y` command over a repo fixture must actually work.**
///
/// `docs/y_language_documentation.md` presented `tests/train_spec.ysu` with two
/// commands, and BOTH failed - `--emit-llvm` because the LLVM host backend
/// correctly refuses `GlobalMemory::load`, and `--emit-ptx` because the fixture
/// carried `@ZeroDrift` on a bare `F32` with no `@bounds`, which the compiler
/// correctly refuses. The compiler was right both times; the fixture and the
/// documentation were wrong. The doc also displayed a completely DIFFERENT
/// kernel under that filename.
///
/// Existence is not enough - `the_readme_layout_names_only_files_that_exist`
/// would have passed this, because the file was there and simply did not
/// compile.
#[test]
fn every_documented_y_command_over_a_repo_fixture_succeeds() {
    let cmds = documented_y_commands();
    assert!(
        cmds.len() >= 3,
        "only {} documented Y commands found over repo fixtures - the scrape \
         has drifted and this gate is checking nothing",
        cmds.len()
    );
    let mut broken = Vec::new();
    for (src, flags) in &cmds {
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(src)
            .args(flags)
            .current_dir(repo())
            .output()
            .expect("run Y");
        if !out.status.success() {
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let why = text
                .lines()
                .find(|l| l.contains("Error") || l.contains("[!]"))
                .unwrap_or("<no diagnostic>")
                .trim()
                .to_string();
            broken.push(format!("`Y {src} {}` -> {why}", flags.join(" ")));
        }
    }
    assert!(
        broken.is_empty(),
        "documented commands that do not work:\n  {}",
        broken.join("\n  ")
    );
}
