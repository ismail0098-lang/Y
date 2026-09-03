//! Two directives reached `FuncDecl` and were read by nothing, and the census
//! predicate that finds them is the CONSUMER COUNT, not the artifact diff.
//!
//! Measured at a8e926e, before anything here was written:
//!
//! * `f.is_hdl_emit` and `f.is_ghost` have **zero readers** anywhere in
//!   `src/`, `tests/` or `crates/` outside `parser.rs` and `ast.rs`. The same
//!   function with and without either attribute emits byte-identical LLVM IR
//!   and a byte-identical `--emit-cpu` blob.
//! * `@hdl_emit` names an HDL backend that **does not exist**, so there is
//!   nothing for it to lower to, ever.
//! * `@ghost` is real at a different SYNTACTIC SITE: `@ghost { .. }` and
//!   `@ghost let ..` are lowered by six backends, while the function attribute
//!   parses and is dropped. That is `feedback-guards-consulted-at-one-site`
//!   read across syntax rather than across call sites, and it is the worse
//!   direction — a user who has used the block form successfully will
//!   reasonably expect the function form to work.
//!
//! **THE OBVIOUS PREDICATE IS A NULL METRIC AND WAS TRIED FIRST.** Diffing the
//! emitted artifact with and without each function-level directive reports
//! SEVEN of eight as inert — `@safe`, `@unsafe`, `@zk_safe` and
//! `@zk_allow_unconstrained` included. Those are CHECKERS: they change what is
//! REFUSED, not what is emitted, so on a program with nothing to check they
//! legitimately change nothing. The probe program was the dead component. The
//! consumer count is what separates a checker with nothing to say from a flag
//! nobody reads.
//!
//! This is the `@zk_target(scheme = "plonkish")` shape — a directive a user can
//! select that changes nothing, with a clean compile and no sign the annotation
//! was discarded — and it gets the same treatment: refused by name.
//!
//! **`@ptx_emit` is deliberately NOT refused and that is the interesting
//! boundary.** It is also inert for an ordinary program (measured
//! byte-identical), but it names a backend that EXISTS and it has a consumer:
//! the LLVM backend refuses a `chisel` block inside it. Refusing one and not
//! the other is not the one-site bug — they differ in whether there is
//! anything to lower to — so the census gate below is written to check the
//! CONSUMER, which `@ptx_emit` has and the other two now have via their
//! refusal.
//!
//! **What is deliberately not claimed: a `@ghost` BLOCK is emitted in full.**
//! `a_ghost_block_is_emitted_not_stripped` compiles, links and RUNS one and
//! measures 7 where a stripped block gives 0, and `tests/ghost_test.ysu` calls
//! `print_int` inside one, so ghost code performs I/O. Stripping it is a
//! feature — it would first have to refuse a ghost block that writes non-ghost
//! state, which is exactly what that probe does — not a typo. §9.13 of the
//! language reference used to claim ghost code was "completely stripped out by
//! codegen and cost zero execution cycles"; that claim is corrected, and the
//! test below pins the correction against the running program.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(repo().join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// Compiles `source` with `flag` in a per-test scratch directory.
///
/// The `tag` is in the SIGNATURE rather than left to the caller's memory: a
/// pid-only temp directory shared by two tests in one binary is a race this
/// repository has now hit six times, most recently in a file written the same
/// afternoon.
fn compile(tag: &str, source: &str, flag: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("y_directives_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("p.ysu");
    std::fs::write(&src, source).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg(flag)
        .current_dir(repo())
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

const PLAIN: &str = "fn tick(a: I32) -> I32 {\n    let b: I32 = a + 1;\n    return b;\n}\n\nfn main() -> I32 {\n    return tick(41);\n}\n";

// ---------------------------------------------------------------------------
// 1. The census. This is the generalisable half: it catches the NEXT one.
// ---------------------------------------------------------------------------

/// Every `is_*` attribute on `FuncDecl` must be read by something outside the
/// front end — by a backend, by the type checker, or by a refusal.
///
/// A refusal counts as a consumer, and that is the point: "honouring it and
/// refusing it are both fine; exiting 0 in silence is not". `is_hdl_emit` and
/// `is_ghost` pass this gate now because `refuse_inert_attributes` reads them.
///
/// The floor counts fields EXAMINED, not fields found in the file: a parse that
/// silently recovers nothing reports "no unread attributes" perfectly.
#[test]
fn every_funcdecl_attribute_has_a_consumer() {
    let ast = read("src/ast.rs");
    let start = ast
        .find("pub struct FuncDecl {")
        .expect("FuncDecl struct not found in src/ast.rs");
    let body = &ast[start..start + ast[start..].find('}').expect("unterminated FuncDecl")];

    let fields: Vec<String> = body
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            let name = l.strip_prefix("pub ")?.split(':').next()?.trim();
            if name.starts_with("is_") {
                Some(name.to_string())
            } else {
                None
            }
        })
        .collect();

    assert!(
        fields.len() >= 6,
        "expected at least 6 `is_*` attributes on FuncDecl, recovered {:?} — \
         the struct was probably not parsed",
        fields
    );

    // Every compiler source EXCEPT the two that merely construct the struct.
    let mut haystack = String::new();
    let mut scanned = 0usize;
    for dir in ["src", "crates"] {
        let mut stack = vec![repo().join(dir)];
        while let Some(p) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&p) else { continue };
            for e in rd.flatten() {
                let path = e.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|x| x == "rs") {
                    let name = path.file_name().unwrap().to_string_lossy().to_string();
                    if name == "parser.rs" || name == "ast.rs" {
                        continue;
                    }
                    haystack.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
                    scanned += 1;
                }
            }
        }
    }
    assert!(scanned > 20, "only {scanned} compiler sources scanned");

    // The exclusion must actually be in force. Scanning `parser.rs` would make
    // every flag look consumed by its own CONSTRUCTION, which is the whole
    // distinction this gate draws - and it is invisible while the refusals are
    // in place, so it is asserted directly rather than left to a mutation.
    assert!(
        !haystack.contains("fn parse_func_decl"),
        "parser.rs leaked into the scan: constructing a flag is not consuming it"
    );
    assert!(
        !haystack.contains("pub struct FuncDecl {"),
        "ast.rs leaked into the scan: declaring a flag is not consuming it"
    );

    let unread = |names: &[String]| -> Vec<String> {
        names
            .iter()
            .filter(|f| !haystack.contains(&format!(".{f}")))
            .cloned()
            .collect()
    };

    // The detector must be able to detect. With the refusals in place there is
    // genuinely nothing to find, so an empty answer is exactly what a census
    // that computes nothing also reports - `feedback-null-metrics-pass-dead-
    // components`, and it survived the first mutation sweep of this very file.
    let canary = "is_a_field_that_nothing_anywhere_reads".to_string();
    assert_eq!(
        unread(std::slice::from_ref(&canary)),
        vec![canary.clone()],
        "the census reported a field that provably has no consumer as consumed - \
         it is not detecting anything, so its empty answer below means nothing"
    );

    let unread: Vec<String> = unread(&fields);

    assert!(
        unread.is_empty(),
        "these `FuncDecl` attributes reach the AST and are read by NOTHING \
         outside parser.rs/ast.rs: {unread:?}.\n\
         A directive a user can write that changes nothing is the \
         `@zk_target(scheme = \"plonkish\")` shape. Consume it, or refuse it by \
         name in `type_checker::refuse_inert_attributes` — a refusal counts.\n\
         (Examined {} attributes across {scanned} sources.)",
        fields.len()
    );
}

// ---------------------------------------------------------------------------
// 2. The two refusals, behaviourally, at every backend.
// ---------------------------------------------------------------------------

/// `@hdl_emit` is refused whatever backend was asked for.
///
/// The refusal lives in `check_func`, which `Item::Func`, `Item::Impl`'s
/// methods and `Item::Module`'s recursion all route through — one choke point
/// rather than three, which is what stops this being the one-site bug.
#[test]
fn hdl_emit_is_refused_by_every_backend() {
    let src = format!("@hdl_emit\n{PLAIN}");
    let mut checked = 0usize;
    for flag in [
        "--emit-llvm",
        "--emit-cpu",
        "--emit-native",
        "--emit-ptx",
        "--emit-coprocessor",
    ] {
        let (ok, text) = compile(&format!("hdl{}", flag.trim_start_matches('-')), &src, flag);
        assert!(!ok, "{flag} accepted `@hdl_emit`:\n{text}");
        assert!(
            text.contains("@hdl_emit") && text.contains("no HDL backend"),
            "{flag} refused, but not by name — a user cannot act on this:\n{text}"
        );
        checked += 1;
    }
    assert_eq!(checked, 5, "backend sweep did not run");
}

/// The refusal reaches a method in an `impl` and a function in a `module`.
///
/// Both go through `check_func`; asserting it rather than reading it is what
/// this repository means by enumerating the SITES.
#[test]
fn the_refusal_reaches_nested_functions() {
    let cases = [
        (
            "impl",
            "struct S { x: I32 }\nimpl S {\n    @hdl_emit\n    fn m(v: I32) -> I32 { return v; }\n}\nfn main() -> I32 { return 0; }\n",
        ),
        (
            "module",
            "module inner {\n    @hdl_emit\n    fn m(v: I32) -> I32 { return v; }\n}\nfn main() -> I32 { return 0; }\n",
        ),
    ];
    for (tag, src) in cases {
        let (ok, text) = compile(&format!("nest{tag}"), src, "--emit-llvm");
        assert!(!ok, "`@hdl_emit` inside a {tag} was accepted:\n{text}");
        assert!(
            text.contains("@hdl_emit"),
            "refused inside a {tag}, but not by name:\n{text}"
        );
    }
}

/// `@ghost` on a function is refused AND the block and variable forms still
/// compile.
///
/// The positive half is what stops "refuse anything spelled `@ghost`" passing:
/// that would be sound, would satisfy every assertion about the refusal, and
/// would delete two working forms — the `ordinary_loop_bodies_still_verify`
/// shape.
#[test]
fn ghost_on_a_function_is_refused_and_the_other_forms_still_work() {
    let (ok, text) = compile(
        "ghostfn",
        "@ghost\nfn spec(a: I32) -> I32 { return a; }\nfn main() -> I32 { return spec(1); }\n",
        "--emit-llvm",
    );
    assert!(!ok, "`@ghost` on a function was accepted:\n{text}");
    assert!(
        text.contains("@ghost") && text.contains("function"),
        "refused, but not by name:\n{text}"
    );

    let (ok, text) = compile(
        "ghostokblock",
        "fn main() -> I32 {\n    let mut acc: I32 = 0;\n    @ghost { acc = acc + 7; }\n    return acc;\n}\n",
        "--emit-llvm",
    );
    assert!(
        ok,
        "the `@ghost` BLOCK form must still compile — it is the one working \
         spelling and six backends lower it:\n{text}"
    );

    // The variable form does not parse, and §9.13's own worked example used it.
    // Found by this test failing on its first run: the positive case was
    // written from the manual, and the manual was wrong. Pinned so that
    // implementing it is a deliberate act rather than a silent drift.
    let (ok, text) = compile(
        "ghostlet",
        "fn main() -> I32 {\n    @ghost let mut s: I32 = 3;\n    return s;\n}\n",
        "--emit-llvm",
    );
    assert!(!ok, "`@ghost let` now parses — update §9.13, which says it does not");
    assert!(
        text.contains("Expected '{' to begin block"),
        "`@ghost let` must still fail as a block-header syntax error:\n{text}"
    );
}

// ---------------------------------------------------------------------------
// 3. The doc claim, tied to a running program.
// ---------------------------------------------------------------------------

/// A `@ghost` block is EMITTED. §9.13 used to say it was "completely stripped
/// out by codegen and cost zero execution cycles".
///
/// Measured by running it: 7 against a stripped block's 0. Both arms are built
/// and run, because "the ghost arm returns 7" is also what a compiler that had
/// deleted the `@ghost` syntax entirely would report — the control is what
/// makes the pair mean something.
///
/// Skipped when `clang` is absent. The skip guard is `clang --version`, which
/// is not computed by anything under test.
#[test]
fn a_ghost_block_is_emitted_not_stripped() {
    let doc = read("docs/y_language_documentation.md");
    let ghost = doc
        .split("### 9.13")
        .nth(1)
        .expect("§9.13 not found")
        .split("### 9.14")
        .next()
        .unwrap()
        .to_string();
    assert!(
        !ghost.contains("completely stripped out by codegen"),
        "§9.13 claims `@ghost` code is stripped. It is emitted; see below."
    );
    assert!(
        ghost.contains("NOT stripped") || ghost.contains("not stripped"),
        "§9.13 must say what was measured — that a `@ghost` block is emitted"
    );

    if Command::new("clang").arg("--version").output().is_err() {
        eprintln!("SKIP: no clang, cannot run the emitted program");
        return;
    }

    let dir = std::env::temp_dir().join(format!("y_ghostrun_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("drv.c"),
        "#include <stdio.h>\nint ysu_main(void);\nint main(void){printf(\"%d\\n\", ysu_main());return 0;}\n",
    )
    .unwrap();

    // Same program twice: once with the ghost block, once without it.
    let arms = [
        ("with", "fn main() -> I32 {\n    let mut acc: I32 = 0;\n    @ghost { acc = acc + 7; }\n    return acc;\n}\n", 7),
        ("without", "fn main() -> I32 {\n    let mut acc: I32 = 0;\n    return acc;\n}\n", 0),
    ];
    for (tag, src, want) in arms {
        let ys = dir.join(format!("{tag}.ysu"));
        std::fs::write(&ys, src).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&ys)
            .arg("--emit-llvm")
            .current_dir(repo())
            .output()
            .expect("run Y");
        assert!(
            out.status.success(),
            "compiling the {tag} arm failed:\n{}",
            String::from_utf8_lossy(&out.stdout)
        );
        let ll = dir.join(format!("{tag}.ll"));
        let bin = dir.join(format!("{tag}.bin"));
        let cc = Command::new("clang")
            .args(["-O0", "-Wno-override-module"])
            .arg(&ll)
            .arg(dir.join("drv.c"))
            .arg("-o")
            .arg(&bin)
            .output()
            .expect("run clang");
        assert!(
            cc.status.success(),
            "clang rejected the {tag} arm:\n{}",
            String::from_utf8_lossy(&cc.stderr)
        );
        let run = Command::new(&bin).output().expect("run the program");
        let got: i32 = String::from_utf8_lossy(&run.stdout).trim().parse().unwrap();
        assert_eq!(
            got, want,
            "the `{tag}` arm returned {got}, expected {want} — a `@ghost` block \
             is lowered like any other block and costs the cycles its statements \
             cost"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. The fossil's signature.
// ---------------------------------------------------------------------------

fn headers(text: &str) -> std::collections::BTreeSet<String> {
    text.lines()
        .filter(|l| l.starts_with('#'))
        .map(|l| l.trim_end().to_string())
        .filter(|l| l.len() > 2)
        .collect()
}

/// A documentation file a gate reads must not have a second copy elsewhere.
///
/// A second copy of the language reference lived under `tests/` — deliberately
/// named here WITHOUT a resolvable path, because this increment deletes it and
/// `every_path_a_proof_or_a_gate_cites_exists` is right to refuse a docstring
/// that cites a deleted file; do not helpfully restore the citation. It was a
/// 3,129-line fossil of `docs/y_language_documentation.md` (4,726 lines), read
/// by none of the seven gates that point at the `docs/` one, last touched
/// before the whole correction series — so it still advertised the **withdrawn and unsound**
/// `@max_iterations` active-mask lowering, three verification scripts that do
/// not exist, a `Y.toml` manifest whose real name is `Ysu.toml`, and two `ypm`
/// subcommands that answer `Unknown command`.
///
/// **The predicate is header overlap, not the file name, and the name was tried
/// first and rejected on measurement.** Three `README.md` files are tracked
/// here and they are a convention rather than copies: pairwise they share
/// **zero** section headers. The fossil shared **170 of its 185** with the file
/// it was a copy of — 0.92. The 0.5 threshold sits in the gap between those,
/// near neither edge.
#[test]
fn no_gated_document_has_a_second_copy() {
    // Every .md path any test reads.
    let mut gated: Vec<String> = Vec::new();
    for e in std::fs::read_dir(repo().join("tests")).unwrap().flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "rs") {
            let s = std::fs::read_to_string(&p).unwrap_or_default();
            let mut rest = s.as_str();
            while let Some(i) = rest.find(".md\"") {
                let head = &rest[..i];
                if let Some(q) = head.rfind('"') {
                    let cand = &head[q + 1..];
                    if !cand.contains(char::is_whitespace) && !cand.is_empty() {
                        gated.push(format!("{cand}.md"));
                    }
                }
                rest = &rest[i + 4..];
            }
        }
    }
    gated.sort();
    gated.dedup();
    assert!(
        gated.len() >= 2,
        "recovered only {gated:?} gated documents — the scan is not working"
    );

    // Every .md in the working tree. Walked rather than taken from
    // `git ls-files`: that lists only TRACKED files, so a copy added and not
    // yet committed would be invisible - which is how the first version of
    // this gate reported a restored fossil as clean. `target/` and `.git/` are
    // excluded because a build tree legitimately contains vendored copies of
    // other people's documents.
    let mut all: Vec<String> = Vec::new();
    let mut stack = vec![repo()];
    while let Some(p) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&p) else { continue };
        for e in rd.flatten() {
            let path = e.path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if path.is_dir() {
                if name != "target" && name != ".git" && name != "node_modules" {
                    stack.push(path);
                }
            } else if path.extension().is_some_and(|x| x == "md") {
                if let Ok(rel) = path.strip_prefix(repo()) {
                    all.push(rel.to_string_lossy().to_string());
                }
            }
        }
    }
    assert!(all.len() > 5, "walk found only {} .md files", all.len());

    let mut compared = 0usize;
    let mut copies: Vec<String> = Vec::new();
    for g in &gated {
        let Ok(gtext) = std::fs::read_to_string(repo().join(g)) else { continue };
        let gh = headers(&gtext);
        if gh.len() < 5 {
            continue;
        }
        for other in &all {
            if other == g {
                continue;
            }
            let Ok(otext) = std::fs::read_to_string(repo().join(other)) else { continue };
            let oh = headers(&otext);
            if oh.len() < 5 {
                continue;
            }
            compared += 1;
            let shared = gh.intersection(&oh).count();
            let overlap = shared as f64 / gh.len().min(oh.len()) as f64;
            if overlap > 0.5 {
                copies.push(format!(
                    "{other} shares {shared} of {} section headers with the gated {g} \
                     (overlap {overlap:.2})",
                    gh.len().min(oh.len())
                ));
            }
        }
    }
    assert!(
        compared > 0,
        "no document pair was compared — the sweep did no work"
    );
    assert!(
        copies.is_empty(),
        "a gated document has a second copy that no gate reads, which is how a \
         correction lands in one of them and not the other:\n  {}\n\
         Delete the copy, or point the gates at both.\n\
         (Compared {compared} pairs.)",
        copies.join("\n  ")
    );
}
