//! `chisel {}` blocks and the register names inside them.
//!
//! §16 of `docs/y_language_documentation.md` is a full user reference for
//! `chisel {}`, and §16.2 documented a naming convention — "Y variables
//! declared before the `chisel` block are accessible using their PTX register
//! names", with a table saying `let x: F32` is `%x`.
//!
//! **The convention did not exist.** The line was written to the module
//! verbatim, and this backend's allocator names registers `%f0`/`%r0`/`%rd0`,
//! never after the source variable. Measured before the fix, on §16.2's own
//! example:
//!
//! ```text
//!     // --- CHISEL INLINE PTX ---
//!     mul.f32 %result, %val, %val;
//! ```
//!
//! `Compilation Successful!`, exit 0, and `ptxas` answers
//! `Unknown symbol '%result'`. Every worked example in §16.4 had the same
//! defect, and **no `.ysu` in this repository uses `chisel` at all**, which is
//! why nothing noticed — the `SmemLayout` profile: a documented surface that
//! no test exercises. It is also the `Expr::Ident` hole one layer over (an
//! unbound name spliced into instruction text) reached through a *string*
//! rather than through the AST, which is why the fix there did not cover it.
//!
//! What is pinned here:
//!
//! * a Y variable resolves to the register the allocator gave it, so §16.2 is
//!   now a true statement about the compiler;
//! * a PTX special register (`%tid.x`, `%globaltimer`) passes through;
//! * a name that is neither is **refused** with a line and column, rather than
//!   handed to `ptxas` on whichever machine runs the kernel;
//! * a `chisel` block naming nothing still compiles — the control that stops
//!   "refuse every `chisel` block" satisfying the rest of this file;
//! * the LLVM backend refuses `chisel` inside `@ptx_emit` (it would put PTX
//!   into an `x86_64` module) while still lowering an ordinary host `chisel`.
//!
//! The `ptxas` assertions are the load-bearing half: a substring check on the
//! emitted text says the register was substituted, and only the assembler says
//! the result is a legal module.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn compiler() -> PathBuf {
    repo_root().join("target/release/Y")
}

/// A per-test scratch directory. The tag is in the SIGNATURE rather than left
/// to the caller's memory: two tests sharing one temp directory is a race this
/// repository has hit five times, most recently in a helper whose only
/// interpolated value was the pid.
fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("y_chisel_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Emitted {
    ok: bool,
    stdout: String,
    ptx: Option<String>,
    ptx_path: PathBuf,
}

/// Compiles `source` with `--emit-ptx` in its own directory. The source is
/// written into the scratch directory rather than into `tests/`, because
/// `--emit-ptx` writes its artifact next to the input and a gate that emits
/// into the repository rewrites committed files and races anything else
/// compiling the same fixture.
fn emit_ptx(tag: &str, source: &str) -> Emitted {
    let dir = scratch(tag);
    let src = dir.join("probe.ysu");
    std::fs::write(&src, source).unwrap();
    let out = Command::new(compiler())
        .arg(&src)
        .arg("--emit-ptx")
        // `current_dir` stays the repo so `.ysu_hw_profile` is found; the
        // artifact still lands beside the source, in the scratch directory.
        .current_dir(repo_root())
        .output()
        .expect("run Y");
    let ptx_path = dir.join("probe.ptx");
    Emitted {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr),
        ptx: std::fs::read_to_string(&ptx_path).ok(),
        ptx_path,
    }
}

fn have_ptxas() -> bool {
    Command::new("ptxas")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Runs the real assembler. Returns `None` only when `ptxas` is absent —
/// distinguished from "ptxas rejected it", which is a failure and never a skip.
fn ptxas_accepts(path: &Path) -> Option<Result<(), String>> {
    if !have_ptxas() {
        return None;
    }
    let out = Command::new("ptxas")
        .arg("-arch=sm_89")
        .arg(path)
        .arg("-o")
        .arg(path.with_extension("cubin"))
        .output()
        .expect("run ptxas");
    if out.status.success() {
        Some(Ok(()))
    } else {
        Some(Err(String::from_utf8_lossy(&out.stderr).to_string()))
    }
}

/// The line the emitter writes for a `chisel` block, without the marker.
fn chisel_lines(ptx: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut inside = false;
    for line in ptx.lines() {
        if line.contains("--- CHISEL INLINE PTX ---") {
            inside = true;
            continue;
        }
        if inside {
            let t = line.trim();
            if t.contains("--- END CHISEL PTX ---") {
                return out;
            }
            out.push(t.to_string());
        }
    }
    // An opening marker with no closing one means the emitter changed shape
    // under this gate. Returning what was collected would silently include
    // whatever followed the block.
    panic!("a CHISEL block was opened and never closed in the emitted PTX");
}

// ── The documented convention ───────────────────────────────────────────

/// §16.2's own worked example, verbatim from the reference.
const DOC_16_2: &str = r#"
kernel k(Out: GlobalMemory<F32>) {
    let val: F32 = 1.0;
    let result: F32 = 0.0;
    chisel {
        "mul.f32 %result, %val, %val;";
    }
    block_ptr2d_store(Out, 0, 0, 1, 1, 1, result);
}
"#;

#[test]
fn the_documented_naming_convention_resolves_to_the_allocated_register() {
    let e = emit_ptx("doc162", DOC_16_2);
    assert!(e.ok, "§16.2's example must compile:\n{}", e.stdout);
    let ptx = e.ptx.expect("a .ptx was written");
    let lines = chisel_lines(&ptx);
    assert_eq!(lines.len(), 1, "one chisel line expected, got {:?}", lines);
    let line = &lines[0];

    // The Y names must be GONE. This is the whole defect: they used to survive
    // into the module and `ptxas` answered `Unknown symbol`.
    assert!(
        !line.contains("%val") && !line.contains("%result"),
        "the Y variable names survived into the module: {}",
        line
    );
    // ...and be replaced by real f32 registers. The SHAPE is asserted rather
    // than a prefix: `result` and `val` are two distinct variables, so the
    // destination must differ from the two sources and the two sources must be
    // the same register. A resolver that answered with one fixed register for
    // every name satisfies "no %val survives" and "starts with %f" perfectly —
    // it was a survivor of exactly that mutation until this assertion existed,
    // caught only incidentally because the kernel parameter happened to sort
    // first. What is checked here is that the map DISTINGUISHES its inputs.
    let ops: Vec<&str> = line
        .trim_end_matches(';')
        .split_whitespace()
        .skip(1)
        .map(|t| t.trim_end_matches(','))
        .collect();
    assert_eq!(ops.len(), 3, "expected `mul.f32 d, a, b`, got: {}", line);
    for o in &ops {
        assert!(
            o.starts_with("%f"),
            "expected the allocator's f32 registers, got {} in: {}",
            o,
            line
        );
    }
    assert_eq!(ops[1], ops[2], "both sources are `val`, so they must be one register: {}", line);
    assert_ne!(
        ops[0], ops[1],
        "`result` and `val` are different variables and must not share a register: {}",
        line
    );

    match ptxas_accepts(&e.ptx_path) {
        Some(Ok(())) => {}
        Some(Err(err)) => panic!("ptxas rejected §16.2's example:\n{}\n{}", line, err),
        None => eprintln!("NOTE: ptxas absent; the substitution above was still checked"),
    }
}

#[test]
fn a_ptx_special_register_passes_through() {
    let e = emit_ptx(
        "special",
        r#"
kernel k(Out: GlobalMemory<F32>) {
    let t: I32 = 0;
    let g: I64 = 0;
    chisel {
        "mov.u32 %t, %tid.x;";
        "mov.u32 %t, %ctaid.y;";
        "mov.u64 %g, %globaltimer;";
    }
    block_ptr2d_store(Out, 0, 0, 1, 1, 1, 1.0);
}
"#,
    );
    assert!(e.ok, "special registers must compile:\n{}", e.stdout);
    let ptx = e.ptx.expect("a .ptx was written");
    let lines = chisel_lines(&ptx);
    assert_eq!(lines.len(), 3, "three chisel lines expected, got {:?}", lines);

    // Hardware-provided names survive verbatim; the Y variables beside them do
    // not. Both halves matter: substituting a special register would be as
    // wrong as failing to substitute a variable.
    assert!(lines[0].ends_with("%tid.x;"), "{}", lines[0]);
    assert!(lines[1].ends_with("%ctaid.y;"), "{}", lines[1]);
    assert!(lines[2].ends_with("%globaltimer;"), "{}", lines[2]);
    for l in &lines {
        assert!(!l.contains("%t,") || l.contains("%r"), "unsubstituted %t: {}", l);
        assert!(!l.contains("%g,"), "unsubstituted %g: {}", l);
    }

    match ptxas_accepts(&e.ptx_path) {
        Some(Ok(())) => {}
        Some(Err(err)) => panic!("ptxas rejected the special-register block:\n{}", err),
        None => eprintln!("NOTE: ptxas absent"),
    }
}

// ── The refusals ────────────────────────────────────────────────────────

#[test]
fn a_name_that_is_neither_a_variable_nor_a_special_register_is_refused() {
    let e = emit_ptx(
        "typo",
        r#"
kernel k(Out: GlobalMemory<F32>) {
    let val: F32 = 1.0;
    chisel {
        "mul.f32 %vla, %val, %val;";
    }
    block_ptr2d_store(Out, 0, 0, 1, 1, 1, val);
}
"#,
    );
    assert!(
        !e.ok,
        "a misspelled register must be refused, not emitted:\n{}",
        e.stdout
    );
    // The diagnosis must name the offending token and where it is, or the
    // user is sent to read the whole block.
    assert!(
        e.stdout.contains("%vla"),
        "the message must name the token:\n{}",
        e.stdout
    );
    assert!(
        e.stdout.contains("line 5"),
        "the message must carry a line number:\n{}",
        e.stdout
    );
}

#[test]
fn a_vector_variable_is_refused_rather_than_substituted() {
    // A `U32x4` lives in `vec_vars` as FOUR registers under one name. There is
    // no single register `%v` can stand for, so substituting any one of them
    // would be a silent choice — the design rule's shape.
    let e = emit_ptx(
        "vec",
        r#"
kernel k(A: GlobalMemory<U32>, Out: GlobalMemory<U32>) {
    let v: U32x4 = block_ptr2d_load_v4(A, 0, 0, 4, 1, 1);
    chisel {
        "mov.u32 %v, 0;";
    }
    block_ptr2d_store(Out, 0, 0, 1, 1, 1, v.x);
}
"#,
    );
    assert!(!e.ok, "a 4-wide vector name must be refused:\n{}", e.stdout);
    assert!(
        e.stdout.contains("%v") && e.stdout.contains("four registers"),
        "the message must say why a vector cannot be one register:\n{}",
        e.stdout
    );
}

// ── The control ─────────────────────────────────────────────────────────

#[test]
fn a_chisel_block_with_no_variable_references_still_compiles() {
    // Without this, "refuse every chisel block" passes every other test in
    // this file. `bar.sync 0;` names no register at all and is the shape
    // §16.4 documents for an explicit barrier.
    let e = emit_ptx(
        "literal",
        r#"
kernel k(Out: GlobalMemory<F32>) {
    chisel {
        "bar.sync 0;";
    }
    block_ptr2d_store(Out, 0, 0, 1, 1, 1, 1.0);
}
"#,
    );
    assert!(e.ok, "a literal-only chisel block must compile:\n{}", e.stdout);
    let ptx = e.ptx.expect("a .ptx was written");
    assert_eq!(chisel_lines(&ptx), vec!["bar.sync 0;".to_string()]);

    match ptxas_accepts(&e.ptx_path) {
        Some(Ok(())) => {}
        Some(Err(err)) => panic!("ptxas rejected a bare bar.sync:\n{}", err),
        None => eprintln!("NOTE: ptxas absent"),
    }
}

// ── The host backend ────────────────────────────────────────────────────

fn emit_llvm(tag: &str, source: &str) -> (bool, String, Option<String>) {
    let dir = scratch(tag);
    let src = dir.join("probe.ysu");
    std::fs::write(&src, source).unwrap();
    let out = Command::new(compiler())
        .arg(&src)
        .arg("--emit-llvm")
        .current_dir(repo_root())
        .output()
        .expect("run Y");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr),
        std::fs::read_to_string(dir.join("probe.ll")).ok(),
    )
}

#[test]
fn the_llvm_backend_refuses_ptx_chisel_and_still_lowers_host_chisel() {
    // `--emit-llvm` writes `Self::host_triple()` and has no path that emits an
    // `nvptx` triple, so the `@ptx_emit` arm put PTX text into an x86_64
    // module. Measured before the fix: exit 0, and the `clang` line the
    // compiler prints answered `<inline asm>:1:10: error: invalid register
    // name`. The failure was real and arrived one step further out than the
    // compiler, which is what made it survivable.
    let (ok, msg, _) = emit_llvm(
        "hostptx",
        r#"
@ptx_emit
fn main() {
    let val: I32 = 3;
    chisel {
        "mov.u32 %val, 7;";
    }
    print_int(val);
}
"#,
    );
    assert!(!ok, "PTX chisel in a host module must be refused:\n{}", msg);
    assert!(
        msg.contains("@ptx_emit") && msg.contains("--emit-ptx"),
        "the message must name the directive and the backend that does lower it:\n{}",
        msg
    );

    // The control: an ordinary host `chisel` is x86 inline asm and is
    // legitimate. Refusing it too would satisfy the assertion above while
    // deleting a working path.
    let (ok2, msg2, ir) = emit_llvm(
        "hostplain",
        r#"
fn main() {
    let val: I32 = 3;
    chisel {
        "nop;";
    }
    print_int(val);
}
"#,
    );
    assert!(ok2, "an ordinary host chisel block must still compile:\n{}", msg2);
    let ir = ir.expect("a .ll was written");
    assert!(
        ir.contains("CHISEL INLINE ASM") && ir.contains("asm sideeffect"),
        "the host path must still emit inline asm"
    );
}

// ── The documentation ───────────────────────────────────────────────────

#[test]
fn the_reference_documents_the_convention_the_compiler_implements() {
    // §16.2 is the claim this increment made true. A gate on the behaviour
    // alone would let the reference drift back to describing something else,
    // and a reader following §16.2 is exactly who the original bug caught.
    let doc = std::fs::read_to_string(repo_root().join("docs/y_language_documentation.md"))
        .expect("read the language reference");
    let start = doc
        .find("### 16.2 Variable Scoping Inside")
        .expect("§16.2 exists");
    let end = doc[start..]
        .find("### 16.3")
        .map(|i| start + i)
        .expect("§16.3 follows");
    let sec = &doc[start..end];

    assert!(
        sec.contains("refused at compile time"),
        "§16.2 must state that an unresolvable %name is refused, not passed through"
    );
    assert!(
        sec.contains("tests/chisel_register_scope.rs"),
        "§16.2 must name what pins it"
    );
    assert!(
        sec.contains("U32x4"),
        "§16.2 must record that a vector name is refused"
    );
}

#[test]
fn the_faq_names_only_flags_the_cli_recognises() {
    // Three of the five backend flags this answer used (`--llvm`, `--cpu`,
    // `--ptx`) were not options at all, and it listed a C backend that has
    // been removed. The CLI prints its own `Known options:` line for an
    // unrecognised flag, so the gate ASKS THE COMPILER rather than carrying a
    // second copy of the list — a hardcoded list here is the drift, not the
    // fix.
    let out = Command::new(compiler())
        .arg(repo_root().join("tests/hello.ysu"))
        .arg("--this-is-not-a-flag")
        .current_dir(repo_root())
        .output()
        .expect("run Y");
    let text = String::from_utf8_lossy(&out.stdout).to_string()
        + &String::from_utf8_lossy(&out.stderr);
    let line = text
        .lines()
        .find(|l| l.contains("Known options:"))
        .expect("the CLI lists its known options for an unrecognised flag");
    let known: Vec<&str> = line
        .split_whitespace()
        .filter(|t| t.starts_with("--"))
        .collect();
    assert!(known.len() > 10, "the known-option list looks truncated: {}", line);

    let doc = std::fs::read_to_string(repo_root().join("docs/y_language_documentation.md"))
        .expect("read the language reference");
    let start = doc
        .find("**Do I need CUDA drivers and a GPU to use Y?**")
        .expect("the FAQ answer exists");
    let sec = &doc[start..start + 1400];

    // Only the flags presented as usable are checked. The answer also quotes
    // the three broken spellings, deliberately, to say they were broken — so
    // the scan stops at that sentence.
    let usable_end = sec
        .find("Three of the flag spellings")
        .expect("the answer records which spellings were wrong");
    let usable = &sec[..usable_end];

    let mut checked = 0usize;
    for tok in usable.split('`') {
        if !tok.starts_with("--") {
            continue;
        }
        let flag = tok.split_whitespace().next().unwrap_or(tok);
        assert!(
            known.contains(&flag),
            "the FAQ offers `{}`, which the CLI does not recognise. Its own list is: {}",
            flag,
            line
        );
        checked += 1;
    }
    // Counts the flags actually COMPARED, not the ones the answer contains: a
    // scan that finds nothing reports "no bad flags" perfectly.
    assert!(
        checked >= 4,
        "expected the answer to offer several flags, compared only {}",
        checked
    );
}
