//! What `--emit-cpu` prints must be Rust that compiles.
//!
//! This backend's entire deliverable is source for a human to paste — Y never
//! compiles it — so nothing checked that the text was even syntactically
//! valid Rust. `tests/hello.ysu`, the program the README uses to demonstrate
//! the flag, produced **six compile errors**:
//!
//!   * `println("FizzBuzz")` — a CALL to Rust's `println` MACRO;
//!   * `print_int(x)` against no definition anywhere in the blob;
//!   * a safe `main` calling an `unsafe fn fizzbuzz` with no `unsafe` block.
//!
//! And across `tests/*.ysu`, **31 of 85 programs** emitted a blob rustc
//! rejects, because GPU intrinsics (`block_idx_x`, `thread_idx_x`,
//! `block_ptr2d_load_v4`, the carry-chain family) were transcribed verbatim
//! into Rust that references nothing. Those are refused by name now.
//!
//! The gate is rustc itself: it is the only thing that can tell "this is Rust"
//! from "this looks like Rust", which is the same reason the PTX backend is
//! gated on `ptxas` rather than on substring assertions.
//!
//! **And for a while it was rustc applied to something else.** The extractor
//! deleted the blob's own `use crate::avx_wrapper::*;` before compiling it, so
//! what passed was a MODIFIED artifact. As delivered every blob failed with
//! `error[E0432]: unresolved import`, because `crate::` names the Y compiler's
//! own crate and the reader is pasting into theirs. Stripping did not rescue
//! it either - it only moved the failure, to `error[E0433]: cannot find type
//! Y256f32` on the one construct that needed the import. Both readings were
//! measured. Nothing is stripped now, and the blob is self-contained.
//!
//! Run with:  cargo test --test cpu_emitter_output_compiles

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static SALT: AtomicUsize = AtomicUsize::new(0);

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Emit {
    ok: bool,
    text: String,
    blob: Option<String>,
}

fn emit_cpu(src_path: &Path) -> Emit {
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(src_path)
        .arg("--emit-cpu")
        .current_dir(repo())
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // Line-wise, not `split_once`: the blob's own banner comment is a row of
    // `=` too, so splitting on the closing marker as a substring truncates it
    // to two characters. (Which it did, and the test caught it by asserting on
    // a specific missing definition rather than just "rustc was unhappy".)
    const OPEN: &str = "======= GENERATED RUST BLOB =======";
    const CLOSE: &str = "=======================================";
    let mut body: Vec<&str> = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        let t = line.trim();
        if !inside {
            if t == OPEN {
                inside = true;
            }
            continue;
        }
        if t == CLOSE {
            inside = false;
            break;
        }
        // NOTHING IS STRIPPED. This used to drop `use crate::avx_wrapper::*;`
        // on the grounds that "the blob is written to be pasted INTO this
        // crate" - which contradicted the README, where `--emit-cpu` prints
        // source "for you to paste". The strip is what made the
        // contradiction invisible: measured, ALL 46 corpus blobs carried that
        // import, NONE referenced a symbol from it, and every one of them
        // failed as delivered with `error[E0432]: unresolved import
        // crate::avx_wrapper`. So this gate was certifying an artifact the
        // compiler does not emit - the same defect as the Python harnesses
        // that stripped the PTX header and substituted their own.
        body.push(line);
    }
    let blob = if body.is_empty() { None } else { Some(body.join("\n")) };
    Emit { ok: out.status.success(), text, blob }
}

/// `rustc` accepts the blob, or the reason it does not.
fn rustc_check(blob: &str) -> Result<(), String> {
    let n = SALT.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("y_cpurs_{}_{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("blob.rs");
    std::fs::write(&f, blob).unwrap();

    let out = Command::new("rustc")
        .arg("--edition")
        .arg("2021")
        .arg("--crate-type")
        .arg("lib")
        .arg("-A")
        .arg("warnings")
        .arg("--out-dir")
        .arg(&dir)
        .arg(&f)
        .output()
        .expect("run rustc");
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    let _ = std::fs::remove_dir_all(&dir);
    if out.status.success() {
        Ok(())
    } else {
        Err(err)
    }
}

/// The README's own demonstration program must produce compiling Rust.
#[test]
fn the_documented_example_emits_valid_rust() {
    let src = repo().join("tests").join("hello.ysu");
    let e = emit_cpu(&src);
    assert!(e.ok, "--emit-cpu failed on hello.ysu:\n{}", e.text);
    let blob = e.blob.expect("no blob in the output");

    // Named individually: each was its own bug, and asserting only "rustc is
    // happy" would not say which came back.
    assert!(
        blob.contains("fn println"),
        "`println` is called but the prelude does not define it, so the call \
         resolves to Rust's MACRO of that name and rustc rejects it:\n{}",
        blob
    );
    assert!(
        blob.contains("fn print_int"),
        "`print_int` is called but the prelude does not define it:\n{}",
        blob
    );

    if let Err(e) = rustc_check(&blob) {
        panic!("the blob --emit-cpu prints does not compile:\n{}", e);
    }
}

/// Across the corpus: every blob that is emitted at all must compile.
///
/// The count matters as much as the property. A version of this backend that
/// refused everything would satisfy "no invalid blob" perfectly, so the floor
/// is asserted too — the same control `the_corpus_is_not_all_skips` provides
/// for the cross-backend differential.
#[test]
fn no_emitted_blob_is_invalid_rust() {
    let dir = repo().join("tests");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "ysu").unwrap_or(false))
        .collect();
    sources.sort();
    assert!(sources.len() > 50, "corpus shrank to {} files", sources.len());

    let mut emitted = 0usize;
    let mut refused = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for src in &sources {
        let e = emit_cpu(src);
        let Some(blob) = e.blob else {
            refused += 1;
            continue;
        };
        emitted += 1;
        if let Err(err) = rustc_check(&blob) {
            let first = err
                .lines()
                .find(|l| l.starts_with("error"))
                .unwrap_or("(no error line)");
            bad.push(format!(
                "  {}: {}",
                src.file_name().unwrap().to_string_lossy(),
                first
            ));
        }
    }

    assert!(
        emitted >= 40,
        "only {} of {} programs emitted a blob - the backend is refusing \
         almost everything, so this test proves nothing",
        emitted,
        sources.len()
    );
    // The list is EMPTY, and that is the resolution of what used to sit here.
    //
    // `math.ysu` did `@unsafe fn matrix_init() { let mut mem_ptr: I32 = 0;
    // *mem_ptr = 0; }` - dereferencing an integer. `type_checker`'s
    // `UnaryOp::Deref` arm returned `Unknown` for anything that was not a
    // `Reference`, so the front end accepted it and the LLVM backend emitted
    // `store i32 0, ptr %_t1` with `%_t1` an `i32`: invalid IR under a green
    // banner. This comment recorded "whether Y should type that as an error or
    // lower it through `inttoptr` is a language decision".
    //
    // It is typed as an error. Y has no raw pointer type - only `&T` - so
    // there is no pointee type for `inttoptr` to take a width from, and
    // guessing one is the substitution the design rule forbids. The front end
    // refuses `*x` on anything it can positively identify as a non-reference
    // and names the type; `math.ysu` uses a `&mut I32` parameter now.
    //
    // Pinned as an EXACT set: a new failure fails this test, and so does
    // fixing an entry without removing it. A silently growing allowlist is how
    // a refusal baseline becomes a list of unexamined bugs.
    const KNOWN_FRONT_END_GAPS: &[&str] = &[];

    let unexpected: Vec<&String> = bad
        .iter()
        .filter(|b| !KNOWN_FRONT_END_GAPS.iter().any(|k| b.contains(k)))
        .collect();
    assert!(
        unexpected.is_empty(),
        "{} of {} emitted blobs do not compile (refused: {}):\n{}",
        unexpected.len(),
        emitted,
        refused,
        unexpected.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
    );
    for known in KNOWN_FRONT_END_GAPS {
        assert!(
            bad.iter().any(|b| b.contains(known)),
            "{} is listed as a known front-end gap but its blob compiles now - \
             remove it from KNOWN_FRONT_END_GAPS",
            known
        );
    }
}

/// A GPU intrinsic must be refused by name, not transcribed.
#[test]
fn a_gpu_intrinsic_is_refused_rather_than_transcribed() {
    let dir = std::env::temp_dir().join(format!("y_cpugpu_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("k.ysu");
    std::fs::write(
        &src,
        "fn main() -> I32 {\n    let t: I32 = thread_idx_x();\n    return t;\n}\n",
    )
    .unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-cpu")
        .current_dir(repo())
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);

    assert!(!out.status.success(), "exited 0 on a GPU intrinsic:\n{}", text);
    assert!(
        text.contains("thread_idx_x"),
        "refused without naming the intrinsic:\n{}",
        text
    );
    assert!(
        text.contains("CPU host backend"),
        "refused without saying which backend:\n{}",
        text
    );
}

/// Refuse a GPU intrinsic by name, given a tagged temp dir of its own.
///
/// The tag is in the signature rather than in a comment asking the caller to
/// remember: two tests sharing a temp-dir name is a race this repository has
/// hit five times.
fn refusal_of(tag: &str, source: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("y_cpuref_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("k.ysu");
    std::fs::write(&src, source).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-cpu")
        .current_dir(repo())
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), text)
}

/// No blob may import a path that only resolves inside the compiler's own
/// crate.
///
/// This gates the DEFECT'S SIGNATURE rather than the one instance of it, so
/// the next one needs no prophet. `use crate::…` in text the reader pastes
/// into their own project cannot resolve by construction, whatever it names;
/// `crate` is the pasting crate, and that is never this one.
///
/// The floor counts blobs actually EMITTED AND SCANNED, not `.ysu` files
/// found: a sweep that compiled nothing would report "no bad imports"
/// perfectly.
#[test]
fn no_emitted_blob_imports_a_path_only_the_compiler_can_resolve() {
    let mut scanned = 0usize;
    let mut offenders: Vec<String> = Vec::new();

    let mut sources: Vec<PathBuf> = std::fs::read_dir(repo().join("tests"))
        .expect("tests dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "ysu"))
        .collect();
    sources.sort();

    for src in &sources {
        let e = emit_cpu(src);
        if e.blob.is_none() {
            continue;
        }
        // THE RAW OUTPUT, not `e.blob`. The extractor is in this file and is
        // exactly what hid this defect before, by filtering the offending line
        // out on its way past. A gate that reads the extractor's output cannot
        // see a strip being re-added; mutation confirmed it - B1+B4 defeats
        // both this and the compile gate when it reads `e.blob`.
        scanned += 1;
        for (i, line) in e.text.lines().enumerate() {
            let t = line.trim();
            if t.starts_with("use crate::") || t.starts_with("pub use crate::") {
                offenders.push(format!(
                    "{}:{} {}",
                    src.file_name().unwrap().to_string_lossy(),
                    i + 1,
                    t
                ));
            }
        }
    }

    assert!(
        scanned >= 40,
        "only {scanned} blobs were emitted and scanned; this sweep is supposed \
         to cover the whole corpus, and a sweep that compiles nothing reports \
         no offenders perfectly"
    );
    assert!(
        offenders.is_empty(),
        "{} blob line(s) import a path that resolves only inside the Y crate, \
         so the text the user is told to paste cannot compile for them \
         (error[E0432]):\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

/// `Fragment::zero` is refused, and it is the fifth member of a family the
/// original sweep left at four.
///
/// A warp-cooperative matrix fragment rendered as eight lanes of host f32 is
/// the same substitution as `ldmatrix(p) -> Y256f32::load_aligned_ptr`, which
/// that sweep DID refuse - it sat one match arm away. It was also the only
/// consumer of the blob's `crate::avx_wrapper` import.
#[test]
fn a_matrix_fragment_has_no_host_equivalent() {
    let (ok, text) = refusal_of(
        "frag",
        "kernel k(A: GlobalMemory<F32>) {\n    let acc = Fragment::zero();\n    return;\n}\nfn main() { return; }\n",
    );
    assert!(!ok, "exited 0 on Fragment::zero:\n{text}");
    assert!(
        text.contains("Fragment::zero"),
        "refused without naming the construct:\n{text}"
    );
    assert!(
        !text.contains("GENERATED RUST BLOB"),
        "printed a blob anyway. main() must fail before printing, or the \
         refusal reaches the user's clipboard as a success:\n{text}"
    );
}

/// THE CONTROL. "Refuse everything" satisfies both tests above while deleting
/// the backend.
///
/// It also pins the half that the import removal could have broken: an
/// ordinary host program must still emit a blob, and that blob must still
/// compile VERBATIM - which is the property the strip was hiding.
#[test]
fn an_ordinary_host_program_still_emits_a_blob_that_compiles_verbatim() {
    let dir = std::env::temp_dir().join(format!("y_cpuctl_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("plain.ysu");
    std::fs::write(
        &src,
        "fn add(a: I32, b: I32) -> I32 { return a + b; }\nfn main() { let x: I32 = add(2, 3); return; }\n",
    )
    .unwrap();

    let e = emit_cpu(&src);
    let _ = std::fs::remove_dir_all(&dir);

    assert!(e.ok, "an ordinary host program was refused:\n{}", e.text);
    let blob = e.blob.expect("an ordinary host program must emit a blob");
    assert!(
        blob.contains("fn add"),
        "the blob does not contain the program:\n{blob}"
    );
    if let Err(msg) = rustc_check(&blob) {
        panic!("the blob does not compile as emitted:\n{msg}");
    }
}
