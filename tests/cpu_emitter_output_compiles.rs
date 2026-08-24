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
    const OPEN: &str = "======= GENERATED RUST/AVX BLOB =======";
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
        // The blob is written to be pasted INTO this crate, so its
        // `use crate::avx_wrapper::*;` cannot resolve standalone.
        if t.starts_with("use crate::avx_wrapper") {
            continue;
        }
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
    // One known failure, and it is NOT a bug in this backend. `math.ysu` does
    //
    //     @unsafe fn matrix_init() { let mut mem_ptr: I32 = 0; *mem_ptr = 0; }
    //
    // - dereferencing an integer. `type_checker`'s `UnaryOp::Deref` arm
    // returns `Unknown` for anything that is not a `Reference`, deliberately
    // and with the reasoning written beside it, so the front end accepts it;
    // the LLVM backend then emits `store i32 0, ptr %_t1` where `%_t1` is an
    // `i32`, which is invalid IR that `clang` rejects downstream. Whether Y
    // should type that as an error or lower it through `inttoptr` is a
    // language decision, not a transcription bug, so it is named here rather
    // than papered over.
    //
    // Pinned as an EXACT set: a new failure fails this test, and so does
    // fixing this one without updating the list. A silently growing allowlist
    // is how a refusal baseline becomes a list of unexamined bugs.
    const KNOWN_FRONT_END_GAPS: &[&str] = &["math.ysu"];

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
