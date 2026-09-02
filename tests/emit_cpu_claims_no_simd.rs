//! `--emit-cpu` claimed AVX in six places and emits no SIMD anywhere.
//!
//! The CLI banner said "Emitting CPU AVX-512 Host Code...", the blob's own
//! marker was `======= GENERATED RUST/AVX BLOB =======`, the README and the
//! language reference both called the output "Rust/AVX source", the reference
//! documented an `@avx_emit` directive, and the roadmap described a "shape
//! dispatcher ... that emits hand-written Rust/AVX".
//!
//! Measured 2026-09-02: **0 of 46 emitted blobs contain a vector intrinsic, a
//! vector type, an `x86_64::` path or a `target_feature` attribute.** The
//! emitter contains no vector intrinsic at all. The five-regime shape
//! dispatcher the roadmap cited has exactly one caller and it is a test, so no
//! `.ysu` compilation reaches it. And the crate's only SIMD - an
//! `avx_wrapper` module, deliberately named here WITHOUT a path, because it no
//! longer exists and `every_path_a_proof_or_a_gate_cites_exists` is right to
//! refuse a docstring that cites a deleted file - was referenced by nothing
//! and has been deleted. An audit before deleting it found a safe `pub fn`
//! dereferencing a raw
//! pointer and an entire AVX2 surface calling `_mm256_*` with no
//! `target_feature` and no runtime guard, while `require_avx2()`, whose doc
//! says "call once at start-up", had zero callers.
//!
//! This is the `@zk_target(scheme = "plonkish")` shape: an artifact naming a
//! capability it did not use. It is fail-loud in no direction at all - it
//! misleads silently, which is why nothing caught it.
//!
//! **Half of this can only be checked at the source.** Whether the compiler
//! PRINTS an AVX claim is a property of a string literal; running it on a
//! machine where it emits scalar code says nothing about the banner. Same
//! device as pinning which CPUID reading a predicate uses.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every token that would mean "this output uses SIMD".
const SIMD_MARKERS: &[&str] = &[
    "_mm_", "_mm256", "_mm512", "__m128", "__m256", "__m512", "target_feature",
    "x86_64::", "zmm", "ymm", "Y256f32", "Y512f32", "Y256i32", "Y256f64",
];

/// Words that ADVERTISE SIMD to a user reading a banner or a doc row.
const SIMD_CLAIMS: &[&str] = &["AVX", "avx", "SIMD", "simd", "vector"];

fn emit_cpu_blob(tag: &str, source: &str) -> Option<String> {
    let dir = std::env::temp_dir().join(format!("y_nosimd_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("p.ysu");
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
    blob_of(&text)
}

fn blob_of(text: &str) -> Option<String> {
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
            break;
        }
        body.push(line);
    }
    if body.is_empty() {
        None
    } else {
        Some(body.join("\n"))
    }
}

/// No emitted blob contains SIMD of any kind.
///
/// The floor counts blobs EMITTED AND SCANNED, not `.ysu` files found: a sweep
/// that compiled nothing would report "no SIMD" perfectly.
#[test]
fn no_emitted_blob_contains_simd() {
    let mut sources: Vec<PathBuf> = std::fs::read_dir(repo().join("tests"))
        .expect("tests dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "ysu"))
        .collect();
    sources.sort();

    let mut scanned = 0usize;
    let mut offenders: Vec<String> = Vec::new();
    for src in &sources {
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(src)
            .arg("--emit-cpu")
            .current_dir(repo())
            .output()
            .expect("run Y");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let Some(blob) = blob_of(&text) else { continue };
        scanned += 1;
        for (i, line) in blob.lines().enumerate() {
            for m in SIMD_MARKERS {
                if line.contains(m) {
                    offenders.push(format!(
                        "{}:{} [{}] {}",
                        src.file_name().unwrap().to_string_lossy(),
                        i + 1,
                        m,
                        line.trim()
                    ));
                }
            }
        }
    }

    assert!(
        scanned >= 40,
        "only {scanned} blobs were emitted and scanned; a sweep that compiles \
         nothing reports no SIMD perfectly"
    );
    assert!(
        offenders.is_empty(),
        "{} blob line(s) contain SIMD, so the backend's description must change \
         with them:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

/// The emitter cannot produce SIMD, so nothing it prints may claim SIMD.
///
/// Both halves are source-level, and they have to be: the first is a property
/// of string literals the compiler prints, and the second is a property of the
/// emitter's own text rather than of any one run.
#[test]
fn the_backend_does_not_advertise_simd_it_cannot_emit() {
    let emitter = std::fs::read_to_string(repo().join("src/cpu_emitter.rs")).expect("cpu_emitter");
    // Comments explain the history of what was removed, so only real code
    // counts here.
    let code: String = emitter
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.starts_with("//") && !t.starts_with("///")
        })
        .collect::<Vec<_>>()
        .join("\n");
    for m in ["_mm256", "_mm512", "__m256", "__m512"] {
        assert!(
            !code.contains(m),
            "src/cpu_emitter.rs contains `{m}`. If this backend has grown SIMD, \
             the banner, the blob marker, the README row and the language \
             reference all describe it wrongly and must change together."
        );
    }

    let main_rs = std::fs::read_to_string(repo().join("src/main.rs")).expect("main.rs");
    let start = main_rs
        .find("} else if emit_cpu {")
        .expect("the --emit-cpu dispatch arm must exist");
    // Bounded window: this arm, up to the next arm.
    let rest = &main_rs[start + 4..];
    let end = rest.find("\n    } else").map(|e| e + 4).unwrap_or(rest.len());
    let arm = &rest[..end];
    assert!(
        arm.contains("GENERATED RUST BLOB"),
        "the extracted window is not the --emit-cpu arm; the anchor has moved"
    );

    for line in arm.lines() {
        // Only user-visible text, not identifiers or comments.
        let t = line.trim_start();
        if t.starts_with("//") || !(t.contains("println!") || t.contains("log_step!")) {
            continue;
        }
        for claim in SIMD_CLAIMS {
            assert!(
                !line.contains(claim),
                "the --emit-cpu path prints a line claiming `{claim}` while the \
                 backend emits no SIMD:\n  {}",
                line.trim()
            );
        }
    }
}

/// The documentation must not advertise it either.
///
/// A user picks a backend from the README's table, not from the emitter.
#[test]
fn the_documentation_does_not_advertise_simd_it_cannot_emit() {
    for (file, needle) in [
        ("README.md", "| `--emit-cpu` |"),
        ("docs/y_language_documentation.md", "| `--emit-cpu` / `--target=cpu` |"),
    ] {
        let text = std::fs::read_to_string(repo().join(file)).unwrap_or_else(|_| panic!("{file}"));
        let row = text
            .lines()
            .find(|l| l.starts_with(needle))
            .unwrap_or_else(|| panic!("{file} has no --emit-cpu row starting `{needle}`"));
        // "no SIMD" is a legitimate thing for the row to SAY, so only a claim
        // that it produces SIMD is refused.
        let claims_simd = row.contains("Rust/AVX")
            || row.contains("AVX source")
            || row.contains("AVX-512 source");
        assert!(
            !claims_simd,
            "{file} describes --emit-cpu as producing AVX, which it does not:\n  {row}"
        );
    }

    // `@avx_emit` is a hard syntax error; the reference must not present it as
    // a working directive.
    let lang = std::fs::read_to_string(repo().join("docs/y_language_documentation.md")).unwrap();
    let idx = lang.find("### 9.7").expect("section 9.7 must exist");
    let section = &lang[idx..idx + 1200.min(lang.len() - idx)];
    assert!(
        section.contains("NOT IMPLEMENTED"),
        "section 9.7 documents `@avx_emit` without saying it is unimplemented. \
         Measured: `@avx_emit` is `Error: Unexpected top-level item`, while \
         `@ptx_emit` and `@hdl_emit` parse and are ignored."
    );
}

/// THE CONTROL. Emitting nothing, or refusing everything, satisfies every
/// assertion above while deleting the backend.
#[test]
fn an_ordinary_program_still_emits_a_blob_containing_its_code() {
    let blob = emit_cpu_blob(
        "ctl",
        "fn add(a: I32, b: I32) -> I32 { return a + b; }\nfn main() { let x: I32 = add(2, 3); return; }\n",
    )
    .expect("an ordinary host program must still emit a blob");
    assert!(
        blob.contains("fn add"),
        "the blob does not contain the program it was given:\n{blob}"
    );
    assert!(
        blob.lines().count() > 10,
        "the blob is suspiciously short; a backend that prints a header and \
         nothing else passes every SIMD assertion in this file:\n{blob}"
    );
}
