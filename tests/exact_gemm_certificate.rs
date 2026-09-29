//! The certificate a compilation carries with the kernel it emitted.
//!
//! `proofs/` is checked once at build time against the shipped schedule. Its
//! theorems are universally quantified over `M`, `N`, `K` and `nthr`, so every
//! shape is covered - but until now a user who compiled their own `@ZeroDrift`
//! nest got a fast kernel and no artifact, and "proof-carrying" described the
//! repository rather than the output. `src/exact_gemm_certificate.rs` renders a
//! `.v` beside the `.ll` instantiating
//! `ExactGemmWhole.the_threaded_gemm_holds_the_source_dot_products` at THIS
//! compilation's flush interval and operand bound.
//!
//! ** WHY THIS IS NOT PAPERWORK, and it is the whole reason the file exists.
//!
//! The one hypothesis that depends on the program is the LICENCE,
//! `2 * Fl * m^2 <= i32::MAX`. Y decides it in FLOATING POINT - a `sqrt` and a
//! `floor` on `f64` in `VnniExact::max_operand_magnitude`. The certificate
//! states it over `Z` and hands it to `coqc`. Two independent derivations of
//! the same obligation, by two tools, one of which does not have floats.
//! [the_certificate_refuses_exactly_the_bounds_the_compiler_refuses] is what
//! checks they agree, at the one-unit edge, at every interval the scheme
//! admits.
//!
//! ** What the tests here deliberately do NOT do.
//!
//! They do not re-check the library proofs - `tests/proofs_are_checked.rs`
//! does that, over every `.v` in `proofs/`. These check the *plumbing*: that a
//! substitution emits a certificate, that a non-substitution does not, that
//! the numbers in it come from the source rather than from a default, and that
//! the artifact `coqc` is handed actually compiles.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A private scratch directory.
///
/// **The tag is in the SIGNATURE, not in a comment asking the caller to
/// remember.** Two tests sharing one temp-dir name, where one calls
/// `remove_dir_all` while the other is writing, is a race this repository has
/// now hit five times; it presents as an intermittent failure with no
/// diagnosis attached.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("y_cert_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn have_coqc() -> bool {
    Command::new("coqc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// An exact nest: `I16` buffers, operands widened to `I64` at the load,
/// accumulating into an `I64` buffer under `@ZeroDrift`, with `@bounds` on
/// BOTH operands - which is what licenses the substitution.
fn exact_source(bound: i64) -> String {
    format!(
        r#"
kernel y_matmul(A: GlobalMemory<I16>, B: GlobalMemory<I16>, C: GlobalMemory<I64>, M: I32, N: I32, K: I32) {{
    @invariant(i >= 0)
    for i in 0..M step 1 {{
        @invariant(j >= 0)
        for j in 0..N step 1 {{
            @ZeroDrift
            let mut sum: I64 = 0;
            @invariant(k >= 0)
            for k in 0..K step 1 {{
                @bounds(min=-{bound}, max={bound})
                let a_val: I64 = block_ptr2d_load(A, i, k, K, M, K);
                @bounds(min=-{bound}, max={bound})
                let b_val: I64 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a_val * b_val;
            }}
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }}
    }}
}}

fn main() {{
}}
"#
    )
}

/// The same nest with NO operand bounds. `@bounds` on the accumulator states
/// the range of the SUM, and a bound on a sum implies nothing about its terms
/// - so this nest is exact and cannot be licensed for the fast kernel.
const UNLICENSED_SOURCE: &str = r#"
kernel y_matmul(A: GlobalMemory<I16>, B: GlobalMemory<I16>, C: GlobalMemory<I64>, M: I32, N: I32, K: I32) {
    @invariant(i >= 0)
    for i in 0..M step 1 {
        @invariant(j >= 0)
        for j in 0..N step 1 {
            @ZeroDrift
            let mut sum: I64 = 0;
            @invariant(k >= 0)
            for k in 0..K step 1 {
                let a_val: I64 = block_ptr2d_load(A, i, k, K, M, K);
                let b_val: I64 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a_val * b_val;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}

fn main() {
}
"#;

/// Compile `src` with `--emit-llvm` inside `dir` and return the compiler's
/// combined output. The `.ll` lands in `dir`, so the certificate does too.
fn compile(dir: &PathBuf, src: &str, extra_env: &[(&str, &str)]) -> String {
    compile_named(dir, "gemm.ysu", src, extra_env)
}

fn compile_named(
    dir: &PathBuf,
    name: &str,
    src: &str,
    extra_env: &[(&str, &str)],
) -> String {
    let path = dir.join(name);
    std::fs::write(&path, src).expect("write source");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&path).arg("--emit-llvm").current_dir(repo());
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run Y");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    assert!(
        out.status.success(),
        "the compiler refused a program it is supposed to accept:\n{text}"
    );
    text
}

/// Copy `proofs/*.v` into `dir` and compile them, retrying so the `Require`
/// graph resolves without this helper knowing its order.
///
/// Done ONCE per directory, because the licence sweep below writes eight
/// probes into one directory and rebuilding the fourteen-file library for each
/// of them took two minutes for no additional coverage.
fn coq_prepare(dir: &PathBuf) {
    let proofs = repo().join("proofs");
    let mut pending: Vec<String> = Vec::new();
    for e in std::fs::read_dir(&proofs).expect("proofs dir") {
        let p = e.expect("dir entry").path();
        if p.extension().and_then(|s| s.to_str()) == Some("v") {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            std::fs::copy(&p, dir.join(&name)).expect("copy a proof");
            pending.push(name);
        }
    }
    for _ in 0..pending.len().max(1) {
        let mut still = Vec::new();
        let mut progressed = false;
        for name in pending.drain(..) {
            if coqc(dir, &name).0 {
                progressed = true;
            } else {
                still.push(name);
            }
        }
        if still.is_empty() || !progressed {
            assert!(
                still.is_empty(),
                "the committed proofs do not compile, so nothing here can be \
                 measured: {still:?}"
            );
            break;
        }
        pending = still;
    }
}

fn coqc(dir: &PathBuf, name: &str) -> (bool, String) {
    let out = Command::new("coqc")
        .args(["-Q", ".", ""])
        .arg(name)
        .current_dir(dir)
        .output()
        .expect("run coqc");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

/// The headline: a substituted exact kernel emits a certificate, and `coqc`
/// accepts it against the committed proofs with no axioms.
///
/// This is the "carrying" half. Everything the certificate says is derived
/// from `ExactGemmWhole`, so the interesting claim is not that the theorem is
/// true - `proofs_are_checked.rs` covers that - but that the artifact Y writes
/// for a real compilation is a well-formed instance of it.
#[test]
fn a_substituted_exact_gemm_emits_a_certificate_that_coqc_accepts() {
    let dir = scratch("accepts");
    let out = compile(&dir, &exact_source(1024), &[]);
    assert!(
        out.contains("EXACT vpdpwssd kernel substituted"),
        "the fixture stopped being substituted, so this test is measuring \
         nothing:\n{out}"
    );

    let cert = dir.join("gemm_certificate.v");
    assert!(
        cert.exists(),
        "no certificate beside the .ll. The compiler said it substituted the \
         exact kernel, which is exactly when the certificate is meaningful:\n{out}"
    );
    let text = std::fs::read_to_string(&cert).expect("read certificate");

    // The numbers come from the SOURCE, not from a default. A renderer that
    // hardcoded the shipped interval and a plausible bound would pass every
    // `coqc` check in this file.
    assert!(text.contains("Definition m : Z := 1024."), "{text}");
    assert!(text.contains("Definition Fl : nat := 64."), "{text}");

    // THE OBLIGATION IS ASSERTED AS TEXT, and that is the right instrument
    // here rather than a weaker one. Coq compares propositions up to
    // CONVERSION, and at a licensed numeral `2*Fl*m <= I32MAX` and
    // `2*Fl*m*m <= I32MAX` both reduce to `Lt <> Gt` - so a certificate
    // stating the wrong obligation still type-checks, still refuses exactly
    // the magnitudes Y refuses (the use site forces the real hypothesis by
    // conversion too), and is indistinguishable from the correct one by any
    // `coqc` run. Measured, not assumed: dropping one factor of `m` here
    // passed all seven tests in this file including the licence sweep.
    //
    // What it changes is the claim a HUMAN auditing the artifact reads, which
    // for a certificate is the point of the artifact. So the check is on the
    // statement's text.
    assert!(
        text.contains("  2 * Z.of_nat Fl * m * m <= ExactGemmMicro.I32MAX."),
        "the certificate does not STATE the overflow obligation it is about:\n{text}"
    );

    // ...and the non-vacuity theorem must EVALUATE the model, for the same
    // reason and one measured the same way. `Theorem ... : True. Proof.
    // exact I. Qed.` compiles, reports "Closed under the global context", and
    // passes the report count below - so `coqc` accepting the file is
    // necessary and not sufficient. This is
    // `every_proof_has_a_content_control` in `proofs_are_checked.rs`, which
    // guards the COMMITTED proofs against exactly this and had no counterpart
    // for the generated one.
    assert!(
        text.contains("W.thread_sum Acert Bcert"),
        "the certificate's non-vacuity theorem does not evaluate the model:\n{text}"
    );

    if !have_coqc() {
        eprintln!("skipping the coqc half: no coqc on PATH - the certificate is NOT being checked");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    coq_prepare(&dir);
    let (ok, log) = coqc(&dir, "gemm_certificate.v");
    assert!(ok, "coqc refused the certificate Y emitted:\n{log}");
    assert!(
        !log.contains("Axioms:"),
        "the certificate rests on an axiom:\n{log}"
    );
    assert_eq!(
        log.matches("Closed under the global context").count(),
        3,
        "expected three `Print Assumptions` reports:\n{log}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The control, and without it "always write a certificate" passes everything
/// above.
///
/// A nest with no operand `@bounds` is still EXACT - scalar lowering honours
/// `@ZeroDrift` - it is simply not licensed for the fast kernel. There is then
/// nothing to certify: the emitted code IS the naive nest, so a certificate
/// asserting it equals the naive nest would be true, vacuous, and misleading
/// about what was substituted.
#[test]
fn a_nest_left_on_the_scalar_path_emits_no_certificate() {
    let dir = scratch("scalar");
    let out = compile(&dir, UNLICENSED_SOURCE, &[]);
    assert!(
        out.contains("using scalar lowering"),
        "the fixture was substituted after all, so it is not the control it \
         claims to be:\n{out}"
    );
    let stray: Vec<_> = std::fs::read_dir(&dir)
        .expect("scratch")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".v"))
        .collect();
    assert!(
        stray.is_empty(),
        "a certificate was written for a nest that was never substituted: {stray:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The bound in the certificate is the source's, at a value that is NOT the
/// default and NOT the one the other tests use.
#[test]
fn the_certificate_states_the_source_s_own_bound() {
    let dir = scratch("bound");
    compile(&dir, &exact_source(4000), &[]);
    let text = std::fs::read_to_string(dir.join("gemm_certificate.v")).expect("certificate");
    assert!(
        text.contains("Definition m : Z := 4000."),
        "the certificate did not carry the source's bound:\n{text}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A FRACTIONAL `@bounds` is rounded UP, and this is the only place the
/// direction is observable end to end.
///
/// Every other fixture here declares an integer bound, where `ceil` and
/// `floor` agree - so a renderer that rounded the wrong way would pass all of
/// them. `floor` is wrong twice over: it would licence against a smaller `m`
/// than the source declared, AND state the guarantee over fewer matrices than
/// the source admits, leaving the operands between `floor(m)` and `m`
/// uncovered by the certificate while the kernel happily consumes them.
#[test]
fn a_fractional_bound_is_certified_at_the_integer_above_it() {
    let dir = scratch("fractional");
    let src = exact_source(1024).replace("=-1024, max=1024", "=-1024.5, max=1024.5");
    let out = compile(&dir, &src, &[]);
    assert!(
        out.contains("|x| <= 1024.5"),
        "the fractional bound did not survive the front end, so this test is \
         measuring the integer case again:\n{out}"
    );
    let text = std::fs::read_to_string(dir.join("gemm_certificate.v")).expect("certificate");
    assert!(
        text.contains("Definition m : Z := 1025."),
        "a fractional bound was not rounded up:\n{text}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// **THE CROSS-CHECK, and the reason the certificate is worth emitting.**
///
/// Y grants the licence with `m <= floor(sqrt(i32::MAX / (2 * Fl)))`, computed
/// on `f64`. The certificate asserts `2 * Fl * m^2 <= i32::MAX` over `Z`. The
/// two must agree at the edge, and the edge is ONE UNIT WIDE - at the default
/// interval, 4095 fits with room to spare and 4096 exceeds `i32::MAX` by
/// exactly one.
///
/// Checked at every interval the scheme admits rather than at the shipped one,
/// because a disagreement is most likely where the `sqrt` is least exact.
#[test]
fn the_certificate_refuses_exactly_the_bounds_the_compiler_refuses() {
    if !have_coqc() {
        eprintln!("skipping: no coqc on PATH - the licence cross-check is NOT running");
        return;
    }
    let dir = scratch("licence");
    coq_prepare(&dir);
    let mut checked = 0usize;
    for shift in [0u32, 3, 6, 9] {
        let fl = 1u32 << shift;
        let scheme = y::zero_drift::VnniExact::new(fl).expect("a power-of-two interval");
        let limit = scheme.max_operand_magnitude();

        for (mag, y_licensed) in [(limit, true), (limit + 1.0, false)] {
            assert_eq!(
                scheme.license(mag).is_ok(),
                y_licensed,
                "the fixture disagrees with Y about magnitude {mag} at interval {fl}"
            );
            let cert = y::exact_gemm_certificate::Certificate {
                operand_magnitude: mag,
                flush_k_pairs: fl,
                extent_m: "M".into(),
                extent_n: "N".into(),
            };
            let stem = format!("probe_{fl}_{}", mag as u64);
            let text = y::exact_gemm_certificate::render(&cert, "licence probe", &stem);
            std::fs::write(dir.join(format!("{stem}.v")), &text).expect("write probe");
            let (ok, log) = coqc(&dir, &format!("{stem}.v"));
            assert_eq!(
                ok, y_licensed,
                "coqc and Y disagree at interval {fl}, magnitude {mag}: Y says \
                 licensed={y_licensed}, coqc says {ok}.\n{log}"
            );
            checked += 1;
        }
    }
    // A sweep that checked nothing reports "no disagreements" perfectly.
    assert_eq!(checked, 8, "the licence sweep did not run");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The escape hatch works, and says so rather than going quiet.
#[test]
fn the_certificate_can_be_suppressed() {
    let dir = scratch("suppressed");
    let out = compile(&dir, &exact_source(1024), &[("Y_NO_CERTIFICATE", "1")]);
    assert!(
        out.contains("suppressed by Y_NO_CERTIFICATE"),
        "suppression was silent:\n{out}"
    );
    assert!(!dir.join("gemm_certificate.v").exists());
    // ...and the kernel it would have certified is still substituted, so the
    // switch controls the paperwork and not the codegen.
    assert!(out.contains("EXACT vpdpwssd kernel substituted"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Lowering the nest as written emits no certificate either: there is no
/// substituted kernel, so there is nothing to certify equal to the nest.
#[test]
fn the_naive_reading_of_the_same_source_emits_no_certificate() {
    let dir = scratch("naive");
    let out = compile(
        &dir,
        &exact_source(1024),
        &[("Y_NO_GEMM_RECOGNISER", "1")],
    );
    assert!(
        !out.contains("EXACT vpdpwssd kernel substituted"),
        "the recogniser switch stopped working, so this control is vacuous:\n{out}"
    );
    assert!(!dir.join("gemm_certificate.v").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A source name that is not a legal Coq identifier still yields a certificate
/// `coqc` can be pointed at.
///
/// `coqc` derives the module's logical name from the FILE name, and `4-bit
/// gemm.ysu` is an ordinary file name. Sanitising in the renderer alone would
/// leave the header's own "check with" line naming a module that does not
/// exist - the certificate would compile if you guessed the name and its
/// instructions would be wrong.
#[test]
fn a_source_name_that_is_not_a_coq_identifier_is_sanitised() {
    let dir = scratch("stem");
    let out = compile_named(&dir, "4-bit gemm.ysu", &exact_source(1024), &[]);
    assert!(out.contains("EXACT vpdpwssd kernel substituted"), "{out}");

    let cert = dir.join("_4_bit_gemm_certificate.v");
    assert!(
        cert.exists(),
        "the certificate is not where the sanitiser says it is:\n{out}"
    );
    let text = std::fs::read_to_string(&cert).expect("certificate");
    assert!(
        text.contains("_4_bit_gemm_certificate.v"),
        "the header names a module other than the file it is in:\n{text}"
    );

    if !have_coqc() {
        eprintln!("skipping the coqc half: no coqc on PATH");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    coq_prepare(&dir);
    let (ok, log) = coqc(&dir, "_4_bit_gemm_certificate.v");
    assert!(ok, "coqc refused a certificate with a sanitised name:\n{log}");
    let _ = std::fs::remove_dir_all(&dir);
}
