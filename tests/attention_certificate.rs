//! The GPU kernel emits a certificate, and it states the `ptxas` boundary.
//!
//! `docs/proof_carrying_kernels.md` Phase 3 says of the GPU pipeline: *"the
//! proof covers source-to-PTX, and `ptxas` is trusted or validated
//! per-translation. **This must be stated in the certificate, never papered
//! over.**"* There was no certificate. `--emit-attention-ptx` wrote a `.ptx` to
//! stdout and nothing else, while `AttentionSchedule.v`, `GridStrideSplit.v`
//! and `SoftmaxErrorBound.v` - 96 theorems - described that exact kernel.
//!
//! WHAT THIS GATE ASSERTS, and why each part is not covered by the others.
//!
//! `coqc` accepting the emitted file is **necessary and not sufficient**, which
//! is a recorded finding rather than a caution: `Theorem x : True.` compiles
//! and reports "Closed under the global context". So the content controls check
//! the certificate STATES its obligation and EVALUATES its model, and the
//! boundary test checks it names `ptxas` as trusted-not-validated rather than
//! merely mentioning it.
//!
//! The load-bearing one is the EDGE. The accumulator bound is decided twice, by
//! tools that share no code and no representation: Y computes it in `usize`,
//! `coqc` checks it over `Z`. They must agree on a boundary one unit wide.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use y::exact_attention::MAX_EXACT_SEQ_LEN;
use y::exact_attention_certificate::{file_stem, render, Certificate};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn have_coqc() -> bool {
    Command::new("coqc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `coqc` runs one at a time in the shared proof directory.
///
/// TWO RECORDED LESSONS PULL IN OPPOSITE DIRECTIONS HERE, and the resolution is
/// to obey both rather than to pick one. "A gate that runs `coqc` must build the
/// library ONCE per directory" (a sweep that rebuilt per probe cost 117 s for
/// zero coverage) says share the directory. "Any temp-dir helper needs a
/// per-test tag" - a race this repository has hit six times - says do not.
///
/// Sharing is what the first lesson wants and `coqc` writes **`.lia.cache` into
/// its working directory**, so two tests compiling different certificates in
/// one directory still contend on one file. Observed as an aggregate run
/// aborting at 56 result lines against 142 on a serial re-run, which is this
/// repository's own documented tell.
///
/// So: one directory, built once, and the compilations serialised. The
/// certificates are seconds each, so the mutex costs nothing measurable.
fn coqc(dir: &PathBuf, name: &str) -> (bool, String) {
    static LOCK: Mutex<()> = Mutex::new(());
    let _held = LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

/// The compiled proof library, built ONCE for this whole test binary.
///
/// The recorded lesson is exact: a licence sweep that rebuilt the proofs per
/// probe cost 117 s for zero additional coverage, against 19.5 s once the
/// build was split out. Four tests here need the library and none needs its
/// own copy.
fn proof_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("y_attn_cert_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
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
        // Dependency order is not known here, so iterate to a fixpoint. The
        // assertion is that it CONVERGES: a proof left over is a proof that
        // does not compile, and then nothing below can be measured.
        for _ in 0..pending.len().max(1) {
            let mut still = Vec::new();
            let mut progressed = false;
            for name in pending.drain(..) {
                if coqc(&dir, &name).0 {
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
        dir
    })
}

fn bin() -> PathBuf {
    repo().join("target/release/Y")
}

/// Run the real compiler and return `(stdout, stderr, wrote_certificate)`.
fn emit(dir: &PathBuf, head_dim: usize, seq_len: usize, suppress: bool) -> (String, String, bool) {
    let mut cmd = Command::new(bin());
    cmd.args(["--emit-attention-ptx", &head_dim.to_string(), &seq_len.to_string()])
        .current_dir(dir);
    if suppress {
        cmd.env("Y_NO_CERTIFICATE", "1");
    } else {
        cmd.env_remove("Y_NO_CERTIFICATE");
    }
    let out = cmd.output().expect("run Y");
    let cert = dir.join(format!(
        "{}.v",
        file_stem(&Certificate { head_dim, seq_len })
    ));
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        cert.exists(),
    )
}

/// The headline: the emitted kernel carries a certificate, and `coqc` accepts
/// it against the committed proofs with no axioms.
#[test]
fn the_emitted_kernel_carries_a_certificate_that_checks() {
    if !have_coqc() {
        eprintln!("SKIP: coqc not installed");
        return;
    }
    let dir = proof_dir();
    let (ptx, err, wrote) = emit(dir, 128, 4096, false);
    assert!(wrote, "no certificate was written; stderr was: {err}");
    assert!(
        ptx.starts_with(".version"),
        "stdout must be PTX and nothing else - the bridge pipes it into \
         cuModuleLoadData, so a diagnostic on that stream is a driver parse \
         error. Got: {:?}",
        &ptx[..ptx.len().min(80)]
    );
    assert!(
        err.contains("certificate written to"),
        "the notice belongs on stderr, and it is missing: {err}"
    );
    let name = format!("{}.v", file_stem(&Certificate { head_dim: 128, seq_len: 4096 }));
    let (ok, log) = coqc(dir, &name);
    assert!(ok, "coqc rejected the emitted certificate:\n{log}");
    assert!(
        !log.contains("Axioms:"),
        "the certificate rests on an axiom:\n{log}"
    );
    let closed = log.matches("Closed under the global context").count();
    assert_eq!(
        closed, 5,
        "expected five Print Assumptions reports, got {closed}:\n{log}"
    );
}

/// **The load-bearing test.** Two tools, no shared code, one obligation, and a
/// boundary one unit wide.
///
/// Y decides the accumulator bound in `usize`; the certificate states it over
/// `Z` and `coqc` decides it. If they disagreed, the compiler would either
/// refuse a kernel that is exact or emit a certificate for one that wraps.
#[test]
fn the_certificate_refuses_exactly_the_lengths_the_compiler_refuses() {
    if !have_coqc() {
        eprintln!("SKIP: coqc not installed");
        return;
    }
    let dir = proof_dir();

    // At the bound: the compiler emits, and coqc accepts.
    let (_, _, wrote) = emit(dir, 128, MAX_EXACT_SEQ_LEN, false);
    assert!(wrote, "the compiler refused a length it says is exact");
    let at = format!(
        "{}.v",
        file_stem(&Certificate { head_dim: 128, seq_len: MAX_EXACT_SEQ_LEN })
    );
    let (ok, log) = coqc(dir, &at);
    assert!(ok, "coqc refused the certificate AT the bound:\n{log}");

    // One past: the compiler refuses outright, so there is no certificate.
    let (_, err, wrote) = emit(dir, 128, MAX_EXACT_SEQ_LEN + 1, false);
    assert!(!wrote, "a certificate was written past the exactness bound");
    assert!(
        err.contains("past") && err.contains(&MAX_EXACT_SEQ_LEN.to_string()),
        "the refusal should name the bound: {err}"
    );

    // And the other half of the agreement: had it been emitted, `coqc` would
    // have rejected it. Rendered directly, because the compiler will not
    // produce this file - which is the point.
    let over = Certificate { head_dim: 128, seq_len: MAX_EXACT_SEQ_LEN + 1 };
    let stem = "attention_over_bound_certificate";
    std::fs::write(
        dir.join(format!("{stem}.v")),
        render(&over, "edge probe", stem),
    )
    .expect("write the over-bound certificate");
    let (ok, log) = coqc(dir, &format!("{stem}.v"));
    assert!(
        !ok,
        "coqc ACCEPTED a certificate one past the bound the compiler refuses. \
         The two tools disagree about the obligation:\n{log}"
    );
}

/// Phase 3's explicit requirement: the `ptxas` boundary is stated, and stated
/// as *trusted, not validated* for this kernel.
///
/// A certificate that merely mentioned `ptxas` would satisfy a substring check
/// while papering over exactly what Phase 3 says must not be papered over, so
/// this asserts the verdict and the route, not the word.
#[test]
fn the_certificate_states_the_ptxas_boundary() {
    let text = render(
        &Certificate { head_dim: 128, seq_len: 4096 },
        "test",
        "attention_probe_certificate",
    );
    for needle in [
        "ptxas",
        "closed",
        "NOT CHECKED",
        "TRUSTED and not validated",
        "tools/ptxas_tval/",
    ] {
        assert!(
            text.contains(needle),
            "the certificate does not state the ptxas boundary: {needle:?} is absent"
        );
    }
    // The floor below the model must be named too. A list that stops before the
    // hardware reads as one that was abandoned rather than one that ends.
    assert!(
        text.contains("GPU executing its own ISA"),
        "the certificate stops before the hardware, so a reader cannot tell a \
         boundary that ENDS from one that trails off"
    );
}

/// `coqc` accepting a generated proof is necessary and not sufficient, so the
/// certificate must STATE the obligation it is about and EVALUATE the model it
/// claims to.
///
/// This is the same control `tests/proofs_are_checked.rs` applies to the
/// committed proofs, which had no counterpart for a GENERATED one until the
/// exact-GEMM certificate grew it. Mutation on the GEMM side established that
/// `Theorem the_certificate_is_not_vacuous : True.` compiles, reports "Closed
/// under the global context", and satisfies a COUNT of such reports.
#[test]
fn the_certificate_states_its_obligation_and_evaluates_its_model() {
    let cert = Certificate { head_dim: 128, seq_len: 4096 };
    let text = render(&cert, "test", "attention_probe_certificate");
    assert!(
        text.contains("Definition seq_len_Z : Z := 4096."),
        "the certificate does not name the length it was emitted for"
    );
    assert!(
        text.contains("seq_len_Z * (2 ^ 28 - 1) * 127 < 2 ^ 63"),
        "the obligation is not stated in the certificate, so a reader auditing \
         the artifact cannot see what it claims"
    );
    assert!(
        text.contains("Theorem the_certificate_is_not_vacuous")
            && text.contains("vm_compute"),
        "the certificate does not evaluate its model, so `coqc` accepting it \
         says only that it type-checks"
    );
    // A `nat` literal is unary. At a production length a literal would be
    // hundreds of millions of constructors, and every normalising tactic would
    // try to evaluate it - a recorded landmine in this repository's proofs.
    assert!(
        text.contains("Definition seq_len : nat := Z.to_nat seq_len_Z."),
        "the length reaches `nat` as a literal rather than through Z.to_nat"
    );
}

/// The escape hatch exists, and it is loud.
///
/// A compiler that quietly stops emitting its proof is the failure this whole
/// programme is about, so suppression has to say so on the stream a human
/// reads.
#[test]
fn suppression_is_available_and_says_so() {
    let dir = std::env::temp_dir().join(format!("y_attn_sup_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let (ptx, err, wrote) = emit(&dir, 64, 1024, true);
    assert!(!wrote, "Y_NO_CERTIFICATE did not suppress the certificate");
    assert!(
        err.contains("suppressed by Y_NO_CERTIFICATE"),
        "suppression was silent, which is indistinguishable from a compiler \
         that never had a certificate: {err}"
    );
    assert!(
        ptx.starts_with(".version"),
        "suppressing the certificate must not disturb the kernel"
    );
    // The control that stops "never emit one" passing every assertion above.
    let (_, _, wrote) = emit(&dir, 64, 1024, false);
    assert!(
        wrote,
        "without the variable the certificate must still be written - otherwise \
         'suppression works' is satisfied by emitting nothing, ever"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
