//! `exact_pv` is the one kernel with BOTH a Rocq proof and a validated SASS.
//!
//! `proofs/ExactPvExact.v` proves what the emitted PTX computes;
//! `tools/ptxas_tval/loopval.py` validates that PTX against the SASS `ptxas`
//! produced from it. Until this file those two sets were DISJOINT -- every
//! kernel the validator had a standing result for carried no proof, and the
//! three proved GPU kernels were not in its corpus -- and
//! `docs/ptxas_translation_validation.md` said
//! otherwise, calling `exact_pv` "the kernel that carries three Rocq files"
//! when it carried none. This gate asserts the overlap instead of describing
//! it, in both directions.
//!
//! FOUR JOBS, and the first is the seam.
//!
//!   1. The TRANSCRIPTION tie. `ptx_emitter.rs` does not go through the `Ix`
//!      extraction layer, so nothing renders the proof and the kernel from one
//!      description the way `exact_attention.rs` does. The proof's model is
//!      therefore asserted against the emitted text: the operand domains come
//!      from the load instructions' own widths and extensions, the accumulator
//!      from `mul.lo.s64`/`add.s64`, the indices from `mul.lo.s32`/`add.s32`,
//!      and the bound from `setp.lt.u32`. Change any one and the proof is
//!      about a different kernel.
//!
//!   2. The OVERLAP. `exact_pv` is named by a proof and is a VALIDATED row.
//!
//!   3. The CONTROL, which is what stops job 2 being satisfied by claiming
//!      everything. The attention kernel is proved and is NOT validated, and
//!      `src/exact_attention_certificate.rs` records `ptxas` as trusted there.
//!      Deleting that distinction is the original defect.
//!
//!   4. The two device boundaries the proof refutes, each one iteration wide.
//!      A licence nothing can violate certifies nothing.
//!
//! The kernel is compiled into a per-test TEMP DIRECTORY. `--emit-ptx` writes
//! next to its input, so compiling in place would rewrite a committed artifact
//! and race `tests/ptx_exact_pv.rs`, which compiles the same source.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn y_binary() -> PathBuf {
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    bin.join("Y")
}

/// The tag is in the SIGNATURE so the next author cannot forget it, and a
/// process-wide counter is appended because a tag makes the requirement
/// visible without making it unique -- two tests passing the same word is the
/// shared-temp-dir race this repository has recorded seven times.
fn emit_ptx(tag: &str) -> String {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "y_exact_pv_proof_{}_{}_{}",
        tag,
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let src = dir.join("exact_pv.ysu");
    std::fs::copy(repo().join("tests/exact_pv.ysu"), &src).expect("copy source");
    let out = Command::new(y_binary())
        .arg(&src)
        .arg("--emit-ptx")
        // The repo is the cwd so `.ysu_hw_profile` is found, exactly as the
        // sibling harness does; only the SOURCE moves.
        .current_dir(repo())
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "exact_pv.ysu did not compile:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(dir.join("exact_pv.ptx")).expect("no .ptx emitted");
    let _ = std::fs::remove_dir_all(&dir);
    ptx
}

fn proof_source() -> String {
    std::fs::read_to_string(repo().join("proofs/ExactPvExact.v")).expect("proofs/ExactPvExact.v")
}

fn tval_doc() -> String {
    std::fs::read_to_string(repo().join("docs/ptxas_translation_validation.md"))
        .expect("docs/ptxas_translation_validation.md")
}

// ---------------------------------------------------------------------------
// 1. The transcription tie: the proof's model IS the emitted arithmetic.
// ---------------------------------------------------------------------------

/// `p_domain` is `[0, 2^32)` and `v_domain` is `[-128, 127]`, and those are not
/// choices -- they are what the emitted load instructions produce. `P` is read
/// with `ld.global.u32` and widened with `cvt.u64.u32`, a ZERO-extend; `V` with
/// `ld.global.s8` and `cvt.s64.s32`, a SIGN-extend. Swap either extension and
/// the proof's licence is computed over a domain the kernel does not have.
#[test]
fn the_emitted_loads_are_the_operand_domains_the_proof_states() {
    let ptx = emit_ptx("loads");
    assert!(
        ptx.contains("ld.global.u32") && ptx.contains("cvt.u64.u32"),
        "the weight load is no longer an unsigned 32-bit load widened by a \
         zero-extend, so `p_domain = [0, 2^32)` in ExactPvExact.v is wrong:\n{ptx}"
    );
    assert!(
        ptx.contains("ld.global.s8") && ptx.contains("cvt.s64.s32"),
        "the activation load is no longer a signed 8-bit load widened by a \
         sign-extend, so `v_domain = [-128, 127]` in ExactPvExact.v is wrong. \
         A ZERO-extend here turns every negative activation into a large \
         positive one:\n{ptx}"
    );
    let src = proof_source();
    assert!(
        src.contains("Definition PMAX : Z := 4294967295.")
            && src.contains("Definition VABS : Z := 128."),
        "the proof's operand bounds moved away from the emitted load widths"
    );
}

/// The accumulator is int64 and both of its operations WRAP, which is what
/// `wrap64` models. A model with a non-wrapping accumulator would prove a
/// theorem about a machine that cannot overflow, and the device does.
#[test]
fn the_emitted_accumulator_is_the_wrapping_int64_the_proof_models() {
    let ptx = emit_ptx("acc");
    assert!(
        ptx.contains("mul.lo.s64") && ptx.contains("add.s64"),
        "the accumulation is no longer a 64-bit multiply-and-add, so `wrap64` \
         and `the_product_never_wraps` describe something else:\n{ptx}"
    );
    // `st.global.u64` is the only store, so the accumulator is what lands.
    assert!(
        ptx.contains("st.global.u64"),
        "the kernel no longer stores a 64-bit accumulator:\n{ptx}"
    );
}

/// **The finding's tie.** Every index is computed in 32-bit arithmetic that
/// WRAPS, and the bounds predicate reads those same bits as UNSIGNED. That
/// pair is why a wrapped index does not fault: it either masks (a very large
/// unsigned value) or, as the device showed, aliases a live element.
#[test]
fn the_emitted_indices_are_thirty_two_bit_and_the_bound_is_unsigned() {
    let ptx = emit_ptx("idx");
    assert!(
        ptx.contains("mul.lo.s32") && ptx.contains("add.s32"),
        "the index arithmetic is no longer 32-bit, so `W = wrap32` in \
         ExactPvExact.v is modelling a width the kernel does not use:\n{ptx}"
    );
    assert!(
        ptx.contains("setp.lt.u32"),
        "the bounds predicate is no longer an UNSIGNED compare, so `u32_of` \
         is wrong and a wrapped-negative index would be caught rather than \
         re-read as a huge address:\n{ptx}"
    );
    assert!(
        ptx.contains("cvt.u64.u32"),
        "the address is no longer a zero-extend of the 32-bit index bits:\n{ptx}"
    );
    let src = proof_source();
    assert!(
        src.contains("Definition W (z : Z) : Z := MC.wrap32 z."),
        "the proof no longer models the index as 32-bit wrapping"
    );
}

// ---------------------------------------------------------------------------
// 2 and 3. The overlap, and the control that stops it being vacuous.
// ---------------------------------------------------------------------------

/// The names the tval results table records as VALIDATED.
fn validated_kernels(doc: &str) -> Vec<String> {
    doc.lines()
        .filter(|l| l.trim_start().starts_with('|') && l.contains("**VALIDATED**"))
        .filter_map(|l| {
            let cell = l.split('|').nth(1)?.trim();
            let name = cell.trim_matches('`').split(" @ ").next()?.trim();
            Some(name.trim_matches('`').to_string())
        })
        .filter(|n| !n.is_empty())
        .collect()
}

/// Every kernel the tval results table names, whatever its verdict.
///
/// The two doc gates below iterate THIS and not `validated_kernels`, because a
/// false proof attribution is not confined to a row that passed: the table
/// carries UNPROVED rows that are results in their own right, and a claim
/// about one of those was invisible while the gates were scoped to the
/// validated set. Found by mutating a new claim into the doc and watching the
/// gate -- written one increment earlier -- stay green.
fn table_kernels(doc: &str) -> Vec<String> {
    doc.lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with('|') && (t.contains("VALIDATED") || t.contains("UNPROVED"))
        })
        .filter_map(|l| {
            let cell = l.split('|').nth(1)?.trim();
            let name = cell.trim_matches('`').split(" @ ").next()?.trim();
            Some(name.trim_matches('`').to_string())
        })
        .filter(|n| !n.is_empty())
        .collect()
}

/// Which files in `proofs/` name a given kernel entry point.
fn proofs_naming(entry: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(repo().join("proofs"))
        .expect("proofs/")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "v").unwrap_or(false))
        .filter(|e| {
            std::fs::read_to_string(e.path())
                .map(|s| s.contains(entry))
                .unwrap_or(false)
        })
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// Which kernel entry names `proofs/` mentions at all.
fn kernel_is_proved(entry: &str) -> bool {
    let dir = repo().join("proofs");
    std::fs::read_dir(dir)
        .expect("proofs/")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "v").unwrap_or(false))
        .any(|e| {
            std::fs::read_to_string(e.path())
                .map(|s| s.contains(entry))
                .unwrap_or(false)
        })
}

/// **The chain.** `exact_pv` is proved AND validated -- the first kernel here
/// for which both steps are covered.
#[test]
fn the_proof_and_the_validator_cover_the_same_kernel() {
    let doc = tval_doc();
    let validated = validated_kernels(&doc);
    // FLOOR: a table that parsed to nothing satisfies every claim below
    // perfectly, which is the null metric this repository keeps finding.
    assert!(
        validated.len() >= 5,
        "parsed only {} VALIDATED rows out of the results table, so this gate \
         is asserting nothing: {validated:?}",
        validated.len()
    );
    assert!(
        validated.iter().any(|k| k == "exact_pv"),
        "`exact_pv` is no longer a VALIDATED row, so the chain's second step \
         is gone: {validated:?}"
    );
    assert!(
        kernel_is_proved("exact_pv"),
        "no file in proofs/ names `exact_pv`, so the chain's first step is gone"
    );
    assert!(
        proof_source().contains("the_emitted_exact_pv_holds_the_source_dot_product"),
        "ExactPvExact.v no longer states the capstone the chain rests on"
    );
}

/// **The control.** The attention kernel is proved and is NOT validated, and
/// its certificate says so. Without this, "claim every kernel is validated"
/// satisfies the test above; with it, the distinction the original defect
/// erased is the thing under test.
#[test]
fn the_attention_kernel_is_proved_and_not_validated() {
    let doc = tval_doc();
    let validated = validated_kernels(&doc);
    // `attn_scores` and not `attn_accum`: only the first is named LITERALLY in
    // proofs/ (`AttentionSchedule.v` twice, `GridStrideSplit.v` once). The
    // accumulating entry's schedule is proved too, abstractly, but this gate
    // matches on names and must use one that is actually written down --
    // otherwise the control fails for a reason unrelated to what it tests.
    for entry in ["attn_scores"] {
        assert!(
            kernel_is_proved(entry),
            "`{entry}` is no longer named by any proof, so this control is vacuous"
        );
        assert!(
            !validated.iter().any(|k| k == entry),
            "`{entry}` is now claimed VALIDATED. If the attention kernel really \
             entered the validator's corpus, \
             src/exact_attention_certificate.rs must stop recording `ptxas` as \
             trusted-and-not-validated in the same commit."
        );
    }
    // Read the RENDERED certificate, not the source file. The first version of
    // this control matched a literal against `src/exact_attention_certificate.rs`
    // and broke the moment that sentence was re-wrapped across two source lines
    // -- the emitted string was byte-identical and the claim unchanged, so the
    // gate was pinned to a text FORM rather than to what the certificate says.
    // A certificate's claim is what it renders; that is the thing to assert on.
    let rendered = y::exact_attention_certificate::render(
        &y::exact_attention_certificate::Certificate { head_dim: 128, seq_len: 4096 },
        "test",
        "attention_probe_certificate",
    );
    let flat = rendered.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        flat.contains("TRUSTED and not validated"),
        "the attention certificate no longer records `ptxas` as trusted; if \
         that changed, this control and the doc must change with it"
    );
}

/// **The defect's signature, generalised.** The original bug was prose crediting
/// a kernel with Rocq files it did not have. So: every kernel name the tval doc
/// mentions in the same sentence as "Rocq" must be named by some file in
/// `proofs/`. That catches the instance without being written for it.
#[test]
fn the_tval_doc_credits_no_kernel_with_a_proof_it_does_not_have() {
    let doc = tval_doc();
    let named = table_kernels(&doc);
    let mut checked = 0usize;
    for sentence in doc.split(['.', '\n']) {
        if !sentence.contains("Rocq") {
            continue;
        }
        for k in &named {
            if sentence.contains(k.as_str()) {
                checked += 1;
                assert!(
                    kernel_is_proved(k),
                    "the tval doc puts `{k}` in the same sentence as \"Rocq\" \
                     and no file in proofs/ names it. That is the defect this \
                     gate exists for: prose crediting a kernel with a proof \
                     that does not exist.\nsentence: {}",
                    sentence.trim()
                );
            }
        }
    }
    // FLOOR: a doc that stopped mentioning Rocq at all would pass silently.
    assert!(
        checked >= 1,
        "no sentence in the tval doc names a table kernel alongside \
         \"Rocq\", so this gate examined nothing"
    );
}

/// **The defect exactly, which the existence gate above does NOT catch.** The
/// original line did not credit a kernel with a proof that did not exist in
/// general -- it credited it with a COUNT, "three Rocq files", and `exact_pv`
/// now has one. So reverting that prose leaves the existence gate green. Any
/// sentence attributing a NUMBER of Rocq files to a kernel must state the
/// number of files in `proofs/` that name it.
#[test]
fn the_tval_doc_states_the_right_number_of_proofs_for_a_kernel() {
    let doc = tval_doc();
    let words = [
        ("no", 0usize), ("one", 1), ("two", 2), ("three", 3),
        ("four", 4), ("five", 5), ("six", 6),
    ];
    let named = table_kernels(&doc);
    let mut checked = 0usize;
    for sentence in doc.split(['.', '\n']) {
        // A correcting sentence has to QUOTE the claim it corrects, so the
        // paragraph recording the old "three Rocq files" wording would trip
        // this gate on its own history. Scoped past it by an explicit
        // historical marker, the same device the README layout gate uses to
        // scope itself to the fenced block.
        if sentence.contains("used to") {
            continue;
        }
        let Some(at) = sentence.find("Rocq file") else { continue };
        // The count word immediately before "Rocq file".
        let before = &sentence[..at];
        let Some(last) = before.split_whitespace().last() else { continue };
        let Some(&(_, want)) = words.iter().find(|(w, _)| *w == last.to_lowercase()) else {
            continue;
        };
        for k in &named {
            if !sentence.contains(k.as_str()) {
                continue;
            }
            checked += 1;
            let have = proofs_naming(k);
            assert_eq!(
                have.len(),
                want,
                "the tval doc says `{k}` carries {want} Rocq file(s); {} name \
                 it: {have:?}.\nsentence: {}",
                have.len(),
                sentence.trim()
            );
        }
    }
    assert!(
        checked >= 1,
        "no sentence attributes a count of Rocq files to a validated kernel, \
         so this gate examined nothing"
    );
}

// ---------------------------------------------------------------------------
// 4. The two device boundaries the proof refutes.
// ---------------------------------------------------------------------------

mod device {
    use super::*;
    use y::cuda_runtime::CudaContext;

    /// One launch of `exact_pv` at `b = q = d = 0`, `V` all ones.
    fn run(
        ctx: &CudaContext,
        ptx: &str,
        p: &[u32],
        v: &[i8],
        t: usize,
        d_dim: usize,
    ) -> i64 {
        let module = ctx.load_ptx(ptx, "exact_pv").expect("PTX failed to load");
        let d_p = ctx.alloc(p.len() * 4).unwrap();
        let d_v = ctx.alloc(v.len()).unwrap();
        let d_o = ctx.alloc(8).unwrap();
        let raw_p: Vec<u8> = p.iter().flat_map(|x| x.to_le_bytes()).collect();
        let raw_v: Vec<u8> = v.iter().map(|x| *x as u8).collect();
        ctx.memcpy_htod_at(&d_p, 0, &raw_p).unwrap();
        ctx.memcpy_htod_at(&d_v, 0, &raw_v).unwrap();
        ctx.memset_u8(&d_o, 0).unwrap();
        let args = vec![
            d_p.device_ptr(),
            d_v.device_ptr(),
            d_o.device_ptr(),
            t as u64,
            d_dim as u64,
            1u64,
            p.len() as u64,
            v.len() as u64,
            1u64,
        ];
        ctx.launch(&module, (1, 1, 1), (1, 1, 1), 0, &args)
            .expect("launch failed");
        ctx.synchronize().unwrap();
        let mut b = vec![0u8; 8];
        ctx.memcpy_dtoh_at(&mut b, &d_o, 0).unwrap();
        i64::from_le_bytes(b.try_into().unwrap())
    }

    /// **The finding, on the silicon.** With `D = 2^30` the V index `t*D`
    /// reaches `2^32` at exactly `t = 4` and wraps to 0 -- the index `t = 0`
    /// uses. `NV = 1`, so an unbounded-index machine reads `V[0]` once; this
    /// one reads it twice. `P[t] = t+1`, so the answer names which `t`
    /// contributed. One iteration wide: `T = 4` agrees, `T = 5` does not.
    ///
    /// This is what `ExactPvExact.the_measured_index_wrap_reads_v0_twice` and
    /// `the_measured_double_count_is_what_the_device_returned` reproduce from
    /// `wrap32` alone.
    #[test]
    fn the_index_ceiling_is_one_iteration_wide_on_the_device() {
        let Some(ctx) = CudaContext::new() else {
            eprintln!("SKIP: no CUDA driver -- the index boundary was not executed.");
            return;
        };
        let ptx = emit_ptx("idxdev");
        const D: usize = 1 << 30;
        let at = |t: usize| {
            let p: Vec<u32> = (0..t).map(|i| i as u32 + 1).collect();
            run(&ctx, &ptx, &p, &[1i8], t, D)
        };
        assert_eq!(
            at(4),
            1,
            "at T = 4 no index reaches 2^32, so the kernel must read V[0] once \
             and answer P[0] = 1"
        );
        assert_eq!(
            at(5),
            6,
            "at T = 5 the index t*D reaches 4*2^30 = 2^32, which wraps to 0. \
             The device is expected to re-read V[0] and answer P[0] + P[4] = 6. \
             If this is now 1, the index arithmetic has been widened and \
             ExactPvExact.v's index licence describes a kernel that no longer \
             exists -- which is a FIX, and the proof must be updated with it."
        );
    }

    /// The accumulator boundary, over the domain the emitted loads permit:
    /// every `P` at `2^32-1` and every `V` at `-128`. `T = 16,777,216` is
    /// exact and `16,777,217` wraps -- and the wrap flips the sign, which is
    /// what `the_measured_overflow_is_two_s_complement` reproduces from
    /// `wrap64` alone.
    #[test]
    fn the_accumulator_ceiling_is_one_iteration_wide_on_the_device() {
        let Some(ctx) = CudaContext::new() else {
            eprintln!("SKIP: no CUDA driver -- the accumulator boundary was not executed.");
            return;
        };
        let ptx = emit_ptx("accdev");
        const PER: i128 = (u32::MAX as i128) * 128;
        let at = |t: usize| {
            let p: Vec<u32> = vec![u32::MAX; t];
            let v: Vec<i8> = vec![-128i8; t];
            // D = 1, so the V index is `t` and every access is in range: this
            // isolates the ACCUMULATOR from the index bound above.
            run(&ctx, &ptx, &p, &v, t, 1)
        };
        let ok = 16_777_216usize;
        assert_eq!(
            at(ok) as i128,
            -(ok as i128) * PER,
            "at T = {ok} the total is inside int64 and must be exact"
        );
        let bad = ok + 1;
        let exact = -(bad as i128) * PER;
        let got = at(bad);
        assert_ne!(
            got as i128, exact,
            "at T = {bad} the total is outside int64, so the device cannot be \
             exact. If it now is, the accumulator has been widened."
        );
        assert_eq!(
            got, 9_223_371_489_246_445_696i64,
            "the wrapped value is the two's-complement one, and \
             ExactPvExact.the_measured_overflow_is_two_s_complement reproduces \
             exactly this number from wrap64 alone"
        );
    }
}
