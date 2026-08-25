//! The Coq proofs are part of the product, so they have to be part of the suite.
//!
//! `proofs/ZkControlFlow.v` was for a long time the one machine-checked
//! artifact in this repo, and it was run by a command in CLAUDE.md and by
//! **nothing else** - no test, no build script, no CI. That is the shape this
//! repo already catalogues under other names: `@ZeroDrift` before it was
//! implemented, the `scheme = "plonkish"` flag, the `wgmma...s4` that existed
//! on no hardware. An unrun proof is the paperwork of a proof.
//!
//! It matters more here than for an ordinary test because a proof is about a
//! MODEL of a lowering rather than about the Rust. The model can drift from the
//! code silently; the least this gate can do is guarantee it still type-checks
//! and still rests on nothing.
//!
//! **This gate sweeps `proofs/` rather than naming one file.** The first
//! version named `ZkControlFlow.v`, which is exactly how the second proof would
//! have gone unrun - the same mistake one level up. Adding a `.v` file now puts
//! it under all three checks automatically; only its content control has to be
//! written by hand, and `every_proof_has_a_content_control` refuses to let that
//! be skipped.
//!
//! Three claims per file, and the third is the one that rots quietly:
//!
//! 1. `coqc` accepts it.
//! 2. Every `Print Assumptions` reports `Closed under the global context` -
//!    counted against the number the SOURCE asks for, so deleting a
//!    `Print Assumptions` line cannot quietly reduce what is checked. An
//!    `Axioms:` section appearing would mean a proof now depends on an
//!    assumption nobody stated.
//! 3. Nothing is `Admitted`. `coqc` accepts an admitted lemma happily and
//!    prints a warning that no exit code reflects.
//!
//! Skipped with a printed notice when `coqc` is absent, like the `ptxas`,
//! `solcjs` and `z3` gates. That is a real hole - CI without Rocq is not
//! checking this - and it is stated rather than hidden.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn proofs_dir() -> PathBuf {
    repo().join("proofs")
}

/// Every `.v` in `proofs/`, sorted so a run is reproducible.
fn proof_sources() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(proofs_dir())
        .expect("read proofs/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "v").unwrap_or(false))
        .collect();
    v.sort();
    v
}

fn have_coqc() -> bool {
    Command::new("coqc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Compile every proof in a private directory.
///
/// Not in `proofs/` on purpose: `coqc` writes `.vo`/`.glob`/`.vok`/`.vos`
/// beside the source, so compiling in place would race a developer running
/// `coqc` by hand and would litter a tracked directory. The `.ptx` race in the
/// GPU harness was exactly this, and it presented as an intermittent failure.
///
/// All sources are copied in together and the loop retries, so one proof may
/// `Require` another without this gate having to know the order.
fn compile_all_proofs() -> Vec<(String, bool, String)> {
    let dir = std::env::temp_dir().join(format!("y_coq_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let sources = proof_sources();
    for src in &sources {
        let dst = dir.join(src.file_name().unwrap());
        std::fs::copy(src, &dst).expect("copy a proof");
    }

    let mut pending: Vec<String> = sources
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let mut done: Vec<(String, bool, String)> = Vec::new();

    // At most one pass per file is enough for any acyclic Require graph.
    for _ in 0..pending.len().max(1) {
        let mut still: Vec<String> = Vec::new();
        let mut progressed = false;
        for name in pending.drain(..) {
            let out = Command::new("coqc")
                .args(["-Q", ".", ""])
                .arg(&name)
                .current_dir(&dir)
                .output()
                .expect("run coqc");
            let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
            text.push_str(&String::from_utf8_lossy(&out.stderr));
            if out.status.success() {
                progressed = true;
                done.push((name, true, text));
            } else {
                still.push(name.clone());
                done.push((name, false, text));
            }
        }
        if still.is_empty() || !progressed {
            break;
        }
        // Retry the failures; drop their earlier (failed) rows first.
        done.retain(|(n, ok, _)| *ok || !still.contains(n));
        pending = still;
    }

    let _ = std::fs::remove_dir_all(&dir);
    done
}

#[test]
fn every_coq_proof_still_checks_and_rests_on_no_axioms() {
    if !have_coqc() {
        eprintln!("skipping: no coqc on PATH - the proofs are NOT being checked");
        return;
    }
    let results = compile_all_proofs();
    assert!(
        !results.is_empty(),
        "no `.v` files found in proofs/ - this gate would pass perfectly while \
         checking nothing"
    );

    for (name, ok, output) in &results {
        assert!(
            ok,
            "`coqc proofs/{name}` failed. The proofs are documented commands and \
             the repo's only machine-checked artifacts.\n{output}"
        );

        // `Print Assumptions` prints either `Closed under the global context` or
        // an `Axioms:` section listing what the proof leans on. Count what the
        // SOURCE asks for rather than hardcoding a number here, so deleting a
        // `Print Assumptions` line is caught by the content control below and
        // cannot silently shrink what this check covers.
        let src = std::fs::read_to_string(proofs_dir().join(name)).expect("read a proof");
        let asked = src
            .lines()
            .filter(|l| l.trim_start().starts_with("Print Assumptions"))
            .count();
        assert!(
            asked >= 1,
            "proofs/{name} has no `Print Assumptions`, so nothing checks what it \
             rests on"
        );
        let closed = output.matches("Closed under the global context").count();
        assert_eq!(
            closed, asked,
            "proofs/{name}: the source asks for {asked} `Print Assumptions` and \
             {closed} reported `Closed under the global context`. Output:\n{output}"
        );
        assert!(
            !output.contains("Axioms:"),
            "proofs/{name} now depends on an axiom. These theorems are supposed \
             to hold unconditionally - an assumption here means each is weaker \
             than its statement reads.\nOutput:\n{output}"
        );
    }
}

/// `coqc` accepts an admitted lemma and exits 0. Only the source says.
#[test]
fn nothing_in_any_proof_is_admitted() {
    let mut offenders: Vec<String> = Vec::new();
    for path in proof_sources() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(&path).expect("read a proof");
        for (i, line) in src.lines().enumerate() {
            let code = line.split("(*").next().unwrap_or("").trim();
            // `Admitted.` closes a proof with a hole; `admit.` leaves one mid-proof.
            if code == "Admitted." || code.starts_with("admit") || code.contains(" admit.") {
                offenders.push(format!("{name}:{}: {}", i + 1, line.trim()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "an admitted proof is the paperwork of a proof without the proof:\n  {}",
        offenders.join("\n  ")
    );
}

/// The content controls. Both tests above pass perfectly against a file
/// containing no theorems at all - "no axioms" and "nothing admitted" are
/// properties an EMPTY file has. Same shape as
/// `the_artifact_sweep_actually_finds_artifacts` and
/// `ordinary_loop_bodies_still_verify`.
///
/// Each entry names the load-bearing theorem AND its refutation counterpart:
/// proving the new thing right without proving the old thing wrong is what let
/// the flat-vs-nested distinction survive Z3 and a fresh test file.
fn content_controls() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        (
            "ZkControlFlow.v",
            &[
                // The shipped lowering agrees with the operational semantics.
                "low_correct",
                "Print Assumptions low_correct",
                // The machine-checked refutation of the lowering it replaced.
                "low_tail_is_wrong_when_nested",
            ][..],
        ),
        (
            "ExactGemmKsplit.v",
            &[
                // The K-split reduction equals the naive nest, every K, every nthr.
                "ksplit_exact",
                "Print Assumptions ksplit_exact",
                // The bands cover [0, K) exactly - the obligation a dropped
                // remainder violates.
                "bands_tile",
                // ...and that mutation refuted concretely.
                "dropping_the_remainder_loses_terms",
                // The roadmap's central claim: the SAME schedule is wrong under
                // a rounding accumulate and right under an exact one.
                "rounding_breaks_the_split",
                "exact_survives_the_same_split",
            ][..],
        ),
        (
            "ExactGemmTiling.v",
            &[
                // The output tiles account for the axis exactly...
                "tiles_cover",
                // ...and, the real obligation, name each position ONCE.
                "tile_index_injective",
                "tile_index_surjective",
                "Print Assumptions c_written_exactly_once",
                // The emitter's own comment about the ragged tail, as a
                // machine-checked refutation rather than a comment.
                "unclamped_tail_writes_out_of_bounds",
                // ...and the precondition nothing in the compiler states.
                "row_stride_below_n_aliases",
            ][..],
        ),
        (
            "ExactGemmPacking.v",
            &[
                // Both packers' destination maps are bijections onto their panel.
                "pack_a_slot_bijective",
                "pack_b_slot_bijective",
                // THE theorem: the full padded product equals the live dot
                // product, which is what licenses running a ragged tile at full
                // width.
                "Print Assumptions padded_product_is_the_live_dot_product",
                // ...and the zero-fill that makes it true, refuted concretely.
                "garbage_in_the_pad_changes_the_answer",
                "the_masked_version_agrees_on_the_same_input",
                // The precise statement of what the packing proof CANNOT say
                // about lane layout: the emitted vector-group form IS the
                // plain interleave, so there is no arithmetic difference for
                // any proof to capture. Deleting this turns the file's own
                // "what this does not prove" section into an unchecked remark.
                "slot_b_is_the_plain_interleave",
            ][..],
        ),
        (
            "ExactGemmMicro.v",
            &[
                // The flush chunks sum to the whole k-pair range, with no
                // hypothesis that the interval divides it.
                "Print Assumptions flush_exact",
                // The int32 accumulator agrees with Z exactly when nothing
                // leaves the range...
                "flush_exact_in_int32",
                // ...and the licence's own bound is what supplies that.
                "operand_bound_gives_no_overflow",
                "the_licence_makes_the_chunk_exact",
                // The overflow refuted concretely rather than assumed, and the
                // boundary pinned at one unit wide.
                "overflow_breaks_the_flush",
                "the_4096_case_exceeds_by_exactly_one",
                // The cross-file tie: the column pack_b routes to a lane is the
                // column the store reads that lane back out to.
                "the_packed_column_is_the_stored_column",
                "a_wrong_lane_stride_permutes_the_columns",
            ][..],
        ),
    ]
}

#[test]
fn each_proof_still_proves_the_thing_it_exists_for() {
    for (name, required) in content_controls() {
        let path = proofs_dir().join(name);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read proofs/{name}: {e}"));
        for needle in required {
            assert!(
                src.contains(needle),
                "proofs/{name} no longer contains `{needle}`, which is what makes \
                 the file worth checking. It is gone or renamed; point this gate \
                 at whatever replaced it rather than deleting the entry."
            );
        }
    }
}

/// A new `.v` file must arrive with a content control, or it gets the two weak
/// checks (compiles, no axioms) that an empty file also passes.
#[test]
fn every_proof_has_a_content_control() {
    let controlled: Vec<&str> = content_controls().iter().map(|(n, _)| *n).collect();
    let missing: Vec<String> = proof_sources()
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .filter(|n| !controlled.contains(&n.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "these proofs have no content control, so they would pass this suite \
         while containing no theorems: {missing:?}. Add each to \
         `content_controls()` naming its load-bearing theorem."
    );
}
