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

/// Blank out everything inside `(* ... *)`, keeping every newline so line
/// numbers survive.
///
/// **Stripping comments LINE-LOCALLY is not enough, and that was found the way
/// these things are found: by writing a docstring.** The first version split
/// each line on `(*` and scanned the prefix, so the *opening* line of a comment
/// was handled and every continuation line was scanned as if it were code - a
/// paragraph containing the ordinary English word "admit" failed the gate.
/// Contorting the prose would have left the hole; Coq comments also NEST, so
/// the depth counter is not optional.
fn strip_coq_comments(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == '(' && i + 1 < b.len() && b[i + 1] == '*' {
            depth += 1;
            out.push(' ');
            out.push(' ');
            i += 2;
        } else if depth > 0 && b[i] == '*' && i + 1 < b.len() && b[i + 1] == ')' {
            depth -= 1;
            out.push(' ');
            out.push(' ');
            i += 2;
        } else {
            // Newlines are kept even inside a comment, so `lines()` still
            // agrees with the file.
            out.push(if depth > 0 && b[i] != '\n' { ' ' } else { b[i] });
            i += 1;
        }
    }
    out
}

/// `coqc` accepts an admitted lemma and exits 0. Only the source says.
#[test]
fn nothing_in_any_proof_is_admitted() {
    let mut offenders: Vec<String> = Vec::new();
    for path in proof_sources() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(&path).expect("read a proof");
        let stripped = strip_coq_comments(&src);
        for (i, (code, line)) in stripped.lines().zip(src.lines()).enumerate() {
            // Match the TOKEN, not a line shape. The previous version tested
            // `code == "Admitted."`, `starts_with("admit")` and
            // `contains(" admit.")`, which between them miss the commonest way
            // of stubbing a Coq lemma at all: `Proof. Admitted.` on ONE line is
            // none of the three. Found by probing the gate with the thing it
            // exists to catch, which is the only way a hole like this surfaces -
            // it passes every real file perfectly.
            //
            // `Abort.` is here for the same reason one level up: it discards the
            // lemma outright, so there is no theorem left to `Print Assumptions`
            // and the file still compiles.
            let holed = code
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .any(|t| t == "Admitted" || t == "admit" || t == "Abort" || t == "give_up");
            if holed {
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
            // GENERATED from `src/cpu_gemm.rs` by
            // `tests/exact_gemm_schedule_proof.rs`, which also gates it on
            // byte-identity. It is the one file here whose CONTENT is not
            // written by hand, and that is exactly why it takes no exemption
            // from this control.
            //
            // A definitions-only file would fail `every_proof_has_a_content_control`,
            // and that gate is right: "compiles" and "no axioms" are properties
            // an EMPTY file has, and a generator emitting nonsense emits it
            // confidently. So the generated file carries theorems, and two of
            // the three are STRUCTURAL - they constrain the shape of the
            // emitted expressions rather than restating their values, so they
            // are not made true merely by being generated alongside what they
            // describe.
            "ExactGemmSchedule.v",
            &[
                // The emitted `vpdpwssd` vector-group form of the B slot map
                // IS the plain interleave. The two sides come from different
                // places in `cpu_gemm.rs` - `pack_b_slot` and the bare `/2`
                // that `panel_slot_decode` inverts it with - so this is a
                // real check on the generator, not a restatement of a value.
                // The Rust asserts their agreement in a doc comment; this
                // proves it.
                "Print Assumptions slot_b_is_the_plain_interleave",
                // The constants are internally consistent. Catches a
                // generator that emitted `NR` and `NRV` from constants that
                // had stopped agreeing, or any degenerate zero.
                "the_tile_geometry_is_consistent",
                // The index arithmetic rendered from the SAME expressions
                // the emitter renders to LLVM (`cpu_gemm::tile_width_ix`,
                // `panel_index_ix`). Everything else in that file is
                // generated from a constant; these are generated from the
                // emitted CODE's arithmetic, and these two theorems are the
                // join to the tiling model's tile-index view.
                "Print Assumptions the_emitted_width_is_the_tiling_model_at_the_loop_variable",
                "the_emitted_panel_index_is_the_tile_index",
                // ...and the same for the two sites emitted as raw LLVM: the
                // micro-kernel's flush clamp, and the threaded wrapper's
                // K-split bands. Every theorem in ExactGemmKsplit.v is about
                // `blen`; the second of these is what says the emitted spawn
                // loop computes it.
                "the_emitted_tile_count_is_the_tiling_model",
                "the_emitted_chunk_end_is_the_flush_model",
                "the_emitted_band_length_is_the_ksplit_model",
                "the_emitted_a_index_is_the_pair_element",
                // The realizability constraint that was prose in a test
                // comment and stated nowhere: MR*NRV accumulators + NRV B
                // vectors + 1 A broadcast must fit 32 zmm. Measured by
                // sweeping VNNI_MR and reading real spill traffic - the cliff
                // is exactly where this flips. It makes the generator unable
                // to emit a schedule that does not fit the register file.
                "the_tile_fits_the_register_file",
                // A genuine cross-file join: every theorem in
                // `ExactGemmKsplit.v` is stated under `0 < nthr`, and nothing
                // proved the emitted thread count satisfies it - the floor
                // was argued in a comment.
                "ksplit_threads_is_never_zero",
                // The non-vacuity control, and it is honestly the weakest of
                // the four: under generation it is self-fulfilling. Its job is
                // the other direction - it makes the shipped values
                // load-bearing inside Coq, so a hand-edit of the committed
                // file fails `coqc` as well as failing byte-identity. That
                // matters most for `MR`, which the measurement recorded in
                // `ExactGemmComposition.v`'s header found was pinned by no
                // theorem in its own file.
                "the_schedule_is_the_shipped_one",
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
            "MixedRadix.v",
            &[
                // Positional indices, proved once. `quot_rem_unique` had TWO
                // independent copies - ExactGemmTiling and ExactGemmPacking,
                // neither requiring the other.
                "quot_rem_unique",
                "pack_unpack",
                "unpack_pack",
                // The leg that was rewritten in three files, and the one the
                // "it is only six lines" argument for a local copy did not
                // cover.
                "pack_onto",
                "pack_in_range",
                // The two-digit peel `pack_b_slot_bijective` was doing by hand.
                "two_digit_unique",
                // ...and that the bound on the low digit is load-bearing, on
                // the high one absent, rather than either being carried along.
                "the_low_digit_bound_cannot_be_dropped",
                "without_the_digit_bound_it_is_not_injective",
                "the_high_digit_needs_no_bound",
            ][..],
        ),
        (
            "Decomposition.v",
            &[
                // The partition obligation, proved once and instantiated by
                // all three kernels. Two theorems, not one: contiguous parts
                // are a re-bracketing and cost only associativity; an
                // arbitrary owner map reorders the terms and costs
                // commutativity as well.
                "contiguous_exact",
                "decomposition_exact",
                "parts_tile",
                "acc_parts_prefix",
                // ...and that the two really are different claims, or
                // "use the general one everywhere" would be free.
                "the_interleaved_split_is_not_contiguous",
                // The refutation every kernel used to carry its own copy of:
                // the theorems are about the ACCUMULATE, not the indices.
                // The clamped edge family, which two obligations turned out to
                // share: the int32 flush interval and the output tiling.
                "clamped_width",
                "widths_cover_the_extent",
                "rounding_breaks_a_contiguous_split",
                "rounding_breaks_an_interleaved_split",
                "exact_survives_the_same_split",
                "exact_survives_the_interleaved_split",
                "the_two_accumulates_differ_on_this_input",
            ][..],
        ),
        (
            "CountingSort.v",
            &[
                // The obligation the schema did not have: a decomposition
                // whose consequence is PLACEMENT, not a folded value.
                // `Decomposition.widths_cover_the_extent` says the widths add
                // up, which is strictly weaker - a decomposition writing one
                // slot twice and another never has exactly the right total.
                "slot_injective",
                "slot_onto",
                "every_slot_is_written_exactly_once",
                // ...and that it IS the same schema and not a rival to it:
                // the width-derived edges satisfy `Decomposition`'s three
                // hypotheses, and its fold theorem instantiates on them.
                "the_bucket_edges_are_a_decomposition_edge_function",
                "the_reduction_theorem_still_applies",
                // The claim a code comment in `scatter` had been making with
                // nothing checking it: the cursors ALONE - each group stopped
                // where the next one started, the last at `off[b+1]` - imply
                // the runs tile the bucket. It assumes nothing about the
                // histogram, which is what makes the runtime check worth its
                // 0.2-1.0 ms.
                "the_chained_runs_tile_the_bucket",
                "the_chained_runs_do_not_overlap",
                "a_chained_run_stays_in_the_bucket",
                // ...and both refutations that make the chaining hypothesis
                // load-bearing, stated as refutations of the WEAKENED theorem
                // rather than as witnesses, so no fixture drift can make them
                // vacuous.
                "without_the_chaining_a_slot_can_go_unwritten",
                "without_the_chaining_two_groups_can_write_one_slot",
                // The empty-group case, which is what `scatter`'s own
                // post-condition got wrong the first time it was written.
                "an_empty_group_still_chains",
                // Two levels - buckets outside, scatter groups inside. No
                // instantiation had composed two before, and the whole cost
                // of the second one is the single hypothesis below, which is
                // `scatter`'s own grouping assertion.
                "dest_injective",
                "dest_onto",
                "every_idx_slot_is_written_exactly_once",
                "the_group_widths_exhaust_every_bucket",
                // `MixedRadix` does not cover it, and that is a property of
                // the decomposition rather than of how it is written: the
                // inner extent is the bucket's own width, and bucket widths
                // differ because they count data.
                "no_uniform_radix_describes_this",
                // Concrete, because every theorem above is satisfied by a
                // histogram that counted nothing and by one bucket with one
                // group.
                "the_msm_instance_is_not_vacuous",
            ][..],
        ),
        (
            "AttentionSchedule.v",
            &[
                // GENERATED from the emitter's own launch-geometry
                // expressions. The bijection is the whole obligation: every
                // thread is a worker the reduction folds, no two share an id,
                // and no id is unclaimed.
                "the_worker_is_below_the_worker_count",
                "distinct_threads_are_distinct_workers",
                "every_worker_is_some_thread",
                // The pairing is load-bearing, stated by refuting the
                // weakened claim rather than by exhibiting a collision.
                "dropping_the_z_extent_overflows_the_worker_count",
                // ... and the control that stops that refutation being read
                // as "attn_scores is wrong": at one CTA in z the two
                // schedules are the same map, so one proof covers both.
                "the_scores_schedule_is_the_accumulate_schedule_at_one_z",
            ],
        ),
        (
            "GridStrideSplit.v",
            &[
                // The third kernel, and the first that is not a GEMM: the
                // residue classes partition the sequence, so any worker count
                // gives the naive sum.
                "grid_stride_exact",
                "stride_classes_partition",
                "any_worker_count_agrees",
                // The instantiation itself: this kernel's classes ARE the
                // schema's parts at the owner map `i |-> i mod n`.
                "combine_is_decomposition",
                // The property neither GEMM proof needed. The atomics supply
                // their operands in whatever order the hardware chooses, so
                // this one needs commutativity and not just associativity.
                "atomics_may_land_in_any_order",
                // The join to the emitted kernel. Everything else in that
                // file is stated for an ABSTRACT worker count; this is the
                // one theorem that says the compiler emits one of them.
                "the_emitted_launch_geometry_visits_every_key_exactly_once",
                "every_folded_worker_is_a_real_thread",
                // Both halves are about the ACCUMULATE - and the second is a
                // failure the GEMM kernels cannot exhibit at all, since they
                // fold their bands in index order.
                "rounding_breaks_the_stride_split",
                "rounding_is_order_dependent",
                "exact_is_order_independent",
                // The accumulator ceiling, which needs 2.7e8 keys to reach and
                // so can never be demonstrated on a device.
                "the_bound_is_one_unit_wide",
            ][..],
        ),
        (
            // Phase 4's first theorem, and the first BOUND in this directory -
            // every other file states an exactness or a coverage claim.
            "SoftmaxErrorBound.v",
            &[
                // The headline: the emitted chain's output against the ideal
                // softmax, and the same expression evaluated at a long context.
                "the_attention_output_is_within_the_bound",
                "Print Assumptions the_attention_output_is_within_the_bound",
                "the_output_quotient_is_within_the_bound",
                "the_bound_at_a_long_context",
                // The three joints, each of which was covered by nothing. The
                // exp itself is NOT one of them - it is discharged
                // exhaustively in Rust, which is stronger than a proof here.
                "an_unsaturated_argument_is_within_a_half_ulp",
                "the_weight_is_within_a_relative_ulp",
                "the_max_subtraction_puts_a_floor_under_the_total_weight",
                // The domain gap: the exhaustive sweep stops at 31<<16 and the
                // emitted saturate admits 2^30. Two lines, and they existed
                // nowhere.
                "the_swept_domain_covers_the_admitted_one",
                "saturation_means_the_exact_exponent_is_astronomically_large",
                // The refutation that makes the max subtraction load-bearing
                // rather than decorative: without its floor the same bracket
                // admits an all-zero denominator.
                "a_total_weight_below_one_ulp_admits_a_zero_denominator",
                // The answer to "does the Decomposition schema extend to
                // approximate arithmetic": the reduction is still EXACT, so
                // GridStrideSplit applies verbatim and the per-element error
                // enters the fold as data.
                "the_bound_holds_at_every_launch_geometry",
                // A hypothesis nothing is shown to satisfy is the proof-shaped
                // version of a licence nothing can violate. Both directions:
                // the obvious interface is INCONSISTENT over Q, and the one
                // used here has a model.
                "exact_homogeneity_is_unsatisfiable_over_Q",
                "no_rational_squares_to_two",
                "the_interface_is_satisfiable",
                // ...and the constant the bracket turns on is not magic.
                "the_per_unit_factor_is_what_one_unit_of_log2_costs",
                // The TEMPERATURE is itself quantized, and the shape of that
                // error is what makes it a head_dim question rather than a
                // KFix one: rounding `C * 2^32` moves the exponent by at most
                // half a score delta, INDEPENDENTLY of KFix.
                "the_quantized_temperature_shifts_the_exponent_by_at_most_half_a_delta",
                "the_ideal_weights_at_two_close_exponents_bracket_each_other",
                "at_head_dim_128_the_temperature_moves_a_weight_by_under_a_two_thousandth",
                // ...and the two multipliers that are not approximations of
                // anything. Both were a bare `continue` in a Python script.
                "a_zero_multiplier_gives_every_key_the_same_weight",
                "the_two_readings_of_the_multiplier_disagree_above_two_to_the_thirty_one",
                "the_signed_reading_zeroes_a_weight_the_unsigned_one_keeps",
            ][..],
        ),
        (
            "GemmBandSplit.v",
            &[
                // The f32 kernel is the SECOND kernel, and its K-split is a
                // different decomposition. The tiling obligation composed; its
                // proof did not transfer.
                "prop_ksplit_exact",
                "prop_bands_tile",
                // The measurement the schema exists to produce: the M/N split
                // had edge theorems and no exactness theorem, because writing
                // the reduction out a third time was not worth it. Under
                // `Decomposition` it costs one application and the three edge
                // facts already proved.
                "granule_split_exact",
                // ...and that it really is a different partition, or the
                // theorem above would be the exact kernel's wearing a hat.
                "the_two_splits_are_different",
                // The two last-band clamps the emitter writes are redundant -
                // the property that was a comment in `emit_entry`.
                "pedge_last",
                "gedge_last",
                // A band boundary inside a tile would make one thread write a
                // partial tile; also only a comment before.
                "every_edge_snaps_to_a_granule_or_the_extent",
                // The exactness obligation provably does NOT transfer, at the
                // same f/K/nthr as the exact kernel's own refutation, with the
                // control that says the failure is the accumulate's.
                "rounding_breaks_the_proportional_split_too",
                "exact_survives_the_proportional_split",
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
                // The fold-back loop, whose BOUND carries the obligation: it
                // runs exactly the clamped tile width, so the copy into C
                // never leaves the rectangle the tile owns.
                "the_emitted_fold_back_runs_the_tile_width",
                "the_fold_back_stays_inside_the_live_rectangle",
                // ...with the trip count doing the work, one unit wide.
                "the_fold_back_trip_count_is_what_keeps_it_in_bounds",
                            // The FIRST tie between a proof and the SHAPE of an emitted
                // loop rather than the value of an emitted expression:
                // `SCH.row_panel_*` is rendered from `cpu_gemm::CountedLoop`,
                // the description the driver opens its loops from.
                "the_emitted_row_loop_enumerates_the_tiles",
                "the_emitted_row_loop_runs_once_per_tile",
                "the_emitted_column_loop_enumerates_the_tiles",
                "the_emitted_column_loop_runs_once_per_tile",
                "the_two_panel_loops_are_different",
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
                // The panel as a FUNCTION - what slot `s` actually holds. This
                // is the statement the composition needed and had to assume;
                // the uniqueness theorem is what makes it a claim about the
                // emitted loop rather than about a convenient model.
                "Print Assumptions panel_decodes_its_own_write",
                "panel_is_the_only_solution",
                // The group bound refuted concretely: without `idx < width` the
                // contract is FALSE of every non-zero operand, because the next
                // index is the following k-pair group's first slot. The control
                // shows the same panel agrees inside the group.
                "the_group_bound_is_load_bearing",
                "inside_the_group_the_same_panel_agrees",
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
        (
            "ExactGemmRegisterTile.v",
            &[
                // THE routing theorem: lane l of vector v consumes exactly the
                // two packed slots of the column the store sends it to.
                "Print Assumptions the_lane_consumes_its_own_column",
                // The i32 load of A aliases the packed pair...
                "the_i32_load_is_the_packed_pair",
                // ...and the endianness that rests on is load-bearing, with a
                // control showing a symmetric operand would hide it.
                "swapping_the_pair_halves_computes_a_different_function",
                "a_symmetric_operand_hides_the_swap",
                // The 24 accumulators of 16 lanes tile the 64 columns.
                "tile_position_injective",
                "tile_position_surjective",
                // The masked tails, and why the packers' row/column masks are
                // redundant while the phantom k-half's is not.
                "only_the_live_rectangle_is_stored",
                "a_padding_column_never_reaches_c",
            ][..],
        ),
        (
            "ExactGemmComposition.v",
            &[
                // The five files must AGREE on the definitions they each
                // declared separately - three copies of the B slot map, two of
                // MR/NR/col_of. Each copy turns out to be pinned by a theorem
                // in its own file already (that was measured, not assumed), so
                // these make an incidental pinning explicit and cross-file.
                "packing_and_register_tile_agree_on_the_b_slot",
                "micro_and_register_tile_agree_on_the_b_slot",
                "the_tile_shape_is_the_same_everywhere",
                "the_agreement_is_not_vacuous",
                // Packing says what a slot holds, the register tile says which
                // lane reads it; together, the lane reads the right source
                // elements. Neither half can state this alone.
                "Print Assumptions the_lane_accumulates_the_source_elements",
                "a_dead_row_contributes_nothing",
                // ...and the packers' contract is DISCHARGED rather than
                // assumed. Deleting this leaves the composition resting on a
                // hypothesis nothing is shown to satisfy - which is the
                // proof-shaped version of a licence nothing can violate, and
                // is the state this file was committed in once.
                "the_packed_panels_route_to_the_right_source_elements",
                            // The two obligations that turned out to be one decomposition:
                // an int32 overflow budget and a memory partition, both
                // `t |-> min(t*X, ext)`, spelled as an END in one emitter site
                // and as a WIDTH in the other.
                "the_flush_interval_and_the_output_tile_are_the_same_family",
                "the_agreement_covers_the_ragged_case",
][..],
        ),
        (
            "ExactGemmChain.v",
            &[
                // THE headline: the emitted flush schedule, accumulating each
                // chunk in int32, over the panels the emitted packers produce,
                // through the emitted routing, computes the SOURCE dot product
                // for one accumulator lane.
                "Print Assumptions the_emitted_lane_computes_the_source_dot_product",
                // The two joins that are the file's own new content. The
                // sibling models were written independently and it was not
                // obvious they would meet: sum_pairs counts k-PAIRS with both
                // halves, sum_from is a flat range.
                "kloop_is_the_padded_product",
                "kloop_is_sum_from_step",
                "the_k_pair_loop_computes_the_source_dot_product",
                "the_flush_schedule_computes_the_source_dot_product",
                // The masking survives to loop scale rather than being
                // re-argued per layer.
                "a_dead_row_accumulates_nothing",
                "a_dead_column_accumulates_nothing",
                // The licence made load-bearing for the WHOLE chain, with the
                // one-unit boundary and its control. Without these the chain
                // is a theorem about an idealised Z accumulator.
                "violating_the_licence_breaks_the_chain",
                "at_the_licensed_magnitude_the_chain_holds",
                // ...and the chain evaluated on concrete numbers, because
                // every equality above is satisfied by a model computing 0.
                "the_whole_chain_is_not_vacuous",
                // The tile lift: the same statement over the whole MR x NR
                // rectangle, with the ACCUMULATE visible (C0 + dot) and dead
                // positions left exactly as they were.
                "Print Assumptions the_tile_holds_the_source_dot_products",
                "a_dead_position_leaves_c_untouched",
                // The join it needs: RT.tile_position_surjective says an
                // inverse of col_of exists, this names it and proves it
                // inverts BOTH ways - which is what licenses "no two tile
                // positions share an accumulator lane".
                "the_lane_map_is_a_two_sided_inverse",
                "distinct_columns_use_distinct_lanes",
                "the_tile_lift_is_not_vacuous",
                // The clamp does no work: a dead position accumulates zero by
                // the packers' masks, so the tile theorem describes the
                // micro-kernel's own effect on C. The proved twin of the
                // redundancy exact_gemm_packing_model.rs measured.
                "the_store_predicate_is_redundant",
            ][..],
        ),
        (
            "ExactGemmWhole.v",
            &[
                // THE capstone: all six exact-GEMM proofs chained. Every
                // element of C, over every thread's K band, is the source dot
                // product - no hypothesis that MR divides M, NR divides N,
                // nthr divides K, or that K is even.
                "Print Assumptions the_threaded_gemm_holds_the_source_dot_products",
                "the_whole_output_holds_the_source_dot_products",
                // Every position is live in its OWN tile, which is what makes
                // the store predicate true for it.
                "the_position_is_live_in_its_own_tile",
                // ...and the (r, c) view addresses the same element the tiling
                // proof's (tile, offset) view does, so
                // ExactGemmTiling.c_written_exactly_once applies here.
                "the_position_decomposition_is_the_tilings",
                // Concrete, because every equality above is satisfied by a
                // model computing 0 - and with BOTH bands contributing, so the
                // K-split is a real reduction rather than one band doing all
                // the work.
                "the_whole_chain_is_not_vacuous",
                "both_bands_contribute",
                // The scratch tile starts at ZERO - a hypothesis `gemm_position`
                // hid inside a definition, now tied to the emitted `vg.z` loop.
                // `%Ctile` is reused across tiles, so one trip short carries the
                // previous tile's accumulator into this one.
                "the_scratch_is_zeroed_wherever_the_fold_back_reads",
                "the_zeroing_loop_covers_exactly_the_tile",
                "one_trip_short_leaves_the_corner_stale",
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
                names_something_real(&src, needle),
                "proofs/{name} no longer STATES `{needle}`, which is what makes \
                 the file worth checking. It is gone or renamed; point this gate \
                 at whatever replaced it rather than deleting the entry. \
                 (A mention in the file's header prose does not count - see \
                 `names_something_real`.)"
            );
        }
    }
}

/// A needle must appear at a **declaration site**, not merely somewhere in the
/// file.
///
/// This used to be a bare `src.contains(needle)`, and that is satisfied by the
/// theorem's own name in the header doc comment - which every proof here has,
/// because each file's header explains what it proves and refers to its
/// theorems as `[the_name]`. Measured rather than reasoned about: deleting
/// `a_total_weight_below_one_ulp_admits_a_zero_denominator` from
/// `SoftmaxErrorBound.v` **together with its `Print Assumptions` line** left
/// all four tests in this file green, because the header still named it. The
/// `Print Assumptions` line has to go too or `coqc` catches the dangling
/// reference; with both gone, nothing did.
///
/// It is a hole in the gate that guards the whole directory, not in one entry,
/// and it is pre-existing - it was found by mutating a new file and applies to
/// all seventeen.
fn names_something_real(src: &str, needle: &str) -> bool {
    // `Print Assumptions foo` IS the site it names; match it literally.
    if needle.starts_with("Print Assumptions ") {
        return src.contains(needle);
    }
    const KEYWORDS: [&str; 9] = [
        "Theorem", "Lemma", "Corollary", "Definition", "Fixpoint", "Example",
        "Remark", "Fact", "Proposition",
    ];
    // ...and a bare name is also satisfied by its own `Print Assumptions`,
    // which cannot survive the theorem being deleted.
    if src.contains(&format!("Print Assumptions {needle}")) {
        return true;
    }
    KEYWORDS.iter().any(|kw| {
        src.match_indices(&format!("{kw} {needle}")).any(|(i, m)| {
            // The declaration must start a line, and the name must end there:
            // `Lemma foo_bar` must not satisfy a needle of `foo`.
            let starts_line = i == 0 || src.as_bytes()[i - 1] == b'\n';
            let after = src.as_bytes().get(i + m.len()).copied().unwrap_or(b' ');
            starts_line && !(after.is_ascii_alphanumeric() || after == b'_' || after == b'\'')
        })
    })
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
