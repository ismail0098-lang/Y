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
