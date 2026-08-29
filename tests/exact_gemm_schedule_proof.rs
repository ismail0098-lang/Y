//! The exact-GEMM schedule has ONE source, and this gate is what makes that
//! structural rather than remembered.
//!
//! # What was wrong
//!
//! Phase 1 of `docs/proof_carrying_kernels.md` is nine proofs in `proofs/`,
//! and the schedule they are about existed in three places:
//!
//! - `src/cpu_gemm.rs`, where the emitter reads it;
//! - the proof files, which each declared their own copy - `MR` and `NR` twice,
//!   `col_of` twice, the B slot map three times;
//! - `proofs/ExactGemmComposition.v`, which asserts the copies agree.
//!
//! That third one is a theorem somebody remembered to write. Nothing forced it
//! to exist, and nothing at all connected any of it to the Rust.
//!
//! # What was MEASURED first, and it corrects a claim this repo had written down
//!
//! `ExactGemmComposition.v`'s header says each duplicated definition "turns out
//! to be pinned by a theorem in its own file", from three attempts:
//! `RegisterTile.slot_b` shifted by one, its `NR` set to 32, `Micro.slot` with
//! the halves swapped. Re-running those three confirms them, and two more
//! (`RegisterTile.NRV` 4 -> 2, `Packing.NR` 64 -> 32) behave the same way.
//!
//! **`MR` is the exception, and it was never tried.** Setting
//! `ExactGemmRegisterTile.MR` or `ExactGemmPacking.MR` to 8 leaves that file
//! compiling perfectly. Both are caught only by `ExactGemmComposition.v` and by
//! the downstream chain - that is, by the remembered theorem, not by anything
//! intrinsic. So the header's claim is true of five definitions out of six and
//! false of the sixth. It is corrected in place rather than left standing.
//!
//! And in the Rust -> Coq direction nothing was pinned at all: `VNNI_MR` could
//! move to 8 and every proof would go on describing a 6-row tile.
//!
//! # What this is
//!
//! `proofs/ExactGemmSchedule.v` is GENERATED from `cpu_gemm.rs`'s own constants
//! and committed. Every sibling proof takes its constants and index maps from
//! it. This gate regenerates the file and fails on byte-identity, exactly like
//! `tests/committed_ptx_artifacts.rs` for `.ptx` and
//! `tools/extract_poseidon.py` for `src/zk_poseidon_constants.rs`.
//!
//! # Why a `#[test]` and not a `[[bin]]` or a `tools/` script
//!
//! **The generator must LINK, not PARSE.** A `tools/` Python script would have
//! to recover `VNNI_MR` with a regex over `cpu_gemm.rs` - a fourth copy of the
//! value, living in the generator, which is the bug rather than the fix. The
//! `extract_poseidon.py` precedent parses because its input is FOREIGN
//! (circomlib's `.circom`); this input is our own Rust and can be `use`d.
//!
//! That leaves `[[bin]]` or `#[test]`. A fifth `[[bin]]` is a permanent user
//! surface - `tests/source_surface.rs` gates those, and the README documents
//! them - bought for a file regenerated perhaps twice a year. A `#[test]` links
//! the same way, costs no surface, and puts the regeneration and the check in
//! one file where they cannot drift apart. There is no `build.rs` to use.
//!
//! Regenerate with:
//!
//! ```text
//! Y_REWRITE_SCHEDULE_PROOF=1 cargo test --release --test exact_gemm_schedule_proof
//! ```
//!
//! # What this closes, and what it does not
//!
//! It closes CONSTANT drift. It does not close the gap
//! `docs/proof_carrying_kernels.md` names about itself: the loop NEST is still
//! hand-written `IrBuilder` calls and the tie is still between two models. That
//! is Phase 2.
//!
//! # Mutation table
//!
//! Measured over all thirteen exact-GEMM suites, each `--test` target run
//! SEPARATELY (`cargo test` aborts the remaining binaries after one fails, so
//! a combined run silently leaves later targets unmeasured).
//!
//! | mutation | this file | proofs_are_checked | the other 11 |
//! |---|---|---|---|
//! | Rust `VNNI_MR` 6 -> 8 | **FAIL** | ok | 6 FAIL, see below |
//! | Rust `KSPLIT_MIN_BAND` 128 -> 256 | **FAIL** | ok | all ok |
//! | committed `.v` hand-edited, `MR := 8` | **FAIL** | **FAIL** | all ok |
//! | `ExactGemmRegisterTile.NR := SCH.MR` | ok | **FAIL** | all ok |
//! | `render` echoes the committed file | **FAIL** | ok | all ok |
//! | `render` hardcodes `MR := 6` | **FAIL** | ok | all ok |
//!
//! **THE `VNNI_MR` ROW WAS MISATTRIBUTED WHEN FIRST RECORDED, AND DIAGNOSING
//! IT IS WHERE `the_tile_fits_the_register_file` CAME FROM.** "6 model suites
//! FAIL" reads as "six suites detect the schedule drift". They do not.
//!
//! `exact_gemm_thread_invariance` - the suite that runs the real threaded
//! kernel at ragged shapes against an independent integer reference - **passes
//! at MR = 8**. The kernel is not wrong. What failed was seven harnesses that
//! each embed a C driver hardcoding `#define MR 6` / `#define NR 64` while
//! their Rust half reads the crate constant, so moving `VNNI_MR` made each
//! test disagree with ITSELF: a 48-element panel against a 64-element
//! expectation, and a child process crashing on buffers sized for the wrong
//! tile. The same defect this file exists to remove, one layer down, in the
//! half of the harness that allocates the memory the kernel writes into.
//!
//! All seven take the constants from `cpu_gemm.rs` now, and with that fixed
//! six of the eight PASS at MR = 8. What still fails is the real constraint -
//! `cpu_gemm_vnni_micro::the_hot_loop_does_not_spill_the_accumulators`, whose
//! own comment already stated the register budget in prose. That is what
//! `the_tile_fits_the_register_file` promotes into a checked statement, with
//! the bound's FORM taken from a sweep of real spill traffic rather than
//! guessed.
//!
//! Three of the rows are worth reading rather than counting.
//!
//! **`KSPLIT_MIN_BAND` is caught by NOTHING else.** That is this gate's
//! clearest single justification: the constant is transcribed into
//! `ExactGemmSchedule` and into the emitted wrapper, and no model test in the
//! repo compares them.
//!
//! **A hand-edited `.v` is caught only here and by `coqc`** - every model test
//! passes, because they drive the Rust and never read the proofs. That is the
//! Coq-side half of the drift, and before this file nothing covered it at all.
//!
//! **The alias mistake is correctly NOT caught here**, and that is a
//! confirmation rather than a hole. `RegisterTile.NR := SCH.MR` leaves the
//! generated file untouched, so byte-identity has nothing to say; it is
//! `ExactGemmComposition.the_tile_shape_is_the_same_everywhere` that fires.
//! Those agreement theorems changed character when the definitions were
//! centralised - they now assert that each file ALIASES the right thing rather
//! than that two independent definitions coincide - and this mutation is what
//! demonstrates the weaker claim is still a real one.
//!
//! One mutation SURVIVED and was sorted rather than recorded: making
//! `Schedule::shipped` read `mr: 6` instead of `mr: VNNI_MR`. It changes
//! nothing observable, because the two are the same number - so it is
//! MIS-AIMED, not a hole. The discriminating experiments confirm it: hardcode
//! `6` **and** move `VNNI_MR` to 8, or read the wrong constant outright
//! (`mr: VNNI_NRV`), and `the_shipped_schedule_is_the_rust_constants` fails in
//! both. A hardcode that agrees today is caught the moment it stops agreeing,
//! which is the property that matters.

use std::path::PathBuf;

use y::cpu_gemm::{
    a_i32_element_ix, a_row_base_ix, band_base_ix, band_len_ix, band_rem_ix, chunk_end_ix,
    granule_band_edge_ix, kpairs_ix, panel_index_ix, prop_band_edge_ix, tile_count_ix,
    tile_width_ix,
    KSPLIT_MAX_THREADS, KSPLIT_MIN_BAND, VNNI_MR, VNNI_NR, VNNI_NRV,
};
use y::zero_drift::VnniExact;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every constant the generated proof carries.
///
/// The generator is a function OF THIS, not of the Rust constants directly,
/// and that is not decoration - it is what makes the control below able to
/// fail. A byte-identity gate whose generator quietly starts echoing the
/// committed file passes forever while checking nothing, and a control that
/// only asks "does the output contain `Definition MR : nat := 6.`" passes too,
/// because the committed file contains exactly that. **That mutation survived
/// the first version of this file.** Rendering a PERTURBED schedule and
/// requiring the result to differ from the committed text is what catches it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Schedule {
    mr: usize,
    nrv: usize,
    lanes: usize,
    nr: usize,
    vec_elems: usize,
    flush: usize,
    minband: usize,
    maxthr: usize,
}

impl Schedule {
    /// The schedule the compiler ships. The ONLY place the Rust constants are
    /// read; nothing here parses anything.
    fn shipped() -> Self {
        // int32 accumulator lanes in one `<16 x i32>` group. Not a named
        // constant in `cpu_gemm.rs` - it is baked into
        // `VNNI_NR = VNNI_NRV * 16` and into `column_of_lane`. DERIVED rather
        // than written down, so it cannot become a fourth copy of 16.
        let lanes = VNNI_NR / VNNI_NRV;
        Schedule {
            mr: VNNI_MR,
            nrv: VNNI_NRV,
            lanes,
            nr: VNNI_NR,
            // int16 elements in one `<32 x i16>` B vector group: two per lane.
            vec_elems: 2 * lanes,
            flush: VnniExact::DEFAULT_FLUSH_K_PAIRS as usize,
            minband: KSPLIT_MIN_BAND,
            maxthr: KSPLIT_MAX_THREADS,
        }
    }
}

fn schedule_path() -> PathBuf {
    repo().join("proofs").join("ExactGemmSchedule.v")
}

/// Bind the schedule expressions' names to Coq variables.
///
/// The LLVM side binds the same names to registers. That is the whole point:
/// [tile_width_ix] and [panel_index_ix] are ONE expression with two renderings,
/// so the definitions below are not a transcription of the emitter - they are
/// the emitter's own arithmetic.
fn coq_names(n: &'static str) -> String {
    match n {
        "ext" => "ext".into(),
        "iv" => "iv".into(),
        "T" => "T".into(),
        "K" => "K".into(),
        "nthr" => "nthr".into(),
        "base" => "base".into(),
        "rem" => "rem".into(),
        "t" => "t".into(),
        "kc" => "kc".into(),
        "Tm1" => "Tm1".into(),
        "p" => "p".into(),
        "MR" => "MR".into(),
        "i" => "i".into(),
        "n" => "n".into(),
        "idx" => "idx".into(),
        "g" => "g".into(),
        "count" => "count".into(),
        "gran" => "gran".into(),
        other => panic!("the schedule expressions gained an unbound name `{other}`"),
    }
}

fn render(s: &Schedule) -> String {
    let tile_width_body = tile_width_ix().coq(&coq_names);
    let panel_index_body = panel_index_ix().coq(&coq_names);
    let chunk_end_body = chunk_end_ix().coq(&coq_names);
    let band_base_body = band_base_ix().coq(&coq_names);
    let band_rem_body = band_rem_ix().coq(&coq_names);
    let band_len_body = band_len_ix().coq(&coq_names);
    let kpairs_body = kpairs_ix().coq(&coq_names);
    let tile_count_body = tile_count_ix().coq(&coq_names);
    let a_row_base_body = a_row_base_ix().coq(&coq_names);
    let a_elem_body = a_i32_element_ix().coq(&coq_names);
    let prop_edge_body = prop_band_edge_ix().coq(&coq_names);
    let granule_edge_body = granule_band_edge_ix().coq(&coq_names);
    let Schedule { mr, nrv, lanes, nr, vec_elems: ve, flush, minband, maxthr } = *s;

    format!(
        r#"(** * The exact-GEMM schedule: constants and index maps, in ONE place.

    *** GENERATED FILE - DO NOT EDIT. ***

    Emitted from `src/cpu_gemm.rs`'s own constants by
    `tests/exact_gemm_schedule_proof.rs`, which links against them rather than
    parsing them. Regenerate with

    <<  Y_REWRITE_SCHEDULE_PROOF=1 cargo test --release --test exact_gemm_schedule_proof  >>

    and the same gate fails the build on any byte of divergence.

    ** Why this file exists.

    The nine Phase 1 proofs each declared their own copy of the schedule: `MR`
    and `NR` in two files, `col_of` in two, the B slot map in three, and the
    flush interval as a bare `64` inside a theorem statement.
    `ExactGemmComposition.v` asserts the copies agree - a theorem somebody
    remembered to write. Nothing forced it to exist, and nothing connected any
    of it to the Rust the kernel is actually emitted from.

    That was MEASURED before it was fixed, and the measurement corrected a
    claim in `ExactGemmComposition.v`'s own header. It says each duplicated
    definition "turns out to be pinned by a theorem in its own file". That
    holds for `RegisterTile.slot_b`, `RegisterTile.NR`, `RegisterTile.NRV`,
    `Packing.NR` and `Micro.slot` - and is FALSE for `MR`, which was never
    tried: setting it to 8 in either file leaves that file compiling. So the
    pinning was incidental, and for one constant of six it was absent.

    ** What this closes.

    Constant drift, in both directions: a `.v` edited by hand and a Rust
    constant moved without regenerating both fail the gate. It does NOT close
    the gap `docs/proof_carrying_kernels.md` names about itself - the loop NEST
    is still hand-written `IrBuilder` calls, and the tie between the model and
    the emitted LLVM is still two models meeting. That is Phase 2.

    ** What is deliberately NOT collapsed.

    [slot_b] and [slot_b_interleave] are two DIFFERENT expressions for one map:
    the emitted `vpdpwssd` vector-group form, and the plain interleave that
    `panel_slot_decode` in `cpu_gemm.rs` inverts with a bare `/2`.
    [slot_b_is_the_plain_interleave] proves they agree. Collapsing them to one
    definition would make that theorem - and
    `ExactGemmComposition.the_agreement_is_not_vacuous` - true by `reflexivity`
    and worth nothing.

    Build:  coqc proofs/ExactGemmSchedule.v      (Rocq 9.1)
*)

From Stdlib Require Import Arith Lia.

(* ------------------------------------------------------------------ *)
(** ** The tile                                                        *)
(* ------------------------------------------------------------------ *)

(** `VNNI_MR`: rows of C the micro-kernel holds in registers. *)
Definition MR : nat := {mr}.

(** `VNNI_NRV`: `<16 x i32>` accumulator groups per row. *)
Definition NRV : nat := {nrv}.

(** int32 lanes in one accumulator group. Derived in the generator as
    `VNNI_NR / VNNI_NRV` rather than written down, so it cannot become a
    fourth copy of 16. *)
Definition LANES : nat := {lanes}.

(** `VNNI_NR`: columns of C the micro-kernel covers. *)
Definition NR : nat := {nr}.

(** int16 elements in one `<32 x i16>` B vector group - two per lane. This is
    the `v * 32` stride of the emitted B load. *)
Definition VEC_ELEMS : nat := {ve}.

(** AVX-512's architectural register count. An ISA FACT, not a schedule
    constant - there is no `cpu_gemm.rs` constant for it, and no proof over
    [nat] establishes it. It sits at the same boundary as `vpdpwssd`'s
    semantics: pinned empirically, here by
    `tests/cpu_gemm_vnni_micro.rs::the_hot_loop_does_not_spill_the_accumulators`
    reading real compiled output. *)
Definition ZMM_REGISTERS : nat := 32.

(* ------------------------------------------------------------------ *)
(** ** The K axis                                                      *)
(* ------------------------------------------------------------------ *)

(** `VnniExact::DEFAULT_FLUSH_K_PAIRS`: k-pairs accumulated in int32 before
    widening into the int64 running sum. This was a bare `64` inside
    `ExactGemmMicro.the_4096_case_exceeds_by_exactly_one`, which is the one
    place a schedule constant had been written into a theorem STATEMENT. *)
Definition FLUSH_K_PAIRS : nat := {flush}.

(** `KSPLIT_MIN_BAND`: shortest K band worth giving a thread. *)
Definition KSPLIT_MIN_BAND : nat := {minband}.

(** `KSPLIT_MAX_THREADS`: the emitted wrapper's clamp. *)
Definition KSPLIT_MAX_THREADS : nat := {maxthr}.

(* ------------------------------------------------------------------ *)
(** ** Panel destination indices                                       *)
(* ------------------------------------------------------------------ *)

(** `pack_a_slot`. Where `pack_a` puts row `i`, half `h`, inside its `2*MR`
    slot group. *)
Definition slot_a (i h : nat) : nat := 2 * i + h.

(** `pack_b_slot`. The emitted `vpdpwssd` lane layout: accumulator group
    `v = j / LANES` is one `<32 x i16>` vector, and lane `l = j mod LANES`
    inside it consumes int16 elements `2l` and `2l+1`. *)
Definition slot_b (j h : nat) : nat := (j / {lanes}) * {ve} + (j mod {lanes}) * 2 + h.

(** The plain interleave - the form `panel_slot_decode` inverts with a bare
    `/2`, and the form the register-tile and micro-kernel models are stated
    over. [slot_b_is_the_plain_interleave] below is what says these are one
    map; the Rust doc comment on `panel_slot_decode` asserts it in prose. *)
Definition slot_b_interleave (j h : nat) : nat := 2 * j + h.

(** `column_of_lane`. Which column of the tile accumulator lane `l` of vector
    `v` holds. *)
Definition col_of (v l : nat) : nat := {lanes} * v + l.

(** `vec_of_slot` / `lane_of_slot`: the inverse legs. *)
Definition vec_of_slot (s : nat) : nat := s / {ve}.
Definition lane_of_slot (s : nat) : nat := (s mod {ve}) / 2.

(** `a_i32_element`. One i32 load fetches both halves of row `i`'s k-pair `p`.
    [ExactGemmRegisterTile.the_i32_load_is_the_packed_pair] is stated over this
    number; [the_emitted_a_index_is_the_pair_element] below is what says the
    emitter computes it. *)
Definition a_i32_element (p i : nat) : nat := p * {mr} + i.

(** `panel_slot_decode`, as its three legs. `width` is [MR] for an A panel and
    [NR] for a B one. *)
Definition panel_group (s width : nat) : nat := s / (2 * width).
Definition panel_idx (s width : nat) : nat := (s mod (2 * width)) / 2.
Definition panel_half (s width : nat) : nat := (s mod (2 * width)) mod 2.

(* ------------------------------------------------------------------ *)
(** ** Loop decompositions                                             *)
(* ------------------------------------------------------------------ *)

(** k-pairs in a K panel of `kc`, rounded up: the phantom half of an odd `kc`
    is what the packers' zero-fill covers.

    **Rendered from `cpu_gemm::kpairs_ix`**, the expression the emitter spells
    at FIVE sites - both packers, the driver, and twice in the threaded
    wrapper. Every packing and flush theorem is stated in terms of this number
    and nothing said the compiler computed the same one. *)
Definition kpairs (kc : nat) : nat := {kpairs_body}.

(** `mn_tiles`. The output partition: a single ragged tail, clamped. *)
Definition tw (ext T t : nat) : nat := Nat.min (ext - t * T) T.
Definition toff (T t : nat) : nat := t * T.
Definition ntiles (ext T : nat) : nat := (ext + T - 1) / T.

(** `ksplit_bands`. The K-split reduction: `base = K/nthr`, `rem = K mod nthr`,
    and the first `rem` bands take one extra k, so the cuts are UNEVEN. A
    different decomposition from [tw] deliberately - do not unify them. *)
Definition blen (K nthr t : nat) : nat :=
  K / nthr + (if Nat.ltb t (K mod nthr) then 1 else 0).

Fixpoint boff (K nthr t : nat) : nat :=
  match t with
  | O => O
  | S t' => boff K nthr t' + blen K nthr t'
  end.

(** `ksplit_threads`. The emitted `__y_gemm_exact_threads`. *)
Definition ksplit_threads (requested K : nat) : nat :=
  let ceil := Nat.min (Nat.max requested 1) {maxthr} in
  let by_k := K / {minband} in
  Nat.max (Nat.min by_k ceil) 1.

(** `flush_chunks`. Chunk `t` starts at `t*Fl` and is CLAMPED - the final
    partial chunk is carried by the same clamp rather than by an epilogue.
    `Fl` stays a parameter: `ExactGemmMicro.v` quantifies over it, and
    [FLUSH_K_PAIRS] above is the value the compiler ships. *)
Definition coff (Fl t : nat) : nat := t * Fl.
Definition cw (Fl n t : nat) : nat := Nat.min (coff Fl (S t)) n - coff Fl t.
Definition nchunks (Fl n : nat) : nat := (n + Fl - 1) / Fl.

(* ------------------------------------------------------------------ *)
(** ** The emitter's own index arithmetic                              *)
(* ------------------------------------------------------------------ *)

(** **These two are rendered from the SAME expressions the emitter renders to
    LLVM** - `cpu_gemm::tile_width_ix` and `panel_index_ix`, via `Ix::coq`
    where the driver uses `Ix::emit`. Everything above is generated from a
    constant; these are generated from the emitted CODE's arithmetic.

    That is the difference between a model that agrees with the emitter and a
    model that IS the emitter, for this slice of the schedule. The slice is
    small and deliberately so: it is the tile-width clamp and the panel index,
    which is the arithmetic the driver's loop nest does on its induction
    variables, and which §1 of `docs/proof_carrying_kernels.md` names as where
    the bugs live ("twelve address computations ... correct only because
    `lda == K` made stride and extent the same number"). Which loops exist, in
    what order, and what they call is still hand-written. *)

(** The live width of the tile starting at induction variable `iv`. *)
Definition tile_width (ext iv T : nat) : nat := {tile_width_body}.

(** Which packed panel the tile at `iv` reads. *)
Definition panel_index (iv T : nat) : nat := {panel_index_body}.

(** **The join.** [tw] is the tiling model, stated over the tile INDEX; the
    emitted loop has the induction variable instead. This says they are the
    same number, which is what lets `ExactGemmTiling.v`'s partition theorems
    describe the emitted driver.

    It was implicit before: the emitter clamped with three `IrBuilder` calls
    and a behavioural test (`exact_gemm_tile_enumeration.rs`) sampled the
    result. *)
Theorem the_emitted_width_is_the_tiling_model_at_the_loop_variable :
  forall ext T t, tw ext T t = tile_width ext (toff T t) T.
Proof. reflexivity. Qed.

(** ...and the emitted `sdiv iv, T` really does recover the tile index, so the
    panel a tile reads is its own. Getting this wrong is a correctly-computed
    tile read from the wrong panel - the shape of bug this repo catalogues as
    invisible to a relative-L2 check. *)
Theorem the_emitted_panel_index_is_the_tile_index :
  forall T t, 0 < T -> panel_index (toff T t) T = t.
Proof.
  intros T t HT. unfold panel_index, toff.
  rewrite Nat.div_mul by lia. reflexivity.
Qed.

(** The flush chunk's END, as the micro-kernel's outer loop computes it.

    The emitter computes an END and [cw] computes a WIDTH; they are the same
    clamp seen from two sides, and the theorem below is that identity. *)
Definition chunk_end (iv T ext : nat) : nat := {chunk_end_body}.

(** The K-split's even share and remainder. Loop-invariant, so the emitted
    wrapper hoists both out of the spawn loop - which is why they are separate
    expressions rather than subterms of [band_len]. An expression split across
    basic blocks is not one contiguous instruction sequence. *)
Definition band_base (K nthr : nat) : nat := {band_base_body}.
Definition band_rem (K nthr : nat) : nat := {band_rem_body}.

(** Band `t`'s length, over the already-hoisted `base` and `rem`. *)
Definition band_len (base rem t : nat) : nat := {band_len_body}.

(** How many tiles an axis is cut into, as the threaded wrapper computes it.

    `T - 1` is a separate parameter because the emitter FOLDS it at compile
    time into a literal (`add i64 %M, 5`); modelling it as `T - 1` inside the
    expression would emit an instruction the compiler does not. *)
Definition tile_count (ext Tm1 T : nat) : nat := {tile_count_body}.

(** The packed-A row base and the per-row element index, as the micro-kernel
    emits them. `MR` is a tile constant and `i` is a Rust-side constant per
    unrolled row, so both reach the emitted instructions as literal operands -
    which makes them bound NAMES here, not a reason the expression cannot be
    extracted.

    They are two expressions rather than one because the base is loop-invariant
    across the unroll and the emitter hoists it, the same split as
    [band_base]/[band_len]. *)
Definition a_base (p MR : nat) : nat := {a_row_base_body}.
Definition a_elem (base i : nat) : nat := {a_elem_body}.

(** ** The f32 kernel's bands - a SECOND consumer, and a different split

    `src/cpu_gemm.rs` emits two GEMMs. Everything above is the exact `vpdpwssd`
    one; these two expressions are the **f32 AVX-512** kernel's partitions, and
    they are here because they are the same KIND of object, not because they
    are the same object. The f32 K-split is proportional
    (`[t*K/n, (t+1)*K/n)`), NOT the exact kernel's even-with-remainder split
    (`[boff, boff + blen)`) - `proofs/GemmBandSplit.v` proves each tiles, and
    exhibits an instance where they disagree.

    `prop_band_edge`: band `t` of the proportional split runs
    `[prop_band_edge t n ext, prop_band_edge (S t) n ext)`, so one expression
    is both ends. *)
Definition prop_band_edge (t n ext : nat) : nat := {prop_edge_body}.

(** `granule_band_edge`: the f32 M and N bands, which partition the GRANULE
    COUNT `g` rather than the extent - snapping a band's position to the tile
    granularity instead dumps the accumulated slack onto one band, and in 2-D
    the two axes' errors multiply. `g` is [tile_count], so this shares an
    expression with the exact kernel above. *)
Definition granule_band_edge (idx g count gran ext : nat) : nat := {granule_edge_body}.

(** **The tiling-count join**, and it needs `0 < T` because `nat` subtraction
    truncates: at `T = 0` the model's `ext + T - 1` is `ext - 1` while the
    emitter's `ext + (T-1)` is `ext`. The emitter cannot reach `T = 0` - the
    tile is a compile-time constant - but the hypothesis is what makes the
    folding legitimate rather than incidental. *)
Theorem the_emitted_tile_count_is_the_tiling_model : forall ext T,
  0 < T -> ntiles ext T = tile_count ext (T - 1) T.
Proof.
  intros ext T HT. unfold ntiles, tile_count.
  replace (ext + T - 1)%nat with (ext + (T - 1))%nat by lia.
  reflexivity.
Qed.

(** **The flush join.** [cw] is the model's chunk width; the emitted loop
    clamps an end instead. *)
Theorem the_emitted_chunk_end_is_the_flush_model : forall Fl n t,
  cw Fl n t = chunk_end (coff Fl t) Fl n - coff Fl t.
Proof.
  intros Fl n t. unfold cw, chunk_end, coff.
  replace (S t * Fl)%nat with (t * Fl + Fl)%nat by lia.
  reflexivity.
Qed.

(** **The K-split join.** The emitted spawn loop's `%klen` is
    `ExactGemmKsplit`'s [blen], recomposed from the two hoisted terms. Every
    theorem in that file is about [blen]; this is what says the emitted wrapper
    computes it. *)
Theorem the_emitted_band_length_is_the_ksplit_model : forall K nthr t,
  blen K nthr t = band_len (band_base K nthr) (band_rem K nthr) t.
Proof. reflexivity. Qed.

(** **The A-index join.** [ExactGemmRegisterTile.the_i32_load_is_the_packed_pair]
    proves the i32 load at element [a_i32_element p i] aliases packed slots
    `2i` and `2i+1` of k-pair group `p`. That theorem says nothing about the
    compiler; this says the compiler computes that element. *)
Theorem the_emitted_a_index_is_the_pair_element : forall p i,
  a_i32_element p i = a_elem (a_base p MR) i.
Proof. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** What this file proves about itself                              *)
(* ------------------------------------------------------------------ *)

(** `tests/proofs_are_checked.rs::every_proof_has_a_content_control` refuses a
    `.v` that names no load-bearing theorem, and it is right to: "compiles"
    and "no axioms" are properties an EMPTY file has. A generated file is
    exactly where that matters most, because a generator emitting nonsense
    emits it confidently.

    So this file is not definitions-only and takes no exemption. The three
    theorems below each catch a distinct way the generator could be wrong, and
    two of them are STRUCTURAL - they constrain the shape of the emitted
    expressions, not their values, so they are not made true merely by being
    generated alongside what they describe. *)

(** The tile geometry is internally consistent. Catches a generator that
    emitted [NR] and [NRV] from constants that had stopped agreeing, or any
    degenerate zero. *)
Theorem the_tile_geometry_is_consistent :
  NR = NRV * LANES
  /\ VEC_ELEMS = 2 * LANES
  /\ 0 < MR /\ 0 < NRV /\ 0 < LANES /\ 0 < NR
  /\ 0 < FLUSH_K_PAIRS /\ 0 < KSPLIT_MIN_BAND /\ 0 < KSPLIT_MAX_THREADS.
Proof. unfold NR, NRV, LANES, VEC_ELEMS, MR, FLUSH_K_PAIRS,
         KSPLIT_MIN_BAND, KSPLIT_MAX_THREADS. repeat split; lia. Qed.

(** The emitted vector-group form of the B map IS the plain interleave.

    This is the load-bearing one, and it is not a restatement of a generated
    value: the two sides come from different places in `cpu_gemm.rs` -
    [slot_b] from `pack_b_slot`, [slot_b_interleave] from the bare `/2` that
    `panel_slot_decode` uses to invert it. The Rust asserts their agreement in
    a doc comment; this proves it.

    `16*(j/16) + (j mod 16) = j`, so the vector-group decomposition folds away
    entirely. `ExactGemmPacking.v` records the consequence: there are not two
    layouts to tell apart, so no proof can pin the lane assignment and
    `tests/cpu_gemm_vnni_micro.rs` on the real instruction is what does. *)
Theorem slot_b_is_the_plain_interleave : forall j h,
  slot_b j h = slot_b_interleave j h.
Proof.
  intros j h. unfold slot_b, slot_b_interleave.
  pose proof (Nat.div_mod_eq j {lanes}) as H. lia.
Qed.

(** **The tile fits the register file**, and this is the constraint that was
    written in a test comment and stated nowhere.

    The micro-kernel holds `MR * NRV` int32 accumulators live across the hot
    loop, plus [NRV] `<32 x i16>` B vectors and one broadcast A vector. On
    AVX-512 that has to fit in 32 zmm. `cpu_gemm.rs` never says so and no
    theorem did either - the arithmetic appeared only as prose beside a spill
    bound ("24 accumulators + 4 B vectors + 1 A broadcast is 29 of 32 zmm").

    **The predicate is MEASURED, not guessed**, by sweeping `VNNI_MR` and
    reading the hot loop's real spill traffic:

<<
      MR   budget   hot-loop spills + reloads
       5   25/32    within bound
       6   29/32    within bound  (10 + 10, the shipped kernel)
       7   33/32    16 + 16
       8   37/32    17 + 17
>>

    The cliff falls exactly where this inequality flips, so the form of the
    bound is the measurement's and not an invention.

    It is STRUCTURAL and it bites: a generator emitting `MR = 8` emits a file
    in which this theorem is FALSE, so `coqc` rejects the schedule outright
    rather than proving nine theorems about a tile that cannot be allocated.
    Note what it does not claim - a spilling kernel is SLOW, not wrong.
    `tests/exact_gemm_thread_invariance.rs` still passes bit-identically at
    `MR = 8`, which is how the diagnosis separated the two. *)
Theorem the_tile_fits_the_register_file :
  MR * NRV + NRV + 1 <= ZMM_REGISTERS.
Proof. unfold MR, NRV, ZMM_REGISTERS. lia. Qed.

(** The thread count is never zero.

    A genuine cross-file join rather than a self-check: every theorem in
    `ExactGemmKsplit.v` is stated under `(0 < nthr)%nat`, and `ksplit_bands`
    in `cpu_gemm.rs` asserts the same precondition at runtime. Nothing proved
    the emitted thread count satisfies it - the floor was argued in a comment
    ("`ksplit_threads` floors at 1"). Here it is discharged, for every request
    and every K, including `K` below [KSPLIT_MIN_BAND] where `by_k` is 0. *)
Theorem ksplit_threads_is_never_zero : forall requested K,
  0 < ksplit_threads requested K.
Proof. intros. unfold ksplit_threads. lia. Qed.

(** The non-vacuity control, and it is honest about being weaker than the two
    above. Under generation this is self-fulfilling - a generator emitting
    `MR := 8` emits this theorem with an 8 in it. Its job is the OTHER
    direction: it makes the shipped values load-bearing inside Coq, so a
    hand-edit of this committed file fails `coqc` as well as failing the
    byte-identity gate. That matters most for [MR], which the measurement
    recorded above found was pinned by nothing in its own file. *)
Theorem the_schedule_is_the_shipped_one :
  MR = {mr} /\ NRV = {nrv} /\ LANES = {lanes} /\ NR = {nr}
  /\ VEC_ELEMS = {ve} /\ FLUSH_K_PAIRS = {flush}
  /\ KSPLIT_MIN_BAND = {minband} /\ KSPLIT_MAX_THREADS = {maxthr}.
Proof. repeat split; reflexivity. Qed.

Print Assumptions the_tile_geometry_is_consistent.
Print Assumptions slot_b_is_the_plain_interleave.
Print Assumptions the_emitted_width_is_the_tiling_model_at_the_loop_variable.
Print Assumptions the_emitted_panel_index_is_the_tile_index.
Print Assumptions the_emitted_tile_count_is_the_tiling_model.
Print Assumptions the_emitted_chunk_end_is_the_flush_model.
Print Assumptions the_emitted_band_length_is_the_ksplit_model.
Print Assumptions the_emitted_a_index_is_the_pair_element.
Print Assumptions the_tile_fits_the_register_file.
Print Assumptions ksplit_threads_is_never_zero.
Print Assumptions the_schedule_is_the_shipped_one.
"#
    )
}

/// The gate. Regenerates and compares byte for byte.
#[test]
fn the_committed_schedule_proof_is_what_cpu_gemm_generates() {
    let want = render(&Schedule::shipped());
    let path = schedule_path();

    if std::env::var("Y_REWRITE_SCHEDULE_PROOF").is_ok() {
        std::fs::write(&path, &want).expect("write ExactGemmSchedule.v");
        eprintln!("rewrote {}", path.display());
        return;
    }

    let have = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "proofs/ExactGemmSchedule.v is missing ({e}). It is GENERATED and \
             committed; regenerate with \
             `Y_REWRITE_SCHEDULE_PROOF=1 cargo test --release --test \
             exact_gemm_schedule_proof`."
        )
    });

    if have == want {
        return;
    }

    // Name the first differing line rather than dumping two files: the usual
    // cause is one constant, and the diff should say which.
    let first = have
        .lines()
        .zip(want.lines())
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| format!("line {}:\n  committed: {a}\n  generated: {b}", i + 1))
        .unwrap_or_else(|| {
            format!(
                "no differing line - the files differ in length ({} vs {} lines)",
                have.lines().count(),
                want.lines().count()
            )
        });

    panic!(
        "proofs/ExactGemmSchedule.v is not what src/cpu_gemm.rs generates.\n\n{first}\n\n\
         The exact-GEMM schedule has one source. Either a constant in \
         cpu_gemm.rs moved and the proofs still describe the old kernel, or \
         this generated file was edited by hand. Regenerate with \
         `Y_REWRITE_SCHEDULE_PROOF=1 cargo test --release --test \
         exact_gemm_schedule_proof` and re-check the proofs - a constant that \
         moves changes what nine theorems are ABOUT."
    );
}

/// The control, and the first version of it was VACUOUS.
///
/// A byte-identity gate is only as good as its generator. Neuter
/// [render] so it returns the committed file verbatim and the gate passes
/// forever while checking nothing - `feedback-null-metrics-pass-dead-components`
/// in the place it bites hardest.
///
/// **The obvious control does not catch that, and mutation is what showed it.**
/// The first version asserted the output *contains*
/// `Definition MR : nat := 6.` and friends. Under the neutering it still does,
/// because the committed file contains exactly those lines. The mutation
/// survived a green run.
///
/// What catches it is rendering a schedule that is NOT the shipped one and
/// requiring the result to differ from the committed text. A generator that
/// echoes the file cannot do that; a generator that is a function of its
/// argument does it for every field.
#[test]
fn the_generator_is_a_function_of_the_schedule_not_of_the_committed_file() {
    let shipped = Schedule::shipped();
    let committed = std::fs::read_to_string(schedule_path()).expect("read the committed proof");

    // Each field, perturbed one at a time, with the line it must move.
    let cases: Vec<(&str, Schedule, String)> = vec![
        (
            "MR",
            Schedule { mr: shipped.mr + 1, ..shipped },
            format!("Definition MR : nat := {}.", shipped.mr + 1),
        ),
        (
            "NRV",
            Schedule { nrv: shipped.nrv + 1, ..shipped },
            format!("Definition NRV : nat := {}.", shipped.nrv + 1),
        ),
        (
            "LANES",
            Schedule { lanes: shipped.lanes + 1, ..shipped },
            format!("Definition LANES : nat := {}.", shipped.lanes + 1),
        ),
        (
            "NR",
            Schedule { nr: shipped.nr + 1, ..shipped },
            format!("Definition NR : nat := {}.", shipped.nr + 1),
        ),
        (
            "VEC_ELEMS",
            Schedule { vec_elems: shipped.vec_elems + 1, ..shipped },
            format!("Definition VEC_ELEMS : nat := {}.", shipped.vec_elems + 1),
        ),
        (
            "FLUSH_K_PAIRS",
            Schedule { flush: shipped.flush + 1, ..shipped },
            format!("Definition FLUSH_K_PAIRS : nat := {}.", shipped.flush + 1),
        ),
        (
            "KSPLIT_MIN_BAND",
            Schedule { minband: shipped.minband + 1, ..shipped },
            format!("Definition KSPLIT_MIN_BAND : nat := {}.", shipped.minband + 1),
        ),
        (
            "KSPLIT_MAX_THREADS",
            Schedule { maxthr: shipped.maxthr + 1, ..shipped },
            format!("Definition KSPLIT_MAX_THREADS : nat := {}.", shipped.maxthr + 1),
        ),
    ];

    for (name, perturbed, expected) in cases {
        assert_ne!(
            perturbed, shipped,
            "the `{name}` case does not actually perturb the schedule, so it \
             asserts nothing"
        );
        let out = render(&perturbed);
        assert_ne!(
            out, committed,
            "rendering a schedule with `{name}` changed produced the COMMITTED \
             file byte for byte. The generator is not a function of its \
             argument - most likely it reads proofs/ExactGemmSchedule.v - which \
             makes the byte-identity gate a check on a constant string."
        );
        assert!(
            out.contains(&expected),
            "rendering with `{name}` perturbed did not emit `{expected}`, so \
             that constant does not reach the generated proof."
        );
    }
}

/// ...and the schedule the generator is handed is the Rust's own.
///
/// The control above proves [render] is a function of its argument. This
/// proves the argument is `cpu_gemm.rs`. Neither claim is worth anything
/// alone: a faithful renderer of the wrong constants, and a faithful reading
/// of constants that never reach the output, both pass one of them.
#[test]
fn the_shipped_schedule_is_the_rust_constants() {
    let s = Schedule::shipped();
    assert_eq!(s.mr, VNNI_MR, "MR is not cpu_gemm::VNNI_MR");
    assert_eq!(s.nrv, VNNI_NRV, "NRV is not cpu_gemm::VNNI_NRV");
    assert_eq!(s.nr, VNNI_NR, "NR is not cpu_gemm::VNNI_NR");
    assert_eq!(
        s.flush,
        VnniExact::DEFAULT_FLUSH_K_PAIRS as usize,
        "FLUSH_K_PAIRS is not zero_drift::VnniExact::DEFAULT_FLUSH_K_PAIRS"
    );
    assert_eq!(s.minband, KSPLIT_MIN_BAND, "KSPLIT_MIN_BAND is not cpu_gemm's");
    assert_eq!(s.maxthr, KSPLIT_MAX_THREADS, "KSPLIT_MAX_THREADS is not cpu_gemm's");

    // The derived lane count has to be exact, not a truncation. If VNNI_NR
    // ever stops being a multiple of VNNI_NRV the generated
    // `the_tile_geometry_is_consistent` is false and `coqc` rejects it - but
    // it is worth failing here, with a reason.
    assert_eq!(
        s.nrv * s.lanes,
        s.nr,
        "VNNI_NR is no longer VNNI_NRV * LANES, so the derived lane count is a \
         truncation and the generated tile-geometry theorem cannot be proved"
    );
    assert_eq!(s.vec_elems, 2 * s.lanes, "VEC_ELEMS is not two int16 per lane");
}

/// The schedule the generator emits must be the schedule the model functions
/// in `cpu_gemm.rs` actually compute.
///
/// The gate above ties the `.v` to the Rust CONSTANTS. This ties it to the
/// Rust FUNCTIONS, which is a different claim: a constant could be carried
/// correctly into Coq while `pack_b_slot` computes something else. Checked by
/// evaluating both readings over a range rather than by inspection.
#[test]
fn the_generated_index_maps_agree_with_the_rust_ones() {
    let Schedule { lanes, vec_elems, .. } = Schedule::shipped();
    use y::cpu_gemm::{
        a_i32_element, column_of_lane, lane_of_slot, pack_a_slot, pack_b_slot,
        panel_slot_decode, vec_of_slot,
    };

    for j in 0..(4 * VNNI_NR) {
        for h in 0..2 {
            // slot_a / slot_b, as the generated file spells them.
            assert_eq!(pack_a_slot(j, h), 2 * j + h, "slot_a at ({j}, {h})");
            assert_eq!(
                pack_b_slot(j, h),
                (j / lanes) * vec_elems + (j % lanes) * 2 + h,
                "slot_b at ({j}, {h})"
            );
            // ...and the interleave the generated theorem says it equals.
            assert_eq!(
                pack_b_slot(j, h),
                2 * j + h,
                "pack_b_slot is no longer the plain interleave at ({j}, {h}); \
                 `slot_b_is_the_plain_interleave` in the generated proof is false"
            );
        }
    }

    for v in 0..VNNI_NRV {
        for l in 0..lanes {
            assert_eq!(column_of_lane(v, l), lanes * v + l, "col_of at ({v}, {l})");
        }
    }

    for s in 0..(8 * vec_elems) {
        assert_eq!(vec_of_slot(s), s / vec_elems, "vec_of_slot at {s}");
        assert_eq!(lane_of_slot(s), (s % vec_elems) / 2, "lane_of_slot at {s}");
    }

    for p in 0..8 {
        for i in 0..VNNI_MR {
            assert_eq!(a_i32_element(p, i), p * VNNI_MR + i, "a_i32_element at ({p}, {i})");
        }
    }

    for width in [VNNI_MR, VNNI_NR] {
        for s in 0..(4 * 2 * width) {
            let (g, idx, h) = panel_slot_decode(s, width);
            assert_eq!(g, s / (2 * width), "panel_group at ({s}, {width})");
            assert_eq!(idx, (s % (2 * width)) / 2, "panel_idx at ({s}, {width})");
            assert_eq!(h, (s % (2 * width)) % 2, "panel_half at ({s}, {width})");
        }
    }
}

/// The claim this file's index-arithmetic half makes, checked against the real
/// emitted module.
///
/// `tile_width_ix` and `panel_index_ix` are ONE expression with two renderings:
/// `Ix::coq` produces the definitions in `proofs/ExactGemmSchedule.v`, and
/// `Ix::emit` produces the driver's instructions. This asserts the second half
/// - that the arithmetic the proof describes is the arithmetic the compiler
/// emits - by rendering each expression standalone and requiring the same
/// instruction SHAPE to appear in `emit_vnni_gemm_module`.
///
/// **It checks the PROPERTY, not the plumbing.** A driver that hand-wrote an
/// identical clamp would pass, and should: there would be no divergence. A
/// driver that hand-wrote a different one - operands swapped, `sle` for `slt`,
/// the clamp dropped - fails, which is the case that matters. Register names
/// are normalised away because they are numbered by the surrounding function.
/// Normalise only the BUILDER'S OWN temporaries (`%g12`, `%iv4`), leaving
/// named values like `%M` and `%lda` verbatim.
///
/// **Normalising every register was the first attempt and it was too
/// loose** - it turns `sub i64 %M, %iv` and `sub i64 %iv, %M` into the same
/// string, which is precisely the distinction this test exists to make.
/// The non-vacuity control below caught that on the first run, which is
/// what a control is for. Alpha-renaming does not fix it either: both
/// operands are distinct registers under any renaming. Keeping the NAMED
/// half anchored is what makes operand order observable.
fn shape(line: &str) -> String {
    let mut out = String::new();
    let ch: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < ch.len() {
        if ch[i] == '%' {
            let start = i + 1;
            let mut j = start;
            while j < ch.len() && (ch[j].is_alphanumeric() || ch[j] == '.' || ch[j] == '_') {
                j += 1;
            }
            let name: String = ch[start..j].iter().collect();
            // A builder temporary: `g` or `iv` followed by digits only.
            let generated = ["g", "iv"].iter().any(|p| {
                name.strip_prefix(*p)
                    .is_some_and(|r| !r.is_empty() && r.chars().all(|c| c.is_ascii_digit()))
            });
            out.push_str(if generated { "%_" } else { line_slice(line, i, j) });
            i = j;
        } else {
            out.push(ch[i]);
            i += 1;
        }
    }
    out
}

/// `line[i..j]` by character index, including the leading `%`.
fn line_slice(line: &str, i: usize, j: usize) -> &str {
    let s: Vec<(usize, char)> = line.char_indices().collect();
    let a = s[i].0;
    let b = if j < s.len() { s[j].0 } else { line.len() };
    &line[a..b]
}

#[test]
fn the_emitted_arithmetic_is_the_arithmetic_the_proof_describes() {
    use y::cpu_gemm::{emit_vnni_gemm_module, render_llvm};

    let module = emit_vnni_gemm_module(64);
    let emitted: Vec<String> = module.lines().map(shape).map(|l| l.trim().to_string()).collect();

    // Every site in the driver, with the tile it is instantiated at. Counting
    // is what makes this a claim about ALL of them.
    //
    // **The first version searched for the arithmetic ANYWHERE in the module
    // and that was too weak** - it is satisfied while one site of three
    // diverges, and mutation proved it: swapping the `min` operands at the
    // i-loop (same VALUE, different instructions) passed this test AND all
    // five correctness suites, because the other two sites still matched.
    // Counting turns an existential into a universal.
    //
    // A count is deliberately brittle against the driver gaining or losing a
    // loop. That should be a deliberate edit, and this is where it gets
    // noticed.
    let sites: &[(&str, y::cpu_gemm::Ix, usize, &str, usize)] = &[
        // The pack-A loop and the i-loop walk M in steps of MR; the j-loop
        // walks N in steps of NR. The extent is part of the site, not a
        // constant of the expression - binding it to `%M` everywhere was the
        // first attempt and the NR site reported 0.
        ("tile_width_ix @ M/MR", tile_width_ix(), VNNI_MR, "%M", 2),
        ("tile_width_ix @ N/NR", tile_width_ix(), VNNI_NR, "%N", 1),
        // The A panel index, at the pack-A loop and the i-loop. It does not
        // mention the extent.
        ("panel_index_ix @ MR", panel_index_ix(), VNNI_MR, "%M", 2),
    ];

    // `kpairs_ix` is emitted through the builder at three sites - both packers
    // and the driver - all of which live in `emit_vnni_gemm_module`. Its
    // operand is a parameter (`%kc` or `%K`), so it is checked separately from
    // the table above, whose expressions all take the extent.
    for (kc_reg, want_n) in [("%kc", 2), ("%K", 1)] {
        let want: Vec<String> = render_llvm(&kpairs_ix(), &|n: &'static str| match n {
            "kc" => kc_reg.to_string(),
            other => panic!("unbound {other}"),
        })
        .iter()
        .map(|l| shape(l))
        .collect();
        let n = emitted.windows(want.len()).filter(|w| *w == want.as_slice()).count();
        assert_eq!(
            n, want_n,
            "`kpairs_ix` over `{kc_reg}` appears {n} times in the gemm module, \
             not {want_n}. Every packing and flush theorem is stated in terms \
             of this number, so a site computing a different one makes those \
             theorems about a kernel that is not this one."
        );
    }

    for (name, ix, tile, ext, want_n) in sites {
        let bind = move |n: &'static str| -> String {
            match n {
                "ext" => (*ext).into(),
                // A builder-shaped name, so `shape` normalises it exactly as
                // it normalises the driver's own induction temporary.
                "iv" => "%g0".into(),
                "T" => tile.to_string(),
                other => panic!("unbound {other}"),
            }
        };
        let want: Vec<String> = render_llvm(ix, &bind).iter().map(|l| shape(l)).collect();
        assert!(!want.is_empty(), "{name} rendered to nothing");

        let n = emitted
            .windows(want.len())
            .filter(|w| *w == want.as_slice())
            .count();
        assert_eq!(
            n, *want_n,
            "`{name}` renders to an instruction sequence the driver contains \
             {n} times, not {want_n}. Either a site stopped using the shared \
             schedule expression - so the arithmetic in \
             proofs/ExactGemmSchedule.v is no longer the arithmetic the \
             compiler emits - or the loop nest gained or lost a site and this \
             table needs updating deliberately.\nrendered:\n  {}\n",
            want.join("\n  ")
        );
    }

    // Non-vacuity: the clamp with its operands reversed must appear NOWHERE.
    // `%M` stays anchored under `shape`, which is what makes operand order
    // observable at all - normalising every register made this control fire on
    // the first run, correctly.
    let reversed = y::cpu_gemm::Ix::Sub(
        Box::new(y::cpu_gemm::Ix::Val("iv")),
        Box::new(y::cpu_gemm::Ix::Val("ext")),
    );
    let bind0 = |n: &'static str| -> String {
        match n {
            "ext" => "%M".into(),
            "iv" => "%g0".into(),
            "T" => VNNI_MR.to_string(),
            other => panic!("unbound {other}"),
        }
    };
    let wrong: Vec<String> = render_llvm(&reversed, &bind0).iter().map(|l| shape(l)).collect();
    assert!(
        !emitted.windows(wrong.len()).any(|w| w == wrong.as_slice()),
        "the driver contains `iv - ext`, the tile-width clamp with its operands \
         reversed. Either the emitter is wrong or this control is matching too \
         loosely to mean anything."
    );
}

/// The same claim for the sites emitted as RAW LLVM rather than through the
/// builder: the flush chunk in the micro-kernel, and the K-split bands in the
/// threaded wrapper.
///
/// These are stronger and simpler than the driver's gate above, for a reason
/// worth stating: those emitters choose their own register names (`%cend`,
/// `%base`, `%klen`), so there is nothing to normalise. The rendered text is
/// compared VERBATIM against the module. If a site stops using the shared
/// expression, or uses it differently, the exact line goes missing.
///
/// The count is asserted for the same reason it is asserted above - an
/// existential check is satisfied while one site of several diverges, which
/// mutation demonstrated on the driver's gate rather than being assumed here.
#[test]
fn the_raw_emitted_sites_use_the_shared_schedule_expressions() {
    use y::cpu_gemm::{emit_vnni_micro_module, emit_vnni_threaded_module, render_named};

    let micro = emit_vnni_micro_module(64);
    let threaded = emit_vnni_threaded_module(true);

    let ksplit = |n: &'static str| -> String {
        match n {
            "K" => "%K".into(),
            "nthr" => "%nthr".into(),
            "base" => "%base".into(),
            "rem" => "%rem".into(),
            "t" => "%t".into(),
            other => panic!("unbound {other}"),
        }
    };

    let cases: Vec<(&str, &str, Vec<String>, Vec<String>)> = vec![
        (
            "chunk_end_ix",
            "the micro-kernel",
            vec!["%cend0".into(), "%clt".into(), "%cend".into()],
            render_named(
                &chunk_end_ix(),
                &mut ["%cend0", "%clt", "%cend"].into_iter().map(String::from),
                &|n| match n {
                    "iv" => "%c".to_string(),
                    "T" => "64".to_string(),
                    "ext" => "%kpairs".to_string(),
                    other => panic!("unbound {other}"),
                },
            )
            .0,
        ),
        (
            "tile_count_ix",
            "the threaded wrapper",
            vec!["%mtiles0".into(), "%mtiles".into()],
            render_named(
                &tile_count_ix(),
                &mut ["%mtiles0", "%mtiles"].into_iter().map(String::from),
                &|n| match n {
                    "ext" => "%M".to_string(),
                    "Tm1" => (VNNI_MR - 1).to_string(),
                    "T" => VNNI_MR.to_string(),
                    other => panic!("unbound {other}"),
                },
            )
            .0,
        ),
        (
            "kpairs_ix @ %K",
            "the threaded wrapper",
            vec!["%kp1".into(), "%kps".into()],
            render_named(
                &kpairs_ix(),
                &mut ["%kp1", "%kps"].into_iter().map(String::from),
                &|n| match n {
                    "kc" => "%K".to_string(),
                    other => panic!("unbound {other}"),
                },
            )
            .0,
        ),
        (
            "kpairs_ix @ %klen",
            "the threaded wrapper",
            vec!["%kp1b".into(), "%kpsb".into()],
            render_named(
                &kpairs_ix(),
                &mut ["%kp1b", "%kpsb"].into_iter().map(String::from),
                &|n| match n {
                    "kc" => "%klen".to_string(),
                    other => panic!("unbound {other}"),
                },
            )
            .0,
        ),
        (
            "band_base_ix",
            "the threaded wrapper",
            vec!["%base".into()],
            render_named(
                &band_base_ix(),
                &mut ["%base"].into_iter().map(String::from),
                &ksplit,
            )
            .0,
        ),
        (
            "band_rem_ix",
            "the threaded wrapper",
            vec!["%rem".into()],
            render_named(
                &band_rem_ix(),
                &mut ["%rem"].into_iter().map(String::from),
                &ksplit,
            )
            .0,
        ),
        (
            "band_len_ix",
            "the threaded wrapper",
            vec!["%extra".into(), "%inc".into(), "%klen".into()],
            render_named(
                &band_len_ix(),
                &mut ["%extra", "%inc", "%klen"].into_iter().map(String::from),
                &ksplit,
            )
            .0,
        ),
    ];

    let mut cases = cases;

    // The A panel index, at the micro-kernel's hoisted base and at all MR
    // unrolled rows. `ExactGemmRegisterTile.the_i32_load_is_the_packed_pair` is
    // stated over that element; these are what tie it to the emitted loads.
    cases.push((
        "a_row_base_ix",
        "the micro-kernel",
        vec!["%aidx".into()],
        render_named(
            &a_row_base_ix(),
            &mut ["%aidx"].into_iter().map(String::from),
            &|n| match n {
                "p" => "%p".to_string(),
                "MR" => VNNI_MR.to_string(),
                other => panic!("unbound {other}"),
            },
        )
        .0,
    ));
    let row_cases: Vec<(String, Vec<String>)> = (0..VNNI_MR)
        .map(|i| {
            (
                format!("a_i32_element_ix @ row {i}"),
                render_named(
                    &a_i32_element_ix(),
                    &mut [format!("%ai{i}")].into_iter(),
                    &|n| match n {
                        "base" => "%aidx".to_string(),
                        "i" => i.to_string(),
                        other => panic!("unbound {other}"),
                    },
                )
                .0,
            )
        })
        .collect();
    for (name, lines) in &row_cases {
        cases.push((name.as_str(), "the micro-kernel", vec![], lines.clone()));
    }

    for (name, module_name, _names, lines) in &cases {
        let module = if *module_name == "the micro-kernel" { &micro } else { &threaded };
        assert!(!lines.is_empty(), "{name} rendered to nothing");
        for l in lines {
            let n = module.lines().filter(|m| m.trim_end() == l.trim_end()).count();
            assert_eq!(
                n, 1,
                "`{name}` renders the line `{}`, which appears {n} times in \
                 {module_name} - expected exactly once. Either that site stopped \
                 using the shared schedule expression, so the arithmetic in \
                 proofs/ExactGemmSchedule.v is no longer the arithmetic the \
                 compiler emits, or the emitter changed and this gate needs \
                 updating deliberately.",
                l.trim()
            );
        }
    }

    // Non-vacuity. The flush clamp with its `min` operands swapped computes the
    // SAME VALUE by different instructions - the mutation that escaped every
    // correctness suite on the driver's gate. It must not be what the emitter
    // writes, or this check is matching something it did not intend to.
    let swapped = render_named(
        &y::cpu_gemm::Ix::Min(
            Box::new(y::cpu_gemm::Ix::Val("ext")),
            Box::new(y::cpu_gemm::Ix::Add(
                Box::new(y::cpu_gemm::Ix::Val("iv")),
                Box::new(y::cpu_gemm::Ix::Val("T")),
            )),
        ),
        &mut ["%cend0", "%clt", "%cend"].into_iter().map(String::from),
        &|n| match n {
            "iv" => "%c".to_string(),
            "T" => "64".to_string(),
            "ext" => "%kpairs".to_string(),
            other => panic!("unbound {other}"),
        },
    )
    .0;
    assert!(
        !swapped.iter().all(|l| micro.lines().any(|m| m.trim_end() == l.trim_end())),
        "the micro-kernel contains the flush clamp with its operands swapped. \
         That computes the same value, so no correctness test can see it - \
         which is exactly the divergence this gate exists to catch."
    );
}

/// The **f32** kernel emits the band arithmetic `proofs/GemmBandSplit.v`
/// describes.
///
/// `src/cpu_gemm.rs` emits two GEMMs, and everything above this test is about
/// the exact one. `__y_sgemm_f32_avx512` partitions the same three axes with a
/// DIFFERENT K-split - proportional rather than even-with-remainder - and its
/// M/N bands partition the granule count. Both now render from the shared
/// expressions, so this is the same universal claim one kernel over: the
/// arithmetic in the proof is the arithmetic the compiler emits.
///
/// Counts are per site, for the reason recorded on the exact kernel's table: an
/// existential search is satisfied while one site of several diverges.
#[test]
fn the_f32_kernel_emits_the_band_arithmetic_the_proof_describes() {
    use y::cpu_gemm::{emit_kernel_module, render_llvm, DEFAULT_TILE};

    let module = emit_kernel_module();
    let emitted: Vec<String> = module.lines().map(shape).map(|l| l.trim().to_string()).collect();

    let find = |want: &[String]| -> usize {
        emitted.windows(want.len()).filter(|w| *w == want).count()
    };
    let render = |ix: &y::cpu_gemm::Ix, b: &dyn Fn(&'static str) -> String| -> Vec<String> {
        render_llvm(ix, b).iter().map(|l| shape(l)).collect()
    };

    // A builder temporary, which `shape` normalises exactly as it normalises
    // the emitter's own. The ANCHORS are the extents (`%M`, `%N`, `%K`) and the
    // M-axis granularity, which is the compile-time `MR`.
    let t = || "%g0".to_string();
    let mr = DEFAULT_TILE.mr.to_string();

    // The granule count is `tile_count_ix` - the same expression the exact
    // kernel's threaded wrapper uses for `(M + MR-1)/MR`. Here `gran` is a
    // register on the N axis, so `Tm1` is emitted rather than folded.
    for (axis, ext, gran, want_n) in [("M", "%M", mr.clone(), 1usize), ("N", "%N", t(), 1)] {
        let want = render(&tile_count_ix(), &|n| match n {
            "ext" => ext.to_string(),
            "Tm1" => t(),
            "T" => gran.clone(),
            other => panic!("unbound {other}"),
        });
        let n = find(&want);
        assert_eq!(
            n, want_n,
            "the f32 kernel's {axis}-axis granule count is not `tile_count_ix` \
             ({n} matches, not {want_n})"
        );
    }

    // The band edge, at both ends of both axes.
    for (axis, ext, gran, want_n) in [("M", "%M", mr.clone(), 2usize), ("N", "%N", t(), 2)] {
        let want = render(&granule_band_edge_ix(), &|n| match n {
            "idx" => t(),
            "g" => t(),
            "count" => t(),
            "gran" => gran.clone(),
            "ext" => ext.to_string(),
            other => panic!("unbound {other}"),
        });
        let n = find(&want);
        assert_eq!(
            n, want_n,
            "the f32 kernel's {axis} band edge appears {n} times, not {want_n}. \
             `GemmBandSplit.gedge_last` proves the emitted last-band clamp is \
             redundant; if the edge is no longer this expression that theorem is \
             about a kernel that is not this one."
        );
    }

    // The proportional K-split: `[t*K/n, (t+1)*K/n)`, one expression, both ends.
    let want = render(&prop_band_edge_ix(), &|n| match n {
        "t" => t(),
        "ext" => "%K".to_string(),
        "n" => t(),
        other => panic!("unbound {other}"),
    });
    let n = find(&want);
    assert_eq!(
        n, 2,
        "the f32 K-split's band edge appears {n} times, not 2 (its two ends). \
         `GemmBandSplit.prop_ksplit_exact` is stated over this decomposition."
    );

    // Non-vacuity, and it is a real distinction rather than a formality: the
    // EXACT kernel's split is `K/nthr` + a remainder test, and if the f32
    // kernel emitted that instead, every theorem in GemmBandSplit.v would be
    // about the wrong partition while still being true of its own definitions.
    let exact_shaped = render(&band_base_ix(), &|n| match n {
        "K" => "%K".to_string(),
        "nthr" => t(),
        other => panic!("unbound {other}"),
    });
    assert_eq!(
        find(&exact_shaped),
        0,
        "the f32 kernel divides %K by the thread count directly, which is the \
         EXACT kernel's decomposition - the two splits are supposed to differ, \
         and GemmBandSplit.the_two_splits_are_different says so"
    );
}
