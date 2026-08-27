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
//! | Rust `VNNI_MR` 6 -> 8 | **FAIL** | ok | 6 FAIL |
//! | Rust `KSPLIT_MIN_BAND` 128 -> 256 | **FAIL** | ok | all ok |
//! | committed `.v` hand-edited, `MR := 8` | **FAIL** | **FAIL** | all ok |
//! | `ExactGemmRegisterTile.NR := SCH.MR` | ok | **FAIL** | all ok |
//! | `render` echoes the committed file | **FAIL** | ok | all ok |
//! | `render` hardcodes `MR := 6` | **FAIL** | ok | all ok |
//!
//! Three of those are worth reading rather than counting.
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

use y::cpu_gemm::{VNNI_MR, VNNI_NR, VNNI_NRV, KSPLIT_MAX_THREADS, KSPLIT_MIN_BAND};
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

fn render(s: &Schedule) -> String {
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

(** `a_i32_element`. One i32 load fetches both halves of row `i`'s k-pair `p`. *)
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
    is what the packers' zero-fill covers. *)
Definition kpairs (kc : nat) : nat := (kc + 1) / 2.

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
