//! The certificate the GPU side did not emit.
//!
//! `docs/proof_carrying_kernels.md` Phase 3 states the obligation this module
//! discharges, in its own words: *"Y emits PTX; `ptxas` produces SASS and is
//! closed source. The proof covers source-to-PTX, and `ptxas` is trusted or
//! validated per-translation. **This must be stated in the certificate, never
//! papered over.**"*
//!
//! There was no certificate. `--emit-attention-ptx` wrote a `.ptx` to stdout
//! and nothing else, while three machine-checked proofs -
//! `AttentionSchedule.v`, `GridStrideSplit.v` and `SoftmaxErrorBound.v`, 96
//! theorems between them - sat in `proofs/` describing that exact kernel. That
//! is the same shape as the finding recorded for the exact GEMM before its
//! certificate landed: **proof-carrying described the repository, not the
//! output.**
//!
//! # What it instantiates, and what it does not
//!
//! It **instantiates**, never re-proves. The theorems are quantified over the
//! sequence length and the launch geometry already; what a compilation fixes is
//! the length, and the certificate states the one obligation that depends on
//! it.
//!
//! **The obligation bites, and that is the point of emitting it.** The kernel
//! reduces into a 64-bit accumulator over `red.global.add.u64`, so exactness
//! needs `S * (2^28 - 1) * 127 < 2^63`. Y decides that in `usize` arithmetic
//! ([`crate::exact_attention::MAX_EXACT_SEQ_LEN`]); the certificate states it
//! over `Z` and hands it to `coqc`, which has no `usize`. Verified at the edge
//! before this module was written: the certificate is **accepted at 270549122
//! and refused at 270549123**, one unit wide, which is the same boundary the
//! emitter refuses at and derived by a different tool from a different
//! representation.
//!
//! # The capstone is the dependency ROOT, and it was measured rather than guessed
//!
//! The rule this repository already established is that only a dependency root
//! can truthfully state a global negative, so the capstone is the file whose
//! exclusion list is the aggregate one. Measured: `AttentionSchedule` <-
//! `GridStrideSplit` <- `SoftmaxErrorBound`, and nothing requires the last. So
//! the GPU capstone is [`CAPSTONE`] - **not** `GridStrideSplit.v`, which is
//! what it looks like from the kernel's side.

use crate::exact_gemm_certificate::{render_trust_boundary_of, Check, TrustItem};

/// The compile-time parameters of one emitted attention kernel.
///
/// `KFix` is deliberately absent: it is a `.param .u32` read at LAUNCH, so no
/// compilation fixes it and the certificate states it as a precondition on the
/// caller instead. See [`TRUST_BOUNDARY`].
#[derive(Debug, Clone, Copy)]
pub struct Certificate {
    /// The per-lane vector width the kernel was emitted for.
    pub head_dim: usize,
    /// The sequence length, which is what the obligation is about.
    pub seq_len: usize,
}

/// The GPU chain's dependency root, and therefore the file whose exclusion
/// list [`TRUST_BOUNDARY`] must mirror.
pub const CAPSTONE: &str = "proofs/SoftmaxErrorBound.v";

/// The trusted computing base of the emitted attention certificate.
///
/// Five items mirror the capstone's own "what this does NOT claim" list; two
/// have no `stated_in` because they sit BELOW the model, and a proof over `Z`
/// and `Q` has no opinion about a closed-source assembler or a GPU. Omitting
/// those two would leave the certificate silent about the half of its boundary
/// a reader most needs - and one of them is the item Phase 3 names explicitly.
pub const TRUST_BOUNDARY: &[TrustItem] = &[
    TrustItem {
        claim: "That the abstract ideal weight `W` describes `2^-u/2^32`, the function \
                the kernel is approximating.",
        because: "`W` is constrained by four properties the true function has, and nothing \
                  in `proofs/` proves the true function has them. Two of the four are \
                  quantitative and their constants are derived rather than assumed, but \
                  believing the whole describes the real exponential is an assumption. It \
                  sits beside `vpdpwssd`'s semantics in the exact-GEMM certificate rather \
                  than above them.",
        check: Check::Unchecked(
            "a real-analytic development of `2^-x`, which over Coq's axiomatized `R` would \
             put axioms under a file whose whole gate is that it has none",
        ),
        stated_in: Some((CAPSTONE, "not a statement about `f64::exp2`")),
    },
    TrustItem {
        claim: "The accuracy of the integer exponential over the domain the kernel can reach.",
        because: "It is a HYPOTHESIS in the proof, discharged by exhaustion in Rust against \
                  `f64::exp2` over the swept domain - which is stronger than a proof about \
                  the table would be, because it is complete over a finite domain rather \
                  than a model of one. The extension from the swept domain to the admitted \
                  one is proved.",
        check: Check::Pinned("tests/softmax_error_bound.rs"),
        stated_in: Some((CAPSTONE, "The exp's accuracy is a hypothesis")),
    },
    TrustItem {
        claim: "That the caller passes a `KFix` in range. The kernel reads it as a \
                `.param .u32` at LAUNCH, so no compilation fixes it and this certificate \
                cannot instantiate it.",
        because: "Both failure modes are proved rather than supposed: `KFix = 0` makes every \
                  weight identical, so the softmax carries no information about the scores \
                  at all; and at `KFix >= 2^31` the SIGNED `mul.wide.s32` reads the \
                  parameter negative and a key's weight becomes ZERO - the key vanishes. \
                  `exact_attention::temperature_fixed_point` refuses both, and it is a HOST \
                  helper: a caller that computes the multiplier some other way is not \
                  checked by anything.",
        check: Check::Unchecked(
            "a check at the launch boundary rather than in a host helper the caller may not \
             use - the kernel cannot check a runtime parameter and say why",
        ),
        stated_in: Some((CAPSTONE, "`KFix` is itself a rounded temperature")),
    },
    TrustItem {
        claim: "Everything about int8 quantization of Q, K and V. The scores are taken as \
                given.",
        because: "This certificate is about what the kernel does with the scores it is \
                  handed, not about whether quantizing the inputs to produce them was a \
                  good idea. That is a different question with its own measurements.",
        check: Check::Pinned("tests/attention_quantization_error.rs"),
        stated_in: Some((CAPSTONE, "int8 quantization of Q, K or V")),
    },
    TrustItem {
        claim: "That the emitted PTX is the arithmetic the proofs describe.",
        because: "The tie is a MODEL checked against the running code - the test drives the \
                  real `exp2_neg_q16_16` and the real emitted arithmetic - and not the exact \
                  GEMM's byte-identity, where one description is rendered both to the IR and \
                  to the proofs. The launch geometry IS rendered that way \
                  (`exact_attention::sched_accum` feeds both), so that half is byte-identical \
                  and the arithmetic half is not.",
        check: Check::Pinned("tests/softmax_error_bound.rs"),
        stated_in: Some((CAPSTONE, "The tie to the kernel is")),
    },
    TrustItem {
        claim: "`ptxas`, which turns this PTX into the SASS the GPU runs. It is closed \
                source and is NOT covered by anything above.",
        because: "Every proof here stops at the PTX Y emits. `ptxas` is free to contract, \
                  reorder and re-materialise, and it demonstrably does: the same repository's \
                  translation validator REFUTES a float kernel because `ptxas` fuses \
                  `mul.f32`+`add.f32` into an `FFMA` that rounds once where the PTX rounds \
                  twice. That is a freedom the ISA grants, not a bug - which is exactly why \
                  it cannot be assumed away.",
        check: Check::Unchecked(
            "validating THIS kernel per-translation with `tools/ptxas_tval/`, which exists \
             and currently covers six kernels; this is not one of them, so for this kernel \
             `ptxas` is TRUSTED and not validated",
        ),
        stated_in: None,
    },
    TrustItem {
        claim: "The GPU executing its own ISA, and the driver loading the module it was \
                given.",
        because: "Below every model here. A proof over `Z` and `Q` has no opinion about \
                  silicon.",
        check: Check::Unchecked(
            "nothing available: this is the floor, and naming it is what makes the list \
             finite rather than merely trailing off",
        ),
        stated_in: None,
    },
];

/// A Coq module name for `stem`. Same sanitiser and same reason as the
/// exact-GEMM certificate's: `coqc` derives the logical name from the file
/// name, so it must be a legal identifier.
pub fn module_stem(stem: &str) -> String {
    crate::exact_gemm_certificate::module_stem(stem)
}

/// The file name this certificate is written to, without its extension.
pub fn file_stem(cert: &Certificate) -> String {
    format!(
        "attention_{}_{}_certificate",
        cert.head_dim, cert.seq_len
    )
}

/// Render the certificate for one emitted attention kernel.
pub fn render(cert: &Certificate, source: &str, stem: &str) -> String {
    let trust = render_trust_boundary_of(TRUST_BOUNDARY);
    let stem = module_stem(stem);
    let hd = cert.head_dim;
    let sl = cert.seq_len;
    format!(
        r#"(** * EXACT-ATTENTION CERTIFICATE - GENERATED BY THE Y COMPILER, DO NOT EDIT.

    Source      : {source}
    Kernel      : head_dim {hd}, seq_len {sl}
    Proofs      : `AttentionSchedule.v` -> `GridStrideSplit.v` -> `{cap}`
                  (the last is the dependency ROOT, which is why its exclusion
                  list is the one this certificate's trust boundary mirrors)

    WHAT THIS INSTANTIATES.  Nothing here is re-proved.  The theorems below are
    already quantified over the sequence length and over the launch geometry;
    what a compilation fixes is the LENGTH, and
    [the_accumulator_does_not_wrap] is the one obligation that depends on it.
    That obligation is decided TWICE by different tools from different
    representations - Y computes it in `usize`, `coqc` checks it over `Z` - and
    it is one unit wide, so the agreement is not a formality.

    WHAT IS NOT PROVED, and by whom it would be noticed:

{trust}
    Check with:

    <<  coqc -Q <dir with proofs/*.v> "" {stem}.v  >>
*)

From Stdlib Require Import ZArith Arith Lia List Permutation.
Require GridStrideSplit.
Require AttentionSchedule.
Open Scope Z_scope.

Module GS := GridStrideSplit.
Module AS := AttentionSchedule.

(** The sequence length this compilation emitted. *)
Definition seq_len_Z : Z := {sl}.

(** ... and the same length as the `nat` the schedule is stated over.

    Written as `Z.to_nat` of the integer rather than as a `nat` literal ON
    PURPOSE: a `nat` literal is UNARY, so at a production length this would be
    hundreds of millions of constructors and every tactic that normalises would
    try to evaluate it. *)
Definition seq_len : nat := Z.to_nat seq_len_Z.

(** The head dimension, which fixes the score delta span the error bound is
    priced against: `|s| <= 127^2 * head_dim` for int8 operands. *)
Definition head_dim_Z : Z := {hd}.

(** *** THE COMPILATION-SPECIFIC OBLIGATION.

    The kernel reduces into a 64-bit accumulator, and a weight is at most
    `2^28 - 1` against a `V` of at most 127 in magnitude.  Past this the sum
    WRAPS, which is a wrong answer rather than an imprecise one. *)
Theorem the_accumulator_does_not_wrap :
  seq_len_Z * (2 ^ 28 - 1) * 127 < 2 ^ 63.
Proof. unfold seq_len_Z. lia. Qed.

(** The two spellings of the length are the same number.  Without this the
    obligation above would be about `seq_len_Z` and the theorems below about a
    `seq_len` that need not be equal to it. *)
Theorem the_length_is_the_emitted_one : Z.of_nat seq_len = seq_len_Z.
Proof. unfold seq_len, seq_len_Z. rewrite Z2Nat.id; lia. Qed.

(** *** THE CERTIFICATE.

    At THIS sequence length, and at every launch geometry with at least one of
    everything, the grid-stride reduction visits every key exactly once and its
    partials sum to the naive sum. *)
Theorem this_kernel_visits_every_key_exactly_once :
  forall f ncz ncx ntx,
    (0 < ncz)%nat -> (0 < ncx)%nat -> (0 < ntx)%nat ->
    GS.combine Z.add f (AS.nworkers_accum ncz ncx ntx) seq_len
               (AS.nworkers_accum ncz ncx ntx)
      = GS.sum_upto Z.add f seq_len.
Proof.
  intros f ncz ncx ntx Hz Hx Ht.
  apply GS.the_emitted_launch_geometry_visits_every_key_exactly_once; assumption.
Qed.

(** And the atomics may land in any order at all.  This is the half no launch
    geometry can express, because it is a property of a RACE: the workers'
    partials reach the accumulator in whatever order the hardware chooses. *)
Theorem this_kernel_is_order_independent :
  forall f n order,
    (0 < n)%nat -> Permutation order (seq 0 n) ->
    fold_right Z.add 0 (map (fun w => GS.class_sum Z.add f w n seq_len) order)
      = GS.sum_upto Z.add f seq_len.
Proof. intros f n order Hn Hp. now apply GS.atomics_may_land_in_any_order. Qed.

(** *** Non-vacuity.

    `coqc` accepting a generated proof is necessary and NOT sufficient - the
    recorded instance is that `Theorem x : True` compiles and reports "Closed
    under the global context".  So the certificate EVALUATES the model on a
    concrete input at a concrete worker count, at a length that does not divide
    evenly by it. *)
Definition fcert (k : nat) : Z := Z.of_nat k.

Theorem the_certificate_is_not_vacuous :
  GS.combine Z.add fcert 3 7 3 = 21.
Proof. vm_compute. reflexivity. Qed.

Print Assumptions the_accumulator_does_not_wrap.
Print Assumptions the_length_is_the_emitted_one.
Print Assumptions this_kernel_visits_every_key_exactly_once.
Print Assumptions this_kernel_is_order_independent.
Print Assumptions the_certificate_is_not_vacuous.
"#,
        source = source,
        hd = hd,
        sl = sl,
        cap = CAPSTONE,
        trust = trust,
        stem = stem,
    )
}
