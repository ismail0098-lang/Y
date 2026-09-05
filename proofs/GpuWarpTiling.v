(** * The GPU warp tiling: a CTA tile's rows are written exactly once.

    The CPU chain has had [ExactGemmTiling.c_written_exactly_once] since the
    tiling increment.  The GPU tensor-core GEMMs had no equivalent, and the
    precondition their tiling rests on was three [debug_assert_eq!]s - which
    [Cargo.toml] compiles out, since it declares no [[profile.release]] and
    rustc's default there is [debug-assertions = false].

    ** The defect this file is about, measured before it was written

    Every tensor-core GEMM here derives its warp geometry by TRUNCATING
    integer division:

    << per_warp = cta / warps ;   num = per_warp / frag >>

    so a warp's base advances by [per_warp] while a warp WRITES [num * frag].
    Those two numbers are equal exactly when [frag * warps] divides [cta].
    When it does not, the emitter still produces a well-formed kernel, [ptxas]
    still accepts it, and the rows in between keep whatever the output buffer
    held.  On [tests/gemm_f16_1024.ysu] with
    [Y_CTA_OVERRIDE=96,128,32,4,2,3]: stride 24, [num = 1], so each CTA
    advances 96 rows and writes 64 - gaps at rows 16-23, 40-47, 64-71, 88-95 -
    under "Compilation Successful!" and exit 0.

    [the_measured_gap] and [the_measured_hole_count] state that instance, so
    the refutation in this file is the program that was actually run rather
    than an illustration.

    ** What is proved

    - [rows_in_range], [rows_injective], [rows_onto], and
      [cta_rows_written_exactly_once]: under divisibility the warp tiles
      PARTITION the CTA tile's rows on one axis.  Injectivity is
      [MixedRadix.two_digit_unique] - a warp row is a two-digit positional
      index and needs no new reasoning, the fourth consumer of that schema.
    - [the_stride_is_the_written_extent]: divisibility is exactly the
      condition under which the emitter's two numbers agree.
    - [the_emitted_count_is_the_model]: the emitter's truncating [num]
      is the model's, so the theorems are about the arithmetic Y performs.

    ** What is NOT proved

    - One axis at a time.  M and N are independent instances of the same
      section; K is a reduction, not a partition, and is [ExactGemmKsplit]'s
      shape rather than this one.
    - Nothing here is about the VALUE written.  These GEMMs are f16 or fp8,
      so their accumulation is floating point and not associative - an
      equality to a naive nest is not available for them at all.  This is a
      claim about the SCHEDULE, which is the half that does not depend on
      precision, and it is the half the measured defect was in.
    - The tie is the emitter's own arithmetic transcribed here and gated by
      [tests/gemm_tile_partition.rs], not the exact GEMM's byte-identity:
      [src/ptx_emitter.rs] does not go through the [Ix] extraction layer
      ([exact_attention.rs] does, this does not).

    Build:  coqc -Q proofs "" proofs/GpuWarpTiling.v      (Rocq 9.1)
*)

From Stdlib Require Import Arith Lia.
Require MixedRadix.
Module MR := MixedRadix.

(** ** The emitter's arithmetic, transcribed *)

(** [per_warp_m = cta_m / warps_m] in every one of the three GEMM emitters. *)
Definition emit_stride (cta warps : nat) : nat := cta / warps.

(** [num_i = per_warp_m / 16], with the fragment extent a parameter because
    the three consumers disagree: F16 and SwiGLU are m16n16k16, FP8 m16n8k32. *)
Definition emit_num (cta warps frag : nat) : nat := (cta / warps) / frag.

(** The row a warp actually writes: base [w * stride], fragment [i], lane [r]. *)
Definition row (stride frag w i r : nat) : nat := w * stride + i * frag + r.

(** ** Under divisibility the warp tiles partition the CTA tile *)

Section Partition.
  Variables cta warps frag : nat.
  Hypothesis warps_pos : 0 < warps.
  Hypothesis frag_pos  : 0 < frag.
  Hypothesis cta_pos   : 0 < cta.
  Hypothesis divides   : cta = warps * (emit_num cta warps frag * frag).

  Let stride := emit_stride cta warps.
  Let num    := emit_num cta warps frag.

  (** [divides] is stated with [emit_num ...] because that is what the caller
      can test.  Restated over the section's [Let] so the rewrites below see
      the same term the goals do. *)
  Lemma divides_num : cta = warps * (num * frag).
  Proof. unfold num. exact divides. Qed.

  (** The two numbers the emitter derives independently agree.  This is the
      whole content of the precondition: [stride] is how far the next warp's
      base moves and [num * frag] is how much this warp wrote. *)
  Theorem the_stride_is_the_written_extent : stride = num * frag.
  Proof.
    unfold stride, num, emit_stride.
    rewrite divides at 1.
    rewrite (Nat.mul_comm warps (emit_num cta warps frag * frag)).
    apply Nat.div_mul. lia.
  Qed.

  Theorem rows_in_range :
    forall w i r, w < warps -> i < num -> r < frag -> row stride frag w i r < cta.
  Proof.
    intros w i r Hw Hi Hr. unfold row.
    rewrite the_stride_is_the_written_extent, divides_num.
    apply Nat.lt_le_trans with (m := (S w) * (num * frag)); [| apply Nat.mul_le_mono_r; lia].
    assert (i * frag + r < num * frag).
    { apply Nat.lt_le_trans with (m := S i * frag); [simpl; lia|].
      apply Nat.mul_le_mono_r; lia. }
    simpl; lia.
  Qed.

  (** Injectivity is a two-digit positional index and nothing else: the
      fourth consumer of [MixedRadix], reached from a GPU warp geometry. *)
  Theorem rows_injective :
    forall w1 i1 r1 w2 i2 r2,
      w1 < warps -> i1 < num -> r1 < frag ->
      w2 < warps -> i2 < num -> r2 < frag ->
      row stride frag w1 i1 r1 = row stride frag w2 i2 r2 ->
      w1 = w2 /\ i1 = i2 /\ r1 = r2.
  Proof.
    intros w1 i1 r1 w2 i2 r2 Hw1 Hi1 Hr1 Hw2 Hi2 Hr2 Heq.
    unfold row in Heq. rewrite the_stride_is_the_written_extent in Heq.
    assert (0 < num) by lia.
    apply (MR.two_digit_unique frag num w1 i1 r1 w2 i2 r2); auto.
  Qed.

  Theorem rows_onto :
    forall x, x < cta ->
    exists w i r, w < warps /\ i < num /\ r < frag /\ row stride frag w i r = x.
  Proof.
    intros x Hx. rewrite the_stride_is_the_written_extent.
    assert (Hd := divides_num).
    assert (Hnf : 0 < num * frag).
    { destruct (Nat.eq_dec (num * frag) 0) as [E|N].
      - rewrite E, Nat.mul_0_r in Hd. lia.
      - lia. }
    set (S := num * frag) in *.
    exists (x / S), (x mod S / frag), (x mod S mod frag).
    assert (H1 := Nat.div_mod_eq x S).
    assert (H2 := Nat.div_mod_eq (x mod S) frag).
    repeat split.
    - apply Nat.Div0.div_lt_upper_bound. rewrite Nat.mul_comm, <- Hd. exact Hx.
    - apply Nat.Div0.div_lt_upper_bound.
      rewrite Nat.mul_comm. apply Nat.mod_upper_bound; lia.
    - apply Nat.mod_upper_bound; lia.
    - unfold row. nia.
  Qed.

  (** The GPU twin of [ExactGemmTiling.c_written_exactly_once]. *)
  Theorem cta_rows_written_exactly_once :
    forall x, x < cta ->
    exists! t, let '(w, i, r) := t in
      w < warps /\ i < num /\ r < frag /\ row stride frag w i r = x.
  Proof.
    intros x Hx.
    destruct (rows_onto x Hx) as [w [i [r [Hw [Hi [Hr He]]]]]].
    exists (w, i, r). split; [tauto|].
    intros [[w' i'] r'] [Hw' [Hi' [Hr' He']]].
    destruct (rows_injective w' i' r' w i r Hw' Hi' Hr' Hw Hi Hr) as [E1 [E2 E3]];
      [lia | subst; reflexivity].
  Qed.
End Partition.

(** ** The refutation: the tile that was actually compiled *)

(** [Y_CTA_OVERRIDE=96,128,32,4,2,3] gives stride 24 and [num = 1], so the
    written rows are [{0..15} u {24..39} u {48..63} u {72..87}] and row 16 -
    the first row of the first gap - is written by no warp. *)
Theorem the_measured_gap :
  emit_stride 96 4 = 24 /\ emit_num 96 4 16 = 1 /\
  ~ (exists w i r, w < 4 /\ i < emit_num 96 4 16 /\ r < 16 /\
                   row (emit_stride 96 4) 16 w i r = 16).
Proof.
  assert (Hs : emit_stride 96 4 = 24) by (vm_compute; reflexivity).
  assert (Hn : emit_num 96 4 16 = 1) by (vm_compute; reflexivity).
  repeat split; try assumption.
  rewrite Hs, Hn. intros [w [i [r [Hw [Hi [Hr He]]]]]].
  unfold row in He. lia.
Qed.

(** 32 of the 96 rows, in four gaps of 8 - which is what was observed. *)
Theorem the_measured_hole_count :
  4 * (emit_num 96 4 16 * 16) = 64 /\ 96 - 4 * (emit_num 96 4 16 * 16) = 32.
Proof. unfold emit_num. split; vm_compute; reflexivity. Qed.

(** And the same tile with a legal [cta_m] has no gap - the control, without
    which "the tiling has holes" could be a property of the theorem rather
    than of the tile. *)
Theorem the_legal_tile_has_no_gap :
  emit_stride 64 4 = 16 /\ emit_num 64 4 16 = 1 /\
  64 = 4 * (emit_num 64 4 16 * 16).
Proof. unfold emit_stride, emit_num. repeat split; vm_compute; reflexivity. Qed.

(** ** The tie: divisibility is exactly what the compiler now checks *)

(** [autotuner::validate_warp_tiling] refuses unless [frag * warps] divides
    [cta].  That is the [divides] hypothesis above, in the form the emitter
    can test before it has computed anything. *)
Theorem the_emitted_count_is_the_model :
  forall cta warps frag,
    0 < warps -> 0 < frag ->
    cta mod (frag * warps) = 0 ->
    cta = warps * (emit_num cta warps frag * frag).
Proof.
  intros cta warps frag Hw Hf Hmod. unfold emit_num.
  apply Nat.Lcm0.mod_divide in Hmod. destruct Hmod as [q Hq]. subst cta.
  rewrite (Nat.mul_assoc q frag warps).
  rewrite (Nat.div_mul (q * frag) warps) by lia.
  rewrite (Nat.div_mul q frag) by lia.
  nia.
Qed.

Print Assumptions the_stride_is_the_written_extent.
Print Assumptions rows_in_range.
Print Assumptions rows_injective.
Print Assumptions rows_onto.
Print Assumptions cta_rows_written_exactly_once.
Print Assumptions the_measured_gap.
Print Assumptions the_measured_hole_count.
Print Assumptions the_legal_tile_has_no_gap.
Print Assumptions the_emitted_count_is_the_model.
