(** * The checked exact_pv launch domain implies the arithmetic licences.

    This connects the positive shape/product/accumulator checks in
    [verified_exact_pv::ExactPvShape] to the existing dot-product theorem.
    It does not prove the Rust checker, buffer ownership, launch API, decoder,
    PTX control flow, output stores or hardware. Those are separate boundaries.
    Zero T is excluded because the imported capstone requires T >= 1. *)

From Stdlib Require Import ZArith Lia.
Require ExactPvExact ExactGemmMicro.

Module EP := ExactPvExact.
Module MC := ExactGemmMicro.
Open Scope Z_scope.

Definition checked_shape (B Q T D : Z) : Prop :=
  1 <= B /\ 1 <= Q /\ 1 <= T /\ 1 <= D /\
  B * Q * T <= MC.I32MAX /\
  B * T * D <= MC.I32MAX /\
  B * Q * D <= MC.I32MAX /\
  T * (EP.PMAX * EP.VABS) <= EP.I64MAX.

Theorem checked_shape_licenses_every_live_index :
  forall B Q T D b q d t,
    checked_shape B Q T D ->
    0 <= b < B -> 0 <= q < Q -> 0 <= d < D -> 0 <= t < T ->
    0 <= (b * Q + q) * T + t < B * Q * T /\
    0 <= (b * T + t) * D + d < B * T * D /\
    0 <= (b * Q + q) * D + d < B * Q * D /\
    (b * Q + q) * T + t <= MC.I32MAX /\
    (b * T + t) * D + d <= MC.I32MAX /\
    (b * Q + q) * D + d <= MC.I32MAX.
Proof.
  intros B Q T D b q d t
    [HB [HQ [HT [HD [HP [HV [HO HA]]]]]]] Hb Hq Hd Ht.
  assert (Hbq : 0 <= b * Q + q < B * Q) by nia.
  assert (Hbt : 0 <= b * T + t < B * T) by nia.
  assert (Hp : 0 <= (b * Q + q) * T + t < B * Q * T) by nia.
  assert (Hv : 0 <= (b * T + t) * D + d < B * T * D) by nia.
  assert (Ho : 0 <= (b * Q + q) * D + d < B * Q * D) by nia.
  repeat split; lia.
Qed.

Theorem checked_shape_licenses_the_output_index :
  forall B Q T D b q d,
    checked_shape B Q T D ->
    0 <= b < B -> 0 <= q < Q -> 0 <= d < D ->
    EP.o_index b q Q D d = (b * Q + q) * D + d.
Proof.
  intros B Q T D b q d Hshape Hb Hq Hd.
  pose proof (checked_shape_licenses_every_live_index B Q T D b q d 0
    Hshape Hb Hq Hd) as Hidx.
  destruct Hshape as [HB [HQ [HT [HD [HP [HV [HO HA]]]]]]].
  specialize (Hidx ltac:(lia)).
  rewrite EP.the_output_index_is_the_mathematical_one by (try lia; tauto).
  ring.
Qed.

Theorem checked_shape_instantiates_the_source_dot_product :
  forall Pf Vf B Q T D b q d (n : nat),
    checked_shape B Q T D ->
    0 <= b < B -> 0 <= q < Q -> 0 <= d < D ->
    Z.of_nat n = T ->
    (forall z, EP.p_domain (Pf z)) ->
    (forall z, EP.v_domain (Vf z)) ->
    EP.wacc (EP.emitted_term Pf Vf b q Q T D d (B * Q * T) (B * T * D)) n
      = EP.sum_upto (EP.src_term Pf Vf b q Q T D d) n.
Proof.
  intros Pf Vf B Q T D b q d n Hshape Hb Hq Hd Hn HPdom HVdom.
  pose proof Hshape as [HB [HQ [HT [HD [HP [HV [HO HA]]]]]]].
  apply EP.the_emitted_exact_pv_holds_the_source_dot_product; try lia; try assumption.
  all: intros t Ht.
  all: assert (Htz : 0 <= Z.of_nat t < T)
    by (split; [apply Nat2Z.is_nonneg | apply Nat2Z.inj_lt in Ht; lia]).
  all: pose proof (checked_shape_licenses_every_live_index B Q T D b q d (Z.of_nat t)
    Hshape Hb Hq Hd Htz) as Hidx; tauto.
Qed.

(** Geometry chooses precisely these coordinates. Coverage and ownership are
    mathematical map facts, not statements that a CUDA launch writes them. *)
Theorem geometry_covers_the_output_rectangle :
  forall B Q D slot : nat,
    (0 < Q)%nat -> (0 < D)%nat -> (slot < B * Q * D)%nat ->
    exists b q d, (b < B)%nat /\ (q < Q)%nat /\ (d < D)%nat /\
      ((b * Q + q) * D + d = slot)%nat.
Proof.
  intros B Q D slot HQ HD Hslot.
  destruct (MixedRadix.pack_onto D (B * Q) slot HD Hslot)
    as [row [d [Hr [Hd Hrow]]]].
  destruct (MixedRadix.pack_onto Q B row HQ Hr)
    as [b [q [Hb [Hq Hbq]]]].
  unfold MixedRadix.pack in *.
  exists b, q, d. repeat split; try assumption. nia.
Qed.

Theorem geometry_output_coordinates_are_unique :
  forall Q D b1 q1 d1 b2 q2 d2 : nat,
    (0 < Q)%nat -> (0 < D)%nat ->
    (q1 < Q)%nat -> (q2 < Q)%nat -> (d1 < D)%nat -> (d2 < D)%nat ->
    ((b1 * Q + q1) * D + d1 = (b2 * Q + q2) * D + d2)%nat ->
    b1 = b2 /\ q1 = q2 /\ d1 = d2.
Proof.
  intros Q D b1 q1 d1 b2 q2 d2 HQ HD Hq1 Hq2 Hd1 Hd2 Heq.
  apply (EP.every_output_element_is_written_by_one_thread Q D); try assumption; nia.
Qed.

(** The checked byte-span test excludes output overlap with either input.
    P/V overlap is deliberately permitted because neither is written. *)
Theorem disjoint_live_byte_spans_never_alias :
  forall input input_bytes output output_bytes i o : Z,
    0 <= i < input_bytes -> 0 <= o < output_bytes ->
    (input + input_bytes <= output \/ output + output_bytes <= input) ->
    input + i <> output + o.
Proof. intros input input_bytes output output_bytes i o Hi Ho Hsep. destruct Hsep; lia. Qed.

Print Assumptions checked_shape_licenses_every_live_index.
Print Assumptions checked_shape_licenses_the_output_index.
Print Assumptions checked_shape_instantiates_the_source_dot_product.
Print Assumptions geometry_covers_the_output_rectangle.
Print Assumptions geometry_output_coordinates_are_unique.
Print Assumptions disjoint_live_byte_spans_never_alias.
