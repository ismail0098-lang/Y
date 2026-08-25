(** * The exact GEMM's PACKING, mechanically verified.

    Third and last of the schedule proofs. [ExactGemmKsplit.v] covers the K
    axis, [ExactGemmTiling.v] the output rectangle; this one covers the operand
    LAYOUT - what `pack_a` and `pack_b` put where, and why computing a full
    padded tile still yields the live dot product.

    *** The code being modelled. ***

    From [cpu_gemm::emit_vnni_pack_a] and [emit_vnni_pack_b], with
    `kpairs = (kc + 1) / 2`:

    << Ap[p*(2*MR) + 2*i + h]                      = A[i][2p+h]  if i < mrows and 2p+h < kc, else 0
       Bp[p*(2*NR) + (j/16)*32 + (j%16)*2 + h]     = B[2p+h][j]  if j < ncols and 2p+h < kc, else 0 >>

    Both packers write EVERY slot of the panel unconditionally and use a
    `select` to supply 0 outside the live region, and both clamp the source
    address to 0 when out of range so nothing is read past the operand. The
    micro-kernel then computes the FULL `MR x NR x kpairs` product regardless of
    how ragged the tile is.

    *** What is proved. ***

    - [pack_a_slot_bijective] / [pack_b_slot_bijective] : each packer's
      destination index is a bijection onto its panel. Nothing is written twice
      and no slot is left holding whatever was there before - which matters
      because the panel buffer is reused across tiles.
    - [padded_product_is_the_live_dot_product] : THE theorem. Summing the full
      padded `kpairs x 2` product gives exactly the dot product over the live
      `kc`. This is what makes the ragged tail correct, and it covers three
      separate paddings at once: rows past `mrows`, columns past `ncols`, and
      the phantom `k = kc` that an ODD `kc` introduces when `kpairs` rounds up.
    - [garbage_in_the_pad_changes_the_answer] : the same statement with the
      zero-fill removed is FALSE. The `select ... i16 0` is load-bearing, not
      defensive.

    *** What this does NOT prove, and it is the interesting half. ***

    **That the lane numbering matches `vpdpwssd`.** And the gap is wider than
    it first looks. [slot_b_is_the_plain_interleave] proves that the emitted
    `(j/16)*32 + (j%16)*2 + h` is not merely *indistinguishable from* the naive
    `2*j + h` by bijectivity - it IS `2*j + h`, identically, because
    `16*(j/16) + (j mod 16) = j`. The vector-group decomposition is a
    DERIVATION of that interleave, written in the emitter to document which
    `<32 x i16>` vector and which of its 16 lanes a column lands in; it folds
    to a shift-free doubling and computes nothing extra.

    So there is no arithmetic content here for a proof to capture at all. That
    a lane consumes int16 elements `2l` and `2l+1` of its own vector is an ISA
    fact, and no proof over [Z] can establish it; it is pinned empirically by
    [tests/cpu_gemm_vnni_micro.rs], which mutates the stride and checks the
    result against a scalar reference on the real instruction.

    So the division of labour is deliberate: the proof says nothing is lost,
    doubled, or contaminated by padding, and the differential says the layout
    is the one the hardware wants.

    Also, as in the sibling files: this is a proof about a MODEL. There is no
    extraction. [tests/exact_gemm_packing_model.rs] is what ties it to the Rust.

    Build:  coqc proofs/ExactGemmPacking.v      (Rocq 9.1)
*)

From Stdlib Require Import ZArith Arith Lia.

Open Scope Z_scope.

(** The micro-kernel's tile. `VNNI_MR` and `VNNI_NR` in `cpu_gemm.rs`. *)
Definition MR : nat := 6.
Definition NR : nat := 64.

(* ------------------------------------------------------------------ *)
(** ** Destination indices                                             *)
(* ------------------------------------------------------------------ *)

Definition slot_a (i h : nat) : nat := (2 * i + h)%nat.

(** The `vpdpwssd` lane layout: accumulator group `v = j / 16` is one
    <32 x i16> vector, and lane `l = j mod 16` inside it consumes int16
    elements `2l` and `2l + 1`. So the two k-values for a column are ADJACENT
    and consecutive columns are 2 int16 apart, not 1. *)
Definition slot_b (j h : nat) : nat := ((j / 16) * 32 + (j mod 16) * 2 + h)%nat.

(** Quotient and remainder are unique. Proved locally rather than imported:
    it is six lines, and a lemma cannot drift into being wrong the way a
    duplicated CONSTANT can - it is re-proved wherever it is stated. *)
Lemma quot_rem_unique : forall B q1 r1 q2 r2,
  (0 < B)%nat -> (r1 < B)%nat -> (r2 < B)%nat ->
  (q1 * B + r1 = q2 * B + r2)%nat -> q1 = q2 /\ r1 = r2.
Proof.
  intros B q1 r1 q2 r2 HB H1 H2 Heq.
  assert (E1 : ((q1 * B + r1) / B = q1)%nat).
  { rewrite Nat.div_add_l by lia. rewrite Nat.div_small by lia. lia. }
  assert (E2 : ((q2 * B + r2) / B = q2)%nat).
  { rewrite Nat.div_add_l by lia. rewrite Nat.div_small by lia. lia. }
  assert (q1 = q2) by (rewrite <- E1, <- E2, Heq; reflexivity).
  split; [assumption | subst; lia].
Qed.

Theorem pack_a_slot_bijective : forall i1 h1 i2 h2,
  (i1 < MR)%nat -> (i2 < MR)%nat -> (h1 < 2)%nat -> (h2 < 2)%nat ->
  slot_a i1 h1 = slot_a i2 h2 -> i1 = i2 /\ h1 = h2.
Proof.
  intros i1 h1 i2 h2 _ _ H1 H2 Heq. unfold slot_a in Heq.
  apply (quot_rem_unique 2); lia.
Qed.

Theorem pack_a_slot_in_panel : forall i h,
  (i < MR)%nat -> (h < 2)%nat -> (slot_a i h < 2 * MR)%nat.
Proof. intros i h Hi Hh. unfold slot_a, MR in *. lia. Qed.

(** Every slot of an A panel is written: the map is onto, so no slot keeps a
    value from the previous tile. The panel buffer really is reused. *)
Theorem pack_a_slot_onto : forall s,
  (s < 2 * MR)%nat -> exists i h, (i < MR)%nat /\ (h < 2)%nat /\ slot_a i h = s.
Proof.
  intros s Hs. exists (s / 2)%nat, (s mod 2)%nat.
  pose proof (Nat.div_mod_eq s 2) as Hdm.
  pose proof (Nat.mod_upper_bound s 2 ltac:(lia)) as Hmod.
  unfold slot_a, MR in *. repeat split; lia.
Qed.

Theorem pack_b_slot_bijective : forall j1 h1 j2 h2,
  (j1 < NR)%nat -> (j2 < NR)%nat -> (h1 < 2)%nat -> (h2 < 2)%nat ->
  slot_b j1 h1 = slot_b j2 h2 -> j1 = j2 /\ h1 = h2.
Proof.
  intros j1 h1 j2 h2 Hj1 Hj2 Hh1 Hh2 Heq.
  unfold slot_b, NR in *.
  pose proof (Nat.div_mod_eq j1 16) as D1.
  pose proof (Nat.div_mod_eq j2 16) as D2.
  pose proof (Nat.mod_upper_bound j1 16 ltac:(lia)) as M1.
  pose proof (Nat.mod_upper_bound j2 16 ltac:(lia)) as M2.
  assert (Hq1 : (j1 / 16 < 4)%nat) by (apply Nat.Div0.div_lt_upper_bound; lia).
  assert (Hq2 : (j2 / 16 < 4)%nat) by (apply Nat.Div0.div_lt_upper_bound; lia).
  (* The index is a three-digit mixed-radix number: group (radix 4), lane
     (radix 16), half (radix 2). Peel one digit at a time. *)
  assert (Hg : (j1 / 16 = j2 / 16)%nat /\ ((j1 mod 16) * 2 + h1 = (j2 mod 16) * 2 + h2)%nat).
  { apply (quot_rem_unique 32); lia. }
  destruct Hg as [Hgrp Hrest].
  assert (Hl : (j1 mod 16 = j2 mod 16)%nat /\ h1 = h2)
    by (apply (quot_rem_unique 2); lia).
  destruct Hl as [Hlane Hh]. split; [lia | exact Hh].
Qed.

(** The emitted vector-group form is the plain interleave.

    `16 * (j / 16) + (j mod 16) = j`, so the whole `(v, l)` decomposition
    collapses. This is stated as a theorem rather than left as a remark
    because it is the precise statement of what the packing proof CANNOT
    say about lane layout: there is no arithmetic difference to say it about.
    It also makes the emitter's constants checkable - changing the `32` or
    the `16` to any inconsistent pair breaks this equality, and the Rust
    transcription asserts the same thing against [pack_b_slot]. *)
Theorem slot_b_is_the_plain_interleave : forall j h,
  slot_b j h = (2 * j + h)%nat.
Proof.
  intros j h. unfold slot_b.
  pose proof (Nat.div_mod_eq j 16) as Hdm.
  lia.
Qed.

Theorem pack_b_slot_in_panel : forall j h,
  (j < NR)%nat -> (h < 2)%nat -> (slot_b j h < 2 * NR)%nat.
Proof.
  intros j h Hj Hh. unfold slot_b, NR in *.
  pose proof (Nat.mod_upper_bound j 16 ltac:(lia)).
  assert ((j / 16 < 4)%nat) by (apply Nat.Div0.div_lt_upper_bound; lia).
  lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The padded product                                              *)
(* ------------------------------------------------------------------ *)

(** Sum over `k < n`. *)
Fixpoint sum_k (f : nat -> Z) (n : nat) : Z :=
  match n with
  | O => 0
  | S n' => sum_k f n' + f n'
  end.

(** Sum over `p < n` of both halves - the shape the micro-kernel's k-pair loop
    actually has. *)
Fixpoint sum_pairs (g : nat -> nat -> Z) (n : nat) : Z :=
  match n with
  | O => 0
  | S n' => sum_pairs g n' + (g n' 0%nat + g n' 1%nat)
  end.

Lemma pairs_flatten : forall f n,
  sum_pairs (fun p h => f (2 * p + h)%nat) n = sum_k f (2 * n).
Proof.
  intros f n. induction n as [| n IH]; cbn [sum_pairs].
  - reflexivity.
  - rewrite IH. replace (2 * S n)%nat with (S (S (2 * n)))%nat by lia.
    cbn [sum_k].
    replace (2 * n + 0)%nat with (2 * n)%nat by lia.
    replace (2 * n + 1)%nat with (S (2 * n))%nat by lia.
    ring.
Qed.

Lemma sum_k_ext : forall f g n,
  (forall k, (k < n)%nat -> f k = g k) -> sum_k f n = sum_k g n.
Proof.
  intros f g n. induction n as [| n IH]; intros Hfg; cbn [sum_k].
  - reflexivity.
  - rewrite IH by (intros k Hk; apply Hfg; lia). rewrite Hfg by lia. reflexivity.
Qed.

(** Terms past the live `kc` are zero, so extending the range costs nothing. *)
Lemma sum_k_masked : forall f kc n,
  (kc <= n)%nat ->
  sum_k (fun k => if Nat.ltb k kc then f k else 0) n = sum_k f kc.
Proof.
  intros f kc n. induction n as [| n IH]; intros Hle.
  - assert (kc = 0)%nat by lia. subst. reflexivity.
  - cbn [sum_k]. destruct (Nat.ltb_spec n kc) as [Hlt | Hge].
    + (* kc <= S n and n < kc, so this is the last live index *)
      assert (kc = S n) by lia. subst.
      rewrite (sum_k_ext (fun k => if Nat.ltb k (S n) then f k else 0) f n)
        by (intros k Hk; destruct (Nat.ltb_spec k (S n)); [reflexivity | lia]).
      cbn [sum_k]. reflexivity.
    + rewrite IH by lia. ring.
Qed.

(** The live element, or zero outside. Exactly the emitted
    `select i1 %ok, i16 %raw, i16 0`. *)
Definition packed (v : nat -> Z) (extent kc : nat) (idx k : nat) : Z :=
  if andb (Nat.ltb idx extent) (Nat.ltb k kc) then v k else 0.

Definition kpairs (kc : nat) : nat := ((kc + 1) / 2)%nat.

Lemma kpairs_covers : forall kc, (kc <= 2 * kpairs kc)%nat.
Proof.
  intros kc. unfold kpairs.
  pose proof (Nat.div_mod_eq (kc + 1) 2) as Hdm.
  pose proof (Nat.mod_upper_bound (kc + 1) 2 ltac:(lia)) as Hmod.
  lia.
Qed.

(** ** THE THEOREM.

    The micro-kernel computes a full `kpairs x 2` product with no knowledge of
    how ragged the tile is. Under the packers' zero-fill that equals the dot
    product over the live `kc` - which is what licenses running the full tile.

    [arow] and [bcol] are the row of A and the column of B as functions of k,
    so this is stated for one (i, j) of the tile. The hypotheses are that the
    row and column are live; the k axis needs no hypothesis, because the
    zero-fill handles both the ragged `kc` and the phantom `k` an odd `kc`
    introduces. *)
Theorem padded_product_is_the_live_dot_product :
  forall (arow bcol : nat -> Z) (i j mrows ncols kc : nat),
    (i < mrows)%nat -> (j < ncols)%nat ->
    sum_pairs
      (fun p h =>
         packed arow mrows kc i (2 * p + h)%nat
         * packed bcol ncols kc j (2 * p + h)%nat)
      (kpairs kc)
    = sum_k (fun k => arow k * bcol k) kc.
Proof.
  intros arow bcol i j mrows ncols kc Hi Hj.
  rewrite (pairs_flatten
             (fun k => packed arow mrows kc i k * packed bcol ncols kc j k)).
  rewrite (sum_k_ext _ (fun k => if Nat.ltb k kc then arow k * bcol k else 0)).
  - apply sum_k_masked. apply kpairs_covers.
  - intros k _. unfold packed.
    apply Nat.ltb_lt in Hi as Hi'. apply Nat.ltb_lt in Hj as Hj'.
    rewrite Hi', Hj'. cbn [andb].
    destruct (Nat.ltb k kc); ring.
Qed.

(** A dead row or column contributes nothing, which is the other half of why a
    ragged tile is safe. *)
Theorem a_dead_row_contributes_nothing :
  forall (arow bcol : nat -> Z) (i j mrows ncols kc : nat),
    (mrows <= i)%nat ->
    sum_pairs
      (fun p h =>
         packed arow mrows kc i (2 * p + h)%nat
         * packed bcol ncols kc j (2 * p + h)%nat)
      (kpairs kc)
    = 0.
Proof.
  intros arow bcol i j mrows ncols kc Hi.
  rewrite (pairs_flatten
             (fun k => packed arow mrows kc i k * packed bcol ncols kc j k)).
  rewrite (sum_k_ext _ (fun _ => 0)).
  - induction (2 * kpairs kc)%nat as [| n IH]; cbn [sum_k]; [reflexivity | rewrite IH; ring].
  - intros k _. unfold packed.
    replace (Nat.ltb i mrows) with false by (symmetry; apply Nat.ltb_ge; lia).
    cbn [andb]. ring.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Refutation: the zero-fill is load-bearing                       *)
(* ------------------------------------------------------------------ *)

(** `pack_a` writes `select i1 %ok, i16 %raw, i16 0`, and `pack_b` the same.
    Reading as it does past the live region and NOT masking - the natural
    "optimisation", since the buffer is about to be overwritten anyway - makes
    the padded product disagree with the live one.

    Concretely: kc = 1, so kpairs = 1 and the pair's high half is the phantom
    k = 1. With the mask, the answer is `arow 0 * bcol 0` = 6. Without it, the
    phantom term 5*7 is added. *)

Definition unmasked (v : nat -> Z) (_ _ : nat) (_ k : nat) : Z := v k.

Definition a_vals (k : nat) : Z := if Nat.eqb k 0 then 2 else 5.
Definition b_vals (k : nat) : Z := if Nat.eqb k 0 then 3 else 7.

Theorem garbage_in_the_pad_changes_the_answer :
  sum_pairs
    (fun p h => unmasked a_vals 1 1 0 (2 * p + h)%nat
                * unmasked b_vals 1 1 0 (2 * p + h)%nat)
    (kpairs 1)
  <> sum_k (fun k => a_vals k * b_vals k) 1.
Proof. vm_compute. discriminate. Qed.

(** ...and with the zero-fill it agrees, on the same numbers. Without this the
    refutation above is satisfied by a model that computes nothing. *)
Theorem the_masked_version_agrees_on_the_same_input :
  sum_pairs
    (fun p h => packed a_vals 1 1 0 (2 * p + h)%nat
                * packed b_vals 1 1 0 (2 * p + h)%nat)
    (kpairs 1)
  = sum_k (fun k => a_vals k * b_vals k) 1.
Proof. vm_compute. reflexivity. Qed.

Print Assumptions pack_a_slot_bijective.
Print Assumptions pack_a_slot_onto.
Print Assumptions pack_b_slot_bijective.
Print Assumptions slot_b_is_the_plain_interleave.
Print Assumptions pack_b_slot_in_panel.
Print Assumptions padded_product_is_the_live_dot_product.
Print Assumptions a_dead_row_contributes_nothing.
Print Assumptions garbage_in_the_pad_changes_the_answer.
Print Assumptions the_masked_version_agrees_on_the_same_input.
