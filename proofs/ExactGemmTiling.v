(** * The exact GEMM's OUTPUT tiling, mechanically verified.

    Companion to [ExactGemmKsplit.v], which proves the K axis. That one is a
    REDUCTION - the bands are summed, so the obligation is that they tile the
    reduction range. This one is a PARTITION OF THE OUTPUT - each tile writes a
    disjoint rectangle of C, so the obligation is that every element is written
    EXACTLY ONCE. Same programme, genuinely different theorem: the K-split's
    bands are uneven by one, while these tiles are uniform with a clamped
    ragged tail, and "sums to K" is not the same claim as "is a bijection".

    *** The code being modelled. ***

    From [cpu_gemm::emit_vnni_gemm_driver]:

    << for j0 in (0, N, NR):
         nw = min(N - j0, NR)
         for i0 in (0, M, MR):
           mw = min(M - i0, MR)
           zero Ct[0 .. MR*NR)
           micro(...)                       (* writes the full MR x NR tile *)
           for fi < mw: for fj < nw:
             C[(i0+fi)*ldc + (j0+fj)] += Ct[fi*NR + fj]        >>

    The micro-kernel always writes a full tile into scratch and only the live
    part is folded back, and the emitter's own comment says why: "Letting it
    write C directly would run past the last row and column whenever M or N is
    not a multiple of the tile - an out-of-bounds WRITE, not a wrong number."
    [unclamped_tail_writes_out_of_bounds] turns that comment into a
    machine-checked refutation.

    *** What is proved. ***

    - [tiles_cover] : the 1-D tiling covers [0, extent) exactly. Applied twice,
      to rows with MR and to columns with NR.
    - [tile_index_injective] / [tile_index_surjective] : (tile, offset) is a
      BIJECTION onto [0, extent). This is the "exactly once" obligation; a
      coverage count alone is satisfied by a tiling that writes one element
      twice and another never.
    - [c_written_exactly_once] : the 2-D consequence - the linear index
      (i0+fi)*ldc + (j0+fj) is injective over the whole loop nest and its image
      is exactly the live rectangle.

    *** THE HYPOTHESIS THAT IS NOT CHECKED ANYWHERE IN THE COMPILER. ***

    [c_written_exactly_once] needs [N <= ldc]. With a shorter row stride two
    distinct (row, col) pairs collapse onto one address - [row_stride_below_n_aliases]
    exhibits the pair - and the kernel would silently accumulate two tiles into
    the same element. Every caller in the repo passes ldc = N, so nothing has
    ever exercised it; the proof is what says the precondition exists.
    [tests/exact_gemm_tiling_model.rs] is what tests it, by running the kernel
    with a PADDED C and asserting the padding is untouched.

    *** What this does NOT prove. ***

    A proof about a MODEL, as the others are. No extraction, no refinement
    against [src/cpu_gemm.rs]. In particular the micro-kernel is again assumed:
    what lands in [Ct] is not modelled here, only where it is folded back to.
    And the scratch tile's own indexing ([fi*NR + fj]) is modelled as a
    faithful copy rather than proved in bounds - that is part of the
    micro-kernel obligation.

    Build:  coqc proofs/ExactGemmTiling.v      (Rocq 9.1)
*)

From Stdlib Require Import ZArith Arith Lia.
Require ExactGemmSchedule.

Module SCH := ExactGemmSchedule.

(* ------------------------------------------------------------------ *)
(** ** One axis: uniform tiles with a clamped ragged tail              *)
(* ------------------------------------------------------------------ *)

(** Tile [t] of an axis of length [ext] blocked by [T]. The width uses nat
    subtraction, which truncates at 0 - so a tile entirely past the end has
    width 0 rather than a negative one, and the results below hold for any
    tile count at or above [ntiles]. *)
Definition tw (ext T t : nat) : nat := SCH.tw ext T t.
Definition toff (T t : nat) : nat := SCH.toff T t.
Definition ntiles (ext T : nat) : nat := SCH.ntiles ext T.

(** The generated definitions in their unfolded shape, named once so a tactic
    never has to see two spellings at the same time. *)
Lemma tw_unfold : forall ext T t, tw ext T t = Nat.min (ext - t * T) T.
Proof. reflexivity. Qed.
Lemma toff_unfold : forall T t, toff T t = (t * T)%nat.
Proof. reflexivity. Qed.
Lemma ntiles_unfold : forall ext T, ntiles ext T = ((ext + T - 1) / T)%nat.
Proof. reflexivity. Qed.

(** How much of the axis the first [n] tiles account for. *)
Fixpoint covered (ext T n : nat) : nat :=
  match n with
  | O => O
  | S n' => (covered ext T n' + tw ext T n')%nat
  end.

Lemma covered_closed : forall ext T n,
  (0 < T)%nat -> covered ext T n = Nat.min (n * T) ext.
Proof.
  intros ext T n HT. induction n as [| n IH]; cbn [covered].
  - rewrite Nat.min_0_l. reflexivity.
  - rewrite IH. unfold tw, SCH.tw.
    (* Three regimes: the tile is entirely past the end, straddles it, or is
       full. `lia` closes each once `min` is resolved. *)
    destruct (Nat.le_gt_cases ext (n * T)) as [Hpast | Hlive].
    + rewrite (Nat.min_r (n * T) ext) by lia.
      replace (ext - n * T)%nat with 0%nat by lia.
      rewrite Nat.min_l by lia.
      rewrite Nat.min_r by nia. lia.
    + rewrite (Nat.min_l (n * T) ext) by lia.
      destruct (Nat.le_gt_cases (ext - n * T) T) as [Hrag | Hfull].
      * rewrite (Nat.min_l (ext - n * T) T) by lia.
        rewrite Nat.min_r by nia. lia.
      * rewrite (Nat.min_r (ext - n * T) T) by lia.
        rewrite Nat.min_l by nia. lia.
Qed.

Lemma ntiles_spans : forall ext T,
  (0 < T)%nat -> (ext <= ntiles ext T * T)%nat.
Proof.
  intros ext T HT. unfold ntiles, SCH.ntiles.
  pose proof (Nat.div_mod_eq (ext + T - 1) T) as Hdm.
  pose proof (Nat.mod_upper_bound (ext + T - 1) T ltac:(lia)) as Hmod.
  nia.
Qed.

(** The axis is covered exactly - no gap, no double count in the total. *)
Theorem tiles_cover : forall ext T,
  (0 < T)%nat -> covered ext T (ntiles ext T) = ext.
Proof.
  intros ext T HT. rewrite covered_closed by exact HT.
  pose proof (ntiles_spans ext T HT). lia.
Qed.

(** ** The bijection, which is the real obligation.

    A coverage count is satisfied by a tiling that writes one element twice and
    another never. "Exactly once" needs both directions. *)

(** Forward: a live position inside a live tile is inside the axis. This is the
    clamp doing its job, and it is the property the emitter's comment is about. *)
Theorem tile_index_in_range : forall ext T t f,
  (f < tw ext T t)%nat -> (toff T t + f < ext)%nat.
Proof.
  intros ext T t f Hf. unfold tw, SCH.tw, toff, SCH.toff in *. lia.
Qed.

(** Uniqueness of quotient and remainder, which is what BOTH injectivity
    results below are. Stated once: a tile index against its width and a row
    index against the row stride are the same arithmetic fact, and writing it
    twice is how two copies of a rule drift apart. *)
Lemma quot_rem_unique : forall T q1 r1 q2 r2,
  (0 < T)%nat -> (r1 < T)%nat -> (r2 < T)%nat ->
  (q1 * T + r1 = q2 * T + r2)%nat -> q1 = q2 /\ r1 = r2.
Proof.
  intros T q1 r1 q2 r2 HT H1 H2 Heq.
  assert (E1 : ((q1 * T + r1) / T = q1)%nat).
  { rewrite Nat.div_add_l by lia. rewrite Nat.div_small by lia. lia. }
  assert (E2 : ((q2 * T + r2) / T = q2)%nat).
  { rewrite Nat.div_add_l by lia. rewrite Nat.div_small by lia. lia. }
  assert (q1 = q2) by (rewrite <- E1, <- E2, Heq; reflexivity).
  split; [assumption | subst; lia].
Qed.

(** Injective: two distinct (tile, offset) pairs never name the same position. *)
Theorem tile_index_injective : forall ext T t1 f1 t2 f2,
  (0 < T)%nat ->
  (f1 < tw ext T t1)%nat -> (f2 < tw ext T t2)%nat ->
  toff T t1 + f1 = toff T t2 + f2 -> t1 = t2 /\ f1 = f2.
Proof.
  intros ext T t1 f1 t2 f2 HT H1 H2 Heq.
  unfold tw, SCH.tw, toff, SCH.toff in *.
  apply (quot_rem_unique T); [lia | lia | lia | exact Heq].
Qed.

(** Surjective: every position is named by some live (tile, offset). *)
Theorem tile_index_surjective : forall ext T r,
  (0 < T)%nat -> (r < ext)%nat ->
  exists t f, (t < ntiles ext T)%nat /\ (f < tw ext T t)%nat /\ toff T t + f = r.
Proof.
  intros ext T r HT Hr.
  pose proof (ntiles_spans ext T HT) as Hspan.
  pose proof (Nat.div_mod_eq r T) as Hdm.
  pose proof (Nat.mod_upper_bound r T ltac:(lia)) as Hmod.
  exists (r / T)%nat, (r mod T)%nat.
  assert (Hlt : (r / T < ntiles ext T)%nat).
  { apply Nat.Div0.div_lt_upper_bound. nia. }
  repeat split.
  - exact Hlt.
  - unfold tw, SCH.tw. nia.
  - unfold toff, SCH.toff. nia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Two axes: the write set of the whole loop nest                  *)
(* ------------------------------------------------------------------ *)

(** The address the driver stores to. *)
Definition addr (ldc i0 fi j0 fj : nat) : nat := ((i0 + fi) * ldc + (j0 + fj))%nat.

(** A linear index is injective on a rectangle only while the row stride is at
    least the row LENGTH. This is the hypothesis nothing in the compiler
    states. *)
Lemma linear_index_injective : forall ldc r1 c1 r2 c2,
  (c1 < ldc)%nat -> (c2 < ldc)%nat ->
  (r1 * ldc + c1 = r2 * ldc + c2)%nat -> r1 = r2 /\ c1 = c2.
Proof.
  intros ldc r1 c1 r2 c2 H1 H2 Heq.
  apply (quot_rem_unique ldc); [lia | lia | lia | exact Heq].
Qed.

(** THE THEOREM. Over the whole nest, distinct iterations write distinct
    addresses - so every element of the live M x N rectangle is written exactly
    once, and (with [tile_index_in_range]) never outside it. *)
Theorem c_written_exactly_once : forall M N MR NR ldc ti1 fi1 tj1 fj1 ti2 fi2 tj2 fj2,
  (0 < MR)%nat -> (0 < NR)%nat -> (N <= ldc)%nat ->
  (fi1 < tw M MR ti1)%nat -> (fj1 < tw N NR tj1)%nat ->
  (fi2 < tw M MR ti2)%nat -> (fj2 < tw N NR tj2)%nat ->
  addr ldc (toff MR ti1) fi1 (toff NR tj1) fj1
  = addr ldc (toff MR ti2) fi2 (toff NR tj2) fj2 ->
  ti1 = ti2 /\ fi1 = fi2 /\ tj1 = tj2 /\ fj1 = fj2.
Proof.
  intros M N MR NR ldc ti1 fi1 tj1 fj1 ti2 fi2 tj2 fj2
         HMR HNR Hld Hi1 Hj1 Hi2 Hj2 Heq.
  unfold addr in Heq.
  (* Both column positions are inside [0, N) and N <= ldc, so the linear index
     splits back into (row, col) uniquely. *)
  pose proof (tile_index_in_range N NR tj1 fj1 Hj1) as Hc1.
  pose proof (tile_index_in_range N NR tj2 fj2 Hj2) as Hc2.
  assert (Hd1 : (toff NR tj1 + fj1 < ldc)%nat) by lia.
  assert (Hd2 : (toff NR tj2 + fj2 < ldc)%nat) by lia.
  destruct (linear_index_injective ldc
              (toff MR ti1 + fi1) (toff NR tj1 + fj1)
              (toff MR ti2 + fi2) (toff NR tj2 + fj2)
              Hd1 Hd2 Heq) as [Hr Hc].
  (* ...and then each axis is the 1-D bijection. *)
  destruct (tile_index_injective M MR ti1 fi1 ti2 fi2 HMR Hi1 Hi2 Hr) as [Hti Hfi].
  destruct (tile_index_injective N NR tj1 fj1 tj2 fj2 HNR Hj1 Hj2 Hc) as [Htj Hfj].
  auto.
Qed.

(** Coverage of the rectangle: nothing in it is missed. *)
Theorem every_live_element_is_written : forall M N MR NR r c,
  (0 < MR)%nat -> (0 < NR)%nat -> (r < M)%nat -> (c < N)%nat ->
  exists ti fi tj fj,
    (fi < tw M MR ti)%nat /\ (fj < tw N NR tj)%nat /\
    toff MR ti + fi = r /\ toff NR tj + fj = c.
Proof.
  intros M N MR NR r c HMR HNR Hr Hc.
  destruct (tile_index_surjective M MR r HMR Hr) as [ti [fi [_ [Hfi Hri]]]].
  destruct (tile_index_surjective N NR c HNR Hc) as [tj [fj [_ [Hfj Hcj]]]].
  exists ti, fi, tj, fj. auto.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Refutation 1: the clamp is load-bearing                         *)
(* ------------------------------------------------------------------ *)

(** The emitter's comment says an unclamped tail "would run past the last row
    and column whenever M or N is not a multiple of the tile - an out-of-bounds
    WRITE, not a wrong number". Here that is a theorem rather than a comment.

    [tw_flat] is the mutation: every tile is a full T wide. At MR = 6 and
    M = 53 the last tile starts at 48 and would write rows 48..53, i.e. one
    past the end. *)
Definition tw_flat (T : nat) (_ _ : nat) : nat := T.

Theorem unclamped_tail_writes_out_of_bounds :
  exists f, (f < tw_flat 6 53 8)%nat /\ ~ (toff 6 8 + f < 53)%nat.
Proof. exists 5%nat. unfold tw_flat, toff, SCH.toff. cbn. lia. Qed.

(** ...and the clamped one does not, on the same numbers. Without this the
    theorem above is satisfied by a tiling that writes nothing. *)
Theorem the_clamped_tail_stays_in_bounds :
  forall f, (f < tw 53 6 8)%nat -> (toff 6 8 + f < 53)%nat.
Proof. intros f H. apply tile_index_in_range. exact H. Qed.

(* ------------------------------------------------------------------ *)
(** ** Refutation 2: the row stride hypothesis is not decorative       *)
(* ------------------------------------------------------------------ *)

(** With ldc < N two distinct (row, col) pairs land on one address, so two
    tiles accumulate into the same element and one element of C is never
    written. Nothing in the compiler checks this; the caller always happens to
    pass ldc = N. *)
Theorem row_stride_below_n_aliases :
  addr 4 0 0 0 4 = addr 4 0 1 0 0 /\ (0, 4) <> (1, 0).
Proof. unfold addr. cbn. split; [reflexivity | discriminate]. Qed.

Print Assumptions tiles_cover.
Print Assumptions tile_index_in_range.
Print Assumptions tile_index_injective.
Print Assumptions tile_index_surjective.
Print Assumptions c_written_exactly_once.
Print Assumptions every_live_element_is_written.
Print Assumptions unclamped_tail_writes_out_of_bounds.
Print Assumptions row_stride_below_n_aliases.
