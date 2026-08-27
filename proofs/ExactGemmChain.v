(** * Where the chain closes: one accumulator lane, end to end.

    The five schedule proofs each cover one obligation over their own model.
    `ExactGemmComposition.v` joined two of them for a SINGLE `vpdpwssd` step and
    said so explicitly - it does not chain that through the k-pair loop, nor
    into the flush. This file does both, for one accumulator lane.

    *** The statement. ***

    [the_emitted_lane_computes_the_source_dot_product]: run the emitted flush
    schedule, accumulating each chunk in INT32 and adding it into an int64
    running sum, over the panels the emitted packers produce, through the
    routing the emitted register tile performs - and lane `l` of vector `v` for
    row `i` holds exactly

    <<  sum over k < kc of  A[i][k] * B[k][16v + l]  >>

    the dot product of the SOURCE matrices. No hypothesis about panel contents,
    no hypothesis that the flush interval divides the k-pair count, and no
    hypothesis that the tile is full - a dead row or column accumulates zero
    ([a_dead_row_accumulates_nothing], [a_dead_column_accumulates_nothing]).

    The one thing it does assume is the LICENCE: `2 * Fl * m^2 <= i32::MAX` for
    an operand magnitude `m`, which is exactly what
    `VnniExact::max_operand_magnitude` computes. That assumption is discharged
    at the top level by `tests/exact_gemm_licence_obligations.rs`, which
    exhausts the finite int16 domain rather than modelling it.

    *** Why it is a chain and not a restatement. ***

    Four files meet here and each supplies something none of the others can:

    - `ExactGemmPacking.panel_decodes_its_own_write`   what a panel slot holds
    - `ExactGemmRegisterTile.the_lane_consumes_...`    which slot a lane reads
    - `ExactGemmComposition.the_packed_panels_...`     their join, one step
    - `ExactGemmMicro.flush_exact` + `...chunk_exact`  the int32 chunking

    The connective tissue is small and is the actual new content:
    [kloop_is_the_padded_product] turns a loop of routing steps into the padded
    product `ExactGemmPacking` already evaluates, and [kloop_is_sum_from_step]
    re-presents the same loop in the shape `ExactGemmMicro`'s flush theorem
    quantifies over. The two models were written independently and it was NOT
    obvious they would meet; `sum_pairs` counts k-PAIRS with both halves, while
    `sum_from` is a flat range.

    *** The licence is load-bearing at chain level, refuted concretely. ***

    [violating_the_licence_breaks_the_chain] runs the whole chain at operand
    magnitude 4096 with the shipped flush interval of 64 and gets
    -2147483648 where the answer is 2147483648;
    [at_the_licensed_magnitude_the_chain_holds] is the same chain at 4095,
    which is the largest magnitude the licence admits. That is the same
    one-unit boundary `ExactGemmMicro.the_4096_case_exceeds_by_exactly_one`
    states as arithmetic and `tests/exact_gemm_micro_model.rs` observes on a
    running kernel - here it is the END-TO-END function that changes.

    *** The tile lift. ***

    [the_tile_holds_the_source_dot_products] takes the same statement from one
    lane to the whole `MR x NR` tile: for every live `(i, j)`,

    <<  C[i][j]  =  C0[i][j] + sum over k < kc of A[i][k] * B[k][j]  >>

    and for every dead one, `C[i][j] = C0[i][j]` exactly
    ([a_dead_position_leaves_c_untouched]). Note the ACCUMULATE - `C0` is what
    was there before, because the kernel adds into C rather than assigning,
    which is what lets a caller split K across threads.

    The join needed is the inverse of `RT.col_of`.
    `RT.tile_position_surjective` says an inverse exists;
    [the_lane_map_is_a_two_sided_inverse] names it (`vec_of j = j / 16`,
    `lane_of j = j mod 16`) and proves it inverts BOTH ways, which is what
    licenses [distinct_columns_use_distinct_lanes] - no two tile positions
    share an accumulator lane, so 384 values really do live in 24 registers of
    16 lanes with nothing aliasing.

    **And the store predicate turns out to be redundant, which is a theorem
    rather than a convenience.** The emitted micro-kernel writes all `MR x NR`
    positions unconditionally - the live-rectangle clamp is the DRIVER's - so
    [tile_after] models two things at once and it is fair to ask whether the
    second does any work. [the_store_predicate_is_redundant] says it does not:
    a dead row or column accumulates zero by the packers' masks, so adding it
    to `C0` leaves `C0`. That settles the faithfulness question - the tile
    theorem describes the micro-kernel's own effect on C, clamp or no clamp -
    and it is the same redundancy `tests/exact_gemm_packing_model.rs`
    MEASURED from the other side, where removing a packer's row or column mask
    leaves every answer correct.

    **The live rectangle here is the tile's CLAMPED extent, not `M` and `N`.**
    `mrows`/`ncols` are `ExactGemmTiling.tw M MR ti` and `tw N NR tj`. That is
    a modelling fact worth stating and not worth a theorem - instantiating
    `RT.only_the_live_rectangle_is_stored` at those arguments proves nothing
    new. It is where the two models would have to be joined, and they are not.

    *** What this does NOT do, stated rather than implied. ***

    - **It does not reach the whole of C.** Three layers sit above the tile.
      `ExactGemmTiling.v`'s output partition and `ExactGemmKsplit.v`'s band
      reduction are each proved over their own model and are not joined to
      this one. The third is not proved anywhere: the **kc-panel loop**, which
      cuts K into panels of `kc` inside one thread. That is the same
      decomposition shape as `ExactGemmKsplit.bands_tile` - a clamped or
      uneven cut of `[0, K)` whose parts sum - so it is probably cheap, but
      "probably cheap" is not "done", and it is named here rather than left to
      be discovered.
    - **The ISA facts are still definitions.** `vpdpwssd`'s semantics and the
      little-endian order of an i32's halves are pinned by
      `tests/cpu_gemm_vnni_micro.rs` on the real instruction, not here.
    - As everywhere in this series, this is a proof about a MODEL. There is no
      extraction; `tests/exact_gemm_chain_model.rs` is what ties it to the Rust.

    Build:  coqc proofs/ExactGemmChain.v      (Rocq 9.1)
*)

From Stdlib Require Import ZArith Arith Lia.
Require ExactGemmSchedule.
Require ExactGemmPacking.
Require ExactGemmRegisterTile.
Require ExactGemmComposition.
Require ExactGemmMicro.
Open Scope Z_scope.

(** Module aliases, because the statements below name four files at once and
    the fully qualified forms make the theorem unreadable. *)
Module RT := ExactGemmRegisterTile.
Module PK := ExactGemmPacking.
Module MC := ExactGemmMicro.
Module SCH := ExactGemmSchedule.

(* ------------------------------------------------------------------ *)
(** ** The k-pair loop                                                  *)
(* ------------------------------------------------------------------ *)

(** Lanes are independent - each accumulator lane is its own `+=` chain - so
    one lane is modelled with one accumulator. *)
Fixpoint kloop (Ap Bp : nat -> Z) (i v l : nat) (n : nat) : Z :=
  match n with
  | O => 0
  | S p =>
      RT.vpdpwssd
        (fun _ => kloop Ap Bp i v l p)
        (RT.broadcast
           (Ap (p * (2 * RT.MR) + RT.slot_a i 0)%nat)
           (Ap (p * (2 * RT.MR) + RT.slot_a i 1)%nat))
        (RT.bvec (fun e => Bp (p * (2 * RT.NR) + e)%nat) v)
        l
  end.

Lemma kloop_is_the_padded_product :
  forall A B mrows ncols kc i v l n,
    (i < RT.MR)%nat -> (v < RT.NRV)%nat -> (l < 16)%nat ->
    kloop (PK.panel A mrows kc PK.MR)
          (PK.panel (fun j k => B k j) ncols kc PK.NR) i v l n
    = PK.sum_pairs
        (fun p h =>
           PK.packed (A i) mrows kc i (2 * p + h)%nat
           * PK.packed (fun k => B k (RT.col_of v l)) ncols kc
               (RT.col_of v l) (2 * p + h)%nat)
        n.
Proof.
  intros A B mrows ncols kc i v l n Hi Hv Hl.
  induction n as [| p IH]; [ reflexivity |].
  cbn [kloop PK.sum_pairs].
  rewrite ExactGemmComposition.the_packed_panels_route_to_the_right_source_elements
    by assumption.
  cbn beta. rewrite IH.
  replace (2 * p + 0)%nat with (2 * p)%nat by lia.
  ring.
Qed.

Lemma sum_pairs_zero : forall (g : nat -> nat -> Z) n,
  (forall p h, g p h = 0) -> PK.sum_pairs g n = 0.
Proof.
  intros g n H. induction n as [| n IH]; cbn [PK.sum_pairs]; [reflexivity |].
  rewrite IH, !H. ring.
Qed.

Theorem the_k_pair_loop_computes_the_source_dot_product :
  forall A B mrows ncols kc i v l,
    (i < RT.MR)%nat -> (v < RT.NRV)%nat -> (l < 16)%nat ->
    (i < mrows)%nat -> (RT.col_of v l < ncols)%nat ->
    kloop (PK.panel A mrows kc PK.MR)
          (PK.panel (fun j k => B k j) ncols kc PK.NR) i v l (PK.kpairs kc)
    = PK.sum_k (fun k => A i k * B k (RT.col_of v l)) kc.
Proof.
  intros A B mrows ncols kc i v l Hi Hv Hl Hrow Hcol.
  rewrite kloop_is_the_padded_product by assumption.
  apply PK.padded_product_is_the_live_dot_product; assumption.
Qed.

Corollary a_dead_row_accumulates_nothing :
  forall A B mrows ncols kc i v l n,
    (i < RT.MR)%nat -> (v < RT.NRV)%nat -> (l < 16)%nat -> (mrows <= i)%nat ->
    kloop (PK.panel A mrows kc PK.MR)
          (PK.panel (fun j k => B k j) ncols kc PK.NR) i v l n = 0.
Proof.
  intros. rewrite kloop_is_the_padded_product by assumption.
  apply sum_pairs_zero. intros p h. unfold PK.packed.
  rewrite (proj2 (Nat.ltb_ge i mrows)) by lia. cbn [andb]. ring.
Qed.

Corollary a_dead_column_accumulates_nothing :
  forall A B mrows ncols kc i v l n,
    (i < RT.MR)%nat -> (v < RT.NRV)%nat -> (l < 16)%nat ->
    (ncols <= RT.col_of v l)%nat ->
    kloop (PK.panel A mrows kc PK.MR)
          (PK.panel (fun j k => B k j) ncols kc PK.NR) i v l n = 0.
Proof.
  intros. rewrite kloop_is_the_padded_product by assumption.
  apply sum_pairs_zero. intros p h. unfold PK.packed.
  rewrite (proj2 (Nat.ltb_ge (RT.col_of v l) ncols)) by lia.
  cbn [andb]. ring.
Qed.

Definition Aex (i k : nat) : Z := Z.of_nat (k + 1).
Definition Bex (k j : nat) : Z := Z.of_nat (k + 4).

Theorem the_chain_is_not_vacuous :
  kloop (PK.panel Aex 1 3 PK.MR) (PK.panel (fun j k => Bex k j) 1 3 PK.NR)
        0 0 0 (PK.kpairs 3) = 32.
Proof. vm_compute. reflexivity. Qed.


(* One k-pair's contribution: the same vpdpwssd step against a zero
   accumulator. This is the `f` the flush theorem quantifies over. *)
Definition step (Ap Bp : nat -> Z) (i v l p : nat) : Z :=
  RT.vpdpwssd (fun _ => 0)
    (RT.broadcast
       (Ap (p * (2 * RT.MR) + RT.slot_a i 0)%nat)
       (Ap (p * (2 * RT.MR) + RT.slot_a i 1)%nat))
    (RT.bvec (fun e => Bp (p * (2 * RT.NR) + e)%nat) v)
    l.

Lemma kloop_is_sum_from_step : forall Ap Bp i v l n,
  kloop Ap Bp i v l n = MC.sum_from (step Ap Bp i v l) 0 n.
Proof.
  intros Ap Bp i v l n. induction n as [| n IH]; [reflexivity |].
  cbn [kloop MC.sum_from]. rewrite IH, Nat.add_0_l.
  unfold step, RT.vpdpwssd. ring.
Qed.

Theorem the_flush_schedule_computes_the_source_dot_product :
  forall A B mrows ncols kc Fl i v l,
    (0 < Fl)%nat ->
    (i < RT.MR)%nat -> (v < RT.NRV)%nat -> (l < 16)%nat ->
    (i < mrows)%nat -> (RT.col_of v l < ncols)%nat ->
    MC.chunk_acc
      (step (PK.panel A mrows kc PK.MR)
            (PK.panel (fun j k => B k j) ncols kc PK.NR) i v l)
      Fl (PK.kpairs kc) (MC.nchunks Fl (PK.kpairs kc))
    = PK.sum_k (fun k => A i k * B k (RT.col_of v l)) kc.
Proof.
  intros A B mrows ncols kc Fl i v l HFl Hi Hv Hl Hrow Hcol.
  rewrite MC.flush_exact by exact HFl.
  rewrite <- kloop_is_sum_from_step.
  apply the_k_pair_loop_computes_the_source_dot_product; assumption.
Qed.

Theorem the_whole_chain_is_not_vacuous :
  MC.chunk_acc (step (PK.panel Aex 1 3 PK.MR)
                     (PK.panel (fun j k => Bex k j) 1 3 PK.NR) 0 0 0)
    1 (PK.kpairs 3) (MC.nchunks 1 (PK.kpairs 3)) = 32.
Proof. vm_compute. reflexivity. Qed.

(* ---------------- the int32 layer ---------------- *)

(** The kernel's real accumulation: each chunk summed in INT32, then added into
    the int64 running total. [MC.chunk_acc] uses [MC.sum_from] for a chunk,
    i.e. Z; this uses [MC.wsum], i.e. int32 with wraparound. *)
Fixpoint chunk_acc_i32 (f : nat -> Z) (Fl n t : nat) : Z :=
  match t with
  | O => 0
  | S t' => chunk_acc_i32 f Fl n t' + MC.wsum f (MC.coff Fl t') (MC.cw Fl n t')
  end.

Lemma chunk_width_fits_the_interval : forall Fl n t,
  (0 < Fl)%nat -> (MC.cw Fl n t <= Fl)%nat.
Proof.
  intros Fl n t HFl. unfold MC.cw, MC.coff, SCH.cw, SCH.coff.
  pose proof (Nat.le_min_l (S t * Fl)%nat n) as H1. lia.
Qed.

Lemma panel_bound : forall v extent kc width m,
  0 <= m -> (forall idx k, Z.abs (v idx k) <= m) ->
  forall s, Z.abs (PK.panel v extent kc width s) <= m.
Proof.
  intros v extent kc width m Hm Hv s. unfold PK.panel, PK.packed.
  destruct (andb _ _); [ apply Hv | cbn; lia ].
Qed.

Lemma prod_pair_bound : forall x y z w m,
  0 <= m -> Z.abs x <= m -> Z.abs y <= m -> Z.abs z <= m -> Z.abs w <= m ->
  Z.abs (x * y + z * w) <= 2 * m * m.
Proof.
  intros x y z w m Hm Hx Hy Hz Hw.
  eapply Z.le_trans; [ apply Z.abs_triangle |].
  rewrite !Z.abs_mul.
  pose proof (Z.abs_nonneg x). pose proof (Z.abs_nonneg y).
  pose proof (Z.abs_nonneg z). pose proof (Z.abs_nonneg w). nia.
Qed.

Lemma step_bound : forall Ap Bp i v l m,
  0 <= m ->
  (forall s, Z.abs (Ap s) <= m) -> (forall s, Z.abs (Bp s) <= m) ->
  forall p, Z.abs (step Ap Bp i v l p) <= 2 * m * m.
Proof.
  intros Ap Bp i v l m Hm HA HB p.
  unfold step, RT.vpdpwssd, RT.broadcast, RT.bvec.
  rewrite Z.add_0_l.
  apply prod_pair_bound;
    [ exact Hm
    | destruct (Nat.even (2 * l)); apply HA
    | apply HB
    | destruct (Nat.even (2 * l + 1)); apply HA
    | apply HB ].
Qed.

Lemma the_licence_makes_every_chunk_exact : forall f Fl n m t,
  (0 < Fl)%nat -> 0 <= m ->
  2 * Z.of_nat Fl * m * m <= MC.I32MAX ->
  (forall p, Z.abs (f p) <= 2 * m * m) ->
  chunk_acc_i32 f Fl n t = MC.chunk_acc f Fl n t.
Proof.
  intros f Fl n m t HFl Hm Hlic Hf.
  induction t as [| t IH]; [reflexivity |].
  cbn [chunk_acc_i32 MC.chunk_acc]. rewrite IH.
  rewrite (MC.the_licence_makes_the_chunk_exact f (MC.coff Fl t) (MC.cw Fl n t)
             Fl m Hm (chunk_width_fits_the_interval Fl n t HFl) Hlic Hf).
  reflexivity.
Qed.

Theorem the_emitted_lane_computes_the_source_dot_product :
  forall A B mrows ncols kc Fl m i v l,
    (0 < Fl)%nat -> 0 <= m ->
    2 * Z.of_nat Fl * m * m <= MC.I32MAX ->
    (forall idx k, Z.abs (A idx k) <= m) ->
    (forall k idx, Z.abs (B k idx) <= m) ->
    (i < RT.MR)%nat -> (v < RT.NRV)%nat -> (l < 16)%nat ->
    (i < mrows)%nat -> (RT.col_of v l < ncols)%nat ->
    chunk_acc_i32
      (step (PK.panel A mrows kc PK.MR)
            (PK.panel (fun j k => B k j) ncols kc PK.NR) i v l)
      Fl (PK.kpairs kc) (MC.nchunks Fl (PK.kpairs kc))
    = PK.sum_k (fun k => A i k * B k (RT.col_of v l)) kc.
Proof.
  intros A B mrows ncols kc Fl m i v l HFl Hm Hlic HA HB Hi Hv Hl Hrow Hcol.
  assert (Hstep : forall p,
    Z.abs (step (PK.panel A mrows kc PK.MR)
                (PK.panel (fun j k => B k j) ncols kc PK.NR) i v l p)
    <= 2 * m * m).
  { apply step_bound.
    - exact Hm.
    - apply panel_bound; [ exact Hm | exact HA ].
    - apply panel_bound; [ exact Hm | intros idx k; apply HB ]. }
  rewrite (the_licence_makes_every_chunk_exact _ Fl _ m _ HFl Hm Hlic Hstep).
  apply the_flush_schedule_computes_the_source_dot_product; assumption.
Qed.


(* ------------------------------------------------------------------ *)
(** ** The licence is load-bearing for the WHOLE chain                 *)
(* ------------------------------------------------------------------ *)

(** Operand magnitude 4096 at the shipped flush interval of 64:
    `2 * 64 * 4096^2 = 2147483648`, over `i32::MAX` by exactly one. *)
Definition A4096 (i k : nat) : Z := 4096.
Definition B4096 (k j : nat) : Z := 4096.

Theorem violating_the_licence_breaks_the_chain :
  chunk_acc_i32
    (step (PK.panel A4096 1 128 PK.MR)
          (PK.panel (fun j k => B4096 k j) 1 128 PK.NR) 0 0 0)
    64 (PK.kpairs 128) (MC.nchunks 64 (PK.kpairs 128))
  <> PK.sum_k (fun k => A4096 0 k * B4096 k 0) 128.
Proof. vm_compute. discriminate. Qed.

(** The control, and the reason the refutation is about the LICENCE rather
    than about the numbers: one less, and the same chain is exact.
    `2 * 64 * 4095^2 = 2146435200`, with 1048447 to spare. *)
Definition A4095 (i k : nat) : Z := 4095.
Definition B4095 (k j : nat) : Z := 4095.

Theorem at_the_licensed_magnitude_the_chain_holds :
  chunk_acc_i32
    (step (PK.panel A4095 1 128 PK.MR)
          (PK.panel (fun j k => B4095 k j) 1 128 PK.NR) 0 0 0)
    64 (PK.kpairs 128) (MC.nchunks 64 (PK.kpairs 128))
  = PK.sum_k (fun k => A4095 0 k * B4095 k 0) 128.
Proof. vm_compute. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** The tile lift: from one lane to the MR x NR rectangle           *)
(* ------------------------------------------------------------------ *)

(* Which accumulator vector and lane hold tile column j - the inverse of
   RT.col_of, which RT.tile_position_surjective says exists. *)
Definition vec_of  (j : nat) : nat := (j / 16)%nat.
Definition lane_of (j : nat) : nat := (j mod 16)%nat.

Lemma the_lane_decomposition_inverts_the_column : forall j,
  (j < RT.NR)%nat ->
  (vec_of j < RT.NRV)%nat /\ (lane_of j < 16)%nat
  /\ RT.col_of (vec_of j) (lane_of j) = j.
Proof.
  intros j Hj.
  unfold vec_of, lane_of, RT.col_of, SCH.col_of,
         RT.NR, SCH.NR, RT.NRV, SCH.NRV in *.
  pose proof (Nat.div_mod_eq j 16) as Hdm.
  pose proof (Nat.mod_upper_bound j 16 ltac:(lia)) as Hub.
  assert ((j / 16 < 4)%nat) by (apply Nat.Div0.div_lt_upper_bound; lia).
  repeat split; lia.
Qed.

(* The kernel's effect on one tile position: the lane's int32-flushed
   accumulator added into C, through the store's live-rectangle predicate. *)
Definition tile_after (C0 : nat -> nat -> Z) (Ap Bp : nat -> Z)
                      (Fl mrows ncols kp i j : nat) : Z :=
  if RT.stored mrows ncols i j
  then C0 i j
       + chunk_acc_i32 (step Ap Bp i (vec_of j) (lane_of j))
           Fl kp (MC.nchunks Fl kp)
  else C0 i j.

Theorem the_tile_holds_the_source_dot_products :
  forall A B C0 mrows ncols kc Fl m i j,
    (0 < Fl)%nat -> 0 <= m ->
    2 * Z.of_nat Fl * m * m <= MC.I32MAX ->
    (forall idx k, Z.abs (A idx k) <= m) ->
    (forall k idx, Z.abs (B k idx) <= m) ->
    (i < RT.MR)%nat -> (j < RT.NR)%nat ->
    tile_after C0
      (PK.panel A mrows kc PK.MR)
      (PK.panel (fun c k => B k c) ncols kc PK.NR)
      Fl mrows ncols (PK.kpairs kc) i j
    = if RT.stored mrows ncols i j
      then C0 i j + PK.sum_k (fun k => A i k * B k j) kc
      else C0 i j.
Proof.
  intros A B C0 mrows ncols kc Fl m i j HFl Hm Hlic HA HB Hi Hj.
  destruct (the_lane_decomposition_inverts_the_column j Hj) as [Hv [Hl Hcol]].
  unfold tile_after.
  destruct (RT.stored mrows ncols i j) eqn:Hs; [| reflexivity ].
  apply (proj1 (RT.only_the_live_rectangle_is_stored mrows ncols i j)) in Hs.
  destruct Hs as [Hrow Hncol].
  rewrite (the_emitted_lane_computes_the_source_dot_product
             A B mrows ncols kc Fl m i (vec_of j) (lane_of j));
    try assumption; rewrite Hcol; try assumption.
  reflexivity.
Qed.

(** The other direction: `vec_of`/`lane_of` is a two-sided inverse of
    `RT.col_of`. `RT.tile_position_surjective` says an inverse exists; this
    names it and proves it inverts BOTH ways, which is what licenses "no two
    tile positions share an accumulator lane". *)
Theorem the_lane_map_is_a_two_sided_inverse : forall v l,
  (v < RT.NRV)%nat -> (l < 16)%nat ->
  vec_of (RT.col_of v l) = v /\ lane_of (RT.col_of v l) = l.
Proof.
  intros v l Hv Hl. unfold vec_of, lane_of, RT.col_of, SCH.col_of.
  split.
  - replace (16 * v + l)%nat with (l + v * 16)%nat by lia.
    rewrite Nat.div_add by lia. rewrite Nat.div_small by lia. lia.
  - replace (16 * v + l)%nat with (l + v * 16)%nat by lia.
    rewrite Nat.Div0.mod_add. apply Nat.mod_small; lia.
Qed.

Corollary distinct_columns_use_distinct_lanes : forall j1 j2,
  (j1 < RT.NR)%nat -> (j2 < RT.NR)%nat ->
  vec_of j1 = vec_of j2 -> lane_of j1 = lane_of j2 -> j1 = j2.
Proof.
  intros j1 j2 H1 H2 Hv Hl.
  destruct (the_lane_decomposition_inverts_the_column j1 H1) as [_ [_ E1]].
  destruct (the_lane_decomposition_inverts_the_column j2 H2) as [_ [_ E2]].
  rewrite <- E1, <- E2, Hv, Hl. reflexivity.
Qed.

(** A dead position is left exactly as it was - the tile runs at full width
    and the store discards the rest. *)
Corollary a_dead_position_leaves_c_untouched :
  forall C0 Ap Bp Fl mrows ncols kp i j,
    (mrows <= i)%nat \/ (ncols <= j)%nat ->
    tile_after C0 Ap Bp Fl mrows ncols kp i j = C0 i j.
Proof.
  intros C0 Ap Bp Fl mrows ncols kp i j Hdead.
  unfold tile_after.
  destruct (RT.stored mrows ncols i j) eqn:Hs; [| reflexivity ].
  apply (proj1 (RT.only_the_live_rectangle_is_stored mrows ncols i j)) in Hs.
  lia.
Qed.

(** Concrete, because every equality above is satisfied by a model computing
    nothing. A 2 x 3 live rectangle inside the 6 x 64 tile, kc = 3 (odd, so the
    phantom k-half is live), C pre-loaded with 100 so the ACCUMULATE is visible
    rather than an assignment.

    Row 1 of A is 2, 3, 4; column 2 of B is 3, 5, 7; so 6 + 15 + 28 = 49. *)
Definition Atc (i k : nat) : Z := Z.of_nat (i + k + 1).
Definition Btc (k j : nat) : Z := Z.of_nat (2 * k + j + 1).
Definition C100 (i j : nat) : Z := 100.

Theorem the_tile_lift_is_not_vacuous :
  tile_after C100 (PK.panel Atc 2 3 PK.MR)
                  (PK.panel (fun c k => Btc k c) 3 3 PK.NR)
             64 2 3 (PK.kpairs 3) 1 2 = 149
  /\ tile_after C100 (PK.panel Atc 2 3 PK.MR)
                     (PK.panel (fun c k => Btc k c) 3 3 PK.NR)
                64 2 3 (PK.kpairs 3) 1 5 = 100.
Proof. split; vm_compute; reflexivity. Qed.

Print Assumptions the_lane_decomposition_inverts_the_column.
Print Assumptions the_lane_map_is_a_two_sided_inverse.
Print Assumptions distinct_columns_use_distinct_lanes.
Print Assumptions the_tile_holds_the_source_dot_products.
Print Assumptions a_dead_position_leaves_c_untouched.
Print Assumptions the_tile_lift_is_not_vacuous.

(** **Is the store predicate needed at all?** The emitted micro-kernel writes
    every one of the `MR x NR` positions unconditionally; the live-rectangle
    clamp lives in the DRIVER. So [tile_after] models two things at once, and
    it is fair to ask whether the second one does any work.

    It does not, and that is a THEOREM rather than a modelling convenience: a
    dead row or column accumulates zero (the packers' masks), so adding it to
    `C0` leaves `C0`. `tests/exact_gemm_packing_model.rs` MEASURED this
    redundancy - removing a packer's row or column mask leaves every answer
    correct because the clamp discards it anyway - and this is the same fact
    proved from the other side. It also settles the faithfulness question:
    [the_tile_holds_the_source_dot_products] describes the micro-kernel's own
    effect on C, clamp or no clamp. *)
Theorem the_store_predicate_is_redundant :
  forall A B C0 mrows ncols kc Fl m i j,
    (0 < Fl)%nat -> 0 <= m ->
    2 * Z.of_nat Fl * m * m <= MC.I32MAX ->
    (forall idx k, Z.abs (A idx k) <= m) ->
    (forall k idx, Z.abs (B k idx) <= m) ->
    (i < RT.MR)%nat -> (j < RT.NR)%nat ->
    tile_after C0
      (PK.panel A mrows kc PK.MR)
      (PK.panel (fun c k => B k c) ncols kc PK.NR)
      Fl mrows ncols (PK.kpairs kc) i j
    = C0 i j
      + chunk_acc_i32
          (step (PK.panel A mrows kc PK.MR)
                (PK.panel (fun c k => B k c) ncols kc PK.NR)
                i (vec_of j) (lane_of j))
          Fl (PK.kpairs kc) (MC.nchunks Fl (PK.kpairs kc)).
Proof.
  intros A B C0 mrows ncols kc Fl m i j HFl Hm Hlic HA HB Hi Hj.
  destruct (the_lane_decomposition_inverts_the_column j Hj) as [Hv [Hl Hcol]].
  unfold tile_after.
  destruct (RT.stored mrows ncols i j) eqn:Hs; [ reflexivity |].
  (* Dead: the accumulator is zero, so the predicate changes nothing. *)
  assert (Hstep : forall p,
    Z.abs (step (PK.panel A mrows kc PK.MR)
                (PK.panel (fun c k => B k c) ncols kc PK.NR)
                i (vec_of j) (lane_of j) p) <= 2 * m * m).
  { apply step_bound.
    - exact Hm.
    - apply panel_bound; [ exact Hm | exact HA ].
    - apply panel_bound; [ exact Hm | intros idx k; apply HB ]. }
  rewrite (the_licence_makes_every_chunk_exact _ Fl _ m _ HFl Hm Hlic Hstep).
  rewrite MC.flush_exact by exact HFl.
  rewrite <- kloop_is_sum_from_step.
  assert (Hdead : (mrows <= i)%nat \/ (ncols <= j)%nat).
  { unfold RT.stored in Hs.
    apply Bool.andb_false_iff in Hs.
    destruct Hs as [H | H]; [ left | right ];
      apply Nat.ltb_ge in H; exact H. }
  destruct Hdead as [Hdr | Hdc].
  - rewrite (a_dead_row_accumulates_nothing A B mrows ncols kc i (vec_of j)
               (lane_of j) _ Hi Hv Hl Hdr). ring.
  - rewrite (a_dead_column_accumulates_nothing A B mrows ncols kc i (vec_of j)
               (lane_of j) _ Hi Hv Hl ltac:(rewrite Hcol; exact Hdc)). ring.
Qed.

Print Assumptions kloop_is_the_padded_product.
Print Assumptions the_k_pair_loop_computes_the_source_dot_product.
Print Assumptions a_dead_row_accumulates_nothing.
Print Assumptions a_dead_column_accumulates_nothing.
Print Assumptions kloop_is_sum_from_step.
Print Assumptions the_flush_schedule_computes_the_source_dot_product.
Print Assumptions the_emitted_lane_computes_the_source_dot_product.
Print Assumptions the_chain_is_not_vacuous.
Print Assumptions the_whole_chain_is_not_vacuous.
Print Assumptions violating_the_licence_breaks_the_chain.
Print Assumptions at_the_licensed_magnitude_the_chain_holds.
Print Assumptions the_store_predicate_is_redundant.
