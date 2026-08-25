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

    *** What this does NOT do, stated rather than implied. ***

    - **It is one LANE, not the tile.** Lanes are independent, but the
      tile-level statement needs the store, which is
      `ExactGemmRegisterTile.only_the_live_rectangle_is_stored` over a
      different model of C.
    - **It does not reach C.** `ExactGemmTiling.v`'s output partition and
      `ExactGemmKsplit.v`'s band reduction sit ABOVE this and are proved over
      their own models. Joining those needs one shared model of the kernel,
      which remains Phase 2's subject.
    - **The ISA facts are still definitions.** `vpdpwssd`'s semantics and the
      little-endian order of an i32's halves are pinned by
      `tests/cpu_gemm_vnni_micro.rs` on the real instruction, not here.
    - As everywhere in this series, this is a proof about a MODEL. There is no
      extraction; `tests/exact_gemm_chain_model.rs` is what ties it to the Rust.

    Build:  coqc proofs/ExactGemmChain.v      (Rocq 9.1)
*)

From Stdlib Require Import ZArith Arith Lia.
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
  intros Fl n t HFl. unfold MC.cw, MC.coff.
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
