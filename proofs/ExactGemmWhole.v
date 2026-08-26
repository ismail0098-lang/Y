(** * The whole of C: all six exact-GEMM proofs, chained.

    Phase 1 accumulated six files, each proving one obligation over its own
    model. `ExactGemmChain.v` joined four of them up to a single `MR x NR`
    tile. This file adds the last two - the output partition and the K-split -
    and states the result for every element of C.

    *** THE THEOREM. ***

    [the_threaded_gemm_holds_the_source_dot_products]: sum the partials of
    `nthr` threads, each handed a K band, each running the emitted driver -
    packing, the register tile's routing, the k-pair loop, the int32 flush,
    the scratch tile and the fold-back - and for every `(r, c)` inside `M x N`
    the result is exactly

    <<  sum over k < K of  A[r][k] * B[k][c]  >>

    with no hypothesis that `MR` divides `M`, that `NR` divides `N`, that
    `nthr` divides `K`, or that `K` is even. The only assumption is the
    LICENCE - `2 * Fl * m^2 <= i32::MAX` for operand magnitude `m` - which is
    what `VnniExact::max_operand_magnitude` computes and what
    `tests/exact_gemm_licence_obligations.rs` discharges by exhausting the
    finite int16 domain.

    *** What each file contributes. ***

    - `ExactGemmPacking.v`      what a packed panel slot holds
    - `ExactGemmRegisterTile.v` which accumulator lane reads it
    - `ExactGemmComposition.v`  their join, for one `vpdpwssd` step
    - `ExactGemmMicro.v`        the int32 flush chunking
    - `ExactGemmChain.v`        the k-pair loop, and the lift to a tile
    - `ExactGemmTiling.v`       the output partition        <- joined here
    - `ExactGemmKsplit.v`       the K-band reduction        <- joined here

    *** A correction to what was previously recorded as missing. ***

    An earlier note named a **kc-panel loop** - K cut into panels of `kc`
    inside one thread - as the one layer proved nowhere. **There is no such
    loop.** `emit_vnni_gemm_driver` passes the FULL `K` to both packers and
    `kpairs = (K+1)/2` to the micro-kernel; the only cut of the K axis is the
    K-split across threads. So `kc` in the sibling files is K, and the gap that
    actually remained was the fold-back into C, which is what
    [gemm_position] models here. Read the emitter before recording a gap.

    *** What is STILL not proved, stated rather than implied. ***

    - **`vpdpwssd`'s semantics and the little-endian order of an i32's two
      halves are DEFINITIONS.** No proof over [Z] can supply them;
      `tests/cpu_gemm_vnni_micro.rs` pins them on the real instruction.
    - **The loop STRUCTURE is modelled, not extracted.** [gemm_position] says
      what the position `(r, c)` receives, on the reading that the driver
      visits tile `(r / MR, c / NR)` at offset `(r mod MR, c mod NR)`.
      [the_position_decomposition_is_the_tilings] ties that to
      `ExactGemmTiling.addr`, and `ExactGemmTiling.c_written_exactly_once`
      proves that enumeration hits every element once - but the emitted loop
      nest is not extracted, so the tie is between two models rather than to
      the LLVM. That gap is named in every file of this series and is what
      Phase 2 exists to close.
    - The threaded wrapper's `pthread` mechanics, the panel buffers' sizes and
      the scratch tile's allocation are not modelled at all.

    Build:  coqc proofs/ExactGemmWhole.v      (Rocq 9.1)
*)

From Stdlib Require Import ZArith Arith Lia.
Require ExactGemmChain.
Require ExactGemmPacking.
Require ExactGemmRegisterTile.
Require ExactGemmTiling.
Require ExactGemmKsplit.
Require ExactGemmMicro.
Open Scope Z_scope.

Module RT := ExactGemmRegisterTile.
Module PK := ExactGemmPacking.
Module TL := ExactGemmTiling.
Module KS := ExactGemmKsplit.
Module CH := ExactGemmChain.

(** Every position of C is live in its OWN tile: `r` lands in tile `r / T` at
    offset `r mod T`, and the clamped width of that tile is wide enough. *)
Lemma the_position_is_live_in_its_own_tile : forall ext T r,
  (0 < T)%nat -> (r < ext)%nat ->
  (r mod T < TL.tw ext T (r / T))%nat.
Proof.
  intros ext T r HT Hr. unfold TL.tw.
  pose proof (Nat.div_mod_eq r T) as Hdm.
  pose proof (Nat.mod_upper_bound r T ltac:(lia)) as Hub.
  apply Nat.min_glb_lt; lia.
Qed.

Lemma the_tile_and_offset_rebuild_the_position : forall T r,
  (0 < T)%nat -> ((r / T) * T + r mod T)%nat = r.
Proof.
  intros T r HT. pose proof (Nat.div_mod_eq r T). lia.
Qed.

(** The driver's effect on one element of C, as the composition the emitted
    code performs: the micro-kernel accumulates into a ZEROED scratch tile, and
    the fold-back adds the live part into C. *)
Definition gemm_position (A B : nat -> nat -> Z) (M N K Fl r c : nat) : Z :=
  CH.tile_after (fun _ _ => 0)
    (PK.panel (fun i k => A ((r / RT.MR) * RT.MR + i)%nat k)
              (TL.tw M RT.MR (r / RT.MR)) K PK.MR)
    (PK.panel (fun cc k => B k ((c / RT.NR) * RT.NR + cc)%nat)
              (TL.tw N RT.NR (c / RT.NR)) K PK.NR)
    Fl (TL.tw M RT.MR (r / RT.MR)) (TL.tw N RT.NR (c / RT.NR))
    (PK.kpairs K) (r mod RT.MR) (c mod RT.NR).

Theorem the_whole_output_holds_the_source_dot_products :
  forall A B M N K Fl m r c,
    (0 < Fl)%nat -> 0 <= m ->
    2 * Z.of_nat Fl * m * m <= ExactGemmMicro.I32MAX ->
    (forall idx k, Z.abs (A idx k) <= m) ->
    (forall k idx, Z.abs (B k idx) <= m) ->
    (r < M)%nat -> (c < N)%nat ->
    gemm_position A B M N K Fl r c = PK.sum_k (fun k => A r k * B k c) K.
Proof.
  intros A B M N K Fl m r c HFl Hm Hlic HA HB Hr Hc.
  assert (HMR : (0 < RT.MR)%nat) by (unfold RT.MR; lia).
  assert (HNR : (0 < RT.NR)%nat) by (unfold RT.NR; lia).
  assert (Hi : (r mod RT.MR < RT.MR)%nat) by (apply Nat.mod_upper_bound; lia).
  assert (Hj : (c mod RT.NR < RT.NR)%nat) by (apply Nat.mod_upper_bound; lia).
  unfold gemm_position.
  rewrite (CH.the_tile_holds_the_source_dot_products _ _ _ _ _ _ _ m);
    try assumption.
  - (* the position is live, so the store predicate is true *)
    replace (RT.stored (TL.tw M RT.MR (r / RT.MR)) (TL.tw N RT.NR (c / RT.NR))
                       (r mod RT.MR) (c mod RT.NR)) with true.
    + rewrite Z.add_0_l.
      apply PK.sum_k_ext. intros k _.
      rewrite (the_tile_and_offset_rebuild_the_position RT.MR r HMR).
      rewrite (the_tile_and_offset_rebuild_the_position RT.NR c HNR).
      reflexivity.
    + symmetry. apply (proj2 (RT.only_the_live_rectangle_is_stored _ _ _ _)).
      split; apply the_position_is_live_in_its_own_tile; assumption.
  - intros idx k. apply HA.
  - intros k idx. apply HB.
Qed.

(* ---------------- the K-split across threads ---------------- *)


Lemma sum_k_shifted_is_acc_range : forall f bo len,
  PK.sum_k (fun k => f (bo + k)%nat) len = KS.acc_range Z.add f bo len.
Proof.
  intros f bo len. induction len as [| n IH]; [reflexivity |].
  cbn [PK.sum_k KS.acc_range]. rewrite IH. reflexivity.
Qed.

Lemma acc_range_from_zero_is_sum_k : forall f n,
  KS.acc_range Z.add f 0 n = PK.sum_k f n.
Proof.
  intros f n. rewrite <- sum_k_shifted_is_acc_range.
  apply PK.sum_k_ext. intros k _. rewrite Nat.add_0_l. reflexivity.
Qed.

(** Thread `t'` is handed the K band `[boff, boff + blen)` - i.e. the same A and
    B with their k axis shifted - and its partials are summed. *)
Fixpoint thread_sum (A B : nat -> nat -> Z) (M N K Fl nthr r c t : nat) : Z :=
  match t with
  | O => 0
  | S t' =>
      thread_sum A B M N K Fl nthr r c t'
      + gemm_position (fun i k => A i (KS.boff K nthr t' + k)%nat)
                      (fun k j => B (KS.boff K nthr t' + k)%nat j)
                      M N (KS.blen K nthr t') Fl r c
  end.

Theorem the_threaded_gemm_holds_the_source_dot_products :
  forall A B M N K Fl m nthr r c,
    (0 < Fl)%nat -> 0 <= m ->
    2 * Z.of_nat Fl * m * m <= ExactGemmMicro.I32MAX ->
    (forall idx k, Z.abs (A idx k) <= m) ->
    (forall k idx, Z.abs (B k idx) <= m) ->
    (0 < nthr)%nat -> (r < M)%nat -> (c < N)%nat ->
    thread_sum A B M N K Fl nthr r c nthr
    = PK.sum_k (fun k => A r k * B k c) K.
Proof.
  intros A B M N K Fl m nthr r c HFl Hm Hlic HA HB Hn Hr Hc.
  assert (Hband : forall t,
    thread_sum A B M N K Fl nthr r c t
    = KS.acc_bands Z.add (fun k => A r k * B k c) K nthr t).
  { induction t as [| t IH]; [reflexivity |].
    cbn [thread_sum KS.acc_bands]. rewrite IH.
    rewrite (the_whole_output_holds_the_source_dot_products _ _ _ _ _ _ m);
      try assumption.
    - (* Higher-order unification cannot guess `f` here, so the instance is
         given explicitly: `f k = A r k * B k c`, shifted by the band offset. *)
      assert (E : PK.sum_k
                    (fun k => A r (KS.boff K nthr t + k)%nat
                              * B (KS.boff K nthr t + k)%nat c)
                    (KS.blen K nthr t)
                  = KS.acc_range Z.add (fun k => A r k * B k c)
                      (KS.boff K nthr t) (KS.blen K nthr t))
        by apply (sum_k_shifted_is_acc_range (fun k => A r k * B k c)).
      rewrite E. reflexivity.
    - intros idx k. apply HA.
    - intros k idx. apply HB. }
  rewrite Hband, KS.ksplit_exact by exact Hn.
  apply acc_range_from_zero_is_sum_k.
Qed.


(* ------------------------------------------------------------------ *)
(** ** The tie to the output partition                                 *)
(* ------------------------------------------------------------------ *)

(** [gemm_position] indexes C by `(r, c)`; `ExactGemmTiling.v` indexes it by
    tile plus offset. They address the same element, which is what lets
    `ExactGemmTiling.c_written_exactly_once` - every element written by exactly
    one `(ti, fi, tj, fj)` - apply to this decomposition. *)
Theorem the_position_decomposition_is_the_tilings : forall ldc r c,
  TL.addr ldc ((r / RT.MR) * RT.MR)%nat (r mod RT.MR)
              ((c / RT.NR) * RT.NR)%nat (c mod RT.NR)
  = (r * ldc + c)%nat.
Proof.
  intros ldc r c. unfold TL.addr.
  assert (HMR : (0 < RT.MR)%nat) by (unfold RT.MR; lia).
  assert (HNR : (0 < RT.NR)%nat) by (unfold RT.NR; lia).
  rewrite (the_tile_and_offset_rebuild_the_position RT.MR r HMR).
  rewrite (the_tile_and_offset_rebuild_the_position RT.NR c HNR).
  reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Concrete, because every equality above is satisfied by zero     *)
(* ------------------------------------------------------------------ *)

(** `M = 2`, `N = 3`, `K = 3` - so M, N and K are all RAGGED against the
    `6 x 64` tile and K is odd, giving a phantom k-half. Row 1 of A is 2, 3, 4;
    column 2 of B is 3, 5, 7; the dot product is 6 + 15 + 28 = 49.

    Checked at one thread and at two - the K = 3 split into bands of 2 and 1 is
    uneven, which is the case `ExactGemmKsplit.bands_tile` exists for. *)
Definition Aex (i k : nat) : Z := Z.of_nat (i + k + 1).
Definition Bex (k j : nat) : Z := Z.of_nat (2 * k + j + 1).

Theorem the_whole_chain_is_not_vacuous :
  thread_sum Aex Bex 2 3 3 64 1 1 2 1 = 49
  /\ thread_sum Aex Bex 2 3 3 64 2 1 2 2 = 49
  /\ PK.sum_k (fun k => Aex 1 k * Bex k 2) 3 = 49.
Proof. repeat split; vm_compute; reflexivity. Qed.

(** ...and it is a real reduction rather than one band doing all the work: at
    two threads the first band carries 2 of the 3 k-values and the second
    carries 1, and neither alone is the answer. *)
Theorem both_bands_contribute :
  thread_sum Aex Bex 2 3 3 64 2 1 2 1 = 21
  /\ thread_sum Aex Bex 2 3 3 64 2 1 2 2 = 49.
Proof. split; vm_compute; reflexivity. Qed.

Print Assumptions the_position_is_live_in_its_own_tile.
Print Assumptions the_whole_output_holds_the_source_dot_products.
Print Assumptions the_threaded_gemm_holds_the_source_dot_products.
Print Assumptions the_position_decomposition_is_the_tilings.
Print Assumptions the_whole_chain_is_not_vacuous.
Print Assumptions both_bands_contribute.
