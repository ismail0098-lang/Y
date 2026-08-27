(** * The exact VNNI micro-kernel's REGISTER TILE.

    Phase 1's schedule is proved elsewhere: `ExactGemmKsplit.v` (the K-split
    reduction), `ExactGemmTiling.v` (the output partition),
    `ExactGemmPacking.v` (the operand packing) and `ExactGemmMicro.v` (the
    int32 flush and the lane round trip). What was left unproved is the
    arithmetic core - the 2-D register tile itself - and this file covers the
    routing half of it.

    *** The division of labour, stated so the composition is checkable. ***

    - `ExactGemmPacking.v` : panel slot `2j + h` of k-pair group `p` holds
      `B[2p+h][j]`, and slot `2i + h` holds `A[i][2p+h]`, zero outside the
      live tile.
    - **This file** : the emitted broadcast, the four `<32 x i16>` B loads and
      `vpdpwssd` route those slots to accumulator `acc[i][v]` lane `l`, whose
      column is `16v + l`; and the store writes that lane to `C[i][16v+l]`.
    - `ExactGemmMicro.v` : the int32 accumulation across k-pair groups is
      exact under the licence's bound.

    Composed, that is "the tile computes the dot product". Neither half is
    interesting alone: the packing is a bijection that says nothing about
    which lane consumes a slot, and this file says nothing about what the
    slots hold.

    *** What is TAKEN AS GIVEN, and could not be otherwise. ***

    [vpdpwssd] is a DEFINITION here, not a theorem - it is an ISA fact, pinned
    empirically by `tests/cpu_gemm_vnni_micro.rs` running the real instruction
    against a scalar reference. Same for the little-endian order of the halves
    of an i32 ([i32_lo]/[i32_hi]): the emitter loads A's k-pair with a single
    `load i32`, and which half is `2p` rather than `2p+1` is a property of the
    machine. What IS proved is that the emitter's index arithmetic agrees with
    those definitions - and
    [swapping_the_pair_halves_computes_a_different_function] shows the
    endianness assumption is load-bearing rather than decorative.

    The masked tails are here too ([only_the_live_rectangle_is_stored]).
*)

Require Import ZArith Lia Arith.
Require ExactGemmSchedule.
Open Scope Z_scope.

Module SCH := ExactGemmSchedule.

(** The tile shape. Taken from [ExactGemmSchedule], which is GENERATED from
    `cpu_gemm.rs`'s own `VNNI_MR` / `VNNI_NRV` / `VNNI_NR` - so these are the
    emitter's constants rather than a transcription of them. They were literals
    here, and `MR` in particular was pinned by nothing in this file: setting it
    to 8 left the file compiling. *)
Definition MR : nat := SCH.MR.
Definition NRV : nat := SCH.NRV.
Definition NR : nat := SCH.NR.

(* ------------------------------------------------------------------ *)
(** ** The ISA facts, written down as definitions                      *)
(* ------------------------------------------------------------------ *)

(** A `<32 x i16>` register, as its 32 int16 elements. *)
Definition v32 := nat -> Z.

(** `vpdpwssd acc, a, b`: lane `l` of 16 consumes int16 elements `2l` and
    `2l+1` of BOTH operands and adds the two products into the int32 lane.
    Non-saturating, which is why `ExactGemmMicro.v` has to bound it. *)
Definition vpdpwssd (acc : nat -> Z) (a b : v32) (l : nat) : Z :=
  acc l + a (2 * l)%nat * b (2 * l)%nat
        + a (2 * l + 1)%nat * b (2 * l + 1)%nat.

(** The emitted A broadcast: `load i32` -> `insertelement` ->
    `shufflevector zeroinitializer` -> `bitcast to <32 x i16>`. Every one of
    the 16 i32 lanes holds the same pair, so int16 element `2l` is the LOW
    half and `2l+1` the high one, for every `l`. *)
Definition broadcast (lo hi : Z) : v32 :=
  fun e => if Nat.even e then lo else hi.

(** The emitted B load: `%bp{v} = bp + v*32` int16 elements, then a whole
    `<32 x i16>`. *)
Definition bvec (Bp : nat -> Z) (v : nat) : v32 :=
  fun e => Bp (v * 32 + e)%nat.

(** Where the packers put things. Both are `ExactGemmPacking.v`'s maps;
    `slot_b_is_the_plain_interleave` there proves the emitted vector-group
    form equals this. *)
Definition slot_a (i h : nat) : nat := SCH.slot_a i h.
Definition slot_b (j h : nat) : nat := SCH.slot_b_interleave j h.

(** Which column of the tile accumulator lane `l` of vector `v` is. *)
Definition col_of (v l : nat) : nat := SCH.col_of v l.

(* ------------------------------------------------------------------ *)
(** ** The broadcast really does pair the row's two k-values            *)
(* ------------------------------------------------------------------ *)

Lemma broadcast_even : forall lo hi l, broadcast lo hi (2 * l)%nat = lo.
Proof.
  intros. unfold broadcast.
  rewrite (proj2 (Nat.even_spec _)); [reflexivity | exists l; lia].
Qed.

Lemma broadcast_odd : forall lo hi l, broadcast lo hi (2 * l + 1)%nat = hi.
Proof.
  intros. unfold broadcast.
  replace (Nat.even (2 * l + 1)) with false; [reflexivity |].
  symmetry. apply Bool.not_true_is_false. rewrite Nat.even_spec.
  intros [k Hk]. lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** THE ROUTING THEOREM                                             *)
(* ------------------------------------------------------------------ *)

(** **Lane `l` of vector `v` consumes exactly its own column's two packed
    slots.** This is the join between the packing proof and the kernel: the
    B panel is addressed by `slot_b`, and this says the lane that reads
    `slot_b (16v+l) h` is the lane the store sends to column `16v+l`.

    A mismatch is a correctly-summed, column-permuted tile - the failure mode
    `ExactGemmMicro.v`'s round trip guards from the other side. *)
Theorem the_lane_consumes_its_own_column : forall acc Bp lo hi v l,
  vpdpwssd acc (broadcast lo hi) (bvec Bp v) l
  = acc l
    + lo * Bp (slot_b (col_of v l) 0)
    + hi * Bp (slot_b (col_of v l) 1).
Proof.
  intros acc Bp lo hi v l.
  unfold vpdpwssd, bvec, slot_b, col_of.
  rewrite broadcast_even, broadcast_odd.
  replace (v * 32 + 2 * l)%nat with (2 * (16 * v + l) + 0)%nat by lia.
  replace (v * 32 + (2 * l + 1))%nat with (2 * (16 * v + l) + 1)%nat by lia.
  reflexivity.
Qed.

(** **The i32 load is the int16 pair the packer wrote.** `%aidx = p * MR`,
    `%ai{i} = aidx + i`, then `getelementptr inbounds i32` - so element `i` of
    i32-group `p`. In int16 terms that is byte-identical to slots `2i` and
    `2i+1` of the panel's k-pair group `p`, whose base is `p * MR * 2`. *)
Definition i32_lo (Ap : nat -> Z) (n : nat) : Z := Ap (2 * n)%nat.
Definition i32_hi (Ap : nat -> Z) (n : nat) : Z := Ap (2 * n + 1)%nat.

Theorem the_i32_load_is_the_packed_pair : forall Ap p i,
  i32_lo Ap (p * MR + i)%nat = Ap (p * (MR * 2) + slot_a i 0)%nat
  /\ i32_hi Ap (p * MR + i)%nat = Ap (p * (MR * 2) + slot_a i 1)%nat.
Proof.
  intros Ap p i. unfold i32_lo, i32_hi, slot_a, SCH.slot_a, MR, SCH.MR.
  split; f_equal; lia.
Qed.

(** Put the two together: one `vpdpwssd` step contributes exactly the two
    k-terms of the dot product for `(row i, column 16v+l)`, read out of the
    panels at the slots `ExactGemmPacking.v` proves the packers wrote. *)
Theorem one_step_is_two_k_terms : forall acc Ap Bp p i v l,
  vpdpwssd acc
           (broadcast (i32_lo Ap (p * MR + i)) (i32_hi Ap (p * MR + i)))
           (bvec (fun e => Bp (p * (NR * 2) + e)%nat) v) l
  = acc l
    + Ap (p * (MR * 2) + slot_a i 0)%nat * Bp (p * (NR * 2) + slot_b (col_of v l) 0)%nat
    + Ap (p * (MR * 2) + slot_a i 1)%nat * Bp (p * (NR * 2) + slot_b (col_of v l) 1)%nat.
Proof.
  intros acc Ap Bp p i v l.
  rewrite the_lane_consumes_its_own_column.
  destruct (the_i32_load_is_the_packed_pair Ap p i) as [Hlo Hhi].
  rewrite Hlo, Hhi. reflexivity.
Qed.

(** **The refutation that makes the endianness assumption load-bearing.**
    Swap which half of the i32 is `2p` and the kernel computes a different
    function - not a rounding difference, a different answer. Concretely: row
    values 1 and 10 against column values 100 and 1. *)
Theorem swapping_the_pair_halves_computes_a_different_function :
  let acc := fun _ : nat => 0%Z in
  let b := fun e : nat => if Nat.even e then 100%Z else 1%Z in
  vpdpwssd acc (broadcast 1 10) b 0 = 110
  /\ vpdpwssd acc (broadcast 10 1) b 0 = 1001.
Proof. split; vm_compute; reflexivity. Qed.

(** ...and it is not vacuous the other way either: with a SYMMETRIC operand
    the swap is invisible, which is why the fixture above is asymmetric. *)
Theorem a_symmetric_operand_hides_the_swap :
  let acc := fun _ : nat => 0%Z in
  let b := fun _ : nat => 7%Z in
  vpdpwssd acc (broadcast 1 10) b 0 = vpdpwssd acc (broadcast 10 1) b 0.
Proof. vm_compute. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** The register tile is a bijection onto the MR x NR tile           *)
(* ------------------------------------------------------------------ *)

(** Every `(i, v, l)` with `i < MR`, `v < NRV`, `l < 16` names a distinct
    position of the tile, and every position is named. The accumulators are
    `MR * NRV = 24` registers of 16 lanes = 384 = `MR * NR` values. *)
Theorem tile_position_injective : forall (i1 v1 l1 i2 v2 l2 : nat),
  (v1 < NRV)%nat -> (l1 < 16)%nat -> (v2 < NRV)%nat -> (l2 < 16)%nat ->
  i1 = i2 -> col_of v1 l1 = col_of v2 l2 ->
  i1 = i2 /\ v1 = v2 /\ l1 = l2.
Proof.
  intros i1 v1 l1 i2 v2 l2 Hv1 Hl1 Hv2 Hl2 Hi Hc.
  unfold col_of, SCH.col_of in Hc. repeat split; lia.
Qed.

Theorem tile_position_in_range : forall v l,
  (v < NRV)%nat -> (l < 16)%nat -> (col_of v l < NR)%nat.
Proof. intros v l Hv Hl. unfold col_of, SCH.col_of, NRV, SCH.NRV, NR, SCH.NR in *. lia. Qed.

Theorem tile_position_surjective : forall j,
  (j < NR)%nat -> exists v l, (v < NRV)%nat /\ (l < 16)%nat /\ col_of v l = j.
Proof.
  intros j Hj. exists (j / 16)%nat, (j mod 16)%nat.
  pose proof (Nat.div_mod_eq j 16) as Hdm.
  pose proof (Nat.mod_upper_bound j 16 ltac:(lia)) as Hub.
  assert ((j / 16 < NRV)%nat).
  { unfold NRV, SCH.NRV. apply Nat.Div0.div_lt_upper_bound.
    unfold NR, SCH.NR in Hj. lia. }
  unfold col_of, SCH.col_of. repeat split; lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The masked tails                                                *)
(* ------------------------------------------------------------------ *)

(** The store is predicated on `i < mrows` and `col < ncols`: a ragged tile
    runs at FULL width and discards the rest. `stored` is what reaches C. *)
Definition stored (mrows ncols : nat) (i j : nat) : bool :=
  andb (Nat.ltb i mrows) (Nat.ltb j ncols).

Theorem only_the_live_rectangle_is_stored : forall mrows ncols i j,
  stored mrows ncols i j = true <-> ((i < mrows)%nat /\ (j < ncols)%nat).
Proof.
  intros. unfold stored. rewrite Bool.andb_true_iff.
  rewrite !Nat.ltb_lt. reflexivity.
Qed.

(** **This is what makes the packers' row and column masks redundant**, which
    `tests/exact_gemm_packing_model.rs` measured rather than assumed: a
    padding row or column is computed at full width and then discarded here,
    so garbage in it never reaches C. The phantom k-half is the one padding
    this does NOT cover - it contributes to a LIVE column - which is why that
    is the only shape whose mask a differential can catch. *)
Theorem a_padding_column_never_reaches_c : forall mrows ncols i j,
  (ncols <= j)%nat -> stored mrows ncols i j = false.
Proof.
  intros mrows ncols i j H. unfold stored.
  rewrite (proj2 (Nat.ltb_ge j ncols) H). apply Bool.andb_false_r.
Qed.

Theorem a_padding_row_never_reaches_c : forall mrows ncols i j,
  (mrows <= i)%nat -> stored mrows ncols i j = false.
Proof.
  intros mrows ncols i j H. unfold stored.
  rewrite (proj2 (Nat.ltb_ge i mrows) H). reflexivity.
Qed.

Print Assumptions the_lane_consumes_its_own_column.
Print Assumptions the_i32_load_is_the_packed_pair.
Print Assumptions one_step_is_two_k_terms.
Print Assumptions swapping_the_pair_halves_computes_a_different_function.
Print Assumptions tile_position_injective.
Print Assumptions tile_position_surjective.
Print Assumptions only_the_live_rectangle_is_stored.
Print Assumptions a_padding_column_never_reaches_c.
