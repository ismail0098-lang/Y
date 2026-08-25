(** * The exact VNNI micro-kernel: the flush, and the lane round trip.

    Phase 1's fourth and last schedule obligation. `ExactGemmKsplit.v` proved
    the K-split reduction, `ExactGemmTiling.v` the output partition, and
    `ExactGemmPacking.v` the operand packing. What is left is the kernel's own
    inner core, and two things in it are provable:

    - **The flush.** `vpdpwssd` accumulates into int32, which wraps. The kernel
      therefore runs a bounded number of k-pairs into int32 accumulators, then
      widens them into an int64 running sum and re-zeroes them
      (`emit_vnni_flush` in `src/cpu_gemm.rs`). [flush_exact] says the chunked
      accumulation equals the plain sum; [flush_exact_in_int32] says the int32
      arithmetic agrees with [Z] provided no partial sum leaves the int32
      range; and [operand_bound_gives_no_overflow] derives that hypothesis from
      the operand magnitude bound the licence actually checks.

    - **The lane round trip.** `pack_b` places column `j`'s two k-values at
      panel slot `2j + h` (`ExactGemmPacking.v`). `vpdpwssd` consumes that slot
      in vector `slot / 32`, lane `(slot mod 32) / 2`; the store writes vector
      `v` lane `l` to column `16v + l`. [the_packed_column_is_the_stored_column]
      says the round trip is the identity. If those two disagreed, the kernel
      would compute a correctly-summed but COLUMN-PERMUTED answer - every
      element on the curve, so to speak, and every element in the wrong place.

    *** What this does NOT prove. ***

    The 2-D register tile and the masked tails are not modelled: a lane's
    contribution `f p` is taken as given rather than derived from the packed
    panels, so nothing here says the `<16 x i32>` accumulator for row `i` is
    fed row `i`'s broadcast. That, and the `vpdpwssd` semantics themselves, are
    ISA facts pinned by `tests/cpu_gemm_vnni_micro.rs` running the real
    instruction against a scalar reference.

    Nor does the int32 model here have anything to say about the WIDTH of the
    operands - that they fit int16 at all is the other half of the licence,
    discharged exhaustively over the finite int16 domain in
    `tests/exact_gemm_licence_obligations.rs`. A proof over [Z] plus an
    exhaustive check over the finite domain is stronger than either alone, and
    only because both are present.
*)

Require Import ZArith Lia Arith.
Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** Sums over a range                                               *)
(* ------------------------------------------------------------------ *)

(** `sum_from f lo len` is the sum of `f lo, ..., f (lo+len-1)`, associated to
    the LEFT, in the order the accumulator actually visits them. *)
Fixpoint sum_from (f : nat -> Z) (lo len : nat) : Z :=
  match len with
  | O => 0
  | S n => sum_from f lo n + f (lo + n)%nat
  end.

Lemma sum_from_split : forall f lo a b,
  sum_from f lo (a + b) = sum_from f lo a + sum_from f (lo + a) b.
Proof.
  intros f lo a b. induction b as [| b IH].
  - rewrite Nat.add_0_r. simpl. lia.
  - replace (a + S b)%nat with (S (a + b))%nat by lia.
    cbn [sum_from]. rewrite IH.
    replace (lo + (a + b))%nat with (lo + a + b)%nat by lia.
    lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The flush chunks partition the k-pair range                     *)
(* ------------------------------------------------------------------ *)

Section Flush.

Variable f : nat -> Z.
Variable Fl : nat.                      (* the flush interval, in k-pairs *)
Hypothesis HFl : (0 < Fl)%nat.

(** The emitted loop is `for c = 0; c < kpairs; c += Fl`, with
    `cend = min(c + Fl, kpairs)` - so chunk `t` starts at `t*Fl` and is
    CLAMPED, exactly like the ragged output tile. *)
Definition coff (t : nat) : nat := (t * Fl)%nat.
Definition cw (n t : nat) : nat := (Nat.min (coff (S t)) n - coff t)%nat.
Definition nchunks (n : nat) : nat := ((n + Fl - 1) / Fl)%nat.

(** The int64 running sum: one `+=` per chunk, which is what the flush block
    emits. *)
Fixpoint chunk_acc (n t : nat) : Z :=
  match t with
  | O => 0
  | S t' => chunk_acc n t' + sum_from f (coff t') (cw n t')
  end.

Lemma chunk_acc_prefix : forall n t,
  chunk_acc n t = sum_from f 0 (Nat.min (coff t) n).
Proof.
  intros n t. induction t as [| t IH].
  - cbn [chunk_acc coff]. rewrite ?Nat.mul_0_l. rewrite Nat.min_0_l. reflexivity.
  - cbn [chunk_acc]. rewrite IH. unfold cw.
    destruct (Nat.le_gt_cases n (coff t)) as [Hle | Hgt].
    + (* this chunk is entirely past the end: it is empty *)
      rewrite (Nat.min_r (coff t) n) by lia.
      assert (Nat.min (coff (S t)) n = n) as ->.
      { apply Nat.min_r. unfold coff in *. nia. }
      replace (n - coff t)%nat with O by lia.
      cbn [sum_from]. lia.
    + rewrite (Nat.min_l (coff t) n) by lia.
      set (e := Nat.min (coff (S t)) n).
      assert (coff t <= e)%nat as He.
      { unfold e, coff in *. apply Nat.min_glb; nia. }
      replace e with (coff t + (e - coff t))%nat at 2 by lia.
      rewrite sum_from_split. rewrite Nat.add_0_l.
      replace (coff t + (e - coff t))%nat with e by lia.
      reflexivity.
Qed.

(** The chunk count really does reach the end. *)
Lemma nchunks_spans : forall n, (n <= coff (nchunks n))%nat.
Proof.
  intros n. unfold coff, nchunks.
  pose proof (Nat.div_mod_eq (n + Fl - 1) Fl) as Hdm.
  pose proof (Nat.mod_upper_bound (n + Fl - 1) Fl ltac:(lia)) as Hub.
  nia.
Qed.

(** **THE FLUSH THEOREM.** However `kpairs` is cut into intervals of `Fl`, the
    per-chunk partials added into the int64 sum give the plain sum. No
    hypothesis that `Fl` divides `n`: the final partial chunk is carried by the
    same clamp. *)
Theorem flush_exact : forall n,
  chunk_acc n (nchunks n) = sum_from f 0 n.
Proof.
  intros n. rewrite chunk_acc_prefix.
  rewrite Nat.min_r by apply nchunks_spans. reflexivity.
Qed.

(** The refutation: a loop that stopped at the last WHOLE interval would drop
    the tail. This is the flush's version of `dropping_the_remainder_loses_terms`. *)
Theorem stopping_at_the_last_whole_interval_loses_the_tail :
  let g := fun _ : nat => 1 in
  sum_from g 0 5 <> sum_from g 0 ((5 / 2) * 2)%nat.
Proof. cbn. lia. Qed.

End Flush.

(* ------------------------------------------------------------------ *)
(** ** int32 accumulation, and when it agrees with [Z]                 *)
(* ------------------------------------------------------------------ *)

Definition I32MIN : Z := -2147483648.
Definition I32MAX : Z := 2147483647.
Definition in_i32 (z : Z) : Prop := I32MIN <= z <= I32MAX.

Definition wrap32 (z : Z) : Z := (z - I32MIN) mod 4294967296 + I32MIN.

Lemma wrap32_id : forall z, in_i32 z -> wrap32 z = z.
Proof.
  intros z [Hlo Hhi]. unfold wrap32, I32MIN, I32MAX in *.
  rewrite Z.mod_small by lia. lia.
Qed.

(** The int32 accumulator, in the order the kernel visits its k-pairs. *)
Fixpoint wsum (f : nat -> Z) (lo len : nat) : Z :=
  match len with
  | O => 0
  | S n => wrap32 (wsum f lo n + f (lo + n)%nat)
  end.

(** **The int32 arithmetic agrees with [Z] exactly when no partial sum leaves
    the range.** This is the precise content of "the licence is the promise
    that it cannot overflow within one flush interval". *)
Theorem flush_exact_in_int32 : forall f lo len,
  (forall i, (i <= len)%nat -> in_i32 (sum_from f lo i)) ->
  wsum f lo len = sum_from f lo len.
Proof.
  intros f lo len. induction len as [| len IH]; intros Hb.
  - reflexivity.
  - cbn [wsum sum_from]. rewrite IH by (intros i Hi; apply Hb; lia).
    apply wrap32_id. apply (Hb (S len)). lia.
Qed.

(** The refutation, computed rather than argued: two k-pairs of 1.5e9 overflow,
    and the wrapped answer is not merely imprecise - it has the wrong SIGN. *)
Theorem overflow_breaks_the_flush :
  let g := fun _ : nat => 1500000000 in
  wsum g 0 2 = -1294967296 /\ sum_from g 0 2 = 3000000000.
Proof. split; vm_compute; reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** The licence's bound is what discharges the hypothesis           *)
(* ------------------------------------------------------------------ *)

Lemma abs_sum_bound : forall f lo len m,
  0 <= m ->
  (forall p, Z.abs (f p) <= 2 * m * m) ->
  Z.abs (sum_from f lo len) <= Z.of_nat len * (2 * m * m).
Proof.
  intros f lo len m Hm Hf. induction len as [| len IH].
  - cbn. lia.
  - cbn [sum_from]. rewrite Nat2Z.inj_succ.
    eapply Z.le_trans; [apply Z.abs_triangle |].
    specialize (Hf (lo + len)%nat). nia.
Qed.

(** `VnniExact::max_operand_magnitude` is `floor(sqrt(i32::MAX / (2 * Fl)))`,
    i.e. it licenses `m` exactly when `2 * Fl * m^2 <= i32::MAX`. Given that,
    every partial sum inside a chunk stays in range - which is precisely the
    hypothesis [flush_exact_in_int32] needs. *)
Theorem operand_bound_gives_no_overflow : forall f lo len Fl m,
  0 <= m ->
  (len <= Fl)%nat ->
  2 * Z.of_nat Fl * m * m <= I32MAX ->
  (forall p, Z.abs (f p) <= 2 * m * m) ->
  forall i, (i <= len)%nat -> in_i32 (sum_from f lo i).
Proof.
  intros f lo len Fl m Hm Hlen Hlic Hf i Hi.
  pose proof (abs_sum_bound f lo i m Hm Hf) as Habs.
  assert (Z.of_nat i <= Z.of_nat Fl) by (apply Nat2Z.inj_le; lia).
  assert (0 <= Z.of_nat i) by apply Nat2Z.is_nonneg.
  unfold in_i32, I32MIN, I32MAX in *.
  assert (Hb2 : Z.abs (sum_from f lo i) <= 2147483647) by nia.
  apply Z.abs_le in Hb2. lia.
Qed.

(** And the two halves compose: bounded operands, a chunk no longer than the
    flush interval, and the int32 accumulator is exact. *)
Corollary the_licence_makes_the_chunk_exact : forall f lo len Fl m,
  0 <= m ->
  (len <= Fl)%nat ->
  2 * Z.of_nat Fl * m * m <= I32MAX ->
  (forall p, Z.abs (f p) <= 2 * m * m) ->
  wsum f lo len = sum_from f lo len.
Proof.
  intros. apply flush_exact_in_int32.
  eapply operand_bound_gives_no_overflow; eauto.
Qed.

(** **The boundary is one unit wide**, at the shipped `DEFAULT_FLUSH_K_PAIRS`
    of 64. `tests/exact_gemm_licence_obligations.rs` exhausts the int16 domain
    against the real function; this pins the same two numbers here, so a change
    to either width or interval that moves the edge breaks both. *)
Theorem the_default_interval_licenses_4095_and_not_4096 :
  2 * 64 * 4095 * 4095 <= I32MAX /\ ~ (2 * 64 * 4096 * 4096 <= I32MAX).
Proof. unfold I32MAX. split; [lia | lia]. Qed.

Theorem the_4096_case_exceeds_by_exactly_one :
  2 * 64 * 4096 * 4096 = I32MAX + 1.
Proof. unfold I32MAX. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** The lane round trip                                             *)
(* ------------------------------------------------------------------ *)

(** `pack_b` writes column `j`'s `h`-th k-value at this slot of the k-pair
    group. `ExactGemmPacking.v` proves the emitted `(j/16)*32 + (j%16)*2 + h`
    equals this. *)
Definition slot (j h : nat) : nat := (2 * j + h)%nat.

(** How `vpdpwssd` reads that slot: 32 int16 per `<32 x i16>` vector, and lane
    `l` of the 16 consumes elements `2l` and `2l+1` of its own vector. *)
Definition vec_of_slot (s : nat) : nat := (s / 32)%nat.
Definition lane_of_slot (s : nat) : nat := ((s mod 32) / 2)%nat.

(** How the store reads the accumulator back out: vector `v`, lane `l` is
    column `16v + l` of the tile. *)
Definition col_of (v l : nat) : nat := (16 * v + l)%nat.

(** **The round trip is the identity.** Pack a column, let the hardware route
    it to a lane, store that lane back out, and you land on the column you
    started from. A mismatch here is a correctly-summed, column-PERMUTED
    result - every partial sum right, every value in the wrong place, and no
    bound or bijection anywhere else able to see it. *)
Theorem the_packed_column_is_the_stored_column : forall j h,
  (h < 2)%nat ->
  col_of (vec_of_slot (slot j h)) (lane_of_slot (slot j h)) = j.
Proof.
  intros j h Hh. unfold col_of, vec_of_slot, lane_of_slot, slot.
  pose proof (Nat.div_mod_eq j 16) as Hj.
  pose proof (Nat.mod_upper_bound j 16 ltac:(lia)) as Hjm.
  set (a := (j / 16)%nat) in *. set (b := (j mod 16)%nat) in *.
  replace (2 * j + h)%nat with ((2 * b + h) + a * 32)%nat by lia.
  rewrite Nat.div_add by lia.
  rewrite Nat.Div0.mod_add.
  rewrite (Nat.mod_small (2 * b + h) 32) by lia.
  rewrite (Nat.div_small (2 * b + h) 32) by lia.
  replace (2 * b + h)%nat with (b * 2 + h)%nat by lia.
  rewrite Nat.div_add_l by lia.
  rewrite (Nat.div_small h 2) by lia.
  lia.
Qed.

(** The control: it is NOT the identity for a lane stride the hardware does not
    use. Without this, "the round trip works" is satisfied by any pair of maps
    that happen to compose to the identity, including a wrong pair. *)
Theorem a_wrong_lane_stride_permutes_the_columns :
  col_of (vec_of_slot (slot 8 0)) ((slot 8 0 mod 32) / 4)%nat <> 8%nat.
Proof. vm_compute. lia. Qed.

Print Assumptions flush_exact.
Print Assumptions flush_exact_in_int32.
Print Assumptions overflow_breaks_the_flush.
Print Assumptions operand_bound_gives_no_overflow.
Print Assumptions the_licence_makes_the_chunk_exact.
Print Assumptions the_default_interval_licenses_4095_and_not_4096.
Print Assumptions the_packed_column_is_the_stored_column.
Print Assumptions a_wrong_lane_stride_permutes_the_columns.
