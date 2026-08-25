(** * Where the five exact-GEMM proofs meet.

    Phase 1 accumulated five files, each proving one obligation:

    - `ExactGemmKsplit.v`       the K-split reduction
    - `ExactGemmTiling.v`       the output partition
    - `ExactGemmPacking.v`      the operand packing
    - `ExactGemmMicro.v`        the int32 flush, and the lane round trip
    - `ExactGemmRegisterTile.v` the register tile's routing

    **Each was written self-contained, and that is a defect this file exists to
    fix.** Three of them define the B panel's slot map, in two different forms:
    `ExactGemmPacking.slot_b` is the emitted `(j/16)*32 + (j%16)*2 + h`, while
    `ExactGemmMicro.slot` and `ExactGemmRegisterTile.slot_b` are the plain
    `2j + h`. `MR`, `NR` and `col_of` are each defined twice. Nothing checked
    that any of those agreed.

    **How bad that is was MEASURED, and it is milder than it looks - the first
    draft of this header overstated it.** The claim to check is whether a
    drifted definition would go on type-checking in its own file while
    silently disagreeing with the others. Three attempts, all CAUGHT by the
    file itself: `ExactGemmRegisterTile.slot_b` shifted by one, its `NR` set
    to 32, and `ExactGemmMicro.slot` with the two halves swapped. Each
    definition turns out to be pinned by a theorem in its own file - the last
    one by a proof script rather than by the statement, which is weaker but
    still a gate.

    So these agreement theorems do not close a hole that was demonstrably
    open. What they do is make the pinning EXPLICIT and cross-file rather than
    incidental: today each file constrains its own copy by accident of what it
    happens to prove, and nothing says the three copies denote one map. That
    is worth stating, and it is a smaller claim than "the files could drift
    apart unnoticed".

    The second half is the file's real new content: the one composition step
    the pieces were built for, which no single file can state.
*)

Require Import ZArith Lia Arith.
Require ExactGemmPacking.
Require ExactGemmMicro.
Require ExactGemmRegisterTile.
Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** The five files agree on their shared definitions                *)
(* ------------------------------------------------------------------ *)

(** The emitted vector-group form and the plain interleave are the same map.
    `ExactGemmPacking.slot_b_is_the_plain_interleave` is what makes this hold;
    without it the register tile would be reasoning about a layout the packer
    does not produce. *)
Theorem packing_and_register_tile_agree_on_the_b_slot : forall j h,
  ExactGemmPacking.slot_b j h = ExactGemmRegisterTile.slot_b j h.
Proof.
  intros. rewrite ExactGemmPacking.slot_b_is_the_plain_interleave.
  reflexivity.
Qed.

Theorem micro_and_register_tile_agree_on_the_b_slot : forall j h,
  ExactGemmMicro.slot j h = ExactGemmRegisterTile.slot_b j h.
Proof. reflexivity. Qed.

Theorem packing_and_micro_agree_on_the_b_slot : forall j h,
  ExactGemmPacking.slot_b j h = ExactGemmMicro.slot j h.
Proof.
  intros. rewrite ExactGemmPacking.slot_b_is_the_plain_interleave.
  reflexivity.
Qed.

Theorem packing_and_register_tile_agree_on_the_a_slot : forall i h,
  ExactGemmPacking.slot_a i h = ExactGemmRegisterTile.slot_a i h.
Proof. reflexivity. Qed.

Theorem micro_and_register_tile_agree_on_the_column : forall v l,
  ExactGemmMicro.col_of v l = ExactGemmRegisterTile.col_of v l.
Proof. reflexivity. Qed.

Theorem the_tile_shape_is_the_same_everywhere :
  ExactGemmPacking.MR = ExactGemmRegisterTile.MR
  /\ ExactGemmPacking.NR = ExactGemmRegisterTile.NR
  /\ ExactGemmRegisterTile.NR = (ExactGemmRegisterTile.NRV * 16)%nat.
Proof. repeat split; reflexivity. Qed.

(** The control. Agreement is only worth asserting if disagreement is
    expressible - two maps that are equal by `reflexivity` on a shared
    definition would prove nothing. These are genuinely different expressions
    that happen to denote the same function, and here is a map that does not. *)
Theorem the_agreement_is_not_vacuous :
  ExactGemmPacking.slot_b 17 0 = ExactGemmRegisterTile.slot_b 17 0
  /\ ExactGemmPacking.slot_b 17 0 <> ((17 / 16) * 32 + (17 mod 16) * 4 + 0)%nat.
Proof. split; vm_compute; lia. Qed.

(* ------------------------------------------------------------------ *)
(** ** The composition step                                            *)
(* ------------------------------------------------------------------ *)

(** The packers' contract, as a hypothesis rather than as something re-derived
    here: panel slot `slot_a i h` of k-pair group `p` holds `A[i][2p+h]`,
    masked to the live tile, and `slot_b j h` holds `B[2p+h][j]`.

    `ExactGemmPacking.v` proves the maps are bijections and that the padded
    product equals the live dot product; it does not state "slot `s` holds
    source element `x`" as one reusable lemma, so the composition takes it as
    an explicit assumption. **That is the seam, and naming it is the point** -
    it is the one step of Phase 1 that is assumed rather than proved. *)
Section Compose.

Variable A B : nat -> nat -> Z.        (* A i k, B k j *)
Variable Ap Bp : nat -> Z.             (* the packed panels *)
Variable mrows ncols kc p : nat.

Hypothesis pack_a_contract : forall i h,
  (h < 2)%nat ->
  Ap (p * (2 * ExactGemmRegisterTile.MR) + ExactGemmRegisterTile.slot_a i h)%nat
  = ExactGemmPacking.packed (A i) mrows kc i (2 * p + h)%nat.

Hypothesis pack_b_contract : forall j h,
  (h < 2)%nat ->
  Bp (p * (2 * ExactGemmRegisterTile.NR) + ExactGemmRegisterTile.slot_b j h)%nat
  = ExactGemmPacking.packed (fun k => B k j) ncols kc j (2 * p + h)%nat.

(** **THE COMPOSITION.** One `vpdpwssd` step on the packed panels contributes
    exactly the two k-terms of the SOURCE matrices' dot product for
    `(row i, column 16v + l)` - masked, so a padding row, column or k-half
    contributes zero.

    Packing says what a slot holds; the register tile says which lane reads it;
    together they say the lane reads the right source elements. Neither half
    can state this alone, which is the whole reason the split was worth
    making. *)
Theorem the_lane_accumulates_the_source_elements : forall acc i v l,
  ExactGemmRegisterTile.vpdpwssd
    acc
    (ExactGemmRegisterTile.broadcast
       (Ap (p * (2 * ExactGemmRegisterTile.MR)
            + ExactGemmRegisterTile.slot_a i 0)%nat)
       (Ap (p * (2 * ExactGemmRegisterTile.MR)
            + ExactGemmRegisterTile.slot_a i 1)%nat))
    (ExactGemmRegisterTile.bvec
       (fun e => Bp (p * (2 * ExactGemmRegisterTile.NR) + e)%nat) v)
    l
  = acc l
    + ExactGemmPacking.packed (A i) mrows kc i (2 * p)%nat
      * ExactGemmPacking.packed
          (fun k => B k (ExactGemmRegisterTile.col_of v l)) ncols kc
          (ExactGemmRegisterTile.col_of v l) (2 * p)%nat
    + ExactGemmPacking.packed (A i) mrows kc i (2 * p + 1)%nat
      * ExactGemmPacking.packed
          (fun k => B k (ExactGemmRegisterTile.col_of v l)) ncols kc
          (ExactGemmRegisterTile.col_of v l) (2 * p + 1)%nat.
Proof.
  intros acc i v l.
  rewrite ExactGemmRegisterTile.the_lane_consumes_its_own_column.
  cbv beta.
  rewrite (pack_a_contract i 0 ltac:(lia)).
  rewrite (pack_a_contract i 1 ltac:(lia)).
  rewrite (pack_b_contract (ExactGemmRegisterTile.col_of v l) 0 ltac:(lia)).
  rewrite (pack_b_contract (ExactGemmRegisterTile.col_of v l) 1 ltac:(lia)).
  replace (2 * p + 0)%nat with (2 * p)%nat by lia.
  reflexivity.
Qed.

(** A dead row contributes nothing, whatever the panel padding holds - the
    masking survives the composition rather than having to be re-argued at
    each layer. *)
Corollary a_dead_row_contributes_nothing : forall acc i v l,
  (mrows <= i)%nat ->
  ExactGemmRegisterTile.vpdpwssd
    acc
    (ExactGemmRegisterTile.broadcast
       (Ap (p * (2 * ExactGemmRegisterTile.MR)
            + ExactGemmRegisterTile.slot_a i 0)%nat)
       (Ap (p * (2 * ExactGemmRegisterTile.MR)
            + ExactGemmRegisterTile.slot_a i 1)%nat))
    (ExactGemmRegisterTile.bvec
       (fun e => Bp (p * (2 * ExactGemmRegisterTile.NR) + e)%nat) v)
    l
  = acc l.
Proof.
  intros acc i v l Hdead.
  rewrite the_lane_accumulates_the_source_elements.
  unfold ExactGemmPacking.packed.
  rewrite (proj2 (Nat.ltb_ge i mrows) Hdead). simpl. lia.
Qed.

End Compose.

(** *** What is still NOT composed, stated rather than implied.

    This file joins PACKING to ROUTING for one `vpdpwssd` step. It does not
    chain that through the k-pair loop into `ExactGemmMicro`'s flush, nor
    through `ExactGemmTiling`'s output partition, nor through
    `ExactGemmKsplit`'s band reduction, into a single "the emitted kernel
    equals the naive nest" statement. Each of those four is proved in its own
    file over its own model; a genuine end-to-end theorem needs one shared
    model of the kernel, which is Phase 2's work, not a missing lemma here.

    And the two ISA facts remain definitions: `vpdpwssd`'s semantics and the
    little-endian order of an i32's halves. No proof over [Z] supplies those;
    `tests/cpu_gemm_vnni_micro.rs` does, on the real instruction. *)

Print Assumptions packing_and_register_tile_agree_on_the_b_slot.
Print Assumptions micro_and_register_tile_agree_on_the_b_slot.
Print Assumptions packing_and_register_tile_agree_on_the_a_slot.
Print Assumptions the_tile_shape_is_the_same_everywhere.
Print Assumptions the_agreement_is_not_vacuous.
Print Assumptions the_lane_accumulates_the_source_elements.
Print Assumptions a_dead_row_contributes_nothing.
