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

    **How bad that is was MEASURED, and the measurement was then RE-RUN and
    found incomplete. Both readings are kept, because the correction is the
    interesting part.**

    The claim to check is whether a drifted definition would go on
    type-checking in its own file while silently disagreeing with the others.
    The first pass tried three: `ExactGemmRegisterTile.slot_b` shifted by one,
    its `NR` set to 32, and `ExactGemmMicro.slot` with the halves swapped. All
    three are caught by the file itself, and this header concluded that "each
    definition turns out to be pinned by a theorem in its own file".

    That conclusion is FALSE, and it is false for the constant nobody tried.
    Re-running the sweep over every duplicated definition: `RegisterTile.NRV`
    (4 -> 2) and `Packing.NR` (64 -> 32) behave like the original three, and
    **`MR` set to 8 - in EITHER `ExactGemmPacking` or
    `ExactGemmRegisterTile` - leaves that file compiling perfectly.** Both are
    caught only here and by the downstream chain. So the pinning was
    incidental for five definitions and ABSENT for the sixth, and the only
    thing standing between `MR` and silent drift was this file - a theorem
    somebody remembered to write.

    **That is what `proofs/ExactGemmSchedule.v` now removes the need for.** It
    is GENERATED from `cpu_gemm.rs`'s own constants, committed, and gated on
    byte-identity by `tests/exact_gemm_schedule_proof.rs`; every file above
    takes its constants and index maps from it. A drift is no longer a
    theorem's job to catch, in either direction: a `.v` edited by hand and a
    Rust constant moved without regenerating both fail the gate.

    *** What that does to the agreement theorems below. ***

    They are kept, and their character has changed - which is worth stating
    rather than leaving for a reader to infer. Where two files' copies now
    resolve to ONE generated definition (`MR`, `NR`, `col_of`, `slot_a`, and
    `Micro.slot` against `RegisterTile.slot_b`), the theorem no longer asserts
    that two independent definitions coincide; it asserts that the two files
    ALIAS the right one. That is a weaker claim, and a real one - nothing
    stops `Definition NR := SCH.MR`.

    [packing_and_register_tile_agree_on_the_b_slot] is the exception and keeps
    its full strength. The generated file deliberately carries the B map in
    TWO forms - the emitted `vpdpwssd` vector-group expression and the plain
    interleave that `panel_slot_decode` inverts - because collapsing them
    would make that theorem, and
    [the_agreement_is_not_vacuous] below, true by `reflexivity` and worth
    nothing.

    *** What is NOT closed. ***

    Constant drift only. The loop NEST is still hand-written `IrBuilder` calls
    in `cpu_gemm.rs`, and the tie between these models and the emitted LLVM is
    still two models meeting - the gap `docs/proof_carrying_kernels.md` names
    about itself. That is Phase 2.

    The second half is the file's real new content: the one composition step
    the pieces were built for, which no single file can state.

    *** What changed after the first commit. ***

    That step was proved under two HYPOTHESES about what the packed panels
    hold, and the file said so - "the one step of Phase 1 that is assumed
    rather than proved". The next thing done to it was to try to violate them,
    and **the real panel does not satisfy them as they were written**: they
    quantified over all `i` and `j`, and at `i = MR` the index named is the
    following k-pair group's first slot, not a pad. So the composition was
    true and unusable - to apply it you had to supply a premise satisfied only
    by a panel one group long.

    Both are now bounded and DISCHARGED, against
    [ExactGemmPacking.panel_decodes_its_own_write] and its uniqueness
    companion, giving [the_packed_panels_route_to_the_right_source_elements]
    with no hypothesis about panel contents at all. **A hypothesis nothing is
    shown to satisfy is the proof-shaped version of a licence nothing can
    violate** - which this repo already knows to check for, and had not
    checked here.
*)

Require Import ZArith Lia Arith.
Require ExactGemmSchedule.
Require ExactGemmPacking.
Require ExactGemmMicro.
Require ExactGemmRegisterTile.
Require ExactGemmTiling.
Require Decomposition.
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

(** The packers' contract. It was a HYPOTHESIS when this file was written,
    and the first thing done to it afterwards was to try to violate it - which
    is how the bound below came to be here.

    **As first stated it quantified over ALL `i` and ALL `j`, and in that form
    no real panel satisfies it.** At `i = MR` the write index
    `p*(2*MR) + 2*MR` is the FIRST slot of k-pair group `p+1`, which holds
    that group's data; an unbounded contract demands a zero there whenever
    row `MR` is past `mrows`. So the theorem below was true and useless: to
    apply it you had to supply a premise satisfied only by a panel one k-pair
    group long, never by the panel the emitted loop builds.
    `ExactGemmPacking.the_group_bound_is_load_bearing` refutes the unbounded
    form on concrete numbers, with a control showing the same panel agrees
    inside the group.

    It is no longer an assumption either way:
    [ExactGemmPacking.panel_decodes_its_own_write] proves it of the decoded
    panel, and [ExactGemmPacking.panel_is_the_only_solution] proves that panel
    is the only array the loop's writes admit. The section keeps the
    hypothesis form so the composition stays stated about *any* panel meeting
    the contract, and [the_packed_panels_route_to_the_right_source_elements]
    below discharges it. *)
Section Compose.

Variable A B : nat -> nat -> Z.        (* A i k, B k j *)
Variable Ap Bp : nat -> Z.             (* the packed panels *)
Variable mrows ncols kc p : nat.

Hypothesis pack_a_contract : forall i h,
  (i < ExactGemmRegisterTile.MR)%nat -> (h < 2)%nat ->
  Ap (p * (2 * ExactGemmRegisterTile.MR) + ExactGemmRegisterTile.slot_a i h)%nat
  = ExactGemmPacking.packed (A i) mrows kc i (2 * p + h)%nat.

Hypothesis pack_b_contract : forall j h,
  (j < ExactGemmRegisterTile.NR)%nat -> (h < 2)%nat ->
  Bp (p * (2 * ExactGemmRegisterTile.NR) + ExactGemmRegisterTile.slot_b j h)%nat
  = ExactGemmPacking.packed (fun k => B k j) ncols kc j (2 * p + h)%nat.

(** The tile's own bounds: 4 accumulator vectors of 16 lanes is 64 columns. *)
Lemma col_of_is_in_the_tile : forall v l,
  (v < ExactGemmRegisterTile.NRV)%nat -> (l < 16)%nat ->
  (ExactGemmRegisterTile.col_of v l < ExactGemmRegisterTile.NR)%nat.
Proof.
  intros v l Hv Hl.
  unfold ExactGemmRegisterTile.col_of, ExactGemmSchedule.col_of,
         ExactGemmRegisterTile.NR, ExactGemmSchedule.NR,
         ExactGemmRegisterTile.NRV, ExactGemmSchedule.NRV in *. lia.
Qed.

(** **THE COMPOSITION.** One `vpdpwssd` step on the packed panels contributes
    exactly the two k-terms of the SOURCE matrices' dot product for
    `(row i, column 16v + l)` - masked, so a padding row, column or k-half
    contributes zero.

    Packing says what a slot holds; the register tile says which lane reads it;
    together they say the lane reads the right source elements. Neither half
    can state this alone, which is the whole reason the split was worth
    making. *)
Theorem the_lane_accumulates_the_source_elements : forall acc i v l,
  (i < ExactGemmRegisterTile.MR)%nat ->
  (v < ExactGemmRegisterTile.NRV)%nat -> (l < 16)%nat ->
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
  intros acc i v l Hi Hv Hl.
  pose proof (col_of_is_in_the_tile v l Hv Hl) as Hcol.
  rewrite ExactGemmRegisterTile.the_lane_consumes_its_own_column.
  cbv beta.
  rewrite (pack_a_contract i 0 Hi ltac:(lia)).
  rewrite (pack_a_contract i 1 Hi ltac:(lia)).
  rewrite (pack_b_contract (ExactGemmRegisterTile.col_of v l) 0 Hcol ltac:(lia)).
  rewrite (pack_b_contract (ExactGemmRegisterTile.col_of v l) 1 Hcol ltac:(lia)).
  replace (2 * p + 0)%nat with (2 * p)%nat by lia.
  reflexivity.
Qed.

(** A dead row contributes nothing, whatever the panel padding holds - the
    masking survives the composition rather than having to be re-argued at
    each layer. *)
Corollary a_dead_row_contributes_nothing : forall acc i v l,
  (i < ExactGemmRegisterTile.MR)%nat ->
  (v < ExactGemmRegisterTile.NRV)%nat -> (l < 16)%nat ->
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
  intros acc i v l Hi Hv Hl Hdead.
  rewrite the_lane_accumulates_the_source_elements by assumption.
  unfold ExactGemmPacking.packed.
  rewrite (proj2 (Nat.ltb_ge i mrows) Hdead). simpl. lia.
Qed.

End Compose.

(* ------------------------------------------------------------------ *)
(** ** Discharging the contract: the seam is closed                    *)
(* ------------------------------------------------------------------ *)

(** The same statement over the panels the packer loops actually produce, with
    no hypothesis about what a panel holds. This is what the section was for;
    [ExactGemmPacking.panel_is_the_only_solution] is what makes it a claim
    about the emitted loop rather than about a chosen model.

    A hypothesis nothing is shown to satisfy is the proof-shaped version of a
    licence nothing can violate. This file spent a commit in that state. *)
Theorem the_packed_panels_route_to_the_right_source_elements :
  forall (A B : nat -> nat -> Z) (mrows ncols kc p : nat) acc i v l,
    (i < ExactGemmRegisterTile.MR)%nat ->
    (v < ExactGemmRegisterTile.NRV)%nat -> (l < 16)%nat ->
    ExactGemmRegisterTile.vpdpwssd
      acc
      (ExactGemmRegisterTile.broadcast
         (ExactGemmPacking.panel A mrows kc ExactGemmPacking.MR
            (p * (2 * ExactGemmRegisterTile.MR)
             + ExactGemmRegisterTile.slot_a i 0)%nat)
         (ExactGemmPacking.panel A mrows kc ExactGemmPacking.MR
            (p * (2 * ExactGemmRegisterTile.MR)
             + ExactGemmRegisterTile.slot_a i 1)%nat))
      (ExactGemmRegisterTile.bvec
         (fun e => ExactGemmPacking.panel (fun j k => B k j) ncols kc
                     ExactGemmPacking.NR
                     (p * (2 * ExactGemmRegisterTile.NR) + e)%nat) v)
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
  intros A B mrows ncols kc p acc i v l Hi Hv Hl.
  apply the_lane_accumulates_the_source_elements; try assumption.
  - intros i' h' Hi' Hh'.
    apply ExactGemmPacking.panel_decodes_its_own_write;
      [ unfold ExactGemmPacking.MR, ExactGemmRegisterTile.MR,
               ExactGemmSchedule.MR in *; lia
      | unfold ExactGemmPacking.MR, ExactGemmRegisterTile.MR,
               ExactGemmSchedule.MR in *; lia
      | exact Hh' ].
  - intros j' h' Hj' Hh'.
    (* [ExactGemmRegisterTile.slot_b] is the plain interleave by definition,
       and [packing_and_register_tile_agree_on_the_b_slot] above is what says
       that is the map the emitter writes. Unfolding rather than rewriting
       with the interleave theorem: `2*j+h` is a shape the accumulator index
       `2*p+h` also has, so a backwards rewrite hits the wrong occurrence. *)
    unfold ExactGemmRegisterTile.slot_b, ExactGemmSchedule.slot_b_interleave.
    apply ExactGemmPacking.panel_decodes_its_own_write;
      [ unfold ExactGemmPacking.NR, ExactGemmRegisterTile.NR,
               ExactGemmSchedule.NR in *; lia
      | unfold ExactGemmPacking.NR, ExactGemmRegisterTile.NR,
               ExactGemmSchedule.NR in *; lia
      | exact Hh' ].
Qed.

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

(* ------------------------------------------------------------------ *)
(** ** The flush interval and the output tile are ONE decomposition     *)
(* ------------------------------------------------------------------ *)

(** [ExactGemmMicro] cuts the k-pair range into intervals of `Fl` because
    `vpdpwssd` accumulates in int32 and must be widened before it wraps - an
    OVERFLOW budget. [ExactGemmTiling] cuts the output axis into tiles of `T`
    because the micro-kernel writes a fixed rectangle - a MEMORY partition.
    CLAUDE.md's own note warns against confusing the two, and it is right about
    their PURPOSE and wrong about their SHAPE: both are `t |-> min(t*X, ext)`.

    The emitter does not make it obvious, because it computes them differently.
    `chunk_end_ix` emits `min(iv + T, ext)` - an END, from which the width is
    recovered by subtracting the offset - while `tile_width_ix` emits
    `min(ext - iv, T)` - a WIDTH, directly. Two instruction sequences, one
    function, and each proof file developed the three-regime reconciliation
    from scratch.

    Both are [Decomposition.clamped] now, and this is the theorem that says the
    two spellings agree. Note what it is NOT: the emitter still emits two
    different sequences, and it should - one site has the offset in hand and the
    other has the end. This is a claim about the SCHEDULE, not about the code
    generator. *)
Theorem the_flush_interval_and_the_output_tile_are_the_same_family :
  forall X ext t, ExactGemmSchedule.tw ext X t = ExactGemmSchedule.cw X ext t.
Proof.
  intros X ext t.
  unfold ExactGemmSchedule.tw, ExactGemmSchedule.cw, ExactGemmSchedule.coff.
  rewrite Nat.mul_succ_l. symmetry. apply Decomposition.min_shift.
Qed.

(** The agreement has to cover the RAGGED case or it says nothing: on a full
    piece both spellings trivially give `X`, and every interesting property of
    a clamped family lives at the boundary. At `ext = 5`, `X = 3`, `t = 1` the
    piece is short - width 2, not 3 - and there the two forms are
    `min(5 - 3, 3)` and `min(2*3, 5) - 3`, which arrive at 2 by different
    routes.

    This does NOT establish that the two are distinct expressions; nothing
    inside Coq can, since `cw` defined as `tw` would make the theorem above
    `reflexivity`. What guards that is `tests/exact_gemm_schedule_proof.rs`,
    which regenerates `ExactGemmSchedule.v` from `cpu_gemm.rs` and requires
    byte-identity - so neither definition can be edited here at all. *)
Theorem the_agreement_covers_the_ragged_case :
  (* the piece really is short - stated, not assumed, because a version of
     this that merely evaluated both sides at some point passed perfectly
     after being moved to a FULL piece, where the agreement is trivial *)
  (ExactGemmSchedule.tw 5 3 1 < 3)%nat
  /\ ExactGemmSchedule.tw 5 3 1 = ExactGemmSchedule.cw 3 5 1.
Proof. split; [ vm_compute; lia | reflexivity ]. Qed.

Print Assumptions packing_and_register_tile_agree_on_the_b_slot.
Print Assumptions micro_and_register_tile_agree_on_the_b_slot.
Print Assumptions packing_and_register_tile_agree_on_the_a_slot.
Print Assumptions the_tile_shape_is_the_same_everywhere.
Print Assumptions the_agreement_is_not_vacuous.
Print Assumptions the_flush_interval_and_the_output_tile_are_the_same_family.
Print Assumptions the_agreement_covers_the_ragged_case.
Print Assumptions the_lane_accumulates_the_source_elements.
Print Assumptions a_dead_row_contributes_nothing.
Print Assumptions the_packed_panels_route_to_the_right_source_elements.
