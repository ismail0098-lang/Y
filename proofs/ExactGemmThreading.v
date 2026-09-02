(** * The threading layer: one record per worker, one private tile per worker,
      and a reduction that reads each of them exactly once.

    This is the arithmetic half of the item
    `exact_gemm_certificate::TRUST_BOUNDARY` carried as `Check::Unchecked` -
    "the threaded wrapper's `pthread` mechanics: the job struct's layout, the
    spawn/join protocol, and the per-thread private C buffer ... not modelled
    at all". [ExactGemmKsplit.v] proves what the K bands COMPUTE and says
    nothing about how they are dispatched, and the behavioural cover
    (`tests/exact_gemm_thread_invariance.rs`) compares ANSWERS across thread
    counts - so a dispatch bug that produces the right answer is invisible to
    it, and so is every over-allocation.

    ** Why this half is separable from the concurrency half.

    Three of the four things that item names are index arithmetic and are
    proved here: the job array is indexed by `(thread, field)`, the `pthread_t`
    array by `thread`, and each worker's private C by `(row, column)`. Each is
    a positional index, so [MixedRadix] carries the disjointness with no new
    reasoning - the fourth consumer of that schema, and the first reached from
    a concurrency layout rather than from a GEMM tile.

    The fourth thing - the happens-before edge that `pthread_join` establishes
    between a worker's last store and the reducer's first load - is NOT here
    and is not arithmetic. It needs a memory model, and saying so is the point
    of splitting the item rather than marking it closed.

    ** What is proved.

    - [job_slot_injective] / [job_slot_in_range] - thread `t`'s record cannot
      overlap thread `t'`'s, and lies inside the `malloc` sized for it.
    - [a_thirteenth_field_is_the_next_threads_first] - the sharp form of why
      `JOB_SLOTS` had to stop being a bare `96` at the `malloc`: one field past
      the record is not padding, it is the next worker's `A` pointer, and that
      worker may already be running.
    - [the_job_array_is_exactly_the_record_set] and its two siblings - the last
      slot written is the last slot allocated. This is the half no answer can
      see, and this repository has an over-allocation of exactly that shape in
      its history.
    - [the_private_c_is_read_back_exactly_once] - the reduction visits each
      element of each worker's buffer once, so no partial is dropped or
      double-counted.
    - [reading_the_private_buffer_at_the_callers_stride_leaves_it] - the
      concrete form of the bug that was observed as `double free or
      corruption`: `rows * cols` and `rows * ldc` are different buffers.
    - [the_reduction_is_the_ksplit_sum] - the join. The emitted `reduce.head`
      loop, started from what the zeroing loop left, is
      [ExactGemmKsplit.acc_bands], so [ExactGemmKsplit.ksplit_exact] applies to
      the loop that actually runs rather than to a fold that resembles it.
    - [a_destination_that_is_not_zeroed_is_wrong] - which makes the separate
      `zero.head` loop load-bearing rather than defensive: this kernel
      ACCUMULATES into C, and that is what lets the bands be summed.

    ** What is NOT proved HERE.

    - The concurrency. Nothing here says a worker's stores are visible to the
      reducer, that `pthread_join` orders them, or that the spawn loop's
      inline-fallback arm leaves the `tid` array in a state the join loop can
      read. Those are the remaining `Check::Unchecked` half.
    - That a successfully created thread never gets `pthread_t = 0`. The join
      loop uses `0` as its "never started" sentinel, which is an assumption
      about the C library's representation, not about this arithmetic.
    - The sizes are read from [ExactGemmSchedule], which is GENERATED from
      `src/cpu_gemm.rs` and byte-identity-gated, so "the number here is the
      number emitted" is that gate's claim rather than this file's. Which
      OFFSET each field is written and read at is
      `tests/exact_gemm_threading_layout.rs`'s claim, because it is a property
      of the emitted text and not of any number.

    Build:  coqc proofs/ExactGemmThreading.v      (Rocq 9.1)
*)

From Stdlib Require Import ZArith Arith Lia.
Require ExactGemmSchedule.
Require ExactGemmKsplit.
Require MixedRadix.
Require Decomposition.

Module SCH := ExactGemmSchedule.
Module KS := ExactGemmKsplit.
Module MR := MixedRadix.
Module D := Decomposition.

Open Scope nat_scope.

(* ------------------------------------------------------------------ *)
(** ** The generated sizes in the unit each index map is stated in      *)
(* ------------------------------------------------------------------ *)

(** The emitted `malloc`s are in BYTES and every map below is in slots or
    elements, so each size is divided by 8 once. `lia` cannot evaluate a
    division, which is why these three are lemmas rather than inlined.

    The first goes through [ExactGemmSchedule.jobs_bytes_is_a_record_per_thread]
    rather than unfolding the generated `96`, so that a change to `JOB_SLOTS`
    moves this file's arithmetic with it instead of failing here. Which is not
    an academic point: the first version unfolded the literal, and a probe that
    bumped `JOB_SLOTS` and regenerated broke `coqc` at this line - reporting a
    schedule change as a broken proof rather than as the over-allocation it
    is. *)
Lemma jobs_slots_total_unfold : forall nthr,
  SCH.jobs_slots_total nthr = nthr * SCH.JOB_SLOTS.
Proof.
  intro nthr. unfold SCH.jobs_slots_total.
  rewrite SCH.jobs_bytes_is_a_record_per_thread, Nat.mul_assoc.
  rewrite Nat.div_mul by lia. reflexivity.
Qed.

Lemma tids_slots_unfold : forall nthr, SCH.tids_slots nthr = nthr.
Proof.
  intro nthr. unfold SCH.tids_slots, SCH.tids_bytes.
  rewrite Nat.div_mul by lia. reflexivity.
Qed.

Lemma private_c_elems_unfold : forall rows cols,
  SCH.private_c_elems rows cols = rows * cols.
Proof.
  intros rows cols. unfold SCH.private_c_elems, SCH.private_c_bytes.
  rewrite Nat.div_mul by lia. reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The three index maps the emitted wrapper uses                   *)
(* ------------------------------------------------------------------ *)

(** Thread `t`'s field `k`, in 8-byte slots. The emitted spawn loop computes
    `%jb = t * 96` and then `getelementptr i8, ptr %jobs, i64 (%jb + 8*k)`. *)
Definition job_slot (t k : nat) : nat := MR.pack SCH.JOB_SLOTS t k.

(** Thread `t`'s `pthread_t`. One digit, so there is nothing to pack. *)
Definition tid_slot (t : nat) : nat := t.

(** Element `(r, c)` of a worker's PRIVATE C buffer, which is compact: the
    worker is handed `cols` as its output stride precisely so this is
    `r * cols + c` and not `r * ldc + c`. *)
Definition private_slot (cols r c : nat) : nat := MR.pack cols r c.

(** Element `(r, c)` of the CALLER's C, at the caller's stride. The reduction
    walks source and destination with different strides, which the flat loop it
    replaces did not. *)
Definition caller_slot (ldc r c : nat) : nat := MR.pack ldc r c.

(* ------------------------------------------------------------------ *)
(** ** The job array                                                   *)
(* ------------------------------------------------------------------ *)

(** Every field of every thread's record is inside the allocation. *)
Theorem job_slot_in_range : forall nthr t k,
  t < nthr -> k < SCH.JOB_SLOTS ->
  job_slot t k < SCH.jobs_slots_total nthr.
Proof.
  intros nthr t k Ht Hk. rewrite jobs_slots_total_unfold.
  unfold job_slot. apply MR.pack_in_range; assumption.
Qed.

(** Distinct `(thread, field)` pairs are distinct slots: no worker's record
    overlaps another's. The whole content is [MixedRadix.quot_rem_unique]. *)
Theorem job_slot_injective : forall t1 k1 t2 k2,
  k1 < SCH.JOB_SLOTS -> k2 < SCH.JOB_SLOTS ->
  job_slot t1 k1 = job_slot t2 k2 -> t1 = t2 /\ k1 = k2.
Proof.
  intros t1 k1 t2 k2 H1 H2 Heq. unfold job_slot, MR.pack in Heq.
  apply (MR.quot_rem_unique SCH.JOB_SLOTS); try assumption.
  unfold SCH.JOB_SLOTS. lia.
Qed.

(** **The last slot written is the last slot allocated.** The half that catches
    an over-allocation, which changes no answer. *)
Theorem the_job_array_is_exactly_the_record_set : forall nthr,
  0 < nthr ->
  S (job_slot (nthr - 1) (SCH.JOB_SLOTS - 1)) = SCH.jobs_slots_total nthr.
Proof.
  intros nthr H. rewrite jobs_slots_total_unfold.
  unfold job_slot, MR.pack, SCH.JOB_SLOTS. lia.
Qed.

(** ...and one slot short is an overflow, so the bound above is tight rather
    than merely sufficient. *)
Theorem one_slot_short_overflows_the_job_array :
  ~ (job_slot 3 (SCH.JOB_SLOTS - 1) < SCH.jobs_slots_total 4 - 1).
Proof. vm_compute. lia. Qed.

(** **A thirteenth field is the next worker's first field**, not padding.

    This is the sharp form of why the emitter's `96` had to become
    `JOB_SLOTS * 8`: the number is the record's SIZE and its STRIDE at once,
    so a field added at one role and not the other does not run into slack -
    it runs into a live neighbour whose worker may already be executing. For
    the last thread it runs off the allocation entirely. *)
Theorem a_thirteenth_field_is_the_next_threads_first : forall t,
  job_slot t SCH.JOB_SLOTS = job_slot (S t) 0.
Proof. intro t. unfold job_slot, MR.pack. lia. Qed.

(* ------------------------------------------------------------------ *)
(** ** The `pthread_t` array                                           *)
(* ------------------------------------------------------------------ *)

Theorem tid_slot_in_range : forall nthr t,
  t < nthr -> tid_slot t < SCH.tids_slots nthr.
Proof. intros nthr t H. rewrite tids_slots_unfold. exact H. Qed.

Theorem the_tid_array_is_exactly_the_thread_set : forall nthr,
  0 < nthr -> S (tid_slot (nthr - 1)) = SCH.tids_slots nthr.
Proof. intros nthr H. rewrite tids_slots_unfold. unfold tid_slot. lia. Qed.

(* ------------------------------------------------------------------ *)
(** ** The private C buffer, and the reduction that reads it           *)
(* ------------------------------------------------------------------ *)

Theorem private_slot_in_range : forall rows cols r c,
  r < rows -> c < cols ->
  private_slot cols r c < SCH.private_c_elems rows cols.
Proof.
  intros rows cols r c Hr Hc. rewrite private_c_elems_unfold.
  unfold private_slot. apply MR.pack_in_range; assumption.
Qed.

(** **Each element is read back exactly once.** Injective, so no partial is
    double-counted; onto, so nothing in the buffer is left behind. Together
    with [the_reduction_is_the_ksplit_sum] below that is what makes the
    reduction the sum of the bands rather than a sum of some of them. *)
Theorem the_private_c_is_read_back_exactly_once : forall cols r1 c1 r2 c2,
  c1 < cols -> c2 < cols ->
  private_slot cols r1 c1 = private_slot cols r2 c2 -> r1 = r2 /\ c1 = c2.
Proof.
  intros cols r1 c1 r2 c2 H1 H2 Heq. unfold private_slot, MR.pack in Heq.
  apply (MR.quot_rem_unique cols); try assumption. lia.
Qed.

Theorem every_element_of_the_private_buffer_is_reached : forall rows cols s,
  0 < cols -> s < SCH.private_c_elems rows cols ->
  exists r c, r < rows /\ c < cols /\ private_slot cols r c = s.
Proof.
  intros rows cols s Hc Hs. rewrite private_c_elems_unfold in Hs.
  apply MR.pack_onto; assumption.
Qed.

Theorem the_private_c_allocation_is_exactly_the_reduced_set :
  forall rows cols,
  0 < rows -> 0 < cols ->
  S (private_slot cols (rows - 1) (cols - 1)) = SCH.private_c_elems rows cols.
Proof.
  intros rows cols Hr Hc. rewrite private_c_elems_unfold.
  unfold private_slot, MR.pack. nia.
Qed.

Theorem one_element_short_overflows_the_private_c :
  ~ (private_slot 5 2 4 < SCH.private_c_elems 3 5 - 1).
Proof. vm_compute. lia. Qed.

(** **The bug that was observed as `double free or corruption`.** Sizing the
    private buffer from `cols` while addressing it at the caller's `ldc` puts
    `(rows-1)*(ldc-cols)` elements past the end. Concretely at
    `rows = 3, cols = 2, ldc = 4`: slot 9 of a 6-element buffer. *)
Theorem reading_the_private_buffer_at_the_callers_stride_leaves_it :
  ~ (caller_slot 4 2 1 < SCH.private_c_elems 3 2).
Proof. vm_compute. lia. Qed.

(** The caller's C needs `cols <= ldc` for its own addressing to be injective -
    the SAME hypothesis [ExactGemmTiling]'s fold-back needs, arrived at from a
    different loop. It is a property of C, not of one site. *)
Theorem a_row_stride_below_cols_aliases :
  caller_slot 3 1 0 = caller_slot 3 0 3.
Proof. reflexivity. Qed.

Theorem the_destination_is_injective_when_the_stride_admits_the_row :
  forall ldc cols r1 c1 r2 c2,
  cols <= ldc -> c1 < cols -> c2 < cols ->
  caller_slot ldc r1 c1 = caller_slot ldc r2 c2 -> r1 = r2 /\ c1 = c2.
Proof.
  intros ldc cols r1 c1 r2 c2 Hle H1 H2 Heq.
  unfold caller_slot, MR.pack in Heq.
  apply (MR.quot_rem_unique ldc); try lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The reduction loop, and its join to the K-split                 *)
(* ------------------------------------------------------------------ *)

Open Scope Z_scope.

(** `reduce.head` in the emitted module: for `t = 0 .. n-1`, load the
    destination, add worker `t`'s value, store it back. `c0` is what the
    separate `zero.head` loop left there. *)
Fixpoint reduce_into (c0 : Z) (g : nat -> Z) (n : nat) : Z :=
  match n with
  | O => c0
  | S k => reduce_into c0 g k + g k
  end.

Lemma reduce_into_is_c0_plus_fold : forall c0 g n,
  reduce_into c0 g n = c0 + D.acc_range Z.add g 0 n.
Proof.
  intros c0 g n. induction n as [| n IH]; cbn [reduce_into D.acc_range].
  - ring.
  - rewrite IH. rewrite Nat.add_0_l. ring.
Qed.

(** The loop IS [ExactGemmKsplit.acc_bands] when each worker's value is its own
    band's partial. Without this the file below is a fold that resembles the
    emitted one; with it, [ksplit_exact] is about the loop that runs. *)
Lemma the_reduction_is_acc_bands : forall f K nthr n,
  D.acc_range Z.add
    (fun t => KS.acc_range Z.add f (KS.boff K nthr t) (KS.blen K nthr t)) 0 n
  = KS.acc_bands Z.add f K nthr n.
Proof.
  intros f K nthr n. induction n as [| n IH];
    cbn [D.acc_range KS.acc_bands]; [ reflexivity | ].
  rewrite Nat.add_0_l, IH. reflexivity.
Qed.

(** **The join.** The emitted reduction, over a destination the zeroing loop
    left at 0, is the naive sum over the whole of K - at every thread count,
    with no hypothesis that the bands are equal or that `nthr` divides `K`. *)
Theorem the_reduction_is_the_ksplit_sum : forall f K nthr,
  (0 < nthr)%nat ->
  reduce_into 0
    (fun t => KS.acc_range Z.add f (KS.boff K nthr t) (KS.blen K nthr t)) nthr
  = KS.acc_range Z.add f 0 K.
Proof.
  intros f K nthr H.
  rewrite reduce_into_is_c0_plus_fold, the_reduction_is_acc_bands.
  rewrite KS.ksplit_exact by exact H. ring.
Qed.

(** **The zeroing loop is load-bearing.** This kernel ACCUMULATES into C -
    which is what lets the bands be summed at all - so a destination carrying
    anything but zero adds it to the answer. The single-threaded arm has the
    same property and the same zeroing loop above it. *)
Theorem a_destination_that_is_not_zeroed_is_wrong :
  reduce_into 5 (fun _ => 1) 3 <> reduce_into 0 (fun _ => 1) 3.
Proof. vm_compute. discriminate. Qed.

(** ...and the control: with the destination zeroed, the same fold is the sum
    of the partials and nothing else, so the theorem above is about the start
    value rather than about the fold. *)
Theorem a_zeroed_destination_is_exactly_the_partials :
  reduce_into 0 (fun _ => 1) 3 = 3.
Proof. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** Assumption reports                                              *)
(* ------------------------------------------------------------------ *)

Print Assumptions job_slot_in_range.
Print Assumptions job_slot_injective.
Print Assumptions the_job_array_is_exactly_the_record_set.
Print Assumptions one_slot_short_overflows_the_job_array.
Print Assumptions a_thirteenth_field_is_the_next_threads_first.
Print Assumptions tid_slot_in_range.
Print Assumptions the_tid_array_is_exactly_the_thread_set.
Print Assumptions private_slot_in_range.
Print Assumptions the_private_c_is_read_back_exactly_once.
Print Assumptions every_element_of_the_private_buffer_is_reached.
Print Assumptions the_private_c_allocation_is_exactly_the_reduced_set.
Print Assumptions one_element_short_overflows_the_private_c.
Print Assumptions reading_the_private_buffer_at_the_callers_stride_leaves_it.
Print Assumptions a_row_stride_below_cols_aliases.
Print Assumptions the_destination_is_injective_when_the_stride_admits_the_row.
Print Assumptions the_reduction_is_acc_bands.
Print Assumptions the_reduction_is_the_ksplit_sum.
Print Assumptions a_destination_that_is_not_zeroed_is_wrong.
Print Assumptions a_zeroed_destination_is_exactly_the_partials.
