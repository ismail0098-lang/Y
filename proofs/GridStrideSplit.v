(** * The exact-attention kernel's grid-stride reduction, and the property the
      GEMM proofs never needed: COMMUTATIVITY.

    `src/exact_attention.rs` states the claim in its module header - "the answer
    does not depend on `blockDim.x`, `gridDim.x`, `gridDim.z`, or the order the
    atomics land" - and `tests/gpu_attention_invariance.rs` demonstrates it on
    the device at nine launch geometries. Nine geometries is not every geometry,
    and "the order the atomics land" is not a geometry at all: it is a property
    of a race, and no test can enumerate it.

    This is the third kernel in the programme and the first that is not a GEMM.
    Its decomposition is structurally different in TWO ways, and both matter:

    - **It is not an interval split.** Worker `w` of `n` takes
      `{ i < S : i mod n = w }` - the residue classes, interleaved, not the
      contiguous bands of `ExactGemmKsplit` or the proportional edges of
      `GemmBandSplit`. [stride_classes_partition] is the obligation, and it is
      a different proof: an index belongs to its class by arithmetic rather
      than by an accumulated offset.
    - **The partials are combined in an ARBITRARY order**, by
      `red.shared.add.u64` and `red.global.add.u64`. Both GEMM proofs fold the
      bands in index order, so associativity alone sufficed. Here the order is
      whatever the hardware chooses, so the argument needs **commutativity**
      too - [atomics_may_land_in_any_order] quantifies over every permutation.

    ** What is proved, and what it rests on.

    [grid_stride_exact] with no hypothesis but `0 < n`; then
    [any_worker_count_agrees], which is the `blockDim`/`gridDim` half of the
    module's claim, and [atomics_may_land_in_any_order], which is the other
    half. The refutations show both halves are about the ACCUMULATE:
    [rounding_breaks_the_stride_split] disagrees across worker counts, and
    [rounding_is_order_dependent] disagrees at a FIXED worker count purely from
    the order - a failure the GEMM kernels cannot even exhibit.

    ** It covers all three entry points now, and did not when it was written.

    `attn_scores` was one thread per key with a bounds guard and no loop, so it
    silently required `gridDim.x * blockDim.x >= S` - the OPPOSITE contract to
    the accumulate's, in the same file, under a module header advertising launch
    invariance. Measured at S=512 launched with 128 threads: 768 of 1024 score
    slots stale and the maximum wrong. It is the same residue-class partition
    now, so everything below applies to it verbatim; nothing here changed to
    make that true.

    Its reduction is a max rather than a sum. Order-independence there needs no
    theorem - max is associative, commutative AND idempotent - but its IDENTITY
    has to come from somewhere, and it used to come from the host: a signed max
    wants `i32::MIN`, which is not a uniform byte pattern, so `M` could not be
    seeded by the `memset(0)` that every other accumulator in the module takes.
    A caller who used one anyway got a silently wrong answer whenever a row's
    scores were ALL negative - a precondition that cannot fire on random int8
    data, which is the worst kind.

    The kernel supplies it now. `x ^ 0x80000000` is the order-preserving
    signed->unsigned bijection, so a max over the biased values is the max over
    the originals and `red.global.max.u32` has identity 0: a zeroed buffer MEANS
    `i32::MIN`. The two accumulating entries undo the bias when they load
    `M[b]`. Guarded by [a_zeroed_max_buffer_is_the_identity] and
    [the_accumulating_entries_undo_the_bias] in
    `tests/gpu_attention_invariance.rs`; the second of those exists because the
    undo is invisible at the temperature every other test shares (2^15 * KFix is
    exactly 2^34 there, so the u32 truncation annihilates a missing undo).

    Nothing here changed to accommodate that: `f` is an arbitrary function of the
    index, so the bias is inside `f` and the partition theorems are untouched.

    ** What is NOT proved.

    The per-thread body - the integer exp, the Q0.28 weight, the int8 `V` load
    - is not modelled; `f` is an arbitrary function of the index. The
    accumulator width obligation is [the_bound_is_one_unit_wide] here and is
    enforced in `src/exact_attention.rs` and checked in
    `tests/exact_attention_bounds.rs`. And unlike the two GEMM kernels this
    one's schedule is a PTX string template rather than an emitted expression,
    so the tie to the artifact is `tests/exact_attention_schedule.rs` reading
    the emitted text - weaker than the byte-identity the `Ix` layer gives, and
    named as such rather than glossed.

    Build:  coqc proofs/GridStrideSplit.v      (Rocq 9.1)
*)

From Stdlib Require Import ZArith Arith Lia List Permutation.
Require ExactGemmKsplit.

Import ListNotations.
Open Scope Z_scope.

Module KS := ExactGemmKsplit.

(* ------------------------------------------------------------------ *)
(** ** The decomposition                                               *)
(* ------------------------------------------------------------------ *)

(** The naive reduction: every term, in index order.
    [ExactGemmKsplit.acc_range] is reused verbatim - a third kernel, and the
    range fold is still not a property of any decomposition. *)
Definition sum_upto (op : Z -> Z -> Z) (f : nat -> Z) (S : nat) : Z :=
  KS.acc_range op f 0 S.

(** Worker `w` of `n`, from `emit`'s grid-stride loop:

      mad.lo.s32 %r7, %r4, %r5, %r6     ; worker = flat_cta * ntid.x + tid.x
      mul.lo.s32 %r9, %r9, %r5          ; nworkers = nctaid.z * nctaid.x * ntid.x
      mov.u32    %r14, %r7              ; i = worker
    LOOP_I:
      ...
      add.s32    %r14, %r14, %r9        ; i += nworkers

    so worker `w` visits exactly the indices below `S` congruent to `w`. *)
Fixpoint class_sum (op : Z -> Z -> Z) (f : nat -> Z) (w n S : nat) : Z :=
  match S with
  | O => 0
  | Datatypes.S k =>
      if Nat.eqb (k mod n) w then op (class_sum op f w n k) (f k)
      else class_sum op f w n k
  end.

(** The reduction: the first [t] workers' partials, folded together. *)
Fixpoint combine (op : Z -> Z -> Z) (f : nat -> Z) (n S t : nat) : Z :=
  match t with
  | O => 0
  | Datatypes.S t' => op (combine op f n S t') (class_sum op f t' n S)
  end.

(* ------------------------------------------------------------------ *)
(** ** The classes partition the sequence                              *)
(* ------------------------------------------------------------------ *)

(** Peeling the last index off the sequence hands one term to exactly one
    worker - the one numbered `S mod n`. This single lemma is the partition;
    everything below is bookkeeping around it. *)
Lemma combine_peel : forall f n S t,
  combine Z.add f n (Datatypes.S S) t
  = combine Z.add f n S t + (if Nat.ltb (S mod n) t then f S else 0).
Proof.
  intros f n S t. induction t as [| t IH].
  - cbn. ring.
  - cbn [combine]. rewrite IH. cbn [class_sum].
    destruct (Nat.eqb_spec (S mod n) t);
      destruct (Nat.ltb_spec (S mod n) t);
      destruct (Nat.ltb_spec (S mod n) (Datatypes.S t)); try lia; ring.
Qed.

(** **The partition.** Every index below `S` is claimed by exactly one of the
    `n` workers, so peeling it off moves exactly one term. *)
Theorem stride_classes_partition : forall f n S,
  (0 < n)%nat ->
  combine Z.add f n (Datatypes.S S) n = combine Z.add f n S n + f S.
Proof.
  intros f n S Hn. rewrite combine_peel.
  assert (S mod n < n)%nat by (apply Nat.mod_upper_bound; lia).
  destruct (Nat.ltb_spec (S mod n) n); [ reflexivity | lia ].
Qed.

Lemma combine_empty : forall f n t, combine Z.add f n 0 t = 0.
Proof. intros f n t. induction t as [| t IH]; cbn [combine]; [ reflexivity | ]. rewrite IH. reflexivity. Qed.

(** **The theorem.** Splitting a sequence across any number of workers by the
    grid-stride rule and summing their partials gives the naive sum. No
    hypothesis that `n` divides `S`, and none on how the workers were cut into
    blocks and grids. *)
Theorem grid_stride_exact : forall f n S,
  (0 < n)%nat ->
  combine Z.add f n S n = sum_upto Z.add f S.
Proof.
  intros f n S Hn. unfold sum_upto. induction S as [| S IH].
  - cbn [KS.acc_range]. apply combine_empty.
  - cbn [KS.acc_range]. rewrite stride_classes_partition by exact Hn.
    rewrite IH, Nat.add_0_l. reflexivity.
Qed.

(** The `blockDim.x` / `gridDim.x` / `gridDim.z` half of the module's claim,
    derived rather than assumed - and note it says nothing about launch
    geometry. Two worker counts agree because each separately equals the naive
    sum, however the workers were cut into blocks and grids. *)
Corollary any_worker_count_agrees : forall f S n1 n2,
  (0 < n1)%nat -> (0 < n2)%nat ->
  combine Z.add f n1 S n1 = combine Z.add f n2 S n2.
Proof.
  intros f S n1 n2 H1 H2.
  rewrite (grid_stride_exact f n1 S H1), (grid_stride_exact f n2 S H2). reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The atomics: an ARBITRARY order                                 *)
(* ------------------------------------------------------------------ *)

(** Neither GEMM proof needed this. Both fold their bands in index order, so
    associativity carried the whole argument; `red.global.add.u64` supplies its
    operands in whatever order the hardware chooses. *)
Lemma fold_right_add_shift : forall (l : list Z) a,
  fold_right Z.add a l = fold_right Z.add 0 l + a.
Proof. induction l as [| x l IH]; intros a; cbn; [ ring | rewrite IH; ring ]. Qed.

(** [combine] is the sum of the partials as a LIST, which is what lets the
    permutation argument below reach it. *)
Lemma combine_is_fold : forall f n S t,
  combine Z.add f n S t
  = fold_right Z.add 0 (map (fun w => class_sum Z.add f w n S) (seq 0 t)).
Proof.
  intros f n S t. induction t as [| t IH].
  - reflexivity.
  - cbn [combine]. rewrite IH, seq_S, map_app, fold_right_app.
    cbn [map fold_right]. rewrite Nat.add_0_l.
    (* the shift lemma must land on the RIGHT side; `rewrite` would otherwise
       take the left one, which is already at accumulator 0. *)
    symmetry. rewrite fold_right_add_shift. ring.
Qed.

Lemma fold_right_add_permutation : forall l1 l2 : list Z,
  Permutation l1 l2 -> fold_right Z.add 0 l1 = fold_right Z.add 0 l2.
Proof.
  intros l1 l2 H. induction H; cbn; try ring.
  - rewrite IHPermutation. ring.
  - rewrite IHPermutation1, IHPermutation2. reflexivity.
Qed.

(** **The other half of the module's claim.** However the atomics interleave -
    any permutation of the workers at all - the answer is the naive sum. *)
Theorem atomics_may_land_in_any_order : forall f n S order,
  (0 < n)%nat ->
  Permutation order (seq 0 n) ->
  fold_right Z.add 0 (map (fun w => class_sum Z.add f w n S) order)
  = sum_upto Z.add f S.
Proof.
  intros f n S order Hn Hperm.
  rewrite (fold_right_add_permutation _ (map (fun w => class_sum Z.add f w n S) (seq 0 n)))
    by (apply Permutation_map; exact Hperm).
  rewrite <- combine_is_fold. now apply grid_stride_exact.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Both halves are about the accumulate                            *)
(* ------------------------------------------------------------------ *)

(** Same crude rounding accumulate as `ExactGemmKsplit` - reused so the three
    kernels' refutations are comparable rather than each inventing a model. *)
Definition spike (k : nat) : Z := if Nat.eqb k 0 then 1500 else 20.

(** A second fixture, and it needs to be a second one: the input that makes the
    worker COUNT matter and the input that makes the atomics' ORDER matter are
    not the same. Under [KS.rnd] a long tail of small terms rounds away against
    a running total, which is what separates a split from the naive fold; two
    orders of three partials separate only when the truncation lands
    differently, which needs partials that are not multiples of the rounding
    unit. Three plausible fixtures agreed on both readings before this pair. *)
Definition tail (k : nat) : Z := if Nat.eqb k 0 then 1050 else 55.

(** Worker counts disagree, exactly as they do for a GEMM's K-split: 1900
    against the naive fold's 1500. *)
Theorem rounding_breaks_the_stride_split :
  combine KS.fadd spike 3 30 3 <> sum_upto KS.fadd spike 30.
Proof. vm_compute. discriminate. Qed.

(** **The failure the GEMM kernels cannot exhibit**: a FIXED worker count, the
    same partials, and two orders of the atomics landing - 1500 against 1600.
    This is what makes commutativity, and not just associativity, the property
    this kernel needs. *)
Theorem rounding_is_order_dependent :
  fold_left KS.fadd (map (fun w => class_sum KS.fadd tail w 3 3) (seq 0 3)) 0
  <> fold_left KS.fadd (map (fun w => class_sum KS.fadd tail w 3 3) (rev (seq 0 3))) 0.
Proof. vm_compute. discriminate. Qed.

(** The control for the pair above: with the exact accumulate the same two
    orders agree, so the refutation is about `fadd` and not about the classes.
    `rev (seq 0 3)` really is a permutation of `seq 0 3`, so this is an
    instance of [atomics_may_land_in_any_order]. *)
Theorem exact_is_order_independent :
  fold_left Z.add (map (fun w => class_sum Z.add tail w 3 3) (seq 0 3)) 0
  = fold_left Z.add (map (fun w => class_sum Z.add tail w 3 3) (rev (seq 0 3))) 0.
Proof. vm_compute. reflexivity. Qed.

(** A control on the controls: the two accumulates really do differ on this
    input, so nothing above passes because `fadd` happens to be `Z.add` here. *)
Theorem the_two_accumulates_differ_on_this_input :
  sum_upto KS.fadd spike 30 <> sum_upto Z.add spike 30
  /\ sum_upto KS.fadd tail 3 <> sum_upto Z.add tail 3.
Proof. vm_compute. split; discriminate. Qed.

(* ------------------------------------------------------------------ *)
(** ** The accumulator bound                                           *)
(* ------------------------------------------------------------------ *)

(** `MAX_EXACT_SEQ_LEN` in `src/exact_attention.rs`. A weight is Q0.28 so
    `p < 2^28`; `V` is int8 so `|v| <= 127`; the accumulator is 64-bit two's
    complement. Past this the sum WRAPS, which is a wrong answer and not an
    imprecise one - and it is untestable on a device, since it needs 2.7e8
    keys. The boundary is one unit wide, which is the property a sampled test
    cannot see. *)
Definition MAX_EXACT_SEQ_LEN : Z := (2 ^ 63) / ((2 ^ 28 - 1) * 127).

Theorem the_bound_is_one_unit_wide :
  MAX_EXACT_SEQ_LEN * ((2 ^ 28 - 1) * 127) < 2 ^ 63
  /\ (MAX_EXACT_SEQ_LEN + 1) * ((2 ^ 28 - 1) * 127) >= 2 ^ 63.
Proof. split; vm_compute; [ reflexivity | discriminate ]. Qed.

Print Assumptions grid_stride_exact.
Print Assumptions any_worker_count_agrees.
Print Assumptions atomics_may_land_in_any_order.
Print Assumptions stride_classes_partition.
Print Assumptions rounding_breaks_the_stride_split.
Print Assumptions rounding_is_order_dependent.
Print Assumptions exact_is_order_independent.
Print Assumptions the_bound_is_one_unit_wide.
