(** * The exact-attention kernel's launch geometry.

    GENERATED from [src/exact_attention.rs]'s `sched_scores` / `sched_accum` by
    `tests/exact_attention_schedule.rs`. Do not edit: regenerate with

      << Y_REWRITE_ATTENTION_PROOF=1 cargo test --release --test exact_attention_schedule >>

    Why it exists. [GridStrideSplit.v] proves that worker `w` of `n` taking
    `{ i < S : i mod n = w }` visits every index exactly once, in any order,
    at any worker count - and it proves it for an ABSTRACT `n`. Nothing said
    the kernel's `n` was the right one. The instructions were hand-written in
    a PTX template, this proof quoted them in a comment, and a test recovered
    them from emitted text with a reaching-definition walker. That was the
    weakest tie in the programme, and the quoted comment had already gone
    stale: it showed `mul.lo.s32 %r9, %r9, %r5`, a two-writes-to-one-register
    form the kernel stopped using.

    Now the emitter and this file render ONE expression. The definitions below
    are not a transcription.

    The obligation is the PAIRING. `worker` mixes three hardware indices with
    two radices; `nworkers` must be the product of exactly those three indices'
    EXTENTS. Drop a factor and two threads share a residue class, so their keys
    are accumulated twice; add one and some class is claimed by no thread, so
    its keys are dropped. Neither is a crash, and on random data neither is
    reliably a wrong-looking number.

    That statement is a BIJECTION from the launch geometry's coordinate box
    onto `[0, nworkers)`, and it is a mixed-radix positional index - so
    [MixedRadix.v] discharges the injectivity with no new reasoning. Third
    consumer of that schema, and the first reached from a GPU launch geometry
    rather than from a GEMM tile.

    What is NOT claimed: nothing here is about the per-thread body, the integer
    exp, the Q0.28 weight or the int8 load. This file is the schedule and only
    the schedule. *)

Require Import Arith Lia.
Require MixedRadix.

Module MR := MixedRadix.

Open Scope nat_scope.

(* ------------------------------------------------------------------ *)
(** ** The emitted expressions                                         *)
(* ------------------------------------------------------------------ *)

Definition worker_scores (ctaid_x ntid_x tid_x : nat) : nat := ((ctaid_x * ntid_x) + tid_x).
Definition nworkers_scores (nctaid_x ntid_x : nat) : nat := (nctaid_x * ntid_x).
Definition worker_accum (ctaid_z nctaid_x ctaid_x ntid_x tid_x : nat) : nat := ((((ctaid_z * nctaid_x) + ctaid_x) * ntid_x) + tid_x).
Definition nworkers_accum (nctaid_z nctaid_x ntid_x : nat) : nat := ((nctaid_z * nctaid_x) * ntid_x).

(* ------------------------------------------------------------------ *)
(** ** Which index plays which role                                    *)
(* ------------------------------------------------------------------ *)

(** **The theorems below are half generated and half fixed, and the asymmetry
    is load-bearing.** Their BINDERS come from the emitted expression's own
    parameter list; their right-hand sides are fixed text naming the roles.

    Without them a relabelling is invisible. Every other theorem here applies
    `worker_accum` POSITIONALLY, and the parameter list is derived from the
    expression - so swapping two radices in the emitter renames the parameters
    in step, the definition stays the same function under different labels,
    and the byte-identity gate passes because the generated file moved too.
    That mutation survived a green sweep until these were added.

    Here the binder list moves and the right-hand side does not, so a swap
    makes the equation false and `coqc` rejects the file. Dropping an extent
    is caught harder still: the binder disappears while the right-hand side
    still names it, so the reference is unbound. *)

Theorem the_worker_index_is_a_mixed_radix_number :
  forall ctaid_z nctaid_x ctaid_x ntid_x tid_x : nat,
    worker_accum ctaid_z nctaid_x ctaid_x ntid_x tid_x
      = ctaid_z * (nctaid_x * ntid_x) + ctaid_x * ntid_x + tid_x.
Proof. intros. unfold worker_accum. ring. Qed.

Theorem the_worker_count_is_the_product_of_the_extents :
  forall nctaid_z nctaid_x ntid_x : nat,
    nworkers_accum nctaid_z nctaid_x ntid_x = nctaid_z * nctaid_x * ntid_x.
Proof. intros. unfold nworkers_accum. ring. Qed.

Theorem the_scores_worker_index_is_a_mixed_radix_number :
  forall ctaid_x ntid_x tid_x : nat,
    worker_scores ctaid_x ntid_x tid_x = ctaid_x * ntid_x + tid_x.
Proof. intros. unfold worker_scores. ring. Qed.

Theorem the_scores_worker_count_is_the_product_of_the_extents :
  forall nctaid_x ntid_x : nat,
    nworkers_scores nctaid_x ntid_x = nctaid_x * ntid_x.
Proof. intros. unfold nworkers_scores. ring. Qed.

(* ------------------------------------------------------------------ *)
(** ** One digit of a positional index                                 *)
(* ------------------------------------------------------------------ *)

(** The bound both radices need, in one place. *)
Lemma digit_bound : forall q r Q R, q < Q -> r < R -> q * R + r < Q * R.
Proof. intros. nia. Qed.

(* ------------------------------------------------------------------ *)
(** ** The worker map is a bijection onto the worker count             *)
(* ------------------------------------------------------------------ *)

(** **In range.** Every thread of the launch is a worker the reduction folds. *)
Theorem the_worker_is_below_the_worker_count :
  forall cz cx tx ncz ncx ntx,
    cz < ncz -> cx < ncx -> tx < ntx ->
    worker_accum cz ncx cx ntx tx < nworkers_accum ncz ncx ntx.
Proof.
  intros cz cx tx ncz ncx ntx Hz Hx Ht.
  unfold worker_accum, nworkers_accum.
  assert (H1 : cz * ncx + cx < ncz * ncx) by (apply digit_bound; assumption).
  assert (H2 : (cz * ncx + cx) * ntx + tx < (ncz * ncx) * ntx)
    by (apply digit_bound; assumption).
  lia.
Qed.

(** **Injective.** Two threads never share a worker id, so no key is
    accumulated twice. Discharged by [MixedRadix.two_digit_unique]: the worker
    index IS `q*(B1*B0) + m*B0 + r` at `B0 = ntid.x`, `B1 = nctaid.x`. *)
Theorem distinct_threads_are_distinct_workers :
  forall cz1 cx1 tx1 cz2 cx2 tx2 ncx ntx,
    0 < ncx -> 0 < ntx ->
    cx1 < ncx -> cx2 < ncx -> tx1 < ntx -> tx2 < ntx ->
    worker_accum cz1 ncx cx1 ntx tx1 = worker_accum cz2 ncx cx2 ntx tx2 ->
    cz1 = cz2 /\ cx1 = cx2 /\ tx1 = tx2.
Proof.
  intros cz1 cx1 tx1 cz2 cx2 tx2 ncx ntx Hncx Hntx Hx1 Hx2 Ht1 Ht2 Heq.
  unfold worker_accum in Heq.
  apply (MR.two_digit_unique ntx ncx cz1 cx1 tx1 cz2 cx2 tx2);
    try assumption.
  transitivity ((cz1 * ncx + cx1) * ntx + tx1); [ ring | ].
  rewrite Heq. ring.
Qed.

(** **Onto.** Every worker id the reduction folds is some thread's, so no
    residue class goes unclaimed and no key is dropped. *)
Theorem every_worker_is_some_thread :
  forall w ncz ncx ntx,
    0 < ncx -> 0 < ntx ->
    w < nworkers_accum ncz ncx ntx ->
    exists cz cx tx,
      cz < ncz /\ cx < ncx /\ tx < ntx /\
      worker_accum cz ncx cx ntx tx = w.
Proof.
  intros w ncz ncx ntx Hncx Hntx Hw.
  unfold nworkers_accum in Hw.
  exists (w / (ncx * ntx)), ((w / ntx) mod ncx), (w mod ntx).
  assert (Hdd : w / (ncx * ntx) = w / ntx / ncx).
  { rewrite Nat.Div0.div_div. f_equal. ring. }
  assert (E1 : (w / ntx / ncx) * ncx + (w / ntx) mod ncx = w / ntx).
  { rewrite Nat.mul_comm. symmetry. apply Nat.div_mod_eq. }
  assert (E2 : (w / ntx) * ntx + w mod ntx = w).
  { rewrite Nat.mul_comm. symmetry. apply Nat.div_mod_eq. }
  split; [ | split; [ | split ] ].
  - apply Nat.Div0.div_lt_upper_bound. nia.
  - apply Nat.mod_upper_bound. lia.
  - apply Nat.mod_upper_bound. lia.
  - unfold worker_accum. rewrite Hdd, E1, E2. reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The pairing is load-bearing                                     *)
(* ------------------------------------------------------------------ *)

(** **The refutation.** Drop `%nctaid.z` from the product - which is exactly
    what the worker count of `attn_scores` is - and the in-range property is
    FALSE, so workers alias.

    Stated as a refuted theorem rather than an exhibited witness: a witness
    shows the case exists, a refutation shows no proof of the weakened claim
    can exist. *)
Theorem dropping_the_z_extent_overflows_the_worker_count :
  ~ (forall cz cx tx ncz ncx ntx,
       cz < ncz -> cx < ncx -> tx < ntx ->
       worker_accum cz ncx cx ntx tx < nworkers_scores ncx ntx).
Proof.
  intro H. specialize (H 1 0 0 2 1 1).
  unfold worker_accum, nworkers_scores in H. simpl in H. lia.
Qed.

(** **The control.** The refutation above must not be read as "`attn_scores`'s
    worker count is wrong". It is right FOR ITS OWN KERNEL, which is launched
    with one CTA in z and does not read `%ctaid.z` at all: at `nctaid.z = 1`
    the two schedules are the same map. So one proof covers both entries. *)
Theorem the_scores_schedule_is_the_accumulate_schedule_at_one_z :
  forall cx tx ncx ntx,
    worker_accum 0 ncx cx ntx tx = worker_scores cx ntx tx
    /\ nworkers_accum 1 ncx ntx = nworkers_scores ncx ntx.
Proof.
  intros. unfold worker_accum, worker_scores, nworkers_accum, nworkers_scores.
  split; ring.
Qed.

Print Assumptions the_worker_is_below_the_worker_count.
Print Assumptions distinct_threads_are_distinct_workers.
Print Assumptions every_worker_is_some_thread.
Print Assumptions dropping_the_z_extent_overflows_the_worker_count.
Print Assumptions the_scores_schedule_is_the_accumulate_schedule_at_one_z.
Print Assumptions the_worker_index_is_a_mixed_radix_number.
Print Assumptions the_worker_count_is_the_product_of_the_extents.
Print Assumptions the_scores_worker_index_is_a_mixed_radix_number.
Print Assumptions the_scores_worker_count_is_the_product_of_the_extents.
Print Assumptions digit_bound.
