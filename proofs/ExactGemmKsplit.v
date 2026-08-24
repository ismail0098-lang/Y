(** * The exact GEMM's K-split schedule, mechanically verified.

    Phase 0 of [docs/proof_carrying_kernels.md] established that Y can
    substitute a tiled, packed, threaded [vpdpwssd] GEMM for a naive
    [@ZeroDrift] nest and get a bit-identical answer at every thread count.
    That is a TESTED property: [tests/exact_gemm_thread_invariance.rs] runs six
    thread counts over one ragged K and compares bytes. This file turns the
    part of it that is a schedule into a THEOREM.

    *** What is proved. ***

    [__y_gemm_exact_vnni_threaded] cuts K into [nthr] contiguous bands - the
    first [K mod nthr] of them one longer than the rest - runs one worker per
    band into a private zeroed C, and sums the partials. Three results:

    - [bands_tile] : the bands cover [0, K) exactly. No gap, no overlap, and
      the last one ends at K. This is the obligation a dropped remainder
      violates, and [dropping_the_remainder_loses_terms] refutes that mutation
      by computation.
    - [ksplit_exact] : summing the per-band partials equals the naive sum over
      the whole of K, for every f, every K and every positive nthr.
    - [any_thread_count_agrees] : the corollary the test asserts - two
      different thread counts give the same answer - which follows from
      [ksplit_exact] rather than from anything about threads.

    *** Why this is a proof at all: the accumulate is EXACT. ***

    Section 2 of the roadmap claims that exactness is what makes the
    equivalence provable, and that under a rounding accumulate the same
    schedule is simply WRONG rather than approximately right. That claim is
    machine-checked here rather than asserted:
    [rounding_breaks_the_split] and [exact_survives_the_same_split] are the
    SAME f, the same K and the same nthr, differing only in the accumulate.
    The rounded one disagrees with its own naive nest (1100 against 1000, the
    textbook loss of small terms to a large accumulator) while the exact one
    agrees (1200 either way).

    *** What this does NOT prove. ***

    A proof about a MODEL, as [ZkControlFlow.v] is. No extraction, no
    refinement proof against [src/cpu_gemm.rs]. What ties them together is
    [tests/exact_gemm_ksplit_model.rs], which checks the Rust transcription of
    [blen]/[boff] against this file's theorem over a finite range and against
    the emitted kernel's observed thread count.

    Three abstractions are deliberate and each one is a real gap:

    - A band's partial is modelled as [acc_range], the exact sum over that
      band's indices. THE MICRO-KERNEL IS NOT VERIFIED. Packing, the 2-D
      register tile, the masked tails and the periodic int32 -> int64 flush are
      all assumed to compute that sum. This file proves the schedule around
      them; Phase 1's other half is proving they fill it.
    - Accumulators are [Z], which is unbounded. Nothing here says a partial
      sum fits in int32 between flushes - that is the licence obligation, and
      it is discharged separately and EXHAUSTIVELY (over all 32768 int16
      magnitudes) in [tests/exact_gemm_licence_obligations.rs]. A proof over
      [Z] plus an exhaustive check over the finite domain is stronger than
      either alone, but only because both are present.
    - Only the K axis is split here. The M and N partitioning inside the
      driver is a disjoint decomposition of the OUTPUT, which is a different
      obligation (every element written once) and is not modelled.

    Build:  coqc proofs/ExactGemmKsplit.v      (Rocq 9.1)
*)

From Stdlib Require Import ZArith Arith Lia.

Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** The band decomposition the emitter computes                     *)
(* ------------------------------------------------------------------ *)

(** Transcribed from [cpu_gemm::emit_vnni_threaded_module]:

      %base = sdiv i64 %K, %nthr
      %rem  = srem i64 %K, %nthr
      %extra = icmp slt i64 %t, %rem      ; first `rem` bands take one more

    The cuts are uneven on purpose - a band boundary that lined up with the
    flush interval would make the whole family of tests agree for the wrong
    reason. *)

Definition blen (K nthr t : nat) : nat :=
  (K / nthr + (if Nat.ltb t (K mod nthr) then 1 else 0))%nat.

Fixpoint boff (K nthr t : nat) : nat :=
  match t with
  | O => O
  | S t' => (boff K nthr t' + blen K nthr t')%nat
  end.

Lemma boff_closed : forall K nthr t,
  boff K nthr t = (t * (K / nthr) + Nat.min t (K mod nthr))%nat.
Proof.
  intros K nthr t. induction t as [| t IH].
  - cbn [boff]. rewrite Nat.min_0_l. lia.
  - cbn [boff]. rewrite IH. unfold blen.
    destruct (Nat.ltb_spec t (K mod nthr)) as [Hlt | Hge].
    + rewrite (Nat.min_l t (K mod nthr)) by lia.
      rewrite (Nat.min_l (S t) (K mod nthr)) by lia. lia.
    + rewrite (Nat.min_r t (K mod nthr)) by lia.
      rewrite (Nat.min_r (S t) (K mod nthr)) by lia. lia.
Qed.

(** The obligation a dropped remainder violates: the bands end exactly at K. *)
Theorem bands_tile : forall K nthr,
  (0 < nthr)%nat -> boff K nthr nthr = K.
Proof.
  intros K nthr Hn. rewrite boff_closed.
  assert (Hm : (K mod nthr < nthr)%nat) by (apply Nat.mod_upper_bound; lia).
  rewrite Nat.min_r by lia.
  pose proof (Nat.div_mod_eq K nthr) as Hdm. lia.
Qed.

(** Contiguity, stated on its own because it is what the induction below
    consumes: band [t] starts exactly where band [t-1] stopped. *)
Lemma bands_contiguous : forall K nthr t,
  boff K nthr (S t) = (boff K nthr t + blen K nthr t)%nat.
Proof. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** The schedule, over an arbitrary accumulate                      *)
(* ------------------------------------------------------------------ *)

Section Schedule.
  (** [op a x] is "fold the next term x into the running accumulator a". It is
      left as a variable so the exact and the rounding accumulate are the SAME
      schedule rather than two schedules that might differ elsewhere. *)
  Variable op : Z -> Z -> Z.

  (** The band partial: fold f over [lo, lo+len) in index order, from 0. *)
  Fixpoint acc_range (f : nat -> Z) (lo len : nat) : Z :=
    match len with
    | O => 0
    | S n => op (acc_range f lo n) (f (lo + n)%nat)
    end.

  (** The reduction: fold the first [t] bands' partials together, from 0.
      This is [reduce.head] in the emitted module. *)
  Fixpoint acc_bands (f : nat -> Z) (K nthr t : nat) : Z :=
    match t with
    | O => 0
    | S t' => op (acc_bands f K nthr t') (acc_range f (boff K nthr t') (blen K nthr t'))
    end.
End Schedule.

(** Splitting a range is where associativity is spent, and it is the only
    place. Everything above is index arithmetic that holds for any [op]. *)
Lemma sum_range_split : forall f lo a b,
  acc_range Z.add f lo (a + b)%nat
  = acc_range Z.add f lo a + acc_range Z.add f (lo + a)%nat b.
Proof.
  intros f lo a b. induction b as [| b IH]; simpl.
  - rewrite Nat.add_0_r. ring.
  - rewrite Nat.add_succ_r. simpl. rewrite IH.
    replace (lo + (a + b))%nat with (lo + a + b)%nat by lia. ring.
Qed.

Lemma acc_bands_prefix : forall f K nthr t,
  acc_bands Z.add f K nthr t = acc_range Z.add f 0 (boff K nthr t).
Proof.
  intros f K nthr t. induction t as [| t IH].
  - cbn. reflexivity.
  - cbn [acc_bands]. rewrite IH, bands_contiguous, sum_range_split. reflexivity.
Qed.

(** ** The theorem.

    Splitting K across any number of workers and summing their partials gives
    the naive sum over the whole of K. Note what it is NOT quantified over:
    there is no hypothesis about K being divisible by nthr, none about the
    bands being equal, and none about how many workers there are beyond one. *)
Theorem ksplit_exact : forall f K nthr,
  (0 < nthr)%nat ->
  acc_bands Z.add f K nthr nthr = acc_range Z.add f 0 K.
Proof.
  intros f K nthr Hn. rewrite acc_bands_prefix, bands_tile by exact Hn. reflexivity.
Qed.

(** The property [tests/exact_gemm_thread_invariance.rs] asserts, derived
    rather than assumed - and note it says nothing about threads. Two thread
    counts agree because each separately equals the naive nest. *)
Corollary any_thread_count_agrees : forall f K n1 n2,
  (0 < n1)%nat -> (0 < n2)%nat ->
  acc_bands Z.add f K n1 n1 = acc_bands Z.add f K n2 n2.
Proof.
  intros f K n1 n2 H1 H2.
  rewrite (ksplit_exact f K n1 H1), (ksplit_exact f K n2 H2). reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Refutation 1: the mutation that drops the remainder             *)
(* ------------------------------------------------------------------ *)

(** Run by hand against the real kernel while building Phase 0 ("drop the
    remainder distribution"), where it failed against the integer reference.
    Here it is refuted for every K and nthr at once by failing [bands_tile] -
    and concretely below, so the file contains a counterexample and not only
    an unproved goal. *)

Definition blen_flat (K nthr : nat) (_ : nat) : nat := (K / nthr)%nat.

Fixpoint boff_flat (K nthr t : nat) : nat :=
  match t with
  | O => O
  | S t' => (boff_flat K nthr t' + blen_flat K nthr t')%nat
  end.

Fixpoint acc_bands_flat (f : nat -> Z) (K nthr t : nat) : Z :=
  match t with
  | O => 0
  | S t' => acc_bands_flat f K nthr t'
            + acc_range Z.add f (boff_flat K nthr t') (blen_flat K nthr t')
  end.

Definition ones (_ : nat) : Z := 1.

Theorem dropping_the_remainder_loses_terms :
  acc_bands_flat ones 3 2 2 <> acc_range Z.add ones 0 3.
Proof. vm_compute. discriminate. Qed.

(* ------------------------------------------------------------------ *)
(** ** Refutation 2: the same schedule under a rounding accumulate     *)
(* ------------------------------------------------------------------ *)

(** [rnd] keeps four significant decimal digits: below 1000 a value is exact,
    at or above it the last two digits are truncated. That is a crude model of
    a float's fixed mantissa and it is not IEEE - it is here to exhibit the
    ONE property that matters, that the accumulate is not associative, without
    dragging in a floating-point formalisation.

    [spike] is a single large term followed by two hundred ones. Folded left,
    each 1 is rounded away against the running 1000. Split in two, the second
    band's hundred ones survive as 100 and then land on the accumulator all at
    once, where they are large enough to be kept. *)

Definition rnd (x : Z) : Z := if Z.ltb x 1000 then x else 100 * (x / 100).
Definition fadd (a x : Z) : Z := rnd (a + x).
Definition spike (k : nat) : Z := if Nat.eqb k 0 then 1000 else 1.

(** 1100 against 1000: the split does not merely lose precision, it disagrees
    with its own reference. No tolerance makes this a proof of anything. *)
Theorem rounding_breaks_the_split :
  acc_bands fadd spike 201 2 2 <> acc_range fadd spike 0 201.
Proof. vm_compute. discriminate. Qed.

(** Same f, same K, same nthr, exact accumulate: 1200 either way. This pair is
    the roadmap's section 2 in two lines. *)
Theorem exact_survives_the_same_split :
  acc_bands Z.add spike 201 2 2 = acc_range Z.add spike 0 201.
Proof. vm_compute. reflexivity. Qed.

(** A control on the pair above: the two accumulates really do disagree on
    this input, so [exact_survives_the_same_split] is not passing because
    [fadd] happens to equal [Z.add] here. *)
Theorem the_two_accumulates_differ_on_this_input :
  acc_range fadd spike 0 201 <> acc_range Z.add spike 0 201.
Proof. vm_compute. discriminate. Qed.

Print Assumptions bands_tile.
Print Assumptions ksplit_exact.
Print Assumptions any_thread_count_agrees.
Print Assumptions dropping_the_remainder_loses_terms.
Print Assumptions rounding_breaks_the_split.
Print Assumptions exact_survives_the_same_split.
