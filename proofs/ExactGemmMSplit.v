(** * The exact GEMM's M-split schedule, and why it is a different KIND of
      parallelisation from the K-split.

    [ExactGemmKsplit.v] proves the shipped wrapper's contraction split correct.
    Reading its own performance back found that the split it proves is the
    expensive one: cutting K is a REDUCTION, so every thread needs a private
    [M x N] C to sum out of, and that bookkeeping is [O(T * M * N)] against
    [O(M * N * K)] of work. Measured at 1024x1024x1024 on 8 threads it cost
    9.311 ms against 7.344 ms of compute - more than the entire GEMM - and 8
    threads ran SLOWER than one.

    Cutting M instead is a PARTITION: thread [t] owns rows
    [[boff M mthr t, boff M mthr t + blen M mthr t)] of C and computes them over
    the whole of K. Nothing is shared, so nothing is summed.

    *** What is proved. ***

    - [rows_tile] : the row bands cover [0, M) exactly. It is [bands_tile] at a
      different extent - the SAME decomposition, which is the point: the
      emitter renders both from one [Ix] description.
    - [owner_unique] : every row belongs to exactly one band. This is the
      obligation a partition has and a reduction does not, and it is what says
      no two threads write the same element of C.
    - [msplit_exact] : the M-split's result is the naive nest's, for every
      [f], every [M] and every positive [mthr].

    *** The theorem that makes this worth a file: [msplit_needs_no_algebra]. ***

    [ksplit_exact] is stated over an arbitrary [op] and then USES associativity
    to move a band boundary; [rounding_breaks_the_split] exhibits an [op] for
    which it fails. [msplit_exact] uses no property of the accumulate at all -
    not associativity, not commutativity, not exactness. A row's accumulator is
    never split, so there is no re-bracketing to justify.

    [an_msplit_survives_the_accumulate_that_breaks_a_ksplit] is the two stated
    against each other: the SAME rounding [op], the same data, the same thread
    count. The K-split answers 1100 where its own naive nest answers 1000; the
    M-split answers exactly what its naive nest answers.

    That is a statement about the PROGRAMME, not only about this kernel. Section
    2 of [docs/proof_carrying_kernels.md] argues that exact accumulation is what
    makes a tiled kernel's equivalence provable. True, and incomplete: it is
    what makes a REDUCTION-shaped parallelisation provable. A partition-shaped
    one is provable without it. Exactness buys the axes you could not otherwise
    cut - it is not the price of cutting anything.

    *** What this does NOT prove. ***

    A proof about a MODEL, as the rest of the series is. A band's contribution
    is [row_naive], the value the naive nest computes for that row; what fills
    it - packing, the register tile, the masked tails, the int32 flush - is
    proved in the sibling files and joined in [ExactGemmChain.v]. Nothing here
    is about memory, threads, or the happens-before edge [pthread_join]
    establishes; that is [tests/exact_gemm_thread_sanitizer.rs].

    Nor does it say the M-split is always the better choice. It is not: below
    two register tiles per thread the bands stop filling the micro-kernel, and
    the emitter takes M only when it saturates the request. That crossover is
    measured, not proved - [cpu_gemm::split_axis] carries the numbers.

    Build: coqc -Q . "" proofs/ExactGemmMSplit.v   (Rocq 9.1) *)

Require Import Coq.ZArith.ZArith.
Require Import Coq.Arith.Arith.
Require Import Coq.micromega.Lia.
Require ExactGemmSchedule.
Require ExactGemmKsplit.

Module SCH := ExactGemmSchedule.
Module KS := ExactGemmKsplit.

Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** The row bands are the SAME decomposition, at a different extent *)
(* ------------------------------------------------------------------ *)

(** The emitter renders [%mbase]/[%mrem]/[%mlen] from [band_base_ix],
    [band_rem_ix] and [band_len_ix] - the identical [Ix] expressions the
    K-split's [%base]/[%rem]/[%klen] come from, bound to [M] and [mthr]
    instead of [K] and [nthr]. So these are not new definitions. *)
Definition rlen (M mthr t : nat) : nat := SCH.blen M mthr t.
Definition roff (M mthr t : nat) : nat := SCH.boff M mthr t.

Lemma rlen_is_blen : forall M mthr t, rlen M mthr t = KS.blen M mthr t.
Proof. reflexivity. Qed.

Lemma roff_is_boff : forall M mthr t, roff M mthr t = KS.boff M mthr t.
Proof. reflexivity. Qed.

(** The bands end exactly at M. Inherited, not re-proved. *)
Theorem rows_tile : forall M mthr,
  (0 < mthr)%nat -> roff M mthr mthr = M.
Proof. intros. apply KS.bands_tile. assumption. Qed.

Lemma roff_step : forall M mthr t,
  roff M mthr (S t) = (roff M mthr t + rlen M mthr t)%nat.
Proof. reflexivity. Qed.

Lemma roff_zero : forall M mthr, roff M mthr 0 = 0%nat.
Proof. reflexivity. Qed.

Lemma roff_monotone : forall M mthr t,
  (roff M mthr t <= roff M mthr (S t))%nat.
Proof. intros. rewrite roff_step. lia. Qed.

Lemma roff_le : forall M mthr t u,
  (t <= u)%nat -> (roff M mthr t <= roff M mthr u)%nat.
Proof.
  intros M mthr t u H. induction H as [| u _ IH].
  - lia.
  - eapply Nat.le_trans; [ exact IH | apply roff_monotone ].
Qed.

(* ------------------------------------------------------------------ *)
(** ** Every row belongs to exactly one band                           *)
(* ------------------------------------------------------------------ *)

(** The obligation a PARTITION carries and a reduction does not. A K-band's
    partials are summed, so two bands overlapping would double-count and the
    coverage theorem alone would not notice; here two bands overlapping means
    two threads writing the same row of C, which no arithmetic can repair. *)
Definition owns (M mthr t r : nat) : Prop :=
  (roff M mthr t <= r)%nat /\ (r < roff M mthr t + rlen M mthr t)%nat.

Theorem owner_unique : forall M mthr t u r,
  owns M mthr t r -> owns M mthr u r -> t = u.
Proof.
  intros M mthr t u r [Ht1 Ht2] [Hu1 Hu2].
  destruct (Nat.lt_trichotomy t u) as [Hlt | [He | Hgt]]; [| exact He |].
  - assert (S t <= u)%nat by lia.
    pose proof (roff_le M mthr (S t) u H) as Hle.
    rewrite roff_step in Hle. lia.
  - assert (S u <= t)%nat by lia.
    pose proof (roff_le M mthr (S u) t H) as Hle.
    rewrite roff_step in Hle. lia.
Qed.

(** ...and every row in range belongs to SOME band. Coverage and uniqueness
    together are what "partition" means; either alone is satisfied by a
    schedule that drops rows or by one that writes them twice. *)
Theorem owner_exists : forall M mthr r,
  (0 < mthr)%nat -> (r < M)%nat -> exists t, (t < mthr)%nat /\ owns M mthr t r.
Proof.
  intros M mthr r Hm Hr.
  assert (Hcover : forall n, (n <= mthr)%nat ->
            (r < roff M mthr n)%nat -> exists t, (t < n)%nat /\ owns M mthr t r).
  { induction n as [| n IH]; intros Hn Hlt.
    - rewrite roff_zero in Hlt. lia.
    - rewrite roff_step in Hlt.
      destruct (Nat.ltb_spec r (roff M mthr n)) as [Hin | Hout].
      + destruct (IH ltac:(lia) Hin) as [t [Ht Ho]]. exists t. split; [lia | exact Ho].
      + exists n. split; [lia |]. unfold owns. split; lia. }
  apply Hcover; [lia |]. rewrite rows_tile by assumption. exact Hr.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The schedule, over an arbitrary accumulate                      *)
(* ------------------------------------------------------------------ *)

Section Schedule.
  (** [op] is left a variable, exactly as in [ExactGemmKsplit.v], so that the
      exact and the rounding accumulate are the same theorem at two
      instantiations rather than two theorems. The difference is that NOTHING
      below assumes anything about it. *)
  Variable op : Z -> Z -> Z.
  Variable seed : Z.

  (** What the naive nest computes for one row: fold the whole of K into a
      fresh accumulator. [f r k] is the term for row [r], index [k]. *)
  Fixpoint row_acc (f : nat -> Z) (n : nat) : Z :=
    match n with
    | O => seed
    | S n' => op (row_acc f n') (f n')
    end.

  Definition row_naive (f : nat -> nat -> Z) (K r : nat) : Z := row_acc (f r) K.

  (** What the M-split computes for row [r]: the band that owns it runs the
      same fold over the same K. The row index is the only thing that moves. *)
  Definition msplit_row (f : nat -> nat -> Z) (M mthr K t i : nat) : Z :=
    row_naive f K (roff M mthr t + i).

  (** The schedule is exact for EVERY [op]. There is no re-bracketing to
      justify, because a row's accumulator is never split. *)
  Theorem msplit_exact : forall f M mthr K t i,
    msplit_row f M mthr K t i = row_naive f K (roff M mthr t + i).
  Proof. reflexivity. Qed.

  (** The corollary the emitted wrapper relies on: whatever thread count is
      chosen, the value written to row [r] is the naive nest's value for row
      [r] - because the band that owns [r] is the only one that writes it and
      it computes exactly that. *)
  Theorem every_row_gets_its_naive_value : forall f M mthr K r,
    (0 < mthr)%nat -> (r < M)%nat ->
    exists t, (t < mthr)%nat /\ owns M mthr t r /\
      msplit_row f M mthr K t (r - roff M mthr t) = row_naive f K r.
  Proof.
    intros f M mthr K r Hm Hr.
    destruct (owner_exists M mthr r Hm Hr) as [t [Ht Ho]].
    exists t. split; [exact Ht |]. split; [exact Ho |].
    unfold msplit_row. destruct Ho as [H1 _].
    replace (roff M mthr t + (r - roff M mthr t))%nat with r by lia.
    reflexivity.
  Qed.

  (** Two thread counts agree, which is what the behavioural test asserts. It
      says nothing about threads: both equal the naive nest. *)
  Theorem any_thread_count_agrees : forall f M m1 m2 K r,
    (0 < m1)%nat -> (0 < m2)%nat -> (r < M)%nat ->
    (exists t1, owns M m1 t1 r /\ msplit_row f M m1 K t1 (r - roff M m1 t1) = row_naive f K r) /\
    (exists t2, owns M m2 t2 r /\ msplit_row f M m2 K t2 (r - roff M m2 t2) = row_naive f K r).
  Proof.
    intros f M m1 m2 K r H1 H2 Hr. split.
    - destruct (every_row_gets_its_naive_value f M m1 K r H1 Hr) as [t [_ [Ho He]]].
      exists t. split; assumption.
    - destruct (every_row_gets_its_naive_value f M m2 K r H2 Hr) as [t [_ [Ho He]]].
      exists t. split; assumption.
  Qed.
End Schedule.

(** The statement above, made explicit: [msplit_exact] holds for an [op] about
    which literally nothing is assumed. Compare [ExactGemmKsplit.ksplit_exact],
    whose proof consumes associativity. *)
Theorem msplit_needs_no_algebra :
  forall (op : Z -> Z -> Z) (seed : Z) f M mthr K t i,
    msplit_row op seed f M mthr K t i = row_naive op seed f K (roff M mthr t + i).
Proof. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** The refutation: the accumulate that breaks a K-split does not
       break an M-split                                                *)
(* ------------------------------------------------------------------ *)

(** [Decomposition.rnd] is the rounding accumulate the series already uses to
    refute the K-split: it keeps values below 1000 and rounds everything else
    to a multiple of 100, which is the textbook loss of small terms to a large
    accumulator. *)
Definition rnd (x : Z) : Z := if Z.ltb x 1000 then x else 100 * (x / 100).
Definition fadd (a x : Z) : Z := rnd (a + x).

(** One row, whose terms are a big one followed by small ones. *)
Definition spike (k : nat) : Z := if Nat.eqb k 0 then 1000 else 1.
Definition two_rows (r k : nat) : Z := spike k.

(** Under the SAME rounding accumulate, the M-split's answer for a row is the
    naive nest's answer for that row, at EVERY thread count - and the values
    are computed, not asserted. Under a K-split of the same 201 terms into two
    bands, [ExactGemmKsplit.rounding_breaks_the_split] gets 1100 against its
    own naive 1000. Same accumulate, same data, opposite outcome; the only
    difference is which axis was cut. *)
Theorem an_msplit_survives_the_accumulate_that_breaks_a_ksplit :
  msplit_row fadd 0 two_rows 4%nat 2%nat 201%nat 0%nat 0%nat = 1000
  /\ msplit_row fadd 0 two_rows 4%nat 4%nat 201%nat 0%nat 0%nat = 1000
  /\ row_naive fadd 0 two_rows 201%nat 0%nat = 1000.
Proof. repeat split; vm_compute; reflexivity. Qed.

(** ...and the controls that stop the theorem above being vacuous. The first
    says this accumulate really does lose a small term to a large accumulator.
    The second is the property [ExactGemmKsplit.ksplit_exact] CONSUMES and this
    file does not: re-bracketing changes the answer.

    The obvious witness for the second is false and was checked rather than
    assumed - [rnd] is the identity on multiples of 100, so
    [fadd (fadd 1000 1) 1] and [fadd 1000 2] are both 1000. A real witness
    needs two terms that cross a rounding boundary together but not
    separately: 50 and 60 sum to 110, which carries, while each alone does
    not. *)
Theorem the_accumulate_really_does_round :
  fadd 1000 1 <> 1000 + 1.
Proof. compute. discriminate. Qed.

Theorem the_rounding_accumulate_is_not_associative :
  fadd (fadd 1000 50) 60 <> fadd 1000 (50 + 60).
Proof. compute. discriminate. Qed.

(** A non-vacuity control for the partition itself: at M = 4 over 2 bands the
    two bands really are different and really do cover the rows. A schedule
    that put every row in band 0 would satisfy [owner_unique] trivially. *)
Theorem the_bands_are_not_all_one :
  roff 4 2 0 = 0%nat /\ roff 4 2 1 = 2%nat /\ roff 4 2 2 = 4%nat.
Proof. repeat split; reflexivity. Qed.

Theorem a_ragged_extent_still_tiles :
  roff 53 4 4 = 53%nat /\ rlen 53 4 0 = 14%nat /\ rlen 53 4 3 = 13%nat.
Proof. repeat split; reflexivity. Qed.

Print Assumptions rows_tile.
Print Assumptions owner_unique.
Print Assumptions owner_exists.
Print Assumptions msplit_exact.
Print Assumptions every_row_gets_its_naive_value.
Print Assumptions any_thread_count_agrees.
Print Assumptions msplit_needs_no_algebra.
Print Assumptions an_msplit_survives_the_accumulate_that_breaks_a_ksplit.
Print Assumptions the_rounding_accumulate_is_not_associative.
Print Assumptions the_bands_are_not_all_one.
Print Assumptions a_ragged_extent_still_tiles.
