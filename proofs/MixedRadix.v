(** * Positional indices, proved once.

    An index in this compiler is almost always a positional number: a tile
    offset is `t*T + f`, a packed A slot is `2*i + h`, a packed B slot is
    `(j/16)*32 + (j mod 16)*2 + h`, an accumulator column is `16*v + l`, a
    linear address is `r*ldc + c`. Each carries the same three obligations -
    the map is injective, it lands inside the buffer, and it is onto, so no
    slot keeps a value from the previous tile.

    Those obligations were spread across three files. [ExactGemmTiling] and
    [ExactGemmPacking] each defined their own [quot_rem_unique], with nothing
    relating them and neither requiring the other; the surjectivity and range
    arguments around it were rewritten in [ExactGemmRegisterTile] as well.

    ** The local-copy argument was RIGHT about the lemma and wrong about what
    ** surrounds it.

    `ExactGemmPacking`'s comment says [quot_rem_unique] is "proved locally
    rather than imported: it is six lines, and a lemma cannot drift into being
    wrong the way a duplicated CONSTANT can - it is re-proved wherever it is
    stated." That reasoning holds: a duplicated *lemma* is checked by `coqc` at
    every copy, unlike a duplicated constant, which is the whole reason
    `ExactGemmSchedule.v` is generated.

    What it does not cover is the rest of the package. Uniqueness is six lines;
    ONTO is eleven in `ExactGemmTiling`, seven in `ExactGemmRegisterTile` and
    four in `ExactGemmPacking`, all the same argument, and the two-digit peel
    in `pack_b_slot_bijective` is sixteen. The thing that recurs is the
    bijection, not the lemma inside it.

    ** What is NOT claimed.

    Nothing here is about memory, a buffer, or a machine word: these are facts
    about [nat]. That the emitted address arithmetic computes [pack] is the tie
    each kernel makes for itself, through `Ix` for the CPU GEMM.

    Checked with Rocq 9.1:  coqc proofs/MixedRadix.v  *)

Require Import Arith Lia.

(* ------------------------------------------------------------------ *)
(** ** One digit                                                       *)
(* ------------------------------------------------------------------ *)

(** `pack B q r` is the index whose low digit is `r` in radix `B`. *)
Definition pack (B q r : nat) : nat := (q * B + r)%nat.
Definition hi (B s : nat) : nat := (s / B)%nat.
Definition lo (B s : nat) : nat := (s mod B)%nat.

(** Quotient and remainder are unique. This is the single copy. *)
Lemma quot_rem_unique : forall B q1 r1 q2 r2,
  (0 < B)%nat -> (r1 < B)%nat -> (r2 < B)%nat ->
  (q1 * B + r1 = q2 * B + r2)%nat -> q1 = q2 /\ r1 = r2.
Proof.
  intros B q1 r1 q2 r2 HB H1 H2 Heq.
  assert (E1 : ((q1 * B + r1) / B = q1)%nat).
  { rewrite Nat.div_add_l by lia. rewrite Nat.div_small by lia. lia. }
  assert (E2 : ((q2 * B + r2) / B = q2)%nat).
  { rewrite Nat.div_add_l by lia. rewrite Nat.div_small by lia. lia. }
  assert (q1 = q2) by (rewrite <- E1, <- E2, Heq; reflexivity).
  split; [assumption | subst; lia].
Qed.

(** The two legs invert [pack] - the digits come back out. *)
Theorem pack_unpack : forall B q r,
  (0 < B)%nat -> (r < B)%nat -> hi B (pack B q r) = q /\ lo B (pack B q r) = r.
Proof.
  intros B q r HB Hr. unfold hi, lo, pack. split.
  - rewrite Nat.div_add_l by lia. rewrite Nat.div_small by lia. lia.
  - rewrite Nat.Div0.add_mod by lia. rewrite Nat.Div0.mod_mul by lia.
    rewrite Nat.add_0_l. rewrite Nat.Div0.mod_mod by lia.
    apply Nat.mod_small. exact Hr.
Qed.

(** ...and [pack] inverts them, so the correspondence is two-sided. *)
Theorem unpack_pack : forall B s, (0 < B)%nat -> pack B (hi B s) (lo B s) = s.
Proof.
  intros B s HB. unfold pack, hi, lo.
  rewrite Nat.mul_comm. symmetry. apply Nat.div_mod_eq.
Qed.

(** **In range.** A digit pair inside its bounds lands inside the buffer. *)
Theorem pack_in_range : forall B Q q r,
  (q < Q)%nat -> (r < B)%nat -> (pack B q r < Q * B)%nat.
Proof. intros B Q q r Hq Hr. unfold pack. nia. Qed.

(** **Onto.** Every position of the buffer is some digit pair, so nothing in it
    keeps a value from a previous tile. This is the leg that was rewritten in
    three files. *)
Theorem pack_onto : forall B Q s,
  (0 < B)%nat -> (s < Q * B)%nat ->
  exists q r, (q < Q)%nat /\ (r < B)%nat /\ pack B q r = s.
Proof.
  intros B Q s HB Hs. exists (hi B s), (lo B s).
  pose proof (Nat.div_mod_eq s B) as Hdm.
  pose proof (Nat.mod_upper_bound s B ltac:(lia)) as Hub.
  unfold hi, lo, pack. repeat split.
  - apply Nat.Div0.div_lt_upper_bound. lia.
  - exact Hub.
  - lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Two digits                                                      *)
(* ------------------------------------------------------------------ *)

(** `q*(B1*B0) + m*B0 + r`: the shape the packed B slot has, where the emitted
    map splits a column into a vector group and a lane before interleaving the
    k-halves. Peeled one digit at a time, which is what the sixteen lines in
    `pack_b_slot_bijective` were doing by hand. *)
Theorem two_digit_unique : forall B0 B1 q1 m1 r1 q2 m2 r2,
  (0 < B0)%nat -> (0 < B1)%nat ->
  (m1 < B1)%nat -> (m2 < B1)%nat -> (r1 < B0)%nat -> (r2 < B0)%nat ->
  (q1 * (B1 * B0) + m1 * B0 + r1 = q2 * (B1 * B0) + m2 * B0 + r2)%nat ->
  q1 = q2 /\ m1 = m2 /\ r1 = r2.
Proof.
  intros B0 B1 q1 m1 r1 q2 m2 r2 HB0 HB1 Hm1 Hm2 Hr1 Hr2 Heq.
  assert (Hhi : q1 = q2 /\ (m1 * B0 + r1 = m2 * B0 + r2)%nat).
  { apply (quot_rem_unique (B1 * B0)); [ nia | nia | nia | lia ]. }
  destruct Hhi as [Hq Hrest].
  assert (Hlo : m1 = m2 /\ r1 = r2) by (apply (quot_rem_unique B0); lia).
  destruct Hlo as [Hm Hr]. auto.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The bound on the LOW digit is what makes it a bijection          *)
(* ------------------------------------------------------------------ *)

(** Without `r < B` the map is not injective, and this says so by REFUTING the
    weakened theorem rather than by exhibiting a collision.

    The first version of this exhibited one - `0*2 + 2 = 1*2 + 0` - and a
    mutation moved the fixture to a legal digit pair, where the statement is
    still true and still proves, leaving every test green. That is the second
    time in this development a control has turned out to be satisfied by an
    uninteresting instance of itself. **A control has to state what makes the
    case interesting as a PROPOSITION**; here, that dropping the hypothesis
    makes the theorem false, which no choice of witness can satisfy vacuously. *)
Theorem the_low_digit_bound_cannot_be_dropped :
  ~ (forall B q1 r1 q2 r2,
        (0 < B)%nat ->
        (q1 * B + r1 = q2 * B + r2)%nat -> q1 = q2 /\ r1 = r2).
Proof.
  intros H. destruct (H 2 0 2 1 0 ltac:(lia) ltac:(reflexivity)) as [Hq _].
  discriminate.
Qed.

(** ...and the collision it rests on, kept because it names the smallest case
    where the bound bites. *)
Theorem without_the_digit_bound_it_is_not_injective :
  pack 2 0 2 = pack 2 1 0 /\ (0 <> 1)%nat.
Proof. unfold pack. split; [ reflexivity | discriminate ]. Qed.

(** The bound is on the LOW digit ONLY: the high one may be anything, which is
    why [pack_unpack] needs no hypothesis about `q` and why a packed panel can
    be addressed before its group count is known. *)
Theorem the_high_digit_needs_no_bound : forall B q r,
  (0 < B)%nat -> (r < B)%nat -> hi B (pack B q r) = q.
Proof. intros B q r HB Hr. now apply pack_unpack. Qed.

Print Assumptions quot_rem_unique.
Print Assumptions pack_unpack.
Print Assumptions unpack_pack.
Print Assumptions pack_in_range.
Print Assumptions pack_onto.
Print Assumptions two_digit_unique.
Print Assumptions the_low_digit_bound_cannot_be_dropped.
Print Assumptions without_the_digit_bound_it_is_not_injective.
Print Assumptions the_high_digit_needs_no_bound.
