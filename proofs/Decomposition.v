(** * The partition obligation, proved once.

    Three kernels in this repo split a reduction across workers, and each was
    proved on its own:

      - [ExactGemmKsplit]   contiguous bands, the first `K mod nthr` one longer
      - [GemmBandSplit]     contiguous bands at proportional edges `t*ext/n`,
                            and a second family snapped to a granularity
      - [GridStrideSplit]   residue classes `{ i : i mod n = w }`, interleaved

    Read side by side they prove the same theorem three times: edges at 0 and
    at N, monotone, the parts tile `[0, N)`, a prefix lemma, and finally
    "the partials combine to the naive fold". `docs/proof_carrying_kernels.md`
    names this as Phase 2's research risk - *if obligations don't compose, the
    thing is a one-off proof rather than a compiler* - and the question is
    answerable from what is already here, with no new kernel and no IR.

    This file is the schema. Each kernel keeps its own theorem STATEMENT
    unchanged and supplies only what is peculiar to its decomposition; nothing
    below mentions a kernel, a tile, a thread or a GPU.

    ** The schema is TWO theorems, and which one a kernel gets is decided by
    ** whether its parts are contiguous.

    [contiguous_exact] spends associativity and nothing else: the parts are
    consecutive stretches of `[0, N)`, so folding them left to right is a
    re-bracketing of the naive fold. Both GEMM splits are instances.

    [decomposition_exact] takes an arbitrary [owner] map, so the parts
    interleave and the terms genuinely change order. That needs commutativity.
    The grid-stride split is an instance, and it is the only one - which is the
    same distinction `GridStrideSplit` records about its atomics, stated here
    as a property of the DECOMPOSITION rather than of the hardware.

    Neither is derived from the other. The general form would give the
    contiguous one only by inverting the edge function, and it would then
    demand commutativity that the contiguous case does not need - so a single
    theorem would be a WEAKER result presented as a simpler one.

    ** What is NOT claimed.

    The operator is `Z.add` in every theorem, so nothing here says anything
    about a machine word: the width obligations live with the kernels
    ([the_bound_is_one_unit_wide], [operand_bound_gives_no_overflow]) and are
    discharged against real types. And a decomposition is modelled as an index
    map; that the EMITTED code walks that map is the tie each kernel makes for
    itself, by byte-identity through `Ix` for the two GEMMs and by reading the
    emitted PTX for attention.

    Checked with Rocq 9.1:  coqc proofs/Decomposition.v  *)

Require Import ZArith Arith Lia.
Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** The naive fold                                                  *)
(* ------------------------------------------------------------------ *)

Section Fold.
  Variable op : Z -> Z -> Z.

  (** Every term of `[lo, lo+len)`, in index order. This is what a
      decomposition has to reproduce, and it is not a property of any
      decomposition - which is why it sat, verbatim, in three files. *)
  Fixpoint acc_range (f : nat -> Z) (lo len : nat) : Z :=
    match len with
    | O => 0
    | S n => op (acc_range f lo n) (f (lo + n)%nat)
    end.
End Fold.

(** Splitting a range is where associativity is spent, and it is the only
    place in the contiguous half of this file. *)
Lemma sum_range_split : forall f lo a b,
  acc_range Z.add f lo (a + b)%nat
  = acc_range Z.add f lo a + acc_range Z.add f (lo + a)%nat b.
Proof.
  intros f lo a b. induction b as [| b IH]; simpl.
  - rewrite Nat.add_0_r. ring.
  - rewrite Nat.add_succ_r. simpl. rewrite IH.
    replace (lo + (a + b))%nat with (lo + a + b)%nat by lia. ring.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Contiguous parts: associativity only                            *)
(* ------------------------------------------------------------------ *)

Section Contiguous.
  Variable op : Z -> Z -> Z.
  (** The decomposition, given by its boundaries: part `t` is
      `[edge t, edge (S t))`. A kernel supplies this and three facts about it. *)
  Variable edge : nat -> nat.

  (** Fold the first [t] parts' partials together, from 0 - the shape every
      band reduction in this repo is emitted in. *)
  Fixpoint acc_parts (f : nat -> Z) (t : nat) : Z :=
    match t with
    | O => 0
    | S t' => op (acc_parts f t') (acc_range op f (edge t') (edge (S t') - edge t'))
    end.
End Contiguous.

(** The prefix lemma: the first [t] parts cover `[0, edge t)`. *)
Lemma acc_parts_prefix : forall edge f t,
  edge 0%nat = 0%nat ->
  (forall u, (edge u <= edge (S u))%nat) ->
  acc_parts Z.add edge f t = acc_range Z.add f 0 (edge t).
Proof.
  intros edge f t Hz Hm. induction t as [| t IH].
  - cbn [acc_parts]. rewrite Hz. cbn [acc_range]. reflexivity.
  - cbn [acc_parts]. rewrite IH.
    specialize (Hm t).
    replace (edge (S t)) with (edge t + (edge (S t) - edge t))%nat by lia.
    rewrite sum_range_split. rewrite Nat.add_0_l.
    replace (edge t + (edge (S t) - edge t) - edge t)%nat
       with (edge (S t) - edge t)%nat by lia.
    reflexivity.
Qed.

(** **The contiguous theorem.** A decomposition whose boundaries start at 0,
    end at N, and never go backwards reproduces the naive fold exactly - with
    no hypothesis that the parts are equal, that `n` divides `N`, or that `N`
    is positive.

    Associativity is the whole cost. Commutativity is never used: the parts are
    consecutive, so this is a re-bracketing and not a reordering. *)
Theorem contiguous_exact : forall edge f n N,
  edge 0%nat = 0%nat ->
  edge n = N ->
  (forall u, (edge u <= edge (S u))%nat) ->
  acc_parts Z.add edge f n = acc_range Z.add f 0 N.
Proof.
  intros edge f n N Hz Hn Hm.
  rewrite acc_parts_prefix by assumption. rewrite Hn. reflexivity.
Qed.

(** The worker count is invisible in the answer - which is what makes a
    schedule bug undetectable from the result, and so is the reason the
    kernels need a separate behavioural tie. *)
Corollary contiguous_count_agrees : forall e1 e2 f n1 n2 N,
  e1 0%nat = 0%nat -> e1 n1 = N -> (forall u, (e1 u <= e1 (S u))%nat) ->
  e2 0%nat = 0%nat -> e2 n2 = N -> (forall u, (e2 u <= e2 (S u))%nat) ->
  acc_parts Z.add e1 f n1 = acc_parts Z.add e2 f n2.
Proof.
  intros e1 e2 f n1 n2 N Z1 L1 M1 Z2 L2 M2.
  rewrite (contiguous_exact e1 f n1 N Z1 L1 M1).
  rewrite (contiguous_exact e2 f n2 N Z2 L2 M2). reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The CLAMPED edge family, which two obligations share            *)
(* ------------------------------------------------------------------ *)

(** Cut `[0, ext)` into pieces of `T`, with the last one short:

      edge t = min(t * T, ext)

    The emitter writes this twice, in two spellings, for two different reasons -
    `min(iv + T, ext)` for the int32 flush interval (an OVERFLOW budget) and
    `min(ext - iv, T)` for the output tile width (a MEMORY partition). Neither
    knows about the other; `ExactGemmMicro` and `ExactGemmTiling` each developed
    it from scratch. It is one family. *)
Definition clamped (T ext t : nat) : nat := Nat.min (t * T) ext.

Lemma clamped_zero : forall T ext, clamped T ext 0 = 0%nat.
Proof. intros T ext. unfold clamped. rewrite Nat.mul_0_l. apply Nat.min_0_l. Qed.

Lemma clamped_monotone : forall T ext t,
  (clamped T ext t <= clamped T ext (S t))%nat.
Proof. intros T ext t. unfold clamped. apply Nat.min_le_compat_r. nia. Qed.

(** The last edge reaches the extent exactly when the piece count spans it,
    which is what `ntiles` and `nchunks` are both `ceil` for. *)
Lemma clamped_last : forall T ext n,
  (ext <= n * T)%nat -> clamped T ext n = ext.
Proof. intros T ext n H. unfold clamped. now apply Nat.min_r. Qed.

(** **The identity that ties both spellings to the schema.** A clamped edge
    function's WIDTH is the `min(ext - t*T, T)` form, so a file that computes
    widths directly and a file that computes ends are describing the same
    decomposition. Each carried its own three-way case split for this; here it
    is once. *)
(** The arithmetic underneath both, isolated so neither has to repeat it: a
    window `[a, a+w)` intersected with `[0, n)` has width `min(n - a, w)`. *)
Lemma min_shift : forall a w n, (Nat.min (a + w) n - a)%nat = Nat.min (n - a) w.
Proof.
  intros a w n.
  destruct (Nat.le_gt_cases n a) as [H1 | H1].
  - rewrite Nat.min_r by lia. replace (n - a)%nat with O by lia.
    rewrite Nat.min_0_l. lia.
  - destruct (Nat.le_gt_cases n (a + w)) as [H2 | H2].
    + rewrite Nat.min_r by lia. rewrite Nat.min_l by lia. reflexivity.
    + rewrite Nat.min_l by lia. rewrite Nat.min_r by lia. lia.
Qed.

Lemma clamped_width : forall T ext t,
  (clamped T ext (S t) - clamped T ext t)%nat = Nat.min (ext - t * T)%nat T.
Proof.
  intros T ext t. unfold clamped. rewrite Nat.mul_succ_l.
  destruct (Nat.le_gt_cases ext (t * T)) as [Hpast | Hlive].
  - rewrite (Nat.min_r (t * T) ext) by lia.
    rewrite (Nat.min_r (t * T + T) ext) by lia.
    replace (ext - t * T)%nat with 0%nat by lia.
    rewrite Nat.min_0_l. lia.
  - rewrite (Nat.min_l (t * T) ext) by lia. apply min_shift.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The same partition, counted in [nat]                            *)
(* ------------------------------------------------------------------ *)

(** [acc_parts] folds VALUES; an output tiling has no values to fold and asks
    instead that the part WIDTHS add up to the extent - no gap and no double
    count. Same decomposition, different consequence, and it needs no algebra
    at all: the sum telescopes. *)
Fixpoint width_sum (edge : nat -> nat) (t : nat) : nat :=
  match t with
  | O => O
  | S t' => (width_sum edge t' + (edge (S t') - edge t'))%nat
  end.

Lemma width_sum_closed : forall edge t,
  edge 0%nat = 0%nat ->
  (forall u, (edge u <= edge (S u))%nat) ->
  width_sum edge t = edge t.
Proof.
  intros edge t Hz Hm. induction t as [| t IH]; cbn [width_sum].
  - symmetry. exact Hz.
  - rewrite IH. specialize (Hm t). lia.
Qed.

Theorem widths_cover_the_extent : forall edge n N,
  edge 0%nat = 0%nat ->
  edge n = N ->
  (forall u, (edge u <= edge (S u))%nat) ->
  width_sum edge n = N.
Proof. intros edge n N Hz Hn Hm. rewrite width_sum_closed by assumption. exact Hn. Qed.

(* ------------------------------------------------------------------ *)
(** ** Arbitrary parts: commutativity as well                          *)
(* ------------------------------------------------------------------ *)

Section Owned.
  Variable op : Z -> Z -> Z.
  (** The decomposition, given by an assignment: index `i` belongs to worker
      `owner i`. Nothing says the parts are intervals, or even connected. *)
  Variable owner : nat -> nat.

  (** Worker `w`'s partial over the indices below `N` that it owns. *)
  Fixpoint part (f : nat -> Z) (w N : nat) : Z :=
    match N with
    | O => 0
    | S k => if Nat.eqb (owner k) w then op (part f w k) (f k) else part f w k
    end.

  (** The first [t] workers' partials, folded together. *)
  Fixpoint combine (f : nat -> Z) (N t : nat) : Z :=
    match t with
    | O => 0
    | S t' => op (combine f N t') (part f t' N)
    end.
End Owned.

(** Peeling the last index off hands one term to exactly one worker - the one
    numbered `owner N`. This single lemma is the partition; everything else is
    bookkeeping around it, and it is where commutativity is spent, because the
    term has to travel out of the middle of its own worker's partial. *)
Lemma combine_peel : forall owner f N t,
  combine Z.add owner f (S N) t
  = combine Z.add owner f N t + (if Nat.ltb (owner N) t then f N else 0).
Proof.
  intros owner f N t. induction t as [| t IH].
  - cbn. ring.
  - cbn [combine]. rewrite IH. cbn [part].
    destruct (Nat.eqb_spec (owner N) t);
      destruct (Nat.ltb_spec (owner N) t);
      destruct (Nat.ltb_spec (owner N) (S t)); try lia; ring.
Qed.

(** **The parts tile.** Every index below `N` is claimed by exactly one of the
    `n` workers, so peeling it off moves exactly one term. *)
Theorem parts_tile : forall owner f N n,
  (owner N < n)%nat ->
  combine Z.add owner f (S N) n = combine Z.add owner f N n + f N.
Proof.
  intros owner f N n H. rewrite combine_peel.
  destruct (Nat.ltb_spec (owner N) n); [ reflexivity | lia ].
Qed.

Lemma combine_empty : forall owner f t, combine Z.add owner f 0 t = 0.
Proof.
  intros owner f t. induction t as [| t IH]; cbn [combine]; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

(** **The general theorem.** Any assignment of indices to `n` workers
    reproduces the naive fold, whatever shape the parts are. The one
    hypothesis is that every index the fold covers is owned by a worker the
    combine visits. *)
Theorem decomposition_exact : forall owner f N n,
  (forall i, (i < N)%nat -> (owner i < n)%nat) ->
  combine Z.add owner f N n = acc_range Z.add f 0 N.
Proof.
  intros owner f N n H. induction N as [| N IH].
  - cbn [acc_range]. apply combine_empty.
  - cbn [acc_range]. rewrite parts_tile by (apply H; lia).
    rewrite IH by (intros i Hi; apply H; lia).
    rewrite Nat.add_0_l. reflexivity.
Qed.

Corollary decomposition_count_agrees : forall o1 o2 f N n1 n2,
  (forall i, (i < N)%nat -> (o1 i < n1)%nat) ->
  (forall i, (i < N)%nat -> (o2 i < n2)%nat) ->
  combine Z.add o1 f N n1 = combine Z.add o2 f N n2.
Proof.
  intros o1 o2 f N n1 n2 H1 H2.
  rewrite (decomposition_exact o1 f N n1 H1).
  rewrite (decomposition_exact o2 f N n2 H2). reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The refutation, stated once                                     *)
(* ------------------------------------------------------------------ *)

(** Every one of the three kernels carries its own version of this, and it is
    the same fact: the theorems above are about the ACCUMULATE, not about the
    indices. A rounding accumulate satisfies every partition property and
    still disagrees with itself across worker counts.

    `rnd` keeps small values and rounds larger ones to a multiple of 100 - a
    model of any accumulate that discards low bits, which is what a float does
    once the running sum outgrows the term being added. *)
Definition rnd (x : Z) : Z := if Z.ltb x 1000 then x else 100 * (x / 100).
Definition fadd (a x : Z) : Z := rnd (a + x).

(** A big first term the small ones vanish against - the textbook shape, and it
    has to be constructed: three plausible rounding models agreed on every
    small input tried before this one. *)
Definition spike (k : nat) : Z := if Nat.eqb k 0 then 1000 else 1.

(** Contiguous halves of `[0, 201)`. *)
Definition half (n : nat) : nat := match n with O => 0%nat | S O => 100%nat | _ => 201%nat end.

Theorem rounding_breaks_a_contiguous_split :
  acc_parts fadd half spike 2 <> acc_range fadd spike 0 201.
Proof. vm_compute. discriminate. Qed.

Theorem exact_survives_the_same_split :
  acc_parts Z.add half spike 2 = acc_range Z.add spike 0 201.
Proof. reflexivity. Qed.

(** And the same for an interleaved one, where the terms genuinely reorder. *)
Definition alternate (i : nat) : nat := (i mod 2)%nat.

Theorem rounding_breaks_an_interleaved_split :
  combine fadd alternate spike 201 2 <> acc_range fadd spike 0 201.
Proof. vm_compute. discriminate. Qed.

Theorem exact_survives_the_interleaved_split :
  combine Z.add alternate spike 201 2 = acc_range Z.add spike 0 201.
Proof. reflexivity. Qed.

(** The control. Without it the two refutations could be passing because the
    two accumulates agree on this input and the decomposition is what differs -
    they do not, and this is what says so. *)
Theorem the_two_accumulates_differ_on_this_input :
  acc_range fadd spike 0 201 <> acc_range Z.add spike 0 201.
Proof. vm_compute. discriminate. Qed.

(** The two theorems are genuinely different: [decomposition_exact] holds of a
    partition [contiguous_exact] cannot describe, because its parts are not
    intervals. Stated so that "just use the general one everywhere" is visibly
    not the same claim - the general one is about arbitrary parts and pays
    commutativity for it. *)
Theorem the_interleaved_split_is_not_contiguous :
  (alternate 0 = alternate 2)%nat /\ (alternate 0 <> alternate 1)%nat.
Proof. cbn. split; [ reflexivity | discriminate ]. Qed.

Print Assumptions sum_range_split.
Print Assumptions min_shift.
Print Assumptions clamped_width.
Print Assumptions widths_cover_the_extent.
Print Assumptions acc_parts_prefix.
Print Assumptions contiguous_exact.
Print Assumptions contiguous_count_agrees.
Print Assumptions combine_peel.
Print Assumptions parts_tile.
Print Assumptions decomposition_exact.
Print Assumptions decomposition_count_agrees.
Print Assumptions rounding_breaks_a_contiguous_split.
Print Assumptions exact_survives_the_same_split.
Print Assumptions rounding_breaks_an_interleaved_split.
Print Assumptions exact_survives_the_interleaved_split.
Print Assumptions the_two_accumulates_differ_on_this_input.
Print Assumptions the_interleaved_split_is_not_contiguous.
