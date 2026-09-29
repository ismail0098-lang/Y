(** * The placement obligation: a decomposition that TILES rather than FOLDS.

    `Decomposition.v` proves the partition obligation once and five files
    instantiate it. Its own header names the limit, and
    `docs/proof_carrying_kernels.md` repeats it: every one of those five is a
    REDUCTION. The parts fold into a value, the theorem is "the partials
    combine to the naive fold", and the two schema theorems differ only in
    whether that fold spends associativity or commutativity as well. Two of
    the kernels are GEMMs and the third is a softmax, so the axes coincide;
    nothing had tested whether the schema describes a decomposition of a
    different KIND.

    This file is that test, and the subject is not a synthetic one. The MSM
    binning in `tests/common/msm.rs` is a parallel COUNTING SORT: per-chunk
    histograms, an exclusive prefix over buckets, and an unsynchronised
    scatter through a private cursor per (writer, bucket). Its parts do not
    fold. They TILE a destination array, and the obligation is that every slot
    of `Idx` is written by exactly one thread, exactly once, in bounds.

    ** Three things here are new, and they are the reason this is the test.

    1. **The consequence is placement, not a value.** `Decomposition` has
       [width_sum] / [widths_cover_the_extent] for the widths of a partition,
       which says there is no gap and no double count in AGGREGATE. That is
       strictly weaker than exactly-once placement: a decomposition that
       writes one slot twice and another never has exactly the right total
       width. What is needed is a bijection, and it is not derivable from the
       fold theorems.

    2. **The edges are DATA-DEPENDENT.** Every edge function instantiated so
       far - `blen`/`boff`, the proportional and granule-snapped families,
       residue classes, [Decomposition.clamped] - is a static function of the
       extents. These come out of a histogram of the input. The schema's
       `edge : nat -> nat` is general enough in principle; that had never been
       exercised, and this file exercises it by building the edge function
       FROM the widths rather than the other way round.

    3. **It is TWO LEVELS.** `Idx` is partitioned by BUCKET, and each bucket's
       slice is partitioned again by scatter GROUP. No instantiation had
       composed two levels, and `MixedRadix.v` does not cover it: its radix is
       fixed, and here the inner extent depends on the outer digit
       ([no_uniform_radix_describes_this]).

    ** The result is positive, and the honest form of it is narrow.

    The schema DOES describe this decomposition - [the_bucket_edges_are_a_
    decomposition_edge_function] hands `Decomposition`'s hypotheses over
    unchanged, and [the_reduction_theorem_still_applies] shows the fold
    theorem instantiates too, so a counting sort is not a different family so
    much as a different CONSEQUENCE of the same one. What the schema did not
    have is the consequence itself, which is the ~90 lines of
    [slot_injective] / [slot_onto] / [dest_*] below. Composing two levels then
    costs nothing but one hypothesis.

    ** What the compiler gets out of it: two `assert_eq!` become hypotheses.

    `scatter` runs a post-condition on every call, and its comment claims
    "this is a PROOF rather than a spot check". That claim was argued in a
    comment and checked by nothing. [the_chained_runs_tile_the_bucket] is it:
    from the cursors ALONE - each group stopped where the next one started,
    and the last stopped at `off[b+1]` - every slot of the bucket is written
    by exactly one group. It assumes nothing about the histogram, which is
    what makes the runtime check worth running.

    Its two other `assert_eq!`s turn out to be the hypotheses of the tiling
    theorems: "scatter grouping does not tile the input" is
    [the_groups_tile_the_chunks], and "bucket offsets disagree with the entry
    total" is `edge tot nb = total`.

    ** What is NOT claimed.

    These are facts about [nat]. Nothing here is about memory, a `u32`, or a
    thread: that the RUNNING binner walks these edges is the tie
    `tests/msm_counting_sort_model.rs` makes, against a plain sequential
    stable counting sort as the oracle. The u32 width obligation is the
    `scatter` assertion that `n * nw < u32::MAX`, discharged there and not
    here. And nothing here says the histogram is right - only that whatever
    it counted is placed exactly once.

    Checked with Rocq 9.1:  coqc proofs/CountingSort.v  *)

Require Import ZArith Arith Lia.
Require Decomposition.
(* `Decomposition` folds over Z, so ZArith has to be in scope; the schedule
   arithmetic here is all [nat], so nat stays the default scope. *)
Open Scope nat_scope.
Module D := Decomposition.

(* ------------------------------------------------------------------ *)
(** ** Edges built FROM widths                                         *)
(* ------------------------------------------------------------------ *)

(** The exclusive prefix sum of the part widths. Every decomposition proved
    before this one supplies an `edge` and derives its widths; a counting sort
    MEASURES its widths and derives the edges, which is the direction the
    schema had not been asked for. *)
Fixpoint edge (w : nat -> nat) (t : nat) : nat :=
  match t with
  | O => O
  | S t' => (edge w t' + w t')%nat
  end.

Lemma edge_step : forall w t, edge w (S t) = (edge w t + w t)%nat.
Proof. reflexivity. Qed.

Lemma edge_mono : forall w t, (edge w t <= edge w (S t))%nat.
Proof. intros w t. rewrite edge_step. lia. Qed.

Lemma edge_le : forall w t u, (t <= u)%nat -> (edge w t <= edge w u)%nat.
Proof.
  intros w t u H. induction u as [| u IH].
  - replace t with O by lia. reflexivity.
  - destruct (Nat.eq_dec t (S u)) as [-> | Hne]; [ lia | ].
    assert (t <= u)%nat by lia.
    pose proof (edge_mono w u). lia.
Qed.

(** **The bridge.** A width-derived edge satisfies exactly the three
    hypotheses `Decomposition`'s theorems take, so a counting sort's
    decomposition is an instance of the schema and not a new family. *)
Theorem the_bucket_edges_are_a_decomposition_edge_function : forall w n N,
  edge w n = N ->
  edge w 0%nat = 0%nat /\ edge w n = N /\ (forall u, (edge w u <= edge w (S u))%nat).
Proof.
  intros w n N H.
  split; [ reflexivity | split; [ exact H | apply edge_mono ] ].
Qed.

(** And the widths recover the edge, which is `Decomposition.width_sum_closed`
    read through this file's constructor rather than re-proved. *)
Lemma edge_is_width_sum : forall e n,
  edge (fun u => (e (S u) - e u)%nat) n = D.width_sum e n.
Proof.
  intros e n. induction n as [| n IH]; cbn [edge D.width_sum].
  - reflexivity.
  - now rewrite IH.
Qed.

Theorem widths_of_an_edge_recover_it : forall e n,
  e 0%nat = 0%nat ->
  (forall u, (e u <= e (S u))%nat) ->
  edge (fun u => (e (S u) - e u)%nat) n = e n.
Proof.
  intros e n H0 Hm. rewrite edge_is_width_sum.
  now apply D.width_sum_closed.
Qed.

(* ------------------------------------------------------------------ *)
(** ** One level: the slot map is a bijection                          *)
(* ------------------------------------------------------------------ *)

(** Part `t`'s `r`-th slot. *)
Definition slot (w : nat -> nat) (t r : nat) : nat := (edge w t + r)%nat.

(** A slot exists only for a part that is visited and a rank inside it. The
    rank bound is what the whole file rests on, exactly as [MixedRadix]'s
    digit bound is: without it the map is not injective. *)
Definition legal (w : nat -> nat) (n t r : nat) : Prop :=
  ((t < n)%nat /\ (r < w t)%nat).

Lemma slot_below_next : forall w t r,
  (r < w t)%nat -> (slot w t r < edge w (S t))%nat.
Proof. intros w t r H. unfold slot. rewrite edge_step. lia. Qed.

(** In bounds: no part writes past the end of the buffer its widths add up to. *)
Theorem slot_in_range : forall w n t r,
  legal w n t r -> (slot w t r < edge w n)%nat.
Proof.
  intros w n t r [Ht Hr].
  pose proof (slot_below_next w t r Hr).
  assert (edge w (S t) <= edge w n)%nat by (apply edge_le; lia).
  lia.
Qed.

(** Injective: no slot is written twice. This is where the placement
    obligation stops being a corollary of the width sum - the aggregate is
    equally happy with one slot written twice and another never. *)
Theorem slot_injective : forall w n t1 r1 t2 r2,
  legal w n t1 r1 -> legal w n t2 r2 ->
  slot w t1 r1 = slot w t2 r2 -> t1 = t2 /\ r1 = r2.
Proof.
  intros w n t1 r1 t2 r2 [Ht1 Hr1] [Ht2 Hr2] Heq.
  destruct (lt_eq_lt_dec t1 t2) as [[Hlt | He] | Hgt].
  - exfalso.
    pose proof (slot_below_next w t1 r1 Hr1).
    assert (edge w (S t1) <= edge w t2)%nat by (apply edge_le; lia).
    unfold slot in *. lia.
  - subst t2. unfold slot in Heq. split; [ reflexivity | lia ].
  - exfalso.
    pose proof (slot_below_next w t2 r2 Hr2).
    assert (edge w (S t2) <= edge w t1)%nat by (apply edge_le; lia).
    unfold slot in *. lia.
Qed.

(** Onto: no slot keeps whatever was in the buffer before. For `Idx` that
    matters more than usual - it is allocated with `vec![0u32; total]`, so an
    unwritten slot reads back as the perfectly legitimate point index 0. *)
Theorem slot_onto : forall w n s,
  (s < edge w n)%nat -> exists t r, legal w n t r /\ slot w t r = s.
Proof.
  intros w n. induction n as [| n IH]; intros s Hs.
  - cbn in Hs. lia.
  - destruct (lt_dec s (edge w n)) as [Hlo | Hhi].
    + destruct (IH s Hlo) as (t & r & [Ht Hr] & Heq).
      exists t, r. split; [ split; [ lia | exact Hr ] | exact Heq ].
    + exists n, (s - edge w n)%nat.
      rewrite edge_step in Hs.
      split; [ split; lia | unfold slot; lia ].
Qed.

(** The three together. *)
Theorem every_slot_is_written_exactly_once : forall w n s,
  (s < edge w n)%nat ->
  exists t r, legal w n t r /\ slot w t r = s /\
    (forall t' r', legal w n t' r' -> slot w t' r' = s -> t' = t /\ r' = r).
Proof.
  intros w n s Hs.
  destruct (slot_onto w n s Hs) as (t & r & Hl & Heq).
  exists t, r. split; [ exact Hl | split; [ exact Heq | ] ].
  intros t' r' Hl' Heq'.
  apply (slot_injective w n t' r' t r Hl' Hl).
  rewrite Heq', Heq. reflexivity.
Qed.

(** The reduction theorem instantiates as well, which is what says this is a
    consequence of the schema rather than a rival to it: the same edges that
    place the entries also reproduce a naive fold over them. Nothing in the
    MSM uses this - a counting sort has no values to add - and it is here
    because the alternative claim ("a placement decomposition is a different
    family") would be wrong. *)
Theorem the_reduction_theorem_still_applies : forall w n f,
  D.acc_parts Z.add (edge w) f n = D.acc_range Z.add f 0%nat (edge w n).
Proof.
  intros w n f.
  apply (D.contiguous_exact (edge w) f n (edge w n));
    [ reflexivity | reflexivity | apply edge_mono ].
Qed.

(* ------------------------------------------------------------------ *)
(** ** The observable post-condition: chaining alone is enough         *)
(* ------------------------------------------------------------------ *)

(** What `scatter` can actually see after its threads have joined is the
    cursor each group STOPPED at, against the cursor it STARTED at. It cannot
    see how many entries a group wrote, and it has no reason to trust the
    histogram it was handed.

    The claim its comment makes is that the chaining is enough:

      "If every group stopped exactly where the next one STARTED - and the
       last stopped at `off[b+1]` - then the groups' runs tile
       `off[b]..off[b+1]` exactly: every slot of `idx` was written, by exactly
       one thread, in bounds."

    That is [the_chained_runs_tile_the_bucket] and the two beside it. The
    proof route is to show the observed starts ARE the prefix edges of the
    (unknown) written counts, at which point the bijection above applies. *)
Theorem chained_starts_are_prefix_edges : forall start written n lo,
  start 0%nat = lo ->
  (forall gi, (gi < n)%nat -> (start gi + written gi)%nat = start (S gi)) ->
  forall gi, (gi <= n)%nat -> start gi = (lo + edge written gi)%nat.
Proof.
  intros start written n lo Hlo Hchain gi Hle.
  induction gi as [| gi IH].
  - cbn. lia.
  - rewrite edge_step. rewrite <- (Hchain gi) by lia.
    rewrite IH by lia. lia.
Qed.

(** In bounds: nothing is written outside the bucket. *)
Theorem a_chained_run_stays_in_the_bucket : forall start written n lo hi gi r,
  start 0%nat = lo -> start n = hi ->
  (forall g, (g < n)%nat -> (start g + written g)%nat = start (S g)) ->
  (gi < n)%nat -> (r < written gi)%nat ->
  (lo <= start gi + r)%nat /\ (start gi + r < hi)%nat.
Proof.
  intros start written n lo hi gi r Hlo Hhi Hchain Hgi Hr.
  rewrite (chained_starts_are_prefix_edges start written n lo Hlo Hchain gi) by lia.
  assert (Hn : hi = (lo + edge written n)%nat).
  { rewrite <- Hhi.
    now apply (chained_starts_are_prefix_edges start written n lo Hlo Hchain). }
  pose proof (slot_in_range written n gi r (conj Hgi Hr)) as Hin.
  unfold slot in Hin. lia.
Qed.

(** No overlap: no two groups write the same slot. *)
Theorem the_chained_runs_do_not_overlap : forall start written n lo gi1 r1 gi2 r2,
  start 0%nat = lo ->
  (forall g, (g < n)%nat -> (start g + written g)%nat = start (S g)) ->
  (gi1 < n)%nat -> (r1 < written gi1)%nat ->
  (gi2 < n)%nat -> (r2 < written gi2)%nat ->
  (start gi1 + r1)%nat = (start gi2 + r2)%nat -> gi1 = gi2 /\ r1 = r2.
Proof.
  intros start written n lo gi1 r1 gi2 r2 Hlo Hchain H1 Hr1 H2 Hr2 Heq.
  rewrite (chained_starts_are_prefix_edges start written n lo Hlo Hchain gi1) in Heq by lia.
  rewrite (chained_starts_are_prefix_edges start written n lo Hlo Hchain gi2) in Heq by lia.
  apply (slot_injective written n gi1 r1 gi2 r2 (conj H1 Hr1) (conj H2 Hr2)).
  unfold slot. lia.
Qed.

(** No gap: every slot of the bucket is written by some group. *)
Theorem the_chained_runs_tile_the_bucket : forall start written n lo hi s,
  start 0%nat = lo -> start n = hi ->
  (forall g, (g < n)%nat -> (start g + written g)%nat = start (S g)) ->
  (lo <= s)%nat -> (s < hi)%nat ->
  exists gi r, (gi < n)%nat /\ (r < written gi)%nat /\ (start gi + r)%nat = s.
Proof.
  intros start written n lo hi s Hlo Hhi Hchain Hge Hlt.
  assert (Hn : hi = (lo + edge written n)%nat).
  { rewrite <- Hhi.
    now apply (chained_starts_are_prefix_edges start written n lo Hlo Hchain). }
  destruct (slot_onto written n (s - lo)%nat) as (gi & r & [Hgi Hr] & Heq); [ lia | ].
  exists gi, r. repeat split; try assumption.
  rewrite (chained_starts_are_prefix_edges start written n lo Hlo Hchain gi) by lia.
  unfold slot in Heq. lia.
Qed.

(** **The chaining cannot be dropped, in either direction.** Stated as
    refutations of the WEAKENED theorems rather than as witnesses: a control
    that merely exhibits an interesting instance stays green when the fixture
    drifts to a boring one, and this file's siblings have been caught by that
    twice. No choice of witness satisfies a refuted universal. *)

(** Without it, a slot can be written by NOBODY - which in `Idx` reads back as
    point index 0, indistinguishable from a legitimate entry. *)
Theorem without_the_chaining_a_slot_can_go_unwritten :
  ~ (forall (start written : nat -> nat) (n lo hi s : nat),
       start 0%nat = lo -> start n = hi ->
       (lo <= s)%nat -> (s < hi)%nat ->
       exists gi r, (gi < n)%nat /\ (r < written gi)%nat /\ (start gi + r)%nat = s).
Proof.
  intros H.
  destruct (H (fun gi => match gi with O => 0%nat | S O => 2%nat | _ => 4%nat end)
              (fun _ => 1%nat) 2%nat 0%nat 4%nat 1%nat
              eq_refl eq_refl ltac:(lia) ltac:(lia))
    as (gi & r & Hgi & Hr & Heq).
  destruct gi as [| [| gi]]; cbn in *; lia.
Qed.

(** And two groups can write the SAME slot, which loses an entry and races. *)
Theorem without_the_chaining_two_groups_can_write_one_slot :
  ~ (forall (start written : nat -> nat) (n gi1 r1 gi2 r2 : nat),
       (gi1 < n)%nat -> (r1 < written gi1)%nat ->
       (gi2 < n)%nat -> (r2 < written gi2)%nat ->
       (start gi1 + r1)%nat = (start gi2 + r2)%nat -> gi1 = gi2).
Proof.
  intros H.
  assert (Hbad : (0 = 1)%nat).
  { apply (H (fun _ => 0%nat) (fun _ => 1%nat) 2%nat 0%nat 0%nat 1%nat 0%nat);
      cbn; lia. }
  discriminate.
Qed.

(** The non-vacuity control for the chaining hypothesis, and an EMPTY group is
    the case it is stated at on purpose: the first version of `scatter`'s
    runtime check compared each group's end against the next group's
    post-scatter cursor rather than against its START, and for an empty group
    those differ. It failed on bucket 1 for exactly that reason. *)
Theorem an_empty_group_still_chains :
  let start := fun gi => match gi with O => 7%nat | S O => 9%nat
                                     | S (S O) => 9%nat | _ => 12%nat end in
  let written := fun gi => match gi with O => 2%nat | S O => 0%nat | _ => 3%nat end in
  start 0%nat = 7%nat /\ start 3%nat = 12%nat /\
  (forall g, (g < 3)%nat -> (start g + written g)%nat = start (S g)) /\
  written 1%nat = 0%nat.
Proof.
  cbn. repeat split; try reflexivity.
  intros g Hg. destruct g as [| [| [| g]]]; cbn; lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Two levels: buckets outside, scatter groups inside              *)
(* ------------------------------------------------------------------ *)

(** `Idx` is cut twice. Bucket `b` owns `[edge tot b, edge tot (S b))`, and
    inside it group `gi` owns `[edge (gw b) gi, edge (gw b) (S gi))`. The
    destination of the `r`-th entry group `gi` puts in bucket `b` is the sum
    of the two offsets - which is exactly the cursor `scatter` builds. *)
Definition dest (tot : nat -> nat) (gw : nat -> nat -> nat) (b gi r : nat) : nat :=
  (edge tot b + edge (gw b) gi + r)%nat.

(** The single hypothesis the second level costs, and it is not an assumption
    invented for the proof: it is `scatter`'s own
    `assert_eq!(.., ngroup, "scatter grouping does not tile the input")`. If
    the groups do not exhaust a bucket's chunks the inner edges stop short,
    and `zip` TRUNCATES rather than complaining. *)
Definition groups_exhaust_the_buckets
    (tot : nat -> nat) (gw : nat -> nat -> nat) (nb ng : nat) : Prop :=
  forall b, (b < nb)%nat -> edge (gw b) ng = tot b.

Theorem dest_in_range : forall tot gw nb ng b gi r,
  groups_exhaust_the_buckets tot gw nb ng ->
  (b < nb)%nat -> (gi < ng)%nat -> (r < gw b gi)%nat ->
  (dest tot gw b gi r < edge tot nb)%nat.
Proof.
  intros tot gw nb ng b gi r Hex Hb Hgi Hr.
  pose proof (slot_in_range (gw b) ng gi r (conj Hgi Hr)) as Hin.
  rewrite (Hex b Hb) in Hin. unfold slot in Hin.
  assert (Hl : legal tot nb b (edge (gw b) gi + r)) by (split; assumption).
  pose proof (slot_in_range tot nb b (edge (gw b) gi + r) Hl) as Hout.
  unfold slot, dest in *. lia.
Qed.

Theorem dest_injective : forall tot gw nb ng b1 gi1 r1 b2 gi2 r2,
  groups_exhaust_the_buckets tot gw nb ng ->
  (b1 < nb)%nat -> (gi1 < ng)%nat -> (r1 < gw b1 gi1)%nat ->
  (b2 < nb)%nat -> (gi2 < ng)%nat -> (r2 < gw b2 gi2)%nat ->
  dest tot gw b1 gi1 r1 = dest tot gw b2 gi2 r2 ->
  b1 = b2 /\ gi1 = gi2 /\ r1 = r2.
Proof.
  intros tot gw nb ng b1 gi1 r1 b2 gi2 r2 Hex Hb1 Hg1 Hr1 Hb2 Hg2 Hr2 Heq.
  (* The inner offset is a legal rank of the OUTER decomposition, because the
     groups exhaust the bucket. So the outer bijection separates the buckets
     first, and only then does the inner one separate the groups. *)
  pose proof (slot_in_range (gw b1) ng gi1 r1 (conj Hg1 Hr1)) as Hi1.
  pose proof (slot_in_range (gw b2) ng gi2 r2 (conj Hg2 Hr2)) as Hi2.
  rewrite (Hex b1 Hb1) in Hi1. rewrite (Hex b2 Hb2) in Hi2.
  unfold slot in Hi1, Hi2.
  assert (Hl1 : legal tot nb b1 (edge (gw b1) gi1 + r1)) by (split; assumption).
  assert (Hl2 : legal tot nb b2 (edge (gw b2) gi2 + r2)) by (split; assumption).
  assert (Hs : slot tot b1 (edge (gw b1) gi1 + r1)
             = slot tot b2 (edge (gw b2) gi2 + r2))
    by (unfold slot, dest in *; lia).
  destruct (slot_injective tot nb b1 (edge (gw b1) gi1 + r1)
                                 b2 (edge (gw b2) gi2 + r2) Hl1 Hl2 Hs)
    as [Hb Hq].
  subst b2. split; [ reflexivity | ].
  apply (slot_injective (gw b1) ng gi1 r1 gi2 r2
           (conj Hg1 Hr1) (conj Hg2 Hr2)).
  unfold slot. lia.
Qed.

Theorem dest_onto : forall tot gw nb ng s,
  groups_exhaust_the_buckets tot gw nb ng ->
  (s < edge tot nb)%nat ->
  exists b gi r, (b < nb)%nat /\ (gi < ng)%nat /\ (r < gw b gi)%nat /\
                 dest tot gw b gi r = s.
Proof.
  intros tot gw nb ng s Hex Hs.
  destruct (slot_onto tot nb s Hs) as (b & q & [Hb Hq] & Heq).
  rewrite <- (Hex b Hb) in Hq.
  destruct (slot_onto (gw b) ng q Hq) as (gi & r & [Hgi Hr] & Heq2).
  exists b, gi, r.
  split; [ exact Hb | split; [ exact Hgi | split; [ exact Hr | ] ] ].
  unfold dest. unfold slot in Heq, Heq2. lia.
Qed.

(** **The capstone.** Every slot of `Idx` holds exactly one entry, placed by
    exactly one scatter thread. *)
Theorem every_idx_slot_is_written_exactly_once : forall tot gw nb ng s,
  groups_exhaust_the_buckets tot gw nb ng ->
  (s < edge tot nb)%nat ->
  exists b gi r,
    (b < nb)%nat /\ (gi < ng)%nat /\ (r < gw b gi)%nat /\
    dest tot gw b gi r = s /\
    (forall b' gi' r',
       (b' < nb)%nat -> (gi' < ng)%nat -> (r' < gw b' gi')%nat ->
       dest tot gw b' gi' r' = s -> b' = b /\ gi' = gi /\ r' = r).
Proof.
  intros tot gw nb ng s Hex Hs.
  destruct (dest_onto tot gw nb ng s Hex Hs) as (b & gi & r & Hb & Hgi & Hr & Heq).
  exists b, gi, r.
  split; [ exact Hb | split; [ exact Hgi | split; [ exact Hr |
    split; [ exact Heq | ] ] ] ].
  intros b' gi' r' Hb' Hgi' Hr' Heq'.
  apply (dest_injective tot gw nb ng b' gi' r' b gi r Hex Hb' Hgi' Hr' Hb Hgi Hr).
  rewrite Heq', Heq. reflexivity.
Qed.

(** **[MixedRadix] does not cover this, and that is a property of the
    decomposition rather than of how it happens to be written.** A positional
    index `q*B + r` needs one radix for every digit; here the inner extent is
    the bucket's own width, and bucket widths differ because they count data.
    Refuted over a bucket set of widths 1 and 2. *)
Definition tot_ex (b : nat) : nat :=
  match b with O => 1%nat | S O => 2%nat | _ => 0%nat end.

Theorem no_uniform_radix_describes_this : forall B,
  ~ (forall b, (b <= 2)%nat -> edge tot_ex b = (b * B)%nat).
Proof.
  intros B H.
  pose proof (H 1%nat ltac:(lia)) as H1.
  pose proof (H 2%nat ltac:(lia)) as H2.
  cbn in H1, H2. lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The MSM's own edges                                             *)
(* ------------------------------------------------------------------ *)

(** `counts b t` is how many entries histogram chunk `t` puts in bucket `b`,
    and a scatter group is a contiguous run of `grp` chunks. So the group
    widths are differences of the chunk prefix - which makes the chunk
    grouping the third member of [Decomposition.clamped], after the int32
    flush interval and the output tile width. Two spellings had already turned
    out to be one family there; this is a third, arrived at from a counting
    sort rather than from a GEMM. *)
Definition gedge (grp nchunk gi : nat) : nat := D.clamped grp nchunk gi.

Definition group_width (counts : nat -> nat -> nat) (grp nchunk : nat)
    (b gi : nat) : nat :=
  (edge (counts b) (gedge grp nchunk (S gi)) - edge (counts b) (gedge grp nchunk gi))%nat.

Definition bucket_total (counts : nat -> nat -> nat) (nchunk b : nat) : nat :=
  edge (counts b) nchunk.

(** The chunk grouping tiles the chunks exactly when the group count spans
    them, `nchunk <= ng * grp` - which is `ngroup = ceil(nchunk / group)`, and
    is what `scatter`'s grouping assertion checks at run time. *)
Theorem the_group_widths_exhaust_every_bucket : forall counts grp nchunk ng nb,
  (nchunk <= ng * grp)%nat ->
  groups_exhaust_the_buckets
    (bucket_total counts nchunk) (group_width counts grp nchunk) nb ng.
Proof.
  intros counts grp nchunk ng nb Hspan b _.
  unfold bucket_total, group_width, gedge.
  rewrite (widths_of_an_edge_recover_it
             (fun gi => edge (counts b) (D.clamped grp nchunk gi)) ng).
  - rewrite D.clamped_last by exact Hspan. reflexivity.
  - rewrite D.clamped_zero. reflexivity.
  - intros u. apply edge_le. apply D.clamped_monotone.
Qed.

(** So the cursor table `scatter` builds is the two-level destination map, and
    every entry it writes lands in its own slot. *)
Definition cursor (counts : nat -> nat -> nat) (grp nchunk b gi : nat) : nat :=
  (edge (bucket_total counts nchunk) b
   + edge (group_width counts grp nchunk b) gi)%nat.

Theorem the_cursor_is_the_destination : forall counts grp nchunk b gi r,
  (cursor counts grp nchunk b gi + r)%nat
  = dest (bucket_total counts nchunk) (group_width counts grp nchunk) b gi r.
Proof. reflexivity. Qed.

Theorem the_msm_scatter_writes_every_slot_exactly_once :
  forall counts grp nchunk nb ng s,
  (nchunk <= ng * grp)%nat ->
  (s < edge (bucket_total counts nchunk) nb)%nat ->
  exists b gi r,
    (b < nb)%nat /\ (gi < ng)%nat /\ (r < group_width counts grp nchunk b gi)%nat /\
    (cursor counts grp nchunk b gi + r)%nat = s /\
    (forall b' gi' r',
       (b' < nb)%nat -> (gi' < ng)%nat ->
       (r' < group_width counts grp nchunk b' gi')%nat ->
       (cursor counts grp nchunk b' gi' + r')%nat = s ->
       b' = b /\ gi' = gi /\ r' = r).
Proof.
  intros counts grp nchunk nb ng s Hspan Hs.
  apply (every_idx_slot_is_written_exactly_once
           (bucket_total counts nchunk) (group_width counts grp nchunk) nb ng s);
    [ apply the_group_widths_exhaust_every_bucket; exact Hspan | exact Hs ].
Qed.

(** The non-vacuity control. Every theorem above is satisfied by a histogram
    that counted nothing, and by a single bucket with a single group - so a
    concrete instance with two buckets of different widths, three groups, and
    one group EMPTY is evaluated here. *)
Definition counts_ex (b t : nat) : nat :=
  match b, t with
  | O, O => 2%nat | O, S O => 0%nat | O, S (S O) => 1%nat
  | S O, O => 1%nat | S O, S O => 3%nat | S O, S (S O) => 2%nat
  | _, _ => 0%nat
  end.

Theorem the_msm_instance_is_not_vacuous :
  bucket_total counts_ex 3%nat 0%nat = 3%nat /\
  bucket_total counts_ex 3%nat 1%nat = 6%nat /\
  edge (bucket_total counts_ex 3%nat) 2%nat = 9%nat /\
  group_width counts_ex 1%nat 3%nat 0%nat 1%nat = 0%nat /\
  cursor counts_ex 1%nat 3%nat 1%nat 2%nat = 7%nat.
Proof. repeat split; vm_compute; reflexivity. Qed.

Print Assumptions edge_le.
Print Assumptions the_bucket_edges_are_a_decomposition_edge_function.
Print Assumptions widths_of_an_edge_recover_it.
Print Assumptions slot_in_range.
Print Assumptions slot_injective.
Print Assumptions slot_onto.
Print Assumptions every_slot_is_written_exactly_once.
Print Assumptions the_reduction_theorem_still_applies.
Print Assumptions chained_starts_are_prefix_edges.
Print Assumptions a_chained_run_stays_in_the_bucket.
Print Assumptions the_chained_runs_do_not_overlap.
Print Assumptions the_chained_runs_tile_the_bucket.
Print Assumptions without_the_chaining_a_slot_can_go_unwritten.
Print Assumptions without_the_chaining_two_groups_can_write_one_slot.
Print Assumptions an_empty_group_still_chains.
Print Assumptions dest_in_range.
Print Assumptions dest_injective.
Print Assumptions dest_onto.
Print Assumptions every_idx_slot_is_written_exactly_once.
Print Assumptions no_uniform_radix_describes_this.
Print Assumptions the_group_widths_exhaust_every_bucket.
Print Assumptions the_cursor_is_the_destination.
Print Assumptions the_msm_scatter_writes_every_slot_exactly_once.
Print Assumptions the_msm_instance_is_not_vacuous.
