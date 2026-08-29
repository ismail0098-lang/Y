(** * The f32 kernel's band decompositions - a SECOND kernel, and what composed.

    `docs/proof_carrying_kernels.md` puts the research risk of the whole
    programme in one sentence: "if obligations don't compose, the thing is a
    one-off proof rather than a compiler", and its Phase 2 "Done when" is a
    second, structurally different kernel. `src/cpu_gemm.rs` already emits one.
    Beside the exact `vpdpwssd` GEMM that the other nine proofs are about sits
    `__y_sgemm_f32_avx512`, which ships for ordinary Y programs, partitions the
    same three axes, and had **no proofs at all**.

    This file is that experiment run early, on a kernel that already exists,
    rather than after building the transformation IR.

    ** The result, stated before the proofs.

    - The **tiling** obligation composes, and its PROOF does not transfer: the
      f32 K-split is proportional ([pedge]), the exact one is
      even-with-remainder ([ExactGemmKsplit.blen]). Both cover `[0, ext)`;
      they are not the same partition, and [the_two_splits_are_different]
      exhibits an instance. So the obligation had to be re-discharged - about
      thirty lines - not re-used.
    - The **range-folding machinery** transfers VERBATIM. [acc_range] and
      [sum_range_split] are [ExactGemmKsplit]'s, used unchanged; only the
      band-indexed fold needed redefining, because that is the part that names
      the decomposition.
    - The **exactness** obligation provably does NOT transfer, and that is not
      a gap. f32 addition is not associative, so per-band partials do not sum
      to the naive sum; [rounding_breaks_the_proportional_split_too] refutes it
      at the same `f`, `K` and thread count where the exact kernel's own
      refutation lives. The f32 kernel is entitled to the tiling theorem and to
      nothing else, and the repo asserts no bit-identity for it anywhere.

    ** What this buys the shipping compiler.

    Both decompositions clamp their last band - `select (t+1 == n) ext hi` -
    with a comment saying the last thread takes the remainder "so no row of B
    is dropped". [pedge_last] and [gedge_last] prove those clamps are
    **redundant**: the arithmetic already lands on `ext`. Redundant is not
    dead - the packers' masks in `ExactGemmPacking.v` are redundant with each
    other and are kept - but it moves the property from a comment to a theorem,
    which is what the whole exercise is for.

    ** What is NOT claimed.

    The definitions here are the emitter's, generated into
    `ExactGemmSchedule.v` from `cpu_gemm.rs` and consumed by both, so there is
    no second description of the arithmetic. There is still no model of the
    f32 micro-kernel, of its packing, or of its scratch reduction; this is the
    schedule and only the schedule. Accumulators are `Z`, so nothing here says
    anything about float rounding beyond the one refutation, which uses
    [ExactGemmKsplit]'s deliberately crude `rnd`.

    Build:  coqc proofs/GemmBandSplit.v      (Rocq 9.1)
*)

From Stdlib Require Import ZArith Arith Lia.
Require ExactGemmSchedule.
Require ExactGemmKsplit.

Open Scope Z_scope.

Module SCH := ExactGemmSchedule.
Module KS := ExactGemmKsplit.

(* ------------------------------------------------------------------ *)
(** ** The proportional split: the f32 K-bands                          *)
(* ------------------------------------------------------------------ *)

(** Transcribed from [cpu_gemm::emit_entry], and shared with it:

      %kl0   = mul nsw i64 %ti, %K
      %kfrom = sdiv i64 %kl0, %nt
      %kl1   = mul nsw i64 %tn, %K        ; tn = ti + 1
      %kto0  = sdiv i64 %kl1, %nt
      %kto   = select i1 %k_last, i64 %K, i64 %kto0

    Band `t` is `[pedge t, pedge (S t))`, so contiguity is definitional and the
    whole tiling obligation reduces to the two endpoints. *)
Definition pedge (t n ext : nat) : nat := SCH.prop_band_edge t n ext.

Definition plen (t n ext : nat) : nat := (pedge (S t) n ext - pedge t n ext)%nat.

Lemma pedge_unfold : forall t n ext, pedge t n ext = ((t * ext) / n)%nat.
Proof. reflexivity. Qed.

Theorem pedge_zero : forall n ext, pedge 0 n ext = 0%nat.
Proof.
  intros n ext. rewrite pedge_unfold, Nat.mul_0_l.
  now apply Nat.Div0.div_0_l.
Qed.

(** **The clamp is redundant**, which is the point of the theorem rather than a
    remark. The emitter writes `select (t+1 == n) K kto0` because the comment
    beside it says the last thread must take the remainder; `(n * ext) / n` is
    already exactly `ext`. *)
Theorem pedge_last : forall n ext, (0 < n)%nat -> pedge n n ext = ext.
Proof.
  intros n ext Hn. rewrite pedge_unfold.
  rewrite Nat.mul_comm, Nat.div_mul by lia. reflexivity.
Qed.

Theorem pedge_monotone : forall t n ext,
  (pedge t n ext <= pedge (S t) n ext)%nat.
Proof.
  intros t n ext. rewrite !pedge_unfold.
  destruct n as [| n']; [ simpl; lia | ].
  apply Nat.Div0.div_le_mono. nia.
Qed.

(** The tiling obligation, in the shape the proportional form makes natural:
    the bands start at 0, end at `ext`, and never run backwards. Contiguity
    needs no lemma - band `t` ends where band `S t` begins, by construction,
    which is exactly the structural difference from the exact kernel's split
    (where [ExactGemmKsplit.bands_contiguous] is a real induction). *)
Theorem prop_bands_tile : forall n ext,
  (0 < n)%nat ->
  pedge 0 n ext = 0%nat
  /\ pedge n n ext = ext
  /\ forall t, (pedge t n ext <= pedge (S t) n ext)%nat.
Proof.
  intros n ext Hn. repeat split.
  - apply pedge_zero.
  - now apply pedge_last.
  - intros t. apply pedge_monotone.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The granule split: the f32 M and N bands                         *)
(* ------------------------------------------------------------------ *)

(** The granule count is [ExactGemmSchedule.tile_count] - the SAME expression
    the exact kernel's threaded wrapper uses for `(M + MR-1)/MR`. That sharing
    is real and not cosmetic: one `Ix` renders both. *)
Definition gcount (ext gran : nat) : nat := SCH.tile_count ext (gran - 1) gran.

Definition gedge (idx count ext gran : nat) : nat :=
  SCH.granule_band_edge idx (gcount ext gran) count gran ext.

Lemma gedge_unfold : forall idx count ext gran,
  gedge idx count ext gran
  = Nat.min ((idx * gcount ext gran) / count * gran) ext.
Proof. reflexivity. Qed.

Lemma gcount_covers : forall ext gran,
  (0 < gran)%nat -> (ext <= gcount ext gran * gran)%nat.
Proof.
  intros ext gran Hg. unfold gcount, SCH.tile_count.
  set (a := (ext + (gran - 1))%nat).
  assert (Ha : a = (ext + (gran - 1))%nat) by reflexivity.
  assert (H : a = (gran * (a / gran) + a mod gran)%nat) by apply Nat.div_mod_eq.
  assert (Hm : (a mod gran < gran)%nat) by (apply Nat.mod_upper_bound; lia).
  rewrite Nat.mul_comm. lia.
Qed.

Theorem gedge_zero : forall count ext gran, gedge 0 count ext gran = 0%nat.
Proof.
  intros count ext gran. rewrite gedge_unfold.
  rewrite Nat.mul_0_l, Nat.Div0.div_0_l, Nat.mul_0_l. lia.
Qed.

(** **The M/N clamp is redundant too**, for the same reason plus the fact that
    a granule count always covers its extent. *)
Theorem gedge_last : forall count ext gran,
  (0 < count)%nat -> (0 < gran)%nat ->
  gedge count count ext gran = ext.
Proof.
  intros count ext gran Hc Hg. rewrite gedge_unfold.
  rewrite (Nat.mul_comm count (gcount ext gran)), Nat.div_mul by lia.
  apply Nat.min_r. now apply gcount_covers.
Qed.

Theorem gedge_monotone : forall idx count ext gran,
  (gedge idx count ext gran <= gedge (S idx) count ext gran)%nat.
Proof.
  intros idx count ext gran. rewrite !gedge_unfold.
  apply Nat.min_le_compat_r.
  apply Nat.mul_le_mono_r.
  destruct count as [| c]; [ simpl; lia | ].
  apply Nat.Div0.div_le_mono. nia.
Qed.

(** Every band edge is a multiple of the tile granularity, or the extent
    itself. That is the property the whole granule-counting design exists for -
    a band boundary inside a tile would make one thread write a partial tile -
    and it was stated only in a comment. *)
Theorem every_edge_snaps_to_a_granule_or_the_extent :
  forall idx count ext gran,
    (exists q, gedge idx count ext gran = (q * gran)%nat)
    \/ gedge idx count ext gran = ext.
Proof.
  intros idx count ext gran. rewrite gedge_unfold.
  destruct (Nat.min_dec ((idx * gcount ext gran) / count * gran) ext) as [H | H];
    rewrite H.
  - left. now exists ((idx * gcount ext gran) / count)%nat.
  - now right.
Qed.

(* ------------------------------------------------------------------ *)
(** ** What composed, and what did not                                  *)
(* ------------------------------------------------------------------ *)

(** The band-indexed fold. [ExactGemmKsplit.acc_range] is reused verbatim -
    folding a contiguous range is not a property of any decomposition - and
    only this is new, because it is the definition that names the bands. *)
Fixpoint pacc_bands (op : Z -> Z -> Z) (f : nat -> Z) (n ext t : nat) : Z :=
  match t with
  | O => 0
  | S t' => op (pacc_bands op f n ext t')
               (KS.acc_range op f (pedge t' n ext) (plen t' n ext))
  end.

Lemma pacc_prefix : forall f n ext t,
  (0 < n)%nat ->
  pacc_bands Z.add f n ext t = KS.acc_range Z.add f 0 (pedge t n ext).
Proof.
  intros f n ext t Hn. induction t as [| t IH].
  - cbn [pacc_bands]. rewrite pedge_zero. reflexivity.
  - cbn [pacc_bands]. rewrite IH.
    unfold plen.
    replace (pedge (S t) n ext)
      with (pedge t n ext + (pedge (S t) n ext - pedge t n ext))%nat
      at 2 by (pose proof (pedge_monotone t n ext); lia).
    rewrite KS.sum_range_split. rewrite Nat.add_0_l. reflexivity.
Qed.

(** **The tiling obligation, discharged for the second kernel.** Same shape as
    [ExactGemmKsplit.ksplit_exact] and a different proof, over a different
    decomposition, with no hypothesis that `n` divides `ext`. *)
Theorem prop_ksplit_exact : forall f n ext,
  (0 < n)%nat ->
  pacc_bands Z.add f n ext n = KS.acc_range Z.add f 0 ext.
Proof.
  intros f n ext Hn. rewrite pacc_prefix by exact Hn.
  now rewrite pedge_last.
Qed.

(** **The two splits are different partitions**, so the theorem above is not
    the previous one wearing a hat. `ext = 5` over `n = 3`: proportional band 0
    is one element, the exact kernel's is two. *)
Theorem the_two_splits_are_different :
  plen 0 3 5 <> KS.blen 5 3 0.
Proof. vm_compute. discriminate. Qed.

(** **The exactness obligation does not transfer, and the failure is about the
    ACCUMULATE and not about the decomposition.** Same `f`, same `K`, same
    thread count as [ExactGemmKsplit.rounding_breaks_the_split], now over the
    proportional bands: 1100 against its own reference's 1000. f32 addition is
    not associative, so the f32 kernel gets [prop_ksplit_exact] only for an
    exact accumulate - which it does not have, and which is the entire reason
    the exact kernel exists. *)
Theorem rounding_breaks_the_proportional_split_too :
  pacc_bands KS.fadd KS.spike 2 201 2 <> KS.acc_range KS.fadd KS.spike 0 201.
Proof. vm_compute. discriminate. Qed.

(** The control: with an exact accumulate the same split is exact, so the
    refutation above is about `fadd` and not about [pedge]. *)
Theorem exact_survives_the_proportional_split :
  pacc_bands Z.add KS.spike 2 201 2 = KS.acc_range Z.add KS.spike 0 201.
Proof. vm_compute. reflexivity. Qed.

Print Assumptions pedge_last.
Print Assumptions prop_bands_tile.
Print Assumptions gedge_last.
Print Assumptions every_edge_snaps_to_a_granule_or_the_extent.
Print Assumptions prop_ksplit_exact.
Print Assumptions the_two_splits_are_different.
Print Assumptions rounding_breaks_the_proportional_split_too.
Print Assumptions exact_survives_the_proportional_split.
