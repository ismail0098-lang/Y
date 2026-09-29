(** * The exact PV kernel computes the source dot product -- and its SASS is validated

    This is the first kernel in the repository to carry BOTH a Rocq proof of
    what its PTX computes AND a per-translation validation of the PTX -> SASS
    step.  Until now those two sets were DISJOINT: [tools/ptxas_tval/] validates
    six kernels and none of them had a proof, while the three proved GPU
    kernels (the int8 GEMM, the attention pair, the warp tiling) are not in the
    validator's corpus.  [tests/exact_pv_proof.rs] asserts that overlap rather
    than leaving it as prose.

    ** THE CHAIN, AND ITS SEAMS

    <<
      Y source (tests/exact_pv.ysu)
        |  [the statement of the capstone below: the PTX loop's result IS
        |   the source's dot product]
      emitted PTX (tests/exact_pv.ptx)
        |  [tools/ptxas_tval/loopval.py at -O1: 14 obligations, VALIDATED]
      SASS the GPU runs
    >>

    Neither seam is free and both are named.  The first is a TRANSCRIPTION
    plus a gate - [ptx_emitter.rs] does not go through the [Ix] extraction
    layer, so [tests/exact_pv_proof.rs] reads the emitted text and asserts it
    is the arithmetic modelled here, rather than the two being rendered from
    one description the way [exact_attention.rs] is.  The second carries the
    validator's own stated assumptions: one multiplier identity ASSUMED, a
    single thread's view, and one optimisation level.  What the chain is not is
    a proof about [ptxas].

    ** WHAT WRITING IT FOUND: THE CEILING NOBODY WROTE DOWN

    [tests/exact_pv.ysu]'s own header states one bound -- "accumulating in
    int64 needs [T * 2^28 * V_LEVELS < 2^63], i.e. about 2.7e8 tokens" -- and
    nothing anywhere checks it.  That bound is correct and it is NOT THE
    BINDING ONE.  Every index in the emitted kernel is computed in 32-bit
    WRAPPING arithmetic ([mul.lo.s32] / [add.s32]), so the V access
    [(b*T + t)*D + d] additionally needs

    <<  B * T * D  <=  i32::MAX  >>

    which at head_dim 64 is 33,554,432 tokens -- **8.06x tighter** than the
    accumulator bound that is documented -- and at [B = 8], head_dim 128 is
    2,097,152, **129x tighter**.  Nothing states it, in the source, in
    [proofs/], or in [tests/].

    **Measured on the device before any of this was written.**  With [D = 2^30]
    so that [t*D] reaches [2^32] at exactly [t = 4], [NV = 1] so only index 0
    is in range, and [P[t] = t+1]:

    | T | device | unbounded-index reference |         |
    |---|--------|---------------------------|---------|
    | 3 |      1 |                         1 | agree   |
    | 4 |      1 |                         1 | agree   |
    | 5 |      6 |                         1 | DIVERGE |
    | 6 |      6 |                         1 | DIVERGE |

    One iteration wide.  The 6 is [P[0] + P[4]]: [4 * 2^30 = 2^32] wraps to 0,
    so [t = 4] re-reads [V[0]].  It is not a masked drop but a read of the
    WRONG ELEMENT, and it happens under a green banner.
    [the_measured_index_wrap_reads_v0_twice] reproduces both the indices and
    the answer from [wrap32] alone.

    The accumulator bound was measured at its own boundary in the same session,
    over the domain the PTX's OWN load instructions permit ([ld.global.u32] +
    [cvt.u64.u32] is a zero-extend, so [P] is in [[0, 2^32)]; [ld.global.s8] +
    [cvt.s64.s32] is a sign-extend, so [V] is in [[-128, 127]]):

    | T          | exact                 | device                |         |
    |------------|-----------------------|-----------------------|---------|
    | 16,777,216 | -9223372034707292160  | -9223372034707292160  | ok      |
    | 16,777,217 | -9223372584463105920  |  9223371489246445696  | WRAPPED |

    One iteration wide, and the wrap FLIPS THE SIGN.
    [the_measured_overflow_is_two_s_complement] reproduces that value from
    [wrap64] alone; a model that merely said "it overflows" would agree with
    any wrong answer.

    ** WHAT THIS DOES NOT CLAIM

    - It is not about [ptxas].  See the chain diagram: that step is the
      validator's, per-translation, with its own assumptions.
    - The licences are CONSERVATIVE, a worst case over the domain the emitted
      load instructions permit, exactly as [VnniExact::license] is over its
      declared operand type.  A caller who guarantees the semantic domain
      ([P <= 2^28], [|V| <= 127]) gets the larger accumulator ceiling stated in
      [the_semantic_domain_raises_the_accumulator_ceiling]; the index ceiling
      does not move, which is the whole point of the finding.
    - [T], [D], [Q] and the extents are RUNTIME parameters, so no compilation
      can check either licence.  Both are launch-boundary obligations and
      neither is checked; that is recorded rather than fixed.
    - It is per-thread and says nothing about how many threads run.  The
      output PARTITION is separate and is
      [every_output_element_is_written_by_one_thread].

    Build:  coqc proofs/ExactPvExact.v
    Gates:  tests/exact_pv_proof.rs, tests/proofs_are_checked.rs
*)

Require Import Coq.Arith.Arith.
Require Import Coq.micromega.Lia.
Require Import Coq.ZArith.ZArith.

Require MixedRadix.
Require ExactGemmMicro.

Module MR := MixedRadix.
Module MC := ExactGemmMicro.

Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** The int64 accumulator model                                     *)
(* ------------------------------------------------------------------ *)

(** [ExactGemmMicro]'s int32 model is concrete at 32 bits ([wrap32] is not
    width-parameterised), and this kernel accumulates in int64 -- [mul.lo.s64]
    then [add.s64], both wrapping.  So the twin is written here.  The int32
    model is still REQUIRED, because every INDEX in this kernel is 32-bit. *)

Definition I64MIN : Z := -9223372036854775808.
Definition I64MAX : Z :=  9223372036854775807.
Definition in_i64 (z : Z) : Prop := I64MIN <= z <= I64MAX.

Definition wrap64 (z : Z) : Z :=
  (z - I64MIN) mod 18446744073709551616 + I64MIN.

Lemma wrap64_id : forall z, in_i64 z -> wrap64 z = z.
Proof.
  intros z [Hlo Hhi]. unfold wrap64, I64MIN, I64MAX in *.
  rewrite Z.mod_small by lia. lia.
Qed.

(** The mathematical sum, and the accumulator the kernel actually runs.  The
    kernel seeds [%rd3] with 0 and folds ASCENDING in [t], one [add.s64] per
    iteration; [wacc] is that fold and nothing else. *)
Fixpoint sum_upto (f : nat -> Z) (n : nat) : Z :=
  match n with O => 0 | S m => sum_upto f m + f m end.

Fixpoint wacc (f : nat -> Z) (n : nat) : Z :=
  match n with O => 0 | S m => wrap64 (wacc f m + f m) end.

(** The int64 arithmetic agrees with [Z] exactly when no partial sum leaves the
    range.  This is the int64 twin of [ExactGemmMicro.flush_exact_in_int32]. *)
Theorem acc_exact_in_int64 : forall f n,
  (forall k, (k <= n)%nat -> in_i64 (sum_upto f k)) ->
  wacc f n = sum_upto f n.
Proof.
  induction n; intros Hb.
  - reflexivity.
  - simpl. rewrite IHn by (intros k Hk; apply Hb; lia).
    apply (wrap64_id (sum_upto f n + f n)).
    apply (Hb (S n)); lia.
Qed.

Lemma sum_abs_bound : forall f c,
  0 <= c -> (forall k, Z.abs (f k) <= c) ->
  forall n, Z.abs (sum_upto f n) <= Z.of_nat n * c.
Proof.
  intros f c Hc Hf n. induction n.
  - simpl. lia.
  - simpl sum_upto. rewrite Nat2Z.inj_succ.
    assert (Ht : Z.abs (sum_upto f n + f n) <= Z.abs (sum_upto f n) + Z.abs (f n))
      by apply Z.abs_triangle.
    assert (Hn : 0 <= Z.of_nat n) by apply Nat2Z.is_nonneg.
    specialize (Hf n). nia.
Qed.

(** **The licence, in the form the capstone consumes.**  If every term is
    bounded by [c] and [n * c] fits, the wrapping fold IS the sum. *)
Theorem bounded_terms_accumulate_exactly : forall f n c,
  0 <= c -> (forall k, Z.abs (f k) <= c) -> Z.of_nat n * c <= I64MAX ->
  wacc f n = sum_upto f n.
Proof.
  intros f n c Hc Hf Hn. apply acc_exact_in_int64.
  intros k Hk.
  assert (Hk' : Z.abs (sum_upto f k) <= Z.of_nat k * c) by (apply (sum_abs_bound f c Hc Hf)).
  assert (Z.of_nat k <= Z.of_nat n) by (apply inj_le; lia).
  assert (Z.of_nat k * c <= Z.of_nat n * c) by nia.
  unfold in_i64, I64MIN, I64MAX in *. lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The operand domains, read off the emitted load instructions      *)
(* ------------------------------------------------------------------ *)

(** [@%p3 ld.global.u32 %r21, [%rd6]; cvt.u64.u32 %rd7, %r21] -- ZERO-extended,
    so a weight is in [[0, 2^32)].  [@%p6 ld.global.s8 %r29, [%rd10];
    cvt.s64.s32 %rd11, %r29] -- SIGN-extended, so an activation is in
    [[-128, 127]].  These are the DECLARED domains; the licence is a worst case
    over them. *)
Definition PMAX : Z := 4294967295.
Definition VABS : Z := 128.

Definition p_domain (z : Z) : Prop := 0 <= z <= PMAX.
Definition v_domain (z : Z) : Prop := -VABS <= z <= VABS - 1.

(** [mul.lo.s64] wraps too, and it never has to: the widest product the two
    load instructions can produce is far inside int64.  Proved rather than
    assumed, because a wrapping multiply in the model would be invisible. *)
Theorem the_product_never_wraps : forall p v,
  p_domain p -> v_domain v -> wrap64 (p * v) = p * v.
Proof.
  intros p v [Hp0 Hp1] [Hv0 Hv1]. apply wrap64_id.
  unfold in_i64, I64MIN, I64MAX, p_domain, v_domain, PMAX, VABS in *. nia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The accumulator licence                                          *)
(* ------------------------------------------------------------------ *)

Definition MAX_EXACT_T : Z := I64MAX / (PMAX * VABS).

Theorem the_accumulator_bound_is_one_iteration_wide :
  MAX_EXACT_T = 16777216
  /\ 16777216 * (PMAX * VABS) <= I64MAX
  /\ I64MAX < 16777217 * (PMAX * VABS).
Proof.
  unfold MAX_EXACT_T, I64MAX, PMAX, VABS.
  repeat split; vm_compute; first [ reflexivity | discriminate ].
Qed.

(** **The refutation, refereed against the silicon.**  At [T = 16,777,217] with
    every [P] at [2^32-1] and every [V] at [-128] the device returned
    9223371489246445696 where the answer is -9223372584463105920.  Reproduced
    from [wrap64] alone -- the model was not fitted to it. *)
Theorem the_measured_overflow_is_two_s_complement :
  wrap64 (-(16777217 * (PMAX * VABS))) = 9223371489246445696
  /\ -(16777217 * (PMAX * VABS)) = -9223372584463105920.
Proof. unfold wrap64, I64MIN, PMAX, VABS. split; vm_compute; reflexivity. Qed.

(** The semantic domain a caller may promise -- a Q0.28 weight and a
    [+-127] activation -- raises the accumulator ceiling by ~16x.  Stated so
    the source's own documented figure is placed rather than contradicted: it
    is right about the accumulator. *)
Definition MAX_EXACT_T_SEMANTIC : Z := I64MAX / (268435456 * 127).

Theorem the_semantic_domain_raises_the_accumulator_ceiling :
  MAX_EXACT_T_SEMANTIC = 270549121 /\ MAX_EXACT_T < MAX_EXACT_T_SEMANTIC.
Proof.
  unfold MAX_EXACT_T_SEMANTIC, MAX_EXACT_T, I64MAX, PMAX, VABS.
  split; vm_compute; first [ reflexivity | discriminate ].
Qed.

(* ------------------------------------------------------------------ *)
(** ** The index model: 32-bit wrapping, and an UNSIGNED bound          *)
(* ------------------------------------------------------------------ *)

(** Every index the kernel computes goes through [mul.lo.s32] / [add.s32],
    which wrap; the bounds predicate is [setp.lt.u32], which reads the same
    register's bits as UNSIGNED; and the address is [cvt.u64.u32], a
    zero-extend of those bits.  So a wrapped-negative index is not a negative
    address -- it is a very large one, which is why it usually MASKS and
    occasionally aliases a live element. *)
Definition W (z : Z) : Z := MC.wrap32 z.
Definition u32_of (z : Z) : Z := if z <? 0 then z + 4294967296 else z.

(** [mul.lo.s32 %r9,%r7,%r2; add.s32 %r10,%r9,%r6; mul.lo.s32 %r11,%r10,%r0;
    add.s32 %r17,%r11,%r14] -- the weight row, then the step. *)
Definition p_index (b q Qx Tx t : Z) : Z := W (W (W (W (b * Qx) + q) * Tx) + t).

(** [mul.lo.s32 %r12,%r7,%r0; add.s32 %r23,%r12,%r14; mul.lo.s32 %r24,%r23,%r1;
    add.s32 %r25,%r24,%r8] -- the shared V row, which is where the wrap bites. *)
Definition v_index (b Tx Dx d t : Z) : Z := W (W (W (W (b * Tx) + t) * Dx) + d).

(** [mul.lo.s32 %r31,%r7,%r2; add.s32 %r32,%r31,%r6; mul.lo.s32 %r33,%r32,%r1;
    add.s32 %r34,%r33,%r8] -- the output. *)
Definition o_index (b q Qx Dx d : Z) : Z := W (W (W (W (b * Qx) + q) * Dx) + d).

(** A masked load: out of range contributes ZERO and the address is the
    UNSIGNED reading of the index bits. *)
Definition mload (arr : Z -> Z) (idx N : Z) : Z :=
  let u := u32_of idx in if u <? N then arr u else 0.

Lemma W_id : forall z, 0 <= z <= MC.I32MAX -> W z = z.
Proof.
  intros z [H0 H1]. unfold W. apply MC.wrap32_id.
  unfold MC.in_i32, MC.I32MIN in *. lia.
Qed.

Lemma u32_of_id : forall z, 0 <= z -> u32_of z = z.
Proof. intros z H. unfold u32_of. destruct (z <? 0) eqn:E; [ | reflexivity ].
  apply Z.ltb_lt in E. lia. Qed.

(** **Under the licence the emitted index IS the mathematical one.**  A single
    bound on the FINAL value suffices, because with [1 <= Dx] every earlier
    intermediate is between 0 and it. *)
Theorem the_v_index_is_the_mathematical_one : forall b Tx Dx d t,
  0 <= b -> 0 <= t -> 0 <= d -> 1 <= Dx -> 0 <= Tx ->
  (b * Tx + t) * Dx + d <= MC.I32MAX ->
  v_index b Tx Dx d t = (b * Tx + t) * Dx + d.
Proof.
  intros b Tx Dx d t Hb Ht Hd HD HT Hfit. unfold v_index.
  assert (H1 : W (b * Tx) = b * Tx) by (apply W_id; nia).
  rewrite H1.
  assert (H2 : W (b * Tx + t) = b * Tx + t) by (apply W_id; nia).
  rewrite H2.
  assert (H3 : W ((b * Tx + t) * Dx) = (b * Tx + t) * Dx) by (apply W_id; nia).
  rewrite H3.
  apply W_id; nia.
Qed.

Theorem the_p_index_is_the_mathematical_one : forall b q Qx Tx t,
  0 <= b -> 0 <= q -> 0 <= t -> 1 <= Tx -> 0 <= Qx ->
  (b * Qx + q) * Tx + t <= MC.I32MAX ->
  p_index b q Qx Tx t = (b * Qx + q) * Tx + t.
Proof.
  intros b q Qx Tx t Hb Hq Ht HT HQ Hfit. unfold p_index.
  assert (H1 : W (b * Qx) = b * Qx) by (apply W_id; nia). rewrite H1.
  assert (H2 : W (b * Qx + q) = b * Qx + q) by (apply W_id; nia). rewrite H2.
  assert (H3 : W ((b * Qx + q) * Tx) = (b * Qx + q) * Tx) by (apply W_id; nia).
  rewrite H3. apply W_id; nia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** THE FINDING: the index ceiling binds first                       *)
(* ------------------------------------------------------------------ *)

(** The largest [T] the V index survives, given a batch and a head dimension.
    [(B-1)*T + (T-1)] is the largest row and [D-1] the largest lane, so the
    largest index is [B*T*D - 1] and the bound is [B*T*D <= i32::MAX + 1]. *)
Definition MAX_T_INDEX (Bx Dx : Z) : Z := (MC.I32MAX + 1) / (Bx * Dx).

(** At the shapes this kernel is FOR -- [tests/ptx_exact_pv.rs] runs head_dim
    64 -- the undocumented index ceiling is 8.06x tighter than the documented
    accumulator one, and at a served batch it is 129x tighter. *)
Theorem the_index_ceiling_binds_first :
  MAX_T_INDEX 1 64 = 33554432
  /\ MAX_T_INDEX 8 128 = 2097152
  /\ MAX_T_INDEX 1 64 < MAX_EXACT_T_SEMANTIC
  /\ MAX_T_INDEX 8 128 < MAX_EXACT_T_SEMANTIC.
Proof.
  unfold MAX_T_INDEX, MAX_EXACT_T_SEMANTIC, MC.I32MAX, I64MAX.
  repeat split; vm_compute; first [ reflexivity | discriminate ].
Qed.

(** **The refutation, refereed against the silicon.**  [D = 2^30], [b = 0],
    [d = 0]: [t = 4] computes [4 * 2^30 = 2^32], which wraps to 0 -- the same
    index [t = 0] uses.  Both reproduced from [wrap32] alone. *)
Theorem the_measured_index_wrap_reads_v0_twice :
  v_index 0 8 1073741824 0 0 = 0
  /\ v_index 0 8 1073741824 0 4 = 0
  /\ (0 * 8 + 4) * 1073741824 + 0 = 4294967296.
Proof.
  unfold v_index, W, MC.wrap32, MC.I32MIN.
  repeat split; vm_compute; reflexivity.
Qed.

(** The device answered 6 where an unbounded-index machine answers 1, and this
    is that arithmetic: [P[t] = t+1], [V] one element, [NV = 1], [T = 5].  The
    wrapping model sums [t = 0] and [t = 4]; the mathematical one sums [t = 0]
    alone.  A theorem that merely said "the index wraps" would not distinguish
    a dropped term from a double-counted one, and the device shows the second. *)
Definition fixture_wrapped (t : nat) : Z :=
  (Z.of_nat t + 1) * mload (fun _ => 1) (v_index 0 8 1073741824 0 (Z.of_nat t)) 1.

Definition fixture_unbounded (t : nat) : Z :=
  (Z.of_nat t + 1) * (if (0 * 8 + Z.of_nat t) * 1073741824 + 0 <? 1 then 1 else 0).

Theorem the_measured_double_count_is_what_the_device_returned :
  sum_upto fixture_wrapped 5 = 6 /\ sum_upto fixture_unbounded 5 = 1.
Proof. split; vm_compute; reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** The output partition                                             *)
(* ------------------------------------------------------------------ *)

(** One thread per [(ctaid.y, ctaid.x, tid.x) = (b, q, d)], and the output
    index is [b*(Q*D) + q*D + d] -- a three-digit positional index, so
    [MixedRadix.two_digit_unique] carries injectivity with no new reasoning.
    This is the EIGHTH consumer of that schema and the first reached from a
    decode kernel's output map. *)
(** First the tie: under the licence the emitted output index IS
    [b*(Q*D) + q*D + d].  Without this the theorem below is a fact about an
    index shape that nothing says this kernel has. *)
Theorem the_output_index_is_the_mathematical_one : forall b q Qx Dx d,
  0 <= b -> 0 <= q -> 0 <= d -> 1 <= Dx -> 0 <= Qx ->
  (b * Qx + q) * Dx + d <= MC.I32MAX ->
  o_index b q Qx Dx d = b * (Qx * Dx) + q * Dx + d.
Proof.
  intros b q Qx Dx d Hb Hq Hd HD HQ Hfit. unfold o_index.
  assert (H1 : W (b * Qx) = b * Qx) by (apply W_id; nia). rewrite H1.
  assert (H2 : W (b * Qx + q) = b * Qx + q) by (apply W_id; nia). rewrite H2.
  assert (H3 : W ((b * Qx + q) * Dx) = (b * Qx + q) * Dx) by (apply W_id; nia).
  rewrite H3.
  assert (H4 : W ((b * Qx + q) * Dx + d) = (b * Qx + q) * Dx + d)
    by (apply W_id; nia).
  rewrite H4. ring.
Qed.

Theorem every_output_element_is_written_by_one_thread :
  forall Qx Dx b1 q1 d1 b2 q2 d2,
    (0 < Dx)%nat -> (0 < Qx)%nat ->
    (q1 < Qx)%nat -> (q2 < Qx)%nat -> (d1 < Dx)%nat -> (d2 < Dx)%nat ->
    (b1 * (Qx * Dx) + q1 * Dx + d1 = b2 * (Qx * Dx) + q2 * Dx + d2)%nat ->
    b1 = b2 /\ q1 = q2 /\ d1 = d2.
Proof.
  intros Qx Dx b1 q1 d1 b2 q2 d2 HD HQ Hq1 Hq2 Hd1 Hd2 Heq.
  apply (MR.two_digit_unique Dx Qx b1 q1 d1 b2 q2 d2); assumption.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The capstone                                                     *)
(* ------------------------------------------------------------------ *)

(** The source's meaning: [out[b][q][d] = sum over t of P[(b*Q+q)*T+t] *
    V[(b*T+t)*D+d]], with no wrapping anywhere and every access in range. *)
Definition src_term (Pf Vf : Z -> Z) (b q Qx Tx Dx d : Z) (t : nat) : Z :=
  Pf ((b * Qx + q) * Tx + Z.of_nat t) * Vf ((b * Tx + Z.of_nat t) * Dx + d).

(** What the kernel runs: wrapped indices, masked loads, wrapping products. *)
Definition emitted_term (Pf Vf : Z -> Z) (b q Qx Tx Dx d NP NV : Z) (t : nat) : Z :=
  wrap64 (mload Pf (p_index b q Qx Tx (Z.of_nat t)) NP
          * mload Vf (v_index b Tx Dx d (Z.of_nat t)) NV).

(** **THE CAPSTONE.**  Under the two licences the emitted loop's accumulator
    holds the source dot product exactly.

    [Hidx] is the index licence -- the one the source's header does not state.
    [Hacc] is the accumulator licence -- the one it does.  [Hin] says the
    accesses the source intends are in range, which is what makes the masks
    inert; [Hdom] is the operand domain the load instructions permit. *)
Theorem the_emitted_exact_pv_holds_the_source_dot_product :
  forall Pf Vf b q Qx Tx Dx d NP NV (n : nat),
    0 <= b -> 0 <= q -> 0 <= d -> 1 <= Dx -> 1 <= Tx -> 0 <= Qx ->
    Z.of_nat n <= Tx ->
    (forall z, p_domain (Pf z)) -> (forall z, v_domain (Vf z)) ->
    (* Hidx: every index this kernel computes stays inside i32 *)
    (forall t, (t < n)%nat -> (b * Qx + q) * Tx + Z.of_nat t <= MC.I32MAX) ->
    (forall t, (t < n)%nat -> (b * Tx + Z.of_nat t) * Dx + d <= MC.I32MAX) ->
    (* Hin: the accesses the source intends are in range *)
    (forall t, (t < n)%nat -> (b * Qx + q) * Tx + Z.of_nat t < NP) ->
    (forall t, (t < n)%nat -> (b * Tx + Z.of_nat t) * Dx + d < NV) ->
    (* Hacc: the accumulator licence *)
    Z.of_nat n * (PMAX * VABS) <= I64MAX ->
    wacc (emitted_term Pf Vf b q Qx Tx Dx d NP NV) n
      = sum_upto (src_term Pf Vf b q Qx Tx Dx d) n.
Proof.
  intros Pf Vf b q Qx Tx Dx d NP NV n Hb Hq Hd HD HT HQ Hn HP HV Hix Hiv Hnp Hnv Hacc.
  (* Step 1: on [0..n) the emitted term IS the source term. *)
  assert (Hterm : forall t, (t < n)%nat ->
            emitted_term Pf Vf b q Qx Tx Dx d NP NV t
            = src_term Pf Vf b q Qx Tx Dx d t).
  { intros t Ht. unfold emitted_term, src_term.
    assert (Hpt : 0 <= Z.of_nat t) by apply Nat2Z.is_nonneg.
    rewrite (the_p_index_is_the_mathematical_one b q Qx Tx (Z.of_nat t))
      by (auto; apply Hix; lia).
    rewrite (the_v_index_is_the_mathematical_one b Tx Dx d (Z.of_nat t))
      by (auto; try lia; apply Hiv; lia).
    unfold mload.
    rewrite (u32_of_id ((b * Qx + q) * Tx + Z.of_nat t)) by nia.
    rewrite (u32_of_id ((b * Tx + Z.of_nat t) * Dx + d)) by nia.
    assert (E1 : ((b * Qx + q) * Tx + Z.of_nat t <? NP) = true)
      by (apply Z.ltb_lt; apply Hnp; lia).
    assert (E2 : ((b * Tx + Z.of_nat t) * Dx + d <? NV) = true)
      by (apply Z.ltb_lt; apply Hnv; lia).
    rewrite E1, E2.
    apply the_product_never_wraps; auto. }
  (* Step 2: the wrapping fold over those terms is the plain sum. *)
  assert (Hfold : wacc (emitted_term Pf Vf b q Qx Tx Dx d NP NV) n
                  = sum_upto (emitted_term Pf Vf b q Qx Tx Dx d NP NV) n).
  { apply (bounded_terms_accumulate_exactly _ _ (PMAX * VABS)).
    - unfold PMAX, VABS. lia.
    - intros k. unfold emitted_term.
      assert (Hm : forall (arr : Z -> Z) idx N,
                (forall z, p_domain (arr z)) -> 0 <= mload arr idx N <= PMAX).
      { intros arr idx N Hdm. unfold mload.
        destruct (u32_of idx <? N); [ apply Hdm | unfold PMAX; lia ]. }
      assert (Hm2 : forall (arr : Z -> Z) idx N,
                (forall z, v_domain (arr z)) -> -VABS <= mload arr idx N <= VABS - 1).
      { intros arr idx N Hdm. unfold mload.
        destruct (u32_of idx <? N); [ apply Hdm | unfold VABS; lia ]. }
      specialize (Hm Pf (p_index b q Qx Tx (Z.of_nat k)) NP HP).
      specialize (Hm2 Vf (v_index b Tx Dx d (Z.of_nat k)) NV HV).
      rewrite the_product_never_wraps
        by (unfold p_domain, v_domain; lia).
      rewrite Z.abs_mul.
      assert (Z.abs (mload Pf (p_index b q Qx Tx (Z.of_nat k)) NP) <= PMAX)
        by (rewrite Z.abs_eq by lia; lia).
      assert (Z.abs (mload Vf (v_index b Tx Dx d (Z.of_nat k)) NV) <= VABS)
        by (apply Z.abs_le; lia).
      assert (0 <= Z.abs (mload Pf (p_index b q Qx Tx (Z.of_nat k)) NP)) by apply Z.abs_nonneg.
      assert (0 <= Z.abs (mload Vf (v_index b Tx Dx d (Z.of_nat k)) NV)) by apply Z.abs_nonneg.
      nia.
    - exact Hacc. }
  rewrite Hfold. clear Hfold.
  (* Step 3: pointwise equality lifts to the sums. *)
  assert (Hlift : forall m, (m <= n)%nat ->
            sum_upto (emitted_term Pf Vf b q Qx Tx Dx d NP NV) m
            = sum_upto (src_term Pf Vf b q Qx Tx Dx d) m).
  { induction m; intros Hm; [ reflexivity | ].
    simpl. rewrite IHm by lia. rewrite Hterm by lia. reflexivity. }
  apply Hlift; lia.
Qed.

(** **Without the index licence the capstone is FALSE**, and this is the
    instance the device produced: the same fixture, one iteration apart.  A
    licence nothing can violate certifies nothing. *)
Theorem without_the_index_licence_the_answer_is_wrong :
  sum_upto fixture_wrapped 5 <> sum_upto fixture_unbounded 5.
Proof. vm_compute. discriminate. Qed.

(** The sum of a constant, so the accumulator boundary can be stated about
    [sum_upto] itself rather than about a 16-million-step [vm_compute]. *)
Lemma sum_const : forall c n, sum_upto (fun _ => c) n = Z.of_nat n * c.
Proof.
  induction n; [ reflexivity | ].
  simpl sum_upto. rewrite IHn, Nat2Z.inj_succ. lia.
Qed.

(** **Without the accumulator licence the capstone is FALSE too**, at its own
    boundary and one iteration wide.  What breaks is precisely
    [acc_exact_in_int64]'s hypothesis: the running total leaves int64.  The
    admissible case beside it is the control - without it this reads as "the
    fold is broken" rather than "the licence is what makes it exact".

    Stated over an ABSTRACT [n1]/[n2] rather than [Z.to_nat 16777216].  A
    [nat] literal is unary, so writing the length down directly puts sixteen
    million constructors in the proof term and [Qed] does not return -- the
    landmine CLAUDE.md records from [SoftmaxErrorBound.v], hit here and
    bisected to this theorem.  [the_loop_lengths_exist] is what stops the
    hypothesis reading as vacuous. *)
Theorem the_accumulator_licence_is_load_bearing : forall n1 n2 : nat,
  Z.of_nat n1 = 16777216 -> Z.of_nat n2 = 16777217 ->
  sum_upto (fun _ => -(PMAX * VABS)) n1 = -9223372034707292160
  /\ sum_upto (fun _ => -(PMAX * VABS)) n2 = -9223372584463105920
  /\ in_i64 (-9223372034707292160)
  /\ ~ in_i64 (-9223372584463105920).
Proof.
  intros n1 n2 H1 H2. repeat split.
  - rewrite sum_const, H1. unfold PMAX, VABS. vm_compute. reflexivity.
  - rewrite sum_const, H2. unfold PMAX, VABS. vm_compute. reflexivity.
  - unfold in_i64, I64MIN, I64MAX. lia.
  - unfold in_i64, I64MIN, I64MAX. lia.
  - unfold in_i64, I64MIN, I64MAX. intros [Ha Hb]. lia.
Qed.

(** The two hypotheses above are satisfiable, so the theorem is not vacuous.
    Proved through [Z2Nat.id], which is a LEMMA rather than a computation -- it
    exhibits the witness without ever building it. *)
Theorem the_loop_lengths_exist :
  (exists n : nat, Z.of_nat n = 16777216) /\ (exists n : nat, Z.of_nat n = 16777217).
Proof.
  split; [ exists (Z.to_nat 16777216) | exists (Z.to_nat 16777217) ];
    apply Z2Nat.id; lia.
Qed.

Print Assumptions acc_exact_in_int64.
Print Assumptions bounded_terms_accumulate_exactly.
Print Assumptions the_product_never_wraps.
Print Assumptions the_accumulator_bound_is_one_iteration_wide.
Print Assumptions the_measured_overflow_is_two_s_complement.
Print Assumptions the_semantic_domain_raises_the_accumulator_ceiling.
Print Assumptions the_v_index_is_the_mathematical_one.
Print Assumptions the_p_index_is_the_mathematical_one.
Print Assumptions the_index_ceiling_binds_first.
Print Assumptions the_measured_index_wrap_reads_v0_twice.
Print Assumptions the_measured_double_count_is_what_the_device_returned.
Print Assumptions the_output_index_is_the_mathematical_one.
Print Assumptions every_output_element_is_written_by_one_thread.
Print Assumptions the_emitted_exact_pv_holds_the_source_dot_product.
Print Assumptions without_the_index_licence_the_answer_is_wrong.
Print Assumptions the_accumulator_licence_is_load_bearing.
Print Assumptions the_loop_lengths_exist.
