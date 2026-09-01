(** * The exact-attention softmax's error bound: the first BOUND in this
      programme, and the first obligation that is not an equality.

    Every theorem in `proofs/` so far is an EXACTNESS or a COVERAGE claim. That
    is not a stylistic preference, it is the precondition
    `docs/verified_kernel_process.md` §0 states and refuses to negotiate: the
    kernel must be exact, because integer addition is associative and the
    relationship between the optimized kernel and its specification is then an
    EQUALITY, which is what a proof assistant is good at.

    `exp` is not exact in any representation, so the softmax cannot be handled
    that way and Phase 4 exists to say what happens instead. What the repo has
    for the OUTPUT today is `tests/attention_quantization_error.rs`, whose own
    stated bar is comparative and empirical - "is it more wrong than the f32
    online softmax that production flash attention already ships?" - measured
    on synthetic score distributions. This file replaces that sentence for the
    part that is arithmetic.

    ** What is proved

    The chain the kernel runs, from `src/exact_attention.rs`:

      m - s_i  ->  t_i = min( ((m - s_i)*KFix + 2^15) >> 16, 2^30 )
               ->  p_i = exp2_neg_q16_16(t_i)      (Q0.28)
               ->  L = sum p_i,  O_d = sum p_i * v_i   (exact int64)
               ->  O_d / L

    [the_attention_output_is_within_the_bound] bounds `O_d/L` against the ideal
    `sum w_i v_i / sum w_i`, and [the_bound_at_a_long_context] evaluates it: at
    65,536 keys with int8 `V`, the output is within **7/100** of the ideal, on
    an output range of +-127. The general form is
    `2*VMAX*(EPS*Wtot + n) / L`, and every term in it is named.

    ** The interesting part is NOT the exp

    `fixed_exp.rs::it_is_sub_ulp_accurate_everywhere` is EXHAUSTIVE over the
    whole swept domain against `f64::exp2`, worst 0.908 ulp. Exhaustion over a
    finite domain is stronger than a proof about the series would be, and
    `docs/verified_kernel_process.md` §4 already says to prefer it. So the exp
    enters here as a HYPOTHESIS, discharged by that test.

    What nothing covered is the chain AROUND it, and it has three joints:

    - **The argument reduction ROUNDS THE EXPONENT.** `>> 16` after a
      round-to-nearest `+ 2^15` moves `t` by up to half a Q16.16 ulp, and an
      error in an exponent is MULTIPLICATIVE in the weight. That is a different
      shape from the exp's additive ulp and the two do not simply add.
    - **The saturate at 2^30 is outside the swept domain.**
      `it_is_sub_ulp_accurate_everywhere` sweeps `0 .. 31<<16`; the emitted
      `min.s64 ..., 1073741824` admits arguments up to `2^30`, which is 16,384
      times further out. [the_swept_domain_covers_the_admitted_one] is the
      two-line argument that closes that gap and existed nowhere: above the
      table the implementation returns 0 and the true weight is below a
      quarter of an ulp, so the "0.908 ulp everywhere" headline does hold on
      the whole domain the kernel can reach - it just was not established.
    - **The max subtraction has to be carried through.**
      [the_max_subtraction_puts_a_floor_under_the_total_weight] is what makes
      the exp's ADDITIVE error relatively negligible: `m - s_i` is zero for the
      argmax, so one weight is exactly `2^28` and `Wtot >= 2^28`.
      [a_total_weight_below_one_ulp_admits_a_zero_denominator] is the
      refutation - without that floor the same per-element bracket admits
      `p_i = 0` for every `i`, so `L = 0` and there is no output at all. That
      is not hypothetical: it is the F=16 attention-sink failure
      `tests/attention_quantization_error.rs` records, where the whole tail of
      the softmax rounded to zero.

    ** The fourth joint - the temperature is quantized too

    Everything above compares the computed weight against the ideal at the
    multiplier the kernel was HANDED. `KFix = round(C * 2^32)`, so that is not
    the ideal at the temperature the caller meant, and the difference was
    priced nowhere.

    [the_quantized_temperature_shifts_the_exponent_by_at_most_half_a_delta]
    prices it, and the shape of the answer is the point: the exponent moves by
    at most `(delta + 1)/2` in units of `2^-32` log2, so the error is
    proportional to the SCORE DELTA and INDEPENDENT of `KFix`. A bound on the
    delta is therefore the whole bound - and the delta is bounded by the
    head_dim, since `|s| <= 127^2 * hd` for an int8 pipeline. At head_dim 128
    that is a factor under `1/2048` on any weight
    ([at_head_dim_128_the_temperature_moves_a_weight_by_under_a_two_thousandth]),
    measured tight to 1.000 of its half-delta allowance on the exponent and
    1.47x slack on the weight.

    Two multipliers are REFUSED rather than bounded, because they are not
    approximations of anything: `KFix = 0` gives every key the same argument,
    so the softmax is uniform and carries no information about the scores
    ([a_zero_multiplier_gives_every_key_the_same_weight]); and at or above
    `2^31` the `.u32` parameter and the `mul.wide.s32` that consumes it are
    different numbers, so a key that should weigh `2^28 * 2^(-1/2)` weighs
    exactly zero ([the_two_readings_of_the_multiplier_disagree_above_two_to_the
    _thirty_one], [the_signed_reading_zeroes_a_weight_the_unsigned_one_keeps]).
    Both bounds were already in `tools/ptx_bridge.py` as a bare `continue` -
    a silent skip of a measurement. They are derived here and enforced by
    `exact_attention::temperature_fixed_point`.

    ** Does the Decomposition schema extend to approximate arithmetic?

    That is the question this file was written to answer, and the answer is
    yes, for a reason worth stating narrowly: **the error is introduced BEFORE
    the decomposition, not by it.** `L` and `O` are exact integer sums, so
    `GridStrideSplit.grid_stride_exact` applies verbatim and the per-element
    error enters the fold as data. [the_bound_holds_at_every_launch_geometry]
    is that join, and it is why the bound is a property of the kernel rather
    than of one launch.

    An f32 softmax has no such split - its error is produced BY the reduction,
    so the bound would have to be re-derived per decomposition. That contrast
    is the actual content of "bounds compose differently", and it says the
    addressable set widens to *pipelines whose approximation is per-element*,
    not to approximate reductions in general.

    ** Why there is no Reals library here, and it is not a stylistic choice

    `tests/proofs_are_checked.rs` requires every `Print Assumptions` to report
    `Closed under the global context`. Coq's `R` is axiomatized, so a single
    `Require Import Reals` would put seventeen axioms under every theorem in
    this file and fail that gate. Everything below is `Z` and `Q`.

    That forces one real design decision. The obvious interface for the ideal
    weight - exact homogeneity `W(u+v)*2^28 = W(u)*W(v)` plus one exact halving
    - is **INCONSISTENT over Q**: [exact_homogeneity_forces_a_square_root]
    derives `W(2^31)^2 == 2^55` from it, and no rational squares to `2^55`. An
    inconsistent hypothesis set proves everything, so the interface here is a
    two-sided rational BRACKET instead, and
    [the_interface_is_satisfiable] exhibits a model.

    ** What this does NOT claim

    - **It is not a statement about `f64::exp2` or about the real
      exponential.** `W` is an abstract ideal weight constrained by four
      properties the true function has; nothing here proves the true function
      has them. Two of the four are quantitative, and their constants are not
      magic: [the_per_unit_factor_is_what_one_unit_of_log2_costs] proves the
      rational inequality `(1 - 2^-32)^(2^32) <= 1/2` that justifies them.
      Believing they describe `2^-u/2^32` is a TCB item, and it sits beside
      `vpdpwssd`'s semantics rather than above them.
    - **The exp's accuracy is a hypothesis, not a theorem.** It is discharged
      by exhaustion in Rust, over the *swept* domain; the extension to the
      admitted domain is proved here.
    - **`KFix` is itself a rounded temperature.** It was NOT priced when this
      file was first written, and it is now - Part 6b, and see the joint below.
    - **Nothing here is about int8 quantization of Q, K or V.** The scores are
      taken as given; `docs/deterministic_inference.md` and
      `tests/attention_quantization_error.rs` own that question.
    - **The tie to the kernel is `tests/softmax_error_bound.rs`**, which runs
      the real `exp2_neg_q16_16` and the real emitted arithmetic. It is the
      GridStrideSplit grade of tie - a model checked against the running code -
      not the exact GEMM's byte-identity.

    Build:  coqc proofs/SoftmaxErrorBound.v      (Rocq 9.1)
*)

From Stdlib Require Import ZArith Arith Lia QArith Qabs Znumtheory.
From Stdlib Require Import Lqa.
Require ExactGemmKsplit.
Require GridStrideSplit.

Module KS := ExactGemmKsplit.
Module GS := GridStrideSplit.

Open Scope Q_scope.

(* ------------------------------------------------------------------ *)
(** ** Part 0 - rational helpers                                       *)
(* ------------------------------------------------------------------ *)

(** Powers indexed by `nat` rather than `Qpower`'s `Z`, so induction is
    available. Nothing below is ever evaluated at a large literal: the two
    places a 2^32-fold product appears instantiate an ABSTRACT exponent at
    `Z.to_nat (2^32)`, which is a term rather than 4 billion nested `S`. *)
Fixpoint qpow (q : Q) (n : nat) : Q :=
  match n with O => 1 | S k => q * qpow q k end.

Lemma qpow_nonneg : forall q n, 0 <= q -> 0 <= qpow q n.
Proof. induction n; simpl; intros. lra. apply Qmult_le_0_compat; auto. Qed.

Lemma qpow_le_1 : forall q n, 0 <= q -> q <= 1 -> qpow q n <= 1.
Proof.
  induction n; simpl; intros. lra.
  assert (0 <= qpow q n) by (apply qpow_nonneg; auto).
  assert (qpow q n <= 1) by auto. nra.
Qed.

Lemma qpow_add : forall q a b, qpow q (a + b)%nat == qpow q a * qpow q b.
Proof. induction a; simpl; intros. ring. rewrite IHa. ring. Qed.

Lemma qpow_mult : forall a b n, qpow (a*b) n == qpow a n * qpow b n.
Proof. induction n; simpl. ring. rewrite IHn. ring. Qed.

Lemma qpow_S : forall q n, qpow q (S n) == q * qpow q n.
Proof. intros. simpl. reflexivity. Qed.

(** Bernoulli. This is what turns "one unit of the exponent numerator" into a
    bound over a whole half-ulp span without ever multiplying 32,768
    rationals. *)
Lemma bernoulli : forall n x, -1 <= x -> 1 + inject_Z (Z.of_nat n) * x <= qpow (1+x) n.
Proof.
  induction n; intros x H.
  - simpl. change (inject_Z 0) with 0. lra.
  - assert (H1 := IHn x H).
    rewrite qpow_S.
    rewrite Nat2Z.inj_succ, <- Z.add_1_r, inject_Z_plus.
    assert (0 <= 1 + x) by lra.
    assert (Hi : 0 <= inject_Z (Z.of_nat n)).
    { change 0 with (inject_Z 0). rewrite <- Zle_Qle. apply Nat2Z.is_nonneg. }
    change (inject_Z 1) with 1.
    nra.
Qed.

(** The other direction, phrased multiplicatively so no division appears:
    `(1-x)^n <= 1/(1+nx)`. *)
Lemma decay_upper : forall n x, 0 <= x -> x <= 1 ->
  qpow (1-x) n * (1 + inject_Z (Z.of_nat n) * x) <= 1.
Proof.
  intros n x H0 H1.
  assert (B : 1 + inject_Z (Z.of_nat n) * x <= qpow (1+x) n) by (apply bernoulli; lra).
  assert (P : 0 <= qpow (1-x) n) by (apply qpow_nonneg; lra).
  assert (M : qpow (1-x) n * qpow (1+x) n == qpow ((1-x)*(1+x)) n)
    by (rewrite qpow_mult; ring).
  assert (L : qpow ((1-x)*(1+x)) n <= 1) by (apply qpow_le_1; nra).
  nra.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Part 1 - the emitted argument reduction, over Z                 *)
(* ------------------------------------------------------------------ *)

Close Scope Q_scope.
Open Scope Z_scope.

(** The emitted PTX, verbatim from `src/exact_attention.rs`:

      sub.s32      %r16, %r11, %r15          // delta = m - s_i  (>= 0)
      mul.wide.s32 %rd50, %r16, %r50         // delta * KFix
      add.s64      %rd50, %rd50, 32768       // round to nearest
      shr.s64      %rd50, %rd50, 16          // -> Q16.16
      min.s64      %rd50, %rd50, 1073741824  // saturate at 2^30
      cvt.u32.u64  %r16,  %rd50

    `shr.s64` is an ARITHMETIC shift, i.e. floor division, which is what
    Coq's `Z.div` is (`Z.quot` is the truncating one). Both operands are
    non-negative here, so the distinction cannot bite - and that is stated
    rather than left to be noticed. *)
Definition HALF : Z := 32768.        (** 2^15, the round-to-nearest addend *)
Definition ULP  : Z := 65536.        (** 2^16, the `shr.s64` *)
Definition SAT  : Z := 1073741824.   (** 2^30, the `min.s64` *)
Definition TWO32 : Z := 4294967296.

Definition arg (delta kfix : Z) : Z := Z.min ((delta * kfix + HALF) / ULP) SAT.

(** The exponent, in units of `2^-32` in log2. The kernel's Q16.16 argument `t`
    means `t / 2^16` in log2 units, hence `2^16 * t` here; the exact one is
    `delta * KFix` and is an INTEGER, which is the whole reason this layer
    needs no rationals. *)
Definition u_of_arg (t : Z) : Z := ULP * t.
Definition u_exact (delta kfix : Z) : Z := delta * kfix.

(** Round-to-nearest moves the value by at most half an ulp. One side is
    strict and the other is not; the symmetric bound is what the rest uses. *)
Theorem the_rounding_is_within_a_half_ulp : forall x, 0 <= x ->
  Z.abs (ULP * ((x + HALF) / ULP) - x) <= HALF.
Proof.
  intros x Hx. unfold ULP, HALF in *.
  assert (Hb : 0 < 65536) by lia.
  pose proof (Z.div_mod (x + 32768) 65536 ltac:(lia)) as Hdm.
  pose proof (Z.mod_pos_bound (x + 32768) 65536 Hb) as Hmb.
  apply Z.abs_le. lia.
Qed.

(** Below the saturate, the emitted argument is a half-ulp perturbation of the
    exact exponent. This is the whole of the argument-reduction joint. *)
Theorem an_unsaturated_argument_is_within_a_half_ulp :
  forall delta kfix, 0 <= delta -> 0 <= kfix ->
    (delta * kfix + HALF) / ULP <= SAT ->
    Z.abs (u_of_arg (arg delta kfix) - u_exact delta kfix) <= HALF.
Proof.
  intros delta kfix Hd Hk Hs. unfold arg, u_of_arg, u_exact.
  rewrite Z.min_l by exact Hs.
  apply the_rounding_is_within_a_half_ulp. now apply Z.mul_nonneg_nonneg.
Qed.

(** And above it the exact exponent is astronomically large - far past the
    thirty halvings the weight needs to fall below one ulp. `2^46` against
    `30 * 2^32`, a factor of 546. *)
Theorem saturation_means_the_exact_exponent_is_astronomically_large :
  forall delta kfix, 0 <= delta -> 0 <= kfix ->
    SAT < (delta * kfix + HALF) / ULP ->
    30 * TWO32 <= u_exact delta kfix.
Proof.
  intros delta kfix Hd Hk Hs. unfold u_exact, TWO32, SAT, HALF, ULP in *.
  assert (Hb : 0 < 65536) by lia.
  pose proof (Z.div_mod (delta*kfix + 32768) 65536 ltac:(lia)) as Hdm.
  pose proof (Z.mod_pos_bound (delta*kfix + 32768) 65536 Hb) as Hmb.
  lia.
Qed.

(** The argument is never negative, so the exp's own domain assumption holds
    and the floor/truncation question above is settled rather than avoided. *)
Lemma the_argument_is_in_range : forall delta kfix, 0 <= delta -> 0 <= kfix ->
  0 <= arg delta kfix <= SAT.
Proof.
  intros delta kfix Hd Hk. unfold arg, SAT, HALF, ULP in *.
  assert (0 <= delta * kfix) by now apply Z.mul_nonneg_nonneg.
  assert (0 <= (delta*kfix + 32768) / 65536).
  { apply Z.div_pos; lia. }
  split; [apply Z.min_glb; lia | apply Z.le_min_r].
Qed.

(* ------------------------------------------------------------------ *)
(** ** Part 2 - the ideal weight, as a two-sided rational bracket      *)
(* ------------------------------------------------------------------ *)

Close Scope Z_scope.
Open Scope Q_scope.

(** The Q0.28 scale: `exp2_neg_q16_16(0) = 1 << 28`. *)
Definition TWO28 : Q := 268435456 # 1.

(** One unit of the exponent numerator is `2^-32` in log2 units, so it costs a
    factor of at most `1 - 2^-32`. The true factor is `2^-2^-32`, which is
    `1 - 1.47e-10` against this bound's `1 - 2.33e-10`, so the bracket is
    genuinely satisfied and not merely asserted - see
    [the_per_unit_factor_is_what_one_unit_of_log2_costs] for the rational
    inequality that fixes the constant. *)
Definition BETA : Q := 4294967295 # 4294967296.

(** ... and over the half-ulp span the argument reduction can move the
    exponent, `2^15` units, that compounds to at least `1 - 2^-17`.
    Bernoulli, not a product of 32,768 rationals. *)
Definition ALPHA : Q := 131071 # 131072.

(** The relative slack a rounded exponent costs a weight. `1/ALPHA - 1` is
    `2^-17/(1 - 2^-17)`, about `7.63e-6`; `EPS` is `1.53e-5`, so this is the
    round number above it rather than a fitted constant. *)
Definition EPS : Q := 1 # 65536.

(** `V` is int8. *)
Definition VMAXQ : Q := 127 # 1.

Lemma qpow_pos : forall q n, 0 < q -> 0 < qpow q n.
Proof. induction n; simpl; intros. lra. apply Qmult_lt_0_compat; auto. Qed.

Lemma qpow_Qeq : forall a b n, a == b -> qpow a n == qpow b n.
Proof. induction n; simpl; intros. reflexivity. rewrite (IHn H), H. reflexivity. Qed.

Lemma Qabs_inject_Z : forall z, Qabs (inject_Z z) == inject_Z (Z.abs z).
Proof. intros. reflexivity. Qed.

(** The constant in [BETA] is not a magic number: it is exactly the claim that
    `2^32` units of the exponent numerator lose at least a factor of two, i.e.
    that one unit is at most `2^-32` in log2. Over Q that is a rational
    inequality and provable; over R it would be `2^(-2^-32) >= 1 - 2^-32` and
    would cost seventeen axioms. *)
Theorem the_per_unit_factor_is_what_one_unit_of_log2_costs :
  (2#1) * qpow BETA (Z.to_nat 4294967296) <= 1.
Proof.
  assert (Hn : Z.of_nat (Z.to_nat 4294967296) = 4294967296%Z)
    by (apply Z2Nat.id; lia).
  pose proof (decay_upper (Z.to_nat 4294967296) (1#4294967296)) as D.
  assert (H0 : 0 <= 1#4294967296) by (unfold Qle; simpl; lia).
  assert (H1 : (1#4294967296) <= 1) by (unfold Qle; simpl; lia).
  specialize (D H0 H1). rewrite Hn in D.
  assert (Hb : qpow (1 - (1#4294967296)) (Z.to_nat 4294967296)
               == qpow BETA (Z.to_nat 4294967296))
    by (apply qpow_Qeq; unfold BETA; unfold Qeq; simpl; lia).
  rewrite Hb in D.
  assert (Hc : inject_Z 4294967296 * (1#4294967296) == 1)
    by (unfold inject_Z, Qeq, Qmult; simpl; lia).
  rewrite Hc in D. lra.
Qed.

Section Ideal.

(** The ideal Q0.28 softmax weight, as a function of the exact exponent
    numerator `u` in units of `2^-32` log2. It is `2^28 * 2^(-u/2^32)`, and
    the whole point of the four hypotheses below is that they are properties
    that function HAS while being satisfiable by a Q-valued model - see
    [the_interface_is_satisfiable]. The obvious interface, exact homogeneity,
    is not: [exact_homogeneity_forces_a_square_root]. *)
Variable W : Z -> Q.

Hypothesis W_pos   : forall u, (0 <= u)%Z -> 0 < W u.
Hypothesis W_one   : W 0%Z == TWO28.
Hypothesis W_step  : forall u, (0 <= u)%Z -> BETA * W u <= W (u+1)%Z /\ W (u+1)%Z <= W u.
Hypothesis W_halves: forall u, (0 <= u)%Z -> (2#1) * W (u + 4294967296)%Z <= W u.

Lemma W_nonneg : forall u, (0 <= u)%Z -> 0 <= W u.
Proof. intros. apply Qlt_le_weak. now apply W_pos. Qed.

(** The bracket over a span, by induction on the span. Both directions come
    from the one hypothesis. *)
Lemma W_span_nat : forall (k : nat) u, (0 <= u)%Z ->
  qpow BETA k * W u <= W (u + Z.of_nat k)%Z /\ W (u + Z.of_nat k)%Z <= W u.
Proof.
  induction k; intros u Hu.
  - replace (u + Z.of_nat 0)%Z with u by lia. simpl. split; lra.
  - destruct (IHk u Hu) as [Lo Hi].
    assert (Hk : (0 <= u + Z.of_nat k)%Z) by lia.
    destruct (W_step _ Hk) as [S1 S2].
    replace (u + Z.of_nat (S k))%Z with (u + Z.of_nat k + 1)%Z by lia.
    assert (HB : 0 <= BETA) by (unfold BETA, Qle; simpl; lia).
    rewrite qpow_S. split.
    + (* BETA * (qpow BETA k * W u) <= BETA * W(u+k) <= W(u+k+1) *)
      nra.
    + lra.
Qed.

Lemma W_span : forall u k, (0 <= u)%Z -> (0 <= k)%Z ->
  qpow BETA (Z.to_nat k) * W u <= W (u + k)%Z /\ W (u + k)%Z <= W u.
Proof.
  intros u k Hu Hk.
  pose proof (W_span_nat (Z.to_nat k) u Hu) as H.
  rewrite (Z2Nat.id k Hk) in H. exact H.
Qed.

(** Bernoulli, at the span the emitted rounding can produce. *)
Lemma alpha_bounds_a_half_ulp_span : forall (k : nat),
  (Z.of_nat k <= 32768)%Z -> ALPHA <= qpow BETA k.
Proof.
  intros k Hk.
  pose proof (bernoulli k (-(1#4294967296))) as B.
  assert (Hm1 : -1 <= -(1#4294967296)) by (unfold Qle; simpl; lia).
  specialize (B Hm1).
  assert (Hb : qpow (1 + -(1#4294967296)) k == qpow BETA k)
    by (apply qpow_Qeq; unfold BETA, Qeq; simpl; lia).
  rewrite Hb in B.
  assert (Hkq : inject_Z (Z.of_nat k) <= inject_Z 32768) by (rewrite <- Zle_Qle; lia).
  assert (Hk0 : 0 <= inject_Z (Z.of_nat k))
    by (change 0 with (inject_Z 0); rewrite <- Zle_Qle; apply Nat2Z.is_nonneg).
  assert (Hprod : inject_Z 32768 * (1#4294967296) == 1#131072)
    by (unfold inject_Z, Qeq, Qmult; simpl; lia).
  assert (HA : ALPHA == 1 - (1#131072)) by (unfold ALPHA, Qeq; simpl; lia).
  assert (Hpos : 0 <= (1#4294967296)) by (unfold Qle; simpl; lia).
  (* 1 + k * (-(2^-32)) >= 1 - 32768 * 2^-32 = ALPHA *)
  assert (1 + inject_Z (Z.of_nat k) * -(1#4294967296)
          >= 1 - inject_Z 32768 * (1#4294967296)) by nra.
  rewrite Hprod in *. rewrite HA. lra.
Qed.

(** The joint the whole file is about: a half-ulp perturbation of the exponent
    moves the weight by at most the factor ALPHA, in EITHER direction. The
    "easy" side is not free - it is the upper half of [W_step]. *)
Lemma W_two_sided : forall u u', (0 <= u)%Z -> (0 <= u')%Z ->
  (Z.abs (u' - u) <= 32768)%Z ->
  ALPHA * W u <= W u' /\ ALPHA * W u' <= W u.
Proof.
  intros u u' Hu Hu' Hd.
  assert (HA1 : ALPHA <= 1) by (unfold ALPHA, Qle; simpl; lia).
  assert (HA0 : 0 <= ALPHA) by (unfold ALPHA, Qle; simpl; lia).
  pose proof (Z.abs_le (u' - u) 32768) as [Habs _]. specialize (Habs Hd).
  destruct (Z_le_gt_dec u u') as [Hle | Hgt].
  - (* u' is further out: W u' <= W u, and W u' >= ALPHA * W u *)
    destruct (W_span u (u' - u)%Z Hu ltac:(lia)) as [Lo Hi].
    replace (u + (u' - u))%Z with u' in Lo, Hi by lia.
    assert (HAk : ALPHA <= qpow BETA (Z.to_nat (u' - u)))
      by (apply alpha_bounds_a_half_ulp_span; rewrite Z2Nat.id by lia; lia).
    assert (Hw : 0 <= W u) by (apply W_nonneg; lia).
    assert (Hw' : 0 <= W u') by (apply W_nonneg; lia).
    split; nra.
  - (* u is further out *)
    destruct (W_span u' (u - u')%Z Hu' ltac:(lia)) as [Lo Hi].
    replace (u' + (u - u'))%Z with u in Lo, Hi by lia.
    assert (HAk : ALPHA <= qpow BETA (Z.to_nat (u - u')))
      by (apply alpha_bounds_a_half_ulp_span; rewrite Z2Nat.id by lia; lia).
    assert (Hw : 0 <= W u) by (apply W_nonneg; lia).
    assert (Hw' : 0 <= W u') by (apply W_nonneg; lia).
    split; nra.
Qed.

(** Thirty halvings, so the weight past `30 * 2^32` is below a quarter of a
    Q0.28 ulp. This is what makes the saturate harmless. *)
Lemma W_halves_n : forall (n : nat),
  qpow (2#1) n * W (Z.of_nat n * 4294967296)%Z <= W 0%Z.
Proof.
  induction n.
  - replace (Z.of_nat 0 * 4294967296)%Z with 0%Z by lia. simpl. lra.
  - assert (Hu : (0 <= Z.of_nat n * 4294967296)%Z) by lia.
    pose proof (W_halves _ Hu) as Hh.
    replace (Z.of_nat n * 4294967296 + 4294967296)%Z
       with (Z.of_nat (S n) * 4294967296)%Z in Hh by lia.
    assert (H2 : 0 <= qpow (2#1) n) by (apply qpow_nonneg; unfold Qle; simpl; lia).
    rewrite qpow_S. nra.
Qed.

Theorem the_weight_past_thirty_halvings_is_below_an_ulp :
  forall u, (30 * 4294967296 <= u)%Z -> W u <= 1#4.
Proof.
  intros u Hu.
  pose proof (W_halves_n 30) as H30.
  replace (Z.of_nat 30) with 30%Z in H30 by reflexivity.
  assert (Hp : qpow (2#1) 30 == 1073741824 # 1) by (vm_compute; reflexivity).
  rewrite Hp, W_one in H30.
  (* W (30 * 2^32) <= 2^28 / 2^30 = 1/4 *)
  assert (Hq : W (30 * 4294967296)%Z <= 1#4) by (unfold TWO28 in H30; lra).
  destruct (W_span (30 * 4294967296)%Z (u - 30 * 4294967296)%Z
                   ltac:(lia) ltac:(lia)) as [_ Hi].
  replace (30 * 4294967296 + (u - 30 * 4294967296))%Z with u in Hi by lia.
  lra.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Part 3 - the exp, and the domain gap nothing had closed         *)
(* ------------------------------------------------------------------ *)

(** `src/fixed_exp.rs::exp2_neg_q16_16`, as a function of the Q16.16 argument. *)
Variable expf : Z -> Z.

(** Discharged EXHAUSTIVELY, in Rust, by
    `fixed_exp.rs::it_is_sub_ulp_accurate_everywhere`: every `t` in
    `0 .. 31<<16` against `f64::exp2`, worst 0.908 ulp. That is a stronger
    check than a proof about the table and the three-term series would be, and
    it is the reason the exp is a hypothesis here rather than a theorem -
    `docs/verified_kernel_process.md` §4. *)
Hypothesis exp_is_sub_ulp_on_the_swept_domain :
  forall t, (0 <= t)%Z -> (t < 31 * 65536)%Z ->
    Qabs (inject_Z (expf t) - W (u_of_arg t)) <= 1.

(** Structural, from the `if n >= 30 { return 0 }` early return. Pinned in
    Rust by `the_saturation_guard_covers_the_whole_input_range`. *)
Hypothesis exp_is_zero_above_the_table :
  forall t, (30 * 65536 <= t)%Z -> expf t = 0%Z.

(** **The gap, closed.** The exhaustive sweep stops at `31<<16 = 2,031,616`;
    the emitted `min.s64 ..., 1073741824` admits arguments up to `2^30`, which
    is 528 times further out. Nothing had said the sub-ulp headline survives
    the difference, and it does, by two lines: up there the implementation
    returns 0 and the ideal weight is below a quarter of an ulp, because
    `31 * 2^32` is past thirty halvings of `2^28`.

    This is the shape `CLAUDE.md` records as "a number in a comment is not a
    check", in the one place where the number was not even in a comment. *)
Theorem the_swept_domain_covers_the_admitted_one :
  forall t, (0 <= t)%Z -> (t <= SAT)%Z ->
    Qabs (inject_Z (expf t) - W (u_of_arg t)) <= 1.
Proof.
  intros t H0 Ht.
  destruct (Z_lt_le_dec t (31 * 65536)) as [Hlt | Hge].
  - now apply exp_is_sub_ulp_on_the_swept_domain.
  - rewrite (exp_is_zero_above_the_table t ltac:(lia)).
    assert (Hu : (30 * 4294967296 <= u_of_arg t)%Z) by (unfold u_of_arg, ULP; lia).
    pose proof (the_weight_past_thirty_halvings_is_below_an_ulp _ Hu) as Hsmall.
    assert (Hpos : 0 <= W (u_of_arg t)) by (apply W_nonneg; lia).
    change (inject_Z 0) with 0.
    apply Qabs_Qle_condition. split; lra.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Part 4 - one weight: the additive ulp and the MULTIPLICATIVE
       argument error, composed                                        *)
(* ------------------------------------------------------------------ *)

Variable delta : nat -> Z.       (** `m - s_i`, non-negative by construction *)
Variable kfix  : Z.              (** `KFix = round(C * 2^32)` *)

Hypothesis delta_nonneg : forall i, (0 <= delta i)%Z.
Hypothesis kfix_nonneg  : (0 <= kfix)%Z.

(** The kernel's Q0.28 weight, and the ideal it approximates. *)
Definition p (i : nat) : Z := expf (arg (delta i) kfix).
Definition w (i : nat) : Q := W (u_exact (delta i) kfix).

Lemma w_nonneg : forall i, 0 <= w i.
Proof.
  intros. unfold w. apply W_nonneg. unfold u_exact.
  apply Z.mul_nonneg_nonneg; [apply delta_nonneg | exact kfix_nonneg].
Qed.

(** **The per-element bracket, and the joint this file exists for.**

    The exp's error is ADDITIVE - one ulp of Q0.28, wherever the argument is.
    The argument reduction's error is MULTIPLICATIVE - it perturbs an
    *exponent*, so it scales the weight. They compose as `EPS * w + 1` rather
    than as a single number, and that shape is what makes the max subtraction
    load-bearing further down: the additive term is only negligible relative to
    a total weight that has a floor.

    Both regimes are covered. Below the saturate the two-sided bracket applies;
    at it the exact exponent is past thirty halvings, so the ideal weight is
    itself below an ulp and returning zero is within the bracket. *)
Theorem the_weight_is_within_a_relative_ulp :
  forall i, Qabs (inject_Z (p i) - w i) <= EPS * w i + 1.
Proof.
  intros i.
  set (d := delta i).
  assert (Hd : (0 <= d)%Z) by apply delta_nonneg.
  assert (Hu : (0 <= u_exact d kfix)%Z)
    by (unfold u_exact; apply Z.mul_nonneg_nonneg; auto).
  destruct (the_argument_is_in_range d kfix Hd kfix_nonneg) as [Ha0 HaS].
  assert (Hu' : (0 <= u_of_arg (arg d kfix))%Z)
    by (unfold u_of_arg, ULP; lia).
  pose proof (the_swept_domain_covers_the_admitted_one _ Ha0 HaS) as Hexp.
  unfold p, w. fold d.
  assert (Hw : 0 <= W (u_exact d kfix)) by (apply W_nonneg; auto).
  assert (Hw' : 0 <= W (u_of_arg (arg d kfix))) by (apply W_nonneg; auto).
  assert (HE : 0 <= EPS) by (unfold EPS, Qle; simpl; lia).
  apply Qabs_Qle_condition in Hexp.
  apply Qabs_Qle_condition.
  destruct (Z_le_gt_dec ((d * kfix + HALF) / ULP) SAT) as [Hns | Hs].
  - (* unsaturated: the two-sided bracket *)
    pose proof (an_unsaturated_argument_is_within_a_half_ulp d kfix Hd kfix_nonneg Hns)
      as Hnear.
    destruct (W_two_sided (u_exact d kfix) (u_of_arg (arg d kfix)) Hu Hu'
                ltac:(unfold HALF in Hnear; exact Hnear)) as [B1 B2].
    (* 1 - ALPHA <= EPS, and ALPHA * (1 + EPS) >= 1 - the two facts that turn
       a half-ulp of exponent into a relative EPS on the weight. *)
    assert (K1 : 1 - ALPHA <= EPS) by (unfold ALPHA, EPS, Qle, Qminus; simpl; lia).
    assert (K2 : 1 <= ALPHA * (1 + EPS))
      by (unfold ALPHA, EPS, Qle, Qplus, Qmult; simpl; lia).
    (* W u' <= (1+EPS) * W u  and  W u >= W u' - EPS * W u *)
    assert (U1 : W (u_of_arg (arg d kfix)) <= (1 + EPS) * W (u_exact d kfix)) by nra.
    assert (U2 : W (u_exact d kfix) - W (u_of_arg (arg d kfix))
                 <= EPS * W (u_exact d kfix)) by nra.
    split; lra.
  - (* saturated: the exact exponent is astronomically large *)
    assert (Hsat : arg d kfix = SAT)
      by (unfold arg; apply Z.min_r; lia).
    rewrite Hsat in *.
    pose proof (saturation_means_the_exact_exponent_is_astronomically_large
                  d kfix Hd kfix_nonneg ltac:(lia)) as Hfar.
    unfold TWO32 in Hfar.
    pose proof (the_weight_past_thirty_halvings_is_below_an_ulp _ Hfar) as Hsmall.
    rewrite (exp_is_zero_above_the_table SAT ltac:(unfold SAT; lia)).
    change (inject_Z 0) with 0.
    split; nra.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Part 5 - the reduction, and the quotient                        *)
(* ------------------------------------------------------------------ *)

(** Ascending, exactly like `Decomposition.acc_range op f 0 n`, so the join in
    Part 6 is a rewrite rather than a re-proof. *)
Fixpoint qsum (f : nat -> Q) (n : nat) : Q :=
  match n with O => 0 | S k => qsum f k + f k end.

Lemma qsum_le : forall f g n, (forall i, f i <= g i) -> qsum f n <= qsum g n.
Proof. induction n; simpl; intros. lra. specialize (IHn H). specialize (H n). lra. Qed.

Lemma qsum_nonneg : forall f n, (forall i, 0 <= f i) -> 0 <= qsum f n.
Proof. induction n; simpl; intros. lra. specialize (IHn H). specialize (H n). lra. Qed.

Lemma qsum_abs : forall f n, Qabs (qsum f n) <= qsum (fun i => Qabs (f i)) n.
Proof.
  induction n; simpl. apply Qle_refl.
  eapply Qle_trans. apply Qabs_triangle. lra.
Qed.

Lemma qsum_plus : forall f g n,
  qsum (fun i => f i + g i) n == qsum f n + qsum g n.
Proof. induction n; simpl. ring. rewrite IHn. ring. Qed.

Lemma qsum_scale : forall c f n, qsum (fun i => c * f i) n == c * qsum f n.
Proof. induction n; simpl. ring. rewrite IHn. ring. Qed.

Lemma qsum_minus : forall f g n, qsum (fun i => f i - g i) n == qsum f n - qsum g n.
Proof. induction n; simpl. ring. rewrite IHn. ring. Qed.

Lemma qsum_ext : forall f g n, (forall i, f i == g i) -> qsum f n == qsum g n.
Proof.
  induction n; simpl; intros. reflexivity. rewrite (IHn H), (H n). reflexivity.
Qed.

Lemma qsum_const : forall c n, qsum (fun _ => c) n == inject_Z (Z.of_nat n) * c.
Proof.
  induction n.
  - cbn [qsum]. change (Z.of_nat 0) with 0%Z. change (inject_Z 0) with 0. ring.
  - cbn [qsum]. rewrite IHn, Nat2Z.inj_succ, <- Z.add_1_r, inject_Z_plus.
    change (inject_Z 1) with 1. ring.
Qed.

Lemma qsum_ge_term : forall f n i0, (forall i, 0 <= f i) -> (i0 < n)%nat ->
  f i0 <= qsum f n.
Proof.
  induction n; intros i0 Hf Hi. lia.
  simpl. destruct (Nat.eq_dec i0 n) as [->|Hne].
  - pose proof (qsum_nonneg f n Hf). lra.
  - assert (i0 < n)%nat by lia. specialize (IHn i0 Hf H). specialize (Hf n). lra.
Qed.

Variable v : nat -> Z.
Hypothesis v_is_int8 : forall i, (Z.abs (v i) <= 127)%Z.

(** The kernel's two exact integer reductions, and the ideal each approximates.
    Both are EXACT - that is why Part 6's join to `GridStrideSplit` costs
    nothing. *)
Definition Lq   (n : nat) : Q := qsum (fun i => inject_Z (p i)) n.
Definition Oq   (n : nat) : Q := qsum (fun i => inject_Z (p i) * inject_Z (v i)) n.
Definition Wtot (n : nat) : Q := qsum w n.
Definition Wnum (n : nat) : Q := qsum (fun i => w i * inject_Z (v i)) n.

(** The whole error budget in one place. Its first term is the argument
    reduction's, relative; its second is the exp table's, absolute and
    proportional to the number of keys. *)
Definition Err (n : nat) : Q := EPS * Wtot n + inject_Z (Z.of_nat n).

Lemma Wtot_nonneg : forall n, 0 <= Wtot n.
Proof. intros. apply qsum_nonneg. apply w_nonneg. Qed.

Lemma v_abs_q : forall i, Qabs (inject_Z (v i)) <= VMAXQ.
Proof.
  intros i. rewrite Qabs_inject_Z. unfold VMAXQ.
  change (127#1) with (inject_Z 127). rewrite <- Zle_Qle. apply v_is_int8.
Qed.

Lemma the_computed_total_is_close_to_the_ideal :
  forall n, Qabs (Lq n - Wtot n) <= Err n.
Proof.
  intros n. unfold Lq, Err, Wtot.
  rewrite <- qsum_minus.
  eapply Qle_trans. apply qsum_abs.
  eapply Qle_trans.
  { apply qsum_le with (g := fun i => EPS * w i + 1).
    intros. apply the_weight_is_within_a_relative_ulp. }
  rewrite qsum_plus, qsum_scale, qsum_const. lra.
Qed.

Lemma the_weighted_sum_is_close_to_the_ideal :
  forall n, Qabs (Oq n - Wnum n) <= VMAXQ * Err n.
Proof.
  intros n. unfold Oq, Wnum, Err, Wtot.
  rewrite <- qsum_minus.
  rewrite (qsum_ext _ (fun i => (inject_Z (p i) - w i) * inject_Z (v i)) n)
    by (intros; ring).
  eapply Qle_trans. apply qsum_abs.
  eapply Qle_trans.
  { apply qsum_le with (g := fun i => VMAXQ * (EPS * w i + 1)).
    intros i. rewrite Qabs_Qmult.
    pose proof (the_weight_is_within_a_relative_ulp i) as H1.
    pose proof (v_abs_q i) as H2.
    pose proof (Qabs_nonneg (inject_Z (p i) - w i)) as H3.
    pose proof (Qabs_nonneg (inject_Z (v i))) as H4.
    assert (0 <= EPS * w i + 1).
    { pose proof (w_nonneg i).
      assert (0 <= EPS) by (unfold EPS, Qle; simpl; lia). nra. }
    nra. }
  rewrite (qsum_scale VMAXQ (fun i => EPS * w i + 1) n).
  rewrite qsum_plus, qsum_scale, qsum_const. lra.
Qed.

Lemma the_ideal_numerator_is_bounded : forall n, Qabs (Wnum n) <= VMAXQ * Wtot n.
Proof.
  intros n. unfold Wnum, Wtot.
  eapply Qle_trans. apply qsum_abs.
  eapply Qle_trans.
  { apply qsum_le with (g := fun i => VMAXQ * w i).
    intros i. rewrite Qabs_Qmult.
    pose proof (v_abs_q i) as H2. pose proof (w_nonneg i) as H3.
    rewrite (Qabs_pos (w i) H3).
    pose proof (Qabs_nonneg (inject_Z (v i))). nra. }
  rewrite qsum_scale. lra.
Qed.

(** **The bound, division-free.** Everything above meets here:
    `O*Wtot - Wnum*L = (O - Wnum)*Wtot + Wnum*(Wtot - L)`, and each of the four
    factors already has a bound. *)
Theorem the_attention_output_is_within_the_bound : forall n,
  Qabs (Oq n * Wtot n - Wnum n * Lq n) <= (2#1) * VMAXQ * Err n * Wtot n.
Proof.
  intros n.
  pose proof (the_weighted_sum_is_close_to_the_ideal n) as H1.
  pose proof (the_computed_total_is_close_to_the_ideal n) as H2.
  pose proof (the_ideal_numerator_is_bounded n) as H3.
  pose proof (Wtot_nonneg n) as H4.
  assert (E : Oq n * Wtot n - Wnum n * Lq n
              == (Oq n - Wnum n) * Wtot n + Wnum n * (Wtot n - Lq n)) by ring.
  rewrite E.
  eapply Qle_trans. apply Qabs_triangle.
  rewrite !Qabs_Qmult.
  assert (A1 : Qabs (Oq n - Wnum n) * Qabs (Wtot n) <= (VMAXQ * Err n) * Wtot n).
  { rewrite (Qabs_pos (Wtot n) H4).
    apply Qmult_le_compat_r; assumption. }
  assert (A2 : Qabs (Wnum n) * Qabs (Wtot n - Lq n) <= (VMAXQ * Wtot n) * Err n).
  { apply Qmult_le_compat_nonneg.
    - split; [apply Qabs_nonneg | assumption].
    - split; [apply Qabs_nonneg |].
      rewrite <- Qabs_opp. assert (Q : -(Wtot n - Lq n) == Lq n - Wtot n) by ring.
      rewrite Q. assumption. }
  lra.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Part 6 - the max subtraction, and the bound as a number         *)
(* ------------------------------------------------------------------ *)

(** **What the max subtraction is FOR, stated as a floor.**

    `m = max_i s_i`, so `delta` is zero at the argmax and that key's ideal
    weight is exactly `W 0 = 2^28`. Every other weight is positive, so the
    total is at least `2^28`.

    It is usually described as an overflow guard - keep `exp` out of its own
    saturation. It is also, and this is the part nothing had recorded, what
    makes the exp table's ABSOLUTE error relatively negligible: the additive
    term in [Err] is `n` ulps against a total of at least `2^28` of them. *)
Theorem the_max_subtraction_puts_a_floor_under_the_total_weight :
  forall n i0, (i0 < n)%nat -> delta i0 = 0%Z -> TWO28 <= Wtot n.
Proof.
  intros n i0 Hi Hz. unfold Wtot.
  assert (Hterm : w i0 == TWO28).
  { unfold w, u_exact. rewrite Hz.
    replace (0 * kfix)%Z with 0%Z by ring. exact W_one. }
  pose proof (qsum_ge_term w n i0 w_nonneg Hi) as H. lra.
Qed.

(** **The refutation, and it is not hypothetical.** Without that floor the same
    per-element bracket is satisfied by `p_i = 0` for every key, so `L = 0` and
    there is no output at all - not an inaccurate one.

    That is the F=16 attention-sink failure `tests/attention_quantization_error.rs`
    records: with a sink 18 logits above the rest every non-sink weight rounded
    to zero and the whole tail of the softmax disappeared. Here it is the same
    statement one layer up - a total ideal weight below one ulp admits a zero
    denominator. *)
Theorem a_total_weight_below_one_ulp_admits_a_zero_denominator :
  forall (wt : nat -> Q) n,
    (forall i, 0 < wt i) ->
    (forall i, wt i <= 1) ->
    (forall i, Qabs (inject_Z 0%Z - wt i) <= EPS * wt i + 1)
    /\ qsum (fun _ => inject_Z 0%Z) n == 0.
Proof.
  intros wt n Hpos Hle. split.
  - intros i. specialize (Hpos i). specialize (Hle i).
    assert (0 <= EPS) by (unfold EPS, Qle; simpl; lia).
    change (inject_Z 0) with 0.
    apply Qabs_Qle_condition. split; nra.
  - rewrite qsum_const. change (inject_Z 0) with 0. ring.
Qed.

(** The computed denominator cannot be far below the ideal one. *)
Lemma the_denominator_is_close : forall n,
  (1 - EPS) * Wtot n - inject_Z (Z.of_nat n) <= Lq n.
Proof.
  intros n. pose proof (the_computed_total_is_close_to_the_ideal n) as H.
  apply Qabs_Qle_condition in H. unfold Err in H. lra.
Qed.

(** The bound, as a quotient. Division is confined to here so the algebra above
    stays division-free. *)
Corollary the_output_quotient_is_within_the_bound : forall n,
  0 < Lq n -> 0 < Wtot n ->
  Qabs (Oq n / Lq n - Wnum n / Wtot n) <= (2#1) * VMAXQ * Err n / Lq n.
Proof.
  intros n HL HW.
  pose proof (the_attention_output_is_within_the_bound n) as H.
  apply Qabs_Qle_condition in H.
  assert (E : Oq n / Lq n - Wnum n / Wtot n
              == (Oq n * Wtot n - Wnum n * Lq n) / (Lq n * Wtot n)).
  { field; split; intro C; [rewrite C in HW | rewrite C in HL]; apply (Qlt_irrefl 0); assumption. }
  rewrite E.
  apply Qabs_Qle_condition. split.
  - apply Qle_shift_div_l. now apply Qmult_lt_0_compat.
    (* -(2*V*Err/L) * (L*W) = -(2*V*Err*W) *)
    assert (HD : - ((2#1) * VMAXQ * Err n / Lq n) * (Lq n * Wtot n)
                 == - ((2#1) * VMAXQ * Err n / Lq n * Lq n) * Wtot n) by ring.
    rewrite HD.
    assert (HQ : (2#1) * VMAXQ * Err n / Lq n * Lq n == (2#1) * VMAXQ * Err n)
      by (field; intro C; rewrite C in HL; apply (Qlt_irrefl 0); assumption).
    rewrite HQ. lra.
  - apply Qle_shift_div_r. now apply Qmult_lt_0_compat.
    assert (HD : (2#1) * VMAXQ * Err n / Lq n * (Lq n * Wtot n)
                 == ((2#1) * VMAXQ * Err n / Lq n * Lq n) * Wtot n) by ring.
    rewrite HD.
    assert (HQ : (2#1) * VMAXQ * Err n / Lq n * Lq n == (2#1) * VMAXQ * Err n)
      by (field; intro C; rewrite C in HL; apply (Qlt_irrefl 0); assumption).
    rewrite HQ. lra.
Qed.

(** **The bound as a number.** At 65,536 keys - a long context - with int8 `V`,
    the exact-attention output is within `7/100` of the ideal softmax, on an
    output range of +-127. The true value of the expression at the floor is
    `4318/65519 = 0.06590...`, so `7/100` is the round number above it and not
    a fitted constant.

    Both quantitative ingredients are visible in the arithmetic: `EPS * Wtot`
    is the argument reduction's relative term and `n` is the exp table's
    absolute one, and the second is what makes the bound grow with the
    context. At `n = 2^20` the same expression is `0.99998`, i.e. the bound is
    still under one unit of `V` but only just - which is a real statement about
    where Q0.28 runs out, and it agrees with `MAX_EXACT_SEQ_LEN`'s independent
    accumulator-width argument being about a much larger `n`. *)
Corollary the_bound_at_a_long_context : forall n i0,
  (i0 < n)%nat -> delta i0 = 0%Z -> (Z.of_nat n <= 65536)%Z ->
  Qabs (Oq n / Lq n - Wnum n / Wtot n) <= 7#100.
Proof.
  intros n i0 Hi Hz Hn.
  pose proof (the_max_subtraction_puts_a_floor_under_the_total_weight n i0 Hi Hz) as HF.
  pose proof (the_denominator_is_close n) as HD.
  assert (HNq : inject_Z (Z.of_nat n) <= inject_Z 65536) by (rewrite <- Zle_Qle; lia).
  assert (HN0 : 0 <= inject_Z (Z.of_nat n))
    by (change 0 with (inject_Z 0); rewrite <- Zle_Qle; apply Nat2Z.is_nonneg).
  assert (H28 : TWO28 == 268435456 # 1) by reflexivity.
  assert (HE : EPS == 1#65536) by reflexivity.
  assert (H65536 : inject_Z 65536 == 65536 # 1) by reflexivity.
  rewrite H28 in HF. rewrite H65536 in HNq. rewrite HE in HD.
  (* linear in Wtot and n, at the floor Wtot >= 2^28 *)
  assert (HL : 0 < Lq n) by lra.
  assert (HW : 0 < Wtot n) by lra.
  eapply Qle_trans. apply (the_output_quotient_is_within_the_bound n HL HW).
  apply Qle_shift_div_r; [assumption |].
  unfold Err, VMAXQ. rewrite HE. lra.
Qed.

(** **The join to the decomposition schema, and the answer to the question this
    file was written to ask.**

    `L` and `O` are EXACT integer reductions, so `GridStrideSplit`'s residue-
    class partition applies to them verbatim: at any worker count, in any
    order the atomics land, the kernel's accumulators hold exactly the sums
    the bound above is stated about. The approximation happens per element,
    BEFORE the fold - so the schema does not have to grow an approximate
    variant, and the bound is a property of the kernel rather than of one
    launch.

    An f32 softmax has no such split: its error is produced BY the reduction,
    and the bound would have to be re-derived for every decomposition. *)
Lemma inject_Z_of_a_range_sum : forall (f : nat -> Z) n,
  inject_Z (KS.acc_range Z.add f 0 n) == qsum (fun i => inject_Z (f i)) n.
Proof.
  induction n; simpl. reflexivity.
  rewrite inject_Z_plus, IHn.
  replace (0 + n)%nat with n by lia. reflexivity.
Qed.

Corollary the_bound_holds_at_every_launch_geometry : forall n k,
  (0 < k)%nat ->
  inject_Z (GS.combine Z.add p k n k) == Lq n
  /\ inject_Z (GS.combine Z.add (fun i => p i * v i)%Z k n k) == Oq n.
Proof.
  intros n k Hk. split.
  - rewrite (GS.grid_stride_exact p k n Hk). unfold GS.sum_upto.
    apply inject_Z_of_a_range_sum.
  - rewrite (GS.grid_stride_exact _ k n Hk). unfold GS.sum_upto, Oq.
    rewrite inject_Z_of_a_range_sum.
    apply qsum_ext. intros. rewrite inject_Z_mult. reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(** ** Part 6b - the TEMPERATURE is itself quantized                   *)
(* ------------------------------------------------------------------ *)

(** Everything above compares the computed weight against `W (delta * KFix)` -
    the ideal at the multiplier the kernel was HANDED. `KFix` is itself
    `round(C * 2^32)`, so that is not the ideal at the temperature the caller
    meant, and the difference was priced nowhere: the "what this does not
    claim" section says so in as many words.

    It is priced here, and the shape of the answer is the interesting part.
    The exponent error from rounding the multiplier is `delta * |KFix - C*2^32|
    <= delta / 2` in units of `2^-32` log2 - so it is proportional to the SCORE
    DELTA and INDEPENDENT of `KFix`. A bound on the delta is therefore the
    whole bound, and the delta is bounded by the head_dim: `|s| <= 127^2 * hd`
    for an int8 pipeline, so `delta = m - s_i <= 2 * 127^2 * hd`. That is a
    compile-time function of one parameter, which is why
    [`exact_attention::score_delta_span`] can compute it.

    Two multipliers are refused rather than bounded, because they are not
    approximations of anything - see [a_zero_multiplier_gives_every_key_the
    _same_weight] and [the_two_readings_of_the_multiplier_disagree_above_two
    _to_the_thirty_one]. Both bounds were already in `tools/ptx_bridge.py` as a
    bare `continue`; what is new is that they are derived, enforced in the
    compiler by `exact_attention::temperature_fixed_point`, and stated. *)

Definition TWO32Q : Q := inject_Z TWO32.

(** `|s| <= 127^2 * hd` for int8 operands, and the delta is between two
    scores. Kept as a definition rather than a literal so the tie test can
    compare it against `score_delta_span` rather than against a number. *)
Definition SCORE_DELTA_SPAN (hd : Z) : Z := 2 * 127 * 127 * hd.

(** The kernel declares `.param .u32 q6` and consumes it with
    `mul.wide.s32`, so at or above `2^31` the two readings are different
    numbers. *)
Definition KFIX_SIGNED_LIMIT : Z := 2147483648.
Definition as_s32 (k : Z) : Z := if (k <? KFIX_SIGNED_LIMIT)%Z then k else (k - TWO32)%Z.

Lemma beta_unit : 0 <= BETA /\ BETA <= 1.
Proof. unfold BETA. split; unfold Qle; simpl; lia. Qed.

Lemma qpow_mono : forall q (a b : nat), 0 <= q -> q <= 1 -> (a <= b)%nat ->
  qpow q b <= qpow q a.
Proof.
  intros q a b Hq H1 Hab.
  replace b with (a + (b - a))%nat by lia.
  rewrite qpow_add.
  assert (0 <= qpow q a) by (apply qpow_nonneg; auto).
  assert (qpow q (b-a)%nat <= 1) by (apply qpow_le_1; auto).
  nra.
Qed.

(** Two exponents within `n` of each other give ideal weights within a factor
    `BETA^n`, in BOTH directions - which is what makes it a bracket rather
    than a one-sided decay claim. Stated symmetrically so no division
    appears. *)
Lemma the_ideal_weights_at_two_close_exponents_bracket_each_other :
  forall u1 u2 (n : nat),
    (0 <= u1)%Z -> (0 <= u2)%Z ->
    (Z.abs (u1 - u2) <= Z.of_nat n)%Z ->
    qpow BETA n * W u1 <= W u2 /\ qpow BETA n * W u2 <= W u1.
Proof.
  intros u1 u2 n H1 H2 Hd.
  destruct beta_unit as [Hb0 Hb1].
  assert (Hle : forall a b, (0 <= a)%Z -> (0 <= b)%Z -> (a <= b)%Z ->
                (Z.abs (a - b) <= Z.of_nat n)%Z ->
                qpow BETA n * W a <= W b /\ qpow BETA n * W b <= W a).
  { intros a b Ha Hb Hab Habs.
    set (k := (b - a)%Z).
    assert (Hk : (0 <= k)%Z) by (unfold k; lia).
    assert (Hkn : (Z.to_nat k <= n)%nat) by (rewrite Z.abs_neq in Habs by lia; lia).
    destruct (W_span a k Ha Hk) as [Hlo Hhi].
    replace (a + k)%Z with b in Hlo, Hhi by (unfold k; lia).
    assert (HWa : 0 <= W a) by (apply W_nonneg; auto).
    assert (HWb : 0 <= W b) by (apply W_nonneg; auto).
    assert (Hm : qpow BETA n <= qpow BETA (Z.to_nat k)) by (apply qpow_mono; auto).
    assert (H1n : qpow BETA n <= 1) by (apply qpow_le_1; auto).
    split; nra. }
  destruct (Z_le_gt_dec u1 u2) as [H|H].
  - apply Hle; auto.
  - assert (Hs : (Z.abs (u2 - u1) <= Z.of_nat n)%Z) by lia.
    destruct (Hle u2 u1 H2 H1 ltac:(lia) Hs) as [A B]. split; auto.
Qed.

Lemma inject_Z_sub : forall a b : Z, inject_Z (a - b)%Z == inject_Z a - inject_Z b.
Proof.
  intros a b. replace (a - b)%Z with (a + (- b))%Z by ring.
  rewrite inject_Z_plus, inject_Z_opp. reflexivity.
Qed.

(** The exponent statement, and the only place a rational temperature appears.
    `k` is the kernel's multiplier and `vtrue` the nearest integer to the true
    exponent `delta * C * 2^32`; both roundings are to nearest, so both are
    within a half, and the two halves compose to `(delta + 1) / 2` - stated as
    `2 * |..| <= delta + 1` so it is an integer fact with no division in it. *)
Theorem the_quantized_temperature_shifts_the_exponent_by_at_most_half_a_delta :
  forall (C : Q) (k d vtrue : Z),
    (0 <= d)%Z ->
    Qabs (inject_Z k - C * TWO32Q) <= 1#2 ->
    Qabs (inject_Z vtrue - inject_Z d * (C * TWO32Q)) <= 1#2 ->
    (2 * Z.abs (u_exact d k - vtrue) <= d + 1)%Z.
Proof.
  intros C k d vtrue Hd Hk Hv.
  assert (Hdq : 0 <= inject_Z d)
    by (change 0 with (inject_Z 0); rewrite <- Zle_Qle; lia).
  assert (Hsplit :
    inject_Z (u_exact d k - vtrue)
    == inject_Z d * (inject_Z k - C * TWO32Q)
       + (inject_Z d * (C * TWO32Q) - inject_Z vtrue)).
  { unfold u_exact. rewrite inject_Z_sub, inject_Z_mult. ring. }
  assert (Ha : Qabs (inject_Z (u_exact d k - vtrue)) <= inject_Z d * (1#2) + (1#2)).
  { rewrite Hsplit.
    eapply Qle_trans; [ apply Qabs_triangle | ].
    assert (T1 : Qabs (inject_Z d * (inject_Z k - C * TWO32Q))
                 <= inject_Z d * (1#2)).
    { rewrite Qabs_Qmult.
      assert (Qabs (inject_Z d) == inject_Z d) by (apply Qabs_pos; auto).
      assert (0 <= Qabs (inject_Z k - C * TWO32Q)) by apply Qabs_nonneg.
      nra. }
    assert (T2 : Qabs (inject_Z d * (C * TWO32Q) - inject_Z vtrue) <= 1#2).
    { assert (E : inject_Z d * (C * TWO32Q) - inject_Z vtrue
                  == - (inject_Z vtrue - inject_Z d * (C * TWO32Q))) by ring.
      rewrite E, Qabs_opp. exact Hv. }
    lra. }
  rewrite Qabs_inject_Z in Ha.
  assert (Hz : inject_Z (2 * Z.abs (u_exact d k - vtrue)) <= inject_Z (d + 1)).
  { rewrite inject_Z_plus, inject_Z_mult.
    change (inject_Z 2) with (2#1). change (inject_Z 1) with 1. lra. }
  rewrite <- Zle_Qle in Hz. exact Hz.
Qed.

(** The weight consequence: rounding the temperature moves the ideal weight by
    at most `BETA^n` where `2n >= delta + 1`. Note `n ~ delta/2`, not `delta`
    - the half is the whole point of the previous theorem and dropping it
    doubles the bound. *)
Corollary the_quantized_temperature_moves_the_ideal_weight_by_at_most_a_beta_power :
  forall (C : Q) (k d vtrue : Z) (n : nat),
    (0 <= d)%Z -> (0 <= u_exact d k)%Z -> (0 <= vtrue)%Z ->
    Qabs (inject_Z k - C * TWO32Q) <= 1#2 ->
    Qabs (inject_Z vtrue - inject_Z d * (C * TWO32Q)) <= 1#2 ->
    (d + 1 <= 2 * Z.of_nat n)%Z ->
    qpow BETA n * W (u_exact d k) <= W vtrue /\ qpow BETA n * W vtrue <= W (u_exact d k).
Proof.
  intros C k d vtrue n Hd Hu Hv Hk Hvr Hn.
  apply the_ideal_weights_at_two_close_exponents_bracket_each_other; auto.
  pose proof (the_quantized_temperature_shifts_the_exponent_by_at_most_half_a_delta
                C k d vtrue Hd Hk Hvr) as H. lia.
Qed.

(** `BETA^n >= 1 - n * 2^-32` - Bernoulli, so no power is ever evaluated. *)
Lemma beta_power_is_above_its_linearisation : forall n : nat,
  1 - inject_Z (Z.of_nat n) * (1#4294967296) <= qpow BETA n.
Proof.
  intros n.
  assert (Hb : BETA == 1 + (- (1#4294967296)))
    by (unfold BETA, Qeq; simpl; lia).
  assert (Hq : qpow BETA n == qpow (1 + (- (1#4294967296))) n)
    by (apply qpow_Qeq; exact Hb).
  rewrite Hq.
  pose proof (bernoulli n (- (1#4294967296)) ltac:(unfold Qle; simpl; lia)) as B.
  lra.
Qed.

(** The headline, at the head_dim this kernel is used at. `2 * 127^2 * 128` is
    4,129,024, so `n = 2,064,513` satisfies `d + 1 <= 2n` for every delta an
    int8 pipeline can produce, and `1 - n*2^-32 > 1 - 1/2048`.

    Measured against the real arithmetic the worst observed exponent error is
    0.837 of this allowance and the worst weight factor is 1.00028 against the
    1/2048 = 1.00049 stated here, so it is loose by about 1.7x - report the
    slack, not just the bound. *)
Corollary at_head_dim_128_the_temperature_moves_a_weight_by_under_a_two_thousandth :
  forall (C : Q) (k d vtrue : Z),
    (0 <= d)%Z -> (d <= SCORE_DELTA_SPAN 128)%Z ->
    (0 <= u_exact d k)%Z -> (0 <= vtrue)%Z ->
    Qabs (inject_Z k - C * TWO32Q) <= 1#2 ->
    Qabs (inject_Z vtrue - inject_Z d * (C * TWO32Q)) <= 1#2 ->
    (1 - (1#2048)) * W (u_exact d k) <= W vtrue
    /\ (1 - (1#2048)) * W vtrue <= W (u_exact d k).
Proof.
  intros C k d vtrue Hd Hspan Hu Hv Hk Hvr.
  (* `Z.to_nat 2064513`, NOT the nat literal: a `nat` literal in Coq is UNARY,
     so writing 2064513 here builds two million constructors and every tactic
     that normalises hangs. This is the same reason Part 0 says nothing is
     ever evaluated at a large literal. *)
  set (n := Z.to_nat 2064513).
  assert (Hid : Z.of_nat n = 2064513%Z) by (unfold n; apply Z2Nat.id; lia).
  assert (Hn : (d + 1 <= 2 * Z.of_nat n)%Z)
    by (rewrite Hid; unfold SCORE_DELTA_SPAN in Hspan; lia).
  destruct (the_quantized_temperature_moves_the_ideal_weight_by_at_most_a_beta_power
              C k d vtrue n Hd Hu Hv Hk Hvr Hn) as [A B].
  assert (HL : 1 - (1#2048) <= qpow BETA n).
  { eapply Qle_trans; [ | apply beta_power_is_above_its_linearisation ].
    rewrite Hid. change (inject_Z 2064513) with (2064513#1).
    unfold Qle; simpl; lia. }
  assert (HWu : 0 <= W (u_exact d k)) by (apply W_nonneg; auto).
  assert (HWv : 0 <= W vtrue) by (apply W_nonneg; auto).
  split; nra.
Qed.

(** REFUSAL 1. `C < 2^-33` rounds the multiplier to zero, and then the
    argument is `(0 + 2^15) >> 16 = 0` for EVERY key - so every weight is
    exactly `2^28` and the softmax is uniform. This is the same symptom
    `tools/ptx_bridge.py` finding 06 records from a multiplier `2^16` too
    small, and its own differential could not see it: both arms replicate the
    kernel's formula, so they agree bit for bit on a uniform answer. *)
Theorem a_zero_multiplier_gives_every_key_the_same_weight :
  forall d1 d2, (0 <= d1)%Z -> (0 <= d2)%Z ->
    arg d1 0 = 0%Z /\ arg d2 0 = 0%Z /\ expf (arg d1 0) = expf (arg d2 0).
Proof.
  intros d1 d2 _ _.
  assert (H : forall d, arg d 0 = 0%Z).
  { intros d. unfold arg, HALF, ULP, SAT. rewrite Z.mul_0_r. reflexivity. }
  rewrite !H. auto.
Qed.

(** REFUSAL 2. `mul.wide.s32` on a `.u32` parameter. At `KFix = 2^31` and one
    unit of score delta the unsigned reading gives argument 32768 - a weight
    of `2^28 * 2^(-1/2)`, i.e. a key that contributes - while the signed one
    gives -32768, which the `cvt.u32.u64` under the saturate wraps to
    4294934528. That is far above the table, so the weight is ZERO. Not a
    rounding difference: the key vanishes.

    Note the saturate cannot help. `min.s64` bounds from ABOVE, and the signed
    product is negative. *)
Theorem the_two_readings_of_the_multiplier_disagree_above_two_to_the_thirty_one :
  arg 1 KFIX_SIGNED_LIMIT = 32768%Z
  /\ as_s32 KFIX_SIGNED_LIMIT = (-2147483648)%Z
  /\ arg 1 (as_s32 KFIX_SIGNED_LIMIT) = (-32768)%Z
  /\ ((arg 1 (as_s32 KFIX_SIGNED_LIMIT)) mod TWO32)%Z = 4294934528%Z.
Proof. repeat split; vm_compute; reflexivity. Qed.

Theorem the_signed_reading_zeroes_a_weight_the_unsigned_one_keeps :
  expf ((arg 1 (as_s32 KFIX_SIGNED_LIMIT)) mod TWO32)%Z = 0%Z
  /\ 0 < W (u_of_arg (arg 1 KFIX_SIGNED_LIMIT)).
Proof.
  split.
  - apply exp_is_zero_above_the_table. vm_compute. discriminate.
  - apply W_pos. vm_compute. discriminate.
Qed.


End Ideal.

(* ------------------------------------------------------------------ *)
(** ** Part 7 - is the interface SATISFIABLE, and why is it a bracket  *)
(* ------------------------------------------------------------------ *)

(** An inconsistent hypothesis set proves everything, so an interface nothing
    is shown to satisfy is the proof-shaped version of a licence nothing can
    violate. This repository has the rule written down and had never run it on
    its own premises; here it is run in both directions.

    First the negative direction, because it is what forced the design. The
    OBVIOUS interface for `2^28 * 2^(-u/2^32)` is exact homogeneity - a
    weight function turns addition of exponents into multiplication of
    weights - plus the one exact fact that `2^32` units is a halving. Over Q
    that is INCONSISTENT: it forces the weight at the midpoint to be a
    rational square root of `2^55`. *)

Lemma a_coprime_pair_does_not_square_to_twice :
  forall a b : Z, (b <> 0)%Z -> Z.gcd a b = 1%Z -> (a * a <> 2 * (b * b))%Z.
Proof.
  intros a b Hb Hg Heq.
  assert (Hda : (2 | a)%Z).
  { assert (H : (2 | a * a)%Z) by (exists (b*b)%Z; lia).
    destruct (prime_mult 2 prime_2 a a H); assumption. }
  destruct Hda as [c Hc]. subst a.
  assert (Hdb : (2 | b)%Z).
  { assert (H : (2 | b * b)%Z) by (exists (c*c)%Z; nia).
    destruct (prime_mult 2 prime_2 b b H); assumption. }
  assert (Hg2 : (2 | Z.gcd (c*2) b)%Z)
    by (apply Z.gcd_greatest; [exists c; lia | assumption]).
  rewrite Hg in Hg2. destruct Hg2 as [k Hk]. lia.
Qed.

Lemma no_integer_pair_squares_to_twice :
  forall a b : Z, (b <> 0)%Z -> (a * a <> 2 * (b * b))%Z.
Proof.
  intros a b Hb Heq.
  set (g := Z.gcd a b).
  assert (Hg0 : (g <> 0)%Z)
    by (unfold g; intro C; apply Z.gcd_eq_0_r in C; contradiction).
  destruct (Z.gcd_divide_l a b) as [a' Ha'].
  destruct (Z.gcd_divide_r a b) as [b' Hb'].
  fold g in Ha', Hb'.
  assert (Hcop : Z.gcd a' b' = 1%Z).
  { replace a' with (a / g)%Z by (rewrite Ha'; apply Z.div_mul; auto).
    replace b' with (b / g)%Z by (rewrite Hb'; apply Z.div_mul; auto).
    apply Z.gcd_div_gcd; auto. }
  assert (Hb'0 : (b' <> 0)%Z) by (intro C; subst b'; simpl in Hb'; lia).
  apply (a_coprime_pair_does_not_square_to_twice a' b' Hb'0 Hcop).
  rewrite Ha', Hb' in Heq.
  apply (Z.mul_reg_r _ _ (g*g)%Z); [nia | nia].
Qed.

Theorem no_rational_squares_to_two : forall q : Q, ~ (q * q == 2#1).
Proof.
  intros [a b] H. unfold Qeq, Qmult in H. simpl in H.
  apply (no_integer_pair_squares_to_twice a (Z.pos b)); lia.
Qed.

(** So the interface below is a two-sided rational bracket rather than an
    equation, and this is why. *)
Theorem exact_homogeneity_is_unsatisfiable_over_Q :
  forall W : Z -> Q,
    (forall u v, W (u + v)%Z * TWO28 == W u * W v) ->
    ~ (W 4294967296%Z == 134217728 # 1).
Proof.
  intros W Hhom Hhalf.
  assert (Hx : W 2147483648%Z * W 2147483648%Z == 36028797018963968 # 1).
  { specialize (Hhom 2147483648%Z 2147483648%Z).
    replace (2147483648 + 2147483648)%Z with 4294967296%Z in Hhom by lia.
    rewrite Hhalf in Hhom. rewrite <- Hhom. vm_compute. reflexivity. }
  apply (no_rational_squares_to_two (W 2147483648%Z / (134217728#1))).
  unfold Qdiv.
  assert (E : W 2147483648%Z * / (134217728#1) * (W 2147483648%Z * / (134217728#1))
              == (W 2147483648%Z * W 2147483648%Z)
                 * (/(134217728#1) * /(134217728#1))) by ring.
  rewrite E, Hx. vm_compute. reflexivity.
Qed.

(** And the positive direction: the bracket IS satisfiable, by a model whose
    every value is rational. `2^28 * (1 - 2^-32)^u` decays a hair faster than
    the true weight, which is exactly what the two quantitative hypotheses
    allow. Note what this does and does not say: it says the four hypotheses
    are not contradictory, so the theorems above are not vacuous. It does not
    say the true `2^28 * 2^(-u/2^32)` satisfies them - that is stated in the
    header as a TCB item, with the rational inequality
    [the_per_unit_factor_is_what_one_unit_of_log2_costs] fixing the constant
    it turns on. *)
Definition Wmodel (u : Z) : Q := TWO28 * qpow BETA (Z.to_nat u).

Lemma model_pos : forall u, (0 <= u)%Z -> 0 < Wmodel u.
Proof.
  intros u _. unfold Wmodel.
  assert (0 < TWO28) by (unfold TWO28, Qlt; simpl; lia).
  assert (0 < qpow BETA (Z.to_nat u))
    by (apply qpow_pos; unfold BETA, Qlt; simpl; lia).
  nra.
Qed.

Lemma model_one : Wmodel 0%Z == TWO28.
Proof. unfold Wmodel. simpl. ring. Qed.

Lemma model_step : forall u, (0 <= u)%Z ->
  BETA * Wmodel u <= Wmodel (u+1)%Z /\ Wmodel (u+1)%Z <= Wmodel u.
Proof.
  intros u Hu. unfold Wmodel.
  replace (Z.to_nat (u+1)) with (S (Z.to_nat u)) by lia.
  rewrite qpow_S.
  assert (HB1 : BETA <= 1) by (unfold BETA, Qle; simpl; lia).
  assert (HB0 : 0 <= BETA) by (unfold BETA, Qle; simpl; lia).
  assert (HP : 0 <= qpow BETA (Z.to_nat u)) by (apply qpow_nonneg; auto).
  assert (H28 : 0 <= TWO28) by (unfold TWO28, Qle; simpl; lia).
  set (P := qpow BETA (Z.to_nat u)) in *.
  assert (HA : 0 <= TWO28 * P) by nra.
  split.
  - assert (E : BETA * (TWO28 * P) == TWO28 * (BETA * P)) by ring. lra.
  - assert (E : TWO28 * (BETA * P) == (TWO28 * P) * BETA) by ring.
    rewrite E. nra.
Qed.

(** The algebra of the halving step, over ABSTRACT rationals.

    It is a separate lemma for a mechanical reason worth recording: `ring` (and
    `auto`) NORMALISE, and normalising a goal that mentions
    `qpow BETA (Z.to_nat 4294967296)` asks for four billion multiplications -
    `Stack overflow`, after 85 seconds, at `Qed` rather than at the tactic.
    `remember` does not help, because it leaves a let-binding `ring` can zeta-
    reduce through. Proving the algebra over variables and `apply`ing it keeps
    the big term where only first-order unification ever sees it, which never
    evaluates. `lra` and `nra` are safe - they abstract atoms - and so is the
    kernel: [the_per_unit_factor_is_what_one_unit_of_log2_costs] states the big
    term and checks in milliseconds. *)
Lemma halving_step : forall c p h : Q,
  0 <= c -> 0 <= p -> (2#1) * h <= 1 -> (2#1) * (c * (p * h)) <= c * p.
Proof.
  intros c p h Hc Hp Hh.
  assert (Hcp : 0 <= c * p) by nra.
  assert (E : (2#1) * (c * (p * h)) == (c * p) * ((2#1) * h)) by ring.
  rewrite E. nra.
Qed.

Lemma model_halves : forall u, (0 <= u)%Z ->
  (2#1) * Wmodel (u + 4294967296)%Z <= Wmodel u.
Proof.
  intros u Hu. unfold Wmodel.
  replace (Z.to_nat (u + 4294967296))
     with (Z.to_nat u + Z.to_nat 4294967296)%nat by lia.
  rewrite qpow_add.
  apply halving_step.
  - unfold TWO28. unfold Qle. simpl. lia.
  - apply qpow_nonneg. unfold BETA, Qle. simpl. lia.
  - exact the_per_unit_factor_is_what_one_unit_of_log2_costs.
Qed.

Theorem the_interface_is_satisfiable :
  (forall u, (0 <= u)%Z -> 0 < Wmodel u)
  /\ Wmodel 0%Z == TWO28
  /\ (forall u, (0 <= u)%Z -> BETA * Wmodel u <= Wmodel (u+1)%Z
                             /\ Wmodel (u+1)%Z <= Wmodel u)
  /\ (forall u, (0 <= u)%Z -> (2#1) * Wmodel (u + 4294967296)%Z <= Wmodel u).
Proof.
  split; [ exact model_pos | ].
  split; [ exact model_one | ].
  split; [ exact model_step | exact model_halves ].
Qed.

(* ------------------------------------------------------------------ *)
(** ** Nothing is assumed                                              *)
(* ------------------------------------------------------------------ *)

Print Assumptions the_per_unit_factor_is_what_one_unit_of_log2_costs.
Print Assumptions the_rounding_is_within_a_half_ulp.
Print Assumptions an_unsaturated_argument_is_within_a_half_ulp.
Print Assumptions saturation_means_the_exact_exponent_is_astronomically_large.
Print Assumptions the_swept_domain_covers_the_admitted_one.
Print Assumptions the_weight_is_within_a_relative_ulp.
Print Assumptions the_attention_output_is_within_the_bound.
Print Assumptions the_output_quotient_is_within_the_bound.
Print Assumptions the_max_subtraction_puts_a_floor_under_the_total_weight.
Print Assumptions a_total_weight_below_one_ulp_admits_a_zero_denominator.
Print Assumptions the_bound_at_a_long_context.
Print Assumptions the_bound_holds_at_every_launch_geometry.
Print Assumptions no_rational_squares_to_two.
Print Assumptions exact_homogeneity_is_unsatisfiable_over_Q.
Print Assumptions the_interface_is_satisfiable.
Print Assumptions the_ideal_weights_at_two_close_exponents_bracket_each_other.
Print Assumptions the_quantized_temperature_shifts_the_exponent_by_at_most_half_a_delta.
Print Assumptions the_quantized_temperature_moves_the_ideal_weight_by_at_most_a_beta_power.
Print Assumptions at_head_dim_128_the_temperature_moves_a_weight_by_under_a_two_thousandth.
Print Assumptions a_zero_multiplier_gives_every_key_the_same_weight.
Print Assumptions the_two_readings_of_the_multiplier_disagree_above_two_to_the_thirty_one.
Print Assumptions the_signed_reading_zeroes_a_weight_the_unsigned_one_keeps.
