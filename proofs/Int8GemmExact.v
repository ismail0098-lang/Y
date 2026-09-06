(** * The int8 tensor-core GEMM computes the source dot products

    [proofs/Int8GemmSchedule.v] proves this kernel's SCHEDULE: which lane owns
    which element of C, that the split-K classes tile the contraction, and that
    the atomic reduction is order-independent.  It says nothing whatever about
    the VALUE that lands at [C[r][c]].  This file closes that, and it is the
    GPU twin of [ExactGemmWhole.the_threaded_gemm_holds_the_source_dot_products]
    - the same claim for the CPU's exact GEMM.

    It is available for THIS kernel and no other GEMM in the repository: 950 of
    the 952 [mma.sync] instructions Y emits are floating point, and an f16
    tensor-core GEMM is simply not equal to the naive nest.  The int8
    instruction accumulates into int32, integer addition is associative, and
    the relationship is an EQUALITY.

    ** THE LICENCE, AND THE DEFECT THAT WRITING THIS FOUND

    The theorem below is FALSE without a hypothesis, and the compiler was not
    checking it.  [mma...s32.s8.s8.s32] accumulates into int32 and this kernel
    has NO FLUSH - the CPU's exact GEMM widens to int64 every [Fl] k-pairs, and
    there is no equivalent here because the OUTPUT is int32 too.  So the bound
    is on the whole contraction:

    <<  | sum over k < K of A[r][k] * B[c][k] |  <=  K * m^2  <=  i32::MAX  >>

    for operand magnitude [m].  The emitter refused only on M % 16, N % 8 and
    K % 32; nothing bounded K, and nothing anywhere in [proofs/] or [tests/]
    mentioned the accumulator's range.

    **Measured on the device before any of this was written**, M=16 N=8, every
    element of A and B set to 127, one warp, grid (1,1,1):

    | K       | exact        | device        |          |
    |---------|--------------|---------------|----------|
    | 133 088 | 2 146 576 352| 2 146 576 352 | ok       |
    | 133 120 | 2 147 092 480| 2 147 092 480 | ok       |
    | 133 152 | 2 147 608 608| -2 147 358 688| WRAPPED  |
    | 133 184 | 2 148 124 736| -2 146 842 560| WRAPPED  |

    One K-step wide.  [the_measured_overflow_is_two_s_complement] reproduces
    that third row from [wrap32] alone, so the model is refereed against the
    silicon rather than asserted to describe it.

    Latent rather than live: the largest K in the corpus is 16 384, four
    orders below the bound.  That is the reason to fix it now - this
    repository's own rule is to find these while the path is still dead.

    ** WHAT IS PROVED

    - [the_lanes_cover_the_a_fragment] / [_b_fragment]: the 32 lanes' register
      bytes are a BIJECTION onto the 16x32 and 32x8 fragments.  Every element
      is loaded, once, so no fragment position keeps a value from the previous
      K step.  Both are [MixedRadix] - the eighth and ninth consumers.
    - [the_emitted_a_address_is_its_fragment_element] / [_b_]: the emitted byte
      offsets - base [(brow + 16*mi + g) * K + 4*t], register steps [8*K] and
      [16], byte [b] - address exactly the source element that bijection names.
      This is where a stride/extent confusion would live, and note A is packed
      so its stride IS K; there is no [lda] to disagree with the extent.
    - [the_k_loop_is_the_contraction]: 32 products per step over [K/32] steps
      re-index to the flat [sum over k < K].
    - [bounded_products_accumulate_exactly]: under the licence the int32
      accumulator ([ExactGemmMicro.wsum], wrapping at every step) equals the
      [Z] sum.  This is what makes the licence load-bearing rather than
      paperwork.
    - [the_emitted_int8_gemm_holds_the_source_dot_products]: the capstone.

    ** WHAT THIS DOES NOT CLAIM

    - [mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32]'s own semantics are a
      DEFINITION here, not a theorem - the trusted base, exactly as
      [vpdpwssd]'s semantics are on the CPU side.  They are pinned empirically
      by [tests/ptx_int8_mma_layout.rs], which runs the instruction on the
      device against a plain integer matmul.  A proof over [Z] cannot supply
      an ISA.
    - The ISA does not specify the order in which one [mma] sums its 32
      products, so nothing here depends on it: the bound is on the sum of
      ABSOLUTE values, which dominates every partial sum in every order.
    - Nothing about int8 QUANTIZATION.  The claim is that the kernel computes
      the integer matrix product its source names.
    - The int32 conjunct of the capstone is stated for the FLAT accumulation
      ([MC.wsum] over the K products in visit order), which is the emitted
      order at [nz = 1] - the default grid, and the case the device
      measurement above was taken in.  At [nz > 1] each class accumulates in
      int32 and the atomics combine in int32; that is covered here in [Z]
      (the first conjunct) plus the fact that the licence bounds the sum of
      ABSOLUTE values, which dominates every partial sum of every class in
      every bracketing - but it is not stated as one theorem over a wrapping
      class fold.  Doing so needs a wrapping twin of [GridStrideSplit.combine]
      and is recorded rather than done.
    - The tie is transcription-plus-gate, as in [Int8GemmSchedule]:
      [ptx_emitter] does not go through the [Ix] extraction layer, so this is a
      model checked against emitted text and against the device by
      [tests/int8_gemm_exactness.rs], not the byte-identity tie the CPU chain
      has.
    - [ExactGemmMicro] is required for its int32 model ([I32MAX], [wrap32],
      [wsum]) and for nothing else.  That model is generic - there is no
      [vpdpwssd] in it - and a second copy of "what an int32 is" is precisely
      the drift this directory exists to prevent.  Extracting it to a shared
      [Int32.v] is recorded and not done; it would touch three CPU proofs and
      muddy this increment.

    Build: coqc -R . Y Int8GemmExact.v   (Rocq 9.1)
*)

Require Import Coq.Arith.Arith.
Require Import Coq.micromega.Lia.
Require Import Coq.ZArith.ZArith.

Require MixedRadix.
Require Decomposition.
Require GridStrideSplit.
Require Int8GemmSchedule.
Require ExactGemmMicro.

Module MR  := MixedRadix.
Module D   := Decomposition.
Module GS  := GridStrideSplit.
Module SCH := Int8GemmSchedule.
Module MC  := ExactGemmMicro.

Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(** ** The range fold                                                  *)
(* ------------------------------------------------------------------ *)

(** Taken from [Decomposition] rather than restated - the fourth file to reuse
    it, and the range fold is still not a property of any decomposition. *)
Definition sum_k (f : nat -> Z) (n : nat) : Z := D.acc_range Z.add f 0 n.

(** Pointwise-equal functions fold alike.  Used instead of
    [functional_extensionality], which is an AXIOM - every [Print Assumptions]
    in this directory must report "Closed under the global context". *)
Lemma acc_range_ext : forall f h lo len,
  (forall i, f i = h i) ->
  D.acc_range Z.add f lo len = D.acc_range Z.add h lo len.
Proof.
  intros f h lo len H. induction len as [| len IH]; cbn [D.acc_range].
  - reflexivity.
  - rewrite IH, H. reflexivity.
Qed.

Lemma acc_range_shift : forall f lo len,
  D.acc_range Z.add f lo len = D.acc_range Z.add (fun p => f (lo + p)%nat) 0 len.
Proof.
  intros f lo len. induction len as [| len IH]; cbn [D.acc_range].
  - reflexivity.
  - rewrite IH. replace (lo + (0 + len))%nat with (lo + len)%nat by lia. reflexivity.
Qed.

(** Peel one step. *)
Lemma acc_range_succ : forall f lo len,
  D.acc_range Z.add f lo (Datatypes.S len)
  = D.acc_range Z.add f lo len + f (lo + len)%nat.
Proof. reflexivity. Qed.

(** Blocks of [B] terms re-index to the flat range.

    **Stated for an ABSTRACT [B], and that is not generality for its own sake -
    it is the difference between 0.27 s and not terminating.**  Proved directly
    at 32 the tactics all succeed and the goal closes to something
    SYNTACTICALLY IDENTICAL on both sides, and then [Qed] does not return: a
    [nat] literal is unary, so a proof term carrying `32` carries thirty-two
    nested [S] through every conversion check, inside a fold that is itself 32
    deep.  With [B] a variable nothing can unfold at all, and the literal
    enters only at the [apply] below, as a single unification.

    Same landmine as [SoftmaxErrorBound]'s [ring] on a large power, wearing a
    different hat: keep every literal out of anything that normalises. *)
Lemma acc_range_blocks : forall (B : nat) (g : nat -> Z) (S : nat),
  D.acc_range Z.add (fun s => D.acc_range Z.add (fun p => g (B * s + p)%nat) 0 B) 0 S
  = D.acc_range Z.add g 0 (B * S)%nat.
Proof.
  intros B g S. induction S as [| S IH].
  - rewrite Nat.mul_0_r. reflexivity.
  - rewrite acc_range_succ. cbv beta.
    replace (0 + S)%nat with S by lia.
    replace (B * Datatypes.S S)%nat with (B * S + B)%nat by lia.
    rewrite D.sum_range_split.
    replace (0 + B * S)%nat with (B * S)%nat by lia.
    rewrite IH.
    rewrite (acc_range_shift g (B * S)%nat B).
    reflexivity.
Qed.

(** 32 products per [mma], [K/32] steps, and the flat contraction. *)
Lemma sum_k_blocks : forall g S,
  sum_k (fun s => sum_k (fun p => g (32 * s + p)%nat) 32) S = sum_k g (32 * S)%nat.
Proof. intros g S. unfold sum_k. apply acc_range_blocks. Qed.

(* ------------------------------------------------------------------ *)
(** ** The fragment maps, and that the lanes cover the fragments       *)
(* ------------------------------------------------------------------ *)

(** With [g = laneid >> 2] in 0..7 and [t = laneid & 3] in 0..3, lane [(g,t)]
    holds four [.b32] registers of A and two of B.  Register [r], byte [b] of
    the A group is fragment element [(a_row g r, a_col t r b)] of the 16x32
    tile; register [r], byte [b] of the B group is [(b_col g, b_k t r b)] of
    the 32x8 tile, which is column-major so its lane holds four CONTIGUOUS
    k-values.

    Derived and validated in [tests/ptx_int8_mma_layout.rs]. *)
Definition a_row (g r : nat) : nat := (g + 8 * (r mod 2))%nat.
Definition a_col (t r b : nat) : nat := (4 * t + 16 * (r / 2) + b)%nat.
Definition b_col (g : nat) : nat := g.
Definition b_k   (t r b : nat) : nat := (4 * t + 16 * r + b)%nat.

(** The accumulator: [d0,d1] at row [g], columns [2t] and [2t+1]; [d2,d3] at
    row [g+8], same columns.  This is the map [Int8GemmSchedule] already
    partitions; it is restated here only to compose with it. *)
Definition d_row (g j : nat) : nat := (g + 8 * (j / 2))%nat.
Definition d_col (t j : nat) : nat := (2 * t + j mod 2)%nat.

Lemma a_in_range : forall g t r b,
  (g < 8)%nat -> (t < 4)%nat -> (r < 4)%nat -> (b < 4)%nat ->
  (a_row g r < 16)%nat /\ (a_col t r b < 32)%nat.
Proof.
  intros g t r b Hg Ht Hr Hb. unfold a_row, a_col.
  pose proof (Nat.mod_upper_bound r 2 ltac:(lia)) as Hm.
  assert (r / 2 < 2)%nat by (apply Nat.Div0.div_lt_upper_bound; lia).
  lia.
Qed.

Lemma b_in_range : forall g t r b,
  (g < 8)%nat -> (t < 4)%nat -> (r < 2)%nat -> (b < 4)%nat ->
  (b_col g < 8)%nat /\ (b_k t r b < 32)%nat.
Proof. intros g t r b Hg Ht Hr Hb. unfold b_col, b_k. lia. Qed.

(** **Injective**: no two lane/register/byte triples name one fragment element,
    so nothing is loaded twice and no two source elements collide.  Both digits
    fall straight out of [MixedRadix] - [a_row] is a two-digit index in radix 8
    and [a_col] a three-digit one in radices 4, 4, 2. *)
Lemma a_frag_injective : forall g1 t1 r1 b1 g2 t2 r2 b2,
  (g1 < 8)%nat -> (t1 < 4)%nat -> (r1 < 4)%nat -> (b1 < 4)%nat ->
  (g2 < 8)%nat -> (t2 < 4)%nat -> (r2 < 4)%nat -> (b2 < 4)%nat ->
  a_row g1 r1 = a_row g2 r2 -> a_col t1 r1 b1 = a_col t2 r2 b2 ->
  g1 = g2 /\ t1 = t2 /\ r1 = r2 /\ b1 = b2.
Proof.
  intros g1 t1 r1 b1 g2 t2 r2 b2 Hg1 Ht1 Hr1 Hb1 Hg2 Ht2 Hr2 Hb2 Hrow Hcol.
  unfold a_row, a_col in *.
  pose proof (Nat.mod_upper_bound r1 2 ltac:(lia)) as M1.
  pose proof (Nat.mod_upper_bound r2 2 ltac:(lia)) as M2.
  assert (Q1 : (r1 / 2 < 2)%nat) by (apply Nat.Div0.div_lt_upper_bound; lia).
  assert (Q2 : (r2 / 2 < 2)%nat) by (apply Nat.Div0.div_lt_upper_bound; lia).
  (* the row: radix 8, high digit `r mod 2`, low digit `g` *)
  assert (Hr : (r1 mod 2)%nat = (r2 mod 2)%nat /\ g1 = g2).
  { apply (MR.quot_rem_unique 8); [ lia | lia | lia | lia ]. }
  (* the column: b + 4*(t + 4*(r/2)) *)
  assert (Hc : (r1 / 2)%nat = (r2 / 2)%nat /\ t1 = t2 /\ b1 = b2).
  { apply (MR.two_digit_unique 4 4); [ lia | lia | lia | lia | lia | lia | lia ]. }
  destruct Hr as [Hrm Hg]. destruct Hc as [Hrd [Ht Hb]].
  pose proof (Nat.div_mod_eq r1 2) as E1.
  pose proof (Nat.div_mod_eq r2 2) as E2.
  repeat split; try assumption. lia.
Qed.

Lemma b_frag_injective : forall g1 t1 r1 b1 g2 t2 r2 b2,
  (g1 < 8)%nat -> (t1 < 4)%nat -> (r1 < 2)%nat -> (b1 < 4)%nat ->
  (g2 < 8)%nat -> (t2 < 4)%nat -> (r2 < 2)%nat -> (b2 < 4)%nat ->
  b_col g1 = b_col g2 -> b_k t1 r1 b1 = b_k t2 r2 b2 ->
  g1 = g2 /\ t1 = t2 /\ r1 = r2 /\ b1 = b2.
Proof.
  intros g1 t1 r1 b1 g2 t2 r2 b2 Hg1 Ht1 Hr1 Hb1 Hg2 Ht2 Hr2 Hb2 Hcol Hk.
  unfold b_col, b_k in *.
  assert (H : r1 = r2 /\ t1 = t2 /\ b1 = b2).
  { apply (MR.two_digit_unique 4 4); [ lia | lia | lia | lia | lia | lia | lia ]. }
  destruct H as [H1 [H2 H3]]. repeat split; assumption.
Qed.

(** **Onto**: every one of the 16x32 A positions and 32x8 B positions is held
    by some lane.  This is the leg that says the fragment the [mma] reads is
    fully populated by the emitted loads - without it the theorem below would
    describe a tile with holes in it. *)
Theorem the_lanes_cover_the_a_fragment : forall i p,
  (i < 16)%nat -> (p < 32)%nat ->
  exists g t r b,
    (g < 8)%nat /\ (t < 4)%nat /\ (r < 4)%nat /\ (b < 4)%nat /\
    a_row g r = i /\ a_col t r b = p.
Proof.
  intros i p Hi Hp.
  destruct (MR.pack_onto 8 2 i ltac:(lia) ltac:(lia)) as [qi [ri [Hqi [Hri Hi']]]].
  destruct (MR.pack_onto 16 2 p ltac:(lia) ltac:(lia)) as [qp [rp [Hqp [Hrp Hp']]]].
  destruct (MR.pack_onto 4 4 rp ltac:(lia) ltac:(lia)) as [qr [rr [Hqr [Hrr Hp'']]]].
  unfold MR.pack in *.
  exists ri, qr, (qi + 2 * qp)%nat, rr.
  assert (Hm : ((qi + 2 * qp) mod 2)%nat = qi).
  { replace (qi + 2 * qp)%nat with (qp * 2 + qi)%nat by lia.
    destruct (MR.pack_unpack 2 qp qi ltac:(lia) ltac:(lia)) as [_ Hlo].
    unfold MR.lo, MR.pack in Hlo. exact Hlo. }
  assert (Hd : ((qi + 2 * qp) / 2)%nat = qp).
  { replace (qi + 2 * qp)%nat with (qp * 2 + qi)%nat by lia.
    destruct (MR.pack_unpack 2 qp qi ltac:(lia) ltac:(lia)) as [Hhi _].
    unfold MR.hi, MR.pack in Hhi. exact Hhi. }
  unfold a_row, a_col. rewrite Hm, Hd.
  repeat split; try lia.
Qed.

Theorem the_lanes_cover_the_b_fragment : forall c p,
  (c < 8)%nat -> (p < 32)%nat ->
  exists g t r b,
    (g < 8)%nat /\ (t < 4)%nat /\ (r < 2)%nat /\ (b < 4)%nat /\
    b_col g = c /\ b_k t r b = p.
Proof.
  intros c p Hc Hp.
  destruct (MR.pack_onto 16 2 p ltac:(lia) ltac:(lia)) as [qp [rp [Hqp [Hrp Hp']]]].
  destruct (MR.pack_onto 4 4 rp ltac:(lia) ltac:(lia)) as [qr [rr [Hqr [Hrr Hp'']]]].
  unfold MR.pack in *.
  exists c, qr, qp, rr. unfold b_col, b_k. repeat split; lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The emitted addresses ARE those fragment elements               *)
(* ------------------------------------------------------------------ *)

(** The emitted A pointer for m-tile [mi] of the warp tile:

      add.u32     row, brow, mi*16      add.u32 row, row, g
      mul.lo.u32  off, row, K           add.u32 off, off, 4*t
      cvt.u64.u32 / add.u64             ->  &A[brow + 16*mi + g][4*t]

    and the four registers are loaded at byte offsets [0], [8*K], [16] and
    [8*K + 16] - one row-half down is [8*K] because A's rows are K bytes apart,
    and [16] is sixteen k-values along because A is int8.  Byte [b] of a 32-bit
    load is the [b]-th of four contiguous k-values.

    A is PACKED: its row stride IS K.  There is no [lda] here to disagree with
    the extent, which is exactly where this repository's recorded address bugs
    live - twelve of them in the CPU GEMM "correct only because [lda == K] made
    stride and extent the same number".  Here they are the same number by the
    kernel's own contract rather than by coincidence. *)
Definition a_emitted_offset (K brow mi kk g t r b : nat) : nat :=
  ((brow + 16 * mi + g) * K + 4 * t + kk
   + 8 * K * (r mod 2) + 16 * (r / 2) + b)%nat.

(** B is [N][K] and its two registers are [16] bytes apart - no [8*K] step,
    because a B lane's k-values are contiguous where an A lane's rows are not.
    The two offsets are DIFFERENT expressions, and the layout note in
    [tests/ptx_int8_mma_layout.rs] records that their coincidence at [r = 0] is
    a coincidence of the two shapes and not a shared rule. *)
Definition b_emitted_offset (K bcol ni kk g t r b : nat) : nat :=
  ((bcol + 8 * ni + g) * K + 4 * t + kk + 16 * r + b)%nat.

(** Row-major addressing, as the kernel's own header comment states it:
    A is [M][K], B is [N][K], both int8. *)
Definition element (K row col : nat) : nat := (row * K + col)%nat.

Theorem the_emitted_a_address_is_its_fragment_element :
  forall K brow mi kk g t r b,
    a_emitted_offset K brow mi kk g t r b
    = element K (brow + 16 * mi + a_row g r)%nat (kk + a_col t r b)%nat.
Proof.
  intros. unfold a_emitted_offset, element, a_row, a_col. ring.
Qed.

Theorem the_emitted_b_address_is_its_fragment_element :
  forall K bcol ni kk g t r b,
    b_emitted_offset K bcol ni kk g t r b
    = element K (bcol + 8 * ni + b_col g)%nat (kk + b_k t r b)%nat.
Proof.
  intros. unfold b_emitted_offset, element, b_col, b_k. ring.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The instruction, as a DEFINITION (the trusted base)             *)
(* ------------------------------------------------------------------ *)

(** [mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32] computes, for every
    element [(i, jj)] of the 16x8 accumulator,

      D[i][jj] = C[i][jj] + sum over p < 32 of Afrag[i][p] * Bfrag[p][jj]

    **This is an ASSUMPTION about the ISA, not a theorem.**  A proof over [Z]
    cannot supply an instruction set.  It sits in the trusted base exactly
    where [vpdpwssd]'s semantics sit for the CPU chain, and it is pinned
    empirically by [tests/ptx_int8_mma_layout.rs], which runs the real
    instruction on the device against a plain integer matmul.

    Note what is deliberately NOT assumed: the ORDER in which one [mma] sums
    its 32 products.  The ISA does not specify it, so nothing below depends on
    it - the licence bounds the sum of ABSOLUTE values, which dominates every
    partial sum in every order and at every granularity. *)
Definition mma (Afrag Bfrag : nat -> nat -> Z) (i jj : nat) : Z :=
  sum_k (fun p => Afrag i p * Bfrag p jj) 32.

(** The fragments the emitted loads populate at K-offset [kk].  That they hold
    these values is [the_emitted_*_address_is_its_fragment_element]; that they
    hold ALL of them is [the_lanes_cover_the_*_fragment]. *)
Definition a_frag (A : nat -> nat -> Z) (brow mi kk : nat) : nat -> nat -> Z :=
  fun i p => A (brow + 16 * mi + i)%nat (kk + p)%nat.
Definition b_frag (B : nat -> nat -> Z) (bcol ni kk : nat) : nat -> nat -> Z :=
  fun p jj => B (bcol + 8 * ni + jj)%nat (kk + p)%nat.

(* ------------------------------------------------------------------ *)
(** ** The K loop, and every split factor                              *)
(* ------------------------------------------------------------------ *)

(** One K step of the emitted loop: the pointers advance by [32 * nz] bytes per
    iteration from a base of [32 * z], so CTA [z] visits step indices congruent
    to [z] modulo [nz] - the residue classes [GridStrideSplit] partitions. *)
Definition step (A B : nat -> nat -> Z) (brow bcol mi ni i jj s : nat) : Z :=
  mma (a_frag A brow mi (32 * s)) (b_frag B bcol ni (32 * s)) i jj.

Definition prod (A B : nat -> nat -> Z) (row col k : nat) : Z :=
  A row k * B col k.

Theorem the_k_loop_is_the_contraction :
  forall A B brow bcol mi ni i jj S,
    sum_k (step A B brow bcol mi ni i jj) S
    = sum_k (prod A B (brow + 16 * mi + i)%nat (bcol + 8 * ni + jj)%nat) (32 * S)%nat.
Proof.
  intros. unfold step, mma, a_frag, b_frag, prod.
  apply (sum_k_blocks
           (fun k => A (brow + 16 * mi + i)%nat k * B (bcol + 8 * ni + jj)%nat k) S).
Qed.

Lemma sum_upto_is_sum_k : forall f S, GS.sum_upto Z.add f S = sum_k f S.
Proof. reflexivity. Qed.

(** **Every split factor gives the whole contraction.**  [GridStrideSplit]
    supplies this with no new reasoning - the residue classes are the same
    decomposition the attention kernel and this kernel's own output tiling use.
    What is new is only that the CLASS TERMS are [mma] steps. *)
Theorem every_split_factor_gives_the_contraction :
  forall A B brow bcol mi ni i jj S nz,
    (0 < nz)%nat ->
    GS.combine Z.add (step A B brow bcol mi ni i jj) nz S nz
    = sum_k (prod A B (brow + 16 * mi + i)%nat (bcol + 8 * ni + jj)%nat) (32 * S)%nat.
Proof.
  intros A B brow bcol mi ni i jj S nz Hnz.
  rewrite GS.grid_stride_exact by assumption.
  rewrite sum_upto_is_sum_k.
  apply the_k_loop_is_the_contraction.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The licence                                                     *)
(* ------------------------------------------------------------------ *)

(** The largest contraction an int32 accumulator holds exactly, for full-range
    int8 operands: [|sum| <= K * 127^2] must fit [i32::MAX].

    The emitter refuses at K-STEP granularity, because [K mod 32 = 0] is
    already its shape precondition and a bound that is not a multiple of 32 is
    not expressible as a refusal it can state. *)
Definition MAX_EXACT_K : Z := MC.I32MAX / (127 * 127).
Definition MAX_EXACT_K_STEPS : Z := MAX_EXACT_K / 32 * 32.

Theorem the_bound_is_one_k_step_wide :
  MAX_EXACT_K = 133144 /\ MAX_EXACT_K_STEPS = 133120.
Proof. split; vm_compute; reflexivity. Qed.

(** The two sides of it.  [MAX_EXACT_K_STEPS] is admissible and the next K step
    is not - one step wide, which is what makes an off-by-one in the emitter's
    constant visible rather than absorbed. *)
Theorem the_licence_admits_133120_and_refuses_133152 :
  133120 * (127 * 127) <= MC.I32MAX /\ MC.I32MAX < 133152 * (127 * 127).
Proof. unfold MC.I32MAX. split; lia. Qed.

(** **The refutation, refereed against the silicon.**  The device returned
    -2147358688 at K = 133152 with every operand 127.  That number is
    reproduced here from [wrap32] alone - the model was not fitted to it, and a
    model that merely said "it overflows" would agree with any wrong answer. *)
Theorem the_measured_overflow_is_two_s_complement :
  133152 * (127 * 127) = 2147608608
  /\ MC.wrap32 2147608608 = -2147358688.
Proof. split; vm_compute; reflexivity. Qed.

(** Every partial sum of a bounded sequence is bounded by its length times the
    bound - which dominates the accumulator's value at every granularity and in
    every summation order, so nothing below depends on the [mma]'s unspecified
    internal order. *)
Lemma acc_range_abs_bound : forall f lo len c,
  0 <= c -> (forall i, Z.abs (f i) <= c) ->
  Z.abs (D.acc_range Z.add f lo len) <= Z.of_nat len * c.
Proof.
  intros f lo len c Hc H. induction len as [| len IH]; cbn [D.acc_range].
  - simpl. lia.
  - rewrite Nat2Z.inj_succ.
    pose proof (H (lo + len)%nat) as Hi.
    pose proof (Z.abs_triangle (D.acc_range Z.add f lo len) (f (lo + len)%nat)) as Ht.
    lia.
Qed.

(** **The licence is load-bearing, and this is where.**  [ExactGemmMicro.wsum]
    wraps at EVERY step; under the licence it equals the [Z] sum. *)
Theorem bounded_products_accumulate_exactly : forall f n c,
  0 <= c -> (forall i, Z.abs (f i) <= c) -> Z.of_nat n * c <= MC.I32MAX ->
  MC.wsum f 0 n = sum_k f n.
Proof.
  intros f n c Hc H Hlic. unfold sum_k.
  rewrite <- MC.sum_from_is_acc_range.
  apply MC.flush_exact_in_int32.
  intros i Hi.
  pose proof (acc_range_abs_bound f 0 i c Hc H) as Hb.
  rewrite MC.sum_from_is_acc_range.
  assert (Z.of_nat i * c <= Z.of_nat n * c) by (apply Z.mul_le_mono_nonneg_r; [ lia | apply Nat2Z.inj_le; lia ]).
  unfold MC.in_i32, MC.I32MIN, MC.I32MAX in *. lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The capstone                                                    *)
(* ------------------------------------------------------------------ *)

(** The row the emitted base register addresses IS the row the schedule proof
    partitions, so every theorem in [Int8GemmSchedule] about [warp_row] applies
    to the element this file gives a value to. *)
Lemma the_addressed_row_is_the_schedules_row : forall mt ty mi i,
  (ty * (mt * 16) + 16 * mi + i)%nat = SCH.warp_row mt ty mi i.
Proof. intros. unfold SCH.warp_row, SCH.MMA_M. lia. Qed.

Lemma the_addressed_col_is_the_schedules_col : forall nt tx ni j,
  (tx * (nt * 8) + 8 * ni + j)%nat = SCH.warp_col nt tx ni j.
Proof. intros. unfold SCH.warp_col, SCH.MMA_N. lia. Qed.

(** **The theorem this file exists for.**

    For every launch geometry - any split factor [nz] - and every lane of every
    mma of every warp tile, the emitted kernel's accumulator holds exactly

      sum over k < K of A[row][k] * B[col][k]

    at the [(row, col)] the schedule proof says that lane owns, with no
    hypothesis that [nz] divides [K/32].  The two conjuncts are the two halves
    the kernel actually has: the grid-stride split in [Z], and the int32
    accumulator that carries it not wrapping.

    The LICENCE is what makes the second conjunct true, and the emitter now
    refuses a K that violates it. *)
Theorem the_emitted_int8_gemm_holds_the_source_dot_products :
  forall A B m K S nz mt nt ty tx mi ni g t j,
    0 <= m ->
    (forall r k, Z.abs (A r k) <= m) ->
    (forall c k, Z.abs (B c k) <= m) ->
    Z.of_nat K * (m * m) <= MC.I32MAX ->
    (K = 32 * S)%nat -> (0 < nz)%nat ->
    let row := SCH.warp_row mt ty mi (d_row g j) in
    let col := SCH.warp_col nt tx ni (d_col t j) in
    GS.combine Z.add
      (step A B (ty * (mt * 16))%nat (tx * (nt * 8))%nat mi ni (d_row g j) (d_col t j))
      nz S nz
    = sum_k (prod A B row col) K
    /\ MC.wsum (prod A B row col) 0 K = sum_k (prod A B row col) K.
Proof.
  intros A B m K S nz mt nt ty tx mi ni g t j Hm HA HB Hlic HK Hnz row col.
  subst row col. split.
  - rewrite (every_split_factor_gives_the_contraction
               A B (ty * (mt * 16))%nat (tx * (nt * 8))%nat mi ni
               (d_row g j) (d_col t j) S nz Hnz).
    rewrite the_addressed_row_is_the_schedules_row,
            the_addressed_col_is_the_schedules_col, HK.
    reflexivity.
  - apply (bounded_products_accumulate_exactly _ _ (m * m)).
    + nia.
    + intros i. unfold prod. rewrite Z.abs_mul.
      apply Z.mul_le_mono_nonneg; try apply Z.abs_nonneg; [ apply HA | apply HB ].
    + exact Hlic.
Qed.

(* ------------------------------------------------------------------ *)
(** ** No axioms                                                       *)
(* ------------------------------------------------------------------ *)

Print Assumptions acc_range_ext.
Print Assumptions acc_range_shift.
Print Assumptions acc_range_blocks.
Print Assumptions sum_k_blocks.
Print Assumptions a_in_range.
Print Assumptions b_in_range.
Print Assumptions a_frag_injective.
Print Assumptions b_frag_injective.
Print Assumptions the_lanes_cover_the_a_fragment.
Print Assumptions the_lanes_cover_the_b_fragment.
Print Assumptions the_emitted_a_address_is_its_fragment_element.
Print Assumptions the_emitted_b_address_is_its_fragment_element.
Print Assumptions the_k_loop_is_the_contraction.
Print Assumptions every_split_factor_gives_the_contraction.
Print Assumptions the_bound_is_one_k_step_wide.
Print Assumptions the_licence_admits_133120_and_refuses_133152.
Print Assumptions the_measured_overflow_is_two_s_complement.
Print Assumptions acc_range_abs_bound.
Print Assumptions bounded_products_accumulate_exactly.
Print Assumptions the_addressed_row_is_the_schedules_row.
Print Assumptions the_addressed_col_is_the_schedules_col.
Print Assumptions the_emitted_int8_gemm_holds_the_source_dot_products.
