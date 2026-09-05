(** * The int8 tensor-core GEMM's schedule, and the launch contract it assumed.

    This is the only GPU GEMM in this repository whose EXACTNESS is available
    to prove.  Counted over every committed [.ptx]: of the 952 [mma.sync]
    instructions this compiler emits, 854 are [f32.f16.f16.f32], 96 are
    [f32.e4m3.e4m3.f32] and **2** are [s32.s8.s8.s32].  Floating-point
    accumulation is not associative, so for 950 of them the kernel-vs-spec
    relationship is not an equality and no theorem of this shape exists.  The
    int8 instruction accumulates into an int32 EXACTLY, so it is.

    Measured before this file was written, on an RTX 4070 Ti SUPER, so that
    what is proved here is a property of a kernel that ships rather than of a
    demonstration:

      - 15,439 G MAC/s at 4096^3, which is **0.41x** cuBLASLt's [_int_mm]
        (38,090) - a real kernel, not a stub.  An earlier note in this repo
        called it a stub on the strength of a GREP for [cp.async] / [ldmatrix]
        / [bar.sync]; running it says otherwise.
      - 4% of the measured int8 [mma] ISA ceiling (374,027 G MAC/s, doubling-
        verified, 32 live [IMMA.16832.S8.S8] in the SASS; the same probe in
        f16 reads 92,482 G MAC/s = 185 TFLOPS against this card's ~176 spec,
        which is what validates the accounting).
      - It is L2-BANDWIDTH bound, at 0.1875 bytes per MAC, because it has no
        shared-memory staging: ~2,890 GB/s flat at 2048/4096/6144, then a
        3.7x collapse at 8192 when the working set leaves the 48 MB L2.

    ** The defect this file is about, measured before it was written

    The schedule gives one 16x8 output tile to one WARP, and the grid is
    [(N/8, M/16, splits)].  A CTA therefore has exactly 32 threads of work
    however many it is launched with - and nothing checked.  The emitted
    kernel contained ONE predicate, the K-loop bound, and never mentioned
    [%ntid.x].

    At M=64 N=32 K=128 with every element of A = 3 and of B = 5, so every
    element of C must be exactly K*15 = 1920:

      block (32,1,1)   correct
      block (64,1,1)   1344 of 2048 elements wrong, C[256] = 3840
      block (128,1,1)  same, and it reads row 79 of a 64-row A

    3840 is EXACTLY DOUBLE, which is the mechanism: warp 1's lane index gives
    [g = tid/4] in 8..15 instead of 0..7, so its two A-row reads land at
    [cy*16+g] and [cy*16+g+8] - the second of which is the NEXT tile's rows -
    and [red.global.add.s32] sums that second product into the same output.

    That is worse here than it would be in an ordinary kernel, because this
    kernel's entire advertised claim is a bit-identical answer at every launch
    geometry ([tests/gpu_batch_invariance.rs]).  A wrong block size falsifies
    the claim silently.  [the_guard_is_what_confines_a_warp_to_its_own_tile]
    below is that mechanism stated as arithmetic, and
    [without_the_guard_a_second_warp_lands_in_the_next_tile] is its refutation.

    ** What is instantiated rather than re-derived

    The split-K is STRIPED over [%ctaid.z]: CTA z takes every [nctaid.z]-th
    32-wide K step starting at z.  That is residue classes, which is exactly
    [GridStrideSplit]'s decomposition, and the partials combine through
    [red.global.add.s32] in whatever order the scheduler produces, which is
    exactly its [atomics_may_land_in_any_order].  The emitter states all of
    this in a prose comment ("order-independent by construction - the same
    result for every grid, every launch, every scheduling accident") and
    nothing had checked it; [tests/gpu_batch_invariance.rs] sweeps seven split
    factors, which is seven points rather than a property.

    The output tiling and the lane decomposition are positional indices, so
    [MixedRadix] discharges them with no new reasoning - the fifth and sixth
    consumers of that schema.

    ** What this does NOT claim

    - Nothing about [mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32]'s own
      semantics, nor about the per-lane fragment layout.  Those are ISA facts,
      pinned empirically by [tests/ptx_int8_mma_layout.rs], which runs the
      instruction on the device against a plain integer matmul.
    - Nothing about int8 QUANTIZATION.  The claim is that the kernel computes
      the integer matrix product its source names, not that that product is a
      good approximation of anything.
    - The tie is transcription-plus-gate, as in [GpuWarpTiling]: [ptx_emitter]
      does not go through the [Ix] extraction layer, so this is a model checked
      against emitted text by [tests/int8_gemm_launch_contract.rs], not the
      byte-identity tie the CPU chain has.
    - One axis of the output at a time, as in [GpuWarpTiling].

    Build: coqc -R . Y Int8GemmSchedule.v   (Rocq 9.1)
*)

Require Import Coq.Arith.Arith.
Require Import Coq.micromega.Lia.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Lists.List.
Require Import Coq.Sorting.Permutation.
Import ListNotations.

Require MixedRadix.
Require GridStrideSplit.

Module MR := MixedRadix.
Module GS := GridStrideSplit.

Open Scope nat_scope.

(* ------------------------------------------------------------------ *)
(** ** The emitted schedule                                            *)
(* ------------------------------------------------------------------ *)

(** The mma shape.  [m16n8k32] means one warp produces a 16x8 tile of C from
    32 k-elements per step, so the emitter's refusal is
    [m mod 16 = 0 /\ n mod 8 = 0 /\ k mod 32 = 0]. *)
Definition MMA_M : nat := 16.
Definition MMA_N : nat := 8.
Definition MMA_K : nat := 32.

(** Lanes.  The emitter writes

      shr.u32 %g, %tid, 2      (* g = tid / 4 *)
      and.b32 %t, %tid, 3      (* t = tid mod 4 *)

    and both are validated against the ISA in tests/ptx_int8_mma_layout.rs. *)
Definition lane_g (tid : nat) : nat := tid / 4.
Definition lane_t (tid : nat) : nat := tid mod 4.

(** The two A rows a lane reads: [&A[cy*16 + g]] and the load at [+8*k], i.e.
    8 rows down. *)
Definition a_row (cy g half : nat) : nat := cy * MMA_M + g + 8 * half.

(** The output tile a CTA owns. *)
Definition tile_row (cy i : nat) : nat := cy * MMA_M + i.
Definition tile_col (cx j : nat) : nat := cx * MMA_N + j.

(** The split-K striping, in units of 32-wide K steps.  The emitter seeds
    [kk] with [z*32] and advances it by [nz*32], so in step-index space CTA
    [z] visits [z, z+nz, z+2nz, ...]. *)
Definition k_steps (k : nat) : nat := k / MMA_K.
Definition owner (nz j : nat) : nat := j mod nz.

(* ------------------------------------------------------------------ *)
(** ** The lane decomposition is a bijection on a warp                 *)
(* ------------------------------------------------------------------ *)

Theorem lane_decomposition_is_injective :
  forall tid1 tid2,
    lane_g tid1 = lane_g tid2 -> lane_t tid1 = lane_t tid2 -> tid1 = tid2.
Proof.
  intros t1 t2 Hg Ht. unfold lane_g, lane_t in *.
  assert (H1 : t1 = 4 * (t1 / 4) + t1 mod 4) by (apply Nat.div_mod_eq).
  assert (H2 : t2 = 4 * (t2 / 4) + t2 mod 4) by (apply Nat.div_mod_eq).
  lia.
Qed.

Theorem a_warps_lanes_fill_the_tile :
  forall tid, tid < 32 -> lane_g tid < 8 /\ lane_t tid < 4.
Proof.
  intros tid H. unfold lane_g, lane_t. split.
  - apply Nat.Div0.div_lt_upper_bound. lia.
  - apply Nat.mod_upper_bound. lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The launch guard                                                *)
(* ------------------------------------------------------------------ *)

(** **The guard, stated as what it buys.**  A lane reads A rows [cy*16+g] and
    [cy*16+g+8]; both lie inside this CTA's own 16-row tile exactly when
    [g < 8], and the emitted guard [tid < 32] gives that. *)
Theorem the_guard_is_what_confines_a_warp_to_its_own_tile :
  forall tid cy half,
    tid < 32 -> half < 2 ->
    cy * MMA_M <= a_row cy (lane_g tid) half
    /\ a_row cy (lane_g tid) half < cy * MMA_M + MMA_M.
Proof.
  intros tid cy half Htid Hh.
  destruct (a_warps_lanes_fill_the_tile tid Htid) as [Hg _].
  unfold a_row, MMA_M. lia.
Qed.

(** **The refutation, at the block size that was measured wrong.**  With a
    64-thread block, lane 32 (the first of warp 1) has [g = 8], and its
    [half = 1] row is [cy*16 + 16] - the first row of the NEXT tile.  That is
    the double-count observed as [C[256] = 3840] against a correct 1920. *)
Theorem without_the_guard_a_second_warp_lands_in_the_next_tile :
  lane_g 32 = 8 /\ a_row 0 (lane_g 32) 1 = 16 /\ ~ (a_row 0 (lane_g 32) 1 < MMA_M).
Proof. unfold lane_g, a_row, MMA_M. cbn. repeat split; lia. Qed.

(** And the guard is not vacuous: it excludes exactly the threads that would
    misbehave, and no thread of warp 0. *)
Theorem the_guard_excludes_nothing_a_warp_needs :
  forall tid, tid < 32 -> lane_g tid < 8.
Proof. intros tid H. apply (a_warps_lanes_fill_the_tile tid H). Qed.

Theorem the_guard_is_warp_uniform :
  forall tid1 tid2, tid1 / 32 = tid2 / 32 -> (tid1 <? 32) = (tid2 <? 32).
Proof.
  intros t1 t2 H.
  destruct (Nat.ltb_spec t1 32) as [H1 | H1]; destruct (Nat.ltb_spec t2 32) as [H2 | H2];
    try reflexivity.
  - assert (t1 / 32 = 0) by (apply Nat.div_small; lia).
    assert (0 < t2 / 32) by (apply Nat.div_str_pos; lia). lia.
  - assert (t2 / 32 = 0) by (apply Nat.div_small; lia).
    assert (0 < t1 / 32) by (apply Nat.div_str_pos; lia). lia.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The output tiling is a partition                                *)
(* ------------------------------------------------------------------ *)

Theorem tile_row_injective :
  forall cy1 i1 cy2 i2,
    i1 < MMA_M -> i2 < MMA_M ->
    tile_row cy1 i1 = tile_row cy2 i2 -> cy1 = cy2 /\ i1 = i2.
Proof.
  intros cy1 i1 cy2 i2 H1 H2 H.
  apply (MR.quot_rem_unique MMA_M); [ unfold MMA_M; lia | exact H1 | exact H2 | exact H ].
Qed.

Theorem tile_col_injective :
  forall cx1 j1 cx2 j2,
    j1 < MMA_N -> j2 < MMA_N ->
    tile_col cx1 j1 = tile_col cx2 j2 -> cx1 = cx2 /\ j1 = j2.
Proof.
  intros cx1 j1 cx2 j2 H1 H2 H.
  apply (MR.quot_rem_unique MMA_N); [ unfold MMA_N; lia | exact H1 | exact H2 | exact H ].
Qed.

Theorem tile_row_onto :
  forall r, exists cy i, i < MMA_M /\ tile_row cy i = r.
Proof.
  intros r. exists (r / MMA_M), (r mod MMA_M). split.
  - apply Nat.mod_upper_bound. unfold MMA_M. lia.
  - unfold tile_row. rewrite (Nat.mul_comm (r / MMA_M) MMA_M). symmetry.
    apply Nat.div_mod_eq.
Qed.

Theorem tile_col_onto :
  forall c, exists cx j, j < MMA_N /\ tile_col cx j = c.
Proof.
  intros c. exists (c / MMA_N), (c mod MMA_N). split.
  - apply Nat.mod_upper_bound. unfold MMA_N. lia.
  - unfold tile_col. rewrite (Nat.mul_comm (c / MMA_N) MMA_N). symmetry.
    apply Nat.div_mod_eq.
Qed.

(** Every element of C is owned by exactly one CTA, for a given split index. *)
Theorem c_element_has_exactly_one_owner :
  forall r c,
    (exists cy i cx j, i < MMA_M /\ j < MMA_N
                       /\ tile_row cy i = r /\ tile_col cx j = c)
    /\ (forall cy1 i1 cx1 j1 cy2 i2 cx2 j2,
          i1 < MMA_M -> i2 < MMA_M -> j1 < MMA_N -> j2 < MMA_N ->
          tile_row cy1 i1 = r -> tile_col cx1 j1 = c ->
          tile_row cy2 i2 = r -> tile_col cx2 j2 = c ->
          cy1 = cy2 /\ cx1 = cx2).
Proof.
  intros r c. split.
  - destruct (tile_row_onto r) as [cy [i [Hi Hr]]].
    destruct (tile_col_onto c) as [cx [j [Hj Hc]]].
    exists cy, i, cx, j. auto.
  - intros cy1 i1 cx1 j1 cy2 i2 cx2 j2 Hi1 Hi2 Hj1 Hj2 Hr1 Hc1 Hr2 Hc2.
    split.
    + apply (tile_row_injective cy1 i1 cy2 i2 Hi1 Hi2). rewrite Hr1, Hr2. reflexivity.
    + apply (tile_col_injective cx1 j1 cx2 j2 Hj1 Hj2). rewrite Hc1, Hc2. reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The split-K, instantiated from GridStrideSplit                  *)
(* ------------------------------------------------------------------ *)

(** [owner] IS the grid-stride rule, so the striped split needs no new
    reasoning about coverage. *)
Theorem the_split_is_the_grid_stride_rule :
  forall nz j, 0 < nz -> owner nz j = j mod nz.
Proof. reflexivity. Qed.

Theorem every_k_step_has_exactly_one_owner :
  forall nz j, 0 < nz -> owner nz j < nz.
Proof. intros nz j H. unfold owner. apply Nat.mod_upper_bound. lia. Qed.

(** **The theorem.**  Summing the CTAs' partial products gives the whole K
    range, for ANY split factor - no divisibility precondition, which is what
    lets the harness sweep [gridDim.z] without recompiling. *)
Theorem the_split_k_sums_to_the_whole_contraction :
  forall f nz S, 0 < nz -> GS.combine Z.add f nz S nz = GS.sum_upto Z.add f S.
Proof. intros f nz S H. apply GS.grid_stride_exact. exact H. Qed.

Corollary any_split_factor_gives_the_same_answer :
  forall f S nz1 nz2,
    0 < nz1 -> 0 < nz2 ->
    GS.combine Z.add f nz1 S nz1 = GS.combine Z.add f nz2 S nz2.
Proof. intros f S nz1 nz2 H1 H2. apply GS.any_worker_count_agrees; assumption. Qed.

(** **[red.global.add.s32] is the whole demonstration**, and this is the
    statement of it: the CTAs' partials may land in any order at all. *)
Theorem the_atomic_reduction_is_order_independent :
  forall f nz S order,
    0 < nz -> Permutation order (seq 0 nz) ->
    fold_right Z.add 0%Z (map (fun w => GS.class_sum Z.add f w nz S) order)
    = GS.sum_upto Z.add f S.
Proof. intros. apply GS.atomics_may_land_in_any_order; assumption. Qed.

(** The refutation that makes the previous theorem worth stating: the SAME
    striped split with a rounding accumulate disagrees with itself, both
    across split factors and across landing orders.  Reused from
    [GridStrideSplit] rather than re-invented, so the three kernels'
    refutations are comparable. *)
Theorem a_rounding_accumulate_would_break_the_split :
  GS.combine GS.KS.fadd GS.spike 3 30 3 <> GS.sum_upto GS.KS.fadd GS.spike 30.
Proof. exact GS.rounding_breaks_the_stride_split. Qed.

(** And the sharper one, which is the failure a GEMM's K-split cannot exhibit
    and this kernel's atomic can: a FIXED split factor, the same partials, and
    two orders of them landing.  That is why the property needed here is
    COMMUTATIVITY and not only associativity. *)
Theorem a_rounding_accumulate_would_break_the_landing_order :
  fold_left GS.KS.fadd
    (map (fun w => GS.class_sum GS.KS.fadd GS.tail w 3 3) (seq 0 3)) 0%Z
  <> fold_left GS.KS.fadd
    (map (fun w => GS.class_sum GS.KS.fadd GS.tail w 3 3) (rev (seq 0 3))) 0%Z.
Proof. exact GS.rounding_is_order_dependent. Qed.

(* ------------------------------------------------------------------ *)
(** ** The emitter's shape refusal is what the partition needs         *)
(* ------------------------------------------------------------------ *)

(** [emit_int8_gemm_kernel] refuses unless [M mod 16 = 0], [N mod 8 = 0] and
    [K mod 32 = 0].  Those are exactly the conditions under which the grid
    [(N/8, M/16, splits)] covers the output and the K steps, with no partial
    tile needing predication. *)
Theorem the_shape_refusal_is_the_covering_condition :
  forall m n k,
    m mod MMA_M = 0 -> n mod MMA_N = 0 -> k mod MMA_K = 0 ->
    m = (m / MMA_M) * MMA_M /\ n = (n / MMA_N) * MMA_N /\ k = k_steps k * MMA_K.
Proof.
  intros m n k Hm Hn Hk. unfold k_steps.
  repeat split.
  - rewrite (Nat.div_mod_eq m MMA_M) at 1. rewrite Hm. lia.
  - rewrite (Nat.div_mod_eq n MMA_N) at 1. rewrite Hn. lia.
  - rewrite (Nat.div_mod_eq k MMA_K) at 1. rewrite Hk. lia.
Qed.

(** And it bites: a shape one short of a tile is not covered. *)
Theorem an_uncovered_shape_is_refused :
  63 mod MMA_M <> 0 /\ 63 <> (63 / MMA_M) * MMA_M.
Proof. unfold MMA_M. cbn. split; lia. Qed.

Print Assumptions lane_decomposition_is_injective.
Print Assumptions a_warps_lanes_fill_the_tile.
Print Assumptions the_guard_is_what_confines_a_warp_to_its_own_tile.
Print Assumptions without_the_guard_a_second_warp_lands_in_the_next_tile.
Print Assumptions the_guard_excludes_nothing_a_warp_needs.
Print Assumptions the_guard_is_warp_uniform.
Print Assumptions tile_row_injective.
Print Assumptions tile_col_injective.
Print Assumptions tile_row_onto.
Print Assumptions tile_col_onto.
Print Assumptions c_element_has_exactly_one_owner.
Print Assumptions the_split_is_the_grid_stride_rule.
Print Assumptions every_k_step_has_exactly_one_owner.
Print Assumptions the_split_k_sums_to_the_whole_contraction.
Print Assumptions any_split_factor_gives_the_same_answer.
Print Assumptions the_atomic_reduction_is_order_independent.
Print Assumptions a_rounding_accumulate_would_break_the_split.
Print Assumptions a_rounding_accumulate_would_break_the_landing_order.
Print Assumptions the_shape_refusal_is_the_covering_condition.
Print Assumptions an_uncovered_shape_is_refused.
