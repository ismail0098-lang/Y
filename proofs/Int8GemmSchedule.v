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

      - 14,571 G MAC/s at 4096^3, which is **0.09x** cuBLASLt's [_int_mm]
        (144,196) - a real kernel, not a stub.  An earlier note in this repo
        called it a stub on the strength of a GREP for [cp.async] / [ldmatrix]
        / [bar.sync]; running it says otherwise.
      - 8% of the measured int8 [mma] ISA ceiling (179,761 G MAC/s), against
        cuBLASLt's 80%.
      - It is L2-BANDWIDTH bound, at 0.1875 bytes per MAC, because it has no
        shared-memory staging: ~2,890 GB/s flat at 2048/4096/6144, then a
        3.7x collapse at 8192 when the working set leaves the 48 MB L2.

    ** BOTH of those figures were published WRONG first, by different
    ** mechanisms, and they partly cancelled.

    The first version of this header said 0.41x and a 374,027 ceiling.  The
    baseline was low because [.contiguous()] sat inside the timed loop,
    materialising a 16 MB copy per call - measured at 31,452 G MAC/s, which is
    where the published 38,090 came from.  The ceiling was HIGH because the
    probe stored only one of its eight accumulator sets, so [ptxas] deleted the
    mma chains feeding the other seven - and **the doubling control did not
    catch it, because a constant dead fraction divides out of the ratio.**

    The tell was there and was filed as a curiosity: the published ceiling made
    int8 **4.04x** the f16 rate where a spec sheet predicts 2x, and the note
    recorded that as "measured and unreconciled rather than adjusted".  The
    corrected ceiling is 1.94x f16 and 102% of [66 SM x 2.61 GHz x 1024
    MAC/SM/cycle].  A measurement that disagrees with a spec sheet by 2x is a
    bug report, not a curiosity.

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

    ** The WARP TILE, added when the kernel gained one

    A warp now issues [mt*nt] mma per K step from ONE set of A and B fragments -
    a pure traffic decision worth **4.21x** at 4096^3 with no shared memory and
    no new instruction.  Its schedule consequence is that an output row becomes
    a THREE-digit index (tile, mma within the tile, row within the mma), which
    is [MixedRadix.two_digit_unique] verbatim: [warp_row] is its **seventh**
    consumer.  [warp_row_is_a_tile_row] shows the warp tile FACTORS THROUGH the
    single-mma tiling, so every theorem above about [tile_row] still describes
    the emitted kernel and none of it had to be redone.

    The output tiles are also GRID-STRIDED in x and y now, so [owner] serves a
    third axis of this one kernel.  That is not a gratuitous generalisation: the
    tile is a compile-time function of M and N and therefore invisible at the
    call site, so a host still launching the pre-tiling [(N/8, M/16, z)] grid
    would compute a base row past the end of the matrix and
    [red.global.add.s32] would write it.
    [without_the_tile_guard_a_cta_addresses_past_the_matrix] is what makes the
    guard load-bearing rather than an optimisation - the overrun is PAST THE
    MATRIX, not a duplicate write.

    ** What this does NOT claim

    - Nothing HERE about the VALUE that lands at [C[r][c]] - this file is about
      the SCHEDULE, i.e. which lane owns which element and that the split-K
      classes tile the contraction.  [Int8GemmExact.v] proves the value, under
      a licence bounding the int32 accumulator, and reuses [warp_row] and
      [warp_col] from here so the two describe the same element.

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
(** ** The WARP TILE, and the grid stride that made it safe to change  *)
(* ------------------------------------------------------------------ *)

(** A warp now issues [mt*nt] mma per K step from ONE set of A and B fragments.
    That is a pure traffic decision - a 16x8 tile moves 768 bytes per 32-wide K
    step to retire 4096 MACs, a 64x64 tile moves 4096 bytes for 131,072, a 6x
    reduction - and it measured **4.21x** at 4096^3 on an RTX 4070 Ti SUPER.

    The SCHEDULE consequence is that an output row is now a THREE-digit index:
    the tile, the mma within the tile, and the row within the mma.  That is
    exactly [MixedRadix.two_digit_unique], so the composition costs no new
    reasoning at all - which is the claim [Decomposition.v] exists to test. *)

Definition warp_row (mt ty mi i : nat) : nat := ty * (mt * MMA_M) + mi * MMA_M + i.
Definition warp_col (nt tx ni j : nat) : nat := tx * (nt * MMA_N) + ni * MMA_N + j.

(** The warp tile does not replace the mma tiling, it factors through it: the
    16-row block a lane addresses is still [tile_row], at the flattened index
    [ty*mt + mi].  Everything already proved about [tile_row] therefore still
    describes the emitted kernel. *)
Theorem warp_row_is_a_tile_row :
  forall mt ty mi i, warp_row mt ty mi i = tile_row (ty * mt + mi) i.
Proof. intros. unfold warp_row, tile_row. ring. Qed.

Theorem warp_col_is_a_tile_col :
  forall nt tx ni j, warp_col nt tx ni j = tile_col (tx * nt + ni) j.
Proof. intros. unfold warp_col, tile_col. ring. Qed.

Theorem warp_row_injective :
  forall mt ty1 mi1 i1 ty2 mi2 i2,
    0 < mt -> mi1 < mt -> mi2 < mt -> i1 < MMA_M -> i2 < MMA_M ->
    warp_row mt ty1 mi1 i1 = warp_row mt ty2 mi2 i2 ->
    ty1 = ty2 /\ mi1 = mi2 /\ i1 = i2.
Proof.
  intros mt ty1 mi1 i1 ty2 mi2 i2 Hmt Hm1 Hm2 Hi1 Hi2 H.
  unfold warp_row in H.
  apply (MR.two_digit_unique MMA_M mt); try assumption; unfold MMA_M; lia.
Qed.

Theorem warp_col_injective :
  forall nt tx1 ni1 j1 tx2 ni2 j2,
    0 < nt -> ni1 < nt -> ni2 < nt -> j1 < MMA_N -> j2 < MMA_N ->
    warp_col nt tx1 ni1 j1 = warp_col nt tx2 ni2 j2 ->
    tx1 = tx2 /\ ni1 = ni2 /\ j1 = j2.
Proof.
  intros nt tx1 ni1 j1 tx2 ni2 j2 Hnt Hn1 Hn2 Hj1 Hj2 H.
  unfold warp_col in H.
  apply (MR.two_digit_unique MMA_N nt); try assumption; unfold MMA_N; lia.
Qed.

Theorem warp_row_onto :
  forall mt r, 0 < mt ->
    exists ty mi i, mi < mt /\ i < MMA_M /\ warp_row mt ty mi i = r.
Proof.
  intros mt r Hmt.
  exists ((r / MMA_M) / mt), ((r / MMA_M) mod mt), (r mod MMA_M).
  split; [ apply Nat.mod_upper_bound; lia | ].
  split; [ apply Nat.mod_upper_bound; unfold MMA_M; lia | ].
  unfold warp_row.
  pose proof (Nat.div_mod_eq r MMA_M) as H1.
  pose proof (Nat.div_mod_eq (r / MMA_M) mt) as H2.
  nia.
Qed.

Theorem warp_col_onto :
  forall nt c, 0 < nt ->
    exists tx ni j, ni < nt /\ j < MMA_N /\ warp_col nt tx ni j = c.
Proof.
  intros nt c Hnt.
  exists ((c / MMA_N) / nt), ((c / MMA_N) mod nt), (c mod MMA_N).
  split; [ apply Nat.mod_upper_bound; lia | ].
  split; [ apply Nat.mod_upper_bound; unfold MMA_N; lia | ].
  unfold warp_col.
  pose proof (Nat.div_mod_eq c MMA_N) as H1.
  pose proof (Nat.div_mod_eq (c / MMA_N) nt) as H2.
  nia.
Qed.

(** Every element of C is still written by exactly one (tile, mma, lane), now
    with the warp tile in between. *)
Theorem c_element_has_exactly_one_owner_under_warp_tiling :
  forall mt nt r c, 0 < mt -> 0 < nt ->
    (exists ty mi i tx ni j,
       mi < mt /\ i < MMA_M /\ ni < nt /\ j < MMA_N
       /\ warp_row mt ty mi i = r /\ warp_col nt tx ni j = c)
    /\ (forall ty1 mi1 i1 ty2 mi2 i2,
          mi1 < mt -> mi2 < mt -> i1 < MMA_M -> i2 < MMA_M ->
          warp_row mt ty1 mi1 i1 = r -> warp_row mt ty2 mi2 i2 = r ->
          ty1 = ty2 /\ mi1 = mi2 /\ i1 = i2).
Proof.
  intros mt nt r c Hmt Hnt. split.
  - destruct (warp_row_onto mt r Hmt) as [ty [mi [i [Hmi [Hi Hr]]]]].
    destruct (warp_col_onto nt c Hnt) as [tx [ni [j [Hni [Hj Hc]]]]].
    exists ty, mi, i, tx, ni, j. repeat split; assumption.
  - intros ty1 mi1 i1 ty2 mi2 i2 Hm1 Hm2 Hi1 Hi2 H1 H2.
    apply (warp_row_injective mt ty1 mi1 i1 ty2 mi2 i2); try assumption.
    rewrite H1, H2. reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The output tiles are grid-strided, in x and y                   *)
(* ------------------------------------------------------------------ *)

(** **Why the stride exists at all.**  The warp tile is a compile-time function
    of M and N, so it is invisible at the call site.  A host that kept launching
    the pre-tiling [(N/8, M/16, z)] grid would start [mt*nt] times too many
    CTAs, and each would compute a base row [ty * (mt*16)] - which for
    [ty >= m/(mt*16)] is at or past the end of the matrix.

    So changing the tile without this would be exactly the defect at the top of
    this file, one axis over: a wrong answer from a host nobody recompiled.  The
    kernel guards the tile index the same way it guards the lane index, and then
    strides, so every grid of at least (1,1,1) is correct. *)

Definition tile_owner (ng ty : nat) : nat := ty mod ng.

(** The tile loop IS the grid-stride rule, so it inherits coverage with no new
    reasoning - the third axis of this kernel to do so, after the K split and
    (in [AttentionSchedule]) the sequence reduction. *)
Theorem the_tile_loop_is_the_grid_stride_rule :
  forall ng ty, tile_owner ng ty = owner ng ty.
Proof. reflexivity. Qed.

Theorem every_output_tile_has_exactly_one_owner :
  forall ng ty, 0 < ng -> tile_owner ng ty < ng.
Proof. intros ng ty H. unfold tile_owner. apply Nat.mod_upper_bound. lia. Qed.

(** **The refutation the guard exists for.**  Any CTA index at or past the tile
    count has its base row at or past the end of the matrix - so without the
    guard those CTAs do not merely idle, they address memory the matrix does not
    own, and [red.global.add.s32] writes it. *)
Theorem without_the_tile_guard_a_cta_addresses_past_the_matrix :
  forall m mt ty,
    0 < mt -> m / (mt * MMA_M) <= ty -> m mod (mt * MMA_M) = 0 ->
    m <= ty * (mt * MMA_M).
Proof.
  intros m mt ty Hmt Hty Hmod.
  assert (Hb : 0 < mt * MMA_M) by (unfold MMA_M; nia).
  pose proof (Nat.div_mod_eq m (mt * MMA_M)) as He.
  rewrite Hmod in He. rewrite Nat.add_0_r in He.
  nia.
Qed.

(** The measured instance, at the shape [tests/int8_gemm_launch_contract.rs]
    sweeps: M=128 with a 64-row warp tile is 2 tiles, while the pre-tiling grid
    launches M/16 = 8 CTAs in y - and CTA 2 starts at row 128, i.e. exactly one
    past the last row of a 128-row matrix. *)
Theorem the_pre_tiling_grid_overruns_the_tile_count :
  128 / MMA_M = 8 /\ 128 / (4 * MMA_M) = 2 /\ 2 * (4 * MMA_M) = 128.
Proof. unfold MMA_M. repeat split; reflexivity. Qed.

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
(** ** The FUSED epilogue STORES, so it may not be split               *)
(* ------------------------------------------------------------------ *)

(** Every theorem above is about the epilogue that REDUCES.  The same emitter
    function has a second one: with [epi.is_some()] it dequantises the int32
    accumulator to f32 (per-row activation scale, per-column weight scale,
    bias) and writes it with [st.global.f32].  A store combines nothing, so
    the striped split above is not merely unproved for it - it is WRONG, and
    the argument one row up says exactly why without ever having been read
    that way: "[red.global.add.s32] ... being an INTEGER add it is
    associative" is a statement about an ADD.

    Measured on the device before this section existed, M=64 N=32 K=128,
    Sa = Sb = 1, Bias = 0 so the f32 output is the integer accumulation
    exactly, three launches per geometry:

      z = 1     0 of 2048 elements wrong
      z = 2  2048 of 2048 wrong,  C[0] = -3016 | -60724 | -3016
      z = 3  2048 of 2048 wrong,  C[0] = -23368
      z = 8  2048 of 2048 wrong,  C[0] = -15591 | -7777 | -15591
                                         (the answer is -63740)

    A different matrix, and a different one BETWEEN LAUNCHES - from the kernel
    whose own source header says "the answer does not depend on how K was
    walked", in the shape the w8a8 inference path is meant to use.

    The repair is not an atomic float add: that is precisely the
    non-reproducibility this family exists to avoid, and the bias would land
    once per z.  It is to make the launch contract not matter, which is what
    [emitted_class] / [emitted_workers] below record. *)

(** The K loop's seed and stride as the emitter chooses them.  Reducing reads
    [%ctaid.z] / [%nctaid.z]; storing emits the constants [0] and [1]. *)
Definition emitted_class (stores : bool) (cz : nat) : nat :=
  if stores then 0 else cz.
Definition emitted_workers (stores : bool) (nz : nat) : nat :=
  if stores then 1 else nz.

(** A storing CTA walks the WHOLE contraction, so what it writes is a final
    value rather than a partial. *)
Theorem a_storing_cta_computes_the_whole_contraction :
  forall f S cz nz,
    GS.class_sum Z.add f (emitted_class true cz) (emitted_workers true nz) S
    = GS.sum_upto Z.add f S.
Proof.
  intros f S cz nz. cbn [emitted_class emitted_workers].
  replace (GS.class_sum Z.add f 0 1 S) with (GS.combine Z.add f 1 S 1)
    by (cbn [GS.combine]; lia).
  apply GS.grid_stride_exact. lia.
Qed.

(** And [cz] and [nz] are not free in that answer: every z-CTA computes the
    same value and writes identical bytes, so the race between their stores is
    benign and the result is bit-identical at every grid.  This is the
    counterpart of [any_split_factor_gives_the_same_answer] for the epilogue
    that cannot combine. *)
Theorem every_storing_cta_writes_the_same_value :
  forall f S cz1 nz1 cz2 nz2,
    GS.class_sum Z.add f (emitted_class true cz1) (emitted_workers true nz1) S
    = GS.class_sum Z.add f (emitted_class true cz2) (emitted_workers true nz2) S.
Proof.
  intros. rewrite !a_storing_cta_computes_the_whole_contraction. reflexivity.
Qed.

(** The refutation that makes those two worth stating, and the one the device
    measurement above exhibits: under the stripe a CTA holds only part of the
    contraction, so a store writes part of the answer. *)
Theorem a_split_cta_holds_only_part_of_the_contraction :
  GS.class_sum Z.add (fun _ => 1%Z) 0 2 4 <> GS.sum_upto Z.add (fun _ => 1%Z) 4.
Proof. vm_compute. lia. Qed.

(** The sharper half, and the reason the measurement changed between launches
    rather than being merely wrong: the partials DISAGREE, so "last writer
    wins" is a scheduling accident with a visible value. *)
Theorem two_split_ctas_would_store_different_values :
  GS.class_sum Z.add (fun j => Z.of_nat j) 0 2 4
  <> GS.class_sum Z.add (fun j => Z.of_nat j) 1 2 4.
Proof. vm_compute. lia. Qed.

(** The reducing epilogue keeps the stripe - the constants are substituted for
    the storing one ALONE.  Without this, "walk the whole contraction
    everywhere" would satisfy every theorem in this file and delete the split
    the batch-invariance harness exists to sweep. *)
Theorem the_reducing_epilogue_still_splits :
  forall cz nz, emitted_class false cz = cz /\ emitted_workers false nz = nz.
Proof. intros. split; reflexivity. Qed.

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
Print Assumptions warp_row_is_a_tile_row.
Print Assumptions warp_col_is_a_tile_col.
Print Assumptions warp_row_injective.
Print Assumptions warp_col_injective.
Print Assumptions warp_row_onto.
Print Assumptions warp_col_onto.
Print Assumptions c_element_has_exactly_one_owner_under_warp_tiling.
Print Assumptions the_tile_loop_is_the_grid_stride_rule.
Print Assumptions every_output_tile_has_exactly_one_owner.
Print Assumptions without_the_tile_guard_a_cta_addresses_past_the_matrix.
Print Assumptions the_pre_tiling_grid_overruns_the_tile_count.
Print Assumptions a_storing_cta_computes_the_whole_contraction.
Print Assumptions every_storing_cta_writes_the_same_value.
Print Assumptions a_split_cta_holds_only_part_of_the_contraction.
Print Assumptions two_split_ctas_would_store_different_values.
Print Assumptions the_reducing_epilogue_still_splits.
