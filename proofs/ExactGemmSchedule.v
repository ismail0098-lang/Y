(** * The exact-GEMM schedule: constants and index maps, in ONE place.

    *** GENERATED FILE - DO NOT EDIT. ***

    Emitted from `src/cpu_gemm.rs`'s own constants by
    `tests/exact_gemm_schedule_proof.rs`, which links against them rather than
    parsing them. Regenerate with

    <<  Y_REWRITE_SCHEDULE_PROOF=1 cargo test --release --test exact_gemm_schedule_proof  >>

    and the same gate fails the build on any byte of divergence.

    ** Why this file exists.

    The nine Phase 1 proofs each declared their own copy of the schedule: `MR`
    and `NR` in two files, `col_of` in two, the B slot map in three, and the
    flush interval as a bare `64` inside a theorem statement.
    `ExactGemmComposition.v` asserts the copies agree - a theorem somebody
    remembered to write. Nothing forced it to exist, and nothing connected any
    of it to the Rust the kernel is actually emitted from.

    That was MEASURED before it was fixed, and the measurement corrected a
    claim in `ExactGemmComposition.v`'s own header. It says each duplicated
    definition "turns out to be pinned by a theorem in its own file". That
    holds for `RegisterTile.slot_b`, `RegisterTile.NR`, `RegisterTile.NRV`,
    `Packing.NR` and `Micro.slot` - and is FALSE for `MR`, which was never
    tried: setting it to 8 in either file leaves that file compiling. So the
    pinning was incidental, and for one constant of six it was absent.

    ** What this closes.

    Constant drift, in both directions: a `.v` edited by hand and a Rust
    constant moved without regenerating both fail the gate. It does NOT close
    the gap `docs/proof_carrying_kernels.md` names about itself - the loop NEST
    is still hand-written `IrBuilder` calls, and the tie between the model and
    the emitted LLVM is still two models meeting. That is Phase 2.

    ** What is deliberately NOT collapsed.

    [slot_b] and [slot_b_interleave] are two DIFFERENT expressions for one map:
    the emitted `vpdpwssd` vector-group form, and the plain interleave that
    `panel_slot_decode` in `cpu_gemm.rs` inverts with a bare `/2`.
    [slot_b_is_the_plain_interleave] proves they agree. Collapsing them to one
    definition would make that theorem - and
    `ExactGemmComposition.the_agreement_is_not_vacuous` - true by `reflexivity`
    and worth nothing.

    Build:  coqc proofs/ExactGemmSchedule.v      (Rocq 9.1)
*)

From Stdlib Require Import Arith Lia.

(* ------------------------------------------------------------------ *)
(** ** The tile                                                        *)
(* ------------------------------------------------------------------ *)

(** `VNNI_MR`: rows of C the micro-kernel holds in registers. *)
Definition MR : nat := 6.

(** `VNNI_NRV`: `<16 x i32>` accumulator groups per row. *)
Definition NRV : nat := 4.

(** int32 lanes in one accumulator group. Derived in the generator as
    `VNNI_NR / VNNI_NRV` rather than written down, so it cannot become a
    fourth copy of 16. *)
Definition LANES : nat := 16.

(** `VNNI_NR`: columns of C the micro-kernel covers. *)
Definition NR : nat := 64.

(** int16 elements in one `<32 x i16>` B vector group - two per lane. This is
    the `v * 32` stride of the emitted B load. *)
Definition VEC_ELEMS : nat := 32.

(** AVX-512's architectural register count. An ISA FACT, not a schedule
    constant - there is no `cpu_gemm.rs` constant for it, and no proof over
    [nat] establishes it. It sits at the same boundary as `vpdpwssd`'s
    semantics: pinned empirically, here by
    `tests/cpu_gemm_vnni_micro.rs::the_hot_loop_does_not_spill_the_accumulators`
    reading real compiled output. *)
Definition ZMM_REGISTERS : nat := 32.

(* ------------------------------------------------------------------ *)
(** ** The K axis                                                      *)
(* ------------------------------------------------------------------ *)

(** `VnniExact::DEFAULT_FLUSH_K_PAIRS`: k-pairs accumulated in int32 before
    widening into the int64 running sum. This was a bare `64` inside
    `ExactGemmMicro.the_4096_case_exceeds_by_exactly_one`, which is the one
    place a schedule constant had been written into a theorem STATEMENT. *)
Definition FLUSH_K_PAIRS : nat := 64.

(** `KSPLIT_MIN_BAND`: shortest K band worth giving a thread. *)
Definition KSPLIT_MIN_BAND : nat := 128.

(** `KSPLIT_MAX_THREADS`: the emitted wrapper's clamp. *)
Definition KSPLIT_MAX_THREADS : nat := 64.

(* ------------------------------------------------------------------ *)
(** ** Panel destination indices                                       *)
(* ------------------------------------------------------------------ *)

(** `pack_a_slot`. Where `pack_a` puts row `i`, half `h`, inside its `2*MR`
    slot group. *)
Definition slot_a (i h : nat) : nat := 2 * i + h.

(** `pack_b_slot`. The emitted `vpdpwssd` lane layout: accumulator group
    `v = j / LANES` is one `<32 x i16>` vector, and lane `l = j mod LANES`
    inside it consumes int16 elements `2l` and `2l+1`. *)
Definition slot_b (j h : nat) : nat := (j / 16) * 32 + (j mod 16) * 2 + h.

(** The plain interleave - the form `panel_slot_decode` inverts with a bare
    `/2`, and the form the register-tile and micro-kernel models are stated
    over. [slot_b_is_the_plain_interleave] below is what says these are one
    map; the Rust doc comment on `panel_slot_decode` asserts it in prose. *)
Definition slot_b_interleave (j h : nat) : nat := 2 * j + h.

(** `column_of_lane`. Which column of the tile accumulator lane `l` of vector
    `v` holds. *)
Definition col_of (v l : nat) : nat := 16 * v + l.

(** `vec_of_slot` / `lane_of_slot`: the inverse legs. *)
Definition vec_of_slot (s : nat) : nat := s / 32.
Definition lane_of_slot (s : nat) : nat := (s mod 32) / 2.

(** `a_i32_element`. One i32 load fetches both halves of row `i`'s k-pair `p`.
    [ExactGemmRegisterTile.the_i32_load_is_the_packed_pair] is stated over this
    number; [the_emitted_a_index_is_the_pair_element] below is what says the
    emitter computes it. *)
Definition a_i32_element (p i : nat) : nat := p * 6 + i.

(** `panel_slot_decode`, as its three legs. `width` is [MR] for an A panel and
    [NR] for a B one. *)
Definition panel_group (s width : nat) : nat := s / (2 * width).
Definition panel_idx (s width : nat) : nat := (s mod (2 * width)) / 2.
Definition panel_half (s width : nat) : nat := (s mod (2 * width)) mod 2.

(* ------------------------------------------------------------------ *)
(** ** Loop decompositions                                             *)
(* ------------------------------------------------------------------ *)

(** k-pairs in a K panel of `kc`, rounded up: the phantom half of an odd `kc`
    is what the packers' zero-fill covers.

    **Rendered from `cpu_gemm::kpairs_ix`**, the expression the emitter spells
    at FIVE sites - both packers, the driver, and twice in the threaded
    wrapper. Every packing and flush theorem is stated in terms of this number
    and nothing said the compiler computed the same one. *)
Definition kpairs (kc : nat) : nat := ((kc + 1) / 2).

(* ------------------------------------------------------------------ *)
(** ** The packing buffers the driver allocates                        *)
(* ------------------------------------------------------------------ *)

(** The stride between two packed A panels in i16 ELEMENTS, and the byte counts
    the driver hands to `malloc`.

    **Rendered from `cpu_gemm::panel_a_stride_ix` / `panel_a_bytes_ix` /
    `panel_b_bytes_ix`.** Until these were extracted the panel geometry existed
    THREE times: as `2 * MR` and `2 * NR` spelled into the emitter's `malloc`
    arithmetic, as `kp * (2 * width)` in [ExactGemmPacking]'s notion of how far
    a panel runs, and as the driver's own separate `panel_stride` multiply.
    Nothing said the three agreed.

    A divergence in the SAFE direction - a buffer larger than the write set -
    changes no answer, so no correctness test in this repository can see one.
    That is not hypothetical: an over-allocation of exactly this shape was
    caught by the schedule gate and by nothing else.

    [ExactGemmAllocation.v] turns these into a bound on the packers' writes.
    Note the trailing `* 2` is `sizeof(i16)` while the inner one is the k-pair
    interleave - two different twos, both named rather than spelled. *)
Definition panel_a_stride (kpairs : nat) : nat := (kpairs * 12).
Definition panel_a_bytes (mtiles kpairs : nat) : nat := (((mtiles * kpairs) * 12) * 2).
Definition panel_b_bytes (kpairs : nat) : nat := ((kpairs * 128) * 2).

(** The same sizes in ELEMENTS, which is the unit every slot map is in. *)
Definition panel_a_elems (mtiles kpairs : nat) : nat :=
  (panel_a_bytes mtiles kpairs) / 2.
Definition panel_b_elems (kpairs : nat) : nat := (panel_b_bytes kpairs) / 2.

(** The scratch tile, in bytes as the emitted `malloc` spells it. *)
Definition SCRATCH_TILE_BYTES : nat := 3072.

(* ------------------------------------------------------------------ *)
(** ** The threading layer's three allocations                         *)
(* ------------------------------------------------------------------ *)

(** Fields in one worker's job record: `A B C M N K lda ldb ldc Ap Bp Ct`.

    **Generated from `cpu_gemm::JOB_SLOTS`**, which the emitter's `malloc`
    arithmetic now derives its `96` from. That number is the record's STRIDE as
    well as its size, so the two roles have to move together: a thirteenth
    field with the stride left behind puts the writer's last store in the next
    thread's record, and past the end of the array entirely for the last
    thread. [jobs_bytes_is_a_record_per_thread] is what states they agree. *)
Definition JOB_SLOTS : nat := 12.

(** **Rendered from `cpu_gemm::jobs_bytes_ix` / `tids_bytes_ix` /
    `private_c_bytes_ix`.**

    `private_c_bytes` is `rows * cols`, NOT `rows * ldc`, and that distinction
    is not cosmetic: the worker is handed `cols` as its output stride precisely
    so its buffer can be compact whatever padding the caller's C carries.
    Sizing it from `ldc` while writing at stride `cols` over-allocates, which no
    answer can see; sizing it from `cols` while WRITING at `ldc` put
    `(rows-1)*(ldc-cols)` elements past the end, which was observed as
    `double free or corruption`. *)
Definition jobs_bytes (nthr : nat) : nat := (nthr * 96).
Definition tids_bytes (nthr : nat) : nat := (nthr * 8).
Definition private_c_bytes (rows cols : nat) : nat := ((rows * cols) * 8).

(** The same sizes in the unit each index map is stated in: job slots, thread
    slots, and i64 elements. *)
Definition jobs_slots_total (nthr : nat) : nat := (jobs_bytes nthr) / 8.
Definition tids_slots (nthr : nat) : nat := (tids_bytes nthr) / 8.
Definition private_c_elems (rows cols : nat) : nat :=
  (private_c_bytes rows cols) / 8.

(** The array's stride IS one record. Self-fulfilling under generation in the
    sense that both sides come from `JOB_SLOTS` - and that is the point: before
    the extraction the `malloc` carried a literal `96` that no definition of
    `JOB_SLOTS` could have contradicted. *)
Theorem jobs_bytes_is_a_record_per_thread : forall nthr,
  jobs_bytes nthr = (nthr * (JOB_SLOTS * 8))%nat.
Proof. intro nthr. unfold jobs_bytes, JOB_SLOTS. reflexivity. Qed.

(** `mn_tiles`. The output partition: a single ragged tail, clamped. *)
Definition tw (ext T t : nat) : nat := Nat.min (ext - t * T) T.
Definition toff (T t : nat) : nat := t * T.
Definition ntiles (ext T : nat) : nat := (ext + T - 1) / T.

(** The driver's two panel loops, **rendered from `cpu_gemm::CountedLoop`** -
    the same description `IrBuilder::loop_begin_counted` emits.

    This is the first slice of the loop NEST to be extracted rather than
    modelled. [toff] and [ntiles] above describe what the row-panel loop does;
    nothing said the emitted loop does it, and a trip count one too large is
    invisible in the answer (the extra tile clamps to zero width and writes
    nothing). `ExactGemmTiling` closes that with
    [the_emitted_row_loop_enumerates_the_tiles].

    Rendered, not restated: `_visit` comes from the emitter's `start`/`step`,
    `_trips` from all three. The emitter never COMPUTES the trip count - the
    loop tests `iv < end` - so that half is a fact about the loop rather than
    an expression it emits. *)
Definition row_panel_visit (M k : nat) : nat := (0 + (k * 6)).
Definition row_panel_trips (M : nat) : nat := (((M - 0) + (6 - 1)) / 6).

Definition col_panel_visit (N k : nat) : nat := (0 + (k * 64)).
Definition col_panel_trips (N : nat) : nat := (((N - 0) + (64 - 1)) / 64).

Definition fold_row_visit (ext iv T k : nat) : nat := (0 + (k * 1)).
Definition fold_row_trips (ext iv T : nat) : nat := (((Nat.min (ext - iv) T - 0) + (1 - 1)) / 1).

Definition fold_col_visit (ext iv T k : nat) : nat := (0 + (k * 1)).
Definition fold_col_trips (ext iv T : nat) : nat := (((Nat.min (ext - iv) T - 0) + (1 - 1)) / 1).

Definition zero_tile_visit (k : nat) : nat := (0 + (k * 1)).
Definition zero_tile_trips : nat := (((384 - 0) + (1 - 1)) / 1).

(** `ksplit_bands`. The K-split reduction: `base = K/nthr`, `rem = K mod nthr`,
    and the first `rem` bands take one extra k, so the cuts are UNEVEN. A
    different decomposition from [tw] deliberately - do not unify them. *)
Definition blen (K nthr t : nat) : nat :=
  K / nthr + (if Nat.ltb t (K mod nthr) then 1 else 0).

Fixpoint boff (K nthr t : nat) : nat :=
  match t with
  | O => O
  | S t' => boff K nthr t' + blen K nthr t'
  end.

(** `ksplit_threads`. The emitted `__y_gemm_exact_threads`. *)
Definition ksplit_threads (requested K : nat) : nat :=
  let ceil := Nat.min (Nat.max requested 1) 64 in
  let by_k := K / 128 in
  Nat.max (Nat.min by_k ceil) 1.

(** `flush_chunks`. Chunk `t` starts at `t*Fl` and is CLAMPED - the final
    partial chunk is carried by the same clamp rather than by an epilogue.
    `Fl` stays a parameter: `ExactGemmMicro.v` quantifies over it, and
    [FLUSH_K_PAIRS] above is the value the compiler ships. *)
Definition coff (Fl t : nat) : nat := t * Fl.
Definition cw (Fl n t : nat) : nat := Nat.min (coff Fl (S t)) n - coff Fl t.
Definition nchunks (Fl n : nat) : nat := (n + Fl - 1) / Fl.

(* ------------------------------------------------------------------ *)
(** ** The emitter's own index arithmetic                              *)
(* ------------------------------------------------------------------ *)

(** **These two are rendered from the SAME expressions the emitter renders to
    LLVM** - `cpu_gemm::tile_width_ix` and `panel_index_ix`, via `Ix::coq`
    where the driver uses `Ix::emit`. Everything above is generated from a
    constant; these are generated from the emitted CODE's arithmetic.

    That is the difference between a model that agrees with the emitter and a
    model that IS the emitter, for this slice of the schedule. The slice is
    small and deliberately so: it is the tile-width clamp and the panel index,
    which is the arithmetic the driver's loop nest does on its induction
    variables, and which §1 of `docs/proof_carrying_kernels.md` names as where
    the bugs live ("twelve address computations ... correct only because
    `lda == K` made stride and extent the same number"). Which loops exist, in
    what order, and what they call is still hand-written. *)

(** The live width of the tile starting at induction variable `iv`. *)
Definition tile_width (ext iv T : nat) : nat := Nat.min (ext - iv) T.

(** Which packed panel the tile at `iv` reads. *)
Definition panel_index (iv T : nat) : nat := (iv / T).

(** **The join.** [tw] is the tiling model, stated over the tile INDEX; the
    emitted loop has the induction variable instead. This says they are the
    same number, which is what lets `ExactGemmTiling.v`'s partition theorems
    describe the emitted driver.

    It was implicit before: the emitter clamped with three `IrBuilder` calls
    and a behavioural test (`exact_gemm_tile_enumeration.rs`) sampled the
    result. *)
Theorem the_emitted_width_is_the_tiling_model_at_the_loop_variable :
  forall ext T t, tw ext T t = tile_width ext (toff T t) T.
Proof. reflexivity. Qed.

(** ...and the emitted `sdiv iv, T` really does recover the tile index, so the
    panel a tile reads is its own. Getting this wrong is a correctly-computed
    tile read from the wrong panel - the shape of bug this repo catalogues as
    invisible to a relative-L2 check. *)
Theorem the_emitted_panel_index_is_the_tile_index :
  forall T t, 0 < T -> panel_index (toff T t) T = t.
Proof.
  intros T t HT. unfold panel_index, toff.
  rewrite Nat.div_mul by lia. reflexivity.
Qed.

(** The flush chunk's END, as the micro-kernel's outer loop computes it.

    The emitter computes an END and [cw] computes a WIDTH; they are the same
    clamp seen from two sides, and the theorem below is that identity. *)
Definition chunk_end (iv T ext : nat) : nat := Nat.min (iv + T) ext.

(** The K-split's even share and remainder. Loop-invariant, so the emitted
    wrapper hoists both out of the spawn loop - which is why they are separate
    expressions rather than subterms of [band_len]. An expression split across
    basic blocks is not one contiguous instruction sequence. *)
Definition band_base (K nthr : nat) : nat := (K / nthr).
Definition band_rem (K nthr : nat) : nat := (K mod nthr).

(** Band `t`'s length, over the already-hoisted `base` and `rem`. *)
Definition band_len (base rem t : nat) : nat := (base + (if Nat.ltb t rem then 1 else 0)).

(** How many tiles an axis is cut into, as the threaded wrapper computes it.

    `T - 1` is a separate parameter because the emitter FOLDS it at compile
    time into a literal (`add i64 %M, 5`); modelling it as `T - 1` inside the
    expression would emit an instruction the compiler does not. *)
Definition tile_count (ext Tm1 T : nat) : nat := ((ext + Tm1) / T).

(** The packed-A row base and the per-row element index, as the micro-kernel
    emits them. `MR` is a tile constant and `i` is a Rust-side constant per
    unrolled row, so both reach the emitted instructions as literal operands -
    which makes them bound NAMES here, not a reason the expression cannot be
    extracted.

    They are two expressions rather than one because the base is loop-invariant
    across the unroll and the emitter hoists it, the same split as
    [band_base]/[band_len]. *)
Definition a_base (p MR : nat) : nat := (p * MR).
Definition a_elem (base i : nat) : nat := (base + i).

(** ** The f32 kernel's bands - a SECOND consumer, and a different split

    `src/cpu_gemm.rs` emits two GEMMs. Everything above is the exact `vpdpwssd`
    one; these two expressions are the **f32 AVX-512** kernel's partitions, and
    they are here because they are the same KIND of object, not because they
    are the same object. The f32 K-split is proportional
    (`[t*K/n, (t+1)*K/n)`), NOT the exact kernel's even-with-remainder split
    (`[boff, boff + blen)`) - `proofs/GemmBandSplit.v` proves each tiles, and
    exhibits an instance where they disagree.

    `prop_band_edge`: band `t` of the proportional split runs
    `[prop_band_edge t n ext, prop_band_edge (S t) n ext)`, so one expression
    is both ends. *)
Definition prop_band_edge (t n ext : nat) : nat := ((t * ext) / n).

(** `granule_band_edge`: the f32 M and N bands, which partition the GRANULE
    COUNT `g` rather than the extent - snapping a band's position to the tile
    granularity instead dumps the accumulated slack onto one band, and in 2-D
    the two axes' errors multiply. `g` is [tile_count], so this shares an
    expression with the exact kernel above. *)
Definition granule_band_edge (idx g count gran ext : nat) : nat := Nat.min (((idx * g) / count) * gran) ext.

(** **The tiling-count join**, and it needs `0 < T` because `nat` subtraction
    truncates: at `T = 0` the model's `ext + T - 1` is `ext - 1` while the
    emitter's `ext + (T-1)` is `ext`. The emitter cannot reach `T = 0` - the
    tile is a compile-time constant - but the hypothesis is what makes the
    folding legitimate rather than incidental. *)
Theorem the_emitted_tile_count_is_the_tiling_model : forall ext T,
  0 < T -> ntiles ext T = tile_count ext (T - 1) T.
Proof.
  intros ext T HT. unfold ntiles, tile_count.
  replace (ext + T - 1)%nat with (ext + (T - 1))%nat by lia.
  reflexivity.
Qed.

(** **The flush join.** [cw] is the model's chunk width; the emitted loop
    clamps an end instead. *)
Theorem the_emitted_chunk_end_is_the_flush_model : forall Fl n t,
  cw Fl n t = chunk_end (coff Fl t) Fl n - coff Fl t.
Proof.
  intros Fl n t. unfold cw, chunk_end, coff.
  replace (S t * Fl)%nat with (t * Fl + Fl)%nat by lia.
  reflexivity.
Qed.

(** **The K-split join.** The emitted spawn loop's `%klen` is
    `ExactGemmKsplit`'s [blen], recomposed from the two hoisted terms. Every
    theorem in that file is about [blen]; this is what says the emitted wrapper
    computes it. *)
Theorem the_emitted_band_length_is_the_ksplit_model : forall K nthr t,
  blen K nthr t = band_len (band_base K nthr) (band_rem K nthr) t.
Proof. reflexivity. Qed.

(** **The A-index join.** [ExactGemmRegisterTile.the_i32_load_is_the_packed_pair]
    proves the i32 load at element [a_i32_element p i] aliases packed slots
    `2i` and `2i+1` of k-pair group `p`. That theorem says nothing about the
    compiler; this says the compiler computes that element. *)
Theorem the_emitted_a_index_is_the_pair_element : forall p i,
  a_i32_element p i = a_elem (a_base p MR) i.
Proof. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** ** What this file proves about itself                              *)
(* ------------------------------------------------------------------ *)

(** `tests/proofs_are_checked.rs::every_proof_has_a_content_control` refuses a
    `.v` that names no load-bearing theorem, and it is right to: "compiles"
    and "no axioms" are properties an EMPTY file has. A generated file is
    exactly where that matters most, because a generator emitting nonsense
    emits it confidently.

    So this file is not definitions-only and takes no exemption. The three
    theorems below each catch a distinct way the generator could be wrong, and
    two of them are STRUCTURAL - they constrain the shape of the emitted
    expressions, not their values, so they are not made true merely by being
    generated alongside what they describe. *)

(** The tile geometry is internally consistent. Catches a generator that
    emitted [NR] and [NRV] from constants that had stopped agreeing, or any
    degenerate zero. *)
Theorem the_tile_geometry_is_consistent :
  NR = NRV * LANES
  /\ VEC_ELEMS = 2 * LANES
  /\ 0 < MR /\ 0 < NRV /\ 0 < LANES /\ 0 < NR
  /\ 0 < FLUSH_K_PAIRS /\ 0 < KSPLIT_MIN_BAND /\ 0 < KSPLIT_MAX_THREADS.
Proof. unfold NR, NRV, LANES, VEC_ELEMS, MR, FLUSH_K_PAIRS,
         KSPLIT_MIN_BAND, KSPLIT_MAX_THREADS. repeat split; lia. Qed.

(** The emitted vector-group form of the B map IS the plain interleave.

    This is the load-bearing one, and it is not a restatement of a generated
    value: the two sides come from different places in `cpu_gemm.rs` -
    [slot_b] from `pack_b_slot`, [slot_b_interleave] from the bare `/2` that
    `panel_slot_decode` uses to invert it. The Rust asserts their agreement in
    a doc comment; this proves it.

    `16*(j/16) + (j mod 16) = j`, so the vector-group decomposition folds away
    entirely. `ExactGemmPacking.v` records the consequence: there are not two
    layouts to tell apart, so no proof can pin the lane assignment and
    `tests/cpu_gemm_vnni_micro.rs` on the real instruction is what does. *)
Theorem slot_b_is_the_plain_interleave : forall j h,
  slot_b j h = slot_b_interleave j h.
Proof.
  intros j h. unfold slot_b, slot_b_interleave.
  pose proof (Nat.div_mod_eq j 16) as H. lia.
Qed.

(** **The tile fits the register file**, and this is the constraint that was
    written in a test comment and stated nowhere.

    The micro-kernel holds `MR * NRV` int32 accumulators live across the hot
    loop, plus [NRV] `<32 x i16>` B vectors and one broadcast A vector. On
    AVX-512 that has to fit in 32 zmm. `cpu_gemm.rs` never says so and no
    theorem did either - the arithmetic appeared only as prose beside a spill
    bound ("24 accumulators + 4 B vectors + 1 A broadcast is 29 of 32 zmm").

    **The predicate is MEASURED, not guessed**, by sweeping `VNNI_MR` and
    reading the hot loop's real spill traffic:

<<
      MR   budget   hot-loop spills + reloads
       5   25/32    within bound
       6   29/32    within bound  (10 + 10, the shipped kernel)
       7   33/32    16 + 16
       8   37/32    17 + 17
>>

    The cliff falls exactly where this inequality flips, so the form of the
    bound is the measurement's and not an invention.

    It is STRUCTURAL and it bites: a generator emitting `MR = 8` emits a file
    in which this theorem is FALSE, so `coqc` rejects the schedule outright
    rather than proving nine theorems about a tile that cannot be allocated.
    Note what it does not claim - a spilling kernel is SLOW, not wrong.
    `tests/exact_gemm_thread_invariance.rs` still passes bit-identically at
    `MR = 8`, which is how the diagnosis separated the two. *)
Theorem the_tile_fits_the_register_file :
  MR * NRV + NRV + 1 <= ZMM_REGISTERS.
Proof. unfold MR, NRV, ZMM_REGISTERS. lia. Qed.

(** The thread count is never zero.

    A genuine cross-file join rather than a self-check: every theorem in
    `ExactGemmKsplit.v` is stated under `(0 < nthr)%nat`, and `ksplit_bands`
    in `cpu_gemm.rs` asserts the same precondition at runtime. Nothing proved
    the emitted thread count satisfies it - the floor was argued in a comment
    ("`ksplit_threads` floors at 1"). Here it is discharged, for every request
    and every K, including `K` below [KSPLIT_MIN_BAND] where `by_k` is 0. *)
Theorem ksplit_threads_is_never_zero : forall requested K,
  0 < ksplit_threads requested K.
Proof. intros. unfold ksplit_threads. lia. Qed.

(** The non-vacuity control, and it is honest about being weaker than the two
    above. Under generation this is self-fulfilling - a generator emitting
    `MR := 8` emits this theorem with an 8 in it. Its job is the OTHER
    direction: it makes the shipped values load-bearing inside Coq, so a
    hand-edit of this committed file fails `coqc` as well as failing the
    byte-identity gate. That matters most for [MR], which the measurement
    recorded above found was pinned by nothing in its own file. *)
Theorem the_schedule_is_the_shipped_one :
  MR = 6 /\ NRV = 4 /\ LANES = 16 /\ NR = 64
  /\ VEC_ELEMS = 32 /\ FLUSH_K_PAIRS = 64
  /\ KSPLIT_MIN_BAND = 128 /\ KSPLIT_MAX_THREADS = 64.
Proof. repeat split; reflexivity. Qed.

Print Assumptions the_tile_geometry_is_consistent.
Print Assumptions slot_b_is_the_plain_interleave.
Print Assumptions the_emitted_width_is_the_tiling_model_at_the_loop_variable.
Print Assumptions the_emitted_panel_index_is_the_tile_index.
Print Assumptions the_emitted_tile_count_is_the_tiling_model.
Print Assumptions the_emitted_chunk_end_is_the_flush_model.
Print Assumptions the_emitted_band_length_is_the_ksplit_model.
Print Assumptions the_emitted_a_index_is_the_pair_element.
Print Assumptions the_tile_fits_the_register_file.
Print Assumptions ksplit_threads_is_never_zero.
Print Assumptions the_schedule_is_the_shipped_one.
