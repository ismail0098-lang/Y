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

(** `a_i32_element`. One i32 load fetches both halves of row `i`'s k-pair `p`. *)
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
    is what the packers' zero-fill covers. *)
Definition kpairs (kc : nat) : nat := (kc + 1) / 2.

(** `mn_tiles`. The output partition: a single ragged tail, clamped. *)
Definition tw (ext T t : nat) : nat := Nat.min (ext - t * T) T.
Definition toff (T t : nat) : nat := t * T.
Definition ntiles (ext T : nat) : nat := (ext + T - 1) / T.

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
Print Assumptions the_tile_fits_the_register_file.
Print Assumptions ksplit_threads_is_never_zero.
Print Assumptions the_schedule_is_the_shipped_one.
