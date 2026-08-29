//! The **f32** GEMM's band decompositions, tied to `proofs/GemmBandSplit.v`.
//!
//! `src/cpu_gemm.rs` emits two GEMMs. Nine proofs cover the exact `vpdpwssd`
//! one; `__y_sgemm_f32_avx512` — the kernel that ships for ordinary Y programs
//! — partitions the same three axes and had **no proofs at all**. That made it
//! the cheapest available version of Phase 2's decisive experiment: the
//! roadmap's stated risk is *"if obligations don't compose, the thing is a
//! one-off proof rather than a compiler"*, and here is a second, structurally
//! different kernel that already exists.
//!
//! **The two kernels split K differently.** The exact one gives the first
//! `rem` bands one extra k (`ExactGemmKsplit.blen`); the f32 one is
//! proportional, `[t*K/n, (t+1)*K/n)`. Both tile `[0, K)` — they are not the
//! same partition, and `the_two_splits_are_different_partitions` below is the
//! instance where they disagree. So the obligation composed and its **proof
//! did not transfer**: `GemmBandSplit.prop_ksplit_exact` is about thirty new
//! lines. What did transfer verbatim is the range-folding machinery
//! (`acc_range`, `sum_range_split`), because folding a contiguous range is not
//! a property of any decomposition.
//!
//! **The exactness obligation provably does not transfer**, and that is a
//! result rather than a gap: f32 addition is not associative, so per-band
//! partials do not sum to the naive sum.
//! `GemmBandSplit.rounding_breaks_the_proportional_split_too` refutes it at the
//! same `f`, `K` and thread count as the exact kernel's own refutation. The
//! repo asserts bit-identity for the exact kernel and nowhere for this one,
//! which is now a stated consequence instead of an omission.
//!
//! **What this file found.** Both decompositions clamp their last band —
//! `select (t+1 == n) ext hi` — under a comment saying the last thread takes
//! the remainder "so no row of B is dropped". The clamp never fires: the
//! arithmetic already lands on `ext`. Redundant is not dead (the packers' masks
//! in `exact_gemm_packing_model.rs` are redundant with each other and are
//! kept), but the property moves from a comment to
//! `GemmBandSplit.pedge_last` / `gedge_last`, and this file measures it.
//!
//! **There is no transcription here.** `Ix::eval` runs the SAME expression the
//! emitter renders to LLVM and the generator renders to Coq, so this file adds
//! a third *consumer*, not a third *description* — which is the defect the
//! whole layer exists to remove. The three renderings agree because every
//! operand is non-negative: `sdiv` truncates toward zero, Coq's `nat` division
//! floors, and those coincide there.
//!
//! The behavioural tie already exists and is not duplicated:
//! `cpu_gemm_threaded.rs` runs the real threaded f32 GEMM at several thread
//! counts and shapes, and its own header states why a wrong partition is a
//! wrong answer rather than a slow one.
//!
//! Run with:  cargo test --release --test f32_band_split_model

use y::cpu_gemm::{
    granule_band_edge_ix, ksplit_bands, prop_band_edge_ix, tile_count_ix, DEFAULT_TILE,
};

/// `(t * ext) / n`, evaluated through the emitter's own expression.
fn pedge(t: i64, n: i64, ext: i64) -> i64 {
    prop_band_edge_ix().eval(&|nm| match nm {
        "t" => t,
        "n" => n,
        "ext" => ext,
        other => panic!("unbound {other}"),
    })
}

/// `ceil(ext / gran)`, which is [`tile_count_ix`] — the SAME expression the
/// exact kernel's threaded wrapper uses for `(M + MR-1)/MR`.
fn gcount(ext: i64, gran: i64) -> i64 {
    tile_count_ix().eval(&|nm| match nm {
        "ext" => ext,
        "Tm1" => gran - 1,
        "T" => gran,
        other => panic!("unbound {other}"),
    })
}

/// `min(gran * ((idx * g) / count), ext)`.
fn gedge(idx: i64, count: i64, ext: i64, gran: i64) -> i64 {
    granule_band_edge_ix().eval(&|nm| match nm {
        "idx" => idx,
        "g" => gcount(ext, gran),
        "count" => count,
        "gran" => gran,
        "ext" => ext,
        other => panic!("unbound {other}"),
    })
}

/// The granularities the emitter actually passes: `MR` on the M axis, and the
/// N-axis one derived from the tile.
fn grans() -> [i64; 3] {
    [1, DEFAULT_TILE.mr as i64, DEFAULT_TILE.nr as i64]
}

/// **The finding.** Neither kernel's last-band clamp ever changes the value.
///
/// `GemmBandSplit.pedge_last` and `gedge_last` are the theorems; this measures
/// them against the emitter's own expressions over the domain the shapes and
/// thread counts can actually reach.
#[test]
fn the_last_band_clamps_are_redundant() {
    let mut k_cases = 0u64;
    for n in 1i64..=64 {
        for ext in 0i64..3000 {
            assert_eq!(
                pedge(n, n, ext),
                ext,
                "the proportional split's last edge is not the extent at \
                 n={n} ext={ext}; the emitted `select (t+1 == n) K kto0` in \
                 emit_entry would then be load-bearing, and GemmBandSplit.\
                 pedge_last is false"
            );
            k_cases += 1;
        }
    }

    let mut g_cases = 0u64;
    for gran in grans() {
        for count in 1i64..=16 {
            for ext in 0i64..800 {
                assert_eq!(
                    gedge(count, count, ext, gran),
                    ext,
                    "the granule split's last edge is not the extent at \
                     count={count} ext={ext} gran={gran}"
                );
                g_cases += 1;
            }
        }
    }

    assert!(k_cases >= 100_000 && g_cases >= 30_000, "the sweep is too small to mean anything");
}

/// The tiling obligation for the proportional split, as
/// `GemmBandSplit.prop_bands_tile` states it: start at 0, end at the extent,
/// never run backwards. Contiguity is definitional — band `t` ends where band
/// `t+1` begins — which is the structural difference from the exact kernel,
/// where it needs an induction.
#[test]
fn the_proportional_bands_tile() {
    for n in 1i64..=64 {
        for ext in 0i64..1500 {
            assert_eq!(pedge(0, n, ext), 0, "n={n} ext={ext}");
            assert_eq!(pedge(n, n, ext), ext, "n={n} ext={ext}");
            let mut prev = 0;
            let mut total = 0;
            for t in 0..n {
                let (lo, hi) = (pedge(t, n, ext), pedge(t + 1, n, ext));
                assert_eq!(lo, prev, "band {t} does not begin where {} ended (n={n} ext={ext})", t - 1);
                assert!(hi >= lo, "band {t} runs backwards (n={n} ext={ext})");
                total += hi - lo;
                prev = hi;
            }
            assert_eq!(total, ext, "the bands do not cover K (n={n} ext={ext})");
        }
    }
}

/// The granule split tiles too, and every edge is a multiple of the tile
/// granularity or the extent itself — `GemmBandSplit`'s
/// `every_edge_snaps_to_a_granule_or_the_extent`. A boundary inside a tile
/// would make one thread write a partial tile, and that was stated only in a
/// comment.
#[test]
fn the_granule_bands_tile_and_snap_to_the_granularity() {
    for gran in grans() {
        for count in 1i64..=16 {
            for ext in 0i64..800 {
                assert_eq!(gedge(0, count, ext, gran), 0);
                assert_eq!(gedge(count, count, ext, gran), ext);
                let mut prev = 0;
                for idx in 0..count {
                    let (lo, hi) = (gedge(idx, count, ext, gran), gedge(idx + 1, count, ext, gran));
                    assert_eq!(lo, prev, "gran={gran} count={count} ext={ext} idx={idx}");
                    assert!(hi >= lo, "gran={gran} count={count} ext={ext} idx={idx}");
                    assert!(
                        lo % gran == 0 || lo == ext,
                        "edge {lo} is neither a multiple of {gran} nor the extent \
                         (count={count} ext={ext})"
                    );
                    prev = hi;
                }
                assert_eq!(prev, ext, "gran={gran} count={count} ext={ext}");
            }
        }
    }
}

/// **The two kernels do not share a partition**, so `prop_ksplit_exact` is not
/// `ExactGemmKsplit.ksplit_exact` wearing a hat. `GemmBandSplit.
/// the_two_splits_are_different` states the same instance in Coq.
///
/// This is also the control on the whole file: if the two agreed, none of the
/// work above would have been necessary.
#[test]
fn the_two_splits_are_different_partitions() {
    let exact = ksplit_bands(3, 5);
    let prop: Vec<(usize, usize)> = (0..3)
        .map(|t| {
            let lo = pedge(t, 3, 5);
            (lo as usize, (pedge(t + 1, 3, 5) - lo) as usize)
        })
        .collect();
    assert_eq!(exact, vec![(0, 2), (2, 2), (4, 1)], "the exact split moved");
    assert_eq!(prop, vec![(0, 1), (1, 2), (3, 2)], "the proportional split moved");
    assert_ne!(exact, prop, "the two decompositions coincide; this file is vacuous");

    // ...and both still cover K, which is what makes them alternatives rather
    // than one of them being wrong.
    for b in [&exact, &prop] {
        assert_eq!(b.iter().map(|(_, l)| l).sum::<usize>(), 5);
    }
}
