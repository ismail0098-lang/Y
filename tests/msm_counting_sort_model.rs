//! The tie between `proofs/CountingSort.v` and the running MSM binner.
//!
//! `bin_by_digit` is a parallel counting sort: per-chunk histograms, an
//! exclusive prefix over buckets, and an unsynchronised scatter through a
//! private cursor per (writer, bucket). `CountingSort.v` proves its
//! destination map is a bijection — every slot of `Idx` written by exactly one
//! thread, exactly once, in bounds. That is a fact about `nat`; this file is
//! what says the shipped code walks it.
//!
//! **What was covered before, and what was not.** `zk_gpu_msm.rs` has
//! `binning_does_not_depend_on_the_thread_count`, which compares the binner
//! against ITSELF at one writer. That is a strong check on the grouping
//! arithmetic and it is blind by construction to anything wrong at every
//! thread count — a cursor prefix that is systematically off, a group whose
//! slice is taken from the wrong point range. The MSM tests above it check the
//! final curve SUM, and this repo already records that an entry in the wrong
//! bucket still yields a valid curve point. Nothing compared `Idx` against an
//! independent placement.
//!
//! **The oracle is a plain sequential stable counting sort** — the
//! specification the parallel one exists to be a fast version of, in the same
//! relationship as the naive triple loop to the tiled GEMM. It deliberately
//! shares `window_digit` and the `base[w] + d` bucket numbering with the
//! implementation: those decide WHICH bucket an entry belongs to, they are not
//! what `CountingSort.v` is about, and they are already checked against
//! arkworks by `gpu_msm_matches_arkworks` and `the_msm_oracle_is_not_vacuous`.
//! What is under test here is placement.
//!
//! No GPU is needed — `bin_by_digit` is pure host code — so unlike its
//! neighbours in `zk_gpu_msm.rs` this file never skips.

#[path = "common/msm.rs"]
mod msm;

use ark_bn254::Fr;
use ark_ff::{PrimeField, UniformRand};
use ark_std::rand::SeedableRng;

use msm::{bin_by_digit, last_scatter_shape, set_scatter_threads, window_digit, Geom};

const N: usize = 40_000;
const SEED: u64 = 0xC0FFEE;

fn scalars() -> Vec<Fr> {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(SEED);
    (0..N).map(|_| Fr::rand(&mut rng)).collect()
}

/// Every (point, window) entry of the input, in the order a single-threaded
/// binner would visit them: point-outer, window-inner, digit 0 dropped.
fn entries(scalars: &[Fr], g: &Geom) -> Vec<(usize, u32)> {
    let mut v = Vec::with_capacity(scalars.len() * g.nw);
    for (i, sc) in scalars.iter().enumerate() {
        let l = sc.into_bigint().0;
        for w in 0..g.nw {
            let d = window_digit(&l, g.shifts[w], g.widths[w]);
            if d != 0 {
                v.push((g.base[w] + d as usize, i as u32));
            }
        }
    }
    v
}

/// **The specification.** One writer, buckets filled in point order, offsets
/// the exclusive prefix of the bucket totals. Obviously correct and far too
/// slow to ship, which is the point.
fn reference_bins(scalars: &[Fr], g: &Geom) -> (Vec<u32>, Vec<u32>) {
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); g.nb];
    for (b, i) in entries(scalars, g) {
        buckets[b].push(i);
    }
    let mut off = Vec::with_capacity(g.nb + 1);
    let mut idx = Vec::new();
    let mut run = 0u32;
    for b in &buckets {
        off.push(run);
        run += b.len() as u32;
        idx.extend_from_slice(b);
    }
    off.push(run);
    (idx, off)
}

/// **The oracle's own liveness canary, and it is a SOURCE check because no
/// behavioural one can work.**
///
/// Replacing `reference_bins`' body with a call to `bin_by_digit` makes every
/// comparison in this file compare the binner against itself, and every
/// control below still passes — they are computed from the (now identical)
/// output. That mutation survived the first sweep of this file.
///
/// It is not fixable by testing harder. A specification and a correct
/// implementation agree exactly, so a behavioural check cannot tell "the
/// oracle is right" from "the oracle IS the implementation" — the same hole
/// every differential in this repo has, where replacing one arm with the other
/// is undetectable from the outside. What distinguishes them is structural, so
/// that is what is checked: the reference must not reach into the parallel
/// binner.
///
/// `window_digit` and `Geom` are deliberately allowed. They decide which
/// bucket an entry belongs to, not where it is placed, and they are checked
/// against arkworks elsewhere.
#[test]
fn the_reference_does_not_call_the_implementation_it_is_the_oracle_for() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/msm_counting_sort_model.rs"),
    )
    .expect("read this test's own source");

    // Anchored on a newline so this matches the DEFINITION at column 0 and not
    // the string literal on this very line. Without the anchor the search
    // finds its own source, extracts this test's body instead, and then
    // reports the failure below for the wrong reason - which is how this was
    // found.
    let start = src
        .find("\nfn reference_bins(")
        .expect("reference_bins is gone or renamed; point this gate at whatever replaced it")
        + 1;
    let body = &src[start..];
    let end = body
        .find("\n}\n")
        .expect("reference_bins has no closing brace at column 0");
    let body = &body[..end];

    // The gate's own non-vacuity: an extraction that landed somewhere else, or
    // a reference that stopped doing the placement, would satisfy the loop
    // below trivially.
    assert!(body.len() > 200, "reference_bins' body extracted as {} bytes", body.len());
    assert!(
        body.contains("buckets[b].push(i)"),
        "the extracted body does not place entries into buckets, so this gate is \
         looking at the wrong function"
    );

    for forbidden in ["bin_by_digit", "last_scatter_shape", "set_scatter_threads"] {
        assert!(
            !body.contains(forbidden),
            "reference_bins calls `{forbidden}`, so it is no longer an independent \
             specification — every comparison in this file would then be the parallel \
             binner against itself"
        );
    }
}

/// The parallel counting sort places exactly what a sequential one places, at
/// every writer count, at three window geometries.
///
/// An unwritten slot of `Idx` reads back as the point index 0, because `idx`
/// is allocated with `vec![0u32; total]` — indistinguishable from a legitimate
/// entry on its own. It is caught here because the reference says what belongs
/// in that slot, which no self-comparison can.
#[test]
fn the_parallel_counting_sort_places_what_a_sequential_one_places() {
    let sc = scalars();
    for nw in [20usize, 25, 31] {
        let g = Geom::new(nw);
        let (want_idx, want_off) = reference_bins(&sc, &g);

        // The controls. Every assertion below is satisfied vacuously by a
        // geometry whose buckets hold at most one entry, and the empty-bucket
        // case is the one `scatter`'s first post-condition check got wrong.
        let deepest = (0..g.nb).map(|b| want_off[b + 1] - want_off[b]).max().unwrap();
        let empties = (0..g.nb).filter(|&b| want_off[b + 1] == want_off[b]).count();
        assert!(deepest > 4, "nw = {nw}: deepest bucket holds {deepest}, too few to order");
        assert!(empties > 0, "nw = {nw}: no empty bucket, so that case is untested");

        for thr in [1usize, 2, 3, 6, 12, 32] {
            set_scatter_threads(Some(thr));
            let (idx, off) = bin_by_digit(&sc, &g);
            set_scatter_threads(None);

            assert_eq!(off, want_off, "nw = {nw}, {thr} writers: offsets disagree");
            assert_eq!(
                idx, want_idx,
                "nw = {nw}, {thr} writers: Idx disagrees with a sequential counting sort"
            );
        }
    }
}

/// **The model tie.** `CountingSort.dest` evaluated on real data, against the
/// real `Idx`.
///
/// The proof's destination map is
///
/// ```text
/// dest b gi r = edge tot b + edge (gw b) gi + r
/// ```
///
/// — `off[b]` from the bucket totals, then the group's own prefix inside the
/// bucket, then the rank. This recomputes both edges from the histogram and
/// asserts every entry is at the slot the map names. It is stronger than the
/// oracle comparison above, which checks the resulting SEQUENCE: this checks
/// the arithmetic that produced it, and would separate two cursor tables that
/// happen to yield the same order.
///
/// The proof states the group widths over chunk ranges; the observable form is
/// point ranges, and they are the same partition — a scatter group owns
/// `group` consecutive histogram chunks, i.e. `chunk * group` consecutive
/// points. `span` is recorded by `scatter` rather than recomputed here.
#[test]
fn the_destination_map_is_the_one_the_proof_describes() {
    let sc = scalars();
    for nw in [20usize, 25] {
        let g = Geom::new(nw);
        for thr in [3usize, 6, 12] {
            set_scatter_threads(Some(thr));
            let (idx, off) = bin_by_digit(&sc, &g);
            let (ngroup, span) = last_scatter_shape();
            set_scatter_threads(None);

            assert!(span > 0, "nw = {nw}, {thr} writers: scatter recorded no span");
            assert_eq!(
                N.div_ceil(span),
                ngroup,
                "nw = {nw}, {thr} writers: the recorded span does not tile the input"
            );

            let ents = entries(&sc, &g);

            // `bucket_total b` = how many entries bucket b holds.
            let mut tot = vec![0u32; g.nb];
            // `group_width b gi`, laid out group-major so the prefix below is
            // a sequential walk.
            let mut gw = vec![0u32; ngroup * g.nb];
            for &(b, i) in &ents {
                tot[b] += 1;
                gw[(i as usize / span) * g.nb + b] += 1;
            }

            // `edge tot b` — the outer decomposition's edges. This is what the
            // binner returns as `Off`, so the model and the code agree on the
            // whole first level before a single entry is placed.
            let mut want_off = Vec::with_capacity(g.nb + 1);
            let mut run = 0u32;
            for b in 0..g.nb {
                want_off.push(run);
                run += tot[b];
            }
            want_off.push(run);
            assert_eq!(
                off, want_off,
                "nw = {nw}, {thr} writers: Off is not the prefix of the histogram totals"
            );

            // `edge tot b + edge (gw b) gi` — the cursor the scatter builds,
            // recomputed from the model rather than read out of the binner.
            let mut cursor = vec![0u32; ngroup * g.nb];
            for b in 0..g.nb {
                let mut run = want_off[b];
                for gi in 0..ngroup {
                    cursor[gi * g.nb + b] = run;
                    run += gw[gi * g.nb + b];
                }
                assert_eq!(run, want_off[b + 1], "nw = {nw}: group widths do not exhaust bucket {b}");
            }

            // Walk the entries in the order the model says the ranks are
            // assigned — a group takes its points in index order — and check
            // each one is where `dest` puts it.
            let mut rank = vec![0u32; ngroup * g.nb];
            let mut degenerate_groups = 0usize;
            for &(b, i) in &ents {
                let gi = i as usize / span;
                let cell = gi * g.nb + b;
                let dest = cursor[cell] + rank[cell];
                assert_eq!(
                    idx[dest as usize], i,
                    "nw = {nw}, {thr} writers: entry (point {i}, bucket {b}) is not at \
                     dest = off[{b}] + edge(gw[{b}])[{gi}] + {} = {dest}",
                    rank[cell]
                );
                rank[cell] += 1;
            }
            for cell in 0..ngroup * g.nb {
                if gw[cell] == 0 {
                    degenerate_groups += 1;
                }
            }

            // The controls that stop this being a one-level test. `ngroup > 1`
            // is what makes the inner decomposition real at all; an empty
            // group inside a non-empty bucket is the case `scatter`'s own
            // post-condition got wrong the first time it was written, and
            // `an_empty_group_still_chains` is the proof's counterpart.
            assert!(
                ngroup > 1,
                "nw = {nw}, {thr} writers: only one scatter group, so the second level \
                 of the decomposition is not exercised"
            );
            assert!(
                degenerate_groups > 0,
                "nw = {nw}, {thr} writers: no empty (group, bucket) cell, so the \
                 zero-width part is untested"
            );
        }
    }
}
