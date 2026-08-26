//! What the packed panel CONTAINS, checked against the emitted packers.
//!
//! `proofs/ExactGemmPacking.v` proved for a long time only which slot a write
//! lands in - that the destination maps are bijections, and that the padded
//! product equals the live dot product. It never said what the panel ARRAY
//! holds, and that gap was not academic: `ExactGemmComposition.v` needed
//! exactly that statement and had to take it as a hypothesis.
//!
//! **The hypothesis as first written was false of the real panel.** It
//! quantified over every `i`, and at `i = MR` the index it names,
//! `p*(2*MR) + 2*MR`, is the FIRST slot of k-pair group `p+1` - which holds
//! that group's data, not a pad. So the composition theorem was true and
//! unusable: applying it meant supplying a premise satisfied only by a panel
//! one group long. `the_group_bound_is_load_bearing` refutes the unbounded
//! form in Coq; `the_next_group_is_not_padding` below refutes it against the
//! bytes the emitted packer actually writes, which is the half a proof cannot
//! reach.
//!
//! `panel_decodes_its_own_write` and `panel_is_the_only_solution` are the
//! discharge. This file is their behavioural tie: run the REAL
//! `__y_gemm_vnni_pack_a` / `__y_gemm_vnni_pack_b` and compare every slot of
//! the resulting panel against `panel_slot_decode`.
//!
//! **Every stride differs from its extent**, per the rule the tiling work
//! established: `lda != kc` and `ldb != ncols`, because equality is a
//! coincidence that hides address bugs. The source values are non-zero
//! everywhere, so a dropped mask shows up as a live value in a pad rather than
//! as a zero that a zero-initialised buffer would have supplied anyway.

use std::process::Command;

use y::cpu_gemm::{panel_slot_decode, VNNI_MR, VNNI_NR};

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// The source matrices, as functions. Non-zero everywhere and asymmetric in
/// both indices, so a transposed or swapped access is visible.
fn a_val(i: usize, k: usize) -> i32 {
    1 + 7 * i as i32 + 3 * k as i32
}
fn b_val(k: usize, j: usize) -> i32 {
    2 + 5 * k as i32 + 3 * j as i32
}

/// `panel` from `proofs/ExactGemmPacking.v`, transcribed.
///
/// `val(idx, k)` is the k-indexed vector row/column `idx` contributes: `A i`
/// for an A panel, `fun k => B k j` for a B one.
fn panel_model(width: usize, extent: usize, kc: usize, s: usize, val: &dyn Fn(usize, usize) -> i32) -> i32 {
    let (p, idx, h) = panel_slot_decode(s, width);
    let k = 2 * p + h;
    if idx < extent && k < kc {
        val(idx, k)
    } else {
        0
    }
}

fn kpairs(kc: usize) -> usize {
    (kc + 1) / 2
}

/// Two exported wrappers, because the packers have `internal` linkage and are
/// callable only from inside their own module.
const WRAPPERS: &str = r#"
define void @y_test_pack_a(ptr noalias %s, i64 %lda, i64 %m, i64 %kc, ptr noalias %d) {
  call void @__y_gemm_vnni_pack_a(ptr %s, i64 %lda, i64 %m, i64 %kc, ptr %d)
  ret void
}
define void @y_test_pack_b(ptr noalias %s, i64 %ldb, i64 %kc, i64 %n, ptr noalias %d) {
  call void @__y_gemm_vnni_pack_b(ptr %s, i64 %ldb, i64 %kc, i64 %n, ptr %d)
  ret void
}
"#;

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>

#define MR 6
#define NR 64

void y_test_pack_a(short *src, long lda, long mrows, long kc, short *dst);
void y_test_pack_b(short *src, long ldb, long kc, long ncols, short *dst);

int main(int argc, char **argv) {
    long mrows = atol(argv[1]);
    long kc    = atol(argv[2]);
    long ncols = atol(argv[3]);
    long lda   = atol(argv[4]);
    long ldb   = atol(argv[5]);
    long kp    = (kc + 1) / 2;

    /* Rows/cols beyond the live tile exist in memory and hold live-looking
       values: the packer must mask them, not rely on them being absent. */
    long arows = MR + 3, acols = lda;
    short *A = malloc(sizeof(short) * arows * acols);
    for (long i = 0; i < arows; i++)
        for (long k = 0; k < acols; k++)
            A[i * lda + k] = (short)(1 + 7 * i + 3 * k);

    long brows = kc + 3, bcols = ldb;
    short *B = malloc(sizeof(short) * brows * bcols);
    for (long k = 0; k < brows; k++)
        for (long j = 0; j < bcols; j++)
            B[k * ldb + j] = (short)(2 + 5 * k + 3 * j);

    long alen = kp * 2 * MR, blen = kp * 2 * NR;
    /* Poisoned, so a slot the packer fails to write is visible rather than
       reading back as the zero a calloc would have supplied. */
    short *Ap = malloc(sizeof(short) * alen);
    short *Bp = malloc(sizeof(short) * blen);
    for (long s = 0; s < alen; s++) Ap[s] = -31337;
    for (long s = 0; s < blen; s++) Bp[s] = -31337;

    y_test_pack_a(A, lda, mrows, kc, Ap);
    y_test_pack_b(B, ldb, kc, ncols, Bp);

    printf("A %ld\n", alen);
    for (long s = 0; s < alen; s++) printf("%d\n", (int)Ap[s]);
    printf("B %ld\n", blen);
    for (long s = 0; s < blen; s++) printf("%d\n", (int)Bp[s]);
    printf("DONE\n");
    return 0;
}
"#;

struct Panels {
    a: Vec<i32>,
    b: Vec<i32>,
}

/// Build the module once, then run it per shape.
///
/// `tag` is per-test and belongs in the SIGNATURE, not in a comment asking the
/// next author to remember. Both tests in this file call this; a shared
/// directory name means one test's `remove_dir_all` lands while the other is
/// writing its IR, which presents as `write IR: NotFound` with the module
/// perfectly valid. That race has now fired five times in this repo (the GPU
/// `.ptx` harness, `backend_differential.rs`, `exact_gemm_chain_model.rs`,
/// `exact_gemm_tile_enumeration.rs`, here) - it is a property of any helper
/// that materialises files in a temp directory.
fn build(tag: &str) -> Option<std::path::PathBuf> {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("y_panel_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let ll = dir.join("m.ll");
    let mut module = y::cpu_gemm::emit_vnni_gemm_module(64);
    module.push_str(WRAPPERS);
    std::fs::write(&ll, module).expect("write IR");
    let drv = dir.join("drv.c");
    std::fs::write(&drv, DRIVER).expect("write driver");
    let exe = dir.join("run");
    let cc = Command::new("clang")
        .args(["-O2", "-x", "ir"])
        .arg(&ll)
        .args(["-x", "c"])
        .arg(&drv)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("clang");
    assert!(
        cc.status.success(),
        "link failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );
    Some(exe)
}

fn run(exe: &std::path::Path, mrows: usize, kc: usize, ncols: usize, lda: usize, ldb: usize) -> Panels {
    let out = Command::new(exe)
        .args([
            mrows.to_string(),
            kc.to_string(),
            ncols.to_string(),
            lda.to_string(),
            ldb.to_string(),
        ])
        .output()
        .expect("run");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success() && text.contains("DONE"),
        "driver did not finish:\n{text}"
    );
    let mut lines = text.lines();
    let mut take = |tag: &str| -> Vec<i32> {
        let hdr = lines.next().unwrap_or_default();
        let n: usize = hdr
            .strip_prefix(&format!("{tag} "))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or_else(|| panic!("expected `{tag} <len>`, got `{hdr}`"));
        (0..n)
            .map(|_| lines.next().unwrap().trim().parse::<i32>().unwrap())
            .collect()
    };
    let a = take("A");
    let b = take("B");
    Panels { a, b }
}

/// The shapes. Ragged M, ragged N, and an odd `kc` whose phantom high half is
/// a third padding distinct from the other two.
const SHAPES: &[(usize, usize, usize)] = &[
    (VNNI_MR, 8, VNNI_NR),  // full tile, even kc
    (VNNI_MR, 7, VNNI_NR),  // phantom k-half only
    (4, 8, VNNI_NR),        // ragged M
    (VNNI_MR, 8, 53),       // ragged N
    (4, 7, 53),             // all three at once
];

/// **Every slot of both panels is what the model says it is.**
///
/// This is the behavioural half of `panel_decodes_its_own_write`. It is
/// stronger than the syntactic constant check in
/// `exact_gemm_packing_model.rs`: that one asserts the group stride appears as
/// a literal in the packer's body, this one asserts the bytes land where the
/// stride says. The panels are poisoned first, so a slot the packer never
/// writes fails rather than reading back as a plausible zero.
#[test]
fn every_panel_slot_is_what_the_model_says() {
    let Some(exe) = build("slots") else { return };

    for &(mrows, kc, ncols) in SHAPES {
        let (lda, ldb) = (kc + 5, ncols + 9);
        let p = run(&exe, mrows, kc, ncols, lda, ldb);
        assert_eq!(p.a.len(), kpairs(kc) * 2 * VNNI_MR);
        assert_eq!(p.b.len(), kpairs(kc) * 2 * VNNI_NR);

        for (s, &got) in p.a.iter().enumerate() {
            let want = panel_model(VNNI_MR, mrows, kc, s, &|i, k| a_val(i, k));
            assert_eq!(
                got, want,
                "A panel slot {s} of shape ({mrows}, {kc}, {ncols}): the emitted \
                 packer wrote {got}, panel_slot_decode says {want}. Decoded as \
                 {:?}",
                panel_slot_decode(s, VNNI_MR)
            );
        }
        for (s, &got) in p.b.iter().enumerate() {
            let want = panel_model(VNNI_NR, ncols, kc, s, &|j, k| b_val(k, j));
            assert_eq!(
                got, want,
                "B panel slot {s} of shape ({mrows}, {kc}, {ncols}): the emitted \
                 packer wrote {got}, panel_slot_decode says {want}. Decoded as \
                 {:?}",
                panel_slot_decode(s, VNNI_NR)
            );
        }
    }
}

/// **The refutation, against real bytes: the slot past a group is the NEXT
/// GROUP'S DATA, not padding.**
///
/// This is what makes the `idx < width` bound on the composition's contract a
/// correction rather than a tightening. With `mrows = MR` the unbounded
/// contract demands a 0 at `p*(2*MR) + 2*MR` - row `MR` is past the live tile,
/// so its packed value would be masked. The panel holds group `p+1`'s
/// `A[0][2p+2]` there instead.
///
/// `kc = 8` gives four k-pair groups, so there is a group after group 0 for
/// this to be about. The control is the test above: inside its own group the
/// same panel agrees with the model everywhere.
#[test]
fn the_next_group_is_not_padding() {
    let Some(exe) = build("group") else { return };

    let (mrows, kc, ncols) = (VNNI_MR, 8, VNNI_NR);
    let p = run(&exe, mrows, kc, ncols, kc + 5, ncols + 9);

    for grp in 0..kpairs(kc) - 1 {
        let s = grp * (2 * VNNI_MR) + 2 * VNNI_MR;
        let unbounded_would_demand = 0; // row MR is past mrows = MR
        let next_groups_first_element = a_val(0, 2 * (grp + 1));
        assert_eq!(
            p.a[s], next_groups_first_element,
            "slot {s} should be group {}'s (i=0, h=0) element",
            grp + 1
        );
        assert_ne!(
            p.a[s], unbounded_would_demand,
            "the refutation is vacuous: group {}'s first element is itself 0, so \
             an unbounded contract would happen to hold here",
            grp + 1
        );
    }
}

/// The pure half: `panel_slot_decode` really is the inverse of the write map,
/// and it is ONTO the panel - every slot decodes to a write some `(p, idx, h)`
/// performs. Those two are `panel_is_the_only_solution`'s combinatorial core:
/// injectivity makes the packer's write specification consistent, surjectivity
/// makes it complete, and together they are why the decoded panel is the only
/// array the loop can produce.
#[test]
fn the_decode_inverts_the_write_map_over_the_whole_panel() {
    for &width in &[VNNI_MR, VNNI_NR] {
        for kp in 1..6usize {
            let mut seen = vec![false; kp * 2 * width];
            for p in 0..kp {
                for idx in 0..width {
                    for h in 0..2 {
                        let s = p * (2 * width) + 2 * idx + h;
                        assert_eq!(
                            panel_slot_decode(s, width),
                            (p, idx, h),
                            "width {width}: slot {s} does not decode to ({p}, {idx}, {h})"
                        );
                        assert!(!seen[s], "width {width}: slot {s} written twice");
                        seen[s] = true;
                    }
                }
            }
            assert!(
                seen.iter().all(|&x| x),
                "width {width}, {kp} groups: the write map misses a slot, so some \
                 slot keeps the previous tile's value"
            );
        }
    }
}
