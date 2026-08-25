//! The register tile's ROUTING, driven through the micro-kernel directly.
//!
//! `proofs/ExactGemmRegisterTile.v` proves that the emitted broadcast, the four
//! `<32 x i16>` B loads and `vpdpwssd` send panel slot `2j + h` to the
//! accumulator lane whose column is `j`, and that the i32 load of A aliases the
//! packed pair `2i`/`2i+1`.
//!
//! **The behavioural half calls `__y_gemm_micro_vnni` with HAND-BUILT panels,
//! and that is the point - DEMONSTRATED, not asserted.** Every other
//! exact-GEMM test goes through the full driver, where the packers and the
//! routing are composed, so a packing bug and a routing bug that are inverse
//! to each other cancel. A single-site mutation does not show this: breaking
//! the B vector offset alone fails every GEMM test in the repo. The
//! compensating pair does. XOR the vector index in BOTH `emit_vnni_pack_b`
//! (`v := v ^ 1`) and the flush's store column (`c = (v ^ 1) * 16`) - the two
//! cancel, and:
//!
//! | | packing_model | thread_invariance | cpu_gemm_exact_threaded | this file |
//! |---|---|---|---|---|
//! | compensating pair | ok | ok | ok | **FAILED** |
//!
//! **That table predates `exact_gemm_panel_model.rs`, which also catches it** -
//! by a different route, and the difference is worth keeping. This file breaks
//! the composition by stating the panel contents; the panel model breaks it by
//! checking the panel the real packers produced, slot by slot, against the
//! model. Either kills the cancellation. `exact_gemm_chain_model.rs`, which
//! composes both halves against a source-level reference, does NOT catch it -
//! measured, and recorded in that file.
//!
//! Building the panel by hand breaks the composition: the panel contents are
//! stated by the test, so only the routing is under test and there is nothing
//! left for a packer to compensate with.
//!
//! The fixture is ASYMMETRIC in both axes on purpose. With a symmetric operand
//! a swapped lane or a swapped pair-half is invisible - `ExactGemmRegisterTile`
//! records that as `a_symmetric_operand_hides_the_swap`, which is why the
//! refutation beside it uses distinct values.

use std::process::Command;

use y::cpu_gemm::{
    a_i32_element, a_pair_slots, b_pair_slots_for_lane, column_of_lane, pack_a_slot,
    pack_b_slot, VNNI_MR, VNNI_NR, VNNI_NRV,
};

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn host_has_vnni() -> bool {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.contains("avx512_vnni"))
        .unwrap_or(false)
}

/// `tile_position_injective` + `_surjective`, finitely: the 24 accumulators of
/// 16 lanes name each of the tile's 64 columns exactly once.
#[test]
fn the_accumulator_lanes_tile_the_columns_exactly_once() {
    let mut seen = vec![0usize; VNNI_NR];
    for v in 0..VNNI_NRV {
        for l in 0..16 {
            let j = column_of_lane(v, l);
            assert!(j < VNNI_NR, "vector {v} lane {l} names column {j}, past the tile");
            seen[j] += 1;
        }
    }
    for (j, n) in seen.iter().enumerate() {
        assert_eq!(*n, 1, "column {j} is named by {n} accumulator lanes, not one");
    }
    assert_eq!(VNNI_NRV * 16, VNNI_NR);
}

/// `the_i32_load_is_the_packed_pair` and `the_lane_consumes_its_own_column`,
/// as the model: the slots the kernel reads are the slots the packers write.
#[test]
fn the_slots_the_tile_reads_are_the_slots_the_packers_write() {
    for p in 0..3 {
        for i in 0..VNNI_MR {
            // The i32 element index, and the int16 pair it aliases.
            assert_eq!(a_i32_element(p, i), p * VNNI_MR + i);
            let (lo, hi) = a_pair_slots(p, i);
            assert_eq!(lo, 2 * a_i32_element(p, i));
            assert_eq!(hi, 2 * a_i32_element(p, i) + 1);
            assert_eq!(lo, p * (VNNI_MR * 2) + pack_a_slot(i, 0));
            assert_eq!(hi, p * (VNNI_MR * 2) + pack_a_slot(i, 1));
        }
        for v in 0..VNNI_NRV {
            for l in 0..16 {
                let j = column_of_lane(v, l);
                let (b0, b1) = b_pair_slots_for_lane(p, v, l);
                // The lane reads elements 2l and 2l+1 of the vector loaded at
                // Bp + p*NR*2 + v*32 - which must be column j's two slots.
                assert_eq!(b0, p * (VNNI_NR * 2) + v * 32 + 2 * l);
                assert_eq!(b1, p * (VNNI_NR * 2) + v * 32 + 2 * l + 1);
                assert_eq!(b0, p * (VNNI_NR * 2) + pack_b_slot(j, 0));
                assert_eq!(b1, p * (VNNI_NR * 2) + pack_b_slot(j, 1));
            }
        }
    }
}

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void __y_gemm_micro_vnni(const int16_t *Ap, const int16_t *Bp, int64_t *C,
                         long ldc, long kpairs);

#define MR 6
#define NR 64

int main(void) {
    const long KP = 3;                 /* three k-pair groups */
    int16_t *Ap = aligned_alloc(64, (size_t)KP * MR * 2 * 2);
    int16_t *Bp = aligned_alloc(64, (size_t)KP * NR * 2 * 2);
    int64_t *C  = aligned_alloc(64, (size_t)MR * NR * 8);

    /* Position-encoded and ASYMMETRIC in every axis: the two halves of a pair
       differ, and the value grows with the row/column index, so a swapped
       half or a permuted lane cannot cancel. Small enough that MR*NR*KP
       products stay far inside int32. */
    for (long p = 0; p < KP; ++p) {
        for (int i = 0; i < MR; ++i)
            for (int h = 0; h < 2; ++h)
                Ap[p*MR*2 + 2*i + h] = (int16_t)(1 + 7*p + 2*i + h);
        for (int j = 0; j < NR; ++j)
            for (int h = 0; h < 2; ++h)
                Bp[p*NR*2 + 2*j + h] = (int16_t)(1 + 3*p + 2*j + h);
    }

    for (long t = 0; t < (long)MR * NR; ++t) C[t] = 0;   /* the kernel ACCUMULATES */
    __y_gemm_micro_vnni(Ap, Bp, C, NR, KP);

    long wrong = 0;
    for (int i = 0; i < MR; ++i)
        for (int j = 0; j < NR; ++j) {
            int64_t want = 0;
            for (long p = 0; p < KP; ++p)
                for (int h = 0; h < 2; ++h)
                    want += (int64_t)Ap[p*MR*2 + 2*i + h] * (int64_t)Bp[p*NR*2 + 2*j + h];
            if (C[(long)i*NR + j] != want) {
                if (wrong < 3)
                    printf("bad i=%d j=%d got %lld want %lld\n",
                           i, j, (long long)C[(long)i*NR + j], (long long)want);
                wrong++;
            }
        }
    printf("wrong %ld\n", wrong);
    printf("DONE\n");
    return 0;
}
"#;

/// **The routing, on the real instruction, with the packers taken out of the
/// loop.**
///
/// `__y_gemm_micro_vnni` is a public symbol taking the panels directly, so the
/// test states the panel contents itself. If lane `l` of vector `v` consumed
/// any column other than `16v + l`, or the i32 load paired the halves the
/// other way round, this disagrees with its own scalar reference - and the
/// full-GEMM tests would not, because there the packers could compensate.
#[test]
fn every_accumulator_lane_gets_its_own_column() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    if !host_has_vnni() {
        eprintln!("skipping: host CPU has no avx512_vnni");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_regtile_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let ll = dir.join("m.ll");
    std::fs::write(
        &ll,
        y::cpu_gemm::emit_vnni_micro_module(y::zero_drift::VnniExact::DEFAULT_FLUSH_K_PAIRS),
    )
    .expect("write IR");
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

    let run = Command::new(&exe).output().expect("run");
    let text = String::from_utf8_lossy(&run.stdout).into_owned();
    assert!(run.status.success() && text.contains("DONE"), "did not finish:\n{text}");
    let wrong: i64 = text
        .lines()
        .find_map(|l| l.strip_prefix("wrong "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or_else(|| panic!("no `wrong` line in:\n{text}"));
    assert_eq!(
        wrong, 0,
        "{wrong} of {} tile positions disagree with the scalar reference. The \
         panels here are built by the test, so the packers cannot be at fault - \
         this is the broadcast, the B vector offsets or the lane mapping:\n{text}",
        VNNI_MR * VNNI_NR
    );

    let _ = std::fs::remove_dir_all(&dir);
}
