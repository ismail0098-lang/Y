//! The micro-kernel's flush and lane routing: the model against the theorem,
//! and the flush interval's bound against the running kernel.
//!
//! `proofs/ExactGemmMicro.v` proves three things:
//!
//! - `flush_exact` — the chunks `[c, min(c+F, kpairs))` sum to the whole range,
//!   with no hypothesis that `F` divides `kpairs`.
//! - `flush_exact_in_int32` + `operand_bound_gives_no_overflow` — the int32
//!   accumulator agrees with `Z` exactly when no partial sum leaves the int32
//!   range, and the licence's `2 * F * m^2 <= i32::MAX` is what supplies that.
//! - `the_packed_column_is_the_stored_column` — the column `pack_b` routes to a
//!   lane is the column the store reads that lane back out to.
//!
//! **The interval-invariance half is already covered** by
//! `cpu_gemm_exact_threaded.rs`, which links three flush intervals into one
//! process and compares them bit for bit. What was NOT covered is the other
//! direction: whether the bound is *necessary*. A licence nothing can violate
//! is indistinguishable from a licence that certifies nothing, so
//! `exceeding_the_bound_by_one_really_does_go_wrong` drives the emitted symbols
//! past it directly and asserts the answer changes — with an in-licence control
//! one unit below, because "it went wrong" is also what a broken harness says.

use std::path::PathBuf;
use std::process::Command;

use y::cpu_gemm::{
    column_of_lane, flush_chunks, lane_of_slot, pack_b_slot, vec_of_slot, VNNI_NR,
};
use y::zero_drift::VnniExact;

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

/// `flush_exact`, finitely: the chunks tile `[0, kpairs)` with no gap, no
/// overlap, and nothing dropped — including when `F` does not divide `kpairs`,
/// which is the case the emitted `select` clamp exists for.
#[test]
fn the_flush_chunks_tile_the_k_pair_range() {
    for flush in [1usize, 2, 3, 7, 64, 512] {
        for kpairs in 0usize..200 {
            let chunks = flush_chunks(kpairs, flush);
            let mut next = 0usize;
            for &(start, len) in &chunks {
                assert_eq!(start, next, "chunk gap or overlap at kpairs={kpairs} F={flush}");
                assert!(len > 0, "empty chunk at kpairs={kpairs} F={flush}");
                assert!(len <= flush, "chunk longer than the interval");
                next = start + len;
            }
            assert_eq!(
                next, kpairs,
                "the chunks stop at {next} instead of {kpairs} (F={flush}); a loop that \
                 halted at the last WHOLE interval would drop exactly this tail"
            );
        }
    }
    // Every chunk but the last is full — this is the clamped-tile shape, not
    // the K-split's uneven bands, and confusing the two is how a schedule
    // proof gets pointed at the wrong theorem.
    let c = flush_chunks(150, 64);
    assert_eq!(c, vec![(0, 64), (64, 64), (128, 22)]);
}

/// The model and the emitter must AGREE on the interval, rather than the test
/// keeping a third copy of it.
#[test]
fn the_emitter_uses_the_interval_it_is_given() {
    for flush in [8u32, 64, 512] {
        let m = y::cpu_gemm::emit_vnni_micro_module(flush);
        assert!(
            m.contains(&format!("add i64 %c, {flush}")),
            "the micro-kernel emitted at interval {flush} never advances by it; \
             `flush_chunks` and proofs/ExactGemmMicro.v would be describing a \
             loop the code does not run"
        );
    }
    // ...and the shipped default is the one the proof's boundary case names.
    assert_eq!(VnniExact::DEFAULT_FLUSH_K_PAIRS, 64);
}

/// `operand_bound_gives_no_overflow` states the licence as
/// `2 * F * m^2 <= i32::MAX`. Assert the real function AGREES with that, rather
/// than re-deriving the table here — a second copy of a rule is how two
/// producers come to disagree.
#[test]
fn the_licence_is_the_bound_the_proof_assumes() {
    for flush in [1u32, 2, 8, 64, 512, 65536] {
        let v = VnniExact::new(flush).expect("a positive interval is constructible");
        for m in [0u64, 1, 100, 127, 128, 4094, 4095, 4096, 4097, 32767] {
            let proof_bound = 2u128 * u128::from(flush) * u128::from(m) * u128::from(m)
                <= u128::from(i32::MAX as u32);
            // The real function also refuses non-representable widths and
            // sub-integer magnitudes, which the proof does not model; restrict
            // the comparison to the domain the proof is about. Every `m` here
            // is a non-negative INTEGER, which is exactly the domain on which
            // the shipping `license` and the `licenses` bool this used to call
            // agree - so switching to `license` strengthens the tie rather
            // than weakening it: it is now the predicate the compiler uses.
            if m > 32767 {
                continue;
            }
            assert_eq!(
                v.license(m as f64).is_ok(),
                proof_bound,
                "flush={flush} m={m}: the licence and \
                 `operand_bound_gives_no_overflow`'s hypothesis disagree"
            );
        }
    }
}

/// The boundary the proof pins at exactly one unit wide, and the exhaustive
/// test in `exact_gemm_licence_obligations.rs` finds by exhausting int16.
#[test]
fn the_default_interval_boundary_is_one_unit_wide() {
    let v = VnniExact::new(VnniExact::DEFAULT_FLUSH_K_PAIRS).unwrap();
    assert!(v.license(4095.0).is_ok());
    assert!(v.license(4096.0).is_err());
    // `the_4096_case_exceeds_by_exactly_one`, in the same arithmetic.
    assert_eq!(2i64 * 64 * 4096 * 4096, i64::from(i32::MAX) + 1);
    assert_eq!(i64::from(i32::MAX) - 2 * 64 * 4095 * 4095, 1_048_447);
}

/// `the_packed_column_is_the_stored_column`: the cross-file tie between
/// `ExactGemmPacking.v` (where a column goes) and `ExactGemmMicro.v` (where it
/// comes back from).
#[test]
fn the_lane_round_trip_is_the_identity() {
    for j in 0..VNNI_NR {
        for h in 0..2 {
            let s = pack_b_slot(j, h);
            assert_eq!(
                column_of_lane(vec_of_slot(s), lane_of_slot(s)),
                j,
                "column {j} (half {h}) is packed into vector {} lane {} and stored \
                 back out to column {} — the kernel would compute a correctly \
                 summed but column-PERMUTED tile, which no bijection or bound \
                 elsewhere can see",
                vec_of_slot(s),
                lane_of_slot(s),
                column_of_lane(vec_of_slot(s), lane_of_slot(s))
            );
        }
    }
    // The control: the round trip is NOT the identity for a lane stride the
    // hardware does not use, so this is a claim about these two maps rather
    // than about any pair that happens to compose.
    let s = pack_b_slot(8, 0);
    assert_ne!(column_of_lane(vec_of_slot(s), (s % 32) / 4), 8);
}

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void __y_gemm_exact_vnni_threaded(const int16_t *A, const int16_t *B, int64_t *C,
                                  long M, long N, long K,
                                  long lda, long ldb, long ldc);

/* One full flush interval and not a k-pair more: K = 128 is kpairs = 64,
   which is exactly DEFAULT_FLUSH_K_PAIRS, so the int32 accumulator carries
   every product and the interval bound is the only thing standing between
   this and a wrap. Constant operands, because the bound is a WORST case and
   random data does not reach it. */
int main(int argc, char **argv) {
    long v = atol(argv[1]);
    const long M = 6, N = 64, K = 128;

    int16_t *A = malloc((size_t)M * K * 2);
    int16_t *B = malloc((size_t)K * N * 2);
    int64_t *C = malloc((size_t)M * N * 8);
    for (long i = 0; i < M * K; ++i) A[i] = (int16_t)v;
    for (long i = 0; i < K * N; ++i) B[i] = (int16_t)v;
    for (long i = 0; i < M * N; ++i) C[i] = 0;

    __y_gemm_exact_vnni_threaded(A, B, C, M, N, K, K, N, N);

    /* Every element is the same sum, in int64 where it cannot wrap. */
    long long want = (long long)K * v * v;
    long bad = 0;
    for (long i = 0; i < M * N; ++i) if (C[i] != want) bad++;
    printf("want %lld got %lld bad %ld\n", want, (long long)C[0], bad);
    printf("DONE\n");
    return 0;
}
"#;

/// **Is the licence necessary, or is it paperwork?** Measured.
///
/// The compiler REFUSES operands beyond the licence, so this is reachable only
/// by calling the emitted symbols directly — which is what makes it worth
/// asserting: it is the behavioural form of `overflow_breaks_the_flush`, and
/// without it "no test ever violates the bound" and "the bound does nothing"
/// look identical.
///
/// `K = 128` is `kpairs = 64`, exactly one full default interval, with constant
/// operands because the bound is a worst case that random data never reaches.
/// At `v = 4095` the interval sum is `2,146,435,200` and fits; at `v = 4096` it
/// is `2,147,483,648`, over by exactly one, and the int32 accumulator wraps to
/// the negative of it.
#[test]
fn exceeding_the_bound_by_one_really_does_go_wrong() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    if !host_has_vnni() {
        eprintln!("skipping: host CPU has no avx512_vnni");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_micro_bound_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    // `malloc`/`free` come from `llvm_emitter`'s prelude, not from either
    // cpu_gemm module - so a module emitted standalone has to supply them.
    // The sibling tests do not hit this because they compile a `.ysu` through
    // the real binary and inherit the prelude.
    let ir = format!(
        "declare ptr @malloc(i64)\ndeclare void @free(ptr)\n{}\n{}",
        y::cpu_gemm::emit_vnni_gemm_module(VnniExact::DEFAULT_FLUSH_K_PAIRS),
        y::cpu_gemm::emit_vnni_threaded_module(true)
    );
    let ll = dir.join("m.ll");
    std::fs::write(&ll, ir).expect("write IR");
    let drv = dir.join("drv.c");
    std::fs::write(&drv, DRIVER).expect("write driver");
    let exe = dir.join("run");
    let cc = Command::new("clang")
        .args(["-O2", "-x", "ir"])
        .arg(&ll)
        .args(["-x", "c"])
        .arg(&drv)
        .args(["-lpthread", "-o"])
        .arg(&exe)
        .output()
        .expect("clang");
    assert!(
        cc.status.success(),
        "link failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    let run = |v: i64| -> (i64, i64) {
        let out = Command::new(&exe)
            .arg(v.to_string())
            .env("Y_NUM_THREADS", "1")
            .output()
            .expect("run");
        let t = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(t.contains("DONE"), "v={v} did not finish:\n{t}");
        let f: Vec<&str> = t.lines().next().unwrap().split_whitespace().collect();
        (f[1].parse().unwrap(), f[3].parse().unwrap())
    };

    // In licence by one: exact.
    let (want_ok, got_ok) = run(4095);
    assert_eq!(want_ok, 2_146_435_200, "the fixture no longer sits on the edge");
    assert_eq!(
        got_ok, want_ok,
        "an operand magnitude the licence ADMITS came back wrong; the bound is \
         not merely tight, it is incorrect"
    );

    // Over by one: the int32 accumulator wraps.
    let (want_bad, got_bad) = run(4096);
    assert_eq!(want_bad, 2_147_483_648, "the fixture no longer sits on the edge");
    assert_ne!(
        got_bad, want_bad,
        "one unit past the licence still gave the exact answer, so the flush \
         interval bound is certifying nothing. Either the accumulator is no \
         longer int32, or the interval is no longer honoured — both make \
         `VnniExact::license` paperwork"
    );
    assert_eq!(
        got_bad, -2_147_483_648,
        "it went wrong, but not by wrapping — check the kernel still accumulates \
         in int32 before concluding the bound is what failed"
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = PathBuf::new();
}
