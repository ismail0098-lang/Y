//! The operand packing: the model against the theorem, and the packers against
//! POISONED operand padding.
//!
//! `proofs/ExactGemmPacking.v` proves two things about `pack_a` / `pack_b`:
//! each destination map is a bijection onto its panel (nothing written twice,
//! no slot left holding the previous tile), and the full padded
//! `kpairs x 2` product equals the dot product over the live `kc`. The second
//! is what licenses the micro-kernel running a ragged tile at full width, and
//! it rests entirely on the packers' `select ... i16 0`.
//!
//! **The behavioural half needs GARBAGE in the operand padding, not zeros.**
//! Both packers read a clamped address and then mask the value. Delete the
//! mask and they return whatever sits at `A[i*lda + k]` for an out-of-range
//! `k` - which, if the buffer came from `calloc`, is **zero**, contributes
//! nothing to the dot product, and leaves every existing test green. The
//! padding is filled with live-range values here specifically so that a
//! dropped mask changes the answer.
//!
//! That is the same trap as pre-zeroing C in `exact_gemm_tiling_model.rs`: a
//! convenient buffer initialiser silently supplies the property under test.

use std::path::PathBuf;
use std::process::Command;

use y::cpu_gemm::{pack_a_slot, pack_b_slot, VNNI_MR, VNNI_NR};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `pack_a_slot_bijective` + `pack_a_slot_onto`, finitely.
#[test]
fn the_a_panel_slots_are_a_bijection() {
    let mut seen = vec![0u32; 2 * VNNI_MR];
    for i in 0..VNNI_MR {
        for h in 0..2 {
            let s = pack_a_slot(i, h);
            assert!(
                s < 2 * VNNI_MR,
                "pack_a_slot({i}, {h}) = {s} is outside the {}-slot group",
                2 * VNNI_MR
            );
            seen[s] += 1;
        }
    }
    for (s, n) in seen.iter().enumerate() {
        assert_eq!(
            *n, 1,
            "A-panel slot {s} is written {n} times, not once. A slot written \
             zero times keeps whatever the PREVIOUS tile left there - the panel \
             buffer is reused across tiles."
        );
    }
}

/// `pack_b_slot_bijective` + `pack_b_slot_in_panel`, finitely. This is the
/// `vpdpwssd` lane layout, so the sweep is over the whole 64-column panel.
#[test]
fn the_b_panel_slots_are_a_bijection() {
    let mut seen = vec![0u32; 2 * VNNI_NR];
    for j in 0..VNNI_NR {
        for h in 0..2 {
            let s = pack_b_slot(j, h);
            assert!(
                s < 2 * VNNI_NR,
                "pack_b_slot({j}, {h}) = {s} is outside the {}-slot group",
                2 * VNNI_NR
            );
            seen[s] += 1;
        }
    }
    for (s, n) in seen.iter().enumerate() {
        assert_eq!(*n, 1, "B-panel slot {s} is written {n} times, not once");
    }
}

/// The property bijectivity CANNOT establish, recorded as a test so the gap is
/// visible rather than implied - and the gap is wider than it looks.
///
/// The first version of this test asserted that `pack_b_slot` DIFFERS from the
/// naive `2*j + h`, on the reasoning that the `(j/16, j%16)` vector-group split
/// must be doing something. It does not: `16*(j/16) + (j%16) == j`, so the
/// whole decomposition folds and the map IS `2*j + h`. So bijectivity is not
/// merely too weak to tell the two apart - there are not two maps.
///
/// Asserting the equality is still worth doing, and is a strictly stronger
/// check than the one it replaces: the emitter carries the `32` and the `16`
/// as separate literals, and any inconsistent pair breaks it. What no test
/// here can cover is which int16 pair a hardware lane consumes; that is
/// `tests/cpu_gemm_vnni_micro.rs`, which mutates the stride against a scalar
/// reference on the real instruction.
#[test]
fn the_b_lane_layout_is_exactly_the_plain_interleave() {
    for j in 0..VNNI_NR {
        for h in 0..2 {
            assert_eq!(
                pack_b_slot(j, h),
                2 * j + h,
                "pack_b_slot({j}, {h}) is no longer the plain interleave; the \
                 vector-group constants have gone inconsistent, and \
                 slot_b_is_the_plain_interleave in proofs/ExactGemmPacking.v \
                 describes a layout the code no longer uses"
            );
        }
    }
    // Spot values, stated in lane terms so the derivation stays readable.
    assert_eq!(pack_b_slot(15, 1), 31, "last lane of the first vector group");
    assert_eq!(pack_b_slot(16, 0), 32, "column 16 opens the second group");
}

/// The model and the emitted packers must AGREE on their constants.
///
/// **Matched against the real emitted text, not a guess at it.** The first
/// version looked for `"i64 12"` and `"i64 128"`; the emitter writes
/// `mul nsw i64 %g9, 12`, so the operand sits between and neither needle was
/// ever a substring. Worse, `"i64 32"` PASSED - by matching a
/// `getelementptr inbounds i16, ptr %bp, i64 32` in the micro-kernel, nowhere
/// near a packer. A needle that matches the wrong function is a test that
/// retires the question, so each check is now scoped to the packer's own body
/// and anchored on the whole instruction.
fn packer_body<'a>(module: &'a str, name: &str) -> &'a str {
    let start = module
        .find(&format!("define internal void @{name}("))
        .unwrap_or_else(|| panic!("{name} is not in the emitted module"));
    let rest = &module[start..];
    let end = rest[1..].find("\ndefine ").map(|i| i + 1).unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn the_model_and_the_packers_agree_on_their_layout() {
    let module = y::cpu_gemm::emit_vnni_gemm_module(64);
    let pa = packer_body(&module, "__y_gemm_vnni_pack_a");
    let pb = packer_body(&module, "__y_gemm_vnni_pack_b");

    let has_mul = |body: &str, lit: usize| {
        body.lines().any(|l| {
            let l = l.trim();
            l.contains("mul nsw i64 ") && l.ends_with(&format!(", {lit}"))
        })
    };

    // The per-k-pair group strides: MR*2 int16 for A, NR*2 for B.
    assert!(
        has_mul(pa, VNNI_MR * 2),
        "pack_a never multiplies by its group stride {}; pack_a_slot and \
         proofs/ExactGemmPacking.v would be describing a layout the code no \
         longer uses",
        VNNI_MR * 2
    );
    assert!(
        has_mul(pb, VNNI_NR * 2),
        "pack_b never multiplies by its group stride {}",
        VNNI_NR * 2
    );

    // The lane split itself: v = j >> 4, l = j & 15, offset v*32 + l*2.
    for needle in ["lshr i64", "and i64", "mul nsw i64"] {
        assert!(
            pb.contains(needle),
            "pack_b no longer contains `{needle}`, so the (j/16, j%16) \
             derivation transcribed in pack_b_slot is gone"
        );
    }
    assert!(
        has_mul(pb, 32) && has_mul(pb, 2),
        "pack_b's vector-group constants (32 and 2) are not both present"
    );
    // A's slot map is a bare doubling, so it must NOT carry a lane split.
    assert!(
        !pa.contains("lshr i64"),
        "pack_a has grown a lane split; pack_a_slot is `2*i + h` and would be stale"
    );
}

/// The recognised exact nest. Identical to the one in
/// `exact_gemm_tiling_model.rs` on purpose: the two files test different
/// obligations (partition vs. packing) over the same program, so a divergence
/// between them would be a difference in the test, not in the compiler.
const SOURCE: &str = r#"
kernel y_matmul(A: GlobalMemory<I16>, B: GlobalMemory<I16>, C: GlobalMemory<I64>, M: I32, N: I32, K: I32) {
    @invariant(i >= 0)
    for i in 0..M step 1 {
        @invariant(j >= 0)
        for j in 0..N step 1 {
            @ZeroDrift
            let mut sum: I64 = 0;
            @invariant(k >= 0)
            for k in 0..K step 1 {
                @bounds(min=-1024, max=1024)
                let a_val: I64 = block_ptr2d_load(A, i, k, K, M, K);
                @bounds(min=-1024, max=1024)
                let b_val: I64 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a_val * b_val;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}

fn main() {
}
"#;

/// The operand padding is filled with LIVE-RANGE values, not zeros.
///
/// `calloc` would leave it zero, and a zero contributes nothing to a dot
/// product - so deleting either packer's mask would leave the answer unchanged
/// and every test green. The poison is chosen inside `@bounds(-1024, 1024)` so
/// it is a legitimate operand the kernel could have been given: if it reaches
/// the accumulator, the result is wrong rather than out of range.
const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

void __y_gemm_exact_vnni_threaded(const int16_t *A, const int16_t *B, int64_t *C,
                                  long M, long N, long K,
                                  long lda, long ldb, long ldc);

static int16_t gen(long i, long salt) {
    unsigned long h = i * 2654435761UL + salt * 40503UL;
    h ^= h >> 13;
    return (int16_t)((long)(h % 2049) - 1024);
}

int main(int argc, char **argv) {
    int M = 53, N = 71, K = 301;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    long lda = K + 5, ldb = N + 9, ldc = N + 7;

    int16_t *A = malloc((size_t)M * lda * 2);
    int16_t *B = malloc((size_t)K * ldb * 2);
    int64_t *C = malloc((size_t)M * ldc * 8);
    int64_t *R = malloc((size_t)M * N * 8);

    /* POISON EVERYTHING FIRST, live region included - it is overwritten below.
       Values alternate +-1000, inside the declared @bounds, so anything that
       leaks into the accumulator is a legal operand and a wrong answer rather
       than an overflow that a range check might catch by luck. */
    for (long i = 0; i < (long)M * lda; ++i) A[i] = (i & 1) ? 1000 : -1000;
    for (long i = 0; i < (long)K * ldb; ++i) B[i] = (i & 1) ? -1000 : 1000;

    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) A[(long)i*lda + k] = gen((long)i*K + k, 1);
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j) B[(long)k*ldb + j] = gen((long)k*N + j, 2);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            int64_t a = 0;
            for (int k = 0; k < K; ++k)
                a += (int64_t)A[(long)i*lda + k] * (int64_t)B[(long)k*ldb + j];
            R[(long)i*N + j] = a;
        }

    for (long i = 0; i < (long)M * ldc; ++i) C[i] = (int64_t)0xABABABABABABABABULL;

    __y_gemm_exact_vnni_threaded(A, B, C, M, N, K, lda, ldb, ldc);

    long wrong = 0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            if (C[(long)i*ldc + j] != R[(long)i*N + j]) wrong++;
    printf("wrong %ld\n", wrong);
    printf("DONE\n");
    return 0;
}
"#;

/// A ragged tile must exclude the operand padding, not merely survive it.
///
/// K is odd on purpose in most shapes: `kpairs = (kc+1)/2` rounds up, so an odd
/// K creates a phantom `k = K` whose high half must be zeroed. That is a THIRD
/// padding, distinct from the ragged M and the ragged N, and the proof's
/// `padded_product_is_the_live_dot_product` covers all three at once.
///
/// **What this test can and cannot catch, MEASURED rather than assumed.**
/// Removing a packer's `select ... i16 0` and sweeping all five shapes:
///
/// | mask removed | 53x71x301 | 48x128x301 | 53x128x300 | 48x71x300 | 48x128x300 |
/// |---|---|---|---|---|---|
/// | pack_a only | 0 | 0 | 0 | 0 | 0 |
/// | pack_b only | 0 | 0 | 0 | 0 | 0 |
/// | both | 3763 | 6144 | 0 | 0 | 0 |
///
/// Two facts fall out, and neither was visible from reading the code:
///
/// 1. **The two masks are redundant with EACH OTHER**, so neither alone is
///    load-bearing and no driver can make it so: a padding term is
///    `a_pad * b_pad`, and a zero on either side kills it. The obligation this
///    test really pins is the conjunction - *the padding contributes nothing* -
///    which is exactly what the proof states. Removing one mask leaves the
///    kernel correct and undefended, not wrong.
/// 2. **Only the phantom k-half can corrupt an answer at all.** The ragged M
///    and ragged N shapes report 0 even with both masks gone, because those
///    accumulator rows and columns are discarded by the C store mask before
///    anything reads them - the property proved in
///    `proofs/ExactGemmTiling.v`, doing double duty. So `48x128x301` is the
///    only shape here that can fail, and the other four are controls.
///
/// The row and column masks are therefore defence in depth against a future
/// change to the store, not correctness today. Recorded rather than deleted.
#[test]
fn the_packers_exclude_everything_outside_the_live_tile() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_exact_pack_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let src = dir.join("m.ysu");
    std::fs::write(&src, SOURCE).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-llvm")
        .current_dir(repo())
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "the exact nest must compile:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let driver = dir.join("drv.c");
    std::fs::write(&driver, DRIVER).expect("write driver");
    let exe = dir.join("run");
    let cc = Command::new("clang")
        .args([
            dir.join("m.ll").to_str().unwrap(),
            driver.to_str().unwrap(),
            "-O2",
            "-lm",
            "-lpthread",
            "-o",
            exe.to_str().unwrap(),
        ])
        .output()
        .expect("run clang");
    assert!(
        cc.status.success(),
        "linking failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    // Each shape isolates a padding: ragged M, ragged N, odd K (the phantom
    // k-pair half), and one with all three at once. The even-K, exact-tile case
    // is the control that proves nothing on its own.
    let shapes: [(usize, usize, usize); 5] = [
        (53, 71, 301),  // all three ragged
        (48, 128, 301), // only the odd-K phantom half
        (53, 128, 300), // only the ragged M
        (48, 71, 300),  // only the ragged N
        (48, 128, 300), // nothing ragged - the control
    ];
    // Exercised is not covered: assert the list really does contain each
    // padding kind, so a future edit to the shapes cannot quietly leave the
    // sweep testing nothing. `kc == K` here (the VNNI path has no k-panel
    // loop), so the phantom half exists exactly when K is odd.
    assert!(
        shapes.iter().any(|&(m, _, _)| m % VNNI_MR != 0),
        "no shape has a ragged M"
    );
    assert!(
        shapes.iter().any(|&(_, n, _)| n % VNNI_NR != 0),
        "no shape has a ragged N"
    );
    assert!(
        shapes.iter().any(|&(_, _, k)| k % 2 == 1),
        "no shape has the phantom k-half - the ONLY padding this test can \
         actually catch (see the table above)"
    );
    assert!(
        shapes
            .iter()
            .any(|&(m, n, k)| m % VNNI_MR == 0 && n % VNNI_NR == 0 && k % 2 == 0),
        "no un-ragged control shape"
    );

    for (m, n, k) in shapes {
        let run = Command::new(&exe)
            .args([m.to_string(), n.to_string(), k.to_string()])
            .env("Y_NUM_THREADS", "4")
            .output()
            .expect("run");
        let text = String::from_utf8_lossy(&run.stdout).into_owned();
        assert!(
            run.status.success() && text.contains("DONE"),
            "M={m} N={n} K={k} did not finish:\n{text}"
        );
        let wrong: i64 = text
            .lines()
            .find_map(|l| l.strip_prefix("wrong "))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or_else(|| panic!("no `wrong` line in:\n{text}"));
        assert_eq!(
            wrong, 0,
            "M={m} N={n} K={k}: {wrong} elements disagree with the reference. The \
             operand padding here holds live-range values rather than zeros, so a \
             packer that fails to mask an out-of-range row, column or k-half \
             feeds them to `vpdpwssd` - which is exactly what \
             `padded_product_is_the_live_dot_product` rules out."
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
