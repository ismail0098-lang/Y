//! The output tiling: the model against the theorem, and the kernel against a
//! PADDED C.
//!
//! `proofs/ExactGemmTiling.v` proves that the driver's `MR x NR` tiles name
//! every element of the live `M x N` rectangle exactly once, and nothing
//! outside it. `cpu_gemm::mn_tiles` is the Rust transcription; the first half
//! of this file checks the transcription against the theorem over a finite
//! range, exactly as `exact_gemm_ksplit_model.rs` does for the K axis.
//!
//! **The second half is the part that could not be written for the K axis.** A
//! correct K-split is invisible in the answer - that is what `ksplit_exact`
//! says - so the model had to predict a thread count instead. An output tiling
//! is different: a tile that runs past the ragged tail writes MEMORY IT DOES
//! NOT OWN, and that is directly observable if the caller gives C a row stride
//! larger than N and poisons the space between rows.
//!
//! Nothing in this repo has ever called the exact kernel with `ldc != N`, and
//! §1 of `docs/proof_carrying_kernels.md` names precisely this class:
//!
//! > Twelve address computations in the CPU GEMM were correct only because
//! > `lda == K` made stride and extent the same number.
//!
//! So all three strides are made distinct from their extents here.

use std::path::PathBuf;
use std::process::Command;

use y::cpu_gemm::{mn_tiles, VNNI_MR, VNNI_NR};

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

/// `tiles_cover`, finitely: contiguous from 0, no gap, no overlap, ending at
/// the extent - and every tile at most one wide.
#[test]
fn the_output_tiles_cover_the_axis_exactly() {
    let mut ragged = 0usize;
    let mut checked = 0usize;
    for tile in 1..=VNNI_NR {
        for extent in 0..=1500usize {
            let tiles = mn_tiles(extent, tile);
            let mut off = 0usize;
            for (i, (o, w)) in tiles.iter().enumerate() {
                assert_eq!(*o, off, "extent={extent} tile={tile}: tile {i} starts wrong");
                assert!(*w > 0, "extent={extent} tile={tile}: tile {i} is empty");
                assert!(*w <= tile, "extent={extent} tile={tile}: tile {i} is too wide");
                off += w;
            }
            assert_eq!(
                off, extent,
                "extent={extent} tile={tile}: the tiles end at {off}. Anything short \
                 leaves elements of C never written; anything long is a write past \
                 the end."
            );
            if extent % tile != 0 {
                ragged += 1;
            }
            checked += 1;
        }
    }
    assert!(checked > 50_000, "only {checked} cases");
    assert!(
        ragged > 20_000,
        "only {ragged} of {checked} had a ragged tail; the clamp is the whole \
         point and a sweep over exact multiples cannot see it"
    );
}

/// `tile_index_injective` + `tile_index_surjective`, finitely: the (tile,
/// offset) pairs are a BIJECTION onto `[0, extent)`.
///
/// This is stronger than the coverage test above and is the actual obligation.
/// A tiling that writes element 3 twice and element 4 never has exactly the
/// right total width.
#[test]
fn every_position_is_named_exactly_once() {
    for tile in [1usize, 2, 5, VNNI_MR, VNNI_NR, 7, 64] {
        for extent in [0usize, 1, 5, 6, 7, 53, 64, 65, 71, 128, 383, 384] {
            let mut seen = vec![0u32; extent];
            for (o, w) in mn_tiles(extent, tile) {
                for f in 0..w {
                    let p = o + f;
                    assert!(
                        p < extent,
                        "extent={extent} tile={tile}: position {p} is outside the \
                         axis - this is the unclamped-tail bug, and in the kernel it \
                         is a write past the end of C rather than a wrong number"
                    );
                    seen[p] += 1;
                }
            }
            for (p, n) in seen.iter().enumerate() {
                assert_eq!(
                    *n, 1,
                    "extent={extent} tile={tile}: position {p} was named {n} times, \
                     not once"
                );
            }
        }
    }
}

/// The mutation the proof refutes, re-run here: an unclamped tail.
///
/// Without it, the two tests above are not obviously testing anything - they
/// would pass against a tiling with no ragged case to get wrong.
#[test]
fn the_unclamped_tail_would_write_past_the_end() {
    // The proof's concrete numbers: MR = 6, extent = 53, tile 8 starts at 48
    // and a full-width tile would reach 54.
    let flat_width = VNNI_MR;
    let last_off = 8 * VNNI_MR;
    assert_eq!(last_off, 48);
    assert!(
        last_off + flat_width > 53,
        "the unclamped tail is supposed to overrun"
    );
    assert_eq!(
        mn_tiles(53, VNNI_MR).last().copied(),
        Some((48, 5)),
        "the shipped tiling clamps the last tile to 5"
    );
}

/// The model and the emitter must AGREE on the tile constants, rather than this
/// file keeping a third copy of them.
#[test]
fn the_model_and_the_driver_agree_on_the_tile_shape() {
    let module = y::cpu_gemm::emit_vnni_gemm_module(64);

    // The two tiling loops this file and proofs/ExactGemmTiling.v transcribe.
    for label in ["vg.i.cond", "vg.j.cond"] {
        assert!(
            module.contains(label),
            "the emitted driver has no `{label}` loop; `mn_tiles` and \
             proofs/ExactGemmTiling.v are transcriptions of the `vg.i` / `vg.j` \
             nest and would be describing a tiling the code no longer has"
        );
    }

    // The clamp itself, which is the whole subject of the proof: a tile width
    // is `select (rem < T) rem T`. Matched on the real instruction rather than
    // on a guess - the first version of this test looked for `llvm.smin`, which
    // the IR builder does not emit.
    let has_clamp = |w: usize| {
        module.lines().any(|l| {
            l.contains("select i1") && l.trim_end().ends_with(&format!("i64 {w}"))
        })
    };
    assert!(
        has_clamp(VNNI_MR),
        "no row-tile clamp against MR = {VNNI_MR} in the emitted driver. An \
         unclamped tail writes past the last row of C - not a wrong number, a \
         write into memory the caller owns."
    );
    assert!(
        has_clamp(VNNI_NR),
        "no column-tile clamp against NR = {VNNI_NR} in the emitted driver."
    );
}

// ── The behavioural tie: a PADDED C ──────────────────────────────────

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

/// Calls the threaded entry directly so the strides can be set independently of
/// the extents. The recognised nest always passes `lda = K`, `ldb = ldc = N`,
/// which is exactly the coincidence that hid twelve address bugs before.
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

/* 0xAB in every byte: a value the kernel cannot produce by accident, and one
   that is obviously wrong if it survives into the live region. */
static const int64_t POISON = (int64_t)0xABABABABABABABABULL;

int main(int argc, char **argv) {
    int M = 53, N = 71, K = 301;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    /* Every stride is strictly larger than its extent, so no address
       computation can be right by the stride == extent coincidence. */
    long lda = K + 5, ldb = N + 9, ldc = N + 7;

    int16_t *A = calloc((size_t)M * lda, 2);
    int16_t *B = calloc((size_t)K * ldb, 2);
    int64_t *C = malloc((size_t)M * ldc * 8);
    int64_t *R = malloc((size_t)M * N * 8);

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

    /* Poison EVERYTHING, live region included, and do not zero it here.
       The wrapper owns that zeroing - "C is ASSIGNED by the nest this replaces,
       while the kernel accumulates INTO it". Pre-zeroing the live region from
       the driver masks exactly half the bug this test exists for: a flat
       `memset(C, 0, M*N*8)` leaves the LAST rows' live cells unzeroed when
       ldc > N, and an accumulate onto poison is a wrong answer, not a crash. */
    for (long i = 0; i < (long)M * ldc; ++i) C[i] = POISON;

    __y_gemm_exact_vnni_threaded(A, B, C, M, N, K, lda, ldb, ldc);

    long wrong = 0, clobbered = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j)
            if (C[(long)i*ldc + j] != R[(long)i*N + j]) wrong++;
        for (long j = N; j < ldc; ++j)
            if (C[(long)i*ldc + j] != POISON) clobbered++;
    }
    printf("wrong %ld\n", wrong);
    printf("clobbered %ld\n", clobbered);
    printf("DONE\n");
    return 0;
}
"#;

/// A tile that runs past the ragged tail writes memory it does not own, and
/// that is exactly what a padded C exposes.
///
/// Shapes are chosen so BOTH axes are ragged against their tile: `M % 6 != 0`
/// and `N % 64 != 0`. A shape that divides evenly has no tail to get wrong -
/// `feedback-exercised-is-not-covered`.
#[test]
fn the_kernel_writes_only_the_live_rectangle() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_exact_tile_{}", std::process::id()));
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
    let ll = dir.join("m.ll");
    let driver = dir.join("drv.c");
    std::fs::write(&driver, DRIVER).expect("write driver");
    let exe = dir.join("run");
    let cc = Command::new("clang")
        .args([
            ll.to_str().unwrap(),
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

    // Both axes ragged, both axes exact, and one of each - so a bug confined to
    // a single axis cannot hide behind the other.
    let shapes: [(usize, usize, usize); 5] = [
        (53, 71, 301),   // both ragged
        (48, 71, 301),   // M exact, N ragged
        (53, 128, 301),  // M ragged, N exact
        (48, 128, 301),  // both exact - the case that proves nothing on its own
        (1, 1, 4099),    // the smallest possible tail on both axes
    ];
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
        let field = |name: &str| -> i64 {
            text.lines()
                .find_map(|l| l.strip_prefix(name))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or_else(|| panic!("no `{name}` line in:\n{text}"))
        };
        assert_eq!(
            field("wrong "),
            0,
            "M={m} N={n} K={k}: the live region disagrees with the integer \
             reference. Strides here are lda=K+5, ldb=N+9, ldc=N+7, so an address \
             computation that assumed stride == extent fails only under this test."
        );
        assert_eq!(
            field("clobbered "),
            0,
            "M={m} N={n} K={k}: the kernel wrote into C's row padding. A tile ran \
             past the ragged tail - `tile_index_in_range` in \
             proofs/ExactGemmTiling.v is the property that just failed, and in a \
             caller's buffer that is memory it does not own."
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
