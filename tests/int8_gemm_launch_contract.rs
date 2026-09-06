//! The int8 tensor-core GEMM's launch contract, and the tie to
//! `proofs/Int8GemmSchedule.v`.
//!
//! ## The defect this file exists for, measured before the guard was written
//!
//! `emit_int8_gemm_kernel` gives one 16x8 output tile to one WARP and launches
//! a grid of `(N/8, M/16, splits)`. A CTA therefore has exactly 32 threads of
//! work however many it is launched with — and nothing checked. The emitted
//! kernel contained ONE predicate, the K-loop bound, and never mentioned
//! `%ntid.x`.
//!
//! At M=64 N=32 K=128 with every element of A = 3 and of B = 5, so every
//! element of C must be exactly K*15 = 1920:
//!
//! | block | result |
//! |---|---|
//! | (32,1,1) | correct |
//! | (64,1,1) | 1344 of 2048 wrong, `C[256] = 3840` |
//! | (128,1,1) | same, and it reads row 79 of a 64-row A |
//!
//! 3840 is EXACTLY DOUBLE, which is the mechanism: warp 1's lane index gives
//! `g = tid/4` in 8..15 instead of 0..7, so its two A-row reads land at
//! `cy*16+g` and `cy*16+g+8` — the second of which is the NEXT tile's rows —
//! and `red.global.add.s32` sums that second product into the same output.
//!
//! That matters more here than in an ordinary kernel: this kernel's entire
//! advertised claim is a bit-identical answer at every launch geometry
//! (`tests/gpu_batch_invariance.rs`), and a wrong block size falsified it
//! silently.
//!
//! ## What each test covers, and what it cannot
//!
//! The device test needs a CUDA driver and SKIPS without one, so it cannot be
//! the only cover — this repo's own recorded failure mode. The two source-level
//! tests run everywhere: one asserts the emitted PTX actually contains a
//! warp-uniform guard branching past the whole body, the other ties the
//! proof's constants to the emitter's shape refusal. Between them a machine
//! with no GPU still fails if the guard is removed.

use std::path::Path;
use std::process::Command;

/// Compile a fixture at `(m, n, k)` in a per-test temp directory and return
/// its PTX. The tag is in the SIGNATURE rather than a comment asking the next
/// author to remember: this helper materialises files in a temp dir and that
/// race has fired six times in this repository.
fn emit(tag: &str, m: usize, n: usize, k: usize) -> Result<String, String> {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    let dir = std::env::temp_dir().join(format!("i8lc_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    // `--emit-ptx` writes next to its input, so compile a COPY: a gate that
    // emits must never rewrite the committed artifacts it is checking.
    if let Ok(p) = std::fs::read(repo.join(".ysu_hw_profile")) {
        let _ = std::fs::write(dir.join(".ysu_hw_profile"), p);
    }
    let src = dir.join("lc.ysu");
    std::fs::write(
        &src,
        format!(
            "@tile({}, {}, {})\n\
             kernel int8_gemm(A: GlobalMemory<I8>, B: GlobalMemory<I8>, C: GlobalMemory<I32>) {{\n}}\n\
             fn main() {{}}\n",
            m, n, k
        ),
    )
    .unwrap();
    let out = Command::new(bin.join("Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let r = if out.status.success() {
        Ok(std::fs::read_to_string(dir.join("lc.ptx")).expect("no .ptx"))
    } else {
        Err(format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    };
    let _ = std::fs::remove_dir_all(&dir);
    r
}

/// The behavioural half: the answer must not depend on the block size.
///
/// The assertion is that every geometry is **correct**, not merely that the
/// geometries agree — a kernel that writes nothing agrees with itself
/// perfectly, and this is exactly the shape where that null metric passes.
#[test]
fn the_answer_does_not_depend_on_the_block_size() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!(
            "SKIP: no CUDA driver. The two source-level tests in this file still \
             cover the guard's presence and its tie to the proof."
        );
        return;
    };
    let (m, n, k) = (64usize, 32usize, 128usize);
    let ptx = emit("dev", m, n, k).expect("the fixture must compile");
    let module = ctx.load_ptx(&ptx, "int8_gemm").expect("PTX failed to load");

    let d_a = ctx.alloc(m * k).unwrap();
    let d_b = ctx.alloc(n * k).unwrap();
    let d_c = ctx.alloc(m * n * 4).unwrap();
    // Every element 3 and 5, so every element of C is exactly k*15. A uniform
    // fill is what makes a doubled contribution read as exactly 2x rather than
    // as noise that has to be diffed against a reference.
    ctx.memset_u8(&d_a, 3).unwrap();
    ctx.memset_u8(&d_b, 5).unwrap();
    let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()];
    let want = (k * 15) as i32;

    for (bx, gz) in [(32u32, 1u32), (64, 1), (128, 1), (256, 1), (32, 4), (64, 4), (128, 3)] {
        // Zero rather than poison: the kernel REDUCES into C with
        // `red.global.add.s32`, so C is an accumulator. A kernel that writes
        // nothing still fails, because `want` is non-zero.
        ctx.memset_u8(&d_c, 0).unwrap();
        ctx.launch(&module, ((n / 8) as u32, (m / 16) as u32, gz), (bx, 1, 1), 0, &args)
            .unwrap_or_else(|e| panic!("launch failed at block {} gridz {}: {:?}", bx, gz, e));
        ctx.synchronize()
            .unwrap_or_else(|e| panic!("kernel faulted at block {} gridz {}: {:?}", bx, gz, e));
        let mut raw = vec![0u8; m * n * 4];
        ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
        let mut bad = 0usize;
        let mut first = (0usize, 0i32);
        for i in 0..m * n {
            let v = i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
            if v != want {
                if bad == 0 {
                    first = (i, v);
                }
                bad += 1;
            }
        }
        assert_eq!(
            bad, 0,
            "block ({},1,1) gridz {}: {} of {} elements wrong; C[{}] = {}, want {}. \
             Before the warp-0 guard existed a 64-thread block gave exactly {} here \
             (double), because warp 1's g = tid/4 lands in 8..15 and its second A-row \
             read is the NEXT tile's rows.",
            bx, gz, bad, m * n, first.0, first.1, want, want * 2
        );
    }
}

/// The structural half, which runs with no GPU: the guard must be IN the
/// emitted kernel, must be warp-uniform, and must branch past the body.
///
/// A device test alone would report `ok` on a machine with no driver while the
/// guard was deleted.
#[test]
fn the_emitted_kernel_carries_a_warp_uniform_guard() {
    let ptx = emit("guard", 64, 32, 128).expect("the fixture must compile");
    let body: Vec<&str> = ptx.lines().map(|l| l.trim()).collect();

    // `%tid.x` must be compared against 31 (i.e. lane < 32), NOT against
    // `%ntid.x` or a runtime value: the predicate has to be constant across a
    // warp or `mma.sync.aligned` sees a divergent warp.
    let cmp = body
        .iter()
        .position(|l| l.starts_with("setp.gt.u32") && l.contains(", 31;"))
        .unwrap_or_else(|| {
            panic!(
                "no `setp.gt.u32 %p, %tid, 31` in the emitted kernel. Without it a \
                 block larger than 32 threads double-counts through \
                 `red.global.add.s32`.\n{}",
                ptx
            )
        });
    let tid_reg = body[cmp]
        .split(", ")
        .nth(1)
        .expect("malformed setp")
        .to_string();
    // The register compared must be the one `%tid.x` was moved into.
    assert!(
        body.iter()
            .any(|l| l.starts_with("mov.u32") && l.contains(&tid_reg) && l.contains("%tid.x")),
        "the guard compares {} against 31, but that register was never loaded from \
         %tid.x. A guard on the wrong register is not a guard.\n{}",
        tid_reg, ptx
    );
    assert!(
        !body[cmp].contains("%ntid"),
        "the guard reads %ntid, which is not warp-uniform in general; \
         `mma.sync.aligned` needs a predicate constant across the warp"
    );

    // The branch must jump past the mma and past the reduction, or the guard
    // skips nothing that matters.
    let br = body
        .iter()
        .skip(cmp)
        .position(|l| l.starts_with('@') && l.contains("bra"))
        .map(|i| i + cmp)
        .expect("the guard's predicate is computed and never branched on");
    let label = body[br]
        .rsplit(' ')
        .next()
        .unwrap()
        .trim_end_matches(';')
        .to_string();
    let target = body
        .iter()
        .position(|l| l.starts_with(&label) && l.ends_with(':'))
        .unwrap_or_else(|| panic!("the guard branches to `{}`, which is not defined", label));
    let skipped = &body[br..target];
    for must in ["mma.sync", "red.global.add.s32"] {
        assert!(
            skipped.iter().any(|l| l.contains(must)),
            "the guard branches past `{}` but that region contains no `{}` - \
             it is skipping nothing that could double-count",
            label, must
        );
    }
}

/// The tie: the proof's shape constants must be the emitter's, and the
/// emitter's shape refusal must be the covering condition the proof states.
///
/// Without this the theorems are about an arithmetic nobody performs.
#[test]
fn the_proof_states_the_shape_the_emitter_enforces() {
    let v = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("proofs/Int8GemmSchedule.v"),
    )
    .expect("proofs/Int8GemmSchedule.v");
    for (name, val) in [("MMA_M", 16), ("MMA_N", 8), ("MMA_K", 32)] {
        assert!(
            v.contains(&format!("Definition {} : nat := {}.", name, val)),
            "the proof must state {} = {}, the shape `mma.sync.m16n8k32` fixes",
            name, val
        );
    }
    // **The confinement theorem must stay TWO-SIDED.** Its lower bound alone
    // (`cy*16 <= row`) is true for every `g` whatsoever, so a proof weakened to
    // it still compiles, still reports "Closed under the global context", and
    // still satisfies `proofs_are_checked`'s content control - which asks that
    // the theorem NAME appear under a `Print Assumptions`, not that it say
    // anything. That mutation survived the first sweep of this file; it is the
    // `Theorem the_certificate_is_not_vacuous : True.` shape, and the guard
    // belongs on the STATEMENT'S TEXT.
    //
    // The upper bound is the whole content: it is what makes `g < 8` do work,
    // and it is exactly what the refutation below negates at `g = 8`.
    let confine = v
        .split("Theorem the_guard_is_what_confines_a_warp_to_its_own_tile")
        .nth(1)
        .expect("the confinement theorem must exist")
        .split("Proof.")
        .next()
        .expect("malformed theorem");
    assert!(
        confine.contains("< cy * MMA_M + MMA_M"),
        "the confinement theorem states no UPPER bound on the row a lane reads.          Its lower bound alone holds for every g, so it says nothing about what          the guard buys:\n{}",
        confine
    );
    let refute = v
        .split("Theorem without_the_guard_a_second_warp_lands_in_the_next_tile")
        .nth(1)
        .expect("the refutation must exist")
        .split("Proof.")
        .next()
        .expect("malformed theorem");
    assert!(
        refute.contains('~') && refute.contains("< MMA_M"),
        "the refutation must NEGATE the same bound the confinement theorem          asserts, or the pair is not a dual and either half can drift:\n{}",
        refute
    );

    // **The warp tile's decomposition must stay THREE-digit**, for the same
    // reason the confinement theorem must stay two-sided. `warp_row` is
    // `ty*(mt*16) + mi*16 + i`: drop the middle digit and it degenerates to the
    // single-mma tiling, which is still true, still compiles, and says nothing
    // about a warp that issues mt*nt of them. The middle term is what makes two
    // mma of ONE warp provably write different rows.
    let wr = v
        .split("Definition warp_row")
        .nth(1)
        .expect("the proof must define the warp tile's row map")
        .split('\n')
        .next()
        .expect("malformed definition");
    for term in ["mt * MMA_M", "mi * MMA_M"] {
        assert!(
            wr.contains(term),
            "warp_row must carry `{}` - without the per-mma digit it is the \
             single-tile map again and proves nothing about a warp tile:\n{}",
            term, wr
        );
    }
    // The tile guard's refutation must say the overrun is PAST THE MATRIX, not
    // merely a duplicate: that is what makes the guard load-bearing rather than
    // an optimisation, and it is the claim the grid sweep exercises.
    let overrun = v
        .split("Theorem without_the_tile_guard_a_cta_addresses_past_the_matrix")
        .nth(1)
        .expect("the tile-guard refutation must exist")
        .split("Proof.")
        .next()
        .expect("malformed theorem");
    assert!(
        overrun.contains("m <= ty * (mt * MMA_M)"),
        "the tile-guard refutation must conclude that the base row reaches or \
         passes M. A weaker statement would leave the guard looking like a \
         performance choice:\n{}",
        overrun
    );

    // The emitter refuses anything the tiling does not cover, and the proof
    // says so. Check both directions on the real compiler: one legal shape and
    // one shape short of a tile on each axis.
    assert!(emit("ok", 64, 32, 128).is_ok(), "a legal shape must compile");
    for (m, n, k, axis) in [(63usize, 32usize, 128usize, "M"), (64, 31, 128, "N"), (64, 32, 127, "K")] {
        let e = emit("bad", m, n, k)
            .expect_err(&format!("M={} N={} K={} does not tile and must be refused", m, n, k));
        assert!(
            e.contains("M % 16 == 0") || e.contains("m16n8k32"),
            "the refusal for a ragged {} must name the shape it needs, not fail obscurely:\n{}",
            axis, e
        );
    }
}

/// The WARP TILE, and why changing it needed the grid to stop mattering.
///
/// `int8_warp_tile` issues `mt*nt` mma per K step from the same A and B
/// fragments, which cuts global traffic by up to 6x and measured **4.21x** at
/// 4096^3 on an RTX 4070 Ti SUPER. The tile is a compile-time function of M and
/// N, so it is invisible at the call site — and a host that kept launching the
/// pre-tiling `(N/8, M/16, z)` grid would then start `mt*nt` times too many
/// CTAs, each writing a wider tile, with `red.global.add.s32` summing the
/// overlaps into a silently wrong answer.
///
/// That is the same class of defect as the block-size bug at the top of this
/// file, so it gets the same answer: the output tiles are grid-strided in x and
/// y, exactly as K already was in z. Every grid of at least (1,1,1) is now
/// correct, and the old launch is correct with idle CTAs.
fn cpu_reference(m: usize, n: usize, k: usize, a: &[i8], b: &[i8]) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0i32;
            for x in 0..k {
                s += a[i * k + x] as i32 * b[j * k + x] as i32;
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Shapes that select DIFFERENT warp tiles must all compute the same matrix.
///
/// The reference is a plain CPU triple loop, not another GPU kernel: the tile
/// decides which lane reads which element, so checking one tiling against
/// another would compare two readings of the same possibly-wrong layout.
#[test]
fn every_warp_tile_the_shape_selects_computes_the_same_matrix() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — warp tiling was not exercised on a device.");
        return;
    };
    let k = 128usize;
    // (M, N) chosen so the selector lands on a different (mt, nt) in each row,
    // and so several of them need MORE THAN ONE output tile — without that the
    // grid-stride loop below runs exactly once and is not tested at all.
    // The shapes must actually select DIFFERENT tiles under the floor rule, or
    // this compares one schedule against itself. A big tile is only taken when
    // it leaves >= 64 output tiles, so reaching 64x64 needs M*N >= 512*512, and
    // reaching an intermediate tile needs a shape long in one axis and short in
    // the other. Both directions are here on purpose.
    let shapes = [
        (16usize, 8usize),   // 1 tile at any size -> 16x8
        (48, 24),            // 9 tiles -> 16x8
        (512, 512),          // 64 tiles at 64x64 -> 64x64, exactly at the floor
        (1024, 1024),        // 256 tiles -> 64x64
        (4096, 16),          // N admits only nt=2 -> 64x16
        (32, 4096),          // M admits only mt=2 -> 32x64
    ];
    let mut seen = std::collections::BTreeSet::new();
    for (m, n) in shapes {
        let ptx = emit(&format!("tile{}x{}", m, n), m, n, k).expect("fixture must compile");
        let tile = ptx
            .lines()
            .find(|l| l.contains("warp tile "))
            .and_then(|l| l.split("warp tile ").nth(1))
            .and_then(|s| s.split_whitespace().next())
            .expect("the emitted kernel must declare its warp tile")
            .to_string();
        seen.insert(tile.clone());
        let module = ctx.load_ptx(&ptx, "int8_gemm").expect("PTX failed to load");

        let a: Vec<i8> = (0..m * k).map(|i| (((i * 37 + 11) % 251) as i32 - 125) as i8).collect();
        let b: Vec<i8> = (0..n * k).map(|i| (((i * 53 + 7) % 251) as i32 - 125) as i8).collect();
        let want = cpu_reference(m, n, k, &a, &b);

        let d_a = ctx.alloc(m * k).unwrap();
        let d_b = ctx.alloc(n * k).unwrap();
        let d_c = ctx.alloc(m * n * 4).unwrap();
        ctx.memcpy_htod_at(&d_a, 0, &a.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
        ctx.memcpy_htod_at(&d_b, 0, &b.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
        ctx.memset_u8(&d_c, 0).unwrap();
        let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()];
        ctx.launch(&module, ((n / 8) as u32, (m / 16) as u32, 1), (32, 1, 1), 0, &args)
            .unwrap_or_else(|e| panic!("launch failed at {}x{}: {:?}", m, n, e));
        ctx.synchronize().unwrap_or_else(|e| panic!("faulted at {}x{}: {:?}", m, n, e));
        let mut raw = vec![0u8; m * n * 4];
        ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
        for i in 0..m * n {
            let got = i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
            assert_eq!(
                got, want[i],
                "M={} N={} K={} warp tile {}: C[{}] = {}, want {}",
                m, n, k, tile, i, got, want[i]
            );
        }
    }
    // Non-vacuity: if the selector collapsed to one tile these shapes would all
    // exercise the same schedule and agreeing would mean nothing.
    assert!(
        seen.len() >= 4,
        "the shapes above must select at least three distinct warp tiles, else this \
         test compares one schedule against itself; saw {:?}",
        seen
    );
}

/// The answer must not depend on the GRID either — including the grid a host
/// written before warp tiling would still be launching.
#[test]
fn the_answer_does_not_depend_on_the_grid() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — grid invariance was not demonstrated.");
        return;
    };
    // 512x512 is the smallest shape that takes the 64x64 warp tile (64 output
    // tiles, exactly at the floor), so the grid sweep below exercises the tile
    // loop against a tile that is NOT the mma shape.
    let (m, n, k) = (512usize, 512usize, 128usize);
    let ptx = emit("grid", m, n, k).expect("fixture must compile");
    let module = ctx.load_ptx(&ptx, "int8_gemm").expect("PTX failed to load");
    let a: Vec<i8> = (0..m * k).map(|i| (((i * 29 + 3) % 251) as i32 - 125) as i8).collect();
    let b: Vec<i8> = (0..n * k).map(|i| (((i * 71 + 17) % 251) as i32 - 125) as i8).collect();
    let want = cpu_reference(m, n, k, &a, &b);

    let d_a = ctx.alloc(m * k).unwrap();
    let d_b = ctx.alloc(n * k).unwrap();
    let d_c = ctx.alloc(m * n * 4).unwrap();
    ctx.memcpy_htod_at(&d_a, 0, &a.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
    ctx.memcpy_htod_at(&d_b, 0, &b.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
    let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()];

    // (n/8, m/16) is the grid a host written before warp tiling launches: with
    // a 64x64 tile that is 8x more CTAs in x and 4x more in y than there are
    // tiles. Without the stride guard every one of them would write a full
    // tile and `red.global.add.s32` would sum 32 copies.
    for (gx, gy, gz) in [
        (2u32, 2u32, 1u32),                            // exactly the tiles
        ((n / 8) as u32, (m / 16) as u32, 1),          // the pre-tiling host grid
        (1, 1, 1),                                     // one CTA does everything
        (1, 2, 1),
        (2, 1, 1),
        (3, 5, 1),                                     // coprime with the tile counts
        (2, 2, 4),                                     // split K as well
        ((n / 8) as u32, (m / 16) as u32, 3),
    ] {
        ctx.memset_u8(&d_c, 0).unwrap();
        ctx.launch(&module, (gx, gy, gz), (32, 1, 1), 0, &args)
            .unwrap_or_else(|e| panic!("launch failed at grid ({},{},{}): {:?}", gx, gy, gz, e));
        ctx.synchronize()
            .unwrap_or_else(|e| panic!("faulted at grid ({},{},{}): {:?}", gx, gy, gz, e));
        let mut raw = vec![0u8; m * n * 4];
        ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
        let mut bad = 0usize;
        let mut first = (0usize, 0i32);
        for i in 0..m * n {
            let got = i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
            if got != want[i] {
                if bad == 0 {
                    first = (i, got);
                }
                bad += 1;
            }
        }
        assert_eq!(
            bad, 0,
            "grid ({},{},{}): {} of {} wrong; C[{}] = {}, want {}. An over-large grid \
             without the tile-stride guard writes each tile several times and \
             `red.global.add.s32` sums the copies.",
            gx, gy, gz, bad, m * n, first.0, first.1, want[first.0]
        );
    }
}

/// The tile is chosen from the shape ALONE, so the set of programs that compile
/// is exactly what it was before warp tiling.
///
/// This runs with no GPU. Without it, a selector that started refusing shapes
/// it used to accept would be caught only by whichever fixture happened to use
/// one.
#[test]
fn the_warp_tile_is_chosen_from_the_shape_and_narrows_nothing() {
    // Every M % 16 == 0 and N % 8 == 0 still compiles, tile or no tile.
    for (m, n) in [(16usize, 8usize), (48, 24), (80, 40), (16, 64), (64, 8), (128, 128)] {
        let ptx = emit(&format!("sel{}_{}", m, n), m, n, 128)
            .unwrap_or_else(|e| panic!("M={} N={} used to compile and must still:\n{}", m, n, e));
        let hdr = ptx
            .lines()
            .find(|l| l.contains("warp tile "))
            .expect("the kernel must declare its warp tile in the header");
        let tile = hdr.split("warp tile ").nth(1).unwrap().split_whitespace().next().unwrap();
        let (tm, tn) = tile.split_once('x').expect("tile must read <rows>x<cols>");
        let (tm, tn): (usize, usize) = (tm.parse().unwrap(), tn.parse().unwrap());
        assert_eq!(m % tm, 0, "M={} does not divide by its own warp tile {}", m, tile);
        assert_eq!(n % tn, 0, "N={} does not divide by its own warp tile {}", n, tile);
        assert!(tm % 16 == 0 && tn % 8 == 0, "warp tile {} is not a multiple of the mma shape", tile);
        // **The tile must be the largest the shape admits, UNLESS that would
        // leave too few tiles to fill the machine** — a tile is one warp, so a
        // 64x64 tile on a 64x64 matrix is one warp for the whole GEMM, and it
        // measures **0.14x** the schedule that shipped. Below the floor the
        // schedule falls back to one mma per warp outright; the intermediate
        // tiles do not win either (32x32 at 256x256 is 0.89x).
        let max_mt = [4usize, 2, 1].into_iter().find(|c| m % (16 * c) == 0).unwrap();
        let max_nt = [8usize, 4, 2, 1].into_iter().find(|c| n % (8 * c) == 0).unwrap();
        let want = if (m / (16 * max_mt)) * (n / (8 * max_nt)) >= 64 {
            (16 * max_mt, 8 * max_nt)
        } else {
            (16, 8)
        };
        assert_eq!(
            (tm, tn),
            want,
            "M={} N={} selected warp tile {} where the rule says {}x{} \
             ({} tiles at the maximal {}x{}, floor is 64)",
            m, n, tile, want.0, want.1,
            (m / (16 * max_mt)) * (n / (8 * max_nt)), 16 * max_mt, 8 * max_nt
        );
    }
}
