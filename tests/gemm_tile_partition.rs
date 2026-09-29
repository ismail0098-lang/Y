//! The GPU warp tiling's precondition, which was checked by nothing that ships.
//!
//! Every tensor-core GEMM here derives its warp geometry by TRUNCATING
//! integer division — `per_warp = cta / warps`, `num = per_warp / frag` — so a
//! warp's base advances by `per_warp` while a warp WRITES `num * frag`. Those
//! agree exactly when `frag * warps` divides `cta`. When they do not the
//! emitted kernel is well-formed, `ptxas` accepts it, and the rows in between
//! keep whatever the output buffer held.
//!
//! That precondition was three `debug_assert_eq!`s at each of three emitters.
//! `Cargo.toml` declares no `[profile.release]`, so rustc's default
//! `debug-assertions = false` applies and **none of them is in the shipping
//! binary** — the same gates-nothing shape as `@require`.
//!
//! Measured before this file was written, on `tests/gemm_f16_1024.ysu` with
//! `Y_CTA_OVERRIDE=96,128,32,4,2,3`: stride 24, `num = 1`, so each CTA
//! advances 96 rows and writes 64 — gaps at rows 16-23, 40-47, 64-71, 88-95 —
//! under "Compilation Successful!", exit 0, and `ptxas` exit 0.
//!
//! `proofs/GpuWarpTiling.v` states the partition and refutes that instance;
//! this file is the tie between the two, plus the census that says which
//! producers can reach the refused state.
//!
//! ## Which of the three sites is actually reachable
//!
//! Measured, not assumed:
//!
//! * **F16 tensor-core GEMM** — reachable two ways, both demonstrated below:
//!   `Y_CTA_OVERRIDE`, and a persisted `AUTOTUNE_*` line in `.ysu_hw_profile`.
//!   This is where the live defect was.
//! * **Fused GEMM+SwiGLU** — `Y_SWIGLU_TILE` **already validated** this exact
//!   constraint and rejected with a notice. So the same rule was written
//!   correctly at one override site and not the other. The guard added at the
//!   emitter still covers the autotuned and analytic-default paths, which the
//!   env-var check does not.
//! * **FP8 GEMM** — its tile comes from compile-time constants only (no
//!   autotuner, no override), so its guard is **DEFENSIVE AND UNREACHABLE**.
//!   Recorded rather than claimed as coverage. Its fragment shape is m16n8k32,
//!   not m16n16k16, which is why `validate_warp_tiling` takes the fragment
//!   extents as parameters rather than hardcoding 16.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("Y")
}

/// Per-test tag in the SIGNATURE, not in a comment asking the next author to
/// remember: this race has fired six times in this repository.
fn workdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("y_tilepart_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    // Compile a COPY: --emit-ptx writes next to its source, so compiling in
    // tests/ would rewrite committed artifacts and race any other binary
    // doing the same.
    std::fs::copy("tests/gemm_f16_1024.ysu", d.join("g.ysu")).unwrap();
    if let Ok(p) = std::fs::read_to_string(".ysu_hw_profile") {
        std::fs::write(d.join(".ysu_hw_profile"), p).unwrap();
    }
    d
}

struct Run {
    ok: bool,
    err: String,
    wrote_ptx: bool,
}

fn compile(dir: &PathBuf, src: &str, env: &[(&str, &str)]) -> Run {
    let ptx = dir.join(src.replace(".ysu", ".ptx"));
    let _ = std::fs::remove_file(&ptx);
    let mut c = Command::new(bin());
    c.arg(src).arg("--emit-ptx").current_dir(dir).env("Y_NO_CERTIFICATE", "1");
    for (k, v) in env {
        c.env(k, v);
    }
    let o = c.output().expect("run Y");
    Run {
        ok: o.status.success(),
        err: format!(
            "{}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        ),
        wrote_ptx: ptx.exists(),
    }
}

/// The live defect, through the override. Refusing is the fix, not a stopgap:
/// the alternative is a kernel with holes in it under a success banner.
#[test]
fn a_tile_that_does_not_tile_is_refused() {
    let d = workdir("override");
    let r = compile(&d, "g.ysu", &[("Y_CTA_OVERRIDE", "96,128,32,4,2,3")]);
    assert!(!r.ok, "a tile that does not tile still compiled:\n{}", r.err);
    assert!(
        !r.wrote_ptx,
        "the compile failed but a .ptx was written anyway — a refusal must \
         leave no artifact, or the next tool reads the broken kernel"
    );
    // The message has to name the constraint AND the repair. "does not tile"
    // alone sends the reader to look for a typo.
    for want in ["cta_m=96", "NEVER WRITTEN", "multiple of 64", "64 or 128"] {
        assert!(
            r.err.contains(want),
            "refusal does not name {:?}; a diagnosis that cannot be acted on \
             is half a refusal:\n{}",
            want,
            r.err
        );
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// The control, and it is what stops "refuse every tile" from passing every
/// other assertion in this file while deleting a working backend.
#[test]
fn a_legal_tile_still_compiles() {
    let d = workdir("legal");
    let r = compile(&d, "g.ysu", &[("Y_CTA_OVERRIDE", "64,128,32,4,2,3")]);
    assert!(r.ok, "a legal 64x128x32 4x2 tile was refused:\n{}", r.err);
    assert!(r.wrote_ptx, "legal tile compiled but wrote no .ptx");
    // And the default path — no override at all — must still work.
    let r2 = compile(&d, "g.ysu", &[]);
    assert!(r2.ok, "the autotuned default tile was refused:\n{}", r2.err);
    let _ = std::fs::remove_dir_all(&d);
}

/// The second live producer, and the one that needs no environment variable:
/// a persisted `AUTOTUNE_*` line is state on disk, written by `--autotune`,
/// keyed by GPU, and `parse_persisted_value` validates nothing but the field
/// count.
///
/// The first version of this probe was MIS-AIMED — it injected a line for a
/// shape with no cached entry, so the analytic model answered and the tile
/// never reached the emitter. It targets a shape the cache actually holds.
#[test]
fn a_corrupted_autotune_cache_line_is_refused() {
    let d = workdir("cache");
    let profile = d.join(".ysu_hw_profile");
    let Ok(text) = std::fs::read_to_string(&profile) else {
        eprintln!("SKIP: no .ysu_hw_profile — the cached-tile path was not demonstrated.");
        return;
    };
    // Find a real cached F16 entry and corrupt only its cta_m.
    let Some(line) = text.lines().find(|l| l.starts_with("AUTOTUNE_F16_")) else {
        eprintln!("SKIP: no cached F16 autotune line — the cached path was not demonstrated.");
        return;
    };
    let (key, val) = line.split_once('=').unwrap();
    let shape: Vec<u32> = key
        .trim_start_matches("AUTOTUNE_F16_")
        .split('_')
        .next()
        .unwrap()
        .split('x')
        .filter_map(|d| d.parse().ok())
        .collect();
    assert_eq!(shape.len(), 3, "unrecognised autotune key {key}");
    let parts: Vec<&str> = val.split(',').collect();
    assert_eq!(parts.len(), 6, "unrecognised autotune value {val}");

    // A source at exactly that shape, so the cached line is the one consulted.
    let src = format!(
        "@tile({}, {}, {})\nkernel gemm_probe(A: GlobalMemory<F16>, B: GlobalMemory<F16>, \
         C: GlobalMemory<F32>) {{\n}}\n\nfn main() {{\n}}\n",
        shape[0], shape[1], shape[2]
    );
    std::fs::write(d.join("probe.ysu"), &src).unwrap();

    // Control first: the untouched cache line must compile, or the refusal
    // below could be about anything.
    let good = compile(&d, "probe.ysu", &[]);
    assert!(good.ok, "the cached tile itself was refused:\n{}", good.err);

    // Now make cta_m indivisible by frag*warps_m, changing nothing else.
    let warps_m: u32 = parts[3].parse().unwrap();
    let bad_m = parts[0].parse::<u32>().unwrap() + 16 * warps_m / 2;
    assert_ne!(bad_m % (16 * warps_m), 0, "constructed tile is still legal");
    let corrupted = text.replace(
        line,
        &format!("{key}={bad_m},{},{},{},{},{}", parts[1], parts[2], parts[3], parts[4], parts[5]),
    );
    std::fs::write(&profile, corrupted).unwrap();

    let r = compile(&d, "probe.ysu", &[]);
    assert!(
        !r.ok && !r.wrote_ptx,
        "a corrupted cached tile compiled — this path needs no environment \
         variable, only a stale file:\n{}",
        r.err
    );
    assert!(r.err.contains("NEVER WRITTEN"), "wrong refusal:\n{}", r.err);
    let _ = std::fs::remove_dir_all(&d);
}

/// The 23 built-in candidates all satisfy the constraint — a CONFIRMATION,
/// which is why the default path was never wrong. The floor counts candidates
/// actually checked: a sweep that finds no candidates reports "none of them
/// violates" perfectly.
#[test]
fn every_builtin_candidate_tiles() {
    use y::autotuner::validate_warp_tiling;
    let src = std::fs::read_to_string("src/autotuner.rs").unwrap();
    let re = regex_lite_tuples(&src);
    let mut checked = 0usize;
    for (m, n, k, wm, wn) in re {
        if m > 1024 || wm > 32 || wm == 0 || wn == 0 {
            continue; // not a tile tuple
        }
        checked += 1;
        assert!(
            validate_warp_tiling(m, n, k, wm, wn, 16, 16, 16).is_ok(),
            "built-in candidate {m}x{n}x{k} warps {wm}x{wn} does not tile"
        );
    }
    assert!(
        checked >= 20,
        "only {checked} candidate tuples found in src/autotuner.rs — the sweep \
         is not reading the candidate list any more"
    );
}

/// Five-integer tuples out of the candidate tables, without pulling in a regex
/// crate for one scan.
fn regex_lite_tuples(src: &str) -> Vec<(u32, u32, u32, u32, u32)> {
    let mut out = Vec::new();
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'(' {
            let end = match src[i..].find(')') {
                Some(e) => i + e,
                None => break,
            };
            let inner = &src[i + 1..end];
            let nums: Vec<u32> = inner.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            if nums.len() == 5 && inner.chars().all(|c| c.is_ascii_digit() || c == ',' || c == ' ') {
                out.push((nums[0], nums[1], nums[2], nums[3], nums[4]));
            }
        }
        i += 1;
    }
    out
}

/// The tie: the proof's model of the emitter's truncating arithmetic must be
/// the arithmetic the emitter performs, and the counterexample the proof
/// refutes must be the tile that was actually compiled.
#[test]
fn the_proof_states_the_tile_that_was_actually_compiled() {
    let v = std::fs::read_to_string("proofs/GpuWarpTiling.v").unwrap();
    // The model.
    assert!(
        v.contains("Definition emit_stride (cta warps : nat) : nat := cta / warps.")
            && v.contains(
                "Definition emit_num (cta warps frag : nat) : nat := (cta / warps) / frag."
            ),
        "the proof no longer models the emitter's truncating derivation"
    );
    // The refuted instance, and its measured hole count.
    for want in ["emit_stride 96 4 = 24", "emit_num 96 4 16 = 1", "96 - 4 * (emit_num 96 4 16 * 16) = 32"] {
        assert!(v.contains(want), "the proof no longer states {want:?}");
    }
    // Rust agrees with the model on that instance.
    let (cta, warps, frag) = (96u32, 4u32, 16u32);
    let stride = cta / warps;
    let num = stride / frag;
    assert_eq!(stride, 24);
    assert_eq!(num, 1);
    assert_eq!(warps * num * frag, 64, "written extent");
    assert_eq!(cta - warps * num * frag, 32, "rows never written");
    // And the guard refuses exactly that instance, and accepts the legal one.
    use y::autotuner::validate_warp_tiling;
    assert!(validate_warp_tiling(96, 128, 32, 4, 2, 16, 16, 16).is_err());
    assert!(validate_warp_tiling(64, 128, 32, 4, 2, 16, 16, 16).is_ok());
    // The fragment extents are parameters because FP8 is m16n8k32: a tile
    // legal for F16 is not necessarily legal for FP8. Hardcoding 16 would
    // pass this.
    assert!(
        validate_warp_tiling(128, 128, 16, 4, 2, 16, 8, 32).is_err(),
        "cta_k=16 is not a multiple of FP8's k32 fragment and must be refused"
    );
    assert!(validate_warp_tiling(128, 128, 64, 4, 2, 16, 8, 32).is_ok());
}

/// The FP8 emitter must pass ITS OWN fragment shape, and this can only be
/// checked at the source.
///
/// Found by mutation, as a real hole in the test above: swapping the FP8 call
/// site from `16, 8, 32` to `16, 16, 16` leaves **all nine suites green**. Two
/// reasons compound, and neither is fixable by testing harder.
///
/// 1. The FP8 site is unreachable — its tile comes from compile-time
///    constants (`FP8_GEMM_CTA_*`), with no autotuner and no override — so
///    nothing it does is observable from any input.
/// 2. Both shipped FP8 tiles satisfy BOTH fragment shapes anyway. Checked:
///    `128x128x64` warps 4x2 and `64x64x64` warps 2x2 are each accepted under
///    m16n8k32 and under m16n16k16. So even if the site were reachable, the
///    wrong shape would still accept.
///
/// The test above asserts the parameterisation MATTERS by calling
/// `validate_warp_tiling` directly with each shape. That is true and says
/// nothing about which shape the emitter passes. Same device as pinning the
/// absence of an environment override: where behaviour cannot reach, pin the
/// source.
#[test]
fn each_emitter_passes_its_own_fragment_shape() {
    let src = std::fs::read_to_string("src/ptx_emitter.rs").unwrap();
    let calls: Vec<&str> = src
        .match_indices("validate_warp_tiling(")
        .map(|(i, _)| {
            let rest = &src[i..];
            &rest[..rest.find(')').unwrap_or(rest.len())]
        })
        .collect();
    assert_eq!(
        calls.len(),
        3,
        "expected exactly three guarded GEMM emitters (F16, FP8, SwiGLU), found {}: {:#?}",
        calls.len(),
        calls
    );
    // m16n8k32 exactly once — the FP8 kernel. `mma.sync.aligned.m16n8k32` is
    // what it emits, and its `num_j = per_warp_n / 8` / `k_substeps = cta_k /
    // 32` are what depend on it.
    let fp8 = calls.iter().filter(|c| c.contains("16, 8, 32")).count();
    assert_eq!(
        fp8, 1,
        "the FP8 GEMM must pass its own m16n8k32 fragment shape, not the F16 \
         one — its `num_j` divides by 8 and its `k_substeps` by 32. Found {} \
         call sites passing 16, 8, 32 among:\n{:#?}",
        fp8, calls
    );
    // m16n16k16 exactly twice — the F16 GEMM and the fused GEMM+SwiGLU.
    let f16 = calls.iter().filter(|c| c.contains("16, 16, 16")).count();
    assert_eq!(
        f16, 2,
        "expected the F16 and SwiGLU GEMMs to pass m16n16k16; found {} among:\n{:#?}",
        f16, calls
    );
    // Non-vacuity: the scan must be reading real call sites, not comments.
    assert!(
        calls.iter().all(|c| c.contains("cta_m, cta_n, cta_k, warps_m, warps_n")),
        "a validate_warp_tiling call site no longer passes the tile it guards:\n{:#?}",
        calls
    );
}
