//! Will this PTX load on a card that is not the one it was built on?
//!
//! `ptxas` can target any architecture it knows, so **this whole question is
//! answerable locally without owning the hardware** - which is the only reason
//! it is answerable at all here.
//!
//! ## The failure this exists to prevent
//!
//! A `.target` ABOVE the running device is a hard load failure:
//!
//! ```text
//! ptxas fatal : SM version specified by .target is higher than default SM
//!               version assumed
//! ```
//!
//! and at runtime, `CUDA_ERROR_NO_BINARY_FOR_GPU`. It is not a slowdown or a
//! wrong answer - the kernel simply does not run, on a machine the author never
//! tested. PTX is FORWARD compatible (sm_80 runs on Ada and Blackwell) and
//! never backward, so the correct target is the LOWEST architecture a kernel's
//! instructions actually require, not the architecture that happened to be
//! plugged in when it was generated.
//!
//! ## What was found when this was written (2026-08-23)
//!
//! - **`.ysu_hw_profile` was COMMITTED**, carrying `SM_VERSION=8.9` and
//!   `GPU_NAME=RTX 4070 Ti SUPER`. `check_or_probe_hardware` skips the probe
//!   whenever that file exists and never compares it against the live device,
//!   so a fresh clone on any other card emitted sm_89 and failed. Now
//!   gitignored.
//! - **`crates/y-gpu` ships five sm_89 kernels via `include_str!`** - MSM, NTT,
//!   permute, field multiply, vector subtract. The entire GPU ZK library was
//!   dead on every Ampere card. All five needed nothing above sm_80.
//! - `exact_attention` and `zero_drift` hardcoded `.target sm_89` in a format
//!   string. Both needed nothing above sm_80.
//!
//! **93 of the repo's 97 committed `.ptx` files assembled at sm_80 unchanged.**
//! Only the four FP8 GEMMs genuinely require sm_89, and correctly so: `e4m3`
//! tensor cores are Ada and later.
//!
//! Run with:  cargo test --test ptx_portability

use std::path::{Path, PathBuf};
use std::process::Command;

/// The floor. Every kernel that does not *need* something newer must load here.
///
/// sm_80 is Ampere (A100, 3060, 3090). Chosen because it is the oldest
/// architecture this project's instruction mix (cp.async, mma.sync, redux)
/// actually supports - not as a guess.
const FLOOR: &str = "sm_80";

/// Architectures a kernel is expected to load on, oldest first. sm_86 is the
/// 3060, sm_89 the 4070 Ti SUPER this was developed on, sm_120 Blackwell.
const ARCHES: [&str; 4] = ["sm_80", "sm_86", "sm_89", "sm_120"];

/// The PTX ISA version shipped artifacts declare.
///
/// `.version` is a DRIVER requirement and is independent of `.target`: 8.4
/// needs CUDA 12.4 or newer, 7.0 needs 11.0. A module can therefore be
/// perfectly portable across architectures and still refuse to load on a
/// machine with an older driver, which is the same failure wearing a different
/// hat. The shipped kernels were 8.4 and needed nothing above 7.0.
const VERSION_FLOOR: &str = "7.0";

fn declared_version(ptx: &str) -> Option<String> {
    ptx.lines()
        .find(|l| l.trim_start().starts_with(".version"))
        .map(|l| l.trim().trim_start_matches(".version").trim().to_string())
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ptxas() -> Option<PathBuf> {
    for c in ["ptxas", "/opt/cuda/bin/ptxas", "/usr/local/cuda/bin/ptxas"] {
        if Command::new(c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            return Some(PathBuf::from(c));
        }
    }
    None
}

/// Assemble `ptx` for `arch`; `Ok(())` or the first line of ptxas's complaint.
fn assembles(tool: &Path, ptx: &str, arch: &str, tag: &str) -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!(
        "y_portability_{}_{}",
        std::process::id(),
        tag.replace(['/', '.'], "_")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("k.ptx");
    std::fs::write(&src, ptx).unwrap();

    let out = Command::new(tool)
        .arg(format!("-arch={}", arch))
        .arg(&src)
        .arg("-o")
        .arg(dir.join("k.cubin"))
        .output()
        .expect("run ptxas");
    let _ = std::fs::remove_dir_all(&dir);

    if out.status.success() {
        Ok(())
    } else {
        let e = String::from_utf8_lossy(&out.stderr);
        Err(e.lines().find(|l| !l.trim().is_empty()).unwrap_or("(no message)").to_string())
    }
}

/// The declared `.target` of a PTX module.
fn declared_target(ptx: &str) -> Option<String> {
    ptx.lines()
        .find(|l| l.trim_start().starts_with(".target"))
        .map(|l| l.trim().trim_start_matches(".target").trim().to_string())
}

// ────────────────────────────────────────────────────────
// Kernels shipped inside the binary
// ────────────────────────────────────────────────────────

/// `crates/y-gpu` compiles these into the binary with `include_str!`, so a
/// wrong `.target` is not a build error anywhere - it is a library that loads
/// nothing, on someone else's machine.
#[test]
fn every_shipped_kernel_loads_on_every_supported_card() {
    let Some(tool) = ptxas() else {
        eprintln!("SKIP: no ptxas");
        return;
    };
    let dir = repo().join("crates/y-gpu/ptx");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {}", dir.display(), e))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "ptx").unwrap_or(false))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no shipped .ptx found - this test would pass vacuously");

    let mut bad = Vec::new();
    for f in &files {
        let ptx = std::fs::read_to_string(f).unwrap();
        let name = f.file_name().unwrap().to_string_lossy().to_string();

        // The declared target IS the load requirement, so check it directly:
        // ptxas would happily assemble an sm_80 module at sm_120 and tell us
        // nothing about whether the shipped file names a floor a 3060 can meet.
        match declared_target(&ptx) {
            Some(t) if t == FLOOR => {}
            Some(t) => bad.push(format!(
                "  {} declares .target {} - above the {} floor, so it cannot load on anything older",
                name, t, FLOOR
            )),
            None => bad.push(format!("  {} declares no .target at all", name)),
        }
        // Independent of the target: `.version` gates the DRIVER.
        match declared_version(&ptx) {
            Some(v) if v == VERSION_FLOOR => {}
            Some(v) => bad.push(format!(
                "  {} declares .version {} - above the {} floor, so it needs a newer CUDA \
                 driver than it has any reason to",
                name, v, VERSION_FLOOR
            )),
            None => bad.push(format!("  {} declares no .version at all", name)),
        }
        for arch in ARCHES {
            if let Err(e) = assembles(&tool, &ptx, arch, &name) {
                bad.push(format!("  {} at {}: {}", name, arch, e));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "shipped kernels are not portable ({} files checked):\n{}",
        files.len(),
        bad.join("\n")
    );
}

// ────────────────────────────────────────────────────────
// Kernels generated at run time
// ────────────────────────────────────────────────────────

/// The exact-attention kernels, which are the whole deterministic-decode path.
#[test]
fn exact_attention_loads_on_every_supported_card() {
    let Some(tool) = ptxas() else {
        eprintln!("SKIP: no ptxas");
        return;
    };
    let mut bad = Vec::new();
    let mut checked = 0;
    for (head_dim, seq) in [(64usize, 128usize), (64, 256), (128, 128)] {
        let ptx = match y::exact_attention::attention_ptx(head_dim, seq) {
            Ok(p) => p,
            // A refusal is fine and is not a portability problem - it is the
            // compiler declining a shape, loudly, which is the intended
            // behaviour everywhere else in this repo.
            Err(_) => continue,
        };
        checked += 1;
        let tag = format!("attn_{}_{}", head_dim, seq);
        if declared_target(&ptx).as_deref() != Some(FLOOR) {
            bad.push(format!("  {} declares {:?}, want {}", tag, declared_target(&ptx), FLOOR));
        }
        for arch in ARCHES {
            if let Err(e) = assembles(&tool, &ptx, arch, &tag) {
                bad.push(format!("  {} at {}: {}", tag, arch, e));
            }
        }
    }
    assert!(checked > 0, "no attention shape was accepted - the test would pass vacuously");
    assert!(bad.is_empty(), "exact attention is not portable:\n{}", bad.join("\n"));
}

/// The `@ZeroDrift` accumulate probe, one kernel per representation.
///
/// This one runs on the user's card to MEASURE it, so targeting the local
/// architecture is the tempting thing to do and is exactly wrong: it is JIT'd
/// for whatever card is present either way, and naming a high floor only makes
/// it unloadable on a low one.
#[test]
fn the_drift_probe_loads_on_every_supported_card() {
    let Some(tool) = ptxas() else {
        eprintln!("SKIP: no ptxas");
        return;
    };
    use y::zero_drift::DriftRepr::*;
    let mut bad = Vec::new();
    for repr in [FixedQ16_16, FixedQ32_32, Int64, Float64] {
        let ptx = y::zero_drift::accumulate_probe_ptx(repr, 8);
        let tag = format!("drift_{:?}", repr);
        if declared_target(&ptx).as_deref() != Some(FLOOR) {
            bad.push(format!("  {} declares {:?}, want {}", tag, declared_target(&ptx), FLOOR));
        }
        for arch in ARCHES {
            if let Err(e) = assembles(&tool, &ptx, arch, &tag) {
                bad.push(format!("  {} at {}: {}", tag, arch, e));
            }
        }
    }
    assert!(bad.is_empty(), "the drift probe is not portable:\n{}", bad.join("\n"));
}

// ────────────────────────────────────────────────────────
// The control, and the source-level guard
// ────────────────────────────────────────────────────────

/// The gate must be able to FAIL. A portability check that passes everything is
/// the shape this repo keeps finding: `ordinary_loop_bodies_still_verify`,
/// `the_corpus_is_not_all_skips`, the quantization liveness canary.
#[test]
fn the_gate_actually_rejects_an_unportable_kernel() {
    let Some(tool) = ptxas() else {
        eprintln!("SKIP: no ptxas");
        return;
    };
    let ptx = ".version 8.4\n.target sm_89\n.address_size 64\n\
               .visible .entry k() { ret; }\n";
    assert!(
        assembles(&tool, ptx, "sm_86", "control").is_err(),
        "an sm_89 module assembled at sm_86 - the gate cannot detect the bug it exists for"
    );
    assert!(
        assembles(&tool, ptx, "sm_89", "control").is_ok(),
        "the control module does not assemble at its own target, so it is testing nothing"
    );
}

/// No source file may hardcode a `.target` above the floor.
///
/// Both of the generator bugs found here were a literal `.target sm_89` inside
/// a Rust format string, which no amount of assembling the CURRENT output can
/// prevent from coming back. FP8 is the one legitimate exception and is named,
/// not pattern-matched, so adding a second exception has to be deliberate.
#[test]
fn no_source_file_hardcodes_a_target_above_the_floor() {
    let src = repo().join("src");
    let mut bad = Vec::new();
    for e in std::fs::read_dir(&src).unwrap().filter_map(|e| e.ok()) {
        let p = e.path();
        if p.extension().map(|x| x != "rs").unwrap_or(true) {
            continue;
        }
        let text = std::fs::read_to_string(&p).unwrap_or_default();
        for (lineno, line) in text.lines().enumerate() {
            // Only literal EMISSION above the floor. Three things are not
            // that, and each was hit on the first run:
            //   - prose,
            //   - a target BELOW the floor (`ysu_gpu_probe` emits sm_75, which
            //     is strictly more portable, not less),
            //   - an `assert!` checking what the emitter produced, which is a
            //     check and not an emission.
            let l = line.trim();
            if l.starts_with("//") || l.starts_with("///") || l.starts_with('*') {
                continue;
            }
            if l.contains("assert") || l.contains(".contains(") {
                continue;
            }
            if let Some(sm) = line.split(".target sm_").nth(1) {
                let n: u32 = sm
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0);
                let floor_n: u32 = FLOOR.trim_start_matches("sm_").parse().unwrap();
                if n > floor_n {
                    bad.push(format!(
                        "  {}:{}: {}",
                        p.file_name().unwrap().to_string_lossy(),
                        lineno + 1,
                        l.chars().take(90).collect::<String>()
                    ));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "a `.target` above {} is hardcoded in source. PTX is forward compatible, so \
         emit the LOWEST architecture the instructions require - naming a higher one \
         makes the kernel unloadable on every older card:\n{}",
        FLOOR,
        bad.join("\n")
    );
}

// ────────────────────────────────────────────────────────
// The one legitimate exception
// ────────────────────────────────────────────────────────

/// FP8 genuinely needs sm_89, so it must REFUSE below it, not emit.
///
/// This is the only kernel family in the repo whose hardware requirement is
/// above the floor, and it is a real one: `e4m3` tensor cores are Ada and
/// later. Below that the instruction does not exist, so the module is rejected
/// at LOAD time with `CUDA_ERROR_NO_BINARY_FOR_GPU` and nothing says why -
/// exactly the surprise this file exists to prevent, in the one case that
/// cannot be fixed by lowering the target.
///
/// Asserted as a BICONDITIONAL. "Refuse FP8 always" satisfies half of it and
/// would delete a working path on the hardware that has it, so the sm_89 arm
/// is what makes the sm_86 arm mean anything.
#[test]
fn fp8_refuses_below_ada_and_still_works_on_it() {
    let src = repo().join("tests/gemm_fp8_256.ysu");
    if !src.exists() {
        eprintln!("SKIP: no FP8 fixture");
        return;
    }
    // A profile is what fixes the target, so writing one is how a card is
    // simulated. Everything runs in a temp dir; the real profile is untouched.
    for (cc, want_refusal) in [("8.6", true), ("8.9", false)] {
        let dir = std::env::temp_dir()
            .join(format!("y_fp8_arch_{}_{}", std::process::id(), cc.replace('.', "")));
        std::fs::create_dir_all(&dir).unwrap();
        let real = repo().join(".ysu_hw_profile");
        let mut profile = std::fs::read_to_string(&real).unwrap_or_default();
        if profile.is_empty() {
            eprintln!("SKIP: no .ysu_hw_profile to base the simulated card on");
            return;
        }
        profile = profile
            .lines()
            .map(|l| {
                if l.starts_with("SM_VERSION=") {
                    format!("SM_VERSION={}", cc)
                } else if l.starts_with("COMPUTE_CAPABILITY=") {
                    format!("COMPUTE_CAPABILITY={}", cc)
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.join(".ysu_hw_profile"), profile).unwrap();
        let local_src = dir.join("gemm_fp8_256.ysu");
        std::fs::copy(&src, &local_src).unwrap();

        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&local_src)
            .arg("--emit-ptx")
            .current_dir(&dir)
            .output()
            .expect("run Y");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let _ = std::fs::remove_dir_all(&dir);

        if want_refusal {
            assert!(
                !out.status.success(),
                "at sm_{} an FP8 kernel was ACCEPTED. The emitted module cannot load on \
                 that card, and the failure would surface as CUDA_ERROR_NO_BINARY_FOR_GPU \
                 at run time on someone else's machine:\n{}",
                cc.replace('.', ""),
                text
            );
            assert!(
                text.contains("FP8") && text.contains("sm_89"),
                "refused, but not by the FP8 arch check - so this case is not gating what \
                 it is named for:\n{}",
                text
            );
        } else {
            assert!(
                out.status.success(),
                "at sm_{} FP8 must still compile - refusing everywhere is sound and \
                 useless:\n{}",
                cc.replace('.', ""),
                text
            );
        }
    }
}

// ────────────────────────────────────────────────────────
// Portability that is not about the GPU at all
// ────────────────────────────────────────────────────────

/// The compiler must work from a directory that is not its own source tree.
///
/// The LLVM backend passed `c_src/runtime.c` to clang as a bare CWD-relative
/// path, so it linked only when Y was invoked from inside the repo. Anywhere
/// else clang reported `no such file or directory: 'c_src/runtime.c'`, which
/// reads as a broken install rather than a wrong working directory - and is
/// exactly the "works on the author's machine" shape this file exists for.
///
/// `-lX11` was also linked unconditionally, so any headless machine without
/// libX11 could not build a Y program at all. It is needed only by the
/// optional GUI surface and is now dropped when the library is absent.
#[test]
fn the_llvm_backend_works_from_a_foreign_directory() {
    if Command::new("clang").arg("--version").output().map(|o| !o.status.success()).unwrap_or(true) {
        eprintln!("SKIP: no clang");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_foreign_cwd_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Carry the hardware profile so this does not trigger a GPU probe.
    let prof = repo().join(".ysu_hw_profile");
    if prof.exists() {
        let _ = std::fs::copy(&prof, dir.join(".ysu_hw_profile"));
    }
    let src = dir.join("foreign.ysu");
    std::fs::write(&src, "fn main() -> I32 { let a: I32 = 9; let b: I32 = 2; return a - b; }\n")
        .unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success(),
        "compiling from a directory outside the repo failed:\n{}",
        text
    );

    // ...and the binary it produced must actually run. "clang exited 0" is not
    // the claim; a linked executable that computes 9 - 2 is.
    let bin = dir.join("foreign");
    assert!(bin.exists(), "no binary was produced:\n{}", text);
    let run = Command::new(&bin).output().expect("run the produced binary");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        run.status.code(),
        Some(7),
        "the produced binary returned the wrong value"
    );
}
