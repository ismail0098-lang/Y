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
use std::sync::atomic::{AtomicUsize, Ordering};

/// Two tests calling the same helper for the same arch would otherwise share a
/// directory and `remove_dir_all` each other's output mid-run - the `.ptx` race
/// this repo has now hit in four files, hit again while adding the gate below.
static SALT: AtomicUsize = AtomicUsize::new(0);

/// The floor. Every kernel that does not *need* something newer must load here.
///
/// sm_80 is Ampere (A100, 3060, 3090). Chosen because it is the oldest
/// architecture this project's instruction mix (cp.async, mma.sync, redux)
/// actually supports - not as a guess.
const FLOOR: &str = "sm_80";

/// Architectures a kernel is expected to load on, oldest first. sm_86 is the
/// 3060, sm_89 the 4070 Ti SUPER this was developed on, sm_90 the H100/H200,
/// sm_100 datacenter Blackwell (B100/B200), sm_120 consumer Blackwell (RTX 50).
///
/// **sm_100 and sm_120 are different architectures, not two names for
/// Blackwell.** Measured with ptxas 13.3: `tcgen05` assembles at sm_100a and is
/// refused at sm_120a, and `wgmma` is refused at both (it is Hopper-only and
/// Blackwell dropped it). Listing only sm_120 therefore left the datacenter
/// part of the line unchecked, which is the same hole sm_90 was in - and for
/// the same reason, that nobody here can spot-check it by running something.
///
/// **sm_90 was missing, and it is the one arch here nobody can spot-check by
/// running something.** The list covered two consumer Ampere parts, the
/// development machine, and consumer Blackwell - and skipped the datacenter
/// architecture entirely. That is the arch where "does it load?" is least
/// answerable by hand, because no one working on this repo has an H100; it is
/// therefore the one that most needs the gate. All 67 committed artifacts
/// already assemble there, so this asserts a property the compiler had rather
/// than demanding a new one - which is precisely why it should have been
/// asserted before somebody changed it.
///
/// Note this says nothing about Hopper-SPECIFIC instructions. TMA and WGMMA
/// are not in this backend (see gotcha #8: the surface that claimed to be was
/// deleted for never having assembled). What is checked is that the ordinary
/// `mma.sync` / `cp.async` instruction mix Y really emits loads on an H100.
const ARCHES: [&str; 6] = ["sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"];

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

// ────────────────────────────────────────────────────────
// The DRIVER floor, which is a separate axis from the arch
// ────────────────────────────────────────────────────────

/// The `.version` the EMITTER picks per architecture must be the real floor.
///
/// A module can be perfectly portable across cards and still refuse to load on
/// a machine whose *driver* is older, with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
/// `ptx_version_for_sm` used to declare **8.4 on sm_89** -- CUDA 12.4 -- for
/// every kernel, because FP8 `mma.sync` needs it on that arch. Everything else
/// needed 7.8 (CUDA 11.8): a whole major version of driver, for nothing. sm_86
/// was 7.5 against a real 7.1.
///
/// **The floor is MEASURED here, not written down**, by bisecting `.version`
/// under `ptxas -arch=<a>`, and the emitter's own choice is then compared to
/// it. The first version of this test hardcoded the expected numbers in a
/// table and so verified the table rather than the compiler -- reverting sm_86
/// to 7.5 left it green. Found by mutation.
/// Archs whose DOCUMENTED PTX ISA floor is above what this assembler measures.
///
/// `ptxas` 13.3 accepts `.version 7.8` at `-arch=sm_90`, but sm_90 shipped in
/// CUDA 12.0, whose ISA is 8.0 - there is no 11.x driver with sm_90 support to
/// be compatible with, so 7.8 buys nothing and claims something the spec does
/// not. `ptx_version_for_sm` keeps 8.0 deliberately and this is where that is
/// written down.
const SPEC_FLOOR_ABOVE_MEASURED: [(&str, &str, &str); 1] = [(
    "sm_90",
    "8.0",
    "sm_90 arrived in CUDA 12.0 (ISA 8.0). This assembler accepting 7.8 at that \
     target is leniency, not a contract, and no driver old enough to care about \
     the difference supports sm_90 at all.",
)];

#[test]
fn the_emitter_declares_the_real_driver_floor() {
    let Some(tool) = ptxas() else {
        eprintln!("ptxas not found; skipping");
        return;
    };
    // Ascending, so the first that assembles is the floor.
    const CANDIDATES: [&str; 12] =
        ["6.0", "6.2", "6.3", "6.5", "7.0", "7.1", "7.2", "7.3", "7.5", "7.8", "8.0", "8.7"];
    let body = ".address_size 64\n.visible .entry k(){ .reg .b32 %r<2>; mov.u32 %r0, 1; ret; }\n";

    for arch in ARCHES {
        let floor = CANDIDATES
            .iter()
            .find(|v| {
                let ptx = format!(".version {}\n.target {}\n{}", v, arch, body);
                assembles(&tool, &ptx, arch, &format!("bisect_{}_{}", arch, v)).is_ok()
            })
            .unwrap_or_else(|| panic!("{}: no candidate .version assembles at all", arch));

        let declared = emitted_version_for(arch);

        // A DOCUMENTED exception, kept narrow on purpose. Where the arch is
        // newer than the ISA version this assembler will accept for it, the
        // measured floor is leniency rather than a contract, and the emitter
        // deliberately declares the SPEC floor instead. Guess down, but not
        // below the spec.
        //
        // Written as a table of named archs, not as a blanket `declared >=
        // measured`: the whole reason this test exists is that four archs were
        // over-stating their floor, and a `>=` comparison passes all four.
        if let Some((_, spec, why)) = SPEC_FLOOR_ABOVE_MEASURED
            .iter()
            .find(|(a, _, _)| *a == arch)
        {
            assert_eq!(
                declared.as_str(),
                *spec,
                "{arch}: the emitter declares .version {declared}, but this arch's \
                 documented floor is {spec}. {why}"
            );
            assert!(
                CANDIDATES.iter().position(|v| v == floor)
                    <= CANDIDATES.iter().position(|v| v == spec),
                "{arch}: the MEASURED floor {floor} is now above the documented \
                 {spec}, so the exception is stale - this assembler no longer \
                 accepts what the table assumes it does."
            );
            continue;
        }

        assert_eq!(
            &declared.as_str(),
            floor,
            "{}: the emitter declares .version {} but the measured floor is {}. \
             Over-stating it makes every kernel need a newer CUDA driver than it uses; \
             under-stating it makes the module fail to assemble.",
            arch,
            declared,
            floor
        );
    }
}

/// Compile a trivial kernel with a hardware profile naming `arch`, and return
/// the `.version` the emitter chose.
///
/// Driving the real binary is what makes the caller a test OF THE EMITTER
/// rather than of a constant repeated in the test file.
/// The `.target` the emitter declares for a probed arch.
///
/// Deliberately separate from `emitted_version_for`, which asserts the target
/// matches as a sanity check on its own probe - an assertion inside a helper is
/// not a test, and the sm_90a promotion is exactly what it failed to report as
/// a finding.
fn emitted_target_for(arch: &str) -> String {
    let ptx = emitted_module_for(arch);
    declared_target(&ptx).unwrap_or_default()
}

fn emitted_module_for(arch: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "y_ptxver_{}_{}_{}",
        std::process::id(),
        arch,
        SALT.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // `PtxEmitter::new_with_profile` builds its target as
    // `sm_` + sm_version with the dot stripped, so "8.6" gives sm_86.
    let sm = arch.trim_start_matches("sm_");
    let dotted = format!("{}.{}", &sm[..sm.len() - 1], &sm[sm.len() - 1..]);
    std::fs::write(
        dir.join(".ysu_hw_profile"),
        format!("SM_VERSION={}\nGPU_NAME=TestCard\nSM_COUNT=66\n", dotted),
    )
    .unwrap();

    let src = dir.join("plain.ysu");
    std::fs::write(&src, "kernel plain_add() {\n    let a: u32 = 1;\n    let b: u32 = a + 2;\n}\n")
        .unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let ptx = std::fs::read_to_string(dir.join("plain.ptx"))
        .unwrap_or_else(|_| String::from_utf8_lossy(&out.stdout).to_string());
    let _ = std::fs::remove_dir_all(&dir);
    ptx
}

fn emitted_version_for(arch: &str) -> String {
    let ptx = emitted_module_for(arch);
    let target = declared_target(&ptx).unwrap_or_default();
    assert_eq!(
        target, arch,
        "the profile asked for {} but the emitter targeted {} -- the probe is not \
         measuring the arch it thinks it is",
        arch, target
    );
    declared_version(&ptx)
        .unwrap_or_else(|| panic!("no .version line in the emitted PTX:\n{}", ptx))
}

/// A kernel with no FP8 in it must not inherit FP8's driver requirement.
///
/// The pair is the point. sm_89's floor is 7.8, and FP8 `mma.sync` needs 8.4 on
/// the very same arch -- so the requirement belongs to the INSTRUCTION, not to
/// the architecture, and `require_ptx_version` raises the module's floor only
/// when the instruction is actually emitted.
#[test]
fn only_the_kernel_that_needs_a_newer_isa_declares_one() {
    assert_eq!(
        emitted_version_for("sm_89"),
        "7.8",
        "a kernel with no FP8 in it declared a version above sm_89's own floor of 7.8 \
         (CUDA 11.8). 8.4 is CUDA 12.4 and is FP8 `mma.sync`'s requirement, not the \
         architecture's."
    );
}

// ────────────────────────────────────────────────────────
// The co-processor backend builds its own module header
// ────────────────────────────────────────────────────────

/// Emit `--emit-coprocessor` under a crafted profile and return `(version, ptx)`.
fn coprocessor_module_for(arch: &str) -> (String, String) {
    let dir = std::env::temp_dir().join(format!(
        "y_copver_{}_{}_{}",
        std::process::id(),
        arch,
        SALT.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let sm = arch.trim_start_matches("sm_");
    let dotted = format!("{}.{}", &sm[..sm.len() - 1], &sm[sm.len() - 1..]);
    std::fs::write(
        dir.join(".ysu_hw_profile"),
        format!("SM_VERSION={}\nGPU_NAME=TestCard\nSM_COUNT=66\n", dotted),
    )
    .unwrap();

    let fixture = repo().join("tests").join("coprocessor_attention.ysu");
    let src = dir.join("case.ysu");
    std::fs::copy(&fixture, &src).expect("the co-processor fixture must exist");

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-coprocessor")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "--emit-coprocessor failed at {}:\n{}{}",
        arch,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let ptx = std::fs::read_to_string(dir.join("case.coprocessor.ptx"))
        .expect("no .coprocessor.ptx was written");
    let target = declared_target(&ptx).unwrap_or_default();
    assert_eq!(
        target, arch,
        "the profile asked for {} but the co-processor backend targeted {}",
        arch, target
    );
    let v = declared_version(&ptx)
        .unwrap_or_else(|| panic!("no .version line in the co-processor PTX:\n{}", ptx));
    let _ = std::fs::remove_dir_all(&dir);
    (v, ptx)
}

/// `--emit-coprocessor` must declare the same driver floor as `--emit-ptx`.
///
/// It does not go through `emit_program`, so it wrote its own header - and the
/// header hardcoded `.version 8.0`. That one literal is wrong in BOTH
/// directions at once:
///
///   - **over-stated** on sm_80/sm_86/sm_89, whose floors are 7.0/7.1/7.8, so
///     a co-processor kernel refuses to load on a driver that is merely older
///     (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`) - invisible to any assemble-only
///     gate, because `ptxas` is happy to assemble an over-stated version;
///   - **under-stated** on Blackwell, where `ptxas` rejects the module outright
///     ("PTX .version 8.0 does not support .target sm_120") while this backend
///     printed "Dual-accelerator PTX generated successfully!" and exited 0.
///
/// The assertion is an AGREEMENT between the two producers rather than a second
/// copy of the table: a third site that re-derives the floor is the bug, not
/// the fix. The existing `coprocessor_ptx_assembles` cannot see any of this -
/// it hardcodes `-arch=sm_89`, which is exactly the arch where 8.0 is legal.
#[test]
fn the_coprocessor_backend_declares_the_same_floor_as_the_ptx_backend() {
    for arch in ARCHES {
        let (cop, _) = coprocessor_module_for(arch);
        let main = emitted_version_for(arch);
        assert_eq!(
            cop, main,
            "at {} the co-processor backend declared .version {} while the PTX backend \
             declared {}. Both are the same driver requirement for the same card, so a \
             disagreement means one of them is not consulting `ptx_version_for_sm`.",
            arch, cop, main
        );
    }
}

/// ...and the module it writes must assemble at the target it names.
#[test]
fn the_coprocessor_module_assembles_at_every_supported_target() {
    let Some(tool) = ptxas() else {
        eprintln!("skipping: no ptxas on PATH");
        return;
    };
    for arch in ARCHES {
        let (_, ptx) = coprocessor_module_for(arch);
        if let Err(e) = assembles(&tool, &ptx, arch, "coprocessor") {
            panic!(
                "the co-processor backend wrote a module for {} that ptxas rejects at \
                 that very target - and reported success and exit 0 while doing it:\n{}",
                arch, e
            );
        }
    }
}

/// No source file may hardcode a `.version` above the floor either.
///
/// Gotcha 8b's standing lesson is that a literal in a Rust format string cannot
/// be prevented from coming back by assembling the CURRENT output. That was
/// written about `.target`, a gate was added for `.target`, and the identical
/// bug was sitting one line above it in `.version` form the whole time.
///
/// `ptx_version_for_sm`'s own body is the table and is exempt by position, not
/// by pattern - a second exemption has to be deliberate. Everything after a
/// `#[cfg(test)]` is not shipped and is not scanned.
#[test]
fn no_source_file_hardcodes_a_ptx_version_above_the_floor() {
    let src = repo().join("src");
    let floor: (u32, u32) = {
        let mut it = VERSION_FLOOR.split('.');
        (
            it.next().unwrap().parse().unwrap(),
            it.next().unwrap().parse().unwrap(),
        )
    };
    let mut bad = Vec::new();
    for e in std::fs::read_dir(&src).unwrap().filter_map(|e| e.ok()) {
        let p = e.path();
        if p.extension().map(|x| x != "rs").unwrap_or(true) {
            continue;
        }
        let text = std::fs::read_to_string(&p).unwrap_or_default();

        // The table itself, exempted by position.
        let mut exempt: Option<(usize, usize)> = None;
        if let Some(start) = text.find("pub fn ptx_version_for_sm(") {
            let before = text[..start].lines().count();
            let rest = &text[start..];
            let end = rest.find("\n}\n").map(|o| rest[..o].lines().count()).unwrap_or(0);
            exempt = Some((before, before + end + 1));
        }

        for (lineno, line) in text.lines().enumerate() {
            if line.trim_start().starts_with("#[cfg(test)]") {
                break;
            }
            if let Some((a, b)) = exempt {
                if lineno >= a && lineno <= b {
                    continue;
                }
            }
            let l = line.trim();
            if l.starts_with("//") || l.starts_with("///") || l.starts_with('*') {
                continue;
            }
            if l.contains("assert") || l.contains(".contains(") {
                continue;
            }
            let Some(rest) = line.split(".version ").nth(1) else {
                continue;
            };
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let mut it = num.split('.');
            let (Some(maj), Some(min)) = (
                it.next().and_then(|x| x.parse::<u32>().ok()),
                it.next().and_then(|x| x.parse::<u32>().ok()),
            ) else {
                continue;
            };
            if (maj, min) > floor {
                bad.push(format!(
                    "  {}:{}: {}",
                    p.file_name().unwrap().to_string_lossy(),
                    lineno + 1,
                    l.chars().take(90).collect::<String>()
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "a `.version` above {} is hardcoded in source. `.version` is the DRIVER \
         requirement: over-stating it makes the kernel refuse to load on an older \
         driver, and under-stating it makes ptxas reject the module on a newer card. \
         Call `ptx_version_for_sm` instead of writing a literal:\n{}",
        VERSION_FLOOR,
        bad.join("\n")
    );
}

/// No emitted `.target` may carry an architecture-SPECIFIC suffix.
///
/// The emitter promoted `sm_90` to `sm_90a` unconditionally and without a
/// stated reason - a leftover from the WGMMA/TMA surface that was deleted for
/// never having assembled. The `a` suffix is not "sm_90 plus extras": it is a
/// different, architecture-specific target that never JITs forward, so every
/// kernel compiled on an H100 was pinned to that exact card.
///
/// It went unnoticed for the most ordinary reason - `ARCHES` did not include
/// sm_90, so nothing ever asked the emitter what it would target there. This
/// test asks the question of every arch, and would catch the same promotion
/// applied to any future one.
#[test]
fn no_kernel_declares_an_architecture_specific_target() {
    for arch in ARCHES {
        let target = emitted_target_for(arch);
        assert_eq!(
            target, arch,
            "asked the emitter for {arch} and it declared `.target {target}`. A \
             target must be what was probed, not a promotion - PTX is forward \
             compatible and an arch-specific variant is not."
        );
        assert!(
            !target
                .trim_start_matches("sm_")
                .chars()
                .last()
                .is_some_and(|c| c.is_ascii_alphabetic()),
            "`.target {target}` carries an architecture-specific suffix. Nothing \
             this backend emits requires one; if a Hopper- or Blackwell-specific \
             instruction is ever added it must request the suffix the way FP8 \
             requests a `.version` - discovered during emission, on the module - \
             not by promoting every kernel on the chance that one needs it."
        );
    }
}

/// The MEASUREMENT the rule above rests on, so it is grounded rather than
/// believed.
///
/// A plain target JITs forward to every later architecture; the `a` variant
/// assembles at neither the later archs NOR its own plain one. Without this,
/// the test above is a style rule.
#[test]
fn an_architecture_specific_target_does_not_travel() {
    let Some(tool) = ptxas() else {
        eprintln!("ptxas not found; skipping");
        return;
    };
    let body = ".address_size 64\n.visible .entry k(){ ret; }\n";
    let plain = format!(".version 8.0\n.target sm_90\n{body}");
    let specific = format!(".version 8.0\n.target sm_90a\n{body}");

    // The control: without this the assertions below could pass because the
    // probe module is malformed for some unrelated reason.
    assert!(
        assembles(&tool, &plain, "sm_90", "travel_plain_90").is_ok(),
        "the plain sm_90 probe must assemble at its own arch, or this test is \
         measuring a broken module rather than the suffix"
    );

    for later in ["sm_100", "sm_120"] {
        if assembles(&tool, &plain, later, &format!("travel_plain_{later}")).is_err() {
            // This assembler may simply not know the arch; skip rather than
            // assert something about the local toolchain.
            eprintln!("skipping {later}: not supported by this ptxas");
            continue;
        }
        assert!(
            assembles(&tool, &specific, later, &format!("travel_spec_{later}")).is_err(),
            "`.target sm_90a` assembled at {later}. If that is now true the \
             suffix has stopped being architecture-specific and the rule in \
             `no_kernel_declares_an_architecture_specific_target` needs revisiting."
        );
    }
    assert!(
        assembles(&tool, &specific, "sm_90", "travel_spec_90").is_err(),
        "`.target sm_90a` assembled at plain -arch=sm_90; the suffix is supposed \
         to require an exactly-matching arch"
    );
}
