//! There must be ONE answer to "does this machine have AVX-512", and it must be
//! the one that asks whether the instruction will EXECUTE.
//!
//! There were three, and the wrong one fed the emitter:
//!
//! | site | input | XCR0 checked | consumer |
//! |---|---|---|---|
//! | `probe_cpu_features` (fresh probe) | CPUID.7.0:EBX[16] | no | `HardwareProfile::has_avx512` -> `attributes #0` |
//! | `check_or_probe_hardware` (cached) | **the profile file** | no probe at all | the same |
//! | `probe_cpu_hardware_profile` | CPUID.7.0:EBX[16] | no | `CpuShapeDispatcher` -> `--emit-cpu` regime |
//! | `host_has_avx512_vnni` | `is_x86_feature_detected!` | yes | the exact GEMM gate |
//!
//! A CPU reports a vector feature in CPUID whether or not the OS has enabled
//! the register state in `XCR0`. Under a hypervisor masking the state, or a
//! kernel booted without it, the CPUID bit stays set and the instruction faults
//! exactly as if the silicon lacked it. That is why the reading matters and not
//! merely the agreement.
//!
//! This is the `VnniExact::licenses` / CUDA driver-binding shape: a rule with
//! one written-down implementation, implemented differently a second time. The
//! repository's precedent is to assert the producers AGREE rather than to
//! re-derive a table, and that is what `the_producers_agree` does. It is not
//! sufficient on its own - on a machine where XCR0 has the state enabled every
//! reading agrees and a gate made only of agreement is silent - so
//! `the_authority_checks_the_register_state` pins the READING at the source,
//! the way `the_override_cannot_claim_hardware_the_machine_lacks` pins the
//! absence of an env var.
//!
//! What none of this can check is the machine where they disagree. That is a
//! stated limit: see `docs/proof_carrying_kernels.md`.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Does this machine have AVX-512, asked WITHOUT going through anything under
/// test?
///
/// `/proc/cpuinfo` is a third mechanism: the Linux kernel clears the AVX-512
/// capability bits when the XSAVE state is not available, so its flags already
/// answer "will it execute" rather than "is the silicon capable". Two readings
/// from different mechanisms can only agree by both being right.
///
/// A skip guard must not be computed by the function under test - that mistake
/// made the previous increment's biconditional skip itself under the exact
/// mutation it existed to catch.
fn machine_really_has_avx512() -> bool {
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        // `avx512f` is the foundation; every other avx512_* flag implies it.
        return info.split_whitespace().any(|w| w.trim_matches(',') == "avx512f");
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The tag is in the signature, not in a comment asking the caller to remember:
/// two tests sharing a temp-dir name is a race this repository has hit five
/// times.
///
/// `avx512_in_profile` is written into the child's `.ysu_hw_profile`, so the
/// test can ask what a STALE profile does without touching the repo's own.
fn emit_with_profile(tag: &str, avx512_in_profile: bool) -> (PathBuf, String) {
    let dir = std::env::temp_dir().join(format!("y_avx512_gate_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let base = std::fs::read_to_string(repo().join(".ysu_hw_profile"))
        .expect("the repo profile is the template");
    let doctored: String = base
        .lines()
        .map(|l| {
            if l.starts_with("AVX512=") {
                format!("AVX512={avx512_in_profile}")
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    // The floor counts the WORK, not the candidate: a version of this helper
    // that forgot to doctor the profile would make
    // `a_stale_profile_cannot_raise_the_answer` compare two identical runs and
    // pass while testing nothing.
    assert!(
        doctored.contains(&format!("AVX512={avx512_in_profile}")),
        "the profile was not doctored, so this proves nothing"
    );
    std::fs::write(dir.join(".ysu_hw_profile"), doctored).expect("write profile");

    // A trivial host program: this gate is about the module PRELUDE, which
    // every emitted module carries whatever is in it.
    let src = dir.join("probe.ysu");
    std::fs::write(&src, "fn main() {\n    return;\n}\n").expect("write source");

    let out = Command::new(repo().join("target/release/Y"))
        .arg(&src)
        .arg("--emit-llvm")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "Y refused a trivial program: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ll = std::fs::read_to_string(dir.join("probe.ll")).expect("emitted .ll");
    let attrs = ll
        .lines()
        .find(|l| l.starts_with("attributes #0"))
        .unwrap_or("<no attributes #0>")
        .to_string();
    (dir, attrs)
}

/// Every producer of the answer returns the same thing.
///
/// This is the assertion the CUDA driver-binding gate makes: the two bindings
/// must AGREE rather than each re-deriving a table. Here there were four
/// producers and one authority.
#[test]
fn the_producers_agree() {
    let authority = y::sentinel::host_has_avx512();

    assert_eq!(
        authority,
        y::sentinel::check_or_probe_hardware().has_avx512,
        "`HardwareProfile::has_avx512` disagrees with the authority. This field \
         selects `attributes #0`, so a wrong-high answer puts AVX-512 into \
         every function of every emitted module."
    );

    assert_eq!(
        authority,
        y::sentinel::probe_cpu_hardware_profile().supports_avx512_masking,
        "`CpuHardwareProfile::supports_avx512_masking` disagrees with the \
         authority. It selects the `--emit-cpu` kernel regime."
    );

    assert_eq!(
        if authority { 16 } else { 8 },
        y::sentinel::probe_cpu_hardware_profile().simd_vector_width_floats,
        "the SIMD width and the masking flag are two spellings of one fact and \
         must not be able to disagree"
    );

    // The assertion above is VACUOUS on a machine that has AVX-512: both
    // readings are 16 whether or not the width is derived from the flag.
    // Mutation confirmed it - decoupling them (`let simd_w = 16;`) is a no-op
    // here and survived the whole sweep. So the coupling is pinned at the
    // source as well, which is the same move as pinning the READING there:
    // a property that cannot fail on this machine needs a check that does not
    // run on it.
    let src = std::fs::read_to_string(repo().join("src/sentinel.rs")).expect("sentinel.rs");
    assert!(
        src.contains("let simd_w = if has_avx512 { 16 } else { 8 };"),
        "`probe_cpu_hardware_profile` must derive the SIMD width from the \
         AVX-512 answer. Hardcoding it hands `CpuShapeDispatcher` a width the \
         machine may not have, and on a machine WITH AVX-512 no behavioural \
         assertion can see the difference."
    );

    // VNNI implies avx512f. The converse does not hold, so this is one-way.
    if y::sentinel::host_has_avx512_vnni() {
        assert!(
            authority,
            "`host_has_avx512_vnni` says yes while the authority says the \
             machine has no AVX-512 at all"
        );
    }
}

/// The authority must ask whether the instruction will EXECUTE, not whether the
/// silicon reports it.
///
/// Asserted at the SOURCE, because it cannot be observed by running the
/// compiler: on a machine where the OS has enabled the register state the two
/// readings agree, so every behavioural assertion in this file passes with the
/// bug put back. Same shape as pinning the absence of an env override.
#[test]
fn the_authority_checks_the_register_state() {
    let src = std::fs::read_to_string(repo().join("src/sentinel.rs")).expect("sentinel.rs");

    for name in ["host_has_avx512", "host_has_avx"] {
        let start = src
            .find(&format!("\npub fn {name}()"))
            .unwrap_or_else(|| panic!("{name} must exist and be the authority"));
        let body = &src[start..];
        let end = body.find("\n}\n").expect("function end") + 3;
        let body = &body[..end];

        assert!(
            body.contains("is_x86_feature_detected!"),
            "{name} must use `is_x86_feature_detected!`, which performs the \
             XGETBV check. Its detection path compiles to code containing \
             `xgetbv`; a raw CPUID read cannot."
        );
        assert!(
            !body.contains("__cpuid"),
            "{name} reads CPUID directly. CPUID answers `is the silicon \
             capable`, and the emitter is asking `will this instruction \
             execute` - they differ whenever the OS has not enabled the \
             register state in XCR0."
        );
    }

    // And nothing else may derive the answer from the raw bits again. The leaf
    // 7 EBX bit 16 test is the shape the bug had; leaf 1 ECX bit 28 is the
    // AVX one beside it.
    for (needle, what) in [
        ("(1 << 16)) != 0", "CPUID leaf 7 EBX[16], the AVX-512 bit"),
        ("(1 << 28)) != 0", "CPUID leaf 1 ECX[28], the AVX bit"),
    ] {
        assert!(
            !src.contains(needle),
            "src/sentinel.rs still derives a vector-feature answer from {what}. \
             Route it through the authority."
        );
    }
}

/// A cached `.ysu_hw_profile` may not decide what the machine can execute.
///
/// `check_or_probe_hardware` skips the probe whenever the file merely exists,
/// so before this gate the file was authoritative: measured, `AVX512=true`
/// gave `target-cpu=znver5 +avx512f,...` and `AVX512=false` gave
/// `target-cpu=haswell +avx2` on one unchanged machine. A profile copied from a
/// better machine - or committed, which has happened in this repository - then
/// put AVX-512 into every function of the module.
///
/// The reasoning that justifies not validating the GPU half of the profile does
/// not transfer: validating `SM_VERSION` costs `cuInit`, and validating a CPU
/// feature costs one CPUID instruction.
#[test]
fn a_stale_profile_cannot_raise_the_answer() {
    let (dir_true, attrs_true) = emit_with_profile("stale_true", true);
    let (dir_false, attrs_false) = emit_with_profile("stale_false", false);

    assert_eq!(
        attrs_true, attrs_false,
        "the cached profile still decides `attributes #0`. The two runs differ \
         only in what the file claims about AVX-512, and the machine did not \
         change between them."
    );

    let _ = std::fs::remove_dir_all(&dir_true);
    let _ = std::fs::remove_dir_all(&dir_false);
}

/// The emitted module may not claim a feature the machine cannot execute.
///
/// This is the defect itself rather than its shape: `attributes #0` is applied
/// to every function in the module, so a wrong-high answer here is an illegal
/// instruction in code that has nothing to do with vectors - measured at 1,637
/// AVX-512 register references in the previous increment.
#[test]
fn the_emitted_attributes_do_not_outrun_the_machine() {
    let (dir, attrs) = emit_with_profile("honest", true);

    if !machine_really_has_avx512() {
        assert!(
            !attrs.contains("avx512"),
            "this machine has no AVX-512 and the emitted module asks for it: {attrs}"
        );
    }
    // Whatever it claims, it must be a claim the authority backs.
    assert_eq!(
        attrs.contains("avx512"),
        y::sentinel::host_has_avx512(),
        "`attributes #0` and the authority disagree about AVX-512: {attrs}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// THE CONTROL. Answering `false` always satisfies every assertion above while
/// deleting a working path.
///
/// It cannot skip itself: `machine_really_has_avx512` reads `/proc/cpuinfo`,
/// which is neither the authority nor anything the authority calls.
#[test]
fn the_fast_path_is_still_taken_on_hardware_that_has_it() {
    let expected = machine_really_has_avx512();
    assert_eq!(
        y::sentinel::host_has_avx512(),
        expected,
        "the authority disagrees with /proc/cpuinfo. Answering `false` on a \
         machine that HAS AVX-512 silently drops every kernel to the `haswell` \
         fallback; answering `true` on one that lacks it emits instructions \
         that fault."
    );

    if expected {
        let (dir, attrs) = emit_with_profile("control", true);
        assert!(
            attrs.contains("avx512f"),
            "this machine has AVX-512 and the emitter did not ask for it: \
             {attrs}. A gate made only of refusals passes perfectly when the \
             feature is never used."
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
