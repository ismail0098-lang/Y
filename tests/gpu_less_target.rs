//! A machine with no NVIDIA GPU emitted `.target sm_00`, and `ptxas` rejects it.
//!
//! The probe records "no GPU" as compute capability `0.0` - `ysu_gpu_probe`'s
//! generic fallback writes `GPU_NAME=Generic GPU Emitter Target`, and
//! `sentinel`'s defaults give `GPU_NAME=Unknown GPU` when the probe binary
//! cannot run. Every PTX consumer then derived the target by deleting the dot
//! and prefixing `sm_`, so `--emit-ptx` on CI, in a container, or on a laptop
//! without an NVIDIA card wrote `.target sm_00` under "Compilation
//! Successful!" and exit 0, and `ptxas` answered `Unsupported .target 'sm_00'`.
//! `@require(sm < 50)` was SATISFIED there - the machine was read as having an
//! architecture-0 card - and `@require(sm >= 89)` was "unsatisfied" rather than
//! unknowable.
//!
//! It survived because the suite only ever ran on the developer's sm_89 card,
//! and because the one consumer that knew `0.0` means "no GPU" was
//! `--emit-coprocessor`, which special-cased it while the PTX emitter, the C
//! API and `@require` did not. `ptx_emitter::ptx_target_for` is the one rule
//! now, and these tests ask every consumer the same question.
//!
//! **Every case pins its own `.ysu_hw_profile` in a private directory.** The
//! verdict must not depend on which card - if any - the test runs on, which is
//! the very property this file exists to establish.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn have(prog: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {} >/dev/null 2>&1", prog))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

const KERNEL: &str = "kernel k(Out: GlobalMemory<F32>) {\n    \
                      let x: F32 = 2.0;\n    \
                      Out[0] = x * x;\n}\nfn main() {}\n";

/// Profiles that name NO architecture: the two the GPU-less probe actually
/// writes, a profile with no SM_VERSION line, an empty one, and nonsense.
const NO_ARCH: &[(&str, &str)] = &[
    ("probe_fallback", "SM_VERSION=0.0\nGPU_NAME=Generic GPU Emitter Target\nSM_COUNT=108\n"),
    ("probe_absent", "SM_VERSION=0.0\nGPU_NAME=Unknown GPU\nSM_COUNT=108\n"),
    ("no_line", "GPU_NAME=Unknown GPU\nSM_COUNT=108\n"),
    ("empty", "SM_VERSION=\nGPU_NAME=Unknown GPU\nSM_COUNT=108\n"),
    ("nonsense", "SM_VERSION=Unknown\nGPU_NAME=Unknown GPU\nSM_COUNT=108\n"),
];

struct Run {
    ok: bool,
    output: String,
    dir: PathBuf,
}

/// Compile `src` with `flag` in a fresh directory holding exactly `profile`.
///
/// The tag is in the signature and a counter makes the path unique, because a
/// shared temp directory is a race this repository has hit six times.
fn compile(tag: &str, profile: &str, src: &str, flag: &str) -> Run {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "y_gpu_less_{}_{}_{}",
        tag,
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    std::fs::write(dir.join(".ysu_hw_profile"), profile).expect("write profile");
    std::fs::write(dir.join("k.ysu"), src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg("k.ysu")
        .arg(flag)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    Run {
        ok: out.status.success(),
        output: format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        dir,
    }
}

fn header_line(ptx: &str, directive: &str) -> String {
    ptx.lines()
        .find(|l| l.starts_with(directive))
        .unwrap_or_else(|| panic!("no `{directive}` line in:\n{ptx}"))
        .trim()
        .to_string()
}

fn assemble(ptx_path: &Path, arch: &str) -> Result<(), String> {
    let out = Command::new("ptxas")
        .arg(format!("-arch={arch}"))
        .arg(ptx_path)
        .arg("-o")
        .arg("/dev/null")
        .output()
        .expect("run ptxas");
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).to_string())
    }
}

#[test]
fn a_profile_that_names_no_gpu_targets_the_floor_and_says_so() {
    let mut assembled = 0;
    for (tag, profile) in NO_ARCH {
        let r = compile(tag, profile, KERNEL, "--emit-ptx");
        assert!(r.ok, "[{tag}] a kernel with no requirements must compile:\n{}", r.output);
        let ptx = std::fs::read_to_string(r.dir.join("k.ptx")).expect("k.ptx written");
        assert_eq!(
            header_line(&ptx, ".target"),
            ".target sm_80",
            "[{tag}] a profile naming no architecture must target the floor, not invent one"
        );
        // The floor's own `.version`, not the 7.8 the unknown-arch default gave.
        assert_eq!(header_line(&ptx, ".version"), ".version 7.0", "[{tag}]");
        assert!(
            r.output.contains("PTX target: sm_80 (ASSUMED"),
            "[{tag}] the compiler must SAY the target was assumed:\n{}",
            r.output
        );
        assert!(!r.output.contains("sm_00"), "[{tag}] `sm_00` is still printed:\n{}", r.output);
        if have("ptxas") {
            assemble(&r.dir.join("k.ptx"), "sm_80")
                .unwrap_or_else(|e| panic!("[{tag}] ptxas rejects the emitted module:\n{e}"));
            assembled += 1;
        }
        let _ = std::fs::remove_dir_all(&r.dir);
    }
    if assembled == 0 {
        eprintln!("SKIP (assembly half): ptxas not found; the .target/.version assertions ran");
    }
}

/// The control: a profile that DOES name an architecture is honoured exactly,
/// and is not called assumed. Without this, "always emit sm_80" passes the
/// test above.
#[test]
fn a_real_architecture_is_honoured_and_not_called_assumed() {
    for (sm, want) in [
        ("8.9", "sm_89"),
        ("12.0", "sm_120"),
        ("9.0", "sm_90"),
        ("sm_86", "sm_86"),
        ("89", "sm_89"),
    ] {
        let profile = format!("SM_VERSION={sm}\nGPU_NAME=Pinned\nSM_COUNT=66\n");
        let r = compile("real", &profile, KERNEL, "--emit-ptx");
        assert!(r.ok, "[{sm}] must compile:\n{}", r.output);
        let ptx = std::fs::read_to_string(r.dir.join("k.ptx")).expect("k.ptx written");
        assert_eq!(header_line(&ptx, ".target"), format!(".target {want}"), "[{sm}]");
        assert!(
            r.output.contains(&format!("PTX target: {want}\n")),
            "[{sm}] a probed target must be reported plainly, with no ASSUMED:\n{}",
            r.output
        );
        let _ = std::fs::remove_dir_all(&r.dir);
    }
}

/// `@require(sm ...)` is a question about the target, and an ASSUMED target
/// is not an answer. Before the fix `0.0` was read as architecture 0, so
/// `sm < 50` was SATISFIED - a requirement answered by a card nobody has - and
/// the module went out as `sm_00`.
#[test]
fn require_sm_is_refused_not_answered_on_a_gpu_less_profile() {
    for cond in ["sm < 50", "sm >= 89"] {
        let src = format!("@require({cond})\n{KERNEL}");
        for (tag, profile) in NO_ARCH {
            let r = compile(tag, profile, &src, "--emit-ptx");
            assert!(!r.ok, "[{tag}] `@require({cond})` was answered on a GPU-less profile:\n{}", r.output);
            assert!(
                r.output.contains("error[R0004]"),
                "[{tag}] `@require({cond})` must be R0004 (supported, not determinable), \
                 not R0001 (a determined value):\n{}",
                r.output
            );
            assert!(
                r.output.contains("SM_VERSION=<major>.<minor>"),
                "[{tag}] the refusal must name the repair that works without a GPU:\n{}",
                r.output
            );
            assert!(!r.dir.join("k.ptx").exists(), "[{tag}] a refused compile wrote a module");
            let _ = std::fs::remove_dir_all(&r.dir);
        }
    }
    // Control: with an architecture named, the same requirement is EVALUATED -
    // so the refusal above is about the missing answer, not about `@require`.
    let pinned = "SM_VERSION=8.9\nGPU_NAME=Pinned\nSM_COUNT=66\n";
    let ok = compile("req_ok", pinned, &format!("@require(sm >= 89)\n{KERNEL}"), "--emit-ptx");
    assert!(ok.ok, "`sm >= 89` on an sm_89 target must compile:\n{}", ok.output);
    let no = compile("req_no", pinned, &format!("@require(sm < 50)\n{KERNEL}"), "--emit-ptx");
    assert!(!no.ok && no.output.contains("error[R0001]"), "{}", no.output);
}

/// The coprocessor backend carried the only correct copy of the rule; both
/// backends must now give the same `.target` for the same profile.
#[test]
fn the_coprocessor_and_ptx_backends_decide_the_target_alike() {
    let src = std::fs::read_to_string(repo().join("tests/coprocessor_attention.ysu"))
        .expect("the coprocessor fixture");
    let mut compared = 0;
    for profile in [
        NO_ARCH[0].1.to_string(),
        "SM_VERSION=8.9\nGPU_NAME=Pinned\nSM_COUNT=66\n".to_string(),
        "SM_VERSION=12.0\nGPU_NAME=Pinned\nSM_COUNT=66\n".to_string(),
    ] {
        let co = compile("co", &profile, &src, "--emit-coprocessor");
        assert!(co.ok, "the coprocessor fixture must compile:\n{}", co.output);
        let co_ptx = std::fs::read_to_string(co.dir.join("k.coprocessor.ptx"))
            .expect("k.coprocessor.ptx written");
        let px = compile("px", &profile, KERNEL, "--emit-ptx");
        assert!(px.ok, "{}", px.output);
        let px_ptx = std::fs::read_to_string(px.dir.join("k.ptx")).expect("k.ptx written");
        assert_eq!(
            header_line(&co_ptx, ".target"),
            header_line(&px_ptx, ".target"),
            "the two PTX producers disagree about the target for profile:\n{profile}"
        );
        compared += 1;
        let _ = std::fs::remove_dir_all(&co.dir);
        let _ = std::fs::remove_dir_all(&px.dir);
    }
    assert_eq!(compared, 3);
}

/// The C API is an entry point of its own - `y_compile_to_ptx` never passes
/// through the CLI's dispatch, and it evaluated no `@require` at all: a kernel
/// declaring `@require(sm >= 89)` came back as `.target sm_80` PTX with no
/// error. It also refuses a target the CALLER names that is no architecture,
/// instead of quietly using the floor.
#[test]
fn the_c_api_evaluates_require_and_refuses_a_target_that_names_nothing() {
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;

    fn call(src: &str, target: &str) -> Result<String, String> {
        let src = CString::new(src).unwrap();
        let target = CString::new(target).unwrap();
        let mut err: *mut c_char = std::ptr::null_mut();
        // SAFETY: valid NUL-terminated strings; both returned pointers are
        // freed with `y_free_string`, as the API documents.
        unsafe {
            let p = y::c_api::y_compile_to_ptx(src.as_ptr(), target.as_ptr(), &mut err);
            if p.is_null() {
                let msg = if err.is_null() {
                    String::new()
                } else {
                    let m = CStr::from_ptr(err).to_string_lossy().to_string();
                    y::c_api::y_free_string(err);
                    m
                };
                Err(msg)
            } else {
                let s = CStr::from_ptr(p as *const c_char).to_string_lossy().to_string();
                y::c_api::y_free_string(p as *mut c_char);
                Ok(s)
            }
        }
    }

    let req = format!("@require(sm >= 89)\n{KERNEL}");
    let refused = call(&req, "sm_80").expect_err("`@require(sm >= 89)` compiled for sm_80");
    assert!(refused.contains("error[R0001]"), "{refused}");
    let ptx = call(&req, "sm_89").expect("`@require(sm >= 89)` for sm_89 must compile");
    assert!(ptx.lines().any(|l| l.trim() == ".target sm_89"), "{ptx}");

    for bad in ["sm_00", "0.0", "garbage"] {
        let e = call(KERNEL, bad).expect_err("an explicit non-architecture target compiled");
        assert!(e.contains("does not name a GPU architecture"), "[{bad}] {e}");
    }
    // Control: an ordinary explicit target still compiles to exactly itself.
    let ok = call(KERNEL, "sm_86").expect("sm_86 must compile");
    assert!(ok.lines().any(|l| l.trim() == ".target sm_86"), "{ok}");
}

/// The source-level half. All four hand-rolled derivations shared one
/// signature - deleting the dot from `sm_version` - and a fifth would be the
/// same bug again. Only `ptx_target_for` may turn a profile into a target.
#[test]
fn no_consumer_derives_a_target_from_sm_version_itself() {
    let mut offenders = Vec::new();
    let mut scanned = 0;
    for e in std::fs::read_dir(repo().join("src")).expect("src/") {
        let p = e.expect("entry").path();
        if p.extension().and_then(|x| x.to_str()) != Some("rs") {
            continue;
        }
        scanned += 1;
        let text = std::fs::read_to_string(&p).expect("read source");
        for (i, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if code.contains("sm_version.replace(") || code.contains("sm_version != \"0.0\"") {
                offenders.push(format!("{}:{}: {}", p.display(), i + 1, line.trim()));
            }
        }
    }
    assert!(scanned > 20, "only {scanned} source files scanned");
    assert!(
        offenders.is_empty(),
        "a target is being derived from `sm_version` outside `ptx_target_for`:\n  {}",
        offenders.join("\n  ")
    );
}
