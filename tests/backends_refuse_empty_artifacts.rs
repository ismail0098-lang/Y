//! A backend with nothing to emit must say so, not write an empty artifact.
//!
//! Two backends were writing files that assemble, launch, and do nothing, both
//! under a green banner and exit 0.
//!
//! **`--emit-ptx`.** `PtxEmitter::emit_program` matches `Item::Kernel` and
//! nothing else, so a source with no `kernel` in it produced a three-line
//! module - `.version`, `.target`, `.address_size`, no instructions. **24 of
//! the 85 programs in `tests/` emitted one.** `tests/hello.ysu` - the program
//! the README uses - compiled to a header.
//!
//! **`--emit-coprocessor`, and this one is worse.** With zero RT Core nodes and
//! zero Tensor Core nodes it still emitted a complete `.visible .entry
//! y_coprocessor_fused` whose body is `ret;`, under a **hardcoded two-parameter
//! signature** (`param_rt_A_ptr`, `param_nns_query_ptr`) byte-identical for
//! every program and derived from none of them, plus a banner comment reporting
//! "RT Nodes: 0 | Tensor Nodes: 0 | Barriers: 0" and "Dual-accelerator PTX
//! generated successfully!". **77 of the 83 programs it accepted were that**;
//! only the six `coprocessor_*.ysu` files were real. `println("hi")` compiled
//! to a launchable GPU kernel.
//!
//! It does not LOOK empty, which is the whole problem - it has an entry point,
//! a parameter list and a comment claiming to be a schedule. Gotcha #8's rule
//! applies: "a named gap costs a user five minutes, a plausible-looking broken
//! kernel costs them however long it takes to suspect the compiler".
//!
//! ## Why no existing gate could see either
//!
//! **An empty module assembles perfectly.** Every `ptxas` gate this repo added
//! after gotcha #8 passes it, and a corpus-wide assemble sweep run immediately
//! before this fix reported **82 of 82 accepted**. That is the same limit
//! recorded there for a *missing instruction*, one level up: here the whole
//! program is missing. `success_banner_means_success` cannot see it either -
//! the banner and the exit code agree with each other, and both are wrong about
//! the artifact.
//!
//! So the sweeps below assert on the CONTENT of what was emitted, not on
//! whether a tool accepted it - `feedback-decorative-codegen-passes-every-test`
//! ("assert the EFFECT not the vocabulary"), which found the same shape in
//! `quantization_pass`.
use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Compile `src` in a private directory; return (ok, output, emitted artifact).
fn compile(name: &str, src: &str, flag: &str, ext: &str) -> (bool, String, Option<String>) {
    let dir = std::env::temp_dir().join(format!("y_empty_{}_{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg(flag)
        .current_dir(repo())
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let art = std::fs::read_to_string(dir.join(format!("{}.{}", name, ext))).ok();
    (out.status.success(), text, art)
}

const HOST_ONLY: &str = "\
@unsafe
fn main() {
    println(\"hi\");
}
";

/// A real kernel, small enough to be obviously one.
const A_KERNEL: &str = "\
kernel touch_it(Out: GlobalMemory<U32>, N: U32) {
    let tid: U32 = thread_idx_x();
    block_ptr2d_store(Out, 0, tid, 1, 1, N, tid);
}
";

/// A real fused program: one RT Core op feeding a Tensor Core MMA.
const FUSED: &str = "\
@unsafe
fn main() {
    let nns_res: I32 = rt_nearest_neighbor(128, 8);
    let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
    let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(nns_res);
    let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(nns_res);
    let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(nns_res);
    acc = mma_sync(frag_A, frag_B, frag_C);
}
";

/// Lines inside a PTX entry that are real instructions.
///
/// Excludes the parameter list, register pools, comments, `ld.param` and the
/// closing `ret;` - i.e. everything the fabricated kernel consisted of. The
/// separation is not marginal: a real fused kernel has ~83 of these and the
/// fabricated one had 0.
fn instruction_lines(ptx: &str) -> usize {
    ptx.lines()
        .skip_while(|l| !l.contains(".visible .entry"))
        .take_while(|l| !l.starts_with('}'))
        .map(str::trim)
        .filter(|l| {
            l.ends_with(';')
                && !l.starts_with("//")
                && !l.starts_with(".reg")
                && !l.starts_with(".shared")
                && !l.starts_with("ld.param")
                && *l != "ret;"
        })
        .count()
}

// ── --emit-ptx ────────────────────────────────────────────

#[test]
fn the_ptx_backend_refuses_a_source_with_no_kernel() {
    let (ok, text, art) = compile("noker", HOST_ONLY, "--emit-ptx", "ptx");
    assert!(!ok, "a source with no `kernel` compiled and exited 0:\n{}", text);
    assert!(
        text.contains("kernel"),
        "refused without naming what was missing:\n{}",
        text
    );
    assert!(
        art.is_none(),
        "refused, but still wrote a .ptx file - the refusal has to happen \
         before the artifact exists, or the user has a file to be misled by"
    );
}

/// The control. "Refuse everything" satisfies the test above and deletes the
/// backend - the shape `ordinary_loop_bodies_still_verify` exists for.
#[test]
fn the_ptx_backend_still_emits_a_real_kernel() {
    let (ok, text, art) = compile("realker", A_KERNEL, "--emit-ptx", "ptx");
    assert!(ok, "a real kernel was refused:\n{}", text);
    let ptx = art.expect("no .ptx written for a real kernel");
    assert!(
        ptx.contains(".visible .entry"),
        "emitted a module with no entry point:\n{}",
        ptx
    );
    assert!(
        instruction_lines(&ptx) > 0,
        "emitted an entry point with no instructions in it:\n{}",
        ptx
    );
}

// ── --emit-coprocessor ────────────────────────────────────

#[test]
fn the_coprocessor_backend_refuses_a_source_with_nothing_to_fuse() {
    let (ok, text, art) =
        compile("nofuse", HOST_ONLY, "--emit-coprocessor", "coprocessor.ptx");
    assert!(
        !ok,
        "a source with no RT and no Tensor work compiled and exited 0:\n{}",
        text
    );
    assert!(
        text.contains("nothing to fuse"),
        "refused without saying why:\n{}",
        text
    );
    assert!(art.is_none(), "refused, but still wrote a .coprocessor.ptx");
}

/// The control, and it is the one that would have caught the original bug from
/// the other side: a real fused kernel must contain real instructions, not the
/// `ret;` the fabricated one had.
#[test]
fn the_coprocessor_backend_still_emits_a_real_fused_kernel() {
    let (ok, text, art) =
        compile("realfuse", FUSED, "--emit-coprocessor", "coprocessor.ptx");
    assert!(ok, "a real RT+Tensor program was refused:\n{}", text);
    let ptx = art.expect("no .coprocessor.ptx written for a real fused program");
    let n = instruction_lines(&ptx);
    assert!(
        n > 5,
        "the fused kernel has {} instruction lines - that is the do-nothing \
         `ret;` body this file exists to stop:\n{}",
        n,
        ptx
    );
}

// ── corpus sweeps ─────────────────────────────────────────

fn corpus() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(repo().join("tests"))
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "ysu").unwrap_or(false))
        .collect();
    v.sort();
    assert!(v.len() > 50, "corpus shrank to {} files", v.len());
    v
}

/// Sweeps run in a private copy of the corpus: `--emit-ptx` writes its artifact
/// beside the input, and some of `tests/*.ptx` is committed.
fn sweep(flag: &str, ext: &str, tag: &str) -> (usize, usize, Vec<String>) {
    let work = std::env::temp_dir().join(format!("y_sweep_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let (mut emitted, mut refused, mut bad) = (0, 0, Vec::new());
    for src in corpus() {
        let stem = src.file_stem().unwrap().to_string_lossy().to_string();
        let copy = work.join(format!("{}.ysu", stem));
        std::fs::copy(&src, &copy).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&copy)
            .arg(flag)
            .current_dir(repo())
            .output()
            .expect("run Y");
        let art = std::fs::read_to_string(work.join(format!("{}.{}", stem, ext)));
        match (out.status.success(), art) {
            (true, Ok(ptx)) => {
                emitted += 1;
                if instruction_lines(&ptx) == 0 {
                    bad.push(format!("  {}: emitted, but contains no instructions", stem));
                }
            }
            (true, Err(_)) => {
                emitted += 1;
                bad.push(format!("  {}: exited 0 but wrote no artifact", stem));
            }
            (false, Ok(_)) => bad.push(format!("  {}: refused, but wrote an artifact anyway", stem)),
            (false, Err(_)) => refused += 1,
        }
    }
    let _ = std::fs::remove_dir_all(&work);
    (emitted, refused, bad)
}

#[test]
fn no_emitted_ptx_module_is_empty() {
    let (emitted, refused, bad) = sweep("--emit-ptx", "ptx", "ptx");
    // 24 of 85 used to be empty and every `ptxas` gate passed them, so the
    // floor is what stops "refuse everything" from being a green run.
    assert!(
        emitted >= 40,
        "only {} programs emitted a module (refused {}), so this sweep proves \
         nothing",
        emitted,
        refused
    );
    assert!(
        bad.is_empty(),
        "{} of {} emitted PTX modules are empty or missing (refused {}):\n{}",
        bad.len(),
        emitted,
        refused,
        bad.join("\n")
    );
}

#[test]
fn no_emitted_coprocessor_kernel_is_a_bare_ret() {
    let (emitted, refused, bad) = sweep("--emit-coprocessor", "coprocessor.ptx", "cop");
    // Only the six `coprocessor_*.ysu` files have both pipelines, so the floor
    // here is small by nature - but it must not be zero, or a backend that
    // refused everything would pass.
    assert!(
        emitted >= 5,
        "only {} programs emitted a fused kernel (refused {}), so this sweep \
         proves nothing",
        emitted,
        refused
    );
    assert!(
        bad.is_empty(),
        "{} of {} emitted co-processor kernels are do-nothing bodies \
         (refused {}):\n{}",
        bad.len(),
        emitted,
        refused,
        bad.join("\n")
    );
}

// ── --emit-native ───────────────────────────────────────

/// A `kernel` reaching the native backend was dropped without a word.
///
/// `native_emitter::emit_program` was `if let Item::Func(f) = item`, so every
/// other `Item` variant fell out of the loop. A `kernel` beside an empty `main`
/// produced a 162-byte executable **byte-identical** to the one for
/// `fn main() {}` alone, under "Compiled to native ELF executable!" and exit 0
/// - the whole kernel contributed nothing and nothing said so.
///
/// This is the mirror of the `ptx_emitter::emit_program` bug in the same commit
/// series, which matched `Item::Kernel` and dropped everything else. The two
/// backends were each keeping only the half they understood.
///
/// `--emit-cpu` already refused this program (it walks the kernel body and
/// rejects `thread_idx_x` by name), so the host backends now agree.
mod native_refuses_a_kernel {
    use std::path::PathBuf;
    use std::process::Command;

    fn dir_for(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("y_nat_{}_{}", std::process::id(), name))
    }

    fn build(name: &str, src: &str) -> (bool, String, bool) {
        let dir = dir_for(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(format!("{}.ysu", name));
        std::fs::write(&path, src).expect("write source");
        let bin = dir.join(format!("{}.bin", name));
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&path)
            .arg("--emit-native")
            .arg("-o")
            .arg(&bin)
            .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
            .output()
            .expect("run Y");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.success(), text, bin.exists())
    }

    const KERNEL_PLUS_STUB: &str = "\
kernel k(Out: GlobalMemory<U32>, N: U32) {
    let t: U32 = thread_idx_x();
    block_ptr2d_store(Out, 0, t, 1, 1, N, t);
}

fn main() {}
";

    #[test]
    fn a_kernel_is_refused_by_name_not_dropped() {
        let (ok, text, wrote) = build("kstub", KERNEL_PLUS_STUB);
        assert!(!ok, "a kernel compiled to a native ELF and exited 0:\n{}", text);
        assert!(
            text.contains("kernel k"),
            "refused without naming the kernel it could not lower:\n{}",
            text
        );
        assert!(
            !wrote,
            "refused, but still wrote an executable - the refusal has to land \
             before the file exists or the user has a binary to be misled by"
        );
    }

    /// The control. Refusing every program satisfies the test above and deletes
    /// the backend; this one RUNS the binary, because emitting something is not
    /// the same as emitting something correct.
    #[test]
    fn an_ordinary_program_still_builds_and_runs() {
        let src = "\
fn main() -> I32 {
    let a: I32 = 9;
    let b: I32 = 2;
    return a - b;
}
";
        let (ok, text, wrote) = build("plain", src);
        assert!(ok, "an ordinary integer program was refused:\n{}", text);
        assert!(wrote, "exited 0 but wrote no executable:\n{}", text);
        let code = Command::new(dir_for("plain").join("plain.bin"))
            .status()
            .expect("run the produced binary")
            .code();
        assert_eq!(code, Some(7), "the produced binary computed the wrong answer");
    }

    /// A type declaration emits no code, so dropping it is correct and must stay
    /// allowed - otherwise "refuse every non-Func item" passes the kernel test
    /// and breaks ordinary programs.
    #[test]
    fn a_type_declaration_beside_main_is_still_fine() {
        let src = "\
struct P { a: I32 }

fn main() -> I32 {
    return 3;
}
";
        let (ok, text, wrote) = build("withstruct", src);
        assert!(ok, "a struct declaration beside main was refused:\n{}", text);
        assert!(wrote, "exited 0 but wrote no executable:\n{}", text);
    }
}

// ── the C API ─────────────────────────────────────────────

/// `y_compile_to_ptx` must report the emitter's refusals, not return a string.
///
/// It discarded `emit_errors` entirely: a program the PTX backend could not
/// lower came back to the C caller as a PTX string with a null error. Every
/// refusal `PtxEmitter` has ever had went out that door silently - `tma_load`
/// and `wgmma_async` (which assemble to nothing), the intrinsic arity gate, the
/// unknown-call gate - and a source with no `kernel` returned a three-line
/// `.version`/`.target` header as though it were a compiled program.
///
/// `y_interpret_kernel`, the CPU entry point in the same file, has this exact
/// fix with this exact comment. The PTX path never got it. That is the
/// "a correct guard is worthless where it is not called" shape for the third
/// time in this repo, so the test asserts BOTH entry points, not just the one
/// that was broken.
mod c_api_reports_refusals {
    use std::ffi::{CStr, CString};
    use std::ptr;
    use y::c_api::{y_compile_to_ptx, y_free_string};

    fn compile(src: &str) -> (bool, String) {
        let c = CString::new(src).unwrap();
        let mut err: *mut std::os::raw::c_char = ptr::null_mut();
        let sm = CString::new("sm_89").unwrap();
        unsafe {
            let out = y_compile_to_ptx(c.as_ptr(), sm.as_ptr(), &mut err);
            let msg = if err.is_null() {
                String::new()
            } else {
                let m = CStr::from_ptr(err).to_string_lossy().into_owned();
                y_free_string(err);
                m
            };
            let ok = !out.is_null();
            if ok {
                y_free_string(out);
            }
            (ok, msg)
        }
    }

    #[test]
    fn a_source_with_no_kernel_is_an_error_not_a_ptx_string() {
        let (ok, msg) = compile("@unsafe\nfn main() {\n    println(\"hi\");\n}\n");
        assert!(
            !ok,
            "returned a PTX string for a source with no kernel; the caller has \
             no way to tell it apart from a compiled program"
        );
        assert!(
            msg.contains("kernel"),
            "returned an error, but it does not say what was missing: {:?}",
            msg
        );
    }

    /// The control: a real kernel must still come back as PTX with a null
    /// error, or "return an error always" passes the test above.
    #[test]
    fn a_real_kernel_still_comes_back_as_ptx() {
        let (ok, msg) = compile(super::A_KERNEL);
        assert!(ok, "a real kernel was refused by the C API: {:?}", msg);
        assert!(msg.is_empty(), "a successful compile set an error: {:?}", msg);
    }
}
