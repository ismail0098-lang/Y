//! The committed `.ptx` files are artifacts, and an artifact drifts from the
//! compiler that produced it in silence.
//!
//! Two gates, both hermetic (no GPU, no `ptxas`, no recompilation), because
//! both failures are invisible to every gate this repo already had:
//!
//! * An EMPTY module assembles perfectly, so no `ptxas` sweep can see it.
//!   `tests/backends_refuse_empty_artifacts.rs` stops the backend PRODUCING
//!   one; this stops one being COMMITTED. 24 were, which is exactly the
//!   "24 of 85" the empty-artifact sweep measured - they were that bug's
//!   output, checked in.
//!
//! * An OVER-STATED `.version` assembles perfectly too, on the machine that
//!   wrote it. It fails at load time on a merely-older driver with
//!   `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, i.e. on a machine nobody tested.
//!   55 of the 59 committed artifacts at `.version 8.4` contained no FP8
//!   instruction at all - they demanded CUDA 12.4 to run two `mov`s.
//!
//! CLAUDE.md already recorded the second one happening ("two committed
//! artifacts were also still at 8.4 - the previous regeneration pass missed
//! them, and no test could tell"). The regeneration pass missed 32, not two.
//! A regeneration is a one-time act and this is the gate that makes it stick.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.ptx` **tracked in the repo**, from `git ls-files`.
///
/// This used to walk the filesystem from the repo root, excluding only
/// `target/`, `.git/` and `node_modules/`, with a comment saying it avoided a
/// subprocess. Two things were wrong with that. The file **already** runs `git
/// ls-files` in `committed_with_extension` for the `.ll` path, so there were
/// two notions of "committed" in one file and no subprocess was being saved;
/// and the walk picked up `tools/ptxas_tval/corpus/`, which is **generated and
/// gitignored** — 66 byte-copies of `tests/*.ptx` that the validator's
/// `build_corpus.sh` writes. They passed every assertion here precisely
/// because they are copies, so nothing failed and the docstring's claim was
/// quietly false from the moment that tool landed.
///
/// One notion of committed, asked of git, used by both.
fn committed_ptx() -> Vec<PathBuf> {
    let out = Command::new("git")
        .args(["ls-files", "*.ptx"])
        .current_dir(repo_root())
        .output()
        .expect("git ls-files");
    let mut v: Vec<PathBuf> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| repo_root().join(l.trim()))
        .filter(|p| p.exists())
        .collect();
    v.sort();
    v
}

/// A line that is not a directive, a comment, a brace or a parameter - i.e.
/// something the GPU would actually execute.
///
/// This is the same "assert the EFFECT, not the vocabulary" measure that
/// `tests/quantization_pass_refuses.rs` uses: a module full of comments and
/// `.reg` declarations assembles and does nothing.
fn instruction_lines(src: &str) -> usize {
    src.lines()
        .map(str::trim)
        .filter(|l| {
            !l.is_empty()
                && !l.starts_with("//")
                && !l.starts_with('.')
                && !l.starts_with('{')
                && !l.starts_with('}')
                && !l.starts_with(')')
        })
        .count()
}

fn header_value(src: &str, directive: &str) -> Option<String> {
    src.lines()
        .map(str::trim)
        .find(|l| l.starts_with(directive))
        .map(|l| l[directive.len()..].trim().to_string())
}

fn rel(p: &Path) -> String {
    p.strip_prefix(repo_root())
        .unwrap_or(p)
        .to_string_lossy()
        .into_owned()
}

#[test]
fn no_committed_ptx_module_is_empty() {
    let files = committed_ptx();
    let mut empty: Vec<String> = Vec::new();
    for p in &files {
        let src = std::fs::read_to_string(p).unwrap();
        if instruction_lines(&src) == 0 {
            empty.push(rel(p));
        }
    }
    assert!(
        empty.is_empty(),
        "these committed .ptx modules contain no instructions at all - they are \
         a `.version`/`.target` header and nothing else, which is what the PTX \
         backend used to emit for a source containing no `kernel`. `ptxas` \
         accepts them, so nothing else in this repo can see them. Delete them, \
         or regenerate from a source the backend accepts:\n  {}",
        empty.join("\n  ")
    );
}

/// The control for the test above: if the sweep found no files, "none of them
/// is empty" is vacuously true. This is the `feedback-null-metrics-pass-dead-
/// components` shape - a count of bad things being zero is also what a broken
/// harness reports.
#[test]
fn the_artifact_sweep_actually_finds_artifacts() {
    let files = committed_ptx();
    assert!(
        files.len() >= 40,
        "expected the repo to carry a substantial body of committed .ptx; found {}. \
         If they have genuinely been removed, lower this floor deliberately - do not \
         let the two gates in this file start passing vacuously.",
        files.len()
    );
    // And they must be real modules, not a directory of stubs.
    let total: usize = files
        .iter()
        .map(|p| instruction_lines(&std::fs::read_to_string(p).unwrap()))
        .sum();
    assert!(
        total > 1000,
        "the committed .ptx corpus holds only {total} instruction lines in total"
    );
}

#[test]
fn every_committed_ptx_declares_the_version_floor_for_its_target() {
    let mut wrong: Vec<String> = Vec::new();
    for p in committed_ptx() {
        let src = std::fs::read_to_string(&p).unwrap();
        let (target, version) = match (header_value(&src, ".target"), header_value(&src, ".version"))
        {
            (Some(t), Some(v)) => (t, v),
            // A file with no header is not a module this gate can judge; the
            // emptiness gate above is what covers a degenerate file.
            _ => continue,
        };

        // The compiler's own table is the reference. Re-deriving it here would
        // make a third copy, which CLAUDE.md records as the bug rather than the
        // fix (`the_coprocessor_backend_declares_the_same_floor_as_the_ptx_backend`
        // asserts the producers AGREE for the same reason).
        let floor = y::ptx_emitter::ptx_version_for_sm(&target)
            .trim_start_matches(".version")
            .trim()
            .to_string();

        // The one legitimate reason to declare MORE than the floor: FP8
        // `mma.sync` really does need ISA 8.4, even on the arch that has the
        // hardware. That is a per-INSTRUCTION requirement, so it is detected
        // per-instruction rather than waved through by architecture.
        let needs_fp8 = src.contains("e4m3") || src.contains("e5m2");
        let expected = if needs_fp8 && floor.as_str() < "8.4" {
            "8.4".to_string()
        } else {
            floor
        };

        if version != expected {
            wrong.push(format!(
                "{}: .target {target} declares .version {version}, expected {expected}{}",
                rel(&p),
                if needs_fp8 { " (FP8 present)" } else { "" }
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "a `.version` above the floor its instructions require is a DRIVER \
         requirement nothing in the artifact needs. It assembles perfectly here \
         and fails to load on an older driver with \
         CUDA_ERROR_UNSUPPORTED_PTX_VERSION. Regenerate these:\n  {}",
        wrong.join("\n  ")
    );
}

/// **A committed artifact whose source no longer compiles is a third staleness
/// class, and neither gate above can see it.**
///
/// `tests/train_spec.ptx` was committed with 11 instructions and no
/// `[Y ZERO DRIFT]` comment - emitted back when the directive did nothing -
/// and its source had been REFUSED ever since `@ZeroDrift` became real
/// (`@ZeroDrift` on a bare `F32` with no `@bounds` cannot be honoured). So the
/// checked-in file was one no run of the current compiler could produce, and
/// it passed `no_committed_ptx_module_is_empty` and
/// `every_committed_ptx_declares_the_version_its_target_requires` perfectly:
/// it is neither empty nor over-stated, just unreachable.
///
/// This asserts only that the source still COMPILES, not that the artifact is
/// byte-identical to a fresh emission. Byte-identity would be machine
/// dependent - emitted PTX can vary with `.ysu_hw_profile` and with tile
/// overrides like `Y_CTA_OVERRIDE`, which is exactly why that variable
/// "persists in whatever `.ptx` files were written while it was set".
///
/// Found by running a documented command. 60 committed `.ptx` have a matching
/// `.ysu`; 59 compiled, and this was the one that did not.

/// Committed artifacts with a given extension, for the source-compiles gate.
fn committed_with_extension(ext: &str) -> Vec<PathBuf> {
    let out = Command::new("git")
        .args(["ls-files", &format!("tests/*.{ext}")])
        .current_dir(repo_root())
        .output()
        .expect("git ls-files");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| repo_root().join(l.trim()))
        .filter(|p| p.exists())
        .collect()
}

#[test]
fn every_committed_artifact_still_has_a_source_that_compiles() {
    let mut refused = Vec::new();
    let mut compiled = 0usize;
    let mut coprocessor_checked = 0usize;
    let dir = std::env::temp_dir().join(format!("y_ptx_sources_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let mut artifacts: Vec<(PathBuf, &str)> = committed_ptx()
        .into_iter()
        // `<stem>.coprocessor.ptx` is emitted by a DIFFERENT backend and names
        // its source `<stem>.ysu`, not `<stem>.coprocessor.ysu`. The pairing
        // below is `with_extension("ysu")`, which asks for the latter, so all
        // seven coprocessor artifacts were SKIPPED - and one of them,
        // `hello.coprocessor.ptx`, was a `ret;`-only kernel the backend now
        // refuses to emit at all ("no RT Core work and no Tensor Core work, so
        // there is nothing to fuse"). A gate written for exactly the stale-
        // artifact class, with a blind spot created by a filename convention.
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if name.ends_with(".coprocessor.ptx") {
                (p, "--emit-coprocessor")
            } else {
                (p, "--emit-ptx")
            }
        })
        .collect();
    // `.ll` is covered by the same argument and had its own instance:
    // `tests/naive_gemm_f32.ll` was committed declaring `block_idx_y` and
    // friends as EXTERNS - it assembles and dies at link with `undefined
    // reference`, which is exactly the bug `llvm_emitter` was fixed to stop
    // producing. Deleted; the file is a GPU kernel and its artifact is the
    // committed `.ptx`.
    artifacts.extend(committed_with_extension("ll").into_iter().map(|p| (p, "--emit-llvm")));
    for (art, flag) in artifacts {
        // Strip EVERY artifact suffix, not just the last one: `.coprocessor` is
        // part of the artifact's name, not part of the source's.
        let stem = art
            .file_name()
            .unwrap()
            .to_string_lossy()
            .trim_end_matches(".ptx")
            .trim_end_matches(".ll")
            .trim_end_matches(".coprocessor")
            .to_string();
        let ysu = art.with_file_name(format!("{stem}.ysu"));
        if !ysu.exists() {
            continue; // generated fixtures without a checked-in source
        }
        // Compile a COPY in a temp directory. `--emit-ptx` writes its output
        // next to the source, so compiling in place would rewrite the very
        // artifacts under test and race any other test doing the same - this
        // repo has already been bitten by exactly that ("a test harness that
        // compiles the same .ysu from several threads races on the .ptx path",
        // which passed for several runs before failing).
        if flag == "--emit-coprocessor" {
            coprocessor_checked += 1;
        }
        let tmp = dir.join(ysu.file_name().unwrap());
        std::fs::copy(&ysu, &tmp).expect("copy fixture");
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&tmp)
            .arg(flag)
            .current_dir(repo_root())
            .output()
            .expect("run Y");
        // Counted AFTER the run, not before: an earlier version incremented
        // on finding the .ysu/.ptx pair, so neutering the compile loop left
        // the floor at 60 and the mutation survived. A non-vacuity floor has
        // to count the work, not the candidates.
        compiled += 1;
        if !out.status.success() {
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let why = text
                .lines()
                .find(|l| l.contains("Error") || l.contains("[!]"))
                .unwrap_or("<no diagnostic>")
                .trim()
                .to_string();
            refused.push(format!("{} ({flag}) -> {why}", rel(&ysu)));
        }
    }
    // NON-VACUITY, and it is load-bearing rather than decorative: the artifact
    // that exposed the blind spot has been DELETED, so reverting the pairing
    // above would otherwise be invisible - there would be nothing left for the
    // gate to fail on. This asserts the coprocessor family is actually reached.
    let committed_coprocessor = committed_ptx()
        .iter()
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".coprocessor.ptx")
        })
        .count();
    assert!(
        committed_coprocessor > 0 && coprocessor_checked == committed_coprocessor,
        "{coprocessor_checked} of {committed_coprocessor} committed \
         `*.coprocessor.ptx` were paired with a source - the rest were SKIPPED, \
         which is exactly the blind spot this pairing was widened to close \
         (`with_extension(\"ysu\")` asks for `<stem>.coprocessor.ysu`, which \
         never exists)"
    );
    assert!(
        compiled > 40,
        "only {compiled} sources were actually compiled - either the .ysu/.ptx \
         pairing has drifted or the loop stopped running the compiler, and \
         this gate is checking almost nothing"
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        refused.is_empty(),
        "these committed artifacts have a source the compiler now \
         REFUSES, so the checked-in file cannot be reproduced by any run:\n  {}",
        refused.join("\n  ")
    );
}

/// Every `.ptx` under `crates/y-gpu/ptx/` is embedded in the shipping library
/// with `include_str!`, so it is not merely an artifact — it is the code that
/// runs. This recompiles each one from its `.ysu` and requires **byte
/// identity**.
///
/// **The crate has its own freshness test and it had been red for a week,
/// because nothing runs it.** `cargo test` at this workspace root builds the
/// root package only; `crates/y-gpu/tests/ptx_is_not_stale.rs` needs
/// `cargo test -p y-gpu` or `--workspace`, and neither is in the documented
/// build commands. Four of the five embedded kernels were stale by the whole
/// carry-flag intrinsic series — `bn254_fr_mul_fast` shipped at 2,112 lines
/// with **zero** `add.cc` against the current 1,255 — so the GPU ZK library
/// was running the pre-carry-chain kernels. Same shape as `self_hosted/`,
/// which "was named only in a documentation file, so it had rotted
/// invisibly".
///
/// **The crate's own gate could not have passed anyway, and that is the more
/// interesting half.** It compares the embedded copy against a compile *on the
/// build machine*, and the whole point of these artifacts is that they are
/// portable: they are committed at `.target sm_80`, while a fresh compile here
/// probes an sm_89 card and says so. Those two requirements contradict each
/// other, and the freshness one loses — a compiler that probes the local
/// machine bakes that machine into its output, which is the bug the sm_80
/// regeneration existed to fix. It also produced a false positive:
/// `bn254_permute` was reported stale and was not, its body being identical.
///
/// So this gate does not choose a target. **It asks the artifact which target
/// it claims and reproduces that**, pinning a `.ysu_hw_profile` in the working
/// directory the way `ptx_portability::emitted_module_for` does. Whether that
/// claimed target is legitimate is a different question, already answered by
/// `every_committed_ptx_declares_the_version_floor_for_its_target` and by the
/// `.target`-suffix gate in `ptx_portability.rs`.
#[test]
fn the_shipped_gpu_kernels_match_their_sources() {
    let ptx_dir = repo_root().join("crates/y-gpu/ptx");
    let mut shipped: Vec<PathBuf> = committed_ptx()
        .into_iter()
        .filter(|p| p.parent() == Some(ptx_dir.as_path()))
        .collect();
    shipped.sort();

    let work = std::env::temp_dir().join(format!("y_shipped_ptx_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).expect("temp dir");

    let mut checked = 0usize;
    let mut stale = Vec::new();
    for art in &shipped {
        let stem = art.file_stem().unwrap().to_string_lossy().into_owned();
        let ysu = repo_root().join(format!("tests/{stem}.ysu"));
        assert!(
            ysu.exists(),
            "crates/y-gpu/ptx/{stem}.ptx ships in the library and has no source at \
             tests/{stem}.ysu, so nothing can say what it should contain"
        );
        let embedded = std::fs::read_to_string(art).expect("read the shipped PTX");
        let target = header_value(&embedded, ".target").unwrap_or_else(|| {
            panic!("crates/y-gpu/ptx/{stem}.ptx declares no .target")
        });

        // `PtxEmitter::new_with_profile` builds `sm_` + the version with the
        // dot stripped, so sm_80 comes from "8.0" and sm_120 from "12.0".
        let digits = target.trim_start_matches("sm_");
        assert!(
            digits.len() >= 2 && digits.bytes().all(|b| b.is_ascii_digit()),
            "crates/y-gpu/ptx/{stem}.ptx declares an unusable .target `{target}`"
        );
        let dotted = format!("{}.{}", &digits[..digits.len() - 1], &digits[digits.len() - 1..]);

        // A directory per kernel: the profile lives beside the source and the
        // compiler writes its output next to the source too, so two kernels
        // sharing a directory would be fine but two RUNS of this gate in
        // different processes would not.
        let dir = work.join(&stem);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("per-kernel temp dir");
        std::fs::write(
            dir.join(".ysu_hw_profile"),
            format!("SM_VERSION={dotted}\nGPU_NAME=ShippedArtifact\nSM_COUNT=66\n"),
        )
        .expect("pin the profile");
        let src = dir.join(format!("{stem}.ysu"));
        std::fs::copy(&ysu, &src).expect("copy the kernel source");

        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&src)
            .arg("--emit-ptx")
            // The working directory is what decides which `.ysu_hw_profile` is
            // read, so it must be the pinned one and NOT the repo. Running it
            // from the repo silently targets whatever card is in this machine,
            // which is exactly how the crate's own gate came to be unpassable.
            .current_dir(&dir)
            .output()
            .expect("run Y");
        assert!(
            out.status.success(),
            "tests/{stem}.ysu no longer compiles, so the kernel this library ships \
             cannot be reproduced:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let fresh = std::fs::read_to_string(dir.join(format!("{stem}.ptx")))
            .expect("no .ptx written");

        // Counted after the run, not on finding the pair: an earlier gate in
        // this file incremented before compiling, so neutering the loop left
        // the floor intact and the mutation survived.
        checked += 1;
        if fresh.trim() != embedded.trim() {
            stale.push(format!(
                "  crates/y-gpu/ptx/{stem}.ptx: {} lines shipped, {} lines fresh",
                embedded.lines().count(),
                fresh.lines().count()
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&work);

    assert!(
        stale.is_empty(),
        "the library ships kernels that no longer match their source:\n{}\n\
         Regenerate with a profile pinned to the artifact's own .target - NOT by \
         compiling in the repo, which bakes in the build machine's card:\n  \
         printf 'SM_VERSION=8.0\\n' > <tmp>/.ysu_hw_profile && cp tests/<k>.ysu <tmp>/ \\\n  \
         && (cd <tmp> && Y ./<k>.ysu --emit-ptx) && cp <tmp>/<k>.ptx crates/y-gpu/ptx/",
        stale.join("\n")
    );
    assert!(
        checked >= 5,
        "only {checked} shipped kernels were recompiled; the sweep found nothing to \
         check, which it would also report if `crates/y-gpu/ptx` had been moved"
    );
}
