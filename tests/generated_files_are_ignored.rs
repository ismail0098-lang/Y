//! `.gitignore` is a claim several other files lean on, and nothing checked it.
//!
//! `tests/ptx_portability.rs` records that `.ysu_hw_profile` was committed once,
//! carrying one card's `SM_VERSION`, and "Now gitignored" - so a fresh clone on
//! another card stops emitting PTX for the committer's GPU. The ptxas
//! translation validator's corpus, the Rocq build artifacts and every emitted
//! `*_certificate.v` are kept out of git the same way, and the comments beside
//! each pattern say why a committed copy would be wrong.
//!
//! **The merge `37651fb` emptied the file** (3,065 bytes to 0). The other side
//! of that merge carried a 112-byte fragment of it, partly UTF-16LE with
//! embedded NULs, so its patterns matched nothing either; the merge resolved
//! the conflict to nothing at all. Every claim above became false at once, and
//! the first sign was `git status` listing `target/` - which a `git add -A`
//! would have committed, ELF binaries and all.
//!
//! The gate is two-sided on purpose. A `.gitignore` of `*` passes every
//! "this generated path is ignored" assertion while hiding real work from
//! `git status`, so the second test requires that no TRACKED file matches an
//! ignore pattern - the committed `.ptx` artifacts and the validator's
//! hand-built `.sass` fixtures live right next to generated ones.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Paths this repository generates and must never commit, each with the
/// reason given where the pattern is written. They need not exist: `git
/// check-ignore` matches the path string against the patterns.
const GENERATED: &[(&str, &str)] = &[
    ("target/release/Y", "cargo's build output"),
    (
        ".ysu_hw_profile",
        "one machine's probed GPU; committed, it fixes every clone's PTX target",
    ),
    (".ysu/cache/0123abcd.ptx", "the Python JIT's disk cache, keyed on source text only"),
    ("proofs/ExactGemmWhole.vo", "coqc output beside a proof"),
    ("proofs/ExactGemmWhole.glob", "coqc output beside a proof"),
    ("proofs/.lia.cache", "the lia tactic's cache"),
    ("proofs/.nra.cache", "the nra tactic's cache"),
    ("tools/ptxas_tval/corpus/rope_64.sass", "the validator's regenerated corpus"),
    ("tools/ptxas_tval/o1/exact_pv.sass", "the validator's -O1 corpus"),
    ("tools/ptxas_tval/idiv/no_second_corr.sass", "derived division twins"),
    ("tools/ptxas_tval/guard_base.tgz", "a mutation harness's baseline archive"),
    ("tools/ptxas_tval/.frontier_cache_O3.json", "the census cache"),
    ("tools/ptxas_tval/fma/rn.cubin", "a machine-specific ELF"),
    // Written beside whatever source produced them, so the pattern must match
    // in any directory. (Not spelled under `tests/`: `proofs_are_checked`'s
    // citation sweep reads that as a reference to a missing file.)
    ("exact_matmul_certificate.v", "an emitted certificate"),
    ("my_kernels/exact_matmul_certificate_2.v", "an emitted certificate"),
    ("node_modules/solc/index.js", "npm's install tree"),
    ("python/y_lang/__pycache__/compiler.cpython-311.pyc", "Python bytecode"),
    ("output_bin", "--emit-native's default output"),
];

fn git(args: &[&str]) -> std::process::Output {
    Command::new("git")
        .args(args)
        .current_dir(repo())
        .output()
        .expect("git must be runnable; this test asks git, not a re-implementation of it")
}

#[test]
fn every_generated_path_is_ignored() {
    let raw = std::fs::read(repo().join(".gitignore")).expect("the repo has a .gitignore");
    // The failure mode the merge's other parent carried: a pattern with NULs in
    // it is not a pattern git can match, so the file looks populated and
    // ignores nothing.
    assert!(
        !raw.contains(&0u8),
        ".gitignore contains NUL bytes - part of it is UTF-16, and git matches none of it"
    );

    let mut not_ignored = Vec::new();
    for (path, why) in GENERATED {
        let out = git(&["check-ignore", "--no-index", "-q", path]);
        // 0 = ignored, 1 = not ignored, anything else = git itself failed.
        match out.status.code() {
            Some(0) => {}
            Some(1) => not_ignored.push(format!("{path}  ({why})")),
            other => panic!(
                "git check-ignore failed on {path} ({other:?}): {}",
                String::from_utf8_lossy(&out.stderr)
            ),
        }
    }
    assert!(
        not_ignored.is_empty(),
        "{} generated path(s) are NOT ignored, so `git add -A` would commit them:\n  {}",
        not_ignored.len(),
        not_ignored.join("\n  ")
    );
}

#[test]
fn no_tracked_file_is_ignored() {
    // `-c` tracked, `-i` matching an ignore pattern, `--exclude-standard` the
    // repo's own rules. A tracked file that matches is one an over-broad
    // pattern would hide from `git status` the moment it changed.
    let out = git(&["ls-files", "-c", "-i", "--exclude-standard"]);
    assert!(out.status.success(), "git ls-files failed");
    let hidden: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.to_string())
        .collect();
    assert!(
        hidden.is_empty(),
        "{} tracked file(s) match an ignore pattern:\n  {}",
        hidden.len(),
        hidden.join("\n  ")
    );

    // Non-vacuity: an empty answer is also what a repository git cannot read
    // returns. Count what it examined.
    let all = git(&["ls-files", "-c"]);
    let tracked = String::from_utf8_lossy(&all.stdout).lines().count();
    assert!(
        tracked > 500,
        "git listed only {tracked} tracked files; this is not the repository the test thinks it is"
    );
}
