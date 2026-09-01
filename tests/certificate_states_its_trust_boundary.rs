//! The emitted certificate must state its own trusted computing base, and the
//! statement must not be a hand-copy of the capstone's.
//!
//! ** WHY THIS FILE EXISTS, measured rather than supposed.
//!
//! `src/exact_gemm_certificate.rs` renders a `.v` beside every `.ll` in which
//! the exact `vpdpwssd` kernel was substituted, instantiating
//! `ExactGemmWhole.the_threaded_gemm_holds_the_source_dot_products`. Its own
//! module doc said - correctly - that "a certificate that overstates its scope
//! is worse than none", and it carried a "What is NOT claimed" section copied
//! by hand from that capstone's exclusion list.
//!
//! **The copy had dropped one of the capstone's three bullets and added one of
//! its own.** Both lists were three bullets long, so a count called them
//! equal. The dropped one is
//!
//!     "The threaded wrapper's `pthread` mechanics, the panel buffers' sizes
//!      and the scratch tile's allocation are not modelled at all."
//!
//! which is where BOTH of this repository's documented out-of-bounds writes
//! were: the three `ldc` sites in `emit_vnni_threaded_module` (observed as
//! `double free or corruption`, found only by writing `ExactGemmTiling.v`), and
//! the tile-count over-allocation caught by the schedule gate and by nothing
//! else. So the emitted certificate was silent about exactly the class of
//! defect this kernel has actually shipped.
//!
//! ** What each test here can and cannot see.
//!
//! - [every_trust_item_reaches_the_emitted_certificate] compiles a real nest
//!   and reads the `.v` a user would get. It catches a renderer whose output
//!   is dropped, and a renderer that skips the unchecked items - which is the
//!   failure mode that matters, since silence about an unchecked item is
//!   indistinguishable from not having one.
//! - [every_named_check_is_a_file_that_exists] catches a check that has been
//!   renamed or deleted. Same class as the citation sweep in
//!   `tests/proofs_are_checked.rs`, which found a stale path on its first run.
//! - [the_capstone_and_the_certificate_claim_the_same_exclusions] is the
//!   direction that actually failed, and it is a BIJECTION rather than a
//!   count. Every bullet of the capstone's list must be claimed by exactly one
//!   `TrustItem`, and every item attributed to the capstone must find exactly
//!   one bullet.
//!
//!   **It cannot see a bullet reworded around its phrase into a different
//!   claim.** Prose staleness is not mechanically decidable - which is the
//!   same limit `proofs/ExactGemmSchedule.v` exists to sidestep by being
//!   GENERATED, and the reason the certificate's section now is too.
//!
//! Note what is NOT asserted anywhere here: that the trust boundary is
//! COMPLETE. Nothing can assert that. What is asserted is that it does not
//! understate the one list in this repository that is obliged to be complete.

use std::path::PathBuf;
use std::process::Command;

use y::exact_gemm_certificate::{Check, CAPSTONE, TRUST_BOUNDARY};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A private scratch directory.
///
/// The tag is in the SIGNATURE rather than in a comment asking the caller to
/// remember: two tests sharing one temp-dir name, one calling `remove_dir_all`
/// while the other writes, is a race this repository has hit five times.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("y_tcb_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// An exact nest that licenses the substitution: `I16` operands widened to
/// `I64` at the load, a `@ZeroDrift` `I64` accumulator, and `@bounds` on BOTH
/// operands - a bound on the accumulator would state the range of the SUM and
/// imply nothing about its terms.
const EXACT_SOURCE: &str = r#"
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

/// Compile the exact nest in a private directory and return the certificate
/// the compiler wrote beside the `.ll`.
fn emitted_certificate(tag: &str) -> String {
    let dir = scratch(tag);
    let src = dir.join("gemm.ysu");
    std::fs::write(&src, EXACT_SOURCE).expect("write source");

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-llvm")
        // The compiler is run from the repository so it finds `c_src/` and the
        // hardware profile; the source is outside it, so nothing committed is
        // rewritten by a test run.
        .current_dir(repo())
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "the exact nest did not compile, so this file is testing nothing:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let cert = dir.join("gemm_certificate.v");
    assert!(
        cert.exists(),
        "no certificate was emitted for a nest that licenses the substitution. \
         Either the substitution stopped happening or the certificate stopped \
         being written; `tests/exact_gemm_certificate.rs` owns telling those apart."
    );
    std::fs::read_to_string(&cert).expect("read certificate")
}

/// Every item of the trust boundary appears in the file a user is handed,
/// including - especially - the ones nothing checks.
#[test]
fn every_trust_item_reaches_the_emitted_certificate() {
    let cert = emitted_certificate("reaches");

    // Rendering wraps, so compare on collapsed whitespace rather than on the
    // exact line breaks the wrapper happened to choose.
    let flat: String = cert.split_whitespace().collect::<Vec<_>>().join(" ");

    let mut pinned = 0usize;
    let mut unchecked = 0usize;
    for item in TRUST_BOUNDARY {
        for phrase in [item.claim, item.because] {
            let want: String = phrase.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(
                flat.contains(&want),
                "the emitted certificate does not carry this part of its own trust \
                 boundary:\n  {want}\nA certificate that omits an item of its trust \
                 boundary overstates what it proves."
            );
        }
        match item.check {
            Check::Pinned(f) => {
                pinned += 1;
                assert!(
                    flat.contains(&format!("Checked by: {f}")),
                    "the certificate does not name the check for `{}`", item.claim
                );
            }
            Check::Unchecked(what) => {
                unchecked += 1;
                let want: String = what.split_whitespace().collect::<Vec<_>>().join(" ");
                assert!(
                    flat.contains("NOT CHECKED.") && flat.contains(&want),
                    "the certificate does not say that `{}` is unchecked, or does not \
                     say what closing it would take. A renderer that quietly skips the \
                     unchecked items turns the honest half of the list into silence.",
                    item.claim
                );
            }
        }
    }

    // Non-vacuity, both ways. With no `Unchecked` item the branch above is
    // dead and this test degenerates into "the checked items are listed";
    // with no `Pinned` one the list is a caveat paragraph again.
    assert!(
        pinned >= 2 && unchecked >= 2,
        "the trust boundary has {pinned} checked and {unchecked} unchecked items; \
         with fewer than two of either, half the assertions above never run"
    );
}

/// A trust boundary that names its checks is only worth the checks existing.
#[test]
fn every_named_check_is_a_file_that_exists() {
    let mut named = 0usize;
    for item in TRUST_BOUNDARY {
        if let Check::Pinned(path) = item.check {
            named += 1;
            assert!(
                repo().join(path).is_file(),
                "the trust boundary says `{}` is checked by `{path}`, which does not \
                 exist. A certificate naming a check that has been renamed or deleted \
                 is worse than one admitting the item is unchecked.",
                item.claim
            );
        }
    }
    assert!(named >= 2, "only {named} items name a check; the sweep is near-vacuous");
}

/// The bullets of the capstone's own exclusion list.
///
/// Located by a marker phrase rather than by line number. If the marker moves
/// this returns nothing and the caller's floor fails loudly, which is the
/// behaviour wanted: a gate that silently finds zero bullets passes.
fn capstone_bullets() -> Vec<String> {
    let src = std::fs::read_to_string(repo().join(CAPSTONE))
        .unwrap_or_else(|e| panic!("read {CAPSTONE}: {e}"));
    const MARKER: &str = "What is STILL not proved";
    let start = src.find(MARKER).unwrap_or_else(|| {
        panic!(
            "{CAPSTONE} no longer contains the marker {MARKER:?} that locates its \
             exclusion list. Update the marker here deliberately - a gate that cannot \
             find the list it is comparing against would otherwise pass silently."
        )
    });

    // The section runs to the header comment's `Build:` line or its close.
    //
    // **It deliberately does NOT end at the first blank line**, which is what
    // an earlier version did. A mutation that appended a bullet after that
    // blank - still inside the exclusion section as any reader sees it, and
    // still before `Build:` - was then invisible, and the gate reported a
    // survivor that was really a hole one line wide. Ending at the section's
    // own terminator instead means a bullet anywhere in it counts.
    let mut bullets: Vec<String> = Vec::new();
    for line in src[start..].lines().skip(1) {
        let t = line.trim_start();
        if t.starts_with("Build:") || t.starts_with("*)") {
            break;
        }
        if let Some(rest) = line.strip_prefix("    - ") {
            bullets.push(rest.trim().to_string());
        } else if line.starts_with("      ") && !line.trim().is_empty() {
            if let Some(last) = bullets.last_mut() {
                last.push(' ');
                last.push_str(line.trim());
            }
        }
    }
    bullets
}

/// The direction that failed: a bullet the capstone states and the certificate
/// does not carry.
///
/// A BIJECTION, not a count. The two lists were both three bullets long while
/// disagreeing on one of them, so a count is exactly the check that would have
/// passed.
#[test]
fn the_capstone_and_the_certificate_claim_the_same_exclusions() {
    let bullets = capstone_bullets();
    assert!(
        bullets.len() >= 3,
        "found only {} bullets in {CAPSTONE}'s exclusion list; the parse has drifted \
         and every comparison below would be vacuous",
        bullets.len()
    );

    let attributed: Vec<_> = TRUST_BOUNDARY
        .iter()
        .filter_map(|i| match i.stated_in {
            Some((file, phrase)) if file == CAPSTONE => Some((i.claim, phrase)),
            _ => None,
        })
        .collect();
    assert!(
        attributed.len() >= 3,
        "only {} trust items are attributed to {CAPSTONE}",
        attributed.len()
    );

    // Every item finds exactly one bullet.
    for (claim, phrase) in &attributed {
        let hits = bullets.iter().filter(|b| b.contains(phrase)).count();
        assert_eq!(
            hits, 1,
            "the trust item `{claim}` locates its bullet in {CAPSTONE} by the phrase \
             {phrase:?}, which matches {hits} bullets rather than one. Either the \
             capstone reworded the bullet or the phrase is not distinctive."
        );
    }

    // Every bullet is claimed by exactly one item. This is the half that was
    // missing: the capstone's third bullet was claimed by nothing.
    for bullet in &bullets {
        let hits = attributed.iter().filter(|(_, p)| bullet.contains(p)).count();
        assert_eq!(
            hits, 1,
            "this bullet of {CAPSTONE}'s exclusion list is claimed by {hits} trust \
             items rather than one, so the emitted certificate does not carry it:\n  \
             {bullet}\nAdd it to `exact_gemm_certificate::TRUST_BOUNDARY` - with the \
             check that would fail if it were false, or `Check::Unchecked` saying what \
             closing it would take."
        );
    }
}

/// The items BELOW the model are the certificate's own, and they must exist.
///
/// A proof over `Z` has no opinion about a toolchain, so the capstone
/// correctly does not mention `clang` or the processor - which means a
/// bijection against the capstone alone can never require them. Without this,
/// deleting every `stated_in: None` item leaves the file green while the
/// certificate stops mentioning that it says nothing about machine code.
#[test]
fn the_certificate_states_the_boundary_below_the_model_too() {
    let own: Vec<_> = TRUST_BOUNDARY
        .iter()
        .filter(|i| i.stated_in.is_none())
        .collect();
    assert!(
        own.len() >= 3,
        "only {} trust items sit below the model. The proofs stop at the emitted IR, \
         so at minimum the toolchain beneath it, the proof checker, and the hardware \
         are trusted, and a certificate silent about them understates its boundary.",
        own.len()
    );

    let flat: String = own
        .iter()
        .flat_map(|i| [i.claim, i.because])
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    for want in ["clang", "rocq", "processor"] {
        assert!(
            flat.contains(want),
            "no trust item below the model mentions {want:?}"
        );
    }
}

/// Every proof of the exact-GEMM chain must be reachable from the capstone.
///
/// **This closes a hole a mutation found rather than a hazard I anticipated.**
/// The certificate mirrors ONE file's exclusion list -
/// [`exact_gemm_certificate::CAPSTONE`] - so a second root is a second list
/// with nothing reconciling it against the certificate. Removing
/// `ExactGemmWhole.v`'s `Require ExactGemmAllocation.` makes the allocation
/// proof exactly that, and every other gate stays green: the bijection still
/// holds (it is stated against the capstone), `only_the_capstone_states_a_global_negative`
/// permits a capstone to state one, and `coqc` does not care.
///
/// What it CANNOT see: a proof outside the `ExactGemm*` naming convention. The
/// convention is what makes "the chain" mechanically identifiable, and a file
/// deliberately named otherwise is outside this check by construction - which
/// is stated rather than implied.
#[test]
fn every_proof_of_the_chain_is_reachable_from_the_capstone() {
    let dir = repo().join("proofs");
    let mut requires: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for e in std::fs::read_dir(&dir).expect("read proofs/") {
        let path = e.expect("dir entry").path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) if n.ends_with(".v") => n.to_string(),
            _ => continue,
        };
        let src = std::fs::read_to_string(&path).expect("read proof");
        let deps: Vec<String> = src
            .lines()
            .filter_map(|l| l.trim().strip_prefix("Require "))
            // `Require Import X Y Z.` lists several; `Require X.` lists one.
            .flat_map(|r| {
                r.trim_end_matches('.')
                    .split_whitespace()
                    .filter(|w| *w != "Import" && *w != "Export")
                    .map(|w| format!("{w}.v"))
                    .collect::<Vec<_>>()
            })
            .collect();
        requires.insert(name, deps);
    }
    assert!(
        requires.len() >= 15,
        "only {} proofs found; the sweep would be near-vacuous",
        requires.len()
    );

    let root = CAPSTONE.rsplit('/').next().expect("capstone file name");
    assert!(requires.contains_key(root), "{CAPSTONE} is not in proofs/");

    // Transitive closure from the root.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut stack = vec![root.to_string()];
    while let Some(f) = stack.pop() {
        if !seen.insert(f.clone()) {
            continue;
        }
        for d in requires.get(&f).map(|v| v.as_slice()).unwrap_or(&[]) {
            stack.push(d.clone());
        }
    }

    let chain: Vec<&String> = requires
        .keys()
        .filter(|k| k.starts_with("ExactGemm"))
        .collect();
    assert!(chain.len() >= 8, "only {} chain files found", chain.len());
    for f in chain {
        assert!(
            seen.contains(f),
            "proofs/{f} is part of the exact-GEMM chain and is NOT reachable from \
             {CAPSTONE} by `Require`. It is therefore a second root, with its own \
             exclusion list that no certificate mirrors and nothing reconciles. Add \
             a `Require` (and USE it - an unused one is a comment) or say why the \
             file is deliberately outside the chain."
        );
    }
}
