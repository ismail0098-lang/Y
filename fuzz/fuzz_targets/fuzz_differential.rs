//! Coverage-guided driver for the generative differential fuzzer.
//!
//! The generator, the reference interpreter and the oracles live in
//! `src/zk_fuzz.rs`, shared with `tests/zk_fuzz_differential.rs`. One
//! implementation, two harnesses — the same arrangement `zk_witness` has, and
//! for the same reason: two that disagreed would make both results meaningless.
//!
//! # What this file used to be
//!
//! It was named `fuzz_differential` and was **not differential**. It built a
//! single `ConstDecl` from a random arithmetic expression — no function, no
//! control flow, no branches — ran it through the CPU and ZK emitters, and then
//! compared *nothing*: it asserted the CPU output string was non-empty and
//! `eprintln!`d any ZK error. It could not have caught any of the six bugs the
//! real differential has now found, because it never asked whether the two
//! backends agreed about a value.
//!
//! The bytes libFuzzer supplies drive generator *choices* directly rather than
//! seeding a PRNG, so mutating one byte perturbs one decision and coverage
//! feedback still has a gradient.
//!
//! Run with:
//!     cargo +nightly fuzz run fuzz_differential

#![no_main]

use libfuzzer_sys::fuzz_target;
use y::zk_fuzz::check_bytes;

fuzz_target!(|data: &[u8]| {
    // Very short inputs generate a degenerate program; skip rather than spend
    // the corpus on them.
    if data.len() < 16 {
        return;
    }

    let findings = check_bytes(data);
    if findings.is_empty() {
        return;
    }

    // Panic, rather than log. A fuzz target that reports findings on stderr and
    // returns cannot fail, so libFuzzer keeps going and the run exits 0 — which
    // is exactly how the previous soundness target could have run for a week
    // over a backend computing the wrong function.
    let mut report = String::new();
    for f in &findings {
        report.push_str(&format!(
            "\n=== {:?} ===\ninputs: {:?}\n{}\n--- program ---\n{}",
            f.severity, f.inputs, f.detail, f.source
        ));
    }
    panic!("{}", report);
});
