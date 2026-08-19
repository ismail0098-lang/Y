//! The generative differential fuzzer, run as a deterministic regression gate.
//!
//! `src/zk_fuzz.rs` holds the generator, the reference interpreter and the
//! oracles; this file drives them from a fixed seed sequence so a finding
//! reproduces from its seed alone, and `fuzz/fuzz_targets/fuzz_differential.rs`
//! drives the same code from libFuzzer's bytes. One generator, two harnesses —
//! the same arrangement `zk_witness` has, and for the same reason.
//!
//! Run the long sweep with:
//!     cargo test --release --features zk --test zk_fuzz_differential -- --ignored --nocapture

#![cfg(feature = "zk")]

use y::zk_fuzz::{
    check, check_bytes, gen_program, gen_value, minimize, render, Entropy, Finding,
    GenConfig, Severity,
};

/// SplitMix64. A seeded byte source, so that "seed 12345 fails" is a complete
/// bug report.
struct Seeded(u64);

impl Seeded {
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) & 0xff) as u8
            })
            .collect()
    }
}

fn report(seed: u64, f: &Finding) -> String {
    format!(
        "\n=== {:?} (seed {}) ===\ninputs: {:?}\n{}\n--- program ---\n{}",
        f.severity, seed, f.inputs, f.detail, f.source
    )
}

/// Sweep `count` programs starting at `base_seed`, returning every finding.
fn sweep(base_seed: u64, count: u64) -> Vec<(u64, Finding)> {
    let mut out = Vec::new();
    for i in 0..count {
        let seed = base_seed.wrapping_add(i);
        let mut rng = Seeded(seed);
        let bytes = rng.bytes(512);
        for f in check_bytes(&bytes) {
            out.push((seed, f));
        }
    }
    out
}

/// The always-on gate. Small enough to run in the default suite.
#[test]
fn generated_programs_agree_with_their_semantics() {
    let findings = sweep(0xC0FFEE, 400);
    let hard: Vec<_> = findings
        .iter()
        .filter(|(_, f)| f.severity == Severity::WrongValue)
        .collect();
    assert!(
        hard.is_empty(),
        "the ZK backend computed a different function than the program on {} of 400 programs:{}",
        hard.len(),
        hard.iter()
            .take(3)
            .map(|(s, f)| report(*s, f))
            .collect::<Vec<_>>()
            .join("")
    );
}

/// Divergence between the constant-folded and gadget paths, held separately so
/// a fold-only disagreement does not read as a wrong-value bug.
#[test]
fn constant_folding_agrees_with_gadget_emission() {
    let findings = sweep(0x5EED, 400);
    let diverged: Vec<_> = findings
        .iter()
        .filter(|(_, f)| f.severity == Severity::FoldingDivergence)
        .collect();
    assert!(
        diverged.is_empty(),
        "{} programs compile to different circuits depending on whether their \
         inputs are literals:{}",
        diverged.len(),
        diverged
            .iter()
            .take(3)
            .map(|(s, f)| report(*s, f))
            .collect::<Vec<_>>()
            .join("")
    );
}

/// Every program this fuzzer renders is well-formed Y by construction, so a
/// parse failure is unambiguously a front-end bug — there is no reading under
/// which it is the right answer.
///
/// This gate exists because the parser bug the fuzzer found (`if p0 { }`
/// misparsed as an empty struct literal) surfaced only as an over-refusal,
/// which is not gated: a semantic refusal is fail-closed and legitimate, and
/// lumping the two together meant reverting the parser fix passed the whole
/// sweep. Mutation testing is what exposed that.
#[test]
fn every_generated_program_parses() {
    let findings = sweep(0x9A75E, 400);
    let broken: Vec<_> = findings
        .iter()
        .filter(|(_, f)| f.severity == Severity::ParseFailure)
        .collect();
    assert!(
        broken.is_empty(),
        "{} well-formed programs failed to parse:{}",
        broken.len(),
        broken
            .iter()
            .take(3)
            .map(|(s, f)| report(*s, f))
            .collect::<Vec<_>>()
            .join("")
    );
}

/// **The control.** A fuzzer that generates nothing interesting passes every
/// assertion above. This asserts the corpus actually reaches the constructs the
/// oracles are about — without it, gutting the generator would look like a
/// clean bill of health.
#[test]
fn the_generated_corpus_is_not_vacuous() {
    let (mut ifs, mut loops, mut cmps, mut returns, mut assigns) = (0, 0, 0, 0, 0);
    let mut compounds = 0;
    for i in 0..400u64 {
        let mut rng = Seeded(0xBEEF + i);
        let bytes = rng.bytes(512);
        let prog = gen_program(&mut Entropy::new(&bytes), &GenConfig::default());
        let src = render(&prog, None);
        if src.contains("if ") {
            ifs += 1;
        }
        if src.contains("for ") {
            loops += 1;
        }
        if src.contains(" < ") || src.contains(" >= ") || src.contains(" <= ") {
            cmps += 1;
        }
        returns += src.matches("return").count();
        if src.contains(" = ") {
            assigns += 1;
        }
        if src.contains(" += ") || src.contains(" -= ") || src.contains(" *= ") {
            compounds += 1;
        }
    }
    assert!(ifs > 100, "only {} programs contained an `if`", ifs);
    assert!(loops > 50, "only {} programs contained a `for`", loops);
    assert!(cmps > 100, "only {} programs contained an ordering", cmps);
    assert!(assigns > 50, "only {} programs contained an assignment", assigns);
    assert!(
        compounds > 30,
        "only {} programs contained a compound assignment. `x += e` was the \
         shape `zk_emitter::emit_stmt` silently dropped, and the generator \
         could not write one - so oracle 1 had nothing to disagree about. A \
         generator restriction is a deleted bug class.",
        compounds
    );
    assert!(
        returns > 800,
        "only {} `return`s across the corpus - the multi-return shapes that \
         carried all four control-flow bugs are not being generated",
        returns
    );
}

/// Print one seed's program and every oracle's verdict, for investigating a
/// finding. Reads the seed rather than setting anything, so it cannot race the
/// rest of the suite.
///
///     Y_FUZZ_SEED=5 cargo test --release --features zk \
///         --test zk_fuzz_differential -- dump_seed --ignored --nocapture
#[test]
#[ignore]
fn dump_seed() {
    let seed: u64 = std::env::var("Y_FUZZ_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let mut rng = Seeded(seed);
    let bytes = rng.bytes(512);
    // Mirror `check_bytes` exactly - one Entropy, program first, then the input
    // vectors from the same stream - or this dumps a different program than the
    // sweep reported.
    let cfg = GenConfig::default();
    let mut e = Entropy::new(&bytes);
    let prog = gen_program(&mut e, &cfg);
    for _ in 0..4 {
        let inputs: Vec<u64> = (0..prog.nparams).map(|_| gen_value(&mut e)).collect();
        let found = check(&prog, &inputs);
        if let Some(f) = found.first() {
            println!("{}", report(seed, f));
            let small = minimize(&prog, &inputs, f.severity.clone());
            println!(
                "\n--- minimised ({:?}, inputs {:?}) ---\n{}",
                f.severity,
                inputs,
                render(&small, None)
            );
            return;
        }
    }
    println!("seed {} produced no finding", seed);
}

/// The long sweep. Not part of the default run.
#[test]
#[ignore]
fn extended_sweep() {
    let findings = sweep(1, 20_000);
    let mut wrong = 0;
    let mut refusal = 0;
    let mut fold = 0;
    let mut unattributed = 0;
    let mut parse_fail = 0;
    for (seed, f) in &findings {
        match f.severity {
            Severity::WrongValue => {
                wrong += 1;
                if wrong <= 5 {
                    println!("{}", report(*seed, f));
                }
            }
            Severity::OverRefusal => {
                refusal += 1;
                if f.detail.contains("UNATTRIBUTED") {
                    unattributed += 1;
                    if unattributed <= 5 {
                        println!("{}", report(*seed, f));
                    }
                }
            }
            Severity::FoldingDivergence => {
                fold += 1;
                if fold <= 5 {
                    println!("{}", report(*seed, f));
                }
            }
            Severity::ParseFailure => {
                parse_fail += 1;
                if parse_fail <= 3 {
                    println!("{}", report(*seed, f));
                }
            }
        }
    }
    println!(
        "\n20,000 programs: {} wrong-value, {} parse failure, {} folding divergence, \
         {} over-refusal ({} of them UNATTRIBUTED)",
        wrong, parse_fail, fold, refusal, unattributed
    );
    assert_eq!(wrong, 0, "wrong-value findings");
    assert_eq!(parse_fail, 0, "parse failures");
}
