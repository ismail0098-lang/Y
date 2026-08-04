//! Witness generation and R1CS satisfiability checking for the ZK backend.
//!
//! `zk_emitter` produces the CONSTRAINTS (the circuit); this module produces
//! the ASSIGNMENT that satisfies them, and checks that it does. Both halves are
//! needed before anything can be proved: a Groth16 prover consumes an R1CS plus
//! a full witness vector, and an R1CS on its own proves nothing.
//!
//! This code previously lived inside `fuzz/fuzz_targets/fuzz_zk_soundness.rs`,
//! which meant the only thing that could generate a witness was the fuzzer, and
//! the only place satisfiability was ever checked was a fuzzing run - not
//! `cargo test`. It is library code now so that the fuzzer and the Groth16
//! end-to-end test (`tests/zk_groth16_end_to_end.rs`) share ONE implementation;
//! a witness solver that disagrees with the one the soundness fuzzer uses would
//! make both results meaningless.
//!
//! The solver is deliberately a general algebraic back-propagation pass rather
//! than a straight-line evaluator, because `WitnessIRGraph` does not always
//! determine every signal directly - constraints introduced by the lowering
//! (and by hint blocks) can leave a wire that is only pinned by the R1CS
//! itself. `solve_r1cs_witness` reports whether it reached a fully satisfying
//! assignment rather than asserting it, so callers can distinguish "this
//! circuit is unsatisfiable" from "this solver could not finish it".

use crate::zk_emitter::{Constraint, Fr, HintOp, LinearCombination, SignalId, WitnessIRGraph, WitnessOp};

/// Evaluates a linear combination against a witness vector.
///
/// Wire 0 is the constant `1` by R1CS convention. Out-of-range wires evaluate
/// to zero rather than panicking: the fuzzer reaches this with deliberately
/// malformed circuits, and a panic there would be a fuzz crash rather than the
/// soundness finding it is looking for.
pub fn eval_lc(lc: &LinearCombination, w: &[Fr]) -> Fr {
    let mut sum = Fr::zero();
    for (wire_id, coeff) in &lc.terms {
        let val = if *wire_id == 0 {
            Fr::one()
        } else if *wire_id < w.len() {
            w[*wire_id].clone()
        } else {
            Fr::zero()
        };
        sum = sum.add(&coeff.mul(&val));
    }
    sum
}

/// Runs the witness IR straight through, producing the signals it determines
/// directly from the inputs. Returns `Err` only for an explicit `AssertEq`
/// violation, which means the inputs themselves are inconsistent with the
/// circuit rather than that the solver fell short.
pub fn execute_host_witness_ir(
    ir: &WitnessIRGraph,
    public_inputs: &[Fr],
    private_inputs: &[Fr],
) -> Result<Vec<Fr>, String> {
    let mut w = vec![Fr::zero(); ir.num_signals];
    if ir.num_signals > 0 {
        w[0] = Fr::one();
    }

    let mut pub_idx = 0;
    let mut priv_idx = 0;

    for (node_idx, op) in ir.nodes.iter().enumerate() {
        if node_idx == 0 {
            continue;
        }

        match op {
            WitnessOp::Const(val) => {
                w[node_idx] = Fr(val.clone());
            }
            WitnessOp::LoadInput { is_public, .. } => {
                if *is_public {
                    if pub_idx < public_inputs.len() {
                        w[node_idx] = public_inputs[pub_idx].clone();
                        pub_idx += 1;
                    }
                } else if priv_idx < private_inputs.len() {
                    w[node_idx] = private_inputs[priv_idx].clone();
                    priv_idx += 1;
                }
            }
            WitnessOp::Add(SignalId(a), SignalId(b)) => {
                if *a < w.len() && *b < w.len() {
                    w[node_idx] = w[*a].add(&w[*b]);
                }
            }
            WitnessOp::Sub(SignalId(a), SignalId(b)) => {
                if *a < w.len() && *b < w.len() {
                    w[node_idx] = w[*a].sub(&w[*b]);
                }
            }
            WitnessOp::Mul(SignalId(a), SignalId(b)) => {
                if *a < w.len() && *b < w.len() {
                    w[node_idx] = w[*a].mul(&w[*b]);
                }
            }
            WitnessOp::Div(SignalId(a), SignalId(b)) => {
                if *a < w.len() && *b < w.len() && w[*b] != Fr::zero() {
                    let inv_b = w[*b].inv();
                    w[node_idx] = w[*a].mul(&inv_b);
                }
            }
            WitnessOp::Inv(SignalId(a)) => {
                if *a < w.len() && w[*a] != Fr::zero() {
                    w[node_idx] = w[*a].inv();
                }
            }
            WitnessOp::HintBlock { ops, .. } => {
                for hint in ops {
                    match hint {
                        HintOp::NonDeterministicInv { src: SignalId(s), dst: SignalId(d) } => {
                            if *s < w.len() && *d < w.len() && w[*s] != Fr::zero() {
                                w[*d] = w[*s].inv();
                            }
                        }
                        HintOp::AssignExpr { src: SignalId(s), dst: SignalId(d) } => {
                            if *s < w.len() && *d < w.len() {
                                w[*d] = w[*s].clone();
                            }
                        }
                        HintOp::BitDecompose { src: SignalId(s), dst_bits } => {
                            if *s < w.len() {
                                let val_is_zero = w[*s].0.is_zero();
                                for dst in dst_bits {
                                    if dst.0 < w.len() {
                                        w[dst.0] = if val_is_zero { Fr::zero() } else { Fr::one() };
                                    }
                                }
                            }
                        }
                    }
                }
            }
            WitnessOp::AssertEq(SignalId(a), SignalId(b)) => {
                if *a < w.len() && *b < w.len() && w[*a] != w[*b] {
                    return Err(format!("Host Witness Assertion Error: w[{}] != w[{}]", a, b));
                }
            }

            // ---- linear-combination recipes ----
            // These evaluate against wires solved earlier in the same forward
            // pass, which is why the emitter must allocate a gadget's inputs
            // before its advice wires. They are what makes `==`, `!=` and the
            // comparisons witnessable at all: their constraints each carry two
            // unknowns, so back-propagation alone can never pin them.
            WitnessOp::IsZeroLc(lc) => {
                let v = eval_lc(lc, &w);
                w[node_idx] = if v.0.is_zero() { Fr::one() } else { Fr::zero() };
            }
            WitnessOp::InvOrZeroLc(lc) => {
                let v = eval_lc(lc, &w);
                w[node_idx] = if v.0.is_zero() { Fr::zero() } else { v.inv() };
            }
            // Deliberately leaves the wire at zero and UNSOLVED, so the
            // back-propagation pass owns it. See `WitnessOp::Unknown`.
            WitnessOp::Unknown => {}
            WitnessOp::BitOfLc { lc, bit } => {
                let v = eval_lc(lc, &w);
                w[node_idx] = if v.0.get_bit(*bit as usize) { Fr::one() } else { Fr::zero() };
            }
            WitnessOp::MulLc(a, b) => {
                w[node_idx] = eval_lc(a, &w).mul(&eval_lc(b, &w));
            }
            WitnessOp::IntDivLc(a, b) => {
                let (bv, av) = (eval_lc(b, &w), eval_lc(a, &w));
                w[node_idx] = if bv.0.is_zero() {
                    Fr::zero()
                } else {
                    Fr(av.0.div_mod(&bv.0).0)
                };
            }
            WitnessOp::IntModLc(a, b) => {
                let (bv, av) = (eval_lc(b, &w), eval_lc(a, &w));
                w[node_idx] = if bv.0.is_zero() {
                    Fr::zero()
                } else {
                    Fr(av.0.div_mod(&bv.0).1)
                };
            }
        }
    }

    Ok(w)
}

/// Seeds from the witness IR, then back-propagates through the R1CS to pin any
/// remaining wires.
///
/// Each constraint `A*B = C` can determine one unknown if the other two linear
/// combinations are fully known and exactly one term in the third is not - so
/// the pass runs to a fixed point (capped at 100 sweeps; every real circuit
/// seen so far converges in far fewer, and the cap keeps a pathological fuzz
/// input from hanging).
///
/// The returned flag is whether the assignment actually satisfies every
/// constraint, which is NOT the same as "the solver terminated": an
/// unsatisfiable circuit and a circuit this solver cannot finish both come back
/// `false`, and callers that need to distinguish them must say so themselves.
pub fn solve_r1cs_witness(
    constraints: &[Constraint],
    witness_ir: &WitnessIRGraph,
    num_vars: usize,
    pub_in: &[Fr],
    priv_in: &[Fr],
) -> (Vec<Fr>, bool) {
    let mut w = execute_host_witness_ir(witness_ir, pub_in, priv_in)
        .unwrap_or_else(|_| vec![Fr::zero(); num_vars]);
    w.resize(num_vars.max(w.len()), Fr::zero());
    if num_vars > 0 {
        w[0] = Fr::one();
    }

    let mut solved_mask = vec![false; num_vars.max(w.len())];
    if !solved_mask.is_empty() {
        solved_mask[0] = true;
    }

    // Everything the forward pass above actually computed counts as solved.
    // The linear-combination recipes MUST be included: they are computed
    // directly, but if they are left unmarked then any constraint referencing a
    // gadget's advice wire looks like it still has an unknown, back-propagation
    // refuses to fire on it, and the wire it would have determined - typically
    // the circuit's own output - stays at zero. That is subtle to spot, because
    // it only shows up when the correct answer is NOT zero.
    // `Unknown` is deliberately excluded; that is the whole point of it.
    for (node_idx, op) in witness_ir.nodes.iter().enumerate() {
        if let WitnessOp::Const(_)
        | WitnessOp::LoadInput { .. }
        | WitnessOp::HintBlock { .. }
        | WitnessOp::IsZeroLc(_)
        | WitnessOp::InvOrZeroLc(_)
        | WitnessOp::BitOfLc { .. }
        | WitnessOp::MulLc(..)
        | WitnessOp::IntDivLc(..)
        | WitnessOp::IntModLc(..) = op
        {
            if node_idx < solved_mask.len() {
                solved_mask[node_idx] = true;
            }
        }
    }

    // Fast path: the forward pass frequently determines the entire circuit on
    // its own, and when it does there is nothing left to back-propagate.
    //
    // This is worth a lot. Measured on the 50,000-constraint polynomial
    // circuit: the forward pass costs 0.36s and already satisfies every
    // constraint, but running the sweep anyway cost **52.87s** - back
    // propagation rediscovering, one constraint at a time, values that were
    // already correct. Witness generation was ~99% of the whole prove pipeline
    // and essentially all of it was this.
    //
    // Checking is not a heuristic: an assignment that satisfies every
    // constraint IS a valid witness, whatever produced it. So this cannot
    // accept anything the sweep would have rejected - it only skips work.
    if check_r1cs_satisfiability(constraints, &w).is_ok() {
        return (w, true);
    }

    let known = |lc: &LinearCombination, mask: &[bool]| {
        lc.terms.iter().all(|(wire, _)| *wire < mask.len() && mask[*wire])
    };
    // Sum of every term except `skip`, i.e. the part of the LC already pinned.
    let known_sum = |lc: &LinearCombination, skip: usize, w: &[Fr]| {
        let mut acc = Fr::zero();
        for (w_id, coeff) in &lc.terms {
            if *w_id != skip {
                let val = if *w_id == 0 { Fr::one() } else { w[*w_id].clone() };
                acc = acc.add(&coeff.mul(&val));
            }
        }
        acc
    };

    for _pass in 0..100 {
        let mut changed = false;

        for c in constraints {
            // Solve the single unknown in C from a known A*B.
            if known(&c.a, &solved_mask) && known(&c.b, &solved_mask) {
                let target = eval_lc(&c.a, &w).mul(&eval_lc(&c.b, &w));
                if let Some((wire, coeff)) = sole_unknown(&c.c, &solved_mask) {
                    if coeff.0 != Fr::zero().0 {
                        let rem = target.sub(&known_sum(&c.c, wire, &w));
                        w[wire] = div_by(&rem, &coeff);
                        solved_mask[wire] = true;
                        changed = true;
                    }
                }
            }

            // Solve the single unknown in A from C/B, and symmetrically in B.
            for (num, den, target_lc) in [(&c.c, &c.b, &c.a), (&c.c, &c.a, &c.b)] {
                if known(den, &solved_mask) && known(num, &solved_mask) {
                    let den_val = eval_lc(den, &w);
                    if den_val.0 == Fr::zero().0 {
                        continue;
                    }
                    let target = div_by(&eval_lc(num, &w), &den_val);
                    if let Some((wire, coeff)) = sole_unknown(target_lc, &solved_mask) {
                        if coeff.0 != Fr::zero().0 {
                            let rem = target.sub(&known_sum(target_lc, wire, &w));
                            w[wire] = div_by(&rem, &coeff);
                            solved_mask[wire] = true;
                            changed = true;
                        }
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    let satisfied = check_r1cs_satisfiability(constraints, &w).is_ok();
    (w, satisfied)
}

/// `rem / coeff`, skipping the modular inversion when the coefficient is 1.
///
/// It essentially always is: the wire being solved for appears in its own
/// constraint with coefficient 1. That matters a lot, because `Fr::inv` is
/// extended Euclid over a hand-rolled `BigUint` - a big division per iteration,
/// ~380 iterations for a 254-bit modulus. Paying it once per solved wire made
/// witness generation cost ~1.07 ms per constraint, roughly 300x the cost of
/// EMITTING a constraint, and put ~99% of the end-to-end proving pipeline in
/// this one call.
#[inline]
fn div_by(rem: &Fr, coeff: &Fr) -> Fr {
    if *coeff == Fr::one() {
        rem.clone()
    } else {
        rem.mul(&coeff.inv())
    }
}

/// The one unsolved term of `lc`, if there is exactly one.
fn sole_unknown(lc: &LinearCombination, mask: &[bool]) -> Option<(usize, Fr)> {
    let mut found: Option<(usize, Fr)> = None;
    for (wire, coeff) in &lc.terms {
        if *wire < mask.len() && mask[*wire] {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some((*wire, coeff.clone()));
    }
    found
}

/// Checks `A(w) * B(w) == C(w)` for every constraint.
///
/// This is the definition of "the witness satisfies the circuit"; a Groth16
/// proof over a witness that fails this cannot verify, so it is the natural
/// gate to run before ever invoking a prover.
pub fn check_r1cs_satisfiability(constraints: &[Constraint], w: &[Fr]) -> Result<(), String> {
    for (i, c) in constraints.iter().enumerate() {
        let a = eval_lc(&c.a, w);
        let b = eval_lc(&c.b, w);
        let c_val = eval_lc(&c.c, w);

        let lhs = a.mul(&b);
        if lhs != c_val {
            return Err(format!(
                "R1CS Satisfiability Failure on Constraint #{}: A(w)*B(w) != C(w). A={}, B={}, A*B={}, C={}",
                i,
                a.to_string(),
                b.to_string(),
                lhs.to_string(),
                c_val.to_string()
            ));
        }
    }
    Ok(())
}
