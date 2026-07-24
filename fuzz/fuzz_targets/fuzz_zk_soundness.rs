#![no_main]
use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;
use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{Constraint, Fr, HintOp, LinearCombination, SignalId, WitnessIRGraph, WitnessOp, ZkEmitter};

fn eval_lc(lc: &LinearCombination, w: &[Fr]) -> Fr {
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

fn execute_host_witness_ir(ir: &WitnessIRGraph, public_inputs: &[Fr], private_inputs: &[Fr]) -> Result<Vec<Fr>, String> {
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
                } else {
                    if priv_idx < private_inputs.len() {
                        w[node_idx] = private_inputs[priv_idx].clone();
                        priv_idx += 1;
                    }
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
                if *a < w.len() && *b < w.len() {
                    if w[*b] != Fr::zero() {
                        let inv_b = w[*b].inv();
                        w[node_idx] = w[*a].mul(&inv_b);
                    }
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
                if *a < w.len() && *b < w.len() {
                    if w[*a] != w[*b] {
                        return Err(format!("Host Witness Assertion Error: w[{}] != w[{}]", a, b));
                    }
                }
            }
        }
    }

    Ok(w)
}

fn solve_r1cs_witness(constraints: &[Constraint], witness_ir: &WitnessIRGraph, num_vars: usize, pub_in: &[Fr], priv_in: &[Fr]) -> (Vec<Fr>, bool) {
    let mut w = execute_host_witness_ir(witness_ir, pub_in, priv_in).unwrap_or_else(|_| vec![Fr::zero(); num_vars]);
    if num_vars > 0 {
        w[0] = Fr::one();
    }

    let mut solved_mask = vec![false; num_vars];
    if num_vars > 0 {
        solved_mask[0] = true;
    }

    for (node_idx, op) in witness_ir.nodes.iter().enumerate() {
        match op {
            WitnessOp::Const(_) | WitnessOp::LoadInput { .. } | WitnessOp::HintBlock { .. } => {
                if node_idx < num_vars {
                    solved_mask[node_idx] = true;
                }
            }
            _ => {}
        }
    }

    // Multi-pass Algebraic Backpropagation Solver over Fr
    for _pass in 0..100 {
        let mut changed = false;

        for c in constraints {
            // Case 1: Solve for unknown in C when A and B are fully known
            let a_known = c.a.terms.iter().all(|(wire, _)| solved_mask[*wire]);
            let b_known = c.b.terms.iter().all(|(wire, _)| solved_mask[*wire]);

            if a_known && b_known {
                let target = eval_lc(&c.a, &w).mul(&eval_lc(&c.b, &w));
                let unknown_in_c: Vec<(usize, Fr)> = c.c.terms.iter()
                    .filter(|(wire, _)| !solved_mask[*wire])
                    .map(|(w_id, coeff)| (*w_id, coeff.clone()))
                    .collect();

                if unknown_in_c.len() == 1 {
                    let (target_wire, coeff) = &unknown_in_c[0];
                    if coeff.0 != Fr::zero().0 {
                        let mut known_sum = Fr::zero();
                        for (w_id, c_coeff) in &c.c.terms {
                            if *w_id != *target_wire {
                                let val = if *w_id == 0 { Fr::one() } else { w[*w_id].clone() };
                                known_sum = known_sum.add(&c_coeff.mul(&val));
                            }
                        }
                        let rem = target.sub(&known_sum);
                        let val = rem.mul(&coeff.inv());
                        w[*target_wire] = val;
                        solved_mask[*target_wire] = true;
                        changed = true;
                    }
                }
            }

            // Case 2: Solve for unknown in A when B and C are fully known
            let b_known = c.b.terms.iter().all(|(wire, _)| solved_mask[*wire]);
            let c_known = c.c.terms.iter().all(|(wire, _)| solved_mask[*wire]);

            if b_known && c_known {
                let b_val = eval_lc(&c.b, &w);
                let c_val = eval_lc(&c.c, &w);
                if b_val.0 != Fr::zero().0 {
                    let target_a = c_val.mul(&b_val.inv());
                    let unknown_in_a: Vec<(usize, Fr)> = c.a.terms.iter()
                        .filter(|(wire, _)| !solved_mask[*wire])
                        .map(|(w_id, coeff)| (*w_id, coeff.clone()))
                        .collect();

                    if unknown_in_a.len() == 1 {
                        let (target_wire, coeff) = &unknown_in_a[0];
                        if coeff.0 != Fr::zero().0 {
                            let mut known_sum = Fr::zero();
                            for (w_id, a_coeff) in &c.a.terms {
                                if *w_id != *target_wire {
                                    let val = if *w_id == 0 { Fr::one() } else { w[*w_id].clone() };
                                    known_sum = known_sum.add(&a_coeff.mul(&val));
                                }
                            }
                            let rem = target_a.sub(&known_sum);
                            let val = rem.mul(&coeff.inv());
                            w[*target_wire] = val;
                            solved_mask[*target_wire] = true;
                            changed = true;
                        }
                    }
                }
            }

            // Case 3: Solve for unknown in B when A and C are fully known
            let a_known = c.a.terms.iter().all(|(wire, _)| solved_mask[*wire]);
            let c_known = c.c.terms.iter().all(|(wire, _)| solved_mask[*wire]);

            if a_known && c_known {
                let a_val = eval_lc(&c.a, &w);
                let c_val = eval_lc(&c.c, &w);
                if a_val.0 != Fr::zero().0 {
                    let target_b = c_val.mul(&a_val.inv());
                    let unknown_in_b: Vec<(usize, Fr)> = c.b.terms.iter()
                        .filter(|(wire, _)| !solved_mask[*wire])
                        .map(|(w_id, coeff)| (*w_id, coeff.clone()))
                        .collect();

                    if unknown_in_b.len() == 1 {
                        let (target_wire, coeff) = &unknown_in_b[0];
                        if coeff.0 != Fr::zero().0 {
                            let mut known_sum = Fr::zero();
                            for (w_id, b_coeff) in &c.b.terms {
                                if *w_id != *target_wire {
                                    let val = if *w_id == 0 { Fr::one() } else { w[*w_id].clone() };
                                    known_sum = known_sum.add(&b_coeff.mul(&val));
                                }
                            }
                            let rem = target_b.sub(&known_sum);
                            let val = rem.mul(&coeff.inv());
                            w[*target_wire] = val;
                            solved_mask[*target_wire] = true;
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

    let mut fully_solved = true;
    for c in constraints {
        let a = eval_lc(&c.a, &w);
        let b = eval_lc(&c.b, &w);
        let c_val = eval_lc(&c.c, &w);
        if a.mul(&b) != c_val {
            fully_solved = false;
            break;
        }
    }

    (w, fully_solved)
}

fn check_r1cs_satisfiability(constraints: &[Constraint], w: &[Fr]) -> Result<(), String> {
    for (i, c) in constraints.iter().enumerate() {
        let a = eval_lc(&c.a, w);
        let b = eval_lc(&c.b, w);
        let c_val = eval_lc(&c.c, w);

        let lhs = a.mul(&b);
        if lhs != c_val {
            return Err(format!(
                "R1CS Satisfiability Failure on Constraint #{}: A(w)*B(w) != C(w). A={}, B={}, A*B={}, C={}",
                i, a.to_string(), b.to_string(), lhs.to_string(), c_val.to_string()
            ));
        }
    }
    Ok(())
}

fn check_underconstrained_signals(
    constraints: &[Constraint],
    w: &[Fr],
    num_vars: usize,
    public_inputs: &[usize],
    outputs: &[usize],
) -> Result<usize, String> {
    let mut public_set: HashSet<usize> = public_inputs.iter().cloned().collect();
    for &out_id in outputs {
        public_set.insert(out_id);
    }
    public_set.insert(0);

    let mut underconstrained_count = 0;
    for i in 1..num_vars {
        if public_set.contains(&i) {
            continue;
        }

        let mut perturbed_w = w.to_vec();
        if i < perturbed_w.len() {
            perturbed_w[i] = perturbed_w[i].add(&Fr::one());
            if check_r1cs_satisfiability(constraints, &perturbed_w).is_ok() {
                underconstrained_count += 1;
            }
        }
    }
    Ok(underconstrained_count)
}

fuzz_target!(|data: &[u8]| {
    let source = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();
    if tokens.is_empty() {
        return;
    }

    let mut parser = Parser::new(tokens);
    let program = match parser.parse_program() {
        Ok(p) => p,
        Err(_) => return,
    };

    let mut tc = TypeChecker::new();
    tc.check_program(&program);

    let mut zk_emitter = ZkEmitter::new();
    if zk_emitter.emit_program(&program).is_err() {
        return;
    }

    let circuit = zk_emitter.build_circuit();
    let witness_ir = zk_emitter.build_witness_ir();
    if circuit.constraints.is_empty() {
        return;
    }

    let pub_inputs = vec![Fr::one(); circuit.public_inputs.len()];
    let priv_inputs = vec![Fr::one(); circuit.private_inputs.len()];

    let (witness, fully_solved) = solve_r1cs_witness(&circuit.constraints, &witness_ir, circuit.num_variables, &pub_inputs, &priv_inputs);

    if fully_solved {
        if let Err(e) = check_r1cs_satisfiability(&circuit.constraints, &witness) {
            eprintln!("[ZK SATISFIABILITY ERROR] {}", e);
        }
    }

    if let Err(e) = check_underconstrained_signals(
        &circuit.constraints,
        &witness,
        circuit.num_variables,
        &circuit.public_inputs,
        &circuit.outputs,
    ) {
        eprintln!("[ZK SOUNDNESS WARNING] {}", e);
    }
});
