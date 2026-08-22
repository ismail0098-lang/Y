#![no_main]
use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;
use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{Constraint, Fr, ZkEmitter};
// Witness generation and satisfiability checking live in the library now, so
// this fuzzer and tests/zk_groth16_end_to_end.rs exercise ONE implementation.
// A solver here that disagreed with the one the proof test uses would make
// both results meaningless.
use y::zk_witness::{check_r1cs_satisfiability, solve_r1cs_witness};

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

    // These were `eprintln!`s, which meant this target COULD NOT FAIL. libFuzzer
    // only registers a finding on a panic or a sanitizer trap, so a run over a
    // backend emitting unsatisfiable circuits printed warnings into the log and
    // exited 0. A fuzz target that cannot fail is a fuzz target that is not
    // testing anything - the same shape as the `_ =>` arms in CLAUDE.md's design
    // rule, one layer out: the obligation was reported to a channel nobody
    // checks.
    if fully_solved {
        if let Err(e) = check_r1cs_satisfiability(&circuit.constraints, &witness) {
            panic!(
                "the witness solver reported success but the witness does not \
                 satisfy its own circuit: {}\n--- program ---\n{}",
                e, source
            );
        }
    }

    // `check_underconstrained_signals` returns `Ok(count)` on every path and
    // never `Err`, so the `if let Err(..)` this replaces was dead code: the
    // count was computed and thrown away, and the scan had never reported
    // anything in its life.
    //
    // It is still not a failure condition, and that is deliberate rather than
    // unfinished. A signal that can be perturbed while the circuit stays
    // satisfiable is not necessarily a bug - some wires are legitimately free -
    // so gating on a non-zero count needs a measured baseline for what this
    // corpus produces, which nothing has established. Reporting it keeps the
    // number visible without asserting a threshold nobody has justified.
    let underconstrained = check_underconstrained_signals(
        &circuit.constraints,
        &witness,
        circuit.num_variables,
        &circuit.public_inputs,
        &circuit.outputs,
    )
    .unwrap_or(0);
    if underconstrained > 0 {
        eprintln!(
            "[ZK SOUNDNESS] {} under-constrained signals (not gated - see comment)",
            underconstrained
        );
    }
});
