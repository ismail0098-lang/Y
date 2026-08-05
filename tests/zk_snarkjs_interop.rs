//! Y's `.r1cs` and `.wtns` must agree with each other, and with iden3's format.
//!
//! Y writes its constraint system in snarkjs's binary format, and has since
//! before anything could check that claim. It holds - `snarkjs r1cs info`,
//! `groth16 setup`, `prove` and `verify` all accept Y's output, so this:
//!
//! ```bash
//! Y circuit.ysu --target=r1cs --witness input.json
//! snarkjs groth16 setup circuit.r1cs pot_final.ptau circuit.zkey
//! snarkjs groth16 prove circuit.zkey circuit.wtns proof.json public.json
//! snarkjs groth16 verify verification_key.json public.json proof.json   # OK!
//! ```
//!
//! works, which is the whole adoption path: Y replaces circom in a pipeline
//! people already run.
//!
//! What nearly broke it is what this file guards. snarkjs requires a specific
//! wire layout - `1`, outputs, public inputs, private inputs, intermediates -
//! and Y allocates wires in whatever order the emitter needs them. The `.r1cs`
//! writer has always permuted its constraint terms into iden3 order. The first
//! `.wtns` writer did not permute the witness, so both files were internally
//! consistent, Y's own solver was perfectly happy, and `snarkjs wtns check`
//! rejected the pair at constraint 1.
//!
//! That failure is invisible from inside Y: every check Y can run works in Y's
//! own numbering. So the test below deliberately re-derives satisfiability in
//! the OTHER numbering.
#![cfg(feature = "zk")]

use std::collections::HashMap;

use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{BigUint, Constraint, Fr, LinearCombination, ZkEmitter};
use y::zk_witness::{check_r1cs_satisfiability, solve_r1cs_witness};

fn remap(lc: &LinearCombination, map: &HashMap<usize, usize>) -> LinearCombination {
    let mut out = LinearCombination::zero();
    for (wire, coeff) in &lc.terms {
        out.add_term(*map.get(wire).expect("every wire is mapped"), coeff.clone());
    }
    out.simplify();
    out
}

/// The permuted constraint system must be satisfied by the permuted witness.
///
/// This is the property `snarkjs wtns check` tests, expressed without snarkjs
/// so it runs in CI. If `snarkjs_wire_map` is ever applied to one file and not
/// the other, or applied inconsistently, this fails.
#[test]
fn r1cs_and_wtns_use_the_same_wire_order() {
    for (src, inputs) in [
        (
            "@unsafe\nfn main(x: I32, y: I32) -> I32 {\n    let mut t = x;\n    for i in 0..8 {\n        t = t * y;\n    }\n    return t;\n}\n",
            vec![3u64, 2],
        ),
        // Gadget-heavy: range checks, bit decomposition and a comparison all
        // allocate wires in orders unrelated to the input/output layout.
        (
            "@unsafe\nfn main(x: I32, y: I32) -> I32 {\n    return ((x / y) & 6) | (x % y);\n}\n",
            vec![123u64, 10],
        ),
        (
            "@unsafe\nfn main(x: I32, y: I32) -> I32 {\n    return poseidon_hash(x, y);\n}\n",
            vec![1u64, 2],
        ),
    ] {
        let tokens = Lexer::new(src).tokenize();
        let program = Parser::new(tokens).parse_program().expect("parse");
        TypeChecker::new().check_program(&program);
        let mut emitter = ZkEmitter::new();
        emitter.emit_program(&program).expect("lower");
        let circuit = emitter.build_circuit();
        let ir = emitter.build_witness_ir();

        let privs: Vec<Fr> = inputs.iter().map(|v| Fr(BigUint::from_u64(*v))).collect();
        let (witness, ok) =
            solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);
        assert!(ok, "witness does not satisfy the circuit in Y's own numbering");

        let (map, n_out, n_pub, n_prv) = ZkEmitter::snarkjs_wire_map(&circuit);

        // A permutation, not merely a mapping: every wire once, no collisions.
        assert_eq!(map.len(), circuit.num_variables, "not every wire is mapped");
        let mut seen = vec![false; circuit.num_variables];
        for (_, new) in &map {
            assert!(*new < circuit.num_variables, "mapped wire {} out of range", new);
            assert!(!seen[*new], "two wires map onto {}", new);
            seen[*new] = true;
        }
        assert_eq!(map[&0], 0, "the constant-1 wire must stay at index 0");

        // The header counts must describe the layout the map produced.
        assert_eq!(n_out, circuit.outputs.len());
        assert_eq!(n_pub, circuit.public_inputs.len());
        assert_eq!(n_prv, circuit.private_inputs.len());
        for (i, w) in circuit.outputs.iter().enumerate() {
            assert_eq!(map[w], 1 + i, "outputs must follow the constant wire");
        }
        for (i, w) in circuit.private_inputs.iter().enumerate() {
            assert_eq!(
                map[w],
                1 + n_out + n_pub + i,
                "private inputs must follow the public ones"
            );
        }

        // Now the real check, in iden3's numbering.
        let permuted_constraints: Vec<Constraint> = circuit
            .constraints
            .iter()
            .map(|c| Constraint {
                a: remap(&c.a, &map),
                b: remap(&c.b, &map),
                c: remap(&c.c, &map),
                span: c.span.clone(),
            })
            .collect();

        let mut permuted_witness = vec![Fr::zero(); witness.len()];
        for (old, new) in &map {
            permuted_witness[*new] = witness[*old].clone();
        }

        check_r1cs_satisfiability(&permuted_constraints, &permuted_witness).unwrap_or_else(|e| {
            panic!(
                "permuted system is unsatisfiable ({}). The .wtns and .r1cs writers have \
                 diverged on wire order - snarkjs will reject the pair even though Y's own \
                 checks pass.",
                e
            )
        });
    }
}
