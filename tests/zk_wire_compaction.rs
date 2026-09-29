//! Wire compaction: `optimize_circuit`'s last pass.
//!
//! The two reduction passes before it both abandon wires. `substitute_linear_
//! constraints` deletes the constraint that defined an intermediate;
//! `dedup_identical_products` points two products at one wire and abandons the
//! loser. Neither renumbers - they run to a fixpoint over arrays indexed by wire
//! id, so renumbering mid-round would renumber under them - and the result was
//! that Y emitted 1.86x fewer constraints than circom for the same circuit while
//! carrying ~50% MORE wires. Groth16 has a proving-key element per wire and an
//! MSM scalar per wire, so every one of those cost a curve operation in every
//! proof, forever.
//!
//! Compaction drops the wires nothing refers to any more and renumbers the rest
//! densely. Its soundness argument is the dual of the substitution pass's: a
//! wire in no surviving constraint is unconstrained and existentially
//! quantified, so projecting it away preserves satisfiability both ways, and the
//! boundary - `1`, the inputs, the outputs - is marked live outright and is
//! exactly what a verifier sees.
//!
//! **Renumbering is the operation with the worst failure mode in this emitter**,
//! and it has already gone wrong once: `optimize_circuit` renamed wires in the
//! constraints and not in the witness recipes, and gadget circuits silently
//! became unprovable (see `zk_cse_gadget_wires.rs`). So the tests here are not
//! "did the wire count go down". They are:
//!
//! 1. a control, so that "compact nothing" cannot pass the file,
//! 2. the circuit still computes the same values - pinned against circomlib,
//! 3. the compacted circuit still *binds* its output,
//! 4. nothing dead survives, which is the pass's actual postcondition,
//! 5. the boundary keeps its identity AND its order - the inputs are consumed
//!    positionally in wire order, so a map that reordered them would feed the
//!    circuit its arguments shuffled with every wire still defined.
//!
//! Run with:  cargo test --features zk --test zk_wire_compaction

#![cfg(feature = "zk")]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use y::circom_lower::compile_file;
use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{set_wire_compaction, Circuit, ZkEmitter};
use y::zk_field::Fr;
use y::zk_witness::{check_r1cs_satisfiability, solve_r1cs_witness};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/circom").join(name)
}

fn circomlib() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("circomlib/circuits")
}

struct Run {
    circuit: Circuit,
    witness: Vec<Fr>,
    satisfied: bool,
    out: usize,
}

fn finish(emitter: ZkEmitter, inputs: &[u64]) -> Run {
    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();
    let privs: Vec<Fr> = inputs.iter().map(|v| Fr::from_u64(*v)).collect();
    let (witness, satisfied) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);
    let out = *circuit.outputs.first().expect("circuit has no output signal");
    Run { circuit, witness, satisfied, out }
}

/// Compile and solve with compaction `on` for this thread.
///
/// Thread-local for the same reason the other two knobs are: the alternative is
/// `env::set_var`, which races every other test in the binary.
fn run(case: &Case, on: bool) -> Run {
    set_wire_compaction(on);
    let r = match case.src {
        Src::Circom(name) => {
            let e = compile_file(&fixture(name), &[circomlib()])
                .unwrap_or_else(|e| panic!("{} failed to compile: {}", name, e));
            finish(e, case.inputs)
        }
        Src::Y(src) => {
            let tokens = Lexer::new(src).tokenize();
            let program = Parser::new(tokens).parse_program().expect("parse");
            TypeChecker::new().check_program(&program);
            let mut e = ZkEmitter::new();
            // `ZkEmitter::new` re-reads the environment, so the thread-local has
            // to be set again after it.
            set_wire_compaction(on);
            e.emit_program(&program).expect("lower");
            finish(e, case.inputs)
        }
    };
    set_wire_compaction(true);
    r
}

enum Src {
    Circom(&'static str),
    Y(&'static str),
}

struct Case {
    name: &'static str,
    src: Src,
    inputs: &'static [u64],
    /// Whether this circuit has any dead wire to drop at all.
    shrinks: bool,
}

/// Both front ends, and that is not redundant coverage.
///
/// They allocate the boundary at opposite ends of the wire table. circom
/// declares `signal output out;` at the top of a template, so the outputs and
/// inputs are a low, entirely-live prefix and compaction maps them to
/// themselves - which means a bug that forgot to renumber the boundary lists is
/// INVISIBLE on circom input. Y binds its return value last (`<lc> * 1 = out`),
/// so its output wire sits above thousands of intermediates and moves. Verified
/// by mutation: dropping the boundary remap passes every circom case and fails
/// the Y ones.
fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "multiplier.circom",
            src: Src::Circom("multiplier.circom"),
            inputs: &[3, 5],
            shrinks: false,
        },
        Case {
            name: "sum_squares.circom",
            src: Src::Circom("sum_squares.circom"),
            inputs: &[1, 2, 3, 4],
            shrinks: true,
        },
        Case {
            name: "linear_chain.circom",
            src: Src::Circom("linear_chain.circom"),
            inputs: &[6],
            shrinks: true,
        },
        Case {
            name: "poseidon2.circom",
            src: Src::Circom("poseidon2.circom"),
            inputs: &[1, 2],
            shrinks: true,
        },
        Case {
            name: "lessthan.circom",
            src: Src::Circom("lessthan.circom"),
            inputs: &[3, 9],
            shrinks: true,
        },
        Case {
            name: "num2bits_sum.circom",
            src: Src::Circom("num2bits_sum.circom"),
            inputs: &[19],
            shrinks: true,
        },
        // An input that appears in no constraint at all. Both front ends allow
        // it, and it is the one boundary wire that liveness-by-constraint-scan
        // would miss.
        Case {
            name: "unused_input.circom",
            src: Src::Circom("unused_input.circom"),
            inputs: &[6, 99],
            shrinks: false,
        },
        Case {
            // `y` is unused and must survive - that is what
            // `unused_boundary_signals_survive` pins. It DOES shrink by one now:
            // the pure-copy rename collapses `mul_tmp * 1 = out`, leaving
            // `mul_tmp` dead for compaction to drop. The boundary is untouched,
            // and `multiplier.circom` and `unused_input.circom` still exercise
            // the `shrinks: false` arm.
            name: "y/unused_param",
            src: Src::Y("@unsafe\nfn main(x: I32, y: I32) -> I32 {\n    return x * x;\n}\n"),
            inputs: &[6, 99],
            shrinks: true,
        },
        // Y native. `x*y` twice is a CSE pair, so the loser is dead and the
        // output wire is allocated after it - exactly the arrangement that makes
        // an un-renumbered boundary list observable.
        Case {
            name: "y/cse_pair",
            src: Src::Y("@unsafe\nfn main(x: I32, y: I32) -> I32 {\n    let a = x * y;\n    let b = x * y;\n    return a + b;\n}\n"),
            inputs: &[3, 5],
            shrinks: true,
        },
        // A gadget, so the witness recipes carry linear combinations that the
        // renumbering has to rewrite. `==` is the is-zero gadget.
        Case {
            name: "y/is_zero",
            src: Src::Y("@unsafe\nfn main(x: I32, y: I32) -> I32 {\n    let a = x * y;\n    let b = x * y;\n    if a == b { return 1; }\n    return 0;\n}\n"),
            inputs: &[3, 5],
            shrinks: true,
        },
        Case {
            name: "y/poseidon_chain",
            src: Src::Y("@unsafe\nfn main(x: I32, y: I32) -> I32 {\n    let mut h = x;\n    for i in 0..8 { h = poseidon_hash(h, y); }\n    return h;\n}\n"),
            inputs: &[3, 7],
            shrinks: true,
        },
    ]
}

/// **The control.** Without this, a pass that dropped nothing would satisfy
/// every other test in this file.
///
/// `multiplier.circom` is the other half of the control and is listed
/// separately: it is `out <== a*b` and has no dead wire to drop, so it must come
/// through at exactly its original size. A pass that "compacted" it would be
/// removing something live.
#[test]
fn the_pass_actually_removes_wires() {
    for case in cases() {
        let name = case.name;
        let off = run(&case, false).circuit.num_variables;
        let on = run(&case, true).circuit.num_variables;
        eprintln!("{:24} {:>7} -> {:>7} wires  ({:.2}x)", name, off, on, off as f64 / on as f64);

        if case.shrinks {
            assert!(
                on < off,
                "{}: compaction dropped nothing ({} wires either way)",
                name,
                off
            );
        } else {
            assert_eq!(on, off, "{}: nothing is dead here, so nothing may go", name);
        }
    }
}

/// Compaction must not change what the circuit computes.
#[test]
fn compaction_preserves_the_computed_value() {
    for case in cases() {
        let name = case.name;
        let off = run(&case, false);
        let on = run(&case, true);
        assert!(off.satisfied, "{}: uncompacted circuit has no witness", name);
        assert!(on.satisfied, "{}: compacted circuit has no witness", name);
        assert_eq!(
            on.witness[on.out].to_decimal_string(),
            off.witness[off.out].to_decimal_string(),
            "{}: compaction changed the circuit's output",
            name
        );
    }
}

/// The compacted circuit must still force its output.
///
/// The failure this exists to catch: dropping a wire the output's defining
/// constraint depended on would leave a circuit that still compiles, still
/// admits a witness, and no longer says anything about `out`. Perturbing the
/// honest witness's output wire has to break satisfiability.
#[test]
fn the_compacted_circuit_still_binds_its_output() {
    for case in cases() {
        let name = case.name;
        let r = run(&case, true);
        assert!(r.satisfied, "{}: no witness", name);
        check_r1cs_satisfiability(&r.circuit.constraints, &r.witness)
            .unwrap_or_else(|e| panic!("{}: honest witness rejected: {}", name, e));

        let mut tampered = r.witness.clone();
        tampered[r.out] = tampered[r.out].add(&Fr::one());
        assert!(
            check_r1cs_satisfiability(&r.circuit.constraints, &tampered).is_err(),
            "{}: the output wire is unconstrained after compaction - changing it \
             still satisfies every constraint",
            name
        );
    }
}

/// The pass's actual postcondition: nothing dead is left.
///
/// Every wire must be the constant, on the boundary, or mentioned by a surviving
/// constraint. This is what says the pass ran to completion rather than dropping
/// the easy half - the control above only proves it dropped *something*.
///
/// If a future gadget introduces a wire that lives only inside a witness recipe
/// and appears in no constraint, this becomes a false failure and the right fix
/// is to widen the expected set to the recipe closure - which is what
/// `compact_wires` itself computes - not to loosen the assertion to an
/// inequality.
#[test]
fn no_dead_wire_survives() {
    for case in cases() {
        let name = case.name;
        let r = run(&case, true);
        let c = &r.circuit;

        let mut reachable: HashSet<usize> = HashSet::new();
        reachable.insert(0);
        reachable.extend(c.public_inputs.iter().copied());
        reachable.extend(c.private_inputs.iter().copied());
        reachable.extend(c.outputs.iter().copied());
        for con in &c.constraints {
            for lc in [&con.a, &con.b, &con.c] {
                reachable.extend(lc.terms.iter().map(|(w, _)| *w));
            }
        }

        assert_eq!(
            reachable.len(),
            c.num_variables,
            "{}: {} wires declared but only {} are reachable - compaction left \
             {} dead",
            name,
            c.num_variables,
            reachable.len(),
            c.num_variables - reachable.len()
        );

        // Dense and zero-based, which is what every consumer assumes: the
        // witness vector is indexed by wire, and `snarkjs_wire_map` builds its
        // permutation by walking `1..num_variables`.
        assert!(
            reachable.iter().all(|w| *w < c.num_variables),
            "{}: a wire id is past the end of the variable table",
            name
        );
        assert_eq!(
            c.variables.len(),
            c.num_variables,
            "{}: the name table and the wire count disagree",
            name
        );
    }
}

/// The boundary keeps its identity and its ORDER.
///
/// Order is the sharp edge here. `execute_host_witness_ir` assigns the public
/// and private inputs positionally in wire order, so a renumbering that
/// reordered them would hand the circuit its arguments shuffled - and every wire
/// would still be defined, every constraint still satisfiable, the proof still
/// valid, for the wrong statement. Names are allocated during emission, before
/// any pass runs, so comparing the name sequence catches a permutation that
/// comparing counts would not.
#[test]
fn the_boundary_keeps_its_identity_and_order() {
    for case in cases() {
        let name = case.name;
        let off = run(&case, false);
        let on = run(&case, true);
        let names = |r: &Run, ws: &[usize]| -> Vec<String> {
            ws.iter().map(|w| r.circuit.variables[*w].clone()).collect()
        };
        for (what, a, b) in [
            ("public inputs", &off.circuit.public_inputs, &on.circuit.public_inputs),
            ("private inputs", &off.circuit.private_inputs, &on.circuit.private_inputs),
            ("outputs", &off.circuit.outputs, &on.circuit.outputs),
        ] {
            assert_eq!(
                names(&off, a),
                names(&on, b),
                "{}: compaction changed the {} - by name, in order",
                name,
                what
            );
        }
    }
}

/// `Poseidon(2)` from unmodified circomlib, compacted, still hashes correctly.
///
/// The end-to-end catch, and the reason the `(7, 0)` vector is in the list: it
/// is asymmetric, so an input permutation that survived every structural
/// assertion above would produce `Poseidon(0, 7)` and fail here.
///
/// The digests are circomlib's own published vectors, pinned identically in
/// `circom_frontend.rs`, `zk_poseidon_interop.rs` and `zk_linear_substitution.rs`.
/// **If they move, the hash has forked - do not update them.**
#[test]
fn compacted_poseidon_still_matches_circomlib() {
    let vectors: &[(u64, u64, &str)] = &[
        (1, 2, "7853200120776062878684798364095072458815029376092732009249414926327459813530"),
        (3, 4, "14763215145315200506921711489642608356394854266165572616578112107564877678998"),
        (0, 0, "14744269619966411208579211824598458697587494354926760081771325075741142829156"),
        (7, 0, "10402197090275139279073177788985849389816807868761640028215734431067655199248"),
    ];
    for (a, b, expected) in vectors {
        let case = Case {
            name: "poseidon2.circom",
            src: Src::Circom("poseidon2.circom"),
            inputs: Box::leak(Box::new([*a, *b])),
            shrinks: true,
        };
        let r = run(&case, true);
        assert!(r.satisfied, "Poseidon({}, {}): no witness", a, b);
        check_r1cs_satisfiability(&r.circuit.constraints, &r.witness)
            .unwrap_or_else(|e| panic!("Poseidon({}, {}): {}", a, b, e));
        assert_eq!(
            r.witness[r.out].to_decimal_string(),
            *expected,
            "the compacted Poseidon(2) disagrees with circomlib"
        );
    }
}

/// Compaction must not disturb the constraints themselves.
///
/// It renumbers wires; it does not delete, merge or reshape a constraint. So the
/// count must be identical with the pass on and off, and so must the total
/// number of non-zero terms. This separates "compaction" from "another
/// reduction pass that happens to run last" - if either number moves, the pass
/// is doing something its soundness argument does not cover.
#[test]
fn compaction_changes_only_the_numbering() {
    for case in cases() {
        let name = case.name;
        let off = run(&case, false).circuit;
        let on = run(&case, true).circuit;
        assert_eq!(
            off.constraints.len(),
            on.constraints.len(),
            "{}: compaction changed the constraint count",
            name
        );
        let terms = |c: &Circuit| -> usize {
            c.constraints
                .iter()
                .map(|k| k.a.terms.len() + k.b.terms.len() + k.c.terms.len())
                .sum()
        };
        assert_eq!(
            terms(&off),
            terms(&on),
            "{}: compaction changed the non-zero term count",
            name
        );
    }
}

/// A boundary signal that appears in NO constraint must still survive.
///
/// This is the one case liveness-by-constraint-scan gets wrong, and it is not
/// hypothetical: both front ends accept an input that is never read. The wire is
/// genuinely unconstrained - nothing mentions it - so the "drop what nothing
/// refers to" rule would collect it, and the circuit would still compile, still
/// prove, and present a DIFFERENT interface than the one the source declared. A
/// verifier compiled against two public inputs would then be checking a proof
/// about one.
///
/// `compact_wires` marks the whole boundary live before it looks at a single
/// constraint, for exactly this reason. Verified by mutation: removing that
/// makes this fail.
#[test]
fn unused_boundary_signals_survive() {
    for case in cases().into_iter().filter(|c| c.name.contains("unused")) {
        let off = run(&case, false);
        let on = run(&case, true);

        assert_eq!(
            on.circuit.private_inputs.len(),
            off.circuit.private_inputs.len(),
            "{}: compaction changed the number of private inputs",
            case.name
        );
        assert_eq!(
            on.circuit.public_inputs.len(),
            off.circuit.public_inputs.len(),
            "{}: compaction changed the number of public inputs",
            case.name
        );

        // And the unused one is a real, addressable wire, not a dangling id.
        for &w in on.circuit.private_inputs.iter().chain(&on.circuit.public_inputs) {
            assert!(
                w < on.circuit.num_variables,
                "{}: input wire {} is past the end of a {}-wire table",
                case.name,
                w,
                on.circuit.num_variables
            );
        }
        assert!(on.satisfied, "{}: no witness after compaction", case.name);
    }
}

/// What compaction is worth downstream, measured rather than assumed.
///
/// Groth16's proving key holds a G1 element per wire and its MSMs run one scalar
/// per wire, so a dead wire is a curve operation in every proof for the lifetime
/// of the circuit. That is the argument for this pass; this is the measurement.
///
/// The comparison is compaction off vs on at a FIXED constraint count - the
/// other two reduction passes run either way, so nothing else moves and the
/// wire count is the only variable.
///
/// `#[ignore]`d - a benchmark, not a gate. Run with:
///
/// ```bash
/// cargo test --release --features zk --test zk_wire_compaction \
///     -- --ignored --nocapture what_compaction_buys
/// ```
#[test]
#[ignore]
fn what_compaction_buys_at_proving_time() {
    use ark_bn254::{Bn254, Fr as ArkFr};
    use ark_ff::PrimeField;
    use ark_groth16::Groth16;
    use ark_relations::r1cs::{
        ConstraintSynthesizer, ConstraintSystemRef, LinearCombination as ArkLc, SynthesisError,
        Variable,
    };
    use ark_serialize::CanonicalSerialize;
    use ark_snark::SNARK;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use std::time::Instant;
    use y::zk_emitter::LinearCombination;

    fn to_ark(v: &Fr) -> ArkFr {
        ArkFr::from_le_bytes_mod_order(&v.to_bytes_le(32))
    }
    fn ark_lc(lc: &LinearCombination, vars: &[Variable]) -> ArkLc<ArkFr> {
        let mut out = ArkLc::zero();
        for (wire, coeff) in &lc.terms {
            out = out + (to_ark(coeff), vars[*wire]);
        }
        out
    }
    #[derive(Clone)]
    struct C {
        circuit: Circuit,
        witness: Vec<Fr>,
        public: Vec<usize>,
    }
    impl ConstraintSynthesizer<ArkFr> for C {
        fn generate_constraints(self, cs: ConstraintSystemRef<ArkFr>) -> Result<(), SynthesisError> {
            let n = self.circuit.num_variables;
            let mut vars = vec![Variable::One; n];
            for &w in &self.public {
                let v = self.witness[w];
                vars[w] = cs.new_input_variable(|| Ok(to_ark(&v)))?;
            }
            for w in 1..n {
                if self.public.contains(&w) {
                    continue;
                }
                let v = self.witness[w];
                vars[w] = cs.new_witness_variable(|| Ok(to_ark(&v)))?;
            }
            for c in &self.circuit.constraints {
                cs.enforce_constraint(
                    ark_lc(&c.a, &vars),
                    ark_lc(&c.b, &vars),
                    ark_lc(&c.c, &vars),
                )?;
            }
            Ok(())
        }
    }

    let case = Case {
        name: "poseidon_chain200.circom",
        src: Src::Circom("poseidon_chain200.circom"),
        inputs: &[3, 7],
        shrinks: true,
    };

    for (label, on) in [("uncompacted", false), ("compacted  ", true)] {
        let r = run(&case, on);
        assert!(r.satisfied, "{}: no witness", label);
        let public = r.circuit.outputs.clone();
        let c = C {
            circuit: r.circuit.clone(),
            witness: r.witness.clone(),
            public: public.clone(),
        };

        let mut rng = StdRng::seed_from_u64(7);
        let t = Instant::now();
        let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).unwrap();
        let setup = t.elapsed();
        let t = Instant::now();
        let proof = Groth16::<Bn254>::prove(&pk, c, &mut rng).unwrap();
        let prove = t.elapsed();
        let inputs: Vec<ArkFr> = public.iter().map(|w| to_ark(&r.witness[*w])).collect();
        assert!(
            Groth16::<Bn254>::verify(&vk, &inputs, &proof).unwrap(),
            "{}: proof does not verify",
            label
        );

        // The proving key is the noise-free half of the answer: Groth16 stores a
        // G1 element per wire, so this shrinks by exactly the wire ratio, where
        // the TIMES do not - the QAP FFTs and the H-query MSM scale with the
        // domain size, which the constraint count fixes and compaction does not
        // touch.
        eprintln!(
            "{}  {:>7} constraints  {:>7} wires   setup {:>6.3} s   prove {:>6.3} s   pk {:>6.1} MB",
            label,
            r.circuit.constraints.len(),
            r.circuit.num_variables,
            setup.as_secs_f64(),
            prove.as_secs_f64(),
            pk.serialized_size(ark_serialize::Compress::Yes) as f64 / (1024.0 * 1024.0)
        );
    }
}
