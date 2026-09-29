//! The generated Solidity verifier, executed on a real EVM.
//!
//! String-matching a verifier contract proves nothing. The failure mode that
//! matters - G2 coordinates in library order rather than the EVM's reversed
//! order - produces a contract that compiles cleanly, deploys, burns the full
//! gas of a pairing check, and returns `false` for every valid proof. No type
//! error, no revert, nothing to grep for.
//!
//! So this test runs the whole chain: a Y circuit is compiled to R1CS, arkworks
//! performs the setup and produces a proof, Y generates the verifier from the
//! resulting key, `solc` compiles it to EVM bytecode, and `revm` executes
//! `verifyProof` against the BN254 precompiles at 0x06/0x07/0x08 - the same
//! ones Ethereum mainnet runs. An honest proof must be accepted and a tampered
//! public input must be rejected, on-chain.
//!
//! Requires `solc` (or `solcjs`) on PATH; skipped with a printed notice when it
//! is absent, in the same spirit as the `ptxas` gate on the PTX emitter.
//!
//! Run with:  cargo test --features zk --test zk_solidity_verifier
#![cfg(feature = "zk")]

use ark_bn254::{Bn254, Fr as ArkFr, G1Affine, G2Affine};
use ark_ff::PrimeField;
use ark_groth16::Groth16;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystemRef, LinearCombination as ArkLc, SynthesisError, Variable,
};
use ark_snark::SNARK;
use ark_std::rand::{rngs::StdRng, SeedableRng};

use revm::database::InMemoryDB;
use revm::primitives::{keccak256, Address, Bytes, TxKind, U256};
use revm::{Context, ExecuteCommitEvm, MainBuilder, MainContext};

use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::{Circuit, Fr, LinearCombination, ZkEmitter};
use y::zk_solidity::{emit_groth16_verifier, Groth16VerifyingKey};
use y::zk_witness::solve_r1cs_witness;

// ─────────────────────────── Y -> arkworks ───────────────────────────

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
struct YCircuit {
    circuit: Circuit,
    witness: Vec<Fr>,
    public_wires: Vec<usize>,
}

impl ConstraintSynthesizer<ArkFr> for YCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<ArkFr>) -> Result<(), SynthesisError> {
        let n = self.circuit.num_variables;
        let mut vars = vec![Variable::One; n];
        for &wire in &self.public_wires {
            let v = self.witness[wire].clone();
            vars[wire] = cs.new_input_variable(|| Ok(to_ark(&v)))?;
        }
        for wire in 1..n {
            if self.public_wires.contains(&wire) {
                continue;
            }
            let v = self.witness[wire].clone();
            vars[wire] = cs.new_witness_variable(|| Ok(to_ark(&v)))?;
        }
        for c in &self.circuit.constraints {
            cs.enforce_constraint(ark_lc(&c.a, &vars), ark_lc(&c.b, &vars), ark_lc(&c.c, &vars))?;
        }
        Ok(())
    }
}

fn compile(source: &str, priv_in: &[u64]) -> YCircuit {
    let tokens = Lexer::new(source).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    TypeChecker::new().check_program(&program);
    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&program).expect("zk lowering");
    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();

    let privs: Vec<Fr> = priv_in.iter().map(|v| Fr::from_u64(*v)).collect();
    let (witness, ok) =
        solve_r1cs_witness(&circuit.constraints, &ir, circuit.num_variables, &[], &privs);
    assert!(ok, "witness does not satisfy the circuit");

    let mut public_wires = circuit.public_inputs.clone();
    for o in &circuit.outputs {
        if !public_wires.contains(o) {
            public_wires.push(*o);
        }
    }
    public_wires.retain(|w| *w != 0);
    YCircuit { circuit, witness, public_wires }
}

// ─────────────────────── arkworks -> decimal strings ───────────────────────

fn fq_dec(v: &ark_bn254::Fq) -> String {
    v.into_bigint().to_string()
}

fn g1_dec(p: &G1Affine) -> [String; 2] {
    [fq_dec(&p.x), fq_dec(&p.y)]
}

/// G2 in mathematical `(c0, c1)` order - the emitter applies the EVM swap.
fn g2_dec(p: &G2Affine) -> [[String; 2]; 2] {
    [
        [fq_dec(&p.x.c0), fq_dec(&p.x.c1)],
        [fq_dec(&p.y.c0), fq_dec(&p.y.c1)],
    ]
}

fn u256_dec(s: &str) -> U256 {
    U256::from_str_radix(s, 10).expect("decimal")
}

// ────────────────────────────── solc + revm ──────────────────────────────

fn solc_binary() -> Option<Vec<String>> {
    for cmd in [vec!["solc".to_string()], vec!["solcjs".to_string()]] {
        if std::process::Command::new(&cmd[0])
            .arg("--version")
            .output()
            .is_ok()
        {
            return Some(cmd);
        }
    }
    // The npm-local copy this repo installs for the benchmark harness.
    let local = std::path::Path::new("node_modules/.bin/solcjs");
    if local.exists() {
        return Some(vec![local.to_string_lossy().into_owned()]);
    }
    None
}

/// Compiles `source` and returns the deployment bytecode of `contract_name`.
fn compile_solidity(source: &str, contract_name: &str) -> Option<Vec<u8>> {
    let solc = solc_binary()?;
    let dir = std::env::temp_dir().join(format!("y_sol_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let sol = dir.join("Verifier.sol");
    std::fs::write(&sol, source).expect("write .sol");

    let out = std::process::Command::new(&solc[0])
        .args(&solc[1..])
        .arg("--bin")
        .arg("--optimize")
        .arg(&sol)
        .arg("-o")
        .arg(&dir)
        .output()
        .expect("run solc");
    assert!(
        out.status.success(),
        "solc failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // solcjs writes `<mangled path>_<Contract>.bin`; solc writes `<Contract>.bin`.
    let mut found = None;
    for entry in std::fs::read_dir(&dir).expect("read outdir") {
        let path = entry.expect("entry").path();
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        if name.ends_with(".bin") && name.contains(contract_name) {
            let hex = std::fs::read_to_string(&path).expect("read bin");
            let hex = hex.trim();
            if !hex.is_empty() {
                found = Some(hex_decode(hex));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    found
}

fn hex_decode(s: &str) -> Vec<u8> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

const CALLER: Address = Address::new([0x11; 20]);

/// Deploys `bytecode` and returns the contract address.
fn deploy(evm: &mut revm::MainnetEvm<Context<revm::context::BlockEnv, revm::context::TxEnv, revm::context::CfgEnv, InMemoryDB>>, bytecode: Vec<u8>) -> Address {
    let tx = revm::context::TxEnv::builder()
        .caller(CALLER)
        .kind(TxKind::Create)
        .data(Bytes::from(bytecode))
        .gas_limit(16_000_000)
        .gas_price(0)
        .build()
        .expect("tx");
    let res = evm.transact_commit(tx).expect("deploy");
    match res {
        revm::context::result::ExecutionResult::Success {
            output: revm::context::result::Output::Create(_, Some(addr)),
            ..
        } => addr,
        other => panic!("deployment failed: {:?}", other),
    }
}

/// Calls `verifyProof` and returns the single returned bool.
fn call_verify(
    evm: &mut revm::MainnetEvm<Context<revm::context::BlockEnv, revm::context::TxEnv, revm::context::CfgEnv, InMemoryDB>>,
    addr: Address,
    nonce: &mut u64,
    n_public: usize,
    words: &[U256],
) -> bool {
    let sig = format!(
        "verifyProof(uint256[2],uint256[2][2],uint256[2],uint256[{}])",
        n_public
    );
    let mut data = keccak256(sig.as_bytes())[..4].to_vec();
    for w in words {
        data.extend_from_slice(&w.to_be_bytes::<32>());
    }

    let tx = revm::context::TxEnv::builder()
        .caller(CALLER)
        .kind(TxKind::Call(addr))
        .data(Bytes::from(data))
        .gas_limit(16_000_000)
        .gas_price(0)
        .nonce(*nonce)
        .build()
        .expect("tx");
    let res = evm.transact_commit(tx).expect("call");
    *nonce += 1;
    match res {
        revm::context::result::ExecutionResult::Success { output, .. } => {
            let bytes = output.data();
            assert_eq!(bytes.len(), 32, "verifyProof should return one word");
            bytes[31] == 1
        }
        // A revert is a rejection, and for some bad inputs it is the ONLY
        // possible answer: a corrupted proof element is usually not a point on
        // the curve at all, so the pairing precompile rejects its input rather
        // than computing a pairing that comes out unequal, and `require(ok)`
        // reverts. Fail-closed, and what snarkjs's verifier does too - either
        // way the caller learns the proof did not verify.
        revm::context::result::ExecutionResult::Revert { .. } => false,
        other => panic!("verifyProof halted unexpectedly: {:?}", other),
    }
}

// ────────────────────────────────── test ──────────────────────────────────

/// The full chain: Y circuit -> R1CS -> Groth16 -> Solidity -> solc -> EVM.
#[test]
fn generated_verifier_accepts_and_rejects_on_a_real_evm() {
    let Some(_) = solc_binary() else {
        eprintln!("skipping: no solc/solcjs on PATH");
        return;
    };

    // 3 * 2^8 = 768, matching the circuit used elsewhere in the ZK tests.
    let src = "@unsafe\nfn main(x: I32, y: I32) -> I32 {\n    let mut t = x;\n    for i in 0..8 {\n        t = t * y;\n    }\n    return t;\n}\n";
    let c = compile(src, &[3, 2]);

    let mut rng = StdRng::seed_from_u64(0);
    let (pk, vk) =
        Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).expect("setup");
    let public: Vec<ArkFr> = c.public_wires.iter().map(|w| to_ark(&c.witness[*w])).collect();
    let proof = Groth16::<Bn254>::prove(&pk, c, &mut rng).expect("prove");
    assert!(
        Groth16::<Bn254>::verify(&vk, &public, &proof).expect("verify"),
        "arkworks rejected its own proof - nothing below is meaningful"
    );

    // ---- Y generates the verifier ----
    let y_vk = Groth16VerifyingKey {
        alpha_g1: g1_dec(&vk.alpha_g1),
        beta_g2: g2_dec(&vk.beta_g2),
        gamma_g2: g2_dec(&vk.gamma_g2),
        delta_g2: g2_dec(&vk.delta_g2),
        ic: vk.gamma_abc_g1.iter().map(g1_dec).collect(),
    };
    assert_eq!(
        y_vk.num_public_inputs(),
        public.len(),
        "verifying key and public input vector disagree on length"
    );
    let sol = emit_groth16_verifier(&y_vk, "Groth16Verifier");

    let Some(bytecode) = compile_solidity(&sol, "Groth16Verifier") else {
        eprintln!("skipping: solc produced no bytecode");
        return;
    };

    // ---- deploy and call ----
    let mut evm = Context::mainnet().with_db(InMemoryDB::default()).build_mainnet();
    let addr = deploy(&mut evm, bytecode);
    let mut nonce = 1u64;

    // Proof calldata, G2 coordinates imaginary part first (snarkjs layout).
    let mut words = vec![
        u256_dec(&fq_dec(&proof.a.x)),
        u256_dec(&fq_dec(&proof.a.y)),
        u256_dec(&fq_dec(&proof.b.x.c1)),
        u256_dec(&fq_dec(&proof.b.x.c0)),
        u256_dec(&fq_dec(&proof.b.y.c1)),
        u256_dec(&fq_dec(&proof.b.y.c0)),
        u256_dec(&fq_dec(&proof.c.x)),
        u256_dec(&fq_dec(&proof.c.y)),
    ];
    let public_words: Vec<U256> = public
        .iter()
        .map(|v| u256_dec(&v.into_bigint().to_string()))
        .collect();
    words.extend_from_slice(&public_words);

    assert!(
        call_verify(&mut evm, addr, &mut nonce, public.len(), &words),
        "the generated contract rejected a proof arkworks accepts - the usual \
         cause is G2 coordinates emitted in (c0, c1) order instead of the EVM's \
         reversed order"
    );

    // ---- and it must be bound to the claimed output ----
    let mut tampered = words.clone();
    let last = tampered.len() - 1;
    tampered[last] += U256::from(1u64);
    assert!(
        !call_verify(&mut evm, addr, &mut nonce, public.len(), &tampered),
        "the contract accepted a proof against a public input it does not match"
    );

    // A mangled proof point must also fail rather than be silently ignored.
    let mut bad_proof = words.clone();
    bad_proof[0] += U256::from(1u64);
    assert!(
        !call_verify(&mut evm, addr, &mut nonce, public.len(), &bad_proof),
        "the contract accepted a corrupted proof element"
    );
}
