#![cfg(feature = "zk")]

use std::collections::{BTreeSet, HashSet};
use y::lexer::Lexer;
use y::parser::Parser;
use y::zk_emitter::{
    active_modulus, set_active_modulus, BigUint, Circuit, FieldConfig, Fr, R1csEncoder,
    ScalarField, ZkEmitter,
};
use y::zk_witness::{check_r1cs_satisfiability, execute_host_witness_ir, solve_r1cs_witness};

struct RestoreField(BigUint);

impl RestoreField {
    fn new() -> Self {
        Self(active_modulus())
    }
}

impl Drop for RestoreField {
    fn drop(&mut self) {
        set_active_modulus(&self.0);
    }
}

fn compile(field: &str) -> ZkEmitter {
    let source = format!(
        r#"
        @zk_target(field = "{field}", scheme = "r1cs")
        module Example {{
            fn main(x: I32, y: I32) -> I32 {{ return x * y + y; }}
        }}
    "#
    );
    let program = Parser::new(Lexer::new(&source).tokenize())
        .parse_program()
        .unwrap();
    let mut emitter = ZkEmitter::new();
    emitter.emit_program(&program).unwrap();
    emitter
}

fn encode(circuit: &Circuit) -> Vec<u8> {
    let (map, outputs, public, private) = ZkEmitter::snarkjs_wire_map(circuit);
    let mut bytes = Vec::new();
    R1csEncoder::new()
        .encode_to_stream(circuit.view(), &map, outputs, public, private, &mut bytes)
        .unwrap();
    bytes
}

#[test]
fn retained_elements_keep_their_value_arithmetic_and_identity() {
    let _restore = RestoreField::new();
    let bn = FieldConfig::get(ScalarField::Bn254);
    let bls = FieldConfig::get(ScalarField::Bls12_381);
    set_active_modulus(&bn.p);
    let seven = Fr::from_u64(7);
    let two = Fr::from_u64(2);
    let one = Fr::one();
    let zero = Fr::zero();
    let minus_one = one.neg();
    let bytes = minus_one.to_bytes_le(32);
    let mut identities = HashSet::from([seven, one, Fr::zero()]);

    set_active_modulus(&bls.p);
    assert_eq!(seven.to_u64(), Some(7));
    assert_eq!(seven.to_decimal_string(), "7");
    assert_eq!(minus_one.to_bytes_le(32), bytes);
    assert_eq!(minus_one.to_biguint(), bn.p.sub(&BigUint::one()));
    assert_eq!(minus_one.add(&one), zero);
    assert_eq!(seven.add(&two).to_u64(), Some(9));
    assert_eq!(seven.sub(&two).to_u64(), Some(5));
    assert_eq!(seven.mul(&two).to_u64(), Some(14));
    assert_eq!(seven.mul(&seven.inv()), one);
    assert_eq!(seven.pow_limbs(&[3, 0, 0, 0]).to_u64(), Some(343));
    let (quotient, remainder) = seven.int_div_rem(&two);
    assert_eq!((quotient.to_u64(), remainder.to_u64()), (Some(3), Some(1)));
    assert_eq!(quotient.mul(&two).add(&remainder), seven);
    assert!(one.is_one());
    assert!(seven > two);

    let bls_seven = Fr::from_u64(7);
    assert_ne!(seven, bls_seven);
    assert_ne!(zero, Fr::zero());
    identities.insert(bls_seven);
    assert_eq!(identities.len(), 4);
    assert_eq!(BTreeSet::from([seven, bls_seven]).len(), 2);
    assert!(std::panic::catch_unwind(|| seven.add(&bls_seven)).is_err());
    assert!(std::panic::catch_unwind(|| seven.mul(&bls_seven)).is_err());
    set_active_modulus(&bn.p);
    assert!(identities.contains(&Fr::from_u64(7)));

    // Interning is process-wide: values keep the same identity on another
    // thread, regardless of that thread's default constructor field.
    std::thread::spawn(move || {
        set_active_modulus(&bls.p);
        assert_eq!(seven.to_u64(), Some(7));
        assert_eq!(seven.mul(&seven.inv()), one);
        assert_eq!(bls_seven, Fr::from_u64(7));
    })
    .join()
    .unwrap();
}

#[test]
fn retained_circuits_and_witnesses_survive_interleaved_fields() {
    let _restore = RestoreField::new();
    let bn_emitter = compile("bn254");
    let bn_circuit = bn_emitter.build_circuit();
    let bn_ir = bn_emitter.build_witness_ir();
    let bn = FieldConfig::get(ScalarField::Bn254);
    let bls = FieldConfig::get(ScalarField::Bls12_381);
    // Construct inputs while their field is active; all subsequent operations
    // must retain it even after another emitter changes the default field.
    let inputs = [Fr::from_u64(7), Fr::from_u64(3)];
    let (expected, satisfied) = solve_r1cs_witness(
        &bn_circuit.constraints,
        &bn_ir,
        bn_circuit.num_variables,
        &[],
        &inputs,
    );
    assert!(satisfied);
    assert_eq!(expected[bn_circuit.outputs[0]].to_u64(), Some(24));
    let r1cs = encode(&bn_circuit);
    assert_eq!(&r1cs[28..60], bn.p.to_bytes_le(32));

    let wtns_path =
        std::env::temp_dir().join(format!("y-field-context-{}.wtns", std::process::id()));
    ZkEmitter::write_wtns_binary(&bn_circuit, &expected, wtns_path.to_str().unwrap()).unwrap();
    let wtns = std::fs::read(&wtns_path).unwrap();

    let bls_emitter = compile("bls12_381");
    let bls_circuit = bls_emitter.build_circuit();
    let bls_ir = bls_emitter.build_witness_ir();
    let bls_inputs = [Fr::from_u64(7), Fr::from_u64(3)];
    // Build another IR from the retained emitter while BLS is active, too.
    let rebuilt_bn_ir = bn_emitter.build_witness_ir();
    for ir in [&bn_ir, &rebuilt_bn_ir] {
        let w = execute_host_witness_ir(ir, &[], &inputs).unwrap();
        assert_eq!(w, expected);
        check_r1cs_satisfiability(&bn_circuit.constraints, &w).unwrap();
    }
    assert_eq!(encode(&bn_circuit), r1cs);
    assert_eq!(encode(&bn_emitter.build_circuit()), r1cs);
    ZkEmitter::write_wtns_binary(&bn_circuit, &expected, wtns_path.to_str().unwrap()).unwrap();
    assert_eq!(std::fs::read(&wtns_path).unwrap(), wtns);
    assert_eq!(&wtns[28..60], bn.p.to_bytes_le(32));

    // Exercise the reverse order and ensure the BLS header/values survive a
    // return to BN254 as well.
    let bls_encoded = encode(&bls_circuit);
    set_active_modulus(&bn.p);
    let (w, satisfied) = solve_r1cs_witness(
        &bls_circuit.constraints,
        &bls_ir,
        bls_circuit.num_variables,
        &[],
        &bls_inputs,
    );
    assert!(satisfied);
    assert_eq!(w[bls_circuit.outputs[0]].to_u64(), Some(24));
    assert_eq!(encode(&bls_circuit), bls_encoded);
    assert_eq!(&bls_encoded[28..60], bls.p.to_bytes_le(32));
    ZkEmitter::write_wtns_binary(&bls_circuit, &w, wtns_path.to_str().unwrap()).unwrap();
    let bls_wtns = std::fs::read(&wtns_path).unwrap();
    assert_eq!(&bls_wtns[28..60], bls.p.to_bytes_le(32));
    std::fs::remove_file(&wtns_path).unwrap();

    assert!(execute_host_witness_ir(&bn_ir, &[], &bls_inputs)
        .unwrap_err()
        .contains("scalar field"));
    assert!(
        !solve_r1cs_witness(
            &bn_circuit.constraints,
            &bn_ir,
            bn_circuit.num_variables,
            &[],
            &bls_inputs,
        )
        .1
    );
    assert!(check_r1cs_satisfiability(&bn_circuit.constraints, &w).is_err());
    assert!(ZkEmitter::write_wtns_binary(&bn_circuit, &w, wtns_path.to_str().unwrap()).is_err());
    assert!(
        !wtns_path.exists(),
        "invalid-field witnesses must be rejected before writing"
    );
}
