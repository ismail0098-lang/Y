#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
use std::ffi::c_void;
use std::rc::Rc;
use y::cpu_jit::{AbiType, CpuJit, CpuJitCache, JitOptions, JitValue as V};

const SOURCE: &str = r#"
fn echo8(x: I8) -> I8 { return x; }
fn echou8(x: U8) -> U8 { return x; }
fn echochar(x: char) -> char { return x; }
fn charcode(x: char) -> I32 { return ychar_to_ascii(x); }
fn highchar() -> char { return 'È'; }
fn echo16(x: I16) -> I16 { return x; }
fn echou16(x: U16) -> U16 { return x; }
fn echo32(x: I32) -> I32 { return x; }
fn echou32(x: U32) -> U32 { return x; }
fn echo64(x: I64) -> I64 { return x; }
fn echou64(x: U64) -> U64 { return x; }
fn size(x: usize) -> usize { return x; }
fn boolean(x: bool) -> bool { return x; }
fn single(x: F32) -> F32 { return x; }
fn double(x: F64) -> F64 { return x; }
@unsafe
fn write_value(p: GlobalMemory<I64>, x: I64) { p[0] = x; }
@unsafe
fn identity(p: GlobalMemory<I64>) -> GlobalMemory<I64> { return p; }
fn many(a: I64, b: F64, c: I8, d: F32, e: U64, f: bool, g: I32, h: I64, i: U16) -> I64 {
    if f && b == 1.25 && d == 2.5 && e == 9000000000 { return a + c + g + h + i; }
    return -1;
}
fn main() {}
struct Pair { x: I64, y: I64, }
fn pair() -> Pair { let mut p: Pair = {}; p.x = 11; p.y = 22; return p; }
"#;

#[test]
fn dynamic_calls_preserve_scalar_types_ieee_bits_and_mixed_abi_arguments() {
    for opt_level in [0, 3] {
        let jit = CpuJit::compile_with_options(
            SOURCE,
            JitOptions {
                opt_level,
                ..JitOptions::default()
            },
        )
        .unwrap();
        for (name, value) in [
            ("echo8", V::I8(-127)),
            ("echou8", V::U8(250)),
            ("echochar", V::U8(200)),
            ("echo16", V::I16(-32000)),
            ("echou16", V::U16(65000)),
            ("echo32", V::I32(i32::MIN)),
            ("echou32", V::U32(u32::MAX)),
            ("echo64", V::I64(i64::MIN)),
            ("echou64", V::U64(u64::MAX)),
            ("size", V::Usize(usize::MAX)),
            ("boolean", V::Bool(true)),
            ("boolean", V::Bool(false)),
            ("single", V::F32(-0.0)),
            ("double", V::F64(-0.0)),
            ("single", V::F32(f32::INFINITY)),
            ("double", V::F64(f64::NEG_INFINITY)),
            ("single", V::F32(f32::from_bits(0x7fc01234))),
            ("double", V::F64(f64::from_bits(0x7ff8000000001234))),
        ] {
            let result = unsafe { jit.call(name, &[value]) }.unwrap();
            assert_eq!(result.abi_type(), value.abi_type(), "{name}");
            assert_eq!(result.bits(), value.bits(), "{name}");
        }
        assert_eq!(
            unsafe {
                jit.call(
                    "many",
                    &[
                        V::I64(10),
                        V::F64(1.25),
                        V::I8(-2),
                        V::F32(2.5),
                        V::U64(9000000000),
                        V::Bool(true),
                        V::I32(3),
                        V::I64(20),
                        V::U16(65535),
                    ],
                )
            }
            .unwrap(),
            V::I64(65566)
        );
        assert_eq!(
            unsafe { jit.call("charcode", &[V::U8(200)]) }.unwrap(),
            V::I32(200)
        );
        assert_eq!(unsafe { jit.call("highchar", &[]) }.unwrap(), V::U8(200));
        assert_eq!(unsafe { jit.call("main", &[]) }.unwrap(), V::Void);
        assert!(!jit
            .functions()
            .any(|name| name.starts_with("__y_jit_dispatch_")));
    }
}

#[test]
fn pointer_and_void_calls_check_arguments_before_memory_mutation() {
    for opt_level in [0, 3] {
        let jit = CpuJit::compile_with_options(
            SOURCE,
            JitOptions {
                opt_level,
                ..JitOptions::default()
            },
        )
        .unwrap();
        let mut storage = [73_i64; 2];
        let pointer = V::Pointer(storage.as_mut_ptr().cast::<c_void>());
        for args in [
            vec![pointer],
            vec![pointer, V::U64(99)],
            vec![pointer, V::I64(99), V::I64(1)],
        ] {
            assert!(unsafe { jit.call("write_value", &args) }.is_err());
            assert_eq!(storage, [73, 73]);
        }
        assert_eq!(
            unsafe { jit.call("identity", &[pointer]) }.unwrap(),
            pointer
        );
        assert_eq!(
            unsafe { jit.call("write_value", &[pointer, V::I64(4294967296)]) }.unwrap(),
            V::Void
        );
        assert_eq!(storage, [4294967296, 73]);
    }
}

#[test]
fn aggregate_signatures_are_inspectable_and_direct_lookup_still_works() {
    let jit = CpuJit::compile(SOURCE).unwrap();
    assert!(unsafe { jit.call("absent", &[]) }
        .unwrap_err()
        .to_string()
        .contains("absent"));
    let signature = jit.function_signature("many").unwrap();
    assert_eq!(
        signature.parameters,
        [
            AbiType::I64,
            AbiType::F64,
            AbiType::I8,
            AbiType::F32,
            AbiType::U64,
            AbiType::Bool,
            AbiType::I32,
            AbiType::I64,
            AbiType::U16
        ]
    );
    assert_eq!(signature.return_type, AbiType::I64);
    assert!(signature.to_json().contains("\"dynamic_call\":true"));
    assert!(!jit
        .function_signature("pair")
        .unwrap()
        .supports_dynamic_call());
    assert!(unsafe { jit.call("pair", &[]) }
        .unwrap_err()
        .to_string()
        .contains("aggregate"));
    assert_ne!(jit.function_address("pair").unwrap(), 0);
}

#[test]
fn adapter_names_do_not_collide_with_user_functions() {
    let jit = CpuJit::compile(
        "fn __y_jit_dispatch_0() -> I32 { return 71; } fn other() -> I32 { return 12; }",
    )
    .unwrap();
    assert_eq!(
        unsafe { jit.call("__y_jit_dispatch_0", &[]) }.unwrap(),
        V::I32(71)
    );
    assert_eq!(unsafe { jit.call("other", &[]) }.unwrap(), V::I32(12));
    assert!(jit.function_address("__y_jit_dispatch_0").is_ok());
}

#[test]
fn bounded_cache_keys_options_and_eviction_preserves_retained_code() {
    let mut cache = CpuJitCache::new(2);
    let source = "fn value() -> I32 { return 19; }";
    let first = cache
        .compile(
            source,
            JitOptions {
                opt_level: 0,
                ..JitOptions::default()
            },
        )
        .unwrap();
    let reused = cache
        .compile(
            source,
            JitOptions {
                opt_level: 0,
                ..JitOptions::default()
            },
        )
        .unwrap();
    assert!(Rc::ptr_eq(&first, &reused));
    assert_eq!(cache.hits(), 1);
    let optimized = cache
        .compile(
            source,
            JitOptions {
                opt_level: 3,
                ..JitOptions::default()
            },
        )
        .unwrap();
    assert!(!Rc::ptr_eq(&first, &optimized));
    let changed = cache
        .compile("fn value() -> I32 { return 31; }", JitOptions::default())
        .unwrap();
    assert_eq!(cache.len(), 2);
    assert_eq!(unsafe { first.call("value", &[]) }.unwrap(), V::I32(19));
    assert_eq!(unsafe { changed.call("value", &[]) }.unwrap(), V::I32(31));
    let rebuilt = cache
        .compile(
            source,
            JitOptions {
                opt_level: 0,
                ..JitOptions::default()
            },
        )
        .unwrap();
    assert!(!Rc::ptr_eq(&first, &rebuilt));
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(unsafe { rebuilt.call("value", &[]) }.unwrap(), V::I32(19));
    assert!(cache
        .compile("fn broken( {", JitOptions::default())
        .is_err());
    assert!(cache.is_empty());
    let mut disabled = CpuJitCache::new(0);
    let a = disabled.compile(source, JitOptions::default()).unwrap();
    let b = disabled.compile(source, JitOptions::default()).unwrap();
    assert!(!Rc::ptr_eq(&a, &b));
    assert!(disabled.is_empty());
}

#[test]
fn cache_distinguishes_all_lowering_and_profile_policy_options() {
    let source = "fn value() -> I32 { return 19; }";
    let mut cache = CpuJitCache::new(17);
    let defaults = JitOptions::default();
    let first = cache.compile(source, defaults).unwrap();
    for options in [
        JitOptions {
            recognize_rotates: false,
            ..defaults
        },
        JitOptions {
            optimize_runtime: false,
            ..defaults
        },
        JitOptions {
            optimize_runtime_mutations: false,
            ..defaults
        },
        JitOptions {
            optimize_runtime_copies: false,
            ..defaults
        },
        JitOptions {
            optimize_helper_effects: false,
            ..defaults
        },
        JitOptions {
            optimize_call_adapters: false,
            ..defaults
        },
        JitOptions {
            profile_loop_controls: true,
            ..defaults
        },
        JitOptions {
            verify_each_pass: false,
            ..defaults
        },
        JitOptions {
            codegen_opt_level: Some(2),
            ..defaults
        },
        // An explicit setting is a different requested policy than inheritance.
        JitOptions {
            codegen_opt_level: Some(3),
            ..defaults
        },
        JitOptions {
            training_opt_level: Some(1),
            ..defaults
        },
        JitOptions {
            training_opt_level: Some(3),
            ..defaults
        },
        JitOptions {
            profile_edge_counters: true,
            ..defaults
        },
        JitOptions {
            profile_loop_edge_counters: true,
            ..defaults
        },
        JitOptions {
            final_loop_unrolling: false,
            ..defaults
        },
        JitOptions {
            final_unroll_outer_loops: false,
            ..defaults
        },
    ] {
        let different = cache.compile(source, options).unwrap();
        assert!(!Rc::ptr_eq(&first, &different));
        assert!(Rc::ptr_eq(
            &different,
            &cache.compile(source, options).unwrap()
        ));
        assert_eq!(unsafe { different.call("value", &[]) }.unwrap(), V::I32(19));
    }
    assert!(Rc::ptr_eq(
        &first,
        &cache.compile(source, defaults).unwrap()
    ));
    assert_eq!((cache.len(), cache.hits(), cache.misses()), (17, 17, 17));
}

#[test]
fn tagged_values_reject_invalid_tags_and_noncanonical_bits() {
    for (tag, bits) in [
        (99, 0),
        (10, 2),
        (2, 256),
        (4, 65536),
        (5, 1_u64 << 32),
        (11, 1_u64 << 32),
        (0, 1),
    ] {
        assert!(V::from_tagged_bits(tag, bits).is_err());
    }
    assert_eq!(V::from_tagged_bits(1, 255).unwrap(), V::I8(-1));
    assert_eq!(V::from_tagged_bits(8, u64::MAX).unwrap(), V::U64(u64::MAX));
}
