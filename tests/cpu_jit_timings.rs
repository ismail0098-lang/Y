#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::rc::Rc;
use std::time::Duration;
use y::cpu_jit::{CpuJit, CpuJitCache, JitOptions, JitValue};
use y::{lexer::Lexer, parser::Parser};

const SOURCE: &str = "fn choose(x: I64) -> I64 { if x < 0 { return x - 1; } return x + 2; }";

fn check_timings(jit: &CpuJit, parsed_source: bool) {
    let timings = jit.compile_timings();
    assert_eq!(timings.accounted_duration(), timings.total);
    assert_eq!(jit.compile_duration(), timings.total);
    assert!(timings.total > Duration::ZERO);
    let optimization = jit.optimization_timings();
    assert_eq!(optimization.accounted_duration(), optimization.total);
    assert_eq!(optimization.total, timings.optimization);
    assert!(optimization.pipeline > Duration::ZERO);
    assert!(matches!(timings.verification_checks, 2 | 3));
    if parsed_source {
        assert!(timings.parse > Duration::ZERO);
    } else {
        assert_eq!(timings.parse, Duration::ZERO);
    }
    let before = *timings;
    assert_eq!(
        unsafe { jit.call("choose", &[JitValue::I64(-8)]) }.unwrap(),
        JitValue::I64(-9)
    );
    assert_eq!(
        *jit.compile_timings(),
        before,
        "calls cannot alter compile timings"
    );
}

#[test]
fn source_and_ast_timings_partition_each_compilation_mode() {
    let ast = Parser::new(Lexer::new(SOURCE).tokenize())
        .parse_program()
        .unwrap();
    for (opt_level, verify_each_pass) in [(0, true), (0, false), (3, true), (3, false)] {
        let options = JitOptions {
            opt_level,
            verify_each_pass,
            ..JitOptions::default()
        };
        let original = CpuJit::compile_with_options(SOURCE, options).unwrap();
        check_timings(&original, true);
        let original_ast = CpuJit::compile_program(&ast, options).unwrap();
        check_timings(&original_ast, false);
        let training = CpuJit::compile_instrumented(SOURCE, options).unwrap();
        let training_ast = CpuJit::compile_program_instrumented(&ast, options).unwrap();
        assert_eq!(training.branch_profile().unwrap().total_observations(), 0);
        assert_eq!(
            training_ast.branch_profile().unwrap().total_observations(),
            0
        );
        check_timings(&training, true);
        check_timings(&training_ast, false);
        let profile = training.branch_profile().unwrap();
        let optimized = CpuJit::compile_with_profile(SOURCE, options, &profile).unwrap();
        let optimized_ast = CpuJit::compile_program_with_profile(&ast, options, &profile).unwrap();
        check_timings(&optimized, true);
        check_timings(&optimized_ast, false);
        assert_eq!(
            training.branch_profile().unwrap(),
            profile,
            "recompile cannot train source"
        );
    }
}

#[test]
fn cache_hits_retain_original_compilation_measurements() {
    let mut cache = CpuJitCache::new(1);
    let first = cache.compile(SOURCE, JitOptions::default()).unwrap();
    let before = *first.compile_timings();
    let hit = cache.compile(SOURCE, JitOptions::default()).unwrap();
    assert!(Rc::ptr_eq(&first, &hit));
    assert_eq!(*hit.compile_timings(), before);
    assert_eq!(hit.optimization_timings(), first.optimization_timings());
    assert_eq!(
        hit.materialization_timings(),
        first.materialization_timings()
    );
    assert_eq!(before.accounted_duration(), before.total);
}
