pub mod ast;
pub mod avx_wrapper;
pub mod bank_conflict;
pub mod cpu_emitter;
pub mod lexer;
pub mod linear_tracker;
pub mod llvm_emitter;
pub mod parser;
pub mod ptx_emitter;
pub mod sentinel;
pub mod type_checker;
pub mod native_emitter;
pub mod ir_grapher;
pub mod rt_core_emitter;
pub mod quantization_pass;
pub mod coprocessor_scheduler;

pub mod autotuner;
pub mod cuda_runtime;
pub mod empirical_autotune;
pub mod zero_drift;
pub mod fixed_exp;
pub mod exact_attention;
pub mod c_api;
pub mod rocm_emitter;
pub mod layout_pass;
pub mod auto_vectorize;
pub mod cpu_specializer;
pub mod cpu_gemm;

#[cfg(feature = "zk")]
pub mod circom_lexer;
#[cfg(feature = "zk")]
pub mod circom_ast;
#[cfg(feature = "zk")]
pub mod circom_parser;
#[cfg(feature = "zk")]
pub mod circom_lower;
#[cfg(feature = "zk")]
pub mod zk_field;
#[cfg(feature = "zk")]
pub mod zk_emitter;
#[cfg(feature = "zk")]
pub mod zk_witness;
#[cfg(feature = "zk")]
pub mod zk_poseidon_constants;
#[cfg(feature = "zk")]
pub mod mini_json;
#[cfg(feature = "zk")]
pub mod zk_solidity;
#[cfg(feature = "zk")]
pub mod zk_fuzz;

/// Runs all advanced compiler optimization passes on a Program AST.
pub fn run_all_optimization_passes(prog: &mut ast::Program) {
    let mut auto_vec = auto_vectorize::AutoVectorizePass::new();
    auto_vec.run(prog);

    let mut unroll_jam = auto_vectorize::UnrollAndJamPass::new();
    unroll_jam.run(prog);

    let mut layout_pass = layout_pass::ConvertLayoutPass::new();
    layout_pass.run(prog);

    let mut smem_swizzle = layout_pass::SmemBankSwizzlePass::new();
    smem_swizzle.run(prog);

    let mut cpu_spec = cpu_specializer::CpuSpecializerPass::new(
        cpu_specializer::CpuHardwareProfile::default()
    );
    cpu_spec.run(prog);
}


