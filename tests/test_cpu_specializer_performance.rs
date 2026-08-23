// ============================================================
//  Y  —  CPU Matrix Specialization Performance & Regimes Test
//  test_cpu_specializer_performance.rs
// ============================================================

use y::cpu_specializer::{CpuHardwareProfile, CpuShapeDispatcher, CpuMatrixRegime};
use y::cpu_emitter::CpuEmitter;
use std::time::Instant;

#[test]
fn test_cpu_specializer_all_five_regimes_dispatch() {
    // AVX-512 named explicitly. The `IrregularMasked` regime is gated on
    // `supports_avx512_masking`, and `CpuHardwareProfile::default()` is now
    // AVX2 - it used to assume AVX-512, which is guessing UP at an ISA whose
    // absence is a SIGILL. This test is about the five regimes, so it states
    // the machine that has all five rather than inheriting one.
    let profile = CpuHardwareProfile {
        simd_vector_width_floats: 16,
        supports_avx512_masking: true,
        ..CpuHardwareProfile::default()
    };
    let dispatcher = CpuShapeDispatcher::new(profile);

    // 1. SmallDirect regime
    let r1 = dispatcher.classify_shape(16, 16, 16, 4);
    assert_eq!(r1, CpuMatrixRegime::SmallDirect);

    // 2. DecodeGEMV regime
    let r2 = dispatcher.classify_shape(1, 4096, 4096, 4);
    assert_eq!(r2, CpuMatrixRegime::DecodeGEMV);

    // 3. DeepK regime
    let r3 = dispatcher.classify_shape(64, 64, 32768, 4);
    assert_eq!(r3, CpuMatrixRegime::DeepK);

    // 4. IrregularMasked regime
    let r4 = dispatcher.classify_shape(137, 391, 1013, 4);
    assert_eq!(r4, CpuMatrixRegime::IrregularMasked);

    // 5. NiceSquare regime
    let r5 = dispatcher.classify_shape(1024, 1024, 1024, 4);
    assert_eq!(r5, CpuMatrixRegime::NiceSquare);
}

#[test]
fn test_cpu_specializer_codegen_emission() {
    let mut emitter = CpuEmitter::new();
    
    let start = Instant::now();
    emitter.emit_specialized_cpu_kernel_dispatch("test_small", 16, 16, 16);
    emitter.emit_specialized_cpu_kernel_dispatch("test_decode", 1, 4096, 4096);
    emitter.emit_specialized_cpu_kernel_dispatch("test_deep_k", 64, 64, 32768);
    emitter.emit_specialized_cpu_kernel_dispatch("test_irregular", 137, 391, 1013);
    emitter.emit_specialized_cpu_kernel_dispatch("test_nice", 1024, 1024, 1024);
    let elapsed = start.elapsed();

    assert!(emitter.host_buffer.contains("test_small_cpu_small_direct"));
    assert!(emitter.host_buffer.contains("test_decode_cpu_decode_gemv"));
    assert!(emitter.host_buffer.contains("test_deep_k_cpu_deep_k_split"));
    assert!(emitter.host_buffer.contains("test_irregular_cpu_irregular_masked"));
    assert!(emitter.host_buffer.contains("test_nice_cpu_blis_packed"));

    println!("[PERF] CPU Specializer Codegen Emission completed in {:?}", elapsed);
}
