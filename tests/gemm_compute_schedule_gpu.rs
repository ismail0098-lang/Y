//! Real-device regression for the compute block shared by GEMM and SwiGLU.
//! Run serially, without concurrent GPU benchmarks:
//! cargo test --test gemm_compute_schedule_gpu -- --ignored --test-threads=1

use y::cuda_runtime::{f16_bits_to_f32, CudaContext};
use y::lexer::Lexer;
use y::parser::Parser;
use y::ptx_emitter::PtxEmitter;
use y::sentinel::HardwareProfile;

fn hardware(ctx: &CudaContext) -> HardwareProfile {
    let attribute = |id| {
        let value = ctx.device_attribute(id).expect("CUDA device attribute");
        u32::try_from(value).expect("nonnegative CUDA device attribute")
    };
    let major = attribute(75);
    let minor = attribute(76);
    assert!(major >= 8, "this test requires an sm_80+ CUDA device");
    let warp_size = attribute(10);
    assert_eq!(warp_size, 32);
    let max_threads_per_sm = attribute(39);
    HardwareProfile {
        gpu_name: ctx.device_name().into(),
        gpu_vendor: "NVIDIA".into(),
        sm_version: format!("sm_{major}{minor}"),
        compute_capability: format!("{major}.{minor}"),
        sm_count: attribute(16),
        max_smem_per_sm_bytes: attribute(81).min(attribute(97)),
        max_regs_per_sm: attribute(82),
        max_regs_per_thread: 255,
        max_threads_per_sm,
        warp_size,
        max_warps_per_sm: max_threads_per_sm / warp_size,
        ..Default::default()
    }
}

// Construct exact normal F16s with independent signs and mantissas. Small
// magnitudes keep the gate projection away from sigmoid saturation; different
// salts distinguish the gate/up operands and expose transposed indexing.
fn input(count: usize, salt: u64) -> (Vec<u8>, Vec<f64>) {
    let mut state = salt;
    let mut bytes = Vec::with_capacity(count * 2);
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let sign = (state >> 63) as u16;
        let exponent = 10 + ((state >> 24) % 3) as u16;
        let fraction = (state & 1023) as u16;
        let bits = (sign << 15) | (exponent << 10) | fraction;
        bytes.extend_from_slice(&bits.to_ne_bytes());
        values.push(f64::from(f16_bits_to_f32(bits)));
    }
    (bytes, values)
}

fn check_swiglu(m: u32, n: u32, k: u32, pipelined: bool) {
    assert!(
        std::env::var_os("Y_SWIGLU_TILE").is_none(),
        "run this regression without the experimental Y_SWIGLU_TILE override"
    );
    let ctx = CudaContext::new().expect("an explicitly requested CUDA GPU is required");
    ctx.require_current().unwrap();
    let hw = hardware(&ctx);
    let source = format!(
        "@tile({m}, {n}, {k})\n\
         kernel schedule_swiglu(X: GlobalMemory<F16>, Wgate: GlobalMemory<F16>, \
         Wup: GlobalMemory<F16>, Out: GlobalMemory<F32>) {{ let x: I32 = 0; }}"
    );
    let mut lexer = Lexer::new(&source);
    let mut parser = Parser::new(lexer.tokenize());
    let ast = parser.parse_program().unwrap();
    let mut emitter = PtxEmitter::new_with_profile(&hw);
    let ptx = emitter.emit_program(&ast, &hw);
    assert!(emitter.emit_errors.is_empty(), "{:?}", emitter.emit_errors);
    assert!(ptx.contains("EPI_SWIGLU"));
    assert_eq!(ptx.contains("cp.async.cg.shared.global"), pipelined);

    // Follow the same emitted launch metadata as the SwiGLU benchmark rather
    // than duplicating its tile/warp/shared-memory selection arithmetic.
    let header = ptx
        .lines()
        .find(|line| line.contains("[Y FUSED LINEAR+SWIGLU GEMM]"))
        .expect("fused SwiGLU launch metadata");
    let fields: Vec<_> = header.split('|').collect();
    let dimensions = |field: &str| -> Vec<u32> {
        field
            .split_whitespace()
            .next()
            .unwrap()
            .split('x')
            .map(|value| value.parse().unwrap())
            .collect()
    };
    let cta = dimensions(fields[1].trim().strip_prefix("CTA ").unwrap());
    let warps = dimensions(fields[2].trim());
    assert_eq!(cta.len(), 3);
    assert_eq!(warps.len(), 2);
    assert_eq!((m % cta[0], n % cta[1], k % cta[2]), (0, 0, 0));
    let shared_bytes: u32 = ptx
        .split("Dynamic shared memory required: ")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(shared_bytes <= hw.max_smem_per_sm_bytes);
    let kernel = ctx.load_ptx(&ptx, "schedule_swiglu").unwrap();
    kernel.set_max_dynamic_smem(shared_bytes).unwrap();

    let (m, n, k) = (m as usize, n as usize, k as usize);
    let (x_bytes, x_host) = input(m * k, 0x1234_5678_9abc_def1);
    let (gate_bytes, gate_host) = input(k * n, 0x9876_5432_fedc_ba91);
    let (up_bytes, up_host) = input(k * n, 0x6ac1_4f79_b35e_820d);
    let x = ctx.alloc(x_bytes.len()).unwrap();
    let gate = ctx.alloc(gate_bytes.len()).unwrap();
    let up = ctx.alloc(up_bytes.len()).unwrap();
    let out = ctx.alloc(m * n * 4).unwrap();
    ctx.memcpy_htod_at(&x, 0, &x_bytes).unwrap();
    ctx.memcpy_htod_at(&gate, 0, &gate_bytes).unwrap();
    ctx.memcpy_htod_at(&up, 0, &up_bytes).unwrap();
    let mut reference = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let (mut gate_sum, mut up_sum) = (0.0, 0.0);
            for inner in 0..k {
                gate_sum += x_host[row * k + inner] * gate_host[inner * n + col];
                up_sum += x_host[row * k + inner] * up_host[inner * n + col];
            }
            reference[row * n + col] = gate_sum / (1.0 + (-gate_sum).exp()) * up_sum;
        }
    }
    let args = [
        x.device_ptr(),
        gate.device_ptr(),
        up.device_ptr(),
        out.device_ptr(),
    ];
    let mut output = vec![0u8; m * n * 4];
    for _ in 0..2 {
        // A stale correct buffer cannot make a missed store or missed reuse
        // launch pass: poison every output element before each enqueue.
        ctx.memset_u8(&out, 0xff).unwrap();
        ctx.launch(
            &kernel,
            (n as u32 / cta[1], m as u32 / cta[0], 1),
            (warps[0] * warps[1] * 32, 1, 1),
            shared_bytes,
            &args,
        )
        .unwrap();
        ctx.synchronize().unwrap();
        ctx.memcpy_dtoh_at(&mut output, &out, 0).unwrap();
        let (mut squared_error, mut squared_ref, mut max_error) = (0.0, 0.0, 0.0f64);
        for (index, (bytes, &expected)) in output.chunks_exact(4).zip(&reference).enumerate() {
            let actual = f64::from(f32::from_ne_bytes(bytes.try_into().unwrap()));
            let error = (actual - expected).abs();
            let tolerance = 1e-5 + 0.002 * expected.abs();
            assert!(
                actual.is_finite() && error <= tolerance,
                "SwiGLU {m}x{n}x{k}, output {index}: got {actual}, expected {expected}, tolerance {tolerance}"
            );
            squared_error += error * error;
            squared_ref += expected * expected;
            max_error = max_error.max(error);
        }
        let relative_l2 = (squared_error / squared_ref.max(f64::MIN_POSITIVE)).sqrt();
        assert!(relative_l2 <= 0.002, "relative L2 {relative_l2}");
        println!(
            "SwiGLU {m}x{n}x{k}: maximum absolute error {max_error:e}, relative L2 {relative_l2:e}"
        );
    }
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU; run serially without concurrent GPU work"]
fn fused_swiglu_synchronous_shared_compute_matches_full_cpu_reference() {
    // SwiGLU currently requires exact multiples of its fixed 128x128x32
    // tile. One K tile covers its synchronous shared-compute path.
    check_swiglu(128, 256, 32, false);
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU; run serially without concurrent GPU work"]
fn fused_swiglu_pipelined_shared_compute_matches_full_cpu_reference() {
    // Multiple K tiles exercise accumulation order and the pipeline drain;
    // reversing the rectangular aspect ratio distinguishes M/N indexing.
    check_swiglu(256, 128, 128, true);
}
