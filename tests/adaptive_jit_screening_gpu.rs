//! Hardware regression for adaptive tuning's inexpensive baseline-retention path.
//! Run serially: cargo test --test adaptive_jit_screening_gpu -- --ignored --test-threads=1

use std::path::PathBuf;
use y::adaptive_jit::{AdaptiveGemm, AdaptiveJitConfig, GemmShape, JitTier};
use y::cuda_runtime::{CudaContext, DeviceBuffer};

struct TemporaryCacheDir(PathBuf);

impl TemporaryCacheDir {
    fn new() -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "y-adaptive-screening-gpu-{}-{timestamp}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TemporaryCacheDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// Exact F16 values and an independent F64 scalar reference exercise every
// output, including the incomplete M and K tiles. Distinct streams and signs
// catch operand swaps and transpositions without a half-float dependency.
fn operand(count: usize, mut state: u64) -> (Vec<u8>, Vec<f64>) {
    const VALUES: [(u16, f64); 10] = [
        (0xbc00, -1.0),
        (0xb800, -0.5),
        (0xb400, -0.25),
        (0x3400, 0.25),
        (0x3800, 0.5),
        (0x3c00, 1.0),
        (0x3e00, 1.5),
        (0xbe00, -1.5),
        (0x3000, 0.125),
        (0xb000, -0.125),
    ];
    let mut bytes = Vec::with_capacity(count * 2);
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let (bits, value) = VALUES[(state % VALUES.len() as u64) as usize];
        bytes.extend_from_slice(&bits.to_ne_bytes());
        values.push(value);
    }
    (bytes, values)
}

struct GemmCase {
    shape: GemmShape,
    a: DeviceBuffer,
    b: DeviceBuffer,
    c: DeviceBuffer,
    reference: Vec<f64>,
}

impl GemmCase {
    fn new(ctx: &CudaContext) -> Self {
        let shape = GemmShape {
            m: 31,
            n: 48,
            k: 80,
        };
        let (m, n, k) = (shape.m as usize, shape.n as usize, shape.k as usize);
        let (a_bytes, a_host) = operand(m * k, 0x17c5_002b_94a1_7f63);
        let (b_bytes, b_host) = operand(k * n, 0xabc5_f829_179f_0821);
        let a = ctx.alloc(a_bytes.len()).unwrap();
        let b = ctx.alloc(b_bytes.len()).unwrap();
        let c = ctx.alloc(m * n * 4).unwrap();
        ctx.memcpy_htod_at(&a, 0, &a_bytes).unwrap();
        ctx.memcpy_htod_at(&b, 0, &b_bytes).unwrap();
        let mut reference = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                reference[row * n + col] = (0..k)
                    .map(|inner| a_host[row * k + inner] * b_host[inner * n + col])
                    .sum();
            }
        }
        Self {
            shape,
            a,
            b,
            c,
            reference,
        }
    }

    fn output_bytes(&self, ctx: &CudaContext) -> Vec<u8> {
        let mut output = vec![0; self.reference.len() * 4];
        ctx.memcpy_dtoh_at(&mut output, &self.c, 0).unwrap();
        output
    }

    fn launch_and_check(&self, ctx: &CudaContext, jit: &mut AdaptiveGemm<'_>) {
        ctx.memset_u8(&self.c, 0xff).unwrap();
        // All buffers belong to the current context, have the required
        // element types and extents, and remain alive until synchronization.
        unsafe {
            jit.launch(
                self.shape,
                self.a.device_ptr(),
                self.b.device_ptr(),
                self.c.device_ptr(),
            )
            .unwrap();
        }
        ctx.synchronize().unwrap();
        for (index, (bytes, expected)) in self
            .output_bytes(ctx)
            .chunks_exact(4)
            .zip(&self.reference)
            .enumerate()
        {
            let actual = f64::from(f32::from_ne_bytes(bytes.try_into().unwrap()));
            assert!(
                actual.is_finite() && (actual - expected).abs() <= 0.0005,
                "output {index}: got {actual}, expected {expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU; run serially without other GPU measurements"]
fn screened_baseline_stays_correct_and_persists_without_repeated_tuning() {
    let directory = TemporaryCacheDir::new();
    let ctx = CudaContext::new().expect("this explicitly requested test requires CUDA");
    let case = GemmCase::new(&ctx);
    let shape = case.shape;
    let config = AdaptiveJitConfig {
        cache_dir: Some(directory.0.clone()),
        hot_threshold: 1,
        max_candidates: 8,
        // A deliberately strict target exercises retention independently of
        // small timing differences. Wall-clock budgets belong in benchmarks,
        // not this correctness test.
        min_improvement: 0.95,
        ..Default::default()
    };
    let mut first = AdaptiveGemm::new(&ctx, config.clone()).unwrap();
    first.prepare(shape).unwrap();
    assert_eq!(first.stats(shape).unwrap().launches, 0);
    case.launch_and_check(&ctx, &mut first);
    assert_eq!(first.pending_shapes(), vec![shape]);

    // Measurement must use its own scratch buffers, leaving user output intact.
    ctx.memset_u8(&case.c, 0xa5).unwrap();
    ctx.synchronize().unwrap();
    let reports = first.tune_hot(1).unwrap();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].shape, shape);
    let trained = reports[0].stats.clone();
    assert_eq!(trained.tier, JitTier::RetainedBaseline, "{trained:?}");
    assert_eq!(trained.tuning_attempts, 1);
    assert!(trained.candidates_measured > 0, "{trained:?}");
    assert!(trained.candidates_measured <= config.max_candidates);
    assert!(trained
        .baseline_us
        .is_some_and(|us| us.is_finite() && us > 0.0));
    assert_eq!(trained.baseline_us, trained.selected_us);
    assert!(trained.tuning_error.is_none(), "{trained:?}");
    assert!(trained.cache_error.is_none(), "{trained:?}");
    assert!(!trained.cache_hit);
    assert_eq!(trained.estimated_break_even_launches(), None);
    assert!(case.output_bytes(&ctx).iter().all(|byte| *byte == 0xa5));
    for _ in 0..3 {
        case.launch_and_check(&ctx, &mut first);
        assert!(first.pending_shapes().is_empty());
        assert!(first.tune_hot(1).unwrap().is_empty());
        assert_eq!(first.stats(shape).unwrap().tuning_attempts, 1);
    }
    assert_eq!(
        std::fs::read_dir(&directory.0)
            .unwrap()
            .filter(|entry| entry
                .as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|e| e == "yjit"))
            .count(),
        1
    );
    drop(first);

    let mut restored = AdaptiveGemm::new(&ctx, config).unwrap();
    restored.prepare(shape).unwrap();
    let cached = restored.stats(shape).unwrap();
    assert_eq!(cached.tier, JitTier::RetainedBaseline);
    assert!(cached.cache_hit, "{cached:?}");
    assert!(cached.cache_error.is_none(), "{cached:?}");
    assert!(cached.tuning_error.is_none(), "{cached:?}");
    assert_eq!(cached.baseline_us, trained.baseline_us);
    assert_eq!(cached.selected_us, trained.selected_us);
    assert_eq!(cached.launches, 0);
    assert_eq!(cached.candidates_measured, 0);
    assert_eq!(cached.tuning_attempts, 0);
    assert!(cached.tuning_time.is_zero());
    for _ in 0..3 {
        case.launch_and_check(&ctx, &mut restored);
        assert!(restored.pending_shapes().is_empty());
        assert!(restored.tune_hot(1).unwrap().is_empty());
        assert_eq!(restored.stats(shape).unwrap().tuning_attempts, 0);
        assert!(restored.stats(shape).unwrap().tuning_time.is_zero());
    }
}
