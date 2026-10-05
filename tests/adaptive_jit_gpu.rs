//! Hardware coverage; run without concurrent GPU benchmarks:
//! cargo test --test adaptive_jit_gpu -- --ignored --test-threads=1

use y::adaptive_jit::{AdaptiveGemm, AdaptiveJitConfig, GemmShape, JitTier, TuningPolicy};
use y::cuda_runtime::{CudaContext, DeviceBuffer};

fn context() -> CudaContext {
    CudaContext::new()
        .expect("these explicitly requested tests require a working NVIDIA CUDA device")
}

// Distinct salts and mixed signs prevent a transpose or a swapped operand
// from accidentally matching the reference. These are exact, normal F16s.
fn input(count: usize, salt: u64) -> (Vec<u8>, Vec<f64>) {
    let mut bytes = Vec::with_capacity(count * 2);
    let mut values = Vec::with_capacity(count);
    let mut state = salt;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let sign = (state >> 63) as u16;
        let exponent = 12 + ((state >> 24) % 3) as u16;
        let fraction = (state & 1023) as u16;
        let bits = (sign << 15) | (exponent << 10) | fraction;
        bytes.extend_from_slice(&bits.to_ne_bytes());
        let magnitude =
            (1.0 + f64::from(fraction) / 1024.0) * 2.0f64.powi(i32::from(exponent) - 15);
        values.push(if sign == 0 { magnitude } else { -magnitude });
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
    fn new(ctx: &CudaContext, shape: GemmShape) -> Self {
        let (m, n, k) = (shape.m as usize, shape.n as usize, shape.k as usize);
        let (a_bytes, a_host) = input(m * k, 0x1234_5678_9abc_def1);
        let (b_bytes, b_host) = input(k * n, 0x9876_5432_fedc_ba91);
        let a = ctx.alloc(a_bytes.len()).unwrap();
        let b = ctx.alloc(b_bytes.len()).unwrap();
        let c = ctx.alloc(m * n * 4).unwrap();
        ctx.memcpy_htod_at(&a, 0, &a_bytes).unwrap();
        ctx.memcpy_htod_at(&b, 0, &b_bytes).unwrap();
        let mut reference = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                for inner in 0..k {
                    reference[row * n + col] += a_host[row * k + inner] * b_host[inner * n + col];
                }
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

    fn launch_and_check(&self, ctx: &CudaContext, jit: &mut AdaptiveGemm<'_>) {
        // Poison output before each launch so repeated reuse cannot pass by
        // leaving a previous correct result untouched, including tail rows.
        ctx.memset_u8(&self.c, 0xff).unwrap();
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
        let mut output = vec![0u8; self.reference.len() * 4];
        ctx.memcpy_dtoh_at(&mut output, &self.c, 0).unwrap();
        for (index, (bytes, &expected)) in output.chunks_exact(4).zip(&self.reference).enumerate() {
            let actual = f64::from(f32::from_ne_bytes(bytes.try_into().unwrap()));
            let tolerance = 0.0005 * (1.0 + expected.abs());
            assert!(actual.is_finite() && (actual - expected).abs() <= tolerance,
                "shape {:?}, output {index}: got {actual}, expected {expected}, tolerance {tolerance}",
                self.shape);
        }
    }
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU; run serially without other GPU measurements"]
fn baseline_hot_tuning_and_reuse_match_cpu_with_ragged_rows() {
    let ctx = context();
    let config = AdaptiveJitConfig {
        tuning_policy: TuningPolicy::OnLaunch,
        hot_threshold: 2,
        max_candidates: 4,
        ..Default::default()
    };
    let mut jit = AdaptiveGemm::new(&ctx, config).unwrap();
    let shape = GemmShape {
        m: 65,
        n: 96,
        k: 128,
    };
    let case = GemmCase::new(&ctx, shape);

    for launches in 1..=2 {
        case.launch_and_check(&ctx, &mut jit);
        let stats = jit.stats(shape).unwrap();
        assert_eq!(stats.launches, launches);
        assert_eq!(stats.tier, JitTier::Baseline);
        assert_eq!(stats.tuning_attempts, 0);
        assert!(stats.baseline_us.is_none());
    }

    case.launch_and_check(&ctx, &mut jit);
    let tuned = jit.stats(shape).unwrap().clone();
    assert_eq!(tuned.launches, 3);
    assert_eq!(tuned.tuning_attempts, 1);
    assert!(
        matches!(tuned.tier, JitTier::Tuned | JitTier::RetainedBaseline),
        "{tuned:?}"
    );
    assert!(tuned.tuning_error.is_none(), "{tuned:?}");
    assert!(!tuned.tuning_time.is_zero());
    let baseline = tuned.baseline_us.unwrap();
    let selected = tuned.selected_us.unwrap();
    assert!(baseline.is_finite() && baseline > 0.0);
    assert!(selected.is_finite() && selected > 0.0);
    if tuned.tier == JitTier::Tuned {
        assert!(selected < baseline * 0.95);
    } else {
        assert_eq!(selected, baseline);
    }
    for launches in 4..=6 {
        case.launch_and_check(&ctx, &mut jit);
        let reused = jit.stats(shape).unwrap();
        assert_eq!(reused.launches, launches);
        assert_eq!(reused.tuning_attempts, 1);
        assert_eq!(reused.tier, tuned.tier);
        assert_eq!(reused.tuning_time, tuned.tuning_time);
        assert_eq!(reused.baseline_us, tuned.baseline_us);
        assert_eq!(reused.selected_us, tuned.selected_us);
    }
    assert_eq!(jit.cached_shapes(), 1);
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU; run serially without other GPU measurements"]
fn cold_shapes_remain_separate_and_evict_the_least_recently_used() {
    let ctx = context();
    let config = AdaptiveJitConfig {
        hot_threshold: u64::MAX,
        max_cached_shapes: 2,
        max_candidates: 4,
        ..Default::default()
    };
    let mut jit = AdaptiveGemm::new(&ctx, config).unwrap();
    // Vary each dimension independently across this set; M=1 and M=17
    // also exercise partial output tiles and decode-sized workloads.
    let a = GemmCase::new(&ctx, GemmShape { m: 1, n: 48, k: 32 });
    let b = GemmCase::new(&ctx, GemmShape { m: 1, n: 48, k: 64 });
    let c = GemmCase::new(
        &ctx,
        GemmShape {
            m: 17,
            n: 48,
            k: 32,
        },
    );
    let d = GemmCase::new(
        &ctx,
        GemmShape {
            m: 17,
            n: 80,
            k: 32,
        },
    );
    a.launch_and_check(&ctx, &mut jit);
    b.launch_and_check(&ctx, &mut jit);
    assert_eq!(jit.cached_shapes(), 2);
    assert_eq!(jit.stats(a.shape).unwrap().launches, 1);
    assert_eq!(jit.stats(b.shape).unwrap().launches, 1);
    a.launch_and_check(&ctx, &mut jit);
    c.launch_and_check(&ctx, &mut jit);
    assert_eq!(jit.cached_shapes(), 2);
    assert!(jit.stats(b.shape).is_none());
    assert_eq!(jit.stats(a.shape).unwrap().launches, 2);
    assert_eq!(jit.stats(c.shape).unwrap().launches, 1);

    b.launch_and_check(&ctx, &mut jit);
    assert!(jit.stats(a.shape).is_none());
    assert_eq!(
        jit.stats(b.shape).unwrap().launches,
        1,
        "eviction resets hotness"
    );
    c.launch_and_check(&ctx, &mut jit);
    d.launch_and_check(&ctx, &mut jit);
    assert!(jit.stats(b.shape).is_none());
    assert_eq!(jit.cached_shapes(), 2);
    assert_eq!(jit.stats(c.shape).unwrap().launches, 2);
    assert_eq!(jit.stats(d.shape).unwrap().launches, 1);
    for shape in [c.shape, d.shape] {
        let stats = jit.stats(shape).unwrap();
        assert_eq!(stats.tier, JitTier::Baseline);
        assert_eq!(stats.tuning_attempts, 0);
    }
    // Include the smallest legal GEMM and a partial K tile beyond 64.
    for shape in [
        GemmShape { m: 1, n: 16, k: 16 },
        GemmShape {
            m: 31,
            n: 48,
            k: 80,
        },
    ] {
        let case = GemmCase::new(&ctx, shape);
        case.launch_and_check(&ctx, &mut jit);
        assert_eq!(jit.cached_shapes(), 2);
        let stats = jit.stats(shape).unwrap();
        assert_eq!(stats.launches, 1);
        assert_eq!(stats.tier, JitTier::Baseline);
        assert_eq!(stats.tuning_attempts, 0);
    }
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU"]
fn rejected_shapes_and_null_or_unaligned_pointers_leave_no_cache_entries() {
    let ctx = context();
    let mut jit = AdaptiveGemm::new(&ctx, AdaptiveJitConfig::default()).unwrap();
    let shape = GemmShape { m: 1, n: 16, k: 16 };
    let case = GemmCase::new(&ctx, shape);
    let valid = [
        case.a.device_ptr(),
        case.b.device_ptr(),
        case.c.device_ptr(),
    ];
    for bad_shape in [GemmShape { m: 0, ..shape }, GemmShape { n: 17, ..shape }] {
        assert!(jit.prepare(bad_shape).is_err());
        assert!(unsafe { jit.launch(bad_shape, valid[0], valid[1], valid[2]) }.is_err());
        assert!(jit.stats(bad_shape).is_none());
    }
    for index in 0..3 {
        for bad_pointer in [0, valid[index] + 1] {
            let mut args = valid;
            args[index] = bad_pointer;
            assert!(unsafe { jit.launch(shape, args[0], args[1], args[2]) }.is_err());
            assert_eq!(jit.cached_shapes(), 0);
            assert!(jit.stats(shape).is_none());
        }
    }
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU"]
fn a_second_context_on_the_same_device_is_rejected_without_switching() {
    let first = context();
    first.require_current().unwrap();
    let mut jit = AdaptiveGemm::new(&first, AdaptiveJitConfig::default()).unwrap();
    let shape = GemmShape { m: 1, n: 16, k: 16 };
    let case = GemmCase::new(&first, shape);
    jit.prepare(shape).unwrap();
    let second = context();
    second.require_current().unwrap();
    assert_eq!(first.device_name(), second.device_name());
    assert!(first.require_current().is_err());
    assert!(AdaptiveGemm::new(&first, AdaptiveJitConfig::default()).is_err());
    assert!(jit.prepare(GemmShape { m: 1, n: 16, k: 16 }).is_err());
    assert!(jit.tune_hot(1).is_err());
    // A resident entry must still reject a switched context before enqueueing.
    assert!(unsafe {
        jit.launch(shape, case.a.device_ptr(), case.b.device_ptr(), case.c.device_ptr())
    }.is_err());
    assert_eq!(jit.cached_shapes(), 1);
    assert_eq!(jit.stats(shape).unwrap().launches, 0);
    assert!(jit.pending_shapes().is_empty());
    second.require_current().unwrap();
    drop(second);
    first.require_current().unwrap();
    case.launch_and_check(&first, &mut jit);
    assert_eq!(jit.stats(shape).unwrap().launches, 1);
    drop(jit);
    drop(case);
    drop(first);
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU; modifies Y_SMEM_PAD, run serially"]
fn tuning_failure_retains_the_working_baseline_and_does_not_retry() {
    struct RestorePadding(Option<std::ffi::OsString>);
    impl Drop for RestorePadding {
        fn drop(&mut self) {
            match &self.0 {
                Some(value) => std::env::set_var("Y_SMEM_PAD", value),
                None => std::env::remove_var("Y_SMEM_PAD"),
            }
        }
    }
    let restore = RestorePadding(std::env::var_os("Y_SMEM_PAD"));
    std::env::remove_var("Y_SMEM_PAD");
    let ctx = context();
    let mut jit = AdaptiveGemm::new(
        &ctx,
        AdaptiveJitConfig {
            tuning_policy: TuningPolicy::OnLaunch,
            hot_threshold: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let shape = GemmShape {
        m: 17,
        n: 48,
        k: 32,
    };
    let case = GemmCase::new(&ctx, shape);
    case.launch_and_check(&ctx, &mut jit);

    // This experimental layout would panic during codegen or emit misaligned
    // shared loads. Reject it before tuning, then execute the cached baseline.
    std::env::set_var("Y_SMEM_PAD", "1");
    assert!(AdaptiveGemm::new(&ctx, AdaptiveJitConfig::default()).is_err());
    case.launch_and_check(&ctx, &mut jit);
    let failed = jit.stats(shape).unwrap();
    assert_eq!(failed.tier, JitTier::TuningFailed);
    assert_eq!(failed.tuning_attempts, 1);
    assert!(failed.tuning_error.as_ref().unwrap().contains("Y_SMEM_PAD"));
    drop(restore);
    for launches in 3..=5 {
        case.launch_and_check(&ctx, &mut jit);
        let stats = jit.stats(shape).unwrap();
        assert_eq!(stats.launches, launches);
        assert_eq!(stats.tuning_attempts, 1);
        assert_eq!(stats.tier, JitTier::TuningFailed);
    }
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU; run serially without other GPU measurements"]
fn prepared_deferred_shapes_tune_only_during_explicit_maintenance() {
    let ctx = context();
    let config = AdaptiveJitConfig {
        hot_threshold: 2,
        max_candidates: 4,
        ..Default::default()
    };
    assert_eq!(config.tuning_policy, TuningPolicy::Deferred);
    let mut jit = AdaptiveGemm::new(&ctx, config).unwrap();
    let shape = GemmShape {
        m: 31,
        n: 48,
        k: 80,
    };

    // Preparing a kernel requires no application buffers and is not a launch.
    jit.prepare(shape).unwrap();
    jit.prepare(shape).unwrap();
    assert_eq!(jit.cached_shapes(), 1);
    let prepared = jit.stats(shape).unwrap();
    assert_eq!(prepared.launches, 0);
    assert_eq!(prepared.tuning_attempts, 0);
    assert_eq!(prepared.tier, JitTier::Baseline);
    assert!(jit.pending_shapes().is_empty());
    assert!(jit.tune_hot(1).unwrap().is_empty());
    let case = GemmCase::new(&ctx, shape);

    for launches in 1..=4 {
        case.launch_and_check(&ctx, &mut jit);
        let stats = jit.stats(shape).unwrap();
        assert_eq!(stats.launches, launches);
        assert_eq!(stats.tuning_attempts, 0);
        assert_eq!(stats.tier, JitTier::Baseline);
        assert!(stats.tuning_time.is_zero());
        assert!(stats.baseline_us.is_none());
        assert!(stats.selected_us.is_none());
        if launches < 2 {
            assert!(jit.pending_shapes().is_empty());
        } else {
            assert_eq!(jit.pending_shapes(), vec![shape]);
        }
    }

    assert!(jit.tune_hot(0).unwrap().is_empty());
    assert_eq!(jit.pending_shapes(), vec![shape]);
    assert_eq!(jit.stats(shape).unwrap().tuning_attempts, 0);
    let reports = jit.tune_hot(1).unwrap();
    assert_eq!(reports.len(), 1);
    let report = &reports[0];
    assert_eq!(report.shape, shape);
    let tuned = &report.stats;
    assert_eq!(
        tuned.launches, 4,
        "scratch measurements are not user launches"
    );
    assert_eq!(tuned.tuning_attempts, 1);
    assert!(
        matches!(tuned.tier, JitTier::Tuned | JitTier::RetainedBaseline),
        "{tuned:?}"
    );
    assert!(tuned.tuning_error.is_none(), "{tuned:?}");
    assert!(!tuned.tuning_time.is_zero());
    let baseline = tuned.baseline_us.unwrap();
    let selected = tuned.selected_us.unwrap();
    assert!(baseline.is_finite() && baseline > 0.0);
    assert!(selected.is_finite() && selected > 0.0);
    if tuned.tier == JitTier::Tuned {
        assert!(selected < baseline * 0.95);
    } else {
        assert_eq!(selected, baseline);
    }
    assert!(jit.pending_shapes().is_empty());
    assert!(jit.tune_hot(1).unwrap().is_empty());
    jit.prepare(shape).unwrap();
    assert_eq!(jit.stats(shape).unwrap().launches, 4);
    for launches in 5..=6 {
        case.launch_and_check(&ctx, &mut jit);
        let stats = jit.stats(shape).unwrap();
        assert_eq!(stats.launches, launches);
        assert_eq!(stats.tuning_attempts, 1);
        assert_eq!(stats.tier, tuned.tier);
        assert_eq!(stats.tuning_time, tuned.tuning_time);
        assert_eq!(stats.baseline_us, tuned.baseline_us);
        assert_eq!(stats.selected_us, tuned.selected_us);
        assert!(jit.pending_shapes().is_empty());
        assert!(jit.tune_hot(1).unwrap().is_empty());
    }
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU"]
fn disabled_tuning_reuses_prepared_kernels_without_pending_work() {
    let ctx = context();
    let mut jit = AdaptiveGemm::new(
        &ctx,
        AdaptiveJitConfig {
            tuning_policy: TuningPolicy::Disabled,
            hot_threshold: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let shape = GemmShape {
        m: 17,
        n: 48,
        k: 32,
    };
    jit.prepare(shape).unwrap();
    let case = GemmCase::new(&ctx, shape);
    for launches in 1..=4 {
        case.launch_and_check(&ctx, &mut jit);
        assert!(jit.pending_shapes().is_empty());
        assert!(jit.tune_hot(1).unwrap().is_empty());
        let stats = jit.stats(shape).unwrap();
        assert_eq!(stats.launches, launches);
        assert_eq!(stats.tuning_attempts, 0);
        assert_eq!(stats.tier, JitTier::Baseline);
        assert!(stats.tuning_time.is_zero());
        assert!(stats.baseline_us.is_none());
        assert!(stats.selected_us.is_none());
    }
    assert_eq!(jit.cached_shapes(), 1);
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU"]
fn preparation_updates_lru_and_eviction_removes_pending_work() {
    let ctx = context();
    let mut jit = AdaptiveGemm::new(
        &ctx,
        AdaptiveJitConfig {
            hot_threshold: 1,
            max_cached_shapes: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let a = GemmShape { m: 1, n: 16, k: 16 };
    let b = GemmShape {
        m: 17,
        n: 16,
        k: 16,
    };
    let c = GemmShape { m: 1, n: 32, k: 16 };
    jit.prepare(a).unwrap();
    jit.prepare(b).unwrap();
    jit.prepare(a).unwrap();
    jit.prepare(c).unwrap();
    assert_eq!(jit.cached_shapes(), 2);
    assert!(
        jit.stats(b).is_none(),
        "preparing a again must refresh its LRU use"
    );
    for shape in [a, c] {
        let stats = jit.stats(shape).unwrap();
        assert_eq!(stats.launches, 0);
        assert_eq!(stats.tuning_attempts, 0);
    }
    assert!(jit.pending_shapes().is_empty());

    GemmCase::new(&ctx, a).launch_and_check(&ctx, &mut jit);
    GemmCase::new(&ctx, c).launch_and_check(&ctx, &mut jit);
    let pending = jit.pending_shapes();
    assert_eq!(pending.len(), 2);
    assert!(pending.contains(&a));
    assert!(pending.contains(&c));
    jit.prepare(b).unwrap();
    assert!(jit.stats(a).is_none());
    assert_eq!(jit.pending_shapes(), vec![c]);
    assert_eq!(jit.stats(b).unwrap().launches, 0);
    jit.prepare(a).unwrap();
    assert!(jit.stats(c).is_none());
    assert!(jit.pending_shapes().is_empty());
    assert!(jit.tune_hot(1).unwrap().is_empty());
    assert_eq!(jit.cached_shapes(), 2);
}

struct TemporaryCacheDir(std::path::PathBuf);

impl TemporaryCacheDir {
    fn new(label: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "y-adaptive-jit-gpu-{label}-{}-{unique}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn records(&self) -> Vec<std::path::PathBuf> {
        std::fs::read_dir(&self.0)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "yjit")
            })
            .collect()
    }
}

impl Drop for TemporaryCacheDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU; run serially without other GPU measurements"]
fn persisted_decisions_reuse_across_runtimes_and_invalid_records_fall_back() {
    let directory = TemporaryCacheDir::new("reuse");
    let ctx = context();
    let shape = GemmShape {
        m: 31,
        n: 48,
        k: 80,
    };
    let case = GemmCase::new(&ctx, shape);
    let config = AdaptiveJitConfig {
        cache_dir: Some(directory.0.clone()),
        max_disk_cache_entries: 4,
        hot_threshold: 2,
        max_candidates: 4,
        ..Default::default()
    };
    let mut first = AdaptiveGemm::new(&ctx, config.clone()).unwrap();
    first.prepare(shape).unwrap();
    assert!(!first.stats(shape).unwrap().cache_hit);
    for _ in 0..config.hot_threshold {
        case.launch_and_check(&ctx, &mut first);
    }
    assert_eq!(first.pending_shapes(), vec![shape]);
    let reports = first.tune_hot(1).unwrap();
    assert_eq!(reports.len(), 1);
    let trained = reports[0].stats.clone();
    assert!(
        matches!(trained.tier, JitTier::Tuned | JitTier::RetainedBaseline),
        "{trained:?}"
    );
    assert_eq!(trained.tuning_attempts, 1);
    assert!(trained.tuning_error.is_none(), "{trained:?}");
    assert!(trained.cache_error.is_none(), "{trained:?}");
    assert!(!trained.cache_hit);
    case.launch_and_check(&ctx, &mut first);
    let records = directory.records();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    let stem = record.file_stem().unwrap().to_str().unwrap();
    assert_eq!(stem.len(), 64);
    assert!(stem.bytes().all(|byte| byte.is_ascii_hexdigit()));
    drop(first);

    // A retained baseline is a successful measured decision too. Do not
    // require a noisy timing run to select a different tile to pass this test.
    let mut restored = AdaptiveGemm::new(&ctx, config.clone()).unwrap();
    restored.prepare(shape).unwrap();
    let stats = restored.stats(shape).unwrap();
    assert!(stats.cache_hit, "{stats:?}");
    assert!(stats.cache_error.is_none(), "{stats:?}");
    assert_eq!(stats.tier, trained.tier);
    assert_eq!(stats.baseline_us, trained.baseline_us);
    assert_eq!(stats.selected_us, trained.selected_us);
    assert_eq!(stats.launches, 0);
    assert_eq!(stats.tuning_attempts, 0);
    assert_eq!(stats.candidates_measured, 0);
    assert!(stats.tuning_time.is_zero());
    for launches in 1..=config.hot_threshold + 2 {
        case.launch_and_check(&ctx, &mut restored);
        assert!(restored.pending_shapes().is_empty());
        assert!(restored.tune_hot(1).unwrap().is_empty());
        let stats = restored.stats(shape).unwrap();
        assert_eq!(stats.launches, launches);
        assert_eq!(stats.tuning_attempts, 0);
        assert_eq!(stats.tier, trained.tier);
        assert!(stats.cache_hit);
        assert!(stats.tuning_time.is_zero());
    }
    drop(restored);

    // Changing either the search budget or the promotion threshold requires
    // a new measurement, even when an otherwise matching record exists.
    for changed in [
        AdaptiveJitConfig {
            min_improvement: 0.15,
            ..config.clone()
        },
        AdaptiveJitConfig {
            max_candidates: config.max_candidates + 1,
            ..config.clone()
        },
    ] {
        let mut jit = AdaptiveGemm::new(&ctx, changed).unwrap();
        jit.prepare(shape).unwrap();
        let stats = jit.stats(shape).unwrap();
        assert!(!stats.cache_hit, "{stats:?}");
        assert_eq!(stats.tier, JitTier::Baseline);
        assert_eq!(stats.tuning_attempts, 0);
        for _ in 0..config.hot_threshold {
            case.launch_and_check(&ctx, &mut jit);
        }
        assert_eq!(jit.pending_shapes(), vec![shape]);
    }
    assert_eq!(directory.records(), records);

    // Recompute the checksum after semantic corruption so these cases reach
    // the runtime's compiler-identity and candidate-membership checks.
    use sha2::Digest;
    let original_record = std::fs::read(record).unwrap();
    let original_text = std::str::from_utf8(&original_record).unwrap();
    let (body, _) = original_text.rsplit_once("checksum=").unwrap();
    for (field, value, expected_error) in [
        ("baseline_hash", "0".repeat(64), "baseline"),
        ("selected_hash", "0".repeat(64), "current compilation"),
        ("cta_m", "1024".to_owned(), "search space"),
    ] {
        let prefix = format!("{field}=");
        let original_line = body.lines().find(|line| line.starts_with(&prefix)).unwrap();
        let changed_line = format!("{field}={value}");
        assert_ne!(original_line, changed_line);
        let changed_body =
            body.replace(&format!("{original_line}\n"), &format!("{changed_line}\n"));
        let resigned = format!(
            "{changed_body}checksum={:x}\n",
            sha2::Sha256::digest(changed_body.as_bytes())
        );
        std::fs::write(record, resigned).unwrap();
        let mut rejected = AdaptiveGemm::new(&ctx, config.clone()).unwrap();
        rejected.prepare(shape).unwrap();
        let stats = rejected.stats(shape).unwrap();
        assert!(!stats.cache_hit, "accepted corrupt {field}: {stats:?}");
        assert_eq!(stats.tier, JitTier::Baseline);
        assert_eq!(stats.tuning_attempts, 0);
        assert!(stats.tuning_error.is_none(), "{stats:?}");
        assert!(
            stats
                .cache_error
                .as_ref()
                .is_some_and(|error| error.contains(expected_error)),
            "wrong rejection for {field}: {stats:?}"
        );
        case.launch_and_check(&ctx, &mut rejected);
        drop(rejected);
        std::fs::write(record, &original_record).unwrap();
    }

    std::fs::write(record, b"corrupt and truncated tuning record\n").unwrap();
    let mut corrupt = AdaptiveGemm::new(&ctx, config.clone()).unwrap();
    corrupt.prepare(shape).unwrap();
    let stats = corrupt.stats(shape).unwrap();
    assert!(!stats.cache_hit);
    assert!(stats.cache_error.is_some(), "{stats:?}");
    assert!(stats.tuning_error.is_none(), "{stats:?}");
    assert_eq!(stats.tier, JitTier::Baseline);
    for _ in 0..config.hot_threshold {
        case.launch_and_check(&ctx, &mut corrupt);
    }
    assert_eq!(corrupt.pending_shapes(), vec![shape]);
    assert_eq!(corrupt.stats(shape).unwrap().tuning_attempts, 0);
    drop(corrupt);

    // Disabled must not even diagnose the invalid disk record: persistence
    // and empirical tuning are both outside this policy's execution path.
    let mut disabled = AdaptiveGemm::new(
        &ctx,
        AdaptiveJitConfig {
            tuning_policy: TuningPolicy::Disabled,
            ..config
        },
    )
    .unwrap();
    disabled.prepare(shape).unwrap();
    for _ in 0..4 {
        case.launch_and_check(&ctx, &mut disabled);
    }
    let stats = disabled.stats(shape).unwrap();
    assert!(!stats.cache_hit);
    assert!(stats.cache_error.is_none(), "{stats:?}");
    assert_eq!(stats.tier, JitTier::Baseline);
    assert_eq!(stats.tuning_attempts, 0);
    assert!(disabled.pending_shapes().is_empty());
    assert!(disabled.tune_hot(1).unwrap().is_empty());
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU; run serially without other GPU measurements"]
fn disk_cache_failures_leave_successfully_tuned_kernels_usable() {
    let directory = TemporaryCacheDir::new("io-failure");
    let file = directory.0.join("regular-file");
    std::fs::write(&file, b"preserve this existing file").unwrap();
    let ctx = context();
    let shape = GemmShape {
        m: 17,
        n: 48,
        k: 32,
    };
    let case = GemmCase::new(&ctx, shape);
    let mut jit = AdaptiveGemm::new(
        &ctx,
        AdaptiveJitConfig {
            cache_dir: Some(file.clone()),
            hot_threshold: 1,
            max_candidates: 4,
            ..Default::default()
        },
    )
    .unwrap();
    jit.prepare(shape).unwrap();
    let cold = jit.stats(shape).unwrap();
    assert_eq!(cold.tier, JitTier::Baseline);
    assert!(cold.cache_error.is_some(), "{cold:?}");
    assert!(!cold.cache_hit);
    case.launch_and_check(&ctx, &mut jit);
    assert_eq!(jit.pending_shapes(), vec![shape]);
    let reports = jit.tune_hot(1).unwrap();
    assert_eq!(reports.len(), 1);
    let tuned = &reports[0].stats;
    assert!(
        matches!(tuned.tier, JitTier::Tuned | JitTier::RetainedBaseline),
        "{tuned:?}"
    );
    assert_eq!(tuned.tuning_attempts, 1);
    assert!(tuned.tuning_error.is_none(), "{tuned:?}");
    assert!(tuned.cache_error.is_some(), "{tuned:?}");
    assert!(!tuned.cache_hit);
    for _ in 0..3 {
        case.launch_and_check(&ctx, &mut jit);
        assert!(jit.pending_shapes().is_empty());
        assert!(jit.tune_hot(1).unwrap().is_empty());
        let stats = jit.stats(shape).unwrap();
        assert_eq!(stats.tier, tuned.tier);
        assert_eq!(stats.tuning_attempts, 1);
        assert!(stats.tuning_error.is_none(), "{stats:?}");
    }
    assert_eq!(
        std::fs::read(&file).unwrap(),
        b"preserve this existing file"
    );
}
