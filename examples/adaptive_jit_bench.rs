//! Internal worker for tools/benchmark_adaptive_jit.py. No runtime behavior changes.
//! Usage: adaptive_jit_bench MODE M N K WEIGHT_COPIES CALLS CACHE_DIR
use std::fmt::Write;
use std::path::PathBuf;
use std::time::Instant;
use y::adaptive_jit::{AdaptiveGemm, AdaptiveJitConfig, GemmShape, JitTier, TuningPolicy};
use y::cuda_runtime::{f16_bits_to_f32, random_f16_bits, CudaContext, DeviceBuffer};

const HOT_THRESHOLD: u64 = 32;
const PAIR_ROUNDS: usize = 15;
const PAIR_CALLS: u64 = 1000;
const RAMP_SECONDS: f64 = 3.0;

struct Inputs {
    a: DeviceBuffer,
    weights: Vec<DeviceBuffer>,
    c: DeviceBuffer,
    shape: GemmShape,
    references: Vec<(usize, f64)>,
}

impl Inputs {
    fn new(ctx: &CudaContext, shape: GemmShape, copies: usize) -> Result<Self, String> {
        let (m, n, k) = (shape.m as usize, shape.n as usize, shape.k as usize);
        let a_host: Vec<u16> = (0..m * k).map(|i| random_f16_bits(i as u64, 123)).collect();
        let b_host: Vec<u16> = (0..k * n).map(|i| random_f16_bits(i as u64, 987)).collect();
        let a = ctx.alloc(a_host.len() * 2)?;
        let b = ctx.alloc(b_host.len() * 2)?;
        let bytes = |values: &[u16]| {
            values
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        ctx.memcpy_htod_at(&a, 0, &bytes(&a_host))?;
        ctx.memcpy_htod_at(&b, 0, &bytes(&b_host))?;
        let mut weights = vec![b];
        for _ in 1..copies {
            let replica = ctx.alloc(b_host.len() * 2)?;
            ctx.memcpy_dtod(&replica, &weights[0])?;
            weights.push(replica);
        }
        let c = ctx.alloc(m * n * 4)?;
        let references = (0..64)
            .map(|sample| {
                let row = sample * 37 % m;
                let col = sample * 71 % n;
                let reference = (0..k)
                    .map(|inner| {
                        f64::from(f16_bits_to_f32(a_host[row * k + inner]))
                            * f64::from(f16_bits_to_f32(b_host[inner * n + col]))
                    })
                    .sum();
                ((row * n + col) * 4, reference)
            })
            .collect();
        Ok(Self {
            a,
            weights,
            c,
            shape,
            references,
        })
    }

    fn enqueue(&self, jit: &mut AdaptiveGemm<'_>, start: u64, count: u64) -> Result<(), String> {
        for i in start..start + count {
            // Allocations are correctly sized, disjoint, and tied to this
            // runtime's context. Every measured batch completes before reuse.
            unsafe {
                jit.launch(
                    self.shape,
                    self.a.device_ptr(),
                    self.weights[i as usize % self.weights.len()].device_ptr(),
                    self.c.device_ptr(),
                )?;
            }
        }
        Ok(())
    }

    fn batch(
        &self,
        ctx: &CudaContext,
        jit: &mut AdaptiveGemm<'_>,
        count: u64,
    ) -> Result<f64, String> {
        ctx.synchronize()?;
        let start = Instant::now();
        self.enqueue(jit, 0, count)?;
        ctx.synchronize()?;
        Ok(start.elapsed().as_secs_f64() * 1e6 / count as f64)
    }

    fn read_error(&self, ctx: &CudaContext) -> Result<f64, String> {
        let (mut squared_error, mut squared_ref) = (0.0, 0.0);
        for &(offset, reference) in &self.references {
            let mut bytes = [0; 4];
            ctx.memcpy_dtoh_at(&mut bytes, &self.c, offset)?;
            let actual = f64::from(f32::from_ne_bytes(bytes));
            squared_error += (actual - reference).powi(2);
            squared_ref += reference.powi(2);
        }
        let error = (squared_error / squared_ref.max(f64::MIN_POSITIVE)).sqrt();
        if !error.is_finite() || error > 0.002 {
            return Err(format!("CPU-reference samples failed: relative L2 {error}"));
        }
        Ok(error)
    }

    fn check(&self, ctx: &CudaContext, jit: &mut AdaptiveGemm<'_>) -> Result<f64, String> {
        ctx.memset_u8(&self.c, 0xff)?;
        self.enqueue(jit, 0, self.weights.len() as u64)?;
        ctx.synchronize()?;
        self.read_error(ctx)
    }
}

fn quote(value: &str) -> String {
    let mut result = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            ch if ch <= '\u{1f}' => {
                write!(result, "\\u{:04x}", ch as u32).unwrap();
            }
            ch => result.push(ch),
        }
    }
    result.push('"');
    result
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!("Internal benchmark worker: MODE M N K WEIGHT_COPIES CALLS CACHE_DIR\nUse python3 tools/benchmark_adaptive_jit.py --help for the complete benchmark.");
        return Ok(());
    }
    if args.len() != 7 {
        return Err("expected MODE M N K WEIGHT_COPIES CALLS CACHE_DIR".into());
    }
    let mode = args[0].as_str();
    if !matches!(mode, "baseline" | "adaptive" | "cached") {
        return Err("MODE must be baseline, adaptive, or cached".into());
    }
    let shape = GemmShape {
        m: args[1].parse().map_err(|_| "invalid M")?,
        n: args[2].parse().map_err(|_| "invalid N")?,
        k: args[3].parse().map_err(|_| "invalid K")?,
    };
    shape.validate()?;
    let copies: usize = args[4].parse().map_err(|_| "invalid WEIGHT_COPIES")?;
    let calls: u64 = args[5].parse().map_err(|_| "invalid CALLS")?;
    if !(1..=16).contains(&copies) || calls < 1000 {
        return Err("WEIGHT_COPIES must be 1..=16 and CALLS must be at least 1000".into());
    }
    let directory = PathBuf::from(&args[6]);
    if mode == "adaptive" {
        // Require a genuinely fresh application decision cache without deleting
        // or overwriting any caller data. This occurs before GPU work/timing.
        std::fs::create_dir(&directory).map_err(|e| format!("fresh cache directory: {e}"))?;
    }
    let config = AdaptiveJitConfig {
        tuning_policy: if mode == "baseline" {
            TuningPolicy::Disabled
        } else {
            TuningPolicy::Deferred
        },
        cache_dir: (mode != "baseline").then_some(directory),
        hot_threshold: HOT_THRESHOLD,
        ..Default::default()
    };
    let baseline_config = AdaptiveJitConfig {
        tuning_policy: TuningPolicy::Disabled,
        ..Default::default()
    };
    let ctx = CudaContext::new().ok_or("a working sm_80+ CUDA GPU is required")?;
    let input = Inputs::new(&ctx, shape, copies)?;

    // Identical warm GPU conditioning in all modes. This necessarily warms the
    // driver's own baseline code cache; 'fresh' below refers to application
    // runtime/decision state, not a pristine CUDA installation or powered-off GPU.
    let mut warmup = AdaptiveGemm::new(&ctx, baseline_config.clone())?;
    warmup.prepare(shape)?;
    let baseline_rel_l2 = input.check(&ctx, &mut warmup)?;
    let ramp = Instant::now();
    while ramp.elapsed().as_secs_f64() < RAMP_SECONDS {
        input.batch(&ctx, &mut warmup, 256)?;
    }
    ctx.synchronize()?;
    drop(warmup);
    ctx.memset_u8(&input.c, 0xff)?;
    ctx.synchronize()?;

    // Everything from runtime creation through the last completion is charged.
    // The only diagnostics inside this interval are the same checkpoint clock
    // reads/vector writes for every mode. Input/CPU checks are outside timing.
    let mut checkpoints = Vec::with_capacity(4);
    let lifecycle = Instant::now();
    let mut jit = AdaptiveGemm::new(&ctx, config)?;
    jit.prepare(shape)?;
    let setup_s = lifecycle.elapsed().as_secs_f64();
    let first = Instant::now();
    input.enqueue(&mut jit, 0, HOT_THRESHOLD)?;
    ctx.synchronize()?;
    let first32_s = first.elapsed().as_secs_f64();
    let maintenance = Instant::now();
    let reports = jit.tune_hot(1)?;
    let maintenance_s = maintenance.elapsed().as_secs_f64();
    let mut completed = HOT_THRESHOLD;
    let mut targets = vec![1000, 10_000, 100_000, calls];
    targets.retain(|n| *n <= calls);
    targets.sort_unstable();
    targets.dedup();
    for target in targets {
        input.enqueue(&mut jit, completed, target - completed)?;
        ctx.synchronize()?;
        checkpoints.push((target, lifecycle.elapsed().as_secs_f64()));
        completed = target;
    }
    let stats = jit
        .stats(shape)
        .ok_or("benchmark shape was evicted")?
        .clone();
    if stats.launches != calls {
        return Err(format!(
            "expected {calls} application calls, got {}",
            stats.launches
        ));
    }
    if stats.tuning_error.is_some() || stats.cache_error.is_some() {
        return Err(format!(
            "tuning/cache failed: {:?} / {:?}",
            stats.tuning_error, stats.cache_error
        ));
    }
    match mode {
        "baseline" if stats.cache_hit || stats.tuning_attempts != 0 || !reports.is_empty() => {
            return Err("baseline unexpectedly tuned/restored".into())
        }
        "adaptive" if stats.cache_hit || stats.tuning_attempts != 1 || reports.len() != 1 => {
            return Err("adaptive worker did not perform exactly one fresh tuning attempt".into())
        }
        "cached" if !stats.cache_hit || stats.tuning_attempts != 0 || !reports.is_empty() => {
            return Err("cached worker did not restore a prior decision without tuning".into())
        }
        _ => {}
    }
    if mode != "baseline" && !matches!(stats.tier, JitTier::Tuned | JitTier::RetainedBaseline) {
        return Err(format!("unsuccessful adaptive decision: {:?}", stats.tier));
    }
    // Check the timed workload output, then poison and independently check the
    // chosen module before believing steady-state timings.
    input.read_error(&ctx)?;
    let selected_rel_l2 = input.check(&ctx, &mut jit)?;

    let mut pairs = Vec::with_capacity(PAIR_ROUNDS);
    let mut reference_setup_s = 0.0;
    if mode != "baseline" {
        let setup = Instant::now();
        let mut baseline = AdaptiveGemm::new(&ctx, baseline_config)?;
        baseline.prepare(shape)?;
        reference_setup_s = setup.elapsed().as_secs_f64();
        input.check(&ctx, &mut baseline)?;
        // Refill working sets after CPU readbacks and module loading. Both
        // variants get the same batch and alternating order in timed rounds.
        input.batch(&ctx, &mut baseline, PAIR_CALLS)?;
        input.batch(&ctx, &mut jit, PAIR_CALLS)?;
        for round in 0..PAIR_ROUNDS {
            let (base_us, selected_us) = if round % 2 == 0 {
                let base = input.batch(&ctx, &mut baseline, PAIR_CALLS)?;
                (base, input.batch(&ctx, &mut jit, PAIR_CALLS)?)
            } else {
                let selected = input.batch(&ctx, &mut jit, PAIR_CALLS)?;
                (input.batch(&ctx, &mut baseline, PAIR_CALLS)?, selected)
            };
            pairs.push((base_us, selected_us));
        }
    }
    ctx.synchronize()?;
    let checkpoints_json = checkpoints
        .iter()
        .map(|(n, t)| format!("{{\"calls\":{n},\"elapsed_s\":{t}}}"))
        .collect::<Vec<_>>()
        .join(",");
    let pairs_json = pairs
        .iter()
        .map(|(b, s)| format!("{{\"baseline_us\":{b},\"selected_us\":{s}}}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        concat!(
            "{{\"mode\":{},\"m\":{},\"n\":{},\"k\":{},\"weight_copies\":{},\"calls\":{},",
            "\"setup_s\":{},\"first32_s\":{},\"maintenance_s\":{},\"tuning_s\":{},",
            "\"cache_hit\":{},\"tier\":{},\"candidates_measured\":{},\"tuning_attempts\":{},",
            "\"checkpoints\":[{}],\"pairs\":[{}],\"reference_setup_s\":{},",
            "\"baseline_rel_l2\":{},\"selected_rel_l2\":{},\"device\":{}}}"
        ),
        quote(mode),
        shape.m,
        shape.n,
        shape.k,
        copies,
        calls,
        setup_s,
        first32_s,
        maintenance_s,
        stats.tuning_time.as_secs_f64(),
        stats.cache_hit,
        quote(&format!("{:?}", stats.tier)),
        stats.candidates_measured,
        stats.tuning_attempts,
        checkpoints_json,
        pairs_json,
        reference_setup_s,
        baseline_rel_l2,
        selected_rel_l2,
        quote(ctx.device_name())
    );
    Ok(())
}
