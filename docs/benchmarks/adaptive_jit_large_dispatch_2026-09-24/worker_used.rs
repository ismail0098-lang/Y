//! Focused resident GEMM dispatch benchmark; compilation and tuning are excluded.
//! Usage: adaptive_jit_dispatch_bench M N K [disabled|deferred [CALLS [ROUNDS]]]
use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use y::adaptive_jit::{AdaptiveGemm, AdaptiveJitConfig, GemmShape, TuningPolicy};
use y::cuda_runtime::{f16_bits_to_f32, random_f16_bits, CudaContext, DeviceBuffer};

// Count Rust allocations only in a separate untimed audit. CUDA's internal
// allocations are outside this allocator. Timing has counting disabled, with
// the same allocator wrapper in both preserved before/after executables.
struct AuditAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for AuditAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}
#[global_allocator]
static ALLOCATOR: AuditAllocator = AuditAllocator;
const DEFAULT_CALLS: u64 = 1000;
const DEFAULT_ROUNDS: usize = 41;

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
    if !(3..=6).contains(&args.len()) {
        return Err(
            "usage: adaptive_jit_dispatch_bench M N K [disabled|deferred [CALLS [ROUNDS]]]".into(),
        );
    }
    let shape = GemmShape {
        m: args[0].parse().map_err(|_| "invalid M")?,
        n: args[1].parse().map_err(|_| "invalid N")?,
        k: args[2].parse().map_err(|_| "invalid K")?,
    };
    shape.validate()?;
    let policy = args.get(3).map(String::as_str).unwrap_or("disabled");
    let tuning_policy = match policy {
        "disabled" => TuningPolicy::Disabled,
        "deferred" => TuningPolicy::Deferred,
        _ => return Err("policy must be disabled or deferred".into()),
    };
    let calls: u64 = args
        .get(4)
        .map(|s| s.parse())
        .transpose()
        .map_err(|_| "invalid CALLS")?
        .unwrap_or(DEFAULT_CALLS);
    let rounds_count: usize = args
        .get(5)
        .map(|s| s.parse())
        .transpose()
        .map_err(|_| "invalid ROUNDS")?
        .unwrap_or(DEFAULT_ROUNDS);
    if !(1..=1_000_000).contains(&calls) || !(3..=10_000).contains(&rounds_count) {
        return Err("CALLS must be 1..=1000000 and ROUNDS must be 3..=10000".into());
    }
    let ctx = CudaContext::new().ok_or("a working sm_80+ CUDA GPU is required")?;
    let input = Inputs::new(&ctx, shape, 1)?;
    let mut jit = AdaptiveGemm::new(
        &ctx,
        AdaptiveJitConfig {
            tuning_policy,
            ..Default::default()
        },
    )?;
    jit.prepare(shape)?;
    let initial_rel_l2 = input.check(&ctx, &mut jit)?;
    let ramp = Instant::now();
    while ramp.elapsed().as_secs_f64() < 3.0 {
        input.batch(&ctx, &mut jit, 256)?;
    }
    ctx.synchronize()?;
    let launches_before = jit.stats(shape).ok_or("missing shape")?.launches;
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let audit = input.enqueue(&mut jit, 0, calls);
    COUNTING.store(false, Ordering::Relaxed);
    audit?;
    ctx.synchronize()?;
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let mut rounds = Vec::with_capacity(rounds_count);
    for _ in 0..rounds_count {
        ctx.synchronize()?;
        let start = Instant::now();
        input.enqueue(&mut jit, 0, calls)?;
        let enqueue_us = start.elapsed().as_secs_f64() * 1e6 / calls as f64;
        ctx.synchronize()?;
        let completed_us = start.elapsed().as_secs_f64() * 1e6 / calls as f64;
        rounds.push((enqueue_us, completed_us));
    }
    let stats = jit.stats(shape).ok_or("missing shape")?;
    if stats.launches - launches_before != calls * (rounds_count as u64 + 1)
        || stats.tuning_attempts != 0
        || stats.cache_hit
    {
        return Err("unexpected dispatch counters or tuning/cache activity".into());
    }
    let final_rel_l2 = input.read_error(&ctx)?;
    let round_json = rounds
        .iter()
        .map(|(enqueue, complete)| {
            format!("{{\"enqueue_us\":{enqueue},\"completed_us\":{complete}}}")
        })
        .collect::<Vec<_>>()
        .join(",");
    println!(
        concat!(
            "{{\"shape\":[{},{},{}],\"policy\":{},\"calls_per_round\":{},",
            "\"audit_calls\":{},\"audit_rust_allocations\":{},\"rounds\":[{}],",
            "\"initial_rel_l2\":{},\"final_rel_l2\":{},\"device\":{}}}"
        ),
        shape.m,
        shape.n,
        shape.k,
        quote(policy),
        calls,
        calls,
        allocations,
        round_json,
        initial_rel_l2,
        final_rel_l2,
        quote(ctx.device_name())
    );
    Ok(())
}
