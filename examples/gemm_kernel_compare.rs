//! Compare preserved PTX artifacts in the same context with paired ordering.
//! export OUTDIR M N K
//! compare BEFOREDIR AFTERDIR M N K [CALLS [ROUNDS]]
//! Compilation, allocation, validation and a three-second warmup are untimed.
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::Path;
use std::time::{Duration, Instant};
use y::adaptive_jit::{AdaptiveJitConfig, GemmShape};
use y::autotuner::{Autotuner, Precision};
use y::cuda_runtime::{f16_bits_to_f32, random_f16_bits, CudaContext, DeviceBuffer, KernelModule};
use y::empirical_autotune::{emit_candidate_ptx, is_emittable, LaunchConfig};
use y::sentinel::HardwareProfile;

const KERNEL_NAME: &str = "y_autotune_probe";
const MAX_DEVICE_BYTES: usize = 1024 * 1024 * 1024;
const SOURCE: &[(&str, &str)] = &[
    ("src/ptx_emitter.rs", include_str!("../src/ptx_emitter.rs")),
    ("src/autotuner.rs", include_str!("../src/autotuner.rs")),
    (
        "src/empirical_autotune.rs",
        include_str!("../src/empirical_autotune.rs"),
    ),
    (
        "src/cuda_runtime.rs",
        include_str!("../src/cuda_runtime.rs"),
    ),
    (
        "src/adaptive_jit.rs",
        include_str!("../src/adaptive_jit.rs"),
    ),
];

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn quote(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            ch if ch <= '\u{1f}' => write!(out, "\\u{:04x}", ch as u32).unwrap(),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn profile(ctx: &CudaContext) -> Result<HardwareProfile, String> {
    // Keep these attributes identical to AdaptiveGemm::with_context.
    let attr = |id| {
        ctx.device_attribute(id)
            .filter(|v| *v > 0)
            .map(|v| v as u32)
            .ok_or_else(|| format!("could not query CUDA device attribute {id}"))
    };
    let major = attr(75)?;
    let minor = ctx
        .device_attribute(76)
        .filter(|v| *v >= 0)
        .ok_or("missing CUDA minor")?;
    if major < 8 {
        return Err("benchmark requires sm_80 or newer".into());
    }
    let max_threads_per_sm = attr(39)?;
    let warp_size = attr(10)?;
    Ok(HardwareProfile {
        gpu_name: ctx.device_name().to_owned(),
        gpu_vendor: "NVIDIA".into(),
        sm_version: format!("sm_{major}{minor}"),
        compute_capability: format!("{major}.{minor}"),
        sm_count: attr(16)?,
        max_smem_per_sm_bytes: attr(81)?.min(attr(97)?),
        max_regs_per_sm: attr(82)?,
        max_regs_per_thread: 255,
        max_threads_per_sm,
        warp_size,
        max_warps_per_sm: max_threads_per_sm / warp_size,
        ..HardwareProfile::default()
    })
}

fn stem(shape: GemmShape) -> String {
    format!("{}_{}_{}", shape.m, shape.n, shape.k)
}

fn load(ctx: &CudaContext, ptx: &str, launch: LaunchConfig) -> Result<KernelModule, String> {
    let module = ctx.load_ptx(ptx, KERNEL_NAME)?;
    if launch.dyn_smem_bytes > 0 {
        module.set_max_dynamic_smem(launch.dyn_smem_bytes)?;
    }
    Ok(module)
}

fn export(ctx: &CudaContext, directory: &Path, shape: GemmShape) -> Result<(), String> {
    let hw = profile(ctx)?;
    let mut candidates = Autotuner::generate_candidates(shape.m, shape.n, shape.k, Precision::F16);
    candidates.retain(|c| is_emittable(c, shape.k));
    candidates.sort_by(|a, b| {
        let score = |c| Autotuner::score_candidate(c, shape.m, shape.n, shape.k, &hw);
        score(b).total_cmp(&score(a))
    });
    let mut last_error = "no emittable candidate".to_owned();
    for candidate in candidates
        .into_iter()
        .take(AdaptiveJitConfig::default().max_candidates)
    {
        let generated = emit_candidate_ptx(shape.m, shape.n, shape.k, &candidate, &hw)
            .and_then(|(ptx, launch)| load(ctx, &ptx, launch).map(|_| (ptx, launch)));
        let (ptx, launch) = match generated {
            Ok(value) => value,
            Err(error) => {
                last_error = error;
                continue;
            }
        };
        let mut manifest = format!(
            "version=1\nm={}\nn={}\nk={}\nentry={}\ngrid_x={}\ngrid_y={}\nthreads={}\ndyn_smem_bytes={}\ncandidate={:?}\nsm_version={}\nptx_sha256={}\n",
            shape.m, shape.n, shape.k, KERNEL_NAME, launch.grid_x, launch.grid_y,
            launch.threads, launch.dyn_smem_bytes, candidate, hw.sm_version, hash(ptx.as_bytes())
        );
        for (path, source) in SOURCE {
            writeln!(manifest, "source_sha256:{path}={}", hash(source.as_bytes())).unwrap();
        }
        std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
        let name = stem(shape);
        std::fs::write(directory.join(format!("{name}.ptx")), &ptx).map_err(|e| e.to_string())?;
        std::fs::write(directory.join(format!("{name}.manifest")), &manifest)
            .map_err(|e| e.to_string())?;
        println!(
            "{{\"mode\":\"export\",\"shape\":[{},{},{}],\"ptx_sha256\":{},\"candidate\":{}}}",
            shape.m,
            shape.n,
            shape.k,
            quote(&hash(ptx.as_bytes())),
            quote(&format!("{candidate:?}"))
        );
        return Ok(());
    }
    Err(format!("could not compile adaptive baseline: {last_error}"))
}

struct Artifact {
    ptx: String,
    launch: LaunchConfig,
    manifest: BTreeMap<String, String>,
}

fn artifact(directory: &Path, shape: GemmShape) -> Result<Artifact, String> {
    let name = stem(shape);
    let ptx = std::fs::read_to_string(directory.join(format!("{name}.ptx")))
        .map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(directory.join(format!("{name}.manifest")))
        .map_err(|e| e.to_string())?;
    let mut manifest = BTreeMap::new();
    for line in text.lines() {
        let (key, value) = line.split_once('=').ok_or("malformed manifest")?;
        if manifest.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(format!("duplicate manifest field {key}"));
        }
    }
    let number = |key: &str| -> Result<u32, String> {
        manifest
            .get(key)
            .ok_or_else(|| format!("missing manifest field {key}"))?
            .parse()
            .map_err(|_| format!("invalid manifest number {key}"))
    };
    if number("version")? != 1
        || number("m")? != shape.m
        || number("n")? != shape.n
        || number("k")? != shape.k
    {
        return Err("manifest version or shape mismatch".into());
    }
    if manifest.get("entry").map(String::as_str) != Some(KERNEL_NAME)
        || manifest.get("ptx_sha256") != Some(&hash(ptx.as_bytes()))
    {
        return Err("manifest entry or PTX checksum mismatch".into());
    }
    let launch = LaunchConfig {
        grid_x: number("grid_x")?,
        grid_y: number("grid_y")?,
        threads: number("threads")?,
        dyn_smem_bytes: number("dyn_smem_bytes")?,
    };
    if launch.grid_x == 0 || launch.grid_y == 0 || launch.threads == 0 || launch.threads > 1024 {
        return Err("invalid launch geometry".into());
    }
    Ok(Artifact {
        ptx,
        launch,
        manifest,
    })
}

struct Inputs {
    a: DeviceBuffer,
    b: DeviceBuffer,
    outputs: [DeviceBuffer; 2],
    references: Vec<(usize, f64)>,
    output_bytes: usize,
}

impl Inputs {
    fn new(ctx: &CudaContext, shape: GemmShape) -> Result<Self, String> {
        let (m, n, k) = (shape.m as usize, shape.n as usize, shape.k as usize);
        let required = 2 * m * k + 2 * k * n + 8 * m * n;
        if required > MAX_DEVICE_BYTES {
            return Err(format!(
                "benchmark buffers require {required} bytes; limit is {MAX_DEVICE_BYTES}"
            ));
        }
        if let Some((free, _)) = ctx.mem_info() {
            if required > free.saturating_sub(256 * 1024 * 1024) {
                return Err("insufficient free GPU memory with 256 MiB reserve".into());
            }
        }
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
        let mut indices = BTreeSet::new();
        if m * n <= 4096 {
            indices.extend(0..m * n);
        } else {
            indices.extend([0, n - 1, (m - 1) * n, m * n - 1]);
            let mut state = 0x72ef_d431_3187_ab09_u64;
            while indices.len() < 128 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                indices.insert(state as usize % (m * n));
            }
        }
        let references = indices
            .into_iter()
            .map(|index| {
                let (row, col) = (index / n, index % n);
                let expected = (0..k)
                    .map(|inner| {
                        f64::from(f16_bits_to_f32(a_host[row * k + inner]))
                            * f64::from(f16_bits_to_f32(b_host[inner * n + col]))
                    })
                    .sum();
                (index, expected)
            })
            .collect();
        let output_bytes = m * n * 4;
        let outputs = [ctx.alloc(output_bytes)?, ctx.alloc(output_bytes)?];
        Ok(Self {
            a,
            b,
            outputs,
            references,
            output_bytes,
        })
    }

    fn args(&self, side: usize) -> [u64; 3] {
        [
            self.a.device_ptr(),
            self.b.device_ptr(),
            self.outputs[side].device_ptr(),
        ]
    }

    fn check(&self, ctx: &CudaContext) -> Result<[f64; 2], String> {
        ctx.synchronize()?;
        let mut results = [Vec::new(), Vec::new()];
        let mut errors = [0.0; 2];
        for side in 0..2 {
            results[side].resize(self.output_bytes, 0);
            ctx.memcpy_dtoh_at(&mut results[side], &self.outputs[side], 0)?;
            if results[side]
                .chunks_exact(4)
                .any(|bytes| !f32::from_ne_bytes(bytes.try_into().unwrap()).is_finite())
            {
                return Err(format!("nonfinite output on side {side}"));
            }
            let (mut squared_error, mut squared_reference) = (0.0, 0.0);
            for &(index, reference) in &self.references {
                let value =
                    f32::from_ne_bytes(results[side][index * 4..index * 4 + 4].try_into().unwrap())
                        as f64;
                squared_error += (value - reference).powi(2);
                squared_reference += reference.powi(2);
            }
            errors[side] = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
            if !errors[side].is_finite() || errors[side] > 0.002 {
                return Err(format!(
                    "CPU reference mismatch on side {side}: relative L2 {}",
                    errors[side]
                ));
            }
        }
        if results[0] != results[1] {
            let mismatches = results[0]
                .chunks_exact(4)
                .zip(results[1].chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count();
            return Err(format!(
                "before/after outputs differ bitwise at {mismatches} elements"
            ));
        }
        Ok(errors)
    }
}

fn enqueue(
    ctx: &CudaContext,
    kernel: &KernelModule,
    launch: LaunchConfig,
    args: &[u64],
    calls: u32,
) -> Result<(), String> {
    for _ in 0..calls {
        ctx.launch(
            kernel,
            (launch.grid_x, launch.grid_y, 1),
            (launch.threads, 1, 1),
            launch.dyn_smem_bytes,
            args,
        )?;
    }
    Ok(())
}

fn compare(
    ctx: &CudaContext,
    before: &Path,
    after: &Path,
    shape: GemmShape,
    calls: u32,
    rounds: usize,
) -> Result<(), String> {
    let artifacts = [artifact(before, shape)?, artifact(after, shape)?];
    for key in [
        "candidate",
        "sm_version",
        "grid_x",
        "grid_y",
        "threads",
        "dyn_smem_bytes",
    ] {
        if artifacts[0].manifest.get(key).is_none()
            || artifacts[0].manifest.get(key) != artifacts[1].manifest.get(key)
        {
            return Err(format!(
                "before/after manifest field {key} differs or is missing"
            ));
        }
    }
    let hw = profile(ctx)?;
    if artifacts[0].manifest.get("sm_version") != Some(&hw.sm_version) {
        return Err("artifact architecture does not match the current GPU".into());
    }
    let kernels = [
        load(ctx, &artifacts[0].ptx, artifacts[0].launch)?,
        load(ctx, &artifacts[1].ptx, artifacts[1].launch)?,
    ];
    let inputs = Inputs::new(ctx, shape)?;
    let args = [inputs.args(0), inputs.args(1)];
    let event_args = [vec![args[0].to_vec()], vec![args[1].to_vec()]];
    for side in 0..2 {
        ctx.memset_u8(&inputs.outputs[side], 0xff)?;
        enqueue(ctx, &kernels[side], artifacts[side].launch, &args[side], 1)?;
    }
    let initial_errors = inputs.check(ctx)?;
    let warm_start = Instant::now();
    let mut warm_round = 0;
    while warm_start.elapsed() < Duration::from_secs(3) {
        for slot in 0..2 {
            let side = (slot + warm_round) % 2;
            enqueue(
                ctx,
                &kernels[side],
                artifacts[side].launch,
                &args[side],
                calls.min(10),
            )?;
            ctx.synchronize()?;
        }
        warm_round += 1;
    }
    let mut raw = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let mut wall = [0.0; 2];
        let mut event = [0.0; 2];
        for slot in 0..2 {
            let side = (slot + round) % 2;
            let launch = artifacts[side].launch;
            ctx.synchronize()?;
            let started = Instant::now();
            enqueue(ctx, &kernels[side], launch, &args[side], calls)?;
            ctx.synchronize()?;
            wall[side] = started.elapsed().as_secs_f64() * 1e6 / calls as f64;
            event[side] = ctx.time_launches(
                &kernels[side],
                (launch.grid_x, launch.grid_y, 1),
                (launch.threads, 1, 1),
                launch.dyn_smem_bytes,
                &event_args[side],
                calls,
            )?;
        }
        raw.push(format!("{{\"round\":{round},\"first\":{},\"before_wall_us\":{},\"after_wall_us\":{},\"before_event_us\":{},\"after_event_us\":{}}}",
            quote(if round % 2 == 0 { "before" } else { "after" }), wall[0], wall[1], event[0], event[1]));
    }
    let final_errors = inputs.check(ctx)?;
    println!("{{\"mode\":\"compare\",\"shape\":[{},{},{}],\"device\":{},\"calls_per_batch\":{calls},\"rounds\":{rounds},\"warmup_seconds\":{},\"candidate\":{},\"before_ptx_sha256\":{},\"after_ptx_sha256\":{},\"reference_outputs\":{},\"total_outputs\":{},\"bitwise_equal\":true,\"initial_relative_l2\":[{},{}],\"final_relative_l2\":[{},{}],\"raw\":[{}]}}",
        shape.m, shape.n, shape.k, quote(ctx.device_name()), 3,
        quote(artifacts[0].manifest.get("candidate").unwrap()),
        quote(&hash(artifacts[0].ptx.as_bytes())), quote(&hash(artifacts[1].ptx.as_bytes())),
        inputs.references.len(), inputs.output_bytes / 4,
        initial_errors[0], initial_errors[1], final_errors[0], final_errors[1], raw.join(","));
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("");
    let offset = match mode {
        "export" if args.len() == 5 => 2,
        "compare" if (6..=8).contains(&args.len()) => 3,
        _ => return Err("usage: gemm_kernel_compare export OUTDIR M N K | compare BEFOREDIR AFTERDIR M N K [CALLS [ROUNDS]]".into()),
    };
    let shape = GemmShape {
        m: args[offset].parse().map_err(|_| "invalid M")?,
        n: args[offset + 1].parse().map_err(|_| "invalid N")?,
        k: args[offset + 2].parse().map_err(|_| "invalid K")?,
    };
    shape.validate()?;
    if std::env::var_os("Y_SMEM_PAD").is_some() || std::env::var_os("Y_FORCE_TILE").is_some() {
        return Err("benchmark requires unset Y_SMEM_PAD and Y_FORCE_TILE".into());
    }
    let calls = args
        .get(6)
        .map(|v| v.parse::<u32>())
        .transpose()
        .map_err(|_| "invalid CALLS")?
        .unwrap_or(100);
    let rounds = args
        .get(7)
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|_| "invalid ROUNDS")?
        .unwrap_or(15);
    if calls == 0 || calls > 10_000 || rounds == 0 || rounds > 1000 {
        return Err("CALLS must be 1..=10000 and ROUNDS 1..=1000".into());
    }
    let ctx = CudaContext::new().ok_or("CUDA unavailable")?;
    if mode == "export" {
        export(&ctx, Path::new(&args[1]), shape)
    } else {
        compare(
            &ctx,
            Path::new(&args[1]),
            Path::new(&args[2]),
            shape,
            calls,
            rounds,
        )
    }
}
