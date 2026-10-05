//! Run with: cargo run --release --example adaptive_gemm
use std::time::Instant;
use y::adaptive_jit::{AdaptiveGemm, AdaptiveJitConfig, GemmShape};
use y::cuda_runtime::{f16_bits_to_f32, random_f16_bits, CudaContext};

fn main() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let mut args = Vec::new();
    let mut cache_dir = None;
    while let Some(arg) = arguments.next() {
        match arg.as_str() {
            "--help" => {
                println!("Usage: cargo run --release --example adaptive_gemm -- [--cache-dir PATH] [M N K]");
                return Ok(());
            }
            "--cache-dir" => {
                if cache_dir.is_some() {
                    return Err("--cache-dir can only be specified once".into());
                }
                cache_dir = Some(std::path::PathBuf::from(
                    arguments.next().ok_or("--cache-dir requires a path")?,
                ));
            }
            value if value.starts_with("--") => return Err(format!("unknown option {value}")),
            _ => args.push(arg),
        }
    }
    let shape = match args.as_slice() {
        [] => GemmShape {
            m: 256,
            n: 256,
            k: 256,
        },
        [m, n, k] => GemmShape {
            m: m.parse().map_err(|_| "M must be a positive integer")?,
            n: n.parse().map_err(|_| "N must be a positive integer")?,
            k: k.parse().map_err(|_| "K must be a positive integer")?,
        },
        _ => return Err("expected either no dimensions or M N K (see --help)".into()),
    };
    shape.validate()?;
    let ctx = CudaContext::new().ok_or("an NVIDIA GPU and CUDA driver are required")?;
    let a: Vec<u16> = (0..shape.m as u64 * shape.k as u64)
        .map(|i| random_f16_bits(i, 123))
        .collect();
    let b: Vec<u16> = (0..shape.k as u64 * shape.n as u64)
        .map(|i| random_f16_bits(i, 987))
        .collect();
    let a_device = ctx.alloc(a.len() * 2)?;
    let b_device = ctx.alloc(b.len() * 2)?;
    let c_device = ctx.alloc(shape.m as usize * shape.n as usize * 4)?;
    let bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>();
    ctx.memcpy_htod_at(&a_device, 0, &bytes(&a))?;
    ctx.memcpy_htod_at(&b_device, 0, &bytes(&b))?;
    let setup_start = Instant::now();
    let mut jit = AdaptiveGemm::new(
        &ctx,
        AdaptiveJitConfig {
            hot_threshold: 2, // Deliberately short to demonstrate the transition.
            cache_dir,
            ..Default::default()
        },
    )?;
    println!(
        "{}: {}x{}x{} GEMM",
        ctx.device_name(),
        shape.m,
        shape.n,
        shape.k
    );
    println!(
        "runtime setup: {:.3} ms",
        setup_start.elapsed().as_secs_f64() * 1000.0
    );
    let prepare_start = Instant::now();
    jit.prepare(shape)?;
    println!(
        "warmup preparation: {:.3} ms; launches={}; cache_hit={}",
        prepare_start.elapsed().as_secs_f64() * 1000.0,
        jit.stats(shape).unwrap().launches,
        jit.stats(shape).unwrap().cache_hit
    );
    for call in 1..=4 {
        if call == 4 {
            let tuning_start = Instant::now();
            let reports = jit.tune_hot(1)?;
            println!(
                "explicit maintenance: {:.3} ms; {} shape(s) processed",
                tuning_start.elapsed().as_secs_f64() * 1000.0,
                reports.len()
            );
        }
        let start = Instant::now();
        // These correctly sized, disjoint allocations belong to ctx and
        // remain alive until every launch has synchronized.
        unsafe {
            jit.launch(
                shape,
                a_device.device_ptr(),
                b_device.device_ptr(),
                c_device.device_ptr(),
            )?;
        }
        ctx.synchronize()?;
        println!(
            "launch {call}: {:.3} ms host elapsed; {:?}; {} pending shape(s)",
            start.elapsed().as_secs_f64() * 1000.0,
            jit.stats(shape).unwrap().tier,
            jit.pending_shapes().len()
        );
    }
    let mut output = vec![0u8; c_device.len_bytes()];
    ctx.memcpy_dtoh_at(&mut output, &c_device, 0)?;
    let mut squared_error = 0.0f64;
    let mut squared_ref = 0.0f64;
    for sample in 0..64usize {
        let row = (sample * 37) % shape.m as usize;
        let col = (sample * 71) % shape.n as usize;
        let reference: f64 = (0..shape.k as usize)
            .map(|k| {
                f16_bits_to_f32(a[row * shape.k as usize + k]) as f64
                    * f16_bits_to_f32(b[k * shape.n as usize + col]) as f64
            })
            .sum();
        let offset = (row * shape.n as usize + col) * 4;
        let actual = f32::from_ne_bytes(output[offset..offset + 4].try_into().unwrap()) as f64;
        squared_error += (reference - actual).powi(2);
        squared_ref += reference.powi(2);
    }
    let error = (squared_error / squared_ref.max(f64::MIN_POSITIVE)).sqrt();
    if !error.is_finite() || error > 0.002 {
        return Err(format!(
            "CPU reference comparison failed: relative L2 error {error}"
        ));
    }
    let stats = jit.stats(shape).unwrap();
    println!("64 CPU-reference samples: relative L2 error {error:.3e}");
    println!(
        "Tuning this run: {:.3} s; {} distinct kernels measured; baseline {:?} us; selected {:?} us",
        stats.tuning_time.as_secs_f64(),
        stats.candidates_measured,
        stats.baseline_us,
        stats.selected_us
    );
    if stats.cache_hit {
        println!("Restored timings are historical; no empirical tuning ran in this process.");
    }
    if let Some(error) = &stats.cache_error {
        eprintln!("Persistent cache unavailable for this shape: {error}");
    }
    if let Some(error) = &stats.tuning_error {
        return Err(format!("baseline worked, but tuning failed: {error}"));
    }
    if let Some(calls) = stats.estimated_break_even_launches() {
        println!("Estimated extra calls to repay tuning: {calls}");
    }
    Ok(())
}
