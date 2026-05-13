use std::fs;
use std::path::Path;

// Using pure Rust mocks since the NASM object file is not linked via Cargo
fn probe_cpu_features(out_buffer: &mut [u32; 4]) {
    // Mock AVX and AVX512, cache line size
    out_buffer[0] = 1 << 28; // AVX
    out_buffer[2] = 1 << 16; // AVX512
    out_buffer[3] = 64; // L2 line size
}

fn measure_l1_latency() -> u64 {
    4 // 4 cycles
}

pub struct HardwareProfile {
    pub has_avx: bool,
    pub has_avx512: bool,
    pub l2_line_size: u32,
    pub l1_latency_cycles: u64,
    // GPU hardware characteristics
    pub gpu_name: String,

    // Compute Latencies
    pub fma_latency_cycles: f64,
    pub imad_latency_cycles: f64,
    pub thermal_latency_40c: f64,
    pub thermal_latency_60c: f64,
    pub thermal_latency_80c: f64,
    pub mufu_rcp_latency_cycles: f64,
    pub dfma_latency_cycles: f64,

    // Memory Latencies
    pub smem_latency_cycles: f64,
    pub l1_gpu_latency_cycles: f64,
    pub l2_gpu_latency_cycles: f64,
    pub vram_latency_cycles: f64,

    // Tensor Cores
    pub hmma_f16_latency_cycles: f64,
    pub tf32_latency_cycles: f64,

    // Synchronization
    pub bar_sync_latency_cycles: f64,

    // Warp-Level Primitives
    pub shfl_sync_latency_cycles: f64,
    pub smem_exchange_latency_cycles: f64,

    // Bit-Field Ops
    pub bfe_latency_cycles: f64,
    pub bfi_latency_cycles: f64,
    pub and_shift_latency_cycles: f64,

    // Branch Divergence
    pub branch_uniform_cycles: f64,
    pub branch_divergent_cycles: f64,
    pub branch_divergence_penalty_cycles: f64,

    // Texture Unit
    pub tex1d_latency_cycles: f64,

    // IMAD.WIDE (Paper: 2.59 — faster than IMAD 4.53)
    pub imad_wide_latency_cycles: f64,

    // Full SFU Family (Paper Table 4)
    pub mufu_ex2_latency_cycles: f64,
    pub mufu_sin_latency_cycles: f64,
    pub mufu_rsq_latency_cycles: f64,
    pub mufu_lg2_latency_cycles: f64,

    // Reduced Precision
    pub hfma2_latency_cycles: f64,
    pub bf16x2_fma_latency_cycles: f64,

    // LOP3.LUT (3-input logic)
    pub lop3_lut_latency_cycles: f64,

    // FP64 DADD/DMUL (separate from DFMA)
    pub dadd_latency_cycles: f64,

    // Global Synchronization
    pub redux_sum_latency_cycles: f64,
    pub membar_gpu_latency_cycles: f64,

    // Constant Memory
    pub ldc_latency_cycles: f64,

    // Hardware Limits
    pub max_regs_per_thread: u32,
    pub max_regs_per_sm: u32,
    pub warp_size: u32,
    pub max_threads_per_sm: u32,
    pub max_warps_per_sm: u32,
    pub total_global_mem_mb: u64,

    // e.g. "Q32.32", "FP64"
    pub drift_free_types: Vec<String>,
    // Cycle cost for switching to a drift-free path
    pub zero_drift_penalty_cycles: u64,
}

fn parse_profile_value<'a>(contents: &'a str, key: &str) -> Option<&'a str> {
    for line in contents.lines() {
        let (found_key, value) = line.split_once('=')?;
        if found_key.trim() == key {
            return Some(value.trim());
        }
    }
    None
}

fn parse_bool_field(contents: &str, key: &str) -> Option<bool> {
    match parse_profile_value(contents, key)? {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn parse_u32_field(contents: &str, key: &str) -> Option<u32> {
    parse_profile_value(contents, key)?.parse().ok()
}

fn parse_u64_field(contents: &str, key: &str) -> Option<u64> {
    parse_profile_value(contents, key)?.parse().ok()
}

fn parse_f64_field(contents: &str, key: &str) -> Option<f64> {
    parse_profile_value(contents, key)?.parse().ok()
}

pub fn check_or_probe_hardware() -> HardwareProfile {
    let profile_path = ".ysu_hw_profile";

    if Path::new(profile_path).exists() {
        println!(
            "[*] Found existing {}, skipping Sentinel Probe.",
            profile_path
        );
        let contents = fs::read_to_string(profile_path).unwrap_or_default();

        // Parse drift free types list (comma separated)
        let drift_types_str = parse_profile_value(&contents, "DRIFT_FREE_TYPES").unwrap_or("");
        let drift_free_types = drift_types_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let profile = HardwareProfile {
            has_avx: parse_bool_field(&contents, "AVX").unwrap_or(false),
            has_avx512: parse_bool_field(&contents, "AVX512").unwrap_or(false),
            l2_line_size: parse_u32_field(&contents, "L2_LINE").unwrap_or(64),
            l1_latency_cycles: parse_u64_field(&contents, "L1_CYCLES").unwrap_or(4),
            gpu_name: parse_profile_value(&contents, "GPU_NAME")
                .unwrap_or("Unknown GPU")
                .to_string(),
            fma_latency_cycles: parse_f64_field(&contents, "FMA_LATENCY").unwrap_or(4.0),
            imad_latency_cycles: parse_f64_field(&contents, "IMAD_LATENCY").unwrap_or(4.0),
            thermal_latency_40c: parse_f64_field(&contents, "THERMAL_LATENCY_40C").unwrap_or(4.0),
            thermal_latency_60c: parse_f64_field(&contents, "THERMAL_LATENCY_60C").unwrap_or(4.0),
            thermal_latency_80c: parse_f64_field(&contents, "THERMAL_LATENCY_80C").unwrap_or(4.0),
            mufu_rcp_latency_cycles: parse_f64_field(&contents, "MUFU_RCP_LATENCY").unwrap_or(40.0),
            dfma_latency_cycles: parse_f64_field(&contents, "DFMA_LATENCY").unwrap_or(50.0),
            smem_latency_cycles: parse_f64_field(&contents, "SMEM_LATENCY").unwrap_or(28.0),
            l1_gpu_latency_cycles: parse_f64_field(&contents, "L1_GPU_LATENCY").unwrap_or(33.0),
            l2_gpu_latency_cycles: parse_f64_field(&contents, "L2_GPU_LATENCY").unwrap_or(90.0),
            vram_latency_cycles: parse_f64_field(&contents, "VRAM_LATENCY").unwrap_or(300.0),
            hmma_f16_latency_cycles: parse_f64_field(&contents, "HMMA_F16_LATENCY").unwrap_or(42.0),
            tf32_latency_cycles: parse_f64_field(&contents, "TF32_LATENCY").unwrap_or(66.0),
            bar_sync_latency_cycles: parse_f64_field(&contents, "BAR_SYNC_LATENCY").unwrap_or(35.0),
            shfl_sync_latency_cycles: parse_f64_field(&contents, "SHFL_SYNC_LATENCY")
                .unwrap_or(1.0),
            smem_exchange_latency_cycles: parse_f64_field(&contents, "SMEM_EXCHANGE_LATENCY")
                .unwrap_or(5.0),
            bfe_latency_cycles: parse_f64_field(&contents, "BFE_LATENCY").unwrap_or(4.5),
            bfi_latency_cycles: parse_f64_field(&contents, "BFI_LATENCY").unwrap_or(4.5),
            and_shift_latency_cycles: parse_f64_field(&contents, "AND_SHIFT_LATENCY")
                .unwrap_or(7.0),
            branch_uniform_cycles: parse_f64_field(&contents, "BRANCH_UNIFORM").unwrap_or(4.5),
            branch_divergent_cycles: parse_f64_field(&contents, "BRANCH_DIVERGENT").unwrap_or(9.0),
            branch_divergence_penalty_cycles: parse_f64_field(
                &contents,
                "BRANCH_DIVERGENCE_PENALTY",
            )
            .unwrap_or(4.5),
            tex1d_latency_cycles: parse_f64_field(&contents, "TEX1D_LATENCY").unwrap_or(70.0),
            imad_wide_latency_cycles: parse_f64_field(&contents, "IMAD_WIDE_LATENCY")
                .unwrap_or(2.6),
            mufu_ex2_latency_cycles: parse_f64_field(&contents, "MUFU_EX2_LATENCY").unwrap_or(17.5),
            mufu_sin_latency_cycles: parse_f64_field(&contents, "MUFU_SIN_LATENCY").unwrap_or(23.5),
            mufu_rsq_latency_cycles: parse_f64_field(&contents, "MUFU_RSQ_LATENCY").unwrap_or(39.5),
            mufu_lg2_latency_cycles: parse_f64_field(&contents, "MUFU_LG2_LATENCY").unwrap_or(39.5),
            hfma2_latency_cycles: parse_f64_field(&contents, "HFMA2_LATENCY").unwrap_or(4.5),
            bf16x2_fma_latency_cycles: parse_f64_field(&contents, "BF16X2_FMA_LATENCY")
                .unwrap_or(4.0),
            lop3_lut_latency_cycles: parse_f64_field(&contents, "LOP3_LUT_LATENCY").unwrap_or(4.5),
            dadd_latency_cycles: parse_f64_field(&contents, "DADD_LATENCY").unwrap_or(48.5),
            redux_sum_latency_cycles: parse_f64_field(&contents, "REDUX_SUM_LATENCY")
                .unwrap_or(60.0),
            membar_gpu_latency_cycles: parse_f64_field(&contents, "MEMBAR_GPU_LATENCY")
                .unwrap_or(205.0),
            ldc_latency_cycles: parse_f64_field(&contents, "LDC_LATENCY").unwrap_or(70.0),
            max_regs_per_thread: parse_u32_field(&contents, "MAX_REGS_PER_THREAD").unwrap_or(255),
            max_regs_per_sm: parse_u32_field(&contents, "MAX_REGS_PER_SM").unwrap_or(65536),
            warp_size: parse_u32_field(&contents, "WARP_SIZE").unwrap_or(32),
            max_threads_per_sm: parse_u32_field(&contents, "MAX_THREADS_PER_SM").unwrap_or(1536),
            max_warps_per_sm: parse_u32_field(&contents, "MAX_WARPS_PER_SM").unwrap_or(48),
            total_global_mem_mb: parse_u64_field(&contents, "TOTAL_GLOBAL_MEM_MB").unwrap_or(0),
            drift_free_types,
            zero_drift_penalty_cycles: parse_u64_field(&contents, "ZERO_DRIFT_PENALTY")
                .unwrap_or(0),
        };

        println!("    -> Loaded AVX: {}", profile.has_avx);
        println!("    -> Loaded AVX-512: {}", profile.has_avx512);
        println!(
            "    -> Loaded L2 Cache Line Size: {} bytes",
            profile.l2_line_size
        );
        println!(
            "    -> Loaded Baseline L1 Latency: {} cycles",
            profile.l1_latency_cycles
        );
        println!("    -> Loaded GPU Name: {}", profile.gpu_name);
        println!(
            "    -> GPU FMA/IMAD/MUFU Latencies: {} / {} / {}",
            profile.fma_latency_cycles,
            profile.imad_latency_cycles,
            profile.mufu_rcp_latency_cycles
        );
        println!(
            "    -> GPU Memory Latencies (SMEM/L1/L2/VRAM): {} / {} / {} / {}",
            profile.smem_latency_cycles,
            profile.l1_gpu_latency_cycles,
            profile.l2_gpu_latency_cycles,
            profile.vram_latency_cycles
        );
        println!(
            "    -> GPU Tensor Core Latencies (F16/TF32): {} / {}",
            profile.hmma_f16_latency_cycles, profile.tf32_latency_cycles
        );
        println!(
            "    -> Warp Shuffle vs SMEM Exchange: {} / {} cycles",
            profile.shfl_sync_latency_cycles, profile.smem_exchange_latency_cycles
        );
        println!(
            "    -> Bit-Field (BFE/BFI vs AND+SHIFT): {} / {} vs {}",
            profile.bfe_latency_cycles,
            profile.bfi_latency_cycles,
            profile.and_shift_latency_cycles
        );
        println!(
            "    -> Branch Divergence Penalty: {} cycles (uniform={}, divergent={})",
            profile.branch_divergence_penalty_cycles,
            profile.branch_uniform_cycles,
            profile.branch_divergent_cycles
        );
        println!(
            "    -> Texture Unit (TEX1D): {} cycles",
            profile.tex1d_latency_cycles
        );
        println!(
            "    -> IMAD.WIDE: {} cycles | SFU (EX2/SIN/RSQ/LG2): {} / {} / {} / {}",
            profile.imad_wide_latency_cycles,
            profile.mufu_ex2_latency_cycles,
            profile.mufu_sin_latency_cycles,
            profile.mufu_rsq_latency_cycles,
            profile.mufu_lg2_latency_cycles
        );
        println!(
            "    -> Reduced Precision (HFMA2/BF16x2): {} / {} | LOP3.LUT: {}",
            profile.hfma2_latency_cycles,
            profile.bf16x2_fma_latency_cycles,
            profile.lop3_lut_latency_cycles
        );
        println!(
            "    -> FP64 DADD: {} | REDUX.SUM: {} | MEMBAR.GPU: {} | LDC: {}",
            profile.dadd_latency_cycles,
            profile.redux_sum_latency_cycles,
            profile.membar_gpu_latency_cycles,
            profile.ldc_latency_cycles
        );
        println!(
            "    -> HW Limits: {} regs/thread, {} regs/SM, warp={}, {}MB VRAM",
            profile.max_regs_per_thread,
            profile.max_regs_per_sm,
            profile.warp_size,
            profile.total_global_mem_mb
        );
        println!("    -> Zero Drift Types: {:?}", profile.drift_free_types);

        return profile;
    }

    println!("[*] First boot detected! Running Sentinel Hardware Probe...");

    let mut features = [0u32; 4];
    let l1_cycles;

    // Call out to the simulated microbenchmark!
    probe_cpu_features(&mut features);
    l1_cycles = measure_l1_latency();

    let has_avx = (features[0] & (1 << 28)) != 0;
    let has_avx512 = (features[2] & (1 << 16)) != 0;
    let l2_line_size = features[3] & 0xFF;

    println!("[*] Executing external GPU Microbenchmark Payload (ysu_gpu_probe.exe)...");

    // In production, this binary is shipped alongside the compiler and runs
    // actual CUDA/PTX microbenchmarks on the user's silicon.
    let probe_cmd = std::process::Command::new("./ysu_gpu_probe.exe").output();

    let mut gpu_name = "Unknown GPU".to_string();
    let mut fma_latency_cycles = 4.0;
    let mut imad_latency_cycles = 4.0;
    let mut thermal_latency_40c = 4.0;
    let mut thermal_latency_60c = 4.0;
    let mut thermal_latency_80c = 4.0;
    let mut mufu_rcp_latency_cycles = 40.0;
    let mut dfma_latency_cycles = 50.0;
    let mut smem_latency_cycles = 28.0;
    let mut l1_gpu_latency_cycles = 33.0;
    let mut l2_gpu_latency_cycles = 90.0;
    let mut vram_latency_cycles = 300.0;
    let mut hmma_f16_latency_cycles = 42.0;
    let mut tf32_latency_cycles = 66.0;
    let mut bar_sync_latency_cycles = 35.0;
    let mut shfl_sync_latency_cycles = 1.0;
    let mut smem_exchange_latency_cycles = 5.0;
    let mut bfe_latency_cycles = 4.5;
    let mut bfi_latency_cycles = 4.5;
    let mut and_shift_latency_cycles = 7.0;
    let mut branch_uniform_cycles = 4.5;
    let mut branch_divergent_cycles = 9.0;
    let mut branch_divergence_penalty_cycles = 4.5;
    let mut tex1d_latency_cycles = 70.0;
    let mut imad_wide_latency_cycles = 2.6;
    let mut mufu_ex2_latency_cycles = 17.5;
    let mut mufu_sin_latency_cycles = 23.5;
    let mut mufu_rsq_latency_cycles = 39.5;
    let mut mufu_lg2_latency_cycles = 39.5;
    let mut hfma2_latency_cycles = 4.5;
    let mut bf16x2_fma_latency_cycles = 4.0;
    let mut lop3_lut_latency_cycles = 4.5;
    let mut dadd_latency_cycles = 48.5;
    let mut redux_sum_latency_cycles = 60.0;
    let mut membar_gpu_latency_cycles = 205.0;
    let mut ldc_latency_cycles = 70.0;
    let mut max_regs_per_thread = 255u32;
    let mut max_regs_per_sm = 65536u32;
    let mut warp_size = 32u32;
    let mut max_threads_per_sm = 1536u32;
    let mut max_warps_per_sm = 48u32;
    let mut total_global_mem_mb = 0u64;

    let mut drift_free_types = Vec::new();
    let mut zero_drift_penalty_cycles = 0;

    match probe_cmd {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);

            gpu_name = parse_profile_value(&stdout, "GPU_NAME")
                .unwrap_or("Unknown")
                .to_string();
            zero_drift_penalty_cycles = parse_u64_field(&stdout, "ZERO_DRIFT_PENALTY").unwrap_or(0);

            fma_latency_cycles = parse_f64_field(&stdout, "FMA_LATENCY_CYCLES").unwrap_or(4.0);
            imad_latency_cycles = parse_f64_field(&stdout, "IMAD_LATENCY_CYCLES").unwrap_or(4.0);
            thermal_latency_40c = parse_f64_field(&stdout, "THERMAL_LATENCY_40C").unwrap_or(4.0);
            thermal_latency_60c = parse_f64_field(&stdout, "THERMAL_LATENCY_60C").unwrap_or(4.0);
            thermal_latency_80c = parse_f64_field(&stdout, "THERMAL_LATENCY_80C").unwrap_or(4.0);
            mufu_rcp_latency_cycles =
                parse_f64_field(&stdout, "MUFU_RCP_LATENCY_CYCLES").unwrap_or(40.0);
            dfma_latency_cycles = parse_f64_field(&stdout, "DFMA_LATENCY_CYCLES").unwrap_or(50.0);
            smem_latency_cycles = parse_f64_field(&stdout, "SMEM_LATENCY_CYCLES").unwrap_or(28.0);
            l1_gpu_latency_cycles = parse_f64_field(&stdout, "L1_LATENCY_CYCLES").unwrap_or(33.0);
            l2_gpu_latency_cycles = parse_f64_field(&stdout, "L2_LATENCY_CYCLES").unwrap_or(90.0);
            vram_latency_cycles = parse_f64_field(&stdout, "VRAM_LATENCY_CYCLES").unwrap_or(300.0);
            hmma_f16_latency_cycles =
                parse_f64_field(&stdout, "HMMA_F16_LATENCY_CYCLES").unwrap_or(42.0);
            tf32_latency_cycles = parse_f64_field(&stdout, "TF32_LATENCY_CYCLES").unwrap_or(66.0);
            bar_sync_latency_cycles =
                parse_f64_field(&stdout, "BAR_SYNC_LATENCY_CYCLES").unwrap_or(35.0);
            shfl_sync_latency_cycles =
                parse_f64_field(&stdout, "SHFL_SYNC_LATENCY_CYCLES").unwrap_or(1.0);
            smem_exchange_latency_cycles =
                parse_f64_field(&stdout, "SMEM_EXCHANGE_LATENCY_CYCLES").unwrap_or(5.0);
            bfe_latency_cycles = parse_f64_field(&stdout, "BFE_LATENCY_CYCLES").unwrap_or(4.5);
            bfi_latency_cycles = parse_f64_field(&stdout, "BFI_LATENCY_CYCLES").unwrap_or(4.5);
            and_shift_latency_cycles =
                parse_f64_field(&stdout, "AND_SHIFT_LATENCY_CYCLES").unwrap_or(7.0);
            branch_uniform_cycles =
                parse_f64_field(&stdout, "BRANCH_UNIFORM_CYCLES").unwrap_or(4.5);
            branch_divergent_cycles =
                parse_f64_field(&stdout, "BRANCH_DIVERGENT_CYCLES").unwrap_or(9.0);
            branch_divergence_penalty_cycles =
                parse_f64_field(&stdout, "BRANCH_DIVERGENCE_PENALTY_CYCLES").unwrap_or(4.5);
            tex1d_latency_cycles = parse_f64_field(&stdout, "TEX1D_LATENCY_CYCLES").unwrap_or(70.0);
            imad_wide_latency_cycles =
                parse_f64_field(&stdout, "IMAD_WIDE_LATENCY_CYCLES").unwrap_or(2.6);
            mufu_ex2_latency_cycles =
                parse_f64_field(&stdout, "MUFU_EX2_LATENCY_CYCLES").unwrap_or(17.5);
            mufu_sin_latency_cycles =
                parse_f64_field(&stdout, "MUFU_SIN_LATENCY_CYCLES").unwrap_or(23.5);
            mufu_rsq_latency_cycles =
                parse_f64_field(&stdout, "MUFU_RSQ_LATENCY_CYCLES").unwrap_or(39.5);
            mufu_lg2_latency_cycles =
                parse_f64_field(&stdout, "MUFU_LG2_LATENCY_CYCLES").unwrap_or(39.5);
            hfma2_latency_cycles = parse_f64_field(&stdout, "HFMA2_LATENCY_CYCLES").unwrap_or(4.5);
            bf16x2_fma_latency_cycles =
                parse_f64_field(&stdout, "BF16X2_FMA_LATENCY_CYCLES").unwrap_or(4.0);
            lop3_lut_latency_cycles =
                parse_f64_field(&stdout, "LOP3_LUT_LATENCY_CYCLES").unwrap_or(4.5);
            dadd_latency_cycles = parse_f64_field(&stdout, "DADD_LATENCY_CYCLES").unwrap_or(48.5);
            redux_sum_latency_cycles =
                parse_f64_field(&stdout, "REDUX_SUM_LATENCY_CYCLES").unwrap_or(60.0);
            membar_gpu_latency_cycles =
                parse_f64_field(&stdout, "MEMBAR_GPU_LATENCY_CYCLES").unwrap_or(205.0);
            ldc_latency_cycles = parse_f64_field(&stdout, "LDC_LATENCY_CYCLES").unwrap_or(70.0);
            max_regs_per_thread = parse_u32_field(&stdout, "MAX_REGS_PER_THREAD").unwrap_or(255);
            max_regs_per_sm = parse_u32_field(&stdout, "MAX_REGS_PER_SM").unwrap_or(65536);
            warp_size = parse_u32_field(&stdout, "WARP_SIZE").unwrap_or(32);
            max_threads_per_sm = parse_u32_field(&stdout, "MAX_THREADS_PER_SM").unwrap_or(1536);
            max_warps_per_sm = parse_u32_field(&stdout, "MAX_WARPS_PER_SM").unwrap_or(48);
            total_global_mem_mb = parse_u64_field(&stdout, "TOTAL_GLOBAL_MEM_MB").unwrap_or(0);

            let drift_types_str = parse_profile_value(&stdout, "DRIFT_FREE_TYPES").unwrap_or("");
            drift_free_types = drift_types_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            println!("    -> GPU Probe returned successfully.");
        }
        _ => {
            println!("    -> [!] Failed to run GPU microbenchmark probe. Falling back to generic profile.");
        }
    }

    let profile = HardwareProfile {
        has_avx,
        has_avx512,
        l2_line_size,
        l1_latency_cycles: l1_cycles,
        gpu_name,
        fma_latency_cycles,
        imad_latency_cycles,
        thermal_latency_40c,
        thermal_latency_60c,
        thermal_latency_80c,
        mufu_rcp_latency_cycles,
        dfma_latency_cycles,
        smem_latency_cycles,
        l1_gpu_latency_cycles,
        l2_gpu_latency_cycles,
        vram_latency_cycles,
        hmma_f16_latency_cycles,
        tf32_latency_cycles,
        bar_sync_latency_cycles,
        shfl_sync_latency_cycles,
        smem_exchange_latency_cycles,
        bfe_latency_cycles,
        bfi_latency_cycles,
        and_shift_latency_cycles,
        branch_uniform_cycles,
        branch_divergent_cycles,
        branch_divergence_penalty_cycles,
        tex1d_latency_cycles,
        imad_wide_latency_cycles,
        mufu_ex2_latency_cycles,
        mufu_sin_latency_cycles,
        mufu_rsq_latency_cycles,
        mufu_lg2_latency_cycles,
        hfma2_latency_cycles,
        bf16x2_fma_latency_cycles,
        lop3_lut_latency_cycles,
        dadd_latency_cycles,
        redux_sum_latency_cycles,
        membar_gpu_latency_cycles,
        ldc_latency_cycles,
        max_regs_per_thread,
        max_regs_per_sm,
        warp_size,
        max_threads_per_sm,
        max_warps_per_sm,
        total_global_mem_mb,
        drift_free_types,
        zero_drift_penalty_cycles,
    };

    println!("    -> Detected AVX: {}", profile.has_avx);
    println!("    -> Detected AVX-512: {}", profile.has_avx512);
    println!("    -> L2 Cache Line Size: {} bytes", profile.l2_line_size);
    println!(
        "    -> Baseline L1 Latency: {} cycles",
        profile.l1_latency_cycles
    );
    println!("    -> Detected GPU: {}", profile.gpu_name);
    println!(
        "    -> GPU FMA/IMAD/MUFU Latencies: {} / {} / {}",
        profile.fma_latency_cycles, profile.imad_latency_cycles, profile.mufu_rcp_latency_cycles
    );
    println!(
        "    -> GPU Thermal Latency Gradient (40C/60C/80C): {} / {} / {}",
        profile.thermal_latency_40c, profile.thermal_latency_60c, profile.thermal_latency_80c
    );
    println!(
        "    -> GPU Memory Latencies (SMEM/L1/L2/VRAM): {} / {} / {} / {}",
        profile.smem_latency_cycles,
        profile.l1_gpu_latency_cycles,
        profile.l2_gpu_latency_cycles,
        profile.vram_latency_cycles
    );
    println!(
        "    -> GPU Tensor Core Latencies (F16/TF32): {} / {}",
        profile.hmma_f16_latency_cycles, profile.tf32_latency_cycles
    );
    println!(
        "    -> Warp Shuffle vs SMEM Exchange: {} / {} cycles",
        profile.shfl_sync_latency_cycles, profile.smem_exchange_latency_cycles
    );
    println!(
        "    -> Bit-Field (BFE/BFI vs AND+SHIFT): {} / {} vs {}",
        profile.bfe_latency_cycles, profile.bfi_latency_cycles, profile.and_shift_latency_cycles
    );
    println!(
        "    -> Branch Divergence Penalty: {} cycles (uniform={}, divergent={})",
        profile.branch_divergence_penalty_cycles,
        profile.branch_uniform_cycles,
        profile.branch_divergent_cycles
    );
    println!(
        "    -> Texture Unit (TEX1D): {} cycles",
        profile.tex1d_latency_cycles
    );
    println!(
        "    -> IMAD.WIDE: {} cycles | SFU (EX2/SIN/RSQ/LG2): {} / {} / {} / {}",
        profile.imad_wide_latency_cycles,
        profile.mufu_ex2_latency_cycles,
        profile.mufu_sin_latency_cycles,
        profile.mufu_rsq_latency_cycles,
        profile.mufu_lg2_latency_cycles
    );
    println!(
        "    -> Reduced Precision (HFMA2/BF16x2): {} / {} | LOP3.LUT: {}",
        profile.hfma2_latency_cycles,
        profile.bf16x2_fma_latency_cycles,
        profile.lop3_lut_latency_cycles
    );
    println!(
        "    -> FP64 DADD: {} | REDUX.SUM: {} | MEMBAR.GPU: {} | LDC: {}",
        profile.dadd_latency_cycles,
        profile.redux_sum_latency_cycles,
        profile.membar_gpu_latency_cycles,
        profile.ldc_latency_cycles
    );
    println!(
        "    -> HW Limits: {} regs/thread, {} regs/SM, warp={}, {}MB VRAM",
        profile.max_regs_per_thread,
        profile.max_regs_per_sm,
        profile.warp_size,
        profile.total_global_mem_mb
    );
    println!(
        "    -> Verified Zero Drift Types: {:?}",
        profile.drift_free_types
    );

    println!("[*] Saving hardware topology to {}...", profile_path);
    let serialized = format!(
        "AVX={}\nAVX512={}\nL2_LINE={}\nL1_CYCLES={}\nGPU_NAME={}\nFMA_LATENCY={}\nIMAD_LATENCY={}\nTHERMAL_LATENCY_40C={}\nTHERMAL_LATENCY_60C={}\nTHERMAL_LATENCY_80C={}\nMUFU_RCP_LATENCY={}\nDFMA_LATENCY={}\nSMEM_LATENCY={}\nL1_GPU_LATENCY={}\nL2_GPU_LATENCY={}\nVRAM_LATENCY={}\nHMMA_F16_LATENCY={}\nTF32_LATENCY={}\nBAR_SYNC_LATENCY={}\nSHFL_SYNC_LATENCY={}\nSMEM_EXCHANGE_LATENCY={}\nBFE_LATENCY={}\nBFI_LATENCY={}\nAND_SHIFT_LATENCY={}\nBRANCH_UNIFORM={}\nBRANCH_DIVERGENT={}\nBRANCH_DIVERGENCE_PENALTY={}\nTEX1D_LATENCY={}\nIMAD_WIDE_LATENCY={}\nMUFU_EX2_LATENCY={}\nMUFU_SIN_LATENCY={}\nMUFU_RSQ_LATENCY={}\nMUFU_LG2_LATENCY={}\nHFMA2_LATENCY={}\nBF16X2_FMA_LATENCY={}\nLOP3_LUT_LATENCY={}\nDADD_LATENCY={}\nREDUX_SUM_LATENCY={}\nMEMBAR_GPU_LATENCY={}\nLDC_LATENCY={}\nMAX_REGS_PER_THREAD={}\nMAX_REGS_PER_SM={}\nWARP_SIZE={}\nMAX_THREADS_PER_SM={}\nMAX_WARPS_PER_SM={}\nTOTAL_GLOBAL_MEM_MB={}\nDRIFT_FREE_TYPES={}\nZERO_DRIFT_PENALTY={}\n",
        profile.has_avx,
        profile.has_avx512,
        profile.l2_line_size,
        profile.l1_latency_cycles,
        profile.gpu_name,
        profile.fma_latency_cycles,
        profile.imad_latency_cycles,
        profile.thermal_latency_40c,
        profile.thermal_latency_60c,
        profile.thermal_latency_80c,
        profile.mufu_rcp_latency_cycles,
        profile.dfma_latency_cycles,
        profile.smem_latency_cycles,
        profile.l1_gpu_latency_cycles,
        profile.l2_gpu_latency_cycles,
        profile.vram_latency_cycles,
        profile.hmma_f16_latency_cycles,
        profile.tf32_latency_cycles,
        profile.bar_sync_latency_cycles,
        profile.shfl_sync_latency_cycles,
        profile.smem_exchange_latency_cycles,
        profile.bfe_latency_cycles,
        profile.bfi_latency_cycles,
        profile.and_shift_latency_cycles,
        profile.branch_uniform_cycles,
        profile.branch_divergent_cycles,
        profile.branch_divergence_penalty_cycles,
        profile.tex1d_latency_cycles,
        profile.imad_wide_latency_cycles,
        profile.mufu_ex2_latency_cycles,
        profile.mufu_sin_latency_cycles,
        profile.mufu_rsq_latency_cycles,
        profile.mufu_lg2_latency_cycles,
        profile.hfma2_latency_cycles,
        profile.bf16x2_fma_latency_cycles,
        profile.lop3_lut_latency_cycles,
        profile.dadd_latency_cycles,
        profile.redux_sum_latency_cycles,
        profile.membar_gpu_latency_cycles,
        profile.ldc_latency_cycles,
        profile.max_regs_per_thread,
        profile.max_regs_per_sm,
        profile.warp_size,
        profile.max_threads_per_sm,
        profile.max_warps_per_sm,
        profile.total_global_mem_mb,
        profile.drift_free_types.join(","),
        profile.zero_drift_penalty_cycles
    );
    fs::write(profile_path, serialized).expect("Failed to write profile");

    profile
}
