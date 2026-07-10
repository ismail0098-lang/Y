use std::fs;
use std::path::Path;

macro_rules! log_info {
    ($($arg:tt)*) => {
        println!("\x1b[1;36m[*]\x1b[0m {}", format_args!($($arg)*));
    };
}

#[cfg(target_arch = "x86_64")]
unsafe fn get_tsc() -> u64 {
    let mut low: u32;
    let mut high: u32;
    std::arch::asm!(
        "lfence",
        "rdtsc",
        out("eax") low,
        out("edx") high,
        options(nostack, nomem)
    );
    ((high as u64) << 32) | (low as u64)
}

fn probe_cpu_features(out_buffer: &mut [u32; 4]) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{__cpuid, __cpuid_count};
        
        // 1. Standard CPUID (EAX=1)
        let res1 = __cpuid(1);
        out_buffer[0] = res1.ecx;
        out_buffer[1] = res1.edx;
        
        // 2. Extended CPUID (EAX=7, ECX=0)
        let res7 = __cpuid_count(7, 0);
        out_buffer[2] = res7.ebx;
        
        // 3. Cache Line Size (EAX=0x80000006)
        let res_cache = __cpuid(0x80000006);
        out_buffer[3] = res_cache.ecx;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Mock fallback for non-x86_64
        out_buffer[0] = 1 << 28; // AVX
        out_buffer[2] = 1 << 16; // AVX512
        out_buffer[3] = 64; // L2 line size
    }
}

fn measure_cache_latency(size_bytes: usize) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let size_elts = size_bytes / 8;
        if size_elts < 128 { return 4; }
        
        // Cache line size is 64 bytes (8 elements of usize)
        let line_stride = 8;
        let num_lines = size_elts / line_stride;
        if num_lines < 2 { return 4; }
        
        let mut array = vec![0usize; size_elts];
        
        // Generate a single-cycle permutation of line indices using Satolo's algorithm
        let mut lines: Vec<usize> = (0..num_lines).collect();
        let mut rng = 0x243F6A8885A308D3u64; // Seed with fractional part of Pi
        for i in 0..num_lines - 1 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let range = num_lines - 1 - i;
            let j = i + 1 + (rng as usize % range);
            lines.swap(i, j);
        }
        
        // Link the lines together in the array
        for i in 0..num_lines {
            let next_line = lines[i];
            array[i * line_stride] = next_line * line_stride;
        }
        
        let mut ptr = 0;
        // Warm up and verify cycle
        for _ in 0..num_lines {
            ptr = array[ptr];
        }
        
        let start = unsafe { get_tsc() };
        for _ in 0..20000 {
            ptr = array[ptr];
            ptr = array[ptr];
        }
        let end = unsafe { get_tsc() };
        
        let cycles = end.saturating_sub(start);
        let avg = cycles / 40000;
        if avg == 0 { 4 } else { avg }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        if size_bytes <= 32 * 1024 { 4 }
        else if size_bytes <= 256 * 1024 { 12 }
        else if size_bytes <= 8 * 1024 * 1024 { 40 }
        else { 120 }
    }
}

#[allow(dead_code)]
fn measure_avx2_throughput(has_avx2: bool) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if !has_avx2 {
            return 0.0;
        }
        let iterations = 1000000;
        let start = unsafe { get_tsc() };
        unsafe {
            std::arch::asm!(
                "2:",
                "vpaddd ymm0, ymm0, ymm1",
                "vpaddd ymm2, ymm2, ymm1",
                "vpaddd ymm3, ymm3, ymm1",
                "vpaddd ymm4, ymm4, ymm1",
                "vpaddd ymm5, ymm5, ymm1",
                "vpaddd ymm6, ymm6, ymm1",
                "vpaddd ymm7, ymm7, ymm1",
                "vpaddd ymm8, ymm8, ymm1",
                "vpaddd ymm9, ymm9, ymm1",
                "vpaddd ymm10, ymm10, ymm11",
                "dec {0:e}",
                "jnz 2b",
                inout(reg) iterations => _,
                out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
                out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
                out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
                options(nostack)
            );
        }
        let end = unsafe { get_tsc() };
        let cycles = end.saturating_sub(start);
        let total_ops = 10_000_000.0;
        cycles as f64 / total_ops
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0.5
    }
}

fn measure_avx512_throughput(has_avx512: bool) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if !has_avx512 {
            return 0.0;
        }
        let iterations = 1000000;
        let start = unsafe { get_tsc() };
        unsafe {
            std::arch::asm!(
                "2:",
                "vpaddd zmm0, zmm0, zmm1",
                "vpaddd zmm2, zmm2, zmm1",
                "vpaddd zmm3, zmm3, zmm1",
                "vpaddd zmm4, zmm4, zmm1",
                "vpaddd zmm5, zmm5, zmm1",
                "vpaddd zmm6, zmm6, zmm1",
                "vpaddd zmm7, zmm7, zmm1",
                "vpaddd zmm8, zmm8, zmm1",
                "vpaddd zmm9, zmm9, zmm1",
                "vpaddd zmm10, zmm10, zmm11",
                "dec {0:e}",
                "jnz 2b",
                inout(reg) iterations => _,
                out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
                out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
                out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
                options(nostack)
            );
        }
        let end = unsafe { get_tsc() };
        let cycles = end.saturating_sub(start);
        let total_ops = 10_000_000.0;
        cycles as f64 / total_ops
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0.5
    }
}

fn measure_thread_scheduling_cost() -> u64 {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    
    let flag1 = Arc::new(AtomicU32::new(0));
    let flag2 = Arc::new(AtomicU32::new(0));
    
    let f1_clone = flag1.clone();
    let f2_clone = flag2.clone();
    
    let handle = std::thread::spawn(move || {
        for i in 1..1000 {
            while f1_clone.load(Ordering::Acquire) != i {
                std::thread::yield_now();
            }
            f2_clone.store(i, Ordering::Release);
        }
    });
    
    let start = unsafe { get_tsc() };
    for i in 1..1000 {
        flag1.store(i, Ordering::Release);
        while flag2.load(Ordering::Acquire) != i {
            std::thread::yield_now();
        }
    }
    let end = unsafe { get_tsc() };
    
    let _ = handle.join();
    
    let cycles = end.saturating_sub(start);
    cycles / 999
}

#[derive(Debug, Clone, Default)]
pub struct HardwareProfile {
    pub has_avx: bool,
    pub has_avx512: bool,
    pub l2_line_size: u32,
    pub l1_latency_cycles: u64,
    pub l2_latency_cycles: u64,
    pub l3_latency_cycles: u64,
    pub mem_latency_cycles: u64,
    pub avx512_throughput_cycles: f64,
    pub thread_scheduling_cost_cycles: u64,
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

    // §16 SMEM Bank Conflict Family
    pub smem_noconflict_cycles: f64,
    pub smem_2way_conflict_cycles: f64,
    pub smem_4way_conflict_cycles: f64,
    pub smem_broadcast_cycles: f64,
    pub smem_2way_conflict_penalty: f64,
    pub smem_4way_conflict_penalty: f64,
    pub smem_padding_needed: bool,

    // §17 Type Conversion Latencies
    pub f2i_latency_cycles: f64,
    pub i2f_latency_cycles: f64,
    pub f2h_latency_cycles: f64,
    pub h2f_latency_cycles: f64,

    // §18 DP4A INT8 Dot Product
    pub dp4a_latency_cycles: f64,

    // §19 Bit Manipulation
    pub popc_latency_cycles: f64,
    pub clz_latency_cycles: f64,
    pub prmt_latency_cycles: f64,

    // §20 Warp Vote Primitives
    pub ballot_sync_latency_cycles: f64,
    pub vote_any_latency_cycles: f64,

    // §21 Read-Only Cache (__ldg)
    pub ldg_nc_latency_cycles: f64,

    // §22 Global Atomics
    pub atom_add_f32_latency_cycles: f64,
    pub atom_add_i32_latency_cycles: f64,

    // §23 Strided Global Access
    pub stride1_cycles: f64,
    pub stride2_cycles: f64,
    pub stride4_cycles: f64,
    pub stride8_cycles: f64,
    pub stride16_cycles: f64,
    pub stride32_cycles: f64,

    // §24 CP.ASYNC Global→Shared
    pub cp_async_latency_cycles: f64,

    // §25 FMA ILP Throughput
    pub fma_ilp_throughput: f64,     // FMAs per cycle (>1 = dual-issue capable)
    pub fma_ilp_cycles_per_op: f64,
}

fn parse_profile_value<'a>(contents: &'a str, key: &str) -> Option<&'a str> {
    for line in contents.lines() {
        if let Some((found_key, value)) = line.split_once('=') {
            if found_key.trim() == key {
                return Some(value.trim());
            }
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
            l2_latency_cycles: parse_u64_field(&contents, "L2_CYCLES").unwrap_or(12),
            l3_latency_cycles: parse_u64_field(&contents, "L3_CYCLES").unwrap_or(40),
            mem_latency_cycles: parse_u64_field(&contents, "MEM_CYCLES").unwrap_or(120),
            avx512_throughput_cycles: parse_f64_field(&contents, "AVX512_THROUGHPUT").unwrap_or(0.5),
            thread_scheduling_cost_cycles: parse_u64_field(&contents, "THREAD_SCHEDULING_COST").unwrap_or(2000),
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
            smem_noconflict_cycles: parse_f64_field(&contents, "SMEM_NOCONFLICT_CYCLES").unwrap_or(4.0),
            smem_2way_conflict_cycles: parse_f64_field(&contents, "SMEM_2WAY_CONFLICT_CYCLES").unwrap_or(8.0),
            smem_4way_conflict_cycles: parse_f64_field(&contents, "SMEM_4WAY_CONFLICT_CYCLES").unwrap_or(16.0),
            smem_broadcast_cycles: parse_f64_field(&contents, "SMEM_BROADCAST_CYCLES").unwrap_or(4.0),
            smem_2way_conflict_penalty: parse_f64_field(&contents, "SMEM_2WAY_CONFLICT_PENALTY").unwrap_or(4.0),
            smem_4way_conflict_penalty: parse_f64_field(&contents, "SMEM_4WAY_CONFLICT_PENALTY").unwrap_or(12.0),
            smem_padding_needed: parse_u32_field(&contents, "SMEM_PADDING_NEEDED").unwrap_or(0) != 0,
            f2i_latency_cycles: parse_f64_field(&contents, "F2I_LATENCY_CYCLES").unwrap_or(4.5),
            i2f_latency_cycles: parse_f64_field(&contents, "I2F_LATENCY_CYCLES").unwrap_or(4.5),
            f2h_latency_cycles: parse_f64_field(&contents, "F2H_LATENCY_CYCLES").unwrap_or(8.0),
            h2f_latency_cycles: parse_f64_field(&contents, "H2F_LATENCY_CYCLES").unwrap_or(8.0),
            dp4a_latency_cycles: parse_f64_field(&contents, "DP4A_LATENCY_CYCLES").unwrap_or(2.0),
            popc_latency_cycles: parse_f64_field(&contents, "POPC_LATENCY_CYCLES").unwrap_or(4.5),
            clz_latency_cycles: parse_f64_field(&contents, "CLZ_LATENCY_CYCLES").unwrap_or(4.5),
            prmt_latency_cycles: parse_f64_field(&contents, "PRMT_LATENCY_CYCLES").unwrap_or(4.5),
            ballot_sync_latency_cycles: parse_f64_field(&contents, "BALLOT_SYNC_LATENCY_CYCLES").unwrap_or(4.5),
            vote_any_latency_cycles: parse_f64_field(&contents, "VOTE_ANY_LATENCY_CYCLES").unwrap_or(4.5),
            ldg_nc_latency_cycles: parse_f64_field(&contents, "LDG_NC_LATENCY_CYCLES").unwrap_or(125.0),
            atom_add_f32_latency_cycles: parse_f64_field(&contents, "ATOM_ADD_F32_LATENCY_CYCLES").unwrap_or(400.0),
            atom_add_i32_latency_cycles: parse_f64_field(&contents, "ATOM_ADD_I32_LATENCY_CYCLES").unwrap_or(400.0),
            stride1_cycles: parse_f64_field(&contents, "STRIDE1_CYCLES").unwrap_or(28.0),
            stride2_cycles: parse_f64_field(&contents, "STRIDE2_CYCLES").unwrap_or(40.0),
            stride4_cycles: parse_f64_field(&contents, "STRIDE4_CYCLES").unwrap_or(60.0),
            stride8_cycles: parse_f64_field(&contents, "STRIDE8_CYCLES").unwrap_or(90.0),
            stride16_cycles: parse_f64_field(&contents, "STRIDE16_CYCLES").unwrap_or(120.0),
            stride32_cycles: parse_f64_field(&contents, "STRIDE32_CYCLES").unwrap_or(125.0),
            cp_async_latency_cycles: parse_f64_field(&contents, "CP_ASYNC_LATENCY_CYCLES").unwrap_or(200.0),
            fma_ilp_throughput: parse_f64_field(&contents, "FMA_ILP_THROUGHPUT").unwrap_or(1.0),
            fma_ilp_cycles_per_op: parse_f64_field(&contents, "FMA_ILP_CYCLES_PER_OP").unwrap_or(4.5),
        };

        println!("    -> Loaded AVX: {}", profile.has_avx);
        println!("    -> Loaded AVX-512: {}", profile.has_avx512);
        println!(
            "    -> Loaded L2 Cache Line Size: {} bytes",
            profile.l2_line_size
        );
        println!(
            "    -> Loaded CPU Memory Latency Sweep (L1/L2/L3/Mem): {} / {} / {} / {} cycles",
            profile.l1_latency_cycles, profile.l2_latency_cycles, profile.l3_latency_cycles, profile.mem_latency_cycles
        );
        println!("    -> Loaded CPU AVX-512 Instruction Throughput: {:.2} cycles per op", profile.avx512_throughput_cycles);
        println!("    -> Loaded CPU Thread Scheduling/Context Switch Handoff Cost: {} cycles", profile.thread_scheduling_cost_cycles);
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

    log_info!("First boot detected! Running Sentinel Hardware Probe...");

    let mut features = [0u32; 4];

    probe_cpu_features(&mut features);
    let has_avx = (features[0] & (1 << 28)) != 0;
    let has_avx512 = (features[2] & (1 << 16)) != 0;
    let l2_line_size = features[3] & 0xFF;

    println!("      -> CPU Features: AVX={}, AVX-512={}, L2 Cache Line Size={}", has_avx, has_avx512, l2_line_size);
    
    log_info!("Running Analytical CPU Memory Bus Latency Sweep...");
    let l1_latency_cycles = measure_cache_latency(16 * 1024);
    let l2_latency_cycles = measure_cache_latency(256 * 1024);
    let l3_latency_cycles = measure_cache_latency(4 * 1024 * 1024);
    let mem_latency_cycles = measure_cache_latency(64 * 1024 * 1024);

    log_info!("Profiling CPU Instruction Throughput...");
    let avx512_throughput_cycles = measure_avx512_throughput(has_avx512);

    log_info!("Measuring CPU Thread Scheduling/Context Switch Handoff Cost...");
    let thread_scheduling_cost_cycles = measure_thread_scheduling_cost();

    // Try to find the probe executable. First try current working directory, 
    // then try the folder of the running compiler binary itself.
    let mut probe_path = std::path::PathBuf::from("./ysu_gpu_probe");
    if cfg!(target_os = "windows") {
        probe_path.set_extension("exe");
    }
    
    if !probe_path.exists() {
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let local_probe = exe_dir.join(if cfg!(target_os = "windows") { "ysu_gpu_probe.exe" } else { "ysu_gpu_probe" });
                if local_probe.exists() {
                    probe_path = local_probe;
                }
            }
        }
    }
    
    log_info!("Executing external GPU Microbenchmark Payload ({:?})...", probe_path);
    let probe_cmd = std::process::Command::new(&probe_path).output();

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

    let mut stdout_content = String::new();

    match probe_cmd {
        Ok(output) if output.status.success() => {
            stdout_content = String::from_utf8_lossy(&output.stdout).to_string();
            let stdout = &stdout_content;

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
        l1_latency_cycles,
        l2_latency_cycles,
        l3_latency_cycles,
        mem_latency_cycles,
        avx512_throughput_cycles,
        thread_scheduling_cost_cycles,
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
        smem_noconflict_cycles: parse_f64_field(&stdout_content, "SMEM_NOCONFLICT_CYCLES").unwrap_or(4.0),
        smem_2way_conflict_cycles: parse_f64_field(&stdout_content, "SMEM_2WAY_CONFLICT_CYCLES").unwrap_or(8.0),
        smem_4way_conflict_cycles: parse_f64_field(&stdout_content, "SMEM_4WAY_CONFLICT_CYCLES").unwrap_or(16.0),
        smem_broadcast_cycles: parse_f64_field(&stdout_content, "SMEM_BROADCAST_CYCLES").unwrap_or(4.0),
        smem_2way_conflict_penalty: parse_f64_field(&stdout_content, "SMEM_2WAY_CONFLICT_PENALTY").unwrap_or(4.0),
        smem_4way_conflict_penalty: parse_f64_field(&stdout_content, "SMEM_4WAY_CONFLICT_PENALTY").unwrap_or(12.0),
        smem_padding_needed: parse_u32_field(&stdout_content, "SMEM_PADDING_NEEDED").unwrap_or(0) != 0,
        f2i_latency_cycles: parse_f64_field(&stdout_content, "F2I_LATENCY_CYCLES").unwrap_or(4.5),
        i2f_latency_cycles: parse_f64_field(&stdout_content, "I2F_LATENCY_CYCLES").unwrap_or(4.5),
        f2h_latency_cycles: parse_f64_field(&stdout_content, "F2H_LATENCY_CYCLES").unwrap_or(8.0),
        h2f_latency_cycles: parse_f64_field(&stdout_content, "H2F_LATENCY_CYCLES").unwrap_or(8.0),
        dp4a_latency_cycles: parse_f64_field(&stdout_content, "DP4A_LATENCY_CYCLES").unwrap_or(2.0),
        popc_latency_cycles: parse_f64_field(&stdout_content, "POPC_LATENCY_CYCLES").unwrap_or(4.5),
        clz_latency_cycles: parse_f64_field(&stdout_content, "CLZ_LATENCY_CYCLES").unwrap_or(4.5),
        prmt_latency_cycles: parse_f64_field(&stdout_content, "PRMT_LATENCY_CYCLES").unwrap_or(4.5),
        ballot_sync_latency_cycles: parse_f64_field(&stdout_content, "BALLOT_SYNC_LATENCY_CYCLES").unwrap_or(4.5),
        vote_any_latency_cycles: parse_f64_field(&stdout_content, "VOTE_ANY_LATENCY_CYCLES").unwrap_or(4.5),
        ldg_nc_latency_cycles: parse_f64_field(&stdout_content, "LDG_NC_LATENCY_CYCLES").unwrap_or(125.0),
        atom_add_f32_latency_cycles: parse_f64_field(&stdout_content, "ATOM_ADD_F32_LATENCY_CYCLES").unwrap_or(400.0),
        atom_add_i32_latency_cycles: parse_f64_field(&stdout_content, "ATOM_ADD_I32_LATENCY_CYCLES").unwrap_or(400.0),
        stride1_cycles: parse_f64_field(&stdout_content, "STRIDE1_CYCLES").unwrap_or(28.0),
        stride2_cycles: parse_f64_field(&stdout_content, "STRIDE2_CYCLES").unwrap_or(40.0),
        stride4_cycles: parse_f64_field(&stdout_content, "STRIDE4_CYCLES").unwrap_or(60.0),
        stride8_cycles: parse_f64_field(&stdout_content, "STRIDE8_CYCLES").unwrap_or(90.0),
        stride16_cycles: parse_f64_field(&stdout_content, "STRIDE16_CYCLES").unwrap_or(120.0),
        stride32_cycles: parse_f64_field(&stdout_content, "STRIDE32_CYCLES").unwrap_or(125.0),
        cp_async_latency_cycles: parse_f64_field(&stdout_content, "CP_ASYNC_LATENCY_CYCLES").unwrap_or(200.0),
        fma_ilp_throughput: parse_f64_field(&stdout_content, "FMA_ILP_THROUGHPUT").unwrap_or(1.0),
        fma_ilp_cycles_per_op: parse_f64_field(&stdout_content, "FMA_ILP_CYCLES_PER_OP").unwrap_or(4.5),
    };

    println!("    -> Detected AVX: {}", profile.has_avx);
    println!("    -> Detected AVX-512: {}", profile.has_avx512);
    println!("    -> L2 Cache Line Size: {} bytes", profile.l2_line_size);
    println!(
        "    -> CPU Memory Latency Sweep (L1/L2/L3/Mem): {} / {} / {} / {} cycles",
        profile.l1_latency_cycles, profile.l2_latency_cycles, profile.l3_latency_cycles, profile.mem_latency_cycles
    );
    println!("    -> CPU AVX-512 Instruction Throughput: {:.2} cycles per op", profile.avx512_throughput_cycles);
    println!("    -> CPU Thread Scheduling/Context Switch Handoff Cost: {} cycles", profile.thread_scheduling_cost_cycles);
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
        "AVX={}\nAVX512={}\nL2_LINE={}\nL1_CYCLES={}\nL2_CYCLES={}\nL3_CYCLES={}\nMEM_CYCLES={}\nAVX512_THROUGHPUT={}\nTHREAD_SCHEDULING_COST={}\nGPU_NAME={}\n\
         FMA_LATENCY={}\nIMAD_LATENCY={}\nTHERMAL_LATENCY_40C={}\n\
         THERMAL_LATENCY_60C={}\nTHERMAL_LATENCY_80C={}\n\
         MUFU_RCP_LATENCY={}\nDFMA_LATENCY={}\nSMEM_LATENCY={}\n\
         L1_GPU_LATENCY={}\nL2_GPU_LATENCY={}\nVRAM_LATENCY={}\n\
         HMMA_F16_LATENCY={}\nTF32_LATENCY={}\nBAR_SYNC_LATENCY={}\n\
         SHFL_SYNC_LATENCY={}\nSMEM_EXCHANGE_LATENCY={}\n\
         BFE_LATENCY={}\nBFI_LATENCY={}\nAND_SHIFT_LATENCY={}\n\
         BRANCH_UNIFORM={}\nBRANCH_DIVERGENT={}\nBRANCH_DIVERGENCE_PENALTY={}\n\
         TEX1D_LATENCY={}\nIMAD_WIDE_LATENCY={}\n\
         MUFU_EX2_LATENCY={}\nMUFU_SIN_LATENCY={}\nMUFU_RSQ_LATENCY={}\nMUFU_LG2_LATENCY={}\n\
         HFMA2_LATENCY={}\nBF16X2_FMA_LATENCY={}\nLOP3_LUT_LATENCY={}\n\
         DADD_LATENCY={}\nREDUX_SUM_LATENCY={}\nMEMBAR_GPU_LATENCY={}\nLDC_LATENCY={}\n\
         MAX_REGS_PER_THREAD={}\nMAX_REGS_PER_SM={}\nWARP_SIZE={}\n\
         MAX_THREADS_PER_SM={}\nMAX_WARPS_PER_SM={}\nTOTAL_GLOBAL_MEM_MB={}\n\
         DRIFT_FREE_TYPES={}\nZERO_DRIFT_PENALTY={}\n\
         SMEM_NOCONFLICT_CYCLES={}\nSMEM_2WAY_CONFLICT_CYCLES={}\n\
         SMEM_4WAY_CONFLICT_CYCLES={}\nSMEM_BROADCAST_CYCLES={}\n\
         SMEM_2WAY_CONFLICT_PENALTY={}\nSMEM_4WAY_CONFLICT_PENALTY={}\n\
         SMEM_PADDING_NEEDED={}\n\
         F2I_LATENCY_CYCLES={}\nI2F_LATENCY_CYCLES={}\n\
         F2H_LATENCY_CYCLES={}\nH2F_LATENCY_CYCLES={}\n\
         DP4A_LATENCY_CYCLES={}\n\
         POPC_LATENCY_CYCLES={}\nCLZ_LATENCY_CYCLES={}\nPRMT_LATENCY_CYCLES={}\n\
         BALLOT_SYNC_LATENCY_CYCLES={}\nVOTE_ANY_LATENCY_CYCLES={}\n\
         LDG_NC_LATENCY_CYCLES={}\n\
         ATOM_ADD_F32_LATENCY_CYCLES={}\nATOM_ADD_I32_LATENCY_CYCLES={}\n\
         STRIDE1_CYCLES={}\nSTRIDE2_CYCLES={}\nSTRIDE4_CYCLES={}\n\
         STRIDE8_CYCLES={}\nSTRIDE16_CYCLES={}\nSTRIDE32_CYCLES={}\n\
         CP_ASYNC_LATENCY_CYCLES={}\n\
         FMA_ILP_THROUGHPUT={}\nFMA_ILP_CYCLES_PER_OP={}\n",
        profile.has_avx, profile.has_avx512, profile.l2_line_size, profile.l1_latency_cycles,
        profile.l2_latency_cycles, profile.l3_latency_cycles, profile.mem_latency_cycles,
        profile.avx512_throughput_cycles, profile.thread_scheduling_cost_cycles,
        profile.gpu_name, profile.fma_latency_cycles, profile.imad_latency_cycles,
        profile.thermal_latency_40c, profile.thermal_latency_60c, profile.thermal_latency_80c,
        profile.mufu_rcp_latency_cycles, profile.dfma_latency_cycles, profile.smem_latency_cycles,
        profile.l1_gpu_latency_cycles, profile.l2_gpu_latency_cycles, profile.vram_latency_cycles,
        profile.hmma_f16_latency_cycles, profile.tf32_latency_cycles, profile.bar_sync_latency_cycles,
        profile.shfl_sync_latency_cycles, profile.smem_exchange_latency_cycles,
        profile.bfe_latency_cycles, profile.bfi_latency_cycles, profile.and_shift_latency_cycles,
        profile.branch_uniform_cycles, profile.branch_divergent_cycles, profile.branch_divergence_penalty_cycles,
        profile.tex1d_latency_cycles, profile.imad_wide_latency_cycles,
        profile.mufu_ex2_latency_cycles, profile.mufu_sin_latency_cycles,
        profile.mufu_rsq_latency_cycles, profile.mufu_lg2_latency_cycles,
        profile.hfma2_latency_cycles, profile.bf16x2_fma_latency_cycles, profile.lop3_lut_latency_cycles,
        profile.dadd_latency_cycles, profile.redux_sum_latency_cycles,
        profile.membar_gpu_latency_cycles, profile.ldc_latency_cycles,
        profile.max_regs_per_thread, profile.max_regs_per_sm, profile.warp_size,
        profile.max_threads_per_sm, profile.max_warps_per_sm, profile.total_global_mem_mb,
        profile.drift_free_types.join(","), profile.zero_drift_penalty_cycles,
        profile.smem_noconflict_cycles, profile.smem_2way_conflict_cycles,
        profile.smem_4way_conflict_cycles, profile.smem_broadcast_cycles,
        profile.smem_2way_conflict_penalty, profile.smem_4way_conflict_penalty,
        profile.smem_padding_needed as u8,
        profile.f2i_latency_cycles, profile.i2f_latency_cycles,
        profile.f2h_latency_cycles, profile.h2f_latency_cycles,
        profile.dp4a_latency_cycles,
        profile.popc_latency_cycles, profile.clz_latency_cycles, profile.prmt_latency_cycles,
        profile.ballot_sync_latency_cycles, profile.vote_any_latency_cycles,
        profile.ldg_nc_latency_cycles,
        profile.atom_add_f32_latency_cycles, profile.atom_add_i32_latency_cycles,
        profile.stride1_cycles, profile.stride2_cycles, profile.stride4_cycles,
        profile.stride8_cycles, profile.stride16_cycles, profile.stride32_cycles,
        profile.cp_async_latency_cycles,
        profile.fma_ilp_throughput, profile.fma_ilp_cycles_per_op
    );
    fs::write(profile_path, serialized).expect("Failed to write profile");

    profile
}
