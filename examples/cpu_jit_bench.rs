//! Process-isolated worker for tools/benchmark_cpu_jit.py.
//! Invoke via the Python runner to validate outputs and preserve metadata.
use std::hint::black_box;
use std::time::Instant;
use y::cpu_jit::{CpuJit, JitOptions, JitValue};

const SOURCE: &str = include_str!("../benchmarks/cpu_jit/kernels.ysu");
const ORIGINAL_SOURCE: &str = include_str!("../benchmarks/cpu_jit/kernels_original.ysu");
const RUNTIME_SOURCE: &str = include_str!("../benchmarks/cpu_jit/kernels_runtime.ysu");
const COPIES_SOURCE: &str = include_str!("../benchmarks/cpu_jit/kernels_copies.ysu");
const HELPERS_SOURCE: &str = include_str!("../benchmarks/cpu_jit/kernels_helpers.ysu");
const UNSIGNED_SEED: u64 = 0xfedc_ba98_7654_3210;

struct ExtraFns {
    unsigned: unsafe extern "C" fn(i64, u64) -> u64,
    dot: unsafe extern "C" fn(*const f64, *const f64, i64, i64) -> f64,
    logical: unsafe extern "C" fn(*mut i64, i64, i64) -> i64,
    search: unsafe extern "C" fn(*const i64, i64, i64) -> i64,
}

impl ExtraFns {
    fn new(jit: &CpuJit) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            Ok(Self {
                unsigned: std::mem::transmute(black_box(jit.function_address("unsigned_mix")?)),
                dot: std::mem::transmute(black_box(jit.function_address("float_dot")?)),
                logical: std::mem::transmute(black_box(jit.function_address("short_circuit")?)),
                search: std::mem::transmute(black_box(jit.function_address("binary_search")?)),
            })
        }
    }
}

struct RuntimeFns {
    string: unsafe extern "C" fn(i64, i64) -> i64,
    vector: unsafe extern "C" fn(i64, i64) -> i64,
}
impl RuntimeFns {
    fn new(jit: &CpuJit) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            Ok(Self {
                string: std::mem::transmute(black_box(jit.function_address("string_scan")?)),
                vector: std::mem::transmute(black_box(jit.function_address("vec_scan_append")?)),
            })
        }
    }
    fn new_helpers(jit: &CpuJit) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            Ok(Self {
                string: std::mem::transmute(black_box(jit.function_address("string_scan_helper")?)),
                vector: std::mem::transmute(black_box(
                    jit.function_address("vec_scan_append_helper")?,
                )),
            })
        }
    }
}

struct CopyFns {
    byte: unsafe extern "C" fn(i64, i64, i32) -> i64,
    word: unsafe extern "C" fn(i64, i64, i32) -> i64,
    bulk: unsafe extern "C" fn(i64, i64) -> i64,
}
impl CopyFns {
    fn new(jit: &CpuJit) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            Ok(Self {
                byte: std::mem::transmute(black_box(jit.function_address("vec_dynamic_byte")?)),
                word: std::mem::transmute(black_box(jit.function_address("vec_dynamic_i64")?)),
                bulk: std::mem::transmute(black_box(jit.function_address("string_bulk_append")?)),
            })
        }
    }
}

fn checked_measurement(
    jit: &CpuJit,
    n: i64,
    calls: usize,
    warmup: usize,
    name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let function: unsafe extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(jit.function_address("integer_branch")?) };
    for i in 0..warmup {
        let seed = 123 + i as i64 * 17;
        black_box(unsafe { function(n, seed) });
        black_box(unsafe { jit.call("integer_branch", &[JitValue::I64(n), JitValue::I64(seed)])? });
    }
    let mut native = vec![0_i64; calls];
    let mut checked = vec![0_i64; calls];
    let start = Instant::now();
    for (i, output) in native.iter_mut().enumerate() {
        *output = unsafe { function(black_box(n), black_box(123 + i as i64 * 17)) };
    }
    let native_ns_per_call = nanos(start) / calls as f64;
    let start = Instant::now();
    for (i, output) in checked.iter_mut().enumerate() {
        let result = unsafe {
            jit.call(
                black_box("integer_branch"),
                &[
                    JitValue::I64(black_box(n)),
                    JitValue::I64(black_box(123 + i as i64 * 17)),
                ],
            )?
        };
        *output = match result {
            JitValue::I64(value) => value,
            _ => return Err("checked call returned unexpected type".into()),
        };
    }
    let checked_ns_per_call = nanos(start) / calls as f64;
    if native != checked {
        return Err("checked calls differ from native calls".into());
    }
    let native_hash = memory_hash(&native);
    let checked_hash = memory_hash(&checked);
    let first = &native[..calls.min(8)];
    let last = &native[calls.saturating_sub(8)..];
    Ok(format!("{{\"name\":{name:?},\"n\":{n},\"calls\":{calls},\"warmup\":{warmup},\"native_ns_per_call\":{native_ns_per_call},\"checked_ns_per_call\":{checked_ns_per_call},\"native_outputs_hash\":\"{native_hash}\",\"checked_outputs_hash\":\"{checked_hash}\",\"outputs_first\":{first:?},\"outputs_last\":{last:?}}}"))
}

fn reset(data: &mut [i64]) {
    for (i, value) in data.iter_mut().enumerate() {
        *value = ((i * 13 + 5) & 1023) as i64;
    }
}

fn memory_hash(data: &[i64]) -> u64 {
    data.iter().fold(14695981039346656037_u64, |hash, &value| {
        (hash ^ value as u64).wrapping_mul(1099511628211)
    })
}

fn nanos(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e9
}

fn compilation_settings(
    options: JitOptions,
    instrumented: bool,
    outer_unroll_annotations: usize,
) -> String {
    let ir_opt_level = if instrumented {
        options.training_opt_level.unwrap_or(options.opt_level)
    } else {
        options.opt_level
    };
    let codegen_opt_level = options.codegen_opt_level.unwrap_or(options.opt_level);
    // true denotes inherited LLVM defaults, rather than a forced unroll.
    let ir_loop_unrolling = instrumented || options.final_loop_unrolling;
    let outer_unroll_policy_active =
        !instrumented && options.final_loop_unrolling && !options.final_unroll_outer_loops;
    let optional = |level: Option<u8>| {
        level
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".into())
    };
    format!("{{\"ir_opt_level\":{ir_opt_level},\"ir_target_opt_level\":{ir_opt_level},\"codegen_opt_level\":{codegen_opt_level},\"codegen_override\":{},\"training_override\":{},\"verify_each_pass\":{},\"profile_edge_counters\":{},\"profile_loop_edge_counters\":{},\"final_loop_unrolling\":{},\"ir_loop_unrolling\":{ir_loop_unrolling},\"final_unroll_outer_loops\":{},\"outer_unroll_policy_active\":{outer_unroll_policy_active},\"outer_unroll_annotations\":{outer_unroll_annotations},\"verification_boundary_policy\":\"input-and-pass-boundaries\"}}",
        optional(options.codegen_opt_level), optional(options.training_opt_level), options.verify_each_pass, options.profile_edge_counters, options.profile_loop_edge_counters, options.final_loop_unrolling, options.final_unroll_outer_loops)
}

// Training uses the same runtime seed distribution as the timed batch. All
// training results and side effects are retained for independent verification.
fn collect_training(
    jit: &CpuJit,
    count: usize,
    dims: [i64; 13],
    expanded: bool,
    runtime: bool,
    copies: bool,
    helpers: bool,
) -> Result<(String, f64), Box<dyn std::error::Error>> {
    let integer: unsafe extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(jit.function_address("integer_branch")?) };
    let fib: unsafe extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(jit.function_address("recursive_fib")?) };
    let float: unsafe extern "C" fn(i64, f64) -> f64 =
        unsafe { std::mem::transmute(jit.function_address("float_recurrence")?) };
    let memory: unsafe extern "C" fn(*mut i64, i64, i64) -> i64 =
        unsafe { std::mem::transmute(jit.function_address("indexed_memory")?) };
    let extra = if expanded {
        Some(ExtraFns::new(jit)?)
    } else {
        None
    };
    let runtime_fns = if runtime {
        Some(RuntimeFns::new(jit)?)
    } else {
        None
    };
    let copy_fns = if copies {
        Some(CopyFns::new(jit)?)
    } else {
        None
    };
    let helper_fns = if helpers {
        Some(RuntimeFns::new_helpers(jit)?)
    } else {
        None
    };
    let mut data = vec![0_i64; 65536];
    reset(&mut data);
    let extra_len = if expanded { 65536 } else { 0 };
    let a: Vec<f64> = (0..extra_len)
        .map(|i| ((i * 17 + 3) & 1023) as f64 / 1024.0)
        .collect();
    let b: Vec<f64> = (0..extra_len)
        .map(|i| ((i * 29 + 7) & 1023) as f64 / 2048.0)
        .collect();
    let sorted: Vec<i64> = (0..extra_len).map(|i| i * 3 + 1).collect();
    let mut counters = [0_i64; 3];
    let mut integers = vec![0_i64; count];
    let mut fibs = vec![0_i64; count];
    let mut floats = vec![0_f64; count];
    let mut memories = vec![0_i64; count];
    let mut unsigneds = vec![0_u64; count];
    let mut dots = vec![0_f64; count];
    let mut logicals = vec![0_i64; count];
    let mut searches = vec![0_i64; count];
    let mut strings = vec![0_i64; count];
    let mut vectors = vec![0_i64; count];
    let mut copy_bytes = vec![0_i64; count];
    let mut copy_words = vec![0_i64; count];
    let mut copy_bulk = vec![0_i64; count];
    let mut helper_strings = vec![0_i64; count];
    let mut helper_vectors = vec![0_i64; count];
    let started = Instant::now();
    for i in 0..count {
        let seed = 123 + i as i64 * 17;
        unsafe {
            integers[i] = integer(dims[0], seed);
            fibs[i] = fib(dims[1] + i as i64 % 2);
            floats[i] = float(dims[2], 0.5 + i as f64 * 0.0001);
            memories[i] = memory(data.as_mut_ptr(), dims[3], seed);
            if let Some(extra) = &extra {
                unsigneds[i] = (extra.unsigned)(dims[4], UNSIGNED_SEED.wrapping_add(i as u64 * 17));
                dots[i] = (extra.dot)(a.as_ptr(), b.as_ptr(), dims[5], seed);
                logicals[i] = (extra.logical)(counters.as_mut_ptr(), dims[6], seed);
                searches[i] = (extra.search)(sorted.as_ptr(), dims[7], seed);
            }
            if let Some(runtime) = &runtime_fns {
                strings[i] = (runtime.string)(dims[8], seed);
                vectors[i] = (runtime.vector)(dims[9], seed);
            }
            if let Some(f) = &copy_fns {
                copy_bytes[i] = (f.byte)(dims[10], seed, 1);
                copy_words[i] = (f.word)(dims[11], seed, 8);
                copy_bulk[i] = (f.bulk)(dims[12], seed);
            }
            if let Some(f) = &helper_fns {
                helper_strings[i] = (f.string)(dims[8], seed);
                helper_vectors[i] = (f.vector)(dims[9], seed);
            }
        }
    }
    let collection_ns = nanos(started);
    let hash = memory_hash(&data);
    let mut results=format!("{{\"name\":\"integer_branch\",\"outputs\":{integers:?}}},\
                            {{\"name\":\"recursive_fib\",\"outputs\":{fibs:?}}},\
                            {{\"name\":\"float_recurrence\",\"outputs\":{floats:?}}},\
                            {{\"name\":\"indexed_memory\",\"outputs\":{memories:?},\"memory_hash\":\"{hash}\"}}");
    if expanded {
        let unsigneds: Vec<String> = unsigneds.iter().map(u64::to_string).collect();
        results.push_str(&format!(",{{\"name\":\"unsigned_mix\",\"outputs\":{unsigneds:?}}},\
                                   {{\"name\":\"float_dot\",\"outputs\":{dots:?}}},\
                                   {{\"name\":\"short_circuit\",\"outputs\":{logicals:?},\"counters\":{counters:?}}},\
                                   {{\"name\":\"binary_search\",\"outputs\":{searches:?}}}"));
    }
    if runtime {
        results.push_str(&format!(",{{\"name\":\"string_scan\",\"outputs\":{strings:?}}},{{\"name\":\"vec_scan_append\",\"outputs\":{vectors:?}}}"));
    }
    if copies {
        results.push_str(&format!(",{{\"name\":\"vec_dynamic_byte\",\"outputs\":{copy_bytes:?}}},{{\"name\":\"vec_dynamic_i64\",\"outputs\":{copy_words:?}}},{{\"name\":\"string_bulk_append\",\"outputs\":{copy_bulk:?}}}"));
    }
    if helpers {
        results.push_str(&format!(",{{\"name\":\"string_scan_helper\",\"outputs\":{helper_strings:?}}},{{\"name\":\"vec_scan_append_helper\",\"outputs\":{helper_vectors:?}}}"));
    }
    Ok((results, collection_ns))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if ![12, 14, 17].contains(&args.len()) {
        return Err(
            "usage: cpu_jit_bench CALLS WARMUP INTEGER_N FIB_N FLOAT_N MEMORY_N UNSIGNED_N DOT_N LOGICAL_N SEARCH_N [STRING_N VEC_N [DYNAMIC_BYTE_N DYNAMIC_I64_N BULK_N]] COLD_ONLY SUITE".into(),
        );
    }
    let calls: usize = args[0].parse()?;
    let warmup: usize = args[1].parse()?;
    let integer_n: i64 = args[2].parse()?;
    let fib_n: i64 = args[3].parse()?;
    let float_n: i64 = args[4].parse()?;
    let memory_n: i64 = args[5].parse()?;
    let unsigned_n: i64 = args[6].parse()?;
    let dot_n: i64 = args[7].parse()?;
    let logical_n: i64 = args[8].parse()?;
    let search_n: i64 = args[9].parse()?;
    let tail = args.len() - 2;
    let string_n: i64 = if tail >= 12 { args[10].parse()? } else { 16384 };
    let vec_n: i64 = if tail >= 12 { args[11].parse()? } else { 16384 };
    let dynamic_byte_n: i64 = if tail == 15 { args[12].parse()? } else { 16384 };
    let dynamic_i64_n: i64 = if tail == 15 { args[13].parse()? } else { 16384 };
    let bulk_n: i64 = if tail == 15 { args[14].parse()? } else { 512 };
    let cold_only = args[tail] == "1";
    let suite = args[tail + 1].as_str();
    let helpers = args[tail + 1] == "helpers";
    let copies = helpers || args[tail + 1] == "copies";
    let runtime = copies || args[tail + 1] == "runtime";
    let expanded = match args[tail + 1].as_str() {
        "expanded" | "runtime" | "copies" | "helpers" => true,
        "original" => false,
        _ => return Err("suite must be original, expanded, runtime, copies or helpers".into()),
    };
    if calls == 0
        || integer_n < 1
        || !(2..=40).contains(&fib_n)
        || float_n < 1
        || memory_n < 1
        || unsigned_n < 1
        || dot_n < 1
        || logical_n < 1
        || search_n < 1
        || string_n < 1
        || vec_n < 1
        || dynamic_byte_n < 1
        || dynamic_i64_n < 1
        || bulk_n < 1
    {
        return Err("invalid benchmark dimensions".into());
    }

    let runtime_source = if helpers {
        format!("{SOURCE}\n{RUNTIME_SOURCE}\n{COPIES_SOURCE}\n{HELPERS_SOURCE}")
    } else if copies {
        format!("{SOURCE}\n{RUNTIME_SOURCE}\n{COPIES_SOURCE}")
    } else if runtime {
        format!("{SOURCE}\n{RUNTIME_SOURCE}")
    } else {
        String::new()
    };
    let source = if runtime {
        runtime_source.as_str()
    } else if expanded {
        SOURCE
    } else {
        ORIGINAL_SOURCE
    };
    let mode = std::env::var("Y_CPU_JIT_BENCH_MODE").unwrap_or_else(|_| "optimized".into());
    let stage = std::env::var("Y_CPU_JIT_BENCH_STAGE").unwrap_or_else(|_| "rotate-profile".into());
    if ![
        "rotate-profile",
        "runtime",
        "runtime-append",
        "runtime-copies",
        "adapters",
        "helper-effects",
        "verify-each",
        "codegen",
        "training-tier",
        "profile-edges",
        "profile-loop-edges",
        "final-unroll",
        "outer-unroll",
    ]
    .contains(&stage.as_str())
    {
        return Err("invalid optimization stage".into());
    }
    if [
        "helper-effects",
        "verify-each",
        "codegen",
        "training-tier",
        "profile-edges",
        "profile-loop-edges",
        "final-unroll",
        "outer-unroll",
    ]
    .contains(&stage.as_str())
        && !helpers
    {
        return Err("this stage requires helpers suite".into());
    }
    if stage == "runtime-copies" && !copies {
        return Err("runtime-copies requires copies suite".into());
    }
    if ["runtime-append", "adapters"].contains(&stage.as_str()) && !runtime {
        return Err("runtime stage requires runtime or copies suite".into());
    }
    if [
        "runtime-append",
        "runtime-copies",
        "adapters",
        "helper-effects",
        "verify-each",
        "codegen",
        "training-tier",
        "profile-edges",
        "profile-loop-edges",
        "final-unroll",
        "outer-unroll",
    ]
    .contains(&stage.as_str())
        && mode == "baseline"
    {
        return Err("this stage requires previous or optimized mode".into());
    }
    let modern = [
        "runtime-copies",
        "adapters",
        "helper-effects",
        "verify-each",
        "codegen",
        "training-tier",
        "profile-edges",
        "profile-loop-edges",
        "final-unroll",
        "outer-unroll",
    ]
    .contains(&stage.as_str());
    let optimize_runtime =
        modern || stage == "runtime-append" || (stage == "runtime" && mode == "optimized");
    let optimize_runtime_mutations = modern || (stage == "runtime-append" && mode == "optimized");
    let optimize_runtime_copies = stage == "outer-unroll"
        || stage == "final-unroll"
        || stage == "profile-loop-edges"
        || stage == "profile-edges"
        || stage == "training-tier"
        || stage == "codegen"
        || stage == "verify-each"
        || stage == "helper-effects"
        || stage == "adapters"
        || (stage == "runtime-copies" && mode == "optimized");
    let optimize_call_adapters = stage == "outer-unroll"
        || stage == "final-unroll"
        || stage == "profile-loop-edges"
        || stage == "profile-edges"
        || stage == "training-tier"
        || stage == "codegen"
        || stage == "verify-each"
        || stage == "helper-effects"
        || stage == "runtime-copies"
        || (stage == "adapters" && mode == "optimized");
    let optimize_helper_effects = stage == "outer-unroll"
        || stage == "final-unroll"
        || stage == "profile-loop-edges"
        || stage == "profile-edges"
        || stage == "training-tier"
        || stage == "codegen"
        || stage == "verify-each"
        || (stage == "helper-effects" && mode == "optimized");
    let verify_each_pass = stage != "verify-each" || mode == "previous";
    let codegen_override = (stage == "codegen" && mode == "optimized").then_some(2);
    let codegen_opt_level = codegen_override.unwrap_or(3);
    let codegen_override_json = codegen_override
        .map(|level| level.to_string())
        .unwrap_or_else(|| "null".into());
    let training_override = (stage == "outer-unroll"
        || stage == "final-unroll"
        || stage == "profile-loop-edges"
        || stage == "profile-edges"
        || (stage == "training-tier" && mode == "optimized"))
        .then_some(1);
    let profile_edge_counters = stage == "profile-edges" && mode == "optimized";
    let profile_loop_edge_counters = stage == "profile-loop-edges" && mode == "optimized";
    let final_loop_unrolling = stage != "final-unroll" || mode != "optimized";
    let ir_loop_unrolling = final_loop_unrolling;
    let final_unroll_outer_loops = stage != "outer-unroll" || mode != "optimized";
    let outer_unroll_policy_active = final_loop_unrolling && !final_unroll_outer_loops;
    let training_opt_level = training_override.unwrap_or(3);
    let training_override_json = training_override
        .map(|level| level.to_string())
        .unwrap_or_else(|| "null".into());
    let recognize_rotates = mode != "baseline";
    let profile_loop_controls = !modern && stage != "runtime-append" && !optimize_runtime;
    let training_count: usize = std::env::var("Y_CPU_JIT_BENCH_PROFILE_WARMUP")
        .unwrap_or_else(|_| "12".into())
        .parse()?;
    let options = JitOptions {
        opt_level: 3,
        codegen_opt_level: codegen_override,
        training_opt_level: training_override,
        profile_edge_counters,
        profile_loop_edge_counters,
        final_loop_unrolling,
        final_unroll_outer_loops,
        recognize_rotates,
        optimize_runtime,
        optimize_runtime_mutations,
        optimize_runtime_copies,
        optimize_call_adapters,
        optimize_helper_effects,
        profile_loop_controls,
        verify_each_pass,
    };
    let prepare_started = Instant::now();
    let mut preparation_json = String::new();
    let mut compilations = Vec::new();
    let mut instrumented_ir = None;
    let jit = match mode.as_str() {
        "baseline" => CpuJit::compile_with_options(source, options)?,
        "optimized" | "previous" => {
            if training_count == 0 {
                return Err("profile training count must be positive".into());
            }
            let trainer = CpuJit::compile_instrumented(source, options)?;
            let instrumented_compile_ns = trainer.compile_duration().as_secs_f64() * 1e9;
            compilations.push(format!(
                "{{\"kind\":\"instrumented\",\"compilation_settings\":{},\"timings\":{},\"optimization_timings\":{},\"materialization_timings\":{},\"verification_checks\":{}}}",
                compilation_settings(options, true, trainer.outer_unroll_annotations()),
                trainer.compile_timings().to_json(),
                trainer.optimization_timings().to_json(),
                trainer.materialization_timings().to_json(),
                trainer.compile_timings().verification_checks,
            ));
            if let Some(path) = std::env::var_os("Y_CPU_JIT_BENCH_INSTRUMENTED_IR") {
                instrumented_ir = Some((path, trainer.optimized_ir().to_owned()));
            }
            let (training_results, profile_collection_ns) = collect_training(
                &trainer,
                training_count,
                [
                    integer_n,
                    fib_n,
                    float_n,
                    memory_n,
                    unsigned_n,
                    dot_n,
                    logical_n,
                    search_n,
                    string_n,
                    vec_n,
                    dynamic_byte_n,
                    dynamic_i64_n,
                    bulk_n,
                ],
                expanded,
                runtime,
                copies,
                helpers,
            )?;
            let snapshot_started = Instant::now();
            let profile = trainer.branch_profile()?;
            let profile_snapshot_ns = nanos(snapshot_started);
            let observations = profile.total_observations();
            if observations == 0 {
                return Err("profile collection recorded no branches".into());
            }
            let fingerprint: String = profile
                .fingerprint()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            let sites: Vec<String> = profile
                .sites()
                .iter()
                .map(|site| {
                    format!(
                        "{{\"function\":{:?},\"block\":{:?},\"true_count\":{},\"false_count\":{}}}",
                        site.function, site.block, site.true_count, site.false_count
                    )
                })
                .collect();
            let started = Instant::now();
            let compiled = CpuJit::compile_with_profile(source, options, &profile)?;
            let optimized_recompile_ns = nanos(started);
            preparation_json=format!(",\"instrumented_compile_ns\":{instrumented_compile_ns},\
                \"profile_collection_ns\":{profile_collection_ns},\"profile_snapshot_ns\":{profile_snapshot_ns},\
                \"optimized_recompile_ns\":{optimized_recompile_ns},\"profile_training_calls\":{training_count},\
                \"profile_fingerprint\":\"{fingerprint}\",\"profile_observations\":{observations},\
                \"profile_sites\":[{}],\"profiled_branches\":{},\"profile_selection_optimization\":{},\
                \"training_results\":[{training_results}]",sites.join(","),
                compiled.profiled_branches(),compiled.profile_selection_optimization());
            compiled
        }
        _ => return Err("Y benchmark mode must be baseline, previous or optimized".into()),
    };
    let prepare_ns = nanos(prepare_started);
    let compile_ns = jit.compile_duration().as_secs_f64() * 1e9;
    let outer_unroll_annotations = jit.outer_unroll_annotations();
    let compilation_kind = if mode == "baseline" {
        "original"
    } else {
        "profiled"
    };
    compilations.push(format!(
        "{{\"kind\":{compilation_kind:?},\"compilation_settings\":{},\"timings\":{},\"optimization_timings\":{},\"materialization_timings\":{},\"verification_checks\":{}}}",
        compilation_settings(options, false, jit.outer_unroll_annotations()),
        jit.compile_timings().to_json(),
        jit.optimization_timings().to_json(),
        jit.materialization_timings().to_json(),
        jit.compile_timings().verification_checks,
    ));
    if let Some(path) = std::env::var_os("Y_CPU_JIT_BENCH_IR") {
        std::fs::write(path, jit.optimized_ir())?;
    }
    if let Some((path, ir)) = instrumented_ir {
        std::fs::write(path, ir)?;
    }
    let llvm = jit.llvm_version();
    // These signatures exactly match the source declarations. `jit` outlives
    // every call and memory has 65536 live, aligned, writable I64 elements.
    let integer: unsafe extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(black_box(jit.function_address("integer_branch")?)) };
    let fib: unsafe extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(black_box(jit.function_address("recursive_fib")?)) };
    let float: unsafe extern "C" fn(i64, f64) -> f64 =
        unsafe { std::mem::transmute(black_box(jit.function_address("float_recurrence")?)) };
    let memory: unsafe extern "C" fn(*mut i64, i64, i64) -> i64 =
        unsafe { std::mem::transmute(black_box(jit.function_address("indexed_memory")?)) };
    let extra = if expanded {
        Some(ExtraFns::new(&jit)?)
    } else {
        None
    };

    let runtime_fns = if runtime {
        Some(RuntimeFns::new(&jit)?)
    } else {
        None
    };

    let copy_fns = if copies {
        Some(CopyFns::new(&jit)?)
    } else {
        None
    };
    let helper_fns = if helpers {
        Some(RuntimeFns::new_helpers(&jit)?)
    } else {
        None
    };

    let extra_len = if expanded { 65536 } else { 0 };
    let a: Vec<f64> = (0..extra_len)
        .map(|i| ((i * 17 + 3) & 1023) as f64 / 1024.0)
        .collect();
    let b: Vec<f64> = (0..extra_len)
        .map(|i| ((i * 29 + 7) & 1023) as f64 / 2048.0)
        .collect();
    let sorted: Vec<i64> = (0..extra_len).map(|i| i * 3 + 1).collect();
    let mut counters = [0_i64; 3];

    let mut data = vec![0_i64; 65536];
    reset(&mut data);
    let started = Instant::now();
    let (first_integer, first_fib, first_float, first_memory) = unsafe {
        (
            integer(integer_n, 123),
            fib(fib_n),
            float(float_n, 0.5),
            memory(data.as_mut_ptr(), memory_n, 123),
        )
    };
    let first_extra_values = if let Some(extra) = &extra {
        unsafe {
            Some((
                (extra.unsigned)(unsigned_n, UNSIGNED_SEED),
                (extra.dot)(a.as_ptr(), b.as_ptr(), dot_n, 123),
                (extra.logical)(counters.as_mut_ptr(), logical_n, 123),
                (extra.search)(sorted.as_ptr(), search_n, 123),
            ))
        }
    } else {
        None
    };
    let first_runtime_values = runtime_fns
        .as_ref()
        .map(|f| unsafe { ((f.string)(string_n, 123), (f.vector)(vec_n, 123)) });
    let first_copy_values = copy_fns.as_ref().map(|f| unsafe {
        (
            (f.byte)(dynamic_byte_n, 123, 1),
            (f.word)(dynamic_i64_n, 123, 8),
            (f.bulk)(bulk_n, 123),
        )
    });
    let first_helper_values = helper_fns
        .as_ref()
        .map(|f| unsafe { ((f.string)(string_n, 123), (f.vector)(vec_n, 123)) });
    let first_call_ns = nanos(started);
    let first_memory_hash = memory_hash(&data);
    let mut first = format!(
        "\"engine\":\"y\",\"mode\":{mode:?},\"suite\":{suite:?},\"optimization_stage\":{stage:?},\"opt_level\":3,\"codegen_opt_level\":{codegen_opt_level},\"codegen_override\":{codegen_override_json},\"training_opt_level\":{training_opt_level},\"training_override\":{training_override_json},\"recognize_rotates\":{recognize_rotates},\"verify_each_pass\":{verify_each_pass},\"profile_edge_counters\":{profile_edge_counters},\"profile_loop_edge_counters\":{profile_loop_edge_counters},\"final_loop_unrolling\":{final_loop_unrolling},\"ir_loop_unrolling\":{ir_loop_unrolling},\"final_unroll_outer_loops\":{final_unroll_outer_loops},\"outer_unroll_policy_active\":{outer_unroll_policy_active},\"outer_unroll_annotations\":{outer_unroll_annotations},\"verification_boundary_policy\":\"input-and-pass-boundaries\",\"optimize_runtime\":{optimize_runtime},\"optimize_runtime_mutations\":{optimize_runtime_mutations},\"optimize_runtime_copies\":{optimize_runtime_copies},\"optimize_call_adapters\":{optimize_call_adapters},\"optimize_helper_effects\":{optimize_helper_effects},\"profile_loop_controls\":{profile_loop_controls},\"llvm\":{llvm:?},\"compile_ns\":{compile_ns},\"prepare_ns\":{prepare_ns},\"first_call_ns\":{first_call_ns},\
         \"first_integer\":\"{first_integer}\",\"first_fib\":\"{first_fib}\",\
         \"first_float\":{first_float:?},\"first_memory\":\"{first_memory}\",\
         \"first_memory_hash\":\"{first_memory_hash}\""
    );
    first.push_str(&preparation_json);
    first.push_str(&format!(",\"compilations\":[{}]", compilations.join(",")));
    if let Some((unsigned, dot, logical, search)) = first_extra_values {
        first.push_str(&format!(
            ",\"first_unsigned\":\"{unsigned}\",\"first_dot\":{dot:?},\
                                \"first_logical\":\"{logical}\",\"first_search\":\"{search}\",\
                                \"first_counters\":{counters:?}"
        ));
    }
    if let Some((string, vector)) = first_runtime_values {
        first.push_str(&format!(
            ",\"first_string\":\"{string}\",\"first_vec\":\"{vector}\""
        ));
    }
    if let Some((byte, word, bulk)) = first_copy_values {
        first.push_str(&format!(",\"first_dynamic_byte\":\"{byte}\",\"first_dynamic_i64\":\"{word}\",\"first_bulk\":\"{bulk}\""));
    }
    if let Some((string, vector)) = first_helper_values {
        first.push_str(&format!(
            ",\"first_helper_string\":\"{string}\",\"first_helper_vec\":\"{vector}\""
        ));
    }
    if cold_only {
        println!("{{{first}}}");
        return Ok(());
    }

    for i in 0..warmup {
        unsafe {
            black_box(integer(integer_n, 123 + i as i64 * 17));
            black_box(fib(fib_n + i as i64 % 2));
            black_box(float(float_n, 0.5 + i as f64 * 0.0001));
            black_box(memory(data.as_mut_ptr(), memory_n, 123 + i as i64 * 17));
            if let Some(extra) = &extra {
                black_box((extra.unsigned)(
                    unsigned_n,
                    UNSIGNED_SEED.wrapping_add(i as u64 * 17),
                ));
                black_box((extra.dot)(
                    a.as_ptr(),
                    b.as_ptr(),
                    dot_n,
                    123 + i as i64 * 17,
                ));
                black_box((extra.logical)(
                    counters.as_mut_ptr(),
                    logical_n,
                    123 + i as i64 * 17,
                ));
                black_box((extra.search)(
                    sorted.as_ptr(),
                    search_n,
                    123 + i as i64 * 17,
                ));
            }
            if let Some(f) = &runtime_fns {
                black_box((f.string)(string_n, 123 + i as i64 * 17));
                black_box((f.vector)(vec_n, 123 + i as i64 * 17));
            }
            if let Some(f) = &copy_fns {
                black_box((f.byte)(dynamic_byte_n, 123 + i as i64 * 17, 1));
                black_box((f.word)(dynamic_i64_n, 123 + i as i64 * 17, 8));
                black_box((f.bulk)(bulk_n, 123 + i as i64 * 17));
            }
            if let Some(f) = &helper_fns {
                black_box((f.string)(string_n, 123 + i as i64 * 17));
                black_box((f.vector)(vec_n, 123 + i as i64 * 17));
            }
        }
    }

    let mut integer_outputs = vec![0_i64; calls];
    let started = Instant::now();
    for (i, output) in integer_outputs.iter_mut().enumerate() {
        *output = unsafe { integer(integer_n, 123 + i as i64 * 17) };
    }
    let integer_ns = nanos(started) / calls as f64;

    let mut fib_outputs = vec![0_i64; calls];
    let started = Instant::now();
    for (i, output) in fib_outputs.iter_mut().enumerate() {
        *output = unsafe { fib(fib_n + i as i64 % 2) };
    }
    let fib_ns = nanos(started) / calls as f64;

    let mut float_outputs = vec![0_f64; calls];
    let started = Instant::now();
    for (i, output) in float_outputs.iter_mut().enumerate() {
        *output = unsafe { float(float_n, 0.5 + i as f64 * 0.0001) };
    }
    let float_ns = nanos(started) / calls as f64;

    reset(&mut data);
    let mut memory_outputs = vec![0_i64; calls];
    let started = Instant::now();
    for (i, output) in memory_outputs.iter_mut().enumerate() {
        *output = unsafe { memory(data.as_mut_ptr(), memory_n, 123 + i as i64 * 17) };
    }
    let memory_ns = nanos(started) / calls as f64;
    let hash = memory_hash(&data);

    let mut extra_json = String::new();
    if let Some(extra) = &extra {
        let mut outputs = vec![0_u64; calls];
        let started = Instant::now();
        for (i, output) in outputs.iter_mut().enumerate() {
            *output =
                unsafe { (extra.unsigned)(unsigned_n, UNSIGNED_SEED.wrapping_add(i as u64 * 17)) };
        }
        let elapsed = nanos(started) / calls as f64;
        let outputs: Vec<String> = outputs.iter().map(u64::to_string).collect();
        extra_json.push_str(&format!(
            ",{{\"name\":\"unsigned_mix\",\"ns_per_call\":{elapsed},\"outputs\":{outputs:?}}}"
        ));

        let mut outputs = vec![0_f64; calls];
        let started = Instant::now();
        for (i, output) in outputs.iter_mut().enumerate() {
            *output = unsafe { (extra.dot)(a.as_ptr(), b.as_ptr(), dot_n, 123 + i as i64 * 17) };
        }
        let elapsed = nanos(started) / calls as f64;
        extra_json.push_str(&format!(
            ",{{\"name\":\"float_dot\",\"ns_per_call\":{elapsed},\"outputs\":{outputs:?}}}"
        ));

        counters.fill(0);
        let mut outputs = vec![0_i64; calls];
        let started = Instant::now();
        for (i, output) in outputs.iter_mut().enumerate() {
            *output =
                unsafe { (extra.logical)(counters.as_mut_ptr(), logical_n, 123 + i as i64 * 17) };
        }
        let elapsed = nanos(started) / calls as f64;
        extra_json.push_str(&format!(",{{\"name\":\"short_circuit\",\"ns_per_call\":{elapsed},\"outputs\":{outputs:?},\"counters\":{counters:?}}}"));

        let mut outputs = vec![0_i64; calls];
        let started = Instant::now();
        for (i, output) in outputs.iter_mut().enumerate() {
            *output = unsafe { (extra.search)(sorted.as_ptr(), search_n, 123 + i as i64 * 17) };
        }
        let elapsed = nanos(started) / calls as f64;
        extra_json.push_str(&format!(
            ",{{\"name\":\"binary_search\",\"ns_per_call\":{elapsed},\"outputs\":{outputs:?}}}"
        ));
    }

    if let Some(f) = &runtime_fns {
        for (name, function, n) in [
            ("string_scan", f.string, string_n),
            ("vec_scan_append", f.vector, vec_n),
        ] {
            let mut outputs = vec![0_i64; calls];
            let started = Instant::now();
            for (i, output) in outputs.iter_mut().enumerate() {
                *output = unsafe { function(n, 123 + i as i64 * 17) };
            }
            let elapsed = nanos(started) / calls as f64;
            extra_json.push_str(&format!(
                ",{{\"name\":{name:?},\"ns_per_call\":{elapsed},\"outputs\":{outputs:?}}}"
            ));
        }
    }

    if let Some(f) = &copy_fns {
        for (name, n, element_size) in [
            ("vec_dynamic_byte", dynamic_byte_n, 1),
            ("vec_dynamic_i64", dynamic_i64_n, 8),
            ("string_bulk_append", bulk_n, 0),
        ] {
            let mut outputs = vec![0_i64; calls];
            let start = Instant::now();
            for (i, output) in outputs.iter_mut().enumerate() {
                let seed = 123 + i as i64 * 17;
                *output = unsafe {
                    if element_size == 1 {
                        (f.byte)(n, seed, 1)
                    } else if element_size == 8 {
                        (f.word)(n, seed, 8)
                    } else {
                        (f.bulk)(n, seed)
                    }
                };
            }
            let elapsed = nanos(start) / calls as f64;
            extra_json.push_str(&format!(
                ",{{\"name\":{name:?},\"ns_per_call\":{elapsed},\"outputs\":{outputs:?}}}"
            ));
        }
    }
    if let Some(f) = &helper_fns {
        for (name, function, n) in [
            ("string_scan_helper", f.string, string_n),
            ("vec_scan_append_helper", f.vector, vec_n),
        ] {
            let mut outputs = vec![0_i64; calls];
            let started = Instant::now();
            for (i, output) in outputs.iter_mut().enumerate() {
                *output = unsafe { function(n, 123 + i as i64 * 17) };
            }
            let elapsed = nanos(started) / calls as f64;
            extra_json.push_str(&format!(
                ",{{\"name\":{name:?},\"ns_per_call\":{elapsed},\"outputs\":{outputs:?}}}"
            ));
        }
    }

    let mut checked_json = String::new();
    if stage == "adapters" {
        let tiny_calls: usize = std::env::var("Y_CPU_JIT_BENCH_CHECKED_TINY_CALLS")
            .unwrap_or_else(|_| "20000".into())
            .parse()?;
        if tiny_calls == 0 {
            return Err("checked tiny calls must be positive".into());
        }
        let tiny = checked_measurement(&jit, 1, tiny_calls, 64, "integer_branch_tiny")?;
        let large = checked_measurement(&jit, integer_n, calls, warmup, "integer_branch_large")?;
        checked_json = format!(",\"checked_results\":[{tiny},{large}]");
    }

    println!(
        "{{{first},\"calls\":{calls},\"warmup\":{warmup},\"results\":[\
         {{\"name\":\"integer_branch\",\"ns_per_call\":{integer_ns},\"outputs\":{integer_outputs:?}}},\
         {{\"name\":\"recursive_fib\",\"ns_per_call\":{fib_ns},\"outputs\":{fib_outputs:?}}},\
         {{\"name\":\"float_recurrence\",\"ns_per_call\":{float_ns},\"outputs\":{float_outputs:?}}},\
         {{\"name\":\"indexed_memory\",\"ns_per_call\":{memory_ns},\"outputs\":{memory_outputs:?},\"memory_hash\":\"{hash}\"}}{extra_json}]{checked_json}}}"
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
