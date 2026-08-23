// ============================================================
//  Y  —  Compiler CLI Driver
//  main.rs
//
//  The main entry point for the compiler. Consumes a .ysu
//  source file, pushes it through the Lexical, Syntax,
//  and Semantic validation phases, and emits to the
//  selected backend (LLVM IR, PTX, CPU, R1CS, Co-Processor).
// ============================================================

mod ast;
mod avx_wrapper;
mod bank_conflict;
mod cpu_emitter;
mod lexer;
mod linear_tracker;
mod llvm_emitter;
mod parser;
mod ptx_emitter;
mod sentinel;
mod type_checker;
mod native_emitter;
mod ir_grapher;
mod rt_core_emitter;
mod quantization_pass;
mod coprocessor_scheduler;
mod autotuner;
mod cuda_runtime;
mod empirical_autotune;
mod zero_drift;
mod cpu_specializer;
mod cpu_gemm;

#[cfg(feature = "zk")]
mod zk_emitter;
#[cfg(feature = "zk")]
mod zk_poseidon_constants;
#[cfg(feature = "zk")]
mod mini_json;
#[cfg(feature = "zk")]
mod circom_lexer;
#[cfg(feature = "zk")]
mod circom_ast;
#[cfg(feature = "zk")]
mod circom_parser;
#[cfg(feature = "zk")]
mod circom_lower;
#[cfg(feature = "zk")]
mod zk_field;
#[cfg(feature = "zk")]
mod zk_solidity;
#[cfg(feature = "zk")]
mod zk_witness;

/// Resident and peak memory, for the `Y_ZK_TIMING` phase report.
///
/// Memory, not time, is what bounds the circuit sizes Y is meant to reach that
/// other compilers cannot: cost is roughly linear per constraint, so the box's
/// RAM sets a hard ceiling on circuit size. A user who hits it needs to know
/// which phase peaked, and a "circuits too big for circom" claim needs the
/// number it depends on to be visible rather than folklore.
///
/// Linux-only (`/proc/self/status`); silently contributes nothing elsewhere.
#[cfg(feature = "zk")]
mod zk_mem {
    pub fn report() -> String {
        #[cfg(target_os = "linux")]
        {
            let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
                return String::new();
            };
            let field = |k: &str| -> Option<f64> {
                status
                    .lines()
                    .find(|l| l.starts_with(k))?
                    .split_whitespace()
                    .nth(1)?
                    .parse::<f64>()
                    .ok()
                    .map(|kb| kb / 1024.0 / 1024.0)
            };
            match (field("VmRSS:"), field("VmHWM:")) {
                (Some(rss), Some(peak)) => {
                    format!("   rss {:>6.2} GB   peak {:>6.2} GB", rss, peak)
                }
                _ => String::new(),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            String::new()
        }
    }
}

/// Counting allocator, reported by `Y_ZK_TIMING=1`.
///
/// Here because the ZK emitter's cost turned out NOT to be where the obvious
/// reasoning put it. A Poseidon circuit is pure field arithmetic, so the natural
/// conclusion is that it is bound by the cost of a field multiply — but
/// measured, the 4.12 M multiplies plus 7.81 M adds in a 1000-hash chain
/// accounted for only ~1.8 s of an 8.6 s emit. The other 81% was **356 million
/// allocations**, because `Fr` was a heap `Vec<u32>` and every element cloned,
/// moved into a HashMap or returned by value hit the allocator.
///
/// `Fr` is a stack `[u64; 4]` now (`zk_field.rs`) and that is down to 7.4 M, so
/// this counter has done its job — it stays because it is the only instrument
/// that could have found the answer, and the same question will be asked again.
///
/// A relaxed atomic increment per allocation is a few nanoseconds and does not
/// distort the ratio it exists to measure.
#[cfg(feature = "alloc-stats")]
mod counting_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub static ALLOCS: AtomicU64 = AtomicU64::new(0);
    pub static BYTES: AtomicU64 = AtomicU64::new(0);

    pub struct Counting;

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Relaxed);
            BYTES.fetch_add(l.size() as u64, Relaxed);
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
            ALLOCS.fetch_add(1, Relaxed);
            BYTES.fetch_add(new as u64, Relaxed);
            unsafe { System.realloc(p, l, new) }
        }
    }

    pub fn counts() -> (u64, u64) {
        (ALLOCS.load(Relaxed), BYTES.load(Relaxed))
    }
}

#[cfg(feature = "alloc-stats")]
#[global_allocator]
static ALLOC: counting_alloc::Counting = counting_alloc::Counting;

/// `(0, 0)` when the counting allocator is not compiled in.
#[cfg(not(feature = "alloc-stats"))]
mod counting_alloc {
    pub fn counts() -> (u64, u64) {
        (0, 0)
    }
}

use std::env;
use std::fs;
use std::process::exit;

use ast::Item;
use cpu_emitter::CpuEmitter;
use lexer::Lexer;
use llvm_emitter::LlvmEmitter;
use parser::Parser;
use ptx_emitter::PtxEmitter;
use type_checker::TypeChecker;
use native_emitter::NativeEmitter;

macro_rules! log_info {
    ($($arg:tt)*) => {
        println!("\x1b[1;36m[*]\x1b[0m {}", format_args!($($arg)*));
    };
}

macro_rules! log_error {
    ($($arg:tt)*) => {
        eprintln!("\x1b[1;31m[!]\x1b[0m {}", format_args!($($arg)*));
    };
}

macro_rules! log_warning {
    ($($arg:tt)*) => {
        println!("\x1b[1;33m[!]\x1b[0m {}", format_args!($($arg)*));
    };
}

macro_rules! log_step {
    ($step:expr, $($arg:tt)*) => {
        println!("\x1b[1;32m[{}]\x1b[0m {}", $step, format_args!($($arg)*));
    };
}

/// `Y --emit-verifier <verification_key.json> [-o Verifier.sol] [--name N]`
///
/// Turns a Groth16 verifying key into a deployable Solidity contract. The key
/// comes from a trusted setup, which Y does not perform - snarkjs
/// (`snarkjs groth16 setup` then `snarkjs zkey export verificationkey`) and
/// arkworks both produce a key this reads.
#[cfg(feature = "zk")]
fn emit_verifier_cli(args: &[String], pos: usize) {
    let vkey_path = match args.get(pos + 1) {
        Some(p) if !p.starts_with('-') => p.clone(),
        _ => {
            log_error!("--emit-verifier needs a verification key: Y --emit-verifier verification_key.json");
            exit(1);
        }
    };
    let out_path = args
        .iter()
        .position(|a| a == "-o" || a == "--output")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let name = args
        .iter()
        .position(|a| a == "--name")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "Groth16Verifier".to_string());

    log_info!("Reading verifying key: {}", vkey_path);
    let json = match fs::read_to_string(&vkey_path) {
        Ok(j) => j,
        Err(e) => {
            log_error!("Failed to read {}: {}", vkey_path, e);
            exit(1);
        }
    };
    let vk = match zk_solidity::parse_snarkjs_vkey(&json) {
        Ok(vk) => vk,
        Err(e) => {
            log_error!("{}", e);
            exit(1);
        }
    };
    log_info!(
        "Groth16 / BN254, {} public input(s), {} IC points",
        vk.num_public_inputs(),
        vk.ic.len()
    );

    let sol = zk_solidity::emit_groth16_verifier(&vk, &name);
    match out_path {
        Some(p) => match fs::write(&p, &sol) {
            Ok(()) => {
                log_step!("done", "Wrote {} ({} bytes)", p, sol.len());
            }
            Err(e) => {
                log_error!("Failed to write {}: {}", p, e);
                exit(1);
            }
        },
        None => print!("{}", sol),
    }
}

/// Flatten an input file into one `(signal name, value text)` pair per WIRE.
///
/// circom's own `input.json` keys a signal by its source name and gives an
/// array signal a JSON array - `{"in": ["1", "2"], "x": 3}` - while the wires
/// behind it are `main.in[0]`, `main.in[1]`, `main.x`. Reading circom's file
/// therefore means flattening it the same way `alloc_signal` flattens a
/// declaration. `parse_scalar_map`, which this replaces, refused an array
/// outright, so no circuit taking an array input could be given one.
///
/// Values keep their source text: a field element routinely exceeds 2^53 and
/// parsing through `f64` would round it silently.
#[cfg(feature = "zk")]
fn flatten_inputs(prefix: &str, v: &mini_json::Json, out: &mut Vec<(String, String)>) -> Result<(), String> {
    match v {
        mini_json::Json::Str(t) | mini_json::Json::Num(t) => {
            out.push((prefix.to_string(), t.clone()));
            Ok(())
        }
        mini_json::Json::Arr(items) => {
            for (i, item) in items.iter().enumerate() {
                flatten_inputs(&format!("{}[{}]", prefix, i), item, out)?;
            }
            Ok(())
        }
        mini_json::Json::Obj(_) => Err(format!(
            "input {:?} is a JSON object; circuit inputs are numbers, strings, or \
             (nested) arrays of them",
            prefix
        )),
        mini_json::Json::Other => Err(format!(
            "input {:?} is not a number, string or array",
            prefix
        )),
    }
}

/// Look each wire's signal name up in the supplied file, marking what it used.
///
/// Two spellings are accepted for the same signal: the source name circom's
/// own `input.json` uses (`a`), and the fully-qualified name that appears in
/// the `.sym` file (`main.a`). Y used to accept only its own third spelling,
/// `maina`, which is neither.
#[cfg(feature = "zk")]
fn bind_inputs(
    names: &[String],
    supplied: &[(String, String)],
    used: &mut [bool],
    inputs_path: &str,
    all_names: &[String],
) -> Result<Vec<zk_emitter::Fr>, String> {
    use zk_emitter::{BigUint, Fr};
    let mut ordered = Vec::with_capacity(names.len());
    for name in names {
        let bare = name.strip_prefix("main.").unwrap_or(name);
        let hit = supplied
            .iter()
            .position(|(k, _)| k == name || k == bare)
            .ok_or_else(|| {
                format!(
                    "input {:?} is missing from {}; this circuit takes [{}]",
                    bare,
                    inputs_path,
                    all_names.join(", ")
                )
            })?;
        used[hit] = true;
        ordered.push(Fr::from_biguint(BigUint::from_str(&supplied[hit].1)));
    }
    Ok(ordered)
}

/// Solves the circuit against a JSON input file and writes a `.wtns`.
///
/// Inputs are matched by NAME against the circuit's input signals, not by
/// position: a file listing them in the wrong order would otherwise produce a
/// valid proof of the wrong statement, which is exactly the class of error this
/// backend cannot afford. Every input must be present and no unknown key is
/// accepted.
#[cfg(feature = "zk")]
fn solve_and_write_witness(
    emitter: &zk_emitter::ZkEmitter,
    inputs_path: &str,
    wtns_path: &str,
) -> Result<usize, String> {
    let json = fs::read_to_string(inputs_path)
        .map_err(|e| format!("Failed to read {}: {}", inputs_path, e))?;
    let root = mini_json::P { b: json.as_bytes(), i: 0 }.value()?;
    let mini_json::Json::Obj(fields) = &root else {
        return Err(
            "expected a JSON object of circuit inputs, e.g. {\"x\": 3, \"in\": [1, 2]}".into(),
        );
    };
    let mut supplied = Vec::new();
    for (k, v) in fields {
        flatten_inputs(k, v, &mut supplied)?;
    }
    let mut used = vec![false; supplied.len()];

    // BOTH lists, and in this order: `execute_host_witness_ir` consumes
    // `pub_in` and `priv_in` positionally. Passing `&[]` for the public list -
    // which is what this did - left every `{public [...]}` signal at zero, so
    // no circom circuit with a public input could be solved.
    let pub_names = emitter.public_input_names();
    let priv_names = emitter.private_input_names();
    let all: Vec<String> = pub_names
        .iter()
        .chain(priv_names.iter())
        .map(|n| n.strip_prefix("main.").unwrap_or(n).to_string())
        .collect();
    let pub_ordered = bind_inputs(&pub_names, &supplied, &mut used, inputs_path, &all)?;
    let ordered = bind_inputs(&priv_names, &supplied, &mut used, inputs_path, &all)?;

    if let Some((k, _)) = supplied.iter().zip(&used).find(|(_, u)| !**u).map(|(kv, _)| kv) {
        return Err(format!(
            "{} sets {:?}, which is not an input of this circuit; it takes [{}]",
            inputs_path,
            k,
            all.join(", ")
        ));
    }

    // Borrowed, not `build_circuit()`: cloning the constraint list to hand it to
    // a read-only consumer is the largest single allocation the ZK pipeline can
    // make, and memory is what bounds circuit size here.
    let circuit = emitter.view();
    let ir = emitter.build_witness_ir();
    let (witness, satisfied) =
        zk_witness::solve_r1cs_witness(circuit.constraints, &ir, circuit.num_variables, &pub_ordered, &ordered);
    if !satisfied {
        return Err(format!(
            "no satisfying witness exists for these inputs. A range check almost \
             certainly failed: comparisons and the bitwise, shift and division \
             gadgets treat values as unsigned {}-bit, so a negative or oversized \
             input is unprovable by design.",
            zk_emitter::ZK_COMPARISON_BITS
        ));
    }
    zk_emitter::ZkEmitter::write_wtns_binary_view(circuit, &witness, wtns_path)
        .map_err(|e| format!("Failed to write {}: {}", wtns_path, e))?;
    Ok(witness.len())
}

/// Compile a circom circuit through Y's R1CS back end.
///
/// The whole point of the front end: a team's existing, audited `.circom`
/// source compiles here with no rewrite, and everything downstream of
/// constraint construction is the same code Y's own language uses.
///
/// `-l <dir>` adds an include search path, matching circom's own flag.
#[cfg(feature = "zk")]
fn compile_circom(path: &str, args: &[String]) {
    use std::path::PathBuf;

    let mut search_paths: Vec<PathBuf> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-l" || args[i] == "--link" {
            if let Some(dir) = args.get(i + 1) {
                search_paths.push(PathBuf::from(dir));
            }
            i += 1;
        } else if let Some(dir) = args[i].strip_prefix("-l") {
            if !dir.is_empty() {
                search_paths.push(PathBuf::from(dir));
            }
        }
        i += 1;
    }

    let output_path = args
        .iter()
        .position(|a| a == "-o" || a == "--output")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| {
            let mut p = PathBuf::from(path);
            p.set_extension("r1cs");
            p.to_string_lossy().to_string()
        });

    log_info!("Reading circom source: {}", path);
    log_step!("1/2", "Parsing and lowering circom to R1CS...");

    let timing = std::env::var("Y_ZK_TIMING").is_ok();
    let t0 = std::time::Instant::now();
    let alloc0 = counting_alloc::counts();

    let emitter = match circom_lower::compile_file(std::path::Path::new(path), &search_paths) {
        Ok(e) => e,
        Err(e) => {
            log_error!("circom compilation failed:");
            eprintln!("    {}", e);
            exit(1);
        }
    };
    if timing {
        eprintln!(
            "[Y ZK TIMING] {:<22} {:>8.3} s{}",
            "circom lower",
            t0.elapsed().as_secs_f64(),
            zk_mem::report()
        );
        // The circom front end shares the linear-combination layer with Y's own,
        // so it can regress into the same quadratic accumulate — report it here
        // too rather than only on the `.ysu` path.
        let (calls, terms) = zk_emitter::lc_simplify_stats();
        eprintln!(
            "[Y ZK TIMING] {:<22} {:>12} calls, {:>12} terms scanned",
            "lc simplify", calls, terms
        );
        let a1 = counting_alloc::counts();
        if a1.0 > alloc0.0 {
            eprintln!(
                "[Y ZK TIMING] {:<22} {:>12} allocs, {:>9.2} GB",
                "lower allocations",
                a1.0 - alloc0.0,
                (a1.1 - alloc0.1) as f64 / 1e9
            );
        }
    }

    let circuit = emitter.view();
    println!(
        "      -> {} constraints, {} wires, {} public input(s), {} private input(s), {} output(s)",
        circuit.constraints.len(),
        circuit.num_variables,
        circuit.public_inputs.len(),
        circuit.private_inputs.len(),
        circuit.outputs.len()
    );

    log_step!("2/2", "Writing R1CS...");
    let t1 = std::time::Instant::now();
    if let Err(e) = emitter.write_r1cs_binary(&output_path) {
        log_error!("failed to write {}: {}", output_path, e);
        exit(1);
    }
    if timing {
        eprintln!(
            "[Y ZK TIMING] {:<22} {:>8.3} s{}",
            "write_r1cs_binary",
            t1.elapsed().as_secs_f64(),
            zk_mem::report()
        );
    }
    println!("      -> Written to: {}", output_path);

    // `--witness inputs.json` also solves and writes the `.wtns`, so the whole
    // prove path is available without a second tool.
    if let Some(i) = args.iter().position(|a| a == "--witness") {
        if let Some(inputs) = args.get(i + 1) {
            let mut wtns = PathBuf::from(&output_path);
            wtns.set_extension("wtns");
            let wtns = wtns.to_string_lossy().to_string();
            match solve_and_write_witness(&emitter, inputs, &wtns) {
                Ok(n) => {
                    println!("      -> Solved {} witness values.", n);
                    println!("      -> Written witness to: {}", wtns);
                }
                Err(e) => {
                    log_error!("witness generation failed:");
                    eprintln!("    {}", e);
                    exit(1);
                }
            }
        }
    }

    println!("\n\x1b[1;32mCompilation Successful!\x1b[0m\n");
}

/// Measured accumulate costs for `@ZeroDrift`, cached in `.ysu_hw_profile`.
///
/// Measuring costs a couple of seconds of GPU time, so it happens once per
/// device and is then read back. Without any measurement the selector still
/// works - it falls back to the narrowest representation that fits, which is
/// deterministic - so a machine with no GPU compiles fine, just less informed.
fn load_or_measure_drift_costs(gpu_name: &str) -> zero_drift::CostTable {
    const PROFILE: &str = ".ysu_hw_profile";

    if let Ok(contents) = fs::read_to_string(PROFILE) {
        let cached = zero_drift::parse_costs(&contents, gpu_name);
        if !cached.is_empty() {
            return cached;
        }
    }

    let Some(costs) = zero_drift::measure_accumulate_costs() else {
        return zero_drift::CostTable::new();
    };

    log_info!("Measured @ZeroDrift accumulate costs on {}", gpu_name);
    let mut sorted: Vec<_> = costs.iter().collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));
    for (repr, ps) in &sorted {
        println!(
            "      -> {:>9}: {:>9.0} ps/acc  {}",
            repr.name(),
            ps,
            if repr.is_exact() { "exact" } else { "not exact (never selected)" }
        );
    }

    // Append rather than rewrite: this file also holds the sentinel probe and
    // the autotuner's measurements.
    use std::io::Write as _;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(PROFILE) {
        let _ = writeln!(f, "{}", zero_drift::serialize_costs(&costs, gpu_name));
    }
    costs
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // `--emit-attention-ptx <head_dim> <seq_len>`
    //
    // Advertised by `src/exact_attention.rs`'s module header ("the
    // `--emit-attention-ptx` CLI ... all take the same string") and invoked by
    // `tools/ptx_bridge.py`, and implemented nowhere: the flag fell through to
    // the ordinary source-file path, which read `64` as `64.ysu` and reported a
    // missing file. The bridge's `check=True` saw the non-zero exit, so the
    // failure was loud rather than silent - but the surface two files documented
    // did not exist.
    //
    // Handled before the banner: stdout is piped straight into
    // `cuModuleLoadData`, so a line of ASCII art on it is a driver parse error
    // rather than a diagnostic.
    if let Some(pos) = args.iter().position(|a| a == "--emit-attention-ptx") {
        let parse = |i: usize, what: &str| -> usize {
            match args.get(i).and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => v,
                None => {
                    log_error!(
                        "--emit-attention-ptx needs {}: Y --emit-attention-ptx <head_dim> <seq_len>",
                        what
                    );
                    exit(1);
                }
            }
        };
        let head_dim = parse(pos + 1, "a head dimension");
        let seq_len = parse(pos + 2, "a sequence length");
        match y::exact_attention::attention_ptx(head_dim, seq_len) {
            // Straight to stdout with no banner: the bridge pipes this into
            // `cuModuleLoadData`, so anything else on the stream is a parse
            // error in the driver rather than a diagnostic.
            Ok(ptx) => print!("{}", ptx),
            Err(why) => {
                log_error!("{}", why);
                exit(1);
            }
        }
        return;
    }

    println!("========================================");
    println!("=== Y Compiler v1.0 ===");
    println!("========================================\n");

    // Verifier generation takes a verifying key, not a .ysu source, and touches
    // none of the compilation pipeline - so handle it before the hardware probe
    // rather than paying for a GPU probe to format a contract.
    #[cfg(feature = "zk")]
    if let Some(pos) = args.iter().position(|a| a == "--emit-verifier") {
        emit_verifier_cli(&args, pos);
        return;
    }
    #[cfg(not(feature = "zk"))]
    if args.iter().any(|a| a == "--emit-verifier") {
        log_error!("--emit-verifier requires a build with the ZK backend: cargo build --release --features zk");
        exit(1);
    }

    // Phase 0: Sentinel Hardware Probe
    let mut hw_profile = sentinel::check_or_probe_hardware();
    if args.iter().any(|a| a == "--portable") {
        hw_profile.has_avx = false;
        hw_profile.has_avx512 = false;
        println!("[*] --portable flag detected. Disabling AVX/AVX-512 target features for maximum compatibility.");
    }

    let mut source_file = None;
    let mut lib_paths = Vec::new();
    // `-o` / `--output` were honoured by `--emit-verifier` and `--target=r1cs`
    // and IGNORED by the main compile path, which read only `--output=`. So
    // `Y foo.ysu -o bar` silently wrote `foo`, and `Y -o bar foo.ysu` compiled
    // `bar` - the value was not consumed here, so it was taken as the source
    // file. Consume it in one place so both readings agree.
    let mut cli_output = None;
    /// Every option this binary reads. An argument starting with `-` that is
    /// not here, and does not begin with one of `KNOWN_FLAG_PREFIXES`, is a
    /// hard error -- see the check after the loop.
    const KNOWN_FLAGS: &[&str] = &[
        "-o", "--output", "-I", "-l", "--link", "--name", "--witness",
        "--portable", "--autotune", "--autotune-force", "--no-autotune",
        "--emit-attention-ptx", "--emit-c", "--emit-coprocessor", "--emit-cpu",
        "--emit-llvm", "--emit-native", "--emit-ptx", "--emit-r1cs",
        "--emit-verifier", "--emit-zk-ptx", "--c",
        "--target=c", "--target=coprocessor", "--target=cpu", "--target=llvm",
        "--target=native", "--target=ptx", "--target=r1cs", "--target=zk-ptx",
    ];
    /// Options that carry their value in the same argument.
    const KNOWN_FLAG_PREFIXES: &[&str] = &["--output=", "--lib-path=", "-I", "-l"];
    let mut unknown_flags: Vec<String> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if (args[i] == "-o" || args[i] == "--output") && i + 1 < args.len() {
            cli_output = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "-I" && i + 1 < args.len() {
            lib_paths.push(std::path::PathBuf::from(&args[i + 1]));
            i += 2;
        } else if args[i].starts_with("-I") {
            lib_paths.push(std::path::PathBuf::from(&args[i][2..]));
            i += 1;
        } else if args[i].starts_with("--lib-path=") {
            lib_paths.push(std::path::PathBuf::from(args[i].trim_start_matches("--lib-path=")));
            i += 1;
        } else if args[i].starts_with('-') {
            if !KNOWN_FLAGS.contains(&args[i].as_str())
                && !KNOWN_FLAG_PREFIXES.iter().any(|p| args[i].starts_with(p))
            {
                unknown_flags.push(args[i].clone());
            }
            i += 1;
        } else {
            if source_file.is_none() {
                source_file = Some(args[i].clone());
            }
            i += 1;
        }
    }

    // An unrecognised flag used to be SKIPPED, so `Y foo.ysu --ptx` ran the
    // LLVM backend and printed "Compiled successfully" over a native ELF. That
    // exact command is in this repo's own build instructions, and `--ptx` is
    // not a flag -- the PTX backend is `--emit-ptx`. `--probe` and
    // `--nonsense-flag` behaved the same way.
    //
    // It is the `--c` bug in its general form: CLAUDE.md records that `--c`
    // "used to be *silently ignored*, so the command this line documented ran
    // the LLVM backend instead", and that one flag was fixed while the arm
    // that ignored ALL of them was left in place. Fixing the instance and not
    // the class is the thing this repo's design rule exists to catch.
    if !unknown_flags.is_empty() {
        log_error!(
            "unrecognised option{}: {}",
            if unknown_flags.len() == 1 { "" } else { "s" },
            unknown_flags.join(", ")
        );
        eprintln!("    Known options: {}", KNOWN_FLAGS.join(" "));
        eprintln!("    (did you mean --emit-ptx? there is no --ptx)");
        exit(1);
    }

    // A `.circom` file is a different language and does not go through Y's
    // lexer, parser or type checker at all - only the R1CS back end is shared.
    // Dispatching on the extension here rather than deeper down keeps that
    // separation honest: there is no point at which circom source is pretended
    // to be Y source.
    #[cfg(feature = "zk")]
    if let Some(ref f) = source_file {
        if std::path::Path::new(f).extension().and_then(|e| e.to_str()) == Some("circom") {
            compile_circom(f, &args);
            return;
        }
    }

    let source_code = if let Some(ref mut file_path) = source_file {
        if std::path::Path::new(&file_path).extension().is_none() {
            file_path.push_str(".ysu");
        }
        log_info!("Reading source: {}", file_path);
        match fs::read_to_string(&file_path) {
            Ok(content) => content,
            Err(e) => {
                log_error!("Failed to read file: {}", e);
                exit(1);
            }
        }
    } else {
        log_info!("No input file provided. Running internal test harness.");
        // A hardcoded mock Y source based on the specification document
        r#"
        @require(avx512 >= 1)

        enum TokenKind {
            Kernel, Let, Type, Ident, Eof
        }

        struct Token {
            kind: TokenKind,
            line: I32,
            lexeme: String,
        }

        struct Lexer {
            tokens: Vec<Token, PageAllocator>,
        }

        @safe
        fn load_source(path: String) -> String {
            let content = File::read(path);
            return content;
        }

        @safe
        fn test_structs() {
            let t = Token { kind: 0, line: 42, lexeme: "EOF" };
            println(t.lexeme);
            print_int(t.line);
        }

        @require(avx512 >= 1)
        kernel matmul(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {
            type ATile = SmemLayout<F16, rows=16, cols=64, swizzle=330>;
            let smem_A = SharedMemory::alloc<ATile>();

            @cache_policy(L2_PERSIST, reuse_count=8)
            let weights: F16 = load(A);

            @cache_policy(L2_EVICT_FIRST)
            let act: F16 = load(B);
            
            let acc: Fragment<MMA_m16n8k16, D, F32> = Fragment::zero();
            let pipe: Pipeline<stages=2, layout=ATile> = Pipeline::init();

            for k in 0..1024 step 16 {
                let tx_A: Transfer<Global, Shared, Async<1>, 128> = cp_async(A[k], smem_A);
                pipe.wait(tx_A);
                barrier::sync();
                
                let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(smem_A);
                let frag_B: Fragment<MMA_m16n8k16, B, F16> = ldmatrix(smem_A);
                let frag_C: Fragment<MMA_m16n8k16, C, F32> = ldmatrix(smem_A);
                
                chisel {
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [smem_ptr];";
                }

                acc = mma_sync(frag_A, frag_B, frag_C); 
            }

            store(acc, C);
        }
        "#
        .to_string()
    };

    // ────────────────────────────────────────────────────────
    // Phase 1: Lexical Analysis
    // ────────────────────────────────────────────────────────
    log_step!("1/4", "Running Lexer...");
    let mut lexer = Lexer::new(&source_code);
    let tokens = lexer.tokenize();
    println!("      -> Extracted {} tokens.", tokens.len());

    // ────────────────────────────────────────────────────────
    // Phase 2: Syntax Parsing (AST)
    // ────────────────────────────────────────────────────────
    log_step!("2/4", "Constructing AST...");
    let mut parser = Parser::new(tokens);
    let mut ast = match parser.parse_program() {
        Ok(program) => program,
        Err(e) => {
            eprintln!("\n[!] Syntax Error:\n    {}", e);
            exit(1);
        }
    };
    println!("      -> Successfully parsed {} item(s).", ast.items.len());

    // Resolve imports recursively
    let parent_dir = if let Some(ref sf) = source_file {
        std::path::Path::new(sf).parent().unwrap_or(std::path::Path::new("")).to_path_buf()
    } else {
        std::path::PathBuf::from("")
    };

    let mut imported_files = std::collections::HashSet::new();
    if let Some(ref sf) = source_file {
        if let Ok(canonical) = fs::canonicalize(sf) {
            imported_files.insert(canonical);
        }
    }

    let mut queue = ast.items;
    let mut index = 0;
    while index < queue.len() {
        if let Item::Import(imp) = &queue[index] {
            let mut relative_path = std::path::PathBuf::new();
            for segment in &imp.path {
                relative_path.push(segment);
            }
            relative_path.set_extension("ysu");

            let mut target_file = parent_dir.join(&relative_path);
            if !target_file.exists() {
                for lib_dir in &lib_paths {
                    let candidate = lib_dir.join(&relative_path);
                    if candidate.exists() {
                        target_file = candidate;
                        break;
                    }
                }
            }

            if target_file.exists() {
                if let Ok(canonical) = fs::canonicalize(&target_file) {
                    if imported_files.insert(canonical) {
                        log_info!("Loading imported module: {}", target_file.display());
                        match fs::read_to_string(&target_file) {
                            Ok(content) => {
                                let mut sub_lexer = Lexer::new(&content);
                                let sub_tokens = sub_lexer.tokenize();
                                let mut sub_parser = Parser::new(sub_tokens);
                                match sub_parser.parse_program() {
                                    Ok(mut sub_prog) => {
                                        sub_prog.items.retain(|item| {
                                            if let Item::Func(f) = item {
                                                f.name != "main"
                                            } else {
                                                true
                                            }
                                        });
                                        queue.extend(sub_prog.items);
                                    }
                                    Err(e) => {
                                        log_error!("Syntax Error in imported module {}:\n    {}", target_file.display(), e);
                                        exit(1);
                                    }
                                }
                            }
                            Err(e) => {
                                log_error!("Failed to read imported file {}: {}", target_file.display(), e);
                                exit(1);
                            }
                        }
                    }
                }
            } else {
                log_warning!("Imported module file not found: {}", target_file.display());
            }
        }
        index += 1;
    }

    // Filter out Item::Import from the final list of items
    ast.items = queue.into_iter().filter(|item| !matches!(item, Item::Import(_))).collect();

    // ────────────────────────────────────────────────────────
    // Phase 3: Semantic Type Checking & Math Verifiers
    // ────────────────────────────────────────────────────────
    log_step!("3/4", "Running Semantic Type-Checker...");
    let mut type_checker = TypeChecker::new();
    type_checker.set_zk_target(
        args.iter().any(|a| a == "--emit-r1cs" || a == "--target=r1cs"),
    );
    type_checker.check_program(&ast);

    if !type_checker.errors.is_empty() {
        log_error!("The Type-Checker caught {} semantic errors:", type_checker.errors.len());
        for err in type_checker.errors {
            eprintln!("    \x1b[1;31m[Error]\x1b[0m {}", err);
        }
        eprintln!("\nCompilation aborted to prevent undefined hardware behavior.");
        exit(1);
    }

    // Check if any transfer obligations were left unconsumed via linear tracking
    if type_checker.linear_tracker.has_errors() {
        log_error!("Linear Type Check Failed!");
        for err in &type_checker.linear_tracker.errors {
            eprintln!("    \x1b[1;31m[Error]\x1b[0m {}", err);
        }
        exit(1);
    }

    println!("      -> 0 Bank Conflicts Detected.");
    println!("      -> Fragment Roles & Linear Obligations verified.");

    // NOTE: @ZeroDrift used to be reported here as a hardware "advisory" that
    // printed what the annotation would cost and claimed the compiler would
    // "insert a software compensation path" - while inserting nothing. It is a
    // real lowering now, chosen per device, and the backends report what they
    // actually selected. See src/zero_drift.rs.


    // ────────────────────────────────────────────────────────
    // Phase 4: Backend Emission
    // ────────────────────────────────────────────────────────
    let mut target_is_cpu = false;
    for item in &ast.items {
        if let Item::Kernel(k) = item {
            for req in &k.requires {
                fn check_expr(e: &ast::Expr, is_cpu: &mut bool) {
                    match e {
                        ast::Expr::Ident(name, _) if name.contains("avx512") => *is_cpu = true,
                        ast::Expr::BinaryOp { left, right, .. } => {
                            check_expr(left, is_cpu);
                            check_expr(right, is_cpu);
                        }
                        _ => {}
                    }
                }
                check_expr(&req.condition, &mut target_is_cpu);
            }
        }
    }

    // Check for target flags
    let emit_c = args
        .iter()
        .any(|a| a == "--emit-c" || a == "--target=c" || a == "--c");
    let emit_llvm = args
        .iter()
        .any(|a| a == "--emit-llvm" || a == "--target=llvm");
    let emit_native = args
        .iter()
        .any(|a| a == "--emit-native" || a == "--target=native");
    let emit_ptx = args
        .iter()
        .any(|a| a == "--emit-ptx" || a == "--target=ptx");
    let emit_cpu = args
        .iter()
        .any(|a| a == "--emit-cpu" || a == "--target=cpu");
    // Empirical autotuning: compile every candidate tile, run it on the GPU
    // that is actually present, correctness-check it, keep the fastest, and
    // cache the answer per (M,N,K,precision,GPU) in `.ysu_hw_profile`.
    //
    // TAKING a measurement is opt-in: it costs seconds of real device time,
    // and a compiler that silently starts benchmarking on the user's GPU
    // during an ordinary build is a bad default - especially on a shared or
    // headless machine, or one whose GPU is busy with the job the user
    // actually cares about. READING a measurement someone already took on
    // this machine is free and strictly better information than the analytic
    // model, so that happens by default (see `Autotuner::autotune`).
    //
    //   --autotune         measure any shape not already cached, then cache it
    //   --autotune-force   re-measure even if cached (after a codegen change)
    //   --no-autotune      analytic model only, ignoring the cache - for
    //                      reproducible codegen that must not depend on what
    //                      is on this machine's disk. Wins over the others.
    if args.iter().any(|a| a == "--no-autotune") {
        autotuner::set_tuning_mode(autotuner::TuningMode::Analytic);
    } else if args.iter().any(|a| a == "--autotune-force") {
        autotuner::set_tuning_mode(autotuner::TuningMode::Remeasure);
    } else if args.iter().any(|a| a == "--autotune") {
        autotuner::set_tuning_mode(autotuner::TuningMode::Measure);
    }
    let emit_r1cs = args
        .iter()
        .any(|a| a == "--emit-r1cs" || a == "--target=r1cs");
    let emit_zk_ptx = args
        .iter()
        .any(|a| a == "--emit-zk-ptx" || a == "--target=zk-ptx");
    let emit_coprocessor = args
        .iter()
        .any(|a| a == "--emit-coprocessor" || a == "--target=coprocessor");

    if emit_zk_ptx {
        #[cfg(not(feature = "zk"))]
        {
            log_error!("The ZK Circuit Backend is not compiled into this binary.");
            eprintln!("    Recompile Y-lang with ZK support enabled: cargo build --features zk");
            exit(1);
        }
        #[cfg(feature = "zk")]
        {
            log_step!("4/4", "Emitting GPU PTX Witness Generator Kernel...");
            let mut emitter = zk_emitter::ZkEmitter::new();
            if let Err(e) = emitter.emit_program(&ast) {
                log_error!("ZK Constraint Lowering Error:\n    {}", e);
                exit(1);
            }
            let graph = emitter.build_witness_ir();
            let mut ptx_emitter = ptx_emitter::PtxEmitter::new();
            let ptx_code = ptx_emitter.emit_witness_generator_ptx(&graph);

            let write_path = if let Some(ref sf) = source_file {
                let path = std::path::Path::new(sf);
                let mut p = path.to_path_buf();
                p.set_extension("witness.ptx");
                p.to_string_lossy().to_string()
            } else {
                "output.witness.ptx".to_string()
            };

            match fs::write(&write_path, &ptx_code) {
                Ok(_) => {
                    println!("      -> GPU PTX Witness Generator Kernel compiled successfully.");
                    println!("      -> Signals compiled: {}", graph.num_signals);
                    println!("      -> Written to: {}", write_path);
                    exit(0);
                }
                Err(e) => {
                    log_error!("Failed to write witness PTX output: {}", e);
                    exit(1);
                }
            }
        }
    }

    if emit_coprocessor {
        log_step!("4/4", "Running Dual-Accelerator Co-Processing Pipeline...");
        println!("      -> Phase A: IR Dependency Graphing...");
        let mut grapher = ir_grapher::DependencyGrapher::new();
        let ir_graph = grapher.analyze_program(&ast).clone();

        let rt_count = ir_graph.rt_core_nodes().len();
        let tensor_count = ir_graph.tensor_core_nodes().len();
        let cross_edges = ir_graph.cross_pipeline_edges().len();

        println!("         RT Core nodes:     {}", rt_count);
        println!("         Tensor Core nodes: {}", tensor_count);
        println!("         Cross-pipe edges:  {}", cross_edges);
        println!("         Sequential total:  {:.0} cycles", ir_graph.total_sequential_cycles());

        println!("      -> Phase B: Co-Processor Scheduling...");
        let mut scheduler = coprocessor_scheduler::CoprocessorScheduler::new();
        scheduler.schedule(&ir_graph, &hw_profile);

        let sched = &scheduler.schedule;
        println!("         SMEM budget:       {} bytes", sched.total_smem_bytes);
        println!("         Sync barriers:     {}", sched.sync_barriers.len());
        println!("         Est. parallel cy:  {:.0}", sched.estimated_total_cycles);
        println!("         Overlap savings:   {:.0} cycles", sched.overlap_savings_cycles);

        for (i, barrier) in sched.sync_barriers.iter().enumerate() {
            if barrier.needs_quantization {
                println!(
                    "         Barrier {}: {:?} → {:?} quantization ({} bytes)",
                    i, barrier.src_precision, barrier.dst_precision, barrier.smem_bytes
                );
            }
        }

        println!("      -> Phase C: Fused PTX Emission...");
        let fused_ptx = match scheduler.emit_fused_ptx(&ir_graph, &hw_profile) {
            Ok(p) => p,
            Err(e) => {
                log_error!("{}", e);
                exit(1);
            }
        };

        let write_path = if let Some(ref sf) = source_file {
            let path = std::path::Path::new(sf);
            let mut p = path.to_path_buf();
            p.set_extension("coprocessor.ptx");
            p.to_string_lossy().to_string()
        } else {
            "output.coprocessor.ptx".to_string()
        };

        // Wrap in a PTX module with dynamic target SM
        let target_sm = if hw_profile.sm_version.starts_with("sm_") {
            hw_profile.sm_version.clone()
        } else if !hw_profile.sm_version.is_empty() && hw_profile.sm_version != "0.0" {
            format!("sm_{}", hw_profile.sm_version.replace('.', ""))
        } else {
            "sm_80".to_string()
        };

        // Emit a COMPLETE module, not an instruction stream.
        //
        // This used to write the scheduler's body directly under a `.version`
        // header: no `.visible .entry`, no `.reg` declarations, and a `.shared`
        // directive sitting at module scope inside the body. `ptxas` rejects
        // that outright, so `--emit-coprocessor` produced a file no tool could
        // consume - while printing "Dual-accelerator PTX generated
        // successfully!". The only thing that knew how to finish the job was a
        // `wrap_ptx` helper inside tests/benchmark_coprocessor_physical.py,
        // which hand-wrote the entry point and register pools in Python.
        //
        // A compiler backend that emits half a kernel and relies on a benchmark
        // script to complete it is not a backend. The envelope belongs here,
        // and `coprocessor_ptx_assembles` now gates it on real `ptxas`.
        //
        // `.shared` declarations are hoisted out of the body because PTX
        // requires them at module scope; the register pools are declared
        // generously because the scheduler does not currently report its own
        // high-water marks, and an over-declared virtual register costs nothing
        // (ptxas allocates what is used).
        let mut shared_decls = String::new();
        let mut body = String::new();
        for line in fused_ptx.lines() {
            if line.trim_start().starts_with(".shared") {
                shared_decls.push_str(line.trim_start());
                shared_decls.push('\n');
            } else {
                body.push_str(line);
                body.push('\n');
            }
        }

        let entry_name = "y_coprocessor_fused";
        let mut full_ptx = String::new();
        full_ptx.push_str(".version 8.0\n");
        full_ptx.push_str(&format!(".target {}\n", target_sm));
        full_ptx.push_str(".address_size 64\n\n");
        full_ptx.push_str("// =======================================================\n");
        full_ptx.push_str("// Y Compiler - Dual-Accelerator Co-Processing Backend\n");
        full_ptx.push_str(&format!("// Hardware: {}\n", hw_profile.gpu_name));
        full_ptx.push_str(&format!("// RT Nodes: {} | Tensor Nodes: {} | Barriers: {}\n",
            rt_count, tensor_count, sched.sync_barriers.len()));
        full_ptx.push_str("// =======================================================\n\n");
        full_ptx.push_str(&shared_decls);
        full_ptx.push('\n');
        full_ptx.push_str(&format!(
            ".visible .entry {}(\n    .param .u64 param_rt_A_ptr,\n    .param .u64 param_nns_query_ptr\n)\n{{\n",
            entry_name
        ));
        for pool in ["%r", "%rt_r", "%qr"] {
            full_ptx.push_str(&format!("    .reg .b32 {}<100>;\n", pool));
        }
        for pool in ["%f", "%rt_f", "%qf"] {
            full_ptx.push_str(&format!("    .reg .f32 {}<100>;\n", pool));
        }
        for pool in ["%rd", "%rt_rd", "%qrd"] {
            full_ptx.push_str(&format!("    .reg .b64 {}<100>;\n", pool));
        }
        for pool in ["%p", "%rt_p", "%qp"] {
            full_ptx.push_str(&format!("    .reg .pred {}<100>;\n", pool));
        }
        full_ptx.push_str("    .reg .b64 rt_A_ptr;\n    .reg .b64 nns_query_ptr;\n\n");
        full_ptx.push_str("    ld.param.u64 rt_A_ptr, [param_rt_A_ptr];\n");
        full_ptx.push_str("    ld.param.u64 nns_query_ptr, [param_nns_query_ptr];\n\n");
        full_ptx.push_str(&body);
        full_ptx.push_str("\n    ret;\n}\n");

        match fs::write(&write_path, &full_ptx) {
            Ok(_) => {
                println!("      -> Written to: {}", write_path);
                println!("      \x1b[1;32mDual-accelerator PTX generated successfully!\x1b[0m");
                std::process::exit(0);
            }
            Err(e) => {
                log_error!("Failed to write co-processor PTX: {}", e);
                exit(1);
            }
        }
    } else if emit_c {
        log_error!("The C backend has been removed. Y now uses LLVM as its primary backend.");
        eprintln!("    To compile your code to a native binary (default behavior), omit backend flags.");
        eprintln!("    To emit LLVM IR, use --emit-llvm.");
        exit(1);
    }

    let mut output_path = args
        .iter()
        .find(|a| a.starts_with("--output="))
        .map(|a| a.trim_start_matches("--output=").to_string())
        .or(cli_output)
        .unwrap_or_else(|| {
            if emit_native {
                "output_bin".to_string()
            } else if emit_llvm {
                if let Some(ref sf) = source_file {
                    let path = std::path::Path::new(sf);
                    let mut p = path.to_path_buf();
                    p.set_extension("ll");
                    p.to_string_lossy().to_string()
                } else {
                    "output.ll".to_string()
                }
            } else if emit_r1cs {
                if let Some(ref sf) = source_file {
                    let path = std::path::Path::new(sf);
                    let mut p = path.to_path_buf();
                    p.set_extension("r1cs");
                    p.to_string_lossy().to_string()
                } else {
                    "output.r1cs".to_string()
                }
            } else {
                if let Some(ref sf) = source_file {
                    let path = std::path::Path::new(sf);
                    let mut p = path.to_path_buf();
                    p.set_extension("");
                    p.to_string_lossy().to_string()
                } else {
                    "output".to_string()
                }
            }
        });

    if output_path.starts_with('-') {
        output_path = format!("./{}", output_path);
    }

    // NOT "Compilation Successful!" - the backend has not run yet. Every arm
    // of the dispatch below can fail and `exit(1)`, and this banner used to
    // print above the failure: `--emit-native` on an `if`, or `--target=r1cs`
    // on an out-of-range comparison operand, both printed success and then a
    // hard error. The real banner is at the end of `main`, which every
    // successful path falls through to (no arm exits 0).
    println!("\n\x1b[1;32mFront-end analysis complete.\x1b[0m\n");

    if emit_r1cs {
        #[cfg(not(feature = "zk"))]
        {
            log_error!("The ZK Circuit Backend is not compiled into this binary.");
            eprintln!("    Recompile Y-lang with ZK support enabled: cargo build --features zk");
            exit(1);
        }
        #[cfg(feature = "zk")]
        {
            log_step!("4/4", "Emitting Rank-1 Constraint System (R1CS)...");

            // `Y_ZK_TIMING=1` prints a per-phase breakdown. The ZK path is a
            // single-threaded pipeline of four phases with very different
            // costs depending on the circuit, and the totals alone are
            // misleading: the polynomial benchmark spends almost everything in
            // `emit_program`, while a Poseidon circuit's linear combinations
            // are ~28 terms wide instead of 1-2 and shift the weight to the
            // writers. Tuning against the totals of one circuit is how
            // `to_decimal_string` came to dominate ZK compilation unnoticed.
            let timing = std::env::var("Y_ZK_TIMING").is_ok();
            macro_rules! phase {
                ($label:expr, $body:expr) => {{
                    let t0 = std::time::Instant::now();
                    let r = $body;
                    if timing {
                        eprintln!(
                            "[Y ZK TIMING] {:<22} {:>8.3} s{}",
                            $label,
                            t0.elapsed().as_secs_f64(),
                            zk_mem::report()
                        );
                    }
                    r
                }};
            }

            let alloc0 = counting_alloc::counts();
            let mut emitter = zk_emitter::ZkEmitter::new();
            let emitted = phase!("emit_program", emitter.emit_program(&ast));
            if timing {
                let a1 = counting_alloc::counts();
                if a1.0 > alloc0.0 {
                    eprintln!(
                        "[Y ZK TIMING] {:<22} {:>12} allocs, {:>9.2} GB",
                        "emit allocations",
                        a1.0 - alloc0.0,
                        (a1.1 - alloc0.1) as f64 / 1e9
                    );
                } else {
                    eprintln!("[Y ZK TIMING] emit allocations       (build with --features alloc-stats)");
                }
            }
            match emitted {
                Ok(r1cs_text) => {
                    // Write binary R1CS format directly to output_path
                    let written = phase!("write_r1cs_binary", emitter.write_r1cs_binary(&output_path));
                    match written {
                        Ok(_) => {
                            println!("      -> R1CS binary target compiled successfully.");
                            println!("      -> Written to: {}", output_path);

                            let prefix = output_path.strip_suffix(".r1cs").unwrap_or(&output_path);

                            // Write symbols file
                            let sym_path = format!("{}.sym", prefix);
                            println!("      -> Written symbols to: {}", sym_path);

                            // Also write human-readable constraints text to .r1cs.txt
                            let txt_path = format!("{}.r1cs.txt", prefix);
                            phase!("write_r1cs_txt", { let _ = fs::write(&txt_path, &r1cs_text); });

                            if timing {
                                let (muls, adds) = zk_emitter::field_op_counts();
                                eprintln!(
                                    "[Y ZK TIMING] {:<22} {:>12} muls, {:>12} adds",
                                    "field ops", muls, adds
                                );
                                let (calls, terms) = zk_emitter::lc_simplify_stats();
                                eprintln!(
                                    "[Y ZK TIMING] {:<22} {:>12} calls, {:>12} terms scanned",
                                    "lc simplify", calls, terms
                                );
                            }

                            // --witness inputs.json also solves the circuit and
                            // writes a .wtns, which is what `snarkjs groth16
                            // prove` needs alongside the .r1cs.
                            if let Some(inputs_path) = args
                                .iter()
                                .position(|a| a == "--witness")
                                .and_then(|i| args.get(i + 1))
                            {
                                let wtns_path = format!("{}.wtns", prefix);
                                match solve_and_write_witness(&emitter, inputs_path, &wtns_path) {
                                    Ok(n) => {
                                        println!("      -> Solved {} witness values.", n);
                                        println!("      -> Written witness to: {}", wtns_path);
                                    }
                                    Err(e) => {
                                        log_error!("{}", e);
                                        exit(1);
                                    }
                                }
                            }
                            println!("      -> Written human-readable constraints to: {}", txt_path);
                        }
                        Err(e) => {
                            log_error!("Failed to write binary R1CS output: {}", e);
                            exit(1);
                        }
                    }
                }
                Err(e) => {
                    log_error!("ZK Constraint Lowering Error:\n    {}", e);
                    exit(1);
                }
            }
        }
    } else if emit_native {
        log_step!("4/4", "Emitting Native x86-64 ELF Binary...");
        let mut emitter = NativeEmitter::new();
        let binary_output = emitter.emit_program(&ast);

        // This path wrote a RUNNABLE ELF whatever the emitter had to skip. Of
        // the backends in this repo it is the one where a silent gap costs the
        // most, so the check goes in before the file is written at all.
        if !emitter.emit_errors.is_empty() {
            for e in &emitter.emit_errors {
                log_error!("{}", e);
            }
            exit(1);
        }

        match fs::write(&output_path, &binary_output) {
            Ok(_) => {
                println!("      -> Written to: {}", output_path);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    if let Ok(metadata) = fs::metadata(&output_path) {
                        let mut perms = metadata.permissions();
                        perms.set_mode(0o755);
                        let _ = fs::set_permissions(&output_path, perms);
                    }
                }
                println!("      \x1b[1;32mCompiled to native ELF executable!\x1b[0m");
            }
            Err(e) => {
                log_error!("Failed to write ELF binary: {}", e);
                exit(1);
            }
        }
    } else if emit_llvm {
        log_step!("4/4", "Emitting LLVM IR...");
        let mut emitter = LlvmEmitter::new();
        emitter.set_drift_costs(load_or_measure_drift_costs(&hw_profile.gpu_name));
        let ll_output = emitter.emit_program(&ast, &hw_profile);
        for line in &emitter.drift_report {
            println!("      -> @ZeroDrift {}", line);
        }
        if !emitter.emit_errors.is_empty() {
            for e in &emitter.emit_errors {
                log_error!("{}", e);
            }
            exit(1);
        }
        match fs::write(&output_path, &ll_output) {
            Ok(_) => println!("      -> Written to: {}", output_path),
            Err(e) => {
                log_error!("Failed to write LLVM IR: {}", e);
                exit(1);
            }
        }
        println!("      Compile manually: clang -O2 -o output {} c_src/runtime.c -lm", &output_path);
    } else if emit_ptx {
        log_step!("4/4", "Emitting NVIDIA PTX Assembly with Triton-Level Optimization Passes...");
        println!("      -> Pass 1: Multi-Stage Asynchronous Software Pipelining Pass (cp.async multi-buffering)");
        println!("      -> Pass 2: Automated Grid Block Swizzling Pass (grouped-raster L2 locality, group size {})", ptx_emitter::GEMM_SWIZZLE_GROUP_SIZE);
        println!(
            "      -> Pass 3: JIT Dynamic Autotuning Pass (CTA tile / num_warps / num_stages search) [{}]",
            match autotuner::tuning_mode() {
                autotuner::TuningMode::Cached =>
                    "cached measurements where available, else the analytic model; \
                     pass --autotune to measure this GPU",
                autotuner::TuningMode::Analytic =>
                    "analytic model only (--no-autotune): cache ignored, nothing measured",
                autotuner::TuningMode::Measure =>
                    "measuring on-device, reusing any cached result for this shape",
                autotuner::TuningMode::Remeasure =>
                    "measuring on-device, ignoring any cached result",
            }
        );
        println!("      -> Pass 4: Automated Shared Memory Layout Permutation Pass (0-bank conflict XOR swizzling)");

        // Autotune diagnostics are printed per @tile'd kernel, using that
        // kernel's own real compile-time M/N/K - not a single hardcoded
        // 1024x1024x1024 guess regardless of what's actually in the source
        // (the previous behavior here). A compile unit can hold multiple
        // kernels, and for any @tile'd one, `emit_program` below now calls
        // `Autotuner::autotune` itself with its real dimensions (see
        // `ptx_emitter::emit_tensor_core_gemm_kernel`) - this block exists
        // only to surface that same result to the CLI's own log output.
        let mut printed_any_tune = false;
        for item in &ast.items {
            if let Item::Kernel(k) = item {
                if let Some(t) = &k.tile {
                    fn as_u32(e: &ast::Expr) -> Option<u32> {
                        match e {
                            ast::Expr::IntLit(v, _) if *v > 0 => u32::try_from(*v).ok(),
                            _ => None,
                        }
                    }
                    if let (Some(m), Some(n), Some(k_dim)) =
                        (as_u32(&t.block_m), as_u32(&t.block_n), t.block_k.as_deref().and_then(as_u32))
                    {
                        let tuned_config = autotuner::Autotuner::autotune(m, n, k_dim, &hw_profile, autotuner::Precision::F16);
                        println!("         [JIT Autotuner Result] `{}` (M={}, N={}, K={}): CTA Tile: {}x{}x{}, Warps: {}, Pipeline Stages: {}",
                            k.name, m, n, k_dim, tuned_config.cta_m, tuned_config.cta_n, tuned_config.cta_k, tuned_config.num_warps, tuned_config.num_stages);
                        printed_any_tune = true;
                    }
                }
            }
        }
        if !printed_any_tune {
            println!("         [JIT Autotuner] No @tile'd kernel in this source - nothing to autotune.");
        }

        let mut emitter = PtxEmitter::new_with_profile(&hw_profile);
        emitter.set_drift_costs(load_or_measure_drift_costs(&hw_profile.gpu_name));
        let ptx_output = emitter.emit_program(&ast, &hw_profile);
        for line in &emitter.drift_report {
            println!("      -> @ZeroDrift {}", line);
        }
        if !emitter.emit_errors.is_empty() {
            for e in &emitter.emit_errors {
                log_error!("{}", e);
            }
            exit(1);
        }
        let write_path = if let Some(ref sf) = source_file {
            let path = std::path::Path::new(sf);
            let mut p = path.to_path_buf();
            p.set_extension("ptx");
            p.to_string_lossy().to_string()
        } else {
            "output.ptx".to_string()
        };
        match fs::write(&write_path, &ptx_output) {
            Ok(_) => println!("      -> Written to: {}", write_path),
            Err(e) => {
                log_error!("Failed to write PTX assembly: {}", e);
                exit(1);
            }
        }
        println!("======= GENERATED PTX BLOB =======");
        println!("{}", ptx_output);
        println!("==================================");
    } else if emit_cpu {
        log_step!("4/4", "Emitting CPU AVX-512 Host Code...");
        let mut emitter = CpuEmitter::new();
        let cpu_output = emitter.emit_program(&ast);

        // Printing the blob regardless of what the emitter refused would hand
        // the user Rust that compiles into a different program - the same
        // "green build, wrong artifact" shape the LLVM and PTX paths were
        // fixed for.
        if !emitter.emit_errors.is_empty() {
            for e in &emitter.emit_errors {
                log_error!("{}", e);
            }
            exit(1);
        }

        println!("======= GENERATED RUST/AVX BLOB =======");
        println!("{}", cpu_output);
        println!("=======================================");
    } else {
        log_step!("4/4", "Compiling via LLVM IR Backend...");
        let mut emitter = LlvmEmitter::new();
        let ll_output = emitter.emit_program(&ast, &hw_profile);

        // This path did not check the emitter's errors at all, so a construct
        // the backend refused still produced a binary and exited 0 — the same
        // "green build, wrong program" shape the PTX backend was fixed for.
        if !emitter.emit_errors.is_empty() {
            for e in &emitter.emit_errors {
                log_error!("{}", e);
            }
            exit(1);
        }

        let ll_path = format!("{}.tmp.ll", &output_path);
        match fs::write(&ll_path, &ll_output) {
            Ok(_) => {}
            Err(e) => {
                log_error!("Failed to write temporary LLVM IR: {}", e);
                exit(1);
            }
        }

        println!("      -> Invoking clang compilation...");

        // `c_src/runtime.c` used to be a bare CWD-relative path, so the LLVM
        // backend only worked when Y was invoked from its own source tree:
        // anywhere else clang reported `no such file or directory:
        // 'c_src/runtime.c'`, which reads as a broken install rather than a
        // wrong working directory. Search the CWD, then the directory the
        // compiler binary lives in and its ancestors (target/release/Y ->
        // repo root), and let `Y_RUNTIME_C` override.
        let runtime_path = match find_runtime_c() {
            Some(p) => p,
            None => {
                log_error!("could not find the Y runtime (`c_src/runtime.c`).");
                eprintln!("    Searched $Y_RUNTIME_C, ./c_src/runtime.c, and the");
                eprintln!("    compiler's own directory upwards. Set Y_RUNTIME_C to its path.");
                exit(1);
            }
        };

        // `-lX11` is needed ONLY by the optional GUI surface, and linking it
        // unconditionally made every headless machine - CI, containers, a
        // server without libX11 - unable to compile any Y program at all.
        // Try with it, and if the LIBRARY is what is missing, link without.
        let base = ["-O2", "-o", output_path.as_str(), ll_path.as_str(), runtime_path.as_str(), "-lm"];
        let with_x11 = std::process::Command::new("clang")
            .args(base)
            .arg("-lX11")
            .output();
        let clang_result = match with_x11 {
            Ok(o) if !o.status.success()
                && String::from_utf8_lossy(&o.stderr).contains("-lX11") =>
            {
                println!("      -> libX11 not present; linking without it (GUI calls will not resolve).");
                std::process::Command::new("clang").args(base).output()
            }
            other => other,
        };

        match clang_result {
            Ok(output) => {
                if output.status.success() {
                    let _ = fs::remove_file(&ll_path);
                    println!("      \x1b[1;32mCompiled successfully to native binary:\x1b[0m {}", output_path);
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    log_error!("clang failed:\n{}", stderr);
                    exit(1);
                }
            }
            Err(e) => {
                let _ = fs::remove_file(&ll_path);
                log_error!("clang not found or failed to execute: {}", e);
                println!("          Make sure clang is installed in your system.");
                exit(1);
            }
        }
    }

    // Reached only when the selected backend produced its artifact. Every
    // failure path above calls `exit(1)` and none of them exits 0.
    println!("\n\x1b[1;32mCompilation Successful!\x1b[0m\n");
}

/// Locate `c_src/runtime.c` without assuming the working directory.
///
/// Order: `$Y_RUNTIME_C`, then `./c_src/runtime.c`, then `c_src/runtime.c`
/// relative to each ancestor of the compiler binary's own directory - which
/// covers the ordinary `target/release/Y` layout from any CWD.
fn find_runtime_c() -> Option<String> {
    if let Ok(p) = std::env::var("Y_RUNTIME_C") {
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
    }
    let rel = std::path::Path::new("c_src").join("runtime.c");
    if rel.exists() {
        return Some(rel.to_string_lossy().into_owned());
    }
    let exe = std::env::current_exe().ok()?;
    let mut dir = exe.parent();
    while let Some(d) = dir {
        let cand = d.join("c_src").join("runtime.c");
        if cand.exists() {
            return Some(cand.to_string_lossy().into_owned());
        }
        dir = d.parent();
    }
    None
}
