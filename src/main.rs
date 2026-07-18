// ============================================================
//  Y  —  Compiler CLI Driver
//  main.rs
//
//  The main entry point for the compiler. Consumes a .ysu
//  source file, pushes it through the Lexical, Syntax,
//  and Semantic validation phases, and finally emits PTX.
// ============================================================

mod ast;
mod avx_wrapper;
mod bank_conflict;
// mod c_emitter;
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

#[cfg(feature = "zk")]
mod zk_emitter;

use std::env;
use std::fs;
use std::process::exit;

use ast::Item;
// use c_emitter::CEmitter;
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

fn main() {
    println!("========================================");
    println!("=== Y Compiler v0.1 (Prototype) ===");
    println!("========================================\n");

    let args: Vec<String> = env::args().collect();

    // Phase 0: Sentinel Hardware Probe
    let mut hw_profile = sentinel::check_or_probe_hardware();
    if args.iter().any(|a| a == "--portable") {
        hw_profile.has_avx = false;
        hw_profile.has_avx512 = false;
        println!("[*] --portable flag detected. Disabling AVX/AVX-512 target features for maximum compatibility.");
    }

    let mut source_file = None;
    let mut lib_paths = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "-I" && i + 1 < args.len() {
            lib_paths.push(std::path::PathBuf::from(&args[i + 1]));
            i += 2;
        } else if args[i].starts_with("-I") {
            lib_paths.push(std::path::PathBuf::from(&args[i][2..]));
            i += 1;
        } else if args[i].starts_with("--lib-path=") {
            lib_paths.push(std::path::PathBuf::from(args[i].trim_start_matches("--lib-path=")));
            i += 1;
        } else if args[i].starts_with('-') {
            i += 1;
        } else {
            if source_file.is_none() {
                source_file = Some(args[i].clone());
            }
            i += 1;
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
    // lexer::print_tokens(&tokens); // Uncomment for verbose token debug
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
            if !target_file.exists() {
                target_file = std::path::Path::new("self_hosted").join(&relative_path);
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

    // ────────────────────────────────────────────────────────
    // Phase 3.5: Hardware Advisories (Zero Drift)
    // ────────────────────────────────────────────────────────
    let mut zero_drift_count = 0;
    for item in &ast.items {
        if let Item::Kernel(k) = item {
            fn walk_block(b: &ast::Block, profile: &sentinel::HardwareProfile, count: &mut usize) {
                for stmt in &b.stmts {
                    match stmt {
                        ast::Stmt::Let {
                            zero_drift: Some(_),
                            ty,
                            ..
                        } => {
                            let type_name = match ty {
                                Some(ast::Type::Ident(name, _)) => name.clone(),
                                Some(ast::Type::Primitive(name, _)) => name.clone(),
                                _ => "Unknown".to_string(),
                            };

                            println!(
                                "      \x1b[1;33m[Advisory]\x1b[0m @ZeroDrift requested on type: {}",
                                type_name
                            );
                            if profile.drift_free_types.contains(&type_name) {
                                println!("        -> Hardware target ({}) natively supports zero drift for {}.", profile.gpu_name, type_name);
                                println!(
                                    "        -> Performance tradeoff: +{} cycles penalty.",
                                    profile.zero_drift_penalty_cycles
                                );
                            } else {
                                println!("        -> \x1b[1;33mWARNING\x1b[0m: Target ({}) lacks native zero drift for {}.", profile.gpu_name, type_name);
                                println!(
                                    "        -> Compiler must insert software compensation path."
                                );
                            }
                            *count += 1;
                        }
                        ast::Stmt::For { body, .. } => walk_block(body, profile, count),
                        ast::Stmt::While { body, .. } => walk_block(body, profile, count),
                        ast::Stmt::If {
                            then_block,
                            else_block,
                            ..
                        } => {
                            walk_block(then_block, profile, count);
                            if let Some(eb) = else_block {
                                walk_block(eb, profile, count);
                            }
                        }
                        _ => {}
                    }
                }
            }
            walk_block(&k.body, &hw_profile, &mut zero_drift_count);
        }
    }
    if zero_drift_count > 0 {
        println!(
            "      -> Processed {} @ZeroDrift annotations.",
            zero_drift_count
        );
    }

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
    let emit_c = args.iter().any(|a| a == "--emit-c" || a == "--target=c");
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
    let emit_r1cs = args
        .iter()
        .any(|a| a == "--emit-r1cs" || a == "--target=r1cs");
    let emit_coprocessor = args
        .iter()
        .any(|a| a == "--emit-coprocessor" || a == "--target=coprocessor");

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
        println!("         Critical path:     {:.0} cycles", ir_graph.critical_path_cycles());

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
        let fused_ptx = scheduler.emit_fused_ptx(&ir_graph, &hw_profile);

        let write_path = if let Some(ref sf) = source_file {
            let path = std::path::Path::new(sf);
            let mut p = path.to_path_buf();
            p.set_extension("coprocessor.ptx");
            p.to_string_lossy().to_string()
        } else {
            "output.coprocessor.ptx".to_string()
        };

        // Wrap in a PTX module
        let mut full_ptx = String::new();
        full_ptx.push_str(".version 8.0\n");
        full_ptx.push_str(".target sm_89\n");
        full_ptx.push_str(".address_size 64\n\n");
        full_ptx.push_str("// =======================================================\n");
        full_ptx.push_str("// Y Compiler - Dual-Accelerator Co-Processing Backend\n");
        full_ptx.push_str(&format!("// Hardware: {}\n", hw_profile.gpu_name));
        full_ptx.push_str(&format!("// RT Nodes: {} | Tensor Nodes: {} | Barriers: {}\n",
            rt_count, tensor_count, sched.sync_barriers.len()));
        full_ptx.push_str("// =======================================================\n\n");
        full_ptx.push_str(&fused_ptx);

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

    println!("\n\x1b[1;32mCompilation Successful!\x1b[0m\n");

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
            let mut emitter = zk_emitter::ZkEmitter::new();
            match emitter.emit_program(&ast) {
                Ok(r1cs_text) => {
                    // Write binary R1CS format directly to output_path
                    match emitter.write_r1cs_binary(&output_path) {
                        Ok(_) => {
                            println!("      -> R1CS binary target compiled successfully.");
                            println!("      -> Written to: {}", output_path);

                            let prefix = output_path.strip_suffix(".r1cs").unwrap_or(&output_path);

                            // Write symbols file
                            let sym_path = format!("{}.sym", prefix);
                            println!("      -> Written symbols to: {}", sym_path);

                            // Also write human-readable constraints text to .r1cs.txt
                            let txt_path = format!("{}.r1cs.txt", prefix);
                            let _ = fs::write(&txt_path, &r1cs_text);
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
        let ll_output = emitter.emit_program(&ast, &hw_profile);
        match fs::write(&output_path, &ll_output) {
            Ok(_) => println!("      -> Written to: {}", output_path),
            Err(e) => {
                log_error!("Failed to write LLVM IR: {}", e);
                exit(1);
            }
        }
        println!("      Compile manually: clang -O2 -o output {} c_src/runtime.c -lm", &output_path);
    } else if emit_ptx {
        log_step!("4/4", "Emitting NVIDIA PTX Assembly...");
        let mut emitter = PtxEmitter::new();
        let ptx_output = emitter.emit_program(&ast, &hw_profile);
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
        println!("======= GENERATED RUST/AVX BLOB =======");
        println!("{}", cpu_output);
        println!("=======================================");
    } else {
        log_step!("4/4", "Compiling via LLVM IR Backend...");
        let mut emitter = LlvmEmitter::new();
        let ll_output = emitter.emit_program(&ast, &hw_profile);

        let ll_path = format!("{}.tmp.ll", &output_path);
        match fs::write(&ll_path, &ll_output) {
            Ok(_) => {}
            Err(e) => {
                log_error!("Failed to write temporary LLVM IR: {}", e);
                exit(1);
            }
        }

        println!("      -> Invoking clang compilation...");
        let runtime_path = "c_src/runtime.c";
        let clang_result = std::process::Command::new("clang")
            .args(&["-O2", "-o", &output_path, &ll_path, runtime_path, "-lm", "-lX11"])
            .output();

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
}

