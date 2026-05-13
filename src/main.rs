// ============================================================
//  Y-Lang  —  Compiler CLI Driver
//  main.rs
//
//  The main entry point for the compiler. Consumes a .ysu
//  source file, pushes it through the Lexical, Syntax,
//  and Semantic validation phases, and finally emits PTX.
// ============================================================

mod ast;
mod avx_wrapper;
mod bank_conflict;
mod c_emitter;
mod cpu_emitter;
mod lexer;
mod linear_tracker;
mod llvm_emitter;
mod parser;
mod ptx_emitter;
mod sentinel;
mod type_checker;

use std::env;
use std::fs;
use std::process::exit;

use ast::Item;
use c_emitter::CEmitter;
use cpu_emitter::CpuEmitter;
use lexer::Lexer;
use llvm_emitter::LlvmEmitter;
use parser::Parser;
use ptx_emitter::PtxEmitter;
use type_checker::TypeChecker;

fn main() {
    println!("========================================");
    println!("=== Y-Lang Compiler v0.1 (Prototype) ===");
    println!("========================================\n");

    // Phase 0: Sentinel Hardware Probe
    let hw_profile = sentinel::check_or_probe_hardware();

    let args: Vec<String> = env::args().collect();

    // Find source file (first arg that isn't a flag)
    let source_file = args.iter().skip(1).find(|a| !a.starts_with("--")).cloned();

    let source_code = if let Some(mut file_path) = source_file {
        if std::path::Path::new(&file_path).extension().is_none() {
            file_path.push_str(".ysu");
        }
        println!("[*] Reading source: {}", file_path);
        match fs::read_to_string(&file_path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("[!] Failed to read file: {}", e);
                exit(1);
            }
        }
    } else {
        println!("[*] No input file provided. Running internal test harness.");
        // A hardcoded mock Y-Lang source based on the specification document
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
    println!("[1/4] Running Lexer...");
    let mut lexer = Lexer::new(&source_code);
    let tokens = lexer.tokenize();
    // lexer::print_tokens(&tokens); // Uncomment for verbose token debug
    println!("      -> Extracted {} tokens.", tokens.len());

    // ────────────────────────────────────────────────────────
    // Phase 2: Syntax Parsing (AST)
    // ────────────────────────────────────────────────────────
    println!("[2/4] Constructing AST...");
    let mut parser = Parser::new(tokens);
    let ast = match parser.parse_program() {
        Ok(program) => program,
        Err(e) => {
            eprintln!("\n[!] Syntax Error:\n    {}", e);
            exit(1);
        }
    };
    println!("      -> Successfully parsed {} item(s).", ast.items.len());

    // ────────────────────────────────────────────────────────
    // Phase 3: Semantic Type Checking & Math Verifiers
    // ────────────────────────────────────────────────────────
    println!("[3/4] Running Semantic Type-Checker...");
    let mut type_checker = TypeChecker::new();
    type_checker.check_program(&ast);

    if !type_checker.errors.is_empty() {
        eprintln!(
            "\n[!] The Type-Checker caught {} semantic errors:",
            type_checker.errors.len()
        );
        for err in type_checker.errors {
            eprintln!("    ❌ {}", err);
        }
        eprintln!("\nCompilation aborted to prevent undefined hardware behavior.");
        exit(1);
    }

    // Check if any transfer obligations were left unconsumed via linear tracking
    if type_checker.linear_tracker.has_errors() {
        eprintln!("\n[!] Linear Type Check Failed!");
        for err in &type_checker.linear_tracker.errors {
            eprintln!("    ❌ {}", err);
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
                                "      [Advisory] @ZeroDrift requested on type: {}",
                                type_name
                            );
                            if profile.drift_free_types.contains(&type_name) {
                                println!("        -> Hardware target ({}) natively supports zero drift for {}.", profile.gpu_name, type_name);
                                println!(
                                    "        -> Performance tradeoff: +{} cycles penalty.",
                                    profile.zero_drift_penalty_cycles
                                );
                            } else {
                                println!("        -> WARNING: Target ({}) lacks native zero drift for {}.", profile.gpu_name, type_name);
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

    // Check for --emit-c flag
    let emit_c = args.iter().any(|a| a == "--emit-c" || a == "--target=c");
    let emit_llvm = args
        .iter()
        .any(|a| a == "--emit-llvm" || a == "--target=llvm");
    let mut output_path = args
        .iter()
        .find(|a| a.starts_with("--output="))
        .map(|a| a.trim_start_matches("--output=").to_string())
        .unwrap_or_else(|| "output.c".to_string());

    if output_path.starts_with('-') {
        output_path = format!("./{}", output_path);
    }

    println!("\n✅ Compilation Successful!\n");

    if emit_llvm {
        println!("[4/4] Emitting LLVM IR...");
        let mut emitter = LlvmEmitter::new();
        let ll_output = emitter.emit_program(&ast, &hw_profile);
        let ll_path = output_path.replace(".c", ".ll");
        match fs::write(&ll_path, &ll_output) {
            Ok(_) => println!("      -> Written to: {}", ll_path),
            Err(e) => {
                eprintln!("[!] Failed to write LLVM IR: {}", e);
                exit(1);
            }
        }
        println!("      Compile: clang -O2 -o output {} -lm", ll_path);
    } else if emit_c {
        println!("[4/4] Emitting C11 Code...");
        let mut emitter = CEmitter::new();
        let c_output = emitter.emit_program(&ast);

        match fs::write(&output_path, &c_output) {
            Ok(_) => println!("      -> Written to: {}", output_path),
            Err(e) => {
                eprintln!("[!] Failed to write C output: {}", e);
                exit(1);
            }
        }

        let exe_path = output_path.replace(".c", ".exe");
        println!("      -> Attempting gcc compilation...");
        let gcc_result = std::process::Command::new("gcc")
            .args(&["-std=c11", "-O2", "-o", &exe_path, &output_path, "-lm"])
            .output();

        match gcc_result {
            Ok(output) => {
                if output.status.success() {
                    println!("      ✅ Compiled to native: {}", exe_path);
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    eprintln!("      [!] gcc failed:\n{}", stderr);
                    println!(
                        "      -> C source saved to {} for manual compilation.",
                        output_path
                    );
                }
            }
            Err(_) => {
                println!("      -> gcc not found. C source saved to {}.", output_path);
                println!(
                    "         Compile manually: gcc -std=c11 -O2 -o output {} -lm",
                    output_path
                );
            }
        }
    } else if target_is_cpu {
        println!("[4/4] Emitting CPU AVX-512 Host Code...");
        let mut emitter = CpuEmitter::new();
        let cpu_output = emitter.emit_program(&ast);
        println!("======= GENERATED RUST/AVX BLOB =======");
        println!("{}", cpu_output);
        println!("=======================================");
    } else {
        println!("[4/4] Emitting NVIDIA PTX Assembly...");
        let mut emitter = PtxEmitter::new();
        let ptx_output = emitter.emit_program(&ast, &hw_profile);
        println!("======= GENERATED PTX BLOB =======");
        println!("{}", ptx_output);
        println!("==================================");
    }
}
