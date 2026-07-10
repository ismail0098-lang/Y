// ============================================================
//  YPM  —  Y Package Manager & Build System
//  ypm.rs
//
//  Manages Y projects, initializes directories, parses Ysu.toml
//  manifests, handles dependencies, and coordinates linking.
// ============================================================

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{exit, Command};

macro_rules! log_info {
    ($($arg:tt)*) => {
        println!("\x1b[1;36m[*]\x1b[0m {}", format_args!($($arg)*));
    };
}

macro_rules! log_success {
    ($($arg:tt)*) => {
        println!("\x1b[1;32m[+]\x1b[0m {}", format_args!($($arg)*));
    };
}

macro_rules! log_error {
    ($($arg:tt)*) => {
        eprintln!("\x1b[1;31m[!]\x1b[0m {}", format_args!($($arg)*));
    };
}

struct Manifest {
    name: String,
    version: String,
    entry: String,
    target: String,
    ld_flags: Vec<String>,
    dependencies: Vec<(String, String)>, // name, path
}

impl Manifest {
    fn parse(toml_content: &str) -> Result<Self, String> {
        let mut name = String::new();
        let mut version = String::new();
        let mut entry = "src/main.ysu".to_string();
        let mut target = "native".to_string();
        let mut ld_flags = Vec::new();
        let mut dependencies = Vec::new();

        let mut current_section = "";

        for (line_num, line) in toml_content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with('[') && line.ends_with(']') {
                current_section = line[1..line.len() - 1].trim();
                continue;
            }

            let parts: Vec<&str> = line.splitn(2, '=').collect();
            if parts.len() != 2 {
                return Err(format!("Line {}: invalid key-value pair", line_num + 1));
            }
            let key = parts[0].trim();
            let value = parts[1].trim();

            match current_section {
                "package" => {
                    match key {
                        "name" => name = value.trim_matches('"').to_string(),
                        "version" => version = value.trim_matches('"').to_string(),
                        _ => {}
                    }
                }
                "build" => {
                    match key {
                        "entry" => entry = value.trim_matches('"').to_string(),
                        "target" => target = value.trim_matches('"').to_string(),
                        "ld_flags" => {
                            let stripped = value.trim_matches(|c| c == '[' || c == ']');
                            for item in stripped.split(',') {
                                let item = item.trim().trim_matches('"').trim();
                                if !item.is_empty() {
                                    ld_flags.push(item.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
                "dependencies" => {
                    if value.starts_with('{') && value.ends_with('}') {
                        let inner = value[1..value.len() - 1].trim();
                        let sub_parts: Vec<&str> = inner.splitn(2, '=').collect();
                        if sub_parts.len() == 2 && sub_parts[0].trim() == "path" {
                            let path = sub_parts[1].trim().trim_matches('"').to_string();
                            dependencies.push((key.to_string(), path));
                        }
                    }
                }
                _ => {}
            }
        }

        if name.is_empty() {
            return Err("Missing package.name in Ysu.toml".to_string());
        }

        Ok(Self {
            name,
            version,
            entry,
            target,
            ld_flags,
            dependencies,
        })
    }
}

fn print_usage() {
    println!("Y Package Manager & Build System (YPM) v0.1");
    println!("Usage:");
    println!("  ypm new <project_name>  - Create a new project directory template");
    println!("  ypm init                - Initialize a project in the current directory");
    println!("  ypm build               - Build the current project and dependencies");
    println!("  ypm run                 - Build and run the current project");
    println!("  ypm test                - Run test modules inside tests/");
    println!("  ypm clean               - Clear build target directory");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        exit(1);
    }

    let command = &args[1];
    match command.as_str() {
        "new" => {
            if args.len() < 3 {
                log_error!("project name required for 'new' command.");
                exit(1);
            }
            cmd_new(&args[2]);
        }
        "init" => {
            cmd_init();
        }
        "build" => {
            cmd_build();
        }
        "run" => {
            cmd_run();
        }
        "test" => {
            cmd_test();
        }
        "clean" => {
            cmd_clean();
        }
        _ => {
            log_error!("Unknown command: {}", command);
            print_usage();
            exit(1);
        }
    }
}

fn cmd_new(name: &str) {
    let path = Path::new(name);
    if path.exists() {
        log_error!("Directory '{}' already exists.", name);
        exit(1);
    }

    fs::create_dir_all(path.join("src")).unwrap();
    fs::create_dir_all(path.join("libs")).unwrap();

    let toml_content = format!(
        r#"[package]
name = "{}"
version = "0.1.0"
description = "A high-performance Y project"

[dependencies]

[build]
target = "native"
entry = "src/main.ysu"
ld_flags = []
"#,
        name
    );

    let main_content = r#"// src/main.ysu
fn main() {
    println("Hello, World from YPM!");
}
"#;

    fs::write(path.join("Ysu.toml"), toml_content).unwrap();
    fs::write(path.join("src/main.ysu"), main_content).unwrap();
    fs::write(path.join(".gitignore"), "/target/\n").unwrap();

    log_info!("Created new Y project: '{}'", name);
}

fn cmd_init() {
    let name = env::current_dir()
        .unwrap()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();

    if Path::new("Ysu.toml").exists() {
        log_error!("Ysu.toml already exists in current directory.");
        exit(1);
    }

    fs::create_dir_all("src").ok();
    fs::create_dir_all("libs").ok();

    let toml_content = format!(
        r#"[package]
name = "{}"
version = "0.1.0"
description = "A high-performance Y project"

[dependencies]

[build]
target = "native"
entry = "src/main.ysu"
ld_flags = []
"#,
        name
    );

    let main_content = r#"// src/main.ysu
fn main() {
    println("Hello, World from YPM!");
}
"#;

    fs::write("Ysu.toml", toml_content).unwrap();
    if !Path::new("src/main.ysu").exists() {
        fs::write("src/main.ysu", main_content).unwrap();
    }
    if !Path::new(".gitignore").exists() {
        fs::write(".gitignore", "/target/\n").unwrap();
    }

    log_info!("Initialized Y project in current directory: '{}'", name);
}

fn find_runtime_c() -> Option<PathBuf> {
    if let Ok(exe) = env::current_exe() {
        let mut dir = exe.parent();
        while let Some(d) = dir {
            let candidate = d.join("c_src").join("runtime.c");
            if candidate.exists() {
                return Some(candidate);
            }
            dir = d.parent();
        }
    }
    let cwd_candidate = Path::new("c_src").join("runtime.c");
    if cwd_candidate.exists() {
        return Some(cwd_candidate);
    }
    None
}

fn cmd_build() -> PathBuf {
    if !Path::new("Ysu.toml").exists() {
        log_error!("Ysu.toml not found in the current directory.");
        exit(1);
    }

    let toml_content = fs::read_to_string("Ysu.toml").unwrap();
    let manifest = match Manifest::parse(&toml_content) {
        Ok(m) => m,
        Err(e) => {
            log_error!("Error parsing Ysu.toml: {}", e);
            exit(1);
        }
    };

    log_info!("Building package '{}' v{}...", manifest.name, manifest.version);

    fs::create_dir_all("target").unwrap();

    let mut compiler_bin = PathBuf::new();
    if let Ok(exe) = env::current_exe() {
        let parent = exe.parent().unwrap();
        let candidate = parent.join("Y");
        if candidate.exists() {
            compiler_bin = candidate;
        }
    }
    if compiler_bin.as_os_str().is_empty() {
        compiler_bin = PathBuf::from("Y");
    }

    let mut include_args = Vec::new();
    for (_dep_name, dep_path) in &manifest.dependencies {
        let dep_dir = Path::new(dep_path);
        let src_candidate = dep_dir.join("src");
        let include_path = if src_candidate.exists() {
            src_candidate
        } else {
            dep_dir.to_path_buf()
        };
        include_args.push("-I".to_string());
        include_args.push(include_path.to_string_lossy().to_string());
        log_info!("Registered dependency include path: {}", include_path.display());
    }

    let out_name = &manifest.name;
    let output_bin = Path::new("target").join(out_name);

    if manifest.target == "ptx" {
        let ptx_out = Path::new("target").join(format!("{}.ptx", out_name));
        log_info!("Compiling to GPU PTX: {}", ptx_out.display());
        let mut cmd = Command::new(&compiler_bin);
        cmd.arg(&manifest.entry)
            .arg("--emit-ptx")
            .arg(format!("--output={}", ptx_out.display()))
            .args(&include_args);

        let status = cmd.status().unwrap_or_else(|e| {
            log_error!("Failed to invoke Y compiler: {}", e);
            exit(1);
        });

        if !status.success() {
            log_error!("Compilation failed.");
            exit(1);
        }
        log_success!("Finished building PTX: {}", ptx_out.display());
        ptx_out
    } else if manifest.target == "llvm" {
        let ll_out = Path::new("target").join(format!("{}.ll", out_name));
        log_info!("Emitting LLVM IR: {}", ll_out.display());
        let mut cmd = Command::new(&compiler_bin);
        cmd.arg(&manifest.entry)
            .arg("--emit-llvm")
            .arg(format!("--output={}", ll_out.display()))
            .args(&include_args);

        let status = cmd.status().unwrap_or_else(|e| {
            log_error!("Failed to invoke Y compiler: {}", e);
            exit(1);
        });

        if !status.success() {
            log_error!("Compilation failed.");
            exit(1);
        }
        log_success!("Finished building LLVM IR: {}", ll_out.display());
        ll_out
    } else {
        let temp_ll = Path::new("target").join(format!("{}.tmp.ll", out_name));
        log_info!("Compiling intermediate LLVM IR...");
        
        let mut cmd = Command::new(&compiler_bin);
        cmd.arg(&manifest.entry)
            .arg("--emit-llvm")
            .arg(format!("--output={}", temp_ll.display()))
            .args(&include_args);

        let status = cmd.status().unwrap_or_else(|e| {
            log_error!("Failed to invoke Y compiler: {}", e);
            exit(1);
        });

        if !status.success() {
            log_error!("Compilation failed.");
            exit(1);
        }

        log_info!("Invoking clang link step...");
        let runtime_c = find_runtime_c().unwrap_or_else(|| {
            log_error!("Could not locate runtime.c. Please ensure c_src/runtime.c is available.");
            exit(1);
        });

        let mut clang_cmd = Command::new("clang");
        clang_cmd.arg("-O2")
            .arg("-o")
            .arg(&output_bin)
            .arg(&temp_ll)
            .arg(&runtime_c)
            .arg("-lm")
            .arg("-lX11");

        for flag in &manifest.ld_flags {
            clang_cmd.arg(flag);
        }

        let clang_status = clang_cmd.status().unwrap_or_else(|e| {
            log_error!("Failed to invoke clang: {}", e);
            exit(1);
        });

        fs::remove_file(&temp_ll).ok();

        if !clang_status.success() {
            log_error!("Clang link step failed.");
            exit(1);
        }

        log_success!("Finished building native executable: {}", output_bin.display());
        output_bin
    }
}

fn cmd_run() {
    let output_bin = cmd_build();
    if output_bin.extension().is_none() {
        log_info!("Running {}...", output_bin.display());
        let status = Command::new(&output_bin)
            .status()
            .unwrap_or_else(|e| {
                log_error!("Failed to execute native binary: {}", e);
                exit(1);
            });
        if !status.success() {
            log_error!("Execution exited with non-zero status code.");
            exit(1);
        }
    } else {
        log_info!("Output target ({}) is not executable directly.", output_bin.display());
    }
}

fn cmd_test() {
    log_info!("Running project tests...");
    let tests_dir = Path::new("tests");
    if !tests_dir.exists() || !tests_dir.is_dir() {
        log_info!("No tests/ directory found.");
        return;
    }

    let mut compiler_bin = PathBuf::new();
    if let Ok(exe) = env::current_exe() {
        let parent = exe.parent().unwrap();
        let candidate = parent.join("Y");
        if candidate.exists() {
            compiler_bin = candidate;
        }
    }
    if compiler_bin.as_os_str().is_empty() {
        compiler_bin = PathBuf::from("Y");
    }

    let mut passed = 0;
    let mut failed = 0;

    for entry in fs::read_dir(tests_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "ysu") {
            log_info!("Running test: {}", path.display());
            
            let status = Command::new(&compiler_bin)
                .arg(&path)
                .status()
                .unwrap();

            if status.success() {
                log_success!("Test {} passed compiler verification.", path.file_name().unwrap().to_string_lossy());
                passed += 1;
            } else {
                log_error!("Test {} failed compiler verification.", path.file_name().unwrap().to_string_lossy());
                failed += 1;
            }
        }
    }

    println!("\nTest Results: {} passed, {} failed", passed, failed);
    if failed > 0 {
        exit(1);
    }
}

fn cmd_clean() {
    let target_dir = Path::new("target");
    if target_dir.exists() {
        log_info!("Cleaning build artifacts...");
        fs::remove_dir_all(target_dir).unwrap_or_else(|e| {
            log_error!("Failed to remove target/ directory: {}", e);
            exit(1);
        });
        log_success!("Clean completed.");
    } else {
        log_info!("target/ directory not found, skipping clean.");
    }
}
