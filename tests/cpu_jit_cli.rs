#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

fn run(source: &str, flags: &[&str]) -> std::process::Output {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "y_cpu_jit_cli_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("program.ysu");
    std::fs::write(&file, source).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&file)
        .args(flags)
        .current_dir(&dir)
        .output()
        .unwrap();
    // CPU JIT runs directly and does not probe GPU hardware or write artifacts.
    let names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names, [std::ffi::OsString::from("program.ysu")]);
    std::fs::remove_dir_all(dir).unwrap();
    result
}

#[test]
fn cli_runs_native_main_with_full_width_printing() {
    for level in ["-O0", "-O3"] {
        let out = run(
            "fn main() -> I32 { println(\"JIT alive\"); print_int(4294967296); return 17; }",
            &["--jit", level],
        );
        assert_eq!(
            out.status.code(),
            Some(17),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(String::from_utf8_lossy(&out.stdout).contains("JIT alive\n4294967296"));
        assert!(String::from_utf8_lossy(&out.stderr).contains("[CPU JIT] LLVM"));
    }
}

#[test]
fn cli_accepts_void_main_and_target_alias() {
    let out = run("fn main() { println(\"done\"); }", &["--target=jit"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("done\n"));
}

#[test]
fn cli_refuses_wrong_entry_abi_and_conflicting_flags() {
    for source in [
        "fn main(x: I64) -> I32 { return 0; }",
        "fn main() -> F64 { return 2.0; }",
        "fn helper() -> I32 { return 0; }",
    ] {
        let out = run(source, &["--jit"]);
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("--jit requires fn main()"));
    }
    for flag in [
        "--emit-llvm",
        "--target=native",
        "-g",
        "--portable",
        "--output=bad",
    ] {
        let out = run("fn main() {}", &["--jit", flag]);
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("--jit cannot be combined"));
    }
}

#[test]
fn cli_frontend_failure_is_a_failure() {
    let out = run("fn main() -> I32 { return unknown; }", &["--jit"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown"));
}
