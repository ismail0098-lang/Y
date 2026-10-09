//! Loader probes execute in fresh processes so failed mock LLVM libraries do
//! not affect the native LLVM singleton or other JIT tests.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(dead_code)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;

// Include the private loader with a local error type. This exercises its
// candidate helper with exact paths without adding a production test env var.
#[derive(Clone, Debug)]
struct JitError(String);
impl JitError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
#[path = "../src/cpu_jit/llvm.rs"]
mod llvm;

#[repr(C)]
struct DlInfo {
    name: *const c_char,
    base: *mut c_void,
    symbol: *const c_char,
    address: *mut c_void,
}
#[link(name = "dl")]
unsafe extern "C" {
    fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
}

fn real_library() -> PathBuf {
    let api = llvm::api().expect("a compatible LLVM 17+ installation is required");
    unsafe {
        let mut info = DlInfo {
            name: std::ptr::null(),
            base: std::ptr::null_mut(),
            symbol: std::ptr::null(),
            address: std::ptr::null_mut(),
        };
        assert_ne!(dladdr(api.LLVMGetVersion as *const c_void, &mut info), 0);
        assert!(!info.name.is_null());
        PathBuf::from(std::ffi::OsStr::from_bytes(
            CStr::from_ptr(info.name).to_bytes(),
        ))
        .canonicalize()
        .expect("LLVM shared library path")
    }
}

fn scratch(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("y-cpu-jit-loader-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn fake_library(directory: &Path, version: u32) -> PathBuf {
    let source = directory.join("mock.c");
    let library = directory.join("libLLVM-mock.so");
    std::fs::write(
        &source,
        format!(
            "#include <stdlib.h>\n\
             void LLVMGetVersion(unsigned *a, unsigned *b, unsigned *c) {{ *a = {version}; *b = 0; *c = 0; }}\n\
             void LLVMInitializeX86TargetInfo(void) {{ abort(); }}\n"
        ),
    )
    .unwrap();
    let output = Command::new(std::env::var_os("CC").unwrap_or_else(|| "cc".into()))
        .args(["-shared", "-fPIC"])
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .expect("C compiler is required for loader fixtures");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    library
}

fn candidate(path: &Path) -> CString {
    CString::new(path.as_os_str().as_bytes()).unwrap()
}

fn child(test: &str, case: &str, paths: &[(&str, &Path)]) {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", test, "--nocapture"])
        .env("Y_CPU_JIT_LOADER_CASE", case)
        .env_remove("Y_LLVM_LIBRARY");
    for (key, path) in paths {
        command.env(key, path);
    }
    let output = command.output().expect("isolated LLVM loader subprocess");
    assert!(
        output.status.success(),
        "case {case} failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn candidate_discovery_falls_back_after_incomplete_and_old_libraries() {
    if std::env::var_os("Y_CPU_JIT_LOADER_CASE").is_some() {
        let mock = PathBuf::from(std::env::var_os("Y_CPU_JIT_LOADER_MOCK").unwrap());
        let valid = PathBuf::from(std::env::var_os("Y_CPU_JIT_LOADER_VALID").unwrap());
        let api = unsafe { llvm::load_candidates(&[candidate(&mock), candidate(&valid)]) }
            .expect("an incompatible first library must not prevent fallback");
        assert!(
            api.version()
                .split('.')
                .next()
                .unwrap()
                .parse::<u32>()
                .unwrap()
                >= 17
        );
        return;
    }
    let real = real_library();
    for version in [16, 24] {
        let directory = scratch(&format!("fallback-{version}"));
        let mock = fake_library(&directory, version);
        let valid = directory.join("libLLVM-valid.so");
        std::os::unix::fs::symlink(&real, &valid).unwrap();
        child(
            "candidate_discovery_falls_back_after_incomplete_and_old_libraries",
            "fallback",
            &[
                ("Y_CPU_JIT_LOADER_MOCK", &mock),
                ("Y_CPU_JIT_LOADER_VALID", &valid),
            ],
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn explicit_override_is_strict_even_when_real_llvm_is_installed() {
    if std::env::var_os("Y_CPU_JIT_LOADER_CASE").is_some() {
        let mock = std::env::var_os("Y_CPU_JIT_LOADER_MOCK").unwrap();
        std::env::set_var("Y_LLVM_LIBRARY", mock);
        let error = match llvm::api() {
            Ok(_) => panic!("an incomplete explicit LLVM override was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("Y_LLVM_LIBRARY override failed"), "{error}");
        assert!(error.contains("LLVMInitializeX86Target"), "{error}");
        return;
    }
    let _ = real_library();
    let directory = scratch("override");
    let mock = fake_library(&directory, 24);
    child(
        "explicit_override_is_strict_even_when_real_llvm_is_installed",
        "override",
        &[("Y_CPU_JIT_LOADER_MOCK", &mock)],
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn failed_discovery_reports_each_candidate_and_its_reason() {
    if std::env::var_os("Y_CPU_JIT_LOADER_CASE").is_some() {
        let mock = PathBuf::from(std::env::var_os("Y_CPU_JIT_LOADER_MOCK").unwrap());
        let absent = mock.with_file_name("does-not-exist.so");
        let error = match unsafe { llvm::load_candidates(&[candidate(&mock), candidate(&absent)]) }
        {
            Ok(_) => panic!("no candidate supplied a compatible LLVM API"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("libLLVM-mock.so"), "{error}");
        assert!(error.contains("LLVMInitializeX86Target"), "{error}");
        assert!(error.contains("does-not-exist.so"), "{error}");
        assert!(error.contains("Install LLVM 17+"), "{error}");
        return;
    }
    let directory = scratch("errors");
    let mock = fake_library(&directory, 24);
    child(
        "failed_discovery_reports_each_candidate_and_its_reason",
        "errors",
        &[("Y_CPU_JIT_LOADER_MOCK", &mock)],
    );
    std::fs::remove_dir_all(directory).unwrap();
}
