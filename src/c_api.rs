// ============================================================
//  Y  —  C-ABI Shared Library Export Interface
//  c_api.rs
//
//  Provides C-compatible export functions for Python (ctypes/PyTorch),
//  C/C++ runtimes, and external JIT execution agents.
// ============================================================

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::autotuner::{Autotuner, Precision};
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::ptx_emitter::PtxEmitter;
use crate::type_checker::TypeChecker;

/// Compiles a Y language source code string into NVIDIA PTX assembly.
///
/// # Safety
/// `source_ptr` and `target_sm` must be valid, null-terminated C strings.
/// The returned pointer must be freed using `y_free_string`.
#[no_mangle]
pub unsafe extern "C" fn y_compile_to_ptx(
    source_ptr: *const c_char,
    target_sm: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    if source_ptr.is_null() {
        if !error_out.is_null() {
            *error_out = CString::new("Source pointer is null").unwrap().into_raw();
        }
        return ptr::null_mut();
    }

    let source = match CStr::from_ptr(source_ptr).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                *error_out = CString::new("Invalid UTF-8 source string").unwrap().into_raw();
            }
            return ptr::null_mut();
        }
    };

    let sm_str = if target_sm.is_null() {
        ""
    } else {
        CStr::from_ptr(target_sm).to_str().unwrap_or("")
    };

    // Step 1: Lexical Analysis
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();

    // Step 2: Parsing
    let mut parser = Parser::new(tokens);
    let ast = match parser.parse_program() {
        Ok(a) => a,
        Err(e) => {
            if !error_out.is_null() {
                *error_out = CString::new(format!("Parser Error: {}", e)).unwrap().into_raw();
            }
            return ptr::null_mut();
        }
    };

    // Step 3: Type Checking
    let mut type_checker = TypeChecker::new();
    type_checker.check_program(&ast);
    if !type_checker.errors.is_empty() {
        if !error_out.is_null() {
            *error_out = CString::new(format!("TypeChecker Error: {}", type_checker.errors.join("; "))).unwrap().into_raw();
        }
        return ptr::null_mut();
    }

    // Step 4: Hardware-Sentient Probe & Target Selection
    let mut hw_profile = crate::sentinel::check_or_probe_hardware();
    if !sm_str.is_empty() && sm_str != "auto" {
        hw_profile.sm_version = sm_str.to_string();
    }
    if hw_profile.sm_version.is_empty() {
        hw_profile.sm_version = "sm_89".to_string();
    }

    let mut emitter = PtxEmitter::new_with_profile(&hw_profile);
    let ptx_output = emitter.emit_program(&ast, &hw_profile);

    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }

    CString::new(ptx_output).unwrap().into_raw()
}

/// Generates candidate tile configurations in JSON format for dynamic autotuning.
///
/// `is_fp8` must reflect the actual precision of the kernel being tuned for -
/// it directly drives shared-memory/occupancy math (FP8 loads half the bytes
/// per element of F16) and cannot be inferred from the tile shapes alone.
///
/// # Safety
/// The returned pointer must be freed using `y_free_string`.
#[no_mangle]
pub unsafe extern "C" fn y_autotune_search_space_json(m: u32, n: u32, k: u32, is_fp8: bool) -> *mut c_char {
    let precision = if is_fp8 { Precision::Fp8 } else { Precision::F16 };
    let candidates = Autotuner::generate_candidates(m, n, k, precision);

    let json_items: Vec<String> = candidates
        .iter()
        .map(|c| {
            format!(
                "{{\"cta_m\":{},\"cta_n\":{},\"cta_k\":{},\"warps_m\":{},\"warps_n\":{},\"num_stages\":{},\"num_warps\":{},\"is_fp8\":{}}}",
                c.cta_m, c.cta_n, c.cta_k, c.warps_m, c.warps_n, c.num_stages, c.num_warps, is_fp8
            )
        })
        .collect();

    let json_output = format!("[{}]", json_items.join(","));
    CString::new(json_output).unwrap().into_raw()
}

/// Executes a Y kernel using CPU Interpreter & Debugger Mode with boundary checks.
///
/// # Safety
/// `source_ptr` must be a valid, null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn y_interpret_kernel(source_ptr: *const c_char, error_out: *mut *mut c_char) -> i32 {
    if source_ptr.is_null() {
        if !error_out.is_null() {
            *error_out = CString::new("Source pointer is null").unwrap().into_raw();
        }
        return -1;
    }

    let source = match CStr::from_ptr(source_ptr).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                *error_out = CString::new("Invalid UTF-8 source string").unwrap().into_raw();
            }
            return -1;
        }
    };

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    let ast = match parser.parse_program() {
        Ok(a) => a,
        Err(e) => {
            if !error_out.is_null() {
                *error_out = CString::new(format!("Parser Error: {}", e)).unwrap().into_raw();
            }
            return -1;
        }
    };

    let mut type_checker = TypeChecker::new();
    type_checker.check_program(&ast);

    let mut cpu_emitter = crate::cpu_emitter::CpuEmitter::new();
    let _code = cpu_emitter.emit_program(&ast);

    // The emitter's refusals were discarded here, so a program it could not
    // lower returned 0 (success) to the C caller.
    if !cpu_emitter.emit_errors.is_empty() {
        if !error_out.is_null() {
            *error_out = CString::new(cpu_emitter.emit_errors.join("\n"))
                .unwrap()
                .into_raw();
        }
        return -1;
    }

    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
    0
}

/// Selects optimal autotuner configuration in JSON format.
///
/// `is_fp8` must reflect the actual precision of the kernel being tuned for -
/// see `y_autotune_search_space_json`.
///
/// # Safety
/// The returned pointer must be freed using `y_free_string`.
#[no_mangle]
pub unsafe extern "C" fn y_autotune_select_config_json(m: u32, n: u32, k: u32, is_fp8: bool) -> *mut c_char {
    let hw = crate::sentinel::HardwareProfile::default();
    let precision = if is_fp8 { Precision::Fp8 } else { Precision::F16 };
    let config = Autotuner::autotune(m, n, k, &hw, precision);
    let json_output = format!(
        "{{\"cta_m\":{},\"cta_n\":{},\"cta_k\":{},\"warps_m\":{},\"warps_n\":{},\"num_stages\":{},\"num_warps\":{}}}",
        config.cta_m, config.cta_n, config.cta_k, config.warps_m, config.warps_n, config.num_stages, config.num_warps
    );
    CString::new(json_output).unwrap().into_raw()
}

/// Performs integer ceiling division (cdiv(a, b)).
#[no_mangle]
pub extern "C" fn y_block_cdiv(a: u32, b: u32) -> u32 {
    if b == 0 {
        return 0;
    }
    (a + b - 1) / b
}

/// Frees a string allocated by Y C-API functions.
///
/// # Safety
/// `s` must be a pointer returned by one of Y's C export functions, or null.
#[no_mangle]
pub unsafe extern "C" fn y_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}


