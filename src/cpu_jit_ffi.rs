//! Embeddable C/ctypes interface to the in-process CPU JIT.
use crate::cpu_jit::{BranchProfile, CpuJit, JitOptions, JitValue};
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

pub const CPU_JIT_OPTIONS_ABI_VERSION: u32 = 1;
pub const CPU_JIT_OPT_LEVEL_INHERIT: i32 = -1;

/// Versioned C compilation options. See `c_src/y_cpu_jit.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YCpuJitOptions {
    pub abi_version: u32,
    pub struct_size: u32,
    pub opt_level: u32,
    /// -1 inherits opt_level; otherwise an explicit level from 0 through 3.
    pub training_opt_level: i32,
    /// -1 inherits opt_level; otherwise an explicit level from 0 through 3.
    pub codegen_opt_level: i32,
}

impl Default for YCpuJitOptions {
    fn default() -> Self {
        Self {
            abi_version: CPU_JIT_OPTIONS_ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            opt_level: JitOptions::default().opt_level as u32,
            training_opt_level: CPU_JIT_OPT_LEVEL_INHERIT,
            codegen_opt_level: CPU_JIT_OPT_LEVEL_INHERIT,
        }
    }
}

fn decode_level(level: u32) -> Result<u8, String> {
    if level > 3 {
        Err("opt_level must be 0, 1, 2, or 3".into())
    } else {
        Ok(level as u8)
    }
}

fn decode_override(level: i32, name: &str) -> Result<Option<u8>, String> {
    match level {
        CPU_JIT_OPT_LEVEL_INHERIT => Ok(None),
        0..=3 => Ok(Some(level as u8)),
        _ => Err(format!("{name} must be -1 (inherit), 0, 1, 2, or 3")),
    }
}

unsafe fn decode_options(options: *const YCpuJitOptions) -> Result<JitOptions, String> {
    if options.is_null() {
        return Ok(JitOptions::default());
    }
    // Only read the common prefix until the version and exact size are checked.
    let header = options.cast::<u32>();
    if *header != CPU_JIT_OPTIONS_ABI_VERSION {
        return Err("unsupported CPU JIT options ABI version".into());
    }
    if *header.add(1) != std::mem::size_of::<YCpuJitOptions>() as u32 {
        return Err("CPU JIT options size does not match this library".into());
    }
    let options = &*options;
    Ok(JitOptions {
        opt_level: decode_level(options.opt_level)?,
        training_opt_level: decode_override(options.training_opt_level, "training_opt_level")?,
        codegen_opt_level: decode_override(options.codegen_opt_level, "codegen_opt_level")?,
        ..JitOptions::default()
    })
}

unsafe fn set_error(out: *mut *mut c_char, message: &str) {
    if !out.is_null() {
        // Returned errors use the existing y_free_string allocator contract.
        *out = CString::new(message.replace('\0', "\\0"))
            .unwrap()
            .into_raw();
    }
}

/// Compile null-terminated UTF-8 Y source. Returns null on failure.
/// Errors are freed with `y_free_string`; handles with `y_cpu_jit_free`.
///
/// # Safety
/// `source` must be readable and NUL-terminated; `error_out`, if nonnull,
/// must be writable. Use/free the handle on the creating thread. Executing
/// native code requires trusted source and the exact lowered function ABI.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_compile(
    source: *const c_char,
    opt_level: u32,
    error_out: *mut *mut c_char,
) -> *mut c_void {
    compile(
        source,
        OptionsInput::Legacy(opt_level),
        error_out,
        CompileMode::Normal,
    )
}

enum OptionsInput {
    Legacy(u32),
    Versioned(*const YCpuJitOptions),
}

enum CompileMode {
    Normal,
    Instrument,
    Profile(*mut c_void),
}

unsafe fn compile(
    source: *const c_char,
    options: OptionsInput,
    error_out: *mut *mut c_char,
    mode: CompileMode,
) -> *mut c_void {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        if source.is_null() {
            return Err("CPU JIT source pointer is null".to_string());
        }
        let options = match options {
            OptionsInput::Legacy(level) => JitOptions {
                opt_level: decode_level(level)?,
                ..JitOptions::default()
            },
            OptionsInput::Versioned(options) => decode_options(options)?,
        };
        let source = CStr::from_ptr(source)
            .to_str()
            .map_err(|_| "CPU JIT source is not UTF-8".to_string())?;
        match mode {
            CompileMode::Normal => CpuJit::compile_with_options(source, options),
            CompileMode::Instrument => CpuJit::compile_instrumented(source, options),
            CompileMode::Profile(training_handle) => {
                if training_handle.is_null() {
                    return Err("CPU JIT training handle pointer is null".to_string());
                }
                let profile = (&*training_handle.cast::<CpuJit>())
                    .branch_profile()
                    .map_err(|e| e.to_string())?;
                CpuJit::compile_with_profile(source, options, &profile)
            }
        }
        .map_err(|error| error.to_string())
    }));
    match result {
        Ok(Ok(jit)) => Box::into_raw(Box::new(jit)).cast(),
        Ok(Err(error)) => {
            set_error(error_out, &error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error(error_out, "CPU JIT compiler panicked");
            ptr::null_mut()
        }
    }
}

/// Compile with atomic branch counters. Compilation does not execute source;
/// subsequent explicit native or dynamic calls supply training observations.
///
/// # Safety
/// Same source/error/thread requirements as `y_cpu_jit_compile`.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_compile_instrumented(
    source: *const c_char,
    opt_level: u32,
    error_out: *mut *mut c_char,
) -> *mut c_void {
    compile(
        source,
        OptionsInput::Legacy(opt_level),
        error_out,
        CompileMode::Instrument,
    )
}

/// Snapshot a training session and compile a new, independently owned session.
/// The source must produce exactly the original training IR. No source runs
/// during compilation, and the training session and its addresses stay valid.
///
/// # Safety
/// Source/error requirements are the same as `y_cpu_jit_compile`.
/// `training_handle` must be a live instrumented handle on its creating thread.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_compile_profiled(
    source: *const c_char,
    opt_level: u32,
    training_handle: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_void {
    compile(
        source,
        OptionsInput::Legacy(opt_level),
        error_out,
        CompileMode::Profile(training_handle),
    )
}

/// Initialize options with inherited training/codegen policy and final O3.
/// On error, the options storage is unchanged.
///
/// # Safety
/// `options` must be writable for `options_size` bytes; `error_out`, if nonnull,
/// must be writable and disjoint from the options storage.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_options_init(
    options: *mut YCpuJitOptions,
    options_size: u32,
    error_out: *mut *mut c_char,
) -> i32 {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        if options.is_null() {
            return Err("CPU JIT options pointer is null".to_string());
        }
        if options_size != std::mem::size_of::<YCpuJitOptions>() as u32 {
            return Err("CPU JIT options size does not match this library".to_string());
        }
        ptr::write(options, YCpuJitOptions::default());
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_error(error_out, &error);
            -1
        }
        Err(_) => {
            set_error(error_out, "CPU JIT options initialization panicked");
            -1
        }
    }
}

/// Compile with versioned options. Null options select unchanged defaults.
///
/// # Safety
/// Same source/error/thread requirements as `y_cpu_jit_compile`. Nonnull
/// options must have a readable two-u32 header, and the full declared object
/// when its ABI version and size match. Compilation copies the options.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_compile_with_options(
    source: *const c_char,
    options: *const YCpuJitOptions,
    error_out: *mut *mut c_char,
) -> *mut c_void {
    compile(
        source,
        OptionsInput::Versioned(options),
        error_out,
        CompileMode::Normal,
    )
}

/// Compile atomic branch instrumentation with versioned options.
///
/// # Safety
/// Same requirements as `y_cpu_jit_compile_with_options`.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_compile_instrumented_with_options(
    source: *const c_char,
    options: *const YCpuJitOptions,
    error_out: *mut *mut c_char,
) -> *mut c_void {
    compile(
        source,
        OptionsInput::Versioned(options),
        error_out,
        CompileMode::Instrument,
    )
}

/// Snapshot training counters and compile an independent session with options.
/// Training IR policy is ignored for this final compilation. The caller selects
/// final options explicitly; they are not recovered from the training handle.
///
/// # Safety
/// Same requirements as `y_cpu_jit_compile_with_options`; `training_handle`
/// must be a live instrumented handle on its creating thread.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_compile_profiled_with_options(
    source: *const c_char,
    options: *const YCpuJitOptions,
    training_handle: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_void {
    compile(
        source,
        OptionsInput::Versioned(options),
        error_out,
        CompileMode::Profile(training_handle),
    )
}

fn json_quote(value: &str) -> String {
    let mut result = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            ch if ch.is_control() => write!(result, "\\u{:04x}", ch as u32).unwrap(),
            ch => result.push(ch),
        }
    }
    result.push('"');
    result
}

fn profile_json(profile: &BranchProfile) -> String {
    let fingerprint = profile
        .fingerprint()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let sites = profile
        .sites()
        .iter()
        .map(|site| {
            format!(
                "{{\"function\":{},\"block\":{},\"true_count\":{},\"false_count\":{}}}",
                json_quote(&site.function),
                json_quote(&site.block),
                site.true_count,
                site.false_count,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"fingerprint\":\"{fingerprint}\",\"total_observations\":{},\"sites\":[{sites}]}}",
        profile.total_observations()
    )
}

/// Return an owned JSON snapshot of branch outcomes. Free with `y_free_string`.
/// JSON contains a hexadecimal IR fingerprint, total_observations, and all
/// original sites with function/block names and true_count/false_count.
/// Counters wrap after 2^64 observations per edge; concurrent snapshots read
/// each counter atomically without a single simultaneous point across sites.
///
/// # Safety
/// `handle` must be a live instrumented CPU JIT handle on this thread;
/// `error_out` must be writable if nonnull.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_branch_profile(
    handle: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            return Err("CPU JIT handle pointer is null".to_string());
        }
        let profile = (&*handle.cast::<CpuJit>())
            .branch_profile()
            .map_err(|error| error.to_string())?;
        CString::new(profile_json(&profile))
            .map_err(|_| "CPU JIT branch profile contains NUL".to_string())
    }));
    match result {
        Ok(Ok(json)) => json.into_raw(),
        Ok(Err(error)) => {
            set_error(error_out, &error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error(error_out, "CPU JIT branch profile snapshot panicked");
            ptr::null_mut()
        }
    }
}

/// Return owned compilation phase timings as integer-nanosecond JSON.
/// Free the returned string with `y_free_string`. Reading timings never runs Y.
///
/// # Safety
/// `handle` must be live on its creating thread; `error_out` writable if nonnull.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_compile_timings(
    handle: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            return Err("CPU JIT handle pointer is null".to_string());
        }
        CString::new((&*handle.cast::<CpuJit>()).compile_timings().to_json())
            .map_err(|_| "CPU JIT compilation timings contain NUL".to_string())
    }));
    match result {
        Ok(Ok(json)) => json.into_raw(),
        Ok(Err(error)) => {
            set_error(error_out, &error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error(error_out, "CPU JIT compilation timings lookup panicked");
            ptr::null_mut()
        }
    }
}

/// Return owned optimization subphase timings as integer-nanosecond JSON.
/// Free the returned string with `y_free_string`. Reading timings never runs Y.
/// The subphase total equals the ordinary compilation optimization interval.
///
/// # Safety
/// `handle` must be live on its creating thread; `error_out` writable if nonnull.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_optimization_timings(
    handle: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            return Err("CPU JIT handle pointer is null".to_string());
        }
        CString::new((&*handle.cast::<CpuJit>()).optimization_timings().to_json())
            .map_err(|_| "CPU JIT optimization timings contain NUL".to_string())
    }));
    match result {
        Ok(Ok(json)) => json.into_raw(),
        Ok(Err(error)) => {
            set_error(error_out, &error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error(error_out, "CPU JIT optimization timings lookup panicked");
            ptr::null_mut()
        }
    }
}

/// Return owned materialization timings and object metadata as JSON.
/// Free the returned string with `y_free_string`. Reading timings never runs Y.
/// The disjoint duration total equals the compilation materialization interval;
/// first-lookup children and object/lookup metadata are excluded from that sum.
///
/// # Safety
/// `handle` must be live on its creating thread; `error_out` writable if nonnull.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_materialization_timings(
    handle: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            return Err("CPU JIT handle pointer is null".to_string());
        }
        CString::new(
            (&*handle.cast::<CpuJit>())
                .materialization_timings()
                .to_json(),
        )
        .map_err(|_| "CPU JIT materialization timings contain NUL".to_string())
    }));
    match result {
        Ok(Ok(json)) => json.into_raw(),
        Ok(Err(error)) => {
            set_error(error_out, &error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error(error_out, "CPU JIT materialization timings lookup panicked");
            ptr::null_mut()
        }
    }
}

/// Look up a native C ABI address. Returns null and an error if absent.
///
/// # Safety
/// `handle` must be a live CPU JIT handle on this thread; `name` a readable
/// NUL-terminated UTF-8 string; `error_out` writable if nonnull. The address
/// must not be called after the handle is freed or with an incorrect signature.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_function(
    handle: *mut c_void,
    name: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_void {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
    if handle.is_null() || name.is_null() {
        set_error(error_out, "CPU JIT handle/name pointer is null");
        return ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let name = CStr::from_ptr(name)
            .to_str()
            .map_err(|_| "CPU JIT function name is not UTF-8".to_string())?;
        (&*handle.cast::<CpuJit>())
            .function_address(name)
            .map_err(|e| e.to_string())
    }));
    match result {
        Ok(Ok(address)) => address as *mut c_void,
        Ok(Err(error)) => {
            set_error(error_out, &error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error(error_out, "CPU JIT lookup panicked");
            ptr::null_mut()
        }
    }
}

/// Stable tagged bits representation for checked dynamic calls.
/// Tags: void=0, I8=1, U8=2, I16=3, U16=4, I32=5, U32=6, I64=7,
/// U64=8, usize=9, bool=10, F32=11, F64=12, pointer=13.
/// Signed integers carry their low-width two's-complement bits; floats carry
/// IEEE bits. `reserved` must be zero. Results use the same representation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct YCpuJitValue {
    pub kind: u32,
    pub reserved: u32,
    pub bits: u64,
}

/// Return a function signature as JSON. Free with `y_free_string`.
///
/// # Safety
/// Same handle/name/error pointer requirements as `y_cpu_jit_function`.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_signature(
    handle: *mut c_void,
    name: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || name.is_null() {
            return Err("CPU JIT handle/name pointer is null".to_string());
        }
        let name = CStr::from_ptr(name)
            .to_str()
            .map_err(|_| "CPU JIT function name is not UTF-8".to_string())?;
        let signature = (&*handle.cast::<CpuJit>())
            .function_signature(name)
            .map_err(|e| e.to_string())?;
        CString::new(signature.to_json()).map_err(|_| "CPU JIT signature contains NUL".to_string())
    }));
    match result {
        Ok(Ok(json)) => json.into_raw(),
        Ok(Err(error)) => {
            set_error(error_out, &error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error(error_out, "CPU JIT signature lookup panicked");
            ptr::null_mut()
        }
    }
}

/// Invoke after checking argument count/types against the source signature.
/// Returns 0 on success, -1 on error. Failure leaves `result_out` unchanged.
///
/// # Safety
/// Handle/name/error must be valid as above. `arguments` must contain
/// `argument_count` readable values (null is allowed for zero values), and
/// `result_out` must be writable. Code must be trusted; pointer arguments
/// must satisfy the called Y function's memory requirements.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_call(
    handle: *mut c_void,
    name: *const c_char,
    arguments: *const YCpuJitValue,
    argument_count: usize,
    result_out: *mut YCpuJitValue,
    error_out: *mut *mut c_char,
) -> i32 {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || name.is_null() || result_out.is_null() {
            return Err("CPU JIT handle/name/result pointer is null".to_string());
        }
        if argument_count > isize::MAX as usize / std::mem::size_of::<YCpuJitValue>() {
            return Err("CPU JIT argument array is too large".to_string());
        }
        if arguments.is_null() && argument_count != 0 {
            return Err("CPU JIT argument pointer is null".to_string());
        }
        let name = CStr::from_ptr(name)
            .to_str()
            .map_err(|_| "CPU JIT function name is not UTF-8".to_string())?;
        let arguments = if argument_count == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(arguments, argument_count)
        };
        let values = arguments
            .iter()
            .map(|value| {
                if value.reserved != 0 {
                    return Err("CPU JIT value reserved field must be zero".to_string());
                }
                JitValue::from_tagged_bits(value.kind, value.bits).map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let value = (&*handle.cast::<CpuJit>())
            .call(name, &values)
            .map_err(|e| e.to_string())?;
        Ok(YCpuJitValue {
            kind: value.abi_type().tag().unwrap(),
            reserved: 0,
            bits: value.bits(),
        })
    }));
    match result {
        Ok(Ok(value)) => {
            ptr::write(result_out, value);
            0
        }
        Ok(Err(error)) => {
            set_error(error_out, &error);
            -1
        }
        Err(_) => {
            set_error(error_out, "CPU JIT call panicked");
            -1
        }
    }
}

/// Release executable code. Null is accepted.
///
/// # Safety
/// `handle` must be a handle returned by a CPU JIT compile function, used on its
/// creating thread and freed exactly once after all native calls finish.
#[no_mangle]
pub unsafe extern "C" fn y_cpu_jit_free(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle.cast::<CpuJit>()));
    }
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::c_api::y_free_string;

    unsafe fn owned_string(value: *mut c_char) -> String {
        assert!(!value.is_null());
        let text = CStr::from_ptr(value).to_string_lossy().into_owned();
        y_free_string(value);
        text
    }

    #[test]
    fn options_ffi_initializes_defaults_and_checks_header_before_fields() {
        unsafe {
            let mut error = ptr::null_mut();
            let mut options = YCpuJitOptions {
                abi_version: 91,
                struct_size: 0,
                opt_level: 99,
                training_opt_level: 99,
                codegen_opt_level: 99,
            };
            let before = options;
            assert_eq!(y_cpu_jit_options_init(&mut options, 8, &mut error), -1);
            assert!(owned_string(error).contains("size"));
            assert_eq!(options, before);
            assert_eq!(y_cpu_jit_options_init(ptr::null_mut(), 20, &mut error), -1);
            assert!(owned_string(error).contains("pointer is null"));
            error = std::ptr::dangling_mut::<c_char>();
            assert_eq!(
                y_cpu_jit_options_init(
                    &mut options,
                    std::mem::size_of::<YCpuJitOptions>() as u32,
                    &mut error,
                ),
                0
            );
            assert!(error.is_null());
            assert_eq!(options, YCpuJitOptions::default());
            assert_eq!(std::mem::size_of::<YCpuJitOptions>(), 20);
            assert_eq!(std::mem::offset_of!(YCpuJitOptions, opt_level), 8);
            assert_eq!(std::mem::offset_of!(YCpuJitOptions, training_opt_level), 12);
            assert_eq!(std::mem::offset_of!(YCpuJitOptions, codegen_opt_level), 16);
            for (prefix, expected) in [([99_u32, 8], "version"), ([1, 8], "size")] {
                assert!(y_cpu_jit_compile_with_options(
                    c"fn broken(".as_ptr(),
                    prefix.as_ptr().cast(),
                    &mut error,
                )
                .is_null());
                assert!(owned_string(error).contains(expected));
            }
            let decoded = decode_options(ptr::null()).unwrap();
            assert_eq!(decoded.opt_level, 3);
            assert_eq!(decoded.training_opt_level, None);
            assert_eq!(decoded.codegen_opt_level, None);
            assert!(decoded.verify_each_pass);
        }
    }

    #[test]
    fn options_ffi_rejects_invalid_levels_in_every_compilation_mode() {
        unsafe {
            let mut cases = Vec::new();
            for level in [4, 256, u32::MAX] {
                cases.push((
                    YCpuJitOptions {
                        opt_level: level,
                        ..Default::default()
                    },
                    "opt_level",
                ));
            }
            for level in [-2, 4, 256, i32::MIN, i32::MAX] {
                cases.push((
                    YCpuJitOptions {
                        training_opt_level: level,
                        ..Default::default()
                    },
                    "training_opt_level",
                ));
                cases.push((
                    YCpuJitOptions {
                        codegen_opt_level: level,
                        ..Default::default()
                    },
                    "codegen_opt_level",
                ));
            }
            for (options, expected) in cases {
                for mode in 0..3 {
                    let mut error = ptr::null_mut();
                    let source = c"fn broken(".as_ptr();
                    let handle = match mode {
                        0 => y_cpu_jit_compile_with_options(source, &options, &mut error),
                        1 => y_cpu_jit_compile_instrumented_with_options(
                            source, &options, &mut error,
                        ),
                        _ => y_cpu_jit_compile_profiled_with_options(
                            source,
                            &options,
                            ptr::null_mut(),
                            &mut error,
                        ),
                    };
                    assert!(handle.is_null());
                    assert!(
                        owned_string(error).contains(expected),
                        "mode {mode}: {expected}"
                    );
                }
            }
            let options = YCpuJitOptions {
                training_opt_level: 4,
                ..Default::default()
            };
            assert!(y_cpu_jit_compile_with_options(
                c"fn value() {}".as_ptr(),
                &options,
                ptr::null_mut()
            )
            .is_null());
        }
    }

    #[test]
    fn options_ffi_matches_rust_training_and_final_ir_and_native_outputs() {
        const SOURCE: &CStr = c"fn choose(x: I64) -> I64 { let mut y: I64 = x + 2; if x < 0 { y = x - 1; } return y; }";
        unsafe {
            let mut error = ptr::null_mut();
            let inherited =
                y_cpu_jit_compile_with_options(SOURCE.as_ptr(), ptr::null(), &mut error);
            assert!(!inherited.is_null());
            assert!(error.is_null());
            let ordinary_reference = CpuJit::compile(SOURCE.to_str().unwrap()).unwrap();
            assert_eq!(
                (&*inherited.cast::<CpuJit>()).optimized_ir(),
                ordinary_reference.optimized_ir()
            );
            y_cpu_jit_free(inherited);

            for tier in 0..=3 {
                let options = YCpuJitOptions {
                    training_opt_level: tier,
                    codegen_opt_level: 3 - tier,
                    ..Default::default()
                };
                let rust_options = decode_options(&options).unwrap();
                assert_eq!(rust_options.training_opt_level, Some(tier as u8));
                assert_eq!(rust_options.codegen_opt_level, Some((3 - tier) as u8));
                assert!(rust_options.verify_each_pass);
                let ordinary =
                    y_cpu_jit_compile_with_options(SOURCE.as_ptr(), &options, &mut error);
                assert!(!ordinary.is_null());
                assert!(error.is_null());
                assert_eq!(
                    (&*ordinary.cast::<CpuJit>()).optimized_ir(),
                    ordinary_reference.optimized_ir(),
                    "training override cannot change ordinary IR"
                );
                y_cpu_jit_free(ordinary);

                let training = y_cpu_jit_compile_instrumented_with_options(
                    SOURCE.as_ptr(),
                    &options,
                    &mut error,
                );
                assert!(!training.is_null());
                assert!(error.is_null());
                let reference =
                    CpuJit::compile_instrumented(SOURCE.to_str().unwrap(), rust_options).unwrap();
                let training_jit = &*training.cast::<CpuJit>();
                assert_eq!(
                    training_jit.optimized_ir(),
                    reference.optimized_ir(),
                    "actual training IR O{tier}"
                );
                assert_eq!(training_jit.compile_timings().verification_checks, 2);
                assert_eq!(
                    training_jit.branch_profile().unwrap().total_observations(),
                    0
                );
                let choose: unsafe extern "C" fn(i64) -> i64 = std::mem::transmute(
                    y_cpu_jit_function(training, c"choose".as_ptr(), &mut error),
                );
                for x in [-3, -1, 0, 2, 5] {
                    assert_eq!(choose(x), if x < 0 { x - 1 } else { x + 2 });
                }
                let profile = training_jit.branch_profile().unwrap();
                assert_eq!(profile.total_observations(), 5);
                assert_eq!(
                    (
                        profile.sites()[0].true_count,
                        profile.sites()[0].false_count
                    ),
                    (2, 3)
                );

                // Final codegen can be selected independently of the trainer.
                let final_options = YCpuJitOptions {
                    codegen_opt_level: 2,
                    ..options
                };
                let optimized = y_cpu_jit_compile_profiled_with_options(
                    SOURCE.as_ptr(),
                    &final_options,
                    training,
                    &mut error,
                );
                assert!(!optimized.is_null());
                assert!(error.is_null());
                let final_reference = CpuJit::compile_with_profile(
                    SOURCE.to_str().unwrap(),
                    decode_options(&final_options).unwrap(),
                    &profile,
                )
                .unwrap();
                let final_jit = &*optimized.cast::<CpuJit>();
                assert_eq!(
                    final_jit.optimized_ir(),
                    final_reference.optimized_ir(),
                    "final IR uses base O3, not training O{tier}"
                );
                assert!(!final_jit.optimized_ir().contains("atomicrmw"));
                assert_eq!(
                    training_jit.branch_profile().unwrap(),
                    profile,
                    "compilation does not execute source"
                );
                let final_choose: unsafe extern "C" fn(i64) -> i64 = std::mem::transmute(
                    y_cpu_jit_function(optimized, c"choose".as_ptr(), &mut error),
                );
                for x in [-50, -1, 0, 17] {
                    assert_eq!(final_choose(x), if x < 0 { x - 1 } else { x + 2 });
                }
                assert_eq!(choose(-9), -10, "original native address stays live");
                y_cpu_jit_free(training);
                assert_eq!(
                    final_choose(9),
                    11,
                    "final session owns its code independently"
                );
                y_cpu_jit_free(optimized);
            }
        }
    }

    #[test]
    fn profile_json_escapes_names_and_controls() {
        assert_eq!(
            json_quote("quoted\" \\ \n\t\0 È"),
            "\"quoted\\\" \\\\ \\u000a\\u0009\\u0000 È\""
        );
    }

    #[test]
    fn profile_ffi_records_explicit_calls_and_owns_each_native_session() {
        const SOURCE: &CStr =
            c"fn choose(x: I64) -> I64 { if x < 0 { return x - 1; } return x + 2; }";
        for opt_level in [0, 3] {
            unsafe {
                let mut error = ptr::null_mut();
                let training =
                    y_cpu_jit_compile_instrumented(SOURCE.as_ptr(), opt_level, &mut error);
                assert!(
                    !training.is_null(),
                    "{}",
                    if error.is_null() {
                        String::new()
                    } else {
                        owned_string(error)
                    }
                );
                assert!(error.is_null());
                let empty = owned_string(y_cpu_jit_branch_profile(training, &mut error));
                assert!(empty.contains("\"total_observations\":0"), "{empty}");
                let fingerprint = empty
                    .split("\"fingerprint\":\"")
                    .nth(1)
                    .unwrap()
                    .split('"')
                    .next()
                    .unwrap();
                assert_eq!(fingerprint.len(), 64);
                assert!(fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit()));
                let choose: unsafe extern "C" fn(i64) -> i64 = std::mem::transmute(
                    y_cpu_jit_function(training, c"choose".as_ptr(), &mut error),
                );
                for x in [-3, -1, 0, 2, 5] {
                    assert_eq!(choose(x), if x < 0 { x - 1 } else { x + 2 });
                }
                let measured = owned_string(y_cpu_jit_branch_profile(training, &mut error));
                assert!(measured.contains("\"function\":\"choose\""), "{measured}");
                assert!(
                    measured.contains("\"true_count\":2,\"false_count\":3"),
                    "{measured}"
                );
                assert!(measured.contains("\"total_observations\":5"), "{measured}");
                // This valid operation must clear a previous error-out value.
                error = std::ptr::dangling_mut::<c_char>();
                let optimized =
                    y_cpu_jit_compile_profiled(SOURCE.as_ptr(), opt_level, training, &mut error);
                assert!(!optimized.is_null());
                assert!(error.is_null());
                let after = owned_string(y_cpu_jit_branch_profile(training, &mut error));
                assert_eq!(
                    after, measured,
                    "profile-use compilation must not execute source"
                );
                let optimized_fn: unsafe extern "C" fn(i64) -> i64 = std::mem::transmute(
                    y_cpu_jit_function(optimized, c"choose".as_ptr(), &mut error),
                );
                for x in [-50, -1, 0, 17] {
                    assert_eq!(optimized_fn(x), if x < 0 { x - 1 } else { x + 2 });
                }
                assert_eq!(choose(-9), -10, "recompilation preserves old addresses");
                let newer = owned_string(y_cpu_jit_branch_profile(training, &mut error));
                assert!(
                    newer.contains("\"true_count\":3,\"false_count\":3"),
                    "{newer}"
                );
                assert_eq!(after, measured, "returned JSON is independently owned");
                assert!(y_cpu_jit_branch_profile(optimized, &mut error).is_null());
                assert!(owned_string(error).contains("instrumentation"));
                y_cpu_jit_free(training);
                assert_eq!(
                    optimized_fn(-100),
                    -101,
                    "optimized session survives training session drop"
                );
                y_cpu_jit_free(optimized);
            }
        }
    }

    #[test]
    fn profile_ffi_rejects_nulls_stale_source_and_nontraining_sessions() {
        unsafe {
            let source = c"fn choose(x: I64) -> I64 { if x < 0 { return x - 1; } return x + 2; }";
            let mut error = ptr::null_mut();
            assert!(y_cpu_jit_compile_instrumented(ptr::null(), 3, &mut error).is_null());
            assert!(owned_string(error).contains("source pointer is null"));
            assert!(y_cpu_jit_compile_instrumented(source.as_ptr(), 4, &mut error).is_null());
            assert!(owned_string(error).contains("opt_level"));
            assert!(y_cpu_jit_compile_instrumented(c"\xff".as_ptr(), 3, &mut error).is_null());
            assert!(owned_string(error).contains("UTF-8"));
            assert!(y_cpu_jit_branch_profile(ptr::null_mut(), &mut error).is_null());
            assert!(owned_string(error).contains("handle pointer is null"));
            assert!(
                y_cpu_jit_compile_profiled(source.as_ptr(), 3, ptr::null_mut(), &mut error)
                    .is_null()
            );
            assert!(owned_string(error).contains("training handle pointer is null"));

            let training = y_cpu_jit_compile_instrumented(source.as_ptr(), 3, &mut error);
            assert!(!training.is_null());
            let stale = c"fn choose(x: I64) -> I64 { if x < 0 { return x - 1; } return x + 7; }";
            assert!(y_cpu_jit_compile_profiled(stale.as_ptr(), 3, training, &mut error).is_null());
            assert!(owned_string(error).contains("profile does not match"));
            let regular = y_cpu_jit_compile(source.as_ptr(), 3, &mut error);
            assert!(!regular.is_null());
            assert!(error.is_null());
            assert!(y_cpu_jit_compile_profiled(source.as_ptr(), 3, regular, &mut error).is_null());
            assert!(owned_string(error).contains("instrumentation"));
            // Errors are optional; failures still return null without unwinding.
            assert!(y_cpu_jit_branch_profile(regular, ptr::null_mut()).is_null());
            let profile = owned_string(y_cpu_jit_branch_profile(training, &mut error));
            assert!(error.is_null());
            assert!(profile.contains("\"total_observations\":0"));
            y_cpu_jit_free(regular);
            y_cpu_jit_free(training);
        }
    }

    #[test]
    fn dynamic_ffi_checks_metadata_tags_and_arity_before_executing() {
        unsafe {
            let mut error = ptr::null_mut();
            let handle = y_cpu_jit_compile(c"@unsafe fn write_value(p: GlobalMemory<I64>, value: I64) { p[0] = value; } fn mixed(a: I64, b: F64) -> F64 { if a < 0 { return b - 3.0; } return b; }".as_ptr(), 3, &mut error);
            assert!(
                !handle.is_null(),
                "{}",
                if error.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(error).to_string_lossy().into_owned()
                }
            );
            let json = y_cpu_jit_signature(handle, c"mixed".as_ptr(), &mut error);
            assert!(!json.is_null());
            assert_eq!(CStr::from_ptr(json).to_str().unwrap(), "{\"name\":\"mixed\",\"parameters\":[\"I64\",\"F64\"],\"return_type\":\"F64\",\"dynamic_call\":true}");
            y_free_string(json);
            let arguments = [
                YCpuJitValue {
                    kind: 7,
                    reserved: 0,
                    bits: (-3_i64) as u64,
                },
                YCpuJitValue {
                    kind: 12,
                    reserved: 0,
                    bits: 1.25_f64.to_bits(),
                },
            ];
            let mut result = YCpuJitValue::default();
            assert_eq!(
                y_cpu_jit_call(
                    handle,
                    c"mixed".as_ptr(),
                    arguments.as_ptr(),
                    2,
                    &mut result,
                    &mut error
                ),
                0
            );
            assert!(error.is_null());
            assert_eq!(result.kind, 12);
            assert_eq!(f64::from_bits(result.bits), -1.75);
            let mut storage = 73_i64;
            let valid = [
                YCpuJitValue {
                    kind: 13,
                    reserved: 0,
                    bits: (&mut storage as *mut i64) as u64,
                },
                YCpuJitValue {
                    kind: 7,
                    reserved: 0,
                    bits: 4294967296,
                },
            ];
            for (kind, reserved, bits) in [(8, 0, 99), (7, 1, 99), (99, 0, 99), (10, 0, 2)] {
                let invalid = [
                    valid[0],
                    YCpuJitValue {
                        kind,
                        reserved,
                        bits,
                    },
                ];
                result = YCpuJitValue {
                    kind: 42,
                    reserved: 0,
                    bits: 100,
                };
                assert_eq!(
                    y_cpu_jit_call(
                        handle,
                        c"write_value".as_ptr(),
                        invalid.as_ptr(),
                        2,
                        &mut result,
                        &mut error
                    ),
                    -1
                );
                assert!(!error.is_null());
                assert_eq!(storage, 73);
                assert_eq!((result.kind, result.bits), (42, 100));
                y_free_string(error);
            }
            assert_eq!(
                y_cpu_jit_call(
                    handle,
                    c"write_value".as_ptr(),
                    ptr::null(),
                    1,
                    &mut result,
                    &mut error
                ),
                -1
            );
            y_free_string(error);
            assert_eq!(
                y_cpu_jit_call(
                    handle,
                    c"write_value".as_ptr(),
                    valid.as_ptr(),
                    2,
                    &mut result,
                    &mut error
                ),
                0
            );
            assert!(error.is_null());
            assert_eq!(storage, 4294967296);
            assert_eq!(result.kind, 0);
            y_cpu_jit_free(handle);
        }
    }

    #[test]
    fn ffi_calls_native_code_and_reports_errors() {
        unsafe {
            let mut error = ptr::null_mut();
            let handle = y_cpu_jit_compile(
                c"fn square(x: I64) -> I64 { return x * x; }".as_ptr(),
                3,
                &mut error,
            );
            assert!(
                !handle.is_null(),
                "{}",
                if error.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(error).to_string_lossy().into_owned()
                }
            );
            assert!(error.is_null());
            let address = y_cpu_jit_function(handle, c"square".as_ptr(), &mut error);
            let function: unsafe extern "C" fn(i64) -> i64 = std::mem::transmute(address);
            assert_eq!(function(123456), 15241383936);
            assert!(y_cpu_jit_function(handle, c"missing".as_ptr(), &mut error).is_null());
            assert!(CStr::from_ptr(error)
                .to_bytes()
                .windows(7)
                .any(|w| w == b"missing"));
            y_free_string(error);
            y_cpu_jit_free(handle);
            assert!(y_cpu_jit_compile(ptr::null(), 3, &mut error).is_null());
            assert!(!error.is_null());
            y_free_string(error);
            assert!(y_cpu_jit_compile(c"fn a() {}".as_ptr(), 256, &mut error).is_null());
            y_free_string(error);
            y_cpu_jit_free(ptr::null_mut());
        }
    }
}
