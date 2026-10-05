//! C ABI for adaptive GEMM in an application's existing CUDA context.
//!
//! See `c_src/y_adaptive_jit.h` for the ABI and lifetime contract. Handles are
//! confined to their creating thread. Returned strings use `y_free_string`.

use crate::adaptive_jit::TuningPolicy;
use crate::adaptive_jit::{AdaptiveGemm, AdaptiveJitConfig, GemmShape, JitTier, KernelStats};
use crate::cuda_runtime::CudaContext;
use std::ffi::{c_char, CStr, CString};
use std::fmt::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr;
use std::thread::{self, ThreadId};

pub const ADAPTIVE_JIT_ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct YAdaptiveJitConfig {
    pub abi_version: u32,
    pub struct_size: u32,
    pub tuning_policy: u32,
    pub max_cached_shapes: u32,
    pub max_candidates: u32,
    pub max_disk_cache_entries: u32,
    pub hot_threshold: u64,
    pub min_improvement: f64,
    pub cache_dir: *const c_char,
}

impl Default for YAdaptiveJitConfig {
    fn default() -> Self {
        let config = AdaptiveJitConfig::default();
        Self {
            abi_version: ADAPTIVE_JIT_ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            tuning_policy: 0,
            max_cached_shapes: config.max_cached_shapes as u32,
            max_candidates: config.max_candidates as u32,
            max_disk_cache_entries: config.max_disk_cache_entries as u32,
            hot_threshold: config.hot_threshold,
            min_improvement: config.min_improvement,
            cache_dir: ptr::null(),
        }
    }
}

/// Opaque to C. Callers must not read, copy, or allocate this structure.
pub struct YAdaptiveJit {
    owner: ThreadId,
    poisoned: bool,
    runtime: AdaptiveGemm<'static>,
}

unsafe fn decode_config(config: *const YAdaptiveJitConfig) -> Result<AdaptiveJitConfig, String> {
    if config.is_null() {
        return Ok(AdaptiveJitConfig::default());
    }
    // Read only the common two-u32 prefix until version and size are checked.
    let header = config.cast::<u32>();
    if *header != ADAPTIVE_JIT_ABI_VERSION {
        return Err("unsupported adaptive JIT config ABI version".into());
    }
    if *header.add(1) != std::mem::size_of::<YAdaptiveJitConfig>() as u32 {
        return Err("adaptive JIT config size does not match this library".into());
    }
    let config = &*config;
    let tuning_policy = match config.tuning_policy {
        0 => TuningPolicy::Deferred,
        1 => TuningPolicy::OnLaunch,
        2 => TuningPolicy::Disabled,
        _ => return Err("unknown adaptive JIT tuning policy".into()),
    };
    let cache_dir = if config.cache_dir.is_null() {
        None
    } else {
        let path = CStr::from_ptr(config.cache_dir)
            .to_str()
            .map_err(|_| "cache_dir must be valid UTF-8")?;
        if path.is_empty() {
            return Err("cache_dir must be nonempty or null".into());
        }
        Some(PathBuf::from(path))
    };
    let result = AdaptiveJitConfig {
        tuning_policy,
        cache_dir,
        max_disk_cache_entries: config.max_disk_cache_entries as usize,
        hot_threshold: config.hot_threshold,
        max_cached_shapes: config.max_cached_shapes as usize,
        max_candidates: config.max_candidates as usize,
        min_improvement: config.min_improvement,
    };
    result.validate()?;
    Ok(result)
}

unsafe fn boundary<T>(
    error_out: *mut *mut c_char,
    failure: T,
    f: impl FnOnce() -> Result<T, String>,
) -> T {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
    let outcome = catch_unwind(AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err("internal adaptive JIT panic".into()));
    match outcome {
        Ok(value) => value,
        Err(error) => {
            if !error_out.is_null() {
                // Driver diagnostics may contain arbitrary text. CString errors
                // must not themselves unwind across the C boundary.
                *error_out = CString::new(error.replace('\0', "\\0")).unwrap().into_raw();
            }
            failure
        }
    }
}

unsafe fn with_handle<T>(
    handle: *mut YAdaptiveJit,
    f: impl FnOnce(&mut AdaptiveGemm<'static>) -> Result<T, String>,
) -> Result<T, String> {
    if handle.is_null() {
        return Err("adaptive JIT handle is null".into());
    }
    let handle = &mut *handle;
    if handle.owner != thread::current().id() {
        return Err("adaptive JIT handle must be used on its creating thread".into());
    }
    if handle.poisoned {
        return Err("adaptive JIT handle is poisoned after a panic; close it".into());
    }
    handle.runtime.require_current()?;
    match catch_unwind(AssertUnwindSafe(|| f(&mut handle.runtime))) {
        Ok(result) => result,
        Err(_) => {
            handle.poisoned = true;
            Err("internal adaptive JIT panic; handle is poisoned and must be closed".into())
        }
    }
}

/// Initialize a caller-allocated configuration with defaults.
///
/// # Safety
/// `out` must point to writable storage of `config_size` bytes with the config's
/// alignment. A nonnull `error_out` must point to a writable pointer slot.
#[no_mangle]
pub unsafe extern "C" fn y_adaptive_jit_config_init(
    out: *mut YAdaptiveJitConfig,
    config_size: u32,
    error_out: *mut *mut c_char,
) -> i32 {
    boundary(error_out, -1, || {
        if out.is_null() || config_size != std::mem::size_of::<YAdaptiveJitConfig>() as u32 {
            return Err("config_init requires nonnull output and the exact config size".into());
        }
        out.write(YAdaptiveJitConfig::default());
        Ok(0)
    })
}

/// Borrow the calling thread's current CUDA context; null config uses defaults.
///
/// # Safety
/// The host must keep its CUDA context alive and current on this thread through
/// destruction. Nonnull config and strings must be valid for this call; the
/// config must contain at least its two-u32 prefix and its claimed size.
/// A nonnull error_out must be writable. Never call concurrently on a handle.
#[no_mangle]
pub unsafe extern "C" fn y_adaptive_jit_create_current(
    config: *const YAdaptiveJitConfig,
    error_out: *mut *mut c_char,
) -> *mut YAdaptiveJit {
    boundary(error_out, ptr::null_mut(), || {
        let config = decode_config(config)?;
        let ctx = CudaContext::borrow_current()?;
        let runtime = AdaptiveGemm::from_owned_context(ctx, config)?;
        Ok(Box::into_raw(Box::new(YAdaptiveJit {
            owner: thread::current().id(),
            poisoned: false,
            runtime,
        })))
    })
}

/// Synchronize and close; null is a no-op. Failure leaves the handle live.
/// Does not destroy the host's CUDA context. A poisoned handle can be closed.
///
/// # Safety
/// Nonnull handle must be a live pointer from create_current, exclusively
/// accessed here. Its original context must remain alive. On success the pointer
/// becomes invalid. A nonnull error_out must be writable and disjoint from it.
#[no_mangle]
pub unsafe extern "C" fn y_adaptive_jit_destroy(
    handle: *mut YAdaptiveJit,
    error_out: *mut *mut c_char,
) -> i32 {
    boundary(error_out, -1, || {
        if handle.is_null() {
            return Ok(0);
        }
        if (*handle).owner != thread::current().id() {
            return Err("adaptive JIT handle must be closed on its creating thread".into());
        }
        (*handle).runtime.synchronize()?;
        drop(Box::from_raw(handle));
        Ok(0)
    })
}

/// Prepare a kernel without accessing application buffers.
///
/// # Safety
/// Handle must be live and exclusively accessed; keep its host context alive.
/// A nonnull error_out must be a writable, disjoint pointer slot.
#[no_mangle]
pub unsafe extern "C" fn y_adaptive_jit_prepare(
    handle: *mut YAdaptiveJit,
    m: u32,
    n: u32,
    k: u32,
    error_out: *mut *mut c_char,
) -> i32 {
    boundary(error_out, -1, || {
        with_handle(handle, |runtime| {
            runtime.prepare(GemmShape { m, n, k })?;
            Ok(0)
        })
    })
}

/// Enqueue row-major F16 A * F16 B -> F32 C on the default CUDA stream.
///
/// # Safety
/// Besides the prepare handle/error contract, a/b/c must be live 16-byte-aligned
/// device allocations in the host context containing M*K F16, K*N F16 and M*N
/// F32 elements. C must not overlap A/B. Retain allocations until completion and
/// coordinate conflicting accesses from other streams and threads.
#[no_mangle]
pub unsafe extern "C" fn y_adaptive_jit_launch(
    handle: *mut YAdaptiveJit,
    m: u32,
    n: u32,
    k: u32,
    a: u64,
    b: u64,
    c: u64,
    error_out: *mut *mut c_char,
) -> i32 {
    boundary(error_out, -1, || {
        with_handle(handle, |runtime| {
            runtime.launch(GemmShape { m, n, k }, a, b, c)?;
            Ok(0)
        })
    })
}

/// Wait for work in the borrowed CUDA context.
///
/// # Safety
/// Same handle/error contract as prepare.
#[no_mangle]
pub unsafe extern "C" fn y_adaptive_jit_synchronize(
    handle: *mut YAdaptiveJit,
    error_out: *mut *mut c_char,
) -> i32 {
    boundary(error_out, -1, || {
        with_handle(handle, |runtime| {
            runtime.synchronize()?;
            Ok(0)
        })
    })
}

fn json_string(value: &str) -> String {
    let mut result = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            ch if ch <= '\u{1f}' => {
                write!(result, "\\u{:04x}", ch as u32).unwrap();
            }
            ch => result.push(ch),
        }
    }
    result.push('"');
    result
}

fn optional_text(value: &Option<String>) -> String {
    value
        .as_deref()
        .map(json_string)
        .unwrap_or_else(|| "null".into())
}

fn optional_number(value: Option<f64>) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".into())
}

fn stats_json(stats: &KernelStats) -> String {
    let tier = match stats.tier {
        JitTier::Baseline => "baseline",
        JitTier::Tuned => "tuned",
        JitTier::RetainedBaseline => "retained_baseline",
        JitTier::TuningFailed => "tuning_failed",
        JitTier::RejectedBaseline => "rejected_baseline",
    };
    format!(
        concat!(
            "{{\"launches\":{},\"tier\":\"{}\",\"tuning_attempts\":{},",
            "\"tuning_time_seconds\":{},\"candidates_measured\":{},\"baseline_us\":{},",
            "\"selected_us\":{},\"tuning_error\":{},\"cache_hit\":{},\"cache_error\":{},",
            "\"estimated_break_even_launches\":{}}}"
        ),
        stats.launches,
        tier,
        stats.tuning_attempts,
        stats.tuning_time.as_secs_f64(),
        stats.candidates_measured,
        optional_number(stats.baseline_us),
        optional_number(stats.selected_us),
        optional_text(&stats.tuning_error),
        stats.cache_hit,
        optional_text(&stats.cache_error),
        stats
            .estimated_break_even_launches()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "null".into())
    )
}

fn string_result(value: String) -> Result<*mut c_char, String> {
    CString::new(value)
        .map(CString::into_raw)
        .map_err(|_| "invalid NUL in JSON output".into())
}

/// Return resident statistics as allocated JSON; a missing shape is an error.
///
/// # Safety
/// Same handle/error contract as prepare. Free the result with y_free_string.
#[no_mangle]
pub unsafe extern "C" fn y_adaptive_jit_stats_json(
    handle: *mut YAdaptiveJit,
    m: u32,
    n: u32,
    k: u32,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    boundary(error_out, ptr::null_mut(), || {
        with_handle(handle, |runtime| {
            let shape = GemmShape { m, n, k };
            shape.validate()?;
            string_result(stats_json(
                runtime
                    .stats(shape)
                    .ok_or("shape is not resident in adaptive JIT cache")?,
            ))
        })
    })
}

/// Return pending shapes as allocated JSON, e.g. [[256,256,256]].
///
/// # Safety
/// Same handle/error contract as prepare. Free the result with y_free_string.
#[no_mangle]
pub unsafe extern "C" fn y_adaptive_jit_pending_json(
    handle: *mut YAdaptiveJit,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    boundary(error_out, ptr::null_mut(), || {
        with_handle(handle, |runtime| {
            string_result(format!(
                "[{}]",
                runtime
                    .pending_shapes()
                    .iter()
                    .map(|s| format!("[{},{},{}]", s.m, s.n, s.k))
                    .collect::<Vec<_>>()
                    .join(",")
            ))
        })
    })
}

/// Tune up to max_shapes synchronously and return allocated JSON reports.
/// Per-shape tuning failures appear in stats; the call itself can still succeed.
///
/// # Safety
/// Same handle/error contract as prepare. Free the result with y_free_string.
#[no_mangle]
pub unsafe extern "C" fn y_adaptive_jit_tune_hot_json(
    handle: *mut YAdaptiveJit,
    max_shapes: u32,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    boundary(error_out, ptr::null_mut(), || {
        with_handle(handle, |runtime| {
            let reports = runtime.tune_hot(max_shapes as usize)?;
            string_result(format!(
                "[{}]",
                reports
                    .iter()
                    .map(|r| format!(
                        "{{\"shape\":[{},{},{}],\"stats\":{}}}",
                        r.shape.m,
                        r.shape.n,
                        r.shape.k,
                        stats_json(&r.stats)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            ))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_api::y_free_string;

    unsafe fn take_error(error: *mut c_char) -> String {
        assert!(!error.is_null());
        let result = CStr::from_ptr(error).to_str().unwrap().to_owned();
        y_free_string(error);
        result
    }

    #[test]
    fn rejects_config_versions_sizes_and_invalid_values_before_cuda() {
        unsafe {
            let mut error = ptr::null_mut();
            for config in [
                YAdaptiveJitConfig {
                    abi_version: 2,
                    ..Default::default()
                },
                YAdaptiveJitConfig {
                    struct_size: 8,
                    ..Default::default()
                },
                YAdaptiveJitConfig {
                    tuning_policy: 99,
                    ..Default::default()
                },
                YAdaptiveJitConfig {
                    max_candidates: 1,
                    ..Default::default()
                },
                YAdaptiveJitConfig {
                    min_improvement: f64::NAN,
                    ..Default::default()
                },
                YAdaptiveJitConfig {
                    hot_threshold: 0,
                    ..Default::default()
                },
                YAdaptiveJitConfig {
                    cache_dir: c"".as_ptr(),
                    ..Default::default()
                },
            ] {
                assert!(y_adaptive_jit_create_current(&config, &mut error).is_null());
                assert!(!take_error(error).is_empty());
            }
            // A short prefix really is sufficient for version/size rejection.
            let prefix = [1u32, 8];
            assert!(y_adaptive_jit_create_current(prefix.as_ptr().cast(), &mut error).is_null());
            assert!(take_error(error).contains("size"));
        }
    }

    #[test]
    fn config_init_and_null_handle_errors_obey_ownership_contract() {
        unsafe {
            let mut error = ptr::null_mut();
            let mut config = YAdaptiveJitConfig::default();
            config.hot_threshold = 123;
            assert_eq!(y_adaptive_jit_config_init(&mut config, 8, &mut error), -1);
            assert_eq!(config.hot_threshold, 123);
            take_error(error);
            assert_eq!(
                y_adaptive_jit_config_init(
                    &mut config,
                    std::mem::size_of_val(&config) as u32,
                    &mut error
                ),
                0
            );
            assert!(error.is_null());
            assert_eq!(config.hot_threshold, 32);
            assert_eq!(
                decode_config(&config).unwrap().tuning_policy,
                TuningPolicy::Deferred
            );
            assert_eq!(
                y_adaptive_jit_prepare(ptr::null_mut(), 16, 16, 16, &mut error),
                -1
            );
            assert!(take_error(error).contains("null"));
            assert!(y_adaptive_jit_pending_json(ptr::null_mut(), ptr::null_mut()).is_null());
            assert_eq!(y_adaptive_jit_destroy(ptr::null_mut(), &mut error), 0);
            assert!(error.is_null());
        }
    }

    #[test]
    fn boundary_catches_panics_and_sanitizes_error_strings() {
        unsafe {
            let mut error = ptr::null_mut();
            assert_eq!(
                boundary(&mut error, -1, || -> Result<i32, String> {
                    panic!("test panic")
                }),
                -1
            );
            assert!(take_error(error).contains("panic"));
            assert_eq!(
                boundary(&mut error, -1, || Err("nul\0diagnostic".into())),
                -1
            );
            assert_eq!(take_error(error), "nul\\0diagnostic");
        }
    }

    #[test]
    fn json_escapes_all_control_characters_and_nonfinite_numbers() {
        assert_eq!(
            json_string("a\"\\\n\r\t\0\u{1f}é"),
            "\"a\\\"\\\\\\n\\r\\t\\u0000\\u001fé\""
        );
        assert_eq!(optional_number(Some(f64::INFINITY)), "null");
        assert_eq!(optional_number(Some(f64::NAN)), "null");
        assert_eq!(optional_number(Some(1.25)), "1.25");
    }
}
