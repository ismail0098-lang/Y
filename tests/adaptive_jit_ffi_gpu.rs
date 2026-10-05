//! Real-device coverage of the C ABI and borrowed-context ownership contract.
//! Run serially: cargo test --test adaptive_jit_ffi_gpu -- --ignored --test-threads=1

use std::ffi::{c_char, CStr};
use std::mem::{size_of, MaybeUninit};
use std::ptr;

use y::adaptive_jit_ffi::*;
use y::c_api::y_free_string;
use y::cuda_runtime::{CudaContext, DeviceBuffer};

fn take_string(value: *mut c_char) -> String {
    assert!(!value.is_null());
    unsafe {
        let result = CStr::from_ptr(value).to_str().unwrap().to_owned();
        y_free_string(value);
        result
    }
}

fn status(result: i32, error: *mut c_char) {
    if !error.is_null() {
        panic!("C ABI failed ({result}): {}", take_string(error));
    }
    assert_eq!(result, 0);
}

fn json(result: *mut c_char, error: *mut c_char) -> String {
    if !error.is_null() {
        panic!("C ABI failed: {}", take_string(error));
    }
    take_string(result)
}

fn config() -> YAdaptiveJitConfig {
    let mut config = MaybeUninit::uninit();
    let mut error = ptr::null_mut();
    status(
        unsafe {
            y_adaptive_jit_config_init(
                config.as_mut_ptr(),
                size_of::<YAdaptiveJitConfig>() as u32,
                &mut error,
            )
        },
        error,
    );
    unsafe { config.assume_init() }
}

struct Handle(*mut YAdaptiveJit);

impl Handle {
    fn create(config: &YAdaptiveJitConfig) -> Self {
        let mut error = ptr::null_mut();
        let raw = unsafe { y_adaptive_jit_create_current(config, &mut error) };
        if !error.is_null() {
            panic!("C ABI creation failed: {}", take_string(error));
        }
        assert!(!raw.is_null());
        Self(raw)
    }

    fn close(&mut self) {
        let mut error = ptr::null_mut();
        status(unsafe { y_adaptive_jit_destroy(self.0, &mut error) }, error);
        self.0 = ptr::null_mut();
    }

    fn prepare(&self, shape: [u32; 3]) {
        let mut error = ptr::null_mut();
        status(
            unsafe { y_adaptive_jit_prepare(self.0, shape[0], shape[1], shape[2], &mut error) },
            error,
        );
    }

    fn stats(&self, shape: [u32; 3]) -> String {
        let mut error = ptr::null_mut();
        let value =
            unsafe { y_adaptive_jit_stats_json(self.0, shape[0], shape[1], shape[2], &mut error) };
        json(value, error)
    }

    fn pending(&self) -> String {
        let mut error = ptr::null_mut();
        let value = unsafe { y_adaptive_jit_pending_json(self.0, &mut error) };
        json(value, error)
    }

    fn tune(&self, max_shapes: u32) -> String {
        let mut error = ptr::null_mut();
        let value = unsafe { y_adaptive_jit_tune_hot_json(self.0, max_shapes, &mut error) };
        json(value, error)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // Keep cleanup best-effort while an assertion is already unwinding;
            // successful test paths explicitly close and assert the result.
            let mut error = ptr::null_mut();
            unsafe {
                y_adaptive_jit_destroy(self.0, &mut error);
                y_free_string(error);
            }
        }
    }
}

fn input(count: usize, salt: u64) -> (Vec<u8>, Vec<f64>) {
    let mut bytes = Vec::with_capacity(count * 2);
    let mut values = Vec::with_capacity(count);
    let mut state = salt;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let sign = (state >> 63) as u16;
        let exponent = 12 + ((state >> 24) % 3) as u16;
        let fraction = (state & 1023) as u16;
        bytes.extend_from_slice(&((sign << 15) | (exponent << 10) | fraction).to_ne_bytes());
        let magnitude =
            (1.0 + f64::from(fraction) / 1024.0) * 2.0f64.powi(i32::from(exponent) - 15);
        values.push(if sign == 0 { magnitude } else { -magnitude });
    }
    (bytes, values)
}

struct Case {
    shape: [u32; 3],
    a: DeviceBuffer,
    b: DeviceBuffer,
    c: DeviceBuffer,
    reference: Vec<f64>,
}

impl Case {
    fn new(ctx: &CudaContext, shape: [u32; 3]) -> Self {
        let [m, n, k] = shape.map(|v| v as usize);
        let (a_bytes, a_host) = input(m * k, 0x1234_5678_9abc_def1);
        let (b_bytes, b_host) = input(k * n, 0x9876_5432_fedc_ba91);
        let a = ctx.alloc(a_bytes.len()).unwrap();
        let b = ctx.alloc(b_bytes.len()).unwrap();
        let c = ctx.alloc(m * n * 4).unwrap();
        ctx.memcpy_htod_at(&a, 0, &a_bytes).unwrap();
        ctx.memcpy_htod_at(&b, 0, &b_bytes).unwrap();
        let mut reference = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                for inner in 0..k {
                    reference[row * n + col] += a_host[row * k + inner] * b_host[inner * n + col];
                }
            }
        }
        Self {
            shape,
            a,
            b,
            c,
            reference,
        }
    }

    fn launch_and_check(&self, ctx: &CudaContext, handle: &Handle) {
        // Fresh poison catches skipped output stores even when reusing a kernel.
        ctx.memset_u8(&self.c, 0xff).unwrap();
        let [m, n, k] = self.shape;
        let mut error = ptr::null_mut();
        status(
            unsafe {
                y_adaptive_jit_launch(
                    handle.0,
                    m,
                    n,
                    k,
                    self.a.device_ptr(),
                    self.b.device_ptr(),
                    self.c.device_ptr(),
                    &mut error,
                )
            },
            error,
        );
        status(
            unsafe { y_adaptive_jit_synchronize(handle.0, &mut error) },
            error,
        );
        let mut output = vec![0; self.reference.len() * 4];
        ctx.memcpy_dtoh_at(&mut output, &self.c, 0).unwrap();
        for (index, (bytes, expected)) in output.chunks_exact(4).zip(&self.reference).enumerate() {
            let actual = f64::from(f32::from_ne_bytes(bytes.try_into().unwrap()));
            let tolerance = 0.0005 * (1.0 + expected.abs());
            assert!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
                "C ABI output {index}: {actual}, expected {expected}, tolerance {tolerance}"
            );
        }
    }
}

fn compact(json: &str) -> String {
    // Only used on fixed-key numerical reports with no error messages.
    json.chars().filter(|c| !c.is_ascii_whitespace()).collect()
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU; run serially without other GPU measurements"]
fn c_abi_deferred_tuning_reuses_external_buffers_and_preserves_host_context() {
    let ctx = CudaContext::new().expect("explicit GPU test requires a CUDA device");
    let mut config = config();
    config.hot_threshold = 2;
    config.max_candidates = 2;
    let mut handle = Handle::create(&config);
    let shape = [31, 48, 80];
    let case = Case::new(&ctx, shape);

    handle.prepare(shape);
    handle.prepare(shape);
    let prepared = compact(&handle.stats(shape));
    assert!(prepared.contains("\"launches\":0"), "{prepared}");
    assert!(prepared.contains("\"tier\":\"baseline\""), "{prepared}");
    assert!(prepared.contains("\"tuning_attempts\":0"), "{prepared}");
    assert!(prepared.contains("\"cache_hit\":false"), "{prepared}");
    assert_eq!(compact(&handle.pending()), "[]");

    case.launch_and_check(&ctx, &handle);
    assert_eq!(compact(&handle.pending()), "[]");
    case.launch_and_check(&ctx, &handle);
    assert_eq!(compact(&handle.pending()), "[[31,48,80]]");
    let hot = compact(&handle.stats(shape));
    assert!(hot.contains("\"launches\":2"), "{hot}");
    assert!(hot.contains("\"tuning_attempts\":0"), "{hot}");
    assert_eq!(compact(&handle.tune(0)), "[]");
    assert_eq!(compact(&handle.pending()), "[[31,48,80]]");

    let report = compact(&handle.tune(1));
    assert!(report.contains("\"shape\":[31,48,80]"), "{report}");
    assert!(report.contains("\"tuning_attempts\":1"), "{report}");
    assert!(report.contains("\"tuning_error\":null"), "{report}");
    assert!(
        report.contains("\"tier\":\"tuned\"") || report.contains("\"tier\":\"retained_baseline\""),
        "{report}"
    );
    assert_eq!(compact(&handle.pending()), "[]");
    assert_eq!(compact(&handle.tune(1)), "[]");
    case.launch_and_check(&ctx, &handle);
    let reused = compact(&handle.stats(shape));
    assert!(reused.contains("\"launches\":3"), "{reused}");
    assert!(reused.contains("\"tuning_attempts\":1"), "{reused}");
    handle.close();

    // Destroying the C handle must leave the host's context, allocations, and
    // ability to load and execute another kernel intact.
    ctx.require_current().unwrap();
    let scratch = ctx.alloc(64).unwrap();
    ctx.memset_u8(&scratch, 0xa5).unwrap();
    let mut copy = vec![0; 64];
    ctx.memcpy_dtoh_at(&mut copy, &scratch, 0).unwrap();
    assert_eq!(copy, vec![0xa5; 64]);
    config.tuning_policy = 2;
    let mut next = Handle::create(&config);
    case.launch_and_check(&ctx, &next);
    assert_eq!(compact(&next.pending()), "[]");
    next.close();
    ctx.require_current().unwrap();
}

#[test]
#[ignore = "requires an sm_80+ CUDA GPU"]
fn c_abi_rejects_other_threads_without_consuming_handle() {
    let ctx = CudaContext::new().expect("explicit GPU test requires a CUDA device");
    let mut handle = Handle::create(&config());
    handle.prepare([1, 16, 16]);
    let address = handle.0 as usize;
    std::thread::spawn(move || {
        let raw = address as *mut YAdaptiveJit;
        let mut error = ptr::null_mut();
        let result = unsafe { y_adaptive_jit_prepare(raw, 1, 16, 16, &mut error) };
        assert_eq!(result, -1);
        let message = take_string(error);
        assert!(message.to_lowercase().contains("thread"), "{message}");
        error = ptr::null_mut();
        let result = unsafe { y_adaptive_jit_destroy(raw, &mut error) };
        assert_eq!(result, -1);
        let message = take_string(error);
        assert!(message.to_lowercase().contains("thread"), "{message}");
    })
    .join()
    .unwrap();
    // No concurrent use occurs; the rejecting thread has finished above.
    let case = Case::new(&ctx, [1, 16, 16]);
    case.launch_and_check(&ctx, &handle);
    handle.close();
    ctx.require_current().unwrap();
}

#[cfg(unix)]
mod context_switch {
    use std::ffi::{c_char, c_void, CString};

    pub struct Driver {
        get: unsafe extern "C" fn(*mut *mut c_void) -> i32,
        set: unsafe extern "C" fn(*mut c_void) -> i32,
    }

    impl Driver {
        pub fn load() -> Self {
            extern "C" {
                fn dlopen(name: *const c_char, flags: i32) -> *mut c_void;
                fn dlsym(library: *mut c_void, symbol: *const c_char) -> *mut c_void;
            }
            unsafe {
                let name = CString::new("libcuda.so.1").unwrap();
                let library = dlopen(name.as_ptr(), 1);
                assert!(!library.is_null());
                let get = dlsym(library, CString::new("cuCtxGetCurrent").unwrap().as_ptr());
                let set = dlsym(library, CString::new("cuCtxSetCurrent").unwrap().as_ptr());
                assert!(!get.is_null() && !set.is_null());
                Self {
                    get: std::mem::transmute(get),
                    set: std::mem::transmute(set),
                }
            }
        }

        pub fn current(&self) -> *mut c_void {
            let mut context = std::ptr::null_mut();
            assert_eq!(unsafe { (self.get)(&mut context) }, 0);
            context
        }

        pub fn set(&self, context: *mut c_void) {
            assert_eq!(unsafe { (self.set)(context) }, 0);
        }
    }

    pub struct Restore<'a> {
        pub driver: &'a Driver,
        pub context: *mut c_void,
    }

    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            // Restore before the original handle/context's own cleanup runs.
            unsafe {
                (self.driver.set)(self.context);
            }
        }
    }
}

#[cfg(unix)]
#[test]
#[ignore = "requires an sm_80+ CUDA GPU"]
fn c_abi_rejects_missing_or_different_context_and_allows_close_retry() {
    let driver = context_switch::Driver::load();
    let ctx = CudaContext::new().expect("explicit GPU test requires a CUDA device");
    let mut handle = Handle::create(&config());
    handle.prepare([1, 16, 16]);
    let original = driver.current();
    assert!(!original.is_null());
    let restore = context_switch::Restore {
        driver: &driver,
        context: original,
    };

    driver.set(ptr::null_mut());
    let mut error = ptr::null_mut();
    let missing = unsafe { y_adaptive_jit_create_current(ptr::null(), &mut error) };
    assert!(missing.is_null());
    let message = take_string(error);
    assert!(message.to_lowercase().contains("context"), "{message}");

    let other = CudaContext::new().expect("second CUDA context");
    assert_ne!(driver.current(), original);
    error = ptr::null_mut();
    let rejected = unsafe { y_adaptive_jit_stats_json(handle.0, 1, 16, 16, &mut error) };
    assert!(rejected.is_null());
    let message = take_string(error);
    assert!(message.to_lowercase().contains("context"), "{message}");
    error = ptr::null_mut();
    let rejected = unsafe { y_adaptive_jit_destroy(handle.0, &mut error) };
    assert_eq!(rejected, -1);
    let message = take_string(error);
    assert!(message.to_lowercase().contains("context"), "{message}");
    other.require_current().unwrap();
    drop(other);
    drop(restore);
    ctx.require_current().unwrap();
    let case = Case::new(&ctx, [1, 16, 16]);
    case.launch_and_check(&ctx, &handle);
    handle.close();
    ctx.require_current().unwrap();
}
