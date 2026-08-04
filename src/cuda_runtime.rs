// ============================================================
//  Y  —  Minimal CUDA Driver API binding (dynamically loaded)
//  cuda_runtime.rs
//
//  Enough of the CUDA Driver API to JIT a PTX string, launch the
//  resulting kernel and time it with CUDA events. Loaded at runtime via
//  `dlopen("libcuda.so.1")` so that the compiler still builds and runs on
//  machines with no CUDA toolkit and no NVIDIA driver installed - every
//  entry point here is reached through `CudaContext::new()`, which returns
//  `None` instead of failing the build/link when the driver is absent.
//
//  This exists because `src/autotuner.rs`'s empirical mode has to run
//  candidate kernels on the real device to rank them. `src/ysu_gpu_probe.rs`
//  already carries its own private copy of a loader like this one, but it is
//  a `[[bin]]` target (see Cargo.toml), so none of it is reachable from the
//  library where the autotuner lives. This module is the library-visible
//  one; the probe's copy is deliberately left alone (it additionally binds
//  NVRTC, which nothing here needs).
//
//  Scope note: this is NOT a general-purpose CUDA binding and should not
//  grow into one. It binds exactly the calls the autotuner's measurement
//  loop issues, with a single device, a single context and a single stream.
// ============================================================

#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::{c_void, CStr, CString};

pub type CUresult = i32;
pub type CUdevice = i32;
/// `CUdeviceptr` is `unsigned long long` in the 64-bit driver ABI. Spelled
/// `u64` rather than `usize` so the FFI signatures stay correct by
/// construction rather than by coincidence on 64-bit hosts.
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;

/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`. A kernel that requests
/// more than 48KB of dynamic shared memory must opt in through
/// `cuFuncSetAttribute` before launch or `cuLaunchKernel` fails with
/// `CUDA_ERROR_INVALID_VALUE` - every pipelined GEMM config the autotuner
/// generates is over that line, so this is the common case, not an edge one.
const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;

/// `CU_JIT_ERROR_LOG_BUFFER` / `..._SIZE_BYTES`, so a candidate whose PTX the
/// driver rejects reports *why* instead of just disappearing from the ranking.
const CU_JIT_ERROR_LOG_BUFFER: i32 = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: i32 = 6;

// ── dynamic library loading ────────────────────────────────

#[cfg(unix)]
unsafe fn load_library(paths: &[&str]) -> Option<*mut c_void> {
    extern "C" {
        fn dlopen(filename: *const u8, flag: i32) -> *mut c_void;
    }
    for path in paths {
        if let Ok(c_path) = CString::new(*path) {
            let handle = dlopen(c_path.as_ptr() as *const u8, 1); // RTLD_LAZY
            if !handle.is_null() {
                return Some(handle);
            }
        }
    }
    None
}

#[cfg(unix)]
unsafe fn get_symbol(lib: *mut c_void, name: &str) -> Option<*mut c_void> {
    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
    }
    let c_name = CString::new(name).ok()?;
    let sym = dlsym(lib, c_name.as_ptr() as *const u8);
    if sym.is_null() {
        None
    } else {
        Some(sym)
    }
}

#[cfg(windows)]
unsafe fn load_library(paths: &[&str]) -> Option<*mut c_void> {
    extern "system" {
        fn LoadLibraryA(lpLibFileName: *const u8) -> *mut c_void;
    }
    for path in paths {
        if let Ok(c_path) = CString::new(*path) {
            let handle = LoadLibraryA(c_path.as_ptr() as *const u8);
            if !handle.is_null() {
                return Some(handle);
            }
        }
    }
    None
}

#[cfg(windows)]
unsafe fn get_symbol(lib: *mut c_void, name: &str) -> Option<*mut c_void> {
    extern "system" {
        fn GetProcAddress(hModule: *mut c_void, lpProcName: *const u8) -> *mut c_void;
    }
    let c_name = CString::new(name).ok()?;
    let sym = GetProcAddress(lib, c_name.as_ptr() as *const u8);
    if sym.is_null() {
        None
    } else {
        Some(sym)
    }
}

// ── resolved entry points ──────────────────────────────────

#[derive(Copy, Clone)]
struct Driver {
    cuInit: unsafe extern "C" fn(u32) -> CUresult,
    cuDeviceGetCount: unsafe extern "C" fn(*mut i32) -> CUresult,
    cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult,
    cuDeviceGetName: unsafe extern "C" fn(*mut u8, i32, CUdevice) -> CUresult,
    cuDeviceGetAttribute: unsafe extern "C" fn(*mut i32, i32, CUdevice) -> CUresult,
    cuCtxCreate: unsafe extern "C" fn(*mut *mut c_void, u32, CUdevice) -> CUresult,
    cuCtxDestroy: unsafe extern "C" fn(*mut c_void) -> CUresult,
    cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
    cuModuleLoadDataEx: unsafe extern "C" fn(
        *mut *mut c_void,
        *const c_void,
        u32,
        *mut i32,
        *mut *mut c_void,
    ) -> CUresult,
    cuModuleUnload: unsafe extern "C" fn(*mut c_void) -> CUresult,
    cuModuleGetFunction: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const u8) -> CUresult,
    cuFuncSetAttribute: unsafe extern "C" fn(*mut c_void, i32, i32) -> CUresult,
    cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    cuMemFree: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    cuMemGetInfo: unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult,
    cuMemsetD8: unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult,
    cuMemcpyHtoD: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    cuMemcpyDtoH: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    cuMemcpyDtoD: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult,
    cuLaunchKernel: unsafe extern "C" fn(
        *mut c_void,
        u32, u32, u32,
        u32, u32, u32,
        u32,
        *mut c_void,
        *const *mut c_void,
        *const *mut c_void,
    ) -> CUresult,
    cuEventCreate: unsafe extern "C" fn(*mut *mut c_void, u32) -> CUresult,
    cuEventRecord: unsafe extern "C" fn(*mut c_void, *mut c_void) -> CUresult,
    cuEventSynchronize: unsafe extern "C" fn(*mut c_void) -> CUresult,
    cuEventElapsedTime: unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> CUresult,
    cuEventDestroy: unsafe extern "C" fn(*mut c_void) -> CUresult,
}

impl Driver {
    unsafe fn load() -> Option<Self> {
        #[cfg(unix)]
        let lib = load_library(&["libcuda.so.1", "libcuda.so"])?;
        #[cfg(windows)]
        let lib = load_library(&["nvcuda.dll"])?;

        // Several driver entry points were ABI-versioned when 64-bit sizes
        // were introduced (CUDA 3.2): the unsuffixed `cuMemAlloc` symbol is
        // the *legacy* 32-bit-size form, and calling it with a `usize` byte
        // count silently truncates above 4GB. Resolve the `_v2` symbol first
        // and only fall back to the plain name if the driver is old enough
        // not to export it.
        macro_rules! resolve_v2 {
            ($name:ident) => {
                let sym = get_symbol(lib, concat!(stringify!($name), "_v2"))
                    .or_else(|| get_symbol(lib, stringify!($name)))?;
                let $name = std::mem::transmute::<*mut c_void, _>(sym);
            };
        }
        macro_rules! resolve {
            ($name:ident) => {
                let sym = get_symbol(lib, stringify!($name))?;
                let $name = std::mem::transmute::<*mut c_void, _>(sym);
            };
        }

        resolve!(cuInit);
        resolve!(cuDeviceGetCount);
        resolve!(cuDeviceGet);
        resolve!(cuDeviceGetName);
        resolve!(cuDeviceGetAttribute);
        resolve_v2!(cuCtxCreate);
        resolve_v2!(cuCtxDestroy);
        resolve!(cuCtxSynchronize);
        resolve!(cuModuleLoadDataEx);
        resolve!(cuModuleUnload);
        resolve!(cuModuleGetFunction);
        resolve!(cuFuncSetAttribute);
        resolve_v2!(cuMemAlloc);
        resolve_v2!(cuMemFree);
        resolve_v2!(cuMemGetInfo);
        resolve_v2!(cuMemsetD8);
        resolve_v2!(cuMemcpyHtoD);
        resolve_v2!(cuMemcpyDtoH);
        resolve_v2!(cuMemcpyDtoD);
        resolve!(cuLaunchKernel);
        resolve!(cuEventCreate);
        resolve!(cuEventRecord);
        resolve!(cuEventSynchronize);
        resolve!(cuEventElapsedTime);
        resolve_v2!(cuEventDestroy);

        Some(Driver {
            cuInit,
            cuDeviceGetCount,
            cuDeviceGet,
            cuDeviceGetName,
            cuDeviceGetAttribute,
            cuCtxCreate,
            cuCtxDestroy,
            cuCtxSynchronize,
            cuModuleLoadDataEx,
            cuModuleUnload,
            cuModuleGetFunction,
            cuFuncSetAttribute,
            cuMemAlloc,
            cuMemFree,
            cuMemGetInfo,
            cuMemsetD8,
            cuMemcpyHtoD,
            cuMemcpyDtoH,
            cuMemcpyDtoD,
            cuLaunchKernel,
            cuEventCreate,
            cuEventRecord,
            cuEventSynchronize,
            cuEventElapsedTime,
            cuEventDestroy,
        })
    }
}

// ── public surface ─────────────────────────────────────────

/// Device memory owned by a `CudaContext`. Freed on drop.
///
/// Holds a raw `Driver` copy rather than a borrow of the context so that a
/// buffer and the context can be held in the same struct without fighting
/// the borrow checker. That is sound only because a `DeviceBuffer` can never
/// outlive its context in this module's usage (both live in
/// `GemmProbeHarness`, and the buffers are declared before the context so
/// they drop first) - do not hand these out across API boundaries.
pub struct DeviceBuffer {
    ptr: CUdeviceptr,
    len_bytes: usize,
    drv: Driver,
}

impl DeviceBuffer {
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                (self.drv.cuMemFree)(self.ptr);
            }
        }
    }
}

/// A JIT-compiled PTX module plus one resolved entry point.
pub struct KernelModule {
    module: *mut c_void,
    func: *mut c_void,
    drv: Driver,
}

impl KernelModule {
    /// Opt a kernel in to more than the default 48KB of dynamic shared
    /// memory. Must be called before any launch that passes a larger
    /// `shared_bytes` - see `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`.
    pub fn set_max_dynamic_smem(&self, bytes: u32) -> Result<(), String> {
        unsafe {
            check(
                (self.drv.cuFuncSetAttribute)(
                    self.func,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                ),
                "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
            )
        }
    }
}

impl Drop for KernelModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                (self.drv.cuModuleUnload)(self.module);
            }
        }
    }
}

fn check(res: CUresult, what: &str) -> Result<(), String> {
    if res == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{} failed (CUresult {})", what, res))
    }
}

/// A CUDA device + context. Bound to the thread that created it, and
/// deliberately neither `Send` nor `Sync` (the raw context pointer is not
/// auto-`Send`, which enforces this for free).
pub struct CudaContext {
    drv: Driver,
    ctx: *mut c_void,
    device: CUdevice,
    name: String,
}

impl CudaContext {
    /// Initialises CUDA and creates a context on device 0.
    ///
    /// Returns `None` - never panics, never aborts - when there is no
    /// driver, no device, or the driver refuses to initialise. Callers are
    /// expected to fall back to a non-measuring code path on `None`.
    pub fn new() -> Option<Self> {
        unsafe {
            let drv = Driver::load()?;
            if (drv.cuInit)(0) != CUDA_SUCCESS {
                return None;
            }
            let mut count = 0i32;
            if (drv.cuDeviceGetCount)(&mut count) != CUDA_SUCCESS || count <= 0 {
                return None;
            }
            let mut device: CUdevice = 0;
            if (drv.cuDeviceGet)(&mut device, 0) != CUDA_SUCCESS {
                return None;
            }
            let mut name_buf = [0u8; 256];
            let name = if (drv.cuDeviceGetName)(name_buf.as_mut_ptr(), 256, device) == CUDA_SUCCESS
            {
                CStr::from_ptr(name_buf.as_ptr() as *const i8)
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            };
            let mut ctx: *mut c_void = std::ptr::null_mut();
            if (drv.cuCtxCreate)(&mut ctx, 0, device) != CUDA_SUCCESS || ctx.is_null() {
                return None;
            }
            Some(CudaContext { drv, ctx, device, name })
        }
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// Reads a `CUdevice_attribute` by raw ordinal. Used to sanity-check that
    /// the device actually present matches the cached hardware profile the
    /// candidates were generated against.
    pub fn device_attribute(&self, attrib: i32) -> Option<i32> {
        unsafe {
            let mut v = 0i32;
            if (self.drv.cuDeviceGetAttribute)(&mut v, attrib, self.device) == CUDA_SUCCESS {
                Some(v)
            } else {
                None
            }
        }
    }

    pub fn alloc(&self, len_bytes: usize) -> Result<DeviceBuffer, String> {
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            check((self.drv.cuMemAlloc)(&mut ptr, len_bytes), "cuMemAlloc")?;
            Ok(DeviceBuffer { ptr, len_bytes, drv: self.drv })
        }
    }

    pub fn memset_u8(&self, buf: &DeviceBuffer, value: u8) -> Result<(), String> {
        unsafe {
            check(
                (self.drv.cuMemsetD8)(buf.ptr, value, buf.len_bytes),
                "cuMemsetD8",
            )
        }
    }

    /// Host->device copy into `buf` at `offset_bytes`.
    pub fn memcpy_htod_at(
        &self,
        buf: &DeviceBuffer,
        offset_bytes: usize,
        src: &[u8],
    ) -> Result<(), String> {
        if offset_bytes + src.len() > buf.len_bytes {
            return Err(format!(
                "memcpy_htod_at out of range: offset {} + {} bytes > buffer {} bytes",
                offset_bytes,
                src.len(),
                buf.len_bytes
            ));
        }
        unsafe {
            check(
                (self.drv.cuMemcpyHtoD)(
                    buf.ptr + offset_bytes as u64,
                    src.as_ptr() as *const c_void,
                    src.len(),
                ),
                "cuMemcpyHtoD",
            )
        }
    }

    /// Device->host copy of `dst.len()` bytes starting at `offset_bytes`.
    /// Reading a handful of scattered elements this way is deliberate: the
    /// autotuner's correctness check samples the output rather than pulling
    /// back a whole M*N f32 matrix (1GB at M=N=16384).
    pub fn memcpy_dtoh_at(
        &self,
        dst: &mut [u8],
        buf: &DeviceBuffer,
        offset_bytes: usize,
    ) -> Result<(), String> {
        if offset_bytes + dst.len() > buf.len_bytes {
            return Err(format!(
                "memcpy_dtoh_at out of range: offset {} + {} bytes > buffer {} bytes",
                offset_bytes,
                dst.len(),
                buf.len_bytes
            ));
        }
        unsafe {
            check(
                (self.drv.cuMemcpyDtoH)(
                    dst.as_mut_ptr() as *mut c_void,
                    buf.ptr + offset_bytes as u64,
                    dst.len(),
                ),
                "cuMemcpyDtoH",
            )
        }
    }

    /// Device-to-device copy. Used to replicate the weight matrix cheaply
    /// when the measurement needs several distinct copies of it (see
    /// `empirical_autotune`'s L2 rotation) - regenerating each copy on the
    /// host and uploading it would cost seconds for no benefit, since the
    /// copies only have to occupy DIFFERENT memory, not hold different data.
    pub fn memcpy_dtod(&self, dst: &DeviceBuffer, src: &DeviceBuffer) -> Result<(), String> {
        let n = dst.len_bytes.min(src.len_bytes);
        unsafe { check((self.drv.cuMemcpyDtoD)(dst.ptr, src.ptr, n), "cuMemcpyDtoD") }
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Option<(usize, usize)> {
        unsafe {
            let (mut free, mut total) = (0usize, 0usize);
            if (self.drv.cuMemGetInfo)(&mut free, &mut total) == CUDA_SUCCESS {
                Some((free, total))
            } else {
                None
            }
        }
    }

    pub fn synchronize(&self) -> Result<(), String> {
        unsafe { check((self.drv.cuCtxSynchronize)(), "cuCtxSynchronize") }
    }

    /// JIT-compiles a PTX string and resolves one entry point out of it.
    ///
    /// This is the same driver-side JIT path the shipping kernels take
    /// (cupy's `RawModule(path=...)`, the Python benchmark harnesses' loader,
    /// hands the driver the identical PTX text), so a candidate measured here
    /// is measured through the machinery it will actually run through - not
    /// through an offline `ptxas` binary that may be a different version than
    /// the installed driver's built-in compiler.
    pub fn load_ptx(&self, ptx: &str, entry: &str) -> Result<KernelModule, String> {
        let ptx_c = CString::new(ptx).map_err(|_| "PTX contains an interior NUL byte".to_string())?;
        let entry_c =
            CString::new(entry).map_err(|_| "entry name contains an interior NUL byte".to_string())?;

        let mut log = vec![0u8; 8192];
        let mut options = [CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
        let mut option_values: [*mut c_void; 2] =
            [log.as_mut_ptr() as *mut c_void, log.len() as *mut c_void];

        unsafe {
            let mut module: *mut c_void = std::ptr::null_mut();
            let res = (self.drv.cuModuleLoadDataEx)(
                &mut module,
                ptx_c.as_ptr() as *const c_void,
                options.len() as u32,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
            );
            if res != CUDA_SUCCESS {
                let msg = CStr::from_ptr(log.as_ptr() as *const i8)
                    .to_string_lossy()
                    .trim()
                    .to_string();
                return Err(format!(
                    "cuModuleLoadDataEx failed (CUresult {}){}",
                    res,
                    if msg.is_empty() { String::new() } else { format!(": {}", msg) }
                ));
            }

            let mut func: *mut c_void = std::ptr::null_mut();
            let res = (self.drv.cuModuleGetFunction)(&mut func, module, entry_c.as_ptr() as *const u8);
            if res != CUDA_SUCCESS {
                (self.drv.cuModuleUnload)(module);
                return Err(format!(
                    "cuModuleGetFunction('{}') failed (CUresult {})",
                    entry, res
                ));
            }
            Ok(KernelModule { module, func, drv: self.drv })
        }
    }

    /// Enqueues one launch. `args` are the raw kernel parameter values in
    /// declaration order; this helper only supports device-pointer
    /// parameters, which is all the GEMM probe kernels take.
    pub fn launch(
        &self,
        kernel: &KernelModule,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        args: &[CUdeviceptr],
    ) -> Result<(), String> {
        // `cuLaunchKernel` takes an array of POINTERS TO the argument values,
        // so the values themselves must outlive the call.
        let mut arg_values: Vec<CUdeviceptr> = args.to_vec();
        let arg_ptrs: Vec<*mut c_void> = arg_values
            .iter_mut()
            .map(|v| v as *mut CUdeviceptr as *mut c_void)
            .collect();
        unsafe {
            check(
                (self.drv.cuLaunchKernel)(
                    kernel.func,
                    grid.0, grid.1, grid.2,
                    block.0, block.1, block.2,
                    shared_bytes,
                    std::ptr::null_mut(), // default stream
                    arg_ptrs.as_ptr(),
                    std::ptr::null(),
                ),
                "cuLaunchKernel",
            )
        }
    }

    /// Times `iters` back-to-back launches with CUDA events and returns the
    /// mean microseconds per launch.
    ///
    /// Events bracket the whole batch rather than each individual launch:
    /// per-launch event overhead is on the order of the kernels being
    /// measured at decode shapes (tens of microseconds), which would swamp
    /// exactly the differences this is here to resolve.
    ///
    /// `arg_sets` is cycled across launches. That is how the caller rotates
    /// over several distinct weight buffers to keep the working set out of
    /// L2 - see `empirical_autotune::GemmProbe`.
    pub fn time_launches(
        &self,
        kernel: &KernelModule,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        arg_sets: &[Vec<CUdeviceptr>],
        iters: u32,
    ) -> Result<f64, String> {
        if iters == 0 {
            return Err("time_launches called with iters = 0".to_string());
        }
        if arg_sets.is_empty() {
            return Err("time_launches called with no argument sets".to_string());
        }
        unsafe {
            let mut ev_start: *mut c_void = std::ptr::null_mut();
            let mut ev_end: *mut c_void = std::ptr::null_mut();
            check((self.drv.cuEventCreate)(&mut ev_start, 0), "cuEventCreate")?;
            check((self.drv.cuEventCreate)(&mut ev_end, 0), "cuEventCreate")?;

            let run = || -> Result<f64, String> {
                self.synchronize()?;
                check((self.drv.cuEventRecord)(ev_start, std::ptr::null_mut()), "cuEventRecord")?;
                for i in 0..iters {
                    self.launch(
                        kernel,
                        grid,
                        block,
                        shared_bytes,
                        &arg_sets[i as usize % arg_sets.len()],
                    )?;
                }
                check((self.drv.cuEventRecord)(ev_end, std::ptr::null_mut()), "cuEventRecord")?;
                check((self.drv.cuEventSynchronize)(ev_end), "cuEventSynchronize")?;
                let mut ms = 0f32;
                check(
                    (self.drv.cuEventElapsedTime)(&mut ms, ev_start, ev_end),
                    "cuEventElapsedTime",
                )?;
                Ok((ms as f64 * 1000.0) / iters as f64)
            };

            let result = run();
            (self.drv.cuEventDestroy)(ev_start);
            (self.drv.cuEventDestroy)(ev_end);
            result
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                (self.drv.cuCtxDestroy)(self.ctx);
            }
        }
    }
}

// ── half-precision helpers ─────────────────────────────────

/// Decodes an IEEE binary16 bit pattern to `f32`. Exact for every normal
/// input, which is all this module's generator produces (see
/// `random_f16_bits`) - subnormals, infinities and NaNs are not handled
/// because they are never generated.
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    debug_assert!(exp != 0 && exp != 31, "f16_bits_to_f32 handles normals only");
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}

/// Deterministic pseudo-random *normal* binary16 bit pattern for element
/// `index`.
///
/// Generating the f16 bits directly and decoding them for the CPU reference
/// (rather than generating f32 and rounding to f16) means the reference and
/// the kernel provably read the exact same values - there is no f32->f16
/// rounding step that could differ between the two and show up as a fake
/// correctness failure.
///
/// The exponent is confined to 12..=14, i.e. magnitudes in [2^-3, 2^0), so
/// that a K-deep dot product cannot overflow f32 accumulation at any K this
/// compiler supports, and no subnormal/Inf/NaN is ever produced.
pub fn random_f16_bits(index: u64, seed: u64) -> u16 {
    // splitmix64
    let mut z = index
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(seed)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;

    let sign = ((z >> 63) & 1) as u16;
    let exp = 12u16 + ((z >> 40) & 0x3) as u16 % 3; // 12, 13 or 14
    let mant = (z & 0x3ff) as u16;
    (sign << 15) | (exp << 10) | mant
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_decode_matches_known_values() {
        // 1.0 = 0x3C00, -2.0 = 0xC000, 0.5 = 0x3800
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0xC000), -2.0);
        assert_eq!(f16_bits_to_f32(0x3800), 0.5);
    }

    #[test]
    fn generated_f16_is_always_a_bounded_normal() {
        for i in 0..10_000u64 {
            let bits = random_f16_bits(i, 0x5EED_5EED);
            let exp = (bits >> 10) & 0x1f;
            assert!(exp >= 12 && exp <= 14, "exponent {} out of the safe band", exp);
            let v = f16_bits_to_f32(bits).abs();
            assert!(v >= 0.125 && v < 1.0, "magnitude {} out of the safe band", v);
        }
    }
}
