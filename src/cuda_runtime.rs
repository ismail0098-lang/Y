// ============================================================
//  Y  —  Minimal CUDA Driver API binding (dynamically loaded)
//  cuda_runtime.rs
//
//  Enough of the CUDA Driver API to JIT PTX or load a validated exact_pv
//  cubin, launch the kernel and time it with CUDA events. Loaded at runtime via
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
//  grow into one. It binds the autotuner's measurement calls and the verified
//  exact_pv artifact loader, with one owned device, context and stream.
// ============================================================

#![allow(non_snake_case)]

use std::ffi::{c_void, CStr, CString};
use std::rc::Rc;
#[cfg(any(target_os = "linux", test))]
use std::io::Read;

pub type CUresult = i32;
pub type CUdevice = i32;
/// `CUdeviceptr` is `unsigned long long` in the 64-bit driver ABI. Spelled
/// `u64` rather than `usize` so the FFI signatures stay correct by
/// construction rather than by coincidence on 64-bit hosts.
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;

// CUDA's CUuuid is a struct containing exactly 16 bytes, including on
// platforms where plain C char is signed. Keep its C layout at the FFI edge.
#[repr(C)]
struct CUuuid {
    bytes: [u8; 16],
}

type DeviceGetUuid = unsafe extern "C" fn(*mut CUuuid, CUdevice) -> CUresult;
type DriverGetVersion = unsafe extern "C" fn(*mut i32) -> CUresult;
type FuncGetAttribute = unsafe extern "C" fn(*mut i32, i32, *mut c_void) -> CUresult;

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

/// Memory sizes and device pointers use the v2 ABI in this binding. An
/// unsuffixed legacy export cannot establish those widths, even when present.
fn required_memory_symbol(
    name: &str,
    mut lookup: impl FnMut(&str) -> Option<*mut c_void>,
) -> Option<*mut c_void> {
    lookup(&format!("{name}_v2"))
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
    cuDeviceGetUuid: Option<DeviceGetUuid>,
    cuDriverGetVersion: Option<DriverGetVersion>,
    cuCtxCreate: unsafe extern "C" fn(*mut *mut c_void, u32, CUdevice) -> CUresult,
    cuCtxDestroy: unsafe extern "C" fn(*mut c_void) -> CUresult,
    cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
    cuCtxGetCurrent: unsafe extern "C" fn(*mut *mut c_void) -> CUresult,
    cuCtxGetDevice: unsafe extern "C" fn(*mut CUdevice) -> CUresult,
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
    cuFuncGetAttribute: Option<FuncGetAttribute>,
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

        // Memory exports must establish the 64-bit device-pointer/size ABI.
        // Legacy unsuffixed exports can truncate a >4GiB allocation while
        // DeviceBuffer records the full requested extent. All real sm_89
        // drivers provide v2; refuse an unidentified ABI at driver loading.
        macro_rules! resolve_memory_v2 {
            ($name:ident) => {
                let sym = required_memory_symbol(stringify!($name), |name| get_symbol(lib, name))?;
                let $name = std::mem::transmute::<*mut c_void, _>(sym);
            };
        }
        // Context/event APIs retain their existing compatibility resolution;
        // these signatures do not carry allocation or transfer byte extents.
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
        // Cache identity is optional: unavailable symbols must not prevent
        // ordinary CUDA execution. v2 distinguishes MIG compute instances.
        let cuDeviceGetUuid = get_symbol(lib, "cuDeviceGetUuid_v2")
            .or_else(|| get_symbol(lib, "cuDeviceGetUuid"))
            .map(|sym| std::mem::transmute::<*mut c_void, DeviceGetUuid>(sym));
        let cuDriverGetVersion = get_symbol(lib, "cuDriverGetVersion")
            .map(|sym| std::mem::transmute::<*mut c_void, DriverGetVersion>(sym));
        resolve_v2!(cuCtxCreate);
        resolve_v2!(cuCtxDestroy);
        resolve!(cuCtxSynchronize);
        resolve!(cuCtxGetCurrent);
        resolve!(cuCtxGetDevice);
        resolve!(cuModuleLoadDataEx);
        resolve!(cuModuleUnload);
        resolve!(cuModuleGetFunction);
        resolve!(cuFuncSetAttribute);
        // Ordinary probes remain usable if this optional query is missing;
        // the checked launch path explicitly refuses an unknown function cap.
        let cuFuncGetAttribute = get_symbol(lib, "cuFuncGetAttribute")
            .map(|sym| std::mem::transmute::<*mut c_void, FuncGetAttribute>(sym));
        resolve_memory_v2!(cuMemAlloc);
        resolve_memory_v2!(cuMemFree);
        resolve_memory_v2!(cuMemGetInfo);
        resolve_memory_v2!(cuMemsetD8);
        resolve_memory_v2!(cuMemcpyHtoD);
        resolve_memory_v2!(cuMemcpyDtoH);
        resolve_memory_v2!(cuMemcpyDtoD);
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
            cuDeviceGetUuid,
            cuDriverGetVersion,
            cuCtxCreate,
            cuCtxDestroy,
            cuCtxSynchronize,
            cuCtxGetCurrent,
            cuCtxGetDevice,
            cuModuleLoadDataEx,
            cuModuleUnload,
            cuModuleGetFunction,
            cuFuncSetAttribute,
            cuFuncGetAttribute,
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

/// Device identity used to scope persistent kernel tuning decisions.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DeviceIdentity {
    /// The UUID reported by CUDA, preferring the MIG-aware v2 entry point.
    pub uuid: [u8; 16],
    /// CUDA version supported by the installed driver, encoded as
    /// `1000 * major + 10 * minor`; this is not its package build number.
    pub driver_version: i32,
    /// Linux kernel driver build identity, read from NVIDIA's proc metadata.
    /// This does not fingerprint the complete userspace driver binaries.
    pub driver_build: String,
}

#[cfg(any(target_os = "linux", test))]
fn read_driver_build(reader: impl Read) -> Result<String, String> {
    const MAX_BUILD_BYTES: u64 = 16 * 1024;
    let mut bytes = Vec::new();
    reader.take(MAX_BUILD_BYTES + 1).read_to_end(&mut bytes)
        .map_err(|err| format!("cannot read NVIDIA driver build identity: {err}"))?;
    if bytes.len() > MAX_BUILD_BYTES as usize {
        return Err("NVIDIA driver build identity exceeds 16 KiB".into());
    }
    let text = String::from_utf8(bytes)
        .map_err(|_| "NVIDIA driver build identity is not UTF-8")?;
    let build = text.trim();
    if build.is_empty() || build.contains('\0') {
        return Err("NVIDIA driver build identity is empty or invalid".into());
    }
    Ok(build.to_owned())
}

fn query_cache_identity(
    device: CUdevice,
    get_uuid: Option<DeviceGetUuid>,
    get_version: Option<DriverGetVersion>,
    driver_build: String,
) -> Result<DeviceIdentity, String> {
    let get_uuid = get_uuid.ok_or("CUDA device UUID query is unavailable")?;
    let get_version = get_version.ok_or("CUDA driver version query is unavailable")?;
    let mut uuid = CUuuid { bytes: [0; 16] };
    let mut driver_version = 0;
    unsafe {
        check(get_uuid(&mut uuid, device), "cuDeviceGetUuid")?;
        check(get_version(&mut driver_version), "cuDriverGetVersion")?;
    }
    if uuid.bytes == [0; 16] {
        return Err("CUDA driver returned an unidentified device UUID".into());
    }
    if driver_version <= 0 {
        return Err("CUDA driver returned an invalid driver version".into());
    }
    Ok(DeviceIdentity { uuid: uuid.bytes, driver_version, driver_build })
}

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
    // An allocation-generation token, not merely a device/context address.
    // Retaining it prevents a destroyed context's reused handle from making
    // an old buffer appear to belong to a newly created wrapper.
    ownership: Rc<()>,
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

/// A loaded CUDA module plus one resolved entry point.
pub struct KernelModule {
    module: *mut c_void,
    func: *mut c_void,
    drv: Driver,
}

/// The reviewed, hash-bound exact_pv module with a live originating context.
///
/// Its kernel handle is private and cannot be implicitly used by the generic
/// launch API. Launch through `CudaContext::launch_checked_exact_pv`; the
/// arithmetic/model/ISA trust boundaries of the artifact still apply.
pub struct VerifiedExactPvModule<'ctx> {
    context: &'ctx CudaContext,
    kernel: KernelModule,
    cubin_sha256: String,
}

impl VerifiedExactPvModule<'_> {
    pub fn cubin_sha256(&self) -> &str { &self.cubin_sha256 }
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

// The same helper is exercised with a recording CUDA API in unit tests, so
// tests cover the actual bytes and options submitted by the verified path.
fn load_verified_image(
    artifact: &crate::verified_exact_pv::ValidatedExactPv,
    major: i32,
    minor: i32,
    load: unsafe extern "C" fn(*mut *mut c_void, *const c_void, u32, *mut i32, *mut *mut c_void) -> CUresult,
    lookup: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const u8) -> CUresult,
    unload: unsafe extern "C" fn(*mut c_void) -> CUresult,
) -> Result<(*mut c_void, *mut c_void), String> {
    if (major, minor) != (8, 9) {
        return Err(format!("validated exact_pv requires sm_89; device is sm_{major}{minor}"));
    }
    // Read and hash AFTER identifying the device. The same owned bytes go
    // into the driver; file replacement after this point cannot change them.
    let cubin = artifact.checked_cubin()?;
    unsafe {
        let mut module = std::ptr::null_mut();
        check(
            load(&mut module, cubin.as_ptr().cast(), 0, std::ptr::null_mut(), std::ptr::null_mut()),
            "cuModuleLoadDataEx(validated exact_pv cubin)",
        )?;
        if module.is_null() {
            return Err("CUDA driver did not identify the loaded exact_pv module".into());
        }
        let mut func = std::ptr::null_mut();
        let result = lookup(&mut func, module, b"exact_pv\0".as_ptr());
        if result != CUDA_SUCCESS || func.is_null() {
            unload(module);
            return Err(format!("cuModuleGetFunction(exact_pv) failed or unidentified (CUresult {result})"));
        }
        Ok((module, func))
    }
}

/// A CUDA device + context, either owned or borrowed from an embedding host.
/// Bound to the thread that created or borrowed it, and
/// deliberately neither `Send` nor `Sync` (the raw context pointer is not
/// auto-`Send`, which enforces this for free).
pub struct CudaContext {
    drv: Driver,
    ctx: *mut c_void,
    owns_context: bool,
    device: CUdevice,
    name: String,
    ownership: Rc<()>,
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
            Some(CudaContext { drv, ctx, owns_context: true, device, name, ownership: Rc::new(()) })
        }
    }

    /// Wrap the exact CUDA context already current on this thread.
    /// Does not create, switch, retain, or destroy the host's context.
    ///
    /// # Safety
    /// The host must keep this context alive and current on this thread while
    /// the wrapper and resources created through it are used or dropped.
    /// All such resources must be released before the host destroys the context.
    pub unsafe fn borrow_current() -> Result<Self, String> {
        let drv = Driver::load().ok_or("CUDA driver is unavailable")?;
        check((drv.cuInit)(0), "cuInit")?;
        Self::borrow_current_from_driver(drv)
    }

    unsafe fn borrow_current_from_driver(drv: Driver) -> Result<Self, String> {
        let mut ctx = std::ptr::null_mut();
        check((drv.cuCtxGetCurrent)(&mut ctx), "cuCtxGetCurrent")?;
        if ctx.is_null() {
            return Err("no CUDA context is current on the calling thread".into());
        }
        let mut device: CUdevice = 0;
        check((drv.cuCtxGetDevice)(&mut device), "cuCtxGetDevice")?;
        let mut name_buf = [0u8; 256];
        check(
            (drv.cuDeviceGetName)(name_buf.as_mut_ptr(), name_buf.len() as i32, device),
            "cuDeviceGetName",
        )?;
        let name = CStr::from_bytes_until_nul(&name_buf)
            .map_err(|_| "CUDA device name is not NUL-terminated")?
            .to_string_lossy()
            .into_owned();
        Ok(Self { drv, ctx, owns_context: false, device, name, ownership: Rc::new(()) })
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// Requires this exact context to be current on the calling thread.
    /// Another context on the same device is not interchangeable: modules
    /// and allocations belong to the context that created them. This check
    /// never changes the calling thread's current context.
    pub fn require_current(&self) -> Result<(), String> {
        let mut current = std::ptr::null_mut();
        unsafe {
            check((self.drv.cuCtxGetCurrent)(&mut current), "cuCtxGetCurrent")?;
        }
        if current != self.ctx {
            return Err("CUDA context is not current on the calling thread".into());
        }
        Ok(())
    }

    /// Queries the actual device UUID, driver's supported CUDA version, and
    /// Linux kernel driver build metadata. Missing symbols, failed queries,
    /// unsupported platforms, and unidentified values disable persistent
    /// caching through an error rather than a guessed identity.
    pub fn cache_identity(&self) -> Result<DeviceIdentity, String> {
        self.require_current()?;
        #[cfg(target_os = "linux")]
        {
            let file = std::fs::File::open("/proc/driver/nvidia/version")
                .map_err(|err| format!("NVIDIA driver build metadata is unavailable: {err}"))?;
            let driver_build = read_driver_build(file)?;
            query_cache_identity(
                self.device, self.drv.cuDeviceGetUuid, self.drv.cuDriverGetVersion, driver_build,
            )
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err("persistent CUDA cache identity requires Linux driver build metadata".into())
        }
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
        self.require_current()?;
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            check((self.drv.cuMemAlloc)(&mut ptr, len_bytes), "cuMemAlloc")?;
            Ok(DeviceBuffer { ptr, len_bytes, drv: self.drv, ownership: Rc::clone(&self.ownership) })
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
    /// This is the driver-side JIT path ordinary, unverified kernels take
    /// (cupy's `RawModule(path=...)`, the Python benchmark harnesses' loader,
    /// hands the driver the identical PTX text), so a candidate measured here
    /// is measured through the machinery it will actually run through - not
    /// through an offline `ptxas` binary that may be a different version than
    /// the installed driver's built-in compiler. Verified `exact_pv` uses
    /// `load_verified_exact_pv` to load its validated offline cubin instead.
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

    /// Load the exact offline cubin accepted by exact_pv translation validation.
    ///
    /// Requires an sm_89 device, an unchanged successful validation receipt,
    /// and matching SHA-256 hashes. Passes the checked ELF bytes directly to
    /// the driver with no JIT options. Any error is returned without PTX fallback.
    /// The returned generic handle carries artifact identity only; generic
    /// launch does not check arithmetic or launch premises. Prefer
    /// `load_checked_exact_pv` for the typed checked-launch interface.
    pub fn load_verified_exact_pv(
        &self,
        artifact: &crate::verified_exact_pv::ValidatedExactPv,
    ) -> Result<KernelModule, String> {
        self.require_current()?;
        // The CUDA API loads into the current context, which another caller
        // could have switched since this CudaContext was constructed.
        let (major, minor) = unsafe {
            let mut current_device = 0;
            check((self.drv.cuCtxGetDevice)(&mut current_device), "cuCtxGetDevice")?;
            let (mut major, mut minor) = (0, 0);
            check((self.drv.cuDeviceGetAttribute)(&mut major, 75, current_device), "CUDA compute capability major")?;
            check((self.drv.cuDeviceGetAttribute)(&mut minor, 76, current_device), "CUDA compute capability minor")?;
            (major, minor)
        };
        let (module, func) = load_verified_image(
            artifact, major, minor,
            self.drv.cuModuleLoadDataEx,
            self.drv.cuModuleGetFunction,
            self.drv.cuModuleUnload,
        )?;
        Ok(KernelModule { module, func, drv: self.drv })
    }

    /// Load a checked artifact without erasing its identity into a generic
    /// kernel handle. The returned module keeps this context borrowed/live.
    pub fn load_checked_exact_pv<'ctx>(
        &'ctx self,
        artifact: &crate::verified_exact_pv::ValidatedExactPv,
    ) -> Result<VerifiedExactPvModule<'ctx>, String> {
        let kernel = self.load_verified_exact_pv(artifact)?;
        self.require_current()?;
        Ok(VerifiedExactPvModule { context: self, kernel, cubin_sha256: artifact.cubin_sha256().to_owned() })
    }

    /// Execute the fixed exact_pv subject inside its checked nonempty domain.
    ///
    /// Shape, real owned allocation extents, alignment, output disjointness,
    /// context identity, device/function limits and the exact parameter ABI
    /// are checked before enqueueing. P/V may alias each other. Synchronizing
    /// before and after keeps referenced buffers/module/context live through
    /// successful completion and orders preceding work in this context. Any
    /// CUDA error returns no checked result; cleanup/error recovery still
    /// relies on driver semantics. External code
    /// must not concurrently write the inputs or destination. This is runtime
    /// enforcement of theorem premises, not an operational PTX proof.
    pub fn launch_checked_exact_pv(
        &self,
        kernel: &VerifiedExactPvModule<'_>,
        shape: crate::verified_exact_pv::ExactPvShape,
        p: &DeviceBuffer,
        v: &DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), String> {
        self.require_current()?;
        if !Rc::ptr_eq(&self.ownership, &kernel.context.ownership) {
            return Err("checked exact_pv module belongs to a different context wrapper".into());
        }
        for (name, buffer) in [("P", p), ("V", v), ("Out", &*out)] {
            if !Rc::ptr_eq(&self.ownership, &buffer.ownership) {
                return Err(format!("checked exact_pv {name} allocation belongs to a different context wrapper"));
            }
        }
        crate::verified_exact_pv::check_launch_buffers(shape,
            [(p.ptr, p.len_bytes), (v.ptr, v.len_bytes), (out.ptr, out.len_bytes)])?;

        let grid = shape.grid();
        let block = shape.block();
        // CUDA header enum ordinals: device MAX_THREADS=1, BLOCK_X/Y/Z=2/3/4,
        // GRID_X/Y/Z=5/6/7; function MAX_THREADS_PER_BLOCK=0.
        for (attribute, required, name) in [
            (1, block.0, "device threads per block"),
            (2, block.0, "block.x"), (3, block.1, "block.y"), (4, block.2, "block.z"),
            (5, grid.0, "grid.x"), (6, grid.1, "grid.y"), (7, grid.2, "grid.z"),
        ] {
            let mut cap = 0;
            unsafe { check((self.drv.cuDeviceGetAttribute)(&mut cap, attribute, self.device),
                "checked exact_pv device launch limit")?; }
            if cap <= 0 || required as u64 > cap as u64 {
                return Err(format!("checked exact_pv {name} exceeds the identified device limit"));
            }
        }
        let query = self.drv.cuFuncGetAttribute
            .ok_or("checked exact_pv function launch limit query is unavailable")?;
        let mut function_cap = 0;
        unsafe { check(query(&mut function_cap, 0, kernel.kernel.func),
            "checked exact_pv function launch limit")?; }
        if function_cap <= 0 || block.0 as u64 > function_cap as u64 {
            return Err("checked exact_pv block.x exceeds the identified function thread limit".into());
        }

        self.synchronize()?;
        self.require_current()?;
        let mut pointers = [p.ptr, v.ptr, out.ptr];
        let mut scalars = shape.scalar_parameters();
        let args: [*mut c_void; 9] = [
            (&mut pointers[0] as *mut CUdeviceptr).cast(),
            (&mut pointers[1] as *mut CUdeviceptr).cast(),
            (&mut pointers[2] as *mut CUdeviceptr).cast(),
            (&mut scalars[0] as *mut i32).cast(),
            (&mut scalars[1] as *mut i32).cast(),
            (&mut scalars[2] as *mut i32).cast(),
            (&mut scalars[3] as *mut i32).cast(),
            (&mut scalars[4] as *mut i32).cast(),
            (&mut scalars[5] as *mut i32).cast(),
        ];
        unsafe {
            check((self.drv.cuLaunchKernel)(kernel.kernel.func,
                grid.0, grid.1, grid.2, block.0, block.1, block.2,
                0, std::ptr::null_mut(), args.as_ptr(), std::ptr::null()),
                "cuLaunchKernel(checked exact_pv)")?;
        }
        self.synchronize()
    }

    /// Enqueues an unverified launch. `args` are the raw kernel parameter values in
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
        let launch = |arg_ptrs: &[*mut c_void]| unsafe {
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
        };
        // CUDA receives pointers to argument values, not the values themselves.
        // GEMM has exactly three parameters, whose owned copies and pointers
        // fit on the stack. Keep both arrays alive until the driver returns.
        if let [a, b, c] = args {
            let mut arg_values = [*a, *b, *c];
            let arg_ptrs: [*mut c_void; 3] = arg_values
                .each_mut()
                .map(|value| (value as *mut CUdeviceptr).cast());
            launch(&arg_ptrs)
        } else {
            // Other probe kernels retain support for any parameter count.
            let mut arg_values: Vec<CUdeviceptr> = args.to_vec();
            let arg_ptrs: Vec<*mut c_void> = arg_values
                .iter_mut()
                .map(|value| (value as *mut CUdeviceptr).cast())
                .collect();
            launch(&arg_ptrs)
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
        if self.owns_context && !self.ctx.is_null() {
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
    use crate::verified_exact_pv::test_support::Bundle;
    use crate::verified_exact_pv::ValidatedExactPv;
    use std::cell::RefCell;

    #[derive(Default)]
    struct ContextSpy {
        current: usize,
        failed_query: &'static str,
        unterminated_name: bool,
        destroys: Vec<usize>,
        unexpected_calls: usize,
        queried_devices: Vec<CUdevice>,
    }

    thread_local! {
        static CONTEXT_SPY: RefCell<ContextSpy> = RefCell::new(ContextSpy::default());
    }

    unsafe extern "C" fn context_test_current(ctx: *mut *mut c_void) -> CUresult {
        CONTEXT_SPY.with(|spy| {
            let spy = spy.borrow();
            *ctx = spy.current as *mut c_void;
            if spy.failed_query == "current" { 999 } else { CUDA_SUCCESS }
        })
    }

    unsafe extern "C" fn context_test_device(device: *mut CUdevice) -> CUresult {
        *device = 7;
        CONTEXT_SPY.with(|spy| {
            if spy.borrow().failed_query == "device" { 999 } else { CUDA_SUCCESS }
        })
    }

    unsafe extern "C" fn context_test_name(name: *mut u8, len: i32, device: CUdevice) -> CUresult {
        CONTEXT_SPY.with(|spy| {
            let mut spy = spy.borrow_mut();
            spy.queried_devices.push(device);
            let name = std::slice::from_raw_parts_mut(name, len as usize);
            if spy.unterminated_name {
                name.fill(b'x');
            } else {
                name[..9].copy_from_slice(b"external\0");
            }
            if spy.failed_query == "name" { 999 } else { CUDA_SUCCESS }
        })
    }

    unsafe extern "C" fn context_test_destroy(ctx: *mut c_void) -> CUresult {
        CONTEXT_SPY.with(|spy| spy.borrow_mut().destroys.push(ctx as usize));
        CUDA_SUCCESS
    }

    fn context_test_driver() -> Driver {
        macro_rules! unused {
            ($($ty:ty),* $(,)?) => {{
                unsafe extern "C" fn unexpected($(_: $ty),*) -> CUresult {
                    CONTEXT_SPY.with(|spy| spy.borrow_mut().unexpected_calls += 1);
                    999
                }
                unexpected
            }};
        }
        Driver {
            cuInit: unused!(u32),
            cuDeviceGetCount: unused!(*mut i32),
            cuDeviceGet: unused!(*mut CUdevice, i32),
            cuDeviceGetName: context_test_name,
            cuDeviceGetAttribute: unused!(*mut i32, i32, CUdevice),
            cuDeviceGetUuid: None,
            cuDriverGetVersion: None,
            cuCtxCreate: unused!(*mut *mut c_void, u32, CUdevice),
            cuCtxDestroy: context_test_destroy,
            cuCtxSynchronize: unused!(),
            cuCtxGetCurrent: context_test_current,
            cuCtxGetDevice: context_test_device,
            cuModuleLoadDataEx: unused!(*mut *mut c_void, *const c_void, u32, *mut i32, *mut *mut c_void),
            cuModuleUnload: unused!(*mut c_void),
            cuModuleGetFunction: unused!(*mut *mut c_void, *mut c_void, *const u8),
            cuFuncSetAttribute: unused!(*mut c_void, i32, i32),
            cuFuncGetAttribute: None,
            cuMemAlloc: unused!(*mut CUdeviceptr, usize),
            cuMemFree: unused!(CUdeviceptr),
            cuMemGetInfo: unused!(*mut usize, *mut usize),
            cuMemsetD8: unused!(CUdeviceptr, u8, usize),
            cuMemcpyHtoD: unused!(CUdeviceptr, *const c_void, usize),
            cuMemcpyDtoH: unused!(*mut c_void, CUdeviceptr, usize),
            cuMemcpyDtoD: unused!(CUdeviceptr, CUdeviceptr, usize),
            cuLaunchKernel: unused!(
                *mut c_void, u32, u32, u32, u32, u32, u32, u32,
                *mut c_void, *const *mut c_void, *const *mut c_void,
            ),
            cuEventCreate: unused!(*mut *mut c_void, u32),
            cuEventRecord: unused!(*mut c_void, *mut c_void),
            cuEventSynchronize: unused!(*mut c_void),
            cuEventElapsedTime: unused!(*mut f32, *mut c_void, *mut c_void),
            cuEventDestroy: unused!(*mut c_void),
        }
    }

    fn reset_context_spy() {
        CONTEXT_SPY.with(|spy| *spy.borrow_mut() = ContextSpy {
            current: 0x1234,
            ..Default::default()
        });
    }

    #[test]
    fn borrowed_context_preserves_host_context_and_device() {
        reset_context_spy();
        let ctx = unsafe { CudaContext::borrow_current_from_driver(context_test_driver()) }.unwrap();
        assert_eq!(ctx.ctx as usize, 0x1234);
        assert_eq!(ctx.device, 7);
        assert_eq!(ctx.device_name(), "external");
        ctx.require_current().unwrap();
        // A different context on the same device cannot use these resources.
        CONTEXT_SPY.with(|spy| spy.borrow_mut().current = 0x5678);
        assert!(ctx.require_current().is_err());
        CONTEXT_SPY.with(|spy| spy.borrow_mut().current = 0x1234);
        drop(ctx);
        CONTEXT_SPY.with(|spy| {
            let spy = spy.borrow();
            assert_eq!(spy.current, 0x1234);
            assert_eq!(spy.queried_devices, vec![7]);
            assert!(spy.destroys.is_empty());
            assert_eq!(spy.unexpected_calls, 0);
        });
    }

    #[test]
    fn borrowed_context_rejects_missing_and_failed_context_queries() {
        for failed_query in ["missing", "current", "device", "name"] {
            reset_context_spy();
            CONTEXT_SPY.with(|spy| {
                let mut spy = spy.borrow_mut();
                spy.failed_query = failed_query;
                if failed_query == "missing" {
                    spy.current = 0;
                }
            });
            assert!(unsafe {
                CudaContext::borrow_current_from_driver(context_test_driver())
            }.is_err());
            CONTEXT_SPY.with(|spy| {
                let spy = spy.borrow();
                assert!(spy.destroys.is_empty());
                assert_eq!(spy.unexpected_calls, 0);
            });
        }
    }

    #[test]
    fn borrowed_context_rejects_unterminated_device_name() {
        reset_context_spy();
        CONTEXT_SPY.with(|spy| spy.borrow_mut().unterminated_name = true);
        let error = unsafe {
            CudaContext::borrow_current_from_driver(context_test_driver())
        }.err().unwrap();
        assert!(error.contains("NUL-terminated"));
        CONTEXT_SPY.with(|spy| assert!(spy.borrow().destroys.is_empty()));
    }

    #[test]
    fn context_drop_only_destroys_owned_contexts() {
        reset_context_spy();
        let mut ctx = unsafe { CudaContext::borrow_current_from_driver(context_test_driver()) }.unwrap();
        // Exercise the same ownership state produced by CudaContext::new().
        ctx.owns_context = true;
        drop(ctx);
        CONTEXT_SPY.with(|spy| assert_eq!(spy.borrow().destroys, vec![0x1234]));
    }

    unsafe extern "C" fn cache_test_uuid(uuid: *mut CUuuid, device: CUdevice) -> CUresult {
        if device != 7 {
            return 101;
        }
        (*uuid).bytes = [0xAB; 16];
        CUDA_SUCCESS
    }

    unsafe extern "C" fn cache_test_version(version: *mut i32) -> CUresult {
        *version = 13020;
        CUDA_SUCCESS
    }

    unsafe extern "C" fn cache_test_unidentified_uuid(_: *mut CUuuid, _: CUdevice) -> CUresult {
        CUDA_SUCCESS
    }

    unsafe extern "C" fn cache_test_unidentified_version(_: *mut i32) -> CUresult {
        CUDA_SUCCESS
    }

    unsafe extern "C" fn cache_test_failed_version(_: *mut i32) -> CUresult {
        999
    }

    #[test]
    fn cache_identity_preserves_device_uuid_and_driver_versions() {
        let identity = query_cache_identity(
            7, Some(cache_test_uuid), Some(cache_test_version), "driver build".into(),
        ).unwrap();
        assert_eq!(identity, DeviceIdentity {
            uuid: [0xAB; 16], driver_version: 13020, driver_build: "driver build".into(),
        });
        assert_eq!(std::mem::size_of::<CUuuid>(), 16);
    }

    #[test]
    fn cache_identity_rejects_unavailable_optional_symbols() {
        assert!(query_cache_identity(
            7, None, Some(cache_test_version), "driver build".into(),
        ).unwrap_err().contains("UUID query is unavailable"));
        assert!(query_cache_identity(
            7, Some(cache_test_uuid), None, "driver build".into(),
        ).unwrap_err().contains("version query is unavailable"));
    }

    #[test]
    fn cache_identity_rejects_failed_driver_queries() {
        assert!(query_cache_identity(
            8, Some(cache_test_uuid), Some(cache_test_version), "driver build".into(),
        ).unwrap_err().contains("cuDeviceGetUuid"));
        assert!(query_cache_identity(
            7, Some(cache_test_uuid), Some(cache_test_failed_version), "driver build".into(),
        ).unwrap_err().contains("cuDriverGetVersion"));
    }

    #[test]
    fn cache_identity_rejects_unidentified_success_results() {
        assert!(query_cache_identity(
            7, Some(cache_test_unidentified_uuid), Some(cache_test_version), "driver build".into(),
        ).unwrap_err().contains("unidentified device UUID"));
        assert!(query_cache_identity(
            7, Some(cache_test_uuid), Some(cache_test_unidentified_version), "driver build".into(),
        ).unwrap_err().contains("invalid driver version"));
    }

    #[test]
    fn cache_identity_driver_build_is_bounded_and_validated() {
        assert_eq!(read_driver_build(&b" driver build 123\n"[..]).unwrap(), "driver build 123");
        assert!(read_driver_build(&b" \n\t"[..]).is_err());
        assert!(read_driver_build(&b"build\0version"[..]).is_err());
        assert!(read_driver_build(&b"\xff"[..]).is_err());
        assert!(read_driver_build(&vec![b'x'; 16 * 1024][..]).is_ok());
        assert!(read_driver_build(&vec![b'x'; 16 * 1024 + 1][..]).unwrap_err().contains("exceeds"));
    }

    #[test]
    fn cache_identity_driver_build_propagates_read_errors() {
        struct FailedReader;
        impl Read for FailedReader {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"))
            }
        }
        assert!(read_driver_build(FailedReader).unwrap_err().contains("denied"));
    }

    #[derive(Default)]
    struct LaunchSpy {
        arg_count: usize,
        args: Vec<CUdeviceptr>,
        arg_addresses: Vec<usize>,
        geometry: [u32; 7],
        function: usize,
        stream: usize,
        extra: usize,
        calls: usize,
        result: CUresult,
    }

    thread_local! {
        static LAUNCH_SPY: RefCell<LaunchSpy> = RefCell::new(LaunchSpy::default());
    }

    unsafe extern "C" fn launch_test_kernel(
        function: *mut c_void,
        grid_x: u32, grid_y: u32, grid_z: u32,
        block_x: u32, block_y: u32, block_z: u32,
        shared_bytes: u32,
        stream: *mut c_void,
        args: *const *mut c_void,
        extra: *const *mut c_void,
    ) -> CUresult {
        LAUNCH_SPY.with(|spy| {
            let mut spy = spy.borrow_mut();
            spy.calls += 1;
            spy.function = function as usize;
            spy.geometry = [grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes];
            spy.stream = stream as usize;
            spy.extra = extra as usize;
            for i in 0..spy.arg_count {
                let value = (*args.add(i)).cast::<CUdeviceptr>();
                spy.arg_addresses.push(value as usize);
                spy.args.push(*value);
                // Stress the owned argument storage, never the caller's slice.
                *value = 0;
            }
            spy.result
        })
    }

    fn launch_test_context_and_kernel() -> (CudaContext, KernelModule) {
        let mut drv = context_test_driver();
        drv.cuLaunchKernel = launch_test_kernel;
        let ctx = CudaContext {
            drv,
            ctx: std::ptr::null_mut(),
            owns_context: false,
            device: 0,
            name: "launch spy".into(),
            ownership: Rc::new(()),
        };
        let kernel = KernelModule {
            module: std::ptr::null_mut(),
            func: 0x4321usize as *mut c_void,
            drv,
        };
        (ctx, kernel)
    }

    #[test]
    fn launch_marshals_owned_pointer_arguments_for_three_and_other_counts() {
        let (ctx, kernel) = launch_test_context_and_kernel();
        for args in [
            vec![],
            vec![0x1234],
            vec![0x1020304050607080, u64::MAX, 0],
            (0..17).map(|i| 0x1000 + i * 16).collect(),
        ] {
            let original = args.clone();
            LAUNCH_SPY.with(|spy| *spy.borrow_mut() = LaunchSpy {
                arg_count: args.len(),
                ..Default::default()
            });
            ctx.launch(&kernel, (2, 3, 4), (5, 6, 7), 8192, &args).unwrap();
            assert_eq!(args, original, "driver argument storage must be an owned copy");
            LAUNCH_SPY.with(|spy| {
                let spy = spy.borrow();
                assert_eq!(spy.calls, 1);
                assert_eq!(spy.args, original);
                assert_eq!(spy.function, 0x4321);
                assert_eq!(spy.geometry, [2, 3, 4, 5, 6, 7, 8192]);
                assert_eq!(spy.stream, 0);
                assert_eq!(spy.extra, 0);
                for (i, address) in spy.arg_addresses.iter().enumerate() {
                    assert_ne!(*address, 0);
                    assert_eq!(address % std::mem::align_of::<CUdeviceptr>(), 0);
                    assert_ne!(*address, &args[i] as *const CUdeviceptr as usize);
                    assert!(!spy.arg_addresses[..i].contains(address));
                }
            });
        }
    }

    #[test]
    fn launch_preserves_driver_errors_for_stack_and_fallback_arguments() {
        let (ctx, kernel) = launch_test_context_and_kernel();
        for args in [vec![16, 32, 48], vec![16, 32, 48, 64]] {
            let original = args.clone();
            LAUNCH_SPY.with(|spy| *spy.borrow_mut() = LaunchSpy {
                arg_count: args.len(),
                result: 201,
                ..Default::default()
            });
            assert_eq!(
                ctx.launch(&kernel, (1, 1, 1), (32, 1, 1), 0, &args).unwrap_err(),
                "cuLaunchKernel failed (CUresult 201)",
            );
            assert_eq!(args, original);
            LAUNCH_SPY.with(|spy| {
                let spy = spy.borrow();
                assert_eq!(spy.calls, 1);
                assert_eq!(spy.args, original);
            });
        }
    }

    #[derive(Default)]
    struct LoadSpy {
        expected: Vec<u8>,
        loads: usize,
        lookups: usize,
        unloads: usize,
        cubin_only: bool,
        lookup_exact_pv: bool,
        reject_load: bool,
    }

    thread_local! {
        static LOAD_SPY: RefCell<LoadSpy> = RefCell::new(LoadSpy::default());
    }

    unsafe extern "C" fn record_load(
        module: *mut *mut c_void, image: *const c_void, count: u32,
        options: *mut i32, values: *mut *mut c_void,
    ) -> CUresult {
        LOAD_SPY.with(|spy| {
            let mut spy = spy.borrow_mut();
            spy.loads += 1;
            let bytes = std::slice::from_raw_parts(image.cast::<u8>(), spy.expected.len());
            spy.cubin_only = bytes == spy.expected && bytes.starts_with(b"\x7fELF")
                && count == 0 && options.is_null() && values.is_null();
            if spy.reject_load {
                return 200;
            }
            *module = 1usize as *mut c_void;
            CUDA_SUCCESS
        })
    }

    unsafe extern "C" fn record_lookup(func: *mut *mut c_void, _: *mut c_void, entry: *const u8) -> CUresult {
        LOAD_SPY.with(|spy| {
            let mut spy = spy.borrow_mut();
            spy.lookups += 1;
            spy.lookup_exact_pv = CStr::from_ptr(entry.cast()).to_bytes() == b"exact_pv";
        });
        *func = 2usize as *mut c_void;
        CUDA_SUCCESS
    }

    unsafe extern "C" fn record_unload(_: *mut c_void) -> CUresult {
        LOAD_SPY.with(|spy| spy.borrow_mut().unloads += 1);
        CUDA_SUCCESS
    }

    fn load_fixture(artifact: &ValidatedExactPv, major: i32, minor: i32) -> Result<(*mut c_void, *mut c_void), String> {
        load_verified_image(artifact, major, minor, record_load, record_lookup, record_unload)
    }

    #[test]
    fn verified_exact_pv_loads_the_identical_cubin_without_ptx_jit() {
        let bundle = Bundle::new();
        LOAD_SPY.with(|spy| *spy.borrow_mut() = LoadSpy { expected: bundle.cubin.clone(), ..Default::default() });
        let artifact = bundle.open();
        let handles = load_fixture(&artifact, 8, 9).unwrap();
        assert_eq!(handles, (1usize as *mut c_void, 2usize as *mut c_void));
        LOAD_SPY.with(|spy| {
            let spy = spy.borrow();
            assert_eq!((spy.loads, spy.lookups), (1, 1));
            assert!(spy.cubin_only && spy.lookup_exact_pv);
        });
        // A failed binary load must return the failure without a second load
        // (in particular, without submitting PTX as a fallback).
        LOAD_SPY.with(|spy| spy.borrow_mut().reject_load = true);
        assert!(load_fixture(&artifact, 8, 9).is_err());
        LOAD_SPY.with(|spy| assert_eq!((spy.borrow().loads, spy.borrow().lookups), (2, 1)));
    }

    #[test]
    fn verified_exact_pv_rejects_mutation_target_and_failed_validation_before_loading() {
        let bundle = Bundle::new();
        LOAD_SPY.with(|spy| *spy.borrow_mut() = LoadSpy { expected: bundle.cubin.clone(), ..Default::default() });
        let artifact = bundle.open();
        assert!(load_fixture(&artifact, 9, 0).is_err());
        let mut modified = bundle.cubin.clone();
        *modified.last_mut().unwrap() ^= 1;
        std::fs::write(bundle.directory.join("exact_pv.cubin"), modified).unwrap();
        assert!(load_fixture(&artifact, 8, 9).is_err());
        std::fs::write(bundle.directory.join("exact_pv.cubin"), &bundle.cubin).unwrap();
        bundle.write_receipt("REFUSED");
        assert!(ValidatedExactPv::open(&bundle.directory, None).is_err());
        assert!(load_fixture(&artifact, 8, 9).is_err());
        std::fs::remove_file(bundle.directory.join("receipt.txt")).unwrap();
        assert!(load_fixture(&artifact, 8, 9).is_err());
        LOAD_SPY.with(|spy| assert_eq!(spy.borrow().loads, 0));
    }

    mod exact_pv_checked_tests {
        use super::*;
        use crate::verified_exact_pv::ExactPvShape;
        use std::collections::VecDeque;

        #[derive(Default)]
        struct CheckedSpy {
            caps: [i32; 8],
            function_cap: i32,
            failed_attribute: Option<i32>,
            function_error: CUresult,
            addresses: VecDeque<u64>,
            allocations: Vec<usize>,
            pointers: Vec<u64>,
            scalars: Vec<i32>,
            argument_addresses: Vec<usize>,
            geometry: [u32; 7],
            events: Vec<&'static str>,
            launches: usize,
            synchronizations: usize,
            launch_error: CUresult,
            failed_synchronization: usize,
        }

        thread_local! {
            static CHECKED_SPY: RefCell<CheckedSpy> = RefCell::new(CheckedSpy::default());
        }

        #[test]
        fn memory_bindings_refuse_legacy_only_size_or_pointer_abis() {
            for name in ["cuMemAlloc", "cuMemFree", "cuMemGetInfo", "cuMemsetD8",
                         "cuMemcpyHtoD", "cuMemcpyDtoH", "cuMemcpyDtoD"] {
                let mut asked = Vec::new();
                let legacy = required_memory_symbol(name, |symbol| {
                    asked.push(symbol.to_owned());
                    (symbol == name).then_some(1usize as *mut c_void)
                });
                assert!(legacy.is_none(), "legacy {name} must not establish v2 ABI widths");
                assert_eq!(asked, [format!("{name}_v2")]);
                let modern = required_memory_symbol(name, |symbol| {
                    (symbol == format!("{name}_v2")).then_some(2usize as *mut c_void)
                });
                assert_eq!(modern, Some(2usize as *mut c_void));
            }
        }

        fn reset() {
            reset_context_spy();
            CHECKED_SPY.with(|spy| *spy.borrow_mut() = CheckedSpy {
                caps: [0, 1024, 1024, 1024, 64, i32::MAX, 65535, 65535],
                function_cap: 1024,
                addresses: [0x1000, 0x2000, 0x3000].into_iter().collect(),
                ..Default::default()
            });
        }

        unsafe extern "C" fn attribute(value: *mut i32, attr: i32, device: CUdevice) -> CUresult {
            assert_eq!(device, 7);
            CHECKED_SPY.with(|spy| {
                let spy = spy.borrow();
                if spy.failed_attribute == Some(attr) { return 999; }
                *value = match attr {
                    75 => 8, 76 => 9,
                    1..=7 => spy.caps[attr as usize],
                    _ => panic!("unexpected checked-launch attribute {attr}"),
                };
                CUDA_SUCCESS
            })
        }

        unsafe extern "C" fn function_attribute(value: *mut i32, attr: i32, func: *mut c_void) -> CUresult {
            assert_eq!(attr, 0);
            assert_eq!(func as usize, 2);
            CHECKED_SPY.with(|spy| {
                let spy = spy.borrow();
                *value = spy.function_cap;
                spy.function_error
            })
        }

        unsafe extern "C" fn allocate(ptr: *mut CUdeviceptr, bytes: usize) -> CUresult {
            CHECKED_SPY.with(|spy| {
                let mut spy = spy.borrow_mut();
                spy.allocations.push(bytes);
                *ptr = spy.addresses.pop_front().expect("unplanned allocation");
            });
            CUDA_SUCCESS
        }

        unsafe extern "C" fn free(_: CUdeviceptr) -> CUresult { CUDA_SUCCESS }

        unsafe extern "C" fn synchronize() -> CUresult {
            CHECKED_SPY.with(|spy| {
                let mut spy = spy.borrow_mut();
                spy.events.push("synchronize");
                spy.synchronizations += 1;
                if spy.failed_synchronization == spy.synchronizations { 999 } else { CUDA_SUCCESS }
            })
        }

        unsafe extern "C" fn launch(
            func: *mut c_void, gx: u32, gy: u32, gz: u32,
            bx: u32, by: u32, bz: u32, shared: u32,
            stream: *mut c_void, args: *const *mut c_void, extra: *const *mut c_void,
        ) -> CUresult {
            assert_eq!(func as usize, 2);
            assert!(stream.is_null() && extra.is_null());
            CHECKED_SPY.with(|spy| {
                let mut spy = spy.borrow_mut();
                spy.events.push("launch");
                spy.launches += 1;
                spy.geometry = [gx, gy, gz, bx, by, bz, shared];
                for i in 0..9 {
                    let address = *args.add(i);
                    spy.argument_addresses.push(address as usize);
                    if i < 3 {
                        spy.pointers.push(*address.cast::<u64>());
                        *address.cast::<u64>() = 0;
                    } else {
                        spy.scalars.push(*address.cast::<i32>());
                        *address.cast::<i32>() = 0;
                    }
                }
                spy.launch_error
            })
        }

        fn context(function_query: bool) -> CudaContext {
            let mut drv = context_test_driver();
            drv.cuDeviceGetAttribute = attribute;
            drv.cuFuncGetAttribute = function_query.then_some(function_attribute);
            drv.cuModuleLoadDataEx = record_load;
            drv.cuModuleGetFunction = record_lookup;
            drv.cuModuleUnload = record_unload;
            drv.cuMemAlloc = allocate;
            drv.cuMemFree = free;
            drv.cuCtxSynchronize = synchronize;
            drv.cuLaunchKernel = launch;
            unsafe { CudaContext::borrow_current_from_driver(drv).unwrap() }
        }

        fn bundle() -> Bundle {
            let bundle = Bundle::new();
            LOAD_SPY.with(|spy| *spy.borrow_mut() = LoadSpy {
                expected: bundle.cubin.clone(), ..Default::default()
            });
            bundle
        }

        fn buffers(ctx: &CudaContext, lengths: [usize; 3]) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
            (ctx.alloc(lengths[0]).unwrap(), ctx.alloc(lengths[1]).unwrap(), ctx.alloc(lengths[2]).unwrap())
        }

        fn assert_not_launched() {
            CHECKED_SPY.with(|spy| {
                let spy = spy.borrow();
                assert_eq!(spy.launches, 0);
                assert_eq!(spy.synchronizations, 0);
            });
        }

        #[test]
        fn checked_launch_uses_the_reviewed_module_exact_geometry_and_typed_parameter_abi() {
            reset();
            let bundle = bundle();
            let ctx = context(true);
            let kernel = ctx.load_checked_exact_pv(&bundle.open()).unwrap();
            let shape = ExactPvShape::new(2, 3, 4, 5).unwrap();
            let (p, v, mut out) = buffers(&ctx, shape.required_bytes());
            ctx.launch_checked_exact_pv(&kernel, shape, &p, &v, &mut out).unwrap();
            assert_eq!(shape.dimensions(), [2, 3, 4, 5]);
            assert_eq!(kernel.cubin_sha256(), bundle.open().cubin_sha256());
            CHECKED_SPY.with(|spy| {
                let spy = spy.borrow();
                assert_eq!(spy.pointers, [0x1000, 0x2000, 0x3000]);
                assert_eq!(spy.scalars, [4, 5, 3, 24, 40, 30]);
                assert_eq!(spy.geometry, [3, 2, 1, 5, 1, 1, 0]);
                assert_eq!(spy.events, ["synchronize", "launch", "synchronize"]);
                for i in 0..9 {
                    assert_ne!(spy.argument_addresses[i], 0);
                    assert_eq!(spy.argument_addresses[i] % if i < 3 { 8 } else { 4 }, 0);
                    assert!(!spy.argument_addresses[..i].contains(&spy.argument_addresses[i]));
                }
                for i in 4..9 {
                    assert_eq!(spy.argument_addresses[i] - spy.argument_addresses[i - 1], 4,
                        "scalar ABI storage must be actual i32 values, not u64 pointer slots");
                }
            });
            LOAD_SPY.with(|spy| assert!(spy.borrow().cubin_only && spy.borrow().lookup_exact_pv));
        }

        #[test]
        fn checked_launch_preserves_read_only_input_aliasing_and_touching_output_boundaries() {
            reset();
            CHECKED_SPY.with(|spy| spy.borrow_mut().addresses = [0x1000, 0x1008].into_iter().collect());
            let bundle = bundle();
            let ctx = context(true);
            let kernel = ctx.load_checked_exact_pv(&bundle.open()).unwrap();
            let shape = ExactPvShape::new(1, 1, 2, 1).unwrap();
            let p = ctx.alloc(shape.required_bytes()[0]).unwrap();
            let mut out = ctx.alloc(shape.required_bytes()[2]).unwrap();
            ctx.launch_checked_exact_pv(&kernel, shape, &p, &p, &mut out).unwrap();
            CHECKED_SPY.with(|spy| assert_eq!(spy.borrow().pointers, [0x1000, 0x1000, 0x1008]));
        }

        #[test]
        fn checked_launch_refuses_short_misaligned_wrapping_or_overlapping_allocations() {
            let shape = ExactPvShape::new(1, 1, 2, 1).unwrap();
            let normal = shape.required_bytes();
            let mut cases = vec![
                ([0, 0x2000, 0x3000], normal, "null"),
                ([0x1001, 0x2000, 0x3000], normal, "misaligned"),
                ([0x1000, 0x2000, 0x3004], normal, "misaligned"),
                ([u64::MAX - 3, 0x2000, 0x3000], normal, "wraps"),
                ([0x1000, u64::MAX, 0x3000], normal, "wraps"),
                ([0x1000, 0x2000, u64::MAX - 7], normal, "wraps"),
                ([0x1000, 0x2000, 0x1000], normal, "overlaps"),
                ([0x1000, 0x2000, 0x2000], normal, "overlaps"),
            ];
            for i in 0..3 {
                let mut lengths = normal;
                lengths[i] -= 1;
                cases.push(([0x1000, 0x2000, 0x3000], lengths, "shorter"));
            }
            for (addresses, lengths, diagnostic) in cases {
                reset();
                CHECKED_SPY.with(|spy| spy.borrow_mut().addresses = addresses.into_iter().collect());
                let bundle = bundle();
                let ctx = context(true);
                let kernel = ctx.load_checked_exact_pv(&bundle.open()).unwrap();
                let (p, v, mut out) = buffers(&ctx, lengths);
                let error = ctx.launch_checked_exact_pv(&kernel, shape, &p, &v, &mut out).unwrap_err();
                assert!(error.contains(diagnostic), "{error}");
                assert_not_launched();
            }
        }

        #[test]
        fn checked_launch_requires_the_originating_current_context_and_allocation_generation() {
            reset();
            let bundle = bundle();
            let ctx = context(true);
            let other = context(true); // Same raw context/device, independent wrapper generation.
            let kernel = ctx.load_checked_exact_pv(&bundle.open()).unwrap();
            let wrong_kernel = other.load_checked_exact_pv(&bundle.open()).unwrap();
            let shape = ExactPvShape::new(1, 1, 2, 1).unwrap();
            let (p, v, mut out) = buffers(&ctx, shape.required_bytes());
            assert!(ctx.launch_checked_exact_pv(&wrong_kernel, shape, &p, &v, &mut out)
                .unwrap_err().contains("module belongs"));
            CHECKED_SPY.with(|spy| spy.borrow_mut().addresses.extend([0x4000, 0x5000, 0x6000]));
            let foreign_p = other.alloc(shape.required_bytes()[0]).unwrap();
            let foreign_v = other.alloc(shape.required_bytes()[1]).unwrap();
            let mut foreign_out = other.alloc(shape.required_bytes()[2]).unwrap();
            assert!(ctx.launch_checked_exact_pv(&kernel, shape, &foreign_p, &v, &mut out)
                .unwrap_err().contains("allocation belongs"));
            assert!(ctx.launch_checked_exact_pv(&kernel, shape, &p, &foreign_v, &mut out)
                .unwrap_err().contains("allocation belongs"));
            assert!(ctx.launch_checked_exact_pv(&kernel, shape, &p, &v, &mut foreign_out)
                .unwrap_err().contains("allocation belongs"));
            CONTEXT_SPY.with(|spy| spy.borrow_mut().current = 0x5678);
            assert!(ctx.launch_checked_exact_pv(&kernel, shape, &p, &v, &mut out)
                .unwrap_err().contains("not current"));
            assert_not_launched();
        }

        #[test]
        fn checked_launch_refuses_unidentified_or_insufficient_device_and_function_limits() {
            for bad in 0..16 {
                reset();
                let bundle = bundle();
                let ctx = context(bad != 8);
                let kernel = ctx.load_checked_exact_pv(&bundle.open()).unwrap();
                let shape = ExactPvShape::new(2, 3, 4, 5).unwrap();
                let (p, v, mut out) = buffers(&ctx, shape.required_bytes());
                CHECKED_SPY.with(|spy| {
                    let mut spy = spy.borrow_mut();
                    match bad {
                        0..=6 => spy.caps[bad + 1] = 0,
                        7 => spy.failed_attribute = Some(5),
                        8 => {}, // Optional entry point absent.
                        9 => spy.function_error = 999,
                        10 => spy.function_cap = 0,
                        11 => spy.function_cap = 4,
                        12 => spy.caps[1] = 4,
                        13 => spy.caps[2] = 4,
                        14 => spy.caps[5] = 2,
                        15 => spy.caps[6] = 1,
                        _ => unreachable!(),
                    }
                });
                assert!(ctx.launch_checked_exact_pv(&kernel, shape, &p, &v, &mut out).is_err());
                assert_not_launched();
            }
        }

        #[test]
        fn checked_launch_propagates_pre_launch_launch_and_completion_errors_without_fallback() {
            for bad in 0..3 {
                reset();
                let bundle = bundle();
                let ctx = context(true);
                let kernel = ctx.load_checked_exact_pv(&bundle.open()).unwrap();
                let shape = ExactPvShape::new(1, 1, 2, 1).unwrap();
                let (p, v, mut out) = buffers(&ctx, shape.required_bytes());
                CHECKED_SPY.with(|spy| {
                    let mut spy = spy.borrow_mut();
                    if bad == 1 { spy.launch_error = 999; }
                    else { spy.failed_synchronization = if bad == 0 { 1 } else { 2 }; }
                });
                assert!(ctx.launch_checked_exact_pv(&kernel, shape, &p, &v, &mut out).is_err());
                CHECKED_SPY.with(|spy| {
                    let spy = spy.borrow();
                    assert_eq!(spy.launches, usize::from(bad != 0));
                    assert_eq!(spy.synchronizations, if bad == 2 { 2 } else { 1 });
                });
                LOAD_SPY.with(|spy| assert_eq!(spy.borrow().loads, 1));
            }
        }
    }

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
