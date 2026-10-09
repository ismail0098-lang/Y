//! Dynamically loaded LLVM C API. No build-time LLVM or new crate dependency.
#![allow(non_snake_case)]

use super::JitError;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::sync::OnceLock;

pub type Ref = *mut c_void;

/// Optional observational APIs; missing symbols leave lookup timing intact.
pub struct ObjectObserverApi {
    pub get_layer: unsafe extern "C" fn(Ref) -> Ref,
    pub set_transform: unsafe extern "C" fn(Ref, unsafe extern "C" fn(Ref, *mut Ref) -> Ref, Ref),
    pub buffer_size: unsafe extern "C" fn(Ref) -> usize,
}

#[repr(C)]
pub struct SymbolFlags {
    pub generic: u8,
    pub target: u8,
}
#[repr(C)]
pub struct EvaluatedSymbol {
    pub address: u64,
    pub flags: SymbolFlags,
}
#[repr(C)]
pub struct SymbolPair {
    pub name: Ref,
    pub symbol: EvaluatedSymbol,
}

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const u8, flags: c_int) -> Ref;
    fn dlsym(handle: Ref, name: *const u8) -> Ref;
    fn dlerror() -> *const c_char;
    fn dlclose(handle: Ref) -> c_int;
}

macro_rules! api {
    ($($name:ident: ($($arg:ty),*) -> $ret:ty),* $(,)?) => {
        pub struct Api {
            _library: Library,
            pub context_from_llvm: Option<unsafe extern "C" fn(Ref) -> Ref>,
            pub context_get_llvm: Option<unsafe extern "C" fn(Ref) -> Ref>,
            pub object_observer: Option<ObjectObserverApi>,
            $(pub $name: unsafe extern "C" fn($($arg),*) -> $ret,)*
        }
        impl Api {
            unsafe fn load(library: Library) -> Result<Self, JitError> {
                // Check the reported version before loading the rest of the
                // entrypoints, and never initialize a rejected LLVM build.
                let get_version = std::mem::transmute::<Ref, unsafe extern "C" fn(*mut u32, *mut u32, *mut u32)>(
                    library.symbol(b"LLVMGetVersion\0")?);
                let (mut major, mut minor, mut patch) = (0, 0, 0);
                get_version(&mut major, &mut minor, &mut patch);
                if major < 17 {
                    return Err(JitError::new(format!(
                        "LLVM {major}.{minor}.{patch} is too old; need LLVM 17+"
                    )));
                }
                $(let $name = std::mem::transmute::<Ref, unsafe extern "C" fn($($arg),*) -> $ret>(
                    library.symbol(concat!(stringify!($name), "\0").as_bytes())?);)*
                // LLVM 23 replaced the context getter with a constructor that
                // adopts a caller-created context. Support both C APIs.
                let context_from_llvm = library.optional_symbol(b"LLVMOrcCreateNewThreadSafeContextFromLLVMContext\0")
                    .map(|p| std::mem::transmute::<Ref, unsafe extern "C" fn(Ref) -> Ref>(p));
                let context_get_llvm = library.optional_symbol(b"LLVMOrcThreadSafeContextGetContext\0")
                    .map(|p| std::mem::transmute::<Ref, unsafe extern "C" fn(Ref) -> Ref>(p));
                if context_from_llvm.is_none() && context_get_llvm.is_none() {
                    return Err(JitError::new("LLVM is missing ORC context access APIs"));
                }
                let object_observer = match (
                    library.optional_symbol(b"LLVMOrcLLJITGetObjTransformLayer\0"),
                    library.optional_symbol(b"LLVMOrcObjectTransformLayerSetTransform\0"),
                    library.optional_symbol(b"LLVMGetBufferSize\0"),
                ) {
                    (Some(get), Some(set), Some(size)) => Some(ObjectObserverApi {
                        get_layer: std::mem::transmute::<Ref, unsafe extern "C" fn(Ref) -> Ref>(get),
                        set_transform: std::mem::transmute::<Ref, unsafe extern "C" fn(Ref, unsafe extern "C" fn(Ref, *mut Ref) -> Ref, Ref)>(set),
                        buffer_size: std::mem::transmute::<Ref, unsafe extern "C" fn(Ref) -> usize>(size),
                    }),
                    _ => None,
                };
                Ok(Self { _library: library, context_from_llvm, context_get_llvm, object_observer, $($name,)* })
            }
        }
    }
}

api! {
    LLVMInitializeX86TargetInfo: () -> (),
    LLVMInitializeX86Target: () -> (),
    LLVMInitializeX86TargetMC: () -> (),
    LLVMInitializeX86AsmPrinter: () -> (),
    LLVMInitializeX86AsmParser: () -> (),
    LLVMGetVersion: (*mut u32, *mut u32, *mut u32) -> (),
    LLVMDisposeMessage: (*mut c_char) -> (),
    LLVMGetErrorMessage: (Ref) -> *mut c_char,
    LLVMDisposeErrorMessage: (*mut c_char) -> (),
    LLVMContextCreate: () -> Ref,
    LLVMOrcCreateNewThreadSafeContext: () -> Ref,
    LLVMOrcDisposeThreadSafeContext: (Ref) -> (),
    LLVMCreateMemoryBufferWithMemoryRangeCopy: (*const c_char, usize, *const c_char) -> Ref,
    LLVMParseIRInContext: (Ref, Ref, *mut Ref, *mut *mut c_char) -> c_int,
    LLVMDisposeModule: (Ref) -> (),
    LLVMVerifyModule: (Ref, c_int, *mut *mut c_char) -> c_int,
    LLVMSetTarget: (Ref, *const c_char) -> (),
    LLVMSetDataLayout: (Ref, *const c_char) -> (),
    LLVMGetFirstFunction: (Ref) -> Ref,
    LLVMGetNextFunction: (Ref) -> Ref,
    LLVMIsDeclaration: (Ref) -> c_int,
    LLVMGetFirstBasicBlock: (Ref) -> Ref,
    LLVMGetNextBasicBlock: (Ref) -> Ref,
    LLVMGetBasicBlockName: (Ref) -> *const c_char,
    LLVMGetBasicBlockTerminator: (Ref) -> Ref,
    LLVMGetNumSuccessors: (Ref) -> u32,
    LLVMGetSuccessor: (Ref, u32) -> Ref,
    LLVMSetSuccessor: (Ref, u32, Ref) -> (),
    LLVMGetInstructionParent: (Ref) -> Ref,
    LLVMGetBasicBlockParent: (Ref) -> Ref,
    LLVMAppendBasicBlockInContext: (Ref, Ref, *const c_char) -> Ref,
    LLVMGetFirstInstruction: (Ref) -> Ref,
    LLVMGetNextInstruction: (Ref) -> Ref,
    LLVMGetInstructionOpcode: (Ref) -> c_int,
    LLVMIsABranchInst: (Ref) -> Ref,
    LLVMIsAPHINode: (Ref) -> Ref,
    LLVMIsConditional: (Ref) -> c_int,
    LLVMGetCondition: (Ref) -> Ref,
    LLVMGetEnumAttributeKindForName: (*const c_char, usize) -> u32,
    LLVMRemoveEnumAttributeAtIndex: (Ref, u32, u32) -> (),
    LLVMRemoveCallSiteEnumAttribute: (Ref, u32, u32) -> (),
    LLVMGetLinkage: (Ref) -> c_int,
    LLVMGetFirstUse: (Ref) -> Ref,
    LLVMGetValueName2: (Ref, *mut usize) -> *const c_char,
    LLVMAddTargetDependentFunctionAttr: (Ref, *const c_char, *const c_char) -> (),
    LLVMPrintModuleToString: (Ref) -> *mut c_char,
    LLVMInt32TypeInContext: (Ref) -> Ref,
    LLVMInt64TypeInContext: (Ref) -> Ref,
    LLVMArrayType: (Ref, u32) -> Ref,
    LLVMConstInt: (Ref, u64, c_int) -> Ref,
    LLVMConstNull: (Ref) -> Ref,
    LLVMAddGlobal: (Ref, Ref, *const c_char) -> Ref,
    LLVMSetInitializer: (Ref, Ref) -> (),
    LLVMSetAlignment: (Ref, u32) -> (),
    LLVMCreateBuilderInContext: (Ref) -> Ref,
    LLVMDisposeBuilder: (Ref) -> (),
    LLVMPositionBuilderBefore: (Ref, Ref) -> (),
    LLVMPositionBuilderAtEnd: (Ref, Ref) -> (),
    LLVMBuildBr: (Ref, Ref) -> Ref,
    LLVMBuildInBoundsGEP2: (Ref, Ref, Ref, *mut Ref, u32, *const c_char) -> Ref,
    LLVMBuildSelect: (Ref, Ref, Ref, Ref, *const c_char) -> Ref,
    LLVMBuildAtomicRMW: (Ref, c_int, Ref, Ref, c_int, c_int) -> Ref,
    LLVMGetMDKindIDInContext: (Ref, *const c_char, u32) -> u32,
    LLVMMDStringInContext2: (Ref, *const c_char, usize) -> Ref,
    LLVMMDNodeInContext2: (Ref, *mut Ref, usize) -> Ref,
    LLVMValueAsMetadata: (Ref) -> Ref,
    LLVMMetadataAsValue: (Ref, Ref) -> Ref,
    LLVMSetMetadata: (Ref, u32, Ref) -> (),
    LLVMGetMetadata: (Ref, u32) -> Ref,
    LLVMTemporaryMDNode: (Ref, *mut Ref, usize) -> Ref,
    LLVMMetadataReplaceAllUsesWith: (Ref, Ref) -> (),
    LLVMGetHostCPUName: () -> *mut c_char,
    LLVMGetHostCPUFeatures: () -> *mut c_char,
    LLVMGetDefaultTargetTriple: () -> *mut c_char,
    LLVMGetTargetFromTriple: (*const c_char, *mut Ref, *mut *mut c_char) -> c_int,
    LLVMCreateTargetMachine: (Ref, *const c_char, *const c_char, *const c_char, c_int, c_int, c_int) -> Ref,
    LLVMDisposeTargetMachine: (Ref) -> (),
    LLVMCreatePassBuilderOptions: () -> Ref,
    LLVMDisposePassBuilderOptions: (Ref) -> (),
    LLVMPassBuilderOptionsSetVerifyEach: (Ref, c_int) -> (),
    LLVMPassBuilderOptionsSetLoopUnrolling: (Ref, c_int) -> (),
    LLVMRunPasses: (Ref, *const c_char, Ref, Ref) -> Ref,
    LLVMOrcCreateLLJIT: (*mut Ref, Ref) -> Ref,
    LLVMOrcCreateLLJITBuilder: () -> Ref,
    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder: (Ref, Ref) -> (),
    LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine: (Ref) -> Ref,
    LLVMOrcDisposeLLJIT: (Ref) -> Ref,
    LLVMOrcLLJITGetMainJITDylib: (Ref) -> Ref,
    LLVMOrcLLJITGetTripleString: (Ref) -> *const c_char,
    LLVMOrcLLJITGetDataLayoutStr: (Ref) -> *const c_char,
    LLVMOrcLLJITGetGlobalPrefix: (Ref) -> c_char,
    LLVMOrcLLJITMangleAndIntern: (Ref, *const c_char) -> Ref,
    LLVMOrcAbsoluteSymbols: (*mut SymbolPair, usize) -> Ref,
    LLVMOrcJITDylibDefine: (Ref, Ref) -> Ref,
    LLVMOrcDisposeMaterializationUnit: (Ref) -> (),
    LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess: (*mut Ref, c_char, Ref, Ref) -> Ref,
    LLVMOrcJITDylibAddGenerator: (Ref, Ref) -> (),
    LLVMOrcCreateNewThreadSafeModule: (Ref, Ref) -> Ref,
    LLVMOrcLLJITAddLLVMIRModule: (Ref, Ref, Ref) -> Ref,
    LLVMOrcLLJITLookup: (Ref, *mut u64, *const c_char) -> Ref,
}

struct Library(Ref);
// LLVM initialization runs once; these are immutable C API entrypoints.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Library {
    unsafe fn open(name: &CStr) -> Result<Self, JitError> {
        // RTLD_NOW | RTLD_LOCAL: LLVM symbols do not pollute the host. Clear
        // old optional-symbol failures before obtaining this attempt's error.
        dlerror();
        let handle = dlopen(name.as_ptr().cast(), 2);
        if handle.is_null() {
            Err(JitError::new(loader_error()))
        } else {
            Ok(Self(handle))
        }
    }

    unsafe fn symbol(&self, name: &[u8]) -> Result<Ref, JitError> {
        dlerror();
        let value = dlsym(self.0, name.as_ptr().cast());
        if value.is_null() {
            return Err(JitError::new(format!(
                "LLVM is missing {}: {}",
                String::from_utf8_lossy(&name[..name.len() - 1]),
                loader_error()
            )));
        }
        Ok(value)
    }

    unsafe fn optional_symbol(&self, name: &[u8]) -> Option<Ref> {
        dlerror();
        let value = dlsym(self.0, name.as_ptr());
        (!value.is_null()).then_some(value)
    }
}

/// Try a complete API load for every candidate, rather than stopping at the
/// first library the OS loader can open. A generic libLLVM.so can refer to an
/// older installation while a usable versioned library exists beside it.
pub(super) unsafe fn load_candidates(candidates: &[CString]) -> Result<Api, JitError> {
    let mut failures = Vec::new();
    for candidate in candidates {
        match Library::open(candidate).and_then(|library| Api::load(library)) {
            Ok(api) => return Ok(api),
            Err(error) => failures.push(format!("{}: {error}", candidate.to_string_lossy())),
        }
    }
    Err(JitError::new(format!(
        "could not load a compatible LLVM shared library:\n{}. Install LLVM 17+ or set Y_LLVM_LIBRARY",
        failures.join("\n")
    )))
}

unsafe fn discover() -> Result<Api, JitError> {
    if let Some(path) = std::env::var_os("Y_LLVM_LIBRARY") {
        let candidate = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| JitError::new("Y_LLVM_LIBRARY path contains NUL"))?;
        // An explicit override is authoritative: never silently select a
        // different installation after an invalid configured path or API.
        return load_candidates(&[candidate])
            .map_err(|error| JitError::new(format!("Y_LLVM_LIBRARY override failed: {error}")));
    }
    let mut candidates = vec![CString::new("libLLVM.so").unwrap()];
    for version in (17..=24).rev() {
        for name in [
            format!("libLLVM-{version}.so"),
            format!("libLLVM-{version}.so.1"),
            format!("libLLVM.so.{version}"),
            format!("libLLVM.so.{version}.1"),
        ] {
            candidates.push(CString::new(name).unwrap());
        }
    }
    load_candidates(&candidates)
}

impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            dlclose(self.0);
        }
    }
}

unsafe fn loader_error() -> String {
    let message = dlerror();
    if message.is_null() {
        "unknown loader error".into()
    } else {
        CStr::from_ptr(message).to_string_lossy().into_owned()
    }
}

pub fn api() -> Result<&'static Api, JitError> {
    static API: OnceLock<Result<Api, JitError>> = OnceLock::new();
    API.get_or_init(|| unsafe {
        let api = discover()?;
        (api.LLVMInitializeX86TargetInfo)();
        (api.LLVMInitializeX86Target)();
        (api.LLVMInitializeX86TargetMC)();
        (api.LLVMInitializeX86AsmPrinter)();
        (api.LLVMInitializeX86AsmParser)();
        Ok(api)
    })
    .as_ref()
    .map_err(Clone::clone)
}

pub unsafe fn process_symbol(name: &CStr) -> Ref {
    // RTLD_DEFAULT searches symbols visible in the embedding process.
    dlsym(std::ptr::null_mut(), name.as_ptr().cast())
}

impl Api {
    pub unsafe fn error(&self, error: Ref) -> Result<(), JitError> {
        if error.is_null() {
            return Ok(());
        }
        let message = (self.LLVMGetErrorMessage)(error);
        let result = JitError::new(CStr::from_ptr(message).to_string_lossy().into_owned());
        (self.LLVMDisposeErrorMessage)(message);
        Err(result)
    }

    pub unsafe fn message(&self, message: *mut c_char) -> String {
        if message.is_null() {
            return String::new();
        }
        let result = CStr::from_ptr(message).to_string_lossy().into_owned();
        (self.LLVMDisposeMessage)(message);
        result
    }

    pub fn version(&self) -> String {
        let (mut major, mut minor, mut patch) = (0, 0, 0);
        unsafe {
            (self.LLVMGetVersion)(&mut major, &mut minor, &mut patch);
        }
        format!("{major}.{minor}.{patch}")
    }
}
