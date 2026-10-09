//! In-process CPU JIT for the existing Y LLVM host language.
//!
//! Source is checked and lowered by the same frontend/backend as the AOT
//! compiler. LLVM ORC owns executable memory; dropping `CpuJit` releases it.
//! Function addresses use the C ABI and remain valid only while their JIT lives.

use crate::ast::{Item, Program, Type};
use crate::{lexer::Lexer, llvm_emitter::LlvmEmitter, parser::Parser, type_checker::TypeChecker};
use std::collections::BTreeMap;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use std::ffi::CString;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;
use std::time::{Duration, Instant};

mod call;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod llvm;
#[cfg(any(test, all(target_os = "linux", target_arch = "x86_64")))]
mod loops;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod runtime;
pub use call::{AbiType, FunctionSignature, JitValue};
mod cache;
pub use cache::CpuJitCache;
mod profile;
pub use profile::{BranchProfile, BranchSite};

#[derive(Debug, Clone)]
pub struct JitError(String);
impl JitError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for JitError {}

#[derive(Debug, Clone, Copy)]
pub struct JitOptions {
    /// LLVM's default optimization pipeline, from 0 through 3.
    pub opt_level: u8,
    /// Override the IR optimization tier for instrumented training, from 0 to 3.
    /// `None` follows `opt_level`; ordinary and profile-use IR keep `opt_level`.
    /// Native codegen still follows `codegen_opt_level` or `opt_level`.
    pub training_opt_level: Option<u8>,
    /// Use fixed-address atomic counters on eligible instrumented branch edges.
    /// Branches with PHI successors retain selected-address instrumentation.
    /// Ordinary and profile-use sessions are unaffected; defaults to false.
    pub profile_edge_counters: bool,
    /// Use fixed-address atomic counters only on natural-loop iteration edges.
    /// Other branches and PHI successors keep selected-address instrumentation.
    /// `profile_edge_counters` takes precedence; both flags default to false.
    pub profile_loop_edge_counters: bool,
    /// Keep LLVM's default loop-unrolling tuning for ordinary and profile-use IR.
    /// `false` disables that tuning; instrumented training keeps LLVM's defaults.
    /// Native codegen and the requested IR optimization level are unchanged.
    pub final_loop_unrolling: bool,
    /// Keep default unrolling for original natural loops containing nested loops.
    /// `false` adds unroll-disable metadata to eligible outer-loop latches only.
    /// Instrument mode and global unrolling disabled by `final_loop_unrolling`
    /// ignore this policy. Existing loop metadata and ambiguous loops are preserved.
    pub final_unroll_outer_loops: bool,
    /// Override ORC machine-code optimization, from 0 through 3.
    /// `None` follows `opt_level`; this does not change the IR pipeline.
    pub codegen_opt_level: Option<u8>,
    /// Diagnose invalid intermediate IR after individual LLVM passes.
    /// Full-module verification before and after optimization is unconditional.
    pub verify_each_pass: bool,
    /// Preserve proven unsigned rotate idioms before arithmetic combining.
    pub recognize_rotates: bool,
    /// Inline bounded queries for proven local native-runtime String/Vec handles.
    pub optimize_runtime: bool,
    /// Append directly to proven local String/Vec storage when capacity permits.
    /// Allocation, growth and freeing keep their ordinary runtime callbacks.
    pub optimize_runtime_mutations: bool,
    /// Extend guarded mutation copies to bulk Strings and exact-width dynamic Vecs.
    /// Requires mutation optimization; growth and unproved copies keep callbacks.
    pub optimize_runtime_copies: bool,
    /// Preserve local runtime fast paths across proved scalar-only source helpers.
    pub optimize_helper_effects: bool,
    /// Keep checked-call adapters compact without restricting source-call inlining.
    pub optimize_call_adapters: bool,
    /// Apply branch profiles to natural-loop iteration control as well as work.
    /// Disabled by default because counts do not identify compiler-created loop versions.
    pub profile_loop_controls: bool,
}
impl Default for JitOptions {
    fn default() -> Self {
        Self {
            opt_level: 3,
            training_opt_level: None,
            profile_edge_counters: false,
            profile_loop_edge_counters: false,
            final_loop_unrolling: true,
            final_unroll_outer_loops: true,
            codegen_opt_level: None,
            verify_each_pass: true,
            recognize_rotates: true,
            optimize_runtime: true,
            optimize_runtime_mutations: true,
            optimize_runtime_copies: true,
            optimize_helper_effects: true,
            optimize_call_adapters: true,
            profile_loop_controls: false,
        }
    }
}

/// Disjoint wall-clock intervals inside the LLVM optimization phase.
///
/// Each pipeline interval includes the LLVM pass-manager call and its requested
/// per-pass verification. `other` covers option setup, error handling, cleanup
/// and timer overhead. These details partition `JitCompileTimings::optimization`
/// and must not be added to the primary compilation phase sum again.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JitOptimizationTimings {
    pub pipeline: Duration,
    pub profile_selection: Duration,
    pub other: Duration,
    pub total: Duration,
}

impl JitOptimizationTimings {
    fn finish(&mut self, total: Duration) {
        self.total = total;
        self.other = total.saturating_sub(self.pipeline + self.profile_selection);
    }

    /// Sum the nested phases without counting `total` twice.
    pub fn accounted_duration(&self) -> Duration {
        self.pipeline + self.profile_selection + self.other
    }

    /// An owned JSON object with integer nanoseconds and stable `*_ns` keys.
    pub fn to_json(&self) -> String {
        use std::fmt::Write;
        let mut json = String::from("{");
        for (index, (name, duration)) in [
            ("pipeline_ns", self.pipeline),
            ("profile_selection_ns", self.profile_selection),
            ("other_ns", self.other),
            ("total_ns", self.total),
        ]
        .iter()
        .enumerate()
        {
            if index != 0 {
                json.push(',');
            }
            write!(json, "\"{name}\":{}", duration.as_nanos()).unwrap();
        }
        json.push('}');
        json
    }
}

/// Disjoint intervals within eager ORC materialization.
///
/// Optional object-ready children partition `first_lookup`, not the total again.
/// The interval before that callback includes native object emission and ORC
/// work; the interval after it includes observer overhead, linking and lookup.
/// Neither interval is an exclusive code-generator or linker measurement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JitMaterializationTimings {
    pub submission: Duration,
    pub first_lookup: Duration,
    pub remaining_function_lookups: Duration,
    pub profile_lookup: Duration,
    pub other: Duration,
    pub total: Duration,
    pub first_lookup_before_object: Option<Duration>,
    pub first_lookup_after_object: Option<Duration>,
    pub object_observer_available: bool,
    pub object_count: u64,
    pub object_bytes: u64,
    pub function_lookup_count: u64,
    pub profile_lookup_count: u64,
}

impl JitMaterializationTimings {
    fn finish(&mut self, total: Duration) {
        self.total = total;
        self.other = total.saturating_sub(
            self.submission
                + self.first_lookup
                + self.remaining_function_lookups
                + self.profile_lookup,
        );
    }

    /// Sum disjoint parent intervals, excluding children, metadata and `total`.
    pub fn accounted_duration(&self) -> Duration {
        self.submission
            + self.first_lookup
            + self.remaining_function_lookups
            + self.profile_lookup
            + self.other
    }

    /// Owned JSON; unavailable object-ready child durations are null.
    pub fn to_json(&self) -> String {
        use std::fmt::Write;
        let mut json = String::from("{");
        for (index, (name, duration)) in [
            ("submission_ns", self.submission),
            ("first_lookup_ns", self.first_lookup),
            (
                "remaining_function_lookups_ns",
                self.remaining_function_lookups,
            ),
            ("profile_lookup_ns", self.profile_lookup),
            ("other_ns", self.other),
            ("total_ns", self.total),
        ]
        .iter()
        .enumerate()
        {
            if index != 0 {
                json.push(',');
            }
            write!(json, "\"{name}\":{}", duration.as_nanos()).unwrap();
        }
        for (name, duration) in [
            (
                "first_lookup_before_object_ns",
                self.first_lookup_before_object,
            ),
            (
                "first_lookup_after_object_ns",
                self.first_lookup_after_object,
            ),
        ] {
            write!(json, ",\"{name}\":").unwrap();
            if let Some(duration) = duration {
                write!(json, "{}", duration.as_nanos()).unwrap();
            } else {
                json.push_str("null");
            }
        }
        write!(json, ",\"object_observer_available\":{},\"object_count\":{},\"object_bytes\":{},\"function_lookup_count\":{},\"profile_lookup_count\":{}}}",
            self.object_observer_available, self.object_count, self.object_bytes,
            self.function_lookup_count, self.profile_lookup_count).unwrap();
        json
    }
}

/// Disjoint wall-clock intervals for one successful compilation, excluding calls.
///
/// `parse` is zero for the pre-parsed program API. `optimization` includes LLVM's
/// verification after individual passes when requested; `verification` aggregates
/// the explicit full-module checks before optimization and after each pipeline
/// invocation. ORC materialization includes eager code generation, linking and
/// address lookup for every public function and checked-call adapter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JitCompileTimings {
    pub parse: Duration,
    pub checks: Duration,
    pub lowering: Duration,
    pub llvm_setup: Duration,
    pub ir_parse: Duration,
    pub profile_setup: Duration,
    pub verification: Duration,
    pub optimization: Duration,
    pub ir_capture: Duration,
    pub symbol_resolution: Duration,
    pub materialization: Duration,
    /// Remaining setup, metadata, cleanup and timer overhead.
    pub other: Duration,
    pub total: Duration,
    /// Nested details of `optimization`, excluded from the primary sum and JSON.
    pub optimization_details: JitOptimizationTimings,
    /// Nested materialization parents and optional object-ready children.
    pub materialization_details: JitMaterializationTimings,
    /// Successful explicit module checks, excluded from the primary sum and JSON.
    pub verification_checks: u32,
}

impl JitCompileTimings {
    fn phases(&self) -> [(&'static str, Duration); 13] {
        [
            ("parse_ns", self.parse),
            ("checks_ns", self.checks),
            ("lowering_ns", self.lowering),
            ("llvm_setup_ns", self.llvm_setup),
            ("ir_parse_ns", self.ir_parse),
            ("profile_setup_ns", self.profile_setup),
            ("verification_ns", self.verification),
            ("optimization_ns", self.optimization),
            ("ir_capture_ns", self.ir_capture),
            ("symbol_resolution_ns", self.symbol_resolution),
            ("materialization_ns", self.materialization),
            ("other_ns", self.other),
            ("total_ns", self.total),
        ]
    }

    fn finish(&mut self, total: Duration) {
        let measured: Duration = self.phases()[..11].iter().map(|(_, time)| *time).sum();
        self.total = total;
        self.other = total.saturating_sub(measured);
    }

    /// Sum every phase, including `other`, without counting `total` twice.
    pub fn accounted_duration(&self) -> Duration {
        self.phases()[..12].iter().map(|(_, time)| *time).sum()
    }

    /// An owned JSON object with integer nanoseconds and stable `*_ns` keys.
    pub fn to_json(&self) -> String {
        use std::fmt::Write;
        let mut json = String::from("{");
        for (index, (name, duration)) in self.phases().iter().enumerate() {
            if index != 0 {
                json.push(',');
            }
            write!(json, "\"{name}\":{}", duration.as_nanos()).unwrap();
        }
        json.push('}');
        json
    }
}

#[derive(Clone, Copy)]
enum ProfileMode<'a> {
    None,
    Instrument,
    Use(&'a BranchProfile),
}

#[cfg(test)]
mod option_tests {
    use super::*;

    #[test]
    fn invalid_training_override_is_rejected_before_semantic_checks_or_llvm() {
        let ast = Program { items: Vec::new() };
        for level in [4, u8::MAX] {
            let options = JitOptions {
                training_opt_level: Some(level),
                ..JitOptions::default()
            };
            for mode in [ProfileMode::None, ProfileMode::Instrument] {
                match CpuJit::compile_program_mode(&ast, options, mode) {
                    Err(error) => assert_eq!(
                        error.to_string(),
                        "training_opt_level must be 0, 1, 2, or 3 when set"
                    ),
                    Ok(_) => panic!("invalid training tier was accepted"),
                }
            }
        }
    }
}

/// A CPU profile without GPU probes, disk caches, or guessed measurements.
pub fn host_profile() -> crate::sentinel::HardwareProfile {
    crate::sentinel::HardwareProfile {
        has_avx: crate::sentinel::host_has_avx(),
        has_avx512: crate::sentinel::host_has_avx512(),
        sm_version: "0.0".into(),
        ..Default::default()
    }
}

pub struct CpuJit {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    _engine: Engine,
    addresses: BTreeMap<String, usize>,
    signatures: BTreeMap<String, FunctionSignature>,
    adapters: BTreeMap<String, usize>,
    main_returns_void: Option<bool>,
    compile_timings: JitCompileTimings,
    optimized_ir: String,
    llvm_version: String,
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    instrumentation: Option<(profile::Instrumentation, usize)>,
    profiled_branches: usize,
    profile_selection_optimization: bool,
    outer_unroll_annotations: usize,
    // Session access and destruction stay on their creating thread. Callers
    // can arrange parallel execution of native functions using unsafe code.
    _thread_bound: PhantomData<Rc<()>>,
}

impl CpuJit {
    pub fn compile(source: &str) -> Result<Self, JitError> {
        Self::compile_with_options(source, JitOptions::default())
    }

    pub fn compile_with_options(source: &str, options: JitOptions) -> Result<Self, JitError> {
        Self::compile_source(source, options, ProfileMode::None)
    }

    /// Compile with atomic counters for the original conditional branches.
    /// Compilation does not execute source; the host supplies training calls.
    pub fn compile_instrumented(source: &str, options: JitOptions) -> Result<Self, JitError> {
        Self::compile_source(source, options, ProfileMode::Instrument)
    }

    /// Apply measured branch counts from the identical lowered compilation unit.
    /// No counters or profiling callbacks remain in this new native session.
    pub fn compile_with_profile(
        source: &str,
        options: JitOptions,
        profile: &BranchProfile,
    ) -> Result<Self, JitError> {
        Self::compile_source(source, options, ProfileMode::Use(profile))
    }

    fn compile_source(
        source: &str,
        options: JitOptions,
        mode: ProfileMode<'_>,
    ) -> Result<Self, JitError> {
        let start = Instant::now();
        let ast = Parser::new(Lexer::new(source).tokenize())
            .parse_program()
            .map_err(|e| JitError::new(format!("parser: {e}")))?;
        let parse = start.elapsed();
        let mut jit = Self::compile_program_mode(&ast, options, mode)?;
        jit.compile_timings.parse = parse;
        jit.compile_timings.finish(start.elapsed());
        Ok(jit)
    }

    /// Compile a program whose imports have already been resolved by the host.
    /// Semantic checks are always performed, including for AST callers.
    pub fn compile_program(ast: &Program, options: JitOptions) -> Result<Self, JitError> {
        Self::compile_program_mode(ast, options, ProfileMode::None)
    }

    pub fn compile_program_instrumented(
        ast: &Program,
        options: JitOptions,
    ) -> Result<Self, JitError> {
        Self::compile_program_mode(ast, options, ProfileMode::Instrument)
    }

    pub fn compile_program_with_profile(
        ast: &Program,
        options: JitOptions,
        profile: &BranchProfile,
    ) -> Result<Self, JitError> {
        Self::compile_program_mode(ast, options, ProfileMode::Use(profile))
    }

    fn compile_program_mode(
        ast: &Program,
        options: JitOptions,
        mode: ProfileMode<'_>,
    ) -> Result<Self, JitError> {
        let start = Instant::now();
        let mut timings = JitCompileTimings::default();
        let checks_start = Instant::now();
        if options.opt_level > 3 {
            return Err(JitError::new("opt_level must be 0, 1, 2, or 3"));
        }
        if options.training_opt_level.is_some_and(|level| level > 3) {
            return Err(JitError::new(
                "training_opt_level must be 0, 1, 2, or 3 when set",
            ));
        }
        if options.codegen_opt_level.is_some_and(|level| level > 3) {
            return Err(JitError::new(
                "codegen_opt_level must be 0, 1, 2, or 3 when set",
            ));
        }
        // The emitter silently ignores imports/modules: refuse unresolved items
        // instead of producing a successful, incomplete program.
        for item in &ast.items {
            match item {
                Item::Import(i) => return Err(JitError::new(format!("Line {}: CPU JIT source API requires resolved imports; use the Y --jit CLI or compile_program", i.span.line))),
                Item::Module(m) => return Err(JitError::new(format!("Line {}: CPU JIT does not lower module declarations", m.span.line))),
                _ => {}
            }
        }
        let mut checker = TypeChecker::new();
        checker.check_program(ast);
        let mut errors = checker.errors;
        errors.extend(checker.linear_tracker.errors);
        if !errors.is_empty() {
            return Err(JitError::new(errors.join("\n")));
        }
        let profile = host_profile();
        let (_, errors) = crate::require::check_program(ast, &profile);
        if !errors.is_empty() {
            return Err(JitError::new(errors.join("\n")));
        }
        timings.checks = checks_start.elapsed();
        let lowering_start = Instant::now();
        let mut emitter = LlvmEmitter::new();
        emitter.set_recognize_rotates(options.recognize_rotates);
        emitter.set_optimize_runtime(options.optimize_runtime);
        emitter.set_optimize_runtime_mutations(options.optimize_runtime_mutations);
        emitter.set_optimize_runtime_copies(options.optimize_runtime_copies);
        emitter.set_optimize_helper_effects(options.optimize_helper_effects);
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        emitter.set_native_string_handle_normalizer(runtime::string_handle_normalizer_address());
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        emitter.register_host_runtime_symbols(
            &runtime::symbols()
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>(),
        );
        let mut ir = emitter.emit_program(ast, &profile);
        if !emitter.emit_errors.is_empty() {
            return Err(JitError::new(emitter.emit_errors.join("\n")));
        }
        let mut signatures = call::signatures(ast);
        let adapter_names =
            call::append_adapters(&mut ir, &signatures, options.optimize_call_adapters);
        let main_returns_void = ast.items.iter().find_map(|item| {
            let Item::Func(f) = item else {
                return None;
            };
            if f.name != "main" || !f.params.is_empty() {
                return None;
            }
            match &f.ret_ty {
                None => Some(true),
                Some(Type::Primitive(n, _) | Type::Ident(n, _)) if n == "I32" || n == "i32" => {
                    Some(false)
                }
                _ => None,
            }
        });
        timings.lowering = lowering_start.elapsed();
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            let CompiledModule {
                engine,
                mut addresses,
                optimized_ir,
                instrumentation,
                profiled_branches,
                profile_selection_optimization,
                outer_unroll_annotations,
            } = Engine::compile(&ir, options, mode, &mut timings)?;
            let adapters = adapter_names
                .into_iter()
                .map(|(name, adapter)| {
                    addresses
                        .remove(&adapter)
                        .map(|address| (name, address))
                        .ok_or_else(|| {
                            JitError::new(format!("CPU JIT adapter `{adapter}` is missing"))
                        })
                })
                .collect::<Result<_, _>>()?;
            signatures.retain(|name, _| addresses.contains_key(name));
            let llvm_version = engine.api.version();
            timings.finish(start.elapsed());
            Ok(Self {
                _engine: engine,
                addresses,
                signatures,
                adapters,
                main_returns_void,
                compile_timings: timings,
                optimized_ir,
                llvm_version,
                instrumentation,
                profiled_branches,
                profile_selection_optimization,
                outer_unroll_annotations,
                _thread_bound: PhantomData,
            })
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            let _ = (
                ir,
                main_returns_void,
                start,
                signatures,
                adapter_names,
                mode,
                timings,
            );
            Err(JitError::new("CPU JIT currently supports Linux x86-64"))
        }
    }

    /// Return a compiled C ABI entrypoint. `main` aliases `ysu_main`.
    ///
    /// Calling the address is unsafe: its signature must exactly match Y's
    /// lowered ABI, pointer arguments must be valid, and this JIT must outlive
    /// every call. No interpreter or dispatch work occurs in native calls.
    pub fn function_address(&self, name: &str) -> Result<usize, JitError> {
        let name = if name == "main" { "ysu_main" } else { name };
        self.addresses
            .get(name)
            .copied()
            .ok_or_else(|| JitError::new(format!("CPU JIT function `{name}` is not defined")))
    }

    pub fn function_signature(&self, name: &str) -> Result<&FunctionSignature, JitError> {
        let name = if name == "main" { "ysu_main" } else { name };
        self.signatures.get(name).ok_or_else(|| {
            JitError::new(format!("CPU JIT function `{name}` has no source signature"))
        })
    }

    /// Call a scalar/pointer function after checking every argument's ABI type.
    /// Aggregate functions remain callable through `function_address`.
    ///
    /// # Safety
    /// Source must be trusted to execute in this process. Pointer values must
    /// satisfy the function's memory and aliasing requirements for the call.
    pub unsafe fn call(&self, name: &str, arguments: &[JitValue]) -> Result<JitValue, JitError> {
        let signature = self.function_signature(name)?;
        if !signature.supports_dynamic_call() {
            return Err(JitError::new(format!("dynamic calls do not support the aggregate signature of `{name}`; use function_address")));
        }
        if arguments.len() != signature.parameters.len() {
            return Err(JitError::new(format!(
                "`{name}` expects {} arguments, received {}",
                signature.parameters.len(),
                arguments.len()
            )));
        }
        for (index, (value, ty)) in arguments.iter().zip(&signature.parameters).enumerate() {
            if value.abi_type() != *ty {
                return Err(JitError::new(format!(
                    "`{name}` argument {index} expects {}, received {}",
                    ty.name(),
                    value.abi_type().name()
                )));
            }
        }
        let address = self
            .adapters
            .get(&signature.name)
            .ok_or_else(|| JitError::new(format!("`{name}` has no dynamic call adapter")))?;
        let bits: Vec<_> = arguments.iter().map(|value| value.bits()).collect();
        let adapter: unsafe extern "C" fn(*const u64) -> u64 = std::mem::transmute(*address);
        JitValue::from_bits(&signature.return_type, adapter(bits.as_ptr()))
    }

    /// Run `fn main()` or `fn main() -> I32` and return its exit status.
    ///
    /// # Safety
    /// The Y program must be trusted to execute in the current process: Y
    /// permits raw memory access, I/O and process termination.
    pub unsafe fn run_main(&self) -> Result<i32, JitError> {
        let returns_void = self.main_returns_void.ok_or_else(|| {
            JitError::new("--jit requires fn main() or fn main() -> I32 (no arguments)")
        })?;
        let address = self.function_address("main")?;
        if returns_void {
            let entry: unsafe extern "C" fn() = std::mem::transmute(address);
            entry();
            Ok(0)
        } else {
            let entry: unsafe extern "C" fn() -> i32 = std::mem::transmute(address);
            Ok(entry())
        }
    }

    /// Frontend + LLVM optimization + executable materialization; excludes calls.
    pub fn compile_duration(&self) -> Duration {
        self.compile_timings.total
    }
    /// Phase measurements for this session's compilation, excluding training/calls.
    /// Cached sessions retain the timings of their original compilation.
    pub fn compile_timings(&self) -> &JitCompileTimings {
        &self.compile_timings
    }
    /// Details that partition this session's LLVM optimization interval.
    pub fn optimization_timings(&self) -> &JitOptimizationTimings {
        &self.compile_timings.optimization_details
    }
    /// Details within eager materialization, excluding later execution/lookup.
    pub fn materialization_timings(&self) -> &JitMaterializationTimings {
        &self.compile_timings.materialization_details
    }
    pub fn optimized_ir(&self) -> &str {
        &self.optimized_ir
    }
    pub fn llvm_version(&self) -> &str {
        &self.llvm_version
    }
    pub fn functions(&self) -> impl Iterator<Item = &str> {
        self.addresses.keys().map(String::as_str)
    }

    /// Snapshot observed edges. Counts accumulate until the instrumented JIT is dropped.
    pub fn branch_profile(&self) -> Result<BranchProfile, JitError> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            let (instrumentation, address) = self.instrumentation.as_ref().ok_or_else(|| {
                JitError::new("CPU JIT was not compiled with branch instrumentation")
            })?;
            // The ORC-owned aligned counter storage is live for this session.
            Ok(unsafe { instrumentation.snapshot(*address) })
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        Err(JitError::new("CPU JIT currently supports Linux x86-64"))
    }

    /// Number of original conditional branches that received observed weights.
    pub fn profiled_branches(&self) -> usize {
        self.profiled_branches
    }

    /// Whether LLVM's targeted select-to-branch optimization ran successfully.
    pub fn profile_selection_optimization(&self) -> bool {
        self.profile_selection_optimization
    }

    /// Number of original natural loops receiving outer-loop unroll metadata.
    /// This counts annotations, not transformations LLVM avoided or live loops.
    pub fn outer_unroll_annotations(&self) -> usize {
        self.outer_unroll_annotations
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
struct CompiledModule {
    engine: Engine,
    addresses: BTreeMap<String, usize>,
    optimized_ir: String,
    instrumentation: Option<(profile::Instrumentation, usize)>,
    profiled_branches: usize,
    profile_selection_optimization: bool,
    outer_unroll_annotations: usize,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
struct Engine {
    api: &'static llvm::Api,
    jit: llvm::Ref,
    // Heap address is stable; the callback context lives through DisposeLLJIT.
    observer: Option<Box<MaterializationObserver>>,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[derive(Default)]
struct ObjectObservation {
    first_start: Option<Instant>,
    first_end: Option<Instant>,
    object_entry: Option<Instant>,
    count: u64,
    bytes: u64,
    outside_first: bool,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
struct MaterializationObserver {
    buffer_size: unsafe extern "C" fn(llvm::Ref) -> usize,
    observation: std::sync::Mutex<ObjectObservation>,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
impl MaterializationObserver {
    fn lock(&self) -> std::sync::MutexGuard<'_, ObjectObservation> {
        self.observation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn snapshot(&self, timings: &mut JitMaterializationTimings) {
        let observation = self.lock();
        timings.object_observer_available = true;
        timings.object_count = observation.count;
        timings.object_bytes = observation.bytes;
        timings.first_lookup_before_object = None;
        timings.first_lookup_after_object = None;
        if observation.count == 1 && !observation.outside_first {
            if let (Some(start), Some(entry), Some(end)) = (
                observation.first_start,
                observation.object_entry,
                observation.first_end,
            ) {
                if let (Some(before), Some(after)) = (
                    entry.checked_duration_since(start),
                    end.checked_duration_since(entry),
                ) {
                    timings.first_lookup_before_object = Some(before);
                    timings.first_lookup_after_object = Some(after);
                }
            }
        }
    }
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod object_observer_tests {
    use super::*;

    unsafe extern "C" fn size(_: llvm::Ref) -> usize {
        64
    }

    #[test]
    fn callback_preserves_buffer_and_handles_poisoned_and_nonunique_observations() {
        let observer = Box::new(MaterializationObserver {
            buffer_size: size,
            observation: std::sync::Mutex::new(ObjectObservation::default()),
        });
        let context = (&*observer as *const MaterializationObserver)
            .cast_mut()
            .cast();
        let mut buffer = std::ptr::without_provenance_mut::<std::ffi::c_void>(0x1234);
        let original = buffer;
        let start = Instant::now();
        observer.lock().first_start = Some(start);
        assert!(unsafe { observe_object(context, &mut buffer) }.is_null());
        assert_eq!(buffer, original);
        let end = Instant::now();
        observer.lock().first_end = Some(end);
        let mut timings = JitMaterializationTimings {
            first_lookup: end.duration_since(start),
            ..Default::default()
        };
        observer.snapshot(&mut timings);
        assert_eq!(timings.object_count, 1);
        assert_eq!(timings.object_bytes, 64);
        assert_eq!(
            timings.first_lookup_before_object.unwrap()
                + timings.first_lookup_after_object.unwrap(),
            timings.first_lookup
        );

        // A poisoned bookkeeping lock must not propagate panic through C.
        assert!(std::panic::catch_unwind(|| {
            let _lock = observer.lock();
            panic!("deliberately poison observer bookkeeping");
        })
        .is_err());
        assert!(unsafe { observe_object(context, &mut buffer) }.is_null());
        assert_eq!(buffer, original);
        observer.snapshot(&mut timings);
        assert_eq!(timings.object_count, 2);
        assert_eq!(timings.object_bytes, 128);
        assert!(timings.first_lookup_before_object.is_none());
        assert!(timings.first_lookup_after_object.is_none());

        let outside = MaterializationObserver {
            buffer_size: size,
            observation: std::sync::Mutex::new(ObjectObservation::default()),
        };
        let context = (&outside as *const MaterializationObserver)
            .cast_mut()
            .cast();
        assert!(unsafe { observe_object(context, &mut buffer) }.is_null());
        let mut outside_timings = JitMaterializationTimings::default();
        outside.snapshot(&mut outside_timings);
        assert_eq!(outside_timings.object_count, 1);
        assert!(outside_timings.first_lookup_before_object.is_none());
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe extern "C" fn observe_object(context: llvm::Ref, buffer: *mut llvm::Ref) -> llvm::Ref {
    let entry = Instant::now();
    // The LLJIT-owned buffer is observed, never changed, retained or disposed.
    // This callback performs no allocation/I/O and never propagates a panic.
    let observer = &*context.cast::<MaterializationObserver>();
    let bytes = (observer.buffer_size)(*buffer) as u64;
    let mut observation = observer.lock();
    observation.count = observation.count.saturating_add(1);
    observation.bytes = observation.bytes.saturating_add(bytes);
    if observation.object_entry.is_none() {
        observation.object_entry = Some(entry);
    }
    observation.outside_first |=
        observation.first_start.is_none() || observation.first_end.is_some();
    std::ptr::null_mut()
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            let _ = self.api.error((self.api.LLVMOrcDisposeLLJIT)(self.jit));
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
struct Module {
    api: &'static llvm::Api,
    module: llvm::Ref,
    context: llvm::Ref,
    target: llvm::Ref,
}
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            if !self.target.is_null() {
                (self.api.LLVMDisposeTargetMachine)(self.target);
            }
            if !self.module.is_null() {
                (self.api.LLVMDisposeModule)(self.module);
            }
            (self.api.LLVMOrcDisposeThreadSafeContext)(self.context);
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
impl Engine {
    unsafe fn verify_module(
        api: &llvm::Api,
        module: llvm::Ref,
        stage: &str,
        timings: &mut JitCompileTimings,
    ) -> Result<(), JitError> {
        let verification_start = Instant::now();
        let mut message = std::ptr::null_mut();
        if (api.LLVMVerifyModule)(module, 2, &mut message) != 0 {
            return Err(JitError::new(format!("{stage}: {}", api.message(message))));
        }
        if !message.is_null() {
            api.message(message);
        }
        timings.verification += verification_start.elapsed();
        timings.verification_checks += 1;
        Ok(())
    }

    fn compile(
        ir: &str,
        options: JitOptions,
        mode: ProfileMode<'_>,
        timings: &mut JitCompileTimings,
    ) -> Result<CompiledModule, JitError> {
        use std::ptr::null_mut;
        // Training may use a cheaper IR pipeline while retaining the requested
        // native codegen tier and the final ordinary/profile-use IR pipeline.
        let ir_opt_level = if matches!(mode, ProfileMode::Instrument) {
            options.training_opt_level.unwrap_or(options.opt_level)
        } else {
            options.opt_level
        };
        let setup_start = Instant::now();
        let api = llvm::api()?;
        unsafe {
            let cpu = CString::new(api.message((api.LLVMGetHostCPUName)())).unwrap();
            let features = CString::new(api.message((api.LLVMGetHostCPUFeatures)())).unwrap();
            let host_triple =
                CString::new(api.message((api.LLVMGetDefaultTargetTriple)())).unwrap();
            let mut target = null_mut();
            let mut message = null_mut();
            if (api.LLVMGetTargetFromTriple)(host_triple.as_ptr(), &mut target, &mut message) != 0 {
                return Err(JitError::new(api.message(message)));
            }
            let codegen_target = (api.LLVMCreateTargetMachine)(
                target,
                host_triple.as_ptr(),
                cpu.as_ptr(),
                features.as_ptr(),
                options
                    .codegen_opt_level
                    .unwrap_or(options.opt_level)
                    .into(),
                2,
                1,
            );
            if codegen_target.is_null() {
                return Err(JitError::new("LLVM could not create host codegen target"));
            }
            // ORC takes ownership of this target template and builder. The
            // override affects machine-code optimization only; the IR pipeline
            // uses a separate target machine at ir_opt_level below.
            let target_builder =
                (api.LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine)(codegen_target);
            let builder = (api.LLVMOrcCreateLLJITBuilder)();
            (api.LLVMOrcLLJITBuilderSetJITTargetMachineBuilder)(builder, target_builder);
            let mut jit = null_mut();
            api.error((api.LLVMOrcCreateLLJIT)(&mut jit, builder))?;
            let observer = api.object_observer.as_ref().and_then(|hooks| {
                let layer = (hooks.get_layer)(jit);
                if layer.is_null() {
                    return None;
                }
                let mut observer = Box::new(MaterializationObserver {
                    buffer_size: hooks.buffer_size,
                    observation: std::sync::Mutex::new(ObjectObservation::default()),
                });
                // LLJIT's default object transform is identity. Preserve the
                // buffer and default IR transform, observing only object handoff.
                (hooks.set_transform)(
                    layer,
                    observe_object,
                    (&mut *observer as *mut MaterializationObserver).cast(),
                );
                Some(observer)
            });
            let engine = Self { api, jit, observer };
            let dylib = (api.LLVMOrcLLJITGetMainJITDylib)(jit);
            let (context, llvm_context) = if let Some(adopt) = api.context_from_llvm {
                let llvm_context = (api.LLVMContextCreate)();
                (adopt(llvm_context), llvm_context)
            } else {
                let context = (api.LLVMOrcCreateNewThreadSafeContext)();
                (context, api.context_get_llvm.unwrap()(context))
            };
            let mut module = Module {
                api,
                module: null_mut(),
                context,
                target: null_mut(),
            };
            timings.llvm_setup = setup_start.elapsed();
            let ir_parse_start = Instant::now();
            let buffer = (api.LLVMCreateMemoryBufferWithMemoryRangeCopy)(
                ir.as_ptr().cast(),
                ir.len(),
                c"Y CPU JIT".as_ptr(),
            );
            // LLVMParseIRInContext consumes the buffer on either outcome.
            if (api.LLVMParseIRInContext)(llvm_context, buffer, &mut module.module, &mut message)
                != 0
            {
                return Err(JitError::new(format!(
                    "LLVM IR parse: {}",
                    api.message(message)
                )));
            }
            timings.ir_parse = ir_parse_start.elapsed();
            let setup_start = Instant::now();
            let triple = (api.LLVMOrcLLJITGetTripleString)(jit);
            (api.LLVMSetTarget)(module.module, triple);
            (api.LLVMSetDataLayout)(module.module, (api.LLVMOrcLLJITGetDataLayoutStr)(jit));
            if (api.LLVMGetTargetFromTriple)(triple, &mut target, &mut message) != 0 {
                return Err(JitError::new(api.message(message)));
            }
            module.target = (api.LLVMCreateTargetMachine)(
                target,
                triple,
                cpu.as_ptr(),
                features.as_ptr(),
                ir_opt_level.into(),
                2,
                1,
            );
            if module.target.is_null() {
                return Err(JitError::new("LLVM could not create host target machine"));
            }
            let mut names = Vec::new();
            let mut function = (api.LLVMGetFirstFunction)(module.module);
            while !function.is_null() {
                if (api.LLVMIsDeclaration)(function) == 0 {
                    let mut len = 0;
                    let name = (api.LLVMGetValueName2)(function, &mut len);
                    // Internal/private implementation helpers are compiled
                    // with their callers but have no public lookup contract.
                    if !matches!((api.LLVMGetLinkage)(function), 8 | 9) {
                        names.push(
                            String::from_utf8_lossy(std::slice::from_raw_parts(name.cast(), len))
                                .into_owned(),
                        );
                    }
                    // Override the emitter's generic target with live host ISA.
                    (api.LLVMAddTargetDependentFunctionAttr)(
                        function,
                        c"target-cpu".as_ptr(),
                        cpu.as_ptr(),
                    );
                    (api.LLVMAddTargetDependentFunctionAttr)(
                        function,
                        c"target-features".as_ptr(),
                        features.as_ptr(),
                    );
                }
                function = (api.LLVMGetNextFunction)(function);
            }
            if names.is_empty() {
                return Err(JitError::new(
                    "CPU JIT program defines no executable functions",
                ));
            }
            timings.llvm_setup += setup_start.elapsed();
            let profile_start = Instant::now();
            let instrumentation = match mode {
                ProfileMode::Instrument => Some(profile::instrument(
                    api,
                    llvm_context,
                    module.module,
                    ir,
                    options.profile_edge_counters,
                    options.profile_loop_edge_counters,
                )?),
                _ => None,
            };
            let profiled_branches = match mode {
                ProfileMode::Use(profile) => profile::apply(
                    api,
                    llvm_context,
                    module.module,
                    ir,
                    profile,
                    options.profile_loop_controls,
                )?,
                _ => 0,
            };
            let outer_unroll_annotations = if !matches!(mode, ProfileMode::Instrument)
                && options.final_loop_unrolling
                && !options.final_unroll_outer_loops
            {
                loops::disable_outer_unrolling(api, llvm_context, module.module)
            } else {
                0
            };
            timings.profile_setup = profile_start.elapsed();
            Self::verify_module(api, module.module, "LLVM verification", timings)?;
            let optimization_start = Instant::now();
            let passes = (api.LLVMCreatePassBuilderOptions)();
            (api.LLVMPassBuilderOptionsSetVerifyEach)(passes, options.verify_each_pass.into());
            if !matches!(mode, ProfileMode::Instrument) && !options.final_loop_unrolling {
                (api.LLVMPassBuilderOptionsSetLoopUnrolling)(passes, 0);
            }
            let pipeline = CString::new(format!("default<O{ir_opt_level}>")).unwrap();
            let pipeline_start = Instant::now();
            let pipeline_error =
                (api.LLVMRunPasses)(module.module, pipeline.as_ptr(), module.target, passes);
            timings.optimization_details.pipeline = pipeline_start.elapsed();
            let result = api.error(pipeline_error);
            if let Err(error) = result {
                (api.LLVMDisposePassBuilderOptions)(passes);
                return Err(error);
            }
            timings.optimization = optimization_start.elapsed();
            if let Err(error) = Self::verify_module(
                api,
                module.module,
                "LLVM verification after optimization",
                timings,
            ) {
                (api.LLVMDisposePassBuilderOptions)(passes);
                return Err(error);
            }
            let mut optimization_start = Instant::now();
            let mut profile_selection_optimization = false;
            if profiled_branches != 0 && options.opt_level != 0 {
                // Older compatible LLVM builds may omit this optional pass.
                // Measured weights still guide their standard O-level pipeline.
                let selection = c"require<profile-summary>,function(select-optimize)";
                let selection_start = Instant::now();
                let selection_error =
                    (api.LLVMRunPasses)(module.module, selection.as_ptr(), module.target, passes);
                timings.optimization_details.profile_selection = selection_start.elapsed();
                match api.error(selection_error) {
                    Ok(()) => profile_selection_optimization = true,
                    Err(error)
                        if error.to_string().contains("unknown function pass")
                            && error.to_string().contains("select-optimize") => {}
                    Err(error) => {
                        (api.LLVMDisposePassBuilderOptions)(passes);
                        return Err(JitError::new(format!(
                            "profile-guided select optimization: {error}"
                        )));
                    }
                }
                timings.optimization += optimization_start.elapsed();
                if let Err(error) = Self::verify_module(
                    api,
                    module.module,
                    "LLVM verification after profile-guided select optimization",
                    timings,
                ) {
                    (api.LLVMDisposePassBuilderOptions)(passes);
                    return Err(error);
                }
                optimization_start = Instant::now();
            }
            (api.LLVMDisposePassBuilderOptions)(passes);
            timings.optimization += optimization_start.elapsed();
            timings.optimization_details.finish(timings.optimization);
            let capture_start = Instant::now();
            let optimized_ir = api.message((api.LLVMPrintModuleToString)(module.module));
            timings.ir_capture = capture_start.elapsed();
            let symbols_start = Instant::now();

            // Diagnose live unresolved calls before ORC materialization. ORC's
            // default reporter sends their names only to stderr, otherwise a
            // library caller receives just a generic materialization error.
            let runtime_symbols = runtime::symbols();
            let mut function = (api.LLVMGetFirstFunction)(module.module);
            while !function.is_null() {
                if (api.LLVMIsDeclaration)(function) != 0
                    && !(api.LLVMGetFirstUse)(function).is_null()
                {
                    let mut len = 0;
                    let name = (api.LLVMGetValueName2)(function, &mut len);
                    let name =
                        String::from_utf8_lossy(std::slice::from_raw_parts(name.cast(), len));
                    if !name.starts_with("llvm.")
                        && !runtime_symbols.iter().any(|(n, _)| *n == name)
                    {
                        let c_name = CString::new(name.as_ref()).unwrap();
                        if llvm::process_symbol(&c_name).is_null() {
                            return Err(JitError::new(format!(
                                "CPU JIT runtime symbol `{name}` is unavailable"
                            )));
                        }
                    }
                }
                function = (api.LLVMGetNextFunction)(function);
            }

            // Native-width runtime functions are local to this JIT dylib.
            // Do not redefine user functions whose names happen to match.
            let mut symbols: Vec<_> = runtime_symbols
                .into_iter()
                .filter(|(name, _)| !names.iter().any(|n| n == name))
                .map(|(name, address)| {
                    let name = CString::new(name).unwrap();
                    llvm::SymbolPair {
                        name: (api.LLVMOrcLLJITMangleAndIntern)(jit, name.as_ptr()),
                        symbol: llvm::EvaluatedSymbol {
                            address: address as u64,
                            flags: llvm::SymbolFlags {
                                generic: 5,
                                target: 0,
                            },
                        },
                    }
                })
                .collect();
            let unit = (api.LLVMOrcAbsoluteSymbols)(symbols.as_mut_ptr(), symbols.len());
            if let Err(error) = api.error((api.LLVMOrcJITDylibDefine)(dylib, unit)) {
                (api.LLVMOrcDisposeMaterializationUnit)(unit);
                return Err(error);
            }
            // libc/memcpy/compiler helper symbols resolve through the host.
            let mut generator = null_mut();
            api.error((api.LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess)(
                &mut generator,
                (api.LLVMOrcLLJITGetGlobalPrefix)(jit),
                null_mut(),
                null_mut(),
            ))?;
            (api.LLVMOrcJITDylibAddGenerator)(dylib, generator);
            timings.symbol_resolution = symbols_start.elapsed();
            let materialization_start = Instant::now();
            let submission_start = Instant::now();
            let safe_module = (api.LLVMOrcCreateNewThreadSafeModule)(module.module, context);
            module.module = null_mut();
            api.error((api.LLVMOrcLLJITAddLLVMIRModule)(jit, dylib, safe_module))?;
            timings.materialization_details.submission = submission_start.elapsed();
            let mut addresses = BTreeMap::new();
            // Force all definitions to native code now, so lookup/calls do not
            // hide compilation in benchmark timings or defer link errors.
            for name in names {
                let first = timings.materialization_details.function_lookup_count == 0;
                let lookup_start = Instant::now();
                if first {
                    if let Some(observer) = &engine.observer {
                        observer.lock().first_start = Some(lookup_start);
                    }
                }
                let c_name = CString::new(name.as_str())
                    .map_err(|_| JitError::new("function name contains NUL"))?;
                let mut address = 0;
                api.error((api.LLVMOrcLLJITLookup)(jit, &mut address, c_name.as_ptr()))?;
                if address == 0 {
                    return Err(JitError::new(format!("LLVM returned null for `{name}`")));
                }
                let lookup_end = Instant::now();
                let lookup_time = lookup_end.duration_since(lookup_start);
                if first {
                    timings.materialization_details.first_lookup = lookup_time;
                    if let Some(observer) = &engine.observer {
                        observer.lock().first_end = Some(lookup_end);
                    }
                } else {
                    timings.materialization_details.remaining_function_lookups += lookup_time;
                }
                timings.materialization_details.function_lookup_count += 1;
                addresses.insert(name, address as usize);
            }
            let instrumentation = if let Some(instrumentation) = instrumentation {
                let lookup_start = Instant::now();
                let name = CString::new(instrumentation.symbol()).unwrap();
                let mut address = 0;
                api.error((api.LLVMOrcLLJITLookup)(jit, &mut address, name.as_ptr()))?;
                if address == 0 {
                    return Err(JitError::new("LLVM returned null branch counters"));
                }
                timings.materialization_details.profile_lookup = lookup_start.elapsed();
                timings.materialization_details.profile_lookup_count = 1;
                Some((instrumentation, address as usize))
            } else {
                None
            };
            if let Some(observer) = &engine.observer {
                observer.snapshot(&mut timings.materialization_details);
            }
            timings.materialization = materialization_start.elapsed();
            timings
                .materialization_details
                .finish(timings.materialization);
            Ok(CompiledModule {
                engine,
                addresses,
                optimized_ir,
                instrumentation,
                profiled_branches,
                profile_selection_optimization,
                outer_unroll_annotations,
            })
        }
    }
}
