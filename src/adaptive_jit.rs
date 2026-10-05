//! Opt-in adaptive execution of row-major F16 x F16 -> F32 GEMMs.
//!
//! Cold shapes retain an analytic tile's driver-compiled module. Hot shapes
//! measure a bounded candidate set on scratch inputs and only promote a
//! correctness-checked, measurably faster tile. By default tuning is deferred
//! to an explicit maintenance call and attempted once per resident shape.
//! Optional disk storage reuses successful decisions across runtimes.
//! This is not a general Y interpreter or a verified execution path.

mod cache;

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::autotuner::{AutotuneCandidate, Autotuner, Precision};
use crate::cuda_runtime::{CUdeviceptr, CudaContext, DeviceIdentity, KernelModule};
use crate::empirical_autotune::{self, LaunchConfig, MeasuredCandidate, Timing, TuneFailure};
use crate::sentinel::HardwareProfile;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl GemmShape {
    /// Initial range: M=1..16384; N and K are positive multiples of 16 up to
    /// 16384. The bound also keeps generated 32-bit offsets safe.
    pub fn validate(self) -> Result<(), String> {
        if self.m == 0
            || self.n == 0
            || self.k == 0
            || self.m > 16384
            || self.n > 16384
            || self.k > 16384
            || self.n % 16 != 0
            || self.k % 16 != 0
        {
            return Err(
                "adaptive GEMM requires M in 1..=16384 and N/K multiples of 16 in 16..=16384"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Controls where expensive empirical measurement may run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TuningPolicy {
    /// Launches track hotness; the caller chooses when to run `tune_hot`.
    #[default]
    Deferred,
    /// The first launch after the hot threshold synchronously tunes.
    OnLaunch,
    /// Compile and cache baselines without scheduling empirical tuning.
    Disabled,
}

#[derive(Clone, Debug)]
pub struct AdaptiveJitConfig {
    pub tuning_policy: TuningPolicy,
    /// Application-controlled cache directory. None (default) disables disk IO.
    /// Stores tuning decisions only, never executable code or PTX.
    pub cache_dir: Option<PathBuf>,
    /// Maximum owned decision files retained by a successful cache write.
    pub max_disk_cache_entries: usize,
    /// Successful enqueues before the following launch may tune.
    pub hot_threshold: u64,
    pub max_cached_shapes: usize,
    /// Maximum distinct generated kernels measured, including the baseline.
    pub max_candidates: usize,
    /// Required fractional latency reduction (0.05 = 5%).
    pub min_improvement: f64,
}

impl Default for AdaptiveJitConfig {
    fn default() -> Self {
        Self {
            tuning_policy: TuningPolicy::Deferred,
            cache_dir: None,
            max_disk_cache_entries: 128,
            hot_threshold: 32,
            max_cached_shapes: 16,
            max_candidates: 8,
            min_improvement: 0.05,
        }
    }
}

impl AdaptiveJitConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.hot_threshold == 0
            || self.max_disk_cache_entries == 0
            || self.max_cached_shapes == 0
            || self.max_candidates < 2
            || self.max_candidates > 64
            || !self.min_improvement.is_finite()
            || !(0.0..1.0).contains(&self.min_improvement)
        {
            return Err("adaptive JIT requires positive hot_threshold/memory and disk cache capacities, 2..=64 candidates, and finite min_improvement in [0, 1)".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitTier {
    Baseline,
    Tuned,
    /// No measured improvement justified replacement.
    RetainedBaseline,
    /// Tuning/code loading failed; subsequent calls reuse the baseline.
    TuningFailed,
    /// The baseline failed the numerical check; this shape can no longer run.
    RejectedBaseline,
}

#[derive(Clone, Debug)]
pub struct KernelStats {
    pub launches: u64,
    pub tier: JitTier,
    pub tuning_attempts: u32,
    pub tuning_time: Duration,
    /// Distinct candidates that passed the numerical gate and were timed.
    pub candidates_measured: usize,
    pub baseline_us: Option<f64>,
    pub selected_us: Option<f64>,
    pub tuning_error: Option<String>,
    /// True when a previous runtime's successful decision was restored.
    pub cache_hit: bool,
    /// Nonfatal persistence diagnostic; does not disable ordinary tuning.
    pub cache_error: Option<String>,
}

impl KernelStats {
    fn new() -> Self {
        Self {
            launches: 0,
            tier: JitTier::Baseline,
            tuning_attempts: 0,
            tuning_time: Duration::ZERO,
            candidates_measured: 0,
            baseline_us: None,
            selected_us: None,
            tuning_error: None,
            cache_hit: false,
            cache_error: None,
        }
    }

    /// Estimated additional calls needed to repay measured tuning wall time.
    /// Uses synthetic best-case kernel timings; this is not a workload forecast.
    /// Returns None without a promotion or when the estimate is not representable.
    pub fn estimated_break_even_launches(&self) -> Option<u64> {
        if self.cache_hit || self.tier != JitTier::Tuned {
            return None;
        }
        let (baseline, selected) = (self.baseline_us?, self.selected_us?);
        if !baseline.is_finite() || !selected.is_finite() || selected <= 0.0 || baseline <= selected
        {
            return None;
        }
        let calls = (self.tuning_time.as_secs_f64() * 1e6 / (baseline - selected)).ceil();
        if !calls.is_finite() || calls >= u64::MAX as f64 {
            None
        } else {
            Some(calls as u64)
        }
    }

    fn record_failure(&mut self, error: TuneFailure) {
        self.tier = if matches!(error, TuneFailure::BaselineIncorrect { .. }) {
            JitTier::RejectedBaseline
        } else {
            JitTier::TuningFailed
        };
        self.tuning_error = Some(error.to_string());
    }

    fn require_launchable(&self) -> Result<(), String> {
        if self.tier == JitTier::RejectedBaseline {
            Err(self
                .tuning_error
                .clone()
                .expect("rejection records its reason"))
        } else {
            Ok(())
        }
    }

    fn should_tune(&self, threshold: u64) -> bool {
        self.launches >= threshold && self.tuning_attempts == 0 && !self.cache_hit
    }
}

/// Result of one explicit tuning attempt, including a per-shape failure.
#[derive(Clone, Debug)]
pub struct TuningReport {
    pub shape: GemmShape,
    pub stats: KernelStats,
}

struct Entry {
    module: KernelModule,
    launch: LaunchConfig,
    candidate: AutotuneCandidate,
    kernel_hash: String,
    stats: KernelStats,
    last_used: u64,
}

impl Entry {
    // Called after validating the shape, pointers, and current context. Keep
    // dispatch and accounting together so failed enqueues never gain hotness
    // or refresh this entry's eviction order.
    fn enqueue(
        &mut self,
        ctx: &CudaContext,
        args: &[CUdeviceptr; 3],
        clock: &mut u64,
    ) -> Result<(), String> {
        self.stats.require_launchable()?;
        ctx.launch(
            &self.module,
            (self.launch.grid_x, self.launch.grid_y, 1),
            (self.launch.threads, 1, 1),
            self.launch.dyn_smem_bytes,
            args,
        )?;
        self.stats.launches = self.stats.launches.saturating_add(1);
        *clock = clock.saturating_add(1);
        self.last_used = *clock;
        Ok(())
    }
}

enum Context<'ctx> {
    Borrowed(&'ctx CudaContext),
    Owned(CudaContext),
}

impl Deref for Context<'_> {
    type Target = CudaContext;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(ctx) => ctx,
            Self::Owned(ctx) => ctx,
        }
    }
}

/// Cache tied to one CUDA context and its actual device profile.
///
/// Keep that context current on this thread while using or dropping the
/// runtime. Construction/launches reject a different current context. No
/// global tuning modes, hardware profiles, or verified artifacts change.
/// Optional cache files contain decisions only. Eviction forgets launch
/// counters; persisted decisions can be reused if their identity still matches.
pub struct AdaptiveGemm<'ctx> {
    // Declaration order releases CUDA modules before an owned context.
    entries: HashMap<GemmShape, Entry>,
    ctx: Context<'ctx>,
    profile: HardwareProfile,
    config: AdaptiveJitConfig,
    clock: u64,
    cache_namespace: Option<String>,
    cache_unavailable: Option<String>,
}

impl AdaptiveGemm<'static> {
    /// Own the context wrapper for an embedding API without self-references.
    /// A wrapper made by `CudaContext::borrow_current` still leaves ownership
    /// of the underlying CUDA context with its host; its safety contract applies.
    pub fn from_owned_context(ctx: CudaContext, config: AdaptiveJitConfig) -> Result<Self, String> {
        Self::with_context(Context::Owned(ctx), config)
    }
}

impl<'ctx> AdaptiveGemm<'ctx> {
    pub fn new(ctx: &'ctx CudaContext, config: AdaptiveJitConfig) -> Result<Self, String> {
        Self::with_context(Context::Borrowed(ctx), config)
    }

    fn with_context(ctx: Context<'ctx>, config: AdaptiveJitConfig) -> Result<Self, String> {
        config.validate()?;
        require_default_padding()?;
        ctx.require_current()?;
        let attribute = |id| {
            ctx.device_attribute(id)
                .filter(|v| *v > 0)
                .map(|v| v as u32)
                .ok_or_else(|| format!("could not query CUDA device attribute {id}"))
        };
        let major = attribute(75)?;
        let minor = ctx
            .device_attribute(76)
            .filter(|v| *v >= 0)
            .ok_or("could not query CUDA compute capability minor")?;
        if major < 8 {
            return Err("adaptive GEMM currently requires sm_80 or newer".into());
        }
        let max_threads_per_sm = attribute(39)?;
        let warp_size = attribute(10)?;
        let profile = HardwareProfile {
            gpu_name: ctx.device_name().to_owned(),
            gpu_vendor: "NVIDIA".into(),
            sm_version: format!("sm_{major}{minor}"),
            compute_capability: format!("{major}.{minor}"),
            sm_count: attribute(16)?,
            max_smem_per_sm_bytes: attribute(81)?.min(attribute(97)?),
            max_regs_per_sm: attribute(82)?,
            max_regs_per_thread: 255,
            max_threads_per_sm,
            warp_size,
            max_warps_per_sm: max_threads_per_sm / warp_size,
            ..HardwareProfile::default()
        };
        let (cache_namespace, cache_unavailable) =
            if config.cache_dir.is_some() && config.tuning_policy != TuningPolicy::Disabled {
                match ctx.cache_identity() {
                    Ok(identity) => (
                        Some(cache_namespace(&identity, &profile, compiler_fingerprint())),
                        None,
                    ),
                    Err(error) => (None, Some(format!("persistent cache unavailable: {error}"))),
                }
            } else {
                (None, None)
            };
        Ok(Self {
            entries: HashMap::new(),
            ctx,
            profile,
            config,
            clock: 0,
            cache_namespace,
            cache_unavailable,
        })
    }

    /// Check context identity without changing the calling thread's context.
    pub fn require_current(&self) -> Result<(), String> {
        self.ctx.require_current()
    }

    /// Finish submitted work in this runtime's context before releasing buffers
    /// or closing an embedding handle. Rejects a different current context.
    pub fn synchronize(&self) -> Result<(), String> {
        self.require_current()?;
        self.ctx.synchronize()
    }

    pub fn stats(&self, shape: GemmShape) -> Option<&KernelStats> {
        self.entries.get(&shape).map(|entry| &entry.stats)
    }

    pub fn cached_shapes(&self) -> usize {
        self.entries.len()
    }

    /// Compile a shape without launching or increasing its hotness. Use this
    /// during warmup to move first-use compilation out of a latency-sensitive
    /// phase. Successful preparation also updates the cache's LRU ordering.
    pub fn prepare(&mut self, shape: GemmShape) -> Result<(), String> {
        shape.validate()?;
        self.ctx.require_current()?;
        self.ensure_entry(shape)?;
        self.entries[&shape].stats.require_launchable()?;
        self.touch(shape);
        Ok(())
    }

    /// Hot resident shapes that have not been tuned. Most-used shapes come
    /// first, with least-recently used first on ties. Disabled policy has none.
    /// Evicted shapes cannot leave stale jobs in this bounded work list.
    pub fn pending_shapes(&self) -> Vec<GemmShape> {
        if self.config.tuning_policy == TuningPolicy::Disabled {
            return Vec::new();
        }
        let mut pending: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, e)| e.stats.should_tune(self.config.hot_threshold))
            .collect();
        pending.sort_by(|(a, ea), (b, eb)| {
            eb.stats
                .launches
                .cmp(&ea.stats.launches)
                .then(ea.last_used.cmp(&eb.last_used))
                .then((a.m, a.n, a.k).cmp(&(b.m, b.n, b.k)))
        });
        pending.into_iter().map(|(shape, _)| *shape).collect()
    }

    /// Synchronously tune up to `max_shapes` pending shapes at a caller-chosen
    /// warmup/maintenance point. This can take seconds per shape. The bound is
    /// a job count, not a wall-time deadline; the measurement clock ramp stays
    /// intact. Zero performs no tuning. No user-buffer launches are counted.
    ///
    /// Per-shape failures are included in reports and recorded in `stats`.
    /// A context error returns Err. A numerically rejected baseline remains
    /// blocked on all subsequent dispatches while it is resident.
    pub fn tune_hot(&mut self, max_shapes: usize) -> Result<Vec<TuningReport>, String> {
        self.ctx.require_current()?;
        let mut reports = Vec::new();
        for shape in self.pending_shapes().into_iter().take(max_shapes) {
            self.ctx.require_current()?;
            self.tune(shape);
            reports.push(TuningReport {
                shape,
                stats: self.entries[&shape].stats.clone(),
            });
        }
        Ok(reports)
    }

    /// Enqueue C = A * B on the default stream. Deferred (default) and Disabled
    /// policies never tune here. Cold compilation, cache eviction, and CUDA
    /// calls can still block; prepare the needed shapes first for steady reuse.
    /// OnLaunch policy can additionally block for seconds once a shape is hot.
    /// Synchronize before reading results or releasing inputs.
    ///
    /// # Safety
    /// A/B/C must be allocations in this context, aligned to at least 16
    /// bytes, live through completion, and contain at least M*K F16, K*N F16,
    /// and M*N F32 elements, in contiguous row-major order. C must not overlap
    /// A or B. Coordinate access with other streams/threads. Pointer ranges
    /// and allocation ownership cannot be checked by this API.
    pub unsafe fn launch(
        &mut self,
        shape: GemmShape,
        a: CUdeviceptr,
        b: CUdeviceptr,
        c: CUdeviceptr,
    ) -> Result<(), String> {
        shape.validate()?;
        self.ctx.require_current()?;
        if [a, b, c].iter().any(|ptr| *ptr == 0 || *ptr % 16 != 0) {
            return Err("adaptive GEMM requires non-null, 16-byte-aligned device pointers".into());
        }
        // A resident shape usually needs only one lookup. The context check
        // above remains mandatory; no operation in this branch can switch it.
        if let Some(entry) = self.entries.get_mut(&shape) {
            let needs_tuning = self.config.tuning_policy == TuningPolicy::OnLaunch
                && entry.stats.should_tune(self.config.hot_threshold);
            if !needs_tuning {
                return entry.enqueue(&self.ctx, &[a, b, c], &mut self.clock);
            }
        }
        self.ensure_entry(shape)?;
        if self.config.tuning_policy == TuningPolicy::OnLaunch
            && self.entries[&shape]
                .stats
                .should_tune(self.config.hot_threshold)
        {
            self.tune(shape);
        }
        // Compilation, restoration, or tuning happened on the slow path;
        // recheck the context before dispatching its resulting module.
        self.ctx.require_current()?;
        self.entries
            .get_mut(&shape)
            .expect("shape compiled above")
            .enqueue(&self.ctx, &[a, b, c], &mut self.clock)
    }

    fn ensure_entry(&mut self, shape: GemmShape) -> Result<(), String> {
        if !self.entries.contains_key(&shape) {
            // A failed compile preserves the existing cache.
            let mut entry = self.compile_baseline(shape)?;
            self.restore_cached(shape, &mut entry);
            if self.entries.len() == self.config.max_cached_shapes {
                self.ctx.synchronize()?;
                let oldest = *self
                    .entries
                    .iter()
                    .min_by_key(|(_, e)| e.last_used)
                    .expect("nonempty bounded cache")
                    .0;
                self.entries.remove(&oldest);
            }
            self.entries.insert(shape, entry);
        }
        Ok(())
    }

    fn touch(&mut self, shape: GemmShape) {
        self.clock = self.clock.saturating_add(1);
        self.entries
            .get_mut(&shape)
            .expect("resident shape")
            .last_used = self.clock;
    }

    fn ranked_candidates(&self, shape: GemmShape) -> Vec<AutotuneCandidate> {
        let mut candidates =
            Autotuner::generate_candidates(shape.m, shape.n, shape.k, Precision::F16);
        candidates.retain(|c| empirical_autotune::is_emittable(c, shape.k));
        candidates.sort_by(|a, b| {
            let score = |c| Autotuner::score_candidate(c, shape.m, shape.n, shape.k, &self.profile);
            score(b).total_cmp(&score(a))
        });
        candidates
    }

    fn compile(
        &self,
        shape: GemmShape,
        candidate: &AutotuneCandidate,
    ) -> Result<(KernelModule, LaunchConfig, String), String> {
        require_default_padding()?;
        let (ptx, launch) = empirical_autotune::emit_candidate_ptx(
            shape.m,
            shape.n,
            shape.k,
            candidate,
            &self.profile,
        )?;
        let module = self.ctx.load_ptx(&ptx, empirical_autotune::PROBE_KERNEL)?;
        if launch.dyn_smem_bytes > 0 {
            module.set_max_dynamic_smem(launch.dyn_smem_bytes)?;
        }
        Ok((module, launch, kernel_fingerprint(&ptx, launch)))
    }

    fn compile_baseline(&self, shape: GemmShape) -> Result<Entry, String> {
        let mut error = "no emittable GEMM candidate".to_owned();
        for candidate in self
            .ranked_candidates(shape)
            .into_iter()
            .take(self.config.max_candidates)
        {
            match self.compile(shape, &candidate) {
                Ok((module, launch, kernel_hash)) => {
                    return Ok(Entry {
                        module,
                        launch,
                        candidate,
                        kernel_hash,
                        stats: KernelStats::new(),
                        last_used: 0,
                    })
                }
                Err(e) => error = e,
            }
        }
        Err(format!("could not compile adaptive GEMM baseline: {error}"))
    }

    fn cache_key(&self, shape: GemmShape) -> Option<String> {
        self.cache_namespace
            .as_ref()
            .map(|namespace| decision_key(namespace, shape, &self.config))
    }

    fn restore_cached(&self, shape: GemmShape, entry: &mut Entry) {
        if self.config.tuning_policy == TuningPolicy::Disabled {
            return;
        }
        if let Some(reason) = &self.cache_unavailable {
            entry.stats.cache_error = Some(reason.clone());
            return;
        }
        let (Some(dir), Some(key)) = (&self.config.cache_dir, self.cache_key(shape)) else {
            return;
        };
        let result = cache::load(dir, &key).and_then(|record| {
            let Some(record) = record else {
                return Ok(false);
            };
            if record.baseline_hash != entry.kernel_hash {
                return Err("cached baseline code does not match current compilation".into());
            }
            // Validate settings against today's generated search space BEFORE
            // codegen: disk integers must not reach unchecked emitter arithmetic.
            if !self.ranked_candidates(shape).contains(&record.candidate) {
                return Err("cached candidate is outside the current search space".into());
            }
            if record.candidates_measured > self.config.max_candidates {
                return Err("cached measurement exceeds the current candidate budget".into());
            }
            if record.promoted {
                if record.candidate == entry.candidate
                    || record.baseline_us - record.selected_us
                        <= record.baseline_us * self.config.min_improvement
                {
                    return Err("cached promotion does not satisfy the current policy".into());
                }
                require_default_padding()?;
                let (ptx, launch) = empirical_autotune::emit_candidate_ptx(
                    shape.m,
                    shape.n,
                    shape.k,
                    &record.candidate,
                    &self.profile,
                )?;
                if launch != record.launch
                    || kernel_fingerprint(&ptx, launch) != record.selected_hash
                {
                    return Err("cached selected kernel does not match current compilation".into());
                }
                // All code comes from the current emitter. No cached code is loaded.
                let module = self.ctx.load_ptx(&ptx, empirical_autotune::PROBE_KERNEL)?;
                if launch.dyn_smem_bytes > 0 {
                    module.set_max_dynamic_smem(launch.dyn_smem_bytes)?;
                }
                entry.module = module;
                entry.launch = launch;
                entry.candidate = record.candidate;
                entry.kernel_hash = record.selected_hash;
                entry.stats.tier = JitTier::Tuned;
            } else {
                if record.candidate != entry.candidate
                    || record.launch != entry.launch
                    || record.selected_hash != entry.kernel_hash
                {
                    return Err(
                        "cached retained baseline does not match current compilation".into(),
                    );
                }
                entry.stats.tier = JitTier::RetainedBaseline;
            }
            entry.stats.baseline_us = Some(record.baseline_us);
            entry.stats.selected_us = Some(record.selected_us);
            entry.stats.cache_hit = true;
            // Historical timings are restored; this session has done no tuning.
            Ok(true)
        });
        if let Err(error) = result {
            entry.stats.cache_error = Some(error);
        }
    }

    fn persist_decision(&mut self, shape: GemmShape, baseline_hash: String) {
        let (Some(dir), Some(key)) = (&self.config.cache_dir, self.cache_key(shape)) else {
            return;
        };
        let entry = self.entries.get_mut(&shape).unwrap();
        if !matches!(entry.stats.tier, JitTier::Tuned | JitTier::RetainedBaseline) {
            return;
        }
        let record = cache::CacheRecord {
            key,
            candidate: entry.candidate.clone(),
            baseline_hash,
            selected_hash: entry.kernel_hash.clone(),
            launch: entry.launch,
            baseline_us: entry.stats.baseline_us.unwrap(),
            selected_us: entry.stats.selected_us.unwrap(),
            candidates_measured: entry.stats.candidates_measured,
            promoted: entry.stats.tier == JitTier::Tuned,
        };
        entry.stats.cache_error =
            cache::store(dir, &record, self.config.max_disk_cache_entries).err();
    }

    fn tune(&mut self, shape: GemmShape) {
        let baseline = self.entries[&shape].candidate.clone();
        let baseline_hash = self.entries[&shape].kernel_hash.clone();
        self.entries.get_mut(&shape).unwrap().stats.tuning_attempts += 1;
        let started = Instant::now();
        // Baseline first so deduplication preserves its measured identity.
        // Bound unique generated kernels, not requests that clamp to the same
        // effective pipeline depth. Cold compilation still stops at its first
        // loadable baseline and does not emit this full search space.
        let mut candidates = vec![baseline.clone()];
        candidates.extend(
            self.ranked_candidates(shape)
                .into_iter()
                .filter(|c| *c != baseline),
        );
        let measured = require_default_padding()
            .map_err(TuneFailure::Cuda)
            .and_then(|()| {
                empirical_autotune::distinct_gemm_candidates(
                    shape.m,
                    shape.n,
                    shape.k,
                    &self.profile,
                    &candidates,
                    self.config.max_candidates,
                )
                .map_err(TuneFailure::Cuda)
            })
            .and_then(|distinct| {
                empirical_autotune::tune_gemm_f16_in_context(
                    &self.ctx,
                    shape.m,
                    shape.n,
                    shape.k,
                    &self.profile,
                    &distinct,
                    false,
                    &baseline,
                    self.config.min_improvement,
                )
            });
        if let Ok(results) = &measured {
            self.entries
                .get_mut(&shape)
                .unwrap()
                .stats
                .candidates_measured = match results {
                empirical_autotune::AdaptiveGemmMeasurement::Screened {
                    candidates_measured,
                    ..
                } => *candidates_measured,
                empirical_autotune::AdaptiveGemmMeasurement::Confirmed(results) => results.len(),
            };
        }
        let result = measured.and_then(|measurement| {
            let measured = match measurement {
                // A short screen can decide to keep a numerically checked
                // baseline, but cannot license a challenger for promotion.
                empirical_autotune::AdaptiveGemmMeasurement::Screened { baseline_us, .. } => {
                    return Ok((baseline_us, None));
                }
                empirical_autotune::AdaptiveGemmMeasurement::Confirmed(results) => results,
            };
            let original = measured
                .iter()
                .find(|r| r.candidate == baseline && r.finalist)
                .ok_or_else(|| {
                    TuneFailure::NoUsableCandidate(
                        "baseline did not pass the tuning correctness/timing gate".into(),
                    )
                })?;
            let winner =
                promotion_candidate(&measured, &original.timing, self.config.min_improvement);
            let compiled = winner
                .map(|r| {
                    self.compile(shape, &r.candidate)
                        .map(|(module, launch, hash)| {
                            (module, launch, hash, r.candidate.clone(), r.us())
                        })
                })
                .transpose()
                .map_err(TuneFailure::Cuda)?;
            self.ctx.synchronize().map_err(TuneFailure::Cuda)?;
            Ok((original.us(), compiled))
        });
        let entry = self.entries.get_mut(&shape).unwrap();
        entry.stats.tuning_time = started.elapsed();
        match result {
            Ok((baseline_us, replacement)) => {
                entry.stats.baseline_us = Some(baseline_us);
                if let Some((module, launch, hash, candidate, selected_us)) = replacement {
                    entry.kernel_hash = hash;
                    entry.module = module;
                    entry.launch = launch;
                    entry.candidate = candidate;
                    entry.stats.selected_us = Some(selected_us);
                    entry.stats.tier = JitTier::Tuned;
                } else {
                    entry.stats.selected_us = Some(baseline_us);
                    entry.stats.tier = JitTier::RetainedBaseline;
                }
            }
            Err(error) => entry.stats.record_failure(error),
        }
        self.persist_decision(shape, baseline_hash);
    }
}

// Length-prefix parts so names/fields cannot ambiguously concatenate.
fn digest_parts(parts: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part);
    }
    format!("{:x}", digest.finalize())
}

fn kernel_fingerprint(ptx: &str, launch: LaunchConfig) -> String {
    let geometry = [
        launch.grid_x,
        launch.grid_y,
        launch.threads,
        launch.dyn_smem_bytes,
    ]
    .into_iter()
    .flat_map(u32::to_le_bytes)
    .collect::<Vec<_>>();
    digest_parts(&[ptx.as_bytes(), &geometry])
}

fn cache_namespace(identity: &DeviceIdentity, profile: &HardwareProfile, compiler: &str) -> String {
    let hardware = format!("{:?}", profile);
    digest_parts(&[
        b"y-adaptive-gemm-f16-row-major-v1",
        &identity.uuid,
        &identity.driver_version.to_le_bytes(),
        identity.driver_build.as_bytes(),
        hardware.as_bytes(),
        compiler.as_bytes(),
    ])
}

fn decision_key(namespace: &str, shape: GemmShape, config: &AdaptiveJitConfig) -> String {
    digest_parts(&[
        namespace.as_bytes(),
        &shape.m.to_le_bytes(),
        &shape.n.to_le_bytes(),
        &shape.k.to_le_bytes(),
        &(config.max_candidates as u64).to_le_bytes(),
        &config.min_improvement.to_bits().to_le_bytes(),
    ])
}

// Source identity includes codegen, ranking, numerical/timing gates, cache
// semantics and locked dependencies. Actual baseline/selected PTX+launch hashes
// are additionally checked on read. Compute once, only with persistence enabled.
fn compiler_fingerprint() -> &'static str {
    static FINGERPRINT: OnceLock<String> = OnceLock::new();
    FINGERPRINT.get_or_init(|| {
        digest_parts(&[
            include_bytes!("adaptive_jit.rs"),
            include_bytes!("adaptive_jit/cache.rs"),
            include_bytes!("ptx_emitter.rs"),
            include_bytes!("empirical_autotune.rs"),
            include_bytes!("autotuner.rs"),
            include_bytes!("cuda_runtime.rs"),
            include_bytes!("ast.rs"),
            include_bytes!("lexer.rs"),
            include_bytes!("parser.rs"),
            include_bytes!("type_checker.rs"),
            include_bytes!("linear_tracker.rs"),
            include_bytes!("intrinsics.rs"),
            include_bytes!("sentinel.rs"),
            include_bytes!("bank_conflict.rs"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            env!("CARGO_PKG_VERSION").as_bytes(),
            &[cfg!(debug_assertions) as u8, cfg!(feature = "zk") as u8],
        ])
    })
}

// The emitter's experimental override is unchecked and can generate misaligned
// vector loads. Keep this first runtime on the scorer's standard +8 layout.
fn require_default_padding() -> Result<(), String> {
    match std::env::var("Y_SMEM_PAD") {
        Err(std::env::VarError::NotPresent) => Ok(()),
        Ok(value) if value.parse::<u32>() == Ok(8) => Ok(()),
        _ => Err("adaptive GEMM requires Y_SMEM_PAD unset or equal to 8".into()),
    }
}

fn promotion_candidate<'a>(
    results: &'a [MeasuredCandidate],
    baseline: &Timing,
    min_improvement: f64,
) -> Option<&'a MeasuredCandidate> {
    results
        .iter()
        .filter(|r| r.finalist && clearly_faster(baseline, &r.timing, min_improvement))
        .min_by(|a, b| a.us().total_cmp(&b.us()))
}

fn clearly_faster(baseline: &Timing, candidate: &Timing, min_improvement: f64) -> bool {
    let valid = |t: &Timing| {
        t.best_us.is_finite()
            && t.best_us > 0.0
            && t.median_us.is_finite()
            && t.median_us >= t.best_us
            && t.max_us.is_finite()
            && t.max_us >= t.median_us
    };
    if !valid(baseline) || !valid(candidate) {
        return false;
    }
    let noise = (baseline.median_us - baseline.best_us)
        .max(candidate.median_us - candidate.best_us)
        .max(0.0025 * baseline.best_us.min(candidate.best_us));
    baseline.best_us - candidate.best_us > (baseline.best_us * min_improvement).max(noise)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_requires_a_resolved_improvement() {
        let timing = |best_us, median_us| Timing {
            best_us,
            median_us,
            max_us: median_us,
        };
        let baseline = timing(100.0, 102.0);
        assert!(clearly_faster(&baseline, &timing(90.0, 92.0), 0.05));
        assert!(!clearly_faster(&baseline, &timing(96.0, 97.0), 0.05));
        assert!(!clearly_faster(&baseline, &timing(90.0, 105.0), 0.05));
        assert!(!clearly_faster(&baseline, &timing(110.0, 112.0), 0.05));
        assert!(!clearly_faster(&baseline, &timing(f64::NAN, 1.0), 0.05));
        assert!(!clearly_faster(&baseline, &timing(0.0, 1.0), 0.05));
    }

    #[test]
    fn hotness_and_failure_do_not_retry_tuning() {
        let mut stats = KernelStats::new();
        stats.launches = 31;
        assert!(!stats.should_tune(32));
        stats.launches = 32;
        assert!(stats.should_tune(32));
        stats.tuning_attempts = 1;
        stats.tier = JitTier::TuningFailed;
        stats.launches = u64::MAX;
        assert!(!stats.should_tune(32));
    }

    #[test]
    fn a_known_incorrect_baseline_is_rejected_instead_of_used_as_fallback() {
        let mut stats = KernelStats::new();
        stats.tuning_attempts = 1;
        stats.launches = 64;
        stats.record_failure(TuneFailure::BaselineIncorrect { relative_l2: 1.0 });
        assert_eq!(stats.tier, JitTier::RejectedBaseline);
        assert!(stats.tuning_error.as_ref().unwrap().contains("baseline"));
        assert!(!stats.should_tune(1));
        assert!(stats.require_launchable().is_err());
        assert!(stats.require_launchable().is_err());
        let mut failed_tuning = KernelStats::new();
        failed_tuning.record_failure(TuneFailure::Cuda("out of memory".into()));
        assert_eq!(failed_tuning.tier, JitTier::TuningFailed);
        assert!(failed_tuning.require_launchable().is_ok());
    }

    #[test]
    fn coarse_screening_results_cannot_trigger_promotion() {
        let candidate = Autotuner::generate_candidates(256, 256, 256, Precision::F16)[0].clone();
        let result = MeasuredCandidate {
            candidate,
            timing: Timing {
                best_us: 10.0,
                median_us: 11.0,
                max_us: 12.0,
            },
            tflops: 1.0,
            finalist: false,
        };
        let baseline = Timing {
            best_us: 100.0,
            median_us: 101.0,
            max_us: 102.0,
        };
        assert!(promotion_candidate(&[result], &baseline, 0.05).is_none());
    }

    #[test]
    fn break_even_estimate_requires_real_positive_savings() {
        let mut stats = KernelStats::new();
        assert_eq!(stats.estimated_break_even_launches(), None);
        stats.tier = JitTier::Tuned;
        stats.tuning_time = Duration::from_secs(5);
        stats.baseline_us = Some(12.0);
        stats.selected_us = Some(10.0);
        assert_eq!(stats.estimated_break_even_launches(), Some(2_500_000));
        stats.tuning_time = Duration::from_micros(5);
        assert_eq!(stats.estimated_break_even_launches(), Some(3));
        for selected in [12.0, 13.0, 0.0, f64::NAN, f64::INFINITY] {
            stats.selected_us = Some(selected);
            assert_eq!(stats.estimated_break_even_launches(), None);
        }
        stats.selected_us = Some(10.0);
        stats.tier = JitTier::RetainedBaseline;
        assert_eq!(stats.estimated_break_even_launches(), None);
    }

    #[test]
    fn persistent_identity_separates_device_driver_compiler_shape_and_policy() {
        let hw = HardwareProfile::default();
        let identity = DeviceIdentity {
            uuid: [1; 16],
            driver_version: 13020,
            driver_build: "driver-build-a".into(),
        };
        let namespace = cache_namespace(&identity, &hw, "compiler-a");
        assert_ne!(
            namespace,
            cache_namespace(
                &DeviceIdentity {
                    uuid: [2; 16],
                    ..identity.clone()
                },
                &hw,
                "compiler-a"
            )
        );
        assert_ne!(
            namespace,
            cache_namespace(
                &DeviceIdentity {
                    driver_version: 13030,
                    ..identity.clone()
                },
                &hw,
                "compiler-a"
            )
        );
        assert_ne!(
            namespace,
            cache_namespace(
                &DeviceIdentity {
                    driver_build: "driver-build-b".into(),
                    ..identity.clone()
                },
                &hw,
                "compiler-a"
            )
        );
        assert_ne!(
            namespace,
            cache_namespace(
                &identity,
                &HardwareProfile {
                    sm_count: 66,
                    ..hw.clone()
                },
                "compiler-a"
            )
        );
        assert_ne!(namespace, cache_namespace(&identity, &hw, "compiler-b"));
        let shape = GemmShape {
            m: 16,
            n: 32,
            k: 64,
        };
        let config = AdaptiveJitConfig::default();
        let key = decision_key(&namespace, shape, &config);
        for different in [
            GemmShape { m: 17, ..shape },
            GemmShape { n: 48, ..shape },
            GemmShape { k: 80, ..shape },
        ] {
            assert_ne!(key, decision_key(&namespace, different, &config));
        }
        assert_ne!(
            key,
            decision_key(
                &namespace,
                shape,
                &AdaptiveJitConfig {
                    max_candidates: 4,
                    ..config.clone()
                }
            )
        );
        assert_ne!(
            key,
            decision_key(
                &namespace,
                shape,
                &AdaptiveJitConfig {
                    min_improvement: 0.1,
                    ..config.clone()
                }
            )
        );
        // Scheduling and storage location do not change what was measured.
        assert_eq!(
            key,
            decision_key(
                &namespace,
                shape,
                &AdaptiveJitConfig {
                    hot_threshold: 5,
                    tuning_policy: TuningPolicy::OnLaunch,
                    cache_dir: Some(PathBuf::from("different-directory")),
                    ..config
                }
            )
        );
    }

    #[test]
    fn persistent_kernel_identity_includes_code_and_each_launch_requirement() {
        let launch = LaunchConfig {
            grid_x: 1,
            grid_y: 2,
            threads: 128,
            dyn_smem_bytes: 8192,
        };
        let hash = kernel_fingerprint("ret;", launch);
        assert_ne!(hash, kernel_fingerprint("exit;", launch));
        for changed in [
            LaunchConfig {
                grid_x: 2,
                ..launch
            },
            LaunchConfig {
                grid_y: 3,
                ..launch
            },
            LaunchConfig {
                threads: 64,
                ..launch
            },
            LaunchConfig {
                dyn_smem_bytes: 4096,
                ..launch
            },
        ] {
            assert_ne!(hash, kernel_fingerprint("ret;", changed));
        }
        assert_ne!(digest_parts(&[b"ab", b"c"]), digest_parts(&[b"a", b"bc"]));
    }

    #[test]
    fn restored_decisions_do_not_claim_new_tuning_or_schedule_more_work() {
        let mut stats = KernelStats::new();
        stats.tier = JitTier::Tuned;
        stats.cache_hit = true;
        stats.baseline_us = Some(10.0);
        stats.selected_us = Some(8.0);
        stats.launches = u64::MAX;
        assert_eq!(stats.tuning_attempts, 0);
        assert_eq!(stats.candidates_measured, 0);
        assert!(stats.tuning_time.is_zero());
        assert!(!stats.should_tune(1));
        assert_eq!(stats.estimated_break_even_launches(), None);
        assert!(stats.require_launchable().is_ok());
    }

    #[test]
    fn invalid_shapes_and_policies_are_rejected() {
        for shape in [
            GemmShape { m: 0, n: 16, k: 16 },
            GemmShape { m: 1, n: 17, k: 16 },
            GemmShape {
                m: u32::MAX,
                n: 16,
                k: 16,
            },
            GemmShape { m: 1, n: 16, k: 8 },
        ] {
            assert!(shape.validate().is_err());
        }
        assert!(GemmShape {
            m: 1,
            n: 4096,
            k: 4096
        }
        .validate()
        .is_ok());
        assert!(AdaptiveJitConfig::default().validate().is_ok());
        assert!(AdaptiveJitConfig {
            max_disk_cache_entries: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(AdaptiveJitConfig {
            min_improvement: f64::NAN,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(AdaptiveJitConfig {
            hot_threshold: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(AdaptiveJitConfig {
            max_candidates: 1,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(AdaptiveJitConfig {
            max_cached_shapes: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
    }
}
