// ============================================================
//  Y  —  Packed AVX-512 SGEMM codegen for the LLVM/CPU backend
//  cpu_gemm.rs
//
//  Recognises the canonical Y matmul loop nest and emits a
//  BLIS-shaped blocked kernel in its place: pack A and B into
//  cache-resident panels, then run a register-blocked
//  MR x NR micro-kernel over explicit <16 x float> vectors.
//
//  The blocking parameters below are measured, not assumed —
//  see `docs/cpu_gemm_tuning.md` for the sweep they came from.
// ============================================================

#![allow(dead_code)]

use crate::ast::*;
use std::fmt::Write;

/// Register-blocking and cache-blocking parameters.
///
/// These were constants, and being constants is why `NRV` was never swept:
/// changing it meant editing and rebuilding the compiler, so every recorded
/// sweep moved `MR` alone and left the `MR x NRV` *shape* at 12x2 by default
/// rather than by measurement. They are overridable at emit time now —
/// `Y_GEMM_TILE=mr,nr,kc,mc,nc` — so a sweep is a shell loop.
///
/// The override behaves like `Y_CTA_OVERRIDE` on the PTX side and carries the
/// same trap: **it persists in whatever `.ll` was emitted while it was set.**
/// Regenerate `tests/y_cpu_matmul.ll` after sweeping or later measurements are
/// silently attributed to the default tile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tile {
    /// Rows of C held in registers by the micro-kernel.
    pub mr: usize,
    /// Columns of C held in registers, in floats. A multiple of 16.
    pub nr: usize,
    /// K-panel depth. `nr * kc * 4` should sit in L1d with the A micro-panel.
    pub kc: usize,
    /// M-block. `mc * kc * 4` should fit L2.
    pub mc: usize,
    /// N-block. `kc * nc * 4` should stream through L3.
    pub nc: usize,
}

impl Tile {
    /// Vectors per micro-kernel row: `nr / 16`.
    pub fn nrv(&self) -> usize {
        self.nr / 16
    }

    /// zmm registers the micro-kernel needs: accumulators, the B vectors, and
    /// two A broadcasts live at once.
    pub fn regs(&self) -> usize {
        self.mr * self.nrv() + self.nrv() + 2
    }

    /// Reject rather than emit a kernel that cannot work. A tile over the
    /// register budget compiles and runs — it just spills every accumulator to
    /// the stack in the innermost loop, which reads as a mysterious 3x slowdown
    /// rather than as a bad tile.
    pub fn check(&self) -> Result<(), String> {
        if self.nr % 16 != 0 || self.nr == 0 {
            return Err(format!("nr={} must be a non-zero multiple of 16", self.nr));
        }
        if self.mr == 0 || self.kc == 0 || self.mc == 0 || self.nc == 0 {
            return Err("mr, kc, mc and nc must all be non-zero".into());
        }
        if self.regs() > 32 {
            return Err(format!(
                "tile {}x{} needs {} zmm registers (max 32); accumulators would spill",
                self.mr,
                self.nr,
                self.regs()
            ));
        }
        if self.mr > MR_MAX || self.nr > NR_MAX || self.kc > KC_MAX
            || self.mc > MC_MAX || self.nc > NC_MAX
        {
            return Err(format!(
                "tile exceeds the scratch reserved by SCRATCH_FLOATS \
                 (max mr={} nr={} kc={} mc={} nc={})",
                MR_MAX, NR_MAX, KC_MAX, MC_MAX, NC_MAX
            ));
        }
        Ok(())
    }
}

/// Measured default: see `docs/cpu_gemm_tuning.md`.
pub const DEFAULT_TILE: Tile = Tile { mr: 6, nr: 64, kc: 256, mc: 192, nc: 2048 };

/// Floats of packed B the whole process may hold across all threads.
///
/// 32 MB against this part's 64 MB L3 — half, because the A panels, the C
/// block being accumulated and OpenBLAS-style streaming all want room too.
/// Divided by `nthr * nc` at runtime to pick the K-panel depth; see the long
/// comment at the `g.pc` loop in `emit_driver`.
pub const L3_PANEL_FLOATS: usize = 8 * 1024 * 1024;

/// Floor for the runtime K-panel depth. Below this the per-panel pack and the
/// C read-modify-write stop being amortised by anything.
pub const KC_MIN: usize = 64;

/// Upper bounds the static scratch is sized for. A tile past these is refused
/// rather than allowed to run off the end of the buffer.
pub const MR_MAX: usize = 16;
pub const NR_MAX: usize = 64;
pub const KC_MAX: usize = 1024;
pub const MC_MAX: usize = 384;
pub const NC_MAX: usize = 2048;

/// The tile in force for this compilation, resolved once from the environment.
pub fn tl() -> Tile {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Tile> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let raw = match std::env::var("Y_GEMM_TILE") {
            Ok(v) => v,
            Err(_) => return DEFAULT_TILE,
        };
        let f: Vec<usize> = raw
            .split(',')
            .map(|x| x.trim().parse().unwrap_or(0))
            .collect();
        if f.len() != 5 {
            eprintln!(
                "[Y CPU GEMM] Y_GEMM_TILE=\"{}\" is not mr,nr,kc,mc,nc — using the default tile",
                raw
            );
            return DEFAULT_TILE;
        }
        let t = Tile { mr: f[0], nr: f[1], kc: f[2], mc: f[3], nc: f[4] };
        if let Err(e) = t.check() {
            eprintln!("[Y CPU GEMM] Y_GEMM_TILE rejected: {} — using the default tile", e);
            return DEFAULT_TILE;
        }
        eprintln!(
            "[Y CPU GEMM] tile override: mr={} nr={} kc={} mc={} nc={} ({} zmm)",
            t.mr, t.nr, t.kc, t.mc, t.nc, t.regs()
        );
        t
    })
}

/// The GEMM shape a recognised loop nest computes: `C = A * B` with the
/// operand names and the three extents, all as Y identifiers.
#[derive(Debug, Clone, PartialEq)]
pub struct GemmShape {
    pub a: String,
    pub b: String,
    pub c: String,
    pub m: String,
    pub n: String,
    pub k: String,
}

// ── Recognition ─────────────────────────────────────────────

fn ident_of(e: &Expr) -> Option<&str> {
    match e {
        Expr::Ident(n, _) => Some(n.as_str()),
        _ => None,
    }
}

fn is_zero_lit(e: &Expr) -> bool {
    matches!(e, Expr::IntLit(0, _)) || matches!(e, Expr::FloatLit(v, _) if *v == 0.0)
}

/// A `step` clause that is absent or literally 1.
fn is_unit_step(step: &Option<Expr>) -> bool {
    match step {
        None => true,
        Some(Expr::IntLit(1, _)) => true,
        _ => false,
    }
}

fn call_parts<'a>(e: &'a Expr, name: &str) -> Option<&'a [Expr]> {
    let Expr::Call { func, args, .. } = e else {
        return None;
    };
    if ident_of(func) == Some(name) {
        Some(args.as_slice())
    } else {
        None
    }
}

/// Matches `block_ptr2d_load(buf, row, col, stride, _, _)` and returns
/// `(buf, row, col, stride)`.
fn match_load(e: &Expr) -> Option<(&str, &str, &str, &str)> {
    let args = call_parts(e, "block_ptr2d_load")?;
    if args.len() != 6 {
        return None;
    }
    Some((
        ident_of(&args[0])?,
        ident_of(&args[1])?,
        ident_of(&args[2])?,
        ident_of(&args[3])?,
    ))
}

/// Recognises exactly this nest, and refuses anything else:
///
/// ```text
/// for i in 0..M { for j in 0..N {
///     let mut sum = 0.0;
///     for k in 0..K { sum = sum + load(A,i,k,K,..) * load(B,k,j,N,..); }
///     store(C, i, j, N, .., sum);
/// } }
/// ```
///
/// The match is deliberately strict. Anything that does not line up falls
/// through to the ordinary scalar lowering, which is correct — so a near-miss
/// costs performance, never an answer. A looser match that "mostly" fit would
/// substitute a different computation for the one that was written.
pub fn recognize_gemm(body: &Block) -> Option<GemmShape> {
    let [Stmt::For {
        loop_var: i,
        start: i_start,
        end: i_end,
        step: i_step,
        body: i_body,
        ..
    }] = body.stmts.as_slice()
    else {
        return None;
    };
    if !is_zero_lit(i_start) || !is_unit_step(i_step) {
        return None;
    }

    let [Stmt::For {
        loop_var: j,
        start: j_start,
        end: j_end,
        step: j_step,
        body: j_body,
        ..
    }] = i_body.stmts.as_slice()
    else {
        return None;
    };
    if !is_zero_lit(j_start) || !is_unit_step(j_step) {
        return None;
    }

    // let sum = 0.0;  /  for k ...  /  store(...)
    let [Stmt::Let {
        name: sum,
        init: Some(sum_init),
        zero_drift,
        ..
    }, Stmt::For {
        loop_var: k,
        start: k_start,
        end: k_end,
        step: k_step,
        body: k_body,
        ..
    }, Stmt::Expr(store_call)] = j_body.stmts.as_slice()
    else {
        return None;
    };
    if !is_zero_lit(sum_init) || !is_zero_lit(k_start) || !is_unit_step(k_step) {
        return None;
    }
    // A `@ZeroDrift` accumulator has different semantics — an exact
    // fixed-point accumulation — and this kernel accumulates in f32.
    // Substituting it would silently discard the guarantee.
    if zero_drift.is_some() {
        return None;
    }

    // sum = sum + load(A,i,k,K,..) * load(B,k,j,N,..)
    let [Stmt::Let {
        name: a_val,
        init: Some(a_init),
        ..
    }, Stmt::Let {
        name: b_val,
        init: Some(b_init),
        ..
    }, Stmt::Assign { target, value, .. }] = k_body.stmts.as_slice()
    else {
        return None;
    };

    let (a_buf, a_row, a_col, a_stride) = match_load(a_init)?;
    let (b_buf, b_row, b_col, b_stride) = match_load(b_init)?;

    // A is indexed [i, k] with row stride K; B is indexed [k, j] with row
    // stride N. Both must be row-major and non-transposed.
    if (a_row, a_col) != (i.as_str(), k.as_str()) {
        return None;
    }
    if (b_row, b_col) != (k.as_str(), j.as_str()) {
        return None;
    }

    if ident_of(target)? != sum.as_str() {
        return None;
    }
    let Expr::BinaryOp {
        op: BinaryOp::Add,
        left,
        right,
        ..
    } = value
    else {
        return None;
    };
    if ident_of(left)? != sum.as_str() {
        return None;
    }
    let Expr::BinaryOp {
        op: BinaryOp::Mul,
        left: ml,
        right: mr,
        ..
    } = &**right
    else {
        return None;
    };
    if ident_of(ml)? != a_val.as_str() || ident_of(mr)? != b_val.as_str() {
        return None;
    }

    // store(C, i, j, N, _, _, sum)
    let sargs = call_parts(store_call, "block_ptr2d_store")?;
    if sargs.len() != 7 {
        return None;
    }
    let c_buf = ident_of(&sargs[0])?;
    if ident_of(&sargs[1])? != i.as_str()
        || ident_of(&sargs[2])? != j.as_str()
        || ident_of(&sargs[6])? != sum.as_str()
    {
        return None;
    }
    let c_stride = ident_of(&sargs[3])?;

    // The strides must be the extents they are supposed to be, or the buffers
    // are not the matrices this kernel assumes.
    let m = ident_of(i_end)?;
    let n = ident_of(j_end)?;
    let k_ext = ident_of(k_end)?;
    if a_stride != k_ext || b_stride != n || c_stride != n {
        return None;
    }

    Some(GemmShape {
        a: a_buf.to_string(),
        b: b_buf.to_string(),
        c: c_buf.to_string(),
        m: m.to_string(),
        n: n.to_string(),
        k: k_ext.to_string(),
    })
}

// ── IR emission ─────────────────────────────────────────────

/// Small helper for emitting structured loops as LLVM IR text.
///
/// Induction variables live in `alloca`s rather than `phi` nodes. That is not
/// laziness: `mem2reg` promotes them in the first function pass, and it keeps
/// this generator free of the SSA bookkeeping that hand-written phi chains
/// need. The vector width, the FMA structure and the blocking are explicit —
/// those are what the performance comes from — while register allocation is
/// left to the backend, which is better at it.
pub struct IrBuilder {
    pub out: String,
    n: usize,
    entry: Vec<String>,
}

pub struct LoopCtx {
    var: String,
    step: String,
    cond: String,
    body: String,
    end: String,
}

impl IrBuilder {
    pub fn new() -> Self {
        IrBuilder {
            out: String::new(),
            n: 0,
            entry: Vec::new(),
        }
    }

    fn t(&mut self) -> String {
        self.n += 1;
        format!("%g{}", self.n)
    }

    fn l(&mut self, tag: &str) -> String {
        self.n += 1;
        format!("{}.{}", tag, self.n)
    }

    fn w(&mut self, s: &str) {
        self.out.push_str("  ");
        self.out.push_str(s);
        self.out.push('\n');
    }

    /// Declare an alloca that must sit in the entry block.
    fn entry_alloca(&mut self, name: &str, ty: &str, align: usize) {
        self.entry
            .push(format!("  {} = alloca {}, align {}", name, ty, align));
    }

    /// `for (var = start; var < end; var += step)`, signed.
    fn loop_begin(&mut self, tag: &str, start: &str, end: &str, step: &str) -> LoopCtx {
        self.n += 1;
        let var = format!("%iv{}", self.n);
        let cond = self.l(&format!("{}.cond", tag));
        let body = self.l(&format!("{}.body", tag));
        let end_l = self.l(&format!("{}.end", tag));
        self.entry_alloca(&var, "i64", 8);
        self.w(&format!("store i64 {}, ptr {}", start, var));
        self.w(&format!("br label %{}", cond));
        self.out.push_str(&format!("{}:\n", cond));
        let cur = self.t();
        self.w(&format!("{} = load i64, ptr {}", cur, var));
        let c = self.t();
        self.w(&format!("{} = icmp slt i64 {}, {}", c, cur, end));
        self.w(&format!("br i1 {}, label %{}, label %{}", c, body, end_l));
        self.out.push_str(&format!("{}:\n", body));
        LoopCtx {
            var,
            step: step.to_string(),
            cond,
            body,
            end: end_l,
        }
    }

    /// Current value of a loop's induction variable.
    fn iv(&mut self, ctx: &LoopCtx) -> String {
        let t = self.t();
        self.w(&format!("{} = load i64, ptr {}", t, ctx.var));
        t
    }

    fn loop_end(&mut self, ctx: LoopCtx) {
        let cur = self.t();
        self.w(&format!("{} = load i64, ptr {}", cur, ctx.var));
        let nx = self.t();
        self.w(&format!("{} = add i64 {}, {}", nx, cur, ctx.step));
        self.w(&format!("store i64 {}, ptr {}", nx, ctx.var));
        self.w(&format!("br label %{}", ctx.cond));
        self.out.push_str(&format!("{}:\n", ctx.end));
        let _ = ctx.body;
    }

    /// `max(a, b)` on i64.
    fn imax(&mut self, a: &str, b: &str) -> String {
        let c = self.t();
        self.w(&format!("{} = icmp sgt i64 {}, {}", c, a, b));
        let r = self.t();
        self.w(&format!("{} = select i1 {}, i64 {}, i64 {}", r, c, a, b));
        r
    }

    /// `min(a, b)` on i64.
    fn imin(&mut self, a: &str, b: &str) -> String {
        let c = self.t();
        self.w(&format!("{} = icmp slt i64 {}, {}", c, a, b));
        let r = self.t();
        self.w(&format!("{} = select i1 {}, i64 {}, i64 {}", r, c, a, b));
        r
    }

    fn gep(&mut self, ty: &str, base: &str, off: &str) -> String {
        let r = self.t();
        self.w(&format!(
            "{} = getelementptr inbounds {}, ptr {}, i64 {}",
            r, ty, base, off
        ));
        r
    }

    fn mul(&mut self, a: &str, b: &str) -> String {
        let r = self.t();
        self.w(&format!("{} = mul nsw i64 {}, {}", r, a, b));
        r
    }

    fn add(&mut self, a: &str, b: &str) -> String {
        let r = self.t();
        self.w(&format!("{} = add nsw i64 {}, {}", r, a, b));
        r
    }

    fn sub(&mut self, a: &str, b: &str) -> String {
        let r = self.t();
        self.w(&format!("{} = sub nsw i64 {}, {}", r, a, b));
        r
    }

    /// Finish a function: splice the entry-block allocas in ahead of the body.
    ///
    /// Every function gets `#0`, the target-cpu/target-features attribute set.
    /// Without it a helper that does not get inlined into the `#0` kernel is
    /// compiled for baseline x86-64, and the `<16 x float>` masked loads and
    /// stores scalarise into per-lane bit-test chains on 4-wide `xmm`. That is
    /// a ~7x slowdown that shows up only after an unrelated change stops the
    /// inliner from firing, which is how it was found.
    fn finish(&mut self, signature: &str) -> String {
        let mut s = String::new();
        let _ = writeln!(&mut s, "{} #0 {{", signature);
        let _ = writeln!(&mut s, "entry:");
        for a in &self.entry {
            let _ = writeln!(&mut s, "{}", a);
        }
        s.push_str(&self.out);
        let _ = writeln!(&mut s, "  ret void");
        let _ = writeln!(&mut s, "}}");
        s
    }
}

/// Name of the emitted kernel.
pub const KERNEL_NAME: &str = "__y_sgemm_f32_avx512";

/// Upper bound on worker threads. Also the size of the stack-allocated task
/// and `pthread_t` arrays, so it is a hard cap, not a hint.
///
/// Capped at physical-core count rather than `_SC_NPROCESSORS_ONLN`: a GEMM
/// micro-kernel saturates a core's FMA pipes on its own, so the second SMT
/// thread on a core adds scheduling pressure and scratch footprint for no
/// extra throughput.
pub const MAX_THREADS: usize = 16;

/// Floats of scratch each thread needs: one A panel plus one B panel.
///
/// Sized for the LARGEST tile `Tile::check` will accept, not for the tile in
/// force, because it sizes a static array and the tile is a runtime value. The
/// buffer is BSS, so the pages a smaller tile never touches are never faulted
/// in and cost nothing resident.
pub const SCRATCH_FLOATS: usize =
    (MC_MAX + MR_MAX) * KC_MAX + KC_MAX * (NC_MAX + NR_MAX);

/// Multiply-adds a thread must be given before it is worth adding.
///
/// OpenBLAS's rule is `nthreads = M*N*K / (SMP_THRESHOLD_MIN *
/// GEMM_MULTITHREAD_THRESHOLD)` with that product equal to `65536 * 4`
/// (`interface/gemm.c`). Using their constant here asks for 16 threads on a
/// 256^3 GEMM, which measures 391 GFLOPS against 465 on four — their figure is
/// calibrated for a per-thread kernel that does roughly half the work per
/// cycle that this one does, and for a driver that shares one packed B panel
/// between threads. This one re-packs B per thread, so each extra thread adds
/// real work rather than only splitting it.
///
/// Measured optima on a clean machine (Y GFLOPS by thread count):
///
/// | shape | 2 | 4 | 8 | 16 |
/// |---|---|---|---|---|
/// | 256^3 | 326 | **465** | 387 | 391 |
/// | 250^3 | 325 | **480** | 449 | 384 |
/// | 512^3 | 372 | 487 | **576** | 483 |
/// | 64x64x32768 | 168 | **257** | 160 | 157 |
/// | 17x4096x4096 | 111 | 168 | 234 | **318** |
pub const WORK_PER_THREAD: usize = 4 << 20;

/// Work above which threads share one packed B panel instead of each packing
/// their own.
///
/// Sharing is **not** a general win, which is the opposite of what the traffic
/// arithmetic predicts. Packing B once instead of `nthreads` times obviously
/// moves fewer bytes, but the panel is then produced by one thread and consumed
/// by all, so it travels through L3 rather than staying in the packer's own L2
/// — and the two barriers per block expose whatever load imbalance the M-split
/// left. Measured over four full runs each (mean Y/OB over 18 shapes):
///
/// | | private B | shared B |
/// |---|---|---|
/// | mean | **0.98** | 0.89 |
/// | 1024³ | 1230 GF | 960 GF |
/// | 1000³ | 1298 GF | 903 GF |
/// | 2048³ | 1770 GF | **2050 GF** |
///
/// So it pays only once the operands stop fitting comfortably in a 32 MB L3,
/// where the redundant re-reads become real DRAM traffic. 2048³ moves 16.8 MB
/// per operand; 1024³ moves 4 MB and simply stays cached however many times it
/// is re-read.
pub const SHARE_B_WORK: usize = 2 << 30;

const SCRATCH: &str = "@__y_gemm_scratch";
const SCRATCH_CAP: &str = "@__y_gemm_scratch_threads";
const NTHREADS_CACHE: &str = "@__y_gemm_nthreads";
/// Statically reserved single-thread panel, used only if `malloc` fails.
const FALLBACK: &str = "@__y_gemm_fallback";
const POOL_N: &str = "@__y_pool_n";
const POOL_GEN: &str = "@__y_pool_gen";
const POOL_DONE: &str = "@__y_pool_done";
const POOL_TASK: &str = "@__y_pool_task";
const POOL_IDS: &str = "@__y_pool_ids";

const POOL_MUTEX: &str = "@__y_pool_mutex";
const POOL_COND: &str = "@__y_pool_cond";
const POOL_BARRIER: &str = "@__y_pool_barrier";
const BARRIER_N: &str = "@__y_barrier_n";
/// One B panel shared by all threads when the split is along M.
const SHARED_B: &str = "@__y_shared_b";

/// Pause iterations a worker spins before blocking on the condition variable.
///
/// Short on purpose. The spin only exists to cover the gap between
/// back-to-back dispatches; past that a worker must genuinely block. An
/// earlier version spun 8192 times and then polled `usleep(100)` forever,
/// which left 15 threads waking 10k times a second and spinning hot into
/// whatever ran next — it corrupted the very A/B benchmark used to tune it,
/// reading OpenBLAS at 15 GFLOPS on a shape where it does 460.
const POOL_SPIN: usize = 512;

/// Module-level declarations the kernel needs.
///
/// Packing scratch is one heap block of `SCRATCH_FLOATS` per thread, allocated
/// once and reused. It used to be two fixed globals, which made the kernel
/// single-threaded by construction — threads sharing a packed panel corrupt
/// each other's data rather than merely contending for it.
pub fn emit_globals() -> String {
    let mut s = String::new();
    let _ = writeln!(&mut s, "; --- Y CPU GEMM: shared state ---");
    let _ = writeln!(
        &mut s,
        "%y_gemm_task = type {{ ptr, ptr, ptr, i64, i64, i64, i64, i64, i64, i64, ptr, i64, i64, i64 }}"
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global ptr null, align 8   ; per-thread packing scratch",
        SCRATCH
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global i64 0, align 8      ; threads the scratch was sized for",
        SCRATCH_CAP
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global i64 0, align 8      ; 0 = not yet resolved",
        NTHREADS_CACHE
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global [{} x float] zeroinitializer, align 64  ; malloc-failure panel",
        FALLBACK, SCRATCH_FLOATS
    );
    let _ = writeln!(
        &mut s,
        "@.y_gemm_env = private unnamed_addr constant [15 x i8] c\"Y_NUM_THREADS\\00\\00\""
    );
    let _ = writeln!(
        &mut s,
        "declare <16 x float> @llvm.fmuladd.v16f32(<16 x float>, <16 x float>, <16 x float>)"
    );
    let _ = writeln!(
        &mut s,
        "declare void @llvm.masked.store.v16f32.p0(<16 x float>, ptr, i32 immarg, <16 x i1>)"
    );
    let _ = writeln!(
        &mut s,
        "declare <16 x float> @llvm.masked.load.v16f32.p0(ptr, i32 immarg, <16 x i1>, <16 x float>)"
    );
    // `malloc`/`free` are already declared by the emitter prelude; redeclaring
    // them is an invalid redefinition, not a duplicate that LLVM merges.
    let _ = writeln!(&mut s, "declare ptr @getenv(ptr)");
    let _ = writeln!(&mut s, "declare i32 @atoi(ptr)");
    let _ = writeln!(&mut s, "declare i64 @sysconf(i32)");
    let _ = writeln!(
        &mut s,
        "declare i32 @pthread_create(ptr, ptr, ptr, ptr)"
    );
    let _ = writeln!(&mut s, "declare i32 @pthread_join(i64, ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_mutex_init(ptr, ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_mutex_lock(ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_mutex_unlock(ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_cond_init(ptr, ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_cond_wait(ptr, ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_cond_broadcast(ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_barrier_init(ptr, ptr, i32)");
    let _ = writeln!(&mut s, "declare i32 @pthread_barrier_destroy(ptr)");
    let _ = writeln!(&mut s, "declare i32 @pthread_barrier_wait(ptr)");
    let _ = writeln!(
        &mut s,
        "{} = internal global [64 x i8] zeroinitializer, align 16",
        POOL_BARRIER
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global i64 0, align 8   ; barrier width, 0 = uninitialised",
        BARRIER_N
    );
    // Shared B panel. When the split is along M every thread walks the same
    // (jc, pc) blocks and therefore wants the same packed B; packing it once
    // instead of once per thread is the difference between adding a thread
    // splitting the work and adding a thread that also re-reads all of B.
    let _ = writeln!(
        &mut s,
        "{} = internal global [{} x float] zeroinitializer, align 64",
        SHARED_B,
        tl().kc * (tl().nc + tl().nr)
    );
    // Over-sized and initialised by call rather than by relying on the
    // all-zero glibc static initialisers, so the layout is not assumed.
    let _ = writeln!(
        &mut s,
        "{} = internal global [64 x i8] zeroinitializer, align 16",
        POOL_MUTEX
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global [64 x i8] zeroinitializer, align 16",
        POOL_COND
    );
    let _ = writeln!(&mut s, "declare void @llvm.x86.sse2.pause()");
    let _ = writeln!(
        &mut s,
        "{} = internal global i64 0, align 8   ; workers running (0 = pool not started)",
        POOL_N
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global i64 0, align 8   ; bumped once per dispatch",
        POOL_GEN
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global [{} x i64] zeroinitializer, align 64  ; generation each worker finished",
        POOL_DONE, MAX_THREADS
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global [{} x %y_gemm_task] zeroinitializer, align 64",
        POOL_TASK, MAX_THREADS
    );
    let _ = writeln!(
        &mut s,
        "{} = internal global [{} x i64] zeroinitializer, align 8    ; each worker's index",
        POOL_IDS, MAX_THREADS
    );
    s
}

/// `<16 x i1>` mask selecting the first `rem` lanes (saturating at 0 and 16).
fn lane_mask(b: &mut IrBuilder, rem: &str) -> String {
    let t = b.t();
    b.w(&format!(
        "{} = insertelement <16 x i64> poison, i64 {}, i32 0",
        t, rem
    ));
    let sp = b.t();
    b.w(&format!(
        "{} = shufflevector <16 x i64> {}, <16 x i64> poison, <16 x i32> zeroinitializer",
        sp, t
    ));
    let m = b.t();
    b.w(&format!(
        "{} = icmp slt <16 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, \
         i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15>, {}",
        m, sp
    ));
    m
}

/// Splat a scalar float across `<16 x float>`.
fn splat(b: &mut IrBuilder, v: &str) -> String {
    let t = b.t();
    b.w(&format!(
        "{} = insertelement <16 x float> poison, float {}, i32 0",
        t, v
    ));
    let s = b.t();
    b.w(&format!(
        "{} = shufflevector <16 x float> {}, <16 x float> poison, <16 x i32> zeroinitializer",
        s, t
    ));
    s
}

/// `packB(src, ldb, kc, nc)` — copy a `kc x nc` block of B into
/// `[panel][k][NR]` order, zero-filling the N tail so the micro-kernel never
/// needs an N-tail branch.
fn emit_pack_b() -> String {
    let mut b = IrBuilder::new();
    let (src, ldb, kc, nc, dst, tid, nthr) =
        ("%src", "%ldb", "%kc", "%nc", "%dst", "%tid", "%nthr");

    // Panel `p` is packed by thread `p % nthr`. With nthr = 1 this is the
    // ordinary sequential pack.
    let start = b.mul(tid, &tl().nr.to_string());
    let stride = b.mul(nthr, &tl().nr.to_string());
    let jl = b.loop_begin("pb.j", &start, nc, &stride);
    let j0 = b.iv(&jl);
    let rem_n = b.sub(nc, &j0);
    let jw = b.imin(&rem_n, &tl().nr.to_string());

    // dst = packB + j0 * kc
    let dbase = b.mul(&j0, kc);
    let dst0 = b.gep("float", dst, &dbase);

    let pl = b.loop_begin("pb.p", "0", kc, "1");
    let p = b.iv(&pl);
    let srow = b.mul(&p, ldb);
    let soff = b.add(&srow, &j0);
    let sp = b.gep("float", src, &soff);
    let drow = b.mul(&p, &tl().nr.to_string());
    let dp = b.gep("float", &dst0, &drow);

    // Copy jw lanes, zero the rest. NR is 32 floats = two vectors.
    for v in 0..tl().nrv() {
        let lo = (v * 16).to_string();
        let rem = b.sub(&jw, &lo);
        let mask = lane_mask(&mut b, &rem);
        let s = b.gep("float", &sp, &lo);
        let d = b.gep("float", &dp, &lo);
        let val = b.t();
        b.w(&format!(
            "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, <16 x i1> {}, \
             <16 x float> zeroinitializer)",
            val, s, mask
        ));
        // Store all 16 lanes: the masked-off ones are zero, which is exactly
        // the padding the micro-kernel needs.
        b.w(&format!("store <16 x float> {}, ptr {}, align 4", val, d));
    }

    b.loop_end(pl);
    b.loop_end(jl);
    b.finish(&format!(
        "define internal void @__y_gemm_pack_b(ptr noalias {}, i64 {}, i64 {}, i64 {}, \
         ptr noalias {}, i64 {}, i64 {})",
        src, ldb, kc, nc, dst, tid, nthr
    ))
}

/// `packA(src, lda, mc, kc)` — copy an `mc x kc` block of A into
/// `[panel][k][MR]` order. This is a transpose: A is row-major, so the
/// micro-kernel would otherwise walk it by column, at one cache miss per
/// element.
fn emit_pack_a() -> String {
    let mut b = IrBuilder::new();
    let (src, lda, mc, kc, dst) = ("%src", "%lda", "%mc", "%kc", "%dst");

    let il = b.loop_begin("pa.i", "0", mc, &tl().mr.to_string());
    let i0 = b.iv(&il);
    let rem_m = b.sub(mc, &i0);
    let iw = b.imin(&rem_m, &tl().mr.to_string());

    let dbase = b.mul(&i0, kc);
    let dst0 = b.gep("float", dst, &dbase);

    let pl = b.loop_begin("pa.p", "0", kc, "1");
    let p = b.iv(&pl);
    let drow = b.mul(&p, &tl().mr.to_string());
    let dp = b.gep("float", &dst0, &drow);

    for i in 0..tl().mr {
        let ic = i.to_string();
        let inb = b.t();
        b.w(&format!("{} = icmp slt i64 {}, {}", inb, ic, iw));
        // Row i0+i of A, column p. Out-of-range rows read element 0 and are
        // then forced to zero, so the padded rows contribute nothing.
        let row = b.add(&i0, &ic);
        let ro = b.mul(&row, lda);
        let so = b.add(&ro, &p);
        let sso = b.t();
        b.w(&format!(
            "{} = select i1 {}, i64 {}, i64 0",
            sso, inb, so
        ));
        let sp = b.gep("float", src, &sso);
        let raw = b.t();
        b.w(&format!("{} = load float, ptr {}, align 4", raw, sp));
        let val = b.t();
        b.w(&format!(
            "{} = select i1 {}, float {}, float 0.0",
            val, inb, raw
        ));
        let d = b.gep("float", &dp, &ic);
        b.w(&format!("store float {}, ptr {}, align 4", val, d));
    }

    b.loop_end(pl);
    b.loop_end(il);
    b.finish(&format!(
        "define internal void @__y_gemm_pack_a(ptr noalias {}, i64 {}, i64 {}, i64 {}, \
         ptr noalias {})",
        src, lda, mc, kc, dst
    ))
}

/// The micro-kernel: `C[mw x jw] (+)= Ap[kc x MR] * B[kc x jw]`, with all
/// `MR * NRV` accumulators live in registers across the whole K loop.
///
/// `ldb` is a parameter rather than a constant so one kernel serves both
/// sources of B: the packed panel (`ldb = NR`, zero-padded, so the loads are
/// always full width) and B in place (`ldb = N`, unpadded, so the final column
/// panel must be masked). Packing B only pays when it is reused across several
/// row panels; when M is small there is one panel and packing is pure loss.
fn emit_micro() -> String {
    let mut b = IrBuilder::new();
    let (ap, bp, ldb, c, ldc, kc, mw, jw, first) = (
        "%ap", "%bp", "%ldb", "%c", "%ldc", "%kc", "%mw", "%jw", "%first",
    );

    // Accumulators. mem2reg promotes these to registers; MR*NRV = 24 vectors
    // plus NRV operands and one broadcast fits the 32 zmm registers.
    for i in 0..tl().mr {
        for v in 0..tl().nrv() {
            let name = format!("%acc{}_{}", i, v);
            b.entry_alloca(&name, "<16 x float>", 64);
            b.w(&format!(
                "store <16 x float> zeroinitializer, ptr {}, align 64",
                name
            ));
        }
    }

    // Two copies of the K loop: full-width loads when the panel is complete,
    // masked loads only for the ragged final panel. Emitting one masked loop
    // would put a mask computation in the innermost loop of every shape to
    // serve the tail of one panel.
    let full = b.t();
    b.w(&format!("{} = icmp eq i64 {}, {}", full, jw, tl().nr));
    let full_l = b.l("mk.full");
    let tail_l = b.l("mk.tail");
    let epi_l = b.l("mk.epi");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        full, full_l, tail_l
    ));

    for (masked, entry) in [(false, full_l), (true, tail_l)] {
        b.out.push_str(&format!("{}:\n", entry));
        let pl = b.loop_begin("mk.p", "0", kc, "1");
        let p = b.iv(&pl);

        // B row: jw contiguous floats at stride ldb.
        let bo = b.mul(&p, ldb);
        let brow = b.gep("float", bp, &bo);
        let mut bv = Vec::new();
        for v in 0..tl().nrv() {
            let off = (v * 16).to_string();
            let q = b.gep("float", &brow, &off);
            let t = b.t();
            if masked {
                let rem = b.sub(jw, &off);
                let mask = lane_mask(&mut b, &rem);
                b.w(&format!(
                    "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, \
                     <16 x i1> {}, <16 x float> zeroinitializer)",
                    t, q, mask
                ));
            } else {
                b.w(&format!("{} = load <16 x float>, ptr {}, align 4", t, q));
            }
            bv.push(t);
        }

        // A panel row: MR contiguous floats, each broadcast.
        let ao = b.mul(&p, &tl().mr.to_string());
        let arow = b.gep("float", ap, &ao);
        for i in 0..tl().mr {
            let off = i.to_string();
            let q = b.gep("float", &arow, &off);
            let s = b.t();
            b.w(&format!("{} = load float, ptr {}, align 4", s, q));
            let a = splat(&mut b, &s);
            for v in 0..tl().nrv() {
                let name = format!("%acc{}_{}", i, v);
                let cur = b.t();
                b.w(&format!(
                    "{} = load <16 x float>, ptr {}, align 64",
                    cur, name
                ));
                let nv = b.t();
                b.w(&format!(
                    "{} = call <16 x float> @llvm.fmuladd.v16f32(<16 x float> {}, \
                     <16 x float> {}, <16 x float> {})",
                    nv, a, bv[v], cur
                ));
                b.w(&format!(
                    "store <16 x float> {}, ptr {}, align 64",
                    nv, name
                ));
            }
        }
        b.loop_end(pl);
        b.w(&format!("br label %{}", epi_l));
    }
    b.out.push_str(&format!("{}:\n", epi_l));

    // Write out the mw live rows, masked to jw columns.
    //
    // The two cases are emitted as separate blocks under one branch rather than
    // as a `select` over both values. A select needs the accumulate operand,
    // which means reading C back on *every* K-block including the first — and
    // when K fits in one block that read is pure waste proportional to the size
    // of C. At 4096x4096x8 it was a 64 MB read supporting 268 MFLOP.
    let acc_l = b.l("mk.acc");
    let st_l = b.l("mk.first");
    let done_l = b.l("mk.done");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        first, st_l, acc_l
    ));

    for (accumulate, entry) in [(false, st_l), (true, acc_l)] {
        b.out.push_str(&format!("{}:\n", entry));
        for i in 0..tl().mr {
            let ic = i.to_string();
            let live = b.t();
            b.w(&format!("{} = icmp slt i64 {}, {}", live, ic, mw));
            let do_l = b.l("mk.st");
            let skip_l = b.l("mk.sk");
            b.w(&format!(
                "br i1 {}, label %{}, label %{}",
                live, do_l, skip_l
            ));
            b.out.push_str(&format!("{}:\n", do_l));

            let ro = b.mul(&ic, ldc);
            let crow = b.gep("float", c, &ro);
            for v in 0..tl().nrv() {
                let lo = (v * 16).to_string();
                let rem = b.sub(jw, &lo);
                let mask = lane_mask(&mut b, &rem);
                let q = b.gep("float", &crow, &lo);
                let name = format!("%acc{}_{}", i, v);
                let acc = b.t();
                b.w(&format!(
                    "{} = load <16 x float>, ptr {}, align 64",
                    acc, name
                ));
                let val = if accumulate {
                    let old = b.t();
                    b.w(&format!(
                        "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, \
                         <16 x i1> {}, <16 x float> zeroinitializer)",
                        old, q, mask
                    ));
                    let sum = b.t();
                    b.w(&format!("{} = fadd <16 x float> {}, {}", sum, acc, old));
                    sum
                } else {
                    acc
                };
                b.w(&format!(
                    "call void @llvm.masked.store.v16f32.p0(<16 x float> {}, ptr {}, i32 4, \
                     <16 x i1> {})",
                    val, q, mask
                ));
            }
            b.w(&format!("br label %{}", skip_l));
            b.out.push_str(&format!("{}:\n", skip_l));
        }
        b.w(&format!("br label %{}", done_l));
    }
    b.out.push_str(&format!("{}:\n", done_l));

    b.finish(&format!(
        "define internal void @__y_gemm_micro(ptr noalias {}, ptr noalias {}, i64 {}, ptr {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, i1 {})",
        ap, bp, ldb, c, ldc, kc, mw, jw, first
    ))
}

/// Rows of C the small-M kernel keeps broadcast in registers.
pub const SM_MR: usize = 8;
/// Above this many rows the packed path wins and this one is not used.
///
/// Measured by sweeping the threshold over the 4096-wide skinny family:
/// against OpenBLAS this path is 4.05x at M=1, 0.88x at M=4 and 0.74x at M=8,
/// but at M=17 it measures 0.467 where the packed path measures 0.663 — B is
/// re-read once per 8-row block, and by M=17 that costs more than the packing
/// it avoids. Thresholds of 8, 16 and 24 measure 1.622 / 1.638 / 1.599 mean
/// ratio over the whole shape set, so this is the conservative end of a flat
/// optimum, not a knife edge.
pub const SM_MAX_M: usize = 8;

/// Small-M kernel: `k` outer, `j` inner.
///
/// The packed path is wrong for a near-GEMM shape for two compounding reasons:
/// `MR = 12` rows are computed for however few exist, and B gets packed with
/// almost no reuse. Inverting the loops fixes both. B is then read exactly
/// once in perfect linear order — it is the large operand and the shape is
/// bandwidth-bound — while C, only `M * N` with M small, stays resident in
/// cache across the whole K loop and absorbs the repeated accumulation.
fn emit_small_m() -> String {
    let mut b = IrBuilder::new();
    let (a, bb, c, _m, n, k, m0, m1, n0, n1) = (
        "%A", "%B", "%C", "%M", "%N", "%K", "%m0", "%m1", "%n0", "%n1",
    );

    // C is accumulated into across all of K, so this thread's slice of it
    // starts at zero. Only the owned columns are cleared: another thread may
    // be writing the rest of the same rows concurrently.
    let zw = b.sub(n1, n0);
    let zb = b.t();
    b.w(&format!("{} = shl i64 {}, 2", zb, zw));
    let zl = b.loop_begin("sm.z", m0, m1, "1");
    let zi = b.iv(&zl);
    let zro = b.mul(&zi, n);
    let zoff = b.add(&zro, n0);
    let zp = b.gep("float", c, &zoff);
    b.w(&format!(
        "call void @llvm.memset.p0.i64(ptr {}, i8 0, i64 {}, i1 false)",
        zp, zb
    ));
    b.loop_end(zl);

    let il = b.loop_begin("sm.i", m0, m1, &SM_MR.to_string());
    let i0 = b.iv(&il);
    let rm = b.sub(m1, &i0);
    let mw = b.imin(&rm, &SM_MR.to_string());

    let pl = b.loop_begin("sm.p", "0", k, "1");
    let p = b.iv(&pl);

    // Broadcast this K-step's A column once, outside the j loop.
    let mut av = Vec::new();
    for i in 0..SM_MR {
        let ic = i.to_string();
        let live = b.t();
        b.w(&format!("{} = icmp slt i64 {}, {}", live, ic, mw));
        let row = b.add(&i0, &ic);
        let ro = b.mul(&row, k);
        let off = b.add(&ro, &p);
        // Rows past mw read element 0 and are then zeroed, so they contribute
        // nothing and the load stays in bounds.
        let soff = b.t();
        b.w(&format!("{} = select i1 {}, i64 {}, i64 0", soff, live, off));
        let q = b.gep("float", a, &soff);
        let raw = b.t();
        b.w(&format!("{} = load float, ptr {}, align 4", raw, q));
        let val = b.t();
        b.w(&format!(
            "{} = select i1 {}, float {}, float 0.0",
            val, live, raw
        ));
        av.push(splat(&mut b, &val));
    }

    let brow_off = b.mul(&p, n);
    let brow = b.gep("float", bb, &brow_off);
    let crow_base = b.mul(&i0, n);

    // Full 16-wide columns, then a masked tail. Keeping the mask out of the
    // main j loop matters: it runs M*K*N/16 times.
    let span = b.sub(n1, n0);
    let span_full = b.t();
    b.w(&format!("{} = and i64 {}, -16", span_full, span));
    let n_full = b.add(n0, &span_full);

    for (masked, tag) in [(false, "sm.j"), (true, "sm.t")] {
        let (start, end, step) = if masked {
            (n_full.clone(), n1.to_string(), "16".to_string())
        } else {
            (n0.to_string(), n_full.clone(), "16".to_string())
        };
        let jl = b.loop_begin(tag, &start, &end, &step);
        let j = b.iv(&jl);

        let bq = b.gep("float", &brow, &j);
        let bvv = b.t();
        let mask = if masked {
            let rem = b.sub(n1, &j);
            let mk = lane_mask(&mut b, &rem);
            b.w(&format!(
                "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, \
                 <16 x i1> {}, <16 x float> zeroinitializer)",
                bvv, bq, mk
            ));
            Some(mk)
        } else {
            b.w(&format!("{} = load <16 x float>, ptr {}, align 4", bvv, bq));
            None
        };

        for i in 0..SM_MR {
            let ic = i.to_string();
            let live = b.t();
            b.w(&format!("{} = icmp slt i64 {}, {}", live, ic, mw));
            let do_l = b.l("sm.do");
            let sk_l = b.l("sm.sk");
            b.w(&format!("br i1 {}, label %{}, label %{}", live, do_l, sk_l));
            b.out.push_str(&format!("{}:\n", do_l));

            let ro = b.mul(&ic, n);
            let co = b.add(&crow_base, &ro);
            let co2 = b.add(&co, &j);
            let cq = b.gep("float", c, &co2);
            let cur = b.t();
            let nv = b.t();
            match &mask {
                Some(mk) => {
                    b.w(&format!(
                        "{} = call <16 x float> @llvm.masked.load.v16f32.p0(ptr {}, i32 4, \
                         <16 x i1> {}, <16 x float> zeroinitializer)",
                        cur, cq, mk
                    ));
                    b.w(&format!(
                        "{} = call <16 x float> @llvm.fmuladd.v16f32(<16 x float> {}, \
                         <16 x float> {}, <16 x float> {})",
                        nv, av[i], bvv, cur
                    ));
                    b.w(&format!(
                        "call void @llvm.masked.store.v16f32.p0(<16 x float> {}, ptr {}, \
                         i32 4, <16 x i1> {})",
                        nv, cq, mk
                    ));
                }
                None => {
                    b.w(&format!(
                        "{} = load <16 x float>, ptr {}, align 4",
                        cur, cq
                    ));
                    b.w(&format!(
                        "{} = call <16 x float> @llvm.fmuladd.v16f32(<16 x float> {}, \
                         <16 x float> {}, <16 x float> {})",
                        nv, av[i], bvv, cur
                    ));
                    b.w(&format!("store <16 x float> {}, ptr {}, align 4", nv, cq));
                }
            }
            b.w(&format!("br label %{}", sk_l));
            b.out.push_str(&format!("{}:\n", sk_l));
        }
        b.loop_end(jl);
    }

    b.loop_end(pl);
    b.loop_end(il);

    b.finish(&format!(
        "define internal void @__y_gemm_small_m(ptr noalias {}, ptr noalias {}, \
         ptr {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {})",
        a, bb, c, _m, n, k, m0, m1, n0, n1
    ))
}

/// The blocked driver: the five-deep BLIS loop nest around the micro-kernel.
fn emit_driver() -> String {
    let mut b = IrBuilder::new();
    let (a, bb, c, m, n, k, m0, m1, n0, n1, scratch, tid, nthr, shared) = (
        "%A", "%B", "%C", "%M", "%N", "%K", "%m0", "%m1", "%n0", "%n1", "%scratch",
        "%tid", "%nthr", "%shared",
    );

    // Nothing to do, and — importantly — no barrier to enter. A pool slot past
    // the active thread count must return before the cooperative section, or
    // it would be counted in a barrier it was never sized into.
    let empty_m = b.t();
    b.w(&format!("{} = icmp sge i64 {}, {}", empty_m, m0, m1));
    let empty_n = b.t();
    b.w(&format!("{} = icmp sge i64 {}, {}", empty_n, n0, n1));
    let empty = b.t();
    b.w(&format!("{} = or i1 {}, {}", empty, empty_m, empty_n));
    let done_l = b.l("g.empty");
    let go_l = b.l("g.go");
    b.w(&format!("br i1 {}, label %{}, label %{}", empty, done_l, go_l));
    b.out.push_str(&format!("{}:\n", done_l));
    b.w("ret void");
    b.out.push_str(&format!("{}:\n", go_l));

    // This thread's A panel. B is either shared with the other threads (when
    // the split is along M, so every thread wants the same packed block) or
    // private (when the split is along N, where the blocks are disjoint and
    // there is nothing to share).
    let packa = scratch;
    // The B panel starts past the LARGEST A panel the scratch is sized for,
    // not past the current tile's. `kc` is chosen at runtime below and can
    // exceed `tl().kc`, so an offset computed from the tile would put the A
    // panel's tail on top of the B panel — a silent wrong answer on exactly
    // the shapes the runtime `kc` exists to speed up. `SCRATCH_FLOATS` already
    // reserves the maximum, so this only makes the two agree.
    let privb = b.gep("float", scratch, &((MC_MAX + MR_MAX) * KC_MAX).to_string());
    let coop = b.t();
    b.w(&format!("{} = icmp ne i64 {}, 0", coop, shared));
    let packb = b.t();
    b.w(&format!(
        "{} = select i1 {}, ptr {}, ptr {}",
        packb, coop, SHARED_B, privb
    ));

    // Route a near-GEMV shape to the k-outer kernel; see `SM_MAX_M`.
    let small = b.t();
    b.w(&format!("{} = icmp sle i64 {}, {}", small, m, SM_MAX_M));
    let sm_l = b.l("g.small");
    let big_l = b.l("g.big");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        small, sm_l, big_l
    ));
    b.out.push_str(&format!("{}:\n", sm_l));
    b.w(&format!(
        "call void @__y_gemm_small_m(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 {}, i64 {}, i64 {}, i64 {})",
        a, bb, c, m, n, k, m0, m1, n0, n1
    ));
    b.w("ret void");
    b.out.push_str(&format!("{}:\n", big_l));

    let jcl = b.loop_begin("g.jc", n0, n1, &tl().nc.to_string());
    let jc = b.iv(&jcl);
    let rn = b.sub(n1, &jc);
    let nc = b.imin(&rn, &tl().nc.to_string());

    // K-panel depth, chosen at RUNTIME from the thread count.
    //
    // This is the one blocking parameter whose optimum is not a property of
    // the machine alone, and a compile-time constant therefore cannot be right
    // for both cases:
    //
    //   - One thread wants `kc` LARGE. C is read and written once per K-panel,
    //     so C traffic scales as `K/kc`. Raising kc 256 -> 1024 measured
    //     2048^3 at 207 -> 298 GFLOPS single-threaded, +44%.
    //   - Sixteen threads want `kc` SMALL. Each thread packs a private
    //     `kc x nc` B panel, so the aggregate footprint is `nthr*kc*nc*4`
    //     bytes. At kc=1024, nc=2048 that is 8.4 MB per thread and 134 MB
    //     across 16 — against a 64 MB L3. Measured with the single-threaded
    //     optimum baked in, 1024^3 collapsed from 2983 to 851 GFLOPS (0.34x
    //     OpenBLAS). Tuning this on one thread and shipping it is a real trap.
    //
    // So budget the aggregate panel at `L3_PANEL_FLOATS` and divide:
    //     kc = clamp(L3_PANEL_FLOATS / (nthr * nc), KC_MIN, KC_MAX)
    // At nc=2048 that yields 1024 for one thread and 256 for sixteen — both of
    // which are the values the sweeps independently picked, which is the only
    // reason to trust the formula rather than a table.
    let denom = b.mul(nthr, &nc);
    let raw = b.t();
    b.w(&format!(
        "{} = udiv i64 {}, {}",
        raw, L3_PANEL_FLOATS, denom
    ));
    let capped = b.imin(&raw, &KC_MAX.to_string());
    let kc_max_eff = b.imax(&capped, &KC_MIN.to_string());

    let pcl = b.loop_begin("g.pc", "0", k, &kc_max_eff);
    let pc = b.iv(&pcl);
    let rk = b.sub(k, &pc);
    let kc = b.imin(&rk, &kc_max_eff);

    // packB(B + pc*N + jc, N, kc, nc)
    //
    // Skipping this pack when M is small was tried and is NOT a win: measured
    // at 4x4096x4096 it moved 0.29x -> 0.33x, nowhere near the 3x the traffic
    // arithmetic suggested. Packing is not overhead to be avoided here — it is
    // what turns B's access pattern linear. Reading B in place walks a
    // 32-float column panel down 256 rows at stride N, touching a new page
    // every row and revisiting each page once per panel. Small M is handled by
    // `__y_gemm_small_m` instead, which changes the loop order rather than the
    // packing.
    let bro = b.mul(&pc, n);
    let bo = b.add(&bro, &jc);
    let bsrc = b.gep("float", bb, &bo);
    // Cooperative: each thread packs every nthr'th panel of the shared block,
    // then all wait. Private: this thread packs the whole block alone.
    let pk_t = b.t();
    b.w(&format!("{} = select i1 {}, i64 {}, i64 0", pk_t, coop, tid));
    let pk_n = b.t();
    b.w(&format!("{} = select i1 {}, i64 {}, i64 1", pk_n, coop, nthr));
    b.w(&format!(
        "call void @__y_gemm_pack_b(ptr {}, i64 {}, i64 {}, i64 {}, ptr {}, i64 {}, i64 {})",
        bsrc, n, kc, nc, packb, pk_t, pk_n
    ));
    let bar1_l = b.l("g.bar1");
    let after1_l = b.l("g.after1");
    b.w(&format!("br i1 {}, label %{}, label %{}", coop, bar1_l, after1_l));
    b.out.push_str(&format!("{}:\n", bar1_l));
    b.w(&format!(
        "call i32 @pthread_barrier_wait(ptr {})",
        POOL_BARRIER
    ));
    b.w(&format!("br label %{}", after1_l));
    b.out.push_str(&format!("{}:\n", after1_l));

    let first = b.t();
    b.w(&format!("{} = icmp eq i64 {}, 0", first, pc));

    let icl = b.loop_begin("g.ic", m0, m1, &tl().mc.to_string());
    let ic = b.iv(&icl);
    let rm = b.sub(m1, &ic);
    let mc = b.imin(&rm, &tl().mc.to_string());

    // packA(A + ic*K + pc, K, mc, kc)
    let aro = b.mul(&ic, k);
    let ao = b.add(&aro, &pc);
    let asrc = b.gep("float", a, &ao);
    b.w(&format!(
        "call void @__y_gemm_pack_a(ptr {}, i64 {}, i64 {}, i64 {}, ptr {})",
        asrc, k, mc, kc, packa
    ));

    let jrl = b.loop_begin("g.jr", "0", &nc, &tl().nr.to_string());
    let jr = b.iv(&jrl);
    let rjw = b.sub(&nc, &jr);
    let jw = b.imin(&rjw, &tl().nr.to_string());

    let bpo = b.mul(&jr, &kc);
    let bpp = b.gep("float", &packb, &bpo);
    let bld = tl().nr.to_string();

    let irl = b.loop_begin("g.ir", "0", &mc, &tl().mr.to_string());
    let ir = b.iv(&irl);
    let rmw = b.sub(&mc, &ir);
    let mw = b.imin(&rmw, &tl().mr.to_string());
    let apo = b.mul(&ir, &kc);
    let app = b.gep("float", packa, &apo);

    // C + (ic+ir)*N + jc + jr
    // C column is jc + jr: `jc` is the absolute start of this NC block and
    // `jr` the offset of the micro-panel within it.
    let crow = b.add(&ic, &ir);
    let cro = b.mul(&crow, n);
    let cc1 = b.add(&cro, &jc);
    let cc2 = b.add(&cc1, &jr);
    let cp = b.gep("float", c, &cc2);

    b.w(&format!(
        "call void @__y_gemm_micro(ptr {}, ptr {}, i64 {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 {}, i1 {})",
        app, bpp, bld, cp, n, kc, mw, jw, first
    ));

    b.loop_end(irl);
    b.loop_end(jrl);
    b.loop_end(icl);

    // Second barrier: the next pc iteration overwrites the shared panel, so no
    // thread may start repacking until every thread has finished reading it.
    let bar2_l = b.l("g.bar2");
    let after2_l = b.l("g.after2");
    b.w(&format!("br i1 {}, label %{}, label %{}", coop, bar2_l, after2_l));
    b.out.push_str(&format!("{}:\n", bar2_l));
    b.w(&format!(
        "call i32 @pthread_barrier_wait(ptr {})",
        POOL_BARRIER
    ));
    b.w(&format!("br label %{}", after2_l));
    b.out.push_str(&format!("{}:\n", after2_l));

    b.loop_end(pcl);
    b.loop_end(jcl);

    b.finish(&format!(
        "define internal void @__y_gemm_run(ptr noalias {}, ptr noalias {}, ptr {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, i64 {}, ptr noalias {}, \
         i64 {}, i64 {}, i64 {})",
        a, bb, c, m, n, k, m0, m1, n0, n1, scratch, tid, nthr, shared
    ))
}

/// `__y_gemm_worker(ptr task) -> ptr` — pthread entry. Unpacks the task
/// record and runs the driver over this thread's slice.
fn emit_worker() -> String {
    let mut b = IrBuilder::new();
    let t = "%t";

    // %y_gemm_task = { A, B, C, M, N, K, m0, m1, n0, n1, scratch }
    let names = [
        ("a", "ptr", 0),
        ("bb", "ptr", 1),
        ("c", "ptr", 2),
        ("m", "i64", 3),
        ("n", "i64", 4),
        ("k", "i64", 5),
        ("m0", "i64", 6),
        ("m1", "i64", 7),
        ("n0", "i64", 8),
        ("n1", "i64", 9),
        ("scr", "ptr", 10),
        ("tid", "i64", 11),
        ("nthr", "i64", 12),
        ("shared", "i64", 13),
    ];
    let mut vals = Vec::new();
    for (_, ty, idx) in names {
        let p = b.t();
        b.w(&format!(
            "{} = getelementptr inbounds %y_gemm_task, ptr {}, i64 0, i32 {}",
            p, t, idx
        ));
        let v = b.t();
        b.w(&format!("{} = load {}, ptr {}, align 8", v, ty, p));
        vals.push(v);
    }

    b.w(&format!(
        "call void @__y_gemm_run(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 {}, i64 {}, i64 {}, i64 {}, ptr {}, i64 {}, i64 {}, i64 {})",
        vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8],
        vals[9], vals[10], vals[11], vals[12], vals[13]
    ));
    b.w("ret ptr null");

    let mut s = b.finish(&format!(
        "define internal ptr @__y_gemm_worker(ptr {})",
        t
    ));
    // `finish` appends `ret void`, which is wrong for a ptr-returning function.
    s = s.replace("  ret void\n}", "}");
    s
}

/// `__y_gemm_threads(M, N, K) -> i64` — how many threads to use.
///
/// This is OpenBLAS's shape, including its constants: refuse to thread below
/// `SMP_THRESHOLD`, then scale linearly with the work until the core count is
/// reached. `Y_NUM_THREADS` overrides the ceiling, and is resolved once and
/// cached because `getenv` on every GEMM would show up on the small shapes
/// this function exists to protect.
fn emit_threads() -> String {
    let mut b = IrBuilder::new();
    let (m, n, k) = ("%M", "%N", "%K");

    // Resolve the ceiling once.
    let cached = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", cached, NTHREADS_CACHE));
    let need = b.t();
    b.w(&format!("{} = icmp eq i64 {}, 0", need, cached));
    let res_l = b.l("nt.resolve");
    let have_l = b.l("nt.have");
    b.w(&format!("br i1 {}, label %{}, label %{}", need, res_l, have_l));

    b.out.push_str(&format!("{}:\n", res_l));
    let env = b.t();
    b.w(&format!("{} = call ptr @getenv(ptr @.y_gemm_env)", env));
    let has_env = b.t();
    b.w(&format!("{} = icmp ne ptr {}, null", has_env, env));
    let env_l = b.l("nt.env");
    let sys_l = b.l("nt.sys");
    let set_l = b.l("nt.set");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        has_env, env_l, sys_l
    ));

    b.out.push_str(&format!("{}:\n", env_l));
    let ev32 = b.t();
    b.w(&format!("{} = call i32 @atoi(ptr {})", ev32, env));
    let ev = b.t();
    b.w(&format!("{} = sext i32 {} to i64", ev, ev32));
    b.w(&format!("br label %{}", set_l));

    b.out.push_str(&format!("{}:\n", sys_l));
    // _SC_NPROCESSORS_ONLN = 84 on Linux.
    let sc = b.t();
    b.w(&format!("{} = call i64 @sysconf(i32 84)", sc));
    b.w(&format!("br label %{}", set_l));

    b.out.push_str(&format!("{}:\n", set_l));
    let raw = b.t();
    b.w(&format!(
        "{} = phi i64 [ {}, %{} ], [ {}, %{} ]",
        raw, ev, env_l, sc, sys_l
    ));
    let atleast1 = b.imax(&raw, "1");
    let capped = b.imin(&atleast1, &MAX_THREADS.to_string());
    b.w(&format!(
        "store i64 {}, ptr {}, align 8",
        capped, NTHREADS_CACHE
    ));
    b.w(&format!("br label %{}", have_l));

    b.out.push_str(&format!("{}:\n", have_l));
    let ceiling = b.t();
    b.w(&format!(
        "{} = load i64, ptr {}, align 8",
        ceiling, NTHREADS_CACHE
    ));

    // work = M*N*K, saturating: the product overflows i64 only at sizes that
    // cannot be allocated, but clamping keeps the comparison meaningful.
    let mn = b.mul(m, n);
    let work = b.mul(&mn, k);
    let big = b.t();
    b.w(&format!(
        "{} = icmp sgt i64 {}, {}",
        big, work, WORK_PER_THREAD
    ));
    let scale_l = b.l("nt.scale");
    let one_l = b.l("nt.one");
    let out_l = b.l("nt.out");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        big, scale_l, one_l
    ));

    b.out.push_str(&format!("{}:\n", scale_l));
    let by_work = b.t();
    b.w(&format!(
        "{} = sdiv i64 {}, {}",
        by_work, work, WORK_PER_THREAD
    ));
    let scaled = b.imin(&by_work, &ceiling);
    let scaled = b.imax(&scaled, "1");
    b.w(&format!("br label %{}", out_l));

    b.out.push_str(&format!("{}:\n", one_l));
    b.w(&format!("br label %{}", out_l));

    b.out.push_str(&format!("{}:\n", out_l));
    let r = b.t();
    b.w(&format!(
        "{} = phi i64 [ {}, %{} ], [ 1, %{} ]",
        r, scaled, scale_l, one_l
    ));
    b.w(&format!("ret i64 {}", r));

    let mut s = b.finish("define internal i64 @__y_gemm_threads(i64 %M, i64 %N, i64 %K)");
    s = s.replace("  ret void\n}", "}");
    s
}

/// `__y_pool_worker(ptr idx)` — a persistent worker. Waits for the dispatch
/// generation to change, runs its slice, publishes completion, waits again.
///
/// This replaces a `pthread_create`/`pthread_join` pair per GEMM. That cost
/// ~25 us per thread, which at 16 threads is ~400 us of fork/join against a
/// 256^3 GEMM that takes 135 us single-threaded — so threading made the small
/// shapes several times *slower* than not threading. OpenBLAS has had a
/// persistent thread server (`driver/others/blas_server.c`) for exactly this
/// reason; dispatch here is one release store and a spin on an acquire load.
///
/// Idle workers spin briefly and then park in `usleep`, so a pool that is not
/// in use costs no CPU and does not perturb anything measured next to it.
fn emit_pool_worker() -> String {
    let mut b = IrBuilder::new();
    let idxp = "%idxp";

    let my = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", my, idxp));
    let donep = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 {}",
        donep, MAX_THREADS, POOL_DONE, my
    ));
    let taskp = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x %y_gemm_task], ptr {}, i64 0, i64 {}",
        taskp, MAX_THREADS, POOL_TASK, my
    ));

    b.entry_alloca("%.last", "i64", 8);
    b.entry_alloca("%.spin", "i64", 8);
    b.entry_alloca("%.gen", "i64", 8);
    b.w("store i64 0, ptr %.last, align 8");

    let outer = b.l("w.outer");
    let wait = b.l("w.wait");
    let work = b.l("w.work");
    let park = b.l("w.park");
    b.w(&format!("br label %{}", outer));

    b.out.push_str(&format!("{}:\n", outer));
    b.w("store i64 0, ptr %.spin, align 8");
    b.w(&format!("br label %{}", wait));

    b.out.push_str(&format!("{}:\n", wait));
    let g = b.t();
    b.w(&format!(
        "{} = load atomic i64, ptr {} acquire, align 8",
        g, POOL_GEN
    ));
    let last = b.t();
    b.w(&format!("{} = load i64, ptr %.last, align 8", last));
    let changed = b.t();
    b.w(&format!("{} = icmp ne i64 {}, {}", changed, g, last));
    b.w(&format!("store i64 {}, ptr %.gen, align 8", g));
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        changed, work, park
    ));

    b.out.push_str(&format!("{}:\n", park));
    let sp = b.t();
    b.w(&format!("{} = load i64, ptr %.spin, align 8", sp));
    let sp1 = b.add(&sp, "1");
    b.w(&format!("store i64 {}, ptr %.spin, align 8", sp1));
    let hot = b.t();
    b.w(&format!("{} = icmp slt i64 {}, {}", hot, sp1, POOL_SPIN));
    let spin_l = b.l("w.spin");
    let block_l = b.l("w.block");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        hot, spin_l, block_l
    ));
    b.out.push_str(&format!("{}:\n", spin_l));
    b.w("call void @llvm.x86.sse2.pause()");
    b.w(&format!("br label %{}", wait));

    // Blocked: hold the mutex, re-check under it, and wait. An idle pool costs
    // nothing here — no wakeups, no cache traffic, no stolen issue slots.
    b.out.push_str(&format!("{}:\n", block_l));
    b.w(&format!("call i32 @pthread_mutex_lock(ptr {})", POOL_MUTEX));
    let recheck = b.l("w.recheck");
    b.w(&format!("br label %{}", recheck));
    b.out.push_str(&format!("{}:\n", recheck));
    let g2 = b.t();
    b.w(&format!(
        "{} = load atomic i64, ptr {} acquire, align 8",
        g2, POOL_GEN
    ));
    let last2 = b.t();
    b.w(&format!("{} = load i64, ptr %.last, align 8", last2));
    let ready = b.t();
    b.w(&format!("{} = icmp ne i64 {}, {}", ready, g2, last2));
    let wake_l = b.l("w.wake");
    let sleep_l = b.l("w.sleep");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        ready, wake_l, sleep_l
    ));
    b.out.push_str(&format!("{}:\n", sleep_l));
    // Spurious wakeups are why this re-checks in a loop rather than once.
    b.w(&format!(
        "call i32 @pthread_cond_wait(ptr {}, ptr {})",
        POOL_COND, POOL_MUTEX
    ));
    b.w(&format!("br label %{}", recheck));
    b.out.push_str(&format!("{}:\n", wake_l));
    b.w(&format!("call i32 @pthread_mutex_unlock(ptr {})", POOL_MUTEX));
    b.w(&format!("store i64 {}, ptr %.gen, align 8", g2));
    b.w(&format!("br label %{}", work));

    b.out.push_str(&format!("{}:\n", work));
    let gw = b.t();
    b.w(&format!("{} = load i64, ptr %.gen, align 8", gw));
    b.w(&format!("store i64 {}, ptr %.last, align 8", gw));
    b.w(&format!("call void @__y_gemm_worker(ptr {})", taskp));
    // Publish the generation this iteration actually ran, not the one the
    // spin path happened to read: a worker that reached here after blocking
    // read its stale pre-block value there, so publishing that left `done[]`
    // one generation behind and the dispatcher waiting forever.
    b.w(&format!(
        "store atomic i64 {}, ptr {} release, align 8",
        gw, donep
    ));
    b.w(&format!("br label %{}", outer));

    let mut s = b.finish(&format!(
        "define internal ptr @__y_pool_worker(ptr {})",
        idxp
    ));
    s = s.replace("  ret void\n}", "}");
    s
}

/// The public entry: pick a thread count, size the scratch, split the work,
/// dispatch to the pool, run one slice inline, wait.
///
/// The split is 1-D and adaptive. OpenBLAS partitions in both M and N with a
/// `switch_ratio` guard so neither dimension is cut below a useful width; the
/// same concern applies here with only one axis to choose, so the axis is
/// picked by which one has enough width to give every thread a whole number of
/// micro-tiles. Cutting M when M is small would hand threads empty slices —
/// which is exactly the skinny/decode family.
fn emit_entry() -> String {
    let mut b = IrBuilder::new();
    let (a, bb, c, m, n, k) = ("%A", "%B", "%C", "%M", "%N", "%K");

    let nt = b.t();
    b.w(&format!(
        "{} = call i64 @__y_gemm_threads(i64 {}, i64 {}, i64 {})",
        nt, m, n, k
    ));

    // Single-threaded shapes skip the pool entirely: no dispatch, no scratch
    // growth, no generation bump.
    let solo = b.t();
    b.w(&format!("{} = icmp sle i64 {}, 1", solo, nt));
    let solo_l = b.l("e.solo");
    let par_l = b.l("e.par");
    b.w(&format!("br i1 {}, label %{}, label %{}", solo, solo_l, par_l));

    b.out.push_str(&format!("{}:\n", solo_l));
    let s0 = b.t();
    b.w(&format!("{} = load ptr, ptr {}, align 8", s0, SCRATCH));
    let has0 = b.t();
    b.w(&format!("{} = icmp ne ptr {}, null", has0, s0));
    let scr0 = b.t();
    b.w(&format!(
        "{} = select i1 {}, ptr {}, ptr {}",
        scr0, has0, s0, FALLBACK
    ));
    b.w(&format!(
        "call void @__y_gemm_run(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 0, i64 {}, i64 0, i64 {}, ptr {}, i64 0, i64 1, i64 0)",
        a, bb, c, m, n, k, m, n, scr0
    ));
    b.w("ret void");

    b.out.push_str(&format!("{}:\n", par_l));

    // Start the pool once, sized to the cap, and give it scratch to match.
    let started = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", started, POOL_N));
    let cold = b.t();
    b.w(&format!("{} = icmp eq i64 {}, 0", cold, started));
    let start_l = b.l("e.start");
    let hot_l = b.l("e.hot");
    b.w(&format!("br i1 {}, label %{}, label %{}", cold, start_l, hot_l));

    b.out.push_str(&format!("{}:\n", start_l));
    let cap = b.t();
    b.w(&format!(
        "{} = load i64, ptr {}, align 8",
        cap, NTHREADS_CACHE
    ));
    let bytes = b.mul(&cap, &(SCRATCH_FLOATS * 4).to_string());
    let fresh = b.t();
    b.w(&format!("{} = call ptr @malloc(i64 {})", fresh, bytes));
    b.w(&format!("store ptr {}, ptr {}, align 8", fresh, SCRATCH));
    let ok = b.t();
    b.w(&format!("{} = icmp ne ptr {}, null", ok, fresh));
    let oom_l = b.l("e.oom");
    let spawn_l = b.l("e.spawn0");
    b.w(&format!("br i1 {}, label %{}, label %{}", ok, spawn_l, oom_l));

    b.out.push_str(&format!("{}:\n", oom_l));
    // Fall back to one thread on the statically reserved panel. This must not
    // be an `alloca`: an entry-block alloca of this size is emitted on every
    // call, not just on the path that uses it, and a 2.3 MB stack frame per
    // GEMM costs more than the whole kernel at small sizes.
    b.w(&format!(
        "call void @__y_gemm_run(ptr {}, ptr {}, ptr {}, i64 {}, i64 {}, i64 {}, \
         i64 0, i64 {}, i64 0, i64 {}, ptr {}, i64 0, i64 1, i64 0)",
        a, bb, c, m, n, k, m, n, FALLBACK
    ));
    b.w("ret void");

    b.out.push_str(&format!("{}:\n", spawn_l));
    b.w(&format!(
        "call i32 @pthread_mutex_init(ptr {}, ptr null)",
        POOL_MUTEX
    ));
    b.w(&format!(
        "call i32 @pthread_cond_init(ptr {}, ptr null)",
        POOL_COND
    ));
    b.entry_alloca("%.tid", "i64", 8);
    let sl = b.loop_begin("e.spawn", "1", &cap, "1");
    let si = b.iv(&sl);
    let idp = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 {}",
        idp, MAX_THREADS, POOL_IDS, si
    ));
    b.w(&format!("store i64 {}, ptr {}, align 8", si, idp));
    b.w(&format!(
        "call i32 @pthread_create(ptr %.tid, ptr null, ptr @__y_pool_worker, ptr {})",
        idp
    ));
    b.loop_end(sl);
    b.w(&format!("store i64 {}, ptr {}, align 8", cap, POOL_N));
    b.w(&format!("br label %{}", hot_l));

    b.out.push_str(&format!("{}:\n", hot_l));
    let pool_n = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", pool_n, POOL_N));
    let scratch = b.t();
    b.w(&format!("{} = load ptr, ptr {}, align 8", scratch, SCRATCH));

    // Choose the split axis by which one yields more whole micro-tiles, and
    // then cap the thread count by that.
    //
    // The previous rule cut M only when `M >= nt * MR` and fell back to N
    // otherwise, which inverts badly when both extents are small: at
    // 64x64x32768 with 16 threads, M=64 fails `>= 192`, so N=64 was cut into
    // two useful NR slices and fourteen threads got nothing — 157 GFLOPS where
    // four threads cutting M reach 257.
    let pm = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", pm, m, tl().mr));
    let pm = b.imax(&pm, "1");
    let pn = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", pn, n, tl().nr));
    let pn = b.imax(&pn, "1");
    let cut_m = b.t();
    b.w(&format!("{} = icmp sge i64 {}, {}", cut_m, pm, pn));
    let par = b.imax(&pm, &pn);
    let nt = b.imin(&nt, &par);
    let axis = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        axis, cut_m, m, n
    ));
    let gran = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        gran, cut_m, tl().mr, tl().nr
    ));

    // Threads sharing one packed B panel must be exactly the threads the
    // barrier is sized for. Only the M-split shares; an N-split gives each
    // thread a disjoint B block, so there is nothing to share and no barrier.
    let mn2 = b.mul(m, n);
    let work2 = b.mul(&mn2, k);
    let big_enough = b.t();
    b.w(&format!(
        "{} = icmp sge i64 {}, {}",
        big_enough, work2, SHARE_B_WORK
    ));
    let do_share = b.t();
    b.w(&format!("{} = and i1 {}, {}", do_share, cut_m, big_enough));
    let share = b.t();
    b.w(&format!("{} = zext i1 {} to i64", share, do_share));
    let resize_l = b.l("e.bar");
    let barok_l = b.l("e.barok");
    let bn = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", bn, BARRIER_N));
    let stale = b.t();
    b.w(&format!("{} = icmp ne i64 {}, {}", stale, bn, nt));
    let need_bar = b.t();
    b.w(&format!("{} = and i1 {}, {}", need_bar, stale, do_share));
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        need_bar, resize_l, barok_l
    ));
    b.out.push_str(&format!("{}:\n", resize_l));
    let had = b.t();
    b.w(&format!("{} = icmp ne i64 {}, 0", had, bn));
    let destroy_l = b.l("e.bardestroy");
    let init_l = b.l("e.barinit");
    b.w(&format!(
        "br i1 {}, label %{}, label %{}",
        had, destroy_l, init_l
    ));
    b.out.push_str(&format!("{}:\n", destroy_l));
    b.w(&format!(
        "call i32 @pthread_barrier_destroy(ptr {})",
        POOL_BARRIER
    ));
    b.w(&format!("br label %{}", init_l));
    b.out.push_str(&format!("{}:\n", init_l));
    let nt32 = b.t();
    b.w(&format!("{} = trunc i64 {} to i32", nt32, nt));
    b.w(&format!(
        "call i32 @pthread_barrier_init(ptr {}, ptr null, i32 {})",
        POOL_BARRIER, nt32
    ));
    b.w(&format!("store i64 {}, ptr {}, align 8", nt, BARRIER_N));
    b.w(&format!("br label %{}", barok_l));
    b.out.push_str(&format!("{}:\n", barok_l));

    // Fill every pool slot, not just the first `nt`: all workers wake on the
    // generation bump, so a stale record would have one recomputing an old
    // slice on top of this call's output. Slots past `nt` get an empty range.
    let fl = b.loop_begin("e.fill", "0", &pool_n, "1");
    let ti = b.iv(&fl);
    // Cut at t*axis/nt, snapped down to the tile granularity, with the last
    // thread taking whatever is left. This keeps every thread busy; rounding
    // each width UP to a whole tile overshoots and leaves the tail threads
    // with an empty range, which is how 256^3 ended up on 11 of 16 threads.
    let lo0 = b.mul(&ti, &axis);
    let lo1 = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", lo1, lo0, nt));
    let lo2 = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", lo2, lo1, gran));
    let lo = b.mul(&lo2, &gran);
    let tn = b.add(&ti, "1");
    let hi0 = b.mul(&tn, &axis);
    let hi1 = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", hi1, hi0, nt));
    let hi2 = b.t();
    b.w(&format!("{} = sdiv i64 {}, {}", hi2, hi1, gran));
    let hi3 = b.mul(&hi2, &gran);
    let is_last = b.t();
    b.w(&format!("{} = icmp eq i64 {}, {}", is_last, tn, nt));
    let hi4 = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        hi4, is_last, axis, hi3
    ));
    let inuse = b.t();
    b.w(&format!("{} = icmp slt i64 {}, {}", inuse, ti, nt));
    let hi = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        hi, inuse, hi4, lo
    ));
    let slot = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x %y_gemm_task], ptr {}, i64 0, i64 {}",
        slot, MAX_THREADS, POOL_TASK, ti
    ));
    let m_from = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 0",
        m_from, cut_m, lo
    ));
    let m_to = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        m_to, cut_m, hi, m
    ));
    let n_from = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 0, i64 {}",
        n_from, cut_m, lo
    ));
    let n_to = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        n_to, cut_m, n, hi
    ));
    // A slot past `nt` must also present an empty range on the *other* axis,
    // or it would recompute the whole of it.
    let m_to2 = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        m_to2, inuse, m_to, m_from
    ));
    let n_to2 = b.t();
    b.w(&format!(
        "{} = select i1 {}, i64 {}, i64 {}",
        n_to2, inuse, n_to, n_from
    ));
    let scr_off = b.mul(&ti, &SCRATCH_FLOATS.to_string());
    let scr = b.gep("float", &scratch, &scr_off);
    for (idx, ty, val) in [
        (0, "ptr", a.to_string()),
        (1, "ptr", bb.to_string()),
        (2, "ptr", c.to_string()),
        (3, "i64", m.to_string()),
        (4, "i64", n.to_string()),
        (5, "i64", k.to_string()),
        (6, "i64", m_from),
        (7, "i64", m_to2),
        (8, "i64", n_from),
        (9, "i64", n_to2),
        (10, "ptr", scr),
        (11, "i64", ti.clone()),
        (12, "i64", nt.clone()),
        (13, "i64", share.clone()),
    ] {
        let f = b.t();
        b.w(&format!(
            "{} = getelementptr inbounds %y_gemm_task, ptr {}, i64 0, i32 {}",
            f, slot, idx
        ));
        b.w(&format!("store {} {}, ptr {}, align 8", ty, val, f));
    }
    b.loop_end(fl);

    // Publish. The release pairs with the workers' acquire load, so every task
    // record above is visible before any worker observes the new generation.
    let gcur = b.t();
    b.w(&format!("{} = load i64, ptr {}, align 8", gcur, POOL_GEN));
    let gnext = b.add(&gcur, "1");
    b.w(&format!("call i32 @pthread_mutex_lock(ptr {})", POOL_MUTEX));
    b.w(&format!(
        "store atomic i64 {}, ptr {} release, align 8",
        gnext, POOL_GEN
    ));
    b.w(&format!("call i32 @pthread_cond_broadcast(ptr {})", POOL_COND));
    b.w(&format!("call i32 @pthread_mutex_unlock(ptr {})", POOL_MUTEX));

    // Slice 0 on this thread, so a core is never idle waiting on its own fork.
    let own = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x %y_gemm_task], ptr {}, i64 0, i64 0",
        own, MAX_THREADS, POOL_TASK
    ));
    b.w(&format!("call void @__y_gemm_worker(ptr {})", own));

    let jl = b.loop_begin("e.join", "1", &pool_n, "1");
    let ji = b.iv(&jl);
    let dp = b.t();
    b.w(&format!(
        "{} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 {}",
        dp, MAX_THREADS, POOL_DONE, ji
    ));
    let spin = b.l("e.spinwait");
    let done = b.l("e.done");
    b.w(&format!("br label %{}", spin));
    b.out.push_str(&format!("{}:\n", spin));
    let dv = b.t();
    b.w(&format!(
        "{} = load atomic i64, ptr {} acquire, align 8",
        dv, dp
    ));
    let fin = b.t();
    b.w(&format!("{} = icmp eq i64 {}, {}", fin, dv, gnext));
    let again = b.l("e.again");
    b.w(&format!("br i1 {}, label %{}, label %{}", fin, done, again));
    b.out.push_str(&format!("{}:\n", again));
    // Without this the waiting thread hammers the very cache line the workers
    // are publishing into, and on an SMT sibling it steals issue slots from a
    // core still doing useful FMAs.
    b.w("call void @llvm.x86.sse2.pause()");
    b.w(&format!("br label %{}", spin));
    b.out.push_str(&format!("{}:\n", done));
    b.loop_end(jl);

    b.finish(&format!(
        "define internal void @{}(ptr noalias {}, ptr noalias {}, ptr noalias {}, \
         i64 {}, i64 {}, i64 {})",
        KERNEL_NAME, a, bb, c, m, n, k
    ))
}

/// `x - 1`, as a fresh temporary.
fn b_sub1(b: &mut IrBuilder, x: &str) -> String {
    let r = b.t();
    b.w(&format!("{} = sub nsw i64 {}, 1", r, x));
    r
}

/// The whole GEMM support module: globals, packing routines, micro-kernel and
/// blocked driver.
pub fn emit_kernel_module() -> String {
    let mut s = String::new();
    s.push_str(&emit_globals());
    s.push('\n');
    s.push_str(&emit_pack_b());
    s.push('\n');
    s.push_str(&emit_pack_a());
    s.push('\n');
    s.push_str(&emit_micro());
    s.push('\n');
    s.push_str(&emit_small_m());
    s.push('\n');
    s.push_str(&emit_driver());
    s.push('\n');
    s.push_str(&emit_worker());
    s.push('\n');
    s.push_str(&emit_threads());
    s.push('\n');
    s.push_str(&emit_pool_worker());
    s.push('\n');
    s.push_str(&emit_entry());
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn kernel_body(src: &str) -> Block {
        let mut lx = Lexer::new(src);
        let toks = lx.tokenize();
        let mut p = Parser::new(toks);
        let prog = p.parse_program().expect("test source should parse");
        for item in prog.items {
            if let Item::Kernel(k) = item {
                return k.body;
            }
        }
        panic!("no kernel in source");
    }

    const CANONICAL: &str = r#"
kernel mm(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, M: I32, N: I32, K: I32) {
    for i in 0..M step 1 {
        for j in 0..N step 1 {
            let mut sum: F32 = 0.0;
            for k in 0..K step 1 {
                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);
                let b_val: F32 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a_val * b_val;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}
"#;

    #[test]
    fn canonical_matmul_is_recognized() {
        let shape = recognize_gemm(&kernel_body(CANONICAL)).expect("should recognize");
        assert_eq!(shape.a, "A");
        assert_eq!(shape.b, "B");
        assert_eq!(shape.c, "C");
        assert_eq!(shape.m, "M");
        assert_eq!(shape.n, "N");
        assert_eq!(shape.k, "K");
    }

    /// The recogniser decides to compute something *different* from what was
    /// written, so every one of these near-misses must be refused. A false
    /// positive here is a wrong answer that no test of the emitted IR would
    /// catch, because the IR would be a correct GEMM — just not this program.
    #[test]
    fn near_misses_are_refused() {
        // B indexed [j, k] instead of [k, j] — that is A * Bᵀ.
        let transposed = CANONICAL.replace(
            "block_ptr2d_load(B, k, j, N, K, N)",
            "block_ptr2d_load(B, j, k, N, K, N)",
        );
        assert!(recognize_gemm(&kernel_body(&transposed)).is_none());

        // Subtraction, not accumulation.
        let minus = CANONICAL.replace("sum = sum + a_val", "sum = sum - a_val");
        assert!(recognize_gemm(&kernel_body(&minus)).is_none());

        // A non-unit step visits a strict subset of K.
        let step2 = CANONICAL.replace("for k in 0..K step 1", "for k in 0..K step 2");
        assert!(recognize_gemm(&kernel_body(&step2)).is_none());

        // A stride that is not the matching extent means a different layout.
        let stride = CANONICAL.replace(
            "block_ptr2d_load(A, i, k, K, M, K)",
            "block_ptr2d_load(A, i, k, M, M, K)",
        );
        assert!(recognize_gemm(&kernel_body(&stride)).is_none());

        // An accumulator that starts from something other than zero.
        let init = CANONICAL.replace("let mut sum: F32 = 0.0;", "let mut sum: F32 = 1.0;");
        assert!(recognize_gemm(&kernel_body(&init)).is_none());
    }

    #[test]
    fn emitted_module_has_the_expected_pieces() {
        let ir = emit_kernel_module();
        assert!(ir.contains(&format!("define internal void @{}", KERNEL_NAME)));
        assert!(ir.contains("@__y_gemm_pack_a"));
        assert!(ir.contains("@__y_gemm_pack_b"));
        assert!(ir.contains("@__y_gemm_micro"));
        assert!(ir.contains("llvm.fmuladd.v16f32"));
        // MR * NRV accumulators, all distinct.
        for i in 0..tl().mr {
            for v in 0..tl().nrv() {
                assert!(ir.contains(&format!("%acc{}_{} = alloca <16 x float>", i, v)));
            }
        }
    }

    /// A tile over the register budget must be REFUSED, not emitted.
    ///
    /// An over-budget tile compiles and runs — it just spills every
    /// accumulator to the stack in the innermost loop. That reads as a
    /// mysterious 3x slowdown rather than as a bad tile, which is exactly the
    /// failure mode `Y_GEMM_TILE` would otherwise invite during a sweep.
    #[test]
    fn over_budget_tiles_are_refused() {
        // 16x4: 64 accumulators + 4 B + 2 A against 32 zmm.
        let t = Tile { mr: 16, nr: 64, kc: 256, mc: 192, nc: 2048 };
        assert!(t.regs() > 32);
        assert!(t.check().is_err(), "16x64 needs {} registers", t.regs());

        // The shipped default and every tile named in docs/cpu_gemm_tuning.md
        // must pass, or the sweep table describes tiles that cannot be built.
        for t in [
            DEFAULT_TILE,
            Tile { mr: 12, nr: 32, kc: 256, mc: 192, nc: 2048 },
            Tile { mr: 8, nr: 48, kc: 256, mc: 192, nc: 2048 },
            Tile { mr: 6, nr: 64, kc: 1024, mc: 384, nc: 2048 },
        ] {
            assert!(t.check().is_ok(), "{:?} should be accepted: {:?}", t, t.check());
            assert!(t.regs() <= 32);
        }

        // A non-vector-width nr is refused rather than silently truncated by
        // the `nr / 16` division.
        assert!(Tile { mr: 6, nr: 40, kc: 256, mc: 192, nc: 2048 }.check().is_err());
        // And anything past what SCRATCH_FLOATS reserves.
        assert!(Tile { mr: 6, nr: 64, kc: KC_MAX + 1, mc: 192, nc: 2048 }.check().is_err());
    }

    /// The packed-A region must not be able to reach the packed-B region, for
    /// ANY `kc` the runtime rule can pick — not merely for the current tile.
    ///
    /// `kc` is chosen at runtime from the thread count, so a B offset derived
    /// from `tl().kc` puts the A panel's tail on top of B whenever the runtime
    /// value is larger. That is a wrong answer, not a crash, and it appears
    /// only on the shapes the runtime `kc` exists to speed up.
    #[test]
    fn scratch_layout_holds_for_the_largest_runtime_kc() {
        let b_offset = (MC_MAX + MR_MAX) * KC_MAX;
        let widest_a = (MC_MAX + MR_MAX) * KC_MAX;
        assert!(b_offset >= widest_a, "A panel can overrun into B");

        let widest_b = KC_MAX * (NC_MAX + NR_MAX);
        assert!(
            SCRATCH_FLOATS >= b_offset + widest_b,
            "SCRATCH_FLOATS={} cannot hold A({}) + B({})",
            SCRATCH_FLOATS,
            b_offset,
            widest_b
        );

        // The emitted offset must be the max-derived one. A tile-derived
        // offset would still pass the arithmetic above while being wrong.
        let ir = emit_kernel_module();
        assert!(
            ir.contains(&format!("i64 {}", b_offset)),
            "packed-B offset is not the SCRATCH_FLOATS-derived one"
        );

        // And the runtime rule must actually be emitted, with the budget the
        // docs quote. If this constant moves, the sweep table is stale.
        assert!(
            ir.contains(&format!("udiv i64 {}", L3_PANEL_FLOATS)),
            "the thread-count-dependent kc rule is not in the emitted IR"
        );
    }

    /// The runtime `kc` rule must reproduce both measured optima.
    ///
    /// These two values were found by independent sweeps before the formula
    /// existed; the formula is only trustworthy because it lands on them.
    #[test]
    fn runtime_kc_rule_matches_the_measured_optima() {
        let rule = |nthr: usize, nc: usize| {
            (L3_PANEL_FLOATS / (nthr * nc)).min(KC_MAX).max(KC_MIN)
        };
        assert_eq!(rule(1, 2048), 1024, "one thread should get the deep panel");
        assert_eq!(rule(16, 2048), 256, "sixteen threads should get the shallow one");
        // Never below the floor, however many threads or however wide nc is.
        assert_eq!(rule(64, 2048), KC_MIN);
    }
}
