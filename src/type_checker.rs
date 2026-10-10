// ============================================================
//  Y  —  Semantic Type Checker
//  type_checker.rs
//
//  The core brain of Y's safety guarantees.
//  Traverses AST, enforces Fragment roles (A vs B vs C),
//  manages linear memory obligations, and runs the
//  0-Bank-Conflict math prover.
// ============================================================


use crate::ast::*;
use crate::bank_conflict::{BankConflictProver, SmemLayout as ProverLayout, SwizzlePattern};
use crate::guarantees::{render, Assumption, Fact, Guarantees, ItemRange, Status};
use crate::linear_tracker::LinearTracker;
use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::io::Write;

thread_local! {
    pub static SAFE_INDICES: std::cell::RefCell<std::collections::HashSet<(usize, usize)>> = std::cell::RefCell::new(std::collections::HashSet::new());
    pub static INDEX_ARRAY_SIZES: std::cell::RefCell<std::collections::HashMap<(usize, usize), usize>> = std::cell::RefCell::new(std::collections::HashMap::new());
    pub static INDEX_SWIZZLES: std::cell::RefCell<std::collections::HashMap<(usize, usize), SwizzlePattern>> = std::cell::RefCell::new(std::collections::HashMap::new());
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub min: i64,
    pub max: i64,
    /// The assumptions this range rests on, as bits of
    /// `TypeChecker::assumptions` (bit 63 stands for the 64th and later). A
    /// `@bounds` the checker cannot check is taken on trust, and a proof that
    /// used a range resting on one is a proof from that assumption - which
    /// `ydb verify` has to be able to say. Every construction site states it,
    /// so none can drop it.
    pub trust: u64,
}

/// The assumption bit for CUDA's launch limits, the ranges
/// `gpu_index_interval` gives the GPU index intrinsics. Always assumption 0.
const LAUNCH_LIMITS: u64 = 1;

/// One array index the checker saw, for `finish_index_facts`.
struct IndexSite {
    item: String,
    what: String,
    /// The array's element count, when the base is a fixed-size array.
    size: Option<usize>,
    /// What the base is, for an index whose length is not known.
    base: String,
    /// The interval that proved it, from a visit that did.
    proving: Option<Interval>,
}

/// What strict mode checks, which every function and kernel not marked
/// `@unsafe` is checked under.
const STRICT_DETAIL: &str = "strict mode: every `let` is initialised, no raw pointer is dereferenced, \
every loop carries an @invariant that z3 must prove, every fixed-size array index is proved in \
bounds, and a `@bounds` variable is only assigned values inside its bounds - or the program is \
refused. An index into a pointer whose length the compiler does not know is not covered.";

#[derive(Clone, Copy)]
enum CompileTimeValue {
    Integer(i64),
    Boolean(bool),
}

/// Evaluate the constant subset without consulting inferred runtime bounds.
/// Unsupported expressions and overflowing arithmetic are proof failures.
fn eval_compile_time(expr: &Expr) -> Result<CompileTimeValue, &'static str> {
    use CompileTimeValue::{Boolean, Integer};
    let invalid = "invalid or overflowing integer constant arithmetic";
    match expr {
        Expr::IntLit(n, _) => Ok(Integer(*n)),
        Expr::BoolLit(b, _) => Ok(Boolean(*b)),
        Expr::UnaryOp { op, operand, .. } => match (op, eval_compile_time(operand)?) {
            (UnaryOp::Neg, Integer(n)) => n.checked_neg().map(Integer).ok_or(invalid),
            (UnaryOp::Not, Boolean(b)) => Ok(Boolean(!b)),
            _ => Err("unsupported constant unary expression"),
        },
        Expr::BinaryOp { left, op, right, .. } => {
            let lhs = eval_compile_time(left)?;
            let rhs = eval_compile_time(right)?;
            match (lhs, rhs) {
                (Boolean(a), Boolean(b)) => match op {
                    BinaryOp::And => Ok(Boolean(a && b)),
                    BinaryOp::Or => Ok(Boolean(a || b)),
                    BinaryOp::Eq => Ok(Boolean(a == b)),
                    BinaryOp::NotEq => Ok(Boolean(a != b)),
                    _ => Err("unsupported constant boolean operator"),
                },
                (Integer(a), Integer(b)) => {
                    let n = match op {
                        BinaryOp::Eq => return Ok(Boolean(a == b)),
                        BinaryOp::NotEq => return Ok(Boolean(a != b)),
                        BinaryOp::Lt => return Ok(Boolean(a < b)),
                        BinaryOp::Le => return Ok(Boolean(a <= b)),
                        BinaryOp::Gt => return Ok(Boolean(a > b)),
                        BinaryOp::Ge => return Ok(Boolean(a >= b)),
                        BinaryOp::Add => a.checked_add(b),
                        BinaryOp::Sub => a.checked_sub(b),
                        BinaryOp::Mul => a.checked_mul(b),
                        BinaryOp::Div => a.checked_div(b),
                        BinaryOp::Mod => a.checked_rem(b),
                        BinaryOp::BitAnd => Some(a & b),
                        BinaryOp::BitOr => Some(a | b),
                        BinaryOp::BitXor => Some(a ^ b),
                        _ => return Err("unsupported constant integer operator"),
                    };
                    n.map(Integer).ok_or(invalid)
                }
                _ => Err("constant operands have incompatible types"),
            }
        }
        _ => Err("expected a constant expression of integer or boolean literals"),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticType {
    Void,
    Primitive(String),
    Fragment {
        op: String,
        role: String,
        dtype: String,
    },
    SharedMemoryTile {
        rows: u32,
        cols: u32,
        swizzle: Option<SwizzlePattern>,
    },
    GlobalMemory(String),
    Vector(Box<SemanticType>, String), // Tuple of inner type and allocator
    Array {
        element: Box<SemanticType>,
        size: usize,
    },
    BlockTile {
        element: Box<SemanticType>,
        size: usize,
    },
    TransferObligation,
    Pipeline,
    /// `&T` / `&mut T`. This used to resolve to `Unknown`, and `Unknown` is
    /// exempted from the assignment mismatch check, so `let r: &F32 = &x;`
    /// with `x: I32` compiled clean.
    Reference {
        inner: Box<SemanticType>,
        mutable: bool,
    },
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnconstrainedReason {
    HintOutput(String),
    UnconstrainedInput(String),
    Merged(Vec<UnconstrainedReason>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintState {
    Constrained,
    TaintedUnconstrained {
        origins: Vec<Span>,
        reasons: Vec<UnconstrainedReason>,
    },
    DeferredObligation {
        origins: Vec<Span>,
        reasons: Vec<UnconstrainedReason>,
        override_span: Span,
    },
    Verified {
        origins: Vec<Span>,
        verified_span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignalConstraintInfo {
    pub name: String,
    pub state: ConstraintState,
    pub declared_span: Span,
}

/// Per-variable entry in a unified scope frame, combining type, interval,
/// explicit-bound flag, and constraint info into a single cache-friendly record.
pub struct SymbolEntry {
    pub ty: SemanticType,
    pub interval: Option<Interval>,
    pub is_explicitly_bounded: bool,
    pub constraint_info: Option<SignalConstraintInfo>,
}

/// A single scope frame that replaces the former four parallel scope stacks.
pub struct ScopeFrame {
    pub symbols: HashMap<String, SymbolEntry>,
}

#[derive(Clone)]
struct FunctionSignature {
    params: Vec<SemanticType>,
    result: SemanticType,
}

impl ScopeFrame {
    fn new() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }
}

pub struct TypeChecker {
    // Unified scope stack: each frame holds all per-variable data
    scopes: Vec<ScopeFrame>,
    pub linear_tracker: LinearTracker,
    pub errors: Vec<String>,
    pub in_unsafe: bool,
    allow_transfer_use: usize,
    current_return_type: Option<SemanticType>,
    functions: HashMap<String, FunctionSignature>,
    structs: HashMap<String, HashMap<String, SemanticType>>,
    enums: HashMap<String, EnumDecl>,

    // Static Under-Constrained Analyzer (@zk_safe) fields
    pub zk_safe_stack: Vec<bool>,
    pub zk_allow_unconstrained_stack: Vec<bool>,
    /// Set by `set_zk_target` when compiling to R1CS. See that method.
    zk_target: bool,

    /// What was checked, proved or assumed, for `ydb verify`. See
    /// `guarantees.rs`.
    pub guarantees: Guarantees,
    /// The function or kernel being checked, spelled as
    /// `debug_info::item_names` spells it.
    current_item: String,
    /// The assumptions a range can rest on, indexed by bit of
    /// `Interval::trust`. Assumption 0 is CUDA's launch limits.
    assumptions: Vec<Assumption>,
    /// The assumptions of every range an SMT query was given as a fact.
    smt_trust: std::cell::Cell<u64>,
    /// Why the invariant just checked was NOT verified, when
    /// `Y_ALLOW_UNVERIFIED_INVARIANTS` let the program through anyway.
    unverified: Option<String>,
    /// Every array index seen, keyed by position.
    index_sites: std::collections::BTreeMap<(usize, usize), IndexSite>,
}

fn reset_thread_locals() {
    SAFE_INDICES.with(|s| s.borrow_mut().clear());
    INDEX_ARRAY_SIZES.with(|s| s.borrow_mut().clear());
    INDEX_SWIZZLES.with(|s| s.borrow_mut().clear());
}

impl TypeChecker {
    pub fn new() -> Self {
        reset_thread_locals();
        Self {
            scopes: vec![ScopeFrame::new()],
            linear_tracker: LinearTracker::new(),
            errors: Vec::new(),
            in_unsafe: false,
            allow_transfer_use: 0,
            current_return_type: None,
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            zk_safe_stack: vec![false],
            zk_allow_unconstrained_stack: vec![false],
            zk_target: false,
            guarantees: Guarantees::default(),
            current_item: String::new(),
            assumptions: vec![Assumption {
                item: String::new(),
                line: 0,
                what: "CUDA's launch limits (threadIdx.x and .y < 1024, .z < 64, blockIdx.x < 2^31 - 1, \
blockIdx.y and .z < 65535)"
                    .to_string(),
            }],
            smt_trust: std::cell::Cell::new(0),
            unverified: None,
            index_sites: std::collections::BTreeMap::new(),
        }
    }

    /// A new assumption, at `line` of the item being checked; its bit.
    fn new_assumption(&mut self, line: usize, what: String) -> u64 {
        let id = self.assumptions.len();
        self.assumptions.push(Assumption { item: self.current_item.clone(), line, what });
        1u64 << id.min(63)
    }

    /// The assumptions `trust` names.
    fn assumptions_of(&self, trust: u64) -> Vec<Assumption> {
        self.assumptions
            .iter()
            .enumerate()
            .filter(|(id, _)| trust & (1u64 << (*id).min(63)) != 0)
            .map(|(_, a)| a.clone())
            .collect()
    }

    /// Record a fact about the item being checked.
    #[allow(clippy::too_many_arguments)]
    fn fact(
        &mut self,
        kind: &'static str,
        status: Status,
        span: &Span,
        end_line: usize,
        what: String,
        detail: String,
        trust: u64,
    ) {
        let rests_on = self.assumptions_of(trust);
        self.guarantees.facts.push(Fact {
            item: self.current_item.clone(),
            line: span.line,
            col: span.col,
            end_line: end_line.max(span.line),
            kind,
            status,
            what,
            detail,
            rests_on,
        });
    }

    /// Start checking an item: what it spans, and whether strict mode holds.
    fn enter_item(&mut self, name: String, kind: &'static str, display: String, span: &Span, body: &Block, strict: bool) {
        self.current_item = name.clone();
        let end_line = crate::ast::last_line(body).max(span.line);
        self.guarantees.items.push(ItemRange { name, kind, line: span.line, end_line });
        if strict {
            self.fact("safe", Status::Checked, span, end_line, display, STRICT_DETAIL.to_string(), 0);
        } else {
            self.fact(
                "safe",
                Status::NotChecked,
                span,
                end_line,
                display,
                "@unsafe: strict mode is off here - an array index the compiler cannot prove is \
checked when the program runs instead, a loop needs no @invariant and one it has is not \
verified, and a raw pointer may be dereferenced"
                    .to_string(),
                0,
            );
        }
    }

    /// The fact for an invariant that was just checked: proved, or let
    /// through unverified. A refuted one has already refused the program.
    fn invariant_fact(&mut self, inv: &Expr, span: &Span, end_line: usize, errors_before: usize) {
        if self.errors.len() > errors_before {
            return;
        }
        let what = format!("@invariant({})", render(inv));
        match self.unverified.take() {
            Some(why) => self.fact(
                "invariant",
                Status::Unverified,
                span,
                end_line,
                what,
                format!("NOT verified, because Y_ALLOW_UNVERIFIED_INVARIANTS let the program through: {}", why),
                0,
            ),
            None => {
                let trust = self.smt_trust.get();
                self.fact(
                    "invariant",
                    Status::Proved,
                    span,
                    end_line,
                    what,
                    "z3 proved it holds when the loop is entered and that every iteration preserves it"
                        .to_string(),
                    trust,
                );
            }
        }
    }

    /// Classify every index seen, once every visit to it is done: proved in
    /// bounds, checked at run time, or not checked at all. This is the rule
    /// all three backends follow (`SAFE_INDICES`, `INDEX_ARRAY_SIZES`).
    fn finish_index_facts(&mut self) {
        let sites = std::mem::take(&mut self.index_sites);
        for ((line, col), site) in sites {
            self.current_item = site.item.clone();
            let span = Span { line, col };
            let safe = SAFE_INDICES.with(|set| set.borrow().contains(&(line, col)));
            let size = INDEX_ARRAY_SIZES.with(|map| map.borrow().get(&(line, col)).cloned());
            if safe {
                let (detail, trust) = match (site.proving, size.or(site.size)) {
                    (Some(iv), Some(n)) => (
                        format!(
                            "in bounds: the index lies in [{}, {}] and there are {} elements, so no \
run-time check is emitted",
                            iv.min, iv.max, n
                        ),
                        iv.trust,
                    ),
                    _ => ("in bounds, so no run-time check is emitted".to_string(), 0),
                };
                self.fact("index", Status::Proved, &span, line, site.what, detail, trust);
            } else if let Some(n) = size {
                self.fact(
                    "index",
                    Status::RunTime,
                    &span,
                    line,
                    site.what,
                    format!(
                        "not proved (the code is @unsafe): it is checked against the {} elements when \
the program runs, and an index outside them stops the program",
                        n
                    ),
                    0,
                );
            } else {
                self.fact(
                    "index",
                    Status::NotChecked,
                    &span,
                    line,
                    site.what,
                    format!(
                        "not checked at compile time or at run time: {}, so the compiler does \
not know its length",
                        site.base
                    ),
                    0,
                );
            }
        }
    }

    /// Note a visit to an index site.
    fn note_index(&mut self, expr: &Expr, span: &Span, size: Option<usize>, base: &SemanticType, proving: Option<Interval>) {
        let item = self.current_item.clone();
        let entry = self.index_sites.entry((span.line, span.col)).or_insert_with(|| IndexSite {
            item,
            what: render(expr),
            size,
            // `GlobalMemory<T>` resolves to `Unknown` here, so the base is
            // named by its expression and its type is said only when known.
            base: match (expr, base) {
                (Expr::Index { base: b, .. }, SemanticType::Unknown) => {
                    format!("`{}` is not a fixed-size array", render(b))
                }
                (Expr::Index { base: b, .. }, t) => {
                    format!("`{}` is `{}`, not a fixed-size array", render(b), Self::semantic_type_name(t))
                }
                _ => "the base is not a fixed-size array".to_string(),
            },
            proving: None,
        });
        if let (Some(iv), None) = (proving, entry.proving) {
            entry.proving = Some(iv);
        } else if let (Some(iv), Some(old)) = (proving, entry.proving) {
            entry.proving = Some(Interval {
                min: old.min.min(iv.min),
                max: old.max.max(iv.max),
                trust: old.trust | iv.trust,
            });
        }
    }

    /// Whether the R1CS backend is the compilation target.
    ///
    /// Only `error[Z0010]` depends on this, and it must: a `while` loop with no
    /// static bound genuinely cannot be lowered to a fixed constraint system,
    /// but it is ordinary code for every other backend. The check used to fire
    /// unconditionally, so **every** un-annotated `while` in the language was
    /// rejected with a message naming a mode that was not active - including in
    /// `tests/hello.ysu`, the first example in the README.
    pub fn set_zk_target(&mut self, on: bool) {
        self.zk_target = on;
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(ScopeFrame::new());
        self.linear_tracker.push_scope();
    }

    pub fn pop_scope(&mut self) {
        self.check_scope_unconstrained_signals();
        self.linear_tracker.pop_scope();
        self.scopes.pop();
    }

    pub fn is_zk_safe_active(&self) -> bool {
        self.zk_safe_stack.iter().rev().copied().any(|b| b)
    }

    pub fn is_zk_allow_unconstrained_active(&self) -> bool {
        self.zk_allow_unconstrained_stack.iter().rev().copied().any(|b| b)
    }

    pub fn set_signal_constraint(&mut self, name: String, state: ConstraintState, span: Span) {
        if let Some(frame) = self.scopes.last_mut() {
            if let Some(entry) = frame.symbols.get_mut(&name) {
                entry.constraint_info = Some(SignalConstraintInfo {
                    name,
                    state,
                    declared_span: span,
                });
            } else {
                frame.symbols.insert(
                    name.clone(),
                    SymbolEntry {
                        ty: SemanticType::Unknown,
                        interval: None,
                        is_explicitly_bounded: false,
                        constraint_info: Some(SignalConstraintInfo {
                            name,
                            state,
                            declared_span: span,
                        }),
                    },
                );
            }
        }
    }

    pub fn lookup_signal_constraint(&self, name: &str) -> Option<&SignalConstraintInfo> {
        for frame in self.scopes.iter().rev() {
            if let Some(entry) = frame.symbols.get(name) {
                if let Some(ref info) = entry.constraint_info {
                    return Some(info);
                }
            }
        }
        None
    }

    pub fn update_signal_constraint_state(&mut self, name: &str, new_state: ConstraintState) {
        for frame in self.scopes.iter_mut().rev() {
            if let Some(entry) = frame.symbols.get_mut(name) {
                if let Some(ref mut info) = entry.constraint_info {
                    info.state = new_state;
                    return;
                }
            }
        }
    }

    pub fn eval_expr_constraint_state(&self, expr: &Expr) -> ConstraintState {
        match expr {
            Expr::Ident(name, _) => {
                if let Some(info) = self.lookup_signal_constraint(name) {
                    info.state.clone()
                } else {
                    ConstraintState::Constrained
                }
            }
            Expr::IntLit(..) | Expr::FloatLit(..) | Expr::StringLit(..) | Expr::CharLit(..) => {
                ConstraintState::Constrained
            }
            Expr::BinaryOp { left, op, right, .. } => {
                if matches!(op, BinaryOp::Eq) {
                    return ConstraintState::Constrained;
                }
                let s_left = self.eval_expr_constraint_state(left);
                let s_right = self.eval_expr_constraint_state(right);

                match (s_left, s_right) {
                    (
                        ConstraintState::TaintedUnconstrained { origins: o1, reasons: r1 },
                        ConstraintState::TaintedUnconstrained { origins: o2, reasons: r2 },
                    ) => {
                        let mut merged_origins = o1;
                        for span in o2 {
                            if !merged_origins.contains(&span) {
                                merged_origins.push(span);
                            }
                        }
                        let mut merged_reasons = r1;
                        for r in r2 {
                            if !merged_reasons.contains(&r) {
                                merged_reasons.push(r);
                            }
                        }
                        ConstraintState::TaintedUnconstrained {
                            origins: merged_origins,
                            reasons: merged_reasons,
                        }
                    }
                    (ConstraintState::TaintedUnconstrained { origins, reasons }, _)
                    | (_, ConstraintState::TaintedUnconstrained { origins, reasons }) => {
                        ConstraintState::TaintedUnconstrained { origins, reasons }
                    }
                    (
                        ConstraintState::DeferredObligation { origins: o1, reasons: r1, override_span },
                        ConstraintState::DeferredObligation { origins: o2, reasons: r2, .. },
                    ) => {
                        let mut merged_origins = o1;
                        for span in o2 {
                            if !merged_origins.contains(&span) {
                                merged_origins.push(span);
                            }
                        }
                        let mut merged_reasons = r1;
                        for r in r2 {
                            if !merged_reasons.contains(&r) {
                                merged_reasons.push(r);
                            }
                        }
                        ConstraintState::DeferredObligation {
                            origins: merged_origins,
                            reasons: merged_reasons,
                            override_span,
                        }
                    }
                    (ConstraintState::DeferredObligation { origins, reasons, override_span }, _)
                    | (_, ConstraintState::DeferredObligation { origins, reasons, override_span }) => {
                        ConstraintState::DeferredObligation { origins, reasons, override_span }
                    }
                    (ConstraintState::Verified { origins, verified_span }, _)
                    | (_, ConstraintState::Verified { origins, verified_span }) => {
                        ConstraintState::Verified { origins, verified_span }
                    }
                    _ => ConstraintState::Constrained,
                }
            }
            Expr::UnaryOp { operand, .. } => {
                self.eval_expr_constraint_state(operand)
            }
            Expr::Call { args, .. } => {
                for arg in args {
                    let st = self.eval_expr_constraint_state(arg);
                    if matches!(st, ConstraintState::TaintedUnconstrained { .. } | ConstraintState::DeferredObligation { .. }) {
                        return st;
                    }
                }
                ConstraintState::Constrained
            }
            Expr::Index { base, index, .. } => {
                let sb = self.eval_expr_constraint_state(base);
                if matches!(sb, ConstraintState::TaintedUnconstrained { .. } | ConstraintState::DeferredObligation { .. }) {
                    return sb;
                }
                let si = self.eval_expr_constraint_state(index);
                if matches!(si, ConstraintState::TaintedUnconstrained { .. } | ConstraintState::DeferredObligation { .. }) {
                    return si;
                }
                ConstraintState::Constrained
            }
            _ => ConstraintState::Constrained,
        }
    }

    pub fn check_verification_transition(&mut self, left: &Expr, right: &Expr, eq_span: &Span) {
        let left_state = self.eval_expr_constraint_state(left);
        let right_state = self.eval_expr_constraint_state(right);

        if let Expr::Ident(name_l, _) = left {
            if let ConstraintState::TaintedUnconstrained { ref origins, .. }
                | ConstraintState::DeferredObligation { ref origins, .. } = left_state
            {
                if matches!(right_state, ConstraintState::Constrained | ConstraintState::Verified { .. }) {
                    self.update_signal_constraint_state(
                        name_l,
                        ConstraintState::Verified {
                            origins: origins.clone(),
                            verified_span: eq_span.clone(),
                        },
                    );
                }
            }
        }
        if let Expr::Ident(name_r, _) = right {
            if let ConstraintState::TaintedUnconstrained { ref origins, .. }
                | ConstraintState::DeferredObligation { ref origins, .. } = right_state
            {
                if matches!(left_state, ConstraintState::Constrained | ConstraintState::Verified { .. }) {
                    self.update_signal_constraint_state(
                        name_r,
                        ConstraintState::Verified {
                            origins: origins.clone(),
                            verified_span: eq_span.clone(),
                        },
                    );
                }
            }
        }
    }

    fn check_scope_unconstrained_signals(&mut self) {
        let is_allow = self.is_zk_allow_unconstrained_active();
        let is_safe = self.is_zk_safe_active();
        let is_top_level = self.scopes.len() <= 2;

        if let Some(current_frame) = self.scopes.last_mut() {
            for (var_name, entry) in current_frame.symbols.iter_mut() {
                let info = match entry.constraint_info.as_mut() {
                    Some(info) => info,
                    None => continue,
                };

                if is_allow {
                    if let ConstraintState::TaintedUnconstrained { origins, reasons } = &info.state {
                        info.state = ConstraintState::DeferredObligation {
                            origins: origins.clone(),
                            reasons: reasons.clone(),
                            override_span: info.declared_span.clone(),
                        };
                        continue;
                    }
                }

                if is_safe {
                    match &info.state {
                        ConstraintState::TaintedUnconstrained { origins, .. } => {
                            let escape_span = &info.declared_span;
                            let origin_span = origins.first().unwrap_or(escape_span);
                            let err_msg = format!(
                                "error[Z0042]: under-constrained signal `{}` detected in @zk_safe context\n  --> line {}, col {}: signal escapes scope unconstrained\n  |\nnote: signal originated from @hint block here\n  --> line {}, col {}: unconstrained witness defined here\n  |\nhelp: add a constraint assertion (e.g., assert({} == expected)) to verify the witness.",
                                var_name, escape_span.line, escape_span.col, origin_span.line, origin_span.col, var_name
                            );
                            self.errors.push(err_msg);
                        }
                        ConstraintState::DeferredObligation { origins, override_span, .. } if is_top_level => {
                            let escape_span = &info.declared_span;
                            let origin_span = origins.first().unwrap_or(escape_span);
                            let err_msg = format!(
                                "error[Z0042]: deferred unconstrained signal `{}` allowed via @zk_allow_unconstrained escaped top-level program boundary unverified\n  --> line {}, col {}: signal reaches circuit output unconstrained\n  |\nnote: deferred override applied here\n  --> line {}, col {}: @zk_allow_unconstrained override\n  |\nnote: signal originated from @hint block here\n  --> line {}, col {}: unconstrained witness defined here\n  |\nhelp: add a constraint assertion to verify the deferred witness.",
                                var_name, escape_span.line, escape_span.col, override_span.line, override_span.col, origin_span.line, origin_span.col
                            );
                            self.errors.push(err_msg);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// The interval each of `names` holds right now.
    ///
    /// Used to preserve the pre-loop state across the invalidation a loop
    /// performs, so the initiation obligation can be stated about it. Names
    /// with no known interval are simply absent, which is the same as before.
    fn snapshot_intervals(
        &self,
        names: &std::collections::HashSet<String>,
    ) -> HashMap<String, Interval> {
        names
            .iter()
            .filter_map(|n| self.lookup_interval(n).map(|i| (n.clone(), *i)))
            .collect()
    }

    fn interval_state(&self) -> Vec<HashMap<String, Option<Interval>>> {
        self.scopes.iter().map(|frame| frame.symbols.iter()
            .map(|(name, entry)| (name.clone(), entry.interval)).collect()).collect()
    }

    fn restore_interval_state(&mut self, state: &[HashMap<String, Option<Interval>>]) {
        for (frame, saved) in self.scopes.iter_mut().zip(state) {
            for (name, entry) in &mut frame.symbols {
                entry.interval = saved.get(name).copied().flatten();
            }
        }
    }

    /// Retain a fact only if every incoming path establishes it. Scope indices
    /// distinguish a shadowing local from the outer binding it must not alter.
    fn join_interval_state(&mut self, other: &[HashMap<String, Option<Interval>>]) {
        for (frame, saved) in self.scopes.iter_mut().zip(other) {
            for (name, entry) in &mut frame.symbols {
                entry.interval = match (entry.interval, saved.get(name).copied().flatten()) {
                    (Some(a), Some(b)) => Some(Interval {
                        min: a.min.min(b.min),
                        max: a.max.max(b.max),
                        trust: a.trust | b.trust,
                    }),
                    _ => None,
                };
            }
        }
    }

    fn update_assignment_facts(&mut self, name: &str, value: &Expr, span: &Span) {
        let val_state = self.eval_expr_constraint_state(value);
        self.update_signal_constraint_state(name, val_state);
        if matches!(self.lookup_var(name), Some(SemanticType::Primitive(p))
            if p.starts_with('F') || p.starts_with('f') || p.starts_with('Q'))
        {
            // This domain proves integer indices. A float/fixed-point data
            // bound also guides the backend's accumulator selection, but it
            // cannot be propagated as an integer interval through arithmetic.
            self.update_interval(name, None);
            return;
        }
        let val_interval = self.eval_interval(value);
        if self.is_explicitly_bounded(name) {
            if let Some(target_interval) = self.lookup_interval(name).copied() {
                match val_interval {
                    Some(v) if v.min < target_interval.min || v.max > target_interval.max => {
                        self.errors.push(format!(
                            "Line {}: [Strict Safety] Bounds Violation: assigned value range [{}, {}] exceeds declared bounds [{}, {}] of `{}`.",
                            span.line, v.min, v.max, target_interval.min, target_interval.max, name
                        ));
                        self.update_interval(name, None);
                    }
                    None => {
                        if !self.in_unsafe {
                            self.errors.push(format!(
                                "Line {}: [Strict Safety] Bounds Violation: assigning an unconstrained value to bounded variable `{}`.", span.line, name
                            ));
                        }
                        // Even an unsafe assignment must not leave a proof
                        // behind that a later safe block could trust.
                        self.update_interval(name, None);
                    }
                    Some(v) => self.add_trust(name, v.trust),
                }
            }
        } else {
            self.update_interval(name, val_interval);
        }
    }

    /// `name`'s range now also rests on `trust`.
    fn add_trust(&mut self, name: &str, trust: u64) {
        if let Some(idx) = self.find_var_scope_index(name) {
            if let Some(entry) = self.scopes[idx].symbols.get_mut(name) {
                if let Some(iv) = entry.interval.as_mut() {
                    iv.trust |= trust;
                }
            }
        }
    }

    fn insert_interval(&mut self, name: String, interval: Interval) {
        if let Some(frame) = self.scopes.last_mut() {
            if let Some(entry) = frame.symbols.get_mut(&name) {
                entry.interval = Some(interval);
            } else {
                frame.symbols.insert(name, SymbolEntry {
                    ty: SemanticType::Unknown,
                    interval: Some(interval),
                    is_explicitly_bounded: false,
                    constraint_info: None,
                });
            }
        }
    }

    fn find_var_scope_index(&self, name: &str) -> Option<usize> {
        for (idx, frame) in self.scopes.iter().enumerate().rev() {
            if frame.symbols.contains_key(name) {
                return Some(idx);
            }
        }
        None
    }

    fn is_explicitly_bounded(&self, name: &str) -> bool {
        if let Some(idx) = self.find_var_scope_index(name) {
            if let Some(frame) = self.scopes.get(idx) {
                if let Some(entry) = frame.symbols.get(name) {
                    return entry.is_explicitly_bounded;
                }
            }
        }
        false
    }

    fn mark_explicitly_bounded(&mut self, name: String) {
        if let Some(frame) = self.scopes.last_mut() {
            if let Some(entry) = frame.symbols.get_mut(&name) {
                entry.is_explicitly_bounded = true;
            }
        }
    }

    fn update_interval(&mut self, name: &str, interval: Option<Interval>) {
        let target_idx = self.find_var_scope_index(name).unwrap_or_else(|| {
            self.scopes.len().saturating_sub(1)
        });
        if let Some(frame) = self.scopes.get_mut(target_idx) {
            if let Some(entry) = frame.symbols.get_mut(name) {
                entry.interval = Self::interval_for_type(interval, &entry.ty);
            } else if let Some(inv) = interval {
                frame.symbols.insert(name.to_string(), SymbolEntry {
                    ty: SemanticType::Unknown,
                    interval: Some(inv),
                    is_explicitly_bounded: false,
                    constraint_info: None,
                });
            }
        }
    }

    fn interval_for_type(interval: Option<Interval>, ty: &SemanticType) -> Option<Interval> {
        let value = interval?;
        let SemanticType::Primitive(name) = ty else { return Some(value) };
        let (min, max) = match name.to_ascii_lowercase().as_str() {
            "i8" => (i8::MIN as i64, i8::MAX as i64),
            "i16" => (i16::MIN as i64, i16::MAX as i64),
            "i32" => (i32::MIN as i64, i32::MAX as i64),
            "u8" => (0, u8::MAX as i64),
            "u16" => (0, u16::MAX as i64),
            "u32" => (0, u32::MAX as i64),
            "u64" => (0, i64::MAX),
            _ => return Some(value),
        };
        // Arithmetic beyond the variable's width may wrap. The mathematical
        // interval then says nothing about the stored machine integer.
        (value.min >= min && value.max <= max).then_some(value)
    }

    fn lookup_interval(&self, name: &str) -> Option<&Interval> {
        if let Some(idx) = self.find_var_scope_index(name) {
            if let Some(frame) = self.scopes.get(idx) {
                if let Some(entry) = frame.symbols.get(name) {
                    return entry.interval.as_ref();
                }
            }
        }
        None
    }

    fn eval_interval(&self, expr: &Expr) -> Option<Interval> {
        let interval = self.eval_interval_unchecked(expr)?;
        if matches!(expr,
            Expr::BinaryOp { op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod, .. }
            | Expr::UnaryOp { op: UnaryOp::Neg, .. })
        {
            // Entry facts must describe machine execution too. Checking only
            // the stored variable misses overflow in a nested subexpression.
            let bits = self.smt_integer_width(expr).ok()?;
            if bits == 32 && Self::contains_unsigned_literal(expr) {
                // PTX keeps this result unsigned, so widening may zero-extend
                // where LLVM produces a negative signed 64-bit value.
                return None;
            }
            if matches!(expr, Expr::BinaryOp { op: BinaryOp::Div | BinaryOp::Mod, .. }) {
                self.expr_to_smt(expr, &HashMap::new(), &mut Vec::new()).ok()?;
            }
            let limit = 1i128 << (bits - 1);
            if (interval.min as i128) < -limit || (interval.max as i128) >= limit {
                return None;
            }
        }
        Some(interval)
    }

    fn eval_interval_unchecked(&self, expr: &Expr) -> Option<Interval> {
        match expr {
            Expr::IntLit(val, _) => Some(Interval { min: *val, max: *val, trust: 0 }),
            Expr::UnaryOp { op: UnaryOp::Neg, operand, .. } => {
                let value = self.eval_interval(operand)?;
                Some(Interval { min: value.max.checked_neg()?, max: value.min.checked_neg()?, trust: value.trust })
            }
            Expr::Ident(name, _) => self.lookup_interval(name).cloned(),
            // The GPU index intrinsics have ranges the HARDWARE guarantees, so
            // they are the one call shape this domain can evaluate. Without
            // them even `let i = thread_idx_x();` had no interval, so
            // `@invariant(i >= 0)` got no precondition. Composed launch
            // indices must also fit their stored integer width; otherwise
            // `interval_for_type` discards their mathematical interval.
            //
            // Everything asserted here makes an obligation EASIER, which is
            // the direction `CLAUDE.md`'s design rule warns about, so these
            // are CUDA launch-configuration limits and nothing more. Widening
            // a `max` is safe (it weakens what can be concluded); narrowing
            // one, or raising a `min`, would not be.
            Expr::Call { func, args, .. } if args.is_empty() => match &**func {
                Expr::Ident(name, _) => gpu_index_interval(name),
                _ => None,
            },
            Expr::BinaryOp { left, op, right, .. } if matches!(op, BinaryOp::BitAnd) => {
                // `x & c` with a NON-NEGATIVE constant mask lies in [0, c]
                // whatever `x` is: the result's set bits are a subset of `c`'s,
                // and `c >= 0` leaves the sign bit clear, so the result cannot
                // be negative and cannot exceed `c`.
                //
                // This is decided BEFORE both operands are required to be
                // bounded, which is the whole point: the masked ring-buffer
                // index (`tail & 1023` into a `[I64; 1024]`) has a completely
                // unbounded left operand, and demanding an interval for it
                // returns `None` before the operator is ever consulted. A
                // negative or non-constant mask falls through to `None`.
                let mask_of = |side: &Expr| -> Option<(i64, u64)> {
                    match self.eval_interval(side) {
                        Some(i) if i.min == i.max && i.min >= 0 => Some((i.min, i.trust)),
                        _ => None,
                    }
                };
                // The result rests on the MASK's range only: it holds whatever
                // the other operand is.
                let (mask, trust) = mask_of(right).or_else(|| mask_of(left))?;
                Some(Interval { min: 0, max: mask, trust })
            }
            Expr::BinaryOp { left, op, right, .. } => {
                let lhs = self.eval_interval(left)?;
                let rhs = self.eval_interval(right)?;
                match op {
                    BinaryOp::Add => Some(Interval {
                        min: lhs.min.checked_add(rhs.min)?,
                        max: lhs.max.checked_add(rhs.max)?,
                        trust: lhs.trust | rhs.trust,
                    }),
                    BinaryOp::Sub => Some(Interval {
                        min: lhs.min.checked_sub(rhs.max)?,
                        max: lhs.max.checked_sub(rhs.min)?,
                        trust: lhs.trust | rhs.trust,
                    }),
                    BinaryOp::Mul => {
                        let candidates = [
                            lhs.min.checked_mul(rhs.min)?,
                            lhs.min.checked_mul(rhs.max)?,
                            lhs.max.checked_mul(rhs.min)?,
                            lhs.max.checked_mul(rhs.max)?,
                        ];
                        Some(Interval {
                            min: *candidates.iter().min().unwrap(),
                            max: *candidates.iter().max().unwrap(),
                            trust: lhs.trust | rhs.trust,
                        })
                    }
                    BinaryOp::Div => {
                        if rhs.min <= 0 && rhs.max >= 0 {
                            None
                        } else {
                            let candidates = [
                                lhs.min.checked_div(rhs.min)?,
                                lhs.min.checked_div(rhs.max)?,
                                lhs.max.checked_div(rhs.min)?,
                                lhs.max.checked_div(rhs.max)?,
                            ];
                            Some(Interval {
                                min: *candidates.iter().min().unwrap(),
                                max: *candidates.iter().max().unwrap(),
                                trust: lhs.trust | rhs.trust,
                            })
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn insert_var(&mut self, name: String, ty: SemanticType) {
        if let Some(frame) = self.scopes.last_mut() {
            if let Some(entry) = frame.symbols.get_mut(&name) {
                entry.interval = Self::interval_for_type(entry.interval, &ty);
                entry.ty = ty;
            } else {
                frame.symbols.insert(name, SymbolEntry {
                    ty,
                    interval: None,
                    is_explicitly_bounded: false,
                    constraint_info: None,
                });
            }
        }
    }

    fn lookup_var(&self, name: &str) -> Option<&SemanticType> {
        for frame in self.scopes.iter().rev() {
            if let Some(entry) = frame.symbols.get(name) {
                return Some(&entry.ty);
            }
        }
        None
    }

    fn check_expr_allowing_transfer_use(&mut self, expr: &Expr) -> SemanticType {
        self.allow_transfer_use += 1;
        let ty = self.check_expr(expr);
        self.allow_transfer_use -= 1;
        ty
    }

    fn reject_transfer_escape(&mut self, ty: &SemanticType, span: &Span, context: &str) {
        if *ty == SemanticType::TransferObligation {
            self.errors.push(format!(
                "Line {}: Transfer obligations are linear and may only be consumed by `pipe.wait(...)`, not {}.",
                span.line, context
            ));
        }
    }

    fn root_ident(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Ident(name, _) => Some(name.clone()),
            Expr::Index { base, .. } => Self::root_ident(base),
            Expr::MemberAccess { base, .. } => Self::root_ident(base),
            _ => None,
        }
    }

    fn collect_assigned_vars_in_block(&self, block: &Block, vars: &mut std::collections::HashSet<String>) {
        for stmt in &block.stmts {
            self.collect_assigned_vars_in_stmt(stmt, vars);
        }
    }

    fn collect_assigned_vars_in_stmt(&self, stmt: &Stmt, vars: &mut std::collections::HashSet<String>) {
        match stmt {
            Stmt::Assign { target, .. } | Stmt::CompoundAssign { target, .. } => {
                if let Some(name) = Self::root_ident(target) {
                    vars.insert(name);
                }
            }
            Stmt::For { body, .. } => {
                self.collect_assigned_vars_in_block(body, vars);
            }
            Stmt::While { body, .. } => {
                self.collect_assigned_vars_in_block(body, vars);
            }
            Stmt::If { then_block, else_block, .. } => {
                self.collect_assigned_vars_in_block(then_block, vars);
                if let Some(eb) = else_block {
                    self.collect_assigned_vars_in_block(eb, vars);
                }
            }
            Stmt::Chisel(block, _) | Stmt::SafeBlock(block, _) | Stmt::GhostBlock(block, _) | Stmt::HintBlock { body: block, .. } => {
                self.collect_assigned_vars_in_block(block, vars);
            }
            Stmt::ClockDomainBlock { body, .. } => {
                self.collect_assigned_vars_in_block(body, vars);
            }
            Stmt::CompileTimeAssert { .. } => {}
            Stmt::Expr(expr) => {
                self.collect_assigned_vars_in_expr(expr, vars);
            }
            Stmt::Return(Some(expr), _) => {
                self.collect_assigned_vars_in_expr(expr, vars);
            }
            _ => {}
        }
    }

    fn collect_assigned_vars_in_expr(&self, expr: &Expr, vars: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::BlockExpr(block, _) => {
                self.collect_assigned_vars_in_block(block, vars);
            }
            Expr::BinaryOp { left, right, .. } => {
                self.collect_assigned_vars_in_expr(left, vars);
                self.collect_assigned_vars_in_expr(right, vars);
            }
            Expr::Call { func, args, .. } => {
                self.collect_assigned_vars_in_expr(func, vars);
                for arg in args {
                    self.collect_assigned_vars_in_expr(arg, vars);
                }
            }
            Expr::GenericCall { func, args, .. } => {
                self.collect_assigned_vars_in_expr(func, vars);
                for arg in args {
                    self.collect_assigned_vars_in_expr(arg, vars);
                }
            }
            Expr::Index { base, index, .. } => {
                self.collect_assigned_vars_in_expr(base, vars);
                self.collect_assigned_vars_in_expr(index, vars);
            }
            Expr::MemberAccess { base, .. } => {
                self.collect_assigned_vars_in_expr(base, vars);
            }
            Expr::UnaryOp { operand, .. } => {
                self.collect_assigned_vars_in_expr(operand, vars);
            }
            Expr::StructLit { fields, .. } => {
                for (_, f_expr) in fields {
                    self.collect_assigned_vars_in_expr(f_expr, vars);
                }
            }
            _ => {}
        }
    }

    fn transfer_destination_from_expr(expr: &Expr) -> Option<String> {
        if let Expr::Call { func, args, .. } = expr {
            if let Expr::Ident(fname, _) = &**func {
                if fname == "cp_async" && args.len() >= 2 {
                    return Self::root_ident(&args[1]);
                }
            }
        }
        None
    }

    fn require_destination_ready(&mut self, expr: &Expr, span: &Span) {
        if let Some(name) = Self::root_ident(expr) {
            self.linear_tracker
                .require_destination_ready(&name, span.clone());
        }
    }

    fn check_wait_call(&mut self, base: &Expr, args: &[Expr], span: &Span) -> SemanticType {
        let base_ty = self.check_expr(base);
        self.reject_transfer_escape(&base_ty, span, "as the receiver of a method call");

        if args.is_empty() {
            self.errors.push(format!(
                "Line {}: `pipe.wait(...)` requires at least one Transfer obligation.",
                span.line
            ));
            return SemanticType::Unknown;
        }

        for arg in args {
            let arg_ty = self.check_expr_allowing_transfer_use(arg);
            if arg_ty != SemanticType::TransferObligation {
                self.errors.push(format!(
                    "Line {}: `pipe.wait(...)` expects Transfer obligations as arguments.",
                    span.line
                ));
                continue;
            }

            if let Expr::Ident(var_name, _) = arg {
                if !self.linear_tracker.is_tracked_obligation(var_name) {
                    self.errors.push(format!(
                        "Line {}: `{}` is not a tracked Transfer obligation in this scope.",
                        span.line, var_name
                    ));
                    continue;
                }
                self.linear_tracker
                    .consume_obligation(var_name, span.clone());
            } else {
                self.errors.push(format!(
                    "Line {}: `pipe.wait(...)` requires named Transfer bindings so the obligation can be consumed exactly once.",
                    span.line
                ));
            }
        }

        SemanticType::Unknown
    }

    /// Is this primitive one of Y's numeric scalar types?
    ///
    /// `Q<i>.<f>` counts: a fixed-point accumulator is initialised from a
    /// float literal (`@ZeroDrift let acc: Q32.32 = 0.0;`) and leaving it out
    /// made that a type mismatch.
    /// A short, user-facing name for a `SemanticType`, for diagnostics.
    fn semantic_type_name(ty: &SemanticType) -> String {
        match ty {
            SemanticType::Primitive(p) => p.clone(),
            SemanticType::GlobalMemory(t) => format!("GlobalMemory<{}>", t),
            SemanticType::Array { element, size } => {
                format!("[{}; {}]", Self::semantic_type_name(element), size)
            }
            SemanticType::Reference { inner, mutable } => format!(
                "&{}{}",
                if *mutable { "mut " } else { "" },
                Self::semantic_type_name(inner)
            ),
            SemanticType::Vector(inner, _) => {
                format!("Vec<{}>", Self::semantic_type_name(inner))
            }
            SemanticType::Pipeline => "Pipeline".into(),
            SemanticType::TransferObligation => "TransferObligation".into(),
            other => format!("{:?}", other),
        }
    }

    fn is_numeric_primitive(name: &str) -> bool {
        if let Some(rest) = name.strip_prefix('Q') {
            if let Some((i, f)) = rest.split_once('.') {
                if !i.is_empty()
                    && !f.is_empty()
                    && i.bytes().all(|b| b.is_ascii_digit())
                    && f.bytes().all(|b| b.is_ascii_digit())
                {
                    return true;
                }
            }
        }
        matches!(
            name,
            "I8" | "I16" | "I32" | "I64"
                | "U8" | "U16" | "U32" | "U64"
                | "F16" | "F32" | "F64"
                | "i8" | "i16" | "i32" | "i64"
                | "u8" | "u16" | "u32" | "u64"
                | "f16" | "f32" | "f64"
        )
    }

    /// A branch condition must be a boolean.
    ///
    /// `if 1 { ... }` type-checked and reached the backends: `--emit-cpu`
    /// printed Rust with a literal `if 1 {` in it, which rustc rejects
    /// ("expected `bool`, found integer"). `tests/test_drift.ysu` is exactly
    /// this program.
    ///
    /// Unknown is not evidence of a boolean. Named functions and scalar
    /// operators now carry result types, so a condition must resolve to bool.
    fn require_bool_condition(&mut self, ty: &SemanticType, kw: &str, span: Span) {
        if !matches!(ty, SemanticType::Primitive(name) if name.eq_ignore_ascii_case("bool")) {
            self.errors.push(format!(
                "Line {}: `{}` condition has type {}, not a boolean. Y has no implicit truthiness; write a comparison such as `{} x != 0`.",
                span.line, kw, Self::semantic_type_name(ty), kw
            ));
        }
    }

    fn check_uniformity(&mut self, expr: &Expr) {
        // Uniformity analysis: fail if the expression relies on thread-local IDs
        let mut is_uniform = true;
        
        // Very basic prototype check: walk the expression and look for known thread-local variables
        // like threadIdx.x, blockDim.x, blockIdx.x, or memory loads that aren't broadcast.
        // For this bootstrap version, we will just check if any Ident contains "threadIdx".
        fn walk_expr(e: &Expr, is_u: &mut bool) {
            match e {
                Expr::Ident(name, _) => {
                    if name.contains("threadIdx") || name.contains("laneId") {
                        *is_u = false;
                    }
                }
                Expr::BinaryOp { left, right, .. } => {
                    walk_expr(left, is_u);
                    walk_expr(right, is_u);
                }
                Expr::UnaryOp { operand, .. } => {
                    walk_expr(operand, is_u);
                }
                Expr::Call { args, .. } => {
                    for arg in args {
                        walk_expr(arg, is_u);
                    }
                }
                Expr::MemberAccess { base, .. } => {
                    walk_expr(base, is_u);
                }
                Expr::Index { base, index, .. } => {
                    // Indexing into a potentially non-uniform array is divergent
                    // unless we prove the array contains uniform data. For now, mark unsafe indexing.
                    walk_expr(base, is_u);
                    walk_expr(index, is_u);
                }
                _ => {}
            }
        }
        
        walk_expr(expr, &mut is_uniform);
        
        if !is_uniform {
            self.errors.push(format!(
                "Line {}: Hardware Constraint Violation: Branch expression is not guaranteed to be uniform. Warp divergence detected.",
                expr.span().line
            ));
        }
    }

    // ── AST Traversal ───────────────────────────────────────

    pub fn check_program(&mut self, prog: &Program) {
        SAFE_INDICES.with(|set| {
            set.borrow_mut().clear();
        });
        INDEX_ARRAY_SIZES.with(|map| {
            map.borrow_mut().clear();
        });
        INDEX_SWIZZLES.with(|map| {
            map.borrow_mut().clear();
        });
        // Predeclare named types before resolving signatures, including types
        // whose declarations follow the functions that use them.
        fn named_types(tc: &mut TypeChecker, items: &[Item]) {
            for item in items {
                match item {
                    Item::Struct(s) => { tc.structs.insert(s.name.clone(), HashMap::new()); }
                    Item::Enum(e) => { tc.enums.insert(e.name.clone(), e.clone()); }
                    Item::Module(m) => named_types(tc, &m.items),
                    _ => {}
                }
            }
        }
        named_types(self, &prog.items);
        self.refuse_duplicate_definitions(&prog.items);
        // Collect function signatures first
        for item in &prog.items {
            self.collect_signatures_item(item);
        }

        for item in &prog.items {
            self.check_item(item);
        }
        self.finish_index_facts();
    }

    /// One namespace holds every top-level `fn` and `kernel`, every `impl`
    /// method (`Type_method`) and every enum constructor (`Enum_Variant`). A
    /// name defined twice in it was not refused: this checker kept the second
    /// signature, the LLVM module then failed inside clang with no reason
    /// given, `--emit-llvm` and `--emit-cpu` wrote output that does not
    /// compile, and only `--emit-native` said what was wrong.
    fn refuse_duplicate_definitions(&mut self, items: &[Item]) {
        let mut defs: Vec<(String, String, usize)> = Vec::new();
        for item in items {
            match item {
                Item::Func(f) => defs.push((f.name.clone(), format!("fn {}", f.name), f.span.line)),
                Item::Kernel(k) => defs.push((k.name.clone(), format!("kernel {}", k.name), k.span.line)),
                Item::Impl(imp) => {
                    for m in &imp.methods {
                        defs.push((
                            format!("{}_{}", imp.target_type, m.name),
                            format!("fn {}::{}", imp.target_type, m.name),
                            m.span.line,
                        ));
                    }
                }
                Item::Enum(e) => {
                    for v in &e.variants {
                        defs.push((
                            format!("{}_{}", e.name, v.name),
                            format!("{}::{}", e.name, v.name),
                            v.span.line,
                        ));
                    }
                }
                _ => {}
            }
        }
        let mut seen: HashMap<String, (String, usize)> = HashMap::new();
        for (key, shown, line) in defs {
            match seen.get(&key) {
                Some((first, first_line)) => self.errors.push(format!(
                    "Line {}: `{}` is defined twice (`{}` at line {}, and again at line {}); \
                     every call would reach only one of them.",
                    line, shown, first, first_line, line
                )),
                None => {
                    seen.insert(key, (shown, line));
                }
            }
        }
    }

    fn collect_signatures_item(&mut self, item: &Item) {
        match item {
            Item::Func(f) => {
                let mut params = Vec::new();
                for p in &f.params {
                    params.push(self.resolve_type(&p.ty));
                }
                let result = f.ret_ty.as_ref().map(|t| self.resolve_type(t)).unwrap_or(SemanticType::Void);
                self.functions.insert(f.name.clone(), FunctionSignature { params, result });
            }
            Item::Kernel(k) => {
                let params = k.params.iter().map(|p| self.resolve_type(&p.ty)).collect();
                self.functions.insert(k.name.clone(), FunctionSignature { params, result: SemanticType::Void });
            }
            Item::Enum(e) => {
                for variant in &e.variants {
                        let params = variant.fields.iter().flatten().map(|t| self.resolve_type(t)).collect();
                        self.functions.insert(format!("{}_{}", e.name, variant.name), FunctionSignature {
                            params, result: SemanticType::Primitive(e.name.clone()),
                        });
                }
            }
            Item::Impl(imp) => {
                for f in &imp.methods {
                    let mut params = Vec::new();
                    for p in &f.params {
                        params.push(self.resolve_type(&p.ty));
                    }
                    let result = f.ret_ty.as_ref().map(|t| self.resolve_type(t)).unwrap_or(SemanticType::Void);
                    self.functions.insert(format!("{}_{}", imp.target_type, f.name), FunctionSignature { params, result });
                }
            }
            Item::Const(c) => {
                let resolved = self.resolve_type(&c.ty);
                self.insert_var(c.name.clone(), resolved);
            }
            Item::Struct(s) => {
                let mut fields = HashMap::new();
                for f in &s.fields {
                    fields.insert(f.name.clone(), self.resolve_type(&f.ty));
                }
                self.structs.insert(s.name.clone(), fields);
            }
            Item::Module(m) => {
                for inner_item in &m.items {
                    self.collect_signatures_item(inner_item);
                }
            }
            _ => {}
        }
    }

    fn check_item(&mut self, item: &Item) {
        match item {
            Item::StaticAssert(a) => self.check_compile_time_assert(&a.condition, &a.message, &a.span),
            Item::Kernel(k) => {
                // A kernel has no `@unsafe`: it is always checked in strict mode.
                self.enter_item(k.name.clone(), "kernel", format!("kernel {}", k.name), &k.span, &k.body, true);
                self.check_kernel(k)
            }
            Item::Func(f) => {
                self.enter_item(f.name.clone(), "fn", format!("fn {}", f.name), &f.span, &f.body, f.is_safe);
                self.check_func(f)
            }
            Item::Impl(imp) => {
                for f in &imp.methods {
                    self.enter_item(
                        format!("{}_{}", imp.target_type, f.name),
                        "method",
                        format!("fn {}::{}", imp.target_type, f.name),
                        &f.span,
                        &f.body,
                        f.is_safe,
                    );
                    self.check_func(f);
                }
            }
            Item::Module(m) => {
                self.zk_safe_stack.push(m.is_zk_safe);
                self.zk_allow_unconstrained_stack.push(m.is_zk_allow_unconstrained);
                for inner_item in &m.items {
                    self.check_item(inner_item);
                }
                self.zk_allow_unconstrained_stack.pop();
                self.zk_safe_stack.pop();
            }
            _ => {}
        }
    }

    fn check_kernel(&mut self, kernel: &KernelDecl) {
        self.push_scope();

        // Register params
        for param in &kernel.params {
            let sty = self.resolve_type(&param.ty);
            if sty == SemanticType::TransferObligation {
                self.errors.push(format!(
                    "Line {}: Kernel parameters cannot have Transfer type. Transfer obligations must be created and discharged within the kernel body.",
                    param.span.line
                ));
            }
            self.insert_var(param.name.clone(), sty);
            self.set_signal_constraint(param.name.clone(), ConstraintState::Constrained, param.span.clone());
        }

        self.check_block(&kernel.body);

        self.verify_kernel_coherence(kernel);
        self.verify_tile_gemm_kernel(kernel);

        self.pop_scope();
    }

    /// Validates a kernel-level `@tile(M, N, K)` directive (see
    /// `KernelDecl::tile`'s doc comment) - promotes it from
    /// parseable-but-unchecked to a real, enforced precondition before
    /// `ptx_emitter` trusts it to dispatch to tile-aware Tensor Core GEMM
    /// codegen instead of the normal generic per-statement lowering. Runs
    /// regardless of target backend, like the rest of type_checker, but only
    /// has any effect on kernels that opt in by writing `@tile(...)` before
    /// `kernel` - kernels without it are completely untouched.
    fn verify_tile_gemm_kernel(&mut self, kernel: &KernelDecl) {
        let tile = match &kernel.tile {
            Some(t) => t,
            None => return,
        };

        fn as_positive_i64(e: &Expr) -> Option<i64> {
            match e {
                Expr::IntLit(v, _) if *v > 0 => Some(*v),
                _ => None,
            }
        }

        if as_positive_i64(&tile.block_m).is_none() || as_positive_i64(&tile.block_n).is_none() {
            self.errors.push(format!(
                "Line {}: Kernel-level @tile(M, N, K) on `{}` requires M and N to be positive integer literals (the compile-time GEMM problem size this kernel is specialized for).",
                tile.span.line, kernel.name
            ));
        }
        if tile.block_k.as_deref().and_then(as_positive_i64).is_none() {
            self.errors.push(format!(
                "Line {}: Kernel-level @tile(M, N, K) on `{}` requires K (the third argument) - unlike the loop-scoped use of @tile, K is not optional here, and must be a positive integer literal.",
                tile.span.line, kernel.name
            ));
        }

        fn is_global_memory_of(ty: &Type, elem: &str) -> bool {
            matches!(
                ty,
                Type::Generic { base, args, .. }
                    if base == "GlobalMemory"
                        && matches!(
                            args.as_slice(),
                            [GenericArg::Type(Type::Primitive(p, _))] if p == elem
                        )
            )
        }
        // A 5-parameter shape (A, B: GlobalMemory<F32>, scale_a, scale_b:
        // F32 scalar, C: GlobalMemory<F32>) is accepted separately from the
        // 3/4-param F16 shapes below: A/B are quantized to e4m3 on the fly
        // (fused - see ptx_emitter::emit_fp8_gemm_kernel's doc comment) via
        // mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32 (Ada/sm_89-compatible,
        // unlike the Hopper-only WGMMA path), scale_a/scale_b are the
        // per-tensor dequant scales (typically amax/448 - the caller's/
        // launcher's responsibility to compute, not this kernel's), applied
        // to the f32 accumulator in the epilogue. Checked positionally, told
        // apart from the F16 shapes purely by param count (5) - see
        // `ptx_emitter::tile_gemm_fp8_operands`'s doc comment, which this
        // must never disagree with about which shape a given kernel is.
        if kernel.params.len() == 5 {
            fn is_scalar_f32(ty: &Type) -> bool {
                matches!(ty, Type::Primitive(p, _) if p == "F32")
            }
            let expected = ["GlobalMemory<F32>", "GlobalMemory<F32>", "F32", "F32", "GlobalMemory<F32>"];
            let checks: [bool; 5] = [
                is_global_memory_of(&kernel.params[0].ty, "F32"),
                is_global_memory_of(&kernel.params[1].ty, "F32"),
                is_scalar_f32(&kernel.params[2].ty),
                is_scalar_f32(&kernel.params[3].ty),
                is_global_memory_of(&kernel.params[4].ty, "F32"),
            ];
            let bad_params: Vec<String> = kernel
                .params
                .iter()
                .zip(checks.iter())
                .enumerate()
                .filter(|(_, (_, ok))| !**ok)
                .map(|(i, (p, _))| format!("{} (expected {})", p.name, expected[i]))
                .collect();
            if !bad_params.is_empty() {
                self.errors.push(format!(
                    "Line {}: Kernel-level @tile(M, N, K) on `{}` with 5 parameters requires (A, B: GlobalMemory<F32>, scale_a, scale_b: F32, C: GlobalMemory<F32>) for a fused FP8 (e4m3) GEMM - A/B are quantized on the fly, scale_a/scale_b dequant the f32 accumulator. Mismatched param(s): {}.",
                    tile.span.line, kernel.name, bad_params.join(", ")
                ));
            }
            return;
        }

        // A, B are the f16 Tensor Core operands; C is the accumulator/output
        // in f32 (matching wmma's f16-in/f32-out contract - see
        // ptx_emitter::emit_tensor_core_gemm_kernel, which hardcodes 4
        // bytes/element and `wmma.store.d...f32` for C specifically). Two
        // 4-parameter shapes are also accepted, told apart purely by
        // param[2]'s element type (matching ptx_emitter's own dispatch
        // logic exactly, so type-checking and codegen never disagree about
        // which shape a given kernel is):
        // - (A, B, Bias, C): Bias is F32, sitting in the same "everything
        //   past the two F16 operands" bucket as C - see
        //   `ptx_emitter::tile_gemm_operands`'s doc comment for the fused
        //   GEMM+Bias+ReLU epilogue this shape dispatches to.
        // - (X, W_gate, W_up, Out): W_up is F16 (unlike Bias) since it's a
        //   second Tensor Core operand, not an epilogue addend - see
        //   `ptx_emitter::tile_gemm_swiglu_operands`'s doc comment for the
        //   fused Linear+SwiGLU epilogue this shape dispatches to.
        // - (A, B: GlobalMemory<I8>, C: GlobalMemory<I32>): the exact int8
        //   Tensor Core GEMM over `mma.sync.m16n8k32.s32.s8.s8.s32`. Told
        //   apart from the f16 shape by its element types alone, which is
        //   enough because no other accepted shape mentions I8.
        //
        //   **This list and `ptx_emitter`'s dispatch chain are two
        //   implementations of one rule** — the comment above says so, and it
        //   is the hazard `CLAUDE.md`'s design-rule table describes. A shape
        //   accepted here but not recognised there falls through to generic
        //   scalar lowering, which silently computes something else; a shape
        //   recognised there but rejected here is unreachable. Change both.
        if kernel.params.len() == 3
            && is_global_memory_of(&kernel.params[0].ty, "I8")
            && is_global_memory_of(&kernel.params[1].ty, "I8")
            && is_global_memory_of(&kernel.params[2].ty, "I32")
        {
            return;
        }

        // - (A, B: I8, Sa, Sb, Bias, C: F32): the same int8 mma with a fused
        //   scaled epilogue, `C = acc * Sa[row] * Sb[col] + Bias[col]`. Told
        //   apart from the plain int8 shape by its parameter COUNT, matching
        //   `ptx_emitter::tile_gemm_int8_scaled_operands`, which is tried
        //   first in the dispatch chain for the same reason.
        if kernel.params.len() == 6
            && is_global_memory_of(&kernel.params[0].ty, "I8")
            && is_global_memory_of(&kernel.params[1].ty, "I8")
            && (2..6).all(|i| is_global_memory_of(&kernel.params[i].ty, "F32"))
        {
            return;
        }

        let is_swiglu_shape = kernel.params.len() == 4 && is_global_memory_of(&kernel.params[2].ty, "F16");
        let expected_elem = |i: usize| if i < 2 || (is_swiglu_shape && i == 2) { "F16" } else { "F32" };
        let bad_params: Vec<String> = kernel
            .params
            .iter()
            .enumerate()
            .filter(|(i, p)| !is_global_memory_of(&p.ty, expected_elem(*i)))
            .map(|(i, p)| format!("{} (expected GlobalMemory<{}>)", p.name, expected_elem(i)))
            .collect();

        if (kernel.params.len() != 3 && kernel.params.len() != 4) || !bad_params.is_empty() {
            self.errors.push(format!(
                "Line {}: Kernel-level @tile(M, N, K) on `{}` accepts these shapes, and codegen binds them POSITIONALLY:\n\
                 \x20 3  (A, B: GlobalMemory<F16>, C: GlobalMemory<F32>)                                  plain GEMM\n\
                 \x20 4  (A, B: GlobalMemory<F16>, Bias, C: GlobalMemory<F32>)                            fused GEMM+Bias+ReLU\n\
                 \x20 4  (X, W_gate, W_up: GlobalMemory<F16>, Out: GlobalMemory<F32>)                     fused Linear+SwiGLU\n\
                 \x20 3  (A, B: GlobalMemory<I8>, C: GlobalMemory<I32>)                                   int8 Tensor Core GEMM\n\
                 \x20 5  (A, B: GlobalMemory<F32>, scale_a, scale_b: F32, C: GlobalMemory<F32>)           fused FP8 (e4m3) GEMM\n\
                 \x20 6  (A, B: GlobalMemory<I8>, Sa, Sb, Bias, C: GlobalMemory<F32>)                     int8 GEMM + scaled epilogue\n\
                 Found {} parameter(s){}.",
                tile.span.line,
                kernel.name,
                kernel.params.len(),
                if bad_params.is_empty() {
                    String::new()
                } else {
                    format!(", mismatched param(s): {}", bad_params.join(", "))
                }
            ));
        }
    }

    /// Refuses the two function attributes that reach `FuncDecl` and are read
    /// by nothing.
    ///
    /// **Both were measured byte-identically inert before this was written**:
    /// the same function with and without the attribute emits the same LLVM IR
    /// and the same `--emit-cpu` blob, and `grep` finds zero readers of
    /// `f.is_hdl_emit` / `f.is_ghost` outside `parser.rs` and `ast.rs`. That
    /// consumer count is the sharp predicate, and the obvious one is a NULL
    /// METRIC: diffing the emitted artifact reports `@safe`, `@unsafe`,
    /// `@zk_safe` and `@zk_allow_unconstrained` as inert too, because those are
    /// CHECKERS - they change what is refused, not what is emitted, so on a
    /// program with nothing to check they legitimately change nothing.
    ///
    /// This is the `@zk_target(scheme = "plonkish")` shape: a directive a user
    /// can select that changes nothing, with a clean compile and no sign the
    /// annotation was discarded. Refusing is the fix, not a stopgap.
    ///
    /// The two are refused for DIFFERENT reasons and the distinction is why
    /// `@ptx_emit` is deliberately left alone (see the census gate):
    ///
    /// * `@hdl_emit` names a backend that **does not exist**. There is no HDL
    ///   emitter in this compiler and nothing for it to lower to, ever.
    /// * `@ghost` is real at a different SYNTACTIC SITE. `@ghost { .. }` and
    ///   `@ghost let ..` are the documented forms and are lowered by six
    ///   backends; the function attribute parses and is dropped. That is
    ///   "a guard consulted at one site" read across syntax rather than across
    ///   call sites, and it is the worse direction, because a user who has used
    ///   the block form successfully will reasonably expect the function form
    ///   to work.
    ///
    /// Note what is NOT claimed: a `@ghost` BLOCK is emitted in full, not
    /// stripped. `tests/directives_are_consumed_or_refused.rs` runs one and
    /// measures 7 where a stripped block gives 0. Stripping it is a feature -
    /// it would have to refuse a ghost block that writes non-ghost state, which
    /// is exactly what that probe does - not a typo, so it is recorded rather
    /// than done.
    fn refuse_inert_attributes(&mut self, f: &FuncDecl) {
        if f.is_hdl_emit {
            self.errors.push(format!(
                "Line {}: `@hdl_emit` on `{}` is not implemented. Y has no HDL backend, so \
                 there is nothing for it to lower to.\n  \
                 hint: remove the attribute. Select a backend with a CLI flag - `--emit-ptx`, \
                 `--emit-llvm`, `--emit-cpu`, `--emit-native` or `--target=r1cs`.\n  \
                 note: this previously parsed, was stored on the function, was read by nothing, \
                 and exited 0 - the emitted artifact was byte-identical to the undecorated one.",
                f.span.line, f.name
            ));
        }
        if f.is_ghost {
            self.errors.push(format!(
                "Line {}: `@ghost` on a function (`{}`) is not implemented and was silently \
                 ignored.\n  \
                 hint: `@ghost` applies to a BLOCK (`@ghost {{ .. }}`) or a variable \
                 (`@ghost let ..`); both of those work. Delete the function instead, or move \
                 its body into a `@ghost` block at the call site.\n  \
                 note: a `@ghost` block is EMITTED, not stripped - it costs the cycles its \
                 statements cost.",
                f.span.line, f.name
            ));
        }
    }

    fn check_func(&mut self, f: &FuncDecl) {
        self.refuse_inert_attributes(f);
        self.zk_safe_stack.push(f.is_safe || f.is_zk_safe);
        self.zk_allow_unconstrained_stack.push(f.is_zk_allow_unconstrained);
        self.push_scope();

        let prev_unsafe = self.in_unsafe;
        if !f.is_safe {
            self.in_unsafe = true;
        }

        for param in &f.params {
            let sty = self.resolve_type(&param.ty);
            if sty == SemanticType::TransferObligation {
                self.errors.push(format!(
                    "Line {}: Function parameters cannot have Transfer type. Linear Transfer obligations cannot cross function boundaries in the bootstrap compiler.",
                    param.span.line
                ));
            }
            self.insert_var(param.name.clone(), sty);
            self.set_signal_constraint(param.name.clone(), ConstraintState::Constrained, param.span.clone());
        }

        let prev_ret_ty = self.current_return_type.clone();
        if let Some(ret_ty) = &f.ret_ty {
            let resolved = self.resolve_type(ret_ty);
            self.current_return_type = Some(resolved.clone());
            if resolved == SemanticType::TransferObligation {
                self.errors.push(format!(
                    "Line {}: Functions cannot return Transfer obligations. They must be consumed by `pipe.wait(...)` in the creating scope.",
                    f.span.line
                ));
            }
        } else {
            self.current_return_type = None;
        }

        self.check_block(&f.body);
        self.current_return_type = prev_ret_ty;

        self.in_unsafe = prev_unsafe;
        self.pop_scope();
        self.zk_allow_unconstrained_stack.pop();
        self.zk_safe_stack.pop();
    }

    fn check_block(&mut self, block: &Block) {
        // Linear obligations are scoped to the block they are defined in.
        // Wait, loop bodies require their own scope.
        self.push_scope();

        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }

        self.pop_scope();
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
            match stmt {
            Stmt::Let {
                name,
                ty,
                init,
                span,
                bounds,
                zero_drift,
                ..
            } => {
                if zero_drift.is_some() {
                    let range = match bounds {
                        Some(b) => format!("inside @bounds({}, {})", render(&b.min), render(&b.max)),
                        None => "inside the range of its declared type".to_string(),
                    };
                    let assumption = Assumption {
                        item: self.current_item.clone(),
                        line: span.line,
                        what: format!(
                            "the running sum of `{}` stays {}: nothing checks it, and past it the \
integer wraps",
                            name, range
                        ),
                    };
                    self.guarantees.facts.push(Fact {
                        item: self.current_item.clone(),
                        line: span.line,
                        col: span.col,
                        end_line: span.line,
                        kind: "drift",
                        status: Status::Checked,
                        what: format!("@ZeroDrift on `{}`", name),
                        detail: "every `+=` and `-=` on it is exact integer or fixed-point arithmetic, so \
the result does not depend on the order of the additions; each term is rounded to the \
representation the backend selects, and anything else is refused"
                            .to_string(),
                        rests_on: vec![assumption],
                    });
                }
                let mut inferred_type = SemanticType::Unknown;
                let mut explicit_resolved = None;

                if let Some(explicit_ty) = ty {
                    explicit_resolved = Some(self.resolve_type(explicit_ty));
                }

                if !self.in_unsafe && init.is_none() {
                    self.errors.push(format!(
                        "Line {}: [Strict Safety] Variables in safe blocks must be explicitly initialized.",
                        span.line
                    ));
                }

                if let Some(init_expr) = init {
                    inferred_type =
                        self.check_expr_with_expected(init_expr, explicit_resolved.as_ref());
                    if let Some(init_interval) = self.eval_interval(init_expr) {
                        self.insert_interval(name.clone(), init_interval);
                    }
                }

                if let Some(bounds_attr) = bounds {
                    let min_iv = self.eval_interval(&bounds_attr.min);
                    let max_iv = self.eval_interval(&bounds_attr.max);
                    if let (Some(min_iv), Some(max_iv)) = (min_iv, max_iv) {
                        let (mn, mx) = (min_iv.min, max_iv.max);
                        let init_interval = init.as_ref().and_then(|e| self.eval_interval(e));
                        if let Some(init_interval) = init_interval {
                            if (init_interval.min < mn || init_interval.max > mx) && !self.in_unsafe {
                                self.errors.push(format!(
                                    "Line {}: [Strict Safety] Bounds Violation: initialized value range [{}, {}] exceeds declared bounds [{}, {}] of `{}`.",
                                    span.line, init_interval.min, init_interval.max, mn, mx, name
                                ));
                            }
                        }
                        // CHECKED when the initializer's range is known and lies
                        // inside: the declared range then rests on whatever the
                        // initializer's did. Otherwise it is TAKEN ON TRUST, and
                        // every proof that uses it says so.
                        let what = format!("@bounds({}, {}) on `{}`", mn, mx, name);
                        let from_bounds = min_iv.trust | max_iv.trust;
                        let trust = match init_interval {
                            Some(iv) if iv.min >= mn && iv.max <= mx => {
                                self.fact(
                                    "bounds",
                                    Status::Checked,
                                    span,
                                    span.line,
                                    what,
                                    format!("checked: the initializer's range [{}, {}] lies inside it", iv.min, iv.max),
                                    iv.trust | from_bounds,
                                );
                                iv.trust | from_bounds
                            }
                            other => {
                                let why = match other {
                                    Some(iv) => format!(
                                        "TRUSTED, and contradicted: the initializer's range [{}, {}] exceeds it, \
which @unsafe lets through. Every proof using this range assumes it",
                                        iv.min, iv.max
                                    ),
                                    None => "TRUSTED: nothing bounds the initializer, so the compiler assumes the \
range without checking it. Every proof using this range assumes it"
                                        .to_string(),
                                };
                                let bit = self.new_assumption(span.line, what.clone());
                                self.fact("bounds", Status::Trusted, span, span.line, what, why, from_bounds);
                                bit | from_bounds
                            }
                        };
                        self.insert_interval(name.clone(), Interval { min: mn, max: mx, trust });
                        self.mark_explicitly_bounded(name.clone());
                    }
                }

                if let Some(resolved) = explicit_resolved {
                    if inferred_type != SemanticType::Unknown
                        && !self.types_are_compatible(&inferred_type, &resolved)
                        && inferred_type != SemanticType::TransferObligation
                    {
                        self.errors.push(format!(
                            "Line {}: Type mismatch in let assignment.",
                            span.line
                        ));
                    }
                    // The backend stores the annotated type after conversion.
                    // Keeping the initializer's type here can discard a valid
                    // widening or retain range facts after a narrowing.
                    // An annotation must never hide a linear obligation.
                    if inferred_type != SemanticType::TransferObligation {
                        inferred_type = resolved;
                    }
                }

                self.insert_var(name.clone(), inferred_type.clone());

                let init_state = if let Some(init_expr) = init {
                    self.eval_expr_constraint_state(init_expr)
                } else {
                    ConstraintState::Constrained
                };
                self.set_signal_constraint(name.clone(), init_state, span.clone());

                // If it's a transfer obligation (`cp_async`), track it linearly.
                if inferred_type == SemanticType::TransferObligation {
                    let destination = init
                        .as_ref()
                        .and_then(|expr| Self::transfer_destination_from_expr(expr));

                    if init.is_none() {
                        self.errors.push(format!(
                            "Line {}: Transfer obligations must be initialized when declared.",
                            span.line
                        ));
                    }

                    if init.is_some() && destination.is_none() {
                        self.errors.push(format!(
                            "Line {}: Transfer obligations must originate from `cp_async(...)` so the compiler can track their destination.",
                            span.line
                        ));
                    }

                    self.linear_tracker.register_obligation(
                        name.clone(),
                        span.clone(),
                        destination,
                    );
                }
            }
            Stmt::TypeAlias { name, ty, span } => {
                let mut resolved = self.resolve_type(ty);
                // If defining a new SmemLayout, run the Bank Conflict Prover!
                if let SemanticType::SharedMemoryTile {
                    rows,
                    cols,
                    swizzle,
                } = &mut resolved
                {
                    let mut prover_layout = ProverLayout {
                        rows: *rows,
                        cols: *cols,
                        swizzle: swizzle.clone(),
                        swizzle_mode: None,
                        bytes_per_element: 2, // Defaulting F16 for prototype logic
                    };

                    let need_autoswizzle = if swizzle.is_none() {
                        true
                    } else {
                        BankConflictProver::prove_ldmatrix_m16n8(&prover_layout).is_err()
                    };

                    if need_autoswizzle {
                        // Find a swizzle pattern that satisfies the proof!
                        let mut found_swizzle = None;
                        'search: for xor_bits in 1..=4 {
                            for base_shift in 0..=4 {
                                for offset in 0..=4 {
                                    let candidate = SwizzlePattern {
                                        xor_bits,
                                        base_shift,
                                        offset,
                                    };
                                    prover_layout.swizzle = Some(candidate.clone());
                                    if BankConflictProver::prove_ldmatrix_m16n8(&prover_layout).is_ok() {
                                        found_swizzle = Some(candidate);
                                        break 'search;
                                    }
                                }
                            }
                        }

                        if let Some(working_swizzle) = found_swizzle {
                            println!(
                                "    [Optimization] Line {}: Auto-swizzling SharedMemoryTile {}x{} to solve bank conflicts: Swizzle<XOR={}, base_shift={}, offset={}>",
                                span.line, rows, cols, working_swizzle.xor_bits, working_swizzle.base_shift, working_swizzle.offset
                            );
                            *swizzle = Some(working_swizzle);
                        } else {
                            println!(
                                "    [Warning] Line {}: Bank Conflict Prover could not find a swizzle pattern to solve conflicts for {}x{}.",
                                span.line, rows, cols
                            );
                        }
                    } else {
                        println!(
                            "    [Optimization] Line {}: SharedMemoryTile {}x{} has verified 0 bank conflicts.",
                            span.line, rows, cols
                        );
                    }
                }
                self.insert_var(name.clone(), resolved);
            }
            Stmt::For { loop_var, start, end, step, body, invariant, is_uniform_branch: _, span, .. } => {
                for expr in std::iter::once(start).chain(std::iter::once(end)).chain(step.iter()) {
                    let ty = self.check_expr(expr);
                    if ty != SemanticType::Unknown && !matches!(&ty, SemanticType::Primitive(p)
                        if matches!(p.to_ascii_lowercase().as_str(), "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64"))
                    {
                        self.errors.push(format!("Line {}: for-loop bounds and step must be integer.", expr.span().line));
                    }
                }
                self.push_scope();

                if !self.in_unsafe && invariant.is_none() {
                    self.errors.push(format!(
                        "Line {}: [Strict Safety] Loops in safe blocks require formal @invariants.",
                        span.line
                    ));
                }

                let start_iv = self.eval_interval(start);
                let end_iv = self.eval_interval(end);
                let range_trust = start_iv.map_or(0, |i| i.trust) | end_iv.map_or(0, |i| i.trust);
                let start_val = start_iv.map(|i| i.min);
                let end_val = end_iv.and_then(|i| i.max.checked_sub(1));
                let bounds_are_known = matches!((start_val, end_val), (Some(_), Some(_)));
                if let (Some(s_min), Some(e_max)) = (start_val, end_val) {
                    self.insert_interval(loop_var.clone(), Interval { min: s_min, max: e_max, trust: range_trust });
                } else {
                    // The loop bounds are not statically known, so the loop
                    // variable has NO provable range and must not be given one.
                    //
                    // This used to fabricate `Interval { min: 0, max: 999999 }`,
                    // which asserts two facts it has not proved. The `max` half
                    // was harmless in practice because 999999 trips the overflow
                    // check for any normal array - but the `min` half claimed the
                    // index is non-negative, and nothing had established that.
                    // `for i in n..3` over a 2,000,000-element array therefore
                    // compiled clean with `n` an unconstrained parameter: the
                    // fabricated max slipped under the array size and the
                    // fabricated min waved the negative check through.
                    //
                    // Removing the interval makes the index unprovable instead,
                    // which is the honest answer and is what
                    // `mark_explicitly_bounded` below must NOT override.
                    self.update_interval(&loop_var, None);
                }

                self.insert_var(loop_var.clone(), SemanticType::Primitive("I32".into()));
                if bounds_are_known {
                    self.mark_explicitly_bounded(loop_var.clone());
                }

                let mut assigned_vars = std::collections::HashSet::new();
                self.collect_assigned_vars_in_block(body, &mut assigned_vars);
                // Taken BEFORE the clearing below, because that is the state
                // the initiation obligation is about. See
                // `generate_smt_decls_and_preconditions_with`.
                let entry_intervals = self.snapshot_intervals(&assigned_vars);
                let shadowed = self.smt_shadowed_binding(&body.stmts);
                for var in &assigned_vars {
                    self.update_interval(var, None);
                }

                // A `pipe.wait` inside this body awaits once per iteration; the
                // tracker needs to know that to compare against where the
                // matching `cp_async` was created.
                self.linear_tracker.enter_loop();
                for s in &body.stmts {
                    self.check_stmt(s);
                }
                self.linear_tracker.exit_loop();

                // Body-exit values are not assumptions for either induction
                // obligation. Initiation uses the separately saved entry facts.
                let mut body_writes = std::collections::HashSet::new();
                Self::collect_assigned(&body.stmts, &mut body_writes);
                for name in &body_writes { self.update_interval(name, None); }

                let loop_end = crate::ast::last_line(body).max(span.line);
                // A shadowed binding is refused (or let through UNVERIFIED) by
                // `smt_unmodellable`, and `invariant_fact` records which: the
                // code is not @unsafe, so "not verified: @unsafe" would be false.
                if !self.in_unsafe {
                    if let Some(inv_expr) = invariant {
                        let errors_before = self.errors.len();
                        self.unverified = None;
                        self.smt_trust.set(0);
                        if let Some(name) = &shadowed {
                            self.smt_unmodellable(inv_expr.span().line, inv_expr,
                                &format!("loop-local `{name}` shadows an existing binding; lexical shadowing is not modelled"));
                        } else {
                            self.verify_for_loop_invariant(
                                loop_var, start, end, step, body, inv_expr, &entry_intervals, span,
                            );
                        }
                        self.invariant_fact(inv_expr, span, loop_end, errors_before);
                    }
                } else if let Some(inv_expr) = invariant {
                    self.fact(
                        "invariant",
                        Status::NotChecked,
                        span,
                        loop_end,
                        format!("@invariant({})", render(inv_expr)),
                        "not verified: the code is @unsafe".to_string(),
                        0,
                    );
                }

                for var in &assigned_vars {
                    self.update_interval(var, None);
                }

                self.pop_scope();
            }
            Stmt::Assign {
                target,
                value,
                span,
            } => {
                let t1 = self.check_expr(target);
                let t2 = self.check_expr_with_expected(value, Some(&t1));
                if t1 == SemanticType::TransferObligation {
                    self.errors.push(format!(
                        "Line {}: Transfer bindings cannot be reassigned. Create a new Transfer with `let` and consume it exactly once with `pipe.wait(...)`.",
                        span.line
                    ));
                }
                if t2 == SemanticType::TransferObligation {
                    self.errors.push(format!(
                        "Line {}: Transfer obligations cannot be assigned or moved into another location. Consume them with `pipe.wait(...)`.",
                        span.line
                    ));
                }
                if !self.types_are_compatible(&t1, &t2) && t1 != SemanticType::Unknown && t2 != SemanticType::Unknown {
                    self.errors.push(format!(
                        "Line {}: Invalid assignment, types do not match.",
                        span.line
                    ));
                }
                if let Expr::Ident(name, _) = target {
                    self.update_assignment_facts(name, value, span);
                } else if self.takes_reference(target) {
                    self.invalidate_aliased_intervals();
                }
            }
            Stmt::Expr(expr) => {
                let ty = self.check_expr(expr);
                if ty == SemanticType::TransferObligation {
                    self.errors.push(format!(
                        "Line {}: Transfer obligations must be bound to a name and later consumed by `pipe.wait(...)`; they cannot be dropped as expression statements.",
                        expr.span().line
                    ));
                }
            }
            Stmt::Return(val, span) => {
                if let Some(expr) = val {
                    let expected_ret_ty = self.current_return_type.clone();

                    // A function with no declared return type returns nothing,
                    // so `return <expr>` in one is a type error - and it was
                    // accepted. `fn f() { let x = 5; return x; }` compiled
                    // clean and made the LLVM backend emit
                    // `sext i32 %t to void` / `ret void %t`, neither of which
                    // is legal LLVM; the build then failed inside `clang`,
                    // pointing at a line number in generated IR instead of at
                    // the user's source. `--emit-llvm` wrote that IR and
                    // exited 0. `tests/test.ysu` is exactly this program.
                    if expected_ret_ty.is_none() {
                        self.errors.push(format!(
                            "Line {}: `return` with a value in a function that declares no return type. Add `-> <type>` to the signature, or drop the value.",
                            span.line
                        ));
                    }

                    let ret_ty = self.check_expr_with_expected(expr, expected_ret_ty.as_ref());
                    if let Some(expected) = expected_ret_ty {
                        self.check_type_match(&expected, &ret_ty, span, "return type");
                    }
                    if ret_ty == SemanticType::TransferObligation {
                        self.errors.push(format!(
                            "Line {}: Returning a Transfer obligation would leak a linear sync proof. Consume it with `pipe.wait(...)` before returning.",
                            span.line
                        ));
                    }
                } else if self.current_return_type.is_some() {
                    self.errors.push(format!("Line {}: bare return does not supply the declared return type.", span.line));
                }
            }
            Stmt::Chisel(block, _) => {
                // Chisel blocks are privileged — type-check their contents normally
                self.check_block(block);
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
                is_uniform_branch,
                ..
            } => {
                if *is_uniform_branch {
                    self.check_uniformity(condition);
                }
                let cond_ty = self.check_expr(condition);
                self.require_bool_condition(&cond_ty, "if", condition.span());
                // Both arms are conditional: a transfer awaited in either one is
                // not awaited on the paths that take the other.
                self.linear_tracker.enter_conditional();
                let entry = self.interval_state();
                self.check_block(then_block);
                let then_exit = self.interval_state();
                self.restore_interval_state(&entry);
                if let Some(eb) = else_block {
                    self.check_block(eb);
                }
                self.join_interval_state(&then_exit);
                self.linear_tracker.exit_conditional();
            }
            Stmt::While {
                condition, body, invariant, max_iterations, is_uniform_branch, span: while_span,
            } => {
                if self.zk_target && max_iterations.is_none() {
                    // A `while` with no static bound cannot be unrolled into a
                    // fixed constraint system. That is true of R1CS and of
                    // nothing else, so the check is gated on the target rather
                    // than applied to the whole language.
                    self.errors.push(format!(
                        "Line {}: error[Z0010]: dynamic 'while' loop prohibited in ZK circuit mode\n  hint: annotate loop with '@max_iterations(N)' where N is a compile-time constant integer",
                        condition.span().line
                    ));
                }
                if !self.in_unsafe && invariant.is_none() {
                    self.errors.push(format!(
                        "Line {}: [Strict Safety] While loops in safe blocks require formal @invariants.",
                        condition.span().line
                    ));
                }
                if *is_uniform_branch {
                    self.check_uniformity(condition);
                }

                let mut assigned_vars = std::collections::HashSet::new();
                self.collect_assigned_vars_in_block(body, &mut assigned_vars);
                let entry_intervals = self.snapshot_intervals(&assigned_vars);
                let shadowed = self.smt_shadowed_binding(&body.stmts);
                for var in &assigned_vars {
                    self.update_interval(var, None);
                }

                let cond_ty = self.check_expr(condition);
                // Same rule at every branching site. Wiring `if` alone and
                // leaving `while` is the shape this repo keeps finding:
                // a correct guard consulted at a subset of its sites.
                self.require_bool_condition(&cond_ty, "while", condition.span());
                // A while body is both conditional (it may run zero times) and a
                // loop (it may run many). Entering both is not redundant: the
                // zero-iteration case is what makes an await inside it unsound
                // even when the copy is also inside.
                self.linear_tracker.enter_loop();
                self.linear_tracker.enter_conditional();
                self.check_block(body);
                self.linear_tracker.exit_conditional();
                self.linear_tracker.exit_loop();

                // Body-exit values are not assumptions for either induction
                // obligation. Initiation uses the separately saved entry facts.
                let mut body_writes = std::collections::HashSet::new();
                Self::collect_assigned(&body.stmts, &mut body_writes);
                for name in &body_writes { self.update_interval(name, None); }

                let loop_end = crate::ast::last_line(body).max(while_span.line);
                // A shadowed binding is refused (or let through UNVERIFIED) by
                // `smt_unmodellable`, and `invariant_fact` records which: the
                // code is not @unsafe, so "not verified: @unsafe" would be false.
                if !self.in_unsafe {
                    if let Some(inv_expr) = invariant {
                        let errors_before = self.errors.len();
                        self.unverified = None;
                        self.smt_trust.set(0);
                        if let Some(name) = &shadowed {
                            self.smt_unmodellable(inv_expr.span().line, inv_expr,
                                &format!("loop-local `{name}` shadows an existing binding; lexical shadowing is not modelled"));
                        } else {
                            self.verify_while_loop_invariant(
                                condition, body, inv_expr, &entry_intervals, &condition.span(),
                            );
                        }
                        self.invariant_fact(inv_expr, while_span, loop_end, errors_before);
                    }
                } else if let Some(inv_expr) = invariant {
                    self.fact(
                        "invariant",
                        Status::NotChecked,
                        while_span,
                        loop_end,
                        format!("@invariant({})", render(inv_expr)),
                        "not verified: the code is @unsafe".to_string(),
                        0,
                    );
                }

                for var in &assigned_vars {
                    self.update_interval(var, None);
                }
            }
            Stmt::Break { .. } => {}
            Stmt::Match {
                scrutinee, arms, ..
            } => {
                // The scrutinee runs whatever the arms do, so it is evaluated
                // outside the conditional.
                let scrutinee_ty = self.check_expr(scrutinee);
                // A `match` IS a branch, and the linear tracker was never told.
                // `if n { pipe.wait(t); }` was rejected as an await on one path
                // out of two, and `match n { _ => pipe.wait(t) }` - the same
                // program - compiled clean, because only `Stmt::If`,
                // `Stmt::For` and `Stmt::While` raised the depth. Same shape as
                // the `takes_reference` gap: a guard is only as good as the
                // list of sites that consult it.
                //
                // A single irrefutable arm is not really conditional, and this
                // over-approximates it. That is the safe direction and it is
                // free here: nothing in the kernel corpus matches on anything.
                self.linear_tracker.enter_conditional();
                let entry = self.interval_state();
                let mut exits = Vec::new();
                for arm in arms {
                    self.restore_interval_state(&entry);
                    self.push_scope();
                    match &arm.pattern {
                        MatchPattern::Ident(name, _) => self.insert_var(name.clone(), scrutinee_ty.clone()),
                        MatchPattern::EnumVariant { path, variant, bindings, span } => {
                            let namespace = if path.is_empty() {
                                match &scrutinee_ty { SemanticType::Primitive(t) => t.as_str(), _ => "" }
                            } else { path.as_str() };
                            let signature = self.functions.get(&format!("{}_{}", namespace, variant)).cloned();
                            let valid_owner = matches!(&scrutinee_ty, SemanticType::Primitive(t) if t == namespace);
                            let valid_variant = self.enums.get(namespace).is_some_and(|e| e.variants.iter().any(|v| v.name == *variant));
                            if !valid_owner || !valid_variant {
                                self.errors.push(format!("Line {}: enum match pattern `{}::{}` does not name a variant of the scrutinee type.", span.line, namespace, variant));
                            }
                            let arity = signature.as_ref().map_or(0, |s| s.params.len());
                            if bindings.len() != arity {
                                self.errors.push(format!("Line {}: enum match pattern `{}::{}` expects {} binding(s), got {}.", span.line, namespace, variant, arity, bindings.len()));
                            }
                            let mut names = std::collections::HashSet::new();
                            for (i, binding) in bindings.iter().enumerate() {
                                if !names.insert(binding) {
                                    self.errors.push(format!("Line {}: duplicate enum match binding `{}`.", span.line, binding));
                                }
                                let ty = signature.as_ref().and_then(|s| s.params.get(i)).cloned().unwrap_or(SemanticType::Unknown);
                                self.insert_var(binding.clone(), ty);
                            }
                        }
                        _ => {}
                    }
                    let arm_ty = self.check_expr(&arm.body);
                    self.reject_transfer_escape(&arm_ty, &arm.span, "as a match arm result");
                    self.pop_scope();
                    exits.push(self.interval_state());
                }
                // Exhaustiveness is not proved here, so include the path on
                // which no arm matches as well as each arm's exit.
                self.restore_interval_state(&entry);
                for exit in exits {
                    self.join_interval_state(&exit);
                }
                self.linear_tracker.exit_conditional();
            }
            Stmt::CompoundAssign { target, op, value, span } => {
                let lhs = self.check_expr(target);
                let rhs = self.check_expr_with_expected(value, Some(&lhs));
                self.reject_transfer_escape(&lhs, &target.span(), "in compound assignment");
                self.reject_transfer_escape(&rhs, &value.span(), "in compound assignment");
                self.binary_result_type(op, &lhs, &rhs, span);
                if let Expr::Ident(name, _) = target {
                    let result = Expr::BinaryOp {
                        left: Box::new(target.clone()), op: op.clone(),
                        right: Box::new(value.clone()), span: span.clone(),
                    };
                    self.update_assignment_facts(name, &result, span);
                } else if self.takes_reference(target) {
                    self.invalidate_aliased_intervals();
                }
            }
            Stmt::SafeBlock(block, block_span) | Stmt::GhostBlock(block, block_span) => {
                let prev_unsafe = self.in_unsafe;
                if prev_unsafe {
                    let what = if matches!(stmt, Stmt::SafeBlock(..)) { "@safe { }" } else { "@ghost { }" };
                    let end = crate::ast::last_line(block).max(block_span.line);
                    self.fact("safe", Status::Checked, block_span, end, what.to_string(), STRICT_DETAIL.to_string(), 0);
                }
                self.in_unsafe = false;
                self.check_block(block);
                self.in_unsafe = prev_unsafe;
            }
            Stmt::HintBlock { outputs, body, span } => {
                self.check_block(body);
                for out_var in outputs {
                    self.set_signal_constraint(
                        out_var.clone(),
                        ConstraintState::TaintedUnconstrained {
                            origins: vec![span.clone()],
                            reasons: vec![UnconstrainedReason::HintOutput(out_var.clone())],
                        },
                        span.clone(),
                    );
                }
            }
            Stmt::ClockDomainBlock { body, span, .. } => {
                // Type-check the body within the clock domain scope
                println!(
                    "      \x1b[1;35m[CDC]\x1b[0m Line {}: @clock_domain block entered.",
                    span.line
                );
                self.check_block(body);
            }
            Stmt::CompileTimeAssert { condition, message, span } => {
                let msg = message.as_deref().unwrap_or("compile-time assertion");
                self.check_compile_time_assert(condition, msg, span);
            }
        }
    }

    fn check_compile_time_assert(&mut self, condition: &Expr, message: &str, span: &Span) {
        match eval_compile_time(condition) {
            Ok(CompileTimeValue::Boolean(true)) => println!(
                "      \x1b[1;36m[Verified]\x1b[0m Line {}: compile-time assertion \"{}\"",
                span.line, message
            ),
            Ok(CompileTimeValue::Boolean(false)) => self.errors.push(format!(
                "Line {}: compile-time assertion failed: {}", span.line, message
            )),
            result => {
                let reason = match result {
                    Err(reason) => reason,
                    _ => "condition must evaluate to a boolean",
                };
                self.errors.push(format!(
                    "Line {}: cannot verify compile-time assertion: {} ({})", span.line, message, reason
                ));
            }
        }
    }

    fn check_type_match(&mut self, expected: &SemanticType, actual: &SemanticType, span: &Span, context: &str) {
        if *expected != SemanticType::Unknown && *actual != SemanticType::Unknown
            && !self.types_are_compatible(expected, actual)
        {
            self.errors.push(format!("Line {}: {} mismatch: expected {}, got {}.",
                span.line, context, Self::semantic_type_name(expected), Self::semantic_type_name(actual)));
        }
    }

    fn known_expr_type(&self, expr: &Expr) -> Option<SemanticType> {
        match expr {
            Expr::Ident(name, _) => self.lookup_var(name).cloned(),
            Expr::UnaryOp { op: UnaryOp::Neg, operand, .. } => self.known_expr_type(operand),
            Expr::Call { func, .. } => {
                let name = match &**func {
                    Expr::Ident(name, _) => name.clone(),
                    Expr::Path { namespace, member, .. } => format!("{}_{}", namespace, member),
                    _ => return None,
                };
                self.functions.get(&name).map(|s| s.result.clone())
                    .or_else(|| crate::intrinsics::scalar_return_type(&name).map(|t| SemanticType::Primitive(t.into())))
            }
            _ => None,
        }
    }

    fn binary_result_type(&mut self, op: &BinaryOp, lhs: &SemanticType, rhs: &SemanticType, span: &Span) -> SemanticType {
        if matches!(op, BinaryOp::And | BinaryOp::Or) {
            self.require_bool_condition(lhs, "boolean operator", span.clone());
            self.require_bool_condition(rhs, "boolean operator", span.clone());
            return SemanticType::Primitive("bool".into());
        }
        self.check_type_match(lhs, rhs, span, "binary operands");
        if !matches!(op, BinaryOp::Eq | BinaryOp::NotEq) {
            for ty in [lhs, rhs] {
                if *ty != SemanticType::Unknown && !matches!(ty, SemanticType::Primitive(p) if Self::is_numeric_primitive(p)) {
                    self.errors.push(format!("Line {}: arithmetic/comparison operands must be numeric.", span.line));
                }
                if matches!(op, BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr | BinaryOp::Mod)
                    && matches!(ty, SemanticType::Primitive(p) if p.starts_with('F') || p.starts_with('f') || p.starts_with('Q'))
                {
                    self.errors.push(format!("Line {}: bitwise, shift and remainder operands must be integer.", span.line));
                }
            }
        }
        if matches!(op, BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge) {
            SemanticType::Primitive("bool".into())
        } else if *lhs == SemanticType::Unknown { rhs.clone() } else { lhs.clone() }
    }

    fn check_named_call(&mut self, name: &str, args: &[Expr], span: &Span) -> SemanticType {
        if self.lookup_var(name).is_some() {
            self.errors.push(format!("Line {}: `{}` is a variable, not callable.", span.line, name));
        }
        let signature = self.functions.get(name).cloned();
        if let Some(sig) = &signature {
            if args.len() != sig.params.len() {
                self.errors.push(format!("Line {}: function `{}` expects {} argument(s), got {}.",
                    span.line, name, sig.params.len(), args.len()));
            }
        }
        let typed_vec_get = name.strip_prefix("Vec_get_").filter(|t|
            Self::is_numeric_primitive(t) || self.structs.contains_key(*t) || self.enums.contains_key(*t));
        if signature.is_none() && !crate::intrinsics::is_known_function(name) && typed_vec_get.is_none() {
            self.errors.push(format!("Line {}: Unknown function `{}`.", span.line, name));
        }
        let mut arg_types = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            let expected = signature.as_ref().and_then(|s| s.params.get(i));
            let actual = self.check_expr_with_expected(arg, expected);
            self.reject_transfer_escape(&actual, &arg.span(), "as a function argument");
            if let Some(expected) = expected {
                self.check_type_match(expected, &actual, &arg.span(), &format!("function `{}` argument {}", name, i + 1));
            }
            arg_types.push(actual);
        }
        // Without an alias/effect model, a callee receiving a reference can
        // invalidate any caller range fact. Keeping an initializer's interval
        // after `bump(saved_reference)` can prove a false loop-entry invariant.
        if arg_types.iter().any(|ty| self.type_contains_reference(ty))
            || args.iter().any(|arg| self.takes_reference(arg))
        {
            self.invalidate_aliased_intervals();
        }
        if let Some(sig) = signature { return sig.result; }
        if let Some(t) = typed_vec_get { return SemanticType::Primitive(t.to_string()); }
        if let Some(t) = crate::intrinsics::scalar_return_type(name) {
            return if t == "void" { SemanticType::Void } else { SemanticType::Primitive(t.to_string()) };
        }
        if matches!(name, "load" | "block_ptr2d_load" | "block_ptr3d_load" | "GlobalMemory_load") {
            if let Some(SemanticType::GlobalMemory(t)) = arg_types.first() {
                return SemanticType::Primitive(t.clone());
            }
        }
        SemanticType::Unknown
    }

    fn check_expr(&mut self, expr: &Expr) -> SemanticType {
        self.check_expr_with_expected(expr, None)
    }

    fn check_expr_with_expected(
        &mut self,
        expr: &Expr,
        expected_type: Option<&SemanticType>,
    ) -> SemanticType {
        let span = expr.span();
        match expr {
            Expr::ZeroInit(span) => {
                if let Some(expected) = expected_type {
                    expected.clone()
                } else {
                    self.errors.push(format!(
                        "Line {}: Ambiguous zero-initializer: cannot infer struct type.",
                        span.line
                    ));
                    SemanticType::Unknown
                }
            }
            Expr::Ident(name, _) => {
                if let Some(ty) = self.lookup_var(name) {
                    let ty = ty.clone();
                    if ty == SemanticType::TransferObligation && self.allow_transfer_use == 0 {
                        self.errors.push(format!(
                            "Line {}: `{}` is a linear Transfer obligation and may only be used as an argument to `pipe.wait(...)`.",
                            span.line, name
                        ));
                    }
                    ty
                } else {
                    self.errors.push(format!("Line {}: Undefined variable `{}`.", span.line, name));
                    SemanticType::Unknown
                }
            }
            Expr::Call { func, args, .. } => {
                if let Expr::Ident(fname, _) = &**func {
                    if fname == "cp_async" {
                        for arg in args {
                            let arg_ty = self.check_expr(arg);
                            self.reject_transfer_escape(
                                &arg_ty,
                                &arg.span(),
                                "as an operand to `cp_async`",
                            );
                        }
                        // Creates an obligation
                        return SemanticType::TransferObligation;
                    }
                    if fname == "ldmatrix" || fname == "load" {
                        if let Some(arg) = args.first() {
                            self.require_destination_ready(arg, &span);
                        }
                    }
                    if fname == "mma_sync" {
                        self.check_mma_sync(args, &span);
                        // Returns 'D' fragment (Accumulator)
                        return SemanticType::Fragment {
                            op: "MMA_m16n8k16".into(),
                            role: "D".into(),
                            dtype: "F32".into(),
                        };
                    }
                }
                if let Expr::MemberAccess { base, member, .. } = &**func {
                    if member == "wait" {
                        return self.check_wait_call(base, args, &span);
                    }
                }
                if let Expr::Path {
                    namespace, member, ..
                } = &**func
                {
                    if namespace == "barrier" && member == "sync" {
                        self.linear_tracker.synchronize_barrier();
                        return SemanticType::Void;
                    }
                    if namespace == "File" && member == "read" {
                        for arg in args {
                            let arg_ty = self.check_expr(arg);
                            self.reject_transfer_escape(
                                &arg_ty,
                                &arg.span(),
                                "as an argument to `File::read`",
                            );
                        }
                        // Prototype read evaluation guarantees String return
                        return SemanticType::Primitive("String".into());
                    }
                    if namespace == "Vec" || namespace == "String" {
                        for arg in args {
                            let arg_ty = self.check_expr(arg);
                            self.reject_transfer_escape(
                                &arg_ty,
                                &arg.span(),
                                "as an argument to a dynamic allocation API",
                            );
                        }
                        if !self.in_unsafe {
                            self.errors.push(format!("Line {}: Dynamic memory operations like {}::{} are mapped to raw void* and require an @unsafe function context.", span.line, namespace, member));
                        }
                        return self.check_named_call(&format!("{}_{}", namespace, member), args, &span);
                    }
                }
                match &**func {
                    Expr::Ident(name, _) => return self.check_named_call(name, args, &span),
                    Expr::Path { namespace, member, .. } => {
                        return self.check_named_call(&format!("{}_{}", namespace, member), args, &span);
                    }
                    _ => {}
                }
                let func_ty = self.check_expr(func);
                self.reject_transfer_escape(&func_ty, &func.span(), "as a callable value");
                for arg in args { self.check_expr(arg); }
                if args.iter().any(|arg| self.takes_reference(arg)) {
                    self.invalidate_aliased_intervals();
                }
                SemanticType::Unknown
            }
            Expr::MemberAccess { base, member, .. } => {
                if let Expr::MemberAccess { base: data, member: variant, .. } = &**base {
                    if let Expr::MemberAccess { base: owner, member: data_name, .. } = &**data {
                        if data_name == "data" {
                            let owner_ty = self.check_expr(owner);
                            let owner_ty = match &owner_ty {
                                SemanticType::Reference { inner, .. } => &**inner,
                                other => other,
                            };
                            if let SemanticType::Primitive(name) = owner_ty {
                                if let Some(e) = self.enums.get(name).cloned() {
                                    let field = e.variants.iter().find(|v| v.name == *variant)
                                        .and_then(|v| v.fields.as_ref())
                                        .and_then(|fields| member.strip_prefix('_').and_then(|s| s.parse::<usize>().ok()).and_then(|i| fields.get(i)));
                                    if let Some(field) = field { return self.resolve_type(field); }
                                    self.errors.push(format!("Line {}: enum payload `{}::{}` has no field `{}`.", span.line, name, variant, member));
                                    return SemanticType::Unknown;
                                }
                            }
                        }
                    }
                }
                let base_ty = self.check_expr(base);
                if member == "wait" {
                    SemanticType::Unknown
                } else {
                    self.reject_transfer_escape(
                        &base_ty,
                        &base.span(),
                        "as the base of member access",
                    );
                    let owner = match &base_ty {
                        SemanticType::Reference { inner, .. } => &**inner,
                        other => other,
                    };
                    if let SemanticType::Primitive(name) = owner {
                        if self.enums.contains_key(name) && member == "tag" {
                            return SemanticType::Primitive("I32".into());
                        }
                        if let Some(fields) = self.structs.get(name) {
                            if let Some(ty) = fields.get(member) { return ty.clone(); }
                        }
                    }
                    SemanticType::Unknown
                }
            }
            Expr::GenericCall {
                func,
                generic_args,
                args,
                ..
            } => {
                if let Expr::Path {
                    namespace, member, ..
                } = &**func
                {
                    if namespace == "SharedMemory" && member == "alloc" {
                        for arg in args {
                            let arg_ty = self.check_expr(arg);
                            self.reject_transfer_escape(
                                &arg_ty,
                                &arg.span(),
                                "as an argument to `SharedMemory::alloc`",
                            );
                        }
                        if let Some(layout_ty) = generic_args.first() {
                            return self.resolve_type(layout_ty);
                        }
                        return SemanticType::Unknown;
                    }
                    if namespace == "Pipeline" && member == "init" {
                        for arg in args {
                            let arg_ty = self.check_expr(arg);
                            self.reject_transfer_escape(
                                &arg_ty,
                                &arg.span(),
                                "as an argument to `Pipeline::init`",
                            );
                        }
                        return SemanticType::Pipeline;
                    }
                }

                match &**func {
                    Expr::Ident(name, _) => self.check_named_call(name, args, &span),
                    Expr::Path { namespace, member, .. } => self.check_named_call(&format!("{}_{}", namespace, member), args, &span),
                    _ => {
                        let func_ty = self.check_expr(func);
                        self.reject_transfer_escape(&func_ty, &func.span(), "as a generic callable value");
                        for arg in args { self.check_expr(arg); }
                        if args.iter().any(|arg| self.takes_reference(arg)) {
                            self.invalidate_aliased_intervals();
                        }
                        SemanticType::Unknown
                    }
                }
            }
            Expr::StructLit { name, fields, .. } => {
                let struct_fields = self.structs.get(name).cloned();
                for (fname, expr) in fields {
                    let expected_ty = struct_fields.as_ref().and_then(|m| m.get(fname));
                    let field_ty = self.check_expr_with_expected(expr, expected_ty);
                    self.reject_transfer_escape(&field_ty, &expr.span(), "inside a struct literal");
                }
                SemanticType::Primitive(name.clone())
            }
            Expr::Index { base, index, .. } => {
                self.require_destination_ready(base, &span);
                let base_ty = self.check_expr(base);
                let index_ty = self.check_expr(index);
                self.reject_transfer_escape(&base_ty, &base.span(), "as an indexed value");
                self.reject_transfer_escape(&index_ty, &index.span(), "as an index expression");

                // An array reached through a reference is the same array. The
                // rule below used to match a plain `Array` only, so `a[k]`
                // through `a: &mut [I16; 4]` was neither proved in bounds nor
                // checked at run time - in strict mode, the default - and
                // `a[9] = 1` compiled clean and wrote past the array. Member
                // access already looked through one reference; this is the
                // same rule at the other site.
                let base_ty = match base_ty {
                    SemanticType::Reference { inner, .. } => *inner,
                    other => other,
                };

                if let SemanticType::Array { element, size } = &base_ty {
                    INDEX_ARRAY_SIZES.with(|map| {
                        map.borrow_mut().insert((span.line, span.col), *size);
                    });

                    let mut is_safe = false;
                    let index_iv = self.eval_interval(index);
                    if let Some(index_interval) = index_iv {
                        let mut min_ok = true;
                        let mut max_ok = true;
                        if index_interval.min < 0 {
                            min_ok = false;
                            self.errors.push(format!(
                                "Line {}: [Strict Safety] Out of bounds: possible negative index access (inferred min: {}).",
                                span.line, index_interval.min
                            ));
                        }
                        if index_interval.max >= *size as i64 {
                            max_ok = false;
                            self.errors.push(format!(
                                "Line {}: [Strict Safety] Out of bounds: possible overflow index access (inferred max: {} >= array size {}).",
                                span.line, index_interval.max, size
                            ));
                        }
                        if min_ok && max_ok {
                            is_safe = true;
                        }
                    } else if !self.in_unsafe {
                        self.errors.push(format!(
                            "Line {}: [Strict Safety] Array access is unsafe: index has no statically provable bounds. Annotate the index variable with @bounds(min, max).",
                            span.line
                        ));
                    }
                    
                    if is_safe {
                        SAFE_INDICES.with(|set| {
                            set.borrow_mut().insert((span.line, span.col));
                        });
                    }
                    self.note_index(expr, &span, Some(*size), &base_ty, if is_safe { index_iv } else { None });

                    return (**element).clone();
                }

                if let SemanticType::SharedMemoryTile { rows, cols, swizzle } = &base_ty {
                    let size = (*rows * *cols) as usize;
                    INDEX_ARRAY_SIZES.with(|map| {
                        map.borrow_mut().insert((span.line, span.col), size);
                    });
                    if let Some(sw) = swizzle {
                        INDEX_SWIZZLES.with(|map| {
                            map.borrow_mut().insert((span.line, span.col), sw.clone());
                        });
                    }

                    let mut is_safe = false;
                    let index_iv = self.eval_interval(index);
                    if let Some(index_interval) = index_iv {
                        let mut min_ok = true;
                        let mut max_ok = true;
                        if index_interval.min < 0 {
                            min_ok = false;
                            self.errors.push(format!(
                                "Line {}: [Strict Safety] Out of bounds: possible negative index access (inferred min: {}).",
                                span.line, index_interval.min
                            ));
                        }
                        if index_interval.max >= size as i64 {
                            max_ok = false;
                            self.errors.push(format!(
                                "Line {}: [Strict Safety] Out of bounds: possible overflow index access (inferred max: {} >= tile size {}).",
                                span.line, index_interval.max, size
                            ));
                        }
                        if min_ok && max_ok {
                            is_safe = true;
                        }
                    } else if !self.in_unsafe {
                        self.errors.push(format!(
                            "Line {}: [Strict Safety] Array access is unsafe: index has no statically provable bounds. Annotate the index variable with @bounds(min, max).",
                            span.line
                        ));
                    }
                    
                    if is_safe {
                        SAFE_INDICES.with(|set| {
                            set.borrow_mut().insert((span.line, span.col));
                        });
                    }
                    self.note_index(expr, &span, Some(size), &base_ty, if is_safe { index_iv } else { None });

                    return SemanticType::Primitive("F16".into());
                }

                // Not a fixed-size array: a pointer (`GlobalMemory<T>`), a
                // string, a vector. Nothing bounds this index.
                self.note_index(expr, &span, None, &base_ty, None);
                SemanticType::Unknown
            }
            Expr::BinaryOp { left, op, right, span } => {
                let compares = matches!(op, BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge);
                let rhs_hint = self.known_expr_type(right);
                let numeric_context = expected_type.filter(|t| matches!(t, SemanticType::Primitive(p) if Self::is_numeric_primitive(p)));
                let operand_expected = if compares { rhs_hint.as_ref().or(numeric_context) } else { expected_type.or(rhs_hint.as_ref()) };
                let lhs = self.check_expr_with_expected(left, operand_expected);
                let rhs = self.check_expr_with_expected(right, if lhs == SemanticType::Unknown { operand_expected } else { Some(&lhs) });
                self.reject_transfer_escape(&lhs, &left.span(), "in a binary expression");
                self.reject_transfer_escape(&rhs, &right.span(), "in a binary expression");

                if *op == BinaryOp::Eq {
                    self.check_verification_transition(left, right, span);
                }

                let result = self.binary_result_type(op, &lhs, &rhs, span);
                // Y's backends also expose comparisons as integer 0/1 in an
                // explicitly numeric result context (including nested integer
                // expressions). In a condition/inferred binding they are bool.
                if compares {
                    if let Some(SemanticType::Primitive(p)) = expected_type {
                        if Self::is_numeric_primitive(p) {
                            return SemanticType::Primitive(p.clone());
                        }
                    }
                }
                result
            }
            Expr::UnaryOp { op, operand, .. } => {
                let span = expr.span();
                if *op == crate::ast::UnaryOp::Deref && !self.in_unsafe {
                    self.errors.push(format!(
                        "Line {}: [Strict Safety] Raw pointer dereferencing is forbidden in safe blocks.",
                        span.line
                    ));
                }
                let operand_ty = self.check_expr_with_expected(operand, expected_type);
                self.reject_transfer_escape(&operand_ty, &operand.span(), "in a unary expression");
                // Every unary expression used to be `Unknown`, which is the
                // same hole as `Type::Reference` above and had to be closed
                // with it: fixing only the annotation leaves the initialiser
                // untyped, and `Unknown` on EITHER side suppresses the
                // mismatch check. `&x` is what makes an annotated reference
                // binding checkable at all.
                match op {
                    crate::ast::UnaryOp::Neg | crate::ast::UnaryOp::Not => operand_ty,
                    crate::ast::UnaryOp::Ref { mutable } => SemanticType::Reference {
                        inner: Box::new(operand_ty),
                        mutable: *mutable,
                    },
                    // Dereferencing a reference yields what it points at.
                    //
                    // Anything we can positively identify as NOT a reference is
                    // refused. Y has no raw pointer type - only `&T` - so `*p`
                    // with `p: I32` is a type error and not an escape hatch,
                    // and treating it as one produced INVALID LLVM IR under a
                    // green banner: `store i32 0, ptr %_t1` where `%_t1` is an
                    // i32. `@unsafe` does not license it either, because there
                    // is no pointee type to derive a width from - lowering it
                    // through `inttoptr` would have to guess one, which is the
                    // substitution the design rule exists to forbid.
                    //
                    // `Unknown` is deliberately NOT refused. It means the
                    // checker has no information, not that it has established
                    // the operand is a non-reference, and this checker leaves
                    // plenty untyped - refusing it would reject correct
                    // programs. That is a whitelist with its reason written
                    // down, per the design rule, and the LLVM backend's own
                    // GPU-construct refusal is what covers what reaches it.
                    crate::ast::UnaryOp::Deref => match &operand_ty {
                        SemanticType::Reference { inner, .. } => (**inner).clone(),
                        SemanticType::Unknown => SemanticType::Unknown,
                        other => {
                            self.errors.push(format!(
                                "Line {}: Cannot dereference `{}` - it is not a reference. \
                                 Y has no raw pointer type; pass a `&mut T` and dereference that.",
                                span.line,
                                Self::semantic_type_name(other)
                            ));
                            SemanticType::Unknown
                        }
                    },
                }
            }
            Expr::BlockExpr(block, _) => {
                self.check_block(block);
                SemanticType::Unknown
            }
            // Literals were falling into the catch-all below and typing as
            // `Unknown`, which the `let` arm reads as "adopt the annotation"
            // and the assignment arm exempts from the mismatch check outright.
            // It also made `require_bool_condition` dormant: `if 1` could not
            // be caught, because the 1 had no type at all.
            //
            // A literal is POLYMORPHIC and takes the expected type where there
            // is one - `let x: F16 = 0.0` is legal and pinning the literal to
            // `F32` made it a mismatch. Where nothing is expected it falls
            // back to the default, which is what gives an `if` condition a
            // type to check.
            //
            // Only the FLOAT arm is observable today: mutation shows that
            // pinning `IntLit` to `I32` changes nothing, because the `let`
            // mismatch check is lenient between integer widths. It is kept
            // anyway - the two arms are one rule, and an int/float asymmetry
            // here would be a trap the moment that leniency is tightened. Not
            // the same case as a clause that is logically IMPLIED, which
            // should be deleted rather than kept.
            Expr::BoolLit(..) => SemanticType::Primitive("bool".into()),
            Expr::StringLit(..) => SemanticType::Primitive("String".into()),
            Expr::CharLit(..) => SemanticType::Primitive("char".into()),
            Expr::SelfLit(..) => self.lookup_var("self").cloned().unwrap_or(SemanticType::Unknown),
            Expr::Path { namespace, member, .. } => {
                if let Some(e) = self.enums.get(namespace) {
                    if let Some(variant) = e.variants.iter().find(|v| v.name == *member) {
                        if variant.fields.as_ref().is_some_and(|fields| !fields.is_empty()) {
                            self.errors.push(format!("Line {}: enum variant `{}::{}` requires constructor arguments.", span.line, namespace, member));
                        }
                        return SemanticType::Primitive(namespace.clone());
                    }
                }
                SemanticType::Unknown
            }
            Expr::IntLit(..) => match expected_type {
                Some(SemanticType::Primitive(p)) if Self::is_numeric_primitive(p) => {
                    SemanticType::Primitive(p.clone())
                }
                _ => SemanticType::Primitive("I32".into()),
            },
            Expr::FloatLit(..) => match expected_type {
                Some(SemanticType::Primitive(p)) if p.starts_with('F') || p.starts_with('f') || p.starts_with('Q') => {
                    SemanticType::Primitive(p.clone())
                }
                _ => SemanticType::Primitive("F32".into()),
            },
        }
    }

    // ── Semantic Verifications ──────────────────────────────

    /// Enforces Phantom Fragment Role types. (A + B + C -> D)
    fn check_mma_sync(&mut self, args: &[Expr], span: &Span) {
        if args.len() != 3 {
            self.errors.push(format!(
                "Line {}: mma_sync requires exactly 3 operands (A, B, C).",
                span.line
            ));
            return;
        }

        let t_a = self.check_expr(&args[0]);
        let t_b = self.check_expr(&args[1]);
        let t_c = self.check_expr(&args[2]);

        let mut require_role = |ty: &SemanticType, expected_roles: &[&str]| {
            if let SemanticType::Fragment { role, .. } = ty {
                if !expected_roles.contains(&role.as_str()) {
                    self.errors.push(format!(
                        "Line {}: Fragment Role Error: expected Fragment<{}, ...>, got Fragment<{}, ...>.",
                        span.line, expected_roles.join("/"), role
                    ));
                }
            }
        };

        require_role(&t_a, &["A"]);
        require_role(&t_b, &["B"]);
        require_role(&t_c, &["C", "D"]); // Or D commonly used for accumulator feedback
    }

    // ── Type Resolution ─────────────────────────────────────

    fn resolve_type(&mut self, ast_ty: &Type) -> SemanticType {
        match ast_ty {
            Type::Primitive(name, _) => SemanticType::Primitive(name.clone()),
            Type::Ident(name, _) => {
                if name == "ptr" {
                    SemanticType::Primitive("ptr".into())
                } else if let Some(t) = self.lookup_var(name) {
                    t.clone() // alias resolution
                } else if self.structs.contains_key(name) || self.enums.contains_key(name) {
                    // A declared struct, resolved the way `Expr::StructLit`
                    // reports itself. Without this arm the name fell through to
                    // `Unknown`, and since `types_are_compatible` treats
                    // `Unknown` as compatible with nothing, an ANNOTATED
                    // binding of a struct was a type error:
                    //
                    //     struct P { x: I32, y: I32 }
                    //     let p: P = P { x: 4, y: 3 };   // "Type mismatch in
                    //                                    //  let assignment."
                    //
                    // The same binding without the annotation compiled, so the
                    // language rejected the more explicit of two spellings of
                    // one program. `resolve_type` consulted only the variable
                    // table (type aliases); the struct table sat beside it and
                    // was read by `Expr::StructLit` alone.
                    SemanticType::Primitive(name.clone())
                } else {
                    SemanticType::Unknown
                }
            }
            Type::Generic { base, args, span, .. } => {
                if base == "Fragment" && args.len() >= 3 {
                    let mut op = "Unknown".to_string();
                    let mut role = "Unknown".to_string();
                    let mut dtype = "Unknown".to_string();

                    if let GenericArg::Type(Type::Ident(o, _)) = &args[0] {
                        op = o.clone();
                    }
                    if let GenericArg::Type(Type::Ident(r, _)) = &args[1] {
                        role = r.clone();
                    }
                    if let GenericArg::Type(Type::Primitive(d, _)) = &args[2] {
                        dtype = d.clone();
                    }

                    return SemanticType::Fragment { op, role, dtype };
                }

                if base == "Vec" {
                    let mut inner_ty = SemanticType::Unknown;
                    let mut allocator = "Standard".to_string();
                    if args.len() >= 1 {
                        if let GenericArg::Type(t) = &args[0] {
                            inner_ty = self.resolve_type(t);
                        }
                    }
                    if args.len() >= 2 {
                        if let GenericArg::Type(Type::Ident(alloc, _)) = &args[1] {
                            allocator = alloc.clone();
                        }
                    }
                    return SemanticType::Vector(Box::new(inner_ty), allocator);
                }

                if base == "SmemLayout" {
                    let mut rows = 0;
                    let mut cols = 0;
                    let mut swizzle = None;

                    for arg in args {
                        if let GenericArg::Named { name, val } = arg {
                            if name == "rows" {
                                if let Expr::IntLit(r, _) = val {
                                    rows = *r as u32;
                                }
                            }
                            if name == "cols" {
                                if let Expr::IntLit(c, _) = val {
                                    cols = *c as u32;
                                }
                            }
                            if name == "swizzle" {
                                // Dummy fill for parser validation context
                                swizzle = Some(SwizzlePattern {
                                    xor_bits: 3,
                                    base_shift: 0,
                                    offset: 0,
                                });
                            }
                        }
                    }

                    return SemanticType::SharedMemoryTile {
                        rows,
                        cols,
                        swizzle,
                    };
                }

                if base == "Transfer" {
                    return SemanticType::TransferObligation;
                }

                if base == "BlockTile" {
                    let mut elem_resolved = SemanticType::Primitive("F32".into());
                    if !args.is_empty() {
                        if let GenericArg::Type(t) = &args[0] {
                            elem_resolved = self.resolve_type(t);
                        }
                    }
                    // A tile's SIZE is part of its type, and `SemanticType`
                    // derives `PartialEq`, so `types_are_compatible` compares
                    // it. A guessed size therefore does not merely lose
                    // information -- it makes two differently-sized tiles
                    // compare EQUAL, and an assignment between them legal.
                    // This used to default to 128 for a missing or
                    // non-literal argument.
                    let sz = match args.get(1) {
                        Some(GenericArg::Value(Expr::IntLit(v, _))) => *v as usize,
                        _ => {
                            self.errors.push(format!(
                                "Line {}: a `BlockTile` needs a literal size as its \
                                 second generic argument; a size the compiler cannot \
                                 evaluate would make tiles of different sizes compare \
                                 equal.",
                                span.line
                            ));
                            return SemanticType::Unknown;
                        }
                    };
                    return SemanticType::BlockTile {
                        element: Box::new(elem_resolved),
                        size: sz,
                    };
                }

                // An unrecognised generic base is REFUSED rather than resolved
                // to `Unknown`. `Unknown` is not a neutral answer here: the
                // `let` arm adopts the annotation and the assignment arm skips
                // the mismatch check, and downstream `llvm_emitter::emit_type`
                // ends its `Type::Generic` match with `_ => "ptr"`. So
                // `let a: Nonsense<F32, 8> = ...` compiled clean and became a
                // POINTER -- a legal LLVM type, so nothing further could
                // object. (The `Type::Ident` spelling of the same typo is at
                // least caught eventually, by clang, as "Cannot allocate
                // unsized type" pointing at generated IR rather than at the
                // user's line.)
                //
                // The list is every base any part of this compiler models. It
                // is deliberately a whitelist: a new generic type must be
                // taught to the checker before it can be written, which is the
                // opposite of the previous arrangement.
                const KNOWN_GENERIC_BASES: [&str; 9] = [
                    "Vec",
                    "Option",
                    "Box",
                    "GlobalMemory",
                    "SharedMemory",
                    "SmemLayout",
                    "Fragment",
                    "Transfer",
                    "BlockTile",
                ];
                if !KNOWN_GENERIC_BASES.contains(&base.as_str()) {
                    self.errors.push(format!(
                        "Line {}: unknown generic type `{}`. Known generic types are: {}.",
                        span.line,
                        base,
                        KNOWN_GENERIC_BASES.join(", ")
                    ));
                }
                SemanticType::Unknown
            }
            Type::Array { element, size, .. } => {
                let elem_resolved = self.resolve_type(element);
                let mut sz = 0;
                if let Expr::IntLit(val, _) = &**size {
                    sz = *val as usize;
                }
                SemanticType::Array {
                    element: Box::new(elem_resolved),
                    size: sz,
                }
            }
            Type::BlockTile { element, size, .. } => {
                let elem_resolved = self.resolve_type(element);
                // See the `BlockTile` path in the generic-application arm
                // above: a guessed size is not a lossy answer, it is a wrong
                // type-equality.
                let sz = match &**size {
                    Expr::IntLit(val, _) => *val as usize,
                    other => {
                        let sp = other.span();
                        self.errors.push(format!(
                            "Line {}: a `BlockTile` size must be a literal; a size \
                             the compiler cannot evaluate would make tiles of \
                             different sizes compare equal.",
                            sp.line
                        ));
                        return SemanticType::Unknown;
                    }
                };
                SemanticType::BlockTile {
                    element: Box::new(elem_resolved),
                    size: sz,
                }
            }
            Type::Reference { inner, mutable, .. } => SemanticType::Reference {
                inner: Box::new(self.resolve_type(inner)),
                mutable: *mutable,
            },
        }
    }

    /// The SSA model is keyed by names, so shadowing cannot safely share it.
    /// Check lexical scopes before the body introduces its local bindings.
    fn smt_shadowed_binding(&self, statements: &[Stmt]) -> Option<String> {
        fn visit(statements: &[Stmt], names: &mut std::collections::HashSet<String>) -> Option<String> {
            for statement in statements {
                match statement {
                    Stmt::Let { name, .. } => {
                        if !names.insert(name.clone()) { return Some(name.clone()); }
                    }
                    Stmt::For { loop_var, body, .. } => {
                        let mut nested = names.clone();
                        if !nested.insert(loop_var.clone()) { return Some(loop_var.clone()); }
                        if let Some(name) = visit(&body.stmts, &mut nested) { return Some(name); }
                    }
                    Stmt::If { then_block, else_block, .. } => {
                        if let Some(name) = visit(&then_block.stmts, &mut names.clone()) { return Some(name); }
                        if let Some(block) = else_block {
                            if let Some(name) = visit(&block.stmts, &mut names.clone()) { return Some(name); }
                        }
                    }
                    Stmt::While { body, .. } | Stmt::SafeBlock(body, _) | Stmt::GhostBlock(body, _)
                    | Stmt::Chisel(body, _) | Stmt::HintBlock { body, .. }
                    | Stmt::ClockDomainBlock { body, .. } => {
                        if let Some(name) = visit(&body.stmts, &mut names.clone()) { return Some(name); }
                    }
                    _ => {}
                }
            }
            None
        }
        let mut names = self.scopes.iter().flat_map(|scope| scope.symbols.keys().cloned()).collect();
        visit(statements, &mut names)
    }

    fn contains_unsigned_literal(expr: &Expr) -> bool {
        match expr {
            Expr::IntLit(n, _) => *n > i32::MAX as i64 && *n <= u32::MAX as i64,
            Expr::UnaryOp { operand, .. } => Self::contains_unsigned_literal(operand),
            Expr::BinaryOp { left, right, .. } =>
                Self::contains_unsigned_literal(left) || Self::contains_unsigned_literal(right),
            _ => false,
        }
    }

    /// Only signed integers have common arithmetic semantics in LLVM and PTX.
    /// Refuse unsigned proofs until the backends agree on their operators.
    fn smt_integer_width(&self, expr: &Expr) -> Result<u32, String> {
        match expr {
            // For interval propagation use the narrower backend width. PTX
            // holds positive u32 literals in 32 bits; LLVM uses 64 bits.
            Expr::IntLit(n, _) => Ok(if *n >= i32::MIN as i64 && *n <= u32::MAX as i64 { 32 } else { 64 }),
            Expr::Ident(name, _) => match self.lookup_var(name) {
                Some(SemanticType::Primitive(ty)) => match ty.to_ascii_lowercase().as_str() {
                    "i8" => Ok(8), "i16" => Ok(16), "i32" => Ok(32), "i64" => Ok(64),
                    _ => Err(format!("`{name}` does not have a supported signed integer type")),
                },
                _ => Err(format!("the integer type of `{name}` is not known")),
            },
            Expr::UnaryOp { op: UnaryOp::Neg, operand, .. } => self.smt_integer_width(operand),
            Expr::BinaryOp { left, op, right, .. }
                if matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod) =>
                Ok(self.smt_integer_width(left)?.max(self.smt_integer_width(right)?)),
            Expr::Call { func, args, .. } if args.is_empty()
                && matches!(&**func, Expr::Ident(name, _) if gpu_index_symbol(name).is_some()) => Ok(32),
            _ => Err("expression has no supported machine integer type".into()),
        }
    }

    fn smt_in_range(value: &str, bits: u32) -> String {
        let limit = 1i128 << (bits - 1);
        format!("(and (>= {value} {}) (<= {value} {}))", -limit, limit - 1)
    }

    fn smt_all(requirements: &[String], value: &str) -> String {
        if requirements.is_empty() { value.to_string() }
        else { format!("(and {} {value})", requirements.join(" ")) }
    }

    /// SMT div rounds down; machine signed division truncates toward zero.
    fn smt_signed_quotient(lhs: &str, rhs: &str) -> String {
        format!("(let ((div_lhs {lhs}) (div_rhs {rhs})) (let ((div_magnitude (div (ite (< div_lhs 0) (- div_lhs) div_lhs) (ite (< div_rhs 0) (- div_rhs) div_rhs)))) (ite (= (< div_lhs 0) (< div_rhs 0)) div_magnitude (- div_magnitude))))")
    }

    /// Translates an expression into SMT-LIB, or reports that it cannot.
    /// Every intermediate must fit its machine width. Requirements are proof
    /// goals, NEVER assumptions that would exclude overflowing executions.
    ///
    /// **Every unhandled node returns `Err`.** That is the entire point, and it
    /// is a reversal: this function used to end in `_ => "0".to_string()`, with
    /// `_ => "+"` for unknown binary operators and `_ => opnd` for unknown unary
    /// ones. So a call, an index, a member access or even a float literal was
    /// handed to Z3 as the constant `0`; `x & y` was proven as `x + y`; and
    /// `*p` was proven as `p`. Z3 then dutifully answered a question about a
    /// different program, and the answer was reported as a verified invariant.
    ///
    /// A verifier that silently approximates is worse than no verifier, because
    /// it produces the paperwork of a proof without the proof. If a construct
    /// is not modelled, the only safe answer is to refuse to make a claim.
    fn expr_to_smt(
        &self,
        expr: &Expr,
        versions: &HashMap<String, usize>,
        requirements: &mut Vec<String>,
    ) -> Result<String, String> {
        match expr {
            Expr::IntLit(n, _) if *n > i32::MAX as i64 && *n <= u32::MAX as i64 =>
                Err("integer literal has different signedness in LLVM and PTX".into()),
            Expr::IntLit(val, _) => Ok(val.to_string()),
            Expr::BoolLit(val, _) => Ok(val.to_string()),
            Expr::Ident(name, _) => {
                self.smt_integer_width(expr)?;
                if let Some(&ver) = versions.get(name) {
                    Ok(format!("{}_{}", name, ver))
                } else {
                    Ok(format!("{}_0", name))
                }
            }
            // The GPU index intrinsics are the one class of call this encoder
            // models rather than refuses. They take no arguments, have no
            // side effects, and their ranges are guaranteed by the hardware -
            // so mapping each to a canonical symbol supplies the launch
            // bounds needed by GPU loop invariants. These bounds describe
            // the intrinsic results, not the absence of overflow in an
            // arbitrary expression that combines them.
            //
            // This is the one place in this file that makes an obligation
            // EASIER, so the facts asserted alongside it (in
            // `gpu_index_bound`) must be hardware guarantees and nothing more.
            Expr::Call { func, args, .. } if args.is_empty() => match &**func {
                Expr::Ident(name, _) => match gpu_index_symbol(name) {
                    Some(sym) => Ok(format!("{}_0", sym)),
                    None => Err(format!("call to `{}` is not modellable", name)),
                },
                _ => Err("indirect call is not modellable".to_string()),
            },
            Expr::BinaryOp { left, op, right, .. } => {
                let lhs = self.expr_to_smt(left, versions, requirements)?;
                let rhs = self.expr_to_smt(right, versions, requirements)?;
                if matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod) {
                    let bits = self.smt_integer_width(expr)?;
                    let result = match op {
                        BinaryOp::Div | BinaryOp::Mod => {
                            requirements.push(format!("(distinct {rhs} 0)"));
                            requirements.push(format!("(not (and (= {lhs} {}) (= {rhs} (- 1))))", -(1i128 << (bits - 1))));
                            let quotient = Self::smt_signed_quotient(&lhs, &rhs);
                            if *op == BinaryOp::Div { quotient }
                            else { format!("(- {lhs} (* {rhs} {quotient}))") }
                        }
                        _ => format!("({} {lhs} {rhs})", match op {
                            BinaryOp::Add => "+", BinaryOp::Sub => "-", _ => "*",
                        }),
                    };
                    requirements.push(Self::smt_in_range(&result, bits));
                    return Ok(result);
                }
                let op_str = match op {
                    BinaryOp::Eq => "=",
                    BinaryOp::NotEq => "distinct",
                    BinaryOp::Lt => "<",
                    BinaryOp::Gt => ">",
                    BinaryOp::Le => "<=",
                    BinaryOp::Ge => ">=",
                    BinaryOp::And => "and",
                    BinaryOp::Or => "or",
                    // Bitwise and shift operators have no direct counterpart in
                    // the integer theory used here. They used to fall through to
                    // "+", which is not an approximation, it is a different
                    // function.
                    other => {
                        return Err(format!(
                            "the operator `{:?}` has no sound encoding in the integer theory \
this verifier uses",
                            other
                        ))
                    }
                };
                Ok(format!("({} {} {})", op_str, lhs, rhs))
            }
            Expr::UnaryOp { op, operand, .. } => {
                let opnd = self.expr_to_smt(operand, versions, requirements)?;
                match op {
                    UnaryOp::Neg => {
                        let result = format!("(- {opnd})");
                        requirements.push(Self::smt_in_range(&result, self.smt_integer_width(expr)?));
                        Ok(result)
                    }
                    UnaryOp::Not => Ok(format!("(not {})", opnd)),
                    other => Err(format!(
                        "the unary operator `{:?}` is not modelled (it used to be encoded as its \
own operand, so `*p` was proven as `p`)",
                        other
                    )),
                }
            }
            other => Err(format!(
                "`{}` is not modelled by the verifier",
                expr_to_string(other)
            )),
        }
    }

    fn generate_smt_decls_and_preconditions(
        &self,
        vars: &std::collections::HashSet<String>,
        declarations: &mut Vec<String>,
        preconditions: &mut Vec<String>,
    ) {
        self.generate_smt_decls_and_preconditions_with(vars, None, declarations, preconditions)
    }

    /// As above, but `at_entry` may supply the interval a variable held
    /// immediately BEFORE the loop.
    ///
    /// `check_stmt` clears the interval of every variable a loop body assigns
    /// before it verifies anything, and it has to: `check_block` reasons about
    /// `@bounds` inside the body, where a range measured before the loop is no
    /// longer true, and the PRESERVATION obligation needs the same. But the
    /// INITIATION obligation is a statement about the state on entry, where
    /// that range is still exactly true - so clearing it first made every
    /// useful invariant unprovable:
    ///
    /// ```text
    /// let acc: I32 = 0;
    /// @invariant(acc >= 0)
    /// for i in 0..4 { acc = acc + 1; }   // initiation check FAILED
    /// ```
    ///
    /// An invariant about a variable the body does not touch is trivial, and
    /// an invariant about one it does was the only kind that could not be
    /// stated - so `while` was unusable outright (its induction variable is
    /// always body-assigned) and `for` worked only for invariants over its own
    /// induction variable, whose range `verify_for_loop_invariant` re-derives
    /// from `start`/`end`. Two of this repo's own test programs, `math.ysu`
    /// and `safe_test.ysu`, were refused by it.
    ///
    /// Passing the snapshot is not an assumption: it is the value the pass had
    /// already computed for the statement before the loop. It must reach the
    /// initiation query ONLY - a fact true on entry is not true after an
    /// iteration, and the preservation query is where that distinction is the
    /// whole point.
    fn generate_smt_decls_and_preconditions_with(
        &self,
        vars: &std::collections::HashSet<String>,
        at_entry: Option<&HashMap<String, Interval>>,
        declarations: &mut Vec<String>,
        preconditions: &mut Vec<String>,
    ) {
        for var in vars {
            declarations.push(format!("(declare-const {}_{} Int)", var, 0));
            let interval = at_entry
                .and_then(|m| m.get(var))
                .or_else(|| self.lookup_interval(var));
            if at_entry.is_none() || interval.is_none() {
                // A stored signed variable is always representable. These are
                // domain facts, unlike the operation-result bounds proved below.
                // At initiation, a concrete interval is kept on its own:
                // intersecting a bad initializer's interval with the machine
                // range could make the entry state contradictory. An unknown
                // runtime parameter has no such interval, and its declared
                // signed type still guarantees its machine range on entry.
                let expression = Expr::Ident(var.clone(), Span { line: 0, col: 0 });
                if let Ok(bits) = self.smt_integer_width(&expression) {
                    preconditions.push(format!("(assert {})", Self::smt_in_range(&format!("{var}_0"), bits)));
                }
            }
            if let Some(interval) = interval {
                self.smt_trust.set(self.smt_trust.get() | interval.trust);
                preconditions.push(format!(
                    "(assert (and (>= {}_{} {}) (<= {}_{} {})))",
                    var, 0, interval.min, var, 0, interval.max
                ));
            }
        }
    }

    /// Backward slice of a loop body against its invariant.
    ///
    /// The SMT encoder puts the WHOLE body into one query per obligation, and
    /// that stops working on generated code: a Pippenger bucket-accumulation
    /// loop whose body is a 30-multiply elliptic-curve point addition is
    /// ~9,000 statements, and z3 never returns. The invariant on that loop is
    /// `k >= 0` - it mentions the loop variable and nothing else, so not one
    /// of those 9,000 statements can affect it.
    ///
    /// So: keep only the statements that can. Starting from the variables the
    /// invariant reads, any assignment INTO a relevant variable makes its
    /// right-hand side relevant too, to a fixpoint; every other assignment is
    /// dropped. Control-flow statements survive if anything inside them
    /// survived, which preserves the havoc sets `trace_body_statements`
    /// computes (havoc of an irrelevant variable cannot change the answer).
    ///
    /// Soundness rests on `relevant` being an OVER-approximation of what the
    /// invariant depends on. `collect_reads` therefore reports failure on any
    /// expression shape it does not fully understand, and `slice_body` then
    /// returns `None`, which makes the caller encode the entire body exactly
    /// as before. A gap in this analysis costs compile time, never a missed
    /// violation - which is the only acceptable direction here, because the
    /// thing being weakened is a safety check.
    fn slice_body_for_invariant(stmts: &[Stmt], invariant: &Expr, loop_var: &str) -> Option<Vec<Stmt>> {
        let mut relevant = std::collections::HashSet::new();
        relevant.insert(loop_var.to_string());
        if !Self::collect_reads(invariant, &mut relevant) {
            return None;
        }
        loop {
            let before = relevant.len();
            if !Self::grow_relevant(stmts, &mut relevant) {
                return None;
            }
            if relevant.len() == before {
                break;
            }
        }
        Some(Self::keep_relevant(stmts, &relevant))
    }

    /// Adds every identifier `e` reads to `out`. Returns false if it met an
    /// expression it cannot fully walk, in which case the caller must not
    /// slice.
    fn collect_reads(e: &Expr, out: &mut std::collections::HashSet<String>) -> bool {
        match e {
            Expr::Ident(n, _) => {
                out.insert(n.clone());
                true
            }
            Expr::IntLit(..) | Expr::FloatLit(..) | Expr::BoolLit(..) | Expr::StringLit(..) => true,
            Expr::BinaryOp { left, right, .. } => {
                Self::collect_reads(left, out) && Self::collect_reads(right, out)
            }
            Expr::UnaryOp { operand, .. } => Self::collect_reads(operand, out),
            Expr::Index { base, index, .. } => {
                Self::collect_reads(base, out) && Self::collect_reads(index, out)
            }
            Expr::Call { func, args, .. } => {
                Self::collect_reads(func, out) && args.iter().all(|a| Self::collect_reads(a, out))
            }
            Expr::MemberAccess { base, .. } => Self::collect_reads(base, out),
            Expr::Path { .. } => true,
            _ => false,
        }
    }

    /// One fixpoint round. Returns false if anything was unanalysable.
    fn grow_relevant(stmts: &[Stmt], relevant: &mut std::collections::HashSet<String>) -> bool {
        for stmt in stmts {
            let ok = match stmt {
                Stmt::Assign { target, value, .. } => match target {
                    Expr::Ident(n, _) if relevant.contains(n) => Self::collect_reads(value, relevant),
                    Expr::Ident(_, _) => true,
                    // A write through anything other than a plain name could
                    // alias a relevant variable; refuse to slice.
                    _ => false,
                },
                Stmt::CompoundAssign { target, value, .. } => match target {
                    Expr::Ident(n, _) if relevant.contains(n) => Self::collect_reads(value, relevant),
                    Expr::Ident(_, _) => true,
                    _ => false,
                },
                Stmt::Let { name, init, .. } => {
                    if relevant.contains(name) {
                        init.as_ref().map_or(true, |e| Self::collect_reads(e, relevant))
                    } else {
                        true
                    }
                }
                Stmt::If { then_block, else_block, .. } => {
                    Self::grow_relevant(&then_block.stmts, relevant)
                        && else_block.as_ref().map_or(true, |b| Self::grow_relevant(&b.stmts, relevant))
                }
                Stmt::For { body, .. } | Stmt::While { body, .. } => {
                    Self::grow_relevant(&body.stmts, relevant)
                }
                Stmt::SafeBlock(b, _) | Stmt::Chisel(b, _) | Stmt::GhostBlock(b, _)
                | Stmt::HintBlock { body: b, .. } | Stmt::ClockDomainBlock { body: b, .. } => {
                    Self::grow_relevant(&b.stmts, relevant)
                }
                Stmt::Expr(_) | Stmt::Return(..) | Stmt::TypeAlias { .. } | Stmt::Break { .. } => true,
                // Anything unrecognised: do not slice.
                _ => false,
            };
            if !ok {
                return false;
            }
        }
        true
    }

    fn keep_relevant(stmts: &[Stmt], relevant: &std::collections::HashSet<String>) -> Vec<Stmt> {
        let mut out = Vec::new();
        for stmt in stmts {
            match stmt {
                Stmt::Assign { target: Expr::Ident(n, _), .. }
                | Stmt::CompoundAssign { target: Expr::Ident(n, _), .. } => {
                    if relevant.contains(n) {
                        out.push(stmt.clone());
                    }
                }
                Stmt::Let { name, .. } => {
                    if relevant.contains(name) {
                        out.push(stmt.clone());
                    }
                }
                Stmt::If { condition, then_block, else_block, is_uniform_branch, span } => {
                    let t = Self::keep_relevant(&then_block.stmts, relevant);
                    let e = else_block.as_ref().map(|b| Self::keep_relevant(&b.stmts, relevant));
                    if !t.is_empty() || e.as_ref().map_or(false, |v| !v.is_empty()) {
                        let mut tb = then_block.clone();
                        tb.stmts = t;
                        let eb = else_block.as_ref().map(|b| {
                            let mut nb = b.clone();
                            nb.stmts = e.clone().unwrap_or_default();
                            nb
                        });
                        out.push(Stmt::If {
                            condition: condition.clone(),
                            then_block: tb,
                            else_block: eb,
                            is_uniform_branch: *is_uniform_branch,
                            span: span.clone(),
                        });
                    }
                }
                Stmt::For { body, .. } | Stmt::While { body, .. } => {
                    if !Self::keep_relevant(&body.stmts, relevant).is_empty() {
                        out.push(stmt.clone());
                    }
                }
                Stmt::SafeBlock(b, _) | Stmt::Chisel(b, _) | Stmt::GhostBlock(b, _)
                | Stmt::HintBlock { body: b, .. } | Stmt::ClockDomainBlock { body: b, .. } => {
                    if !Self::keep_relevant(&b.stmts, relevant).is_empty() {
                        out.push(stmt.clone());
                    }
                }
                _ => out.push(stmt.clone()),
            }
        }
        out
    }

    fn trace_body_statements(
        &self,
        stmts: &[Stmt],
        versions: &mut HashMap<String, usize>,
        declarations: &mut Vec<String>,
        body_assertions: &mut Vec<String>,
        requirements: &mut Vec<String>,
    ) -> Result<(), String> {
        for stmt in stmts {
            match stmt {
                Stmt::Assign { target, value, .. } | Stmt::CompoundAssign { target, value, .. } => {
                    if let Expr::Ident(name, _) = target {
                        if versions.contains_key(name) {
                            let expression = match stmt {
                                Stmt::CompoundAssign { op, span, .. } => Expr::BinaryOp {
                                    left: Box::new(target.clone()), op: op.clone(),
                                    right: Box::new(value.clone()), span: span.clone(),
                                },
                                _ => value.clone(),
                            };
                            let mut local_requirements = Vec::new();
                            let encoded = self.smt_integer_width(target).and_then(|bits| {
                                self.expr_to_smt(&expression, versions, &mut local_requirements)
                                    .map(|rhs| (bits, rhs))
                            });
                            let Ok((bits, rhs)) = encoded else {
                                let mut one = std::collections::HashSet::new();
                                one.insert(name.clone());
                                Self::havoc(&one, versions, declarations);
                                continue;
                            };
                            // LLVM compound operations use the destination width,
                            // while PTX promotes first. Prove that converting their
                            // RHS is lossless before using a common arithmetic model.
                            if matches!(stmt, Stmt::CompoundAssign { .. }) {
                                let operand = self.expr_to_smt(value, versions, &mut local_requirements)?;
                                local_requirements.push(Self::smt_in_range(&operand, bits));
                            }
                            // Conversions must also be lossless. Checking only the
                            // final expression's width misses narrowing stores.
                            local_requirements.push(Self::smt_in_range(&rhs, bits));
                            requirements.extend(local_requirements);
                            let next = versions[name] + 1;
                            versions.insert(name.clone(), next);
                            declarations.push(format!("(declare-const {name}_{next} Int)"));
                            body_assertions.push(format!("(assert (= {name}_{next} {rhs}))"));
                        }
                    }
                }
                Stmt::Let { name, init, span, .. } => {
                    let target = Expr::Ident(name.clone(), span.clone());
                    let mut local_requirements = Vec::new();
                    let encoded = self.smt_integer_width(&target).and_then(|bits| {
                        let init = init.as_ref().ok_or("uninitialized integer binding")?;
                        self.expr_to_smt(init, versions, &mut local_requirements)
                            .map(|rhs| (bits, rhs))
                    });
                    // Never reset a reused name to version zero: that can
                    // contradict entry facts and make the proof vacuous.
                    let next = versions.get(name).map_or(0, |v| v + 1);
                    versions.insert(name.clone(), next);
                    declarations.push(format!("(declare-const {name}_{next} Int)"));
                    if let Ok((bits, rhs)) = encoded {
                        local_requirements.push(Self::smt_in_range(&rhs, bits));
                        requirements.extend(local_requirements);
                        body_assertions.push(format!("(assert (= {name}_{next} {rhs}))"));
                    }
                }
                Stmt::SafeBlock(block, _) | Stmt::Chisel(block, _) | Stmt::GhostBlock(block, _) | Stmt::HintBlock { body: block, .. } => {
                    self.trace_body_statements(&block.stmts, versions, declarations, body_assertions, requirements)?;
                }
                Stmt::ClockDomainBlock { body, .. } => {
                    self.trace_body_statements(&body.stmts, versions, declarations, body_assertions, requirements)?;
                }
                // A branch is modelled by HAVOC: every variable it might assign
                // gets a fresh, unconstrained version. That is a sound
                // over-approximation - the invariant must then hold for any
                // value the branch could have produced, which includes the real
                // one - so it can only make preservation harder to prove, never
                // easier. Skipping the branch entirely had the opposite effect,
                // which is what made `if i >= 0 { i = i - 100; }` satisfy
                // `@invariant(i >= 0)`.
                Stmt::If { then_block, else_block, .. } => {
                    let mut touched = std::collections::HashSet::new();
                    Self::collect_assigned(&then_block.stmts, &mut touched);
                    if let Some(eb) = else_block {
                        Self::collect_assigned(&eb.stmts, &mut touched);
                    }
                    Self::havoc(&touched, versions, declarations);
                }
                // Nested loops likewise: whatever they assign becomes unknown.
                Stmt::For { body, loop_var, .. } => {
                    let mut touched = std::collections::HashSet::new();
                    Self::collect_assigned(&body.stmts, &mut touched);
                    touched.insert(loop_var.clone());
                    Self::havoc(&touched, versions, declarations);
                }
                Stmt::While { body, .. } => {
                    let mut touched = std::collections::HashSet::new();
                    Self::collect_assigned(&body.stmts, &mut touched);
                    Self::havoc(&touched, versions, declarations);
                }
                // A bare expression is almost always a call. Y has no way for a
                // callee to write a caller's local integer unless the caller
                // hands over a reference, so a call with no `&` cannot disturb
                // the tracked variables - all of which are integer scalars.
                // Anything that does take a reference is refused.
                Stmt::Expr(e) => {
                    if self.takes_reference(e) {
                        return Err(
                            "the loop body passes a reference to a call, which could modify a \
tracked variable in a way this verifier cannot see"
                                .to_string(),
                        );
                    }
                }
                // Anything else is NOT modelled, and an unmodelled statement is
                // rejected rather than skipped.
                //
                // This used to be `_ => {}`. Dropping a statement's effects
                // makes the preservation check strictly easier, so it fails in
                // the unsound direction: the identical violation was caught
                // when written plainly and ACCEPTED when wrapped in a
                // trivially-true `if`, because `Stmt::If` had no arm here.
                // Guarded by `nested_if_cannot_hide_a_false_invariant`.
                other => {
                    return Err(format!(
                        "the loop body contains a `{}` statement, which this verifier does not \
model",
                        Self::stmt_kind(other)
                    ))
                }
            }
        }
        Ok(())
    }

    /// Names assigned anywhere in `stmts`, including nested blocks.
    ///
    /// This is the havoc set for a branch or a nested loop, so a name it misses
    /// keeps its stale version and the preservation obligation gets STRICTLY
    /// EASIER - the unsound direction. Matched exhaustively with no `_ =>` arm
    /// for that reason; every empty arm below states why it is empty.
    fn collect_assigned(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
        for stmt in stmts {
            match stmt {
                Stmt::Assign { target, .. } | Stmt::CompoundAssign { target, .. } => {
                    if let Expr::Ident(n, _) = target {
                        out.insert(n.clone());
                    }
                }
                Stmt::Let { name, .. } => {
                    out.insert(name.clone());
                }
                Stmt::If { then_block, else_block, .. } => {
                    Self::collect_assigned(&then_block.stmts, out);
                    if let Some(eb) = else_block {
                        Self::collect_assigned(&eb.stmts, out);
                    }
                }
                Stmt::For { body, loop_var, .. } => {
                    out.insert(loop_var.clone());
                    Self::collect_assigned(&body.stmts, out);
                }
                Stmt::While { body, .. } => Self::collect_assigned(&body.stmts, out),
                Stmt::SafeBlock(b, _)
                | Stmt::Chisel(b, _)
                | Stmt::GhostBlock(b, _)
                | Stmt::HintBlock { body: b, .. } => Self::collect_assigned(&b.stmts, out),
                Stmt::ClockDomainBlock { body, .. } => Self::collect_assigned(&body.stmts, out),
                // A match arm's body is an `Expr`, not a `Block` - the parser
                // builds no `Expr::BlockExpr` - so an arm cannot contain an
                // assignment. It can still contain a CALL, which is why
                // `stmts_take_reference` walks arms and this does not.
                Stmt::Match { .. } => {}
                // None of these can write a tracked integer scalar. The one
                // way they could - handing a callee a reference to one - is
                // refused for the whole body by `stmts_take_reference` before
                // any of this runs, which is what makes leaving them empty a
                // stated assumption rather than a silent one.
                Stmt::Expr(_)
                | Stmt::Return(..)
                | Stmt::Break { .. }
                | Stmt::TypeAlias { .. }
                | Stmt::CompileTimeAssert { .. } => {}
            }
        }
    }

    /// Gives each tracked name in `touched` a fresh unconstrained version.
    fn havoc(
        touched: &std::collections::HashSet<String>,
        versions: &mut HashMap<String, usize>,
        declarations: &mut Vec<String>,
    ) {
        for name in touched {
            if let Some(cur) = versions.get(name).cloned() {
                let next = cur + 1;
                versions.insert(name.clone(), next);
                declarations.push(format!("(declare-const {}_{} Int)", name, next));
            }
        }
    }

    /// A write through an alias has no reliable name-based range update.
    fn invalidate_aliased_intervals(&mut self) {
        for scope in &mut self.scopes {
            for entry in scope.symbols.values_mut() {
                entry.interval = None;
            }
        }
    }

    /// References can also be carried by arrays, structs, and enum payloads.
    /// Follow named types with a cycle guard rather than treating an aggregate as
    /// an independent scalar just because it has no `&` at the call site.
    fn type_contains_reference(&self, ty: &SemanticType) -> bool {
        fn visit(checker: &TypeChecker, ty: &SemanticType,
                 seen: &mut std::collections::HashSet<String>) -> bool {
            match ty {
                SemanticType::Reference { .. } => true,
                SemanticType::Array { element, .. } | SemanticType::BlockTile { element, .. }
                | SemanticType::Vector(element, _) => visit(checker, element, seen),
                SemanticType::Primitive(name) if seen.insert(name.clone()) => {
                    checker.structs.get(name).is_some_and(|fields|
                        fields.values().any(|field| visit(checker, field, seen)))
                        || checker.enums.get(name).is_some_and(|decl|
                            decl.variants.iter().any(|variant|
                                checker.functions.get(&format!("{name}_{}", variant.name))
                                    .is_some_and(|signature| signature.params.iter()
                                        .any(|field| visit(checker, field, seen)))))
                }
                _ => false,
            }
        }
        visit(self, ty, &mut std::collections::HashSet::new())
    }

    /// Whether an expression creates OR carries a reference. Inspect types
    /// as well as syntax: `bump(saved_reference)` can mutate exactly the same
    /// caller state as `bump(&mut x)`.
    fn takes_reference(&self, expr: &Expr) -> bool {
        if self.known_expr_type(expr).is_some_and(|ty| self.type_contains_reference(&ty)) {
            return true;
        }
        match expr {
            Expr::UnaryOp { op: UnaryOp::Ref { .. }, .. } => true,
            Expr::UnaryOp { operand, .. } => self.takes_reference(operand),
            Expr::BinaryOp { left, right, .. } => {
                self.takes_reference(left) || self.takes_reference(right)
            }
            Expr::Call { func, args, .. } => {
                self.takes_reference(func) || args.iter().any(|e| self.takes_reference(e))
            }
            // `func` was not visited here, only `args`. A callee named by an
            // expression that itself hands out a reference is exotic, but the
            // asymmetry with `Expr::Call` one arm up was an oversight, not a
            // decision.
            Expr::GenericCall { func, args, .. } => {
                self.takes_reference(func) || args.iter().any(|e| self.takes_reference(e))
            }
            // Indexing DEREFERENCES: `a[i]` with `a: &mut [T; N]` reads or
            // writes an element and hands no reference to anyone, and an array
            // element is never one of the tracked integer scalars. So a
            // reference-to-array base counts only if its ELEMENTS carry a
            // reference. Without this every loop over an array reached through
            // a reference was refused - the arrays `a[i]` is now bounds-checked
            // through (`tests/reference_array_bounds.rs`).
            Expr::Index { base, index, .. } => {
                let through_array_reference = match &**base {
                    Expr::Ident(name, _) => match self.lookup_var(name) {
                        Some(SemanticType::Reference { inner, .. }) => match &**inner {
                            SemanticType::Array { element, .. } => {
                                Some(self.type_contains_reference(element))
                            }
                            _ => None,
                        },
                        _ => None,
                    },
                    _ => None,
                };
                through_array_reference.unwrap_or_else(|| self.takes_reference(base))
                    || self.takes_reference(index)
            }
            Expr::MemberAccess { base, .. } => self.takes_reference(base),
            Expr::StructLit { fields, .. } => {
                fields.iter().any(|(_, e)| self.takes_reference(e))
            }
            Expr::BlockExpr(b, _) => self.stmts_take_reference(&b.stmts),
            Expr::Ident(..)
            | Expr::IntLit(..)
            | Expr::FloatLit(..)
            | Expr::StringLit(..)
            | Expr::CharLit(..)
            | Expr::BoolLit(..)
            | Expr::Path { .. }
            | Expr::SelfLit(..)
            | Expr::ZeroInit(..) => false,
        }
    }

    /// The same question asked of a whole statement tree.
    ///
    /// `takes_reference` was consulted at exactly ONE site - the top-level
    /// `Stmt::Expr` arm of `trace_body_statements` - while the assumption it
    /// enforces has to hold of the ENTIRE loop body. Everywhere else the
    /// reference was invisible, and each of these compiled clean with
    /// "Compilation Successful!" and the invariant "verified" against a
    /// variable the callee was free to overwrite:
    ///
    /// ```text
    /// if i >= 0 { bump(&i); }     // inside a branch
    /// y = bump(&i);               // an assignment's right-hand side
    /// let y: I32 = bump(&i);      // a `let` initialiser
    /// match i { _ => bump(&i) }   // a match arm
    /// for j in 0..2 { bump(&i); } // a nested loop's body
    /// ```
    ///
    /// The top-level arm rejected the first of those when the `if` was removed,
    /// which is the same one-level-deep shape as the `Stmt::If` bug that
    /// `nested_if_cannot_hide_a_false_invariant` pins.
    ///
    /// Asked of the FULL body, never of the slice: `slice_body_for_invariant`
    /// drops statements it judges irrelevant to the invariant, and relevance is
    /// computed from names - which is exactly the reasoning a reference
    /// invalidates.
    ///
    /// Exhaustive over `Stmt` with no `_ =>` arm, so a new statement kind is a
    /// compile error here rather than an unvisited subtree.
    fn stmts_take_reference(&self, stmts: &[Stmt]) -> bool {
        stmts.iter().any(|stmt| match stmt {
            Stmt::Let { init, .. } => init.as_ref().is_some_and(|e| self.takes_reference(e)),
            Stmt::Assign { target, value, .. }
            | Stmt::CompoundAssign { target, value, .. } => {
                self.takes_reference(target) || self.takes_reference(value)
            }
            Stmt::Expr(e) => self.takes_reference(e),
            Stmt::Return(e, _) => e.as_ref().is_some_and(|e| self.takes_reference(e)),
            Stmt::If { condition, then_block, else_block, .. } => {
                self.takes_reference(condition)
                    || self.stmts_take_reference(&then_block.stmts)
                    || else_block
                        .as_ref()
                        .is_some_and(|b| self.stmts_take_reference(&b.stmts))
            }
            Stmt::For { start, end, step, body, .. } => {
                self.takes_reference(start)
                    || self.takes_reference(end)
                    || step.as_ref().is_some_and(|e| self.takes_reference(e))
                    || self.stmts_take_reference(&body.stmts)
            }
            Stmt::While { condition, body, .. } => {
                self.takes_reference(condition) || self.stmts_take_reference(&body.stmts)
            }
            // The two loop arms above are redundant TODAY and are kept anyway:
            // `check_stmt` requires an `@invariant` on every loop outside an
            // `unsafe` context, so a nested loop is verified in its own right
            // and catches its own body first. Mutation-verified - deleting
            // either traversal leaves the suite green. This check must not
            // depend on a rule a different pass happens to enforce.
            Stmt::Match { scrutinee, arms, .. } => {
                self.takes_reference(scrutinee)
                    || arms.iter().any(|a| self.takes_reference(&a.body))
            }
            Stmt::Chisel(b, _)
            | Stmt::SafeBlock(b, _)
            | Stmt::GhostBlock(b, _)
            | Stmt::HintBlock { body: b, .. } => self.stmts_take_reference(&b.stmts),
            Stmt::ClockDomainBlock { clock, body, .. } => {
                self.takes_reference(clock) || self.stmts_take_reference(&body.stmts)
            }
            Stmt::CompileTimeAssert { condition, .. } => self.takes_reference(condition),
            // Nothing to hand out: `break` has no operands, and a type alias is
            // erased before anything runs.
            Stmt::Break { .. } | Stmt::TypeAlias { .. } => false,
        })
    }

    /// A short name for a statement kind, for diagnostics.
    fn stmt_kind(stmt: &Stmt) -> &'static str {
    match stmt {
            Stmt::Let { .. } => "let",
            Stmt::Assign { .. } => "assignment",
            Stmt::CompoundAssign { .. } => "compound assignment",
            Stmt::If { .. } => "if",
            Stmt::For { .. } => "nested for",
            Stmt::While { .. } => "nested while",
            Stmt::Return(..) => "return",
            Stmt::Expr(..) => "expression",
            Stmt::TypeAlias { .. } => "type alias",
            Stmt::Chisel(..) => "chisel",
            Stmt::SafeBlock(..) => "safe block",
            Stmt::GhostBlock(..) => "ghost block",
            // These produced "a `unsupported` statement" - ungrammatical, and
            // it named neither the construct nor a reason. The message is a
            // user's only handle on why a loop will not verify.
            Stmt::Break { .. } => "break",
            Stmt::Match { .. } => "match",
            Stmt::ClockDomainBlock { .. } => "clock domain block",
            Stmt::CompileTimeAssert { .. } => "compile-time assert",
            Stmt::HintBlock { .. } => "hint block",
        }
    }


    /// Handles an SMT solver that could not be run at all.
    ///
    /// This FAILS THE BUILD by default, and that is a deliberate reversal. It
    /// used to print `[Warning] SMT Solver execution failed` to stdout and
    /// carry on, which meant that on any machine without z3 - the default -
    /// every `@invariant` in every `@safe` block was accepted unchecked. A
    /// deliberately false invariant like `@invariant(i > 1000)` on a `0..10`
    /// loop compiled cleanly and reported "Compilation Successful!".
    ///
    /// An `@invariant` is a proof obligation, not a comment. A `@safe` block
    /// whose obligations were never discharged guarantees nothing, and the
    /// whole point of the annotation is that the guarantee is checkable. Being
    /// unable to check it is a failure, not a detail - the same reasoning that
    /// makes an out-of-range operand unprovable in the ZK backend rather than
    /// silently accepted.
    ///
    /// `Y_ALLOW_UNVERIFIED_INVARIANTS=1` restores the old behaviour for anyone
    /// who genuinely cannot install a solver. It is loud on purpose.
    /// The verifier met a construct it does not model.
    ///
    /// Rejects, and says what it could not model. The alternative - skipping
    /// the construct and asking Z3 anyway - answers a question about a
    /// different program and reports the answer as a verified invariant. That
    /// is how `if i >= 0 { i = i - 100; }` came to satisfy `@invariant(i >= 0)`
    /// while the identical `i = i - 100;` was correctly rejected.
    ///
    /// The rule this encodes, for any future soundness-critical pass: an
    /// unhandled AST node must reject, never be treated as identity or a no-op.
    /// Silent approximation produces the paperwork of a proof without the proof.
    fn smt_unmodellable(&mut self, line: usize, invariant: &Expr, what: &str) {
        if std::env::var("Y_ALLOW_UNVERIFIED_INVARIANTS").is_ok() {
            self.unverified = Some(format!("the verifier cannot model it: {}", what));
            println!(
                "[Warning] invariant `{}` was NOT verified: {}.",
                expr_to_string(invariant),
                what
            );
            return;
        }
        self.errors.push(format!(
            "Line {}: [Strict Safety] Cannot verify invariant `{}`: {}.\n  The verifier refuses \
to reason about a construct it does not model, because skipping it would make the proof \
obligation strictly easier and could accept an invariant that is false.\n  Rewrite the loop \
body using constructs the verifier supports, or set Y_ALLOW_UNVERIFIED_INVARIANTS=1 to compile \
with this invariant UNVERIFIED.",
            line,
            expr_to_string(invariant),
            what
        ));
    }

    fn smt_unavailable(&mut self, line: usize, invariant: &Expr, phase: &str, err: &str) {
        if std::env::var("Y_ALLOW_UNVERIFIED_INVARIANTS").is_ok() {
            self.unverified = Some(format!("the SMT solver could not be run ({} check)", phase));
            println!(
                "[Warning] SMT solver unavailable; invariant `{}` was NOT verified ({} check). {}",
                expr_to_string(invariant),
                phase,
                err
            );
            return;
        }
        self.errors.push(format!(
            "Line {}: [Strict Safety] Could not verify invariant `{}` ({} check) because the \
SMT solver could not be run.\n  {}\n  An @invariant is a proof obligation - accepting it \
unchecked would make @safe guarantee nothing on this machine.\n  Install z3 (`pip install \
z3-solver`, or your package manager) or set Y_Z3_PATH to its absolute path.\n  To compile \
anyway with invariants UNVERIFIED, set Y_ALLOW_UNVERIFIED_INVARIANTS=1.",
            line,
            expr_to_string(invariant),
            phase,
            err
        ));
    }

    fn verify_smt_requirements(
        &mut self, declarations: &[String], preconditions: &[String], assumption: &str,
        requirements: &[String], invariant: &Expr, span: &Span,
    ) -> bool {
        if requirements.is_empty() { return true; }
        let query = format!("{}\n{}\n(assert {})\n(assert (not {}))\n(check-sat)\n",
            declarations.join("\n"), preconditions.join("\n"), assumption,
            Self::smt_all(requirements, "true"));
        match run_z3(&query) {
            Ok(result) if result == "unsat" => true,
            Ok(result) => {
                self.errors.push(format!("Line {}: [SMT Safety Verification Failed] Loop condition arithmetic is not provably representable for invariant `{}`. Z3 returned: {}",
                    span.line, expr_to_string(invariant), result));
                false
            }
            Err(e) => {
                self.smt_unavailable(span.line, invariant, "condition arithmetic", &e);
                false
            }
        }
    }

    fn verify_while_loop_invariant(
        &mut self,
        condition: &Expr,
        body: &Block,
        invariant: &Expr,
        entry_intervals: &HashMap<String, Interval>,
        span: &Span,
    ) {
        // The SMT model rests on one assumption: no callee can write a
        // caller's local integer scalar unless it is handed a reference. That
        // has to be checked of the WHOLE body, and of the unsliced body -
        // `slice_body_for_invariant` decides relevance from names, which is
        // precisely the reasoning a reference invalidates.
        if self.stmts_take_reference(&body.stmts) {
            return self.smt_unmodellable(
                span.line,
                invariant,
                "the loop body passes a reference to a call, which could modify a \
tracked variable in a way this verifier cannot see",
            );
        }

        let mut vars = std::collections::HashSet::new();
        for frame in &self.scopes {
            for (name, entry) in &frame.symbols {
                if let SemanticType::Primitive(prim_name) = &entry.ty {
                    if matches!(prim_name.to_ascii_lowercase().as_str(), "i32" | "u32" | "usize" | "i64") {
                        vars.insert(name.clone());
                    }
                }
            }
        }

        // --- 1. CHECK INITIATION ---
        let mut decls_init = Vec::new();
        let mut preconditions_init = Vec::new();
        // The snapshot goes HERE and only here - see
        // `generate_smt_decls_and_preconditions_with`.
        self.generate_smt_decls_and_preconditions_with(
            &vars,
            Some(entry_intervals),
            &mut decls_init,
            &mut preconditions_init,
        );

        let mut versions_init = std::collections::HashMap::new();
        for var in &vars {
            versions_init.insert(var.clone(), 0);
        }
        let mut init_requirements = Vec::new();
        let inv_init_smt = match self.expr_to_smt(invariant, &versions_init, &mut init_requirements) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        let inv_init_smt = Self::smt_all(&init_requirements, &inv_init_smt);
        decls_init.sort();
        decls_init.dedup();

        let query_init = format!(
            "{}\n{}\n(assert (not {}))\n(check-sat)\n",
            decls_init.join("\n"),
            preconditions_init.join("\n"),
            inv_init_smt
        );

        match run_z3(&query_init) {
            Ok(result) => {
                if result != "unsat" {
                    self.errors.push(format!(
                        "Line {}: [SMT Safety Verification Failed] Loop invariant initiation check failed. Invariant `{}` may not hold on loop entry. Z3 returned: {}",
                        span.line, expr_to_string(invariant), result
                    ));
                    return;
                }
            }
            Err(e) => {
                self.smt_unavailable(span.line, invariant, "initiation", &e);
            }
        }

        // --- 2. CHECK PRESERVATION ---
        let mut decls_pres = Vec::new();
        let mut preconditions_pres = Vec::new();
        self.generate_smt_decls_and_preconditions(&vars, &mut decls_pres, &mut preconditions_pres);

        let mut invariant_requirements = Vec::new();
        let inv_start_smt = match self.expr_to_smt(invariant, &versions_init, &mut invariant_requirements) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };
        let inv_start_smt = Self::smt_all(&invariant_requirements, &inv_start_smt);
        let mut condition_requirements = Vec::new();
        let cond_start_smt = match self.expr_to_smt(condition, &versions_init, &mut condition_requirements) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        if !self.verify_smt_requirements(&decls_pres, &preconditions_pres, &inv_start_smt,
                                        &condition_requirements, invariant, span) { return; }
        let mut requirements = Vec::new();
        let mut versions_pres = versions_init.clone();
        let mut body_assertions = Vec::new();
        // Encode only the statements that can affect the invariant. On an
        // ordinary loop this changes nothing; on a generated one it is the
        // difference between a query z3 answers and one it never returns
        // from. `None` means the analysis met something it did not fully
        // understand, and the whole body is encoded as before.
        let sliced = Self::slice_body_for_invariant(&body.stmts, invariant, "");
        let to_encode: &[Stmt] = sliced.as_deref().unwrap_or(&body.stmts);
        if let Err(why) =
            self.trace_body_statements(to_encode, &mut versions_pres, &mut decls_pres, &mut body_assertions, &mut requirements)
        {
            return self.smt_unmodellable(span.line, invariant, &why);
        }

        let inv_end_smt = match self.expr_to_smt(invariant, &versions_pres, &mut requirements) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        let inv_end_smt = Self::smt_all(&requirements, &inv_end_smt);
        decls_pres.sort();
        decls_pres.dedup();

        let query_pres = format!(
            "{}\n{}\n(assert {})\n(assert {})\n{}\n(assert (not {}))\n(check-sat)\n",
            decls_pres.join("\n"),
            preconditions_pres.join("\n"),
            inv_start_smt,
            cond_start_smt,
            body_assertions.join("\n"),
            inv_end_smt
        );

        match run_z3(&query_pres) {
            Ok(result) => {
                if result != "unsat" {
                    self.errors.push(format!(
                        "Line {}: [SMT Safety Verification Failed] Loop invariant preservation check failed. Invariant `{}` is not preserved by the loop body. Z3 returned: {}",
                        span.line, expr_to_string(invariant), result
                    ));
                }
            }
            Err(e) => {
                self.smt_unavailable(span.line, invariant, "preservation", &e);
            }
        }
    }

    fn verify_for_loop_invariant(
        &mut self,
        loop_var: &str,
        start: &Expr,
        end: &Expr,
        step: &Option<Expr>,
        body: &Block,
        invariant: &Expr,
        entry_intervals: &HashMap<String, Interval>,
        span: &Span,
    ) {
        // The SMT model rests on one assumption: no callee can write a
        // caller's local integer scalar unless it is handed a reference. That
        // has to be checked of the WHOLE body, and of the unsliced body -
        // `slice_body_for_invariant` decides relevance from names, which is
        // precisely the reasoning a reference invalidates.
        if self.stmts_take_reference(&body.stmts) {
            return self.smt_unmodellable(
                span.line,
                invariant,
                "the loop body passes a reference to a call, which could modify a \
tracked variable in a way this verifier cannot see",
            );
        }

        // LLVM reevaluates a step while PTX captures it before the loop.
        // A shared proof is valid only when header inputs stay unchanged.
        let mut writes = std::collections::HashSet::new();
        Self::collect_assigned(&body.stmts, &mut writes);
        writes.insert(loop_var.to_string());
        for expression in std::iter::once(start).chain(std::iter::once(end)).chain(step.iter()) {
            let mut reads = std::collections::HashSet::new();
            if !Self::collect_reads(expression, &mut reads) || !reads.is_disjoint(&writes) {
                return self.smt_unmodellable(span.line, invariant,
                    "a for-loop bound or step depends on a variable changed by the loop");
            }
        }

        let mut vars = std::collections::HashSet::new();
        for frame in &self.scopes {
            for (name, entry) in &frame.symbols {
                if let SemanticType::Primitive(prim_name) = &entry.ty {
                    if matches!(prim_name.to_ascii_lowercase().as_str(), "i32" | "u32" | "usize" | "i64") {
                        vars.insert(name.clone());
                    }
                }
            }
        }
        vars.insert(loop_var.to_string());

        // --- 1. CHECK INITIATION ---
        let mut decls_init = Vec::new();
        let mut preconditions_init = Vec::new();
        // The snapshot goes HERE and only here - see
        // `generate_smt_decls_and_preconditions_with`.
        let mut entry_vars = vars.clone();
        entry_vars.remove(loop_var);
        // The inferred loop range describes executed iterations, not entry:
        // it can be empty, or exclude the initializer before an unsafe cast.
        decls_init.push(format!("(declare-const {loop_var}_0 Int)"));
        self.generate_smt_decls_and_preconditions_with(
            &entry_vars,
            Some(entry_intervals),
            &mut decls_init,
            &mut preconditions_init,
        );

        let mut init_requirements = Vec::new();
        let start_smt = match self.expr_to_smt(start, &std::collections::HashMap::new(), &mut init_requirements) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };
        init_requirements.push(Self::smt_in_range(&start_smt, 32));
        preconditions_init.push(format!("(assert (= {}_{} {}))", loop_var, 0, start_smt));

        let mut versions_init = std::collections::HashMap::new();
        for var in &vars {
            versions_init.insert(var.clone(), 0);
        }
        let inv_init_smt = match self.expr_to_smt(invariant, &versions_init, &mut init_requirements) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        let inv_init_smt = Self::smt_all(&init_requirements, &inv_init_smt);
        decls_init.sort();
        decls_init.dedup();

        let query_init = format!(
            "{}\n{}\n(assert (not {}))\n(check-sat)\n",
            decls_init.join("\n"),
            preconditions_init.join("\n"),
            inv_init_smt
        );

        match run_z3(&query_init) {
            Ok(result) => {
                if result != "unsat" {
                    self.errors.push(format!(
                        "Line {}: [SMT Safety Verification Failed] Loop invariant initiation check failed. Invariant `{}` may not hold on loop entry. Z3 returned: {}",
                        span.line, expr_to_string(invariant), result
                    ));
                    return;
                }
            }
            Err(e) => {
                self.smt_unavailable(span.line, invariant, "initiation", &e);
            }
        }

        // --- 2. CHECK PRESERVATION ---
        let mut decls_pres = Vec::new();
        let mut preconditions_pres = Vec::new();
        self.generate_smt_decls_and_preconditions(&vars, &mut decls_pres, &mut preconditions_pres);

        let mut invariant_requirements = Vec::new();
        let inv_start_smt = match self.expr_to_smt(invariant, &versions_init, &mut invariant_requirements) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        let inv_start_smt = Self::smt_all(&invariant_requirements, &inv_start_smt);
        let mut condition_requirements = Vec::new();
        let loop_var_start_smt = match self.expr_to_smt(start, &versions_init, &mut condition_requirements) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };
        let loop_var_end_smt = match self.expr_to_smt(end, &versions_init, &mut condition_requirements) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };
        // The induction variable is I32 in both backends, and their loop
        // comparisons are signed. A runtime negative endpoint therefore
        // describes an empty loop when start >= end; it is not an assumed
        // nonnegative extent. Wider header expressions still have to fit
        // I32 so the mathematical comparison describes the emitted bits.
        condition_requirements.push(Self::smt_in_range(&loop_var_start_smt, 32));
        condition_requirements.push(Self::smt_in_range(&loop_var_end_smt, 32));
        let cond_start_smt = format!(
            "(and (>= {}_{} {}) (< {}_{} {}))",
            loop_var, 0, loop_var_start_smt, loop_var, 0, loop_var_end_smt
        );

        if !self.verify_smt_requirements(&decls_pres, &preconditions_pres, &inv_start_smt,
                                        &condition_requirements, invariant, span) { return; }
        let mut requirements = Vec::new();
        let mut versions_pres = versions_init.clone();
        let mut body_assertions = Vec::new();
        // Encode only the statements that can affect the invariant. On an
        // ordinary loop this changes nothing; on a generated one it is the
        // difference between a query z3 answers and one it never returns
        // from. `None` means the analysis met something it did not fully
        // understand, and the whole body is encoded as before.
        let sliced = Self::slice_body_for_invariant(&body.stmts, invariant, loop_var);
        let to_encode: &[Stmt] = sliced.as_deref().unwrap_or(&body.stmts);
        if let Err(why) =
            self.trace_body_statements(to_encode, &mut versions_pres, &mut decls_pres, &mut body_assertions, &mut requirements)
        {
            return self.smt_unmodellable(span.line, invariant, &why);
        }

        let current_loop_var_ver = versions_pres.get(loop_var).cloned().unwrap_or(0);
        let next_loop_var_ver = current_loop_var_ver + 1;
        decls_pres.push(format!("(declare-const {}_{} Int)", loop_var, next_loop_var_ver));

        let step_smt = if let Some(st) = step {
            match self.expr_to_smt(st, &versions_pres, &mut requirements) {
                Ok(v) => v,
                Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
            }
        } else {
            "1".to_string()
        };
        // The inferred lower bound i >= start is valid only for ascending
        // loops. A negative dynamic step invalidates that induction premise.
        requirements.push(format!("(> {step_smt} 0)"));
        requirements.push(Self::smt_in_range(&step_smt, 32));
        requirements.push(Self::smt_in_range(&format!("(+ {loop_var}_{current_loop_var_ver} {step_smt})"), 32));
        // Body writes to the induction variable must preserve the lower
        // bound used by the next iteration's proof, too.
        requirements.push(format!("(>= {loop_var}_{next_loop_var_ver} {loop_var_start_smt})"));
        versions_pres.insert(loop_var.to_string(), next_loop_var_ver);
        body_assertions.push(format!(
            "(assert (= {}_{} (+ {}_{} {})))",
            loop_var, next_loop_var_ver, loop_var, current_loop_var_ver, step_smt
        ));

        let inv_end_smt = match self.expr_to_smt(invariant, &versions_pres, &mut requirements) {
            Ok(v) => v,
            Err(why) => return self.smt_unmodellable(span.line, invariant, &why),
        };

        let inv_end_smt = Self::smt_all(&requirements, &inv_end_smt);
        decls_pres.sort();
        decls_pres.dedup();

        let query_pres = format!(
            "{}\n{}\n(assert {})\n(assert {})\n{}\n(assert (not {}))\n(check-sat)\n",
            decls_pres.join("\n"),
            preconditions_pres.join("\n"),
            inv_start_smt,
            cond_start_smt,
            body_assertions.join("\n"),
            inv_end_smt
        );

        match run_z3(&query_pres) {
            Ok(result) => {
                if result != "unsat" {
                    self.errors.push(format!(
                        "Line {}: [SMT Safety Verification Failed] Loop invariant preservation check failed. Invariant `{}` is not preserved by the loop body. Z3 returned: {}",
                        span.line, expr_to_string(invariant), result
                    ));
                }
            }
            Err(e) => {
                self.smt_unavailable(span.line, invariant, "preservation", &e);
            }
        }
    }
}

/// Z3 binaries to try, in priority order.
///
/// `Y_Z3_PATH` wins when set, then bare `z3` from `PATH`. The rest exist
/// because it is easy to have a perfectly good solver installed that the
/// compiler cannot see: a `pip install z3-solver` inside a project virtualenv
/// puts one in `venv/bin/z3`, which the old two-entry search missed - this
/// repo had exactly that, while the type checker reported the solver as
/// missing and waved every invariant through.
pub fn z3_candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("Y_Z3_PATH") {
        if !p.is_empty() {
            v.push(p);
        }
    }
    v.push("z3".to_string());
    for p in [
        "venv/bin/z3",
        "./venv/bin/z3",
        ".venv/bin/z3",
        "./.venv/bin/z3",
        "./z3/build/z3",
        "z3/build/z3",
    ] {
        v.push(p.to_string());
    }
    if let Ok(home) = std::env::var("HOME") {
        v.push(format!("{}/.local/bin/z3", home));
    }
    v
}

/// Declares any `name_version` symbol the query REFERENCES but never declares.
///
/// A loop whose bounds are variables (`for k in ks..e`) puts `ks_0` into the
/// assertions, but `ks` is not one of the tracked variables the declaration
/// pass walks, so z3 got an unknown constant and exited 1 - which the caller
/// then reported as "the SMT solver could not be run". The solver ran fine;
/// it was handed a malformed query. Literal bounds never hit this, which is
/// why every existing test passed.
///
/// Declaring the symbol unconstrained is the sound direction: the invariant
/// must then hold for ANY value it could have taken, exactly as with the
/// havoc used for branches. It can only make an obligation harder to
/// discharge, never easier.
/// Canonical SMT symbol for a GPU index intrinsic, if it is one.
/// Hardware-guaranteed range of a GPU index intrinsic, as a CUDA launch limit.
fn gpu_index_interval(name: &str) -> Option<Interval> {
    let (min, max) = match name {
        "thread_idx_x" | "thread_idx_y" => (0, 1023),
        "thread_idx_z" => (0, 63),
        "block_dim_x" | "block_dim_y" => (1, 1024),
        "block_dim_z" => (1, 64),
        "block_idx_x" => (0, 2_147_483_646),
        "block_idx_y" | "block_idx_z" => (0, 65_534),
        "grid_dim_x" => (1, 2_147_483_647),
        "grid_dim_y" | "grid_dim_z" => (1, 65_535),
        _ => return None,
    };
    Some(Interval { min, max, trust: LAUNCH_LIMITS })
}

fn gpu_index_symbol(name: &str) -> Option<&'static str> {
    Some(match name {
        "thread_idx_x" => "gpuTidX",
        "thread_idx_y" => "gpuTidY",
        "thread_idx_z" => "gpuTidZ",
        "block_idx_x" => "gpuCtaX",
        "block_idx_y" => "gpuCtaY",
        "block_idx_z" => "gpuCtaZ",
        "block_dim_x" => "gpuNtidX",
        "block_dim_y" => "gpuNtidY",
        "block_dim_z" => "gpuNtidZ",
        "grid_dim_x" => "gpuNctaX",
        "grid_dim_y" => "gpuNctaY",
        "grid_dim_z" => "gpuNctaZ",
        _ => return None,
    })
}

/// The hardware guarantee for a GPU index symbol, as an SMT assertion.
///
/// **Only lower bounds.** Every fact asserted here makes a proof obligation
/// easier, which is the direction the design rule in `CLAUDE.md` warns about,
/// so the set is kept to what the hardware actually promises and to the
/// minimum that lets ordinary kernels verify: indices are non-negative, and
/// extents are at least one (a launch with a zero dimension is rejected by the
/// driver, not executed). Upper bounds would also be true but are not needed
/// by anything, and each one is another chance to be wrong in the unsafe
/// direction.
fn gpu_index_bound(sym: &str) -> Option<String> {
    let lower = match sym {
        "gpuTidX" | "gpuTidY" | "gpuTidZ" | "gpuCtaX" | "gpuCtaY" | "gpuCtaZ" => 0,
        "gpuNtidX" | "gpuNtidY" | "gpuNtidZ" | "gpuNctaX" | "gpuNctaY" | "gpuNctaZ" => 1,
        _ => return None,
    };
    Some(format!("(assert (>= {}_0 {}))", sym, lower))
}

fn declare_free_symbols(query: &str) -> String {
    let mut declared = std::collections::HashSet::new();
    for line in query.lines() {
        if let Some(rest) = line.trim().strip_prefix("(declare-const ") {
            if let Some(name) = rest.split_whitespace().next() {
                declared.insert(name.to_string());
            }
        }
    }
    let mut missing: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for line in query.lines() {
        if line.trim().starts_with("(declare-const ") {
            continue;
        }
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i] as char;
            if c.is_ascii_alphabetic() || c == '_' {
                let start = i;
                while i < bytes.len() {
                    let d = bytes[i] as char;
                    if d.is_ascii_alphanumeric() || d == '_' {
                        i += 1;
                    } else {
                        break;
                    }
                }
                let tok = &line[start..i];
                // Only `name_<digits>`, the shape the version-mangler emits.
                let versioned = tok
                    .rsplit_once('_')
                    .map_or(false, |(h, t)| !h.is_empty() && !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit()));
                if versioned && !declared.contains(tok) && seen.insert(tok.to_string()) {
                    missing.push(tok.to_string());
                }
            } else {
                i += 1;
            }
        }
    }
    if missing.is_empty() {
        return query.to_string();
    }
    missing.sort();
    let mut out = String::new();
    for m in missing {
        out.push_str(&format!("(declare-const {} Int)\n", m));
        // A GPU index symbol is free, but not arbitrary.
        if let Some(sym) = m.strip_suffix("_0") {
            if let Some(bound) = gpu_index_bound(sym) {
                out.push_str(&bound);
                out.push('\n');
            }
        }
    }
    out.push_str(query);
    out
}

fn run_z3(query: &str) -> Result<String, String> {
    let candidates = z3_candidates();
    let mut spawned = None;
    for cand in &candidates {
        let attempt = Command::new(cand)
            .args(&["-smt2", "-in"])
            .env("Z3_GPU_THRESHOLD", "2147483647")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        if let Ok(child) = attempt {
            spawned = Some(child);
            break;
        }
    }
    let mut child = spawned.ok_or_else(|| {
        format!(
            "No Z3 binary could be started. Searched: {}",
            candidates.join(", ")
        )
    })?;

    {
        let stdin = child.stdin.as_mut().ok_or("Failed to open stdin")?;
        let query = declare_free_symbols(query);
        stdin.write_all(query.as_bytes()).map_err(|e| e.to_string())?;
    }

    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr).to_string();
        let code = output.status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".to_string());
        return Err(format!("Z3 error (code {}): {}\nQuery:\n{}", code, err, query));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn expr_to_string(expr: &Expr) -> String {
    match expr {
        Expr::Ident(name, _) => name.clone(),
        Expr::IntLit(val, _) => val.to_string(),
        Expr::FloatLit(val, _) => val.to_string(),
        Expr::StringLit(val, _) => format!("\"{}\"", val),
        Expr::CharLit(val, _) => format!("'{}'", val),
        Expr::BoolLit(val, _) => val.to_string(),
        Expr::BinaryOp { left, op, right, .. } => {
            let op_str = match op {
                BinaryOp::Add => "+",
                BinaryOp::Sub => "-",
                BinaryOp::Mul => "*",
                BinaryOp::Div => "/",
                BinaryOp::Mod => "%",
                BinaryOp::Eq => "==",
                BinaryOp::NotEq => "!=",
                BinaryOp::Lt => "<",
                BinaryOp::Gt => ">",
                BinaryOp::Le => "<=",
                BinaryOp::Ge => ">=",
                BinaryOp::And => "&&",
                BinaryOp::Or => "||",
                BinaryOp::BitAnd => "&",
                BinaryOp::BitOr => "|",
                BinaryOp::BitXor => "^",
                BinaryOp::Shl => "<<",
                BinaryOp::Shr => ">>",
            };
            format!("({} {} {})", expr_to_string(left), op_str, expr_to_string(right))
        }
        Expr::UnaryOp { op, operand, .. } => {
            let op_str = match op {
                UnaryOp::Neg => "-",
                UnaryOp::Not => "!",
                UnaryOp::Ref { mutable } => {
                    if *mutable {
                        "&mut "
                    } else {
                        "&"
                    }
                }
                UnaryOp::Deref => "*",
            };
            format!("{}{}", op_str, expr_to_string(operand))
        }
        _ => format!("{:?}", expr),
    }
}

fn is_shared_resource(ty: &SemanticType) -> bool {
    match ty {
        SemanticType::Array { .. } => true,
        SemanticType::SharedMemoryTile { .. } => true,
        SemanticType::GlobalMemory(_) => true,
        SemanticType::Primitive(name) => name == "ptr",
        _ => false,
    }
}

struct CoherenceAnalyzer<'a> {
    type_checker: &'a TypeChecker,
    segments: Vec<BarrierSegment>,
    current_segment: BarrierSegment,
}

#[derive(Clone, Default)]
struct BarrierSegment {
    reads: std::collections::HashMap<String, Span>,
    writes: std::collections::HashMap<String, Span>,
    barrier_span: Option<Span>,
}

impl<'a> CoherenceAnalyzer<'a> {
    fn analyze_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.analyze_stmt(stmt);
        }
    }

    fn collect_reads_writes(&mut self, expr: &Expr, is_write: bool) {
        self.collect_expr_accesses(expr, is_write);
    }

    fn collect_expr_accesses(&mut self, expr: &Expr, is_write: bool) {
        match expr {
            Expr::Ident(name, span) => {
                if let Some(ty) = self.type_checker.lookup_var(name) {
                    if is_shared_resource(ty) {
                        if is_write {
                            self.current_segment.writes.insert(name.clone(), span.clone());
                        } else {
                            self.current_segment.reads.insert(name.clone(), span.clone());
                        }
                    }
                }
            }
            Expr::Index { base, index, .. } => {
                self.collect_expr_accesses(base, is_write);
                self.collect_expr_accesses(index, false);
            }
            Expr::MemberAccess { base, .. } => {
                self.collect_expr_accesses(base, is_write);
            }
            Expr::BinaryOp { left, right, .. } => {
                self.collect_expr_accesses(left, false);
                self.collect_expr_accesses(right, false);
            }
            Expr::UnaryOp { operand, .. } => {
                self.collect_expr_accesses(operand, is_write);
            }
            Expr::Call { func, args, .. } => {
                if let Expr::Ident(fname, _) = &**func {
                    if fname == "cp_async" && args.len() >= 2 {
                        self.collect_expr_accesses(&args[0], false);
                        self.collect_expr_accesses(&args[1], true);
                    } else if fname == "store" && args.len() >= 2 {
                        self.collect_expr_accesses(&args[0], true);
                        self.collect_expr_accesses(&args[1], false);
                    } else if (fname == "load" || fname == "ldmatrix") && !args.is_empty() {
                        self.collect_expr_accesses(&args[0], false);
                    } else if fname == "mma_sync" {
                        for arg in args {
                            self.collect_expr_accesses(arg, false);
                        }
                    } else {
                        for arg in args {
                            self.collect_expr_accesses(arg, false);
                        }
                    }
                } else {
                    self.collect_expr_accesses(func, false);
                    for arg in args {
                        self.collect_expr_accesses(arg, false);
                    }
                }
            }
            Expr::GenericCall { func, args, .. } => {
                self.collect_expr_accesses(func, false);
                for arg in args {
                    self.collect_expr_accesses(arg, false);
                }
            }
            Expr::StructLit { fields, .. } => {
                for (_, f_expr) in fields {
                    self.collect_expr_accesses(f_expr, false);
                }
            }
            _ => {}
        }
    }

    fn analyze_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { init, .. } => {
                if let Some(init_expr) = init {
                    self.collect_reads_writes(init_expr, false);
                }
            }
            Stmt::Assign { target, value, .. } => {
                self.collect_reads_writes(value, false);
                self.collect_reads_writes(target, true);
            }
            Stmt::Expr(expr) => {
                let is_barrier = match expr {
                    Expr::Path { namespace, member, .. } => namespace == "barrier" && member == "sync",
                    Expr::Call { func, .. } => match &**func {
                        Expr::Path { namespace, member, .. } => namespace == "barrier" && member == "sync",
                        Expr::Ident(fname, _) => fname == "membar" || fname == "barrier_sync",
                        _ => false,
                    },
                    _ => false,
                };
                
                if is_barrier {
                    let prev_segment = std::mem::take(&mut self.current_segment);
                    self.segments.push(prev_segment);
                    self.current_segment = BarrierSegment {
                        reads: std::collections::HashMap::new(),
                        writes: std::collections::HashMap::new(),
                        barrier_span: Some(expr.span()),
                    };
                } else {
                    self.collect_reads_writes(expr, false);
                }
            }
            Stmt::For { body, start, end, .. } => {
                self.collect_reads_writes(start, false);
                self.collect_reads_writes(end, false);
                self.analyze_block(body);
            }
            Stmt::While { body, condition, .. } => {
                self.collect_reads_writes(condition, false);
                self.analyze_block(body);
            }
            Stmt::If { condition, then_block, else_block, .. } => {
                self.collect_reads_writes(condition, false);
                self.analyze_block(then_block);
                if let Some(el) = else_block {
                    self.analyze_block(el);
                }
            }
            _ => {}
        }
    }
}

impl TypeChecker {
    fn verify_kernel_coherence(&mut self, kernel: &KernelDecl) {
        let segments = {
            let mut analyzer = CoherenceAnalyzer {
                type_checker: self,
                segments: Vec::new(),
                current_segment: BarrierSegment::default(),
            };

            analyzer.analyze_block(&kernel.body);
            analyzer.segments.push(analyzer.current_segment);
            analyzer.segments
        };

        for (idx, segment) in segments.iter().enumerate() {
            // 1. Check RAW / WAR hazards (read and write to same variable on different lines)
            for (var_name, read_span) in &segment.reads {
                if let Some(write_span) = segment.writes.get(var_name) {
                    if read_span.line != write_span.line {
                        let second_line = std::cmp::max(read_span.line, write_span.line);
                        self.errors.push(format!(
                            "Line {}: [Coherence Hazard] Read-After-Write (or Write-After-Read) hazard detected on shared/global memory `{}`. Accesses at line {} and line {} are not separated by a `barrier::sync()`.",
                            second_line, var_name, read_span.line, write_span.line
                        ));
                    }
                }
            }

            // 2. Check redundant barriers (optimize barrier placement)
            if idx + 1 < segments.len() {
                if let Some(next_barrier_span) = &segments[idx + 1].barrier_span {
                    if segment.writes.is_empty() {
                        println!(
                            "    [Warning] Line {}: [Barrier Optimization] Redundant barrier synchronization. No shared memory writes occurred since the last barrier.",
                            next_barrier_span.line
                        );
                    }
                }
            }
        }
    }

    fn types_are_compatible(&self, t1: &SemanticType, t2: &SemanticType) -> bool {
        if t1 == t2 {
            return true;
        }
        // `Unknown` means "this checker could not type it", and the mismatch
        // check exempts it precisely so an untypeable value is not reported as
        // a WRONG one. That exemption has to survive being placed behind a
        // reference. `String_new` returns a non-scalar the intrinsic registry
        // cannot type, so `&s_str` is `&Unknown`; comparing it against a
        // declared `&String` refused `tests/test_struct.ysu`, a correct
        // program. A reference whose inner types are both KNOWN and different
        // is still a mismatch - that is the `let r: &F32 = &x` case with
        // `x: I32` that `Reference` was introduced to catch, and it is what
        // stops this from reverting `Reference` to the old blanket `Unknown`.
        // The MUTABILITY must still match exactly. Relaxing only the inner
        // type is what keeps `let r: &mut I32 = &x;` refused - a shared borrow
        // does not satisfy a `&mut` annotation, and that is the half of this
        // check the parser's dropped `mut` token used to lose entirely.
        if let (
            SemanticType::Reference { inner: a, mutable: am },
            SemanticType::Reference { inner: b, mutable: bm },
        ) = (t1, t2)
        {
            return am == bm
                && (**a == SemanticType::Unknown
                    || **b == SemanticType::Unknown
                    || self.types_are_compatible(a, b));
        }
        let is_int_or_ptr = |t: &SemanticType| {
            if let SemanticType::Primitive(p) = t {
                let p_lower = p.to_lowercase();
                p_lower == "i8" || p_lower == "i16" || p_lower == "i32" || p_lower == "i64" ||
                p_lower == "u8" || p_lower == "u16" || p_lower == "u32" || p_lower == "u64" ||
                p_lower == "ptr"
            } else {
                false
            }
        };
        is_int_or_ptr(t1) && is_int_or_ptr(t2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_checker_starts_with_clean_state() {
        let tc = TypeChecker::new();

        assert!(tc.errors.is_empty());
        assert!(!tc.in_unsafe);
    }

    #[test]
    fn test_enum_item_does_not_produce_type_errors() {
        let mut tc = TypeChecker::new();
        let program = Program {
            items: vec![Item::Enum(EnumDecl {
                name: "TestEnum".into(),
                generic_params: vec![],
                variants: vec![],
                span: Span { line: 0, col: 0 },
            })],
        };

        tc.check_program(&program);

        assert!(tc.errors.is_empty());
    }

    #[test]
    fn test_eval_interval_div() {
        let tc = TypeChecker::new();

        // 1. Division by interval containing zero -> None
        let expr_zero = Expr::BinaryOp {
            left: Box::new(Expr::IntLit(10, Span { line: 0, col: 0 })),
            op: BinaryOp::Div,
            right: Box::new(Expr::IntLit(0, Span { line: 0, col: 0 })),
            span: Span { line: 0, col: 0 },
        };
        assert!(tc.eval_interval(&expr_zero).is_none());

        // 2. Division by positive divisor interval
        let expr_pos = Expr::BinaryOp {
            left: Box::new(Expr::IntLit(20, Span { line: 0, col: 0 })),
            op: BinaryOp::Div,
            right: Box::new(Expr::IntLit(4, Span { line: 0, col: 0 })),
            span: Span { line: 0, col: 0 },
        };
        let res = tc.eval_interval(&expr_pos).unwrap();
        assert_eq!(res.min, 5);
        assert_eq!(res.max, 5);
    }

    fn parse_src(src: &str) -> Program {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(tokens);
        parser.parse_program().expect("parse should succeed")
    }

    #[test]
    fn test_kernel_level_tile_valid_shape_passes() {
        let program = parse_src(
            "
            @tile(4096, 4096, 4096)
            kernel gemm(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {
                let x: I32 = 0;
            }
            ",
        );
        let mut tc = TypeChecker::new();
        tc.check_program(&program);
        assert!(tc.errors.is_empty(), "unexpected errors: {:?}", tc.errors);
    }

    #[test]
    fn test_kernel_level_tile_fused_bias_relu_shape_passes() {
        let program = parse_src(
            "
            @tile(4096, 4096, 4096)
            kernel fused_gemm(A: GlobalMemory<F16>, B: GlobalMemory<F16>, Bias: GlobalMemory<F32>, C: GlobalMemory<F32>) {
                let x: I32 = 0;
            }
            ",
        );
        let mut tc = TypeChecker::new();
        tc.check_program(&program);
        assert!(tc.errors.is_empty(), "unexpected errors: {:?}", tc.errors);
    }

    #[test]
    fn test_kernel_level_tile_swiglu_shape_passes() {
        let program = parse_src(
            "
            @tile(4096, 4096, 4096)
            kernel fused_swiglu(X: GlobalMemory<F16>, Wgate: GlobalMemory<F16>, Wup: GlobalMemory<F16>, Out: GlobalMemory<F32>) {
                let x: I32 = 0;
            }
            ",
        );
        let mut tc = TypeChecker::new();
        tc.check_program(&program);
        assert!(tc.errors.is_empty(), "unexpected errors: {:?}", tc.errors);
    }

    #[test]
    fn test_kernel_level_tile_rejects_wrong_param_shape() {
        let program = parse_src(
            "
            @tile(4096, 4096, 4096)
            kernel bad_gemm(A: GlobalMemory<F16>, B: GlobalMemory<F16>, M: I32) {
                let x: I32 = 0;
            }
            ",
        );
        let mut tc = TypeChecker::new();
        tc.check_program(&program);
        // Assert the SUBSTANCE, not the phrasing. This used to match on
        // "requires 3 parameters", so rewriting the message to list every
        // accepted shape broke a test that had no opinion about shapes -- the
        // diagnostic was still correct and still rejected the program. What
        // the test actually cares about is that the offending parameter is
        // named and that the message tells the user what IS accepted.
        assert!(
            tc.errors.iter().any(|e| e.contains("@tile")
                && e.contains("M (expected")
                && e.contains("GlobalMemory<F16>")),
            "expected a validation error naming the non-GlobalMemory<F16> param \
             and the accepted shapes, got: {:?}",
            tc.errors
        );
    }

    #[test]
    fn test_kernel_level_tile_rejects_missing_k() {
        let program = parse_src(
            "
            @tile(4096, 4096)
            kernel bad_gemm2(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {
                let x: I32 = 0;
            }
            ",
        );
        let mut tc = TypeChecker::new();
        tc.check_program(&program);
        assert!(
            tc.errors.iter().any(|e| e.contains("requires K")),
            "expected a missing-K validation error, got: {:?}",
            tc.errors
        );
    }
}
