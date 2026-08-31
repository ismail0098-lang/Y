//! A structured generative differential fuzzer for the ZK backend.
//!
//! # Why this exists, when `fuzz/` already had eight targets
//!
//! None of them could have found the four control-flow bugs of 2026-08-16, and
//! the reasons are worth stating because they are the design constraints on
//! this module:
//!
//! - `fuzz_zk_emitter` / `fuzz_zk_soundness` feed **raw bytes** to the lexer.
//!   The chance that random bytes form a program containing an `if` with a
//!   `return` in one arm is negligible, so the interesting fragment was never
//!   reached at all.
//! - `fuzz_differential` was **not differential**: it built a single
//!   `ConstDecl` — no function, no control flow — and then compared nothing.
//!   It asserted the CPU emitter's output was non-empty and logged ZK errors.
//! - `fuzz_zk_soundness` reports every finding with `eprintln!` and never
//!   panics, so libFuzzer cannot see a failure. It could run for a week over a
//!   backend that computes the wrong function and exit 0.
//!
//! The common thread: **they checked that the compiler did not crash, never
//! that it emitted the right circuit.** Every one of the four bugs compiled
//! cleanly, printed "Compilation Successful!", produced a satisfiable circuit,
//! and computed a different function than the source. A crash oracle is blind
//! to all of it.
//!
//! # The three oracles here
//!
//! 1. **An independent interpreter.** [`interpret`] evaluates the generated
//!    program directly from the language semantics. It is written against
//!    `FProgram`, this module's own IR — *not* against Y's AST — so the path
//!    under test (render to source, lex, parse, emit, solve) shares no code
//!    with the oracle. A parser bug is therefore visible; it would not be if
//!    the oracle walked the AST the parser produced.
//!
//! 2. **A metamorphic constant-folding check.** The same program is rendered
//!    twice: once taking its inputs as parameters, once with those inputs
//!    substituted as literals. The emitter takes a *completely different path*
//!    through each — `is_constant()` folding versus gadget emission — and the
//!    two must agree. This oracle needs no reference implementation at all,
//!    which makes it the one that cannot be wrong in the same direction as the
//!    interpreter.
//!
//! 3. **Structural checks.** The solved witness must satisfy the circuit it
//!    came from, and compilation must be deterministic.
//!
//! # Determinism, and why generation is byte-driven
//!
//! [`Entropy`] reads choices from a byte slice. Under `cargo fuzz` those bytes
//! are libFuzzer's, so coverage-guided mutation still has a gradient — mutating
//! a byte changes one choice rather than reseeding a PRNG and rerolling the
//! whole program. Under `cargo test` they come from a seeded counter, so a
//! finding reproduces from its seed alone. One generator serves both, for the
//! same reason `zk_witness` is library code: two implementations that disagree
//! would make both results meaningless.

use crate::zk_emitter::ZkEmitter;
use crate::zk_field::Fr;
use crate::zk_witness::{check_r1cs_satisfiability, solve_r1cs_witness};

/// Gadget operand width. Mirrors `ZK_COMPARISON_BITS` in the emitter.
const COMPARISON_BITS: u32 = 32;

/// `2^32`, as the field element the range check is against.
fn two_pow_32() -> Fr {
    Fr::from_u64(1u64 << COMPARISON_BITS)
}

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

/// A byte-driven source of generator choices.
///
/// Wraps around when exhausted rather than failing, so a short input still
/// yields a complete program. Termination is guaranteed by the generator's
/// depth and statement budgets, never by running out of bytes.
pub struct Entropy<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Entropy<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Entropy { bytes, pos: 0 }
    }

    fn byte(&mut self) -> u8 {
        if self.bytes.is_empty() {
            return 0;
        }
        let b = self.bytes[self.pos % self.bytes.len()];
        self.pos = self.pos.wrapping_add(1);
        b
    }

    /// A choice in `0..n`. `n == 0` is a caller error and yields 0.
    pub fn choose(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        (self.byte() as usize) % n
    }

    pub fn flip(&mut self, percent: u8) -> bool {
        (self.byte() % 100) < percent
    }
}

// ---------------------------------------------------------------------------
// The generated language
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    // The gadget-backed operators. Each lowers to an `emit_num2bits`
    // decomposition of one or both operands, so each costs 33-135 constraints
    // against the 3 an `+` costs - and each carries a range check whose
    // soundness argument is written down in `CLAUDE.md` and was, until now,
    // tested by nothing this generator could write.
    Div,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

impl Op {
    fn spelling(self) -> &'static str {
        match self {
            Op::Add => "+",
            Op::Sub => "-",
            Op::Mul => "*",
            Op::Eq => "==",
            Op::Ne => "!=",
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Gt => ">",
            Op::Ge => ">=",
            Op::And => "&&",
            Op::Or => "||",
            Op::Div => "/",
            Op::Mod => "%",
            Op::BitAnd => "&",
            Op::BitOr => "|",
            Op::BitXor => "^",
            Op::Shl => "<<",
            Op::Shr => ">>",
        }
    }

    /// Operators whose gadget range-checks its operands to 32 bits.
    fn is_ordering(self) -> bool {
        matches!(self, Op::Lt | Op::Le | Op::Gt | Op::Ge)
    }

    /// Operators that require both operands to already be booleans.
    fn is_logical(self) -> bool {
        matches!(self, Op::And | Op::Or)
    }

    /// `/` and `%`, which share one gadget.
    fn is_divmod(self) -> bool {
        matches!(self, Op::Div | Op::Mod)
    }

    /// Bitwise operators, which decompose BOTH operands to 32 bits.
    fn is_bitwise(self) -> bool {
        matches!(self, Op::BitAnd | Op::BitOr | Op::BitXor)
    }

    /// Shifts, which decompose only the shifted value and require the amount
    /// to be a compile-time constant.
    fn is_shift(self) -> bool {
        matches!(self, Op::Shl | Op::Shr)
    }
}

#[derive(Clone, Debug)]
pub enum FExpr {
    Lit(u64),
    /// A function parameter, by index.
    Param(usize),
    /// A `let`-bound local, by index.
    Local(usize),
    /// An enclosing loop's induction variable, by loop nesting index.
    LoopVar(usize),
    Bin(Op, Box<FExpr>, Box<FExpr>),
}

#[derive(Clone, Debug)]
pub enum FStmt {
    Return(FExpr),
    Assign(usize, FExpr),
    If {
        cond: FExpr,
        then_b: Vec<FStmt>,
        else_b: Option<Vec<FStmt>>,
    },
    For {
        depth: usize,
        count: u64,
        body: Vec<FStmt>,
    },
}

/// A generated program: `fn main(p0.., ) -> I32` with `nlocals` top-level
/// `let` bindings and a body.
///
/// Locals are declared only at the top level and assigned anywhere. That keeps
/// lexical scoping trivially agreed between the interpreter and Y — a `let`
/// inside a branch is out of scope after it, and modelling that adds no
/// coverage of the branch-merge logic, which is driven by *assignment*.
#[derive(Clone, Debug)]
pub struct FProgram {
    pub nparams: usize,
    pub locals_init: Vec<u64>,
    pub body: Vec<FStmt>,
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Values worth hitting far more often than uniform sampling would.
///
/// The 32-bit boundary and the values either side of it are where the
/// comparison gadget's range check changes its answer; 0 and 1 are where the
/// booleanity constraints do.
const INTERESTING: &[u64] = &[
    0,
    1,
    2,
    3,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_fffe,
    0xffff_ffff,
    0x1_0000_0000,
    0x1_0000_0001,
];

pub struct GenConfig {
    pub max_depth: usize,
    pub max_stmts: usize,
    pub max_params: usize,
    pub max_locals: usize,
    pub max_nesting: usize,
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig {
            max_depth: 3,
            max_stmts: 4,
            max_params: 3,
            max_locals: 2,
            max_nesting: 3,
        }
    }
}

struct Ctx {
    nparams: usize,
    nlocals: usize,
    loop_depth: usize,
    nesting: usize,
}

pub fn gen_program(e: &mut Entropy, cfg: &GenConfig) -> FProgram {
    let nparams = 1 + e.choose(cfg.max_params);
    let nlocals = e.choose(cfg.max_locals + 1);
    let locals_init = (0..nlocals).map(|_| gen_value(e)).collect();

    let mut ctx = Ctx {
        nparams,
        nlocals,
        loop_depth: 0,
        nesting: 0,
    };
    let mut body = gen_block(e, cfg, &mut ctx);

    // A function must return on every path, so guarantee a trailing `return`.
    // Generating one only sometimes would mean most programs fail to compile
    // for an uninteresting reason.
    body.push(FStmt::Return(gen_expr(e, cfg, &ctx, 0)));

    FProgram {
        nparams,
        locals_init,
        body,
    }
}

fn gen_block(e: &mut Entropy, cfg: &GenConfig, ctx: &mut Ctx) -> Vec<FStmt> {
    let n = e.choose(cfg.max_stmts + 1);
    (0..n).map(|_| gen_stmt(e, cfg, ctx)).collect()
}

fn gen_stmt(e: &mut Entropy, cfg: &GenConfig, ctx: &mut Ctx) -> FStmt {
    // Past the nesting budget, only leaf statements.
    let leaf_only = ctx.nesting >= cfg.max_nesting;
    let choice = if leaf_only { e.choose(2) } else { e.choose(4) };

    match choice {
        0 => FStmt::Return(gen_expr(e, cfg, ctx, 0)),
        1 if ctx.nlocals > 0 => {
            let slot = e.choose(ctx.nlocals);
            let value = gen_expr(e, cfg, ctx, 0);
            // About a third of assignments are built in the `x = x op e`
            // shape, because that is the shape the renderer spells `x op= e`.
            //
            // The generator had no way to reach a compound assignment at all,
            // and `zk_emitter::emit_stmt` was silently dropping every one of
            // them - `let x = a; x += 5; return x;` emitted a circuit
            // computing `out = a`. The bug is precisely what oracle 1 exists
            // to catch and the fuzzer could not produce the program. Same
            // lesson as the `if` condition restriction recorded in
            // `CLAUDE.md`: a generator that cannot write a construct proves
            // nothing about it.
            if e.flip(33) {
                let op = match e.choose(3) {
                    0 => Op::Add,
                    1 => Op::Sub,
                    _ => Op::Mul,
                };
                FStmt::Assign(
                    slot,
                    FExpr::Bin(op, Box::new(FExpr::Local(slot)), Box::new(value)),
                )
            } else {
                FStmt::Assign(slot, value)
            }
        }
        1 => FStmt::Return(gen_expr(e, cfg, ctx, 0)),
        2 => {
            let cond = gen_cond(e, cfg, ctx);
            ctx.nesting += 1;
            let then_b = gen_block(e, cfg, ctx);
            let else_b = if e.flip(40) {
                Some(gen_block(e, cfg, ctx))
            } else {
                None
            };
            ctx.nesting -= 1;
            FStmt::If {
                cond,
                then_b,
                else_b,
            }
        }
        _ => {
            let count = 1 + e.choose(3) as u64;
            let depth = ctx.loop_depth;
            ctx.loop_depth += 1;
            ctx.nesting += 1;
            let body = gen_block(e, cfg, ctx);
            ctx.nesting -= 1;
            ctx.loop_depth -= 1;
            FStmt::For {
                depth,
                count,
                body,
            }
        }
    }
}

/// An `if` condition: always a comparison, and always mentioning a parameter.
///
/// Both constraints exist to keep the branch **dynamic**, and they are about
/// the oracle rather than about coverage:
///
/// - A comparison is booleanity-constrained by construction, so the branch can
///   actually be taken. A raw integer condition must instead make the circuit
///   unprovable, which is real behaviour but not one this fuzzer can compare
///   against: for a *constant* condition the emitter prunes on `!is_zero()`
///   without a booleanity constraint, so `if 2` and a dynamic `a = 2` mean
///   different things. `tests/zk_control_flow.rs` covers that case directly.
/// - Mentioning a parameter keeps the condition's linear combination
///   non-constant, so the emitter cannot statically prune the branch. Pruning
///   is correct and desirable, but it depends on the emitter's constant
///   propagation through locals — which this module would have to reimplement
///   to predict, and a model that reimplements the thing it checks is not an
///   independent oracle.
///
/// A residual case survives: an operand can cancel to a constant, as in
/// `(p0 - p0) < 5`. That is rare, and the oracles below tolerate the one
/// direction it produces rather than reporting it as a bug.
fn gen_cond(e: &mut Entropy, cfg: &GenConfig, ctx: &Ctx) -> FExpr {
    // A bare parameter as the condition. Kept deliberately, and it is the one
    // non-comparison shape that is safe here: a parameter is never
    // constant-folded, so no branch can be statically pruned and the model
    // stays exact. It buys two things nothing else in the corpus reaches —
    // the booleanity constraint on a raw integer condition, and the
    // `if <ident> {` parse, which is where the struct-literal ambiguity lives.
    // Mutation testing is what established both: with only comparison
    // conditions generated, removing the booleanity constraint and reverting
    // the parser fix were BOTH missed by the whole sweep.
    if e.flip(20) {
        return FExpr::Param(e.choose(ctx.nparams));
    }
    let op = [Op::Lt, Op::Le, Op::Gt, Op::Ge, Op::Eq, Op::Ne][e.choose(6)];
    let l = gen_expr(e, cfg, ctx, cfg.max_depth.saturating_sub(1));
    let r = gen_expr(e, cfg, ctx, cfg.max_depth.saturating_sub(1));
    let (l, r) = if mentions_param(&l) || mentions_param(&r) {
        (l, r)
    } else {
        (FExpr::Param(e.choose(ctx.nparams)), r)
    };
    FExpr::Bin(op, Box::new(l), Box::new(r))
}

fn mentions_param(e: &FExpr) -> bool {
    match e {
        FExpr::Param(_) => true,
        FExpr::Lit(_) | FExpr::Local(_) | FExpr::LoopVar(_) => false,
        FExpr::Bin(_, l, r) => mentions_param(l) || mentions_param(r),
    }
}

pub fn gen_value(e: &mut Entropy) -> u64 {
    if e.flip(55) {
        INTERESTING[e.choose(INTERESTING.len())]
    } else {
        // Small values keep field and integer arithmetic in agreement most of
        // the time, so that the *control flow* is what varies between programs
        // rather than every operand tripping a range check.
        e.choose(64) as u64
    }
}

fn gen_expr(e: &mut Entropy, cfg: &GenConfig, ctx: &Ctx, depth: usize) -> FExpr {
    if depth >= cfg.max_depth || e.flip(35) {
        return gen_atom(e, ctx);
    }
    let op = match e.choose(16) {
        0 => Op::Add,
        1 => Op::Sub,
        2 => Op::Mul,
        3 => Op::Eq,
        4 => Op::Ne,
        5 => Op::Lt,
        6 => Op::Le,
        7 => Op::Gt,
        8 => Op::Ge,
        9 => Op::Div,
        10 => Op::Mod,
        11 => Op::BitAnd,
        12 => Op::BitOr,
        13 => Op::BitXor,
        14 => Op::Shl,
        _ => Op::Shr,
    };
    let l = gen_expr(e, cfg, ctx, depth + 1);
    // A variable shift amount is REFUSED by the emitter - it would be a
    // multiplexer over all 32 amounts - so the generator writes a literal one.
    // The range deliberately spans 32, because `amount >= 32` is not out of
    // range, it is "everything shifted out", which the gadget and the fold
    // both answer as 0; that is a boundary the two could disagree on.
    let r = if op.is_shift() {
        FExpr::Lit(e.choose(36) as u64)
    } else {
        gen_expr(e, cfg, ctx, depth + 1)
    };
    FExpr::Bin(op, Box::new(l), Box::new(r))
}

fn gen_atom(e: &mut Entropy, ctx: &Ctx) -> FExpr {
    let mut options = 1; // literal
    options += ctx.nparams;
    options += ctx.nlocals;
    options += ctx.loop_depth;

    let mut k = e.choose(options);
    if k == 0 {
        return FExpr::Lit(gen_value(e));
    }
    k -= 1;
    if k < ctx.nparams {
        return FExpr::Param(k);
    }
    k -= ctx.nparams;
    if k < ctx.nlocals {
        return FExpr::Local(k);
    }
    k -= ctx.nlocals;
    FExpr::LoopVar(k)
}

// ---------------------------------------------------------------------------
// Rendering to Y source
// ---------------------------------------------------------------------------

fn param_name(i: usize) -> String {
    format!("p{}", i)
}
fn local_name(i: usize) -> String {
    format!("v{}", i)
}
fn loop_name(d: usize) -> String {
    format!("i{}", d)
}

/// Render `prog` as Y source.
///
/// When `inline` is `Some(values)` the parameters are dropped and every use is
/// replaced by its literal value, which is what drives the constant-folding
/// metamorphic oracle. The function still takes one unused parameter, because
/// the emitter needs an input to build a witness against and a zero-argument
/// circuit is a different shape than the one under test.
pub fn render(prog: &FProgram, inline: Option<&[u64]>) -> String {
    let mut s = String::new();
    if inline.is_some() {
        s.push_str("fn main(unused: I32) -> I32 {\n");
    } else {
        let params: Vec<String> = (0..prog.nparams)
            .map(|i| format!("{}: I32", param_name(i)))
            .collect();
        s.push_str(&format!("fn main({}) -> I32 {{\n", params.join(", ")));
    }
    for (i, v) in prog.locals_init.iter().enumerate() {
        s.push_str(&format!("    let {}: I32 = {};\n", local_name(i), v));
    }
    render_block(&mut s, &prog.body, 1, inline);
    s.push_str("}\n");
    s
}

fn render_block(s: &mut String, stmts: &[FStmt], indent: usize, inline: Option<&[u64]>) {
    let pad = "    ".repeat(indent);
    for st in stmts {
        match st {
            FStmt::Return(e) => {
                s.push_str(&format!("{}return {};\n", pad, render_expr(e, inline)));
            }
            FStmt::Assign(slot, e) => {
                // `x = x op rhs` is spelled `x op= rhs`. The IR is unchanged,
                // so the interpreter, the minimiser and the refusal analysis
                // all see one statement with one meaning and cannot disagree
                // with the source about which - which is what makes this a
                // free way to cover an operator the generator could not
                // otherwise reach.
                if let FExpr::Bin(op, lhs, rhs) = e {
                    if matches!(op, Op::Add | Op::Sub | Op::Mul)
                        && matches!(**lhs, FExpr::Local(l) if l == *slot)
                    {
                        s.push_str(&format!(
                            "{}{} {}= {};\n",
                            pad,
                            local_name(*slot),
                            op.spelling(),
                            render_expr(rhs, inline)
                        ));
                        continue;
                    }
                }
                s.push_str(&format!(
                    "{}{} = {};\n",
                    pad,
                    local_name(*slot),
                    render_expr(e, inline)
                ));
            }
            FStmt::If {
                cond,
                then_b,
                else_b,
            } => {
                s.push_str(&format!("{}if {} {{\n", pad, render_expr(cond, inline)));
                render_block(s, then_b, indent + 1, inline);
                if let Some(eb) = else_b {
                    s.push_str(&format!("{}}} else {{\n", pad));
                    render_block(s, eb, indent + 1, inline);
                }
                s.push_str(&format!("{}}}\n", pad));
            }
            FStmt::For {
                depth,
                count,
                body,
            } => {
                // The `@invariant` is what the type checker demands of any
                // loop. The emitter path this fuzzer drives does not run the
                // type checker, but emitting it keeps the programs valid for
                // the full pipeline too.
                s.push_str(&format!("{}@invariant({} >= 0)\n", pad, loop_name(*depth)));
                s.push_str(&format!(
                    "{}for {} in 0..{} {{\n",
                    pad,
                    loop_name(*depth),
                    count
                ));
                render_block(s, body, indent + 1, inline);
                s.push_str(&format!("{}}}\n", pad));
            }
        }
    }
}

fn render_expr(e: &FExpr, inline: Option<&[u64]>) -> String {
    match e {
        FExpr::Lit(v) => v.to_string(),
        FExpr::Param(i) => match inline {
            Some(vals) => vals.get(*i).copied().unwrap_or(0).to_string(),
            None => param_name(*i),
        },
        FExpr::Local(i) => local_name(*i),
        FExpr::LoopVar(d) => loop_name(*d),
        FExpr::Bin(op, l, r) => format!(
            "({} {} {})",
            render_expr(l, inline),
            op.spelling(),
            render_expr(r, inline)
        ),
    }
}

// ---------------------------------------------------------------------------
// The reference interpreter
// ---------------------------------------------------------------------------

/// What running a program produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Value(Fr),
    /// The circuit admits no satisfying assignment. This is a legitimate,
    /// specified result — an out-of-range comparison operand or a non-boolean
    /// `if` condition is meant to be unprovable rather than wrong.
    Unprovable,
    /// The emitter refused to compile the program. A legitimate, fail-closed
    /// answer.
    Rejected,
    /// The program did not parse. Never legitimate here: every program this
    /// module renders is well-formed Y by construction, so a parse failure is
    /// a front-end bug and is graded separately from a semantic refusal.
    ParseError,
}

/// Evaluation can end early, either by returning or by becoming unprovable.
enum Flow {
    Normal,
    Returned(Fr),
    Stuck,
}

struct Env {
    params: Vec<Fr>,
    locals: Vec<Fr>,
    loops: Vec<Fr>,
}

/// Evaluate `prog` on `inputs` under the documented semantics.
///
/// This is deliberately a second implementation, written from the language
/// definition rather than from `zk_emitter.rs`. The semantics it encodes:
///
/// - Values are field elements. `+`, `-` and `*` are **field** operations, so
///   `a - b` with `b > a` is `p - (b - a)`, an enormous number — not a wrap to
///   a negative 32-bit integer.
/// - `==` and `!=` are field equality and need no range check (3 constraints).
/// - `<`, `<=`, `>`, `>=` range-check **both** operands to 32 bits. An operand
///   outside that range makes the circuit unsatisfiable. This is the
///   fail-closed choice that costs ~101 constraints where circomlib's
///   `LessThan(32)` costs 36 and answers `p-1 < 0` incorrectly.
/// - `&&` and `||` constrain both operands boolean.
/// - An `if` condition is a selector and so must be a bit.
pub fn interpret(prog: &FProgram, inputs: &[u64]) -> Outcome {
    // A circuit has no control flow: every branch is emitted, so a gadget
    // operand that is out of range statically is a compile error even when it
    // sits in a branch this input never takes. Modelling that here rather than
    // letting it surface as a stream of false "over-refusal" findings.
    if statically_refused(&prog.body) {
        return Outcome::Rejected;
    }
    let mut env = Env {
        params: (0..prog.nparams)
            .map(|i| Fr::from_u64(inputs.get(i).copied().unwrap_or(0)))
            .collect(),
        locals: prog.locals_init.iter().map(|v| Fr::from_u64(*v)).collect(),
        loops: Vec::new(),
    };
    match exec_block(&prog.body, &mut env) {
        Flow::Returned(v) => Outcome::Value(v),
        Flow::Stuck => Outcome::Unprovable,
        // Falling off the end cannot happen: `gen_program` appends a return.
        Flow::Normal => Outcome::Unprovable,
    }
}

fn exec_block(stmts: &[FStmt], env: &mut Env) -> Flow {
    for st in stmts {
        match st {
            FStmt::Return(e) => {
                return match eval(e, env) {
                    Some(v) => Flow::Returned(v),
                    None => Flow::Stuck,
                }
            }
            FStmt::Assign(slot, e) => match eval(e, env) {
                Some(v) => env.locals[*slot] = v,
                None => return Flow::Stuck,
            },
            FStmt::If {
                cond,
                then_b,
                else_b,
            } => {
                let c = match eval(cond, env) {
                    Some(v) => v,
                    None => return Flow::Stuck,
                };
                // The condition is a selector, so it must be a bit.
                let taken = if c == Fr::zero() {
                    false
                } else if c == Fr::one() {
                    true
                } else {
                    return Flow::Stuck;
                };
                let branch = if taken { Some(then_b) } else { else_b.as_ref() };
                if let Some(b) = branch {
                    match exec_block(b, env) {
                        Flow::Normal => {}
                        other => return other,
                    }
                }
            }
            FStmt::For {
                depth,
                count,
                body,
            } => {
                for k in 0..*count {
                    if env.loops.len() <= *depth {
                        env.loops.resize(*depth + 1, Fr::zero());
                    }
                    env.loops[*depth] = Fr::from_u64(k);
                    match exec_block(body, env) {
                        Flow::Normal => {}
                        other => return other,
                    }
                }
            }
        }
    }
    Flow::Normal
}

/// Whether some branch this input does NOT take contains an expression whose
/// range check cannot be satisfied.
///
/// A circuit has no control flow. Both arms of an `if` are emitted, and the
/// range check inside `emit_num2bits` is unconditional, so an operand that
/// underflows in an arm the input never takes still constrains the witness and
/// can make the whole circuit unprovable. Minimal case, pinned by
/// `tests/zk_dead_branch_range.rs`:
///
/// ```text
/// if p0 < 5 { return 1; } else { return (0 - p0) < 3; }   // p0 = 1
/// ```
///
/// The taken branch is `return 1`, and the circuit is unprovable.
///
/// This exists to *attribute* over-refusals rather than to excuse them: it lets
/// the sweep say how many of its findings are this one known limitation and how
/// many are something else.
pub fn dead_branch_range_violation(prog: &FProgram, inputs: &[u64]) -> bool {
    let mut env = Env {
        params: (0..prog.nparams)
            .map(|i| Fr::from_u64(inputs.get(i).copied().unwrap_or(0)))
            .collect(),
        locals: prog.locals_init.iter().map(|v| Fr::from_u64(*v)).collect(),
        loops: Vec::new(),
    };
    arm_has_violation(&prog.body, &mut env)
}

/// Does any expression in this arm fail its range check, ignoring control flow?
fn arm_has_violation(stmts: &[FStmt], env: &mut Env) -> bool {
    for st in stmts {
        match st {
            FStmt::Return(e) => {
                if eval(e, env).is_none() {
                    return true;
                }
            }
            FStmt::Assign(slot, e) => match eval(e, env) {
                Some(v) => env.locals[*slot] = v,
                None => return true,
            },
            FStmt::If {
                cond,
                then_b,
                else_b,
            } => {
                if eval(cond, env).is_none() {
                    return true;
                }
                if arm_has_violation(then_b, env) {
                    return true;
                }
                if let Some(b) = else_b {
                    if arm_has_violation(b, env) {
                        return true;
                    }
                }
            }
            FStmt::For {
                depth,
                count,
                body,
            } => {
                for k in 0..*count {
                    if env.loops.len() <= *depth {
                        env.loops.resize(*depth + 1, Fr::zero());
                    }
                    env.loops[*depth] = Fr::from_u64(k);
                    if arm_has_violation(body, env) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Whether the emitter must refuse this program outright, for any input.
///
/// The only such case the generated fragment can produce is a gadget operand
/// that is a closed constant expression outside the 32-bit range. The check is
/// deliberately conservative — it looks only at expressions built from
/// literals, not at values the emitter's constant propagation might additionally
/// know — so it can under-report but never over-report. An under-report shows
/// up as an `OverRefusal` finding, which is reported and not gating.
fn statically_refused(stmts: &[FStmt]) -> bool {
    stmts.iter().any(|st| match st {
        FStmt::Return(e) => expr_statically_refused(e),
        FStmt::Assign(_, e) => expr_statically_refused(e),
        FStmt::If {
            cond,
            then_b,
            else_b,
        } => {
            expr_statically_refused(cond)
                || statically_refused(then_b)
                || else_b.as_ref().is_some_and(|b| statically_refused(b))
        }
        FStmt::For { body, .. } => statically_refused(body),
    })
}

fn expr_statically_refused(e: &FExpr) -> bool {
    match e {
        FExpr::Lit(_) | FExpr::Param(_) | FExpr::Local(_) | FExpr::LoopVar(_) => false,
        FExpr::Bin(op, l, r) => {
            if op.is_ordering() {
                for side in [l, r] {
                    if let Some(v) = closed_value(side) {
                        if !in_32_bit_range(v) {
                            return true;
                        }
                    }
                }
            }
            expr_statically_refused(l) || expr_statically_refused(r)
        }
    }
}

/// The value of an expression built only from literals, if it has one.
fn closed_value(e: &FExpr) -> Option<Fr> {
    match e {
        FExpr::Lit(v) => Some(Fr::from_u64(*v)),
        FExpr::Param(_) | FExpr::Local(_) | FExpr::LoopVar(_) => None,
        FExpr::Bin(op, l, r) => {
            let a = closed_value(l)?;
            let b = closed_value(r)?;
            apply(*op, a, b)
        }
    }
}

/// `None` means the circuit is unsatisfiable for this input.
fn eval(e: &FExpr, env: &Env) -> Option<Fr> {
    match e {
        FExpr::Lit(v) => Some(Fr::from_u64(*v)),
        FExpr::Param(i) => env.params.get(*i).copied(),
        FExpr::Local(i) => env.locals.get(*i).copied(),
        FExpr::LoopVar(d) => env.loops.get(*d).copied(),
        FExpr::Bin(op, l, r) => {
            let a = eval(l, env)?;
            let b = eval(r, env)?;
            apply(*op, a, b)
        }
    }
}

fn in_32_bit_range(v: Fr) -> bool {
    v < two_pow_32()
}

fn as_bit(v: Fr) -> Option<bool> {
    if v == Fr::zero() {
        Some(false)
    } else if v == Fr::one() {
        Some(true)
    } else {
        None
    }
}

fn bit(b: bool) -> Fr {
    if b {
        Fr::one()
    } else {
        Fr::zero()
    }
}

fn apply(op: Op, a: Fr, b: Fr) -> Option<Fr> {
    match op {
        Op::Add => Some(a.add(&b)),
        Op::Sub => Some(a.sub(&b)),
        Op::Mul => Some(a.mul(&b)),
        Op::Eq => Some(bit(a == b)),
        Op::Ne => Some(bit(a != b)),
        _ if op.is_ordering() => {
            // Both operands are range-checked, so an out-of-range operand is
            // unprovable rather than compared.
            if !in_32_bit_range(a) || !in_32_bit_range(b) {
                return None;
            }
            Some(bit(match op {
                Op::Lt => a < b,
                Op::Le => a <= b,
                Op::Gt => a > b,
                Op::Ge => a >= b,
                _ => unreachable!(),
            }))
        }
        _ if op.is_logical() => {
            let (x, y) = (as_bit(a)?, as_bit(b)?);
            Some(bit(if op == Op::And { x && y } else { x || y }))
        }
        _ if op.is_divmod() => {
            // `emit_int_div_mod` range-checks the QUOTIENT, the remainder and
            // the DIVISOR - and not the dividend, which is pinned instead by
            // `q * b = a - r`. So a dividend may be up to 2^64 and still be
            // provable, while a divisor may not, and `b = 0` makes `r < b`
            // unsatisfiable. This asymmetry is the gadget's, not a choice made
            // here: the oracle states what the constraints admit.
            if b == Fr::zero() || !in_32_bit_range(b) {
                return None;
            }
            let (q, r) = a.int_div_rem(&b);
            if !in_32_bit_range(q) {
                return None;
            }
            Some(if op == Op::Div { q } else { r })
        }
        _ if op.is_bitwise() => {
            // Both operands are decomposed, so both are range-checked.
            if !in_32_bit_range(a) || !in_32_bit_range(b) {
                return None;
            }
            let (x, y) = (a.to_u64()?, b.to_u64()?);
            Some(Fr::from_u64(match op {
                Op::BitAnd => x & y,
                Op::BitOr => x | y,
                _ => x ^ y,
            }))
        }
        _ if op.is_shift() => {
            // Only the shifted VALUE is decomposed; the amount is a literal.
            if !in_32_bit_range(a) {
                return None;
            }
            let (v, amount) = (a.to_u64()?, b.to_u64()?);
            let n = COMPARISON_BITS as u64;
            let mask = (1u64 << n) - 1;
            Some(Fr::from_u64(if amount >= n {
                0
            } else if op == Op::Shl {
                (v << amount) & mask
            } else {
                (v & mask) >> amount
            }))
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Running the real compiler
// ---------------------------------------------------------------------------

/// Compile `source` to R1CS and solve it for `inputs`.
///
/// Mirrors what `tests/zk_control_flow.rs` does, deliberately: the lexer,
/// parser and emitter are the path under test, and the type checker is not run
/// (nor is it on that test's path).
pub fn run_circuit(source: &str, inputs: &[u64]) -> Outcome {
    let tokens = crate::lexer::Lexer::new(source).tokenize();
    let prog = match crate::parser::Parser::new(tokens).parse_program() {
        Ok(p) => p,
        Err(_) => return Outcome::ParseError,
    };
    let mut emitter = ZkEmitter::new();
    if emitter.emit_program(&prog).is_err() {
        return Outcome::Rejected;
    }

    let circuit = emitter.build_circuit();
    let ir = emitter.build_witness_ir();
    let privs: Vec<Fr> = inputs.iter().map(|v| Fr::from_u64(*v)).collect();
    let (witness, satisfied) = solve_r1cs_witness(
        &circuit.constraints,
        &ir,
        circuit.num_variables,
        &[],
        &privs,
    );
    if !satisfied {
        return Outcome::Unprovable;
    }
    // A solved witness that does not satisfy its own circuit is a bug in its
    // own right, and a much worse one than a wrong value.
    if check_r1cs_satisfiability(&circuit.constraints, &witness).is_err() {
        return Outcome::Rejected;
    }
    match circuit.outputs.first() {
        Some(out) => Outcome::Value(witness[*out]),
        None => Outcome::Rejected,
    }
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    /// The circuit computes a different value than the program means. A
    /// Groth16 proof over it attests to the wrong statement.
    WrongValue,
    /// The compiler refused, or made unprovable, a program the semantics
    /// accept. Fail-closed and so not a soundness hole, but it may be
    /// over-refusal and is worth a look.
    OverRefusal,
    /// Two paths through the compiler disagree about the same program.
    FoldingDivergence,
    /// A well-formed program failed to parse.
    ParseFailure,
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub severity: Severity,
    pub source: String,
    pub inputs: Vec<u64>,
    pub detail: String,
}

/// Both non-value outcomes are fail-closed: no proof can be produced either
/// way, and they differ only in whether the user finds out at compile time or
/// at witness time. Treating them as one avoids a stream of findings that name
/// a difference nobody can exploit.
fn is_refusal(o: &Outcome) -> bool {
    matches!(o, Outcome::Unprovable | Outcome::Rejected)
}

fn describe(o: &Outcome) -> String {
    match o {
        Outcome::Value(v) => v.to_decimal_string(),
        Outcome::Unprovable => "<unprovable>".to_string(),
        Outcome::Rejected => "<rejected>".to_string(),
        Outcome::ParseError => "<parse error>".to_string(),
    }
}

/// Run every oracle over one generated program at one input vector.
pub fn check(prog: &FProgram, inputs: &[u64]) -> Vec<Finding> {
    let mut findings = Vec::new();

    let source = render(prog, None);
    let expected = interpret(prog, inputs);
    let actual = run_circuit(&source, inputs);

    // A parse failure is graded first and on its own. Every program rendered
    // here is well-formed by construction, so this is unambiguous - unlike a
    // semantic refusal, there is no reading under which it is the right answer,
    // and the remaining oracles would only report it a second time as noise.
    if actual == Outcome::ParseError {
        findings.push(Finding {
            severity: Severity::ParseFailure,
            source: source.clone(),
            inputs: inputs.to_vec(),
            detail: "a well-formed generated program failed to parse".to_string(),
        });
        return findings;
    }

    // Oracle 1: the circuit against the independent interpreter.
    match (&expected, &actual) {
        (Outcome::Value(a), Outcome::Value(b)) if a != b => findings.push(Finding {
            severity: Severity::WrongValue,
            source: source.clone(),
            inputs: inputs.to_vec(),
            detail: format!(
                "interpreter says {}, circuit says {}",
                a.to_decimal_string(),
                b.to_decimal_string()
            ),
        }),
        (Outcome::Value(_), b) if is_refusal(b) => findings.push(Finding {
            severity: Severity::OverRefusal,
            source: source.clone(),
            inputs: inputs.to_vec(),
            detail: format!(
                "interpreter says {}, compiler says {}{}",
                describe(&expected),
                describe(&actual),
                if dead_branch_range_violation(prog, inputs) {
                    " [attributed: unconditional range check in an untaken branch]"
                } else {
                    " [UNATTRIBUTED]"
                }
            ),
        }),
        // The dangerous direction: a range violation reached the witness on
        // this input and the compiler produced a proof anyway.
        //
        // Only the DYNAMIC refusal counts here. `Outcome::Rejected` from
        // `statically_refused` is a whole-program judgement that ignores static
        // branch pruning — the emitter constant-propagates locals and drops
        // branches this module cannot predict, so a program it compiles despite
        // a dead out-of-range operand is the model being conservative, not the
        // compiler being wrong.
        (Outcome::Unprovable, Outcome::Value(v)) => findings.push(Finding {
            severity: Severity::WrongValue,
            source: source.clone(),
            inputs: inputs.to_vec(),
            detail: format!(
                "semantics say this input admits no proof, circuit produced {}",
                v.to_decimal_string()
            ),
        }),
        _ => {}
    }

    // Oracle 2: constant folding must agree with gadget emission.
    //
    // Rendering the inputs as literals sends the emitter down its
    // `is_constant()` paths, which are a separate implementation of every
    // operator. Nothing but this compares them.
    let folded_src = render(prog, Some(inputs));
    let folded = run_circuit(&folded_src, &[0]);
    let diverged = match (&folded, &actual) {
        // The case this oracle exists for: two paths, two answers.
        (Outcome::Value(a), Outcome::Value(b)) => a != b,
        // Refusing at compile time and being unprovable are the same answer.
        (a, b) if is_refusal(a) && is_refusal(b) => false,
        // Folded produces a value where the parameterised form refuses. This is
        // legitimate and expected: with every input a literal, the emitter can
        // fold conditions and prune branches, so an out-of-range operand in a
        // branch that is dead FOR THESE INPUTS is never emitted. Constant
        // folding is allowed to make more programs provable.
        (Outcome::Value(_), b) if is_refusal(b) => false,
        // The reverse is not legitimate: folding must never LOSE the ability to
        // compile a program the gadget path handles.
        _ => true,
    };
    if diverged {
        // A refusal on one side only is still a divergence, but rank a value
        // disagreement higher: that one is a wrong answer either way.
        let severity = match (&folded, &actual) {
            (Outcome::Value(a), Outcome::Value(b)) if a != b => Severity::WrongValue,
            _ => Severity::FoldingDivergence,
        };
        findings.push(Finding {
            severity,
            source: format!("{}\n/* folded form: */\n{}", source, folded_src),
            inputs: inputs.to_vec(),
            detail: format!(
                "parameterised form gives {}, constant-folded form gives {}",
                describe(&actual),
                describe(&folded)
            ),
        });
    }

    findings
}

// ---------------------------------------------------------------------------
// Minimisation
// ---------------------------------------------------------------------------

/// Shrink a failing program while it keeps producing a finding of the same
/// severity.
///
/// Delta debugging over the IR rather than over the rendered text, so every
/// candidate is still a well-formed program and the reduction never has to
/// re-parse its own output. A generated counterexample is 40 lines of nested
/// loops; the same finding usually survives down to three or four, and reading
/// the small one is the difference between attributing a bug and guessing at
/// it.
pub fn minimize(prog: &FProgram, inputs: &[u64], want: Severity) -> FProgram {
    let fails = |p: &FProgram| -> bool {
        // A shrink that changes which parameters exist would change the
        // meaning of `inputs`, so `nparams` is held fixed throughout.
        check(p, inputs).iter().any(|f| f.severity == want)
    };
    if !fails(prog) {
        return prog.clone();
    }

    let mut best = prog.clone();
    let mut improved = true;
    while improved {
        improved = false;

        // 1. Drop a statement.
        for idx in 0..count_stmts(&best.body) {
            let mut cand = best.clone();
            let mut n = idx;
            if drop_stmt(&mut cand.body, &mut n) && fails(&cand) {
                best = cand;
                improved = true;
                break;
            }
        }
        if improved {
            continue;
        }

        // 2. Simplify an expression to one of its own operands, which keeps
        //    any variable reference the finding might depend on.
        for idx in 0..count_exprs(&best.body) {
            for side in [true, false] {
                let mut cand = best.clone();
                let mut n = idx;
                if simplify_expr(&mut cand.body, &mut n, side) && fails(&cand) {
                    best = cand;
                    improved = true;
                    break;
                }
            }
            if improved {
                break;
            }
        }
        if improved {
            continue;
        }

        // 3. Drop an unused local.
        if !best.locals_init.is_empty() {
            let mut cand = best.clone();
            cand.locals_init.pop();
            let slot = cand.locals_init.len();
            if !mentions_local(&cand.body, slot) && fails(&cand) {
                best = cand;
                improved = true;
            }
        }
    }
    best
}

fn count_stmts(stmts: &[FStmt]) -> usize {
    stmts
        .iter()
        .map(|s| {
            1 + match s {
                FStmt::If {
                    then_b, else_b, ..
                } => count_stmts(then_b) + else_b.as_ref().map_or(0, |b| count_stmts(b)),
                FStmt::For { body, .. } => count_stmts(body),
                _ => 0,
            }
        })
        .sum()
}

/// Remove the `n`th statement in pre-order. Returns whether one was removed.
fn drop_stmt(stmts: &mut Vec<FStmt>, n: &mut usize) -> bool {
    for i in 0..stmts.len() {
        if *n == 0 {
            stmts.remove(i);
            return true;
        }
        *n -= 1;
        let removed = match &mut stmts[i] {
            FStmt::If {
                then_b, else_b, ..
            } => {
                drop_stmt(then_b, n)
                    || else_b.as_mut().map_or(false, |b| drop_stmt(b, n))
            }
            FStmt::For { body, .. } => drop_stmt(body, n),
            _ => false,
        };
        if removed {
            return true;
        }
    }
    false
}

fn count_exprs(stmts: &[FStmt]) -> usize {
    stmts
        .iter()
        .map(|s| match s {
            FStmt::Return(e) | FStmt::Assign(_, e) => count_expr(e),
            FStmt::If {
                cond,
                then_b,
                else_b,
            } => {
                count_expr(cond)
                    + count_stmts_exprs(then_b)
                    + else_b.as_ref().map_or(0, |b| count_stmts_exprs(b))
            }
            FStmt::For { body, .. } => count_stmts_exprs(body),
        })
        .sum()
}

fn count_stmts_exprs(stmts: &[FStmt]) -> usize {
    count_exprs(stmts)
}

fn count_expr(e: &FExpr) -> usize {
    match e {
        FExpr::Bin(_, l, r) => 1 + count_expr(l) + count_expr(r),
        _ => 0,
    }
}

fn simplify_expr(stmts: &mut [FStmt], n: &mut usize, left: bool) -> bool {
    for st in stmts.iter_mut() {
        let done = match st {
            FStmt::Return(e) | FStmt::Assign(_, e) => simplify_in(e, n, left),
            FStmt::If {
                cond,
                then_b,
                else_b,
            } => {
                simplify_in(cond, n, left)
                    || simplify_expr(then_b, n, left)
                    || else_b.as_mut().map_or(false, |b| simplify_expr(b, n, left))
            }
            FStmt::For { body, .. } => simplify_expr(body, n, left),
        };
        if done {
            return true;
        }
    }
    false
}

fn simplify_in(e: &mut FExpr, n: &mut usize, left: bool) -> bool {
    if let FExpr::Bin(_, l, r) = e {
        if *n == 0 {
            *e = if left { (**l).clone() } else { (**r).clone() };
            return true;
        }
        *n -= 1;
        let mut l2 = (**l).clone();
        if simplify_in(&mut l2, n, left) {
            **l = l2;
            return true;
        }
        let mut r2 = (**r).clone();
        if simplify_in(&mut r2, n, left) {
            **r = r2;
            return true;
        }
    }
    false
}

fn mentions_local(stmts: &[FStmt], slot: usize) -> bool {
    stmts.iter().any(|s| match s {
        FStmt::Return(e) => expr_mentions_local(e, slot),
        FStmt::Assign(sl, e) => *sl == slot || expr_mentions_local(e, slot),
        FStmt::If {
            cond,
            then_b,
            else_b,
        } => {
            expr_mentions_local(cond, slot)
                || mentions_local(then_b, slot)
                || else_b.as_ref().is_some_and(|b| mentions_local(b, slot))
        }
        FStmt::For { body, .. } => mentions_local(body, slot),
    })
}

fn expr_mentions_local(e: &FExpr, slot: usize) -> bool {
    match e {
        FExpr::Local(i) => *i == slot,
        FExpr::Bin(_, l, r) => expr_mentions_local(l, slot) || expr_mentions_local(r, slot),
        _ => false,
    }
}

/// Generate one program from `bytes` and check it on several input vectors.
///
/// Reusing a program across inputs is what makes a constant circuit visible:
/// the four bugs all produced circuits whose output did not depend on their
/// inputs, and a single input vector cannot distinguish that from a correct
/// answer that happens to agree.
pub fn check_bytes(bytes: &[u8]) -> Vec<Finding> {
    let mut e = Entropy::new(bytes);
    let cfg = GenConfig::default();
    let prog = gen_program(&mut e, &cfg);

    let mut findings = Vec::new();
    for _ in 0..4 {
        let inputs: Vec<u64> = (0..prog.nparams).map(|_| gen_value(&mut e)).collect();
        findings.extend(check(&prog, &inputs));
        if !findings.is_empty() {
            break;
        }
    }
    findings
}
