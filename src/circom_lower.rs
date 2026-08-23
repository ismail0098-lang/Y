// ============================================================
//  Y — circom front end: lowering to R1CS
//  circom_lower.rs
// ============================================================
//
// Executes a circom program at compile time and emits the constraints it
// describes into a `ZkEmitter`, reusing everything downstream of constraint
// construction: the CSE pass, the snarkjs wire map, the `.r1cs`/`.wtns`/`.sym`
// writers, and the witness solver.
//
// The value model is circom's own. Every expression is one of:
//
//   Const  a field element known now
//   Lin    a linear combination of wires
//   Quad   `a * b + c`, the most a single R1CS constraint can express
//
// and anything that would exceed `Quad` is REFUSED with the same words circom
// uses ("non-quadratic"), never approximated. That is not politeness: a circuit
// missing a constraint still produces proofs, of a weaker statement than its
// author wrote, and no artifact downstream records the difference.

#![allow(dead_code)]

use crate::circom_ast::*;
use crate::zk_emitter::{LinearCombination, WitnessOp, ZkEmitter};
use crate::zk_field::{BigUint, Fr};
use std::collections::HashMap;

type LResult<T> = Result<T, String>;

fn err(pos: Pos, msg: impl std::fmt::Display) -> String {
    format!("{}:{}: {}", pos.line, pos.col, msg)
}

// ────────────────────────────────────────────────────────
// circom's SIGNED comparison of `var`s
// ────────────────────────────────────────────────────────
//
// circom orders field elements by their SIGNED representative: a value above
// `(p-1)/2` denotes `v - p`, a negative number. Y's `Fr: Ord` is CANONICAL and
// deliberately so — `zk_emitter` folds Y's own `<`/`<=`/`>`/`>=` through it,
// where operands carry a 32-bit range check and the canonical order is the
// intended one. The two disagree on every pair straddling `(p-1)/2`, so this
// front end must not borrow that ordering.
//
// Measured against circom 2.2.3, `var a = 0 - 1; if (a < 1)` takes the TRUE
// branch there and took the FALSE branch here — the same source, one
// constraint each, computing different functions. Both compile; both prove.
//
// Only the four comparisons are signed. `\`, `%`, `<<`, `>>` and the bitwise
// operators were probed at `p-1` and all return the UNSIGNED result in circom,
// so they are left alone.

thread_local! {
    /// `(p-1)/2` for the active modulus: the largest value circom reads as
    /// positive. Cached, and keyed on the modulus, because Montgomery form is
    /// defined relative to it — the same discipline as `POSEIDON_T3_CACHE`.
    static SIGNED_HALF: std::cell::RefCell<Option<(BigUint, Fr)>> =
        const { std::cell::RefCell::new(None) };
}

fn signed_half() -> Fr {
    let modulus = crate::zk_field::active_modulus();
    if let Some(hit) = SIGNED_HALF.with(|cell| {
        cell.borrow().as_ref().filter(|(m, _)| *m == modulus).map(|(_, h)| *h)
    }) {
        return hit;
    }
    let two = BigUint::from_u64(2);
    let half = Fr::from_biguint(modulus.sub(&BigUint::from_u64(1)).div_mod(&two).0);
    SIGNED_HALF.with(|cell| *cell.borrow_mut() = Some((modulus, half)));
    half
}

/// True when circom would read `v` as negative, i.e. `v > (p-1)/2`.
fn is_signed_negative(v: &Fr) -> bool {
    *v > signed_half()
}

/// Compare two field elements the way circom compares `var`s.
///
/// Within one sign class the canonical order already agrees: for `a, b` both
/// above `(p-1)/2`, `a - p < b - p` exactly when `a < b`. Only a pair
/// straddling the boundary needs the explicit case.
fn signed_cmp(a: &Fr, b: &Fr) -> std::cmp::Ordering {
    match (is_signed_negative(a), is_signed_negative(b)) {
        (false, true) => std::cmp::Ordering::Greater,
        (true, false) => std::cmp::Ordering::Less,
        _ => a.cmp(b),
    }
}

// ────────────────────────────────────────────────────────
// Values
// ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum Val {
    Const(Fr),
    Lin(LinearCombination),
    /// `a * b + c`
    Quad(LinearCombination, LinearCombination, LinearCombination),
    /// A value the compiler cannot compute, carrying the id of the reason it
    /// could not — see `Lowerer::opaque_reasons`.
    ///
    /// circom's `var`s are not restricted to compile-time constants: a template
    /// may write `var y2 = out[1] * out[1];` and then compute with it, and a
    /// `function` may be called on signal arrays. The results are *witness*
    /// values, and circom's own compiler carries them as unknowns.
    ///
    /// **The rule that makes this safe is that an `Opaque` may only ever reach
    /// a `<--`.** `<--` emits no constraint, so nothing about the `.r1cs`
    /// depends on a value Y did not compute — the emitted circuit is exactly
    /// what a compiler that *could* compute it would emit. Reaching a `<==`, a
    /// `===`, an array index, an array dimension, a template argument or an
    /// `assert` is refused, and the refusal names the position where the value
    /// first became unknown rather than the position where it was used.
    ///
    /// What is given up is Y's ability to solve the witness for those signals
    /// by itself: they get `WitnessOp::Unknown`, and `lower` reports how many.
    /// That is fail-closed — an unsolved wire stays zero and the satisfiability
    /// check says so — and it is the artifact users of a circom compiler want
    /// least, which is the whole reason the trade is worth making.
    Opaque(u32),
}

impl Val {
    /// Consume the value and take its linear combination **without copying**.
    ///
    /// `lc(&self)` clones, and the arithmetic helpers take their operands by
    /// value and then drop them — so every `+` and `*` in a circom source file
    /// was allocating and copying a `Vec` it was about to throw away. Poseidon's
    /// combinations are ~28 terms wide and a round touches them constantly,
    /// which is why lowering cost ~135 allocations per constraint against the
    /// native emitter's 12.
    fn into_lc(self) -> Option<LinearCombination> {
        match self {
            Val::Const(c) => Some(LinearCombination::constant(c)),
            Val::Lin(l) => Some(l),
            Val::Quad(..) | Val::Opaque(_) => None,
        }
    }

    /// Add `self * scale` into `dst` in place.
    ///
    /// The point is the `Const` arm: `dst.add_linear(&v.lc().unwrap(), scale)`
    /// builds a one-term `Vec` purely to pass a reference to it.
    fn add_into(&self, dst: &mut LinearCombination, scale: Fr) -> bool {
        match self {
            Val::Const(c) => {
                dst.add_constant(c.mul(&scale));
                true
            }
            Val::Lin(l) => {
                dst.add_linear(l, scale);
                true
            }
            Val::Quad(..) | Val::Opaque(_) => false,
        }
    }

    fn lc(&self) -> Option<LinearCombination> {
        match self {
            Val::Const(c) => Some(LinearCombination::constant(*c)),
            Val::Lin(l) => Some(l.clone()),
            Val::Quad(..) | Val::Opaque(_) => None,
        }
    }

    fn as_const(&self) -> Option<Fr> {
        match self {
            Val::Const(c) => Some(*c),
            _ => None,
        }
    }

    fn opaque_id(&self) -> Option<u32> {
        match self {
            Val::Opaque(id) => Some(*id),
            _ => None,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Val::Const(_) => "constant",
            Val::Lin(_) => "linear",
            Val::Quad(..) => "quadratic",
            Val::Opaque(_) => "not computable at compile time",
        }
    }
}

/// A variable or signal slot, which may be an arbitrarily nested array.
#[derive(Clone, Debug)]
enum Slot<T> {
    Leaf(T),
    Array(Vec<Slot<T>>),
}

impl<T> Slot<T> {
    fn leaf(&self, pos: Pos, what: &str) -> LResult<&T> {
        match self {
            Slot::Leaf(v) => Ok(v),
            Slot::Array(_) => Err(err(pos, format!("{} is an array; expected a single value", what))),
        }
    }
}

type VarSlot = Slot<Val>;
type SigSlot = Slot<usize>;

/// One instantiated template.
struct Instance {
    template: String,
    /// Signal name -> wires, in declaration order for the input/output lists.
    signals: HashMap<String, SigSlot>,
    input_order: Vec<String>,
    output_order: Vec<String>,
}

#[derive(Clone)]
enum CompSlot {
    /// Declared but not yet assigned a template.
    Empty,
    Inst(usize),
    Array(Vec<CompSlot>),
}

/// One template activation: its parameters, vars, signals and subcomponents.
struct Frame {
    vars: Vec<HashMap<String, VarSlot>>,
    signals: HashMap<String, SigSlot>,
    input_order: Vec<String>,
    output_order: Vec<String>,
    components: HashMap<String, CompSlot>,
    prefix: String,
    /// Serial number for the components an anonymous instantiation creates.
    /// Per FRAME rather than per Lowerer so a template's `.sym` names do not
    /// depend on how many times unrelated templates were instantiated first.
    anon_count: usize,
}

enum Flow {
    Normal,
    Return(VarSlot),
}

pub struct Lowerer<'a> {
    program: &'a Program,
    pub emitter: ZkEmitter,
    instances: Vec<Instance>,
    depth: usize,
    /// Results of compile-time function calls, keyed by name and argument
    /// values.
    ///
    /// circom `function`s cannot touch signals - they are pure over
    /// compile-time values - so a call with the same arguments has the same
    /// result, and this is a cache rather than a guess. It matters because of
    /// how circomlib is written: `POSEIDON_C(t)` returns a 195-element table of
    /// literals, and `Poseidon(2)` calls it once per instantiation. A 200-hash
    /// chain therefore rebuilt those tables 200 times over.
    fn_cache: HashMap<(String, Vec<Fr>), VarSlot>,
    /// Why each `Val::Opaque` is opaque, already formatted as `line:col: text`.
    ///
    /// A value that cannot be computed is refused at the point it is *used*,
    /// which is usually far from the point it became unknown - in
    /// `Bits2Point_Strict` the two are a `sqrt` call apart. Carrying the origin
    /// with the value means the message names the cause rather than the symptom.
    opaque_reasons: Vec<String>,
    /// Wires whose `<--` recipe came out `WitnessOp::Unknown`, with the reason
    /// id. Reported at the end of `lower`, because a silently unsolvable
    /// witness is exactly the kind of quiet degradation this repo refuses.
    opaque_witness: Vec<(usize, u32)>,
    /// `assert`s over witness-domain values, which emit no constraint and
    /// which Y does not evaluate. Reported at the end of lowering; see
    /// `report_unchecked_asserts`.
    unchecked_asserts: Vec<String>,
    /// Set by `witness_only_recipe` when it emits `WitnessOp::Unknown`, read
    /// and cleared by `substitute`, which is the only caller that knows the
    /// wire being assigned.
    last_opaque: Option<u32>,
}

const MAX_DEPTH: usize = 256;
const MAX_LOOP_ITERS: usize = 20_000_000;

impl<'a> Lowerer<'a> {
    pub fn new(program: &'a Program) -> Self {
        Lowerer {
            program,
            emitter: ZkEmitter::new(),
            instances: Vec::new(),
            depth: 0,
            fn_cache: HashMap::new(),
            opaque_reasons: Vec::new(),
            opaque_witness: Vec::new(),
            unchecked_asserts: Vec::new(),
            last_opaque: None,
        }
    }

    // ────────────────────────────────────────────────────
    // Values the compiler cannot compute
    // ────────────────────────────────────────────────────

    /// Mint an opaque value, recording why.
    fn opaque(&mut self, pos: Pos, what: impl std::fmt::Display) -> Val {
        self.opaque_reasons.push(format!("{}:{}: {}", pos.line, pos.col, what));
        Val::Opaque((self.opaque_reasons.len() - 1) as u32)
    }

    /// `" (it became unknown at ...)"`, or `""` for a value that is known.
    ///
    /// Appended to the existing refusals rather than replacing them, so a
    /// diagnostic that was already good stays word for word what it was and
    /// merely gains the origin.
    fn why_unknown(&self, v: &Val) -> String {
        match v {
            Val::Opaque(id) => {
                format!(" (this value became unknown at {})", self.opaque_reasons[*id as usize])
            }
            _ => String::new(),
        }
    }

    /// Refuse an opaque value at a site where the circuit depends on it.
    ///
    /// Every caller of this is a place where using an unknown would change the
    /// emitted `.r1cs` - which is the one thing the opaque model promises it
    /// cannot do.
    fn require_known(&self, v: &Val, pos: Pos, ctx: &str) -> LResult<()> {
        match v {
            Val::Opaque(id) => Err(err(
                pos,
                format!(
                    "{} cannot use a value the compiler was unable to compute{}. Only a `<--` \
                     may take one, because it emits no constraint.",
                    ctx,
                    format_args!(" — it became unknown at {}", self.opaque_reasons[*id as usize]),
                ),
            )),
            _ => Ok(()),
        }
    }

    fn require_known_slot(&self, s: &VarSlot, pos: Pos, ctx: &str) -> LResult<()> {
        match s {
            Slot::Leaf(v) => self.require_known(v, pos, ctx),
            Slot::Array(items) => {
                items.iter().try_for_each(|i| self.require_known_slot(i, pos, ctx))
            }
        }
    }

    /// Compile `component main` and return the populated emitter.
    pub fn lower(mut self) -> LResult<ZkEmitter> {
        let main = self
            .program
            .main
            .clone()
            .ok_or_else(|| "no `component main` declaration; nothing to compile".to_string())?;

        let tmpl = self.program.template(&main.template).ok_or_else(|| {
            err(main.pos, format!("`component main` names unknown template `{}`", main.template))
        })?;
        if tmpl.is_custom {
            return Err(err(
                main.pos,
                format!(
                    "`{}` is a `custom` template. Custom templates are PLONKish custom gates and \
                     have no R1CS meaning; Y emits R1CS only, and compiling one as an ordinary \
                     template would silently drop the gate it stands for.",
                    main.template
                ),
            ));
        }

        // Template arguments are compile-time by definition - but not
        // necessarily SCALAR. `component main = EscalarMul(8, [x, y])` passes a
        // curve base point as an array literal, and this used to call
        // `eval_expr`, which refuses one. The identical path for a nested
        // component (`Stmt::DeclComponent`) already used `eval_to_slot`, so the
        // construct worked everywhere except at the top level - which is the
        // only place a user writes it directly.
        let mut args = Vec::new();
        for a in &main.args {
            let mut root = Frame::new(String::new());
            args.push(self.eval_to_slot(a, &mut root)?);
        }

        let inst_id = self.instantiate(&main.template, args, "main", main.pos)?;

        // Public inputs are the ones `{public [...]}` names; everything else
        // declared `signal input` is private. Outputs are always public.
        let (inputs, outputs) = {
            let inst = &self.instances[inst_id];
            (inst.input_order.clone(), inst.output_order.clone())
        };
        for name in &main.public {
            if !inputs.contains(name) {
                return Err(err(
                    main.pos,
                    format!(
                        "`public [{}]` names `{}`, which is not an input signal of `{}`",
                        main.public.join(", "),
                        name,
                        main.template
                    ),
                ));
            }
        }
        for name in &outputs {
            let wires = self.flatten_signal(inst_id, name);
            self.emitter.outputs.extend(wires);
        }
        for name in &inputs {
            let wires = self.flatten_signal(inst_id, name);
            if main.public.contains(name) {
                self.emitter.public_inputs.extend(wires);
            } else {
                self.emitter.private_inputs.extend(wires);
            }
        }

        self.report_unsolved_witness();
        self.report_unchecked_asserts();
        self.emitter.run_optimizer();
        Ok(self.emitter)
    }

    /// Say out loud which signals Y cannot solve a witness for.
    ///
    /// The `.r1cs` is complete either way - a `<--` emits no constraint - so
    /// nothing about the circuit is weaker, and a proof built from a witness
    /// obtained elsewhere (circom's own calculator, snarkjs) is unaffected. But
    /// `--witness` will fail the satisfiability check on these circuits, and
    /// finding that out from a satisfiability error rather than from the
    /// compiler is exactly the quiet degradation this repo refuses.
    fn report_unsolved_witness(&self) {
        if self.opaque_witness.is_empty() {
            return;
        }
        // Grouped by the reason TEXT, not by the id: every evaluation of the
        // same expression mints a fresh id, so a loop that shifts 256 signals
        // produces 256 ids all saying the same thing.
        let mut by_reason: HashMap<&str, usize> = HashMap::new();
        for (_, id) in &self.opaque_witness {
            *by_reason.entry(self.opaque_reasons[*id as usize].as_str()).or_insert(0) += 1;
        }
        let mut reasons: Vec<_> = by_reason.into_iter().collect();
        reasons.sort_unstable();
        eprintln!(
            "[Warning] {} signal(s) are assigned with `<--` from a value this compiler cannot \
             evaluate. The emitted .r1cs is complete and correct - `<--` emits no constraint. \
             The witness may still solve: where the author's own constraints determine the \
             signal, Y derives it anyway (circomlib's Sha256 is exactly this - all 256 digest \
             bits are `<--` advice and every one is pinned by a `===` below it). Where they do \
             not, `--witness` reports that no satisfying witness exists rather than writing a \
             zero.",
            self.opaque_witness.len()
        );
        for (reason, n) in reasons {
            eprintln!("          {:>6} from {}", n, reason);
        }
    }

    /// Say out loud which `assert`s Y did not evaluate.
    ///
    /// The `.r1cs` is byte-for-byte circom's either way - `assert` emits no
    /// constraint in circom, measured rather than assumed - so nothing about
    /// the circuit is weaker and no proof is affected. What differs is that
    /// circom's witness calculator ABORTS on a false assert and Y's solver does
    /// not, so a circuit used outside its documented preconditions will produce
    /// a witness here and be refused there.
    ///
    /// Reported for the same reason `report_unsolved_witness` is: learning this
    /// from a divergence against circom later, rather than from the compiler
    /// now, is the quiet degradation this repo refuses.
    fn report_unchecked_asserts(&self) {
        if self.unchecked_asserts.is_empty() {
            return;
        }
        eprintln!(
            "[Warning] {} `assert`(s) over witness-domain values were NOT evaluated. \
             The emitted .r1cs is unaffected - `assert` emits no constraint in circom \
             either - but circom's witness calculator aborts on a false one and Y's \
             solver does not, so a precondition violation will be caught there and not \
             here. Write it as `===` if it is meant to bind the circuit.",
            self.unchecked_asserts.len()
        );
        let mut at = self.unchecked_asserts.clone();
        at.sort_unstable();
        at.dedup();
        for pos in at.iter().take(10) {
            eprintln!("          at {}", pos);
        }
        if at.len() > 10 {
            eprintln!("          ... and {} more", at.len() - 10);
        }
    }

    fn flatten_signal(&self, inst: usize, name: &str) -> Vec<usize> {
        fn walk(s: &SigSlot, out: &mut Vec<usize>) {
            match s {
                Slot::Leaf(w) => out.push(*w),
                Slot::Array(items) => items.iter().for_each(|i| walk(i, out)),
            }
        }
        let mut out = Vec::new();
        if let Some(s) = self.instances[inst].signals.get(name) {
            walk(s, &mut out);
        }
        out
    }

    // ────────────────────────────────────────────────────
    // Template instantiation
    // ────────────────────────────────────────────────────

    fn instantiate(
        &mut self,
        name: &str,
        args: Vec<VarSlot>,
        prefix: &str,
        pos: Pos,
    ) -> LResult<usize> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(err(
                pos,
                format!("template instantiation nested more than {} deep; recursive templates have no finite circuit", MAX_DEPTH),
            ));
        }
        // A template argument sizes arrays and picks loop counts, so it decides
        // the shape of the circuit itself. An unknown one is refused here
        // rather than deep inside the body where the symptom appears.
        for a in &args {
            self.require_known_slot(a, pos, "a template argument")?;
        }
        let tmpl = self
            .program
            .template(name)
            .ok_or_else(|| err(pos, format!("unknown template `{}`", name)))?;
        if tmpl.is_custom {
            return Err(err(
                pos,
                format!("`{}` is a `custom` template; Y emits R1CS and has no PLONKish backend", name),
            ));
        }
        if args.len() != tmpl.params.len() {
            return Err(err(
                pos,
                format!(
                    "template `{}` takes {} parameter(s), got {}",
                    name,
                    tmpl.params.len(),
                    args.len()
                ),
            ));
        }

        let mut frame = Frame::new(prefix.to_string());
        for (p, a) in tmpl.params.iter().zip(args) {
            frame.vars[0].insert(p.clone(), a);
        }

        let body = tmpl.body.clone();
        self.exec_stmt(&body, &mut frame)?;

        let inst = Instance {
            template: name.to_string(),
            signals: frame.signals,
            input_order: frame.input_order,
            output_order: frame.output_order,
        };
        self.instances.push(inst);
        self.depth -= 1;
        Ok(self.instances.len() - 1)
    }

    // ────────────────────────────────────────────────────
    // Statements
    // ────────────────────────────────────────────────────

    /// What in this statement would change the emitted circuit, if anything.
    ///
    /// **This is the guard that makes havoc sound.** When a condition cannot be
    /// evaluated, Y runs neither branch — so it must first be certain that
    /// running neither cannot change the `.r1cs`. If a skipped body could emit
    /// a constraint, declare a signal or instantiate a component, then the set
    /// of constraints would depend on a witness value, and that is refused.
    /// circom refuses the same programs for the same reason.
    ///
    /// Matched exhaustively with no `_ =>` arm, per the design rule in
    /// CLAUDE.md: a new statement kind must be classified deliberately, because
    /// the failure mode of guessing "no effect" is a circuit quietly missing
    /// the constraints its source describes.
    fn constraint_effect(s: &Stmt) -> Option<&'static str> {
        match s {
            Stmt::Block(stmts, _) | Stmt::Seq(stmts, _) => {
                stmts.iter().find_map(Self::constraint_effect)
            }
            Stmt::DeclSignal { .. } => Some("a `signal` declaration"),
            Stmt::DeclComponent { .. } => Some("a `component` declaration"),
            Stmt::Substitution { op, .. } => match op {
                AssignOp::SignalConstrain => Some("a `<==` constraint"),
                // `<--` emits no constraint, so skipping one cannot change the
                // `.r1cs`. It is refused anyway: skipping it leaves the signal
                // with no witness recipe at all, which is a strictly worse
                // outcome than a named refusal, and no circuit here needs it.
                AssignOp::SignalOnly => Some("a `<--` witness assignment to a signal"),
                AssignOp::Var => None,
            },
            Stmt::ConstraintEq(..) => Some("a `===` constraint"),
            Stmt::If { then_branch, else_branch, .. } => Self::constraint_effect(then_branch)
                .or_else(|| else_branch.as_deref().and_then(Self::constraint_effect)),
            Stmt::For { init, step, body, .. } => Self::constraint_effect(init)
                .or_else(|| Self::constraint_effect(step))
                .or_else(|| Self::constraint_effect(body)),
            Stmt::While { body, .. } => Self::constraint_effect(body),
            // A `var` declared inside the branch is scoped to it, so skipping
            // the branch cannot leave a stale value behind; and a `var` emits
            // no constraint whatever it holds.
            Stmt::DeclVar { .. } => None,
            Stmt::Return(..) | Stmt::Assert(..) | Stmt::Log(..) => None,
        }
    }

    /// Every variable name a statement might assign to.
    ///
    /// Over-collection is harmless — a name not in scope is skipped, and a name
    /// that is in scope merely becomes unknown, which is the whole point.
    /// Under-collection is not: it would leave a variable holding the value it
    /// had before a branch that may have changed it.
    fn assigned_vars(s: &Stmt, out: &mut Vec<String>) {
        match s {
            Stmt::Block(stmts, _) | Stmt::Seq(stmts, _) => {
                stmts.iter().for_each(|x| Self::assigned_vars(x, out))
            }
            Stmt::DeclVar { name, .. } => out.push(name.clone()),
            Stmt::Substitution { lhs, op, .. } => {
                if let (AssignOp::Var, Some(n)) = (op, Self::base_name(lhs)) {
                    out.push(n);
                }
            }
            Stmt::If { then_branch, else_branch, .. } => {
                Self::assigned_vars(then_branch, out);
                if let Some(e) = else_branch {
                    Self::assigned_vars(e, out);
                }
            }
            Stmt::For { init, step, body, .. } => {
                Self::assigned_vars(init, out);
                Self::assigned_vars(step, out);
                Self::assigned_vars(body, out);
            }
            Stmt::While { body, .. } => Self::assigned_vars(body, out),
            Stmt::DeclSignal { .. }
            | Stmt::DeclComponent { .. }
            | Stmt::ConstraintEq(..)
            | Stmt::Return(..)
            | Stmt::Assert(..)
            | Stmt::Log(..) => {}
        }
    }

    fn contains_return(s: &Stmt) -> bool {
        match s {
            Stmt::Block(stmts, _) | Stmt::Seq(stmts, _) => stmts.iter().any(Self::contains_return),
            Stmt::Return(..) => true,
            Stmt::If { then_branch, else_branch, .. } => {
                Self::contains_return(then_branch)
                    || else_branch.as_deref().is_some_and(Self::contains_return)
            }
            Stmt::For { init, step, body, .. } => {
                Self::contains_return(init)
                    || Self::contains_return(step)
                    || Self::contains_return(body)
            }
            Stmt::While { body, .. } => Self::contains_return(body),
            Stmt::DeclSignal { .. }
            | Stmt::DeclVar { .. }
            | Stmt::DeclComponent { .. }
            | Stmt::Substitution { .. }
            | Stmt::ConstraintEq(..)
            | Stmt::Assert(..)
            | Stmt::Log(..) => false,
        }
    }

    /// Replace every leaf of a variable slot with the same opaque value.
    fn blank_slot(slot: &mut VarSlot, id: u32) {
        match slot {
            Slot::Leaf(v) => *v = Val::Opaque(id),
            Slot::Array(items) => items.iter_mut().for_each(|i| Self::blank_slot(i, id)),
        }
    }

    /// Execute an `if` / `for` / `while` whose condition Y cannot evaluate.
    ///
    /// Neither branch runs, and every variable a branch might assign becomes
    /// opaque. That is the sound over-approximation in the sense CLAUDE.md's
    /// design rule means: "this value is unknown" is true whichever way the
    /// branch would have gone, so nothing downstream can rely on a guess. The
    /// precondition is `constraint_effect`; without it this would be the
    /// silent-approximation bug the rule exists to prevent.
    ///
    /// A `return` inside a skipped body makes the enclosing function return
    /// opaque immediately: the function may have returned there with an unknown
    /// value, or fallen through to return something else, and opaque is the
    /// join of those two. That is what lets circomlib's `sqrt` — five branches
    /// and two `while` loops over the value being rooted — be called at all.
    fn havoc_branch(
        &mut self,
        cond: &Val,
        bodies: &[&Stmt],
        f: &mut Frame,
        pos: Pos,
    ) -> LResult<Flow> {
        let id = match cond.opaque_id() {
            Some(id) => id,
            None => {
                // A `Lin` or `Quad` condition: the value is a signal expression,
                // known to the witness and not to the compiler.
                let v = self.opaque(
                    pos,
                    format!("a condition whose value is {}, i.e. known only at witness time", cond.kind()),
                );
                v.opaque_id().unwrap()
            }
        };
        for b in bodies {
            if let Some(what) = Self::constraint_effect(b) {
                return Err(err(
                    pos,
                    format!(
                        "this condition cannot be evaluated at compile time, and the branch \
                         contains {}. Which constraints a circuit emits may not depend on a \
                         witness value; use a multiplexer (`sel * a + (1 - sel) * b`) instead. \
                         The condition became unknown at {}",
                        what, self.opaque_reasons[id as usize]
                    ),
                ));
            }
        }
        let mut names = Vec::new();
        for b in bodies {
            Self::assigned_vars(b, &mut names);
        }
        for n in &names {
            if let Some(slot) = f.lookup_var_mut(n) {
                Self::blank_slot(slot, id);
            }
        }
        if bodies.iter().any(|b| Self::contains_return(b)) {
            return Ok(Flow::Return(Slot::Leaf(Val::Opaque(id))));
        }
        Ok(Flow::Normal)
    }

    fn exec_stmt(&mut self, stmt: &Stmt, f: &mut Frame) -> LResult<Flow> {
        match stmt {
            Stmt::Block(stmts, _) => {
                f.push_scope();
                for s in stmts {
                    if let Flow::Return(v) = self.exec_stmt(s, f)? {
                        f.pop_scope();
                        return Ok(Flow::Return(v));
                    }
                }
                f.pop_scope();
                Ok(Flow::Normal)
            }

            // Deliberately NOT a scope: `var a, b;` must leave both names in
            // the scope the declaration was written in.
            Stmt::Seq(stmts, _) => {
                for s in stmts {
                    if let Flow::Return(v) = self.exec_stmt(s, f)? {
                        return Ok(Flow::Return(v));
                    }
                }
                Ok(Flow::Normal)
            }

            Stmt::DeclSignal { kind, name, dims, init, pos } => {
                let dims = self.eval_dims(dims, f)?;
                let slot = self.alloc_signal(&f.prefix, name, &dims);
                f.signals.insert(name.clone(), slot);
                match kind {
                    SignalKind::Input => f.input_order.push(name.clone()),
                    SignalKind::Output => f.output_order.push(name.clone()),
                    SignalKind::Intermediate => {}
                }
                if let Some((op, rhs)) = init {
                    let target = Expr::Var(name.clone(), *pos);
                    self.substitute(&target, *op, rhs, f, *pos)?;
                }
                Ok(Flow::Normal)
            }

            Stmt::DeclVar { name, dims, init, pos } => {
                let dims = self.eval_dims(dims, f)?;
                let slot = match init {
                    Some(e) => self.eval_to_slot(e, f)?,
                    None => Self::zero_slot(&dims),
                };
                f.declare_var(name.clone(), slot, *pos)?;
                Ok(Flow::Normal)
            }

            Stmt::DeclComponent { name, dims, init, pos } => {
                let dims = self.eval_dims(dims, f)?;
                let slot = match init {
                    Some(Expr::TemplateInst(t, args, ipos)) => {
                        if !dims.is_empty() {
                            return Err(err(*pos, "cannot assign a template to a component ARRAY in one statement; declare it then index it"));
                        }
                        let mut argv = Vec::new();
                        for a in args {
                            argv.push(self.eval_to_slot(a, f)?);
                        }
                        let prefix = format!("{}{}.", f.prefix, name);
                        CompSlot::Inst(self.instantiate(t, argv, &prefix, *ipos)?)
                    }
                    Some(other) => {
                        return Err(err(other.pos(), "expected `TemplateName(args)` after `=` in a component declaration"))
                    }
                    None => Self::empty_comp_slot(&dims),
                };
                f.components.insert(name.clone(), slot);
                Ok(Flow::Normal)
            }

            Stmt::Substitution { lhs, op, rhs, pos } => {
                self.substitute(lhs, *op, rhs, f, *pos)?;
                Ok(Flow::Normal)
            }

            Stmt::ConstraintEq(a, b, pos) => {
                let va = self.eval_expr(a, f)?;
                let vb = self.eval_expr(b, f)?;
                self.require_known(&va, *pos, "non-quadratic: a `===` constraint")?;
                self.require_known(&vb, *pos, "non-quadratic: a `===` constraint")?;
                self.constrain_equal(va, vb, *pos)?;
                Ok(Flow::Normal)
            }

            Stmt::If { cond, then_branch, else_branch, pos } => {
                let cv = self.eval_expr(cond, f)?;
                let Some(c) = cv.as_const() else {
                    let mut bodies: Vec<&Stmt> = vec![then_branch];
                    if let Some(e) = else_branch {
                        bodies.push(e);
                    }
                    return self.havoc_branch(&cv, &bodies, f, *pos);
                };
                if !c.is_zero() {
                    self.exec_stmt(then_branch, f)
                } else if let Some(e) = else_branch {
                    self.exec_stmt(e, f)
                } else {
                    Ok(Flow::Normal)
                }
            }

            Stmt::For { init, cond, step, body, pos } => {
                f.push_scope();
                self.exec_stmt(init, f)?;
                let mut iters = 0usize;
                loop {
                    let cv = self.eval_expr(cond, f)?;
                    let Some(c) = cv.as_const() else {
                        // The trip count is a witness value. Run no iteration
                        // and make everything the body or step could assign
                        // unknown - see `havoc_branch`.
                        let r = self.havoc_branch(&cv, &[body, step], f, *pos);
                        f.pop_scope();
                        return r;
                    };
                    if c.is_zero() {
                        break;
                    }
                    iters += 1;
                    if iters > MAX_LOOP_ITERS {
                        f.pop_scope();
                        return Err(err(*pos, format!("loop exceeded {} iterations; circuits are fully unrolled, so this is almost certainly a non-terminating condition", MAX_LOOP_ITERS)));
                    }
                    if let Flow::Return(v) = self.exec_stmt(body, f)? {
                        f.pop_scope();
                        return Ok(Flow::Return(v));
                    }
                    self.exec_stmt(step, f)?;
                }
                f.pop_scope();
                Ok(Flow::Normal)
            }

            Stmt::While { cond, body, pos } => {
                let mut iters = 0usize;
                loop {
                    let cv = self.eval_expr(cond, f)?;
                    let Some(c) = cv.as_const() else {
                        return self.havoc_branch(&cv, &[body], f, *pos);
                    };
                    if c.is_zero() {
                        return Ok(Flow::Normal);
                    }
                    iters += 1;
                    if iters > MAX_LOOP_ITERS {
                        return Err(err(*pos, format!("loop exceeded {} iterations", MAX_LOOP_ITERS)));
                    }
                    if let Flow::Return(v) = self.exec_stmt(body, f)? {
                        return Ok(Flow::Return(v));
                    }
                }
            }

            Stmt::Return(e, _) => Ok(Flow::Return(self.eval_to_slot(e, f)?)),

            Stmt::Assert(e, pos) => {
                // A compile-time `assert` is checked now, and a false one is a
                // hard error - that is the case circom-ecdsa's `assert(n <=
                // 252)` template-parameter guards are.
                //
                // One over a WITNESS-domain value emits no constraint, and that
                // is not an approximation: measured against circom 2.2.3, a
                // circuit with and without `assert(a >= 0)` over a signal has
                // identical non-linear, linear and wire counts. `assert` is a
                // witness-time check in circom too - `ModSubThree` in
                // circom-ecdsa writes `assert(a - b - c + (1 << n) >= 0)` under
                // a comment reading "assume a - b - c + 2**n >= 0", i.e. it
                // documents a PRECONDITION on the caller, not a constraint.
                //
                // So the emitted `.r1cs` is byte-for-byte what circom emits,
                // which is exactly the argument that makes `Val::Opaque` inside
                // a `<--` safe. What is lost is circom's witness-time abort,
                // and Y says so out loud rather than letting it be discovered
                // later - see `report_unchecked_asserts`.
                let v = self.eval_expr(e, f)?;
                match v.as_const() {
                    Some(c) if !c.is_zero() => Ok(Flow::Normal),
                    Some(_) => Err(err(*pos, "assertion failed at compile time")),
                    None => {
                        self.unchecked_asserts
                            .push(format!("{}:{}", pos.line, pos.col));
                        Ok(Flow::Normal)
                    }
                }
            }

            // `log` has no circuit meaning; circom itself strips it from the R1CS.
            Stmt::Log(_, _) => Ok(Flow::Normal),
        }
    }

    // ────────────────────────────────────────────────────
    // Substitution: `<==`, `<--`, `=`
    // ────────────────────────────────────────────────────

    fn substitute(
        &mut self,
        lhs: &Expr,
        op: AssignOp,
        rhs: &Expr,
        f: &mut Frame,
        pos: Pos,
    ) -> LResult<()> {
        // circom 2.1 anonymous components. Handled before `op` is looked at
        // because the left may be a tuple, which is not an lvalue in any other
        // statement.
        if let Expr::AnonComp { template, targs, inputs, pos: apos } = rhs {
            return self.substitute_anon(lhs, op, template, targs, inputs, f, *apos);
        }
        if let Expr::Tuple(items, tpos) = lhs {
            if !items.is_empty() {
                return Err(err(
                    *tpos,
                    "a tuple on the left of a substitution needs an anonymous component \
                     `T(...)(...)` on the right",
                ));
            }
        }
        match op {
            AssignOp::Var => {
                // `sq[i] = Square();` is a component instantiation, but it is
                // lexically identical to a variable assignment - the parser
                // cannot tell them apart without knowing what `sq` was declared
                // as, so it is resolved here.
                if let Some(name) = Self::base_name(lhs) {
                    if f.components.contains_key(&name) {
                        return self.instantiate_into(lhs, &name, rhs, f, pos);
                    }
                }
                let v = self.eval_to_slot(rhs, f)?;
                self.assign_var(lhs, v, f, pos)
            }
            AssignOp::SignalConstrain | AssignOp::SignalOnly => {
                // A signal lvalue may name a whole ARRAY: circom 2.1 allows
                // `c.in <== [a, b]` and `c.in <== other`. `substitute_slot`
                // is the scalar path when the slot turns out to be a leaf.
                let slot = self.resolve_signal_slot(lhs, f, pos)?;
                self.substitute_slot(&slot, op, rhs, f, pos)
            }
        }
    }

    /// Drive one signal slot - a single wire or a whole array - from `rhs`.
    fn substitute_slot(
        &mut self,
        slot: &SigSlot,
        op: AssignOp,
        rhs: &Expr,
        f: &mut Frame,
        pos: Pos,
    ) -> LResult<()> {
        // `c.in <== T()(x)`, and the nested `One()(One()(i))` argument form,
        // both land here. circom allows exactly one output in this position.
        if let Expr::AnonComp { template, targs, inputs, pos: apos } = rhs {
            let outs = self.anon_outputs(template, targs, inputs, f, *apos)?;
            if outs.len() != 1 {
                return Err(err(
                    *apos,
                    format!(
                        "`{}` has {} output signals; a single value is expected here. Use a \
                         tuple `(a, b) <== {}(...)(...)` to take them all",
                        template,
                        outs.len(),
                        template
                    ),
                ));
            }
            let v = Self::sig_slot_to_vals(&outs[0].1);
            return self.constrain_slot_vals(slot, op, &v, pos);
        }
        match slot {
            Slot::Leaf(w) => self.substitute_wire(*w, op, rhs, f, pos),
            Slot::Array(items) => match rhs {
                // Element-wise on the SOURCE EXPRESSIONS, so `<--` keeps its
                // witness-domain lowering per element rather than being forced
                // through the constraint value model.
                Expr::ArrayInline(elems, epos) => {
                    if elems.len() != items.len() {
                        return Err(err(
                            *epos,
                            format!(
                                "array literal has {} element(s) but the signal array has {}",
                                elems.len(),
                                items.len()
                            ),
                        ));
                    }
                    for (sub, e) in items.iter().zip(elems) {
                        self.substitute_slot(sub, op, e, f, pos)?;
                    }
                    Ok(())
                }
                other => {
                    let v = self.eval_to_slot(other, f)?;
                    self.constrain_slot_vals(slot, op, &v, pos)
                }
            },
        }
    }

    /// Constrain a signal slot against an already-evaluated value slot,
    /// leaf by leaf. The shapes must agree exactly.
    fn constrain_slot_vals(
        &mut self,
        slot: &SigSlot,
        op: AssignOp,
        v: &VarSlot,
        pos: Pos,
    ) -> LResult<()> {
        match (slot, v) {
            (Slot::Leaf(w), Slot::Leaf(val)) => {
                if let AssignOp::SignalOnly = op {
                    // `c.in <-- other` would need a witness recipe per wire,
                    // and the value model this arm has already gone through is
                    // the constraint one. Refused by name rather than silently
                    // promoted to `<==`, which would add a constraint the
                    // author chose not to write. The literal form
                    // `c.in <-- [a, b]` recurses on expressions above and is
                    // unaffected.
                    return Err(err(
                        pos,
                        "`<--` onto a whole signal array is only supported from an array \
                         literal `[a, b]`; assign the elements individually",
                    ));
                }
                self.constrain_wire_val(*w, val.clone(), pos)
            }
            (Slot::Array(ws), Slot::Array(vs)) => {
                if ws.len() != vs.len() {
                    return Err(err(
                        pos,
                        format!(
                            "signal array has {} element(s) but the right-hand side has {}",
                            ws.len(),
                            vs.len()
                        ),
                    ));
                }
                for (a, b) in ws.iter().zip(vs) {
                    self.constrain_slot_vals(a, op, b, pos)?;
                }
                Ok(())
            }
            (Slot::Array(_), Slot::Leaf(_)) => Err(err(
                pos,
                "the left-hand side is a signal array but the right-hand side is a single value",
            )),
            (Slot::Leaf(_), Slot::Array(_)) => Err(err(
                pos,
                "the left-hand side is a single signal but the right-hand side is an array",
            )),
        }
    }

    /// The scalar substitution: one signal wire, one right-hand expression.
    fn substitute_wire(
        &mut self,
        wire: usize,
        op: AssignOp,
        rhs: &Expr,
        f: &mut Frame,
        pos: Pos,
    ) -> LResult<()> {
        if let AssignOp::SignalOnly = op {
            // `<--` must NOT be evaluated through the constraint value
            // model. Doing so is what made circomlib's own
            // `bitify.circom` uncompilable: `Num2Bits` writes
            // `out[i] <-- (in >> i) & 1`, and a shift over a signal was
            // refused for having "no R1CS form" - which is true, and
            // irrelevant, because a `<--` right-hand side never becomes
            // a constraint. It is a witness computation, checked
            // afterwards by the `===` the author writes.
            self.last_opaque = None;
            let recipe = self.witness_only_recipe(rhs, f, pos)?;
            if let Some(id) = self.last_opaque.take() {
                self.opaque_witness.push((wire, id));
            }
            self.emitter.set_witness_recipe(wire, recipe);
            return Ok(());
        }
        let v = self.eval_expr(rhs, f)?;
        self.constrain_wire_val(wire, v, pos)
    }

    /// `wire <== value`: one constraint, plus a witness recipe when the value
    /// is quadratic.
    fn constrain_wire_val(&mut self, wire: usize, v: Val, pos: Pos) -> LResult<()> {
        self.require_known(&v, pos, "non-quadratic: a `<==` constraint")?;
        let target = LinearCombination::variable(wire);
        // `a * b + c = target`, one constraint.
        match &v {
            Val::Const(_) | Val::Lin(_) => {
                                // No witness recipe. The constraint is `lc * 1 =
                                // wire`, which is exactly the shape
                                // `build_witness_ir`'s `lc_by_output` scan
                                // reconstructs, so storing one would be a
                                // verbatim second copy of the constraint - and
                                // this is the single most common statement in
                                // any circom program. If the linear-substitution
                                // pass later deletes the constraint, it writes
                                // the wire a recipe of its own in exchange.
                                self.emitter.push_constraint(
                                    v.lc().unwrap(),
                                    LinearCombination::constant(Fr::one()),
                                    target,
                                );
                            }
                            Val::Quad(a, b, c) => {
                                // a * b = target - c
                                let mut rhs_lc = target.clone();
                                rhs_lc.add_linear(c, Fr::zero().sub(&Fr::one()));
                                rhs_lc.simplify();
                                self.emitter.push_constraint(a.clone(), b.clone(), rhs_lc);
                                self.emitter.set_witness_recipe(
                                    wire,
                                    WitnessOp::MulAddLc(a.clone(), b.clone(), c.clone()),
                                );
                            }
                            // Rejected by `require_known` above, ten lines up.
                            // Spelled out rather than folded into an `_` arm so
                            // that a future `Val` variant is a compile error
                            // here, where it would silently emit no constraint.
            Val::Opaque(_) => unreachable!(
                "`require_known` rejects an opaque `<==` right-hand side"
            ),
        }
        Ok(())
    }

    /// The witness recipe for a `<--` right-hand side.
    ///
    /// `<--` is where circuits legitimately compute things R1CS cannot express
    /// directly - field division above all - so this accepts a little more than
    /// `<==` does, and still refuses by name rather than guessing.
    fn witness_only_recipe(
        &mut self,
        rhs: &Expr,
        f: &mut Frame,
        _pos: Pos,
    ) -> LResult<WitnessOp> {
        // Bit extraction: `(e >> k) & 1` and `e & 1`, with `k` compile-time.
        //
        // This is `Num2Bits`, and through it `Bits2Num`, `comparators.circom`,
        // `aliascheck.circom` and every range check and comparison built on
        // them - i.e. most of a real circuit. The shift has no R1CS form and
        // does not need one; `out[i]` is constrained afterwards by
        // `out[i] * (out[i] - 1) === 0` and the recomposition `lc1 === in`,
        // both of which this front end already emits.
        if let Expr::Binary(BinOp::BitAnd, masked, mask, _) = rhs {
            if let Ok(Val::Const(m)) = self.eval_expr(mask, f) {
                if m == Fr::one() {
                    let (base, bit) = match &**masked {
                        Expr::Binary(BinOp::Shr, b, k, _) => {
                            let kv = self.eval_expr(k, f)?;
                            match kv {
                                Val::Const(c) => match c.to_u64() {
                                    // A shift wider than the field is not a bit
                                    // of anything; fall through and let the
                                    // ordinary path refuse it by name.
                                    Some(n) if n < 256 => (&**b, n as u32),
                                    _ => (rhs, u32::MAX),
                                },
                                _ => (rhs, u32::MAX),
                            }
                        }
                        other => (other, 0),
                    };
                    if bit != u32::MAX {
                        if let Some(lc) = self.eval_expr(base, f)?.lc() {
                            return Ok(WitnessOp::BitOfLc { lc, bit });
                        }
                    }
                }
            }
        }

        // `cond ? t : e` over signals. Legal in a `<--` and nowhere else.
        //
        // This is circomlib's `IsZero`, verbatim:
        //
        //     inv <-- in != 0 ? 1/in : 0;
        //     out <== -in*inv + 1;
        //     in*out === 0;
        //
        // and `IsZero` is underneath `IsEqual`, `ForceEqualIfEnabled`, the SMT
        // circuits, `Multiplexer` and every EdDSA verifier - so refusing it cost
        // a third of circomlib. Both comparisons reduce to one test against
        // zero: `a == b` is `a - b == 0`, and `!=` is the same with the branches
        // swapped.
        //
        // Being permissive HERE is safe in a way it would not be one line over,
        // and the reason is worth stating: `<--` emits no constraint. A witness
        // value this computes is not trusted by anything - it still has to
        // satisfy the `===` the author wrote (`in*out === 0` above). Getting it
        // wrong yields an unsatisfiable circuit, never an unsound proof. The
        // same expression on the left of a `<==` is refused, as it must be.
        if let Expr::Ternary(cond, then_e, else_e, tpos) = rhs {
            if let Expr::Binary(op @ (BinOp::Eq | BinOp::Neq), a, b, _) = &**cond {
                let va = self.eval_expr(a, f)?;
                let vb = self.eval_expr(b, f)?;
                if let (Some(la), Some(lb)) = (va.lc(), vb.lc()) {
                    // cond_lc = a - b, zero exactly when they are equal.
                    let mut cond_lc = la;
                    cond_lc.add_linear(&lb, Fr::zero().sub(&Fr::one()));
                    cond_lc.simplify();
                    let t = self.witness_only_recipe(then_e, f, *tpos)?;
                    let e = self.witness_only_recipe(else_e, f, *tpos)?;
                    // `IfZeroLc` takes the ZERO branch first, i.e. the `==` one.
                    let (zero_branch, nonzero_branch) = match op {
                        BinOp::Eq => (t, e),
                        _ => (e, t),
                    };
                    return Ok(WitnessOp::IfZeroLc(
                        cond_lc,
                        Box::new(zero_branch),
                        Box::new(nonzero_branch),
                    ));
                }
            }
        }

        // `x <-- a / b` with a signal divisor is the canonical use.
        if let Expr::Binary(BinOp::Div, a, b, _) = rhs {
            let va = self.eval_expr(a, f)?;
            let vb = self.eval_expr(b, f)?;
            if let (Some(la), Some(lb)) = (va.lc(), vb.lc()) {
                return Ok(WitnessOp::DivLc(la, lb));
            }
        }
        if let Expr::Binary(BinOp::IntDiv, a, b, _) = rhs {
            let va = self.eval_expr(a, f)?;
            let vb = self.eval_expr(b, f)?;
            if let (Some(la), Some(lb)) = (va.lc(), vb.lc()) {
                return Ok(WitnessOp::IntDivLc(la, lb));
            }
        }
        if let Expr::Binary(BinOp::Mod, a, b, _) = rhs {
            let va = self.eval_expr(a, f)?;
            let vb = self.eval_expr(b, f)?;
            if let (Some(la), Some(lb)) = (va.lc(), vb.lc()) {
                return Ok(WitnessOp::IntModLc(la, lb));
            }
        }
        // Anything else still goes through the ordinary value model, so an
        // unrecognised construct is refused by name rather than guessed at.
        let v = self.eval_expr(rhs, f)?;
        match &v {
            Val::Const(_) | Val::Lin(_) => Ok(WitnessOp::MulLc(
                v.lc().unwrap(),
                LinearCombination::constant(Fr::one()),
            )),
            Val::Quad(a, b, c) => Ok(WitnessOp::MulAddLc(a.clone(), b.clone(), c.clone())),
            // The one position where an opaque value is legal, and the reason
            // it is legal is that this emits no constraint: the `.r1cs` is
            // byte-for-byte what a compiler that could evaluate the expression
            // would produce. Only the witness is poorer, and `lower` says so.
            Val::Opaque(id) => {
                self.last_opaque = Some(*id);
                Ok(WitnessOp::Unknown)
            }
        }
    }

    fn constrain_equal(&mut self, a: Val, b: Val, pos: Pos) -> LResult<()> {
        // Move everything to `lhs - rhs = 0`, then express as A * B = C.
        let neg = Fr::zero().sub(&Fr::one());
        match (a, b) {
            (Val::Quad(qa, qb, qc), other) | (other, Val::Quad(qa, qb, qc)) => {
                let Some(o) = other.lc() else {
                    return Err(err(
                        pos,
                        "non-quadratic constraint: both sides of `===` are quadratic, which \
                         exceeds what one R1CS constraint can express. Introduce an intermediate \
                         signal for one side.",
                    ));
                };
                // qa*qb + qc = o   =>   qa*qb = o - qc
                let mut c = o;
                c.add_linear(&qc, neg);
                c.simplify();
                self.emitter.push_constraint(qa, qb, c);
                Ok(())
            }
            (x, y) => {
                // (lx - ly) * 1 = 0
                let mut d = x.into_lc().unwrap();
                y.add_into(&mut d, neg);
                d.simplify();
                self.emitter.push_constraint(
                    d,
                    LinearCombination::constant(Fr::one()),
                    LinearCombination::zero(),
                );
                Ok(())
            }
        }
    }

    // ────────────────────────────────────────────────────
    // Names, arrays, signals
    // ────────────────────────────────────────────────────

    fn eval_dims(&mut self, dims: &[Expr], f: &mut Frame) -> LResult<Vec<usize>> {
        let mut out = Vec::new();
        for d in dims {
            let v = self.eval_expr(d, f)?;
            let c = v.as_const().ok_or_else(|| {
                err(
                    d.pos(),
                    format!("array dimensions must be known at compile time{}", self.why_unknown(&v)),
                )
            })?;
            let n = c.to_u64().ok_or_else(|| err(d.pos(), "array dimension does not fit in 64 bits"))?;
            out.push(n as usize);
        }
        Ok(out)
    }

    fn alloc_signal(&mut self, prefix: &str, name: &str, dims: &[usize]) -> SigSlot {
        if dims.is_empty() {
            let w = self.emitter.alloc_wire(&format!("{}{}", prefix, name));
            return Slot::Leaf(w);
        }
        let (head, tail) = (dims[0], &dims[1..]);
        let mut items = Vec::with_capacity(head);
        for i in 0..head {
            items.push(self.alloc_signal(prefix, &format!("{}[{}]", name, i), tail));
        }
        Slot::Array(items)
    }

    fn zero_slot(dims: &[usize]) -> VarSlot {
        if dims.is_empty() {
            return Slot::Leaf(Val::Const(Fr::zero()));
        }
        Slot::Array((0..dims[0]).map(|_| Self::zero_slot(&dims[1..])).collect())
    }

    fn empty_comp_slot(dims: &[usize]) -> CompSlot {
        if dims.is_empty() {
            return CompSlot::Empty;
        }
        CompSlot::Array((0..dims[0]).map(|_| Self::empty_comp_slot(&dims[1..])).collect())
    }

    /// Split `a[i][j]` into the base name and the evaluated indices.
    fn split_indices(&mut self, e: &Expr, f: &mut Frame) -> LResult<(Expr, Vec<usize>)> {
        match e {
            Expr::Index(base, idx, pos) => {
                let (b, mut idxs) = self.split_indices(base, f)?;
                let v = self.eval_expr(idx, f)?;
                let c = v.as_const().ok_or_else(|| {
                    err(
                        *pos,
                        format!(
                            "array indices must be known at compile time; a signal-dependent \
                             index needs an explicit multiplexer{}",
                            self.why_unknown(&v)
                        ),
                    )
                })?;
                let n = c.to_u64().ok_or_else(|| err(*pos, "array index does not fit in 64 bits"))?;
                idxs.push(n as usize);
                Ok((b, idxs))
            }
            other => Ok((other.clone(), Vec::new())),
        }
    }

    /// Indexing an UNKNOWN yields an unknown.
    ///
    /// A `function` whose branch depends on a signal value is havoc'd, and a
    /// `return` inside a skipped branch makes it return `Val::Opaque` - a
    /// SCALAR - where the caller declared an array. Indexing that is not the
    /// user error "too many indices"; it is the opaque value propagating,
    /// exactly as it already does through `add_vals` and `mul_vals`.
    ///
    /// circom-ecdsa's `secp256k1_addunequal_func` is the motivating case:
    /// `mod_inv` over signal-derived values cannot be evaluated, so the whole
    /// function comes back opaque and every `out[0][i]` below it looked like a
    /// mis-indexed scalar.
    ///
    /// Sound for the same reason the rest of the opaque model is: the value
    /// may only reach a site that emits no constraint, and `require_known`
    /// refuses it everywhere else.
    ///
    /// Returns the id when the walk meets an opaque leaf before the indices
    /// are exhausted. `None` covers the ordinary walk AND a genuine
    /// too-many-indices, which the caller still reports as before.
    fn opaque_through_index(slot: &VarSlot, idxs: &[usize]) -> Option<u32> {
        let mut cur = slot;
        for i in idxs {
            match cur {
                Slot::Leaf(Val::Opaque(id)) => return Some(*id),
                Slot::Leaf(_) => return None,
                Slot::Array(items) => cur = items.get(*i)?,
            }
        }
        None
    }

    fn index_slot<'s, T>(slot: &'s Slot<T>, idxs: &[usize], pos: Pos) -> LResult<&'s Slot<T>> {
        let mut cur = slot;
        for (d, i) in idxs.iter().enumerate() {
            match cur {
                Slot::Array(items) => {
                    cur = items.get(*i).ok_or_else(|| {
                        err(pos, format!("index {} out of bounds in dimension {} (length {})", i, d, items.len()))
                    })?;
                }
                Slot::Leaf(_) => return Err(err(pos, "too many indices for this array")),
            }
        }
        Ok(cur)
    }

    /// Resolve an lvalue to the signal SLOT it names, following `c.out[i]`
    /// into subcomponents. The slot may be a single wire or a whole array;
    /// circom 2.1 lets both sides of a substitution be arrays.
    fn resolve_signal_slot(&mut self, e: &Expr, f: &mut Frame, pos: Pos) -> LResult<SigSlot> {
        let (base, idxs) = self.split_indices(e, f)?;
        match &base {
            Expr::Var(name, p) => {
                let slot = f
                    .signals
                    .get(name)
                    .ok_or_else(|| err(*p, format!("`{}` is not a signal of this template", name)))?
                    .clone();
                Ok(Self::index_slot(&slot, &idxs, pos)?.clone())
            }
            Expr::Member(obj, field, p) => {
                let inst = self.resolve_component(obj, f, *p)?;
                let slot = self.instances[inst]
                    .signals
                    .get(field)
                    .ok_or_else(|| {
                        err(*p, format!("component of template `{}` has no signal `{}`", self.instances[inst].template, field))
                    })?
                    .clone();
                Ok(Self::index_slot(&slot, &idxs, pos)?.clone())
            }
            other => Err(err(other.pos(), "left-hand side of a signal assignment must be a signal")),
        }
    }

    /// Resolve an lvalue to the wire of a signal. `resolve_signal_slot` plus
    /// the requirement that it name a single one.
    fn resolve_signal_wire(&mut self, e: &Expr, f: &mut Frame, pos: Pos) -> LResult<usize> {
        let name = Self::base_name(e).unwrap_or_else(|| "this signal".to_string());
        Ok(*self.resolve_signal_slot(e, f, pos)?.leaf(pos, &name)?)
    }

    // ────────────────────────────────────────────────────
    // circom 2.1 anonymous components
    // ────────────────────────────────────────────────────

    /// `lhs <== T(targs)(inputs)`, where `lhs` may be a tuple.
    fn substitute_anon(
        &mut self,
        lhs: &Expr,
        op: AssignOp,
        template: &str,
        targs: &[Expr],
        inputs: &[(Option<String>, Expr)],
        f: &mut Frame,
        pos: Pos,
    ) -> LResult<()> {
        // circom: "Anonymous components only admit the use of the operator
        // <==". Matched exactly - accepting `<--` here would bind the outputs
        // with no constraint while still constraining the inputs, which is a
        // circuit the author cannot have meant and which circom will not
        // produce.
        if let AssignOp::SignalOnly = op {
            return Err(err(
                pos,
                "an anonymous component may only be used with `<==`; `<--` would leave its \
                 outputs unconstrained",
            ));
        }
        let outs = self.anon_outputs(template, targs, inputs, f, pos)?;
        let targets: Vec<&Expr> = match lhs {
            Expr::Tuple(items, _) => items.iter().collect(),
            other => vec![other],
        };
        if targets.len() != outs.len() {
            return Err(err(
                pos,
                format!(
                    "`{}` has {} output signal(s) but {} target(s) were given",
                    template,
                    outs.len(),
                    targets.len()
                ),
            ));
        }
        for (t, (_, slot)) in targets.into_iter().zip(outs) {
            // `_` discards an output. circom spells it the same way, and the
            // component is still instantiated and its inputs still
            // constrained - only the binding is dropped.
            if matches!(t, Expr::Var(n, _) if n == "_") {
                continue;
            }
            let v = Self::sig_slot_to_vals(&slot);
            self.substitute_val(t, op, v, f, pos)?;
        }
        Ok(())
    }

    /// Bind an already-evaluated value slot to an lvalue.
    fn substitute_val(
        &mut self,
        lhs: &Expr,
        op: AssignOp,
        v: VarSlot,
        f: &mut Frame,
        pos: Pos,
    ) -> LResult<()> {
        match op {
            AssignOp::Var => self.assign_var(lhs, v, f, pos),
            AssignOp::SignalConstrain => {
                let slot = self.resolve_signal_slot(lhs, f, pos)?;
                self.constrain_slot_vals(&slot, op, &v, pos)
            }
            // Refused by `substitute_anon`, the only caller, before it gets
            // here. Named rather than left to an `_` arm so a second caller
            // cannot acquire a silent `<--` lowering.
            AssignOp::SignalOnly => unreachable!("`<--` is refused for anonymous components"),
        }
    }

    /// Instantiate `T(targs)`, constrain its inputs from `inputs`, and return
    /// its output signals in declaration order.
    ///
    /// This is a desugaring, and a checkable one: circom's own `.r1cs` for
    /// `o <== One()(i)` is byte-identical to the one for
    /// `component c = One(); c.a <== i; o <== c.out;`, which is what
    /// `tests/circom_anonymous_components.rs` asserts of Y's.
    fn anon_outputs(
        &mut self,
        template: &str,
        targs: &[Expr],
        inputs: &[(Option<String>, Expr)],
        f: &mut Frame,
        pos: Pos,
    ) -> LResult<Vec<(String, SigSlot)>> {
        if self.program.template(template).is_none() {
            return Err(err(pos, format!("unknown template `{}`", template)));
        }
        let mut argv = Vec::new();
        for a in targs {
            argv.push(self.eval_to_slot(a, f)?);
        }
        let n = f.anon_count;
        f.anon_count += 1;
        let prefix = format!("{}anon{}_{}.", f.prefix, n, template);
        let inst = self.instantiate(template, argv, &prefix, pos)?;

        let input_order = self.instances[inst].input_order.clone();
        if inputs.len() != input_order.len() {
            return Err(err(
                pos,
                format!(
                    "the number of template input signals must coincide with the number of \
                     input parameters: `{}` declares {} input(s), {} given",
                    template,
                    input_order.len(),
                    inputs.len()
                ),
            ));
        }

        // circom takes either an all-positional or an all-named list, never a
        // mixture, and rejects a repeated or unknown name.
        let named = inputs.iter().filter(|(n, _)| n.is_some()).count();
        let bound: Vec<(String, &Expr)> = if named == 0 {
            input_order.iter().cloned().zip(inputs.iter().map(|(_, e)| e)).collect()
        } else if named == inputs.len() {
            let mut seen: Vec<&str> = Vec::new();
            let mut out = Vec::new();
            for (name, e) in inputs {
                let name = name.as_ref().unwrap();
                if !input_order.iter().any(|s| s == name) {
                    return Err(err(
                        e.pos(),
                        format!("`{}` has no input signal `{}`", template, name),
                    ));
                }
                if seen.contains(&name.as_str()) {
                    return Err(err(
                        e.pos(),
                        format!("input signal `{}` of `{}` is given twice", name, template),
                    ));
                }
                seen.push(name);
                out.push((name.clone(), e));
            }
            out
        } else {
            return Err(err(
                pos,
                "an anonymous component's inputs must be either all positional or all named",
            ));
        };

        for (name, e) in bound {
            let slot = self.instances[inst].signals[&name].clone();
            self.substitute_slot(&slot, AssignOp::SignalConstrain, e, f, pos)?;
        }

        let outs = self.instances[inst].output_order.clone();
        Ok(outs
            .iter()
            .map(|o| (o.clone(), self.instances[inst].signals[o].clone()))
            .collect())
    }

    fn resolve_component(&mut self, e: &Expr, f: &mut Frame, pos: Pos) -> LResult<usize> {
        let (base, idxs) = self.split_indices(e, f)?;
        let Expr::Var(name, p) = &base else {
            return Err(err(pos, "expected a component name"));
        };
        let slot = f
            .components
            .get(name)
            .ok_or_else(|| err(*p, format!("`{}` is not a component", name)))?;
        let mut cur = slot;
        for i in &idxs {
            match cur {
                CompSlot::Array(items) => {
                    cur = items
                        .get(*i)
                        .ok_or_else(|| err(pos, format!("component index {} out of bounds", i)))?
                }
                _ => return Err(err(pos, "too many indices for this component array")),
            }
        }
        match cur {
            CompSlot::Inst(id) => Ok(*id),
            CompSlot::Empty => Err(err(
                pos,
                format!("component `{}` is used before it is assigned a template", name),
            )),
            CompSlot::Array(_) => Err(err(pos, format!("component `{}` needs an index", name))),
        }
    }

    fn base_name(e: &Expr) -> Option<String> {
        match e {
            Expr::Var(n, _) => Some(n.clone()),
            Expr::Index(b, _, _) => Self::base_name(b),
            _ => None,
        }
    }

    /// `c = T(args)` / `c[i] = T(args)`.
    fn instantiate_into(
        &mut self,
        lhs: &Expr,
        name: &str,
        rhs: &Expr,
        f: &mut Frame,
        pos: Pos,
    ) -> LResult<()> {
        let (tname, targs, tpos) = match rhs {
            Expr::Call(t, a, p) => (t.clone(), a.clone(), *p),
            Expr::TemplateInst(t, a, p) => (t.clone(), a.clone(), *p),
            other => {
                return Err(err(
                    other.pos(),
                    format!("`{}` is a component; the right-hand side must be `TemplateName(...)`", name),
                ))
            }
        };
        if self.program.template(&tname).is_none() {
            return Err(err(tpos, format!("unknown template `{}`", tname)));
        }
        let (_, idxs) = self.split_indices(lhs, f)?;

        let mut argv = Vec::new();
        for a in &targs {
            argv.push(self.eval_to_slot(a, f)?);
        }
        let suffix: String = idxs.iter().map(|i| format!("[{}]", i)).collect();
        let prefix = format!("{}{}{}.", f.prefix, name, suffix);
        let inst = self.instantiate(&tname, argv, &prefix, tpos)?;

        let slot = f.components.get_mut(name).unwrap();
        Self::store_comp_slot(slot, &idxs, CompSlot::Inst(inst), pos)
    }

    fn store_comp_slot(
        slot: &mut CompSlot,
        idxs: &[usize],
        v: CompSlot,
        pos: Pos,
    ) -> LResult<()> {
        if idxs.is_empty() {
            *slot = v;
            return Ok(());
        }
        match slot {
            CompSlot::Array(items) => {
                let i = idxs[0];
                let len = items.len();
                let item = items.get_mut(i).ok_or_else(|| {
                    err(pos, format!("component index {} out of bounds (length {})", i, len))
                })?;
                Self::store_comp_slot(item, &idxs[1..], v, pos)
            }
            _ => Err(err(pos, "too many indices for this component")),
        }
    }

    fn assign_var(&mut self, lhs: &Expr, v: VarSlot, f: &mut Frame, pos: Pos) -> LResult<()> {
        let (base, idxs) = self.split_indices(lhs, f)?;
        // `c[i] = T(...)` assigns into a component array.
        if let Expr::Var(name, _) = &base {
            let slot = f
                .lookup_var_mut(name)
                .ok_or_else(|| err(pos, format!("`{}` is not a variable in scope", name)))?;
            return Self::store_slot(slot, &idxs, v, pos);
        }
        Err(err(pos, "left-hand side of `=` must be a variable"))
    }

    fn store_slot(slot: &mut VarSlot, idxs: &[usize], v: VarSlot, pos: Pos) -> LResult<()> {
        if idxs.is_empty() {
            *slot = v;
            return Ok(());
        }
        match slot {
            Slot::Array(items) => {
                let i = idxs[0];
                let len = items.len();
                let item = items
                    .get_mut(i)
                    .ok_or_else(|| err(pos, format!("index {} out of bounds (length {})", i, len)))?;
                Self::store_slot(item, &idxs[1..], v, pos)
            }
            Slot::Leaf(_) => Err(err(pos, "too many indices for this variable")),
        }
    }

    // ────────────────────────────────────────────────────
    // Expressions
    // ────────────────────────────────────────────────────

    fn eval_to_slot(&mut self, e: &Expr, f: &mut Frame) -> LResult<VarSlot> {
        match e {
            // `var c = Poseidon(2)([a, b]);` — an anonymous component in a
            // value position. semaphore is written this way. Handled here and
            // NOT in `eval_expr`, so `o <== One()(i) + 3` is still refused
            // exactly where circom refuses it: an operand of an arithmetic
            // expression reaches `eval_expr` directly.
            Expr::AnonComp { template, targs, inputs, pos } => {
                let outs = self.anon_outputs(template, targs, inputs, f, *pos)?;
                if outs.len() != 1 {
                    return Err(err(
                        *pos,
                        format!(
                            "`{}` has {} output signals; a single value is expected here. Use a \
                             tuple `(a, b) = {}(...)(...)` to take them all",
                            template,
                            outs.len(),
                            template
                        ),
                    ));
                }
                Ok(Self::sig_slot_to_vals(&outs[0].1))
            }
            Expr::ArrayInline(items, _) => {
                let mut out = Vec::with_capacity(items.len());
                for i in items {
                    out.push(self.eval_to_slot(i, f)?);
                }
                Ok(Slot::Array(out))
            }
            Expr::Call(name, args, pos) if self.program.function(name).is_some() => {
                self.call_function(name, args, f, *pos)
            }
            Expr::Var(name, _) if f.lookup_var(name).is_some() => {
                Ok(f.lookup_var(name).unwrap().clone())
            }
            // A whole signal ARRAY, passed by name. circom allows this and
            // circomlib relies on it: `sha256compression(hin, inp)` hands a
            // function 256 and 512 input signals. Evaluating it as a scalar is
            // what produced "hin is an array; expected a single value".
            Expr::Var(name, _) if f.signals.contains_key(name) => {
                Ok(Self::sig_slot_to_vals(&f.signals[name]))
            }
            Expr::Index(..) => {
                // Could be a whole sub-array of a var or of a signal.
                let (base, idxs) = self.split_indices(e, f)?;
                if let Expr::Var(name, _) = &base {
                    if let Some(slot) = f.lookup_var(name) {
                        if let Some(id) = Self::opaque_through_index(slot, &idxs) {
                            return Ok(Slot::Leaf(Val::Opaque(id)));
                        }
                        let sub = Self::index_slot(slot, &idxs, e.pos())?;
                        return Ok(sub.clone());
                    }
                    if let Some(slot) = f.signals.get(name) {
                        let sub = Self::index_slot(slot, &idxs, e.pos())?;
                        return Ok(Self::sig_slot_to_vals(sub));
                    }
                }
                if let Expr::Member(obj, field, p) = &base {
                    if let Some(sub) = self.member_signal_slot(obj, field, &idxs, f, *p)? {
                        return Ok(sub);
                    }
                }
                Ok(Slot::Leaf(self.eval_expr(e, f)?))
            }
            Expr::Member(obj, field, p) => {
                if let Some(sub) = self.member_signal_slot(obj, field, &[], f, *p)? {
                    return Ok(sub);
                }
                Ok(Slot::Leaf(self.eval_expr(e, f)?))
            }
            _ => Ok(Slot::Leaf(self.eval_expr(e, f)?)),
        }
    }

    /// A signal slot read as values: every wire becomes a one-term linear
    /// combination, preserving the array shape.
    fn sig_slot_to_vals(s: &SigSlot) -> VarSlot {
        match s {
            Slot::Leaf(w) => Slot::Leaf(Val::Lin(LinearCombination::variable(*w))),
            Slot::Array(items) => {
                Slot::Array(items.iter().map(Self::sig_slot_to_vals).collect())
            }
        }
    }

    /// `c.out` / `c.out[i]` as a value slot, or `None` if it is a single wire
    /// (which the ordinary scalar path already handles).
    fn member_signal_slot(
        &mut self,
        obj: &Expr,
        field: &str,
        idxs: &[usize],
        f: &mut Frame,
        pos: Pos,
    ) -> LResult<Option<VarSlot>> {
        let inst = self.resolve_component(obj, f, pos)?;
        let Some(slot) = self.instances[inst].signals.get(field) else {
            return Ok(None);
        };
        let sub = Self::index_slot(slot, idxs, pos)?;
        match sub {
            Slot::Leaf(_) => Ok(None),
            Slot::Array(_) => Ok(Some(Self::sig_slot_to_vals(sub))),
        }
    }

    fn call_function(
        &mut self,
        name: &str,
        args: &[Expr],
        f: &mut Frame,
        pos: Pos,
    ) -> LResult<VarSlot> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(err(pos, format!("function calls nested more than {} deep", MAX_DEPTH)));
        }
        // Copied out of `self` so the borrow follows the program, not the
        // lowerer. This used to `.clone()` the whole function - body AST
        // included - on every call, which for circomlib's constant tables means
        // deep-copying a 195-element array literal to read one entry of it.
        let func = self.program.function(name).unwrap();
        if args.len() != func.params.len() {
            return Err(err(
                pos,
                format!("function `{}` takes {} argument(s), got {}", name, func.params.len(), args.len()),
            ));
        }

        let mut argv = Vec::with_capacity(args.len());
        for a in args {
            argv.push(self.eval_to_slot(a, f)?);
        }

        // Only scalar arguments are used as a cache key. Array arguments are
        // rare, and hashing one costs more than most function bodies.
        let key = argv
            .iter()
            .map(|s| match s {
                Slot::Leaf(Val::Const(c)) => Some(*c),
                _ => None,
            })
            .collect::<Option<Vec<Fr>>>()
            .map(|k| (name.to_string(), k));
        if let Some(k) = &key {
            if let Some(hit) = self.fn_cache.get(k) {
                self.depth -= 1;
                return Ok(hit.clone());
            }
        }

        let mut inner = Frame::new(f.prefix.clone());
        for (p, v) in func.params.iter().zip(argv) {
            inner.vars[0].insert(p.clone(), v);
        }
        let flow = self.exec_stmt(&func.body, &mut inner)?;
        self.depth -= 1;
        match flow {
            Flow::Return(v) => {
                if let Some(k) = key {
                    self.fn_cache.insert(k, v.clone());
                }
                Ok(v)
            }
            Flow::Normal => Err(err(pos, format!("function `{}` returned no value on this path", name))),
        }
    }

    fn eval_expr(&mut self, e: &Expr, f: &mut Frame) -> LResult<Val> {
        match e {
            Expr::Number(n, _) => Ok(Val::Const(Fr::from_biguint_ref(n))),

            Expr::Var(name, pos) => {
                if let Some(slot) = f.lookup_var(name) {
                    return Ok(slot.leaf(*pos, name)?.clone());
                }
                if let Some(slot) = f.signals.get(name) {
                    let w = *slot.leaf(*pos, name)?;
                    return Ok(Val::Lin(LinearCombination::variable(w)));
                }
                Err(err(*pos, format!("`{}` is not defined", name)))
            }

            Expr::Index(..) => {
                let (base, idxs) = self.split_indices(e, f)?;
                let pos = e.pos();
                if let Expr::Var(name, _) = &base {
                    if let Some(slot) = f.lookup_var(name) {
                        if let Some(id) = Self::opaque_through_index(slot, &idxs) {
                            return Ok(Val::Opaque(id));
                        }
                        return Ok(Self::index_slot(slot, &idxs, pos)?.leaf(pos, name)?.clone());
                    }
                    if let Some(slot) = f.signals.get(name) {
                        let w = *Self::index_slot(slot, &idxs, pos)?.leaf(pos, name)?;
                        return Ok(Val::Lin(LinearCombination::variable(w)));
                    }
                    return Err(err(pos, format!("`{}` is not defined", name)));
                }
                if let Expr::Member(obj, field, p) = &base {
                    let inst = self.resolve_component(obj, f, *p)?;
                    let slot = self.instances[inst]
                        .signals
                        .get(field)
                        .ok_or_else(|| err(*p, format!("no signal `{}` on that component", field)))?
                        .clone();
                    let w = *Self::index_slot(&slot, &idxs, pos)?.leaf(pos, field)?;
                    return Ok(Val::Lin(LinearCombination::variable(w)));
                }
                Err(err(pos, "unsupported indexed expression"))
            }

            Expr::Member(obj, field, pos) => {
                let inst = self.resolve_component(obj, f, *pos)?;
                let slot = self.instances[inst]
                    .signals
                    .get(field)
                    .ok_or_else(|| {
                        err(*pos, format!("component of template `{}` has no signal `{}`", self.instances[inst].template, field))
                    })?
                    .clone();
                let w = *slot.leaf(*pos, field)?;
                Ok(Val::Lin(LinearCombination::variable(w)))
            }

            Expr::Unary(op, a, pos) => {
                let v = self.eval_expr(a, f)?;
                match op {
                    UnOp::Neg => self.mul_vals(Val::Const(Fr::zero().sub(&Fr::one())), v, *pos),
                    UnOp::Not => {
                        let Some(c) = v.as_const() else {
                            return Ok(self.opaque(*pos, "`!` applied to a value that is not a compile-time constant; it has no constraint form"));
                        };
                        Ok(Val::Const(if c.is_zero() { Fr::one() } else { Fr::zero() }))
                    }
                    UnOp::BitNot => {
                        let Some(c) = v.as_const() else {
                            return Ok(self.opaque(*pos, "`~` applied to a value that is not a compile-time constant"));
                        };
                        // circom's ~ is over 254-bit two's complement in Fr.
                        let l = c.to_limbs();
                        let inv = [!l[0], !l[1], !l[2], !l[3]];
                        Ok(Val::Const(Fr::from_limbs_reduce(inv)))
                    }
                }
            }

            Expr::Ternary(c, a, b, pos) => {
                let cv = self.eval_expr(c, f)?;
                let Some(cc) = cv.as_const() else {
                    // Neither branch is evaluated. An expression cannot emit a
                    // constraint in this front end - only statements can - so
                    // unlike `Stmt::If` there is nothing to check first.
                    // `<--` has its own richer handling of `? :` in
                    // `witness_only_recipe`, which runs before this and is
                    // what `IsZero` needs; this is the fallback for every
                    // other position, where the value is refused anyway.
                    if let Some(id) = cv.opaque_id() {
                        return Ok(Val::Opaque(id));
                    }
                    return Ok(self.opaque(
                        *pos,
                        format!(
                            "a `? :` whose condition is {}, i.e. known only at witness time. \
                             As a constraint, use `sel * a + (1 - sel) * b`",
                            cv.kind()
                        ),
                    ));
                };
                if cc.is_zero() {
                    self.eval_expr(b, f)
                } else {
                    self.eval_expr(a, f)
                }
            }

            Expr::Call(name, args, pos) => {
                if self.program.function(name).is_some() {
                    let slot = self.call_function(name, args, f, *pos)?;
                    return Ok(slot.leaf(*pos, name)?.clone());
                }
                if self.program.template(name).is_some() {
                    return Err(err(
                        *pos,
                        format!("`{}` is a template, not a function; instantiate it with `component c = {}(...)`", name, name),
                    ));
                }
                Err(err(*pos, format!("unknown function `{}`", name)))
            }

            Expr::ArrayInline(_, pos) => {
                Err(err(*pos, "an array literal cannot be used as a scalar value"))
            }

            Expr::TemplateInst(name, _, pos) => Err(err(
                *pos,
                format!("`{}(...)` may only appear on the right of a `component` declaration", name),
            )),

            // circom refuses an anonymous component in an arithmetic position
            // too ("This is the anonymous component whose use is not allowed"),
            // so this is a match rather than a limitation. The two legal
            // positions - the whole right-hand side of a substitution, and an
            // argument of another anonymous component - are handled in
            // `substitute` and `anon_outputs` before ever reaching here.
            Expr::AnonComp { template, pos, .. } => Err(err(
                *pos,
                format!(
                    "an anonymous component `{}(...)(...)` may only be the entire right-hand \
                     side of a substitution, not part of a larger expression",
                    template
                ),
            )),

            Expr::Tuple(_, pos) => Err(err(
                *pos,
                "a tuple `(a, b)` may only appear on the left of a substitution",
            )),

            Expr::Binary(op, a, b, pos) => {
                let va = self.eval_expr(a, f)?;
                let vb = self.eval_expr(b, f)?;
                self.binary(op, va, vb, *pos)
            }
        }
    }

    fn binary(&mut self, op: &BinOp, a: Val, b: Val, pos: Pos) -> LResult<Val> {
        let neg_one = Fr::zero().sub(&Fr::one());
        // An unknown operand makes the result unknown, whatever the operator.
        // The reason travels with the value so that the eventual refusal names
        // where it first became unknown, not where it was finally used.
        if let Some(id) = a.opaque_id().or_else(|| b.opaque_id()) {
            return Ok(Val::Opaque(id));
        }
        match op {
            BinOp::Add => self.add_vals(a, b, pos),
            BinOp::Sub => {
                let nb = self.mul_vals(Val::Const(neg_one), b, pos)?;
                self.add_vals(a, nb, pos)
            }
            BinOp::Mul => self.mul_vals(a, b, pos),
            BinOp::Div => {
                // A signal divisor has no constraint form, but it does have a
                // witness value - which is exactly what `<--` is for, and what
                // `Bits2Point_Strict` writes. Carry it as unknown and let the
                // use site decide: `<--` accepts it, `<==` and `===` refuse it
                // with this message quoted back.
                let Some(d) = b.as_const() else {
                    return Ok(self.opaque(
                        pos,
                        "`/` by a signal, which has no R1CS form - the quotient is a witness \
                         value, so it can be computed with `<--` and constrained with `===`",
                    ));
                };
                if d.is_zero() {
                    return Err(err(pos, "division by zero"));
                }
                self.mul_vals(Val::Const(d.inv()), a, pos)
            }
            BinOp::Pow => {
                let Some(e) = b.as_const() else {
                    return Ok(self.opaque(pos, "`**` with an exponent that is not known at compile time"));
                };
                let n = e.to_u64().ok_or_else(|| err(pos, "exponent too large"))?;
                if let Some(base) = a.as_const() {
                    return Ok(Val::Const(base.pow_limbs(&[n, 0, 0, 0])));
                }
                match n {
                    0 => Ok(Val::Const(Fr::one())),
                    1 => Ok(a),
                    2 => self.mul_vals(a.clone(), a, pos),
                    // Same as the degree>2 product: witness-domain, refused at
                    // the point of use rather than here.
                    _ => Ok(self.opaque(
                        pos,
                        format!("a signal raised to the power {} exceeds degree 2", n),
                    )),
                }
            }
            // The remaining operators are integer / boolean and have no
            // constraint form. circom evaluates them at compile time too.
            _ => {
                let (ca, cb) = match (a.as_const(), b.as_const()) {
                    (Some(x), Some(y)) => (x, y),
                    // Over signals these are gadgets, not operators: there is no
                    // R1CS form, and building one is what `comparators.circom`
                    // and `bitify.circom` are. There IS a witness value though,
                    // and circom's own `var`s carry it - `sha256compression`
                    // shifts and xors 768 input signals into a digest and feeds
                    // the result to `<--`. So this is unknown rather than an
                    // error, and the error appears if it reaches a constraint.
                    _ => {
                        return Ok(self.opaque(
                            pos,
                            format!(
                                "`{:?}` over signals, which has no R1CS form - over constraints \
                                 it is a gadget (circomlib's `comparators.circom`, \
                                 `bitify.circom`), and over the witness it is an ordinary \
                                 integer operation",
                                op
                            ),
                        ))
                    }
                };
                let to_big = |v: Fr| v.to_biguint();
                let bool_val = |t: bool| Val::Const(if t { Fr::one() } else { Fr::zero() });
                Ok(match op {
                    BinOp::IntDiv => {
                        if cb.is_zero() {
                            return Err(err(pos, "integer division by zero"));
                        }
                        Val::Const(ca.int_div_rem(&cb).0)
                    }
                    BinOp::Mod => {
                        if cb.is_zero() {
                            return Err(err(pos, "modulo by zero"));
                        }
                        Val::Const(ca.int_div_rem(&cb).1)
                    }
                    BinOp::Eq => bool_val(ca == cb),
                    BinOp::Neq => bool_val(ca != cb),
                    // SIGNED, not canonical — see `signed_cmp`. circom reads a
                    // value above `(p-1)/2` as negative, and `Fr: Ord` does not.
                    BinOp::Lt => bool_val(signed_cmp(&ca, &cb).is_lt()),
                    BinOp::Gt => bool_val(signed_cmp(&ca, &cb).is_gt()),
                    BinOp::Le => bool_val(signed_cmp(&ca, &cb).is_le()),
                    BinOp::Ge => bool_val(signed_cmp(&ca, &cb).is_ge()),
                    BinOp::And => bool_val(!ca.is_zero() && !cb.is_zero()),
                    BinOp::Or => bool_val(!ca.is_zero() || !cb.is_zero()),
                    BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
                        let (x, y) = (ca.to_limbs(), cb.to_limbs());
                        let mut r = [0u64; 4];
                        for i in 0..4 {
                            r[i] = match op {
                                BinOp::BitAnd => x[i] & y[i],
                                BinOp::BitOr => x[i] | y[i],
                                _ => x[i] ^ y[i],
                            };
                        }
                        Val::Const(Fr::from_limbs_reduce(r))
                    }
                    BinOp::Shl | BinOp::Shr => {
                        let sh = cb.to_u64().ok_or_else(|| err(pos, "shift amount too large"))?;
                        let big = to_big(ca);
                        let mut acc = big;
                        let two = BigUint::from_u64(2);
                        if matches!(op, BinOp::Shl) {
                            for _ in 0..sh {
                                acc = acc.mul(&two);
                            }
                            Val::Const(Fr::from_biguint(acc))
                        } else {
                            for _ in 0..sh {
                                acc = acc.div_mod(&two).0;
                            }
                            Val::Const(Fr::from_biguint(acc))
                        }
                    }
                    _ => unreachable!(),
                })
            }
        }
    }

    fn add_vals(&mut self, a: Val, b: Val, pos: Pos) -> LResult<Val> {
        // Reachable without going through `binary` (`Sub` builds its negation
        // here, and `Expr::Unary(Neg)` calls `mul_vals` directly), so the
        // opaque guard has to be repeated rather than assumed.
        if let Some(id) = a.opaque_id().or_else(|| b.opaque_id()) {
            return Ok(Val::Opaque(id));
        }
        Ok(match (a, b) {
            (Val::Const(x), Val::Const(y)) => Val::Const(x.add(&y)),
            (Val::Quad(qa, qb, qc), other) | (other, Val::Quad(qa, qb, qc)) => {
                let mut c = qc;
                if !other.add_into(&mut c, Fr::one()) {
                    // WITNESS-domain, not an error. The value exceeds what one
                    // R1CS constraint can hold, so it cannot appear IN one -
                    // but circom `var`s routinely accumulate such sums as
                    // advice that only ever reaches a `<--`. `BigMult` is the
                    // canonical case: `prod_val[i+j] += a[i] * b[j]` builds a
                    // full polynomial product, feeds it to `out[i] <-- ...`,
                    // and binds `out` with its own `===` identity below.
                    //
                    // Refusing here refused the arithmetic; the refusal belongs
                    // at the point of USE, where `require_known` already names
                    // both the site and this reason. Same correction as
                    // `assert` over signals, one layer down.
                    return Ok(self.opaque(
                        pos,
                        "the sum of two quadratic expressions exceeds what one R1CS constraint \
                         can hold",
                    ));
                }
                c.simplify();
                Val::Quad(qa, qb, c)
            }
            (x, y) => {
                let mut l = x.into_lc().unwrap();
                y.add_into(&mut l, Fr::one());
                l.simplify();
                Val::Lin(l)
            }
        })
    }

    fn mul_vals(&mut self, a: Val, b: Val, pos: Pos) -> LResult<Val> {
        if let Some(id) = a.opaque_id().or_else(|| b.opaque_id()) {
            return Ok(Val::Opaque(id));
        }
        Ok(match (a, b) {
            (Val::Const(x), Val::Const(y)) => Val::Const(x.mul(&y)),
            (Val::Const(k), other) | (other, Val::Const(k)) => {
                if k.is_zero() {
                    return Ok(Val::Const(Fr::zero()));
                }
                match other {
                    // Both are excluded by the arms above: `Const` by the
                    // preceding pattern, `Opaque` by the guard at the top.
                    Val::Const(_) | Val::Opaque(_) => unreachable!(),
                    Val::Lin(mut l) => {
                        l.scale_assign(k);
                        Val::Lin(l)
                    }
                    Val::Quad(mut qa, qb, mut qc) => {
                        qa.scale_assign(k);
                        qc.scale_assign(k);
                        Val::Quad(qa, qb, qc)
                    }
                }
            }
            (Val::Lin(x), Val::Lin(y)) => Val::Quad(x, y, LinearCombination::zero()),
            // Degree > 2. Witness-domain for the same reason as the sum above:
            // it cannot appear in a constraint, and the check that refuses it
            // there is `require_known`, which names this reason.
            _ => self.opaque(
                pos,
                "this product has degree greater than 2, which no single R1CS constraint can \
                 express",
            ),
        })
    }
}

impl Frame {
    fn new(prefix: String) -> Self {
        Frame {
            vars: vec![HashMap::new()],
            signals: HashMap::new(),
            input_order: Vec::new(),
            output_order: Vec::new(),
            components: HashMap::new(),
            prefix,
            anon_count: 0,
        }
    }

    fn push_scope(&mut self) {
        self.vars.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.vars.pop();
    }

    fn declare_var(&mut self, name: String, slot: VarSlot, _pos: Pos) -> LResult<()> {
        self.vars.last_mut().unwrap().insert(name, slot);
        Ok(())
    }

    fn lookup_var(&self, name: &str) -> Option<&VarSlot> {
        self.vars.iter().rev().find_map(|s| s.get(name))
    }

    fn lookup_var_mut(&mut self, name: &str) -> Option<&mut VarSlot> {
        self.vars.iter_mut().rev().find_map(|s| s.get_mut(name))
    }
}

/// Compile a circom file to a populated `ZkEmitter`.
pub fn compile_file(
    path: &std::path::Path,
    search_paths: &[std::path::PathBuf],
) -> LResult<ZkEmitter> {
    let timing = std::env::var("Y_ZK_TIMING").is_ok();
    let t = std::time::Instant::now();
    let program = crate::circom_parser::Parser::parse_file(path, search_paths)?;
    if timing {
        eprintln!("[Y ZK TIMING]   circom parse         {:>8.3} s", t.elapsed().as_secs_f64());
    }
    let t = std::time::Instant::now();
    let lowerer = Lowerer::new(&program);
    let out = lowerer.lower();
    if timing {
        eprintln!("[Y ZK TIMING]   lower + optimize     {:>8.3} s", t.elapsed().as_secs_f64());
    }
    out
}
