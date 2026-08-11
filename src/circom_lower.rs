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
// Values
// ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum Val {
    Const(Fr),
    Lin(LinearCombination),
    /// `a * b + c`
    Quad(LinearCombination, LinearCombination, LinearCombination),
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
            Val::Quad(..) => None,
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
            Val::Quad(..) => false,
        }
    }

    fn lc(&self) -> Option<LinearCombination> {
        match self {
            Val::Const(c) => Some(LinearCombination::constant(*c)),
            Val::Lin(l) => Some(l.clone()),
            Val::Quad(..) => None,
        }
    }

    fn as_const(&self) -> Option<Fr> {
        match self {
            Val::Const(c) => Some(*c),
            _ => None,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Val::Const(_) => "constant",
            Val::Lin(_) => "linear",
            Val::Quad(..) => "quadratic",
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

        self.emitter.run_optimizer();
        Ok(self.emitter)
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
                self.constrain_equal(va, vb, *pos)?;
                Ok(Flow::Normal)
            }

            Stmt::If { cond, then_branch, else_branch, pos } => {
                let c = self.eval_expr(cond, f)?;
                let c = c.as_const().ok_or_else(|| {
                    err(
                        *pos,
                        format!(
                            "an `if` condition must be known at compile time, but this one is {}. \
                             The branch decides which constraints exist, so it cannot depend on a \
                             signal's value; use a multiplexer (`sel * a + (1 - sel) * b`) instead.",
                            c.kind()
                        ),
                    )
                })?;
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
                    let c = self.eval_expr(cond, f)?;
                    let c = c.as_const().ok_or_else(|| {
                        err(*pos, format!("a `for` condition must be known at compile time, but this one is {}", c.kind()))
                    })?;
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
                    let c = self.eval_expr(cond, f)?;
                    let c = c.as_const().ok_or_else(|| {
                        err(*pos, format!("a `while` condition must be known at compile time, but this one is {}", c.kind()))
                    })?;
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
                // A compile-time assert is checked now. One over signals is a
                // real constraint, and dropping it would weaken the circuit.
                let v = self.eval_expr(e, f)?;
                match v.as_const() {
                    Some(c) if !c.is_zero() => Ok(Flow::Normal),
                    Some(_) => Err(err(*pos, "assertion failed at compile time")),
                    None => Err(err(
                        *pos,
                        "`assert` over signal values is not supported by Y's circom front end. \
                         It is a constraint on the witness, so ignoring it would emit a circuit \
                         weaker than the source; write it as `===` to make it explicit.",
                    )),
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
                let wire = self.resolve_signal_wire(lhs, f, pos)?;
                if let AssignOp::SignalOnly = op {
                    // `<--` must NOT be evaluated through the constraint value
                    // model. Doing so is what made circomlib's own
                    // `bitify.circom` uncompilable: `Num2Bits` writes
                    // `out[i] <-- (in >> i) & 1`, and a shift over a signal was
                    // refused for having "no R1CS form" - which is true, and
                    // irrelevant, because a `<--` right-hand side never becomes
                    // a constraint. It is a witness computation, checked
                    // afterwards by the `===` the author writes.
                    let recipe = self.witness_only_recipe(rhs, f, pos)?;
                    self.emitter.set_witness_recipe(wire, recipe);
                    return Ok(());
                }
                let v = self.eval_expr(rhs, f)?;
                let target = LinearCombination::variable(wire);

                match op {
                    AssignOp::SignalConstrain => {
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
                        }
                    }
                    // `<--` returned above: it emits NO constraint, and turning
                    // it into `<==` would silently add one the author chose not
                    // to write.
                    AssignOp::SignalOnly | AssignOp::Var => unreachable!(),
                }
                Ok(())
            }
        }
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
                err(d.pos(), "array dimensions must be known at compile time")
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
                let c = v
                    .as_const()
                    .ok_or_else(|| err(*pos, "array indices must be known at compile time; a signal-dependent index needs an explicit multiplexer"))?;
                let n = c.to_u64().ok_or_else(|| err(*pos, "array index does not fit in 64 bits"))?;
                idxs.push(n as usize);
                Ok((b, idxs))
            }
            other => Ok((other.clone(), Vec::new())),
        }
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

    /// Resolve an lvalue to the wire of a signal, following `c.out[i]` into
    /// subcomponents.
    fn resolve_signal_wire(&mut self, e: &Expr, f: &mut Frame, pos: Pos) -> LResult<usize> {
        let (base, idxs) = self.split_indices(e, f)?;
        match &base {
            Expr::Var(name, p) => {
                let slot = f
                    .signals
                    .get(name)
                    .ok_or_else(|| err(*p, format!("`{}` is not a signal of this template", name)))?;
                Ok(*Self::index_slot(slot, &idxs, pos)?.leaf(pos, name)?)
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
                Ok(*Self::index_slot(&slot, &idxs, pos)?.leaf(pos, field)?)
            }
            other => Err(err(other.pos(), "left-hand side of a signal assignment must be a signal")),
        }
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
            Expr::Index(..) => {
                // Could be a whole sub-array of a var.
                let (base, idxs) = self.split_indices(e, f)?;
                if let Expr::Var(name, _) = &base {
                    if let Some(slot) = f.lookup_var(name) {
                        let sub = Self::index_slot(slot, &idxs, e.pos())?;
                        return Ok(sub.clone());
                    }
                }
                Ok(Slot::Leaf(self.eval_expr(e, f)?))
            }
            _ => Ok(Slot::Leaf(self.eval_expr(e, f)?)),
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
                        let c = v.as_const().ok_or_else(|| {
                            err(*pos, "`!` needs a compile-time value; it is not expressible as a constraint")
                        })?;
                        Ok(Val::Const(if c.is_zero() { Fr::one() } else { Fr::zero() }))
                    }
                    UnOp::BitNot => {
                        let c = v.as_const().ok_or_else(|| {
                            err(*pos, "`~` needs a compile-time value")
                        })?;
                        // circom's ~ is over 254-bit two's complement in Fr.
                        let l = c.to_limbs();
                        let inv = [!l[0], !l[1], !l[2], !l[3]];
                        Ok(Val::Const(Fr::from_limbs_reduce(inv)))
                    }
                }
            }

            Expr::Ternary(c, a, b, pos) => {
                let cv = self.eval_expr(c, f)?;
                let cc = cv.as_const().ok_or_else(|| {
                    err(*pos, format!("a `? :` condition must be known at compile time, but this one is {}. Use `sel * a + (1 - sel) * b`.", cv.kind()))
                })?;
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

            Expr::Binary(op, a, b, pos) => {
                let va = self.eval_expr(a, f)?;
                let vb = self.eval_expr(b, f)?;
                self.binary(op, va, vb, *pos)
            }
        }
    }

    fn binary(&mut self, op: &BinOp, a: Val, b: Val, pos: Pos) -> LResult<Val> {
        let neg_one = Fr::zero().sub(&Fr::one());
        match op {
            BinOp::Add => self.add_vals(a, b, pos),
            BinOp::Sub => {
                let nb = self.mul_vals(Val::Const(neg_one), b, pos)?;
                self.add_vals(a, nb, pos)
            }
            BinOp::Mul => self.mul_vals(a, b, pos),
            BinOp::Div => {
                let d = b.as_const().ok_or_else(|| {
                    err(pos, "non-quadratic: `/` by a signal cannot be expressed as an R1CS constraint. \
                              Compute the inverse with `<--` and constrain it with `===`.")
                })?;
                if d.is_zero() {
                    return Err(err(pos, "division by zero"));
                }
                self.mul_vals(Val::Const(d.inv()), a, pos)
            }
            BinOp::Pow => {
                let e = b
                    .as_const()
                    .ok_or_else(|| err(pos, "the exponent of `**` must be known at compile time"))?;
                let n = e.to_u64().ok_or_else(|| err(pos, "exponent too large"))?;
                if let Some(base) = a.as_const() {
                    return Ok(Val::Const(base.pow_limbs(&[n, 0, 0, 0])));
                }
                match n {
                    0 => Ok(Val::Const(Fr::one())),
                    1 => Ok(a),
                    2 => self.mul_vals(a.clone(), a, pos),
                    _ => Err(err(
                        pos,
                        format!("non-quadratic: a signal raised to the power {} exceeds degree 2; introduce intermediate signals", n),
                    )),
                }
            }
            // The remaining operators are integer / boolean and have no
            // constraint form. circom evaluates them at compile time too.
            _ => {
                let (ca, cb) = match (a.as_const(), b.as_const()) {
                    (Some(x), Some(y)) => (x, y),
                    _ => {
                        return Err(err(
                            pos,
                            format!(
                                "non-quadratic: `{:?}` needs compile-time operands. It has no R1CS \
                                 form over signals - build it from constraints (circomlib's \
                                 `comparators.circom`, `bitify.circom`) instead.",
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
                    BinOp::Lt => bool_val(ca < cb),
                    BinOp::Gt => bool_val(ca > cb),
                    BinOp::Le => bool_val(ca <= cb),
                    BinOp::Ge => bool_val(ca >= cb),
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
        Ok(match (a, b) {
            (Val::Const(x), Val::Const(y)) => Val::Const(x.add(&y)),
            (Val::Quad(qa, qb, qc), other) | (other, Val::Quad(qa, qb, qc)) => {
                let mut c = qc;
                if !other.add_into(&mut c, Fr::one()) {
                    return Err(err(
                        pos,
                        "non-quadratic: the sum of two quadratic expressions exceeds what one R1CS \
                         constraint can hold. Assign one to an intermediate signal first.",
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
        Ok(match (a, b) {
            (Val::Const(x), Val::Const(y)) => Val::Const(x.mul(&y)),
            (Val::Const(k), other) | (other, Val::Const(k)) => {
                if k.is_zero() {
                    return Ok(Val::Const(Fr::zero()));
                }
                match other {
                    Val::Const(_) => unreachable!(),
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
            _ => {
                return Err(err(
                    pos,
                    "non-quadratic: this product has degree greater than 2, which no single R1CS \
                     constraint can express. Assign the inner product to a signal first \
                     (`signal t; t <== a * b; ... t * c ...`).",
                ))
            }
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
