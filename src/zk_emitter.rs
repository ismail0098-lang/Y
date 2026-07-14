// ============================================================
//  Y — ZK Circuit Backend Emitter
//  zk_emitter.rs
//
//  Translates Y AST / IR into Rank-1 Constraint Systems (R1CS)
//  of the form (A · x) * (B · x) = C · x over the BN254 Fr field.
// ============================================================

#![allow(dead_code)]

use crate::ast::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

// ────────────────────────────────────────────────────────
// 1. BigUint and BN254 Fr Field Arithmetic
// ────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct BigUint {
    // Digits in little-endian order, base 2^32
    pub digits: Vec<u32>,
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigUint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let mut lhs = self.clone();
        lhs.trim();
        let mut rhs = other.clone();
        rhs.trim();
        if lhs.digits.len() != rhs.digits.len() {
            return lhs.digits.len().cmp(&rhs.digits.len());
        }
        for i in (0..lhs.digits.len()).rev() {
            if lhs.digits[i] != rhs.digits[i] {
                return lhs.digits[i].cmp(&rhs.digits[i]);
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl BigUint {
    pub fn zero() -> Self {
        Self { digits: vec![0] }
    }

    pub fn one() -> Self {
        Self { digits: vec![1] }
    }

    pub fn is_zero(&self) -> bool {
        self.digits.iter().all(|&d| d == 0)
    }

    pub fn from_u64(mut val: u64) -> Self {
        let mut digits = Vec::new();
        if val == 0 {
            digits.push(0);
        } else {
            while val > 0 {
                digits.push((val & 0xffffffff) as u32);
                val >>= 32;
            }
        }
        Self { digits }
    }

    pub fn trim(&mut self) {
        while self.digits.len() > 1 && *self.digits.last().unwrap() == 0 {
            self.digits.pop();
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut digits = Vec::new();
        let mut carry = 0u64;
        let len = std::cmp::max(self.digits.len(), other.digits.len());
        for i in 0..len {
            let d1 = self.digits.get(i).cloned().unwrap_or(0) as u64;
            let d2 = other.digits.get(i).cloned().unwrap_or(0) as u64;
            let sum = d1 + d2 + carry;
            digits.push((sum & 0xffffffff) as u32);
            carry = sum >> 32;
        }
        if carry > 0 {
            digits.push(carry as u32);
        }
        let mut res = Self { digits };
        res.trim();
        res
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut digits = Vec::new();
        let mut borrow = 0i64;
        let len = std::cmp::max(self.digits.len(), other.digits.len());
        for i in 0..len {
            let d1 = self.digits.get(i).cloned().unwrap_or(0) as i64;
            let d2 = other.digits.get(i).cloned().unwrap_or(0) as i64;
            let diff = d1 - d2 - borrow;
            if diff < 0 {
                digits.push((diff + 0x100000000) as u32);
                borrow = 1;
            } else {
                digits.push(diff as u32);
                borrow = 0;
            }
        }
        if borrow > 0 {
            panic!("BigUint subtraction underflow");
        }
        let mut res = Self { digits };
        res.trim();
        res
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut digits = vec![0u32; self.digits.len() + other.digits.len()];
        for i in 0..self.digits.len() {
            let mut carry = 0u64;
            for j in 0..other.digits.len() {
                let prod = (self.digits[i] as u64) * (other.digits[j] as u64) + (digits[i + j] as u64) + carry;
                digits[i + j] = (prod & 0xffffffff) as u32;
                carry = prod >> 32;
            }
            if carry > 0 {
                digits[i + other.digits.len()] += carry as u32;
            }
        }
        let mut res = Self { digits };
        res.trim();
        res
    }

    pub fn bit_len(&self) -> usize {
        if self.is_zero() {
            return 0;
        }
        let last_idx = self.digits.len() - 1;
        let last_digit = self.digits[last_idx];
        let bits_in_last = 32 - last_digit.leading_zeros() as usize;
        last_idx * 32 + bits_in_last
    }

    pub fn get_bit(&self, bit_idx: usize) -> bool {
        let digit_idx = bit_idx / 32;
        let shift = bit_idx % 32;
        if digit_idx >= self.digits.len() {
            false
        } else {
            ((self.digits[digit_idx] >> shift) & 1) == 1
        }
    }

    pub fn set_bit(&mut self, bit_idx: usize, val: bool) {
        let digit_idx = bit_idx / 32;
        let shift = bit_idx % 32;
        while self.digits.len() <= digit_idx {
            self.digits.push(0);
        }
        if val {
            self.digits[digit_idx] |= 1 << shift;
        } else {
            self.digits[digit_idx] &= !(1 << shift);
        }
        self.trim();
    }

    pub fn shl1(&self) -> Self {
        let mut digits = Vec::new();
        let mut carry = 0u32;
        for &d in &self.digits {
            digits.push((d << 1) | carry);
            carry = d >> 31;
        }
        if carry > 0 {
            digits.push(carry);
        }
        let mut res = Self { digits };
        res.trim();
        res
    }

    pub fn div_mod(&self, other: &Self) -> (Self, Self) {
        if other.is_zero() {
            panic!("Division by zero");
        }
        let mut quotient = Self::zero();
        let mut remainder = Self::zero();
        for i in (0..self.bit_len()).rev() {
            remainder = remainder.shl1();
            if self.get_bit(i) {
                remainder.set_bit(0, true);
            }
            if remainder >= *other {
                remainder = remainder.sub(other);
                quotient.set_bit(i, true);
            }
        }
        (quotient, remainder)
    }

    pub fn from_str(s: &str) -> Self {
        let mut res = Self::zero();
        let ten = Self::from_u64(10);
        for c in s.chars() {
            if let Some(digit) = c.to_digit(10) {
                res = res.mul(&ten).add(&Self::from_u64(digit as u64));
            }
        }
        res
    }

    pub fn to_decimal_string(&self) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        let mut temp = self.clone();
        let ten = Self::from_u64(10);
        let mut chars = Vec::new();
        while !temp.is_zero() {
            let (q, r) = temp.div_mod(&ten);
            let digit = r.digits[0];
            chars.push(std::char::from_digit(digit, 10).unwrap());
            temp = q;
        }
        chars.into_iter().rev().collect()
    }

    pub fn to_bytes_le(&self, byte_len: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(byte_len);
        for &digit in &self.digits {
            bytes.extend_from_slice(&digit.to_le_bytes());
        }
        bytes.resize(byte_len, 0);
        bytes
    }
}


#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Fr(pub BigUint);

impl Fr {
    pub fn modulus() -> BigUint {
        BigUint::from_str("21888242871839275222246405745257275088548364400416034343698204186575808495617")
    }

    pub fn zero() -> Self {
        Fr(BigUint::zero())
    }

    pub fn one() -> Self {
        Fr(BigUint::one())
    }

    pub fn from_u64(val: u64) -> Self {
        let bi = BigUint::from_u64(val);
        let (_, r) = bi.div_mod(&Self::modulus());
        Fr(r)
    }

    pub fn from_biguint(bi: BigUint) -> Self {
        let (_, r) = bi.div_mod(&Self::modulus());
        Fr(r)
    }

    pub fn add(&self, other: &Self) -> Self {
        let sum = self.0.add(&other.0);
        if sum >= Self::modulus() {
            Fr(sum.sub(&Self::modulus()))
        } else {
            Fr(sum)
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        if self.0 >= other.0 {
            Fr(self.0.sub(&other.0))
        } else {
            Fr(self.0.add(&Self::modulus()).sub(&other.0))
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let prod = self.0.mul(&other.0);
        let (_, r) = prod.div_mod(&Self::modulus());
        Fr(r)
    }

    pub fn inv(&self) -> Self {
        if self.0.is_zero() {
            panic!("Zero has no modular inverse");
        }
        let p = Self::modulus();
        let mut t = BigUint::zero();
        let mut newt = BigUint::one();
        let mut r = p.clone();
        let mut newr = self.0.clone();
        
        let mut t_neg = false;
        let mut newt_neg = false;

        while !newr.is_zero() {
            let (quotient, remainder) = r.div_mod(&newr);
            r = newr;
            newr = remainder;

            let prod = quotient.mul(&newt);
            let next_t;
            let next_t_neg;

            if t_neg == newt_neg {
                if t >= prod {
                    next_t = t.sub(&prod);
                    next_t_neg = t_neg;
                } else {
                    next_t = prod.sub(&t);
                    next_t_neg = !t_neg;
                }
            } else {
                next_t = t.add(&prod);
                next_t_neg = t_neg;
            }

            t = newt;
            t_neg = newt_neg;
            newt = next_t;
            newt_neg = next_t_neg;
        }

        if r > BigUint::one() {
            panic!("Modular inverse does not exist");
        }

        if t_neg {
            Fr(p.sub(&t))
        } else {
            Fr(t)
        }
    }

    pub fn to_string(&self) -> String {
        self.0.to_decimal_string()
    }
}

// ────────────────────────────────────────────────────────
// 2. R1CS Structural Definitions & Linear Combinations
// ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct LinearCombination {
    // List of (wire_id, coefficient). By convention, wire_id = 0 is constant 1
    pub terms: Vec<(usize, Fr)>,
    pub is_simplified: bool,
}

impl LinearCombination {
    pub fn zero() -> Self {
        Self { terms: Vec::new(), is_simplified: true }
    }

    pub fn constant(val: Fr) -> Self {
        if val.0.is_zero() {
            Self::zero()
        } else {
            Self { terms: vec![(0, val)], is_simplified: true }
        }
    }

    pub fn variable(id: usize) -> Self {
        Self { terms: vec![(id, Fr::one())], is_simplified: true }
    }

    pub fn add_constant(&mut self, val: Fr) {
        if !val.0.is_zero() {
            self.terms.push((0, val));
            self.is_simplified = false;
        }
    }

    pub fn add_term(&mut self, wire_id: usize, val: Fr) {
        if !val.0.is_zero() {
            self.terms.push((wire_id, val));
            self.is_simplified = false;
        }
    }

    pub fn add_linear(&mut self, other: &Self, scale: Fr) {
        if scale.0.is_zero() {
            return;
        }
        let can_keep_simplified = self.is_simplified 
            && other.is_simplified 
            && (self.terms.is_empty() || other.terms.is_empty() || self.terms.last().unwrap().0 < other.terms[0].0);
        
        for (wire, coeff) in &other.terms {
            self.terms.push((*wire, coeff.mul(&scale)));
        }
        self.is_simplified = can_keep_simplified;
    }

    pub fn scale(&self, factor: Fr) -> Self {
        if factor.0.is_zero() {
            return Self::zero();
        }
        let terms = self.terms.iter().map(|(w, c)| (*w, c.mul(&factor))).collect();
        Self { terms, is_simplified: self.is_simplified }
    }

    pub fn simplify(&mut self) {
        if self.is_simplified {
            return;
        }
        if self.terms.len() <= 1 {
            if self.terms.len() == 1 && self.terms[0].1.0.is_zero() {
                self.terms.clear();
            }
            self.is_simplified = true;
            return;
        }

        // Check if terms are already sorted and have no duplicate wire IDs
        let mut already_simple = true;
        for i in 0..self.terms.len() - 1 {
            if self.terms[i].0 >= self.terms[i+1].0 {
                already_simple = false;
                break;
            }
        }
        if already_simple {
            let has_zeros = self.terms.iter().any(|(_, coeff)| coeff.0.is_zero());
            if !has_zeros {
                self.is_simplified = true;
                return; // Already sorted, distinct, and non-zero
            }
        }

        let mut merged: HashMap<usize, Fr> = HashMap::new();
        for (wire, coeff) in &self.terms {
            let entry = merged.entry(*wire).or_insert_with(Fr::zero);
            *entry = entry.add(coeff);
        }
        self.terms = merged.into_iter()
            .filter(|(_, coeff)| !coeff.0.is_zero())
            .collect();
        self.terms.sort_by_key(|t| t.0);
        self.is_simplified = true;
    }

    pub fn is_constant(&self) -> Option<Fr> {
        if self.terms.is_empty() {
            return Some(Fr::zero());
        }
        
        // A linear combination is not a constant if it contains any variable wire (id > 0)
        let mut has_variables = false;
        for (wire, coeff) in &self.terms {
            if *wire != 0 && !coeff.0.is_zero() {
                has_variables = true;
                break;
            }
        }
        if has_variables {
            return None;
        }

        // It only contains constant terms (wire 0), sum them up
        let mut sum = Fr::zero();
        for (wire, coeff) in &self.terms {
            if *wire == 0 {
                sum = sum.add(coeff);
            }
        }
        Some(sum)
    }

    pub fn to_string(&self, var_names: &HashMap<usize, String>) -> String {
        let mut simplified = self.clone();
        simplified.simplify();
        if simplified.terms.is_empty() {
            return "0".to_string();
        }
        let mut s = String::new();
        for (i, (wire, coeff)) in simplified.terms.iter().enumerate() {
            if i > 0 {
                s.push_str(" + ");
            }
            let name = if *wire == 0 {
                "1".to_string()
            } else {
                var_names.get(wire).cloned().unwrap_or_else(|| format!("w_{}", wire))
            };
            if coeff.0 == BigUint::one() {
                s.push_str(&name);
            } else {
                s.push_str(&format!("{} * {}", coeff.to_string(), name));
            }
        }
        s
    }
}

impl PartialEq for LinearCombination {
    fn eq(&self, other: &Self) -> bool {
        self.terms == other.terms
    }
}

impl Eq for LinearCombination {}

impl std::hash::Hash for LinearCombination {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.terms.hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct Constraint {
    pub a: LinearCombination,
    pub b: LinearCombination,
    pub c: LinearCombination,
    pub span: Option<Span>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WireBinding {
    Wire(usize),
    Linear(LinearCombination),
}

pub struct Circuit {
    pub num_variables: usize,
    pub variables: Vec<String>,
    pub public_inputs: Vec<usize>,
    pub private_inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub constraints: Vec<Constraint>,
}

// ────────────────────────────────────────────────────────
// 3. Lowering and Optimization Pass
// ────────────────────────────────────────────────────────

pub struct ZkEmitter {
    pub variables: Vec<String>,
    pub public_inputs: Vec<usize>,
    pub private_inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub constraints: Vec<Constraint>,
    pub next_var_id: usize,

    // Scope management: Maps variables to their bound representation
    scopes: Vec<HashMap<String, WireBinding>>,
    // Constant bindings for static loop evaluation
    const_bindings: HashMap<String, Fr>,
    // Tracker for active calls to reject recursive loops
    active_calls: Vec<String>,
}

fn expr_references_var(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Ident(n, _) => n == name,
        Expr::BinaryOp { left, right, .. } => {
            expr_references_var(left, name) || expr_references_var(right, name)
        }
        Expr::Call { args, .. } => {
            args.iter().any(|arg| expr_references_var(arg, name))
        }
        _ => false,
    }
}

impl ZkEmitter {
    pub fn new() -> Self {
        Self {
            variables: vec!["const_1".to_string()], // wire 0 is constant 1
            public_inputs: Vec::new(),
            private_inputs: Vec::new(),
            outputs: Vec::new(),
            constraints: Vec::new(),
            next_var_id: 1,
            scopes: vec![HashMap::new()],
            const_bindings: HashMap::new(),
            active_calls: Vec::new(),
        }
    }

    fn new_wire(&mut self, name: &str) -> usize {
        let id = self.next_var_id;
        self.next_var_id += 1;
        self.variables.push(format!("{}_{}", name, id));
        id
    }

    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn bind(&mut self, name: &str, binding: WireBinding) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), binding);
        }
    }

    fn lookup(&self, name: &str) -> Option<WireBinding> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.get(name) {
                return Some(binding.clone());
            }
        }
        None
    }

    fn bind_update(&mut self, name: &str, binding: WireBinding) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), binding);
                return true;
            }
        }
        false
    }

    fn take_binding_from_scope(&mut self, name: &str) -> Option<(usize, WireBinding)> {
        for (idx, scope) in self.scopes.iter_mut().enumerate().rev() {
            if let Some(binding) = scope.remove(name) {
                return Some((idx, binding));
            }
        }
        None
    }

    fn bind_to_scope(&mut self, scope_idx: usize, name: &str, binding: WireBinding) {
        if scope_idx < self.scopes.len() {
            self.scopes[scope_idx].insert(name.to_string(), binding);
        }
    }

    fn lookup_in_scopes(&self, scopes_stack: &[HashMap<String, WireBinding>], start_idx: usize, name: &str) -> Option<LinearCombination> {
        for scope in scopes_stack[..=start_idx].iter().rev() {
            if let Some(binding) = scope.get(name) {
                return Some(match binding {
                    WireBinding::Wire(w) => LinearCombination::variable(*w),
                    WireBinding::Linear(lc) => lc.clone(),
                });
            }
        }
        None
    }

    fn lookup_const(&self, name: &str) -> Option<Fr> {
        self.const_bindings.get(name).cloned()
    }

    // ────────────────────────────────────────────────────────
    // Compilation Entry Points
    // ────────────────────────────────────────────────────────

    pub fn emit_program(&mut self, prog: &Program) -> Result<String, String> {
        // Collect all top-level functions and structs
        let mut target_func: Option<&FuncDecl> = None;
        for item in &prog.items {
            match item {
                Item::Func(f) => {
                    // Compile the function main or any function as the circuit entry
                    if f.name == "main" || f.name == "circuit" {
                        target_func = Some(f);
                    }
                }
                _ => {}
            }
        }

        let f = match target_func {
            Some(func) => func,
            None => return Err("No entry function 'main' or 'circuit' found for ZK Circuit target.".to_string()),
        };

        self.emit_circuit_entry(f, &prog.items)?;

        // Run optimization pass: dead-wire elimination & constraint reduction
        self.optimize_circuit();

        // Format R1CS Output
        let mut out = String::new();
        writeln!(&mut out, "=========================================================").unwrap();
        writeln!(&mut out, "   Y-lang Native ZK Circuit Target: Rank-1 Constraint System").unwrap();
        writeln!(&mut out, "=========================================================\n").unwrap();
        writeln!(&mut out, "Curve Field: BN254 Fr (prime size: 254 bits)").unwrap();
        writeln!(&mut out, "Modulus r: 21888242871839275222246405745257275088548364400416034343698204186575808495617\n").unwrap();

        writeln!(&mut out, "Parameters:").unwrap();
        writeln!(&mut out, "  - Total wires (including intermediate): {}", self.next_var_id).unwrap();
        writeln!(&mut out, "  - Constraints: {}", self.constraints.len()).unwrap();
        writeln!(&mut out, "  - Public inputs: {:?}", self.public_inputs).unwrap();
        writeln!(&mut out, "  - Private inputs: {:?}", self.private_inputs).unwrap();
        writeln!(&mut out, "  - Outputs: {:?}\n", self.outputs).unwrap();

        if self.variables.len() <= 1000 {
            writeln!(&mut out, "Wire Assignments:").unwrap();
            for (i, name) in self.variables.iter().enumerate() {
                let role = if self.public_inputs.contains(&i) {
                    " [Public Input]"
                } else if self.private_inputs.contains(&i) {
                    " [Private Witness Input]"
                } else if self.outputs.contains(&i) {
                    " [Output]"
                } else if i == 0 {
                    " [Constant 1]"
                } else {
                    ""
                };
                writeln!(&mut out, "  w_{} = {}{}", i, name, role).unwrap();
            }
        } else {
            writeln!(&mut out, "Wire Assignments:\n  [Detailed wire assignments list omitted for circuits with > 1000 variables to optimize compilation performance]").unwrap();
        }

        writeln!(&mut out, "\nR1CS Equations (A * B = C):").unwrap();
        if self.constraints.len() <= 1000 {
            let var_map: HashMap<usize, String> = self.variables.iter().enumerate()
                .map(|(i, _name)| (i, format!("w_{}", i)))
                .collect();

            for (idx, c) in self.constraints.iter().enumerate() {
                writeln!(
                    &mut out,
                    "  Constraint #{}:\n    A: ({})\n    B: ({})\n    C: ({})",
                    idx + 1,
                    c.a.to_string(&var_map),
                    c.b.to_string(&var_map),
                    c.c.to_string(&var_map)
                ).unwrap();
                if let Some(ref span) = c.span {
                    writeln!(&mut out, "    Source Location: line {}, col {}", span.line, span.col).unwrap();
                }
                writeln!(&mut out, "").unwrap();
            }
        } else {
            writeln!(&mut out, "  [Detailed constraints list omitted for circuits with > 1000 constraints to optimize compilation performance]").unwrap();
        }

        Ok(out)
    }

    fn emit_circuit_entry(&mut self, f: &FuncDecl, items: &[Item]) -> Result<(), String> {
        self.active_calls.push(f.name.clone());

        // Setup parameters as inputs
        for param in &f.params {
            let param_wire = self.new_wire(&param.name);
            // By convention, circuit parameters are treated as Private Inputs unless marked specifically.
            // Let's make them Private Inputs, and any returned value as Output.
            self.private_inputs.push(param_wire);
            self.bind(&param.name, WireBinding::Wire(param_wire));
        }

        // Lower body
        let ret_lc = self.emit_block(&f.body, items)?;

        // If there is a return expression, register it as Output wire
        if let Some(lc) = ret_lc {
            let out_wire = self.new_wire("out_ret");
            self.outputs.push(out_wire);
            // Constrain out_wire = lc
            // R1CS: (lc) * (1) = out_wire
            self.constraints.push(Constraint {
                a: lc,
                b: LinearCombination::constant(Fr::one()),
                c: LinearCombination::variable(out_wire),
                span: Some(f.span.clone()),
            });
        }

        self.active_calls.pop();
        Ok(())
    }

    fn emit_block(&mut self, block: &Block, items: &[Item]) -> Result<Option<LinearCombination>, String> {
        self.enter_scope();
        let mut last_ret = None;
        for stmt in &block.stmts {
            if let Some(ret) = self.emit_stmt(stmt, items)? {
                last_ret = Some(ret);
            }
        }
        self.exit_scope();
        Ok(last_ret)
    }

    fn emit_stmt(&mut self, stmt: &Stmt, items: &[Item]) -> Result<Option<LinearCombination>, String> {
        match stmt {
            Stmt::Let { name, init, .. } => {
                if let Some(expr) = init {
                    let mut lc = self.emit_expr(expr, items)?;
                    lc.simplify();
                    if let Some(c) = lc.is_constant() {
                        self.const_bindings.insert(name.clone(), c);
                    }
                    self.bind(name, WireBinding::Linear(lc));
                } else {
                    // Uninitialized variable: allocate a raw witness wire
                    let wire = self.new_wire(name);
                    self.bind(name, WireBinding::Wire(wire));
                }
            }
            Stmt::Assign { target, value, span } => {
                let target_name = match target {
                    Expr::Ident(name, _) => name.clone(),
                    _ => return Err(format!("Circuit target error: Assignments to non-identifiers are not supported. Line {}", span.line)),
                };
                
                let mut optimized = false;
                if let Expr::BinaryOp { left, op, right, .. } = value {
                    // Case 1: target = target + expr
                    if let Expr::Ident(ref left_name, _) = **left {
                        if left_name == &target_name && !expr_references_var(right, &target_name) {
                            if let Some((scope_idx, binding)) = self.take_binding_from_scope(&target_name) {
                                let mut target_lc = match binding {
                                    WireBinding::Wire(w) => LinearCombination::variable(w),
                                    WireBinding::Linear(lc) => lc,
                                };
                                let right_lc = self.emit_expr(right, items)?;
                                
                                match op {
                                    BinaryOp::Add => {
                                        target_lc.add_linear(&right_lc, Fr::one());
                                        target_lc.simplify();
                                        if let Some(c) = target_lc.is_constant() {
                                            self.const_bindings.insert(target_name.clone(), c);
                                        } else {
                                            self.const_bindings.remove(&target_name);
                                        }
                                        self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                        optimized = true;
                                    }
                                    BinaryOp::Sub => {
                                        let neg_one = Fr::from_u64(0).sub(&Fr::one());
                                        target_lc.add_linear(&right_lc, neg_one);
                                        target_lc.simplify();
                                        if let Some(c) = target_lc.is_constant() {
                                            self.const_bindings.insert(target_name.clone(), c);
                                        } else {
                                            self.const_bindings.remove(&target_name);
                                        }
                                        self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                        optimized = true;
                                    }
                                    _ => {
                                        self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                    }
                                }
                            }
                        }
                    }
                    
                    // Case 2: target = expr + target
                    if !optimized {
                        if let Expr::Ident(ref right_name, _) = **right {
                            if right_name == &target_name && !expr_references_var(left, &target_name) {
                                if let Some((scope_idx, binding)) = self.take_binding_from_scope(&target_name) {
                                    let mut target_lc = match binding {
                                        WireBinding::Wire(w) => LinearCombination::variable(w),
                                        WireBinding::Linear(lc) => lc,
                                    };
                                    let left_lc = self.emit_expr(left, items)?;
                                    
                                    match op {
                                        BinaryOp::Add => {
                                            target_lc.add_linear(&left_lc, Fr::one());
                                            target_lc.simplify();
                                            if let Some(c) = target_lc.is_constant() {
                                                self.const_bindings.insert(target_name.clone(), c);
                                            } else {
                                                self.const_bindings.remove(&target_name);
                                            }
                                            self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                            optimized = true;
                                        }
                                        BinaryOp::Sub => {
                                            target_lc = target_lc.scale(Fr::from_u64(0).sub(&Fr::one()));
                                            target_lc.add_linear(&left_lc, Fr::one());
                                            target_lc.simplify();
                                            if let Some(c) = target_lc.is_constant() {
                                                self.const_bindings.insert(target_name.clone(), c);
                                            } else {
                                                self.const_bindings.remove(&target_name);
                                            }
                                            self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                            optimized = true;
                                        }
                                        _ => {
                                            self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if !optimized {
                    // Validate circuit restrictions: mutable reassignment must create a multiplexed node or single-assign binding
                    let mut lc_val = self.emit_expr(value, items)?;
                    lc_val.simplify();
                    
                    // Track constant bindings
                    if let Some(c) = lc_val.is_constant() {
                        self.const_bindings.insert(target_name.clone(), c);
                    } else {
                        self.const_bindings.remove(&target_name);
                    }

                    // In ZK, re-assigning is done by binding the target to the new linear combination (SSA-style name rebinding)
                    if !self.bind_update(&target_name, WireBinding::Linear(lc_val.clone())) {
                        self.bind(&target_name, WireBinding::Linear(lc_val));
                    }
                }
            }
            Stmt::For {
                loop_var,
                start,
                end,
                step,
                body,
                span,
                ..
            } => {
                // Loop unrolling validation: loop bounds MUST be compile-time constants
                let start_lc = self.emit_expr(start, items)?;
                let end_lc = self.emit_expr(end, items)?;
                
                let start_const = start_lc.is_constant()
                    .ok_or_else(|| format!("Circuit limitation: Loop start bound must be a compile-time constant. Line {}", span.line))?;
                let end_const = end_lc.is_constant()
                    .ok_or_else(|| format!("Circuit limitation: Loop end bound must be a compile-time constant. Line {}", span.line))?;
                
                let step_val = if let Some(step_expr) = step {
                    let step_lc = self.emit_expr(step_expr, items)?;
                    step_lc.is_constant()
                        .ok_or_else(|| format!("Circuit limitation: Loop step must be a compile-time constant. Line {}", span.line))?
                } else {
                    Fr::one()
                };

                let mut current = start_const;
                let step_bi = step_val.0;
                let end_bi = end_const.0;

                while current.0 < end_bi {
                    self.enter_scope();
                    self.const_bindings.insert(loop_var.clone(), current.clone());
                    self.bind(loop_var, WireBinding::Linear(LinearCombination::constant(current.clone())));

                    // Inline loop body
                    for body_stmt in &body.stmts {
                        self.emit_stmt(body_stmt, items)?;
                    }

                    self.exit_scope();
                    current = Fr(current.0.add(&step_bi));
                }
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
                span,
                ..
            } => {
                // Compile-time multiplexed assignment or static if evaluation
                let cond_lc = self.emit_expr(condition, items)?;
                
                if let Some(c) = cond_lc.is_constant() {
                    // Static branch pruning: evaluate only the active branch
                    if !c.0.is_zero() {
                        return self.emit_block(then_block, items);
                    } else if let Some(eb) = else_block {
                        return self.emit_block(eb, items);
                    }
                } else {
                    // Dynamic conditional execution: both branches must be evaluated and output wires merged using selectors
                    
                    // 1. Clone the current scopes and const_bindings for branch isolation
                    let mut then_scopes = self.scopes.clone();
                    let mut else_scopes = self.scopes.clone();
                    let pre_const = self.const_bindings.clone();

                    // Evaluate then branch
                    let original_scopes = std::mem::replace(&mut self.scopes, then_scopes);
                    self.const_bindings = pre_const.clone();
                    let then_ret = self.emit_block(then_block, items)?;
                    then_scopes = std::mem::replace(&mut self.scopes, original_scopes);
                    let then_const = self.const_bindings.clone();

                    // Evaluate else branch
                    let mut else_ret = None;
                    let mut else_const = pre_const.clone();
                    if let Some(eb) = else_block {
                        let original_scopes = std::mem::replace(&mut self.scopes, else_scopes);
                        self.const_bindings = pre_const.clone();
                        else_ret = self.emit_block(eb, items)?;
                        else_scopes = std::mem::replace(&mut self.scopes, original_scopes);
                        else_const = self.const_bindings.clone();
                    }

                    // Merge const bindings: only keep those that are identical in both branches
                    let mut merged_const = HashMap::new();
                    for (k, v1) in then_const {
                        if let Some(v2) = else_const.get(&k) {
                            if v1 == *v2 {
                                merged_const.insert(k, v1);
                            }
                        }
                    }
                    self.const_bindings = merged_const;

                    // Compare and merge the mutated scopes
                    for i in 0..self.scopes.len() {
                        let then_map = &then_scopes[i];
                        let else_map = &else_scopes[i];

                        // Collect all keys in either map at this scope level
                        let mut all_keys = HashSet::new();
                        for k in then_map.keys() {
                            all_keys.insert(k.clone());
                        }
                        for k in else_map.keys() {
                            all_keys.insert(k.clone());
                        }

                        for var in all_keys {
                            let then_val = then_map.get(&var).cloned();
                            let else_val = else_map.get(&var).cloned();

                            if then_val != else_val {
                                let then_lc = match then_val {
                                    Some(WireBinding::Wire(w)) => LinearCombination::variable(w),
                                    Some(WireBinding::Linear(lc)) => lc,
                                    None => self.lookup_in_scopes(&then_scopes, i, &var)
                                        .unwrap_or_else(LinearCombination::zero),
                                };

                                let else_lc = match else_val {
                                    Some(WireBinding::Wire(w)) => LinearCombination::variable(w),
                                    Some(WireBinding::Linear(lc)) => lc,
                                    None => self.lookup_in_scopes(&else_scopes, i, &var)
                                        .unwrap_or_else(LinearCombination::zero),
                                };

                                // Merged value wire:
                                let merged_wire = self.new_wire(&format!("{}_mux", var));
                                let merged_lc = LinearCombination::variable(merged_wire);

                                // Constraints: (cond_lc) * (then_lc - else_lc) = merged_lc - else_lc
                                let mut b_term = then_lc.clone();
                                b_term.add_linear(&else_lc, Fr::from_u64(0).sub(&Fr::one()));

                                let mut c_term = merged_lc.clone();
                                c_term.add_linear(&else_lc, Fr::from_u64(0).sub(&Fr::one()));

                                self.constraints.push(Constraint {
                                    a: cond_lc.clone(),
                                    b: b_term,
                                    c: c_term,
                                    span: Some(span.clone()),
                                });

                                // Update the actual scope level i with the multiplexed wire binding!
                                self.scopes[i].insert(var.clone(), WireBinding::Wire(merged_wire));
                            }
                        }
                    }

                    // Return values multiplexing
                    if then_ret.is_some() || else_ret.is_some() {
                        let tr = then_ret.unwrap_or_else(LinearCombination::zero);
                        let er = else_ret.unwrap_or_else(LinearCombination::zero);

                        let merged_ret_wire = self.new_wire("ret_mux");
                        let merged_ret_lc = LinearCombination::variable(merged_ret_wire);

                        let mut b_term = tr;
                        b_term.add_linear(&er, Fr::from_u64(0).sub(&Fr::one()));

                        let mut c_term = merged_ret_lc.clone();
                        c_term.add_linear(&er, Fr::from_u64(0).sub(&Fr::one()));

                        self.constraints.push(Constraint {
                            a: cond_lc,
                            b: b_term,
                            c: c_term,
                            span: Some(span.clone()),
                        });

                        return Ok(Some(merged_ret_lc));
                    }
                }
            }
            Stmt::Return(expr_opt, _span) => {
                if let Some(expr) = expr_opt {
                    let lc = self.emit_expr(expr, items)?;
                    return Ok(Some(lc));
                }
            }
            Stmt::Expr(expr) => {
                self.emit_expr(expr, items)?;
            }
            Stmt::While { span, .. } => {
                return Err(format!("Circuit target error: Dynamic 'while' loops are non-deterministic and forbidden in ZK circuits. Line {}", span.line));
            }
            Stmt::Break { span } => {
                return Err(format!("Circuit target error: 'break' statements are forbidden in ZK circuits. Line {}", span.line));
            }
            Stmt::Match { span, .. } => {
                return Err(format!("Circuit target error: Pattern matching is currently not supported in ZK circuits. Line {}", span.line));
            }
            _ => {}
        }
        Ok(None)
    }

    fn emit_expr(&mut self, expr: &Expr, items: &[Item]) -> Result<LinearCombination, String> {
        match expr {
            Expr::IntLit(val, _) => {
                Ok(LinearCombination::constant(Fr::from_u64(*val as u64)))
            }
            Expr::BoolLit(val, _) => {
                let v = if *val { Fr::one() } else { Fr::zero() };
                Ok(LinearCombination::constant(v))
            }
            Expr::Ident(name, span) => {
                if let Some(c) = self.lookup_const(name) {
                    return Ok(LinearCombination::constant(c));
                }
                match self.lookup(name) {
                    Some(WireBinding::Wire(id)) => Ok(LinearCombination::variable(id)),
                    Some(WireBinding::Linear(lc)) => Ok(lc),
                    None => Err(format!("Undefined variable {} in circuit expression. Line {}", name, span.line)),
                }
            }
            Expr::BinaryOp { left, op, right, span } => {
                let left_lc = self.emit_expr(left, items)?;
                let right_lc = self.emit_expr(right, items)?;

                match op {
                    BinaryOp::Add => {
                        let mut res = left_lc;
                        res.add_linear(&right_lc, Fr::one());
                        res.simplify();
                        Ok(res)
                    }
                    BinaryOp::Sub => {
                        let mut res = left_lc;
                        let neg_one = Fr::from_u64(0).sub(&Fr::one());
                        res.add_linear(&right_lc, neg_one);
                        res.simplify();
                        Ok(res)
                    }
                    BinaryOp::Mul => {
                        // Optimizations:
                        // Constant * Constant -> Constant
                        // Constant * LC -> LC scaled
                        if let Some(lc) = left_lc.is_constant() {
                            return Ok(right_lc.scale(lc));
                        }
                        if let Some(rc) = right_lc.is_constant() {
                            return Ok(left_lc.scale(rc));
                        }

                        // Otherwise, non-linear multiplication requires a new constraint wire
                        let out_wire = self.new_wire("mul_tmp");
                        self.constraints.push(Constraint {
                            a: left_lc,
                            b: right_lc,
                            c: LinearCombination::variable(out_wire),
                            span: Some(span.clone()),
                        });
                        Ok(LinearCombination::variable(out_wire))
                    }
                    BinaryOp::Div => {
                        // Division: left / right
                        // Constant divisor: scale by inverse
                        if let Some(rc) = right_lc.is_constant() {
                            return Ok(left_lc.scale(rc.inv()));
                        }

                        // Dynamic divisor: w_out * right = left
                        let out_wire = self.new_wire("div_tmp");
                        self.constraints.push(Constraint {
                            a: LinearCombination::variable(out_wire),
                            b: right_lc,
                            c: left_lc,
                            span: Some(span.clone()),
                        });
                        Ok(LinearCombination::variable(out_wire))
                    }
                    BinaryOp::Eq => {
                        // Equality: left == right -> outputs 1 if equal, 0 if not
                        // Constraint-based check:
                        // Let d = left - right
                        // We introduce a helper wire inv_d.
                        // Constrain:
                        // 1) d * (1 - eq) = 0
                        // 2) d * inv_d = eq
                        let mut d = left_lc;
                        d.add_linear(&right_lc, Fr::from_u64(0).sub(&Fr::one()));

                        if let Some(dc) = d.is_constant() {
                            let eq_val = if dc.0.is_zero() { Fr::one() } else { Fr::zero() };
                            return Ok(LinearCombination::constant(eq_val));
                        }

                        let eq_wire = self.new_wire("eq_tmp");
                        let inv_d_wire = self.new_wire("inv_d_tmp");

                        // Constraint 1: d * (1 - eq) = 0
                        // A: d, B: 1 - eq, C: 0
                        let mut b_term = LinearCombination::constant(Fr::one());
                        b_term.add_term(eq_wire, Fr::from_u64(0).sub(&Fr::one()));

                        self.constraints.push(Constraint {
                            a: d.clone(),
                            b: b_term,
                            c: LinearCombination::zero(),
                            span: Some(span.clone()),
                        });

                        // Constraint 2: d * inv_d = eq
                        self.constraints.push(Constraint {
                            a: d,
                            b: LinearCombination::variable(inv_d_wire),
                            c: LinearCombination::variable(eq_wire),
                            span: Some(span.clone()),
                        });

                        Ok(LinearCombination::variable(eq_wire))
                    }
                    BinaryOp::NotEq => {
                        // Not Equal: (left == right) == 0
                        let eq_lc = self.emit_expr(&Expr::BinaryOp {
                            left: left.clone(),
                            op: BinaryOp::Eq,
                            right: right.clone(),
                            span: span.clone(),
                        }, items)?;

                        // return 1 - eq
                        let mut res = LinearCombination::constant(Fr::one());
                        res.add_linear(&eq_lc, Fr::from_u64(0).sub(&Fr::one()));
                        Ok(res)
                    }
                    _ => Err(format!("Circuit target error: Operator {:?} is not natively supported in ZK field constraints. Line {}", op, span.line)),
                }
            }
            Expr::Call { func, args, span } => {
                let func_name = match &**func {
                    Expr::Ident(name, _) => name.clone(),
                    _ => return Err(format!("Invalid function call in circuit. Line {}", span.line)),
                };

                // Validate circuit restrictions: Recursion check
                if self.active_calls.contains(&func_name) {
                    return Err(format!("Circuit limitation violated: Recursion is strictly forbidden in ZK circuits (found recursion path in {}). Line {}", func_name, span.line));
                }

                // Lookup function declaration
                let mut target_decl = None;
                for item in items {
                    if let Item::Func(f) = item {
                        if f.name == func_name {
                            target_decl = Some(f);
                            break;
                        }
                    }
                }

                let f = target_decl.ok_or_else(|| format!("Undefined function {} called. Line {}", func_name, span.line))?;

                // Inline function execution: evaluate arguments and bind to parameters
                self.enter_scope();
                self.active_calls.push(func_name.clone());

                for (param, arg) in f.params.iter().zip(args) {
                    let arg_lc = self.emit_expr(arg, items)?;
                    self.bind(&param.name, WireBinding::Linear(arg_lc));
                }

                let ret_lc = self.emit_block(&f.body, items)?;

                self.active_calls.pop();
                self.exit_scope();

                ret_lc.ok_or_else(|| format!("Function {} did not return a value in circuit context. Line {}", func_name, span.line))
            }
            Expr::Index { base, index, span } => {
                // Validate circuit restrictions: No dynamic pointer arithmetic / dynamic indexing
                // Array elements must be indexed with compile-time constants
                let index_lc = self.emit_expr(index, items)?;
                let index_const = index_lc.is_constant()
                    .ok_or_else(|| format!("Circuit limitation violated: Dynamic pointer arithmetic or dynamic array indexing is forbidden in ZK circuits. Index must be a compile-time constant. Line {}", span.line))?;
                
                // Let's resolve the array lookup
                let base_name = match &**base {
                    Expr::Ident(name, _) => name,
                    _ => return Err(format!("Forbidden index target in ZK backend. Line {}", span.line)),
                };

                // Look up in scopes for bindings
                let index_val = index_const.0.to_decimal_string();
                let indexed_name = format!("{}_{}", base_name, index_val);
                
                if let Some(c) = self.lookup_const(&indexed_name) {
                    return Ok(LinearCombination::constant(c));
                }
                match self.lookup(&indexed_name) {
                    Some(WireBinding::Wire(id)) => Ok(LinearCombination::variable(id)),
                    Some(WireBinding::Linear(lc)) => Ok(lc),
                    None => Err(format!("Undefined array element {}[{}] in circuit. Line {}", base_name, index_val, span.line)),
                }
            }
            _ => Err(format!("Circuit target error: Expression {:?} is unsupported in ZK backends. Line {}", expr, expr.span().line)),
        }
    }

    // ────────────────────────────────────────────────────────
    // 4. Circuit Optimizations
    // ────────────────────────────────────────────────────────

    fn optimize_circuit(&mut self) {
        // Runs constraint-reduction pass:
        // - Merges terms, normalizes, and flattens
        for c in &mut self.constraints {
            c.a.simplify();
            c.b.simplify();
            c.c.simplify();
        }

        let mut iteration = 0;
        loop {
            let mut replacements = HashMap::new();
            let mut seen: HashMap<u64, Vec<(LinearCombination, LinearCombination, usize)>> = HashMap::new();
            let mut duplicate_indices = HashSet::new();

            for (idx, c) in self.constraints.iter().enumerate() {
                // We only optimize constraints where C is a single intermediate wire: w_j
                // A wire is intermediate if it is not 0, not in public/private inputs, and not in outputs.
                if c.c.terms.len() == 1 {
                    let (wire_j, coeff) = (c.c.terms[0].0, &c.c.terms[0].1);
                    if coeff.0 == BigUint::one()
                        && wire_j != 0
                        && !self.public_inputs.contains(&wire_j)
                        && !self.private_inputs.contains(&wire_j)
                        && !self.outputs.contains(&wire_j)
                    {
                        // Compute commutative hash for (A, B) using symmetric addition of individual hashes
                        use std::hash::{Hash, Hasher};
                        use std::collections::hash_map::DefaultHasher;

                        let hash_a = {
                            let mut s = DefaultHasher::new();
                            c.a.hash(&mut s);
                            s.finish()
                        };
                        let hash_b = {
                            let mut s = DefaultHasher::new();
                            c.b.hash(&mut s);
                            s.finish()
                        };
                        let combined_hash = hash_a.wrapping_add(hash_b);

                        let mut found_wire = None;
                        if let Some(candidates) = seen.get(&combined_hash) {
                            for (seen_a, seen_b, seen_wire) in candidates {
                                if (c.a == *seen_a && c.b == *seen_b) || (c.a == *seen_b && c.b == *seen_a) {
                                    found_wire = Some(*seen_wire);
                                    break;
                                }
                            }
                        }

                        if let Some(wire_i) = found_wire {
                            replacements.insert(wire_j, wire_i);
                            duplicate_indices.insert(idx);
                        } else {
                            seen.entry(combined_hash)
                                .or_insert_with(Vec::new)
                                .push((c.a.clone(), c.b.clone(), wire_j));
                        }
                    }
                }
            }

            if replacements.is_empty() {
                break;
            }

            // Apply replacements to all remaining constraints
            let mut new_constraints = Vec::new();
            for (idx, mut c) in self.constraints.drain(..).enumerate() {
                if duplicate_indices.contains(&idx) {
                    continue; // Remove the duplicate constraint
                }

                // Helper to replace wires in a linear combination
                let replace_lc = |lc: &mut LinearCombination, reps: &HashMap<usize, usize>| {
                    let mut changed = false;
                    for term in &mut lc.terms {
                        if let Some(&new_w) = reps.get(&term.0) {
                            term.0 = new_w;
                            changed = true;
                        }
                    }
                    if changed {
                        lc.is_simplified = false;
                        lc.simplify();
                    }
                };

                replace_lc(&mut c.a, &replacements);
                replace_lc(&mut c.b, &replacements);
                replace_lc(&mut c.c, &replacements);
                new_constraints.push(c);
            }
            self.constraints = new_constraints;
            iteration += 1;
            if iteration > 10 {
                break; // Safety limit
            }
        }
    }


    pub fn build_circuit(&self) -> Circuit {
        Circuit {
            num_variables: self.next_var_id,
            variables: self.variables.clone(),
            public_inputs: self.public_inputs.clone(),
            private_inputs: self.private_inputs.clone(),
            outputs: self.outputs.clone(),
            constraints: self.constraints.clone(),
        }
    }

    pub fn write_r1cs_binary(&self, output_path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::BufWriter;
        use std::io::Write;

        let circuit = self.build_circuit();

        // 1. Determine the old-to-new wire mapping
        let mut old_to_new = HashMap::new();
        old_to_new.insert(0, 0); // Constant 1 is mapped to wire 0

        let mut next_new_id = 1;

        // Public outputs map next
        for &w in &circuit.outputs {
            if !old_to_new.contains_key(&w) {
                old_to_new.insert(w, next_new_id);
                next_new_id += 1;
            }
        }
        let n_pub_out = next_new_id - 1;

        // Public inputs map next
        for &w in &circuit.public_inputs {
            if !old_to_new.contains_key(&w) {
                old_to_new.insert(w, next_new_id);
                next_new_id += 1;
            }
        }
        let n_pub_in = next_new_id - 1 - n_pub_out;

        // Private inputs map next
        for &w in &circuit.private_inputs {
            if !old_to_new.contains_key(&w) {
                old_to_new.insert(w, next_new_id);
                next_new_id += 1;
            }
        }
        let n_prv_in = next_new_id - 1 - n_pub_out - n_pub_in;

        // Intermediate/Auxiliary wires map last
        let mut aux_wires = Vec::new();
        for w in 1..circuit.num_variables {
            if !old_to_new.contains_key(&w) {
                aux_wires.push(w);
            }
        }
        aux_wires.sort();
        for w in aux_wires {
            old_to_new.insert(w, next_new_id);
            next_new_id += 1;
        }

        // 2. Write binary .r1cs file
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);
        let encoder = R1csEncoder::new();
        encoder.encode_to_stream(&circuit, &old_to_new, n_pub_out, n_pub_in, n_prv_in, &mut writer)?;

        // 3. Write symbols (.sym) file
        let sym_path = format!("{}.sym", output_path.strip_suffix(".r1cs").unwrap_or(output_path));
        let sym_file = File::create(&sym_path)?;
        let mut sym_writer = BufWriter::new(sym_file);

        let mut entries = Vec::new();
        for (&old, &new) in &old_to_new {
            if old < circuit.variables.len() {
                entries.push((old, new, &circuit.variables[old]));
            }
        }
        entries.sort_by_key(|e| e.1);
        for (old, new, name) in entries {
            writeln!(sym_writer, "{},{},0,{}", old, new, name)?;
        }

        Ok(())
    }
}

pub struct R1csEncoder;

impl R1csEncoder {
    pub fn new() -> Self {
        Self
    }

    pub fn encode_to_stream<W: std::io::Write>(
        &self,
        circuit: &Circuit,
        old_to_new: &HashMap<usize, usize>,
        n_pub_out: usize,
        n_pub_in: usize,
        n_prv_in: usize,
        writer: &mut W,
    ) -> std::io::Result<()> {
        use std::io::Write as IoWrite;

        // Write Magic: "r1cs"
        writer.write_all(b"r1cs")?;

        // Write Version: 1
        writer.write_all(&1u32.to_le_bytes())?;

        // Write nSections: 3 (Header, Constraints, Wire2Label)
        writer.write_all(&3u32.to_le_bytes())?;

        // --- 1. HEADER SECTION ---
        let mut header_buf = Vec::new();
        let fs = 32u32;
        header_buf.write_all(&fs.to_le_bytes())?;

        let prime_bytes = Fr::modulus().to_bytes_le(32);
        header_buf.write_all(&prime_bytes)?;

        let n_wires = circuit.num_variables;
        header_buf.write_all(&(n_wires as u32).to_le_bytes())?;
        header_buf.write_all(&(n_pub_out as u32).to_le_bytes())?;
        header_buf.write_all(&(n_pub_in as u32).to_le_bytes())?;
        header_buf.write_all(&(n_prv_in as u32).to_le_bytes())?;

        let n_labels = n_wires as u64;
        header_buf.write_all(&n_labels.to_le_bytes())?;
        header_buf.write_all(&(circuit.constraints.len() as u32).to_le_bytes())?;

        self.write_section(writer, 1, &header_buf)?;

        // --- 2. CONSTRAINTS SECTION ---
        let mut constraints_buf = Vec::new();
        for c in &circuit.constraints {
            for lc in &[&c.a, &c.b, &c.c] {
                let mut remapped_terms: Vec<(u32, Vec<u8>)> = Vec::new();
                for &(old_wire, ref coeff) in &lc.terms {
                    let new_wire = *old_to_new.get(&old_wire).unwrap_or(&0) as u32;
                    let coeff_bytes = coeff.0.to_bytes_le(32);
                    remapped_terms.push((new_wire, coeff_bytes));
                }
                // Sort by new wire ID ascending
                remapped_terms.sort_by_key(|t| t.0);

                let n_terms = remapped_terms.len() as u32;
                constraints_buf.write_all(&n_terms.to_le_bytes())?;

                for (wire_id, val_bytes) in remapped_terms {
                    constraints_buf.write_all(&wire_id.to_le_bytes())?;
                    constraints_buf.write_all(&val_bytes)?;
                }
            }
        }
        self.write_section(writer, 2, &constraints_buf)?;

        // --- 3. WIRE TO LABEL MAP SECTION ---
        let mut map_buf = Vec::new();
        let mut new_to_old = vec![0u64; n_wires];
        for (&old, &new) in old_to_new {
            if new < n_wires {
                new_to_old[new] = old as u64;
            }
        }
        for label_id in new_to_old {
            map_buf.write_all(&label_id.to_le_bytes())?;
        }
        self.write_section(writer, 3, &map_buf)?;

        Ok(())
    }

    fn write_section<W: std::io::Write>(
        &self,
        writer: &mut W,
        section_type: u32,
        content: &[u8],
    ) -> std::io::Result<()> {
        writer.write_all(&section_type.to_le_bytes())?;
        let size = content.len() as u64;
        writer.write_all(&size.to_le_bytes())?;
        writer.write_all(content)?;
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biguint_ops() {
        let p_str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";
        let p = BigUint::from_str(p_str);
        
        let zero = BigUint::zero();
        let one = BigUint::one();
        
        assert!(zero < one);
        assert!(one < p);
        
        // Addition
        let p_plus_one = p.add(&one);
        assert_eq!(p_plus_one.sub(&one), p);

        // Multiplication
        let two = BigUint::from_u64(2);
        let doubled = p.mul(&two);
        assert_eq!(doubled.sub(&p), p);
    }

    #[test]
    fn test_field_ops() {
        let zero = Fr::from_u64(0);
        let one = Fr::from_u64(1);
        let neg_one = zero.sub(&one);
        
        // neg_one + 1 = 0
        assert_eq!(neg_one.add(&one), zero);
    }
}
