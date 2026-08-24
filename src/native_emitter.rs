// ============================================================
//  Y — Native x86-64 ELF Emitter (Rust Driver Backend)
//  native_emitter.rs
//
//  Directly translates Y AST into a minimal executable ELF64 binary.
//  No gcc, no clang. Pure machine code generation.
// ============================================================

use crate::ast::*;
use std::collections::HashMap;

pub struct CodeBuffer {
    pub bytes: Vec<u8>,
}

impl CodeBuffer {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn emit8(&mut self, val: u8) {
        self.bytes.push(val);
    }

    pub fn emit16(&mut self, val: u16) {
        self.emit8((val & 0xFF) as u8);
        self.emit8(((val >> 8) & 0xFF) as u8);
    }

    pub fn emit32(&mut self, val: u32) {
        self.emit16((val & 0xFFFF) as u16);
        self.emit16(((val >> 16) & 0xFFFF) as u16);
    }

    pub fn emit64(&mut self, val: u64) {
        self.emit32((val & 0xFFFFFFFF) as u32);
        self.emit32(((val >> 32) & 0xFFFFFFFF) as u32);
    }

    pub fn patch32(&mut self, offset: usize, val: u32) {
        self.bytes[offset] = (val & 0xFF) as u8;
        self.bytes[offset + 1] = ((val >> 8) & 0xFF) as u8;
        self.bytes[offset + 2] = ((val >> 16) & 0xFF) as u8;
        self.bytes[offset + 3] = ((val >> 24) & 0xFF) as u8;
    }
}

pub struct Reloc {
    pub offset: usize,
    pub target_name: String,
}

pub struct NativeEmitter {
    pub code: CodeBuffer,
    pub symbols: HashMap<String, usize>,
    pub relocs: Vec<Reloc>,
    pub stack_offset: usize,
    pub base_addr: u64,
    /// Name -> positive byte offset below `rbp`. Without this, `Expr::Ident`
    /// emitted `mov eax, [rbp-4]` for EVERY identifier, so every name in a
    /// function read the value of its first local.
    locals: HashMap<String, usize>,
    /// Constructs this backend cannot encode. It writes an executable ELF, so
    /// a silent gap here is a runnable artifact that computes the wrong thing -
    /// the most severe form of the failure this repo's design rule describes.
    pub emit_errors: Vec<String>,
}

/// The prologue reserves 64 bytes, and the `disp8` addressing this emitter uses
/// only reaches -128.
const STACK_BYTES: usize = 64;

impl NativeEmitter {
    pub fn new() -> Self {
        Self {
            code: CodeBuffer::new(),
            symbols: HashMap::new(),
            relocs: Vec::new(),
            stack_offset: 0,
            base_addr: 0x400000,
            locals: HashMap::new(),
            emit_errors: Vec::new(),
        }
    }

    pub fn emit_program(&mut self, prog: &Program) -> Vec<u8> {
        self.emit_elf_header();
        self.emit_entry_point();

        for item in &prog.items {
            match item {
                Item::Func(f) => self.emit_func(f),

                // A `kernel` is GPU code and this backend emits an x86-64 ELF,
                // so it cannot be lowered. It used to be DROPPED SILENTLY:
                //
                //     kernel k(Out: GlobalMemory<U32>, N: U32) { ... }
                //     fn main() {}
                //
                // produced a 162-byte executable BYTE-IDENTICAL to the one for
                // `fn main() {}` on its own, under "Compiled to native ELF
                // executable!" and exit 0. The whole kernel contributed
                // nothing and nothing said so.
                //
                // This is the mirror of the bug in `ptx_emitter::emit_program`,
                // which matched `Item::Kernel` and dropped everything else. The
                // two backends were each keeping only the half they understood
                // and discarding the rest without a word.
                //
                // `--emit-cpu` already refuses this program - it walks the
                // kernel body and rejects `thread_idx_x` by name - so the host
                // backends now agree.
                Item::Kernel(k) => {
                    self.emit_errors.push(format!(
                        "[Native x86-64 Backend] `kernel {}` is GPU code and has \
                         no native lowering. Dropping it would produce an \
                         executable that silently does none of the work. Compile \
                         kernels with --emit-ptx.",
                        k.name
                    ));
                }

                // These declare types or names and emit no code of their own.
                //
                // An `impl` method and a `module` function ARE code, and this
                // loop does drop them - but the only way to reach one is a
                // `Path` callee (`P::get()`, `M::helper()`), which `emit_expr`
                // already refuses by name as "a call through a computed
                // callee". Verified with both spellings rather than assumed,
                // because that is the difference between a covered gap and an
                // unexamined one. A method nothing calls contributes nothing.
                //
                // Matched exhaustively, with no `_ =>` arm, so a new `Item`
                // variant is a compile error here rather than another silent
                // drop.
                Item::Struct(_)
                | Item::Enum(_)
                | Item::Import(_)
                | Item::StaticAssert(_)
                | Item::Impl(_)
                | Item::Const(_)
                | Item::Module(_) => {}
            }
        }

        self.emit_syscall_wrappers();

        self.patch_relocs();
        self.code.bytes.clone()
    }

    fn emit_elf_header(&mut self) {
        let cb = &mut self.code;
        // ELF Magic: 0x7F 'E' 'L' 'F'
        cb.emit8(0x7F);
        cb.emit8(b'E');
        cb.emit8(b'L');
        cb.emit8(b'F');
        cb.emit8(2); // Class: 64-bit
        cb.emit8(1); // Data: Little-endian
        cb.emit8(1); // Version: 1
        cb.emit8(0); // OS ABI: System V
        for _ in 0..8 {
            cb.emit8(0); // padding
        }
        cb.emit16(2); // Type: ET_EXEC (Executable)
        cb.emit16(62); // Machine: EM_X86_64
        cb.emit32(1); // Version: 1
        // Entry point virtual address (assuming text starts right after headers at base + 120)
        cb.emit64(self.base_addr + 120);
        cb.emit64(64); // Program header offset (64 bytes)
        cb.emit64(0); // Section header offset (none)
        cb.emit32(0); // Flags
        cb.emit16(64); // ELF header size (64 bytes)
        cb.emit16(56); // Program header size (56 bytes)
        cb.emit16(1); // Program header entry count
        cb.emit16(0); // Section header size
        cb.emit16(0); // Section header count
        cb.emit16(0); // Section header string table index

        // Program Header (PT_LOAD, Readable/Executable)
        cb.emit32(1); // p_type: PT_LOAD
        cb.emit32(5); // p_flags: PF_R | PF_X
        cb.emit64(0); // p_offset
        cb.emit64(self.base_addr); // p_vaddr
        cb.emit64(self.base_addr); // p_paddr
        cb.emit64(0); // p_filesz (patched later or dynamically sized)
        cb.emit64(0); // p_memsz (patched later or dynamically sized)
        cb.emit64(0x1000); // p_align (4KB alignment)
    }

    fn emit_entry_point(&mut self) {
        self.symbols.insert("_start".to_string(), self.code.len());
        // Call main
        self.emit_call_rel32("main");
        // Exit process: syscall(60, eax)
        // mov edi, eax
        self.code.emit8(0x89);
        self.code.emit8(0xC7);
        // mov eax, 60 (sys_exit)
        self.code.emit8(0xB8);
        self.code.emit32(60);
        // syscall
        self.code.emit8(0x0F);
        self.code.emit8(0x05);
    }

    fn emit_func(&mut self, f: &FuncDecl) {
        self.symbols.insert(f.name.clone(), self.code.len());

        // push rbp
        self.code.emit8(0x55);
        // mov rbp, rsp
        self.code.emit8(0x48);
        self.code.emit8(0x89);
        self.code.emit8(0xE5);
        // sub rsp, 64 (stack reservation)
        self.code.emit8(0x48);
        self.code.emit8(0x81);
        self.code.emit8(0xEC);
        self.code.emit32(64);

        self.stack_offset = 0;
        self.locals.clear();

        // System V passes the first six integer arguments in registers, and
        // nothing used to store them anywhere - so a function with parameters
        // read whatever happened to be on the stack.
        const ARG_STORE: [&[u8]; 6] = [
            &[0x89, 0x7D], // mov [rbp-N], edi
            &[0x89, 0x75], // mov [rbp-N], esi
            &[0x89, 0x55], // mov [rbp-N], edx
            &[0x89, 0x4D], // mov [rbp-N], ecx
            &[0x44, 0x89, 0x45], // mov [rbp-N], r8d
            &[0x44, 0x89, 0x4D], // mov [rbp-N], r9d
        ];
        if let Some(rt) = &f.ret_ty {
            self.check_type_width(rt, "a return type", &f.body.span);
        }
        for (i, param) in f.params.iter().enumerate() {
            if i >= ARG_STORE.len() {
                let _ = param;
                let sp = f.body.span.clone();
                self.unsupported("a function with more than six parameters", &sp);
                break;
            }
            self.check_type_width(&param.ty, "a parameter", &param.span);
            let Some(off) = self.alloc_local(&param.name, &f.body.span) else {
                break;
            };
            for b in ARG_STORE[i] {
                self.code.emit8(*b);
            }
            self.code.emit8((256 - off) as u8);
        }

        for stmt in &f.body.stmts {
            self.emit_stmt(stmt);
        }

        // add rsp, 64
        self.code.emit8(0x48);
        self.code.emit8(0x81);
        self.code.emit8(0xC4);
        self.code.emit32(64);
        // pop rbp
        self.code.emit8(0x5D);
        // ret
        self.code.emit8(0xC3);
    }

    /// Reserves four bytes of frame for `name` and returns its offset, or
    /// refuses when the fixed 64-byte frame is exhausted. Returning `None`
    /// rather than wrapping the `disp8` matters: `(256 - offset) as u8` past
    /// 128 addresses memory ABOVE `rbp`, i.e. the caller's frame.
    fn alloc_local(&mut self, name: &str, span: &Span) -> Option<usize> {
        self.stack_offset += 4;
        if self.stack_offset > STACK_BYTES {
            let off = self.stack_offset;
            self.unsupported(
                &format!(
                    "a function needing more than {} bytes of locals ({} so far)",
                    STACK_BYTES, off
                ),
                span,
            );
            return None;
        }
        let off = self.stack_offset;
        self.locals.insert(name.to_string(), off);
        Some(off)
    }

    fn unsupported(&mut self, what: &str, span: &Span) {
        self.emit_errors.push(format!(
            "[Native x86-64 Backend] {} (line {}, col {}) cannot be encoded.",
            what, span.line, span.col
        ));
    }

    /// Refuse a declared type this backend cannot represent.
    ///
    /// The whole datapath is 32 bits - `eax`, `ecx`, `imul`, and a `mov eax,
    /// imm32` for every literal - so a 64-bit type here is a claim the
    /// generated code does not honour. It is not merely a large-value problem:
    ///
    /// ```text
    /// let a: I64 = 100000; let b: I64 = 100000; return (a * b) >> 32;
    /// ```
    ///
    /// is 2, and this backend answered 0 under "Compiled to native ELF
    /// executable!" and exit 0. Widening the datapath is a feature (REX.W on
    /// every instruction, `movabs` for the immediates), not a typo, so the
    /// answer is a named refusal - the same choice the branch-free statements
    /// above make.
    ///
    /// This is the `ptx_emitter` integer-width gotcha found in a THIRD
    /// backend, after `llvm_emitter`. When a gotcha is written for one
    /// backend, grep the others for its shape.
    fn check_type_width(&mut self, ty: &Type, what: &str, span: &Span) {
        let name = match ty {
            Type::Primitive(n, _) | Type::Ident(n, _) => n.as_str(),
            _ => return,
        };
        if matches!(name, "I64" | "U64" | "i64" | "u64" | "isize" | "usize") {
            self.unsupported(
                &format!("{} of 64-bit type `{}` (this backend's datapath is 32 bits)", what, name),
                span,
            );
        }
    }

    fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Return(expr_opt, _) => {
                if let Some(expr) = expr_opt {
                    self.emit_expr(expr);
                }
            }
            Stmt::Let {
                name, ty, init, span, ..
            } => {
                if let Some(t) = ty {
                    self.check_type_width(t, "a `let`", span);
                }
                if let Some(expr) = init {
                    self.emit_expr(expr);
                }
                if let Some(off) = self.alloc_local(name, span) {
                    // mov [rbp - off], eax
                    self.code.emit8(0x89);
                    self.code.emit8(0x45);
                    self.code.emit8((256 - off) as u8);
                }
            }
            Stmt::Expr(expr) => {
                self.emit_expr(expr);
            }
            Stmt::SafeBlock(block, _) => {
                for stmt in &block.stmts {
                    self.emit_stmt(stmt);
                }
            }
            Stmt::GhostBlock(block, _) => {
                for stmt in &block.stmts {
                    self.emit_stmt(stmt);
                }
            }
            Stmt::HintBlock { body, .. } => {
                for stmt in &body.stmts {
                    self.emit_stmt(stmt);
                }
            }
            Stmt::ClockDomainBlock { body, .. } => {
                for stmt in &body.stmts {
                    self.emit_stmt(stmt);
                }
            }
            Stmt::TypeAlias { .. } | Stmt::CompileTimeAssert { .. } => {}
            // `_ => {}` used to live here. This backend has no branches and no
            // assignment, so `if`, `while`, `for`, `=` and `+=` all emitted
            // NOTHING - the ELF ran, exited 0, and computed a different program.
            other => {
                let span = other.span();
                let what = match other {
                    Stmt::If { .. } => "`if` (this backend emits no branches)",
                    Stmt::While { .. } => "`while` (this backend emits no branches)",
                    Stmt::For { .. } => "`for` (this backend emits no branches)",
                    Stmt::Break { .. } => "`break`",
                    Stmt::Match { .. } => "`match`",
                    Stmt::Assign { .. } => "assignment to an existing variable",
                    Stmt::CompoundAssign { .. } => "a compound assignment (`+=` and friends)",
                    Stmt::Chisel(..) => "a `chisel` block",
                    _ => "this statement",
                };
                self.unsupported(what, &span);
            }
        }
    }

    fn emit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::IntLit(val, span) => {
                // `emit32(*val as u32)` silently dropped the top half of any
                // literal that did not fit. `let a: I64 = 4294967296; return
                // a >> 32;` is 1 and this emitted 0 - in a RUNNABLE ELF, under
                // a success banner. Same defect as `llvm_emitter::infer_type`
                // typing every literal `i32`, and as the PTX one before it.
                if *val > i32::MAX as i64 || *val < i32::MIN as i64 {
                    let v = *val;
                    self.unsupported(
                        &format!(
                            "the integer literal {} (this backend's datapath is 32 bits, so it \
                             would be truncated to {})",
                            v, v as i32
                        ),
                        span,
                    );
                    return;
                }
                // mov eax, val
                self.code.emit8(0xB8);
                self.code.emit32(*val as u32);
            }
            Expr::BinaryOp { left, op, right, .. } => {
                self.emit_expr(left);
                // push rax
                self.code.emit8(0x50);
                self.emit_expr(right);
                // mov ecx, eax
                self.code.emit8(0x89);
                self.code.emit8(0xC1);
                // pop rax
                self.code.emit8(0x58);

                // `_ => {}` used to close this match, so Div, Mod, every
                // comparison, every bitwise op and both shifts emitted NO
                // instruction at all - leaving the LEFT operand in eax and
                // calling it the answer. `9 / 2` returned 9.
                match op {
                    // add eax, ecx
                    BinaryOp::Add => self.emit_bytes(&[0x01, 0xC8]),
                    // sub eax, ecx
                    BinaryOp::Sub => self.emit_bytes(&[0x29, 0xC8]),
                    // imul eax, ecx
                    BinaryOp::Mul => self.emit_bytes(&[0x0F, 0xAF, 0xC1]),
                    // cdq ; idiv ecx   -> quotient in eax
                    BinaryOp::Div => self.emit_bytes(&[0x99, 0xF7, 0xF9]),
                    // cdq ; idiv ecx ; mov eax, edx  -> remainder
                    BinaryOp::Mod => self.emit_bytes(&[0x99, 0xF7, 0xF9, 0x89, 0xD0]),
                    // and/or/xor eax, ecx
                    BinaryOp::BitAnd => self.emit_bytes(&[0x21, 0xC8]),
                    BinaryOp::BitOr => self.emit_bytes(&[0x09, 0xC8]),
                    BinaryOp::BitXor => self.emit_bytes(&[0x31, 0xC8]),
                    // shl/sar eax, cl - arithmetic shift right, to match I32
                    BinaryOp::Shl => self.emit_bytes(&[0xD3, 0xE0]),
                    BinaryOp::Shr => self.emit_bytes(&[0xD3, 0xF8]),
                    // cmp eax, ecx ; setcc al ; movzx eax, al
                    BinaryOp::Eq
                    | BinaryOp::NotEq
                    | BinaryOp::Lt
                    | BinaryOp::Gt
                    | BinaryOp::Le
                    | BinaryOp::Ge => {
                        let cc = match op {
                            BinaryOp::Eq => 0x94,
                            BinaryOp::NotEq => 0x95,
                            BinaryOp::Lt => 0x9C,
                            BinaryOp::Gt => 0x9F,
                            BinaryOp::Le => 0x9E,
                            _ => 0x9D,
                        };
                        self.emit_bytes(&[0x39, 0xC8, 0x0F, cc, 0xC0, 0x0F, 0xB6, 0xC0]);
                    }
                    // `&&` and `||` short-circuit, which needs a branch this
                    // backend cannot emit. Evaluating both sides is a different
                    // language, so refuse rather than approximate.
                    BinaryOp::And | BinaryOp::Or => {
                        let sp = expr.span();
                        self.unsupported("`&&` / `||` (short-circuit needs a branch)", &sp);
                    }
                }
            }
            Expr::Call { func, args, .. } => {
                let num_args = args.len();
                for i in 0..num_args {
                    self.emit_expr(&args[i]);
                    // push rax
                    self.code.emit8(0x50);
                }

                if num_args >= 6 {
                    // pop r9
                    self.code.emit8(0x41);
                    self.code.emit8(0x59);
                }
                if num_args >= 5 {
                    // pop r8
                    self.code.emit8(0x41);
                    self.code.emit8(0x58);
                }
                if num_args >= 4 {
                    // pop rcx
                    self.code.emit8(0x59);
                }
                if num_args >= 3 {
                    // pop rdx
                    self.code.emit8(0x5A);
                }
                if num_args >= 2 {
                    // pop rsi
                    self.code.emit8(0x5E);
                }
                if num_args >= 1 {
                    // pop rdi
                    self.code.emit8(0x5F);
                }

                if let Expr::Ident(name, _) = &**func {
                    self.emit_call_rel32(name);
                } else {
                    // The argument setup was emitted and then no CALL, so the
                    // callee's return value was whatever was already in eax.
                    let sp = func.span();
                    self.unsupported("a call through a computed callee", &sp);
                }
            }
            Expr::Ident(name, span) => match self.locals.get(name).copied() {
                Some(off) => {
                    // mov eax, [rbp - off]
                    self.code.emit8(0x8B);
                    self.code.emit8(0x45);
                    self.code.emit8((256 - off) as u8);
                }
                None => {
                    let span = span.clone();
                    self.unsupported(&format!("the name `{}` (no local of that name)", name), &span);
                }
            },
            // Floats, strings, chars, bools, indexing, member access, struct
            // literals and unary operators all reached a `_ => {}` here and
            // emitted nothing, leaving whatever was already in eax.
            other => {
                let span = other.span();
                let what = match other {
                    Expr::FloatLit(..) => "a float literal (this backend is integer-only)",
                    Expr::StringLit(..) => "a string literal",
                    Expr::CharLit(..) => "a char literal",
                    Expr::BoolLit(..) => "a bool literal",
                    Expr::UnaryOp { .. } => "a unary operator",
                    Expr::Index { .. } => "an index expression",
                    Expr::MemberAccess { .. } => "a field access",
                    Expr::Path { .. } => "a `Namespace::member` path",
                    Expr::StructLit { .. } => "a struct literal",
                    Expr::BlockExpr(..) => "a block expression",
                    _ => "this expression",
                };
                self.unsupported(what, &span);
            }
        }
    }

    fn emit_bytes(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.code.emit8(*b);
        }
    }

    fn emit_call_rel32(&mut self, target: &str) {
        // call rel32 (opcode 0xE8)
        self.code.emit8(0xE8);
        self.relocs.push(Reloc {
            offset: self.code.len(),
            target_name: target.to_string(),
        });
        self.code.emit32(0);
    }

    fn patch_relocs(&mut self) {
        // Patch ELF sizes in Program Header
        let filesz = self.code.len() as u64;
        // p_filesz is at offset 64 + 32 = 96
        self.code.patch32(96, filesz as u32);
        self.code.patch32(100, (filesz >> 32) as u32);
        // p_memsz is at offset 64 + 40 = 104
        self.code.patch32(104, filesz as u32);
        self.code.patch32(108, (filesz >> 32) as u32);

        for reloc in &self.relocs {
            if let Some(&target_offset) = self.symbols.get(&reloc.target_name) {
                // rel32 offset is relative to the instruction after the call (reloc.offset + 4)
                let call_next = reloc.offset + 4;
                let rel = (target_offset as isize) - (call_next as isize);
                self.code.patch32(reloc.offset, rel as u32);
            }
        }
    }

    fn emit_syscall_wrappers(&mut self) {
        let offset = self.code.len();
        self.symbols.insert("sys_write".to_string(), offset);
        self.symbols.insert("write".to_string(), offset);

        // sys_write:
        // mov eax, 1 (sys_write syscall number)
        self.code.emit8(0xB8);
        self.code.emit32(1);
        // syscall
        self.code.emit8(0x0F);
        self.code.emit8(0x05);
        // ret
        self.code.emit8(0xC3);
    }
}
