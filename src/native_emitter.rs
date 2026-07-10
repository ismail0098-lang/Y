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
}

impl NativeEmitter {
    pub fn new() -> Self {
        Self {
            code: CodeBuffer::new(),
            symbols: HashMap::new(),
            relocs: Vec::new(),
            stack_offset: 0,
            base_addr: 0x400000,
        }
    }

    pub fn emit_program(&mut self, prog: &Program) -> Vec<u8> {
        self.emit_elf_header();
        self.emit_entry_point();

        for item in &prog.items {
            if let Item::Func(f) = item {
                self.emit_func(f);
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

    fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Return(expr_opt, _) => {
                if let Some(expr) = expr_opt {
                    self.emit_expr(expr);
                }
            }
            Stmt::Let { init, .. } => {
                if let Some(expr) = init {
                    self.emit_expr(expr);
                }
                self.stack_offset += 4;
                // store eax into [rbp - stack_offset]
                // mov [rbp - stack_offset], eax
                self.code.emit8(0x89);
                self.code.emit8(0x45);
                self.code.emit8((256 - self.stack_offset) as u8);
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
            Stmt::ClockDomainBlock { body, .. } => {
                for stmt in &body.stmts {
                    self.emit_stmt(stmt);
                }
            }
            _ => {}
        }
    }

    fn emit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::IntLit(val, _) => {
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

                match op {
                    BinaryOp::Add => {
                        // add eax, ecx
                        self.code.emit8(0x01);
                        self.code.emit8(0xC8);
                    }
                    BinaryOp::Sub => {
                        // sub eax, ecx
                        self.code.emit8(0x29);
                        self.code.emit8(0xC8);
                    }
                    BinaryOp::Mul => {
                        // imul eax, ecx
                        self.code.emit8(0x0F);
                        self.code.emit8(0xAF);
                        self.code.emit8(0xC1);
                    }
                    _ => {}
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
                }
            }
            Expr::Ident(_, _) => {
                // load first local: mov eax, [rbp - 4]
                self.code.emit8(0x8B);
                self.code.emit8(0x45);
                self.code.emit8(252);
            }
            _ => {}
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
