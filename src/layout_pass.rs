// ============================================================
//  Y  —  Compiler IR Layout System & Layout Transformation Pass
//  layout_pass.rs
//
//  Formalizes tensor layout encodings (Blocked, Shared, MMA, Slice)
//  and implements ConvertLayoutPass to lower and swizzle tensor
//  fragment layouts across memory boundaries.
// ============================================================

#![allow(dead_code)]

use crate::ast::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutEncoding {
    /// Contiguous multi-dimensional block layout (threads per warp, block shape)
    Blocked {
        size_per_thread: Vec<u32>,
        threads_per_warp: Vec<u32>,
        warps_per_cta: Vec<u32>,
        order: Vec<u32>,
    },
    /// Shared memory layout with swizzle padding parameters
    Shared {
        vec_size: u32,
        per_phase: u32,
        max_phase: u32,
        order: Vec<u32>,
    },
    /// Hardware Tensor Core MMA fragment layout
    Mma {
        version_major: u32,
        version_minor: u32,
        warps_per_cta: Vec<u32>,
    },
    /// Slice layout for reduced rank tensors
    Slice {
        dim: u32,
        parent: Box<LayoutEncoding>,
    },
}

pub struct ConvertLayoutPass {
    pub conversions_performed: usize,
}

impl ConvertLayoutPass {
    pub fn new() -> Self {
        ConvertLayoutPass {
            conversions_performed: 0,
        }
    }

    /// Analyzes Program AST and transforms layout conversions between memory spaces.
    pub fn run(&mut self, prog: &mut Program) {
        for item in &mut prog.items {
            if let Item::Kernel(kernel) = item {
                self.transform_block(&mut kernel.body);
            }
        }
    }

    fn transform_block(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::Let { init: Some(expr), .. } => {
                    self.transform_expr(expr);
                }
                Stmt::Expr(expr) => {
                    self.transform_expr(expr);
                }
                Stmt::For { body, .. } => {
                    self.transform_block(body);
                }
                _ => {}
            }
        }
    }

    fn transform_expr(&mut self, expr: &mut Expr) {
        if let Expr::Call { func, args, .. } = expr {
            if let Expr::Ident(name, _) = &mut **func {
                if name == "convert_layout" && args.len() == 2 {
                    self.conversions_performed += 1;
                    *name = "swizzle_layout_transform".into();
                }
            }
            for arg in args {
                self.transform_expr(arg);
            }
        }
    }
}

/// Pass 2: Eliminates 32-bank shared memory stalls by swizzling 2D matrix fragment indices
/// using bitwise XOR: swizzled_index = row ^ (col >> 2).
pub struct SmemBankSwizzlePass {
    pub swizzles_applied: usize,
}

impl SmemBankSwizzlePass {
    pub fn new() -> Self {
        SmemBankSwizzlePass {
            swizzles_applied: 0,
        }
    }

    pub fn run(&mut self, prog: &mut Program) {
        for item in &mut prog.items {
            if let Item::Kernel(kernel) = item {
                self.swizzle_block(&mut kernel.body);
            }
        }
    }

    fn swizzle_block(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::Let { init: Some(expr), .. } => {
                    self.swizzle_expr(expr);
                }
                Stmt::Expr(expr) => {
                    self.swizzle_expr(expr);
                }
                Stmt::For { body, .. } => {
                    self.swizzle_block(body);
                }
                _ => {}
            }
        }
    }

    fn swizzle_expr(&mut self, expr: &mut Expr) {
        if let Expr::Call { func, args, .. } = expr {
            if let Expr::Ident(name, _) = &mut **func {
                if (name == "load_shared" || name == "store_shared") && args.len() >= 2 {
                    self.swizzles_applied += 1;
                    *name = format!("{}_swizzled_xor", name);
                }
            }
            for arg in args {
                self.swizzle_expr(arg);
            }
        }
    }
}

