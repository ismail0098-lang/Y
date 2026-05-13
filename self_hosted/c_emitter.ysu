import lib;
import parser;

struct CEmitter {
    c_buffer: String,
    indent_level: usize
}

impl CEmitter {
    @unsafe
    fn new() -> CEmitter {
        let e: CEmitter;
        e.c_buffer = String::new("");
        e.indent_level = 0;
        return e;
    }

    @unsafe
    fn indent(e: &mut CEmitter) {
        let mut i: usize = 0;
        while i < (*e).indent_level {
            String::push_str(&mut (*e).c_buffer, &String::new("    "));
            i += 1;
        }
    }

    @unsafe
    fn emit_func(e: &mut CEmitter, arena: &AstArena, func_idx: usize) {
        let fdecl: FuncDecl = Vec::get_FuncDecl(&(*arena).funcs, func_idx);
        
        CEmitter::indent(e);
        String::push_str(&mut (*e).c_buffer, &String::new("int32_t "));
        String::push_str(&mut (*e).c_buffer, &fdecl.name);
        String::push_str(&mut (*e).c_buffer, &String::new("("));
        
        let mut p: usize = 0;
        while p < fdecl.param_count {
            let param: ParamDecl = Vec::get_ParamDecl(&(*arena).params, fdecl.param_start + p);
            // Default to int32_t for all params in bootstrap to keep it simple
            String::push_str(&mut (*e).c_buffer, &String::new("int32_t "));
            String::push_str(&mut (*e).c_buffer, &param.name);
            if p + 1 < fdecl.param_count {
                String::push_str(&mut (*e).c_buffer, &String::new(", "));
            }
            p += 1;
        }
        
        String::push_str(&mut (*e).c_buffer, &String::new(") {\n"));
        (*e).indent_level += 1;
        
        let mut s: usize = 0;
        while s < fdecl.body_count {
            CEmitter::emit_stmt(e, arena, fdecl.body_start + s);
            s += 1;
        }
        
        (*e).indent_level -= 1;
        CEmitter::indent(e);
        String::push_str(&mut (*e).c_buffer, &String::new("}\n\n"));
    }

    @unsafe
    fn emit_stmt(e: &mut CEmitter, arena: &AstArena, stmt_idx: usize) {
        let stmt: Stmt = Vec::get_Stmt(&(*arena).stmts, stmt_idx);
        CEmitter::indent(e);
        
        if stmt.tag == Stmt_TAG_Let {
            let var_name: String = stmt.data.Let._0;
            let init_idx: usize = stmt.data.Let._2;
            
            String::push_str(&mut (*e).c_buffer, &String::new("int32_t "));
            String::push_str(&mut (*e).c_buffer, &var_name);
            if init_idx > 0 {
                String::push_str(&mut (*e).c_buffer, &String::new(" = "));
                CEmitter::emit_expr(e, arena, init_idx - 1);
            } else {
                String::push_str(&mut (*e).c_buffer, &String::new(" = 0"));
            }
            String::push_str(&mut (*e).c_buffer, &String::new(";\n"));
        } else if stmt.tag == Stmt_TAG_Return {
            let ret_idx: usize = stmt.data.Return._0;
            String::push_str(&mut (*e).c_buffer, &String::new("return "));
            if ret_idx > 0 {
                CEmitter::emit_expr(e, arena, ret_idx - 1);
            }
            String::push_str(&mut (*e).c_buffer, &String::new(";\n"));
        } else if stmt.tag == Stmt_TAG_ExprStmt {
            let expr_idx: usize = stmt.data.ExprStmt._0;
            CEmitter::emit_expr(e, arena, expr_idx - 1);
            String::push_str(&mut (*e).c_buffer, &String::new(";\n"));
        } else if stmt.tag == Stmt_TAG_Assign {
            let target_idx: usize = stmt.data.Assign._0;
            let value_idx: usize = stmt.data.Assign._1;
            CEmitter::emit_expr(e, arena, target_idx - 1);
            String::push_str(&mut (*e).c_buffer, &String::new(" = "));
            CEmitter::emit_expr(e, arena, value_idx - 1);
            String::push_str(&mut (*e).c_buffer, &String::new(";\n"));
        } else if stmt.tag == Stmt_TAG_CompoundAssign {
            let target_idx: usize = stmt.data.CompoundAssign._0;
            let value_idx: usize = stmt.data.CompoundAssign._2;
            CEmitter::emit_expr(e, arena, target_idx - 1);
            // hardcoded to += for now as binaryop isn't fully matched
            String::push_str(&mut (*e).c_buffer, &String::new(" += "));
            CEmitter::emit_expr(e, arena, value_idx - 1);
            String::push_str(&mut (*e).c_buffer, &String::new(";\n"));
        } else if stmt.tag == Stmt_TAG_If {
            let cond_idx: usize = stmt.data.If._0;
            let then_start: usize = stmt.data.If._1;
            let then_count: usize = stmt.data.If._2;
            let else_start: usize = stmt.data.If._3;
            let else_count: usize = stmt.data.If._4;
            
            String::push_str(&mut (*e).c_buffer, &String::new("if ("));
            CEmitter::emit_expr(e, arena, cond_idx - 1);
            String::push_str(&mut (*e).c_buffer, &String::new(") {\n"));
            
            (*e).indent_level += 1;
            let mut i: usize = 0;
            while i < then_count {
                CEmitter::emit_stmt(e, arena, then_start + i);
                i += 1;
            }
            (*e).indent_level -= 1;
            CEmitter::indent(e);
            String::push_str(&mut (*e).c_buffer, &String::new("}\n"));
            
            if else_count > 0 {
                CEmitter::indent(e);
                String::push_str(&mut (*e).c_buffer, &String::new("else {\n"));
                (*e).indent_level += 1;
                let mut j: usize = 0;
                while j < else_count {
                    CEmitter::emit_stmt(e, arena, else_start + j);
                    j += 1;
                }
                (*e).indent_level -= 1;
                CEmitter::indent(e);
                String::push_str(&mut (*e).c_buffer, &String::new("}\n"));
            }
        } else if stmt.tag == Stmt_TAG_While {
            let cond_idx: usize = stmt.data.While._0;
            let body_start: usize = stmt.data.While._1;
            let body_count: usize = stmt.data.While._2;
            
            String::push_str(&mut (*e).c_buffer, &String::new("while ("));
            CEmitter::emit_expr(e, arena, cond_idx - 1);
            String::push_str(&mut (*e).c_buffer, &String::new(") {\n"));
            
            (*e).indent_level += 1;
            let mut i: usize = 0;
            while i < body_count {
                CEmitter::emit_stmt(e, arena, body_start + i);
                i += 1;
            }
            (*e).indent_level -= 1;
            CEmitter::indent(e);
            String::push_str(&mut (*e).c_buffer, &String::new("}\n"));
        }
    }

    @unsafe
    fn emit_expr(e: &mut CEmitter, arena: &AstArena, expr_idx: usize) {
        let expr: Expr = Vec::get_Expr(&(*arena).exprs, expr_idx);
        
        if expr.tag == Expr_TAG_IntLit {
            // Self-hosted Y-Lang doesn't have format! or int->str yet.
            // In the bootstrap, we assume 0 or we'll rely on C's string lit
            String::push_str(&mut (*e).c_buffer, &String::new("0"));
        } else if expr.tag == Expr_TAG_StringLit {
            String::push_str(&mut (*e).c_buffer, &String::new("\""));
            String::push_str(&mut (*e).c_buffer, &expr.data.StringLit._0);
            String::push_str(&mut (*e).c_buffer, &String::new("\""));
        } else if expr.tag == Expr_TAG_BoolLit {
            if expr.data.BoolLit._0 == 1 {
                String::push_str(&mut (*e).c_buffer, &String::new("true"));
            } else {
                String::push_str(&mut (*e).c_buffer, &String::new("false"));
            }
        } else if expr.tag == Expr_TAG_Ident {
            String::push_str(&mut (*e).c_buffer, &expr.data.Ident._0);
        } else if expr.tag == Expr_TAG_BinaryExpr {
            let lhs: usize = expr.data.BinaryExpr._0;
            let rhs: usize = expr.data.BinaryExpr._2;
            String::push_str(&mut (*e).c_buffer, &String::new("("));
            CEmitter::emit_expr(e, arena, lhs - 1);
            // Op matching is skipped in this demo but could be expanded
            String::push_str(&mut (*e).c_buffer, &String::new(" + "));
            CEmitter::emit_expr(e, arena, rhs - 1);
            String::push_str(&mut (*e).c_buffer, &String::new(")"));
        } else if expr.tag == Expr_TAG_Call {
            let func_idx: usize = expr.data.Call._0;
            let args_start: usize = expr.data.Call._1;
            let arg_count: usize = expr.data.Call._2;
            CEmitter::emit_expr(e, arena, func_idx - 1);
            String::push_str(&mut (*e).c_buffer, &String::new("("));
            
            let mut i: usize = 0;
            while i < arg_count {
                let arg_idx: usize = Vec::get_usize(&(*arena).arg_indices, args_start + i);
                CEmitter::emit_expr(e, arena, arg_idx - 1);
                if i + 1 < arg_count {
                    String::push_str(&mut (*e).c_buffer, &String::new(", "));
                }
                i += 1;
            }
            String::push_str(&mut (*e).c_buffer, &String::new(")"));
        } else if expr.tag == Expr_TAG_MemberAccess {
            let base_idx: usize = expr.data.MemberAccess._0;
            let member: String = expr.data.MemberAccess._1;
            CEmitter::emit_expr(e, arena, base_idx - 1);
            String::push_str(&mut (*e).c_buffer, &String::new("."));
            String::push_str(&mut (*e).c_buffer, &member);
        } else if expr.tag == Expr_TAG_Index {
            let base_idx: usize = expr.data.Index._0;
            let idx_idx: usize = expr.data.Index._1;
            CEmitter::emit_expr(e, arena, base_idx - 1);
            String::push_str(&mut (*e).c_buffer, &String::new("["));
            CEmitter::emit_expr(e, arena, idx_idx - 1);
            String::push_str(&mut (*e).c_buffer, &String::new("]"));
        } else if expr.tag == Expr_TAG_UnaryExpr {
            let operand_idx: usize = expr.data.UnaryExpr._1;
            // Assumes Deref for now
            String::push_str(&mut (*e).c_buffer, &String::new("*("));
            CEmitter::emit_expr(e, arena, operand_idx - 1);
            String::push_str(&mut (*e).c_buffer, &String::new(")"));
        }
    }
}
