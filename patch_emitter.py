import sys

content = open('llvm_emitter.rs', 'r', encoding='utf-8').read()

content = content.replace(
    '    locals: HashMap<String, String>,\n    /// Track function return types',
    '    locals: HashMap<String, String>,\n    /// Track what struct type a pointer local variable points to\n    pointee_types: HashMap<String, String>,\n    /// Track function return types'
)

content = content.replace(
    '            locals: HashMap::new(),\n            functions: HashMap::new(),',
    '            locals: HashMap::new(),\n            pointee_types: HashMap::new(),\n            functions: HashMap::new(),'
)

content = content.replace(
    '            self.locals.insert(p.name.clone(), ty.clone());\n            writeln!(&mut self.output, "  %{} = alloca {}", p.name, ty).unwrap();',
    '            self.locals.insert(p.name.clone(), ty.clone());\n            if let Some(pty) = self.get_pointee_type(&p.ty) {\n                self.pointee_types.insert(p.name.clone(), pty);\n            }\n            writeln!(&mut self.output, "  %{} = alloca {}", p.name, ty).unwrap();'
)

content = content.replace(
'''                    let ir_ty = match ty {
                        Some(t) => self.emit_type(t),
                        None => {
                            if let Some(init_expr) = init {
                                self.infer_type(init_expr)
                            } else {
                                "i32".into()
                            }
                        }
                    };
                    self.locals.insert(name.clone(), ir_ty.clone());''',
'''                    let ir_ty = match ty {
                        Some(t) => {
                            if let Some(pty) = self.get_pointee_type(t) {
                                self.pointee_types.insert(name.clone(), pty);
                            }
                            self.emit_type(t)
                        },
                        None => {
                            if let Some(init_expr) = init {
                                let init_ty = self.infer_type(init_expr);
                                let pty = self.infer_struct_type(init_expr);
                                if pty != "i32" {
                                    self.pointee_types.insert(name.clone(), pty);
                                }
                                init_ty
                            } else {
                                "i32".into()
                            }
                        }
                    };
                    self.locals.insert(name.clone(), ir_ty.clone());'''
)

content = content.replace(
    'let base_ty = self.infer_type(base);\n                let tmp = self.fresh_tmp();\n                \n                let mut field_index = 0;\n                if let Some(fields) = self.structs.get(base_ty.trim_start_matches(\'%\')) {',
    'let base_ty = self.infer_struct_type(base);\n                let tmp = self.fresh_tmp();\n                \n                let mut field_index = 0;\n                if let Some(fields) = self.structs.get(base_ty.trim_start_matches(\'%\')) {'
)

content = content.replace(
    'let base_ty = self.infer_type(base);\n                let mut field_ty = "i32".to_string(); // fallback\n                if let Some(fields) = self.structs.get(base_ty.trim_start_matches(\'%\')) {',
    'let base_ty = self.infer_struct_type(base);\n                let mut field_ty = "i32".to_string(); // fallback\n                if let Some(fields) = self.structs.get(base_ty.trim_start_matches(\'%\')) {'
)

# Replace trailing `}\n` with the new methods
if content.endswith('}\n'):
    content = content[:-2] + '''
    fn get_pointee_type(&self, ty: &Type) -> Option<String> {
        match ty {
            Type::Reference { base, .. } => {
                if let Type::Ident(name, _) = &**base {
                    if self.structs.contains_key(name) {
                        return Some(format!("%{}", name));
                    }
                }
                None
            }
            Type::Ident(name, _) => {
                if self.structs.contains_key(name) {
                    return Some(format!("%{}", name));
                }
                None
            }
            _ => None
        }
    }

    fn infer_struct_type(&self, expr: &Expr) -> String {
        match expr {
            Expr::Ident(name, _) => self.pointee_types.get(name).cloned().unwrap_or_else(|| "i32".into()),
            Expr::MemberAccess { base, member, .. } => {
                let base_ty = self.infer_struct_type(base);
                if let Some(fields) = self.structs.get(base_ty.trim_start_matches('%')) {
                    for (fname, fty) in fields {
                        if fname == member {
                            if fty.starts_with('%') { return fty.clone(); }
                            return "i32".into();
                        }
                    }
                }
                "i32".into()
            }
            Expr::Call { func, .. } => {
                let func_name = self.emit_call_target(func);
                self.functions.get(&func_name).cloned().unwrap_or_else(|| "i32".into())
            }
            Expr::UnaryOp { op: UnaryOp::Deref, operand, .. } => self.infer_struct_type(operand),
            _ => "i32".into()
        }
    }
}
'''

open('llvm_emitter.rs', 'w', encoding='utf-8').write(content)
print("Done!")
