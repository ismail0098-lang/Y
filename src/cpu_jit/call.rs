//! Checked dynamic calls use compiler-generated C ABI adapters. The host
//! passes a uniform array of bits instead of transmuting heterogeneous types.
use super::JitError;
use crate::ast::{FuncDecl, Item, Program, Type};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt::Write;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbiType {
    Void,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Usize,
    Bool,
    F32,
    F64,
    Pointer,
    /// Address lookup remains available for aggregates and other native types.
    Unsupported(String),
}

impl AbiType {
    pub fn name(&self) -> &str {
        match self {
            Self::Void => "void",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::I64 => "I64",
            Self::U64 => "U64",
            Self::Usize => "usize",
            Self::Bool => "bool",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::Pointer => "ptr",
            Self::Unsupported(name) => name,
        }
    }

    pub fn tag(&self) -> Option<u32> {
        Some(match self {
            Self::Void => 0,
            Self::I8 => 1,
            Self::U8 => 2,
            Self::I16 => 3,
            Self::U16 => 4,
            Self::I32 => 5,
            Self::U32 => 6,
            Self::I64 => 7,
            Self::U64 => 8,
            Self::Usize => 9,
            Self::Bool => 10,
            Self::F32 => 11,
            Self::F64 => 12,
            Self::Pointer => 13,
            Self::Unsupported(_) => return None,
        })
    }

    fn llvm_type(&self) -> Option<&'static str> {
        Some(match self {
            Self::Void => "void",
            Self::Bool => "i1",
            Self::I8 | Self::U8 => "i8",
            Self::I16 | Self::U16 => "i16",
            Self::I32 | Self::U32 => "i32",
            Self::I64 | Self::U64 | Self::Usize => "i64",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::Pointer => "ptr",
            Self::Unsupported(_) => return None,
        })
    }

    fn from_ast(ty: &Type) -> Self {
        match ty {
            Type::Primitive(name, _) | Type::Ident(name, _) => match name.as_str() {
                "I8" | "i8" => Self::I8,
                "U8" | "u8" | "char" => Self::U8,
                "I16" | "i16" => Self::I16,
                "U16" | "u16" => Self::U16,
                "I32" | "i32" => Self::I32,
                "U32" | "u32" => Self::U32,
                "I64" | "i64" | "isize" => Self::I64,
                "U64" | "u64" => Self::U64,
                "usize" => Self::Usize,
                "F32" | "f32" => Self::F32,
                "F64" | "f64" => Self::F64,
                "bool" => Self::Bool,
                "String" | "Vec" | "ptr" => Self::Pointer,
                name => Self::Unsupported(name.into()),
            },
            Type::Reference { .. } | Type::Array { .. } => Self::Pointer,
            Type::Generic { base, .. }
                if matches!(base.as_str(), "GlobalMemory" | "SharedMemory" | "Vec") =>
            {
                Self::Pointer
            }
            ty => Self::Unsupported(format!("{ty:?}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JitValue {
    Void,
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    Usize(usize),
    Bool(bool),
    F32(f32),
    F64(f64),
    Pointer(*mut c_void),
}

impl JitValue {
    pub fn abi_type(self) -> AbiType {
        match self {
            Self::Void => AbiType::Void,
            Self::I8(_) => AbiType::I8,
            Self::U8(_) => AbiType::U8,
            Self::I16(_) => AbiType::I16,
            Self::U16(_) => AbiType::U16,
            Self::I32(_) => AbiType::I32,
            Self::U32(_) => AbiType::U32,
            Self::I64(_) => AbiType::I64,
            Self::U64(_) => AbiType::U64,
            Self::Usize(_) => AbiType::Usize,
            Self::Bool(_) => AbiType::Bool,
            Self::F32(_) => AbiType::F32,
            Self::F64(_) => AbiType::F64,
            Self::Pointer(_) => AbiType::Pointer,
        }
    }

    pub fn bits(self) -> u64 {
        match self {
            Self::Void => 0,
            Self::I8(v) => v as u8 as u64,
            Self::U8(v) => v as u64,
            Self::I16(v) => v as u16 as u64,
            Self::U16(v) => v as u64,
            Self::I32(v) => v as u32 as u64,
            Self::U32(v) => v as u64,
            Self::I64(v) => v as u64,
            Self::U64(v) => v,
            Self::Usize(v) => v as u64,
            Self::Bool(v) => v as u64,
            Self::F32(v) => v.to_bits() as u64,
            Self::F64(v) => v.to_bits(),
            Self::Pointer(v) => v as usize as u64,
        }
    }

    pub(super) fn from_bits(ty: &AbiType, bits: u64) -> Result<Self, JitError> {
        Ok(match ty {
            AbiType::Void => Self::Void,
            AbiType::I8 => Self::I8(bits as i8),
            AbiType::U8 => Self::U8(bits as u8),
            AbiType::I16 => Self::I16(bits as i16),
            AbiType::U16 => Self::U16(bits as u16),
            AbiType::I32 => Self::I32(bits as i32),
            AbiType::U32 => Self::U32(bits as u32),
            AbiType::I64 => Self::I64(bits as i64),
            AbiType::U64 => Self::U64(bits),
            AbiType::Usize => Self::Usize(bits as usize),
            AbiType::Bool => Self::Bool(bits & 1 != 0),
            AbiType::F32 => Self::F32(f32::from_bits(bits as u32)),
            AbiType::F64 => Self::F64(f64::from_bits(bits)),
            AbiType::Pointer => Self::Pointer(bits as usize as *mut c_void),
            AbiType::Unsupported(name) => {
                return Err(JitError::new(format!(
                    "dynamic calls do not support `{name}`"
                )))
            }
        })
    }

    pub fn from_tagged_bits(tag: u32, bits: u64) -> Result<Self, JitError> {
        let ty = match tag {
            0 => AbiType::Void,
            1 => AbiType::I8,
            2 => AbiType::U8,
            3 => AbiType::I16,
            4 => AbiType::U16,
            5 => AbiType::I32,
            6 => AbiType::U32,
            7 => AbiType::I64,
            8 => AbiType::U64,
            9 => AbiType::Usize,
            10 => AbiType::Bool,
            11 => AbiType::F32,
            12 => AbiType::F64,
            13 => AbiType::Pointer,
            _ => return Err(JitError::new(format!("unknown CPU JIT value tag {tag}"))),
        };
        if ty == AbiType::Bool && bits > 1 {
            return Err(JitError::new("CPU JIT bool value must be 0 or 1"));
        }
        let max = match ty {
            AbiType::Void => Some(0),
            AbiType::I8 | AbiType::U8 => Some(u8::MAX as u64),
            AbiType::I16 | AbiType::U16 => Some(u16::MAX as u64),
            AbiType::I32 | AbiType::U32 | AbiType::F32 => Some(u32::MAX as u64),
            _ => None,
        };
        if max.is_some_and(|max| bits > max) {
            return Err(JitError::new(format!(
                "noncanonical {} value bits",
                ty.name()
            )));
        }
        Self::from_bits(&ty, bits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSignature {
    pub name: String,
    pub parameters: Vec<AbiType>,
    pub return_type: AbiType,
}

impl FunctionSignature {
    pub fn supports_dynamic_call(&self) -> bool {
        self.parameters
            .iter()
            .all(|ty| ty.llvm_type().is_some() && *ty != AbiType::Void)
            && self.return_type.llvm_type().is_some()
    }

    pub fn to_json(&self) -> String {
        fn quote(value: &str) -> String {
            let mut result = String::from("\"");
            for ch in value.chars() {
                match ch {
                    '"' => result.push_str("\\\""),
                    '\\' => result.push_str("\\\\"),
                    ch if ch.is_control() => {
                        write!(result, "\\u{:04x}", ch as u32).unwrap();
                    }
                    ch => result.push(ch),
                }
            }
            result.push('"');
            result
        }
        format!(
            "{{\"name\":{},\"parameters\":[{}],\"return_type\":{},\"dynamic_call\":{}}}",
            quote(&self.name),
            self.parameters
                .iter()
                .map(|ty| quote(ty.name()))
                .collect::<Vec<_>>()
                .join(","),
            quote(self.return_type.name()),
            self.supports_dynamic_call()
        )
    }
}

pub(super) fn signatures(ast: &Program) -> BTreeMap<String, FunctionSignature> {
    fn add(function: &FuncDecl, name: String, result: &mut BTreeMap<String, FunctionSignature>) {
        result.insert(
            name.clone(),
            FunctionSignature {
                name,
                parameters: function
                    .params
                    .iter()
                    .map(|p| AbiType::from_ast(&p.ty))
                    .collect(),
                return_type: function
                    .ret_ty
                    .as_ref()
                    .map_or(AbiType::Void, AbiType::from_ast),
            },
        );
    }
    let mut result = BTreeMap::new();
    for item in &ast.items {
        match item {
            Item::Func(f) => add(
                f,
                if f.name == "main" {
                    "ysu_main".into()
                } else {
                    f.name.clone()
                },
                &mut result,
            ),
            Item::Impl(b) => {
                for f in &b.methods {
                    add(f, format!("{}_{}", b.target_type, f.name), &mut result);
                }
            }
            Item::Kernel(k) => {
                result.insert(
                    k.name.clone(),
                    FunctionSignature {
                        name: k.name.clone(),
                        parameters: k.params.iter().map(|p| AbiType::from_ast(&p.ty)).collect(),
                        return_type: AbiType::Void,
                    },
                );
            }
            _ => {}
        }
    }
    result
}

/// Append adapters for supported exported functions and return their names.
pub(super) fn append_adapters(
    ir: &mut String,
    signatures: &BTreeMap<String, FunctionSignature>,
    optimize_call_adapters: bool,
) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    let mut counter = 0;
    // Keep the ABI conversion wrapper small without changing whether source
    // functions inline into other source functions. Inlining large kernels
    // here duplicates their optimization and machine-code compilation.
    let call_attribute = if optimize_call_adapters {
        " noinline"
    } else {
        ""
    };
    for (name, signature) in signatures {
        if !signature.supports_dynamic_call() {
            continue;
        }
        // Functions erased by the host emitter (e.g. ghost declarations) do
        // not get adapters that would introduce a new undefined call.
        if !ir
            .lines()
            .any(|line| line.starts_with("define ") && line.contains(&format!("@{name}(")))
        {
            continue;
        }
        let wrapper = loop {
            let wrapper = format!("__y_jit_dispatch_{counter}");
            counter += 1;
            if !ir.contains(&format!("@{wrapper}(")) {
                break wrapper;
            }
        };
        writeln!(ir, "\ndefine i64 @{wrapper}(ptr %args) {{\nentry:").unwrap();
        let mut arguments = Vec::new();
        for (i, ty) in signature.parameters.iter().enumerate() {
            writeln!(ir, "  %slot{i} = getelementptr i64, ptr %args, i64 {i}\n  %raw{i} = load i64, ptr %slot{i}, align 8").unwrap();
            let llvm = ty.llvm_type().unwrap();
            let value = match ty {
                AbiType::I64 | AbiType::U64 | AbiType::Usize => format!("%raw{i}"),
                AbiType::Pointer => {
                    writeln!(ir, "  %value{i} = inttoptr i64 %raw{i} to ptr").unwrap();
                    format!("%value{i}")
                }
                AbiType::F64 => {
                    writeln!(ir, "  %value{i} = bitcast i64 %raw{i} to double").unwrap();
                    format!("%value{i}")
                }
                AbiType::F32 => {
                    writeln!(ir, "  %narrow{i} = trunc i64 %raw{i} to i32\n  %value{i} = bitcast i32 %narrow{i} to float").unwrap();
                    format!("%value{i}")
                }
                _ => {
                    writeln!(ir, "  %value{i} = trunc i64 %raw{i} to {llvm}").unwrap();
                    format!("%value{i}")
                }
            };
            arguments.push(format!("{llvm} {value}"));
        }
        let ret = signature.return_type.llvm_type().unwrap();
        if signature.return_type == AbiType::Void {
            writeln!(
                ir,
                "  call void @{name}({}){call_attribute}\n  ret i64 0\n}}",
                arguments.join(", ")
            )
            .unwrap();
        } else {
            writeln!(
                ir,
                "  %result = call {ret} @{name}({}){call_attribute}",
                arguments.join(", ")
            )
            .unwrap();
            let value = match &signature.return_type {
                AbiType::I64 | AbiType::U64 | AbiType::Usize => "%result",
                AbiType::Pointer => {
                    writeln!(ir, "  %bits = ptrtoint ptr %result to i64").unwrap();
                    "%bits"
                }
                AbiType::F64 => {
                    writeln!(ir, "  %bits = bitcast double %result to i64").unwrap();
                    "%bits"
                }
                AbiType::F32 => {
                    writeln!(ir, "  %narrow = bitcast float %result to i32\n  %bits = zext i32 %narrow to i64").unwrap();
                    "%bits"
                }
                _ => {
                    writeln!(ir, "  %bits = zext {ret} %result to i64").unwrap();
                    "%bits"
                }
            };
            writeln!(ir, "  ret i64 {value}\n}}").unwrap();
        }
        result.insert(name.clone(), wrapper);
    }
    result
}
