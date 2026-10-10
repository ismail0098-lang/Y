//! The LLVM backend declares the C runtime's functions itself (`emit_prelude`),
//! and nothing checked those declarations against `c_src/runtime.c`.
//!
//! Seven of them were declared to return `ptr` or `i64` while the C definition
//! returned `int32_t` - `ystr_len` among them - so the caller read a register
//! whose upper half the callee never set. Four integer arguments went the other
//! way, declared `i64` and received as `int32_t`: `print_int(5000000000)`
//! printed `705032704`.
//!
//! This file asks clang what each definition's type IS (its AST dump), and
//! compares every declaration the backend emits against it:
//!
//! * a return is never wider in the declaration than in the definition;
//! * an integer argument has the same width on both sides, except where the
//!   definition is shown to use only the low bits (`EXCEPTIONS`);
//! * a `ptr` argument is a pointer, or one of the runtime's 32-bit handles -
//!   the pool is mapped `MAP_32BIT`, so a handle fits in an `int32_t`.
use std::collections::BTreeMap;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

/// (function, argument index, why the narrower declaration is safe).
const EXCEPTIONS: &[(&str, usize, &str)] = &[(
    "ystr_push",
    1,
    "declared `i8`, received as `int32_t` and stored as `(char)c`: only the low byte is read",
)];

fn clang_available() -> bool {
    Command::new("clang").arg("--version").output().is_ok()
}

fn llvm_width(ty: &str) -> Option<u32> {
    Some(match ty.trim() {
        "void" => 0,
        "i1" => 1,
        "i8" => 8,
        "i16" => 16,
        "i32" | "float" => 32,
        "i64" | "double" | "ptr" => 64,
        _ => return None,
    })
}

fn c_width(ty: &str) -> Option<u32> {
    let t = ty.trim().trim_start_matches("const ").trim();
    if t.ends_with('*') {
        return Some(64);
    }
    Some(match t {
        "void" => 0,
        "_Bool" | "bool" | "char" | "int8_t" | "uint8_t" => 8,
        "int16_t" | "uint16_t" | "short" => 16,
        "int32_t" | "uint32_t" | "int" | "unsigned int" | "float" => 32,
        "int64_t" | "uint64_t" | "long" | "long long" | "size_t" | "double" => 64,
        _ => return None,
    })
}

/// `declare RET @NAME(PARAMS)` lines of a module.
fn declarations(ir: &str) -> BTreeMap<String, (String, Vec<String>)> {
    let mut out = BTreeMap::new();
    for line in ir.lines() {
        let Some(rest) = line.trim().strip_prefix("declare ") else { continue };
        let Some(at) = rest.find(" @") else { continue };
        let ret = rest[..at].trim().to_string();
        let after = &rest[at + 2..];
        let Some(open) = after.find('(') else { continue };
        let name = after[..open].to_string();
        let close = after.rfind(')').unwrap();
        let params = after[open + 1..close]
            .split(',')
            .map(|p| p.split_whitespace().next().unwrap_or("").to_string())
            .filter(|p| !p.is_empty())
            .collect();
        out.insert(name, (ret, params));
    }
    out
}

/// The C type clang gives `name`'s definition: `(return, [params])`.
fn c_signature(runtime: &std::path::Path, name: &str) -> Option<(String, Vec<String>)> {
    let out = Command::new("clang")
        .args(["-fsyntax-only", "-DY_NO_MAIN", "-fno-color-diagnostics", "-Xclang", "-ast-dump"])
        .args(["-Xclang", "-ast-dump-filter", "-Xclang", name])
        .arg(runtime)
        .output()
        .expect("run clang");
    let text = String::from_utf8_lossy(&out.stdout);
    let needle = format!(" {name} '");
    let line = text
        .lines()
        .find(|l| l.starts_with("FunctionDecl") && l.contains(&needle) && l.contains("runtime.c"))?;
    let ty = line.split(&needle).nth(1)?.split('\'').next()?;
    let open = ty.find('(')?;
    let ret = ty[..open].trim().to_string();
    let params = ty[open + 1..ty.rfind(')')?]
        .split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty() && p != "void")
        .collect();
    Some((ret, params))
}

#[test]
fn every_runtime_declaration_agrees_with_its_definition() {
    if !clang_available() {
        eprintln!("SKIP: no clang on this machine, so this test checked NOTHING");
        return;
    }
    let dir = pinned::pinned_scratch("runtime_abi", pinned::SM_PINNED);
    let src = dir.join("p.ysu");
    std::fs::write(&src, "fn main() -> I32 {\n    return 0;\n}\n").unwrap();
    let ll = dir.join("p.ll");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-llvm")
        .arg("-o")
        .arg(&ll)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let ir = std::fs::read_to_string(&ll).unwrap();
    let runtime = pinned::repo().join("c_src/runtime.c");

    let mut problems = Vec::new();
    let mut compared = 0;
    for (name, (ret, params)) in declarations(&ir) {
        let Some((c_ret, c_params)) = c_signature(&runtime, &name) else { continue };
        compared += 1;
        let (lw, cw) = (llvm_width(&ret), c_width(&c_ret));
        match (lw, cw) {
            (Some(l), Some(c)) if l > c => {
                problems.push(format!("{name}: declared to return {ret}, defined to return {c_ret}"))
            }
            (Some(_), Some(_)) => {}
            _ => problems.push(format!("{name}: return {ret} vs {c_ret} has no width here")),
        }
        if params.len() != c_params.len() {
            problems.push(format!("{name}: {} arguments declared, {} defined", params.len(), c_params.len()));
            continue;
        }
        for (i, (p, c)) in params.iter().zip(&c_params).enumerate() {
            if p == "ptr" {
                if !(c.ends_with('*') || c == "int32_t") {
                    problems.push(format!("{name} argument {i}: ptr declared, {c} defined"));
                }
                continue;
            }
            let excepted = EXCEPTIONS.iter().any(|(f, k, _)| *f == name && *k == i);
            match (llvm_width(p), c_width(c)) {
                (Some(l), Some(w)) if l == w || excepted => {}
                _ => problems.push(format!("{name} argument {i}: {p} declared, {c} defined")),
            }
        }
    }
    // The prelude declares a dozen runtime functions; a parse that matched
    // none of them would report agreement about nothing.
    assert!(compared >= 10, "only {compared} declarations compared:\n{ir}");
    assert!(problems.is_empty(), "the declarations disagree with c_src/runtime.c:\n  {}", problems.join("\n  "));
}

/// The observable case: a 64-bit value printed through the runtime.
#[test]
fn print_int_prints_a_64_bit_value_whole() {
    let dir = pinned::pinned_scratch("runtime_print_int", pinned::SM_PINNED);
    let src = dir.join("p.ysu");
    std::fs::write(
        &src,
        "fn main() -> I32 {\n    let x: I64 = 5000000000;\n    print_int(x);\n    print_int(-3);\n    return 0;\n}\n",
    )
    .unwrap();
    let bin = dir.join("p");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    if !bin.exists() {
        if !clang_available() {
            eprintln!("SKIP: no clang on this machine, so this test checked NOTHING");
            return;
        }
        panic!("did not build:\n{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    }
    let ran = Command::new(&bin).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&ran.stdout), "5000000000-3");
}
