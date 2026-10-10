//! A name defined twice in the function namespace is refused by the front end.
//!
//! `fn f() -> I32 { return 1; } fn f() -> I32 { return 2; }` was not refused:
//! the type checker kept the second signature, the default backend then
//! failed inside clang ("clang failed", no reason), `--emit-llvm` and
//! `--emit-cpu` wrote output that does not compile, and only `--emit-native`
//! named the problem. One namespace holds top-level functions and kernels,
//! `impl` methods (`Type_method`) and enum constructors (`Enum_Variant`), so a
//! collision between any two of those is the same defect.
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

fn front_end(name: &str, src: &str, flag: Option<&str>) -> (bool, String) {
    let dir = pinned::pinned_scratch(&format!("dupdef_{name}"), pinned::SM_PINNED);
    let path = dir.join(format!("{name}.ysu"));
    std::fs::write(&path, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&path).arg("-o").arg(dir.join(format!("{name}.out"))).current_dir(&dir);
    if let Some(f) = flag {
        cmd.arg(f);
    }
    let out = cmd.output().expect("run Y");
    (
        out.status.success(),
        format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)),
    )
}

const MAIN: &str = "fn main() -> I32 {\n    return 0;\n}\n";

#[test]
fn every_kind_of_repeated_definition_is_refused_on_every_backend() {
    for (name, items, shown) in [
        ("fn", "fn f() -> I32 {\n    return 1;\n}\n\nfn f() -> I32 {\n    return 2;\n}\n", "`fn f` is defined twice"),
        (
            "method",
            "struct Pa { x: I32 }\n\nimpl Pa {\n    fn get(v: I32) -> I32 {\n        return 1;\n    }\n}\n\n\
             impl Pa {\n    fn get(v: I32) -> I32 {\n        return 2;\n    }\n}\n",
            "`fn Pa::get` is defined twice",
        ),
        (
            "ctor",
            "enum Shape { Dot, Rect(I32, I32) }\n\nfn Shape_Dot() -> I32 {\n    return 3;\n}\n",
            "is defined twice",
        ),
    ] {
        let src = format!("{items}\n{MAIN}");
        for flag in [None, Some("--emit-llvm"), Some("--emit-cpu"), Some("--emit-native")] {
            let (ok, out) = front_end(&format!("{name}{}", flag.unwrap_or("").replace('-', "_")), &src, flag);
            assert!(!ok, "{name} with {flag:?} was accepted:\n{out}");
            assert!(out.contains(shown), "{name} with {flag:?}: the refusal does not name it:\n{out}");
        }
    }
}

/// The control: the same method name in two different `impl`s is two
/// functions, and a repeated LOCAL name is a shadowing `let`, not a definition.
#[test]
fn distinct_owners_may_share_a_method_name() {
    let src = format!(
        "struct Pa {{ x: I32 }}\n\nstruct Pb {{ x: I32 }}\n\n\
         impl Pa {{\n    fn get(v: I32) -> I32 {{\n        return 1;\n    }}\n}}\n\n\
         impl Pb {{\n    fn get(v: I32) -> I32 {{\n        return 2;\n    }}\n}}\n\n{MAIN}"
    );
    let (ok, out) = front_end("control", &src, Some("--emit-llvm"));
    assert!(ok, "{out}");
    assert!(!out.contains("defined twice"), "{out}");
}
