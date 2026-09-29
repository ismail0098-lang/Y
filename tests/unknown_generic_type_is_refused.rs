// ============================================================
//  An unknown generic type became a POINTER, silently.
//
//  `type_checker::resolve_type` ended its `Type::Generic` arm with
//  `SemanticType::Unknown`, and `llvm_emitter::emit_type` ends its
//  `Type::Generic` match with `_ => "ptr"`. `ptr` is a perfectly legal
//  LLVM type, so nothing downstream could object either:
//
//      struct Z { a: Nonsense<F32, 8>, b: I32 }
//      -> %Z = type { ptr, i32 }        // clang: fine
//
//  The same field written `[F32; 8]` -- the spelling the parser
//  actually models -- emits `{ [8 x float], i32 }`. So the two are a
//  different struct layout, and only one of them is what was written.
//
//  Note the `Type::Ident` spelling of the same typo (`let a: F23 = 3;`)
//  is NOT silent: it emits `alloca %F23` for an undeclared type and
//  clang rejects it with "Cannot allocate unsized type", pointing at
//  generated IR rather than the user's line. Bad diagnostics, not a
//  wrong artifact -- which is why only the generic arm is treated as
//  the design-rule violation here.
// ============================================================

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static N: AtomicUsize = AtomicUsize::new(0);

fn bin() -> std::path::PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("Y")
}

fn compile(src: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "y_ungen_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("t.ysu");
    std::fs::write(&f, src).unwrap();
    let out = Command::new(bin())
        .arg(&f)
        .arg("--emit-llvm")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    // The emitted IR lands in the working directory as a file, so read it
    // back BEFORE the temp dir goes away -- the first version of this helper
    // deleted the dir first and then looked for `[8 x float]` in stdout, where
    // it never appears.
    let ir = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "ll"))
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .collect::<String>();
    let text = format!(
        "{}{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
        ir
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !text.contains("Syntax Error"),
        "fixture did not parse, so it proves nothing about the checker:\n{text}"
    );
    text
}

fn binding(ty: &str) -> String {
    compile(&format!(
        "fn main() -> I32 {{\n    let a: {ty} = ZeroInit;\n    return 0;\n}}\n"
    ))
}

#[test]
fn an_undefined_generic_type_is_refused_by_name() {
    let out = binding("Nonsense<F32, 8>");
    assert!(
        out.contains("unknown generic type") && out.contains("Nonsense"),
        "an undefined generic type was accepted:\n{out}"
    );
}

#[test]
fn a_plausible_but_undefined_generic_is_refused_too() {
    // `Array<T, N>` reads like it must be part of the language. It is not:
    // it appears in no `.ysu`, no doc, and no arm of the parser or checker,
    // so it was an undefined name that happened to parse -- and became a
    // pointer where `[F32; 8]` becomes an inline array.
    let out = binding("Array<F32, 8>");
    assert!(
        out.contains("unknown generic type") && out.contains("Array"),
        "`Array<F32, 8>` was accepted as a type:\n{out}"
    );
}

#[test]
fn every_generic_the_corpus_actually_uses_still_compiles() {
    // The control, and it is not decorative: `GlobalMemory` appears 254 times
    // in `tests/*.ysu` and `Fragment` 27, so a whitelist that forgot either
    // would take out most of the corpus. Refusing everything unknown is sound
    // and useless if it also refuses what the language is made of.
    for ty in ["GlobalMemory<F32>", "Vec<F32>"] {
        let out = binding(ty);
        assert!(
            !out.contains("unknown generic type"),
            "`{ty}` was refused:\n{out}"
        );
    }
}

#[test]
fn the_bracket_array_spelling_is_an_inline_array_not_a_pointer() {
    // What the wrong answer was measured against. This is the spelling the
    // parser models (`Type::Array`), and it must keep producing storage.
    let out = compile(
        "struct Z {\n    a: [F32; 8],\n    b: I32,\n}\n\nfn main() -> I32 { return 0; }\n",
    );
    assert!(
        out.contains("%Z = type { [8 x float], i32 }"),
        "an inline array field stopped being inline:\n{out}"
    );
}
