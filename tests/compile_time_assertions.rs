//! An erased assertion must first have been proved true by the frontend.
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn compile(source: &str) -> (bool, String, bool) {
    let dir = std::env::temp_dir().join(format!(
        "y_const_assert_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("case.ysu");
    let artifact = dir.join("case.ll");
    std::fs::write(&src, source).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-llvm")
        .arg("-o")
        .arg(&artifact)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let emitted = artifact.exists();
    std::fs::remove_dir_all(dir).unwrap();
    (out.status.success(), text, emitted)
}

#[test]
fn false_assertions_fail_before_emission_at_both_syntax_sites() {
    for source in [
        "fn main() { compile_time::assert!(false, \"must fail\"); }",
        "@static_assert(2 * 3 == 7, \"must fail\"); fn main() {}",
    ] {
        let (ok, text, emitted) = compile(source);
        assert!(
            !ok && !emitted,
            "false assertion emitted an artifact: {text}"
        );
        assert!(
            text.contains("Line 1")
                && text.contains("must fail")
                && text.contains("assertion failed"),
            "{text}"
        );
        assert!(!text.contains("[Verified]"), "{text}");
    }
}

#[test]
fn true_assertions_are_evaluated_and_erased() {
    let (ok, text, emitted) = compile(
        "@static_assert(1024 % 32 == 0, \"tile\"); fn main() { compile_time::assert!(!(3 > 4) && 2 * (3 + 4) == 14, \"arithmetic\"); }"
    );
    assert!(ok && emitted, "{text}");
    assert!(text.contains("[Verified]"), "{text}");
}

#[test]
fn unsupported_nonboolean_and_invalid_arithmetic_are_not_verified() {
    for condition in [
        "n > 0",
        "1",
        "1 / 0 == 0",
        "9223372036854775807 + 1 > 0",
        "1.0 == 1.0",
    ] {
        let (ok, text, emitted) = compile(&format!(
            "fn f(n: I32) {{ compile_time::assert!({condition}, \"unproved\"); }} fn main() {{}}"
        ));
        assert!(!ok && !emitted, "{condition}: {text}");
        assert!(
            text.contains("compile-time assertion"),
            "{condition}: {text}"
        );
        assert!(!text.contains("[Verified]"), "{condition}: {text}");
    }
}

#[test]
fn embedded_cpu_entrypoint_propagates_frontend_assertion_errors() {
    use std::ffi::{CStr, CString};
    use y::c_api::{y_free_string, y_interpret_kernel};
    for (source, expected) in [
        (
            "fn main() { compile_time::assert!(false, \"ffi assertion\"); }",
            -1,
        ),
        (
            "fn main() { compile_time::assert!(true, \"ffi control\"); }",
            0,
        ),
    ] {
        let source = CString::new(source).unwrap();
        let mut error = std::ptr::null_mut();
        unsafe {
            assert_eq!(y_interpret_kernel(source.as_ptr(), &mut error), expected);
            if expected == -1 {
                assert!(!error.is_null());
                assert!(CStr::from_ptr(error)
                    .to_str()
                    .unwrap()
                    .contains("assertion failed"));
            } else {
                assert!(error.is_null());
            }
            y_free_string(error);
        }
    }
}
