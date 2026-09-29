//! Array bounds behind a struct field, and the masked index idiom.
//!
//! Two findings, and the second was made reachable by the first.
//!
//! 1. **An array reached through a struct field was never bounds-checked.**
//!    `@safe`'s array rule is one of the three `[Strict Safety]` guarantees,
//!    and `s.buffer[k]` for a completely unconstrained `k` compiled clean and
//!    exited 0. The check was there and the member-access path never reached
//!    it, so the guarantee simply did not apply to any array behind a field -
//!    `feedback-guards-consulted-at-one-site`, with the site being a whole
//!    access form rather than a call site.
//!
//! 2. **`eval_interval` does not model `BitAnd`, and that is the safest index
//!    idiom there is.** Closing (1) immediately refused `tests/ring_buffer.ysu`
//!    on `s.buffer[current_tail & 1023]` into a `[I64; 1024]` - a correct
//!    program, and the canonical masked ring-buffer index. The refusal was
//!    right about what it could prove and wrong about the program.
//!
//!    `x & c` with a NON-NEGATIVE constant `c` lies in `[0, c]` whatever `x`
//!    is: the result's set bits are a subset of `c`'s, and `c >= 0` leaves the
//!    sign bit clear, so the result can be neither negative nor larger than
//!    `c`. That holds for an entirely UNBOUNDED `x`, which is the whole point
//!    and is why it has to be decided before `eval_interval` demands an
//!    interval for both operands - `current_tail` is a struct field read and
//!    has none, so the operator was never consulted.
//!
//! The controls below are what separate "the mask is modelled" from "the check
//! is being skipped": a mask WIDER than the array must be refused naming the
//! inferred maximum, which is only possible if the interval is really computed.
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static SEQ: AtomicUsize = AtomicUsize::new(0);

/// Compile `src` through the real binary. The tag is in the signature and is
/// paired with an atomic counter: a tag makes a left-behind directory legible,
/// the counter is what guarantees two tests cannot collide on one path.
fn compile(tag: &str, src: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!(
        "y_sfab_{}_{}_{}",
        std::process::id(),
        tag,
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let file = dir.join("probe.ysu");
    std::fs::write(&file, src).expect("write probe");

    let exe = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release/Y");
    let exe = if exe.exists() {
        exe
    } else {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/debug/Y")
    };
    let out = Command::new(&exe)
        .arg(&file)
        .arg("--emit-llvm")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), text)
}

const STRUCT: &str = "struct S { buffer: [I64; 1024], }\n";
const MAIN: &str = "\nfn main() -> I32 { return 0; }\n";

fn field_index(expr: &str) -> String {
    format!(
        "{}fn f(s: &mut S, k: I64, m: I64) -> I32 {{ s.buffer[{}] = 1; return 1; }}{}",
        STRUCT, expr, MAIN
    )
}

/// Finding 1. This compiled and exited 0 before the member-access path reached
/// the bounds check, so `@safe` silently exempted every array behind a field.
#[test]
fn an_unprovable_index_into_a_struct_field_array_is_refused() {
    let (ok, text) = compile("plain", &field_index("k"));
    assert!(!ok, "an unconstrained index into a struct field compiled:\n{}", text);
    assert!(
        text.contains("no statically provable bounds"),
        "refused, but not by the bounds check:\n{}",
        text
    );
}

/// Finding 2. The masked ring-buffer index, which is `tests/ring_buffer.ysu`.
#[test]
fn a_constant_masked_index_is_provably_in_range() {
    let (ok, text) = compile("mask", &field_index("k & 1023"));
    assert!(ok, "`k & 1023` into a [I64; 1024] was refused:\n{}", text);
}

/// The control that says the mask is MODELLED rather than the check skipped.
/// A skipped check cannot know the maximum; this refusal has to name it.
#[test]
fn a_mask_wider_than_the_array_is_refused_naming_the_bound() {
    let (ok, text) = compile("wide", &field_index("k & 2047"));
    assert!(!ok, "`k & 2047` into a [I64; 1024] compiled:\n{}", text);
    assert!(
        text.contains("inferred max: 2047") && text.contains("array size 1024"),
        "refused, but without naming the interval it inferred - so the mask \
         may not be modelled at all:\n{}",
        text
    );
}

/// A negative mask leaves the sign bit set, so `[0, c]` does not hold and the
/// index is not provable. Fail-closed rather than a guessed bound.
#[test]
fn a_negative_mask_is_refused() {
    let (ok, text) = compile("neg", &field_index("k & (0 - 2)"));
    assert!(!ok, "a negative mask was treated as a provable bound:\n{}", text);
    assert!(
        text.contains("no statically provable bounds"),
        "refused, but not by the bounds check:\n{}",
        text
    );
}

/// A non-constant mask says nothing about the result's range.
#[test]
fn a_non_constant_mask_is_refused() {
    let (ok, text) = compile("var", &field_index("k & m"));
    assert!(!ok, "a variable mask was treated as a provable bound:\n{}", text);
    assert!(
        text.contains("no statically provable bounds"),
        "refused, but not by the bounds check:\n{}",
        text
    );
}

/// The mask rule is a property of the index expression, not of where the array
/// lives. Without this, a fix that only worked behind a struct field would pass
/// every test above.
#[test]
fn the_mask_rule_holds_for_a_local_array_too() {
    let (ok, text) = compile(
        "local",
        "fn f(k: I64) -> I64 { let arr: [I64; 1024] = {}; return arr[k & 1023]; }\nfn main() -> I32 { return 0; }\n",
    );
    assert!(ok, "`k & 1023` into a local [I64; 1024] was refused:\n{}", text);
}

/// The hole a mutation found in the five cases above, and the reason they did
/// not cover it: `a_non_constant_mask_is_refused` uses a bare parameter, which
/// has no interval AT ALL, so it is refused before the mask rule is consulted
/// and says nothing about what that rule does with a mask it CAN see.
///
/// A mask whose range includes a negative value is the unsound case: at `m =
/// -1`, `k & m` is `k`, entirely unbounded. Widening the rule to a non-constant
/// mask that is provably NON-NEGATIVE would be sound (`x & m <= m`), so this
/// pins the negative-spanning case specifically rather than forbidding that.
#[test]
fn a_mask_whose_range_includes_a_negative_value_is_refused() {
    let (ok, text) = compile(
        "negrange",
        &format!(
            "{}fn f(s: &mut S, k: I64) -> I32 {{\n    @bounds(-1, 1023) let m: I64 = 1023;\n    s.buffer[k & m] = 1;\n    return 1;\n}}{}",
            STRUCT, MAIN
        ),
    );
    assert!(
        !ok,
        "a mask ranging over [-1, 1023] was treated as a provable bound - at \
         `m = -1` the index is `k`, which is unbounded:\n{}",
        text
    );
    assert!(
        text.contains("no statically provable bounds"),
        "refused, but not by the bounds check:\n{}",
        text
    );
}
