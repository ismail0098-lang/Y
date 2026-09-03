//! `@require(condition)` — the hardware gate, actually evaluated.
//!
//! §9.1 of the language reference documents this as terminating compilation
//! when the host cannot satisfy the condition, and even gives the diagnostic:
//! `error[R0001]: hardware requirement unsatisfied`. Measured at 5ff9f40,
//! before this module existed:
//!
//! * **`error[R0001]` appeared nowhere in the compiler.**
//! * **`sentinel.rs` — which the architecture section names as the module that
//!   "matches hardware constraints specified by `@require` decorators against
//!   physical microarchitectural capabilities" — contained the string
//!   `require` zero times.**
//! * `@require(1 == 0)` compiled and emitted PTX, exit 0.
//! * Its **sole** reader scanned the condition for an identifier whose name
//!   contains `avx512` and set a local `target_is_cpu`, **which was written and
//!   never read**. rustc's dead-code lint cannot see that: the write goes
//!   through an `&mut` parameter.
//!
//! So a directive whose entire purpose is to REFUSE refused nothing. That is
//! the `@zk_target(scheme = "plonkish")` shape at its purest, and unlike
//! `@hdl_emit` there is no missing backend behind it — `sentinel` already
//! probes exactly the capabilities `@require` names, so the fix is to evaluate
//! the condition rather than to refuse the directive.
//!
//! **Which reading answers each feature is a decision, not a detail.**
//!
//! * CPU features come from the **live authority predicates**
//!   (`sentinel::host_has_avx512` and friends), NOT from `.ysu_hw_profile`. A
//!   cached profile can be stale or copied from a better machine — that is a
//!   recorded finding, not a hypothetical — and here a wrong-high answer is an
//!   illegal instruction rather than a slow kernel. The machine wins.
//! * GPU facts come from the **profile**, because that is what §9.1 says
//!   (`Checks the user's .ysu_hw_profile`) and because the profile is what
//!   selects the PTX target in the first place. A GPU `@require` is therefore a
//!   claim about the compilation target, and a CPU one is a claim about this
//!   machine. They are different questions and are answered from different
//!   places on purpose.
//!
//! Everything it cannot answer is **refused by name**: an unknown feature, and
//! any condition shape outside `feature <op> integer`. Fail-closed — a
//! requirement the compiler silently treats as satisfied is worse than no
//! requirement at all, which is precisely the state this replaces.

use crate::ast::Expr;
use crate::sentinel::{self, HardwareProfile};

/// The features `@require` can answer for, and where each answer comes from.
///
/// Kept as one table rather than a `match` with a fallback, so that adding a
/// feature is a deliberate act and an unknown name cannot become "satisfied".
pub const KNOWN_FEATURES: &[&str] = &[
    "avx", "avx2", "avx512", "avx512f", "avx512_vnni", "sm", "sm_count",
];

/// `8.9` -> `89`, `12.0` -> `120`. The corpus and the docs both write the
/// two-digit form (`@require(sm >= 89)` in `tests/test_drift.ysu`), while the
/// profile stores `SM_VERSION=8.9`.
fn sm_as_two_digits(sm: &str) -> Option<i64> {
    let s = sm.trim().trim_start_matches("sm_");
    if let Some((maj, min)) = s.split_once('.') {
        let maj: i64 = maj.trim().parse().ok()?;
        let min: i64 = min.trim().parse().ok()?;
        Some(maj * 10 + min)
    } else {
        s.parse().ok()
    }
}

/// Whether `name` is a feature this compiler knows about at all.
///
/// Deliberately separate from [`feature_value`], because "I do not know that
/// name" and "I know it and cannot determine it here" are different failures
/// with different repairs, and collapsing them gives the wrong diagnosis. On a
/// machine with no probed GPU profile `sm_version` is empty, and a single
/// `Option` would report `@require(sm >= 89)` as an UNKNOWN FEATURE - telling
/// the user to check their spelling when the real answer is "run the probe".
/// Found by the unit test below, which is the only thing that exercises an
/// unprobed profile.
fn is_known_feature(name: &str) -> bool {
    KNOWN_FEATURES.contains(&name)
}

/// The value this machine (for CPU features) or this compilation target (for
/// GPU facts) reports for `name`, when it can be determined.
fn feature_value(name: &str, hw: &HardwareProfile) -> Option<i64> {
    let b = |v: bool| Some(if v { 1 } else { 0 });
    match name {
        // Live, not cached. See the module docstring.
        "avx" | "avx2" => b(sentinel::host_has_avx()),
        "avx512" | "avx512f" => b(sentinel::host_has_avx512()),
        "avx512_vnni" => b(sentinel::host_has_avx512_vnni()),
        // The compilation target, from the profile, as §9.1 specifies.
        "sm" => sm_as_two_digits(&hw.sm_version),
        "sm_count" => Some(hw.sm_count as i64),
        _ => None,
    }
}

fn apply(op: &str, lhs: i64, rhs: i64) -> Option<bool> {
    Some(match op {
        ">=" => lhs >= rhs,
        ">" => lhs > rhs,
        "<=" => lhs <= rhs,
        "<" => lhs < rhs,
        "==" => lhs == rhs,
        "!=" => lhs != rhs,
        _ => return None,
    })
}

/// Renders the condition back as source, for a diagnostic that quotes what the
/// user wrote rather than an AST dump.
fn render(cond: &Expr) -> String {
    match cond {
        Expr::Ident(n, _) => n.clone(),
        Expr::IntLit(v, _) => v.to_string(),
        Expr::BinaryOp { left, op, right, .. } => {
            format!("{} {} {}", render(left), op_str(op), render(right))
        }
        _ => "<condition>".to_string(),
    }
}

fn op_str(op: &crate::ast::BinaryOp) -> &'static str {
    use crate::ast::BinaryOp as B;
    match op {
        B::Ge => ">=",
        B::Gt => ">",
        B::Le => "<=",
        B::Lt => "<",
        B::Eq => "==",
        B::NotEq => "!=",
        _ => "<op>",
    }
}

/// Evaluates one `@require` condition.
///
/// `Ok(())` means satisfied. `Err(msg)` means the build must stop, and covers
/// all three failure modes deliberately: unsatisfied, unknown feature, and a
/// shape this cannot evaluate. **None of them may be silently treated as
/// satisfied** — that is the state this module replaces.
pub fn check(cond: &Expr, hw: &HardwareProfile, line: usize) -> Result<(), String> {
    let (name, op, want) = match cond {
        Expr::BinaryOp { left, op, right, .. } => match (&**left, &**right) {
            (Expr::Ident(n, _), Expr::IntLit(v, _)) => (n.clone(), op_str(op), *v as i64),
            _ => return Err(shape_error(cond, line)),
        },
        _ => return Err(shape_error(cond, line)),
    };

    if !is_known_feature(&name) {
        return Err(format!(
            "Line {line}: error[R0002]: `@require({})` names an unknown hardware feature \
             `{name}`.\n  \
             hint: the features this compiler can answer for are: {}.\n  \
             note: an unknown feature is refused rather than assumed satisfied - a \
             requirement the compiler quietly ignores is worse than no requirement.",
            render(cond),
            KNOWN_FEATURES.join(", ")
        ));
    }

    let Some(have) = feature_value(&name, hw) else {
        return Err(format!(
            "Line {line}: error[R0004]: `@require({})` names `{name}`, which this compiler \
             supports but cannot determine here - there is no probed value for it.\n  \
             hint: `{name}` comes from `.ysu_hw_profile`; delete that file to force a \
             re-probe, or run the hardware probe on a machine with the device present.\n  \
             note: refused rather than assumed satisfied. This is NOT an unknown feature \
             (that is R0002) - the name is right and the value is missing.",
            render(cond)
        ));
    };

    let Some(ok) = apply(op, have, want) else {
        return Err(shape_error(cond, line));
    };

    if ok {
        Ok(())
    } else {
        Err(format!(
            "Line {line}: error[R0001]: hardware requirement unsatisfied: `{}` required, \
             but this host reports `{name} = {have}`.\n  \
             note: CPU features are read from the running machine (CPUID plus the XGETBV \
             check), GPU facts from `.ysu_hw_profile`; delete the profile to re-probe.",
            render(cond)
        ))
    }
}

fn shape_error(cond: &Expr, line: usize) -> String {
    format!(
        "Line {line}: error[R0003]: `@require({})` is not a shape this compiler can \
         evaluate.\n  \
         hint: write `feature <op> integer`, for example `@require(avx512 >= 1)` or \
         `@require(sm >= 89)`; operators are >=, >, <=, <, == and !=.\n  \
         note: refused rather than assumed satisfied.",
        render(cond)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_versions_parse_both_spellings() {
        assert_eq!(sm_as_two_digits("8.9"), Some(89));
        assert_eq!(sm_as_two_digits("12.0"), Some(120));
        assert_eq!(sm_as_two_digits("sm_90"), Some(90));
        assert_eq!(sm_as_two_digits("89"), Some(89));
        assert_eq!(sm_as_two_digits("not a version"), None);
    }

    #[test]
    fn every_advertised_feature_is_recognised_even_when_unprobed() {
        // An UNPROBED profile is the case that matters: `sm_version` is empty,
        // so `feature_value` legitimately answers `None` while the name is
        // perfectly valid. The first version of this asserted every feature
        // RESOLVES against `HardwareProfile::default()` and failed - correctly,
        // and it is what turned up that the two failures were being collapsed
        // into "unknown feature". Recognition and determination are checked
        // separately now, which is the distinction the diagnostics draw.
        let unprobed = HardwareProfile::default();
        for f in KNOWN_FEATURES {
            assert!(is_known_feature(f), "`{f}` is advertised and not recognised");
        }
        assert!(!is_known_feature("nonsense_feature"));

        // CPU features are answerable with no profile at all; GPU ones are not.
        assert!(feature_value("avx512", &unprobed).is_some());
        assert!(
            feature_value("sm", &unprobed).is_none(),
            "an unprobed profile has no SM version, and pretending otherwise would \
             answer a hardware question with a default"
        );
    }

    #[test]
    fn the_operators_are_the_ones_the_message_advertises() {
        for op in [">=", ">", "<=", "<", "==", "!="] {
            assert!(apply(op, 1, 1).is_some(), "`{op}` is advertised and not handled");
        }
        assert!(apply("&&", 1, 1).is_none());
    }
}
