//! The two CUDA driver bindings must agree about which entry points are
//! ABI-versioned.
//!
//! `libcuda` exports `cuMemAlloc` and `cuMemAlloc_v2` as **different functions
//! at different addresses** (checked here with `nm -D` while writing this: two
//! distinct `T` symbols). The unsuffixed one is the legacy CUDA 3.x form whose
//! byte count is a 32-bit `unsigned int`, so a `usize` passed to it truncates
//! above 4GB. C code never meets this because `cuda.h` `#define`s the plain
//! names to the `_v2` ones; a hand-written `dlsym` binding does.
//!
//! `src/cuda_runtime.rs` has resolved `_v2` first since it was written, with
//! that reason in a comment beside the macro. `src/ysu_gpu_probe.rs` — a second
//! binding of the same API, in a separate binary — resolved the LEGACY names
//! for all six of the versioned entry points it uses. It had never produced a
//! wrong answer, because the largest thing the probe allocates is a 64MB array,
//! four orders of magnitude below where the truncation begins. That is a rule
//! with one implementation, implemented differently a second time — the shape
//! `VnniExact::licenses` had — and it surfaced from the same place: a
//! `never used` warning (on the probe's unused `cuMemcpyDtoH` binding) that a
//! blanket `#![allow(dead_code)]` had been hiding.
//!
//! **The standing limit of an agreement assertion**, stated because this
//! repository has recorded it before for the PTX `.version` floors: flattening
//! *both* files to `resolve!` satisfies agreement. `each_binding_still_uses_the_versioned_form`
//! closes the total-flattening case; a consistent wrong choice for one specific
//! symbol in both files is not caught here, and would need a third copy of the
//! table — which is the bug, not the fix.

use std::collections::BTreeMap;
use std::path::Path;

/// `resolve!(name)` -> `false`, `resolve_v2!(name)` -> `true`.
fn resolutions(src: &str) -> BTreeMap<String, bool> {
    let mut out = BTreeMap::new();
    for line in src.lines() {
        let l = line.trim();
        for (prefix, versioned) in [("resolve_v2!(", true), ("resolve!(", false)] {
            if let Some(rest) = l.strip_prefix(prefix) {
                if let Some(name) = rest.split(')').next() {
                    // A `macro_rules!` definition line is `($name:ident) => {`,
                    // never a call site, so this cannot pick one up.
                    out.insert(name.to_string(), versioned);
                }
                break;
            }
        }
    }
    out
}

fn read(rel: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn the_two_driver_bindings_agree_on_which_symbols_are_abi_versioned() {
    let runtime = resolutions(&read("src/cuda_runtime.rs"));
    let probe = resolutions(&read("src/ysu_gpu_probe.rs"));

    // Non-vacuity: parsing that found nothing would agree perfectly.
    assert!(
        runtime.len() >= 15 && probe.len() >= 15,
        "expected both bindings to resolve at least 15 symbols; parsed \
         {} from cuda_runtime.rs and {} from ysu_gpu_probe.rs - the scan is not \
         reading what it thinks",
        runtime.len(),
        probe.len()
    );

    let shared: Vec<&String> = runtime.keys().filter(|k| probe.contains_key(*k)).collect();
    assert!(
        shared.len() >= 10,
        "only {} symbols are bound by both files; the comparison below is nearly \
         vacuous",
        shared.len()
    );

    let disagree: Vec<String> = shared
        .iter()
        .filter(|k| runtime[**k] != probe[**k])
        .map(|k| {
            format!(
                "{k}: cuda_runtime.rs uses {}, ysu_gpu_probe.rs uses {}",
                if runtime[*k] { "resolve_v2!" } else { "resolve!" },
                if probe[*k] { "resolve_v2!" } else { "resolve!" },
            )
        })
        .collect();

    assert!(
        disagree.is_empty(),
        "the two CUDA driver bindings disagree about which entry points are \
         ABI-versioned. The unsuffixed symbol is the legacy 32-bit-size form and \
         truncates a `usize` byte count above 4GB; `cuda.h` hides this from C by \
         `#define`-ing the plain names to the `_v2` ones. Whichever is right, \
         both files must make the same choice:\n  {}",
        disagree.join("\n  ")
    );
}

#[test]
fn each_binding_still_uses_the_versioned_form() {
    for f in ["src/cuda_runtime.rs", "src/ysu_gpu_probe.rs"] {
        let r = resolutions(&read(f));
        let n = r.values().filter(|v| **v).count();
        assert!(
            n >= 5,
            "{f} resolves only {n} symbols with `resolve_v2!`. Agreement alone is \
             satisfied by flattening BOTH files to the legacy form, which is why \
             this floor exists beside it."
        );
    }
}
