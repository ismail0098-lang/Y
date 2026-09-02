//! One module, one definition per attribute group - and the reason is a bug,
//! not a style rule.
//!
//! `llvm_emitter::emit_prelude` writes `attributes #0` from the probed host.
//! `cpu_gemm::emit_vnni_micro_module` is a COMPLETE standalone module and wrote
//! `attributes #0` too, because the exact `vpdpwssd` kernels need features the
//! host probe may not have found. Splicing the second into the first gave one
//! module with two definitions of `#0`.
//!
//! **That is not a redefinition error.** `llvm-as` accepts it silently and the
//! LAST one wins - so the second group did not ADD its features to the host's,
//! it REPLACED them, for every function in the module. Measured on a prelude
//! declaring `target-cpu="haswell" target-features="+avx2,+avx,+fma"` - a
//! machine with no AVX-512 - the f32 GEMM compiled to **1,637 `zmm` register
//! references** across `f32_matmul`, `__y_gemm_run`, `__y_gemm_small_m` and
//! `__y_pool_worker`. An illegal-instruction fault, in a kernel that has
//! nothing to do with the exact path. On the AVX-512 host it was silently
//! discarding `target-cpu="znver5"`, `+fma`, `+avx512cd`, `+avx512dq`,
//! `+avx512vl` and `+avx512bf16` from everything.
//!
//! **PERFORMANCE IS NOT THE STORY, and the first record of this said it was.**
//! The note that deferred this increment called it "a silent performance
//! regression in the f32 path". Measured at 512x512x2048, interleaved, best of
//! 40 reps x 8 rounds: f32 **0.727 ms before, 0.729 after**; the exact kernel
//! 6.99 both, its codegen being unchanged by construction. The extra 395 lines
//! of assembly `target-cpu="znver5"` buys in `f32_matmul` are worth nothing at
//! this shape. What the fix buys is the illegal instruction above.
//!
//! **AND THE FIRST TWO MEASUREMENTS OF THAT WERE OF DEAD CODE.** The obvious
//! fixture - the exact nest alone - emits `__y_sgemm_f32_avx512` as `internal`
//! with no caller, so `-O2` deletes it. Both arms reported zero `zmm` outside
//! the VNNI kernels and the asm came back byte-identical, which reads exactly
//! like "the f32 path is unaffected". The fixture here declares BOTH kernels so
//! both modules are live. `feedback-null-metrics-pass-dead-components`, in the
//! measurement written to check for it.
//!
//! **The group cannot simply be deleted**, which is what makes this a
//! renumbering rather than a removal: with the VNNI kernels left on a host
//! group lacking `+avx512vnni`, `clang` does not emit worse code, it aborts -
//! `fatal error: Do not know how to split the result of this operator!` in
//! `__y_gemm_micro_vnni`. `the_vnni_group_still_carries_what_vpdpwssd_needs`
//! is the control that stops "delete the second group" passing.
//!
//! The first three tests are structural and need no external tool: they check
//! the SIGNATURE of the defect class - a group defined twice, a group named but
//! never defined, a signature naming one twice - so the next collision is
//! caught without anyone guessing which emitter introduces it.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

/// Both GEMM modules, both reachable. See the note above on why one kernel is
/// not enough: the f32 entry is `internal` and DCE deletes it.
const SOURCE: &str = r#"
kernel f32_matmul(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, M: I32, N: I32, K: I32) {
    @invariant(i >= 0)
    for i in 0..M step 1 {
        @invariant(j >= 0)
        for j in 0..N step 1 {
            let mut sum: F32 = 0.0;
            @invariant(k >= 0)
            for k in 0..K step 1 {
                let a_val: F32 = block_ptr2d_load(A, i, k, K, M, K);
                let b_val: F32 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a_val * b_val;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}

kernel exact_matmul(A: GlobalMemory<I16>, B: GlobalMemory<I16>, C: GlobalMemory<I64>, M: I32, N: I32, K: I32) {
    @invariant(i >= 0)
    for i in 0..M step 1 {
        @invariant(j >= 0)
        for j in 0..N step 1 {
            @ZeroDrift
            let mut sum: I64 = 0;
            @invariant(k >= 0)
            for k in 0..K step 1 {
                @bounds(min=-1024, max=1024)
                let a_val: I64 = block_ptr2d_load(A, i, k, K, M, K);
                @bounds(min=-1024, max=1024)
                let b_val: I64 = block_ptr2d_load(B, k, j, N, K, N);
                sum = sum + a_val * b_val;
            }
            block_ptr2d_store(C, i, j, N, M, N, sum);
        }
    }
}

fn main() {
}
"#;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// The tag is in the signature rather than in a comment asking the caller to
/// remember: two tests sharing a temp-dir name is a race this repository has
/// hit five times.
fn emit(tag: &str) -> (PathBuf, String) {
    let dir = std::env::temp_dir().join(format!("y_attr_groups_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let src = dir.join("m.ysu");
    std::fs::write(&src, SOURCE).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-llvm")
        .current_dir(repo())
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "the two-kernel fixture must compile:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ir = std::fs::read_to_string(dir.join("m.ll")).expect("emitted IR");
    // Both modules must really be present, or every assertion below is about a
    // module that never had a collision to make.
    assert!(
        ir.contains("__y_sgemm_f32_avx512") && ir.contains("__y_gemm_micro_vnni"),
        "the fixture must emit BOTH the f32 and the exact GEMM module"
    );
    (dir, ir)
}

/// `attributes #N = { .. }` -> N, in source order.
fn group_definitions(ir: &str) -> Vec<u32> {
    ir.lines()
        .filter_map(|l| l.strip_prefix("attributes #"))
        .filter_map(|r| r.split(' ').next())
        .filter_map(|n| n.parse::<u32>().ok())
        .collect()
}

/// function name -> the group numbers its `define` line names, in order.
fn groups_named_by_each_function(ir: &str) -> BTreeMap<String, Vec<u32>> {
    let mut out = BTreeMap::new();
    for l in ir.lines() {
        if !l.starts_with("define ") {
            continue;
        }
        let name = match l.split('@').nth(1).and_then(|r| r.split('(').next()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        // Only the trailing attribute list, after the parameter list closes.
        let tail = match l.rfind(')') {
            Some(i) => &l[i + 1..],
            None => continue,
        };
        let nums: Vec<u32> = tail
            .split_whitespace()
            .filter_map(|t| t.strip_prefix('#'))
            .filter_map(|n| n.parse::<u32>().ok())
            .collect();
        out.insert(name, nums);
    }
    out
}

#[test]
fn no_attribute_group_is_defined_twice() {
    let (dir, ir) = emit("dup");
    let defs = group_definitions(&ir);
    assert!(
        defs.len() >= 2,
        "the fixture must produce more than one group, or this asserts nothing: {defs:?}"
    );
    let mut seen = BTreeMap::new();
    for n in &defs {
        *seen.entry(*n).or_insert(0usize) += 1;
    }
    let dups: Vec<_> = seen.iter().filter(|(_, c)| **c > 1).collect();
    assert!(
        dups.is_empty(),
        "an attribute group is defined more than once, and `llvm-as` will keep \
         only the last: {dups:?} (all definitions: {defs:?}). See this file's \
         header for what that measured.",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_group_a_function_names_is_defined() {
    let (dir, ir) = emit("defined");
    let defined: Vec<u32> = group_definitions(&ir);
    let named = groups_named_by_each_function(&ir);
    assert!(
        named.len() >= 10,
        "the fixture must emit a real module: {} functions",
        named.len()
    );
    let mut carriers = 0usize;
    for (f, gs) in &named {
        for g in gs {
            carriers += 1;
            assert!(
                defined.contains(g),
                "@{f} names attribute group #{g}, which the module never defines"
            );
        }
    }
    assert!(
        carriers >= 10,
        "the fixture must have functions that actually carry a group: {carriers}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_function_names_the_same_group_twice() {
    let (dir, ir) = emit("twice");
    let named = groups_named_by_each_function(&ir);
    for (f, gs) in &named {
        let mut s = gs.clone();
        s.sort_unstable();
        let before = s.len();
        s.dedup();
        assert_eq!(
            before,
            s.len(),
            "@{f} names an attribute group more than once ({gs:?}). \
             `__y_gemm_exact_vnni` carried `#0 #0` - the driver's signature \
             wrote one and `IrBuilder::finish` stamped another."
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_host_target_cpu_is_what_llvm_actually_reads() {
    if !have("llvm-as") || !have("llvm-dis") {
        eprintln!("SKIP: llvm-as/llvm-dis not found");
        return;
    }
    let (dir, ir) = emit("roundtrip");
    let declared = ir
        .lines()
        .find(|l| l.starts_with("attributes #") && l.contains("target-cpu"))
        .map(|l| l.to_string())
        .expect("the prelude must declare a target-cpu");
    let cpu = declared
        .split("\"target-cpu\"=\"")
        .nth(1)
        .and_then(|r| r.split('"').next())
        .expect("target-cpu value")
        .to_string();

    let bc = dir.join("m.bc");
    let back = dir.join("m.dis.ll");
    assert!(Command::new("llvm-as")
        .arg(dir.join("m.ll"))
        .arg("-o")
        .arg(&bc)
        .status()
        .expect("llvm-as")
        .success());
    assert!(Command::new("llvm-dis")
        .arg(&bc)
        .arg("-o")
        .arg(&back)
        .status()
        .expect("llvm-dis")
        .success());
    let round = std::fs::read_to_string(&back).expect("round-tripped IR");
    assert!(
        round.contains(&format!("\"target-cpu\"=\"{cpu}\"")),
        "the module declares target-cpu=\"{cpu}\" and LLVM does not read it \
         back - a duplicate attribute group has replaced the host's"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The bug itself, not its signature.
///
/// Rewrite the prelude's group to a machine with no AVX-512 and compile. Only
/// the exact `vpdpwssd` kernels may use `zmm`; anything else is an instruction
/// the target cannot execute.
#[test]
fn on_a_host_without_avx512_only_the_vnni_kernels_use_it() {
    if !have("clang") {
        eprintln!("SKIP: clang not found");
        return;
    }
    let (dir, ir) = emit("nonavx512");
    let haswell =
        "attributes #0 = { \"target-cpu\"=\"haswell\" \"target-features\"=\"+avx2,+avx,+fma\" }";
    let mut lines: Vec<String> = ir.lines().map(|s| s.to_string()).collect();
    let host = lines
        .iter()
        .position(|l| l.starts_with("attributes #") && l.contains("target-cpu"))
        .expect("prelude group");
    lines[host] = haswell.to_string();
    let patched = dir.join("hasw.ll");
    std::fs::write(&patched, lines.join("\n")).expect("write patched IR");

    let asm = dir.join("hasw.s");
    let out = Command::new("clang")
        .args(["-O2", "-S", "-o"])
        .arg(&asm)
        .arg(&patched)
        .output()
        .expect("clang");
    assert!(
        out.status.success(),
        "the AVX2 module must compile:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = std::fs::read_to_string(&asm).expect("asm");

    // Split into functions the way an assembly listing labels them.
    let mut cur = String::new();
    let mut offenders: Vec<(String, usize)> = Vec::new();
    let mut zmm_in_vnni = 0usize;
    let mut body = String::new();
    let mut flush = |name: &str, body: &str, off: &mut Vec<(String, usize)>, ok: &mut usize| {
        let z = body.matches("%zmm").count();
        if z == 0 {
            return;
        }
        if name.contains("vnni") {
            *ok += z;
        } else {
            off.push((name.to_string(), z));
        }
    };
    for l in s.lines() {
        // Every label carries a `# @name` comment, and an operand never
        // contains `#` in AT&T syntax - so cutting there is safe, and it is
        // what makes the label test work at all. Counting `%zmm` on the cut
        // line also keeps a comment that merely mentions a register out of the
        // tally.
        let t = l.split('#').next().unwrap_or("").trim_end();
        let is_label = t.ends_with(':')
            && !t.starts_with('.')
            && !t.starts_with(' ')
            && !t.starts_with('\t')
            && t[..t.len() - 1]
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == '$');
        if is_label {
            flush(&cur, &body, &mut offenders, &mut zmm_in_vnni);
            cur = t[..t.len() - 1].to_string();
            body.clear();
        } else {
            body.push_str(t);
            body.push('\n');
        }
    }
    flush(&cur, &body, &mut offenders, &mut zmm_in_vnni);

    // Non-vacuity: if the exact kernels stopped using AVX-512 the sweep would
    // report "no offenders" while checking nothing.
    assert!(
        zmm_in_vnni > 100,
        "the exact kernels must still use AVX-512, or this test is vacuous: {zmm_in_vnni}"
    );
    assert!(
        offenders.is_empty(),
        "on a target with no AVX-512, these functions use `zmm` anyway: {offenders:?}. \
         A duplicate `attributes #0` has overridden the host's target-features."
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The control that stops "delete the second group" passing every test above.
///
/// Without a group carrying `+avx512vnni`, the VNNI kernels do not get worse
/// code - `clang` aborts in the backend. So the group is required, and the fix
/// is a distinct number rather than a deletion.
#[test]
fn the_vnni_group_still_carries_what_vpdpwssd_needs() {
    let (dir, ir) = emit("vnnigroup");
    let group = y::cpu_gemm::VNNI_ATTR_GROUP;
    let want = format!("attributes {group} = ");
    let line = ir
        .lines()
        .find(|l| l.starts_with(&want))
        .unwrap_or_else(|| panic!("the module must define {group} for the exact kernels"));
    for f in y::cpu_gemm::VNNI_TARGET_FEATURES.split(',') {
        assert!(
            line.contains(f),
            "{group} must carry {f}, or `__y_gemm_micro_vnni` cannot be lowered: {line}"
        );
    }
    let named = groups_named_by_each_function(&ir);
    let n: u32 = group.trim_start_matches('#').parse().expect("group number");
    let carriers: Vec<_> = named
        .iter()
        .filter(|(_, gs)| gs.contains(&n))
        .map(|(f, _)| f.as_str())
        .collect();
    assert!(
        carriers.len() >= 4,
        "the four exact kernels must name {group}: {carriers:?}"
    );
    for f in ["__y_gemm_micro_vnni", "__y_gemm_exact_vnni"] {
        assert!(carriers.contains(&f), "@{f} must name {group}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
