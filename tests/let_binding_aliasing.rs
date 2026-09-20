//! A `let` binding must not share a register with the binding it was
//! initialised from.
//!
//! `emit_convert` returns its argument UNCHANGED when no conversion is needed,
//! so `let mut k: I32 = tid;` bound `k` to the very register `tid` lives in.
//! `Stmt::Assign` then writes through that alias, so **assigning one variable
//! silently changed another**:
//!
//! ```text
//!     mov.u32 %r1, %tid.x;      // tid
//!     add.s32 %r4, %r1, %r2;    // k + bsz
//!     mov.u32 %r1, %r4;         // k = ...   <- clobbers tid
//! ```
//!
//! It compiles clean, `ptxas` accepts it, and it is only correct when nothing
//! reads the aliased name afterwards. That is why it survived: a grid-stride
//! kernel launched with enough workers to cover the range in one pass never
//! re-reads the initialiser, so it matches a CPU reference exactly and
//! diverges at every other launch geometry.
//!
//! `tests/deterministic_reduce.ysu` -- a kernel this repository carries proofs
//! about -- writes `let mut i: I64 = worker;` and was clobbering `worker`. It
//! never reads `worker` again, so that instance is latent; `train_spec.ysu`'s
//! `let idx: I32 = i;` aliases a LOOP COUNTER, which is the same hazard one
//! assignment away. Found while a grid-strided scatter kernel outside this
//! repository hit it live.
//!
//! Three tests, because the device test cannot be the only cover: two read the
//! emitted PTX and run anywhere, and the control is what stops the fix
//! degenerating into "emit a copy for every binding".

use std::path::Path;
use std::process::Command;

const BLOCK: u32 = 64;
const N: u32 = 256;

fn compile(fixture: &str) -> String {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    // Compile a COPY: `--emit-ptx` writes next to its input, and rewriting a
    // committed artifact from a test races every other binary doing the same.
    // The fixture name alone is NOT unique: two tests in this file compile
    // `let_alias_wide.ysu`, so one test's `remove_dir_all` landed while the
    // other was reading -- presenting as "no .ptx emitted" with the module
    // perfectly valid. A name makes the requirement visible; only a counter
    // makes it unique.
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "y_let_alias_{}_{}_{}",
        std::process::id(),
        fixture.replace('/', "_"),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("k.ysu");
    std::fs::copy(repo.join(fixture), &src).expect("fixture missing");
    let _ = std::fs::copy(repo.join(".ysu_hw_profile"), dir.join(".ysu_hw_profile"));

    let out = Command::new(bin.join("Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(repo)
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "{} did not compile:\n{}{}",
        fixture,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(dir.join("k.ptx")).expect("no .ptx emitted");
    let _ = std::fs::remove_dir_all(&dir);
    ptx
}

/// Strip comments, so nothing below can match the emitter's own prose about
/// an instruction instead of the instruction.
fn code(ptx: &str) -> Vec<String> {
    ptx.lines()
        .map(|l| l.split("//").next().unwrap().trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// The register a `mov.<ty> %rX, <src>;` writes into.
fn dst_of(line: &str) -> Option<String> {
    let rest = line.strip_prefix("mov.")?;
    let (_, operands) = rest.split_once(' ')?;
    Some(operands.split(',').next()?.trim().to_string())
}

#[test]
fn a_binding_initialised_from_another_does_not_share_its_register() {
    let lines = code(&compile("tests/let_alias.ysu"));

    // `tid` is whatever register `%tid.x` was moved into.
    let tid = lines
        .iter()
        .find(|l| l.contains("%tid.x"))
        .and_then(|l| dst_of(l))
        .expect("no register was loaded from %tid.x");

    // The loop's induction update is the last `mov` before the back edge.
    let back = lines
        .iter()
        .position(|l| l.starts_with("bra $WHILE_START"))
        .expect("the fixture's while loop did not emit a back edge");
    let updated = lines[..back]
        .iter()
        .rev()
        .find_map(|l| dst_of(l))
        .expect("the loop body assigned nothing");

    assert_ne!(
        updated, tid,
        "`let mut k: I32 = tid;` bound k to tid's own register ({tid}), so the \
         loop's `k = k + bsz` writes through it and `tid` is destroyed. \
         Emitted:\n{}",
        lines.join("\n")
    );

    // And the value the program actually uses afterwards must still be `tid`.
    // Without this, binding `k` to a fresh register while ALSO re-pointing the
    // later read at it would satisfy the assertion above.
    let atomic = lines
        .iter()
        .position(|l| l.starts_with("red.global"))
        .expect("the fixture's atomic_add did not emit a reduction");
    let feeds_index = lines[..atomic]
        .iter()
        .rev()
        .take(6)
        .any(|l| l.contains(&format!(", {tid};")));
    assert!(
        feeds_index,
        "the reduction's index is not derived from {tid}, the register holding \
         the thread index. Emitted:\n{}",
        lines.join("\n")
    );
}

#[test]
fn an_ordinary_binding_still_costs_no_copy() {
    // The control. "Copy every binding" satisfies the test above and makes the
    // backend emit a redundant `mov` for every `let` in every kernel, which no
    // correctness test in this repository would notice.
    let lines = code(&compile("tests/let_no_alias.ysu"));

    let add = lines
        .iter()
        .position(|l| l.starts_with("add.s32"))
        .expect("the control fixture emitted no add");
    let produced = lines[add]
        .split_whitespace()
        .nth(1)
        .unwrap()
        .trim_end_matches(',')
        .to_string();

    // `let doubled: I32 = tid + tid;` -- the add's destination is a fresh
    // temporary that no other binding owns, so the binding takes it directly.
    if let Some(next) = lines.get(add + 1) {
        assert!(
            dst_of(next).is_none() || !next.ends_with(&format!(", {produced};")),
            "a binding whose initialiser produced a FRESH register was copied \
             anyway: `{next}` duplicates {produced}. The copy is for aliased \
             registers only. Emitted:\n{}",
            lines.join("\n")
        );
    }
}

/// The register width a PTX type suffix names, and the width a register's
/// name class implies. `%rd` is checked before `%r` because it starts with it.
fn width_of_suffix(sfx: &str) -> Option<u8> {
    match sfx {
        "u32" | "s32" | "b32" | "f32" => Some(32),
        "u64" | "s64" | "b64" | "f64" => Some(64),
        _ => None,
    }
}

fn width_of_reg(reg: &str) -> Option<u8> {
    if reg.starts_with("%rd") {
        Some(64)
    } else if reg.starts_with("%r") || reg.starts_with("%f") {
        Some(32)
    } else {
        None
    }
}

#[test]
fn the_copy_is_emitted_at_the_width_of_the_register_it_copies() {
    // `tests/let_alias.ysu` is 32-bit throughout, so it cannot see a copy
    // emitted at the wrong width -- and `mov.u32 %rd16, %rd12;` is not a
    // subtly wrong value, it is a module `ptxas` REFUSES. Nothing else in this
    // repository catches that: `ptx_portability` assembles a fixture set with
    // no 64-bit aliased binding in it, which is the standing limit of an
    // assemble gate ("it cannot see a construct no fixture uses").
    //
    // Every grid-stride reduction here widens to I64 before multiplying,
    // because the permitted grid dimensions overflow a signed 32-bit index --
    // so 64 bits is the common case, not the corner one.
    let mut checked = 0usize;
    for fixture in ["tests/let_alias.ysu", "tests/let_alias_wide.ysu"] {
        let ptx = compile(fixture);
        for line in code(&ptx) {
            let Some(rest) = line.strip_prefix("mov.") else {
                continue;
            };
            let Some((sfx, operands)) = rest.split_once(' ') else {
                continue;
            };
            let mut it = operands.trim_end_matches(';').split(',');
            let (Some(dst), Some(src)) = (it.next(), it.next()) else {
                continue;
            };
            let (dst, src) = (dst.trim(), src.trim());
            if !src.starts_with('%') || src.contains('.') {
                continue; // an immediate, or a special register like %tid.x
            }
            let (Some(w), Some(wd), Some(ws)) = (
                width_of_suffix(sfx),
                width_of_reg(dst),
                width_of_reg(src),
            ) else {
                continue;
            };
            assert!(
                w == wd && w == ws,
                "{fixture}: `{line}` copies between {ws}-bit and {wd}-bit registers \
                 with a {w}-bit suffix. ptxas answers \"Arguments mismatch for \
                 instruction 'mov'\" and refuses the whole module."
            );
            checked += 1;
        }
    }
    // Without a floor, a sweep that recognises nothing reports no offenders.
    assert!(
        checked >= 2,
        "only {checked} register-to-register copies were examined across both \
         fixtures; the scan is not reading them"
    );
}

#[test]
fn the_wide_fixture_assembles() {
    // The behavioural half of the width claim: the structural test above
    // reasons about suffixes, this one asks the assembler.
    let ptx = compile("tests/let_alias_wide.ysu");
    let dir = std::env::temp_dir().join(format!(
        "y_let_alias_asm_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("k.ptx");
    std::fs::write(&f, &ptx).unwrap();
    match Command::new("ptxas")
        .args(["-arch=sm_89"])
        .arg(&f)
        .arg("-o")
        .arg(dir.join("k.cubin"))
        .output()
    {
        Ok(out) => assert!(
            out.status.success(),
            "ptxas refused the emitted module:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(_) => eprintln!("SKIP: no ptxas — the wide fixture was not assembled."),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_grid_stride_loop_does_not_destroy_the_thread_index() {
    use y::cuda_runtime::CudaContext;

    let Some(ctx) = CudaContext::new() else {
        eprintln!(
            "SKIP: no CUDA driver — the aliasing bug was not demonstrated on the device. \
             The two source-level tests in this file still ran."
        );
        return;
    };
    let ptx = compile("tests/let_alias.ysu");
    let module = ctx.load_ptx(&ptx, "let_alias").expect("PTX failed to load");

    // Sized so a clobbered index is still INSIDE the allocation: this test
    // must observe a wrong answer, not a fault.
    let slots = (N + 2 * BLOCK) as usize;
    let d_out = ctx.alloc(slots * 4).unwrap();
    ctx.memset_u8(&d_out, 0).unwrap();

    let n_arg = N as i32;
    let args = vec![d_out.device_ptr(), (&n_arg as *const i32) as u64];
    ctx.launch(&module, (1, 1, 1), (BLOCK, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");

    let mut raw = vec![0u8; slots * 4];
    ctx.memcpy_dtoh_at(&mut raw, &d_out, 0).unwrap();
    let got: Vec<i32> = (0..slots)
        .map(|i| i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]))
        .collect();

    // `tid` is assigned nowhere in the fixture, so thread t adds 1 at index t.
    let mut want = vec![0i32; slots];
    for t in 0..BLOCK as usize {
        want[t] = 1;
    }
    assert_eq!(
        got,
        want,
        "the kernel did not write where its source says. Under the aliasing bug \
         the loop's `k = k + bsz` clobbers `tid`, so every thread lands at the \
         first index >= N in its own residue class: indices {}..{} instead of \
         0..{}.",
        N,
        N + BLOCK,
        BLOCK
    );
}
