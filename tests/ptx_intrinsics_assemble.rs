//! Every PTX intrinsic Y advertises must produce PTX that `ptxas` accepts -
//! or must refuse to compile with a diagnostic.
//!
//! # Why this file exists
//!
//! `ptx_emitter.rs` recognises ~55 intrinsic names in `emit_expr`. Not one
//! `.ysu` file in `tests/` called any of the Hopper ones, and every test that
//! covered them asserted that a substring appeared in the output buffer. So
//! two of them - `tma_load` and `wgmma_async` - emitted instructions with the
//! right mnemonic and the wrong operand shape for as long as they had existed:
//!
//! ```text
//! cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd1], [%rd0];
//! wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f0,%f1,%f2,%f3}, %r0, %r1;
//! ```
//!
//! The first needs a host-built tensor map, an mbarrier and per-dimension
//! coordinates; the second needs 64-bit shared-memory matrix descriptors and
//! four immediate operands. `ptxas` rejects both - "Arguments mismatch" - and
//! the `wgmma` line additionally referenced `%f1`..`%f3` when the kernel had
//! declared `.reg .f32 %f<1>`, and required `sm_90a` while the compiler had
//! written `.target sm_89` from the probed hardware profile. Five distinct
//! errors in two instructions, and `Y --emit-ptx` printed "PTX Assembly
//! generated successfully!" and exited 0.
//!
//! Both intrinsics now refuse, which is what this file's second half checks:
//! **refusing is a pass, silently emitting a broken kernel is not.** A gap in
//! the backend that names itself costs a user five minutes. A gap that emits
//! something plausible costs them however long it takes to suspect the
//! compiler.
//!
//! # Why it drives the real binary
//!
//! Calling `PtxEmitter` methods directly cannot catch a register-pool overflow
//! or a wrong `.target`, because both are decided by the module envelope that
//! only the full `emit_program` path writes. The whole class of bug here lived
//! in the gap between "the instruction looks right" and "the module assembles",
//! so the test has to assemble the module.
//!
//! Requires `ptxas`; skipped with a notice otherwise, matching the gate in
//! `tests_paged_decode_attention` and `tests/coprocessor_ptx_assembles.rs`.
//!
//! Run with:  cargo test --test ptx_intrinsics_assemble

use std::path::PathBuf;
use std::process::Command;

fn ptxas_present() -> bool {
    Command::new("ptxas").arg("--version").output().is_ok()
}

/// Compiles a kernel body through the real `Y` binary with `--emit-ptx`.
///
/// Returns `(compiler_output, emitted_ptx_if_any)`.
fn compile(name: &str, body: &str) -> (String, Option<String>) {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("y_intr_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let src = dir.join(format!("{}.ysu", name));
    std::fs::write(
        &src,
        format!(
            "kernel {}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, N: I32) {{\n{}\n}}\n\nfn main() {{\n}}\n",
            name, body
        ),
    )
    .expect("write source");

    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(&repo)
        .output()
        .expect("run Y");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = dir.join(format!("{}.ptx", name));
    let text = std::fs::read_to_string(&ptx).ok();
    (log, text)
}

/// Runs `ptxas` over `ptx`, returning its stderr on failure.
fn assemble(name: &str, ptx: &str) -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("y_intr_{}", std::process::id()));
    let f = dir.join(format!("{}.check.ptx", name));
    std::fs::write(&f, ptx).expect("write ptx");
    let res = Command::new("ptxas")
        .arg("-arch=sm_89")
        .arg(&f)
        .arg("-o")
        .arg("/dev/null")
        .output()
        .expect("run ptxas");
    if res.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&res.stderr).to_string())
    }
}

/// Every intrinsic that is meant to work must assemble for `sm_89`.
///
/// The bodies are deliberately minimal - one call, plus whatever operands it
/// needs. This is a syntax and register-allocation gate, not a semantics test:
/// it answers "can a GPU load this", which is the question no test in this
/// repo was asking of the intrinsic surface.
#[test]
fn supported_intrinsics_emit_assemblable_ptx() {
    if !ptxas_present() {
        eprintln!("skipping: ptxas not on PATH");
        return;
    }

    // (probe name, kernel body). Each exercises one dispatch arm of `emit_expr`.
    let cases: &[(&str, &str)] = &[
        ("i_thread_idx", "let t: I32 = thread_idx_x();\n    store(A, t);"),
        ("i_thread_idx_y", "let t: I32 = thread_idx_y();\n    store(A, t);"),
        ("i_thread_idx_z", "let t: I32 = thread_idx_z();\n    store(A, t);"),
        ("i_block_idx", "let b: I32 = block_idx_x();\n    store(A, b);"),
        ("i_block_idx_y", "let b: I32 = block_idx_y();\n    store(A, b);"),
        ("i_block_idx_z", "let b: I32 = block_idx_z();\n    store(A, b);"),
        ("i_block_dim", "let d: I32 = block_dim_x();\n    store(A, d);"),
        ("i_block_dim_y", "let d: I32 = block_dim_y();\n    store(A, d);"),
        ("i_block_dim_z", "let d: I32 = block_dim_z();\n    store(A, d);"),
        ("i_global_thread_id", "let g: I32 = global_thread_id();\n    store(A, g);"),
        ("i_barrier_sync", "barrier_sync();"),
        ("i_membar", "membar();"),
        ("i_store", "let t: I32 = thread_idx_x();\n    store(A, t);"),
        ("i_load_v4", "let v: F32 = load_v4(A);"),
        ("i_store_v4", "store_v4(A, B);"),
        ("i_ld_global_v4", "let v: F32 = ld_global_v4_f32(A);"),
        ("i_st_global_v4", "st_global_v4_f32(A, B);"),
        ("i_warp_reduce_sum", "let x: F32 = 1.0;\n    let v: F32 = warp_reduce_sum(x);"),
        ("i_warp_reduce_max", "let x: F32 = 1.0;\n    let v: F32 = warp_reduce_max(x);"),
        ("i_shfl_bfly", "let x: F32 = 1.0;\n    let v: F32 = shfl_sync_bfly(x, 16);"),
        ("i_shfl_bfly_b32", "let v: I32 = shfl_sync_bfly_b32(N, 16);"),
        // `cp_async` returns a linear token `linear_tracker` requires be consumed
        // exactly once, so the probe has to bind and await it - which is also
        // what makes this case cover `pipe.wait`.
        ("i_cp_async", "let tok: AsyncToken = cp_async(A, B, 16);\n    pipe.wait(tok);"),
        ("i_block_cdiv", "let c: I32 = block_cdiv(N, 128);"),
        ("i_block_arange", "let r: I32 = block_arange(0, 128);"),
        ("i_tile_load", "let v: F32 = tile_load(A, N, 128);"),
        ("i_tile_store", "tile_store(A, N, B);"),
        ("i_block_tile_load", "let v: F32 = block_tile_load(A, N, 128);"),
        ("i_block_tile_store", "block_tile_store(A, N, B);"),
        ("i_vec_add_v4", "vec_add_v4(A, B, A);"),
        ("i_swiglu_v4", "swiglu_v4(A, B, A);"),
        ("i_rmsnorm_v4", "rmsnorm_v4(A, B, A);"),
        ("i_mbarrier_init", "mbarrier_init(A, 128);"),
        ("i_mbarrier_arrive", "mbarrier_arrive(A);"),
    ];

    let mut passed = 0;
    let mut failures: Vec<String> = Vec::new();
    for (name, body) in cases {
        let (log, ptx) = compile(name, body);
        let Some(ptx) = ptx else {
            failures.push(format!("{}: no PTX was written.\n{}", name, log));
            continue;
        };
        match assemble(name, &ptx) {
            Ok(()) => passed += 1,
            Err(e) => failures.push(format!(
                "{}: emitted PTX does not assemble.\n{}\n--- emitted ---\n{}",
                name,
                e.lines().take(4).collect::<Vec<_>>().join("\n"),
                ptx
            )),
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} intrinsics emitted PTX ptxas rejects:\n\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n\n")
    );
    eprintln!("{} intrinsics assembled for sm_89", passed);
}

/// An intrinsic the backend cannot lower must fail the build, not emit a file.
///
/// This is the half that would have caught the original bug. Both of these
/// used to compile clean and produce a `.ptx` that `ptxas` refuses, so the
/// assertion is threefold: non-zero exit, no output file, and a diagnostic
/// that names the intrinsic. Reporting success is the specific thing being
/// ruled out - a backend is allowed to lack a feature, and is not allowed to
/// pretend it has one.
#[test]
fn unlowerable_intrinsics_fail_the_build() {
    for (name, body, intrinsic) in [
        ("u_tma_load", "tma_load(A, B);", "tma_load"),
        ("u_tma_load_2d", "tma_load_2d(A, B);", "tma_load_2d"),
        ("u_cp_async_bulk", "cp_async_bulk(A, B);", "cp_async_bulk"),
        ("u_wgmma_async", "wgmma_async();", "wgmma_async"),
        ("u_wgmma_mma_async", "wgmma_mma_async();", "wgmma_mma_async"),
    ] {
        let (log, ptx) = compile(name, body);
        assert!(
            !log.contains("generated successfully"),
            "{} reported success for an intrinsic the backend cannot lower:\n{}",
            intrinsic,
            log
        );
        assert!(
            ptx.is_none(),
            "{} wrote a .ptx file despite being unlowerable - the file cannot \
             be assembled, so writing it only defers the error to the user:\n{}",
            intrinsic,
            log
        );
        assert!(
            log.contains(intrinsic),
            "the diagnostic for {} must name the intrinsic so the user knows \
             which call to remove:\n{}",
            intrinsic,
            log
        );
    }
}

/// The await that `linear_tracker` forces you to write must reach the PTX.
///
/// Assembling is not enough to catch this one: the bug was that `pipe.wait(t)`
/// emitted *nothing*, and a kernel missing an instruction assembles perfectly.
/// So this checks for the three instructions by name, and for their order.
///
/// What made it invisible is worth recording. `pipe.wait` without an argument
/// list parses as `Expr::MemberAccess` and was handled; `pipe.wait(token)`
/// parses as an `Expr::Call` whose callee is a `MemberAccess`, and the `Call`
/// arm matched only `Ident` and `Path` callees before falling through to
/// `_ => "".into()`. Since `linear_tracker` *requires* the token be passed to
/// something, the only spelling a user can compile was the one that silently
/// did nothing - the front end proved the copy was awaited exactly once and the
/// back end then dropped the await, leaving a race that reads as intermittently
/// wrong output rather than as a compiler bug.
///
/// `cp.async.commit_group` matters just as much as the wait: `wait_group 0`
/// waits for all but zero of the *committed* groups, so with nothing committed
/// it is satisfied immediately and the await is a no-op even when present.
#[test]
fn async_copy_is_committed_and_awaited() {
    let (log, ptx) = compile(
        "async_await",
        "let tok: AsyncToken = cp_async(A, B, 16);\n    pipe.wait(tok);",
    );
    let ptx = ptx.unwrap_or_else(|| panic!("no PTX emitted:\n{}", log));

    let copy = ptx.find("cp.async.cg.shared.global");
    let commit = ptx.find("cp.async.commit_group");
    let wait = ptx.find("cp.async.wait_group");

    assert!(copy.is_some(), "no cp.async copy was emitted:\n{}", ptx);
    assert!(
        commit.is_some(),
        "cp_async emitted no commit_group, so the wait_group below it waits on \
         an empty set of groups and returns immediately:\n{}",
        ptx
    );
    assert!(
        wait.is_some(),
        "pipe.wait(token) emitted no cp.async.wait_group - the await the linear \
         tracker requires was dropped by the backend:\n{}",
        ptx
    );
    assert!(
        copy < commit && commit < wait,
        "the copy must be committed before it is awaited:\n{}",
        ptx
    );
}

/// cp.async encodes its size as an immediate, so a bad one must be refused.
///
/// The byte count used to be hardcoded to 16 with the third argument thrown
/// away, which meant `cp_async(dst, src, 4)` copied 16 bytes - a 12-byte
/// overrun of whatever the user had sized their shared tile to.
#[test]
fn cp_async_honours_its_byte_count_and_rejects_illegal_ones() {
    for (n, want) in [(4u32, "], 4;"), (8, "], 8;"), (16, "], 16;")] {
        let (log, ptx) = compile(
            &format!("async_size_{}", n),
            &format!(
                "let tok: AsyncToken = cp_async(A, B, {});\n    pipe.wait(tok);",
                n
            ),
        );
        let ptx = ptx.unwrap_or_else(|| panic!("no PTX for {} bytes:\n{}", n, log));
        assert!(
            ptx.contains(want),
            "cp_async({}) did not emit a {}-byte transfer:\n{}",
            n,
            n,
            ptx
        );
    }

    // 12 is not an encodable cp.async width. Rounding it to 16 would silently
    // overrun; rounding it to 8 would silently truncate. Refuse.
    let (log, ptx) = compile(
        "async_size_bad",
        "let tok: AsyncToken = cp_async(A, B, 12);\n    pipe.wait(tok);",
    );
    assert!(
        ptx.is_none() && !log.contains("generated successfully"),
        "cp_async with an illegal 12-byte width must fail the build:\n{}",
        log
    );
}
