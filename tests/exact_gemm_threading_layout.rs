//! The threaded wrapper's job record, read back at the offsets it was written
//! at - and every buffer zeroed for exactly as many bytes as it was allocated.
//!
//! `proofs/ExactGemmThreading.v` proves the ARITHMETIC of the threading
//! layer: the job array is a `(thread, field)` positional index so no worker's
//! record can overlap another's, each worker's private C buffer is exactly the
//! rectangle the reduction reads back, and the emitted `reduce.head` loop is
//! `ExactGemmKsplit.acc_bands`. None of that says which OFFSET each field is
//! written and read at, because that is a property of the emitted text rather
//! than of any number. This file is that half.
//!
//! **What a correctness test cannot see here.** `tests/exact_gemm_thread_
//! invariance.rs` runs the real kernel at ragged shapes and compares answers
//! bit for bit across thread counts. It cannot see:
//!
//! - an OVER-sized job record, which wastes bytes and changes no answer;
//! - a field written at one offset and read at another, when the field
//!   happens not to matter for the shapes under test;
//! - a SHORT `memset`, which is the sharpest of the three. A large `malloc`
//!   comes from a fresh `mmap` and is already zero, so a buffer zeroed one
//!   byte short gives the right answer on every run of a fresh process. The
//!   per-thread buffers here are freed and re-allocated on the next call, so
//!   the tail comes back holding whatever the allocator recycled - a wrong
//!   answer that appears only under a workload nobody benchmarks.
//!
//! The two consistent-renaming cases are deliberately NOT claimed: shifting
//! every offset by one slot, or permuting writer and reader together, is a
//! relabelling and a relabelling is not a bug. What is checked is that the
//! three readers agree with the one writer, and that the record is exactly
//! `JOB_SLOTS` slots wide - which is what makes a thirteenth field a compile
//! -time move rather than a store into the next worker's record.

use std::collections::{BTreeMap, BTreeSet};
use y::cpu_gemm::{emit_vnni_threaded_module, job_bytes, JOB_SLOTS};

/// Every `getelementptr i8, ptr <base>, i64 <off>` in the module, as
/// `result register -> byte offset`.
///
/// The three roles have three distinct base registers - `%j` for the spawn
/// loop that writes the record, `%arg` for the worker that reads it, `%rj` for
/// the reduction that reads back the pointers it must free - so no block
/// tracking is needed to tell them apart.
fn gep_offsets(module: &str, base: &str) -> BTreeMap<String, i64> {
    let needle = format!("= getelementptr i8, ptr {base}, i64 ");
    let mut out = BTreeMap::new();
    for line in module.lines() {
        let line = line.trim();
        let Some(pos) = line.find(&needle) else {
            continue;
        };
        let reg = line[..pos].trim().to_string();
        let off: i64 = line[pos + needle.len()..]
            .trim()
            .parse()
            .unwrap_or_else(|e| panic!("offset in `{line}`: {e}"));
        assert!(
            out.insert(reg.clone(), off).is_none(),
            "{reg} is computed twice; the emitted module is not SSA"
        );
    }
    out
}

/// `offset -> LLVM type` for every `<verb>` whose pointer operand is one of
/// `geps`' registers. `verb` is `store` or `load`.
///
/// The TYPE is recorded as well as the offset because a set of offsets cannot
/// see a permutation, and half of the permutations here cross a type boundary:
/// the record is a mix of pointers and `i64` extents, so reading `%lda` where
/// `%C` was stored dereferences an integer. The other half - swapping two
/// pointer fields, say `Ap` and `Bp` - is invisible to any text gate and is
/// caught by the correctness suites instead, because the two packers then
/// write into each other's buffers.
fn accessed(module: &str, geps: &BTreeMap<String, i64>, verb: &str) -> BTreeMap<i64, String> {
    let mut out = BTreeMap::new();
    for line in module.lines() {
        let line = line.trim();
        // `store <ty> <val>, ptr %sN, align 8` / `%x = load <ty>, ptr %pN, align 8`
        let ty = match verb {
            "store" => line.strip_prefix("store ").and_then(|r| {
                r.split_whitespace().next().map(str::to_string)
            }),
            "load" => line.split(" = load ").nth(1).and_then(|r| {
                r.split(',').next().map(|t| t.trim().to_string())
            }),
            other => panic!("unknown verb {other}"),
        };
        let Some(ty) = ty else { continue };
        for (reg, off) in geps {
            // `, ptr %s1,` must not match `%s10`.
            if line.contains(&format!(", ptr {reg},")) {
                out.insert(*off, ty.clone());
            }
        }
    }
    out
}

/// Just the offsets, for the assertions that are about coverage rather than
/// about what each slot holds.
fn offsets(m: &BTreeMap<i64, String>) -> BTreeSet<i64> {
    m.keys().copied().collect()
}

/// The record is exactly `JOB_SLOTS` eight-byte slots, and every one of them
/// is used.
///
/// Both directions matter and only one of them is a memory-safety claim. A
/// slot at or past the stride is a store into the NEXT worker's record - and
/// past the end of the array entirely for the last worker, which is a heap
/// overflow. A slot the writer never uses is the over-allocation case: nothing
/// crashes, nothing is wrong, and nothing in this repository would notice.
#[test]
fn the_job_record_is_exactly_the_slots_the_writer_uses() {
    let m = emit_vnni_threaded_module(true);
    let writes = offsets(&accessed(&m, &gep_offsets(&m, "%j"), "store"));

    assert_eq!(
        writes.len(),
        JOB_SLOTS,
        "the spawn loop writes {} fields into a {JOB_SLOTS}-slot record: {writes:?}",
        writes.len()
    );
    let expected: BTreeSet<i64> = (0..JOB_SLOTS as i64).map(|k| k * 8).collect();
    assert_eq!(
        writes, expected,
        "the record's slots are not 8-byte aligned and contiguous"
    );
    let last = *writes.iter().next_back().expect("a non-empty record");
    assert_eq!(
        last + 8,
        job_bytes() as i64,
        "the last slot written ends at {}, and the record stride is {} - a gap \
         is bytes nobody can see, and an overhang is the next worker's record",
        last + 8,
        job_bytes()
    );
}

/// The worker reads every field at the offset the spawn loop wrote it to.
///
/// A mismatch is not a subtle one: the record is a mix of pointers and `i64`
/// extents, so reading `%lda` where `%C` was stored dereferences an integer.
/// It would be caught by a correctness test the moment it mattered - and only
/// for the fields that matter at the shapes under test.
#[test]
fn the_worker_reads_the_record_the_spawn_loop_wrote() {
    let m = emit_vnni_threaded_module(true);
    let writes = accessed(&m, &gep_offsets(&m, "%j"), "store");
    let reads = accessed(&m, &gep_offsets(&m, "%arg"), "load");
    assert_eq!(
        offsets(&writes),
        offsets(&reads),
        "the spawn loop writes {:?} and the worker reads {:?}",
        offsets(&writes),
        offsets(&reads)
    );
    // ...and at each offset the two agree on what is stored there. This is the
    // half of a permutation a text gate can see: a swap that crosses the
    // pointer / `i64` boundary shows up here, a swap of two same-typed fields
    // does not and cannot.
    for (off, ty) in &writes {
        assert_eq!(
            reads.get(off),
            Some(ty),
            "slot {off} is stored as `{ty}` and read as `{:?}`",
            reads.get(off)
        );
    }
}

/// The reduction reads back only fields the spawn loop wrote - and the four it
/// needs: the private C pointer it accumulates from, and the three buffers it
/// frees.
///
/// Freeing a pointer loaded from an offset nothing stored a pointer to is heap
/// corruption with no wrong answer in front of it.
#[test]
fn the_reduction_frees_pointers_the_spawn_loop_stored() {
    let m = emit_vnni_threaded_module(true);
    let writes = accessed(&m, &gep_offsets(&m, "%j"), "store");
    let reads = accessed(&m, &gep_offsets(&m, "%rj"), "load");

    assert!(!reads.is_empty(), "the reduction reads no field of the record");
    for (off, ty) in &reads {
        assert_eq!(
            writes.get(off),
            Some(ty),
            "the reduction loads offset {off} as `{ty}`, and the spawn loop \
             stores `{:?}` there",
            writes.get(off)
        );
        assert_eq!(ty, "ptr", "the reduction only ever reads pointers back");
    }
    // Slot 2 is `C`, the private buffer. The other three are `Ap`, `Bp`, `Ct`.
    for slot in [2usize, 9, 10, 11] {
        assert!(
            reads.contains_key(&((slot * 8) as i64)),
            "the reduction never loads slot {slot}, so that buffer is leaked"
        );
    }
}

/// The worker's output stride is `%N`, not the caller's `%ldc`.
///
/// This is the exact line whose other reading was a heap overflow: the private
/// C buffer is sized `M * N * 8`, so handing the worker `%ldc` made it write
/// `(M-1)*(ldc-N)` elements past the end - observed as `double free or
/// corruption`, and invisible at every call site in the compiler because they
/// all pass `ldc == N`.
#[test]
fn the_worker_is_handed_the_compact_stride_for_its_private_buffer() {
    let m = emit_vnni_threaded_module(true);
    let geps = gep_offsets(&m, "%j");
    let stride_reg = geps
        .iter()
        .find(|(_, off)| **off == 64)
        .map(|(r, _)| r.clone())
        .expect("no slot at offset 64");

    let store = m
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("store ") && l.contains(&format!(", ptr {stride_reg},")))
        .unwrap_or_else(|| panic!("nothing is stored to {stride_reg}"));

    assert!(
        store.starts_with("store i64 %N,"),
        "the worker's output stride slot is `{store}`; it must be `%N`, because \
         the buffer at slot 2 is sized M*N and the worker writes at this stride"
    );
}

/// Every allocation that is zeroed is zeroed for exactly the byte count it was
/// allocated with.
///
/// The sharp case in this module is the per-thread private C buffer. A fresh
/// `mmap` is already zero, so a short `memset` is right on the first call of a
/// process and wrong on a later one, once the allocator has recycled the block
/// the previous band freed. No answer-comparing test can be relied on to see
/// that, and none here runs two GEMMs in one process at a size that would.
#[test]
fn every_zeroed_buffer_is_zeroed_for_its_whole_allocation() {
    let m = emit_vnni_threaded_module(true);

    let mut alloc: BTreeMap<String, String> = BTreeMap::new();
    for line in m.lines().map(str::trim) {
        let needle = " = call ptr @__y_gemm_exact_alloc(i64 ";
        let Some(pos) = line.find(needle) else {
            continue;
        };
        let reg = line[..pos].trim().to_string();
        let size = line[pos + needle.len()..].trim_end_matches(')').trim().to_string();
        alloc.insert(reg, size);
    }
    assert!(
        alloc.len() >= 9,
        "expected the threaded wrapper's nine allocations, found {}",
        alloc.len()
    );

    let mut zeroed = BTreeSet::new();
    for line in m.lines().map(str::trim) {
        let needle = "call void @llvm.memset.p0.i64(ptr ";
        let Some(pos) = line.find(needle) else {
            continue;
        };
        let rest = &line[pos + needle.len()..];
        let mut parts = rest.split(", ");
        let ptr = parts.next().expect("memset destination").trim().to_string();
        let _fill = parts.next();
        let len = parts
            .next()
            .expect("memset length")
            .trim()
            .trim_start_matches("i64 ")
            .trim()
            .to_string();
        let Some(size) = alloc.get(&ptr) else {
            // A `memset` of something this wrapper did not allocate: the
            // caller's C, zeroed a row at a time at its own stride.
            continue;
        };
        assert_eq!(
            &len, size,
            "{ptr} is allocated with {size} bytes and zeroed for {len}"
        );
        zeroed.insert(ptr);
    }

    // The two that are deliberately NOT zeroed, with the reason: every slot of
    // each is written before anything reads it. `%tids` matters most - the
    // join loop uses `0` as its "never started" sentinel, so a slot left
    // uninitialised would be read as a thread id.
    let unzeroed: Vec<&String> = alloc.keys().filter(|k| !zeroed.contains(*k)).collect();
    assert_eq!(
        unzeroed,
        vec![&"%jobs".to_string(), &"%tids".to_string()],
        "an allocation is neither zeroed nor fully written before it is read"
    );
}

/// The floor. Every assertion above is stated over a set the parser produced,
/// and a parser that produces nothing satisfies most of them - `writes` empty
/// would fail the count, but `reads` empty against `writes` empty would not.
#[test]
fn the_parser_actually_found_the_record() {
    let m = emit_vnni_threaded_module(true);
    assert!(
        gep_offsets(&m, "%j").len() >= JOB_SLOTS,
        "no job-record offsets recovered; the emitted module changed shape"
    );
    assert!(
        gep_offsets(&m, "%arg").len() >= JOB_SLOTS,
        "no worker offsets recovered"
    );
    assert!(
        !gep_offsets(&m, "%rj").is_empty(),
        "no reduction offsets recovered"
    );
    assert_eq!(job_bytes(), JOB_SLOTS * 8);
}
