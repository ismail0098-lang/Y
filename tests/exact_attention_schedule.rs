//! The exact-attention kernel's grid-stride reduction is the one
//! `proofs/GridStrideSplit.v` describes.
//!
//! `src/exact_attention.rs` claims in its module header that "the answer does
//! not depend on `blockDim.x`, `gridDim.x`, `gridDim.z`, or the order the
//! atomics land". `tests/gpu_attention_invariance.rs` demonstrates it on the
//! device at nine launch geometries — genuinely, on a real card, not a silent
//! skip. Nine geometries is not every geometry, and **"the order the atomics
//! land" is not a geometry at all**: it is a property of a race, and no test
//! can enumerate it. `GridStrideSplit.v` proves both halves; this file is what
//! ties the proof to the emitted PTX.
//!
//! **The obligation is different from the two GEMM kernels'**, in two ways that
//! the proof file spells out and this file pins:
//!
//!   - it is a residue-class partition, not an interval split — worker `w` of
//!     `n` takes `{ i < S : i mod n = w }`;
//!   - the partials are combined by `red.shared.add.u64` / `red.global.add.u64`
//!     in **arbitrary order**, so the argument needs commutativity and not just
//!     the associativity both GEMM proofs rely on.
//!
//! **The precondition the partition theorem needs is `worker ∈ [0, nworkers)`,
//! and it is a claim about two separate expressions agreeing.** `worker` mixes
//! three indices — `ctaid.z`, `ctaid.x`, `tid.x` — with two radices, and
//! `nworkers` must be the product of those three indices' EXTENTS. Drop
//! `%nctaid.z` from the product and workers alias: some `i` counted twice,
//! others never. `the_worker_index_and_the_worker_count_use_paired_registers`
//! is that check, and it is the one assertion here that is not just a
//! transcription of the template.
//!
//! **This tie is weaker than the GEMM kernels' and is named as such.** Those
//! render their schedule from an `Ix` shared with the proof generator, so a
//! divergence is a byte-identity failure. This kernel is a PTX string template,
//! so the best available check is reading the emitted text. Making it an `Ix`
//! would mean routing the attention kernel through `IrBuilder`, which is a
//! larger change than this file.
//!
//! Run with:  cargo test --release --test exact_attention_schedule

use y::exact_attention::{attention_ptx, MAX_EXACT_SEQ_LEN};

const HEAD_DIM: usize = 64;
const SEQ_LEN: usize = 512;

/// The marker `src/exact_attention.rs` writes above the sequence reduction.
///
/// `attn_accum` contains a SECOND grid-stride loop, over `d`, that zeroes the
/// CTA's shared accumulators - identical in shape to the sequence loop and
/// distinguished from it by nothing structural. The first version of this file
/// told them apart by their BOUND, which meant the test only worked because the
/// fixture happened to use `head_dim != seq_len`: the `lda == K` coincidence,
/// one layer over. The kernel names the loop now, the way this backend already
/// names `[Y PAGED DECODE ATTENTION]` and `[Y ZERO DRIFT]`, so nothing here
/// depends on the shape chosen.
const SEQ_MARKER: &str = "[Y SEQUENCE REDUCTION]";

fn ptx() -> String {
    attention_ptx(HEAD_DIM, SEQ_LEN).expect("a shape well inside the exactness argument")
}

/// The accumulating entry points. `attn_scores` is one thread per key and
/// reduces nothing, so it is deliberately not here.
const ACCUM_ENTRIES: [&str; 2] = ["attn_accum", "attn_accum_naive"];

/// The body of `.visible .entry <name>`, up to the next entry.
fn entry_body(ptx: &str, name: &str) -> String {
    let head = format!(".visible .entry {name}(");
    let start = ptx.find(&head).unwrap_or_else(|| panic!("no entry `{name}` in the emitted PTX"));
    let rest = &ptx[start + head.len()..];
    let end = rest.find(".visible .entry ").unwrap_or(rest.len());
    rest[..end].to_string()
}

/// Instructions, comments stripped - except the sequence-reduction marker,
/// which is kept as a line of its own so the loop can be located structurally.
fn norm(s: &str) -> Vec<String> {
    s.lines()
        .filter_map(|l| {
            if l.contains(SEQ_MARKER) {
                return Some(SEQ_MARKER.to_string());
            }
            let code = l.split("//").next().unwrap_or("").trim();
            (!code.is_empty()).then(|| code.to_string())
        })
        .collect()
}

/// Which special registers a virtual register depends on, by walking the
/// entry's `mov` / `mad.lo.s32` / `mul.lo.s32` definitions backwards.
///
/// **Last definition wins**, because PTX virtual registers are not SSA here:
/// `%r9` is written twice while the worker count is built up, and the loop
/// reads the second one.
///
/// This is a derivation rather than a transcription, which is the point. The
/// two accumulating entries build the SAME decomposition in a different
/// instruction ORDER - `attn_accum` hoists `%ntid.x` and `%tid.x` above the
/// shared-memory zeroing loop, `attn_accum_naive` does not - so a literal
/// window matches at most one of them, and asserting one window per entry
/// would be asserting the template back to itself.
fn deps(body: &[String], reg: &str) -> Vec<String> {
    /// The last definition of `reg` strictly before `bound`, and its index.
    fn def_of(body: &[String], reg: &str, bound: usize) -> Option<(usize, String)> {
        body[..bound].iter().enumerate().rev().find_map(|(i, l)| {
            let (_, r) = l.split_once(' ')?;
            (r.split(',').next()?.trim() == reg).then(|| (i, l.clone()))
        })
    }
    fn walk(body: &[String], reg: &str, bound: usize, depth: usize, out: &mut Vec<String>) {
        assert!(depth < 64, "register chain reaching {reg} is too deep");
        if reg.contains('.') {
            // A special register (`%tid.x`, `%nctaid.z`, ...) is a leaf.
            out.push(reg.to_string());
            return;
        }
        if !reg.starts_with("%r") || reg.starts_with("%rd") {
            return;
        }
        let Some((i, def)) = def_of(body, reg, bound) else { return };
        let (op, rest) = def.split_once(' ').unwrap();
        let ops: Vec<&str> = rest.trim_end_matches(';').split(',').map(str::trim).collect();
        if !matches!(op, "mov.u32" | "mad.lo.s32" | "mul.lo.s32" | "add.s32") {
            return;
        }
        // **Operands resolve at the definition's own position, not at the use.**
        // `%r9 = %r9 * %r5` is a real instruction here - the worker count is
        // built in two steps - and resolving its first operand at the use finds
        // the same line again.
        for o in &ops[1..] {
            walk(body, o, i, depth + 1, out);
        }
    }
    let mut out = Vec::new();
    walk(body, reg, body.len(), 0, &mut out);
    out.sort();
    out.dedup();
    out
}

/// The SEQUENCE loop's counter and stride, located by the kernel's own marker.
///
/// Nothing here names a register number or a bound: the marker says which of
/// the entry's grid-stride loops is the reduction, and the loop's own update
/// says which registers it uses.
fn loop_counter_and_stride(body: &[String]) -> (String, String) {
    let at = body.iter().position(|l| l == SEQ_MARKER).unwrap_or_else(|| {
        panic!(
            "no `{SEQ_MARKER}` in this entry. The kernel must say which of its \
             grid-stride loops is the reduction over the sequence - `attn_accum` \
             has two of identical shape, and picking the wrong one reports the \
             decomposition as depending on `tid.x` and nothing else"
        )
    });
    let upd = body[at..]
        .iter()
        .find(|l| {
            l.starts_with("add.s32 ") && {
                let o: Vec<&str> =
                    l[8..].trim_end_matches(';').split(',').map(str::trim).collect();
                o.len() == 3 && o[0] == o[1] && o[2].starts_with("%r")
            }
        })
        .expect("the marked sequence loop never advances");
    let o: Vec<&str> = upd[8..].trim_end_matches(';').split(',').map(str::trim).collect();
    (o[0].to_string(), o[2].to_string())
}

/// **The precondition the partition theorem needs, and the one assertion here
/// that is not a transcription.**
///
/// `worker` is a mixed-radix index over `(ctaid.z, ctaid.x, tid.x)`;
/// `GridStrideSplit.stride_classes_partition` needs it to range over exactly
/// `[0, nworkers)`, so `nworkers` must be the product of those three indices'
/// EXTENTS. Drop `%nctaid.z` from the product and the stride is smaller than
/// the number of workers: the residue classes overlap, some keys are counted
/// twice and others never. Only a launch with `gridDim.z > 1` can see that on a
/// device, and only when there is a device.
#[test]
fn the_worker_index_and_the_worker_count_use_paired_registers() {
    let ptx = ptx();
    let pairs = [("%ctaid.z", "%nctaid.z"), ("%ctaid.x", "%nctaid.x"), ("%tid.x", "%ntid.x")];

    for name in ACCUM_ENTRIES {
        let body = norm(&entry_body(&ptx, name));
        let (counter, stride) = loop_counter_and_stride(&body);

        // The counter is seeded from the worker index.
        let seed = body
            .iter()
            .find(|l| l.starts_with(&format!("mov.u32 {counter},")))
            .unwrap_or_else(|| panic!("{name}: the loop counter {counter} is never seeded"));
        let worker = seed.split(',').nth(1).unwrap().trim().trim_end_matches(';').to_string();

        let wdeps = deps(&body, &worker);
        let sdeps = deps(&body, &stride);

        for (idx, extent) in pairs {
            assert!(
                wdeps.iter().any(|d| d == idx),
                "{name}: the worker index depends on {wdeps:?}, which does not \
                 include {idx} - this is not the decomposition \
                 GridStrideSplit.v proves"
            );
            assert!(
                sdeps.iter().any(|d| d == extent),
                "{name}: the worker index uses {idx} but the worker count \
                 depends on {sdeps:?}, which never mentions its extent \
                 {extent}. The workers then alias: `worker` ranges past \
                 `nworkers`, the residue classes overlap, and some keys are \
                 counted twice while others are missed"
            );
        }

        // ...and nothing else. An extra factor in the stride is the mirror
        // failure: the classes no longer cover the sequence.
        let mut want: Vec<String> = pairs.iter().map(|(_, e)| e.to_string()).collect();
        want.sort();
        assert_eq!(
            sdeps, want,
            "{name}: the worker count is a product over {sdeps:?}, not over the \
             extents of the three indices the worker index actually uses"
        );
    }
}

/// The loop advances by the worker count, and does so exactly once - the
/// `i += nworkers` that makes the visited set a residue class rather than an
/// arbitrary subset.
#[test]
fn the_stride_is_the_worker_count() {
    let ptx = ptx();
    for name in ACCUM_ENTRIES {
        let body = norm(&entry_body(&ptx, name));
        let (counter, _) = loop_counter_and_stride(&body);
        let n = body
            .iter()
            .filter(|l| l.starts_with(&format!("add.s32 {counter}, {counter},")))
            .count();
        assert_eq!(n, 1, "{name}: expected exactly one stride update, found {n}");
    }
}

/// A control on the two tests above: they must be looking at a real
/// decomposition in BOTH entries, not silently at one.
///
/// `attn_scores` is deliberately excluded - it is one thread per key and
/// reduces nothing, so it has no stride loop to check.
#[test]
fn both_accumulating_entries_were_actually_examined() {
    let ptx = ptx();
    for name in ACCUM_ENTRIES {
        let body = norm(&entry_body(&ptx, name));
        assert!(body.len() > 40, "{name}: the entry body is too short to be the kernel");
        let (counter, stride) = loop_counter_and_stride(&body);
        assert_ne!(counter, stride, "{name}: the loop strides by itself");
        assert_eq!(deps(&body, &stride).len(), 3, "{name}: the worker count is not a triple product");
    }
    assert!(
        !entry_body(&ptx, "attn_scores").is_empty(),
        "attn_scores vanished; the exclusion above would then be hiding it"
    );
}

/// The proof's accumulator bound is the compiler's.
///
/// Parsed out of the `.v` rather than restated here — a third copy of
/// `2^63 / ((2^28 - 1) * 127)` is the defect, not the check.
#[test]
fn the_proof_and_the_compiler_agree_on_the_accumulator_bound() {
    let v = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("proofs/GridStrideSplit.v"),
    )
    .expect("proofs/GridStrideSplit.v");
    let line = v
        .lines()
        .find(|l| l.starts_with("Definition MAX_EXACT_SEQ_LEN"))
        .expect("GridStrideSplit.v no longer defines MAX_EXACT_SEQ_LEN");
    let nums: Vec<u128> = line
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    // `(2 ^ 63) / ((2 ^ 28 - 1) * 127)`
    assert_eq!(nums, vec![2, 63, 2, 28, 1, 127], "the bound's shape moved: {line}");
    let from_proof = (1u128 << nums[1]) / (((1u128 << nums[3]) - nums[4]) * nums[5]);
    assert_eq!(
        from_proof as usize, MAX_EXACT_SEQ_LEN,
        "proofs/GridStrideSplit.v and src/exact_attention.rs disagree on where \
         the 64-bit accumulator wraps. Past it the sum is a WRONG answer rather \
         than an imprecise one, and it is untestable on a device — it needs \
         2.7e8 keys — so this agreement is the only thing checking it"
    );
}

/// The marker is load-bearing, not decoration: `attn_accum` really does contain
/// more than one grid-stride loop, and both accumulating entries carry the
/// marker.
///
/// A marker present in one of two reductions is worse than none - the test
/// would silently examine one entry and skip the other.
#[test]
fn the_sequence_marker_is_doing_work() {
    let ptx = ptx();
    for name in ACCUM_ENTRIES {
        let body = norm(&entry_body(&ptx, name));
        let markers = body.iter().filter(|l| *l == SEQ_MARKER).count();
        assert_eq!(markers, 1, "`{name}` carries {markers} sequence markers, not 1");
    }

    // The hazard the marker exists for: two loops of identical shape.
    let accum = norm(&entry_body(&ptx, "attn_accum"));
    let strides = accum
        .iter()
        .filter(|l| {
            l.starts_with("add.s32 ") && {
                let o: Vec<&str> =
                    l[8..].trim_end_matches(';').split(',').map(str::trim).collect();
                o.len() == 3 && o[0] == o[1] && o[2].starts_with("%r")
            }
        })
        .count();
    assert!(
        strides >= 2,
        "`attn_accum` now has only {strides} grid-stride loop(s). If the \
         shared-memory zeroing loop is gone the marker no longer disambiguates \
         anything, and this file should say so rather than keep implying it does"
    );
}

/// `deps` must resolve an operand at its DEFINITION's position, not at the use.
///
/// The kernel used to build its worker count as `%r9 = %r8 * %r2` then
/// `%r9 = %r9 * %r5` - legal PTX, and a "last definition wins" walk resolves
/// that second instruction's first operand to itself. `src/exact_attention.rs`
/// no longer reuses the register, **so the real kernel stops exercising this
/// path** - which is exactly when a capability quietly rots. Pinned here on a
/// synthetic body instead, because hand-written PTX elsewhere in this repo is
/// full of non-SSA reuse and the next caller of `deps` will meet it.
#[test]
fn the_dependency_walk_handles_a_redefined_register() {
    let body: Vec<String> = [
        "mov.u32 %r1, %ctaid.z;",
        "mov.u32 %r2, %nctaid.x;",
        "mov.u32 %r5, %ntid.x;",
        "mul.lo.s32 %r9, %r1, %r2;",
        "mul.lo.s32 %r9, %r9, %r5;", // reads its own previous definition
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    assert_eq!(
        deps(&body, "%r9"),
        vec!["%ctaid.z".to_string(), "%nctaid.x".to_string(), "%ntid.x".to_string()],
        "the walk lost a factor across a redefinition - resolving `%r9`'s \
         operand at the USE finds the same instruction again and the chain \
         either cycles or truncates"
    );

    // The control: without the reuse the answer must be the same, or the test
    // above is passing for a reason unrelated to redefinition.
    let ssa: Vec<String> = [
        "mov.u32 %r1, %ctaid.z;",
        "mov.u32 %r2, %nctaid.x;",
        "mov.u32 %r5, %ntid.x;",
        "mul.lo.s32 %r20, %r1, %r2;",
        "mul.lo.s32 %r9, %r20, %r5;",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(deps(&body, "%r9"), deps(&ssa, "%r9"));
}
