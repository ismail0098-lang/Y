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
//! **That tie used to be the weakest in the programme, and this file said so.**
//! The two GEMM kernels render their schedule from an `Ix` shared with the
//! proof generator, so a divergence is a byte-identity failure; this one read
//! emitted text with a reaching-definition walker, and `GridStrideSplit.v`
//! quoted the instructions in a COMMENT that had already gone stale.
//!
//! It is an `Ix` now - `sched_scores` / `sched_accum` in
//! `src/exact_attention.rs` - rendered to PTX by the emitter and to Coq by the
//! generator at the foot of this file. **The extraction reproduced all three
//! hand-written sequences instruction for instruction**, register numbering,
//! `mad` fusion and lazy `mov` placement included, so the refactor is checked
//! by the artifact rather than by reading it; only comments moved.
//!
//! What that buys is the theorem `GridStrideSplit.v` could not state. Its
//! partition is proved for an ABSTRACT worker count, and nothing said the
//! kernel's count was one of them: `AttentionSchedule.v` is the emitter's own
//! `nworkers`, and `the_emitted_launch_geometry_visits_every_key_exactly_once`
//! is the instantiation.
//!
//! The dataflow walker below is KEPT rather than deleted. It answers a
//! question the rendering cannot: whether the sequence loop actually READS the
//! schedule it was given. A block that is emitted and used by nothing is the
//! decorative-codegen failure this repository catalogues.
//!
//! Run with:  cargo test --release --test exact_attention_schedule

use std::path::PathBuf;

use y::cpu_gemm::{render_ptx, Ix, PtxEnv};
use y::exact_attention::{attention_ptx, sched_accum, sched_scores, MAX_EXACT_SEQ_LEN};

/// The repository root, from the test binary's own manifest directory.
fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

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

/// Every entry point that reduces over the sequence.
///
/// `attn_scores` was NOT here originally, because it was one thread per key
/// with a bounds guard and no loop - which silently required
/// `gridDim.x * blockDim.x >= S`. Measured at S=512 launched with 128 threads:
/// 768 of 1024 score slots stale and the maximum wrong (34647 against 34752),
/// which is worse, since `attn_accum` subtracts that maximum from every score.
/// So the two kernels in one file carried OPPOSITE launch contracts while the
/// module header advertised launch invariance. It is a grid-stride loop now and
/// the contract is uniform.
const REDUCING_ENTRIES: [&str; 3] = ["attn_scores", "attn_accum", "attn_accum_naive"];

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

/// The SEQUENCE loop's counter and stride, located by the kernel's own marker
/// and by the loop's BACK-EDGE.
///
/// Nothing here names a register number or a bound: the marker says which of
/// the entry's grid-stride loops is the reduction, and the back-edge says where
/// its body ends.
///
/// **The back-edge is load-bearing, and this is the third time in this file
/// that a structural pattern turned out to match more than one thing.**
/// "`add.s32 %X, %X, %rY` after the marker" also matches
/// `add.s32 %r12, %r12, %r4` - an ordinary flat-index computation
/// (`b * S + i`) inside `attn_scores` - and taking the first one made the loop
/// counter the address arithmetic. A loop's increment is the last such
/// instruction before the branch that closes it; that is a property of loops,
/// not a shape that happens to be unique today.
fn loop_counter_and_stride(body: &[String]) -> (String, String) {
    let at = body.iter().position(|l| l == SEQ_MARKER).unwrap_or_else(|| {
        panic!(
            "no `{SEQ_MARKER}` in this entry. The kernel must say which of its \
             grid-stride loops is the reduction over the sequence - `attn_accum` \
             has two of identical shape, and picking the wrong one reports the \
             decomposition as depending on `tid.x` and nothing else"
        )
    });
    let label = body[at..]
        .iter()
        .find_map(|l| l.strip_suffix(':').filter(|t| !t.contains(' ')))
        .expect("the marked sequence loop has no label")
        .to_string();
    let back = body
        .iter()
        .rposition(|l| l.ends_with(&format!("bra {label};")))
        .expect("the marked sequence loop has no back-edge");
    let is_stride = |l: &String| {
        l.starts_with("add.s32 ") && {
            let o: Vec<&str> = l[8..].trim_end_matches(';').split(',').map(str::trim).collect();
            o.len() == 3 && o[0] == o[1] && o[2].starts_with("%r")
        }
    };
    let upd = body[at..back]
        .iter()
        .rev()
        .find(|l| is_stride(l))
        .expect("the marked sequence loop never advances");
    let o: Vec<&str> = upd[8..].trim_end_matches(';').split(',').map(str::trim).collect();
    (o[0].to_string(), o[2].to_string())
}

/// **The precondition the partition theorem needs, and the one assertion here
/// that is not a transcription.**
///
/// `worker` is a mixed-radix index over some set of hardware indices;
/// `GridStrideSplit.stride_classes_partition` needs it to range over exactly
/// `[0, nworkers)`, so `nworkers` must be the product of exactly those indices'
/// EXTENTS - no fewer, or the classes overlap and keys are counted twice; no
/// more, or they no longer cover the sequence.
///
/// **Which indices is DERIVED from the kernel, not listed here.** `attn_accum`
/// mixes three (`ctaid.z`, `ctaid.x`, `tid.x`); `attn_scores` mixes two. The
/// first version of this test hardcoded the triple, which is a rule that
/// happens to hold for two of the three entries.
#[test]
fn the_worker_index_and_the_worker_count_use_paired_registers() {
    let ptx = ptx();

    /// `%ctaid.x` -> `%nctaid.x`, `%tid.x` -> `%ntid.x`. A hardware INDEX is
    /// what a worker is built from; its extent is what a worker count is built
    /// from, and the two spellings differ by exactly one `n`.
    fn extent_of(idx: &str) -> Option<String> {
        let bare = idx.strip_prefix('%')?;
        (bare.starts_with("ctaid.") || bare.starts_with("tid."))
            .then(|| format!("%n{bare}"))
    }

    for name in REDUCING_ENTRIES {
        let body = norm(&entry_body(&ptx, name));
        let (counter, stride) = loop_counter_and_stride(&body);

        // The counter is seeded from the worker index - or, where the loop
        // counter IS the worker index, from it directly.
        let worker = body
            .iter()
            .find(|l| l.starts_with(&format!("mov.u32 {counter},")))
            .map(|l| l.split(',').nth(1).unwrap().trim().trim_end_matches(';').to_string())
            .unwrap_or_else(|| counter.clone());

        let wdeps = deps(&body, &worker);
        let sdeps = deps(&body, &stride);
        assert!(!wdeps.is_empty(), "{name}: the worker index depends on nothing");

        let mut want: Vec<String> = wdeps.iter().filter_map(|d| extent_of(d)).collect();
        want.sort();
        want.dedup();
        assert!(
            !want.is_empty(),
            "{name}: the worker index {wdeps:?} contains no hardware index at \
             all, so there is nothing for the worker count to pair with"
        );

        assert_eq!(
            sdeps, want,
            "{name}: the worker index is built from {wdeps:?}, so the worker \
             count must be the product of exactly {want:?} - it is a product \
             over {sdeps:?}. Too few and the residue classes overlap, so some \
             keys are counted twice and others missed; too many and they no \
             longer cover the sequence. Only a launch that uses the missing \
             dimension can see it on a device, and only when there is a device"
        );
    }
}

/// The loop advances by the worker count, and does so exactly once - the
/// `i += nworkers` that makes the visited set a residue class rather than an
/// arbitrary subset.
#[test]
fn the_stride_is_the_worker_count() {
    let ptx = ptx();
    for name in REDUCING_ENTRIES {
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
/// decomposition in EVERY entry, not silently at one.
///
/// `attn_scores` was excluded here while it was one thread per key with no
/// stride loop to check. That exclusion was the bug, not a scoping decision:
/// the entry carried the opposite launch contract to its two siblings and
/// nothing in this file could say so. It is in `REDUCING_ENTRIES` now, and the
/// assertion below is what stops it being dropped again.
#[test]
fn every_reducing_entry_was_actually_examined() {
    let ptx = ptx();
    for name in REDUCING_ENTRIES {
        let body = norm(&entry_body(&ptx, name));
        assert!(body.len() > 40, "{name}: the entry body is too short to be the kernel");
        let (counter, stride) = loop_counter_and_stride(&body);
        assert_ne!(counter, stride, "{name}: the loop strides by itself");
        let n = deps(&body, &stride).len();
        assert!(
            (2..=3).contains(&n),
            "{name}: the worker count is a product over {n} extents. \
             `attn_scores` uses two hardware dimensions and the accumulating \
             entries three; anything else means the decomposition changed and \
             this file has not noticed"
        );
    }
    assert!(
        !entry_body(&ptx, "attn_scores").is_empty(),
        "attn_scores vanished; the loop above would then be checking two \
         entries while claiming three"
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
    for name in REDUCING_ENTRIES {
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

// ==================================================================
// The generated half: one description, rendered to PTX and to Coq.
// ==================================================================

/// Bind a hardware index's PTX name to a legal Coq identifier.
///
/// `%ctaid.x` cannot be a Coq name, and that is the only transformation. The
/// expression itself is not transcribed - it is the same [`Ix`] the emitter
/// renders to PTX, so a divergence between the proof's arithmetic and the
/// kernel's is not expressible.
fn coq_name(n: &'static str) -> String {
    match n {
        "ctaid.x" | "ctaid.y" | "ctaid.z" | "tid.x" | "ntid.x" | "nctaid.x" | "nctaid.z" => {
            n.replace('.', "_")
        }
        other => panic!(
            "the attention schedule gained an unbound name `{other}`. Add it \
             here deliberately: an unrecognised hardware index silently \
             renamed is how a proof comes to be about a different kernel"
        ),
    }
}

/// The free names of an expression, in first-occurrence order.
///
/// Derived rather than hardcoded, so a schedule that grows or loses an index
/// changes the generated Coq SIGNATURE - at which point the hand-written
/// theorems below stop compiling. A hardcoded parameter list would absorb the
/// change silently and leave the theorems talking about the old kernel.
fn free_names(ix: &Ix, out: &mut Vec<&'static str>) {
    match ix {
        Ix::Val(n) => {
            if !out.contains(n) {
                out.push(n);
            }
        }
        Ix::Lit(_) => {}
        Ix::Add(a, b) | Ix::Sub(a, b) | Ix::Mul(a, b) | Ix::Min(a, b) | Ix::Div(a, b)
        | Ix::Mod(a, b) => {
            free_names(a, out);
            free_names(b, out);
        }
        Ix::SelLt(a, b, c, d) => {
            for x in [a, b, c, d] {
                free_names(x, out);
            }
        }
    }
}

/// The derived parameter list of a rendered definition, space separated.
fn binders(ix: &Ix) -> String {
    let mut ns = Vec::new();
    free_names(ix, &mut ns);
    ns.iter().map(|n| coq_name(n)).collect::<Vec<_>>().join(" ")
}

fn coq_def(name: &str, ix: &Ix) -> String {
    let mut ns = Vec::new();
    free_names(ix, &mut ns);
    let params: Vec<String> = ns.iter().map(|n| coq_name(n)).collect();
    format!(
        "Definition {name} ({} : nat) : nat := {}.",
        params.join(" "),
        ix.coq(&coq_name)
    )
}

fn attention_schedule_path() -> PathBuf {
    repo().join("proofs").join("AttentionSchedule.v")
}

/// Render `proofs/AttentionSchedule.v`.
///
/// The DEFINITIONS come from the emitter's own expressions. The THEOREMS are
/// fixed text, deliberately: that is what makes the file bite. Move a radix in
/// `sched_accum` and the definitions move under theorems that did not, so
/// `coqc` rejects the file rather than quietly proving something true of a
/// kernel nobody ships.
fn render_attention_schedule(scores: (Ix, Ix), accum: (Ix, Ix)) -> String {
    let defs = [
        coq_def("worker_scores", &scores.0),
        coq_def("nworkers_scores", &scores.1),
        coq_def("worker_accum", &accum.0),
        coq_def("nworkers_accum", &accum.1),
    ]
    .join("\n");

    // The binders of the ROLE theorems below are derived; their right-hand
    // sides are fixed text. That asymmetry is the whole point - see the
    // comment on those theorems.
    let bind_ws = binders(&scores.0);
    let bind_ns = binders(&scores.1);
    let bind_wa = binders(&accum.0);
    let bind_na = binders(&accum.1);

    format!(
        r#"(** * The exact-attention kernel's launch geometry.

    GENERATED from [src/exact_attention.rs]'s `sched_scores` / `sched_accum` by
    `tests/exact_attention_schedule.rs`. Do not edit: regenerate with

      << Y_REWRITE_ATTENTION_PROOF=1 cargo test --release --test exact_attention_schedule >>

    Why it exists. [GridStrideSplit.v] proves that worker `w` of `n` taking
    `{{ i < S : i mod n = w }}` visits every index exactly once, in any order,
    at any worker count - and it proves it for an ABSTRACT `n`. Nothing said
    the kernel's `n` was the right one. The instructions were hand-written in
    a PTX template, this proof quoted them in a comment, and a test recovered
    them from emitted text with a reaching-definition walker. That was the
    weakest tie in the programme, and the quoted comment had already gone
    stale: it showed `mul.lo.s32 %r9, %r9, %r5`, a two-writes-to-one-register
    form the kernel stopped using.

    Now the emitter and this file render ONE expression. The definitions below
    are not a transcription.

    The obligation is the PAIRING. `worker` mixes three hardware indices with
    two radices; `nworkers` must be the product of exactly those three indices'
    EXTENTS. Drop a factor and two threads share a residue class, so their keys
    are accumulated twice; add one and some class is claimed by no thread, so
    its keys are dropped. Neither is a crash, and on random data neither is
    reliably a wrong-looking number.

    That statement is a BIJECTION from the launch geometry's coordinate box
    onto `[0, nworkers)`, and it is a mixed-radix positional index - so
    [MixedRadix.v] discharges the injectivity with no new reasoning. Third
    consumer of that schema, and the first reached from a GPU launch geometry
    rather than from a GEMM tile.

    What is NOT claimed: nothing here is about the per-thread body, the integer
    exp, the Q0.28 weight or the int8 load. This file is the schedule and only
    the schedule. *)

Require Import Arith Lia.
Require MixedRadix.

Module MR := MixedRadix.

Open Scope nat_scope.

(* ------------------------------------------------------------------ *)
(** ** The emitted expressions                                         *)
(* ------------------------------------------------------------------ *)

{defs}

(* ------------------------------------------------------------------ *)
(** ** Which index plays which role                                    *)
(* ------------------------------------------------------------------ *)

(** **The theorems below are half generated and half fixed, and the asymmetry
    is load-bearing.** Their BINDERS come from the emitted expression's own
    parameter list; their right-hand sides are fixed text naming the roles.

    Without them a relabelling is invisible. Every other theorem here applies
    `worker_accum` POSITIONALLY, and the parameter list is derived from the
    expression - so swapping two radices in the emitter renames the parameters
    in step, the definition stays the same function under different labels,
    and the byte-identity gate passes because the generated file moved too.
    That mutation survived a green sweep until these were added.

    Here the binder list moves and the right-hand side does not, so a swap
    makes the equation false and `coqc` rejects the file. Dropping an extent
    is caught harder still: the binder disappears while the right-hand side
    still names it, so the reference is unbound. *)

Theorem the_worker_index_is_a_mixed_radix_number :
  forall {bind_wa} : nat,
    worker_accum {bind_wa}
      = ctaid_z * (nctaid_x * ntid_x) + ctaid_x * ntid_x + tid_x.
Proof. intros. unfold worker_accum. ring. Qed.

Theorem the_worker_count_is_the_product_of_the_extents :
  forall {bind_na} : nat,
    nworkers_accum {bind_na} = nctaid_z * nctaid_x * ntid_x.
Proof. intros. unfold nworkers_accum. ring. Qed.

Theorem the_scores_worker_index_is_a_mixed_radix_number :
  forall {bind_ws} : nat,
    worker_scores {bind_ws} = ctaid_x * ntid_x + tid_x.
Proof. intros. unfold worker_scores. ring. Qed.

Theorem the_scores_worker_count_is_the_product_of_the_extents :
  forall {bind_ns} : nat,
    nworkers_scores {bind_ns} = nctaid_x * ntid_x.
Proof. intros. unfold nworkers_scores. ring. Qed.

(* ------------------------------------------------------------------ *)
(** ** One digit of a positional index                                 *)
(* ------------------------------------------------------------------ *)

(** The bound both radices need, in one place. *)
Lemma digit_bound : forall q r Q R, q < Q -> r < R -> q * R + r < Q * R.
Proof. intros. nia. Qed.

(* ------------------------------------------------------------------ *)
(** ** The worker map is a bijection onto the worker count             *)
(* ------------------------------------------------------------------ *)

(** **In range.** Every thread of the launch is a worker the reduction folds. *)
Theorem the_worker_is_below_the_worker_count :
  forall cz cx tx ncz ncx ntx,
    cz < ncz -> cx < ncx -> tx < ntx ->
    worker_accum cz ncx cx ntx tx < nworkers_accum ncz ncx ntx.
Proof.
  intros cz cx tx ncz ncx ntx Hz Hx Ht.
  unfold worker_accum, nworkers_accum.
  assert (H1 : cz * ncx + cx < ncz * ncx) by (apply digit_bound; assumption).
  assert (H2 : (cz * ncx + cx) * ntx + tx < (ncz * ncx) * ntx)
    by (apply digit_bound; assumption).
  lia.
Qed.

(** **Injective.** Two threads never share a worker id, so no key is
    accumulated twice. Discharged by [MixedRadix.two_digit_unique]: the worker
    index IS `q*(B1*B0) + m*B0 + r` at `B0 = ntid.x`, `B1 = nctaid.x`. *)
Theorem distinct_threads_are_distinct_workers :
  forall cz1 cx1 tx1 cz2 cx2 tx2 ncx ntx,
    0 < ncx -> 0 < ntx ->
    cx1 < ncx -> cx2 < ncx -> tx1 < ntx -> tx2 < ntx ->
    worker_accum cz1 ncx cx1 ntx tx1 = worker_accum cz2 ncx cx2 ntx tx2 ->
    cz1 = cz2 /\ cx1 = cx2 /\ tx1 = tx2.
Proof.
  intros cz1 cx1 tx1 cz2 cx2 tx2 ncx ntx Hncx Hntx Hx1 Hx2 Ht1 Ht2 Heq.
  unfold worker_accum in Heq.
  apply (MR.two_digit_unique ntx ncx cz1 cx1 tx1 cz2 cx2 tx2);
    try assumption.
  transitivity ((cz1 * ncx + cx1) * ntx + tx1); [ ring | ].
  rewrite Heq. ring.
Qed.

(** **Onto.** Every worker id the reduction folds is some thread's, so no
    residue class goes unclaimed and no key is dropped. *)
Theorem every_worker_is_some_thread :
  forall w ncz ncx ntx,
    0 < ncx -> 0 < ntx ->
    w < nworkers_accum ncz ncx ntx ->
    exists cz cx tx,
      cz < ncz /\ cx < ncx /\ tx < ntx /\
      worker_accum cz ncx cx ntx tx = w.
Proof.
  intros w ncz ncx ntx Hncx Hntx Hw.
  unfold nworkers_accum in Hw.
  exists (w / (ncx * ntx)), ((w / ntx) mod ncx), (w mod ntx).
  assert (Hdd : w / (ncx * ntx) = w / ntx / ncx).
  {{ rewrite Nat.Div0.div_div. f_equal. ring. }}
  assert (E1 : (w / ntx / ncx) * ncx + (w / ntx) mod ncx = w / ntx).
  {{ rewrite Nat.mul_comm. symmetry. apply Nat.div_mod_eq. }}
  assert (E2 : (w / ntx) * ntx + w mod ntx = w).
  {{ rewrite Nat.mul_comm. symmetry. apply Nat.div_mod_eq. }}
  split; [ | split; [ | split ] ].
  - apply Nat.Div0.div_lt_upper_bound. nia.
  - apply Nat.mod_upper_bound. lia.
  - apply Nat.mod_upper_bound. lia.
  - unfold worker_accum. rewrite Hdd, E1, E2. reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(** ** The pairing is load-bearing                                     *)
(* ------------------------------------------------------------------ *)

(** **The refutation.** Drop `%nctaid.z` from the product - which is exactly
    what the worker count of `attn_scores` is - and the in-range property is
    FALSE, so workers alias.

    Stated as a refuted theorem rather than an exhibited witness: a witness
    shows the case exists, a refutation shows no proof of the weakened claim
    can exist. *)
Theorem dropping_the_z_extent_overflows_the_worker_count :
  ~ (forall cz cx tx ncz ncx ntx,
       cz < ncz -> cx < ncx -> tx < ntx ->
       worker_accum cz ncx cx ntx tx < nworkers_scores ncx ntx).
Proof.
  intro H. specialize (H 1 0 0 2 1 1).
  unfold worker_accum, nworkers_scores in H. simpl in H. lia.
Qed.

(** **The control.** The refutation above must not be read as "`attn_scores`'s
    worker count is wrong". It is right FOR ITS OWN KERNEL, which is launched
    with one CTA in z and does not read `%ctaid.z` at all: at `nctaid.z = 1`
    the two schedules are the same map. So one proof covers both entries. *)
Theorem the_scores_schedule_is_the_accumulate_schedule_at_one_z :
  forall cx tx ncx ntx,
    worker_accum 0 ncx cx ntx tx = worker_scores cx ntx tx
    /\ nworkers_accum 1 ncx ntx = nworkers_scores ncx ntx.
Proof.
  intros. unfold worker_accum, worker_scores, nworkers_accum, nworkers_scores.
  split; ring.
Qed.

Print Assumptions the_worker_is_below_the_worker_count.
Print Assumptions distinct_threads_are_distinct_workers.
Print Assumptions every_worker_is_some_thread.
Print Assumptions dropping_the_z_extent_overflows_the_worker_count.
Print Assumptions the_scores_schedule_is_the_accumulate_schedule_at_one_z.
Print Assumptions the_worker_index_is_a_mixed_radix_number.
Print Assumptions the_worker_count_is_the_product_of_the_extents.
Print Assumptions the_scores_worker_index_is_a_mixed_radix_number.
Print Assumptions the_scores_worker_count_is_the_product_of_the_extents.
Print Assumptions digit_bound.
"#
    )
}

/// The gate. Regenerates and compares byte for byte.
#[test]
fn the_committed_attention_schedule_is_what_the_emitter_generates() {
    let want = render_attention_schedule(sched_scores(), sched_accum());
    let path = attention_schedule_path();

    if std::env::var("Y_REWRITE_ATTENTION_PROOF").is_ok() {
        std::fs::write(&path, &want).expect("write AttentionSchedule.v");
        eprintln!("rewrote {}", path.display());
        return;
    }

    let have = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "proofs/AttentionSchedule.v is missing ({e}). It is GENERATED and \
             committed; regenerate with `Y_REWRITE_ATTENTION_PROOF=1 cargo \
             test --release --test exact_attention_schedule`."
        )
    });
    if have == want {
        return;
    }
    let first = have
        .lines()
        .zip(want.lines())
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| format!("line {}:\n  committed: {a}\n  generated: {b}", i + 1))
        .unwrap_or_else(|| {
            format!(
                "no differing line - the files differ in length ({} vs {})",
                have.lines().count(),
                want.lines().count()
            )
        });
    panic!(
        "proofs/AttentionSchedule.v is not what src/exact_attention.rs \
         generates.\n\n{first}\n\nThe launch geometry has one source. Either \
         `sched_scores` / `sched_accum` moved and the proof still describes \
         the old kernel, or this generated file was edited by hand. \
         Regenerate with `Y_REWRITE_ATTENTION_PROOF=1 cargo test --release \
         --test exact_attention_schedule` and re-check the proofs."
    );
}

/// The control, and it is the one the exact-GEMM gate had to learn the hard
/// way: a byte-identity gate is only as good as its generator, and a generator
/// neutered to echo the committed file passes forever while checking nothing.
///
/// Asserting the output CONTAINS the shipped definitions does not catch that -
/// the committed file contains them. What catches it is rendering a schedule
/// that is NOT the shipped one and requiring the result to differ.
#[test]
fn the_generator_is_a_function_of_the_schedule_not_of_the_committed_file() {
    let committed =
        std::fs::read_to_string(attention_schedule_path()).expect("read the committed proof");

    // A worker count with the z extent dropped - the exact bug the refutation
    // in the generated file is about.
    let perturbed = (
        sched_accum().0,
        Ix::mul(Ix::val("nctaid.x"), Ix::val("ntid.x")),
    );
    let out = render_attention_schedule(sched_scores(), perturbed);
    assert_ne!(
        out, committed,
        "the generator ignored its argument: a schedule with `nctaid.z` \
         dropped from the worker count rendered the committed file. A \
         generator that echoes cannot fail its own byte-identity gate"
    );
    assert!(
        out.contains("Definition nworkers_accum (nctaid_x ntid_x : nat)"),
        "the perturbed schedule did not reach the rendered definition"
    );

    // And the signature really is derived: dropping an index must change the
    // parameter list, not just the body.
    assert!(
        committed.contains("Definition nworkers_accum (nctaid_z nctaid_x ntid_x : nat)"),
        "the shipped worker count is no longer the product of the three \
         extents. That is either a real schedule change or a rendering bug, \
         and the difference matters: this parameter list IS the pairing \
         obligation"
    );
}

/// The tie to the artifact: the rendered blocks are what the kernel contains.
///
/// The byte-identity gate above ties the PROOF to the description. This ties
/// the DESCRIPTION to the emitted PTX. Without it both could be perfectly
/// consistent with each other and with nothing the compiler produces - which
/// is the shape of "proof-carrying described the repository, not the output".
#[test]
fn the_emitted_kernels_contain_the_rendered_schedule() {
    let p = ptx();

    // `attn_accum_naive`: no pre-bound registers, no trailing comments, so the
    // block is exactly what a fresh render produces.
    let mut env = PtxEnv::default();
    let regs: Vec<(String, Option<&'static str>)> = ["%r1", "%r2", "%r3", "%r4", "%r5", "%r6",
        "%r7", "%r8", "%r20", "%r9"]
        .iter()
        .map(|r| (r.to_string(), None))
        .collect();
    let mut it = regs.into_iter();
    let (worker, nworkers) = sched_accum();
    let (mut lines, wreg) = render_ptx(&worker, &mut it, &mut env);
    let (rest, nreg) = render_ptx(&nworkers, &mut it, &mut env);
    lines.extend(rest);
    let block = lines.join("\n");

    let naive = entry_body(&p, "attn_accum_naive");
    assert!(
        p.contains(&block),
        "the emitted module does not contain the rendered schedule block. \
         Either the emitter stopped rendering it or the register supply \
         changed.\n--- rendered ---\n{block}\n--- attn_accum_naive ---\n{naive}"
    );

    // And the loop must actually USE what was rendered. A block that is
    // present but read by nothing is the decorative-codegen failure this
    // repository catalogues: `mov.u32 %r14, %r7` seeds the induction variable
    // with the worker, `add.s32 %r14, %r14, %r9` strides by the worker count.
    let body = norm(&entry_body(&p, "attn_accum"));
    let (counter, stride) = loop_counter_and_stride(&body);
    assert_eq!(
        stride, nreg,
        "the sequence loop strides by {stride}, but the rendered worker count \
         is in {nreg}. A grid-stride loop whose stride is not the worker count \
         does not partition anything"
    );
    assert!(
        body.iter()
            .any(|l| l.starts_with(&format!("mov.u32 {counter}, {wreg}"))),
        "the sequence loop's induction variable {counter} is not seeded from \
         the rendered worker index {wreg}"
    );
}
