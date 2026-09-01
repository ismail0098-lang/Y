//! The exact-attention kernel, as PTX.
//!
//! Lives here rather than inside a test so there is ONE copy: the device
//! tests, the `--emit-attention-ptx` CLI, and the Python bridge in
//! `tools/ptx_bridge.py` all take the same string. A kernel that exists only
//! in a test file cannot honestly be described as something the compiler
//! produces.
//!
//! See `docs/deterministic_inference.md` M4 for the design. Every reduction
//! over the sequence is an exact integer one, so the answer does not depend on
//! `blockDim.x`, `gridDim.x`, `gridDim.z`, or the order the atomics land.

use crate::cpu_gemm::{render_ptx, Ix, PtxEnv};
use crate::fixed_exp::ptx_device_function;

/// The largest `seq_len` this kernel is exact for.
///
/// A softmax weight is Q0.28, so `p < 2^28`; `V` is int8, so `|v| <= 127`; and
/// `sum p_i * v_i` accumulates in a 64-bit two's-complement register (the
/// `red.shared.add.u64` / `red.global.add.u64` chain is signed arithmetic under
/// another name). Exactness therefore needs `S * (2^28 - 1) * 127 < 2^63`.
///
/// This bound was written down in a comment in
/// `tests/gpu_attention_invariance.rs` ("the i64 accumulator holds sequences to
/// 2^28 keys") and enforced nowhere. Past it the sum WRAPS, which is a wrong
/// answer rather than an imprecise one - the same failure the Python
/// prototype's `K >= 133,153` int32 wrap turned out to be, recorded in
/// `docs/bit_identical_decode.md` finding 04.
///
/// Note this is a much higher ceiling than the prototype's 264,208 tokens: that
/// one comes from recombining in float64 (`2^53`), and this kernel recombines
/// in integers.
pub const MAX_EXACT_SEQ_LEN: usize = (1usize << 63) / (((1usize << 28) - 1) * 127);

/// The smallest multiplier the kernel's `mul.wide.s32` reads as positive.
///
/// The parameter is declared `.param .u32 q6` and consumed by a SIGNED wide
/// multiply, so the two readings diverge at `2^31`. Nothing in the emitter can
/// check a runtime value; [`temperature_fixed_point`] is where the check goes,
/// and `proofs/SoftmaxErrorBound.v`'s
/// `the_two_readings_of_the_multiplier_disagree_above_two_to_the_thirty_one`
/// exhibits the divergence: at `KFix = 2^31` and one unit of score delta, the
/// unsigned reading gives argument `32768` (a weight of `2^28 * 2^-0.5`) and
/// the signed one gives `-32768`, which the `cvt.u32.u64` below the saturate
/// wraps to `4294934528` -- above the table, so the weight is ZERO. Not a
/// rounding difference: one key contributes and the other does not.
pub const KFIX_SIGNED_LIMIT: u64 = 1 << 31;

/// `KFix = round(C * 2^32)`, the kernel's fixed-point temperature.
///
/// `C` is log2 units per unit of integer score -- for an int8 pipeline,
/// `q_scale * k_scale * log2(e) / sqrt(head_dim)`. The kernel forms the exp's
/// Q16.16 argument as `t = ((m - s) * KFix + 2^15) >> 16`, so one factor of
/// `2^16` is consumed by the shift and `KFix` must carry `2^32`.
///
/// **This existed in seven transcriptions and was checked in one.** Every test
/// that needs a multiplier wrote `(c * 2f64.powi(32)).round()` itself, and only
/// `tools/ptx_bridge.py` guarded the result -- with a bare `continue`, so a
/// temperature outside the representable range silently removed a case from a
/// measurement rather than failing it. Both bounds below come from that guard;
/// what is new is that they are derived rather than asserted, enforced in the
/// compiler rather than in a script, and stated as hypotheses of a theorem.
///
/// The two refusals are the two ways the multiplier stops meaning a
/// temperature, and both are silent in the kernel:
///
/// * **`KFix == 0`** -- reachable whenever `C < 2^-33`. Then `t` is `0` for
///   EVERY key (`(0 + 2^15) >> 16 == 0`), so every weight is exactly `2^28`
///   and the softmax is UNIFORM. That is the same symptom
///   `tools/ptx_bridge.py`'s finding 06 records from the opposite cause -- the
///   bridge passed `C * 2^16`, i.e. a multiplier `2^16` too small -- and its
///   differential could not see it, because both arms replicate the kernel's
///   formula and agree bit for bit on a uniform answer just as readily.
///   `a_zero_multiplier_gives_every_key_the_same_weight` is the proof of it.
/// * **`KFix >= 2^31`** -- see [`KFIX_SIGNED_LIMIT`].
///
/// What is NOT refused is the quantization error itself, because it is bounded
/// rather than catastrophic: rounding `C * 2^32` to an integer moves the
/// exponent by at most `delta / 2` in Q32 log2 units, i.e. `delta * 2^-33` log2
/// units, INDEPENDENTLY of `KFix`. Use [`score_delta_span`] to price it for a
/// given head_dim; at head_dim 128 it is a factor of 1.00034 on any weight.
pub fn temperature_fixed_point(c: f64) -> Result<u32, String> {
    if !c.is_finite() || c <= 0.0 {
        return Err(format!(
            "the softmax temperature C must be finite and positive, got {c:e}; \
             it is log2 units per unit of integer score, so a non-positive value \
             is not a temperature"
        ));
    }
    let scaled = (c * 4294967296.0).round();
    if scaled < 1.0 {
        return Err(format!(
            "C = {c:e} rounds to KFix = 0 (C < 2^-33), which makes the exp's \
             argument 0 for every key: every weight is exactly 2^28 and the \
             softmax is UNIFORM. That is a wrong answer, not an imprecise one, \
             and nothing downstream can see it"
        ));
    }
    if scaled >= KFIX_SIGNED_LIMIT as f64 {
        return Err(format!(
            "C = {c:e} gives KFix = {scaled:.0}, at or above {KFIX_SIGNED_LIMIT}; \
             the kernel's `mul.wide.s32` reads the .u32 parameter as SIGNED, so \
             the product is negative, the saturate bounds nothing and the \
             narrowing wraps -- keys get the weight of an unrelated argument"
        ));
    }
    Ok(scaled as u32)
}

/// The largest `m - s_i` an int8 score pipeline can produce at this head_dim.
///
/// `s = sum_d q_d * k_d` with `|q|, |k| <= 127`, so `|s| <= 127^2 * head_dim`
/// and the delta between the maximum and any other score is at most twice
/// that. This is the quantity the temperature's quantization error is stated
/// over -- the exponent moves by at most `span / 2` in Q32 log2 units, so a
/// bound on the span IS the bound on the error, and it is a compile-time
/// function of head_dim alone.
///
/// It is not a refusal: an `i32` score holds `127^2 * head_dim` for every
/// head_dim this kernel accepts (the 48 KB shared-memory limit caps head_dim at
/// 6143, where `127^2 * 6143` is 99,059,807, well inside `i32`).
pub fn score_delta_span(head_dim: usize) -> u64 {
    2 * 127 * 127 * head_dim as u64
}

// ------------------------------------------------------------------
// The launch-geometry schedule, as ONE description.
//
// `proofs/GridStrideSplit.v` proves that worker `w` of `n` taking
// `{ i < S : i mod n = w }` visits every index exactly once, in any order, at
// any worker count. That theorem is stated over an abstract `n` - and until
// now nothing said the kernel's `n` was the right one. The instructions were
// hand-written in the template below, `GridStrideSplit.v` quoted them in a
// COMMENT, and `tests/exact_attention_schedule.rs` matched the two back up by
// parsing emitted PTX with a reaching-definition walker.
//
// That is the weakest tie in the programme. Every other proof here shares one
// description with the emitter, so a divergence is a byte-identity failure;
// this one read text, and CLAUDE.md already records two traps where a
// structural pattern matched more than one thing.
//
// So the schedule is an [`Ix`] now - the same type the exact-GEMM driver's
// schedule is extracted into - rendered to PTX by the emitter and to Coq by
// `tools/gen_attention_schedule` (`tests/exact_attention_schedule.rs` with
// `Y_REWRITE_ATTENTION_PROOF=1`). The obligation `GridStrideSplit.v` needs is
// then a theorem about the emitted expressions rather than about a comment.
// ------------------------------------------------------------------

/// `attn_scores`: one worker per thread of a 1-D grid.
///
/// `worker = ctaid.x * ntid.x + tid.x`, `nworkers = nctaid.x * ntid.x`.
pub fn sched_scores() -> (Ix, Ix) {
    (
        Ix::add(
            Ix::mul(Ix::val("ctaid.x"), Ix::val("ntid.x")),
            Ix::val("tid.x"),
        ),
        Ix::mul(Ix::val("nctaid.x"), Ix::val("ntid.x")),
    )
}

/// `attn_accum` and `attn_accum_naive`: the sequence is ALSO partitioned across
/// `ctaid.z`, so the CTA index is flattened first.
///
/// `worker = (ctaid.z * nctaid.x + ctaid.x) * ntid.x + tid.x`,
/// `nworkers = nctaid.z * nctaid.x * ntid.x`.
///
/// The pairing is the whole obligation: `nworkers` must be the product of
/// exactly the extents of the hardware indices `worker` depends on. Fewer and
/// two threads share a residue class - they accumulate the same keys twice;
/// more and some class is claimed by no thread, so keys are dropped. Neither
/// shows up as a crash. `AttentionSchedule.the_worker_index_is_a_bijection`
/// is that statement, and it is a mixed-radix positional index - the third
/// consumer of `proofs/MixedRadix.v`, and the first reached from a GPU launch
/// geometry rather than from a GEMM tile.
pub fn sched_accum() -> (Ix, Ix) {
    (
        Ix::add(
            Ix::mul(
                Ix::add(
                    Ix::mul(Ix::val("ctaid.z"), Ix::val("nctaid.x")),
                    Ix::val("ctaid.x"),
                ),
                Ix::val("ntid.x"),
            ),
            Ix::val("tid.x"),
        ),
        Ix::mul(
            Ix::mul(Ix::val("nctaid.z"), Ix::val("nctaid.x")),
            Ix::val("ntid.x"),
        ),
    )
}

/// Render one entry's `(worker, nworkers)` pair as a contiguous PTX block.
///
/// `regs` supplies the result register of each instruction in emission order.
/// `pre` names values the surrounding kernel has already `mov`'d into a
/// register, so they are not materialised twice - which is what makes
/// `attn_accum`, whose `%ntid.x` and `%tid.x` are loaded far above, render to
/// the same instructions as `attn_accum_naive`, whose are not.
fn render_sched(
    sched: (Ix, Ix),
    pre: &[(&'static str, &str)],
    regs: &[(&str, Option<&'static str>)],
) -> String {
    let (worker, nworkers) = sched;
    let mut env = PtxEnv::default();
    for (n, r) in pre {
        env.bind(n, r);
    }
    let mut names = regs.iter().map(|(r, n)| (r.to_string(), *n));
    let (mut lines, _) = render_ptx(&worker, &mut names, &mut env);
    let (rest, _) = render_ptx(&nworkers, &mut names, &mut env);
    lines.extend(rest);
    assert!(
        names.next().is_none(),
        "render_sched was given more result registers than the schedule has \
         instructions - the extra ones are silently unused, which is how a \
         renumbering drifts"
    );
    lines.join("\n")
}

/// The two entry points: `attn_scores` (exact int32 scores + global integer
/// max) and `attn_accum` / `attn_accum_naive` (Q0.28 weights via the integer
/// exp, then an exact integer accumulation).
///
/// `Err` rather than a kernel, for anything outside the range the exactness
/// argument covers. This used to take a third argument, `c_hex`, "the softmax
/// temperature folded with log2(e)" - documented as needing to stay a power of
/// two "because the kernel's argument conversion is a shift". The kernel was
/// later changed to take the temperature as a RUNTIME parameter, precisely so
/// that an arbitrary `q_scale * k_scale / sqrt(d)` would work, and the constant
/// was left behind: `$C` appears nowhere in the template, so the `.replace` for
/// it matched nothing and the argument was accepted and discarded. The
/// doc-comment then asserted a restriction that the change existed to remove.
pub fn attention_ptx(head_dim: usize, seq_len: usize) -> Result<String, String> {
    if head_dim == 0 || seq_len == 0 {
        return Err(format!(
            "exact attention needs head_dim > 0 and seq_len > 0, got {head_dim} and {seq_len}"
        ));
    }
    // `osm` is one u64 per output element, and `lsm` one more.
    let smem = head_dim
        .checked_mul(8)
        .and_then(|b| b.checked_add(8))
        .ok_or_else(|| format!("head_dim {head_dim} overflows the shared-memory size"))?;
    if smem > 48 * 1024 {
        return Err(format!(
            "head_dim {head_dim} needs {smem} bytes of shared memory, past the 48 KB static limit"
        ));
    }
    if seq_len > MAX_EXACT_SEQ_LEN {
        return Err(format!(
            "seq_len {seq_len} is past {MAX_EXACT_SEQ_LEN}, where the 64-bit accumulator \
             wraps and the result stops being exact. This kernel's whole claim is that \
             the answer does not depend on how the reduction was split; a wrapped sum \
             is a wrong answer, not an imprecise one"
        ));
    }
    // The three schedule sites, rendered from `sched_scores` / `sched_accum`.
    // The register names are supplied here, in emission order, so the rendered
    // block reproduces what was hand-written - which is what lets this
    // extraction be checked by byte-identity rather than by reading it.
    let blk_scores = render_sched(
        sched_scores(),
        &[],
        &[
            ("%r1", None),
            ("%r2", None),
            ("%r3", None),
            ("%r4", Some("i = worker")),
            ("%r20", None),
            ("%r21", Some("nworkers")),
        ],
    );
    // `%ntid.x` and `%tid.x` are already in `%r5` / `%r6` here: the zeroing
    // loop above needs them, so they are loaded long before the schedule.
    let blk_accum = render_sched(
        sched_accum(),
        &[("ntid.x", "%r5"), ("tid.x", "%r6")],
        &[
            ("%r1", None),
            ("%r2", None),
            ("%r3", None),
            ("%r4", Some("flat CTA index")),
            ("%r7", Some("worker")),
            ("%r8", None),
            ("%r20", None),
            ("%r9", Some("nworkers")),
        ],
    );
    let blk_naive = render_sched(
        sched_accum(),
        &[],
        &[
            ("%r1", None),
            ("%r2", None),
            ("%r3", None),
            ("%r4", None),
            ("%r5", None),
            ("%r6", None),
            ("%r7", None),
            ("%r8", None),
            ("%r20", None),
            ("%r9", None),
        ],
    );
    let body = KERNELS
        .replace("$SCHED_SCORES", &blk_scores)
        .replace("$SCHED_ACCUM", &blk_accum)
        .replace("$SCHED_NAIVE", &blk_naive)
        .replace("$D8", &(head_dim * 8).to_string())
        .replace("$D", &head_dim.to_string())
        .replace("$S", &seq_len.to_string());
    Ok(format!(
        // sm_80, NOT sm_89. This kernel uses nothing newer than Ampere -
        // verified by assembling it at every target in
        // `tests/ptx_portability.rs`. It said sm_89 for no reason, and a
        // `.target` ABOVE the device is a hard load failure ("SM version
        // specified by .target is higher than default SM version assumed"),
        // so that one token locked the exact-attention path out of every
        // Ampere card: 3060, 3090, A100. PTX is forward compatible, so
        // sm_80 still runs on Ada and Blackwell.
        ".version 7.0\n.target sm_80\n.address_size 64\n{}{}",
        ptx_device_function(),
        body
    ))
}

const KERNELS: &str = r#"

// s_i = q . k_i  (exact int32), and the global max by integer atomic.
.visible .entry attn_scores(
    .param .u64 p0,   // Q      int8  [B][D]
    .param .u64 p1,   // K      int8  [S][D]
    .param .u64 p2,   // Scores int32 [B][S]
    .param .u64 p3    // M      int32 [B]
)
{
    .reg .pred %p<4>;
    .reg .s32  %r<32>;
    .reg .s64  %rd<32>;

    ld.param.u64 %rd1, [p0];
    ld.param.u64 %rd2, [p1];
    ld.param.u64 %rd3, [p2];
    ld.param.u64 %rd4, [p3];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;

    // The launch-geometry schedule, RENDERED rather than written: see
    // `sched_scores` below and `proofs/AttentionSchedule.v`, which is the same
    // expression rendered to Coq. `GridStrideSplit.v` used to quote these
    // instructions in a comment and a test used to recover them from emitted
    // text with a dataflow walker; there is one description now.
$SCHED_SCORES
    mov.u32 %r5, %ctaid.y;              // b

    // [Y SEQUENCE REDUCTION] grid-stride over S, stride = nworkers.
    //
    // This used to be ONE THREAD PER KEY with a bounds guard and no loop, so it
    // silently required `gridDim.x * blockDim.x >= S`. Measured on an RTX 4070
    // Ti SUPER at S=512: launched with 128 threads it left 768 of 1024 score
    // slots stale AND computed the wrong maximum (34647 against 34752). A wrong
    // max is worse than a partly-filled buffer - `attn_accum` subtracts it from
    // every score, so every softmax weight moves.
    //
    // That made the two kernels in this file carry OPPOSITE launch contracts
    // while the module header advertises launch invariance. The precondition is
    // removed rather than documented: this is now the same residue-class
    // partition `attn_accum` uses, so proofs/GridStrideSplit.v covers it and any
    // geometry visits every key exactly once.
SLOOP_I:
    setp.ge.s32 %p1, %r4, $S;
    @%p1 bra DONE_A;

    mul.lo.s32 %r6, %r5, $D;
    mul.wide.s32 %rd5, %r6, 1;
    add.s64 %rd6, %rd1, %rd5;           // &Q[b][0], recomputed: LOOP_A walks it
    mul.wide.s32 %rd7, %r4, $D;
    add.s64 %rd8, %rd2, %rd7;           // &K[i][0]

    mov.u32 %r8, 0;                     // acc
    mov.u32 %r9, 0;                     // d
LOOP_A:
    ld.global.s8 %r10, [%rd6];
    ld.global.s8 %r11, [%rd8];
    mad.lo.s32 %r8, %r10, %r11, %r8;
    add.s64 %rd6, %rd6, 1;
    add.s64 %rd8, %rd8, 1;
    add.s32 %r9, %r9, 1;
    setp.lt.s32 %p2, %r9, $D;
    @%p2 bra LOOP_A;

    mul.lo.s32 %r12, %r5, $S;
    add.s32 %r12, %r12, %r4;
    mul.wide.s32 %rd9, %r12, 4;
    add.s64 %rd10, %rd3, %rd9;
    st.global.s32 [%rd10], %r8;

    mul.wide.s32 %rd11, %r5, 4;
    add.s64 %rd12, %rd4, %rd11;
    // `max` is associative, commutative AND idempotent, so the atomics' order
    // does not matter. Its IDENTITY is the remaining question, and it is
    // answered here rather than left to the host.
    //
    // A signed max wants i32::MIN, which is not a uniform byte pattern - so the
    // caller could not seed M with the same `memset(0)` that L, O and P need,
    // and a caller who used one anyway got a silently wrong answer whenever a
    // row's scores were ALL negative (with max(0, s) = 0 the softmax then
    // subtracts a maximum no key attains). That never fires on random int8
    // data, which is the worst kind of precondition.
    //
    // `x ^ 0x80000000` is the order-preserving signed->unsigned bijection, so a
    // max over the biased values is the max over the originals, and UNSIGNED
    // max has identity 0. A zeroed buffer therefore MEANS i32::MIN. The two
    // accumulating entries undo the bias when they load M[b]; nothing else
    // reads it.
    xor.b32 %r22, %r8, -2147483648;
    red.global.max.u32 [%rd12], %r22;

    add.s32 %r4, %r4, %r21;
    bra SLOOP_I;
DONE_A:
    ret;
}

// p_i, then l and o by a TWO-LEVEL integer reduction: a per-thread register
// for `l`, shared memory for `o`, and one global atomic per CTA per output
// element. The sequence is partitioned across (ctaid.z, ctaid.x, tid.x) with a
// grid-stride loop, so blockDim.x, gridDim.x and gridDim.z may all be varied
// freely and every one of them changes which thread accumulates which term.
//
// The reduction tree here is not fixed -- it has a different SHAPE at every
// block size, and the shared-memory atomics inside a CTA interleave in
// hardware-chosen order just as the global ones do. That is the point: this
// kernel is order-independent because integer addition is, not because
// anything about the order was pinned down. `attn_accum_naive` below is the
// one-atomic-per-term version, kept so the two structures can be checked
// against each other bit for bit.
.visible .entry attn_accum(
    .param .u64 q0,   // Scores int32 [B][S]
    .param .u64 q1,   // V      int8  [S][D]
    .param .u64 q2,   // M      int32 [B]
    .param .u64 q3,   // L      u64   [B]
    .param .u64 q4,   // O      s64   [B][D]
    .param .u64 q5,   // P      int32 [B][S]   (the quantised weights, for the oracle)
    .param .u32 q6    // KFix = C * 2^32, where C is log2 units per unit of
                      // integer score. t = ((m - s) * KFix + 2^15) >> 16,
                      // and t is the exp's Q16.16 argument. NOT C * 2^16.
)
{
    .shared .align 8 .b8 osm[$D8];
    .shared .align 8 .b8 lsm[8];

    .reg .pred %p<8>;
    .reg .s32  %r<64>;
    .reg .s64  %rd<64>;

    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;

    // Zero the CTA's private accumulators. Written as a grid-stride loop over
    // d so that blockDim.x may be smaller OR larger than D.
    mov.u64 %rd40, 0;
    mov.u32 %r40, %r6;
ZLOOP:
    setp.ge.s32 %p3, %r40, $D;
    @%p3 bra ZDONE;
    mov.u32 %r41, osm;
    mad.lo.s32 %r41, %r40, 8, %r41;
    st.shared.u64 [%r41], %rd40;
    add.s32 %r40, %r40, %r5;
    bra ZLOOP;
ZDONE:
    setp.ne.s32 %p4, %r6, 0;
    @%p4 bra ZSKIP;
    mov.u32 %r42, lsm;
    st.shared.u64 [%r42], %rd40;
ZSKIP:
    bar.sync 0;

    ld.param.u64 %rd1, [q0];
    ld.param.u64 %rd2, [q1];
    ld.param.u64 %rd3, [q2];
    ld.param.u64 %rd4, [q3];
    ld.param.u64 %rd5, [q4];
    ld.param.u64 %rd6, [q5];
    ld.param.u32 %r50, [q6];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;
    cvta.to.global.u64 %rd5, %rd5;
    cvta.to.global.u64 %rd6, %rd6;

    // Rendered from `sched_accum` - see the note in `attn_scores`. The worker
    // count comes out in two steps and TWO registers because the name supply
    // hands a fresh register to every instruction. Writing `%r9` twice is
    // legal PTX and was a trap for anything reading this back: a
    // reaching-definition walk that resolves an operand at the USE finds the
    // same instruction again. Nothing needs the reuse, and now nothing can
    // reintroduce it by hand.
$SCHED_ACCUM
    mov.u32 %r10, %ctaid.y;             // b

    mul.wide.s32 %rd10, %r10, 4;
    add.s64 %rd11, %rd3, %rd10;
    ld.global.s32 %r11, [%rd11];        // m = M[b], biased
    // Undo `attn_scores`'s unsigned bias. Deleting this line from ONE of the
    // two accumulating entries used to pass every test in the repo, including
    // the accum-vs-naive differential on `p` that ought to catch a one-sided
    // change: the bias is 2^31, the exp argument below is
    // `((m - s) * KFix + 2^15) >> 16` computed in 64 bits and then TRUNCATED to
    // u32, so a missing undo shifts it by `2^15 * KFix` -- and every test used
    // `KFix = 8 << 16 = 2^19`, for which that is exactly 2^34, i.e. zero
    // modulo 2^32. The truncation annihilated the bug, at the one temperature
    // every arm of the differential shared. A real model's scale is
    // `q_scale * k_scale / sqrt(d)` and does not oblige;
    // `the_accumulating_entries_undo_the_bias` uses 2^19 + 1 for exactly that
    // reason.
    xor.b32 %r11, %r11, -2147483648;

    mul.wide.s32 %rd12, %r10, 8;
    add.s64 %rd13, %rd4, %rd12;         // &L[b]
    mul.lo.s32 %r12, %r10, $D;
    mul.wide.s32 %rd14, %r12, 8;
    add.s64 %rd15, %rd5, %rd14;         // &O[b][0]
    mul.lo.s32 %r13, %r10, $S;
    mul.wide.s32 %rd16, %r13, 4;
    add.s64 %rd17, %rd1, %rd16;         // &Scores[b][0]
    add.s64 %rd25, %rd6, %rd16;         // &P[b][0]

    mov.u64 %rd33, 0;                   // per-thread l accumulator
    // [Y SEQUENCE REDUCTION] grid-stride over S, stride = nworkers. This is
    // the partition proofs/GridStrideSplit.v is about. The marker is here
    // because the entry contains a SECOND grid-stride loop of identical shape
    // (ZLOOP, over d) and nothing else distinguishes them structurally - they
    // differ only in their bound, so a reader or a tool picking "the first
    // one" gets the wrong loop.
    mov.u32 %r14, %r7;                  // i = worker
LOOP_I:
    setp.ge.s32 %p1, %r14, $S;
    @%p1 bra DONE_B;

    mul.wide.s32 %rd18, %r14, 4;
    add.s64 %rd19, %rd17, %rd18;
    ld.global.s32 %r15, [%rd19];
    // p_i by the INTEGER exp: no floating-point operation anywhere on this
    // path, and the result is already Q0.28, so there is no rescale either.
    sub.s32 %r16, %r11, %r15;           // m - s_i  (>= 0)
    // -> Q16.16. The temperature is a RUNTIME parameter: a real model's
    // softmax scale is `q_scale * k_scale / sqrt(d)`, an arbitrary number, not
    // the power of two the tests use. Hardcoding a shift made this kernel
    // usable only on synthetic data.
    //
    // KFix is therefore `C * 2^32`, NOT `C * 2^16`: the argument this feeds is
    // in Q16.16, and one factor of 2^16 is consumed by the shift below. The
    // synthetic tests pass `8 << 16`, which reproduces the old `shl 3` exactly.
    // `tools/ptx_bridge.py` passed `C * 2^16` and got a uniform softmax that
    // its own differential could not see -- see finding 06.
    mul.wide.s32 %rd50, %r16, %r50;
    add.s64 %rd50, %rd50, 32768;
    shr.s64 %rd50, %rd50, 16;
    min.s64 %rd50, %rd50, 1073741824;   // saturate; the exp returns 0 up here
    cvt.u32.u64 %r16, %rd50;
    call.uni (%r17), y_exp2_neg_q16_16, (%r16);

    add.s64 %rd26, %rd25, %rd18;
    st.global.s32 [%rd26], %r17;

    cvt.s64.s32 %rd20, %r17;
    add.s64 %rd33, %rd33, %rd20;        // level 1: a plain register add

    mul.wide.s32 %rd21, %r14, $D;
    add.s64 %rd22, %rd2, %rd21;         // &V[i][0]
    mov.u32 %r43, osm;
    mov.u32 %r18, 0;
LOOP_D:
    ld.global.s8 %r19, [%rd22];
    mul.wide.s32 %rd24, %r17, %r19;     // 35 bits: mul.lo.s32 would overflow
    red.shared.add.u64 [%r43], %rd24;   // level 2: shared, CTA-private
    add.s64 %rd22, %rd22, 1;
    add.s32 %r43, %r43, 8;
    add.s32 %r18, %r18, 1;
    setp.lt.s32 %p2, %r18, $D;
    @%p2 bra LOOP_D;

    add.s32 %r14, %r14, %r9;
    bra LOOP_I;
DONE_B:
    // Every thread reaches here -- the loop guard is at the top, so no thread
    // can return early and leave the barrier below short of arrivals.
    mov.u32 %r44, lsm;
    red.shared.add.u64 [%r44], %rd33;
    bar.sync 0;

    // Level 3: one global atomic per CTA per output element, instead of one
    // per (key, element).
    mov.u32 %r45, %r6;
FLOOP:
    setp.ge.s32 %p5, %r45, $D;
    @%p5 bra FDONE;
    mov.u32 %r46, osm;
    mad.lo.s32 %r46, %r45, 8, %r46;
    ld.shared.u64 %rd35, [%r46];
    mul.wide.s32 %rd36, %r45, 8;
    add.s64 %rd37, %rd15, %rd36;
    red.global.add.u64 [%rd37], %rd35;
    add.s32 %r45, %r45, %r5;
    bra FLOOP;
FDONE:
    setp.ne.s32 %p6, %r6, 0;
    @%p6 bra FSKIP;
    mov.u32 %r47, lsm;
    ld.shared.u64 %rd38, [%r47];
    red.global.add.u64 [%rd13], %rd38;
FSKIP:
    ret;
}

// The one-global-atomic-per-term version. Kept as a structurally different
// implementation of the same mathematics: it and `attn_accum` group the terms
// completely differently and must still agree bit for bit, which is a stronger
// statement than either agreeing with itself across geometries.
.visible .entry attn_accum_naive(
    .param .u64 q0,
    .param .u64 q1,
    .param .u64 q2,
    .param .u64 q3,
    .param .u64 q4,
    .param .u64 q5,
    .param .u32 q6
)
{
    .reg .pred %p<8>;
    .reg .s32  %r<64>;
    .reg .s64  %rd<64>;

    ld.param.u64 %rd1, [q0];
    ld.param.u64 %rd2, [q1];
    ld.param.u64 %rd3, [q2];
    ld.param.u64 %rd4, [q3];
    ld.param.u64 %rd5, [q4];
    ld.param.u64 %rd6, [q5];
    ld.param.u32 %r50, [q6];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;
    cvta.to.global.u64 %rd5, %rd5;
    cvta.to.global.u64 %rd6, %rd6;

    // Rendered from `sched_accum`, the SAME description `attn_accum` uses.
    // The two entries must partition the sequence identically or the
    // differential between them compares two different schedules.
$SCHED_NAIVE
    mov.u32 %r10, %ctaid.y;

    mul.wide.s32 %rd10, %r10, 4;
    add.s64 %rd11, %rd3, %rd10;
    ld.global.s32 %r11, [%rd11];
    xor.b32 %r11, %r11, -2147483648;    // undo `attn_scores`'s unsigned bias

    mul.wide.s32 %rd12, %r10, 8;
    add.s64 %rd13, %rd4, %rd12;
    mul.lo.s32 %r12, %r10, $D;
    mul.wide.s32 %rd14, %r12, 8;
    add.s64 %rd15, %rd5, %rd14;
    mul.lo.s32 %r13, %r10, $S;
    mul.wide.s32 %rd16, %r13, 4;
    add.s64 %rd17, %rd1, %rd16;
    add.s64 %rd25, %rd6, %rd16;

    // [Y SEQUENCE REDUCTION] grid-stride over S, stride = nworkers. The naive
    // entry has no zeroing loop to be confused with, but the marker is what
    // tests/exact_attention_schedule.rs looks for and both entries must carry
    // it - a marker present in one of two reductions is worse than none.
    mov.u32 %r14, %r7;
NLOOP_I:
    setp.ge.s32 %p1, %r14, $S;
    @%p1 bra NDONE_B;

    mul.wide.s32 %rd18, %r14, 4;
    add.s64 %rd19, %rd17, %rd18;
    ld.global.s32 %r15, [%rd19];
    // p_i by the INTEGER exp: no floating-point operation anywhere on this
    // path, and the result is already Q0.28, so there is no rescale either.
    sub.s32 %r16, %r11, %r15;           // m - s_i  (>= 0)
    // -> Q16.16. The temperature is a RUNTIME parameter: a real model's
    // softmax scale is `q_scale * k_scale / sqrt(d)`, an arbitrary number, not
    // the power of two the tests use. Hardcoding a shift made this kernel
    // usable only on synthetic data.
    //
    // KFix is therefore `C * 2^32`, NOT `C * 2^16`: the argument this feeds is
    // in Q16.16, and one factor of 2^16 is consumed by the shift below. The
    // synthetic tests pass `8 << 16`, which reproduces the old `shl 3` exactly.
    // `tools/ptx_bridge.py` passed `C * 2^16` and got a uniform softmax that
    // its own differential could not see -- see finding 06.
    mul.wide.s32 %rd50, %r16, %r50;
    add.s64 %rd50, %rd50, 32768;
    shr.s64 %rd50, %rd50, 16;
    min.s64 %rd50, %rd50, 1073741824;   // saturate; the exp returns 0 up here
    cvt.u32.u64 %r16, %rd50;
    call.uni (%r17), y_exp2_neg_q16_16, (%r16);

    add.s64 %rd26, %rd25, %rd18;
    st.global.s32 [%rd26], %r17;

    cvt.s64.s32 %rd20, %r17;
    red.global.add.u64 [%rd13], %rd20;

    mul.wide.s32 %rd21, %r14, $D;
    add.s64 %rd22, %rd2, %rd21;
    mov.u64 %rd23, %rd15;
    mov.u32 %r18, 0;
NLOOP_D:
    ld.global.s8 %r19, [%rd22];
    mul.wide.s32 %rd24, %r17, %r19;     // 35 bits: mul.lo.s32 would overflow
    red.global.add.u64 [%rd23], %rd24;
    add.s64 %rd22, %rd22, 1;
    add.s64 %rd23, %rd23, 8;
    add.s32 %r18, %r18, 1;
    setp.lt.s32 %p2, %r18, $D;
    @%p2 bra NLOOP_D;

    add.s32 %r14, %r14, %r9;
    bra NLOOP_I;
NDONE_B:
    ret;
}
"#;
