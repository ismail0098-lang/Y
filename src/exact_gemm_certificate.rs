//! The certificate a compilation carries with the kernel it emitted.
//!
//! `proofs/` is checked once, at build time, against the SHIPPED schedule
//! constants. Its theorems are universally quantified over `M`, `N`, `K` and
//! `nthr`, so every shape is covered - but a user who compiles their own
//! `@ZeroDrift` nest gets a fast kernel and no artifact. "Proof-carrying"
//! described the repository, not the output.
//!
//! This module closes that. When `llvm_emitter` substitutes the exact
//! `vpdpwssd` kernel it renders a `.v` file stating, at THIS compilation's
//! numbers, that the emitted kernel computes the source's dot products - by
//! instantiating [`ExactGemmWhole.the_threaded_gemm_holds_the_source_dot_products`]
//! rather than by re-proving anything. The certificate travels with the `.ll`.
//!
//! # Why this is not paperwork
//!
//! The interesting hypothesis is the LICENCE, `2 * Fl * m^2 <= i32::MAX`. Y
//! decides it in [`crate::zero_drift::VnniExact::max_operand_magnitude`] with
//! **floating-point arithmetic** - a `sqrt` and a `floor` on `f64`. The
//! certificate states it as **exact integer arithmetic over `Z`** and hands it
//! to `coqc`. So emitting the certificate makes the compiler's own
//! floating-point reasoning checkable by an independent integer tool, and a
//! certificate that fails to compile where Y said yes is a real finding rather
//! than a formality.
//!
//! Verified at the boundary before this module was written: at the default
//! interval the certificate is ACCEPTED at `m = 4095` and REFUSED at `m = 4096`
//! - the same one-unit edge `tests/exact_gemm_licence_obligations.rs` finds by
//! exhausting the int16 domain, reproduced here by a different tool.
//!
//! # The trust boundary, and why it is a LIST rather than a paragraph
//!
//! A certificate that overstates its scope is worse than none, so the emitted
//! file carries what it does not prove. That section used to be prose copied
//! by hand from `ExactGemmWhole.v`, and **the copy had dropped one of the
//! capstone's three bullets while adding one of its own** - so both lists were
//! three bullets long and a count would have called them equal. The dropped
//! one is the panel buffers' sizing and the scratch tile's allocation, which
//! is where both of this repository's documented out-of-bounds writes were.
//!
//! [`TRUST_BOUNDARY`] is now the single list, rendered into every certificate,
//! and each item carries a third field the prose never had: **whether anything
//! in this repository would FAIL if the item were false.** A caveat list says
//! what is not proved; a trust boundary says what would notice. Only the
//! second is worth handing to someone deciding whether to rely on the kernel.
//!
//! `tests/certificate_states_its_trust_boundary.rs` gates it, including the
//! direction that failed here: every bullet of the capstone's own exclusion
//! list must be claimed by exactly one item, so a proof file that adds an
//! exclusion cannot leave the certificate behind.

/// One substituted exact GEMM, and the numbers its certificate is about.
#[derive(Debug, Clone, PartialEq)]
pub struct Certificate {
    /// The operand magnitude the licence was granted against, as `@bounds`
    /// stated it - a real number.
    pub operand_magnitude: f64,
    /// k-pairs accumulated in int32 between flushes into int64.
    pub flush_k_pairs: u32,
    /// The extent names of the recognised nest, for the header only.
    pub extent_m: String,
    pub extent_n: String,
}

/// The integer bound the certificate is stated at, from the real-valued
/// `@bounds` magnitude.
///
/// **Rounded UP, and the direction matters in both of the places it is used.**
/// The certificate's theorem carries `m` in two roles at once:
///
/// - the LICENCE, `2 * Fl * m^2 <= i32::MAX`, which gets HARDER as `m` grows;
/// - the data hypothesis, `|A[i,k]| <= m`, which gets EASIER as `m` grows.
///
/// So `ceil` is the conservative direction for both: a stricter obligation
/// discharged, over every matrix the declared bound admits. `floor` would be
/// wrong twice - it would licence a wider interval than the source justifies
/// AND state the guarantee over fewer matrices than the source declares.
///
/// This cannot disagree with Y's own licence, and the argument is short enough
/// to write down. Y licenses exactly when `m <= L` where
/// `L = floor(sqrt(i32::MAX / (2*Fl)))` is an integer; `m <= L` with `L`
/// integral gives `ceil(m) <= L`, so the certificate's obligation holds
/// whenever Y granted the licence. It is checked rather than trusted: see
/// `tests/exact_gemm_certificate.rs`.
pub fn integer_bound(magnitude: f64) -> u32 {
    if !magnitude.is_finite() || magnitude <= 0.0 {
        return 0;
    }
    magnitude.ceil() as u32
}

/// The file whose exclusion list this certificate must mirror.
///
/// It is the CAPSTONE of the exact-GEMM chain - the proof nothing else
/// `Require`s - and that is not an arbitrary choice of file. A proof with
/// something above it cannot know what the programme still leaves open, so
/// only a dependency root can truthfully state a global negative; that rule is
/// derived and gated by `tests/proofs_are_checked.rs`. The capstone is
/// therefore exactly the file whose exclusion list is the AGGREGATE one, and
/// exactly the file a certificate instantiating it must not understate.
pub const CAPSTONE: &str = "proofs/ExactGemmWhole.v";

/// Whether anything in this repository can FAIL on a trust-boundary item.
///
/// Two grades rather than three, and the missing third one is deliberate.
/// "A test exercises it" is not a grade between these two unless the test can
/// fail on the thing. `tests/exact_gemm_thread_invariance.rs` runs the
/// threaded kernel at ragged shapes and compares answers; a panel buffer sized
/// wrongly in the safe direction changes no answer, and this repository has
/// found that exact shape twice - an over-allocation caught by the schedule
/// gate and by nothing else, and three `ldc` sites whose overrun was invisible
/// until `ExactGemmTiling.v` was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Check {
    /// Something here can fail on it. The string is the repo-relative file.
    Pinned(&'static str),
    /// Nothing here can. The string says what closing it would take, because
    /// an unchecked item with no route to closing it reads as an excuse rather
    /// than as a work item.
    Unchecked(&'static str),
}

/// One item of the certificate's trusted computing base.
///
/// The point of the struct is that `check` is not optional. A caveat list
/// says what is not proved; a trust boundary says, for each item, whether
/// anything would notice if it were false. Those are different documents and
/// only the second is worth handing to an auditor.
#[derive(Debug, Clone, Copy)]
pub struct TrustItem {
    /// What is trusted, as the certificate states it.
    pub claim: &'static str,
    /// Why no proof in `proofs/` supplies it.
    pub because: &'static str,
    /// Whether anything can fail on it.
    pub check: Check,
    /// The proof file stating this exclusion, and a phrase distinctive enough
    /// to locate its bullet. `None` for the items BELOW the model - a proof
    /// about `Z` has no opinion on a toolchain, so the capstone correctly does
    /// not mention one, and a certificate that omitted them would be silent
    /// about the half of its trust boundary a reader most needs.
    pub stated_in: Option<(&'static str, &'static str)>,
}

/// The trusted computing base of the emitted exact-GEMM certificate.
///
/// **This list existed nowhere.** It was prose in `ExactGemmWhole.v`, prose in
/// `ExactGemmMicro.v`, prose in `docs/proof_carrying_kernels.md`, and a
/// hand-copied subset in the certificate's own header - and the copy had
/// DROPPED one of the capstone's three bullets while adding one of its own, so
/// both lists were three long and a count would have called them equal. The
/// dropped one was the buffer sizing, which is where this repository's two
/// documented out-of-bounds writes were.
///
/// That item has since been **split by being half discharged**:
/// `ExactGemmAllocation.v` proves the sizes, and what is left of it is the
/// allocation failing. The split had to happen in the capstone and here
/// together, which is the bijection gate doing its job.
pub const TRUST_BOUNDARY: &[TrustItem] = &[
    TrustItem {
        claim: "`vpdpwssd`'s arithmetic, and the little-endian order of an i32's two halves.",
        because: "These are ISA facts. No proof over `Z` can supply one, so they are \
                  DEFINITIONS in `ExactGemmRegisterTile.v` rather than theorems, and every \
                  result above them is conditional on their being right.",
        check: Check::Pinned("tests/cpu_gemm_vnni_micro.rs"),
        stated_in: Some((CAPSTONE, "little-endian order of an i32")),
    },
    TrustItem {
        claim: "The structure of the emitted loop nest.",
        because: "The nest's ARITHMETIC is extracted - one description in `cpu_gemm::Ix` and \
                  `cpu_gemm::CountedLoop` is rendered both to the emitted IR and to these \
                  proofs, so a divergence is a byte-identity failure. Which loops exist, in \
                  what order, in which blocks and calling what is still hand-written, so for \
                  that part the tie is between two models rather than to the IR.",
        check: Check::Pinned("tests/exact_gemm_schedule_proof.rs"),
        stated_in: Some((CAPSTONE, "The loop STRUCTURE is modelled")),
    },
    TrustItem {
        claim: "There is no FALLBACK when an allocation fails: the kernel exits.",
        because: "The buffers' SIZES are proved - `ExactGemmAllocation.v` shows every slot \
                  the packers and the flush write lands inside the allocation, and that the \
                  allocation is not one element larger than the write set. Failure is \
                  DEFINED rather than proved: every allocation goes through \
                  `@__y_gemm_exact_alloc`, which prints the byte count and exits 1, so an \
                  out-of-memory condition is a diagnosis rather than a null dereference. \
                  What it cannot do is what the f32 kernel in this same emitter does - fall \
                  back to a static panel - because this kernel packs the WHOLE of A at \
                  once, which is the property that makes its packing asymptotically free \
                  and leaves its panel size unbounded in M and K.",
        check: Check::Pinned("tests/exact_gemm_allocation_failure.rs"),
        stated_in: Some((CAPSTONE, "packs the whole of A at once")),
    },
    TrustItem {
        claim: "The threading layer's LAYOUT: one job record per worker, one private C \
                buffer per worker, and a reduction that reads each of them once.",
        because: "Proved, in `ExactGemmThreading.v`. The job array is a `(thread, field)` \
                  positional index, so no worker's record can overlap another's and the \
                  array is exactly one record per thread; each private C buffer is exactly \
                  the rectangle the reduction reads back, at the worker's own compact \
                  stride rather than the caller's; and the emitted reduction loop over a \
                  zeroed destination IS the fold `ExactGemmKsplit.ksplit_exact` is about. \
                  Which OFFSET each field is written and read at is not a number and is \
                  checked against the emitted text instead.",
        check: Check::Pinned("tests/exact_gemm_threading_layout.rs"),
        stated_in: Some((CAPSTONE, "threading layer's LAYOUT is modelled")),
    },
    TrustItem {
        claim: "The CONCURRENCY those records are dispatched with: that a worker's stores \
                are visible to the reduction, and that the join loop can read the thread \
                ids the spawn loop left.",
        because: "Not modelled. Nothing here says `pthread_join` orders a worker's last \
                  store before the reducer's first load, nor that the spawn loop's \
                  inline-fallback arm leaves the thread-id array in a readable state - and \
                  that loop's `tid == 0` sentinel is an assumption about the C library's \
                  representation of a thread id rather than about any arithmetic. The \
                  behavioural cover is `tests/exact_gemm_thread_invariance.rs`, which \
                  compares ANSWERS across thread counts, so a dispatch bug that produces \
                  the right answer is invisible to it.",
        check: Check::Unchecked(
            "a memory model, which is a concurrency obligation rather than an arithmetic \
             one",
        ),
        stated_in: Some((CAPSTONE, "The CONCURRENCY is not")),
    },
    TrustItem {
        claim: "Everything below the LLVM IR this compilation emitted: `clang`, its optimiser, \
                the assembler and the linker.",
        because: "The proofs stop at the IR. Nothing here is a statement about the machine \
                  code, and the guarantee above is void if the translation to it is not \
                  faithful.",
        check: Check::Unchecked(
            "translation validation - checking THIS object against THIS IR per compilation, \
             which is not performed",
        ),
        stated_in: None,
    },
    TrustItem {
        claim: "Rocq's kernel, and the `coqc` that checks this file.",
        because: "A proof is worth exactly what its checker is worth.",
        check: Check::Pinned("tests/proofs_are_checked.rs"),
        stated_in: None,
    },
    TrustItem {
        claim: "The processor executes its own ISA as documented.",
        because: "Naming the bottom is what makes this list finite rather than open-ended. \
                  Errata are published for every part in production.",
        check: Check::Unchecked(
            "hardware validation, which no software check in any repository can \
             substitute for",
        ),
        stated_in: None,
    },
];

/// Render [`TRUST_BOUNDARY`] as the certificate's exclusion section.
///
/// Every item is rendered, including the unchecked ones - which is the whole
/// point, since a renderer that skipped them would turn the honest half of the
/// list into silence.
pub fn render_trust_boundary() -> String {
    let mut out = String::new();
    for item in TRUST_BOUNDARY {
        out.push_str(&wrap(item.claim, "    - ", "      "));
        out.push_str(&wrap(item.because, "      ", "      "));
        let line = match item.check {
            Check::Pinned(f) => format!("Checked by: {f}"),
            Check::Unchecked(what) => format!("NOT CHECKED. Closing it means {what}."),
        };
        out.push_str(&wrap(&line, "      ", "      "));
    }
    // The section is spliced into a Coq comment, so the trailing newline the
    // last bullet carries is the blank line before what follows.
    out
}

/// Greedy wrap to the width the rest of this file is written at.
fn wrap(text: &str, first: &str, rest: &str) -> String {
    const WIDTH: usize = 76;
    let mut out = String::new();
    let mut line = String::from(first);
    let mut empty = true;
    for word in text.split_whitespace() {
        if !empty && line.chars().count() + 1 + word.chars().count() > WIDTH {
            out.push_str(line.trim_end());
            out.push('\n');
            line = String::from(rest);
            empty = true;
        }
        if !empty {
            line.push(' ');
        }
        line.push_str(word);
        empty = false;
    }
    if !empty {
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

/// Render the certificate for one substituted kernel.
///
/// `source` is a label for the header only - it identifies the compilation to
/// a human and is not read by `coqc`. `stem` is the certificate's own file name
/// without its extension, which `coqc` turns into the module's logical name.
pub fn render(cert: &Certificate, source: &str, stem: &str) -> String {
    let m = integer_bound(cert.operand_magnitude);
    let fl = cert.flush_k_pairs;
    format!(
        r#"(** * EXACT-GEMM CERTIFICATE - GENERATED BY THE Y COMPILER, DO NOT EDIT.

    Source      : {source}
    Matmul      : {em} x {en}
    Operands    : |x| <= {mag} as declared by `@bounds`, certified at the
                  integer bound {m} (rounded UP: see
                  `exact_gemm_certificate::integer_bound`, where the direction
                  is argued for both roles `m` plays)
    Flush       : {fl} k-pairs accumulated in int32 between widenings into int64
    Schedule    : `ExactGemmSchedule.v`, generated from `src/cpu_gemm.rs`

    ** What this file asserts.

    [this_kernel_computes_the_source_dot_products]: for ANY A and B respecting
    the bound declared above, ANY M, N, K, ANY positive thread count, and every
    position of C, the kernel Y emitted for this compilation - packing, the
    register tile's routing, the k-pair loop, the int32 flush, the output
    tiling and the K-split across threads - produces exactly

    <<  sum over k < K of  A[r][k] * B[k][c]  >>

    the naive nest's own value, not a value close to it. Integer addition is
    associative and commutative, so this is an EQUALITY rather than a
    tolerance; that is the whole reason the exactness constraint is worth its
    range.

    ** The compilation-specific obligation.

    [the_licence_holds] is the only thing here that depends on this program.
    `vpdpwssd` accumulates into int32, which wraps silently; the emitted kernel
    is safe exactly when one flush interval of products cannot overflow it:

    <<  2 * {fl} * {m}^2 <= 2147483647  >>

    Y decides that in FLOATING POINT (a `sqrt` and a `floor` on `f64`). This
    file states it over `Z` and lets `coqc` check it, so the two derivations
    are independent. At the default interval the edge is one unit wide - `4095`
    is accepted and `4096` is refused - and a bound Y should have rejected
    makes this file fail to compile rather than certifying anything.

    ** THE TRUST BOUNDARY - what this certificate rests on and does not
       prove, with, for each item, whether anything in the compiler's own
       repository would FAIL if it were false.

       It is rendered from ONE list (`exact_gemm_certificate::TRUST_BOUNDARY`)
       rather than copied. It used to be copied, and the copy had silently
       dropped the buffer-sizing item while adding one of its own - both lists
       three bullets long, so a count called them equal.

{trust}
    Check with:

    <<  coqc -Q <dir with proofs/*.v> "" {stem}.v  >>
*)

From Stdlib Require Import ZArith Arith Lia.
Require ExactGemmWhole.
Require ExactGemmMicro.
Require ExactGemmPacking.
Open Scope Z_scope.

Module W := ExactGemmWhole.
Module PK := ExactGemmPacking.

(** The flush interval this compilation emitted. *)
Definition Fl : nat := {fl}.

(** The operand bound this compilation was licensed against. *)
Definition m : Z := {m}.

(** *** THE COMPILATION-SPECIFIC OBLIGATION. *)
Theorem the_licence_holds :
  2 * Z.of_nat Fl * m * m <= ExactGemmMicro.I32MAX.
Proof. unfold Fl, m, ExactGemmMicro.I32MAX. lia. Qed.

(** *** THE CERTIFICATE. *)
Theorem this_kernel_computes_the_source_dot_products :
  forall A B M N K nthr r c,
    (forall i k, Z.abs (A i k) <= m) ->
    (forall k j, Z.abs (B k j) <= m) ->
    (0 < nthr)%nat -> (r < M)%nat -> (c < N)%nat ->
    W.thread_sum A B M N K Fl nthr r c nthr
    = PK.sum_k (fun k => A r k * B k c) K.
Proof.
  intros A B M N K nthr r c HA HB Hn Hr Hc.
  apply (W.the_threaded_gemm_holds_the_source_dot_products
           A B M N K Fl m nthr r c); try assumption.
  - unfold Fl. lia.
  - unfold m. lia.
  - exact the_licence_holds.
Qed.

(** *** Non-vacuity.

    A theorem whose hypotheses cannot be satisfied is true and worthless, and
    that is a reachable failure here rather than a hypothetical one: a negative
    `m` makes `|A[i,k]| <= m` unsatisfiable and the certificate above
    vacuously true. So the certificate exhibits a matrix the declared bound
    admits, and EVALUATES the model on it - at two threads over an uneven
    K-split, which is the case `ExactGemmKsplit.bands_tile` exists for. *)
Definition Acert (i k : nat) : Z := m.
Definition Bcert (k j : nat) : Z := m.

Theorem the_declared_bound_admits_this_matrix :
  (forall i k, Z.abs (Acert i k) <= m) /\ (forall k j, Z.abs (Bcert k j) <= m).
Proof. unfold Acert, Bcert, m. split; intros; compute; discriminate. Qed.

Theorem the_certificate_is_not_vacuous :
  W.thread_sum Acert Bcert 2 3 3 Fl 2 1 2 2 = 3 * m * m.
Proof. vm_compute. reflexivity. Qed.

Print Assumptions the_licence_holds.
Print Assumptions this_kernel_computes_the_source_dot_products.
Print Assumptions the_certificate_is_not_vacuous.
"#,
        source = source,
        em = cert.extent_m,
        en = cert.extent_n,
        mag = cert.operand_magnitude,
        m = m,
        fl = fl,
        stem = module_stem(stem),
        trust = render_trust_boundary(),
    )
}

/// A Coq module name for `stem`, which is the certificate's file name without
/// its extension.
///
/// `coqc` derives the logical name from the file name, so it has to be a legal
/// identifier: alphanumerics and underscores, not starting with a digit. A
/// source called `4-bit gemm.ysu` is a perfectly ordinary file name and an
/// illegal module name, so this is a sanitiser rather than an assertion.
pub fn module_stem(stem: &str) -> String {
    let mut out: String = stem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    if out.is_empty() || out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rounding direction, at the values where it decides something.
    #[test]
    fn the_integer_bound_rounds_up() {
        assert_eq!(integer_bound(1024.0), 1024);
        assert_eq!(integer_bound(1024.5), 1025);
        assert_eq!(integer_bound(0.0), 0);
        // A magnitude below 1 is refused by `VnniExact::license` before it can
        // reach here; `0` is the honest answer rather than a rounded-up `1`,
        // which would certify a bound the source never stated.
        assert_eq!(integer_bound(f64::NAN), 0);
    }

    /// `ceil` cannot turn a licensed magnitude into an unlicensed integer
    /// bound. The argument is in `integer_bound`'s doc comment; this is the
    /// check, over every interval the scheme admits.
    #[test]
    fn rounding_up_never_breaks_a_licence_y_granted() {
        for shift in 0..12u32 {
            let scheme = crate::zero_drift::VnniExact::new(1u32 << shift).unwrap();
            let limit = scheme.max_operand_magnitude();
            for probe in [limit, limit - 0.5, limit - 1.0, 1.0, 2.5] {
                if probe < 1.0 || scheme.license(probe).is_err() {
                    continue;
                }
                let m = u64::from(integer_bound(probe));
                let fl = u64::from(scheme.flush_k_pairs);
                assert!(
                    2 * fl * m * m <= i32::MAX as u64,
                    "ceil({probe}) = {m} breaks the licence at flush {fl}, which \
                     `integer_bound`'s argument says is impossible"
                );
            }
        }
    }

    #[test]
    fn a_module_stem_is_a_legal_coq_identifier() {
        assert_eq!(module_stem("gemm"), "gemm");
        assert_eq!(module_stem("4bit-gemm"), "_4bit_gemm");
        assert_eq!(module_stem(""), "_");
    }
}
