//! **Is A really packed once, and B once per column panel?**
//!
//! `emit_vnni_gemm_driver`'s own comment states the schedule as a measured
//! fact and nothing checks it:
//!
//! > The first version packed A inside the i-loop and B inside the j-loop
//! > nested within it, so B was re-packed once per ROW panel - `M/MR` times
//! > over, which is `M*N*K/MR` packing work against the `M*N*K` of arithmetic.
//! > At MR=6 that is a sixth of the total run spent copying B.
//!
//! **A re-packing bug is invisible in the answer.** Packing A inside the
//! j-loop, or B inside the i-loop, computes exactly the right result - it just
//! does `ntiles_n` or `ntiles_m` times the packing work. So no correctness
//! test in this repo can fail on it, and the only thing standing between the
//! current schedule and the one it replaced is that paragraph of prose. Same
//! shape as `a_deep_constant_chain_collapses_completely` (a convergence
//! property, guarded by a SIZE assertion) and as the shared-memory swizzle
//! (a bank-conflict property, guarded by a measurement).
//!
//! The observable is the same one `exact_gemm_tile_enumeration.rs` uses, for
//! the same reason: `--wrap` cannot redirect an INTRA-MODULE call, so the
//! emitted module has all three callees - `__y_gemm_vnni_pack_a`,
//! `__y_gemm_vnni_pack_b`, `__y_gemm_micro_vnni` - **excised** and replaced by
//! `declare`s, and the C driver supplies recording stubs. The driver under
//! test is byte-identical otherwise. The two packers are `internal`, so the
//! excision has to match `define internal void @...` as well as `define void`;
//! a `declare` is never `internal`.
//!
//! Recording all three in ONE event stream is what makes the schedule
//! checkable rather than three separate counts. The whole of it is one
//! assertion:
//!
//! ```text
//!   [A(0) .. A(ntm-1)]  ++  concat over j of ( [B(j)] ++ [K(0) .. K(ntm-1)] )
//! ```
//!
//! - every A pack precedes every B pack (A is packed before either loop);
//! - there are `ntm` of them, not `ntm * ntn`;
//! - there are `ntn` B packs, not `ntn * ntm`;
//! - each B pack is followed by exactly `ntm` micro-kernel calls before the
//!   next one, which is what makes "B is packed once per column panel" mean
//!   the panel is LIVE for exactly those tiles rather than merely being
//!   written that many times.
//!
//! **What it isolates - MEASURED, and wider than predicted.** Six mutations
//! of the driver, `caught` = that suite fails. Each mutation was applied to
//! `src/cpu_gemm.rs`, the build checked, and every suite run SEPARATELY -
//! `cargo test` aborts the remaining binaries after one fails, so a run that
//! lists several `--test` targets can leave the important one unmeasured.
//!
//! | mutation | pack_sched | tile_enum | tiling_model | thr-inv | exact-thr | packing_model |
//! |---|---|---|---|---|---|---|
//! | **A packed inside the j-loop** | **caught** | ok | ok | ok | ok | ok |
//! | **B packed inside the i-loop** | **caught** | ok | ok | ok | ok | ok |
//! | **pack_a's row count unclamped (`MR`)** | **caught** | ok | ok | ok | ok | ok |
//! | **pack_b's column count unclamped (`NR`)** | **caught** | ok | ok | ok | ok | ok |
//! | A panel offset is the row index, not the tile index | caught | ok | caught | caught | caught | caught |
//! | both packers handed `K-1` | caught | ok | caught | caught | caught | caught |
//!
//! **Four of the six are caught by this file and nothing else**, and the two I
//! did not expect are the more serious pair. Dropping either packer's clamp at
//! the CALL SITE makes it read up to `MR - 1` rows past the end of `A`, or
//! `NR - 1` columns past the end of `B` - and every answer stays bit-identical,
//! because the FOLD-BACK's own `mw`/`nw` clamp discards exactly the rows and
//! columns the packer over-read. So the observable consequence of a live
//! out-of-bounds READ is nothing at all, until the buffer happens to end at a
//! page boundary. That is the redundant-guard pattern `exact_gemm_packing_model`
//! already records for the masks INSIDE the packers, one layer up at the call
//! site, and it is why "the correctness suites cover the packers" is false.
//!
//! **`exact_gemm_tile_enumeration` catches none of the six**, which is the
//! result that says the two files are complementary rather than overlapping:
//! it excises the micro-kernel only, so what the packers are handed and in what
//! order is invisible to it. Between them they cover the driver's two loops -
//! that one the tiles it visits, this one the panels it prepares.

use std::process::Command;

use y::cpu_gemm::{mn_tiles, VNNI_MR, VNNI_NR};

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn host_has_vnni() -> bool {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.contains("avx512_vnni"))
        .unwrap_or(false)
}

/// Remove one top-level `define` by name, leaving a `declare` in its place.
///
/// Matches the DEFINITION rather than the first mention, so a call site above
/// it cannot be mistaken for one; and it accepts `define internal void @f(` as
/// well as `define void @f(`, because both packers are `internal`. The body
/// ends at the first line that is exactly `}` - true of every function this
/// emitter writes, whose inner blocks are all indented.
fn excise_define(module: &str, name: &str, declare: &str) -> String {
    let needle = format!("@{name}(");
    let start = module
        .match_indices(&needle)
        .map(|(i, _)| {
            let ls = module[..i].rfind('\n').map(|p| p + 1).unwrap_or(0);
            (ls, i)
        })
        .find(|&(ls, i)| module[ls..i].trim_start().starts_with("define"))
        .map(|(ls, _)| ls)
        .unwrap_or_else(|| panic!("{name} is not defined in the emitted module"));
    let rest = &module[start..];
    let end_rel = rest
        .match_indices("\n}")
        .next()
        .map(|(i, _)| i + 2)
        .unwrap_or_else(|| panic!("{name}'s body has no closing brace"));
    let mut out = String::with_capacity(module.len());
    out.push_str(&module[..start]);
    out.push_str(declare);
    out.push('\n');
    out.push_str(&module[start + end_rel..]);
    out
}

/// The schedule constants, as C `#define`s taken from `cpu_gemm.rs` itself.
///
/// **This driver used to hardcode `#define MR 6` / `#define NR 64`.** That is a
/// second copy of the tile shape, in the half of the harness that allocates the
/// buffers the emitted kernel writes into - so a change to `VNNI_MR` did not
/// make this test report a schedule mismatch, it made the test disagree with
/// itself and crash or mis-size a panel. Found by diagnosing exactly that: with
/// `VNNI_MR = 8`, `exact_gemm_thread_invariance` (which checks the ANSWER)
/// passes, while seven harnesses carrying their own `6` fail.
///
/// Same defect `proofs/ExactGemmSchedule.v` exists to remove, one layer down.
fn schedule_defines() -> String {
    format!("#define MR {VNNI_MR}\n#define NR {VNNI_NR}\n")
}

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define CAP 8192

void __y_gemm_exact_vnni(const int16_t *A, const int16_t *B, int64_t *C,
                         long M, long N, long K, long lda, long ldb, long ldc,
                         int16_t *Apanel, int16_t *Bpanel, int64_t *Ctile);

static const int16_t *g_apanel, *g_bpanel;
static long g_stride;              /* int16 elements per row-panel of A */
static long g_n;
static char g_kind[CAP];           /* 'A', 'B' or 'K' */
static long g_off[CAP];            /* destination offset from the panel base */
static long g_width[CAP];          /* mrows / ncols / ldc */
static long g_depth[CAP];          /* kc / kc / kpairs */

static void rec(char kind, long off, long width, long depth) {
    if (g_n < CAP) {
        g_kind[g_n] = kind; g_off[g_n] = off;
        g_width[g_n] = width; g_depth[g_n] = depth;
    }
    g_n++;
}

/* The three excised callees. None writes anything: the ANSWER is other tests'
   business, the SCHEDULE is this one's. Leaving the panels as the driver's
   caller allocated them (calloc, so zero) is safe for exactly that reason. */
void __y_gemm_vnni_pack_a(const int16_t *src, long lda, long mrows, long kc,
                          int16_t *dst) {
    (void)src; (void)lda;
    rec('A', (long)(dst - g_apanel), mrows, kc);
}

void __y_gemm_vnni_pack_b(const int16_t *src, long ldb, long kc, long ncols,
                          int16_t *dst) {
    (void)src; (void)ldb;
    rec('B', (long)(dst - g_bpanel), ncols, kc);
}

void __y_gemm_micro_vnni(const int16_t *Ap, const int16_t *Bp, int64_t *C,
                         long ldc, long kpairs) {
    (void)Bp; (void)C;
    rec('K', (long)(Ap - g_apanel) / g_stride, ldc, kpairs);
}

int main(int argc, char **argv) {
    (void)argc;
    long M = atol(argv[1]), N = atol(argv[2]), K = atol(argv[3]);
    long lda = K + 5, ldb = N + 9, ldc = N + 7;
    long kp = (K + 1) / 2;
    long ntm = (M + MR - 1) / MR;

    int16_t *A = calloc((size_t)M * lda, 2);
    int16_t *B = calloc((size_t)K * ldb, 2);
    int64_t *C = calloc((size_t)M * ldc, 8);

    g_stride = kp * MR * 2;
    int16_t *Apanel = calloc((size_t)ntm * g_stride, 2);
    int16_t *Bpanel = calloc((size_t)kp * NR * 2, 2);
    int64_t *Ctile  = calloc((size_t)MR * NR, 8);
    g_apanel = Apanel;
    g_bpanel = Bpanel;

    __y_gemm_exact_vnni(A, B, C, M, N, K, lda, ldb, ldc, Apanel, Bpanel, Ctile);

    printf("events %ld\n", g_n);
    long n = g_n < CAP ? g_n : CAP;
    for (long i = 0; i < n; i++)
        printf("ev %c %ld %ld %ld\n", g_kind[i], g_off[i], g_width[i], g_depth[i]);
    printf("DONE\n");
    return 0;
}
"#;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct Ev {
    kind: char,
    off: i64,
    width: i64,
    depth: i64,
}

/// `tag` is per-test, in the SIGNATURE rather than in a comment asking the next
/// author to remember. Two tests sharing a temp directory name means one
/// `remove_dir_all` lands while the other is writing its IR - the race this
/// repo has now hit four times.
fn build(tag: &str) -> Option<std::path::PathBuf> {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return None;
    }
    if !host_has_vnni() {
        eprintln!("skipping: host CPU has no avx512_vnni");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("y_packsched_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let module = y::cpu_gemm::emit_vnni_gemm_module(64);
    let mut patched = module;
    for (name, decl) in [
        (
            "__y_gemm_vnni_pack_a",
            "declare void @__y_gemm_vnni_pack_a(ptr noalias, i64, i64, i64, ptr noalias)",
        ),
        (
            "__y_gemm_vnni_pack_b",
            "declare void @__y_gemm_vnni_pack_b(ptr noalias, i64, i64, i64, ptr noalias)",
        ),
        (
            "__y_gemm_micro_vnni",
            "declare void @__y_gemm_micro_vnni(ptr noalias, ptr noalias, ptr noalias, i64, i64)",
        ),
    ] {
        patched = excise_define(&patched, name, decl);
        assert!(
            !patched.contains(&format!("void @{name}(%")),
            "{name}'s definition survived the excision"
        );
        assert!(
            patched.contains(&format!("call void @{name}")),
            "nothing calls @{name} any more, so there is no schedule to record"
        );
    }

    let ll = dir.join("m.ll");
    std::fs::write(&ll, patched).expect("write IR");
    let drv = dir.join("drv.c");
    std::fs::write(&drv, schedule_defines() + DRIVER).expect("write driver");
    let exe = dir.join("run");
    let cc = Command::new("clang")
        .args(["-O2", "-x", "ir"])
        .arg(&ll)
        .args(["-x", "c"])
        .arg(&drv)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("clang");
    assert!(
        cc.status.success(),
        "link failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );
    Some(exe)
}

fn run(exe: &std::path::Path, m: usize, n: usize, k: usize) -> Vec<Ev> {
    let out = Command::new(exe)
        .args([m.to_string(), n.to_string(), k.to_string()])
        .output()
        .expect("run");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success() && text.contains("DONE"),
        "driver did not finish:\n{text}"
    );
    let evs: Vec<Ev> = text
        .lines()
        .filter_map(|l| l.strip_prefix("ev "))
        .map(|v| {
            let f: Vec<&str> = v.split_whitespace().collect();
            Ev {
                kind: f[0].chars().next().unwrap(),
                off: f[1].parse().unwrap(),
                width: f[2].parse().unwrap(),
                depth: f[3].parse().unwrap(),
            }
        })
        .collect();
    let declared: usize = text
        .lines()
        .find_map(|l| l.strip_prefix("events "))
        .and_then(|v| v.trim().parse().ok())
        .expect("no `events` line");
    assert_eq!(
        declared,
        evs.len(),
        "more events than the recorder's capacity; widen CAP"
    );
    evs
}

/// Ragged in every axis, plus the degenerate and the exactly-divisible cases.
/// The middle three tile in BOTH axes, which is what the two headline
/// mutations need in order to be visible at all.
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (6, 64, 8),    // exactly one tile in each axis
    (53, 71, 33),  // both axes ragged
    (12, 128, 16), // both exact multiples
    (7, 65, 9),    // one over a tile in both
    (30, 200, 5),
];

/// **A is packed once up front; B is packed once per column panel, and its
/// panel is live for exactly that panel's tiles.**
#[test]
fn the_driver_packs_a_once_and_b_once_per_column_panel() {
    let Some(exe) = build("sched") else { return };

    for &(m, n, k) in SHAPES {
        let evs = run(&exe, m, n, k);
        let atiles = mn_tiles(m, VNNI_MR);
        let btiles = mn_tiles(n, VNNI_NR);
        let (ntm, ntn) = (atiles.len(), btiles.len());
        let kp = ((k + 1) / 2) as i64;
        let stride = kp * (VNNI_MR as i64) * 2;

        // The whole schedule, as one sequence.
        let mut want: Vec<(char, i64)> = (0..ntm).map(|i| ('A', i as i64 * stride)).collect();
        for _ in 0..ntn {
            want.push(('B', 0));
            want.extend((0..ntm).map(|i| ('K', i as i64)));
        }
        let got: Vec<(char, i64)> = evs.iter().map(|e| (e.kind, e.off)).collect();
        assert_eq!(
            got,
            want,
            "M={m} N={n} K={k}: the packing schedule is not \
             [A x {ntm}] then {ntn} x ([B] then [K x {ntm}]).\n\
             `A` appearing {} times instead of {ntm} means A is packed inside \
             the j-loop; `B` appearing {} times instead of {ntn} means B is \
             re-packed per ROW panel, which is the regression \
             emit_vnni_gemm_driver's comment records - a sixth of the run at \
             MR={VNNI_MR}, and BIT-IDENTICAL output either way.",
            got.iter().filter(|e| e.0 == 'A').count(),
            got.iter().filter(|e| e.0 == 'B').count(),
        );

        // Each A pack gets its own tile's clamped row count and the whole K.
        let apacks: Vec<&Ev> = evs.iter().filter(|e| e.kind == 'A').collect();
        for (t, (e, &(_, w))) in apacks.iter().zip(atiles.iter()).enumerate() {
            assert_eq!(
                (e.width, e.depth),
                (w as i64, k as i64),
                "M={m} K={k}: A panel {t} was packed as {} rows x {} k, want \
                 {w} x {k}. An unclamped row count reads past the last row of \
                 A; a short `kc` drops k-values from every tile in the panel.",
                e.width,
                e.depth
            );
        }

        // Each B pack gets its own panel's clamped column count and the whole K.
        let bpacks: Vec<&Ev> = evs.iter().filter(|e| e.kind == 'B').collect();
        for (t, (e, &(_, w))) in bpacks.iter().zip(btiles.iter()).enumerate() {
            assert_eq!(
                (e.width, e.depth),
                (w as i64, k as i64),
                "N={n} K={k}: B panel {t} was packed as {} columns x {} k, want \
                 {w} x {k}.",
                e.width,
                e.depth
            );
        }

        // The A panels are disjoint, and the micro-kernel reads the panel the
        // packer wrote. Both fall out of the sequence above; asserted
        // separately so a failure names the memory property rather than the
        // schedule.
        let mut offs: Vec<i64> = apacks.iter().map(|e| e.off).collect();
        offs.sort_unstable();
        offs.dedup();
        assert_eq!(
            offs.len(),
            ntm,
            "M={m}: two A panels share a destination offset, so one row panel \
             overwrites another's packed data"
        );
        assert!(
            bpacks.iter().all(|e| e.off == 0),
            "N={n}: a B pack wrote somewhere other than the base of Bpanel, \
             which is a single reused buffer - the caller allocates \
             ceil(K/2) * NR * 2 int16 and nothing more"
        );
    }
}

/// The control. The schedule assertion is satisfied trivially whenever
/// `ntm == 1` (A-once and A-per-column-panel coincide) or `ntn == 1` (B-once
/// and B-per-row-panel coincide), so the sweep must contain shapes that tile
/// in BOTH axes - otherwise neither headline mutation is observable.
#[test]
fn the_schedule_sweep_is_not_vacuous() {
    let Some(exe) = build("vacuity") else { return };

    let mut both = 0usize;
    let mut packs = 0usize;
    for &(m, n, k) in SHAPES {
        let evs = run(&exe, m, n, k);
        packs += evs.iter().filter(|e| e.kind != 'K').count();
        if mn_tiles(m, VNNI_MR).len() > 1 && mn_tiles(n, VNNI_NR).len() > 1 {
            both += 1;
        }
    }
    assert!(packs > 0, "no packing call was recorded at any shape");
    assert!(
        both >= 2,
        "only {both} shape(s) tile in BOTH axes, so `A packed once` and \
         `A packed per column panel` are the same assertion across the sweep \
         and neither re-packing mutation could be seen"
    );
}
