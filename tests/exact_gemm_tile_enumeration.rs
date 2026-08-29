//! **Does the driver visit the tiles the model says it does?**
//!
//! `proofs/ExactGemmWhole.v`'s capstone says what position `(r, c)` receives
//! *on the reading that* the driver visits tile `(r / MR, c / NR)` at offset
//! `(r mod MR, c mod NR)`. `ExactGemmTiling.v` proves that enumeration is a
//! bijection onto C, and `cpu_gemm::mn_tiles` transcribes it - but until now
//! nothing asked the emitted driver how many tiles it actually runs, or in
//! what order. `exact_gemm_tiling_model.rs` checks the loops EXIST (by label)
//! and that the clamp is present; neither is the enumeration.
//!
//! **A correct tiling is invisible in the answer** - that is what
//! `c_written_exactly_once` says - so the answer cannot arbitrate this. The
//! same problem the K-split had, where the `--wrap=pthread_create` spawn count
//! is the only thing the model predicts that the running code reveals. Here the
//! observable is the sequence of micro-kernel invocations.
//!
//! `--wrap` does not work for it: `__y_gemm_micro_vnni` is called from the
//! driver in the SAME module, so the call is resolved at compile time and there
//! is no relocation for the linker to redirect. Instead the emitted module has
//! that definition **excised** and replaced by a `declare`, and the C driver
//! supplies a recording stub. The driver under test is byte-identical
//! otherwise.
//!
//! Three things are then pinned that no other test covers:
//!
//! 1. the number of calls is `mn_tiles(M, MR).len() * mn_tiles(N, NR).len()`;
//! 2. the ORDER is column-panel outer, row-panel inner - the sequence of row
//!    tile indices is `0..ntiles_m` repeated `ntiles_n` times, which is what
//!    makes "B is packed once per column panel" true rather than a comment;
//! 3. every call receives the **full** `kpairs = (K+1)/2` and `ldc = NR`.
//!    Point 3 is the machine-checked form of a fact an earlier session got
//!    wrong from structural resemblance: there is no kc-panel loop, so a tile
//!    is run over the whole K in one call.
//!
//! **What it isolates - MEASURED, and less than it first looked.** Five
//! mutations of the driver, `caught` = that suite fails:
//!
//! | mutation | tile_enum | tiling_model | thr-inv | exact-thr | packing | chain |
//! |---|---|---|---|---|---|---|
//! | row loop strides by `MR-1` | caught | caught | caught | caught | - | - |
//! | column loop strides by `2*NR` | caught | caught | caught | caught | - | - |
//! | micro-kernel handed half the k-pairs | caught | caught | caught | caught | - | - |
//! | scratch row stride is `ldc`, not `NR` | caught | caught | caught | caught | - | - |
//! | **one extra row panel past M** | **caught** | ok | caught | ok | ok | ok |
//!
//! The first four are caught by everything, so this file does NOT isolate the
//! obvious single-site errors - they change the answer, and the correctness
//! suites see that. The last row is the one it earns its place on: an extra
//! panel is clamped to zero width, so the fold-back writes nothing and the
//! ANSWER IS UNCHANGED. Four of the six suites miss it entirely. (`thr-inv`
//! catches it, but as a wrong answer or a crash from the resulting read past
//! the A panel buffer - not as "the driver ran a tile it should not have".)
//!
//! So the honest claim is: this covers enumeration errors that do not change
//! the answer, and diagnoses the rest by name instead of as a bad number.
//!
//! **The companion is `exact_gemm_packing_schedule.rs`**, which excises all
//! three callees instead of one and covers the driver's OTHER loop - the panels
//! it prepares rather than the tiles it visits. The two catch disjoint sets:
//! this file catches none of that file's six packing mutations, because
//! excising the micro-kernel alone leaves what the packers are handed, and in
//! what order, invisible.

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
/// The body ends at the first line that is exactly `}` - true of every function
/// this emitter writes, whose inner blocks are all indented.
fn excise_define(module: &str, name: &str, declare: &str) -> String {
    let head = format!("define void @{name}(");
    let start = module
        .find(&head)
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

#define CAP 4096

void __y_gemm_exact_vnni(const int16_t *A, const int16_t *B, int64_t *C,
                         long M, long N, long K, long lda, long ldb, long ldc,
                         int16_t *Apanel, int16_t *Bpanel, int64_t *Ctile);

static const int16_t *g_apanel;
static long g_stride;          /* int16 elements per row-panel */
static long g_calls;
static long g_row[CAP], g_ldc[CAP], g_kp[CAP];

/* The excised micro-kernel. Records the call and writes nothing - the answer
   is other tests' business; the ENUMERATION is this one's. */
void __y_gemm_micro_vnni(const int16_t *Ap, const int16_t *Bp, int64_t *C,
                         long ldc, long kpairs) {
    (void)Bp; (void)C;
    if (g_calls < CAP) {
        g_row[g_calls] = (long)(Ap - g_apanel) / g_stride;
        g_ldc[g_calls] = ldc;
        g_kp[g_calls]  = kpairs;
    }
    g_calls++;
}

int main(int argc, char **argv) {
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

    __y_gemm_exact_vnni(A, B, C, M, N, K, lda, ldb, ldc, Apanel, Bpanel, Ctile);

    printf("calls %ld\n", g_calls);
    long n = g_calls < CAP ? g_calls : CAP;
    for (long i = 0; i < n; i++)
        printf("call %ld %ld %ld\n", g_row[i], g_ldc[i], g_kp[i]);
    printf("DONE\n");
    return 0;
}
"#;

struct Calls {
    rows: Vec<i64>,
    ldcs: Vec<i64>,
    kps: Vec<i64>,
}

/// `tag` is per-test. Both tests here call this, and a shared directory name
/// means one `remove_dir_all` lands while the other is writing its IR - the
/// race this repo has now hit four times (the GPU `.ptx` harness,
/// `backend_differential.rs`, `exact_gemm_chain_model.rs`, here). It is a
/// property of the HELPER, not of any one test, so the tag belongs in the
/// signature rather than in a comment telling the next author to remember.
fn build(tag: &str) -> Option<std::path::PathBuf> {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return None;
    }
    if !host_has_vnni() {
        eprintln!("skipping: host CPU has no avx512_vnni");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("y_tileenum_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let module = y::cpu_gemm::emit_vnni_gemm_module(64);
    let patched = excise_define(
        &module,
        "__y_gemm_micro_vnni",
        "declare void @__y_gemm_micro_vnni(ptr noalias, ptr noalias, ptr noalias, i64, i64)",
    );
    assert!(
        !patched.contains("define void @__y_gemm_micro_vnni("),
        "the micro-kernel definition survived the excision"
    );
    assert!(
        patched.contains("call void @__y_gemm_micro_vnni"),
        "the driver no longer calls the micro-kernel, so there is nothing to count"
    );

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

fn run(exe: &std::path::Path, m: usize, n: usize, k: usize) -> Calls {
    let out = Command::new(exe)
        .args([m.to_string(), n.to_string(), k.to_string()])
        .output()
        .expect("run");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success() && text.contains("DONE"),
        "driver did not finish:\n{text}"
    );
    let mut c = Calls { rows: vec![], ldcs: vec![], kps: vec![] };
    for l in text.lines() {
        if let Some(v) = l.strip_prefix("call ") {
            let f: Vec<i64> = v.split_whitespace().map(|x| x.parse().unwrap()).collect();
            c.rows.push(f[0]);
            c.ldcs.push(f[1]);
            c.kps.push(f[2]);
        }
    }
    let declared: usize = text
        .lines()
        .find_map(|l| l.strip_prefix("calls "))
        .and_then(|v| v.trim().parse().ok())
        .expect("no `calls` line");
    assert_eq!(
        declared,
        c.rows.len(),
        "more micro-kernel calls than the recorder's capacity; widen CAP"
    );
    c
}

/// Ragged in every axis, plus the degenerate 1x1 and an exact multiple.
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (6, 64, 8),      // exactly one tile
    (53, 71, 33),    // both axes ragged
    (12, 128, 16),   // both exact multiples
    (7, 65, 9),      // one over a tile in both
    (30, 200, 5),
];

/// **The driver runs exactly the tiles `mn_tiles` names, in column-outer
/// order, each over the full K.**
#[test]
fn the_driver_enumerates_the_tiles_the_model_names() {
    let Some(exe) = build("enum") else { return };

    for &(m, n, k) in SHAPES {
        let calls = run(&exe, m, n, k);
        let ntm = mn_tiles(m, VNNI_MR).len();
        let ntn = mn_tiles(n, VNNI_NR).len();

        assert_eq!(
            calls.rows.len(),
            ntm * ntn,
            "M={m} N={n}: the driver ran {} tiles, `mn_tiles` says {ntm} x {ntn} \
             = {}. proofs/ExactGemmWhole.v assumes the driver visits tile \
             (r/MR, c/NR) for every position, and that assumption is what this \
             count checks.",
            calls.rows.len(),
            ntm * ntn
        );

        // Column-panel outer, row-panel inner: 0..ntm repeated ntn times.
        let expected: Vec<i64> = (0..ntn)
            .flat_map(|_| (0..ntm).map(|i| i as i64))
            .collect();
        assert_eq!(
            calls.rows, expected,
            "M={m} N={n}: the row-panel sequence is not `0..{ntm}` repeated \
             {ntn} times. B is packed once per COLUMN panel, which is only \
             correct if the column loop is the outer one."
        );

        // Every tile gets the whole K. There is no kc-panel loop.
        let kp = ((k + 1) / 2) as i64;
        assert!(
            calls.kps.iter().all(|&x| x == kp),
            "M={m} N={n} K={k}: not every call received kpairs = {kp}: {:?}. A \
             tile is run over the whole K in one call - if that stops being \
             true, `kc` in the proof series is no longer K and every sibling \
             file's statement changes.",
            calls.kps
        );
        assert!(
            calls.ldcs.iter().all(|&x| x == VNNI_NR as i64),
            "M={m} N={n}: the scratch tile's row stride is not NR = {VNNI_NR}: \
             {:?}. The micro-kernel writes a full MR x NR tile into scratch; a \
             different stride there is an out-of-bounds write, not a wrong \
             number.",
            calls.ldcs
        );
    }
}

/// The control. Every assertion above is satisfied by a driver that runs no
/// tiles at all on some shape, so at least one shape must be non-trivial in
/// both axes - and the whole sweep must actually invoke the kernel.
#[test]
fn the_enumeration_sweep_is_not_vacuous() {
    let Some(exe) = build("vacuity") else { return };

    let mut total = 0usize;
    let mut multi_tile_shapes = 0usize;
    for &(m, n, k) in SHAPES {
        let calls = run(&exe, m, n, k);
        total += calls.rows.len();
        if mn_tiles(m, VNNI_MR).len() > 1 && mn_tiles(n, VNNI_NR).len() > 1 {
            multi_tile_shapes += 1;
        }
    }
    assert!(total > 0, "no micro-kernel call was recorded at any shape");
    assert!(
        multi_tile_shapes >= 2,
        "only {multi_tile_shapes} shape(s) tile in BOTH axes, so the \
         column-outer ordering assertion is nearly vacuous"
    );
}
