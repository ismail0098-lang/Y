//! The exact `vpdpwssd` micro-kernel must be exact AND order-independent.
//!
//! This is the first piece of `docs/deterministic_inference.md`'s M0 that
//! computes anything, and it is the piece the whole thesis rests on:
//!
//! > Integer addition is associative, so any reordering of the reduction — a
//! > different tile, a different K-split, a different thread count, a different
//! > batch size — produces a **bit-identical** result.
//!
//! Two properties are checked, and they are genuinely different:
//!
//! 1. **Exactness** — the kernel agrees with a scalar reference, bit for bit,
//!    at sizes chosen to exercise the flush boundary (shorter than one
//!    interval, exactly one, and several partial tails).
//! 2. **Order independence** — splitting the same k-range into different chunks
//!    and accumulating into the same `C` gives a byte-identical result. This is
//!    the *product* claim; exactness alone does not imply it, because a kernel
//!    could be exact and still depend on how the range was cut if the flush
//!    dropped a partial interval.
//!
//! **The control is what makes (2) mean anything.** A test that merely observes
//! "the splits agreed" would pass on a kernel that ignored its inputs, and —
//! more plausibly — on data too benign to expose reassociation at all. So the
//! same products are also summed in `f32` under the same splits, and that
//! *must* disagree. Without it this file would have kept passing when the
//! interesting property was gone, which is the failure mode `INTT(NTT(a)) == a`
//! and the shared-memory barrier race both had.
//!
//! Requires `clang` and a CPU with AVX512-VNNI; skipped with a notice
//! otherwise, like the `ptxas` and `solc` gates elsewhere in the suite.
//!
//! Run with:  cargo test --test cpu_gemm_vnni_micro -- --nocapture

use std::path::PathBuf;
use std::process::Command;
use y::cpu_gemm::{VNNI_MR, VNNI_NR};

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// AVX512-VNNI is required to *run* the kernel. Emitting and assembling it does
/// not need the host to have it, but every assertion here is about the numbers
/// it produces, so a host without it can only skip.
fn host_has_vnni() -> bool {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.contains("avx512_vnni"))
        .unwrap_or(false)
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void __y_gemm_micro_vnni(const int32_t*, const int16_t*, int64_t*, long, long);

static unsigned s;
static void seed(unsigned v){ s=v; }
static int rnd(int lo,int hi){ s=s*1103515245u+12345u; return lo+(int)((s>>16)%(unsigned)(hi-lo+1)); }

static int16_t *A, *B;
static void fill_at(long kpairs, unsigned sd, int mag){
  seed(sd);
  for(long t=0;t<kpairs*MR*2;t++) A[t]=(int16_t)rnd(-mag,mag);
  for(long t=0;t<kpairs*NR*2;t++) B[t]=(int16_t)rnd(-mag,mag);
}
static void fill(long kpairs, unsigned sd){ fill_at(kpairs,sd,1024); }

/* Every operand at the licence's exact maximum, alternating sign so nothing
   cancels. This is the worst case the bound admits: one flush interval holds
   2*64 products of 4095^2 = 2,146,435,200, which is 1,048,447 below INT32_MAX.
   It is the empirical check on the derivation in VnniExact::max_operand_magnitude. */
static void fill_worst(long kpairs, int mag){
  for(long t=0;t<kpairs*MR*2;t++) A[t]=(int16_t)mag;
  for(long t=0;t<kpairs*NR*2;t++) B[t]=(int16_t)mag;
}

/* C[i][j] = sum_k A[i][k]*B[k][j], with k running over the pair (2p, 2p+1).
   Written independently of the kernel's layout reasoning so that a layout bug
   shows up as a mismatch rather than being shared by both sides. */
static void reference(long kpairs, int64_t *R){
  memset(R,0,MR*NR*8);
  for(long p=0;p<kpairs;p++)
    for(int i=0;i<MR;i++){
      int64_t a0=A[(p*MR+i)*2+0], a1=A[(p*MR+i)*2+1];
      for(int v=0;v<4;v++)
        for(int l=0;l<16;l++){
          int64_t b0=B[p*NR*2+v*32+2*l], b1=B[p*NR*2+v*32+2*l+1];
          R[i*NR+v*16+l]+=a0*b0+a1*b1;
        }
    }
}

/* The control, and it has to mirror the kernel's TWO-LEVEL structure to mean
   anything.

   The first version of this simply summed p ascending in f32 and chunked the
   loop — and it did not drift, correctly, because chunking a sequential loop
   reorders nothing. It adds the same terms in the same order either way.

   The reassociation the kernel actually performs is between its two levels: a
   narrow accumulator holds a run of products and is flushed into a wide one at
   every interval boundary. Splitting the k-range moves those boundaries (the
   flush index is call-relative, so a 7-pair call flushes at its tail), which
   regroups the outer additions. Integer addition is associative so the regroup
   is invisible; f32 addition is not, so mirroring the structure here is what
   makes the control able to fail. */
static void reference_f32_2level(long off, long kpairs, float *R, long flush){
  static float acc[MR*NR];
  memset(acc,0,sizeof acc);
  for(long q=0;q<kpairs;q++){
    long p=off+q;
    for(int i=0;i<MR;i++){
      float a0=A[(p*MR+i)*2+0], a1=A[(p*MR+i)*2+1];
      for(int v=0;v<4;v++)
        for(int l=0;l<16;l++){
          float b0=B[p*NR*2+v*32+2*l], b1=B[p*NR*2+v*32+2*l+1];
          acc[i*NR+v*16+l]+=a0*b0+a1*b1;
        }
    }
    if((q%flush)==flush-1){ for(int t=0;t<MR*NR;t++){ R[t]+=acc[t]; acc[t]=0.0f; } }
  }
  for(int t=0;t<MR*NR;t++) R[t]+=acc[t];
}

int main(void){
  const long MAXK=512;
  A=aligned_alloc(64,(size_t)MAXK*MR*2*2);
  B=aligned_alloc(64,(size_t)MAXK*NR*2*2);
  int64_t *C=aligned_alloc(64,MR*NR*8), *R=aligned_alloc(64,MR*NR*8);
  int fail=0;

  /* 1 and 63 are shorter than one 64-pair flush interval, so ONLY the tail
     flush fires; 64 is exactly one; 65/100/193 leave a partial tail. */
  long sizes[]={1,63,64,65,100,128,193,512};
  for(unsigned i=0;i<sizeof(sizes)/sizeof(*sizes);i++){
    long k=sizes[i];
    fill(k, 1234567u+(unsigned)k);
    memset(C,0,MR*NR*8);
    __y_gemm_micro_vnni((const int32_t*)A,B,C,NR,k);
    reference(k,R);
    int bad=0;
    for(int t=0;t<MR*NR;t++) if(C[t]!=R[t]) bad++;
    printf("EXACT kpairs=%ld differ=%d\n",k,bad);
    if(bad) fail=1;
  }

  /* The licence boundary. At |x| = 1024 over 512 pairs the whole reduction is
     only 2^30, so it fits int32 outright and a kernel with NO flush at all
     would still be exact — the cases above cannot see a missing flush. At the
     licensed maximum the unflushed total is ~1.7e10, far past int32, so this
     is the case that actually exercises the int32->int64 flush AND confirms
     the derived bound of 4095 is safe on real hardware rather than only on
     paper. */
  {
    long k=512;
    fill_worst(k,4095);
    memset(C,0,MR*NR*8);
    __y_gemm_micro_vnni((const int32_t*)A,B,C,NR,k);
    reference(k,R);
    int bad=0;
    for(int t=0;t<MR*NR;t++) if(C[t]!=R[t]) bad++;
    printf("BOUND mag=4095 kpairs=%ld differ=%d\n",k,bad);
    if(bad) fail=1;
  }

  /* Order independence over one fixed input. */
  const long K=512;
  fill(K, 99991u);
  int64_t *ref=aligned_alloc(64,MR*NR*8); memset(ref,0,MR*NR*8);
  __y_gemm_micro_vnni((const int32_t*)A,B,ref,NR,K);

  long splits[][16]={ {256,256,0},{64,64,64,64,64,64,64,64,0},{1,511,0},
                      {100,200,212,0},{511,1,0},{7,13,101,391,0} };
  for(unsigned si=0; si<sizeof(splits)/sizeof(*splits); si++){
    memset(C,0,MR*NR*8);
    long off=0;
    for(int j=0;splits[si][j];j++){
      long n=splits[si][j];
      __y_gemm_micro_vnni((const int32_t*)(A+off*MR*2), B+off*NR*2, C, NR, n);
      off+=n;
    }
    int d=memcmp(C,ref,MR*NR*8)!=0;
    printf("SPLIT %u differs=%d\n",si,d);
    if(d) fail=1;
  }

  /* The control: the same two-level structure over the same data and the same
     splits, with an f32 accumulator instead of int32->int64. */
  {
    float *f1=calloc(MR*NR,sizeof(float)), *f2=calloc(MR*NR,sizeof(float));
    reference_f32_2level(0,K,f1,64);
    long parts[]={7,13,101,391};
    long off=0;
    for(unsigned j=0;j<4;j++){ reference_f32_2level(off,parts[j],f2,64); off+=parts[j]; }
    int diff=0;
    for(int t=0;t<MR*NR;t++) if(f1[t]!=f2[t]) diff++;
    printf("CONTROL f32 differ=%d of %d\n",diff,MR*NR);
    if(diff==0) fail=1;   /* a benign dataset would make the result vacuous */
    free(f1); free(f2);
  }

  printf("%s\n", fail?"RESULT FAIL":"RESULT PASS");
  return fail;
}
"#;

#[test]
fn the_exact_micro_kernel_is_exact_and_order_independent() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    if !host_has_vnni() {
        eprintln!("skipping: host CPU has no avx512_vnni");
        return;
    }

    let dir = std::env::temp_dir().join(format!("y_vnni_micro_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    // The IR comes from the compiler library, not a checked-in copy, so this
    // test tracks the emitter rather than a snapshot of it.
    let ir_path = dir.join("micro.ll");
    std::fs::write(
        &ir_path,
        y::cpu_gemm::emit_vnni_micro_module(y::zero_drift::VnniExact::DEFAULT_FLUSH_K_PAIRS),
    )
    .expect("write IR");

    let obj = dir.join("micro.o");
    let cc = Command::new("clang")
        .args(["-O2", "-c", "-x", "ir"])
        .arg(&ir_path)
        .arg("-o")
        .arg(&obj)
        .output()
        .expect("run clang on the emitted IR");
    assert!(
        cc.status.success(),
        "the emitted IR did not compile:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    // The instruction must actually survive to machine code. If LLVM scalarised
    // the intrinsic, every numeric assertion below would still pass and the
    // kernel would simply be slow — so this is checked separately.
    let asm = Command::new("clang")
        .args(["-O2", "-S", "-x", "ir"])
        .arg(&ir_path)
        .args(["-o", "-"])
        .output()
        .expect("run clang -S");
    let asm_text = String::from_utf8_lossy(&asm.stdout);
    let n = asm_text.matches("vpdpwssd").count();
    assert_eq!(
        n,
        y::cpu_gemm::VNNI_MR * y::cpu_gemm::VNNI_NRV,
        "expected one vpdpwssd per (row, accumulator group); got {n}. \
         A lower count means LLVM scalarised or merged the intrinsic and the \
         kernel is no longer doing what it claims"
    );

    let drv_c = dir.join("drv.c");
    std::fs::write(&drv_c, schedule_defines() + DRIVER).expect("write driver");
    let exe = dir.join("drv");
    let link = Command::new("clang")
        .arg("-O2")
        .arg(&drv_c)
        .arg(&obj)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("link driver");
    assert!(
        link.status.success(),
        "driver failed to build:\n{}",
        String::from_utf8_lossy(&link.stderr)
    );

    let run = Command::new(&exe).output().expect("run driver");
    let out = String::from_utf8_lossy(&run.stdout);
    eprintln!("{out}");

    // Exactness at every size.
    for line in out.lines().filter(|l| l.starts_with("EXACT")) {
        assert!(
            line.ends_with("differ=0"),
            "the kernel disagrees with the scalar reference: {line}"
        );
    }
    // And at the licence's own boundary, which is the case that can actually
    // overflow int32 if the flush is wrong.
    let bound = out
        .lines()
        .find(|l| l.starts_with("BOUND"))
        .expect("bound line missing");
    assert!(
        bound.ends_with("differ=0"),
        "operands at the licensed maximum overflowed: either the flush is \
         broken or VnniExact::max_operand_magnitude is too loose. {bound}"
    );
    // Order independence at every split.
    for line in out.lines().filter(|l| l.starts_with("SPLIT")) {
        assert!(
            line.ends_with("differs=0"),
            "splitting the k-range changed the result — exact accumulation is \
             supposed to make this impossible: {line}"
        );
    }
    // The control must have found real drift, or the split result is vacuous.
    let control = out
        .lines()
        .find(|l| l.starts_with("CONTROL"))
        .expect("control line missing");
    let drifted: usize = control
        .split("differ=")
        .nth(1)
        .and_then(|s| s.split(' ').next())
        .and_then(|s| s.parse().ok())
        .expect("parse control");
    assert!(
        drifted > 0,
        "the f32 control did NOT drift under the same splits, so 'the exact \
         kernel is order-independent' proves nothing about this dataset: {control}"
    );

    assert!(out.contains("RESULT PASS"), "driver reported failure:\n{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The emitted chunk step must equal the licensed flush interval.
///
/// These are separate pieces of code — `VnniExact` decides how many k-pairs may
/// accumulate in int32 before overflow is possible, and the emitter decides how
/// many it actually accumulates. If they disagree, the kernel silently violates
/// the bound the licence was granted against and overflows in exactly the cases
/// the licence promised were safe.
#[test]
fn the_emitter_and_the_licence_agree_on_legal_intervals() {
    for t in [1u32, 2, 4, 64, 1024] {
        assert!(
            y::zero_drift::VnniExact::new(t).is_some(),
            "{t} is a power of two and must be legal"
        );
        let ir = y::cpu_gemm::emit_vnni_micro_module(t);
        // The chunk loop advances by exactly one flush interval, in both the
        // clamp that bounds the inner loop and the outer latch.
        assert!(
            ir.contains(&format!("%cend0 = add i64 %c, {t}")),
            "the inner loop's chunk bound must be the interval {t}"
        );
        assert!(
            ir.contains(&format!("%cnext = add i64 %c, {t}")),
            "the outer loop must step by the interval {t}"
        );
    }
    for t in [0u32, 3, 48, 100] {
        assert!(
            y::zero_drift::VnniExact::new(t).is_none(),
            "{t} is not a power of two and must be refused"
        );
    }
}

/// The hot loop must contain no flush, and therefore no accumulator spills.
///
/// **This is a performance property with no correctness symptom, so it needs
/// its own guard.** The first version of this kernel put the flush in a
/// conditional block inside the k-loop. It was exactly as correct — every test
/// in this file passed — and 2.4x slower, because a block that reads and writes
/// all 24 accumulators forces them out of registers for the whole loop. The
/// measured hot loop carried 24 `vpdpwssd` alongside 15 spill stores and 14
/// reloads.
///
/// Same shape as the bank-conflict swizzle and the constant-chain convergence
/// tests elsewhere in this repo: a regression here would be invisible to every
/// correctness assertion.
#[test]
fn the_hot_loop_does_not_spill_the_accumulators() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_vnni_spill_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let ir_path = dir.join("m.ll");
    std::fs::write(&ir_path, y::cpu_gemm::emit_vnni_micro_module(64)).expect("write IR");

    let out = Command::new("clang")
        .args(["-O2", "-S", "-x", "ir"])
        .arg(&ir_path)
        .args(["-o", "-"])
        .output()
        .expect("clang -S");
    let asm = String::from_utf8_lossy(&out.stdout);

    // The hot block is the first one containing vpdpwssd.
    //
    // Labels carry trailing comments (`.LBB0_4:   # %body`), so the block
    // boundary test must not require the line to *end* in a colon — an earlier
    // version did, matched nothing, and scanned the entire file as one block.
    // It then reported the flush as "leaked into the loop" when the flush was
    // simply elsewhere in the file, which is a false alarm rather than a missed
    // one, but would have sent the next person chasing a fixed bug.
    let is_label = |l: &str| l.starts_with(".LBB") && l.split(':').next().is_some_and(|h| !h.contains(char::is_whitespace)) && l.contains(':');
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for line in asm.lines() {
        if is_label(line.trim()) {
            blocks.push(std::mem::take(&mut cur));
        }
        cur.push(line);
    }
    blocks.push(cur);
    let hot = blocks
        .into_iter()
        .find(|b| b.iter().any(|l| l.contains("vpdpwssd")))
        .expect("no basic block with vpdpwssd found");

    let macs = hot.iter().filter(|l| l.contains("vpdpwssd")).count();
    assert_eq!(
        macs,
        y::cpu_gemm::VNNI_MR * y::cpu_gemm::VNNI_NRV,
        "the hot block should hold the whole tile's vpdpwssd"
    );

    // The structural property, and the one that actually moved the number:
    // the flush must not be in the loop at all. `vpmovsxdq` (sext i32->i64)
    // and `vpaddq` (the int64 accumulate) are its signature and appear
    // nowhere else in this kernel.
    for insn in ["vpmovsxdq", "vpaddq"] {
        let n = hot.iter().filter(|l| l.contains(insn)).count();
        assert_eq!(
            n, 0,
            "the flush has leaked back into the k-loop ({n} x {insn}). It \
             belongs in the outer chunk loop: a block that reads and writes all \
             {} accumulators forces them out of registers for the whole loop, \
             measured at 0.85x f32 against 1.13x with the flush hoisted — and \
             with no correctness symptom whatsoever.\nHot block:\n{}",
            y::cpu_gemm::VNNI_MR * y::cpu_gemm::VNNI_NRV,
            hot.join("\n")
        );
    }

    // Residual spills are a tuning matter, not a structural one: 24
    // accumulators + 4 B vectors + 1 A broadcast is 29 of 32 zmm, so the
    // allocator has almost no slack and some traffic is expected. Bounded to
    // catch a regression rather than pinned, with the measured figure recorded
    // so a change is visible.
    let spills = hot.iter().filter(|l| l.contains("Spill")).count();
    let reloads = hot.iter().filter(|l| l.contains("Reload")).count();
    assert!(
        spills + reloads <= 24,
        "hot-loop stack traffic regressed to {spills} spills + {reloads} \
         reloads; it was 10 + 10 when this bound was set (and 15 + 14 with the \
         flush inside the loop)"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
