//! M0's completion criterion: the exact GEMM is bit-identical across every
//! thread count and every flush interval.
//!
//! `docs/deterministic_inference.md` M0 is done when
//!
//! > a tiled, threaded, K-split GEMM returns results **bit-identical** to the
//! > naive nest across every thread count and every blocking parameter.
//!
//! Three axes are varied, and each is a different way of reordering the same
//! reduction:
//!
//! 1. **Thread count** — 1, 2, 3, 5, 8 and 16 threads each take a slice of the
//!    k-range into a private `C`, which are then summed. Every count regroups
//!    the additions differently.
//! 2. **Flush interval** — 64, 512 and 65536 k-pairs. The interval decides how
//!    many products accumulate in int32 before being widened into int64, so it
//!    directly changes the grouping of the outer sum.
//! 3. **Uneven splits** — deliberately ragged k-slices (1, 7, K/3, ...) so the
//!    boundaries do not align with the flush interval.
//!
//! All three must produce **byte-identical** output. Integer addition is
//! associative; that is the entire product claim, and this is the file that
//! demonstrates it at the level a user would actually call.
//!
//! **The control is not optional.** An f32 accumulator over the same data and
//! the same two-level structure must DISAGREE across the same splits — without
//! that, "the results matched" would also pass on a dataset too benign to
//! expose reassociation, which is the trap the first version of
//! `cpu_gemm_vnni_micro.rs` fell into.
//!
//! Requires `clang` and a CPU with AVX512-VNNI; skipped with a notice
//! otherwise.
//!
//! Run with:  cargo test --release --test cpu_gemm_exact_threaded -- --nocapture

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

fn host_has_vnni() -> bool {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.contains("avx512_vnni"))
        .unwrap_or(false)
}

/// Flush intervals compiled into the harness, each under its own symbol.
const INTERVALS: [u32; 3] = [64, 512, 65536];

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
#include <pthread.h>
typedef void (*gemm_fn)(const int16_t*,const int16_t*,int64_t*,long,long,long,long,long,long,int16_t*,int16_t*,int64_t*);
void __y_gemm_exact_vnni_64(const int16_t*,const int16_t*,int64_t*,long,long,long,long,long,long,int16_t*,int16_t*,int64_t*);
void __y_gemm_exact_vnni_512(const int16_t*,const int16_t*,int64_t*,long,long,long,long,long,long,int16_t*,int16_t*,int64_t*);
void __y_gemm_exact_vnni_65536(const int16_t*,const int16_t*,int64_t*,long,long,long,long,long,long,int16_t*,int16_t*,int64_t*);

static long M,N,K;
static int16_t *A,*B;
static unsigned s=20260817u;
static int rnd(int lo,int hi){s=s*1103515245u+12345u;return lo+(int)((s>>16)%(unsigned)(hi-lo+1));}
static long kp(long k){return (k+1)/2;}

typedef struct { gemm_fn f; long k0,klen; int64_t *C; int16_t *Ap,*Bp; int64_t *Ct; } job;
static void* worker(void*p){
  job*j=(job*)p;
  j->f(A + j->k0, B + (long)j->k0*N, j->C, M,N,j->klen, K,N,N, j->Ap,j->Bp,j->Ct);
  return NULL;
}

/* Run the GEMM with `nt` threads, each owning a k-slice and a private C.
   `ragged` cuts the k-range unevenly so slice boundaries never line up with
   the flush interval. */
static void run(gemm_fn f,int nt,int ragged,int64_t *out){
  pthread_t th[32]; job jb[32]; long cuts[32];
  long base=K/nt, extra=K%nt, off=0;
  for(int t=0;t<nt;t++){ cuts[t]=base+(t<extra?1:0); }
  if(ragged&&nt>1){ /* move work between neighbours, keeping the total */
    for(int t=0;t+1<nt;t++){ long d=(t*7+1)%(cuts[t]>1?cuts[t]:1); cuts[t]-=d; cuts[t+1]+=d; }
  }
  for(int t=0;t<nt;t++){
    jb[t].f=f; jb[t].k0=off; jb[t].klen=cuts[t]; off+=cuts[t];
    jb[t].C=(int64_t*)calloc((size_t)M*N,8);
    jb[t].Ap=(int16_t*)calloc((size_t)((M+MR-1)/MR)*kp(K)*MR*2,2);
    jb[t].Bp=(int16_t*)calloc((size_t)kp(K)*NR*2,2);
    jb[t].Ct=(int64_t*)calloc(MR*NR,8);
  }
  for(int t=0;t<nt;t++) pthread_create(&th[t],NULL,worker,&jb[t]);
  for(int t=0;t<nt;t++) pthread_join(th[t],NULL);
  memset(out,0,(size_t)M*N*8);
  for(int t=0;t<nt;t++){
    for(long q=0;q<M*N;q++) out[q]+=jb[t].C[q];
    free(jb[t].C);free(jb[t].Ap);free(jb[t].Bp);free(jb[t].Ct);
  }
}

/* f32 control with the same two-level structure: narrow accumulator per chunk,
   widened into the running total at chunk boundaries. */
static void f32_ref(long k0,long klen,float *R,long flush){
  static float acc[64*64];
  for(long i=0;i<M;i++)for(long j=0;j<N;j++){
    float tot=0.0f,a=0.0f; long c=0;
    for(long q=k0;q<k0+klen;q++){
      a+=(float)A[i*K+q]*(float)B[q*N+j];
      if(++c==flush){ tot+=a; a=0.0f; c=0; }
    }
    R[i*N+j]+=tot+a;
  }
  (void)acc;
}

int main(void){
  /* None a multiple of the tile. K is large for the CONTROL's sake, not the
     kernel's: the products are integers, so with |x| <= 1024 a dot product of
     301 terms peaks near 1.7e7 and f32 represents every integer below 2^24
     (1.68e7) exactly — the control drifted on 4 of 3071 elements and would
     have gone to zero under another seed. At K=1201 the sums reach ~3.5e7,
     past f32's exact-integer range, and reassociation becomes visible.

     Note the effective flush interval is min(T, kpairs) = 601 here, so the
     |x| <= 1024 data is within the licence at every T tested even though the
     nominal T=65536 arm would license only |x| <= 128 over a FULL interval. */
  M=37; N=83; K=1201;
  A=(int16_t*)malloc((size_t)M*K*2); B=(int16_t*)malloc((size_t)K*N*2);
  for(long t=0;t<M*K;t++)A[t]=(int16_t)rnd(-1024,1024);
  for(long t=0;t<K*N;t++)B[t]=(int16_t)rnd(-1024,1024);

  int64_t *ref=(int64_t*)calloc((size_t)M*N,8);
  for(long i=0;i<M;i++)for(long j=0;j<N;j++){
    int64_t a=0; for(long q=0;q<K;q++)a+=(int64_t)A[i*K+q]*B[q*N+j]; ref[i*N+j]=a; }

  gemm_fn fns[3]={__y_gemm_exact_vnni_64,__y_gemm_exact_vnni_512,__y_gemm_exact_vnni_65536};
  long ivs[3]={64,512,65536};
  int threads[6]={1,2,3,5,8,16};
  int64_t *got=(int64_t*)calloc((size_t)M*N,8);
  int fail=0;

  for(int v=0;v<3;v++)
    for(int t=0;t<6;t++)
      for(int ragged=0;ragged<2;ragged++){
        run(fns[v],threads[t],ragged,got);
        int d=memcmp(got,ref,(size_t)M*N*8)!=0;
        printf("GEMM flush=%5ld threads=%2d ragged=%d  differs=%d\n",ivs[v],threads[t],ragged,d);
        if(d) fail=1;
      }

  /* Control: the same two axes the integer path was just shown to be immune
     to — flush interval and split shape — applied to an f32 accumulator.

     Both axes are varied at once (64 vs 7, whole range vs ragged) because the
     first version varied only the split at a fixed interval and drifted on 4
     of 3071 elements. That is non-zero, so it was technically a valid control,
     but a margin that thin would go to zero under a different seed and turn a
     real property into a flaky test. */
  {
    float *f1=(float*)calloc((size_t)M*N,4),*f2=(float*)calloc((size_t)M*N,4);
    f32_ref(0,K,f1,64);
    long parts[5]={7,13,101,97,83}; long off=0;
    for(int i=0;i<5;i++){ long n=parts[i]; if(off+n>K)n=K-off; if(n>0)f32_ref(off,n,f2,7); off+=n; }
    if(off<K) f32_ref(off,K-off,f2,7);
    long diff=0; for(long q=0;q<M*N;q++) if(f1[q]!=f2[q]) diff++;
    printf("CONTROL f32 differ=%ld of %ld\n",diff,M*N);
    if(diff==0) fail=1;
    free(f1);free(f2);
  }
  printf("%s\n",fail?"RESULT FAIL":"RESULT PASS");
  return fail;
}
"#;

#[test]
fn the_exact_gemm_is_bit_identical_across_threads_and_intervals() {
    if !have("clang") {
        eprintln!("skipping: clang not found");
        return;
    }
    if !host_has_vnni() {
        eprintln!("skipping: host CPU has no avx512_vnni");
        return;
    }

    let dir = std::env::temp_dir().join(format!("y_vnni_thr_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let mut objs: Vec<PathBuf> = Vec::new();
    for t in INTERVALS {
        // One module per interval, with every externally visible symbol
        // suffixed so all three can be linked into one harness and compared
        // against each other in a single process.
        let ir = y::cpu_gemm::emit_vnni_gemm_module(t)
            .replace(
                y::cpu_gemm::VNNI_GEMM_NAME,
                &format!("{}_{t}", y::cpu_gemm::VNNI_GEMM_NAME),
            )
            .replace(
                y::cpu_gemm::VNNI_MICRO_NAME,
                &format!("{}_{t}", y::cpu_gemm::VNNI_MICRO_NAME),
            )
            .replace(
                "@__y_gemm_vnni_pack_a",
                &format!("@__y_gemm_vnni_pack_a_{t}"),
            )
            .replace(
                "@__y_gemm_vnni_pack_b",
                &format!("@__y_gemm_vnni_pack_b_{t}"),
            );
        let ll = dir.join(format!("g{t}.ll"));
        std::fs::write(&ll, ir).expect("write IR");
        let obj = dir.join(format!("g{t}.o"));
        let cc = Command::new("clang")
            .args(["-O2", "-c", "-x", "ir"])
            .arg(&ll)
            .arg("-o")
            .arg(&obj)
            .output()
            .expect("clang");
        assert!(
            cc.status.success(),
            "IR for interval {t} did not compile:\n{}",
            String::from_utf8_lossy(&cc.stderr)
        );
        objs.push(obj);
    }

    let drv = dir.join("drv.c");
    std::fs::write(&drv, schedule_defines() + DRIVER).expect("write driver");
    let exe = dir.join("drv");
    let mut link = Command::new("clang");
    link.arg("-O2").arg(&drv);
    for o in &objs {
        link.arg(o);
    }
    let out = link
        .arg("-lpthread")
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("link");
    assert!(
        out.status.success(),
        "harness failed to build:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let run = Command::new(&exe).output().expect("run harness");
    let log = String::from_utf8_lossy(&run.stdout);
    eprintln!("{log}");

    for line in log.lines().filter(|l| l.starts_with("GEMM")) {
        assert!(
            line.ends_with("differs=0"),
            "the exact GEMM is NOT reproducible across this configuration — \
             which is the entire product claim: {line}"
        );
    }

    let control = log
        .lines()
        .find(|l| l.starts_with("CONTROL"))
        .expect("control line missing");
    let drifted: i64 = control
        .split("differ=")
        .nth(1)
        .and_then(|s| s.split(' ').next())
        .and_then(|s| s.parse().ok())
        .expect("parse control");
    assert!(
        drifted > 0,
        "the f32 control did not drift under the same splits, so the \
         reproducibility result above proves nothing about this dataset: {control}"
    );

    assert!(log.contains("RESULT PASS"), "harness reported failure:\n{log}");
    let _ = std::fs::remove_dir_all(&dir);
}
