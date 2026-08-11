# Where y-gpu actually stands

Measured on one machine: RTX 4070 Ti SUPER, 32-thread CPU, CUDA 13.3.

## Against icicle (Ingonyama), BN254 G1 MSM

icicle v1.3.0, built from source with `CUDAARCHS=89`. Same GPU, same `n`, same
curve, single MSM (batch 1). Both figures are host-to-host for one call, best
of 5 after a warm-up. "staged" means the bases already live on the device,
which is what a prover with a loaded proving key has.

| n | y-gpu cold | icicle cold | y-gpu staged | icicle staged | icicle is |
|---|---|---|---|---|---|
| 2^18 |  37.8 ms | **13.6 ms** |  29.5 ms | **12.3 ms** | 2.8x faster |
| 2^20 | 111.5 ms | **28.9 ms** |  83.3 ms | **21.4 ms** | 3.9x faster |
| 2^22 | 419.1 ms | **88.2 ms** | 309.3 ms | **63.8 ms** | 4.8x faster |

**icicle is 2.8–4.8x faster and the gap grows with n.** Its entire staged MSM
at 2^20 (21.4 ms) is faster than y-gpu's *kernel alone* with every host phase
excluded (53.5 ms).

Against the same CPU baseline that gives y-gpu 2.6–3.5x, icicle is ~10–14x.

### Where the gap is

Not yet attributed by profiling, so this is a list of differences, not a
diagnosis. In rough order of suspected size:

1. **The bucket reduction is on the host.** y-gpu downloads every bucket and
   reduces on the CPU; the table above shows that as `d2h` + `reduce`. icicle
   returns one point. This also blocks wider windows: at n=2^22 the `nw=16`
   geometry does 147 ms of kernel work against 240 ms at `nw=25`, but its
   d2h+reduce is 146 ms and eats the gain.
2. **Binning is on the host.** A counting sort over `n * windows` entries,
   plus an `O(threads * nb)` cursor table that gets expensive exactly when the
   bucket count grows.
3. **One thread per bucket, accumulating serially.** Even after the
   even-window fix, the kernel is bounded by the most-loaded bucket. icicle
   parallelises within a bucket.
4. **No signed-digit recoding.** NAF-style recoding halves the bucket count
   for the same window width; y-gpu does not do it.
5. **Jacobian + Jacobian addition.** Mixed (affine + Jacobian) addition is
   cheaper and is what a bases-are-affine MSM should use.

## Against arkworks (CPU), whole Groth16 prove

arkworks is parallel here (~12.5 cores at n=2^20 — via feature unification, not
by declaration). Best of 3.

| circuit | ~250k constraints | ~1M |
|---|---|---|
| sparse (multiplication chain) | ~4.6–6.0x | ~5.1x |
| dense (Poseidon chain) | ~5.7x | ~6.2x |

Run-to-run variance is ~15%.

**Read these two tables together.** The prover speedup over a CPU is real, but
the MSM inside it is ~4x off the state of the art, so the same prover built on
icicle's MSM would be substantially faster than this one. The honest summary
is: y-gpu beats a parallel CPU prover and does not currently compete with a
mature GPU MSM library.

## Reproducing the icicle comparison

```sh
curl -sL https://codeload.github.com/ingonyama-zk/icicle/tar.gz/refs/tags/v1.3.0 | tar xz
# icicle hardcodes /usr/local/cuda; on Arch it is /opt/cuda
sed -i 's|/usr/local/cuda|/opt/cuda|g' icicle-1.3.0/wrappers/rust/icicle-cuda-runtime/build.rs
# CUDA 13 dropped sm_50, which is CMake's default compiler probe
CUDAARCHS=89 cargo build --release
```
