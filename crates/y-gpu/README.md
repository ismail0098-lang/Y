# y-gpu

GPU-accelerated BN254 primitives for Groth16 proving. The kernels are written
in **Y**, not CUDA C, and compiled by the Y compiler; the resulting PTX is
embedded in this crate so there is no runtime dependency on the build tree.

```rust
let Some(prover) = GpuProver::new()? else {
    // No CUDA device. Fall back to a CPU prover -- this is not an error.
    return cpu_prove();
};
let key   = prover.prepare(&pk, &matrices)?;   // once per proving key
let proof = prover.prove(&pk, &key, &matrices, &full_assignment, r, s)?;
```

## What runs where

| phase | where |
|---|---|
| R1CS matrices, sparse matvec | CPU, parallel |
| QAP witness map (7 transforms) | **GPU** |
| G1 MSMs (`h`, `l`, `a`, `b_g1`) | **GPU** above a measured size, CPU below |
| G2 MSM (`b_g2`) | CPU — no `Fq2` kernel exists yet |

Dispatch is per **query**, not per proof: a Groth16 proof does four G1 MSMs of
different sizes and they can land on opposite sides of the threshold. The GPU
loses below ~57,000 terms cold / ~40,000 with staged bases; `Y_MSM_GPU_MIN`
overrides.

## Measured

Against arkworks (which is itself parallel here — ~12.5 cores at n=2^20), on
an RTX 4070 Ti SUPER with a 32-thread CPU:

| circuit | ~250k constraints | ~1M |
|---|---|---|
| sparse (multiplication chain) | ~4.6–6.0x | ~5.1x |
| dense (Poseidon chain) | ~5.7x | ~6.2x |

Run-to-run variance is ~15%; do not quote these more precisely.

**Against a mature GPU MSM library, y-gpu loses.** icicle v1.3.0 on the same
GPU is **2.8–4.8x faster** at the BN254 G1 MSM, and the gap grows with `n` —
its entire staged MSM at n=2^20 beats y-gpu's kernel alone with all host work
excluded. See [BENCHMARKS.md](BENCHMARKS.md) for the table and the list of
suspected causes (host-side bucket reduction and binning, serial per-bucket
accumulation, no signed-digit recoding, no mixed addition).

So: this beats a parallel CPU prover, and does not currently compete with
icicle. Both statements are measured.

## Correctness

The acceptance criterion is exact equality with arkworks, not "it verifies":

- proofs match `Groth16::create_proof_with_reduction` **element for element**
  at the same `r`/`s` (Groth16 is deterministic given its randomness, so this
  catches MSMs that are wrong in compensating ways — `verify` does not);
- the QAP matches `LibsnarkReduction::witness_map_from_matrices` exactly;
- every non-empty MSM bucket is asserted on-curve;
- the embedded PTX is checked against a fresh compile of the `.ysu` and run
  through `ptxas`.

## Known limits

- **G2 MSM is on the CPU**, and on a dense circuit it is the single largest
  remaining phase (~47%). Needs `Fq2` arithmetic, which does not exist here.
- `add-2007-bl` requires `P != ±Q` within a bucket. Identity bases are
  filtered; **duplicate or negated bases are not detected**. A violation
  produces a wrong proof, which verification rejects — but it is not checked
  up front.
- Single device, single stream. No H2D/compute overlap, no multi-GPU.
- Memory is not chunked: a circuit whose bases and tables exceed VRAM fails
  with `Error::Alloc` rather than spilling.
- BN254 + Groth16 only.
- Dispatch thresholds were measured on one machine. `Y_MSM_GPU_MIN` overrides;
  auto-calibration via `.ysu_hw_profile` is not wired in.
