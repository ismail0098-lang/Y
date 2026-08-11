//! GPU-accelerated BN254 primitives for Groth16 proving, with the kernels
//! written in Y.
//!
//! What this accelerates, and what it does not:
//!
//! | phase | where |
//! |---|---|
//! | R1CS matrices + sparse matvec | CPU (parallel) |
//! | QAP witness map, 7 transforms | **GPU** |
//! | G1 MSMs (`h`, `l`, `a`, `b_g1`) | **GPU** above a measured size, CPU below |
//! | G2 MSM (`b_g2`) | CPU — no `Fq2` kernel exists |
//!
//! Correctness is checked by exact equality against arkworks: the QAP against
//! `LibsnarkReduction`, and whole proofs element for element against
//! `Groth16::create_proof_with_reduction` at the same `r` and `s`. Groth16 is
//! deterministic given its randomness, so a wrong MSM cannot hide inside a
//! proof that happens to verify.

pub mod error;
pub use error::{Error, Result};

mod device;
pub mod kernels;
pub mod msm;
pub mod qap;
pub mod groth16;
