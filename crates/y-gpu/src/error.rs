//! Errors.
//!
//! Every fallible entry point returns `Result` rather than panicking. The test
//! code this crate grew out of used `unwrap()` throughout, which is fine for a
//! test — a panic is a failed test — and unacceptable in a library, where a
//! CUDA allocation failure on a large circuit is an ordinary runtime condition
//! a caller should be able to fall back from, not a crash.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// No CUDA driver, or no device. Callers should fall back to a CPU prover.
    NoDevice,
    /// A CUDA driver call failed.
    Cuda(String),
    /// A kernel could not be loaded, usually because the embedded PTX does not
    /// match the device architecture.
    KernelLoad { kernel: &'static str, detail: String },
    /// A device allocation failed. Carries the size so a caller can decide to
    /// chunk or to fall back.
    Alloc { bytes: usize },
    /// The input violates a documented precondition.
    Invalid(String),
    /// Something this crate does not implement for the given input.
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoDevice => write!(f, "no CUDA device available"),
            Error::Cuda(m) => write!(f, "CUDA error: {m}"),
            Error::KernelLoad { kernel, detail } => {
                write!(f, "could not load kernel `{kernel}`: {detail}")
            }
            Error::Alloc { bytes } => {
                write!(f, "device allocation of {bytes} bytes failed")
            }
            Error::Invalid(m) => write!(f, "invalid input: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
