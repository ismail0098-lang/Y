//! Use a working CUDA assembler consistently across host-only PTX gates.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "verification.rs"]
mod verification;

pub fn find_ptxas(candidates: &[PathBuf]) -> Option<PathBuf> {
    let tool = candidates.iter().find(|path| {
        Command::new(path)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
    });
    verification::prerequisite_available(tool.is_some(), "working ptxas for emitted PTX assembly");
    tool.cloned()
}

pub fn ptxas() -> Option<PathBuf> {
    find_ptxas(&[
        "ptxas".into(),
        "/opt/cuda/bin/ptxas".into(),
        "/usr/local/cuda/bin/ptxas".into(),
    ])
}
