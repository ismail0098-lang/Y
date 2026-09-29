//! `--emit-coprocessor` must produce PTX a GPU can actually load.
//!
//! It did not. The backend wrote the scheduler's instruction stream under a
//! `.version` header and stopped there: no `.visible .entry`, no `.reg`
//! declarations, and a `.shared` directive left at module scope inside the
//! body. `ptxas` rejects that at the first instruction, so the file the CLI
//! produced could not be assembled, loaded, or run - while the compiler printed
//! "Dual-accelerator PTX generated successfully!".
//!
//! The reason it went unnoticed is that something else was finishing the job. A
//! `wrap_ptx` helper inside `tests/benchmark_coprocessor_physical.py` hand-wrote
//! the entry point, the parameter list and the register pools in Python before
//! handing the result to CuPy - so the benchmark worked while the compiler's own
//! output did not. A backend that emits half a kernel and relies on a benchmark
//! script to complete it is not a backend.
//!
//! Requires `ptxas`; skipped with a notice otherwise, matching the gate on the
//! PTX emitter. String-matching would not substitute here: the whole failure was
//! structural, and every individual line in the file looked fine.
//!
//! Run with:  cargo test --test coprocessor_ptx_assembles

use std::path::PathBuf;
use std::process::Command;

/// Every co-processor workload in `tests/` must assemble for `sm_89`.
#[test]
fn every_coprocessor_workload_assembles() {
    if Command::new("ptxas").arg("--version").output().is_err() {
        eprintln!("skipping: ptxas not on PATH");
        return;
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("y_cop_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");

    let mut checked = 0;
    let mut refused = 0;
    for entry in std::fs::read_dir(repo.join("tests")).expect("read tests/") {
        let path = entry.expect("entry").path();
        let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        if path.extension().and_then(|e| e.to_str()) != Some("ysu")
            || !name.starts_with("coprocessor_")
        {
            continue;
        }

        // Compile into a scratch directory so the repo's checked-in .ptx files
        // are left alone.
        let local = dir.join(format!("{}.ysu", name));
        std::fs::copy(&path, &local).expect("copy source");
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&local)
            .arg("--emit-coprocessor")
            .current_dir(&repo)
            .output()
            .expect("run Y");
        let log = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );

        let ptx = dir.join(format!("{}.coprocessor.ptx", name));
        if !ptx.exists() {
            // Refusing is a legitimate outcome - `coprocessor_nerf` needs more
            // shared memory than a static `.shared` array can hold, and saying
            // so is better than emitting a module no GPU can load. What is not
            // acceptable is refusing quietly, or claiming success anyway.
            assert!(
                !log.contains("generated successfully"),
                "{} emitted no PTX but still reported success:\n{}",
                name,
                log
            );
            assert!(
                log.contains("[!]"),
                "{} emitted no PTX and gave no diagnostic:\n{}",
                name,
                log
            );
            refused += 1;
            continue;
        }

        let text = std::fs::read_to_string(&ptx).expect("read ptx");
        assert!(
            text.contains(".visible .entry"),
            "{}: emitted PTX has no entry point, so it is a fragment rather than a \
             module - the register pools and entry envelope must come from the \
             compiler, not from a benchmark script:\n{}",
            name,
            &text[..text.len().min(400)]
        );

        let res = Command::new("ptxas")
            .arg("-arch=sm_89")
            .arg(&ptx)
            .arg("-o")
            .arg("/dev/null")
            .output()
            .expect("run ptxas");
        assert!(
            res.status.success(),
            "{}: emitted PTX does not assemble:\n{}",
            name,
            String::from_utf8_lossy(&res.stderr)
        );
        checked += 1;
    }

    assert!(checked > 0, "no coprocessor_*.ysu workloads were found to check");
    eprintln!("assembled {} co-processor workloads, {} refused with a diagnostic", checked, refused);
}
