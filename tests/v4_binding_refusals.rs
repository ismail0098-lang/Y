//! A `U32x4` names FOUR registers, and every site that resolves a name through
//! `variables` alone is blind to it.
//!
//! This started as a recorded residue of the `let`-aliasing fix, which said a
//! `U32x4` "can still alias ... unreachable rather than safe". **Measured, that
//! is wrong in a useful direction: a v4 cannot alias at all.** Both producers of
//! a v4 marker (`block_ptr2d_load_v4`, `shared_load_v4`) allocate four FRESH
//! registers, and `Expr::Ident` never consults `vec_vars`, so `let b: U32x4 = a;`
//! is refused before it can reach `Stmt::Let`'s binding site. The hazard is not
//! unreachable, it is structurally impossible, for a stateable reason.
//!
//! Asking the artifact that question found a different, LIVE defect one
//! statement over. `Stmt::Assign` resolved its target through `variables` with
//! no `else`, so a write to a v4 fell out of the `if let` and vanished:
//!
//! ```text
//!     a = 7;      ->   mov.u32 %r20, 7;     // value computed
//!                      ...                  // and the write never happens
//! ```
//!
//! Exit 0, an artifact written, and `ptxas` accepts the module -- the
//! design-rule shape exactly, invisible to every assemble gate. Nothing in the
//! corpus mutates a v4, so it was latent; *find these while the path is still
//! dead*.
//!
//! The third test is the control, and it is what stops "refuse anything that
//! touches a v4" from passing: `bn254_permute.ysu` reads v4 lanes and is how
//! every eight-limb BN254 kernel in this repository is written. Deleting that
//! path would take out the whole field-arithmetic corpus.

use std::path::Path;
use std::process::Command;

struct Run {
    ok: bool,
    text: String,
    artifact: bool,
}

/// Compiles a COPY. `--emit-ptx` writes next to its input, so sweeping in place
/// rewrites committed artifacts and races every other binary compiling the same
/// fixture. The fixture name alone is not unique -- a name makes the
/// requirement visible, only a counter makes it hold -- so the directory
/// carries an atomic counter as well.
fn compile(fixture: &str) -> Run {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "y_v4_refusal_{}_{}_{}",
        std::process::id(),
        fixture.replace('/', "_").replace('.', "_"),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("k.ysu");
    std::fs::copy(repo.join(fixture), &src).expect("fixture missing");
    let _ = std::fs::copy(repo.join(".ysu_hw_profile"), dir.join(".ysu_hw_profile"));

    let out = Command::new(bin.join("Y"))
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(repo)
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let artifact = dir.join("k.ptx").exists();
    let r = Run {
        ok: out.status.success(),
        text,
        artifact,
    };
    let _ = std::fs::remove_dir_all(&dir);
    r
}

/// THE LIVE ONE. Assigning to a vector used to emit nothing at all.
///
/// Asserting only "the compile failed" would pass for a compiler that refused
/// this for some unrelated reason, so the message has to name the vector and
/// say what to do instead. **Measured: the message assertion is what catches
/// the defect** -- restoring it with the artifact assertion deleted still
/// fails here. The artifact assertion is kept anyway, because it pins the part
/// that made this worse than a bad message: the dropped write shipped a `.ptx`
/// and exited 0, and a future refusal that wrote one would otherwise pass.
#[test]
fn assigning_to_a_vector_is_refused_rather_than_dropped() {
    let r = compile("tests/v4_assign.ysu");
    assert!(
        !r.ok,
        "assigning to a U32x4 compiled successfully; it used to be a silent \
         no-op that ptxas accepts:\n{}",
        r.text
    );
    assert!(
        !r.artifact,
        "a refused compile still wrote a .ptx -- the whole point is that the \
         dropped write never reaches an artifact:\n{}",
        r.text
    );
    assert!(
        r.text.contains("assigning to `a`") && r.text.contains("4-wide vector"),
        "the refusal does not name the target as a vector, so a reader cannot \
         act on it:\n{}",
        r.text
    );
}

/// The DIAGNOSIS, not the refusal: this was always rejected, and it blamed a
/// name the user had just declared. A test asserting only `!ok` would have
/// passed against the misleading message, which is why this one asserts what
/// the message may NOT say.
#[test]
fn a_vector_read_as_a_scalar_is_not_reported_as_an_undefined_name() {
    let r = compile("tests/v4_read_as_scalar.ysu");
    assert!(!r.ok, "reading a U32x4 as a scalar compiled:\n{}", r.text);
    assert!(
        !r.text.contains("the undefined name `a`"),
        "`a` is declared on the line above; reporting it as undefined sends the \
         reader looking for a typo:\n{}",
        r.text
    );
    assert!(
        r.text.contains("4-wide vector") && r.text.contains(".x"),
        "the refusal should name the vector and point at the lane read:\n{}",
        r.text
    );
}

/// THE CONTROL. "Refuse anything mentioning a v4" satisfies both tests above
/// and deletes the idiom every BN254 kernel here is built on.
#[test]
fn reading_a_vector_lane_still_compiles() {
    let r = compile("tests/bn254_permute.ysu");
    assert!(
        r.ok && r.artifact,
        "the v4 lane-read path is how every eight-limb BN254 kernel in this \
         repository is written, and it must keep compiling:\n{}",
        r.text
    );
}
