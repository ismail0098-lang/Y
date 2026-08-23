// ============================================================
//  An unrecognised option was SILENTLY IGNORED.
//
//      Y foo.ysu --ptx        ->  ran the LLVM backend, wrote a native
//                                 ELF, printed "Compiled successfully"
//      Y foo.ysu --probe      ->  same
//      Y foo.ysu --nonsense   ->  same
//
//  `--ptx` is not a flag (the PTX backend is `--emit-ptx`), and that
//  exact command was in this repo's own build instructions and in the
//  language documentation's flag table, alongside `--llvm` and
//  `--probe`, neither of which exists either.
//
//  This is the `--c` bug in its general form. CLAUDE.md records that
//  `--c` "used to be *silently ignored*, so the command this line
//  documented ran the LLVM backend instead" -- and that one flag was
//  given a real refusal while the `else if args[i].starts_with('-') {
//  i += 1; }` arm that ignored ALL of them was left in place. Fixing
//  the instance and not the class is the thing the design rule exists
//  to catch.
// ============================================================

use std::path::PathBuf;
use std::process::Command;

fn run(extra: &[&str]) -> (bool, String) {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(repo.join("tests/hello.ysu"))
        .args(extra)
        .arg("-o")
        .arg(std::env::temp_dir().join(format!("y_flagtest_{}", std::process::id())))
        .current_dir(&repo)
        .output()
        .expect("run Y");
    (
        out.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

#[test]
fn a_flag_that_does_not_exist_is_an_error_not_a_no_op() {
    for flag in ["--ptx", "--probe", "--llvm", "--definitely-not-a-flag"] {
        let (ok, text) = run(&[flag]);
        assert!(!ok, "`{flag}` exited 0:\n{text}");
        assert!(
            text.contains("unrecognised option") && text.contains(flag),
            "`{flag}` was not named in the diagnostic:\n{text}"
        );
    }
}

#[test]
fn the_message_points_at_the_flag_people_actually_want() {
    // `--ptx` is the one in the build instructions, so the near-miss is worth
    // naming explicitly rather than leaving the user to diff two lists.
    let (_, text) = run(&["--ptx"]);
    assert!(
        text.contains("--emit-ptx"),
        "the refusal does not mention the flag that works:\n{text}"
    );
}

#[test]
fn every_real_flag_still_compiles() {
    // The control, and it is load-bearing: a whitelist that forgot an option
    // would turn "silently ignored" into "silently refused", which is worse
    // for anyone with a working command line. These are every emit mode that
    // needs no extra argument.
    for flag in [
        "--emit-ptx",
        "--emit-cpu",
        "--emit-llvm",
        "--emit-coprocessor",
        "--target=ptx",
        "--target=cpu",
        "--target=llvm",
        "--portable",
        "--no-autotune",
    ] {
        let (_, text) = run(&[flag]);
        assert!(
            !text.contains("unrecognised option"),
            "`{flag}` is a real option and was refused:\n{text}"
        );
    }
}

#[test]
fn the_bare_invocation_still_works() {
    let (ok, text) = run(&[]);
    assert!(ok && !text.contains("unrecognised option"), "{text}");
}
