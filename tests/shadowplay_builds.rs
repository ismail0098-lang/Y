//! `shadowplay/shadowplay.ysu` is the only end-user application in this repo,
//! and nothing built it. It had stopped compiling AND stopped linking, in two
//! independent ways, and both were invisible because no test named it:
//!
//! * **The compiler refused it.** `llvm_emitter`'s auto-declare allowlist is
//!   the list of symbols the host link provides, and it was derived from
//!   `c_src/runtime.c` alone - never from `c_src/shadowplay_gui.h`, which
//!   `runtime.c` includes. So all eight GUI entry points, plus `usleep`, were
//!   "symbols that do not exist" and `Y shadowplay.ysu` exited 1 by name.
//!
//! * **It would not have linked anyway.** Every entry point in the header had
//!   been changed to `static inline`, which emits no symbol at all. The
//!   committed binary still carries them as `T` (global text), so this is
//!   drift: they were exported when it was last built. The likely cause is
//!   recorded in `tests/zero_drift_end_to_end.rs`, which says `runtime.c`
//!   "currently fails to compile ... static-after-non-static declaration in
//!   shadowplay_gui.h" - somebody silenced that by making everything
//!   `static`, which fixed the warning and killed the application.
//!
//! The lesson is the repo's own: a documented command that does not run is a
//! bug, and an allowlist is a second copy of something, so it will drift. The
//! symbol-level guard lives in `llvm_refuses_gpu_constructs.rs`
//! (`runtime_symbols_match_the_runtime`, which now also rejects a `static`
//! definition); this file is the end-to-end one, because the two failures
//! above are at different layers and neither guard sees the other's.
//!
//! **These tests never run the X11 binary.** `init_shadowplay_gui` grabs the
//! keyboard and takes bare F12 on the root window, so executing it would
//! hijack the developer's session. The run-the-artifact check is done against
//! the `Y_NO_X11` build instead, which is a real execution of the same
//! program over a surface that refuses by name.
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A Y program that drives the ShadowPlay GUI surface the way the application
/// does: `init` checked for failure (printing the application's own message),
/// then a poll loop over `update` and two state accessors with a `usleep`.
const GUI_SURFACE: &str = "\
// The ShadowPlay GUI surface, driven the way the application drives it.

@unsafe
fn main() {
    let ok: I32 = init_shadowplay_gui();
    if ok < 0 {
        println(\"Could not connect to X server\");
        return;
    }
    let running: I32 = 1;
    while running == 1 {
        let r: I32 = update_shadowplay_gui();
        if r < 0 {
            running = 0;
        }
        let rec: I32 = get_recording_state();
        let ind: I32 = get_indicator_state();
        if rec + ind > 1 {
            println(\"recording\");
        }
        usleep(16000);
    }
    cleanup_shadowplay_gui();
}
";

/// The programs the three build tests compile, each written into `dir` as
/// `<stem>.ysu`. Returns the stems.
///
/// **The application is not in this repository.** `../shadowplay/shadowplay.ysu`
/// lived in the old outer checkout, beside the cargo project, and no commit
/// reachable here has ever tracked it. So on a fresh clone these three tests
/// failed with "copy shadowplay.ysu: NotFound" and checked nothing. What they
/// exist to catch is the compiler and runtime half - a GUI entry point missing
/// from the allowlist (the compiler refuses), a `static` entry point (the
/// linker refuses), a headless stub that does not refuse - and `GUI_SURFACE`
/// reaches all three on every machine. The application is checked as well
/// wherever it exists, and its absence is announced rather than silent.
fn subjects(dir: &Path) -> Vec<&'static str> {
    std::fs::write(dir.join("gui_surface.ysu"), GUI_SURFACE).expect("write gui_surface.ysu");
    let mut stems = vec!["gui_surface"];
    let app = repo().join("../shadowplay/shadowplay.ysu");
    if app.exists() {
        std::fs::copy(&app, dir.join("shadowplay.ysu")).expect("copy shadowplay.ysu");
        stems.push("shadowplay");
    } else {
        eprintln!(
            "NOTE: the ShadowPlay application ({}) is not present - it is outside this \
             repository - so only the in-repo GUI-surface program is checked",
            app.display()
        );
    }
    // Non-vacuity: with no subject the three tests below check nothing and
    // pass, which is exactly what they did on a fresh clone before the
    // in-repo program existed - except that they failed on the missing copy.
    assert!(!stems.is_empty(), "no Y program to build; these tests would check nothing");
    stems
}

fn have(prog: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {} >/dev/null 2>&1", prog))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run a child and give up on it, rather than waiting forever.
///
/// This is not defensive boilerplate. The Y program's whole body is a 60Hz
/// `while` loop guarded by init's return value, so a stub that answers 0
/// instead of -1 does not fail this test - it HANGS it, and a hung test is a
/// stuck CI rather than a red build. Verified by mutation: making the headless
/// stub return 0 ran past 60 seconds before this deadline existed.
fn run_with_deadline(bin: &PathBuf, limit: std::time::Duration) -> String {
    use std::io::Read;
    use std::process::Stdio;

    let mut child = Command::new(bin)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn headless shadowplay");

    let start = std::time::Instant::now();
    loop {
        match child.try_wait().expect("try_wait") {
            Some(_) => break,
            None if start.elapsed() >= limit => {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "the headless build did not exit within {:?}; it acted on \
                     a successful init and entered the 60Hz poll loop",
                    limit
                );
            }
            None => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    }

    let mut out = String::new();
    let mut err = String::new();
    if let Some(mut h) = child.stdout.take() {
        let _ = h.read_to_string(&mut out);
    }
    if let Some(mut h) = child.stderr.take() {
        let _ = h.read_to_string(&mut err);
    }
    format!("{}{}", out, err)
}

fn workdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("y_shadowplay_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// Compile the app the way the documented command does, and link it.
///
/// This passes on a headless machine too: `main.rs` retries with `-DY_NO_X11`
/// when X11 is absent, so the link succeeds either way. What it cannot survive
/// is a missing allowlist entry (the compiler refuses) or a `static` entry
/// point (the linker refuses) - which is exactly the pair it exists to catch.
#[test]
fn the_shadowplay_application_compiles_and_links() {
    if !have("clang") {
        eprintln!("skipping: clang not installed");
        return;
    }
    let dir = workdir("build");
    for stem in subjects(&dir) {
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(dir.join(format!("{stem}.ysu")))
            .current_dir(&dir)
            .output()
            .expect("run Y");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success(), "`Y {stem}.ysu` failed:\n{}", text);

        let bin = dir.join(stem);
        assert!(
            bin.exists(),
            "the compiler reported success for {stem}.ysu but wrote no binary:\n{}",
            text
        );
    }
}

/// The build above must not be passing because the calls vanished.
///
/// Non-vacuity: assert the GUI surface really is reaching codegen. A program
/// whose calls were silently dropped would compile and link perfectly.
#[test]
fn the_gui_calls_survive_to_the_emitted_module() {
    let dir = workdir("ir");
    for stem in subjects(&dir) {
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(dir.join(format!("{stem}.ysu")))
            .arg("--emit-llvm")
            .current_dir(&dir)
            .output()
            .expect("run Y");
        assert!(out.status.success(), "--emit-llvm failed on {stem}.ysu");

        let ir = std::fs::read_to_string(dir.join(format!("{stem}.ll")))
            .unwrap_or_else(|e| panic!("read {stem}.ll: {e}"));
        let mut syms = vec![
            "init_shadowplay_gui",
            "update_shadowplay_gui",
            "get_recording_state",
            "get_indicator_state",
            "usleep",
        ];
        // The application also drives voice recording and capture, which the
        // in-repo surface program does not call.
        if stem == "shadowplay" {
            syms.extend(["get_voice_recording_state", "get_capture_failure_count", "print_microphone_label"]);
        }
        for sym in syms {
            assert!(
                ir.contains(&format!("declare i32 @{}(", sym)),
                "`{}` is not declared in {stem}.ll, so the call was dropped \
                 rather than lowered",
                sym
            );
            assert!(
                ir.contains(&format!("call i32 @{}(", sym)),
                "`{}` is declared in {stem}.ll but never called - the module \
                 would link and do nothing",
                sym
            );
        }
    }
}

/// Run the artifact, on the one configuration that is safe to execute.
///
/// `Y_NO_X11` compiles the GUI surface as refusing stubs. Building the app
/// against it and running it proves the whole pipeline end to end - lex,
/// parse, lower, link, execute - without grabbing the developer's keyboard.
/// It also pins that the headless fallback still LINKS, which is the property
/// that stops exporting the GUI symbols from making libX11 mandatory for
/// every Y program on every machine.
#[test]
fn the_headless_build_links_and_refuses_by_name() {
    if !have("clang") {
        eprintln!("skipping: clang not installed");
        return;
    }
    let dir = workdir("headless");
    for stem in subjects(&dir) {
        let emit = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(dir.join(format!("{stem}.ysu")))
            .arg("--emit-llvm")
            .current_dir(&dir)
            .output()
            .expect("run Y");
        assert!(emit.status.success(), "--emit-llvm failed on {stem}.ysu");

        let bin = dir.join(format!("{stem}_headless"));
        let link = Command::new("clang")
            .arg("-O1")
            .arg("-DY_NO_X11")
            .arg("-o")
            .arg(&bin)
            .arg(dir.join(format!("{stem}.ll")))
            .arg(repo().join("c_src/runtime.c"))
            .arg("-lm")
            .output()
            .expect("run clang");
        assert!(
            link.status.success(),
            "the headless build of {stem}.ysu does not link, so a machine \
             without libX11 cannot build any Y program:\n{}",
            String::from_utf8_lossy(&link.stderr)
        );

        let text = run_with_deadline(&bin, std::time::Duration::from_secs(20));
        assert!(
            text.contains("without X11 support"),
            "the headless build of {stem}.ysu ran but did not refuse by name:\n{}",
            text
        );
        // It must REFUSE, not proceed: the Y program checks init's return value
        // and bails. A stub returning 0 would drop the application into the 60Hz
        // poll loop forever, which is how this would hang CI.
        assert!(
            text.contains("Could not connect to X server"),
            "{stem}.ysu did not act on the refusal:\n{}",
            text
        );
    }
}

/// The two branches of the `Y_NO_X11` switch must export the SAME surface.
///
/// They are two hand-written copies of one API. If the real branch gains an
/// accessor the stub branch does not, a headless build stops linking - and
/// only on machines without libX11, which is where nobody looks.
#[test]
fn both_branches_of_the_x11_switch_export_the_same_api() {
    let header = std::fs::read_to_string(repo().join("c_src/shadowplay_gui.h"))
        .expect("read shadowplay_gui.h");
    let (stub, real) = header
        .split_once("\n#else\n")
        .expect("the Y_NO_X11 switch is gone");

    let exported = |text: &str| -> std::collections::BTreeSet<String> {
        let mut out = std::collections::BTreeSet::new();
        for line in text.lines() {
            let t = line.trim_start();
            // A definition at file scope: `<type> name(...) {`, not indented,
            // not `static`, and with a body.
            if line.starts_with(' ') || line.starts_with('\t') || t.starts_with("static") {
                continue;
            }
            let (head, rest) = match t.split_once('(') {
                Some(p) => p,
                None => continue,
            };
            if !rest.contains(')') || !rest.contains('{') {
                continue;
            }
            let mut words = head.split_whitespace();
            let ty = match words.next() {
                Some(w) => w,
                None => continue,
            };
            if !matches!(ty, "int32_t" | "void" | "int") {
                continue;
            }
            if let Some(name) = words.next() {
                out.insert(name.trim_start_matches('*').to_string());
            }
        }
        out
    };

    let stub_api = exported(stub);
    let real_api = exported(real);
    assert!(
        !stub_api.is_empty() && !real_api.is_empty(),
        "the API scraper found nothing, so this test proves nothing \
         (stub: {:?}, real: {:?})",
        stub_api,
        real_api
    );
    assert_eq!(
        stub_api, real_api,
        "the Y_NO_X11 stub surface and the real surface have diverged; a \
         headless build will fail to link on exactly the machines that need it"
    );

    // And every one of them must be callable from Y.
    let allowed: std::collections::BTreeSet<&str> =
        y::llvm_emitter::RUNTIME_SYMBOLS.iter().copied().collect();
    let unreachable: Vec<&String> = real_api
        .iter()
        .filter(|n| !allowed.contains(n.as_str()))
        .collect();
    assert!(
        unreachable.is_empty(),
        "these GUI entry points are exported but absent from RUNTIME_SYMBOLS, \
         so the LLVM backend refuses a Y program that calls them: {:?}",
        unreachable
    );
}
