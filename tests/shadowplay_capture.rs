//! ShadowPlay's microphone discovery and recorder command lines
//! (`c_src/shadowplay_capture.h`), and the two process rules the GUI header
//! enforces around them.
//!
//! Voice capture was broken three ways at once, and each one hid the others:
//!
//! * The device list was read out of the error older gpu-screen-recorder
//!   releases printed for `-a check_devices`. 6.x stops at "missing argument
//!   '-w'" before it reads `-a`, so the list was always just "Disabled" and
//!   "Default Input".
//! * "Default Input" is the default SOURCE. On the machine this was written on
//!   that is `...analog-stereo.monitor` - what the headphones play - so the
//!   "microphone" recorded the desktop a second time and never the voice.
//!   `the_real_machine_picks_the_headset_microphone_not_the_monitor` is that
//!   machine's actual `pactl` output.
//! * The microphone reached one recorder of three. gpu-screen-recorder got it;
//!   the wf-recorder and ffmpeg fallbacks dropped it while the log said
//!   "Desktop + Mic [Merged]".
//!
//! The header is pure (no X11, no exec), so a small C harness drives it on any
//! machine. The ffmpeg command lines are then RUN, through the same builder,
//! with lavfi tone generators standing in for the sound card: a 440 Hz
//! "desktop" and an 880 Hz "microphone". The assertion is on the file, not the
//! flags - the 880 Hz tone must be in the recording, and must be absent from a
//! recording made with the microphone off, or the test proves nothing.
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn have(prog: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {} >/dev/null 2>&1", prog))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// A fresh directory for one test. The tag is for legibility; the counter is
/// what makes it unique - this repo has lost a day to two tests sharing a
/// pid-only temp dir, one deleting it while the other wrote.
fn workdir(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "y_sp_capture_{}_{}_{}",
        std::process::id(),
        tag,
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn c_compiler() -> Option<&'static str> {
    ["clang", "cc", "gcc"].into_iter().find(|c| have(c))
}

const HARNESS: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "shadowplay_capture.h"

static int g_argc;
static char** g_argv;

static const char* kv(const char* key) {
    size_t k = strlen(key);
    for (int i = 2; i < g_argc; i++) {
        if (strncmp(g_argv[i], key, k) == 0 && g_argv[i][k] == '=') return g_argv[i] + k + 1;
    }
    return NULL;
}

static int kvi(const char* key, int dflt) {
    const char* v = kv(key);
    return v ? atoi(v) : dflt;
}

static char* slurp(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* b = malloc((size_t)n + 1);
    size_t got = fread(b, 1, (size_t)n, f);
    b[got] = '\0';
    fclose(f);
    return b;
}

static const char* reason(sp_pick_reason r) {
    switch (r) {
    case SP_PICK_NONE: return "none";
    case SP_PICK_DEFAULT: return "default";
    case SP_PICK_SAME_DEVICE: return "same-device";
    case SP_PICK_NAMED_MIC: return "named-mic";
    case SP_PICK_FIRST: return "first";
    }
    return "?";
}

static void print_argv(const sp_argv* a) {
    printf("OVERFLOW\t%d\n", a->overflow);
    for (int i = 0; i < a->n; i++) printf("ARG\t%s\n", a->v[i]);
}

int main(int argc, char** argv) {
    g_argc = argc;
    g_argv = argv;
    if (argc < 2) return 2;

    if (!strcmp(argv[1], "pactl") || !strcmp(argv[1], "gsr")) {
        char* text = slurp(kv("list"));
        sp_mic m[SP_MAX_MICS];
        int max = kvi("max", SP_MAX_MICS);
        int n = !strcmp(argv[1], "pactl") ? sp_parse_pactl_sources(text, m, max)
                                          : sp_parse_gsr_devices(text, m, max);
        char src[SP_NAME_LEN] = "", sink[SP_NAME_LEN] = "";
        if (kv("info")) {
            char* info = slurp(kv("info"));
            sp_pactl_info_field(info, "Default Source:", src, sizeof src);
            sp_pactl_info_field(info, "Default Sink:", sink, sizeof sink);
            free(info);
        }
        for (int i = 0; i < n; i++) printf("MIC\t%s\t%s\n", m[i].name, m[i].label);
        sp_pick_reason why;
        int pick = sp_pick_mic(m, n, src, sink, &why);
        printf("DEFAULT_SOURCE\t%s\nDEFAULT_SINK\t%s\n", src, sink);
        printf("PICK\t%d\t%s\t%s\n", pick, reason(why), pick >= 0 ? m[pick].name : "");
        free(text);
        return 0;
    }
    if (!strcmp(argv[1], "capture")) {
        sp_capture c;
        memset(&c, 0, sizeof c);
        c.replay = kvi("replay", 0);
        c.replay_seconds = kvi("replay_seconds", 30);
        c.quality = kvi("quality", 1);
        c.codec = kvi("codec", 0);
        c.container = kvi("container", 0);
        c.mic = kv("mic");
        c.desktop = kv("desktop");
        c.out = kv("out");
        c.wayland = kvi("wayland", 0);
        c.display = kv("display");
        c.screen_w = kvi("w", 1920);
        c.screen_h = kvi("h", 1080);
        c.has_gsr = kvi("gsr", 0);
        c.has_wf = kvi("wf", 0);
        c.has_ffmpeg = kvi("ffmpeg", 0);
        c.ffmpeg_pulse = kvi("pulse", 1);
        c.audio_format = kv("audio_format");
        const char* video[5] = {"-f", "lavfi", "-i", NULL, NULL};
        if (kv("video_lavfi")) {
            video[3] = kv("video_lavfi");
            c.video_input = video;
        }
        sp_argv a;
        char note[512];
        sp_backend b = sp_build_capture(&a, &c, note, sizeof note);
        printf("BACKEND\t%s\nNOTE\t%s\n", sp_backend_name(b), note);
        print_argv(&a);
        return 0;
    }
    if (!strcmp(argv[1], "voice")) {
        sp_voice v;
        memset(&v, 0, sizeof v);
        v.mic = kv("mic");
        v.out_base = kv("out_base");
        v.has_ffmpeg = kvi("ffmpeg", 0);
        v.has_parecord = kvi("parecord", 0);
        v.audio_format = kv("audio_format");
        sp_argv a;
        char path[512];
        sp_backend b = sp_build_voice(&a, &v, path, sizeof path);
        printf("BACKEND\t%s\nPATH\t%s\n", sp_backend_name(b), path);
        print_argv(&a);
        return 0;
    }
    return 2;
}
"#;

/// The harness, built once per test process. `-Werror`: the header is
/// compiled into every Y program's runtime, so a warning in it is a warning
/// in every Y build.
fn harness() -> Option<&'static PathBuf> {
    static BIN: OnceLock<Option<PathBuf>> = OnceLock::new();
    BIN.get_or_init(|| {
        let cc = match c_compiler() {
            Some(cc) => cc,
            None => {
                eprintln!("SKIP: no C compiler, so the capture harness cannot be built");
                return None;
            }
        };
        let dir = workdir("harness");
        let src = dir.join("harness.c");
        std::fs::write(&src, HARNESS).expect("write harness");
        let bin = dir.join("harness");
        let out = Command::new(cc)
            .args(["-std=c11", "-O1", "-Wall", "-Wextra", "-Werror"])
            .arg("-I")
            .arg(repo().join("c_src"))
            .arg(&src)
            .arg("-o")
            .arg(&bin)
            .output()
            .expect("run the C compiler");
        assert!(
            out.status.success(),
            "c_src/shadowplay_capture.h does not compile cleanly:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        Some(bin)
    })
    .as_ref()
}

struct Built {
    backend: String,
    note: String,
    path: String,
    overflow: bool,
    args: Vec<String>,
}

fn run_harness(args: &[&str]) -> Option<String> {
    let bin = harness()?;
    let out = Command::new(bin).args(args).output().expect("run harness");
    assert!(
        out.status.success(),
        "harness {:?} failed:\n{}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    Some(String::from_utf8(out.stdout).expect("utf-8"))
}

fn build(kind: &str, kv: &[&str]) -> Option<Built> {
    let mut args = vec![kind];
    args.extend_from_slice(kv);
    let text = run_harness(&args)?;
    let mut b = Built {
        backend: String::new(),
        note: String::new(),
        path: String::new(),
        overflow: false,
        args: Vec::new(),
    };
    for line in text.lines() {
        let (key, val) = line.split_once('\t').unwrap_or((line, ""));
        match key {
            "BACKEND" => b.backend = val.to_string(),
            "NOTE" => b.note = val.to_string(),
            "PATH" => b.path = val.to_string(),
            "OVERFLOW" => b.overflow = val == "1",
            "ARG" => b.args.push(val.to_string()),
            _ => panic!("unexpected harness output: {:?}", line),
        }
    }
    Some(b)
}

struct Mics {
    names: Vec<String>,
    labels: Vec<String>,
    pick: i64,
    reason: String,
    picked: String,
}

fn parse_mics(kind: &str, list: &str, info: Option<&str>, tag: &str) -> Option<Mics> {
    harness()?;
    let dir = workdir(tag);
    let list_path = dir.join("list.txt");
    std::fs::write(&list_path, list).unwrap();
    let list_arg = format!("list={}", list_path.display());
    let mut args = vec![kind, list_arg.as_str()];
    let info_arg;
    if let Some(info) = info {
        let info_path = dir.join("info.txt");
        std::fs::write(&info_path, info).unwrap();
        info_arg = format!("info={}", info_path.display());
        args.push(info_arg.as_str());
    }
    let text = run_harness(&args)?;
    let mut m = Mics {
        names: vec![],
        labels: vec![],
        pick: -2,
        reason: String::new(),
        picked: String::new(),
    };
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "MIC" => {
                m.names.push(f[1].to_string());
                m.labels.push(f[2].to_string());
            }
            "PICK" => {
                m.pick = f[1].parse().unwrap();
                m.reason = f[2].to_string();
                m.picked = f.get(3).unwrap_or(&"").to_string();
            }
            _ => {}
        }
    }
    Some(m)
}

// ------------------------------------------------------------------ fixtures

/// `LC_ALL=C pactl list sources` from the machine this was written on, with
/// the property blocks trimmed. The default source is the headphones' MONITOR,
/// which is exactly the configuration that made "Default Input" record the
/// desktop instead of the voice.
const REAL_SOURCES: &str = "Source #74
\tState: SUSPENDED
\tName: alsa_output.usb-1130_USB_AUDIO-00.analog-stereo.monitor
\tDescription: Monitor of USB  AUDIO   Analog Stereo
\tDriver: PipeWire
\tMonitor of Sink: alsa_output.usb-1130_USB_AUDIO-00.analog-stereo
\tProperties:
\t\tnode.name = \"alsa_output.usb-1130_USB_AUDIO-00.analog-stereo.monitor\"
Source #78
\tState: SUSPENDED
\tName: alsa_input.usb-Generic_USB_Audio-00.HiFi__Line__source
\tDescription: USB Audio Line Input
\tDriver: PipeWire
\tVolume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB
\t        balance 0.00
\tMonitor of Sink: n/a
\tProperties:
\t\tdevice.description = \"USB Audio\"
\tPorts:
\t\tanalog-input-linein: Line In (type: Line, priority: 8100, availability unknown)
\tActive Port: analog-input-linein
\tFormats:
\t\tpcm
Source #79
\tState: SUSPENDED
\tName: alsa_input.usb-Generic_USB_Audio-00.HiFi__Mic__source
\tDescription: USB Audio Microphone
\tMonitor of Sink: n/a
\tProperties:
\t\tnode.name = \"alsa_input.usb-Generic_USB_Audio-00.HiFi__Mic__source\"
Source #2450
\tState: SUSPENDED
\tName: alsa_output.pci-0000_01_00.1.hdmi-stereo.monitor
\tDescription: Monitor of AD103 High Definition Audio Controller Digital Stereo (HDMI) [U28D590]
\tMonitor of Sink: alsa_output.pci-0000_01_00.1.hdmi-stereo
Source #5346
\tState: RUNNING
\tName: alsa_output.usb-SteelSeries_Arctis_Nova_7P-00.analog-stereo.monitor
\tDescription: Monitor of Arctis Nova 7P Analog Stereo
\tMonitor of Sink: alsa_output.usb-SteelSeries_Arctis_Nova_7P-00.analog-stereo
\tProperties:
\tPorts:
\tActive Port: analog-output
\tFormats:
Source #5347
\tState: RUNNING
\tName: alsa_input.usb-SteelSeries_Arctis_Nova_7P-00.mono-fallback
\tDescription: Arctis Nova 7P Mono
\tDriver: PipeWire
\tSample Specification: s16le 1ch 48000Hz
\tMonitor of Sink: n/a
\tFlags: HARDWARE HW_MUTE_CTRL HW_VOLUME_CTRL DECIBEL_VOLUME LATENCY
\tProperties:
\t\tnode.name = \"alsa_input.usb-SteelSeries_Arctis_Nova_7P-00.mono-fallback\"
\t\tdevice.description = \"Arctis Nova 7P\"
\t\tdevice.form_factor = \"headset\"
";

const HEADSET: &str = "alsa_input.usb-SteelSeries_Arctis_Nova_7P-00.mono-fallback";
const USB_MIC: &str = "alsa_input.usb-Generic_USB_Audio-00.HiFi__Mic__source";
const USB_LINE: &str = "alsa_input.usb-Generic_USB_Audio-00.HiFi__Line__source";

fn info(source: &str, sink: &str) -> String {
    format!(
        "Server Name: PulseAudio (on PipeWire 1.6.8)\n\
         Default Sample Specification: float32le 2ch 48000Hz\n\
         Default Sink: {}\n\
         Default Source: {}\n\
         Cookie: 6616:a588\n",
        sink, source
    )
}

// ------------------------------------------------------------------ discovery

#[test]
fn only_real_inputs_are_offered_as_microphones() {
    let Some(m) = parse_mics("pactl", REAL_SOURCES, None, "list") else { return };
    assert_eq!(
        m.names,
        vec![USB_LINE, USB_MIC, HEADSET],
        "the microphone list must be every input and no monitor"
    );
    assert_eq!(m.labels, vec!["USB Audio Line Input", "USB Audio Microphone", "Arctis Nova 7P Mono"]);
}

/// The configuration this whole change is about, from the real machine: the
/// default source is the headphones' monitor, so "use the default input"
/// would record the desktop. The headset's own microphone must win.
#[test]
fn the_real_machine_picks_the_headset_microphone_not_the_monitor() {
    let info = info(
        "alsa_output.usb-SteelSeries_Arctis_Nova_7P-00.analog-stereo.monitor",
        "alsa_output.usb-SteelSeries_Arctis_Nova_7P-00.analog-stereo",
    );
    let Some(m) = parse_mics("pactl", REAL_SOURCES, Some(&info), "real") else { return };
    assert_eq!(m.picked, HEADSET, "picked {:?} for reason {}", m.picked, m.reason);
    assert_eq!(m.reason, "same-device");
}

/// Control for the test above: when the default source IS a microphone, it
/// is the user's choice and it wins, even over the headset heuristic.
#[test]
fn a_default_source_that_is_a_microphone_is_honoured() {
    let info = info(USB_MIC, "alsa_output.usb-SteelSeries_Arctis_Nova_7P-00.analog-stereo");
    let Some(m) = parse_mics("pactl", REAL_SOURCES, Some(&info), "honour") else { return };
    assert_eq!(m.picked, USB_MIC);
    assert_eq!(m.reason, "default");
}

/// One card with a line input AND a microphone: the one named like a
/// microphone, not the first one listed.
#[test]
fn on_a_card_with_several_inputs_the_microphone_wins_over_line_in() {
    let info = info(
        "alsa_output.usb-Generic_USB_Audio-00.HiFi__Speaker__sink.monitor",
        "alsa_output.usb-Generic_USB_Audio-00.HiFi__Speaker__sink",
    );
    let Some(m) = parse_mics("pactl", REAL_SOURCES, Some(&info), "line") else { return };
    assert_eq!(m.picked, USB_MIC, "picked {:?}", m.picked);
    assert_eq!(m.reason, "same-device");
}

/// Output on HDMI, which has no input: fall back to anything named like a
/// microphone - and still never to the monitor that is the default source.
#[test]
fn with_no_input_on_the_output_device_a_named_microphone_is_used() {
    let info = info(
        "alsa_output.pci-0000_01_00.1.hdmi-stereo.monitor",
        "alsa_output.pci-0000_01_00.1.hdmi-stereo",
    );
    let Some(m) = parse_mics("pactl", REAL_SOURCES, Some(&info), "hdmi") else { return };
    assert_eq!(m.picked, USB_MIC);
    assert_eq!(m.reason, "named-mic");
}

#[test]
fn a_machine_with_only_monitors_has_no_microphone() {
    let only_monitors = REAL_SOURCES
        .split("Source #")
        .filter(|s| s.contains(".monitor\n"))
        .map(|s| format!("Source #{}", s))
        .collect::<String>();
    let Some(m) = parse_mics("pactl", &only_monitors, Some(&info("", "")), "none") else { return };
    assert!(m.names.is_empty(), "offered a monitor as a microphone: {:?}", m.names);
    assert_eq!((m.pick, m.reason.as_str()), (-1, "none"));
}

/// The `Monitor of Sink:` field is the authority; the `.monitor` suffix is
/// only a fallback for output that lacks it. A virtual input may be named
/// anything, including something ending in `.monitor`.
#[test]
fn the_monitor_field_outranks_the_name() {
    let text = "Source #1\n\tName: virtual.mic.monitor\n\tDescription: Weird Name\n\tMonitor of Sink: n/a\n\
                Source #2\n\tName: plainname\n\tDescription: Loopback\n\tMonitor of Sink: some_sink\n\
                Source #3\n\tName: old_pactl.monitor\n\tDescription: No Field\n\
                Source #4\n\tName: old_pactl_input\n\tDescription: No Field Either\n";
    let Some(m) = parse_mics("pactl", text, None, "field") else { return };
    assert_eq!(m.names, vec!["virtual.mic.monitor", "old_pactl_input"]);
}

/// A device name longer than the buffer is dropped, not truncated: a
/// truncated name is a DIFFERENT device, and the recorder would refuse it -
/// or open the wrong one.
#[test]
fn an_overlong_device_name_is_dropped_not_truncated() {
    let long = format!("alsa_input.{}.mono", "x".repeat(250));
    let text = format!(
        "Source #1\n\tName: {}\n\tDescription: Long\n\tMonitor of Sink: n/a\n\
         Source #2\n\tName: alsa_input.short.mono\n\tDescription: Short\n\tMonitor of Sink: n/a\n",
        long
    );
    let Some(m) = parse_mics("pactl", &text, None, "long") else { return };
    assert_eq!(m.names, vec!["alsa_input.short.mono"]);
}

/// The fallback list, `gpu-screen-recorder --list-audio-devices`, in 6.x's
/// `name|label` format. The two abstract entries are not microphones.
#[test]
fn the_gpu_screen_recorder_list_is_read_without_its_abstract_entries() {
    let text = "default_output|Default output\n\
                default_input|Default input\n\
                alsa_output.usb-SteelSeries_Arctis_Nova_7P-00.analog-stereo.monitor|Monitor of Arctis Nova 7P Analog Stereo\n\
                alsa_input.usb-SteelSeries_Arctis_Nova_7P-00.mono-fallback|Arctis Nova 7P Mono\n\
                alsa_input.usb-Generic_USB_Audio-00.HiFi__Mic__source|USB Audio Microphone\n";
    let Some(m) = parse_mics("gsr", text, None, "gsr") else { return };
    assert_eq!(m.names, vec![HEADSET, USB_MIC]);
    assert_eq!(m.labels, vec!["Arctis Nova 7P Mono", "USB Audio Microphone"]);
}

/// pactl translates its field names, and the parser matches English ones. A
/// `pactl list`/`pactl info` call without `LC_ALL=C` works for an
/// English-speaking developer and silently finds no microphone for everyone
/// else. The behaviour cannot be tested without every translation, so the
/// call sites are checked instead.
#[test]
fn every_parsed_pactl_call_runs_in_the_c_locale() {
    let gui = std::fs::read_to_string(repo().join("c_src/shadowplay_gui.h")).unwrap();
    let mut checked = 0;
    for (i, line) in gui.lines().enumerate() {
        for cmd in ["pactl list", "pactl info"] {
            if line.contains(cmd) && line.contains('"') && !line.trim_start().starts_with("//") {
                checked += 1;
                assert!(
                    line.contains(&format!("LC_ALL=C {}", cmd)),
                    "c_src/shadowplay_gui.h:{} runs `{}` without LC_ALL=C:\n{}",
                    i + 1,
                    cmd,
                    line
                );
            }
        }
    }
    assert!(checked >= 2, "found {} pactl calls to check; the scan is looking at nothing", checked);
}

/// The fixtures above are this machine's output as of writing. If pactl is
/// here, check the parser against the audio server as it is NOW, through a
/// second view of it: `pactl list short sources`, whose names are the ground
/// truth for which sources are monitors.
#[test]
fn the_parser_agrees_with_the_live_audio_server() {
    if !have("pactl") {
        eprintln!("SKIP: pactl not installed");
        return;
    }
    let long = Command::new("pactl")
        .env("LC_ALL", "C")
        .args(["list", "sources"])
        .output()
        .expect("pactl");
    let short = Command::new("pactl")
        .env("LC_ALL", "C")
        .args(["list", "short", "sources"])
        .output()
        .expect("pactl");
    if !long.status.success() || !short.status.success() {
        eprintln!("SKIP: pactl cannot reach an audio server here");
        return;
    }
    let Some(m) = parse_mics(
        "pactl",
        &String::from_utf8_lossy(&long.stdout),
        None,
        "live",
    ) else {
        return;
    };
    let mut expected: Vec<String> = String::from_utf8_lossy(&short.stdout)
        .lines()
        .filter_map(|l| l.split('\t').nth(1).map(str::to_string))
        .filter(|n| !n.ends_with(".monitor") && n.len() < 192)
        .collect();
    let mut got = m.names.clone();
    expected.sort();
    got.sort();
    assert_eq!(got, expected, "the long-list parser and the short list disagree about which sources are inputs");
}

// ------------------------------------------------------------ command lines

fn arg_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

/// gpu-screen-recorder gets the desktop and the microphone MERGED into one
/// track, and otherwise exactly the command line it got before the refactor -
/// pinned whole, so a reordering or a dropped flag is a failure here rather
/// than a surprise on the user's machine.
#[test]
fn gpu_screen_recorder_merges_the_microphone_into_one_track() {
    let mic = format!("mic={}", HEADSET);
    let Some(rec) = build("capture", &["gsr=1", &mic, "codec=2", "quality=1", "out=/v/Manual.mp4"]) else { return };
    assert_eq!(rec.backend, "gpu-screen-recorder");
    assert_eq!(
        rec.args,
        [
            "gpu-screen-recorder", "-w", "screen", "-f", "60", "-s", "1920x1080",
            "-a", &format!("default_output|{}", HEADSET),
            "-k", "av1", "-q", "high", "-o", "/v/Manual.mp4",
        ]
    );
    assert!(rec.note.contains("microphone") && rec.note.contains(HEADSET), "{}", rec.note);

    let Some(replay) = build(
        "capture",
        &["gsr=1", &mic, "codec=1", "quality=2", "replay=1", "replay_seconds=40", "container=1", "wayland=1", "out=/v/dir"],
    ) else { return };
    assert_eq!(
        replay.args,
        [
            "gpu-screen-recorder", "-w", "portal", "-f", "60", "-s", "3840x2160",
            "-a", &format!("default_output|{}", HEADSET),
            "-r", "40", "-k", "hevc", "-q", "very_high", "-c", "mkv", "-o", "/v/dir",
        ]
    );

    let Some(off) = build("capture", &["gsr=1", "mic=", "out=/v/x.mp4"]) else { return };
    assert_eq!(arg_after(&off.args, "-a"), Some("default_output"));
    assert!(off.note.contains("microphone disabled"), "{}", off.note);
}

/// wf-recorder can take ONE audio source, so it records the desktop - and the
/// log must SAY the microphone is missing. It used to print "Desktop + Mic
/// [Merged]" over a command that never mentioned the microphone. Its `-a`
/// takes an optional value, so the device must be attached: `--audio=DEV`.
#[test]
fn wf_recorder_says_the_microphone_is_missing_instead_of_claiming_it() {
    let mic = format!("mic={}", HEADSET);
    let Some(b) = build("capture", &["wf=1", "wayland=1", &mic, "desktop=sink.monitor", "out=/v/a.mp4"]) else { return };
    assert_eq!(b.backend, "wf-recorder");
    assert_eq!(b.args, ["wf-recorder", "--audio=sink.monitor", "-f", "/v/a.mp4"]);
    assert!(b.note.contains("NOT in this recording"), "the note hides the missing microphone: {}", b.note);
    assert!(!b.note.contains("merged") && !b.note.contains("mixed"), "{}", b.note);
}

/// The ffmpeg fallback: both sources as inputs, mixed at full level into one
/// track. amix's default halves every input, which would put the voice 6 dB
/// down - `normalize=0` is load-bearing, not decoration.
#[test]
fn ffmpeg_mixes_the_microphone_in() {
    let mic = format!("mic={}", HEADSET);
    let Some(b) = build("capture", &["ffmpeg=1", "pulse=1", &mic, "desktop=sink.monitor", "out=/v/a.mp4"]) else { return };
    assert_eq!(b.backend, "ffmpeg");
    let a = &b.args;
    let pulse_inputs: Vec<&str> = a
        .windows(4)
        .filter(|w| w[0] == "-f" && w[1] == "pulse" && w[2] == "-i")
        .map(|w| w[3].as_str())
        .collect();
    assert_eq!(pulse_inputs, ["sink.monitor", HEADSET], "{:?}", a);
    assert_eq!(
        arg_after(a, "-filter_complex"),
        Some("[1:a][2:a]amix=inputs=2:duration=longest:normalize=0[a]")
    );
    let maps: Vec<&str> = a.windows(2).filter(|w| w[0] == "-map").map(|w| w[1].as_str()).collect();
    assert_eq!(maps, ["0:v", "[a]"]);
    assert!(b.note.contains("mixed into one track"), "{}", b.note);
}

/// AV1 through SVT-AV1 takes a numeric preset. "veryfast" made ffmpeg refuse
/// the whole command - and AV1 is this app's default codec on an RTX 40 card,
/// so the ffmpeg fallback could not record at all on the hardware it was
/// tuned for.
#[test]
fn ffmpeg_gives_svt_av1_a_preset_it_accepts() {
    let Some(av1) = build("capture", &["ffmpeg=1", "codec=2", "desktop=d", "out=/v/a.mp4"]) else { return };
    assert_eq!(arg_after(&av1.args, "-c:v"), Some("libsvtav1"));
    assert_eq!(arg_after(&av1.args, "-preset"), Some("10"));
    let Some(h264) = build("capture", &["ffmpeg=1", "codec=0", "desktop=d", "out=/v/a.mp4"]) else { return };
    assert_eq!(arg_after(&h264.args, "-preset"), Some("veryfast"));

    if !have("ffmpeg") {
        eprintln!("SKIP: ffmpeg not installed, so the preset is checked but not run");
        return;
    }
    for (codec, preset) in [("libsvtav1", "10"), ("libx264", "veryfast"), ("libx265", "veryfast")] {
        let ok = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi", "-i"])
            .arg("testsrc=size=320x240:rate=15:d=0.2")
            .args(["-c:v", codec, "-preset", preset, "-crf", "28", "-f", "null", "-"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        let listed = Command::new("ffmpeg")
            .args(["-hide_banner", "-encoders"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(codec))
            .unwrap_or(false);
        if listed {
            assert!(ok, "ffmpeg refuses `-c:v {} -preset {}`", codec, preset);
        }
    }
}

/// When ffmpeg cannot reach the audio server there is no audio at all, and
/// the note says so rather than naming sources that are not in the command.
#[test]
fn ffmpeg_without_an_audio_server_records_no_audio_and_says_so() {
    let mic = format!("mic={}", HEADSET);
    let Some(b) = build("capture", &["ffmpeg=1", "pulse=0", &mic, "desktop=d", "out=/v/a.mp4"]) else { return };
    assert!(!b.args.iter().any(|a| a == "pulse"), "{:?}", b.args);
    assert!(!b.args.iter().any(|a| a == "-c:a"), "{:?}", b.args);
    assert!(b.note.starts_with("NO AUDIO"), "{}", b.note);
}

/// Nothing to record with, and a command too long for its buffer, both build
/// nothing - never a truncated command line.
#[test]
fn no_recorder_or_an_overlong_command_builds_nothing() {
    let Some(none) = build("capture", &["out=/v/a.mp4"]) else { return };
    assert_eq!((none.backend.as_str(), none.args.len()), ("none", 0));

    let huge = format!("out=/{}", "d/".repeat(2100));
    let Some(over) = build("capture", &["ffmpeg=1", "desktop=d", &huge]) else { return };
    assert_eq!(over.backend, "none", "an overflowing command was reported as buildable");
    assert!(over.overflow);
}

/// Voice recording: ffmpeg writes AAC in .m4a and never overwrites; without
/// ffmpeg, parecord writes .wav with the device attached to its flag.
#[test]
fn voice_recording_uses_ffmpeg_then_parecord() {
    let mic = format!("mic={}", HEADSET);
    let Some(ff) = build("voice", &["ffmpeg=1", "parecord=1", &mic, "out_base=/v/Voice_1"]) else { return };
    assert_eq!(ff.backend, "ffmpeg");
    assert_eq!(ff.path, "/v/Voice_1.m4a");
    assert_eq!(ff.args.last().map(String::as_str), Some("/v/Voice_1.m4a"));
    assert!(ff.args.iter().any(|a| a == "-n") && !ff.args.iter().any(|a| a == "-y"), "{:?}", ff.args);
    assert_eq!(arg_after(&ff.args, "-i"), Some(HEADSET));
    assert!(ff.args.iter().any(|a| a == "-vn"), "{:?}", ff.args);

    let Some(pa) = build("voice", &["ffmpeg=0", "parecord=1", &mic, "out_base=/v/Voice_1"]) else { return };
    assert_eq!(pa.backend, "parecord");
    assert_eq!(pa.args, ["parecord", &format!("--device={}", HEADSET), "--file-format=wav", "/v/Voice_1.wav"]);

    let Some(neither) = build("voice", &["ffmpeg=0", "parecord=0", &mic, "out_base=/v/Voice_1"]) else { return };
    assert_eq!(neither.backend, "none");

    // A voice recording without a microphone is not a recording.
    let Some(no_mic) = build("voice", &["ffmpeg=1", "mic=", "out_base=/v/Voice_1"]) else { return };
    assert_eq!((no_mic.backend.as_str(), no_mic.args.len()), ("none", 0));
}

// ------------------------------------------------------- the tone reaches it

/// Mean level, in dB, of the part of `file`'s audio near `freq` Hz.
fn band_level(file: &Path, freq: u32) -> f64 {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-nostdin", "-i"])
        .arg(file)
        .args(["-vn", "-af"])
        .arg(format!("bandpass=f={}:width_type=h:w=40,volumedetect", freq))
        .args(["-f", "null", "-"])
        .output()
        .expect("run ffmpeg volumedetect");
    let err = String::from_utf8_lossy(&out.stderr);
    let line = err
        .lines()
        .find(|l| l.contains("mean_volume:"))
        .unwrap_or_else(|| panic!("no volumedetect output for {}:\n{}", file.display(), err));
    let v = line.split("mean_volume:").nth(1).unwrap().trim().trim_end_matches("dB").trim();
    if v == "-inf" { -1000.0 } else { v.parse().expect("dB") }
}

fn streams(file: &Path) -> Vec<String> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "stream=codec_type,codec_name", "-of", "csv=p=0"])
        .arg(file)
        .output()
        .expect("ffprobe");
    String::from_utf8_lossy(&out.stdout).lines().map(|l| l.trim().to_string()).collect()
}

fn run(args: &[String]) -> std::process::Output {
    Command::new(&args[0]).args(&args[1..]).stdin(Stdio::null()).output().expect("run the built command")
}

/// The builder's ffmpeg command, RUN, with tone generators where the sound
/// card would be: a 440 Hz desktop and an 880 Hz microphone. The microphone's
/// tone must be in the recording - and must be absent from the same recording
/// made with the microphone off, or the level comparison measures nothing.
#[test]
fn the_microphone_is_audible_in_an_ffmpeg_recording() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIP: ffmpeg/ffprobe not installed");
        return;
    }
    let dir = workdir("mix");
    let with = dir.join("with_mic.mp4");
    let without = dir.join("without_mic.mp4");
    let common = [
        "ffmpeg=1",
        "codec=0",
        "quality=0",
        "audio_format=lavfi",
        "video_lavfi=testsrc=size=320x240:rate=15:d=2",
        "desktop=sine=f=440:d=2",
    ];
    for (out, mic) in [(&with, "mic=sine=f=880:d=2"), (&without, "mic=")] {
        let out_arg = format!("out={}", out.display());
        let mut kv: Vec<&str> = common.to_vec();
        kv.push(mic);
        kv.push(&out_arg);
        let Some(b) = build("capture", &kv) else { return };
        let r = run(&b.args);
        assert!(r.status.success(), "the built ffmpeg command failed:\n{:?}\n{}", b.args, String::from_utf8_lossy(&r.stderr));
    }

    for f in [&with, &without] {
        let s = streams(f);
        assert!(s.iter().any(|l| l.starts_with("h264,video")) || s.iter().any(|l| l == "h264,video"), "{}: {:?}", f.display(), s);
        assert_eq!(s.iter().filter(|l| l.ends_with("audio")).count(), 1, "{}: one audio track expected, got {:?}", f.display(), s);
    }

    let voice_with = band_level(&with, 880);
    let voice_without = band_level(&without, 880);
    let desktop_with = band_level(&with, 440);
    let desktop_without = band_level(&without, 440);
    eprintln!(
        "880 Hz (voice): {:.1} dB with the mic, {:.1} dB without | 440 Hz (desktop): {:.1} / {:.1} dB",
        voice_with, voice_without, desktop_with, desktop_without
    );
    assert!(voice_with - voice_without >= 15.0, "the microphone is not audible in the recording: {:.1} vs {:.1} dB", voice_with, voice_without);
    assert!(voice_with > -35.0, "the microphone is in the recording but barely: {:.1} dB", voice_with);
    // Mixing the voice in must not have dropped or buried the desktop.
    assert!(desktop_with > -35.0 && (desktop_with - desktop_without).abs() < 3.0, "desktop audio moved: {:.1} vs {:.1} dB", desktop_with, desktop_without);
}

/// A voice recording, RUN: an audio-only AAC file holding the microphone. And
/// a second recording aimed at the same path must not replace it.
///
/// That last check is on the FILE, not ffmpeg's exit status: ffmpeg 9 prints
/// "already exists. Exiting." and exits 0 when `-n` refuses. (Which is why the
/// HUD treats any recorder exit it did not ask for as a failure, whatever the
/// code.) And the second run records a DIFFERENT tone, because re-encoding the
/// same deterministic input would rewrite identical bytes and a byte
/// comparison would pass whether or not the file was replaced.
#[test]
fn a_voice_recording_is_an_audio_file_of_the_microphone() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIP: ffmpeg/ffprobe not installed");
        return;
    }
    let dir = workdir("voice");
    let base = dir.join("Voice_Recording_1");
    let base_arg = format!("out_base={}", base.display());
    let Some(b) = build("voice", &["ffmpeg=1", "audio_format=lavfi", "mic=sine=f=880:d=2", &base_arg]) else { return };
    let path = PathBuf::from(&b.path);
    assert_eq!(path.extension().and_then(|e| e.to_str()), Some("m4a"));

    let r = run(&b.args);
    assert!(r.status.success(), "the built voice command failed:\n{:?}\n{}", b.args, String::from_utf8_lossy(&r.stderr));
    assert_eq!(streams(&path), ["aac,audio"], "a voice recording must be audio only");
    let level = band_level(&path, 880);
    assert!(level > -35.0, "the microphone is not in the voice recording: {:.1} dB", level);

    let Some(second) = build("voice", &["ffmpeg=1", "audio_format=lavfi", "mic=sine=f=440:d=1", &base_arg]) else { return };
    assert_eq!(second.path, b.path, "the collision this checks needs the same path");
    let _ = run(&second.args);
    let first_tone = band_level(&path, 880);
    let second_tone = band_level(&path, 440);
    assert!(
        first_tone > -35.0 && second_tone < first_tone - 15.0,
        "a second voice recording replaced the first: 880 Hz {:.1} dB, 440 Hz {:.1} dB",
        first_tone,
        second_tone
    );
}

/// The value this builder passes to gpu-screen-recorder's `-a` must be one
/// the installed gpu-screen-recorder ACCEPTS - its command-line contract is
/// what broke the device list in the first place. A bogus window id makes it
/// stop after validating audio and before capturing anything, so the first
/// error it prints says whether the audio spec passed. The bogus-device run
/// is the control that shows the probe can tell the difference.
#[test]
fn the_installed_gpu_screen_recorder_accepts_the_merged_audio_spec() {
    if !have("gpu-screen-recorder") || !have("pactl") {
        eprintln!("SKIP: gpu-screen-recorder or pactl not installed");
        return;
    }
    let long = Command::new("pactl").env("LC_ALL", "C").args(["list", "sources"]).output().expect("pactl");
    if !long.status.success() {
        eprintln!("SKIP: no audio server");
        return;
    }
    let Some(m) = parse_mics("pactl", &String::from_utf8_lossy(&long.stdout), None, "gsrlive") else { return };
    let Some(mic) = m.names.first() else {
        eprintln!("SKIP: this machine has no microphone input");
        return;
    };
    let dir = workdir("gsrprobe");
    let probe = |audio: &str| -> String {
        let mut child = Command::new("gpu-screen-recorder")
            .args(["-w", "0x7ffffff0", "-a", audio, "-o"])
            .arg(dir.join("never.mp4"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("run gpu-screen-recorder");
        let start = Instant::now();
        while child.try_wait().unwrap().is_none() {
            if start.elapsed() > Duration::from_secs(10) {
                let _ = child.kill();
                let _ = child.wait();
                panic!("gpu-screen-recorder did not stop at the bogus window id - it may be capturing");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let out = child.wait_with_output().unwrap();
        format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
    };
    let bogus = probe("default_output|y_no_such_device_xyz");
    assert!(
        bogus.contains("is not a valid audio device"),
        "control failed: a bogus device was not refused, so this probe cannot tell anything apart:\n{}",
        bogus
    );
    let merged = format!("default_output|{}", mic);
    let real = probe(&merged);
    assert!(
        !real.contains("is not a valid audio device"),
        "gpu-screen-recorder refuses `-a {}`:\n{}",
        merged,
        real
    );
    assert!(!dir.join("never.mp4").exists(), "the probe wrote a recording");
}

// ------------------------------------------------ the GUI header's two rules

const GUI_HARNESS: &str = r#"
#include "shadowplay_gui.h"
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    const char* dir = argv[2];
    char log[1024];
    snprintf(log, sizeof log, "%s/child.log", dir);

    if (!strcmp(argv[1], "orphan")) {
        // A stand-in recorder that reports SIGINT - the signal every real
        // recorder finalizes its file on - and otherwise runs forever.
        char script[2048];
        snprintf(script, sizeof script,
                 "trap 'echo INT > \"%s/got_int\"; exit 0' INT; : > \"%s/ready\"; "
                 "while :; do sleep 0.05; done", dir, dir);
        sp_argv cmd;
        sp_argv_init(&cmd);
        sp_arg(&cmd, "sh");
        sp_arg(&cmd, "-c");
        sp_arg(&cmd, script);
        pid_t pid = spawn_encoder(&cmd, log);
        printf("%d\n", (int)pid);
        fflush(stdout);
        for (;;) pause();
    }
    if (!strcmp(argv[1], "ctrlc")) {
        // Ctrl+C in a terminal, end to end: a recorder started the way the
        // HUD starts one, SIGINT from the test to this whole process group,
        // and the HUD's own main loop doing the stopping.
        signal(SIGINT, handle_sigint);
        signal(SIGTERM, handle_sigint);
        sp_argv cmd;
        if (argc > 3 && !strcmp(argv[3], "ffmpeg")) {
            char base[1024];
            snprintf(base, sizeof base, "%s/Voice_Recording_ctrlc", dir);
            sp_voice v;
            memset(&v, 0, sizeof v);
            v.mic = "sine=f=880,arealtime";
            v.out_base = base;
            v.has_ffmpeg = 1;
            v.audio_format = "lavfi";
            if (sp_build_voice(&cmd, &v, voice_path, sizeof voice_path) != SP_FFMPEG) return 3;
        } else {
            // Counts every SIGINT it gets, and exits half a second after the
            // first - long enough for a second one to arrive and be counted.
            char script[2048];
            snprintf(script, sizeof script,
                     "trap 'echo INT >> \"%s/ints\"; stopping=1' INT; : > \"%s/ready\"; "
                     "while :; do if [ -n \"$stopping\" ]; then sleep 0.5; exit 0; fi; sleep 0.05; done",
                     dir, dir);
            sp_argv_init(&cmd);
            sp_arg(&cmd, "sh"); sp_arg(&cmd, "-c"); sp_arg(&cmd, script);
        }
        voice_pid = spawn_encoder(&cmd, log);
        voice_recording = 1;
        voice_started = time(NULL);
        printf("%d\n", (int)voice_pid);
        fflush(stdout);
        for (;;) {
            update_shadowplay_gui();
            usleep(16666);
        }
    }
    if (!strcmp(argv[1], "watchdog")) {
        sp_argv dies, lives;
        sp_argv_init(&dies);
        sp_arg(&dies, "sh"); sp_arg(&dies, "-c"); sp_arg(&dies, "exit 3");
        sp_argv_init(&lives);
        sp_arg(&lives, "sleep"); sp_arg(&lives, "30");
        record_pid = spawn_encoder(&dies, log);
        recording = 1;
        voice_pid = spawn_encoder(&lives, log);
        voice_recording = 1;
        usleep(500000);
        reap_if_dead(&record_pid, &recording, "screen recorder", log, "Recording failed");
        reap_if_dead(&voice_pid, &voice_recording, "voice recorder", log, "Voice recording failed");
        printf("recording=%d record_pid=%d voice_recording=%d voice_alive=%d failures=%d\n",
               recording, (int)record_pid, voice_recording, voice_pid > 0, capture_failures);
        if (voice_pid > 0) { kill(voice_pid, SIGKILL); waitpid(voice_pid, NULL, 0); }
        return 0;
    }
    return 2;
}
"#;

/// The GUI header compiled against real X11 headers, for the two process
/// rules that live there. Skipped - with a notice - where X11 development
/// files are absent; `shadowplay_builds.rs` covers the headless build.
fn gui_harness() -> Option<&'static PathBuf> {
    static BIN: OnceLock<Option<PathBuf>> = OnceLock::new();
    BIN.get_or_init(|| {
        let cc = c_compiler()?;
        let dir = workdir("gui_harness");
        let src = dir.join("gui.c");
        std::fs::write(&src, GUI_HARNESS).unwrap();
        let bin = dir.join("gui");
        let out = Command::new(cc)
            .args(["-std=gnu11", "-O1", "-w"])
            .arg("-I")
            .arg(repo().join("c_src"))
            .arg(&src)
            .arg("-o")
            .arg(&bin)
            .arg("-lX11")
            .output()
            .expect("run the C compiler");
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            if err.contains("X11/Xlib.h") || err.contains("-lX11") {
                eprintln!("SKIP: no X11 development files; the GUI header's process rules are unchecked here");
                return None;
            }
            panic!("the GUI header does not compile:\n{}", err);
        }
        Some(bin)
    })
    .as_ref()
}

fn wait_for(path: &Path, limit: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < limit {
        if path.exists() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    path.exists()
}

/// A recorder must not outlive the HUD. Kill the HUD with SIGKILL - no
/// cleanup can run - and the recorder must still be told SIGINT, which is
/// what makes it finalize its file and, for a voice recording, stop
/// listening to the microphone.
#[test]
fn a_recorder_is_stopped_when_the_hud_dies_without_cleaning_up() {
    let Some(bin) = gui_harness() else { return };
    let dir = workdir("orphan");
    let mut hud = Command::new(bin)
        .arg("orphan")
        .arg(&dir)
        .stdout(Stdio::piped())
        .spawn()
        .expect("start the HUD stand-in");
    let mut first = String::new();
    {
        use std::io::BufRead;
        let stdout = hud.stdout.take().unwrap();
        std::io::BufReader::new(stdout).read_line(&mut first).unwrap();
    }
    let recorder: i32 = first.trim().parse().expect("recorder pid");
    assert!(recorder > 0);
    assert!(wait_for(&dir.join("ready"), Duration::from_secs(5)), "the recorder stand-in never started");

    hud.kill().expect("SIGKILL the HUD");
    let _ = hud.wait();

    let told = wait_for(&dir.join("got_int"), Duration::from_secs(5));
    if !told {
        let _ = Command::new("kill").arg("-9").arg(recorder.to_string()).status();
    }
    assert!(told, "the recorder (pid {}) kept running after the HUD was killed", recorder);
}

/// A recorder that exits on its own must turn its state off and count as a
/// failure; one that is still running must be left alone. This is the check
/// that stops the HUD saying "RECORDING" over a recorder that is gone.
#[test]
fn a_recorder_that_dies_on_its_own_turns_its_state_off() {
    let Some(bin) = gui_harness() else { return };
    let dir = workdir("watchdog");
    let out = Command::new(bin).arg("watchdog").arg(&dir).output().expect("run watchdog");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{}\n{}", text, String::from_utf8_lossy(&out.stderr));
    assert!(
        text.contains("recording=0 record_pid=0"),
        "a dead recorder was left looking alive: {}",
        text
    );
    assert!(text.contains("failures=1"), "the failure was not counted: {}", text);
    assert!(
        text.contains("voice_recording=1 voice_alive=1"),
        "a running recorder was reaped: {}",
        text
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("exit code 3") && err.contains("child.log"), "the error does not say what happened or where to look:\n{}", err);
}

/// Process group of `pid`, from /proc (field 5; parsed after the last `)`
/// because the command name may itself contain spaces or parentheses).
fn process_group(pid: u32) -> Option<u32> {
    let stat = std::fs::read_to_string(format!("/proc/{}/stat", pid)).ok()?;
    let after = &stat[stat.rfind(')')? + 1..];
    after.split_whitespace().nth(2)?.parse().ok()
}

/// What a terminal does on Ctrl+C: SIGINT to every process in the job's
/// process group at once.
fn terminal_ctrl_c(pgid: u32) {
    let ok = Command::new("kill")
        .args(["-INT", "--", &format!("-{}", pgid)])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    assert!(ok, "could not signal process group {}", pgid);
}

fn wait_exit(child: &mut std::process::Child, limit: Duration) -> Option<std::process::ExitStatus> {
    let start = Instant::now();
    while start.elapsed() < limit {
        if let Some(st) = child.try_wait().unwrap() {
            return Some(st);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    None
}

/// Start the HUD stand-in as a job of its own, the way a shell starts one.
///
/// Its output goes to a FILE, not a pipe. The first version read the recorder
/// pid from a pipe and dropped it, so the HUD's shutdown message hit a closed
/// pipe and SIGPIPE killed it before it cleaned up - and the ffmpeg variant
/// still passed, because the kernel's parent-death SIGINT finalized the file
/// instead. The tests below now also require the HUD's own "saved" line.
fn start_hud_job(bin: &Path, dir: &Path, recorder: &str) -> (std::process::Child, u32, PathBuf) {
    use std::os::unix::process::CommandExt;
    let out_path = dir.join("hud_out.txt");
    let out = std::fs::File::create(&out_path).unwrap();
    let err = out.try_clone().unwrap();
    let hud = Command::new(bin)
        .arg("ctrlc")
        .arg(dir)
        .arg(recorder)
        .process_group(0)
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .spawn()
        .expect("start the HUD stand-in");
    let start = Instant::now();
    loop {
        let text = std::fs::read_to_string(&out_path).unwrap_or_default();
        if let Some(line) = text.lines().next().filter(|_| text.contains('\n')) {
            let recorder_pid: u32 = line.trim().parse().expect("recorder pid");
            return (hud, recorder_pid, out_path);
        }
        assert!(start.elapsed() < Duration::from_secs(10), "the HUD stand-in never reported its recorder");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Ctrl+C in a terminal signals the whole foreground job. If the recorders
/// are in that job they get the terminal's SIGINT AND the HUD's, and ffmpeg
/// reads the second one as "exit immediately" - measured: it abandoned the
/// file's index halfway through, leaving a voice recording no player opens,
/// while the HUD printed "Voice recording saved". A stand-in recorder counts
/// the SIGINTs it receives, and the answer must be exactly one.
#[test]
fn ctrl_c_in_the_terminal_reaches_a_recorder_exactly_once() {
    let Some(bin) = gui_harness() else { return };
    let dir = workdir("ctrlc_count");
    let (mut hud, recorder, out_path) = start_hud_job(bin, &dir, "standin");
    assert!(wait_for(&dir.join("ready"), Duration::from_secs(5)), "the stand-in recorder never started");

    let hud_group = process_group(hud.id()).expect("HUD process group");
    let recorder_group = process_group(recorder).expect("recorder process group");
    assert_ne!(
        recorder_group, hud_group,
        "the recorder is in the HUD's process group, so the terminal's Ctrl+C reaches it directly"
    );

    terminal_ctrl_c(hud_group);
    let status = wait_exit(&mut hud, Duration::from_secs(10));
    if status.is_none() {
        let _ = hud.kill();
        let _ = Command::new("kill").arg("-9").arg(recorder.to_string()).status();
    }
    let out = std::fs::read_to_string(&out_path).unwrap_or_default();
    assert_eq!(status.and_then(|s| s.code()), Some(0), "the HUD did not shut down cleanly on Ctrl+C:\n{}", out);
    assert!(out.contains("Voice recording saved"), "the HUD's own shutdown did not stop the recorder:\n{}", out);
    let ints = std::fs::read_to_string(dir.join("ints")).unwrap_or_default();
    assert_eq!(
        ints.lines().count(),
        1,
        "the recorder received {} SIGINTs for one Ctrl+C",
        ints.lines().count()
    );
}

/// The same Ctrl+C with a real ffmpeg voice recorder: the file must come out
/// finished - ffprobe must be able to read its duration, which it cannot for
/// an .m4a whose index was never written.
#[test]
fn ctrl_c_in_the_terminal_leaves_a_playable_voice_recording() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIP: ffmpeg/ffprobe not installed");
        return;
    }
    let Some(bin) = gui_harness() else { return };
    let dir = workdir("ctrlc_ffmpeg");
    let (mut hud, recorder, out_path) = start_hud_job(bin, &dir, "ffmpeg");
    let file = dir.join("Voice_Recording_ctrlc.m4a");
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10)
        && std::fs::metadata(&file).map(|m| m.len()).unwrap_or(0) < 4096
    {
        std::thread::sleep(Duration::from_millis(50));
    }
    std::thread::sleep(Duration::from_millis(500));

    terminal_ctrl_c(process_group(hud.id()).expect("HUD process group"));
    let status = wait_exit(&mut hud, Duration::from_secs(10));
    if status.is_none() {
        let _ = hud.kill();
        let _ = Command::new("kill").arg("-9").arg(recorder.to_string()).status();
    }
    let out = std::fs::read_to_string(&out_path).unwrap_or_default();
    assert_eq!(status.and_then(|s| s.code()), Some(0), "the HUD did not shut down cleanly on Ctrl+C:\n{}", out);
    assert!(out.contains("Voice recording saved"), "the file was not finished by the HUD's own shutdown:\n{}", out);

    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"])
        .arg(&file)
        .output()
        .expect("ffprobe");
    let text = format!("{}{}", String::from_utf8_lossy(&probe.stdout), String::from_utf8_lossy(&probe.stderr));
    let secs: f64 = text.trim().parse().unwrap_or_else(|_| panic!("the voice recording is unreadable after Ctrl+C: {}", text.trim()));
    assert!(secs > 0.5, "the voice recording is only {:.2}s long", secs);
}
