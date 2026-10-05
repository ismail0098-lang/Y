// c_src/shadowplay_gui.h
// Graphical X11 ShadowPlay Overlay HUD for Y compiler runtime

#ifndef SHADOWPLAY_GUI_H
#define SHADOWPLAY_GUI_H

#include <stdint.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// The entry points below are EXPORTED (non-static) on purpose: `Y` compiles a
// `.ysu` program to a module that `declare`s them, and a `static` definition
// emits no symbol at all, so the link fails with `undefined reference to
// 'init_shadowplay_gui'`. They were made `static inline` at some point to
// silence a static-after-non-static warning, which fixed the warning and
// killed the only application that calls them. `shadowplay_api_is_linkable`
// pins that they stay exported; `tests/shadowplay_builds.rs` builds the app.
//
// This header is included by exactly one translation unit (`c_src/runtime.c`),
// so exporting them cannot produce a duplicate symbol.
//
// Y_NO_X11 compiles a stub surface instead. Without it, un-static-ing the
// implementation would leave every Y binary referencing XOpenDisplay, making
// libX11 a hard requirement for compiling ANY Y program - the exact
// regression that removing the unconditional `-lX11` was meant to fix.
// ---------------------------------------------------------------------------

#if defined(Y_NO_X11)

static void y_no_x11_notice(const char* fn) {
    fprintf(stderr,
            "[ShadowPlay] %s: this binary was built without X11 support "
            "(libX11 was not available at compile time).\n", fn);
}

int32_t init_shadowplay_gui(void)      { y_no_x11_notice("init_shadowplay_gui"); return -1; }
int32_t update_shadowplay_gui(void)    { return -1; }
void    cleanup_shadowplay_gui(void)   { }
int32_t is_overlay_visible(void)       { return 0; }
int32_t get_instant_replay_state(void) { return 0; }
int32_t get_recording_state(void)      { return 0; }
int32_t get_broadcast_state(void)      { return 0; }
int32_t get_file_format_state(void)    { return 0; }
int32_t get_quality_state(void)        { return 0; }
int32_t get_codec_state(void)          { return 0; }
int32_t get_replay_duration(void)      { return 0; }
int32_t get_replay_duration_idx(void)  { return 0; }
int32_t get_microphone_index(void)     { return 0; }
int32_t get_indicator_state(void)      { return 0; }
int32_t get_voice_recording_state(void) { return 0; }
int32_t get_capture_failure_count(void) { return 0; }
int32_t print_microphone_label(void)   { return 0; }
void    get_microphone_name(char* out_buf) { if (out_buf) out_buf[0] = '\0'; }

#else

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/prctl.h>
#include <sys/stat.h>

// Which microphones exist, and what each recorder is told. Pure code with no
// X11 in it, split out so it can be tested without a display or a sound card.
#include "shadowplay_capture.h"

// Window geometry. These were five separate literals (600/480 at the centring
// site, again at creation, and the footer's y hardcoded against them), so a
// layout change had to be made in every one of them or the footer fell off.
#define HUD_W 720
#define HUD_H 520
#define CARD_GAP 20
// The right-hand settings column. At the old 300px, "[  AV1  ]" ran into the
// "Replay Length:" label beside it.
#define RIGHT_COL 340

// Custom X11 Error Handler to prevent crashes if a key is already grabbed
static int x11_error_handler(Display* d, XErrorEvent* e) {
    char err_msg[256];
    XGetErrorText(d, e->error_code, err_msg, sizeof(err_msg));
    fprintf(stderr, "[X11 Warning] Handled X11 protocol error: %s (request_code=%d)\n", err_msg, e->request_code);
    return 0;
}

void cleanup_shadowplay_gui(void);
static void draw_ui(void);

static Display* dpy;

// The connection is gone, so every Xlib entry point is now off limits --
// including the ones cleanup reaches through stop_manual_recording ->
// show_toast -> update_window_layout/draw_ui. This used to call cleanup with
// `dpy` still set and recurse straight back into XMoveResizeWindow on the dead
// connection. Nulling `dpy` FIRST is what makes the teardown X-free:
// show_toast, update_window_layout and draw_ui all early-return on `!dpy`.
static int x11_io_error_handler(Display* d) {
    (void)d;
    dpy = NULL;
    fprintf(stderr, "[X11 Fatal] X connection lost/IO Error occurred.\n");
    cleanup_shadowplay_gui();
    fflush(NULL);
    _exit(1);
}

// ShadowPlay Overlay States
static Window win = 0;
static Window root = 0;
static int screen = 0;
static int visible = 1;
static XFontStruct* hud_font = NULL;

static int screen_width = 1920;
static int screen_height = 1080;
static char default_audio_dev[256] = "default";
static int instant_replay = 0; // 0 = OFF, 1 = ON
static int recording = 0;       // 0 = OFF, 1 = ON
static int broadcast = 0;       // 0 = OFF, 1 = ON
static int file_format = 0;     // 0 = MP4, 1 = MKV
static int quality = 1;         // 0 = 720p, 1 = 1080p, 2 = 4K
static int video_codec = 2;     // 0 = H264, 1 = HEVC, 2 = AV1
static int show_indicator = 1;  // 0 = hide the on-screen recording dot, 1 = show it
static int voice_recording = 0; // 0 = OFF, 1 = recording the microphone on its own

// Keyboard navigation
static int selected_idx = 0;    // 0=Replay, 1=Record, 2=Broadcast, 3=Format, 4=Quality,
                                // 5=Codec, 6=ReplayLength, 7=Keybind, 8=Mic, 9=Indicator,
                                // 10=Voice Record
#define SEL_VOICE 10
#define NUM_ITEMS 11

// The four cards across the top, left to right. Voice Record arrived after
// the settings rows were numbered, so its index is not its position; every
// piece of card navigation and drawing goes through this table instead of
// assuming the cards are 0..2.
static const int card_order[4] = {0, 1, SEL_VOICE, 2};

static int card_position(int idx) {
    for (int p = 0; p < 4; p++) {
        if (card_order[p] == idx) return p;
    }
    return -1;
}

// Is the small corner dot supposed to be on screen right now? Four call sites
// asked this question and would otherwise have to stay in agreement by hand:
// the window layout, draw_ui's early-out, draw_ui's indicator branch, and the
// event loop's per-tick repaint. Disagreement leaves a mapped window with
// nothing drawn in it, or a dot that never repaints.
static int indicator_wanted(void) {
    return show_indicator && (recording || instant_replay || voice_recording);
}

// One mark per active capture: replay, screen recording, voice.
static int indicator_marks(void) {
    return (instant_replay != 0) + (recording != 0) + (voice_recording != 0);
}

// Two marks fit the original 70px corner box; the third needs room.
static int indicator_width(void) {
    return indicator_marks() > 2 ? 94 : 70;
}

static pid_t record_pid = 0;
static pid_t replay_pid = 0;
static pid_t voice_pid = 0;
static time_t voice_started = 0;
static char voice_path[512] = "";

// Bumped whenever a recorder exits on its own. The Y program polls it to tell
// "the recording stopped because it failed" from "the user stopped it and it
// was saved", which a single on/off state cannot say.
static int capture_failures = 0;

static int has_gpu_screen_recorder = 0;
static int has_wf_recorder = 0;
static int has_ffmpeg = 0;
static int has_parecord = 0;

#define RECORD_LOG "/tmp/y_recording_log.txt"
#define REPLAY_LOG "/tmp/y_replay_log.txt"
#define VOICE_LOG  "/tmp/y_voice_log.txt"

static int replay_duration_idx = 2; // 0=20s, 1=30s, 2=40s, 3=60s
static int replay_durations[] = {20, 30, 40, 60};
static int replay_keybind_idx = 0; // 0=Alt+S, 1=Alt+F10, 2=Alt+R, 3=Alt+X

static void grab_hotkey(KeySym sym) {
    Window root = DefaultRootWindow(dpy);
    KeyCode code = XKeysymToKeycode(dpy, sym);
    if (code == 0) return;
    XGrabKey(dpy, code, Mod1Mask, root, True, GrabModeAsync, GrabModeAsync);
    XGrabKey(dpy, code, Mod1Mask | ShiftMask, root, True, GrabModeAsync, GrabModeAsync);
    XGrabKey(dpy, code, Mod1Mask | Mod2Mask, root, True, GrabModeAsync, GrabModeAsync);
    XGrabKey(dpy, code, Mod1Mask | LockMask, root, True, GrabModeAsync, GrabModeAsync);
    XGrabKey(dpy, code, Mod1Mask | Mod2Mask | LockMask, root, True, GrabModeAsync, GrabModeAsync);
}

static void ungrab_hotkey(KeySym sym) {
    Window root = DefaultRootWindow(dpy);
    KeyCode code = XKeysymToKeycode(dpy, sym);
    if (code == 0) return;
    XUngrabKey(dpy, code, Mod1Mask, root);
    XUngrabKey(dpy, code, Mod1Mask | ShiftMask, root);
    XUngrabKey(dpy, code, Mod1Mask | Mod2Mask, root);
    XUngrabKey(dpy, code, Mod1Mask | LockMask, root);
    XUngrabKey(dpy, code, Mod1Mask | Mod2Mask | LockMask, root);
}

static void update_grabbed_keys() {
    if (!dpy) return;
    
    ungrab_hotkey(XK_s);
    ungrab_hotkey(XK_S);
    ungrab_hotkey(XK_F10);
    ungrab_hotkey(XK_r);
    ungrab_hotkey(XK_R);
    ungrab_hotkey(XK_x);
    ungrab_hotkey(XK_X);

    KeySym target_sym = XK_s;
    if (replay_keybind_idx == 1) target_sym = XK_F10;
    else if (replay_keybind_idx == 2) target_sym = XK_r;
    else if (replay_keybind_idx == 3) target_sym = XK_x;

    grab_hotkey(target_sym);
    if (target_sym == XK_s) grab_hotkey(XK_S);
    else if (target_sym == XK_r) grab_hotkey(XK_R);
    else if (target_sym == XK_x) grab_hotkey(XK_X);

    XFlush(dpy);
}

static char hw_gpu_name[128] = "Unknown GPU";
static int hw_vram_mb = 0;
static int hw_avx = 0;
static int hw_avx512 = 0;

static void load_hw_profile() {
    FILE* f = fopen(".ysu_hw_profile", "r");
    if (!f) {
        f = fopen("../.ysu_hw_profile", "r");
    }
    if (!f) return;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = '\0';
        char* eq = strchr(line, '=');
        if (eq) {
            *eq = '\0';
            char* key = line;
            char* val = eq + 1;
            if (strcmp(key, "GPU_NAME") == 0) {
                strncpy(hw_gpu_name, val, sizeof(hw_gpu_name) - 1);
            } else if (strcmp(key, "TOTAL_GLOBAL_MEM_MB") == 0) {
                hw_vram_mb = atoi(val);
            } else if (strcmp(key, "AVX") == 0) {
                hw_avx = (strcmp(val, "true") == 0);
            } else if (strcmp(key, "AVX512") == 0) {
                hw_avx512 = (strcmp(val, "true") == 0);
            }
        }
    }
    fclose(f);
}

static char toast_text[128] = "";
static int toast_active = 0;
static time_t toast_start_time = 0;

static void update_window_layout() {
    if (!dpy || !win) return;
    int screen_w = DisplayWidth(dpy, screen);
    int screen_h = DisplayHeight(dpy, screen);

    if (visible) {
        // Center settings menu
        XMoveResizeWindow(dpy, win, (screen_w - HUD_W) / 2, (screen_h - HUD_H) / 2, HUD_W, HUD_H);
        XMapWindow(dpy, win);
    } else if (toast_active) {
        // Top-right toast notification
        XMoveResizeWindow(dpy, win, screen_w - 300, 40, 280, 80);
        XMapWindow(dpy, win);
    } else if (indicator_wanted()) {
        // Top-right tiny status indicator
        int w = indicator_width();
        XMoveResizeWindow(dpy, win, screen_w - w - 20, 40, w, 32);
        XMapWindow(dpy, win);
    } else {
        // Completely hidden
        XUnmapWindow(dpy, win);
    }
}

static void show_toast(const char* message) {
    if (!dpy || !win) return;
    
    strncpy(toast_text, message, sizeof(toast_text) - 1);
    toast_active = 1;
    toast_start_time = time(NULL);

    update_window_layout();
    draw_ui();
}

// Microphones the audio server offers, found at startup. The HUD's Microphone
// row cycles through "Disabled" (index 0) and these (1..num_mics).
static sp_mic mics[SP_MAX_MICS];
static int num_mics = 0;
static int selected_mic_idx = 0; // 0 = microphone off, k = mics[k - 1]

// The source every recorder is given, or NULL when the microphone is off.
static const char* selected_mic_name(void) {
    if (selected_mic_idx < 1 || selected_mic_idx > num_mics) return NULL;
    return mics[selected_mic_idx - 1].name;
}

static const char* selected_mic_label(void) {
    if (selected_mic_idx < 1 || selected_mic_idx > num_mics) return "Disabled";
    return mics[selected_mic_idx - 1].label;
}

// Is `name` runnable? `command -v` rather than `which`, which is a separate
// package that a minimal system does not have.
static int have_command(const char* name) {
    char cmd[256];
    snprintf(cmd, sizeof cmd, "command -v '%s' >/dev/null 2>&1", name);
    return system(cmd) == 0;
}

// Run a command and return what it printed, NUL-terminated (malloc'd), or
// NULL if it could not run. Output past 256 KiB is dropped - but still READ:
// a child blocked writing to a full pipe would hang pclose() forever.
static char* read_command(const char* cmd) {
    FILE* fp = popen(cmd, "r");
    if (!fp) return NULL;
    size_t cap = 256 * 1024, len = 0;
    char* buf = malloc(cap);
    if (!buf) {
        pclose(fp);
        return NULL;
    }
    char chunk[4096];
    size_t got;
    while ((got = fread(chunk, 1, sizeof chunk, fp)) > 0) {
        size_t room = cap - 1 - len;
        size_t take = got < room ? got : room;
        memcpy(buf + len, chunk, take);
        len += take;
    }
    buf[len] = '\0';
    pclose(fp);
    return buf;
}

// Find the microphones and choose one. This used to parse the error older
// gpu-screen-recorder releases printed for `-a check_devices`; 6.x never gets
// as far as `-a` without a `-w`, so the list was always just "Disabled" and
// "Default Input" - see shadowplay_capture.h for why the latter can be the
// desktop instead of a voice.
static void detect_audio_devices() {
    char default_source[SP_NAME_LEN] = "";
    char default_sink[SP_NAME_LEN] = "";
    char* text;

    num_mics = 0;
    selected_mic_idx = 0;

    // LC_ALL=C: pactl translates its field names, and the parser reads them.
    text = read_command("LC_ALL=C pactl list sources 2>/dev/null");
    if (text) {
        num_mics = sp_parse_pactl_sources(text, mics, SP_MAX_MICS);
        free(text);
    }
    if (num_mics == 0) {
        text = read_command("gpu-screen-recorder --list-audio-devices 2>/dev/null");
        if (text) {
            num_mics = sp_parse_gsr_devices(text, mics, SP_MAX_MICS);
            free(text);
        }
    }
    text = read_command("LC_ALL=C pactl info 2>/dev/null");
    if (text) {
        sp_pactl_info_field(text, "Default Source:", default_source, sizeof default_source);
        sp_pactl_info_field(text, "Default Sink:", default_sink, sizeof default_sink);
        free(text);
    }

    sp_pick_reason why = SP_PICK_NONE;
    int pick = sp_pick_mic(mics, num_mics, default_source, default_sink, &why);
    selected_mic_idx = pick + 1;   // no microphone (-1) becomes 0, "Disabled"

    printf("[Audio] %d microphone input(s) found:\n", num_mics);
    for (int i = 0; i < num_mics; i++) {
        printf("        %s %s  (%s)\n", i == pick ? "*" : " ", mics[i].label, mics[i].name);
    }
    if (pick < 0) {
        printf("[Audio] No microphone found: voice recording is unavailable, and recordings "
               "will carry desktop audio only.\n");
    } else if (why != SP_PICK_DEFAULT && default_source[0]) {
        printf("[Audio] Your default input is '%s' - %s - so voice will use '%s'%s.\n",
               default_source,
               sp_ends_with(default_source, ".monitor")
                   ? "that is what your speakers play, not a microphone"
                   : "that is not one of the inputs above",
               mics[pick].label,
               why == SP_PICK_SAME_DEVICE ? " (the microphone on the device you listen on)" : "");
    } else {
        printf("[Audio] Voice will use '%s'%s.\n", mics[pick].label,
               why == SP_PICK_DEFAULT ? " (your default input)" : "");
    }
    if (pick >= 0) {
        printf("[Audio] To change it: open the HUD (Alt+Z), go to 'Microphone' and press Enter.\n");
    }
}

static int is_wsl() {
    FILE* f = fopen("/proc/sys/kernel/osrelease", "r");
    if (!f) return 0;
    char buf[256];
    if (fgets(buf, sizeof(buf), f)) {
        if (strstr(buf, "microsoft") || strstr(buf, "Microsoft")) {
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    return 0;
}

static void get_default_monitor_device(char* buf, size_t max_len) {
    FILE* pipe = popen("pactl get-default-sink 2>/dev/null", "r");
    if (pipe) {
        char sink_name[256];
        if (fgets(sink_name, sizeof(sink_name), pipe)) {
            sink_name[strcspn(sink_name, "\n")] = '\0';
            if (strlen(sink_name) > 0) {
                snprintf(buf, max_len, "%s.monitor", sink_name);
                pclose(pipe);
                return;
            }
        }
        pclose(pipe);
    }
    // Not "default": that is the default SOURCE - a microphone on most
    // machines - and mixing it in as "desktop audio" would record the voice
    // twice. @DEFAULT_MONITOR@ is the audio server's name for what the
    // default output plays.
    strncpy(buf, "@DEFAULT_MONITOR@", max_len);
}

// Can ffmpeg open audio-server sources at all? Asked once, and only when
// ffmpeg is going to be the recorder, because the probe takes a moment. It
// runs in the HUD process now, so `-nostdin`: without it ffmpeg reads the
// terminal and swallows keypresses.
static int ffmpeg_pulse_state = -1;
static int ffmpeg_can_record_audio(void) {
    if (ffmpeg_pulse_state < 0) {
        ffmpeg_pulse_state = system("ffmpeg -nostdin -y -f pulse -i default -t 0.1 -f null - "
                                    ">/dev/null 2>&1") == 0;
    }
    return ffmpeg_pulse_state;
}

static const char* home_dir(void) {
    const char* h = getenv("HOME");
    return (h && h[0]) ? h : "/tmp";
}

// ~/Videos/Y_Captures, created if it does not exist.
static void ensure_captures_dir(void) {
    char path[512];
    snprintf(path, sizeof path, "%s/Videos", home_dir());
    mkdir(path, 0755);
    snprintf(path, sizeof path, "%s/Videos/Y_Captures", home_dir());
    mkdir(path, 0755);
}

// Start a recorder. The command was built in the parent, so the child only has
// to let go of X, tie its lifetime to ours, and exec.
static pid_t spawn_encoder(sp_argv* cmd, const char* log_path) {
    pid_t parent = getpid();
    pid_t pid = fork();
    if (pid > 0) setpgid(pid, pid);   // both sides call it, so there is no window
    if (pid != 0) return pid;         // the parent, or -1

    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    // A recorder must not outlive the HUD. If the HUD dies without cleaning
    // up - a crash, a SIGKILL - the kernel sends the recorder SIGINT, which is
    // the signal every recorder here finalizes its file on. Without this, a
    // voice recording would go on listening to the microphone after the
    // program that started it was gone.
    prctl(PR_SET_PDEATHSIG, SIGINT);
    if (getppid() != parent) _exit(0);   // it died before prctl took effect
    // Its own process group, so Ctrl+C in the terminal reaches the HUD alone
    // and the HUD stops each recorder with exactly ONE SIGINT. A second one -
    // the terminal's, landing while ffmpeg writes the file's index - means
    // "exit immediately" to ffmpeg, and the recording is lost.
    setpgid(0, 0);

    if (dpy) close(ConnectionNumber(dpy));
    int in = open("/dev/null", O_RDONLY);
    if (in >= 0) dup2(in, STDIN_FILENO);
    // ONE open shared by stdout and stderr. Opening the file twice with "w"
    // gave the two streams separate offsets, so they overwrote each other.
    int log = open(log_path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0644);
    if (log >= 0) {
        dup2(log, STDOUT_FILENO);
        dup2(log, STDERR_FILENO);
    }
    execvp(cmd->v[0], cmd->v);
    fprintf(stderr, "[ShadowPlay] could not run %s: %s\n", cmd->v[0], strerror(errno));
    _exit(127);
}

// The screen-recorder settings, in the form the pure builder takes.
static void fill_capture(sp_capture* c, int replay, const char* out) {
    const char* session = getenv("XDG_SESSION_TYPE");
    const char* display = getenv("DISPLAY");
    memset(c, 0, sizeof *c);
    c->replay = replay;
    c->replay_seconds = replay_durations[replay_duration_idx];
    c->quality = quality;
    c->codec = video_codec;
    c->container = file_format;
    c->mic = selected_mic_name();
    c->desktop = default_audio_dev;
    c->out = out;
    c->wayland = session && strcmp(session, "wayland") == 0;
    c->display = display ? display : ":0.0";
    c->screen_w = screen_width;
    c->screen_h = screen_height;
    c->has_gsr = has_gpu_screen_recorder;
    c->has_wf = has_wf_recorder;
    c->has_ffmpeg = has_ffmpeg;
    c->ffmpeg_pulse = (!has_gpu_screen_recorder && has_ffmpeg) ? ffmpeg_can_record_audio() : 0;
}

// Build the screen-recorder command and start it. Returns its pid, or 0 when
// nothing was started - the reason has been printed. The "Audio:" line comes
// from the same builder as the command, so it cannot promise a microphone the
// command leaves out.
static pid_t launch_capture(int replay, const char* out, const char* log_path) {
    sp_capture c;
    sp_argv cmd;
    char note[512];

    fill_capture(&c, replay, out);
    sp_backend b = sp_build_capture(&cmd, &c, note, sizeof note);
    if (b == SP_NONE) {
        if (cmd.overflow) {
            printf("[ShadowPlay Error] The recorder command did not fit (a path is too long).\n");
        } else {
            printf("[ShadowPlay Error] No screen recorder is installed. "
                   "Install gpu-screen-recorder (or ffmpeg).\n");
        }
        fflush(stdout);
        return 0;
    }
    printf("[ShadowPlay] Recorder: %s\n", sp_backend_name(b));
    printf("[ShadowPlay] Audio: %s\n", note);
    fflush(stdout);

    pid_t pid = spawn_encoder(&cmd, log_path);
    if (pid < 0) {
        printf("[ShadowPlay Error] Could not start %s: %s\n", sp_backend_name(b), strerror(errno));
        fflush(stdout);
        return 0;
    }
    return pid;
}

static void start_manual_recording() {
    if (record_pid > 0) return;

    printf("[ShadowPlay] Starting manual recording...\n");
    fflush(stdout);
    if (is_wsl()) {
        printf("[ShadowPlay Warning] You are running inside WSL. Linux screen recorders (like ffmpeg/wf-recorder) can only capture Linux GUI windows inside WSLg, not your main Windows host desktop screen.\n");
    }

    ensure_captures_dir();
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/Videos/Y_Captures/Manual_Capture_%ld.%s",
             home_dir(), (long)time(NULL), file_format == 0 ? "mp4" : "mkv");

    record_pid = launch_capture(0, filepath, RECORD_LOG);
    if (record_pid <= 0) {
        record_pid = 0;
        recording = 0;
        show_toast("Recording failed to start");
        return;
    }
    printf("[ShadowPlay] Manual screen recording started: %s\n", filepath);
    fflush(stdout);
    show_toast("Recording Started");
}

// Ask the encoder to finish, and make sure it actually did.
//
// The previous version polled for 500ms and then zeroed the pid whether or not
// the child had exited. That leaves a zombie, and - much worse - leaves the
// encoder still running and still writing to the capture while the UI reports
// "Saved". A 4K AV1 mux can take seconds to flush, so SIGINT gets a real grace
// period; if it is ignored the child is killed and reaped rather than dropped.
static void reap_encoder(pid_t* pid, const char* what) {
    if (*pid <= 0) return;
    int status;
    kill(*pid, SIGINT);
    for (int i = 0; i < 300; i++) {          // up to 3s
        if (waitpid(*pid, &status, WNOHANG) > 0) { *pid = 0; return; }
        usleep(10000);
    }
    fprintf(stderr, "[ShadowPlay] %s (pid %d) ignored SIGINT after 3s; sending SIGKILL. "
                    "The capture may be truncated.\n", what, (int)*pid);
    kill(*pid, SIGKILL);
    waitpid(*pid, &status, 0);
    *pid = 0;
}

static void stop_manual_recording() {
    if (record_pid > 0) {
        reap_encoder(&record_pid, "recorder");
        printf("[ShadowPlay] Manual screen recording saved to ~/Videos/Y_Captures/.\n");
        show_toast("Recording Saved");
    }
}

static void start_replay_buffer() {
    if (replay_pid > 0) return;

    printf("[ShadowPlay] Starting Instant Replay buffer...\n");
    fflush(stdout);
    if (is_wsl()) {
        printf("[ShadowPlay Warning] You are running inside WSL. Linux screen recorders (like ffmpeg/wf-recorder) can only capture Linux GUI windows inside WSLg, not your main Windows host desktop screen.\n");
    }

    // gpu-screen-recorder keeps the buffer itself and writes each saved clip
    // into a directory; the fallbacks record to a file that save_replay_clip()
    // cuts the end off.
    char out[512];
    if (has_gpu_screen_recorder) {
        ensure_captures_dir();
        snprintf(out, sizeof out, "%s/Videos/Y_Captures", home_dir());
    } else {
        snprintf(out, sizeof out, "/tmp/y_replay_buffer.mp4");
    }

    replay_pid = launch_capture(1, out, REPLAY_LOG);
    if (replay_pid <= 0) {
        replay_pid = 0;
        instant_replay = 0;
        show_toast("Instant Replay failed to start");
        return;
    }
    printf("[ShadowPlay] Instant Replay background buffer activated (%s).\n",
           has_gpu_screen_recorder ? "clips are saved to ~/Videos/Y_Captures/"
                                   : "buffering to /tmp/y_replay_buffer.mp4");
    fflush(stdout);
    show_toast("Instant Replay ON");
}

static void stop_replay_buffer() {
    reap_encoder(&replay_pid, "replay buffer");
    usleep(100000);
    if (!has_gpu_screen_recorder) {
        unlink("/tmp/y_replay_buffer.mp4");
    }
    printf("[ShadowPlay] Instant Replay buffer deactivated.\n");
    show_toast("Instant Replay OFF");
}


static void save_replay_clip() {
    static time_t last_save_time = 0;
    time_t now = time(NULL);
    if (now - last_save_time < 2) {
        printf("[ShadowPlay] Rate limit: please wait before saving another clip.\n");
        return;
    }
    last_save_time = now;

    if (replay_pid == 0) {
        printf("[ShadowPlay Warning] Cannot save replay: buffer is not active.\n");
        show_toast("Buffer Not Active");
        return;
    }
    
    system("mkdir -p ~/Videos/Y_Captures");
    if (has_gpu_screen_recorder) {
        kill(replay_pid, SIGUSR1);
        printf("[ShadowPlay] Replay clip successfully saved to ~/Videos/Y_Captures/ (handled by gpu-screen-recorder)!\n");
        char msg[128];
        snprintf(msg, sizeof(msg), "Saved last %ds clip", replay_durations[replay_duration_idx]);
        show_toast(msg);
    } else {
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "cp /tmp/y_replay_buffer.mp4 /tmp/y_replay_temp.mp4 && ffmpeg -y -err_detect ignore_err -sseof -%d -i /tmp/y_replay_temp.mp4 -c copy ~/Videos/Y_Captures/Instant_Replay_%ld.mp4 > /dev/null 2>&1 && rm -f /tmp/y_replay_temp.mp4", replay_durations[replay_duration_idx], time(NULL));
        
        printf("[ShadowPlay] Extracting last %d seconds of recording...\n", replay_durations[replay_duration_idx]);
        int status = system(cmd);
        if (status == 0) {
            printf("[ShadowPlay] Replay clip successfully saved to ~/Videos/Y_Captures/!\n");
            char msg[128];
            snprintf(msg, sizeof(msg), "Saved last %ds clip", replay_durations[replay_duration_idx]);
            show_toast(msg);
        } else {
            printf("[ShadowPlay Error] Failed to slice replay clip. Make sure it has been running for a few seconds first!\n");
            show_toast("Failed to save clip");
        }
    }
}

// ------------------------------------------------------------------ voice --

// A name for a new voice recording that no file has yet. The recorder picks
// the extension (.m4a or .wav), so both are checked. Two recordings started in
// the same second would otherwise share a name.
static void new_voice_base(char* out, size_t cap) {
    long stamp = (long)time(NULL);
    for (int n = 1; n < 1000; n++) {
        if (n == 1) snprintf(out, cap, "%s/Videos/Y_Captures/Voice_Recording_%ld", home_dir(), stamp);
        else snprintf(out, cap, "%s/Videos/Y_Captures/Voice_Recording_%ld_%d", home_dir(), stamp, n);
        char probe[600];
        snprintf(probe, sizeof probe, "%s.m4a", out);
        if (access(probe, F_OK) == 0) continue;
        snprintf(probe, sizeof probe, "%s.wav", out);
        if (access(probe, F_OK) == 0) continue;
        return;
    }
}

// Record the selected microphone on its own, into a file of its own.
static void start_voice_recording(void) {
    if (voice_pid > 0) return;

    const char* mic = selected_mic_name();
    if (!mic) {
        printf("[ShadowPlay] Voice recording needs a microphone: open the HUD (Alt+Z), "
               "go to 'Microphone' and press Enter to pick one.\n");
        fflush(stdout);
        show_toast("No microphone selected");
        return;
    }

    ensure_captures_dir();
    char base[512];
    new_voice_base(base, sizeof base);
    sp_voice v;
    memset(&v, 0, sizeof v);
    v.mic = mic;
    v.out_base = base;
    v.has_ffmpeg = has_ffmpeg;
    v.has_parecord = has_parecord;

    sp_argv cmd;
    sp_backend b = sp_build_voice(&cmd, &v, voice_path, sizeof voice_path);
    if (b == SP_NONE) {
        printf("[ShadowPlay Error] Voice recording needs ffmpeg or parecord%s.\n",
               cmd.overflow ? " (or the output path is too long)" : "");
        fflush(stdout);
        show_toast("Voice needs ffmpeg");
        return;
    }
    voice_pid = spawn_encoder(&cmd, VOICE_LOG);
    if (voice_pid < 0) {
        voice_pid = 0;
        printf("[ShadowPlay Error] Could not start %s: %s\n", sp_backend_name(b), strerror(errno));
        fflush(stdout);
        show_toast("Voice recording failed");
        return;
    }
    voice_recording = 1;
    voice_started = time(NULL);
    printf("[ShadowPlay] Voice recording started (microphone: %s, recorder: %s): %s\n",
           selected_mic_label(), sp_backend_name(b), voice_path);
    fflush(stdout);
    show_toast("Voice Recording Started");
}

static void stop_voice_recording(void) {
    if (voice_pid <= 0) {
        voice_recording = 0;
        return;
    }
    reap_encoder(&voice_pid, "voice recorder");
    voice_recording = 0;
    long secs = (long)(time(NULL) - voice_started);
    printf("[ShadowPlay] Voice recording saved (%ld:%02ld): %s\n", secs / 60, secs % 60, voice_path);
    fflush(stdout);
    show_toast("Voice Recording Saved");
}

static void toggle_voice_recording(void) {
    if (voice_recording) stop_voice_recording();
    else start_voice_recording();
}

// A recorder that exits on its own has FAILED: every one of them runs until it
// is told to stop. Without this check the HUD went on saying "RECORDING..."
// after gpu-screen-recorder refused a device or the Wayland screen-share
// dialog was cancelled, while nothing was being written. Polled every tick.
static void reap_if_dead(pid_t* pid, int* active, const char* what, const char* log_path,
                         const char* toast) {
    if (*pid <= 0) return;
    int status = 0;
    pid_t r = waitpid(*pid, &status, WNOHANG);
    if (r == 0 || (r < 0 && errno != ECHILD)) return;   // still running

    *pid = 0;
    *active = 0;
    capture_failures++;
    if (r > 0 && WIFEXITED(status)) {
        fprintf(stderr, "[ShadowPlay Error] The %s stopped on its own (exit code %d), so nothing "
                        "is being recorded. Its output is in %s.\n",
                what, WEXITSTATUS(status), log_path);
    } else if (r > 0 && WIFSIGNALED(status)) {
        fprintf(stderr, "[ShadowPlay Error] The %s was killed by signal %d, so nothing is being "
                        "recorded. Its output is in %s.\n",
                what, WTERMSIG(status), log_path);
    } else {
        fprintf(stderr, "[ShadowPlay Error] The %s is gone, so nothing is being recorded. "
                        "Its output is in %s.\n", what, log_path);
    }
    show_toast(toast);
    update_window_layout();
}

void cleanup_shadowplay_gui(void) {
    stop_voice_recording();
    stop_manual_recording();
    if (replay_pid > 0) stop_replay_buffer();   // it announces "deactivated" unconditionally
    if (dpy) {
        XCloseDisplay(dpy);
        dpy = NULL;
    }
}

// Colors
static unsigned long color_bg;
static unsigned long color_card;
static unsigned long color_green;
static unsigned long color_white;
static unsigned long color_grey;
static unsigned long color_red;
static unsigned long color_voice;

// Helper to allocate colors
static unsigned long get_color(const char* hex) {
    XColor col;
    Colormap cmap = DefaultColormap(dpy, screen);
    XParseColor(dpy, cmap, hex, &col);
    XAllocColor(dpy, cmap, &col);
    return col.pixel;
}

static void draw_rounded_rect(Display* d, Drawable dr, GC gc, int x, int y, int w, int h, int r) {
    XDrawArc(d, dr, gc, x, y, r*2, r*2, 90*64, 90*64);
    XDrawArc(d, dr, gc, x+w-r*2, y, r*2, r*2, 0, 90*64);
    XDrawArc(d, dr, gc, x, y+h-r*2, r*2, r*2, 180*64, 90*64);
    XDrawArc(d, dr, gc, x+w-r*2, y+h-r*2, r*2, r*2, 270*64, 90*64);
    XDrawLine(d, dr, gc, x+r, y, x+w-r, y);
    XDrawLine(d, dr, gc, x+r, y+h, x+w-r, y+h);
    XDrawLine(d, dr, gc, x, y+r, x, y+h-r);
    XDrawLine(d, dr, gc, x+w, y+r, x+w, y+h-r);
}

static void fill_rounded_rect(Display* d, Drawable dr, GC gc, int x, int y, int w, int h, int r) {
    XFillArc(d, dr, gc, x, y, r*2, r*2, 90*64, 90*64);
    XFillArc(d, dr, gc, x+w-r*2, y, r*2, r*2, 0, 90*64);
    XFillArc(d, dr, gc, x, y+h-r*2, r*2, r*2, 180*64, 90*64);
    XFillArc(d, dr, gc, x+w-r*2, y+h-r*2, r*2, r*2, 270*64, 90*64);
    XFillRectangle(d, dr, gc, x+r, y, w-r*2, h);
    XFillRectangle(d, dr, gc, x, y+r, r, h-r*2);
    XFillRectangle(d, dr, gc, x+w-r, y+r, r, h-r*2);
}

// A microphone `h` pixels tall with its top-left at (x, y), about h*3/4 wide:
// a capsule, the cradle curving under it, and a stand.
static void draw_mic_icon(GC gc, int x, int y, int h) {
    int cap_w = h * 3 / 8;
    int cap_h = h * 5 / 8;
    int cradle_w = h * 3 / 4;
    int cx = x + cradle_w / 2;
    // The capsule as two full discs and the band between them: quarter arcs
    // (fill_rounded_rect) come out lumpy at the 16px the corner indicator uses.
    XFillArc(dpy, win, gc, cx - cap_w / 2, y, cap_w, cap_w, 0, 360 * 64);
    XFillArc(dpy, win, gc, cx - cap_w / 2, y + cap_h - cap_w, cap_w, cap_w, 0, 360 * 64);
    XFillRectangle(dpy, win, gc, cx - cap_w / 2, y + cap_w / 2, cap_w, cap_h - cap_w);
    XSetLineAttributes(dpy, gc, h >= 16 ? 2 : 1, LineSolid, CapRound, JoinRound);
    XDrawArc(dpy, win, gc, x, y + cap_h / 3, cradle_w, cap_h, 180 * 64, 180 * 64);
    XDrawLine(dpy, win, gc, cx, y + cap_h / 3 + cap_h, cx, y + h - 1);
    XDrawLine(dpy, win, gc, cx - cap_w / 2, y + h - 1, cx + cap_w / 2, y + h - 1);
}

static void draw_ui() {
    if (!dpy || !win) return;
    if (!visible && !toast_active && !indicator_wanted()) return;

    // A toast is for when the HUD is closed. Drawn while it is open, it
    // replaced the whole HUD with a 280x80 box for three seconds - every time
    // a setting change restarted Instant Replay, and every time the Voice
    // card (which keeps the HUD open) was used. The HUD shows the state itself.
    if (toast_active && !visible) {
        // Clear window to dark card background instead of fullscreen bg
        XSetWindowBackground(dpy, win, color_card);
        XClearWindow(dpy, win);
        
        GC gc = XCreateGC(dpy, win, 0, NULL);
        if (hud_font) XSetFont(dpy, gc, hud_font->fid);
        
        // Draw a green border around the toast window
        XSetForeground(dpy, gc, color_green);
        XSetLineAttributes(dpy, gc, 2, LineSolid, CapButt, JoinMiter);
        XDrawRectangle(dpy, win, gc, 0, 0, 278, 78);
        
        // Draw Green NVIDIA Icon or Dot
        fill_rounded_rect(dpy, win, gc, 20, 25, 30, 30, 4);
        
        // Draw Text inside the toast
        XSetForeground(dpy, gc, color_white);
        XDrawString(dpy, win, gc, 65, 35, "Y ShadowPlay", 12);
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, 65, 55, toast_text, strlen(toast_text));
        
        XFreeGC(dpy, gc);
        XFlush(dpy);
        return;
    }

    if (!visible && indicator_wanted()) {
        // Clear window to dark card background
        XSetWindowBackground(dpy, win, color_card);
        XClearWindow(dpy, win);
        
        GC gc = XCreateGC(dpy, win, 0, NULL);
        if (hud_font) XSetFont(dpy, gc, hud_font->fid);
        
        // Draw border
        XSetForeground(dpy, gc, color_grey);
        XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        XDrawRectangle(dpy, win, gc, 0, 0, indicator_width() - 2, 30);

        int draw_x = 12;
        if (instant_replay) {
            // Draw green circular arrow indicator / dot
            XSetForeground(dpy, gc, color_green);
            XFillArc(dpy, win, gc, draw_x, 10, 12, 12, 0, 360 * 64);
            draw_x += 24;
        }
        if (recording) {
            // Draw red recording dot
            XSetForeground(dpy, gc, color_red);
            XFillArc(dpy, win, gc, draw_x, 10, 12, 12, 0, 360 * 64);
            draw_x += 24;
        }
        if (voice_recording) {
            // A microphone rather than a third coloured dot: three dots would
            // leave the user to remember which colour meant what.
            XSetForeground(dpy, gc, color_voice);
            draw_mic_icon(gc, draw_x, 8, 16);
        }
        
        XFreeGC(dpy, gc);
        XFlush(dpy);
        return;
    }

    // Clear window background
    XSetWindowBackground(dpy, win, color_bg);
    XClearWindow(dpy, win);

    GC gc = XCreateGC(dpy, win, 0, NULL);
    
    // Set custom font if successfully loaded
    if (hud_font) {
        XSetFont(dpy, gc, hud_font->fid);
    }
    
    // Title text
    XSetForeground(dpy, gc, color_green);
    XDrawString(dpy, win, gc, 30, 42, "NVIDIA GEFORCE EXPERIENCE", 25);
    XSetForeground(dpy, gc, color_white);
    
    char title_buf[128];
    snprintf(title_buf, sizeof(title_buf), "- SHADOWPLAY OVERLAY (Y) [%s]", hw_gpu_name);
    XDrawString(dpy, win, gc, 235, 42, title_buf, strlen(title_buf));
    
    // Draw the four cards (Replay, Record, Voice, Broadcast), in card_order
    int col_width = (HUD_W - 60 - 3 * CARD_GAP) / 4;
    int col_height = 100;
    int start_y = 70;
    int r = 8; // rounded corner radius

    for (int pos = 0; pos < 4; pos++) {
        int i = card_order[pos];
        int start_x = 30 + pos * (col_width + CARD_GAP);
        
        // Fill card background
        XSetForeground(dpy, gc, color_card);
        fill_rounded_rect(dpy, win, gc, start_x, start_y, col_width, col_height, r);

        // Draw Card Border
        if (selected_idx == i) {
            XSetForeground(dpy, gc, color_green);
            XSetLineAttributes(dpy, gc, 2, LineSolid, CapButt, JoinMiter);
        } else {
            XSetForeground(dpy, gc, color_card);
            XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        }
        draw_rounded_rect(dpy, win, gc, start_x, start_y, col_width, col_height, r);
        
        // Draw Text inside Card
        XSetForeground(dpy, gc, color_white);
        if (i == 0) {
            XDrawString(dpy, win, gc, start_x + 15, start_y + 35, "Instant Replay", 14);
            if (instant_replay) {
                XSetForeground(dpy, gc, color_green);
                XDrawString(dpy, win, gc, start_x + 15, start_y + 70, "Status: ACTIVE", 14);
            } else {
                XSetForeground(dpy, gc, color_grey);
                XDrawString(dpy, win, gc, start_x + 15, start_y + 70, "Status: OFF", 11);
            }
            // Draw circular arrow (replay)
            int rx = start_x + col_width - 32;
            int ry = start_y + 15;
            XSetForeground(dpy, gc, instant_replay ? color_green : color_grey);
            XSetLineAttributes(dpy, gc, 2, LineSolid, CapButt, JoinMiter);
            XDrawArc(dpy, win, gc, rx, ry, 16, 16, 45*64, 270*64);
            XDrawLine(dpy, win, gc, rx + 13, ry + 2, rx + 13, ry - 2);
            XDrawLine(dpy, win, gc, rx + 13, ry + 2, rx + 9, ry + 2);
        } else if (i == 1) {
            XDrawString(dpy, win, gc, start_x + 15, start_y + 35, "Manual Record", 13);
            if (recording) {
                XSetForeground(dpy, gc, color_red);
                XDrawString(dpy, win, gc, start_x + 15, start_y + 70, "RECORDING...", 12);
            } else {
                XSetForeground(dpy, gc, color_grey);
                XDrawString(dpy, win, gc, start_x + 15, start_y + 70, "Status: OFF", 11);
            }
            // Draw recording red dot
            int rx = start_x + col_width - 28;
            int ry = start_y + 15;
            if (recording) {
                XSetForeground(dpy, gc, color_red);
                XFillArc(dpy, win, gc, rx, ry, 14, 14, 0, 360*64);
            } else {
                XSetForeground(dpy, gc, color_grey);
                XDrawArc(dpy, win, gc, rx, ry, 14, 14, 0, 360*64);
                XFillArc(dpy, win, gc, rx + 3, ry + 3, 8, 8, 0, 360*64);
            }
        } else if (i == SEL_VOICE) {
            XDrawString(dpy, win, gc, start_x + 15, start_y + 35, "Voice Record", 12);
            char status[32];
            if (voice_recording) {
                long secs = (long)(time(NULL) - voice_started);
                snprintf(status, sizeof status, "REC %ld:%02ld", secs / 60, secs % 60);
                XSetForeground(dpy, gc, color_voice);
            } else if (!selected_mic_name()) {
                snprintf(status, sizeof status, "No mic selected");
                XSetForeground(dpy, gc, color_grey);
            } else {
                snprintf(status, sizeof status, "Status: OFF");
                XSetForeground(dpy, gc, color_grey);
            }
            XDrawString(dpy, win, gc, start_x + 15, start_y + 70, status, strlen(status));
            XSetForeground(dpy, gc, voice_recording ? color_voice : color_grey);
            draw_mic_icon(gc, start_x + col_width - 30, start_y + 12, 20);
        } else if (i == 2) {
            XDrawString(dpy, win, gc, start_x + 15, start_y + 35, "Live Broadcast", 14);
            if (broadcast) {
                XSetForeground(dpy, gc, color_green);
                XDrawString(dpy, win, gc, start_x + 15, start_y + 70, "Status: LIVE", 12);
            } else {
                XSetForeground(dpy, gc, color_grey);
                XDrawString(dpy, win, gc, start_x + 15, start_y + 70, "Status: OFF", 11);
            }
            // Draw broadcast icon (antenna + waves)
            int rx = start_x + col_width - 30;
            int ry = start_y + 15;
            XSetForeground(dpy, gc, broadcast ? color_green : color_grey);
            XSetLineAttributes(dpy, gc, 2, LineSolid, CapButt, JoinMiter);
            XDrawLine(dpy, win, gc, rx + 8, ry + 14, rx + 8, ry + 6);
            XFillArc(dpy, win, gc, rx + 6, ry + 3, 5, 5, 0, 360*64);
            XDrawArc(dpy, win, gc, rx + 1, ry + 1, 14, 14, 120*64, 120*64);
            XDrawArc(dpy, win, gc, rx - 3, ry + 1, 14, 14, 300*64, 120*64);
        }
    }

    // Draw Settings section (Format, Quality, Codec, Replay Length, Keybind, Mic)
    int settings_y = 200;
    XSetForeground(dpy, gc, color_green);
    XDrawString(dpy, win, gc, 30, settings_y, "SETTINGS", 8);

    // Row 1 starts at settings_y + 25 (y = 225)
    int row1_y = settings_y + 25;

    // Format selection row
    int format_x = 150;
    if (selected_idx == 3) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, 30, row1_y + 15, "> File Format:", 14);
        XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        draw_rounded_rect(dpy, win, gc, format_x - 10, row1_y, 140, 22, 4);
    } else {
        XSetForeground(dpy, gc, color_white);
        XDrawString(dpy, win, gc, 30, row1_y + 15, "  File Format:", 14);
    }

    if (file_format == 0) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, format_x, row1_y + 16, "[ MP4 ]", 7);
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, format_x + 70, row1_y + 16, "  MKV  ", 7);
    } else {
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, format_x, row1_y + 16, "  MP4  ", 7);
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, format_x + 70, row1_y + 16, "[ MKV ]", 7);
    }

    // Quality selection row
    int quality_x = RIGHT_COL + 130;
    if (selected_idx == 4) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, RIGHT_COL, row1_y + 15, "> Video Quality:", 16);
        XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        draw_rounded_rect(dpy, win, gc, quality_x - 10, row1_y, 150, 22, 4);
    } else {
        XSetForeground(dpy, gc, color_white);
        XDrawString(dpy, win, gc, RIGHT_COL, row1_y + 15, "  Video Quality:", 16);
    }

    if (quality == 0) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, quality_x, row1_y + 16, "[ 720p ]", 8);
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, quality_x + 55, row1_y + 16, "  1080p  ", 9);
        XDrawString(dpy, win, gc, quality_x + 110, row1_y + 16, "  4K ", 5);
    } else if (quality == 1) {
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, quality_x, row1_y + 16, "  720p  ", 8);
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, quality_x + 55, row1_y + 16, "[ 1080p ]", 9);
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, quality_x + 110, row1_y + 16, "  4K ", 5);
    } else {
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, quality_x, row1_y + 16, "  720p  ", 8);
        XDrawString(dpy, win, gc, quality_x + 55, row1_y + 16, "  1080p  ", 9);
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, quality_x + 110, row1_y + 16, "[  4K  ]", 8);
    }

    // Row 2 starts at settings_y + 65 (y = 265)
    int row2_y = settings_y + 65;

    // Codec selection row
    int codec_x = 150;
    if (selected_idx == 5) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, 30, row2_y + 15, "> Video Codec:", 14);
        XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        draw_rounded_rect(dpy, win, gc, codec_x - 10, row2_y, 140, 22, 4);
    } else {
        XSetForeground(dpy, gc, color_white);
        XDrawString(dpy, win, gc, 30, row2_y + 15, "  Video Codec:", 14);
    }

    if (video_codec == 0) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, codec_x, row2_y + 16, "[ H264 ]", 8);
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, codec_x + 55, row2_y + 16, "  HEVC ", 7);
        XDrawString(dpy, win, gc, codec_x + 105, row2_y + 16, "  AV1", 5);
    } else if (video_codec == 1) {
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, codec_x, row2_y + 16, "  H264  ", 8);
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, codec_x + 55, row2_y + 16, "[ HEVC ]", 8);
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, codec_x + 105, row2_y + 16, "  AV1", 5);
    } else {
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, codec_x, row2_y + 16, "  H264  ", 8);
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, codec_x + 55, row2_y + 16, "  HEVC ", 7);
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, codec_x + 105, row2_y + 16, "[  AV1  ]", 9);
    }

    // Replay Length selection row
    int length_x = RIGHT_COL + 130;
    if (selected_idx == 6) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, RIGHT_COL, row2_y + 15, "> Replay Length:", 16);
        XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        draw_rounded_rect(dpy, win, gc, length_x - 10, row2_y, 150, 22, 4);
    } else {
        XSetForeground(dpy, gc, color_white);
        XDrawString(dpy, win, gc, RIGHT_COL, row2_y + 15, "  Replay Length:", 16);
    }

    for (int idx = 0; idx < 4; idx++) {
        char label[16];
        snprintf(label, sizeof(label), "%ds", replay_durations[idx]);
        if (replay_duration_idx == idx) {
            XSetForeground(dpy, gc, color_green);
            char opt[32];
            snprintf(opt, sizeof(opt), "[ %s ]", label);
            XDrawString(dpy, win, gc, length_x + idx * 35, row2_y + 16, opt, strlen(opt));
        } else {
            XSetForeground(dpy, gc, color_grey);
            char opt[32];
            snprintf(opt, sizeof(opt), "  %s  ", label);
            XDrawString(dpy, win, gc, length_x + idx * 35, row2_y + 16, opt, strlen(opt));
        }
    }

    // Row 3 starts at settings_y + 105 (y = 305)
    int row3_y = settings_y + 105;

    // Save Keybind selection row
    int bind_x = 150;
    if (selected_idx == 7) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, 30, row3_y + 15, "> Save Keybind:", 15);
        XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        draw_rounded_rect(dpy, win, gc, bind_x - 10, row3_y, 140, 22, 4);
    } else {
        XSetForeground(dpy, gc, color_white);
        XDrawString(dpy, win, gc, 30, row3_y + 15, "  Save Keybind:", 15);
    }

    const char* bind_labels[] = {"Alt+S", "Alt+F10", "Alt+R", "Alt+X"};
    XSetForeground(dpy, gc, color_green);
    char opt_bind[32];
    snprintf(opt_bind, sizeof(opt_bind), "[ %s ]", bind_labels[replay_keybind_idx]);
    XDrawString(dpy, win, gc, bind_x, row3_y + 16, opt_bind, strlen(opt_bind));

    // Row 4 starts at settings_y + 145 (y = 345)
    int row4_y = settings_y + 145;

    // Microphone selection row
    int mic_x = 150;
    if (selected_idx == 8) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, 30, row4_y + 15, "> Microphone:", 13);
        XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        draw_rounded_rect(dpy, win, gc, mic_x - 10, row4_y, HUD_W - mic_x - 20, 22, 4);
    } else {
        XSetForeground(dpy, gc, color_white);
        XDrawString(dpy, win, gc, 30, row4_y + 15, "  Microphone:", 13);
    }

    // The microphone feeds Voice Record and is mixed into recordings and
    // replays, so the row says so - "Default Input" said nothing about which
    // device that was, and on some machines it was not a microphone at all.
    XSetForeground(dpy, gc, selected_mic_name() ? color_green : color_grey);
    char mic_opt[SP_LABEL_LEN + 64];
    snprintf(mic_opt, sizeof(mic_opt), "[ %s ]", selected_mic_label());
    XDrawString(dpy, win, gc, mic_x, row4_y + 16, mic_opt, strlen(mic_opt));
    if (selected_mic_name() && num_mics > 1) {
        char hint[48];
        snprintf(hint, sizeof hint, "%d of %d", selected_mic_idx, num_mics);
        XSetForeground(dpy, gc, color_grey);
        int hint_w = hud_font ? XTextWidth(hud_font, hint, strlen(hint)) : 60;
        XDrawString(dpy, win, gc, HUD_W - 40 - hint_w, row4_y + 16, hint, strlen(hint));
    }

    // Row 5 starts at settings_y + 185 (y = 385)
    int row5_y = settings_y + 185;

    // On-screen recording indicator toggle
    int ind_x = 150;
    if (selected_idx == 9) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, 30, row5_y + 15, "> Rec Indicator:", 16);
        XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        draw_rounded_rect(dpy, win, gc, ind_x - 10, row5_y, 180, 22, 4);
    } else {
        XSetForeground(dpy, gc, color_white);
        XDrawString(dpy, win, gc, 30, row5_y + 15, "  Rec Indicator:", 16);
    }

    if (show_indicator) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, ind_x, row5_y + 16, "[ Shown ]", 9);
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, ind_x + 80, row5_y + 16, "  Hidden  ", 10);
    } else {
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, ind_x, row5_y + 16, "  Shown  ", 9);
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, ind_x + 80, row5_y + 16, "[ Hidden ]", 10);
    }

    // Help Footer
    XSetForeground(dpy, gc, color_card);
    XSetLineAttributes(dpy, gc, 2, LineSolid, CapButt, JoinMiter);
    XDrawLine(dpy, win, gc, 30, HUD_H - 85, HUD_W - 30, HUD_H - 85);

    XSetForeground(dpy, gc, color_grey);
    char footer_msg[160];
    if (instant_replay) {
        snprintf(footer_msg, sizeof(footer_msg), "Alt+Z: Hide HUD | %s: Save Last %ds Replay | Alt+V: Voice | Enter: Toggle", bind_labels[replay_keybind_idx], replay_durations[replay_duration_idx]);
    } else {
        snprintf(footer_msg, sizeof(footer_msg), "Alt+Z: Hide HUD | Alt+V: Voice Record | Arrows: Navigate | Enter: Toggle | Esc: Close");
    }
    XDrawString(dpy, win, gc, 30, HUD_H - 55, footer_msg, strlen(footer_msg));

    XFreeGC(dpy, gc);
    XFlush(dpy);
}

// Ctrl+C and SIGTERM. The handler only records that a stop was asked for;
// update_shadowplay_gui() does the stopping, outside signal context. Stopping
// a recorder is Xlib, stdio and waitpid, and none of that may run inside a
// signal handler - which is where all of it used to run. A second press while
// the stop is under way changes nothing; a third exits at once.
//
// (Which process receives the terminal's Ctrl+C is spawn_encoder's business:
// the recorders run in their own process group, so the HUD is the only thing
// that ever signals them.)
static volatile sig_atomic_t stop_requested = 0;

static void handle_sigint(int sig) {
    (void)sig;
    if (stop_requested >= 2) {
        // The third press means "now". The kernel still sends each recorder
        // SIGINT (PR_SET_PDEATHSIG) - a second one for a recorder the shutdown
        // had already stopped, so a file being finalized at that instant can
        // be lost. That is the price of pressing Ctrl+C three times.
        static const char msg[] = "\n[ShadowPlay] Forced exit.\n";
        ssize_t w = write(STDERR_FILENO, msg, sizeof msg - 1);
        (void)w;
        _exit(130);
    }
    stop_requested++;
}

// Global initialization
int32_t init_shadowplay_gui(void) {
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    dpy = XOpenDisplay(NULL);
    if (!dpy) {
        fprintf(stderr, "[X11] Failed to open X display\n");
        return -1;
    }

    screen = DefaultScreen(dpy);
    root = DefaultRootWindow(dpy);
    
    screen_width = DisplayWidth(dpy, screen);
    screen_height = DisplayHeight(dpy, screen);
    printf("[X11] Initialized Display. Screen resolution: %dx%d\n", screen_width, screen_height);
    
    get_default_monitor_device(default_audio_dev, sizeof(default_audio_dev));
    printf("[Pulse] Default audio recording monitor device: %s\n", default_audio_dev);
    load_hw_profile();
    printf("[Y Hardware Sentient] Detected GPU: %s | VRAM: %d MB | AVX: %s | AVX512: %s\n", 
           hw_gpu_name, hw_vram_mb, hw_avx ? "Yes" : "No", hw_avx512 ? "Yes" : "No");
    if (hw_vram_mb >= 12000) {
        quality = 2; // 4K for RTX 4070 Ti and higher
        printf("[Y Hardware Sentient] High-end GPU detected. Defaulting recording quality to 4K UHD.\n");
    } else if (hw_vram_mb >= 8000) {
        quality = 1; // 1080p for 8GB VRAM
        printf("[Y Hardware Sentient] Mid-range GPU detected. Defaulting recording quality to 1080p Full HD.\n");
    } else {
        quality = 0; // 720p for low VRAM
        printf("[Y Hardware Sentient] Budget GPU/VRAM detected. Defaulting recording quality to 720p HD.\n");
    }

    if (strstr(hw_gpu_name, "RTX 40") || strstr(hw_gpu_name, "RTX 50") || strstr(hw_gpu_name, "RX 7") || strstr(hw_gpu_name, "Arc")) {
        video_codec = 2; // AV1
        printf("[Y Hardware Sentient] AV1 encoding GPU detected. Defaulting video codec to AV1.\n");
    } else {
        video_codec = 1; // HEVC
        printf("[Y Hardware Sentient] Defaulting video codec to HEVC for hardware-accelerated compression.\n");
    }
    detect_audio_devices();
    fflush(stdout);

    // Setup colors
    color_bg = get_color("#18181b");
    color_card = get_color("#27272a");
    color_green = get_color("#76b900");
    color_white = get_color("#f4f4f5");
    color_grey = get_color("#71717a");
    color_red = get_color("#ef4444");
    color_voice = get_color("#f59e0b");

    // Set custom error handlers
    XSetErrorHandler(x11_error_handler);
    XSetIOErrorHandler(x11_io_error_handler);

    // Check system dependencies
    has_gpu_screen_recorder = have_command("gpu-screen-recorder");
    has_wf_recorder = have_command("wf-recorder");
    has_ffmpeg = have_command("ffmpeg");
    has_parecord = have_command("parecord");

    if (!has_ffmpeg && !has_parecord) {
        printf("[ShadowPlay WARNING] Voice Record needs 'ffmpeg' (or 'parecord'); neither was found. Please run: sudo pacman -S ffmpeg\n");
    }

    if (has_gpu_screen_recorder) {
        printf("[ShadowPlay] Found 'gpu-screen-recorder'. Recording tasks will use hardware-accelerated GPU capture.\n");
    } else {
        printf("[ShadowPlay NOTE] 'gpu-screen-recorder' was not found. For near-zero overhead recording on your RTX 4070 Ti, please run: sudo pacman -S gpu-screen-recorder\n");
        if (!has_ffmpeg) {
            printf("[ShadowPlay WARNING] 'ffmpeg' was not found in your PATH! Recording features will not work without a fallback. Please run: sudo pacman -S ffmpeg\n");
        }
        char* session = getenv("XDG_SESSION_TYPE");
        if (session && strcmp(session, "wayland") == 0) {
            if (!has_wf_recorder) {
                printf("[ShadowPlay WARNING] You are running a Wayland session, but 'wf-recorder' was not found! Please run 'sudo pacman -S wf-recorder' for native Wayland capture fallback.\n");
            }
        }
    }

    // Grab Alt+z key combinations globally
    KeyCode z_code = XKeysymToKeycode(dpy, XK_z);
    KeyCode f12_code = XKeysymToKeycode(dpy, XK_F12);

    // Grab Alt+z with various lock modifier combinations (NumLock, CapsLock)
    XGrabKey(dpy, z_code, Mod1Mask, root, True, GrabModeAsync, GrabModeAsync);
    XGrabKey(dpy, z_code, Mod1Mask | ShiftMask, root, True, GrabModeAsync, GrabModeAsync);
    XGrabKey(dpy, z_code, Mod1Mask | Mod2Mask, root, True, GrabModeAsync, GrabModeAsync);
    XGrabKey(dpy, z_code, Mod1Mask | LockMask, root, True, GrabModeAsync, GrabModeAsync);
    XGrabKey(dpy, z_code, Mod1Mask | Mod2Mask | LockMask, root, True, GrabModeAsync, GrabModeAsync);

    // Grab F12 as a robust single-key backup
    XGrabKey(dpy, f12_code, 0, root, True, GrabModeAsync, GrabModeAsync);
    XGrabKey(dpy, f12_code, Mod1Mask, root, True, GrabModeAsync, GrabModeAsync);

    update_grabbed_keys();

    // Alt+V starts and stops a voice recording from anywhere.
    grab_hotkey(XK_v);

    // Create borderless window (CWOverrideRedirect)
    XSetWindowAttributes attrs;
    attrs.override_redirect = True;
    attrs.background_pixel = color_bg;
    attrs.event_mask = ExposureMask | KeyPressMask;

    int screen_w = DisplayWidth(dpy, screen);
    int screen_h = DisplayHeight(dpy, screen);
    int win_w = HUD_W;
    int win_h = HUD_H;

    win = XCreateWindow(dpy, root,
                       (screen_w - win_w) / 2,
                       (screen_h - win_h) / 2,
                       win_w, win_h, 0,
                       CopyFromParent, InputOutput, CopyFromParent,
                       CWOverrideRedirect | CWBackPixel | CWEventMask, &attrs);

    // Load standard clean font (e.g. Helvetica bold/medium or fixed fallback)
    hud_font = XLoadQueryFont(dpy, "-*-helvetica-bold-r-normal--14-*-*-*-*-*-*-*");
    if (!hud_font) {
        hud_font = XLoadQueryFont(dpy, "-*-helvetica-medium-r-normal--14-*-*-*-*-*-*-*");
    }
    if (!hud_font) {
        hud_font = XLoadQueryFont(dpy, "fixed");
    }

    XStoreName(dpy, win, "Y ShadowPlay HUD");
    
    // Map window and grab keyboard focus immediately on startup
    XMapWindow(dpy, win);
    XRaiseWindow(dpy, win);
    XGrabKeyboard(dpy, win, True, GrabModeAsync, GrabModeAsync, CurrentTime);

    printf("[X11] ShadowPlay overlay initialized and visible on startup.\n");
    return 0;
}

// State accessors for Y code. Exported, not `static inline` - see the note at
// the top of this file; a static definition emits no symbol and the Y program
// that declares these cannot link.
int32_t is_overlay_visible(void)       { return visible; }
int32_t get_instant_replay_state(void) { return instant_replay; }
int32_t get_recording_state(void)      { return recording; }
int32_t get_broadcast_state(void)      { return broadcast; }
int32_t get_file_format_state(void)    { return file_format; }
int32_t get_quality_state(void)        { return quality; }
int32_t get_codec_state(void)          { return video_codec; }
int32_t get_replay_duration(void)      { return replay_durations[replay_duration_idx]; }
int32_t get_replay_duration_idx(void)  { return replay_duration_idx; }
int32_t get_microphone_index(void)     { return selected_mic_idx; }
int32_t get_indicator_state(void)      { return show_indicator; }
int32_t get_voice_recording_state(void) { return voice_recording; }
int32_t get_capture_failure_count(void) { return capture_failures; }
void get_microphone_name(char* out_buf) { if (out_buf) strcpy(out_buf, selected_mic_label()); }

// Print the selected microphone's name, without a newline. The Y program logs
// it, and Y has no way to receive a C string, so the C side does the printing.
int32_t print_microphone_label(void) {
    printf("%s", selected_mic_label());
    fflush(stdout);
    return 0;
}

// Check X11 events and update the display
int32_t update_shadowplay_gui(void) {
    // Before the display check: stopping the recorders does not need X.
    if (stop_requested) {
        printf("\n[ShadowPlay] Interrupted! Performing clean shutdown...\n");
        fflush(stdout);
        cleanup_shadowplay_gui();
        fflush(NULL);
        exit(0);
    }
    if (!dpy) return -1;

    reap_if_dead(&record_pid, &recording, "screen recorder", RECORD_LOG, "Recording failed");
    reap_if_dead(&replay_pid, &instant_replay, "instant-replay buffer", REPLAY_LOG, "Instant Replay failed");
    reap_if_dead(&voice_pid, &voice_recording, "voice recorder", VOICE_LOG, "Voice recording failed");

    if (toast_active && (time(NULL) - toast_start_time >= 3)) {
        toast_active = 0;
        update_window_layout();
    }

    XEvent ev;
    while (XPending(dpy)) {
        XNextEvent(dpy, &ev);
        
        if (ev.type == KeyPress) {
            KeySym keysym = XLookupKeysym(&ev.xkey, 0);
            
            // Per-keypress diagnostics, opt-in. This printed on every single
            // grabbed keypress in the shipped build.
            if (getenv("Y_SHADOWPLAY_DEBUG")) {
                printf("[X11 debug] KeyPress detected: keycode=%d, keysym=%lu (%s), state=%u\n",
                       ev.xkey.keycode, (unsigned long)keysym, XKeysymToString(keysym), ev.xkey.state);
            }
            
            int is_toggle = 0;
            // Alt+Z or Alt+z
            if ((keysym == XK_z || keysym == XK_Z) && (ev.xkey.state & Mod1Mask)) {
                is_toggle = 1;
            }
            // F12 or Alt+F12
            else if (keysym == XK_F12) {
                is_toggle = 1;
            }
            // Check if key matches the selected save keybind globally
            KeySym save_keysym_lower = XK_s;
            KeySym save_keysym_upper = XK_S;
            if (replay_keybind_idx == 1) {
                save_keysym_lower = XK_F10;
                save_keysym_upper = XK_F10;
            } else if (replay_keybind_idx == 2) {
                save_keysym_lower = XK_r;
                save_keysym_upper = XK_R;
            } else if (replay_keybind_idx == 3) {
                save_keysym_lower = XK_x;
                save_keysym_upper = XK_X;
            }

            KeySym save_keysym = save_keysym_lower; // Keep in scope for inner check

            // Alt+V: start or stop a voice recording, HUD open or not.
            if ((keysym == XK_v || keysym == XK_V) && (ev.xkey.state & Mod1Mask)) {
                toggle_voice_recording();
                if (visible) draw_ui();
                continue;
            }

            if ((keysym == save_keysym_lower || keysym == save_keysym_upper) && (ev.xkey.state & Mod1Mask)) {
                if (instant_replay) {
                    save_replay_clip();
                }
                continue;
            }

            if (is_toggle) {
                visible = !visible;
                update_window_layout();
                if (visible) {
                    XGrabKeyboard(dpy, win, True, GrabModeAsync, GrabModeAsync, CurrentTime);
                    printf("[X11] ShadowPlay overlay mapped/visible.\n");
                } else {
                    XUngrabKeyboard(dpy, CurrentTime);
                    printf("[X11] ShadowPlay overlay hidden.\n");
                }
                draw_ui();
                continue;
            }

            // Keyboard navigation when visible
            if (visible) {
                if (keysym == XK_Escape) {
                    visible = 0;
                    XUngrabKeyboard(dpy, CurrentTime);
                    update_window_layout();
                    draw_ui();
                } else if (keysym == XK_s || keysym == XK_S || keysym == save_keysym) {
                    if (instant_replay) {
                        save_replay_clip();
                    }
                } else if (keysym == XK_Left) {
                    if (card_position(selected_idx) >= 0) {
                        selected_idx = card_order[(card_position(selected_idx) + 3) % 4];
                    } else if (selected_idx == 3) {
                        selected_idx = 4;
                    } else if (selected_idx == 4) {
                        selected_idx = 3;
                    } else if (selected_idx == 5) {
                        selected_idx = 6;
                    } else if (selected_idx == 6) {
                        selected_idx = 5;
                    }
                    draw_ui();
                } else if (keysym == XK_Right) {
                    if (card_position(selected_idx) >= 0) {
                        selected_idx = card_order[(card_position(selected_idx) + 1) % 4];
                    } else if (selected_idx == 3) {
                        selected_idx = 4;
                    } else if (selected_idx == 4) {
                        selected_idx = 3;
                    } else if (selected_idx == 5) {
                        selected_idx = 6;
                    } else if (selected_idx == 6) {
                        selected_idx = 5;
                    }
                    draw_ui();
                } else if (keysym == XK_Up) {
                    if (card_position(selected_idx) >= 0) selected_idx = 9;
                    else if (selected_idx == 3) selected_idx = 0;
                    else if (selected_idx == 4) selected_idx = SEL_VOICE;   // the card above Video Quality
                    else if (selected_idx == 5) selected_idx = 3;
                    else if (selected_idx == 6) selected_idx = 4;
                    else if (selected_idx == 7) selected_idx = 5;
                    else if (selected_idx == 8) selected_idx = 7;
                    else if (selected_idx == 9) selected_idx = 8;
                    draw_ui();
                } else if (keysym == XK_Down) {
                    // The two left cards sit over File Format, the two right ones over Video Quality.
                    if (card_position(selected_idx) >= 0) selected_idx = card_position(selected_idx) < 2 ? 3 : 4;
                    else if (selected_idx == 3) selected_idx = 5;
                    else if (selected_idx == 4) selected_idx = 6;
                    else if (selected_idx == 5) selected_idx = 7;
                    else if (selected_idx == 6) selected_idx = 7;
                    else if (selected_idx == 7) selected_idx = 8;
                    else if (selected_idx == 8) selected_idx = 9;
                    else if (selected_idx == 9) selected_idx = 0;
                    draw_ui();
                } else if (keysym == XK_Return) {
                    // Activate focused item
                    if (selected_idx == 0) {
                        instant_replay = !instant_replay;
                        if (instant_replay) {
                            visible = 0;
                            XUngrabKeyboard(dpy, CurrentTime);
                            update_window_layout();
                            draw_ui();
                            start_replay_buffer();
                        } else {
                            stop_replay_buffer();
                        }
                    } else if (selected_idx == 1) {
                        recording = !recording;
                        if (recording) {
                            visible = 0;
                            XUngrabKeyboard(dpy, CurrentTime);
                            update_window_layout();
                            draw_ui();
                            start_manual_recording();
                        } else {
                            stop_manual_recording();
                        }
                    } else if (selected_idx == SEL_VOICE) {
                        // The HUD stays open, unlike the two video cards:
                        // nothing on screen is being captured, and the card
                        // shows the running time.
                        toggle_voice_recording();
                    } else if (selected_idx == 2) {
                        broadcast = !broadcast;
                    } else if (selected_idx == 3) {
                        file_format = !file_format;
                        if (instant_replay) {
                            stop_replay_buffer();
                            start_replay_buffer();
                        }
                    } else if (selected_idx == 4) {
                        quality = (quality + 1) % 3;
                        if (instant_replay) {
                            stop_replay_buffer();
                            start_replay_buffer();
                        }
                    } else if (selected_idx == 5) {
                        video_codec = (video_codec + 1) % 3;
                        if (instant_replay) {
                            stop_replay_buffer();
                            start_replay_buffer();
                        }
                    } else if (selected_idx == 6) {
                        replay_duration_idx = (replay_duration_idx + 1) % 4;
                        if (instant_replay) {
                            stop_replay_buffer();
                            start_replay_buffer();
                        }
                    } else if (selected_idx == 7) {
                        replay_keybind_idx = (replay_keybind_idx + 1) % 4;
                        update_grabbed_keys();
                    } else if (selected_idx == 8) {
                        selected_mic_idx = (selected_mic_idx + 1) % (num_mics + 1);
                        printf("[ShadowPlay] Microphone: %s%s\n", selected_mic_label(),
                               voice_recording ? " (from the next voice recording on)" : "");
                        fflush(stdout);
                        if (instant_replay) {
                            stop_replay_buffer();
                            start_replay_buffer();
                        }
                    } else if (selected_idx == 9) {
                        // Hiding the indicator is purely cosmetic - it maps or
                        // unmaps the corner window and does not touch the
                        // encoder, so a capture in progress keeps running.
                        show_indicator = !show_indicator;
                        printf("[ShadowPlay] On-screen recording indicator %s.\n",
                               show_indicator ? "shown" : "hidden");
                        fflush(stdout);
                        update_window_layout();
                    }
                    draw_ui();
                }
            }
        }
        else if (ev.type == Expose) {
            draw_ui();
        }
    }
    
    // Continual drawing to handle potential frame refreshes
    if (visible || toast_active || indicator_wanted()) {
        draw_ui();
    }
    
    return 0;
}

#endif // Y_NO_X11

#endif // SHADOWPLAY_GUI_H
