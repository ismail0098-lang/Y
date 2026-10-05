// c_src/shadowplay_capture.h
// Microphone discovery and recorder command lines for Y ShadowPlay.
//
// Everything in this file is PURE: it parses text and fills argument vectors.
// No X11, no fork, no exec, no popen. `shadowplay_gui.h` decides WHEN to
// record; this file decides WHAT the recorder is told. Keeping the two apart
// is what lets `tests/shadowplay_capture.rs` compile this file into a small
// harness, check the exact command lines on any machine, and run the ffmpeg
// ones against synthetic sources to confirm the microphone reaches the file.
//
// Why it exists. Voice capture was broken three ways at once, and each one hid
// the others:
//
//  * The device list came from parsing the error older gpu-screen-recorder
//    releases printed for `-a check_devices` ("expected one of:"). 6.x stops
//    at "missing argument '-w'" before it looks at `-a`, so nothing was ever
//    parsed and the only choices left were "Disabled" and "Default Input".
//  * "Default Input" is the audio server's default SOURCE. On a machine whose
//    default source is the monitor of the headphones, it records the desktop
//    a second time and never the voice.
//  * The microphone setting reached one recorder of three. gpu-screen-recorder
//    got `-a default_output|<mic>`; the wf-recorder and ffmpeg fallbacks
//    ignored it while the log said "Desktop + Mic [Merged]".
//
// Included by `shadowplay_gui.h` only, which is itself included by exactly one
// translation unit, so everything here is `static`.

#ifndef SHADOWPLAY_CAPTURE_H
#define SHADOWPLAY_CAPTURE_H

#include <stdio.h>
#include <string.h>

#define SP_MAX_MICS  16
#define SP_NAME_LEN  192
#define SP_LABEL_LEN 128

typedef struct {
    char name[SP_NAME_LEN];    // the audio-server source name every recorder is given
    char label[SP_LABEL_LEN];  // its description, for the HUD and the log
} sp_mic;

// ------------------------------------------------------------------ text --

static void sp_copy(char* dst, size_t cap, const char* src, size_t len) {
    if (cap == 0) return;
    if (len >= cap) len = cap - 1;
    memcpy(dst, src, len);
    dst[len] = '\0';
}

static int sp_ends_with(const char* s, const char* suffix) {
    size_t a = strlen(s), b = strlen(suffix);
    return a >= b && strcmp(s + a - b, suffix) == 0;
}

// ASCII case-insensitive substring test.
static int sp_contains_ci(const char* hay, const char* needle) {
    size_t n = strlen(needle);
    for (; *hay; hay++) {
        size_t i = 0;
        for (; i < n && hay[i]; i++) {
            char a = hay[i], b = needle[i];
            if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
            if (a != b) break;
        }
        if (i == n) return 1;
    }
    return n == 0;
}

// Step through `text` one line at a time. `*line`/`*len` are the line with its
// leading blanks and trailing CR removed; `*indented` says whether it had any
// leading blanks. Returns 0 at the end of the text.
static int sp_next_line(const char** cursor, const char** line, size_t* len, int* indented) {
    const char* p = *cursor;
    if (!p || !*p) return 0;
    const char* eol = strchr(p, '\n');
    size_t l = eol ? (size_t)(eol - p) : strlen(p);
    *cursor = eol ? eol + 1 : p + l;
    *indented = (l > 0 && (*p == ' ' || *p == '\t'));
    while (l > 0 && (*p == ' ' || *p == '\t')) { p++; l--; }
    while (l > 0 && p[l - 1] == '\r') l--;
    *line = p;
    *len = l;
    return 1;
}

// If `line` reads `<key> <value>`, point `*val` at the value.
static int sp_field(const char* line, size_t len, const char* key,
                    const char** val, size_t* vlen) {
    size_t k = strlen(key);
    if (len < k || strncmp(line, key, k) != 0) return 0;
    const char* v = line + k;
    size_t n = len - k;
    while (n > 0 && (*v == ' ' || *v == '\t')) { v++; n--; }
    *val = v;
    *vlen = n;
    return 1;
}

// ------------------------------------------------------ device discovery --

// Parse `LC_ALL=C pactl list sources` into the sources a voice can come from.
//
// A source is a microphone candidate unless it MONITORS a sink - that is
// desktop audio, not a voice. pactl says which in so many words ("Monitor of
// Sink: n/a" on a real input); the `.monitor` name suffix is consulted only
// when that field is missing, because a virtual sink may be named anything.
//
// The caller must run pactl under LC_ALL=C. The field names are translated,
// and in another language nothing below would match: the list would come back
// empty and the microphone would be silently unavailable.
static int sp_parse_pactl_sources(const char* text, sp_mic* out, int max) {
    int n = 0;
    int in_source = 0;
    int monitor = -1;          // -1 = the field was absent, 0 = a real input, 1 = a monitor
    int name_too_long = 0;     // a truncated device name names a different device
    char name[SP_NAME_LEN] = "";
    char label[SP_LABEL_LEN] = "";
    const char* cur = text;
    const char* line;
    size_t len;
    int indented;

    for (;;) {
        int more = sp_next_line(&cur, &line, &len, &indented);
        int header = more && !indented && len >= 8 && strncmp(line, "Source #", 8) == 0;

        if ((header || !more) && in_source) {
            int is_monitor = monitor == 1 || (monitor == -1 && sp_ends_with(name, ".monitor"));
            if (name[0] && !name_too_long && !is_monitor && n < max) {
                const char* shown = label[0] ? label : name;
                sp_copy(out[n].name, sizeof out[n].name, name, strlen(name));
                sp_copy(out[n].label, sizeof out[n].label, shown, strlen(shown));
                n++;
            }
        }
        if (!more) break;
        if (header) {
            in_source = 1;
            monitor = -1;
            name_too_long = 0;
            name[0] = label[0] = '\0';
            continue;
        }
        if (!in_source || !indented) continue;

        const char* v;
        size_t vl;
        if (sp_field(line, len, "Name:", &v, &vl)) {
            name_too_long = vl >= sizeof name;
            sp_copy(name, sizeof name, v, vl);
        } else if (sp_field(line, len, "Description:", &v, &vl)) {
            sp_copy(label, sizeof label, v, vl);
        } else if (sp_field(line, len, "Monitor of Sink:", &v, &vl)) {
            monitor = !(vl == 3 && strncmp(v, "n/a", 3) == 0);
        }
    }
    return n;
}

// Parse `gpu-screen-recorder --list-audio-devices` (`name|label` per line).
// Only a fallback for when pactl is unavailable. The two abstract entries are
// skipped for the same reason the pactl list has none: `default_input` is
// whatever the default source is, and that can be a monitor.
static int sp_parse_gsr_devices(const char* text, sp_mic* out, int max) {
    int n = 0;
    const char* cur = text;
    const char* line;
    size_t len;
    int indented;

    while (n < max && sp_next_line(&cur, &line, &len, &indented)) {
        const char* bar = memchr(line, '|', len);
        size_t name_len = bar ? (size_t)(bar - line) : len;
        if (name_len == 0 || name_len >= SP_NAME_LEN) continue;

        char name[SP_NAME_LEN];
        sp_copy(name, sizeof name, line, name_len);
        if (strcmp(name, "default_output") == 0 || strcmp(name, "default_input") == 0 ||
            sp_ends_with(name, ".monitor")) {
            continue;
        }
        sp_copy(out[n].name, sizeof out[n].name, name, name_len);
        if (bar && len > name_len + 1) {
            sp_copy(out[n].label, sizeof out[n].label, bar + 1, len - name_len - 1);
        } else {
            sp_copy(out[n].label, sizeof out[n].label, name, name_len);
        }
        n++;
    }
    return n;
}

// The value of one `Key: value` line of `LC_ALL=C pactl info`, for instance
// "Default Source:". Empty when the key is absent.
static void sp_pactl_info_field(const char* text, const char* key, char* out, size_t cap) {
    const char* cur = text;
    const char* line;
    size_t len;
    int indented;

    if (cap) out[0] = '\0';
    while (sp_next_line(&cur, &line, &len, &indented)) {
        const char* v;
        size_t vl;
        if (sp_field(line, len, key, &v, &vl)) {
            if (vl < cap) sp_copy(out, cap, v, vl);   // a truncated name is a wrong name
            return;
        }
    }
}

// Reduce an audio-server node name to the hardware it names:
// `alsa_output.usb-SteelSeries_Arctis_Nova_7P-00.analog-stereo` and
// `alsa_input.usb-SteelSeries_Arctis_Nova_7P-00.mono-fallback` both give
// `usb-SteelSeries_Arctis_Nova_7P-00`; a Bluetooth headset's output and input
// both give its address. Returns 0 - no key - for a name without the
// `<kind>.<device>.<profile>` shape, so two unrelated virtual devices can
// never match each other on an empty key.
static int sp_device_key(const char* name, char* key, size_t cap) {
    const char* first = strchr(name, '.');
    const char* last = strrchr(name, '.');
    if (!first || last <= first + 1) return 0;
    size_t len = (size_t)(last - first - 1);
    if (len >= cap) return 0;
    sp_copy(key, cap, first + 1, len);
    return 1;
}

static int sp_named_like_mic(const sp_mic* m) {
    return sp_contains_ci(m->name, "mic") || sp_contains_ci(m->label, "mic");
}

typedef enum {
    SP_PICK_NONE,         // there is no input source at all
    SP_PICK_DEFAULT,      // the system default source, which is a real input
    SP_PICK_SAME_DEVICE,  // an input on the same hardware as the default output
    SP_PICK_NAMED_MIC,    // the first input whose name says "mic"
    SP_PICK_FIRST         // the first input
} sp_pick_reason;

// Which microphone to select at startup. Returns an index into `mics`, or -1.
//
//  1. The default source, if it is a real input: that is the user's choice.
//  2. Otherwise the default source is a monitor (or unknown) - desktop audio,
//     not a voice - so prefer an input on the same hardware as the default
//     OUTPUT, which is a headset's own microphone. When that hardware has
//     several inputs (line-in and mic), the one named like a microphone.
//  3. Then any input named like a microphone, then the first input.
static int sp_pick_mic(const sp_mic* mics, int n, const char* default_source,
                       const char* default_sink, sp_pick_reason* why) {
    sp_pick_reason reason = SP_PICK_NONE;
    int pick = -1;
    char sink_key[SP_NAME_LEN];
    char key[SP_NAME_LEN];

    if (n > 0 && default_source && default_source[0]) {
        for (int i = 0; i < n && pick < 0; i++) {
            if (strcmp(mics[i].name, default_source) == 0) { pick = i; reason = SP_PICK_DEFAULT; }
        }
    }
    if (pick < 0 && n > 0 && default_sink && sp_device_key(default_sink, sink_key, sizeof sink_key)) {
        int first_same = -1;
        for (int i = 0; i < n && pick < 0; i++) {
            if (!sp_device_key(mics[i].name, key, sizeof key) || strcmp(key, sink_key) != 0) continue;
            if (sp_named_like_mic(&mics[i])) pick = i;
            else if (first_same < 0) first_same = i;
        }
        if (pick < 0) pick = first_same;
        if (pick >= 0) reason = SP_PICK_SAME_DEVICE;
    }
    for (int i = 0; i < n && pick < 0; i++) {
        if (sp_named_like_mic(&mics[i])) { pick = i; reason = SP_PICK_NAMED_MIC; }
    }
    if (pick < 0 && n > 0) { pick = 0; reason = SP_PICK_FIRST; }

    if (why) *why = reason;
    return pick;
}

// ------------------------------------------------------ recorder commands --

// A fixed-capacity argument vector for execvp. Every string is copied into
// `arena`, so callers may pass stack buffers. It points into itself: build it
// in place and never copy it by value (fork() is fine - it copies the address
// space, addresses included).
typedef struct {
    char* v[64];
    int n;
    int overflow;   // something did not fit: the command is incomplete and must not run
    size_t used;
    char arena[4096];
} sp_argv;

static void sp_argv_init(sp_argv* a) {
    a->n = 0;
    a->overflow = 0;
    a->used = 0;
    a->v[0] = NULL;
}

static void sp_arg(sp_argv* a, const char* s) {
    size_t l = strlen(s) + 1;
    if (a->n + 1 >= (int)(sizeof a->v / sizeof a->v[0]) || l > sizeof a->arena - a->used) {
        a->overflow = 1;
        return;
    }
    char* d = a->arena + a->used;
    memcpy(d, s, l);
    a->used += l;
    a->v[a->n++] = d;
    a->v[a->n] = NULL;
}

typedef enum { SP_NONE = 0, SP_GSR, SP_WF_RECORDER, SP_FFMPEG, SP_PARECORD } sp_backend;

static const char* sp_backend_name(sp_backend b) {
    switch (b) {
    case SP_GSR:         return "gpu-screen-recorder";
    case SP_WF_RECORDER: return "wf-recorder";
    case SP_FFMPEG:      return "ffmpeg";
    case SP_PARECORD:    return "parecord";
    case SP_NONE:        break;
    }
    return "none";
}

typedef struct {
    int replay;            // 1 = instant-replay buffer, 0 = a straight recording
    int replay_seconds;
    int quality;           // 0 = 720p, 1 = 1080p, 2 = 4K
    int codec;             // 0 = H.264, 1 = HEVC, 2 = AV1
    int container;         // 0 = mp4, 1 = mkv (gpu-screen-recorder's replay clips)
    const char* mic;       // microphone source; NULL or "" = microphone off
    const char* desktop;   // desktop-audio source for ffmpeg and wf-recorder (a sink monitor)
    const char* out;       // output file; for a gpu-screen-recorder replay buffer, the clip directory
    int wayland;
    const char* display;   // the X display ffmpeg grabs
    int screen_w, screen_h;
    int has_gsr, has_wf, has_ffmpeg;
    int ffmpeg_pulse;      // ffmpeg can open audio-server sources at all
    // Test seams, left NULL by the application. The tests feed ffmpeg lavfi
    // generators in place of a screen and a sound card THROUGH THIS BUILDER,
    // so the command they run is the command the application runs.
    const char* audio_format;          // "pulse" when NULL
    const char* const* video_input;    // x11grab of `display` when NULL
} sp_capture;

// Build the screen-recorder command for `c` into `a`, and describe in `note`
// what audio it will capture. The note is written by the same code that
// builds the command, so the log cannot claim a microphone the command does
// not contain - which is how "Desktop + Mic [Merged]" was printed for the two
// recorders that dropped the microphone. Returns SP_NONE when there is no
// recorder or the command did not fit.
static sp_backend sp_build_capture(sp_argv* a, const sp_capture* c, char* note, size_t note_cap) {
    static const char* const gsr_codecs[] = {"h264", "hevc", "av1"};
    static const char* const gsr_quality[] = {"medium", "high", "very_high"};
    static const char* const resolutions[] = {"1280x720", "1920x1080", "3840x2160"};
    static const char* const ff_codecs[] = {"libx264", "libx265", "libsvtav1"};
    static const char* const ff_scale[] = {"scale=1280:-2", "scale=1920:-2", "scale=3840:-2"};
    static const char* const ff_crf[] = {"28", "24", "22"};
    const char* mic = (c->mic && c->mic[0]) ? c->mic : NULL;
    int q = (c->quality < 0 || c->quality > 2) ? 1 : c->quality;
    int k = (c->codec < 0 || c->codec > 2) ? 0 : c->codec;

    sp_argv_init(a);
    if (note_cap) note[0] = '\0';
    if (!c->out || !c->out[0]) return SP_NONE;

    if (c->has_gsr) {
        char audio[SP_NAME_LEN + 32];
        char secs[16];
        if (mic) {
            if (snprintf(audio, sizeof audio, "default_output|%s", mic) >= (int)sizeof audio) return SP_NONE;
            snprintf(note, note_cap, "desktop audio + microphone (%s), merged into one track", mic);
        } else {
            snprintf(audio, sizeof audio, "default_output");
            snprintf(note, note_cap, "desktop audio only (microphone disabled)");
        }
        sp_arg(a, "gpu-screen-recorder");
        sp_arg(a, "-w"); sp_arg(a, c->wayland ? "portal" : "screen");
        sp_arg(a, "-f"); sp_arg(a, "60");
        sp_arg(a, "-s"); sp_arg(a, resolutions[q]);
        sp_arg(a, "-a"); sp_arg(a, audio);
        if (c->replay) {
            snprintf(secs, sizeof secs, "%d", c->replay_seconds);
            sp_arg(a, "-r"); sp_arg(a, secs);
        }
        sp_arg(a, "-k"); sp_arg(a, gsr_codecs[k]);
        sp_arg(a, "-q"); sp_arg(a, gsr_quality[q]);
        if (c->replay) { sp_arg(a, "-c"); sp_arg(a, c->container ? "mkv" : "mp4"); }
        sp_arg(a, "-o"); sp_arg(a, c->out);
        return a->overflow ? SP_NONE : SP_GSR;
    }

    if (c->wayland && c->has_wf && c->desktop && c->desktop[0]) {
        // `--audio=DEVICE`, one argument: wf-recorder's -a takes an OPTIONAL
        // value, so the old `-a DEVICE` recorded the default source and left
        // DEVICE behind as a stray argument.
        char audio[SP_NAME_LEN + 16];
        if (snprintf(audio, sizeof audio, "--audio=%s", c->desktop) >= (int)sizeof audio) return SP_NONE;
        if (mic) {
            snprintf(note, note_cap,
                     "desktop audio only - wf-recorder records ONE audio source, so the "
                     "microphone (%s) is NOT in this recording. Install gpu-screen-recorder "
                     "to record both", mic);
        } else {
            snprintf(note, note_cap, "desktop audio only (microphone disabled)");
        }
        sp_arg(a, "wf-recorder");
        sp_arg(a, audio);
        sp_arg(a, "-f"); sp_arg(a, c->out);
        return a->overflow ? SP_NONE : SP_WF_RECORDER;
    }

    if (c->has_ffmpeg) {
        const char* fmt = c->audio_format ? c->audio_format : "pulse";
        int audio_ok = c->ffmpeg_pulse || c->audio_format != NULL;
        const char* inputs[2];
        int n_audio = 0;
        char size[32];

        if (audio_ok && c->desktop && c->desktop[0]) inputs[n_audio++] = c->desktop;
        if (audio_ok && mic) inputs[n_audio++] = mic;

        sp_arg(a, "ffmpeg"); sp_arg(a, "-y"); sp_arg(a, "-nostdin");
        if (c->video_input) {
            for (const char* const* p = c->video_input; *p; p++) sp_arg(a, *p);
        } else {
            snprintf(size, sizeof size, "%dx%d", c->screen_w, c->screen_h);
            sp_arg(a, "-f"); sp_arg(a, "x11grab");
            sp_arg(a, "-r"); sp_arg(a, "30");
            sp_arg(a, "-s"); sp_arg(a, size);
            sp_arg(a, "-i"); sp_arg(a, c->display ? c->display : ":0");
        }
        for (int i = 0; i < n_audio; i++) {
            // A live audio source that is not read quickly enough drops
            // samples while the video encoder is busy.
            sp_arg(a, "-thread_queue_size"); sp_arg(a, "1024");
            sp_arg(a, "-f"); sp_arg(a, fmt);
            sp_arg(a, "-i"); sp_arg(a, inputs[i]);
        }
        if (n_audio == 2) {
            // One track, both sources at full level. amix's default halves
            // each input, which would put the voice 6 dB down.
            sp_arg(a, "-filter_complex");
            sp_arg(a, "[1:a][2:a]amix=inputs=2:duration=longest:normalize=0[a]");
            sp_arg(a, "-map"); sp_arg(a, "0:v");
            sp_arg(a, "-map"); sp_arg(a, "[a]");
        } else if (n_audio == 1) {
            sp_arg(a, "-map"); sp_arg(a, "0:v");
            sp_arg(a, "-map"); sp_arg(a, "1:a");
        } else {
            sp_arg(a, "-map"); sp_arg(a, "0:v");
        }
        sp_arg(a, "-vf"); sp_arg(a, ff_scale[q]);
        sp_arg(a, "-c:v"); sp_arg(a, ff_codecs[k]);
        // SVT-AV1 takes a numeric preset; "veryfast" makes ffmpeg refuse the
        // whole command, and AV1 is the default codec on an RTX 40 card.
        sp_arg(a, "-preset"); sp_arg(a, k == 2 ? "10" : "veryfast");
        sp_arg(a, "-crf"); sp_arg(a, ff_crf[q]);
        if (n_audio > 0) { sp_arg(a, "-c:a"); sp_arg(a, "aac"); }
        sp_arg(a, c->out);

        if (!audio_ok) {
            snprintf(note, note_cap,
                     "NO AUDIO - ffmpeg cannot open the audio server, so neither the "
                     "desktop nor the microphone is recorded");
        } else if (n_audio == 2) {
            snprintf(note, note_cap, "desktop audio + microphone (%s), mixed into one track", mic);
        } else if (mic) {
            snprintf(note, note_cap, "microphone only (%s)", mic);
        } else if (n_audio == 1) {
            snprintf(note, note_cap, "desktop audio only (microphone disabled)");
        } else {
            snprintf(note, note_cap, "no audio");
        }
        return a->overflow ? SP_NONE : SP_FFMPEG;
    }

    return SP_NONE;
}

typedef struct {
    const char* mic;           // required
    const char* out_base;      // output path WITHOUT an extension
    int has_ffmpeg, has_parecord;
    const char* audio_format;  // test seam: "pulse" when NULL
} sp_voice;

// Build an audio-only recording of the microphone into `a`, and write the
// output path - with the extension the chosen recorder produces - to `path`.
//
// gpu-screen-recorder cannot do this: it always needs something to capture
// with -w. So it is ffmpeg, writing AAC in .m4a (which every phone and desktop
// plays), and failing that parecord, from the audio server's own tools,
// writing .wav. `-n` rather than `-y`: a voice recording never overwrites a
// file, whatever the caller did about unique names. No `-movflags +faststart`:
// its second pass rewrites the whole file after the recording stops, which
// lengthens the moment in which an interruption leaves a file with no index.
static sp_backend sp_build_voice(sp_argv* a, const sp_voice* v, char* path, size_t path_cap) {
    sp_argv_init(a);
    if (path_cap) path[0] = '\0';
    if (!v->mic || !v->mic[0] || !v->out_base || !v->out_base[0]) return SP_NONE;

    if (v->has_ffmpeg) {
        if (snprintf(path, path_cap, "%s.m4a", v->out_base) >= (int)path_cap) return SP_NONE;
        sp_arg(a, "ffmpeg"); sp_arg(a, "-n"); sp_arg(a, "-nostdin");
        sp_arg(a, "-thread_queue_size"); sp_arg(a, "1024");
        sp_arg(a, "-f"); sp_arg(a, v->audio_format ? v->audio_format : "pulse");
        sp_arg(a, "-i"); sp_arg(a, v->mic);
        sp_arg(a, "-vn");
        sp_arg(a, "-c:a"); sp_arg(a, "aac");
        sp_arg(a, "-b:a"); sp_arg(a, "128k");
        sp_arg(a, path);
        return a->overflow ? SP_NONE : SP_FFMPEG;
    }
    if (v->has_parecord) {
        char device[SP_NAME_LEN + 16];
        if (snprintf(device, sizeof device, "--device=%s", v->mic) >= (int)sizeof device) return SP_NONE;
        if (snprintf(path, path_cap, "%s.wav", v->out_base) >= (int)path_cap) return SP_NONE;
        sp_arg(a, "parecord");
        sp_arg(a, device);
        sp_arg(a, "--file-format=wav");
        sp_arg(a, path);
        return a->overflow ? SP_NONE : SP_PARECORD;
    }
    return SP_NONE;
}

#endif // SHADOWPLAY_CAPTURE_H
