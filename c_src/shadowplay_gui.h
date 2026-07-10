// c_src/shadowplay_gui.h
// Graphical X11 ShadowPlay Overlay HUD for Y compiler runtime

#ifndef SHADOWPLAY_GUI_H
#define SHADOWPLAY_GUI_H

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <time.h>
#include <fcntl.h>

// Custom X11 Error Handler to prevent crashes if a key is already grabbed
static int x11_error_handler(Display* d, XErrorEvent* e) {
    char err_msg[256];
    XGetErrorText(d, e->error_code, err_msg, sizeof(err_msg));
    fprintf(stderr, "[X11 Warning] Handled X11 protocol error: %s (request_code=%d)\n", err_msg, e->request_code);
    return 0;
}

extern void cleanup_shadowplay_gui(void);
static void draw_ui(void);

static int x11_io_error_handler(Display* d) {
    fprintf(stderr, "[X11 Fatal] X connection lost/IO Error occurred.\n");
    cleanup_shadowplay_gui();
    exit(1);
}

// ShadowPlay Overlay States
static Display* dpy = NULL;
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

// Keyboard navigation
static int selected_idx = 0;    // 0=Replay, 1=Record, 2=Broadcast, 3=Format, 4=Quality, 5=Codec, 6=ReplayLength, 7=Keybind, 8=Mic
#define NUM_ITEMS 9

static pid_t record_pid = 0;
static pid_t replay_pid = 0;

static int has_gpu_screen_recorder = 0;
static int has_wf_recorder = 0;
static int has_ffmpeg = 0;

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
        XMoveResizeWindow(dpy, win, (screen_w - 600) / 2, (screen_h - 480) / 2, 600, 480);
        XMapWindow(dpy, win);
    } else if (toast_active) {
        // Top-right toast notification
        XMoveResizeWindow(dpy, win, screen_w - 300, 40, 280, 80);
        XMapWindow(dpy, win);
    } else if (recording || instant_replay) {
        // Top-right tiny status indicator
        XMoveResizeWindow(dpy, win, screen_w - 90, 40, 70, 32);
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

static char mic_devices[16][128]; // support up to 16 mics
static char mic_labels[16][128];
static int num_mic_devices = 0;
static int selected_mic_idx = 0; // 0=OFF, 1=Default Mic, ...

static void detect_audio_devices() {
    strcpy(mic_devices[0], "OFF");
    strcpy(mic_labels[0], "Disabled");
    
    strcpy(mic_devices[1], "default_input");
    strcpy(mic_labels[1], "Default Input");
    
    num_mic_devices = 2;
    selected_mic_idx = 1; // Default to Default Input

    FILE* fp = popen("gpu-screen-recorder -a check_devices 2>&1", "r");
    if (!fp) return;

    char line[256];
    int start_parsing = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "expected one of:")) {
            start_parsing = 1;
            continue;
        }
        if (start_parsing) {
            char* ptr = line;
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            if (*ptr == '\n' || *ptr == '\0') continue;
            
            char dev[128];
            char lbl[128];
            char* open_paren = strchr(ptr, '(');
            if (open_paren) {
                int dev_len = open_paren - ptr;
                while (dev_len > 0 && (ptr[dev_len - 1] == ' ' || ptr[dev_len - 1] == '\t')) dev_len--;
                if (dev_len >= 127) dev_len = 127;
                strncpy(dev, ptr, dev_len);
                dev[dev_len] = '\0';
                
                char* close_paren = strchr(open_paren, ')');
                if (close_paren) {
                    int lbl_len = close_paren - (open_paren + 1);
                    if (lbl_len >= 127) lbl_len = 127;
                    strncpy(lbl, open_paren + 1, lbl_len);
                    lbl[lbl_len] = '\0';
                } else {
                    strcpy(lbl, dev);
                }
            } else {
                char* nl = strchr(ptr, '\n');
                if (nl) *nl = '\0';
                strcpy(dev, ptr);
                strcpy(lbl, ptr);
            }
            
            if (strstr(dev, "input") || strstr(dev, "source")) {
                if (strcmp(dev, "default_input") != 0 && num_mic_devices < 16) {
                    strcpy(mic_devices[num_mic_devices], dev);
                    strcpy(mic_labels[num_mic_devices], lbl);
                    num_mic_devices++;
                }
            }
        }
    }
    pclose(fp);
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
    strncpy(buf, "default", max_len);
}

static int has_pulse_audio() {
    int status = system("ffmpeg -y -f pulse -i default -t 0.1 -f null - >/dev/null 2>&1");
    return (status == 0);
}

static void start_manual_recording() {
    if (record_pid > 0) return;
    
    printf("[ShadowPlay] Starting manual recording...\n");
    if (selected_mic_idx == 0) {
        printf("[ShadowPlay] Audio Source: Desktop only (default_output)\n");
    } else {
        printf("[ShadowPlay] Audio Source: Desktop (default_output) + Mic (%s) [Merged]\n", mic_devices[selected_mic_idx]);
    }
    fflush(stdout);
    
    if (is_wsl()) {
        printf("[ShadowPlay Warning] You are running inside WSL. Linux screen recorders (like ffmpeg/wf-recorder) can only capture Linux GUI windows inside WSLg, not your main Windows host desktop screen.\n");
    }

    // Create output folder
    system("mkdir -p ~/Videos/Y_Captures");
    
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "%s/Videos/Y_Captures/Manual_Capture_%ld.%s", 
             getenv("HOME"), time(NULL), file_format == 0 ? "mp4" : "mkv");

    record_pid = fork();
    if (record_pid == 0) {
        if (dpy) {
            close(ConnectionNumber(dpy));
        }
        char* dpy_env = getenv("DISPLAY");
        if (!dpy_env) dpy_env = ":0.0";
        
        char grab_res[64];
        snprintf(grab_res, sizeof(grab_res), "%dx%d", screen_width, screen_height);
        
        char scale_filter[64] = "";
        char crf_val[8] = "24";
        if (quality == 0) {
            strcpy(scale_filter, "scale=1280:-2");
            strcpy(crf_val, "28");
        } else if (quality == 1) {
            strcpy(scale_filter, "scale=1920:-2");
            strcpy(crf_val, "24");
        } else {
            strcpy(scale_filter, "scale=3840:-2");
            strcpy(crf_val, "22");
        }

        freopen("/dev/null", "r", stdin);
        freopen("/tmp/y_recording_log.txt", "w", stdout);
        freopen("/tmp/y_recording_log.txt", "w", stderr);

        if (has_gpu_screen_recorder) {
            char res_str[64];
            if (quality == 0) strcpy(res_str, "1280x720");
            else if (quality == 1) strcpy(res_str, "1920x1080");
            else strcpy(res_str, "3840x2160");
            
            char* session = getenv("XDG_SESSION_TYPE");
            int is_wayland = (session && strcmp(session, "wayland") == 0);
            
            char audio_arg[256];
            if (selected_mic_idx == 0) {
                strcpy(audio_arg, "default_output");
            } else {
                snprintf(audio_arg, sizeof(audio_arg), "default_output|%s", mic_devices[selected_mic_idx]);
            }

            char quality_preset[16] = "very_high";
            if (quality == 0) {
                strcpy(quality_preset, "medium");
            } else if (quality == 1) {
                strcpy(quality_preset, "high");
            }

            char* codec_str = "h264";
            if (video_codec == 1) codec_str = "hevc";
            else if (video_codec == 2) codec_str = "av1";

            execlp("gpu-screen-recorder", "gpu-screen-recorder", "-w", is_wayland ? "portal" : "screen", "-f", "60", "-s", res_str, "-a", audio_arg, "-k", codec_str, "-q", quality_preset, "-o", filepath, NULL);
        } else {
            char* session = getenv("XDG_SESSION_TYPE");
            int is_wayland = (session && strcmp(session, "wayland") == 0);
            
            if (is_wayland && has_wf_recorder) {
                execlp("wf-recorder", "wf-recorder", "-a", default_audio_dev, "-f", filepath, NULL);
            }

            if (has_ffmpeg) {
                int has_audio = has_pulse_audio();
                char ffmpeg_codec[32] = "libx264";
                if (video_codec == 1) {
                    strcpy(ffmpeg_codec, "libx265");
                } else if (video_codec == 2) {
                    strcpy(ffmpeg_codec, "libsvtav1");
                }

                if (has_audio) {
                    execlp("ffmpeg", "ffmpeg", "-y", "-nostdin", "-f", "x11grab", "-r", "30", "-s", grab_res, "-i", dpy_env, 
                           "-f", "pulse", "-i", default_audio_dev, "-c:v", ffmpeg_codec, "-preset", "veryfast", "-crf", crf_val, 
                           "-vf", scale_filter, "-c:a", "aac", filepath, NULL);
                } else {
                    execlp("ffmpeg", "ffmpeg", "-y", "-nostdin", "-f", "x11grab", "-r", "30", "-s", grab_res, "-i", dpy_env, 
                           "-c:v", ffmpeg_codec, "-preset", "veryfast", "-crf", crf_val, 
                           "-vf", scale_filter, filepath, NULL);
                }
            }
        }
        _exit(1);
    }
    printf("[ShadowPlay] Manual screen recording started: %s\n", filepath);
    show_toast("Recording Started");
}

static void stop_manual_recording() {
    if (record_pid > 0) {
        kill(record_pid, SIGINT);
        int status;
        for (int i = 0; i < 50; i++) {
            if (waitpid(record_pid, &status, WNOHANG) > 0) break;
            usleep(10000);
        }
        record_pid = 0;
        printf("[ShadowPlay] Manual screen recording saved to ~/Videos/Y_Captures/.\n");
        show_toast("Recording Saved");
    }
}

static void start_replay_buffer() {
    if (replay_pid > 0) return;
    
    printf("[ShadowPlay] Starting Instant Replay buffer...\n");
    if (selected_mic_idx == 0) {
        printf("[ShadowPlay] Audio Source: Desktop only (default_output)\n");
    } else {
        printf("[ShadowPlay] Audio Source: Desktop (default_output) + Mic (%s) [Merged]\n", mic_devices[selected_mic_idx]);
    }
    fflush(stdout);
    
    if (is_wsl()) {
        printf("[ShadowPlay Warning] You are running inside WSL. Linux screen recorders (like ffmpeg/wf-recorder) can only capture Linux GUI windows inside WSLg, not your main Windows host desktop screen.\n");
    }

    replay_pid = fork();
    if (replay_pid == 0) {
        if (dpy) {
            close(ConnectionNumber(dpy));
        }
        char* dpy_env = getenv("DISPLAY");
        if (!dpy_env) dpy_env = ":0.0";
        
        char grab_res[64];
        snprintf(grab_res, sizeof(grab_res), "%dx%d", screen_width, screen_height);
        
        char scale_filter[64] = "";
        char crf_val[8] = "24";
        if (quality == 0) {
            strcpy(scale_filter, "scale=1280:-2");
            strcpy(crf_val, "28");
        } else if (quality == 1) {
            strcpy(scale_filter, "scale=1920:-2");
            strcpy(crf_val, "24");
        } else {
            strcpy(scale_filter, "scale=3840:-2");
            strcpy(crf_val, "22");
        }

        freopen("/dev/null", "r", stdin);
        freopen("/tmp/y_recording_log.txt", "w", stdout);
        freopen("/tmp/y_recording_log.txt", "w", stderr);

        if (has_gpu_screen_recorder) {
            char res_str[64];
            if (quality == 0) strcpy(res_str, "1280x720");
            else if (quality == 1) strcpy(res_str, "1920x1080");
            else strcpy(res_str, "3840x2160");
            
            char fmt_str[8];
            strcpy(fmt_str, file_format == 0 ? "mp4" : "mkv");
            
            char out_dir[256];
            snprintf(out_dir, sizeof(out_dir), "%s/Videos/Y_Captures", getenv("HOME"));
            
            char dur_str[16];
            snprintf(dur_str, sizeof(dur_str), "%d", replay_durations[replay_duration_idx]);
            
            char* session = getenv("XDG_SESSION_TYPE");
            int is_wayland = (session && strcmp(session, "wayland") == 0);
            
            char audio_arg[256];
            if (selected_mic_idx == 0) {
                strcpy(audio_arg, "default_output");
            } else {
                snprintf(audio_arg, sizeof(audio_arg), "default_output|%s", mic_devices[selected_mic_idx]);
            }

            char quality_preset[16] = "very_high";
            if (quality == 0) {
                strcpy(quality_preset, "medium");
            } else if (quality == 1) {
                strcpy(quality_preset, "high");
            }

            char* codec_str = "h264";
            if (video_codec == 1) codec_str = "hevc";
            else if (video_codec == 2) codec_str = "av1";

            execlp("gpu-screen-recorder", "gpu-screen-recorder", "-w", is_wayland ? "portal" : "screen", "-f", "60", "-s", res_str, "-a", audio_arg, "-r", dur_str, "-k", codec_str, "-q", quality_preset, "-c", fmt_str, "-o", out_dir, NULL);
        } else {
            char* session = getenv("XDG_SESSION_TYPE");
            int is_wayland = (session && strcmp(session, "wayland") == 0);
            
            if (is_wayland && has_wf_recorder) {
                execlp("wf-recorder", "wf-recorder", "-a", default_audio_dev, "-f", "/tmp/y_replay_buffer.mp4", NULL);
            }

            if (has_ffmpeg) {
                int has_audio = has_pulse_audio();
                char ffmpeg_codec[32] = "libx264";
                if (video_codec == 1) {
                    strcpy(ffmpeg_codec, "libx265");
                } else if (video_codec == 2) {
                    strcpy(ffmpeg_codec, "libsvtav1");
                }

                if (has_audio) {
                    execlp("ffmpeg", "ffmpeg", "-y", "-nostdin", "-f", "x11grab", "-r", "30", "-s", grab_res, "-i", dpy_env, 
                           "-f", "pulse", "-i", default_audio_dev, "-c:v", ffmpeg_codec, "-preset", "veryfast", "-crf", crf_val, 
                           "-vf", scale_filter, "-c:a", "aac", "/tmp/y_replay_buffer.mp4", NULL);
                } else {
                    execlp("ffmpeg", "ffmpeg", "-y", "-nostdin", "-f", "x11grab", "-r", "30", "-s", grab_res, "-i", dpy_env, 
                           "-c:v", ffmpeg_codec, "-preset", "veryfast", "-crf", crf_val, 
                           "-vf", scale_filter, "/tmp/y_replay_buffer.mp4", NULL);
                }
            }
        }
        _exit(1);
    }
    printf("[ShadowPlay] Instant Replay background buffer activated (saving to /tmp/y_replay_buffer.mp4).\n");
    show_toast("Instant Replay ON");
}

static void stop_replay_buffer() {
    if (replay_pid > 0) {
        kill(replay_pid, SIGINT);
        int status;
        for (int i = 0; i < 50; i++) {
            if (waitpid(replay_pid, &status, WNOHANG) > 0) break;
            usleep(10000);
        }
        replay_pid = 0;
    }
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

extern void cleanup_shadowplay_gui() {
    stop_manual_recording();
    stop_replay_buffer();
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

static void draw_ui() {
    if (!dpy || !win) return;
    if (!visible && !toast_active && !recording && !instant_replay) return;

    if (toast_active) {
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
        XDrawString(dpy, win, gc, 65, 35, "ShadowPlay Replay", 17);
        XSetForeground(dpy, gc, color_grey);
        XDrawString(dpy, win, gc, 65, 55, toast_text, strlen(toast_text));
        
        XFreeGC(dpy, gc);
        XFlush(dpy);
        return;
    }

    if (!visible && (recording || instant_replay)) {
        // Clear window to dark card background
        XSetWindowBackground(dpy, win, color_card);
        XClearWindow(dpy, win);
        
        GC gc = XCreateGC(dpy, win, 0, NULL);
        if (hud_font) XSetFont(dpy, gc, hud_font->fid);
        
        // Draw border
        XSetForeground(dpy, gc, color_grey);
        XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        XDrawRectangle(dpy, win, gc, 0, 0, 68, 30);
        
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
    
    // Draw 3 primary columns/cards (Replay, Record, Broadcast)
    int col_width = 160;
    int col_height = 100;
    int start_y = 70;
    int r = 8; // rounded corner radius
    
    for (int i = 0; i < 3; i++) {
        int start_x = 30 + i * (col_width + 30);
        
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
    int quality_x = 430;
    if (selected_idx == 4) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, 300, row1_y + 15, "> Video Quality:", 16);
        XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        draw_rounded_rect(dpy, win, gc, quality_x - 10, row1_y, 150, 22, 4);
    } else {
        XSetForeground(dpy, gc, color_white);
        XDrawString(dpy, win, gc, 300, row1_y + 15, "  Video Quality:", 16);
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
    int length_x = 430;
    if (selected_idx == 6) {
        XSetForeground(dpy, gc, color_green);
        XDrawString(dpy, win, gc, 300, row2_y + 15, "> Replay Length:", 16);
        XSetLineAttributes(dpy, gc, 1, LineSolid, CapButt, JoinMiter);
        draw_rounded_rect(dpy, win, gc, length_x - 10, row2_y, 150, 22, 4);
    } else {
        XSetForeground(dpy, gc, color_white);
        XDrawString(dpy, win, gc, 300, row2_y + 15, "  Replay Length:", 16);
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

    char* bind_labels[] = {"Alt+S", "Alt+F10", "Alt+R", "Alt+X"};
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
        draw_rounded_rect(dpy, win, gc, mic_x - 10, row4_y, 430, 22, 4);
    } else {
        XSetForeground(dpy, gc, color_white);
        XDrawString(dpy, win, gc, 30, row4_y + 15, "  Microphone:", 13);
    }

    XSetForeground(dpy, gc, color_green);
    char mic_opt[128];
    snprintf(mic_opt, sizeof(mic_opt), "[ %s ]", mic_labels[selected_mic_idx]);
    XDrawString(dpy, win, gc, mic_x, row4_y + 16, mic_opt, strlen(mic_opt));

    // Help Footer
    XSetForeground(dpy, gc, color_card);
    XSetLineAttributes(dpy, gc, 2, LineSolid, CapButt, JoinMiter);
    XDrawLine(dpy, win, gc, 30, 395, 570, 395);

    XSetForeground(dpy, gc, color_grey);
    char footer_msg[128];
    if (instant_replay) {
        snprintf(footer_msg, sizeof(footer_msg), "Alt+Z: Hide HUD | %s: Save Last %ds Replay | Arrows: Navigate | Enter: Toggle", bind_labels[replay_keybind_idx], replay_durations[replay_duration_idx]);
    } else {
        snprintf(footer_msg, sizeof(footer_msg), "Alt+Z: Hide HUD | Arrows: Navigate | Enter: Toggle Option | Esc: Close");
    }
    XDrawString(dpy, win, gc, 30, 425, footer_msg, strlen(footer_msg));

    XFreeGC(dpy, gc);
    XFlush(dpy);
}

static void handle_sigint(int sig) {
    printf("\n[ShadowPlay] Interrupted! Performing clean shutdown...\n");
    cleanup_shadowplay_gui();
    exit(0);
}

// Global initialization
extern int32_t init_shadowplay_gui() {
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

    // Set custom error handlers
    XSetErrorHandler(x11_error_handler);
    XSetIOErrorHandler(x11_io_error_handler);

    // Check system dependencies
    has_gpu_screen_recorder = (system("which gpu-screen-recorder > /dev/null 2>&1") == 0);
    has_wf_recorder = (system("which wf-recorder > /dev/null 2>&1") == 0);
    has_ffmpeg = (system("which ffmpeg > /dev/null 2>&1") == 0);

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
    KeyCode s_code = XKeysymToKeycode(dpy, XK_s);

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

    // Create borderless window (CWOverrideRedirect)
    XSetWindowAttributes attrs;
    attrs.override_redirect = True;
    attrs.background_pixel = color_bg;
    attrs.event_mask = ExposureMask | KeyPressMask;

    int screen_w = DisplayWidth(dpy, screen);
    int screen_h = DisplayHeight(dpy, screen);
    int win_w = 600;
    int win_h = 480;

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

// State accessors for Y code
extern int32_t is_overlay_visible() { return visible; }
extern int32_t get_instant_replay_state() { return instant_replay; }
extern int32_t get_recording_state() { return recording; }
extern int32_t get_broadcast_state() { return broadcast; }
extern int32_t get_file_format_state() { return file_format; }
extern int32_t get_quality_state() { return quality; }
extern int32_t get_codec_state() { return video_codec; }
extern int32_t get_replay_duration() { return replay_durations[replay_duration_idx]; }
extern int32_t get_replay_duration_idx() { return replay_duration_idx; }
extern int32_t get_microphone_index() { return selected_mic_idx; }
extern void get_microphone_name(char* out_buf) { strcpy(out_buf, mic_labels[selected_mic_idx]); }

// Check X11 events and update the display
extern int32_t update_shadowplay_gui() {
    if (!dpy) return -1;

    if (toast_active && (time(NULL) - toast_start_time >= 3)) {
        toast_active = 0;
        update_window_layout();
    }

    XEvent ev;
    while (XPending(dpy)) {
        XNextEvent(dpy, &ev);
        
        if (ev.type == KeyPress) {
            KeySym keysym = XLookupKeysym(&ev.xkey, 0);
            
            // Print diagnostic debug logs for any keypresses caught by our grab list
            printf("[X11 debug] KeyPress detected: keycode=%d, keysym=%lu (%s), state=%u\n",
                   ev.xkey.keycode, (unsigned long)keysym, XKeysymToString(keysym), ev.xkey.state);
            
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
                    if (selected_idx < 3) {
                        selected_idx = (selected_idx - 1 + 3) % 3;
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
                    if (selected_idx < 3) {
                        selected_idx = (selected_idx + 1) % 3;
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
                    if (selected_idx == 0 || selected_idx == 1 || selected_idx == 2) selected_idx = 8;
                    else if (selected_idx == 3) selected_idx = 0;
                    else if (selected_idx == 4) selected_idx = 1;
                    else if (selected_idx == 5) selected_idx = 3;
                    else if (selected_idx == 6) selected_idx = 4;
                    else if (selected_idx == 7) selected_idx = 5;
                    else if (selected_idx == 8) selected_idx = 7;
                    draw_ui();
                } else if (keysym == XK_Down) {
                    if (selected_idx == 0) selected_idx = 3;
                    else if (selected_idx == 1 || selected_idx == 2) selected_idx = 4;
                    else if (selected_idx == 3) selected_idx = 5;
                    else if (selected_idx == 4) selected_idx = 6;
                    else if (selected_idx == 5) selected_idx = 7;
                    else if (selected_idx == 6) selected_idx = 7;
                    else if (selected_idx == 7) selected_idx = 8;
                    else if (selected_idx == 8) selected_idx = 0;
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
                        selected_mic_idx = (selected_mic_idx + 1) % num_mic_devices;
                        if (instant_replay) {
                            stop_replay_buffer();
                            start_replay_buffer();
                        }
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
    if (visible || toast_active || recording || instant_replay) {
        draw_ui();
    }
    
    return 0;
}

#endif
