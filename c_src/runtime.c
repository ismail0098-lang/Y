#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

void* ystr_new(const char* s) {
    return (void*)s;
}

void ystr_push(void* s, char c) {}
void ystr_push_str(void* s, void* s2) {}
int32_t ystr_eq_cstr(void* s, const char* s2) {
    return strcmp((const char*)s, s2) == 0;
}
int64_t ystr_len(void* s) {
    return strlen((const char*)s);
}
char ystr_char_at(void* s, int64_t i) {
    return ((const char*)s)[i];
}
void* ystr_clone(void* s) {
    return (void*)s;
}

// Vector
void* yvec_new(int64_t cap) { return malloc(8); }
void yvec_push(void* v, void* item) {}
void* yvec_get(void* v, int64_t idx) { return NULL; }
int64_t yvec_len(void* v) { return 0; }

// File
void* yfile_read_to_string(void* path) { return NULL; }
void yfile_write(void* path, void* contents) {}

// Utilities
void print_int(int32_t val) {
    printf("%d", val);
}

void println(void* s) {
    printf("%s\n", (const char*)s);
}
