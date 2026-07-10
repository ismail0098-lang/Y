#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

// --- Core Structs ---

typedef struct {
    char*  data;
    size_t len;
    size_t cap;
} YStr;

void yprint_char(char c) {
    printf("%c", c);
    fflush(stdout);
}

typedef struct {
    void*  data;
    size_t len;
    size_t cap;
    size_t elem_size;
} YVec;

// --- String Primitives ---

YStr* ystr_new(const char* s) {
    size_t len = s ? strlen(s) : 0;
    size_t cap = len + 1;
    char* data = (char*)malloc(cap);
    if (s) memcpy(data, s, len);
    data[len] = '\0';

    YStr* result = (YStr*)malloc(sizeof(YStr));
    result->data = data;
    result->len = len;
    result->cap = cap;
    return result;
}

YStr* ystr_clone(const YStr* s) {
    if (!s) return ystr_new("");
    return ystr_new(s->data);
}

void ystr_push(YStr* s, char c) {
    if (s->len + 1 >= s->cap) {
        s->cap = s->cap == 0 ? 8 : s->cap * 2;
        s->data = (char*)realloc(s->data, s->cap);
    }
    s->data[s->len++] = c;
    s->data[s->len] = '\0';
}

void ystr_push_str(YStr* s, const YStr* other) {
    size_t olen = other->len;
    while (s->len + olen >= s->cap) {
        s->cap = s->cap == 0 ? (olen + 8) : s->cap * 2;
        s->data = (char*)realloc(s->data, s->cap);
    }
    memcpy(s->data + s->len, other->data, olen);
    s->len += olen;
    s->data[s->len] = '\0';
}

bool ystr_eq(const YStr* a, const YStr* b) {
    if (!a || !b) return false;
    if (a->len != b->len) return false;
    return memcmp(a->data, b->data, a->len) == 0;
}

bool ystr_eq_cstr(const YStr* a, const char* b) {
    if (!a || !a->data || !b) return false;
    return strcmp(a->data, b) == 0;
}

char ystr_char_at(const YStr* s, size_t i) {
    if (!s || i >= s->len) return '\0';
    return s->data[i];
}

size_t ystr_len(const YStr* s) {
    return s ? s->len : 0;
}

void ystr_free(YStr* s) {
    if (s) {
        free(s->data);
        free(s);
    }
}

// --- Vector Primitives ---

YVec* yvec_new(size_t elem_size) {
    YVec* result = (YVec*)malloc(sizeof(YVec));
    result->data = NULL;
    result->len = 0;
    result->cap = 0;
    result->elem_size = elem_size;
    return result;
}

void yvec_push(YVec* v, const void* elem) {
    if (v->len >= v->cap) {
        v->cap = v->cap == 0 ? 8 : v->cap * 2;
        v->data = realloc(v->data, v->cap * v->elem_size);
    }
    memcpy((char*)v->data + v->len * v->elem_size, elem, v->elem_size);
    v->len++;
}

void* yvec_get(const YVec* v, size_t i) {
    if (!v || i >= v->len) return NULL;
    return (char*)v->data + i * v->elem_size;
}

char yvec_get_char(const YVec* v, size_t i) {
    void* ptr = yvec_get(v, i);
    char ch = ptr ? *((char*)ptr) : '\0';
    return ch;
}

size_t yvec_len(const YVec* v) {
    return v ? v->len : 0;
}

void yvec_free(YVec* v) {
    if (v) {
        free(v->data);
        free(v);
    }
}

// --- I/O Primitives ---

void yprint_str(const YStr* s) {
    if (s && s->data) {
        printf("%s", s->data);
        fflush(stdout);
    }
}

void yprintln_str(const YStr* s) {
    if (s && s->data) printf("%s\n", s->data);
    else printf("\n");
    fflush(stdout);
}

void yprint_int(int64_t v) {
    printf("%lld\n", (long long)v);
    fflush(stdout);
}

YStr* yfile_read_to_string(const YStr* path_str) {
    if (!path_str) return ystr_new("");
    const char* path = path_str->data;
    FILE* f = fopen(path, "rb");
    if (!f) return ystr_new("");

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buffer = (char*)malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);

    YStr* result = ystr_new(buffer);
    free(buffer);
    return result;
}

void yfile_write(const YStr* path_str, const YStr* content) {
    if (!path_str || !content) return;
    FILE* f = fopen(path_str->data, "wb");
    if (!f) return;
    fwrite(content->data, 1, content->len, f);
    fclose(f);
}

// --- Standard Names (for LlvmEmitter) ---

void print(const YStr* s) { yprint_str(s); }
void println(const YStr* s) { yprintln_str(s); }
void print_int(int64_t v) { yprint_int(v); }

// String, Vec, File methods are now implemented in Y (compiler.yy) 
// and call the y-prefixed primitives above.
// No need for duplicate definitions here.

