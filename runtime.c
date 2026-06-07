#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <sys/mman.h>
#include "shadowplay_gui.h"

#ifndef MAP_32BIT
#define MAP_32BIT 0x40
#endif

#define POOL_SIZE (128 * 1024 * 1024) // 128 MB pool
static uint8_t* g_alloc_pool = NULL;
static size_t g_alloc_offset = 0;

static void init_allocator() {
    if (!g_alloc_pool) {
        g_alloc_pool = mmap(NULL, POOL_SIZE, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
        if (g_alloc_pool == MAP_FAILED) {
            perror("mmap MAP_32BIT failed");
            // Fallback to normal malloc
            g_alloc_pool = malloc(POOL_SIZE);
        }
        g_alloc_offset = 0;
    }
}

void* ymalloc(size_t size) {
    init_allocator();
    size = (size + 7) & ~7; // 8-byte alignment
    if (g_alloc_offset + size > POOL_SIZE) {
        fprintf(stderr, "Out of memory in 32-bit arena!\n");
        exit(1);
    }
    void* ptr = g_alloc_pool + g_alloc_offset;
    g_alloc_offset += size;
    return ptr;
}

void* yrealloc(void* ptr, size_t old_size, size_t new_size) {
    if (!ptr) return ymalloc(new_size);
    size_t aligned_old = (old_size + 7) & ~7;
    size_t aligned_new = (new_size + 7) & ~7;
    
    // In-place grow if it was the last allocated block
    if (g_alloc_pool + g_alloc_offset - aligned_old == (uint8_t*)ptr) {
        if (g_alloc_offset - aligned_old + aligned_new <= POOL_SIZE) {
            g_alloc_offset = g_alloc_offset - aligned_old + aligned_new;
            return ptr;
        }
    }
    
    void* new_ptr = ymalloc(new_size);
    memcpy(new_ptr, ptr, old_size);
    return new_ptr;
}

// ── String (YStr) ───────────────────────────────────────────

typedef struct {
    char* data;
    int32_t len;
    int32_t cap;
} YStr;

#define YSTR_HASH_SIZE 1048576
static void* g_ystr_hash[YSTR_HASH_SIZE];

static inline uint32_t ystr_hash_fn(void* p) {
    uintptr_t val = (uintptr_t)p;
    return (uint32_t)((val ^ (val >> 16)) & (YSTR_HASH_SIZE - 1));
}

static void register_ystr(void* p) {
    uint32_t h = ystr_hash_fn(p);
    while (g_ystr_hash[h] != NULL) {
        if (g_ystr_hash[h] == p) return;
        h = (h + 1) & (YSTR_HASH_SIZE - 1);
    }
    g_ystr_hash[h] = p;
}

static int is_valid_ystr(void* p) {
    if (!p) return 0;
    uint32_t h = ystr_hash_fn(p);
    while (g_ystr_hash[h] != NULL) {
        if (g_ystr_hash[h] == p) return 1;
        h = (h + 1) & (YSTR_HASH_SIZE - 1);
    }
    return 0;
}

static inline YStr* resolve_ystr(int32_t s) {
    if (s == 0) return NULL;
    if (is_valid_ystr((void*)(uintptr_t)s)) {
        return (YStr*)(uintptr_t)s;
    }
    int32_t actual_ptr = *(int32_t*)(uintptr_t)s;
    return (YStr*)(uintptr_t)actual_ptr;
}

int32_t ystr_new(const char* s) {
    YStr* str = ymalloc(sizeof(YStr));
    str->len = s ? strlen(s) : 0;
    str->cap = str->len + 1;
    str->data = ymalloc(str->cap);
    if (s) {
        strcpy(str->data, s);
    } else {
        str->data[0] = '\0';
    }
    register_ystr(str);
    // printf("[runtime] ystr_new: registered=%p, len=%d, data='%s'\n", (void*)str, str->len, str->len < 100 ? str->data : "...long...");
    return (int32_t)(uintptr_t)str;
}

int32_t ystr_clone(int32_t s) {
    YStr* src = resolve_ystr(s);
    if (!src) return ystr_new("");
    return ystr_new(src->data);
}

int32_t ystr_push(int32_t s, int32_t c) {
    YStr* str = resolve_ystr(s);
    if (!str) return 0;
    if (str->len + 1 >= str->cap) {
        int32_t old_cap = str->cap;
        str->cap = str->cap == 0 ? 8 : str->cap * 2;
        str->data = yrealloc(str->data, old_cap, str->cap);
    }
    str->data[str->len++] = (char)c;
    str->data[str->len] = '\0';
    return 0;
}

int32_t ystr_push_str(int32_t s, int32_t other) {
    YStr* str = resolve_ystr(s);
    YStr* oth = resolve_ystr(other);
    if (!str || !oth) return 0;
    int32_t new_len = str->len + oth->len;
    if (new_len >= str->cap) {
        int32_t old_cap = str->cap;
        str->cap = new_len + 1;
        str->data = yrealloc(str->data, old_cap, str->cap);
    }
    memcpy(str->data + str->len, oth->data, oth->len);
    str->len = new_len;
    str->data[str->len] = '\0';
    return 0;
}

int32_t ystr_eq(int32_t a, int32_t b) {
    YStr* strA = resolve_ystr(a);
    YStr* strB = resolve_ystr(b);
    if (!strA || !strB) return strA == strB;
    if (strA->len != strB->len) return 0;
    return memcmp(strA->data, strB->data, strA->len) == 0;
}

int32_t ystr_eq_cstr(int32_t a, int32_t b) {
    YStr* strA = resolve_ystr(a);
    const char* cstr = (const char*)(uintptr_t)b;
    if (!strA || !cstr) return 0;
    return strcmp(strA->data, cstr) == 0;
}

int32_t ystr_len(int32_t s) {
    YStr* str = resolve_ystr(s);
    if (!str) return 0;
    return str->len;
}

int32_t ystr_char_at(int32_t s, int32_t idx) {
    YStr* str = resolve_ystr(s);
    if (!str || idx < 0 || idx >= str->len) return 0;
    return str->data[idx];
}

int32_t ystr_free(int32_t s) {
    // Bump allocated, no-op
    return 0;
}

// ── Vector (YVec) ───────────────────────────────────────────

typedef struct {
    void* data;
    int32_t len;
    int32_t cap;
    int32_t elem_size;
} YVec;

int32_t yvec_new(int32_t elem_size) {
    YVec* v = ymalloc(sizeof(YVec));
    v->elem_size = elem_size;
    v->len = 0;
    v->cap = 8;
    v->data = ymalloc(v->cap * elem_size);
    // printf("[runtime] yvec_new: elem_size=%d -> v=0x%x, data=0x%x\n", elem_size, (int32_t)(uintptr_t)v, (int32_t)(uintptr_t)v->data);
    return (int32_t)(uintptr_t)v;
}

int32_t yvec_push(int32_t v, int32_t item) {
    YVec* vec = (YVec*)(uintptr_t)v;
    if (!vec) return 0;
    if (vec->len >= vec->cap) {
        int32_t old_cap = vec->cap;
        vec->cap *= 2;
        vec->data = yrealloc(vec->data, old_cap * vec->elem_size, vec->cap * vec->elem_size);
    }
    memcpy((char*)vec->data + vec->len * vec->elem_size, (void*)(uintptr_t)item, vec->elem_size);
    vec->len++;
    // if (vec->elem_size == 1 && vec->len < 10) {
    //     printf("[runtime] yvec_push_char: v=0x%x, len=%d, cap=%d, pushed='%c'\n", v, vec->len, vec->cap, ((char*)vec->data)[vec->len - 1]);
    // }
    return 0;
}

int32_t yvec_get(int32_t v, int32_t idx) {
    YVec* vec = (YVec*)(uintptr_t)v;
    if (!vec || idx < 0 || idx >= vec->len) return 0;
    return (int32_t)(uintptr_t)((char*)vec->data + idx * vec->elem_size);
}

int32_t yvec_get_char(int32_t v, int32_t idx) {
    YVec* vec = (YVec*)(uintptr_t)v;
    if (!vec) return 0;
    if (idx < 0 || idx >= vec->len) {
        // printf("[runtime] yvec_get_char OOB: v=0x%x, idx=%d, len=%d\n", v, idx, vec->len);
        return 0;
    }
    int32_t res = ((char*)vec->data)[idx];
    // printf("[runtime] yvec_get_char: v=0x%x, idx=%d, len=%d, val='%c'\n", v, idx, vec->len, res);
    return res;
}

int32_t yvec_len(int32_t v) {
    YVec* vec = (YVec*)(uintptr_t)v;
    if (!vec) return 0;
    // printf("[runtime] yvec_len: v=0x%x, len=%d\n", v, vec->len);
    return vec->len;
}

int32_t yvec_free(int32_t v) {
    // Bump allocated, no-op
    return 0;
}

// ── File I/O ─────────────────────────────────────────────

int32_t yfile_read_to_string(int32_t path) {
    YStr* p = resolve_ystr(path);
    if (!p) {
        // printf("[runtime] yfile_read_to_string: p is NULL!\n");
        return ystr_new("");
    }
    // printf("[runtime] yfile_read_to_string: path='%s'\n", p->data ? p->data : "NULL");
    FILE* f = fopen(p->data, "rb");
    if (!f) {
        // printf("[runtime] yfile_read_to_string: failed to open '%s'\n", p->data);
        return ystr_new("");
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(len + 1);
    size_t read_bytes = fread(buf, 1, len, f);
    buf[read_bytes] = '\0';
    fclose(f);
    // printf("[runtime] yfile_read_to_string: read %ld bytes\n", (long)read_bytes);
    int32_t res = ystr_new(buf);
    free(buf);
    return res;
}

int32_t yfile_write(int32_t path, int32_t contents) {
    YStr* p = resolve_ystr(path);
    YStr* c = resolve_ystr(contents);
    if (!p || !c) return 0;
    FILE* f = fopen(p->data, "wb");
    if (f) {
        fwrite(c->data, 1, c->len, f);
        fclose(f);
    }
    return 0;
}

// ── Utilities / Builtins ────────────────────────────────────

int32_t print_int(int32_t val) {
    printf("%d", val);
    fflush(stdout);
    return 0;
}

int32_t print(int32_t s) {
    YStr* str = resolve_ystr(s);
    if (str && str->data) {
        printf("%s", str->data);
    }
    fflush(stdout);
    return 0;
}

int32_t println(int32_t s) {
    YStr* str = resolve_ystr(s);
    if (str && str->data) {
        printf("%s\n", str->data);
    } else {
        printf("\n");
    }
    fflush(stdout);
    return 0;
}

int32_t ychar_to_ascii(int32_t c) {
    return c;
}

// ── AST Enum Variant Constructors ───────────────────────────

static int32_t make_enum(int32_t tag, int32_t count, ...) {
    int32_t* P = NULL;
    if (count > 0) {
        P = ymalloc(count * sizeof(int32_t));
        va_list args;
        va_start(args, count);
        for (int32_t i = 0; i < count; i++) {
            P[i] = va_arg(args, int32_t);
        }
        va_end(args);
    }
    
    int32_t* U = ymalloc(1 * sizeof(int32_t));
    U[0] = (int32_t)(uintptr_t)P;
    
    int32_t* E = ymalloc(2 * sizeof(int32_t));
    E[0] = tag;
    E[1] = (int32_t)(uintptr_t)U;
    return (int32_t)(uintptr_t)E;
}

// TokenKind
int32_t TokenKind_MmaMod(int32_t f0) { return make_enum(40, 1, f0); }
int32_t TokenKind_HardwareTarget(int32_t f0) { return make_enum(71, 1, f0); }
int32_t TokenKind_AtUnknown(int32_t f0) { return make_enum(88, 1, f0); }
int32_t TokenKind_IntLit(int32_t f0) { return make_enum(127, 1, f0); }
int32_t TokenKind_FloatLit(int32_t f0) { return make_enum(128, 1, f0); }
int32_t TokenKind_StringLit(int32_t f0) { return make_enum(129, 1, f0); }
int32_t TokenKind_CharLit(int32_t f0) { return make_enum(130, 1, f0); }
int32_t TokenKind_Ident(int32_t f0) { return make_enum(131, 1, f0); }
int32_t TokenKind_Unknown(int32_t f0) { return make_enum(133, 1, f0); }

// Expr
int32_t Expr_Ident(int32_t f0) { return make_enum(4, 1, f0); }
int32_t Expr_IntLit(int32_t f0) { return make_enum(0, 1, f0); }
int32_t Expr_FloatLit(int32_t f0) { return make_enum(12, 1, f0); }
int32_t Expr_StringLit(int32_t f0) { return make_enum(3, 1, f0); }
int32_t Expr_CharLit(int32_t f0) { return make_enum(1, 1, f0); }
int32_t Expr_BoolLit(int32_t f0) { return make_enum(2, 1, f0); }
int32_t Expr_Call(int32_t f0, int32_t f1, int32_t f2) { return make_enum(6, 3, f0, f1, f2); }
int32_t Expr_Index(int32_t f0, int32_t f1) { return make_enum(8, 2, f0, f1); }
int32_t Expr_MemberAccess(int32_t f0, int32_t f1) { return make_enum(7, 2, f0, f1); }
int32_t Expr_Path(int32_t f0, int32_t f1) { return make_enum(9, 2, f0, f1); }
int32_t Expr_BinaryExpr(int32_t f0, int32_t f1, int32_t f2) { return make_enum(5, 3, f0, f1, f2); }
int32_t Expr_UnaryExpr(int32_t f0, int32_t f1) { return make_enum(10, 2, f0, f1); }
int32_t Expr_StructLit(int32_t f0, int32_t f1, int32_t f2) { return make_enum(11, 3, f0, f1, f2); }

// Stmt
int32_t Stmt_Let(int32_t f0, int32_t f1, int32_t f2) { return make_enum(0, 3, f0, f1, f2); }
int32_t Stmt_Return(int32_t f0) { return make_enum(1, 1, f0); }
int32_t Stmt_If(int32_t f0, int32_t f1, int32_t f2, int32_t f3, int32_t f4) { return make_enum(2, 5, f0, f1, f2, f3, f4); }
int32_t Stmt_While(int32_t f0, int32_t f1, int32_t f2, int32_t f3) { return make_enum(3, 4, f0, f1, f2, f3); }
int32_t Stmt_For(int32_t f0, int32_t f1, int32_t f2, int32_t f3, int32_t f4, int32_t f5, int32_t f6) { return make_enum(4, 7, f0, f1, f2, f3, f4, f5, f6); }
int32_t Stmt_Match(int32_t f0, int32_t f1, int32_t f2) { return make_enum(5, 3, f0, f1, f2); }
int32_t Stmt_Assign(int32_t f0, int32_t f1) { return make_enum(6, 2, f0, f1); }
int32_t Stmt_CompoundAssign(int32_t f0, int32_t f1, int32_t f2) { return make_enum(7, 3, f0, f1, f2); }
int32_t Stmt_ExprStmt(int32_t f0) { return make_enum(8, 1, f0); }
int32_t Stmt_SafeBlock(int32_t f0, int32_t f1) { return make_enum(9, 2, f0, f1); }

// MatchPattern
int32_t MatchPattern_Ident(int32_t f0) { return make_enum(0, 1, f0); }
int32_t MatchPattern_EnumVariant(int32_t f0, int32_t f1) { return make_enum(1, 2, f0, f1); }
int32_t MatchPattern_Literal(int32_t f0) { return make_enum(2, 1, f0); }

int32_t String_new(const char* s) {
    return ystr_new(s);
}

int64_t str_to_i64(int32_t s) {
    YStr* str = resolve_ystr(s);
    if (!str || !str->data) return 0;
    return strtoll(str->data, NULL, 10);
}

// Entry Point
int32_t ysu_main(void);
#ifndef Y_NO_MAIN
int main(int argc, char** argv) {
    init_allocator();
    size_t stack_size = 4 * 1024 * 1024; // 4 MB stack
    void* stack_alloc = ymalloc(stack_size);
    void* stack_top = (char*)stack_alloc + stack_size;
    
    // Align stack pointer to 16-byte boundary for ABI compliance
    uintptr_t stack_top_aligned = ((uintptr_t)stack_top) & ~0xfULL;

    // printf("[runtime] 32-bit stack allocated at %p, top aligned at %p\n", stack_alloc, (void*)stack_top_aligned);
    // fflush(stdout);

    register uintptr_t rsp_val __asm__("rbx") = stack_top_aligned;
    __asm__ volatile(
        "movq %0, %%rsp\n\t"
        "call ysu_main\n\t"
        "movl %%eax, %%edi\n\t"
        "call exit\n\t"
        :
        : "r"(rsp_val)
        : "rdi", "memory"
    );
    return 0;
}
#endif
