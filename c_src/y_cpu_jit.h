#ifndef Y_CPU_JIT_H
#define Y_CPU_JIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Y_CPU_JIT_OPTIONS_ABI_VERSION 1u
#define Y_CPU_JIT_OPT_LEVEL_INHERIT (-1)

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t opt_level;
    int32_t training_opt_level;
    int32_t codegen_opt_level;
} YCpuJitOptions;

typedef struct {
    uint32_t kind;
    uint32_t reserved;
    uint64_t bits;
} YCpuJitValue;

/* Source must be trusted, NUL-terminated UTF-8. Source compilation accepts a
 * resolved compilation unit. Compile/free/handle operations run on the creating
 * thread. Native addresses expire when the handle is freed; all native calls
 * must finish first. Pointer arguments and native ABI signatures remain caller
 * responsibilities. Every non-NULL pointer must be valid for its documented
 * object and lifetime; error_out storage must be disjoint from other arguments.
 *
 * Pointer results use NULL for failure; status results use 0/-1. error_out may
 * be NULL; otherwise each call clears it and supplies an allocated message on
 * failure. Free previous messages before reusing error_out. Release messages
 * and returned JSON with y_free_string. Compilation never executes source.
 */
void y_free_string(char *string);
void y_cpu_jit_free(void *jit);

/* Initializes O3 plus inherited training/codegen tiers. Wrong size or NULL
 * storage returns -1 without writing the options. Pass sizeof(*options). */
int32_t y_cpu_jit_options_init(
    YCpuJitOptions *options, uint32_t options_size, char **error_out);

/* NULL options selects defaults. Non-NULL options require a readable two-u32
 * header and the full declared object when version and exact size match.
 * Options are copied during compilation. opt_level accepts 0..3; optional
 * tiers accept -1 (inherit) or 0..3. All fields are validated in every mode.
 * training_opt_level changes only instrumented IR. Final/ordinary IR follows
 * opt_level; native codegen follows codegen_opt_level or opt_level in all modes.
 * Per-pass verification and mandatory module checks remain enabled. */
void *y_cpu_jit_compile_with_options(
    const char *source, const YCpuJitOptions *options, char **error_out);
void *y_cpu_jit_compile_instrumented_with_options(
    const char *source, const YCpuJitOptions *options, char **error_out);

/* Snapshots a live instrumented handle and returns an independently owned
 * session. Source must match the original lowered IR; training remains usable.
 * Final options are explicit, not recovered from the training handle. */
void *y_cpu_jit_compile_profiled_with_options(
    const char *source, const YCpuJitOptions *options,
    void *training_jit, char **error_out);

/* Existing entrypoints retain inherited training/codegen policy. */
void *y_cpu_jit_compile(const char *source, uint32_t opt_level, char **error_out);
void *y_cpu_jit_compile_instrumented(
    const char *source, uint32_t opt_level, char **error_out);
void *y_cpu_jit_compile_profiled(
    const char *source, uint32_t opt_level, void *training_jit, char **error_out);
void *y_cpu_jit_function(void *jit, const char *name, char **error_out);
char *y_cpu_jit_signature(void *jit, const char *name, char **error_out);
char *y_cpu_jit_branch_profile(void *training_jit, char **error_out);
char *y_cpu_jit_compile_timings(void *jit, char **error_out);
char *y_cpu_jit_optimization_timings(void *jit, char **error_out);
char *y_cpu_jit_materialization_timings(void *jit, char **error_out);

/* Checked scalar/pointer call. Failure preserves result. reserved must be zero.
 * Tags: void=0, I8=1, U8=2, I16=3, U16=4, I32=5, U32=6, I64=7, U64=8,
 * usize=9, bool=10, F32=11, F64=12, pointer=13. Integer bits are low-width
 * two's complement; float bits are IEEE. Pointer storage must remain live. */
int32_t y_cpu_jit_call(
    void *jit, const char *name, const YCpuJitValue *arguments, size_t count,
    YCpuJitValue *result, char **error_out);

#ifdef __cplusplus
}
#endif

#endif /* Y_CPU_JIT_H */
