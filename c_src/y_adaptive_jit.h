#ifndef Y_ADAPTIVE_JIT_H
#define Y_ADAPTIVE_JIT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Y_ADAPTIVE_JIT_ABI_VERSION 1u
#define Y_ADAPTIVE_JIT_DEFERRED 0u
#define Y_ADAPTIVE_JIT_ON_LAUNCH 1u
#define Y_ADAPTIVE_JIT_DISABLED 2u

/* Opaque, thread-confined runtime. Never copy or free its storage yourself. */
typedef struct YAdaptiveJit YAdaptiveJit;

/* Initialize with y_adaptive_jit_config_init before changing individual fields.
 * ABI version 1 requires an exact struct_size match. cache_dir is an optional
 * UTF-8 C string copied by create_current; NULL disables persistent decisions.
 * Deferred tuning is the default. ON_LAUNCH may block for seconds on a hot
 * shape; DEFERRED performs measurements only in tune_hot_json. Capacities and
 * hot_threshold must be positive, max_candidates must be 2..64, and
 * min_improvement is a finite fractional reduction in [0,1), e.g. 0.05 = 5%. */
typedef struct YAdaptiveJitConfig {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t tuning_policy;
    uint32_t max_cached_shapes;
    uint32_t max_candidates;
    uint32_t max_disk_cache_entries;
    uint64_t hot_threshold;
    double min_improvement;
    const char *cache_dir;
} YAdaptiveJitConfig;

/* Common conventions:
 * - Integer results: 0 = success, -1 = failure. Pointer results: NULL = failure.
 * - error_out may be NULL. Otherwise it must point to writable char * storage;
 *   disjoint from other arguments and handle storage. Every call clears it,
 *   then stores an allocated message on failure. Free any
 *   previous message before reusing that storage. Returned messages and JSON
 *   strings belong to the caller and must be released with y_free_string.
 * - All handle calls must run on the creating OS thread, with the EXACT CUDA
 *   context borrowed at creation current. The host owns that context and must
 *   keep it alive through successful destroy. No context is created, switched,
 *   or destroyed by this API. Never call one handle concurrently or reentrantly.
 * - All non-NULL pointers must be valid for their documented object and lifetime.
 *   Freed or fabricated handles and invalid host/device pointers are undefined
 *   behavior. Device allocation bounds and ownership cannot be verified here.
 * - A caught Rust panic returns an error and poisons the handle. Destroy remains
 *   available; other operations reject the poisoned handle.
 */
void y_free_string(char *string);

int32_t y_adaptive_jit_config_init(
    YAdaptiveJitConfig *config, uint32_t config_size, char **error_out);

/* NULL config selects defaults. An initialized CUDA context must already be
 * current on this thread (for example the host framework's primary context).
 * The context's device must support sm_80 or newer. */
YAdaptiveJit *y_adaptive_jit_create_current(
    const YAdaptiveJitConfig *config, char **error_out);

/* Synchronizes before releasing kernels. NULL is a successful no-op. On
 * failure the handle remains owned by the caller: restore the original thread
 * and context and retry. On success it is invalid and must not be reused. */
int32_t y_adaptive_jit_destroy(YAdaptiveJit *handle, char **error_out);

/* Shapes: 1 <= M <= 16384; N and K are positive multiples of 16 <= 16384.
 * Prepares a kernel without accessing application buffers or counting a launch.
 * Cold compilation and cache eviction can block. */
int32_t y_adaptive_jit_prepare(
    YAdaptiveJit *handle, uint32_t m, uint32_t n, uint32_t k, char **error_out);

/* Enqueues contiguous row-major C[F32] = A[F16] * B[F16] on CUDA's legacy
 * default stream. A/B/C must be nonzero device addresses aligned to 16 bytes
 * in the borrowed context, containing at least M*K / K*N / M*N elements.
 * C must not overlap A or B. All allocations must remain live until completion.
 * The caller must order access with other streams and threads; synchronize
 * producer work before launch and complete this work before consuming or
 * freeing buffers. A successful launch does not mean GPU completion.
 * Shape compilation, eviction, and ON_LAUNCH tuning can make this call block. */
int32_t y_adaptive_jit_launch(
    YAdaptiveJit *handle, uint32_t m, uint32_t n, uint32_t k,
    uint64_t a, uint64_t b, uint64_t c, char **error_out);

/* Waits for outstanding work in the borrowed CUDA context. */
int32_t y_adaptive_jit_synchronize(YAdaptiveJit *handle, char **error_out);

/* JSON for one resident shape. Fields include launches, tier,
 * tuning_attempts, tuning_time_seconds, candidates_measured, baseline_us,
 * selected_us, tuning_error, cache_hit, cache_error, and
 * estimated_break_even_launches. Optional values use JSON null.
 * A shape absent from the resident cache returns NULL and an error.
 * Tier values: baseline, tuned, retained_baseline, tuning_failed,
 * rejected_baseline. Cached times are historical observations. */
char *y_adaptive_jit_stats_json(
    YAdaptiveJit *handle, uint32_t m, uint32_t n, uint32_t k, char **error_out);

/* Returns [[M,N,K], ...], most launches first, then oldest use on ties. */
char *y_adaptive_jit_pending_json(YAdaptiveJit *handle, char **error_out);

/* Synchronously measures up to max_shapes pending shapes on scratch buffers;
 * 0 performs no work. This is a count limit, not a time deadline. Returns
 * [{"shape":[M,N,K],"stats":{...}}, ...]. A per-shape tuning failure appears
 * in its stats; a non-NULL JSON result does not imply every shape tuned well.
 * Numeric baseline rejection prevents further launches for that resident shape.
 */
char *y_adaptive_jit_tune_hot_json(
    YAdaptiveJit *handle, uint32_t max_shapes, char **error_out);

#ifdef __cplusplus
}
#endif

#endif /* Y_ADAPTIVE_JIT_H */
