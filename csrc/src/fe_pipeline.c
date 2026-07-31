// Public API: fe_init / fe_run / fe_free / fe_reset.
#include "fe.h"
#include "fe_internal.h"
#include "qgemm/qgemm_dispatch.h"

#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#  include <windows.h>   /* VirtualLock -- Windows page-pin equivalent */
#else
#  include <sys/mman.h>  /* mlock -- best-effort page lock to reduce peak jitter */
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  include <immintrin.h>
#  include <xmmintrin.h>
#  include <pmmintrin.h>
#endif

// Enable FTZ/DAZ so denormals are flushed to zero. Silence frames here
// produce very small fp values that would otherwise hit the 50-200x
// per-op denormal penalty on x86 (smaller but still measurable on ARM).
// The downstream mag<eps masks already produce exact zero, so flushing
// has no audible effect on signal frames.
static void fe_set_denormal_flush(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#elif defined(__aarch64__) || defined(_M_ARM64)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ULL << 24);   /* FZ:  flush-to-zero            */
    __asm__ volatile("msr fpcr, %0" :: "r"(fpcr));
#elif defined(__arm__)
    uint32_t fpscr;
    __asm__ volatile("vmrs %0, fpscr" : "=r"(fpscr));
    fpscr |= (1U << 24);
    __asm__ volatile("vmsr fpscr, %0" :: "r"(fpscr));
#endif
}

static FeWeights g_w;
static FeState  *g_s = NULL;

/* Page-lock weight + state buffers to reduce peak jitter from soft page
 * faults / swap pressure. Best-effort: mlock failure (RLIMIT_MEMLOCK, no
 * cap, etc.) is silently ignored -- the engine functions identically, just
 * with the original jitter envelope. macOS default RLIMIT_MEMLOCK is
 * effectively unlimited for unprivileged processes. Linux default is 64 KB
 * unless CAP_IPC_LOCK or the calling process is root. */
static void try_mlock(const void *p, size_t n) {
    if (!p || !n) return;
#if defined(_WIN32)
    (void)VirtualLock((LPVOID)p, n);
#else
    (void)mlock(p, n);
#endif
}

static void fe_lock_pages(const void *weights_blob, size_t weights_size) {
    try_mlock(weights_blob,        weights_size);
    try_mlock(g_w.q_buf,           g_w.q_buf_size);
    try_mlock(g_w.q_scales_buf,    g_w.q_scales_size);
    try_mlock(g_w.q_row_sums_buf,  g_w.q_row_sums_size);
    try_mlock(g_w.q_wino_buf,      g_w.q_wino_buf_size);
    try_mlock(g_w.q_wino_scales_buf, g_w.q_wino_scales_size);
    try_mlock(g_w.prepack_buf,     g_w.prepack_size);
    try_mlock(g_s, sizeof(*g_s));
}

int fe_init(const void *weights_blob, int weights_size) {
    if (g_s) return 0;
    if (!weights_blob || weights_size <= 0) return -1;
    fe_set_denormal_flush();  /* FTZ/DAZ -- see comment above */
    if (fe_qgemm_init() != 0) return -1;   // unsupported CPU
    if (fe_load_weights(&g_w, weights_blob, (size_t)weights_size) != 0) {
        fe_free_weights(&g_w);
        return -1;
    }
    fe_weights_finalize_for_tier(&g_w, fe_qgemm_ops.tier);
    g_s = fe_state_create();
    if (!g_s) {
        fe_free_weights(&g_w);
        return -1;
    }

    /* Lock pages before warmup so the warmup itself doesn't generate the
     * soft faults we're trying to avoid in production. */
    fe_lock_pages(weights_blob, (size_t)weights_size);

    // Warmup with silent frames: faults weight pages in, populates L1/L2,
    // resolves runtime branches, and touches tier-specific scratch before
    // the first user frame. Activation quantization stays per-frame dynamic;
    // no activation scale is frozen here.
    float warm_in [FE_FRAME_SIZE];
    float warm_out[FE_FRAME_SIZE];
    memset(warm_in, 0, sizeof(warm_in));
    for (int i = 0; i < 128; ++i)
        fe_process_frame(g_s, &g_w, warm_in, warm_out);
    fe_reset();   /* zero streaming state so warmup doesn't leak           */
    return 0;
}

void fe_run(const float *in, float *out) {
    if (!g_s) return;
    fe_process_frame(g_s, &g_w, in, out);
}

void fe_free(void) {
    if (g_s) { fe_state_destroy(g_s); g_s = NULL; }
    fe_free_weights(&g_w);
}

void fe_reset(void) {
    if (!g_s) return;
    // Zero GRU hidden states + STFT/iSTFT overlap caches.
    for (int b = 0; b < FE_RF_BLOCKS; ++b)
        memset(g_s->gru_h[b], 0, sizeof(g_s->gru_h[b]));
    memset(g_s->cache_stft,  0, sizeof(g_s->cache_stft));
    memset(g_s->cache_istft, 0, sizeof(g_s->cache_istft));
}
