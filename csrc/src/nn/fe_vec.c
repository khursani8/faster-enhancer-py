/* Elementwise vector add. x86 paths use wider ymm/zmm explicitly; the
 * 128b fe_simd.h intrinsics don't auto-widen. */
#include "fe_internal.h"
#include "fe_simd.h"
#include "qgemm/qgemm_arch.h"

/* Two-tier layout:
 *   Tier 1 -- a wide zmm/ymm pass (x86 only) chews through the bulk.
 *   Tier 2 -- a 128-bit fe_simd pass, 4x-unrolled for ILP / dual-issue.
 * Both exist on purpose: the wide path covers x86, while the unrolled 128-bit
 * path is the portable fallback (on NEON there is no wider-than-128 vector, so
 * the wide #if is empty and this becomes the main loop). The 4x unroll keeps
 * several independent load/op/store chains in flight to hide load-use latency.
 * A 4-wide pass then a scalar tail mop up the remainder. */
void fe_vec_add(float *dst, const float *src, int n) {
    int i = 0;
#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
    for (; i + 15 < n; i += 16) {
        __m512 d = _mm512_loadu_ps(dst + i);
        __m512 s = _mm512_loadu_ps(src + i);
        _mm512_storeu_ps(dst + i, _mm512_add_ps(d, s));
    }
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
    for (; i + 7 < n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        __m256 s = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(d, s));
    }
#endif
    for (; i + 15 < n; i += 16) {
        fe_store(dst + i + 0,  fe_add(fe_load(dst + i + 0),  fe_load(src + i + 0)));
        fe_store(dst + i + 4,  fe_add(fe_load(dst + i + 4),  fe_load(src + i + 4)));
        fe_store(dst + i + 8,  fe_add(fe_load(dst + i + 8),  fe_load(src + i + 8)));
        fe_store(dst + i + 12, fe_add(fe_load(dst + i + 12), fe_load(src + i + 12)));
    }
    for (; i + 3 < n; i += 4) {
        fe_store(dst + i, fe_add(fe_load(dst + i), fe_load(src + i)));
    }
    for (; i < n; ++i) dst[i] += src[i];
}
