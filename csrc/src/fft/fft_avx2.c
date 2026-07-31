/* x86 AVX2 FFT TU. Compiled with `-mavx2 -mfma` per-file via CMake.
 *
 * Also exports `fft_x86_radix4_stage` (128-bit, L=4 small-stage helper)
 * that both the AVX2 and AVX-512 wrappers fall back to. Pre-AVX2 x86 tiers
 * were dropped (no FMA3 → cross-tier byte-id break), but the 4-wide
 * radix-4 kernel is still needed for the L<8 stages on AVX2/AVX-512.
 * Hosting it here keeps the AVX2-or-above promise without a separate
 * 128-bit TU.
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include "fft_arch.h"
#include <immintrin.h>

/* ---------------- 128-bit radix-4 helpers ---------------- *
 *  Reused by L=4 stages in the AVX2/AVX-512 inner loops.
 */
static inline void fe_x86_cmul(__m128 a_re, __m128 a_im,
                               __m128 b_re, __m128 b_im,
                               __m128 *out_re, __m128 *out_im) {
    __m128 r = _mm_sub_ps(_mm_mul_ps(a_re, b_re), _mm_mul_ps(a_im, b_im));
    __m128 i = _mm_add_ps(_mm_mul_ps(a_re, b_im), _mm_mul_ps(a_im, b_re));
    *out_re = r;
    *out_im = i;
}

static inline void fe_x86_radix4_butterfly(
        __m128 a0_re, __m128 a0_im,
        __m128 b1_re, __m128 b1_im,
        __m128 b2_re, __m128 b2_im,
        __m128 b3_re, __m128 b3_im,
        __m128 *o0_re, __m128 *o0_im,
        __m128 *o1_re, __m128 *o1_im,
        __m128 *o2_re, __m128 *o2_im,
        __m128 *o3_re, __m128 *o3_im) {
    __m128 t0_re = _mm_add_ps(a0_re, b2_re), t0_im = _mm_add_ps(a0_im, b2_im);
    __m128 t1_re = _mm_sub_ps(a0_re, b2_re), t1_im = _mm_sub_ps(a0_im, b2_im);
    __m128 t2_re = _mm_add_ps(b1_re, b3_re), t2_im = _mm_add_ps(b1_im, b3_im);
    __m128 t3_re = _mm_sub_ps(b1_re, b3_re), t3_im = _mm_sub_ps(b1_im, b3_im);

    *o0_re = _mm_add_ps(t0_re, t2_re); *o0_im = _mm_add_ps(t0_im, t2_im);
    *o1_re = _mm_add_ps(t1_re, t3_im); *o1_im = _mm_sub_ps(t1_im, t3_re);
    *o2_re = _mm_sub_ps(t0_re, t2_re); *o2_im = _mm_sub_ps(t0_im, t2_im);
    *o3_re = _mm_sub_ps(t1_re, t3_im); *o3_im = _mm_add_ps(t1_im, t3_re);
}

/* Generic radix-4 stage (L >= 4) — exported, referenced by both
 * fft_avx2.inl and fft_avx512.inl when L is too small for their wider
 * native kernel. */
void fft_x86_radix4_stage(const float *in_re, const float *in_im,
                          float *out_re, float *out_im,
                          const float *twr, const float *twi,
                          int L) {
    const int Ls = L * 4;
    const int n_groups = FE_FFT_HALF / Ls;
    const int stride = FE_FFT_HALF / 4;
    for (int j = 0; j < n_groups; ++j) {
        for (int k = 0; k < L; k += 4) {
            const int in_base = j * L + k;
            __m128 a0_re = _mm_loadu_ps(in_re + in_base + 0 * stride);
            __m128 a0_im = _mm_loadu_ps(in_im + in_base + 0 * stride);
            __m128 a1_re = _mm_loadu_ps(in_re + in_base + 1 * stride);
            __m128 a1_im = _mm_loadu_ps(in_im + in_base + 1 * stride);
            __m128 a2_re = _mm_loadu_ps(in_re + in_base + 2 * stride);
            __m128 a2_im = _mm_loadu_ps(in_im + in_base + 2 * stride);
            __m128 a3_re = _mm_loadu_ps(in_re + in_base + 3 * stride);
            __m128 a3_im = _mm_loadu_ps(in_im + in_base + 3 * stride);

            __m128 w1_re = _mm_loadu_ps(twr + 0 * L + k);
            __m128 w1_im = _mm_loadu_ps(twi + 0 * L + k);
            __m128 w2_re = _mm_loadu_ps(twr + 1 * L + k);
            __m128 w2_im = _mm_loadu_ps(twi + 1 * L + k);
            __m128 w3_re = _mm_loadu_ps(twr + 2 * L + k);
            __m128 w3_im = _mm_loadu_ps(twi + 2 * L + k);

            __m128 b1_re, b1_im, b2_re, b2_im, b3_re, b3_im;
            fe_x86_cmul(a1_re, a1_im, w1_re, w1_im, &b1_re, &b1_im);
            fe_x86_cmul(a2_re, a2_im, w2_re, w2_im, &b2_re, &b2_im);
            fe_x86_cmul(a3_re, a3_im, w3_re, w3_im, &b3_re, &b3_im);

            __m128 o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
            fe_x86_radix4_butterfly(a0_re, a0_im, b1_re, b1_im, b2_re, b2_im, b3_re, b3_im,
                                     &o0_re, &o0_im, &o1_re, &o1_im,
                                     &o2_re, &o2_im, &o3_re, &o3_im);

            const int out_base = j * Ls + k;
            _mm_storeu_ps(out_re + out_base + 0 * L, o0_re);
            _mm_storeu_ps(out_im + out_base + 0 * L, o0_im);
            _mm_storeu_ps(out_re + out_base + 1 * L, o1_re);
            _mm_storeu_ps(out_im + out_base + 1 * L, o1_im);
            _mm_storeu_ps(out_re + out_base + 2 * L, o2_re);
            _mm_storeu_ps(out_im + out_base + 2 * L, o2_im);
            _mm_storeu_ps(out_re + out_base + 3 * L, o3_re);
            _mm_storeu_ps(out_im + out_base + 3 * L, o3_im);
        }
    }
}

/* ---------------- AVX2 native FFT path ---------------- */
#include "fft_avx2.inl"
#endif
