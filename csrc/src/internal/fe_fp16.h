/* fp16 (IEEE 754 binary16 / S1E5M10) helpers for runtime state storage.
 *
 * Hardware fp16 conversion is GUARANTEED on all deployment ISAs:
 *   x86: the minimum tier is AVX2+FMA3+F16C — VCVTPS2PH / VCVTPH2PS.
 *        1 µop, 5-7c latency, 1/cycle throughput on Haswell+.
 *   ARM: NEON baseline (aarch64) mandates FCVTN / FCVTL.
 *        Available on every aarch64 target (M2, X3, A78, A55).
 *
 * Dimension contract: all engine call sites pass n that is a multiple of
 * the vector width (8 on x86, 4 on ARM). FE_C1=96, FE_F1=128 and
 * FE_FREQ_BINS=512 are 16-aligned; FE_C2=72 and FE_F2=72 are
 * 8-aligned. The misaligned-n scalar fallback helper variants were removed
 * — if a misaligned n ever reaches one of these helpers it's a contract
 * violation; the abort() guard catches it loudly in release builds rather
 * than silently producing wrong output.
 *
 * Bit-id within fp16 path: hardware conversion is deterministic (RNE
 * rounding), so cross-tier byte-id holds for the fp16 path itself —
 * just different output than the fp32 path.
 */
#ifndef FE_FP16_H
#define FE_FP16_H

#include <stdint.h>
#include <stdlib.h>   /* abort */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  include <immintrin.h>
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#  include <arm_neon.h>
#else
#  error "fe_fp16.h requires x86_64 (AVX2/FMA3/F16C) or aarch64 (NEON FCVT)"
#endif

/* Single-line guard for callers: any leftover after the SIMD loop is a
 * dimension contract violation; abort loudly. */
static inline void fe_fp16_alignment_violation(void) { abort(); }

/* ---------------------------------------------------------------- *
 *  Vector SIMD conversion (the only path — no scalar fallback)
 * ---------------------------------------------------------------- */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

/* Store 8 fp32 lanes as 8 fp16 (RNE). */
__attribute__((target("avx2,f16c,fma")))
static inline void fe_fp16_store8(uint16_t *dst, __m256 v) {
    _mm_storeu_si128((__m128i *)dst,
                     _mm256_cvtps_ph(v,
                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

/* Load 8 fp16 lanes and expand to fp32. */
__attribute__((target("avx2,f16c,fma")))
static inline __m256 fe_fp16_load8(const uint16_t *src) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)src));
}

#endif /* x86 */

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)

/* Store 4 fp32 lanes as 4 fp16. */
static inline void fe_fp16_store4(uint16_t *dst, float32x4_t v) {
    float16x4_t h = vcvt_f16_f32(v);
    vst1_u16(dst, vreinterpret_u16_f16(h));
}

/* Load 4 fp16 lanes and expand to fp32. */
static inline float32x4_t fe_fp16_load4(const uint16_t *src) {
    float16x4_t h = vreinterpret_f16_u16(vld1_u16(src));
    return vcvt_f32_f16(h);
}

#endif /* aarch64 */

/* ---------------------------------------------------------------- *
 *  Buffer-level conversions
 *  Caller contract: n must be a multiple of the SIMD width
 *  (8 on x86, 4 on ARM). Violation → abort().
 * ---------------------------------------------------------------- */

/* fp32 [n] -> fp16 [n]. */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
__attribute__((target("avx2,f16c,fma")))
#endif
static inline void fe_fp16_pack_buf(uint16_t *dst, const float *src, int n) {
    int i = 0;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        fe_fp16_store8(dst + i, v);
    }
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        fe_fp16_store4(dst + i, v);
    }
#endif
    if (i != n) fe_fp16_alignment_violation();
}

/* fp16-input min/max scan. Reads fp16 once with on-the-fly cvt; no
 * fp32 materialization. Output matches fe_qg_min_max signature. */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
__attribute__((target("avx2,f16c,fma")))
#endif
static inline void fe_fp16_min_max(const uint16_t *p, int n,
                                    float *min_out, float *max_out) {
    if (n <= 0) { *min_out = 0.0f; *max_out = 0.0f; return; }
    int i = 0;
    float vmin = 0.0f, vmax = 0.0f;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    {
        __m256 v0 = fe_fp16_load8(p);
        __m256 vmn = v0, vmx = v0;
        i = 8;
        for (; i + 7 < n; i += 8) {
            __m256 v = fe_fp16_load8(p + i);
            vmn = _mm256_min_ps(vmn, v);
            vmx = _mm256_max_ps(vmx, v);
        }
        __m128 lo = _mm256_castps256_ps128(vmn);
        __m128 hi = _mm256_extractf128_ps(vmn, 1);
        __m128 mn = _mm_min_ps(lo, hi);
        mn = _mm_min_ps(mn, _mm_movehl_ps(mn, mn));
        mn = _mm_min_ss(mn, _mm_shuffle_ps(mn, mn, 1));
        vmin = _mm_cvtss_f32(mn);
        lo = _mm256_castps256_ps128(vmx);
        hi = _mm256_extractf128_ps(vmx, 1);
        __m128 mx = _mm_max_ps(lo, hi);
        mx = _mm_max_ps(mx, _mm_movehl_ps(mx, mx));
        mx = _mm_max_ss(mx, _mm_shuffle_ps(mx, mx, 1));
        vmax = _mm_cvtss_f32(mx);
    }
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    {
        float32x4_t v0 = fe_fp16_load4(p);
        float32x4_t vmn = v0, vmx = v0;
        i = 4;
        for (; i + 3 < n; i += 4) {
            float32x4_t v = fe_fp16_load4(p + i);
            vmn = vminq_f32(vmn, v);
            vmx = vmaxq_f32(vmx, v);
        }
        vmin = vminvq_f32(vmn);
        vmax = vmaxvq_f32(vmx);
    }
#endif
    if (i != n) fe_fp16_alignment_violation();
    *min_out = vmin;
    *max_out = vmax;
}

/* fp16-input asymmetric quantize. q = clamp(round(x*inv) + zp_off). */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
__attribute__((target("avx2,f16c,fma")))
#endif
static inline void fe_fp16_quantize_asym(const uint16_t *p, int n,
                                          int8_t *out, float inv,
                                          int32_t zp_off) {
    int i = 0;
    const float zof = (float)zp_off;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    const __m256 vinv = _mm256_set1_ps(inv);
    const __m256 vzof = _mm256_set1_ps(zof);
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_fmadd_ps(fe_fp16_load8(p + i), vinv, vzof);
        __m256i q = _mm256_cvtps_epi32(v);
        __m128i lo = _mm256_castsi256_si128(q);
        __m128i hi = _mm256_extracti128_si256(q, 1);
        __m128i s  = _mm_packs_epi32(lo, hi);
        __m128i b  = _mm_packs_epi16(s, s);
        _mm_storel_epi64((__m128i *)(out + i), b);
    }
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    const float32x4_t vinv = vdupq_n_f32(inv);
    const float32x4_t vzof = vdupq_n_f32(zof);
    for (; i + 15 < n; i += 16) {
        float32x4_t v0 = fe_fp16_load4(p + i +  0);
        float32x4_t v1 = fe_fp16_load4(p + i +  4);
        float32x4_t v2 = fe_fp16_load4(p + i +  8);
        float32x4_t v3 = fe_fp16_load4(p + i + 12);
        int32x4_t q0 = vcvtnq_s32_f32(vfmaq_f32(vzof, v0, vinv));
        int32x4_t q1 = vcvtnq_s32_f32(vfmaq_f32(vzof, v1, vinv));
        int32x4_t q2 = vcvtnq_s32_f32(vfmaq_f32(vzof, v2, vinv));
        int32x4_t q3 = vcvtnq_s32_f32(vfmaq_f32(vzof, v3, vinv));
        int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(out + i, vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
    }
    for (; i + 3 < n; i += 4) {
        float32x4_t v = fe_fp16_load4(p + i);
        int32x4_t q = vcvtnq_s32_f32(vfmaq_f32(vzof, v, vinv));
        int16x4_t s = vqmovn_s32(q);
        int8x8_t  b = vqmovn_s16(vcombine_s16(s, vdup_n_s16(0)));
        uint32_t  w = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(b), 0);
        for (int k = 0; k < 4; ++k) out[i + k] = (int8_t)((w >> (k * 8)) & 0xff);
    }
#endif
    if (i != n) fe_fp16_alignment_violation();
}

#endif /* FE_FP16_H */
