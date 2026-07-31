/*
 * 128-bit SIMD abstraction + inline transcendentals.
 * Backends: ARM NEON, x86 128-bit. No scalar tier (unsupported ISAs fail to compile).
 * -DFE_USE_ACCURATE_POLY swaps these shared 128-bit/NEON aliases for
 * Remez approximations. Wider x86 qgemm helpers define their own fixed
 * native-width approximations in qgemm_simd_post.inl.
 */
#ifndef FE_SIMD_H
#define FE_SIMD_H

#include <stdint.h>
#include <string.h>

/* ARM AdvSIMD. */
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
    #include <arm_neon.h>
    #define FE_SIMD_NEON 1
    typedef float32x4_t fe_f32x4;

    #define fe_load(p)        vld1q_f32((const float *)(p))
    #define fe_store(p, v)    vst1q_f32((float *)(p), (v))
    #define fe_set1(x)        vdupq_n_f32(x)
    #define fe_add(a, b)      vaddq_f32((a), (b))
    #define fe_sub(a, b)      vsubq_f32((a), (b))
    #define fe_mul(a, b)      vmulq_f32((a), (b))
    #define fe_neg(a)         vnegq_f32(a)
    #define fe_fma(a, b, c)   vfmaq_f32((c), (a), (b))   /* a*b + c */

/* x86 (128-bit). */
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    /* immintrin.h is the umbrella x86 intrinsic header: on the hot TUs
     * (-mavx2) it exposes the 256-bit set; at the x86_64 baseline only the
     * 128-bit ops defined below are used. Safe to include unconditionally. */
    #include <immintrin.h>
    #define FE_SIMD_X86 1
    typedef __m128 fe_f32x4;

    #define fe_load(p)        _mm_loadu_ps((const float *)(p))
    #define fe_store(p, v)    _mm_storeu_ps((float *)(p), (v))
    #define fe_set1(x)        _mm_set1_ps(x)
    #define fe_add(a, b)      _mm_add_ps((a), (b))
    #define fe_sub(a, b)      _mm_sub_ps((a), (b))
    #define fe_mul(a, b)      _mm_mul_ps((a), (b))
    #define fe_neg(a)         _mm_sub_ps(_mm_setzero_ps(), (a))
    #define fe_fma(a, b, c)   _mm_add_ps(_mm_mul_ps((a), (b)), (c))

/* No scalar tier: unsupported ISAs do not run (project policy). */
#else
    #error "faster-enhancer.c requires NEON (aarch64) or x86_64 128-bit SIMD; no scalar fallback"
#endif

/* Element-wise min/max/div. */
#if defined(FE_SIMD_NEON)
    #define fe_min(a, b)            vminq_f32((a), (b))
    #define fe_max(a, b)            vmaxq_f32((a), (b))
    #define fe_div(a, b)            vdivq_f32((a), (b))
#elif defined(FE_SIMD_X86)
    #define fe_min(a, b)            _mm_min_ps((a), (b))
    #define fe_max(a, b)            _mm_max_ps((a), (b))
    #define fe_div(a, b)            _mm_div_ps((a), (b))
#endif

/*
 * Polynomial log2 / exp2 for STFT power compression.
 *   log2(x): split exponent + mantissa, minimax poly on log2(1+y).
 *   exp2(x): split into integer k + r, poly for 2^r, 2^k bit-build.
 */
static inline fe_f32x4 fe_log2f4_fast(fe_f32x4 x) {
#if defined(FE_SIMD_NEON)
    int32x4_t ix = vreinterpretq_s32_f32(x);
    int32x4_t e  = vsubq_s32(vshrq_n_s32(ix, 23), vdupq_n_s32(127));
    int32x4_t mb = vorrq_s32(vandq_s32(ix, vdupq_n_s32(0x007FFFFF)),
                              vdupq_n_s32(0x3F800000));
    fe_f32x4 f   = vreinterpretq_f32_s32(mb);
    fe_f32x4 ef  = vcvtq_f32_s32(e);
#elif defined(FE_SIMD_X86)
    __m128i ix = _mm_castps_si128(x);
    __m128i e  = _mm_sub_epi32(_mm_srli_epi32(ix, 23), _mm_set1_epi32(127));
    __m128i mb = _mm_or_si128(_mm_and_si128(ix, _mm_set1_epi32(0x007FFFFF)),
                               _mm_set1_epi32(0x3F800000));
    fe_f32x4 f  = _mm_castsi128_ps(mb);
    fe_f32x4 ef = _mm_cvtepi32_ps(e);
#endif
    fe_f32x4 y = fe_sub(f, fe_set1(1.0f));
    /* Minimax poly for log2(1+y), y in [0,1). */
    fe_f32x4 p = fe_set1(0.05500914f);
    p = fe_fma(p, y, fe_set1(-0.20652259f));
    p = fe_fma(p, y, fe_set1( 0.47917641f));
    p = fe_fma(p, y, fe_set1(-0.71567517f));
    p = fe_fma(p, y, fe_set1( 1.44256800f));
    return fe_add(ef, fe_mul(y, p));
}

static inline fe_f32x4 fe_exp2f4_fast(fe_f32x4 x) {
    /* Clamp to safe fp32 exponent range. */
    x = fe_min(fe_max(x, fe_set1(-126.0f)), fe_set1(127.0f));
#if defined(FE_SIMD_NEON)
    /* round-down to integer k */
    fe_f32x4 fk = vrndmq_f32(x);
    int32x4_t k = vcvtq_s32_f32(fk);
#elif defined(FE_SIMD_X86)
    fe_f32x4 fk = _mm_cvtepi32_ps(_mm_cvttps_epi32(fe_sub(x, fe_set1(0.4999999f))));
    __m128i k   = _mm_cvtps_epi32(fk);
#endif
    fe_f32x4 r = fe_sub(x, fk);
    /* 5-term poly for 2^r, r in [0,1). */
    fe_f32x4 p = fe_set1(0.0096180f);
    p = fe_fma(p, r, fe_set1(0.0555041f));
    p = fe_fma(p, r, fe_set1(0.2402265f));
    p = fe_fma(p, r, fe_set1(0.6931472f));
    p = fe_fma(p, r, fe_set1(1.0f));
#if defined(FE_SIMD_NEON)
    int32x4_t ks = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
    fe_f32x4 pow2k = vreinterpretq_f32_s32(ks);
#elif defined(FE_SIMD_X86)
    __m128i ks = _mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23);
    fe_f32x4 pow2k = _mm_castsi128_ps(ks);
#endif
    return fe_mul(p, pow2k);
}

/* Horizontal sum / max. */
#if defined(FE_SIMD_NEON)
    #define fe_hsum(v)        vaddvq_f32(v)
    #define fe_hmax(v)        vmaxvq_f32(v)
#elif defined(FE_SIMD_X86)
    /* Horizontal sum / max using only baseline 128-bit ops (no FMA, no
     * 256-bit), so it compiles on every supported x86 TU including the
     * baseline ones; _mm_shuffle_ps keeps it portable. */
    static inline float fe_hsum(fe_f32x4 v) {
        fe_f32x4 sh = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        fe_f32x4 sm = _mm_add_ps(v, sh);
        sh = _mm_movehl_ps(sh, sm);
        return _mm_cvtss_f32(_mm_add_ss(sm, sh));
    }
    static inline float fe_hmax(fe_f32x4 v) {
        fe_f32x4 sh = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        fe_f32x4 sm = _mm_max_ps(v, sh);
        sh = _mm_movehl_ps(sh, sm);
        return _mm_cvtss_f32(_mm_max_ss(sm, sh));
    }
#endif

/* Masking helpers for branchless conditionals. */
#if defined(FE_SIMD_NEON)
    static inline fe_f32x4 fe_ge_mask(fe_f32x4 a, fe_f32x4 b) {
        return vreinterpretq_f32_u32(vcgeq_f32(a, b));
    }
    static inline fe_f32x4 fe_and_mask(fe_f32x4 mask, fe_f32x4 x) {
        return vreinterpretq_f32_u32(
            vandq_u32(vreinterpretq_u32_f32(mask), vreinterpretq_u32_f32(x)));
    }
#elif defined(FE_SIMD_X86)
    #define fe_ge_mask(a, b)        _mm_cmpge_ps((a), (b))
    #define fe_and_mask(mask, x)    _mm_and_ps((mask), (x))
#endif

/*
 * RNNoise rational-poly tanh / sigmoid:
 *   tanh(x) = clamp(N(x^2)*x / D(x^2), -1, 1)
 *   sigmoid(x) = clamp(0.5 + N'(x^2)*x / D(x^2), 0, 1)
 * 5 FMAs + reciprocal-estimate per 4 lanes (no fp-divide).
 */
#if defined(FE_SIMD_NEON)
    #define fe_recip_est(x)   vrecpeq_f32(x)
#elif defined(FE_SIMD_X86)
    #define fe_recip_est(x)   _mm_rcp_ps(x)
#endif

static inline fe_f32x4 fe_tanhf4_fast(fe_f32x4 x) {
    fe_f32x4 x2 = fe_mul(x, x);
    fe_f32x4 num = fe_fma(fe_set1(0.60863042f), x2, fe_set1(96.39235687f));
    num = fe_fma(num, x2, fe_set1(952.52801514f));
    num = fe_mul(num, x);
    fe_f32x4 den = fe_fma(fe_set1(11.88600922f), x2, fe_set1(413.36801147f));
    den = fe_fma(den, x2, fe_set1(952.72399902f));
    fe_f32x4 r = fe_mul(num, fe_recip_est(den));
    return fe_max(fe_set1(-1.0f), fe_min(fe_set1(1.0f), r));
}

static inline fe_f32x4 fe_sigmoidf4_fast(fe_f32x4 x) {
    fe_f32x4 x2 = fe_mul(x, x);
    fe_f32x4 num = fe_fma(fe_set1(0.00950985f), x2, fe_set1(6.02452230f));
    num = fe_fma(num, x2, fe_set1(238.13200378f));
    num = fe_mul(num, x);
    fe_f32x4 den = fe_fma(fe_set1(0.74287558f), x2, fe_set1(103.34200287f));
    den = fe_fma(den, x2, fe_set1(952.72399902f));
    fe_f32x4 r = fe_fma(num, fe_recip_est(den), fe_set1(0.5f));
    return fe_max(fe_set1(0.0f), fe_min(fe_set1(1.0f), r));
}

/*
 * Higher-precision polynomial transcendentals.
 * Range reduction + Remez poly + Schraudolph 2^k bit-build (Sleef-style).
 * Activated by -DFE_USE_ACCURATE_POLY.
 */

/* exp(x): 7-term Remez on x in [-87.336, 88.722]. */
static inline fe_f32x4 fe_expf4_accurate(fe_f32x4 x) {
    x = fe_min(fe_max(x, fe_set1(-87.336f)), fe_set1(88.722f));
    fe_f32x4 t = fe_mul(x, fe_set1(1.4426950408889634f));
#if defined(FE_SIMD_NEON)
    int32x4_t k = vcvtnq_s32_f32(t);
    fe_f32x4 kf = vcvtq_f32_s32(k);
#elif defined(FE_SIMD_X86)
    __m128i k = _mm_cvtps_epi32(t);
    fe_f32x4 kf = _mm_cvtepi32_ps(k);
#endif
    /* Cody-Waite split of ln2 for accurate range reduction. */
    fe_f32x4 r = fe_sub(x, fe_mul(kf, fe_set1(0.693145751953125f)));
    r = fe_sub(r, fe_mul(kf, fe_set1(1.428606820e-6f)));
    /* 7-term Remez for exp(r), |r| <= ln2/2. */
    fe_f32x4 p = fe_set1(0.000198757250f);
    p = fe_fma(p, r, fe_set1(0.001391086980f));
    p = fe_fma(p, r, fe_set1(0.008333079190f));
    p = fe_fma(p, r, fe_set1(0.041666559130f));
    p = fe_fma(p, r, fe_set1(0.166666552010f));
    p = fe_fma(p, r, fe_set1(0.500000000000f));
    p = fe_fma(p, r, fe_set1(1.000000000000f));
    p = fe_fma(p, r, fe_set1(1.000000000000f));
#if defined(FE_SIMD_NEON)
    int32x4_t ks = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
    return fe_mul(p, vreinterpretq_f32_s32(ks));
#elif defined(FE_SIMD_X86)
    __m128i ks = _mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23);
    return fe_mul(p, _mm_castsi128_ps(ks));
#endif
}

/* tanh(x) = (e^2x - 1) / (e^2x + 1). */
static inline fe_f32x4 fe_tanhf4_accurate(fe_f32x4 x) {
    fe_f32x4 lim = fe_set1(8.6f);
    x = fe_min(fe_max(x, fe_neg(lim)), lim);
    fe_f32x4 e2x = fe_expf4_accurate(fe_mul(x, fe_set1(2.0f)));
    return fe_div(fe_sub(e2x, fe_set1(1.0f)),
                  fe_add(e2x, fe_set1(1.0f)));
}

/* sigmoid(x) = 1 / (1 + e^-x). */
static inline fe_f32x4 fe_sigmoidf4_accurate(fe_f32x4 x) {
    fe_f32x4 lim = fe_set1(16.6f);
    x = fe_min(fe_max(x, fe_neg(lim)), lim);
    fe_f32x4 e = fe_expf4_accurate(fe_neg(x));
    return fe_div(fe_set1(1.0f), fe_add(fe_set1(1.0f), e));
}

/* log2(x): exponent split + 6-term Remez on log2(1+y). */
static inline fe_f32x4 fe_log2f4_accurate(fe_f32x4 x) {
#if defined(FE_SIMD_NEON)
    int32x4_t ix = vreinterpretq_s32_f32(x);
    int32x4_t e  = vsubq_s32(vshrq_n_s32(ix, 23), vdupq_n_s32(127));
    int32x4_t mb = vorrq_s32(vandq_s32(ix, vdupq_n_s32(0x007FFFFF)),
                              vdupq_n_s32(0x3F800000));
    fe_f32x4 f   = vreinterpretq_f32_s32(mb);
    fe_f32x4 ef  = vcvtq_f32_s32(e);
#elif defined(FE_SIMD_X86)
    __m128i ix = _mm_castps_si128(x);
    __m128i e  = _mm_sub_epi32(_mm_srli_epi32(ix, 23), _mm_set1_epi32(127));
    __m128i mb = _mm_or_si128(_mm_and_si128(ix, _mm_set1_epi32(0x007FFFFF)),
                               _mm_set1_epi32(0x3F800000));
    fe_f32x4 f  = _mm_castsi128_ps(mb);
    fe_f32x4 ef = _mm_cvtepi32_ps(e);
#endif
    fe_f32x4 y = fe_sub(f, fe_set1(1.0f));
    /* 6-term Remez for log2(1+y), y in [0,1). */
    fe_f32x4 p = fe_set1( 0.035410893707f);
    p = fe_fma(p, y, fe_set1(-0.142251601f));
    p = fe_fma(p, y, fe_set1( 0.287258148f));
    p = fe_fma(p, y, fe_set1(-0.428774504f));
    p = fe_fma(p, y, fe_set1( 0.576093800f));
    p = fe_fma(p, y, fe_set1(-0.721347524f));
    p = fe_fma(p, y, fe_set1( 1.442694925f));
    return fe_add(ef, fe_mul(y, p));
}

/* exp2(x) = 2^k * 2^r with 7-term Remez. */
static inline fe_f32x4 fe_exp2f4_accurate(fe_f32x4 x) {
    x = fe_min(fe_max(x, fe_set1(-126.0f)), fe_set1(127.0f));
#if defined(FE_SIMD_NEON)
    fe_f32x4 fk = vrndmq_f32(x);
    int32x4_t k = vcvtq_s32_f32(fk);
#elif defined(FE_SIMD_X86)
    fe_f32x4 fk = _mm_cvtepi32_ps(_mm_cvttps_epi32(fe_sub(x, fe_set1(0.4999999f))));
    __m128i k   = _mm_cvtps_epi32(fk);
#endif
    fe_f32x4 r = fe_sub(x, fk);
    /* 7-term Remez for 2^r, r in [0,1). */
    fe_f32x4 p = fe_set1(0.000154538f);
    p = fe_fma(p, r, fe_set1(0.001339892f));
    p = fe_fma(p, r, fe_set1(0.009618017f));
    p = fe_fma(p, r, fe_set1(0.055504109f));
    p = fe_fma(p, r, fe_set1(0.240226507f));
    p = fe_fma(p, r, fe_set1(0.693147181f));
    p = fe_fma(p, r, fe_set1(1.000000000f));
#if defined(FE_SIMD_NEON)
    int32x4_t ks = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
    return fe_mul(p, vreinterpretq_f32_s32(ks));
#elif defined(FE_SIMD_X86)
    __m128i ks = _mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23);
    return fe_mul(p, _mm_castsi128_ps(ks));
#endif
}

/* Schraudolph 1999 fast exp (bit-hack):
 *   exp(x) ~ reinterpret((int)(12102203*x + 1065054451), float)
 * One FMA + cvt + reinterpret per 4 lanes. */
static inline fe_f32x4 fe_expf4_schraudolph(fe_f32x4 x) {
    /* Clamp to keep the int cast in range. */
    x = fe_min(fe_max(x, fe_set1(-86.0f)), fe_set1(86.0f));
    fe_f32x4 a = fe_set1(12102203.0f);
    fe_f32x4 b = fe_set1(1065054451.0f);
#if defined(FE_SIMD_NEON)
    int32x4_t y_int = vcvtq_s32_f32(vfmaq_f32(b, x, a));
    return vreinterpretq_f32_s32(y_int);
#elif defined(FE_SIMD_X86)
    __m128 t = _mm_add_ps(_mm_mul_ps(x, a), b);
    return _mm_castsi128_ps(_mm_cvtps_epi32(t));
#endif
}

/* Active alias -- fast or accurate per FE_USE_ACCURATE_POLY. */
#ifdef FE_USE_ACCURATE_POLY
    #define FE_TANHF4(x)    fe_tanhf4_accurate(x)
    #define FE_SIGMOIDF4(x) fe_sigmoidf4_accurate(x)
    #define FE_LOG2F4(x)    fe_log2f4_accurate(x)
    #define FE_EXP2F4(x)    fe_exp2f4_accurate(x)
#else
    #define FE_TANHF4(x)    fe_tanhf4_fast(x)
    #define FE_SIGMOIDF4(x) fe_sigmoidf4_fast(x)
    #define FE_LOG2F4(x)    fe_log2f4_fast(x)
    #define FE_EXP2F4(x)    fe_exp2f4_fast(x)
#endif

#endif /* FE_SIMD_H */
