/*
 * Multi-ISA helpers for activation quantize, dequant+bias+SiLU
 * +requantize, and max-abs scan. Each function does its hot loop at the
 * widest available SIMD width; tails step down through narrower widths.
 * Including TUs must have pulled in qgemm_arch.h ISA macros.
 */
#include <math.h>
#include <string.h>

/* Capability flags from compile-time tier. AVX2 is the x86 minimum on
 * runtime tiers; no separate 128-bit tier — those ops live under the AVX2 path. ARM uses
 * FE_QGEMM_NEON directly. */
#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
  #define FE_QPOST_AVX512 1
  #define FE_QPOST_AVX2   1
#elif defined(FE_QGEMM_HAVE_AVXVNNI) || defined(FE_QGEMM_HAVE_AVX2)
  #define FE_QPOST_AVX2   1
#endif

/* Native-width sigmoid (RNNoise rational poly). */
#if defined(FE_QPOST_AVX2)
static inline __m256 fe_qg_sigmoid8(__m256 x) {
    __m256 x2  = _mm256_mul_ps(x, x);
    __m256 num = _mm256_fmadd_ps(_mm256_set1_ps(0.00950985f), x2,
                                  _mm256_set1_ps(6.02452230f));
    num = _mm256_fmadd_ps(num, x2, _mm256_set1_ps(238.13200378f));
    num = _mm256_mul_ps(num, x);
    __m256 den = _mm256_fmadd_ps(_mm256_set1_ps(0.74287558f), x2,
                                  _mm256_set1_ps(103.34200287f));
    den = _mm256_fmadd_ps(den, x2, _mm256_set1_ps(952.72399902f));
    __m256 inv = _mm256_rcp_ps(den);
    __m256 r   = _mm256_fmadd_ps(num, inv, _mm256_set1_ps(0.5f));
    return _mm256_max_ps(_mm256_setzero_ps(),
                         _mm256_min_ps(_mm256_set1_ps(1.0f), r));
}
#endif

/* log2 / exp2 polynomials, native widths. */
#if defined(FE_QPOST_AVX2)
static inline __m256 fe_qg_log2f8(__m256 x) {
    __m256i ix = _mm256_castps_si256(x);
    __m256i e  = _mm256_sub_epi32(_mm256_srli_epi32(ix, 23), _mm256_set1_epi32(127));
    __m256i mb = _mm256_or_si256(_mm256_and_si256(ix, _mm256_set1_epi32(0x007FFFFF)),
                                  _mm256_set1_epi32(0x3F800000));
    __m256 f   = _mm256_castsi256_ps(mb);
    __m256 ef  = _mm256_cvtepi32_ps(e);
    __m256 y   = _mm256_sub_ps(f, _mm256_set1_ps(1.0f));
    __m256 p   = _mm256_set1_ps(0.05500914f);
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(-0.20652259f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps( 0.47917641f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(-0.71567517f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps( 1.44256800f));
    return _mm256_add_ps(ef, _mm256_mul_ps(y, p));
}
static inline __m256 fe_qg_exp2f8(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-126.0f)),
                      _mm256_set1_ps(127.0f));
    __m256 fk = _mm256_floor_ps(x);
    __m256i k = _mm256_cvtps_epi32(fk);
    __m256 r  = _mm256_sub_ps(x, fk);
    __m256 p  = _mm256_set1_ps(0.0096180f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.0555041f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.2402265f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.6931472f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i ks = _mm256_slli_epi32(_mm256_add_epi32(k, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(ks));
}
#endif
#if defined(FE_QPOST_AVX512)
static inline __m512 fe_qg_log2f16(__m512 x) {
    __m512i ix = _mm512_castps_si512(x);
    __m512i e  = _mm512_sub_epi32(_mm512_srli_epi32(ix, 23), _mm512_set1_epi32(127));
    __m512i mb = _mm512_or_si512(_mm512_and_si512(ix, _mm512_set1_epi32(0x007FFFFF)),
                                  _mm512_set1_epi32(0x3F800000));
    __m512 f   = _mm512_castsi512_ps(mb);
    __m512 ef  = _mm512_cvtepi32_ps(e);
    __m512 y   = _mm512_sub_ps(f, _mm512_set1_ps(1.0f));
    __m512 p   = _mm512_set1_ps(0.05500914f);
    p = _mm512_fmadd_ps(p, y, _mm512_set1_ps(-0.20652259f));
    p = _mm512_fmadd_ps(p, y, _mm512_set1_ps( 0.47917641f));
    p = _mm512_fmadd_ps(p, y, _mm512_set1_ps(-0.71567517f));
    p = _mm512_fmadd_ps(p, y, _mm512_set1_ps( 1.44256800f));
    return _mm512_add_ps(ef, _mm512_mul_ps(y, p));
}
static inline __m512 fe_qg_exp2f16(__m512 x) {
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-126.0f)),
                      _mm512_set1_ps(127.0f));
    __m512 fk = _mm512_floor_ps(x);
    __m512i k = _mm512_cvtps_epi32(fk);
    __m512 r  = _mm512_sub_ps(x, fk);
    __m512 p  = _mm512_set1_ps(0.0096180f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.0555041f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.2402265f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.6931472f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    __m512i ks = _mm512_slli_epi32(_mm512_add_epi32(k, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(ks));
}
#endif

/* tanh polynomial, native widths. */
#if defined(FE_QPOST_AVX2)
static inline __m256 fe_qg_tanh8(__m256 x) {
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 num = _mm256_fmadd_ps(_mm256_set1_ps(0.60863042f), x2,
                                  _mm256_set1_ps(96.39235687f));
    num = _mm256_fmadd_ps(num, x2, _mm256_set1_ps(952.52801514f));
    num = _mm256_mul_ps(num, x);
    __m256 den = _mm256_fmadd_ps(_mm256_set1_ps(11.88600922f), x2,
                                  _mm256_set1_ps(413.36801147f));
    den = _mm256_fmadd_ps(den, x2, _mm256_set1_ps(952.72399902f));
    __m256 r = _mm256_mul_ps(num, _mm256_rcp_ps(den));
    return _mm256_max_ps(_mm256_set1_ps(-1.0f),
                         _mm256_min_ps(_mm256_set1_ps(1.0f), r));
}
#endif

#if defined(FE_QPOST_AVX512)
/* Cross-tier bit-id fix: the ymm sigmoid8/tanh8 use VRCPPS (`_mm256_rcp_ps`,
 * ~12-bit). AVX-512 VRCP14PS (`_mm512_rcp14_ps`) is 14-bit → DIFFERENT mantissa
 * bits → avx512vnni diverged from avx2/avxvnni (SNR ~44 dB, bit-id=no — a latent
 * bug, never caught without AVX-512 hw/SDE). Reproduce the exact 12-bit VRCPPS
 * by running `_mm256_rcp_ps` on the two 256-bit halves (same instruction, same
 * bits as the ymm path). AVX-512F-only ops (no DQ dependency). */
static inline __m512 fe_avx512_rcp_match256(__m512 d) {
    __m256 lo = _mm256_rcp_ps(_mm512_castps512_ps256(d));
    __m256 hi = _mm256_rcp_ps(_mm256_castsi256_ps(
                    _mm512_extracti64x4_epi64(_mm512_castps_si512(d), 1)));
    __m512i z = _mm512_castsi256_si512(_mm256_castps_si256(lo));
    z = _mm512_inserti64x4(z, _mm256_castps_si256(hi), 1);
    return _mm512_castsi512_ps(z);
}
static inline __m512 fe_qg_tanh16(__m512 x) {
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 num = _mm512_fmadd_ps(_mm512_set1_ps(0.60863042f), x2,
                                  _mm512_set1_ps(96.39235687f));
    num = _mm512_fmadd_ps(num, x2, _mm512_set1_ps(952.52801514f));
    num = _mm512_mul_ps(num, x);
    __m512 den = _mm512_fmadd_ps(_mm512_set1_ps(11.88600922f), x2,
                                  _mm512_set1_ps(413.36801147f));
    den = _mm512_fmadd_ps(den, x2, _mm512_set1_ps(952.72399902f));
    __m512 r = _mm512_mul_ps(num, fe_avx512_rcp_match256(den));
    return _mm512_max_ps(_mm512_set1_ps(-1.0f),
                         _mm512_min_ps(_mm512_set1_ps(1.0f), r));
}
#endif

#if defined(FE_QPOST_AVX512)
static inline __m512 fe_qg_sigmoid16(__m512 x) {
    __m512 x2  = _mm512_mul_ps(x, x);
    __m512 num = _mm512_fmadd_ps(_mm512_set1_ps(0.00950985f), x2,
                                  _mm512_set1_ps(6.02452230f));
    num = _mm512_fmadd_ps(num, x2, _mm512_set1_ps(238.13200378f));
    num = _mm512_mul_ps(num, x);
    __m512 den = _mm512_fmadd_ps(_mm512_set1_ps(0.74287558f), x2,
                                  _mm512_set1_ps(103.34200287f));
    den = _mm512_fmadd_ps(den, x2, _mm512_set1_ps(952.72399902f));
    __m512 inv = fe_avx512_rcp_match256(den);
    __m512 r   = _mm512_fmadd_ps(num, inv, _mm512_set1_ps(0.5f));
    return _mm512_max_ps(_mm512_setzero_ps(),
                         _mm512_min_ps(_mm512_set1_ps(1.0f), r));
}
#endif

/* NEON 4-lane: int32 -> fp32 FMA with combined scale + optional bias
 * and SiLU. */
#if defined(FE_QGEMM_NEON)
#endif

/* max |p[i]| over n elements. */
static inline float fe_qg_max_abs(const float *p, int n) {
    int i = 0;
    float m = 0.0f;

#if defined(FE_QPOST_AVX512)
    /* AVX-512F has no vand_ps (DQ only) -- use integer AND on the cast.
     * 4-way unrolled, 64 fp32/iter. */
    __m512 vm0 = _mm512_setzero_ps(), vm1 = vm0, vm2 = vm0, vm3 = vm0;
    const __m512i sm = _mm512_set1_epi32(0x7FFFFFFF);
    for (; i + 63 < n; i += 64) {
        __m512i r0 = _mm512_castps_si512(_mm512_loadu_ps(p + i +  0));
        __m512i r1 = _mm512_castps_si512(_mm512_loadu_ps(p + i + 16));
        __m512i r2 = _mm512_castps_si512(_mm512_loadu_ps(p + i + 32));
        __m512i r3 = _mm512_castps_si512(_mm512_loadu_ps(p + i + 48));
        vm0 = _mm512_max_ps(vm0, _mm512_castsi512_ps(_mm512_and_si512(r0, sm)));
        vm1 = _mm512_max_ps(vm1, _mm512_castsi512_ps(_mm512_and_si512(r1, sm)));
        vm2 = _mm512_max_ps(vm2, _mm512_castsi512_ps(_mm512_and_si512(r2, sm)));
        vm3 = _mm512_max_ps(vm3, _mm512_castsi512_ps(_mm512_and_si512(r3, sm)));
    }
    vm0 = _mm512_max_ps(_mm512_max_ps(vm0, vm1), _mm512_max_ps(vm2, vm3));
    for (; i + 15 < n; i += 16) {
        __m512i raw = _mm512_castps_si512(_mm512_loadu_ps(p + i));
        __m512  v   = _mm512_castsi512_ps(_mm512_and_si512(raw, sm));
        vm0 = _mm512_max_ps(vm0, v);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)((1u << (n - i)) - 1);
        __m512i raw = _mm512_castps_si512(_mm512_maskz_loadu_ps(mask, p + i));
        __m512  v   = _mm512_castsi512_ps(_mm512_and_si512(raw, sm));
        vm0 = _mm512_mask_max_ps(vm0, mask, vm0, v);
        i = n;
    }
    m = _mm512_reduce_max_ps(vm0);
#elif defined(FE_QPOST_AVX2)
    /* 4-way unrolled AVX2: 32 fp32/iter. */
    __m256 vm0 = _mm256_setzero_ps(), vm1 = vm0, vm2 = vm0, vm3 = vm0;
    const __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (; i + 31 < n; i += 32) {
        vm0 = _mm256_max_ps(vm0, _mm256_and_ps(_mm256_loadu_ps(p + i +  0), signmask));
        vm1 = _mm256_max_ps(vm1, _mm256_and_ps(_mm256_loadu_ps(p + i +  8), signmask));
        vm2 = _mm256_max_ps(vm2, _mm256_and_ps(_mm256_loadu_ps(p + i + 16), signmask));
        vm3 = _mm256_max_ps(vm3, _mm256_and_ps(_mm256_loadu_ps(p + i + 24), signmask));
    }
    vm0 = _mm256_max_ps(_mm256_max_ps(vm0, vm1), _mm256_max_ps(vm2, vm3));
    for (; i + 7 < n; i += 8) {
        vm0 = _mm256_max_ps(vm0, _mm256_and_ps(_mm256_loadu_ps(p + i), signmask));
    }
    /* fold ymm -> xmm */
    __m128 hi = _mm256_extractf128_ps(vm0, 1);
    __m128 lo = _mm256_castps256_ps128(vm0);
    __m128 v128 = _mm_max_ps(lo, hi);
    /* tail 4-wide */
    const __m128 signmask128 = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    for (; i + 3 < n; i += 4) {
        __m128 v = _mm_and_ps(_mm_loadu_ps(p + i), signmask128);
        v128 = _mm_max_ps(v128, v);
    }
    __m128 sh = _mm_movehdup_ps(v128);
    __m128 r  = _mm_max_ps(v128, sh);
    sh = _mm_movehl_ps(sh, r);
    r  = _mm_max_ss(r, sh);
    m  = _mm_cvtss_f32(r);
#elif defined(FE_QGEMM_NEON)
    /* 4-way unrolled NEON: 16 fp32/iter. */
    float32x4_t vm0 = vdupq_n_f32(0), vm1 = vm0, vm2 = vm0, vm3 = vm0;
    for (; i + 15 < n; i += 16) {
        vm0 = vmaxq_f32(vm0, vabsq_f32(vld1q_f32(p + i +  0)));
        vm1 = vmaxq_f32(vm1, vabsq_f32(vld1q_f32(p + i +  4)));
        vm2 = vmaxq_f32(vm2, vabsq_f32(vld1q_f32(p + i +  8)));
        vm3 = vmaxq_f32(vm3, vabsq_f32(vld1q_f32(p + i + 12)));
    }
    vm0 = vmaxq_f32(vmaxq_f32(vm0, vm1), vmaxq_f32(vm2, vm3));
    for (; i + 3 < n; i += 4) {
        vm0 = vmaxq_f32(vm0, vabsq_f32(vld1q_f32(p + i)));
    }
    m = vmaxvq_f32(vm0);
#endif

    /* Scalar tail. */
    for (; i < n; ++i) {
        float a = p[i];
        if (a < 0) a = -a;
        if (a > m) m = a;
    }
    return m;
}

/* Per-tensor min and max over n floats in one SIMD pass. Used by
 * per-frame asymmetric uint8 activation quantization. */
static inline void fe_qg_min_max(const float *p, int n,
                                  float *min_out, float *max_out) {
    int i = 0;
    float vmin = 0.0f, vmax = 0.0f;
    if (n > 0) { vmin = p[0]; vmax = p[0]; }

#if defined(FE_QPOST_AVX512)
    if (n >= 64) {
        /* 4-way unrolled: 64 fp32/iter. */
        __m512 vmn0 = _mm512_set1_ps(p[0]), vmn1 = vmn0, vmn2 = vmn0, vmn3 = vmn0;
        __m512 vmx0 = vmn0, vmx1 = vmn0, vmx2 = vmn0, vmx3 = vmn0;
        for (; i + 63 < n; i += 64) {
            __m512 v0 = _mm512_loadu_ps(p + i +  0);
            __m512 v1 = _mm512_loadu_ps(p + i + 16);
            __m512 v2 = _mm512_loadu_ps(p + i + 32);
            __m512 v3 = _mm512_loadu_ps(p + i + 48);
            vmn0 = _mm512_min_ps(vmn0, v0); vmx0 = _mm512_max_ps(vmx0, v0);
            vmn1 = _mm512_min_ps(vmn1, v1); vmx1 = _mm512_max_ps(vmx1, v1);
            vmn2 = _mm512_min_ps(vmn2, v2); vmx2 = _mm512_max_ps(vmx2, v2);
            vmn3 = _mm512_min_ps(vmn3, v3); vmx3 = _mm512_max_ps(vmx3, v3);
        }
        vmn0 = _mm512_min_ps(_mm512_min_ps(vmn0, vmn1), _mm512_min_ps(vmn2, vmn3));
        vmx0 = _mm512_max_ps(_mm512_max_ps(vmx0, vmx1), _mm512_max_ps(vmx2, vmx3));
        for (; i + 15 < n; i += 16) {
            __m512 v = _mm512_loadu_ps(p + i);
            vmn0 = _mm512_min_ps(vmn0, v);
            vmx0 = _mm512_max_ps(vmx0, v);
        }
        vmin = _mm512_reduce_min_ps(vmn0);
        vmax = _mm512_reduce_max_ps(vmx0);
    } else if (n >= 16) {
        __m512 vmn = _mm512_set1_ps(p[0]);
        __m512 vmx = vmn;
        for (; i + 15 < n; i += 16) {
            __m512 v = _mm512_loadu_ps(p + i);
            vmn = _mm512_min_ps(vmn, v);
            vmx = _mm512_max_ps(vmx, v);
        }
        vmin = _mm512_reduce_min_ps(vmn);
        vmax = _mm512_reduce_max_ps(vmx);
    }
#elif defined(FE_QPOST_AVX2)
    if (n >= 32) {
        /* 4-way unrolled AVX2: 32 fp32/iter. */
        __m256 vmn0 = _mm256_set1_ps(p[0]), vmn1 = vmn0, vmn2 = vmn0, vmn3 = vmn0;
        __m256 vmx0 = vmn0, vmx1 = vmn0, vmx2 = vmn0, vmx3 = vmn0;
        for (; i + 31 < n; i += 32) {
            __m256 v0 = _mm256_loadu_ps(p + i +  0);
            __m256 v1 = _mm256_loadu_ps(p + i +  8);
            __m256 v2 = _mm256_loadu_ps(p + i + 16);
            __m256 v3 = _mm256_loadu_ps(p + i + 24);
            vmn0 = _mm256_min_ps(vmn0, v0); vmx0 = _mm256_max_ps(vmx0, v0);
            vmn1 = _mm256_min_ps(vmn1, v1); vmx1 = _mm256_max_ps(vmx1, v1);
            vmn2 = _mm256_min_ps(vmn2, v2); vmx2 = _mm256_max_ps(vmx2, v2);
            vmn3 = _mm256_min_ps(vmn3, v3); vmx3 = _mm256_max_ps(vmx3, v3);
        }
        __m256 vmn = _mm256_min_ps(_mm256_min_ps(vmn0, vmn1), _mm256_min_ps(vmn2, vmn3));
        __m256 vmx = _mm256_max_ps(_mm256_max_ps(vmx0, vmx1), _mm256_max_ps(vmx2, vmx3));
        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_loadu_ps(p + i);
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
    } else if (n >= 8) {
        __m256 vmn = _mm256_set1_ps(p[0]);
        __m256 vmx = vmn;
        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_loadu_ps(p + i);
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
#elif defined(FE_QGEMM_NEON)
    if (n >= 16) {
        /* 4-way unrolled: 16 fp32/iter. */
        float32x4_t vmn0 = vdupq_n_f32(p[0]), vmn1 = vmn0,
                    vmn2 = vmn0, vmn3 = vmn0;
        float32x4_t vmx0 = vmn0, vmx1 = vmn0,
                    vmx2 = vmn0, vmx3 = vmn0;
        for (; i + 15 < n; i += 16) {
            float32x4_t v0 = vld1q_f32(p + i +  0);
            float32x4_t v1 = vld1q_f32(p + i +  4);
            float32x4_t v2 = vld1q_f32(p + i +  8);
            float32x4_t v3 = vld1q_f32(p + i + 12);
            vmn0 = vminq_f32(vmn0, v0); vmx0 = vmaxq_f32(vmx0, v0);
            vmn1 = vminq_f32(vmn1, v1); vmx1 = vmaxq_f32(vmx1, v1);
            vmn2 = vminq_f32(vmn2, v2); vmx2 = vmaxq_f32(vmx2, v2);
            vmn3 = vminq_f32(vmn3, v3); vmx3 = vmaxq_f32(vmx3, v3);
        }
        /* Reduce 4 chains. */
        vmn0 = vminq_f32(vminq_f32(vmn0, vmn1), vminq_f32(vmn2, vmn3));
        vmx0 = vmaxq_f32(vmaxq_f32(vmx0, vmx1), vmaxq_f32(vmx2, vmx3));
        for (; i + 3 < n; i += 4) {
            float32x4_t v = vld1q_f32(p + i);
            vmn0 = vminq_f32(vmn0, v);
            vmx0 = vmaxq_f32(vmx0, v);
        }
        vmin = vminvq_f32(vmn0);
        vmax = vmaxvq_f32(vmx0);
    } else if (n >= 4) {
        float32x4_t vmn = vdupq_n_f32(p[0]);
        float32x4_t vmx = vmn;
        for (; i + 3 < n; i += 4) {
            float32x4_t v = vld1q_f32(p + i);
            vmn = vminq_f32(vmn, v);
            vmx = vmaxq_f32(vmx, v);
        }
        vmin = vminvq_f32(vmn);
        vmax = vmaxvq_f32(vmx);
    }
#endif

    for (; i < n; ++i) {
        float v = p[i];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    *min_out = vmin;
    *max_out = vmax;
}

/* Quantize n fp32 -> int8 with inv = 1/scale; round-to-nearest, sat. */

/* Fused dequant + optional bias + optional SiLU + requant:
 *   dst[n] = clip(round(silu(src[n]*combined[n] + bias[n]) * inv_out)) */

/* Fused dequant + optional bias + optional SiLU -> fp32. */

/* y[n] = a * x[n] (used to build combined[n] = scale_x * scales_w[n]). */
static inline void fe_qg_scale_vec(const float *x, int n, float a, float *out) {
    int i = 0;
#if defined(FE_QPOST_AVX512)
    __m512 va = _mm512_set1_ps(a);
    for (; i + 63 < n; i += 64) {
        _mm512_storeu_ps(out + i +  0, _mm512_mul_ps(_mm512_loadu_ps(x + i +  0), va));
        _mm512_storeu_ps(out + i + 16, _mm512_mul_ps(_mm512_loadu_ps(x + i + 16), va));
        _mm512_storeu_ps(out + i + 32, _mm512_mul_ps(_mm512_loadu_ps(x + i + 32), va));
        _mm512_storeu_ps(out + i + 48, _mm512_mul_ps(_mm512_loadu_ps(x + i + 48), va));
    }
    for (; i + 15 < n; i += 16)
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), va));
    if (i < n) {
        __mmask16 mask = (__mmask16)((1u << (n - i)) - 1);
        _mm512_mask_storeu_ps(out + i, mask,
            _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, x + i), va));
        i = n;
    }
#elif defined(FE_QPOST_AVX2)
    __m256 va = _mm256_set1_ps(a);
    for (; i + 31 < n; i += 32) {
        _mm256_storeu_ps(out + i +  0, _mm256_mul_ps(_mm256_loadu_ps(x + i +  0), va));
        _mm256_storeu_ps(out + i +  8, _mm256_mul_ps(_mm256_loadu_ps(x + i +  8), va));
        _mm256_storeu_ps(out + i + 16, _mm256_mul_ps(_mm256_loadu_ps(x + i + 16), va));
        _mm256_storeu_ps(out + i + 24, _mm256_mul_ps(_mm256_loadu_ps(x + i + 24), va));
    }
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), va));
#elif defined(FE_QGEMM_NEON)
    float32x4_t va = vdupq_n_f32(a);
    for (; i + 15 < n; i += 16) {
        vst1q_f32(out + i +  0, vmulq_f32(vld1q_f32(x + i +  0), va));
        vst1q_f32(out + i +  4, vmulq_f32(vld1q_f32(x + i +  4), va));
        vst1q_f32(out + i +  8, vmulq_f32(vld1q_f32(x + i +  8), va));
        vst1q_f32(out + i + 12, vmulq_f32(vld1q_f32(x + i + 12), va));
    }
    for (; i + 3 < n; i += 4)
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(x + i), va));
#endif
    for (; i < n; ++i) out[i] = x[i] * a;
}

/* Dequantize n int8 -> fp32 with per-tensor scale. */

/* Rescale int8 -> int8 with ratio = scale_src / scale_dst. */
