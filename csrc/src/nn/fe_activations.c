/* SiLU and softmax variants. Vectorized via fe_simd.h with polynomial
 * sigmoid/exp approximations. */
#include "fe_internal.h"
#include "fe_simd.h"
#include "fe_qgemm.h"
#include "fe_fp16.h"
#include "qgemm/qgemm_arch.h"
#include "qgemm/qgemm_simd_post.inl"
#include <math.h>
#include <string.h>

#if defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
/* 8-lane horizontal sum: fold hi+lo to 128-bit, then movehdup/movehl reduce.
 * The softmax row-sum used this exact inline sequence in three places — sharing
 * it preserves the same fixed reduction order, so it stays bit-id-neutral. */
static inline float fe_hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 v128 = _mm_add_ps(lo, hi);
    __m128 sh  = _mm_movehdup_ps(v128);
    __m128 r1  = _mm_add_ps(v128, sh);
    __m128 r2  = _mm_movehl_ps(sh, r1);
    return _mm_cvtss_f32(_mm_add_ss(r1, r2));
}
#endif

/* Fused SiLU + fp16 skip-write. buf stays fp32 (in-place SiLU on the
 * working ping-pong buffer); skip_out is the fp16-storage enc_skip slot.
 * Per-tier SIMD: x86 256-bit VCVTPS2PH, ARM Q-form FCVTN. Same throughput
 * as fp32 stores plus one conversion µop. */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
__attribute__((target("avx2,f16c,fma")))
#endif
void fe_silu_skip_fp16(float *buf, int n, uint16_t *skip_out) {
    int i = 0;
#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
    for (; i + 15 < n; i += 16) {
        __m512 v    = _mm512_loadu_ps(buf + i);
        __m512 s    = fe_qg_sigmoid16(v);
        __m512 silu = _mm512_mul_ps(v, s);
        _mm512_storeu_ps(buf + i, silu);
        /* AVX-512 lacks fp16 store on 512-bit form; split to 2x 256-bit. */
        __m256 lo = _mm512_castps512_ps256(silu);
        __m256 hi = _mm512_extractf32x8_ps(silu, 1);
        fe_fp16_store8(skip_out + i + 0, lo);
        fe_fp16_store8(skip_out + i + 8, hi);
    }
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
    for (; i + 7 < n; i += 8) {
        __m256 v    = _mm256_loadu_ps(buf + i);
        __m256 s    = fe_qg_sigmoid8(v);
        __m256 silu = _mm256_mul_ps(v, s);
        _mm256_storeu_ps(buf + i, silu);
        fe_fp16_store8(skip_out + i, silu);
    }
#elif defined(FE_QGEMM_NEON)
    /* NEON 4-way unrolled: 16 elements/iter via 4 independent SiLU chains. */
    for (; i + 15 < n; i += 16) {
        fe_f32x4 v0 = fe_load(buf + i +  0);
        fe_f32x4 v1 = fe_load(buf + i +  4);
        fe_f32x4 v2 = fe_load(buf + i +  8);
        fe_f32x4 v3 = fe_load(buf + i + 12);
        fe_f32x4 s0 = FE_SIGMOIDF4(v0);
        fe_f32x4 s1 = FE_SIGMOIDF4(v1);
        fe_f32x4 s2 = FE_SIGMOIDF4(v2);
        fe_f32x4 s3 = FE_SIGMOIDF4(v3);
        fe_f32x4 si0 = fe_mul(v0, s0);
        fe_f32x4 si1 = fe_mul(v1, s1);
        fe_f32x4 si2 = fe_mul(v2, s2);
        fe_f32x4 si3 = fe_mul(v3, s3);
        fe_store(buf + i +  0, si0);
        fe_store(buf + i +  4, si1);
        fe_store(buf + i +  8, si2);
        fe_store(buf + i + 12, si3);
        fe_fp16_store4(skip_out + i +  0, si0);
        fe_fp16_store4(skip_out + i +  4, si1);
        fe_fp16_store4(skip_out + i +  8, si2);
        fe_fp16_store4(skip_out + i + 12, si3);
    }
    for (; i + 3 < n; i += 4) {
        fe_f32x4 v    = fe_load(buf + i);
        fe_f32x4 s    = FE_SIGMOIDF4(v);
        fe_f32x4 silu = fe_mul(v, s);
        fe_store(buf + i, silu);
        fe_fp16_store4(skip_out + i, silu);
    }
#endif
    /* Dimension contract: n is a SIMD multiple (engine sites pass
     * FE_C1*FE_F1 = 12288, which is 16-aligned). Misalignment → abort. */
    if (i != n) fe_fp16_alignment_violation();
}

/* Schraudolph 1999 fast exp: reinterpret a linear fp32 fit as the
 * exponent+mantissa bits of expf(x). Single FMA + cvt.
 *
 * Domain guard: the fit is y = x*A + B with A = 12102203, B = 1064866805.
 * Outside [-B/A, (INT32_MAX-B)/A] = [-88.0, +89.4] the int32 cast leaves the
 * representable range and the reinterpret produces garbage rather than a
 * saturating 0/inf -- x < -88 yields a large *negative* float, and x > +89.4
 * yields INT_MIN -> -0.0f. Either poisons the softmax row sum. Softmax here
 * does not subtract the row max (logits are bounded by qkv_scale^2/sqrt(HD)
 * in practice), so the guard is what makes that safe rather than lucky.
 * Clamping to [-87, 88] keeps both ends inside the valid domain; within it
 * the clamp is a no-op, so tier output is unchanged. The ARM 128-bit path
 * (fe_expf4_schraudolph) carries the equivalent guard. */
#define FE_SCHED_EXP_LO (-87.0f)
#define FE_SCHED_EXP_HI ( 88.0f)

#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
static inline __m512 fe_sched_expf16(__m512 x) {
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(FE_SCHED_EXP_LO)),
                                       _mm512_set1_ps(FE_SCHED_EXP_HI));
    __m512 y = _mm512_fmadd_ps(x, _mm512_set1_ps(12102203.0f),
                                  _mm512_set1_ps(1064866805.0f));
    return _mm512_castsi512_ps(_mm512_cvtps_epi32(y));
}
#endif
#if defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI) || defined(FE_QGEMM_HAVE_AVX512_VNNI)
static inline __m256 fe_sched_expf8(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(FE_SCHED_EXP_LO)),
                                       _mm256_set1_ps(FE_SCHED_EXP_HI));
    __m256 y = _mm256_fmadd_ps(x, _mm256_set1_ps(12102203.0f),
                                  _mm256_set1_ps(1064866805.0f));
    return _mm256_castsi256_ps(_mm256_cvtps_epi32(y));
}
#endif

/*
 * Fused dequant + softmax + i8 quantise. Reads int32 c32 from a
 * gemm_int32 call, applies scalar combined_scale (uniform across N for
 * attention Q@K^T), softmaxes per row, writes int8 output.
 * scratch_fp32 buffers one row of exp() values between the two passes.
 */
__attribute__((hot))
void fe_softmax_rows_quant_from_int32(const int32_t *c32, int rows, int cols,
                                       float combined_scale,
                                       float *scratch_fp32,
                                       int8_t *out_q, float inv_scale) {
    /* scratch_fp32 holds one row's exp() values; recycled per row. */
    for (int r = 0; r < rows; ++r) {
        const int32_t *src = c32 + (size_t)r * cols;
        float *exp_row = scratch_fp32;
        int8_t *qrow = out_q + (size_t)r * cols;
        int c = 0;
        float sum;

#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
        const __m512 vcs16 = _mm512_set1_ps(combined_scale);
        __m512 vsum16 = _mm512_setzero_ps();
        for (; c + 15 < cols; c += 16) {
            __m512i vi = _mm512_loadu_si512((const __m512i *)(src + c));
            __m512  vf = _mm512_mul_ps(_mm512_cvtepi32_ps(vi), vcs16);
            __m512  e  = fe_sched_expf16(vf);
            _mm512_storeu_ps(exp_row + c, e);
            vsum16 = _mm512_add_ps(vsum16, e);
        }
        sum = _mm512_reduce_add_ps(vsum16);
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
        const __m256 vcs8 = _mm256_set1_ps(combined_scale);
        __m256 vsum8 = _mm256_setzero_ps();
        for (; c + 7 < cols; c += 8) {
            __m256i vi = _mm256_loadu_si256((const __m256i *)(src + c));
            __m256  vf = _mm256_mul_ps(_mm256_cvtepi32_ps(vi), vcs8);
            __m256  e  = fe_sched_expf8(vf);
            _mm256_storeu_ps(exp_row + c, e);
            vsum8 = _mm256_add_ps(vsum8, e);
        }
        sum = fe_hsum256(vsum8);
#else
        sum = 0.0f;
#endif

        fe_f32x4 vsum0 = fe_set1(0.0f), vsum1 = vsum0,
                 vsum2 = vsum0, vsum3 = vsum0;
        /* 4-way unrolled NEON: 16 cols/iter with 4 independent exp+sum
         * chains. Reassociating the partial sums is ULP-level, well
         * below the int8 output's effective resolution. */
#if defined(FE_QGEMM_NEON)
        fe_f32x4 vcs = fe_set1(combined_scale);
        for (; c + 15 < cols; c += 16) {
            fe_f32x4 vf0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + c +  0)), vcs);
            fe_f32x4 vf1 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + c +  4)), vcs);
            fe_f32x4 vf2 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + c +  8)), vcs);
            fe_f32x4 vf3 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + c + 12)), vcs);
            fe_f32x4 e0 = fe_expf4_schraudolph(vf0);
            fe_f32x4 e1 = fe_expf4_schraudolph(vf1);
            fe_f32x4 e2 = fe_expf4_schraudolph(vf2);
            fe_f32x4 e3 = fe_expf4_schraudolph(vf3);
            fe_store(exp_row + c +  0, e0);
            fe_store(exp_row + c +  4, e1);
            fe_store(exp_row + c +  8, e2);
            fe_store(exp_row + c + 12, e3);
            vsum0 = fe_add(vsum0, e0);
            vsum1 = fe_add(vsum1, e1);
            vsum2 = fe_add(vsum2, e2);
            vsum3 = fe_add(vsum3, e3);
        }
#endif
        /* Fixed reduction order across tiers: (s0+s1) + (s2+s3). */
        vsum0 = fe_add(fe_add(vsum0, vsum1), fe_add(vsum2, vsum3));
        for (; c + 3 < cols; c += 4) {
#if defined(FE_QGEMM_NEON)
            int32x4_t vi = vld1q_s32(src + c);
            fe_f32x4 vf = vmulq_f32(vcvtq_f32_s32(vi), vcs);
#else
            fe_f32x4 vf;
            for (int k = 0; k < 4; ++k)
                ((float *)&vf)[k] = (float)src[c + k] * combined_scale;
#endif
            fe_f32x4 e = fe_expf4_schraudolph(vf);
            fe_store(exp_row + c, e);
            vsum0 = fe_add(vsum0, e);
        }
        sum += fe_hsum(vsum0);
        for (; c < cols; ++c) {
            float v = (float)src[c] * combined_scale;
            float e = expf(v);
            exp_row[c] = e;
            sum += e;
        }

        /* Pass 2: normalize + quantize (vrecpe + Newton on NEON). */
#if defined(FE_QGEMM_NEON)
        float inv;
        {
            float32x2_t vs = vdup_n_f32(sum);
            float32x2_t rr = vrecpe_f32(vs);
            rr = vmul_f32(vrecps_f32(vs, rr), rr);
            inv = vget_lane_f32(rr, 0);
        }
#else
        const float inv = 1.0f / sum;
#endif
        const float combined = inv * inv_scale;
        c = 0;
#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
        {
            __m512 vcomb16 = _mm512_set1_ps(combined);
            for (; c + 15 < cols; c += 16) {
                __m512 e     = _mm512_loadu_ps(exp_row + c);
                __m512i qi   = _mm512_cvtps_epi32(_mm512_mul_ps(e, vcomb16));
                __m128i b    = _mm512_cvtsepi32_epi8(qi);
                _mm_storeu_si128((__m128i *)(qrow + c), b);
            }
        }
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
        {
            __m256 vcomb8 = _mm256_set1_ps(combined);
            for (; c + 7 < cols; c += 8) {
                __m256 e   = _mm256_loadu_ps(exp_row + c);
                __m256i q  = _mm256_cvtps_epi32(_mm256_mul_ps(e, vcomb8));
                __m128i lo = _mm256_castsi256_si128(q);
                __m128i hi = _mm256_extracti128_si256(q, 1);
                __m128i s16 = _mm_packs_epi32(lo, hi);
                __m128i s8  = _mm_packs_epi16(s16, s16);
                _mm_storel_epi64((__m128i *)(qrow + c), s8);
            }
        }
#endif
        {
            fe_f32x4 vcomb4 = fe_set1(combined);
#if defined(FE_QGEMM_NEON)
            /* 4-way unrolled NEON: 16 lanes/iter to a single 16-byte
             * int8 store via vqmovn -> vcombine -> vqmovn. */
            for (; c + 15 < cols; c += 16) {
                fe_f32x4 e0 = fe_load(exp_row + c +  0);
                fe_f32x4 e1 = fe_load(exp_row + c +  4);
                fe_f32x4 e2 = fe_load(exp_row + c +  8);
                fe_f32x4 e3 = fe_load(exp_row + c + 12);
                int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(e0, vcomb4));
                int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(e1, vcomb4));
                int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(e2, vcomb4));
                int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(e3, vcomb4));
                int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
                int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
                vst1q_s8(qrow + c, vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
            }
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            /* x86 128-bit 4-way unrolled: same 16-lane consolidated path. */
            __m128 vcomb_x86 = _mm_set1_ps(combined);
            for (; c + 15 < cols; c += 16) {
                __m128 e0 = _mm_loadu_ps(exp_row + c +  0);
                __m128 e1 = _mm_loadu_ps(exp_row + c +  4);
                __m128 e2 = _mm_loadu_ps(exp_row + c +  8);
                __m128 e3 = _mm_loadu_ps(exp_row + c + 12);
                __m128i q0 = _mm_cvtps_epi32(_mm_mul_ps(e0, vcomb_x86));
                __m128i q1 = _mm_cvtps_epi32(_mm_mul_ps(e1, vcomb_x86));
                __m128i q2 = _mm_cvtps_epi32(_mm_mul_ps(e2, vcomb_x86));
                __m128i q3 = _mm_cvtps_epi32(_mm_mul_ps(e3, vcomb_x86));
                __m128i s01 = _mm_packs_epi32(q0, q1);
                __m128i s23 = _mm_packs_epi32(q2, q3);
                _mm_storeu_si128((__m128i *)(qrow + c), _mm_packs_epi16(s01, s23));
            }
#endif
            for (; c + 3 < cols; c += 4) {
                fe_f32x4 e   = fe_load(exp_row + c);
                fe_f32x4 e_q = fe_mul(e, vcomb4);
#if defined(FE_QGEMM_NEON)
                int32x4_t qi = vcvtnq_s32_f32(e_q);
                int16x4_t s  = vqmovn_s32(qi);
                int8x8_t  b  = vqmovn_s16(vcombine_s16(s, vdup_n_s16(0)));
                uint32_t w   = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(b), 0);
                memcpy(qrow + c, &w, 4);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                __m128i qi = _mm_cvtps_epi32(e_q);
                __m128i s  = _mm_packs_epi32(qi, qi);
                __m128i b  = _mm_packs_epi16(s, s);
                int32_t w  = _mm_cvtsi128_si32(b);
                memcpy(qrow + c, &w, 4);
#else
                for (int k = 0; k < 4; ++k) {
                    int q = (int)lroundf(((float *)&e_q)[k]);
                    if (q < -127) q = -127; if (q > 127) q = 127;
                    qrow[c + k] = (int8_t)q;
                }
#endif
            }
        }
        for (; c < cols; ++c) {
            int q = (int)lroundf(exp_row[c] * combined);
            if (q < -127) q = -127; if (q > 127) q = 127;
            qrow[c] = (int8_t)q;
        }
    }
}
