/*
 * Winograd F(2,3) for k=3 1-D convolution (fused).
 *
 * Inner GEMMs run via gemm_int32 (no per-call fp32 dequant); the B^T
 * input transform runs once with max-abs tracking for all 4 components,
 * and a single batched epilogue applies dequant + A^T + bias + SiLU.
 *
 *   B^T = [[1, 0,-1, 0],
 *          [0, 1, 1, 0],
 *          [0,-1, 1, 0],
 *          [0, 1, 0,-1]]
 *
 *   G   = [[1, 0, 0],
 *          [.5,.5,.5],
 *          [.5,-.5,.5],
 *          [0, 0, 1]]
 *
 *   A^T = [[1, 1, 1, 0],
 *          [0, 1,-1,-1]]
 *
 * Per tile t:
 *     d_i[t, c]    = (B^T row i) . x[2t-1..2t+2, c]
 *     U_i[t, n]    = sum_c Wg_i[n, c] * d_i[t, c]    (4 int32 GEMMs)
 *     y[2t,   n]   = U_0 + U_1 + U_2 + bias
 *     y[2t+1, n]   = U_1 - U_2 - U_3 + bias
 */

#include "fe_internal.h"
#include "fe_qgemm.h"
#include "fe_simd.h"
#include "fe_fp16.h"
#include "fe_profile.h"
#include "fe_config_medium.h"
#include "qgemm/qgemm_arch.h"
#include "qgemm/qgemm_simd_post.inl"
#include "qgemm/qgemm_dispatch.h"
#include <math.h>
#include <string.h>

/* Scratch pointers (single-threaded). Backing memory is the MHSA/Wino
 * union in FeState; set once at fe_state_create. */
#define FE_WINO_MAX_TILES  (FE_F1 / 2)
#define FE_WINO_MAX_C      FE_C1

static int8_t  *g_wino_dq;   /* [4, NTiles, Ci] int8  */
static int32_t *g_wino_U;    /* [4, NTiles, Co] int32 */

void fe_winograd_set_scratch(int32_t *u, int8_t *dq) {
    g_wino_U  = u;
    g_wino_dq = dq;
}

/* Offline weight derivation: fp32 G*W*G^T -> per-row symmetric int8
 * quant + DOTPROD pack. */

static void unpack_row_fp32(const int8_t *Wq_packed, const float *scales_w,
                             int N, int K, int n, float *out_row) {
    (void)N;
    const int NR = FE_QGEMM_NR;
    const int k4_groups = (K + 3) / 4;
    const int bn   = n / NR;
    const int lane = n % NR;
    const int8_t *blk = Wq_packed + (size_t)bn * (size_t)k4_groups * NR * 4;
    const float scale = scales_w[n];
    for (int g = 0; g < k4_groups; ++g) {
        const int8_t *q = blk + (size_t)g * NR * 4 + (size_t)lane * 4;
        for (int kl = 0; kl < 4; ++kl) {
            int k = g * 4 + kl;
            if (k < K) out_row[k] = scale * (float)q[kl];
        }
    }
}

void fe_winograd_f23_derive_weights(
        const int8_t *Wq_packed, const float *scales_w,
        int Co, int Ci,
        int8_t *wg_packed_out[4], float *wg_scales_out[4],
        float *scratch_fp32) {
    const int K = Ci * 3;
    float *W_fp32 = scratch_fp32;
    for (int n = 0; n < Co; ++n)
        unpack_row_fp32(Wq_packed, scales_w, Co, K, n, W_fp32 + (size_t)n * K);

    float *Wg_fp32 = scratch_fp32 + (size_t)Co * K;
    const size_t plane = (size_t)Co * Ci;
    for (int n = 0; n < Co; ++n) {
        for (int c = 0; c < Ci; ++c) {
            float w0 = W_fp32[(size_t)n * K + (size_t)c * 3 + 0];
            float w1 = W_fp32[(size_t)n * K + (size_t)c * 3 + 1];
            float w2 = W_fp32[(size_t)n * K + (size_t)c * 3 + 2];
            Wg_fp32[0 * plane + (size_t)n * Ci + c] = w0;
            Wg_fp32[1 * plane + (size_t)n * Ci + c] = 0.5f * (w0 + w1 + w2);
            Wg_fp32[2 * plane + (size_t)n * Ci + c] = 0.5f * (w0 - w1 + w2);
            Wg_fp32[3 * plane + (size_t)n * Ci + c] = w2;
        }
    }
    for (int i = 0; i < 4; ++i) {
        const float *Wg_i = Wg_fp32 + (size_t)i * plane;
        fe_qgemm_pack_W(Wg_i, Co, Ci, wg_packed_out[i], wg_scales_out[i]);
    }
}

/*
 * Pass A: B^T transform + max-abs tracking, no storage.
 * Pass B: same B^T, quantized with the known per-component scales.
 * Two-pass form keeps the 4 fp32 transforms out of L1d (recompute is
 * cheaper than the extra memory traffic).
 */
/* abs(x) -- single op on NEON, max(x,-x) elsewhere. */
#if defined(FE_QGEMM_NEON)
#  define WINO_ABSQ(v) vabsq_f32(v)
#elif defined(FE_QPOST_AVX2)
static inline __m128 wino_absq(__m128 v) {
    return _mm_and_ps(v, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
}
#  define WINO_ABSQ(v) wino_absq(v)
#else
#  define WINO_ABSQ(v) fe_max((v), fe_neg(v))
#endif

static void wino_input_transform_pass_a(const float *in, int freq, int Ci,
                                         float max_abs_out[4]) {
    const int NTiles = freq / 2;
    fe_f32x4 vmax_0 = fe_set1(0.0f), vmax_1 = fe_set1(0.0f);
    fe_f32x4 vmax_2 = fe_set1(0.0f), vmax_3 = fe_set1(0.0f);
    const fe_f32x4 vzero = fe_set1(0.0f);

    for (int t = 0; t < NTiles; ++t) {
        const int p0 = 2 * t - 1;
        const int p1 = 2 * t;
        const int p2 = 2 * t + 1;
        const int p3 = 2 * t + 2;
        const float *r_p1 = in + (size_t)p1 * Ci;
        const float *r_p2 = in + (size_t)p2 * Ci;
        const float *r_p0 = (p0 >= 0)   ? in + (size_t)p0 * Ci : NULL;
        const float *r_p3 = (p3 < freq) ? in + (size_t)p3 * Ci : NULL;

        int c = 0;
        for (; c + 3 < Ci; c += 4) {
            fe_f32x4 v_p1 = fe_load(r_p1 + c);
            fe_f32x4 v_p2 = fe_load(r_p2 + c);
            fe_f32x4 v_p0 = r_p0 ? fe_load(r_p0 + c) : vzero;
            fe_f32x4 v_p3 = r_p3 ? fe_load(r_p3 + c) : vzero;
            fe_f32x4 d0 = fe_sub(v_p0, v_p2);
            fe_f32x4 d1 = fe_add(v_p1, v_p2);
            fe_f32x4 d2 = fe_sub(v_p2, v_p1);
            fe_f32x4 d3 = fe_sub(v_p1, v_p3);
            vmax_0 = fe_max(vmax_0, WINO_ABSQ(d0));
            vmax_1 = fe_max(vmax_1, WINO_ABSQ(d1));
            vmax_2 = fe_max(vmax_2, WINO_ABSQ(d2));
            vmax_3 = fe_max(vmax_3, WINO_ABSQ(d3));
        }
        if (c < Ci) {
            float m0 = fe_hmax(vmax_0), m1 = fe_hmax(vmax_1);
            float m2 = fe_hmax(vmax_2), m3 = fe_hmax(vmax_3);
            for (; c < Ci; ++c) {
                float v_p0 = r_p0 ? r_p0[c] : 0.0f;
                float v_p1 = r_p1[c];
                float v_p2 = r_p2[c];
                float v_p3 = r_p3 ? r_p3[c] : 0.0f;
                float d0 = v_p0 - v_p2, ad0 = d0 < 0 ? -d0 : d0;
                float d1 = v_p1 + v_p2, ad1 = d1 < 0 ? -d1 : d1;
                float d2 = v_p2 - v_p1, ad2 = d2 < 0 ? -d2 : d2;
                float d3 = v_p1 - v_p3, ad3 = d3 < 0 ? -d3 : d3;
                if (ad0 > m0) m0 = ad0;
                if (ad1 > m1) m1 = ad1;
                if (ad2 > m2) m2 = ad2;
                if (ad3 > m3) m3 = ad3;
            }
            vmax_0 = fe_max(vmax_0, fe_set1(m0));
            vmax_1 = fe_max(vmax_1, fe_set1(m1));
            vmax_2 = fe_max(vmax_2, fe_set1(m2));
            vmax_3 = fe_max(vmax_3, fe_set1(m3));
        }
    }
    max_abs_out[0] = fe_hmax(vmax_0);
    max_abs_out[1] = fe_hmax(vmax_1);
    max_abs_out[2] = fe_hmax(vmax_2);
    max_abs_out[3] = fe_hmax(vmax_3);
}

#if defined(FE_QGEMM_NEON)
static inline void wino_quant4_store(int8_t *dst, float32x4_t v, float32x4_t vinv) {
    int32x4_t q = vcvtnq_s32_f32(vmulq_f32(v, vinv));
    int8x8_t  s = vqmovn_s16(vcombine_s16(vqmovn_s32(q), vdup_n_s16(0)));
    s = vmax_s8(s, vdup_n_s8(-127));
    uint32_t  w = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(s), 0);
    memcpy(dst, &w, 4);
}
#elif defined(FE_QPOST_AVX2)
static inline void wino_quant4_store(int8_t *dst, __m128 v, __m128 vinv) {
    __m128i q  = _mm_cvtps_epi32(_mm_mul_ps(v, vinv));
    __m128i s  = _mm_packs_epi32(q, q);
    __m128i b  = _mm_packs_epi16(s, s);
    b = _mm_max_epi8(b, _mm_set1_epi8(-127));
    int32_t w  = _mm_cvtsi128_si32(b);
    memcpy(dst, &w, 4);
}
#endif

static void wino_input_transform_pass_b(const float *in, int freq, int Ci,
                                         const float inv_scales[4],
                                         int8_t *dq_out[4]) {
    const int NTiles = freq / 2;
    const fe_f32x4 vinv0 = fe_set1(inv_scales[0]);
    const fe_f32x4 vinv1 = fe_set1(inv_scales[1]);
    const fe_f32x4 vinv2 = fe_set1(inv_scales[2]);
    const fe_f32x4 vinv3 = fe_set1(inv_scales[3]);
    const fe_f32x4 vzero = fe_set1(0.0f);

    for (int t = 0; t < NTiles; ++t) {
        const int p0 = 2 * t - 1;
        const int p1 = 2 * t;
        const int p2 = 2 * t + 1;
        const int p3 = 2 * t + 2;
        const float *r_p1 = in + (size_t)p1 * Ci;
        const float *r_p2 = in + (size_t)p2 * Ci;
        const float *r_p0 = (p0 >= 0)   ? in + (size_t)p0 * Ci : NULL;
        const float *r_p3 = (p3 < freq) ? in + (size_t)p3 * Ci : NULL;
        int8_t *o0 = dq_out[0] + (size_t)t * Ci;
        int8_t *o1 = dq_out[1] + (size_t)t * Ci;
        int8_t *o2 = dq_out[2] + (size_t)t * Ci;
        int8_t *o3 = dq_out[3] + (size_t)t * Ci;

        int c = 0;
#if defined(FE_QGEMM_NEON) || defined(FE_QPOST_AVX2)
        for (; c + 3 < Ci; c += 4) {
            fe_f32x4 v_p1 = fe_load(r_p1 + c);
            fe_f32x4 v_p2 = fe_load(r_p2 + c);
            fe_f32x4 v_p0 = r_p0 ? fe_load(r_p0 + c) : vzero;
            fe_f32x4 v_p3 = r_p3 ? fe_load(r_p3 + c) : vzero;
            fe_f32x4 d0 = fe_sub(v_p0, v_p2);
            fe_f32x4 d1 = fe_add(v_p1, v_p2);
            fe_f32x4 d2 = fe_sub(v_p2, v_p1);
            fe_f32x4 d3 = fe_sub(v_p1, v_p3);
            wino_quant4_store(o0 + c, d0, vinv0);
            wino_quant4_store(o1 + c, d1, vinv1);
            wino_quant4_store(o2 + c, d2, vinv2);
            wino_quant4_store(o3 + c, d3, vinv3);
        }
#endif
        for (; c < Ci; ++c) {
            float v_p0 = r_p0 ? r_p0[c] : 0.0f;
            float v_p1 = r_p1[c];
            float v_p2 = r_p2[c];
            float v_p3 = r_p3 ? r_p3[c] : 0.0f;
            float d0 = v_p0 - v_p2, d1 = v_p1 + v_p2;
            float d2 = v_p2 - v_p1, d3 = v_p1 - v_p3;
            int qq0 = (int)lroundf(d0 * inv_scales[0]); if (qq0 < -127) qq0 = -127; if (qq0 > 127) qq0 = 127;
            int qq1 = (int)lroundf(d1 * inv_scales[1]); if (qq1 < -127) qq1 = -127; if (qq1 > 127) qq1 = 127;
            int qq2 = (int)lroundf(d2 * inv_scales[2]); if (qq2 < -127) qq2 = -127; if (qq2 > 127) qq2 = 127;
            int qq3 = (int)lroundf(d3 * inv_scales[3]); if (qq3 < -127) qq3 = -127; if (qq3 > 127) qq3 = 127;
            o0[c] = (int8_t)qq0; o1[c] = (int8_t)qq1;
            o2[c] = (int8_t)qq2; o3[c] = (int8_t)qq3;
        }
    }
}

/*
 * Batched fused epilogue. For each (t, n):
 *   u_i = combined[i][n] * (float)U_i[t, n]
 *   y[2t,   n] = u_0 + u_1 + u_2 + bias[n]    [+ SiLU]
 *   y[2t+1, n] = u_1 - u_2 - u_3 + bias[n]    [+ SiLU]
 * combined[i][n] = qx_scale[i] * wino_scales[i][n] (precomputed).
 */
static __attribute__((always_inline)) inline void wino_fused_epilogue_impl(
        int has_bias, int act_silu_const, int has_skip,
        const int32_t *U[4],
        const float   *combined[4],
        const float   *bias,
        int NTiles, int Co,
        float *out, float *skip_out) {
    for (int t = 0; t < NTiles; ++t) {
        const int32_t *u0r = U[0] + (size_t)t * Co;
        const int32_t *u1r = U[1] + (size_t)t * Co;
        const int32_t *u2r = U[2] + (size_t)t * Co;
        const int32_t *u3r = U[3] + (size_t)t * Co;
        float *y_even = out + (size_t)(2 * t)     * Co;
        float *y_odd  = out + (size_t)(2 * t + 1) * Co;
        float *s_even = has_skip ? skip_out + (size_t)(2 * t)     * Co : NULL;
        float *s_odd  = has_skip ? skip_out + (size_t)(2 * t + 1) * Co : NULL;

        int n = 0;
#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
        for (; n + 15 < Co; n += 16) {
            __m512 cs0 = _mm512_loadu_ps(combined[0] + n);
            __m512 cs1 = _mm512_loadu_ps(combined[1] + n);
            __m512 cs2 = _mm512_loadu_ps(combined[2] + n);
            __m512 cs3 = _mm512_loadu_ps(combined[3] + n);
            __m512 u0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(u0r + n))), cs0);
            __m512 u1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(u1r + n))), cs1);
            __m512 u2 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(u2r + n))), cs2);
            __m512 u3 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(u3r + n))), cs3);
            __m512 b  = has_bias ? _mm512_loadu_ps(bias + n) : _mm512_setzero_ps();
            __m512 ye = _mm512_add_ps(_mm512_add_ps(u0, u1), _mm512_add_ps(u2, b));
            __m512 yo = _mm512_add_ps(_mm512_sub_ps(_mm512_sub_ps(u1, u2), u3), b);
            if (act_silu_const) {
                ye = _mm512_mul_ps(ye, fe_qg_sigmoid16(ye));
                yo = _mm512_mul_ps(yo, fe_qg_sigmoid16(yo));
            }
            _mm512_storeu_ps(y_even + n, ye);
            _mm512_storeu_ps(y_odd  + n, yo);
            if (has_skip) {
                _mm512_storeu_ps(s_even + n, ye);
                _mm512_storeu_ps(s_odd  + n, yo);
            }
        }
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
        for (; n + 7 < Co; n += 8) {
            __m256 cs0 = _mm256_loadu_ps(combined[0] + n);
            __m256 cs1 = _mm256_loadu_ps(combined[1] + n);
            __m256 cs2 = _mm256_loadu_ps(combined[2] + n);
            __m256 cs3 = _mm256_loadu_ps(combined[3] + n);
            __m256 u0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(u0r + n))), cs0);
            __m256 u1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(u1r + n))), cs1);
            __m256 u2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(u2r + n))), cs2);
            __m256 u3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(u3r + n))), cs3);
            __m256 b  = has_bias ? _mm256_loadu_ps(bias + n) : _mm256_setzero_ps();
            __m256 ye = _mm256_add_ps(_mm256_add_ps(u0, u1), _mm256_add_ps(u2, b));
            __m256 yo = _mm256_add_ps(_mm256_sub_ps(_mm256_sub_ps(u1, u2), u3), b);
            if (act_silu_const) {
                ye = _mm256_mul_ps(ye, fe_qg_sigmoid8(ye));
                yo = _mm256_mul_ps(yo, fe_qg_sigmoid8(yo));
            }
            _mm256_storeu_ps(y_even + n, ye);
            _mm256_storeu_ps(y_odd  + n, yo);
            if (has_skip) {
                _mm256_storeu_ps(s_even + n, ye);
                _mm256_storeu_ps(s_odd  + n, yo);
            }
        }
#endif
#if defined(FE_QGEMM_NEON)
        for (; n + 3 < Co; n += 4) {
            float32x4_t cs0 = vld1q_f32(combined[0] + n);
            float32x4_t cs1 = vld1q_f32(combined[1] + n);
            float32x4_t cs2 = vld1q_f32(combined[2] + n);
            float32x4_t cs3 = vld1q_f32(combined[3] + n);
            float32x4_t u0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(u0r + n)), cs0);
            float32x4_t u1 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(u1r + n)), cs1);
            float32x4_t u2 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(u2r + n)), cs2);
            float32x4_t u3 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(u3r + n)), cs3);
            float32x4_t b  = has_bias ? vld1q_f32(bias + n) : vdupq_n_f32(0.0f);
            float32x4_t ye = vaddq_f32(vaddq_f32(u0, u1), vaddq_f32(u2, b));
            float32x4_t yo = vaddq_f32(vsubq_f32(vsubq_f32(u1, u2), u3), b);
            if (act_silu_const) {
                ye = vmulq_f32(ye, FE_SIGMOIDF4(ye));
                yo = vmulq_f32(yo, FE_SIGMOIDF4(yo));
            }
            vst1q_f32(y_even + n, ye);
            vst1q_f32(y_odd  + n, yo);
            if (has_skip) {
                vst1q_f32(s_even + n, ye);
                vst1q_f32(s_odd  + n, yo);
            }
        }
#endif
        for (; n < Co; ++n) {
            float u0 = (float)u0r[n] * combined[0][n];
            float u1 = (float)u1r[n] * combined[1][n];
            float u2 = (float)u2r[n] * combined[2][n];
            float u3 = (float)u3r[n] * combined[3][n];
            float b  = has_bias ? bias[n] : 0.0f;
            float ye = u0 + u1 + u2 + b;
            float yo = u1 - u2 - u3 + b;
            if (act_silu_const) {
                ye = ye / (1.0f + expf(-ye));
                yo = yo / (1.0f + expf(-yo));
            }
            y_even[n] = ye;
            y_odd[n]  = yo;
            if (has_skip) {
                s_even[n] = ye;
                s_odd[n]  = yo;
            }
        }
    }
}

/* Dispatch into one of 8 (has_bias, act_silu, has_skip) specializations
 * so the inner body sees them all as constants. skip_out may be NULL;
 * when set, both stores happen in the same epilogue pass. */
static void wino_fused_epilogue(const int32_t *U[4],
                                 const float   *combined[4],
                                 const float   *bias,
                                 int NTiles, int Co,
                                 int act_silu,
                                 float *out, float *skip_out) {
    if (skip_out) {
        if (bias) {
            if (act_silu) wino_fused_epilogue_impl(1, 1, 1, U, combined, bias, NTiles, Co, out, skip_out);
            else          wino_fused_epilogue_impl(1, 0, 1, U, combined, bias, NTiles, Co, out, skip_out);
        } else {
            if (act_silu) wino_fused_epilogue_impl(0, 1, 1, U, combined, NULL, NTiles, Co, out, skip_out);
            else          wino_fused_epilogue_impl(0, 0, 1, U, combined, NULL, NTiles, Co, out, skip_out);
        }
    } else {
        if (bias) {
            if (act_silu) wino_fused_epilogue_impl(1, 1, 0, U, combined, bias, NTiles, Co, out, NULL);
            else          wino_fused_epilogue_impl(1, 0, 0, U, combined, bias, NTiles, Co, out, NULL);
        } else {
            if (act_silu) wino_fused_epilogue_impl(0, 1, 0, U, combined, NULL, NTiles, Co, out, NULL);
            else          wino_fused_epilogue_impl(0, 0, 0, U, combined, NULL, NTiles, Co, out, NULL);
        }
    }
}

/*
 * fp16 skip-write epilogue. Same math as wino_fused_epilogue
 * but skip_out is uint16_t* (IEEE binary16). Eliminates the engine-side
 * pack pass on encoder Winograd writes.
 *
 * Conversion is in-line at store time (VCVTPS2PH on x86, FCVTN on ARM),
 * which produces deterministic per-tier output. Cross-tier byte-id within
 * the fp16 path requires the same fp32 SiLU register values across tiers —
 * which holds because the polynomial is identical and the fp16 RNE
 * rounding is hardware-deterministic.
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
__attribute__((target("avx2,f16c,fma")))
#endif
static __attribute__((always_inline)) inline void wino_fused_epilogue_fp16skip_impl(
        int has_bias, int act_silu_const,
        const int32_t *U[4],
        const float   *combined[4],
        const float   *bias,
        int NTiles, int Co,
        float *out, uint16_t *skip_out) {
    for (int t = 0; t < NTiles; ++t) {
        const int32_t *u0r = U[0] + (size_t)t * Co;
        const int32_t *u1r = U[1] + (size_t)t * Co;
        const int32_t *u2r = U[2] + (size_t)t * Co;
        const int32_t *u3r = U[3] + (size_t)t * Co;
        float    *y_even = out      + (size_t)(2 * t)     * Co;
        float    *y_odd  = out      + (size_t)(2 * t + 1) * Co;
        uint16_t *s_even = skip_out + (size_t)(2 * t)     * Co;
        uint16_t *s_odd  = skip_out + (size_t)(2 * t + 1) * Co;

        int n = 0;
#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
        for (; n + 15 < Co; n += 16) {
            __m512 cs0 = _mm512_loadu_ps(combined[0] + n);
            __m512 cs1 = _mm512_loadu_ps(combined[1] + n);
            __m512 cs2 = _mm512_loadu_ps(combined[2] + n);
            __m512 cs3 = _mm512_loadu_ps(combined[3] + n);
            __m512 u0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(u0r + n))), cs0);
            __m512 u1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(u1r + n))), cs1);
            __m512 u2 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(u2r + n))), cs2);
            __m512 u3 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(u3r + n))), cs3);
            __m512 b  = has_bias ? _mm512_loadu_ps(bias + n) : _mm512_setzero_ps();
            __m512 ye = _mm512_add_ps(_mm512_add_ps(u0, u1), _mm512_add_ps(u2, b));
            __m512 yo = _mm512_add_ps(_mm512_sub_ps(_mm512_sub_ps(u1, u2), u3), b);
            if (act_silu_const) {
                ye = _mm512_mul_ps(ye, fe_qg_sigmoid16(ye));
                yo = _mm512_mul_ps(yo, fe_qg_sigmoid16(yo));
            }
            _mm512_storeu_ps(y_even + n, ye);
            _mm512_storeu_ps(y_odd  + n, yo);
            /* AVX-512 has no 512-bit fp16 store; split to 2x 256-bit. */
            __m256 ye_lo = _mm512_castps512_ps256(ye);
            __m256 ye_hi = _mm512_extractf32x8_ps(ye, 1);
            __m256 yo_lo = _mm512_castps512_ps256(yo);
            __m256 yo_hi = _mm512_extractf32x8_ps(yo, 1);
            fe_fp16_store8(s_even + n + 0, ye_lo);
            fe_fp16_store8(s_even + n + 8, ye_hi);
            fe_fp16_store8(s_odd  + n + 0, yo_lo);
            fe_fp16_store8(s_odd  + n + 8, yo_hi);
        }
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
        for (; n + 7 < Co; n += 8) {
            __m256 cs0 = _mm256_loadu_ps(combined[0] + n);
            __m256 cs1 = _mm256_loadu_ps(combined[1] + n);
            __m256 cs2 = _mm256_loadu_ps(combined[2] + n);
            __m256 cs3 = _mm256_loadu_ps(combined[3] + n);
            __m256 u0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(u0r + n))), cs0);
            __m256 u1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(u1r + n))), cs1);
            __m256 u2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(u2r + n))), cs2);
            __m256 u3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(u3r + n))), cs3);
            __m256 b  = has_bias ? _mm256_loadu_ps(bias + n) : _mm256_setzero_ps();
            __m256 ye = _mm256_add_ps(_mm256_add_ps(u0, u1), _mm256_add_ps(u2, b));
            __m256 yo = _mm256_add_ps(_mm256_sub_ps(_mm256_sub_ps(u1, u2), u3), b);
            if (act_silu_const) {
                ye = _mm256_mul_ps(ye, fe_qg_sigmoid8(ye));
                yo = _mm256_mul_ps(yo, fe_qg_sigmoid8(yo));
            }
            _mm256_storeu_ps(y_even + n, ye);
            _mm256_storeu_ps(y_odd  + n, yo);
            fe_fp16_store8(s_even + n, ye);
            fe_fp16_store8(s_odd  + n, yo);
        }
#endif
#if defined(FE_QGEMM_NEON)
        for (; n + 3 < Co; n += 4) {
            float32x4_t cs0 = vld1q_f32(combined[0] + n);
            float32x4_t cs1 = vld1q_f32(combined[1] + n);
            float32x4_t cs2 = vld1q_f32(combined[2] + n);
            float32x4_t cs3 = vld1q_f32(combined[3] + n);
            float32x4_t u0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(u0r + n)), cs0);
            float32x4_t u1 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(u1r + n)), cs1);
            float32x4_t u2 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(u2r + n)), cs2);
            float32x4_t u3 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(u3r + n)), cs3);
            float32x4_t b  = has_bias ? vld1q_f32(bias + n) : vdupq_n_f32(0.0f);
            float32x4_t ye = vaddq_f32(vaddq_f32(u0, u1), vaddq_f32(u2, b));
            float32x4_t yo = vaddq_f32(vsubq_f32(vsubq_f32(u1, u2), u3), b);
            if (act_silu_const) {
                ye = vmulq_f32(ye, FE_SIGMOIDF4(ye));
                yo = vmulq_f32(yo, FE_SIGMOIDF4(yo));
            }
            vst1q_f32(y_even + n, ye);
            vst1q_f32(y_odd  + n, yo);
            fe_fp16_store4(s_even + n, ye);
            fe_fp16_store4(s_odd  + n, yo);
        }
#endif
        /* Co dimension contract: 8-aligned on x86, 4-aligned on ARM.
         * FE_C1=96 satisfies both. Misalignment → abort. */
        if (n != Co) fe_fp16_alignment_violation();
    }
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
__attribute__((target("avx2,f16c,fma")))
#endif
static void wino_fused_epilogue_fp16skip(const int32_t *U[4],
                                          const float   *combined[4],
                                          const float   *bias,
                                          int NTiles, int Co,
                                          int act_silu,
                                          float *out, uint16_t *skip_out) {
    if (bias) {
        if (act_silu) wino_fused_epilogue_fp16skip_impl(1, 1, U, combined, bias, NTiles, Co, out, skip_out);
        else          wino_fused_epilogue_fp16skip_impl(1, 0, U, combined, bias, NTiles, Co, out, skip_out);
    } else {
        if (act_silu) wino_fused_epilogue_fp16skip_impl(0, 1, U, combined, NULL, NTiles, Co, out, skip_out);
        else          wino_fused_epilogue_fp16skip_impl(0, 0, U, combined, NULL, NTiles, Co, out, skip_out);
    }
}

/* Public driver -- fused Winograd F(2,3) k=3 1-D conv. */
static void k3_winograd_v2_core(FeConv1d *c, const float *in, float *out,
                                 int freq, int8_t *aq, int32_t *c32,
                                 int act_silu, float *skip_out) {
    (void)aq; (void)c32;             /* dedicated scratch above */
    const int Ci = c->in_ch;
    const int Co = c->out_ch;
    const int NTiles = freq / 2;

    /* Slice scratch into 4 per-component slabs. */
    int8_t *dq[4];
    int32_t *U[4];
    const size_t d_plane = (size_t)NTiles * Ci;
    const size_t u_plane = (size_t)NTiles * Co;
    for (int i = 0; i < 4; ++i) {
        dq[i] = g_wino_dq + (size_t)i * d_plane;
        U[i]  = g_wino_U  + (size_t)i * u_plane;
    }

    /* 1. Pass A: per-component max-abs scan. */
    float max_abs[4];
    wino_input_transform_pass_a(in, freq, Ci, max_abs);

    /* 2. Per-component symmetric int8 scale. */
    float qx_scale[4], qx_inv[4];
    for (int i = 0; i < 4; ++i) {
        float m = max_abs[i];
        if (m == 0.0f) m = 1.0f;
        qx_scale[i] = m / 127.0f;
        qx_inv[i]   = 127.0f / m;
    }

    /* 3. Pass B: transform + quantize on the fly. */
    wino_input_transform_pass_b(in, freq, Ci, qx_inv, dq);

    /* 4. Combined scale: combined[i][n] = qx_scale[i] * wino_scales[i][n]. */
    float combined[4][FE_WINO_MAX_C];
    for (int i = 0; i < 4; ++i) {
        const float *sw = c->wino_scales[i];
        float qs = qx_scale[i];
        for (int n = 0; n < Co; ++n) combined[i][n] = qs * sw[n];
    }

    /* 4 inner GEMM calls + a single batched epilogue. Per-tile fused
     * microkernels were prototyped and dropped: at MR=2 they lose weight
     * reuse and regress vs this batched path on every tier we measured. */
    for (int i = 0; i < 4; ++i) {
        fe_qgemm_ops.gemm_int32(NTiles, Co, Ci,
                                dq[i], c->wino_weight_q[i],
                                U[i], Co);
    }
    const float *combined_ptr[4] = { combined[0], combined[1],
                                     combined[2], combined[3] };
    const int32_t *U_ptr[4] = { U[0], U[1], U[2], U[3] };
    wino_fused_epilogue(U_ptr, combined_ptr, c->bias,
                        NTiles, Co, act_silu, out, skip_out);
}

void fe_conv1d_k3_winograd_silu(FeConv1d *c, const float *in, float *out,
                                 int freq, int8_t *aq, int32_t *c32) {
    k3_winograd_v2_core(c, in, out, freq, aq, c32, 1, NULL);
}

/*
 * fp16 skip-write variant. Same as _silu_skip but the
 * skip slot is uint16_t* (IEEE binary16). One fused pass: SIMD VCVTPS2PH
 * / FCVTN at the store side eliminates the engine-side scratch + pack.
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
__attribute__((target("avx2,f16c,fma")))
#endif
static void k3_winograd_v2_core_fp16skip(FeConv1d *c, const float *in,
                                          float *out, uint16_t *skip_out,
                                          int freq, int8_t *aq, int32_t *c32,
                                          int act_silu) {
    (void)aq; (void)c32;
    const int Ci = c->in_ch;
    const int Co = c->out_ch;
    const int NTiles = freq / 2;

    int8_t *dq[4];
    int32_t *U[4];
    const size_t d_plane = (size_t)NTiles * Ci;
    const size_t u_plane = (size_t)NTiles * Co;
    for (int i = 0; i < 4; ++i) {
        dq[i] = g_wino_dq + (size_t)i * d_plane;
        U[i]  = g_wino_U  + (size_t)i * u_plane;
    }

    float max_abs[4];
    wino_input_transform_pass_a(in, freq, Ci, max_abs);

    float qx_scale[4], qx_inv[4];
    for (int i = 0; i < 4; ++i) {
        float m = max_abs[i];
        if (m == 0.0f) m = 1.0f;
        qx_scale[i] = m / 127.0f;
        qx_inv[i]   = 127.0f / m;
    }

    wino_input_transform_pass_b(in, freq, Ci, qx_inv, dq);

    float combined[4][FE_WINO_MAX_C];
    for (int i = 0; i < 4; ++i) {
        const float *sw = c->wino_scales[i];
        float qs = qx_scale[i];
        for (int n = 0; n < Co; ++n) combined[i][n] = qs * sw[n];
    }

    for (int i = 0; i < 4; ++i) {
        fe_qgemm_ops.gemm_int32(NTiles, Co, Ci,
                                dq[i], c->wino_weight_q[i],
                                U[i], Co);
    }
    const float *combined_ptr[4] = { combined[0], combined[1],
                                     combined[2], combined[3] };
    const int32_t *U_ptr[4] = { U[0], U[1], U[2], U[3] };
    wino_fused_epilogue_fp16skip(U_ptr, combined_ptr, c->bias,
                                  NTiles, Co, act_silu, out, skip_out);
}

void fe_conv1d_k3_winograd_silu_skip_fp16(FeConv1d *c, const float *in,
                                           float *out, uint16_t *skip_out,
                                           int freq, int8_t *aq, int32_t *c32) {
    k3_winograd_v2_core_fp16skip(c, in, out, skip_out, freq, aq, c32, 1);
}
