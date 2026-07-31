/*
 * High-level int8 W8A8 GEMM API and dynamic activation quantization.
 *
 * Quant scheme:
 *   Weights: per-output-channel symmetric int8 (max-abs / 127).
 *   Activations: per-tensor asymmetric uint8 stored as int8 = u8 - 128
 *     so signed kernels (DOTPROD/I8MM/AVX2/AVX-VNNI/AVX-512-VNNI) consume
 *     the same buffer.
 *
 * Dequant:
 *     C32[m,n] = sum_k A_i8[m,k] * W_i8[n,k]
 *     y[m,n]   = scale_x * scales_w[n] *
 *                (C32[m,n] + (128 - zp_x) * row_sums[n]) + bias[n]
 *
 * The row-sum correction folds into an effective bias:
 *     bias_eff[n] = bias[n] + scale_x*scales_w[n]*(128 - zp_x)*row_sums[n]
 *
 * Calibration is dynamic per-frame (matches ORT DynamicQuantizeLinear);
 * scales never freeze.
 */
#include "fe_internal.h"
#include "fe_qgemm.h"
#include "fe_simd.h"
#include "fe_fp16.h"
#include "fe_config_medium.h"
#include "qgemm/qgemm_arch.h"
#include "qgemm/qgemm_simd_post.inl"
#include "qgemm/qgemm_dispatch.h"
#include "qgemm/arch_kernels.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Worst-case N: attention QKV proj at 3*C2 bounds all callers. */
#define FE_QGEMM_MAX_N (3 * FE_C2)

/* File-static dequant scratch (engine is single-threaded). */
static float g_combined_scale[FE_QGEMM_MAX_N];
static float g_bias_eff_buf  [FE_QGEMM_MAX_N];

/* bias_eff[n] = bias[n] + combined_scale[n]*(128 - zp_x)*row_sums[n].
 * When zp_x == 128 the correction is zero (caller returns bias as-is).
 * Templated on has_bias so the bias-load path folds away for the
 * bias=NULL case (GRU W_ih). */
static __attribute__((always_inline)) inline void build_bias_eff_impl(
        int has_bias,
        const float *bias,
        const float *combined_scale,
        const int32_t *row_sums,
        float k, int N,
        float *bias_eff_buf) {
    int n = 0;

#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
    /* 4-way unrolled AVX-512: 64 N/iter. */
    const __m512 vk = _mm512_set1_ps(k);
    const __m512 vzero = _mm512_setzero_ps();
    for (; n + 63 < N; n += 64) {
        __m512 cs0 = _mm512_loadu_ps(combined_scale + n +  0);
        __m512 cs1 = _mm512_loadu_ps(combined_scale + n + 16);
        __m512 cs2 = _mm512_loadu_ps(combined_scale + n + 32);
        __m512 cs3 = _mm512_loadu_ps(combined_scale + n + 48);
        __m512 rs0 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(row_sums + n +  0)));
        __m512 rs1 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(row_sums + n + 16)));
        __m512 rs2 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(row_sums + n + 32)));
        __m512 rs3 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i *)(row_sums + n + 48)));
        __m512 b0  = has_bias ? _mm512_loadu_ps(bias + n +  0) : vzero;
        __m512 b1  = has_bias ? _mm512_loadu_ps(bias + n + 16) : vzero;
        __m512 b2  = has_bias ? _mm512_loadu_ps(bias + n + 32) : vzero;
        __m512 b3  = has_bias ? _mm512_loadu_ps(bias + n + 48) : vzero;
        _mm512_storeu_ps(bias_eff_buf + n +  0, _mm512_fmadd_ps(_mm512_mul_ps(cs0, vk), rs0, b0));
        _mm512_storeu_ps(bias_eff_buf + n + 16, _mm512_fmadd_ps(_mm512_mul_ps(cs1, vk), rs1, b1));
        _mm512_storeu_ps(bias_eff_buf + n + 32, _mm512_fmadd_ps(_mm512_mul_ps(cs2, vk), rs2, b2));
        _mm512_storeu_ps(bias_eff_buf + n + 48, _mm512_fmadd_ps(_mm512_mul_ps(cs3, vk), rs3, b3));
    }
    for (; n + 15 < N; n += 16) {
        __m512  cs = _mm512_loadu_ps(combined_scale + n);
        __m512i rsi = _mm512_loadu_si512((const __m512i *)(row_sums + n));
        __m512  rs  = _mm512_cvtepi32_ps(rsi);
        __m512  b   = has_bias ? _mm512_loadu_ps(bias + n) : vzero;
        __m512  v   = _mm512_fmadd_ps(_mm512_mul_ps(cs, vk), rs, b);
        _mm512_storeu_ps(bias_eff_buf + n, v);
    }
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
    /* 4-way unrolled AVX2: 32 N/iter. */
    const __m256 vk = _mm256_set1_ps(k);
    const __m256 vzero = _mm256_setzero_ps();
    for (; n + 31 < N; n += 32) {
        __m256 cs0 = _mm256_loadu_ps(combined_scale + n +  0);
        __m256 cs1 = _mm256_loadu_ps(combined_scale + n +  8);
        __m256 cs2 = _mm256_loadu_ps(combined_scale + n + 16);
        __m256 cs3 = _mm256_loadu_ps(combined_scale + n + 24);
        __m256 rs0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(row_sums + n +  0)));
        __m256 rs1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(row_sums + n +  8)));
        __m256 rs2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(row_sums + n + 16)));
        __m256 rs3 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(row_sums + n + 24)));
        __m256 b0  = has_bias ? _mm256_loadu_ps(bias + n +  0) : vzero;
        __m256 b1  = has_bias ? _mm256_loadu_ps(bias + n +  8) : vzero;
        __m256 b2  = has_bias ? _mm256_loadu_ps(bias + n + 16) : vzero;
        __m256 b3  = has_bias ? _mm256_loadu_ps(bias + n + 24) : vzero;
        _mm256_storeu_ps(bias_eff_buf + n +  0, _mm256_fmadd_ps(_mm256_mul_ps(cs0, vk), rs0, b0));
        _mm256_storeu_ps(bias_eff_buf + n +  8, _mm256_fmadd_ps(_mm256_mul_ps(cs1, vk), rs1, b1));
        _mm256_storeu_ps(bias_eff_buf + n + 16, _mm256_fmadd_ps(_mm256_mul_ps(cs2, vk), rs2, b2));
        _mm256_storeu_ps(bias_eff_buf + n + 24, _mm256_fmadd_ps(_mm256_mul_ps(cs3, vk), rs3, b3));
    }
    for (; n + 7 < N; n += 8) {
        __m256  cs = _mm256_loadu_ps(combined_scale + n);
        __m256i rsi = _mm256_loadu_si256((const __m256i *)(row_sums + n));
        __m256  rs  = _mm256_cvtepi32_ps(rsi);
        __m256  b   = has_bias ? _mm256_loadu_ps(bias + n) : vzero;
        __m256  v   = _mm256_fmadd_ps(_mm256_mul_ps(cs, vk), rs, b);
        _mm256_storeu_ps(bias_eff_buf + n, v);
    }
#elif defined(FE_QGEMM_NEON)
    const float32x4_t vk = vdupq_n_f32(k);
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    /* 4-way unrolled NEON: 16 N/iter. */
    for (; n + 15 < N; n += 16) {
        float32x4_t cs0 = vld1q_f32(combined_scale + n +  0);
        float32x4_t cs1 = vld1q_f32(combined_scale + n +  4);
        float32x4_t cs2 = vld1q_f32(combined_scale + n +  8);
        float32x4_t cs3 = vld1q_f32(combined_scale + n + 12);
        float32x4_t rs0 = vcvtq_f32_s32(vld1q_s32(row_sums + n +  0));
        float32x4_t rs1 = vcvtq_f32_s32(vld1q_s32(row_sums + n +  4));
        float32x4_t rs2 = vcvtq_f32_s32(vld1q_s32(row_sums + n +  8));
        float32x4_t rs3 = vcvtq_f32_s32(vld1q_s32(row_sums + n + 12));
        float32x4_t b0  = has_bias ? vld1q_f32(bias + n +  0) : vzero;
        float32x4_t b1  = has_bias ? vld1q_f32(bias + n +  4) : vzero;
        float32x4_t b2  = has_bias ? vld1q_f32(bias + n +  8) : vzero;
        float32x4_t b3  = has_bias ? vld1q_f32(bias + n + 12) : vzero;
        vst1q_f32(bias_eff_buf + n +  0, vfmaq_f32(b0, vmulq_f32(cs0, vk), rs0));
        vst1q_f32(bias_eff_buf + n +  4, vfmaq_f32(b1, vmulq_f32(cs1, vk), rs1));
        vst1q_f32(bias_eff_buf + n +  8, vfmaq_f32(b2, vmulq_f32(cs2, vk), rs2));
        vst1q_f32(bias_eff_buf + n + 12, vfmaq_f32(b3, vmulq_f32(cs3, vk), rs3));
    }
    for (; n + 3 < N; n += 4) {
        float32x4_t cs = vld1q_f32(combined_scale + n);
        float32x4_t rs = vcvtq_f32_s32(vld1q_s32(row_sums + n));
        float32x4_t b  = has_bias ? vld1q_f32(bias + n) : vzero;
        vst1q_f32(bias_eff_buf + n, vfmaq_f32(b, vmulq_f32(cs, vk), rs));
    }
#endif

    /* Scalar tail. */
    if (has_bias) {
        for (; n < N; ++n)
            bias_eff_buf[n] = bias[n] + combined_scale[n] * k * (float)row_sums[n];
    } else {
        for (; n < N; ++n)
            bias_eff_buf[n] = combined_scale[n] * k * (float)row_sums[n];
    }
}

/* Branch once on bias!=NULL into a constant-folded specialization. */
static inline const float *build_bias_eff(const float *bias,
                                          const float *combined_scale,
                                          const int32_t *row_sums,
                                          int32_t zp_x, int N,
                                          float *bias_eff_buf) {
    if (!row_sums || zp_x == 128) return bias;
    const float k = (float)(128 - zp_x);
    if (bias) build_bias_eff_impl(1, bias, combined_scale, row_sums, k, N, bias_eff_buf);
    else      build_bias_eff_impl(0, NULL, combined_scale, row_sums, k, N, bias_eff_buf);
    return bias_eff_buf;
}

void fe_qgemm_build_bias_eff_nobias(const float *combined_scale,
                                    const int32_t *row_sums,
                                    float k, int N,
                                    float *bias_eff_buf) {
    build_bias_eff_impl(0, NULL, combined_scale, row_sums, k, N, bias_eff_buf);
}

/* Track running min/max for diagnostics; scales never freeze. */
static inline void calib_update(FeActScale *act, FeActQuant q) {
    float observed_min = -(float)q.zp * q.scale;
    float observed_max = (255.0f - (float)q.zp) * q.scale;
    if (observed_max > act->max_running) act->max_running = observed_max;
    if (observed_min < act->min_running) act->min_running = observed_min;
    act->samples++;
}

void fe_qgemm_prequant(int M, int N, int K,
                       const int8_t *A_q, float scale_x, int32_t zp_x,
                       const int8_t *Bp,
                       const float  *scales_w,
                       const int32_t *row_sums,
                       const float  *bias,
                       float *C, int ldc,
                       int32_t *C32,
                       int act_silu) {
    float *combined_scale = g_combined_scale;
    fe_qg_scale_vec(scales_w, N, scale_x, combined_scale);
    const float *bias_eff = build_bias_eff(bias, combined_scale, row_sums,
                                           zp_x, N, g_bias_eff_buf);
    fe_qgemm_ops.gemm_fp32_fused(M, N, K, A_q, Bp,
                                 combined_scale, bias_eff, C, ldc,
                                 act_silu, C32);
#ifdef FE_DEBUG_NAN_TRACE
    {
        static int call_id = 0;
        int has_nan = 0, has_inf = 0;
        float maxv = -1e30f, minv = 1e30f;
        for (int r = 0; r < M; ++r) {
            for (int j = 0; j < N; ++j) {
                float v = C[(size_t)r*ldc + j];
                if (v != v) has_nan = 1;
                if (v == (1.0f/0.0f) || v == -(1.0f/0.0f)) has_inf = 1;
                if (v > maxv) maxv = v;
                if (v < minv) minv = v;
            }
        }
        if (has_nan || has_inf) {
            fprintf(stderr, "[NaN] prequant M=%d N=%d K=%d sx=%g zp=%d "
                    "rs=%p bias=%p silu=%d range=[%g,%g] nan=%d inf=%d (call %d)\n",
                    M, N, K, scale_x, zp_x, (void*)row_sums, (void*)bias,
                    act_silu, minv, maxv, has_nan, has_inf, call_id);
        }
        call_id++;
    }
#endif
}

void fe_qgemm_packed_calib_transposed_in(int M, int N, int K,
                                          const float *A_KM,
                                          const int8_t *Bp,
                                          const float  *scales_w,
                                          const int32_t *row_sums,
                                          const float  *bias,
                                          float *C, int ldc,
                                          int8_t  *aq_scratch,
                                          int32_t *c32_scratch,
                                          FeActScale *act) {
    /* Transpose-quantize into aq_scratch, then run prequant chain. */
    FeActQuant q = fe_quantize_activation_transposed(A_KM, M, K, aq_scratch);
    float *combined_scale = g_combined_scale;
    fe_qg_scale_vec(scales_w, N, q.scale, combined_scale);
    const float *bias_eff = build_bias_eff(bias, combined_scale, row_sums,
                                           q.zp, N, g_bias_eff_buf);
    fe_qgemm_ops.gemm_fp32_fused(M, N, K, aq_scratch, Bp,
                                 combined_scale, bias_eff, C, ldc,
                                 0, c32_scratch);
    calib_update(act, q);
}

/* fp16-input transposed_in. Eliminates the engine-side
 * unpack pass — fp16 enc_skip goes straight into transposed quantize. */
void fe_qgemm_packed_calib_transposed_in_fp16(int M, int N, int K,
                                               const uint16_t *A_KM_fp16,
                                               const int8_t *Bp,
                                               const float  *scales_w,
                                               const int32_t *row_sums,
                                               const float  *bias,
                                               float *C, int ldc,
                                               int8_t  *aq_scratch,
                                               int32_t *c32_scratch,
                                               FeActScale *act) {
    FeActQuant q = fe_quantize_activation_transposed_fp16(A_KM_fp16, M, K,
                                                           aq_scratch);
    float *combined_scale = g_combined_scale;
    fe_qg_scale_vec(scales_w, N, q.scale, combined_scale);
    const float *bias_eff = build_bias_eff(bias, combined_scale, row_sums,
                                           q.zp, N, g_bias_eff_buf);
    fe_qgemm_ops.gemm_fp32_fused(M, N, K, aq_scratch, Bp,
                                 combined_scale, bias_eff, C, ldc,
                                 0, c32_scratch);
    calib_update(act, q);
}

/* B operand is fp16. Joint min/max over fp32 A and fp16 B
 * (cvt on the fly for B), then quantize with the shared (scale, zp). The
 * B-row quantize reads fp16 directly and writes the second half of each
 * aq_scratch row. Eliminates the engine-side unpack pass.                */
void fe_qgemm_packed_silu_calib_concat2_fp16b(int M, int N, int K_half,
                                               const float *A,
                                               const uint16_t *B_fp16,
                                               const int8_t *Bp,
                                               const float  *scales_w,
                                               const int32_t *row_sums,
                                               const float  *bias,
                                               float *C, int ldc,
                                               int8_t  *aq_scratch,
                                               int32_t *c32_scratch,
                                               FeActScale *act) {
    const int K = 2 * K_half;

    float vmin_a = 0.0f, vmax_a = 0.0f, vmin_b = 0.0f, vmax_b = 0.0f;
    fe_qg_min_max  (A,      M * K_half, &vmin_a, &vmax_a);
    fe_fp16_min_max(B_fp16, M * K_half, &vmin_b, &vmax_b);
    float vmin = vmin_a < vmin_b ? vmin_a : vmin_b;
    float vmax = vmax_a > vmax_b ? vmax_a : vmax_b;

    if (vmin > 0.0f) vmin = 0.0f;
    if (vmax < 0.0f) vmax = 0.0f;
    float span = vmax - vmin;
    if (span < 1e-12f) span = 1e-12f;
    FeActQuant q;
    q.scale = span / 255.0f;
    {
        int zp = (int)lroundf(-vmin / q.scale);
        if (zp < 0)   zp = 0;
        if (zp > 255) zp = 255;
        q.zp = zp;
    }

    const float   inv = 1.0f / q.scale;
    const int32_t zof = q.zp - 128;

    /* A row: existing fp32 path. B row: fp16-direct quantize. */
    for (int m = 0; m < M; ++m) {
        fe_quantize_activation_with_scale(A + (size_t)m * K_half, K_half,
                                          aq_scratch + (size_t)m * K,
                                          q.scale, q.zp);
        fe_fp16_quantize_asym(B_fp16 + (size_t)m * K_half, K_half,
                              aq_scratch + (size_t)m * K + K_half,
                              inv, zof);
    }

    fe_qgemm_prequant(M, N, K, aq_scratch, q.scale, q.zp,
                      Bp, scales_w, row_sums, bias,
                      C, ldc, c32_scratch, 1);

    calib_update(act, q);
}

void fe_qgemm_packed_calib_acc(int M, int N, int K,
                                const float *A,
                                const int8_t *Bp,
                                const float  *scales_w,
                                const int32_t *row_sums,
                                const float  *bias,
                                float *C, int ldc,
                                int8_t  *aq_scratch,
                                int32_t *c32_scratch,
                                float   *fp32_scratch,
                                FeActScale *act) {
    /* Standard quantize + combined / bias_eff setup. */
    FeActQuant q = fe_quantize_activation(A, M * K, aq_scratch);
    float *combined_scale = g_combined_scale;
    fe_qg_scale_vec(scales_w, N, q.scale, combined_scale);
    const float *bias_eff = build_bias_eff(bias, combined_scale, row_sums,
                                           q.zp, N, g_bias_eff_buf);

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_I8MM) {
        qgemm_i8mm_fp32_fused_acc(M, N, K, aq_scratch, Bp,
                                    combined_scale, bias_eff,
                                    C, ldc, c32_scratch);
        calib_update(act, q);
        return;
    }
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_DOTPROD && (K & 3) == 0) {
        qgemm_dotprod_fp32_fused_acc(M, N, K, aq_scratch, Bp,
                                   combined_scale, bias_eff,
                                   C, ldc, c32_scratch);
        calib_update(act, q);
        return;
    }
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_NEON && (K & 3) == 0) {
        qgemm_neon_fp32_fused_acc(M, N, K, aq_scratch, Bp,
                                   combined_scale, bias_eff,
                                   C, ldc, c32_scratch);
        calib_update(act, q);
        return;
    }
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX512_VNNI) {
        qgemm_avx512vnni_fp32_fused_acc(M, N, K, aq_scratch, Bp,
                                         combined_scale, bias_eff,
                                         C, ldc, c32_scratch);
        calib_update(act, q);
        return;
    }
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX_VNNI) {
        qgemm_avxvnni_fp32_fused_acc(M, N, K, aq_scratch, Bp,
                                      combined_scale, bias_eff,
                                      C, ldc, c32_scratch);
        calib_update(act, q);
        return;
    }
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX2) {
        qgemm_avx2_fp32_fused_acc(M, N, K, aq_scratch, Bp,
                                   combined_scale, bias_eff,
                                   C, ldc, c32_scratch);
        calib_update(act, q);
        return;
    }
#endif
    /* Fallback: fp32 GEMM into fp32_scratch, then vec_add onto C. */
    (void)fp32_scratch;
    fe_qgemm_ops.gemm_fp32_fused(M, N, K, aq_scratch, Bp,
                                 combined_scale, bias_eff,
                                 fp32_scratch, ldc, 0, c32_scratch);
    /* C += fp32_scratch. */
    for (int m = 0; m < M; ++m)
        fe_vec_add(C + (size_t)m * ldc, fp32_scratch + (size_t)m * ldc, N);
    calib_update(act, q);
}

/*
 * fp32 -> int8 path used by the attention QKV projection.
 * Input is asymmetric uint8; output is symmetric int8 (zp=128) because
 * the downstream Q*K^T / S*V chain is activation*activation and double
 * asymmetric would need row-sums on both axes. Returns the output
 * (scale, zp=128).
 */
FeActQuant fe_qgemm_packed_calib_to_int8out(int M, int N, int K,
                                       const float *A,
                                       const int8_t *Bp,
                                       const float  *scales_w,
                                       const int32_t *row_sums,
                                       const float  *bias,
                                       int8_t *C_q, int ldc,
                                       float  *C_fp32_scratch, int ldc_fp32,
                                       int8_t  *aq_scratch,
                                       int32_t *c32_scratch,
                                       FeActScale *act_in,
                                       FeActScale *act_out) {
    FeActQuant qin = fe_quantize_activation(A, M * K, aq_scratch);
    calib_update(act_in, qin);

    float *combined = g_combined_scale;
    fe_qg_scale_vec(scales_w, N, qin.scale, combined);
    const float *bias_eff = build_bias_eff(bias, combined, row_sums,
                                           qin.zp, N, g_bias_eff_buf);

    /* Materialise fp32 output and capture max-abs. SIMD tiers use a
     * kernel variant that tracks max-abs in registers during the
     * dequant-store; others run a discrete max-abs scan afterwards. */
    float mx = 0.0f;
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_I8MM) {
        qgemm_i8mm_fp32_fused_track_maxabs(M, N, K, aq_scratch, Bp,
                                             combined, bias_eff,
                                             C_fp32_scratch, ldc_fp32,
                                             c32_scratch, &mx);
    } else if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_DOTPROD && (K & 3) == 0) {
        qgemm_dotprod_fp32_fused_track_maxabs(M, N, K, aq_scratch, Bp,
                                            combined, bias_eff,
                                            C_fp32_scratch, ldc_fp32,
                                            c32_scratch, &mx);
    } else if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_NEON && (K & 3) == 0) {
        qgemm_neon_fp32_fused_track_maxabs(M, N, K, aq_scratch, Bp,
                                            combined, bias_eff,
                                            C_fp32_scratch, ldc_fp32,
                                            c32_scratch, &mx);
    } else
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX512_VNNI) {
        qgemm_avx512vnni_fp32_fused_track_maxabs(M, N, K, aq_scratch, Bp,
                                                  combined, bias_eff,
                                                  C_fp32_scratch, ldc_fp32,
                                                  c32_scratch, &mx);
    } else if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX_VNNI) {
        qgemm_avxvnni_fp32_fused_track_maxabs(M, N, K, aq_scratch, Bp,
                                               combined, bias_eff,
                                               C_fp32_scratch, ldc_fp32,
                                               c32_scratch, &mx);
    } else if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX2) {
        qgemm_avx2_fp32_fused_track_maxabs(M, N, K, aq_scratch, Bp,
                                            combined, bias_eff,
                                            C_fp32_scratch, ldc_fp32,
                                            c32_scratch, &mx);
    } else
#endif
    {
        fe_qgemm_ops.gemm_fp32_fused(M, N, K, aq_scratch, Bp,
                                     combined, bias_eff,
                                     C_fp32_scratch, ldc_fp32,
                                     0, c32_scratch);
        /* Symmetric output: scale = max|y|/127, zp = 128. */
        if (ldc_fp32 == N) {
            mx = fe_qg_max_abs(C_fp32_scratch, M * N);
        } else {
            for (int r = 0; r < M; ++r) {
                float rm = fe_qg_max_abs(C_fp32_scratch + (size_t)r * ldc_fp32, N);
                if (rm > mx) mx = rm;
            }
        }
    }
    if (mx > act_out->max_running) act_out->max_running = mx;
    act_out->samples++;
    if (mx == 0.0f) mx = 1.0f;
    FeActQuant qout;
    qout.scale = mx / 127.0f;
    qout.zp    = 128;             /* symmetric int8 */

    /* zp=128 -> zp_off=0 -> pure symmetric quantize. */
    if (ldc == ldc_fp32) {
        fe_quantize_activation_with_scale(C_fp32_scratch, M * ldc,
                                          C_q, qout.scale, qout.zp);
    } else {
        for (int m = 0; m < M; ++m) {
            fe_quantize_activation_with_scale(
                C_fp32_scratch + (size_t)m * ldc_fp32, N,
                C_q + (size_t)m * ldc, qout.scale, qout.zp);
        }
    }
#ifdef FE_DEBUG_NAN_TRACE
    {
        static int call_id = 0;
        int has_nan = 0, has_inf = 0;
        for (int r = 0; r < M; ++r) for (int j = 0; j < N; ++j) {
            float v = C_fp32_scratch[(size_t)r*ldc_fp32 + j];
            if (v != v) has_nan = 1;
            if (v == (1.0f/0.0f) || v == -(1.0f/0.0f)) has_inf = 1;
        }
        if (has_nan || has_inf)
            fprintf(stderr, "[NaN] calib_to_int8out M=%d N=%d K=%d qin.scale=%g zp=%d qout.scale=%g mx=%g nan=%d inf=%d (call %d)\n",
                    M, N, K, qin.scale, qin.zp, qout.scale, mx, has_nan, has_inf, call_id);
        call_id++;
    }
#endif
    return qout;
}
